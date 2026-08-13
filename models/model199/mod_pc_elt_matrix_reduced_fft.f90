!> Element matrix of the REDUCED PDE operator P_full for the physics-based PC.
!!
!! This is the Milestone-1 artifact of the physics-based preconditioner
!! programme (docs/physics_pc/Development_Plan_Physics-Based.md).
!!
!! The true model-199 Jacobian is a mixed 6-variable system in
!!   (psi, u, j, w, rho, T),
!! where j and w are carried as explicit unknowns constrained by elliptic
!! equations (rows 3 and 4 of mod_elt_matrix_fft). This routine assembles a
!! DIFFERENT, 4-variable operator obtained by substituting the constraints at the
!! CONTINUOUS level and then discretizing:
!!
!!   J(dpsi) = dpsi_xx + dpsi_yy - dpsi_x/R      (Grad-Shafranov Delta*)
!!   W(du)   = du_xx   + du_yy   + du_x/R
!!
!! Both are read off the constraint rows themselves: mod_elt_matrix_fft.f90
!! amat_31/amat_33 give   int v*dj/R  = -int grad(v).grad(dpsi)/R,
!! amat_42/amat_44 give   int v*dw*R  = -int grad(v).grad(du)*R.
!! W is confirmed independently: the operator (v_xx + v_x/R + v_yy) already
!! appears verbatim in the existing visco_num term (mod_elt_matrix_fft.f90:516).
!!
!! Reduced variable ordering (n_var_red = 4):
!!   1 = psi   (<- full-system variable 1)
!!   2 = u     (<- full-system variable 2)
!!   3 = rho   (<- full-system variable 5)
!!   4 = T     (<- full-system variable 6)
!!
!! Block structure produced (row = equation, col = variable):
!!
!!        psi     u      rho     T
!!   psi [ D_psi  U_pu    0     U_pT ]
!!   u   [ L_up   D_u    L_ur   L_uT ]
!!   rho [ L_rp   U_ru   D_rho   0   ]
!!   T   [ L_Tp   U_Tu    0     D_T  ]
!!
!! The three psi-couplings inside the transport block -- U_pT, L_rp, L_Tp -- are
!! the "group-psi" terms. They are gated by physics_pc_drop_psi_coupling so that
!! Milestone 3 (arrow approximation) is a runtime switch, not a code change.
!!
!! IMPORTANT -- deliberate departures from the true Jacobian. These are
!! preconditioner-only approximations; the true residual and Jacobian are
!! untouched. Each is measured separately by verify_reduced_pde_operator.
!!
!!  (a) The visco_num hyperviscosity term of amat_24 is DROPPED. Under the
!!      substitution it becomes W(v)*W(W(du)), a SIXTH-order operator in u, which
!!      no integration by parts reduces to second derivatives on both test and
!!      trial. It is not representable in a C1 space.
!!
!!  (b) Three terms are integrated by parts once, to keep every substituted term
!!      at most second-order in both test and trial function (H2-conforming, which
!!      the C1 Bezier basis supplies), and to preserve the sign/symmetry of the
!!      stiff part:
!!        amat_13 hyper-resistivity:  -eta_num*grad(v).grad(dj)
!!                                 -> +eta_num*J(v)*J(dpsi)          (symmetric +)
!!        amat_23 Poisson bracket:    -v*[ps0, dj]
!!                                 -> -J(dpsi)*(v_s*ps0_t - v_t*ps0_s)
!!        amat_24 viscosity:          +visco_T*R*grad(v).grad(dw)
!!                                 -> -visco_T*R*W(v)*W(du)          (symmetric -)
!!      Each discards a boundary term. The amat_13 row additionally replaces the
!!      exact Laplacian of the test function by J(v), an O(1/R) difference, chosen
!!      so the contribution is symmetric positive-definite.
!!
!!  (c) Eliminating j and w also removes their wall constraints, so Delta*psi is
!!      no longer forced to vanish on the boundary.
!!
!! The LINEARIZATION STATE is not reduced: background fields (ps0, u0, zj0, w0,
!! r0, T0) are still read from all six components of the solution vector, exactly
!! as in mod_elt_matrix_fft. Only the trial/test space is reduced.
!!
!! No RHS is produced -- a preconditioner needs only the operator.
module mod_pc_elt_matrix_reduced_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm,   scatter_fft_to_elm_n, &
                                scatter_fft_to_elm_k, scatter_fft_to_elm_kn

  implicit none
  private

  public :: pc_elt_matrix_reduced_fft
  public :: n_var_red

  !> Number of variables in the reduced system: (psi, u, rho, T)
  integer, parameter :: n_var_red = 4

contains

subroutine pc_elt_matrix_reduced_fft(element, nodes, xpoint2, xcase2, &
                                     R_axis, Z_axis, psi_axis, psi_bnd, &
                                     R_xpoint, Z_xpoint, ELM)

  use constants
  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module
  use diffusivities, only: get_dperp, get_zkperp
  use corr_neg
  use equil_info,  only: get_psi_n

  implicit none

  type(type_element), intent(in)    :: element
  type(type_node),    intent(in)    :: nodes(n_vertex_max)
  logical,            intent(in)    :: xpoint2
  integer,            intent(in)    :: xcase2
  real*8,             intent(in)    :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,             intent(in)    :: R_xpoint(2), Z_xpoint(2)

  ! Poloidal-block and full (toroidally expanded) dimensions of the reduced system
#define NRV (n_var_red*n_vertex_max*n_degrees)
#define DRV (n_tor*n_var_red*n_vertex_max*n_degrees)

  real*8, dimension(DRV, DRV), intent(out) :: ELM

  ! Toroidal channels: p = no d/dphi, n = d/dphi on trial, k = d/dphi on test,
  ! kn = d/dphi on both.
  real*8, dimension(n_plane, NRV, NRV) :: ELM_p, ELM_n, ELM_k, ELM_kn

  ! Geometry at Gauss points
  real*8, dimension(n_gauss,n_gauss) :: x_g, x_s, x_t, x_ss, x_st, x_tt
  real*8, dimension(n_gauss,n_gauss) :: y_g, y_s, y_t, y_ss, y_st, y_tt

  ! Background fields at Gauss points (all six components -- the linearization
  ! state is NOT reduced). Only first derivatives are needed by the retained terms.
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_g, eq_s, eq_t, eq_p

  !> Only current_source is needed: it enters the U_psi_T block (amat_16). The
  !! particle/heat sources of the true model appear only in the RHS, which a
  !! preconditioner does not need.
  real*8 :: current_source(n_gauss,n_gauss)

  integer :: i, j, k, l, ms, mt, mp, in
  integer :: index_ij, index_kl
  integer :: ij1, ij2, ij3, ij4, kl1, kl2, kl3, kl4

  real*8 :: wst, xjac, xjac_x, xjac_y, BigR, BigR_x, eps_cyl
  real*8 :: theta, zeta

  ! Test function and its derivatives
  real*8 :: v, v_x, v_y, v_s, v_t, v_p, v_ss, v_st, v_tt, v_xx, v_yy
  ! Trial function and its derivatives (shared by all four reduced variables)
  real*8 :: psi, psi_x, psi_y, psi_p, psi_s, psi_t, psi_ss, psi_st, psi_tt
  real*8 :: psi_xx, psi_yy
  real*8 :: u, u_x, u_y, u_s, u_t, u_p
  real*8 :: rho, rho_x, rho_y, rho_s, rho_t, rho_p, rho_hat, rho_x_hat, rho_y_hat
  real*8 :: T, T_x, T_y, T_s, T_t, T_p

  !> Substituted constraint operators applied to the TRIAL function, and the
  !! test-function counterparts used by the symmetric weak forms.
  real*8 :: J_tri, W_tri, J_tst, W_tst

  ! Background values at the current Gauss point
  real*8 :: ps0, ps0_x, ps0_y, ps0_p, ps0_s, ps0_t
  real*8 :: u0, u0_x, u0_y, u0_p, u0_s, u0_t
  real*8 :: zj0, zj0_x, zj0_y, zj0_p, zj0_s, zj0_t
  real*8 :: w0, w0_x, w0_y, w0_p, w0_s, w0_t
  real*8 :: r0, r0_x, r0_y, r0_p, r0_s, r0_t, r0_hat, r0_x_hat, r0_y_hat
  real*8 :: T0, T0_x, T0_y, T0_p, T0_s, T0_t
  real*8 :: vv2

  ! Coefficients
  real*8 :: eta_T, deta_dT, d2eta_d2T, eta_T_ohm, deta_dT_ohm
  real*8 :: visco_T, dvisco_dT
  real*8 :: D_prof, D_par_local, ZK_prof, psi_norm, BB2

  ! Parallel-transport linearization helpers (identical to mod_elt_matrix_fft)
  real*8 :: Bgrad_rho_star, Bgrad_rho_k_star, Bgrad_rho
  real*8 :: Bgrad_rho_star_psi, Bgrad_rho_psi, Bgrad_rho_rho, Bgrad_rho_rho_n
  real*8 :: Bgrad_T_star, Bgrad_T_k_star, Bgrad_T
  real*8 :: Bgrad_T_star_psi, Bgrad_T_psi, Bgrad_T_T, Bgrad_T_T_n
  real*8 :: BB2_psi

  ! Reduced-system matrix entries. Naming: a_<row><col> with rows/cols in
  ! (psi, u, rho, T) = (p, u, r, T).
  real*8 :: a_pp, a_pu, a_pu_n, a_pT
  real*8 :: a_up, a_up_n, a_uu, a_uu_kn, a_ur, a_uT
  real*8 :: a_rp, a_rp_k, a_ru, a_rr, a_rr_k, a_rr_n, a_rr_kn
  real*8 :: a_Tp, a_Tp_k, a_Tu, a_TT, a_TT_k, a_TT_n, a_TT_kn

  logical :: drop_psi_coupling

  ! FFT workspace
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  drop_psi_coupling = physics_pc_drop_psi_coupling

  ELM    = 0.d0
  ELM_p  = 0.d0
  ELM_n  = 0.d0
  ELM_k  = 0.d0
  ELM_kn = 0.d0

  ! mod_elt_matrix_fft sets GAMMA locally; mirror it so the reduced operator uses
  ! exactly the same adiabatic index as the true Jacobian.
  GAMMA = 5.d0 / 3.d0

  theta = time_evol_theta
  zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)

  current_source  = 0.d0

  !--------------------------------------------------------------------------
  ! Geometry and background fields at the Gauss points
  !--------------------------------------------------------------------------
  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0
  y_g = 0.d0; y_s = 0.d0; y_t = 0.d0; y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0; eq_p = 0.d0

  do i = 1, n_vertex_max
    do j = 1, n_degrees
      do ms = 1, n_gauss
        do mt = 1, n_gauss

          x_g(ms,mt)  = x_g(ms,mt)  + nodes(i)%x(1,j,1) * element%size(i,j) * H(i,j,ms,mt)
          x_s(ms,mt)  = x_s(ms,mt)  + nodes(i)%x(1,j,1) * element%size(i,j) * H_s(i,j,ms,mt)
          x_t(ms,mt)  = x_t(ms,mt)  + nodes(i)%x(1,j,1) * element%size(i,j) * H_t(i,j,ms,mt)
          x_ss(ms,mt) = x_ss(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_ss(i,j,ms,mt)
          x_st(ms,mt) = x_st(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_st(i,j,ms,mt)
          x_tt(ms,mt) = x_tt(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_tt(i,j,ms,mt)

          y_g(ms,mt)  = y_g(ms,mt)  + nodes(i)%x(1,j,2) * element%size(i,j) * H(i,j,ms,mt)
          y_s(ms,mt)  = y_s(ms,mt)  + nodes(i)%x(1,j,2) * element%size(i,j) * H_s(i,j,ms,mt)
          y_t(ms,mt)  = y_t(ms,mt)  + nodes(i)%x(1,j,2) * element%size(i,j) * H_t(i,j,ms,mt)
          y_ss(ms,mt) = y_ss(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_ss(i,j,ms,mt)
          y_st(ms,mt) = y_st(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_st(i,j,ms,mt)
          y_tt(ms,mt) = y_tt(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_tt(i,j,ms,mt)

          do mp = 1, n_plane
            do k = 1, n_var
              do in = 1, n_tor
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)   * HZ(in,mp)
                eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt) * HZ(in,mp)
                eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt) * HZ(in,mp)
                eq_p(mp,k,ms,mt) = eq_p(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)   * HZ_p(in,mp)
              enddo
            enddo
          enddo

          if (keep_current_prof) &
            call current(xpoint2, xcase2, x_g(ms,mt), y_g(ms,mt), Z_xpoint, &
                         eq_g(1,1,ms,mt), psi_axis, psi_bnd, current_source(ms,mt))

        enddo
      enddo
    enddo
  enddo

  !--------------------------------------------------------------------------
  ! Gauss integration
  !--------------------------------------------------------------------------
  do ms = 1, n_gauss
   do mt = 1, n_gauss

    wst  = wgauss(ms)*wgauss(mt)
    xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)

    xjac_x = (x_ss(ms,mt)*y_t(ms,mt)**2 - y_ss(ms,mt)*x_t(ms,mt)*y_t(ms,mt)             &
            - 2.d0*x_st(ms,mt)*y_s(ms,mt)*y_t(ms,mt)                                    &
            + y_st(ms,mt)*(x_s(ms,mt)*y_t(ms,mt) + x_t(ms,mt)*y_s(ms,mt))               &
            + x_tt(ms,mt)*y_s(ms,mt)**2 - y_tt(ms,mt)*x_s(ms,mt)*y_s(ms,mt)) / xjac

    xjac_y = (y_tt(ms,mt)*x_s(ms,mt)**2 - x_tt(ms,mt)*y_s(ms,mt)*x_s(ms,mt)             &
            - 2.d0*y_st(ms,mt)*x_t(ms,mt)*x_s(ms,mt)                                    &
            + x_st(ms,mt)*(y_t(ms,mt)*x_s(ms,mt) + y_s(ms,mt)*x_t(ms,mt))               &
            + y_ss(ms,mt)*x_t(ms,mt)**2 - x_ss(ms,mt)*y_t(ms,mt)*x_t(ms,mt)) / xjac

    BigR    = x_g(ms,mt)
    BigR_x  = 1.d0
    eps_cyl = 1.d0

    do mp = 1, n_plane

      ps0   = eq_g(mp,1,ms,mt)
      ps0_x = (  y_t(ms,mt)*eq_s(mp,1,ms,mt) - y_s(ms,mt)*eq_t(mp,1,ms,mt) ) / xjac
      ps0_y = ( -x_t(ms,mt)*eq_s(mp,1,ms,mt) + x_s(ms,mt)*eq_t(mp,1,ms,mt) ) / xjac
      ps0_p = eq_p(mp,1,ms,mt)
      ps0_s = eq_s(mp,1,ms,mt)
      ps0_t = eq_t(mp,1,ms,mt)

      u0    = eq_g(mp,2,ms,mt)
      u0_x  = (  y_t(ms,mt)*eq_s(mp,2,ms,mt) - y_s(ms,mt)*eq_t(mp,2,ms,mt) ) / xjac
      u0_y  = ( -x_t(ms,mt)*eq_s(mp,2,ms,mt) + x_s(ms,mt)*eq_t(mp,2,ms,mt) ) / xjac
      u0_p  = eq_p(mp,2,ms,mt)
      u0_s  = eq_s(mp,2,ms,mt)
      u0_t  = eq_t(mp,2,ms,mt)

      vv2   = BigR**2 * ( u0_x*u0_x + u0_y*u0_y )

      zj0   = eq_g(mp,3,ms,mt)
      zj0_x = (  y_t(ms,mt)*eq_s(mp,3,ms,mt) - y_s(ms,mt)*eq_t(mp,3,ms,mt) ) / xjac
      zj0_y = ( -x_t(ms,mt)*eq_s(mp,3,ms,mt) + x_s(ms,mt)*eq_t(mp,3,ms,mt) ) / xjac
      zj0_p = eq_p(mp,3,ms,mt)
      zj0_s = eq_s(mp,3,ms,mt)
      zj0_t = eq_t(mp,3,ms,mt)

      w0    = eq_g(mp,4,ms,mt)
      w0_x  = (  y_t(ms,mt)*eq_s(mp,4,ms,mt) - y_s(ms,mt)*eq_t(mp,4,ms,mt) ) / xjac
      w0_y  = ( -x_t(ms,mt)*eq_s(mp,4,ms,mt) + x_s(ms,mt)*eq_t(mp,4,ms,mt) ) / xjac
      w0_p  = eq_p(mp,4,ms,mt)
      w0_s  = eq_s(mp,4,ms,mt)
      w0_t  = eq_t(mp,4,ms,mt)

      r0    = abs(eq_g(mp,5,ms,mt))
      r0_x  = (  y_t(ms,mt)*eq_s(mp,5,ms,mt) - y_s(ms,mt)*eq_t(mp,5,ms,mt) ) / xjac
      r0_y  = ( -x_t(ms,mt)*eq_s(mp,5,ms,mt) + x_s(ms,mt)*eq_t(mp,5,ms,mt) ) / xjac
      r0_p  = eq_p(mp,5,ms,mt)
      r0_s  = eq_s(mp,5,ms,mt)
      r0_t  = eq_t(mp,5,ms,mt)

      r0_hat   = BigR**2 * r0
      r0_x_hat = 2.d0 * BigR * BigR_x * r0 + BigR**2 * r0_x
      r0_y_hat = BigR**2 * r0_y

      T0    = abs(eq_g(mp,6,ms,mt))
      T0_x  = (  y_t(ms,mt)*eq_s(mp,6,ms,mt) - y_s(ms,mt)*eq_t(mp,6,ms,mt) ) / xjac
      T0_y  = ( -x_t(ms,mt)*eq_s(mp,6,ms,mt) + x_s(ms,mt)*eq_t(mp,6,ms,mt) ) / xjac
      T0_p  = eq_p(mp,6,ms,mt)
      T0_s  = eq_s(mp,6,ms,mt)
      T0_t  = eq_t(mp,6,ms,mt)

      ! --- Temperature dependent resistivity
      if ( eta_T_dependent .and. corr_neg_temp(T0) <= T_max_eta) then
        eta_T     = eta * (corr_neg_temp(T0)/T_0)**(-1.5d0)
        deta_dT   = - eta * (1.5d0)  * corr_neg_temp(T0)**(-2.5d0) * T_0**(1.5d0)
        d2eta_d2T =   eta * (3.75d0) * corr_neg_temp(T0)**(-3.5d0) * T_0**(1.5d0)
      else if ( eta_T_dependent .and. corr_neg_temp(T0) > T_max_eta) then
        eta_T     = eta * (T_max_eta/T_0)**(-1.5d0)
        deta_dT   = 0.d0
        d2eta_d2T = 0.d0
      else
        eta_T     = eta
        deta_dT   = 0.d0
        d2eta_d2T = 0.d0
      end if

      ! --- Eta for ohmic heating
      if ( eta_T_dependent .and. corr_neg_temp(T0) <= T_max_eta_ohm) then
        eta_T_ohm   = eta_ohmic * (corr_neg_temp(T0)/T_0)**(-1.5d0)
        deta_dT_ohm = - eta_ohmic * (1.5d0) * corr_neg_temp(T0)**(-2.5d0) * T_0**(1.5d0)
      else if ( eta_T_dependent .and. corr_neg_temp(T0) > T_max_eta_ohm) then
        eta_T_ohm   = eta_ohmic * (T_max_eta_ohm/T_0)**(-1.5d0)
        deta_dT_ohm = 0.d0
      else
        eta_T_ohm   = eta_ohmic
        deta_dT_ohm = 0.d0
      end if

      ! --- Temperature dependent viscosity
      if ( visco_T_dependent ) then
        visco_T   = visco * (corr_neg_temp(T0)/T_0)**(-1.5d0)
        dvisco_dT = - visco * (1.5d0) * corr_neg_temp(T0)**(-2.5d0) * T_0**(1.5d0)
      else
        visco_T   = visco
        dvisco_dT = 0.d0
      end if

      psi_norm    = get_psi_n(ps0, y_g(ms,mt))
      D_prof      = get_dperp (psi_norm)
      D_par_local = D_par
      ZK_prof     = get_zkperp(psi_norm)

      if (xpoint2) then
        if (r0 .lt. D_prof_neg_thresh)  D_prof  = D_prof_neg
        if (T0 .lt. ZK_prof_neg_thresh) ZK_prof = ZK_prof_neg
      endif

      BB2 = (F0*F0 + ps0_x*ps0_x + ps0_y*ps0_y) / BigR**2

      !----------------------------------------------------------------------
      ! Test function loop
      !----------------------------------------------------------------------
      do i = 1, n_vertex_max
       do j = 1, n_degrees

        index_ij = n_var_red*n_degrees*(i-1) + n_var_red*(j-1) + 1

        v   =  H(i,j,ms,mt) * element%size(i,j)
        v_x = (  y_t(ms,mt)*h_s(i,j,ms,mt) - y_s(ms,mt)*h_t(i,j,ms,mt) ) * element%size(i,j) / xjac
        v_y = ( -x_t(ms,mt)*h_s(i,j,ms,mt) + x_s(ms,mt)*h_t(i,j,ms,mt) ) * element%size(i,j) / xjac
        v_s = h_s(i,j,ms,mt)  * element%size(i,j)
        v_t = h_t(i,j,ms,mt)  * element%size(i,j)
        v_p = H(i,j,ms,mt)    * element%size(i,j)
        v_ss = h_ss(i,j,ms,mt) * element%size(i,j)
        v_tt = h_tt(i,j,ms,mt) * element%size(i,j)
        v_st = h_st(i,j,ms,mt) * element%size(i,j)

        v_xx = (v_ss*y_t(ms,mt)**2 - 2.d0*v_st*y_s(ms,mt)*y_t(ms,mt) + v_tt*y_s(ms,mt)**2  &
              + v_s*(y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt))                      &
              + v_t*(y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt)) ) / xjac**2          &
              - xjac_x*(v_s*y_t(ms,mt) - v_t*y_s(ms,mt)) / xjac**2

        v_yy = (v_ss*x_t(ms,mt)**2 - 2.d0*v_st*x_s(ms,mt)*x_t(ms,mt) + v_tt*x_s(ms,mt)**2  &
              + v_s*(x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt))                      &
              + v_t*(x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt)) ) / xjac**2          &
              - xjac_y*(-v_s*x_t(ms,mt) + v_t*x_s(ms,mt)) / xjac**2

        ! Constraint operators applied to the TEST function (symmetric weak forms)
        J_tst = v_xx + v_yy - v_x/BigR
        W_tst = v_xx + v_yy + v_x/BigR

        Bgrad_rho_star   = ( v_x*ps0_y - v_y*ps0_x ) / BigR
        Bgrad_rho_k_star = ( F0/BigR * v_p ) / BigR
        Bgrad_rho        = ( F0/BigR * r0_p + r0_x*ps0_y - r0_y*ps0_x ) / BigR
        Bgrad_T_star     = ( v_x*ps0_y - v_y*ps0_x ) / BigR
        Bgrad_T_k_star   = ( F0/BigR * v_p ) / BigR
        Bgrad_T          = ( F0/BigR * T0_p + T0_x*ps0_y - T0_y*ps0_x ) / BigR

        ij1 = index_ij       ! psi equation
        ij2 = index_ij + 1   ! u   equation
        ij3 = index_ij + 2   ! rho equation
        ij4 = index_ij + 3   ! T   equation

        !--------------------------------------------------------------------
        ! Trial function loop
        !--------------------------------------------------------------------
        do k = 1, n_vertex_max
         do l = 1, n_degrees

          psi   = H(k,l,ms,mt) * element%size(k,l)
          psi_x = (  y_t(ms,mt)*h_s(k,l,ms,mt) - y_s(ms,mt)*h_t(k,l,ms,mt) ) / xjac * element%size(k,l)
          psi_y = ( -x_t(ms,mt)*h_s(k,l,ms,mt) + x_s(ms,mt)*h_t(k,l,ms,mt) ) / xjac * element%size(k,l)
          psi_p = H(k,l,ms,mt)    * element%size(k,l)
          psi_s = h_s(k,l,ms,mt)  * element%size(k,l)
          psi_t = h_t(k,l,ms,mt)  * element%size(k,l)
          psi_ss = h_ss(k,l,ms,mt) * element%size(k,l)
          psi_tt = h_tt(k,l,ms,mt) * element%size(k,l)
          psi_st = h_st(k,l,ms,mt) * element%size(k,l)

          psi_xx = (psi_ss*y_t(ms,mt)**2 - 2.d0*psi_st*y_s(ms,mt)*y_t(ms,mt) + psi_tt*y_s(ms,mt)**2 &
                  + psi_s*(y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt))                         &
                  + psi_t*(y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt)) ) / xjac**2             &
                  - xjac_x*(psi_s*y_t(ms,mt) - psi_t*y_s(ms,mt)) / xjac**2

          psi_yy = (psi_ss*x_t(ms,mt)**2 - 2.d0*psi_st*x_s(ms,mt)*x_t(ms,mt) + psi_tt*x_s(ms,mt)**2 &
                  + psi_s*(x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt))                         &
                  + psi_t*(x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt)) ) / xjac**2             &
                  - xjac_y*(-psi_s*x_t(ms,mt) + psi_t*x_s(ms,mt)) / xjac**2

          ! All reduced trial variables share the same scalar basis function
          u   = psi   ;  rho   = psi   ;  T   = psi
          u_x = psi_x ;  rho_x = psi_x ;  T_x = psi_x
          u_y = psi_y ;  rho_y = psi_y ;  T_y = psi_y
          u_p = psi_p ;  rho_p = psi_p ;  T_p = psi_p
          u_s = psi_s ;  rho_s = psi_s ;  T_s = psi_s
          u_t = psi_t ;  rho_t = psi_t ;  T_t = psi_t

          rho_hat   = BigR**2 * rho
          rho_x_hat = 2.d0 * BigR * BigR_x * rho + BigR**2 * rho_x
          rho_y_hat = BigR**2 * rho_y

          !--- THE SUBSTITUTION: dj -> J(dpsi),  dw -> W(du) ------------------
          J_tri = psi_xx + psi_yy - psi_x/BigR    ! Grad-Shafranov Delta*
          W_tri = psi_xx + psi_yy + psi_x/BigR

          index_kl = n_var_red*n_degrees*(k-1) + n_var_red*(l-1) + 1
          kl1 = index_kl       ! psi
          kl2 = index_kl + 1   ! u
          kl3 = index_kl + 2   ! rho
          kl4 = index_kl + 3   ! T

          !====================================================================
          ! Row psi  (full-system equation 1)
          !====================================================================
          ! D_psi = amat_11 + amat_13[dj -> J(dpsi)]
          a_pp = v * psi / BigR * xjac * (1.d0+zeta)                                     &
               - v * (psi_s * u0_t - psi_t * u0_s)                       * theta * tstep &
               !--- amat_13, resistive part: dj -> J(dpsi), substituted directly
               - eta_T * v * J_tri / BigR                        * xjac * theta * tstep  &
               !--- amat_13, hyper-resistive part: -eta_num*grad(v).grad(dj)
               !    integrated by parts once -> +eta_num*J(v)*J(dpsi)  (symmetric, +)
               + eta_num * J_tst * J_tri                         * xjac * theta * tstep

          ! U_psi_u = amat_12 (+ toroidal channel)
          a_pu   = - v * (ps0_s * u_t - ps0_t * u_s)                     * theta * tstep
          a_pu_n = + eps_cyl * F0 / BigR * v * u_p              * xjac * theta * tstep

          ! U_psi_T = amat_16  -- group-psi term
          if (drop_psi_coupling) then
            a_pT = 0.d0
          else
            a_pT = - deta_dT * v * T * (zj0 - current_source(ms,mt)) / BigR &
                                                                 * xjac * theta * tstep
          endif

          !====================================================================
          ! Row u  (full-system equation 2)
          !====================================================================
          ! L_u_psi = amat_21 + amat_23[dj -> J(dpsi)]
          a_up = - v * (psi_s * zj0_t - psi_t * zj0_s)                   * theta * tstep &
               !--- amat_23 = -v*[ps0,dj]; bracket integrated by parts once so that
               !    J(dpsi) appears undifferentiated:
               !      int v*(ps0_s*f_t - ps0_t*f_s) = int f*(v_s*ps0_t - v_t*ps0_s)
               - J_tri * (v_s * ps0_t - v_t * ps0_s)                     * theta * tstep

          ! amat_23_n: toroidal channel of the Lorentz-force term
          a_up_n = + eps_cyl * F0 / BigR * v * J_tri            * xjac * theta * tstep

          ! D_u = amat_22 + amat_24[dw -> W(du)]
          a_uu = - BigR * r0_hat * (v_x * u_x + v_y * u_y)      * xjac * (1.d0+zeta)     &
               + r0_hat * BigR**2 * w0 * (v_s * u_t - v_t * u_s)         * theta * tstep &
               + 0.5d0 * BigR**2 * (u_x * u0_x + u_y * u0_y)                             &
                       * (v_x * r0_y_hat - v_y * r0_x_hat)      * xjac * theta * tstep   &
               !--- amat_24 inertia: dw -> W(du), substituted directly
               + r0_hat * BigR**2 * W_tri * (v_s * u0_t - v_t * u0_s)    * theta * tstep &
               !--- amat_24 viscosity: +visco_T*R*grad(v).grad(dw) integrated by parts
               !    once. div(R*grad(v)) = R*W(v) exactly, so this is exact IBP:
               !      int visco_T*R*grad(v).grad(W(du)) = -int visco_T*R*W(v)*W(du)
               - visco_T * BigR * W_tst * W_tri                 * xjac * theta * tstep
               !--- amat_24 hyperviscosity visco_num*W(v)*W(dw) is DROPPED: under the
               !    substitution it is W(v)*W(W(du)), a 6th-order operator that no
               !    integration by parts brings into the C1 space. See header (a).

          ! amat_24_kn: toroidal viscous channel, dw -> W(du)
          a_uu_kn = + visco_T / BigR * v_p * W_tri              * xjac * theta * tstep

          ! L_u_rho = amat_25
          a_ur = + 0.5d0 * vv2 * (v_x * rho_y_hat - v_y * rho_x_hat) * xjac * theta * tstep &
               + rho_hat * BigR**2 * w0 * (v_s * u0_t - v_t * u0_s)         * theta * tstep &
               + 2.d0 * BigR * v * (rho_y * T0 + T0_y * rho)         * xjac * theta * tstep

          ! L_u_T = amat_26
          a_uT = + 2.d0 * BigR * v * (r0_y * T + T_y * r0)           * xjac * theta * tstep &
               + dvisco_dT * T * (v_x * w0_x + v_y * w0_y) * BigR    * xjac * theta * tstep

          !====================================================================
          ! Row rho  (full-system equation 5)
          !====================================================================
          Bgrad_rho_star_psi = ( v_x  * psi_y - v_y  * psi_x ) / BigR
          Bgrad_rho_psi      = ( r0_x * psi_y - r0_y * psi_x ) / BigR
          Bgrad_rho_rho      = ( rho_x * ps0_y - rho_y * ps0_x ) / BigR
          Bgrad_rho_rho_n    = ( F0 / BigR * rho_p ) / BigR
          BB2_psi            = 2.d0 * (psi_x * ps0_x + psi_y * ps0_y) / BigR**2

          ! L_rho_psi = amat_51 -- group-psi term
          if (drop_psi_coupling) then
            a_rp   = 0.d0
            a_rp_k = 0.d0
          else
            a_rp = - (D_par-D_prof) * BigR * BB2_psi/BB2**2 * Bgrad_rho_star     * Bgrad_rho     * xjac * theta * tstep &
                 + (D_par-D_prof) * BigR / BB2             * Bgrad_rho_star_psi * Bgrad_rho     * xjac * theta * tstep &
                 + (D_par-D_prof) * BigR / BB2             * Bgrad_rho_star     * Bgrad_rho_psi * xjac * theta * tstep

            a_rp_k = - (D_par-D_prof) * BigR * BB2_psi/BB2**2 * Bgrad_rho_k_star * Bgrad_rho     * xjac * theta * tstep &
                   + (D_par-D_prof) * BigR / BB2             * Bgrad_rho_k_star * Bgrad_rho_psi * xjac * theta * tstep
          endif

          ! U_rho_u = amat_52
          a_ru = - v * BigR**2 * ( r0_s * u_t - r0_t * u_s)                    * theta * tstep &
               - v * 2.d0 * BigR * r0 * u_y                             * xjac * theta * tstep

          ! D_rho = amat_55
          a_rr = v * rho * BigR * (1.d0 + zeta)                          * xjac                &
               - v * BigR**2 * ( rho_s * u0_t - rho_t * u0_s)                   * theta * tstep &
               - v * 2.d0 * BigR * rho * u0_y                           * xjac * theta * tstep &
               + (D_par-D_prof) * BigR / BB2 * Bgrad_rho_star * Bgrad_rho_rho * xjac * theta * tstep &
               + D_prof * BigR * (v_x*rho_x + v_y*rho_y)                * xjac * theta * tstep

          a_rr_k  = + (D_par-D_prof) * BigR / BB2 * Bgrad_rho_k_star * Bgrad_rho_rho   * xjac * theta * tstep
          a_rr_n  = + (D_par-D_prof) * BigR / BB2 * Bgrad_rho_star   * Bgrad_rho_rho_n * xjac * theta * tstep
          a_rr_kn = + (D_par-D_prof) * BigR / BB2 * Bgrad_rho_k_star * Bgrad_rho_rho_n * xjac * theta * tstep &
                  + D_prof * BigR * ( v_p*rho_p * eps_cyl**2 / BigR**2 )               * xjac * theta * tstep

          !====================================================================
          ! Row T  (full-system equation 6)
          !====================================================================
          Bgrad_T_star_psi = ( v_x  * psi_y - v_y  * psi_x ) / BigR
          Bgrad_T_psi      = ( T0_x * psi_y - T0_y * psi_x ) / BigR
          Bgrad_T_T        = ( T_x * ps0_y - T_y * ps0_x ) / BigR
          Bgrad_T_T_n      = ( F0 / BigR * T_p ) / BigR

          ! L_T_psi = amat_61 + amat_63[dj -> J(dpsi)] -- group-psi term
          if (drop_psi_coupling) then
            a_Tp   = 0.d0
            a_Tp_k = 0.d0
          else
            a_Tp = - (ZK_par-ZK_prof) * BigR * BB2_psi/BB2**2 * Bgrad_T_star     * Bgrad_T     * xjac * theta * tstep &
                 + (ZK_par-ZK_prof) * BigR / BB2              * Bgrad_T_star_psi * Bgrad_T     * xjac * theta * tstep &
                 + (ZK_par-ZK_prof) * BigR / BB2              * Bgrad_T_star     * Bgrad_T_psi * xjac * theta * tstep &
                 !--- amat_63 ohmic heating: dj -> J(dpsi), substituted directly
                 - v * (gamma-1.d0) * eta_T_ohm * 2.d0 * J_tri * zj0/(BigR**2.d0) * BigR * xjac * theta * tstep

            a_Tp_k = - (ZK_par-ZK_prof) * BigR * BB2_psi/BB2**2 * Bgrad_T_k_star * Bgrad_T     * xjac * theta * tstep &
                   + (ZK_par-ZK_prof) * BigR / BB2              * Bgrad_T_k_star * Bgrad_T_psi * xjac * theta * tstep
          endif

          ! U_T_u = amat_62
          a_Tu = - v * BigR**2 * ( T0_s * u_t - T0_t * u_s)                        * theta * tstep &
               - v * 2.d0 * (GAMMA-1.d0) * BigR * T0 * u_y                  * xjac * theta * tstep

          ! D_T = amat_66
          a_TT = v * T * BigR * xjac * (1.d0 + zeta)                                               &
               - v * BigR**2 * ( T_s * u0_t - T_t * u0_s)                        * theta * tstep &
               - v * 2.d0 * (GAMMA-1.d0) * BigR * T * u0_y                  * xjac * theta * tstep &
               + (ZK_par-ZK_prof) * BigR / BB2 * Bgrad_T_star * Bgrad_T_T   * xjac * theta * tstep &
               + ZK_prof * BigR * ( v_x*T_x + v_y*T_y )                     * xjac * theta * tstep &
               - v * T * (gamma-1.d0) * deta_dT_ohm * (zj0/BigR)**2.d0 * BigR * xjac * theta * tstep

          a_TT_k  = + (ZK_par-ZK_prof) * BigR / BB2 * Bgrad_T_k_star * Bgrad_T_T   * xjac * theta * tstep
          a_TT_n  = + (ZK_par-ZK_prof) * BigR / BB2 * Bgrad_T_star   * Bgrad_T_T_n * xjac * theta * tstep
          a_TT_kn = + (ZK_par-ZK_prof) * BigR / BB2 * Bgrad_T_k_star * Bgrad_T_T_n * xjac * theta * tstep &
                  + ZK_prof * BigR * ( v_p*T_p / BigR**2 )                         * xjac * theta * tstep

          !====================================================================
          ! Accumulate into the toroidal channels
          !====================================================================
          ! Row psi
          ELM_p (mp,ij1,kl1) = ELM_p (mp,ij1,kl1) + wst * a_pp
          ELM_p (mp,ij1,kl2) = ELM_p (mp,ij1,kl2) + wst * a_pu
          ELM_n (mp,ij1,kl2) = ELM_n (mp,ij1,kl2) + wst * a_pu_n
          ELM_p (mp,ij1,kl4) = ELM_p (mp,ij1,kl4) + wst * a_pT

          ! Row u
          ELM_p (mp,ij2,kl1) = ELM_p (mp,ij2,kl1) + wst * a_up
          ELM_n (mp,ij2,kl1) = ELM_n (mp,ij2,kl1) + wst * a_up_n
          ELM_p (mp,ij2,kl2) = ELM_p (mp,ij2,kl2) + wst * a_uu
          ELM_kn(mp,ij2,kl2) = ELM_kn(mp,ij2,kl2) + wst * a_uu_kn
          ELM_p (mp,ij2,kl3) = ELM_p (mp,ij2,kl3) + wst * a_ur
          ELM_p (mp,ij2,kl4) = ELM_p (mp,ij2,kl4) + wst * a_uT

          ! Row rho
          ELM_p (mp,ij3,kl1) = ELM_p (mp,ij3,kl1) + wst * a_rp
          ELM_k (mp,ij3,kl1) = ELM_k (mp,ij3,kl1) + wst * a_rp_k
          ELM_p (mp,ij3,kl2) = ELM_p (mp,ij3,kl2) + wst * a_ru
          ELM_p (mp,ij3,kl3) = ELM_p (mp,ij3,kl3) + wst * a_rr
          ELM_k (mp,ij3,kl3) = ELM_k (mp,ij3,kl3) + wst * a_rr_k
          ELM_n (mp,ij3,kl3) = ELM_n (mp,ij3,kl3) + wst * a_rr_n
          ELM_kn(mp,ij3,kl3) = ELM_kn(mp,ij3,kl3) + wst * a_rr_kn

          ! Row T
          ELM_p (mp,ij4,kl1) = ELM_p (mp,ij4,kl1) + wst * a_Tp
          ELM_k (mp,ij4,kl1) = ELM_k (mp,ij4,kl1) + wst * a_Tp_k
          ELM_p (mp,ij4,kl2) = ELM_p (mp,ij4,kl2) + wst * a_Tu
          ELM_p (mp,ij4,kl4) = ELM_p (mp,ij4,kl4) + wst * a_TT
          ELM_k (mp,ij4,kl4) = ELM_k (mp,ij4,kl4) + wst * a_TT_k
          ELM_n (mp,ij4,kl4) = ELM_n (mp,ij4,kl4) + wst * a_TT_n
          ELM_kn(mp,ij4,kl4) = ELM_kn(mp,ij4,kl4) + wst * a_TT_kn

         enddo  ! l
        enddo   ! k

       enddo  ! j
      enddo   ! i

    enddo  ! mp

   enddo  ! mt
  enddo   ! ms

  !--------------------------------------------------------------------------
  ! Transform each toroidal channel and scatter into the mode-space ELM
  !--------------------------------------------------------------------------
  do i = 1, NRV
    do j = 1, NRV

      if (maxval(abs(ELM_p(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_p(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm(out_fft, i, j, ELM, DRV)
      endif

      if (maxval(abs(ELM_n(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_n(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_n(out_fft, i, j, ELM, DRV)
      endif

      if (maxval(abs(ELM_k(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_k(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_k(out_fft, i, j, ELM, DRV)
      endif

      if (maxval(abs(ELM_kn(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_kn(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_kn(out_fft, i, j, ELM, DRV)
      endif

    enddo
  enddo

  ! The scatter routines accumulate each (row, col) pair twice -- once for the
  ! l = (k-1)+(m-1) branch and once for l = (k-1)-(m-1) -- so the assembled
  ! matrix must be halved. Same convention as mod_elt_matrix_fft ("ELM = 0.5*ELM")
  ! and mod_elt_matrix_elliptic.
  ELM = 0.5d0 * ELM

end subroutine pc_elt_matrix_reduced_fft

end module mod_pc_elt_matrix_reduced_fft
