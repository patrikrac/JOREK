module mod_elt_matrix_elliptic
!----------------------------------------------------------------
! Element-level elliptic sub-matrices for the physics-based PC.
!
! Assembles four mode-space element matrices needed by the
! PCSHELL preconditioner for model199:
!
!   ELM_j   :  eq.3 (j)  x var j    1-var, amat_33 (mass)
!   ELM_w   :  eq.4 (w)  x var w    1-var, amat_44 (= ELM_j)
!   ELM_jpsi:  eq.3 (j)  x var psi  1-var, amat_31 (psi->j coupling)
!   ELM_wu  :  eq.4 (w)  x var u    1-var, amat_42 (u->w coupling)
!   
!   ELM_psi_correction: Schur correction coming from amat_31 * amat_33^{-1} * amat_13
!   ELM_u_correction  : Schur correction coming from amat_42 * amat_44^{-1} * amat_24
!   ELM_21_correction : Schur correction coming from amat_23 * amat_33^{-1} * amat_31  (off-diag, u-eq x psi-col)
!   ELM_61_correction : Schur correction coming from amat_63 * amat_33^{-1} * amat_31  (off-diag, T-eq x psi-col)
!
! Most integrands live only in ELM_p (mode-diagonal); the K_21 toroidal term
! also uses ELM_n (mode-off-diagonal, multiplied by toroidal mode number at scatter).
! FFT reconstruction is performed to obtain the mode-space blocks.
!----------------------------------------------------------------
implicit none
public :: element_matrix_elliptic, scatter_fft_to_elm

type :: type_fct_values
  real*8 :: v, v_x, v_y, v_s, v_t, v_p, v_ss, v_st, v_tt, v_xx, v_yy, v_xs, v_ys, v_xt, v_yt, v_xy
end type type_fct_values

contains

subroutine element_matrix_elliptic(element, nodes, ELM_j, ELM_w, ELM_jpsi, ELM_wu, &
                                   ELM_psi_correction, ELM_u_correction, &
                                   ELM_21_correction,  ELM_61_correction, &
                                   ELM_schur_PBP)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module, only: fftw_plan, time_evol_theta, time_evol_zeta, tstep, eta_T_dependent, eta, T_max_eta, T_0, visco_T_dependent, visco, &
                         eta_ohmic, T_max_eta_ohm, gamma, F0, mode
  use corr_neg

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)

  ! Output dimensions (compile-time constants from mod_parameters)
#define N1V  (n_vertex_max*n_degrees)
#define D1V  (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(D1V, D1V), intent(out) :: ELM_j, ELM_w
  real*8, dimension(D1V, D1V), intent(out) :: ELM_jpsi, ELM_wu
  real*8, dimension(D1V, D1V), intent(out), optional :: ELM_psi_correction, ELM_u_correction
  real*8, dimension(D1V, D1V), intent(out), optional :: ELM_21_correction,  ELM_61_correction
  real*8, dimension(D1V, D1V), intent(out), optional :: ELM_schur_PBP

  ! ELM_p workspace — local thread-stack buffers, zeroed each call
  ! ELM_p_w omitted: A_w uses the same mass integrand as A_j, so ELM_w = ELM_j
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_j
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_jpsi, ELM_p_wu
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_psi_correction, ELM_p_u_correction
  real*8, dimension(n_plane, N1V, N1V) :: ELM_kn_u_correction
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_21_correction,  ELM_p_61_correction
  real*8, dimension(n_plane, N1V, N1V) :: ELM_n_21_correction

  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_schur
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_schur_n
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_schur_kn

  ! Geometry at Gauss points (first derivatives only — no 2nd derivs needed)
  real*8, dimension(n_gauss,n_gauss)    :: x_g, x_s, x_t
  real*8, dimension(n_gauss,n_gauss)    :: x_ss, x_st, x_tt
  real*8, dimension(n_gauss,n_gauss)    :: y_g, y_s, y_t
  real*8, dimension(n_gauss,n_gauss)    :: y_ss, y_st, y_tt

  ! Variables at Gauss points
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_g
  ! Computational-coordinate s,t derivatives of background fields at Gauss points
  ! (only synthesized when off-diagonal Schur corrections are requested)
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_s, eq_t
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_ss, eq_st, eq_tt

  ! Loop indices
  integer :: i, j, ms, mt, mp, in, k, l, m, index_k, index_m
  integer :: idx_ij, idx_kl

  ! Gauss-point scalars
  real*8 :: wst, xjac, xjac_x, xjac_y, BigR
  type(type_fct_values) :: v_fct, psi_fct, u_fct
  real*8 :: amat_mass, amat_31, amat_42, amat_psi_correction, amat_u_correction
  real*8 :: amat_21_correction, amat_61_correction, amat_21_n_correction, amat_u_kn_correction

  ! Background-field values/derivatives at the current Gauss point (off-diag corrections)
  real*8 :: r0, r0_hat
  real*8 :: u0, u0_x, u0_y
  real*8 :: ps0_x, ps0_y, zj0
  real*8 :: eta_T_ohm

  ! Variables for the Schur operator weak form
  real*8 :: ps0_xx, ps0_yy, ps0_xy
  real*8 :: Q0, Q0x, Q0y, Q1, Q1x, Q1y
  real*8 :: W0, W0x, W0y, W1, W1x, W1y
  real*8 :: amat_schur, amat_schur_n, amat_schur_kn, amat_schur_inertia
  real*8 :: amat_01, amat_10

  ! --- S_PBP default model (design doc Definition 1): the assembled momentum-Schur
  !     approximation is S_PBP = inertia + viscosity + relaxed tension + S_geo. All
  !     four terms are unconditional; the gated/optional terms (kink, interchange
  !     drive, flutter, equilibrium flow) are not part of this operator. ---
  real*8, parameter :: c_lambda = 1.0d0   ! poloidal lambda_elt scale (tension relaxation)

  real*8  :: amat_schur_geo, p0, zeta

  ! Element-representative geometry for the resistive tension relaxation:
  !   area_elt ~ h^2 (poloidal 1/h^2 proxy), R0 = area-weighted mean major radius.
  real*8  :: area_elt, R0_elt

  ! Split S_PBP accumulators: inertia (Term 1, from Atilde_22, NEVER relaxed) is kept
  ! separate from the ideal tension (Term 2 = Atilde_21 Atilde_11^-1 B_12) so the
  ! per-harmonic resistive relaxation can scale ONLY the tension.
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_schur_inertia
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_schur_geo      ! geodesic compression S_geo (n0 channel)
  real*8, dimension(D1V, D1V)          :: ELM_tension

  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_visco     ! poloidal viscous integrand
  real*8, dimension(n_plane, N1V, N1V) :: ELM_kn_visco    ! toroidal viscous integrand (kn channel)
  real*8 :: amat_visco, amat_visco_kn

  ! Values at Gauss points
  real*8 :: T0

  ! Time stepping variables
  real*8 :: theta
  real*8 :: eta_T, visco_T

  ! Schur correction
  logical :: psi_correction, u_correction, correction_21, correction_61
  logical :: schur_PBP

  ! FFT workspace
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ! Check for optional Schur correction output
  psi_correction = present(ELM_psi_correction)
  u_correction = present(ELM_u_correction)

  correction_21 = present(ELM_21_correction)
  correction_61 = present(ELM_61_correction)

  schur_PBP = present(ELM_schur_PBP)

  !-----------------------------------------------------------------
  ! Initialise
  !-----------------------------------------------------------------
  ELM_j    = 0.d0
  ELM_jpsi = 0.d0;  ELM_wu     = 0.d0
  if (psi_correction) ELM_psi_correction = 0.d0
  if (u_correction) ELM_u_correction = 0.d0
  if (correction_21) ELM_21_correction = 0.d0
  if (correction_61) ELM_61_correction = 0.d0
  ELM_p_j  = 0.d0
  ELM_p_jpsi = 0.d0;  ELM_p_wu = 0.d0
  if (psi_correction) ELM_p_psi_correction = 0.d0
  if (u_correction) ELM_p_u_correction = 0.d0
  if (u_correction)   ELM_kn_u_correction  = 0.d0
  if (correction_21)  ELM_p_21_correction    = 0.d0
  if (correction_21)  ELM_n_21_correction  = 0.d0
  if (correction_61)  ELM_p_61_correction    = 0.d0
  if (schur_PBP) then
    ELM_p_schur_inertia = 0.d0   ! Term 1 (inertia, never relaxed)
    ELM_p_schur    = 0.d0        ! Term 2 poloidal tension (relaxable)
    ELM_p_schur_n  = 0.d0        ! Term 2 toroidal-cross tension
    ELM_p_schur_kn = 0.d0        ! Term 2 toroidal-squared tension
    ELM_tension    = 0.d0
    ELM_p_schur_geo = 0.d0       ! S_geo geodesic compression
    ELM_p_visco  = 0.d0
    ELM_kn_visco = 0.d0
    area_elt = 0.d0
    R0_elt   = 0.d0
  endif
  

  !-----------------------------------------------------------------
  ! Geometry at Gauss points
  !-----------------------------------------------------------------
  x_g  = 0.d0; x_s  = 0.d0; x_t  = 0.d0; x_st  = 0.d0; x_ss  = 0.d0; x_tt  = 0.d0;
  y_g  = 0.d0; y_s  = 0.d0; y_t  = 0.d0; y_st  = 0.d0; y_ss  = 0.d0; y_tt  = 0.d0;
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0
  eq_ss = 0.d0; eq_st = 0.d0; eq_tt = 0.d0

  theta = time_evol_theta
  zeta  = time_evol_zeta

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
          do mp=1,n_plane
            do k=1,n_var
              do in=1,n_tor
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)   * HZ(in,mp)
                eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt) * HZ(in,mp)
                eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt) * HZ(in,mp)
                eq_ss(mp,k,ms,mt) = eq_ss(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_ss(i,j,ms,mt) * HZ(in,mp)
                eq_st(mp,k,ms,mt) = eq_st(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_st(i,j,ms,mt) * HZ(in,mp)
                eq_tt(mp,k,ms,mt) = eq_tt(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_tt(i,j,ms,mt) * HZ(in,mp)
              enddo
            enddo
          enddo

        enddo
      enddo
    enddo
  enddo

  !-----------------------------------------------------------------
  ! Gauss integration loop
  !-----------------------------------------------------------------
  do ms = 1, n_gauss
    do mt = 1, n_gauss

      wst  = wgauss(ms) * wgauss(mt)
      xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)
      xjac_x  = (x_ss(ms,mt)*y_t(ms,mt)**2 - y_ss(ms,mt)*x_t(ms,mt)*y_t(ms,mt) - 2.d0*x_st(ms,mt)*y_s(ms,mt)*y_t(ms,mt)   &
                + y_st(ms,mt)*(x_s(ms,mt)*y_t(ms,mt) + x_t(ms,mt)*y_s(ms,mt))                                               &
                + x_tt(ms,mt)*y_s(ms,mt)**2 - y_tt(ms,mt)*x_s(ms,mt)*y_s(ms,mt)) / xjac
      xjac_y  = (y_tt(ms,mt)*x_s(ms,mt)**2 - x_tt(ms,mt)*y_s(ms,mt)*x_s(ms,mt) - 2.d0*y_st(ms,mt)*x_t(ms,mt)*x_s(ms,mt)   &
                + x_st(ms,mt)*(y_t(ms,mt)*x_s(ms,mt) + y_s(ms,mt)*x_t(ms,mt))                                               &
                + y_ss(ms,mt)*x_t(ms,mt)**2 - x_ss(ms,mt)*y_t(ms,mt)*x_t(ms,mt)) / xjac
      BigR = x_g(ms,mt)

      ! Element geometry accumulation for the resistive relaxation (mode 1).
      ! area_elt ~ h^2; R0_elt = area-weighted mean major radius.
      if (present(ELM_schur_PBP)) then
        area_elt = area_elt + wst * xjac
        R0_elt   = R0_elt   + wst * xjac * BigR
      endif

      do mp = 1, n_plane

        T0 = abs(eq_g(mp,6,ms,mt))
        ! --- Temperature dependent resistivity
        if (eta_T_dependent .and. corr_neg_temp(T0) <= T_max_eta) then
          eta_T     = eta   * (corr_neg_temp(T0)/T_0)**(-1.5d0)
        else if ( eta_T_dependent .and. corr_neg_temp(T0) > T_max_eta) then
          eta_T     = eta   * (T_max_eta/T_0)**(-1.5d0)
        else
          eta_T     = eta
        end if
        ! --- Temperature dependent ohmic-heating resistivity (model199 lines 308-317)
        if (eta_T_dependent .and. corr_neg_temp(T0) <= T_max_eta_ohm) then
          eta_T_ohm = eta_ohmic * (corr_neg_temp(T0)/T_0)**(-1.5d0)
        else if (eta_T_dependent .and. corr_neg_temp(T0) > T_max_eta_ohm) then
          eta_T_ohm = eta_ohmic * (T_max_eta_ohm/T_0)**(-1.5d0)
        else
          eta_T_ohm = eta_ohmic
        end if
        ! --- Temperature dependent viscosity
        if (visco_T_dependent) then
          visco_T   = visco * (corr_neg_temp(T0)/T_0)**(-1.5d0)
        else
          visco_T   = visco
        end if

        r0    = abs(eq_g(mp,5,ms,mt))
        r0_hat   = BigR**2 * r0

        u0    = eq_g(mp,2,ms,mt)
        u0_x  = (   y_t(ms,mt) * eq_s(mp,2,ms,mt) - y_s(ms,mt) * eq_t(mp,2,ms,mt) ) / xjac
        u0_y  = ( - x_t(ms,mt) * eq_s(mp,2,ms,mt) + x_s(ms,mt) * eq_t(mp,2,ms,mt) ) / xjac

        ! --- Background field extraction for off-diagonal Schur corrections
        ! ps0 spatial gradient (Cartesian) from synthesized s,t derivatives
        ps0_x = (   y_t(ms,mt) * eq_s(mp,var_psi,ms,mt) - y_s(ms,mt) * eq_t(mp,var_psi,ms,mt) ) / xjac
        ps0_y = ( - x_t(ms,mt) * eq_s(mp,var_psi,ms,mt) + x_s(ms,mt) * eq_t(mp,var_psi,ms,mt) ) / xjac

        ! Second derivatives of ps0 via coordinate transformation
        ps0_xx = (eq_ss(mp,var_psi,ms,mt) * y_t(ms,mt)**2 - 2.d0*eq_st(mp,var_psi,ms,mt) * y_s(ms,mt)*y_t(ms,mt) + eq_tt(mp,var_psi,ms,mt) * y_s(ms,mt)**2  &
              + eq_s(mp,var_psi,ms,mt) * (y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt) )                              &
              + eq_t(mp,var_psi,ms,mt) * (y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt) ) )    / xjac**2               &
              - xjac_x * (eq_s(mp,var_psi,ms,mt) * y_t(ms,mt) - eq_t(mp,var_psi,ms,mt) * y_s(ms,mt)) / xjac**2
        ps0_yy = (eq_ss(mp,var_psi,ms,mt) * x_t(ms,mt)**2 - 2.d0*eq_st(mp,var_psi,ms,mt) * x_s(ms,mt)*x_t(ms,mt) + eq_tt(mp,var_psi,ms,mt) * x_s(ms,mt)**2  &
              + eq_s(mp,var_psi,ms,mt) * (x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt) )                              &
              + eq_t(mp,var_psi,ms,mt) * (x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt) ) )    / xjac**2               &
              - xjac_y * (- eq_s(mp,var_psi,ms,mt) * x_t(ms,mt) + eq_t(mp,var_psi,ms,mt) * x_s(ms,mt) ) / xjac**2
        ps0_xy = (- eq_ss(mp,var_psi,ms,mt) * y_t(ms,mt)*x_t(ms,mt) - eq_tt(mp,var_psi,ms,mt) * x_s(ms,mt)*y_s(ms,mt)                    &
              + eq_st(mp,var_psi,ms,mt) * (y_s(ms,mt)*x_t(ms,mt)  + y_t(ms,mt)*x_s(ms,mt)  )                             &
              - eq_s(mp,var_psi,ms,mt)  * (x_st(ms,mt)*y_t(ms,mt) - x_tt(ms,mt)*y_s(ms,mt) )                             &
              - eq_t(mp,var_psi,ms,mt)  * (x_st(ms,mt)*y_s(ms,mt) - x_ss(ms,mt)*y_t(ms,mt) )  )  / xjac**2               &
              - xjac_x * (- eq_s(mp,var_psi,ms,mt) * x_t(ms,mt) + eq_t(mp,var_psi,ms,mt) * x_s(ms,mt) )   / xjac**2

        ! Background current density (value only — used as scalar weight in K_61)
        zj0   = eq_g(mp,var_zj,ms,mt)

        ! Background pressure p0 = rho0 * T0 (for S_geo geodesic compression)
        p0    = eq_g(mp,var_rho,ms,mt) * eq_g(mp,var_T,ms,mt)

        do i = 1, n_vertex_max
          do j = 1, n_degrees

            ! 1-var DOF index (test, row)
            idx_ij = n_degrees*(i-1) + (j-1) + 1

            ! test function value and gradients
            v_fct%v   =   H(i,j,ms,mt)   * element%size(i,j)
            v_fct%v_x = (   y_t(ms,mt) * H_s(i,j,ms,mt) &
                    - y_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac
            v_fct%v_y = ( - x_t(ms,mt) * H_s(i,j,ms,mt) &
                    + x_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac

            v_fct%v_s = h_s(i,j,ms,mt) * element%size(i,j)
            v_fct%v_t = h_t(i,j,ms,mt) * element%size(i,j)
            v_fct%v_p = H(i,j,ms,mt)   * element%size(i,j)
            
            v_fct%v_ss = h_ss(i,j,ms,mt) * element%size(i,j) 
            v_fct%v_tt = h_tt(i,j,ms,mt) * element%size(i,j) 
            v_fct%v_st = h_st(i,j,ms,mt) * element%size(i,j)

            v_fct%v_xx = (v_fct%v_ss * y_t(ms,mt)**2 - 2.d0*v_fct%v_st * y_s(ms,mt)*y_t(ms,mt) + v_fct%v_tt * y_s(ms,mt)**2  &
                  + v_fct%v_s * (y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt) )                          &
                  + v_fct%v_t * (y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt) ) )  / xjac**2             &	
                  - xjac_x * (v_fct%v_s * y_t(ms,mt) - v_fct%v_t * y_s(ms,mt)) / xjac**2

            v_fct%v_yy = (v_fct%v_ss * x_t(ms,mt)**2 - 2.d0*v_fct%v_st * x_s(ms,mt)*x_t(ms,mt) + v_fct%v_tt * x_s(ms,mt)**2  &
                  + v_fct%v_s * (x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt) )                          &
                  + v_fct%v_t * (x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt) ) )     / xjac**2          &	
                  - xjac_y * (- v_fct%v_s * x_t(ms,mt) + v_fct%v_t * x_s(ms,mt) ) / xjac**2

            v_fct%v_xy = (- v_fct%v_ss * y_t(ms,mt)*x_t(ms,mt) - v_fct%v_tt * x_s(ms,mt)*y_s(ms,mt)                    &
                  + v_fct%v_st * (y_s(ms,mt)*x_t(ms,mt)  + y_t(ms,mt)*x_s(ms,mt)  )                         &
                  - v_fct%v_s  * (x_st(ms,mt)*y_t(ms,mt) - x_tt(ms,mt)*y_s(ms,mt) )                         &
                  - v_fct%v_t  * (x_st(ms,mt)*y_s(ms,mt) - x_ss(ms,mt)*y_t(ms,mt) )  )  / xjac**2           &
                  - xjac_x * (- v_fct%v_s * x_t(ms,mt) + v_fct%v_t * x_s(ms,mt) )   / xjac**2

            do k = 1, n_vertex_max
              do l = 1, n_degrees

                ! 1-var DOF index (trial, col)
                idx_kl = n_degrees*(k-1) + (l-1) + 1

                ! trial function value and gradients
                psi_fct%v   =   H(k,l,ms,mt)   * element%size(k,l)
                psi_fct%v_x = (   y_t(ms,mt) * H_s(k,l,ms,mt) &
                          - y_s(ms,mt) * H_t(k,l,ms,mt) ) / xjac * element%size(k,l)
                psi_fct%v_y = ( - x_t(ms,mt) * H_s(k,l,ms,mt) &
                          + x_s(ms,mt) * H_t(k,l,ms,mt) ) / xjac * element%size(k,l)
                psi_fct%v_p = H(k,l,ms,mt)   * element%size(k,l)
                psi_fct%v_s = h_s(k,l,ms,mt) * element%size(k,l)
                psi_fct%v_t = h_t(k,l,ms,mt) * element%size(k,l)
                psi_fct%v_ss = h_ss(k,l,ms,mt) * element%size(k,l) 
                psi_fct%v_tt = h_tt(k,l,ms,mt) * element%size(k,l) 
                psi_fct%v_st = h_st(k,l,ms,mt) * element%size(k,l)      
                psi_fct%v_xx = (psi_fct%v_ss * y_t(ms,mt)**2 - 2.d0*psi_fct%v_st * y_s(ms,mt)*y_t(ms,mt) + psi_fct%v_tt * y_s(ms,mt)**2  &
                        + psi_fct%v_s * (y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt) )                              &
                        + psi_fct%v_t * (y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt) ) )    / xjac**2               &	
                        - xjac_x * (psi_fct%v_s * y_t(ms,mt) - psi_fct%v_t * y_s(ms,mt)) / xjac**2
                psi_fct%v_yy = (psi_fct%v_ss * x_t(ms,mt)**2 - 2.d0*psi_fct%v_st * x_s(ms,mt)*x_t(ms,mt) + psi_fct%v_tt * x_s(ms,mt)**2  &
                        + psi_fct%v_s * (x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt) )                              &
                        + psi_fct%v_t * (x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt) ) )    / xjac**2               &
                        - xjac_y * (- psi_fct%v_s * x_t(ms,mt) + psi_fct%v_t * x_s(ms,mt) ) / xjac**2
                psi_fct%v_xy = (- psi_fct%v_ss * y_t(ms,mt)*x_t(ms,mt) - psi_fct%v_tt * x_s(ms,mt)*y_s(ms,mt)                    &
                      + psi_fct%v_st * (y_s(ms,mt)*x_t(ms,mt)  + y_t(ms,mt)*x_s(ms,mt)  )                         &
                      - psi_fct%v_s  * (x_st(ms,mt)*y_t(ms,mt) - x_tt(ms,mt)*y_s(ms,mt) )                         &
                      - psi_fct%v_t  * (x_st(ms,mt)*y_s(ms,mt) - x_ss(ms,mt)*y_t(ms,mt) )  )  / xjac**2           &
                      - xjac_x * (- psi_fct%v_s * x_t(ms,mt) + psi_fct%v_t * x_s(ms,mt) )   / xjac**2

                u_fct = psi_fct

                !--- eq.3/4: amat_33 = amat_44 = v_fct%v * psi_fct%v * BigR * xjac
                amat_mass = v_fct%v * psi_fct%v * BigR * xjac

                !--- eq.3: amat_31 (psi->j coupling)
                amat_31 = (v_fct%v_x*psi_fct%v_x + v_fct%v_y*psi_fct%v_y) * BigR * xjac &
                        + 2.d0 * v_fct%v * psi_fct%v_x * BigR * xjac

                !--- eq.4: amat_42 (u->w coupling)
                amat_42 = (v_fct%v_x*psi_fct%v_x + v_fct%v_y*psi_fct%v_y) * BigR * xjac

                ! ============================================================
                ! S_PBP: element approximation of the velocity Schur complement
                !   S_u = Atilde_22 - Atilde_21 * Atilde_11^-1 * B_12 - (pressure channel)
                ! Default model (design doc Definition 1), all terms unconditional:
                !   inertia (from Atilde_22) + viscosity (Atilde_22 enrichment)
                !   + relaxed Alfven tension (tau_R(n), resistive Atilde_11^-1)
                !   + S_geo geodesic compression (pressure channel).
                ! ============================================================
                ! --- Schur correction equation ---
                amat_psi_correction = - (eta_T / BigR) * (v_fct%v_x * psi_fct%v_x + v_fct%v_y * psi_fct%v_y) * xjac * theta * tstep

                amat_u_correction = - r0_hat * BigR**2 * (u_fct%v_xx + u_fct%v_x/BigR + u_fct%v_yy) * ( v_fct%v_x * u0_y - v_fct%v_y * u0_x) * xjac * theta * tstep  &
                                      + visco_T * BigR * (v_fct%v_xx + v_fct%v_x/BigR + v_fct%v_yy) * (u_fct%v_xx + u_fct%v_x/BigR + u_fct%v_yy) * xjac * theta * tstep 
                                      
                amat_u_kn_correction = - visco_T * (1.d0 / BigR) * v_fct%v * (u_fct%v_xx + u_fct%v_x/BigR + u_fct%v_yy) * xjac * theta * tstep 
                                        !+ visco_T * 1.d0 / BigR * (v_fct%v_x * u_fct%v_x + v_fct%v_y * u_fct%v_y) * xjac * theta * tstep &
                                        !- visco_T * (1.d0 + (1.d0 / BigR**2)) * v_fct%v * u_fct%v_x * xjac * theta * tstep

                ! Off-diagonal Schur corrections 
                amat_21_correction = (v_fct%v_x * ps0_y - v_fct%v_y * ps0_x) &
                                     * (psi_fct%v_xx - psi_fct%v_x/BigR + psi_fct%v_yy) * xjac * theta * tstep

                amat_21_n_correction =  F0 / BigR * (v_fct%v_x * psi_fct%v_x + v_fct%v_y * psi_fct%v_y) * xjac * theta * tstep

                amat_61_correction = - 2.d0 * (gamma - 1.d0) * eta_T_ohm * zj0 / BigR &
                                     * (v_fct%v_x * psi_fct%v_x + v_fct%v_y * psi_fct%v_y) &
                                     * xjac * theta * tstep

                ! 1. Poloidal parts of grad_parallel (Q0 for test v, W0 for trial u)
                Q0  = (ps0_x * v_fct%v_y - ps0_y * v_fct%v_x) / BigR
                Q0x = -(ps0_x * v_fct%v_y - ps0_y * v_fct%v_x) / BigR**2 &
                      + (ps0_xx * v_fct%v_y + ps0_x * v_fct%v_xy - ps0_xy * v_fct%v_x - ps0_y * v_fct%v_xx) / BigR
                Q0y = (ps0_xy * v_fct%v_y + ps0_x * v_fct%v_yy - ps0_yy * v_fct%v_x - ps0_y * v_fct%v_xy) / BigR
                
                W0  = (ps0_x * u_fct%v_y - ps0_y * u_fct%v_x) / BigR
                W0x = -(ps0_x * u_fct%v_y - ps0_y * u_fct%v_x) / BigR**2 &
                      + (ps0_xx * u_fct%v_y + ps0_x * u_fct%v_xy - ps0_xy * u_fct%v_x - ps0_y * u_fct%v_xx) / BigR
                W0y = (ps0_xy * u_fct%v_y + ps0_x * u_fct%v_yy - ps0_yy * u_fct%v_x - ps0_y * u_fct%v_xy) / BigR
                
                ! 2. Toroidal parts of grad_parallel (Q1 for test v, W1 for trial u)
                Q1  = F0 / BigR**2 * v_fct%v
                Q1x = -2.d0 * F0 / BigR**3 * v_fct%v + F0 / BigR**2 * v_fct%v_x
                Q1y = F0 / BigR**2 * v_fct%v_y

                W1  = F0 / BigR**2 * u_fct%v
                W1x = -2.d0 * F0 / BigR**3 * u_fct%v + F0 / BigR**2 * u_fct%v_x
                W1y = F0 / BigR**2 * u_fct%v_y

                ! 3. Construct the Matrix Entries
                ! Term 1: Inertia (from Atilde_22) — NOT relaxed
                ! Carries the Gears mass factor (1+zeta), matching model199 amat_22
                ! (unity for Crank-Nicolson, zeta=0). Keeps the inertia:tension:S_geo
                ! scaling consistent with the exact S_u (mass ~ (1+zeta), couplings ~
                ! (theta*dt)^2/(1+zeta)).
                amat_schur_inertia = - r0_hat * (v_fct%v_x * u_fct%v_x + v_fct%v_y * u_fct%v_y) * BigR * xjac * (1.d0 + zeta)
                ! Term 2: ideal poloidal Alfven tension (from Atilde_21 Atilde_11^-1 B_12) — relaxable
                ! Gears 1/(1+zeta) factor (design doc Remark 2; unity for Crank-Nicolson, zeta=0).
                amat_schur = - (theta*tstep)**2 / (1.d0+zeta) * ( (Q0x * W0x + Q0y * W0y) + (2.d0 / BigR) * Q0 * W0x ) * BigR * xjac

                ! Geodesic compression S_geo (design doc eq.24):
                !   -(4 gamma/(1+zeta)) (theta*dt)^2 * R * p0 * (d_Z v)(d_Z u).
                ! d_Z = d_y (poloidal Z). Negative leading sign -> ADDS dissipation
                ! in the negative-definite S_PBP convention (same as amat_schur).
                amat_schur_geo = - (4.d0*gamma/(1.d0+zeta)) * (theta*tstep)**2 &
                                   * p0 * v_fct%v_y * u_fct%v_y * BigR * xjac

                ! Viscosity (Atilde_22 enrichment): poloidal biharmonic + toroidal companion.
                ! SIGN: all S_PBP integrands carry the SAME leading minus (inertia, tension,
                ! S_geo are all negative). S_PBP is assembled negative-(semi)definite, so a
                ! negative integrand ADDS dissipation. The n0 (scatter_fft_to_elm) and n2
                ! (scatter_fft_to_elm_kn) channels both match the validated amat_schur_kn
                ! convention.
                amat_visco    = - visco_T * BigR * (v_fct%v_xx + v_fct%v_x/BigR + v_fct%v_yy) &
                                            * (u_fct%v_xx + u_fct%v_x/BigR + u_fct%v_yy) * xjac * theta * tstep
                amat_visco_kn = - visco_T * (1.d0 / BigR) * v_fct%v &
                                            * (u_fct%v_xx + u_fct%v_x/BigR + u_fct%v_yy) * xjac * theta * tstep

                ! Term 2: dt^2 * [grad_pol q * grad_pol w + 2/R * q * w_R]
                ! Part 00: purely poloidal (no phi derivatives)
                !amat_schur = amat_schur + (tstep**2) * ( (Q0x * W0x + Q0y * W0y) + (2.d0 / BigR) * Q0 * W0x ) * BigR * xjac

                ! Part n: single phi derivative cross-terms. 
                ! amat_01 is d/dphi on trial function (u).
                ! amat_10 is d/dphi on test function (v). Toroidal IBP flips the sign and transfers it to 'u'.
                amat_01 = (Q0x * W1x + Q0y * W1y) + (2.d0 / BigR) * Q0 * W1x
                amat_10 = (Q1x * W0x + Q1y * W0y) + (2.d0 / BigR) * Q1 * W0x
                amat_schur_n = - (theta*tstep)**2 / (1.d0+zeta) * (amat_01 - amat_10) * BigR * xjac

                ! Part kn: d/dphi on BOTH test and trial functions
                amat_schur_kn = - (theta*tstep)**2 / (1.d0+zeta) * ((Q1x * W1x + Q1y * W1y) + (2.d0 / BigR) * Q1 * W1x ) * BigR * xjac !- amat_u_kn_correction
                
                ! --- 1-var mass (A_w = A_j assigned after FFT) ---
                ELM_p_j(mp, idx_ij, idx_kl) = ELM_p_j(mp, idx_ij, idx_kl) + wst * amat_mass

                ! --- 1-var off-diagonal couplings ---
                ELM_p_jpsi(mp, idx_ij, idx_kl) = ELM_p_jpsi(mp, idx_ij, idx_kl) + wst * amat_31
                ELM_p_wu  (mp, idx_ij, idx_kl) = ELM_p_wu  (mp, idx_ij, idx_kl) + wst * amat_42

                ! --- Schur correction ---
                if (psi_correction) then
                  ELM_p_psi_correction(mp, idx_ij, idx_kl) = ELM_p_psi_correction(mp, idx_ij, idx_kl) + wst * amat_psi_correction
                endif
                if (u_correction) then
                  ELM_p_u_correction(mp, idx_ij, idx_kl) = ELM_p_u_correction(mp, idx_ij, idx_kl) + wst * amat_u_correction
                  ELM_kn_u_correction(mp, idx_ij, idx_kl) = ELM_kn_u_correction(mp, idx_ij, idx_kl) + wst * amat_u_kn_correction
                endif
                if (correction_21) then
                  ELM_p_21_correction(mp, idx_ij, idx_kl) = ELM_p_21_correction(mp, idx_ij, idx_kl) + wst * amat_21_correction
                  ELM_n_21_correction(mp, idx_ij, idx_kl) = ELM_n_21_correction(mp, idx_ij, idx_kl) + wst * amat_21_n_correction
                endif
                if (correction_61) then
                  ELM_p_61_correction(mp, idx_ij, idx_kl) = ELM_p_61_correction(mp, idx_ij, idx_kl) + wst * amat_61_correction
                endif

                if (schur_PBP) then
                  ELM_p_schur_inertia(mp, idx_ij, idx_kl) = ELM_p_schur_inertia(mp, idx_ij, idx_kl) + wst * amat_schur_inertia
                  ELM_p_schur(mp, idx_ij, idx_kl)    = ELM_p_schur(mp, idx_ij, idx_kl)    + wst * amat_schur
                  ELM_p_schur_n(mp, idx_ij, idx_kl)  = ELM_p_schur_n(mp, idx_ij, idx_kl)  + wst * amat_schur_n
                  ELM_p_schur_kn(mp, idx_ij, idx_kl) = ELM_p_schur_kn(mp, idx_ij, idx_kl) + wst * amat_schur_kn
                  ELM_p_schur_geo(mp, idx_ij, idx_kl) = ELM_p_schur_geo(mp, idx_ij, idx_kl) + wst * amat_schur_geo
                  ELM_p_visco(mp, idx_ij, idx_kl)  = ELM_p_visco(mp, idx_ij, idx_kl)  + wst * amat_visco
                  ELM_kn_visco(mp, idx_ij, idx_kl) = ELM_kn_visco(mp, idx_ij, idx_kl) + wst * amat_visco_kn
                endif

              enddo  ! l
            enddo    ! k

          enddo  ! j
        enddo    ! i

      enddo  ! mp

    enddo  ! mt
  enddo    ! ms

  !-----------------------------------------------------------------
  ! FFT reconstruction — 1-var matrix 
  !-----------------------------------------------------------------
  do i = 1, N1V
    do j = 1, N1V
      in_fft = ELM_p_j(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_j, D1V)

      if (psi_correction) then
        in_fft = ELM_p_psi_correction(1:n_plane, i, j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm(out_fft, i, j, ELM_psi_correction, D1V)
      endif
      if (u_correction) then
        in_fft = ELM_p_u_correction(1:n_plane, i, j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm(out_fft, i, j, ELM_u_correction, D1V)

        in_fft = ELM_kn_u_correction(1:n_plane, i, j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_kn(out_fft, i, j, ELM_u_correction, D1V)
      endif
      if (correction_21) then
        in_fft = ELM_p_21_correction(1:n_plane, i, j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm(out_fft, i, j, ELM_21_correction, D1V)

        in_fft = ELM_n_21_correction(1:n_plane, i, j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_n(out_fft, i, j, ELM_21_correction, D1V)
      endif
      if (correction_61) then
        in_fft = ELM_p_61_correction(1:n_plane, i, j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm(out_fft, i, j, ELM_61_correction, D1V)
      endif

        if (schur_PBP) then
          ! Inertia channel -> ELM_schur_PBP directly (never relaxed)
          in_fft = ELM_p_schur_inertia(1:n_plane, i, j)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
          call scatter_fft_to_elm(out_fft, i, j, ELM_schur_PBP, D1V)

          ! Geodesic compression S_geo: poloidal (n0) channel, directly into
          ! ELM_schur_PBP (not relaxed; the d_Z v * d_Z u factor has no phi deriv,
          ! the phi-coupling comes from the 3D p0). Gets the 0.5 scaling below.
          in_fft = ELM_p_schur_geo(1:n_plane, i, j)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
          call scatter_fft_to_elm(out_fft, i, j, ELM_schur_PBP, D1V)

          ! Tension channels -> ELM_tension (relaxed later)
          in_fft = ELM_p_schur(1:n_plane, i, j)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
          call scatter_fft_to_elm(out_fft, i, j, ELM_tension, D1V)

          in_fft = ELM_p_schur_n(1:n_plane, i, j)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
          call scatter_fft_to_elm_n(out_fft, i, j, ELM_tension, D1V)

          in_fft = ELM_p_schur_kn(1:n_plane, i, j)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
          call scatter_fft_to_elm_kn(out_fft, i, j, ELM_tension, D1V)

          ! Viscosity: n0 (poloidal) + kn (toroidal) channels into ELM_schur_PBP.
          in_fft = ELM_p_visco(1:n_plane, i, j)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
          call scatter_fft_to_elm(out_fft, i, j, ELM_schur_PBP, D1V)

          in_fft = ELM_kn_visco(1:n_plane, i, j)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
          call scatter_fft_to_elm_kn(out_fft, i, j, ELM_schur_PBP, D1V)
        endif

    enddo
  enddo

  ELM_j = 0.5d0 * ELM_j
  ELM_w = ELM_j   ! identical mass integrand: amat_33 = amat_44
  if (psi_correction) ELM_psi_correction = 0.5d0 * ELM_psi_correction
  if (u_correction) ELM_u_correction = 0.5d0 * ELM_u_correction
  if (correction_21) ELM_21_correction = 0.5d0 * ELM_21_correction
  if (correction_61) ELM_61_correction = 0.5d0 * ELM_61_correction
  if (schur_PBP) then
    ! Resistive relaxation of the ideal tension (models the resistive Atilde_11^-1):
    !   tau_R(n) = (theta*dt)^2 / (1 + theta*eta_T*dt*lambda_elt(n))
    !   lambda_elt(n) = c_lambda/area_elt + (mode(n)/R0_elt)^2   (poloidal 1/h^2 + exact toroidal)
    ! Applied per-harmonic (nonlinear in n) as a row scaling of the harmonic-diagonal
    ! tension block by 1/(1 + theta*eta_T*dt*lambda_elt(n)) (the (theta*dt)^2 already lives
    ! in amat_schur). Reduces to the ideal limit as eta_T -> 0.
    if (area_elt > 0.d0) R0_elt = R0_elt / area_elt   ! area-weighted mean major radius
    call apply_schur_relaxation(ELM_tension, D1V, area_elt, R0_elt, &
                                c_lambda, eta_T, theta, tstep)

    ELM_schur_PBP = ELM_schur_PBP + ELM_tension
    ELM_schur_PBP = 0.5d0 * ELM_schur_PBP
  endif

  !-----------------------------------------------------------------
  ! FFT reconstruction — 1-var off-diagonal coupling matrices (A_jpsi, A_wu)
  !-----------------------------------------------------------------
  do i = 1, N1V
    do j = 1, N1V

      in_fft = ELM_p_jpsi(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_jpsi, D1V)

      in_fft = ELM_p_wu(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_wu, D1V)

    enddo
  enddo

  ELM_jpsi = 0.5d0 * ELM_jpsi
  ELM_wu   = 0.5d0 * ELM_wu

end subroutine element_matrix_elliptic

!-----------------------------------------------------------------
! Scatter FFT output for row i, col j into ELM.
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    index_k = n_tor*(i-1) + max(2*(k-1), 1)

    do m = 1, (n_tor+1)/2

      index_m = n_tor*(j-1) + max(2*(m-1), 1)

      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(l+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(l+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(l+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) -  real(out_fft(l+1))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(abs(l)+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(abs(l)+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(abs(l)+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) -  real(out_fft(abs(l)+1))
      endif

      l = (k-1) - (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(l+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(l+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(l+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) +  real(out_fft(l+1))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(abs(l)+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(abs(l)+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(abs(l)+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) +  real(out_fft(abs(l)+1))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm

!-----------------------------------------------------------------
! Scatter FFT output for row i, col j into ELM with toroidal
! mode-number weighting: ELM_n equivalent of scatter_fft_to_elm.
! Multiplies each entry by float(mode(im)) where im indexes the
! toroidal wavenumber of the trial (column) function.
! Sign pattern corresponds to d/dphi -> i*n on Fourier modes.
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm_n(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane
  use phys_module,    only: mode

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, im, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    index_k = n_tor*(i-1) + max(2*(k-1), 1)

    do m = 1, (n_tor+1)/2

      im      = max(2*(m-1), 1)
      index_m = n_tor*(j-1) + im

      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(l+1))        * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(l+1))        * float(mode(im))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(abs(l)+1))   * float(mode(im))
      endif

      l = (k-1) - (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - real(out_fft(l+1))        * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(l+1))        * float(mode(im))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(abs(l)+1))   * float(mode(im))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm_n

!-----------------------------------------------------------------
! Scatter FFT output for row i, col j into ELM with toroidal
! mode-number weighting on both test AND trial functions: ELM_kn
! equivalent. Multiplies by float(mode(ik)) * float(mode(im)).
! Sign pattern corresponds to d/dphi -> i*n on both functions,
! giving a real factor -n_k*n_m (i² = -1).
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm_kn(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane
  use phys_module,    only: mode

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, ik, im, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    ik      = max(2*(k-1), 1)
    index_k = n_tor*(i-1) + ik

    do m = 1, (n_tor+1)/2

      im      = max(2*(m-1), 1)
      index_m = n_tor*(j-1) + im

      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
      endif

      l = (k-1) - (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm_kn

!-----------------------------------------------------------------
! Per-harmonic resistive relaxation of the ideal Alfven tension block.
!
! Scales each row of the (harmonic-diagonal) tension matrix by
!   f(n) = 1 / (1 + theta*eta*dt*lambda_elt(n)),
!   lambda_elt(n) = c_lambda/area_elt + (mode(n)/R0)^2,
! where the toroidal number for global 1-var index idx is
!   n = mode( mod(idx-1, n_tor) + 1 )
! (the `mode` array is offset-indexed; cos/sin of a harmonic share n).
! Symmetry-preserving: the tension is block-diagonal in the toroidal harmonic, so
! for every non-zero entry row-harmonic == col-harmonic, hence row scaling by f(n)
! scales each diagonal block by the scalar f(n).
!-----------------------------------------------------------------
subroutine apply_schur_relaxation(ELM, ndim, area_elt, R0, c_lambda, eta_loc, theta, dt)

  use mod_parameters, only: n_tor
  use phys_module,    only: mode

  implicit none

  integer, intent(in)    :: ndim
  real*8,  intent(inout) :: ELM(ndim, ndim)
  real*8,  intent(in)    :: area_elt, R0, c_lambda, eta_loc, theta, dt

  integer :: r, off, n_tor_num
  real*8  :: lambda_elt, fac, kpol2, ktor2

  kpol2 = 0.d0
  if (area_elt > 0.d0) kpol2 = c_lambda / area_elt

  do r = 1, ndim
    off       = mod(r-1, n_tor) + 1
    n_tor_num = mode(off)
    ktor2     = 0.d0
    if (R0 > 0.d0) ktor2 = (dble(n_tor_num) / R0)**2
    lambda_elt = kpol2 + ktor2
    fac = 1.d0 / (1.d0 + theta * eta_loc * dt * lambda_elt)
    ELM(r, 1:ndim) = fac * ELM(r, 1:ndim)
  enddo

end subroutine apply_schur_relaxation

end module mod_elt_matrix_elliptic
