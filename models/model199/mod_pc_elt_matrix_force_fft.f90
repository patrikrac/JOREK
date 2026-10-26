!> The force operator W of the SF preconditioner's pair_w, the matrix ADDED
!! to B_22: the curvature part of the composed momentum Schur complement
!! (Chacon JCP 526 (2025) Eq. (19); the bending and kink parts go through
!! pair_w's explicit (psi, j) rows, mod_petsc_pc_sf_mixed). With
!! [a,b] = a_R b_Z - a_Z b_R and p0 = rho0 T0,
!!
!!   W(du,v) = (theta dt)^2/opz int ( - 2 R^2 dZ v [p0,du] - 4 gamma R p0 dZ v dZ du ) dR dZ
!!
!! both halves of -grad(dv.grad p0 + gamma p0 div dv): dp = theta dt/opz
!! (R[du,p0] - 2 gamma p0 dZ du) from the rho and T rows' couplings and
!! masses. The gamma part is symmetric negative semi-definite. Poloidal only:
!! one toroidal channel.
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_n, scatter_fft_to_elm_kn, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft, pc_elt_matrix_peel_fft

  !> The peeled-Schur arm of pair_w (mod_petsc_pc_sf_peel) has its operators here
  logical, parameter, public :: pc_peel_supported = .true.
  integer, parameter, public :: PEEL_NOP = 15   !< pc_elt_matrix_peel_fft's operators
  !> element products: leave the axis nodes' DOFs out (diagnostic)
  logical, public :: peel_ebe_axis_mask = .false.
  !> diagnostic weights of W's density and temperature halves (dp = T0 drho +
  !! rho0 dT), from the environment JOREK_W_RHO / JOREK_W_T (default 1)
  real*8, save :: wf_rho = 1.d0, wf_T = 1.d0
  logical, save :: wf_read = .false.

contains

subroutine pc_elt_matrix_force_fft(element, nodes, ELM, ELM_v, gamma_w, vu_full)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)

#define NFV (n_vertex_max*n_degrees)
#define DFV (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(DFV, DFV), intent(out) :: ELM
  !> v_par's W terms (model600 only): zero here, for the common interface
  real*8, dimension(DFV, DFV, 3), intent(out), optional :: ELM_v
  real*8, intent(in), optional :: gamma_w
  logical, intent(in), optional :: vu_full

  real*8, dimension(n_plane, NFV, NFV) :: ELM_p
  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  !> rho0 (1) and T0 (2) and their s, t derivatives
  real*8, dimension(n_plane, 2, n_gauss, n_gauss) :: eq_g, eq_s, eq_t
  real*8 :: f_x(NFV), f_y(NFV)              !< basis gradients at the Gauss point

  integer :: i, j, k, ms, mt, mp, in, q
  integer :: index_ij, index_kl
  real*8  :: wst, xjac, BigR, theta, zeta_t, pref, GAMMA_l, sz
  real*8  :: r0, r0_x, r0_y, T0, T0_x, T0_y, p0, p0_x, p0_y
  real*8  :: f_s, f_t, v_y, u_x, u_y, a_curv, a_uu
  integer, parameter :: vrt(2) = [var_rho, var_T]
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ELM   = 0.d0
  if (present(ELM_v)) ELM_v = 0.d0
  ELM_p = 0.d0
  if (.not. wf_read) then
    block
      character(len=32) :: ev
      integer :: ln, st
      call get_environment_variable("JOREK_W_RHO", ev, ln, st)
      if (st == 0 .and. ln > 0) read(ev, *) wf_rho
      call get_environment_variable("JOREK_W_T", ev, ln, st)
      if (st == 0 .and. ln > 0) read(ev, *) wf_T
    end block
    wf_read = .true.
  endif

  ! the model's own element routine overrides GAMMA locally
  GAMMA_l = 5.d0 / 3.d0
  theta  = time_evol_theta
  zeta_t = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  pref   = (theta * tstep)**2 / (1.d0 + zeta_t)

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_s = 0.d0; y_t = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0
  do i = 1, n_vertex_max
    do j = 1, n_degrees
      sz = element%size(i,j)
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * sz * H(i,j,ms,mt)
          x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * sz * H_s(i,j,ms,mt)
          x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * sz * H_t(i,j,ms,mt)
          y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * sz * H_s(i,j,ms,mt)
          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * sz * H_t(i,j,ms,mt)
          do mp = 1, n_plane
            do k = 1, 2
              do in = 1, n_tor
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vrt(k)) * sz * H(i,j,ms,mt)   * HZ(in,mp)
                eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vrt(k)) * sz * H_s(i,j,ms,mt) * HZ(in,mp)
                eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vrt(k)) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  do ms = 1, n_gauss
   do mt = 1, n_gauss
    wst  = wgauss(ms)*wgauss(mt)
    xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)
    BigR = x_g(ms,mt)
    do q = 1, NFV
      i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
      f_s = h_s(i,j,ms,mt) * element%size(i,j)
      f_t = h_t(i,j,ms,mt) * element%size(i,j)
      f_x(q) = (  y_t(ms,mt)*f_s - y_s(ms,mt)*f_t ) / xjac
      f_y(q) = ( -x_t(ms,mt)*f_s + x_s(ms,mt)*f_t ) / xjac
    enddo

    do mp = 1, n_plane
      r0   = abs(eq_g(mp,1,ms,mt))
      r0_x = (  y_t(ms,mt)*eq_s(mp,1,ms,mt) - y_s(ms,mt)*eq_t(mp,1,ms,mt) ) / xjac
      r0_y = ( -x_t(ms,mt)*eq_s(mp,1,ms,mt) + x_s(ms,mt)*eq_t(mp,1,ms,mt) ) / xjac
      T0   = abs(eq_g(mp,2,ms,mt))
      T0_x = (  y_t(ms,mt)*eq_s(mp,2,ms,mt) - y_s(ms,mt)*eq_t(mp,2,ms,mt) ) / xjac
      T0_y = ( -x_t(ms,mt)*eq_s(mp,2,ms,mt) + x_s(ms,mt)*eq_t(mp,2,ms,mt) ) / xjac
      p0_x = r0_x*T0 + r0*T0_x
      p0_y = r0_y*T0 + r0*T0_y
      p0   = r0 * T0

      do index_ij = 1, NFV
        v_y = f_y(index_ij)
        do index_kl = 1, NFV
          u_x = f_x(index_kl); u_y = f_y(index_kl)
          ! density half (drho = -[u, rho0] + 2 rho0 dZ u, times T0) and
          ! temperature half (dT = -[u, T0] + 2 (gamma - 1) T0 dZ u, times rho0)
          a_curv = wf_rho * ( - 2.d0 * BigR**2 * v_y * T0 * ( r0_x*u_y - r0_y*u_x ) &
                              - 4.d0 * BigR * p0 * v_y * u_y ) &
                 + wf_T   * ( - 2.d0 * BigR**2 * v_y * r0 * ( T0_x*u_y - T0_y*u_x ) &
                              - 4.d0 * (GAMMA_l - 1.d0) * BigR * p0 * v_y * u_y )
          a_uu = pref * ( a_curv ) * xjac
          ELM_p(mp,index_ij,index_kl) = ELM_p(mp,index_ij,index_kl) + wst * a_uu
        enddo
      enddo
    enddo
   enddo
  enddo

  do i = 1, NFV
    do j = 1, NFV
      if (maxval(abs(ELM_p(1:n_plane,i,j))) == 0.d0) cycle
      in_fft = ELM_p(1:n_plane,i,j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
      call pc_my_fft(in_fft, out_fft, n_plane)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM, DFV)
    enddo
  enddo
  ! the scatter accumulates each (row, col) pair twice (as mod_elt_matrix_fft)
  ELM = 0.5d0 * ELM

end subroutine pc_elt_matrix_force_fft

!> The scalar operators of the peeled Schur preconditioner for pair_w
!! (mod_petsc_pc_sf_peel), all on the u unknown, one element:
!!   ELM(:,:,1) = opz K_rho, K_rho = int R^3 rho0 grad v . grad u   (-B_22 without flow)
!!   ELM(:,:,2) = W_comp = (theta dt)^2/opz int 4 gamma R p0 dZ v dZ u (the gamma half of -W)
!!   ELM(:,:,3) = M_rho  = int R^3 rho0 v u
!!   ELM(:,:,4) = K_nu   = int visco_T (R grad v . grad u + d_phi v d_phi u / R)
!!                (the viscous part of B_24 without theta dt and hyper-viscosity)
!!   ELM(:,:,5) = K_eta  = int eta_T R^3 rho0 grad v . grad u   (v2's R_hat, B_13's eta_T)
!!   ELM(:,:,6) = K_w    = int w grad v . grad u, w = sqrt(eta_num R^3 rho0)
!!                (v2's hyper-resistive sandwich K_w X K_w ~ eta_num R^4 rho0 k^4)
!!   ELM(:,:,7) = B_12's element matrix (psi row, u column; model199's amat_12, amat_12_n)
!!   ELM(:,:,8) = B_33's element matrix (= M_psi, int v zj / R)
!!   ELM(:,:,9)  = K_w^T B_33^e+ K_w             (R_hat's hyper-resistive sandwich)
!!   ELM(:,:,10) = B_12^eT B_33^e+ B_12^e        (the bending term)
!!   ELM(:,:,11) = theta dt sym(R_hat^e M_rho^e+ K_nu), R_hat^e with slot 9
!!   ELM(:,:,12) = int R L_w v L_w u, L_w u = div(w grad u): the sandwich in
!!                 strong form (K_w X K_w with X the 1 / R mass inverse, for
!!                 smooth u), the element stencil without element-boundary terms
!!   ELM(:,:,13) = K_nu in psi weights, int visco_T / (R^4 rho0) (R grad v . grad u
!!                 + d_phi v d_phi u / R): v3's Y^-1 = B_33 + theta dt / opz K_nu^psi
!!                 carries K_nu's symbol divided by M_rho's and multiplied by B_33's
!!   ELM(:,:,14) = K_alpha = int alpha / R grad v . grad u, poloidal (B_31's metric)
!!   ELM(:,:,15) = K_gamma = int gamma / R grad v . grad u: the factorised psi-Schur
!!                 opz (B_33 + K_alpha) B_33^-1 (B_33 + K_gamma) of the (psi, j)
!!                 pair, alpha + gamma = theta dt eta_T / opz, alpha gamma = theta dt
!!                 eta_num R / opz pointwise (complex roots: alpha = gamma = sqrt of
!!                 the product)
!! 9..11 are v2's products formed per element (-sf_peel_ebe): their sums keep
!! the element stencil, and 9, 10 bound the global products from above (the
!! joint convexity of (B, M) -> B^T M^-1 B). "+" is the pseudo-inverse on the
!! element's DOFs without the Dirichlet ones (the X_AS patch, one element).
!! rho0, T0 per plane as in W, so a non-axisymmetric state couples the slots.
subroutine pc_elt_matrix_peel_fft(element, nodes, ELM)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module
  use corr_neg

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  real*8, dimension(DFV, DFV, PEEL_NOP), intent(out) :: ELM

  real*8, dimension(n_plane, NFV, NFV, PEEL_NOP) :: ELM_p
  real*8, dimension(n_plane, NFV, NFV) :: ELM_kn, ELM_n12, ELM_kn13
  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  real*8, dimension(n_gauss, n_gauss) :: x_ss, x_st, x_tt, y_ss, y_st, y_tt
  real*8, dimension(n_plane, 2, n_gauss, n_gauss) :: eq_g
  real*8, dimension(n_plane, n_gauss, n_gauss) :: ps_s, ps_t, rs_s, rs_t
  real*8 :: f_s(NFV), f_t(NFV), f_lap(NFV), lw(NFV)
  real*8 :: xjac_x, xjac_y, f_ss, f_st, f_tt, r0_x, r0_y, wh, wh_x, wh_y
  real*8 :: f_x(NFV), f_y(NFV), f_v(NFV)

  integer :: i, j, k, ms, mt, mp, in, q, op
  real*8  :: wst, xjac, BigR, zeta_t, opz, pref, GAMMA_l, sz
  real*8  :: r0, T0, p0, vis, eta_l, gg, a_k, a_w, a_m, a_v, a_e, a_h, thdt, s_ag, p_ag, d_ag, al, ga
  integer, parameter :: vrt(2) = [var_rho, var_T]
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ELM = 0.d0; ELM_p = 0.d0; ELM_kn = 0.d0; ELM_n12 = 0.d0; ELM_kn13 = 0.d0
  ps_s = 0.d0; ps_t = 0.d0; rs_s = 0.d0; rs_t = 0.d0
  x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0; y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0
  thdt = time_evol_theta * tstep

  GAMMA_l = 5.d0 / 3.d0                ! as the model's element routine
  zeta_t  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  opz     = 1.d0 + zeta_t
  pref    = (time_evol_theta * tstep)**2 / opz

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_s = 0.d0; y_t = 0.d0; eq_g = 0.d0
  do i = 1, n_vertex_max
    do j = 1, n_degrees
      sz = element%size(i,j)
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * sz * H(i,j,ms,mt)
          x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * sz * H_s(i,j,ms,mt)
          x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * sz * H_t(i,j,ms,mt)
          y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * sz * H_s(i,j,ms,mt)
          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * sz * H_t(i,j,ms,mt)
          x_ss(ms,mt) = x_ss(ms,mt) + nodes(i)%x(1,j,1) * sz * H_ss(i,j,ms,mt)
          x_st(ms,mt) = x_st(ms,mt) + nodes(i)%x(1,j,1) * sz * H_st(i,j,ms,mt)
          x_tt(ms,mt) = x_tt(ms,mt) + nodes(i)%x(1,j,1) * sz * H_tt(i,j,ms,mt)
          y_ss(ms,mt) = y_ss(ms,mt) + nodes(i)%x(1,j,2) * sz * H_ss(i,j,ms,mt)
          y_st(ms,mt) = y_st(ms,mt) + nodes(i)%x(1,j,2) * sz * H_st(i,j,ms,mt)
          y_tt(ms,mt) = y_tt(ms,mt) + nodes(i)%x(1,j,2) * sz * H_tt(i,j,ms,mt)
          do mp = 1, n_plane
            do k = 1, 2
              do in = 1, n_tor
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vrt(k)) * sz * H(i,j,ms,mt) * HZ(in,mp)
              enddo
            enddo
            do in = 1, n_tor
              ps_s(mp,ms,mt) = ps_s(mp,ms,mt) + nodes(i)%values(in,j,var_psi) * sz * H_s(i,j,ms,mt) * HZ(in,mp)
              ps_t(mp,ms,mt) = ps_t(mp,ms,mt) + nodes(i)%values(in,j,var_psi) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
              rs_s(mp,ms,mt) = rs_s(mp,ms,mt) + nodes(i)%values(in,j,var_rho) * sz * H_s(i,j,ms,mt) * HZ(in,mp)
              rs_t(mp,ms,mt) = rs_t(mp,ms,mt) + nodes(i)%values(in,j,var_rho) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  do ms = 1, n_gauss
   do mt = 1, n_gauss
    wst  = wgauss(ms)*wgauss(mt)
    xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)
    BigR = x_g(ms,mt)
    do q = 1, NFV
      i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
      f_v(q) = H(i,j,ms,mt) * element%size(i,j)
      f_s(q) = H_s(i,j,ms,mt) * element%size(i,j)
      f_t(q) = H_t(i,j,ms,mt) * element%size(i,j)
    enddo
    ! the basis' poloidal Laplacian u_xx + u_yy (model199's v_xx, v_yy)
    xjac_x = (x_ss(ms,mt)*y_t(ms,mt)**2 - y_ss(ms,mt)*x_t(ms,mt)*y_t(ms,mt) - 2.d0*x_st(ms,mt)*y_s(ms,mt)*y_t(ms,mt) &
           + y_st(ms,mt)*(x_s(ms,mt)*y_t(ms,mt) + x_t(ms,mt)*y_s(ms,mt)) &
           + x_tt(ms,mt)*y_s(ms,mt)**2 - y_tt(ms,mt)*x_s(ms,mt)*y_s(ms,mt)) / xjac
    xjac_y = (y_tt(ms,mt)*x_s(ms,mt)**2 - x_tt(ms,mt)*y_s(ms,mt)*x_s(ms,mt) - 2.d0*y_st(ms,mt)*x_t(ms,mt)*x_s(ms,mt) &
           + x_st(ms,mt)*(y_t(ms,mt)*x_s(ms,mt) + y_s(ms,mt)*x_t(ms,mt)) &
           + y_ss(ms,mt)*x_t(ms,mt)**2 - x_ss(ms,mt)*y_t(ms,mt)*x_t(ms,mt)) / xjac
    do q = 1, NFV
      i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
      f_ss = H_ss(i,j,ms,mt) * element%size(i,j)
      f_st = H_st(i,j,ms,mt) * element%size(i,j)
      f_tt = H_tt(i,j,ms,mt) * element%size(i,j)
      f_lap(q) = (f_ss * y_t(ms,mt)**2 - 2.d0*f_st * y_s(ms,mt)*y_t(ms,mt) + f_tt * y_s(ms,mt)**2 &
               + f_s(q) * (y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt)) &
               + f_t(q) * (y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt))) / xjac**2 &
               - xjac_x * (f_s(q) * y_t(ms,mt) - f_t(q) * y_s(ms,mt)) / xjac**2 &
               + (f_ss * x_t(ms,mt)**2 - 2.d0*f_st * x_s(ms,mt)*x_t(ms,mt) + f_tt * x_s(ms,mt)**2 &
               + f_s(q) * (x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt)) &
               + f_t(q) * (x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt))) / xjac**2 &
               - xjac_y * (- f_s(q) * x_t(ms,mt) + f_t(q) * x_s(ms,mt)) / xjac**2
      f_x(q) = (  y_t(ms,mt)*h_s(i,j,ms,mt) - y_s(ms,mt)*h_t(i,j,ms,mt) ) / xjac * element%size(i,j)
      f_y(q) = ( -x_t(ms,mt)*h_s(i,j,ms,mt) + x_s(ms,mt)*h_t(i,j,ms,mt) ) / xjac * element%size(i,j)
    enddo
    do mp = 1, n_plane
      r0 = abs(eq_g(mp,1,ms,mt))
      T0 = abs(eq_g(mp,2,ms,mt))
      p0 = r0 * T0
      if (visco_T_dependent) then
        vis = visco * (corr_neg_temp(T0) / T_0)**(-1.5d0)
      else
        vis = visco
      endif
      ! the sandwich's L_w = div(w grad .), w = sqrt(eta_num R^3 rho0)
      r0_x = (  y_t(ms,mt)*rs_s(mp,ms,mt) - y_s(ms,mt)*rs_t(mp,ms,mt) ) / xjac
      r0_y = ( -x_t(ms,mt)*rs_s(mp,ms,mt) + x_s(ms,mt)*rs_t(mp,ms,mt) ) / xjac
      wh = sqrt(eta_num * BigR**3 * max(r0, 1.d-12))
      wh_x = wh * (1.5d0 / BigR + 0.5d0 * r0_x / max(r0, 1.d-12))
      wh_y = wh * 0.5d0 * r0_y / max(r0, 1.d-12)
      lw = wh * f_lap + wh_x * f_x + wh_y * f_y
      ! eta_T as model199's element matrix forms it
      if (eta_T_dependent .and. corr_neg_temp(T0) <= T_max_eta) then
        eta_l = eta * (corr_neg_temp(T0) / T_0)**(-1.5d0)
      else if (eta_T_dependent) then
        eta_l = eta * (T_max_eta / T_0)**(-1.5d0)
      else
        eta_l = eta
      endif
      ! the factorised psi-Schur's pointwise roots
      s_ag = thdt * eta_l / opz
      p_ag = thdt * eta_num * BigR / opz
      d_ag = s_ag**2 - 4.d0 * p_ag
      if (d_ag >= 0.d0) then
        al = 0.5d0 * (s_ag - sqrt(d_ag)); ga = 0.5d0 * (s_ag + sqrt(d_ag))
      else
        al = sqrt(p_ag); ga = al
      endif
      do i = 1, NFV
        do j = 1, NFV
          gg  = f_x(i)*f_x(j) + f_y(i)*f_y(j)
          a_k = opz * BigR**3 * r0 * gg
          a_w = pref * 4.d0 * GAMMA_l * BigR * p0 * f_y(i) * f_y(j)
          a_m = BigR**3 * r0 * f_v(i) * f_v(j)
          a_v = vis * BigR * gg
          a_e = eta_l * BigR**3 * r0 * gg
          a_h = sqrt(eta_num * BigR**3 * r0) * gg
          ELM_p(mp,i,j,1) = ELM_p(mp,i,j,1) + wst * a_k * xjac
          ELM_p(mp,i,j,2) = ELM_p(mp,i,j,2) + wst * a_w * xjac
          ELM_p(mp,i,j,3) = ELM_p(mp,i,j,3) + wst * a_m * xjac
          ELM_p(mp,i,j,4) = ELM_p(mp,i,j,4) + wst * a_v * xjac
          ELM_p(mp,i,j,5) = ELM_p(mp,i,j,5) + wst * a_e * xjac
          ELM_p(mp,i,j,6) = ELM_p(mp,i,j,6) + wst * a_h * xjac
          ELM_kn(mp,i,j)  = ELM_kn(mp,i,j)  + wst * vis / BigR * f_v(i) * f_v(j) * xjac
          ! B_12 (amat_12, amat_12_n) and B_33 (amat_33), as model199's element matrix
          ELM_p(mp,i,j,7) = ELM_p(mp,i,j,7) - wst * f_v(i) * (ps_s(mp,ms,mt) * f_t(j) - ps_t(mp,ms,mt) * f_s(j)) * thdt
          ELM_n12(mp,i,j) = ELM_n12(mp,i,j) + wst * F0 / BigR * f_v(i) * f_v(j) * xjac * thdt
          ELM_p(mp,i,j,8) = ELM_p(mp,i,j,8) + wst * f_v(i) * f_v(j) / BigR * xjac
          ELM_p(mp,i,j,12) = ELM_p(mp,i,j,12) + wst * BigR * lw(i) * lw(j) * xjac
          ELM_p(mp,i,j,13) = ELM_p(mp,i,j,13) + wst * a_v / (BigR**4 * max(r0, 1.d-12)) * xjac
          ELM_kn13(mp,i,j) = ELM_kn13(mp,i,j) + wst * vis / (BigR**5 * max(r0, 1.d-12)) * f_v(i) * f_v(j) * xjac
          ELM_p(mp,i,j,14) = ELM_p(mp,i,j,14) + wst * al / BigR * gg * xjac
          ELM_p(mp,i,j,15) = ELM_p(mp,i,j,15) + wst * ga / BigR * gg * xjac
        enddo
      enddo
    enddo
   enddo
  enddo

  do op = 1, PEEL_NOP
    if (op >= 9 .and. op <= 11) cycle            ! the element products, below
    do i = 1, NFV
      do j = 1, NFV
        if (maxval(abs(ELM_p(1:n_plane,i,j,op))) == 0.d0) cycle
        in_fft = ELM_p(1:n_plane,i,j,op)
        call fft1(in_fft, out_fft)
        call scatter_fft_to_elm(out_fft, i, j, ELM(:,:,op), DFV)
      enddo
    enddo
  enddo
  do i = 1, NFV
    do j = 1, NFV
      if (maxval(abs(ELM_kn(1:n_plane,i,j))) == 0.d0) cycle
      in_fft = ELM_kn(1:n_plane,i,j)
      call fft1(in_fft, out_fft)
      call scatter_fft_to_elm_kn(out_fft, i, j, ELM(:,:,4), DFV)
      in_fft = ELM_kn13(1:n_plane,i,j)
      call fft1(in_fft, out_fft)
      call scatter_fft_to_elm_kn(out_fft, i, j, ELM(:,:,13), DFV)
      if (maxval(abs(ELM_n12(1:n_plane,i,j))) == 0.d0) cycle
      in_fft = ELM_n12(1:n_plane,i,j)
      call fft1(in_fft, out_fft)
      call scatter_fft_to_elm_n(out_fft, i, j, ELM(:,:,7), DFV)
    enddo
  enddo
  ELM = 0.5d0 * ELM

  call element_products()

contains

  subroutine fft1(a, b)
    real*8, intent(inout)   :: a(1:n_plane)
    complex*16, intent(out) :: b(1:n_plane)
#ifdef USE_FFTW
    call dfftw_execute_dft_r2c(fftw_plan, a, b)
#else
    call pc_my_fft(a, b, n_plane)
#endif
  end subroutine fft1

  !> ELM(:,:,9..11): v2's element-local products from the element's blocks
  subroutine element_products()
    use mod_node_indices, only: calculate_node_indices
    real*8, dimension(DFV, DFV) :: Bi, Mi, R, V
    logical :: keep(DFV)
    integer :: node_ind((n_order+1)/2, (n_order+1)/2), kk, ll, a

    ! the Dirichlet DOFs (as model199's boundary_conditions, all slots)
    call calculate_node_indices(node_ind)
    keep = .true.
    do i = 1, n_vertex_max
      if (peel_ebe_axis_mask .and. nodes(i)%axis_node) &
        keep(n_tor * n_degrees * (i - 1) + 1 : n_tor * n_degrees * i) = .false.
      if (nodes(i)%boundary == 0) cycle
      do kk = 1, (n_order+1)/2
        do ll = 1, (n_order+1)/2
          a = 0
          if ((nodes(i)%boundary == 1 .or. nodes(i)%boundary == 3) .and. ll == 1) a = node_ind(kk, ll)
          if ((nodes(i)%boundary == 2 .or. nodes(i)%boundary == 3) .and. kk == 1) a = node_ind(kk, ll)
          if (a == 0) cycle
          keep(n_tor * ((i - 1) * n_degrees + a - 1) + 1 : n_tor * ((i - 1) * n_degrees + a)) = .false.
        enddo
      enddo
    enddo

    call pinv(ELM(:,:,8), keep, Bi)
    call pinv(ELM(:,:,3), keep, Mi)
    ELM(:,:,9)  = matmul(transpose(ELM(:,:,6)), matmul(Bi, ELM(:,:,6)))
    ELM(:,:,10) = matmul(transpose(ELM(:,:,7)), matmul(Bi, ELM(:,:,7)))
    R = opz * ELM(:,:,3) + thdt * ELM(:,:,5) + thdt * ELM(:,:,9)
    V = matmul(R, matmul(Mi, ELM(:,:,4)))
    ELM(:,:,11) = 0.5d0 * thdt * (V + transpose(V))
  end subroutine element_products

  !> Ai = the pseudo-inverse of sym(A) on the kept DOFs (eigenvalues below
  !! 1e-10 of the largest dropped: the zero-size axis DOFs), 0 elsewhere
  subroutine pinv(A, keep, Ai)
    real*8, intent(in)  :: A(DFV, DFV)
    logical, intent(in) :: keep(DFV)
    real*8, intent(out) :: Ai(DFV, DFV)
    real*8 :: S(DFV, DFV), w(DFV), work(DFV * 34), tol
    integer :: idx(DFV), n, ia, ib, info
    external :: dsyev
    n = 0
    do ia = 1, DFV
      if (keep(ia)) then
        n = n + 1; idx(n) = ia
      endif
    enddo
    Ai = 0.d0
    if (n == 0) return
    do ib = 1, n
      do ia = 1, n
        S(ia, ib) = 0.5d0 * (A(idx(ia), idx(ib)) + A(idx(ib), idx(ia)))
      enddo
    enddo
    call dsyev('V', 'U', n, S, DFV, w, work, size(work), info)
    if (info /= 0) return
    tol = 1.d-10 * maxval(abs(w(1:n)))
    do ia = 1, n
      if (abs(w(ia)) > tol) then
        w(ia) = 1.d0 / w(ia)
      else
        w(ia) = 0.d0
      endif
    enddo
    do ib = 1, n
      do ia = 1, n
        Ai(idx(ia), idx(ib)) = sum(S(ia, 1:n) * w(1:n) * S(ib, 1:n))
      enddo
    enddo
  end subroutine pinv

end subroutine pc_elt_matrix_peel_fft

end module mod_pc_elt_matrix_force_fft
