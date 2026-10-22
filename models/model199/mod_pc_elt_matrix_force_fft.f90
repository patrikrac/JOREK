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

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_kn, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft, pc_elt_matrix_peel_fft

  !> The peeled-Schur arm of pair_w (mod_petsc_pc_sf_peel) has its operators here
  logical, parameter, public :: pc_peel_supported = .true.
  integer, parameter, public :: PEEL_NOP = 6    !< pc_elt_matrix_peel_fft's operators

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
          a_curv = - 2.d0 * BigR**2 * v_y * ( p0_x*u_y - p0_y*u_x ) &
                   - 4.d0 * GAMMA_l * BigR * p0 * v_y * u_y
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
  real*8, dimension(n_plane, NFV, NFV) :: ELM_kn
  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  real*8, dimension(n_plane, 2, n_gauss, n_gauss) :: eq_g
  real*8 :: f_x(NFV), f_y(NFV), f_v(NFV)

  integer :: i, j, k, ms, mt, mp, in, q, op
  real*8  :: wst, xjac, BigR, zeta_t, opz, pref, GAMMA_l, sz
  real*8  :: r0, T0, p0, vis, eta_l, gg, a_k, a_w, a_m, a_v, a_e, a_h
  integer, parameter :: vrt(2) = [var_rho, var_T]
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ELM = 0.d0; ELM_p = 0.d0; ELM_kn = 0.d0

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
          do mp = 1, n_plane
            do k = 1, 2
              do in = 1, n_tor
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vrt(k)) * sz * H(i,j,ms,mt) * HZ(in,mp)
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
      f_v(q) = H(i,j,ms,mt) * element%size(i,j)
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
      ! eta_T as model199's element matrix forms it
      if (eta_T_dependent .and. corr_neg_temp(T0) <= T_max_eta) then
        eta_l = eta * (corr_neg_temp(T0) / T_0)**(-1.5d0)
      else if (eta_T_dependent) then
        eta_l = eta * (T_max_eta / T_0)**(-1.5d0)
      else
        eta_l = eta
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
        enddo
      enddo
    enddo
   enddo
  enddo

  do op = 1, PEEL_NOP
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
    enddo
  enddo
  ELM = 0.5d0 * ELM

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

end subroutine pc_elt_matrix_peel_fft

end module mod_pc_elt_matrix_force_fft
