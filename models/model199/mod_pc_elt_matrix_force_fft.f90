!> The force operator W of the SF preconditioner's pair_w, the matrix ADDED
!! to B_22: the curvature part of the composed momentum Schur complement
!! (Chacon JCP 526 (2025) Eq. (19); the bending and kink parts go through
!! pair_w's explicit (psi, j) rows, mod_petsc_pc_sf). With
!! [a,b] = a_R b_Z - a_Z b_R and p0 = rho0 T0,
!!
!!   W(du,v) = (theta dt)^2/opz int ( - 2 R^2 dZ v [p0,du] - 4 gamma R p0 dZ v dZ du ) dR dZ
!!
!! both halves of -grad(dv.grad p0 + gamma p0 div dv): dp = theta dt/opz
!! (R[du,p0] - 2 gamma p0 dZ du) from the rho and T rows' couplings and
!! masses. The gamma part is symmetric negative semi-definite. Poloidal only:
!! one toroidal channel.
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft, pc_elt_matrix_mass_rho, pc_elt_matrix_eta

contains

subroutine pc_elt_matrix_force_fft(element, nodes, ELM, ELM_v)

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

!> The inertia of the vorticity-space Schur PC (mod_petsc_pc_vs), one
!! variable: M_rho(v, w) = opz int rho0 R^3 v w dR dZ, with rho0 the full
!! density (all harmonics, through the FFT scatter). With B_44 = (v, w R) and
!! B_42 = (grad v, R grad u), the inertia part of B_22,
!! -opz (grad v, rho0 R^3 grad u), is -M_rho B_44^-1 B_42 up to the
!! gradients of rho0 R^2.
subroutine pc_elt_matrix_mass_rho(element, nodes, ELM)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  real*8, dimension(DFV, DFV), intent(out) :: ELM

  real*8, dimension(n_plane, NFV, NFV) :: ELM_p
  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  real*8, dimension(n_plane, n_gauss, n_gauss) :: r_g     !< rho0 on the planes
  real*8 :: f_v(NFV)                                       !< basis values at the Gauss point
  integer :: i, j, ms, mt, mp, in, q
  real*8  :: wst, xjac, wr, zeta_t, sz
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ELM   = 0.d0
  ELM_p = 0.d0
  zeta_t = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_s = 0.d0; y_t = 0.d0; r_g = 0.d0
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
            do in = 1, n_tor
              r_g(mp,ms,mt) = r_g(mp,ms,mt) + nodes(i)%values(in,j,var_rho) * sz * H(i,j,ms,mt) * HZ(in,mp)
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
      do q = 1, NFV
        f_v(q) = H((q - 1) / n_degrees + 1, mod(q - 1, n_degrees) + 1, ms, mt) &
                 * element%size((q - 1) / n_degrees + 1, mod(q - 1, n_degrees) + 1)
      enddo
      do mp = 1, n_plane
        ! the model's inertia: r0_hat BigR = rho0 R^3, rho0 = |rho| as in its element matrix
        wr = wst * abs(r_g(mp,ms,mt)) * x_g(ms,mt)**3 * xjac
        do i = 1, NFV
          do j = 1, NFV
            ELM_p(mp,i,j) = ELM_p(mp,i,j) + wr * f_v(i) * f_v(j)
          enddo
        enddo
      enddo
    enddo
  enddo

  do i = 1, NFV
    do j = 1, NFV
      in_fft = ELM_p(1:n_plane,i,j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
      call pc_my_fft(in_fft, out_fft, n_plane)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM, DFV)
    enddo
  enddo
  ELM = 0.5d0 * (1.d0 + zeta_t) * ELM

end subroutine pc_elt_matrix_mass_rho

!> The resistive line bending of vs_lumped (mod_petsc_pc_vs, -vs_bending
!! resistive), one variable each, with eta_T(T0) the coefficient of amat_13
!! (model_eta_T) on the toroidal planes, T0 = |T| with all harmonics (through
!! the FFT scatter, as M_rho):
!!   ELM_m = M_eta(v, w) = theta dt int eta_T v w / R dR dZ          the eta part of -B_13
!!   ELM_k = K_eta(v, w) = theta dt int eta_T grad v . grad w / R    its self-adjoint form
!!   ELM_h = K_1(v, w)   = theta dt eta_num int grad v . grad w      the hyper-resistive part
!!                                                                    of -B_13 (optional, the
!!                                                                    preconditioner's self-check)
!! The Gauss-point quantities are formed as in mod_elt_matrix_fft, so that
!! M_eta + K_1 is -B_13 to rounding.
subroutine pc_elt_matrix_eta(element, nodes, ELM_m, ELM_k, ELM_h)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module
  use mod_model_resistivity, only: model_eta_T

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  real*8, dimension(DFV, DFV), intent(out) :: ELM_m, ELM_k
  real*8, dimension(DFV, DFV), intent(out), optional :: ELM_h

  real*8, dimension(n_plane, NFV, NFV) :: Em_p, Ek_p, Eh_p
  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  real*8, dimension(n_plane, n_gauss, n_gauss) :: T_g       !< T on the planes
  real*8 :: f_v(NFV), f_x(NFV), f_y(NFV)                    !< basis values and gradients at the Gauss point
  integer :: i, j, ms, mt, mp, in, q
  real*8  :: wst, xjac, BigR, theta, T0, eta_T, deta_dT, d2eta_d2T, sz, wm, wk, wh, gg
  logical :: do_h
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  do_h = present(ELM_h)
  ELM_m = 0.d0; ELM_k = 0.d0
  Em_p = 0.d0; Ek_p = 0.d0
  if (do_h) then
    ELM_h = 0.d0; Eh_p = 0.d0
  endif
  theta = time_evol_theta

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_s = 0.d0; y_t = 0.d0; T_g = 0.d0
  do i = 1, n_vertex_max
    do j = 1, n_degrees
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H(i,j,ms,mt)
          x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_s(i,j,ms,mt)
          x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_t(i,j,ms,mt)
          y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_s(i,j,ms,mt)
          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_t(i,j,ms,mt)
          do mp = 1, n_plane
            do in = 1, n_tor
              T_g(mp,ms,mt) = T_g(mp,ms,mt) + nodes(i)%values(in,j,var_T) * element%size(i,j) * H(i,j,ms,mt) &
                              * HZ(in,mp)
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
      ! the basis functions as the model forms v, v_x, v_y
      do q = 1, NFV
        i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
        sz = element%size(i,j)
        f_v(q) = H(i,j,ms,mt) * sz
        f_x(q) = (  y_t(ms,mt) * h_s(i,j,ms,mt) - y_s(ms,mt) * h_t(i,j,ms,mt) ) * sz / xjac
        f_y(q) = (- x_t(ms,mt) * h_s(i,j,ms,mt) + x_s(ms,mt) * h_t(i,j,ms,mt) ) * sz / xjac
      enddo
      do mp = 1, n_plane
        T0 = abs(T_g(mp,ms,mt))
        call model_eta_T(T0, eta_T, deta_dT, d2eta_d2T)
        do i = 1, NFV
          do j = 1, NFV
            gg = f_x(i) * f_x(j) + f_y(i) * f_y(j)
            ! amat_13 = - eta_num (v_x zj_x + v_y zj_y) xjac theta dt - eta_T v zj / R xjac theta dt
            wm = eta_T * f_v(i) * f_v(j) / BigR * xjac * theta * tstep
            wk = eta_T * gg / BigR * xjac * theta * tstep
            Em_p(mp,i,j) = Em_p(mp,i,j) + wst * wm
            Ek_p(mp,i,j) = Ek_p(mp,i,j) + wst * wk
            if (do_h) then
              wh = eta_num * gg * xjac * theta * tstep
              Eh_p(mp,i,j) = Eh_p(mp,i,j) + wst * wh
            endif
          enddo
        enddo
      enddo
    enddo
  enddo

  do i = 1, NFV
    do j = 1, NFV
      in_fft = Em_p(1:n_plane,i,j)
      call fft_plane(in_fft, out_fft)
      call scatter_fft_to_elm(out_fft, i, j, ELM_m, DFV)
      in_fft = Ek_p(1:n_plane,i,j)
      call fft_plane(in_fft, out_fft)
      call scatter_fft_to_elm(out_fft, i, j, ELM_k, DFV)
      if (do_h) then
        in_fft = Eh_p(1:n_plane,i,j)
        call fft_plane(in_fft, out_fft)
        call scatter_fft_to_elm(out_fft, i, j, ELM_h, DFV)
      endif
    enddo
  enddo
  ! the scatter accumulates each (row, col) pair twice (as mod_elt_matrix_fft)
  ELM_m = 0.5d0 * ELM_m
  ELM_k = 0.5d0 * ELM_k
  if (do_h) ELM_h = 0.5d0 * ELM_h

contains

  subroutine fft_plane(a, b)
    real*8, intent(inout)   :: a(1:n_plane)
    complex*16, intent(out) :: b(1:n_plane)
#ifdef USE_FFTW
    call dfftw_execute_dft_r2c(fftw_plan, a, b)
#else
    call pc_my_fft(a, b, n_plane)
#endif
  end subroutine fft_plane

end subroutine pc_elt_matrix_eta

end module mod_pc_elt_matrix_force_fft
