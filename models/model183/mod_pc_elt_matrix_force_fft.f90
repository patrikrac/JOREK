!> model183's force operator W (stellarator reduced MHD, Nikulsin et al.:
!! B = grad chi + grad Psi x grad chi, geometry R(s,t,phi), Z(s,t,phi)), the
!! matrix ADDED to B_22; see models/model199/mod_pc_elt_matrix_force_fft.f90.
!! With Bv2 = |grad chi|^2, [a,b] = grad chi . (grad a x grad b) and
!! pref = (theta dt)^2/(1+zeta), the curvature part of the composed Schur
!! complement (bending and kink go through pair_w's explicit (psi, j)):
!!
!!   W(du,v) = - pref int D(du) [1/Bv2, v] dV,  D(du) = ( [p0,du] - gamma p0 [Bv2,du]/Bv2 ) / Bv2
!!
!! (the tokamak limit, Bv2 ~ 1/R^2, gives model199's curvature term). Each
!! side is split by the phi-derivative order a = 0, 1 on its toroidal basis
!! function (HZ, HZ_p), scattered through the p / n / k / kn channels.
!! Physical derivatives on the phi-dependent grid: f_phi = f_p - x_p f_R - y_p f_Z.
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_n, &
                                scatter_fft_to_elm_k, scatter_fft_to_elm_kn, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft, pc_elt_matrix_peel_fft

  !> The peeled-Schur arm of pair_w (mod_petsc_pc_sf_peel) is model199's only
  logical, parameter, public :: pc_peel_supported = .false.
  integer, parameter, public :: PEEL_NOP = 6

contains

!> [a,b] = grad chi . (grad a x grad b), from (R, Z, phi) gradients
pure real*8 function vbr(a, b, chR, chZ, chP, R)
  real*8, intent(in) :: a(3), b(3), chR, chZ, chP, R
  vbr = ( (a(2)*b(3) - a(3)*b(2))*chR + (a(3)*b(1) - a(1)*b(3))*chZ      &
        + (a(1)*b(2) - a(2)*b(1))*chP ) / R
end function vbr

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

  !> channels (a, b): phi-derivative order on test, trial
  real*8, allocatable :: ELM_c(:,:,:,:,:)
  real*8, allocatable :: ELM_tmp(:,:)

  real*8, dimension(n_plane,n_gauss,n_gauss) :: x_g, x_s, x_t, x_p, y_s, y_t, y_p
  !> rho0 (1) and T0 (2): value, s, t and fixed-(s,t) phi derivatives
  real*8, dimension(n_plane,2,n_gauss,n_gauss) :: eq_g, eq_s, eq_t, eq_p
  integer, parameter :: vrt(2) = [var_rho, var_T]

  integer :: i, j, k, ms, mt, mp, in, a, b, q
  integer :: index_ij, index_kl
  real*8  :: wst, BigR, xjac, theta, zeta_t, pref, GAMMA_l, wvol, sz, xp, yp
  real*8  :: dr0(3), dT0(3), r0, T0, p0, gp0(3)
  real*8  :: ch(0:2,0:2,0:2), chR, chZ, chP, Bv2, gBv2(3)
  real*8  :: dv(3,0:1)                                  !< (R, Z, phi) gradient per channel
  real*8  :: fc(0:1,NFV), fd(0:1,NFV)                   !< [1/Bv2, f], D(f)
  real*8  :: val

  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  allocate(ELM_c(n_plane, NFV, NFV, 0:1, 0:1))
  ELM   = 0.d0
  if (present(ELM_v)) ELM_v = 0.d0
  ELM_c = 0.d0

  ! the model's own element routine overrides GAMMA locally
  GAMMA_l = 5.d0 / 3.d0
  theta  = time_evol_theta
  zeta_t = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  pref   = (theta * tstep)**2 / (1.d0 + zeta_t)

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_p = 0.d0
  y_s = 0.d0; y_t = 0.d0; y_p = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0; eq_p = 0.d0
  do i = 1, n_vertex_max
    do j = 1, n_degrees
      sz = element%size(i,j)
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          do mp = 1, n_plane
            do in = 1, n_coord_tor
              x_g(mp,ms,mt) = x_g(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H  (i,j,ms,mt)*HZ_coord  (in,mp)
              x_s(mp,ms,mt) = x_s(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H_s(i,j,ms,mt)*HZ_coord  (in,mp)
              x_t(mp,ms,mt) = x_t(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H_t(i,j,ms,mt)*HZ_coord  (in,mp)
              x_p(mp,ms,mt) = x_p(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H  (i,j,ms,mt)*HZ_coord_p(in,mp)
              y_s(mp,ms,mt) = y_s(mp,ms,mt) + nodes(i)%x(in,j,2)*sz*H_s(i,j,ms,mt)*HZ_coord  (in,mp)
              y_t(mp,ms,mt) = y_t(mp,ms,mt) + nodes(i)%x(in,j,2)*sz*H_t(i,j,ms,mt)*HZ_coord  (in,mp)
              y_p(mp,ms,mt) = y_p(mp,ms,mt) + nodes(i)%x(in,j,2)*sz*H  (i,j,ms,mt)*HZ_coord_p(in,mp)
            enddo
            do k = 1, 2
              do in = 1, n_tor
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) + nodes(i)%values(in,j,vrt(k))*sz*H  (i,j,ms,mt)*HZ  (in,mp)
                eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) + nodes(i)%values(in,j,vrt(k))*sz*H_s(i,j,ms,mt)*HZ  (in,mp)
                eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) + nodes(i)%values(in,j,vrt(k))*sz*H_t(i,j,ms,mt)*HZ  (in,mp)
                eq_p(mp,k,ms,mt) = eq_p(mp,k,ms,mt) + nodes(i)%values(in,j,vrt(k))*sz*H  (i,j,ms,mt)*HZ_p(in,mp)
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  do ms = 1, n_gauss
   do mt = 1, n_gauss
    wst = wgauss(ms)*wgauss(mt)

    do mp = 1, n_plane
      xjac = x_s(mp,ms,mt)*y_t(mp,ms,mt) - x_t(mp,ms,mt)*y_s(mp,ms,mt)
      xp   = x_p(mp,ms,mt); yp = y_p(mp,ms,mt)
      BigR = x_g(mp,ms,mt)
      wvol = wst * BigR * xjac                       ! dV = R dR dZ dphi

      dr0 = grad(eq_s(mp,1,ms,mt), eq_t(mp,1,ms,mt), eq_p(mp,1,ms,mt))
      dT0 = grad(eq_s(mp,2,ms,mt), eq_t(mp,2,ms,mt), eq_p(mp,2,ms,mt))
      r0  = eq_g(mp,1,ms,mt)
      T0  = eq_g(mp,2,ms,mt)
      p0  = r0*T0
      gp0 = dr0*T0 + r0*dT0

      !--- vacuum field: chi(mp,ms,mt,i,j,k) = d_R^i d_Z^j d_phi^k chi
      ch  = element%chi(mp,ms,mt,0:2,0:2,0:2)
      chR = ch(1,0,0); chZ = ch(0,1,0); chP = ch(0,0,1)
      Bv2 = chR**2 + chZ**2 + chP**2/BigR**2
      gBv2(1) = 2.d0*(chR*ch(2,0,0) + chZ*ch(1,1,0) + chP*ch(1,0,1)/BigR**2) - 2.d0*chP**2/BigR**3
      gBv2(2) = 2.d0*(chR*ch(1,1,0) + chZ*ch(0,2,0) + chP*ch(0,1,1)/BigR**2)
      gBv2(3) = 2.d0*(chR*ch(1,0,1) + chZ*ch(0,1,1) + chP*ch(0,0,2)/BigR**2)

      ! per basis function (test and trial alike), channel 0: HZ, 1: HZ_p
      do q = 1, NFV
        i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
        dv(:,0) = grad(h_s(i,j,ms,mt)*element%size(i,j), h_t(i,j,ms,mt)*element%size(i,j), 0.d0)
        dv(:,1) = grad(0.d0, 0.d0, h(i,j,ms,mt)*element%size(i,j))
        do a = 0, 1
          fc(a,q) = - vbr(gBv2, dv(:,a), chR, chZ, chP, BigR)/Bv2**2
          fd(a,q) = ( vbr(gp0, dv(:,a), chR, chZ, chP, BigR)             &
                      - GAMMA_l*p0*vbr(gBv2, dv(:,a), chR, chZ, chP, BigR)/Bv2 ) / Bv2
        enddo
      enddo

      do index_ij = 1, NFV
        do index_kl = 1, NFV
          do a = 0, 1
            do b = 0, 1
              val = pref * ( - fd(b,index_kl)*fc(a,index_ij) )
              ELM_c(mp,index_ij,index_kl,a,b) = ELM_c(mp,index_ij,index_kl,a,b) + wvol*val
            enddo
          enddo
        enddo
      enddo

    enddo  ! mp
   enddo  ! mt
  enddo   ! ms

  allocate(ELM_tmp(DFV, DFV))
  do a = 0, 1
    do b = 0, 1
      ELM_tmp = 0.d0
      do i = 1, NFV
        do j = 1, NFV
          if (maxval(abs(ELM_c(1:n_plane,i,j,a,b))) == 0.d0) cycle
          in_fft = ELM_c(1:n_plane,i,j,a,b)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
          call pc_my_fft(in_fft, out_fft, n_plane)
#endif
          select case (2*a + b)
          case (0); call scatter_fft_to_elm   (out_fft, i, j, ELM_tmp, DFV)
          case (1); call scatter_fft_to_elm_n (out_fft, i, j, ELM_tmp, DFV)
          case (2); call scatter_fft_to_elm_k (out_fft, i, j, ELM_tmp, DFV)
          case (3); call scatter_fft_to_elm_kn(out_fft, i, j, ELM_tmp, DFV)
          end select
        enddo
      enddo
      ELM = ELM + ELM_tmp
    enddo
  enddo
  deallocate(ELM_tmp, ELM_c)
  ! the scatter accumulates each (row, col) pair twice (as mod_elt_matrix_fft)
  ELM = 0.5d0 * ELM

contains

  !> (R, Z, phi) gradient from the (s, t) derivatives and the fixed-(s,t)
  !! phi derivative f_p, on the current plane
  function grad(f_s, f_t, f_p) result(d)
    real*8, intent(in) :: f_s, f_t, f_p
    real*8 :: d(3)
    d(1) = (  y_t(mp,ms,mt)*f_s - y_s(mp,ms,mt)*f_t ) / xjac
    d(2) = ( -x_t(mp,ms,mt)*f_s + x_s(mp,ms,mt)*f_t ) / xjac
    d(3) = f_p - xp*d(1) - yp*d(2)
  end function grad

end subroutine pc_elt_matrix_force_fft

!> No peeled-Schur operators in this model (mod_petsc_pc_sf_peel refuses it)
subroutine pc_elt_matrix_peel_fft(element, nodes, ELM)
  use mod_parameters
  use data_structure, only: type_element, type_node
  implicit none
  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  real*8, intent(out) :: ELM(:,:,:)
  ELM = 0.d0
end subroutine pc_elt_matrix_peel_fft

end module mod_pc_elt_matrix_force_fft
