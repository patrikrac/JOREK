module mod_elt_matrix_commutator
!----------------------------------------------------------------
! Element-level BUILDING-BLOCK operators for the commutator-PC
! candidate framework (note JOREK_commutator_preconditioner_baseline,
! Sec. 8.5). Each block is a pure integrand (NO time/opz/eta factors --
! those live in the candidate coefficient table in the analysis module),
! a 1-var mode-space operator on the shared Bezier basis:
!
!   CM_ADV1 : v (u_s u0_t - u_t u0_s)            E x B advection, amat_11 form
!   CM_ADVR : v R^2 (u_s u0_t - u_t u0_s)        E x B advection, amat_55 form
!   CM_COMP : v 2R u u0_y xjac                   compression, amat_55 form
!   CM_S1R  : (grad v . grad u) / R xjac         1/R poloidal stiffness
!   CM_SR   : (grad v . grad u) R xjac           R   poloidal stiffness
!
! Candidates are linear combinations of these blocks together with the
! extracted A_full masses (B33 = 1/R, B44 = R) and B11 = A_pp; e.g.
!   M1  = opz*B33 - (theta dt)*ADV1
!   M3  = opz*B44 - (theta dt)*ADVR - (theta dt)*COMP
!   M2  = M1 + eta*(theta dt)*S1R
! so a NEW candidate is just a new coefficient row -- no re-assembly.
!
! All blocks are p-channel only (advection/compression pick up toroidal
! mode coupling through the background flow u0 across planes, via the FFT
! of the plane-wise integrand -- exactly as amat_55's p-channel). Geometry
! and FFT machinery use first derivatives only; 0.5 = FFT normalisation,
! matching element_matrix_fft.
!----------------------------------------------------------------
implicit none
public :: element_matrix_commutator
public :: CM_NB, CM_ADV1, CM_ADVR, CM_COMP, CM_S1R, CM_SR

integer, parameter :: CM_NB   = 5   !< number of assembled building blocks
integer, parameter :: CM_ADV1 = 1
integer, parameter :: CM_ADVR = 2
integer, parameter :: CM_COMP = 3
integer, parameter :: CM_S1R  = 4
integer, parameter :: CM_SR   = 5

type :: type_fct_vals_cm
  real*8 :: v, v_x, v_y, v_s, v_t
end type type_fct_vals_cm

contains

subroutine element_matrix_commutator(element, nodes, ELM_blk)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module, only: fftw_plan
  use mod_elt_matrix_elliptic, only: scatter_fft_to_elm

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)

#define N1V  (n_vertex_max*n_degrees)
#define D1V  (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(D1V, D1V, CM_NB), intent(out) :: ELM_blk

  ! Plane-workspace accumulators (p channel), one per block
  real*8, dimension(n_plane, N1V, N1V, CM_NB) :: ELM_p

  ! Geometry at Gauss points (first derivatives only)
  real*8, dimension(n_gauss,n_gauss) :: x_g, x_s, x_t
  real*8, dimension(n_gauss,n_gauss) :: y_g, y_s, y_t

  ! Background flow potential u0: s,t derivatives at Gauss points/planes
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_s, eq_t

  integer :: i, j, k, l, ms, mt, mp, in, ib
  integer :: idx_ij, idx_kl

  real*8 :: wst, xjac, BigR
  real*8 :: u0_s, u0_t, u0_y, brk
  type(type_fct_vals_cm) :: v_fct, u_fct

  ! FFT workspace
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ELM_blk = 0.d0
  ELM_p   = 0.d0

  !-----------------------------------------------------------------
  ! Geometry and background (u0) s,t derivatives at Gauss points
  !-----------------------------------------------------------------
  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0
  y_g = 0.d0; y_s = 0.d0; y_t = 0.d0
  eq_s = 0.d0; eq_t = 0.d0

  do i = 1, n_vertex_max
    do j = 1, n_degrees
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H(i,j,ms,mt)
          x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_s(i,j,ms,mt)
          x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_t(i,j,ms,mt)
          y_g(ms,mt) = y_g(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H(i,j,ms,mt)
          y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_s(i,j,ms,mt)
          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_t(i,j,ms,mt)
          do mp = 1, n_plane
            do k = 1, n_var
              do in = 1, n_tor
                eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt) * HZ(in,mp)
                eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt) * HZ(in,mp)
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
      BigR = x_g(ms,mt)

      do mp = 1, n_plane

        u0_s = eq_s(mp,var_u,ms,mt)
        u0_t = eq_t(mp,var_u,ms,mt)
        u0_y = ( - x_t(ms,mt) * u0_s + x_s(ms,mt) * u0_t ) / xjac

        do i = 1, n_vertex_max
          do j = 1, n_degrees

            idx_ij = n_degrees*(i-1) + (j-1) + 1

            v_fct%v   =   H(i,j,ms,mt)   * element%size(i,j)
            v_fct%v_s =   H_s(i,j,ms,mt) * element%size(i,j)
            v_fct%v_t =   H_t(i,j,ms,mt) * element%size(i,j)
            v_fct%v_x = (   y_t(ms,mt) * H_s(i,j,ms,mt) - y_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac
            v_fct%v_y = ( - x_t(ms,mt) * H_s(i,j,ms,mt) + x_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac

            do k = 1, n_vertex_max
              do l = 1, n_degrees

                idx_kl = n_degrees*(k-1) + (l-1) + 1

                u_fct%v   =   H(k,l,ms,mt)   * element%size(k,l)
                u_fct%v_s =   H_s(k,l,ms,mt) * element%size(k,l)
                u_fct%v_t =   H_t(k,l,ms,mt) * element%size(k,l)
                u_fct%v_x = (   y_t(ms,mt) * H_s(k,l,ms,mt) - y_s(ms,mt) * H_t(k,l,ms,mt) ) * element%size(k,l) / xjac
                u_fct%v_y = ( - x_t(ms,mt) * H_s(k,l,ms,mt) + x_s(ms,mt) * H_t(k,l,ms,mt) ) * element%size(k,l) / xjac

                brk = u_fct%v_s * u0_t - u_fct%v_t * u0_s          ! [u, u0] * xjac (computational bracket)

                ELM_p(mp,idx_ij,idx_kl,CM_ADV1) = ELM_p(mp,idx_ij,idx_kl,CM_ADV1) &
                  + wst * ( v_fct%v * brk )
                ELM_p(mp,idx_ij,idx_kl,CM_ADVR) = ELM_p(mp,idx_ij,idx_kl,CM_ADVR) &
                  + wst * ( v_fct%v * BigR**2 * brk )
                ELM_p(mp,idx_ij,idx_kl,CM_COMP) = ELM_p(mp,idx_ij,idx_kl,CM_COMP) &
                  + wst * ( v_fct%v * 2.d0 * BigR * u_fct%v * u0_y * xjac )
                ELM_p(mp,idx_ij,idx_kl,CM_S1R) = ELM_p(mp,idx_ij,idx_kl,CM_S1R) &
                  + wst * ( (v_fct%v_x*u_fct%v_x + v_fct%v_y*u_fct%v_y) / BigR * xjac )
                ELM_p(mp,idx_ij,idx_kl,CM_SR) = ELM_p(mp,idx_ij,idx_kl,CM_SR) &
                  + wst * ( (v_fct%v_x*u_fct%v_x + v_fct%v_y*u_fct%v_y) * BigR * xjac )

              enddo  ! l
            enddo    ! k
          enddo  ! j
        enddo    ! i
      enddo  ! mp
    enddo  ! mt
  enddo    ! ms

  !-----------------------------------------------------------------
  ! FFT reconstruction (p channel only); 0.5 = FFT normalisation
  !-----------------------------------------------------------------
  do ib = 1, CM_NB
    do i = 1, N1V
      do j = 1, N1V
        in_fft = ELM_p(1:n_plane, i, j, ib)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm(out_fft, i, j, ELM_blk(:,:,ib), D1V)
      enddo
    enddo
    ELM_blk(:,:,ib) = 0.5d0 * ELM_blk(:,:,ib)
  enddo

end subroutine element_matrix_commutator

end module mod_elt_matrix_commutator
