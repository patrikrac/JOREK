module mod_elt_matrix_commutator
!----------------------------------------------------------------
! Element-level operators for the commutator-preconditioner analysis
! (note: JOREK_commutator_preconditioner_baseline, Table 1 / Sec. 8.5).
!
! Assembles TWO 1-var mode-space element matrices, clones of model199
! diagonal kernels with the field variable replaced by the trial u:
!
!   ELM_S1r : 1/R-weighted poloidal stiffness  (v_x u_x + v_y u_y)/R
!             -- the candidate-M2 diffusion piece and the M4 basis
!             stiffness (note Table 1, listing M2).
!   ELM_M3  : conservative rho-form clone of amat_55 (mass + E x B
!             conservative advection, diffusion dropped), R-weighted:
!               v u R (1+zeta)
!             - v R^2 (u_s u0_t - u_t u0_s) theta dt        (bracket)
!             - v 2 R u u0_y xjac theta dt                  (compression)
!             (note Table 1, listing M3; pairs with R-mass Q_u = amat_44).
!
! Both operators are p-channel only (no d_phi on trial/test): S1r has no
! background dependence (toroidally diagonal); M3's only phi-coupling is
! through the background flow u0 across planes, captured by the FFT of the
! plane-wise integrand -- exactly as amat_55's p-channel is assembled.
!
! Structure, geometry and FFT reconstruction follow
! mod_elt_matrix_metriplectic (first derivatives only; the 0.5 factor is
! the FFT normalisation, matching element_matrix_fft). Coefficient
! conventions (zeta = time_evol_zeta*2*dt/(dt+dt_prev), theta, dt) match
! models/model199/mod_elt_matrix_fft.f90 so the operators are directly
! comparable to the extracted A_full blocks.
!----------------------------------------------------------------
implicit none
public :: element_matrix_commutator

type :: type_fct_vals_cm
  real*8 :: v, v_x, v_y, v_s, v_t
end type type_fct_vals_cm

contains

subroutine element_matrix_commutator(element, nodes, ELM_S1r, ELM_M3)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module, only: fftw_plan, time_evol_theta, time_evol_zeta, tstep, tstep_prev
  use mod_elt_matrix_elliptic, only: scatter_fft_to_elm

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)

#define N1V  (n_vertex_max*n_degrees)
#define D1V  (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(D1V, D1V), intent(out) :: ELM_S1r, ELM_M3

  ! Plane-workspace accumulators (p channel)
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_S1r, ELM_p_M3

  ! Geometry at Gauss points (first derivatives only)
  real*8, dimension(n_gauss,n_gauss) :: x_g, x_s, x_t
  real*8, dimension(n_gauss,n_gauss) :: y_g, y_s, y_t

  ! Background flow potential u0: s,t derivatives at Gauss points/planes
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_s, eq_t

  integer :: i, j, k, l, ms, mt, mp, in
  integer :: idx_ij, idx_kl

  real*8 :: wst, xjac, BigR
  real*8 :: theta, zeta, opz
  real*8 :: u0_s, u0_t, u0_y
  type(type_fct_vals_cm) :: v_fct, u_fct

  real*8 :: amat_S1r, amat_M3

  ! FFT workspace
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  theta = time_evol_theta
  zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  opz   = 1.d0 + zeta

  ELM_S1r = 0.d0; ELM_M3 = 0.d0
  ELM_p_S1r = 0.d0; ELM_p_M3 = 0.d0

  !-----------------------------------------------------------------
  ! Geometry and background (u0) s,t derivatives at Gauss points
  ! (pattern: mod_elt_matrix_metriplectic :144-175, first order only)
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

        ! Background flow potential derivatives (u0 = var_u):
        !   u0_s, u0_t : computational-coord derivatives (bracket)
        !   u0_y       : Cartesian d/dZ (compression term)
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

                ! --- S1r: (1/R) grad(v).grad(u) ---
                amat_S1r = (v_fct%v_x*u_fct%v_x + v_fct%v_y*u_fct%v_y) / BigR * xjac

                ! --- M3: R-mass (1+zeta) + conservative E x B advection
                !     (clone of amat_55, rho -> trial u, diffusion dropped) ---
                amat_M3 =   v_fct%v * u_fct%v * BigR * opz * xjac                                          &
                          - v_fct%v * BigR**2 * ( u_fct%v_s * u0_t - u_fct%v_t * u0_s )   * theta * tstep   &
                          - v_fct%v * 2.d0 * BigR * u_fct%v * u0_y                * xjac  * theta * tstep

                ELM_p_S1r(mp, idx_ij, idx_kl) = ELM_p_S1r(mp, idx_ij, idx_kl) + wst * amat_S1r
                ELM_p_M3 (mp, idx_ij, idx_kl) = ELM_p_M3 (mp, idx_ij, idx_kl) + wst * amat_M3

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
  do i = 1, N1V
    do j = 1, N1V

      in_fft = ELM_p_S1r(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_S1r, D1V)

      in_fft = ELM_p_M3(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_M3, D1V)

    enddo
  enddo

  ELM_S1r = 0.5d0 * ELM_S1r
  ELM_M3  = 0.5d0 * ELM_M3

end subroutine element_matrix_commutator

end module mod_elt_matrix_commutator
