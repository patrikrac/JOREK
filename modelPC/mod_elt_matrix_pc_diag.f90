module mod_elt_matrix_pc_diag
!----------------------------------------------------------------
! Simplified diagonal-block element matrices for the physics PC.
!
! Assembles four 1-var mode-space element matrices for the
! diagonal blocks of the reduced 4x4 system (model199):
!
!   ELM_11 : eq.1 (psi) x var psi  — mass / R
!   ELM_22 : eq.2 (u)   x var u    — R^3-weighted stiffness
!   ELM_55 : eq.5 (rho) x var rho  — mass + perp diffusion
!   ELM_66 : eq.6 (T)   x var T    — mass + perp conduction
!
! These are SIMPLIFIED versions of the full Jacobian blocks,
! retaining only the dominant physics needed for preconditioning.
! They do not depend on equilibrium field data and are therefore
! axisymmetric (mode-diagonal in toroidal harmonics).
!
! Simplifications relative to the full Jacobian:
!   B_11: dropped flow advection term
!   B_22: dropped density weighting (rho -> 1), advection
!   B_55: dropped parallel transport, advection; constant D_perp
!   B_66: dropped parallel transport, ohmic heating, advection;
!          constant ZK_perp
!----------------------------------------------------------------
implicit none
public :: element_matrix_pc_diag

contains

subroutine element_matrix_pc_diag(element, nodes, ELM_11, ELM_22, ELM_55, ELM_66)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module, only: fftw_plan, time_evol_theta, time_evol_zeta, &
                         tstep, tstep_prev, D_perp, ZK_perp
  use mod_elt_matrix_elliptic, only: scatter_fft_to_elm

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)

  ! Output dimensions (compile-time constants from mod_parameters)
#define N1V_D  (n_vertex_max*n_degrees)
#define D1V_D  (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(D1V_D, D1V_D), intent(out) :: ELM_11, ELM_22, ELM_55, ELM_66

  ! Physical-space accumulators (one per toroidal plane)
  real*8, dimension(n_plane, N1V_D, N1V_D) :: ELM_p_11, ELM_p_22
  real*8, dimension(n_plane, N1V_D, N1V_D) :: ELM_p_55, ELM_p_66

  ! Geometry at Gauss points (first derivatives only)
  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t
  real*8, dimension(n_gauss, n_gauss) :: y_g, y_s, y_t

  ! Loop indices
  integer :: i, j, ms, mt, mp, k, l
  integer :: idx_ij, idx_kl

  ! Gauss-point scalars
  real*8 :: wst, xjac, BigR
  real*8 :: v, v_x, v_y
  real*8 :: psi, psi_x, psi_y
  real*8 :: theta, zeta
  real*8 :: D_perp_val, ZK_perp_val
  real*8 :: amat_pc_11, amat_pc_22, amat_pc_55, amat_pc_66

  ! FFT workspace
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  !-----------------------------------------------------------------
  ! Time evolution parameters (same convention as model199)
  !-----------------------------------------------------------------
  theta = time_evol_theta
  zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)

  ! Constant transport coefficients (central/reference values)
  D_perp_val  = D_perp(1)
  ZK_perp_val = ZK_perp(1)

  !-----------------------------------------------------------------
  ! Initialise
  !-----------------------------------------------------------------
  ELM_11 = 0.d0;  ELM_22 = 0.d0;  ELM_55 = 0.d0;  ELM_66 = 0.d0
  ELM_p_11 = 0.d0;  ELM_p_22 = 0.d0
  ELM_p_55 = 0.d0;  ELM_p_66 = 0.d0

  !-----------------------------------------------------------------
  ! Geometry at Gauss points
  !-----------------------------------------------------------------
  x_g = 0.d0;  x_s = 0.d0;  x_t = 0.d0
  y_g = 0.d0;  y_s = 0.d0;  y_t = 0.d0

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

        do i = 1, n_vertex_max
          do j = 1, n_degrees

            ! 1-var DOF index (test function, row)
            idx_ij = n_degrees*(i-1) + (j-1) + 1

            ! Test function value and Cartesian gradients
            v   =   H(i,j,ms,mt)   * element%size(i,j)
            v_x = (   y_t(ms,mt) * H_s(i,j,ms,mt) &
                    - y_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac
            v_y = ( - x_t(ms,mt) * H_s(i,j,ms,mt) &
                    + x_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac

            do k = 1, n_vertex_max
              do l = 1, n_degrees

                ! 1-var DOF index (trial function, col)
                idx_kl = n_degrees*(k-1) + (l-1) + 1

                ! Trial function value and Cartesian gradients
                psi   =   H(k,l,ms,mt)   * element%size(k,l)
                psi_x = (   y_t(ms,mt) * H_s(k,l,ms,mt) &
                          - y_s(ms,mt) * H_t(k,l,ms,mt) ) / xjac * element%size(k,l)
                psi_y = ( - x_t(ms,mt) * H_s(k,l,ms,mt) &
                          + x_s(ms,mt) * H_t(k,l,ms,mt) ) / xjac * element%size(k,l)

                !--- B_11 (psi eq, psi var): mass / R
                ! Full: v*psi/BigR*xjac*(1+zeta) - v*(psi_s*u0_t - psi_t*u0_s)*theta*tstep
                ! PC:   mass term only (drop equilibrium flow advection)
                amat_pc_11 = v * psi / BigR * xjac * (1.d0 + zeta)

                !--- B_22 (u eq, u var): R^3-weighted stiffness
                ! Full: -BigR*R^2*rho*(v_x*u_x + v_y*u_y)*xjac*(1+zeta) + advection + ...
                ! PC:   stiffness with rho=1 (drop density variation and advection)
                amat_pc_22 = -BigR**3 * (v_x * psi_x + v_y * psi_y) * xjac * (1.d0 + zeta)

                !--- B_55 (rho eq, rho var): mass + perpendicular diffusion
                ! Full: mass + parallel transport + perp diffusion + advection
                ! PC:   mass + constant perp diffusion
                amat_pc_55 = v * psi * BigR * xjac * (1.d0 + zeta) &
                           + D_perp_val * BigR * (v_x*psi_x + v_y*psi_y) * xjac * theta * tstep

                !--- B_66 (T eq, T var): mass + perpendicular conduction
                ! Full: mass + parallel transport + perp conduction + ohmic + advection
                ! PC:   mass + constant perp conduction
                amat_pc_66 = v * psi * BigR * xjac * (1.d0 + zeta) &
                           + ZK_perp_val * BigR * (v_x*psi_x + v_y*psi_y) * xjac * theta * tstep

                ! Accumulate into physical-space arrays
                ELM_p_11(mp, idx_ij, idx_kl) = ELM_p_11(mp, idx_ij, idx_kl) + wst * amat_pc_11
                ELM_p_22(mp, idx_ij, idx_kl) = ELM_p_22(mp, idx_ij, idx_kl) + wst * amat_pc_22
                ELM_p_55(mp, idx_ij, idx_kl) = ELM_p_55(mp, idx_ij, idx_kl) + wst * amat_pc_55
                ELM_p_66(mp, idx_ij, idx_kl) = ELM_p_66(mp, idx_ij, idx_kl) + wst * amat_pc_66

              enddo  ! l
            enddo    ! k

          enddo  ! j
        enddo    ! i

      enddo  ! mp

    enddo  ! mt
  enddo    ! ms

  !-----------------------------------------------------------------
  ! FFT reconstruction — physical space to mode space
  !-----------------------------------------------------------------
  do i = 1, N1V_D
    do j = 1, N1V_D

      in_fft = ELM_p_11(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_11, D1V_D)

      in_fft = ELM_p_22(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_22, D1V_D)

      in_fft = ELM_p_55(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_55, D1V_D)

      in_fft = ELM_p_66(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_66, D1V_D)

    enddo
  enddo

  ELM_11 = 0.5d0 * ELM_11
  ELM_22 = 0.5d0 * ELM_22
  ELM_55 = 0.5d0 * ELM_55
  ELM_66 = 0.5d0 * ELM_66

end subroutine element_matrix_pc_diag

end module mod_elt_matrix_pc_diag
