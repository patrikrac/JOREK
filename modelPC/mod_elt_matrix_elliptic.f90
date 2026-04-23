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
!
! All four integrands live only in ELM_p (mode-diagonal).
! FFT reconstruction is performed to obtain the mode-space blocks.
!----------------------------------------------------------------
implicit none
public :: element_matrix_elliptic, scatter_fft_to_elm

type :: type_fct_values
  real*8 :: v, v_x, v_y, v_s, v_t, v_p, v_ss, v_st, v_tt, v_xx, v_yy, v_xs, v_ys, v_xt, v_yt, v_xy
end type type_fct_values

contains

subroutine element_matrix_elliptic(element, nodes, ELM_j, ELM_w, ELM_jpsi, ELM_wu, ELM_psi_correction, ELM_u_correction)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module, only: fftw_plan, time_evol_theta, tstep, eta_T_dependent, eta, T_max_eta, T_0, visco_T_dependent, visco
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

  ! ELM_p workspace — local thread-stack buffers, zeroed each call
  ! ELM_p_w omitted: A_w uses the same mass integrand as A_j, so ELM_w = ELM_j
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_j
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_jpsi, ELM_p_wu
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_psi_correction, ELM_p_u_correction

  ! Geometry at Gauss points (first derivatives only — no 2nd derivs needed)
  real*8, dimension(n_gauss,n_gauss)    :: x_g, x_s, x_t
  real*8, dimension(n_gauss,n_gauss)    :: x_ss, x_st, x_tt
  real*8, dimension(n_gauss,n_gauss)    :: y_g, y_s, y_t
  real*8, dimension(n_gauss,n_gauss)    :: y_ss, y_st, y_tt

  ! Variables at Gauss points
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_g

  ! Loop indices
  integer :: i, j, ms, mt, mp, in, k, l, m, index_k, index_m
  integer :: idx_ij, idx_kl

  ! Gauss-point scalars
  real*8 :: wst, xjac, xjac_x, xjac_y, BigR
  type(type_fct_values) :: v_fct, psi_fct, u_fct
  real*8 :: amat_mass, amat_31, amat_42, amat_psi_correction, amat_u_correction

  ! Values at Gauss points
  real*8 :: T0

  ! Time stepping variables
  real*8 :: theta
  real*8 :: eta_T, visco_T

  ! Schur correction
  logical :: psi_correction, u_correction

  ! FFT workspace
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ! Check for optional Schur correction output
  if (present(ELM_psi_correction)) then
    psi_correction = .true.
  else
    psi_correction = .false.
  endif

  if (present(ELM_u_correction)) then
    u_correction = .true.
  else
    u_correction = .false.
  endif

  !-----------------------------------------------------------------
  ! Initialise
  !-----------------------------------------------------------------
  ELM_j    = 0.d0
  ELM_jpsi = 0.d0;  ELM_wu     = 0.d0
  if (psi_correction) ELM_psi_correction = 0.d0
  if (u_correction) ELM_u_correction = 0.d0
  ELM_p_j  = 0.d0
  ELM_p_jpsi = 0.d0;  ELM_p_wu = 0.d0
  if (psi_correction) ELM_p_psi_correction = 0.d0
  if (u_correction) ELM_p_u_correction = 0.d0

  !-----------------------------------------------------------------
  ! Geometry at Gauss points
  !-----------------------------------------------------------------
  x_g  = 0.d0; x_s  = 0.d0; x_t  = 0.d0; x_st  = 0.d0; x_ss  = 0.d0; x_tt  = 0.d0;
  y_g  = 0.d0; y_s  = 0.d0; y_t  = 0.d0; y_st  = 0.d0; y_ss  = 0.d0; y_tt  = 0.d0;
  eq_g = 0.d0

  theta = time_evol_theta

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
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)  * HZ(in,mp)
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
        ! --- Temperature dependent viscosity
        if (visco_T_dependent) then
          visco_T   = visco * (corr_neg_temp(T0)/T_0)**(-1.5d0)
        else
          visco_T   = visco
        end if

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

                u_fct = psi_fct

                !--- eq.3/4: amat_33 = amat_44 = v_fct%v * psi_fct%v * BigR * xjac
                amat_mass = v_fct%v * psi_fct%v * BigR * xjac

                !--- eq.3: amat_31 (psi->j coupling)
                amat_31 = (v_fct%v_x*psi_fct%v_x + v_fct%v_y*psi_fct%v_y) * BigR * xjac &
                        + 2.d0 * v_fct%v * psi_fct%v_x * BigR * xjac

                !--- eq.4: amat_42 (u->w coupling)
                amat_42 = (v_fct%v_x*psi_fct%v_x + v_fct%v_y*psi_fct%v_y) * BigR * xjac

                ! --- Schur correction equation ---
                amat_psi_correction = - (v_fct%v_x * psi_fct%v_x + v_fct%v_y * psi_fct%v_y) * (eta_T / BigR) * xjac * theta * tstep
                ! amat_psi_correction = (v_fct%v_x * psi_fct%v_x + v_fct%v_y * psi_fct%v_y + 2 * v_fct%v * psi_fct%v_x) * (eta_T / BigR) * xjac * theta * tstep
                !amat_u_correction = 0.d0
                amat_u_correction = visco_T * BigR * (v_fct%v_xx + v_fct%v_x/BigR + v_fct%v_yy) * (u_fct%v_xx + u_fct%v_x/BigR + u_fct%v_yy) * xjac * theta * tstep
                
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
                endif

              enddo  ! l
            enddo    ! k

          enddo  ! j
        enddo    ! i

      enddo  ! mp

    enddo  ! mt
  enddo    ! ms

  !-----------------------------------------------------------------
  ! FFT reconstruction — 1-var matrix (A_j); A_w is a copy
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
      endif
      
    enddo
  enddo

  ELM_j = 0.5d0 * ELM_j
  ELM_w = ELM_j   ! identical mass integrand: amat_33 = amat_44
  if (psi_correction) ELM_psi_correction = 0.5d0 * ELM_psi_correction
  if (u_correction) ELM_u_correction = 0.5d0 * ELM_u_correction

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

end module mod_elt_matrix_elliptic
