module mod_elt_matrix_metriplectic
!----------------------------------------------------------------
! Element-level operators for the metriplectic HSS preconditioner
! (Slice A: analysis operators).
!
! Assembles six mode-space element matrices, derived from the
! continuous ideal Alfven pair with the elliptic constraints
! substituted (j = Delta*psi never an independent field):
!
!   ELM_Mpsi      : psi-row mass, weight 1/R
!   ELM_D         : weak R^2(B.grad), 1/R-tested   (psi-row, u-col)
!   ELM_Dp        : weak R(B.grad)Delta*, flat-tested, skew-moved form
!                     -(B.grad w)(Delta* psi) R     (u-row, psi-col)
!   ELM_Dp_struct : same operator, fully-by-parts form
!                     (1/R) grad(R^2 B.grad w) . grad(psi)
!                   (T1 partner: difference = discrete flat IBP + BT2)
!   ELM_Lrho      : A_rho = rho-hat R grad(w).grad(u)  (SPD)
!   ELM_Wpara     : (1/R) grad(R^2 B.grad w) . grad(R^2 B.grad u)
!                   (symmetric PSD parabolized operator)
!
! Coefficient conventions (R-exponents, signs, Gears normalization
! tau = theta*dt/(1+zeta)): docs/notes/metriplectic_parabolization_note.tex,
! Sec. 7 coefficient table. The COEFF block below is the ONLY place
! those constants live in code.
!
! The u-row operators are assembled in the SPD-oriented convention
! (JOREK's u-row times -1); the apply must negate the u-residual on
! input (note, Sec. 6).
!
! FFT reconstruction to mode space follows mod_elt_matrix_elliptic;
! channels: p (plain), n (d_phi on trial, column-sided mode factor),
! nrow (d_phi on test, row-sided; local scatter routine), kn (both).
!----------------------------------------------------------------
implicit none
public :: element_matrix_metriplectic

type :: type_fct_vals
  real*8 :: v, v_x, v_y, v_s, v_t, v_ss, v_st, v_tt, v_xx, v_yy, v_xy
end type type_fct_vals

! =====================================================================
! COEFF block — single source for R-exponents and signs
! (docs/notes/metriplectic_parabolization_note.tex, Sec. 7).
! =====================================================================
integer, parameter :: EXP_MPSI  = -1  !< M_psi:  v*psi * R**EXP_MPSI
integer, parameter :: EXP_DBRK  =  0  !< D bracket: v*[u,psi0] * R**EXP_DBRK
integer, parameter :: EXP_DTOR  = -1  !< D toroidal: v*F0*(d_phi u) * R**EXP_DTOR
integer, parameter :: EXP_A     =  2  !< R**EXP_A inside grad(R**EXP_A * B.grad w)
integer, parameter :: EXP_DPW   = -1  !< D' by-parts outer weight R**EXP_DPW
integer, parameter :: EXP_WPARA = -1  !< W_para outer weight R**EXP_WPARA
real*8,  parameter :: SGN_D     = +1.d0
real*8,  parameter :: SGN_DP    = +1.d0  !< after u-row negation (note Sec. 6)
real*8,  parameter :: SGN_WPARA = +1.d0

contains

subroutine element_matrix_metriplectic(element, nodes, ELM_Mpsi, ELM_D, &
                                       ELM_Dp, ELM_Dp_struct, ELM_Lrho, ELM_Wpara)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module, only: fftw_plan, F0
  use mod_elt_matrix_elliptic, only: scatter_fft_to_elm, scatter_fft_to_elm_n, &
                                     scatter_fft_to_elm_kn

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)

#define N1V  (n_vertex_max*n_degrees)
#define D1V  (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(D1V, D1V), intent(out) :: ELM_Mpsi, ELM_D
  real*8, dimension(D1V, D1V), intent(out) :: ELM_Dp, ELM_Dp_struct
  real*8, dimension(D1V, D1V), intent(out) :: ELM_Lrho, ELM_Wpara

  ! Plane-workspace accumulators (channel-separated)
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_Mpsi
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_D,    ELM_n_D
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_Dp,   ELM_nr_Dp
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_Dps,  ELM_nr_Dps
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_Lrho
  real*8, dimension(n_plane, N1V, N1V) :: ELM_p_Wp, ELM_n_Wp, ELM_nr_Wp, ELM_kn_Wp

  ! Geometry at Gauss points
  real*8, dimension(n_gauss,n_gauss) :: x_g, x_s, x_t, x_ss, x_st, x_tt
  real*8, dimension(n_gauss,n_gauss) :: y_g, y_s, y_t, y_ss, y_st, y_tt

  ! Background fields at Gauss points (values + s,t first/second derivatives)
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_g, eq_s, eq_t
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_ss, eq_st, eq_tt

  integer :: i, j, k, l, ms, mt, mp, in
  integer :: idx_ij, idx_kl

  real*8 :: wst, xjac, xjac_x, xjac_y, BigR
  type(type_fct_vals) :: v_fct, u_fct

  ! Background values at the current Gauss point / plane
  real*8 :: r0, r0_hat
  real*8 :: ps0_x, ps0_y, ps0_xx, ps0_yy, ps0_xy

  ! B.grad building blocks
  real*8 :: brk_u, brk_v                    ! [f,psi0]
  real*8 :: dbrk_u_x, dbrk_u_y, dbrk_v_x, dbrk_v_y
  real*8 :: Ax_u, Ay_u, Ax_v, Ay_v          ! grad(R**(EXP_A-1) * [f,psi0])
  real*8 :: gs_u                            ! Delta* of trial (GS operator)

  ! Integrands
  real*8 :: amat_Mpsi, amat_D_p, amat_D_n
  real*8 :: amat_Dp_p, amat_Dp_nr, amat_Dps_p, amat_Dps_nr
  real*8 :: amat_Lrho
  real*8 :: amat_Wp_p, amat_Wp_n, amat_Wp_nr, amat_Wp_kn

  ! FFT workspace
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  !-----------------------------------------------------------------
  ! Initialise
  !-----------------------------------------------------------------
  ELM_Mpsi = 0.d0; ELM_D = 0.d0
  ELM_Dp = 0.d0;   ELM_Dp_struct = 0.d0
  ELM_Lrho = 0.d0; ELM_Wpara = 0.d0
  ELM_p_Mpsi = 0.d0
  ELM_p_D    = 0.d0; ELM_n_D   = 0.d0
  ELM_p_Dp   = 0.d0; ELM_nr_Dp = 0.d0
  ELM_p_Dps  = 0.d0; ELM_nr_Dps = 0.d0
  ELM_p_Lrho = 0.d0
  ELM_p_Wp = 0.d0; ELM_n_Wp = 0.d0; ELM_nr_Wp = 0.d0; ELM_kn_Wp = 0.d0

  !-----------------------------------------------------------------
  ! Geometry and background fields at Gauss points
  ! (pattern: mod_elt_matrix_elliptic.f90:186-229)
  !-----------------------------------------------------------------
  x_g  = 0.d0; x_s  = 0.d0; x_t  = 0.d0; x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0
  y_g  = 0.d0; y_s  = 0.d0; y_t  = 0.d0; y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0
  eq_ss = 0.d0; eq_st = 0.d0; eq_tt = 0.d0

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
                eq_g(mp,k,ms,mt)  = eq_g(mp,k,ms,mt)  + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)    * HZ(in,mp)
                eq_s(mp,k,ms,mt)  = eq_s(mp,k,ms,mt)  + nodes(i)%values(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ(in,mp)
                eq_t(mp,k,ms,mt)  = eq_t(mp,k,ms,mt)  + nodes(i)%values(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ(in,mp)
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
                + y_st(ms,mt)*(x_s(ms,mt)*y_t(ms,mt) + x_t(ms,mt)*y_s(ms,mt))                                             &
                + x_tt(ms,mt)*y_s(ms,mt)**2 - y_tt(ms,mt)*x_s(ms,mt)*y_s(ms,mt)) / xjac
      xjac_y  = (y_tt(ms,mt)*x_s(ms,mt)**2 - x_tt(ms,mt)*y_s(ms,mt)*x_s(ms,mt) - 2.d0*y_st(ms,mt)*x_t(ms,mt)*x_s(ms,mt)   &
                + x_st(ms,mt)*(y_t(ms,mt)*x_s(ms,mt) + y_s(ms,mt)*x_t(ms,mt))                                             &
                + y_ss(ms,mt)*x_t(ms,mt)**2 - x_ss(ms,mt)*y_t(ms,mt)*x_t(ms,mt)) / xjac
      BigR = x_g(ms,mt)

      do mp = 1, n_plane

        r0     = abs(eq_g(mp,var_rho,ms,mt))
        r0_hat = BigR**2 * r0

        ! Background psi gradient and second derivatives (Cartesian R,Z)
        ! (pattern: mod_elt_matrix_elliptic.f90:289-305)
        ps0_x = (   y_t(ms,mt) * eq_s(mp,var_psi,ms,mt) - y_s(ms,mt) * eq_t(mp,var_psi,ms,mt) ) / xjac
        ps0_y = ( - x_t(ms,mt) * eq_s(mp,var_psi,ms,mt) + x_s(ms,mt) * eq_t(mp,var_psi,ms,mt) ) / xjac

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

        do i = 1, n_vertex_max
          do j = 1, n_degrees

            idx_ij = n_degrees*(i-1) + (j-1) + 1

            ! Test function values/derivatives (pattern: elliptic :320-348)
            v_fct%v   =   H(i,j,ms,mt)   * element%size(i,j)
            v_fct%v_x = (   y_t(ms,mt) * H_s(i,j,ms,mt) - y_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac
            v_fct%v_y = ( - x_t(ms,mt) * H_s(i,j,ms,mt) + x_s(ms,mt) * H_t(i,j,ms,mt) ) * element%size(i,j) / xjac
            v_fct%v_s  = H_s(i,j,ms,mt)  * element%size(i,j)
            v_fct%v_t  = H_t(i,j,ms,mt)  * element%size(i,j)
            v_fct%v_ss = H_ss(i,j,ms,mt) * element%size(i,j)
            v_fct%v_tt = H_tt(i,j,ms,mt) * element%size(i,j)
            v_fct%v_st = H_st(i,j,ms,mt) * element%size(i,j)
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

            ! Test-side B.grad building blocks
            brk_v    = v_fct%v_x*ps0_y - v_fct%v_y*ps0_x
            dbrk_v_x = v_fct%v_xx*ps0_y + v_fct%v_x*ps0_xy - v_fct%v_xy*ps0_x - v_fct%v_y*ps0_xx
            dbrk_v_y = v_fct%v_xy*ps0_y + v_fct%v_x*ps0_yy - v_fct%v_yy*ps0_x - v_fct%v_y*ps0_xy
            ! grad(R**(EXP_A-1) * [v,psi0]):
            Ax_v = dble(EXP_A-1)*BigR**(EXP_A-2)*brk_v + BigR**(EXP_A-1)*dbrk_v_x
            Ay_v = BigR**(EXP_A-1)*dbrk_v_y

            do k = 1, n_vertex_max
              do l = 1, n_degrees

                idx_kl = n_degrees*(k-1) + (l-1) + 1

                ! Trial function values/derivatives (pattern: elliptic :357-380)
                u_fct%v   =   H(k,l,ms,mt)   * element%size(k,l)
                u_fct%v_x = (   y_t(ms,mt) * H_s(k,l,ms,mt) - y_s(ms,mt) * H_t(k,l,ms,mt) ) / xjac * element%size(k,l)
                u_fct%v_y = ( - x_t(ms,mt) * H_s(k,l,ms,mt) + x_s(ms,mt) * H_t(k,l,ms,mt) ) / xjac * element%size(k,l)
                u_fct%v_s  = H_s(k,l,ms,mt)  * element%size(k,l)
                u_fct%v_t  = H_t(k,l,ms,mt)  * element%size(k,l)
                u_fct%v_ss = H_ss(k,l,ms,mt) * element%size(k,l)
                u_fct%v_tt = H_tt(k,l,ms,mt) * element%size(k,l)
                u_fct%v_st = H_st(k,l,ms,mt) * element%size(k,l)
                u_fct%v_xx = (u_fct%v_ss * y_t(ms,mt)**2 - 2.d0*u_fct%v_st * y_s(ms,mt)*y_t(ms,mt) + u_fct%v_tt * y_s(ms,mt)**2  &
                        + u_fct%v_s * (y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt) )                              &
                        + u_fct%v_t * (y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt) ) )    / xjac**2               &
                        - xjac_x * (u_fct%v_s * y_t(ms,mt) - u_fct%v_t * y_s(ms,mt)) / xjac**2
                u_fct%v_yy = (u_fct%v_ss * x_t(ms,mt)**2 - 2.d0*u_fct%v_st * x_s(ms,mt)*x_t(ms,mt) + u_fct%v_tt * x_s(ms,mt)**2  &
                        + u_fct%v_s * (x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt) )                              &
                        + u_fct%v_t * (x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt) ) )    / xjac**2               &
                        - xjac_y * (- u_fct%v_s * x_t(ms,mt) + u_fct%v_t * x_s(ms,mt) ) / xjac**2
                u_fct%v_xy = (- u_fct%v_ss * y_t(ms,mt)*x_t(ms,mt) - u_fct%v_tt * x_s(ms,mt)*y_s(ms,mt)                    &
                      + u_fct%v_st * (y_s(ms,mt)*x_t(ms,mt)  + y_t(ms,mt)*x_s(ms,mt)  )                         &
                      - u_fct%v_s  * (x_st(ms,mt)*y_t(ms,mt) - x_tt(ms,mt)*y_s(ms,mt) )                         &
                      - u_fct%v_t  * (x_st(ms,mt)*y_s(ms,mt) - x_ss(ms,mt)*y_t(ms,mt) )  )  / xjac**2           &
                      - xjac_x * (- u_fct%v_s * x_t(ms,mt) + u_fct%v_t * x_s(ms,mt) )   / xjac**2

                ! Trial-side B.grad building blocks
                brk_u    = u_fct%v_x*ps0_y - u_fct%v_y*ps0_x
                dbrk_u_x = u_fct%v_xx*ps0_y + u_fct%v_x*ps0_xy - u_fct%v_xy*ps0_x - u_fct%v_y*ps0_xx
                dbrk_u_y = u_fct%v_xy*ps0_y + u_fct%v_x*ps0_yy - u_fct%v_yy*ps0_x - u_fct%v_y*ps0_xy
                Ax_u = dble(EXP_A-1)*BigR**(EXP_A-2)*brk_u + BigR**(EXP_A-1)*dbrk_u_x
                Ay_u = BigR**(EXP_A-1)*dbrk_u_y

                ! Grad-Shafranov operator of the trial (psi-slot): Delta* psi
                gs_u = u_fct%v_xx - u_fct%v_x/BigR + u_fct%v_yy

                ! --- M_psi (p): v*psi * R**EXP_MPSI ---
                amat_Mpsi = v_fct%v * u_fct%v * BigR**EXP_MPSI * xjac

                ! --- D (psi-row, u-col): bracket (p) + toroidal (n, d_phi on trial) ---
                amat_D_p = SGN_D * v_fct%v * brk_u * BigR**EXP_DBRK * xjac
                amat_D_n = SGN_D * F0 * v_fct%v * u_fct%v * BigR**EXP_DTOR * xjac

                ! --- D' (u-row, psi-col), skew-moved form: -(B.grad w)(Delta* psi)*R
                !     bracket part: -(1/R)[w,psi0]*Delta*psi*R = -[w,psi0]*Delta*psi
                !     toroidal part (d_phi on TEST, nrow): -(F0/R)*w*Delta*psi ---
                amat_Dp_p  = -SGN_DP * brk_v * gs_u * xjac
                amat_Dp_nr = -SGN_DP * F0/BigR * v_fct%v * gs_u * xjac

                ! --- D' by-parts form (T1 partner):
                !     (1/R) grad(R**EXP_A * B.grad w) . grad(psi)
                !     pol (p) + toroidal on TEST (nrow) ---
                amat_Dps_p  = SGN_DP * (Ax_v*u_fct%v_x + Ay_v*u_fct%v_y) * BigR**EXP_DPW * xjac
                amat_Dps_nr = SGN_DP * F0 * (v_fct%v_x*u_fct%v_x + v_fct%v_y*u_fct%v_y) &
                              * BigR**(EXP_DPW + EXP_A - 2) * xjac

                ! --- L_rho = A_rho: rho_hat * R * grad(w).grad(u), SPD ---
                amat_Lrho = r0_hat * BigR * (v_fct%v_x*u_fct%v_x + v_fct%v_y*u_fct%v_y) * xjac

                ! --- W_para: (1/R) grad(R^2 B.grad w).grad(R^2 B.grad u), 4 channels ---
                amat_Wp_p  = SGN_WPARA * (Ax_v*Ax_u + Ay_v*Ay_u) * BigR**EXP_WPARA * xjac
                amat_Wp_n  = SGN_WPARA * F0 * (Ax_v*u_fct%v_x + Ay_v*u_fct%v_y) &
                             * BigR**(EXP_WPARA + EXP_A - 2) * xjac
                amat_Wp_nr = SGN_WPARA * F0 * (v_fct%v_x*Ax_u + v_fct%v_y*Ay_u) &
                             * BigR**(EXP_WPARA + EXP_A - 2) * xjac
                amat_Wp_kn = SGN_WPARA * F0**2 * (v_fct%v_x*u_fct%v_x + v_fct%v_y*u_fct%v_y) &
                             * BigR**(EXP_WPARA + 2*EXP_A - 4) * xjac

                ! --- Accumulate ---
                ELM_p_Mpsi(mp, idx_ij, idx_kl) = ELM_p_Mpsi(mp, idx_ij, idx_kl) + wst * amat_Mpsi
                ELM_p_D   (mp, idx_ij, idx_kl) = ELM_p_D   (mp, idx_ij, idx_kl) + wst * amat_D_p
                ELM_n_D   (mp, idx_ij, idx_kl) = ELM_n_D   (mp, idx_ij, idx_kl) + wst * amat_D_n
                ELM_p_Dp  (mp, idx_ij, idx_kl) = ELM_p_Dp  (mp, idx_ij, idx_kl) + wst * amat_Dp_p
                ELM_nr_Dp (mp, idx_ij, idx_kl) = ELM_nr_Dp (mp, idx_ij, idx_kl) + wst * amat_Dp_nr
                ELM_p_Dps (mp, idx_ij, idx_kl) = ELM_p_Dps (mp, idx_ij, idx_kl) + wst * amat_Dps_p
                ELM_nr_Dps(mp, idx_ij, idx_kl) = ELM_nr_Dps(mp, idx_ij, idx_kl) + wst * amat_Dps_nr
                ELM_p_Lrho(mp, idx_ij, idx_kl) = ELM_p_Lrho(mp, idx_ij, idx_kl) + wst * amat_Lrho
                ELM_p_Wp  (mp, idx_ij, idx_kl) = ELM_p_Wp  (mp, idx_ij, idx_kl) + wst * amat_Wp_p
                ELM_n_Wp  (mp, idx_ij, idx_kl) = ELM_n_Wp  (mp, idx_ij, idx_kl) + wst * amat_Wp_n
                ELM_nr_Wp (mp, idx_ij, idx_kl) = ELM_nr_Wp (mp, idx_ij, idx_kl) + wst * amat_Wp_nr
                ELM_kn_Wp (mp, idx_ij, idx_kl) = ELM_kn_Wp (mp, idx_ij, idx_kl) + wst * amat_Wp_kn

              enddo  ! l
            enddo    ! k
          enddo  ! j
        enddo    ! i
      enddo  ! mp
    enddo  ! mt
  enddo    ! ms

  !-----------------------------------------------------------------
  ! FFT reconstruction and mode-space scatter
  ! (pattern: mod_elt_matrix_elliptic.f90:529-653; 0.5 factor = FFT norm)
  !-----------------------------------------------------------------
  do i = 1, N1V
    do j = 1, N1V

      in_fft = ELM_p_Mpsi(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_Mpsi, D1V)

      in_fft = ELM_p_D(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_D, D1V)

      in_fft = ELM_n_D(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm_n(out_fft, i, j, ELM_D, D1V)

      in_fft = ELM_p_Dp(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_Dp, D1V)

      in_fft = ELM_nr_Dp(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm_nrow(out_fft, i, j, ELM_Dp, D1V)

      in_fft = ELM_p_Dps(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_Dp_struct, D1V)

      in_fft = ELM_nr_Dps(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm_nrow(out_fft, i, j, ELM_Dp_struct, D1V)

      in_fft = ELM_p_Lrho(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_Lrho, D1V)

      in_fft = ELM_p_Wp(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm(out_fft, i, j, ELM_Wpara, D1V)

      in_fft = ELM_n_Wp(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm_n(out_fft, i, j, ELM_Wpara, D1V)

      in_fft = ELM_nr_Wp(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm_nrow(out_fft, i, j, ELM_Wpara, D1V)

      in_fft = ELM_kn_Wp(1:n_plane, i, j)
#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
      call scatter_fft_to_elm_kn(out_fft, i, j, ELM_Wpara, D1V)

    enddo
  enddo

  ELM_Mpsi      = 0.5d0 * ELM_Mpsi
  ELM_D         = 0.5d0 * ELM_D
  ELM_Dp        = 0.5d0 * ELM_Dp
  ELM_Dp_struct = 0.5d0 * ELM_Dp_struct
  ELM_Lrho      = 0.5d0 * ELM_Lrho
  ELM_Wpara     = 0.5d0 * ELM_Wpara

end subroutine element_matrix_metriplectic


!-----------------------------------------------------------------
! Scatter FFT output for row i, col j into ELM with toroidal
! mode-number weighting on the TEST (row) function: row-sided
! companion of scatter_fft_to_elm_n (which is column-sided).
!
! Derivation: for a kernel c(phi), the (k,m) harmonic block of
! int (d_phi v_k) u_m c dphi equals the TRANSPOSE of the _n pattern
! with the roles of the harmonics exchanged: the sum branch
! l=(k-1)+(m-1) is symmetric under the exchange, the difference
! branch flips sign, l -> (m-1)-(k-1). Factor float(mode(ik)).
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm_nrow(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane
  use phys_module,    only: mode

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, ik, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    ik      = max(2*(k-1), 1)
    index_k = n_tor*(i-1) + ik

    do m = 1, (n_tor+1)/2

      index_m = n_tor*(j-1) + max(2*(m-1), 1)

      ! Sum branch: same pattern as scatter_fft_to_elm_n (symmetric block)
      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(l+1))      * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(l+1))      * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(l+1))      * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(l+1))      * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(abs(l)+1)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(abs(l)+1)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(abs(l)+1)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(abs(l)+1)) * float(mode(ik))
      endif

      ! Difference branch: l = (m-1)-(k-1), pattern = transpose of _n's
      l = (m-1) - (k-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(l+1))      * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(l+1))      * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - real(out_fft(l+1))      * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(l+1))      * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(abs(l)+1)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(abs(l)+1)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - real(out_fft(abs(l)+1)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(abs(l)+1)) * float(mode(ik))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm_nrow

end module mod_elt_matrix_metriplectic
