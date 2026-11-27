module mod_elt_matrix_fft
implicit none
contains
subroutine element_matrix_fft(element, nodes, xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, ELM, RHS, tid, &
  ELM_p, ELM_n, ELM_k, ELM_kn, RHS_p, RHS_k,  eq_g, eq_s, eq_t, eq_p, eq_ss, eq_st, eq_tt, delta_g, delta_s, delta_t, i_tor_min, i_tor_max, aux_nodes, ELM_pnn)
!---------------------------------------------------------------
! calculates the matrix contribution of one element
!
! TO BE CHECKED (see element_matrix)
!
!---------------------------------------------------------------
use constants
use mod_parameters
use data_structure
use gauss
use basis_at_gaussian
use phys_module
use tr_module
use diffusivities, only: get_dperp, get_zkperp, get_zk_iperp, get_zk_eperp
use corr_neg
use equil_info, only: get_psi_n
use mod_semianalytical
use mod_equations
use mod_chi
use mod_sources
use mod_pfafem, only: pfafem_ng_tor

implicit none
 
type (type_element), intent(in)           :: element
type (type_node),    intent(in)           :: nodes(n_vertex_max)
type (type_node),    intent(in), optional :: aux_nodes(n_vertex_max) ! particle moments

#define DIM0 n_tor*n_vertex_max*(n_order+1)*n_var

real*8, dimension (DIM0,DIM0)       :: ELM
real*8, dimension (DIM0)            :: RHS
integer                , intent(in) :: tid
integer                , intent(in) :: i_tor_min, i_tor_max

integer    :: i, j, ms, mt, mp, k, l, index_ij, index_kl, index, index_k, index_m, m, ik, xcase2
integer    :: in, im, ij, kl, i_var, j_var, n_tor_start, n_tor_end, n_tor_local
real*8     :: wst, prefactor, xjac, xjac_x, xjac_y, xjac_s, xjac_t, BigR, r2, phi, eps_cyl
real*8     :: R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint(2), Z_xpoint(2), dj_dpsi, dj_dz
real*8     :: Bgrad_rho_star,     Bgrad_rho,     Bgrad_T_star,  Bgrad_T, BB2
real*8     :: Bgrad_rho_star_psi, Bgrad_rho_psi, Bgrad_rho_rho, Bgrad_T_star_psi, Bgrad_T_psi, Bgrad_T_T, BB2_psi
real*8     :: Bgrad_rho_rho_n, Bgrad_T_T_n, Bgrad_rho_k_star, Bgrad_T_k_star
real*8     :: D_prof, ZK_prof, psi_norm

real*8     :: x_p_x, x_p_y, y_p_x, y_p_y, v_px, v_py, u_px, u_py
real*8     :: v, v_x, v_y, v_s, v_t, v_p, v_ss, v_st, v_tt, v_xx, v_yy, v_xs, v_ys, v_xt, v_yt, v_xy
real*8     :: ps0, ps0_x, ps0_y, ps0_p,ps0_s,ps0_t,  zj0, zj0_x, zj0_y, zj0_p, zj0_s, zj0_t
real*8     :: u0, u0_x, u0_y, u0_p, u0_s, u0_t,  w0, w0_x, w0_y, w0_p, w0_s, w0_t
real*8     :: r0, r0_x, r0_y, r0_p, r0_s, r0_t, r0_ss, r0_hat, r0_x_hat, r0_y_hat, T0, T0_x, T0_y, T0_p, T0_s, T0_t, T0_ss
real*8     :: psi, psi_x, psi_y, psi_p, psi_s, psi_t, psi_ss, psi_st, psi_tt, psi_xs, psi_ys, psi_xt, psi_yt, psi_xx, psi_yy
real*8     :: zj, zj_x, zj_y, zj_p, zj_s, zj_t
real*8     :: u, u_x, u_y, u_p, u_s, u_t, w, w_x, w_y, w_p, w_s, w_t, w_xx, w_yy
real*8     :: rho, rho_x, rho_y, rho_s, rho_t, rho_p, rho_hat, rho_x_hat, rho_y_hat, T, T_x, T_y, T_s, T_t, T_p
real*8     :: w0_xs, w0_xt, w0_ys, w0_yt, w0_xx, w0_yy, w0_xy, w0_ss, w0_tt, w0_st, P0, P0_s, P0_t, P0_x, P0_y
real*8     :: rho0, rho0_s, rho0_ss
real*8     :: T0_i, T0_i_s, T0_i_ss
real*8     :: T0_e, T0_e_s, T0_e_ss
real*8     :: BigR_x, vv2, eta_T, visco_T, deta_dT, d2eta_d2T, dvisco_dT
real*8     :: theta, zeta, reta
logical    :: xpoint2, use_fft

real*8     :: T0_i_corr, T0_e_corr                      !> temperature for T^(-1.5)
real*8     :: T0_e_corr_eV, dT0_e_corr_eV_dT            !> corrected T0_e in eV and derivative
real*8     :: T0_i_corr_eV, dT0_i_corr_eV_dT            !> corrected T0_i in eV and derivative
real*8     :: rho0_corr                                 !> corresponding density
real*8     :: ne_SI, lambda_e_bg, nu_e_bg, t_norm       !> parameters for temperature exchange term between ion and electrons
real*8     :: dT0_i_corr_dT, dT0_e_corr_dT              !> derivatives of corrected temperatures  
real*8     :: drho0_corr_dn                             !> derivative of corrected density
real*8     :: dnu_e_bg_dTi, dnu_e_bg_dTe                !> derivative of dnu_e_bg
real*8     :: dnu_e_bg_drho                             !> derivative of dnu_e_bg

real*8, dimension(n_var,4)       :: rhs_ij
real*8, dimension(n_var,n_var,4) :: amat_ij

integer*8  :: plan
real*8     :: in_fft(1:n_plane)
complex*16 :: out_fft(1:n_plane)

#define DIM1 n_plane
#define DIM2 1:n_vertex_max*n_var*(n_order+1)

real*8, dimension(DIM1, DIM2, DIM2) :: ELM_p
real*8, dimension(DIM1, DIM2, DIM2) :: ELM_n
real*8, dimension(DIM1, DIM2, DIM2) :: ELM_k
real*8, dimension(DIM1, DIM2, DIM2) :: ELM_kn
real*8, dimension(DIM1, DIM2)       :: RHS_p
real*8, dimension(DIM1, DIM2)       :: RHS_k
real*8, dimension(DIM1, DIM2, DIM2) :: ELM_pnn

real*8, dimension(n_plane,n_gauss,n_gauss) :: x_g, x_s, x_t, x_p, x_ss, x_st, x_tt, x_sp, x_tp, x_pp
real*8, dimension(n_plane,n_gauss,n_gauss) :: y_g, y_s, y_t, y_p, y_ss, y_st, y_tt, y_sp, y_tp, y_pp
real*8, dimension(n_plane,n_gauss,n_gauss) :: current_source, particle_source
real*8, dimension(n_plane,n_gauss,n_gauss) :: heat_source, heat_source_i, heat_source_e
real*8, dimension(n_gauss, n_gauss)        :: s_norm

real*8, dimension(n_plane,n_var,n_gauss,n_gauss) :: eq_g, eq_s, eq_t, eq_p, eq_ss, eq_st, eq_tt, eq_pp, eq_sp, eq_tp
real*8, dimension(n_plane,n_var,n_gauss,n_gauss) :: delta_g, delta_s, delta_t, delta_p

real*8, dimension(:,:,:,:,:), pointer :: eq
real*8, dimension(n_var)              :: eq_px, eq_py

real*8                                :: phi_source, dPhi_source_dpsi,dummy1,dummy2, dummy3, dummy4, dummy5, dummy6, dPhi_source_dpsi2, s_factor

real*8, dimension(4), parameter  :: vr = (/1.,0.,1.,0./), vp = (/0.,1.,0.,1./), dur = (/1.,1.,0.,0./), dup = (/0.,0.,1.,1./)

eq => thread_eq(tid)%eq

ELM_p = 0.d0; ELM_n = 0.d0; ELM_k = 0.d0; ELM_kn = 0.d0
RHS_p = 0.d0; RHS_k = 0.d0
ELM   = 0.d0; RHS   = 0.d0
rhs_ij  = 0.d0
amat_ij = 0.d0

! --- Take time evolution parameters from phys_module
theta = time_evol_theta
! change zeta for variable dt
zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)

! --- Do we need to use the FFT or non-FFT version?
if ( (i_tor_min == 1) .and. (i_tor_max == n_tor) ) then
  ! In case of global matrix construction:
  use_fft = n_tor > n_tor_fft_thresh
else
  ! In case of "direct construction" of harmonic matrix never FFT:
  use_fft = .false.
end if

if (use_fft) then
  ! In case of FFT, don't loop over toroidal harmonics:
  n_tor_start = 1
  n_tor_end   = 1
else
  n_tor_start = i_tor_min
  n_tor_end   = i_tor_max
end if

n_tor_local = n_tor_end - n_tor_start + 1

!---------------------------------------------------- value of (x,y) and derivatives on Gaussian points
x_g  = 0.d0; x_s   = 0.d0; x_t   = 0.d0; x_p = 0.d0; x_st  = 0.d0; x_ss  = 0.d0; x_tt  = 0.d0; x_sp = 0.d0; x_tp = 0.d0; x_pp = 0.d0;
y_g  = 0.d0; y_s   = 0.d0; y_t   = 0.d0; y_p = 0.d0; y_st  = 0.d0; y_ss  = 0.d0; y_tt  = 0.d0; y_sp = 0.d0; y_tp = 0.d0; y_pp = 0.d0;
eq_g = 0.d0; eq_s  = 0.d0; eq_t  = 0.d0; eq_st = 0.d0; eq_ss = 0.d0; eq_tt = 0.d0;
eq_p = 0.d0; eq_pp = 0.d0; eq_sp = 0.d0; eq_tp = 0.d0

delta_g = 0.d0; delta_s = 0.d0; delta_t = 0.d0; delta_p = 0.d0

eq = 0.d0

s_factor = 1.0 / (n_flux - 1)
s_norm          = 0.d0
current_source  = 0.d0
particle_source = 0.d0
heat_source     = 0.d0
heat_source_i   = 0.d0
heat_source_e   = 0.d0

! --- Integrate
do i=1,n_vertex_max
  do j=1,n_order+1
    do ms=1, n_gauss
      do mt=1, n_gauss
        s_norm(ms, mt) = s_norm(ms, mt) + nodes(i)%r_tor_eq(j)*element%size(i,j)*H(i,j,ms,mt)

        do mp=1,n_plane
          do in=1,n_coord_tor
            x_g(mp,ms,mt)  = x_g(mp,ms,mt)  + nodes(i)%x(in,j,1)*element%size(i,j)*H(i,j,ms,mt)   *HZ_coord(in,mp)
            x_s(mp,ms,mt)  = x_s(mp,ms,mt)  + nodes(i)%x(in,j,1)*element%size(i,j)*H_s(i,j,ms,mt) *HZ_coord(in,mp)
            x_t(mp,ms,mt)  = x_t(mp,ms,mt)  + nodes(i)%x(in,j,1)*element%size(i,j)*H_t(i,j,ms,mt) *HZ_coord(in,mp)
            x_p(mp,ms,mt)  = x_p(mp,ms,mt)  + nodes(i)%x(in,j,1)*element%size(i,j)*H(i,j,ms,mt)   *HZ_coord_p(in,mp)
            x_ss(mp,ms,mt) = x_ss(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_ss(i,j,ms,mt)*HZ_coord(in,mp)
            x_st(mp,ms,mt) = x_st(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_st(i,j,ms,mt)*HZ_coord(in,mp)
            x_tt(mp,ms,mt) = x_tt(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_tt(i,j,ms,mt)*HZ_coord(in,mp)
            x_sp(mp,ms,mt) = x_sp(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_s(i,j,ms,mt) *HZ_coord_p(in,mp)
            x_tp(mp,ms,mt) = x_tp(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_t(i,j,ms,mt) *HZ_coord_p(in,mp)
            x_pp(mp,ms,mt) = x_pp(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H(i,j,ms,mt)   *HZ_coord_pp(in,mp)

            y_g(mp,ms,mt)  = y_g(mp,ms,mt)  + nodes(i)%x(in,j,2)*element%size(i,j)*H(i,j,ms,mt)   *HZ_coord(in,mp)
            y_s(mp,ms,mt)  = y_s(mp,ms,mt)  + nodes(i)%x(in,j,2)*element%size(i,j)*H_s(i,j,ms,mt) *HZ_coord(in,mp)
            y_t(mp,ms,mt)  = y_t(mp,ms,mt)  + nodes(i)%x(in,j,2)*element%size(i,j)*H_t(i,j,ms,mt) *HZ_coord(in,mp)
            y_p(mp,ms,mt)  = y_p(mp,ms,mt)  + nodes(i)%x(in,j,2)*element%size(i,j)*H(i,j,ms,mt)   *HZ_coord_p(in,mp)
            y_ss(mp,ms,mt) = y_ss(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_ss(i,j,ms,mt)*HZ_coord(in,mp)
            y_st(mp,ms,mt) = y_st(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_st(i,j,ms,mt)*HZ_coord(in,mp)
            y_tt(mp,ms,mt) = y_tt(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_tt(i,j,ms,mt)*HZ_coord(in,mp)
            y_sp(mp,ms,mt) = y_sp(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_s(i,j,ms,mt) *HZ_coord_p(in,mp)
            y_tp(mp,ms,mt) = y_tp(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_t(i,j,ms,mt) *HZ_coord_p(in,mp)
            y_pp(mp,ms,mt) = y_pp(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H(i,j,ms,mt)   *HZ_coord_pp(in,mp)
          end do
 
          do k=1,n_var
            do in=1,n_tor
              eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)  * HZ(in,mp)
              eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt)* HZ(in,mp)
              eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt)* HZ(in,mp)
              eq_p(mp,k,ms,mt) = eq_p(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)  * HZ_p(in,mp)
              eq_pp(mp,k,ms,mt) = eq_pp(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)   * HZ_pp(in,mp)
              eq_sp(mp,k,ms,mt) = eq_sp(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt) * HZ_p(in,mp)
              eq_tp(mp,k,ms,mt) = eq_tp(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt) * HZ_p(in,mp)
              eq_ss(mp,k,ms,mt) = eq_ss(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_ss(i,j,ms,mt)* HZ(in,mp)
              eq_st(mp,k,ms,mt) = eq_st(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_st(i,j,ms,mt)* HZ(in,mp)
              eq_tt(mp,k,ms,mt) = eq_tt(mp,k,ms,mt) + nodes(i)%values(in,j,k) * element%size(i,j) * H_tt(i,j,ms,mt)* HZ(in,mp)

              delta_g(mp,k,ms,mt) = delta_g(mp,k,ms,mt) + nodes(i)%deltas(in,j,k) * element%size(i,j) * H(i,j,ms,mt)   * HZ(in,mp)
              delta_s(mp,k,ms,mt) = delta_s(mp,k,ms,mt) + nodes(i)%deltas(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt) * HZ(in,mp)
              delta_t(mp,k,ms,mt) = delta_t(mp,k,ms,mt) + nodes(i)%deltas(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt) * HZ(in,mp)
              delta_p(mp,k,ms,mt) = delta_p(mp,k,ms,mt) + nodes(i)%deltas(in,j,k) * element%size(i,j) * H(i,j,ms,mt)   * HZ_p(in,mp)
            enddo
          enddo

          if (keep_current_prof) then
            do in=1,n_tor
              current_source(mp,ms,mt) = current_source(mp,ms,mt) + nodes(i)%j_source(in,j)*element%size(i,j)*H(i,j,ms,mt)*HZ(in,mp)
            end do
          end if
        enddo
      enddo
    enddo
  enddo
enddo

if (.not. maintain_profiles) then
  do ms=1, n_gauss
    do mt=1, n_gauss
      do mp=1,n_plane
        if (with_TiTe) then
          call sources(xpoint2, xcase2, y_g(mp,ms,mt), Z_xpoint, s_norm(ms,mt),psi_axis,psi_bnd,particle_source(mp,ms,mt),heat_source_i(mp,ms,mt),heat_source_e(mp,ms,mt))
        else
          call sources(xpoint2, xcase2, y_g(mp,ms,mt), Z_xpoint, s_norm(ms,mt),0.0,1.0,particle_source(mp,ms,mt),heat_source(mp,ms,mt))
        end if
      enddo
    enddo
  enddo
endif

if (eta .ne. 0.d0) then
  reta = eta_ohmic/eta
else
  reta = 0.d0
end if

! changes deltas for variable time steps
delta_g = delta_g*tstep/tstep_prev
delta_s = delta_s*tstep/tstep_prev
delta_t = delta_t*tstep/tstep_prev

!--------------------------------------------------- sum over the Gaussian integration points
do ms=1, n_gauss
  do mt=1, n_gauss
    wst = wgauss(ms)*wgauss(mt)

    do mp = 1, n_plane
      phi = 2.d0*pi*float(mp-1)/float(n_plane*n_period)
      xjac    = x_s(mp,ms,mt)*y_t(mp,ms,mt)  - x_t(mp,ms,mt)*y_s(mp,ms,mt)
      xjac_x  = (x_ss(mp,ms,mt)*y_t(mp,ms,mt)**2 - y_ss(mp,ms,mt)*x_t(mp,ms,mt)*y_t(mp,ms,mt) - 2.d0*x_st(mp,ms,mt)*y_s(mp,ms,mt)*y_t(mp,ms,mt) &
              + y_st(mp,ms,mt)*(x_s(mp,ms,mt)*y_t(mp,ms,mt) + x_t(mp,ms,mt)*y_s(mp,ms,mt))                                                      &
              + x_tt(mp,ms,mt)*y_s(mp,ms,mt)**2 - y_tt(mp,ms,mt)*x_s(mp,ms,mt)*y_s(mp,ms,mt)) / xjac
      xjac_y  = (y_tt(mp,ms,mt)*x_s(mp,ms,mt)**2 - x_tt(mp,ms,mt)*y_s(mp,ms,mt)*x_s(mp,ms,mt) - 2.d0*y_st(mp,ms,mt)*x_t(mp,ms,mt)*x_s(mp,ms,mt) &
              + x_st(mp,ms,mt)*(y_t(mp,ms,mt)*x_s(mp,ms,mt) + y_s(mp,ms,mt)*x_t(mp,ms,mt))                                                      &
              + y_ss(mp,ms,mt)*x_t(mp,ms,mt)**2 - x_ss(mp,ms,mt)*y_t(mp,ms,mt)*x_t(mp,ms,mt)) / xjac
      x_p_x = (x_sp(mp,ms,mt)*y_t(mp,ms,mt) - x_tp(mp,ms,mt)*y_s(mp,ms,mt))/xjac
      x_p_y = (x_tp(mp,ms,mt)*x_s(mp,ms,mt) - x_sp(mp,ms,mt)*x_t(mp,ms,mt))/xjac
      y_p_x = (y_sp(mp,ms,mt)*y_t(mp,ms,mt) - y_tp(mp,ms,mt)*y_s(mp,ms,mt))/xjac
      y_p_y = (y_tp(mp,ms,mt)*x_s(mp,ms,mt) - y_sp(mp,ms,mt)*x_t(mp,ms,mt))/xjac
      BigR = x_g(mp,ms,mt)

      ! Values at current time step (u^n)
      eq(1:n_var,0,0,0,1) = eq_g(mp,:,ms,mt)
      eq(1:n_var,1,0,0,1) = (y_t(mp,ms,mt)*eq_s(mp,:,ms,mt) - y_s(mp,ms,mt)*eq_t(mp,:,ms,mt))/xjac
      eq(1:n_var,0,1,0,1) = (-x_t(mp,ms,mt)*eq_s(mp,:,ms,mt) + x_s(mp,ms,mt)*eq_t(mp,:,ms,mt))/xjac
      eq(1:n_var,0,0,1,1) = eq_p(mp,:,ms,mt) - eq(1:n_var,1,0,0,1)*x_p(mp,ms,mt) - eq(1:n_var,0,1,0,1)*y_p(mp,ms,mt)
      eq(1:n_var,2,0,0,1) = (eq_ss(mp,:,ms,mt)*y_t(mp,ms,mt)**2 - 2.d0*eq_st(mp,:,ms,mt)*y_s(mp,ms,mt)*y_t(mp,ms,mt) &
                          + eq_tt(mp,:,ms,mt)*y_s(mp,ms,mt)**2                                                       &
                          + eq_s(mp,:,ms,mt)*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt))           &
                          + eq_t(mp,:,ms,mt)*(y_st(mp,ms,mt)*y_s(mp,ms,mt) - y_ss(mp,ms,mt)*y_t(mp,ms,mt)))/xjac**2  &
                          - xjac_x*(eq_s(mp,:,ms,mt)*y_t(mp,ms,mt) - eq_t(mp,:,ms,mt)*y_s(mp,ms,mt))/xjac**2
      eq(1:n_var,0,2,0,1) = (eq_ss(mp,:,ms,mt)*x_t(mp,ms,mt)**2 - 2.d0*eq_st(mp,:,ms,mt)*x_s(mp,ms,mt)*x_t(mp,ms,mt) &
                          + eq_tt(mp,:,ms,mt)*x_s(mp,ms,mt)**2                                                       &
                          + eq_s(mp,:,ms,mt)*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt))           &
                          + eq_t(mp,:,ms,mt)*(x_st(mp,ms,mt)*x_s(mp,ms,mt) - x_ss(mp,ms,mt)*x_t(mp,ms,mt)))/xjac**2  &
                          - xjac_y*(-eq_s(mp,:,ms,mt)*x_t(mp,ms,mt) + eq_t(mp,:,ms,mt)*x_s(mp,ms,mt))/xjac**2
      eq(1:n_var,1,1,0,1) = (-eq_ss(mp,:,ms,mt)*y_t(mp,ms,mt)*x_t(mp,ms,mt) - eq_tt(mp,:,ms,mt)*x_s(mp,ms,mt)*y_s(mp,ms,mt) &
                          + eq_st(mp,:,ms,mt)*(y_s(mp,ms,mt)*x_t(mp,ms,mt) + y_t(mp,ms,mt)*x_s(mp,ms,mt))                   &
                          - eq_s(mp,:,ms,mt)*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt))                  &
                          - eq_t(mp,:,ms,mt)*(x_st(mp,ms,mt)*y_s(mp,ms,mt) - x_ss(mp,ms,mt)*y_t(mp,ms,mt)))/xjac**2         &
                          - xjac_x*(-eq_s(mp,:,ms,mt)*x_t(mp,ms,mt) + eq_t(mp,:,ms,mt)*x_s(mp,ms,mt))/xjac**2
      eq_px               = (y_t(mp,ms,mt)*eq_sp(mp,:,ms,mt) - y_s(mp,ms,mt)*eq_tp(mp,:,ms,mt))/xjac
      eq_py               = (-x_t(mp,ms,mt)*eq_sp(mp,:,ms,mt) + x_s(mp,ms,mt)*eq_tp(mp,:,ms,mt))/xjac
      ! Second derivatives wrt phi not implemented in FFT, and not necessary for stabilization (2nd derivatives only appear in hyperdissipation terms)
!      eq(1:n_var,0,0,2,1) = eq_pp(mp,:,ms,mt) - x_pp(mp,ms,mt)*eq(1:n_var,1,0,0,1) - 2.d0*(x_p(mp,ms,mt)*eq_px + y_p(mp,ms,mt)*eq_py)            &
!                          - y_pp(mp,ms,mt)*eq(1:n_var,0,1,0,1) + 2.d0*(x_p(mp,ms,mt)*x_p_x*eq(1:n_var,1,0,0,1) + x_p(mp,ms,mt)*y_p_x*eq(1:n_var,0,1,0,1)&
!                          + y_p(mp,ms,mt)*x_p_y*eq(1:n_var,1,0,0,1) + y_p(mp,ms,mt)*y_p_y*eq(1:n_var,0,1,0,1)) + x_p(mp,ms,mt)**2*eq(1:n_var,2,0,0,1)   &
!                          + 2.d0*x_p(mp,ms,mt)*y_p(mp,ms,mt)*eq(1:n_var,1,1,0,1) + y_p(mp,ms,mt)**2*eq(1:n_var,0,2,0,1)
      eq(1:n_var,1,0,1,1) = eq_px - x_p_x*eq(1:n_var,1,0,0,1) - x_p(mp,ms,mt)*eq(1:n_var,2,0,0,1) - y_p_x*eq(1:n_var,0,1,0,1) &
                          - y_p(mp,ms,mt)*eq(1:n_var,1,1,0,1)
      eq(1:n_var,0,1,1,1) = eq_py - x_p_y*eq(1:n_var,1,0,0,1) - x_p(mp,ms,mt)*eq(1:n_var,1,1,0,1) - y_p_y*eq(1:n_var,0,1,0,1) &
                          - y_p(mp,ms,mt)*eq(1:n_var,0,2,0,1)

      ! Increments since previous time step (delta_u^(n-1))
      eq(n_var+1:2*n_var,0,0,0,1) = delta_g(mp,:,ms,mt)
      eq(n_var+1:2*n_var,1,0,0,1) = (y_t(mp,ms,mt)*delta_s(mp,:,ms,mt) - y_s(mp,ms,mt)*delta_t(mp,:,ms,mt))/xjac
      eq(n_var+1:2*n_var,0,1,0,1) = (-x_t(mp,ms,mt)*delta_s(mp,:,ms,mt) + x_s(mp,ms,mt)*delta_t(mp,:,ms,mt))/xjac
      eq(n_var+1:2*n_var,0,0,1,1) = delta_p(mp,:,ms,mt) - eq(n_var+1:2*n_var,1,0,0,1)*x_p(mp,ms,mt) - eq(n_var+1:2*n_var,0,1,0,1)*y_p(mp,ms,mt)

      ! Vacuum scalar magnetic potential (chi) and field (grad chi)
      eq(var_chi,:,:,:,1) = element%chi(mp,ms,mt,:,:,:)
      
      ! Duplicate t^n and t^(n-1) terms in FFT matrix contributions
      do m=2,4
        eq(1:n_var,:,:,:,m) = eq(1:n_var,:,:,:,1)
        eq(n_var+1:2*n_var,:,:,:,m) = eq(n_var+1:2*n_var,:,:,:,1)
        eq(var_chi,:,:,:,m) = eq(var_chi,:,:,:,1)
      end do
      
      eq(var_R,0,0,0,:) = x_g(mp,ms,mt); eq(var_R,1,0,0,:) = 1.d0 ! Cylindrical R coordinate

      psi_norm = s_norm(ms,mt)
      
      ! The Psi in the equations differs by a factor of F0 from the normal JOREK Psi
      eq( var_Psi,:,:,:,:) = eq( var_Psi,:,:,:,:)/F0
      eq(var_dPsi,:,:,:,:) = eq(var_dPsi,:,:,:,:)/F0
      eq(  var_zj,:,:,:,:) = eq(  var_zj,:,:,:,:)/F0
      eq( var_dzj,:,:,:,:) = eq( var_dzj,:,:,:,:)/F0

      eq(var_D_perp,0,0,0,:) = get_dperp(psi_norm)         ! D_perp
      eq( var_D_par,0,0,0,:) = D_par                       ! D_par

      eq(var_S_rho,0,0,0,:) = particle_source(mp,ms,mt)   ! S_rho
      eq(  var_S_j,0,0,0,:) = current_source(mp,ms,mt)/F0 ! S_j

      ! Poloidal momentum source based on artificial ExB flow - note only first order derivatives are implemented
      call potential_source(xpoint2, xcase2, y_g(mp,ms,mt), Z_xpoint, psi_norm, 0.0, 1.0, phi_source, dPhi_source_dpsi,dummy1,dPhi_source_dpsi2,dummy2,dummy3,dummy4,dummy5,dummy6)
      dphi_source_dpsi = dphi_source_dpsi*s_factor ! convert to local derivative within element
      eq(var_S_phi_pol,0,0,0,:) = phi_source
      eq(var_S_phi_pol,1,0,0,:) = (y_t(mp,ms,mt)*dphi_source_dpsi)/xjac
      eq(var_S_phi_pol,0,1,0,:) = (-x_t(mp,ms,mt)*dphi_source_dpsi)/xjac
      eq(var_S_phi_pol,0,0,1,:) = -eq(var_S_Phi_pol,1,0,0,:)*x_p(mp,ms,mt) - eq(var_S_Phi_pol,0,1,0,:)*y_p(mp,ms,mt)
      ! Poloidal momentum source needs to be compared with Phi excluding t derivatives - note this assumes a flux surface aligned grid
      eq(var_Phi_pol,0,0,0,1) = eq_g(mp,var_Phi,ms,mt)
      eq(var_Phi_pol,1,0,0,1) = (y_t(mp,ms,mt)*eq_s(mp,var_Phi,ms,mt))/xjac
      eq(var_Phi_pol,0,1,0,1) = (-x_t(mp,ms,mt)*eq_s(mp,var_Phi,ms,mt))/xjac
      eq(var_Phi_pol,0,0,1,1) = eq_p(mp,var_Phi,ms,mt) - eq(var_Phi_pol,1,0,0,1)*x_p(mp,ms,mt) - eq(var_Phi_pol,0,1,0,1)*y_p(mp,ms,mt)
      
      ! Equilibrium temperature and density
      if (maintain_profiles) then
        call density(xpoint2, xcase2, y_g(mp,ms,mt), Z_xpoint, psi_norm, 0.0, 1.0, rho0, rho0_s,dummy1,rho0_ss,dummy2,dummy3,dummy4,dummy5,dummy6)
        rho0_s = rho0_s*s_factor; rho0_ss = rho0_ss*s_factor**2    ! convert to local derivative within element
        eq(var_rho_equil,0,0,0,:) = rho0
        eq(var_rho_equil,1,0,0,:) = (y_t(mp,ms,mt)*rho0_s)/xjac
        eq(var_rho_equil,0,1,0,:) = (-x_t(mp,ms,mt)*rho0_s)/xjac
        eq(var_rho_equil,0,0,1,:) = -eq(var_rho_equil,1,0,0,:)*x_p(mp,ms,mt) - eq(var_rho_equil,0,1,0,:)*y_p(mp,ms,mt)
        eq(var_rho_equil,2,0,0,1) = (rho0_ss*y_t(mp,ms,mt)**2 + rho0_s*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2  &
                                  - xjac_x*(rho0_s*y_t(mp,ms,mt))/xjac**2
        eq(var_rho_equil,0,2,0,1) = (rho0_ss*x_t(mp,ms,mt)**2 + rho0_s*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt)))/xjac**2  &
                                  - xjac_y*(-rho0_s*x_t(mp,ms,mt))/xjac**2
        eq(var_rho_equil,1,1,0,1) = (-rho0_ss*y_t(mp,ms,mt)*x_t(mp,ms,mt) - rho0_s*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2         &
                                  - xjac_x*(-rho0_s*x_t(mp,ms,mt))/xjac**2
        ! Second derivatives wrt phi not implemented in FFT, and not necessary for stabilization (2nd derivatives only appear in hyperdissipation terms)
        eq(var_rho_equil,1,0,1,1) = -x_p_x*eq(var_rho_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_rho_equil,2,0,0,1) - y_p_x*eq(var_rho_equil,0,1,0,1) &
                                  - y_p(mp,ms,mt)*eq(var_rho_equil,1,1,0,1)
        eq(var_rho_equil,0,1,1,1) = -x_p_y*eq(var_rho_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_rho_equil,1,1,0,1) - y_p_y*eq(var_rho_equil,0,1,0,1) &
                                  - y_p(mp,ms,mt)*eq(var_rho_equil,0,2,0,1)

        if (with_TiTe) then
          call temperature_i(xpoint2, xcase2, y_g(mp,ms,mt), Z_xpoint, psi_norm, 0.0, 1.0, T0_i, T0_i_s,dummy1,T0_i_ss,dummy2,dummy3,dummy4,dummy5,dummy6)
          T0_i_s = T0_i_s*s_factor; T0_i_ss = T0_i_ss*s_factor**2   ! convert to local derivative within element
          eq(var_Ti_equil,0,0,0,:) = T0_i
          eq(var_Ti_equil,1,0,0,:) = (y_t(mp,ms,mt)*T0_i_s)/xjac
          eq(var_Ti_equil,0,1,0,:) = (-x_t(mp,ms,mt)*T0_i_s)/xjac
          eq(var_Ti_equil,0,0,1,:) = -eq(var_Ti_equil,1,0,0,:)*x_p(mp,ms,mt) - eq(var_Ti_equil,0,1,0,:)*y_p(mp,ms,mt)
          eq(var_Ti_equil,2,0,0,1) = (T0_i_ss*y_t(mp,ms,mt)**2 + T0_i_s*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2  &
                                    - xjac_x*(T0_i_s*y_t(mp,ms,mt))/xjac**2
          eq(var_Ti_equil,0,2,0,1) = (T0_i_ss*x_t(mp,ms,mt)**2 + T0_i_s*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt)))/xjac**2  &
                                    - xjac_y*(-T0_i_s*x_t(mp,ms,mt))/xjac**2
          eq(var_Ti_equil,1,1,0,1) = (-T0_i_ss*y_t(mp,ms,mt)*x_t(mp,ms,mt) - T0_i_s*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2         &
                                    - xjac_x*(-T0_i_s*x_t(mp,ms,mt))/xjac**2
          ! Second derivatives wrt phi not implemented in FFT, and not necessary for stabilization (2nd derivatives only appear in hyperdissipation terms)
          eq(var_Ti_equil,1,0,1,1) = -x_p_x*eq(var_Ti_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_Ti_equil,2,0,0,1) - y_p_x*eq(var_Ti_equil,0,1,0,1) &
                                    - y_p(mp,ms,mt)*eq(var_Ti_equil,1,1,0,1)
          eq(var_Ti_equil,0,1,1,1) = -x_p_y*eq(var_Ti_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_Ti_equil,1,1,0,1) - y_p_y*eq(var_Ti_equil,0,1,0,1) &
                                    - y_p(mp,ms,mt)*eq(var_Ti_equil,0,2,0,1)

          call temperature_e(xpoint2, xcase2, y_g(mp,ms,mt), Z_xpoint, psi_norm, 0.0, 1.0, T0_e, T0_e_s,dummy1,T0_e_ss,dummy2,dummy3,dummy4,dummy5,dummy6)
          T0_e_s = T0_e_s*s_factor; T0_e_ss = T0_e_ss*s_factor**2   ! convert to local derivative within element
          eq(var_Te_equil,0,0,0,:) = T0_e
          eq(var_Te_equil,1,0,0,:) = (y_t(mp,ms,mt)*T0_e_s)/xjac
          eq(var_Te_equil,0,1,0,:) = (-x_t(mp,ms,mt)*T0_e_s)/xjac
          eq(var_Te_equil,0,0,1,:) = -eq(var_Te_equil,1,0,0,:)*x_p(mp,ms,mt) - eq(var_Te_equil,0,1,0,:)*y_p(mp,ms,mt)
          eq(var_Te_equil,2,0,0,1) = (T0_e_ss*y_t(mp,ms,mt)**2 + T0_e_s*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2  &
                                    - xjac_x*(T0_e_s*y_t(mp,ms,mt))/xjac**2
          eq(var_Te_equil,0,2,0,1) = (T0_e_ss*x_t(mp,ms,mt)**2 + T0_e_s*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt)))/xjac**2  &
                                    - xjac_y*(-T0_e_s*x_t(mp,ms,mt))/xjac**2
          eq(var_Te_equil,1,1,0,1) = (-T0_e_ss*y_t(mp,ms,mt)*x_t(mp,ms,mt) - T0_e_s*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2         &
                                    - xjac_x*(-T0_e_s*x_t(mp,ms,mt))/xjac**2
          ! Second derivatives wrt phi not implemented in FFT, and not necessary for stabilization (2nd derivatives only appear in hyperdissipation terms)
          eq(var_Te_equil,1,0,1,1) = -x_p_x*eq(var_Te_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_Te_equil,2,0,0,1) - y_p_x*eq(var_Te_equil,0,1,0,1) &
                                    - y_p(mp,ms,mt)*eq(var_Te_equil,1,1,0,1)
          eq(var_Te_equil,0,1,1,1) = -x_p_y*eq(var_Te_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_Te_equil,1,1,0,1) - y_p_y*eq(var_Te_equil,0,1,0,1) &
                                    - y_p(mp,ms,mt)*eq(var_Te_equil,0,2,0,1)
        else
          call temperature(xpoint2, xcase2, y_g(mp,ms,mt), Z_xpoint, psi_norm, 0.0, 1.0, T0, T0_s,dummy1,T0_ss,dummy2,dummy3,dummy4,dummy5,dummy6)
          T0_s = T0_s*s_factor; T0_ss = T0_ss*s_factor**2   ! convert to local derivative within element
          eq(var_T_equil,0,0,0,:) = T0
          eq(var_T_equil,1,0,0,:) = (y_t(mp,ms,mt)*T0_s)/xjac
          eq(var_T_equil,0,1,0,:) = (-x_t(mp,ms,mt)*T0_s)/xjac
          eq(var_T_equil,0,0,1,:) = -eq(var_T_equil,1,0,0,:)*x_p(mp,ms,mt) - eq(var_T_equil,0,1,0,:)*y_p(mp,ms,mt)
          eq(var_T_equil,2,0,0,1) = (T0_ss*y_t(mp,ms,mt)**2 + T0_s*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2  &
                                    - xjac_x*(T0_s*y_t(mp,ms,mt))/xjac**2
          eq(var_T_equil,0,2,0,1) = (T0_ss*x_t(mp,ms,mt)**2 + T0_s*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt)))/xjac**2  &
                                    - xjac_y*(-T0_s*x_t(mp,ms,mt))/xjac**2
          eq(var_T_equil,1,1,0,1) = (-T0_ss*y_t(mp,ms,mt)*x_t(mp,ms,mt) - T0_s*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt)))/xjac**2         &
                                    - xjac_x*(-T0_s*x_t(mp,ms,mt))/xjac**2
          ! Second derivatives wrt phi not implemented in FFT, and not necessary for stabilization (2nd derivatives only appear in hyperdissipation terms)
          eq(var_T_equil,1,0,1,1) = -x_p_x*eq(var_T_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_T_equil,2,0,0,1) - y_p_x*eq(var_T_equil,0,1,0,1) &
                                    - y_p(mp,ms,mt)*eq(var_T_equil,1,1,0,1)
          eq(var_T_equil,0,1,1,1) = -x_p_y*eq(var_T_equil,1,0,0,1) - x_p(mp,ms,mt)*eq(var_T_equil,1,1,0,1) - y_p_y*eq(var_T_equil,0,1,0,1) &
                                    - y_p(mp,ms,mt)*eq(var_T_equil,0,2,0,1)
        end if
      endif

      if (with_TiTe) then
        
        ! Perpendicular thermal conductivity
        eq(var_k_perp_i,0,0,0,:)        = get_zk_iperp(psi_norm)  ! k_i_perp
        eq(var_k_perp_e,0,0,0,:)        = get_zk_eperp(psi_norm)  ! k_e_perp

        ! Heat sources for ions and electrons
        eq(var_S_e_i,0,0,0,:) = heat_source_i(mp,ms,mt)        ! S_e_i
        eq(var_S_e_e,0,0,0,:) = heat_source_e(mp,ms,mt)        ! S_e_e
        
        ! Resistivity
        if (eta_T_dependent) then                                                        
          eq(    var_eta,0,0,0,:) = eta*(corr_neg_temp(eq(var_Te,0,0,0,1))/Te_0)**(-1.5d0)               ! eta 
          eq(var_deta_dT,0,0,0,:) = -1.5d0*eta*corr_neg_temp(eq(var_Te,0,0,0,1))**(-2.5d0)*Te_0**(1.5d0) ! deta/dT 
        else
          eq(    var_eta,0,0,0,:) = eta
          eq(var_deta_dT,0,0,0,:) = 0.d0
        end if

        ! Perpendicular viscosity
        if (visco_T_dependent) then  
          eq(var_visco    ,0,0,0,:) = visco*(corr_neg_temp(eq(var_Te,0,0,0,1))/Te_0)**(-1.5d0)               ! visco
          eq(var_dvisco_dT,0,0,0,:) = -1.5d0*visco*corr_neg_temp(eq(var_Te,0,0,0,1))**(-2.5d0)*Te_0**(1.5d0) ! dvisco/dT
        else
          eq(var_visco    ,0,0,0,:) = visco
          eq(var_dvisco_dT,0,0,0,:) = 0.d0
        end if

        ! Parallel thermal conductivity
        if (zkpar_T_dependent) then                                                                 
          eq(    var_k_par_i,0,0,0,:) = ZK_i_par*(corr_neg_temp(eq(var_Ti,0,0,0,1))/Ti_0)**(2.5d0)                       ! k_par for ions
          eq(var_dk_par_dT_i,0,0,0,:) = 2.5d0*ZK_i_par*corr_neg_temp(eq(var_Ti,0,0,0,1))**(1.5d0)*Ti_0**(-2.5d0)         ! dk_par_i_dT_i
          eq(    var_k_par_e,0,0,0,:) = ZK_e_par*(corr_neg_temp(eq(var_Te,0,0,0,1))/Te_0)**(2.5d0)                       ! k_par for e
          eq(var_dk_par_dT_e,0,0,0,:) = 2.5d0*ZK_e_par*corr_neg_temp(eq(var_Te,0,0,0,1))**(1.5d0)*Te_0**(-2.5d0)         ! dk_par_e_dT_e
          if (eq(var_k_par_i,0,0,0,1) .gt. zk_par_max) then
            eq(    var_k_par_i,0,0,0,:) = zk_par_max
            eq(var_dk_par_dT_i,0,0,0,:) = 0.d0
          end if
          if (eq(var_k_par_e,0,0,0,1) .gt. zk_par_max) then
            eq(    var_k_par_e,0,0,0,:) = zk_par_max
            eq(var_dk_par_dT_e,0,0,0,:) = 0.d0
          end if
        else
          eq(    var_k_par_i,0,0,0,:) = zk_i_par
          eq(var_dk_par_dT_i,0,0,0,:) = 0.d0
          eq(    var_k_par_e,0,0,0,:) = zk_e_par
          eq(var_dk_par_dT_e,0,0,0,:) = 0.d0
        end if
         
        ! Thermalization in Temperature evolution

        ! see corr_neg page in the jorek wiki, short explanation in models/corr_neg_include.f90 
        T0_i_corr    = corr_neg_temp(eq(var_Ti,0,0,0,1))  ! corrected ion temperature
        T0_e_corr    = corr_neg_temp(eq(var_Te,0,0,0,1))  ! corrected electron temperature
        rho0_corr    = corr_neg_dens(eq(var_rho,0,0,0,1))  ! corrected density

        dT0_i_corr_dT= dcorr_neg_temp_dT(eq(var_Ti,0,0,0,1))   ! derivative of corrected ion temperature
        dT0_e_corr_dT= dcorr_neg_temp_dT(eq(var_Te,0,0,0,1))   ! derivative of corrected electron temperature
        drho0_corr_dn= dcorr_neg_dens_drho(eq(var_rho,0,0,0,1)) ! dericative of corrected density

        T0_e_corr_eV = T0_e_corr/(EL_CHG*MU_ZERO*central_density*1.d20) ! electron temperature in eV
        T0_i_corr_eV = T0_i_corr/(EL_CHG*MU_ZERO*central_density*1.d20) ! ion temperature in eV
        dT0_e_corr_eV_dT = dT0_e_corr_dT/(EL_CHG*MU_ZERO*central_density*1.d20)
        dT0_i_corr_eV_dT = dT0_i_corr_dT/(EL_CHG*MU_ZERO*central_density*1.d20)

        ne_SI = rho0_corr * 1.d20 * central_density  ! density in SI units
        if (ne_SI < 1.d16) ne_SI = 1.d16   ! prevent absurd number in the coulomb lambda

        ! the equations for collision frequency nu_e_bg and Coulomb lambda were taken from NRL plasmaformulary 2013 p.34
        ! equations were written down in cgs in the source and were modfied to match SI units below
        if ((T0_i_corr*MASS_ELECTRON/(central_mass*ATOMIC_MASS_UNIT)<T0_e_corr) .and. (T0_e_corr_eV<10)) then 
          lambda_e_bg  = 23.d0 - log((ne_SI*1.d-6)**0.5*T0_e_corr_eV**(-1.5)) ! Assuming bg_charge is 1! --> Coulomb lambda
        else if ((T0_i_corr_eV*MASS_ELECTRON/(central_mass*ATOMIC_MASS_UNIT) < 10) .and. (10<T0_e_corr_eV)) then
          lambda_e_bg  = 24.d0 - log((ne_SI*1.d-6)**0.5*T0_e_corr_eV**(-1.0))
        else
          lambda_e_bg  = 30.d0 - log((ne_SI*1.d-6)**0.5*T0_i_corr_eV**(-1.5)/MU_ZERO)
        end if
        nu_e_bg      = 1.8d-19*(1.d6*MASS_ELECTRON*ATOMIC_MASS_UNIT*central_mass) ** 0.5&
                     * (1.d14*central_density*rho0_corr) * lambda_e_bg &
                     / (1.d3*(MASS_ELECTRON*T0_i_corr+T0_e_corr*ATOMIC_MASS_UNIT*central_mass)&
                     / (EL_CHG * MU_ZERO * central_density * 1.d20)) ** 1.5 ! Assuming bg_charge is 1! --> collision frequency
        if (nu_e_bg < 0.) nu_e_bg = 0.
      
        !Converting the energy transfer rate from s^-1 to JOREK unit
        t_norm   = sqrt(MU_ZERO * central_mass * ATOMIC_MASS_UNIT * central_density * 1.d20) ! normalization coefficient
        nu_e_bg  = nu_e_bg * t_norm                                                     ! normalised collision frequency

        eq(var_dTe_i,0,0,0,:) = nu_e_bg * (T0_e_corr - T0_i_corr) * rho0_corr          ! dTe_i - temperature exchange term

        ! --- derivatives of dTe_i for amats (of temperatures and density)
        ! --- We negelect the coulomb log's derivatives due to their smallness

        dnu_e_bg_dTi    = -1.5*MASS_ELECTRON*nu_e_bg*dT0_i_corr_dT / (MASS_ELECTRON*T0_i_corr + ATOMIC_MASS_UNIT*central_mass*T0_e_corr)
        dnu_e_bg_dTe    = -1.5*ATOMIC_MASS_UNIT*central_mass*nu_e_bg*dT0_e_corr_dT &
                        / (MASS_ELECTRON*T0_i_corr + ATOMIC_MASS_UNIT*central_mass*T0_e_corr)

        dnu_e_bg_drho   = nu_e_bg * drho0_corr_dn / rho0_corr

        eq(var_ddTe_i_dT_i,0,0,0,:)  = dnu_e_bg_dTi * (T0_e_corr - T0_i_corr) * rho0_corr - nu_e_bg * dT0_i_corr_dT * rho0_corr
        eq(var_ddTe_i_dT_e,0,0,0,:)  = dnu_e_bg_dTe * (T0_e_corr - T0_i_corr) * rho0_corr + nu_e_bg * dT0_e_corr_dT * rho0_corr
        eq(var_ddTe_i_drho,0,0,0,:)  = dnu_e_bg_drho * (T0_e_corr - T0_i_corr) * rho0_corr &
                                + nu_e_bg * (T0_e_corr - T0_i_corr) * drho0_corr_dn

      else !>>> if not with_TiTe
        eq(var_k_perp,0,0,0,:)        = get_zkperp(psi_norm)        ! Perpendicular thermal conductivity
        eq(   var_S_e,0,0,0,:)        = heat_source(mp,ms,mt)       ! S_e
        ! Resistivity
        if (eta_T_dependent) then
          eq(    var_eta,0,0,0,:) = eta*(corr_neg_temp(eq(var_T,0,0,0,1))/T_0)**(-1.5d0)               ! eta 
          eq(var_deta_dT,0,0,0,:) = -1.5d0*eta*corr_neg_temp(eq(var_T,0,0,0,1))**(-2.5d0)*T_0**(1.5d0) ! deta/dT 
        else
          eq(    var_eta,0,0,0,:) = eta
          eq(var_deta_dT,0,0,0,:) = 0.d0
        end if
        ! Viscosity
        if (visco_T_dependent) then
          eq(    var_visco,0,0,0,:) = visco*(corr_neg_temp(eq(var_T,0,0,0,1))/T_0)**(-1.5d0)               ! visco 
          eq(var_dvisco_dT,0,0,0,:) = -1.5d0*visco*corr_neg_temp(eq(var_T,0,0,0,1))**(-2.5d0)*T_0**(1.5d0) ! dvisco/dT
        else
          eq(    var_visco,0,0,0,:) = visco
          eq(var_dvisco_dT,0,0,0,:) = 0.d0
        end if
        ! Parallel thermal conductivity
        if (zkpar_T_dependent) then
          eq(    var_k_par,0,0,0,:) = zk_par*(corr_neg_temp(eq(var_T,0,0,0,1))/T_0)**(2.5d0)               ! k_par 
          eq(var_dk_par_dT,0,0,0,:) = 2.5d0*zk_par*corr_neg_temp(eq(var_T,0,0,0,1))**(1.5d0)*T_0**(-2.5d0) ! dk_par_dT 
          if (eq(var_k_par,0,0,0,1) .gt. zk_par_max) then
            eq(    var_k_par,0,0,0,:) = zk_par_max
            eq(var_dk_par_dT,0,0,0,:) = 0.d0
          end if
        else
          eq(    var_k_par,0,0,0,:) = zk_par
          eq(var_dk_par_dT,0,0,0,:) = 0.d0
        end if
      end if
      
      ! --- Increase diffusivity if very small density or temperature
      if (eq(var_rho,0,0,0,1) .lt. D_prof_neg_thresh) then
        eq(var_D_perp,0,0,0,:) = D_prof_neg       
      endif
      if ( with_TiTe ) then ! (with_TiTe) ****************************************************
        if (eq(var_Ti,0,0,0,1) .lt. ZK_prof_neg_thresh) then
          eq(var_k_perp_i,0,0,0,:) = ZK_prof_neg       
        endif
        if (eq(var_Te,0,0,0,1) .lt. ZK_prof_neg_thresh) then
          eq(var_k_perp_e,0,0,0,:) = ZK_prof_neg       
        endif
      else ! (with_TiTe), i.e. with single temperature ***************************************
        if (eq(var_T,0,0,0,1) .lt. ZK_prof_neg_thresh) then
          eq(var_k_perp,0,0,0,:) = ZK_prof_neg       
        endif
      endif ! (with_TiTe) ********************************************************************

      ! Auxiliary variables (aux)
      call get_auxiliary(eq)

      do i=1,n_vertex_max
        do j=1,n_order+1
          do im=n_tor_start,n_tor_end
            if (use_fft) then
              index_ij =       n_var*(n_order+1)*(i-1) +       n_var*(j-1) + 1
            else
              index_ij = n_tor_local*n_var*(n_order+1)*(i-1) + n_tor_local*n_var*(j-1) + im - n_tor_start +1 
            end if
            
            ! Test function (v)
            eq(var_v,0,0,0,:) =  H(i,j,ms,mt)*element%size(i,j)*vr
            eq(var_v,1,0,0,:) = (y_t(mp,ms,mt)*h_s(i,j,ms,mt) - y_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)*vr/xjac
            eq(var_v,0,1,0,:) = (-x_t(mp,ms,mt)*h_s(i,j,ms,mt) + x_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)*vr/xjac
            eq(var_v,0,0,1,:) = H(i,j,ms,mt)*element%size(i,j)*vp - eq(var_v,1,0,0,:)*x_p(mp,ms,mt) - eq(var_v,0,1,0,:)*y_p(mp,ms,mt)
            eq(var_v,2,0,0,:) = (h_ss(i,j,ms,mt)*y_t(mp,ms,mt)**2 - 2.d0*h_st(i,j,ms,mt)*y_s(mp,ms,mt)*y_t(mp,ms,mt)                       &
	                              + h_tt(i,j,ms,mt)*y_s(mp,ms,mt)**2                                                                           &
	                              + h_s(i,j,ms,mt)*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt))                               &
                                  + h_t(i,j,ms,mt)*(y_st(mp,ms,mt)*y_s(mp,ms,mt) - y_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(i,j)*vr/xjac**2 &
                                  - xjac_x*(h_s(i,j,ms,mt)*y_t(mp,ms,mt) - h_t(i,j,ms,mt)*y_s(mp,ms,mt))*element%size(i,j)*vr/xjac**2
            eq(var_v,0,2,0,:) = (h_ss(i,j,ms,mt)*x_t(mp,ms,mt)**2 - 2.d0*h_st(i,j,ms,mt)*x_s(mp,ms,mt)*x_t(mp,ms,mt)                    &
                                  + h_tt(i,j,ms,mt)*x_s(mp,ms,mt)**2                                                                        &
                                  + h_s(i,j,ms,mt)*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt))                            &
                                  + h_t(i,j,ms,mt)*(x_st(mp,ms,mt)*x_s(mp,ms,mt) - x_ss(mp,ms,mt)*x_t(mp,ms,mt)))*element%size(i,j)*vr/xjac**2 &
           	                      - xjac_y*(-h_s(i,j,ms,mt)*x_t(mp,ms,mt) + h_t(i,j,ms,mt)*x_s(mp,ms,mt))*element%size(i,j)*vr/xjac**2
            eq(var_v,1,1,0,:) = (-h_ss(i,j,ms,mt)*y_t(mp,ms,mt)*x_t(mp,ms,mt) - h_tt(i,j,ms,mt)*x_s(mp,ms,mt)*y_s(mp,ms,mt)                &
                                  + h_st(i,j,ms,mt)*(y_s(mp,ms,mt)*x_t(mp,ms,mt) + y_t(mp,ms,mt)*x_s(mp,ms,mt))                                &
                                  - h_s(i,j,ms,mt)*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt))                               &
                                  - h_t(i,j,ms,mt)*(x_st(mp,ms,mt)*y_s(mp,ms,mt) - x_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(i,j)*vr/xjac**2 &
                                  - xjac_x*(-h_s(i,j,ms,mt)*x_t(mp,ms,mt) + h_t(i,j,ms,mt)*x_s(mp,ms,mt))*element%size(i,j)*vr/xjac**2
            v_px                  = (y_t(mp,ms,mt)*h_s(i,j,ms,mt) - y_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)/xjac
            v_py                  = (-x_t(mp,ms,mt)*h_s(i,j,ms,mt) + x_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)/xjac
            ! Second derivatives wrt phi not implemented in FFT,
            ! and not necessary for stabilization (2nd derivatives only appear in hyperdissipation terms)
!            eq(var_v,0,0,2,:) = -H(i,j,ms,mt)*element%size(i,j)*vr - x_pp(mp,ms,mt)*eq(var_v,1,0,0,:) - 2.d0*(x_p(mp,ms,mt)*v_px &
!                                  + y_p(mp,ms,mt)*v_py)*vp - y_pp(mp,ms,mt)*eq(var_v,0,1,0,:) + 2.d0*(x_p(mp,ms,mt)*x_p_x*eq(var_v,1,0,0,:) &
!                                  + x_p(mp,ms,mt)*y_p_x*eq(var_v,0,1,0,:) + y_p(mp,ms,mt)*x_p_y*eq(var_v,1,0,0,:) &
!                                  + y_p(mp,ms,mt)*y_p_y*eq(var_v,0,1,0,:)) + x_p(mp,ms,mt)**2*eq(var_v,2,0,0,:) &
!                                  + 2.d0*x_p(mp,ms,mt)*y_p(mp,ms,mt)*eq(var_v,1,1,0,:) + y_p(mp,ms,mt)**2*eq(var_v,0,2,0,:)
            eq(var_v,1,0,1,:) = v_px*vp - x_p_x*eq(var_v,1,0,0,:) - x_p(mp,ms,mt)*eq(var_v,2,0,0,:) - y_p_x*eq(var_v,0,1,0,:) &
                                  - y_p(mp,ms,mt)*eq(var_v,1,1,0,:)
            eq(var_v,0,1,1,:) = v_py*vp - x_p_y*eq(var_v,1,0,0,:) - y_p(mp,ms,mt)*eq(var_v,0,2,0,:) - y_p_y*eq(var_v,0,1,0,:) &
                                  - x_p(mp,ms,mt)*eq(var_v,1,1,0,:)
            
            
          call get_rhs(rhs_ij, eq)
           
           ! Add Jacobian pre-factor to contributions
           do i_var=1,n_var
             rhs_ij(i_var,:) = rhs_ij(i_var,:)*wst*BigR*xjac
           enddo

           ! Fill up RHS
           do i_var = 1, n_var
             if (use_fft) then
                RHS_p(mp, index_ij+(i_var-1)*(n_tor_local)) = RHS_p(mp, index_ij+(i_var-1)*(n_tor_local)) + rhs_ij(i_var, 1)
                RHS_k(mp, index_ij+(i_var-1)*(n_tor_local)) = RHS_k(mp, index_ij+(i_var-1)*(n_tor_local)) + rhs_ij(i_var, 2)
             else
              RHS(index_ij+(i_var-1)*(n_tor_local)) = RHS(index_ij+(i_var-1)*(n_tor_local)) + (rhs_ij(i_var, 1)*HZ(im,mp) + rhs_ij(i_var, 2)*HZ_p(im,mp))
             endif
           enddo
            
            do k=1,n_vertex_max
              do l=1,n_order+1
                do in = n_tor_start,n_tor_end
                  if (use_fft) then
                    index_kl =       n_var*(n_order+1)*(k-1) +       n_var*(l-1) + 1
                  else
                    index_kl = n_tor_local*n_var*(n_order+1)*(k-1) + n_tor_local*n_var*(l-1) + in - n_tor_start +1
                  endif
                  
                  ! Unknown increments to next time step (delta u^n)
                  eq(var_varStar,0,0,0,:) = H(k,l,ms,mt)*element%size(k,l)*dur
                  eq(var_varStar,1,0,0,:) = (y_t(mp,ms,mt)*h_s(k,l,ms,mt) - y_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)*dur/xjac
                  eq(var_varStar,0,1,0,:) = (-x_t(mp,ms,mt)*h_s(k,l,ms,mt) + x_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)*dur/xjac
                  eq(var_varStar,0,0,1,:) = H(k,l,ms,mt)*element%size(k,l)*dup - eq(var_varStar,1,0,0,:)*x_p(mp,ms,mt) - eq(var_varStar,0,1,0,:)*y_p(mp,ms,mt)
                  eq(var_varStar,2,0,0,:) = (h_ss(k,l,ms,mt)*y_t(mp,ms,mt)**2 - 2.d0*h_st(k,l,ms,mt)*y_s(mp,ms,mt)*y_t(mp,ms,mt)                        &
                                        + h_tt(k,l,ms,mt)*y_s(mp,ms,mt)**2                                                                            &
                                        + h_s(k,l,ms,mt)*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt))                                &
                                        + h_t(k,l,ms,mt)*(y_st(mp,ms,mt)*y_s(mp,ms,mt) - y_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(k,l)*dur/xjac**2 &	
                                        - xjac_x*(h_s(k,l,ms,mt)*y_t(mp,ms,mt) - h_t(k,l,ms,mt)*y_s(mp,ms,mt))*element%size(k,l)*dur/xjac**2
                  eq(var_varStar,0,2,0,:) = (h_ss(k,l,ms,mt)*x_t(mp,ms,mt)**2 - 2.d0*h_st(k,l,ms,mt)*x_s(mp,ms,mt)*x_t(mp,ms,mt)                        &
                                        + h_tt(k,l,ms,mt)*x_s(mp,ms,mt)**2                                                                            &
                                        + h_s(k,l,ms,mt)*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt))                                &
                                        + h_t(k,l,ms,mt)*(x_st(mp,ms,mt)*x_s(mp,ms,mt) - x_ss(mp,ms,mt)*x_t(mp,ms,mt)))*element%size(k,l)*dur/xjac**2 &
                                        - xjac_y*(-h_s(k,l,ms,mt)*x_t(mp,ms,mt) + h_t(k,l,ms,mt)*x_s(mp,ms,mt))*element%size(k,l)*dur/xjac**2
                  eq(var_varStar,1,1,0,:) = (-h_ss(k,l,ms,mt)*y_t(mp,ms,mt)*x_t(mp,ms,mt) - h_tt(k,l,ms,mt)*x_s(mp,ms,mt)*y_s(mp,ms,mt)                 &
     	                                  + h_st(k,l,ms,mt)*(y_s(mp,ms,mt)*x_t(mp,ms,mt)  + y_t(mp,ms,mt)*x_s(mp,ms,mt))                                &
                                        - h_s(k,l,ms,mt)*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt))                                &
                                        - h_t(k,l,ms,mt)*(x_st(mp,ms,mt)*y_s(mp,ms,mt) - x_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(k,l)*dur/xjac**2 &
                                        - xjac_x*(-h_s(k,l,ms,mt)*x_t(mp,ms,mt) + h_t(k,l,ms,mt)*x_s(mp,ms,mt))*element%size(k,l)*dur/xjac**2
                  u_px                  = (y_t(mp,ms,mt)*h_s(k,l,ms,mt) - y_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)/xjac
                  u_py                  = (-x_t(mp,ms,mt)*h_s(k,l,ms,mt) + x_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)/xjac
                  ! Second derivatives wrt phi not implemented in FFT, 
                  ! and not necessary for stabilization (2nd derivatives only appear in hyperdissipation terms)
!                  eq(var_varStar,0,0,2,:) = -H(k,l,ms,mt)*element%size(k,l)*dur - x_pp(mp,ms,mt)*eq(var_varStar,1,0,0,:) - 2.d0*(x_p(mp,ms,mt)*u_px &
!                                        + y_p(mp,ms,mt)*u_py)*dup - y_pp(mp,ms,mt)*eq(var_varStar,0,1,0,:) + 2.d0*(x_p(mp,ms,mt)*x_p_x*eq(var_varStar,1,0,0,:) &
!                                        + x_p(mp,ms,mt)*y_p_x*eq(var_varStar,0,1,0,:) + y_p(mp,ms,mt)*x_p_y*eq(var_varStar,1,0,0,:) &
!                                        + y_p(mp,ms,mt)*y_p_y*eq(var_varStar,0,1,0,:)) + x_p(mp,ms,mt)**2*eq(var_varStar,2,0,0,:)             &
!                                        + 2.d0*x_p(mp,ms,mt)*y_p(mp,ms,mt)*eq(var_varStar,1,1,0,:) + y_p(mp,ms,mt)**2*eq(var_varStar,0,2,0,:)
                  eq(var_varStar,1,0,1,:) = u_px*dup - x_p_x*eq(var_varStar,1,0,0,:) - x_p(mp,ms,mt)*eq(var_varStar,2,0,0,:) - y_p_x*eq(var_varStar,0,1,0,:) &
                                        - y_p(mp,ms,mt)*eq(var_varStar,1,1,0,:)
                  eq(var_varStar,0,1,1,:) = u_py*dup - x_p_y*eq(var_varStar,1,0,0,:) - y_p(mp,ms,mt)*eq(var_varStar,0,2,0,:) - y_p_y*eq(var_varStar,0,1,0,:) &
                                        - x_p(mp,ms,mt)*eq(var_varStar,1,1,0,:)
  
                  ! Unknown increments to next time step, neglecting poloidal derivatives (for poloidal momentum source)
                  eq(var_varStar_pol,0,0,0,:) = H(k,l,ms,mt)*element%size(k,l)*dur
                  eq(var_varStar_pol,1,0,0,:) = (y_t(mp,ms,mt)*h_s(k,l,ms,mt))*element%size(k,l)*dur/xjac
                  eq(var_varStar_pol,0,1,0,:) = (-x_t(mp,ms,mt)*h_s(k,l,ms,mt))*element%size(k,l)*dur/xjac
                  eq(var_varStar_pol,0,0,1,:) = H(k,l,ms,mt)*element%size(k,l)*dup - eq(var_varStar_pol,1,0,0,:)*x_p(mp,ms,mt) - eq(var_varStar_pol,0,1,0,:)*y_p(mp,ms,mt)      

                  call get_amat(amat_ij, eq)                
                  
                  ! Include pre-factor to contribution
                  do i_var = 1, n_var
                    do j_var = 1, n_var 
                      if ((j_var .eq. var_Psi) .or. (j_var .eq. var_zj)) then
                        prefactor =  wst*BigR*xjac/F0
                      else
                        prefactor =  wst*BigR*xjac
                      endif
                    
                      amat_ij(i_var, j_var,:) = amat_ij(i_var, j_var,:)*prefactor
                    enddo ! j_var
                  enddo ! i_var

                  ! Fill up ELM
                  do i_var = 1, n_var
                    ij = index_ij + (i_var - 1)*n_tor_local
                    do j_var = 1, n_var 
                      kl = index_kl + (j_var - 1)*n_tor_local
                  
                      if (use_fft) then
                        ELM_p(mp,ij,kl)  =  ELM_p(mp,ij,kl)  + amat_ij(i_var, j_var, 1)
                        ELM_k(mp,ij,kl)  =  ELM_k(mp,ij,kl)  + amat_ij(i_var, j_var, 2)
                        ELM_n(mp,ij,kl)  =  ELM_n(mp,ij,kl)  + amat_ij(i_var, j_var, 3)
                        ELM_kn(mp,ij,kl) =  ELM_kn(mp,ij,kl) + amat_ij(i_var, j_var, 4)
                      else
                        ELM(ij,kl) = ELM(ij,kl) + (amat_ij(i_var,j_var,1)*HZ(im,mp)*HZ(in,mp)   + amat_ij(i_var,j_var,2)*HZ_p(im,mp)*HZ(in,mp) &
                                                 +  amat_ij(i_var,j_var,3)*HZ(im,mp)*HZ_p(in,mp) + amat_ij(i_var,j_var,4)*HZ_p(im,mp)*HZ_p(in,mp))
                      endif
                    enddo ! j_var
                  enddo ! i_var
                end do ! in loop (n_tor, or not...)
              end do ! l loop (n_order+1)
            end do ! k loop (n_vertex)
          end do ! im loop (n_tor, or not...)
        end do ! j loop
      end do ! i loop
    end do ! mp loop (n_plane)
  end do ! ms loop
end do ! mt loop


if (use_fft) then
  do i=1,n_vertex_max*n_var*(n_order+1)
    do j=1, n_vertex_max*n_var*(n_order+1)
      in_fft =  ELM_p(1:n_plane,i,j)

#ifdef USE_FFTW
      call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
      call my_fft(in_fft, out_fft, n_plane)
#endif
    
      do k=1,(n_tor+1)/2
        index_k = n_tor*(i-1) + max(2*(k-1),1)

        do m=1,(n_tor+1)/2
          index_m = n_tor*(j-1) + max(2*(m-1),1)
          l = (k-1) + (m-1)

          if ((l .ge. 0) .and. (l .le. n_plane/2)) then
            ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + real(out_fft(l+1))
            ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   - imag(out_fft(l+1))
            ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) - imag(out_fft(l+1))
            ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) - real(out_fft(l+1))
          else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
            ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + real(out_fft(abs(l)+1))
            ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + imag(out_fft(abs(l)+1))
            ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + imag(out_fft(abs(l)+1))
            ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) - real(out_fft(abs(l)+1))
          end if

          l = (k-1) - (m-1)

          if ((l .ge. 0) .and. (l .le. n_plane/2)) then
            ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + real(out_fft(l+1))
            ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   - imag(out_fft(l+1))
            ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + imag(out_fft(l+1))
            ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + real(out_fft(l+1))
          else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
            ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + real(out_fft(abs(l)+1))
            ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + imag(out_fft(abs(l)+1))
            ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) - imag(out_fft(abs(l)+1))
            ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + real(out_fft(abs(l)+1))
          end if
        end do
      end do
    end do
  end do

  do i=1,n_vertex_max*n_var*(n_order+1)
    do j=1, n_vertex_max*n_var*(n_order+1)
      if (maxval(abs(ELM_n(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft =  ELM_n(1:n_plane,i,j)

#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
        call my_fft(in_fft, out_fft, n_plane)
#endif

        do k=1,(n_tor+1)/2
          index_k = n_tor*(i-1) + max(2*(k-1),1)

          do m=1,(n_tor+1)/2
            im = max(2*(m-1),1)
            index_m = n_tor*(j-1) + max(2*(m-1),1)
            l = (k-1) + (m-1)

            if ((l .ge. 0) .and. (l .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + imag(out_fft(l+1)) * float(mode(im))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + real(out_fft(l+1)) * float(mode(im))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + real(out_fft(l+1)) * float(mode(im))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) - imag(out_fft(l+1)) * float(mode(im))
            else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   - imag(out_fft(abs(l)+1)) * float(mode(im))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + real(out_fft(abs(l)+1)) * float(mode(im))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + real(out_fft(abs(l)+1)) * float(mode(im))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + imag(out_fft(abs(l)+1)) * float(mode(im))
            end if

            l = (k-1) - (m-1)

            if ((l .ge. 0) .and. (l .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   - imag(out_fft(l+1)) * float(mode(im))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   - real(out_fft(l+1)) * float(mode(im))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + real(out_fft(l+1)) * float(mode(im))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) - imag(out_fft(l+1)) * float(mode(im))
            else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + imag(out_fft(abs(l)+1)) * float(mode(im))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   - real(out_fft(abs(l)+1)) * float(mode(im))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + real(out_fft(abs(l)+1)) * float(mode(im))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + imag(out_fft(abs(l)+1)) * float(mode(im))
            end if
          end do
        end do
      end if
    end do
  end do

  do i=1,n_vertex_max*n_var*(n_order+1)
    do j=1, n_vertex_max*n_var*(n_order+1)
      if (maxval(abs(ELM_k(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft =  ELM_k(1:n_plane,i,j)

#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
        call my_fft(in_fft, out_fft, n_plane)
#endif
    
        do k=1,(n_tor+1)/2
          ik      = max(2*(k-1),1)
          index_k = n_tor*(i-1) + max(2*(k-1),1)

          do m=1,(n_tor+1)/2
            index_m = n_tor*(j-1) + max(2*(m-1),1)
            l = (k-1) + (m-1)

            if ((l .ge. 0) .and. (l .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + imag(out_fft(l+1)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + real(out_fft(l+1)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + real(out_fft(l+1)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) - imag(out_fft(l+1)) * float(mode(ik))
            else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   - imag(out_fft(abs(l)+1)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + real(out_fft(abs(l)+1)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + real(out_fft(abs(l)+1)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + imag(out_fft(abs(l)+1)) * float(mode(ik))
            end if

            l = (k-1) - (m-1)

            if ((l .ge. 0) .and. (l .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + imag(out_fft(l+1)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + real(out_fft(l+1)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) - real(out_fft(l+1)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + imag(out_fft(l+1)) * float(mode(ik))
            else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   - imag(out_fft(abs(l)+1)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + real(out_fft(abs(l)+1)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) - real(out_fft(abs(l)+1)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) - imag(out_fft(abs(l)+1)) * float(mode(ik))
            end if
          end do
        end do
      end if
    end do
  end do

  do i=1,n_vertex_max*n_var*(n_order+1)
    do j=1, n_vertex_max*n_var*(n_order+1)
      if (maxval(abs(ELM_kn(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft =  ELM_kn(1:n_plane,i,j)

#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
        call my_fft(in_fft, out_fft, n_plane)
#endif
    
        do k=1,(n_tor+1)/2
          ik      = max(2*(k-1),1)
          index_k = n_tor*(i-1) + max(2*(k-1),1)

          do m=1,(n_tor+1)/2
            im      = max(2*(m-1),1)
            index_m = n_tor*(j-1) + max(2*(m-1),1)
            l = (k-1) + (m-1)

            if ((l .ge. 0) .and. (l .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   - real(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + imag(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + imag(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + real(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
            else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   - real(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   - imag(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) - imag(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + real(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
            end if

            l = (k-1) - (m-1)

            if ((l .ge. 0) .and. (l .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + real(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   - imag(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) + imag(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + real(out_fft(l+1)) * float(mode(im)) * float(mode(ik))
            else if ((l .lt. 0) .and. (abs(l) .le. n_plane/2)) then
              ELM(index_k,  index_m  ) = ELM(index_k,  index_m)   + real(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m  ) = ELM(index_k+1,index_m)   + imag(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k,  index_m+1) = ELM(index_k,  index_m+1) - imag(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
              ELM(index_k+1,index_m+1) = ELM(index_k+1,index_m+1) + real(out_fft(abs(l)+1)) * float(mode(im)) * float(mode(ik))
            end if
          end do
        end do
      end if
    end do
  end do

  ELM = 0.5d0 * ELM

  do j=1, n_vertex_max*n_var*(n_order+1)
    in_fft = RHS_p(1:n_plane,j)

#ifdef USE_FFTW
    call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
    call my_fft(in_fft, out_fft, n_plane)
#endif
  
    index = n_tor*(j-1) + 1
    RHS(index) = real(out_fft(1))

    do k=2,(n_tor+1)/2
      index = n_tor*(j-1) + 2*(k-1)
      RHS(index)   =   real(out_fft(k))
      RHS(index+1) = - imag(out_fft(k))
    end do
  end do

  do j=1, n_vertex_max*n_var*(n_order+1)
    in_fft = RHS_k(1:n_plane,j)

#ifdef USE_FFTW
    call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
    call my_fft(in_fft, out_fft, n_plane)
#endif

    index = n_tor*(j-1) + 1
    ik    = 1
    RHS(index) = RHS(index) + imag(out_fft(1)) * float(mode(ik))

    do k=2,(n_tor+1)/2
      ik    = max(2*(k-1),1)
      index = n_tor*(j-1) + 2*(k-1)
      RHS(index)   = RHS(index)   + imag(out_fft(k)) * float(mode(ik))
      RHS(index+1) = RHS(index+1) + real(out_fft(k)) * float(mode(ik))
    end do
  end do
else
  return
end if

return
end subroutine element_matrix_fft


subroutine my_fft(in_fft,out_fft,n)

implicit none
      
real*8     :: in_fft(*)
complex*16 :: out_fft(*)

integer    :: i, n    
real*8     :: tmp_fft(2*n+2)
      
tmp_fft(1:n) = in_fft(1:n)
      
call RFT2(tmp_fft,n,1)
      
do i=1,n
  out_fft(i) = cmplx(tmp_fft(2*i-1),tmp_fft(2*i))
enddo
      
return
end subroutine my_fft

subroutine get_auxiliary(eq)
  use data_structure
  use constants
  use mod_parameters
  use phys_module
  use mod_equations

  implicit none

  real*8, dimension(:,:,:,:,:), pointer, intent(inout) :: eq
  
#include "aux_automatic.h"
end subroutine

subroutine get_rhs(rhs_ij, eq)
  use data_structure
  use constants
  use mod_parameters
  use phys_module
  use mod_equations
  
  implicit none

  real*8, dimension(n_var,4), intent(inout)       :: rhs_ij
  real*8, dimension(:,:,:,:,:), pointer, intent(in) :: eq
  real*8     :: theta, zeta, reta
  
  ! --- Take time evolution parameters from phys_module
  theta = time_evol_theta
  ! change zeta for variable dt
  zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  if (eta .ne. 0.d0) then
    reta = eta_ohmic/eta
  else
    reta = 0.d0
  end if
  
#include "rhs_automatic.h"
end subroutine

subroutine get_amat(amat_ij, eq)
  use data_structure
  use constants
  use mod_parameters
  use phys_module
  use mod_equations
  
  implicit none

  real*8, dimension(n_var,n_var,4), intent(inout) :: amat_ij
  real*8, dimension(:,:,:,:,:), pointer, intent(in) :: eq
  real*8     :: theta, zeta, reta
  
  ! --- Take time evolution parameters from phys_module
  theta = time_evol_theta
  ! change zeta for variable dt
  zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  if (eta .ne. 0.d0) then
    reta = eta_ohmic/eta
  else
    reta = 0.d0
  end if
  
#include "amat_automatic.h"
end subroutine

subroutine element_matrix_pfafem(node_list, element_list, ielm, xpoint2, xcase2, &
                                 R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, &
                                 ELM_sp, RHS_u, tid, aux_node_list)
  use constants
  use mod_parameters
  use data_structure
  use gauss
  use basis_at_gaussian, only: H, H_s, H_t, H_ss, H_st, H_tt
  use phys_module
  use diffusivities, only: get_dperp, get_zkperp, get_zk_iperp, get_zk_eperp
  use corr_neg
  use mod_equations
  use mod_sources
  use mod_pfafem, only: pfafem_cache, pfafem_el_lo, pfafem_el_hi, pfafem_ep_dim, &
                        pfafem_ng_tor, pfafem_ng_pol, pfafem_tau_q, pfafem_w_tor, &
                        pfafem_t_pol, pfafem_w_pol, pfafem_dphi, pfafem_chi_q, &
                        pfafem_qp_cached, pfafem_qp_geom, pfafem_qp_ngrp, &
                        pfafem_qp_gu, pfafem_qp_gjb, pfafem_qp_giuj, pfafem_qp_db, &
                        pfafem_eval_coord_hz, pfafem_hermite2d_jet, pfafem_eval_qp_basis
  implicit none

  type(type_node_list),    intent(in)           :: node_list
  type(type_element_list), intent(in)           :: element_list
  integer,                 intent(in)           :: ielm, xcase2, tid
  logical,                 intent(in)           :: xpoint2
  real*8,                  intent(in)           :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,                  intent(in)           :: R_xpoint(2), Z_xpoint(2)
  real*8,                  intent(out)          :: ELM_sp(:, :, :, :, :)
  real*8,                  intent(out)          :: RHS_u(:, :, :)
  type(type_node_list),    intent(in), optional :: aux_node_list

  type(type_element) :: element
  type(type_node)    :: nodes(n_vertex_max)
  real*8, dimension(:,:,:,:,:), pointer :: eq

  integer :: i, j, m, in_c, ms, mt, eP, ep1, iep_q, iq, iel_loc
  integer :: n_raw, raw_u(64), raw_io(64), raw_jb(64), n_grp
  integer :: n_act, act_u(64), act_io(64), act_jb(64)
  integer :: grp_u(16), grp_jb(16), grp_iuj(16), iuja, iujb, isp
  integer :: a, b, unode, i_var, j_var, r_loc
  real*8  :: s_factor, dt_ratio, phi_q, t_q, w_t, sz
  real*8  :: hz_c(n_coord_tor), hzp_c(n_coord_tor), hzpp_c(n_coord_tor)
  real*8  :: H0_q(4, n_degrees), Hs_q(4, n_degrees), Ht_q(4, n_degrees)
  real*8  :: Hss_q(4, n_degrees), Hst_q(4, n_degrees), Htt_q(4, n_degrees)
  real*8  :: x_g, x_s, x_t, x_p, x_ss, x_st, x_tt, x_sp, x_tp, x_pp
  real*8  :: y_g, y_s, y_t, y_p, y_ss, y_st, y_tt, y_sp, y_tp, y_pp
  real*8  :: xjac, xjac_x, xjac_y, x_p_x, x_p_y, y_p_x, y_p_y, BigR, wvol, prefactor
  real*8  :: psi_norm
  real*8  :: current_source, particle_source, heat_source, heat_source_i, heat_source_e
  real*8  :: phi_source, dPhi_source_dpsi, dPhi_source_dpsi2
  real*8  :: dummy1, dummy2, dummy3, dummy4, dummy5, dummy6
  real*8  :: rho0, rho0_s, rho0_ss, T0, T0_s, T0_ss, T0_i, T0_i_s, T0_i_ss, T0_e, T0_e_s, T0_e_ss
  real*8  :: T0_i_corr, T0_e_corr, rho0_corr, dT0_i_corr_dT, dT0_e_corr_dT, drho0_corr_dn
  real*8  :: T0_e_corr_eV, T0_i_corr_eV, dT0_e_corr_eV_dT, dT0_i_corr_eV_dT
  real*8  :: ne_SI, lambda_e_bg, nu_e_bg, t_norm, dnu_e_bg_dTi, dnu_e_bg_dTe, dnu_e_bg_drho
  real*8  :: phi0_s, phi0_p_st, b_s, b_p_st
  real*8  :: rv(64), rR(64), rZ(64), rP(64), rRR(64), rRZ(64), rZZ(64), rRP(64), rZP(64), rPP(64)
  real*8  :: bv(64), bR(64), bZ(64), bP(64), bRR(64), bRZ(64), bZZ(64), bRP(64), bZP(64), bPP(64)
  real*8  :: val_k(n_var), del_k(n_var), r_jet(10)
  real*8  :: rhs_ij(n_var, 4), amat_ij(n_var, n_var, 4)
  integer :: ip_d, iq_d, idx_pq, pq_k, lane_k
  integer :: n_ij, k_ij, sq, g_a, g_b, a0, b0, io_a, io_b, u_a, u_b, jb_a, jb_b
  integer :: iv_ij(n_var * n_var), jv_ij(n_var * n_var), nq_ij(n_var * n_var), q_ij(10, n_var * n_var)
  integer, parameter :: d_ix(3, 10) = reshape([ &
    0,0,0,  1,0,0,  0,1,0,  0,0,1,  2,0,0, &
    0,2,0,  1,1,0,  0,0,2,  1,0,1,  0,1,1 ], [3, 10])
  real*8  :: db_mat(10, 64), R_p(n_var, 10), C_pq(n_var, n_var, 10, 10)
  real*8  :: T_kqa(64, 10, n_var * n_var), m44(4, 4), db_val, amat_ab(n_var, n_var), r_val
  real*8  :: x_iq0(4, 4), x_iq1(4, 4), x_iq2(4, 4), y_iq0(4, 4), y_iq1(4, 4), y_iq2(4, 4)
  logical :: has_axis_u, use_qp_c

  eq => thread_eq(tid)%eq
  ELM_sp  = 0.d0
  RHS_u   = 0.d0
  rhs_ij  = 0.d0
  amat_ij = 0.d0

  s_factor = 1.d0 / dble(n_flux - 1)
  if (tstep_prev /= 0.d0) then
    dt_ratio = tstep / tstep_prev
  else
    dt_ratio = 1.d0
  end if

  element = element_list%element(ielm)
  do i = 1, n_vertex_max
    nodes(i) = node_list%node(element%vertex(i))
  end do

  iel_loc  = ielm - pfafem_el_lo + 1
  use_qp_c = (pfafem_qp_cached .and. ielm >= pfafem_el_lo .and. ielm <= pfafem_el_hi)
  has_axis_u = .false.
  if (treat_axis) then
    do a = 1, pfafem_cache(ielm)%n_unodes
      if (node_list%node(pfafem_cache(ielm)%unodes(a))%axis_node) then
        has_axis_u = .true.
        exit
      end if
    end do
  end if

  do eP = 0, n_tor - 1
    ep1   = eP + 1
    iep_q = merge(1, ep1, pfafem_ep_dim == 1)
    do iq = 1, pfafem_ng_tor
      phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
      if (.not. use_qp_c) then
        if (n_coord_tor == 1) then
          hz_c(1)   = 1.d0
          hzp_c(1)  = 0.d0
          hzpp_c(1) = 0.d0
        else
          call pfafem_eval_coord_hz(phi_q, hz_c, hzp_c, hzpp_c)
        end if

        do i = 1, n_vertex_max
          do j = 1, n_order + 1
            sz = element%size(i, j)
            x_iq0(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 1), hz_c)
            x_iq1(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 1), hzp_c)
            x_iq2(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 1), hzpp_c)
            y_iq0(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 2), hz_c)
            y_iq1(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 2), hzp_c)
            y_iq2(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 2), hzpp_c)
          end do
        end do
      end if

      do ms = 1, n_gauss
        do mt = 1, pfafem_ng_pol
          if (use_qp_c) then
            BigR     = pfafem_qp_geom(1,  ms, mt, iq, ep1, iel_loc)
            y_g      = pfafem_qp_geom(2,  ms, mt, iq, ep1, iel_loc)
            x_s      = pfafem_qp_geom(3,  ms, mt, iq, ep1, iel_loc)
            x_t      = pfafem_qp_geom(4,  ms, mt, iq, ep1, iel_loc)
            x_p      = pfafem_qp_geom(5,  ms, mt, iq, ep1, iel_loc)
            y_s      = pfafem_qp_geom(6,  ms, mt, iq, ep1, iel_loc)
            y_t      = pfafem_qp_geom(7,  ms, mt, iq, ep1, iel_loc)
            y_p      = pfafem_qp_geom(8,  ms, mt, iq, ep1, iel_loc)
            x_st     = pfafem_qp_geom(9,  ms, mt, iq, ep1, iel_loc)
            x_tt     = pfafem_qp_geom(10, ms, mt, iq, ep1, iel_loc)
            y_st     = pfafem_qp_geom(11, ms, mt, iq, ep1, iel_loc)
            y_tt     = pfafem_qp_geom(12, ms, mt, iq, ep1, iel_loc)
            xjac     = pfafem_qp_geom(13, ms, mt, iq, ep1, iel_loc)
            xjac_x   = pfafem_qp_geom(14, ms, mt, iq, ep1, iel_loc)
            xjac_y   = pfafem_qp_geom(15, ms, mt, iq, ep1, iel_loc)
            x_p_x    = pfafem_qp_geom(16, ms, mt, iq, ep1, iel_loc)
            x_p_y    = pfafem_qp_geom(17, ms, mt, iq, ep1, iel_loc)
            y_p_x    = pfafem_qp_geom(18, ms, mt, iq, ep1, iel_loc)
            y_p_y    = pfafem_qp_geom(19, ms, mt, iq, ep1, iel_loc)
            wvol     = pfafem_qp_geom(20, ms, mt, iq, ep1, iel_loc)
            psi_norm = pfafem_qp_geom(21, ms, mt, iq, ep1, iel_loc)
          else
            if (ielm >= pfafem_el_lo .and. ielm <= pfafem_el_hi) then
              t_q = pfafem_t_pol(ms, mt, iq, iep_q, iel_loc)
              w_t = pfafem_w_pol(ms, mt, iq, iep_q, iel_loc)
            else
              t_q = xgauss(min(n_gauss, mt))
              w_t = wgauss(min(n_gauss, mt))
            end if
            call pfafem_hermite2d_jet(xgauss(ms), t_q, H0_q, Hs_q, Ht_q, Hss_q, Hst_q, Htt_q)

            x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_p = 0.d0
            x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0; x_sp = 0.d0; x_tp = 0.d0; x_pp = 0.d0
            y_g = 0.d0; y_s = 0.d0; y_t = 0.d0; y_p = 0.d0
            y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0; y_sp = 0.d0; y_tp = 0.d0; y_pp = 0.d0
            psi_norm = 0.d0

            do i = 1, n_vertex_max
              do j = 1, n_order + 1
                x_g  = x_g  + x_iq0(i, j) * H0_q(i, j)
                x_s  = x_s  + x_iq0(i, j) * Hs_q(i, j)
                x_t  = x_t  + x_iq0(i, j) * Ht_q(i, j)
                x_p  = x_p  + x_iq1(i, j) * H0_q(i, j)
                x_ss = x_ss + x_iq0(i, j) * Hss_q(i, j)
                x_st = x_st + x_iq0(i, j) * Hst_q(i, j)
                x_tt = x_tt + x_iq0(i, j) * Htt_q(i, j)
                x_sp = x_sp + x_iq1(i, j) * Hs_q(i, j)
                x_tp = x_tp + x_iq1(i, j) * Ht_q(i, j)
                x_pp = x_pp + x_iq2(i, j) * H0_q(i, j)

                y_g  = y_g  + y_iq0(i, j) * H0_q(i, j)
                y_s  = y_s  + y_iq0(i, j) * Hs_q(i, j)
                y_t  = y_t  + y_iq0(i, j) * Ht_q(i, j)
                y_p  = y_p  + y_iq1(i, j) * H0_q(i, j)
                y_ss = y_ss + y_iq0(i, j) * Hss_q(i, j)
                y_st = y_st + y_iq0(i, j) * Hst_q(i, j)
                y_tt = y_tt + y_iq0(i, j) * Htt_q(i, j)
                y_sp = y_sp + y_iq1(i, j) * Hs_q(i, j)
                y_tp = y_tp + y_iq1(i, j) * Ht_q(i, j)
                y_pp = y_pp + y_iq2(i, j) * H0_q(i, j)

                psi_norm = psi_norm + nodes(i)%r_tor_eq(j) * element%size(i, j) * H0_q(i, j)
              end do
            end do

            xjac   = x_s * y_t - x_t * y_s
            xjac_x = (x_ss*y_t**2 - y_ss*x_t*y_t - 2.d0*x_st*y_s*y_t &
                    + y_st*(x_s*y_t + x_t*y_s) + x_tt*y_s**2 - y_tt*x_s*y_s) / xjac
            xjac_y = (y_tt*x_s**2 - x_tt*y_s*x_s - 2.d0*y_st*x_t*x_s &
                    + x_st*(y_t*x_s + y_s*x_t) + y_ss*x_t**2 - x_ss*y_t*x_t) / xjac
            x_p_x  = (x_sp * y_t - x_tp * y_s) / xjac
            x_p_y  = (x_tp * x_s - x_sp * x_t) / xjac
            y_p_x  = (y_sp * y_t - y_tp * y_s) / xjac
            y_p_y  = (y_tp * x_s - y_sp * x_t) / xjac

            BigR = x_g
            wvol = wgauss(ms) * w_t * pfafem_w_tor(iq) * BigR * xjac
          end if

          eq = 0.d0
          current_source  = 0.d0
          particle_source = 0.d0
          heat_source     = 0.d0
          heat_source_i   = 0.d0
          heat_source_e   = 0.d0

          if (use_qp_c) then
            n_grp = int(pfafem_qp_ngrp(ms, mt, iq, ep1, iel_loc))
            n_raw = 4 * n_grp
            do g_a = 1, n_grp
              u_a   = int(pfafem_qp_gu(g_a, ms, mt, iq, ep1, iel_loc))
              jb_a  = int(pfafem_qp_gjb(g_a, ms, mt, iq, ep1, iel_loc))
              unode = pfafem_cache(ielm)%unodes(u_a)
              a0    = 4 * (g_a - 1)
              do io_a = 1, 4
                a     = a0 + io_a
                r_jet = pfafem_qp_db(1:10, a, ms, mt, iq, ep1, iel_loc)
                val_k = node_list%node(unode)%values(jb_a, io_a, 1:n_var)
                del_k = node_list%node(unode)%deltas(jb_a, io_a, 1:n_var) * dt_ratio

                eq(1:n_var, 0, 0, 0, 1) = eq(1:n_var, 0, 0, 0, 1) + val_k * r_jet(1)
                eq(1:n_var, 1, 0, 0, 1) = eq(1:n_var, 1, 0, 0, 1) + val_k * r_jet(2)
                eq(1:n_var, 0, 1, 0, 1) = eq(1:n_var, 0, 1, 0, 1) + val_k * r_jet(3)
                eq(1:n_var, 0, 0, 1, 1) = eq(1:n_var, 0, 0, 1, 1) + val_k * r_jet(4)
                eq(1:n_var, 2, 0, 0, 1) = eq(1:n_var, 2, 0, 0, 1) + val_k * r_jet(5)
                eq(1:n_var, 0, 2, 0, 1) = eq(1:n_var, 0, 2, 0, 1) + val_k * r_jet(6)
                eq(1:n_var, 1, 1, 0, 1) = eq(1:n_var, 1, 1, 0, 1) + val_k * r_jet(7)
                eq(1:n_var, 0, 0, 2, 1) = eq(1:n_var, 0, 0, 2, 1) + val_k * r_jet(8)
                eq(1:n_var, 1, 0, 1, 1) = eq(1:n_var, 1, 0, 1, 1) + val_k * r_jet(9)
                eq(1:n_var, 0, 1, 1, 1) = eq(1:n_var, 0, 1, 1, 1) + val_k * r_jet(10)

                eq(n_var+1:2*n_var, 0, 0, 0, 1) = eq(n_var+1:2*n_var, 0, 0, 0, 1) + del_k * r_jet(1)
                eq(n_var+1:2*n_var, 1, 0, 0, 1) = eq(n_var+1:2*n_var, 1, 0, 0, 1) + del_k * r_jet(2)
                eq(n_var+1:2*n_var, 0, 1, 0, 1) = eq(n_var+1:2*n_var, 0, 1, 0, 1) + del_k * r_jet(3)
                eq(n_var+1:2*n_var, 0, 0, 1, 1) = eq(n_var+1:2*n_var, 0, 0, 1, 1) + del_k * r_jet(4)

                if (keep_current_prof) then
                  current_source = current_source + node_list%node(unode)%j_source(jb_a, io_a) * r_jet(1)
                end if
              end do
            end do
          else
            call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                      BigR, y_g, phi_q, .false., n_raw, raw_u, raw_io, raw_jb, &
                                      rv, rR, rZ, rP, rRR, rRZ, rZZ, rRP, rZP, rPP)
            do a = 1, n_raw
              unode = pfafem_cache(ielm)%unodes(raw_u(a))
              val_k = node_list%node(unode)%values(raw_jb(a), raw_io(a), 1:n_var)
              del_k = node_list%node(unode)%deltas(raw_jb(a), raw_io(a), 1:n_var) * dt_ratio

              eq(1:n_var, 0, 0, 0, 1) = eq(1:n_var, 0, 0, 0, 1) + val_k * rv(a)
              eq(1:n_var, 1, 0, 0, 1) = eq(1:n_var, 1, 0, 0, 1) + val_k * rR(a)
              eq(1:n_var, 0, 1, 0, 1) = eq(1:n_var, 0, 1, 0, 1) + val_k * rZ(a)
              eq(1:n_var, 0, 0, 1, 1) = eq(1:n_var, 0, 0, 1, 1) + val_k * rP(a)
              eq(1:n_var, 2, 0, 0, 1) = eq(1:n_var, 2, 0, 0, 1) + val_k * rRR(a)
              eq(1:n_var, 0, 2, 0, 1) = eq(1:n_var, 0, 2, 0, 1) + val_k * rZZ(a)
              eq(1:n_var, 1, 1, 0, 1) = eq(1:n_var, 1, 1, 0, 1) + val_k * rRZ(a)
              eq(1:n_var, 0, 0, 2, 1) = eq(1:n_var, 0, 0, 2, 1) + val_k * rPP(a)
              eq(1:n_var, 1, 0, 1, 1) = eq(1:n_var, 1, 0, 1, 1) + val_k * rRP(a)
              eq(1:n_var, 0, 1, 1, 1) = eq(1:n_var, 0, 1, 1, 1) + val_k * rZP(a)

              eq(n_var+1:2*n_var, 0, 0, 0, 1) = eq(n_var+1:2*n_var, 0, 0, 0, 1) + del_k * rv(a)
              eq(n_var+1:2*n_var, 1, 0, 0, 1) = eq(n_var+1:2*n_var, 1, 0, 0, 1) + del_k * rR(a)
              eq(n_var+1:2*n_var, 0, 1, 0, 1) = eq(n_var+1:2*n_var, 0, 1, 0, 1) + del_k * rZ(a)
              eq(n_var+1:2*n_var, 0, 0, 1, 1) = eq(n_var+1:2*n_var, 0, 0, 1, 1) + del_k * rP(a)

              if (keep_current_prof) then
                current_source = current_source + node_list%node(unode)%j_source(raw_jb(a), raw_io(a)) * rv(a)
              end if
            end do
          end if

          if (.not. maintain_profiles) then
            if (with_TiTe) then
              call sources(xpoint2, xcase2, y_g, Z_xpoint, psi_norm, psi_axis, psi_bnd, &
                           particle_source, heat_source_i, heat_source_e)
            else
              call sources(xpoint2, xcase2, y_g, Z_xpoint, psi_norm, 0.d0, 1.d0, &
                           particle_source, heat_source)
            end if
          end if

          ! Poloidal-only Phi0 (neglecting t derivative along flux surface)
          phi0_s    = x_s * eq(var_Phi, 1, 0, 0, 1) + y_s * eq(var_Phi, 0, 1, 0, 1)
          phi0_p_st = eq(var_Phi, 0, 0, 1, 1) + x_p * eq(var_Phi, 1, 0, 0, 1) + y_p * eq(var_Phi, 0, 1, 0, 1)
          eq(var_Phi_pol, 0, 0, 0, 1) = eq(var_Phi, 0, 0, 0, 1)
          eq(var_Phi_pol, 1, 0, 0, 1) = ( y_t * phi0_s) / xjac
          eq(var_Phi_pol, 0, 1, 0, 1) = (-x_t * phi0_s) / xjac
          eq(var_Phi_pol, 0, 0, 1, 1) = phi0_p_st - eq(var_Phi_pol, 1, 0, 0, 1) * x_p - eq(var_Phi_pol, 0, 1, 0, 1) * y_p

          ! Vacuum scalar magnetic potential chi and cylindrical R
          eq(var_chi, :, :, :, 1) = pfafem_chi_q(:, :, :, ms, mt, iq, eP + 1, iel_loc)
          eq(var_R,   0, 0, 0, 1) = BigR
          eq(var_R,   1, 0, 0, 1) = 1.d0

          ! Psi and zj scaling by F0
          eq(var_Psi,  :, :, :, 1) = eq(var_Psi,  :, :, :, 1) / F0
          eq(var_dPsi, :, :, :, 1) = eq(var_dPsi, :, :, :, 1) / F0
          eq(var_zj,   :, :, :, 1) = eq(var_zj,   :, :, :, 1) / F0
          eq(var_dzj,  :, :, :, 1) = eq(var_dzj,  :, :, :, 1) / F0

          eq(var_D_perp, 0, 0, 0, 1) = get_dperp(psi_norm)
          eq(var_D_par,  0, 0, 0, 1) = D_par
          eq(var_S_rho,  0, 0, 0, 1) = particle_source
          eq(var_S_j,    0, 0, 0, 1) = current_source / F0

          call potential_source(xpoint2, xcase2, y_g, Z_xpoint, psi_norm, 0.d0, 1.d0, &
                                phi_source, dPhi_source_dpsi, dummy1, dPhi_source_dpsi2, &
                                dummy2, dummy3, dummy4, dummy5, dummy6)
          dPhi_source_dpsi = dPhi_source_dpsi * s_factor
          eq(var_S_phi_pol, 0, 0, 0, 1) = phi_source
          eq(var_S_phi_pol, 1, 0, 0, 1) = ( y_t * dPhi_source_dpsi) / xjac
          eq(var_S_phi_pol, 0, 1, 0, 1) = (-x_t * dPhi_source_dpsi) / xjac
          eq(var_S_phi_pol, 0, 0, 1, 1) = -eq(var_S_phi_pol, 1, 0, 0, 1) * x_p - eq(var_S_phi_pol, 0, 1, 0, 1) * y_p

          if (maintain_profiles) then
            call density(xpoint2, xcase2, y_g, Z_xpoint, psi_norm, 0.d0, 1.d0, &
                         rho0, rho0_s, dummy1, rho0_ss, dummy2, dummy3, dummy4, dummy5, dummy6)
            rho0_s  = rho0_s  * s_factor
            rho0_ss = rho0_ss * s_factor**2
            eq(var_rho_equil, 0, 0, 0, 1) = rho0
            eq(var_rho_equil, 1, 0, 0, 1) = ( y_t * rho0_s) / xjac
            eq(var_rho_equil, 0, 1, 0, 1) = (-x_t * rho0_s) / xjac
            eq(var_rho_equil, 0, 0, 1, 1) = -eq(var_rho_equil, 1, 0, 0, 1) * x_p - eq(var_rho_equil, 0, 1, 0, 1) * y_p
            eq(var_rho_equil, 2, 0, 0, 1) = (rho0_ss*y_t**2 + rho0_s*(y_st*y_t - y_tt*y_s)) / xjac**2 - xjac_x*(rho0_s*y_t) / xjac**2
            eq(var_rho_equil, 0, 2, 0, 1) = (rho0_ss*x_t**2 + rho0_s*(x_st*x_t - x_tt*x_s)) / xjac**2 - xjac_y*(-rho0_s*x_t) / xjac**2
            eq(var_rho_equil, 1, 1, 0, 1) = (-rho0_ss*y_t*x_t - rho0_s*(x_st*y_t - x_tt*y_s)) / xjac**2 - xjac_x*(-rho0_s*x_t) / xjac**2
            eq(var_rho_equil, 1, 0, 1, 1) = -x_p_x*eq(var_rho_equil, 1, 0, 0, 1) - x_p*eq(var_rho_equil, 2, 0, 0, 1) &
                                          - y_p_x*eq(var_rho_equil, 0, 1, 0, 1) - y_p*eq(var_rho_equil, 1, 1, 0, 1)
            eq(var_rho_equil, 0, 1, 1, 1) = -x_p_y*eq(var_rho_equil, 1, 0, 0, 1) - x_p*eq(var_rho_equil, 1, 1, 0, 1) &
                                          - y_p_y*eq(var_rho_equil, 0, 1, 0, 1) - y_p*eq(var_rho_equil, 0, 2, 0, 1)

            if (with_TiTe) then
              call temperature_i(xpoint2, xcase2, y_g, Z_xpoint, psi_norm, 0.d0, 1.d0, &
                                 T0_i, T0_i_s, dummy1, T0_i_ss, dummy2, dummy3, dummy4, dummy5, dummy6)
              T0_i_s  = T0_i_s  * s_factor
              T0_i_ss = T0_i_ss * s_factor**2
              eq(var_Ti_equil, 0, 0, 0, 1) = T0_i
              eq(var_Ti_equil, 1, 0, 0, 1) = ( y_t * T0_i_s) / xjac
              eq(var_Ti_equil, 0, 1, 0, 1) = (-x_t * T0_i_s) / xjac
              eq(var_Ti_equil, 0, 0, 1, 1) = -eq(var_Ti_equil, 1, 0, 0, 1) * x_p - eq(var_Ti_equil, 0, 1, 0, 1) * y_p
              eq(var_Ti_equil, 2, 0, 0, 1) = (T0_i_ss*y_t**2 + T0_i_s*(y_st*y_t - y_tt*y_s)) / xjac**2 - xjac_x*(T0_i_s*y_t) / xjac**2
              eq(var_Ti_equil, 0, 2, 0, 1) = (T0_i_ss*x_t**2 + T0_i_s*(x_st*x_t - x_tt*x_s)) / xjac**2 - xjac_y*(-T0_i_s*x_t) / xjac**2
              eq(var_Ti_equil, 1, 1, 0, 1) = (-T0_i_ss*y_t*x_t - T0_i_s*(x_st*y_t - x_tt*y_s)) / xjac**2 - xjac_x*(-T0_i_s*x_t) / xjac**2
              eq(var_Ti_equil, 1, 0, 1, 1) = -x_p_x*eq(var_Ti_equil, 1, 0, 0, 1) - x_p*eq(var_Ti_equil, 2, 0, 0, 1) &
                                            - y_p_x*eq(var_Ti_equil, 0, 1, 0, 1) - y_p*eq(var_Ti_equil, 1, 1, 0, 1)
              eq(var_Ti_equil, 0, 1, 1, 1) = -x_p_y*eq(var_Ti_equil, 1, 0, 0, 1) - x_p*eq(var_Ti_equil, 1, 1, 0, 1) &
                                            - y_p_y*eq(var_Ti_equil, 0, 1, 0, 1) - y_p*eq(var_Ti_equil, 0, 2, 0, 1)

              call temperature_e(xpoint2, xcase2, y_g, Z_xpoint, psi_norm, 0.d0, 1.d0, &
                                 T0_e, T0_e_s, dummy1, T0_e_ss, dummy2, dummy3, dummy4, dummy5, dummy6)
              T0_e_s  = T0_e_s  * s_factor
              T0_e_ss = T0_e_ss * s_factor**2
              eq(var_Te_equil, 0, 0, 0, 1) = T0_e
              eq(var_Te_equil, 1, 0, 0, 1) = ( y_t * T0_e_s) / xjac
              eq(var_Te_equil, 0, 1, 0, 1) = (-x_t * T0_e_s) / xjac
              eq(var_Te_equil, 0, 0, 1, 1) = -eq(var_Te_equil, 1, 0, 0, 1) * x_p - eq(var_Te_equil, 0, 1, 0, 1) * y_p
              eq(var_Te_equil, 2, 0, 0, 1) = (T0_e_ss*y_t**2 + T0_e_s*(y_st*y_t - y_tt*y_s)) / xjac**2 - xjac_x*(T0_e_s*y_t) / xjac**2
              eq(var_Te_equil, 0, 2, 0, 1) = (T0_e_ss*x_t**2 + T0_e_s*(x_st*x_t - x_tt*x_s)) / xjac**2 - xjac_y*(-T0_e_s*x_t) / xjac**2
              eq(var_Te_equil, 1, 1, 0, 1) = (-T0_e_ss*y_t*x_t - T0_e_s*(x_st*y_t - x_tt*y_s)) / xjac**2 - xjac_x*(-T0_e_s*x_t) / xjac**2
              eq(var_Te_equil, 1, 0, 1, 1) = -x_p_x*eq(var_Te_equil, 1, 0, 0, 1) - x_p*eq(var_Te_equil, 2, 0, 0, 1) &
                                            - y_p_x*eq(var_Te_equil, 0, 1, 0, 1) - y_p*eq(var_Te_equil, 1, 1, 0, 1)
              eq(var_Te_equil, 0, 1, 1, 1) = -x_p_y*eq(var_Te_equil, 1, 0, 0, 1) - x_p*eq(var_Te_equil, 1, 1, 0, 1) &
                                            - y_p_y*eq(var_Te_equil, 0, 1, 0, 1) - y_p*eq(var_Te_equil, 0, 2, 0, 1)
            else
              call temperature(xpoint2, xcase2, y_g, Z_xpoint, psi_norm, 0.d0, 1.d0, &
                               T0, T0_s, dummy1, T0_ss, dummy2, dummy3, dummy4, dummy5, dummy6)
              T0_s  = T0_s  * s_factor
              T0_ss = T0_ss * s_factor**2
              eq(var_T_equil, 0, 0, 0, 1) = T0
              eq(var_T_equil, 1, 0, 0, 1) = ( y_t * T0_s) / xjac
              eq(var_T_equil, 0, 1, 0, 1) = (-x_t * T0_s) / xjac
              eq(var_T_equil, 0, 0, 1, 1) = -eq(var_T_equil, 1, 0, 0, 1) * x_p - eq(var_T_equil, 0, 1, 0, 1) * y_p
              eq(var_T_equil, 2, 0, 0, 1) = (T0_ss*y_t**2 + T0_s*(y_st*y_t - y_tt*y_s)) / xjac**2 - xjac_x*(T0_s*y_t) / xjac**2
              eq(var_T_equil, 0, 2, 0, 1) = (T0_ss*x_t**2 + T0_s*(x_st*x_t - x_tt*x_s)) / xjac**2 - xjac_y*(-T0_s*x_t) / xjac**2
              eq(var_T_equil, 1, 1, 0, 1) = (-T0_ss*y_t*x_t - T0_s*(x_st*y_t - x_tt*y_s)) / xjac**2 - xjac_x*(-T0_s*x_t) / xjac**2
              eq(var_T_equil, 1, 0, 1, 1) = -x_p_x*eq(var_T_equil, 1, 0, 0, 1) - x_p*eq(var_T_equil, 2, 0, 0, 1) &
                                           - y_p_x*eq(var_T_equil, 0, 1, 0, 1) - y_p*eq(var_T_equil, 1, 1, 0, 1)
              eq(var_T_equil, 0, 1, 1, 1) = -x_p_y*eq(var_T_equil, 1, 0, 0, 1) - x_p*eq(var_T_equil, 1, 1, 0, 1) &
                                           - y_p_y*eq(var_T_equil, 0, 1, 0, 1) - y_p*eq(var_T_equil, 0, 2, 0, 1)
            end if
          end if

          if (with_TiTe) then
            eq(var_k_perp_i, 0, 0, 0, 1) = get_zk_iperp(psi_norm)
            eq(var_k_perp_e, 0, 0, 0, 1) = get_zk_eperp(psi_norm)
            eq(var_S_e_i,    0, 0, 0, 1) = heat_source_i
            eq(var_S_e_e,    0, 0, 0, 1) = heat_source_e

            if (eta_T_dependent) then
              eq(var_eta,     0, 0, 0, 1) = eta * (corr_neg_temp(eq(var_Te, 0, 0, 0, 1)) / Te_0)**(-1.5d0)
              eq(var_deta_dT, 0, 0, 0, 1) = -1.5d0 * eta * corr_neg_temp(eq(var_Te, 0, 0, 0, 1))**(-2.5d0) * Te_0**(1.5d0)
            else
              eq(var_eta,     0, 0, 0, 1) = eta
              eq(var_deta_dT, 0, 0, 0, 1) = 0.d0
            end if

            if (visco_T_dependent) then
              eq(var_visco,     0, 0, 0, 1) = visco * (corr_neg_temp(eq(var_Te, 0, 0, 0, 1)) / Te_0)**(-1.5d0)
              eq(var_dvisco_dT, 0, 0, 0, 1) = -1.5d0 * visco * corr_neg_temp(eq(var_Te, 0, 0, 0, 1))**(-2.5d0) * Te_0**(1.5d0)
            else
              eq(var_visco,     0, 0, 0, 1) = visco
              eq(var_dvisco_dT, 0, 0, 0, 1) = 0.d0
            end if

            if (zkpar_T_dependent) then
              eq(var_k_par_i,     0, 0, 0, 1) = ZK_i_par * (corr_neg_temp(eq(var_Ti, 0, 0, 0, 1)) / Ti_0)**(2.5d0)
              eq(var_dk_par_dT_i, 0, 0, 0, 1) = 2.5d0 * ZK_i_par * corr_neg_temp(eq(var_Ti, 0, 0, 0, 1))**(1.5d0) * Ti_0**(-2.5d0)
              eq(var_k_par_e,     0, 0, 0, 1) = ZK_e_par * (corr_neg_temp(eq(var_Te, 0, 0, 0, 1)) / Te_0)**(2.5d0)
              eq(var_dk_par_dT_e, 0, 0, 0, 1) = 2.5d0 * ZK_e_par * corr_neg_temp(eq(var_Te, 0, 0, 0, 1))**(1.5d0) * Te_0**(-2.5d0)
              if (eq(var_k_par_i, 0, 0, 0, 1) > zk_par_max) then
                eq(var_k_par_i,     0, 0, 0, 1) = zk_par_max
                eq(var_dk_par_dT_i, 0, 0, 0, 1) = 0.d0
              end if
              if (eq(var_k_par_e, 0, 0, 0, 1) > zk_par_max) then
                eq(var_k_par_e,     0, 0, 0, 1) = zk_par_max
                eq(var_dk_par_dT_e, 0, 0, 0, 1) = 0.d0
              end if
            else
              eq(var_k_par_i,     0, 0, 0, 1) = zk_i_par
              eq(var_dk_par_dT_i, 0, 0, 0, 1) = 0.d0
              eq(var_k_par_e,     0, 0, 0, 1) = zk_e_par
              eq(var_dk_par_dT_e, 0, 0, 0, 1) = 0.d0
            end if

            T0_i_corr     = corr_neg_temp(eq(var_Ti,  0, 0, 0, 1))
            T0_e_corr     = corr_neg_temp(eq(var_Te,  0, 0, 0, 1))
            rho0_corr     = corr_neg_dens(eq(var_rho, 0, 0, 0, 1))
            dT0_i_corr_dT = dcorr_neg_temp_dT(eq(var_Ti,  0, 0, 0, 1))
            dT0_e_corr_dT = dcorr_neg_temp_dT(eq(var_Te,  0, 0, 0, 1))
            drho0_corr_dn = dcorr_neg_dens_drho(eq(var_rho, 0, 0, 0, 1))

            T0_e_corr_eV    = T0_e_corr / (EL_CHG * MU_ZERO * central_density * 1.d20)
            T0_i_corr_eV    = T0_i_corr / (EL_CHG * MU_ZERO * central_density * 1.d20)
            dT0_e_corr_eV_dT = dT0_e_corr_dT / (EL_CHG * MU_ZERO * central_density * 1.d20)
            dT0_i_corr_eV_dT = dT0_i_corr_dT / (EL_CHG * MU_ZERO * central_density * 1.d20)

            ne_SI = max(1.d16, rho0_corr * 1.d20 * central_density)
            if ((T0_i_corr * MASS_ELECTRON / (central_mass * ATOMIC_MASS_UNIT) < T0_e_corr) .and. (T0_e_corr_eV < 10.d0)) then
              lambda_e_bg = 23.d0 - log((ne_SI * 1.d-6)**0.5d0 * T0_e_corr_eV**(-1.5d0))
            else if ((T0_i_corr_eV * MASS_ELECTRON / (central_mass * ATOMIC_MASS_UNIT) < 10.d0) .and. (10.d0 < T0_e_corr_eV)) then
              lambda_e_bg = 24.d0 - log((ne_SI * 1.d-6)**0.5d0 * T0_e_corr_eV**(-1.0d0))
            else
              lambda_e_bg = 30.d0 - log((ne_SI * 1.d-6)**0.5d0 * T0_i_corr_eV**(-1.5d0) / MU_ZERO)
            end if
            nu_e_bg = 1.8d-19 * (1.d6 * MASS_ELECTRON * ATOMIC_MASS_UNIT * central_mass)**0.5d0 &
                    * (1.d14 * central_density * rho0_corr) * lambda_e_bg &
                    / (1.d3 * (MASS_ELECTRON * T0_i_corr + T0_e_corr * ATOMIC_MASS_UNIT * central_mass) &
                    / (EL_CHG * MU_ZERO * central_density * 1.d20))**1.5d0
            if (nu_e_bg < 0.d0) nu_e_bg = 0.d0
            t_norm  = sqrt(MU_ZERO * central_mass * ATOMIC_MASS_UNIT * central_density * 1.d20)
            nu_e_bg = nu_e_bg * t_norm

            eq(var_dTe_i, 0, 0, 0, 1) = nu_e_bg * (T0_e_corr - T0_i_corr) * rho0_corr
            dnu_e_bg_dTi  = -1.5d0 * MASS_ELECTRON * nu_e_bg * dT0_i_corr_dT &
                          / (MASS_ELECTRON * T0_i_corr + ATOMIC_MASS_UNIT * central_mass * T0_e_corr)
            dnu_e_bg_dTe  = -1.5d0 * ATOMIC_MASS_UNIT * central_mass * nu_e_bg * dT0_e_corr_dT &
                          / (MASS_ELECTRON * T0_i_corr + ATOMIC_MASS_UNIT * central_mass * T0_e_corr)
            dnu_e_bg_drho = nu_e_bg * drho0_corr_dn / rho0_corr

            eq(var_ddTe_i_dT_i, 0, 0, 0, 1) = dnu_e_bg_dTi  * (T0_e_corr - T0_i_corr) * rho0_corr - nu_e_bg * dT0_i_corr_dT * rho0_corr
            eq(var_ddTe_i_dT_e, 0, 0, 0, 1) = dnu_e_bg_dTe  * (T0_e_corr - T0_i_corr) * rho0_corr + nu_e_bg * dT0_e_corr_dT * rho0_corr
            eq(var_ddTe_i_drho, 0, 0, 0, 1) = dnu_e_bg_drho * (T0_e_corr - T0_i_corr) * rho0_corr + nu_e_bg * (T0_e_corr - T0_i_corr) * drho0_corr_dn
          else
            eq(var_k_perp, 0, 0, 0, 1) = get_zkperp(psi_norm)
            eq(var_S_e,    0, 0, 0, 1) = heat_source

            if (eta_T_dependent) then
              eq(var_eta,     0, 0, 0, 1) = eta * (corr_neg_temp(eq(var_T, 0, 0, 0, 1)) / T_0)**(-1.5d0)
              eq(var_deta_dT, 0, 0, 0, 1) = -1.5d0 * eta * corr_neg_temp(eq(var_T, 0, 0, 0, 1))**(-2.5d0) * T_0**(1.5d0)
            else
              eq(var_eta,     0, 0, 0, 1) = eta
              eq(var_deta_dT, 0, 0, 0, 1) = 0.d0
            end if

            if (visco_T_dependent) then
              eq(var_visco,     0, 0, 0, 1) = visco * (corr_neg_temp(eq(var_T, 0, 0, 0, 1)) / T_0)**(-1.5d0)
              eq(var_dvisco_dT, 0, 0, 0, 1) = -1.5d0 * visco * corr_neg_temp(eq(var_T, 0, 0, 0, 1))**(-2.5d0) * T_0**(1.5d0)
            else
              eq(var_visco,     0, 0, 0, 1) = visco
              eq(var_dvisco_dT, 0, 0, 0, 1) = 0.d0
            end if

            if (zkpar_T_dependent) then
              eq(var_k_par,     0, 0, 0, 1) = zk_par * (corr_neg_temp(eq(var_T, 0, 0, 0, 1)) / T_0)**(2.5d0)
              eq(var_dk_par_dT, 0, 0, 0, 1) = 2.5d0 * zk_par * corr_neg_temp(eq(var_T, 0, 0, 0, 1))**(1.5d0) * T_0**(-2.5d0)
              if (eq(var_k_par, 0, 0, 0, 1) > zk_par_max) then
                eq(var_k_par,     0, 0, 0, 1) = zk_par_max
                eq(var_dk_par_dT, 0, 0, 0, 1) = 0.d0
              end if
            else
              eq(var_k_par,     0, 0, 0, 1) = zk_par
              eq(var_dk_par_dT, 0, 0, 0, 1) = 0.d0
            end if
          end if

          if (eq(var_rho, 0, 0, 0, 1) < D_prof_neg_thresh) eq(var_D_perp, 0, 0, 0, 1) = D_prof_neg
          if (with_TiTe) then
            if (eq(var_Ti, 0, 0, 0, 1) < ZK_prof_neg_thresh) eq(var_k_perp_i, 0, 0, 0, 1) = ZK_prof_neg
            if (eq(var_Te, 0, 0, 0, 1) < ZK_prof_neg_thresh) eq(var_k_perp_e, 0, 0, 0, 1) = ZK_prof_neg
          else
            if (eq(var_T,  0, 0, 0, 1) < ZK_prof_neg_thresh) eq(var_k_perp,   0, 0, 0, 1) = ZK_prof_neg
          end if

          ! Replicate background/state across lanes 1..4 so all denominators in aux/rhs/amat are nonzero
          do m = 2, 4
            eq(:, :, :, :, m) = eq(:, :, :, :, 1)
          end do
          call get_auxiliary(eq)

          if (has_axis_u) then
            call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                      BigR, y_g, phi_q, .true., n_act, act_u, act_io, act_jb, &
                                      bv, bR, bZ, bP, bRR, bRZ, bZZ, bRP, bZP, bPP)
            n_grp = n_act / 4
            do g_a = 1, n_grp
              a0           = 4 * (g_a - 1)
              grp_u(g_a)   = act_u(a0 + 1)
              grp_jb(g_a)  = act_jb(a0 + 1)
              grp_iuj(g_a) = pfafem_cache(ielm)%uj_map(grp_u(g_a), grp_jb(g_a))
            end do
            do a = 1, n_act
              db_mat(1,  a) = bv(a)
              db_mat(2,  a) = bR(a)
              db_mat(3,  a) = bZ(a)
              db_mat(4,  a) = bP(a)
              db_mat(5,  a) = bRR(a)
              db_mat(6,  a) = bZZ(a)
              db_mat(7,  a) = bRZ(a)
              db_mat(8,  a) = bPP(a)
              db_mat(9,  a) = bRP(a)
              db_mat(10, a) = bZP(a)
            end do
          else if (use_qp_c) then
            n_act = n_raw
            do g_a = 1, n_grp
              grp_u(g_a)   = int(pfafem_qp_gu(g_a, ms, mt, iq, ep1, iel_loc))
              grp_jb(g_a)  = int(pfafem_qp_gjb(g_a, ms, mt, iq, ep1, iel_loc))
              grp_iuj(g_a) = int(pfafem_qp_giuj(g_a, ms, mt, iq, ep1, iel_loc))
            end do
            if (n_act > 0) then
              db_mat(1:10, 1:n_act) = pfafem_qp_db(1:10, 1:n_act, ms, mt, iq, ep1, iel_loc)
            end if
            if (nu_phi_source /= 0.d0) then
              do g_a = 1, n_grp
                a0 = 4 * (g_a - 1)
                do io_a = 1, 4
                  a          = a0 + io_a
                  act_u(a)   = grp_u(g_a)
                  act_io(a)  = io_a
                  act_jb(a)  = grp_jb(g_a)
                  bv(a)  = db_mat(1,  a); bR(a)  = db_mat(2,  a); bZ(a)  = db_mat(3,  a); bP(a)  = db_mat(4,  a)
                  bRR(a) = db_mat(5,  a); bZZ(a) = db_mat(6,  a); bRZ(a) = db_mat(7,  a)
                  bPP(a) = db_mat(8,  a); bRP(a) = db_mat(9,  a); bZP(a) = db_mat(10, a)
                end do
              end do
            end if
          else
            n_act = n_raw
            n_grp = n_act / 4
            act_u(1:n_act)  = raw_u(1:n_act)
            act_io(1:n_act) = raw_io(1:n_act)
            act_jb(1:n_act) = raw_jb(1:n_act)
            bv(1:n_act)  = rv(1:n_act);  bR(1:n_act)  = rR(1:n_act);  bZ(1:n_act)  = rZ(1:n_act);  bP(1:n_act)  = rP(1:n_act)
            bRR(1:n_act) = rRR(1:n_act); bRZ(1:n_act) = rRZ(1:n_act); bZZ(1:n_act) = rZZ(1:n_act)
            bRP(1:n_act) = rRP(1:n_act); bZP(1:n_act) = rZP(1:n_act); bPP(1:n_act) = rPP(1:n_act)
            do g_a = 1, n_grp
              a0           = 4 * (g_a - 1)
              grp_u(g_a)   = act_u(a0 + 1)
              grp_jb(g_a)  = act_jb(a0 + 1)
              grp_iuj(g_a) = pfafem_cache(ielm)%uj_map(grp_u(g_a), grp_jb(g_a))
            end do
            do a = 1, n_act
              db_mat(1,  a) = bv(a)
              db_mat(2,  a) = bR(a)
              db_mat(3,  a) = bZ(a)
              db_mat(4,  a) = bP(a)
              db_mat(5,  a) = bRR(a)
              db_mat(6,  a) = bZZ(a)
              db_mat(7,  a) = bRZ(a)
              db_mat(8,  a) = bPP(a)
              db_mat(9,  a) = bRP(a)
              db_mat(10, a) = bZP(a)
            end do
          end if

          if (nu_phi_source == 0.d0 .and. n_act > 0) then
            eq(var_v,           :, :, :, :) = 0.d0
            eq(var_varStar,     :, :, :, :) = 0.d0
            eq(var_varStar_pol, :, :, :, :) = 0.d0
            R_p  = 0.d0
            C_pq = 0.d0

            ! Phase 10: In model183, 2nd derivatives (slots 5..10) appear ONLY via Lap(v)*Lap(u)
            ! where Lap(f) = f_R/R + f_RR + f_ZZ + f_PP/R^2 (slots 2, 5, 6, 8; slots 7, 9, 10 == 0).
            ! Evaluate 1st-order RHS (lanes 1..4) + 1 Lap(v) probe (slot 5) in 2 calls instead of 3:
            do lane_k = 1, 4
              eq(var_v, d_ix(1, lane_k), d_ix(2, lane_k), d_ix(3, lane_k), lane_k) = 1.d0
            end do
            rhs_ij = 0.d0
            call get_rhs(rhs_ij, eq)
            do lane_k = 1, 4
              R_p(:, lane_k) = rhs_ij(:, lane_k)
              eq(var_v, d_ix(1, lane_k), d_ix(2, lane_k), d_ix(3, lane_k), lane_k) = 0.d0
            end do

            eq(var_v, 2, 0, 0, 1) = 1.d0
            rhs_ij = 0.d0
            call get_rhs(rhs_ij, eq)
            eq(var_v, 2, 0, 0, 1) = 0.d0
            r_val = 1.d0 / (BigR * BigR)
            R_p(:, 5) = rhs_ij(:, 1)
            R_p(:, 6) = rhs_ij(:, 1)
            R_p(:, 8) = rhs_ij(:, 1) * r_val

            ! Evaluate 4x4 = 16 1st-order Jacobian pairs (4 calls) + 1 Lap(v)*Lap(u) probe (1 call)
            ! instead of 25 calls (5.0x fewer get_amat evaluations, algebraically identical):
            do idx_pq = 1, 16, 4
              do lane_k = 1, 4
                pq_k = idx_pq + lane_k - 1
                ip_d = (pq_k - 1) / 4 + 1
                iq_d = mod(pq_k - 1, 4) + 1
                eq(var_v,       d_ix(1, ip_d), d_ix(2, ip_d), d_ix(3, ip_d), lane_k) = 1.d0
                eq(var_varStar, d_ix(1, iq_d), d_ix(2, iq_d), d_ix(3, iq_d), lane_k) = 1.d0
              end do
              amat_ij = 0.d0
              call get_amat(amat_ij, eq)
              do lane_k = 1, 4
                pq_k = idx_pq + lane_k - 1
                ip_d = (pq_k - 1) / 4 + 1
                iq_d = mod(pq_k - 1, 4) + 1
                C_pq(:, :, ip_d, iq_d) = amat_ij(:, :, lane_k)
                eq(var_v,       d_ix(1, ip_d), d_ix(2, ip_d), d_ix(3, ip_d), lane_k) = 0.d0
                eq(var_varStar, d_ix(1, iq_d), d_ix(2, iq_d), d_ix(3, iq_d), lane_k) = 0.d0
              end do
            end do

            eq(var_v,       2, 0, 0, 1) = 1.d0
            eq(var_varStar, 2, 0, 0, 1) = 1.d0
            amat_ij = 0.d0
            call get_amat(amat_ij, eq)
            eq(var_v,       2, 0, 0, 1) = 0.d0
            eq(var_varStar, 2, 0, 0, 1) = 0.d0
            amat_ab = amat_ij(:, :, 1)
            if (any(amat_ab /= 0.d0)) then
              db_val = 1.d0 / BigR
              r_val  = db_val * db_val
              C_pq(:, :, 2, 5) = amat_ab * db_val
              C_pq(:, :, 2, 6) = amat_ab * db_val
              C_pq(:, :, 2, 8) = amat_ab * (db_val * r_val)
              C_pq(:, :, 5, 2) = amat_ab * db_val
              C_pq(:, :, 5, 5) = amat_ab
              C_pq(:, :, 5, 6) = amat_ab
              C_pq(:, :, 5, 8) = amat_ab * r_val
              C_pq(:, :, 6, 2) = amat_ab * db_val
              C_pq(:, :, 6, 5) = amat_ab
              C_pq(:, :, 6, 6) = amat_ab
              C_pq(:, :, 6, 8) = amat_ab * r_val
              C_pq(:, :, 8, 2) = amat_ab * (db_val * r_val)
              C_pq(:, :, 8, 5) = amat_ab * r_val
              C_pq(:, :, 8, 6) = amat_ab * r_val
              C_pq(:, :, 8, 8) = amat_ab * (r_val * r_val)
            end if

            ! Accumulate RHS_u and pre-filter active (i_var, j_var) blocks + derivative slots
            do g_a = 1, n_grp
              u_a  = grp_u(g_a)
              jb_a = grp_jb(g_a)
              a0   = 4 * (g_a - 1)
              do io_a = 1, 4
                a = a0 + io_a
                do i_var = 1, n_var
                  r_val = dot_product(R_p(i_var, 1:10), db_mat(1:10, a))
                  r_loc = (i_var - 1) * n_tor + jb_a
                  RHS_u(r_loc, io_a, u_a) = RHS_u(r_loc, io_a, u_a) + wvol * r_val
                end do
              end do
            end do

            n_ij = 0
            do j_var = 1, n_var
              if ((j_var .eq. var_Psi) .or. (j_var .eq. var_zj)) then
                prefactor = wvol / F0
              else
                prefactor = wvol
              end if
              do i_var = 1, n_var
                sq = 0
                do iq_d = 1, 10
                  if (any(C_pq(i_var, j_var, 1:10, iq_d) /= 0.d0)) then
                    if (sq == 0) then
                      n_ij = n_ij + 1
                      iv_ij(n_ij) = i_var
                      jv_ij(n_ij) = j_var
                    end if
                    sq = sq + 1
                    q_ij(sq, n_ij) = iq_d
                    do a = 1, n_act
                      T_kqa(a, sq, n_ij) = prefactor * dot_product(C_pq(i_var, j_var, 1:10, iq_d), db_mat(1:10, a))
                    end do
                  end if
                end do
                if (sq > 0) nq_ij(n_ij) = sq
              end do
            end do

            ! Unit-stride register-blocked 4x4 contraction into sparse (u, jb) pair slot isp
            do g_a = 1, n_grp
              a0   = 4 * (g_a - 1)
              iuja = grp_iuj(g_a)
              do g_b = 1, n_grp
                b0   = 4 * (g_b - 1)
                iujb = grp_iuj(g_b)
                isp  = pfafem_cache(ielm)%sp_map(iuja, iujb)
                do k_ij = 1, n_ij
                  m44 = 0.d0
                  do sq = 1, nq_ij(k_ij)
                    iq_d = q_ij(sq, k_ij)
                    do io_b = 1, 4
                      db_val = db_mat(iq_d, b0 + io_b)
                      do io_a = 1, 4
                        m44(io_a, io_b) = m44(io_a, io_b) + T_kqa(a0 + io_a, sq, k_ij) * db_val
                      end do
                    end do
                  end do
                  ELM_sp(1:4, 1:4, iv_ij(k_ij), jv_ij(k_ij), isp) = &
                    ELM_sp(1:4, 1:4, iv_ij(k_ij), jv_ij(k_ij), isp) + m44
                end do
              end do
            end do
          else
            do a = 1, n_act
              eq(var_v, :, :, :, :) = 0.d0
              eq(var_v, 0, 0, 0, 1) = bv(a)
              eq(var_v, 1, 0, 0, 1) = bR(a)
              eq(var_v, 0, 1, 0, 1) = bZ(a)
              eq(var_v, 0, 0, 1, 1) = bP(a)
              eq(var_v, 2, 0, 0, 1) = bRR(a)
              eq(var_v, 0, 2, 0, 1) = bZZ(a)
              eq(var_v, 1, 1, 0, 1) = bRZ(a)
              eq(var_v, 0, 0, 2, 1) = bPP(a)
              eq(var_v, 1, 0, 1, 1) = bRP(a)
              eq(var_v, 0, 1, 1, 1) = bZP(a)

              call get_rhs(rhs_ij, eq)
              do i_var = 1, n_var
                r_loc = (i_var - 1) * n_tor + act_jb(a)
                RHS_u(r_loc, act_io(a), act_u(a)) = RHS_u(r_loc, act_io(a), act_u(a)) + wvol * rhs_ij(i_var, 1)
              end do

              iuja = pfafem_cache(ielm)%uj_map(act_u(a), act_jb(a))
              do b = 1, n_act
                eq(var_varStar,     :, :, :, :) = 0.d0
                eq(var_varStar_pol, :, :, :, :) = 0.d0

                eq(var_varStar, 0, 0, 0, 1) = bv(b)
                eq(var_varStar, 1, 0, 0, 1) = bR(b)
                eq(var_varStar, 0, 1, 0, 1) = bZ(b)
                eq(var_varStar, 0, 0, 1, 1) = bP(b)
                eq(var_varStar, 2, 0, 0, 1) = bRR(b)
                eq(var_varStar, 0, 2, 0, 1) = bZZ(b)
                eq(var_varStar, 1, 1, 0, 1) = bRZ(b)
                eq(var_varStar, 0, 0, 2, 1) = bPP(b)
                eq(var_varStar, 1, 0, 1, 1) = bRP(b)
                eq(var_varStar, 0, 1, 1, 1) = bZP(b)

                if (nu_phi_source /= 0.d0) then
                  b_s    = x_s * bR(b) + y_s * bZ(b)
                  b_p_st = bP(b) + x_p * bR(b) + y_p * bZ(b)
                  eq(var_varStar_pol, 0, 0, 0, 1) = bv(b)
                  eq(var_varStar_pol, 1, 0, 0, 1) = ( y_t * b_s) / xjac
                  eq(var_varStar_pol, 0, 1, 0, 1) = (-x_t * b_s) / xjac
                  eq(var_varStar_pol, 0, 0, 1, 1) = b_p_st - eq(var_varStar_pol, 1, 0, 0, 1) * x_p - eq(var_varStar_pol, 0, 1, 0, 1) * y_p
                end if

                call get_amat(amat_ij, eq)

                iujb = pfafem_cache(ielm)%uj_map(act_u(b), act_jb(b))
                isp  = pfafem_cache(ielm)%sp_map(iuja, iujb)
                do j_var = 1, n_var
                  if ((j_var .eq. var_Psi) .or. (j_var .eq. var_zj)) then
                    prefactor = wvol / F0
                  else
                    prefactor = wvol
                  end if
                  do i_var = 1, n_var
                    ELM_sp(act_io(a), act_io(b), i_var, j_var, isp) = &
                      ELM_sp(act_io(a), act_io(b), i_var, j_var, isp) + prefactor * amat_ij(i_var, j_var, 1)
                  end do
                end do
              end do
            end do
          end if

        end do
      end do
    end do
  end do

end subroutine element_matrix_pfafem

end module mod_elt_matrix_fft
