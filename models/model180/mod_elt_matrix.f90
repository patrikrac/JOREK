module mod_elt_matrix
  implicit none
contains

subroutine element_matrix(element,nodes, xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, ELM, RHS, tid, i_tor_min, i_tor_max, aux_nodes)
  !---------------------------------------------------------------
  ! calculates the matrix contribution of one element
  !---------------------------------------------------------------
use constants
use mod_parameters
use data_structure
use gauss
use basis_at_gaussian
use phys_module
use tr_module
use diffusivities, only: get_dperp, get_zkperp    
use corr_neg
use equil_info, only: get_psi_n
use mod_semianalytical
use mod_equations
use mod_chi

implicit none

type (type_element), intent(in)           :: element
type (type_node),    intent(in)           :: nodes(n_vertex_max)
type (type_node),    intent(in), optional :: aux_nodes(n_vertex_max)

real*8, dimension (:,:), allocatable  :: ELM
real*8, dimension (:)  , allocatable  :: RHS
integer, intent(in) :: tid, i_tor_min, i_tor_max

integer    :: i, j, ms, mt, mp, k, l, index_ij, index_kl, index, xcase2
integer    :: in, im, ij, kl, i_var, j_var
real*8     :: wst, prefactor, xjac, xjac_x, xjac_y, x_p_x, x_p_y, y_p_x, y_p_y, BigR, phi
real*8     :: R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint(2), Z_xpoint(2)
real*8     :: s_norm_s, psi_norm, psi_norm_R, psi_norm_Z, psi_norm_p, reta, zeta, theta
real*8     :: v_px, v_py, u_px, u_py
real*8     :: BR_R, BR_z, BR_p, Bz_R, Bz_z, Bz_p, Bp_R, Bp_z, Bp_p

real*8, dimension(n_var)        :: rhs_ij
real*8, dimension(n_var, n_var) :: amat_ij

logical    :: xpoint2
integer    :: n_tor_local

real*8, dimension(n_plane,n_gauss,n_gauss) :: x_g, x_s, x_t, x_p, x_ss, x_st, x_tt, x_sp, x_tp, x_pp
real*8, dimension(n_plane,n_gauss,n_gauss) :: y_g, y_s, y_t, y_p, y_ss, y_st, y_tt, y_sp, y_tp, y_pp
real*8, dimension(n_gauss, n_gauss)        :: s_norm

real*8, dimension(:,:,:,:) , pointer :: eq_g, eq_s, eq_t
real*8, dimension(:,:,:,:) , pointer :: eq_st, eq_ss, eq_tt
real*8, dimension(:,:,:,:) , pointer :: eq_p, eq_pp, eq_sp, eq_tp

real*8, dimension(:,:,:,:,:), pointer :: eq
real*8, dimension(n_var)            :: eq_px, eq_py

real*8, dimension(n_gauss,n_gauss) :: press_gvec
real*8, dimension(n_dim+1,n_plane,n_gauss,n_gauss) :: B_gvec, B_gvec_s, B_gvec_t, B_gvec_p
           

eq_g    => thread_struct(tid)%eq_g
eq_s    => thread_struct(tid)%eq_s
eq_t    => thread_struct(tid)%eq_t
eq_p    => thread_struct(tid)%eq_p
eq_pp   => thread_struct(tid)%eq_pp
eq_sp   => thread_struct(tid)%eq_sp
eq_tp   => thread_struct(tid)%eq_tp
eq_ss   => thread_struct(tid)%eq_ss
eq_st   => thread_struct(tid)%eq_st
eq_tt   => thread_struct(tid)%eq_tt

eq => thread_eq(tid)%eq

rhs_ij = 0.d0
amat_ij = 0.d0
ELM = 0.d0
RHS = 0.d0

! --- Take time evolution parameters from phys_module
theta = time_evol_theta
! change zeta for variable dt
zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)

s_norm_s = 1.d0 * bloating_factor /float(n_flux-1)

!---------------------------------------------------- value of (x,y) and derivatives on Gaussian points
x_g  = 0.d0; x_s   = 0.d0; x_t   = 0.d0; x_p = 0.d0; x_st  = 0.d0; x_ss  = 0.d0; x_tt  = 0.d0; x_sp = 0.d0; x_tp = 0.d0; x_pp = 0.d0;
y_g  = 0.d0; y_s   = 0.d0; y_t   = 0.d0; y_p = 0.d0; y_st  = 0.d0; y_ss  = 0.d0; y_tt  = 0.d0; y_sp = 0.d0; y_tp = 0.d0; y_pp = 0.d0;
eq_g = 0.d0; eq_s  = 0.d0; eq_t  = 0.d0; eq_st = 0.d0; eq_ss = 0.d0; eq_tt = 0.d0;
eq_p = 0.d0; eq_pp = 0.d0; eq_sp = 0.d0; eq_tp = 0.d0

eq = 0.d0
press_gvec = 0.d0; B_gvec = 0.d0; B_gvec_s(:,:,:,:) = 0.d0; B_gvec_t(:,:,:,:) = 0.d0; B_gvec_p(:,:,:,:) = 0.d0
s_norm = 0.d0

do i=1,n_vertex_max
 do j=1,n_order+1

   do ms=1, n_gauss
     do mt=1, n_gauss

       press_gvec(ms,mt) = press_gvec(ms,mt) + nodes(i)%pressure(j)*element%size(i,j)*H(i,j,ms,mt)
       s_norm(ms,mt)     = s_norm(ms,mt)     + nodes(i)%r_tor_eq(j)*element%size(i,j)*H(i,j,ms,mt)

       do mp=1,n_plane
         do in=1,n_coord_tor
           x_g(mp,ms,mt)  = x_g(mp,ms,mt)  + nodes(i)%x(in,j,1) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord(in,mp)
           x_s(mp,ms,mt)  = x_s(mp,ms,mt)  + nodes(i)%x(in,j,1) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord(in,mp)
           x_t(mp,ms,mt)  = x_t(mp,ms,mt)  + nodes(i)%x(in,j,1) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord(in,mp)
           x_p(mp,ms,mt)  = x_p(mp,ms,mt)  + nodes(i)%x(in,j,1) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_p(in,mp)
           x_ss(mp,ms,mt) = x_ss(mp,ms,mt) + nodes(i)%x(in,j,1) * element%size(i,j) * H_ss(i,j,ms,mt) * HZ_coord(in,mp)
           x_st(mp,ms,mt) = x_st(mp,ms,mt) + nodes(i)%x(in,j,1) * element%size(i,j) * H_st(i,j,ms,mt) * HZ_coord(in,mp)
           x_tt(mp,ms,mt) = x_tt(mp,ms,mt) + nodes(i)%x(in,j,1) * element%size(i,j) * H_tt(i,j,ms,mt) * HZ_coord(in,mp)
           x_sp(mp,ms,mt) = x_sp(mp,ms,mt) + nodes(i)%x(in,j,1) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord_p(in,mp)
           x_tp(mp,ms,mt) = x_tp(mp,ms,mt) + nodes(i)%x(in,j,1) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord_p(in,mp)
           x_pp(mp,ms,mt) = x_pp(mp,ms,mt) + nodes(i)%x(in,j,1) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_pp(in,mp)

           y_g(mp,ms,mt)  = y_g(mp,ms,mt)  + nodes(i)%x(in,j,2) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord(in,mp)
           y_s(mp,ms,mt)  = y_s(mp,ms,mt)  + nodes(i)%x(in,j,2) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord(in,mp)
           y_t(mp,ms,mt)  = y_t(mp,ms,mt)  + nodes(i)%x(in,j,2) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord(in,mp)
           y_p(mp,ms,mt)  = y_p(mp,ms,mt)  + nodes(i)%x(in,j,2) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_p(in,mp)
           y_ss(mp,ms,mt) = y_ss(mp,ms,mt) + nodes(i)%x(in,j,2) * element%size(i,j) * H_ss(i,j,ms,mt) * HZ_coord(in,mp)
           y_st(mp,ms,mt) = y_st(mp,ms,mt) + nodes(i)%x(in,j,2) * element%size(i,j) * H_st(i,j,ms,mt) * HZ_coord(in,mp)
           y_tt(mp,ms,mt) = y_tt(mp,ms,mt) + nodes(i)%x(in,j,2) * element%size(i,j) * H_tt(i,j,ms,mt) * HZ_coord(in,mp)
           y_sp(mp,ms,mt) = y_sp(mp,ms,mt) + nodes(i)%x(in,j,2) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord_p(in,mp)
           y_tp(mp,ms,mt) = y_tp(mp,ms,mt) + nodes(i)%x(in,j,2) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord_p(in,mp)
           y_pp(mp,ms,mt) = y_pp(mp,ms,mt) + nodes(i)%x(in,j,2) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_pp(in,mp)

           B_gvec(1,mp,ms,mt) = B_gvec(1,mp,ms,mt) + nodes(i)%b_field(in,j,1)*element%size(i,j)*H(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec(2,mp,ms,mt) = B_gvec(2,mp,ms,mt) + nodes(i)%b_field(in,j,2)*element%size(i,j)*H(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec(3,mp,ms,mt) = B_gvec(3,mp,ms,mt) + nodes(i)%b_field(in,j,3)*element%size(i,j)*H(i,j,ms,mt)*HZ_coord(in,mp)

           ! Interpolate s, t, and zeta derivatives
           
           B_gvec_s(1,mp,ms,mt) = B_gvec_s(1,mp,ms,mt) + nodes(i)%b_field(in,j,1)*element%size(i,j)*H_s(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec_t(1,mp,ms,mt) = B_gvec_t(1,mp,ms,mt) + nodes(i)%b_field(in,j,1)*element%size(i,j)*H_t(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec_p(1,mp,ms,mt) = B_gvec_p(1,mp,ms,mt) + nodes(i)%b_field(in,j,1)*element%size(i,j)*H(i,j,ms,mt)*HZ_coord_p(in,mp)
           
           B_gvec_s(2,mp,ms,mt) = B_gvec_s(2,mp,ms,mt) + nodes(i)%b_field(in,j,2)*element%size(i,j)*H_s(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec_t(2,mp,ms,mt) = B_gvec_t(2,mp,ms,mt) + nodes(i)%b_field(in,j,2)*element%size(i,j)*H_t(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec_p(2,mp,ms,mt) = B_gvec_p(2,mp,ms,mt) + nodes(i)%b_field(in,j,2)*element%size(i,j)*H(i,j,ms,mt)*HZ_coord_p(in,mp)
          
           B_gvec_s(3,mp,ms,mt) = B_gvec_s(3,mp,ms,mt) + nodes(i)%b_field(in,j,3)*element%size(i,j)*H_s(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec_t(3,mp,ms,mt) = B_gvec_t(3,mp,ms,mt) + nodes(i)%b_field(in,j,3)*element%size(i,j)*H_t(i,j,ms,mt)*HZ_coord(in,mp)
           B_gvec_p(3,mp,ms,mt) = B_gvec_p(3,mp,ms,mt) + nodes(i)%b_field(in,j,3)*element%size(i,j)*H(i,j,ms,mt)*HZ_coord_p(in,mp)
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
           enddo
         enddo
       enddo
     enddo
   enddo
 enddo
enddo

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

     ! Values
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
     eq(1:n_var,0,0,2,1) = eq_pp(mp,:,ms,mt) - x_pp(mp,ms,mt)*eq(1:n_var,1,0,0,1) - 2.d0*(x_p(mp,ms,mt)*eq_px + y_p(mp,ms,mt)*eq_py)                   &
                         - y_pp(mp,ms,mt)*eq(1:n_var,0,1,0,1) + 2.d0*(x_p(mp,ms,mt)*x_p_x*eq(1:n_var,1,0,0,1) + x_p(mp,ms,mt)*y_p_x*eq(1:n_var,0,1,0,1)&
                         + y_p(mp,ms,mt)*x_p_y*eq(1:n_var,1,0,0,1) + y_p(mp,ms,mt)*y_p_y*eq(1:n_var,0,1,0,1)) + x_p(mp,ms,mt)**2*eq(1:n_var,2,0,0,1)   &
                         + 2.d0*x_p(mp,ms,mt)*y_p(mp,ms,mt)*eq(1:n_var,1,1,0,1) + y_p(mp,ms,mt)**2*eq(1:n_var,0,2,0,1)
     eq(1:n_var,1,0,1,1) = eq_px - x_p_x*eq(1:n_var,1,0,0,1) - x_p(mp,ms,mt)*eq(1:n_var,2,0,0,1) - y_p_x*eq(1:n_var,0,1,0,1) - y_p(mp,ms,mt)*eq(1:n_var,1,1,0,1)
     eq(1:n_var,0,1,1,1) = eq_py - x_p_y*eq(1:n_var,1,0,0,1) - x_p(mp,ms,mt)*eq(1:n_var,1,1,0,1) - y_p_y*eq(1:n_var,0,1,0,1) - y_p(mp,ms,mt)*eq(1:n_var,0,2,0,1)

     eq(var_chi,:,:,:,1) = element%chi(mp,ms,mt,:,:,:) ! Vacuum scalar magnetic potential (chi) and field (grad chi)
     eq(  var_R,0,0,0,1) = x_g(mp,ms,mt); eq(  var_R,1,0,0,1) = 1.d0 ! Cylindrical R coordinate
     
     eq(var_p0_gvec,0,0,0,1) = mu_zero*press_gvec(ms,mt)  ! Pressure, as imported from GVEC
     eq(var_B0x_gvec:var_B0p_gvec,0,0,0,1) = B_gvec(:,mp,ms,mt) ! Magnetic field, as imported from GVEC
     
     ! Compute R, Z, phi derivatives
     BR_R = ( y_t(mp,ms,mt)*B_gvec_s(1,mp,ms,mt) - y_s(mp,ms,mt)*B_gvec_t(1,mp,ms,mt))/xjac
     BR_z = (-x_t(mp,ms,mt)*B_gvec_s(1,mp,ms,mt) + x_s(mp,ms,mt)*B_gvec_t(1,mp,ms,mt))/xjac
     BR_p = B_gvec_p(1,mp,ms,mt) - x_p(mp,ms,mt)*BR_R - y_p(mp,ms,mt)*BR_z
     eq(var_B0x_gvec,1,0,0,1) = BR_R   
     eq(var_B0x_gvec,0,1,0,1) = BR_z  
     eq(var_B0x_gvec,0,0,1,1) = BR_p

     Bz_R = ( y_t(mp,ms,mt)*B_gvec_s(2,mp,ms,mt) - y_s(mp,ms,mt)*B_gvec_t(2,mp,ms,mt))/xjac
     Bz_z = (-x_t(mp,ms,mt)*B_gvec_s(2,mp,ms,mt) + x_s(mp,ms,mt)*B_gvec_t(2,mp,ms,mt))/xjac
     Bz_p = B_gvec_p(2,mp,ms,mt) - x_p(mp,ms,mt)*Bz_R - y_p(mp,ms,mt)*Bz_z
     eq(var_B0y_gvec,1,0,0,1) = Bz_R     
     eq(var_B0y_gvec,0,1,0,1) = Bz_z
     eq(var_B0y_gvec,0,0,1,1) = Bz_p

     Bp_R = ( y_t(mp,ms,mt)*B_gvec_s(3,mp,ms,mt) - y_s(mp,ms,mt)*B_gvec_t(3,mp,ms,mt))/xjac
     Bp_z = (-x_t(mp,ms,mt)*B_gvec_s(3,mp,ms,mt) + x_s(mp,ms,mt)*B_gvec_t(3,mp,ms,mt))/xjac
     Bp_p = B_gvec_p(3,mp,ms,mt) - x_p(mp,ms,mt)*Bp_R - y_p(mp,ms,mt)*Bp_z
     eq(var_B0p_gvec,1,0,0,1) = Bp_R     
     eq(var_B0p_gvec,0,1,0,1) = Bp_z
     eq(var_B0p_gvec,0,0,1,1) = Bp_p

     psi_norm = s_norm(ms,mt)
     psi_norm_R = y_t(mp,ms,mt)*s_norm_s/xjac
     psi_norm_Z = -x_t(mp,ms,mt)*s_norm_s/xjac
     psi_norm_p = -x_p(mp,ms,mt)*psi_norm_R       - y_p(mp,ms,mt)*psi_norm_Z
     eq(var_heaviside,0,0,0,1) = 0.5-0.5*tanh((psi_norm-j_cutoff_rcoord)/j_cutoff_sig)
     eq(var_heaviside,1,0,0,1) = -psi_norm_R/(2*j_cutoff_sig * (cosh((psi_norm-j_cutoff_rcoord)/j_cutoff_sig)**2))
     eq(var_heaviside,0,1,0,1) = -psi_norm_Z/(2*j_cutoff_sig * (cosh((psi_norm-j_cutoff_rcoord)/j_cutoff_sig)**2))
     eq(var_heaviside,0,0,1,1) = -psi_norm_p/(2*j_cutoff_sig * (cosh((psi_norm-j_cutoff_rcoord)/j_cutoff_sig)**2))


     ! The Psi in the equations differs by a factor of F0 from the normal JOREK Psi
     eq(var_Psi,:,:,:,1) = eq(var_Psi,:,:,:,1)/F0
     eq( var_zj,:,:,:,1) = eq( var_zj,:,:,:,1)/F0

     ! Auxiliary variables (aux)
#include "aux_automatic.h"
     
     n_tor_local = i_tor_max - i_tor_min + 1
     do i=1,n_vertex_max

       do j=1,n_order+1

         do im=i_tor_min,i_tor_max

           index_ij = n_tor_local*n_var*(n_order+1)*(i-1) + n_tor_local*n_var*(j-1) + im - i_tor_min + 1   ! index in the ELM matrix

           ! Test function (v)
           eq(var_v,0,0,0,1) =  H(i,j,ms,mt)*element%size(i,j)*HZ(im,mp)
           eq(var_v,1,0,0,1) = (y_t(mp,ms,mt)*h_s(i,j,ms,mt) - y_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)*HZ(im,mp)/xjac
           eq(var_v,0,1,0,1) = (-x_t(mp,ms,mt)*h_s(i,j,ms,mt) + x_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)*HZ(im,mp)/xjac
           eq(var_v,0,0,1,1) = H(i,j,ms,mt)*element%size(i,j)*HZ_p(im,mp) - eq(n_var+1,1,0,0,1)*x_p(mp,ms,mt) - eq(n_var+1,0,1,0,1)*y_p(mp,ms,mt)
           eq(var_v,2,0,0,1) = (h_ss(i,j,ms,mt)*y_t(mp,ms,mt)**2 - 2.d0*h_st(i,j,ms,mt)*y_s(mp,ms,mt)*y_t(mp,ms,mt)                             &
	                           + h_tt(i,j,ms,mt)*y_s(mp,ms,mt)**2                                                                                 &
	                           + h_s(i,j,ms,mt)*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt))                                     &
                             + h_t(i,j,ms,mt)*(y_st(mp,ms,mt)*y_s(mp,ms,mt) - y_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(i,j)*HZ(im,mp)/xjac**2&
                             - xjac_x*(h_s(i,j,ms,mt)*y_t(mp,ms,mt) - h_t(i,j,ms,mt)*y_s(mp,ms,mt))*element%size(i,j)*HZ(im,mp)/xjac**2
           eq(var_v,0,2,0,1) = (h_ss(i,j,ms,mt)*x_t(mp,ms,mt)**2 - 2.d0*h_st(i,j,ms,mt)*x_s(mp,ms,mt)*x_t(mp,ms,mt)                             &
                             + h_tt(i,j,ms,mt)*x_s(mp,ms,mt)**2                                                                                 &
                             + h_s(i,j,ms,mt)*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt))                                     &
                             + h_t(i,j,ms,mt)*(x_st(mp,ms,mt)*x_s(mp,ms,mt) - x_ss(mp,ms,mt)*x_t(mp,ms,mt)))*element%size(i,j)*HZ(im,mp)/xjac**2&
           	                 - xjac_y*(-h_s(i,j,ms,mt)*x_t(mp,ms,mt) + h_t(i,j,ms,mt)*x_s(mp,ms,mt))*element%size(i,j)*HZ(im,mp)/xjac**2
           eq(var_v,1,1,0,1) = (-h_ss(i,j,ms,mt)*y_t(mp,ms,mt)*x_t(mp,ms,mt) - h_tt(i,j,ms,mt)*x_s(mp,ms,mt)*y_s(mp,ms,mt)                      &
       	                     + h_st(i,j,ms,mt)*(y_s(mp,ms,mt)*x_t(mp,ms,mt) + y_t(mp,ms,mt)*x_s(mp,ms,mt))                                      &
                             - h_s(i,j,ms,mt)*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt))                                     &
                             - h_t(i,j,ms,mt)*(x_st(mp,ms,mt)*y_s(mp,ms,mt) - x_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(i,j)*HZ(im,mp)/xjac**2&
                             - xjac_x*(-h_s(i,j,ms,mt)*x_t(mp,ms,mt) + h_t(i,j,ms,mt)*x_s(mp,ms,mt))*element%size(i,j)*HZ(im,mp)/xjac**2
           v_px              = (y_t(mp,ms,mt)*h_s(i,j,ms,mt) - y_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)*HZ_p(im,mp)/xjac
           v_py              = (-x_t(mp,ms,mt)*h_s(i,j,ms,mt) + x_s(mp,ms,mt)*h_t(i,j,ms,mt))*element%size(i,j)*HZ_p(im,mp)/xjac
           eq(var_v,0,0,2,1) = H(i,j,ms,mt)*element%size(i,j)*HZ_pp(im,mp) - x_pp(mp,ms,mt)*eq(n_var+1,1,0,0,1) - 2.d0*(x_p(mp,ms,mt)*v_px &
                             + y_p(mp,ms,mt)*v_py) - y_pp(mp,ms,mt)*eq(n_var+1,0,1,0,1) + 2.d0*(x_p(mp,ms,mt)*x_p_x*eq(n_var+1,1,0,0,1) &
                             + x_p(mp,ms,mt)*y_p_x*eq(n_var+1,0,1,0,1) + y_p(mp,ms,mt)*x_p_y*eq(n_var+1,1,0,0,1) + y_p(mp,ms,mt)*y_p_y*eq(n_var+1,0,1,0,1)) &
                             + x_p(mp,ms,mt)**2*eq(n_var+1,2,0,0,1) + 2.d0*x_p(mp,ms,mt)*y_p(mp,ms,mt)*eq(n_var+1,1,1,0,1) &
                             + y_p(mp,ms,mt)**2*eq(n_var+1,0,2,0,1)
           eq(var_v,1,0,1,1) = v_px - x_p_x*eq(n_var+1,1,0,0,1) - x_p(mp,ms,mt)*eq(n_var+1,2,0,0,1) - y_p_x*eq(n_var+1,0,1,0,1) - y_p(mp,ms,mt)*eq(n_var+1,1,1,0,1)
           eq(var_v,0,1,1,1) = v_py - x_p_y*eq(n_var+1,1,0,0,1) - y_p(mp,ms,mt)*eq(n_var+1,0,2,0,1) - y_p_y*eq(n_var+1,0,1,0,1) - x_p(mp,ms,mt)*eq(n_var+1,1,1,0,1)

#include "rhs_automatic.h"
           do i_var=1,n_var
             rhs_ij(i_var) = rhs_ij(i_var)*wst*BigR*xjac
           enddo

           do i_var = 1, n_var
                RHS(index_ij+(i_var-1)*(n_tor_local)) = RHS(index_ij+(i_var-1)*(n_tor_local)) + rhs_ij(i_var)
           enddo
           
           do k=1,n_vertex_max

             do l=1,n_order+1

               do in=i_tor_min,i_tor_max

                 ! Unknown increments to next time step (delta u^n)
                 eq(var_varStar,0,0,0,1) = H(k,l,ms,mt)*element%size(k,l)*HZ(in,mp)
                 eq(var_varStar,1,0,0,1) = (y_t(mp,ms,mt)*h_s(k,l,ms,mt) - y_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)*HZ(in,mp)/xjac
                 eq(var_varStar,0,1,0,1) = (-x_t(mp,ms,mt)*h_s(k,l,ms,mt) + x_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)*HZ(in,mp)/xjac
                 eq(var_varStar,0,0,1,1) = H(k,l,ms,mt)*element%size(k,l)*HZ_p(in,mp) - eq(var_varStar,1,0,0,1)*x_p(mp,ms,mt) - eq(var_varStar,0,1,0,1)*y_p(mp,ms,mt)
                 eq(var_varStar,2,0,0,1) = (h_ss(k,l,ms,mt)*y_t(mp,ms,mt)**2 - 2.d0*h_st(k,l,ms,mt)*y_s(mp,ms,mt)*y_t(mp,ms,mt)                             &
                                     + h_tt(k,l,ms,mt)*y_s(mp,ms,mt)**2                                                                                 &
                                     + h_s(k,l,ms,mt)*(y_st(mp,ms,mt)*y_t(mp,ms,mt) - y_tt(mp,ms,mt)*y_s(mp,ms,mt))                                     &
                                     + h_t(k,l,ms,mt)*(y_st(mp,ms,mt)*y_s(mp,ms,mt) - y_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(k,l)*HZ(in,mp)/xjac**2&	
                                     - xjac_x*(h_s(k,l,ms,mt)*y_t(mp,ms,mt) - h_t(k,l,ms,mt)*y_s(mp,ms,mt))*element%size(k,l)*HZ(in,mp)/xjac**2
                 eq(var_varStar,0,2,0,1) = (h_ss(k,l,ms,mt)*x_t(mp,ms,mt)**2 - 2.d0*h_st(k,l,ms,mt)*x_s(mp,ms,mt)*x_t(mp,ms,mt)                             &
                                     + h_tt(k,l,ms,mt)*x_s(mp,ms,mt)**2                                                                                 &
                                     + h_s(k,l,ms,mt)*(x_st(mp,ms,mt)*x_t(mp,ms,mt) - x_tt(mp,ms,mt)*x_s(mp,ms,mt))                                     &
                                     + h_t(k,l,ms,mt)*(x_st(mp,ms,mt)*x_s(mp,ms,mt) - x_ss(mp,ms,mt)*x_t(mp,ms,mt)))*element%size(k,l)*HZ(in,mp)/xjac**2&
                                     - xjac_y*(-h_s(k,l,ms,mt)*x_t(mp,ms,mt) + h_t(k,l,ms,mt)*x_s(mp,ms,mt))*element%size(k,l)*HZ(in,mp)/xjac**2
                 eq(var_varStar,1,1,0,1) = (-h_ss(k,l,ms,mt)*y_t(mp,ms,mt)*x_t(mp,ms,mt) - h_tt(k,l,ms,mt)*x_s(mp,ms,mt)*y_s(mp,ms,mt)                      &
     	                             + h_st(k,l,ms,mt)*(y_s(mp,ms,mt)*x_t(mp,ms,mt)  + y_t(mp,ms,mt)*x_s(mp,ms,mt))                                     &
                                     - h_s(k,l,ms,mt)*(x_st(mp,ms,mt)*y_t(mp,ms,mt) - x_tt(mp,ms,mt)*y_s(mp,ms,mt))                                     &
                                     - h_t(k,l,ms,mt)*(x_st(mp,ms,mt)*y_s(mp,ms,mt) - x_ss(mp,ms,mt)*y_t(mp,ms,mt)))*element%size(k,l)*HZ(in,mp)/xjac**2&
                                     - xjac_x*(-h_s(k,l,ms,mt)*x_t(mp,ms,mt) + h_t(k,l,ms,mt)*x_s(mp,ms,mt))*element%size(k,l)*HZ(in,mp)/xjac**2
                 u_px                = (y_t(mp,ms,mt)*h_s(k,l,ms,mt) - y_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)*HZ_p(in,mp)/xjac
                 u_py                = (-x_t(mp,ms,mt)*h_s(k,l,ms,mt) + x_s(mp,ms,mt)*h_t(k,l,ms,mt))*element%size(k,l)*HZ_p(in,mp)/xjac
                 eq(var_varStar,0,0,2,1) = H(k,l,ms,mt)*element%size(k,l)*HZ_pp(in,mp) - x_pp(mp,ms,mt)*eq(var_varStar,1,0,0,1) - 2.d0*(x_p(mp,ms,mt)*u_px &
                                     + y_p(mp,ms,mt)*u_py) - y_pp(mp,ms,mt)*eq(var_varStar,0,1,0,1) + 2.d0*(x_p(mp,ms,mt)*x_p_x*eq(var_varStar,1,0,0,1) &
                                     + x_p(mp,ms,mt)*y_p_x*eq(var_varStar,0,1,0,1) + y_p(mp,ms,mt)*x_p_y*eq(var_varStar,1,0,0,1) &
                                     + y_p(mp,ms,mt)*y_p_y*eq(var_varStar,0,1,0,1)) + x_p(mp,ms,mt)**2*eq(var_varStar,2,0,0,1)             &
                                     + 2.d0*x_p(mp,ms,mt)*y_p(mp,ms,mt)*eq(var_varStar,1,1,0,1) + y_p(mp,ms,mt)**2*eq(var_varStar,0,2,0,1)
                 eq(var_varStar,1,0,1,1) = u_px - x_p_x*eq(var_varStar,1,0,0,1) - x_p(mp,ms,mt)*eq(var_varStar,2,0,0,1) - y_p_x*eq(var_varStar,0,1,0,1) &
                                     - y_p(mp,ms,mt)*eq(var_varStar,1,1,0,1)
                 eq(var_varStar,0,1,1,1) = u_py - x_p_y*eq(var_varStar,1,0,0,1) - y_p(mp,ms,mt)*eq(var_varStar,0,2,0,1) - y_p_y*eq(var_varStar,0,1,0,1) &
                                     - x_p(mp,ms,mt)*eq(var_varStar,1,1,0,1)
                 
                 index_kl = n_tor_local*n_var*(n_order+1)*(k-1) + n_tor_local*n_var*(l-1) + in - i_tor_min + 1   ! index in the ELM matrix
                 
#include "amat_automatic.h"
                 
                 ! Include pre-factor to contribution
                 do i_var = 1, n_var
                   if ((i_var .eq. var_Psi) .or. (i_var .eq.var_zj)) then
                     prefactor =  wst*BigR*xjac/F0
                   else
                     prefactor =  wst*BigR*xjac
                   endif

                   do j_var = 1, n_var 
                     amat_ij(i_var, j_var) = amat_ij(i_var, j_var)*prefactor
                   enddo
                 enddo
                 
                 ! Fill up ELM
                 do i_var = 1, n_var
                   ij = index_ij + (i_var - 1)*n_tor_local
                   do j_var = 1, n_var 
                     kl = index_kl + (j_var - 1)*n_tor_local
                     ELM(ij,kl) =  ELM(ij,kl) + amat_ij(i_var, j_var)
                   enddo
                 enddo

               enddo
             enddo
           enddo

         enddo
       enddo

     enddo
     
   enddo

 enddo
enddo

return

end subroutine element_matrix
end module mod_elt_matrix
