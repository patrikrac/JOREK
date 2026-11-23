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
use mod_pfafem, only: pfafem_ng_tor

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

s_norm_s = 1.d0 / float(n_flux-1)

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

subroutine element_matrix_pfafem(node_list, element_list, ielm, xpoint2, xcase2, &
                                 R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, &
                                 ELM_sp, RHS_u, tid, aux_node_list)
  use constants
  use mod_parameters
  use data_structure
  use gauss
  use basis_at_gaussian, only: H, H_s, H_t
  use phys_module
  use mod_equations
  use mod_pfafem, only: pfafem_cache, pfafem_el_lo, pfafem_el_hi, pfafem_ep_dim, &
                        pfafem_ng_tor, pfafem_ng_pol, pfafem_tau_q, pfafem_w_tor, &
                        pfafem_t_pol, pfafem_w_pol, pfafem_dphi, pfafem_chi_q, &
                        pfafem_qp_cached, pfafem_qp_geom, pfafem_qp_ngrp, &
                        pfafem_qp_gu, pfafem_qp_gjb, pfafem_qp_db, &
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

  integer :: i, j, in_c, ms, mt, eP, ep1, iep_q, iq, iel_loc
  integer :: n_raw, raw_u(64), raw_io(64), raw_jb(64)
  integer :: n_act, act_u(64), act_io(64), act_jb(64)
  integer :: n_grp, g_a, a0, io_a, u_a, jb_a
  integer :: a, b, unode, iuja, iujb, isp, i_var, j_var, r_loc
  real*8  :: theta, zeta, reta, s_norm_s, phi_q, t_q, w_t, sz
  real*8  :: hz_c(n_coord_tor), hzp_c(n_coord_tor)
  real*8  :: H0_q(4, n_degrees), Hs_q(4, n_degrees), Ht_q(4, n_degrees)
  real*8  :: x_g, x_s, x_t, x_p, y_g, y_s, y_t, y_p, xjac, BigR, wvol, prefactor
  real*8  :: press_q
  real*8  :: B_gvec(3), B_gvec_s(3), B_gvec_t(3), B_gvec_p(3)
  real*8  :: BR_R, BR_z, BR_p, Bz_R, Bz_z, Bz_p, Bp_R, Bp_z, Bp_p
  real*8  :: psi_norm, psi_norm_R, psi_norm_Z, psi_norm_p, sech2
  real*8  :: rv(64), rR(64), rZ(64), rP(64), rRR(64), rRZ(64), rZZ(64), rRP(64), rZP(64), rPP(64)
  real*8  :: bv(64), bR(64), bZ(64), bP(64), bRR(64), bRZ(64), bZZ(64), bRP(64), bZP(64), bPP(64)
  real*8  :: r_jet(10)
  real*8  :: val_k(n_var)
  real*8  :: rhs_ij(n_var), amat_ij(n_var, n_var)
  integer :: ip_d, iq_d
  integer, parameter :: d_ix(3, 10) = reshape([ &
    0,0,0,  1,0,0,  0,1,0,  0,0,1,  2,0,0, &
    0,2,0,  1,1,0,  0,0,2,  1,0,1,  0,1,1 ], [3, 10])
  real*8  :: db_mat(10, 64), R_p(n_var, 10), C_pq(n_var, n_var, 10, 10)
  real*8  :: T_aq(n_var, n_var, 10, 64), amat_ab(n_var, n_var), r_val
  real*8  :: x_iq0(4, 4), x_iq1(4, 4), y_iq0(4, 4), y_iq1(4, 4), b_iq0(3, 4, 4), b_iq1(3, 4, 4)
  logical :: has_axis_u, use_qp_c

  eq => thread_eq(tid)%eq
  ELM_sp  = 0.d0
  RHS_u   = 0.d0
  rhs_ij  = 0.d0
  amat_ij = 0.d0
  reta    = 0.d0

  theta    = time_evol_theta
  if (tstep + tstep_prev > 0.d0) then
    zeta   = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  else
    zeta   = 0.d0
  endif
  s_norm_s = 1.d0 / float(n_flux - 1)

  element = element_list%element(ielm)
  do i = 1, n_vertex_max
    nodes(i) = node_list%node(element%vertex(i))
  end do

  iel_loc  = ielm - pfafem_el_lo + 1
  use_qp_c = pfafem_qp_cached .and. (ielm >= pfafem_el_lo) .and. (ielm <= pfafem_el_hi)
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
      if (n_coord_tor == 1) then
        hz_c(1)  = 1.d0
        hzp_c(1) = 0.d0
      else
        call pfafem_eval_coord_hz(phi_q, hz_c, hzp_c)
      end if

      do i = 1, n_vertex_max
        do j = 1, n_order + 1
          sz = element%size(i, j)
          if (.not. use_qp_c) then
            x_iq0(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 1), hz_c)
            x_iq1(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 1), hzp_c)
            y_iq0(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 2), hz_c)
            y_iq1(i, j) = sz * dot_product(nodes(i)%x(1:n_coord_tor, j, 2), hzp_c)
          end if
          b_iq0(1, i, j) = sz * dot_product(nodes(i)%b_field(1:n_coord_tor, j, 1), hz_c)
          b_iq0(2, i, j) = sz * dot_product(nodes(i)%b_field(1:n_coord_tor, j, 2), hz_c)
          b_iq0(3, i, j) = sz * dot_product(nodes(i)%b_field(1:n_coord_tor, j, 3), hz_c)
          b_iq1(1, i, j) = sz * dot_product(nodes(i)%b_field(1:n_coord_tor, j, 1), hzp_c)
          b_iq1(2, i, j) = sz * dot_product(nodes(i)%b_field(1:n_coord_tor, j, 2), hzp_c)
          b_iq1(3, i, j) = sz * dot_product(nodes(i)%b_field(1:n_coord_tor, j, 3), hzp_c)
        end do
      end do

      do ms = 1, n_gauss
        do mt = 1, pfafem_ng_pol
          if (ielm >= pfafem_el_lo .and. ielm <= pfafem_el_hi) then
            t_q = pfafem_t_pol(ms, mt, iq, iep_q, iel_loc)
            w_t = pfafem_w_pol(ms, mt, iq, iep_q, iel_loc)
          else
            t_q = xgauss(min(n_gauss, mt))
            w_t = wgauss(min(n_gauss, mt))
          end if
          call pfafem_hermite2d_jet(xgauss(ms), t_q, H0_q, Hs_q, Ht_q)

          B_gvec = 0.d0; B_gvec_s = 0.d0; B_gvec_t = 0.d0; B_gvec_p = 0.d0
          press_q  = 0.d0
          psi_norm = 0.d0
          if (use_qp_c) then
            BigR     = pfafem_qp_geom(1,  ms, mt, iq, ep1, iel_loc)
            y_g      = pfafem_qp_geom(2,  ms, mt, iq, ep1, iel_loc)
            x_s      = pfafem_qp_geom(3,  ms, mt, iq, ep1, iel_loc)
            x_t      = pfafem_qp_geom(4,  ms, mt, iq, ep1, iel_loc)
            x_p      = pfafem_qp_geom(5,  ms, mt, iq, ep1, iel_loc)
            y_s      = pfafem_qp_geom(6,  ms, mt, iq, ep1, iel_loc)
            y_t      = pfafem_qp_geom(7,  ms, mt, iq, ep1, iel_loc)
            y_p      = pfafem_qp_geom(8,  ms, mt, iq, ep1, iel_loc)
            xjac     = pfafem_qp_geom(13, ms, mt, iq, ep1, iel_loc)
            wvol     = pfafem_qp_geom(20, ms, mt, iq, ep1, iel_loc)
            psi_norm = pfafem_qp_geom(21, ms, mt, iq, ep1, iel_loc)
            do i = 1, n_vertex_max
              do j = 1, n_order + 1
                press_q       = press_q       + nodes(i)%pressure(j) * element%size(i, j) * H0_q(i, j)
                B_gvec(1:3)   = B_gvec(1:3)   + b_iq0(1:3, i, j) * H0_q(i, j)
                B_gvec_s(1:3) = B_gvec_s(1:3) + b_iq0(1:3, i, j) * Hs_q(i, j)
                B_gvec_t(1:3) = B_gvec_t(1:3) + b_iq0(1:3, i, j) * Ht_q(i, j)
                B_gvec_p(1:3) = B_gvec_p(1:3) + b_iq1(1:3, i, j) * H0_q(i, j)
              end do
            end do
          else
            x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_p = 0.d0
            y_g = 0.d0; y_s = 0.d0; y_t = 0.d0; y_p = 0.d0
            do i = 1, n_vertex_max
              do j = 1, n_order + 1
                sz  = element%size(i, j)
                x_g = x_g + x_iq0(i, j) * H0_q(i, j)
                x_s = x_s + x_iq0(i, j) * Hs_q(i, j)
                x_t = x_t + x_iq0(i, j) * Ht_q(i, j)
                x_p = x_p + x_iq1(i, j) * H0_q(i, j)

                y_g = y_g + y_iq0(i, j) * H0_q(i, j)
                y_s = y_s + y_iq0(i, j) * Hs_q(i, j)
                y_t = y_t + y_iq0(i, j) * Ht_q(i, j)
                y_p = y_p + y_iq1(i, j) * H0_q(i, j)

                press_q  = press_q  + nodes(i)%pressure(j) * sz * H0_q(i, j)
                psi_norm = psi_norm + nodes(i)%r_tor_eq(j) * sz * H0_q(i, j)

                B_gvec(1:3)   = B_gvec(1:3)   + b_iq0(1:3, i, j) * H0_q(i, j)
                B_gvec_s(1:3) = B_gvec_s(1:3) + b_iq0(1:3, i, j) * Hs_q(i, j)
                B_gvec_t(1:3) = B_gvec_t(1:3) + b_iq0(1:3, i, j) * Ht_q(i, j)
                B_gvec_p(1:3) = B_gvec_p(1:3) + b_iq1(1:3, i, j) * H0_q(i, j)
              end do
            end do

            xjac = x_s * y_t - x_t * y_s
            BigR = x_g
            wvol = wgauss(ms) * w_t * pfafem_w_tor(iq) * BigR * xjac
          end if

          eq = 0.d0
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
                raw_u(a)  = u_a
                raw_io(a) = io_a
                raw_jb(a) = jb_a
                r_jet = pfafem_qp_db(1:10, a, ms, mt, iq, ep1, iel_loc)
                val_k = node_list%node(unode)%values(jb_a, io_a, 1:n_var)
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
              end do
            end do
          else
            call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                      BigR, y_g, phi_q, .false., n_raw, raw_u, raw_io, raw_jb, &
                                      rv, rR, rZ, rP, rRR, rRZ, rZZ, rRP, rZP, rPP)
            do a = 1, n_raw
              unode = pfafem_cache(ielm)%unodes(raw_u(a))
              val_k = node_list%node(unode)%values(raw_jb(a), raw_io(a), 1:n_var)
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
            end do
          end if

          eq(var_chi, :, :, :, 1) = pfafem_chi_q(:, :, :, ms, mt, iq, eP + 1, iel_loc)
          eq(var_R,   0, 0, 0, 1) = BigR
          eq(var_R,   1, 0, 0, 1) = 1.d0

          eq(var_p0_gvec, 0, 0, 0, 1)               = mu_zero * press_q
          eq(var_B0x_gvec:var_B0p_gvec, 0, 0, 0, 1) = B_gvec(1:3)

          BR_R = ( y_t * B_gvec_s(1) - y_s * B_gvec_t(1)) / xjac
          BR_z = (-x_t * B_gvec_s(1) + x_s * B_gvec_t(1)) / xjac
          BR_p = B_gvec_p(1) - x_p * BR_R - y_p * BR_z
          eq(var_B0x_gvec, 1, 0, 0, 1) = BR_R
          eq(var_B0x_gvec, 0, 1, 0, 1) = BR_z
          eq(var_B0x_gvec, 0, 0, 1, 1) = BR_p

          Bz_R = ( y_t * B_gvec_s(2) - y_s * B_gvec_t(2)) / xjac
          Bz_z = (-x_t * B_gvec_s(2) + x_s * B_gvec_t(2)) / xjac
          Bz_p = B_gvec_p(2) - x_p * Bz_R - y_p * Bz_z
          eq(var_B0y_gvec, 1, 0, 0, 1) = Bz_R
          eq(var_B0y_gvec, 0, 1, 0, 1) = Bz_z
          eq(var_B0y_gvec, 0, 0, 1, 1) = Bz_p

          Bp_R = ( y_t * B_gvec_s(3) - y_s * B_gvec_t(3)) / xjac
          Bp_z = (-x_t * B_gvec_s(3) + x_s * B_gvec_t(3)) / xjac
          Bp_p = B_gvec_p(3) - x_p * Bp_R - y_p * Bp_z
          eq(var_B0p_gvec, 1, 0, 0, 1) = Bp_R
          eq(var_B0p_gvec, 0, 1, 0, 1) = Bp_z
          eq(var_B0p_gvec, 0, 0, 1, 1) = Bp_p

          psi_norm_R =  y_t * s_norm_s / xjac
          psi_norm_Z = -x_t * s_norm_s / xjac
          psi_norm_p = -x_p * psi_norm_R - y_p * psi_norm_Z
          sech2 = 1.d0 / (2.d0 * j_cutoff_sig * (cosh((psi_norm - j_cutoff_rcoord) / j_cutoff_sig)**2))
          eq(var_heaviside, 0, 0, 0, 1) = 0.5d0 - 0.5d0 * tanh((psi_norm - j_cutoff_rcoord) / j_cutoff_sig)
          eq(var_heaviside, 1, 0, 0, 1) = -psi_norm_R * sech2
          eq(var_heaviside, 0, 1, 0, 1) = -psi_norm_Z * sech2
          eq(var_heaviside, 0, 0, 1, 1) = -psi_norm_p * sech2

          eq(var_Psi, :, :, :, 1) = eq(var_Psi, :, :, :, 1) / F0
          eq(var_zj,  :, :, :, 1) = eq(var_zj,  :, :, :, 1) / F0

#include "aux_automatic.h"

          if (has_axis_u) then
            call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                      BigR, y_g, phi_q, .true., n_act, act_u, act_io, act_jb, &
                                      bv, bR, bZ, bP, bRR, bRZ, bZZ, bRP, bZP, bPP)
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
            act_u(1:n_act)  = raw_u(1:n_act)
            act_io(1:n_act) = raw_io(1:n_act)
            act_jb(1:n_act) = raw_jb(1:n_act)
            if (n_act > 0) then
              db_mat(1:10, 1:n_act) = pfafem_qp_db(1:10, 1:n_act, ms, mt, iq, ep1, iel_loc)
            end if
          else
            n_act = n_raw
            act_u(1:n_act)  = raw_u(1:n_act)
            act_io(1:n_act) = raw_io(1:n_act)
            act_jb(1:n_act) = raw_jb(1:n_act)
            do a = 1, n_act
              db_mat(1,  a) = rv(a)
              db_mat(2,  a) = rR(a)
              db_mat(3,  a) = rZ(a)
              db_mat(4,  a) = rP(a)
              db_mat(5,  a) = rRR(a)
              db_mat(6,  a) = rZZ(a)
              db_mat(7,  a) = rRZ(a)
              db_mat(8,  a) = rPP(a)
              db_mat(9,  a) = rRP(a)
              db_mat(10, a) = rZP(a)
            end do
          end if
          if (n_act > 0) then

            eq(var_v,       :, :, :, 1) = 0.d0
            eq(var_varStar, :, :, :, 1) = 0.d0
            R_p  = 0.d0
            C_pq = 0.d0

            do ip_d = 1, 10
              eq(var_v, d_ix(1, ip_d), d_ix(2, ip_d), d_ix(3, ip_d), 1) = 1.d0
              rhs_ij = 0.d0
#include "rhs_automatic.h"
              R_p(:, ip_d) = rhs_ij(:)
              do iq_d = 1, 10
                eq(var_varStar, d_ix(1, iq_d), d_ix(2, iq_d), d_ix(3, iq_d), 1) = 1.d0
                amat_ij = 0.d0
#include "amat_automatic.h"
                C_pq(:, :, ip_d, iq_d) = amat_ij(:, :)
                eq(var_varStar, d_ix(1, iq_d), d_ix(2, iq_d), d_ix(3, iq_d), 1) = 0.d0
              end do
              eq(var_v, d_ix(1, ip_d), d_ix(2, ip_d), d_ix(3, ip_d), 1) = 0.d0
            end do

            do a = 1, n_act
              do i_var = 1, n_var
                r_val = dot_product(R_p(i_var, 1:10), db_mat(1:10, a))
                r_loc = (i_var - 1) * n_tor + act_jb(a)
                RHS_u(r_loc, act_io(a), act_u(a)) = RHS_u(r_loc, act_io(a), act_u(a)) + wvol * r_val
              end do
              T_aq(:, :, 1:10, a) = 0.d0
              do iq_d = 1, 10
                do ip_d = 1, 10
                  if (db_mat(ip_d, a) /= 0.d0) then
                    T_aq(:, :, iq_d, a) = T_aq(:, :, iq_d, a) + C_pq(:, :, ip_d, iq_d) * db_mat(ip_d, a)
                  end if
                end do
              end do
            end do

            do a = 1, n_act
              iuja = pfafem_cache(ielm)%uj_map(act_u(a), act_jb(a))
              do b = 1, n_act
                amat_ab = 0.d0
                do iq_d = 1, 10
                  if (db_mat(iq_d, b) /= 0.d0) then
                    amat_ab = amat_ab + T_aq(:, :, iq_d, a) * db_mat(iq_d, b)
                  end if
                end do
                iujb = pfafem_cache(ielm)%uj_map(act_u(b), act_jb(b))
                isp  = pfafem_cache(ielm)%sp_map(iuja, iujb)
                do i_var = 1, n_var
                  if ((i_var .eq. var_Psi) .or. (i_var .eq. var_zj)) then
                    prefactor = wvol / F0
                  else
                    prefactor = wvol
                  end if
                  do j_var = 1, n_var
                    ELM_sp(act_io(a), act_io(b), i_var, j_var, isp) = &
                      ELM_sp(act_io(a), act_io(b), i_var, j_var, isp) + prefactor * amat_ab(i_var, j_var)
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

end module mod_elt_matrix
