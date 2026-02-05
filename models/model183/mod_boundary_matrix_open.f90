module mod_boundary_matrix_open
  implicit none
contains
subroutine boundary_matrix_open(vertex, direction, element, nodes, xpoint2, xcase2, R_axis, Z_axis, psi_axis, &
                                psi_bnd, R_xpoint, Z_xpoint, ELM, RHS, i_tor_min, i_tor_max, ielm)
!--------------------------------------------------------------------------------------------------------------------
! implements a boundary condition for zj, which allows the values of zj on the boundary to change during a simulation
!--------------------------------------------------------------------------------------------------------------------
use mod_parameters
use data_structure
use gauss
use basis_at_gaussian
use phys_module
use nodes_elements
use mod_chi
implicit none

type(type_element)   :: element
type(type_node)      :: nodes(2)        ! the two nodes containing the boundary nodes

real*8, dimension(:,:), allocatable :: ELM
real*8, dimension(:),   allocatable :: RHS
integer,                intent(in)  :: i_tor_min, i_tor_max, ielm

integer :: vertex(2), direction(2), xcase2
real*8  :: psi_axis, R_axis, Z_axis, psi_bnd, R_xpoint(2), Z_xpoint(2)
logical :: xpoint2

integer :: vertex2(2), direction_perp(2), i, i2, i3, j, j2, j3, k, l, ms, mp, im, in, index_ij, index_kl, ij3, kl1, kl3, n_tor_local
real*8  :: element_size_ij, element_size_kl, element_size_perp, BigR, phi, Bv2, dl, xjac, xjac_x, xjac_y, grad_chi(3), grad_Bv2(3)
real*8  :: Psi0_x, Psi0_y, Psi0_phi, Psi0_xx, Psi0_yy, Psi0_xy, Psi0_xphi, Psi0_yphi, Psi0_phiphi, Psi0_px, Psi0_py, Lap_Psi0
real*8  :: grad_Psi0(3), grad_Psi(3), Bv_parderiv_Bv_parderiv_Psi0, div_Bv2_pgrad_Psi0, v, rhs_ij_3, amat_31, amat_33, zbig, x_p_x
real*8  :: Psi, Psi_s, Psi_t, Psi_p, Psi_ss, Psi_tt, Psi_st, Psi_sp, Psi_tp, Psi_pp, theta, zeta, Bv_parderiv_Bv_parderiv_Psi, x_p_y
real*8  :: Psi_x, Psi_y, Psi_phi, Psi_xx, Psi_yy, Psi_xy, Psi_xphi, Psi_yphi, Psi_phiphi, Psi_px, Psi_py, zj, Lap_Psi, y_p_x, y_p_y

real*8, dimension(n_plane,n_gauss) :: x_g, x_s, x_t, x_p, x_ss, x_tt, x_st, x_sp, x_tp, x_pp
real*8, dimension(n_plane,n_gauss) :: y_g, y_s, y_t, y_p, y_ss, y_tt, y_st, y_sp, y_tp, y_pp
real*8, dimension(n_plane,n_gauss) :: Psi0_s, Psi0_t, Psi0_p, Psi0_ss, Psi0_tt, Psi0_st, Psi0_sp, Psi0_tp, Psi0_pp, zj0

real*8, dimension(n_vertex_max,n_order+1,n_gauss) :: H1_full, H1_s_full, H1_t_full, H1_ss_full, H1_tt_full, H1_st_full

real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: chi

type(type_node) :: nodes2(2), tmp_node

theta = time_evol_theta
!zeta  = time_evol_zeta
! change zeta for variable dt
zeta  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
Zbig = 1.d12

!--------------------- reorder the nodes to have the same direction as full element (maybe not necesary)
if ((vertex(1) .eq. 3) .and. (vertex(2) .eq. 4)) then
  tmP_node = nodes(1)
  nodes(1)  = nodes(2)
  nodes(2)  = tmp_node
  vertex(1) = 4
  vertex(2) = 3
else if ((vertex(1) .eq. 4) .and. (vertex(2) .eq. 1)) then
  tmP_node = nodes(1)
  nodes(1)  = nodes(2)
  nodes(2)  = tmp_node
  vertex(1) = 1
  vertex(2) = 4
else if ((vertex(1) .eq. 3) .and. (vertex(2) .eq. 2)) then
  tmP_node = nodes(1)
  nodes(1)  = nodes(2)
  nodes(2)  = tmp_node
  vertex(1) = 2
  vertex(2) = 3
else if ((vertex(1) .eq. 2) .and. (vertex(2) .eq. 1)) then
  tmP_node = nodes(1)
  nodes(1)  = nodes(2)
  nodes(2)  = tmp_node
  vertex(1) = 1
  vertex(2) = 2
end if

if ((vertex(1) .eq. 1) .and. (vertex(2) .eq. 2)) then
  vertex2(1) = 4
  vertex2(2) = 3
  nodes2(1) = node_list%node(element%vertex(4))
  nodes2(2) = node_list%node(element%vertex(3))
else if ((vertex(1) .eq. 2) .and. (vertex(2) .eq. 3)) then
  vertex2(1) = 1
  vertex2(2) = 4
  nodes2(1) = node_list%node(element%vertex(1))
  nodes2(2) = node_list%node(element%vertex(4))
else if ((vertex(1) .eq. 4) .and. (vertex(2) .eq. 3)) then
  vertex2(1) = 1
  vertex2(2) = 2
  nodes2(1) = node_list%node(element%vertex(1))
  nodes2(2) = node_list%node(element%vertex(2))
else if ((vertex(1) .eq. 1) .and. (vertex(2) .eq. 4)) then
  vertex2(1) = 2
  vertex2(2) = 3
  nodes2(1) = node_list%node(element%vertex(2))
  nodes2(2) = node_list%node(element%vertex(3))
end if

x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_p = 0.d0; x_ss = 0.d0; x_tt = 0.d0; x_st = 0.d0; x_sp = 0.d0; x_tp = 0.d0; x_pp = 0.d0
y_g = 0.d0; y_s = 0.d0; y_t = 0.d0; y_p = 0.d0; y_ss = 0.d0; y_tt = 0.d0; y_st = 0.d0; y_sp = 0.d0; y_tp = 0.d0; y_pp = 0.d0
Psi0_s = 0.d0; Psi0_t = 0.d0; Psi0_p = 0.d0; Psi0_ss = 0.d0; Psi0_tt = 0.d0
Psi0_st = 0.d0; Psi0_sp = 0.d0; Psi0_tp = 0.d0; Psi0_pp = 0.d0; zj0 = 0.d0
H1_full = 0.d0; H1_s_full = 0.d0; H1_t_full = 0.d0; H1_ss_full = 0.d0; H1_tt_full = 0.d0; H1_st_full = 0.d0

direction_perp(1) = 6/direction(2)     ! =3 if direction(2)=2, =2 if direction(2)=3
direction_perp(2) = 4

do i=1,2    ! sum over 2 verices
  do j=1,2  ! sum over two basis functions
    i2 = vertex(i)
    j2 = direction(j)
    element_size_ij = element%size(i2,j2)
    H1_full(i2,j2,:)    = H1(i,j,:)
    H1_s_full(i2,j2,:)  = H1_s(i,j,:)
    H1_ss_full(i2,j2,:) = H1_ss(i,j,:)

    i3 = vertex2(i)
    j3 = direction_perp(j)
    if (vertex(1) .eq. 1) then
      element_size_perp = 3.d0*element%size(i2,direction_perp(1))
      H1_t_full(i2,j3,:)  = 3.d0*H1(i,j,:)
      H1_st_full(i2,j3,:) = 3.d0*H1_s(i,j,:)
    else
      element_size_perp = -3.d0*element%size(i2,direction_perp(1))
      H1_t_full(i2,j3,:)  = -3.d0*H1(i,j,:)
      H1_st_full(i2,j3,:) = -3.d0*H1_s(i,j,:)
    end if
    H1_tt_full(i2,j2,:) = -6.d0*H1(i,j,:)
    H1_tt_full(i2,j3,:) = -12.d0*H1(i,j,:)
    H1_tt_full(i3,j2,:) = 6.d0*H1(i,j,:)
    H1_tt_full(i3,j3,:) = 6.d0*H1(i,j,:)
    
    do ms=1,n_gauss
      do mp=1,n_plane
        do in=1,n_coord_tor
          x_g(mp,ms)  = x_g(mp,ms)  + nodes(i)%x(in,j2,1)*element_size_ij*H1(i,j,ms)   *HZ_coord(in,mp)
          x_s(mp,ms)  = x_s(mp,ms)  + nodes(i)%x(in,j2,1)*element_size_ij*H1_s(i,j,ms) *HZ_coord(in,mp)
          x_t(mp,ms)  = x_t(mp,ms)  + nodes(i)%x(in,j3,1)*element_size_ij*H1(i,j,ms)   *HZ_coord(in,mp)  *element_size_perp
          x_p(mp,ms)  = x_p(mp,ms)  + nodes(i)%x(in,j2,1)*element_size_ij*H1(i,j,ms)   *HZ_coord_p(in,mp)
          x_ss(mp,ms) = x_ss(mp,ms) + nodes(i)%x(in,j2,1)*element_size_ij*H1_ss(i,j,ms)*HZ_coord(in,mp)
          x_st(mp,ms) = x_st(mp,ms) + nodes(i)%x(in,j3,1)*element_size_ij*H1_s(i,j,ms) *HZ_coord(in,mp)  *element_size_perp
          x_sp(mp,ms) = x_sp(mp,ms) + nodes(i)%x(in,j2,1)*element_size_ij*H1_s(i,j,ms) *HZ_coord_p(in,mp)
          x_tp(mp,ms) = x_tp(mp,ms) + nodes(i)%x(in,j3,1)*element_size_ij*H1(i,j,ms)   *HZ_coord_p(in,mp)*element_size_perp
          x_pp(mp,ms) = x_pp(mp,ms) + nodes(i)%x(in,j2,1)*element_size_ij*H1(i,j,ms)   *HZ_coord_pp(in,mp)
          x_tt(mp,ms) = x_tt(mp,ms) - 6.d0*(nodes(i)%x(in,j2,1)*element_size_ij + 2.d0*nodes(i)%x(in,j3,1)*element%size(i2,j3) &
                      - nodes2(i)%x(in,j2,1)*element%size(i3,j2) - nodes2(i)%x(in,j3,1)*element%size(i3,j3))*H1(i,j,ms)*HZ_coord(in,mp)

          y_g(mp,ms)  = y_g(mp,ms)  + nodes(i)%x(in,j2,2)*element_size_ij*H1(i,j,ms)  *HZ_coord(in,mp)
          y_s(mp,ms)  = y_s(mp,ms)  + nodes(i)%x(in,j2,2)*element_size_ij*H1_s(i,j,ms)*HZ_coord(in,mp)
          y_t(mp,ms)  = y_t(mp,ms)  + nodes(i)%x(in,j3,2)*element_size_ij*H1(i,j,ms)  *HZ_coord(in,mp)*element_size_perp
          y_p(mp,ms)  = y_p(mp,ms)  + nodes(i)%x(in,j2,2)*element_size_ij*H1(i,j,ms)  *HZ_coord_p(in,mp)
          y_ss(mp,ms) = y_ss(mp,ms) + nodes(i)%x(in,j2,2)*element_size_ij*H1_ss(i,j,ms)*HZ_coord(in,mp)
          y_st(mp,ms) = y_st(mp,ms) + nodes(i)%x(in,j3,2)*element_size_ij*H1_s(i,j,ms) *HZ_coord(in,mp)  *element_size_perp
          y_sp(mp,ms) = y_sp(mp,ms) + nodes(i)%x(in,j2,2)*element_size_ij*H1_s(i,j,ms) *HZ_coord_p(in,mp)
          y_tp(mp,ms) = y_tp(mp,ms) + nodes(i)%x(in,j3,2)*element_size_ij*H1(i,j,ms)   *HZ_coord_p(in,mp)*element_size_perp
          y_pp(mp,ms) = y_pp(mp,ms) + nodes(i)%x(in,j2,2)*element_size_ij*H1(i,j,ms)   *HZ_coord_pp(in,mp)
          y_tt(mp,ms) = y_tt(mp,ms) - 6.d0*(nodes(i)%x(in,j2,2)*element_size_ij + 2.d0*nodes(i)%x(in,j3,2)*element%size(i2,j3) &
                      - nodes2(i)%x(in,j2,2)*element%size(i3,j2) - nodes2(i)%x(in,j3,2)*element%size(i3,j3))*H1(i,j,ms)*HZ_coord(in,mp)
        end do
        
        do in=1,n_tor
          Psi0_s(mp,ms)  = Psi0_s(mp,ms)  + nodes(i)%values(in,j2,var_Psi)*element_size_ij*H1_s(i,j,ms) *HZ(in,mp)
          Psi0_t(mp,ms)  = Psi0_t(mp,ms)  + nodes(i)%values(in,j3,var_Psi)*element_size_ij*H1(i,j,ms)   *HZ(in,mp)  *element_size_perp
          Psi0_p(mp,ms)  = Psi0_p(mp,ms)  + nodes(i)%values(in,j2,var_Psi)*element_size_ij*H1(i,j,ms)   *HZ_p(in,mp)
          Psi0_ss(mp,ms) = Psi0_ss(mp,ms) + nodes(i)%values(in,j2,var_Psi)*element_size_ij*H1_ss(i,j,ms)*HZ(in,mp)
          Psi0_st(mp,ms) = Psi0_st(mp,ms) + nodes(i)%values(in,j3,var_Psi)*element_size_ij*H1_s(i,j,ms) *HZ(in,mp)  *element_size_perp
          Psi0_sp(mp,ms) = Psi0_sp(mp,ms) + nodes(i)%values(in,j2,var_Psi)*element_size_ij*H1_s(i,j,ms) *HZ_p(in,mp)
          Psi0_tp(mp,ms) = Psi0_tp(mp,ms) + nodes(i)%values(in,j3,var_Psi)*element_size_ij*H1(i,j,ms)   *HZ_p(in,mp)*element_size_perp
          Psi0_pp(mp,ms) = Psi0_pp(mp,ms) + nodes(i)%values(in,j2,var_Psi)*element_size_ij*H1(i,j,ms)   *HZ_pp(in,mp)
          Psi0_tt(mp,ms) = Psi0_tt(mp,ms) - 6.d0*(nodes(i)%values(in,j2,var_Psi)*element_size_ij + 2.d0*nodes(i)%values(in,j3,var_Psi)*element%size(i2,j3) &
                         - nodes2(i)%values(in,j2,var_Psi)*element%size(i3,j2) - nodes2(i)%values(in,j3,var_Psi)*element%size(i3,j3))*H1(i,j,ms)*HZ(in,mp)

          zj0(mp,ms) = zj0(mp,ms) + nodes(i)%values(in,j2,var_zj)*element_size_ij*H1(i,j,ms)*HZ(in,mp)
        end do
      end do
    end do
  end do
end do

n_tor_local = i_tor_max - i_tor_min + 1
do ms=1,n_gauss
  do mp=1,n_plane
    BigR = x_g(mp,ms)
    phi = 2.d0*pi*float(mp-1)/float(n_plane*n_period)
    chi = get_chi(x_g(mp,ms),y_g(mp,ms),phi,node_list,element_list,ielm,1.d0,xgauss(ms))
    grad_chi = (/ chi(1,0,0), chi(0,1,0), chi(0,0,1)/BigR /)
    Bv2 = dot_product(grad_chi,grad_chi)
    grad_Bv2 = 2.d0*(/ chi(1,0,0)*chi(2,0,0) + chi(0,1,0)*chi(1,1,0) + chi(0,0,1)*chi(1,0,1)/BigR**2 - chi(0,0,1)**2/BigR**3, &
                       chi(1,0,0)*chi(1,1,0) + chi(0,1,0)*chi(0,2,0) + chi(0,0,1)*chi(0,1,1)/BigR**2, &
                      (chi(1,0,0)*chi(1,0,1) + chi(0,1,0)*chi(0,1,1) + chi(0,0,1)*chi(0,0,2)/BigR**2)/BigR /)

    dl = sqrt(x_s(mp,ms)**2 + y_s(mp,ms)**2)
    xjac = x_t(mp,ms)*y_s(mp,ms) - x_s(mp,ms)*y_t(mp,ms)
    xjac_x = (x_tt(mp,ms)*y_s(mp,ms)**2 - y_tt(mp,ms)*x_s(mp,ms)*y_s(mp,ms) - 2.d0*x_st(mp,ms)*y_t(mp,ms)*y_s(mp,ms) &
	       + y_st(mp,ms)*(x_t(mp,ms)*y_s(mp,ms) + x_s(mp,ms)*y_t(mp,ms)) &
	       + x_ss(mp,ms)*y_t(mp,ms)**2 - y_ss(mp,ms)*x_t(mp,ms)*y_t(mp,ms))/xjac
    xjac_y = (y_ss(mp,ms)*x_t(mp,ms)**2 - x_ss(mp,ms)*y_t(mp,ms)*x_t(mp,ms) - 2.d0*y_st(mp,ms)*x_s(mp,ms)*x_t(mp,ms) &
	       + x_st(mp,ms)*(y_s(mp,ms)*x_t(mp,ms) + y_t(mp,ms)*x_s(mp,ms)) &
	       + y_tt(mp,ms)*x_s(mp,ms)**2 - x_tt(mp,ms)*y_s(mp,ms)*x_s(mp,ms))/xjac
    
    Psi0_x = (Psi0_t(mp,ms)*y_s(mp,ms) - Psi0_s(mp,ms)*y_t(mp,ms))/xjac
    Psi0_y = (Psi0_s(mp,ms)*x_t(mp,ms) - Psi0_t(mp,ms)*x_s(mp,ms))/xjac
    Psi0_phi = Psi0_p(mp,ms) - Psi0_x*x_p(mp,ms) - Psi0_y*y_p(mp,ms)
    Psi0_xx = (Psi0_tt(mp,ms)*y_s(mp,ms)**2 - 2.d0*Psi0_st(mp,ms)*y_t(mp,ms)*y_s(mp,ms) + Psi0_ss(mp,ms)*y_t(mp,ms)**2 &
           + Psi0_t(mp,ms)*(y_st(mp,ms)*y_s(mp,ms) - y_ss(mp,ms)*y_t(mp,ms)) &
           + Psi0_s(mp,ms)*(y_st(mp,ms)*y_t(mp,ms) - y_tt(mp,ms)*y_s(mp,ms)))/xjac**2 &
           - xjac_x*(Psi0_t(mp,ms)*y_s(mp,ms) - Psi0_s(mp,ms)*y_t(mp,ms))/xjac**2
    Psi0_yy = (Psi0_tt(mp,ms)*x_s(mp,ms)**2 - 2.d0*Psi0_st(mp,ms)*x_t(mp,ms)*x_s(mp,ms) + Psi0_ss(mp,ms)*x_t(mp,ms)**2 &
           + Psi0_t(mp,ms)*(x_st(mp,ms)*x_s(mp,ms) - x_ss(mp,ms)*x_t(mp,ms)) &
           + Psi0_s(mp,ms)*(x_st(mp,ms)*x_t(mp,ms) - x_tt(mp,ms)*x_s(mp,ms)))/xjac**2 &
           - xjac_y*(-Psi0_t(mp,ms)*x_s(mp,ms) + Psi0_s(mp,ms)*x_t(mp,ms))/xjac**2
    Psi0_xy = (-Psi0_ss(mp,ms)*y_t(mp,ms)*x_t(mp,ms) - Psi0_tt(mp,ms)*x_s(mp,ms)*y_s(mp,ms) &
           + Psi0_st(mp,ms)*(y_s(mp,ms)*x_t(mp,ms) + y_t(mp,ms)*x_s(mp,ms)) &
           - Psi0_s(mp,ms)*(x_st(mp,ms)*y_t(mp,ms) - x_tt(mp,ms)*y_s(mp,ms)) &
           - Psi0_t(mp,ms)*(x_st(mp,ms)*y_s(mp,ms) - x_ss(mp,ms)*y_t(mp,ms)))/xjac**2 &
           - xjac_x*(-Psi0_s(mp,ms)*x_t(mp,ms) + Psi0_t(mp,ms)*x_s(mp,ms))/xjac**2
    Psi0_px = (y_s(mp,ms)*Psi0_tp(mp,ms) - y_t(mp,ms)*Psi0_sp(mp,ms))/xjac
    Psi0_py = (-x_s(mp,ms)*Psi0_tp(mp,ms) + x_t(mp,ms)*Psi0_sp(mp,ms))/xjac
    Psi0_phiphi = Psi0_pp(mp,ms) - x_pp(mp,ms)*Psi0_x - 2.d0*(x_p(mp,ms)*Psi0_px + y_p(mp,ms)*Psi0_py) - y_pp(mp,ms)*Psi0_y &
               + 2.d0*(x_p(mp,ms)*x_p_x*Psi0_x + x_p(mp,ms)*y_p_x*Psi0_y + y_p(mp,ms)*x_p_y*Psi0_x + y_p(mp,ms)*y_p_y*Psi0_y) &
               + x_p(mp,ms)**2*Psi0_xx + 2.d0*x_p(mp,ms)*y_p(mp,ms)*Psi0_xy + y_p(mp,ms)**2*Psi0_yy
    Psi0_xphi = Psi0_px - x_p_x*Psi0_x - x_p(mp,ms)*Psi0_xx - y_p_x*Psi0_y - y_p(mp,ms)*Psi0_xy
    Psi0_yphi = Psi0_py - x_p_y*Psi0_x - x_p(mp,ms)*Psi0_xy - y_p_y*Psi0_y - y_p(mp,ms)*Psi0_yy
    grad_Psi0 = (/ Psi0_x, Psi0_y, Psi0_phi/BigR /)
    
    Lap_Psi0 = Psi0_xx + Psi0_x/BigR + Psi0_yy + Psi0_phiphi/BigR**2
    Bv_parderiv_Bv_parderiv_Psi0 = chi(1,0,0)*(chi(2,0,0)*Psi0_x + chi(1,0,0)*Psi0_xx + chi(1,1,0)*Psi0_y + chi(0,1,0)*Psi0_xy &
                                + (chi(1,0,1)*Psi0_phi + chi(0,0,1)*Psi0_xphi)/BigR**2 - 2.d0*chi(0,0,1)*Psi0_phi/BigR**3) &
                                + chi(0,1,0)*(chi(1,1,0)*Psi0_x + chi(1,0,0)*Psi0_xy + chi(0,2,0)*Psi0_y + chi(0,1,0)*Psi0_yy &
                                + (chi(0,1,1)*Psi0_phi + chi(0,0,1)*Psi0_yphi)/BigR**2) &
                                + chi(0,0,1)*(chi(1,0,1)*Psi0_x + chi(1,0,0)*Psi0_xphi + chi(0,1,1)*Psi0_y + chi(0,1,0)*Psi0_yphi &
                                + (chi(0,0,2)*Psi0_phi + chi(0,0,1)*Psi0_phiphi)/BigR**2)/BigR
    div_Bv2_pgrad_Psi0 = Bv2*Lap_Psi0 + dot_product(grad_Bv2,grad_Psi0) - Bv_parderiv_Bv_parderiv_Psi0
    
    do i=1,2
      do j=1,2
        j2 = direction(j)
        element_size_ij = element%size(vertex(i),j2)
        
        do im=i_tor_min,i_tor_max
          index_ij = n_tor_local*n_var*(n_order+1)*(vertex(i)-1) + n_tor_local * n_var * (j2-1) + im - i_tor_min + 1  ! index in the ELM matrix
          
          v = H1(i,j,ms)*element_size_ij*HZ(im,mp)
          rhs_ij_3 = v*(div_Bv2_pgrad_Psi0 - Bv2*zj0(mp,ms))*dl*zbig
          
          ij3 = index_ij + 2*n_tor_local
          
          RHS(ij3) = RHS(ij3) + rhs_ij_3*wgauss(ms)
          
          do k=1,n_vertex_max
            do l=1,n_order+1
              do in=i_tor_min,i_tor_max
                index_kl = n_tor_local*n_var*(n_order+1)*(k-1) + n_tor_local*n_var*(l-1) + in - i_tor_min + 1   ! index in the ELM matrix
                
                Psi = H1_full(k,l,ms)*element%size(k,l)*HZ(in,mp)
                Psi_s = H1_s_full(k,l,ms)*element%size(k,l)*HZ(in,mp)
                Psi_t = H1_t_full(k,l,ms)*element%size(k,l)*HZ(in,mp)
                Psi_p = H1_full(k,l,ms)*element%size(k,l)*HZ_p(in,mp)
                Psi_ss = H1_ss_full(k,l,ms)*element%size(k,l)*HZ(in,mp)
                Psi_tt = H1_tt_full(k,l,ms)*element%size(k,l)*HZ(in,mp)
                Psi_st = H1_st_full(k,l,ms)*element%size(k,l)*HZ(in,mp)
                Psi_sp = H1_s_full(k,l,ms)*element%size(k,l)*HZ_p(in,mp)
                Psi_tp = H1_t_full(k,l,ms)*element%size(k,l)*HZ_p(in,mp)
                Psi_pp = H1_full(k,l,ms)*element%size(k,l)*HZ_pp(in,mp)
                
                Psi_x = (Psi_t*y_s(mp,ms) - Psi_s*y_t(mp,ms))/xjac
                Psi_y = (Psi_s*x_t(mp,ms) - Psi_t*x_s(mp,ms))/xjac
                Psi_phi = Psi_p - Psi_x*x_p(mp,ms) - Psi_y*y_p(mp,ms)
                Psi_xx = (Psi_tt*y_s(mp,ms)**2 - 2.d0*Psi_st*y_t(mp,ms)*y_s(mp,ms) + Psi_ss*y_t(mp,ms)**2 &
                       + Psi_t*(y_st(mp,ms)*y_s(mp,ms) - y_ss(mp,ms)*y_t(mp,ms)) &
                       + Psi_s*(y_st(mp,ms)*y_t(mp,ms) - y_tt(mp,ms)*y_s(mp,ms)))/xjac**2 &
                       - xjac_x*(Psi_t*y_s(mp,ms) - Psi_s*y_t(mp,ms))/xjac**2
                Psi_yy = (Psi_tt*x_s(mp,ms)**2 - 2.d0*Psi_st*x_t(mp,ms)*x_s(mp,ms) + Psi_ss*x_t(mp,ms)**2 &
                       + Psi_t*(x_st(mp,ms)*x_s(mp,ms) - x_ss(mp,ms)*x_t(mp,ms)) &
                       + Psi_s*(x_st(mp,ms)*x_t(mp,ms) - x_tt(mp,ms)*x_s(mp,ms)))/xjac**2 &
                       - xjac_y*(-Psi_t*x_s(mp,ms) + Psi_s*x_t(mp,ms))/xjac**2
                Psi_xy = (-Psi_ss*y_t(mp,ms)*x_t(mp,ms) - Psi_tt*x_s(mp,ms)*y_s(mp,ms) &
                       + Psi_st*(y_s(mp,ms)*x_t(mp,ms) + y_t(mp,ms)*x_s(mp,ms)) &
                       - Psi_s*(x_st(mp,ms)*y_t(mp,ms) - x_tt(mp,ms)*y_s(mp,ms)) &
                       - Psi_t*(x_st(mp,ms)*y_s(mp,ms) - x_ss(mp,ms)*y_t(mp,ms)))/xjac**2 &
                       - xjac_x*(-Psi_s*x_t(mp,ms) + Psi_t*x_s(mp,ms))/xjac**2
                Psi_px = (y_s(mp,ms)*Psi_tp - y_t(mp,ms)*Psi_sp)/xjac
                Psi_py = (-x_s(mp,ms)*Psi_tp + x_t(mp,ms)*Psi_sp)/xjac
                Psi_phiphi = Psi_pp - x_pp(mp,ms)*Psi_x - 2.d0*(x_p(mp,ms)*Psi_px + y_p(mp,ms)*Psi_py) - y_pp(mp,ms)*Psi_y &
                           + 2.d0*(x_p(mp,ms)*x_p_x*Psi_x + x_p(mp,ms)*y_p_x*Psi_y + y_p(mp,ms)*x_p_y*Psi_x + y_p(mp,ms)*y_p_y*Psi_y) &
                           + x_p(mp,ms)**2*Psi_xx + 2.d0*x_p(mp,ms)*y_p(mp,ms)*Psi_xy + y_p(mp,ms)**2*Psi_yy
                Psi_xphi = Psi_px - x_p_x*Psi_x - x_p(mp,ms)*Psi_xx - y_p_x*Psi_y - y_p(mp,ms)*Psi_xy
                Psi_yphi = Psi_py - x_p_y*Psi_x - x_p(mp,ms)*Psi_xy - y_p_y*Psi_y - y_p(mp,ms)*Psi_yy
                grad_Psi = (/ Psi_x, Psi_y, Psi_phi/BigR /)
                
                zj = Psi
                
                Lap_Psi = Psi_xx + Psi_x/BigR + Psi_yy + Psi_phiphi/BigR**2
                Bv_parderiv_Bv_parderiv_Psi = chi(1,0,0)*(chi(2,0,0)*Psi_x + chi(1,0,0)*Psi_xx + chi(1,1,0)*Psi_y + chi(0,1,0)*Psi_xy &
                                            + (chi(1,0,1)*Psi_phi + chi(0,0,1)*Psi_xphi)/BigR**2 - 2.d0*chi(0,0,1)*Psi_phi/BigR**3) &
                                            + chi(0,1,0)*(chi(1,1,0)*Psi_x + chi(1,0,0)*Psi_xy + chi(0,2,0)*Psi_y + chi(0,1,0)*Psi_yy &
                                            + (chi(0,1,1)*Psi_phi + chi(0,0,1)*Psi_yphi)/BigR**2) &
                                            + chi(0,0,1)*(chi(1,0,1)*Psi_x + chi(1,0,0)*Psi_xphi + chi(0,1,1)*Psi_y + chi(0,1,0)*Psi_yphi &
                                            + (chi(0,0,2)*Psi_phi + chi(0,0,1)*Psi_phiphi)/BigR**2)/BigR
                
                amat_31 = -theta*v*(Bv2*Lap_Psi + dot_product(grad_Bv2,grad_Psi) - Bv_parderiv_Bv_parderiv_Psi)*dl*zbig
                amat_33 = theta*v*Bv2*zj*dl*zbig
                
                kl1 = index_kl
                kl3 = index_kl + 2*n_tor_local
                
                ELM(ij3,kl1) = ELM(ij3,kl1) + amat_31*wgauss(ms)
                ELM(ij3,kl3) = ELM(ij3,kl3) + amat_33*wgauss(ms)
              end do
            end do
          end do
        end do
      end do
    end do
  end do
end do

return
end subroutine boundary_matrix_open
end module mod_boundary_matrix_open
