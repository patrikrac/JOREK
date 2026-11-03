subroutine initial_conditions(my_id,node_list,element_list,bnd_node_list, bnd_elm_list, xpoint2, xcase2)
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
use data_structure
use phys_module
use mod_poiss
use mod_interp, only: interp
use constants, only: mu_zero
implicit none

type (type_node_list)        :: node_list
type (type_element_list)     :: element_list
type (type_surface_list)     :: surface_list
type (type_bnd_node_list)    :: bnd_node_list
type (type_bnd_element_list) :: bnd_elm_list

integer    :: my_id, i, i2, j, in, mm, i_elm_axis, i_elm_xpoint(2), ifail, i_elm, xcase2
real*8     :: amplitude, psi, psi_axis
real*8     :: zn, dn_dpsi, dn_dpsi2, dn_dz, dn_dz2, dn_dpsi_dz, dn_dpsi3, dn_dpsi2_dz, dn_dpsi_dz2
real*8     :: zT, dT_dpsi, dT_dpsi2, dT_dz, dT_dz2, dT_dpsi_dz, dT_dpsi3, dT_dpsi2_dz, dT_dpsi_dz2
real*8     :: Phi, dPhi_dpsi,dPhi_dz,dPhi_dpsi2,dPhi_dz2,dPhi_dpsi_dz,dPhi_dpsi3,dPhi_dpsi_dz2, dPhi_dpsi2_dz 
real*8     :: R_axis, Z_axis, s_axis, t_axis, R, Z, BigR
real*8     :: R_out, Z_out, s_out, t_out, R_lim, Z_lim, s_lim, t_lim, psi_lim
real*8     :: psi_n, psi_bnd
real*8     :: p_s, p_t, p_ss, p_st, p_tt
logical    :: xpoint2
real*8     :: s_norm, s_factor
real*8     :: Z_xpoint(2) =  (/ 0.0, 0.0 /)

if (my_id .eq. 0) then
  write(*,*) '***************************************'
  write(*,*) '*      initial conditions  (180)      *'
  write(*,*) '***************************************'
endif

if (my_id .eq. 0) then
  s_factor = 1.0 * bloating_factor / (n_flux - 1)

  do i=1,node_list%n_nodes

    psi = node_list%node(i)%values(1,1,var_Psi)
    R   = node_list%node(i)%x(1,1,1)
    Z   = node_list%node(i)%x(1,1,2)
    s_norm = node_list%node(i)%r_tor_eq(1)

    ! Initialise density
    call density(    xpoint2, xcase2, Z, Z_xpoint, s_norm, 0.0, 1.0, zn, dn_dpsi, dn_dz, dn_dpsi2, dn_dz2, dn_dpsi_dz, dn_dpsi3, dn_dpsi_dz2, dn_dpsi2_dz)
    node_list%node(i)%values(1,1,var_rho) = zn
    node_list%node(i)%values(1,2,var_rho) = dn_dpsi * s_factor * 1.0 / 3.0
    node_list%node(i)%values(1,3,var_rho) = 0.d0
    node_list%node(i)%values(1,4,var_rho) = 0.d0
    
    ! Initialise temperature based on density and pressure
    if (with_TiTe) then
      node_list%node(i)%values(1,1,var_Ti) = TiTe_ratio*mu_zero*node_list%node(i)%pressure(1) / zn
      node_list%node(i)%values(1,2,var_Ti) = mu_zero * (TiTe_ratio*node_list%node(i)%pressure(2) - TiTe_ratio*node_list%node(i)%pressure(1) / zn * node_list%node(i)%values(1,2,5)) / zn   ! dT/ds = (dp/ds - dn/ds*T) / n  ... scale factor is already taken into account
      node_list%node(i)%values(1,3,var_Ti) = 0.d0
      node_list%node(i)%values(1,4,var_Ti) = 0.d0

      node_list%node(i)%values(1,1,var_Te) = (1.d0-TiTe_ratio)*mu_zero*node_list%node(i)%pressure(1) / zn
      node_list%node(i)%values(1,2,var_Te) = mu_zero * ((1.d0-TiTe_ratio)*node_list%node(i)%pressure(2) - (1.d0-TiTe_ratio)*node_list%node(i)%pressure(1) / zn * node_list%node(i)%values(1,2,5)) / zn   ! dT/ds = (dp/ds - dn/ds*T) / n  ... scale factor is already taken into account
      node_list%node(i)%values(1,3,var_Te) = 0.d0
      node_list%node(i)%values(1,4,var_Te) = 0.d0
    else
      node_list%node(i)%values(1,1,var_T) = mu_zero*node_list%node(i)%pressure(1) / zn
      node_list%node(i)%values(1,2,var_T) = mu_zero * (node_list%node(i)%pressure(2) - node_list%node(i)%pressure(1) / zn * node_list%node(i)%values(1,2,5)) / zn   ! dT/ds = (dp/ds - dn/ds*T) / n  ... scale factor is already taken into account
      node_list%node(i)%values(1,3,var_T) = 0.d0
      node_list%node(i)%values(1,4,var_T) = 0.d0
    end if

    call potential_source(xpoint2, xcase2, Z, Z_xpoint, s_norm, 0.0, 1.0, Phi, dPhi_dpsi,dPhi_dz,dPhi_dpsi2,dPhi_dz2,dPhi_dpsi_dz,dPhi_dpsi3,dPhi_dpsi_dz2, dPhi_dpsi2_dz)
    ! values(i_tor, i_dof, i_variable)
    node_list%node(i)%values(1,1,2) = Phi
    node_list%node(i)%values(1,2,2) = dPhi_dpsi * s_factor * 1.0 / 3.0
    node_list%node(i)%values(1,3,2) = 0.d0
    node_list%node(i)%values(1,4,2) = 0.d0 
  enddo

endif

! Set T_0, or Ti_0 and Te_0 based on density, pressure and t_rat on axis
call density(    xpoint2, xcase2, Z, Z_xpoint, 0.0, 0.0, 1.0, zn, dn_dpsi, dn_dz, dn_dpsi2, dn_dz2, dn_dpsi_dz, dn_dpsi3, dn_dpsi_dz2, dn_dpsi2_dz)
do i=1,node_list%n_nodes
  if (node_list%node(i)%axis_node) then
    if (with_TiTe) then
      Ti_0 = TiTe_ratio*mu_zero*node_list%node(i)%pressure(1) / zn
      Te_0 = (1.d0-TiTe_ratio)*mu_zero*node_list%node(i)%pressure(1) / zn
      write(*,*) "Calculated values of Ti_0, Te_0 (to be set in input files!): ", Ti_0, Te_0
    else
      write(*,*) "Calculated value of T_0 (to be set in input files!): ", T_0
      T_0 = mu_zero*node_list%node(i)%pressure(1) / zn
    endif
    exit
  endif
enddo


! initialise perturbations
amplitude = 1.d-10
mm = 2

do in=2,n_tor

  if (my_id .eq. 0) then

    do i=1,node_list%n_nodes

      node_list%node(i)%values(in,:,:) = 0.d0

      node_list%node(i)%deltas = 0.d0

    enddo

  endif

enddo

#ifdef altcs
do i=1,node_list%n_nodes
  node_list%node(i)%psi_eq(:) = node_list%node(i)%values(1,:,1)
end do
#endif

!call add_pellet(node_list,element_list,50.,0.06,0.02,R_axis-0.96,Z_axis)

return
end
