module mod_flux_grid



implicit none



contains



!< Create a grid from parameters n_flux, n_pol
subroutine flux_grid(node_list, element_list, bnd_node_list, bnd_elm_list, my_id, n_mpi)
  use phys_module
  use data_structure
  use mpi_mod
  use vacuum
  use vacuum_response
  use vacuum_equilibrium,  only: import_external_fields
  use mod_boundary,        only: boundary_from_grid
  use mod_element_rtree,   only: populate_element_rtree
  use mod_exchange_indices
  
  implicit none
  
  type(type_node_list),        intent(inout) :: node_list
  type(type_element_list),     intent(inout) :: element_list
  type(type_bnd_node_list),    intent(inout) :: bnd_node_list
  type(type_bnd_element_list), intent(inout) :: bnd_elm_list
  integer,                     intent(in)    :: my_id
  integer,                     intent(in)    :: n_mpi

  type (type_surface_list) :: surface_list
  integer                  :: list_to_be_refined(n_ref_list), n_to_be_refined    
  integer                  :: ierr

  if (my_id == 0) then

    if (xpoint)  then

      if ( (xcase .ge. UPPER_XPOINT) .or. (grid_to_wall .and. (n_wall_blocks .gt. 0)) .or. RZ_grid_inside_wall ) then
        if (grid_to_wall) then
          if ( (xcase .eq. UPPER_XPOINT) .and. (n_wall_blocks .eq. 0) ) then
            if(my_id == 0 ) call grid_upper_xpoint_wall(node_list,element_list,n_flux,n_open,n_up_priv,n_up_leg,n_up_leg_out,  &
                                                    n_tht,n_ext,xr_closed,SIG_open,SIG_closed,SIG_up_priv,SIG_theta_up,SIG_up_leg_0, &
                                                    SIG_up_leg_1,dPSI_open,dPSI_up_priv)
          else
            call grid_double_xpoint_inside_wall(node_list, element_list)
          endif
        else
          call grid_double_xpoint(node_list, element_list)
        endif
      else
  
        if (.not. grid_to_wall) then
          call grid_xpoint(node_list,element_list,n_flux,n_open,n_private,n_leg,n_tht,xr_closed,   &
                           SIG_open,SIG_closed,SIG_private,SIG_theta,SIG_leg_0,SIG_leg_1,dPSI_open,dPSI_private, xcase)
        else
!!rks only for ITER wall for the moment
 !        write(*,*) 'ITER wall started'
          if(my_id == 0 ) call grid_xpoint_wall(node_list,element_list,n_flux,n_open,n_private,n_leg,n_leg_out,n_tht, n_ext,xr_closed,  &
                                SIG_open,SIG_closed,SIG_private,SIG_theta,SIG_leg_0,SIG_leg_1,dPSI_open,dPSI_private)
        endif !  if (.not. grid_to_wall) then
         
      endif !if (xcase .ge. 2) then
               
        call plot_grid(node_list,element_list,bnd_elm_list,bnd_node_list,.false.,.false.,'xpoint')
      
    else ! (if xpoint)
      
      call grid_flux_surface(xpoint,xcase, node_list, element_list, surface_list, n_flux, n_tht,     &
                             xr1, sig1, xr2, sig2,refinement)
      
      call plot_grid(node_list, element_list, bnd_elm_list, bnd_node_list, .true., .false.,'fluxsurface')
      
      ! --- Refine elements (equilibrum)
      if (refinement) then
        n_to_be_refined=0
        call Refine_Elem_List(node_list, element_list, list_to_be_refined, n_to_be_refined)
        call Ref_Update_Index(element_list, node_list)
      end if
         
    end if ! (if xpoint)

    ! --- Optional: Add patches to an existing grid imported from restart file
    if (extend_existing_grid) &
        call grid_patches_on_existing_grid(node_list, element_list)

    if ( freeboundary .and. freeb_change_indices ) call exchange_indices(node_list, my_id, n_mpi, .false.)

    ! --- Determine boundary information from the grid
    call boundary_from_grid(node_list, element_list, bnd_node_list, bnd_elm_list, .false.) 
    call export_boundary(node_list, bnd_elm_list, bnd_node_list)

  endif ! if (my_id == 0) then        

  ! --- Check sanity of grid
  call check_grid(my_id, node_list, element_list)

  call broadcast_boundary(my_id, bnd_elm_list, bnd_node_list) 

end subroutine flux_grid



end module mod_flux_grid
