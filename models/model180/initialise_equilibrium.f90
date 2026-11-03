!> import MHD equilibrium for model 180 from compatable equilibrium solvers
!!
!! Currently it is possible to import:
!!  i) GVEC equilibrium data
subroutine initialise_equilibrium(my_id,node_list,element_list,bnd_node_list, bnd_elm_list, ierr)

use mod_import_gvec
use phys_module
use mod_boundary_conditions
use mod_boundary,            only: boundary_from_grid
use mod_poiss
use equil_info
#ifdef USE_NO_TREE
  use mod_no_tree
#elif USE_QUADTREE
  use mod_quadtree
#else
  use mod_element_rtree, only: populate_element_rtree
#endif

implicit none

integer,                      intent(in)    :: my_id
type (type_node_list),        intent(inout) :: node_list
type (type_element_list),     intent(inout) :: element_list
type (type_bnd_node_list),    intent(inout) :: bnd_node_list
type (type_bnd_element_list), intent(inout) :: bnd_elm_list

integer                                     :: ierr


if (my_id .eq. 0) then
  element_list%n_elements      = 0
  bnd_elm_list%n_bnd_elements  = 0
  node_list%n_nodes            = 0
  
  ! load GVEC data
  if (gvec_grid_import) then
    
    write(*,*)  "Reading GVEC Import..."
    call import_gvec(node_list, element_list, 'gvec2jorek.dat', ierr)
    if (ierr /= 0) then
      write(*, *) 'Error in import gvec routine'
      stop
    end if
    write(*,*)  "Finished GVEC Import..."
  else
    write(*,*) "Incompatable equilibrium import option specified"
    stop
  end if ! gvec_grid_import
    
  call plot_grid(node_list,element_list,bnd_elm_list,bnd_node_list,.true.,.false.,'axisym')
  call plot_grid_3d(node_list,element_list,bnd_elm_list,bnd_node_list,.true.,.false.,'initial')
  call plot_povray_3d(node_list,element_list,bnd_elm_list,bnd_node_list,.true.,.false.)
  
  ! Initialise boundary and initial conditions for T and rho
  call boundary_from_grid(node_list, element_list, bnd_node_list, bnd_elm_list, .false.) 
  
  call populate_element_rtree(node_list, element_list)
  call update_equil_state(my_id, node_list, element_list, bnd_elm_list, xpoint, xcase)
  call print_equil_state(.true.)
  call save_special_points('special_equilibrium_points.dat', .false., ierr)
  
  call initial_conditions(my_id,node_list,element_list,bnd_node_list, bnd_elm_list, xpoint,xcase)
    
#ifdef USE_DOMM
  call solve_Psi_boundary_eqn(node_list, bnd_elm_list, ierr)
  if (ierr == 3) then
    write(*,*) 'Error in solve_Psi_boundary_eqn routine, no fields_xyz.dat file found.'
    return
  end if
  call setup_boundary_condition(node_list, bnd_node_list)
#else
#ifndef USE_EXT_FIELD
  if ( n_plane < 2*(n_coord_tor-1) ) then
    write(*,*) 'Vacuum field solve requires n_plane > 2*(n_coord_tor-1) to avoid aliasing!'
    stop
  endif
  call poisson(my_id,4,node_list,element_list,bnd_node_list,bnd_elm_list,1,1,1, &
               0.0,1.0,xpoint,xcase,(/ -99.0, 99.0 /),freeboundary_equil,refinement,1)   !----------- for GS use -1
#endif  ! USE_EXT_FIELD
#endif

  if (my_id .eq. 0) call determine_boundary_flux(node_list, element_list)
end if ! my_id .eq. 0

return
end subroutine initialise_equilibrium
