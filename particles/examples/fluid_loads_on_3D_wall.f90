!> What does this code do?
!>    Calculates fluid fluxes (heat and current loads) on a 3D thin wall discretized with triangles.
!>    The wall loads are exported in VTK format (3D_wall_fluid_loads.vtk)
!> How?
!>    It works like the SMITER code, it traces Field Lines (FLs) from the wall triangles towards the plasma.
!>    If the FLs do not intersect a wall component after a given distance (l_par_min), then the wall triangle
!>    is considered wetted by the plasma.
!>    If a triangle is wet, the perpendicular heat flux, current density and other quantities are given
!>    there by calling expressions from jorek2_postroc
!> Usage?
!>    ./executable < input_file_jorek
!>    l_par_min is hardcoded, you may want to adapt it to your tokamak
!> Required inputs?
!>    You need to provide two walls in .h5 format
!>        1. wall_to_load.h5 : This is the wall where FL tracing will be done and the fluxes will be calcaulted
!>        2. wall.h5         : This wall is the wall with which field lines intersect, normally you will add here
!>                             structures around the loaded wall + the loaded structure itself
!>    The format for these files is explained here https://www.jorek.eu/wiki/doku.php?id=particles_wall_load
!> Other important things to know
!>    1. For the FL tracing to be effective, the wall triangles must be within the JOREK domain, otherwise those
!>    triangles are considered shadowed (not connected to the plasma) and there are no loads there. So choose
!>    your wall carefully
!>    2. Be careful with the ordering of the wall triangles, we assume the normals point towards the plasma

Program  fluid_loads_on_3D_wall

use particle_tracer
use mod_particle_io
use mod_particle_diagnostics
use mod_fields_linear   
use mod_fields_hermite_birkhoff
use mod_gc_relativistic
use mod_kinetic_relativistic
use mod_coordinate_transforms, only: cylindrical_to_cartesian, cartesian_to_cylindrical, vector_cartesian_to_cylindrical 
use mod_wall_collision
use hdf5_io_module
use constants, only: PI
use phys_module, only : sqrt_mu0_rho0, F0
use equil_info, only: ES, update_equil_state
use mod_new_diag, only: init_new_diag
use mod_expression
use exec_commands,  only: expr_list, clean_up
use mod_position
use nodes_elements
use mod_boundary

implicit none

! --- Important hard-coded parameters for the simulation
real*8,  parameter :: tstep_field = 1.d-2  !< 
real*8,  parameter :: l_par_min   = 4.9d0  !< Distance above which a field line is considered to connect the plasma and the wall (wetted wall)
integer, parameter :: n_steps = 1000       !< Maximum number of steps to advance field lines
integer, parameter :: n_sbnd  = 20         !< Number of boundary element subdivisions to find closest point to the boundary
integer, parameter :: filehandle = 60      !< File handle for vtk file
logical, parameter :: map_wall_to_JOREK_bnd = .true. !< Maps 3D wall points to JOREK's 2D boundary to evaluate parallel fluxes,
                                                     !< otherwise fluxes are calculated inside the domain at exact given locations
! --- Related to the input wall
type(octree_node) :: wall
type(octree_triangle), allocatable :: triangles(:)
integer(HID_T)                     :: file_id
integer*4                          :: n_part, max_depth, wall_id, n_tri
integer*4, allocatable             :: indices(:,:)
real*8, allocatable                :: wtmp(:),nodes_xyz(:,:), normals_all(:,:)

! --- Vectors
real*8, dimension(3) :: pos_prev, wall_pos, xyz_tria, norm_tria, &
                        v21, v31, Bphi_v, xyz_prev, xyz, r_cyl, norm_cyl, B
! --- Others
integer :: i, j, k, l, ifail
real*8  :: phi, norm_R, dir_sign 

! --- To calculate closest point to the boundary from the wall
real*8, allocatable :: R_elm(:), Z_elm(:), distance(:)   
real*8              :: R1, R_s, R_t, Z1, Z_s, Z_t, R_min, Z_min, smin, tmin, dist_min, s, t
integer             :: ierr, i_elm, i_best(2), i_min, i_s, side, i_elm_min, i_bnd

! --- To calculate quantities with posptroc commmands
type(t_pol_pos),      pointer :: pos
type(t_pol_pos_list), target  :: pol_pos_list
type(t_tor_pos_list)          :: tor_pos_list
integer                       :: n_nodes, n_tri_wet, i_count
real*8, allocatable           :: result(:,:,:,:), iangle(:,:), l_part(:), q_heat_perp_3d(:), field_wall_angle(:)
real*8                        :: BR, BZ, Btor, Btot, Jpar, qpar, s_bnd
logical                       :: s_const

  ! --- Ensure that aux_node_list is associated
if (.not. associated(aux_node_list)) allocate(aux_node_list)
 
! --- Read wall for collisions with octree
max_depth = 6
call mod_wall_collision_init('wall.h5',max_depth,wall)

! --- Read wall to initialize particles (this is the wall where loads are calculated)
file_id = 1
call HDF5_open('wall_to_load.h5',file_id)
call HDF5_integer_reading(file_id,n_tri,'ntriangle')

allocate(wtmp(9*n_tri))
call HDF5_array1D_reading(file_id, wtmp, 'nodes')

allocate(indices(n_tri,3))
call HDF5_array2D_reading_int(file_id, indices, 'indices')
call HDF5_close(file_id)

! --- Convert wall array into triangles
allocate(triangles(n_tri))
do i=1,n_tri
   triangles(i)%triangle_id = i
   triangles(i)%v0 = (/ wtmp( (i-1)*9 + 1 ), wtmp( (i-1)*9 + 2 ), wtmp( (i-1)*9 + 3 ) /)
   triangles(i)%v1 = (/ wtmp( (i-1)*9 + 4 ), wtmp( (i-1)*9 + 5 ), wtmp( (i-1)*9 + 6 ) /)
   triangles(i)%v2 = (/ wtmp( (i-1)*9 + 7 ), wtmp( (i-1)*9 + 8 ), wtmp( (i-1)*9 + 9 ) /)
end do

call sim%initialize(num_groups=1)

! Run first event to read the JOREK fields
events = [event(read_jorek_fields_interp_linear(i=-1))] !< i=-1 means read the jorek_restart.h5 file
call with(sim, events, at=0.d0)

! ------------- Initialize FL particles (one per wall triangle) ---------------------------------------------------------
sim%groups(:)%Z    = 1
sim%groups(:)%mass = 2.d0 !< atomic mass units

n_part = n_tri  !< As many particle-field lines as wall triangles

allocate(particle_fieldline::sim%groups(1)%particles(n_tri))
allocate(normals_all(n_tri,3))  !--- Wall triangle normals

do i=1, n_tri
  
  select type (p=>sim%groups(1)%particles(i))
  
  type is (particle_fieldline)
  
   !--- Calculate wall normals (we assume they point towards the domain)
   v21 = triangles(i)%v1 - triangles(i)%v0 
   v31 = triangles(i)%v2 - triangles(i)%v0 

   norm_tria = [v21(2)*v31(3)- v21(3)*v31(2), v21(3)*v31(1)- v21(1)*v31(3), v21(1)*v31(2)- v21(2)*v31(1)]

   norm_tria        = norm_tria / norm2(norm_tria)
   normals_all(i,:) = norm_tria
   
   ! Assign initial particle-FL position to triangle center
   xyz_tria = (triangles(i)%v0 + triangles(i)%v1 + triangles(i)%v2) / 3.d0

   p%x = cartesian_to_cylindrical(xyz_tria)

   norm_R = (xyz_tria(1)*norm_tria(1) + xyz_tria(2)*norm_tria(2))/p%x(1)
  !write(121,'(6ES16.6)') p%x(1), p%x(2), p%x(3), norm_R, norm_tria(3)

   call find_RZ(sim%fields%node_list, sim%fields%element_list, &
            p%x(1), p%x(2), &
            p%x(1), p%x(2), p%i_elm, p%st(1), p%st(2), ifail)
   
   !if (ifail /= 0) write(*,*) 'Initial position of triangle id = ', i, ' not found in JOREK grid, move the wall inside'

   p%weight = 1.0
   p%v      = 1.d0 ! --- not really used

   end select
end do

allocate(iangle(1,n_part))
allocate(l_part(n_part))
iangle = 0
! ----------- end paticle initialization -----------------------------------------------------------------------

! ----------- Start field line tracing -------------------------------------------------------------------------
select type (particles => sim%groups(1)%particles)
type is (particle_fieldline)	
    l_part = 0.d0
    !$omp parallel do default(private) &
    !$omp shared (sim, wall, iangle, l_part, normals_all)
    do j=1,size(particles,1)

      do k=1,n_steps

          if(particles(j)%i_elm .le. 0) exit !--- Don't trace lost particles
          if (l_part(j) > l_par_min)   exit  !--- If the FL is able to move away X m from the wall w/o collision, it is wet

          pos_prev = particles(j)%x

          ! --- Do a step and check if the particle moves in the normal direction (away from the wall), otherwise correct direction
          if (k==1) then
            call field_line_runge_kutta_fixed_dt_push_jorek(sim%fields, particles(j), sim%time, tstep_field*0.1d0)
            xyz_prev = cylindrical_to_cartesian(pos_prev)
            xyz      = cylindrical_to_cartesian(particles(j)%x)

            if (dot_product(xyz-xyz_prev,normals_all(j,:)) > 0.d0) then
              dir_sign =  1.d0 
            else
              dir_sign = -1.d0
            endif
          endif
          ! --- Advance the field line
          call field_line_runge_kutta_fixed_dt_push_jorek(sim%fields, particles(j), sim%time, tstep_field*dir_sign)

          if (any(isnan(pos_prev)) .or. any(isnan(particles(j)%x))) cycle

          xyz_prev = cylindrical_to_cartesian(pos_prev)
          xyz      = cylindrical_to_cartesian(particles(j)%x)

          if (particles(j)%i_elm .le. 0) exit   !< If the particle leaves the domain, stop tracing it

          l_part(j) = l_part(j) + norm2(xyz-xyz_prev)

          if (k < 3) cycle  ! don't check collisions during the initial steps to allow the particle to move away from the wall
          call mod_wall_collision_check(pos_prev, particles(j)%x, wall, wall_id, wall_pos, iangle(1,j))
          if(wall_id .gt. 0) then
            particles(j)%x      = wall_pos
            particles(j)%i_elm  = -wall_id
          end if
      end do
    end do
end select

call mod_wall_collision_free(wall)

! ----------- End field line tracing --------------------------------------------------------------------------

!call write_simulation_hdf5(sim, 'part_out.h5')
!call mod_wall_collision_export(sim, 'wallload.h5', iangle)

! --- Initialize postproc tools
call init_new_diag(.false.)
call clean_up()

! --- Allocate and initialize some values
allocate( q_heat_perp_3d(n_part), field_wall_angle(n_part) )
q_heat_perp_3d(:)   = 0.d0
field_wall_angle(:) = 0.d0

! --- Count wet wall triangles
n_tri_wet = 0
do i=1, n_tri
  if (l_part(i) > l_par_min) then
    n_tri_wet = n_tri_wet + 1
  endif
enddo

! --- Additional initialization required for postproc
call boundary_from_grid(sim%fields%node_list, sim%fields%element_list, bnd_node_list, bnd_elm_list, .false.)
call update_equil_state(0,sim%fields%node_list, sim%fields%element_list, bnd_elm_list, xpoint, xcase)

! --- Get list of local coordinates for wet wall triangles for postproc
! --- Initialize list of poloidal positions
call alloc_pol_pos(pol_pos_list, (/1, n_tri_wet /))
pol_pos_list%has_dedicated_tor_pos = .true.

! --- Get R, Z coordinates of the middle of the boundary elements (necessary to find closest point to bnd later)
allocate(R_elm(bnd_elm_list%n_bnd_elements), Z_elm(bnd_elm_list%n_bnd_elements))
allocate(distance(bnd_elm_list%n_bnd_elements))
do i_bnd = 1, bnd_elm_list%n_bnd_elements
  i_elm = bnd_elm_list%bnd_element(i_bnd)%element 
  call interp_RZ(sim%fields%node_list, sim%fields%element_list, i_elm, 0.5d0, 0.5d0, R1, R_s, R_t, Z1, Z_s, Z_t)
  R_elm(i_bnd) = R1
  Z_elm(i_bnd) = Z1
enddo

i_count = 0
do i=1, n_tri

  if (l_part(i) < l_par_min) cycle  

  i_count = i_count + 1

  ! Field line position is at triangle center
  xyz_tria = (triangles(i)%v0 + triangles(i)%v1 + triangles(i)%v2) / 3.d0

  r_cyl   = cartesian_to_cylindrical(xyz_tria)

  pos     => pol_pos_list%pos(1,i_count)
  pos%R   = r_cyl(1)
  pos%Z   = r_cyl(2)
  pos%phi = r_cyl(3)

  if (map_wall_to_JOREK_bnd) then

    ! --- Find closest point to the boundary of the domain
    distance  = sqrt( (R_elm-pos%R)**2  + (Z_elm-pos%Z)**2)
    i_best(1) = minloc(distance,dim=1)
    dist_min  = distance(i_best(1))
    
    distance(i_best(1)) = 1d99             ! Mask the minimum value to find the second minimum
    i_best(2)           = minloc(distance, dim=1)
    distance(i_best(1)) = dist_min      ! Restore the original minimum distance value

    dist_min = 1.d99

    ! Go along two best boundary elements and find closest local coordinate
    do i_min=1, 2
      side   = bnd_elm_list%bnd_element(i_best(i_min))%side
      i_elm  = bnd_elm_list%bnd_element(i_best(i_min))%element
      ! Go along discretized element
      do i_s=1, n_sbnd
        s_bnd = float(i_s-1)/float(n_sbnd-1)
        call get_st_on_bnd(s_bnd, side, s, t, s_const)
        call interp_RZ(sim%fields%node_list, sim%fields%element_list, i_elm, s, t, R1, R_s, R_t, Z1, Z_s, Z_t)
        if ( sqrt((R1-pos%R)**2  + (Z1-pos%Z)**2) < dist_min ) then
          R_min     = R1
          Z_min     = Z1
          i_elm_min = i_elm
          smin      = s
          tmin      = t
          dist_min  = sqrt((R1-pos%R)**2  + (Z1-pos%Z)**2)
        endif
      end do
    enddo

    ! Assign point to closest boundary point (to evaluate expressions)
    pos%R    = R_min
    pos%Z    = Z_min
    pos%ielm = i_elm_min
    pos%s    = smin
    pos%t    = tmin
  
  else
    ! Directly evaluate the fluxes at the given points of the 3D wall
    call find_RZ(sim%fields%node_list, sim%fields%element_list, &
            pos%R, pos%Z, &
            pos%R, pos%Z, pos%ielm, pos%s, pos%t, ifail)
            
  endif ! mapping points to 2D boundary or not

  call fill_pol_pos(pos, sim%fields%node_list, sim%fields%element_list)
end do

tor_pos_list = tor_pos(nphi=1) 

!--- Evaluate a list of expressions
expr_list = exprs((/'Psi_N   ', 'BR      ', 'BZ      ', 'Btor    ', 'Psi     ', 'ne      ', &
  'T_e     ', 'Jpar    ', 'qpar_tot'/), 9)
call eval_expr(ES, 1, expr_list, pol_pos_list, tor_pos_list, result, ierr)

! --- Write results to vtk
open(filehandle, file='3D_wall_fluid_loads.vtk', status='replace', action='write')
140 format(a)
141 format(a,i8,a)
142 format(3es16.8)
143 format(a,2i8)
144 format(4i8)
write(filehandle,140) '# vtk DataFile Version 2.0'
write(filehandle,140) 'testdata'
write(filehandle,140) 'ASCII'
write(filehandle,140) 'DATASET POLYDATA'

n_nodes = maxval(indices) + 1
write(*,*) n_nodes
allocate(nodes_xyz(n_nodes,3))
do i = 1, n_tri
   nodes_xyz(indices(i,1)+1,:) = triangles(i)%v0
   nodes_xyz(indices(i,2)+1,:) = triangles(i)%v1
   nodes_xyz(indices(i,3)+1,:) = triangles(i)%v2
end do

! --- Triangle node positions
write(filehandle,141) 'POINTS', n_nodes, ' float'
do i = 1, n_nodes
  write(filehandle,142) nodes_xyz(i,:)*1000
end do

! --- Node indices corresponding to triangles
write(filehandle,143) 'POLYGONS', n_tri, n_tri * 4
do i = 1, n_tri
  write(filehandle,144) 3, indices(i,:) 
end do

! --- Write cell data
write(filehandle,141) 'CELL_DATA', n_tri
write(filehandle,140) 'SCALARS Jperp[A/m2] float'
write(filehandle,140) 'LOOKUP_TABLE default'

i_count = 0
do i = 1, n_tri
   if (l_part(i)>l_par_min) then
      i_count = i_count + 1
      BR   = result(1,1,i_count,2)
      BZ   = result(1,1,i_count,3)
      Btor = result(1,1,i_count,4)
      Jpar = result(1,1,i_count,8)
      qpar = result(1,1,i_count,9)
      B    = (/ BR, BZ, Btor /)

      Btot = norm2( B ) 

      xyz_tria = (triangles(i)%v0 + triangles(i)%v1 + triangles(i)%v2) / 3.d0
      r_cyl    = cartesian_to_cylindrical(xyz_tria)
      phi      = r_cyl(3)
      norm_cyl = vector_cartesian_to_cylindrical(phi,  normals_all(i,:) ) 

      q_heat_perp_3d(i)   = abs( qpar * sum( norm_cyl(:) * B(:) ) / Btot )
      field_wall_angle(i) = ASIN(abs( sum( norm_cyl(:) * B(:) ) / Btot ))*180.d0/PI

      write(filehandle,142) Jpar * sum( norm_cyl(:) * B(:) ) / Btot 
   else 
      write(filehandle,142) 0.d0
   endif
 end do

write(filehandle,140) 'SCALARS L_pre_collision[m] float'
write(filehandle,140) 'LOOKUP_TABLE default'

do i = 1, n_tri
  write(filehandle,142) l_part(i) 
end do

write(filehandle,140) 'SCALARS q_perp[W/m2] float'
write(filehandle,140) 'LOOKUP_TABLE default'
do i = 1, n_tri
  write(filehandle,142) q_heat_perp_3d(i)
end do

write(filehandle,140) 'SCALARS B_wall_angle[deg] float'
write(filehandle,140) 'LOOKUP_TABLE default'
do i = 1, n_tri
  write(filehandle,142) field_wall_angle(i)
end do


! --- Close file, clean up
close(filehandle)

deallocate(iangle)

call clean_up()
! Finalize the simulation
call sim%finalize

end program fluid_loads_on_3D_wall

