!> Module containing routines to project a function of particles onto the JOREK
!> finite elements.
!>
!> The general routine allows the user to specify a transformation object
!> to calculate the quantity to be projected.
!> This quantity is multiplied by the particle weight behind the scenes.
!> Included transformation types
!>   - 1 (proj_one)
!>
!> These functions must subclass proj_transform in this module.
!> Other modules defining transformations:
!>   - tools/mod_radiation.f90
!>
!>
!> Projected results can be written to vtk or hdf5 by specifying the appropriate
!> options at creation time of project_particles_base.
!>
!> Projections are written to a node_list. The number of projections is thus limited
!> by the number of JOREK variables. If you need more, use multiple instances.
module mod_project_particles

use mod_io_actions
use data_structure
use phys_module, only: n_aux_var
use mod_particle_types
use mod_fields
use mod_projection_type, only: t_projection
#if STELLARATOR_MODEL
use mod_coupled_projection, only: assemble_system, project_only
#else
use mod_uncoupled_projection, only: assemble_system, project_only
#endif

implicit none
private
public projection, new_projection, sample_rhs
public write_particle_distribution_to_vtk, write_particle_distribution_to_h5 !< public for testing reasons, please don't use directly

type, extends(t_projection) :: projection
  contains
  procedure :: do => project
  procedure :: close_projection => close_projection
end type projection
interface projection
  module procedure new_projection
end interface projection

contains

function new_projection(node_list, element_list,                                                    &
                        filter,    filter_hyper,    filter_parallel,                                &
                        filter_n0, filter_hyper_n0, filter_parallel_n0,                             &
                        f, do_zonal, to_h5, to_vtk, index_h5,                                       &
                        nsub, filename, basename, decimal_digits, fractional_digits, calc_integrals,&
                        do_dirichlet) result(new)

  use phys_module, only: use_mumps_prj, use_pastix_prj, use_strumpack_prj
  use mod_sparse_data, only: type_SP_SOLVER, mumps, pastix, strumpack
  use mod_rhs_projections, only: proj_f

  use mpi_mod
  use mod_vtk

  type(projection) :: new
  type(type_node_list), intent(in)       :: node_list
  type(type_element_list), intent(in), target    :: element_list
  real*8, intent(in), optional           :: filter,    filter_hyper,    filter_parallel    !< normal, hyper and parallel smoothing
  real*8, intent(in), optional           :: filter_n0, filter_hyper_n0, filter_parallel_n0 !< for n=0
  type(proj_f), intent(in), dimension(:), optional :: f !< Type with function to map over particles before projection
  logical, intent(in), optional          :: do_zonal    !< solve zonal flow  system for n=0 instead of projection (false if omitted)
  logical, intent(in), optional          :: to_h5 !< Write HDF5 output after projecting (false if omitted)
  logical, intent(in), optional          :: to_vtk !< Write vtk output after projecting (false if omitted)
  logical, intent(in), optional          :: index_h5 !< numbering projection outputs in the same way as fluid output: e.g. projections00100.vtk(h5) (false if omitted)
  integer, intent(in), optional          :: nsub !< number of subdivisions of the finite elements
  character(len=*), intent(in), optional :: filename
  character(len=*), intent(in), optional :: basename
  integer, intent(in), optional          :: decimal_digits
  integer, intent(in), optional          :: fractional_digits
  logical, intent(in), optional          :: calc_integrals !< After projecting, calculate and print the integral of each projected quantity
  logical, intent(in), optional          :: do_dirichlet

  integer              :: ierr, my_nsub, inode, n_masters, i
  integer, allocatable :: i_tor(:)
  integer              :: i_rank(n_tor)

  ! for stellarators we solve just one big matrix system and therefore we need just one communicator
  call MPI_Comm_dup(MPI_COMM_WORLD, new%mpi_comm_world, ierr)

  call MPI_COMM_RANK(new%mpi_comm_world, new%my_id, ierr)
  call MPI_COMM_SIZE(new%mpi_comm_world, new%n_cpu, ierr)
#if !STELLARATOR_MODEL 
  ! in the tokamak case, we solve each uncoupled system indipendently and we therefore need indipendent communicators
  n_masters = (n_tor+1)/2
  if (MOD(new%n_cpu, n_masters) == 0) then
    new%m_cpu = new%n_cpu / (n_masters)
  else
    new%m_cpu = (new%n_cpu - MOD(new%n_cpu, n_masters))/n_masters +1
  end if

  if (allocated(i_tor)) call tr_deallocate(i_tor,"i_tor",CAT_UNKNOWN)
  call tr_allocate(i_tor,1,new%n_cpu,"i_tor",CAT_UNKNOWN)

  do i=1,new%n_cpu
    i_tor(i) = ((i-1) - MOD(i-1, new%m_cpu))/ new%m_cpu  + 1
  enddo

  call MPI_COMM_SPLIT(new%mpi_comm_world,i_tor(new%my_id+1),new%my_id,new%mpi_comm_n,ierr)
  
  do i=1,n_masters
    i_rank(i) = (i-1) * new%m_cpu
  enddo

  call MPI_COMM_GROUP(new%mpi_comm_world,new%mpi_group_world,ierr)
  call MPI_GROUP_INCL(new%mpi_group_world,n_masters,i_rank,new%mpi_group_master,ierr)
  call MPI_COMM_CREATE(new%mpi_comm_world,new%mpi_group_master,new%mpi_comm_master,ierr)
  call MPI_COMM_RANK(new%mpi_comm_n, new%my_id_n, ierr) ! id of this cpu in local comm
  
  if (i_tor(new%my_id+1) .eq. 1) then
    new%n_tor_local = 1
    new%i_tor_local = 1
  else
    new%n_tor_local = 2
    new%i_tor_local = 2*i_tor(new%my_id+1) - 2       ! i_tor_local is the (starting) index in HZ
  endif
#endif

  allocate(new%node_list,    source=node_list)
  call make_deep_copy_node_list(node_list, new%node_list)

  do inode = 1, new%node_list%n_nodes
    new%node_list%node(inode)%values = 0.d0
    new%node_list%node(inode)%deltas = 0.d0
  end do

  allocate(new%element_list, source=element_list)
  new%element_list = element_list

  new%filter             = 0.d0
  new%filter_hyper       = 0.d0
  new%filter_parallel    = 0.d0
  new%filter_n0          = 0.d0
  new%filter_hyper_n0    = 0.d0
  new%filter_parallel_n0 = 0.d0

  if (present(filter))             new%filter             = filter
  if (present(filter_hyper))       new%filter_hyper       = filter_hyper
  if (present(filter_parallel))    new%filter_parallel    = filter_parallel
  if (present(filter_n0))          new%filter_n0          = filter_n0
  if (present(filter_hyper_n0))    new%filter_hyper_n0    = filter_hyper_n0
  if (present(filter_parallel_n0)) new%filter_parallel_n0 = filter_parallel_n0

  new%do_zonal = .false.
  if (present(do_zonal)) new%do_zonal = do_zonal
  new%apply_dirichlet = .true.
  if (present(do_dirichlet)) new%apply_dirichlet = do_dirichlet

  if (present(to_vtk)) then
    if (to_vtk) then
      my_nsub = 4
      if (present(nsub)) my_nsub = nsub
      ! Precalculate the node positions in the vtk file and the connectivity
      if (new%my_id .eq. 0) then
        new%vtk_grid = vtk_grid(new%node_list, new%element_list, my_nsub)
      endif
      ! We don't set the extension here since this is dynamically set in the do
      ! action (to support both vtk and h5 output)
    end if
  end if
  if (present(to_h5)) new%to_h5 = to_h5
  if (present(index_h5)) new%index_h5 = index_h5

  new%basename = "proj"
  if (present(filename)) new%filename = filename
  if (present(basename)) new%basename = basename
  if (present(decimal_digits)) new%decimal_digits = decimal_digits
  if (present(fractional_digits)) new%fractional_digits = fractional_digits
  if (present(calc_integrals)) new%calc_integrals = calc_integrals
  new%name = "Project"
  new%log = .true.

  !----------------------------
#if STELLARATOR_MODEL
  call assemble_system(node_list, element_list,                           &
                        new%mpi_comm_world, new%a_mat,                     &
                        new%filter, new%filter_hyper, new%filter_parallel, &
                        new%apply_dirichlet, new%system_size, new%n_dof)
#else 
  call assemble_system(node_list, element_list,                                       &
                        new%n_tor_local, new%i_tor_local,                              & 
                        new%mpi_comm_world, new%mpi_comm_n, new%mpi_comm_master,       &
                        new%a_mat, new%area, new%volume,                               &
                        new%filter, new%filter_hyper, new%filter_parallel,             &
                        new%filter_n0, new%filter_hyper_n0, new%filter_parallel_n0,    &
                        new%integral_weights, new%do_zonal,                            &
                        new%apply_dirichlet, new%system_size, new%n_dof) 
#endif
  !----------------------------

  if (present(f)) then
    new%f = f
  endif

  new%solver%verbose = .true.
  new%solver%projection = .true.

  if (use_strumpack_prj) then
#ifdef USE_STRUMPACK
  new%solver%library = strumpack
#else
  write(*,*) ' FATAL : use_strumpack_prj requires defined USE_STRUMPACK'
  call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
#endif
  else if (use_pastix_prj) then
#ifdef USE_PASTIX
  new%solver%library = pastix
#else
  write(*,*) ' FATAL : use_pastix_prj requires defined USE_PASTIX'
  call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
#endif
  else if (use_mumps_prj) then
#ifdef USE_MUMPS
  new%solver%library = mumps
#else
  write(*,*) ' FATAL : use_mumps_prj requires defined USE_MUMPS'
  call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
#endif
endif

end function new_projection


subroutine project(this, sim, ev)
  use mod_event
  use phys_module, only: nout, nout_projection, index_now
  use mod_particle_sim, only: particle_sim

  class(projection), intent(inout)     :: this
  type(particle_sim), intent(inout)    :: sim
  type(event), intent(inout), optional :: ev

  ! Sample projection functions for each particle
  call sample_rhs(this, sim)

  ! Project all right-hand sides
  call project_only(this, sim)

  ! Some checks for 'nout_projection'
  if ((this%to_h5) .or. (allocated(this%vtk_grid))) then
    if (nout_projection .lt. 0) then
      nout_projection = nout
      if (this%my_id .eq. 0) then
        write(*,*) "WARNING: Trying to write projection output files without specifying 'nout_projection'"
        write(*,*) "         Projections will be written in every 'nout' timesteps"
      end if
    else if (.not. (mod(nout,nout_projection) .eq. 0)) then
      if (this%my_id .eq. 0) then
        write(*,*) "WARNING: Double check 'nout' and 'nout_projection' in the namelist"
        write(*,*) "         You will get staggered projection outputs with JOREK restart files"
      end if
    end if
  end if

  ! Save output if requested
  if (this%to_h5) then
    if (mod(index_now,nout_projection) .eq. 0) then
      call save_to_h5(this, sim)
    end if
  end if
  if (allocated(this%vtk_grid)) then
    if (mod(index_now,nout_projection) .eq. 0) then
      call save_to_vtk(this, sim)
    end if
  end if

  ! Clean up storage
  if (allocated(this%rhs)) this%rhs = 0.d0
  if (allocated(this%rhs_f)) deallocate(this%rhs_f)
end subroutine project

!> Add samples to the right-hand side, stored in `this` by calling this%f(i)%f
!> for every particle and saving the contribution multiplied by each of the basis
!> functions (poloidal and toroidal) in `this%rhs_f`
subroutine sample_rhs(this, sim)
  use mpi_mod
  use mod_event
  use mod_interp, only: mode_moivre, interp_RZ
  use constants, only: PI
  use mod_basisfunctions
  use mod_parameters, only: n_degrees
  use mod_particle_sim, only: particle_sim
  use phys_module, only: n_aux_var
  use mod_parameters, only: n_tor, n_vertex_max, n_degrees

  !$ use omp_lib
  class(projection), intent(inout)  :: this
  type(particle_sim), intent(inout) :: sim
  integer :: n_sample !< number of groups to sample
  real*8 :: HP(n_tor)
  real*8 :: v, R_g, R_s, R_t, Z_g, Z_s, Z_t, xjac, x(3), HH(4,n_degrees), HH_s(4,n_degrees), HH_t(4,n_degrees)
  integer :: i_group, m, i, j, im, im_index, i_f
  integer :: index_ij, i_tor
  ! For openmp reduce
  real*8, allocatable :: my_rhs(:,:,:,:,:)


  ! Safety checks
  if (.not. allocated(sim%groups)) return
  
  if (.not. allocated(this%f)) allocate(this%f(0))

  if (allocated(this%rhs)) then
    n_sample = min(size(this%f),n_aux_var-min(size(this%rhs,5),n_aux_var))  ! because we have only n_var storage for now
  else
    n_sample = min(size(this%f),n_aux_var)  ! because we have only n_var storage for now
  end if

  if (n_sample .lt. size(this%f) .and. sim%my_id .eq. 0) then
    write(*,*) 'WARNING: ignoring proj_f after ', n_sample, ' due to lack of output space'
  end if

  if (.not. allocated(this%rhs_f)) then
    if (.not. associated(this%element_list)) then
      write(*,*) "ERROR: Call the constructor for projection or allocate element_list"
      return
    end if

    allocate(this%rhs_f(n_vertex_max,n_degrees,this%element_list%n_elements,n_tor,n_sample))
  
  end if
  
  this%rhs_f = 0.d0


  ! We now write multiple omp regions since this allows us to do reductions per
  ! variable instead of all-at-once, helping with the stack size requirements
  ! there is however some thread-creation overhead

  allocate(my_rhs(n_vertex_max,n_degrees,this%element_list%n_elements,n_tor,1))

  do i_f=1,n_sample

    i_group = this%f(i_f)%group

    if (.not. allocated(sim%groups(i_group)%particles)) cycle
  
    my_rhs = 0.d0

    !$omp parallel do default(none)                                            &
    !$omp shared(this, sim, n_sample, i_group, i_f)                            &
    !$omp private(x, xjac, HH, HH_s, HH_t, R_g, R_s, R_t, Z_g, Z_s, Z_t,       &
    !$omp         m, i, j, i_tor, index_ij, v, HP),                            &
    !$omp reduction(+:my_rhs) schedule(dynamic,10)
    ! Note that the stack per thread can become larger than the 2MB default in
    ! this case. You could try setting the OMP_STACKSIZE environment variable
    ! somewhat larger than the size of my_rhs
    do m=1,size(sim%groups(i_group)%particles,1)

      if (sim%groups(i_group)%particles(m)%i_elm .le. 0) cycle

      x(1:2) = sim%groups(i_group)%particles(m)%st
      x(3)   = sim%groups(i_group)%particles(m)%x(3)
      
      call basisfunctions(x(1), x(2), HH, HH_s, HH_t)
      call mode_moivre(x(3), HP)
            
      do i=1,n_vertex_max
        do j=1,n_degrees
       
          v = HH(i,j) * this%element_list%element(sim%groups(i_group)%particles(m)%i_elm)%size(i,j)

          v = v * this%f(i_f)%f(sim, i_group, sim%groups(i_group)%particles(m)) * sim%groups(i_group)%particles(m)%weight
       
          do i_tor = 1, n_tor

            my_rhs(j,i,sim%groups(i_group)%particles(m)%i_elm,i_tor,1) = &
            my_rhs(j,i,sim%groups(i_group)%particles(m)%i_elm,i_tor,1) + HP(i_tor) * v
          
          enddo

        enddo
      enddo
    enddo

    !$omp end parallel do
    this%rhs_f(:,:,:,:,i_f) = my_rhs(:,:,:,:,1)
  enddo
 
  deallocate(my_rhs)

end subroutine sample_rhs

  !> Save an already-projected set to a vtk file with current parameters
subroutine save_to_vtk(this, sim)
  use mod_event
  use phys_module, only: index_now
  use mod_particle_sim, only: particle_sim
  use mod_import_restart, only: rst_file_ind_fmt
  use phys_module, only: n_aux_var

  !$ use omp_lib
  class(t_projection), intent(inout)  :: this
  type(particle_sim), intent(inout) :: sim
  integer :: i, ierr, n_proj
  real*8 :: t0, t1, ostart, oend
  character(len=120) :: filename, tmp_name

  this%extension = '.vtk'
  if (.not. allocated(this%vtk_grid)) then
    write(*,*) "ERROR: Trying to write unprepared vtk file"
    return
  end if

  if (.not. this%index_h5) then ! put file name with physical time
    if (len_trim(this%filename) .eq. 0) then
      filename = this%get_filename(sim%time)
    else
      filename = this%filename
    end if
  else ! put file name with 'index_now'
    write(tmp_name,rst_file_ind_fmt(1)) trim(this%basename), index_now
    filename = trim(tmp_name)//this%extension
  end if

  call cpu_time(t0)
  !$ ostart = omp_get_wtime()
  if (sim%my_id .eq. 0) then
    ! write only on the host

    n_proj = size(this%f)
    if (allocated(this%rhs)) n_proj = n_proj + size(this%rhs,5)

    call write_particle_distribution_to_vtk(this%node_list, this%element_list, &
      trim(filename), this%vtk_grid%nsub, min(n_proj,n_aux_var), this%vtk_grid%xyz, this%vtk_grid%ien)
      
    write(*,*) "Written projection to ", trim(filename)
  end if
  call cpu_time(t1)
  !$ oend = omp_get_wtime()
#ifdef DEBUG
  if (sim%my_id .eq. 0 ) then
    write(*,"(A,2g12.5)") "writing cpu time", t1-t0
    !$ write(*,"(A,2g12.5)") "writing wall time", oend-ostart
  end if
#endif
end subroutine save_to_vtk

!> Action for projecting all particles and writing output to a hdf5 file
subroutine save_to_h5(this, sim)
  use mpi_mod
  use mod_event
  use phys_module, only: index_now
  use mod_particle_sim, only: particle_sim
  use mod_import_restart, only: rst_file_ind_fmt
  use phys_module, only: n_aux_var

  !$ use omp_lib
  class(t_projection),  intent(inout)  :: this
  type(particle_sim), intent(inout)  :: sim
  integer :: my_id, ierr, n_proj
  character(len=120) :: filename, tmp_name
  real*8 :: t0, t1, ostart, oend

  this%extension = '.h5'

  if (.not. this%index_h5) then ! put file name with physical time
    if (len_trim(this%filename) .eq. 0) then
      filename = this%get_filename(sim%time)
    else
      filename = this%filename
    end if
  else ! put file name with 'index_now'
    write(tmp_name,rst_file_ind_fmt(1)) trim(this%basename), index_now
    filename = trim(tmp_name)//this%extension
  end if

  call cpu_time(t0)
  !$ ostart = omp_get_wtime()
  call MPI_COMM_RANK(MPI_COMM_WORLD, my_id, ierr)
  if (my_id .eq. 0) then
    ! write only on the host
    n_proj = size(this%f)
    if (allocated(this%rhs)) n_proj = n_proj + size(this%rhs,5)
    call write_particle_distribution_to_h5(this%node_list, this%element_list, &
      trim(filename), min(n_proj,n_aux_var), sim%time)

    write(*,*) "Written projection to ", trim(filename)
  end if
  call cpu_time(t1)
  !$ oend = omp_get_wtime()
#ifdef DEBUG
  if (my_id .eq. 0 ) then
    write(*,"(A,2g12.5)") "writing cpu time", t1-t0
    !$ write(*,"(A,2g12.5)") "writing wall time", oend-ostart
  end if
#endif
end subroutine save_to_h5

!> Save a particle distribution in a sort of minimized JOREK restart format.
!> This contains only: n_tor, n_period, n_fields->n_var, n_vertex_max, vertex, x, size, n_elements, values
!> neighbours, RCS_version
!> model=-1 to signify that the variables have not the expected meaning
subroutine write_particle_distribution_to_h5(node_list,element_list,filename,n_fields,time)
use data_structure
use hdf5
use hdf5_io_module
use tr_module
!$ use omp_lib
implicit none

!> Input parameters
type(type_node_list), intent(in)    :: node_list
type(type_element_list), intent(in) :: element_list
character*(*), intent(in)           :: filename
integer, intent(in) :: n_fields !< number of different particle groups to output
real*8, intent(in) :: time

 
#include "version.h"
! --- Local variables
integer :: i
character*50             :: version_control

integer(HID_T)     :: file_id
integer            :: ierr

! type_node, node_list%n_nodes
real(RKIND), allocatable :: t_x(:,:,:,:)                   ! n_coord_tor, n_degrees, n_dim
real(RKIND), allocatable :: t_values(:,:,:,:)              !       n_tor, n_degrees, n_fields

! element, element_list%n_elements
integer,     allocatable :: t_vertex(:,:)                  ! n_vertex_max
integer,     allocatable :: t_neighbours(:,:)              ! n_vertex_max
real(RKIND), allocatable :: t_size(:,:,:)                  ! n_vertex_max,n_degrees

! type_node, node_list%n_nodes
call tr_allocate(t_x,     1,node_list%n_nodes,1,n_coord_tor,1,n_degrees,1,n_dim,   "node_list%x",     CAT_UNKNOWN)
call tr_allocate(t_values,1,node_list%n_nodes,1,n_tor,      1,n_degrees,1,n_fields,"node_list%values",CAT_UNKNOWN)

! element_list%n_elements
call tr_allocate(t_vertex,    1,element_list%n_elements,1,n_vertex_max,"vertex",CAT_UNKNOWN)
call tr_allocate(t_neighbours,1,element_list%n_elements,1,n_vertex_max,"neighbours",CAT_UNKNOWN)
call tr_allocate(t_size,      1,element_list%n_elements,1,n_vertex_max,1,n_degrees,"size",CAT_UNKNOWN)

do i=1,node_list%n_nodes
   t_x(i,:,:,:)      = node_list%node(i)%x
   t_values(i,:,:,:) = node_list%node(i)%values(:,:,1:n_fields)
end do

do i=1,element_list%n_elements
  t_vertex(i,:)       = element_list%element(i)%vertex
  t_neighbours(i,:)   = element_list%element(i)%neighbours
  t_size(i,:,:)       = element_list%element(i)%size
end do

! -> Create and open HDF5 file
call HDF5_create(trim(filename),file_id,ierr)
if (ierr.ne.0) then
  write(*,*) ' ==> error for opening of HDF5 file',filename
end if
  
! -> Save version of revision control system
write(version_control,'(A)') trim(adjustl(RCS_VERSION))
version_control = trim(adjustl(version_control))
call HDF5_char_saving(file_id,version_control,"RCS_version"//char(0))

! -> Save parameters
call HDF5_integer_saving(file_id,-1,'jorek_model'//char(0)) ! Indicate that this is not a normal JOREK model
call HDF5_integer_saving(file_id,n_fields,'n_var'//char(0))
call HDF5_integer_saving(file_id,n_dim,'n_dim'//char(0))
call HDF5_integer_saving(file_id,n_order,'n_order'//char(0))
call HDF5_integer_saving(file_id,n_tor,'n_tor'//char(0))
call HDF5_integer_saving(file_id,n_coord_tor,'n_coord_tor'//char(0))
call HDF5_integer_saving(file_id,n_period,'n_period'//char(0))
call HDF5_integer_saving(file_id,n_vertex_max,'n_vertex_max'//char(0))
call HDF5_integer_saving(file_id,n_nodes_max,'n_nodes_max'//char(0))
call HDF5_integer_saving(file_id,n_elements_max,'n_elements_max'//char(0))

! -> 
call HDF5_integer_saving(file_id,node_list%n_nodes,'n_nodes'//char(0))
call HDF5_integer_saving(file_id,element_list%n_elements,'n_elements'//char(0))
call HDF5_integer_saving(file_id,node_list%n_dof,'n_dof'//char(0))

call HDF5_array4D_saving(file_id,t_x, &
     node_list%n_nodes,n_coord_tor,n_degrees,n_dim,'x'//char(0))
call HDF5_array4D_saving(file_id,t_values, &
     node_list%n_nodes,n_tor,n_degrees,n_fields,'values'//char(0))

call HDF5_array2D_saving_int(file_id,t_vertex, &
     element_list%n_elements,n_vertex_max,'vertex'//char(0))
call HDF5_array2D_saving_int(file_id,t_neighbours, &
     element_list%n_elements,n_vertex_max,'neighbours'//char(0))
call HDF5_array3D_saving(file_id,t_size, &
     element_list%n_elements,n_vertex_max,n_degrees,'size'//char(0))
call HDF5_real_saving(file_id,time,'t_now'//char(0))

! -> close file
call HDF5_close(file_id)
end subroutine write_particle_distribution_to_h5


!> Helper subroutine to write a particle distribution, saved in variables 1:n,
!> to `filename`. `nsub` is the number of subdivisions to make per element.
!> Should be called only by process 1
subroutine write_particle_distribution_to_vtk(node_list,element_list,filename,nsub,n_fields,xyz,ien)
  use data_structure
  use phys_module, only: mode, n_tor, n_plane, n_period
  use mod_vtk
  use mod_interp
  use basis_at_gaussian, only: HZ
  use constants, only: PI

  !> Input parameters
  type(type_node_list), intent(in)    :: node_list
  type(type_element_list), intent(in) :: element_list
  character*(*), intent(in)           :: filename
  integer, intent(in) :: nsub         !< Number of subdivisions of each element for visualization
  integer, intent(in) :: n_fields     !< number of different particle groups to output
  real*4, intent(in)  :: xyz(:,:,:)
  integer, intent(in) :: ien(:,:,:)

  integer :: n_nodes, i, j, k, l, m, mp, inode, ivar
  real*4, allocatable :: scalars(:,:,:), vectors(:,:,:)
  ! No vectors in this example, but could be added:
  ! real*4, allocatable :: vectors(:,:,:)
  integer :: n_scalars, n_vectors = 0
  character*36, allocatable :: scalar_names(:), vector_names(:)

  character*100 :: i_filename
  
  real*8 :: s, t, toroidal_angle
  real*8 :: P, R, Z

  integer, parameter :: vtk_quad_type = 9 ! VTK_QUAD

  n_scalars = (n_tor+1) * n_fields

  n_nodes =  nsub*nsub*element_list%n_elements

  allocate(scalar_names(n_scalars),vector_names(n_vectors))

#ifdef STELLARATOR_MODEL
  allocate(scalars(n_plane, n_nodes, n_scalars), vectors(n_nodes,3,n_vectors))
#else
  allocate(scalars(1, n_nodes, n_scalars), vectors(n_nodes,3,n_vectors))
#endif

  ivar = 0
  do i = 1, n_fields  ! Particle group
    do k = 1, n_tor+1 ! Toroidal mode component
      ivar = ivar + 1
      if (k == 1) then
        write(scalar_names(ivar), '(A,I0.2,A)') "rho_", i, "_0"
      else if (k == n_tor+1) then
        write(scalar_names(ivar), '(A,I0.2,A,I0.2,A,I0.2,A,I0.3)') "allsum_rho_", i
      else if (mod(k, 2) == 0) then ! cosine_mode
        write(scalar_names(ivar), '(A,I0.2,A,I0.2,A,I0.2,A,I0.3)') "rho_", i, "_cos", mode(k)
      else ! sine_mode
        write(scalar_names(ivar), '(A,I0.2,A,I0.2,A,I0.2,A,I0.3)') "rho_", i, "_sin", mode(k)
      endif
    end do
  end do
  
  if (ivar /= n_scalars) then
      print *, "Error: Mismatch in number of scalars and generated names."
      stop
  endif

  scalars(:,:,:) = 0.e0
  vectors        = 0.e0

#ifndef STELLARATOR_MODEL

  ! Create points for each element
  !$omp parallel do default(none) &
  !$omp shared(element_list,nsub,node_list,n_fields,scalars) &
  !$omp private(i,j,k,l,m,inode,ivar,s,t,P,R,Z) schedule(static)
  do i=1,element_list%n_elements
    do j=1,n_fields
      do k=1,n_tor+1
        ivar = (j-1)*(n_tor+1) + k
        do l=1,nsub
          s = real(l-1,8)/real(nsub-1,8)
          do m=1,nsub
            t = real(m-1,8)/real(nsub-1,8)
            inode = (i-1)*nsub*nsub+(l-1)*nsub+m
            if (k == n_tor+1) then  ! add only the cosine terms (i.e. sum at phi=0)
              scalars(1,inode, ivar) = scalars(1, inode, (j-1)*(n_tor+1) + 1)  & 
                                     + sum(scalars(1, inode, (j-1)*(n_tor+1)+2 : (j-1)*(n_tor+1)+n_tor:2))
            else
              call interp(node_list, element_list, i, j, k, s, t, P)
              scalars(1,inode,ivar) = real(P,4)
            end if
          end do
        end do
      end do
    end do
  end do
  !$omp end parallel do

#else

  do mp=1, n_plane

    toroidal_angle = real((mp - 1), 8) * 2.d0 * PI / n_plane / n_period 

    ! Create points for each element
    !$omp parallel do default(none) &
    !$omp shared(element_list,nsub,node_list,n_fields,scalars,HZ, mp, toroidal_angle) &
    !$omp private(i,j,k,l,m,inode,ivar,s,t,P,R,Z) schedule(static)
    do i=1,element_list%n_elements
      do j=1,n_fields
        do k=1,n_tor+1
          ivar = (j-1)*(n_tor+1) + k
          do l=1,nsub
            s = real(l-1,8)/real(nsub-1,8)
            do m=1,nsub
              t = real(m-1,8)/real(nsub-1,8)
              inode = (i-1)*nsub*nsub+(l-1)*nsub+m
              if (k == n_tor+1) then 
                scalars(mp,inode, ivar) = sum(scalars(mp, inode, (j-1)*(n_tor+1)+1 : (j-1)*(n_tor+1)+n_tor))
              else
                call interp(node_list, element_list, i, j, k, s, t, P)
                scalars(mp,inode,ivar) = real(P * HZ(k,mp), 4)
              end if
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do

  end do
#endif

#ifdef STELLARATOR_MODEL
  do mp=1,n_plane
    write(i_filename, '(A,A,I0.3,A)') filename(:len_trim(filename) - 4), "_proj_plane", mp, ".vtk"
    call write_vtk(i_filename, xyz(mp,:,:), ien(mp,:,:), vtk_quad_type, scalar_names, scalars(mp,:,:))
  enddo
#else 
  call write_vtk(filename, xyz(1,:,:), ien(1,:,:), vtk_quad_type, scalar_names, scalars(1,:,:))
#endif

  deallocate(scalars, scalar_names, vectors, vector_names)

end subroutine write_particle_distribution_to_vtk


subroutine close_projection(this)
  class(projection), intent(inout) :: this
  call this%solver%finalize()
end subroutine close_projection

end module mod_project_particles
