!> mod_particle_io_mpi_test contains andata and procedures
!> for testing the writing and the reading of particles
!> data to/from HDF5 files (MPI enables).
module mod_particle_io_mpi_test
use fruit
use mpi
use mod_particle_sim, only: particle_sim
implicit none

private
public :: run_fruit_particle_io_mpi

!> Variables --------------------------------------------
type(particle_sim)          :: sim_particles
logical,parameter           :: test=.true.
character(len=28),parameter :: test_filename="test_particle_io_mpi_hdf5.h5"
character(len=37),parameter :: test_filename_gatherv="test_particle_io_mpi_hdf5_gatherv.h5"
!> the number of particle groups is set equal to the number
!> of particle types for testing all of them
!> particle_gc_Qin is commented because the I/O for particle_gc_Qin
!> has not been implemented yet
integer,parameter :: fill_type=-1
integer,parameter :: n_groups=8
integer,parameter :: n_particles=5 !< N# of particles per group per task
real*8,parameter  :: tol_real8=1.d-15
integer           :: rank_loc,n_tasks_loc,ifail_loc
integer           :: mpi_comm_test,mpi_info_test
logical           :: use_hdf5_access_properties,mpio_collective
logical           :: use_native_hdf5_mpio

!> Interfaces -------------------------------------------
contains

!> Fruit basket -----------------------------------------
!> run_fruit_particle_io_mpi performs the set-up, 
!> execution and tear-down of test features
!> inputs:
!>   rank:    (integer) mpi task rank
!>   n_tasks: (integer) number of tasks in the commworld
!>   ifail:   (integer) 0 if success
!> outputs:
!>   ifail:   (integer) 0 if success
subroutine run_fruit_particle_io_mpi(rank,n_tasks,ifail)
  implicit none
  !> inputs
  integer,intent(in) :: rank,n_tasks
  !> inputs-outputs
  integer,intent(inout) :: ifail
  if(rank.eq.0) write(*,'(/A)') "  ... setting-up: particle io mpi tests"
  call setup(rank,n_tasks,ifail)
  if(rank.eq.0) write(*,'(/A)') "  ... running: particle io mpi tests"
  call run_test_case(test_particle_mpi_io_write_native_read,'test_particle_mpi_io_write_native_read')
  call run_test_case(test_particle_mpi_io_write_gatherv_read,'test_particle_mpi_io_write_gatherv_read')
  call run_test_case(test_get_simulation_hdf5_time_write_native,'test_get_simulation_hdf5_time_write_native')
  call run_test_case(test_get_simulation_hdf5_time_write_gatherv,'test_get_simulation_hdf5_time_write_gatherv')
  if(rank.eq.0) write(*,'(/A)') "  ... tearing-down: particle io mpi tests"
  call teardown(rank,n_tasks,ifail)
end subroutine run_fruit_particle_io_mpi

!> Set-up and tear-down ---------------------------------
!> set-up the test features
!> inputs:
!>   rank:    (integer) mpi task rank
!>   n_tasks: (integer) number of tasks in the commworld
!>   ifail:   (integer) 0 if success
!> outputs:
!>   ifail:   (integer) 0 if success
subroutine setup(rank,n_tasks,ifail)
  use mod_particle_common_test_tools, only: fill_particles
  use mod_particle_common_test_tools, only: sim_time_interval
  use mod_particle_common_test_tools, only: fill_groups
  use mod_particle_types,           only: particle_kinetic,particle_kinetic_leapfrog
  use mod_particle_types,           only: particle_gc,particle_fieldline
  use mod_particle_types,           only: particle_kinetic_relativistic
  use mod_particle_types,           only: particle_gc_relativistic
  use mod_particle_types,           only: particle_gc_vpar,particle_gc_Qin
  use mod_particle_io,              only: write_simulation_hdf5
  use mod_gnu_rng,                  only: gnu_rng_interval
  implicit none
  !> inputs
  integer,intent(in) :: rank,n_tasks
  !> variables
  integer :: ii,jj

  !> inputs-outputs
  integer,intent(inout) :: ifail

  !> store mpi variables
  rank_loc = rank; n_tasks_loc = n_tasks; ifail_loc = ifail;
  mpi_comm_test = MPI_COMM_WORLD; call MPI_Info_create(mpi_info_test,ifail);
  use_hdf5_access_properties=.false.; mpio_collective=.true.;
  use_native_hdf5_mpio=.true.

  !> initialize the particle simulation
  call sim_particles%initialize(.false.,rank,n_tasks,.false.,.true.,n_groups)

  !> set and broadcast simulation time
  if(rank.eq.0) then
    call gnu_rng_interval(sim_time_interval,sim_particles%time)
  endif
  call MPI_Bcast(sim_particles%time,1,MPI_REAL8,0,mpi_comm_test,ifail)

  !> allocate particle lists for different particle types
  allocate(particle_fieldline::sim_particles%groups(1)%particles(n_particles))
  allocate(particle_gc::sim_particles%groups(2)%particles(n_particles))
  allocate(particle_gc_vpar::sim_particles%groups(3)%particles(n_particles))
  allocate(particle_kinetic::sim_particles%groups(4)%particles(n_particles))
  allocate(particle_kinetic_leapfrog::sim_particles%groups(5)%particles(n_particles))
  allocate(particle_kinetic_relativistic::sim_particles%groups(6)%particles(n_particles))
  allocate(particle_gc_relativistic::sim_particles%groups(7)%particles(n_particles))
  allocate(particle_gc_Qin::sim_particles%groups(8)%particles(n_particles))

  !> fill-up the group and particle base variables
  call fill_groups(n_groups,sim_particles%groups,rank,ifail)
  call fill_particles(n_groups,sim_particles%groups,fill_type,rank)

  !> write filename with native and gatherv methods
  call write_simulation_hdf5(sim_particles,trim(test_filename),&
  use_native_hdf5_mpio_in=use_native_hdf5_mpio,use_hdf5_access_properties=use_hdf5_access_properties,&
  collective_mpio_in=mpio_collective,mpi_comm_in=mpi_comm_test,mpi_info_in=mpi_info_test)
  call write_simulation_hdf5(sim_particles,trim(test_filename_gatherv),&
  use_native_hdf5_mpio_in=.false.,use_hdf5_access_properties=.true.,&
  mpi_comm_in=mpi_comm_test)
end subroutine setup

!> tear-down the test simulation features
!> inputs:
!>   rank:    (integer) mpi task rank
!>   n_tasks: (integer) number of tasks in the commworld
!>   ifail:   (integer) 0 if success
!> outputs:
!>   ifail:   (integer) 0 if success
subroutine teardown(rank,n_tasks,ifail)
  use mod_common_test_tools, only: remove_file
  implicit none
  !> inputs
  integer,intent(in) :: rank,n_tasks
  !> inputs-outputs
  integer,intent(inout) :: ifail

  call MPI_Barrier(mpi_comm_test,ifail)
  !> remove test file
  call remove_file(trim(test_filename),rank_in=rank_loc)
  call remove_file(trim(test_filename_gatherv),rank_in=rank_loc)
  call MPI_Info_free(mpi_info_test,ifail);
  rank_loc = -1; n_tasks_loc = -1; mpi_comm_test = -1; 
  use_hdf5_access_properties=.true.; ifail = ifail_loc;
end subroutine teardown

!> Tests ------------------------------------------------
!> procedure for testing the particle io native writier/reader
subroutine test_particle_mpi_io_write_native_read
  use mod_particle_assert_equal,      only: assert_equal_particle_group
  use mod_particle_sim,               only: particle_sim
  use mod_particle_io,                only: read_simulation_hdf5
  implicit none
  !> variables
  type(particle_sim) :: sim_particles_new
  integer :: ii
  real*8 :: comp_real8_1,comp_real8_2

  !> initialize parameters
  use_hdf5_access_properties = .false.

  !> initialize the new particle simulation
  call sim_particles_new%initialize(.false.,rank_loc,n_tasks_loc,.false.,.true.,n_groups)

  !> read default simulation from file and store in new sim
  call read_simulation_hdf5(sim_particles_new,trim(test_filename),&
  use_hdf5_access_properties=use_hdf5_access_properties,&
  mpi_comm_in=mpi_comm_test,mpi_info_in=mpi_info_test,test_in=test)

  !> check simulation 
  call assert_equals(sim_particles_new%time,sim_particles%time,tol_real8,&
  "Error writing (native)/reading  particle simulation: time mismatch!")

  !> check groups
  call assert_equal_particle_group(n_groups,sim_particles_new%groups,sim_particles%groups)
end subroutine test_particle_mpi_io_write_native_read

!> procedure for testing the particle io gatherv writier/ reader
subroutine test_particle_mpi_io_write_gatherv_read
  use mod_particle_assert_equal,      only: assert_equal_particle_group
  use mod_particle_sim,               only: particle_sim
  use mod_particle_io,                only: read_simulation_hdf5
  implicit none
  !> variables
  type(particle_sim) :: sim_particles_new
  integer :: ii
  real*8 :: comp_real8_1,comp_real8_2

  !> initialize parameters
  use_hdf5_access_properties = .false.;

  !> initialize the new particle simulation
  call sim_particles_new%initialize(.false.,rank_loc,n_tasks_loc,.false.,.true.,n_groups)

  !> read default simulation from file and store in new sim
  call read_simulation_hdf5(sim_particles_new,trim(test_filename_gatherv),&
  use_hdf5_access_properties=use_hdf5_access_properties,&
  mpi_comm_in=mpi_comm_test,mpi_info_in=mpi_info_test,test_in=test)

  !> check simulation 
  call assert_equals(sim_particles_new%time,sim_particles%time,tol_real8,&
  "Error writing (gatherv)/reading particle simulation: time mismatch!")

  !> check groups
  call assert_equal_particle_group(n_groups,sim_particles_new%groups,sim_particles%groups)
end subroutine test_particle_mpi_io_write_gatherv_read

!> subroutine for testing the reading of simulation time with writer native
subroutine test_get_simulation_hdf5_time_write_native()
  use mod_particle_io, only: get_simulation_hdf5_time
  implicit none
  real*8 :: time_new
  !> initialize parameters
  use_hdf5_access_properties = .false.
  !> read time
  time_new = get_simulation_hdf5_time(trim(test_filename),&
  use_hdf5_access_properties=use_hdf5_access_properties,&
  mpi_comm_loc=mpi_comm_test,mpi_info_loc=mpi_info_test,n_mpi=n_tasks_loc)
  call assert_equals(sim_particles%time,time_new,tol_real8,&
  "Error get simulation time hdf5 (write hdf5 native): time mismatch!")
end subroutine test_get_simulation_hdf5_time_write_native

!> subroutine for testing the reading of simulation time new with writer gatherv
subroutine test_get_simulation_hdf5_time_write_gatherv()
  use mod_particle_io, only: get_simulation_hdf5_time
  implicit none
  real*8 :: time_new
  !> initialize parameters
  use_hdf5_access_properties = .false.
  !> read time
  time_new = get_simulation_hdf5_time(trim(test_filename_gatherv),&
  use_hdf5_access_properties=use_hdf5_access_properties,&
  mpi_comm_loc=mpi_comm_test,mpi_info_loc=mpi_info_test,n_mpi=n_tasks_loc)
  call assert_equals(sim_particles%time,time_new,tol_real8,&
  "Error get simulation time (write gatherv): time mismatch!")
end subroutine test_get_simulation_hdf5_time_write_gatherv

!>-------------------------------------------------------
end module mod_particle_io_mpi_test
