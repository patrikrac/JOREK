!> This module contains some testcases for hdf5 io
module mod_sim_hdf5_io_spec_mpi_test
use fruit
use mod_io_actions
use mod_particle_sim
use mod_particle_types
implicit none
private
public :: run_fruit_sim_hdf5_io_spec_mpi
!> Variables --------------------------------------
integer,parameter :: master_rank=1
integer           :: rank_loc,n_tasks_loc,ifail_loc
integer           :: mpi_comm_loc,mpi_info_loc,file_access
contains
!> Fruit basket -----------------------------------
subroutine run_fruit_sim_hdf5_io_spec_mpi(rank,n_tasks,ifail)
  implicit none
  integer,intent(inout) :: ifail
  integer,intent(in)    :: rank,n_tasks
  if(rank.eq.master_rank) write(*,'(/A)') "  ... setting-up: "
  call setup(rank,n_tasks,ifail)
  if(rank.eq.master_rank) write(*,'(/A)') "  ... running: "
  call run_test_case(test_get_filename,'test_get_filename')
  call run_test_case(test_write_native_read_sim_time,'test_write_native_read_sim_time')
  call run_test_case(test_write_gatherv_read_sim_time,'test_write_gatherv_read_sim_time')
  call run_test_case(test_write_native_sim_one_particle_kinetic_leapfrog,&
  'test_write_native_sim_one_particle_kinetic_leapfrog')
  call run_test_case(test_write_gatherv_sim_one_particle_kinetic_leapfrog,&
  'test_write_gatherv_sim_one_particle_kinetic_leapfrog')
  call run_test_case(test_write_native_sim_one_group_boris,'test_write_native_sim_one_group_boris')
  call run_test_case(test_write_gatherv_sim_one_group_boris,'test_write_gatherv_sim_one_group_boris')
  call run_test_case(test_write_native_sim_all_particles,'test_write_native_sim_all_particles')
  call run_test_case(test_write_gatherv_sim_all_particles,'test_write_gatherv_sim_all_particles')
  call run_test_case(test_write_native_sim_two_groups_boris,'test_write_native_sim_two_groups_boris')
  call run_test_case(test_write_gatherv_sim_two_groups_boris,'test_write_gatherv_sim_two_groups_boris')
  call run_test_case(test_time_loop_gather_sim_two_groups_boris,'test_time_loop_gather_sim_two_groups_boris')
  if(rank.eq.master_rank) write(*,'(/A)') "  ... tearing-down: "
  call teardown(rank,n_tasks,ifail)
end subroutine run_fruit_sim_hdf5_io_spec_mpi

!> Set-up and Tear-down
subroutine setup(rank,n_tasks,ifail)
  use mpi
  use hdf5, only: H5F_ACC_TRUNC_F
  implicit none
  integer,intent(inout) :: ifail
  integer,intent(in) :: rank,n_tasks
  rank_loc     = rank
  n_tasks_loc  = n_tasks
  ifail_loc    = ifail
  mpi_comm_loc = MPI_COMM_WORLD
  mpi_info_loc = MPI_INFO_NULL
  file_access  = H5F_ACC_TRUNC_F
end subroutine setup

subroutine teardown(rank,n_tasks,ifail)
  implicit none
  integer,intent(inout) :: ifail
  integer,intent(in) :: rank,n_tasks
  ifail = ifail_loc
  n_tasks_loc = 0; rank_loc = 0;
  mpi_comm_loc = 0
  mpi_info_loc = 0
  file_access  = 0
end subroutine teardown

!> Tests ------------------------------------------
!> Test the filename generating routine for times > 0
subroutine test_get_filename
  class(write_action), allocatable :: writer
  allocate(writer)

  call assert_equals('part000.00000000.h5', trim(writer%get_filename(0.d0)), 'test default settings')
  call assert_equals('part001.10000000.h5', trim(writer%get_filename(1.1d0)), 'test default settings 2')
  writer%decimal_digits = 2; writer%fractional_digits = 0
  call assert_equals('part21.h5', trim(writer%get_filename(21d0)), 'decimal point should be removed if fractional_digits = 0')
  writer%decimal_digits = 0; writer%fractional_digits = 0
  call assert_equals('part.h5', trim(writer%get_filename(12.d0)), 'test without numbers')
  writer%decimal_digits = 1; writer%fractional_digits = 3
  call assert_equals('part1.123.h5', trim(writer%get_filename(1.123d0)), 'test manual format')
  call assert_equals('part*.123.h5', trim(writer%get_filename(13.123d0)), 'test manual format overflow')
  writer%basename = 'testing'; writer%extension = '.rst'
  call assert_equals('testing2.123.rst', trim(writer%get_filename(2.123d0)), 'test manual format with extension and basename')
  deallocate(writer)
end subroutine test_get_filename

subroutine test_write_native_read_sim_time
  use mpi
  implicit none
  real*8,parameter                 :: filename_time=21.19d0
  character(len=9),parameter       :: expected_filename='part21.h5'
  type(particle_sim)               :: sim_to_write, sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  allocate(writer, reader)
  sim_to_write%time = filename_time; sim_to_write%my_id = rank_loc;
  sim_to_write%n_mpi = n_tasks_loc;
  writer%decimal_digits = 2; writer%fractional_digits = 0
  writer%mpi_comm_io = mpi_comm_loc; writer%mpi_info_io = mpi_info_loc;
  writer%file_access = file_access; writer%use_native_hdf5_mpio = .true.;
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  reader%filename = expected_filename; reader%use_hdf5_access_properties=.false.;
  reader%mpi_comm_io = mpi_comm_loc; reader%mpi_info_io = mpi_info_loc; 
  call reader%run(sim_to_read)
  ! Test that the right time was read
  call assert_equals(sim_to_write%time, sim_to_read%time, "(native) time should be read from the file")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
end subroutine test_write_native_read_sim_time

subroutine test_write_gatherv_read_sim_time
  use mpi
  implicit none
  real*8,parameter                 :: filename_time=21.19d0
  character(len=9),parameter       :: expected_filename='part21.h5'
  type(particle_sim)               :: sim_to_write, sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  allocate(writer, reader)
  sim_to_write%time = filename_time; sim_to_write%my_id = rank_loc;
  sim_to_write%n_mpi = n_tasks_loc; writer%use_native_hdf5_mpio = .false.;
  writer%decimal_digits = 2; writer%fractional_digits = 0;
  writer%file_access = file_access; writer%mpi_info_io = mpi_info_loc;
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  reader%filename = expected_filename; reader%use_hdf5_access_properties=.false.;
  reader%mpi_comm_io = mpi_comm_loc; reader%mpi_info_io = mpi_info_loc;
  call reader%run(sim_to_read)
  ! Test that the right time was read
  call assert_equals(sim_to_write%time, sim_to_read%time, "time should be read from the file (write gatherv)")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
end subroutine test_write_gatherv_read_sim_time

subroutine test_write_native_sim_one_particle_kinetic_leapfrog
  use mpi
  implicit none
  integer,parameter                :: n_groups_expect=1
  integer,dimension(1),parameter   :: n_particles_expect=(/1/)
  real*8,parameter                 :: filename_time=20d0
  character(len=19),parameter      :: expected_filename='part020.00000000.h5'
  type(particle_sim)               :: sim_to_write, sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  integer :: i, n_groups, n_particles
  allocate(writer, reader) 
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  call allocate_particles_here(sim_to_write%groups(1)%particles, n_particles_expect(1))
  sim_to_write%time = filename_time; sim_to_write%my_id = rank_loc;
  sim_to_write%n_mpi = n_tasks_loc; sim_to_write%groups(1)%Z = 10;
  sim_to_write%groups(1)%mass = 1d3
  writer%mpi_comm_io = mpi_comm_loc; writer%mpi_info_io = mpi_info_loc;
  writer%file_access = file_access; writer%use_native_hdf5_mpio = .true.;
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc; 
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc;
  call reader%run(sim_to_read)
  ! Test that we have the right stuff in sim_to_read now
  call groups_same(sim_to_write,sim_to_read,n_groups_expect,n_particles_expect,&
  " (native write /native read)")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_native_sim_one_particle_kinetic_leapfrog

subroutine test_write_gatherv_sim_one_particle_kinetic_leapfrog
  use mpi
  implicit none
  integer,parameter                :: n_groups_expect=1
  integer,dimension(1),parameter   :: n_particles_expect=(/1/)
  real*8,parameter                 :: filename_time=30d0
  character(len=19),parameter      :: expected_filename='part030.00000000.h5'
  type(particle_sim)               :: sim_to_write
  type(particle_sim)               :: sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  integer :: i, n_groups, n_particles
  allocate(writer, reader); 
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.)
  call allocate_particles_here(sim_to_write%groups(1)%particles, n_particles_expect(1))
  sim_to_write%time = filename_time; sim_to_write%my_id = rank_loc;
  sim_to_write%n_mpi = n_tasks_loc; sim_to_write%groups(1)%Z = 10;
  sim_to_write%groups(1)%mass = 1d3; writer%file_access = file_access;&
  writer%use_native_hdf5_mpio = .false.; writer%mpi_comm_io = mpi_comm_loc;
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc;
  call reader%run(sim_to_read);
  ! Test that we have the right stuff in sim_to_read now
  call groups_same(sim_to_write,sim_to_read,n_groups_expect,&
  n_particles_expect," (gatherv write / native read)")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_gatherv_sim_one_particle_kinetic_leapfrog

subroutine test_write_native_sim_one_group_boris
  use mpi
  implicit none
  integer,parameter                :: n_groups_expect=1
  integer,dimension(1),parameter   :: n_particles_expect=(/2/)
  real*8,parameter                 :: filename_time=21d0
  character(len=19),parameter      :: expected_filename='part021.00000000.h5'
  type(particle_sim)               :: sim_to_write, sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  integer :: i, n_groups, n_particles
  allocate(writer, reader)
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.)
  call allocate_particles_here(sim_to_write%groups(1)%particles, n_particles_expect(1))
  sim_to_write%my_id = rank_loc; sim_to_write%n_mpi = n_tasks_loc;
  sim_to_write%time = filename_time; sim_to_write%groups(1)%Z = 2;
  sim_to_write%groups(1)%mass = 2.0; writer%mpi_comm_io = mpi_comm_loc; 
  writer%mpi_info_io = mpi_info_loc; writer%file_access = file_access; 
  writer%use_native_hdf5_mpio = .true.; call writer%run(sim_to_write);
  call MPI_Barrier(mpi_comm_loc,ifail_loc)
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc;
  call reader%run(sim_to_read)
  ! Test that we have the right stuff in sim_to_read now
  call groups_same(sim_to_write,sim_to_read,n_groups_expect,&
  n_particles_expect,"(native writer/native reader)")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_native_sim_one_group_boris

subroutine test_write_gatherv_sim_one_group_boris
  use mpi
  implicit none
  integer,parameter                :: n_groups_expect=1
  integer,dimension(1),parameter   :: n_particles_expect=(/2/)
  real*8,parameter                 :: filename_time=31d0
  character(len=19),parameter      :: expected_filename='part031.00000000.h5'
  type(particle_sim)               :: sim_to_write
  type(particle_sim)               :: sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  integer :: i, n_groups, n_particles
  allocate(writer, reader)
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.)
  call allocate_particles_here(sim_to_write%groups(1)%particles, n_particles_expect(1))
  sim_to_write%my_id = rank_loc; sim_to_write%n_mpi = n_tasks_loc;
  sim_to_write%time = filename_time; writer%mpi_comm_io = mpi_comm_loc;
  sim_to_write%groups(1)%Z = 2; sim_to_write%groups(1)%mass = 2.0;
  writer%file_access = file_access; writer%use_native_hdf5_mpio = .false.;
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc;
  call reader%run(sim_to_read)
  ! Test that we have the right stuff in sim_to_read now
  call groups_same(sim_to_write,sim_to_read,n_groups_expect,&
  n_particles_expect,"(gatherv writer / native reader)")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_gatherv_sim_one_group_boris

subroutine test_write_native_sim_two_groups_boris
  use mpi
  implicit none
  integer,parameter                :: n_groups_expect=2
  integer,dimension(2),parameter   :: n_particles_expect=(/2,2/)
  real*8,parameter                 :: filename_time=22d0;
  character(len=19),parameter      :: expected_filename='part022.00000000.h5'
  type(particle_sim)               :: sim_to_write, sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  integer :: i, j, n_groups, n_particles
  allocate(writer, reader)
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.)
  do i=1,n_groups_expect
    call allocate_particles_here(sim_to_write%groups(i)%particles,n_particles_expect(i))
  enddo
  sim_to_write%time = filename_time; sim_to_write%my_id = rank_loc;
  sim_to_write%n_mpi = n_tasks_loc; sim_to_write%groups(1)%Z = 324;
  sim_to_write%groups(1)%mass = 53.0; sim_to_write%groups(2)%Z = 765;
  sim_to_write%groups(2)%mass = 13.0; writer%mpi_comm_io = mpi_comm_loc; 
  writer%mpi_info_io = mpi_info_loc; writer%file_access = file_access; 
  writer%use_native_hdf5_mpio = .true.; call writer%run(sim_to_write);
  call MPI_Barrier(mpi_comm_loc,ifail_loc)
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc; reader%test = .true.; call reader%run(sim_to_read)
  ! Test that we have the right stuff in sim_to_read now
  call groups_same(sim_to_write,sim_to_read,n_groups_expect,&
  n_particles_expect,"(native writer/native reader)")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_native_sim_two_groups_boris

!> test the writing and reading of all particle types
subroutine test_write_native_sim_all_particles
  use mpi
  use mod_particle_common_test_tools, only: allocate_one_particle_list_type
  use mod_particle_common_test_tools, only: fill_groups,fill_particles
  use mod_particle_assert_equal,      only: assert_equal_particle_group
  implicit none
  integer,parameter               :: n_groups_expect=8
  integer,parameter               :: n_particles_expect=10
  real*8,parameter                :: filename_time=27d0;
  character(len=19),parameter     :: expected_filename='part027.00000000.h5'
  type(particle_sim)              :: sim_to_write, sim_to_read
  class(write_action),allocatable :: writer
  class(read_action),allocatable  :: reader
  logical :: file_exists
  allocate(writer, reader)
  call sim_to_write%initialize(my_id=rank_loc,n_mpi=n_tasks_loc,num_groups=n_groups_expect,do_jorek_init_in=.false.)
  sim_to_write%time = filename_time;
  call allocate_one_particle_list_type(n_groups_expect,n_particles_expect,sim_to_write%groups,ifail_loc)
  call fill_groups(n_groups_expect,sim_to_write%groups,rank_loc,ifail_loc)
  call fill_particles(n_groups_expect,sim_to_write%groups,rank_in=rank_loc)
  writer%mpi_comm_io = mpi_comm_loc; writer%mpi_info_io = mpi_info_loc;
  writer%file_access = file_access; writer%use_native_hdf5_mpio = .true.;
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc; reader%test = .true.;
  call reader%run(sim_to_read)
  ! Test that we have the right stuff in sim_to_read now
  call assert_equal_particle_group(n_groups_expect,sim_to_write%groups,sim_to_read%groups)
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_native_sim_all_particles

!> test the writing gatherv and reading of all particle types
subroutine test_write_gatherv_sim_all_particles
  use mpi
  use mod_particle_common_test_tools, only: allocate_one_particle_list_type
  use mod_particle_common_test_tools, only: fill_groups,fill_particles
  use mod_particle_assert_equal,      only: assert_equal_particle_group
  implicit none
  integer,parameter               :: n_groups_expect=8
  integer,parameter               :: n_particles_expect=10
  real*8,parameter                :: filename_time=37d0;
  character(len=19),parameter     :: expected_filename='part037.00000000.h5'
  type(particle_sim)              :: sim_to_write, sim_to_read
  class(write_action),allocatable :: writer
  class(read_action),allocatable  :: reader
  logical :: file_exists
  allocate(writer, reader)
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.)
  sim_to_write%time = filename_time; sim_to_write%my_id = rank_loc;
  sim_to_write%n_mpi = n_tasks_loc; 
  call allocate_one_particle_list_type(n_groups_expect,n_particles_expect,&
  sim_to_write%groups,ifail_loc); call fill_groups(n_groups_expect,&
  sim_to_write%groups,rank_loc,ifail_loc); call fill_particles(n_groups_expect,&
  sim_to_write%groups,rank_in=rank_loc); writer%mpi_comm_io = mpi_comm_loc;
  writer%file_access = file_access; writer%use_native_hdf5_mpio = .false.; 
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc; reader%test = .true.;
  call reader%run(sim_to_read)
  ! Test that we have the right stuff in sim_to_read now
  call assert_equal_particle_group(n_groups_expect,sim_to_write%groups,sim_to_read%groups)
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_gatherv_sim_all_particles

subroutine test_write_gatherv_sim_two_groups_boris
  use mpi
  implicit none
  integer,parameter                :: n_groups_expect=2
  integer,dimension(2),parameter   :: n_particles_expect=(/2,2/)
  real*8,parameter                 :: filename_time=22d0;
  character(len=19),parameter      :: expected_filename='part022.00000000.h5'
  type(particle_sim)               :: sim_to_write, sim_to_read
  class(write_action), allocatable :: writer
  class(read_action), allocatable  :: reader
  logical :: file_exists
  integer :: i, j, n_groups, n_particles
  allocate(writer, reader); 
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.)
  do i=1,n_groups_expect
    call allocate_particles_here(sim_to_write%groups(i)%particles,n_particles_expect(i))
  enddo
  sim_to_write%time = filename_time; sim_to_write%my_id = rank_loc;
  sim_to_write%n_mpi = n_tasks_loc; sim_to_write%groups(1)%Z = 324;
  sim_to_write%groups(1)%mass = 53.0; sim_to_write%groups(2)%Z = 765;
  sim_to_write%groups(2)%mass = 13.0; writer%file_access = file_access; 
  writer%mpi_comm_io = mpi_comm_loc; writer%use_native_hdf5_mpio = .false.; 
  call writer%run(sim_to_write); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  ! test if a file with the right name was created
  inquire(file=expected_filename, exist=file_exists)
  call assert_true(file_exists, 'file with the right name should be created')
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc;
  sim_to_read%time = sim_to_write%time; reader%time = sim_to_write%time; 
  reader%use_hdf5_access_properties=.false.; reader%mpi_comm_io = mpi_comm_loc; 
  reader%mpi_info_io = mpi_info_loc; 
  call reader%run(sim_to_read)
  ! Test that we have the right stuff in sim_to_read now
  call groups_same(sim_to_write,sim_to_read,n_groups_expect,&
  n_particles_expect,"(gatherv writer/native reader)")
  ! Delete the file
  call remove_file(rank_loc,expected_filename,mpi_comm_loc,ifail_loc)
  deallocate(writer, reader)
  if(allocated(sim_to_write%groups)) deallocate(sim_to_write%groups)
  if(allocated(sim_to_read%groups))  deallocate(sim_to_read%groups)
end subroutine test_write_gatherv_sim_two_groups_boris

!> test the writing/reading operation for time loop
!> the writing gatherv is used but nothing prevents to
!> use the parallel native write/read operations
subroutine test_time_loop_gather_sim_two_groups_boris
  use mpi
  use mod_io_actions, only: get_filename
  use mod_event
  implicit none
  real*8,parameter                  :: stop_time=1.d0
  real*8,parameter                  :: io_step=1d-1
  type(event), dimension(2), target :: write_events,read_events
  integer,parameter                 :: n_groups_expect=2
  integer,dimension(2),parameter    :: n_particles_expect=(/2,2/)
  type(particle_sim)                :: sim_to_write, sim_to_read
  class(write_action), allocatable  :: writer
  class(read_action), allocatable   :: reader
  integer :: ii, n_groups, n_particles
  !> initialise write particle simulation
  call sim_to_write%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.)
  do ii=1,n_groups_expect
    call allocate_particles_here(sim_to_write%groups(ii)%particles,n_particles_expect(ii))
  enddo
  sim_to_write%my_id = rank_loc; sim_to_write%n_mpi = n_tasks_loc; 
  sim_to_write%groups(1)%Z = 324; sim_to_write%groups(1)%mass = 53.0; 
  sim_to_write%groups(2)%Z = 765; sim_to_write%groups(2)%mass = 13.0; 
  sim_to_write%time = 0d0; write_events = [(event(stop_action(), start=stop_time)),\
  event(write_action(), step=io_step)]
  !> write files
  do while (.not. sim_to_write%stop_now)
    call with(sim_to_write,write_events,at=sim_to_write%time) !< write file
    sim_to_write%time = sim_to_write%time + io_step !< update time
  enddo
  !> initialise read particle simulation
  call sim_to_read%initialize(num_groups=n_groups_expect,do_jorek_init_in=.false.);
  sim_to_read%my_id = rank_loc; sim_to_read%n_mpi = n_tasks_loc; 
  sim_to_read%time = 0d0; read_events = [(event(stop_action(), start=stop_time)),\
  event(read_action(), step=io_step)]
  do while (.not. sim_to_read%stop_now)
   ! if(allocated(sim_to_read%groups)) deallocate(sim_to_read%groups)
    call with(sim_to_read,read_events,at=sim_to_read%time)
    !> compare results
    call groups_same(sim_to_write,sim_to_read,n_groups_expect,&
    n_particles_expect,"(time loop gatherv writer/native reader)")
    !> remove file
    select type (act=>read_events(2)%stored_action)
    type is (read_action)
      call remove_file(rank_loc,act%get_filename(\
      sim_to_read%time),mpi_comm_loc,ifail_loc)
    end select
    sim_to_read%time = sim_to_read%time + io_step !< update time
  enddo
end subroutine test_time_loop_gather_sim_two_groups_boris

!> Tools ------------------------------------------
!> Helper function for removing files
subroutine remove_file(rank,filename,mpi_comm_in,ifail)
  use mpi
  implicit none
  integer,intent(inout)       :: ifail
  integer,intent(in)          :: rank,mpi_comm_in
  character(len=*),intent(in) :: filename
  integer                     :: u, stat
  if(rank.eq.master_rank) then
    open(newunit=u, iostat=stat, file=trim(filename), status='old')
    if (stat .eq. 0) close(u, status='delete')
  endif
  call MPI_Barrier(mpi_comm_in,ifail)
end subroutine remove_file

!> A specific version of allocate particles for this test
subroutine allocate_particles_here(particles, n)
  class(particle_base), allocatable, dimension(:), intent(out) :: particles
  integer, intent(in) :: n
  integer :: i
  ! Prepare some particles
  allocate(particle_kinetic_leapfrog::particles(n))
  select type (p => particles)
  type is (particle_kinetic_leapfrog)
    do i=1,size(p,1)
      p(i)%x = real((/i,i+1,i+2/),8)
      p(i)%weight = real(i+1,4)/4
      call random_number(p(i)%st)
      p(i)%i_elm = 100*i
      p(i)%q = int(3*i,1)
      p(i)%v = real((/2*i,2*i+1,2*i+2/),8)
    end do
  end select
end subroutine allocate_particles_here

!> test if groups of a simulations are equals
subroutine groups_same(sim,sim_target,n_groups_expect,n_particles_expect,message_in)
  use mod_particle_sim, only: particle_sim
  implicit none
  !> inputs
  type(particle_sim),intent(in)                 :: sim,sim_target
  integer,intent(in)                            :: n_groups_expect
  integer,dimension(n_groups_expect),intent(in) :: n_particles_expect
  character(len=*),intent(in),optional          :: message_in
  !> variables
  integer :: ii,jj,n_groups,n_particles
  character(len=:),allocatable                  :: message
  if(present(message_in)) then
    allocate(character(len=len(message_in))::message); message=message_in;
  else
    allocate(character(len=1)::message); message = "";
  endif
  ! Test that we have the right stuff in sim_to_read now
  n_groups = size(sim%groups,1)
  call assert_equals(n_groups_expect, n_groups, &
  'number of expected groups mismatch! '//trim(message))
  if (n_groups .eq. n_groups_expect) then
    do jj=1,n_groups
      n_particles = size(sim%groups(jj)%particles,1)
      call assert_equals(n_particles_expect(jj), n_particles, &
      'number of expected particles mismatch! '//trim(message))
      if (n_particles .eq. n_particles_expect(jj)) then
        do ii=1,n_particles
          call assert_true(particles_same(sim_target%groups(jj)%particles(ii), sim%groups(jj)%particles(ii)), &
              'particle must be as written '//trim(message))
        end do
      end if
      call assert_equals(sim_target%groups(jj)%Z, sim%groups(jj)%Z,&
      'Z equal mismatch! '//trim(message))
      call assert_equals(sim_target%groups(jj)%mass, sim%groups(jj)%mass, &
      'mass equal mismatch! '//trim(message))
    enddo
  end if
  if(allocated(message)) deallocate(message)
end subroutine groups_same

!> Helper function for comparing particles
function particles_same(p1, p2) result(same)
  class(particle_base), intent(in) :: p1, p2
  real*8, parameter :: tolerance = 1d-30
  logical :: same
  same = .true.

  ! Base attributes testing
  if (norm2(p1%x-p2%x)         .gt. tolerance) then
    write(*,*) 'x different', p1%x, p2%x
    same = .false.
  end if
  if (abs(p1%weight-p2%weight) .gt. tolerance) then
    write(*,*) 'weight different', p1%weight, p2%weight
    same = .false.
  end if
  if (norm2(p1%st-p2%st)       .gt. tolerance) then
    write(*,*) 'st different', p1%st, p2%st
    same = .false.
  end if
  if (p1%i_elm .ne. p2%i_elm) then
    write(*,*) 'i_elm different', p1%i_elm, p2%i_elm
    same = .false.
  end if

  select type(p1 => p1)
    type is (particle_kinetic_leapfrog)
      select type (p2 => p2)
        type is (particle_kinetic_leapfrog)
          if (norm2(p1%v-p2%v)         .gt. tolerance) then
            write(*,*) 'v different', p1%v, p2%v
            same = .false.
          end if
          if (p1%q .ne. p2%q) then
            write(*,*) 'q different', p1%q, p2%q
            same = .false.
          end if
        class default
          same = .false.
      end select
  end select
end function particles_same
!> ------------------------------------------------
end module mod_sim_hdf5_io_spec_mpi_test
