! Program use for testing the particle IO
program test_particle_io
  use mpi
  use fruit
  use fruit_mpi
  use hdf5,                           only: H5F_ACC_TRUNC_F
  use mod_gnu_rng,                    only: set_seed_sys_time
  use mod_gnu_rng,                    only: gnu_rng_interval
  use mod_mpi_tools,                  only: get_mpi_wtime
  use mod_particle_types,             only: particle_fieldline_id,particle_fieldline
  use mod_particle_types,             only: particle_gc_id,particle_gc_vpar_id
  use mod_particle_types,             only: particle_kinetic_id,particle_kinetic_leapfrog_id
  use mod_particle_types,             only: particle_kinetic_relativistic_id
  use mod_particle_types,             only: particle_gc_relativistic_id
  use mod_particle_sim,               only: particle_sim
  use mod_event,                      only: event,with
  use mod_io_actions,                 only: read_action,write_action
  use mod_particle_common_test_tools, only: allocate_one_particle_list_type
  use mod_particle_common_test_tools, only: fill_particles
  use mod_particle_assert_equal,      only: assert_equal_particle
  implicit none
  !> Variable declaration ----------------------------------------
  integer,parameter                :: master_task=0
  type(particle_sim)               :: sim_write,sim_read
  type(event),dimension(1),target  :: event_write,event_read
  type(read_action)                :: read_particle
  type(write_action)               :: write_particle
  integer                          :: ii,jj,ifail
  integer                          :: n_groups,fill_particle_type
  integer                          :: mpi_comm_io,mpi_info_io
  integer                          :: my_id,n_tasks
  integer                          :: file_access_write
  integer,dimension(2)             :: Z_interval,seed_interval
  integer,dimension(:),allocatable :: p_types,n_particles,Z_values
  real*8                           :: start_time,end_time
  real*8                           :: write_time,read_time
  real*8,dimension(2)              :: mass_interval
  real*8,dimension(:),allocatable  :: mass_values,write_times,read_times
  logical                          :: do_write,do_read,do_test
  logical                          :: use_hdf5_access_properties
  logical                          :: use_native_hdf5_mpio
  logical                          :: mpio_collective
  logical                          :: remove_file,file_exist
  character(len=24)                :: filename
  !> Define test parameters --------------------------------------
  filename           = "particle_restart_test.h5"
  do_write           = .true.
  do_read            = .true.
  do_test            = .true.
  ifail              = 0 
  n_groups           = 7  !< number of groups = number of particle types
  allocate(n_particles(n_groups)); allocate(p_types(n_groups))
  n_particles        = [1000,2000,1000,3000,4000,7000,1000] !< number of particles
  seed_interval      = [12,31244553]
  mass_interval      = [1d-5,1d0] 
  Z_interval         = [-1,100] 
  p_types = [particle_fieldline_id,particle_gc_id,particle_gc_vpar_id,&
            particle_kinetic_id,particle_kinetic_leapfrog_id,&
            particle_kinetic_relativistic_id,particle_gc_relativistic_id]
  mpi_comm_io                = MPI_COMM_WORLD
  mpi_info_io                = MPI_INFO_NULL
  use_hdf5_access_properties = .true.
  use_native_hdf5_mpio       = .true.
  file_access_write          = H5F_ACC_TRUNC_F
  mpio_collective            = .true.
  remove_file                = .true.
  !> Initialise --------------------------------------------------
  call sim_write%initialize(num_groups=n_groups,skip_jorek2help=.true.,do_jorek_init_in=.false.)
  if(do_write) then
    write_particle = write_action(filename,file_access_in=file_access_write,&
    use_native_hdf5_mpio_in=use_native_hdf5_mpio,&
    use_hdf5_access_properties_in=use_hdf5_access_properties,&
    mpi_comm_in=mpi_comm_io,mpi_info_in=mpi_info_io,mpio_collective_in=mpio_collective)
    event_write = [event(write_particle)]
    call allocate_one_particle_list_type(n_groups,n_particles,p_types,sim_write%groups,ifail)
    call fill_particles(n_groups,sim_write%groups,fill_particle_in=fill_particle_type,&
    rank_in=sim_write%my_id)
    call set_seed_sys_time(seed_interval)
    if(allocated(mass_values)) deallocate(mass_values); allocate(mass_values(n_groups));
    if(allocated(Z_values))    deallocate(Z_values);    allocate(Z_values(n_groups));
    if(sim_write%my_id.eq.master_task) then
      do ii=1,n_groups
        call gnu_rng_interval(mass_interval,mass_values(ii))
        call gnu_rng_interval(Z_interval,Z_values(ii))
      enddo
    endif
    call MPI_Bcast(mass_values,n_groups,MPI_REAL8,master_task,mpi_comm_io,ifail)
    call MPI_Bcast(Z_values,n_groups,MPI_INTEGER,master_task,mpi_comm_io,ifail)
    do ii=1,n_groups
      sim_write%groups(ii)%mass = mass_values(ii); sim_write%groups(ii)%Z = Z_values(ii);
    enddo
    if(allocated(mass_values)) deallocate(mass_values); if(allocated(Z_values)) deallocate(Z_values);
  endif
  !> Read
  if(do_read) then
    read_particle = read_action(filename=filename,mpi_comm_in=mpi_comm_io,&
    mpi_info_in=mpi_info_io,use_hdf5_access_properties_in=use_hdf5_access_properties)
    event_read = [event(read_particle)];
    call sim_read%initialize(num_groups=n_groups,skip_jorek2help=.true.,my_id=sim_write%my_id,&
    n_mpi=sim_write%n_mpi,do_jorek_init_in=.false.)
  endif
  !> Read - write operations -------------------------------------
  if(do_write) then
    start_time = 0d0; end_time = 0d0; write_time = 0d0;
    start_time = get_mpi_wtime()
    call with(sim_write,event_write)
    end_time   = get_mpi_wtime()
    write_time = end_time - start_time
  endif
  if(do_read) then
    start_time = 0d0; end_time = 0d0; read_time = 0d0;
    start_time = get_mpi_wtime()
    call with(sim_read,event_read)
    end_time   = get_mpi_wtime()
    read_time = end_time - start_time
  endif
  !> Check that files are equals ---------------------------------
  if(do_write.and.do_read.and.do_test) then
    do ii=2,n_groups
      select type (p=>sim_write%groups(ii)%particles)
      type is (particle_fieldline)
        select type(p2=>sim_read%groups(ii)%particles)
          type is (particle_fieldline)
          do jj=1,size(p)
            p(jj)%B_hat_prev = 0d0;
          enddo
        endselect
      end select
      write(*,*) "Size particle old: ",size(sim_write%groups(ii)%particles),&
      " size new: ",size(sim_read%groups(ii)%particles)
      call init_fruit
      call fruit_init_mpi_xml(sim_write%my_id)
      call fruit_hide_dots
      call assert_equals(sim_write%groups(ii)%Z,sim_read%groups(ii)%Z,"List Z mismatch!")
      call assert_equals(sim_write%groups(ii)%mass,&
      sim_read%groups(ii)%mass,"List mass mismatch!")
      call assert_equal_particle(size(sim_write%groups(ii)%particles),&
      sim_write%groups(ii)%particles,sim_read%groups(ii)%particles)
      call fruit_summary_mpi(sim_write%n_mpi,sim_write%my_id)
      call fruit_summary_mpi_xml(sim_write%n_mpi,sim_write%my_id)
      call fruit_finalize_mpi(sim_write%n_mpi,sim_write%my_id)
    enddo
  endif
  !> Log data ---------------------------------------------------- 
  call MPI_comm_rank(mpi_comm_io,my_id,ifail)
  call MPI_comm_size(mpi_comm_io,n_tasks,ifail)
  if(allocated(write_times)) deallocate(write_times); allocate(write_times(n_tasks))
  if(allocated(read_times))  deallocate(read_times);  allocate(read_times(n_tasks));
  call MPI_Gather(write_time,1,MPI_REAL8,write_times,1,MPI_REAL8,master_task,mpi_comm_io,ifail)
  call MPI_Gather(read_time,1,MPI_REAL8,read_times,1,MPI_REAL8,master_task,mpi_comm_io,ifail)
  if(my_id.eq.master_task) then
    write(*,*) "Time for writing data [s]: ",maxval(write_times)
    write(*,*) "Time for reading data [s]: ",maxval(read_times)
  endif
  !> Clean-up ---------------------------------------------------- 
  if(allocated(p_types))     deallocate(p_types)
  if(allocated(n_particles)) deallocate(n_particles)
  if(allocated(write_times)) deallocate(write_times)
  if(allocated(read_times))  deallocate(read_times)
  if((sim_write%my_id.eq.master_task).and.remove_file) then
    inquire(file=filename,exist=file_exist)
    if(file_exist) call system("rm -rf "//trim(filename))
  endif
  call sim_write%finalize
end program test_particle_io

