!< This program reads jorek_restart files for the stellarator model 180
!! and calculates the induced magnetic field at arbitrary xyz points. It involves the 
!! computation of expensive volume integrals in the plasma, so run it
!! in paralllel with several MPI processes for large cases. 
!! As input you must provide the points in cartesian coordinates in the file
!! xyz.dat in the format
!! 
!!   n (number_of_points)
!!   x_1       y_1       z_1  
!!   x_2       y_2       z_2  
!!    :         :         :   
!!    :         :         :   
!!   x_n       y_n       z_n  
!!
!! The fields are exported in the file "fields_xyz.dat".
!! Note that to calculate the require plasma fields n_plane typically should be
!! more than 60 (even for 2D)
!!
!! See more documentation here
!!   https://www.jorek.eu/wiki/doku.php?id=jorek2_fields_xyz
program jorek2_fields_xyz_stel

  use mod_vacuum_fields
  use mod_plasma_response,  only: plasma_fields_at_xyz_gvec
  use data_structure
  use phys_module
  use mod_parameters
  use mod_chi
  use nodes_elements
  use mod_boundary,            only: boundary_from_grid
  use vacuum
  use vacuum_response,     only: get_vacuum_response, broadcast_starwall_response, init_wall_currents
  use vacuum_equilibrium,  only: import_external_fields
  use mod_import_restart
  use mod_element_rtree, only: populate_element_rtree
  use basis_at_gaussian, only: initialise_basis
  use tr_module
#ifdef USE_HDF5
  use hdf5
  use hdf5_io_module
  use matio_module, only: timestamp
#endif
  use mpi_mod

  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
                                            stdout=>output_unit, &
                                            stderr=>error_unit
  implicit none
 
  integer   :: my_id, my_id_n, my_id_master, ierr, ierr2
  integer   :: i_rank(n_tor), n_cpu, n_cpu_n, n_cpu_master, m_cpu, n_masters, n_cpu_trans, my_id_trans
  integer   :: MPI_COMM_N, MPI_GROUP_MASTER, MPI_GROUP_WORLD, MPI_COMM_MASTER, MPI_COMM_TRANS
  integer   :: required,provided,StatInfo
  integer   :: istep, delta_step, istart, iend, np, i
  integer*4 :: rank, comm_size 

  real*8               :: bx,      by,      bz
  real*8, allocatable  :: bx_p(:), by_p(:), bz_p(:)
  real*8, allocatable  ::    x(:),    y(:),    z(:)

  character*17      :: file_in
  character(len=MPI_MAX_PROCESSOR_NAME) :: name
  integer :: resultlength
 
 
  !***********************************************************************
  !*                  initialisation (copied from jorek2_main)           *
  !***********************************************************************
#ifdef FUNNELED
  required = MPI_THREAD_FUNNELED
#else
  required = MPI_THREAD_MULTIPLE
#endif

  if (n_period .ne. 1) then
    write(*,*) 'This Biot-Savart integration requires n_period == 1 to integrate over the full torus!'
    stop
  endif

  call MPI_Init_thread(required, provided, StatInfo)

  call init_threads()  ! on some systems init_threads needs to come after mpi_init_thread
  
  ! --- Determine number of MPI procs
  call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_size, ierr)
  n_cpu = comm_size
  
  ! --- Determine ID of each MPI proc
  call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
  my_id = rank
  
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  CALL MPI_GET_PROCESSOR_NAME (name,resultlength,ierr)
  write(*,'(A,I5,2A)') '  #MPI id, ProcessorName ', rank, ': ', name
  call MPI_Barrier(MPI_COMM_WORLD,ierr)

  ! -------------------------------------------------------------
  ! --------------- Initialise parameters -----------------------
  ! -------------------------------------------------------------
  ! --- Initialize mode and mode_type arrays
  call det_modes()
 
  ! --- Preset input parameters to reasonable defaults, then read the input file.
  call initialise_and_broadcast_parameters(my_id, "__NO_FILENAME__")
  
  ! --- Initialize the vacuum part.
  call vacuum_init(my_id, freeboundary_equil, freeboundary, resistive_wall)
  
  call MPI_Barrier(MPI_COMM_WORLD,ierr)

  ! --- Define the basis functions at the Gaussian points
  call initialise_basis()

  ! --------------------------------------------------------------
  ! ------------------  Read input points  -----------------------
  ! --------------------------------------------------------------
  open(26,file='xyz.dat',action='read',iostat=ierr)
  if (ierr/=0) then
    write(*,*) 'Could not read xyz.dat'
    stop
  endif
  read(26,*) np 
  allocate(x(np), y(np), z(np))
  do i=1, np
    read(26,*) x(i), y(i), z(i)
  enddo
  close(26)
  
  ! ---------------------------------------------------------------
  ! ------------  Open output file and set up header  -------------
  ! ---------------------------------------------------------------
  if (my_id==0) then
    open(87,file='fields_xyz.dat',action='write') 
    write(87,'(a)', advance='no')  '#Step  '
    write(87,'(a)', advance='no')  'time(norm)    '
    write(87,'(a)', advance='no')  'time(ms)      '
    write(87,'(a)', advance='no')  'x(m)          '
    write(87,'(a)', advance='no')  'y(m)          '
    write(87,'(a)', advance='no')  'z(m)          '
    write(87,'(a)', advance='no')  'Bx_p(T)       '
    write(87,'(a)', advance='no')  'By_p(T)       '
    write(87,'(a)', advance='yes') 'Bz_p(T)       '
  endif
  
  ! Allocate array for computed plasma magnetic field
  allocate(bx_p(np), by_p(np), bz_p(np))

  ! Read JOREK equilibrium restart file
  istep = 0
  write(file_in,'(A5,i6.6)') 'jorek', istep
  if ( my_id == 0 ) then
    call import_restart(node_list, element_list, file_in, rst_format, ierr)
  endif

  call init_chi_basis

  ! ------------------------------------------------------
  ! ----------- Broadcast across MPI tasks ---------------
  ! ------------------------------------------------------
  call MPI_BCAST(ierr,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr2)

  call broadcast_phys(my_id)  
  call broadcast_elements(my_id, element_list)                ! elements
  call broadcast_nodes(my_id, node_list)                      ! nodes
  
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  
  if ( my_id == 0 ) call boundary_from_grid(node_list, element_list, bnd_node_list, bnd_elm_list, output_bnd_elements)
  call broadcast_boundary(my_id, bnd_elm_list, bnd_node_list)
  
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  
  ! -------------------------------------------------------
  ! -----Compute magnetic field via Biot-Savart law -------
  ! -------------------------------------------------------
  call plasma_fields_at_xyz_gvec(my_id, node_list,element_list, x, y, z, bx_p, by_p, bz_p)
  
  ! Write out computed plasma field to file
  if (my_id==0) then
    do i=1, np
      bx = bx_p(i)
      by = by_p(i)
      bz = bz_p(i)
      write(87,'(I5.5,8ES16.8)') istep, t_start, t_start*sqrt_mu0_rho0*1.d3, x(i),y(i),z(i),bx_p(i), by_p(i), bz_p(i)      
    enddo
  endif
  
  write(87,*) ' '
  write(87,*) ' '

  close(87)

  call MPI_FINALIZE(IERR)                                ! clean up MPI

end program jorek2_fields_xyz_stel 

