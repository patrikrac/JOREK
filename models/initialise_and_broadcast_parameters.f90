!> Initialize parameters and broadcast them to all MPI procs.
subroutine initialise_and_broadcast_parameters(my_id, filename)
  
  use constants, only: mu_zero
  use mod_parameters,  only: n_tor, n_period
  use mod_plasma_functions, only: initialise_reference_parameters
  use phys_module
  
  implicit none
  
  ! --- Routine parameters
  integer,                      intent(in) :: my_id
  character(len=*),             intent(in) :: filename
  
  call initialise_parameters(my_id, filename)
  
  ! --- Broadcast input parameters from MPI thread 0 to the others.
  call broadcast_phys(my_id)
  
  ! --- Broadcast numerical input profiles from MPI thread 0 to the others.
  call broadcast_num_profiles(my_id)
  
  ! --- Initialize the time-stepping parameters.
  call update_time_evol_params()
  
  ! --- Initialize derived reference parameters
  call initialise_reference_parameters()

  ! --- Assign minimum values for parallel conduction if not given
  if (T_min_ZKpar  < -1.d10) T_min_ZKpar  = T_min   
  if (Ti_min_ZKpar < -1.d10) Ti_min_ZKpar = T_min   
  if (Te_min_ZKpar < -1.d10) Te_min_ZKpar = T_min   
  
  ! --- Deprecated input parameters ---
  if ( use_murge ) then
    write(*,*) 'ERROR: use_murge=.true. is not supported any more. Remove this parameter from the namelist input file.'
    stop
  else if ( use_murge_element ) then
    write(*,*) 'ERROR: use_murge_element=.true. is not supported any more. Remove this parameter from the namelist input file.'
    stop
  end if
  ! -----------------------------------
  ! -- Set equilibrium solver if not defined by user --
  if ((.not.use_mumps_eq).and.(.not.use_pastix_eq).and.(.not.use_strumpack_eq)) then
#ifdef USE_COMPLEX_PRECOND
    use_mumps_eq = .true.
    use_pastix_eq = .false.
    use_strumpack_eq = .false.
#else
    use_mumps_eq = use_mumps
    use_pastix_eq = use_pastix
    use_strumpack_eq = use_strumpack
#endif
  endif
  ! -----------------------------------
  ! -- Set projection solver if not defined by user --
  if ((.not.use_mumps_prj).and.(.not.use_pastix_prj).and.(.not.use_strumpack_prj)) then
    write(*,*) 'WARNING: No projection solver defined. Using fluid solver by default.'
    use_mumps_prj     = use_mumps
    use_pastix_prj    = use_pastix
    use_strumpack_prj = use_strumpack
  endif
  ! -----------------------------------

  prev_FB_fact = 1.d0 ! needed to make sure current_FB_fact is applied correctly in import_restart
  
end subroutine initialise_and_broadcast_parameters
