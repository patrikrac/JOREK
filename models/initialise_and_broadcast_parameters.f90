!> Initialize parameters and broadcast them to all MPI procs.
subroutine initialise_and_broadcast_parameters(my_id, filename, init_particles)
  
  use constants, only: mu_zero
  use mod_parameters,  only: n_tor, n_period
  use mod_plasma_functions, only: initialise_reference_parameters
  use mod_particle_group_id
  use mod_coupling_settings
  use phys_module
  
  implicit none
  
  ! --- Routine parameters
  integer,                      intent(in) :: my_id
  character(len=*),             intent(in) :: filename
  logical,                      intent(in) :: init_particles
  
  call initialise_parameters(my_id, filename)

  ! Determine coupling parameters
  if (my_id .eq. 0) then
    if (init_particles) then

      ! --- Initialize part_groups_in_use and determine n_part_groups
      if (part_groups_in_use(1) == 'non') then !< part_groups_in_use not manually defined
        !> generate the particle groups in use based on the defined groups in part_group_configs
        call generate_part_groups_in_use()
      endif

      n_part_groups = count(part_groups_in_use /= 'non')
      !> find the matching part_group_config for each group specified in part_groups_in_use
      call match_part_groups_and_configs()
        
      ! --- Scan over n_part_groups and determine the coupling scheme parameters as well as their compatibility
      call check_compatibility_and_determine_coupling_schemes()
  
      ! --- Determine the coupling variables used, their index, and n_aux_var
      call determine_coupling_variables() 

      n_fluid_groups = count(fluid_configs%Z /= -999)
    endif
  endif

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
  if ((.not.use_mumps_eq).and.(.not.use_pastix_eq).and.(.not.use_strumpack_eq).and.(.not.use_petsc_eq)) then
#ifdef USE_COMPLEX_PRECOND
    use_mumps_eq = .true.
    use_pastix_eq = .false.
    use_strumpack_eq = .false.
#else
    use_mumps_eq = use_mumps
    use_pastix_eq = use_pastix
    use_strumpack_eq = use_strumpack
#endif
  else if (.not. use_petsc_eq) then
    use_mumps_eq = use_mumps
    use_pastix_eq = use_pastix
    use_strumpack_eq = use_strumpack
  endif
  ! -----------------------------------
  ! -- Set projection solver if not defined by user --
  if ((.not.use_mumps_prj).and.(.not.use_pastix_prj).and.(.not.use_strumpack_prj)) then
    if (my_id .eq. 0) write(*,*) 'WARNING: No projection solver defined. Using fluid solver by default.'
    use_mumps_prj     = use_mumps
    use_pastix_prj    = use_pastix
    use_strumpack_prj = use_strumpack
  endif
  ! -----------------------------------

  prev_FB_fact = 1.d0 ! needed to make sure current_FB_fact is applied correctly in import_restart
  
end subroutine initialise_and_broadcast_parameters