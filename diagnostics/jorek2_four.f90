!> Allows to Fourier-analyse the physical variables of a JOREK restart file in magnetic coordinates.
program JOREK2_FOUR

  use mod_parameters,     only: n_tor, n_var, n_period, variable_names
  use nodes_elements, only: element_list, node_list
  use fourier,        only: t_theta_mapping, determine_theta_mag, transform_qttys
  use phys_module,    only: rst_format, xpoint, xcase
  use mod_import_restart
  use equil_info
  use data_structure
  use mod_boundary
  use mod_log_params
  use basis_at_gaussian, only: initialise_basis

  implicit none
  
  integer                      :: i, j, k, l, ierr, ivar, n_mpi, err, vars_per_cpu, nTht
  type(t_theta_mapping)        :: mapping        ! Mapping between theta_mag and theta_geo
  type (type_bnd_element_list) :: bnd_elm_list
  type (type_bnd_node_list)    :: bnd_node_list
  complex, allocatable         :: vfour(:,:,:,:) ! Transformed quantities (m,n,irad,ivar)
  character(len=128)           :: filename
  character(len=64), parameter :: THIS_ROUTINE_NAME = 'JOREK2_FOUR'

  ! ---Field line tracing parameters
  integer                  :: nstpts, nmaxsteps, nsmallsteps, nmaxsteps_corr
  real*8                   :: deltaphi, rad_range(2), deltaphi_corr
  namelist / four_params / nstpts, nmaxsteps, deltaphi, nsmallsteps ,rad_range, nTht
  
  ! --- Initialize mode and mode_type arrays
  call det_modes()
  
  ! --- Initialization
  write(*,*)
  write(*,*) '>>> Initialization <<<'
  call initialise_parameters(0, "__NO_FILENAME__")                 ! default values and namelist input
  call log_parameters(0)
  call initialise_basis                         ! define the basis functions at the Gaussian points
  call import_restart(node_list, element_list, 'jorek_restart', rst_format, ierr, .true.)   ! read restart file

  call boundary_from_grid(node_list, element_list, bnd_node_list, bnd_elm_list, .false.)

  !   --- Preset field line tracing parameters.
  nstpts      = 30
  nTht        = 32
  nsmallsteps = 3 !###
  nmaxsteps   = 2500
  deltaphi    = 0.3
  nsmallsteps = 3
  rad_range   = (/0.001, 0.999/)
  !   --- If four_params.nml file exists, read field line tracing parameters from that file.
  open(42, file='./four_params.nml', action='READ', status='OLD', iostat=err)
  if ( err == 0 ) then
    write(*,*) 'Reading parameters from four_params.nml'
    read(42, four_params) ! read namelist with Fourier parameters
    close(42)
  else
    write(*,*) 'WARNING: Could not find file four_params.nml -- using default parameters.'
  end if
  write(42,'(a)') '#      Psi_N     absolute_value     real_part    imaginary_part      phase'
  deltaphi_corr  = min(0.3d0, 20.d0 / real(nTht), deltaphi) !###
  nmaxsteps_corr = max(nmaxsteps, int(400 / deltaphi_corr)) !###
  
  ! --- Log field line tracing parameters.
  write(*,*)
  write(*,*) '>>> Fourier transformation parameters <<<'
  111 format(1x,a,2i6)
  112 format(1x,a,2es12.4)
  113 format(1x,a,2l6)
  write(*,112) 'rad_range (PsiNmin:PsiNmax) =', rad_range 
  write(*,111) 'nstpts (NPsiN) =', nstpts
  write(*,111) 'nTht          =', nTht
  write(*,111) 'nsmallsteps   =', nsmallsteps
  write(*,112) 'deltaphi      =', deltaphi
  write(*,112) 'deltaphi corr =', deltaphi_corr
  write(*,111) 'nmaxsteps     =', nmaxsteps
  write(*,111) 'nmaxsteps corr=', nmaxsteps_corr

  
  if ( (mod(nTht, 4) .ne. 0) .or. (nTht .lt. 32) ) then
    write(*,*) 'ERROR in '//trim(THIS_ROUTINE_NAME)//': nTht must be a factor of 4, minimum acceptable value=32, recommanded value(jorek2_postproc four2d)=6*4*n_tor'
    ierr=300
    stop
  end if

  ! --- Determine magnetic coordinates by field line tracing.
  write(*,*)
  write(*,*) '>>> Determining the poloidal straight field line angle theta_star <<<'
  call determine_theta_mag(mapping, node_list, element_list, ES, rad_range, nstpts, nTht, ierr, nmaxsteps_corr, deltaphi_corr, nsmallsteps)

  ! --- Transform the quantities
  write(*,*)
  write(*,*) '>>> Transforming the variables <<<'
  call transform_qttys(mapping, vfour, nTht)
  
  ! --- Output Fourier modes of the physical quantities.
  write(*,*)
  write(*,*) '>>> Writing the data to ascii files <<<'
  do ivar = 1, n_var
    do j = 1, (n_tor+1)/2 ! tor
    
      write(filename,'(a,a,i3.3)') trim(variable_names(ivar)), '_modes_n', (j-1)*n_period
      open(42, file=trim(filename), status='REPLACE', action='WRITE')
      l = 0

      do i = 1, nTht/2+1
        write(42,'("# ",I3,":   m=",I3,", n=",I3)') l, i-1, (j-1)*n_period
        l = l + 1
        do k = 1, mapping%nstpts
          write(42,'(5es16.7)') mapping%psin(k), ABS(vfour(i,j,k,ivar)), REAL(vfour(i,j,k,ivar)), AIMAG(vfour(i,j,k,ivar)), ATAN2(AIMAG(vfour(i,j,k,ivar)), REAL(vfour(i,j,k,ivar)))
        end do
        write(42,*)
        write(42,*)
        
      end do
      close(42)
      
    end do
  end do
  
  write(*,*) 'done.'
  
end program JOREK2_FOUR
