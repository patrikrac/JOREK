!> Program to compare jorek imported 3D equilibrium points to GVEC equilibrium points 
!! from a higher resolution import. This test program is used for model 180 only.
program test_gvec2jorek_import

use data_structure
use phys_module
use basis_at_gaussian
use elements_nodes_neighbours
use mod_neighbours
use mod_import_restart
use mod_log_params
use mod_interp
use mod_import_gvec

implicit none

! Local variables  
integer                               :: ierr

! --- Initialise memory tracing
call tr_meminit(0, 1)

! Initialise mode and mode type arrays
call det_modes()
call initialise_basis()

ierr = 0
write(*,*) "Reading GVEC import"
call read_gvec_import(node_list, element_list, 'gvec2jorek.dat', .true., ierr)
write(*,*) "Start GVEC comparison"
call compare_gvec_data_points(node_list, element_list, 'gvec2jorek_testdata.dat', ierr)


end program test_gvec2jorek_import

! Reads a high resolution grid of data points from GVEC to be tested against JOREK representation
subroutine compare_gvec_data_points(node_list, element_list, file_name, ierr)
  use tr_module
  use data_structure
  use phys_module
  use mod_interp

  type(type_node_list),    intent(in)  :: node_list
  type(type_element_list), intent(in)  :: element_list
  
  real*8, allocatable    :: s(:,:,:), theta(:,:,:), phi(:,:,:)            ! Arrays for GVEC test data
  real*8, allocatable    :: R(:,:,:,:)
  real*8, allocatable    :: Z(:,:,:,:)
  real*8, allocatable    :: P(:,:,:,:)
  real*8, allocatable    :: B_R(:,:,:,:), B_Z(:,:,:,:), B_phi(:,:,:,:)
  real*8, allocatable    :: J_R(:,:,:,:), J_Z(:,:,:,:), J_phi(:,:,:,:)
  character(*), intent(in)              :: file_name                          ! Test file name
  integer, intent(inout)                :: ierr                               ! IO error status
  integer          :: n_rad, n_theta, n_phi                    ! Number of radial, poloidal and toroidal points in GVEC test data
  integer          :: n_rad_jorek, n_theta_jorek, n_phi_jorek  ! Number of radial, poloidal and toroidal points in JOREK
  integer          :: coord_type, num_periods, asym            ! GVEC coordinate type, number of field periods, equilibrium symmetry
  integer          :: m_max, n_max, n_modes                    ! Maximum poloidal mode number, maximum toroidal mode number, number of modes
  integer          :: sin_range(2), cos_range(2)               ! Indices in fourier representation for sinusoidal and cosinusoidal modes

  integer          :: idx, i_rad, i_theta, i_phi, i_rad_loc, i_theta_loc, i_elm                ! Generic, radial, poloidal, toroidal, and element indices
  real*8           :: s_loc, theta_loc, phi_loc, dtheta, ds                                    ! Local s, t, phi coordinate and element ds, dt
  real*8           :: s_factor, t_factor, st_factor                                            ! Factors converting derivative units from global to local elements
  real*8           :: R_error, Z_error, R_error_max, Z_error_max, B_R_error_max, B_Z_error_max ! Errors in coordinates between JOREK and GVEC
  real*8           :: R_error_sum, Z_error_sum, B_R_error_sum, B_Z_error_sum                               
  real*8           :: R_s_error, Z_s_error, R_s_error_max, Z_s_error_max, R_s_error_sum, Z_s_error_sum                              
  real*8           :: R_t_error, Z_t_error, R_t_error_max, Z_t_error_max, R_t_error_sum, Z_t_error_sum                             
  real*8           :: R_st_error, Z_st_error, R_st_error_max, Z_st_error_max, R_st_error_sum, Z_st_error_sum

  real*8           :: Ri, Ri_s, Ri_t, Ri_p, Ri_st, Ri_ss, Ri_tt, Ri_sp, Ri_tp, Ri_pp           ! Interpolated values for R, Z and derivatives
  real*8           :: Zi, Zi_s, Zi_t, Zi_p, Zi_st, Zi_ss, Zi_tt, Zi_sp, Zi_tp, Zi_pp
  real*8           :: dumm_var, dumm_var_s, dumm_var_t, dumm_var_st, dumm_var_ss, dumm_var_tt
  real*8           :: B_R_loc, B_R_s, B_R_t, B_R_st, B_Z_loc, B_Z_s, B_Z_t, B_Z_st, B_phi_loc, B_phi_s, B_phi_t, B_phi_st
  real*8           :: J_R_loc, J_R_s, J_R_t, J_R_st, J_Z_loc, J_Z_s, J_Z_t, J_Z_st, J_phi_loc, J_phi_s, J_phi_t, J_phi_st
  
  integer          :: i_harm
  real*8           :: HZ_coord(n_coord_tor)                      ! Local grid toroidal basis functions

  integer          :: i_var=1, i_s=2, i_t=3, i_st=4              ! Indexes for 4D test data
  integer          :: n_skip=0                                   ! Number of skipped test data points
  real*8           :: abs_tol=1.d-5                              ! Tolerance for accepting test data points
  integer          :: in_gvec=11                                 ! IO handle for gvec input
  integer          :: gvec_preamble_lines=124                    ! Number of preamble lines in gvec import

  ! Read test data parameters parameters
  open(in_gvec, file=trim(file_name), status='old', iostat=ierr, form='formatted', access='sequential')
  if (ierr /= 0) then
    write(*, *) "Cannot open GVEC test file...", file_name
    stop
  end if
  do idx=1, gvec_preamble_lines
    read(in_gvec, *)
  enddo
  read(in_gvec, *) n_rad, n_theta, n_phi
  write(*, *)  "n_rad    n_theta    n_phi: ", n_rad, n_theta, n_phi
  read(in_gvec,*)
  read(in_gvec, *) coord_type, num_periods, asym, m_max, n_max, n_modes, sin_range(:), cos_range(:)
  write(*, *) "nfp    asym    m_max    n_max", num_periods, asym, m_max, n_max
  if (n_coord_period .ne. num_periods) then
    write(*, *) "Number of field periods in GVEC equilibrium does not match input file: ", num_periods, n_coord_period
    stop
  endif

  ! Chack JOREK is compiled correctly from equilibrium
  if (n_coord_tor .ne. n_modes) then
    write(*,*) "Number of modes in JOREK and GVEC do not match! (n_coord_tor, n_modes):" , n_coord_tor, n_modes
  endif

  ! Read 3D data
  call tr_allocate(s, 1, n_theta, 1, n_phi, 1, n_rad,   "s", CAT_GRID)
  call tr_allocate(theta, 1, n_theta, 1, n_phi, 1, n_rad,"theta", CAT_GRID)
  call tr_allocate(phi, 1, n_theta, 1, n_phi, 1, n_rad, "phi", CAT_GRID)
  call tr_allocate(R, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad,   "R", CAT_GRID)
  call tr_allocate(Z, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad,   "Z", CAT_GRID)
  call tr_allocate(P, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad,   "P", CAT_GRID)
  call tr_allocate(B_R, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad, "B_R", CAT_GRID)
  call tr_allocate(B_Z, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad, "B_Z", CAT_GRID)
  call tr_allocate(B_phi, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad, "B_phi", CAT_GRID)
  call tr_allocate(J_R, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad, "J_R", CAT_GRID)
  call tr_allocate(J_Z, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad, "J_Z", CAT_GRID)
  call tr_allocate(J_phi, 1, 4, 1, n_theta, 1, n_phi, 1, n_rad, "J_phi", CAT_GRID)
  read(in_gvec, '(A)')
  read(in_gvec,'(*(6(e23.15,:,1X),/))') s
  read(in_gvec, '(A)')
  read(in_gvec,'(*(6(e23.15,:,1X),/))') theta
  read(in_gvec, '(A)')
  read(in_gvec,'(*(6(e23.15,:,1X),/))') phi
  do idx=1,4                                                  ! R, Z
    read(in_gvec, '(A)')
    read(in_gvec,'(*(6(e23.15,:,1X),/))') R(idx,:,:,:)
  enddo
  do idx=1,4
    read(in_gvec, '(A)')
    read(in_gvec,'(*(6(e23.15,:,1X),/))') Z(idx,:,:,:)
  enddo
  do idx=1,2
    read(in_gvec, '(A)')                                    ! Read Pressure and dPds
    read(in_gvec,'(*(6(e23.15,:,1X),/))') P(idx,:,:,:)
  enddo
  P(3:4,:,:,:)=0                                            ! Set poloidal derivatives to zero
  do idx=1,12
    read(in_gvec, '(A)')                                    ! Skip vector potential
    read(in_gvec,'(*(6(e23.15,:,1X),/))') B_R(idx,:,:,:)
  enddo
  do idx=1,4                                                  ! Magnetic field components
    read(in_gvec, '(A)')
    read(in_gvec,'(*(6(e23.15,:,1X),/))') B_R(idx,:,:,:)  
  enddo
  do idx=1,4
    read(in_gvec, '(A)')
    read(in_gvec,'(*(6(e23.15,:,1X),/))') B_Z(idx,:,:,:)   
  enddo
  do idx=1,4
    read(in_gvec, '(A)')
    read(in_gvec,'(*(6(e23.15,:,1X),/))') B_phi(idx,:,:,:) 
  enddo
  do idx=1,4
    read(in_gvec, '(A)')                                    ! Current density Components
    read(in_gvec,'(*(6(e23.15,:,1X),/))') J_R(idx,:,:,:)   
  enddo
  do idx=1,4
    read(in_gvec, '(A)')
    read(in_gvec,'(*(6(e23.15,:,1X),/))') J_Z(idx,:,:,:)
  enddo
  do idx=1,4
    read(in_gvec, '(A)')
    read(in_gvec,'(*(6(e23.15,:,1X),/))') J_phi(idx,:,:,:)   
  enddo
  
  close(in_gvec)
  
  ! Get n_rad, n_phi and n_theta used in import
  open(in_gvec, file=trim("gvec2jorek.dat"), status='old', iostat=ierr, form='formatted', access='sequential')
  if (ierr /= 0) then
    write(*, *) "Cannot open GVEC test file...", file_name
    stop
  end if
  do idx=1, gvec_preamble_lines
    read(in_gvec, *)
  enddo
  read(in_gvec, *) n_rad_jorek, n_theta_jorek, n_phi_jorek
  close(in_gvec)

  ! Write debug file
  open(in_gvec, file="gvec_debug", status='replace', iostat=ierr, form='formatted', access='sequential')
  if (ierr /= 0) then
    write(*, *) "Cannot open GVEC debug file...", file_name
    stop
  end if
  do idx=1, gvec_preamble_lines
    write(in_gvec, '(A)')  '##<< '
  enddo
  write(in_gvec, *)  n_rad, n_theta, n_phi
  write(in_gvec, '(A)')  '##<< '
  write(in_gvec, '(12X,*(I6,:,1X))')  coord_type, num_periods, asym, m_max, n_max, n_modes, sin_range(:), cos_range(:)

  ! Loop through points and compare Ri, Zi to the value from GVEC
  write(*,*) "Running tests"
  dtheta = 1.0 / float(n_theta_jorek)
  ds = 1.0 / float(n_rad_jorek-1)
  R_error_sum = 0.0;   R_s_error_sum=0.0;   R_t_error_sum=0.0;   R_st_error_sum=0.0
  Z_error_sum = 0.0;   Z_s_error_sum=0.0;   Z_t_error_sum=0.0;   Z_st_error_sum=0.0
  B_R_error_sum = 0.0; B_Z_error_sum=0.0;
  R_error_max = 0.0;   R_s_error_max=0.0;   R_t_error_max=0.0;   R_st_error_max=0.0
  Z_error_max = 0.0;   Z_s_error_max=0.0;   Z_t_error_max=0.0;   Z_st_error_max=0.0
  B_R_error_max = 0.0; B_Z_error_max = 0.0;
  s_factor = (n_rad_jorek-1) / bloating_factor                            ! WARNING: this only works for a uniform grid
  t_factor = (n_theta_jorek) / (2 * PI)
  st_factor = s_factor * t_factor
  do i_rad=1, n_rad
    do i_phi=1, n_phi
      do i_theta=1, n_theta
        s_loc = s(i_theta, i_phi, i_rad)
        theta_loc = theta(i_theta, i_phi, i_rad)
        phi_loc = phi(i_theta, i_phi, i_rad)
        
        ! WARNING - assumes that the element list was empty before importing gvec mesh
        i_rad_loc = int(s_loc / ds) + 1
        if (s_loc .eq. 1) i_rad_loc = n_rad_jorek-1
        i_theta_loc = int(theta_loc / dtheta) + 1
        if (theta_loc .ge. 1) i_theta_loc = 1
        i_elm  = n_theta_jorek*(i_rad_loc-1) + i_theta_loc
        if (i_elm .gt. element_list%n_elements) then
          !write(*,*) "ERROR: i_elm is outside grid: ", i_elm, element_list%n_elements
          !write(*, *) s_loc, i_rad_loc, theta_loc, i_theta_loc
          n_skip = n_skip+1
          cycle
        endif

        ! Interpolate element location
        if (s_loc .ne. 1) s_loc = mod(s_loc, ds) / ds
        theta_loc = mod(theta_loc, dtheta) / dtheta
        
        ! Avoid floating point errors in determining i_elm by avoiding points close to nodes 
        if ((abs(s_loc).lt.abs_tol) .or. (abs(s_loc-1.0).lt.abs_tol)) then
          !write(*,*) "RAD SKIP: ", s_loc, abs_tol
          n_skip = n_skip+1; cycle
        endif
        if ((theta_loc.lt.abs_tol) .or. (abs(theta_loc-1.0).lt.abs_tol)) then
          !write(*,*) "THETA SKIP: ", theta_loc, abs_tol
          n_skip = n_skip+1; cycle
        endif

        call interp_RZP(node_list,element_list,i_elm,s_loc,theta_loc,phi_loc,               &
                        Ri, Ri_s, Ri_t, Ri_p, Ri_st, Ri_ss, Ri_tt, Ri_sp, Ri_tp, Ri_pp,     &
                        Zi, Zi_s, Zi_t, Zi_p, Zi_st, Zi_ss, Zi_tt, Zi_sp, Zi_tp, Zi_pp)

        HZ_coord(1)   = 1.d0
        do i_harm=1,(n_coord_tor-1)/2
          HZ_coord(2*i_harm)      = + cos(mode_coord(2*i_harm)  *phi_loc)
          HZ_coord(2*i_harm+1)    = - sin(mode_coord(2*i_harm+1)*phi_loc)
        enddo
        B_R_loc = 0.0; B_Z_loc = 0.0;
        do i_harm=1, n_coord_tor
          call interp_gvec(node_list, element_list, i_elm, 1, 1, i_harm, s_loc, theta_loc, dumm_var, dumm_var_s, dumm_var_t, dumm_var_st, dumm_var_ss, dumm_var_tt)      
          B_R_loc = B_R_loc+dumm_var*HZ_coord(i_harm);
          call interp_gvec(node_list, element_list, i_elm, 1, 2, i_harm, s_loc, theta_loc, dumm_var, dumm_var_s, dumm_var_t, dumm_var_st, dumm_var_ss, dumm_var_tt)      
          B_Z_loc = B_Z_loc+dumm_var*HZ_coord(i_harm);
        enddo

        ! Calculate errors
        R_error_sum    = R_error_sum        + abs(Ri - R(i_var,i_theta,i_phi,i_rad))
        Z_error_sum    = Z_error_sum        + abs(Zi - Z(i_var,i_theta,i_phi,i_rad))
        R_s_error_sum  = R_s_error_sum      + abs(Ri_s * s_factor - R(i_s,i_theta,i_phi,i_rad))
        Z_s_error_sum  = Z_s_error_sum      + abs(Zi_s * s_factor - Z(i_s,i_theta,i_phi,i_rad))
        R_t_error_sum  = R_t_error_sum      + abs(Ri_t * t_factor - R(i_t,i_theta,i_phi,i_rad))
        Z_t_error_sum  = Z_t_error_sum      + abs(Zi_t * t_factor - Z(i_t,i_theta,i_phi,i_rad))
        R_st_error_sum = R_st_error_sum     + abs(Ri_st * st_factor - R(i_st,i_theta,i_phi,i_rad))
        Z_st_error_sum = Z_st_error_sum     + abs(Zi_st * st_factor - Z(i_st,i_theta,i_phi,i_rad))
        B_R_error_sum    = B_R_error_sum    + abs(B_R_loc - B_R(i_var, i_theta,i_phi,i_rad))
        B_Z_error_sum    = B_Z_error_sum    + abs(B_Z_loc - B_Z(i_var, i_theta,i_phi,i_rad))
        
        R_error_max       = max(R_error_max,        abs(Ri-R(i_var,i_theta,i_phi,i_rad)))  
        R_s_error_max     = max(R_s_error_max,      abs(Ri_s * s_factor-R(i_s,i_theta,i_phi,i_rad)))  
        R_t_error_max     = max(R_t_error_max,      abs(Ri_t * t_factor-R(i_t,i_theta,i_phi,i_rad))) 
        R_st_error_max    = max(R_st_error_max,     abs(Ri_st * st_factor-R(i_st,i_theta,i_phi,i_rad)))  
        Z_error_max       = max(Z_error_max,        abs(Zi-Z(i_var,i_theta,i_phi,i_rad)))
        Z_s_error_max     = max(Z_s_error_max,      abs(Zi_s * s_factor-Z(i_s,i_theta,i_phi,i_rad)))  
        Z_t_error_max     = max(Z_t_error_max,      abs(Zi_t * t_factor-Z(i_t,i_theta,i_phi,i_rad))) 
        Z_st_error_max    = max(Z_st_error_max,     abs(Zi_st * st_factor-Z(i_st,i_theta,i_phi,i_rad)))  
        B_R_error_max     = max(B_R_error_max,      abs(B_R_loc-B_R(i_var,i_theta,i_phi,i_rad)))  
        B_Z_error_max     = max(B_Z_error_max,      abs(B_Z_loc-B_Z(i_var,i_theta,i_phi,i_rad)))  

        ! Use data arrays for storing debug output 
        R(i_var, i_theta, i_phi, i_rad)      = abs(Ri-R(i_var,i_theta,i_phi,i_rad))  
        R(i_s,   i_theta, i_phi, i_rad)    = abs(Ri_s * s_factor-R(i_s,i_theta,i_phi,i_rad))  
        R(i_t,   i_theta, i_phi, i_rad)    = abs(Ri_t * t_factor-R(i_t,i_theta,i_phi,i_rad)) 
        R(i_st,  i_theta, i_phi, i_rad)   = abs(Ri_st * st_factor-R(i_st,i_theta,i_phi,i_rad))  
        Z(i_var, i_theta, i_phi, i_rad)      = abs(Zi-Z(i_var,i_theta,i_phi,i_rad))
        Z(i_s,   i_theta, i_phi, i_rad)    = abs(Zi_s * s_factor-Z(i_s,i_theta,i_phi,i_rad))  
        Z(i_t,   i_theta, i_phi, i_rad)    = abs(Zi_t * t_factor-Z(i_t,i_theta,i_phi,i_rad)) 
        Z(i_st,  i_theta, i_phi, i_rad)   = abs(Zi_st * st_factor-Z(i_st,i_theta,i_phi,i_rad))  
      enddo
    enddo
  enddo
  write(*,*) "Running tests"
  R_error_sum      = R_error_sum      / abs(float(n_rad * n_phi * n_theta) - n_skip)
  Z_error_sum      = Z_error_sum      / abs(float(n_rad * n_phi * n_theta) - n_skip)
  R_s_error_sum    = R_s_error_sum    / abs(float(n_rad * n_phi * n_theta) - n_skip)
  Z_s_error_sum    = Z_s_error_sum    / abs(float(n_rad * n_phi * n_theta) - n_skip)
  R_t_error_sum    = R_t_error_sum    / abs(float(n_rad * n_phi * n_theta) - n_skip)
  Z_t_error_sum    = Z_t_error_sum    / abs(float(n_rad * n_phi * n_theta) - n_skip)
  R_st_error_sum   = R_st_error_sum   / abs(float(n_rad * n_phi * n_theta) - n_skip)
  Z_st_error_sum   = Z_st_error_sum   / abs(float(n_rad * n_phi * n_theta) - n_skip)
  B_R_error_sum    = B_R_error_sum    / abs(float(n_rad * n_phi * n_theta) - n_skip)
  B_Z_error_sum    = B_Z_error_sum    / abs(float(n_rad * n_phi * n_theta) - n_skip)
  write(*,*) "Errors R, Z, B_R, B_Z, B_phi, J_R, J_Z, J_phi          "
  write(*,*) "Errors R_s, Z_s, B_R_s, B_Z_s, B_phi_s, J_R_s, J_Z_s, J_phi_s    "
  write(*,*) "Errors R_t, Z_t, B_R_t, B_Z_t, B_phi_t, J_R_t, J_Z_t, J_phi_t    "
  write(*,*) "Errors R_st, Z_st, B_R_st, B_Z_st, B_phi_st, J_R_st, J_Z_st, J_phi_st "
  write(*,'(*(6(e23.15,:,1X),/))')  R_error_sum, Z_error_sum, R_error_max, Z_error_max
  write(*,'(*(6(e23.15,:,1X),/))')  R_s_error_sum, Z_s_error_sum, R_s_error_max, Z_s_error_max
  write(*,'(*(6(e23.15,:,1X),/))')  R_t_error_sum, Z_t_error_sum, R_t_error_max, Z_t_error_max
  write(*,'(*(6(e23.15,:,1X),/))')  R_st_error_sum, Z_st_error_sum, R_st_error_max, Z_st_error_max
  
  ! Write out interpolated values for debugging against original data
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X1(R)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') R(i_var,:,:,:)
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X1_s(R)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') R(i_s,:,:,:)
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X1_t(R)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') R(i_t,:,:,:)
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X1_st(R)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') R(i_st,:,:,:)
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X2(Z)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') Z(i_var,:,:,:)
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X2_s(Z)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') Z(i_s,:,:,:)
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X2_t(Z)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') Z(i_t,:,:,:)
  write(in_gvec, '(A)') '##<< 3D Scalar Variable: "X2_st(Z)"'
  write(in_gvec,'(*(6(e23.15,:,1X),/))') Z(i_st,:,:,:)
  close(in_gvec)

  ! Clean up
  call tr_deallocate(s,   "s", CAT_GRID)
  call tr_deallocate(theta,"theta", CAT_GRID)
  call tr_deallocate(phi, "phi", CAT_GRID)
  call tr_deallocate(R,   "R", CAT_GRID)
  call tr_deallocate(Z,   "Z", CAT_GRID)
  call tr_deallocate(B_R,   "B_R", CAT_GRID)
  call tr_deallocate(B_Z,   "B_Z", CAT_GRID)
  call tr_deallocate(B_phi, "B_phi", CAT_GRID)
  call tr_deallocate(J_R,   "J_R", CAT_GRID)
  call tr_deallocate(J_Z,   "J_Z", CAT_GRID)
  call tr_deallocate(J_phi, "J_phi", CAT_GRID)
end subroutine 
  
