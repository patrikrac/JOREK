!> Testing the coupling of the projections of particles to JOREK

program recobmination_loop

!use mod_integrate_recombination
use particle_tracer
use mod_particle_diagnostics
use mpi
use mod_interp
use mod_atomic_elements
use mod_particle_io
use mod_event
use mod_project_particles
!use mod_particle_loop
use mod_jorek_timestepping
use mod_random_seed
use mod_basisfunctions
use nodes_elements
use phys_module, only: tstep,restart_particles, restart, t_start, nout
use phys_module, only: CENTRAL_MASS, CENTRAL_DENSITY, xcase, xpoint
use phys_module, only: n_particles, nstep_particles, nsubstep_particles, tstep_particles
use phys_module, only: use_ncs, use_pcs, use_ccs, deuterium_adas,sqrt_mu0_over_rho0
use phys_module, only: filter_perp, filter_hyper, filter_par, filter_perp_n0, filter_hyper_n0, filter_par_n0
! use phys_module, only: use_kn_sputtering , use_kn_cx, use_kn_ionisation, use_kn_sputtering

use constants,   only: MU_ZERO, ATOMIC_MASS_UNIT, K_BOLTZ, EL_CHG

use mod_particle_sputtering, only: particle_sputter, sample_fluid_particle_energy
use mod_projection_functions, only: proj_f_combined_density, &
                                    proj_f_combined_energy, proj_f_combined_par_momentum
! use mod_radiation, only : proj_PLT
use mod_particle_puffing
use mod_edge_domain
use mod_edge_elements

 use mod_atomic_coeff_deuterium, only: ad_deuterium 

use data_structure, only: type_bnd_element_list, type_bnd_node_list 
use mod_boundary,   only: boundary_from_grid
use equil_info

!$ use omp_lib

implicit none

type(event)                                       :: fieldreader, partreader
type(event)                                       :: D_sputter_event,gas_puff_event ,gas_puff2_event, gas_puff3_event!, partwriter
type(adf11_all)                                   :: adas
type(pcg32_rng), dimension(:), allocatable        :: rng
type(count_action)                                :: counter
type(projection), target                          :: jorek_feedback, project_density, project_current
type(jorek_timestep_action), target               :: jorek_stepper
type(particle_sputter)                            :: D_sputter_source
type(type_edge_domain), allocatable, dimension(:) :: edge_domains
type(edge_elements)                               :: D_edge
type(particle_puffing)                            :: gas_puff
type(particle_puffing)                            :: gas_puff2,gas_puff3
type(write_particle_diagnostics)                  :: diag

real*8, parameter  :: binding_energy = 2.18d-18 ! ionization energy of a hydrogen atom [J] (= 13.6 eV)
real*8    :: target_time, projection_time
real*8    :: physical_particles, weight
real*8    :: tstep_keep,oldtime, step_rest_time, particle_step_time, particle_start_time, diag_time
real*8    :: rho_norm, t_norm, v_norm, E_norm, M_norm, N_norm, tstep_si, timesteps
real*8    :: v_kin_temp, E(3), B(3), psi, U, B_norm(3)
real*8    :: rescale_coef, T_axis(1), E_axis, E_hot, rho_part, v2, tstart_jorek
!$ real*8 :: w0, w1, mmm(3)

integer   :: n_particles_local,n_reflect,ifail
integer   :: i, j, k, l, m, n_steps, i_elm_old,ierr
integer   :: seed, i_rng, n_stream

! Puffing parameters
real*8  :: r_valve, R_valve_loc, Z_valve,  R_valve_loc2, Z_valve2, puff_rate,t_puff_start,t_puff_slope, puffing_rate_start
real*8   ::r_valve3, R_valve_loc3, Z_valve3,puff_rate3,poly_R(4),poly_Z(4)
integer :: n_puff
logical :: puff_t_dependent,boxpuff


!use physics
logical :: use_kn_recombination, use_kn_puffing, use_kn_cx, use_kn_ionisation , use_kn_sputtering,use_kn_line_radiation
logical  :: run_stepper, run_rec !, one_rec_only !< when recombination is used

! diagnostics
real*8    :: density_tot, density_in, density_out,  pressure, pressure_in, pressure_out
real*8    :: mom_par_tot, mom_par_in, mom_par_out, kin_par_tot, kin_par_out, kin_par_in
real*8    :: particles_remaining, momentum_remaining, energy_remaining, all_particles, all_momentum, all_energy
integer   :: superparticles_remaining,all_superparticles,closest_iteration!, part_i_save,part_n_save
!integer   :: particles_per_element


! Start up MPI, jorek
call sim%initialize(num_groups=1)

!> make sure tstep from namelist doesn't get overwritten
tstep_keep        = tstep
timesteps         = tstep_particles

!> saving part_restart every part_n_save steps
!part_i_save = 1
!part_n_save = 500

! Set up the field reader
fieldreader = event(read_jorek_fields_interp_linear(basename='jorek', i=-1))
call with(sim, fieldreader)
tstep = tstep_keep !< fieldreader overwrites tstep, do this to counter that

if (restart_particles) then 

   if (sim%my_id == 0) write(*,*) 'INFO: READING PARTICLES RESTART FILE'
   partreader = event(read_action(filename='part_restart.h5'))
   call with(sim, partreader) !<defines sim%groups en particles, if more than one group, change num_groups

   n_particles_local = size(sim%groups(1)%particles(:)) !< sputtering and puffing amount is function 
   write(*,*) "n_particles_local = ", n_particles_local
   !We should make an option to use partreader but increase n_particles
   !may be similar to phi_zero_whrite to a sim_in and sim_out but with different allocation size.
      
else
	if (sim%my_id == 0) write(*,*) 'INFO: INITIALIZING PARTICLES', sim%n_cpu, " cpus "

	! Read Open ADAS data
	write(*,*) "deuterium_adas (12)",  deuterium_adas
	adas = read_adf11(sim%my_id,'12_h')

	!> is this needed for neutrals?
	if (sim%my_id .eq. 0) call boundary_from_grid(sim%fields%node_list, sim%fields%element_list, bnd_node_list, bnd_elm_list, .false.)
	call broadcast_boundary(sim%my_id, bnd_elm_list, bnd_node_list)

	call update_equil_state(sim%my_id, sim%fields%node_list, sim%fields%element_list, bnd_elm_list, xpoint, xcase )
	!<

	! Set up particle group characteristics
	sim%groups(1)%Z    = -2 !< for deuterium 1
	sim%groups(1)%mass = atomic_weights(-2) !< atomic mass units
	sim%groups(1)%ad   = adas
	
	! setting up particles per MPI node
	n_particles_local = int(n_particles/sim%n_cpu) 
	allocate(particle_kinetic_leapfrog::sim%groups(1)%particles(n_particles_local))

	!>initialise particles here if needed
	! call initialise_particles_H_mu_psi 
	! call adjust weights

	select type (p => sim%groups(1)%particles)
	type is (particle_kinetic_leapfrog)  
	!> only set everything to zero when you do not initialise particles
	p(:)%q      = 0 !< for neutrals
	p(:)%weight = 0.0!weight
	p(:)%i_elm  = 0
	p(:)%v(1)   = 0.d0 
	p(:)%v(2)   = 0.d0
	p(:)%v(3)   = 0.d0
	! call boris_all_initial_half_step_backwards_RZPhi(p, sim%groups(1)%mass, sim%fields, sim%time, timesteps)

  end select

endif ! (restart_particles)

! setting up particles per MPI node and timestep
rho_part    = 1.195d19 !(corrected value to obtain density=1.441e17 (as in benchmark, for original profile with toroidal flux) 
! n_particles_local = int(n_particles/sim%n_cpu) 
! timesteps         = tstep_particles
! tstep_keep        = tstep

! selecting physics (should be done in input file)
use_kn_puffing       = .true. !.false. 
use_kn_cx            = .true. !.true.
use_kn_ionisation    = .true. !.false.!.false.
use_kn_sputtering    = .true. !.false. !false
use_kn_recombination = .true.  !
use_kn_line_radiation= .true.

! Read Open ADAS data for plasma fluid
 if (deuterium_adas .and. use_kn_recombination) ad_deuterium =  read_adf11(sim%my_id,'96_h') !< move to core (jorek2_main for particles)
 
n_norm    = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
rho_norm  = central_mass * ATOMIC_MASS_UNIT * n_norm                  ! rho_SI = rho_norm * rho
t_norm    = sqrt((MU_ZERO * rho_norm))                           ! t_SI   = t_norm * t_jorek 
 
! Setting up edge_elements and amount of sputtered super particles per event
if (use_kn_sputtering) then  
  n_reflect = int(n_particles_local* sim%n_cpu * 5.d-4) !1.d-3 int(n_particles_local * 2.d-3)
  D_sputter_source = initialise_sputtering(sim%fields%node_list, sim%fields%element_list, n_reflect)
  D_sputter_event = event(D_sputter_source)
endif

! setting up particle puffing Top puff
! puff_t_dependent = .true. !.true. !< select if you want time dependent puffing
! puff_rate = 40.d21 !40.d21!100.d21 !8.85d21 !4.d21 !8.d22 !4.d22 !4.d21
! puffing_rate_start = 10.d21 !10 40 worked, 20 before
! r_valve     = 0.02d0!              0.01d0 !0.04d0 !0.02d0 !0.04d0 !.005d0
! R_valve_loc = 4.27d0!               4.4d0 !4.42787 !4.42787!2.33!2.6!2.1 !< for JET test !1.98991!2.58888  or 1.98991
! Z_valve     = -3.74d0!             -3.8d0 !-3.7 !-3.77948! -1.86 !-1.0!-1.75 !-0.550736!1.86579   or -0.550736

! R_valve_loc2 = 5.55d0!                  5.4d0 !5.46d0
! Z_valve2     = -4.35d0!                  -4.19d0 !-4.2d0

! puff_rate3 = 160.d21 !136.d21 ! 109.d21 !72.d21 !160.d21 !160.d21!85.d21
! R_valve_loc3 = 6.05d0!                  5.4d0 !5.46d0
! Z_valve3     = 4.15d0! 
! r_valve3    = 0.10d0!  .12

!Bot puff
puff_t_dependent = .true. !.true. !< select if you want time dependent puffing
puff_rate = 40.d21 !40.d21!100.d21 !8.85d21 !4.d21 !8.d22 !4.d22 !4.d21
puffing_rate_start = 10.d21 !10 40 worked, 20 before
r_valve     = 0.05d0 !0.02d0!              0.01d0 !0.04d0 !0.02d0 !0.04d0 !.005d0
R_valve_loc = 4.3d0 !4.27d0!               4.4d0 !4.42787 !4.42787!2.33!2.6!2.1 !< for JET test !1.98991!2.58888  or 1.98991
Z_valve     = -3.8d0 !-3.74d0!             -3.8d0 !-3.7 !-3.77948! -1.86 !-1.0!-1.75 !-0.550736!1.86579   or -0.550736

R_valve_loc2 = 5.5d0 !5.55d0!                  5.4d0 !5.46d0
Z_valve2     = -4.35d0!                  -4.19d0 !-4.2d0

puff_rate3 = 160.d21 !136.d21 ! 109.d21 !72.d21 !160.d21 !160.d21!85.d21
R_valve_loc3 = 6.05d0!                  5.4d0 !5.46d0
Z_valve3     = 4.15d0! 
r_valve3    = 0.10d0!  .12
poly_R = (/5.77d0 ,6.735d0 ,5.72d0 ,6.68d0 /)
poly_Z = (/4.51d0 ,3.760d0 ,4.46d0 ,3.71d0 /)
boxpuff = .true.

!R_valve_loc = 4.307! touching leg
!Z_valve     = -3.7898!
if (use_kn_puffing) then  
  n_puff      = int(5.d-5*n_particles_local* sim%n_cpu) !0.25 0.5d-4 !< now total n_puff
  if (puff_t_dependent) then
	t_puff_start = 5000*t_norm !25000*t_norm !34995*t_norm !5000*t_norm !< start puffing after this amount of seconds, t_SI = t_jorek*t_norm jorek time units
	t_puff_slope = 8.d-3 !4.d-3 !< linearly ramps up the puffing during this time
	!puffing_rate_start = 5.d21 !40 worked, 20 before
	!gas_puff = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc, Z_valve, puff_t_dependent, t_puff_start, t_puff_slope)
	!gas_puff2 = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc2, Z_valve2, puff_t_dependent, t_puff_start, t_puff_slope)
    gas_puff = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc, Z_valve, puff_t_dependent=puff_t_dependent,t_puff_start=t_puff_start,t_puff_slope=t_puff_slope,puffing_rate_start=puffing_rate_start/2.d0)
	gas_puff2 = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc2, Z_valve2, puff_t_dependent=puff_t_dependent,t_puff_start=t_puff_start,t_puff_slope=t_puff_slope,puffing_rate_start=puffing_rate_start/2.d0)
	gas_puff3 = particle_puffing(n_puff, puff_rate3   , r_valve3, R_valve_loc3, Z_valve3, puff_t_dependent=puff_t_dependent,t_puff_start=t_puff_start,t_puff_slope=t_puff_slope, &
				puffing_rate_start=40.d21,poly_R=poly_R,poly_Z=poly_Z,boxpuff=boxpuff) !20.d21
  else

	! n_puff      = int(0.25d-4*n_particles_local* sim%n_cpu) !0.5d-4
	gas_puff = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc, Z_valve) ! was 1
	!gas_puff2 = particle_puffing(n_puff, 0.5d21, r_valve, 5.41058, -4.20272)!-0.0) !-1.77 ! jet 2.8d0, -1.77
	gas_puff2 = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc2, Z_valve2)!-0.0) !-1.77 ! jet 2.8d0, -1.77
	gas_puff3 = particle_puffing(n_puff, 20.d21, 0.12d0, 6.05, 4.15)
  end if
  gas_puff_event = event(gas_puff)
  gas_puff2_event = event(gas_puff2)
  gas_puff3_event = event(gas_puff3)
	!gas_puff = particle_puffing(n_puff, 5d22, r_valve, R_valve_loc, Z_valve)
	
	if (sim%my_id .eq.0) then
	write(*,*) "Gas puffing rate [#/s] : ", puff_rate
	write(*,*) "puff_t_dependent : ",puff_t_dependent, "with puff slope",t_puff_slope,"starting at", t_puff_start, "s"
	endif
else 
	n_puff = 0.d0
	gas_puff = particle_puffing(n_puff, 5d20, r_valve, R_valve_loc, Z_valve)
	gas_puff2 = particle_puffing(n_puff, 5d20, r_valve, R_valve_loc, Z_valve)
	gas_puff3 = particle_puffing(n_puff, 5d20, r_valve, R_valve_loc, Z_valve)
endif

! write(*,*) 'main : t_start = ',t_start

! n_norm    = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
! rho_norm  = central_mass * ATOMIC_MASS_UNIT * n_norm                  ! rho_SI = rho_norm * rho
! t_norm    = sqrt((MU_ZERO * rho_norm))                           ! t_SI   = t_norm * t_jorek

tstep_si  = tstep * t_norm
n_steps   = floor(tstep_si / timesteps)
timesteps = tstep_si / n_steps
n_steps   = tstep_si / timesteps

if (sim%my_id .eq.0) then
  write(*,*) 'main : t_start = ',t_start
  write(*,*) ' adapt time step to be multiple of jorek time step'
  write(*,*) "tstep = ", tstep_si, n_steps, timesteps
  write(*,*) "check :", n_steps, tstep_si - n_steps*timesteps
endif


!partwriter = event(write_action()) !< event writing particle restart files
! Set up feedback
jorek_feedback = new_projection(sim%fields%node_list, sim%fields%element_list, &
                                filter_n0 = filter_perp_n0, filter_hyper_n0 = filter_hyper_n0, filter_parallel_n0=filter_par_n0,      &
                                filter = filter_perp, filter_hyper = filter_hyper, filter_parallel=filter_par, fractional_digits = 9, &
                                do_zonal = .false., calc_integrals=.false., to_vtk=.TRUE., to_h5 = .false., basename='projections', nsub=2)


aux_node_list => jorek_feedback%node_list

!> define feedback size as function of the coupling scheme
if (use_ncs) then
  allocate(jorek_feedback%rhs(n_order+1, n_vertex_max, sim%fields%element_list%n_elements, n_tor, 4)) !< stacksize should be big enough
elseif (use_pcs) then  ! not implemented yet!
  allocate(jorek_feedback%rhs(n_order+1, n_vertex_max, sim%fields%element_list%n_elements, n_tor, 1))
elseif (use_ccs) then  ! not implemented yet!
  allocate(jorek_feedback%rhs(n_order+1, n_vertex_max, sim%fields%element_list%n_elements, n_tor, 4))
else
  stop 'define use_ncs, use_pcs or use_ccs'
endif
jorek_feedback%rhs = 0.d0

!Setting up projections 
project_density = new_projection(sim%fields%node_list, sim%fields%element_list, &
                     filter    = filter_perp,    filter_hyper    = filter_hyper,    filter_parallel    = filter_par, &
                     filter_n0 = 5.d-6, filter_hyper_n0 = 2.d-11, filter_parallel_n0 = filter_par_n0, &
                     f=[proj_f(proj_one, group = 1)], &
                     fractional_digits = 9,  to_vtk=.TRUE., to_h5=.FALSE., basename='density', nsub=2)

call with(sim, project_density)


! For proper timestepping, the projections need to be defined before the jorek timestepper
jorek_stepper = new_jorek_timestep_action(jorek_feedback%node_list)

diag = write_particle_diagnostics(filename='diag.h5', append=.true.)

if (restart_particles) then
   tstart_jorek = sim%time + tstep_si
else
   tstart_jorek = sim%time
endif

if (sim%my_id .eq. 0) write(*,*) 'tstart_jorek : ',tstart_jorek

diag_time = 1.1*n_steps*timesteps
events = [ new_event_ptr(jorek_feedback,   start = tstart_jorek),            &
           new_event_ptr(jorek_stepper,    start = tstart_jorek),            & 
!		   event(D_sputter_source      ,start = tstart_jorek+tstep_si/2.d0, step=tstep_si),&
!		   event(gas_puff, step = 5.d-6),                                &
!		   event(gas_puff2, step = 5.d-6),                                & !
!          new_event_ptr(D_sputter_source, start = tstart_jorek, step=diag_time), & !20.d-7 1 sputter per jorek timestep step=1.d-6), &
!          event(count_action(),           start = tstart_jorek, step=1d-5), &
!          event(write_particle_diagnostics(filename='diag.h5'), step=diag_time), &
!         event(write_action(), step=diag_time),                        &
!           new_event_ptr(project_density, step=0.11d-6),                  &
!		   new_event_ptr(project_density, step=2.5d-6),                  &
!          event(diag, step=diag_time)									 &
           event(stop_action(), start=1d12)                              &
        ]


!< physics and projection like gas_puff, sputtering and projection density can be used in the events[] list as well.
!< We've decided to call them separately as this then doesn't interrupt the timestepping and may be altering the result.  ~Sven
!< look at the "if (run_stepper)then " part to see how to call events or projections as function of the jorek timestepper 
!< without interupting the simulation


! if(.not. restart_particles) then
jorek_stepper%extra_event => events(1) !< is used as first event before enetering particle loop (skipped if particles_restart)
! endif !restart_particles
!================================================================================================
!                                      MAIN PARTICLE LOOP
!================================================================================================


! Set up random numbers for ionisation probability
seed = random_seed()
n_stream = 1
!$ n_stream = omp_get_max_threads()
allocate(rng(n_stream))
do i=1,n_stream
  call rng(i)%initialize(1, seed, n_stream, i)
end do

! Call events at sim%time once to help event scheduler, before entering particle loop
step_rest_time = 0.d0
call with(sim, events, at=sim%time)

! do recombination to initialise particles. Maintains conservation with first jorek timestep
if (.not. restart_particles) then 
	if (sim%my_id .eq. 0) write(*,*) "Do 1 particle recombination"
	call do_1particle_recombination_3D(element_list,node_list,jorek_stepper,rng) 

	call with(sim, project_density) !< directly project the first recombination at t=0
endif !.not. restart_particles

do while (.not. sim%stop_now)

  target_time = next_event_at(sim, events) 
  particle_start_time = (sim%time - step_rest_time)
  particle_step_time  = target_time - particle_start_time
  n_steps             = particle_step_time/timesteps
  step_rest_time      = particle_step_time - real(n_steps,8) * timesteps

  if (sim%my_id .eq. 0) then
     if (n_steps < 10) write(*,*) 'low n_steps,', n_steps
     write(*,*) 'Time difference between particles and jorek: ', step_rest_time
     write(*,*) "PARTICLE : target time         : ",target_time
     write(*,*) "PARTICLE : timesteps           : ",timesteps
     write(*,*) "PARTICLE : sim%time            : ",sim%time
     write(*,*) "PARTICLE : particle_start_time : ",particle_start_time
     write(*,*) "PARTICLE : particle_step_time  : ",particle_step_time
     write(*,*) "PARTICLE : n_steps             : ",n_steps
     write(*,*) "PARTICLE : step_rest_time      : ",step_rest_time
  endif

  !> ionisation + CX + pushing the particles + calculating the feedback
  call loop_particle_kinetic_local(sim, jorek_feedback, rng, timesteps, n_steps, particle_start_time)
  
  
  !> do particle sources if next event is jorek_stepper (event(2))
  !> run_stepper tells if particle sources will run. run_stepper must be before call with(sim..
  run_stepper = events(2)%run_at(target_time) !
  projection_time = target_time !< = target_time, but for the purpose of projections. needed for call to project diagnostics such as project_density
  sim%time = target_time !< set time to exactly target_time for calling next event
  
  call with(sim, events, at=sim%time) !< gives new target time

  !> run particle source routines directly after the jorek_stepper
  !> Density projection added which now run every nout steps
  !> You can put anything in here that you want to solely depend on the jorek timestep.
  if (run_stepper)then

	!> call projection only every nout jorek steps. useful for longer runs
	closest_iteration = nint((projection_time - tstart_jorek)/(tstep_si*nout)) !< very similar to run_at function. May be put this in a function?
	if ( (abs((tstart_jorek +closest_iteration*tstep_si*nout) -projection_time) .le. 1.d-13) .or. sim%stop_now) then !< == true every tstep * nout steps
		call with(sim, project_density)
		
		 ! if (use_kn_line_radiation) call with(sim, project_PLT)
	endif !< write projection or diagnostics
	
	!if (part_i_save .ge. part_n_save) then
	!	call with(sim, partwriter)
	!	part_i_save = 0
	!endif
	!part_i_save = part_i_save + 1
	
	if (use_kn_recombination) then
	  !call recombination
	  call do_1particle_recombination_3D(element_list,node_list,jorek_stepper,rng) 
    endif !use_kn_recombination
	
	if (use_kn_sputtering) then
	  ! call sputtering
	   call with(sim, D_sputter_event) !event(D_sputter_source))
	endif   !use_kn_sputtering
	  
	if (use_kn_puffing) then
      call with(sim, gas_puff_event) 
	  call with(sim, gas_puff2_event)
	  call with(sim, gas_puff3_event)
    endif ! use_kn_puffing	
  endif !run_stepper
  

!>  separate subroutine?
!=======================================================================
!                     Run diagnostics for conservation check
!======================================================================== 
  call Integrals_3D(sim%my_id, sim%fields%node_list, sim%fields%element_list, density_tot, density_in, density_out, &
                    pressure, pressure_in, pressure_out, kin_par_tot, kin_par_in, kin_par_out, mom_par_tot, mom_par_in, mom_par_out)

  particles_remaining = 0.d0
  momentum_remaining  = 0.d0
  energy_remaining    = 0.d0
  superparticles_remaining = 0 !< just for debugging. Use count_action in actual simulation.!.d0!.d0

  select type (particles => sim%groups(1)%particles)
  type is (particle_kinetic_leapfrog)
#ifdef __GFORTRAN__
    !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
    !$omp parallel do default(none) &
    !$omp shared(sim, particles) &
#endif
    !$omp reduction(+:particles_remaining, momentum_remaining, energy_remaining,superparticles_remaining) &
    !$omp private(j, E, B, psi, U, B_norm)
    do j=1,size(particles,1)

      if (particles(j)%i_elm .le. 0) cycle

      call sim%fields%calc_EBpsiU(sim%time , particles(j)%i_elm, particles(j)%st, particles(j)%x(3), E, B, psi, U)
      B_norm = B/norm2(B)

      particles_remaining = particles_remaining + particles(j)%weight
      momentum_remaining  = momentum_remaining  + particles(j)%weight * dot_product(B_norm,particles(j)%v) *sim%groups(1)%mass * ATOMIC_MASS_UNIT
      energy_remaining    = energy_remaining    + particles(j)%weight * dot_product(particles(j)%v,particles(j)%v) *sim%groups(1)%mass * ATOMIC_MASS_UNIT /2.d0
!      energy_remaining    = energy_remaining    + particles(j)%weight * 2.18d-15
      
      superparticles_remaining = superparticles_remaining + 1

    enddo !j
	!omp end parallel do


  end select

  call MPI_REDUCE(particles_remaining, all_particles, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(momentum_remaining,  all_momentum,  1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(energy_remaining,    all_energy,    1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(superparticles_remaining,all_superparticles,1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)   
  !write(33,'(A,126e16.8)') ' TOTAL P% info : ',sim%time,density_tot+all_particles/1.d20, density_tot, all_particles/1.d20, &
  !                                      mom_par_tot+all_momentum, mom_par_tot, all_momentum, &
  !                                      pressure+kin_par_tot+all_energy, pressure, all_energy, kin_par_tot 
  
  if (sim%my_id .eq. 0) then
    write(*,'(A,3e16.8)') 'REMAINING (START) : ',all_particles, all_momentum, all_energy

    write(*,'(A,126e16.8)') ' TOTAL1 : ',sim%time,density_tot+all_particles/1.d20, density_tot, all_particles/1.d20, &
                                        mom_par_tot+all_momentum, mom_par_tot, all_momentum, &
                                        pressure+kin_par_tot+all_energy, pressure, all_energy, kin_par_tot
										
	!write(33,'(A,126e16.8)') ' TOTAL P% info : ',sim%time,density_tot+all_particles/1.d20, density_tot, all_particles/1.d20, &
    !                                    mom_par_tot+all_momentum, mom_par_tot, all_momentum, &
    !                                    pressure+kin_par_tot+all_energy, pressure, all_energy, kin_par_tot 

	write(*,'(A,I13,A,E8.2,A,F13.10,A)') 'Superparticles in use :',all_superparticles,' of ', n_particles, '| in use :', &
				real(all_superparticles)/n_particles*100.d0,'%'
				
	if ( all_superparticles .gt. 0 )	then
		write(*,'(A,2E16.8)') 'Average weight of particles',(all_particles)/all_superparticles
	endif	!real(count(
	
  endif !(sim%my_id .eq. 0)  
!===================================================  End diagnostics for conservation  
end do ! while

call write_simulation_hdf5(sim, 'part_restart.h5')
! call write_simulation_hdf5(sim, 'restart_part.h5') !< make sure part_restart won't be overwritten (guido solution)

call sim%finalize

contains


subroutine loop_particle_kinetic_local(sim, jorek_feedback, rng, timesteps, n_steps, particle_start_time)
use mod_project_particles
use mod_random_seed
use mod_interp, only: mode_moivre
use mod_basisfunctions
use mod_particle_types, only: copy_particle_kinetic_leapfrog
use mod_sampling, only: boxmueller_transform,sample_chi_squared_3

implicit none

class(particle_sim), target, intent(inout)                :: sim
type(projection), target, intent(inout)                   :: jorek_feedback
type(count_action)                                        :: counter
type(pcg32_rng), dimension(:), allocatable, intent(inout) :: rng
type(particle_kinetic_leapfrog)                           :: particle_tmp

real*8, intent(in)     :: timesteps, particle_start_time 
real*8    :: n_norm, rho_norm, t_norm, v_norm, E_norm, M_norm
real*8    :: t, E(3), B(3), psi, U, n_e, T_e, rz_old(2), st_old(2)
! real*8    :: v_temp(3), T_eV, K_eV, v_kin_temp, B_norm(3)!, v
real*8    :: R_g, Z_g, R_s, R_t, Z_s, Z_t, xjac, HZ(n_tor), HH(4,4), HH_s(4,4), HH_t(4,4)
real*8    :: ion_rate, ion_source, ion_prob, ion_ran(1), cx_ran(8),st_ran(2), cx_source, cx_energy ,PLT
real*8    :: cx_prob, CX_rate
real*8    :: kinetic_energy, ion_energy,line_rad_energy
real*8    :: n_lost_ion, n_lost_ion_all, p_plt_lost,p_plt_lost_all,p_cx_lost,p_cx_lost_all,p_lost_ion,p_lost_ion_all
integer   :: n_super_ionized, n_super_ionized_all
real*8    :: particle_source, velocity_par_source, energy_source
real*8    :: v_temp(3), T_eV, K_eV, v_kin_temp, B_norm(3), v, v_v, v_E,extra_proj
real*8    :: vvector(3),sum_ran(3), E_th, v_th,ran_norm(4)
!$ real*8 :: w0, w1, mmm(3)

integer, intent(in)   :: n_steps
integer   :: i, j, k, l, m, i_elm_old, i_elm 
integer   :: seed, i_rng, n_stream, ierr, nthreads
integer   :: i_tor, index_lm, i_elm_temp
integer   :: n_particles, ifail
logical   :: limits
real*8,allocatable :: feedback_rhs(:,:,:,:,:)

!$ w0 = omp_get_wtime()

n_norm   = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
rho_norm = central_mass * ATOMIC_MASS_UNIT * n_norm                  ! rho_SI = rho_norm * rho
t_norm   = sqrt((MU_ZERO * rho_norm))                           ! t_SI   = t_norm * t_jorek
v_norm   = 1.d0 / t_norm                                        ! V_SI   = v_norm * v_jorek
E_norm   = 1.5d0 / MU_ZERO                                      ! E_SI   = E_norm * E_jorek
M_norm   = rho_norm * v_norm                                    ! momentum normalisation

  n_lost_ion = 0.d0
  n_lost_ion_all = 0.d0
  p_lost_ion   = 0.d0
  p_lost_ion_all   = 0.d0
  p_plt_lost  = 0.d0
  p_plt_lost_all  = 0.d0
  p_cx_lost   = 0.d0
  p_cx_lost_all   = 0.d0
  
  n_super_ionized = 0
  n_super_ionized_all = 0
  
  
jorek_feedback%rhs_gather_time = jorek_feedback%rhs_gather_time + n_steps * timesteps

allocate(feedback_rhs,source=jorek_feedback%rhs)

jorek_feedback%rhs = 0.d0
feedback_rhs       = 0.d0

call with(sim, counter)

select type (particles => sim%groups(1)%particles)
type is (particle_kinetic_leapfrog)
#ifdef __GFORTRAN__
 !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
 !$omp parallel do default(none) &
 !$omp shared(sim, particles, n_steps, timesteps, rng, particle_start_time,        &
 !$omp rho_norm, t_norm, v_norm, E_norm, M_norm, N_norm,                           &
 !$omp use_kn_cx, use_kn_ionisation,use_kn_line_radiation,                                  &
 !$omp CENTRAL_DENSITY, CENTRAL_MASS)                                              &
#endif
 !$omp schedule(dynamic,10) &
 !$omp private(particle_tmp, i_rng, i,j,k,l,m, t, E, B, psi, U, rz_old, st_old,    &
 !$omp i_elm_old, i_elm, n_e, T_e,                                                 &
 !$omp PLT,ion_rate, ion_prob, ion_ran, ion_source, ion_energy, kinetic_energy, line_rad_energy,       &  
 !$omp R_g, R_s, R_t, Z_g, Z_s, Z_t, xjac, HH, HH_s, HH_t, HZ, index_lm, ifail,limits,    &
 !$omp CX_rate, CX_prob, CX_source, CX_energy, v, v_E, v_v,extra_proj,                        &
 !$omp particle_source, velocity_par_source, energy_source, v_temp, K_eV, T_eV, cx_ran,&
 !$omp E_th, v_th,sum_ran,vvector,ran_norm)                                                                 &
 !$omp reduction(+:feedback_rhs,n_lost_ion,p_plt_lost,p_cx_lost,p_lost_ion,n_super_ionized)
 
 ! shared jorek_feedback
 !private 
 do j=1,size(particles,1)

    call copy_particle_kinetic_leapfrog(particles(j),particle_tmp)
!      i_rng = 1
  !$ i_rng = omp_get_thread_num()+1
	!if (particle_tmp%i_elm .le. 0) cycle
    do k=1,n_steps

      if (particle_tmp%i_elm .le. 0) exit

      t = particle_start_time + (k-1)*timesteps

      call sim%fields%calc_EBpsiU(t, particle_tmp%i_elm, particle_tmp%st, particle_tmp%x(3), E, B, psi, U)
      rz_old    = particle_tmp%x(1:2)
      st_old    = particle_tmp%st
      i_elm_old = particle_tmp%i_elm
	  
	  !> in use ionisation as well?
	  call sim%fields%calc_NeTe(t, particle_tmp%i_elm, particle_tmp%st, particle_tmp%x(3), n_e, T_e)
	  ! call calc_ U ne Te vpar
	  ion_source = 0.d0
	  ion_energy = 0.d0
	  
	  call sim%fields%calc_vvector(t, particle_tmp%i_elm, particle_tmp%st, particle_tmp%x(3), vvector)
	  !vvector is fluid flow velocity [v_R, v_Z, v_phi] m/s
	  !TODO: add upper limits if necessary
	  limits = n_e .le. 1e14 .or. T_e * K_BOLTZ / EL_CHG .le. 1.d0 !ADAS limits
	  if (particle_tmp%weight .lt. 0.0d0) write(*,*) "Negative particle weight p(j)%w=", particle_tmp%weight
	  
	  !>for impurities, bremsstrahlung and CX radiation can be added here as well. (see W_rad_example)
	  line_rad_energy = 0.d0
	  if (use_kn_line_radiation .and. .not. limits) then !< before or after Ionisation and CX ??
			call sim%groups(1)%ad%PLT%interp(int(particle_tmp%q), log10(n_e), log10(T_e), PLT) ! [J m^3/s]
			! call ad_deuterium%plt%interp( 1, ne_si_log10, Te_si_log10, LradDrays_T, dLradDrays_dT)
			line_rad_energy = n_e * particle_tmp%weight * PLT * timesteps
	  endif ! use_kn_line_radiation
	  
	  if (use_kn_ionisation .and. .not. limits) then
       
          call sim%groups(1)%ad%SCD%interp(int(particle_tmp%q), log10(n_e), log10(T_e), ion_rate) ! [m^3/s]
          ion_prob = 1.d0 - exp(-ion_rate * n_e * timesteps) ! [0] poisson point process, exponential 

          ! If the weight is to small throw away the particle with the probability, else reduce weight with ionising probability
          ion_source = 0.d0

          if (particle_tmp%weight .le. 1.0d9) then !1.0d9 !1.0d10 1.0d7
            call rng(i_rng)%next(ion_ran)
            if (ion_ran(1) .le. ion_prob) then
              particle_tmp%i_elm  = 0
              ion_source = particle_tmp%weight
			  !superparticles ionized
			  n_super_ionized = n_super_ionized +1
            else
              ion_source = 0.d0
            endif

          else 
            ion_source = particle_tmp%weight * ion_prob
            particle_tmp%weight = particle_tmp%weight * (1.d0 - ion_prob)
          endif 

          kinetic_energy = dot_product(particle_tmp%v,particle_tmp%v) *sim%groups(1)%mass * ATOMIC_MASS_UNIT /2.d0

          ion_energy     = kinetic_energy - binding_energy !<binding energy should be here
		  !<including binding energy will make ion_energy negative, so it becomes a sink for the plasma

	  endif ! use_kn_ionisation

	  
	  ! Charge Exchange
	  ! It is assumed that we will have a exchange between hydrogen isotopes
	  v_temp    = particle_tmp%v
	  cx_source = 0.d0
	  cx_energy = 0.d0
	  
	  if (use_kn_cx  .and. .not. limits) then !< CX uses adas as well. Te limit could be lower.
	  
          call sim%groups(1)%ad%CCD%interp(int(particle_tmp%q+1), log10(n_e), log10(T_e), CX_rate) ! [m^3/s]
          CX_prob = 1.d0 - exp(-CX_rate * n_e * timesteps)

          call rng(i_rng)%next(cx_ran)
           if (cx_ran(1) .le. CX_prob) then
            ! sample boltzman, randomize velocity
            T_eV = T_e * K_BOLTZ / EL_CHG !< T_eV = electron T in [eV]

			!============== NEW CX PARTICLE
			  !Box-Mueller sample velocities with st.dev=1
			  ran_norm = boxmueller_transform(cx_ran(2:5))
			  !>v_temp = sqrt(kT/m) * ran_norm
			  v_temp = sqrt(T_e * K_BOLTZ/(sim%groups(1)%mass * ATOMIC_MASS_UNIT))*ran_norm(2:4)
			  !write(*,*) "vtemp", v_temp
			  !>add bulk fluid flow
			  v_temp = v_temp + vvector 

              CX_source = particle_tmp%weight
              CX_energy   = 0.5d0 * sim%groups(1)%mass * ATOMIC_MASS_UNIT *  (dot_product(particle_tmp%v,particle_tmp%v) - dot_product(v_temp,v_temp))
			
              !write(*,*) "neTe",n_e,T_e			
			  !write(*,*) "CX", vvector
          endif ! cx_ran
	  endif ! use_kn_cx
	  
	  if (isnan(ion_source * ion_energy + cx_source * cx_energy - line_rad_energy)) then
		write(*,*) "ion_energy", ion_energy
		write(*,*) "cx_energy", cx_energy
		write(*,*) "line_rad_energy", line_rad_energy
		particle_tmp%i_elm  = 0
		CYCLE !< don't feed this particle into the feedback
		
	  endif
	  
	  
	  ! feedback from each particle at each timestep
	  energy_source       = ion_source * ion_energy + cx_source * cx_energy - line_rad_energy
	  particle_source     = ion_source * sim%groups(1)%mass * ATOMIC_MASS_UNIT !< mass source in SI
	  velocity_par_source = ion_source * dot_product(B, particle_tmp%v) * sim%groups(1)%mass * ATOMIC_MASS_UNIT &	
			+ CX_source  * dot_product(B, particle_tmp%v - v_temp) * sim%groups(1)%mass * ATOMIC_MASS_UNIT 
			   
	  particle_tmp%v = v_temp 
	  n_lost_ion = n_lost_ion + ion_source	!< local sum #particles lost due to ionisation
	  p_lost_ion = p_lost_ion + ion_source * ion_energy
	  p_plt_lost = p_plt_lost + line_rad_energy
	  p_cx_lost  = p_cx_lost + cx_source * cx_energy
	  !Calculate the projection of the ion source in real-time
		call basisfunctions(particle_tmp%st(1), particle_tmp%st(2), HH, HH_s, HH_t)
		call mode_moivre(particle_tmp%x(3), HZ)
			  
		do l=1,n_vertex_max
		  do m=1,n_order+1

			index_lm = (l-1)*(n_order+1) + m

			v   = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * particle_source     * t_norm / rho_norm
			v_E = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * energy_source       * t_norm / E_norm
			v_v = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) * velocity_par_source * t_norm / m_norm
			extra_proj = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) *particle_tmp%weight * 1.d0/real(n_steps,8) !<average density over jorek timesteps!real(floor(k/n_steps))!1.d0 !<density proj

			do i_tor=1,n_tor
			  feedback_rhs(m,l,i_elm_old,i_tor,1) = feedback_rhs(m,l,i_elm_old,i_tor,1) + HZ(i_tor) * v
			  feedback_rhs(m,l,i_elm_old,i_tor,2) = feedback_rhs(m,l,i_elm_old,i_tor,2) + HZ(i_tor) * v_E
			  feedback_rhs(m,l,i_elm_old,i_tor,3) = feedback_rhs(m,l,i_elm_old,i_tor,3) + HZ(i_tor) * v_v
			  feedback_rhs(m,l,i_elm_old,i_tor,4) = feedback_rhs(m,l,i_elm_old,i_tor,4) + HZ(i_tor) * extra_proj !< buiten de steps loop
			enddo

		  enddo
		enddo
	  
	  
      if (particle_tmp%i_elm .gt. 0) then
        ! Push the particle and determine it's new location.
        call boris_push_cylindrical(particle_tmp, sim%groups(1)%mass, E, B, timesteps)

        call find_RZ_nearby(sim%fields%node_list, sim%fields%element_list, rz_old(1), rz_old(2), st_old(1), st_old(2), i_elm_old, &
                            particle_tmp%x(1), particle_tmp%x(2), particle_tmp%st(1), particle_tmp%st(2), particle_tmp%i_elm, ifail)
      end if
   
    end do ! steps 

    call copy_particle_kinetic_leapfrog(particle_tmp, particles(j))

  
  end do   ! particles
  !$omp end parallel do
  
end select

if (use_ncs) then
    write(*,*) 'GATHER TIME : ',jorek_feedback%rhs_gather_time
    !jorek_feedback%rhs = feedback_rhs / jorek_feedback%rhs_gather_time !* TWOPI
	jorek_feedback%rhs(:,:,:,:,1:3) = feedback_rhs(:,:,:,:,1:3) / jorek_feedback%rhs_gather_time !* TWOPI
	jorek_feedback%rhs(:,:,:,:,4) = feedback_rhs(:,:,:,:,4)
    jorek_feedback%rhs_gather_time = 0.d0
else
    jorek_feedback%rhs = feedback_rhs 
  endif
  
deallocate(feedback_rhs)

call MPI_REDUCE(n_lost_ion, n_lost_ion_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
call MPI_REDUCE(p_lost_ion, p_lost_ion_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
call MPI_REDUCE(p_plt_lost, p_plt_lost_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
call MPI_REDUCE(p_cx_lost, p_cx_lost_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
call MPI_REDUCE(n_super_ionized, n_super_ionized_all, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

if (sim%my_id .eq. 0) write(*,'(A46,E14.6,I6)') "Lost superparticles at t due to ionisation: ", sim%time, n_super_ionized_all
if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') " Lost particles at t due to ionisation: ", sim%time, n_lost_ion_all
p_lost_ion_all = p_lost_ion_all / (timesteps * n_steps)
p_plt_lost_all = p_plt_lost_all / (timesteps * n_steps)
p_cx_lost_all = p_cx_lost_all / (timesteps * n_steps)
if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost energy [W] at t due to ionisation: ", sim%time, p_lost_ion_all
if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost energy [W] at t due to line radiation: ", sim%time, p_plt_lost_all
if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost energy [W] at t due to CX radiation: ", sim%time, p_cx_lost_all

! if (sim%my_id .eq. 0) write(*,*) " Lost energy [J] at t due to line radiation: ", sim%time, p_plt_lost_all
!$ w1 = omp_get_wtime()
!$ mmm = mpi_minmeanmax(w1-w0)
!$ if (sim%my_id .eq. 0) write(*,"(f10.7,A,3f9.4,A)") sim%time, " Particle stepping complete in ", mmm, "s"


!  write(*,*) 'CAREFUL: averaging over n_steps : ',n_steps
!  jorek_feedback%rhs = jorek_feedback%rhs / real(n_steps,8)
if (sim%my_id .eq. 0) write(*,*) 'done loop_particle_kinetic_local'

end subroutine

!================================================================================
!                                 RECOMBINATION
!================================================================================
subroutine do_1particle_recombination_3D(element_list,node_list,jorek_stepper,rng)
  use mod_jorek_timestepping !< gives us access to sim?
  ! use mod_ionisation_recombination, only : rec_rate_local, rec_rate_global, rec_mom_local,rec_energy_local, rec_v_R, rec_v_Z, rec_v_phi
  use particle_tracer
  !use mod_particle_diagnostics
  use mpi
  use mod_atomic_elements
  use mod_particle_io
  use mod_integrate_recomb_3D, only : integrate_recombination
  !mod_integrate_recomb.f90
  use mod_parameters, only : n_period, n_plane
  
  implicit none
  
  !class(particle_sim), target, intent(inout)                :: sim
  !type(pcg32_rng), dimension(:), allocatable      :: rng
  type(pcg32_rng), dimension(:), intent(inout)    :: rng
  type(jorek_timestep_action),target           :: jorek_stepper !target
  TYPE (type_node_list),         intent(in)     :: node_list
  TYPE (type_element_list),      intent(in)     :: element_list
    
  !internal variables
  type (type_element)               :: element
  logical, allocatable, dimension(:) :: is_free
  integer, allocatable, dimension(:) :: i_free
  integer             :: Nrec_part, particles_per_element
  real*8              :: total_rec,total_rec_all ,total_volume,total_volume_all
  real*8, dimension(n_plane) :: total_rec_nplane, total_volume_nplane, total_rec_nplane_all, total_volume_nplane_all
  integer             :: n_free,i, k,ielm,ife, i_rng, mp!, element_loc
  real*8              :: s, t,R, Z, phi_plane, delta_phi, st_ran(3)
  
  !rec variables
  real*8, dimension(:,:), allocatable  :: rec_rate_local , rec_v_R, rec_v_Z, rec_v_phi 
  real*8, dimension(:,:), allocatable  :: volume_check  
  
  !Call mod_integrate_recombination
  call integrate_recombination(sim%my_id,sim%n_cpu, rec_rate_local, rec_v_R, rec_v_Z, rec_v_phi,volume_check)
  
  
  total_volume = sum(sum( volume_check, DIM = 1 ), DIM=1)
  total_rec = sum(sum( rec_rate_local, DIM = 1 ), DIM=1)
  total_rec_nplane = sum( rec_rate_local, DIM = 1 )
  total_volume_nplane = sum( volume_check, DIM = 1 )
  ! total recombination
  call MPI_REDUCE(total_rec, total_rec_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(total_volume, total_volume_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(total_rec_nplane, total_rec_nplane_all, n_plane, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(total_volume_nplane, total_volume_nplane_all, n_plane, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  if (sim%my_id .eq. 0) then
      write(*,'(A30,E14.6)') 'total recombination weight : ' , total_rec_all* central_density* 1.d20 
    write(*,*) 'total volume : ' , total_volume_all
      write(*,*) 'total recombination weight per plane : ' , total_rec_nplane_all* central_density* 1.d20 
    write(*,*) 'total volume per plane: ' , total_volume_nplane_all	
  endif
  !Nrec_part amount of particles needed for this amount of recombination
  Nrec_part = int( max(n_particles * 1.d-2 ,total_rec/1.d14 ) )!< assumed average weight per particle (not necesarily the actual weight, as that depends on Srec)
  !< limited to 1% of the total initialized particles
  
  
  !============== Finding free particles !< make into a function?
  !> # is_free > n_elements * particles_per_element 
  allocate(is_free(size(sim%groups(1)%particles,1))) 
  !$omp parallel do default(none) shared(sim, n_free, i_free, is_free) &
  !$omp private(j) schedule(dynamic, 100)
  do j=1,size(sim%groups(1)%particles,1) !sim%groups(1)%particles
    is_free(j) = sim%groups(1)%particles(j)%i_elm .le. 0  !< array T/F is particle is free
  end do
  !$omp end parallel do
  !$omp barrier
  n_free = count(is_free)
  allocate(i_free(n_free))
  k = 1
  do j=1,size(is_free,1)
    if (is_free(j)) then
      i_free(k) = j !< i_free(k) has index of free particle in  sim%groups(1)%particles(j)
      k = k+1
      !if (sim%my_id .eq. 0) write(*,*) "Adding to the list number: ", j
    end if
  end do
  ! ==================
  
  !calc phi
  ! phi_plane     = 2.d0 * PI / real(n_period,8) / real(n_plane,8) * i_plane
  delta_phi     = 2.d0 * PI / real(n_plane,8) / real(n_period,8)
  
  ! loop over all elements
  k = 0 !< first free particle
  particles_per_element = 1	
  !write(*,*) "Doing 1 particle recombination over total n_elemnts", element_list%n_elements
  !write(*,*) "Doing 1 particle recombination over n_local_elms", jorek_stepper%n_local_elms
  !write(*,*) "Size rec_rate_local", SHAPE(rec_rate_local)
  select type (particles => sim%groups(1)%particles)
  type is (particle_kinetic_leapfrog)
  !omp
#ifdef __GFORTRAN__
  !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
  !$omp parallel do default(shared) &
  !$omp shared(sim, particles,jorek_stepper, element_list, node_list, rec_v_R,rec_v_Z,rec_v_phi, &
  !$omp i_free,rng,rec_rate_local,delta_phi, &
  !$omp CENTRAL_DENSITY, CENTRAL_MASS,sqrt_mu0_over_rho0,particles_per_element) &
#endif
  !$omp schedule(dynamic,10)      &
  !$omp private(ife,ielm,k,i,element,s,t,R, Z ,mp, &
  !$omp st_ran, i_rng,phi_plane )
  do ife = 1, size(rec_rate_local,1) ! loop over all local elements
      !write(*,*) "ife", ife
      ! --- Get element
    !ielm = jorek_stepper%local_elms(ife) !< actual element number
    ielm    = (sim%my_id+1) + sim%n_cpu*(ife - 1)
    element = element_list%element(ielm)
  
    
    !if (rec_rate_local(ife) .le. 1.d3) CYCLE
    
    
    do mp = 1, n_plane
      if (isnan(rec_v_R(ife,mp)) .or. isnan(rec_v_Z(ife,mp)) .or. isnan(rec_v_phi(ife,mp))) CYCLE !NaN check
      phi_plane     = delta_phi * (mp-1)
      
      !$ i_rng = omp_get_thread_num()+1
      !if (rec_rate_local(ife) / real(particles_per_element)* central_density* 1.d20 .le. 1.d7) cycle
       
      k = ife +(mp-1)* size(rec_rate_local,1)!< every OMP thread gets different values
      !< every MPI process has it's own list of i_free.
       
  
      
      ! initialise particle in the element with Position, Weight, Energy, Momentum			
      do i = 1, particles_per_element
        k = k *i !< update free particle index ! at begin of loop as k is initialized at k =0
        particles(i_free(k))%weight = rec_rate_local(ife,mp) / real(particles_per_element)* central_density* 1.d20 !< rec_rate = in jorek units?
        particles(i_free(k))%i_elm  = ielm  !x, i_elm, st
        particles(i_free(k))%q      = 0
        
        !write(31,*) "ielm,",ielm, "k,",k,"i_free(k)",i_free(k) , "particles(i_free(k))%weight,",particles(i_free(k))%weight
        
        !call rng(1)%next(st_ran) !< i_rng should be thread dependent
        call rng(i_rng)%next(st_ran)
        ! sample random st combination
        !particles(i_free(k))%st(1:2) = st_ran(2)! [s, t] !< dummi for later
        particles(i_free(k))%st(1) = 0.5d0
        particles(i_free(k))%st(2) = 0.5d0
        
        
        s = particles(i_free(k))%st(1)
        t = particles(i_free(k))%st(2)
        
        !> uses i_elm and s,t to give us R,Z
        call interp_RZ(node_list,element_list,ielm,s,t,R,Z)
        particles(i_free(k))%x(1:2)  = [R, Z]!  = [R, Z, phi] no phi for axisymmetrix particles
        particles(i_free(k))%x(3)    = phi_plane + delta_phi*(st_ran(3)-0.5d0)
        
        !> distribute directly fluid velocity?
        particles(i_free(k))%v(1)  = rec_v_R(ife,mp)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0 !m/s
        particles(i_free(k))%v(2)  = rec_v_Z(ife,mp)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
        particles(i_free(k))%v(3)  = rec_v_phi(ife,mp) / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
        !< v = momentum fluid lost to recombination / (mass of superparticle)
      end do ! parts_per_element
  
    end do !mp = 1, n_plane
  enddo   !ife 
  !$omp end parallel do
  end select
  !end omp
  
  
      
  !!!!!--------------------------------------------------------------------------
  end subroutine !do_1particle_recombination_3D

function initialise_sputtering(node_list, element_list, n_reflect) result(D_sputter_source)

  use mod_edge_domain
  use mod_edge_elements

  type(type_node_list), intent(in)    :: node_list
  type(type_element_list)             :: element_list
  type(particle_sputter)              :: D_sputter_source
  integer                             :: n_reflect
  !real*8, allocatable, dimension(:)   :: wall_albedo
  type(type_edge_domain), allocatable, dimension(:) :: edge_domains

  ! number of particles to sputter per species (should be renormalized to yield)

  call find_edge_domains(node_list,element_list, edge_domains)!, discont_corner=.true.)
  if (sim%my_id .eq. 0) write(*,*) "n_domains = ", size(edge_domains,1)
  ! allocate(wall_albedo(size(edge_domains,1)))
  ! wall_albedo(:) = 1.d0 !0.9d0
  ! wall_albedo(size(edge_domains,1)) = 0.1d0
  
  call D_edge%prepare(node_list, element_list, edge_domains, nsub=6, nsub_toroidal=1)!,wall_albedo=wall_albedo)

  ! target group, number of particles per mpi task, densities, Zs, basename
  D_sputter_source = particle_sputter(D_edge, 1, n_reflect, basename='D_reflect')
  D_sputter_source%use_Yn_func = .false.
  D_sputter_source%n_save = nout !10 ! or nout
  !

end function

pure function f_adapted(n, P, grad_P) result(f)
  integer, intent(in) :: n
  real*8, intent(in) :: P(n), grad_P(3,n)
  real*8 ::s, coeff(0:4)
  real*4 :: f

  coeff(0)=0.53
  coeff(1)=0.3
  coeff(2)=0.2
  coeff(3)=0.52
  coeff(4)=0.26

  s = max((P(1) - ES%Psi_axis) / ( ES%Psi_bnd - ES%Psi_axis),0.d0)

  f = coeff(3)*exp(-coeff(2)/coeff(1)*(tanh((sqrt(s)-coeff(0))/coeff(2))))

  f = (f - coeff(4)) / (1.d0 - coeff(4))

end function f_adapted

pure function f_original(n, P, grad_P) result(f)
  integer, intent(in) :: n
  real*8, intent(in) :: P(n), grad_P(3,n)
  real*8 ::s, coeff(0:3)
  real*4 :: f

  ! central densiy should be 1.44131x10^17

  coeff(0)=0.49123
  coeff(1)=0.298228
  coeff(2)=0.198739
  coeff(3)=0.521298

  s = max((P(1) - ES%Psi_axis) / ( ES%Psi_bnd - ES%Psi_axis),0.d0)

  f = coeff(3)*exp(-coeff(2)/coeff(1)*(tanh((sqrt(s)-coeff(0))/coeff(2))))

end function f_original

pure function f_toroidal_flux(n, P, grad_P) result(f)
  integer, intent(in) :: n
  real*8, intent(in) :: P(n), grad_P(3,n)
  real*8 :: s, psi_norm, coeff(0:3)
  real*4 :: f

  ! central densiy should be 1.44131x10^17

  coeff(0)=0.49123
  coeff(1)=0.298228
  coeff(2)=0.198739
  coeff(3)=0.521298

  psi_norm = max((P(1) - ES%Psi_axis) / ( ES%Psi_bnd - ES%Psi_axis),0.d0)

  s = 0.957 * psi_norm + 0.043 * psi_norm**2 

  f = coeff(3)*exp(-coeff(2)/coeff(1)*(tanh((sqrt(s)-coeff(0))/coeff(2))))

end function f_toroidal_flux


end program recobmination_loop
