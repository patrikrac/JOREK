!> Standard neutral atomic particle example in 2D
!>
!> The physics model includes puffing, recombination, ionisation, recycling and charge exchange for atomic Deuterium (no molecules).
!> OpenADAS is used for atomic physics
!> Plasma and neutral wall interaction are based on SDTRIM coefficients. 
!> External files y_DD.dat and ye_DD.dat are used to determine wall recombination of plasma into atomic neutral deuterium. 
!> These are based on interaction with a W wall
!>
!> To adjust the puff to your scenario, see  "! --- Setting up puffing" below
!>
!> To use a particle restart file: use restart_particles=.t. in the input file.
!>
!> To change to 3D recombination:
!> Change subroutine do1particlerecombination: use mod_integrate_recomb3D, only : integrate_recombination
!> Particle puffing is axisymmetric by default.

program kinetic_neutral_loop

use particle_tracer
use mod_particle_diagnostics
use mpi
use mod_interp
use mod_atomic_elements
use mod_particle_io
use mod_event
use mod_project_particles
use mod_jorek_timestepping
use mod_random_seed
use mod_basisfunctions
use nodes_elements
use constants,   only: MU_ZERO, ATOMIC_MASS_UNIT, K_BOLTZ, EL_CHG
use mod_particle_sputtering, only: particle_sputter, sample_fluid_particle_energy
use mod_projection_functions, only: proj_f_combined_density, proj_f_combined_energy, proj_f_combined_par_momentum
use mod_particle_puffing
use mod_edge_domain
use mod_edge_elements
use mod_atomic_coeff_deuterium, only: ad_deuterium 
use data_structure, only: type_bnd_element_list, type_bnd_node_list 
use mod_boundary,   only: boundary_from_grid
use equil_info

use phys_module, only: tstep,tstep_n,restart_particles, restart, t_start, nout
use phys_module, only: CENTRAL_MASS, CENTRAL_DENSITY, xcase, xpoint
use phys_module, only: n_particles, nstep_particles, nsubstep_particles, tstep_particles
use phys_module, only: use_ncs, use_pcs, use_ccs, deuterium_adas,sqrt_mu0_over_rho0
use phys_module, only: filter_perp, filter_hyper, filter_par, filter_perp_n0, filter_hyper_n0, filter_par_n0
use phys_module, only: use_kn_sputtering, use_kn_cx, use_kn_ionisation, use_kn_line_radiation, use_kn_recombination, use_kn_puffing
use phys_module, only: puff_rate, r_valve, R_valve_loc, Z_valve, R_valve_loc2, Z_valve2, n_puff
use phys_module, only: use_manual_random_seed, manual_seed

!$ use omp_lib

implicit none

type(event)                                       :: fieldreader, partreader
type(event)                                       :: D_sputter_event, gas_puff_event, gas_puff2_event
type(event), target                               :: project_jorek_feedback, jorek_stepper_event
type(pcg32_rng), dimension(:), allocatable        :: rng
type(count_action)                                :: counter
type(projection), target                          :: jorek_feedback
type(jorek_timestep_action), target               :: jorek_stepper
type(particle_sputter)                            :: D_sputter_source
type(type_edge_domain), allocatable, dimension(:) :: edge_domains
type(edge_elements)                               :: D_edge
type(particle_puffing)                            :: gas_puff, gas_puff2

real*8    :: rho_norm, t_norm, n_norm, tstep_fluid_si 
real*8    :: tstep_part_adj !< tstep_particles adjusted so that an integer amount of steps (nstep_particles) fit into a fluid step (tstep)
!$ real*8 :: w0, w1, mmm(3)

integer   :: n_reflect
integer   :: i, j, istep
integer   :: seed, i_rng, n_stream

! Puffing parameters
real*8  :: t_puff_start          !< [s] time to start ramping the puff rate if puff_t_dependent=.true.
real*8  :: t_puff_slope          !< [s] time over which the puff rated is ramped to puff_rate (input) from t_puff_start where the puff rate was still puffing_rate_start
real*8  :: puffing_rate_start    !< [atoms/s] initial puff rate before the ramp if puff_t_dependent=.true.
real*8  :: poly_R(4)             !< [m] R coordinates of the quadrangular puffing valve if boxpuff=.true.
real*8  :: poly_Z(4)             !< [m] Z coordinates of the quadrangular puffing valve if boxpuff=.true.
real*8  :: poly_R2(4),poly_Z2(4) !  [m] second puffing valve location
logical :: puff_t_dependent      !< puff time dependent using a flat - ramp - flat pattern (=.true.) or no time dependence at all (.false.) 
logical :: boxpuff               !< whether to puff in a simple (=.false., uses r_valve etc) or quadrangular (=.true., uses pol_R, poly_Z) puff valve

!***********************************************************************
!*                            intialisation                            *
!***********************************************************************

! Start up MPI, jorek
call sim%initialize(num_groups=1)

! Set up the field reader < can this be moved to sim%initialize
fieldreader = event(read_jorek_fields_interp_linear(basename='jorek', i=-1))
call with(sim, fieldreader)
tstep = tstep_n(1) !< the field reader overwrites tstep for some reason, this resets that

! setting up the particles
if (restart_particles) then
  ! reading the particles from a file
  if (sim%my_id == 0) write(*,*) 'INFO: READING PARTICLES RESTART FILE'
  partreader = event(read_action(filename='part_restart.h5'))
  call with(sim, partreader) !<defines sim%groups and the corresponding particles

  !TODO? Sven: We should make an option to use partreader but increase n_particles; may be similar to phi_zero_whrite to a sim_in and sim_out but with different allocation size.
else
  ! setting up empty particle array
  if (sim%my_id == 0) write(*,*) 'INFO: INITIALIZING PARTICLES', sim%n_cpu, " cpus "

  !> is this needed for neutrals?
  if (sim%my_id .eq. 0) call boundary_from_grid(sim%fields%node_list, sim%fields%element_list, bnd_node_list, bnd_elm_list, .false.)
  call broadcast_boundary(sim%my_id, bnd_elm_list, bnd_node_list)

  call update_equil_state(sim%my_id, sim%fields%node_list, sim%fields%element_list, bnd_elm_list, xpoint, xcase )

  ! Set up particle group characteristics
  sim%groups(1)%Z    = -2 !< for deuterium 1
  sim%groups(1)%mass = atomic_weights(-2) !< atomic mass units
  sim%groups(1)%ad   = read_adf11(sim%my_id,'12_h')
  
  ! setting up particles per MPI node
  allocate(particle_kinetic_leapfrog::sim%groups(1)%particles( ceiling(n_particles/sim%n_cpu) ))
  
  ! setting up empty particle array
  select type (p => sim%groups(1)%particles)
  type is (particle_kinetic_leapfrog)  
    p(:)%q      = 0 !< for neutrals
    p(:)%weight = 0.0!weight
    p(:)%i_elm  = 0
    p(:)%v(1)   = 0.d0 
    p(:)%v(2)   = 0.d0
    p(:)%v(3)   = 0.d0
  end select
endif ! (restart_particles)

! Read Open ADAS data for plasma fluid
if (deuterium_adas .and. use_kn_recombination) ad_deuterium =  read_adf11(sim%my_id,'96_h') !< move to core (jorek2_main for particles)

! --- Setting up random numbers for ionisation probability
seed = random_seed()
n_stream = 1
!$ n_stream = omp_get_max_threads()
write(*,*) "id, n_cpu, n_stream",sim%my_id, sim%n_cpu, n_stream
allocate(rng(n_stream))
do i=1,n_stream
  call rng(i)%initialize(1, seed, n_stream, i)
end do

! --- Check if the user tried to use nstep_particles rather than tstep_particles to define the particle timestepping
if (nstep_particles .ne. 0 .and. sim%my_id .eq. 0) then
  write(*,*) "ERROR: nstep_particles is defined in the input file, while for this example the combination tstep_particles, nstep and tstep define nstep_particles. Please remove nstep_particles from your input file to avoid ambiguity. Stopping now."
  stop
end if

!***********************************************************************
!*                         setting up physics                          *
!***********************************************************************

! --- Calculating normalisation constants
n_norm    = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
rho_norm  = central_mass * ATOMIC_MASS_UNIT * n_norm                  ! rho_SI = rho_norm * rho
t_norm    = sqrt((MU_ZERO * rho_norm))                           ! t_SI   = t_norm * t_jorek 

! --- Setting up sputtering
if (use_kn_sputtering) then  
  n_reflect = ceiling(n_particles * 5.d-4) !< should be a group dependent input parameter with this as default value
  D_sputter_source = initialise_sputtering(sim%fields%node_list, sim%fields%element_list, n_reflect)
  D_sputter_event = event(D_sputter_source)
endif

! --- Setting up puffing

!> Adapt the following to customize the time dependent puff rate:
!> puffing_rate_start = initial puffing rate [atoms/s]
!> puff_rate = final puffing rate [atoms/s] <input parameter>
!> t_puff_start = At what time the puffing rate starts to increase [s]
!> t_puff_slope = How much time it takes to increase linearly from puffing_rate_start to puff_rate [s]
!> n_puff = number of super particles puffed per valve per jorek timestep (should be small fraction of total number of super particles) <input parameter>
puff_t_dependent = .true. 
puffing_rate_start = puff_rate/1.5d0 !< initial puffing rate [atoms/s]

!> puff location can be determined for a circular valve by setting input parameters: 
!> r_valve (valve radius), R_valve_loc, Z_valve (R,Z, coordinates of simple valve)
!> if boxpuff=.true., poly_R(4),poly_Z(4) are the vertices of the quadrangular puffing valve
boxpuff = .true. !< whether to puff using 

!> puff location for simple xpoint case
if(sim%my_id .eq. 0) write(*,*) "puff location for xpoint reg_test example"
poly_R  = (/3.86d0, 3.9d0, 3.86d0, 3.9d0/)
poly_Z  = (/0.1d0,  0.1d0,  0.0d0, 0.0d0/)
poly_R2 = poly_R 
poly_Z2 = poly_Z

if (use_kn_puffing) then
  t_puff_start = 5000*t_norm !< start puffing after this amount of seconds, t_SI = t_jorek*t_norm jorek time units
  t_puff_slope = 4.d-3       !< [s] linearly ramps up the puffing during this time

  gas_puff = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc, Z_valve, puff_t_dependent=puff_t_dependent,t_puff_start=t_puff_start,t_puff_slope=t_puff_slope, & 
      puffing_rate_start=puffing_rate_start/2.d0,poly_R=poly_R,poly_Z=poly_Z,boxpuff=boxpuff)
  gas_puff2 = particle_puffing(n_puff, puff_rate/2.d0, r_valve, R_valve_loc2, Z_valve2, puff_t_dependent=puff_t_dependent,t_puff_start=t_puff_start,t_puff_slope=t_puff_slope, &
      puffing_rate_start=puffing_rate_start/2.d0,poly_R=poly_R2,poly_Z=poly_Z2,boxpuff=boxpuff)
  
  gas_puff_event  = event(gas_puff)
  gas_puff2_event = event(gas_puff2)
  
  if (sim%my_id .eq.0) then
    write(*,*) "Gas puffing rate [#/s] : ", puff_rate
    write(*,*) "puff_t_dependent : ",puff_t_dependent, "with puff slope",t_puff_slope,"starting at", t_puff_start, "s"
  endif
else 
  gas_puff = particle_puffing(0, 5d20, r_valve, R_valve_loc, Z_valve)
  gas_puff2 = particle_puffing(0, 5d20, r_valve, R_valve_loc, Z_valve)
endif

! --- Set up feedback to the plasma (does not currently include recombination)
jorek_feedback = new_projection(sim%fields%node_list, sim%fields%element_list, &
                                filter_n0 = filter_perp_n0, filter_hyper_n0 = filter_hyper_n0, filter_parallel_n0=filter_par_n0,      &
                                filter = filter_perp, filter_hyper = filter_hyper, filter_parallel=filter_par, fractional_digits = 9, &
                                do_zonal = .false., calc_integrals=.false., to_vtk=.TRUE., to_h5 = .false., basename='projections', nsub=2)
aux_node_list => jorek_feedback%node_list

! Allocate feedback according to coupling scheme
if (use_ncs) then
  allocate(jorek_feedback%rhs(n_order+1, n_vertex_max, sim%fields%element_list%n_elements, n_tor, 4)) !< stacksize should be big enough
elseif (use_pcs) then
  stop 'pcs not implemented in this example'
elseif (use_ccs) then
  stop 'ccs not implemented in this example'
else
  stop 'define use_ncs, use_pcs or use_ccs'
endif
jorek_feedback%rhs = 0.d0

! --- Setting up jorek timestepper
! For proper timestepping, the projections need to be defined before the jorek timestepper
jorek_stepper = new_jorek_timestep_action(jorek_feedback%node_list)

project_jorek_feedback = new_event_ptr(jorek_feedback,   start = sim%time)
jorek_stepper_event    = new_event_ptr(jorek_stepper,    start = sim%time)


!***********************************************************************
!*                           main loop                                 *
!***********************************************************************

istep = 0
do while (.not. sim%stop_now)
  istep = istep + 1
  if(sim%my_id .eq. 0) write(*,'(A100)'  ) "===================================================================================================="
  if(sim%my_id .eq. 0) write(*,'(A37,I6)') "Starting main loop iteration istep = ",istep
  if(sim%my_id .eq. 0) write(*,'(A100)'  ) "===================================================================================================="

  ! --- Determining the time stepping for this fluid step
  tstep = get_tstep_n(istep) ! tstep is also set in stepper, but tstep is already used in the calls before the stepper
  tstep_fluid_si = tstep*t_norm
  sim%time = sim%time + tstep_fluid_si ! carries the time at the end of the current step

  nstep_particles = ceiling(tstep_fluid_si / tstep_particles) ! ceiling makes sure tstep_part_adj is never bigger than tstep_particles
  tstep_part_adj = tstep_fluid_si / nstep_particles ! slightly smaller tstep_particles to fit an exact integer amount in one fluid timestep
  
  if (sim%my_id .eq. 0) then
     write(*,*) "PARTICLE : tstep_particles : ",tstep_particles
     write(*,*) "PARTICLE : tstep_part_adj  : ",tstep_part_adj
     write(*,*) "PARTICLE : sim%time        : ",sim%time
     write(*,*) "PARTICLE : nstep_particles : ",nstep_particles
     write(*,*) "PARTICLE : tstep_fluid_si  : ",tstep_fluid_si
     write(*,*) "PARTICLE : n*dt_part - dt  : ",nstep_particles*tstep_part_adj - tstep_fluid_si
  endif


  ! --- Interactions that happen on the fluid timestep (creating kinetic particles)
  
  !> The sputtering modules actually contains 3 different effects: sputtering (plasma to W, which is not used in this example), 
  !> kinetic particle reflection off the wall, and wall recombination of the plasma into kinetic neutrals (i.e. recycling)
  !> Do this call before recombination and puffing. Otherwise to-be-reflected particles can be overwritten.
  if (use_kn_sputtering) then
    call write_to_outputfile(sim%my_id, "Sputtering")
    call with(sim, D_sputter_event)
  endif   !use_kn_sputtering
  
  if (use_kn_recombination) then
    call write_to_outputfile(sim%my_id, "Recombination")
    call do_1particle_recombination(element_list,node_list,jorek_stepper,rng, tstep_fluid_si) 
  endif !use_kn_recombination
    
  if (use_kn_puffing) then
    call write_to_outputfile(sim%my_id, "Puffing")
    call with(sim, gas_puff_event) 
    call with(sim, gas_puff2_event)
  endif ! use_kn_puffing  


  ! --- Interactions that happen on the particle timesteps
  
  !> ionisation + CX + pushing the particles + calculating the feedback
  call write_to_outputfile(sim%my_id, "Particle loop")
  call loop_particle_kinetic_local(sim, jorek_feedback, rng, tstep_part_adj)
  

  ! --- Update the fluid
  
  !> Project the collected feedback from the particles onto the finite element grid so that the MHD solver can use it
  !> Also writes the projection.vtk file which contains the interaction terms (particle, energy and momentum exchange to the fluid) and neutral density
  call write_to_outputfile(sim%my_id, "Projecting feedback from particles to fluid FE grid")
  call with(sim, project_jorek_feedback)
  
  !> Calls the MHD solver which timesteps the MHD fluid based on the fluid itself using the projected
  !> feedback of the particles as sources and sinks in the MHD equations 
  !> Also writes .h5 file, updates tstep and sets sim%stop_now = .true. if all fluid steps are done
  call write_to_outputfile(sim%my_id, "Fluid stepper")
  call with(sim, jorek_stepper_event) 
  

  ! -- Finalising the fluid timestep
  
  !Writing interim particle restart files every 500 fluid steps done. Overwrites previous restart file to save space
  if ( mod(istep,500) .eq. 0 ) then
    call write_to_outputfile(sim%my_id, "Writing interim_part_restart.h5")
    call write_simulation_hdf5(sim, 'interim_part_restart.h5')
  endif

  ! Writing some conservation checks to the ouput file
  call write_to_outputfile(sim%my_id, "Conservation checks")
  call conservation_checks(sim)

end do ! while

!***********************************************************************
!*                          end of simulation                          *
!***********************************************************************
call write_to_outputfile(sim%my_id, "End of simulation")
  
call write_simulation_hdf5(sim, 'part_restart.h5')

call sim%finalize

!***********************************************************************
!*                          end of main program                        *
!***********************************************************************

contains

!> Evolves the particles with every interaction that only depends on the particles except for wall reflections, 
!> i.e. ionisation + CX + pushing the particles + calculating the feedback
subroutine loop_particle_kinetic_local(sim, jorek_feedback, rng, tstep_part_adj)
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

  real*8, intent(in)     :: tstep_part_adj

  real*8, parameter  :: binding_energy = 2.18d-18 ! ionization energy of a hydrogen atom [J] (= 13.6 eV)
  real*8    :: n_norm, rho_norm, t_norm, v_norm, E_norm, M_norm
  real*8    :: t, E(3), B(3), psi, U, n_e, T_e, rz_old(2), st_old(2)
  real*8    :: R_g, Z_g, R_s, R_t, Z_s, Z_t, xjac, HZ(n_tor), HH(4,4), HH_s(4,4), HH_t(4,4)
  real*8    :: ion_rate, ion_source, ion_prob, ion_ran(1), cx_ran(8),st_ran(2), cx_source, cx_energy ,PLT
  real*8    :: cx_prob, CX_rate
  real*8    :: kinetic_energy, ion_energy,line_rad_energy
  real*8    :: n_lost_ion, n_lost_ion_all, p_plt_lost,p_plt_lost_all,p_cx_lost,p_cx_lost_all,p_lost_ion,p_lost_ion_all
  integer   :: n_super_ionized, n_super_ionized_all
  real*8    :: particle_source, velocity_par_source, energy_source
  real*8    :: v_temp(3), K_eV, v_kin_temp, B_norm(3), v, v_v, v_E,extra_proj
  real*8    :: vvector(3),sum_ran(3), E_th, v_th,ran_norm(4)
  !$ real*8 :: w0, w1, mmm(3)

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
    
    
  jorek_feedback%rhs_gather_time = jorek_feedback%rhs_gather_time + nstep_particles * tstep_part_adj

  allocate(feedback_rhs,source=jorek_feedback%rhs)

  jorek_feedback%rhs = 0.d0
  feedback_rhs       = 0.d0

  call with(sim, counter)

  select type (particles => sim%groups(1)%particles)
  type is (particle_kinetic_leapfrog)
  if(use_manual_random_seed) then
    !$ call omp_set_schedule(omp_sched_static,10)
  else
    !$ call omp_set_schedule(omp_sched_dynamic,10)
  end if
#ifdef __GFORTRAN__
  !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
  !$omp parallel do default(none) &
  !$omp shared(sim, particles, nstep_particles, tstep_part_adj, rng,                                &
  !$omp rho_norm, t_norm, v_norm, E_norm, M_norm, N_norm,                                         &
  !$omp use_kn_cx, use_kn_ionisation,use_kn_line_radiation,                                       &
  !$omp CENTRAL_DENSITY, CENTRAL_MASS)                                                            &
#endif
  !$omp schedule(runtime)                                                                         &
  !$omp private(particle_tmp, i_rng, i,j,k,l,m, t, E, B, psi, U, rz_old, st_old,                  &
  !$omp i_elm_old, i_elm, n_e, T_e,                                                               &
  !$omp PLT,ion_rate, ion_prob, ion_ran, ion_source, ion_energy, kinetic_energy, line_rad_energy, &
  !$omp R_g, R_s, R_t, Z_g, Z_s, Z_t, xjac, HH, HH_s, HH_t, HZ, index_lm, ifail,limits,           &
  !$omp CX_rate, CX_prob, CX_source, CX_energy, v, v_E, v_v,extra_proj,                           &
  !$omp particle_source, velocity_par_source, energy_source, v_temp, K_eV, cx_ran,                &
  !$omp E_th, v_th,sum_ran,vvector,ran_norm)                                                      &
  !$omp reduction(+:feedback_rhs,n_lost_ion,p_plt_lost,p_cx_lost,p_lost_ion,n_super_ionized)
  
  do j=1,size(particles,1)

    call copy_particle_kinetic_leapfrog(particles(j),particle_tmp)
    !$ i_rng = omp_get_thread_num()+1
    
    do k=1,nstep_particles

      if (particle_tmp%i_elm .le. 0) exit

      t = sim%time + (k-1)*tstep_part_adj

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
        line_rad_energy = n_e * particle_tmp%weight * PLT * tstep_part_adj
      endif ! use_kn_line_radiation
      
      if (use_kn_ionisation .and. .not. limits) then
        
        call sim%groups(1)%ad%SCD%interp(int(particle_tmp%q), log10(n_e), log10(T_e), ion_rate) ! [m^3/s]
        ion_prob = 1.d0 - exp(-ion_rate * n_e * tstep_part_adj) ! [0] poisson point process, exponential 

        ! If the weight is too small throw away the particle with the probability, else reduce weight with ionising probability
        ion_source = 0.d0

        if (particle_tmp%weight .le. 1.0d9) then
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
        CX_prob = 1.d0 - exp(-CX_rate * n_e * tstep_part_adj)

        call rng(i_rng)%next(cx_ran)
        if (cx_ran(1) .le. CX_prob) then
          ! sample boltzman, randomize velocity

          !============== NEW CX PARTICLE
          !Box-Mueller sample velocities with st.dev=1
          ran_norm = boxmueller_transform(cx_ran(2:5))
          v_temp = sqrt(T_e * K_BOLTZ/(sim%groups(1)%mass * ATOMIC_MASS_UNIT))*ran_norm(2:4)

          !>add bulk fluid flow
          v_temp = v_temp + vvector 

          CX_source = particle_tmp%weight
          CX_energy   = 0.5d0 * sim%groups(1)%mass * ATOMIC_MASS_UNIT *  (dot_product(particle_tmp%v,particle_tmp%v) - dot_product(v_temp,v_temp))
        

          endif ! cx_ran
      endif ! use_kn_cx
      
      if (isnan(ion_source * ion_energy + cx_source * cx_energy - line_rad_energy)) then
        write(*,*) "ion_energy", ion_energy
        write(*,*) "cx_energy", cx_energy
        write(*,*) "line_rad_energy", line_rad_energy
        particle_tmp%i_elm  = 0
        cycle !< don't feed this particle into the feedback
      endif
      
      
      ! feedback from each particle at each timestep
      energy_source       = ion_source * ion_energy + cx_source * cx_energy - line_rad_energy
      particle_source     = ion_source * sim%groups(1)%mass * ATOMIC_MASS_UNIT !< mass source in SI
      velocity_par_source = ion_source * dot_product(B, particle_tmp%v) * sim%groups(1)%mass * ATOMIC_MASS_UNIT &
            + CX_source  * dot_product(B, particle_tmp%v - v_temp) * sim%groups(1)%mass * ATOMIC_MASS_UNIT 
          
      particle_tmp%v = v_temp 
      n_lost_ion = n_lost_ion + ion_source  !< local sum #particles lost due to ionisation
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
          extra_proj = HH(l,m) * sim%fields%element_list%element(i_elm_old)%size(l,m) *particle_tmp%weight * 1.d0/real(nstep_particles,8) !<average density over jorek tstep_part_adj!real(floor(k/nstep_particles))!1.d0 !<density proj

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
        call boris_push_cylindrical(particle_tmp, sim%groups(1)%mass, E, B, tstep_part_adj)

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
  if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') " Ionization rate at time t [#/s]: ", sim%time, n_lost_ion_all / (tstep_part_adj * nstep_particles)
  p_lost_ion_all = p_lost_ion_all / (tstep_part_adj * nstep_particles)
  p_plt_lost_all = p_plt_lost_all / (tstep_part_adj * nstep_particles)
  p_cx_lost_all = p_cx_lost_all / (tstep_part_adj * nstep_particles)
  if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Energy exchange to plasma [W] at t due to ionisation: ", sim%time, p_lost_ion_all ! energy gain
  if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Lost energy [W] at t due to line radiation: ", sim%time, p_plt_lost_all
  if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Energy exchange to plasma [W] at t due to CX radiation: ", sim%time, p_cx_lost_all
  if (sim%my_id .eq. 0) write(*,'(A17,5E14.6)') 'TOTAL Exchange , delta t: ' ,sim%time,p_lost_ion_all, -p_plt_lost_all, p_cx_lost_all, tstep_part_adj * nstep_particles
  if (sim%my_id .eq. 0) write(*,'(A46,2E14.6)') "Total energy exchange to plasma [W]: ", sim%time, p_lost_ion_all -p_plt_lost_all+ p_cx_lost_all

  ! if (sim%my_id .eq. 0) write(*,*) " Lost energy [J] at t due to line radiation: ", sim%time, p_plt_lost_all
  !$ w1 = omp_get_wtime()
  !$ mmm = mpi_minmeanmax(w1-w0)
  !$ if (sim%my_id .eq. 0) write(*,"(f10.7,A,3f9.4,A)") sim%time, " Particle stepping complete in ", mmm, "s"


  if (sim%my_id .eq. 0) write(*,*) 'done loop_particle_kinetic_local'

end subroutine

!================================================================================
!                                 RECOMBINATION
!================================================================================
subroutine do_1particle_recombination(element_list,node_list,jorek_stepper,rng,tstep_fluid_si)
  use mod_jorek_timestepping !< gives us access to sim?
  use particle_tracer
  use mpi
  use mod_atomic_elements
  use mod_particle_io
  use mod_integrate_recomb, only : integrate_recombination

  implicit none

  type(pcg32_rng), dimension(:), intent(inout)  :: rng
  type(jorek_timestep_action),target            :: jorek_stepper
  TYPE (type_node_list),         intent(in)     :: node_list
  TYPE (type_element_list),      intent(in)     :: element_list
  real*8, intent(in)                            :: tstep_fluid_si
    
  !internal variables
  type (type_element)               :: element
  logical, allocatable, dimension(:) :: is_free
  integer, allocatable, dimension(:) :: i_free
  integer             :: Nrec_part, particles_per_element
  real*8              :: total_rec,total_rec_all ,total_volume,total_volume_all
  real*8              :: total_Erec_neutral,total_Erec_neutral_all, total_Erec_rad,total_Erec_rad_all
  integer             :: n_free,i, k,ielm,ife, i_rng, ierr
  real*8              :: s, t,R, Z, st_ran(2)

  !debug rec
  real*8                              :: sanity_rec_local,total_sanity_rec
  !rec variables
  real*8, dimension(:), allocatable  :: rec_rate_local , rec_v_R, rec_v_Z, rec_v_phi 
  real*8, dimension(:), allocatable  :: volume_check, energy_neutrals, energy_radiation

  !Call mod_integrate_recombination
  call integrate_recombination(sim%my_id,sim%n_cpu, rec_rate_local, rec_v_R, rec_v_Z, rec_v_phi,volume_check, energy_neutrals, energy_radiation)

  sanity_rec_local = 0.d0
  !calculate total recombination per mpi proces
  total_volume = sum( volume_check(:) )
  total_Erec_neutral = sum( energy_neutrals(:) )
  total_Erec_rad = sum( energy_radiation(:) )
  total_rec = sum( rec_rate_local(:) )
  ! total recombination
  call MPI_REDUCE(total_rec, total_rec_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(total_volume, total_volume_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(total_Erec_neutral, total_Erec_neutral_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(total_Erec_rad, total_Erec_rad_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  if (sim%my_id .eq. 0) then
    write(*,'(A30,2E16.8)') 'total recombination weight : ' , sim%time,total_rec_all* central_density* 1.d20 
    write(*,*) 'total energy to recombined neutrals [J] : ' , total_Erec_neutral_all *1.5d0 / MU_ZERO
    write(*,*) 'total energy lost to Prb [J]: ' , total_Erec_rad_all *1.5d0 / MU_ZERO
    write(*,*) 'total volume : ' , total_volume_all
    write(*,'(A30,2E16.8)') 'Recombination rate  [#/s] : ', sim%time, total_rec_all* central_density* 1.d20 /tstep_fluid_si
    write(*,*) 'total power to recombined neutrals [MW]: ' , total_Erec_neutral_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
    write(*,*) 'total power lost to Prb [MW]: ' , total_Erec_rad_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
    write(*,'(A15,6E14.6)') 'TOTAL RECOMB: ',sim%time, total_rec_all* central_density* 1.d20 , total_Erec_neutral_all *1.5d0 / MU_ZERO, total_Erec_neutral_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6, &
                              total_Erec_rad_all *1.5d0 / MU_ZERO, total_Erec_rad_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
  endif
  !Nrec_part amount of particles needed for this amount of recombination
  Nrec_part = int( max(n_particles * 1.d-2 ,total_rec/1.d14 ) )!< assumed average weight per particle (not necesarily the actual weight, as that depends on Srec)
  !< limited to 1% of the total initialized particles


  !============== Finding free particles !< make into a function?
  !> # is_free > n_elements * particles_per_element 
  if(use_manual_random_seed) then
    !$ call omp_set_schedule(omp_sched_static,100)
  else
    !$ call omp_set_schedule(omp_sched_dynamic,100)
  end if

  allocate(is_free(size(sim%groups(1)%particles,1))) 
  !$omp parallel do default(none) shared(sim, n_free, i_free, is_free) &
  !$omp private(j) schedule(runtime)
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

  ! loop over all elements
  k = 0 !< first free particle
  particles_per_element = 1  
  select type (particles => sim%groups(1)%particles)
  type is (particle_kinetic_leapfrog)
  if(use_manual_random_seed) then
    !$ call omp_set_schedule(omp_sched_static,10)
  else
    !$ call omp_set_schedule(omp_sched_dynamic,10)
  end if
  !omp
#ifdef __GFORTRAN__
  !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
  !$omp parallel do default(shared) &
  !$omp shared(sim, particles,jorek_stepper, element_list, node_list, rec_v_R,rec_v_Z,rec_v_phi, &
  !$omp i_free,rng,rec_rate_local, &
  !$omp CENTRAL_DENSITY, CENTRAL_MASS,sqrt_mu0_over_rho0,particles_per_element ) &
#endif
  !$omp schedule(runtime)      &
  !$omp private(ife,ielm,k,i,element,s,t,R, Z , &
  !$omp st_ran, i_rng ) &
  !$omp reduction(+:sanity_rec_local)
  do ife = 1, size(rec_rate_local) ! loop over all local elements

    if (isnan(rec_v_R(ife)) .or. isnan(rec_v_Z(ife)) .or. isnan(rec_v_phi(ife))) CYCLE !NaN check
    if (rec_rate_local(ife)* central_density* 1.d20 .le. 1.d3) CYCLE
    
    !$ i_rng = omp_get_thread_num()+1
    !if (rec_rate_local(ife) / real(particles_per_element)* central_density* 1.d20 .le. 1.d7) cycle
    
    k = ife !< every OMP thread gets different values
    !< every MPI process has it's own list of i_free.
    
      ! --- Get element
    !ielm = jorek_stepper%local_elms(ife) !< actual element number
    ielm    = (sim%my_id+1) + sim%n_cpu*(ife - 1)
    element = element_list%element(ielm)
    
    ! initialise particle in the element with Position, Weight, Energy, Momentum
    do i = 1, particles_per_element
        k = k *i !< update free particle index ! at begin of loop as k is initialized at k =0
      particles(i_free(k))%weight = rec_rate_local(ife) / real(particles_per_element,8)* central_density* 1.d20 !< rec_rate = in jorek units?
      particles(i_free(k))%i_elm  = ielm  !x, i_elm, st
      particles(i_free(k))%q      = 0
      
      sanity_rec_local = sanity_rec_local + particles(i_free(k))%weight
      
      call rng(i_rng)%next(st_ran)
      !< sample random st combination
      particles(i_free(k))%st(1) = 0.5d0
      particles(i_free(k))%st(2) = 0.5d0
      
      s = particles(i_free(k))%st(1)
      t = particles(i_free(k))%st(2)
      
      !> uses i_elm and s,t to give us R,Z
      call interp_RZ(node_list,element_list,ielm,s,t,R,Z)
      particles(i_free(k))%x(1:2)  = [R, Z]!  = [R, Z, phi] no phi for axisymmetrix particles
      
      particles(i_free(k))%v(1)  = rec_v_R(ife)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0 !m/s
      particles(i_free(k))%v(2)  = rec_v_Z(ife)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
      particles(i_free(k))%v(3)  = rec_v_phi(ife) / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
          !< v = momentum fluid lost to recombination / (mass of superparticle)
    end do ! parts_per_element
  
  enddo   !ife 
  !$omp end parallel do
  end select
  !end omp

  call MPI_REDUCE(sanity_rec_local, total_sanity_rec, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  if (sim%my_id .eq. 0) then
    write(*,*) 'SANITY recombination weight : ' , total_sanity_rec 
    write(*,*) 'SANITY Recombination rate  [#/s] : ' , total_sanity_rec /tstep_fluid_si
  endif    

end subroutine !do_1particle_recombination

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
  
  call D_edge%prepare(node_list, element_list, edge_domains, nsub=6, nsub_toroidal=1)!,wall_albedo=wall_albedo)

  ! target group, number of particles per mpi task, densities, Zs, basename
  D_sputter_source = particle_sputter(D_edge, 1, n_reflect, basename='D_reflect')
  D_sputter_source%use_Yn_func = .false.
  D_sputter_source%n_save = nout !10 ! or nout
  D_sputter_source%albedo_for_neutrals = 1.d0
  D_sputter_source%sputtered_particle_weight_threshold = 1.d0

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

subroutine conservation_checks(sim)
  implicit none
  
  class(particle_sim), target, intent(in) :: sim

  integer :: j, ierr

  real*8    :: v_kin_temp, E(3), B(3), psi, U, B_norm(3)

  real*8    :: density_tot, density_in, density_out,  pressure, pressure_in, pressure_out
  real*8    :: mom_par_tot, mom_par_in, mom_par_out, kin_par_tot, kin_par_out, kin_par_in
  real*8    :: particles_remaining, momentum_remaining, energy_remaining, all_particles, all_momentum, all_energy
  integer   :: superparticles_remaining,all_superparticles,closest_iteration!, part_i_save,part_n_save
  real*8, dimension(n_var) :: varminout, varmaxout

  call Integrals_3D(sim%my_id, sim%fields%node_list, sim%fields%element_list, density_tot, density_in, density_out, &
  pressure, pressure_in, pressure_out, kin_par_tot, kin_par_in, kin_par_out, mom_par_tot, mom_par_in, mom_par_out,varminout,varmaxout)

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
      superparticles_remaining = superparticles_remaining + 1

      enddo !j
    !omp end parallel do
  end select

  call MPI_REDUCE(particles_remaining, all_particles,         1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(momentum_remaining,  all_momentum,          1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(energy_remaining,    all_energy,            1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(superparticles_remaining,all_superparticles,1, MPI_INTEGER,          MPI_SUM, 0, MPI_COMM_WORLD, ierr) 

  if (sim%my_id .eq. 0) then
    write(*,'(A,3e16.8)') 'REMAINING (START) : ',all_particles, all_momentum, all_energy

    write(*,'(A,126e16.8)') ' TOTAL1 : ',sim%time,density_tot+all_particles/1.d20, density_tot, all_particles/1.d20, &
                        mom_par_tot+all_momentum, mom_par_tot, all_momentum, &
                        pressure+kin_par_tot+all_energy, pressure, all_energy, kin_par_tot

    write(*,'(A,I13,A,E8.2,A,F13.10,A)') 'Superparticles in use :',all_superparticles,' of ', n_particles, '| in use :', &
    real(all_superparticles)/n_particles*100.d0,'%'

    if ( all_superparticles .gt. 0 ) then
      write(*,'(A,2E16.8)') 'Average weight of particles',(all_particles)/all_superparticles
    endif !real(count(

  endif !(sim%my_id .eq. 0)
end subroutine conservation_checks

subroutine write_to_outputfile(id,what)
  implicit none
  
  integer, intent(in) :: id
  character(len=*),intent(in) :: what

  if(id .ne. 0) return

  write(*,'(A100)') "===================================================================================================="
  write(*,*) what
  write(*,'(A100)') "===================================================================================================="

end subroutine

end program kinetic_neutral_loop