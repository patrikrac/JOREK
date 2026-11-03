!> Boris push test for W7-AS stellarator: Gaussian blob of W impurities,
!> thermal velocity init, ADAS ioniz/recomb, Boris push + find_RZ_nearby.
!> Killer test: energy conservation |v|^2 = const (Boris in static B, no E field).
!> Derived from stel_ex2_push.f90 (W7-A); only change: blob center R=1.934 m (W7-AS axis).
program stel_ex2_push_w7as
use data_structure
use mod_find_rz_nearby
use mod_interp,        only: interp_PRZP, interp_gvec, mode_moivre
use mod_pcg32_rng,     only: pcg32_rng
use mod_random_seed
use omp_lib
use mpi
use constants
use mod_fields,        only: fields_base
use mod_fields_linear, only: read_jorek_fields_interp_linear

use mod_jorek_timestepping

use mod_event
use mod_io_actions
use mod_particle_types
use mod_particle_io
use mod_particle_allocation
use mod_boris
use mod_openadas,                 only: read_adf11
use mod_ionisation_recombination, only: new_charge
use mod_collisions
use mod_project_particles
use mod_rhs_projections
use mod_basisfunctions

use phys_module, only: CENTRAL_MASS, CENTRAL_DENSITY
use phys_module, only: restart, restart_particles, tstep_particles, nstep_n
use phys_module, only: use_manual_random_seed, nout, nout_projection, nout_particles
!use phys_module, only: filter_perp, filter_hyper, filter_par, filter_perp_n0, filter_hyper_n0, filter_par_n0

! check this for psi_n diagnostics (see if can expand to include stellarator as well)
use mod_particle_diagnostics

implicit none

interface
  subroutine find_RZP(node_list,element_list,R_find,Z_find,phi_find,R_out,Z_out,ielm_out,s_out,t_out,ifail,checked_elms)
    use data_structure
    type (type_node_list),    intent(in)    :: node_list
    type (type_element_list), intent(in)    :: element_list
    real*8,                   intent(in)    :: R_find, Z_find, phi_find
    real*8,                   intent(out)   :: R_out,Z_out,s_out,t_out
    integer,                  intent(inout) :: ielm_out
    integer,                  intent(out)   :: ifail, checked_elms
  end subroutine find_RZP
end interface

!***********************************************************************
!*                  Set up simulation variables                        *
!***********************************************************************

type(particle_sim)  :: sim
class(*), pointer   :: pa
integer             :: i, j, k, n_lost, n_lost_global, istep, my_nstep_particles, ierr
real*8              :: n_norm, rho_norm, t_norm
real*8              :: tstep_, tstep_fluid_si, tstep_part_adj
character(len=1024) :: filename       ! input
character(len=25)   :: hdf5_file_name ! output
real*8              :: E(3), B(3), rz_old(2), st_old(2), psi, U
type(event)         :: fieldreader, partreader
integer             :: i_elm, ifail_, i_elm_old, q_old
real*8              :: s, t, psi_norm, dummy_1, dummy_2, dummy_3, dummy_4, dummy_5

! --- Initialization ---
logical :: with_v_thermal_init, with_psi_n_init
real*8  :: R, Z, phi
real*8  :: sigma_r, sigma_z, sigma_phi

! --- Physics ---
! Coronal equilibrium
real*8                                      :: n_e, T_e, n_e_raw, T_e_raw, Temp_norm
real*8                                      :: ionize_ran_imp(2)
type(pcg32_rng), dimension(:), allocatable  :: rng
integer                                     :: n_stream, l, seed_size, i_rng, seed, limits, limits_coll
! Collisions
real*8,          dimension(1)               :: P, P_s, P_t, P_phi!, P_time
real*8                                      :: R_s, R_t, Z_s, Z_t, R_phi, Z_phi
integer,         parameter                  :: n_coll = 15   ! number of particles sampled from background with which to collide
integer*1                                   :: q_b
real*8                                      :: n_b, m_b, kTb, coulomb_log
real*8                                      :: grad_T_e(3), q(3), ran2(6, n_coll), v_b(3, n_coll), ran(6)
real*8                                      :: Du, v(3), nv, c_si
logical                                     :: with_push, with_coll, fix_ne_Te, with_thermal_force, with_E_field
real*8                                      :: v_par_init ! m/s

! --- Diagnostics ---
! Projections
type(projection), target  :: project_density
real*8, allocatable       :: feedback_rhs(:,:,:,:,:), feedback_rhs_thread(:,:,:,:,:,:)
real*8                    :: HZ(n_tor), HH(4,4), HH_s(4,4), HH_t(4,4), dens
integer                   :: m, i_tor
logical                   :: with_JOREK_proj

!***********************************************************************
!*                     Simulation configuration                        *
!***********************************************************************

with_psi_n_init =     .false. ! kinetic_develop_stellarator_elias: pa%n_e/T_e/psi_n removed; use Gaussian blob init
with_v_thermal_init = .true.  ! PUSH TEST: sample v from Maxwell at T_e=43 eV
with_push =           .true.  ! PUSH TEST: enable Boris push + find_RZ_nearby
with_coll =           .false. ! no Coulomb collisions: isolate push + ADAS
with_thermal_force =  .false. ! no thermal force: irrelevant without collisions
with_JOREK_proj =     .false.

fix_ne_Te =           .true.  ! override T_e to ~43 eV (20*Temp_norm K) so ADAS limits are satisfied

with_E_field =        .false. ! E=0: static equilibrium, no work done -> |v|^2 conserved by Boris
v_par_init =          0.d0

!***********************************************************************
!*                          Initialisation                             *
!***********************************************************************

! -- Start up MPI, JOREK
call sim%initialize()

! --- Loading JOREK fields
if (restart) then
    fieldreader = event(read_jorek_fields_interp_linear(basename='jorek', i=-1))
    call with(sim, fieldreader)
  else
    if (sim%my_id == 0) write(*,*) 'ERROR: using this program without restarting from a jorek field is not possible. Please set restart=.t. in the namelist and provide a jorek_restart.h5 file'
    stop
end if

! --- Set up the particles
if (restart_particles) then
  ! Read particles from a file
  if (sim%my_id == 0) write(*,*) '========== INFO: READING PARTICLES RESTART FILE =========='
  partreader = event(read_action(filename='part_restart.h5'))
  call with(sim, partreader) !<defines sim%groups and the corresponding particles
else
  call allocate_particles_for_sim(sim)  ! populate particle arrays in the particle groups considering mpi

  ! Retrieving ADAS data
  sim%groups(1)%ad = read_adf11(0, '89_w')

  !> initialise particles: around toroidal direction with psi_n, OR in localised gaussian blob 
  if (with_psi_n_init .eq. .true.) then
    call intialise_particles_with_psi_n(sim, with_v_thermal_init)
  else
    call initiliase_particles_as_gaussian_blob(sim, R_=1.934d0, Z_=0.d0, phi_=0.d0, & ! near W7-AS axis (R_axis~1.934 m): high n_e ensures ADAS limits not triggered
      sigma_r_=.005d0, sigma_z_=.005d0, sigma_phi_=0.d0, with_v_thermal_init_=with_v_thermal_init)
  end if

  ! Write particle restart file so one knows where particles are being initialized
  if (sim%my_id == 0) write(*,*) 'Writing particle restart file'
  write(hdf5_file_name, '(A, F11.9, A)') 'jorek_part_', sim%time, '.h5'
  call write_simulation_hdf5(sim, hdf5_file_name)
end if

!***********************************************************************
!*                         Set up physics                              *
!***********************************************************************

! --- Check if tstep_particles is positive
if (tstep_particles <= 0.d0) then
  if (sim%my_id == 0) write(*,*) "ERROR: tstep_particles <= 0 which is not allowed. Stopping now."
  stop
end if

! --- Set dpsi_dt to be zero to treat JOREK as an equilibrium field
call sim%fields%set_flag_dpsidt(.true.)

! --- Setting up random numbers for ionisation probability
seed = random_seed()
n_stream = 1
!$ n_stream = omp_get_max_threads()
write(*,*) "id, n_mpi, n_stream",sim%my_id, sim%n_mpi, n_stream
allocate(rng(n_stream))
do i=1,n_stream
  call rng(i)%initialize(1, seed, n_stream, i)
end do

! --- Set up JOREK projection scheme for diagnostics
if (with_JOREK_proj .eq. .true.) then
  ! Type of projection: proj_one -> density
  project_density = new_projection(sim%fields%node_list, sim%fields%element_list, &
                                    !filter    = filter_perp,    filter_hyper    = filter_hyper,    filter_parallel    = filter_par, &
                                    !filter_n0 = filter_perp_n0, filter_hyper_n0 = filter_hyper_n0, filter_parallel_n0 = filter_par_n0, &
                                    f=[                                           &
                                      proj_f(proj_one, group=1)                   &
                                      !proj_f(proj_Z, group=1)                     &
                                      ],                                          &
                                    !fractional_digits = 9,                        &
                                    to_vtk=.true., to_h5 = .false.,               &!, index_h5=.false.,
                                    basename='')
  allocate(project_density%rhs(n_degrees, n_vertex_max, sim%fields%element_list%n_elements, n_tor, 1))
  allocate(feedback_rhs,source=project_density%rhs)
  allocate(feedback_rhs_thread(n_degrees, n_vertex_max, sim%fields%element_list%n_elements, n_tor, 1, omp_get_max_threads()))
  project_density%rhs = 0.d0
  feedback_rhs        = 0.d0
  feedback_rhs_thread = 0.d0
end if

! --- Calculate normalisation constants
n_norm    = CENTRAL_DENSITY * 1.d20                              ! (number) density normalisation
rho_norm  = CENTRAL_MASS * ATOMIC_MASS_UNIT * n_norm             ! rho_SI = rho_norm * rho
t_norm    = sqrt((MU_ZERO * rho_norm))                           ! t_SI   = t_norm * t_jorek 


!***********************************************************************
!*                            Main loop                                *
!***********************************************************************

istep = 0
do while (.not. sim%stop_now)
  istep = istep + 1
  if(sim%my_id == 0) write(*,'(A80)'  ) "================================================================================"
  if(sim%my_id == 0) write(*,'(A37,I6)') "Starting main loop iteration istep = ", istep
  if(sim%my_id == 0) write(*,'(A80)'  ) "================================================================================"

  ! --- Determining the time stepping for this fluid step
  tstep_ = get_tstep_n(istep)            ! fluid dt in JOREK units
  tstep_fluid_si = tstep_*t_norm         ! fuild dt in SI units
  sim%time = sim%time + tstep_fluid_si   ! carries the time at the end of the current step

  my_nstep_particles = ceiling(tstep_fluid_si / tstep_particles) ! ceiling makes sure tstep_part_adj is never bigger than tstep_particles
  tstep_part_adj = tstep_fluid_si / my_nstep_particles ! slightly smaller tstep_particles to fit an exact integer amount in one fluid timestep

  if (sim%my_id == 0) then
    write(*,*) "PARTICLE : tstep_particles    : ",tstep_particles
    write(*,*) "PARTICLE : tstep_part_adj     : ",tstep_part_adj
    write(*,*) "PARTICLE : sim%time           : ",sim%time
    write(*,*) "PARTICLE : my_nstep_particles : ",my_nstep_particles
    write(*,*) "PARTICLE : tstep_fluid_si     : ",tstep_fluid_si
    write(*,*) "PARTICLE : n*dt_part - dt     : ",my_nstep_particles*tstep_part_adj - tstep_fluid_si
  endif

  if (with_JOREK_proj) then
    feedback_rhs        = 0.d0
    feedback_rhs_thread = 0.d0
  end if
  n_lost              = 0
  n_lost_global       = 0
  Temp_norm = (1.d0/EL_CHG/(2.d0*MU_ZERO*CENTRAL_DENSITY*1.d20)) ! T in eV
  ! --- Loop over all particle groups
  do i=1,1
    if(use_manual_random_seed) then
      !$ call omp_set_schedule(omp_sched_static,10)
    else
      !$ call omp_set_schedule(omp_sched_dynamic,10)
    end if
    !$omp parallel do default(none)                                                                                   &
    !$omp shared( sim, i, rng, my_nstep_particles, tstep_part_adj, Temp_norm, CENTRAL_DENSITY, CENTRAL_MASS,          & 
    !$omp         nout_projection, with_coll, with_JOREK_proj, with_push, with_E_field,                               &
    !$omp         with_thermal_force, v_par_init, istep, fix_ne_Te, feedback_rhs_thread)                              &
    !$omp private(j, k, t, pa, i_elm_old, q_old, E, B, psi, U, rz_old, st_old, n_e, T_e, n_e_raw, T_e_raw, c_si, ionize_ran_imp,        & 
    !$omp         ifail_, filename, i_rng, i_tor, l, m, dens, psi_norm, dummy_1, dummy_2, dummy_3, dummy_4, dummy_5,  & 
    !$omp         grad_T_e, kTb, n_b, q_b, m_b, q, coulomb_log, ran2, v_b, ran, HH, HZ, HH_s, HH_t, limits_coll, limits)                   &
    !$omp schedule(runtime)                                                                                           &
    !$omp reduction(+: n_lost)
    do j=1, size(sim%groups(i)%particles) !< particles

      !$ i_rng = omp_get_thread_num()+1
      do k=1, my_nstep_particles !< steps

        !> exit evolution loop if particle is outside domain
        if (sim%groups(i)%particles(j)%i_elm .le. 0) then
          n_lost = n_lost + 1
          exit
        end if

        !> check that particle weight is non negative
        !if (sim%groups(i)%particles(j)%weight .lt. 0.0d0) write(*,*) "Negative particle weight pa(j)%w=", sim%groups(i)%particles(j)%weight
        
        t = sim%time + (k-1)*tstep_part_adj
        
        ! --- Integrate time evolution scheme of particles
        select type (pa => sim%groups(i)%particles(j))
        type is (particle_kinetic_leapfrog)
          
          !> calculate local fields
          call sim%fields%calc_EBpsiU(t, pa%i_elm, pa%st, pa%x(3), E, B, psi, U)
          rz_old    = pa%x(1:2)
          st_old    = pa%st
          i_elm_old = pa%i_elm
          q_old     = pa%q
          !write(*,*) "w_ci = ", norm2(B) * pa%q * EL_CHG / (183.84 * 1.660539e-27) ! to determine ion timestep -> there should be at least ~100 time steps per gyro orbit
          
          !write(*,*) ""
          !write(*,*) ""
          !write(*,*) "(R,Z,phi) [m,m,rad]"
          !write(*,*) "", pa%x

          !> calculate n_i [m^-3] and T_e [K] (jorek model assumption: n_e = n_i)
          !call sim%fields%calc_NeTe(t, pa%i_elm, pa%st, pa%x(3), n_e=n_e, T_e=T_e, grad_T_e=grad_T_e)  ! &
          ! always use grad_T_e, otherwise T2_1 is NaN matrix when sampling shifted Maxwellian

          call sim%fields%calc_NeTe(t, pa%i_elm, pa%st, pa%x(3), n_e=n_e, T_e=T_e, n_e_raw=n_e_raw, T_e_raw=T_e_raw, grad_T_e=grad_T_e)

          ! Override T_e BEFORE limits check so ADAS activates even in cold equilibrium
          if (fix_ne_Te .eq. .true.) then
            T_e = 20 * Temp_norm ! ~4.97e5 K = 42.8 eV (20 * T_norm_eV_per_JOREK_unit)
            grad_T_e = 0
          end if

          ! Limits: use (possibly overridden) T_e, not T_e_raw, so fix_ne_Te takes effect
          limits = (n_e_raw .le. 1e14) .or. (T_e * K_BOLTZ / EL_CHG .le. 1.d0)
          limits_coll = T_e * K_BOLTZ / EL_CHG .lt. 1.d0

          ! pa%n_e = n_e          ! removed from particle_base in kinetic_develop
          ! pa%T_e = T_e * K_BOLTZ / EL_CHG ! removed from particle_base in kinetic_develop
          
          !> IONISATION AND RECOMBINATION
          if (.not. limits) then
            call rng(i_rng)%next(ionize_ran_imp)
            pa%q = int(new_charge(int(q_old,4), sim%groups(i)%ad, log10(n_e), log10(T_e), tstep_part_adj, ionize_ran_imp(1:2)),1)
            !write(*,*) "charge state Z", pa%q
          end if
          !< IONISATION AND RECOMBINATION
          
          !> COLLISIONS
          if ((with_coll .eq. .true.) .and. (.not. limits_coll)) then
            if (pa%q .gt. 0) then
              ! variables
              kTb = T_e * K_BOLTZ ! [J] E = T * k_B [K * J/K]
              n_b = n_e
              q_b = 1             ! deuterium is always ionized under fusion conditions
              m_b = CENTRAL_MASS
            
              if (with_thermal_force .eq. .true.) then
                q = q_homma2013(kTb, grad_T_e*K_BOLTZ, B, n_b, m_b, q_b)
                !write(*,*) "heat flux q", q
                !q(1)=0
                !q(2)=0
              else
                q = 0
              end if
            
              !> Calculate coulomb logarithm and limit it to reasonable values
              coulomb_log = coulomb_logarithm(kTb, n_b, pa%q, q_b, sim%groups(1)%mass, m_b)
              !write(*,*) "coulomb_log", coulomb_log
              coulomb_log = max(10.d0, coulomb_log)
              coulomb_log = min(20.d0, coulomb_log)
            
              !> Get parallel flow velocity (-> not physical from fluid side -> background flow velocity is not properly calculated (no sheath BCs))
              !call interp_PRZP(sim%fields%node_list, sim%fields%element_list, pa%i_elm, [var_Vpar], 1, pa%st(1), pa%st(2), pa%x(3), & 
              !                 P_, P_s, P_t, P_phi, R_, R_s, R_t, R_phi, Z_, Z_s, Z_t, Z_phi)
            
              do l=1,n_coll
                call rng(i_rng)%next(ran2(:,l))
              end do
            
              call sample_velocity_dist_magnetized(n_coll, ran2(1:6,:), kTb, q, n_b, m_b, q_b, v_par_init*B/sim%t_norm, v_b) ! P_(1)*B/sim%t_norm, v_b)
              !write(*,*) "v_b from sample_velocity_dist_magnetized"
              !write(*,*) "", v_b
              !write(*,*) "pa%v"
              !write(*,*) "", pa%v
            
              do l=1,n_coll
                call rng(i_rng)%next(ran)
                call collide_particles(ran(1:3), pa%q, sim%groups(1)%mass, pa%v, &
                     q_b, m_b, v_b(:,l), n_b, coulomb_log, tstep_part_adj/real(n_coll,8))
              end do
            !write(*,*) "pa%v after coll"
            !write(*,*) "", pa%v
            end if
          end if !< COLLISIONS
          
          ! if no E-field
          if (with_E_field .eq. .false.) then
            E = 0.d0
          end if
          
          ! --- Particle pushing and find out where they are next ---
          if (with_push .eq. .true.) then
            call boris_push_cylindrical(pa, m=sim%groups(i)%mass, E=E, B=B, dt=tstep_part_adj)
            call find_RZ_nearby(sim%fields%node_list, sim%fields%element_list, rz_old(1), rz_old(2), st_old(1), st_old(2), i_elm_old, pa%x(1), pa%x(2), pa%st(1), pa%st(2), pa%i_elm, ifail_, pa%x(3))
          end if

          !> particle got pushed outside of the grid -> skip to next particle
          if (pa%i_elm .le. 0) then
            n_lost = n_lost + 1
            exit
          end if

          call interp_gvec(sim%fields%node_list, sim%fields%element_list, pa%i_elm, 4, 1, 1, pa%st(1), pa%st(2), psi_norm, dummy_1, dummy_2, dummy_3, dummy_4, dummy_5) !s_norm, BRg_s, BRg_t, BRg_st, BRg_ss, BRg_tt
          psi_norm = psi_norm * psi_norm
          ! pa%psi_n = psi_norm    ! removed from particle_base in kinetic_develop

          !> PROJECTIONS
          if ((with_JOREK_proj .eq. .true.) .and. (mod(istep, nout_projection) .eq. 0)) then
            !> Calculate the projection of the ion source in real-time
            call basisfunctions(pa%st(1), pa%st(2), HH, HH_s, HH_t)
            call mode_moivre(pa%x(3), HZ)
            
            do l=1, n_vertex_max
              do m=1, n_degrees
                
                dens = HH(l,m) * sim%fields%element_list%element(pa%i_elm)%size(l,m) * 1 !particle_source * t_norm / rho_norm
                
                do i_tor=1,n_tor
                  feedback_rhs_thread(m,l,pa%i_elm,i_tor,1, i_rng) = feedback_rhs_thread(m,l,pa%i_elm,i_tor,1, i_rng) + HZ(i_tor) * dens
                enddo
                feedback_rhs_thread(m,l,pa%i_elm, 1, 1, i_rng) = sum(feedback_rhs_thread(m,l,pa%i_elm, :, 1, i_rng))                     
              enddo
            enddo
          end if !< PROJECTIONS
        end select
      end do !< steps
    end do !< particles
    !$omp end parallel do
    
    ! Perform particle projections in JOREK Fourier basis
    if ((with_JOREK_proj .eq. .true.) .and. (mod(istep, nout_projection) .eq. 0)) then
      do i_rng = 1, omp_get_max_threads()
        feedback_rhs = feedback_rhs + feedback_rhs_thread(:,:,:,:,:,i_rng) ! Sequentially reduce thread arrays for bitwise determinism
      end do
      
      feedback_rhs = feedback_rhs/real(my_nstep_particles,8)
      project_density%rhs = feedback_rhs
      call with(sim, project_density)
    end if
    
    ! Count total number of lost particles and exit main loop if all particles are lost
    call MPI_ALLREDUCE(n_lost, n_lost_global, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
    if (sim%my_id == 0) then
      write(*,*) "number/% of lost particles: ", n_lost_global, real(n_lost_global)/sim%groups(1)%n_particles*100.0
    end if

    if (n_lost_global .eq. sim%groups(1)%n_particles) then
      if (sim%my_id == 0) write(*,*) "All particles lost."
      exit
    endif
  end do ! groups
  
  ! --- Write restart files
  if (mod(istep, nout) .eq. 0) then
    if (sim%my_id == 0) write(*,*) 'Writing particle restart file'
    write(hdf5_file_name, '(A, F11.9, A)') 'jorek_part_', sim%time, '.h5'
    call write_simulation_hdf5(sim, hdf5_file_name)
  end if
  
  ! --- Finish loop according to nstep_n variable, which can be assigned in namelist
  if (istep .ge. nstep_n(1)) then
    sim%stop_now = .true.
    call write_simulation_hdf5(sim, 'part_restart.h5') ! Write last particle restart file as part_restart.h5
  end if
  
  ! Exit sim if more than 99.9% of particles are lost
  if (real(n_lost_global / sim%groups(1)%n_particles, 8) .gt. 0.999d0) then
    if (sim%my_id == 0) write(*,*) "Exiting simulation."
    exit
  endif
  
end do ! while

deallocate(rng)

call sim%finalize

!***********************************************************************
!*                      End of main program                            *
!***********************************************************************

contains

!***********************************************************************
!*                        Helper functions                             *
!***********************************************************************

!> Initialise particles as a gaussian blob around a given location
subroutine initiliase_particles_as_gaussian_blob(sim_, R_, Z_, phi_, sigma_r_, sigma_z_, sigma_phi_, with_v_thermal_init_)
  !use data_structure
  use mod_particle_sim
  use mpi
  !use mod_particle_types
  use mod_find_rz_nearby
  !use mod_openadas, only: read_adf11
  
  implicit none

  class(particle_sim), intent(inout)  :: sim_
  real*8,              intent(in)     :: R_, Z_, phi_, sigma_r_, sigma_z_, sigma_phi_
  logical,             intent(in)     :: with_v_thermal_init_

  class(*), pointer                   :: pa_
  real*8, allocatable, dimension(:,:) :: positions
  real*8                              :: DUMMY_R, DUMMY_Z, s_, t_
  integer                             :: j_, i_elm_, ifail__, ierr_, checked_elms, current_offset
  integer, dimension(:), allocatable  :: n_particles_per_mpi, global_start_index

  ! Generate Gaussian distributed blob of particles
  allocate(positions(3, int(sim_%groups(1)%n_particles)))
  if (sim_%my_id == 0) positions = generate_3d_gaussian(int(sim_%groups(1)%n_particles), R_, Z_, phi_, sigma_r_, sigma_z_, sigma_phi_)
 
  ! Broadcast (share) positions to all MPI threads
  call MPI_BCAST(positions, 3 * int(sim_%groups(1)%n_particles), MPI_REAL8, 0, MPI_COMM_WORLD, ierr_)
  if (ierr_ /= 0) then
    write(*,*) 'Error in MPI_BCAST on rank', sim_%my_id
  end if

  ! To get correct positions' indices in routine below (needed because of MPI parallelization)
  n_particles_per_mpi = calc_n_particles_per_mpi_array(int(sim_%groups(1)%n_particles), sim_%n_mpi)
  allocate(global_start_index(sim_%n_mpi))
  current_offset = 1
  do j_=1, sim_%n_mpi
    global_start_index(j_) = current_offset
    current_offset = current_offset + n_particles_per_mpi(j_)
  end do

  ! Get particle location in element, s_, t_, space
  !$omp parallel do default(none)                                                             &
  !$omp shared(sim_, positions, global_start_index, n_particles_per_mpi, with_v_thermal_init_)  &
  !$omp private(j_, pa_, DUMMY_R, DUMMY_Z, i_elm_, s_, t_, ifail__, checked_elms)
  do j_ = 1, n_particles_per_mpi(sim_%my_id + 1)  ! each MPI thread allocates a position for its local number of particles
    select type (pa_ => sim_%groups(1)%particles(j_))
    type is (particle_kinetic_leapfrog)

      call find_RZP(sim_%fields%node_list, sim_%fields%element_list,              &
                    positions(1, global_start_index(sim_%my_id + 1) + (j_ - 1)),  & 
                    positions(2, global_start_index(sim_%my_id + 1) + (j_ - 1)),  &
                    positions(3, global_start_index(sim_%my_id + 1) + (j_ - 1)),  &
                    DUMMY_R, DUMMY_Z, i_elm_, s_, t_, ifail__, checked_elms)

      ! Check if particles are initialized inside the JOREK grid domain
      if (i_elm_ .le. 0) then
        write(*,*) "ERROR: particle initalized outside of the grid (i_elm_ .le. 0)", &
                    positions(:, global_start_index(sim_%my_id + 1) + (j_ - 1))
        stop
      end if
      pa_%x      = positions(:, global_start_index(sim_%my_id + 1) + (j_ - 1))
      pa_%i_elm  = i_elm_
      pa_%st     = [s_,t_]
      pa_%weight = 1.0d0
      pa_%q      = 0
      ! pa_%n_e    = 0  ! removed from particle_base in kinetic_develop
      ! pa_%T_e    = 0  ! removed from particle_base in kinetic_develop
      ! pa_%psi_n  = 0  ! removed from particle_base in kinetic_develop
      if (with_v_thermal_init_ .eq. .false.) then
        pa_%v    = [0., 0., 0.] ! [v_R, v_Z, v_phi]
      else
        call initialise_veloc_from_T(pa_, sim_%groups(1)%mass, sim_%fields%node_list, sim_%fields%element_list, &
                                     global_start_index(sim_%my_id + 1) + (j_ - 1))
      end if
    end select
  end do
  !$omp end parallel do

  deallocate(positions, global_start_index, n_particles_per_mpi)

end subroutine initiliase_particles_as_gaussian_blob


!> Initialise particles in specific region of psi_norm
subroutine intialise_particles_with_psi_n(sim_, with_v_thermal_init_)
  !use data_structure
  !use mpi
  use mod_particle_sim
  use mod_particle_allocation
  use initialisers_base,       only: initialise_particles_H_mu_psi
  use mod_pcg32_rng,           only: pcg32_rng
  use mod_random_seed,         only: random_seed

  implicit none

  class(particle_sim), intent(inout) :: sim_
  logical,             intent(in)    :: with_v_thermal_init_

  class(*), pointer                  :: pa_
  integer                            :: j_, ifail__, global_offset, current_offset
  integer, dimension(:), allocatable :: n_particles_per_mpi, global_start_index
  integer, dimension(3)              :: my_rej_vars = [-3, -2, -1] !(phi, Z, R)
  type(pcg32_rng)                    :: init_rng

  ! Initialize the RNG using the JOREK manual seed with MPI parallelization
  call init_rng%initialize(1, random_seed() + sim_%my_id, 1, 1, ierr=ifail__)

  ! Initialise particle positions based on psi_n values
  call initialise_particles_H_mu_psi(sim_%groups(1)%particles, sim_%fields, init_rng, sim_%groups(1)%mass, &
                                    uniform_space=.true., uniform_space_rej_vars=my_rej_vars, uniform_space_rej_f=filter_by_psin)

  ! Calculate global offset for this MPI task to ensure MPI-invariant reproducibility
  n_particles_per_mpi = calc_n_particles_per_mpi_array(int(sim_%groups(1)%n_particles), sim_%n_mpi)
  allocate(global_start_index(sim_%n_mpi))
  current_offset = 1
  do j_=1, sim_%n_mpi
    global_start_index(j_) = current_offset
    current_offset = current_offset + n_particles_per_mpi(j_)
  end do

  ! Allocate velocity and initialise final parameters
  !$omp parallel do default(none)                                                   &
  !$omp shared(sim_, n_particles_per_mpi, global_start_index, with_v_thermal_init_) &
  !$omp private(j_, pa_)
  do j_ = 1, n_particles_per_mpi(sim_%my_id + 1)  ! each MPI thread allocates a position for its local number of particles
    select type (pa_ => sim_%groups(1)%particles(j_))
    type is (particle_kinetic_leapfrog)
      if (pa_%i_elm .le. 0) then
        write(*,*) "ERROR: particle initalized outside of the grid (i_elm .le. 0)", pa_%x
        stop
      end if
      pa_%q      = 0
      ! pa_%n_e    = 0  ! removed from particle_base in kinetic_develop
      ! pa_%T_e    = 0  ! removed from particle_base in kinetic_develop
      ! pa_%psi_n  = 0  ! removed from particle_base in kinetic_develop
      if (with_v_thermal_init_ .eq. .false.) then
        pa_%v    = [0., 0., 0.] ! [v_R, v_Z, v_phi]
      else
        call initialise_veloc_from_T(pa_, sim_%groups(1)%mass, sim_%fields%node_list, sim_%fields%element_list, &
                                     global_start_index(sim_%my_id + 1) + (j_ - 1)) ! Gets v sampled from Maxwell distribution
      end if
    end select
  end do
  !$omp end parallel do

end subroutine intialise_particles_with_psi_n


!> Create mask based on psi_n to initialise particles via rejection sampling
function filter_by_psin(n_vars, P2, grad_P2) result(prob)
  use mod_particle_sim
  use mod_find_rz_nearby

  implicit none
  
  integer, intent(in) :: n_vars
  real*8, dimension(n_vars), intent(in) :: P2
  real*8, dimension(3, n_vars), intent(in) :: grad_P2
  real*4 :: prob
  
  real*8 :: Z_, R_, phi_, psi_norm_, s_, t_, dummy_R, dummy_Z, dummy_1_, dummy_2_, dummy_3_, dummy_4_, dummy_5_
  integer :: i_elm_, ifail__, checked_elms

  ! 1. Extract R_ and Z_ from P2 (assuming uniform_space_rej_vars = [-3, -2, -1])
  phi_ = P2(1)
  Z_   = P2(2)
  R_   = P2(3)

  ! 2. Find the element (i_elm_) and local coordinates (s_, t_) for this R_, Z_
  call find_RZP(sim%fields%node_list, sim%fields%element_list, R_, Z_, phi_, dummy_R, dummy_Z, i_elm_, s_, t_, ifail__, checked_elms)

  ! If particle was generated outside simulation grid, reject it immediately
  if (i_elm_ .le. 0) then
    prob = 0.d0
    return
  end if

  ! 3. Call interp_gvec to get psi_norm_
  call interp_gvec(sim%fields%node_list, sim%fields%element_list, i_elm_, &
                   4, 1, 1, s_, t_, psi_norm_, dummy_1_, dummy_2_, dummy_3_, dummy_4_, dummy_5_)
  psi_norm_ = psi_norm_*psi_norm_

  ! 4. Apply boundaries in psi_n (case dependent)
  if (psi_norm_ .ge. 0.76d0 .and. psi_norm_ .le. 0.9d0) then
    prob = 1.0d0
  else
    prob = 0.0d0
  end if

end function filter_by_psin


!> Initialise velocity of a single kinetic particle from a Maxwell distribution
subroutine initialise_veloc_from_T(particle, mass, node_list, element_list, global_index)
  use data_structure
  use constants,    only: MU_ZERO, ATOMIC_MASS_UNIT
  use phys_module,  only: central_density, F0
  use mod_sampling, only: boxmueller_transform, normal_vectors
  use mod_interp,   only: interp_PRZP
  use mod_random_seed
#if STELLARATOR_MODEL
  use mod_chi
#endif

  implicit none

  class(particle_base),   intent(inout)       :: particle !< whose velocity to initialize
  real*8,                 intent(in)          :: mass     !< [u]
  type(type_node_list),   intent(in)          :: node_list
  type(type_element_list),intent(in)          :: element_list
  integer,                intent(in)          :: global_index ! < global index of the particle (used for RNG seeding to ensure different velocity values for each particle)

  class(*), pointer :: pa_
  type(pcg32_rng) :: rng_
  integer :: ifail__
  real*8, dimension(4) :: P_, P_s_, P_t_, P_phi_, rans
#if STELLARATOR_MODEL
  real*8  :: R_phi_, Z_phi_, psi_phi
  real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: chi
#endif
  real*8 :: R_, R_s_, R_t_, Z_, Z_s_, Z_t_, psi_R, psi_Z, B_(3)
  real*8 :: v_out(3), B_norm, B_hat(3), e1(3), e2(3)
  real*8 :: background_kbT, background_density, v_thermal
  logical:: use_anysotropic_sampling

  use_anysotropic_sampling = .true.

  ! Only an implementation for particle_kinetic_leapfrog now
  select type (pa_ => particle)
  type is (particle_kinetic_leapfrog)

#ifdef WITH_TiTe
#define VAR_T_IDX var_Te
#else
#define VAR_T_IDX var_T
#endif

    if (pa_%i_elm .le. 0) then
      write(*,*) "Particle being initialized is out of domain"
      stop
    else
#if STELLARATOR_MODEL
      call interp_PRZP(node_list,element_list,pa_%i_elm,[1,5,VAR_T_IDX,7],4,pa_%st(1),pa_%st(2),pa_%x(3),&
        P_,P_s_,P_t_,P_phi_,R_,R_s_,R_t_,R_phi_,Z_,Z_s_,Z_t_,Z_phi_) ! density and temperature
#else
      call interp_PRZ (node_list,element_list,pa_%i_elm,[1,5,VAR_T_IDX,7],4,pa_%st(1),pa_%st(2),pa_%x(3),&
        P_,P_s_,P_t_,P_phi_,R_,R_s_,R_t_,       Z_,Z_s_,Z_t_       ) ! density and temperature
#endif

      background_density = P_(2) * 1d20                            ! plasma density [1/m^3]
      ! Assume that particle has the same temperature as electrons
#ifdef WITH_TiTe
      background_kbT = P_(3)/(MU_ZERO*central_density*1.d20)       ! P_(3) contains the electron temperature
#else
      background_kbT = P_(3)/(2.d0*MU_ZERO*central_density*1.d20)  ! P_(3) contains the total plasma temperature in J/kB = T = Te + Ti
#endif
      v_thermal = sqrt(background_kbT / (mass*ATOMIC_MASS_UNIT))   ! variance in each of the velocity dimensions [m/s]

      ! Prepare uniformly distributed random numbers --------- should perhaps use random_seed() + particle_index so that every particle gets a different velocity value
      call rng_%initialize(n_dims=1, seed=random_seed() + global_index, n_streams=1, i_stream=1, ierr=ifail__)
      call rng_%next(rans)

      v_out(1:2) = boxmueller_transform(rans(1:2)) * v_thermal  ! v_perp1 and v_perp2
      v_out(3:4) = boxmueller_transform(rans(3:4)) * v_thermal  ! v_par (and dummy)

      if (use_anysotropic_sampling .eq. .true.) then
        ! v_out sampled contains v_par, v_perp1 and v_perp2 to then distribute in (R_,Z_,phi) components

        ! Calculate psi derivatives (and chi in stellarator model)
        psi_R   = (  P_s_(1) * Z_t_ - P_t_(1) * Z_s_ )/(R_s_ * Z_t_ - R_t_ * Z_s_)
        psi_Z   = (- P_s_(1) * R_t_ + P_t_(1) * R_s_ )/(R_s_ * Z_t_ - R_t_ * Z_s_)
#if STELLARATOR_MODEL
        psi_phi = P_phi_(1) - R_phi_*psi_R - Z_phi_*psi_Z
        chi = get_chi(R_,Z_,pa_%x(3),node_list,element_list,pa_%i_elm,pa_%st(1),pa_%st(2))

        ! Calculate magnetic field
        B_ = (/chi(1,0,0)     + (psi_Z*chi(0,0,1) - psi_phi*chi(0,1,0))/(F0*R_), &
               chi(0,1,0)     - (psi_R*chi(0,0,1) - psi_phi*chi(1,0,0))/(F0*R_), &
               chi(0,0,1)/R_  + (psi_R*chi(0,1,0) - psi_Z * chi(1,0,0))/F0         /)
#else
        B_ = [psi_Z, -psi_R, F0]/(R_)
#endif
        B_norm = sqrt(B_(1)*B_(1) + B_(2)*B_(2) + B_(3)*B_(3))
        B_hat = B_/B_norm

        ! Construct orthonormal Basis (b, e1, e2) using Frisvad method
        call normal_vectors(B_hat, e1, e2)

        ! Transform v_par and v_perp into v_R, v_Z and v_Phi
        pa_%v = v_out(3)*B_hat + v_out(1)*e1 + v_out(2)*e2
      else
        ! v_out sampled is v_R, v_Z and v_phi directly
        pa_%v = v_out
      end if
    end if
  class default
    write(*,*) "initialise_veloc_from_T not implemented for this particle type"
    stop
  end select
end subroutine initialise_veloc_from_T


!> Generates positions following a 3D Gaussian distribution
!! Creates an array of points with x, y, and z coordinates sampled from
!! independent Gaussian distributions.
!!
!! @param n_points Number of positions to generate
!! @param x0       Mean of the x-coordinate Gaussian distribution
!! @param y0       Mean of the y-coordinate Gaussian distribution
!! @param z0       Mean of the z-coordinate Gaussian distribution
!! @param sigma_x  Standard deviation for the x-coordinate
!! @param param sigma_y Standard deviation for the y-coordinate
!! @param sigma_z  Standard deviation for the z-coordinate
!! @return positions Array containing [x, y, z] coordinates for each point
function generate_3d_gaussian(n_points, x0, y0, z0, sigma_x_, sigma_y_, sigma_z_) result(positions)
  use constants, only: pi
  use mod_pcg32_rng, only: pcg32_rng
  use mod_random_seed, only: random_seed
  implicit none

  integer, intent(in) :: n_points
  real*8,  intent(in) :: x0, y0, z0, sigma_x_, sigma_y_, sigma_z_
  real*8, dimension(3,n_points) :: positions
  
  type(pcg32_rng) :: local_rng
  integer :: i_, ifail__
  real*8 :: u1, u2, u3, u4, r_gauss, theta_gauss, r2_gauss, theta2_gauss
  real*8 :: norm1, norm2, norm3
  real*8 :: rans(4)

  ! Initialize local RNG once using JOREK's manual seed
  call local_rng%initialize(1, random_seed(), 1, 1, ierr=ifail__)

  do i_ = 1, n_points
    ! Draw 4 random numbers at once from the seeded generator
    call local_rng%next(rans)
    u1 = rans(1); u2 = rans(2); u3 = rans(3); u4 = rans(4)

    if(u1 .le. 0.0d0) u1 = 1.0d-10
    if(u3 .le. 0.0d0) u3 = 1.0d-10
    
    r_gauss = sqrt(-2.0d0 * log(u1))
    theta_gauss = 2.0d0 * pi * u2
    norm1 = r_gauss * cos(theta_gauss)
    norm2 = r_gauss * sin(theta_gauss)
      
    r2_gauss = sqrt(-2.0d0 * log(u3))
    theta2_gauss = 2.0d0 * pi * u4
    norm3 = r2_gauss * cos(theta2_gauss)

    positions(1,i_) = x0 + sigma_x_ * norm1
    positions(2,i_) = y0 + sigma_y_ * norm2
    positions(3,i_) = z0 + sigma_z_ * norm3
  end do
end function generate_3d_gaussian

end program stel_ex2_push_w7as
              
