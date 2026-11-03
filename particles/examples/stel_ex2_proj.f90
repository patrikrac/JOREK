!> Minimal coronal equilibrium + JOREK projection test for stellarators.
!> No Boris push, no Coulomb collisions. Static JOREK field from restart H5.
!>
!> What it does:
!>   1. Place N neutral W particles in a Gaussian blob (fix_ne_Te = .true. -> ~43 eV)
!>   2. Advance ionization/recombination (new_charge) until coronal equilibrium is reached
!>   3. Project particle count (proj_one) and charge-weighted count (proj_q_int) onto JOREK grid
!>   4. Write two VTK files per nout: "proj_count_*.vtk" and "proj_charge_*.vtk"
!>
!> Validation (post-process in Python):
!>   mean_q(R,Z) = proj_charge / proj_count
!>   This should equal the ADAS coronal equilibrium charge at T_e ~ 43 eV: <Z> ~ 8 for W
!>   (compare Daan Van Vugt PhD thesis Fig. 3.10: W8+ dominant at 40-50 eV)
program stel_ex2_proj
use data_structure
use mod_pcg32_rng,     only: pcg32_rng
use mod_random_seed
use mod_find_rz_nearby
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
use mod_openadas,                 only: read_adf11
use mod_ionisation_recombination, only: new_charge
use mod_project_particles
use mod_rhs_projections

use phys_module, only: CENTRAL_MASS, CENTRAL_DENSITY
use phys_module, only: restart, restart_particles, tstep_particles, nstep_n
use phys_module, only: use_manual_random_seed, nout, nout_particles, nout_projection

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
!*                        Variables                                    *
!***********************************************************************

type(particle_sim)  :: sim
class(*), pointer   :: pa
integer             :: i, j, k, istep, my_nstep_particles, ierr
real*8              :: n_norm, rho_norm, t_norm, Temp_norm
real*8              :: tstep_, tstep_fluid_si, tstep_part_adj
character(len=25)   :: hdf5_file_name
type(event)         :: fieldreader

! Physics
real*8                                      :: n_e, T_e, n_e_raw, T_e_raw, grad_T_e(3)
real*8                                      :: ionize_ran_imp(2)
type(pcg32_rng), dimension(:), allocatable  :: rng
integer                                     :: n_stream, i_rng, seed
integer                                     :: q_old
logical                                     :: limits

! Projection: count (proj_one weight) and charge-weighted count (proj_q weight)
type(projection), target :: project_count, project_charge

!***********************************************************************
!*                      Initialisation                                 *
!***********************************************************************

call sim%initialize()

if (.not. restart) then
  if (sim%my_id == 0) write(*,*) 'ERROR: restart=.t. required; provide jorek_restart.h5'
  stop
end if
fieldreader = event(read_jorek_fields_interp_linear(basename='jorek', i=-1))
call with(sim, fieldreader)

call allocate_particles_for_sim(sim)
sim%groups(1)%ad = read_adf11(0, '89_w')

! Gaussian blob at W7-A axis (sigma=5mm, all at phi=0)
call initiliase_particles_as_gaussian_blob(sim, &
  1.99d0, 0.d0, 0.d0, &
  5.d-3, 5.d-3, 0.d0, &
  .false.)

! Write initial state (q=0, before main loop) for reference
write(hdf5_file_name, '(A, F11.9, A)') 'jorek_part_', sim%time, '.h5'
call write_simulation_hdf5(sim, hdf5_file_name)

! Freeze dψ/dt -> static field, no loop voltage
call sim%fields%set_flag_dpsidt(.true.)

! Random number streams (one per OMP thread)
seed = random_seed()
n_stream = 1
!$ n_stream = omp_get_max_threads()
allocate(rng(n_stream))
do i = 1, n_stream
  call rng(i)%initialize(1, seed, n_stream, i)
end do

! Projection objects: count density and charge-weighted density
!   proj_count  -> particle count per grid node  (weight = 1)
!   proj_charge -> sum(q * el_chg) per grid node (weight = proj_q)
!   Post-process: mean_q = proj_charge_values / EL_CHG / proj_count_values
project_count  = new_projection(sim%fields%node_list, sim%fields%element_list, &
  f=[proj_f(proj_one, group=1)], to_vtk=.true., to_h5=.false., basename='proj_count')
project_charge = new_projection(sim%fields%node_list, sim%fields%element_list, &
  f=[proj_f(proj_q,   group=1)], to_vtk=.true., to_h5=.false., basename='proj_charge')

! Force VTK write on every call: nout_projection=-1 (default) -> set to nout on first call,
! then mod(index_now, nout) may not be 0 (index_now from restart, never incremented here).
! Setting to 1 bypasses the check entirely.
nout_projection = 1

! Normalisations
n_norm   = CENTRAL_DENSITY * 1.d20
rho_norm = CENTRAL_MASS * ATOMIC_MASS_UNIT * n_norm
t_norm   = sqrt(MU_ZERO * rho_norm)

!***********************************************************************
!*                          Main loop                                  *
!***********************************************************************

istep = 0
do while (.not. sim%stop_now)
  istep = istep + 1
  if (sim%my_id == 0) write(*,'(A,I6)') 'istep = ', istep

  tstep_         = get_tstep_n(istep)
  tstep_fluid_si = tstep_ * t_norm
  sim%time       = sim%time + tstep_fluid_si
  my_nstep_particles = ceiling(tstep_fluid_si / tstep_particles)
  tstep_part_adj = tstep_fluid_si / my_nstep_particles
  Temp_norm = 1.d0 / (EL_CHG * 2.d0 * MU_ZERO * CENTRAL_DENSITY * 1.d20)

  do i = 1, 1  ! single particle group
    !$omp parallel do default(none)                                               &
    !$omp shared(sim, i, rng, my_nstep_particles, tstep_part_adj, Temp_norm,     &
    !$omp        CENTRAL_DENSITY, CENTRAL_MASS)                                  &
    !$omp private(j, k, pa, q_old, n_e, T_e, n_e_raw, T_e_raw, grad_T_e, i_rng, &
    !$omp         ionize_ran_imp, limits)                                         &
    !$omp schedule(dynamic, 10)
    do j = 1, size(sim%groups(i)%particles)
      !$ i_rng = omp_get_thread_num() + 1
      do k = 1, my_nstep_particles

        select type (pa => sim%groups(i)%particles(j))
        type is (particle_kinetic_leapfrog)
          q_old = pa%q

          call sim%fields%calc_NeTe(sim%time, pa%i_elm, pa%st, pa%x(3), &
            n_e=n_e, T_e=T_e, n_e_raw=n_e_raw, T_e_raw=T_e_raw, grad_T_e=grad_T_e)

          ! Fix T_e: cold W7-A equilibrium (T_peak~0.03 eV) would disable ADAS
          ! 20 * Temp_norm ~ 43 eV -> W8+ coronal equilibrium
          T_e = 20.d0 * Temp_norm

          limits = (n_e_raw .le. 1d14) .or. (T_e * K_BOLTZ / EL_CHG .le. 1.d0)

          if (.not. limits) then
            call rng(i_rng)%next(ionize_ran_imp)
            pa%q = int(new_charge(int(q_old, 4), sim%groups(i)%ad, &
              log10(n_e), log10(T_e), tstep_part_adj, ionize_ran_imp(1:2)), 1)
          end if

        end select
      end do ! k substeps
    end do ! j particles
    !$omp end parallel do
  end do ! i groups

  ! --- Projection + H5 write at output cadence
  ! sample_rhs iterates over all particles using the f=[...] weight functions (OMP-parallel)
  ! project_only solves the mass-matrix system; result goes to node_list%node%values
  ! save_to_vtk writes proj_count_*.vtk and proj_charge_*.vtk (nout_projection=1 ensures write)
  if (mod(istep, nout) .eq. 0) then
    call with(sim, project_count)   ! proj_count_*.vtk:  particle count on grid (weight=1)
    call with(sim, project_charge)  ! proj_charge_*.vtk: sum(q*el_chg) on grid (weight=proj_q)

    write(hdf5_file_name, '(A, F11.9, A)') 'jorek_part_', sim%time, '.h5'
    call write_simulation_hdf5(sim, hdf5_file_name)
  end if

  if (istep .ge. nstep_n(1)) then
    sim%stop_now = .true.
    call write_simulation_hdf5(sim, 'part_restart.h5')
  end if

end do ! while

deallocate(rng)
call sim%finalize

contains

  subroutine initiliase_particles_as_gaussian_blob(sim_, R_, Z_, phi_, sigma_r_, sigma_z_, sigma_phi_, with_v_thermal_init_)
    use mod_particle_sim
    use mod_particle_allocation, only: calc_n_particles_per_mpi_array
    use mpi
    implicit none
    class(particle_sim), intent(inout) :: sim_
    real*8,              intent(in)    :: R_, Z_, phi_, sigma_r_, sigma_z_, sigma_phi_
    logical,             intent(in)    :: with_v_thermal_init_
    class(*), pointer                  :: pa_
    real*8, allocatable, dimension(:,:) :: positions
    real*8              :: DUMMY_R, DUMMY_Z, s_, t_
    integer             :: j_, i_elm_, ifail__, ierr_, checked_elms, current_offset
    integer, dimension(:), allocatable :: n_particles_per_mpi, global_start_index

    allocate(positions(3, int(sim_%groups(1)%n_particles)))
    if (sim_%my_id == 0) positions = generate_3d_gaussian(int(sim_%groups(1)%n_particles), R_, Z_, phi_, sigma_r_, sigma_z_, sigma_phi_)
    call MPI_BCAST(positions, 3 * int(sim_%groups(1)%n_particles), MPI_REAL8, 0, MPI_COMM_WORLD, ierr_)

    n_particles_per_mpi = calc_n_particles_per_mpi_array(int(sim_%groups(1)%n_particles), sim_%n_mpi)
    allocate(global_start_index(sim_%n_mpi))
    current_offset = 1
    do j_=1, sim_%n_mpi
      global_start_index(j_) = current_offset
      current_offset = current_offset + n_particles_per_mpi(j_)
    end do

    !$omp parallel do default(none)                                                                  &
    !$omp shared(sim_, positions, global_start_index, n_particles_per_mpi, with_v_thermal_init_)    &
    !$omp private(j_, pa_, DUMMY_R, DUMMY_Z, i_elm_, s_, t_, ifail__, checked_elms)
    do j_ = 1, n_particles_per_mpi(sim_%my_id + 1)
      select type (pa_ => sim_%groups(1)%particles(j_))
      type is (particle_kinetic_leapfrog)
        call find_RZP(sim_%fields%node_list, sim_%fields%element_list,                          &
                      positions(1, global_start_index(sim_%my_id+1) + (j_-1)),                  &
                      positions(2, global_start_index(sim_%my_id+1) + (j_-1)),                  &
                      positions(3, global_start_index(sim_%my_id+1) + (j_-1)),                  &
                      DUMMY_R, DUMMY_Z, i_elm_, s_, t_, ifail__, checked_elms)
        if (i_elm_ .le. 0) then
          write(*,*) 'ERROR: particle initialized outside grid'
          stop
        end if
        pa_%x     = positions(:, global_start_index(sim_%my_id+1) + (j_-1))
        pa_%i_elm = i_elm_
        pa_%st    = [s_, t_]
        pa_%weight = 1.0d0
        pa_%q      = 0
        pa_%v      = [0.d0, 0.d0, 0.d0]
      end select
    end do
    !$omp end parallel do
    deallocate(positions, global_start_index, n_particles_per_mpi)
  end subroutine initiliase_particles_as_gaussian_blob

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
    real*8  :: u1, u2, u3, u4, r_gauss, theta_gauss, r2_gauss, theta2_gauss, norm1, norm2, norm3, rans(4)
    call local_rng%initialize(1, random_seed(), 1, 1, ierr=ifail__)
    do i_ = 1, n_points
      call local_rng%next(rans)
      u1=rans(1); u2=rans(2); u3=rans(3); u4=rans(4)
      if(u1 .le. 0.d0) u1 = 1.d-10
      if(u3 .le. 0.d0) u3 = 1.d-10
      r_gauss = sqrt(-2.d0*log(u1)); theta_gauss = 2.d0*pi*u2
      norm1 = r_gauss*cos(theta_gauss); norm2 = r_gauss*sin(theta_gauss)
      r2_gauss = sqrt(-2.d0*log(u3)); theta2_gauss = 2.d0*pi*u4
      norm3 = r2_gauss*cos(theta2_gauss)
      positions(1,i_) = x0 + sigma_x_*norm1
      positions(2,i_) = y0 + sigma_y_*norm2
      positions(3,i_) = z0 + sigma_z_*norm3
    end do
  end function generate_3d_gaussian

end program stel_ex2_proj
