!>#Example 1: single particle in JOREK field
!> This example follows a particle in a static JOREK equilibrium field
program ex1_stel
use particle_tracer
use mod_interp, only: interp_gvec

implicit none

interface
  subroutine find_RZ(node_list,element_list,R_find,Z_find,R_out,Z_out,ielm_out,s_out,t_out,ifail)
    use data_structure
    type (type_node_list), intent(in)    :: node_list
    type (type_element_list), intent(in) :: element_list
    real*8, intent(in)     :: R_find, Z_find
    real*8, intent(out)    :: R_out,Z_out,s_out,t_out
    integer, intent(inout) :: ielm_out
    integer, intent(out)   :: ifail
  end subroutine find_RZ
end interface

! 1. Set up the simulation variables containing
!    sim: particles, time, and io.
!    events: halting points for the pushers and actions to run.
real*8 :: timesteps(1) = [1d-4]
class(*), pointer :: p
integer :: i, j, k, n_steps, n_lost
real*8  :: target_time, time
real*8 :: E(3), B(3), rz_old(2), st_old(2), psi, U
type(read_jorek_fields_interp_linear) :: fieldreader
character(len=1024) :: filename
integer :: i_elm, ifail, i_elm_old
real*8  :: dummy, DUMMY_R, DUMMY_Z, s, t, s_norm
real*8, parameter  :: R=2.04, Z=0.0, phi=0.0
integer, parameter :: n_particles = 1

! 2. Allocate a group.
call sim%initialize(num_groups=1)

! 3. Set up the field reader
fieldreader = read_jorek_fields_interp_linear(basename='jorek', i=-1)
call with(sim, fieldreader)

! 4. Allocate a particle of type particle_kinetic_leapfrog
allocate(particle_kinetic_leapfrog::sim%groups(1)%particles(n_particles))
!allocate(particle_fieldline::sim%groups(1)%particles(n_particles))

! 5. Get particle location in element, s, t, space
call find_RZ(sim%fields%node_list, sim%fields%element_list,R,Z,DUMMY_R,DUMMY_Z,i_elm,s,t,ifail)
write(*,*) 'Initial particle position: ', R, Z, phi

! 6. Initialize the particle.
!    This should usually be done by a dedicated initialization routine
!    or by reading existing files.
select type (p => sim%groups(1)%particles(1))
type is (particle_kinetic_leapfrog)
  p%x = [R,Z,phi]
  p%v = [1.55d6, 0.d0, 1.55d6]  ! He-4 at ~100 keV: gyroradius ~1.4 cm, fits in W7-A (a=9 cm)
  p%q = 2_1
  p%i_elm = i_elm
  p%st = [s, t]
type is (particle_fieldline)
  p%x = [R,Z,phi]
  p%i_elm = i_elm
  p%st = [s, t]
  E = 0.0
  B = 0.0
end select
sim%groups(1)%mass = 4.0
write(*,*) "Print kinetic energy at start of simulation time"
select type (p => sim%groups(1)%particles(1))
  type is (particle_kinetic_leapfrog) 
    write(*,*) norm2(p%v)
end select

! 7. Set an event to stop the simulation.
events  = [event(stop_action(), start=20.0d0)]

! 8. Check whether all events conform to the requested timestep
call check_and_fix_timesteps(timesteps, events)

! 9. Run first events
call with(sim, events, at=0.d0)

! 10. Set dpsi_dt to be zero as treating JOREK as an equilibrium field
call sim%fields%set_flag_dpsidt(.true.)

! 11. Loop until we the simulation requests a stop
write(*,*) "Start tracing"
do while (.not. sim%stop_now)
  ! 11.1 Find out which events are next and when they will run
  target_time = next_event_at(sim, events)
  ! 11.2 Loop over all particle groups
  do i=1,1
    n_steps = nint((target_time - sim%time)/timesteps(i))
    write(*,*) "Target time, n_steps", target_time, n_steps
    ! 11.3 Loop first over particles, and then over how many steps we can take
    !$omp parallel do default(private) shared(sim, n_steps, timesteps, i, p) &
    !$omp reduction(+:n_lost)
    do j=1,size(sim%groups(i)%particles)
      write(filename, '(A12,I1,A4)') 'particle_xyz', i, '.dat'
      ! 11.4 copy the particle j in the i-th groups to the dummy structure particle
      !< get particle base attributes from sim%groups(i)%particles(j)
      open(21,file=filename)
      do k=1,n_steps
        t = sim%time + k*timesteps(i)
        ! 11.5 integrating the particle with corresponding time evolution scheme
        select type (p => sim%groups(1)%particles)
        type is (particle_kinetic_leapfrog)

          call sim%fields%calc_EBpsiU(t, p(j)%i_elm, p(j)%st, p(j)%x(3), E, B, psi, U)
          rz_old    = p(j)%x(1:2)
          st_old    = p(j)%st
          i_elm_old = p(j)%i_elm
          
          call boris_push_cylindrical(p(j), m=sim%groups(i)%mass, E=E, B=B, dt=timesteps(i)*sim%t_norm)
          call find_RZ_nearby(sim%fields%node_list, sim%fields%element_list, rz_old(1), rz_old(2), st_old(1), st_old(2), i_elm_old, p(j)%x(1), p(j)%x(2), p(j)%st(1), p(j)%st(2), p(j)%i_elm, ifail, p(j)%x(3))
        
          if (p(j)%i_elm .le. 0) exit  ! particle lost
          call interp_gvec(sim%fields%node_list, sim%fields%element_list, p(j)%i_elm, 4, 1, 1, p(j)%st(1), p(j)%st(2), s_norm, dummy, dummy, dummy, dummy, dummy)
          if (mod(k, 10) .eq. 0) write(21, "(15e16.8)") p(j)%x, p(j)%v, B, E, p(j)%st(1), p(j)%st(2), s_norm
        type is (particle_fieldline)
          call field_line_runge_kutta_fixed_dt_push_jorek(sim%fields, p(j), sim%time, t)
          call interp_gvec(sim%fields%node_list, sim%fields%element_list, p(j)%i_elm, 4, 1, 1, p(j)%st(1), p(j)%st(2), s_norm, dummy, dummy, dummy, dummy, dummy)
          if (mod(k, 1) .eq. 0) write(21, "(6e16.8)") p(j)%x, p(j)%st(1), p(j)%st(2), s_norm
        end select      
        
        if (sim%groups(i)%particles(j)%i_elm .le. 0) n_lost = n_lost + 1
      end do ! steps
      close(21)
    end do ! particles
    !$omp end parallel do
    write(*,*) "number/% of lost particles: ", n_lost, n_lost/n_particles*100.0
  end do ! groups

  ! 11.6 Update the current time and run events
  sim%time = target_time
  call with(sim, events, at=sim%time)
end do

! 11. Print some info of the particle
write(*,*) "Print kinetic energy at end of simulation time"
select type (p => sim%groups(1)%particles(1))
  type is (particle_kinetic_leapfrog) 
    write(*,*) norm2(p%v)
end select
call sim%finalize
end program ex1_stel
