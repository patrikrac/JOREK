!> Module for routines doing general tasks in the output file such as general diagnostics (non-interaction specific) and writing new header lines
module mod_output_file_routines
  use mod_particle_sim
  use mod_particle_types
  use constants, only: ATOMIC_MASS_UNIT, SPEED_OF_LIGHT
  use mod_event, only: mpi_minmeanmax
  use phys_module, only: n_part_groups_max, n_part_groups
  use mpi
  !$ use omp_lib

  implicit none
  private
  public write_to_outputfile

  integer,parameter :: num_tracked=7        !< number of tracked variables (see conserv_objs)

  !> timer object to time anything
  type :: timer
    real*8 :: last_time = -1.d0             !< last time the timer was called
    character(len=100) :: output_line(2)="" !< what to say exactly in the output file when writing the time spent
    logical :: started=.false.              !< whether the timer has started running (.true.) or not (.false.)
  contains
    procedure :: write => write_cpu_time
  end type timer

  real*8  :: conserv_objs(n_part_groups_max,num_tracked)=0.d0 !< for every particle group, contains: superparticles, real particles, momentum (3 directions), energy
  logical :: next_call_write_conserv=.true.                   !< whether the coming block should have conservation information (.true.) or not (.false.)
  logical :: next_call_write_timing =.true.                   !< whether the coming block should be timed (.true.) or not (.false.)

  type(timer) :: time_block, time_main_loop
contains

!> Write the new header line for the coming block, with possibly extra general output of the previous block such as
!> how much computational time the block previous block took, and the changes in particles/momentum/energy of a tracked species
subroutine write_to_outputfile(sim,what,next_block_write_conserv,next_block_write_timing,call_main_loop_timer)
  implicit none

  type(particle_sim), intent(in) :: sim
  character(len=*),   intent(in) :: what
  logical, optional,  intent(in) :: next_block_write_conserv !< whether the coming block should have conservation information (.true.) or not (.false.). Default behaviour: .true.
  logical, optional,  intent(in) :: next_block_write_timing  !< whether the coming block should be timed (.true.) or not (.false.). Default behaviour: .true.
  logical, optional,  intent(in) :: call_main_loop_timer       !< whether this is the start of 

  real*8  :: now, mmm_main(3)
  logical :: call_main_loop_timer_loc
  integer :: group_num

  if(next_call_write_conserv) then
    do group_num=1,n_part_groups
      if (sim%groups(group_num)%do_conservation_checks) call conservation_block(sim,group_num)
    enddo
  end if
  if(next_call_write_timing) then
    if(.not. time_block%started) then
      ! setting up the timer
      time_block%output_line(1) = "block"
    endif
    call time_block%write(sim)
  endif

  if(present(next_block_write_conserv)) then
    next_call_write_conserv=next_block_write_conserv
  else
    next_call_write_conserv=.true.
  endif
  if(present(next_block_write_timing)) then
    next_call_write_timing=next_block_write_timing
  else
    next_call_write_timing=.true.
  endif

  call_main_loop_timer_loc = .false.
  if(present(call_main_loop_timer)) call_main_loop_timer_loc = call_main_loop_timer
  if(call_main_loop_timer_loc) then
    if(.not. time_main_loop%started) then
      ! setting up the timer
      time_main_loop%output_line(1) = "===== Main loop iteration"
      time_main_loop%output_line(2) = "====="
    endif
    call time_main_loop%write(sim)
  endif

  if(sim%my_id .ne. 0) return

  write(*,'(A100)') "===================================================================================================="
  write(*,"(2X,A)") what
  write(*,'(A100)') "===================================================================================================="

end subroutine


!> keeps track of global superparticle, particle, momentum and energy balances for 1 group,
!> and writes the change and new values to the logfile, and times the previous block
subroutine conservation_block(sim,group_num)
  implicit none
  
  class(particle_sim), target, intent(in) :: sim
  integer,                     intent(in) :: group_num

  real*8    :: old(num_tracked), diff(num_tracked)
  integer   :: j, ierr
  real*8    :: mass, gamma_m
  real*8    :: particles_remaining, particles_elm_lt0, momentum_remaining(3), energy_remaining, all_particles, all_momentum(3), all_energy, all_elm_lt0
  integer   :: superparticles_remaining,all_superparticles

  particles_remaining = 0.d0
  particles_elm_lt0   = 0.d0
  momentum_remaining  = 0.d0
  energy_remaining    = 0.d0
  superparticles_remaining = 0

  mass = sim%groups(group_num)%mass * ATOMIC_MASS_UNIT !< particle mass [kg]

  select type (particles => sim%groups(group_num)%particles)
  type is (particle_kinetic_leapfrog)
#ifdef __GFORTRAN__
    !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
    !$omp parallel do default(none)  &
    !$omp shared(sim, mass)    &
    !$omp private(j)                 &
#endif
    !$omp reduction(+:particles_remaining, particles_elm_lt0, momentum_remaining, energy_remaining,superparticles_remaining)
      do j=1,size(particles,1)

        if (particles(j)%i_elm .lt. 0) particles_elm_lt0 = particles_elm_lt0 + particles(j)%weight


        if (particles(j)%i_elm .le. 0) cycle

        particles_remaining = particles_remaining + particles(j)%weight
        momentum_remaining  = momentum_remaining  + particles(j)%weight * particles(j)%v * mass
        energy_remaining    = energy_remaining    + particles(j)%weight * 0.5d0 * mass * dot_product(particles(j)%v,particles(j)%v)
        superparticles_remaining = superparticles_remaining + 1

      enddo !j
    !omp end parallel do

  type is (particle_kinetic_relativistic)
#ifdef __GFORTRAN__
    !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
    !$omp parallel do default(none)   &
    !$omp shared(sim, mass)     &
    !$omp private(gamma_m,j)          &
#endif
    !$omp reduction(+:particles_remaining, particles_elm_lt0, momentum_remaining, energy_remaining,superparticles_remaining)
      do j=1,size(particles,1)

        !< account for lost particles
        if (particles(j)%i_elm .lt. 0) particles_elm_lt0 = particles_elm_lt0 + particles(j)%weight
        if (particles(j)%i_elm .le. 0) cycle

        !< account for remaining (super)particles
        superparticles_remaining = superparticles_remaining + 1
        particles_remaining      = particles_remaining + particles(j)%weight

        !< calculate Lorentz factor
        gamma_m = sqrt(mass**2 + dot_product(particles(j)%p, particles(j)%p)*ATOMIC_MASS_UNIT**2/SPEED_OF_LIGHT**2)

        momentum_remaining = momentum_remaining + particles(j)%weight * particles(j)%p * ATOMIC_MASS_UNIT
        energy_remaining   = energy_remaining   + particles(j)%weight *(gamma_m - mass*ATOMIC_MASS_UNIT) * SPEED_OF_LIGHT**2

      enddo !j
    !omp end parallel do

  class default
      if(sim%my_id == 0) write(*,*) "conservation_block() only implemented for particle_kinetic_leapfrog and particle_kinetic_relativistic, not for particle type of group ",sim%groups(group_num)%id
      return
  end select

  call MPI_REDUCE(particles_remaining, all_particles,         1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(particles_elm_lt0,   all_elm_lt0,           1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(momentum_remaining,  all_momentum,          3, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(energy_remaining,    all_energy,            1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_REDUCE(superparticles_remaining,all_superparticles,1, MPI_INTEGER,          MPI_SUM, 0, MPI_COMM_WORLD, ierr) 
  
  if (sim%my_id .eq. 0) then
    old = conserv_objs(group_num,:)

    conserv_objs(group_num,1:3) = [real(all_superparticles),all_particles, all_elm_lt0]
    conserv_objs(group_num,4:6) = all_momentum
    conserv_objs(group_num,7)   = all_energy

    diff = 0.d0
    diff = conserv_objs(group_num,:) - old
    
    write(*,"(3A)") "conservation checks for group with id: ",sim%groups(group_num)%id," -------------------------------------------------------------------"
    write(*,"(A,7A15)") "qty: ", "superparticles", "particles", "w ielm < 0", "momentum R", "momentum Z", "momentum phi", "energy [J]"
    write(*,"(A,7es15.5)") "diff ",diff
    write(*,"(A,7es15.5)") "new  ",conserv_objs(group_num,:)
  endif !(sim%my_id .eq. 0)

end subroutine conservation_block

!> determines the cpu time spent since the last time this function was called
!> and writes this to the output file
subroutine write_cpu_time(this, sim)
  implicit none
  
  class(timer),                intent(inout) :: this
  class(particle_sim), target, intent(in)    :: sim

  real*8 :: now, mmm(3)
  logical :: has_omp
  has_omp = .false.
  !$ has_omp = .true.
  ! the !$ lines are executed only if omp is used, they are not actual comments
  
  if(has_omp) then
    !$ now = omp_get_wtime()
  else
    call cpu_time(now)
  endif
  if(sim%n_mpi > 1) then
    mmm = mpi_minmeanmax(now - this%last_time)
  else
    mmm(:) = now - this%last_time
  endif
  this%last_time = now

  ! don't write timing information the first time around, because it makes no sense
  if(.not. this%started) then
    this%started=.true.
    return
  end if

  if(sim%my_id == 0) write(*,"(2A,3f17.5,2A)") trim(this%output_line(1))," done in (min/mean/max) ", mmm, " s ",trim(this%output_line(2))
end subroutine write_cpu_time

end module mod_output_file_routines
