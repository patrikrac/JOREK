!> Module for puffing gas into the plasma
!> This module will create new particles at the locations where gas valves will be.

module mod_particle_puffing
  use mod_edge_elements
  use mod_io_actions, only: io_action
  use mod_sampling
  use mod_particle_types
  use constants, only: TWOPI, K_BOLTZ, ATOMIC_MASS_UNIT
  use mod_rng, only: type_rng, setup_shared_rngs
  use mod_boundary, only: wall_normal_vector
  use mod_atomic_elements 
  use mod_particle_sim
  use mod_event
  use mod_find_rz_nearby, only: find_rz_nearby

  implicit none

  private
  public  :: particle_puffing

  ! Extend type
  type, extends(io_action) :: particle_puffing
   
    class(type_rng), dimension(:), allocatable :: rng  !< one RNG per openmp thread
   
    ! number of simulation particles/s to puff across all processes
    integer :: n_puff = -1 
    ! Average fueling rate: 9.7d22; max fueling rate 18d22
    real*8  :: puffing_rate = -1.d0
    real*8  :: R = -1.d0, Z = -1.d0, phi = -1.d0
    real*8  :: valve_r = -1.d0  !< radius of gas valve
    
    !box volume puff, define 4 RZ points to determine volume
    real*8  :: poly_R(4) = -1.d0
    real*8  :: poly_Z(4) = -1.d0
    logical :: boxpuff = .false.
    
    !Time dependent puffing
    logical :: puff_t_dependent = .false.
    real*8  :: puffing_rate_start = 0.d0
    real*8  :: t_puff_start = 0.d0 !< defined in JOREK time units
    real*8  :: t_puff_slope = 0.d0 !<defined in SI
    
  contains
    procedure :: do => do_particle_puffing
  end type particle_puffing

  interface particle_puffing
    module procedure new_particle_puffing
  end interface particle_puffing
contains

!> To do: add generalization to choose group number.
function new_particle_puffing(n_puff, puffing_rate, valve_r, R, Z, phi, rng, seed, puff_t_dependent,t_puff_start,t_puff_slope,puffing_rate_start,poly_R,poly_Z,boxpuff) result(new)
  use mod_pcg32_rng,   only: pcg32_rng
  use mod_random_seed, only: random_seed
  
  type(particle_puffing)    :: new

  integer, intent(in)           :: n_puff
  real*8, intent(in)            :: puffing_rate
  real*8, intent(in)            :: valve_r
  real*8, intent(in)            :: R, Z
  real*8, intent(in), optional  :: phi ! If no phi is given axisymmetric puffing will be excecuted.
  logical, intent(in), optional :: puff_t_dependent
  real*8, intent(in), optional  :: t_puff_start,t_puff_slope
  real*8, intent(in), optional  :: puffing_rate_start 
  
  real*8, intent(in), optional  :: poly_R(4)
  real*8, intent(in), optional  :: poly_Z(4)
  logical, intent(in), optional :: boxpuff
  
  class(type_rng), intent(in), optional :: rng !< random number generator to use (deafult PCG32)
  integer, intent(in), optional         :: seed !< Seed for the RNG (default random_seed() on my_id + bcast)
  integer                               :: my_seed
  
  new%n_puff       = n_puff
  new%puffing_rate = puffing_rate
  new%R            = R
  new%Z            = Z
  new%valve_r      = valve_r
  if (present(phi))  new%phi = phi
  
  if (present(puff_t_dependent))  new%puff_t_dependent  = puff_t_dependent
  if (present(t_puff_start)) new%t_puff_start = t_puff_start
  if (present(t_puff_slope)) new%t_puff_slope = t_puff_slope
  if (present(puffing_rate_start)) new%puffing_rate_start = puffing_rate_start

  if (present(poly_R)) new%poly_R = poly_R
  if (present(poly_Z)) new%poly_Z = poly_Z
  if (present(boxpuff)) new%boxpuff = boxpuff
  !> allocate random seed for sampling
  if (present(seed)) then
    my_seed = seed
  else
    my_seed = random_seed()
  end if
  if (present(rng)) then
    call setup_shared_rngs(n_dim=3, seed=my_seed, rng_type=rng, rngs=new%rng)
  else
    ! default to pcg32_rng for reflection
    call setup_shared_rngs(n_dim=3, seed=my_seed, rng_type=pcg32_rng(), rngs=new%rng)
  end if

end function new_particle_puffing

!> Actually puff gass
subroutine do_particle_puffing(this,sim, ev)
  use mpi_mod
  use phys_module, only: tstep, central_mass, central_density
  use constants, only: ATOMIC_MASS_UNIT, MU_ZERO
  ! !$ use omp_lib

  class(particle_puffing) , intent(inout) :: this
  type(particle_sim), intent(inout)       :: sim
  type(event), intent(inout), optional    :: ev 

  integer :: ierr,i_scalar, n_free, j, k, n_group, i_elm, i_elm_new, ifail, i_p, to_puff, n_puff_local,i_rng
  logical, allocatable, dimension(:) :: is_free
  integer, allocatable, dimension(:) :: i_free
  real*8  :: tstep_fluid_si, c, R, Z, phi, s, t
  real*8  :: R_new, Z_new, s_new, t_new, r_valve, theta
  real*8  :: vector_normal(3), u(5)
  real*8  :: puffing_rate_t !< possibly time dependent fueling rate

  integer ::    puffed_this_step_local, all_puffed_this_step
  real*8  ::    puff_weight_local, all_puff_weight

  tstep_fluid_si = tstep*sqrt((MU_ZERO * central_mass * ATOMIC_MASS_UNIT * CENTRAL_DENSITY * 1.d20))

  if (sim%my_id .eq. 0) write(*,*) "Started puffing!"
  
  if (this%n_puff .le. -1.d-6) then ! 0.d0
    if (sim%my_id .eq. 0) write(*,*) 'No puf quota set, exiting. --- n_puff == 0 this will now stop the program'
    stop
  end if
  
  n_puff_local = this%n_puff / sim%n_cpu !n_puff local is the amount of superparticles that will be puffed per MPI process.
 
!============== Finding free particles !< make into a function?
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
  
  
  n_group = 1   ! Puffing Hydrogen (or actually the element at groups(1)) only
  ! Assuming the incoming gas at T=300C and a diatomic gas
  c = sqrt((7.d0/5.d0)*(300.d0+273.d0)*K_BOLTZ/(2.d0*sim%groups(n_group)%mass*ATOMIC_MASS_UNIT))
  if (.not. this%boxpuff) then
    call find_RZ(sim%fields%node_list, sim%fields%element_list, this%R, this%Z, R, Z, &
           i_elm, s, t ,ifail)
           
  else
    call find_RZ(sim%fields%node_list, sim%fields%element_list, sum(this%poly_R(1:2))/2.d0, sum(this%poly_Z(1:2))/2.d0, R, Z, &
           i_elm, s, t ,ifail)
  endif
  if (ifail .ne. 0) then
    if (sim%my_id .eq. 0) write(*,*) "Warning: The valve location for puffing could not be found, maybe it was placed outside of the grid?"
    stop
  end if

  vector_normal = wall_normal_vector(sim%fields%node_list, sim%fields%element_list, &
          i_elm, s, t)

!------------- Decide how many superparticles to initiate     
!Adjust amount of superparticles + fueling rate if we use time dependent puffing 
  if (this%puff_t_dependent) then
    to_puff        = n_puff_local !int( maxval((/ time_dependent_puff(real(n_puff_local,8)       ,sim%time, this%t_puff_start,this%t_puff_slope) ,10.d0 /)))
    puffing_rate_t = time_dependent_puff(this%puffing_rate ,sim%time, this%t_puff_start,this%t_puff_slope, this%puffing_rate_start)
   
    if (sim%my_id .eq.0) write(*,"(A,g12.4,A,g12.4, A)") "Actual puffing rate at time t:", sim%time, " is puffing_rate_t:",puffing_rate_t, "atoms/s"
   
    if (to_puff .ge. n_free) then
      write(*,*) "Warning could not puff the requested amount."
      to_puff = n_free
    end if
  else
    puffing_rate_t = this%puffing_rate
    if (n_puff_local .ge. n_free) then
      write(*,*) "Warning could not puff the requested amount."
      to_puff = n_free
    else
      to_puff = n_puff_local
      if (sim%my_id .eq.0) write(*,"(A,g12.4, A)") "puffing_rate:",puffing_rate_t, "atoms/s"
    end if
  end if !< time dependent puffing
!-------------  
  
  
  puffed_this_step_local = 0
  puff_weight_local      = 0.d0
  select type (pa => sim%groups(1)%particles)
  type is (particle_kinetic_leapfrog)
  !> To do: Parallellize loop 
  !> This loop initializes the to be puffed particles and places then in the computational domain.
  !> It counts the amount of marker particles and weight that was initialized.
  !> This loop can be OMP parallel. there was a small bug in the OMP implementation.
  !> The almost done loop is written below.
  !> reduction of puffed_this_step_local,puff_weight_local for diagnostics
  !>
  ! #ifdef __GFORTRAN__
  !  !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
  ! #else
  ! !$omp parallel do default(shared) &
  ! !$omp schedule(dynamic,10) &
  ! !$omp shared(sim, pa, this,i_free,c, vector_normal,                       &
  ! !$omp   to_puff,n_puff_local, tstep_fluid_si,puffing_rate_t )                        &
  ! !$omp private(i_p, i_rng, j,k,u , R,Z,s,t,R_new,Z_new,s_new,t_new,     &
  ! !$omp  i_elm,i_elm_new,r_valve, theta,                                         &
  ! !$omp ifail)                                                                    &
  ! !$omp reduction(+:puffed_this_step_local,puff_weight_local)
    do j = 1, to_puff
      i_p = i_free(j)
      do 
    
        !      !$ i_rng = omp_get_thread_num()+1
        call this%rng(1)%next(u) !rng(1)
        if (.not. this%boxpuff) then
          r_valve = this%valve_r*sample_piecewise_linear(2, [0.d0, 1.d0], [1.d0, 0.d0], u(1))
          theta = TWOPI * u(2)
          R_new = this%R + r_valve * cos(theta)
          Z_new = this%Z + r_valve * sin(theta)
        else
          s = u(1)
          t = u(2)
          R_new = this%poly_R(1)*(1.d0-s)*(1.d0-t) + this%poly_R(2)*s*(1.d0-t) + this%poly_R(3)*(1.d0-s)*t + this%poly_R(4)*s*t
          Z_new = this%poly_Z(1)*(1.d0-s)*(1.d0-t) + this%poly_Z(2)*s*(1.d0-t) + this%poly_Z(3)*(1.d0-s)*t + this%poly_Z(4)*s*t
        endif
    
        call find_RZ_nearby(sim%fields%node_list, sim%fields%element_list, R, Z, s, t, i_elm, &
        R_new, Z_new, s_new, t_new, i_elm_new, ifail)
        if (ifail .ge. 0) exit
      end do
      R     = R_new
      Z     = Z_new
      s     = s_new
      t     = t_new
      i_elm = i_elm_new
      if (this%phi .lt. 0.d0) then
        pa(i_p)%x(3) = TWOPI*u(3)
      else 
        pa(i_p)%x(3) = this%phi 
      end if
      pa(i_p)%x(1:2)  = [R, Z]
      pa(i_p)%st(1:2) = [s, t]
      pa(i_p)%i_elm   = i_elm
      pa(i_p)%weight  = real(1.d0/n_puff_local) * tstep_fluid_si * puffing_rate_t 
      pa(i_p)%v       = c * sample_cosine(u(4:5), vector_normal)   
      pa(i_p)%q       = 0_1
      if (sim%groups(1)%particles(i_p)%weight  .le. 1.d-2) then ! if the weight is too low. 
        sim%groups(1)%particles(i_p)%i_elm = 0
        cycle       
      end if
    puffed_this_step_local = puffed_this_step_local+1
    puff_weight_local      = puff_weight_local + pa(i_p)%weight 
    end do
  ! !$omp end parallel do  
  class default
    write(*,*) 'Particle type not implemented for gas fueling.'
    stop
  end select

! puffed_this_step_local
call MPI_REDUCE(puffed_this_step_local,all_puffed_this_step,1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)   
call MPI_REDUCE(puff_weight_local,all_puff_weight,1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)  
if (sim%my_id .eq. 0) then
write(*,'(A60,I7,E14.6)') "Superparticles, weight puffed this puffing action action = ", all_puffed_this_step, all_puff_weight
endif
  
  
end subroutine do_particle_puffing

pure function time_dependent_puff(max_puff,time, t_puff_start,t_puff_slope, min_puff) result(to_puff)
real*8,intent(in)   :: max_puff, min_puff
real*8              :: to_puff
real*8,intent(in)    :: t_puff_start,t_puff_slope
real*8,intent(in)    :: time

if (time-(t_puff_start+t_puff_slope) .ge. 0.d0) then
  to_puff = max_puff
elseif (time-t_puff_start .ge. 0.d0) then
  to_puff = min_puff+ (max_puff -min_puff) * (time-t_puff_start)/(t_puff_slope)  
else
    to_puff = min_puff !default = 0.d0
endif
end function time_dependent_puff

end module