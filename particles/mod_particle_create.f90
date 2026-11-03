!> module to contain common tools for initialisation and destruction of particles
module mod_particle_create

  implicit none
   
  private
  public :: free_particle_indices, part_create_scheme, type_part_create_scheme, create_scheme_is_set

  !> holds the super particle creation scheme information for this action
  type :: type_part_create_scheme
    integer           :: supers_num    = -1     !< number of new superparticles initialised at each puff action
    real*8            :: supers_weight = -1.d0  !< aimed weight (no. real particles per superparticle) of the new superparticles initialised at each puff action
    real*8            :: supers_ratio  = -1.d0  !< fraction of the total number of superparticles allocated for this group (i.e. part_group_configs(i)%n_particles) to use for each puff action
    character(len=10) :: scheme        = "none" !< either "num", "weight" or "ratio"
    real*8            :: n_particles   = -1     !< total number of particles (of the group this event creates). Is used for 
  contains
    procedure :: supers_to_create
  end type

contains


!> check whether the create scheme has been set (.true.) or still has it default values (.false.)
function create_scheme_is_set(supers_num, supers_weight, supers_ratio) result(is_set)
  implicit none
  
  integer,                       intent(in)              :: supers_num
  real*8,                        intent(in)              :: supers_weight 
  real*8,                        intent(in)              :: supers_ratio  
  logical                                                :: is_set

  if (supers_num == -1.d0 .and. supers_weight == -1.d0 .and. supers_ratio == -1.d0) then
    is_set = .false.
  else
    is_set = .true.
  end if
  
end function create_scheme_is_set


!> checks the set parameters for the create scheme to determine the scheme, or stop for wrong input 
function part_create_scheme(supers_num, supers_weight, supers_ratio, n_particles, default, my_id, identifier) result(this)
  implicit none
  
  integer,                       intent(in)              :: supers_num
  real*8,                        intent(in)              :: supers_weight 
  real*8,                        intent(in)              :: supers_ratio  
  real*8,                        intent(in)              :: n_particles
  real*8,                        intent(in), optional    :: default       !< default for supers_ratio for this scheme
  integer,                       intent(in), optional    :: my_id         !< for printing errors
  character(len=*),              intent(in), optional    :: identifier    !< for tracing back the origin of the error
  type(type_part_create_scheme)                          :: this 
  
  integer :: id, n_create_schemes
  real*8  :: supers_ratio_default
  character(len=1000) :: identifier_local

  !setting values
  this%supers_num    = supers_num
  this%supers_weight = supers_weight
  this%supers_ratio  = supers_ratio
  this%n_particles   = n_particles

  !determining presets
  id = 0
  if(present(my_id)) id = my_id
  
  identifier_local = ""
  if(present(identifier)) identifier_local = identifier

  supers_ratio_default = 1.d-4
  if(present(default)) supers_ratio_default = default
  
  !> determine supers_create_scheme (which scheme is used to calculate supers_to_create) -----
  n_create_schemes = 0

  ! check that the supers_... settings are non-negative
  if (supers_num /= -1.d0) then 
    if (supers_num < 0) then
      if (id == 0) write(*,"(A,A,A)") "ERROR: ",trim(identifier_local),"%supers_num is negative"
      stop
    endif 
    this%scheme = "num"
    n_create_schemes = n_create_schemes + 1
  endif
  if (supers_weight /= -1.d0) then
    if (supers_weight < 0) then
      if (id == 0) write(*,"(A,A,A)") "ERROR: ",trim(identifier_local),"%supers_weight is negative"
      stop
    endif 
    this%scheme = "weight"
    n_create_schemes = n_create_schemes + 1
  endif
  if (supers_ratio /= -1.d0) then
    if (supers_ratio < 0) then
      if (id == 0) write(*,"(A,A,A)") "ERROR: ",trim(identifier_local),"%supers_ratio is negative"
      stop
    endif
    if (n_particles < 1.d0) then
      if (id == 0) write(*,"(A,A,A)") "ERROR: ",trim(identifier_local),"%n_particles is < 1"
      stop
    end if
    this%scheme = "ratio"
    n_create_schemes = n_create_schemes + 1
  endif

  ! check that only 1 types of supers_... settings are set, or else use default setting
  if (n_create_schemes > 1) then
    if (id == 0) then 
      write(*,"(A,A)") "ERROR: in create scheme ",trim(identifier_local)
      write(*,*) "  Only one type of supers_... can be used per creation event"
    endif
    stop
  else if (n_create_schemes == 0) then
    this%scheme = "ratio"
    this%supers_ratio = supers_ratio_default
    if (id == 0) then
      write(*,"(2A)") "WARNING: in create scheme ",trim(identifier_local)
      write(*,*) "  no scheme for determining the number of superparticles (supers_...)"
      write(*,*) "  has been assigned. Using the default"
      write(*,*) "  setting of supers_ratio = ", supers_ratio_default
    endif
  endif

end function part_create_scheme


!> determine the number of super particles to create for this action
function supers_to_create(this, my_id, weight) result(n_supers)
  implicit none
  class(type_part_create_scheme), intent(in) :: this
  integer,                        intent(in) :: my_id    !< for printing warning
  real*8,                         intent(in) :: weight   !< [weight] weight of real particles created, used for weight scheme
  
  integer                                    :: n_supers !< number of superparticles to create this creation action

  n_supers = 0
  if (trim(this%scheme) == "num")    n_supers = this%supers_num
  if (trim(this%scheme) == "weight") n_supers = nint((weight) / this%supers_weight)
  if (trim(this%scheme) == "ratio")  n_supers = nint(this%n_particles * this%supers_ratio)

  !> forces that at least one particle is puffed
  if (n_supers < 1) then
    if (my_id == 0) then
      write(*,*) "WARNING: The number of superparticles to be initialized for this "
      write(*,*) "  action is calculated to be less than 1. It will be overwritten to 1."
    endif
    n_supers = 1
  endif

end function supers_to_create


!> determines the indices of all free particles for a particles array (typically sim%particle_group(group_num)%particles)
subroutine free_particle_indices(part_arr, i_free, n_needed)
  use mod_particle_types, only: particle_base
  use phys_module, only: use_manual_random_seed
  !$ use omp_lib 

  implicit none
  
  class(particle_base), dimension(:), allocatable, intent(in)  :: part_arr          !< typically sim%particle_group(group_num)%particles
  integer,             dimension(:), allocatable,  intent(out) :: i_free            !< indices of free particles (size: n_free or n_needed if specified)
  integer,                           optional,     intent(in)  :: n_needed          !< number of free particles needed. If this is specified, routine returns first n_needed free particles instead 
  
  logical, dimension(:), allocatable :: is_free
  integer, dimension(:), allocatable :: i_free_tmp
  integer :: i, j, k, k_thread, n_part, n_free
  
  n_part = size(part_arr)

  ! Step 1: find which particles are free using a mask is_free
  allocate(is_free(n_part)) 

  ! might be replaced with omp workshare, or just the array expression.
  ! there is an issue with derived type arrays in gfortran though, and this works
  if(use_manual_random_seed) then
    !$ call omp_set_schedule(omp_sched_static,100)
  else
    !$ call omp_set_schedule(omp_sched_dynamic,100)
  end if
  !$omp parallel do default(none) shared(part_arr, n_part, is_free) &
  !$omp private(j) schedule(runtime)
  do j=1,n_part
    ! i_elm < 0 technically means the particle left the domain rather than that it is free, thus to easier catch bugs, only i_elm = 0 particles are considered free by default 
    !however for reg test purpose we need .le.
    !is_free(j) = part_arr(j)%i_elm .eq. 0 
    is_free(j) = part_arr(j)%i_elm .le. 0 
  end do
  !$omp end parallel do
  
  ! what is the function of this barrier?
  !$omp barrier
  
  n_free = count(is_free)
  allocate(i_free(n_free))
  
  ! Step 2: write their indices in an array
  k = 1

  ! omp atomic capture is not deterministic and thus can never satisfy any reg test
  ! if(use_manual_random_seed) then
  !   !$ call omp_set_schedule(omp_sched_static,1)
  ! else
  !   !$ call omp_set_schedule(omp_sched_dynamic,100)
  ! end if
  ! !$omp parallel do default(none) shared(is_free, i_free, n_part, k) &
  ! !$omp private(j, k_thread) schedule(runtime)
  do j=1,n_part
    if (is_free(j)) then
      ! !$omp atomic capture
      k_thread = k
      k = k+1
      ! !$omp end atomic
      i_free(k_thread) = j
    end if
  end do
  ! !$omp end parallel do

  ! Step 3: give first n_needed elements back if n_needed was specified, else return array with all free particle indices
  if(present(n_needed)) then
    if(n_free < n_needed) then
      write(*,*) "ERROR: More free particles needed than available, returning only available free particles (avail/needed)", n_free, n_needed
      ! then just send the full i_free
      return
    end if
    ! make i_free smaller
    i_free_tmp = i_free(1:n_needed)
    deallocate(i_free)
    i_free = i_free_tmp
  end if

end subroutine free_particle_indices

end module mod_particle_create
