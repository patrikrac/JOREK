!> Module for sorting particles based on their location
!> Currently only contains an implementation for sorting particles into their elements
module mod_particle_sorting
  use mod_particle_types
  use phys_module, only: use_manual_random_seed
  use mpi
  
  !$ use omp_lib

  implicit none
   
  private
  public :: indices_in_elm, sort_particles_in_elm

  !> an object containing the array of particle indices of particles in this element 
  !> s.t. an array of these objects can store all particle indices as array per element
  type :: indices_in_elm
    integer, dimension(:), allocatable :: pa_ind
  end type

contains

!> Deterministically sort the particle indices (of the particle array pa given as input)
!> into arrays by their element number
!> Also return the number of particles in each element (as that is a useful intermediate result)
subroutine sort_particles_in_elm(pa, n_elm, sorted_ind_arr, pa_in_elm_arr)
  implicit none

  class(particle_base),  dimension(:), allocatable, intent(in)  :: pa             !< particle array to be sorted (n_pa)
  integer,                                          intent(in)  :: n_elm          !< number of elements in the grid
  type(indices_in_elm),  dimension(:), allocatable, intent(out) :: sorted_ind_arr !< object containing all particle indices as arrays per element number (n_elm)
  integer,               dimension(:), allocatable, intent(out) :: pa_in_elm_arr  !< number of particles in each element (n_elm)
  
  ! arrays used for determining sorted_ind_arr
  integer, dimension(:),       allocatable :: pa_elm_arr        !< precalculated pa(:)%i_elm for faster masking over i_elm  (n_pa)
  integer, dimension(:,:),     allocatable :: pa_in_thread_arr  !< number of particles in the thread for this element (n_elm, n_thread)
  integer, dimension(:,:),     allocatable :: offset_thread_arr !< offset of this thread's contribution to sorted_ind_arr for this element (n_elm, n_thread)
  integer, dimension(:,:),     allocatable :: i_loc_thread_arr  !< ith particle index insertion into sorted_ind_arr for this element by this thread (n_elm, n_thread)

  integer :: i, i_elm, i_loc
  integer :: i_thread, n_thread

  ! --- start of code
  
  !initialisation
  i_thread = 1 !default if not using OMP
  n_thread = 1
  !$ n_thread = omp_get_max_threads()
  
  allocate(pa_elm_arr(size(pa)),source=0)
  allocate(pa_in_elm_arr(n_elm),source=0)
  allocate(pa_in_thread_arr(n_elm,n_thread),source=0)
  
  !find out how many particles per element and thread
  !$omp parallel do default(none)  &
  !$omp shared(pa,pa_elm_arr,pa_in_thread_arr)      &
  !$omp private(i_elm, i_thread)   &
  !$omp schedule(static,100)
  do i=1,size(pa)
    !$ i_thread = omp_get_thread_num()+1
    i_elm = pa(i)%i_elm
    if(i_elm < 1) cycle
    pa_elm_arr(i) = i_elm
    pa_in_thread_arr(i_elm,i_thread) = pa_in_thread_arr(i_elm,i_thread) + 1
  end do
  !$omp end parallel do

  !find out how many particles per element total, and getting the offset of each thread 
  !so that each thread writes to its own part of the sorted_ind_arr(i_elm)%pa_ind(:) array later on
  if(allocated(offset_thread_arr)) deallocate(offset_thread_arr)
  allocate(offset_thread_arr(n_elm,n_thread))
  !$omp parallel do default(none)  &
  !$omp shared(pa_in_elm_arr,pa_in_thread_arr,offset_thread_arr,n_thread,n_elm) &
  !$omp private(i_thread)
  do i_elm=1,n_elm
    pa_in_elm_arr(i_elm) = sum(pa_in_thread_arr(i_elm,:))
    
    !the offset must be the sum over the particles in that element of lower thread numbers, we can determine this iteratively from the previous offset
    offset_thread_arr(i_elm,1) = 0
    do i_thread=2,n_thread
      offset_thread_arr(i_elm,i_thread) = offset_thread_arr(i_elm,i_thread - 1) + pa_in_thread_arr(i_elm,i_thread - 1)
    end do
  end do
  !$omp end parallel do

  !allocating sorted_ind_arr object
  if(.not. allocated(sorted_ind_arr)) allocate(sorted_ind_arr(n_elm))
  !$omp parallel do default(none) &
  !$omp shared(pa_in_elm_arr, sorted_ind_arr, n_elm)
  do i_elm=1,n_elm
    if(allocated(sorted_ind_arr(i_elm)%pa_ind)) deallocate(sorted_ind_arr(i_elm)%pa_ind) ! this can be done here as we know size(sorted_ind_arr)=n_elm is fixed
    allocate(sorted_ind_arr(i_elm)%pa_ind(pa_in_elm_arr(i_elm)))
  end do
  !$omp end parallel do

  !filling sorted_ind_arr object
  if(allocated(i_loc_thread_arr)) deallocate(i_loc_thread_arr)
  allocate(i_loc_thread_arr(n_elm,n_thread),source=1) !index 1
  !$omp parallel do default(none)                     &
  !$omp shared(pa_elm_arr, offset_thread_arr, i_loc_thread_arr, sorted_ind_arr) &
  !$omp private(i_elm, i_loc, i_thread) &
  !$omp schedule(static,100)
  do i=1,size(pa_elm_arr)
    !$ i_thread = omp_get_thread_num()+1
    i_elm = pa_elm_arr(i)
    if(i_elm < 1) cycle
    !find out where to insert this particles' index in the element array (i_loc) from the thread offset + the number already filled in by this thread
    i_loc = offset_thread_arr(i_elm,i_thread) + i_loc_thread_arr(i_elm,i_thread) 
    !update the number of indices filled in by this thread for this element
    i_loc_thread_arr(i_elm,i_thread) = i_loc_thread_arr(i_elm,i_thread) + 1
    !insert the particle index in the element array (at the right spot i_loc)
    sorted_ind_arr(i_elm)%pa_ind(i_loc) = i
  end do
  !$omp end parallel do

end subroutine sort_particles_in_elm

end module mod_particle_sorting
