!> Module containing type for multi-dimensional (Q)RNGs
!> composited of interspersing output from two rngs
module mod_mix_rng
  use mod_rng
  use phys_module, only: use_fixed_rng_value, fixed_rng_value
  implicit none
  private
  public mix_rng

  type, extends(type_rng) :: mix_rng
    class(type_rng), allocatable :: rng_a
    class(type_rng), allocatable :: rng_b
    logical, dimension(:), allocatable :: use_a !< if not, use b
    contains
      procedure :: initialize => init_all
      procedure :: next => next_all
      procedure :: jump_ahead => jump_ahead_all
  end type mix_rng

  interface mix_rng
    module procedure new_mix_rng
  end interface mix_rng
    

contains

  function new_mix_rng(rng_a, rng_b, use_a)
    class(type_rng), intent(in) :: rng_a, rng_b
    logical, dimension(:), intent(in) :: use_a
    type(mix_rng) :: new_mix_rng
    allocate(new_mix_rng%rng_a, source=rng_a)
    allocate(new_mix_rng%rng_b, source=rng_b)
    new_mix_rng%use_a = use_a
  end function new_mix_rng

  subroutine init_all(rng, n_dims, seed, n_streams, i_stream, ierr, round_off_n_streams_in)
    use mpi_mod
    class(mix_rng), intent(inout) :: rng
    integer, intent(in)  :: n_dims !< Dimension of the generated output vector
    integer, intent(in)  :: seed !< Seed for the RNG if required
    integer, intent(in)  :: n_streams !< Number of output streams needed
    integer, intent(in)  :: i_stream !< Index of rng output stream (1<=i_stream<=n_streams)
    logical, intent(in),  optional :: round_off_n_streams_in !< If true n_streams is rounded-off to 2**ceil
    integer, intent(out), optional :: ierr !< Error code. If present, return on error, otherwise call mpi_abort
    integer :: mpi_err
    logical :: round_off_n_streams_loc
    round_off_n_streams_loc = .false.
    if(present(round_off_n_streams_in)) round_off_n_streams_loc = round_off_n_streams_in

    if (.not. allocated(rng%use_a)) then
      write(*,*) "You need to set the use_a array to indicate which rng you would like to use for which position"
      if (present(ierr)) then
        ierr = 5
      else
        call MPI_ABORT(MPI_COMM_WORLD, ierr, mpi_err)
      end if
      return
    else
      if (n_dims .ne. size(rng%use_a)) then
        write(*,*) "Incorrect dimension size versus size of use_a list"
        if (present(ierr)) then
          ierr = 6
        else
          call MPI_ABORT(MPI_COMM_WORLD, ierr, mpi_err)
        end if
      return
      else
        if (present(ierr)) then
          call rng%rng_a%initialize(count(rng%use_a), seed, n_streams, i_stream, ierr, &
          round_off_n_streams_in=round_off_n_streams_loc)
          call rng%rng_b%initialize(count(.not. rng%use_a), seed, n_streams, i_stream, ierr, &
          round_off_n_streams_in=round_off_n_streams_loc)
        else
          call rng%rng_a%initialize(count(rng%use_a), seed, n_streams, i_stream, &
          round_off_n_streams_in=round_off_n_streams_loc)
          call rng%rng_b%initialize(count(.not. rng%use_a), seed, n_streams, i_stream, &
          round_off_n_streams_in=round_off_n_streams_loc)
        end if
      end if
    end if
  end subroutine init_all

  subroutine next_all(rng, out)
    class(mix_rng), intent(inout) :: rng
    real*8, dimension(:), intent(out) :: out
    real*8, dimension(:), allocatable :: tmp_a, tmp_b
    allocate(tmp_a(count(rng%use_a)),tmp_b(count(.not. rng%use_a)))
    call rng%rng_a%next(tmp_a)
    call rng%rng_b%next(tmp_b)
    out = unpack(tmp_a, rng%use_a, out)
    out = unpack(tmp_b, .not. rng%use_a, out)
    if (use_fixed_rng_value) out = fixed_rng_value
  end subroutine next_all

  subroutine jump_ahead_all(rng, delta)
    class(mix_rng), intent(inout) :: rng
    integer, intent(in) :: delta
    call rng%rng_a%jump_ahead(delta)
    call rng%rng_b%jump_ahead(delta)
  end subroutine jump_ahead_all
end module mod_mix_rng
