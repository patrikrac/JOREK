module mod_pcg32
use iso_c_binding, only: c_int32_t, c_int64_t
use phys_module, only: use_fixed_rng_value, fixed_rng_value
implicit none
private
public pcg_state_setseq_64, pcg32_random_doubles_r, pcg32_random_double_r, pcg32_srandom_r, pcg32_jumpahead

type, bind(c) :: pcg_state_setseq_64
  integer(c_int64_t) :: state
  integer(c_int64_t) :: inc
end type

! PCG-random integration
interface
  function pcg32_random_double_r(rng) bind(C, name="pcg32_random_double_r")
    use iso_c_binding, only: c_double
    import :: pcg_state_setseq_64
    type(pcg_state_setseq_64), intent(inout) :: rng
    real(c_double) :: pcg32_random_double_r
  end function pcg32_random_double_r
  subroutine pcg32_srandom_r(rng, initstate, initseq) bind(C, name="pcg32_srandom_r")
    use iso_c_binding, only: c_int64_t
    import :: pcg_state_setseq_64
    type(pcg_state_setseq_64) :: rng
    integer(c_int64_t), value :: initstate
    integer(c_int64_t), value :: initseq
  end subroutine pcg32_srandom_r
  !> Advance by delta steps
  subroutine pcg32_advance_r(rng, delta) bind(C, name="pcg32_advance_r")
    use iso_c_binding, only: c_int64_t
    import :: pcg_state_setseq_64
    type(pcg_state_setseq_64) :: rng
    integer(c_int64_t), value :: delta !< delta can be passed as a signed integer
    !< and will cause a little bit of extra work then
  end subroutine pcg32_advance_r
end interface

contains
  subroutine pcg32_random_doubles_r(rng, out)
    implicit none
    type(pcg_state_setseq_64), intent(inout) :: rng
    real*8, dimension(:), intent(out) :: out
    integer :: i
    do i=1,size(out,1)
      out(i) = pcg32_random_double_r(rng)
      if (use_fixed_rng_value) out(i) = fixed_rng_value
    end do
  end subroutine pcg32_random_doubles_r

  subroutine pcg32_jumpahead(rng, n)
    use iso_c_binding, only: c_int64_t
    implicit none
    type(pcg_state_setseq_64), intent(inout) :: rng
    integer, intent(in) :: n
    integer(c_int64_t) :: delta
    ! Delta is a signed integer. Subtract to set the sign bit correctly
    delta = -n
    call pcg32_advance_r(rng, delta)
  end subroutine pcg32_jumpahead
end module mod_pcg32
