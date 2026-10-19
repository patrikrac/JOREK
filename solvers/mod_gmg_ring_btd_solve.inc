! The ring solves of mod_gmg_ring_btd, written once for both precisions:
! included with RK (the real kind of the factors and the vector), GEMV /
! GETRS (d* / s*) and the routine names defined. a holds the factors as
! btd_factor leaves them, piv their pivots, x a vector in block order.
! w / ipv (optional): the hybrid separators' workspace, >= rg%nf entries
! each, which the smoothing solves take from per-thread heap arrays; absent,
! it is an automatic array on the (thread) stack.

  !> x = A^-1 x, serially
  subroutine BTD_SOLVE(rg, a, piv, x, w, ipv)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*)
    real(RK), intent(inout) :: x(*)
    real(RK), intent(inout), optional :: w(*)     !< hybrid: separator workspace (>= rg%nf)
    integer, intent(inout), optional  :: ipv(*)   !< hybrid: separator pivot workspace (>= rg%nf)
    integer :: k
    if (rg%meth == RS_THOMAS) then
      call THOMAS_SOLVE(rg, a, piv, x)
      return
    endif
    do k = 0, rg%ns - 1
      call HYB_SOL_FWD(rg, a, piv, x, k)
    enddo
    call HYB_SOL_SEP(rg, a, piv, x, w, ipv)
    do k = 0, rg%ns - 1
      call HYB_SOL_BWD(rg, a, x, k)
    enddo
  end subroutine BTD_SOLVE

  !> bordered block LU: forward over nodes 0..c-1 (the corner gathers R z),
  !> the corner, then backward
  subroutine THOMAS_SOLVE(rg, a, piv, x)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*)
    real(RK), intent(inout) :: x(*)
    real(RK), parameter :: one = 1, mone = -1
    integer :: J, c, bj, bc, info
    c = rg%nn - 1; bc = rg%bs(c)
    do J = 0, c - 1
      bj = rg%bs(J)
      if (J > 0) call GEMV('N', bj, rg%bs(J - 1), mone, a(rg%ol(J) + 1), bj, x(rg%rp(J - 1) + 1), 1, &
                           one, x(rg%rp(J) + 1), 1)
      call GETRS('N', bj, 1, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), x(rg%rp(J) + 1), bj, info)
      call GEMV('N', bc, bj, mone, a(rg%orr(J) + 1), bc, x(rg%rp(J) + 1), 1, one, x(rg%rp(c) + 1), 1)
    enddo
    call GETRS('N', bc, 1, a(rg%od(c) + 1), bc, piv(rg%rp(c) + 1), x(rg%rp(c) + 1), bc, info)
    do J = c - 1, 0, -1
      bj = rg%bs(J)
      call GEMV('N', bj, bc, mone, a(rg%oc(J) + 1), bj, x(rg%rp(c) + 1), 1, one, x(rg%rp(J) + 1), 1)
      if (J < c - 1) call GEMV('N', bj, rg%bs(J + 1), mone, a(rg%ou(J) + 1), bj, x(rg%rp(J + 1) + 1), 1, &
                               one, x(rg%rp(J) + 1), 1)
    enddo
  end subroutine THOMAS_SOLVE

  !> hybrid, segment k forward: z_J = D~_J^-1 (x_J - L_J z_J-1) over its
  !! interior, and its left separator's x_a -= R~_J z_J (each separator is
  !! written by one segment only)
  subroutine HYB_SOL_FWD(rg, a, piv, x, k)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*), k
    real(RK), intent(inout) :: x(*)
    real(RK), parameter :: one = 1, mone = -1
    integer :: s, J, bj, ba, info
    s = rg%sep(k); ba = rg%bs(s)
    do J = s + 1, hyb_end(rg, k)
      bj = rg%bs(J)
      if (J > s + 1) call GEMV('N', bj, rg%bs(J - 1), mone, a(rg%ol(J) + 1), bj, x(rg%rp(J - 1) + 1), 1, &
                               one, x(rg%rp(J) + 1), 1)
      call GETRS('N', bj, 1, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), x(rg%rp(J) + 1), bj, info)
      call GEMV('N', ba, bj, mone, a(rg%orr(J) + 1), ba, x(rg%rp(J) + 1), 1, one, x(rg%rp(s) + 1), 1)
    enddo
  end subroutine HYB_SOL_FWD

  !> hybrid, after all segments' forward sweeps: every separator's x -= L z
  !! of the previous segment's last node, then the separators' system (w /
  !! ipv: the caller's workspace, else automatic)
  subroutine HYB_SOL_SEP(rg, a, piv, x, w, ipv)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*)
    real(RK), intent(inout) :: x(*)
    real(RK), intent(inout), optional :: w(*)     !< separator workspace (>= rg%nf)
    integer, intent(inout), optional  :: ipv(*)   !< separator pivot workspace (>= rg%nf)
    if (present(w) .and. present(ipv)) then
      call HYB_SEP_CORE(rg, a, piv, x, w, ipv)
    else
      block
        real(RK) :: ws(rg%nf)
        integer  :: iw(rg%nf)
        call HYB_SEP_CORE(rg, a, piv, x, ws, iw)
      end block
    endif
  end subroutine HYB_SOL_SEP

  !> HYB_SOL_SEP on the workspace w / ipv: the separators' right-hand sides
  !! and pivots gathered in separator order, solved, scattered back
  subroutine HYB_SEP_CORE(rg, a, piv, x, w, ipv)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*)
    real(RK), intent(inout) :: x(*)
    real(RK), intent(inout) :: w(rg%nf)
    integer, intent(inout)  :: ipv(rg%nf)
    real(RK), parameter :: one = 1, mone = -1
    integer :: k, n, np, b, q, info
    q = 0
    do k = 0, rg%ns - 1
      n = rg%sep(k); b = rg%bs(n); np = modulo(n - 1, rg%nn)
      call GEMV('N', b, rg%bs(np), mone, a(rg%ol(n) + 1), b, x(rg%rp(np) + 1), 1, one, x(rg%rp(n) + 1), 1)
      w(q + 1 : q + b) = x(rg%rp(n) + 1 : rg%rp(n) + b)
      ipv(q + 1 : q + b) = piv(rg%rp(n) + 1 : rg%rp(n) + b)
      q = q + b
    enddo
    if (rg%ns <= 3) then
      call GETRS('N', rg%nf, 1, a(rg%of + 1), rg%nf, ipv, w, rg%nf, info)
    else
      call THOMAS_SOLVE(rg%sub, a(rg%of + 1), ipv, w)
    endif
    q = 0
    do k = 0, rg%ns - 1
      n = rg%sep(k); b = rg%bs(n)
      x(rg%rp(n) + 1 : rg%rp(n) + b) = w(q + 1 : q + b)
      q = q + b
    enddo
  end subroutine HYB_SEP_CORE

  !> hybrid, segment k backward: x_J = z_J - H_J x_a - G_J x_J+1 (x_b at the end)
  subroutine HYB_SOL_BWD(rg, a, x, k)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: k
    real(RK), intent(inout) :: x(*)
    real(RK), parameter :: one = 1, mone = -1
    integer :: s, J, bj, jn
    s = rg%sep(k)
    do J = hyb_end(rg, k), s + 1, -1
      bj = rg%bs(J); jn = modulo(J + 1, rg%nn)
      call GEMV('N', bj, rg%bs(s), mone, a(rg%oc(J) + 1), bj, x(rg%rp(s) + 1), 1, one, x(rg%rp(J) + 1), 1)
      call GEMV('N', bj, rg%bs(jn), mone, a(rg%ou(J) + 1), bj, x(rg%rp(jn) + 1), 1, one, x(rg%rp(J) + 1), 1)
    enddo
  end subroutine HYB_SOL_BWD
