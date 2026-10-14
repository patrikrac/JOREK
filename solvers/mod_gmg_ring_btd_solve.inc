! The ring solves of mod_gmg_ring_btd, written once for both precisions:
! included with RK (the real kind of the factors and the vector), GEMV /
! GETRS (d* / s*) and the routine names defined. a holds the factors as
! btd_factor leaves them, piv their pivots, x a vector in block order.

  !> x = A^-1 x, serially
  subroutine BTD_SOLVE(rg, a, piv, x)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*)
    real(RK), intent(inout) :: x(*)
    integer :: k
    if (rg%meth == RS_THOMAS) then
      call THOMAS_SOLVE(rg, a, piv, x)
      return
    endif
    do k = 0, rg%ns - 1
      call HYB_SOL_FWD(rg, a, piv, x, k)
    enddo
    call HYB_SOL_SEP(rg, a, piv, x)
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
  !! of the previous segment's last node, then the separators' system
  subroutine HYB_SOL_SEP(rg, a, piv, x)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*)
    real(RK), intent(inout) :: x(*)
    real(RK), parameter :: one = 1, mone = -1
    integer :: k, n, np, b, q, info, ipv(rg%nf)
    real(RK) :: w(rg%nf)
    q = 0
    do k = 0, rg%ns - 1
      n = rg%sep(k); b = rg%bs(n); np = modulo(n - 1, rg%nn)
      call GEMV('N', b, rg%bs(np), mone, a(rg%ol(n) + 1), b, x(rg%rp(np) + 1), 1, one, x(rg%rp(n) + 1), 1)
      w(q + 1 : q + b) = x(rg%rp(n) + 1 : rg%rp(n) + b)
      ipv(q + 1 : q + b) = piv(rg%rp(n) + 1 : rg%rp(n) + b)
      q = q + b
    enddo
    if (rg%sepm == 1) then
      call BCR_SOLVE(rg, a, w)
    else if (rg%ns <= 3) then
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
  end subroutine HYB_SOL_SEP

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

  !> the separators' system by block cyclic reduction (sepm 1, bcr_factor):
  !! w = the separators' right-hand sides packed by rg%co, solution out.
  !! Down the levels the kept nodes gather w_k -= A w_k-h + B w_k+h, the top
  !! pair is solved whole, and back up each eliminated node takes
  !! w_k = Dinv_k (w_k - L_k w_k-h - U_k w_k+h).
  subroutine BCR_SOLVE(rg, a, w)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    real(RK), intent(inout) :: w(*)
    real(RK), parameter :: one = 1, mone = -1, zero = 0
    integer :: ns, lv, h, k, km, kp, bk, b0, b1
    real(RK) :: t(rg%nf), u(rg%nf)
    ns = rg%ns
    h = 1
    do lv = 0, rg%nlev - 1
      do k = 0, ns - 1, 2 * h
        km = modulo(k - h, ns); kp = modulo(k + h, ns); bk = nb(k)
        call GEMV('N', bk, nb(km), mone, a(rg%ba(k, lv) + 1), bk, w(rg%co(km) + 1), 1, one, w(rg%co(k) + 1), 1)
        call GEMV('N', bk, nb(kp), mone, a(rg%bb(k, lv) + 1), bk, w(rg%co(kp) + 1), 1, one, w(rg%co(k) + 1), 1)
      enddo
      h = 2 * h
    enddo
    b0 = nb(0); b1 = nb(h)
    t(1:b0) = w(rg%co(0) + 1 : rg%co(0) + b0)
    t(b0 + 1 : b0 + b1) = w(rg%co(h) + 1 : rg%co(h) + b1)
    call GEMV('N', b0 + b1, b0 + b1, one, a(rg%bt + 1), b0 + b1, t, 1, zero, u, 1)
    w(rg%co(0) + 1 : rg%co(0) + b0) = u(1:b0)
    w(rg%co(h) + 1 : rg%co(h) + b1) = u(b0 + 1 : b0 + b1)
    do lv = rg%nlev - 1, 0, -1
      h = 2**lv
      do k = h, ns - 1, 2 * h
        km = modulo(k - h, ns); kp = modulo(k + h, ns); bk = nb(k)
        t(1:bk) = w(rg%co(k) + 1 : rg%co(k) + bk)
        call GEMV('N', bk, nb(km), mone, a(rg%bl(k) + 1), bk, w(rg%co(km) + 1), 1, one, t, 1)
        call GEMV('N', bk, nb(kp), mone, a(rg%bu(k) + 1), bk, w(rg%co(kp) + 1), 1, one, t, 1)
        call GEMV('N', bk, bk, one, a(rg%bd(k) + 1), bk, t, 1, zero, w(rg%co(k) + 1), 1)
      enddo
    enddo

  contains

    integer function nb(k_)
      integer, intent(in) :: k_
      nb = rg%bs(rg%sep(k_))
    end function nb
  end subroutine BCR_SOLVE
