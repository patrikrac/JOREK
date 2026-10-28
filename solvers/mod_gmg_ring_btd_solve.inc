! The ring solves of mod_gmg_ring_btd, written once for both precisions:
! included with RK (the real kind of the factors and the vector), GEMV /
! GETRS (d* / s*) and the routine names defined. a holds the factors as
! btd_factor leaves them, piv their pivots, x a vector in block order.

  !> y = alpha A x + beta y (A m x n, column-major, unit strides), the BLAS
  !! gemv's interface: blocks up to GEMV_SMALL entries as column axpys the
  !! compiler vectorises. A ring of 16-row node blocks (a cos / sin pair of
  !! two fields) does 4-5 such products per node and sweep, where the BLAS
  !! call itself cost more than the 256 multiply-adds.
  subroutine GEMV_S(tr, m, n, alpha, a, lda, x, incx, beta, y, incy)
    character, intent(in) :: tr
    integer, intent(in)   :: m, n, lda, incx, incy
    real(RK), intent(in)  :: alpha, beta, a(lda, *), x(*)
    real(RK), intent(inout) :: y(*)
    real(RK) :: t
    integer :: j
    if (m * n > GEMV_SMALL) then
      call GEMV(tr, m, n, alpha, a, lda, x, incx, beta, y, incy)
      return
    endif
    if (beta == 0) then
      y(1:m) = 0
    else if (beta /= 1) then
      y(1:m) = beta * y(1:m)
    endif
    do j = 1, n
      t = alpha * x(j)
      y(1:m) = y(1:m) + t * a(1:m, j)
    enddo
  end subroutine GEMV_S

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

  !> x_J = D~_J^-1 x_J: LU solve, or (btd_invert) one gemv with the inverse
  subroutine NODE_SOLVE(rg, a, piv, J, x)
    type(btd_ring), intent(in) :: rg
    real(RK), intent(in)    :: a(*)
    integer, intent(in)     :: piv(*), J
    real(RK), intent(inout) :: x(*)
    real(RK), parameter :: one = 1, zero = 0
    real(RK) :: t(rg%bs(J))
    integer :: bj, info
    bj = rg%bs(J)
    if (rg%inv) then
      t = x(rg%rp(J) + 1 : rg%rp(J) + bj)
      call GEMV_S('N', bj, bj, one, a(rg%od(J) + 1), bj, t, 1, zero, x(rg%rp(J) + 1), 1)
    else
      call GETRS('N', bj, 1, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), x(rg%rp(J) + 1), bj, info)
    endif
  end subroutine NODE_SOLVE

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
      if (J > 0) call GEMV_S('N', bj, rg%bs(J - 1), mone, a(rg%ol(J) + 1), bj, x(rg%rp(J - 1) + 1), 1, &
                           one, x(rg%rp(J) + 1), 1)
      call NODE_SOLVE(rg, a, piv, J, x)
      call GEMV_S('N', bc, bj, mone, a(rg%orr(J) + 1), bc, x(rg%rp(J) + 1), 1, one, x(rg%rp(c) + 1), 1)
    enddo
    call NODE_SOLVE(rg, a, piv, c, x)
    do J = c - 1, 0, -1
      bj = rg%bs(J)
      call GEMV_S('N', bj, bc, mone, a(rg%oc(J) + 1), bj, x(rg%rp(c) + 1), 1, one, x(rg%rp(J) + 1), 1)
      if (J < c - 1) call GEMV_S('N', bj, rg%bs(J + 1), mone, a(rg%ou(J) + 1), bj, x(rg%rp(J + 1) + 1), 1, &
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
      if (J > s + 1) call GEMV_S('N', bj, rg%bs(J - 1), mone, a(rg%ol(J) + 1), bj, x(rg%rp(J - 1) + 1), 1, &
                               one, x(rg%rp(J) + 1), 1)
      call NODE_SOLVE(rg, a, piv, J, x)
      call GEMV_S('N', ba, bj, mone, a(rg%orr(J) + 1), ba, x(rg%rp(J) + 1), 1, one, x(rg%rp(s) + 1), 1)
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
      call GEMV_S('N', b, rg%bs(np), mone, a(rg%ol(n) + 1), b, x(rg%rp(np) + 1), 1, one, x(rg%rp(n) + 1), 1)
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
      call GEMV_S('N', bj, rg%bs(s), mone, a(rg%oc(J) + 1), bj, x(rg%rp(s) + 1), 1, one, x(rg%rp(J) + 1), 1)
      call GEMV_S('N', bj, rg%bs(jn), mone, a(rg%ou(J) + 1), bj, x(rg%rp(jn) + 1), 1, one, x(rg%rp(J) + 1), 1)
    enddo
  end subroutine HYB_SOL_BWD
