! The ring solves of mod_gmg_ring_btd (included there), in kind rk_ring (single by
! default, double with -DSF_RING_DP). a holds the factors as btd_factor /
! btd_invert leave them, rounded to rk_ring, piv their pivots, x a vector in
! block order. The _sp names are kept for both precisions.
#ifdef SF_RING_DP
#define GEMV_RK dgemv
#define GETRS_RK dgetrs
#else
#define GEMV_RK sgemv
#define GETRS_RK sgetrs
#endif

  !> y = alpha A x + beta y (A m x n, column-major): node blocks up to
  !! GEMV_SMALL entries as column axpys, where a BLAS call costs more than the
  !! multiply-adds, larger ones by sgemv.
  subroutine sgemv_s(m, n, alpha, a, lda, x, beta, y)
    integer, intent(in)   :: m, n, lda
    real(rk_ring), intent(in)    :: alpha, beta, a(lda, *), x(*)
    real(rk_ring), intent(inout) :: y(*)
    real(rk_ring) :: t
    integer :: j
    if (m * n > GEMV_SMALL) then
      call GEMV_RK('N', m, n, alpha, a, lda, x, 1, beta, y, 1)
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
  end subroutine sgemv_s

  !> x = A^-1 x, serially
  subroutine btd_solve_sp(rg, a, piv, x)
    type(btd_ring), intent(in) :: rg
    real(rk_ring), intent(in)    :: a(*)
    integer, intent(in)   :: piv(*)
    real(rk_ring), intent(inout) :: x(*)
    integer :: k
    if (rg%meth == RS_THOMAS) then
      call thomas_solve_sp(rg, a, piv, x)
      return
    endif
    do k = 0, rg%ns - 1
      call hyb_sol_fwd_sp(rg, a, piv, x, k)
    enddo
    call hyb_sol_sep_sp(rg, a, piv, x)
    do k = 0, rg%ns - 1
      call hyb_sol_bwd_sp(rg, a, x, k)
    enddo
  end subroutine btd_solve_sp

  !> x_J = D~_J^-1 x_J: one gemv with the inverse (btd_invert), else the LU solve
  subroutine node_solve_sp(rg, a, piv, J, x)
    type(btd_ring), intent(in) :: rg
    real(rk_ring), intent(in)    :: a(*)
    integer, intent(in)   :: piv(*), J
    real(rk_ring), intent(inout) :: x(*)
    real(rk_ring), parameter :: one = 1, zero = 0
    real(rk_ring) :: t(rg%bs(J))
    integer :: bj, info
    bj = rg%bs(J)
    if (rg%inv) then
      t = x(rg%rp(J) + 1 : rg%rp(J) + bj)
      call sgemv_s(bj, bj, one, a(rg%od(J) + 1), bj, t, zero, x(rg%rp(J) + 1))
    else
      call GETRS_RK('N', bj, 1, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), x(rg%rp(J) + 1), bj, info)
    endif
  end subroutine node_solve_sp

  !> bordered block LU: forward over nodes 0..c-1 (the corner gathers R z),
  !! the corner, then backward
  subroutine thomas_solve_sp(rg, a, piv, x)
    type(btd_ring), intent(in) :: rg
    real(rk_ring), intent(in)    :: a(*)
    integer, intent(in)   :: piv(*)
    real(rk_ring), intent(inout) :: x(*)
    real(rk_ring), parameter :: one = 1, mone = -1
    integer :: J, c, bj, bc
    c = rg%nn - 1; bc = rg%bs(c)
    do J = 0, c - 1
      bj = rg%bs(J)
      if (J > 0) call sgemv_s(bj, rg%bs(J - 1), mone, a(rg%ol(J) + 1), bj, x(rg%rp(J - 1) + 1), &
                              one, x(rg%rp(J) + 1))
      call node_solve_sp(rg, a, piv, J, x)
      call sgemv_s(bc, bj, mone, a(rg%orr(J) + 1), bc, x(rg%rp(J) + 1), one, x(rg%rp(c) + 1))
    enddo
    call node_solve_sp(rg, a, piv, c, x)
    do J = c - 1, 0, -1
      bj = rg%bs(J)
      call sgemv_s(bj, bc, mone, a(rg%oc(J) + 1), bj, x(rg%rp(c) + 1), one, x(rg%rp(J) + 1))
      if (J < c - 1) call sgemv_s(bj, rg%bs(J + 1), mone, a(rg%ou(J) + 1), bj, x(rg%rp(J + 1) + 1), &
                                  one, x(rg%rp(J) + 1))
    enddo
  end subroutine thomas_solve_sp

  !> hybrid, segment k forward: z_J = D~_J^-1 (x_J - L_J z_J-1) over its
  !! interior, and its left separator's x_a -= R~_J z_J (each separator is
  !! written by one segment only)
  subroutine hyb_sol_fwd_sp(rg, a, piv, x, k)
    type(btd_ring), intent(in) :: rg
    real(rk_ring), intent(in)    :: a(*)
    integer, intent(in)   :: piv(*), k
    real(rk_ring), intent(inout) :: x(*)
    real(rk_ring), parameter :: one = 1, mone = -1
    integer :: s, J, bj, ba
    s = rg%sep(k); ba = rg%bs(s)
    do J = s + 1, hyb_end(rg, k)
      bj = rg%bs(J)
      if (J > s + 1) call sgemv_s(bj, rg%bs(J - 1), mone, a(rg%ol(J) + 1), bj, x(rg%rp(J - 1) + 1), &
                                  one, x(rg%rp(J) + 1))
      call node_solve_sp(rg, a, piv, J, x)
      call sgemv_s(ba, bj, mone, a(rg%orr(J) + 1), ba, x(rg%rp(J) + 1), one, x(rg%rp(s) + 1))
    enddo
  end subroutine hyb_sol_fwd_sp

  !> hybrid, after all segments' forward sweeps: every separator's x -= L z
  !! of the previous segment's last node, then the separators' system
  subroutine hyb_sol_sep_sp(rg, a, piv, x)
    type(btd_ring), intent(in) :: rg
    real(rk_ring), intent(in)    :: a(*)
    integer, intent(in)   :: piv(*)
    real(rk_ring), intent(inout) :: x(*)
    real(rk_ring), parameter :: one = 1, mone = -1
    integer :: k, n, np, b, q, info, ipv(rg%nf)
    real(rk_ring) :: w(rg%nf)
    q = 0
    do k = 0, rg%ns - 1
      n = rg%sep(k); b = rg%bs(n); np = modulo(n - 1, rg%nn)
      call sgemv_s(b, rg%bs(np), mone, a(rg%ol(n) + 1), b, x(rg%rp(np) + 1), one, x(rg%rp(n) + 1))
      w(q + 1 : q + b) = x(rg%rp(n) + 1 : rg%rp(n) + b)
      ipv(q + 1 : q + b) = piv(rg%rp(n) + 1 : rg%rp(n) + b)
      q = q + b
    enddo
    if (rg%ns <= 3) then
      call GETRS_RK('N', rg%nf, 1, a(rg%of + 1), rg%nf, ipv, w, rg%nf, info)
    else
      call thomas_solve_sp(rg%sub, a(rg%of + 1), ipv, w)
    endif
    q = 0
    do k = 0, rg%ns - 1
      n = rg%sep(k); b = rg%bs(n)
      x(rg%rp(n) + 1 : rg%rp(n) + b) = w(q + 1 : q + b)
      q = q + b
    enddo
  end subroutine hyb_sol_sep_sp

  !> hybrid, segment k backward: x_J = z_J - H_J x_a - G_J x_J+1 (x_b at the end)
  subroutine hyb_sol_bwd_sp(rg, a, x, k)
    type(btd_ring), intent(in) :: rg
    real(rk_ring), intent(in)    :: a(*)
    integer, intent(in)   :: k
    real(rk_ring), intent(inout) :: x(*)
    real(rk_ring), parameter :: one = 1, mone = -1
    integer :: s, J, bj, jn
    s = rg%sep(k)
    do J = hyb_end(rg, k), s + 1, -1
      bj = rg%bs(J); jn = modulo(J + 1, rg%nn)
      call sgemv_s(bj, rg%bs(s), mone, a(rg%oc(J) + 1), bj, x(rg%rp(s) + 1), one, x(rg%rp(J) + 1))
      call sgemv_s(bj, rg%bs(jn), mone, a(rg%ou(J) + 1), bj, x(rg%rp(jn) + 1), one, x(rg%rp(J) + 1))
    enddo
  end subroutine hyb_sol_bwd_sp
