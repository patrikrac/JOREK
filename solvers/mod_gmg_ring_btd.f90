!> Direct solvers for the GMG's ring smoother blocks (mod_petsc_pc_gmg).
!!
!! A ring block holds the rows of all nodes J = 0..nn-1 of one flux surface
!! (and, with harm_pair 2, all toroidal slots). C1 elements couple a node to
!! J +- 1 only, so the block is a PERIODIC block-tridiagonal chain of dense
!! node blocks: D_J = A(J, J), U_J = A(J, J+1), L_J = A(J, J-1), indices mod
!! nn. Two factorisations of it, both with partial pivoting inside the node
!! blocks only (none across nodes):
!!  - RS_THOMAS: block LU of nodes 0..nn-2 bordered by the last node (the
!!    periodic coupling becomes a fill column C and row R);
!!  - RS_HYB: Thomas / reduction hybrid (partitioned block LU). ns separator
!!    nodes cut the ring into ns segments; block Thomas eliminates every
!!    segment's interior, independently of the others, with its fill towards
!!    the segment's two separators; the separators' Schur complement, again
!!    a periodic block-tridiagonal chain of ns nodes, is solved dense (ns <=
!!    3) or by block Thomas. Thomas's work (ns = 1 is Thomas, with node 0 as
!!    the corner), with ns-way parallelism inside a ring and two barriers per
!!    solve.
!! Both store ~5 bs^2 per node (the LAPACK band of the interleaved ring
!! order ~8.4 bs^2) and work on dense blocks (BLAS 3 / 2).
!!
!! Block cyclic reduction (as BCYCLIC, Hirshman et al., JCP 229 (2010) 6392,
!! but periodic and threaded level by level) was tried and removed
!! (2026-10-13): no faster than Thomas with many rings per rank, slower
!! than the hybrid with few, and ~2 barriers per level.
!!
!! A ring's storage is one segment a(1:tot): per node D, U, L (bs_J x bs_J,
!! bs_J x bs_J+1, bs_J x bs_J-1, column-major), then the method's fill. The
!! pivot blocks stay LU-factored (explicit inverses, one gemv per solve,
!! gave no gain on tok, the solves being bandwidth-bound, and raised the
!! ring residual 5e-10 -> 2e-8); their pivots take one integer per row,
!! piv(rp(J)+1 : rp(J+1)). Vectors are in block order: node J's rows at
!! x(rp(J)+1 : rp(J+1)). No PETSc here.
!!
!! Mixed precision (gmg ring_single): the factorisation runs in double; its
!! result, rounded to single (btd_to_sp), serves the *_sp solves, which read
!! half the bytes. The smoothing solves are bandwidth-bound (~46 GB/s per
!! 12-thread rank on tok), and a smoother needs no more than single accuracy.
!! The solves exist once, in mod_gmg_ring_btd_solve.inc, included for both
!! kinds.
module mod_gmg_ring_btd
  implicit none
  private

  integer, parameter, public :: RS_LAPACK = 0, RS_THOMAS = 1, RS_HYB = 2

  type, public :: btd_ring
    integer :: meth = 0, nn = 0
    integer, allocatable :: bs(:), rp(:)            !< (0:nn-1) node rows, (0:nn) row offsets
    integer(8), allocatable :: od(:), ou(:), ol(:)  !< (0:nn-1) offsets of D, U, L (0-based)
    integer(8), allocatable :: oc(:), orr(:)        !< Thomas, hybrid: border column C_J, row R_J
    ! hybrid: separator k = node sep(k) (sep(0) = 0), the left end of
    ! segment k, whose interior runs to hyb_end(k); per segment the Schur
    ! blocks of its separators a = sep(k), b = sep(k+1): S_ab, S_ba, S_bb
    ! (S_aa goes straight into D_a). The separators' system (nf rows) is
    ! dense at of (ns <= 3), else the Thomas ring sub at of.
    integer :: ns = 0, nf = 0
    integer(8) :: of = 0
    integer, allocatable :: sep(:)
    integer(8), allocatable :: osab(:), osba(:), osbb(:)
    type(btd_ring), allocatable :: sub
    ! sepm 1: the separators' system by periodic block cyclic reduction
    ! (bcr_factor; ns >= 4 a power of 2) instead of dense / Thomas. Separator
    ! k (rows co(k)+1..co(k+1) of the packed separator vector) is eliminated
    ! at level lev = trailing zeros of k (k = 0 and ns/2 reach the top): its
    ! inverse pivot block at bd(k), its couplings L / U to k -+ 2^lev at bl(k)
    ! / bu(k); kept nodes at level l store A = L Dinv(k-h), B = U Dinv(k+h) at
    ! ba(k, l) / bb(k, l) (h = 2^l); bt: the top pair's dense inverse. All
    ! offsets are into the ring's storage, like od.
    integer :: sepm = 0, nlev = 0
    integer, allocatable :: co(:)
    integer(8), allocatable :: bd(:), bl(:), bu(:), ba(:,:), bb(:,:)
    integer(8) :: bt = 0
    integer(8) :: tot = 0                           !< storage (doubles)
  end type btd_ring

  public :: btd_plan, btd_index, btd_factor, btd_solve, btd_matvec, btd_to_sp, btd_solve_sp
  public :: btd_equil
  public :: hyb_end, hyb_fac_seg, hyb_fac_sep, hyb_sol_fwd, hyb_sol_sep, hyb_sol_bwd
  public :: hyb_sol_fwd_sp, hyb_sol_sep_sp, hyb_sol_bwd_sp
  public :: bcr_ok, btd_dev_pack

  external :: dgetrf, dgetrs, dgetri, dgemm, dgemv, sgetrs, sgemv

contains

  !> Storage layout of a ring of nn >= 4 nodes with bs(0:nn-1) rows each;
  !! ns: the hybrid's segments (1..nn/2); bcr: its separators by block cyclic
  !! reduction where bcr_ok(ns)
  recursive subroutine btd_plan(rg, meth, nn, bs, ns, bcr)
    type(btd_ring), intent(inout) :: rg
    integer, intent(in) :: meth, nn, bs(0:nn - 1)
    integer, intent(in), optional :: ns
    logical, intent(in), optional :: bcr
    integer :: J, c, k, i, l, h, kk
    integer(8) :: t

    rg%meth = meth; rg%nn = nn
    if (allocated(rg%bs)) deallocate(rg%bs, rg%rp, rg%od, rg%ou, rg%ol)
    if (allocated(rg%oc)) deallocate(rg%oc, rg%orr)
    if (allocated(rg%sep)) deallocate(rg%sep, rg%osab, rg%osba, rg%osbb)
    if (allocated(rg%sub)) deallocate(rg%sub)
    if (allocated(rg%co)) deallocate(rg%co, rg%bd, rg%bl, rg%bu, rg%ba, rg%bb)
    rg%sepm = 0; rg%nlev = 0
    allocate(rg%bs(0:nn - 1), rg%rp(0:nn), rg%od(0:nn - 1), rg%ou(0:nn - 1), rg%ol(0:nn - 1))
    rg%bs = bs
    rg%rp(0) = 0
    do J = 0, nn - 1
      rg%rp(J + 1) = rg%rp(J) + bs(J)
    enddo
    t = 0
    do J = 0, nn - 1
      rg%od(J) = t; t = t + int(bs(J), 8) * bs(J)
      rg%ou(J) = t; t = t + int(bs(J), 8) * bs(nxt(J))
      rg%ol(J) = t; t = t + int(bs(J), 8) * bs(prv(J))
    enddo
    if (meth == RS_THOMAS) then
      c = nn - 1
      allocate(rg%oc(0:c - 1), rg%orr(0:c - 1))
      do J = 0, c - 1
        rg%oc(J) = t;  t = t + int(bs(J), 8) * bs(c)
        rg%orr(J) = t; t = t + int(bs(c), 8) * bs(J)
      enddo
    else if (meth == RS_HYB) then
      rg%ns = 2
      if (present(ns)) rg%ns = ns
      rg%ns = max(1, min(rg%ns, nn / 2))
      allocate(rg%sep(0:rg%ns - 1), rg%osab(0:rg%ns - 1), rg%osba(0:rg%ns - 1), rg%osbb(0:rg%ns - 1))
      allocate(rg%oc(0:nn - 1), rg%orr(0:nn - 1))
      rg%oc = 0; rg%orr = 0
      do k = 0, rg%ns - 1
        rg%sep(k) = (k * nn) / rg%ns
      enddo
      do k = 0, rg%ns - 1
        c = rg%sep(k); i = rg%sep(modulo(k + 1, rg%ns))
        do J = c + 1, hyb_end(rg, k)
          rg%oc(J) = t;  t = t + int(bs(J), 8) * bs(c)
          rg%orr(J) = t; t = t + int(bs(c), 8) * bs(J)
        enddo
        rg%osab(k) = t; t = t + int(bs(c), 8) * bs(i)
        rg%osba(k) = t; t = t + int(bs(i), 8) * bs(c)
        rg%osbb(k) = t; t = t + int(bs(i), 8) * bs(i)
      enddo
      rg%nf = sum(bs(rg%sep))
      rg%of = t
      if (present(bcr)) then
        if (bcr .and. bcr_ok(rg%ns)) rg%sepm = 1
      endif
      if (rg%sepm == 1) then
        associate (n_ => rg%ns)
          allocate(rg%co(0:n_), rg%bd(0:n_ - 1), rg%bl(0:n_ - 1), rg%bu(0:n_ - 1))
          rg%nlev = nint(log(dble(n_)) / log(2.d0)) - 1
          allocate(rg%ba(0:n_ - 1, 0:rg%nlev - 1), rg%bb(0:n_ - 1, 0:rg%nlev - 1))
          rg%bd = 0; rg%bl = 0; rg%bu = 0; rg%ba = 0; rg%bb = 0
          rg%co(0) = 0
          do k = 0, n_ - 1
            rg%co(k + 1) = rg%co(k) + sb(k)
          enddo
          h = 1
          do l = 0, rg%nlev - 1
            do k = h, n_ - 1, 2 * h                      ! eliminated at level l
              rg%bd(k) = t; t = t + int(sb(k), 8)**2
              rg%bl(k) = t; t = t + int(sb(k), 8) * sb(modulo(k - h, n_))
              rg%bu(k) = t; t = t + int(sb(k), 8) * sb(modulo(k + h, n_))
            enddo
            do kk = 0, n_ - 1, 2 * h                     ! kept
              rg%ba(kk, l) = t; t = t + int(sb(kk), 8) * sb(modulo(kk - h, n_))
              rg%bb(kk, l) = t; t = t + int(sb(kk), 8) * sb(modulo(kk + h, n_))
            enddo
            h = 2 * h
          enddo
          rg%bt = t; t = t + int(sb(0) + sb(h), 8)**2
        end associate
      else if (rg%ns <= 3) then
        t = t + int(rg%nf, 8)**2
      else
        allocate(rg%sub)
        call btd_plan(rg%sub, RS_THOMAS, rg%ns, bs(rg%sep))
        t = t + rg%sub%tot
      endif
    endif
    rg%tot = t

  contains

    integer function nxt(j_)
      integer, intent(in) :: j_
      nxt = modulo(j_ + 1, nn)
    end function nxt

    integer function prv(j_)
      integer, intent(in) :: j_
      prv = modulo(j_ - 1, nn)
    end function prv

    !> rows of separator k_
    integer function sb(k_)
      integer, intent(in) :: k_
      sb = bs(rg%sep(k_))
    end function sb
  end subroutine btd_plan

  !> the separators' count suits block cyclic reduction: a power of 2, >= 4
  logical function bcr_ok(ns)
    integer, intent(in) :: ns
    bcr_ok = (ns >= 4 .and. iand(ns, ns - 1) == 0)
  end function bcr_ok

  !> last interior node of hybrid segment k
  integer function hyb_end(rg, k)
    type(btd_ring), intent(in) :: rg
    integer, intent(in) :: k
    if (k == rg%ns - 1) then
      hyb_end = rg%nn - 1
    else
      hyb_end = rg%sep(k + 1) - 1
    endif
  end function hyb_end

  !> 1-based index in the ring's storage of entry (row ir of node jr, column
  !! ic of node jc), ir / ic 1-based; 0 if the nodes are not neighbours.
  integer(8) function btd_index(rg, jr, ir, jc, ic)
    type(btd_ring), intent(in) :: rg
    integer, intent(in) :: jr, ir, jc, ic
    integer :: d
    d = modulo(jc - jr, rg%nn)
    if (d == 0) then
      btd_index = rg%od(jr)
    else if (d == 1) then
      btd_index = rg%ou(jr)
    else if (d == rg%nn - 1) then
      btd_index = rg%ol(jr)
    else
      btd_index = 0
      return
    endif
    btd_index = btd_index + int(ic - 1, 8) * rg%bs(jr) + ir
  end function btd_index

  !> y = A x from the unfactored storage a (the D, U, L blocks)
  subroutine btd_matvec(rg, a, x, y)
    type(btd_ring), intent(in) :: rg
    real*8, intent(in)  :: a(*), x(*)
    real*8, intent(out) :: y(*)
    integer :: J, b, jn, jp
    do J = 0, rg%nn - 1
      b = rg%bs(J); jn = modulo(J + 1, rg%nn); jp = modulo(J - 1, rg%nn)
      call dgemv('N', b, b, 1.d0, a(rg%od(J) + 1), b, x(rg%rp(J) + 1), 1, 0.d0, y(rg%rp(J) + 1), 1)
      call dgemv('N', b, rg%bs(jn), 1.d0, a(rg%ou(J) + 1), b, x(rg%rp(jn) + 1), 1, 1.d0, y(rg%rp(J) + 1), 1)
      call dgemv('N', b, rg%bs(jp), 1.d0, a(rg%ol(J) + 1), b, x(rg%rp(jp) + 1), 1, 1.d0, y(rg%rp(J) + 1), 1)
    enddo
  end subroutine btd_matvec

  !> Factor the ring in place, serially. info /= 0: a node block was singular.
  subroutine btd_factor(rg, a, piv, info)
    type(btd_ring), intent(in) :: rg
    real*8, intent(inout)  :: a(*)
    integer, intent(inout) :: piv(*)
    integer, intent(out)   :: info
    integer :: k
    if (rg%meth == RS_THOMAS) then
      call thomas_factor(rg, a, piv, info)
      return
    endif
    do k = 0, rg%ns - 1
      call hyb_fac_seg(rg, a, piv, k, info)
      if (info /= 0) return
    enddo
    call hyb_fac_sep(rg, a, piv, info)
  end subroutine btd_factor

  !> Bordered block LU: nodes 0..c-1 (c = nn-1) as a block-tridiagonal chain
  !! whose couplings to node c form the border column C and row R. In place:
  !! D_J -> LU(D~_J), U_J -> G_J = D~_J^-1 U_J, C_J -> H_J = D~_J^-1 C~_J,
  !! R_J -> R~_J, D_c -> LU of the corner's Schur complement.
  subroutine thomas_factor(rg, a, piv, info)
    type(btd_ring), intent(in) :: rg
    real*8, intent(inout)  :: a(*)
    integer, intent(inout) :: piv(*)
    integer, intent(out)   :: info
    integer :: J, c, bj, bc, bn, inf2
    c = rg%nn - 1; bc = rg%bs(c)
    info = 0
    do J = 0, c - 1
      a(rg%oc(J) + 1 : rg%oc(J) + int(rg%bs(J), 8) * bc) = 0.d0
      a(rg%orr(J) + 1 : rg%orr(J) + int(rg%bs(J), 8) * bc) = 0.d0
    enddo
    ! the periodic couplings: C_0 = L_0, C_c-1 = U_c-1, R_0 = U_c, R_c-1 = L_c
    call addb(rg%oc(0), rg%ol(0), int(rg%bs(0), 8) * bc)
    call addb(rg%oc(c - 1), rg%ou(c - 1), int(rg%bs(c - 1), 8) * bc)
    call addb(rg%orr(0), rg%ou(c), int(rg%bs(0), 8) * bc)
    call addb(rg%orr(c - 1), rg%ol(c), int(rg%bs(c - 1), 8) * bc)
    do J = 0, c - 1
      bj = rg%bs(J)
      call dgetrf(bj, bj, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), info)
      if (info /= 0) return
      call dgetrs('N', bj, bc, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), a(rg%oc(J) + 1), bj, inf2)
      if (J < c - 1) then
        bn = rg%bs(J + 1)
        call dgetrs('N', bj, bn, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), a(rg%ou(J) + 1), bj, inf2)
        call dgemm('N', 'N', bn, bn, bj, -1.d0, a(rg%ol(J + 1) + 1), bn, a(rg%ou(J) + 1), bj, &
                   1.d0, a(rg%od(J + 1) + 1), bn)
        call dgemm('N', 'N', bn, bc, bj, -1.d0, a(rg%ol(J + 1) + 1), bn, a(rg%oc(J) + 1), bj, &
                   1.d0, a(rg%oc(J + 1) + 1), bn)
        call dgemm('N', 'N', bc, bn, bj, -1.d0, a(rg%orr(J) + 1), bc, a(rg%ou(J) + 1), bj, &
                   1.d0, a(rg%orr(J + 1) + 1), bc)
      endif
      call dgemm('N', 'N', bc, bc, bj, -1.d0, a(rg%orr(J) + 1), bc, a(rg%oc(J) + 1), bj, &
                 1.d0, a(rg%od(c) + 1), bc)
    enddo
    call dgetrf(bc, bc, a(rg%od(c) + 1), bc, piv(rg%rp(c) + 1), info)

  contains

    subroutine addb(dst, src, n)
      integer(8), intent(in) :: dst, src, n
      a(dst + 1 : dst + n) = a(dst + 1 : dst + n) + a(src + 1 : src + n)
    end subroutine addb
  end subroutine thomas_factor

  ! ---- Thomas / reduction hybrid: the segments are independent (each
  ! writes only its own interior, its fill and its left separator's D), so
  ! a caller may run them on threads, across rings too; hyb_fac_sep then
  ! needs all of a ring's segments.

  !> segment k: block Thomas over its interior z = a+1..e (a = sep(k), b =
  !! the next separator). In place: D_z -> LU(D~_z), U_z -> D~_z^-1 U~_z (G,
  !! towards b at z = e), C_z -> H_z = D~_z^-1 C~_z (a's fill column), R_z ->
  !! R~_z (a's fill row); D_a -= sum R~ H, and S_ab, S_ba, S_bb stored
  subroutine hyb_fac_seg(rg, a, piv, k, info)
    type(btd_ring), intent(in) :: rg
    real*8, intent(inout)  :: a(*)
    integer, intent(inout) :: piv(*)
    integer, intent(in)    :: k
    integer, intent(out)   :: info
    integer :: s, b, e, J, bj, bp, ba, bb, inf2
    s = rg%sep(k); b = rg%sep(modulo(k + 1, rg%ns)); e = hyb_end(rg, k)
    ba = rg%bs(s); bb = rg%bs(b)
    info = 0
    ! the couplings to a: C_a+1 = L_a+1, R_a+1 = U_a
    a(rg%oc(s + 1) + 1 : rg%oc(s + 1) + int(rg%bs(s + 1), 8) * ba) = &
      a(rg%ol(s + 1) + 1 : rg%ol(s + 1) + int(rg%bs(s + 1), 8) * ba)
    a(rg%orr(s + 1) + 1 : rg%orr(s + 1) + int(rg%bs(s + 1), 8) * ba) = &
      a(rg%ou(s) + 1 : rg%ou(s) + int(rg%bs(s + 1), 8) * ba)
    do J = s + 1, e
      bj = rg%bs(J)
      if (J > s + 1) then
        bp = rg%bs(J - 1)
        call dgemm('N', 'N', bj, bj, bp, -1.d0, a(rg%ol(J) + 1), bj, a(rg%ou(J - 1) + 1), bp, &
                   1.d0, a(rg%od(J) + 1), bj)
        call dgemm('N', 'N', bj, ba, bp, -1.d0, a(rg%ol(J) + 1), bj, a(rg%oc(J - 1) + 1), bp, &
                   0.d0, a(rg%oc(J) + 1), bj)
        call dgemm('N', 'N', ba, bj, bp, -1.d0, a(rg%orr(J - 1) + 1), ba, a(rg%ou(J - 1) + 1), bp, &
                   0.d0, a(rg%orr(J) + 1), ba)
      endif
      call dgetrf(bj, bj, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), info)
      if (info /= 0) return
      call dgetrs('N', bj, ba, a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), a(rg%oc(J) + 1), bj, inf2)
      call dgetrs('N', bj, rg%bs(modulo(J + 1, rg%nn)), a(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), &
                  a(rg%ou(J) + 1), bj, inf2)
      call dgemm('N', 'N', ba, ba, bj, -1.d0, a(rg%orr(J) + 1), ba, a(rg%oc(J) + 1), bj, &
                 1.d0, a(rg%od(s) + 1), ba)
    enddo
    bj = rg%bs(e)
    call dgemm('N', 'N', ba, bb, bj, -1.d0, a(rg%orr(e) + 1), ba, a(rg%ou(e) + 1), bj, &
               0.d0, a(rg%osab(k) + 1), ba)
    call dgemm('N', 'N', bb, ba, bj, -1.d0, a(rg%ol(b) + 1), bb, a(rg%oc(e) + 1), bj, &
               0.d0, a(rg%osba(k) + 1), bb)
    call dgemm('N', 'N', bb, bb, bj, -1.d0, a(rg%ol(b) + 1), bb, a(rg%ou(e) + 1), bj, &
               0.d0, a(rg%osbb(k) + 1), bb)
  end subroutine hyb_fac_seg

  !> the separators' system, after all segments: separator k's D (with
  !! S_aa) + S_bb of segment k-1, couplings S_ab(k) to k+1 and S_ba(k-1) to
  !! k-1; LU-factored, dense or by Thomas, its pivots in the separators'
  !! piv segments in order
  subroutine hyb_fac_sep(rg, a, piv, info)
    type(btd_ring), intent(in) :: rg
    real*8, intent(inout)  :: a(*)
    integer, intent(inout) :: piv(*)
    integer, intent(out)   :: info
    integer :: ns, k, kp, kn, q, co(0:rg%ns), ipv(rg%nf)
    ns = rg%ns
    co(0) = 0
    do k = 0, ns - 1
      co(k + 1) = co(k) + rg%bs(rg%sep(k))
    enddo
    if (rg%sepm == 1) then
      call bcr_factor(rg, a, info)
      do q = 1, rg%nf                       ! explicit inverses: no pivots
        ipv(q) = q - co(count(co(1:ns) < q))
      enddo
    else if (ns <= 3) then
      a(rg%of + 1 : rg%of + int(rg%nf, 8)**2) = 0.d0
      do k = 0, ns - 1
        kp = modulo(k - 1, ns); kn = modulo(k + 1, ns)
        call addf(k, k, rg%od(rg%sep(k)))
        call addf(k, k, rg%osbb(kp))
        call addf(k, kn, rg%osab(k))
        call addf(kn, k, rg%osba(k))
      enddo
      call dgetrf(rg%nf, rg%nf, a(rg%of + 1), rg%nf, ipv, info)
    else
      associate (sb => rg%sub)
        do k = 0, ns - 1
          kp = modulo(k - 1, ns)
          call cpy(sb%od(k), rg%od(rg%sep(k)), int(sb%bs(k), 8)**2, .false.)
          call cpy(sb%od(k), rg%osbb(kp), int(sb%bs(k), 8)**2, .true.)
          call cpy(sb%ou(k), rg%osab(k), int(sb%bs(k), 8) * sb%bs(modulo(k + 1, ns)), .false.)
          call cpy(sb%ol(k), rg%osba(kp), int(sb%bs(k), 8) * sb%bs(kp), .false.)
        enddo
        call thomas_factor(sb, a(rg%of + 1), ipv, info)
      end associate
    endif
    do k = 0, ns - 1
      q = rg%sep(k)
      piv(rg%rp(q) + 1 : rg%rp(q + 1)) = ipv(co(k) + 1 : co(k + 1))
    enddo

  contains

    !> dense F(block kr, block kc) += the block at offset src (column-major,
    !! rows of separator kr)
    subroutine addf(kr, kc, src)
      integer, intent(in) :: kr, kc
      integer(8), intent(in) :: src
      integer :: c, br
      integer(8) :: d
      br = co(kr + 1) - co(kr)
      do c = 1, co(kc + 1) - co(kc)
        d = rg%of + int(co(kc) + c - 1, 8) * rg%nf + co(kr)
        a(d + 1 : d + br) = a(d + 1 : d + br) + a(src + int(c - 1, 8) * br + 1 : src + int(c, 8) * br)
      enddo
    end subroutine addf

    !> the sub-ring's block at offset dst (relative to of) = (or +=) the block at src
    subroutine cpy(dst, src, n, add)
      integer(8), intent(in) :: dst, src, n
      logical, intent(in) :: add
      integer(8) :: d
      d = rg%of + dst
      if (add) then
        a(d + 1 : d + n) = a(d + 1 : d + n) + a(src + 1 : src + n)
      else
        a(d + 1 : d + n) = a(src + 1 : src + n)
      endif
    end subroutine cpy
  end subroutine hyb_fac_sep

  !> The separators' periodic block-tridiagonal chain by block cyclic
  !! reduction (sepm 1), its blocks from the segments' Schur parts as in
  !! hyb_fac_sep. Level l (h = 2^l, nodes k = 0, h, 2h, ...) inverts the
  !! odd nodes' pivot blocks and folds them into the even ones:
  !!   A = L_k Dinv_k-h, B = U_k Dinv_k+h,  D_k -= A U_k-h + B L_k+h,
  !!   L_k <- -A L_k-h,  U_k <- -B U_k+h   (couplings to k -+ 2h),
  !! until the pair 0, ns/2 is left, inverted whole (its two couplings, one
  !! across each way round the ring, summed). Explicit inverses throughout:
  !! the solve is then products only, level-parallel (the device kernel).
  subroutine bcr_factor(rg, a, info)
    type(btd_ring), intent(in) :: rg
    real*8, intent(inout) :: a(*)
    integer, intent(out)  :: info
    type :: mat_t
      real*8, allocatable :: m(:,:)
    end type mat_t
    type(mat_t), allocatable :: D(:), L(:), U(:)
    real*8, allocatable :: AA(:,:), BB(:,:), T(:,:)
    integer :: ns, k, km, kp, h, lv, b0, b1
    ns = rg%ns
    allocate(D(0:ns - 1), L(0:ns - 1), U(0:ns - 1))
    do k = 0, ns - 1
      km = modulo(k - 1, ns); kp = modulo(k + 1, ns)
      D(k)%m = blk(rg%od(rg%sep(k)), sb(k), sb(k)) + blk(rg%osbb(km), sb(k), sb(k))
      U(k)%m = blk(rg%osab(k), sb(k), sb(kp))
      L(k)%m = blk(rg%osba(km), sb(k), sb(km))
    enddo
    info = 0
    h = 1
    do lv = 0, rg%nlev - 1
      do k = h, ns - 1, 2 * h
        call inv(D(k)%m, info)
        if (info /= 0) return
        call put(rg%bd(k), D(k)%m); call put(rg%bl(k), L(k)%m); call put(rg%bu(k), U(k)%m)
      enddo
      do k = 0, ns - 1, 2 * h
        km = modulo(k - h, ns); kp = modulo(k + h, ns)
        AA = mm(L(k)%m, D(km)%m); BB = mm(U(k)%m, D(kp)%m)
        call put(rg%ba(k, lv), AA); call put(rg%bb(k, lv), BB)
        D(k)%m = D(k)%m - mm(AA, U(km)%m) - mm(BB, L(kp)%m)
        L(k)%m = -mm(AA, L(km)%m)
        U(k)%m = -mm(BB, U(kp)%m)
      enddo
      h = 2 * h
    enddo
    b0 = sb(0); b1 = sb(h)
    allocate(T(b0 + b1, b0 + b1))
    T(1:b0, 1:b0) = D(0)%m;           T(1:b0, b0 + 1:) = L(0)%m + U(0)%m
    T(b0 + 1:, 1:b0) = L(h)%m + U(h)%m; T(b0 + 1:, b0 + 1:) = D(h)%m
    call inv(T, info)
    if (info /= 0) return
    call put(rg%bt, T)

  contains

    integer function sb(k_)
      integer, intent(in) :: k_
      sb = rg%bs(rg%sep(k_))
    end function sb

    function blk(o, m, n) result(r)
      integer(8), intent(in) :: o
      integer, intent(in) :: m, n
      real*8 :: r(m, n)
      r = reshape(a(o + 1 : o + int(m, 8) * n), [m, n])
    end function blk

    subroutine put(o, r)
      integer(8), intent(in) :: o
      real*8, intent(in) :: r(:,:)
      a(o + 1 : o + size(r, kind=8)) = reshape(r, [size(r, kind=8)])
    end subroutine put

    function mm(x, y) result(r)
      real*8, intent(in) :: x(:,:), y(:,:)
      real*8 :: r(size(x, 1), size(y, 2))
      call dgemm('N', 'N', size(x, 1), size(y, 2), size(x, 2), 1.d0, x, size(x, 1), y, size(y, 1), 0.d0, r, size(x, 1))
    end function mm
  end subroutine bcr_factor

  !> x = x^-1 in place (dgetrf + dgetri); info /= 0: singular
  subroutine inv(x, info)
    real*8, intent(inout) :: x(:,:)
    integer, intent(out)  :: info
    integer :: n, ipv(size(x, 1)), lw
    real*8, allocatable :: wk(:)
    real*8 :: q(1)
    n = size(x, 1)
    call dgetrf(n, n, x, n, ipv, info)
    if (info /= 0) return
    call dgetri(n, x, n, ipv, q, -1, info)
    lw = max(1, int(q(1)))
    allocate(wk(lw))
    call dgetri(n, x, n, ipv, wk, lw, info)
  end subroutine inv

  !> The device kernel's copy of a factored ring (jorek_ring_kokkos): the
  !! factors with the interior nodes' pivot blocks as explicit inverses and,
  !! for ns <= 3, the dense separators' system inverted. ok = .false.: the
  !! layout has no device form (a sub-Thomas separator system).
  subroutine btd_dev_pack(rg, a, piv, out, ok)
    type(btd_ring), intent(in) :: rg
    real*8, intent(in)  :: a(*)
    integer, intent(in) :: piv(*)
    real*8, intent(out) :: out(*)
    logical, intent(out) :: ok
    integer :: k, J, bj, info, lw, q, ipv(max(rg%nf, 1))
    real*8, allocatable :: wk(:)
    real*8 :: qw(1)
    ok = (rg%meth == RS_HYB .and. (rg%sepm == 1 .or. rg%ns <= 3))
    if (.not. ok) return
    out(1:rg%tot) = a(1:rg%tot)
    do k = 0, rg%ns - 1
      do J = rg%sep(k) + 1, hyb_end(rg, k)
        bj = rg%bs(J)
        call dgetri(bj, out(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), qw, -1, info)
        lw = max(1, int(qw(1)))
        if (.not. allocated(wk)) allocate(wk(lw))
        if (size(wk) < lw) then
          deallocate(wk); allocate(wk(lw))
        endif
        call dgetri(bj, out(rg%od(J) + 1), bj, piv(rg%rp(J) + 1), wk, size(wk), info)
      enddo
    enddo
    if (rg%sepm == 0) then
      q = 0
      do k = 0, rg%ns - 1
        J = rg%sep(k)
        ipv(q + 1 : q + rg%bs(J)) = piv(rg%rp(J) + 1 : rg%rp(J + 1))
        q = q + rg%bs(J)
      enddo
      call dgetri(rg%nf, out(rg%of + 1), rg%nf, ipv, qw, -1, info)
      lw = max(1, int(qw(1)))
      if (allocated(wk)) deallocate(wk)
      allocate(wk(lw))
      call dgetri(rg%nf, out(rg%of + 1), rg%nf, ipv, wk, lw, info)
    endif
  end subroutine btd_dev_pack

  !> a4 = a, rounded to single: the factors of one ring for the *_sp solves
  !> Symmetric equilibration S A S of an unfactored ring, S = |diag A|^-1/2
  !! (1 where the diagonal is zero), before ring_single's factorisation. The
  !! C1 DOFs of one node differ in scale by the element sizes (d^2/ds dt by
  !! their product: 1e-11..1e-16 of the value DOF on fine X-point grids), so
  !! an unscaled block's single-precision factor is cond * 6e-8 wrong; the
  !! scaled one only by its physical condition. The solves then take
  !! x = S (S A S)^-1 S b.
  subroutine btd_equil(rg, a, sc)
    type(btd_ring), intent(in) :: rg
    real*8, intent(inout) :: a(*)
    real*8, intent(out)   :: sc(*)
    integer :: J, Jn, Jp, i, k, nb, nn
    real*8 :: d
    nn = rg%nn
    do J = 0, nn - 1
      nb = rg%bs(J)
      do i = 1, nb
        d = abs(a(rg%od(J) + int(i - 1, 8) * nb + i))
        sc(rg%rp(J) + i) = 1.d0
        if (d > 0.d0) sc(rg%rp(J) + i) = 1.d0 / sqrt(d)
      enddo
    enddo
    do J = 0, nn - 1
      nb = rg%bs(J); Jn = modulo(J + 1, nn); Jp = modulo(J - 1, nn)
      do k = 1, nb
        do i = 1, nb
          a(rg%od(J) + int(k - 1, 8) * nb + i) = a(rg%od(J) + int(k - 1, 8) * nb + i) &
            * sc(rg%rp(J) + i) * sc(rg%rp(J) + k)
        enddo
      enddo
      do k = 1, rg%bs(Jn)
        do i = 1, nb
          a(rg%ou(J) + int(k - 1, 8) * nb + i) = a(rg%ou(J) + int(k - 1, 8) * nb + i) &
            * sc(rg%rp(J) + i) * sc(rg%rp(Jn) + k)
        enddo
      enddo
      do k = 1, rg%bs(Jp)
        do i = 1, nb
          a(rg%ol(J) + int(k - 1, 8) * nb + i) = a(rg%ol(J) + int(k - 1, 8) * nb + i) &
            * sc(rg%rp(J) + i) * sc(rg%rp(Jp) + k)
        enddo
      enddo
    enddo
  end subroutine btd_equil

  subroutine btd_to_sp(rg, a, a4)
    type(btd_ring), intent(in) :: rg
    real*8, intent(in)  :: a(*)
    real*4, intent(out) :: a4(*)
    a4(1:rg%tot) = real(a(1:rg%tot), 4)
  end subroutine btd_to_sp

  ! ---- the solves, double (as factored) and single (btd_to_sp)
#define RK 8
#define GEMV dgemv
#define GETRS dgetrs
#define BTD_SOLVE btd_solve
#define THOMAS_SOLVE thomas_solve
#define HYB_SOL_FWD hyb_sol_fwd
#define HYB_SOL_SEP hyb_sol_sep
#define HYB_SOL_BWD hyb_sol_bwd
#define BCR_SOLVE bcr_solve
#include "mod_gmg_ring_btd_solve.inc"
#undef RK
#undef GEMV
#undef GETRS
#undef BTD_SOLVE
#undef THOMAS_SOLVE
#undef HYB_SOL_FWD
#undef HYB_SOL_SEP
#undef HYB_SOL_BWD
#undef BCR_SOLVE
#define RK 4
#define GEMV sgemv
#define GETRS sgetrs
#define BTD_SOLVE btd_solve_sp
#define THOMAS_SOLVE thomas_solve_sp
#define HYB_SOL_FWD hyb_sol_fwd_sp
#define HYB_SOL_SEP hyb_sol_sep_sp
#define HYB_SOL_BWD hyb_sol_bwd_sp
#define BCR_SOLVE bcr_solve_sp
#include "mod_gmg_ring_btd_solve.inc"

end module mod_gmg_ring_btd
