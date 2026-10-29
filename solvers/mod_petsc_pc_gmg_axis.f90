!> Distributed exact solve of a GMG axis block over J-sectors (workstream H2).
!!
!! THE BLOCK. The axis block of a GMG level (rings 0..K, every J, the slots of
!! one |n| group or all of them) is what the line smoothers solve exactly
!! near the axis, where the circle couplings dominate. An element couples
!! column J only to J +- 1, so the block is a periodic chain of columns plus a
!! small BORDER: the axis value DOFs force_central_node shares between all J,
!! which couple to every column. An operator product reaches further (pair_w's
!! Schur channel -Ltil Dh B_12 couples J +- 2 or 3 in the axis rings): the
!! chain then has half-width w > 1, measured from the pattern, and every
!! separator below is w columns wide instead of one.
!!
!! WHY. The sparse LU of rds_t solves the whole block on the ranks that own it
!! (rank 0: the partition is ring by ring). Its size depends on n_tht only, so
!! under strong scaling it becomes the critical path of every smoothing step.
!!
!! THE METHOD: one level of nested dissection in J (the partitioned solvers of
!! Wang, ACM TOMS 7 (1981), and SPIKE). nsec sector ranks each take a
!! contiguous range of columns; w columns between neighbouring sectors are a
!! separator S_s, the border is B. Ordering interiors I_s first,
!!   A = [ A_II  A_IG ]   (A_II block-diagonal over sectors),
!!       [ A_GI  A_GG ]   G = S_0 .. S_nsec-1, B,
!! the reduced system R = A_GG - sum_s A_G,Is A_Is^-1 A_Is,G is again a small
!! periodic chain (of separators) plus the border. A solve is
!!   w_s = A_Is^-1 x_Is                        (sector ranks, in parallel)
!!   g   = x_G - sum_s A_G,Is w_s              (one Allreduce, size |G|)
!!   y_G = R^-1 g                              (redundant on every sector rank)
!!   y_Is = A_Is^-1 (x_Is - A_Is,G y_G)        (in parallel)
!! A sector's interior is a chain of columns, factored as dense blocks
!! (chain_t, threaded BLAS); R is factored dense up to RDENSE_MAX rows.
!! Otherwise (couplings beyond the neighbouring column, a larger R, or
!! -sf_gmg_axis_chain / _axis_dense 0) PETSc's sparse LU, as rds_t. It is
!! the same elimination as an LU with the separators ordered last, so it is
!! exact; the first build checks its backward error (axd_gate in
!! mod_petsc_pc_gmg).
!!
!! Its cost per solve on a sector rank is ~ 4 nnz(L_Is) + 2 nnz(L_R): the
!! interiors shrink like n_tht/nsec while the separator chain grows like nsec,
!! so the best nsec is ~ sqrt(2 n_tht / 3) (axd_nsec), a gain of about
!! 3x at n_tht = 64 and 6x at 256 over the sequential LU.
!!
!! Communication per solve: one neighbour scatter of the axis rows to the
!! sector ranks, one Allreduce among them, one scatter back. Ranks outside
!! the sectors and without axis rows only enter the (empty) scatters.
!!
!! Structure (row classes, index sets, scatters, the reduced pattern) once per
!! run; values and numeric factorisations per rebuild, in two parts: the
!! collective refill of the sector's couplings (axd_refill) and the
!! factorisations, local to the sector ranks (axd_factor).
!!
!! The sector ranks are roff .. roff + nsec - 1 (mod np). The caller gives the
!! levels of a hierarchy different offsets and factors them after all refills,
!! so their factorisations run at the same time on different ranks instead of
!! one after the other on ranks 0 .. nsec - 1.
module mod_petsc_pc_gmg_axis
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: axd_t, axd_nsec, axd_setup, axd_refill, axd_factor, axd_solve

  !> A block-tridiagonal chain of dense blocks (open, no wrap): the rows of a
  !! matrix grouped into columns c = 1..cm that couple to c +- 1 only. Block
  !! LU with partial pivoting within the blocks, S_c = D_c - L_c G_c-1,
  !! G_c = S_c^-1 U_c. A J-sector's interior is such a chain: columns = w
  !! columns J, each block holding every ring, field, C1 DOF and slot.
  !! cpos: matrix row -> chain position; cof: column offsets (0-based);
  !! sof / lof / gof: offsets of S_c (its LU), L_c = A(c, c-1), G_c in
  !! cs / cl / cg (column-major); cw: solve workspace.
  type :: chain_t
    logical :: on = .false.
    integer :: ni = 0, cm = 0
    integer, allocatable :: cpos(:), cof(:), cpiv(:)
    integer(8), allocatable :: sof(:), lof(:), gof(:)
    real*8, allocatable :: cs(:), cl(:), cg(:), cw(:)
  end type chain_t

  type :: axd_t
    logical :: on = .false.                !< in use (the same on every rank)
    logical :: sector = .false.            !< this rank holds a sector
    integer :: nsec = 0, sec = -1
    integer :: comm = MPI_COMM_NULL        !< the sector ranks
    integer :: ni = 0, nga = 0, nred = 0, nout = 0
    integer, allocatable :: gpos(:)        !< Gamma_s row k -> reduced row (1-based)
    integer, allocatable :: gcls(:)        !< Gamma_s row k: 1 left separator S_s, 2 right S_s+1, 3 border
    logical, allocatable :: gown(:)        !< this sector contributes its x and writes its y:
                                           !< its left separator, and on sector 0 the border
    ! every sector's Gamma rows as reduced rows (concatenated) and the
    ! Allgatherv layout of their Schur blocks
    integer, allocatable :: ng_all(:), goff(:), rpos(:), bcnt(:), bdsp(:)
    real*8, allocatable  :: cblk(:), call_(:), g(:)
    IS  :: isI, isG, isAll, isOut
    Mat, pointer :: sub(:) => null()       !< A(I,I), A(I,G), A(G,I), A(G,G) of this sector
    KSP :: kI, kR
    Mat :: R
    Vec :: xl, yo, wI, tI, yI, rG, yG, gr, yr
    VecScatter :: sin, sout
    logical :: numeric_done = .false.
    logical :: failed = .false.            !< this rank's last factorisation failed (zero pivot)
    type(chain_t) :: ch                    !< the interior as a chain (ch%on), else kI
    logical :: ki_set = .false.            !< kI created
    logical :: rdn = .false.               !< R as a dense LU (rd, rpiv), else kR
    real*8, allocatable :: rd(:)
    integer, allocatable :: rpiv(:)
  end type axd_t

  !> reduced systems up to this many rows are factored dense (12000: 1.2 GB)
  integer, parameter :: RDENSE_MAX = 12000

contains

  !> Sector count for a chain of nj columns on np ranks: the cost model's
  !! optimum sqrt(2 nj / 3), at least 4 columns per sector, at most np.
  !! Below 2 the sequential LU is used.
  integer function axd_nsec(nj, np)
    integer, intent(in) :: nj, np
    axd_nsec = min(np, max(1, nint(sqrt(2.d0 * nj / 3.d0))), nj / 4)
  end function axd_nsec

  !--------------------------------------------------------------------
  !> Structure, first build only (collective on comm). loc = the rank's
  !! local rows of the block (0-based, ascending), jl = their column J; nj =
  !! n_tht of the level; nsec from axd_nsec; chain / rdense: interiors as
  !! dense block chains, R dense (up to RDENSE_MAX rows). Leaves D%on =
  !! .false. (the same on every rank) if the block is not a chain the method
  !! applies to.
  !--------------------------------------------------------------------
  subroutine axd_setup(D, A, loc, jl, nj, nsec_req, comm, tag, chain, rdense, roff)
    type(axd_t), intent(inout) :: D
    Mat, intent(in)      :: A
    integer, intent(in)  :: loc(:), jl(:), nj, nsec_req, comm
    character(len=*), intent(in) :: tag
    logical, intent(in)  :: chain, rdense
    integer, intent(in), optional :: roff   !< first sector rank (default 0)
    PetscInt :: rst, ren, ncols, nn
    PetscInt, pointer :: cols(:)
    PetscInt, parameter :: izero = 0, four = 4
    PetscErrorCode :: ierr
    integer, parameter :: WMAX = 3         !< widest chain the separators handle
    integer :: me, np, mpierr, nl, ntot, k, q, c, s, nd, j, color, t, nsec, w, thr, vme
    integer, allocatable :: cnts(:), dsps(:), gall(:), jall(:), ball(:), bmine(:), seen(:)
    integer, allocatable :: cls(:), csec(:), rid(:), jsep(:), nnz(:)
    PetscInt, allocatable :: pcol(:)
    integer :: npc, pjl
    PetscInt, allocatable :: li(:), lg(:), lo(:)
    Vec :: tmpl

    D%on = .false.
    call MPI_Comm_rank(comm, me, mpierr)
    call MPI_Comm_size(comm, np, mpierr)
    if (nsec_req < 2) return
    call MatGetOwnershipRange(A, rst, ren, ierr)

    ! all rows of the block with their column, on every rank
    nl = size(loc)
    allocate(cnts(0:np - 1), dsps(0:np))
    call MPI_Allgather(nl, 1, MPI_INTEGER, cnts, 1, MPI_INTEGER, comm, mpierr)
    dsps(0) = 0
    do k = 0, np - 1
      dsps(k + 1) = dsps(k) + cnts(k)
    enddo
    ntot = dsps(np)
    allocate(gall(ntot), jall(ntot), ball(ntot), bmine(nl))
    call MPI_Allgatherv(int(rst) + loc, nl, MPI_INTEGER, gall, cnts, dsps(0:np - 1), MPI_INTEGER, comm, mpierr)
    call MPI_Allgatherv(jl, nl, MPI_INTEGER, jall, cnts, dsps(0:np - 1), MPI_INTEGER, comm, mpierr)
    ! border rows: couple to more than 2 WMAX + 2 distinct columns (a chain
    ! row of half-width w couples to J-w .. J+w and to the border rows, which
    ! all report one J; the border couples to every column). On a short
    ! ring the threshold drops to nj - 1, which keeps the w = 1 classification
    allocate(seen(0:nj - 1))
    seen = -1
    thr = max(4, min(2 * WMAX + 2, nj - 1))
    ! a node's rows (fields, C1 DOFs, slots) share one column pattern: a row
    ! with the previous row's pattern and column J has its result (same_row)
    allocate(pcol(0))
    npc = -1; pjl = -1
    do q = 1, nl
      nd = 0
      call MatGetRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      if (same_row(q)) then
        bmine(q) = bmine(q - 1)
        call MatRestoreRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
        cycle
      endif
      k = 0
      do c = 1, int(ncols)
        k = find(gall, int(cols(c)), k)
        if (k == 0) cycle
        if (seen(jall(k)) /= q) then
          seen(jall(k)) = q; nd = nd + 1
        endif
      enddo
      call MatRestoreRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      bmine(q) = merge(1, 0, nd > thr)
    enddo
    call MPI_Allgatherv(bmine, nl, MPI_INTEGER, ball, cnts, dsps(0:np - 1), MPI_INTEGER, comm, mpierr)

    ! the chain's half-width w: the widest chain-chain coupling, measured on
    ! the owners. 1 for an element operator; an operator product reaches
    ! further (pair_w's Schur channel -Ltil Dh B_12 in the axis rings)
    w = 0
    npc = -1; pjl = -1
    do q = 1, nl
      k = dsps(me) + q
      if (ball(k) == 1) cycle
      call MatGetRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      if (same_row(q)) then
        call MatRestoreRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
        cycle
      endif
      t = 0
      do c = 1, int(ncols)
        t = find(gall, int(cols(c)), t)
        if (t == 0) cycle
        if (ball(t) == 1) cycle
        w = max(w, min(abs(jall(t) - jall(k)), nj - abs(jall(t) - jall(k))))
      enddo
      call MatRestoreRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, w, 1, MPI_INTEGER, MPI_MAX, comm, mpierr)
    w = max(w, 1)
    if (w > WMAX) then
      if (me == 0) write(*,'(A,A,A,I0,A)') "[Physics PC]   GMG ", tag, &
        ": not a chain of columns (half-width ", w, "), no sector solve"
      return
    endif
    ! a sector keeps a separator of w columns and at least w + 2 interior
    ! ones, so its interior couples only to its own two separators
    nsec = min(nsec_req, nj / (2 * w + 2))
    if (nsec < 2) return

    ! classes: 0 interior, 1 separator (w columns), 2 border; csec = the sector
    allocate(jsep(0:nsec), cls(ntot), csec(ntot))
    do s = 0, nsec
      jsep(s) = (s * nj) / nsec
    enddo
    do k = 1, ntot
      if (ball(k) == 1) then
        cls(k) = 2; csec(k) = -1
        cycle
      endif
      j = jall(k)
      do s = 0, nsec - 1
        if (j >= jsep(s) .and. j < jsep(s) + w) then
          cls(k) = 1; csec(k) = s
          exit
        else if (j >= jsep(s) + w .and. j < jsep(s + 1)) then
          cls(k) = 0; csec(k) = s
          exit
        endif
      enddo
    enddo

    ! the reduced system: S_0 .. S_nsec-1, then B, each ascending
    allocate(rid(ntot))
    rid = 0
    D%nred = 0
    do s = 0, nsec - 1
      do k = 1, ntot
        if (cls(k) == 1 .and. csec(k) == s) then
          D%nred = D%nred + 1; rid(k) = D%nred
        endif
      enddo
    enddo
    do k = 1, ntot
      if (cls(k) == 2) then
        D%nred = D%nred + 1; rid(k) = D%nred
      endif
    enddo
    ! every sector's Gamma_s = S_s, S_s+1 (periodic), B, as reduced rows
    allocate(D%ng_all(0:nsec - 1), D%goff(0:nsec))
    D%goff(0) = 0
    do s = 0, nsec - 1
      D%ng_all(s) = count((cls == 1 .and. (csec == s .or. csec == mod(s + 1, nsec))) .or. cls == 2)
      D%goff(s + 1) = D%goff(s) + D%ng_all(s)
    enddo
    allocate(D%rpos(D%goff(nsec)))
    do s = 0, nsec - 1
      q = D%goff(s)
      call add_rows(1, s); call add_rows(1, mod(s + 1, nsec)); call add_rows(2, -1)
    enddo

    ! this rank's sector
    D%nsec = nsec
    vme = me
    if (present(roff)) vme = modulo(me - roff, np)
    D%sector = (vme < nsec)
    D%sec = -1
    if (D%sector) D%sec = vme
    color = MPI_UNDEFINED
    if (D%sector) color = 1
    ! keyed by the sector: the Allgatherv of the Schur blocks is in sector order
    call MPI_Comm_split(comm, color, vme, D%comm, mpierr)
    D%ni = 0; D%nga = 0; D%nout = 0
    if (D%sector) then
      s = D%sec
      D%ni = count(cls == 0 .and. csec == s)
      D%nga = D%ng_all(s)
      allocate(li(D%ni), lg(D%nga), D%gpos(D%nga), D%gcls(D%nga), D%gown(D%nga))
      q = 0
      do k = 1, ntot
        if (cls(k) == 0 .and. csec(k) == s) then
          q = q + 1; li(q) = gall(k)
        endif
      enddo
      q = 0
      call gamma_rows(1, s, 1)
      call gamma_rows(1, mod(s + 1, nsec), 2)
      call gamma_rows(2, -1, 3)
      D%gown = (D%gcls == 1 .or. (D%gcls == 3 .and. s == 0))
      D%nout = D%ni + count(D%gown)
      allocate(lo(D%nout))
      lo(1:D%ni) = li
      lo(D%ni + 1:) = pack(lg, D%gown)
      if (chain) call chain_setup()
    else
      allocate(li(0), lg(0), lo(0), D%gpos(0), D%gcls(0), D%gown(0))
    endif
    call ISCreateGeneral(PETSC_COMM_SELF, int(D%ni, kind(nn)), li, PETSC_COPY_VALUES, D%isI, ierr)
    call ISCreateGeneral(PETSC_COMM_SELF, int(D%nga, kind(nn)), lg, PETSC_COPY_VALUES, D%isG, ierr)
    call ISCreateGeneral(PETSC_COMM_SELF, int(D%ni + D%nga, kind(nn)), [li, lg], PETSC_COPY_VALUES, &
                         D%isAll, ierr)
    call ISCreateGeneral(PETSC_COMM_SELF, int(D%nout, kind(nn)), lo, PETSC_COPY_VALUES, D%isOut, ierr)
    call MatCreateVecs(A, tmpl, PETSC_NULL_VEC, ierr)
    call VecCreateSeq(PETSC_COMM_SELF, int(D%ni + D%nga, kind(nn)), D%xl, ierr)
    call VecCreateSeq(PETSC_COMM_SELF, int(D%nout, kind(nn)), D%yo, ierr)
    call VecScatterCreate(tmpl, D%isAll, D%xl, PETSC_NULL_IS, D%sin, ierr)
    call VecScatterCreate(D%yo, PETSC_NULL_IS, tmpl, D%isOut, D%sout, ierr)
    call VecDestroy(tmpl, ierr)

    if (D%sector) then
      call VecCreateSeq(PETSC_COMM_SELF, int(D%ni, kind(nn)), D%wI, ierr)
      call VecDuplicate(D%wI, D%tI, ierr); call VecDuplicate(D%wI, D%yI, ierr)
      call VecCreateSeq(PETSC_COMM_SELF, int(D%nga, kind(nn)), D%rG, ierr)
      call VecDuplicate(D%rG, D%yG, ierr)
      allocate(D%g(D%nred), D%cblk(D%nga * D%nga), D%call_(sum(D%ng_all**2)))
      allocate(D%bcnt(0:nsec - 1), D%bdsp(0:nsec - 1))
      D%bcnt = D%ng_all**2
      D%bdsp(0) = 0
      do s = 1, nsec - 1
        D%bdsp(s) = D%bdsp(s - 1) + D%bcnt(s - 1)
      enddo
      D%rdn = rdense .and. D%nred <= RDENSE_MAX
      if (D%rdn) then
        allocate(D%rd(int(D%nred, 8) * D%nred), D%rpiv(D%nred))
      else
        call VecCreateSeq(PETSC_COMM_SELF, int(D%nred, kind(nn)), D%gr, ierr)
        call VecDuplicate(D%gr, D%yr, ierr)
        ! the reduced pattern: row r couples to the Gamma rows of every sector
        ! holding it (an upper bound; the first assembly fixes the pattern)
        allocate(nnz(D%nred))
        nnz = 0
        do s = 0, nsec - 1
          do k = D%goff(s) + 1, D%goff(s + 1)
            nnz(D%rpos(k)) = nnz(D%rpos(k)) + D%ng_all(s)
          enddo
        enddo
        nnz = min(nnz, D%nred)
        call MatCreateSeqAIJ(PETSC_COMM_SELF, int(D%nred, kind(nn)), int(D%nred, kind(nn)), &
                             izero, int(nnz, kind(nn)), D%R, ierr)
        call MatSetOption(D%R, MAT_ROW_ORIENTED, PETSC_FALSE, ierr)
      endif
    endif
    ! the four couplings of this sector (non-sector ranks take part with
    ! empty index sets: the call is collective)
    call MatCreateSubMatrices(A, four, [D%isI, D%isI, D%isG, D%isG], &
                              [D%isI, D%isG, D%isI, D%isG], MAT_INITIAL_MATRIX, D%sub, ierr)
    D%on = .true.
    if (me == 0) write(*,'(A,A,A,I0,A,I0,A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG ", tag, ": ", nsec, &
      " J-sectors (", ntot, " rows, reduced system ", D%nred, ", ", count(cls == 2), &
      " shared rows, separators ", w, " column(s))"

  contains

    !> Row q has the pattern cols(1:ncols) and the column J of the last row
    !! looked at; otherwise q becomes that row.
    logical function same_row(q_)
      integer, intent(in) :: q_
      same_row = .false.
      if (int(ncols) == npc .and. jl(q_) == pjl) then
        same_row = all(cols(1:ncols) == pcol(1:npc))
        if (same_row) return
      endif
      if (size(pcol) < ncols) then
        deallocate(pcol)
        allocate(pcol(2 * ncols))
      endif
      npc = int(ncols); pjl = jl(q_)
      pcol(1:npc) = cols(1:ncols)
    end function same_row

    !> The interior's chain layout (D%ch): every interior row's chain column,
    !! w consecutive J per column.
    subroutine chain_setup()
      integer :: kk, qq, nint_
      integer, allocatable :: ccol(:)
      if (D%ni == 0) return
      allocate(ccol(D%ni))
      qq = 0
      do kk = 1, ntot
        if (cls(kk) == 0 .and. csec(kk) == s) then
          qq = qq + 1; ccol(qq) = (jall(kk) - jsep(s) - w) / w + 1
        endif
      enddo
      nint_ = jsep(s + 1) - jsep(s) - w                ! the interior's J columns
      call chain_layout(D%ch, ccol, (nint_ + w - 1) / w)
    end subroutine chain_setup

    subroutine add_rows(c_, s_)
      integer, intent(in) :: c_, s_
      integer :: kk
      do kk = 1, ntot
        if (cls(kk) /= c_) cycle
        if (c_ == 1 .and. csec(kk) /= s_) cycle
        q = q + 1; D%rpos(q) = rid(kk)
      enddo
    end subroutine add_rows

    subroutine gamma_rows(c_, s_, gc)
      integer, intent(in) :: c_, s_, gc
      integer :: kk
      do kk = 1, ntot
        if (cls(kk) /= c_) cycle
        if (c_ == 1 .and. csec(kk) /= s_) cycle
        q = q + 1; lg(q) = gall(kk); D%gpos(q) = rid(kk); D%gcls(q) = gc
      enddo
    end subroutine gamma_rows

    !> position of x in the ascending list l, or 0. hint: the previous
    !! column's position; a row's columns come in runs of consecutive indices
    !! (the slots and C1 DOFs of a node), so the next one is tried first
    integer function find(l, x, hint)
      integer, intent(in) :: l(:), x, hint
      integer :: lo_, hi, mid
      if (hint > 0 .and. hint < size(l)) then
        if (l(hint + 1) == x) then
          find = hint + 1
          return
        endif
      endif
      find = 0
      lo_ = 1; hi = size(l)
      do while (lo_ <= hi)
        mid = (lo_ + hi) / 2
        if (l(mid) == x) then
          find = mid
          return
        else if (l(mid) < x) then
          lo_ = mid + 1
        else
          hi = mid - 1
        endif
      enddo
    end function find
  end subroutine axd_setup

  !> Numeric part, every rebuild, collective on A's communicator: the
  !! sector's couplings refilled from A (nothing on the first build,
  !! axd_setup extracted them). axd_factor follows.
  subroutine axd_refill(D, A)
    type(axd_t), intent(inout) :: D
    Mat, intent(in) :: A
    PetscErrorCode :: ierr
    PetscInt, parameter :: four = 4
    integer :: k
    if (.not. D%numeric_done) return
    call MatCreateSubMatrices(A, four, [D%isI, D%isI, D%isG, D%isG], [D%isI, D%isG, D%isI, D%isG], &
                              MAT_REUSE_MATRIX, D%sub, ierr)
    ! MAT_REUSE_MATRIX does not advance the submatrices' state: without the
    ! assembly the LU would not be refactored (the rds_t trap)
    do k = 1, 4
      call MatAssemblyBegin(D%sub(k), MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(D%sub(k), MAT_FINAL_ASSEMBLY, ierr)
    enddo
  end subroutine axd_refill

  !> Sector-rank part of the numeric phase (others return): factor A_Is, form the
  !! Schur block, gather all blocks among the sector ranks, assemble and
  !! factor R.
  subroutine axd_factor(D)
    type(axd_t), intent(inout) :: D
    PetscErrorCode :: ierr
    PC  :: pc
    Mat :: Bd, Zd, Cd
    PetscScalar, pointer :: a2(:,:), z2(:,:)
    PetscInt, allocatable :: ri(:)
    PetscInt :: nr, nc, k2
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    integer :: s, k, c, mpierr, cr, cc
    logical :: fresh

    fresh = .not. D%numeric_done
    D%numeric_done = .true.
    D%failed = .false.
    if (.not. D%sector) return

    ! the interior LU; a coupling beyond the neighbouring chain column
    ! clears ch%on and falls back to PETSc's LU
    if (D%ch%on) then
      call chain_factor(D%ch, D%sub(1), D%failed)
      if (.not. D%ch%on) call chain_free(D%ch)
    endif
    if (.not. D%ch%on) then
      if (.not. D%ki_set) then
        call make_lu(D%kI, D%sub(1), "gmg_axsec_", MATORDERINGQMD)
        D%ki_set = .true.
      else
        call KSPSetOperators(D%kI, D%sub(1), D%sub(1), ierr)
      endif
      call KSPSetUp(D%kI, ierr)
      call check_failed(D%kI)
    endif

    ! this sector's Schur block C = A_GG(owned part) - A_GI A_II^-1 A_IG
    call MatConvert(D%sub(2), MATSEQDENSE, MAT_INITIAL_MATRIX, Bd, ierr)
    call MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, Zd, ierr)
    if (D%ch%on) then
      call MatDenseGetArrayRead(Bd, a2, ierr)
      call MatDenseGetArray(Zd, z2, ierr)
      call chain_solve(D%ch, a2, z2, D%nga)
      call MatDenseRestoreArray(Zd, z2, ierr)
      call MatDenseRestoreArrayRead(Bd, a2, ierr)
      deallocate(D%ch%cw)                        ! sized for nga columns, the solves need one
    else
      call KSPMatSolve(D%kI, Bd, Zd, ierr)
    endif
    call MatMatMult(D%sub(3), Zd, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Cd, ierr)
    call MatDenseGetArrayRead(Cd, a2, ierr)
    do c = 1, D%nga
      do k = 1, D%nga
        D%cblk((c - 1) * D%nga + k) = -a2(k, c)
      enddo
    enddo
    call MatDenseRestoreArrayRead(Cd, a2, ierr)
    call MatDestroy(Bd, ierr); call MatDestroy(Zd, ierr); call MatDestroy(Cd, ierr)
    ! A_GG: rows of the left separator, the border's columns of it, and on
    ! sector 0 the border's own block -- each entry exactly once
    call MatGetSize(D%sub(4), nr, nc, ierr)
    do k = 1, D%nga
      call MatGetRow(D%sub(4), int(k - 1, kind(nr)), k2, cols, vals, ierr)
      do c = 1, int(k2)
        cr = k; cc = int(cols(c)) + 1
        if (take(cr, cc)) D%cblk((cc - 1) * D%nga + cr) = D%cblk((cc - 1) * D%nga + cr) + vals(c)
      enddo
      call MatRestoreRow(D%sub(4), int(k - 1, kind(nr)), k2, cols, vals, ierr)
    enddo

    ! every sector's block to every sector rank, then R
    call MPI_Allgatherv(D%cblk, D%nga * D%nga, MPI_DOUBLE_PRECISION, D%call_, D%bcnt, D%bdsp, &
                        MPI_DOUBLE_PRECISION, D%comm, mpierr)
    if (D%rdn) then
      ! R dense, column-major (the blocks are, too), LU with partial pivoting
      D%rd = 0.d0
      do s = 0, D%nsec - 1
        do c = 1, D%ng_all(s)
          cc = D%rpos(D%goff(s) + c)
          do k = 1, D%ng_all(s)
            cr = D%rpos(D%goff(s) + k)
            D%rd((cc - 1) * int(D%nred, 8) + cr) = D%rd((cc - 1) * int(D%nred, 8) + cr) + &
              D%call_(D%bdsp(s) + (c - 1) * D%ng_all(s) + k)
          enddo
        enddo
      enddo
      call dgetrf(D%nred, D%nred, D%rd, D%nred, D%rpiv, k)
      if (k /= 0) D%failed = .true.
      return
    endif
    if (.not. fresh) call MatZeroEntries(D%R, ierr)
    do s = 0, D%nsec - 1
      allocate(ri(D%ng_all(s)))
      ri = D%rpos(D%goff(s) + 1:D%goff(s + 1)) - 1
      call MatSetValues(D%R, int(D%ng_all(s), kind(nr)), ri, int(D%ng_all(s), kind(nr)), ri, &
                        D%call_(D%bdsp(s) + 1:D%bdsp(s) + D%bcnt(s)), ADD_VALUES, ierr)
      deallocate(ri)
    enddo
    call MatAssemblyBegin(D%R, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(D%R, MAT_FINAL_ASSEMBLY, ierr)
    if (fresh) then
      call MatSetOption(D%R, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
      call make_lu(D%kR, D%R, "gmg_axred_", MATORDERINGND)
    else
      call KSPSetOperators(D%kR, D%R, D%R, ierr)
    endif
    call KSPSetUp(D%kR, ierr)
    call check_failed(D%kR)

  contains

    !> PETSc's LU does not pivot: a zero pivot leaves a failed factor, which the
    !! caller must replace (axd_factor_pending falls back to the block's LU).
    !! The rest of the factorisation still runs, keeping the sector collectives
    !! matched.
    subroutine check_failed(ksp)
      KSP :: ksp
      PCFailedReason :: why
      call KSPGetPC(ksp, pc, ierr)
      call PCGetFailedReason(pc, why, ierr)
      if (why /= PC_NOERROR) D%failed = .true.
    end subroutine check_failed

    !> entry (row k, column c) of A_GG belongs to this sector's block: the
    !! rows of its left separator, the border's couplings to that separator,
    !! and on sector 0 the border's own block -- so each entry is added once
    logical function take(k_, c_)
      integer, intent(in) :: k_, c_
      take = D%gcls(k_) == 1 .or. (D%gcls(k_) == 3 .and. D%gcls(c_) == 1) &
             .or. (D%gcls(k_) == 3 .and. D%gcls(c_) == 3 .and. D%sec == 0)
    end function take

    !> ord: quotient minimum degree for the interiors (less fill than nested
    !! dissection on the mixed pair_w block: factorisation, Schur block and
    !! triangular solves all cheaper), nested dissection for R
    subroutine make_lu(ksp, M, pre, ord)
      KSP :: ksp
      Mat :: M
      character(len=*), intent(in) :: pre, ord
      call KSPCreate(PETSC_COMM_SELF, ksp, ierr)
      call KSPSetOperators(ksp, M, M, ierr)
      call KSPSetType(ksp, KSPPREONLY, ierr)
      call KSPGetPC(ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERPETSC, ierr)
      call PCFactorSetMatOrderingType(pc, ord, ierr)
      call KSPSetOptionsPrefix(ksp, pre, ierr)
      call KSPSetFromOptions(ksp, ierr)
    end subroutine make_lu
  end subroutine axd_factor

  !> yy(axis rows) = A_ax^-1 xx(axis rows). Collective on the vectors'
  !! communicator (the two scatters); the rest runs on the sector ranks.
  subroutine axd_solve(D, xx, yy)
    type(axd_t), intent(inout) :: D
    Vec :: xx, yy
    PetscErrorCode :: ierr
    PetscScalar, pointer :: xp(:), p1(:), p2(:), p3(:)
    integer :: k, mpierr, q

    call VecScatterBegin(D%sin, xx, D%xl, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(D%sin, xx, D%xl, INSERT_VALUES, SCATTER_FORWARD, ierr)
    if (D%sector) then
      ! w = A_II^-1 x_I
      call VecGetArrayRead(D%xl, xp, ierr)
      call VecGetArray(D%tI, p1, ierr)
      p1(1:D%ni) = xp(1:D%ni)
      call VecRestoreArray(D%tI, p1, ierr)
      call interior_solve(D%tI, D%wI)
      ! g = x_G(owned) - A_GI w, summed over the sectors
      call MatMult(D%sub(3), D%wI, D%rG, ierr)
      call VecGetArrayRead(D%rG, p2, ierr)
      D%g = 0.d0
      do k = 1, D%nga
        D%g(D%gpos(k)) = D%g(D%gpos(k)) - p2(k)
        if (D%gown(k)) D%g(D%gpos(k)) = D%g(D%gpos(k)) + xp(D%ni + k)
      enddo
      call VecRestoreArrayRead(D%rG, p2, ierr)
      call MPI_Allreduce(MPI_IN_PLACE, D%g, D%nred, MPI_DOUBLE_PRECISION, MPI_SUM, D%comm, mpierr)
      ! y_G = R^-1 g, in g
      if (D%rdn) then
        call dgetrs('N', D%nred, 1, D%rd, D%nred, D%rpiv, D%g, D%nred, k)
      else
        call VecGetArray(D%gr, p3, ierr)
        p3 = D%g
        call VecRestoreArray(D%gr, p3, ierr)
        call KSPSolve(D%kR, D%gr, D%yr, ierr)
        call VecGetArrayRead(D%yr, p3, ierr)
        D%g = p3
        call VecRestoreArrayRead(D%yr, p3, ierr)
      endif
      ! y_G on this sector's Gamma rows, then y_I = A_II^-1 (x_I - A_IG y_G)
      call VecGetArray(D%yG, p2, ierr)
      do k = 1, D%nga
        p2(k) = D%g(D%gpos(k))
      enddo
      call VecRestoreArray(D%yG, p2, ierr)
      call MatMult(D%sub(2), D%yG, D%tI, ierr)
      call VecGetArray(D%tI, p1, ierr)
      p1(1:D%ni) = xp(1:D%ni) - p1(1:D%ni)
      call VecRestoreArray(D%tI, p1, ierr)
      call VecRestoreArrayRead(D%xl, xp, ierr)
      call interior_solve(D%tI, D%yI)
      ! the rows this sector writes: its interior, its separator, (sector 0) B
      call VecGetArray(D%yo, p1, ierr)
      call VecGetArrayRead(D%yI, p2, ierr)
      p1(1:D%ni) = p2(1:D%ni)
      call VecRestoreArrayRead(D%yI, p2, ierr)
      call VecGetArrayRead(D%yG, p2, ierr)
      q = D%ni
      do k = 1, D%nga
        if (.not. D%gown(k)) cycle
        q = q + 1; p1(q) = p2(k)
      enddo
      call VecRestoreArrayRead(D%yG, p2, ierr)
      call VecRestoreArray(D%yo, p1, ierr)
    endif
    call VecScatterBegin(D%sout, D%yo, yy, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(D%sout, D%yo, yy, INSERT_VALUES, SCATTER_FORWARD, ierr)

  contains

    !> v = A_II^-1 u
    subroutine interior_solve(u, v)
      Vec :: u, v
      PetscScalar, pointer :: up(:), vp(:)
      if (.not. D%ch%on) then
        call KSPSolve(D%kI, u, v, ierr)
        return
      endif
      call VecGetArrayRead(u, up, ierr)
      call VecGetArray(v, vp, ierr)
      call chain_solve(D%ch, up, vp, 1)
      call VecRestoreArray(v, vp, ierr)
      call VecRestoreArrayRead(u, up, ierr)
    end subroutine interior_solve
  end subroutine axd_solve

  !> ch's layout from the column (1..cm) of every matrix row; ch%on = .false.
  !! if a column is empty.
  subroutine chain_layout(ch, ccol, cm)
    type(chain_t), intent(inout) :: ch
    integer, intent(in) :: ccol(:), cm
    integer :: q, c
    integer, allocatable :: nxt(:)
    call chain_free(ch)
    ch%ni = size(ccol); ch%cm = cm
    allocate(ch%cpos(ch%ni), ch%cof(cm + 1), nxt(cm))
    ch%cof = 0
    do q = 1, ch%ni
      ch%cof(ccol(q) + 1) = ch%cof(ccol(q) + 1) + 1
    enddo
    do c = 2, cm + 1
      ch%cof(c) = ch%cof(c) + ch%cof(c - 1)
    enddo
    ! cof(c) = the start of column c (0-based), cof(cm + 1) = ni
    nxt = ch%cof(1:cm)
    do q = 1, ch%ni
      c = ccol(q)
      nxt(c) = nxt(c) + 1
      ch%cpos(q) = nxt(c)
    enddo
    ch%on = all(ch%cof(2:) - ch%cof(:cm) > 0)
  end subroutine chain_layout

  subroutine chain_free(ch)
    type(chain_t), intent(inout) :: ch
    ch%on = .false.
    if (allocated(ch%cs)) deallocate(ch%cs, ch%cl, ch%cg, ch%cpiv, ch%sof, ch%lof, ch%gof)
    if (allocated(ch%cw)) deallocate(ch%cw)
  end subroutine chain_free

  !> The block LU of amat (sequential AIJ, rows and columns in the order
  !! chain_layout was given): its blocks, then S_c = D_c - L_c G_c-1, getrf,
  !! G_c = S_c^-1 U_c. A coupling beyond the neighbouring column clears ch%on
  !! (the caller falls back); a zero pivot sets failed.
  subroutine chain_factor(ch, amat, failed)
    type(chain_t), intent(inout) :: ch
    Mat, intent(in) :: amat
    logical, intent(inout) :: failed
    PetscInt :: k2
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    integer :: c, q, p, pc, cr, cc, info, m
    integer(8) :: ts, tl, tg
    integer, allocatable :: colof(:)
    integer :: n(0:ch%cm + 1)

    m = ch%cm
    n = 0
    do c = 1, m
      n(c) = ch%cof(c + 1) - ch%cof(c)
    enddo
    if (.not. allocated(ch%cs)) then
      allocate(ch%sof(m), ch%lof(m), ch%gof(m))
      ts = 0; tl = 0; tg = 0
      do c = 1, m
        ch%sof(c) = ts; ts = ts + int(n(c), 8) * n(c)
        ch%lof(c) = tl; tl = tl + int(n(c), 8) * n(c - 1)
        ch%gof(c) = tg; tg = tg + int(n(c), 8) * n(c + 1)
      enddo
      allocate(ch%cs(ts), ch%cl(tl), ch%cg(tg), ch%cpiv(ch%ni))
    endif
    allocate(colof(ch%ni))
    do c = 1, m
      colof(ch%cof(c) + 1:ch%cof(c + 1)) = c
    enddo
    ch%cs = 0.d0; ch%cl = 0.d0; ch%cg = 0.d0
    do q = 1, ch%ni
      p = ch%cpos(q); cr = colof(p)
      call MatGetRow(amat, int(q - 1, kind(k2)), k2, cols, vals, ierr)
      do c = 1, int(k2)
        pc = ch%cpos(cols(c) + 1); cc = colof(pc)
        if (cc == cr) then
          ch%cs(ch%sof(cr) + int(pc - ch%cof(cc) - 1, 8) * n(cr) + p - ch%cof(cr)) = vals(c)
        else if (cc == cr - 1) then
          ch%cl(ch%lof(cr) + int(pc - ch%cof(cc) - 1, 8) * n(cr) + p - ch%cof(cr)) = vals(c)
        else if (cc == cr + 1) then
          ch%cg(ch%gof(cr) + int(pc - ch%cof(cc) - 1, 8) * n(cr) + p - ch%cof(cr)) = vals(c)
        else
          ch%on = .false.
        endif
      enddo
      call MatRestoreRow(amat, int(q - 1, kind(k2)), k2, cols, vals, ierr)
      if (.not. ch%on) return
    enddo
    do c = 1, m
      if (c > 1) call dgemm('N', 'N', n(c), n(c), n(c - 1), -1.d0, ch%cl(ch%lof(c) + 1), n(c), &
                            ch%cg(ch%gof(c - 1) + 1), n(c - 1), 1.d0, ch%cs(ch%sof(c) + 1), n(c))
      call dgetrf(n(c), n(c), ch%cs(ch%sof(c) + 1), n(c), ch%cpiv(ch%cof(c) + 1), info)
      if (info /= 0) then
        failed = .true.
        return
      endif
      if (c < m) call dgetrs('N', n(c), n(c + 1), ch%cs(ch%sof(c) + 1), n(c), ch%cpiv(ch%cof(c) + 1), &
                             ch%cg(ch%gof(c) + 1), n(c), info)
    enddo
  end subroutine chain_factor

  !> y = A^-1 x for nrhs columns (leading dimension ni), in the matrix's row
  !! order: permuted into chain order, forward and back substitution.
  subroutine chain_solve(ch, x, y, nrhs)
    type(chain_t), intent(inout) :: ch
    PetscScalar, intent(in)    :: x(*)
    PetscScalar, intent(inout) :: y(*)
    integer, intent(in) :: nrhs
    integer :: c, q, r, info, m, ni, n0, n1
    integer(8) :: o

    m = ch%cm; ni = ch%ni
    if (allocated(ch%cw)) then
      if (size(ch%cw, kind=8) < int(ni, 8) * nrhs) deallocate(ch%cw)
    endif
    if (.not. allocated(ch%cw)) allocate(ch%cw(int(ni, 8) * nrhs))
    do r = 1, nrhs
      o = int(r - 1, 8) * ni
      do q = 1, ni
        ch%cw(o + ch%cpos(q)) = x(o + q)
      enddo
    enddo
    do c = 1, m
      n1 = ch%cof(c + 1) - ch%cof(c)
      if (c > 1) then
        n0 = ch%cof(c) - ch%cof(c - 1)
        call dgemm('N', 'N', n1, nrhs, n0, -1.d0, ch%cl(ch%lof(c) + 1), n1, ch%cw(ch%cof(c - 1) + 1), ni, &
                   1.d0, ch%cw(ch%cof(c) + 1), ni)
      endif
      call dgetrs('N', n1, nrhs, ch%cs(ch%sof(c) + 1), n1, ch%cpiv(ch%cof(c) + 1), ch%cw(ch%cof(c) + 1), ni, info)
    enddo
    do c = m - 1, 1, -1
      n1 = ch%cof(c + 1) - ch%cof(c)
      n0 = ch%cof(c + 2) - ch%cof(c + 1)
      call dgemm('N', 'N', n1, nrhs, n0, -1.d0, ch%cg(ch%gof(c) + 1), n1, ch%cw(ch%cof(c + 1) + 1), ni, &
                 1.d0, ch%cw(ch%cof(c) + 1), ni)
    enddo
    do r = 1, nrhs
      o = int(r - 1, 8) * ni
      do q = 1, ni
        y(o + q) = ch%cw(o + ch%cpos(q))
      enddo
    enddo
  end subroutine chain_solve

#endif
end module mod_petsc_pc_gmg_axis
