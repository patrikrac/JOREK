!> Exact solve of a GMG axis block distributed over J-sector ranks
!! (mod_petsc_pc_gmg). The axis block (rings 0..K, every J and slot) is a
!! periodic chain of columns J of half-width w <= 3 plus a border, the axis value
!! DOFs shared by all J. One level of nested dissection in J (partitioned solvers:
!! Wang, ACM TOMS 7 (1981) 170; SPIKE, Polizzi & Sameh, Parallel Comput. 32
!! (2006) 177): each of nsec sector ranks factors its interior, a chain of dense
!! column blocks; the reduced system R of the separators and the border (dense up
!! to RDENSE_MAX rows, else a sparse LU) is factored redundantly on every sector
!! rank. A solve is two sweeps over the interiors around one Allreduce of size
!! |R| among the sector ranks, between a scatter of the axis rows in and out.
!! Structure once per run (axd_setup); values per rebuild (axd_refill,
!! collective; axd_factor, sector ranks only).
module mod_petsc_pc_gmg_axis
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
#if PETSC_VERSION_LT(3,23,0)
  use mod_petsc_raw_csr, only: mat_get_row, mat_restore_row, mat_create_submatrices, mat_destroy_submatrices
#endif
#include "petsc_f90_compat.h"
  implicit none
  private

  public :: axd_t, axd_nsec, axd_setup, axd_refill, axd_factor, axd_solve, axd_destroy

  !> An open block-tridiagonal chain of dense blocks: matrix rows grouped into
  !! columns c = 1..cm coupling to c +- 1 only, block LU with partial pivoting
  !! inside the blocks (S_c = D_c - L_c G_c-1, G_c = S_c^-1 U_c).
  type :: chain_t
    logical :: on = .false.
    integer :: ni = 0, cm = 0
    integer, allocatable :: cpos(:), cof(:), cpiv(:)   !< row -> chain position; column offsets (0-based); pivots
    integer(8), allocatable :: sof(:), lof(:), gof(:)  !< offsets of S_c (LU), L_c, G_c in cs / cl / cg
    real*8, allocatable :: cs(:), cl(:), cg(:), cw(:)  !< column-major blocks; cw: solve workspace
  end type chain_t

  type :: axd_t
    logical :: on = .false.                !< in use (the same on every rank)
    logical :: sector = .false.            !< this rank holds a sector
    integer :: nsec = 0, sec = -1
    integer :: comm = MPI_COMM_NULL        !< the sector ranks
    integer :: ni = 0, nga = 0, nred = 0, nout = 0
    integer, allocatable :: gpos(:)        !< Gamma_s row k -> reduced row (1-based)
    integer, allocatable :: gcls(:)        !< Gamma_s row k: 1 left separator S_s, 2 right S_s+1, 3 border
    logical, allocatable :: gown(:)        !< Gamma_s row k is written by this sector
    integer, allocatable :: ng_all(:), goff(:), rpos(:), bcnt(:), bdsp(:)   !< all sectors' Gamma rows, Schur gather layout
    real*8, allocatable  :: cblk(:), call_(:), g(:)
    IS  :: isI, isG, isAll, isOut
    Mat, pointer :: sub(:) => null()       !< A(I,I), A(I,G), A(G,I), A(G,G) of this sector
    KSP :: kR
    Mat :: R
    Vec :: xl, yo, wI, tI, yI, rG, yG, gr, yr
    VecScatter :: sin, sout
    logical :: numeric_done = .false.
    logical :: failed = .false.            !< this rank's last factorisation failed
    type(chain_t) :: ch                    !< the sector's interior
    logical :: rdn = .false.               !< R as a dense LU (rd, rpiv), else kR
    logical :: kr_set = .false.
    real*8, allocatable :: rd(:)
    integer, allocatable :: rpiv(:)
    character(len=32) :: pre = ""          !< options prefix of kR
  end type axd_t

  !> reduced systems up to this many rows are factored dense
  integer, parameter :: RDENSE_MAX = 12000

contains

  !> Sector count for a chain of nj columns on np ranks: the optimum of the
  !! solve cost ~ 4 nnz(interior) + 2 nnz(R), sqrt(2 nj / 3), at least 4
  !! columns per sector, at most np. Below 2 the caller keeps its LU.
  integer function axd_nsec(nj, np)
    integer, intent(in) :: nj, np
    axd_nsec = min(np, max(1, nint(sqrt(2.d0 * nj / 3.d0))), nj / 4)
  end function axd_nsec

  !> Structure, first build only (collective on comm). loc = the rank's local
  !! rows of the block (0-based, ascending), jl = their column J; nj = the
  !! columns of the chain; pre = kR's options prefix; the sector ranks are roff ..
  !! roff + nsec - 1 (mod np). D%on = .false. (on every rank) if the method does
  !! not apply.
  subroutine axd_setup(D, A, loc, jl, nj, nsec_req, comm, tag, pre, roff)
    type(axd_t), intent(inout) :: D
    Mat, intent(in)      :: A
    integer, intent(in)  :: loc(:), jl(:), nj, nsec_req, comm, roff
    character(len=*), intent(in) :: tag, pre
    PetscInt :: rst, ren, ncols, nn
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, parameter :: izero = 0, four = 4
    PetscErrorCode :: ierr
    integer, parameter :: WMAX = 3         !< widest chain the separators handle
    integer :: me, np, mpierr, nl, ntot, k, q, c, s, nd, j, color, t, nsec, w, thr, vme
    integer, allocatable :: cnts(:), dsps(:), gall(:), jall(:), ball(:), bmine(:), seen(:)
    integer, allocatable :: cls(:), csec(:), rid(:), jsep(:), nnz(:)
    PetscInt, allocatable :: pcol(:)
    integer :: npc, pjl
    logical :: ok
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
    ! border rows couple to more than 2 WMAX + 2 distinct columns (a chain row
    ! of half-width w reaches J-w .. J+w and the border, which reports one J);
    ! on a short ring the threshold drops to nj - 1
    allocate(seen(0:nj - 1))
    seen = -1
    thr = max(4, min(2 * WMAX + 2, nj - 1))
    allocate(pcol(0))
    npc = -1; pjl = -1
    do q = 1, nl
      nd = 0
      call MatGetRow(A, rst + loc(q), ncols, cols, vals, ierr)
      if (same_row(q)) then
        bmine(q) = bmine(q - 1)
        call MatRestoreRow(A, rst + loc(q), ncols, cols, vals, ierr)
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
      call MatRestoreRow(A, rst + loc(q), ncols, cols, vals, ierr)
      bmine(q) = merge(1, 0, nd > thr)
    enddo
    call MPI_Allgatherv(bmine, nl, MPI_INTEGER, ball, cnts, dsps(0:np - 1), MPI_INTEGER, comm, mpierr)

    ! the chain's half-width w: the widest chain-chain coupling (1 for an
    ! element operator, more for an operator product)
    w = 0
    npc = -1; pjl = -1
    do q = 1, nl
      k = dsps(me) + q
      if (ball(k) == 1) cycle
      call MatGetRow(A, rst + loc(q), ncols, cols, vals, ierr)
      if (same_row(q)) then
        call MatRestoreRow(A, rst + loc(q), ncols, cols, vals, ierr)
        cycle
      endif
      t = 0
      do c = 1, int(ncols)
        t = find(gall, int(cols(c)), t)
        if (t == 0) cycle
        if (ball(t) == 1) cycle
        w = max(w, min(abs(jall(t) - jall(k)), nj - abs(jall(t) - jall(k))))
      enddo
      call MatRestoreRow(A, rst + loc(q), ncols, cols, vals, ierr)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, w, 1, MPI_INTEGER, MPI_MAX, comm, mpierr)
    w = max(w, 1)
    if (w > WMAX) then
      if (me == 0) write(*,'(A,A,A,I0,A)') "[Physics PC]   ", tag, &
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

    ! this rank's sector, and its interior as a chain of w-column blocks
    vme = modulo(me - roff, np)
    D%sector = (vme < nsec)
    D%sec = -1
    D%ni = 0
    ok = .true.
    if (D%sector) then
      D%sec = vme
      s = D%sec
      D%ni = count(cls == 0 .and. csec == s)
      call chain_setup()
      ok = D%ch%on
    endif
    call MPI_Allreduce(MPI_IN_PLACE, ok, 1, MPI_LOGICAL, MPI_LAND, comm, mpierr)
    if (.not. ok) then
      call chain_free(D%ch)
      if (me == 0) write(*,'(A,A,A)') "[Physics PC]   ", tag, ": a sector interior is not a chain, no sector solve"
      return
    endif

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

    D%nsec = nsec
    D%pre = pre
    color = MPI_UNDEFINED
    if (D%sector) color = 1
    ! keyed by the sector: the Allgatherv of the Schur blocks is in sector order
    call MPI_Comm_split(comm, color, vme, D%comm, mpierr)
    D%nga = 0; D%nout = 0
    if (D%sector) then
      s = D%sec
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
      ! a sector writes its left separator, and sector 0 the border
      D%gown = (D%gcls == 1 .or. (D%gcls == 3 .and. s == 0))
      D%nout = D%ni + count(D%gown)
      allocate(lo(D%nout))
      lo(1:D%ni) = li
      lo(D%ni + 1:) = pack(lg, D%gown)
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
      D%rdn = D%nred <= RDENSE_MAX
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
    ! the four couplings of this sector (collective: the other ranks pass empty sets)
    call MatCreateSubMatrices(A, four, [D%isI, D%isI, D%isG, D%isG], &
                              [D%isI, D%isG, D%isI, D%isG], MAT_INITIAL_MATRIX, D%sub, ierr)
    D%on = .true.
    if (me == 0) write(*,'(A,A,A,I0,A,I0,A,I0,A,I0,A,I0,A)') "[Physics PC]   ", tag, ": ", nsec, &
      " J-sectors (", ntot, " rows, reduced system ", D%nred, ", ", count(cls == 2), &
      " shared rows, separators ", w, " column(s))"

  contains

    !> Row q has the pattern cols(1:ncols) and the column J of the last row
    !! looked at (a node's rows share one pattern); otherwise q becomes that row.
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
    !! w consecutive J per column; D%ch%on = .false. if it is empty or has a gap.
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

    !> position of x in the ascending list l, or 0; the position after hint
    !! is tried first (a row's columns come in runs of consecutive indices)
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

  !> Numeric part, every rebuild after the first, collective on A's
  !! communicator: the sector's couplings refilled from A. axd_factor follows.
  subroutine axd_refill(D, A)
    type(axd_t), intent(inout) :: D
    Mat, intent(in) :: A
    PetscErrorCode :: ierr
    PetscInt, parameter :: four = 4
    integer :: k
    if (.not. D%numeric_done) return
    call MatCreateSubMatrices(A, four, [D%isI, D%isI, D%isG, D%isG], [D%isI, D%isG, D%isI, D%isG], &
                              MAT_REUSE_MATRIX, D%sub, ierr)
    ! MAT_REUSE_MATRIX does not advance the submatrices' state: assemble
    do k = 1, 4
      call MatAssemblyBegin(D%sub(k), MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(D%sub(k), MAT_FINAL_ASSEMBLY, ierr)
    enddo
  end subroutine axd_refill

  !> Sector ranks (others return): factor the interior, form the Schur block,
  !! gather all blocks among the sector ranks, factor R. A failed factorisation
  !! sets D%failed and skips the local work but not the gather.
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

    call chain_factor(D%ch, D%sub(1), D%failed)

    ! this sector's Schur block C = A_GG(owned part) - A_GI A_II^-1 A_IG
    D%cblk = 0.d0
    if (.not. D%failed) then
      call MatConvert(D%sub(2), MATSEQDENSE, MAT_INITIAL_MATRIX, Bd, ierr)
      call MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, Zd, ierr)
      call MatDenseGetArrayRead(Bd, a2, ierr)
      call MatDenseGetArray(Zd, z2, ierr)
      call chain_solve(D%ch, a2, z2, D%nga)
      call MatDenseRestoreArray(Zd, z2, ierr)
      call MatDenseRestoreArrayRead(Bd, a2, ierr)
      deallocate(D%ch%cw)                        ! sized for nga columns, the solves need one
      call MatMatMult(D%sub(3), Zd, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, Cd, ierr)
      call MatDenseGetArrayRead(Cd, a2, ierr)
      do c = 1, D%nga
        do k = 1, D%nga
          D%cblk((c - 1) * D%nga + k) = -a2(k, c)
        enddo
      enddo
      call MatDenseRestoreArrayRead(Cd, a2, ierr)
      call MatDestroy(Bd, ierr); call MatDestroy(Zd, ierr); call MatDestroy(Cd, ierr)
      ! A_GG: each entry exactly once over the sectors (take)
      call MatGetSize(D%sub(4), nr, nc, ierr)
      do k = 1, D%nga
        call MatGetRow(D%sub(4), int(k - 1, kind(nr)), k2, cols, vals, ierr)
        do c = 1, int(k2)
          cr = k; cc = int(cols(c)) + 1
          if (take(cr, cc)) D%cblk((cc - 1) * D%nga + cr) = D%cblk((cc - 1) * D%nga + cr) + vals(c)
        enddo
        call MatRestoreRow(D%sub(4), int(k - 1, kind(nr)), k2, cols, vals, ierr)
      enddo
    endif

    ! every sector's block to every sector rank, then R
    call MPI_Allgatherv(D%cblk, D%nga * D%nga, MPI_DOUBLE_PRECISION, D%call_, D%bcnt, D%bdsp, &
                        MPI_DOUBLE_PRECISION, D%comm, mpierr)
    if (D%failed) return
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
    if (.not. D%kr_set) then
      call MatSetOption(D%R, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
      call KSPCreate(PETSC_COMM_SELF, D%kR, ierr)
      call KSPSetOperators(D%kR, D%R, D%R, ierr)
      call KSPSetType(D%kR, KSPPREONLY, ierr)
      call KSPGetPC(D%kR, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERPETSC, ierr)
      call PCFactorSetMatOrderingType(pc, MATORDERINGND, ierr)
      call KSPSetOptionsPrefix(D%kR, trim(D%pre), ierr)
      call KSPSetFromOptions(D%kR, ierr)
      D%kr_set = .true.
    else
      call KSPSetOperators(D%kR, D%R, D%R, ierr)
    endif
    call KSPSetUp(D%kR, ierr)
    block
      PCFailedReason :: why
      call KSPGetPC(D%kR, pc, ierr)
      call PCGetFailedReason(pc, why, ierr)
      if (why /= PC_NOERROR) D%failed = .true.   ! PETSc's LU does not pivot
    end block

  contains

    !> entry (row k, column c) of A_GG belongs to this sector's block: the
    !! rows of its left separator, the border's couplings to that separator,
    !! and on sector 0 the border's own block
    logical function take(k_, c_)
      integer, intent(in) :: k_, c_
      take = D%gcls(k_) == 1 .or. (D%gcls(k_) == 3 .and. D%gcls(c_) == 1) &
             .or. (D%gcls(k_) == 3 .and. D%gcls(c_) == 3 .and. D%sec == 0)
    end function take
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
      call VecGetArrayRead(u, up, ierr)
      call VecGetArray(v, vp, ierr)
      call chain_solve(D%ch, up, vp, 1)
      call VecRestoreArray(v, vp, ierr)
      call VecRestoreArrayRead(u, up, ierr)
    end subroutine interior_solve
  end subroutine axd_solve

  !> Free D's objects and its communicator. Collective on the communicator D
  !! was set up on; returns at once where D is not in use (D%on, the same on
  !! every rank).
  subroutine axd_destroy(D)
    type(axd_t), intent(inout) :: D
    PetscInt, parameter :: four = 4
    PetscErrorCode :: ierr
    integer :: mpierr
    if (.not. D%on) return
    call MatDestroySubMatrices(four, D%sub, ierr)
    call VecScatterDestroy(D%sin, ierr); call VecScatterDestroy(D%sout, ierr)
    call VecDestroy(D%xl, ierr); call VecDestroy(D%yo, ierr)
    call ISDestroy(D%isI, ierr); call ISDestroy(D%isG, ierr)
    call ISDestroy(D%isAll, ierr); call ISDestroy(D%isOut, ierr)
    if (D%sector) then
      call VecDestroy(D%wI, ierr); call VecDestroy(D%tI, ierr); call VecDestroy(D%yI, ierr)
      call VecDestroy(D%rG, ierr); call VecDestroy(D%yG, ierr)
      if (D%rdn) then
        deallocate(D%rd, D%rpiv)
      else
        if (D%kr_set) call KSPDestroy(D%kR, ierr)
        call MatDestroy(D%R, ierr)
        call VecDestroy(D%gr, ierr); call VecDestroy(D%yr, ierr)
      endif
      call MPI_Comm_free(D%comm, mpierr)
      deallocate(D%g, D%cblk, D%call_, D%bcnt, D%bdsp)
    endif
    deallocate(D%gpos, D%gcls, D%gown, D%ng_all, D%goff, D%rpos)
    call chain_free(D%ch)
    if (allocated(D%ch%cpos)) deallocate(D%ch%cpos, D%ch%cof)
    D%on = .false.; D%sector = .false.; D%kr_set = .false.; D%numeric_done = .false.
  end subroutine axd_destroy

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
  !! G_c = S_c^-1 U_c. A coupling beyond the neighbouring column or a zero
  !! pivot sets failed; the factors are then incomplete and must not be used.
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
    integer, allocatable :: colof(:), n(:)

    m = ch%cm
    allocate(n(0:m + 1))
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
          failed = .true.
        endif
      enddo
      call MatRestoreRow(amat, int(q - 1, kind(k2)), k2, cols, vals, ierr)
      if (failed) return
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
