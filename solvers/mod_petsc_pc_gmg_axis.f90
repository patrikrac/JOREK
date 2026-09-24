!> Distributed exact solve of a GMG axis block over J-sectors (workstream H2).
!!
!! THE BLOCK. The axis block of a GMG level (rings 0..K, every J, the slots of
!! one |n| group or all of them) is what the line smoothers solve exactly
!! near the axis, where the circle couplings dominate. An element couples
!! column J only to J +- 1, so the block is a periodic chain of columns plus a
!! small BORDER: the axis value DOFs force_central_node shares between all J,
!! which couple to every column.
!!
!! WHY. The sparse LU of rds_t solves the whole block on the ranks that own it
!! (rank 0: the partition is ring by ring). Its size depends on n_tht only, so
!! under strong scaling it becomes the critical path of every smoothing step.
!!
!! THE METHOD: one level of nested dissection in J (the partitioned solvers of
!! Wang, ACM TOMS 7 (1981), and SPIKE). nsec sector ranks each take a
!! contiguous range of columns; one column between neighbouring sectors is a
!! separator S_s, the border is B. Ordering interiors I_s first,
!!   A = [ A_II  A_IG ]   (A_II block-diagonal over sectors),
!!       [ A_GI  A_GG ]   G = S_0 .. S_nsec-1, B,
!! the reduced system R = A_GG - sum_s A_G,Is A_Is^-1 A_Is,G is again a small
!! periodic chain (of separators) plus the border. A solve is
!!   w_s = A_Is^-1 x_Is                        (sector ranks, in parallel)
!!   g   = x_G - sum_s A_G,Is w_s              (one Allreduce, size |G|)
!!   y_G = R^-1 g                              (redundant on every sector rank)
!!   y_Is = A_Is^-1 (x_Is - A_Is,G y_G)        (in parallel)
!! with sparse LUs throughout (PETSc's, nested dissection, as rds_t). It is
!! the same elimination as a sparse LU with the separators ordered last, so
!! it is exact; the first build compares it with the rds_t LU.
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
!! run; values and numeric factorisations per rebuild (axd_numeric).
module mod_petsc_pc_gmg_axis
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: axd_t, axd_nsec, axd_setup, axd_numeric, axd_solve

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
  end type axd_t

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
  !! n_tht of the level; nsec from axd_nsec. Leaves D%on = .false. (the same
  !! on every rank) if the block is not a chain the method applies to.
  !--------------------------------------------------------------------
  subroutine axd_setup(D, A, loc, jl, nj, nsec, comm, tag)
    type(axd_t), intent(inout) :: D
    Mat, intent(in)      :: A
    integer, intent(in)  :: loc(:), jl(:), nj, nsec, comm
    character(len=*), intent(in) :: tag
    PetscInt :: rst, ren, ncols, nn
    PetscInt, pointer :: cols(:)
    PetscInt, parameter :: izero = 0, four = 4
    PetscErrorCode :: ierr
    integer :: me, np, mpierr, nl, ntot, k, q, c, s, nd, j, color, t
    integer, allocatable :: cnts(:), dsps(:), gall(:), jall(:), ball(:), bmine(:), seen(:)
    integer, allocatable :: cls(:), csec(:), rid(:), jsep(:), nnz(:)
    PetscInt, allocatable :: li(:), lg(:), lo(:)
    logical :: ok
    Vec :: tmpl

    D%on = .false.
    call MPI_Comm_rank(comm, me, mpierr)
    call MPI_Comm_size(comm, np, mpierr)
    if (nsec < 2) return
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
    ! border rows: couple to more than four distinct columns (a chain row
    ! couples to J-1, J, J+1 and to the border rows, which all report one J)
    allocate(seen(0:nj - 1))
    seen = -1
    do q = 1, nl
      nd = 0
      call MatGetRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      do c = 1, int(ncols)
        k = find(gall, int(cols(c)))
        if (k == 0) cycle
        if (seen(jall(k)) /= q) then
          seen(jall(k)) = q; nd = nd + 1
        endif
      enddo
      call MatRestoreRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      bmine(q) = merge(1, 0, nd > 4)
    enddo
    call MPI_Allgatherv(bmine, nl, MPI_INTEGER, ball, cnts, dsps(0:np - 1), MPI_INTEGER, comm, mpierr)

    ! classes: 0 interior, 1 separator, 2 border; csec = the sector
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
        if (j == jsep(s)) then
          cls(k) = 1; csec(k) = s
          exit
        else if (j > jsep(s) .and. j < jsep(s + 1)) then
          cls(k) = 0; csec(k) = s
          exit
        endif
      enddo
    enddo
    ! a chain-chain coupling across more than one column would break the
    ! dissection: check on the owners
    ok = .true.
    do q = 1, nl
      k = dsps(me) + q
      if (cls(k) == 2) cycle
      call MatGetRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      do c = 1, int(ncols)
        t = find(gall, int(cols(c)))
        if (t == 0) cycle
        if (cls(t) == 2) cycle
        if (min(abs(jall(t) - jall(k)), nj - abs(jall(t) - jall(k))) > 1) ok = .false.
      enddo
      call MatRestoreRow(A, rst + loc(q), ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, ok, 1, MPI_LOGICAL, MPI_LAND, comm, mpierr)
    if (.not. ok) then
      if (me == 0) write(*,'(A,A,A)') "[Physics PC]   GMG ", tag, &
        ": not a chain of columns, no sector solve"
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

    ! this rank's sector
    D%nsec = nsec
    D%sector = (me < nsec)
    D%sec = -1
    if (D%sector) D%sec = me
    color = MPI_UNDEFINED
    if (D%sector) color = 1
    call MPI_Comm_split(comm, color, me, D%comm, mpierr)
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
      call VecCreateSeq(PETSC_COMM_SELF, int(D%nred, kind(nn)), D%gr, ierr)
      call VecDuplicate(D%gr, D%yr, ierr)
      allocate(D%g(D%nred), D%cblk(D%nga * D%nga), D%call_(sum(D%ng_all**2)))
      allocate(D%bcnt(0:nsec - 1), D%bdsp(0:nsec - 1))
      D%bcnt = D%ng_all**2
      D%bdsp(0) = 0
      do s = 1, nsec - 1
        D%bdsp(s) = D%bdsp(s - 1) + D%bcnt(s - 1)
      enddo
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
    ! the four couplings of this sector (non-sector ranks take part with
    ! empty index sets: the call is collective)
    call MatCreateSubMatrices(A, four, [D%isI, D%isI, D%isG, D%isG], &
                              [D%isI, D%isG, D%isI, D%isG], MAT_INITIAL_MATRIX, D%sub, ierr)
    D%on = .true.
    if (me == 0) write(*,'(A,A,A,I0,A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG ", tag, ": ", nsec, &
      " J-sectors (", ntot, " rows, reduced system ", D%nred, ", ", count(cls == 2), " shared rows)"

  contains

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

    !> position of x in the ascending list l, or 0
    integer function find(l, x)
      integer, intent(in) :: l(:), x
      integer :: lo_, hi, mid
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

  !--------------------------------------------------------------------
  !> Numeric part, every rebuild (collective on A's communicator): refill
  !! the sector's couplings, factor A_Is, form its Schur block
  !! A_G,Is A_Is^-1 A_Is,G, gather all blocks, assemble and factor R.
  !--------------------------------------------------------------------
  subroutine axd_numeric(D, A)
    type(axd_t), intent(inout) :: D
    Mat, intent(in) :: A
    PetscErrorCode :: ierr
    PetscInt, parameter :: four = 4
    PC  :: pc
    Mat :: Bd, Zd, Cd
    PetscScalar, pointer :: a2(:,:)
    PetscInt, allocatable :: ri(:)
    PetscInt :: nr, nc, k2
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    integer :: s, k, c, mpierr, cr, cc
    logical :: fresh

    fresh = .not. D%numeric_done
    if (.not. fresh) then
      call MatCreateSubMatrices(A, four, [D%isI, D%isI, D%isG, D%isG], [D%isI, D%isG, D%isI, D%isG], &
                                MAT_REUSE_MATRIX, D%sub, ierr)
      ! MAT_REUSE_MATRIX does not advance the submatrices' state: without the
      ! assembly the LU would not be refactored (the rds_t trap)
      do k = 1, 4
        call MatAssemblyBegin(D%sub(k), MAT_FINAL_ASSEMBLY, ierr)
        call MatAssemblyEnd(D%sub(k), MAT_FINAL_ASSEMBLY, ierr)
      enddo
    endif
    D%numeric_done = .true.
    if (.not. D%sector) return

    ! the interior LU
    if (fresh) then
      call make_lu(D%kI, D%sub(1), "gmg_axsec_")
    else
      call KSPSetOperators(D%kI, D%sub(1), D%sub(1), ierr)
    endif
    call KSPSetUp(D%kI, ierr)

    ! this sector's Schur block C = A_GG(owned part) - A_GI A_II^-1 A_IG
    call MatConvert(D%sub(2), MATSEQDENSE, MAT_INITIAL_MATRIX, Bd, ierr)
    call MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, Zd, ierr)
    call KSPMatSolve(D%kI, Bd, Zd, ierr)
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
      call make_lu(D%kR, D%R, "gmg_axred_")
    else
      call KSPSetOperators(D%kR, D%R, D%R, ierr)
    endif
    call KSPSetUp(D%kR, ierr)

  contains

    !> entry (row k, column c) of A_GG belongs to this sector's block: the
    !! rows of its left separator, the border's couplings to that separator,
    !! and on sector 0 the border's own block -- so each entry is added once
    logical function take(k_, c_)
      integer, intent(in) :: k_, c_
      take = D%gcls(k_) == 1 .or. (D%gcls(k_) == 3 .and. D%gcls(c_) == 1) &
             .or. (D%gcls(k_) == 3 .and. D%gcls(c_) == 3 .and. D%sec == 0)
    end function take

    subroutine make_lu(ksp, M, pre)
      KSP :: ksp
      Mat :: M
      character(len=*), intent(in) :: pre
      call KSPCreate(PETSC_COMM_SELF, ksp, ierr)
      call KSPSetOperators(ksp, M, M, ierr)
      call KSPSetType(ksp, KSPPREONLY, ierr)
      call KSPGetPC(ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERPETSC, ierr)
      call PCFactorSetMatOrderingType(pc, MATORDERINGND, ierr)
      call KSPSetOptionsPrefix(ksp, pre, ierr)
      call KSPSetFromOptions(ksp, ierr)
    end subroutine make_lu
  end subroutine axd_numeric

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
      call KSPSolve(D%kI, D%tI, D%wI, ierr)
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
      call VecGetArray(D%gr, p3, ierr)
      p3 = D%g
      call VecRestoreArray(D%gr, p3, ierr)
      call KSPSolve(D%kR, D%gr, D%yr, ierr)
      ! y_G on this sector's Gamma rows, then y_I = A_II^-1 (x_I - A_IG y_G)
      call VecGetArrayRead(D%yr, p3, ierr)
      call VecGetArray(D%yG, p2, ierr)
      do k = 1, D%nga
        p2(k) = p3(D%gpos(k))
      enddo
      call VecRestoreArray(D%yG, p2, ierr)
      call VecRestoreArrayRead(D%yr, p3, ierr)
      call MatMult(D%sub(2), D%yG, D%tI, ierr)
      call VecGetArray(D%tI, p1, ierr)
      p1(1:D%ni) = xp(1:D%ni) - p1(1:D%ni)
      call VecRestoreArray(D%tI, p1, ierr)
      call VecRestoreArrayRead(D%xl, xp, ierr)
      call KSPSolve(D%kI, D%tI, D%yI, ierr)
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
  end subroutine axd_solve

#endif
end module mod_petsc_pc_gmg_axis
