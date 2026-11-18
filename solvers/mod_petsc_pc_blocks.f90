!> Operations on the packed diagonal blocks of the physics preconditioners:
!! the symmetric field scaling and the decoupling of penalty rows; and the
!! penalty rows of the full Jacobian with their back-substitution.
module mod_petsc_pc_blocks
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  !> The full Jacobian's penalty rows (eliminate_boundary_dofs = .f.; with
  !! .t. the test finds the eliminated rows, their diagonal alone): local
  !! offsets and diagonals, refreshed on every build. With the compact
  !! option, ap holds those rows of A alone (A's row and column layout), so
  !! that the back-substitution's product and its halo exchange are of
  !! boundary size rather than the whole Jacobian's.
  type, public :: penalty_rows_t
    Mat     :: a                           !< the full Jacobian
    integer :: n = 0                       !< penalty rows on the communicator
    integer, allocatable :: loc(:)
    real*8, allocatable  :: d(:)
    Vec     :: w
    logical :: w_made = .false.
    Mat     :: ap                          !< the penalty rows of a (compact option)
    logical :: ap_made = .false.
  end type penalty_rows_t

  public :: pc_print_block_setup, make_field_block_scale, decouple_penalty_rows
  public :: full_penalty_rows, penalty_backsub

contains

  !> "[Physics PC]   <label>: <method>" on the rank 0 of comm.
  subroutine pc_print_block_setup(comm, label, method)
    integer,          intent(in) :: comm
    character(len=*), intent(in) :: label, method
    integer        :: rank
    PetscErrorCode :: ierr
    call MPI_Comm_rank(comm, rank, ierr)
    if (rank == 0) write(*,'(A)') "[Physics PC]   " // trim(label) // ": " // trim(method)
  end subroutine pc_print_block_setup

  !> A <- D A D with D = diag(I, s_2 I, ..., s_nf I), s_f = sqrt(mean|diag| of
  !! field 1 / mean|diag| of field f). The caller solves (D A D) z = D b and
  !! takes x = D z: an exact similarity that evens out the scales of the
  !! operator and mass blocks and keeps the block structure.
  subroutine make_field_block_scale(A, nloc, dvec, comm, my_id, label)
    Mat, intent(inout)           :: A
    PetscInt, intent(in)         :: nloc(:)  !< local rows of each field, in pack order
    Vec, intent(inout)           :: dvec     !< out: D
    integer, intent(in)          :: comm, my_id
    character(len=*), intent(in) :: label
    PetscErrorCode :: ierr
    integer :: kk, f, nf, mpierr
    PetscScalar, pointer :: dptr(:)
    real*8, allocatable :: acc(:), m(:), sc(:)
    integer, allocatable :: fld(:)
    character(len=16) :: t1
    character(len=256) :: ss

    nf = size(nloc)
    allocate(acc(2 * nf), m(nf), sc(nf), fld(sum(nloc)))
    kk = 0
    do f = 1, nf
      fld(kk + 1:kk + nloc(f)) = f
      kk = kk + int(nloc(f))
    enddo

    call MatCreateVecs(A, dvec, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, dvec, ierr)
    acc = 0.d0
    call VecGetArray(dvec, dptr, ierr)
    do kk = 1, size(fld)
      acc(fld(kk))      = acc(fld(kk)) + abs(dptr(kk))
      acc(nf + fld(kk)) = acc(nf + fld(kk)) + 1.d0
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, acc, 2 * nf, MPI_DOUBLE_PRECISION, MPI_SUM, comm, mpierr)
    m = acc(1:nf) / max(acc(nf + 1:2 * nf), 1.d0)
    if (all(m > 0.d0)) then
      sc = sqrt(m(1) / m)
    else
      sc = 1.d0
      if (my_id == 0) write(*,'(A,A,A)') "[Physics PC]   WARNING: ", trim(label), &
        " block scaling skipped (a field has zero mean |diag|)"
    endif
    do kk = 1, size(fld)
      dptr(kk) = sc(fld(kk))
    enddo
    call VecRestoreArray(dvec, dptr, ierr)
    call MatDiagonalScale(A, dvec, dvec, ierr)

    if (my_id == 0 .and. nf > 1) then
      ss = ""
      do f = 2, nf
        write(t1, '(ES11.4)') sc(f)
        ss = trim(ss)//" "//adjustl(t1)
      enddo
      write(*,'(A,A,A,A)') "[Physics PC]   ", trim(label), " block scale s =", trim(ss)
    endif
  end subroutine make_field_block_scale

  !> Make the penalty rows of a diagonal block the Dirichlet rows that
  !! eliminate_boundary_dofs builds. Penalty boundary conditions put zbig on
  !! the diagonal of the whole row; left in, those rows set every field's mean
  !! |diag|, dominate the inner Krylov norms and stay in the GMG coarse spaces,
  !! which drop only diagonal-only rows. A row with |a_ii| > PEN_TAU max_j
  !! |a_ij| gets its row and column zeroed and its field's bulk mean |diag| on
  !! the diagonal; pfac = bulk / a_ii there (1 elsewhere), so a caller scaling
  !! the right-hand side by pfac still gets x_i = r_i / a_ii. The pattern is
  !! kept.
  subroutine decouple_penalty_rows(A, nloc, pfac, made, comm, my_id, label)
    Mat, intent(inout)           :: A
    PetscInt, intent(in)         :: nloc(:)  !< local rows of each field, in pack order
    Vec, intent(inout)           :: pfac     !< out: the right-hand-side factor
    logical, intent(inout)       :: made     !< pfac exists (refilled in place)
    integer, intent(in)          :: comm, my_id
    character(len=*), intent(in) :: label
    real*8, parameter :: PEN_TAU = 1.d6
    PetscErrorCode :: ierr
    PetscInt :: rs, re, row, ncols, npen
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:), dptr(:), pptr(:)
    PetscInt, allocatable :: prow(:)
    logical, allocatable :: pen(:)
    integer, allocatable :: fld(:)
    real*8, allocatable :: acc(:), m(:)
    integer :: nf, f, kk, jc, mpierr
    real*8 :: dg, off
    Vec :: dv
    character(len=16)  :: t1
    character(len=256) :: sn

    nf = size(nloc)
    call MatGetOwnershipRange(A, rs, re, ierr)
    allocate(pen(re - rs), fld(re - rs), prow(re - rs), acc(3 * nf), m(nf))
    kk = 0
    do f = 1, nf
      fld(kk + 1:kk + nloc(f)) = f
      kk = kk + int(nloc(f))
    enddo

    ! detect, and each field's bulk mean |diag|
    acc = 0.d0
    do row = rs, re - 1
      kk = int(row - rs) + 1
      call MatGetRow(A, row, ncols, cols, vals, ierr)
      dg = 0.d0; off = 0.d0
      do jc = 1, int(ncols)
        if (cols(jc) == row) then
          dg = abs(vals(jc))
        else
          off = max(off, abs(vals(jc)))
        endif
      enddo
      call MatRestoreRow(A, row, ncols, cols, vals, ierr)
      pen(kk) = (dg > 0.d0) .and. (dg > PEN_TAU * off)
      if (pen(kk)) then
        acc(2 * nf + fld(kk)) = acc(2 * nf + fld(kk)) + 1.d0
      else
        acc(fld(kk))      = acc(fld(kk)) + dg
        acc(nf + fld(kk)) = acc(nf + fld(kk)) + 1.d0
      endif
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, acc, 3 * nf, MPI_DOUBLE_PRECISION, MPI_SUM, comm, mpierr)
    m = acc(1:nf) / max(acc(nf + 1:2 * nf), 1.d0)
    where (.not. (m > 0.d0)) m = 1.d0

    if (.not. made) call MatCreateVecs(A, pfac, PETSC_NULL_VEC, ierr)
    made = .true.
    call MatCreateVecs(A, dv, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, dv, ierr)
    call VecGetArray(dv, dptr, ierr)
    call VecGetArray(pfac, pptr, ierr)
    npen = 0
    do kk = 1, size(pen)
      if (pen(kk)) then
        npen = npen + 1
        prow(npen) = rs + kk - 1
        pptr(kk) = m(fld(kk)) / dptr(kk)
        dptr(kk) = m(fld(kk))
      else
        pptr(kk) = 1.d0
      endif
    enddo
    call VecRestoreArray(pfac, pptr, ierr)
    call VecRestoreArray(dv, dptr, ierr)

    call MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    call MatZeroRowsColumns(A, npen, prow(1:npen), 1.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
    call MatDiagonalSet(A, dv, INSERT_VALUES, ierr)
    call VecDestroy(dv, ierr)

    if (my_id == 0) then
      sn = ""
      do f = 1, nf
        write(t1, '(I0)') nint(acc(2 * nf + f))
        sn = trim(sn)//merge(" ", "/", f == 1)//trim(t1)
      enddo
      write(*,'(A,A,A,A)') "[Physics PC]   ", trim(label), " penalty rows decoupled:", trim(sn)
    endif
  end subroutine decouple_penalty_rows

  !> The full Jacobian's penalty rows, for the back-substitution closing a
  !! preconditioner apply: |a_ii| > 1e6 max |a_ij| over the row's own
  !! variable (the Mach-1 rows also hold zbig couplings to other variables).
  !! compact (default .f.): also keep those rows as the matrix pr%ap, refilled
  !! in place while the detected rows stay the same, rebuilt when they change.
  subroutine full_penalty_rows(pr, A, comm, my_id, report, compact)
    use mod_parameters, only: n_tor
    type(penalty_rows_t), intent(inout) :: pr
    Mat :: A
    integer, intent(in) :: comm, my_id
    logical, intent(in) :: report             !< print the count
    logical, intent(in), optional :: compact
    PetscInt :: rs, re, cs, ce, row, ncols, bs
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    integer :: jc, n, iv, mpierr
    logical :: want_ap
    real*8 :: dg, off
    integer, allocatable :: tl(:)
    real*8, allocatable :: td(:)
    PetscInt, allocatable :: tnd(:), tno(:)  !< a detected row's entries inside / outside A's diagonal block
    want_ap = .false.
    if (present(compact)) want_ap = compact
    pr%a = A
    call MatGetBlockSize(A, bs, ierr)
    call MatGetOwnershipRange(A, rs, re, ierr)
    call MatGetOwnershipRangeColumn(A, cs, ce, ierr)
    allocate(tl(re - rs), td(re - rs), tnd(re - rs), tno(re - rs))
    n = 0
    do row = rs, re - 1
      iv = int(mod(row, bs)) / n_tor
      call MatGetRow(A, row, ncols, cols, vals, ierr)
      dg = 0.d0; off = 0.d0
      do jc = 1, int(ncols)
        if (cols(jc) == row) then
          dg = vals(jc)
        else if (int(mod(cols(jc), bs)) / n_tor == iv) then
          off = max(off, abs(vals(jc)))
        endif
      enddo
      if (abs(dg) > 0.d0 .and. abs(dg) > 1.d6 * off) then
        n = n + 1; tl(n) = int(row - rs) + 1; td(n) = dg
        tnd(n) = count(cols(1:ncols) >= cs .and. cols(1:ncols) < ce)
        tno(n) = ncols - tnd(n)
      endif
      call MatRestoreRow(A, row, ncols, cols, vals, ierr)
    enddo
    if (want_ap) call penalty_matrix()
    pr%loc = tl(1:n); pr%d = td(1:n)
    call MPI_Allreduce(n, pr%n, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
    if (.not. pr%w_made) call MatCreateVecs(A, pr%w, PETSC_NULL_VEC, ierr)
    pr%w_made = .true.
    if (my_id == 0 .and. report) write(*,'(A,I0,A)') "[Physics PC]   full Jacobian: ", pr%n, &
      " penalty rows, back-substituted after every sweep"

  contains

    !> pr%ap = the detected rows of A: refilled in place if they are the
    !! stored ones on every rank, created anew otherwise. Before pr%loc is
    !! overwritten.
    subroutine penalty_matrix()
      PetscInt, parameter :: ZERO = 0, ONE = 1
      PetscInt, allocatable :: dnz(:), onz(:)
      PetscInt :: q
      logical :: same, same_all
      same = pr%ap_made
      if (same) same = (size(pr%loc) == n)
      if (same) same = all(pr%loc == tl(1:n))
      call MPI_Allreduce(same, same_all, 1, MPI_LOGICAL, MPI_LAND, comm, mpierr)
      if (.not. same_all) then
        if (pr%ap_made) call MatDestroy(pr%ap, ierr)
        allocate(dnz(re - rs), onz(re - rs))
        dnz = 0; onz = 0
        do q = 1, n
          dnz(tl(q)) = tnd(q); onz(tl(q)) = tno(q)
        enddo
        call MatCreateAIJ(comm, re - rs, ce - cs, PETSC_DETERMINE, PETSC_DETERMINE, ZERO, dnz, ZERO, onz, pr%ap, ierr)
        pr%ap_made = .true.
      endif
      do q = 1, n
        row = rs + tl(q) - 1
        call MatGetRow(A, row, ncols, cols, vals, ierr)
        call MatSetValues(pr%ap, ONE, [row], ncols, cols, vals, INSERT_VALUES, ierr)
        call MatRestoreRow(A, row, ncols, cols, vals, ierr)
      enddo
      call MatAssemblyBegin(pr%ap, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(pr%ap, MAT_FINAL_ASSEMBLY, ierr)
    end subroutine penalty_matrix

  end subroutine full_penalty_rows

  !> The penalty rows' z_b += (r_b - sum_j a_bj z_j) / a_bb with the
  !! preconditioner's z: their residual is zbig times z_b's error, which the
  !! decoupled block rows leave at the size of the whole outer residual.
  !! Changes the other rows' residual by O(1 / zbig). The product is pr%ap's
  !! (the penalty rows alone) where it exists, the whole Jacobian's otherwise.
  subroutine penalty_backsub(pr, r, z)
    type(penalty_rows_t), intent(inout) :: pr
    Vec :: r, z
    PetscScalar, pointer :: zp(:), rp(:), wp(:)
    PetscErrorCode :: ierr
    integer :: q
    if (pr%ap_made) then
      call MatMult(pr%ap, z, pr%w, ierr)
    else
      call MatMult(pr%a, z, pr%w, ierr)
    endif
    call VecGetArray(z, zp, ierr)
    call VecGetArrayRead(r, rp, ierr)
    call VecGetArrayRead(pr%w, wp, ierr)
    do q = 1, size(pr%loc)
      zp(pr%loc(q)) = zp(pr%loc(q)) + (rp(pr%loc(q)) - wp(pr%loc(q))) / pr%d(q)
    enddo
    call VecRestoreArrayRead(pr%w, wp, ierr)
    call VecRestoreArrayRead(r, rp, ierr)
    call VecRestoreArray(z, zp, ierr)
  end subroutine penalty_backsub

#endif
end module mod_petsc_pc_blocks
