!> Operations on the packed diagonal blocks of the SF preconditioner: the
!! symmetric field scaling and the decoupling of penalty rows.
module mod_petsc_pc_blocks
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
#if PETSC_VERSION_LT(3,23,0)
  use mod_petsc_raw_csr, only: mat_get_row, mat_restore_row
#endif
#include "petsc_f90_compat.h"
  implicit none
  private

  public :: pc_print_block_setup, make_field_block_scale, decouple_penalty_rows

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
    real*8, parameter :: PEN_TAU = 1.d6, PEN_DIAG_MIN = 1.d7
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
    allocate(pen(re - rs), fld(re - rs), prow(max(1, int(re - rs))), acc(3 * nf), m(nf))
    kk = 0
    do f = 1, nf
      fld(kk + 1:kk + nloc(f)) = f
      kk = kk + int(nloc(f))
    enddo

    ! detect, and each field's bulk mean |diag| (require dg >= PEN_DIAG_MIN so
    ! eliminated Dirichlet rows with diag = 1, off = 0 are not misclassified)
    call MatCreateVecs(A, dv, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, dv, ierr)
    call VecGetArray(dv, dptr, ierr)
    acc = 0.d0
    do row = rs, re - 1
      kk = int(row - rs) + 1
      dg = abs(dptr(kk))
      if (dg >= PEN_DIAG_MIN) then
        call MatGetRow(A, row, ncols, cols, vals, ierr)
        off = 0.d0
        do jc = 1, int(ncols)
          if (cols(jc) /= row) off = max(off, abs(vals(jc)))
        enddo
        call MatRestoreRow(A, row, ncols, cols, vals, ierr)
        pen(kk) = (dg > PEN_TAU * off)
      else
        pen(kk) = .false.
      endif
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
    call MatZeroRowsColumns(A, npen, prow, 1.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
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

#endif
end module mod_petsc_pc_blocks
