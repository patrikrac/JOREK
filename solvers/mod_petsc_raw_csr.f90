module mod_petsc_raw_csr
#ifdef USE_PETSC
  use iso_c_binding
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
#if defined(PETSC_USE_COMPLEX) || !defined(PETSC_USE_REAL_DOUBLE)
#error "mod_petsc_raw_csr: the value maps are real(c_double)"
#endif

  !> Raw CSR access to PETSc (MPI)AIJ/BAIJ matrices and the C kernels, via iso_c_binding (the Fortran
  !! MatMPIBAIJGetSeqBAIJ mishandles its colmap). Any failure ends the run on all ranks.

  interface
    integer(c_int) function c_mpibaij_seq(A, Ad, Ao, cmap) bind(C, name="MatMPIBAIJGetSeqBAIJ")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: A
      integer(c_intptr_t)        :: Ad, Ao
      type(c_ptr)                :: cmap
    end function c_mpibaij_seq
    integer(c_int) function c_mpiaij_seq(A, Ad, Ao, cmap) bind(C, name="MatMPIAIJGetSeqAIJ")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: A
      integer(c_intptr_t)        :: Ad, Ao
      type(c_ptr)                :: cmap
    end function c_mpiaij_seq
    integer(c_int) function c_baij_get(M, arr) bind(C, name="MatSeqBAIJGetArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_baij_get
    integer(c_int) function c_baij_restore(M, arr) bind(C, name="MatSeqBAIJRestoreArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_baij_restore
    integer(c_int) function c_aij_get(M, arr) bind(C, name="MatSeqAIJGetArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_get
    integer(c_int) function c_aij_restore(M, arr) bind(C, name="MatSeqAIJRestoreArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_restore
    ! read-only access: does not advance the matrix's object state
    integer(c_int) function c_aij_get_read(M, arr) bind(C, name="MatSeqAIJGetArrayRead")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_get_read
    integer(c_int) function c_aij_restore_read(M, arr) bind(C, name="MatSeqAIJRestoreArrayRead")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_restore_read
    ! the threaded block matvec of jorek_blockmv_attach.c (this declaration's
    ! form is what util/makedepend links the C file by)
    function jorek_blockmv_attach(M, bs, ok) bind(C)
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t
      integer(c_int)             :: jorek_blockmv_attach
      integer(c_intptr_t), value :: M
      integer(c_int), value      :: bs
      integer(c_int)             :: ok
    end function jorek_blockmv_attach
    ! the Galerkin product on dense slot blocks of jorek_galerkin_blk.c
    function jorek_galerkin_blk(A, Ps, bs, reuse, C, ok) bind(C)
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t
      integer(c_int)             :: jorek_galerkin_blk
      integer(c_intptr_t), value :: A, Ps
      integer(c_int), value      :: bs, reuse
      integer(c_intptr_t)        :: C
      integer(c_int)             :: ok
    end function jorek_galerkin_blk
#if defined(PETSC_USE_64BIT_INDICES)
#define C_PETSC_INT c_int64_t
#else
#define C_PETSC_INT c_int
#endif
    integer(c_int) function c_mat_get_row(M, row, ncols, cols, vals) bind(C, name="MatGetRow")
      use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_intptr_t, c_ptr
      integer(c_intptr_t), value  :: M
      integer(C_PETSC_INT), value :: row
      integer(C_PETSC_INT)        :: ncols
      type(c_ptr)                 :: cols, vals
    end function c_mat_get_row
    integer(c_int) function c_mat_restore_row(M, row, ncols, cols, vals) bind(C, name="MatRestoreRow")
      use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_intptr_t, c_ptr
      integer(c_intptr_t), value  :: M
      integer(C_PETSC_INT), value :: row
      integer(C_PETSC_INT)        :: ncols
      type(c_ptr)                 :: cols, vals
    end function c_mat_restore_row
    integer(c_int) function c_mat_create_submatrices(M, n, irow, icol, scall, submat) bind(C, name="MatCreateSubMatrices")
      use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_intptr_t, c_ptr
      integer(c_intptr_t), value  :: M
      integer(C_PETSC_INT), value :: n
      integer(c_intptr_t)         :: irow(*), icol(*)
      integer(c_int), value       :: scall
      type(c_ptr)                 :: submat
    end function c_mat_create_submatrices
    integer(c_int) function c_mat_destroy_submatrices(n, submat) bind(C, name="MatDestroySubMatrices")
      use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr
      integer(C_PETSC_INT), value :: n
      type(c_ptr)                 :: submat
    end function c_mat_destroy_submatrices
    integer(c_int) function c_pc_asm_get_sub_ksp(pc, n_local, first_local, ksp) bind(C, name="PCASMGetSubKSP")
      use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_intptr_t, c_ptr
      integer(c_intptr_t), value  :: pc
      integer(C_PETSC_INT)        :: n_local, first_local
      type(c_ptr)                 :: ksp
    end function c_pc_asm_get_sub_ksp
    integer(c_int) function c_petsc_object_get_id(obj, id) bind(C, name="PetscObjectGetId")
      use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_intptr_t
      integer(c_intptr_t), value  :: obj
      integer(c_int64_t)          :: id
    end function c_petsc_object_get_id
    integer(c_int) function c_mat_get_nonzero_state(M, state) bind(C, name="MatGetNonzeroState")
      use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_intptr_t
      integer(c_intptr_t), value  :: M
      integer(c_int64_t)          :: state
    end function c_mat_get_nonzero_state
#undef C_PETSC_INT
  end interface

  public :: c_baij_get, c_baij_restore, c_aij_get, c_aij_restore
  public :: split_parts, get_ij, put_ij, aij_parts, aij_vals_read, aij_vals_done
  public :: blockmv_attach, galerkin_blk
  public :: mat_get_row, mat_restore_row, mat_create_submatrices, mat_destroy_submatrices, pc_asm_get_sub_ksp
  public :: petsc_object_get_id, mat_get_nonzero_state

contains

  !> Ends the run on all ranks (a PETSc error may come from one rank only, inside collective work).
  subroutine fatal(what)
    character(len=*), intent(in) :: what
    integer :: ierr
    write(*,'(A,A)') "[Physics PC]   FATAL: ", what
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end subroutine fatal

  !> fatal() unless rc (a PETSc error code) is 0; PETSc has printed the error on this rank.
  subroutine chk(rc, what)
    integer(c_int), intent(in) :: rc
    character(len=*), intent(in) :: what
    character(len=16) :: num
    if (rc == 0) return
    write(num, '(I0)') rc
    call fatal(what//": PETSc error "//trim(num))
  end subroutine chk

  !> M's MatMult/MatMultAdd on the OpenMP kernel of jorek_blockmv_attach.c (.false.: PETSc's kept); collective, idempotent.
  logical function blockmv_attach(M, bs)
    Mat, intent(in)     :: M
    integer, intent(in) :: bs
    integer(c_int) :: ok
    ok = 0
    call chk(jorek_blockmv_attach(transfer(M%v, 0_c_intptr_t), int(bs, c_int), ok), "blockmv_attach")
    blockmv_attach = (ok /= 0)
  end function blockmv_attach

  !> C = P^T A P on dense slot blocks (jorek_galerkin_blk.c), reuse refills; .false. on all ranks if A / Ps do not fit.
  logical function galerkin_blk(A, Ps, bs, reuse, C)
    Mat, intent(in)     :: A, Ps
    integer, intent(in) :: bs
    logical, intent(in) :: reuse
    Mat, intent(inout)  :: C
    integer(c_int) :: ok
    integer(c_intptr_t) :: cv
    cv = 0
    if (reuse) cv = transfer(C%v, 0_c_intptr_t)
    ok = 0
    call chk(jorek_galerkin_blk(transfer(A%v, 0_c_intptr_t), transfer(Ps%v, 0_c_intptr_t), int(bs, c_int), &
                                merge(1_c_int, 0_c_int, reuse), cv, ok), "galerkin_blk")
    galerkin_blk = (ok /= 0)
    if (galerkin_blk .and. .not. reuse) C%v = transfer(cv, C%v)
  end function galerkin_blk

  !> Diagonal/off-diagonal sequential parts of an MPI(B)AIJ matrix, and the
  !! off-diagonal part's global (block) column of each local column.
  subroutine split_parts(M, is_baij, Md, Mo, garr)
    Mat, intent(in)  :: M
    logical, intent(in) :: is_baij
    Mat, intent(out) :: Md, Mo
    PetscInt, pointer, intent(out) :: garr(:)
    integer(c_intptr_t) :: pd, po
    type(c_ptr) :: cm
    PetscInt :: nr, nc, bs
    PetscErrorCode :: ierr
    if (is_baij) then
      call chk(c_mpibaij_seq(transfer(M%v, 0_c_intptr_t), pd, po, cm), "split_parts: MatMPIBAIJGetSeqBAIJ")
    else
      call chk(c_mpiaij_seq(transfer(M%v, 0_c_intptr_t), pd, po, cm), "split_parts: MatMPIAIJGetSeqAIJ")
    endif
    Md%v = transfer(pd, Md%v); Mo%v = transfer(po, Mo%v)
    call MatGetSize(Mo, nr, nc, ierr)
    call chk(ierr, "split_parts: MatGetSize")
    bs = 1
    if (is_baij) then
      call MatGetBlockSize(Mo, bs, ierr)
      call chk(ierr, "split_parts: MatGetBlockSize")
    endif
    if (nc / bs > 0) then
      call c_f_pointer(cm, garr, [nc / bs])
    else
      garr => null()
    endif
  end subroutine split_parts

  !> split_parts for an AIJ matrix that may also be sequential (a one-rank
  !! communicator can give either): has_o = .false. means M is its own
  !! diagonal part and there is no off-diagonal part. The AIJ families as
  !! MatMPIAIJGetSeqAIJ takes them (types beginning mpiaij / seqaij).
  subroutine aij_parts(M, Md, Mo, garr, has_o)
    Mat, intent(in)  :: M
    Mat, intent(out) :: Md, Mo
    PetscInt, pointer, intent(out) :: garr(:)
    logical, intent(out) :: has_o
    MatType :: mt
    PetscErrorCode :: ierr
    call MatGetType(M, mt, ierr)
    call chk(ierr, "aij_parts: MatGetType")
    has_o = (index(mt, "mpiaij") == 1)
    if (has_o) then
      call split_parts(M, .false., Md, Mo, garr)
    else if (index(mt, "seqaij") == 1) then
      Md = M; garr => null()
    else
      call fatal("raw CSR access needs an AIJ matrix, got "//trim(mt))
    endif
  end subroutine aij_parts

  !> Block CSR (BAIJ) or CSR (AIJ) of a sequential part, 0-based.
  subroutine get_ij(M, blocked, n, ia, ja)
    Mat, intent(in) :: M
    logical, intent(in) :: blocked
    PetscInt, intent(out) :: n
    PetscInt, pointer :: ia(:), ja(:)
    PetscBool :: done, bc
    PetscErrorCode :: ierr
    PetscInt, parameter :: zero = 0
    bc = PETSC_FALSE
    if (blocked) bc = PETSC_TRUE
#if PETSC_VERSION_LT(3,23,0)
    call MatGetRowIJF90(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
#else
    call MatGetRowIJ(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
#endif
    call chk(ierr, "get_ij: MatGetRowIJ")
    if (.not. done) call fatal("get_ij: MatGetRowIJ gave no row IJ")
  end subroutine get_ij

  subroutine put_ij(M, blocked, n, ia, ja)
    Mat, intent(in) :: M
    logical, intent(in) :: blocked
    PetscInt, intent(inout) :: n
    PetscInt, pointer :: ia(:), ja(:)
    PetscBool :: done, bc
    PetscErrorCode :: ierr
    PetscInt, parameter :: zero = 0
    bc = PETSC_FALSE
    if (blocked) bc = PETSC_TRUE
#if PETSC_VERSION_LT(3,23,0)
    call MatRestoreRowIJF90(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
#else
    call MatRestoreRowIJ(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
#endif
    call chk(ierr, "put_ij: MatRestoreRowIJ")
    if (.not. done) call fatal("put_ij: MatRestoreRowIJ failed")
  end subroutine put_ij

  !> Read-only value array (length n) of a sequential AIJ matrix; p is the
  !! handle aij_vals_done needs.
  subroutine aij_vals_read(M, n, p, v)
    Mat, intent(in) :: M
    integer(8), intent(in) :: n
    type(c_ptr), intent(out) :: p
    real(c_double), pointer, intent(out) :: v(:)
    call chk(c_aij_get_read(transfer(M%v, 0_c_intptr_t), p), "aij_vals_read: MatSeqAIJGetArrayRead")
    call c_f_pointer(p, v, [max(n, 1_8)])
  end subroutine aij_vals_read

  subroutine aij_vals_done(M, p)
    Mat, intent(in) :: M
    type(c_ptr), intent(inout) :: p
    call chk(c_aij_restore_read(transfer(M%v, 0_c_intptr_t), p), "aij_vals_done: MatSeqAIJRestoreArrayRead")
  end subroutine aij_vals_done

  !> PETSc < 3.23 Fortran pointer shims via stable C API.
  subroutine mat_get_row(M, row, ncols, cols, vals, ierr)
    Mat, intent(in) :: M
    PetscInt, intent(in) :: row
    PetscInt, intent(out) :: ncols
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode, intent(out) :: ierr
    type(c_ptr) :: c_cols, c_vals
    c_cols = c_null_ptr
    c_vals = c_null_ptr
    ierr = int(c_mat_get_row(transfer(M%v, 0_c_intptr_t), row, ncols, c_cols, c_vals), kind(ierr))
    if (ierr == 0 .and. c_associated(c_cols)) then
      call c_f_pointer(c_cols, cols, [max(int(ncols), 1)])
    else
      cols => null()
    endif
    if (ierr == 0 .and. c_associated(c_vals)) then
      call c_f_pointer(c_vals, vals, [max(int(ncols), 1)])
    else
      vals => null()
    endif
  end subroutine mat_get_row

  subroutine mat_restore_row(M, row, ncols, cols, vals, ierr)
    Mat, intent(in) :: M
    PetscInt, intent(in) :: row
    PetscInt, intent(inout) :: ncols
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode, intent(out) :: ierr
    type(c_ptr) :: c_cols, c_vals
    if (associated(cols)) then
      c_cols = c_loc(cols(1))
    else
      c_cols = c_null_ptr
    endif
    if (associated(vals)) then
      c_vals = c_loc(vals(1))
    else
      c_vals = c_null_ptr
    endif
    ierr = int(c_mat_restore_row(transfer(M%v, 0_c_intptr_t), row, ncols, c_cols, c_vals), kind(ierr))
    cols => null()
    vals => null()
  end subroutine mat_restore_row

  subroutine mat_create_submatrices(M, n, irow, icol, scall, sub, ierr)
    Mat, intent(in) :: M
    PetscInt, intent(in) :: n
    IS, intent(in) :: irow(*), icol(*)
    MatReuse, intent(in) :: scall
    Mat, pointer :: sub(:)
    PetscErrorCode, intent(out) :: ierr
    integer(c_intptr_t) :: crow(max(int(n), 1)), ccol(max(int(n), 1))
    type(c_ptr) :: c_sub
    integer :: k
    do k = 1, int(n)
      crow(k) = transfer(irow(k)%v, 0_c_intptr_t)
      ccol(k) = transfer(icol(k)%v, 0_c_intptr_t)
    enddo
    if (scall == MAT_REUSE_MATRIX .and. associated(sub)) then
      c_sub = c_loc(sub(1)%v)
    else
      c_sub = c_null_ptr
    endif
    ierr = int(c_mat_create_submatrices(transfer(M%v, 0_c_intptr_t), n, crow, ccol, int(scall, c_int), c_sub), kind(ierr))
    if (ierr == 0 .and. scall /= MAT_REUSE_MATRIX) then
      if (c_associated(c_sub)) then
        call c_f_pointer(c_sub, sub, [max(int(n), 1)])
      else
        sub => null()
      endif
    endif
  end subroutine mat_create_submatrices

  subroutine mat_destroy_submatrices(n, sub, ierr)
    PetscInt, intent(in) :: n
    Mat, pointer :: sub(:)
    PetscErrorCode, intent(out) :: ierr
    type(c_ptr) :: c_sub
    if (.not. associated(sub)) then
      ierr = 0
      return
    endif
    c_sub = c_loc(sub(1)%v)
    ierr = int(c_mat_destroy_submatrices(n, c_sub), kind(ierr))
    sub => null()
  end subroutine mat_destroy_submatrices

  subroutine pc_asm_get_sub_ksp(pc_in, n_local, first_local, sub, ierr)
    PC, intent(in) :: pc_in
    PetscInt, intent(out) :: n_local, first_local
    KSP, pointer :: sub(:)
    PetscErrorCode, intent(out) :: ierr
    type(c_ptr) :: c_sub
    c_sub = c_null_ptr
    ierr = int(c_pc_asm_get_sub_ksp(transfer(pc_in%v, 0_c_intptr_t), n_local, first_local, c_sub), kind(ierr))
    if (ierr == 0 .and. c_associated(c_sub)) then
      call c_f_pointer(c_sub, sub, [max(int(n_local), 1)])
    else
      sub => null()
    endif
  end subroutine pc_asm_get_sub_ksp

  subroutine petsc_object_get_id(M, id, ierr)
    Mat, intent(in) :: M
    PetscInt64, intent(out) :: id
    PetscErrorCode, intent(out) :: ierr
    integer(c_int64_t) :: cid
    cid = 0
    ierr = int(c_petsc_object_get_id(transfer(M%v, 0_c_intptr_t), cid), kind(ierr))
    id = int(cid, kind(id))
  end subroutine petsc_object_get_id

  subroutine mat_get_nonzero_state(M, state, ierr)
    Mat, intent(in) :: M
    PetscObjectState, intent(out) :: state
    PetscErrorCode, intent(out) :: ierr
    integer(c_int64_t) :: cst
    cst = 0
    ierr = int(c_mat_get_nonzero_state(transfer(M%v, 0_c_intptr_t), cst), kind(ierr))
    state = int(cst, kind(state))
  end subroutine mat_get_nonzero_state

#endif
end module mod_petsc_raw_csr
