module mod_petsc_raw_csr
#ifdef USE_PETSC
  use iso_c_binding
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  !--------------------------------------------------------------------
  !> Raw CSR access to PETSc AIJ/BAIJ matrices, for the value maps of the
  !! physics PC (mod_petsc_pc_sf_gather, the GMG smoother blocks).
  !!
  !! A value map records, once per run, where each stored entry of a target
  !! comes from in a source's value array; a rebuild is then one indexed copy.
  !! That needs the diagonal/off-diagonal split of an MPI matrix and the
  !! value arrays of its sequential parts, which PETSc exposes only in C (the
  !! Fortran binding of MatMPIBAIJGetSeqBAIJ mishandles its colmap), so those
  !! calls go through iso_c_binding on the handles' addresses.
  !--------------------------------------------------------------------

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
  end interface

  public :: c_baij_get, c_baij_restore, c_aij_get, c_aij_restore
  public :: split_parts, get_ij, put_ij, aij_parts, aij_vals_read, aij_vals_done

contains

  !> Diagonal/off-diagonal sequential parts of an MPI(B)AIJ matrix, and the
  !! off-diagonal part's global (block) column of each local column.
  subroutine split_parts(M, is_baij, Md, Mo, garr)
    Mat, intent(in)  :: M
    logical, intent(in) :: is_baij
    Mat, intent(out) :: Md, Mo
    PetscInt, pointer, intent(out) :: garr(:)
    integer(c_intptr_t) :: pd, po
    type(c_ptr) :: cm
    integer(c_int) :: rc
    PetscInt :: nr, nc, bs
    PetscErrorCode :: ierr
    if (is_baij) then
      rc = c_mpibaij_seq(transfer(M%v, 0_c_intptr_t), pd, po, cm)
    else
      rc = c_mpiaij_seq(transfer(M%v, 0_c_intptr_t), pd, po, cm)
    endif
    Md%v = transfer(pd, Md%v); Mo%v = transfer(po, Mo%v)
    call MatGetSize(Mo, nr, nc, ierr)
    bs = 1
    if (is_baij) call MatGetBlockSize(Mo, bs, ierr)
    if (nc / bs > 0) then
      call c_f_pointer(cm, garr, [nc / bs])
    else
      garr => null()
    endif
  end subroutine split_parts

  !> split_parts for an AIJ matrix that may also be sequential (a one-rank
  !! communicator can give either): has_o = .false. means M is its own
  !! diagonal part and there is no off-diagonal part.
  subroutine aij_parts(M, Md, Mo, garr, has_o)
    Mat, intent(in)  :: M
    Mat, intent(out) :: Md, Mo
    PetscInt, pointer, intent(out) :: garr(:)
    logical, intent(out) :: has_o
    MatType :: mt
    PetscErrorCode :: ierr
    call MatGetType(M, mt, ierr)
    has_o = (index(mt, "mpi") == 1)
    if (has_o) then
      call split_parts(M, .false., Md, Mo, garr)
    else if (index(mt, "seqaij") == 1) then
      Md = M; garr => null()
    else
      write(*,'(A,A)') "[Physics PC]   FATAL: raw CSR access needs an AIJ matrix, got ", trim(mt)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
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
    call MatGetRowIJ(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
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
    call MatRestoreRowIJ(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
  end subroutine put_ij

  !> Read-only value array (length n) of a sequential AIJ matrix; p is the
  !! handle aij_vals_done needs.
  subroutine aij_vals_read(M, n, p, v)
    Mat, intent(in) :: M
    integer(8), intent(in) :: n
    type(c_ptr), intent(out) :: p
    real(c_double), pointer, intent(out) :: v(:)
    integer(c_int) :: rc
    rc = c_aij_get_read(transfer(M%v, 0_c_intptr_t), p)
    call c_f_pointer(p, v, [max(n, 1_8)])
  end subroutine aij_vals_read

  subroutine aij_vals_done(M, p)
    Mat, intent(in) :: M
    type(c_ptr), intent(inout) :: p
    integer(c_int) :: rc
    rc = c_aij_restore_read(transfer(M%v, 0_c_intptr_t), p)
  end subroutine aij_vals_done

#endif
end module mod_petsc_raw_csr
