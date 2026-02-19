module mod_petsc
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc

  implicit none


  type type_PETSC_SYSTEM
    Mat :: A
    Vec :: x, b
    KSP :: ksp
  end type type_PETSC_SYSTEM


contains

  subroutine petsc_initialize()
    PetscErrorCode :: ierr
    call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
    if (ierr /= 0) print *, "Error initializing PETSc"
  end subroutine


  subroutine petsc_finalize()
    PetscErrorCode :: ierr
    call PetscFinalize(ierr)
  end subroutine petsc_finalize


  subroutine petsc_print_version()
    PetscErrorCode :: ierr
    character(len=256) :: version_string

    call PetscGetVersion(version_string, ierr)
    if (ierr == 0) then
      print *, "----------------------------------------"
      print *, "JOREK linked with ", trim(version_string)
      print *, "----------------------------------------"
    end if
  end subroutine petsc_print_version


  subroutine petsc_convert_jorek_system(a_mat, rhs_vec, petsc_sys)
    use data_structure, only: type_SP_MATRIX, type_RHS 

    type(type_SP_MATRIX), intent(in) :: a_mat
    type(type_RHS), intent(in) :: rhs_vec
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys

    integer :: i, k
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, block_size2, row_start_idx, row_end_idx
    integer :: r_start , r_end, c_global, val_ptr_start, val_ptr_end
    PetscInt, allocatable :: indices_petsc(:)
    PetscInt :: num_rows_petsc
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscInt, allocatable :: rows_petsc(:), cols_petsc(:)
    PetscScalar, allocatable :: vals_petsc(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm

    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size = a_mat%block_size
    block_size2 = block_size * block_size
    n_global = a_mat%ng
    n_local = (a_mat%my_ind_max - a_mat%my_ind_min + 1) * block_size
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    row_start_idx = (a_mat%my_ind_min - 1)*block_size + 1
    row_end_idx = a_mat%my_ind_max*block_size


    if ((row_end_idx - row_start_idx + 1) /= n_local) print *, "WARNING: Somthing is wrong!"

    ! --- 1. Create vectors
    call VecCreateMPI(comm, n_local, n_global, petsc_sys%x, ierr)
    call VecDuplicate(petsc_sys%x, petsc_sys%b, ierr)

    print *, "PETSC Vectors created!"
    
    allocate(indices_petsc(n_local))
    do i = 1, n_local
        indices_petsc(i) = (row_start_idx-1 + i) - 1
    end do

    call VecSetValues(petsc_sys%b, n_local, indices_petsc, rhs_vec%val(row_start_idx:row_end_idx), INSERT_VALUES, ierr)
    call VecSetValues(petsc_sys%x, n_local, indices_petsc, rhs_vec%val(row_start_idx:row_end_idx), INSERT_VALUES, ierr)
    deallocate(indices_petsc)
    print *, "PETSC Vector values set!"
    call VecAssemblyBegin(petsc_sys%b, ierr)
    call VecAssemblyEnd(petsc_sys%b, ierr)
    call VecAssemblyBegin(petsc_sys%x, ierr)
    call VecAssemblyEnd(petsc_sys%x, ierr)
    
    ! --- 1. Create matrix
    call MatCreate(comm, petsc_sys%A, ierr)
    call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_sys%A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_sys%A, block_size, ierr)

    print *, "PETSC Matrix created!"

    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0; o_nnz = 0
    do i = 1, n_block_local
      r_start = a_mat%iblockptr(i); r_end = a_mat%iblockptr(i+1)-1
      do k = r_start, r_end
        c_global = a_mat%jcn((k-1)*block_size2+1)

        if (c_global >= row_start_idx .and. c_global <= row_end_idx) then
          d_nnz(i) = d_nnz(i) + 1  ! Diagonal (Local)
        else
          o_nnz(i) = o_nnz(i) + 1  ! Off-Diagonal (Remote)
        endif
      enddo
    enddo

    call MatMPIBAIJSetPreallocation(petsc_sys%A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    deallocate(d_nnz, o_nnz)

    allocate(rows_petsc(1), cols_petsc(1))
    allocate(vals_petsc(block_size2))
    do i = 1, n_block_local
      rows_petsc(1) = (a_mat%my_ind_min + i - 1) - 1
      r_start = a_mat%iblockptr(i); r_end = a_mat%iblockptr(i+1)-1
      do k = r_start, r_end
        cols_petsc(1) = a_mat%jcn((k-1)*block_size2+1)/block_size

        val_ptr_start = (k - 1) * block_size2 + 1
        val_ptr_end   = val_ptr_start + block_size2 - 1
        vals_petsc = a_mat%val(val_ptr_start : val_ptr_end)
        call MatSetValuesBlocked(petsc_sys%A, 1, rows_petsc, 1, cols_petsc, vals_petsc, INSERT_VALUES, ierr)
      enddo
    enddo

    print *, "PETSC Matrix values set!" 

    call MatAssemblyBegin(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)
    deallocate(rows_petsc, cols_petsc, vals_petsc)

    if (my_id .eq. 0) print *, "System conversion successfull"
  end subroutine petsc_convert_jorek_system


  subroutine petsc_calc_vec_norm(petsc_sys, b_norm)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    real*8, intent(out) :: b_norm

    PetscErrorCode :: ierr

    call VecNorm(petsc_sys%b, NORM_2, b_norm, ierr)
  end subroutine petsc_calc_vec_norm


  subroutine petsc_print_matrix_info(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    PetscErrorCode :: ierr
    integer ::  my_id, comm, mpierr
    logical :: speaker
    PetscInt :: M, N
    MatInfo :: info(MAT_INFO_SIZE)
    PetscReal :: norm
    PetscBool :: flg

    call PetscObjectGetComm(petsc_sys%A, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)

    speaker = (my_id .eq. 0)

    if (speaker) print *, "Start PETSC Matrix info ---"
    call MatGetSize(petsc_sys%A, M, N, ierr)
    if (speaker) print *, "Matrix size M = ", M, " ; N = ", N
    call MatGetInfo(petsc_sys%A, MAT_GLOBAL_SUM, info, ierr) 
    if (speaker) then
       print *, "NNZ used = ", info(MAT_INFO_NZ_USED)
       print *, "NNZ stored = ", info(MAT_INFO_NZ_ALLOCATED)
    endif

    if (speaker) print *, "End PETSC Matrix info ---"

  end subroutine petsc_print_matrix_info


  subroutine petsc_test_matv(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX 
    use mod_matv, only: bcsr_matv

    type(type_PETSC_SYSTEM), intent(in) :: petsc_sys
    type(type_SP_MATRIX), intent(in) :: a_mat
    PetscErrorCode :: ierr
    integer :: comm, my_id, mpi_err

    real*8, allocatable :: x_global(:) 
    real*8, allocatable :: y_jorek(:) 
    Vec :: x, y_petsc
    PetscInt       :: i_start, i_end, n_local
    PetscScalar, pointer :: x_arr(:)
    PetscScalar, pointer :: y_arr(:)
    PetscLogDouble :: t1, t2, t3, t4

    comm = a_mat%comm

    allocate(x_global(a_mat%ng))

    call MPI_Comm_rank(MPI_COMM_WORLD, my_id, mpi_err)

    if (my_id .eq. 0) then
      call random_seed()
      call random_number(x_global)
    endif

    call MPI_Bcast(x_global, a_mat%ng, MPI_DOUBLE_PRECISION, 0, comm, mpi_err)

    call MatCreateVecs(petsc_sys%A, x, PETSC_NULL_VEC, ierr)
    call VecGetOwnershipRange(x, i_start, i_end, ierr)
    n_local = i_end - i_start
    call VecGetArrayF90(x, x_arr, ierr)
    x_arr(1:n_local) = x_global(i_start+1:i_end) 
    call VecRestoreArrayF90(x, x_arr, ierr)
    call VecAssemblyBegin(x, ierr)
    call VecAssemblyEnd(x, ierr)

    call MatCreateVecs(petsc_sys%A, PETSC_NULL_VEC, y_petsc, ierr)
    call PetscTime(t1, ierr)
    call MatMult(petsc_sys%A, x, y_petsc, ierr)
    call PetscTime(t2, ierr)
    if (my_id .eq. 0) print *, "PETSc MatMult time: ", t2-t1


    allocate(y_jorek(a_mat%ng))
    call PetscTime(t3, ierr)
    call bcsr_matv(a_mat, x_global, y_jorek)
    call PetscTime(t4, ierr)
    if (my_id .eq. 0) print *, "JOREK MatVec time: ", t4-t3

    call VecGetArrayReadF90(y_petsc, y_arr, ierr)
    write(*,*) "Max local diff:", maxval(abs(y_arr - y_jorek(i_start+1:i_end)))
    call VecRestoreArrayReadF90(y_petsc, y_arr, ierr)

    deallocate(x_global, y_jorek)
    call VecDestroy(x, ierr)
    call VecDestroy(y_petsc, ierr)
  end subroutine petsc_test_matv

  subroutine petsc_cleanup(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    PetscErrorCode :: ierr

    call VecDestroy(petsc_sys%b, ierr)
    call VecDestroy(petsc_sys%x, ierr)
    call MatDestroy(petsc_sys%A, ierr)

  end subroutine petsc_cleanup


  


  

#endif
end module mod_petsc