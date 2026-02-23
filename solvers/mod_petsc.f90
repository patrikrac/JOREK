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
    integer :: n_local, n_global, n_block_local, n_block_global, block_size, block_size2, row_start_idx, row_end_idx
    integer :: r_start , r_end, c_global, block_col, val_ptr_start, val_ptr_end
    PetscInt, allocatable :: indices_petsc(:)
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscInt :: idxm(1), idxn(1)
    PetscScalar, allocatable :: vals_petsc(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm

    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size = a_mat%block_size
    block_size2 = block_size * block_size
    n_global = a_mat%ng
    n_block_global = n_global / block_size
    n_local = (a_mat%my_ind_max - a_mat%my_ind_min + 1) * block_size
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    row_start_idx = (a_mat%my_ind_min - 1)*block_size + 1
    row_end_idx = a_mat%my_ind_max*block_size

    if ((row_end_idx - row_start_idx + 1) /= n_local) print *, "[RANK ", my_id, "] WARNING: Somthing is wrong!"

    ! --- 1. Create vectors
    call VecCreateMPI(comm, n_local, n_global, petsc_sys%x, ierr)
    call VecDuplicate(petsc_sys%x, petsc_sys%b, ierr)

    allocate(indices_petsc(n_local))
    do i = 1, n_local
        indices_petsc(i) = (row_start_idx-1 + i) - 1
    end do

    call VecSetValues(petsc_sys%b, n_local, indices_petsc, rhs_vec%val(row_start_idx:row_end_idx), INSERT_VALUES, ierr)
    call VecSetValues(petsc_sys%x, n_local, indices_petsc, rhs_vec%val(row_start_idx:row_end_idx), INSERT_VALUES, ierr)
    deallocate(indices_petsc)

    call VecAssemblyBegin(petsc_sys%b, ierr)
    call VecAssemblyEnd(petsc_sys%b, ierr)
    call VecAssemblyBegin(petsc_sys%x, ierr)
    call VecAssemblyEnd(petsc_sys%x, ierr)

    print *, "[RANK ", my_id, "] PETSc Vectors created!"
    
    ! --- 1. Create matrix
    call MatCreate(comm, petsc_sys%A, ierr)
    call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_sys%A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_sys%A, block_size, ierr)

    print *, "[RANK ", my_id, "] Matrix created (MATMPIBAIJ, block_size=", block_size, ")"

    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0
    o_nnz = 0
    do i = 1, n_block_local
      r_start = a_mat%iblockptr(i)
      r_end = a_mat%iblockptr(i+1) - 1
        do k = r_start, r_end
          c_global = a_mat%jcn((k-1)*block_size2 + 1)
          block_col = (c_global / block_size) + 1 ! Fortran 1-based indexing
          if (block_col >= a_mat%my_ind_min .and. block_col <= a_mat%my_ind_max) then
            d_nnz(i) = d_nnz(i) + 1
          else
            o_nnz(i) = o_nnz(i) + 1
          endif
        enddo
    enddo

    print *, "[RANK ", my_id, "] d_nnz sum: ", sum(d_nnz), " o_nnz sum: ", sum(o_nnz)

    call MatMPIBAIJSetPreallocation(petsc_sys%A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    if (ierr /= 0) print *, "[RANK ", my_id, "] WARNING: MatMPIBAIJSetPreallocation ierr=", ierr
    deallocate(d_nnz, o_nnz)

    !call MatSetOption(petsc_sys%A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

    allocate(vals_petsc(block_size2))
    do i = 1, n_block_local
      idxm(1) = (a_mat%my_ind_min - 1) + (i - 1) ! 0-based PETSc indexing
      r_start = a_mat%iblockptr(i)
      r_end = a_mat%iblockptr(i+1) - 1

      do k = r_start, r_end
        c_global = a_mat%jcn((k-1)*block_size2 + 1)
        idxn(1) = c_global / block_size  ! 0-based PETSc indexing
        
        val_ptr_start = (k - 1) * block_size2 + 1
        val_ptr_end   = val_ptr_start + block_size2
        vals_petsc(1:block_size2) = a_mat%val(val_ptr_start : val_ptr_end)
        
        PetscCallA(MatSetValuesBlocked(petsc_sys%A, 1, idxm, 1, idxn, vals_petsc, INSERT_VALUES, ierr)) ! Secure PETSc call
      enddo
    enddo

    call MatAssemblyBegin(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)
    deallocate(vals_petsc)

    print *, "[RANK ", my_id, "] PETSc Matrix created!"
    if (my_id .eq. 0) print *, " --- System conversion successfull"
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
       print "(A, I0)", " NNZ used = ", int(info(MAT_INFO_NZ_USED))
       print "(A, I0)", " NNZ stored = ", int(info(MAT_INFO_NZ_ALLOCATED))
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
    integer :: i
    real*8, allocatable :: x_global(:) 
    real*8, allocatable :: y_jorek(:) 
    real*8 :: jorek_sum_sq, jorek_norm, petsc_norm
    Vec :: x, y_petsc
    PetscInt :: i_start, i_end, n_local
    PetscScalar, pointer :: x_arr(:)
    PetscLogDouble :: t1, t2, t3, t4

    comm = a_mat%comm

    allocate(x_global(a_mat%ng))
    allocate(y_jorek(a_mat%ng))

    call MPI_Comm_rank(MPI_COMM_WORLD, my_id, mpi_err)

    if (my_id .eq. 0) then
      call random_seed()
      call random_number(x_global)
      x_global = x_global * 1e-3
    endif

    call MPI_Bcast(x_global, a_mat%ng, MPI_DOUBLE_PRECISION, 0, comm, mpi_err)

    if (my_id .eq.0) then
      jorek_sum_sq = 0.0d0
      do i = 1,a_mat%ng
        jorek_sum_sq = jorek_sum_sq + (x_global(i))**2
      enddo
      jorek_norm = sqrt(jorek_sum_sq)
      print *, "JOREK Manual Norm (x): ", jorek_norm
    endif


    call MatCreateVecs(petsc_sys%A, x, PETSC_NULL_VEC, ierr)
    call VecGetOwnershipRange(x, i_start, i_end, ierr)
    n_local = i_end - i_start
    call VecGetArrayF90(x, x_arr, ierr)
    do i = 1, n_local
      x_arr(i) = x_global(i_start + i)  ! i_start+1 to i_end maps to x_global indices
    enddo 
    call VecRestoreArrayF90(x, x_arr, ierr)
    call VecAssemblyBegin(x, ierr)
    call VecAssemblyEnd(x, ierr)

    call VecNorm(x, NORM_2, petsc_norm, ierr)
    if (my_id .eq.0) print *, "PETSc Norm (x): ", petsc_norm

    call MatCreateVecs(petsc_sys%A, PETSC_NULL_VEC, y_petsc, ierr)
    call PetscTime(t1, ierr)
    PetscCallA(MatMult(petsc_sys%A, x, y_petsc, ierr))
    call PetscTime(t2, ierr)

    call PetscTime(t3, ierr)
    call bcsr_matv(a_mat, x_global, y_jorek)
    call PetscTime(t4, ierr)

    if (my_id .eq.0) then
      jorek_sum_sq = 0.0d0
      do i = 1,a_mat%ng
        jorek_sum_sq = jorek_sum_sq + (y_jorek(i))**2
      enddo
      jorek_norm = sqrt(jorek_sum_sq)
      print *, "JOREK Manual Norm (x): ", jorek_norm
    endif

    call VecNorm(y_petsc, NORM_2, petsc_norm, ierr)

    if (my_id .eq. 0) then
      print *, ""
      print *, "===== MATVEC COMPARISON ====="
      print *, "PETSc MatMult time: ", t2-t1
      print *, "JOREK MatVec time:  ", t4-t3
      print *, ""
      
      print *, "JOREK Norm: ", jorek_norm
      print *, "PETSc norm: ", petsc_norm
    endif

    deallocate(x_global, y_jorek)
    call VecDestroy(x, ierr)
    call VecDestroy(y_petsc, ierr)
  end subroutine petsc_test_matv


  subroutine petsc_solve_and_retrieve(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    PetscErrorCode :: ierr
    integer :: comm, my_id, mpierr
    PetscLogDouble :: t1, t2
    PC :: pc ! Maybe should be part of petsc_sys in the future
    KSPConvergedReason :: reason

    call PetscObjectGetComm(petsc_sys%A, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)

    ! Create solver Object
    call KSPCreate(comm, petsc_sys%ksp, ierr)
    print *, "DEBUG: Setting Operators"
    ! Link the operator A  
    call KSPSetOperators(petsc_sys%ksp, petsc_sys%A, petsc_sys%A, ierr)

    ! Configure the solver
    call KSPGetPC(petsc_sys%ksp, pc, ierr)
    call KSPSetType(petsc_sys%ksp, KSPPREONLY, ierr)
    call PCSetType(pc, PCLU, ierr)

    print *, "DEBUG: Setting MUMPS"

    call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
                            "-mat_mumps_icntl_14", "50", ierr)
    call KSPSetFromOptions(petsc_sys%ksp, ierr)

    print *, "DEBUG: Calling Solve"

    print *, "DEBUG: 4a. Setup (Symbolic Factorization)"
    call KSPSetUp(petsc_sys%ksp, ierr)  ! Does symbolic analysis
    print *, "DEBUG: 4b. Setup Done"

    print *, "DEBUG: 4c. Solve (Numeric Factorization)"
    call KSPSolve(petsc_sys%ksp, petsc_sys%b, petsc_sys%x, ierr)
    print *, "DEBUG: 4d. Solve Done"
    ! Perform the solve 
    !call KSPSolve(petsc_sys%ksp, petsc_sys%b, petsc_sys%x, ierr)

    print *, "DEBUG: Solve Done"

    ! Check result
    call KSPGetConvergedReason(petsc_sys%ksp, reason, ierr)
    if (reason < 0) then
        print *, "CRITICAL: Direct Solver Failed! Reason:", reason
        ! KSP_DIVERGED_NANORINF (-9) is common if matrix is singular
    end if

    ! Destroy the solver Object
    call KSPDestroy(petsc_sys%ksp, ierr)

  end subroutine petsc_solve_and_retrieve


  subroutine petsc_cleanup(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    PetscErrorCode :: ierr

    call VecDestroy(petsc_sys%b, ierr)
    call VecDestroy(petsc_sys%x, ierr)
    call MatDestroy(petsc_sys%A, ierr)

  end subroutine petsc_cleanup


  


  

#endif
end module mod_petsc