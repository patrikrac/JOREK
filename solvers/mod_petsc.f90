module mod_petsc
#ifdef USE_PETSC
  use mpi_mod
  use mod_petsc_pc
#include "petsc/finclude/petsc.h"
  use petsc
#ifdef USE_SLEPC
#include "slepc/finclude/slepceps.h"
  use slepceps
#endif

  implicit none


  type type_PETSC_SYSTEM
    Mat  :: A              ! BAIJ system matrix (from JOREK block-CSR)
    Mat  :: A_aij          ! AIJ version used by KSP (persistent)
    Vec  :: x, b           ! BAIJ solution/RHS vectors
    Vec  :: x_aij, b_aij   ! AIJ solution/RHS vectors for KSP (persistent)
    KSP  :: ksp            ! Krylov solver context (persistent)
    logical :: initialized   = .false.  ! A, x, b created
    logical :: owns_A        = .false.  ! .true. when A was created by petsc_init_system (old path)
    logical :: ksp_ready     = .false.  ! KSP, A_aij, PC setup + factored
    PetscLogStage :: stage_setup = -1
    PetscLogStage :: stage_solve = -1
  end type type_PETSC_SYSTEM


contains

  subroutine petsc_initialize()
    PetscErrorCode :: ierr
    ! PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-log_view", PETSC_NULL_CHARACTER, ierr))
#ifdef USE_SLEPC
    call SlepcInitialize(PETSC_NULL_CHARACTER, ierr)  ! superset of PetscInitialize
    if (ierr /= 0) print *, "Error initializing SLEPc/PETSc"
#else
    call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
    if (ierr /= 0) print *, "Error initializing PETSc"
#endif
  end subroutine


  subroutine petsc_finalize()
    PetscErrorCode :: ierr
#ifdef USE_SLEPC
    call SlepcFinalize(ierr)
#else
    call PetscFinalize(ierr)
#endif
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


  !> Initialize BAIJ matrix structure and BAIJ vecs — called once when !initialized
  !TODO: Redundant with petsc_create_matrix. One must go...
  subroutine petsc_init_system(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(in) :: a_mat

    integer :: i, k
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, block_size2, row_start_idx, row_end_idx
    integer :: r_start, r_end, c_global, block_col
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
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

    if ((row_end_idx - row_start_idx + 1) /= n_local) &
      write(*,*) "[RANK ", my_id, "] WARNING: Something is wrong in petsc_init_system!"

    ! Create matrix
    call MatCreate(comm, petsc_sys%A, ierr)
    call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_sys%A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_sys%A, block_size, ierr)

    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0
    o_nnz = 0
    do i = 1, n_block_local
      r_start = a_mat%iblockptr(i)
      r_end = a_mat%iblockptr(i+1) - 1
      do k = r_start, r_end
        c_global = a_mat%jcn((k-1)*block_size2 + 1)
        block_col = (c_global / block_size) + 1
        if (block_col >= a_mat%my_ind_min .and. block_col <= a_mat%my_ind_max) then
          d_nnz(i) = d_nnz(i) + 1
        else
          o_nnz(i) = o_nnz(i) + 1
        endif
      enddo
    enddo

    call MatMPIBAIJSetPreallocation(petsc_sys%A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: MatMPIBAIJSetPreallocation ierr=", ierr
    deallocate(d_nnz, o_nnz)

    call MatCreateVecs(petsc_sys%A, petsc_sys%x, petsc_sys%b, ierr)

    petsc_sys%initialized = .true.
    petsc_sys%owns_A = .true.
    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0)') "[PETSc] init: BAIJ matrix ", n_global, "x", n_global, &
                                                    ", block_size=", block_size
  end subroutine petsc_init_system


  !> Create and preallocate a PETSc MPIBAIJ matrix from the JOREK block structure
  !! (ijA_size, irn_jcn). Does not require iblockptr (block-CSR).
  subroutine petsc_create_matrix(petsc_A, a_mat)
    use data_structure, only: type_SP_MATRIX

    Mat, intent(out)                    :: petsc_A
    type(type_SP_MATRIX), intent(in)    :: a_mat

    integer :: i, j
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, col_block
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size    = a_mat%block_size
    n_global      = a_mat%ng
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local       = n_block_local * block_size

    ! Compute diagonal/off-diagonal block counts per block row
    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0
    o_nnz = 0
    do i = 1, n_block_local
      do j = 1, a_mat%ijA_size(i)
        col_block = a_mat%irn_jcn(i, j)
        if (col_block >= a_mat%my_ind_min .and. col_block <= a_mat%my_ind_max) then
          d_nnz(i) = d_nnz(i) + 1
        else
          o_nnz(i) = o_nnz(i) + 1
        endif
      enddo
    enddo

    ! Create and preallocate
    call MatCreate(comm, petsc_A, ierr)
    call MatSetSizes(petsc_A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_A, block_size, ierr)
    call MatMPIBAIJSetPreallocation(petsc_A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: petsc_create_matrix preallocation ierr=", ierr
    deallocate(d_nnz, o_nnz)

    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0)') &
      "[PETSc] create_matrix: BAIJ ", n_global, "x", n_global, ", block_size=", block_size
  end subroutine petsc_create_matrix


  !> Create an n_vars-variable MPIBAIJ PC matrix.
  !! block_size = n_vars * n_tor_local;  n_global = n_vars * a_mat%ng / n_var.
  subroutine petsc_create_pc_matrix(petsc_A, a_mat, n_vars)
    use data_structure,  only: type_SP_MATRIX
    use mod_parameters,  only: n_var

    Mat,                   intent(out) :: petsc_A
    type(type_SP_MATRIX),  intent(in)  :: a_mat
    integer,               intent(in)  :: n_vars   ! 1 or 2

    integer :: i, j
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, col_block
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size    = n_vars * a_mat%block_size / n_var
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local       = n_block_local * block_size
    n_global      = n_vars * a_mat%ng / n_var

    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0
    o_nnz = 0
    do i = 1, n_block_local
      do j = 1, a_mat%ijA_size(i)
        col_block = a_mat%irn_jcn(i, j)
        if (col_block >= a_mat%my_ind_min .and. col_block <= a_mat%my_ind_max) then
          d_nnz(i) = d_nnz(i) + 1
        else
          o_nnz(i) = o_nnz(i) + 1
        endif
      enddo
    enddo

    call MatCreate(comm, petsc_A, ierr)
    call MatSetSizes(petsc_A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_A, block_size, ierr)
    call MatMPIBAIJSetPreallocation(petsc_A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: petsc_create_pc_matrix ierr=", ierr
    deallocate(d_nnz, o_nnz)

    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0,A,I0)') &
      "[PETSc] create_pc_matrix (", n_vars, "-var): BAIJ ", n_global, "x", n_global, &
      ", block_size=", block_size
  end subroutine petsc_create_pc_matrix


  !> Fill matrix values from JOREK block-CSR
  subroutine petsc_update_matrix(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(in) :: a_mat

    integer :: i, k
    integer :: my_id, mpierr
    integer :: n_block_local, block_size, block_size2
    integer :: r_start, r_end, c_global, val_ptr_start, val_ptr_end
    PetscInt :: idxm(1), idxn(1)
    PetscScalar, allocatable :: vals_petsc(:)
    PetscErrorCode :: ierr

    call MPI_COMM_RANK(a_mat%comm, my_id, mpierr)

    block_size = a_mat%block_size
    block_size2 = block_size * block_size
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1

    call MatZeroEntries(petsc_sys%A, ierr)

    allocate(vals_petsc(block_size2))
    do i = 1, n_block_local
      idxm(1) = (a_mat%my_ind_min - 1) + (i - 1)
      r_start = a_mat%iblockptr(i)
      r_end = a_mat%iblockptr(i+1) - 1
      do k = r_start, r_end
        c_global = a_mat%jcn((k-1)*block_size2 + 1)
        idxn(1) = c_global / block_size
        val_ptr_start = (k - 1) * block_size2 + 1
        val_ptr_end   = val_ptr_start + block_size2
        vals_petsc(1:block_size2) = a_mat%val(val_ptr_start : val_ptr_end)
        PetscCallA(MatSetValuesBlocked(petsc_sys%A, 1, idxm, 1, idxn, vals_petsc, INSERT_VALUES, ierr))
      enddo
    enddo

    PetscCallA(MatAssemblyBegin(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr))
    deallocate(vals_petsc)
  end subroutine petsc_update_matrix


  !> Fill RHS vector from JOREK rhs — called every time
  subroutine petsc_update_rhs(petsc_sys, rhs_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(in) :: rhs_vec

    integer :: i, my_id, mpierr, comm
    PetscInt :: i_start, i_end, n_local
    PetscInt, allocatable :: indices_petsc(:)
    PetscErrorCode :: ierr

    PetscCallA(PetscObjectGetComm(petsc_sys%b, comm, ierr))
    call MPI_COMM_RANK(comm, my_id, mpierr)

    PetscCallA(VecGetOwnershipRange(petsc_sys%b, i_start, i_end, ierr))
    n_local = i_end - i_start

    allocate(indices_petsc(n_local))
    do i = 1, n_local
      indices_petsc(i) = i_start + (i - 1)
    end do

    PetscCallA(VecSetValues(petsc_sys%b, n_local, indices_petsc, rhs_vec%val(i_start+1:i_end), INSERT_VALUES, ierr))
    PetscCallA(VecAssemblyBegin(petsc_sys%b, ierr))
    PetscCallA(VecAssemblyEnd(petsc_sys%b, ierr))
    deallocate(indices_petsc)

  end subroutine petsc_update_rhs


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
    PetscReal :: petsc_norm
    PC :: pc ! Maybe should be part of petsc_sys in the future
    PetscViewerAndFormat :: vf
    KSPConvergedReason :: reason
    Mat :: A_aij, F
    Vec :: b_aij, x_aij
    KSPType :: ksp_type

    PetscCallA(PetscObjectGetComm(petsc_sys%A, comm, ierr))
    call MPI_COMM_RANK(comm, my_id, mpierr)

    !PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_INITIAL_MATRIX, A_aij, ierr))
    !PetscCallA(MatCreateVecs(A_aij, x_aij, b_aij, ierr))
    !PetscCallA(VecCopy(petsc_sys%b, b_aij, ierr))

    PetscCallA(KSPCreate(comm, petsc_sys%ksp, ierr))
    PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A, petsc_sys%A, ierr))

    PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf, ierr))
    PetscCallA(KSPMonitorSet(petsc_sys%ksp, KSPMonitorResidual, vf, PetscViewerAndFormatDestroy, ierr))

    ! PetscCallA(KSPSetType(petsc_sys%ksp, KSPDGMRES, ierr))
    PetscCallA(KSPSetType(petsc_sys%ksp, KSPPREONLY, ierr))
    !PetscCallA(KSPSetType(petsc_sys%ksp, KSPGMRES, ierr))

    ! Set the preconditioner
    PetscCallA(KSPGetPC(petsc_sys%ksp, pc, ierr))
    ! --- Additive Schwarz
    !PetscCallA(PCSetType(pc, PCASM, ierr)) ! Set additive Schwarz method
    !PetscCallA(PCASMSetTotalSubdomains(pc, 5, PETSC_NULL_IS, PETSC_NULL_IS, ierr))
    !PetscCallA(PCASMSetOverlap(pc, 2, ierr))
    !PetscCallA(PCASMSetType(pc, PC_ASM_BASIC, ierr)) ! Set type of restriction/interpolation
    ! --- AMG
    !PetscCallA(PCSetType(pc, PCGAMG, ierr))
    !PetscCallA(PCGAMGSetThreshold(pc, [0.1], 1, ierr))
    !PetscCallA(PCGAMGSetAggressiveLevels(pc, 1, ierr))
    ! --- LU
    PetscCallA(PCSetType(pc, PCLU, ierr))
    PetscCallA(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr))

    !PetscCallA(KSPSetFromOptions(petsc_sys%ksp, ierr))

    PetscCallA(KSPGetPC(petsc_sys%ksp, pc, ierr))

    PetscCallA(PCFactorSetMatOrderingType(pc,MATORDERINGND,ierr))
    PetscCallA(PCFactorGetMatrix(pc, F, ierr))
    PetscCallA(MatMumpsSetIcntl(F, 7,  7,  ierr))  ! fill-reducing ordering
    PetscCallA(MatMumpsSetIcntl(F, 14, 50, ierr))  ! workspace expansion %
    PetscCallA(MatMumpsSetIcntl(F, 8,  77, ierr))  ! numerical scaling (auto)
    PetscCallA(MatMumpsSetIcntl(F, 21, 1, ierr))

    PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))

    PetscCallA(KSPGetType(petsc_sys%ksp, ksp_type, ierr))
    if (my_id == 0) print *, "KSP type:", ksp_type

    ! Set the maximum iterations of the linear system
    PetscCallA(KSPSetTolerances(petsc_sys%ksp, PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, 400, ierr))
    PetscCallA(KSPGMRESSetRestart(petsc_sys%ksp, 40, ierr))

    if (my_id .eq. 0) print *, "Solving the system using PETSc"
    PetscCallA(KSPSolve(petsc_sys%ksp, petsc_sys%b, petsc_sys%x, ierr))
    PetscCallA(KSPDestroy(petsc_sys%ksp, ierr))
    !PetscCallA(MatDestroy(A_aij, ierr))
    !PetscCallA(VecCopy(x_aij, petsc_sys%x, ierr))
    !PetscCallA(VecDestroy(b_aij, ierr))
    !PetscCallA(VecDestroy(x_aij, ierr))

    ! Calculate the norm of the solution
    PetscCallA(VecNorm(petsc_sys%x, NORM_2, petsc_norm, ierr))
    if (my_id .eq.0) print *, "PETSc Norm (solution): ", petsc_norm
  end subroutine petsc_solve_and_retrieve


  !> Iterative solve with persistent KSP/PC across time steps.
  !! On first call (!ksp_ready): creates AIJ matrix, KSP, sets up PCFIELDSPLIT+MUMPS.
  !! When !solve_only: converts A to AIJ (reuse sparsity), calls KSPSetUp to refactorize.
  !! When solve_only:  converts A to AIJ, sets KSPSetReusePreconditioner to skip refactorization.
  subroutine petsc_solve_iterative_and_retrieve(petsc_sys, solve_only, n_iter, converged)
    use mod_clock, only: FMT_TIMING
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    logical, intent(in) :: solve_only
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged

    PetscErrorCode :: ierr
    integer :: comm, my_id, mpierr
    KSPConvergedReason :: reason
    PetscLogDouble :: t1, t2
    KSPType :: ksp_type
    PetscInt :: its
    PetscReal :: petsc_norm
    PetscViewerAndFormat :: vf

    call PetscObjectGetComm(petsc_sys%A, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)

    if (.not. petsc_sys%ksp_ready) then
      ! First solve: create AIJ matrix, vecs, KSP, and set up PCFIELDSPLIT+MUMPS
      PetscCallA(PetscLogStageRegister("KSP Setup", petsc_sys%stage_setup, ierr))
      PetscCallA(PetscLogStageRegister("KSP Solve", petsc_sys%stage_solve, ierr))
      PetscCallA(PetscLogStagePush(petsc_sys%stage_setup, ierr))

      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_INITIAL_MATRIX, petsc_sys%A_aij, ierr))
      PetscCallA(MatCreateVecs(petsc_sys%A_aij, petsc_sys%x_aij, petsc_sys%b_aij, ierr))

      PetscCallA(KSPCreate(comm, petsc_sys%ksp, ierr))
      PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A_aij, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetType(petsc_sys%ksp, KSPDGMRES, ierr))

      ! Set the maximum iterations and restart
      PetscCallA(KSPSetTolerances(petsc_sys%ksp, 1.d-8, 1.d-36, PETSC_CURRENT_REAL, 400, ierr))
      PetscCallA(KSPGMRESSetRestart(petsc_sys%ksp, 40, ierr))

      if (my_id .eq. 0) write(*,*) "[PETSc] setup: DGMRES + PCFIELDSPLIT + MUMPS"
      PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf, ierr))
      PetscCallA(KSPMonitorSet(petsc_sys%ksp, KSPMonitorResidual, vf, PetscViewerAndFormatDestroy, ierr))
      call petsc_setup_toroidal_harmonic_pc(petsc_sys%ksp, petsc_sys%A)

      PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))
      petsc_sys%ksp_ready = .true.

      PetscCallA(PetscLogStagePop(ierr))

    else if (.not. solve_only) then
      if (my_id .eq. 0) write(*,*) "[PETSc] PC rebuild: refactorizing"
      PetscCallA(PetscLogStagePush(petsc_sys%stage_setup, ierr))

      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_REUSE_MATRIX, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A_aij, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetReusePreconditioner(petsc_sys%ksp, PETSC_FALSE, ierr))
      PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))

      PetscCallA(PetscLogStagePop(ierr))

    else
      ! solve_only: update A for mat-vec products but reuse PC factorization
      if (my_id .eq. 0) write(*,*) "[PETSc] PC reuse: solve_only, skipping refactorization"
      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_REUSE_MATRIX, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A_aij, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetReusePreconditioner(petsc_sys%ksp, PETSC_TRUE, ierr))
    end if

    ! Copy RHS, solve, copy solution back
    PetscCallA(VecCopy(petsc_sys%b, petsc_sys%b_aij, ierr))

    PetscCallA(PetscTime(t1, ierr))
    PetscCallA(PetscLogStagePush(petsc_sys%stage_solve, ierr))
    PetscCallA(KSPSolve(petsc_sys%ksp, petsc_sys%b_aij, petsc_sys%x_aij, ierr))
    PetscCallA(PetscLogStagePop(ierr))
    PetscCallA(PetscTime(t2, ierr))

    PetscCallA(VecCopy(petsc_sys%x_aij, petsc_sys%x, ierr))

    PetscCallA(KSPGetConvergedReason(petsc_sys%ksp, reason, ierr))
    PetscCallA(KSPGetIterationNumber(petsc_sys%ksp, its, ierr))
    n_iter    = its
    converged = (reason > 0)

    if (my_id == 0) write(*,FMT_TIMING) my_id, '[PETSc] Elapsed time in solve :', t2-t1

    ! Calculate the norm of the solution
    PetscCallA(VecNorm(petsc_sys%x, NORM_2, petsc_norm, ierr))
    if (my_id .eq.0) write(*,'(A,ES12.4)') "[PETSc] solution norm: ", petsc_norm
  end subroutine petsc_solve_iterative_and_retrieve


  subroutine petsc_recover_solution(petsc_sys, sol_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(inout) :: sol_vec

    Vec             :: x_seq
    VecScatter      :: scatter
    PetscScalar, pointer :: x_arr(:)
    PetscErrorCode :: ierr

    PetscCallA(VecScatterCreateToAll(petsc_sys%x, scatter, x_seq, ierr))

    PetscCallA(VecScatterBegin(scatter, petsc_sys%x, x_seq, INSERT_VALUES, SCATTER_FORWARD, ierr))
    PetscCallA(VecScatterEnd(scatter, petsc_sys%x, x_seq, INSERT_VALUES, SCATTER_FORWARD, ierr))

    PetscCallA(VecGetArrayF90(x_seq, x_arr, ierr))

    sol_vec%val(:) = x_arr(:)

    PetscCallA(VecRestoreArrayF90(x_seq, x_arr, ierr))

    PetscCallA(VecScatterDestroy(scatter, ierr))
    PetscCallA(VecDestroy(x_seq, ierr))
  end subroutine petsc_recover_solution


  !> Destroy all persistent PETSc objects; safe to call even if never initialized.
  !! petsc_sys%A is only destroyed if owns_A=.true. (old path via petsc_init_system).
  !! In the direct assembly path, A is owned by a_mat%petsc_A.
  subroutine petsc_cleanup(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    PetscErrorCode :: ierr

    if (petsc_sys%ksp_ready) then
      call KSPDestroy(petsc_sys%ksp, ierr)
      call MatDestroy(petsc_sys%A_aij, ierr)
      call VecDestroy(petsc_sys%b_aij, ierr)
      call VecDestroy(petsc_sys%x_aij, ierr)
      petsc_sys%ksp_ready = .false.
    endif
    if (petsc_sys%initialized) then
      call VecDestroy(petsc_sys%b, ierr)
      call VecDestroy(petsc_sys%x, ierr)
      if (petsc_sys%owns_A) then
        call MatDestroy(petsc_sys%A, ierr)
        petsc_sys%owns_A = .false.
      endif
      petsc_sys%initialized = .false.
    endif
  end subroutine petsc_cleanup

#endif
end module mod_petsc
