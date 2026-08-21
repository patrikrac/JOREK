module mod_petsc
#ifdef USE_PETSC
  use mpi_mod
  use mod_petsc_pc
  use mod_petsc_direct_solver, only: petsc_configure_direct_solver
  use mod_petsc_dump,          only: petsc_dump_operator
#include "petsc/finclude/petsc.h"
  use petsc
#ifdef USE_SLEPC
#include "slepc/finclude/slepceps.h"
  use slepceps
#endif

  implicit none


  !> Run-directory file read by PetscInitialize for solver options.
  character(len=*), parameter :: PETSC_OPTIONS_FILE = 'jorek.petsc'
  !> Options prefix of the main iterative solve, e.g. -jorek_ksp_rtol 1e-9
  character(len=*), parameter :: PETSC_MAIN_PREFIX   = 'jorek_'
  !> Options prefix of the one-shot direct solve path
  character(len=*), parameter :: PETSC_DIRECT_PREFIX = 'jorek_direct_'

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

  !> Initialize PETSc, reading solver options from PETSC_OPTIONS_FILE in the run
  !! directory when it exists. That file is the supported way to reconfigure the
  !! solvers at run time; see namelist/jorek.petsc.example for the available
  !! prefixes. Passing a file name that does not exist is a fatal error in PETSc,
  !! hence the inquire.
  subroutine petsc_initialize()
    PetscErrorCode :: ierr
    logical :: have_opts
    ! PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-log_view", PETSC_NULL_CHARACTER, ierr))

    inquire(file=PETSC_OPTIONS_FILE, exist=have_opts)
#ifdef USE_SLEPC
    if (have_opts) then
      call SlepcInitialize(PETSC_OPTIONS_FILE, ierr)  ! superset of PetscInitialize
    else
      call SlepcInitialize(PETSC_NULL_CHARACTER, ierr)
    endif
    if (ierr /= 0) print *, "Error initializing SLEPc/PETSc"
#else
    if (have_opts) then
      call PetscInitialize(PETSC_OPTIONS_FILE, ierr)
    else
      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
    endif
    if (ierr /= 0) print *, "Error initializing PETSc"
#endif
  end subroutine


  !> Report the main solve configuration actually resolved from the defaults
  !! plus whatever jorek.petsc supplied.
  subroutine report_main_solver(ksp, my_id)
    KSP, intent(in)     :: ksp
    integer, intent(in) :: my_id

    KSPType        :: ktype
    PetscReal      :: rtol, abstol, dtol
    PetscInt       :: maxits
    PetscErrorCode :: ierr

    if (my_id /= 0) return

    PetscCallA(KSPGetType(ksp, ktype, ierr))
    PetscCallA(KSPGetTolerances(ksp, rtol, abstol, dtol, maxits, ierr))
    write(*,*) '[PETSc] main solve (-'//PETSC_MAIN_PREFIX//'...): '//trim(ktype) &
               //' + '//PCFIELDSPLIT
    write(*,*) '[PETSc]   rtol =', rtol, ' max_it =', maxits
  end subroutine report_main_solver


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
    integer :: my_id, mpierr
    character(len=256) :: version_string

    call PetscGetVersion(version_string, ierr)
    if (ierr == 0) then
      call MPI_COMM_RANK(MPI_COMM_WORLD, my_id, mpierr)
      if (my_id == 0) then
        print *, "----------------------------------------"
        print *, "JOREK linked with ", trim(version_string)
        print *, "----------------------------------------"
      end if
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
        val_ptr_end   = val_ptr_start + block_size2 - 1
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

    PetscInt :: i_start, i_end, n_local
    PetscScalar, pointer :: b_arr(:)
    PetscErrorCode :: ierr

    ! rhs_vec%val is the full global rhs on every rank, so the owned slice can be
    ! written straight into the Vec's local array - no index list, no assembly pass.
    PetscCallA(VecGetOwnershipRange(petsc_sys%b, i_start, i_end, ierr))
    n_local = i_end - i_start

    PetscCallA(VecGetArray(petsc_sys%b, b_arr, ierr))
    b_arr(1:n_local) = rhs_vec%val(i_start+1:i_end)
    PetscCallA(VecRestoreArray(petsc_sys%b, b_arr, ierr))

  end subroutine petsc_update_rhs


  !> Load an initial guess into petsc_sys%x from the JOREK sol_vec (the previous
  !! time-step increment). Mirrors petsc_update_rhs ownership slicing so that x
  !! and b share the same global layout. If sol_vec is not yet allocated (e.g. the
  !! first step of a fresh, non-restart run) the guess degrades to zero, reproducing
  !! the standard zero-start. Used together with KSPSetInitialGuessNonzero.
  subroutine petsc_update_initial_guess(petsc_sys, sol_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(in) :: sol_vec

    PetscInt :: i_start, i_end, n_local
    PetscScalar, pointer :: x_arr(:)
    PetscErrorCode :: ierr

    if (.not. associated(sol_vec%val)) then
      PetscCallA(VecSet(petsc_sys%x, 0.0d0, ierr))
      return
    endif

    ! Same ownership slicing as petsc_update_rhs, written directly into the local array.
    PetscCallA(VecGetOwnershipRange(petsc_sys%x, i_start, i_end, ierr))
    n_local = i_end - i_start

    PetscCallA(VecGetArray(petsc_sys%x, x_arr, ierr))
    x_arr(1:n_local) = sol_vec%val(i_start+1:i_end)
    PetscCallA(VecRestoreArray(petsc_sys%x, x_arr, ierr))

  end subroutine petsc_update_initial_guess


  subroutine petsc_solve_and_retrieve(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys

    PetscErrorCode :: ierr
    integer :: comm, my_id, mpierr
    PetscLogDouble :: t1, t2
    PetscReal :: petsc_norm
    PC :: pc ! Maybe should be part of petsc_sys in the future
    PetscViewerAndFormat :: vf
    KSPConvergedReason :: reason
    KSPType :: ksp_type

    PetscCallA(PetscObjectGetComm(petsc_sys%A, comm, ierr))
    call MPI_COMM_RANK(comm, my_id, mpierr)

    PetscCallA(KSPCreate(comm, petsc_sys%ksp, ierr))
    PetscCallA(KSPSetOptionsPrefix(petsc_sys%ksp, PETSC_DIRECT_PREFIX, ierr))
    PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A, petsc_sys%A, ierr))

    PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf, ierr))
    PetscCallA(KSPMonitorSet(petsc_sys%ksp, KSPMonitorResidual, vf, PetscViewerAndFormatDestroy, ierr))

    PetscCallA(KSPSetType(petsc_sys%ksp, KSPPREONLY, ierr))
    PetscCallA(KSPSetFromOptions(petsc_sys%ksp, ierr))

    ! Set the preconditioner: direct solve, LU via MUMPS unless overridden by
    ! -jorek_direct_pc_* options.
    PetscCallA(KSPGetPC(petsc_sys%ksp, pc, ierr))
    call petsc_configure_direct_solver(pc)

    PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))
    petsc_sys%ksp_ready = .true.

    PetscCallA(KSPGetType(petsc_sys%ksp, ksp_type, ierr))
    if (my_id == 0) print *, "KSP type:", ksp_type

    if (my_id .eq. 0) print *, "Solving the system using PETSc"
    PetscCallA(KSPSolve(petsc_sys%ksp, petsc_sys%b, petsc_sys%x, ierr))
    PetscCallA(KSPDestroy(petsc_sys%ksp, ierr))
    petsc_sys%ksp_ready = .false.

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
    PetscLogDouble :: ts1, ts2
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
      PetscCallA(PetscTime(ts1, ierr))

      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_INITIAL_MATRIX, petsc_sys%A_aij, ierr))
      PetscCallA(MatCreateVecs(petsc_sys%A_aij, petsc_sys%x_aij, petsc_sys%b_aij, ierr))
      ! Opt-in, inert unless -jorek_dump_mat is set.
      call petsc_dump_operator(petsc_sys%A_aij, 'system')

      PetscCallA(KSPCreate(comm, petsc_sys%ksp, ierr))
      PetscCallA(KSPSetOptionsPrefix(petsc_sys%ksp, PETSC_MAIN_PREFIX, ierr))
      PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A_aij, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetType(petsc_sys%ksp, KSPGMRES, ierr))

      ! Warm-start: consume the JOREK initial guess (previous-step increment) loaded into
      ! petsc_sys%x before each KSPSolve. The convergence reference stays the rhs norm
      ! ||b|| (PETSc default; KSPConvergedDefaultSetUIRNorm is NOT set), so the stopping
      ! criterion is identical to the zero-start behaviour - only the starting point changes.
      PetscCallA(KSPSetInitialGuessNonzero(petsc_sys%ksp, PETSC_TRUE, ierr))

      ! Set GMRES parameters
      PetscCallA(KSPSetTolerances(petsc_sys%ksp, 1.d-8, 1.d-36, PETSC_CURRENT_REAL, 400, ierr))
      PetscCallA(KSPGMRESSetRestart(petsc_sys%ksp, 40, ierr))
      PetscCallA(KSPGMRESSetOrthogonalization(petsc_sys%ksp, KSPGMRESClassicalGramSchmidtOrthogonalization, ierr))
      PetscCallA(KSPGMRESSetCGSRefinementType(petsc_sys%ksp, KSP_GMRES_CGS_REFINE_IFNEEDED, ierr))

      PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf, ierr))
      PetscCallA(KSPMonitorSet(petsc_sys%ksp, KSPMonitorResidual, vf, PetscViewerAndFormatDestroy, ierr))

      ! Options after the defaults above, so anything in jorek.petsc overrides
      ! them: -jorek_ksp_rtol, -jorek_ksp_max_it, -jorek_ksp_gmres_restart, ...
      ! The outer PC type is deliberately NOT a knob - PCFIELDSPLIT carries the
      ! toroidal mode-family decomposition and is structural, so petsc_setup_pc
      ! below always sets it. The blocks inside it are fully configurable under
      ! the -jorek_pcblock_ prefix, and report themselves from there.
      PetscCallA(KSPSetFromOptions(petsc_sys%ksp, ierr))
      call report_main_solver(petsc_sys%ksp, my_id)
      call petsc_setup_pc(petsc_sys%ksp, petsc_sys%A, PETSC_PC_TOROIDAL_HARMONIC)

      PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))
      ! Force the sub-KSP factorizations here so they are timed as setup, not solve
      ! (no-op for the toroidal PC, which sets its sub-KSPs up explicitly).
      PetscCallA(KSPSetUpOnBlocks(petsc_sys%ksp, ierr))
      petsc_sys%ksp_ready = .true.

      PetscCallA(PetscTime(ts2, ierr))
      PetscCallA(PetscLogStagePop(ierr))
      if (my_id == 0) write(*,FMT_TIMING) my_id, '[PETSc] Elapsed time in solver setup :', ts2-ts1

    else if (.not. solve_only) then
      if (my_id .eq. 0) write(*,*) "[PETSc] PC rebuild: refactorizing"
      PetscCallA(PetscLogStagePush(petsc_sys%stage_setup, ierr))
      PetscCallA(PetscTime(ts1, ierr))

      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_REUSE_MATRIX, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A_aij, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetReusePreconditioner(petsc_sys%ksp, PETSC_FALSE, ierr))
      PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))
      ! KSPSetUp only rebuilds PCFIELDSPLIT itself; the sub-KSP LU/MUMPS numerical
      ! factorizations are otherwise deferred to KSPSetUpOnBlocks inside KSPSolve, which
      ! would charge the whole factorization cost to the GMRES solve timer below.
      PetscCallA(KSPSetUpOnBlocks(petsc_sys%ksp, ierr))

      PetscCallA(PetscTime(ts2, ierr))
      PetscCallA(PetscLogStagePop(ierr))
      if (my_id == 0) write(*,FMT_TIMING) my_id, '[PETSc] Elapsed time in solver setup :', ts2-ts1

    else
      ! solve_only: update A for mat-vec products but reuse PC factorization
      if (my_id .eq. 0) write(*,*) "[PETSc] PC reuse: solve_only, skipping refactorization"
      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_REUSE_MATRIX, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A_aij, petsc_sys%A_aij, ierr))
      PetscCallA(KSPSetReusePreconditioner(petsc_sys%ksp, PETSC_TRUE, ierr))
    end if

    ! Copy RHS and warm-start guess, solve, copy solution back
    PetscCallA(VecCopy(petsc_sys%b, petsc_sys%b_aij, ierr))
    PetscCallA(VecCopy(petsc_sys%x, petsc_sys%x_aij, ierr))   ! initial guess for GMRES

    PetscCallA(PetscTime(t1, ierr))
    PetscCallA(PetscLogStagePush(petsc_sys%stage_solve, ierr))
    PetscCallA(KSPSolve(petsc_sys%ksp, petsc_sys%b_aij, petsc_sys%x_aij, ierr))
    PetscCallA(PetscLogStagePop(ierr))
    PetscCallA(PetscTime(t2, ierr))

    PetscCallA(VecCopy(petsc_sys%x_aij, petsc_sys%x, ierr))

    PetscCallA(KSPGetConvergedReason(petsc_sys%ksp, reason, ierr))
    PetscCallA(KSPGetIterationNumber(petsc_sys%ksp, its, ierr))
    n_iter    = its
    converged = (reason%v > 0)

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

    PetscCallA(VecGetArray(x_seq, x_arr, ierr))

    if (associated(sol_vec%val)) then
      sol_vec%val(1:sol_vec%n) = x_arr(1:sol_vec%n)
    end if

    PetscCallA(VecRestoreArray(x_seq, x_arr, ierr))

    PetscCallA(VecScatterDestroy(scatter, ierr))
    PetscCallA(VecDestroy(x_seq, ierr))
  end subroutine petsc_recover_solution


  !> Assemble a scalar MATAIJ equilibrium matrix directly from JOREK COO
  !! triplets (a_mat%irn/jcn/val, 1-based, with duplicate entries).
  !! ADD_VALUES sums duplicates natively. Serial (n_cpu==1, COMM_SELF) preallocates
  !! from COO row counts; parallel (n_cpu>1, COMM_WORLD) lets PETSc choose the row
  !! distribution and scatters rank-0-set entries during MatAssembly.
  subroutine petsc_equilibrium_assemble(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(in)       :: a_mat

    integer :: k, r, c, j, owner
    integer :: comm, my_id, n_cpu, mpierr
    PetscInt :: n_global, n_local
    PetscInt :: row(1), col(1)
    PetscScalar :: v(1)
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    integer, allocatable  :: d_all(:), o_all(:), d_loc(:), o_loc(:), counts(:), displs(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)
    call MPI_COMM_SIZE(comm, n_cpu, mpierr)

    n_global = a_mat%ng

    call MatCreate(comm, petsc_sys%A, ierr)

    if (n_cpu .eq. 1) then
      n_local = a_mat%ng
      call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
      call MatSetType(petsc_sys%A, MATAIJ, ierr)
      ! Per-row count from COO (duplicate-overestimate, safe); rank owns all rows.
      allocate(d_nnz(n_global))
      d_nnz = 0
      do k = 1, a_mat%nnz
        r = a_mat%irn(k)
        d_nnz(r) = d_nnz(r) + 1
      enddo
      call MatSeqAIJSetPreallocation(petsc_sys%A, 0, d_nnz, ierr)
      deallocate(d_nnz)
    else
      ! Fix the row split up front so rank 0, which holds the whole COO, can work out
      ! every rank's ownership range and count its rows exactly. The previous
      ! n_global-per-row preallocation was effectively dense and grew as O(n^2).
      n_local = PETSC_DECIDE
      call PetscSplitOwnership(comm, n_local, n_global, ierr)
      call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
      call MatSetType(petsc_sys%A, MATAIJ, ierr)

      allocate(counts(n_cpu), displs(n_cpu))
      call MPI_Allgather(int(n_local), 1, MPI_INTEGER, counts, 1, MPI_INTEGER, comm, mpierr)
      displs(1) = 0
      do k = 2, n_cpu
        displs(k) = displs(k-1) + counts(k-1)
      enddo

      ! Per global row, count entries inside vs outside the diagonal block of the rank
      ! owning that row (duplicates included - a safe overestimate).
      if (my_id .eq. 0) then
        allocate(d_all(n_global), o_all(n_global))
        d_all = 0
        o_all = 0
        do k = 1, a_mat%nnz
          r = a_mat%irn(k)
          c = a_mat%jcn(k)
          owner = 1
          do j = n_cpu, 1, -1
            if (r-1 >= displs(j)) then
              owner = j
              exit
            endif
          enddo
          if (c-1 >= displs(owner) .and. c-1 < displs(owner) + counts(owner)) then
            d_all(r) = d_all(r) + 1
          else
            o_all(r) = o_all(r) + 1
          endif
        enddo
      else
        allocate(d_all(1), o_all(1))
      endif

      allocate(d_loc(n_local), o_loc(n_local))
      call MPI_Scatterv(d_all, counts, displs, MPI_INTEGER, d_loc, int(n_local), MPI_INTEGER, 0, comm, mpierr)
      call MPI_Scatterv(o_all, counts, displs, MPI_INTEGER, o_loc, int(n_local), MPI_INTEGER, 0, comm, mpierr)
      deallocate(d_all, o_all)

      ! Clamp to the width actually available in each block; PETSc rejects larger values.
      allocate(d_nnz(n_local), o_nnz(n_local))
      do k = 1, n_local
        d_nnz(k) = min(d_loc(k), int(n_local))
        o_nnz(k) = min(o_loc(k), int(n_global - n_local))
      enddo
      deallocate(d_loc, o_loc, counts, displs)

      call MatMPIAIJSetPreallocation(petsc_sys%A, 0, d_nnz, 0, o_nnz, ierr)
      deallocate(d_nnz, o_nnz)
    endif

    call MatSetOption(petsc_sys%A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

    ! Only rank 0 holds the COO; off-rank entries are cached and shipped at assembly.
    if (my_id .eq. 0) then
      do k = 1, a_mat%nnz
        row(1) = a_mat%irn(k) - 1
        col(1) = a_mat%jcn(k) - 1
        v(1)   = a_mat%val(k)
        call MatSetValues(petsc_sys%A, 1, row, 1, col, v, ADD_VALUES, ierr)
      enddo
    endif

    call MatAssemblyBegin(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)

    call MatCreateVecs(petsc_sys%A, petsc_sys%x, petsc_sys%b, ierr)

    petsc_sys%initialized = .true.
    petsc_sys%owns_A       = .true.
    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0)') &
      "[PETSc] equilibrium: AIJ matrix ", n_global, "x", n_global, " on ranks=", n_cpu
  end subroutine petsc_equilibrium_assemble


  !> Fill the equilibrium RHS: rank 0 sets all global entries, assembly scatters.
  !! Works for serial (COMM_SELF) and distributed (COMM_WORLD) equilibrium solves.
  subroutine petsc_equilibrium_rhs(petsc_sys, rhs_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(in)             :: rhs_vec

    integer :: i, comm, my_id, mpierr
    PetscInt :: ng
    PetscInt, allocatable :: idx(:)
    PetscErrorCode :: ierr

    call PetscObjectGetComm(petsc_sys%b, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)

    call VecSet(petsc_sys%b, 0.0d0, ierr)

    if (my_id .eq. 0) then
      ng = rhs_vec%n
      allocate(idx(ng))
      do i = 1, ng
        idx(i) = i - 1
      enddo
      call VecSetValues(petsc_sys%b, ng, idx, rhs_vec%val(1:ng), INSERT_VALUES, ierr)
      deallocate(idx)
    endif

    call VecAssemblyBegin(petsc_sys%b, ierr)
    call VecAssemblyEnd(petsc_sys%b, ierr)
  end subroutine petsc_equilibrium_rhs


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
