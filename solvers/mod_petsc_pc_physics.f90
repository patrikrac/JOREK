module mod_petsc_pc_physics
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_setup_physics_pc, petsc_create_pc_matrices, &
            petsc_assemble_pc_matrices, petsc_update_physics_pc_ctx, &
            petsc_analyze_pc_matrices, petsc_test_pc_matrices, &
            petsc_physics_pc_build_reduced

  type :: type_physics_pc_ctx
    logical :: initialized    = .false.
    logical :: matrices_ready = .false.
    logical :: reduced_ready  = .false.
    integer :: comm           = -1
    !> 1-var BAIJ matrices from separate element-level assembly
    Mat :: A_j, A_w, A_jpsi, A_wu

    !> Index sets for each variable in the full system vector (1=psi..6=T)
    IS :: is_var(6)
    logical :: is_created = .false.

    !> Lumped mass inverse vectors (1-var layout from extracted sub-blocks)
    Vec :: diag_Mj_inv, diag_Mw_inv

    !> Block diagonal inverse matrices (BAIJ, block_size = n_tor)
    !! Computed via MatInvertBlockDiagonalMat from B_33/B_44
    Mat :: Dinv_Mj, Dinv_Mw
    logical :: dinv_created = .false.

    !> Sub-blocks extracted from the full system AIJ matrix.
    !! Naming: B_ij = block at (equation i, variable j).
    !! Coupling blocks for Schur corrections (TO j,w from other eqs):
    Mat :: B_13, B_23, B_24, B_63
    !! Constraint blocks (FROM j,w eqs):
    Mat :: B_31, B_42
    !! Diagonal blocks of the constraint equations:
    Mat :: B_33, B_44
    !! Diagonal blocks of the 4x4 reduced system:
    Mat :: B_11, B_22, B_55, B_66
    !! Off-diagonal blocks of the 4x4 reduced system:
    Mat :: B_12, B_16                  ! row psi
    Mat :: B_21, B_25, B_26            ! row u
    Mat :: B_51, B_52                  ! row rho
    Mat :: B_61, B_62                  ! row T

    !> Reassembled diagonal blocks (simplified integrands, used when physics_pc_reassemble = .true.)
    Mat :: R_11, R_22, R_55, R_66
    logical :: reassembled_ready = .false.

    !> Schur-corrected blocks:
    !! Diagonal:
    !! Atilde_11 = B_11 - B_13 * D_j^{-1} * B_31
    !! Atilde_22 = B_22 - B_24 * D_w^{-1} * B_42
    Mat :: Atilde_11, Atilde_22
    !! Off-diagonal (psi-column and u-column get corrections):
    !! Atilde_21 = B_21 - B_23 * D_j^{-1} * B_31
    !! Atilde_61 = B_61 - B_63 * D_j^{-1} * B_31
    Mat :: Atilde_21, Atilde_61

    !> KSP for each diagonal block of the reduced 4x4 system
    KSP :: ksp_psi, ksp_u, ksp_rho, ksp_T
    logical :: ksp_created = .false.

    !> KSP for elliptic constraint mass matrices (replaces approx mass inverse)
    KSP :: ksp_Mj, ksp_Mw
    logical :: ksp_elliptic_created = .false.

    !> Work vectors (1-var size) — allocated once, reused in every apply
    Vec :: work_1, work_2, work_3, work_4, work_5

    !> Monolithic 4x4 reduced system (stage-one test mode)
    Mat :: A_reduced_4x4
    KSP :: ksp_reduced
    logical :: ksp_reduced_created = .false.
    Vec :: work_rhs_4v, work_sol_4v
    logical :: work_4v_created = .false.
    IS :: is_reduced(4)
    logical :: is_reduced_created = .false.
  end type type_physics_pc_ctx

  type(type_physics_pc_ctx), save :: g_ctx

contains

  !--------------------------------------------------------------------
  !> Register the physics-based PCSHELL on an existing KSP.
  !--------------------------------------------------------------------
  subroutine petsc_setup_physics_pc(ksp, A)
    KSP, intent(inout) :: ksp
    Mat, intent(in)    :: A
    PC :: pc
    PetscErrorCode :: ierr

    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(pc, physics_pc_apply, ierr))
    g_ctx%initialized = .true.
  end subroutine petsc_setup_physics_pc


  !--------------------------------------------------------------------
  !> Create variable index sets for extracting sub-vectors from the
  !! full 6-variable system vector.
  !!
  !! DOF ordering within each BAIJ block (block_size = n_var*n_tor):
  !!   var v (0-based) at node block i: i*block_size + v*n_tor + m
  !!   for m = 0..n_tor-1
  !--------------------------------------------------------------------
  subroutine create_variable_index_sets(A_full, comm)
    use mod_parameters, only: n_var, n_tor

    Mat, intent(in) :: A_full
    integer, intent(in) :: comm

    PetscInt :: n_local, n_global, rstart, rend
    PetscInt :: block_size, n_block_local, n_var_dofs, n_block_global
    !PetscInt :: out_local, out_global, out_start, out_end
    PetscInt, allocatable :: indices(:)
    PetscErrorCode :: ierr
    integer :: v, i, m, k

    ! Get parallel layout from the full system matrix
    PetscCallA(MatGetLocalSize(A_full, n_local, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetSize(A_full, n_global, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetOwnershipRange(A_full, rstart, rend, ierr))

    !write(*,'(A,I8,A,I8)') "[Physics PC]   Creating variable index sets: local DOFs ", n_local, " [", rstart, "-", rend-1, "]"

    block_size    = n_var * n_tor
    n_block_local = n_local / block_size
    n_block_global = n_global / block_size
    n_var_dofs    = n_block_local * n_tor

    allocate(indices(n_var_dofs))

    do v = 1, 6
      k = 0
      do i = 0, n_block_local - 1
        do m = 0, n_tor - 1
          k = k + 1
          !k = i*(n_tor-1) + m + 1
          indices(k) = rstart + i * block_size + (v-1) * n_tor + m
        enddo
      enddo
      PetscCallA(ISCreateGeneral(comm, n_var_dofs, indices, PETSC_COPY_VALUES, &
                           g_ctx%is_var(v), ierr))

      ! Print information about the created IS for debugging
      !PetscCallA(ISGetSize(g_ctx%is_var(v), out_global, ierr))
      !PetscCallA(ISGetLocalSize(g_ctx%is_var(v), out_local, ierr))
      !PetscCallA(ISGetMinMax(g_ctx%is_var(v), out_start, out_end, ierr))
      !write(*,*) "[Physics PC]     Variable ", v, ": local DOFs : ", out_local," global DOFs ", out_global, " [", out_start, "-", out_end, "]"
    enddo

    deallocate(indices)
    g_ctx%is_created = .true.
  end subroutine create_variable_index_sets


  !--------------------------------------------------------------------
  !> Extract a sub-block A_ij from the full system matrix.
  !! A_ij has rows corresponding to equation eq_row and columns
  !! corresponding to variable var_col.
  !--------------------------------------------------------------------
  subroutine extract_sub_block(A_full, eq_row, var_col, B, first_time)
    Mat, intent(in)    :: A_full
    integer, intent(in) :: eq_row, var_col
    Mat, intent(inout)  :: B
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (first_time) then
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), &
                              MAT_INITIAL_MATRIX, B, ierr))
    else
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), &
                              MAT_REUSE_MATRIX, B, ierr))
    endif
  end subroutine extract_sub_block


  !--------------------------------------------------------------------
  !> Compute diagnonal mass inverse from diagonal blocks B_33 and B_44.
  !! diag_M_inv = 1 / diag(B)
  !--------------------------------------------------------------------
  subroutine compute_diag_mass_inverse(B, diag_M_inv, first_time)
    Mat, intent(in)    :: B
    Vec, intent(inout) :: diag_M_inv
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (first_time) then
      PetscCallA(MatCreateVecs(B, PETSC_NULL_VEC, diag_M_inv, ierr))
    endif
    !PetscCallA(MatGetRowSum(B, diag_M_inv, ierr))
    PetscCallA(MatGetDiagonal(B, diag_M_inv, ierr))
    PetscCallA(VecReciprocal(diag_M_inv, ierr))
  end subroutine compute_diag_mass_inverse


  !--------------------------------------------------------------------
  !> Compute block diagonal inverse of a 1-var AIJ matrix.
  !!
  !! The extracted sub-blocks (B_33, B_44) are AIJ, but
  !! MatInvertBlockDiagonalMat requires BAIJ. This routine:
  !!   1. Converts AIJ -> BAIJ (block_size = n_tor) as a temporary
  !!   2. Calls MatInvertBlockDiagonalMat to get the inverse
  !!   3. Destroys the temporary BAIJ copy
  !--------------------------------------------------------------------
  subroutine compute_block_diagonal_inverse(B_aij, Dinv, block_size, first_time)
    Mat, intent(in)    :: B_aij
    Mat, intent(inout) :: Dinv
    PetscInt, intent(in) :: block_size
    logical, intent(in)  :: first_time

    Mat :: B_baij, Dinv_baij
    PetscErrorCode :: ierr
    integer :: comm

    ! Destroy previous inverse if rebuilding
    if (.not. first_time) call MatDestroy(Dinv, ierr)

    ! Set block size on AIJ so that MatConvert creates BAIJ with correct blocking
    call MatSetBlockSize(B_aij, block_size, ierr)

    ! Convert AIJ -> BAIJ with the given block size
    call MatConvert(B_aij, MATBAIJ, MAT_INITIAL_MATRIX, B_baij, ierr)

    ! Pre-create output matrix (required by MatInvertBlockDiagonalMat)
    call PetscObjectGetComm(B_baij, comm, ierr)
    call MatCreate(comm, Dinv_baij, ierr)

    ! Compute the inverse of the block diagonal (result is BAIJ)
    call MatInvertBlockDiagonalMat(B_baij, Dinv_baij, ierr)

    ! Convert result to AIJ for compatibility with MatMatMult(AIJ, AIJ)
    call MatConvert(Dinv_baij, MATAIJ, MAT_INITIAL_MATRIX, Dinv, ierr)

    ! Clean up temporaries
    call MatDestroy(Dinv_baij, ierr)
    call MatDestroy(B_baij, ierr)
  end subroutine compute_block_diagonal_inverse


  !--------------------------------------------------------------------
  !> Compute a Schur-corrected diagonal block using diagonal mass inverse:
  !! Atilde = B_diag - B_coupling * diag(M_inv) * B_constraint
  !!
  !! Steps:
  !!   1. B_scaled = diag(M_inv) * B_constraint  (scale rows)
  !!   2. C = B_coupling * B_scaled               (mat-mat product)
  !!   3. Atilde = B_diag - C                     (subtract)
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block_diag(B_diag, B_coupling, B_constraint, &
                                            diag_M_inv, Atilde, first_time)
    Mat, intent(in)    :: B_diag, B_coupling, B_constraint
    Vec, intent(in)    :: diag_M_inv
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    Mat :: B_scaled, C
    PetscErrorCode :: ierr

    ! B_scaled = diag(M_inv) * B_constraint  (left-scale rows)
    call MatDuplicate(B_constraint, MAT_COPY_VALUES, B_scaled, ierr)
    call MatDiagonalScale(B_scaled, diag_M_inv, PETSC_NULL_VEC, ierr)

    ! C = B_coupling * B_scaled
    call MatMatMult(B_coupling, B_scaled, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, C, ierr)

    ! Atilde = B_diag - C
    ! Always destroy and recreate: sparsity pattern may change between rebuilds
    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)
    call MatAXPY(Atilde, -1.0d0, C, DIFFERENT_NONZERO_PATTERN, ierr)

    call MatDestroy(B_scaled, ierr)
    call MatDestroy(C, ierr)
  end subroutine compute_schur_corrected_block_diag


  !--------------------------------------------------------------------
  !> Compute a Schur-corrected diagonal block using block diagonal inverse:
  !! Atilde = B_diag - B_coupling * Dinv * B_constraint
  !!
  !! Steps:
  !!   1. B_scaled = Dinv * B_constraint            (mat-mat product)
  !!   2. C = B_coupling * B_scaled                 (mat-mat product)
  !!   3. Atilde = B_diag - C                       (subtract)
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block(B_diag, B_coupling, B_constraint, &
                                            Dinv, Atilde, first_time)
    Mat, intent(in)    :: B_diag, B_coupling, B_constraint
    Mat, intent(in)    :: Dinv
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    Mat :: B_scaled, C
    PetscErrorCode :: ierr

    ! B_scaled = Dinv * B_constraint  (block-diagonal mat-mat product)
    call MatMatMult(Dinv, B_constraint, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, B_scaled, ierr)

    ! C = B_coupling * B_scaled
    call MatMatMult(B_coupling, B_scaled, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, C, ierr)

    ! Atilde = B_diag - C
    ! Always destroy and recreate: sparsity pattern may change between rebuilds
    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)
    call MatAXPY(Atilde, -1.0d0, C, DIFFERENT_NONZERO_PATTERN, ierr)

    call MatDestroy(B_scaled, ierr)
    call MatDestroy(C, ierr)
  end subroutine compute_schur_corrected_block


  !--------------------------------------------------------------------
  !> Compute a Schur-corrected block using exact M^{-1} via MatMatSolve:
  !! Atilde = B_diag - B_coupling * M^{-1} * B_constraint
  !!
  !! Uses the already-factored KSP (MUMPS) to solve M * X = B_constraint
  !! for X, then forms C = B_coupling * X.
  !!
  !! NOTE: This converts B_constraint to dense for MatMatSolve, so it is
  !! only suitable for proof-of-concept / small test cases.
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block_exact(B_diag, B_coupling, &
                                          B_constraint, ksp_M, Atilde, first_time)
    Mat, intent(in)    :: B_diag, B_coupling, B_constraint
    KSP, intent(in)    :: ksp_M
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    Mat :: F              ! factor matrix from KSP
    Mat :: B_dense        ! dense copy of B_constraint
    Mat :: X_dense        ! dense solution: M^{-1} * B_constraint
    Mat :: X_aij          ! sparse conversion of X_dense
    Mat :: C              ! B_coupling * X
    PC  :: pc_obj
    PetscErrorCode :: ierr

    ! Get factor matrix from already-factored KSP
    call KSPGetPC(ksp_M, pc_obj, ierr)
    call PCFactorGetMatrix(pc_obj, F, ierr)

    ! Convert B_constraint to dense for MatMatSolve
    call MatConvert(B_constraint, MATDENSE, MAT_INITIAL_MATRIX, B_dense, ierr)

    ! Create dense solution matrix of same size
    call MatDuplicate(B_dense, MAT_DO_NOT_COPY_VALUES, X_dense, ierr)

    ! Solve M * X = B_constraint  (uses MUMPS factorization)
    call MatMatSolve(F, B_dense, X_dense, ierr)

    ! Convert X back to sparse AIJ for MatMatMult
    call MatConvert(X_dense, MATAIJ, MAT_INITIAL_MATRIX, X_aij, ierr)

    ! C = B_coupling * X
    call MatMatMult(B_coupling, X_aij, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, C, ierr)

    ! Atilde = B_diag - C
    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)
    call MatAXPY(Atilde, -1.0d0, C, DIFFERENT_NONZERO_PATTERN, ierr)

    ! Clean up
    call MatDestroy(B_dense, ierr)
    call MatDestroy(X_dense, ierr)
    call MatDestroy(X_aij, ierr)
    call MatDestroy(C, ierr)
  end subroutine compute_schur_corrected_block_exact


  !--------------------------------------------------------------------
  !> Set up a sub-KSP for a diagonal block: PREONLY + LU + MUMPS.
  !--------------------------------------------------------------------
  subroutine setup_block_ksp(ksp_block, B_block, comm, first_time)
    KSP, intent(inout) :: ksp_block
    Mat, intent(in)    :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time

    PC :: pc
    PetscErrorCode :: ierr

    if (first_time) then
      call KSPCreate(comm, ksp_block, ierr)
    endif
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    call KSPSetType(ksp_block, KSPPREONLY, ierr)
    call KSPGetPC(ksp_block, pc, ierr)
    call PCSetType(pc, PCLU, ierr)
    call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_block, ierr)
  end subroutine setup_block_ksp


  !--------------------------------------------------------------------
  !> Assemble the monolithic 4x4 reduced system via MatCreateNest +
  !! MatConvert, and set up a single KSP (PREONLY+LU+MUMPS).
  !--------------------------------------------------------------------
  subroutine assemble_monolithic_4x4(use_reassembled, comm, first_time, my_id)
    logical, intent(in) :: use_reassembled, first_time
    integer, intent(in) :: comm, my_id

    Mat :: mats_nest(16), A_nest   ! 1D row-major: (row0,col0), (row0,col1), ...
    Mat :: diag_55, diag_66
    PC  :: pc_obj
    PetscErrorCode :: ierr
    PetscInt :: rstart, rend, n_local, n_global
    PetscInt, parameter :: nblocks = 4
    integer :: k

    ! Choose diagonal blocks B_55/B_66 or R_55/R_66
    if (use_reassembled) then
      diag_55 = g_ctx%R_55
      diag_66 = g_ctx%R_66
    else
      diag_55 = g_ctx%B_55
      diag_66 = g_ctx%B_66
    endif

    ! Populate nest in row-major order (PETSc Fortran convention for MatCreateNest)
    ! Row 1 (psi): Atilde_11  B_12        0          B_16
    mats_nest( 1) = g_ctx%Atilde_11
    mats_nest( 2) = g_ctx%B_12
    mats_nest( 3) = PETSC_NULL_MAT
    mats_nest( 4) = g_ctx%B_16
    ! Row 2 (u):   Atilde_21  Atilde_22   B_25       B_26
    mats_nest( 5) = g_ctx%Atilde_21
    mats_nest( 6) = g_ctx%Atilde_22
    mats_nest( 7) = g_ctx%B_25
    mats_nest( 8) = g_ctx%B_26
    ! Row 3 (rho): B_51       B_52        B_55/R_55  0
    mats_nest( 9) = g_ctx%B_51
    mats_nest(10) = g_ctx%B_52
    mats_nest(11) = diag_55
    mats_nest(12) = PETSC_NULL_MAT
    ! Row 4 (T):   Atilde_61  B_62        0          B_66/R_66
    mats_nest(13) = g_ctx%Atilde_61
    mats_nest(14) = g_ctx%B_62
    mats_nest(15) = PETSC_NULL_MAT
    mats_nest(16) = diag_66

    PetscCallA(MatCreateNest(comm, nblocks, PETSC_NULL_IS, nblocks, PETSC_NULL_IS, mats_nest, A_nest, ierr))

    ! Destroy old monolithic AIJ if rebuilding
    if (.not. first_time .and. g_ctx%ksp_reduced_created) then
      call MatDestroy(g_ctx%A_reduced_4x4, ierr)
    endif

    ! Convert nest to concrete AIJ
    call MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%A_reduced_4x4, ierr)
    call MatDestroy(A_nest, ierr)

    ! Set up monolithic KSP
    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_reduced, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_reduced, g_ctx%A_reduced_4x4, g_ctx%A_reduced_4x4, ierr)
    call KSPSetType(g_ctx%ksp_reduced, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_reduced, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_reduced, ierr)
    g_ctx%ksp_reduced_created = .true.

    ! Allocate 4-var work vectors and create index sets (first time only)
    if (.not. g_ctx%work_4v_created) then
      call MatCreateVecs(g_ctx%A_reduced_4x4, g_ctx%work_sol_4v, g_ctx%work_rhs_4v, ierr)
      g_ctx%work_4v_created = .true.

      ! Create IS for each variable in the monolithic vector
      ! Each sub-block has identical parallel layout, so variable k
      ! occupies rows [rstart + k*n_local .. rstart + (k+1)*n_local - 1]
      call VecGetOwnershipRange(g_ctx%work_rhs_4v, rstart, rend, ierr)
      n_local = (rend - rstart) / 4
      call VecGetSize(g_ctx%work_rhs_4v, n_global, ierr)
      do k = 1, 4
        call ISCreateStride(comm, n_local, rstart + (k-1)*n_local, 1, &
                            g_ctx%is_reduced(k), ierr)
      enddo
      g_ctx%is_reduced_created = .true.
    endif

    if (my_id == 0) write(*,'(A)') "[Physics PC]   Monolithic 4x4 KSP set up (PREONLY+LU+MUMPS)"
  end subroutine assemble_monolithic_4x4


  !--------------------------------------------------------------------
  !> Build the reduced 4x4 system from the full system matrix.
  !!
  !! Extracts sub-blocks, approx mass, forms Schur corrections,
  !! and sets up sub-KSPs for the diagonal blocks.
  !--------------------------------------------------------------------
  subroutine petsc_physics_pc_build_reduced(A_full)
    use mod_parameters, only: n_var, n_tor, n_degrees, var_psi, var_u, var_zj, var_w, var_rho, var_T
    use phys_module, only: physics_pc_reassemble, debug_physics_pc, physics_pc_monolithic

    Mat, intent(in) :: A_full

    PetscErrorCode :: ierr
    PetscInt :: bs_ntor
    integer :: comm, my_id, mpierr
    logical :: first_time, use_reassembled
    PetscReal :: norm_val
    Mat :: diag_11, diag_22, diag_55, diag_66  ! pointers to chosen diagonal blocks

    call PetscObjectGetComm(A_full, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)


    first_time = .not. g_ctx%reduced_ready
    use_reassembled = physics_pc_reassemble

    if (my_id == 0) then
      if (use_reassembled) then
        write(*,'(A)') "[Physics PC] Building reduced 4x4 system (reassembled diagonal blocks)..."
      else
        write(*,'(A)') "[Physics PC] Building reduced 4x4 system (extracted blocks)..."
      endif
    endif

    ! --- Step 1: Create index sets (first time only) ---
    if (.not. g_ctx%is_created) then
      call create_variable_index_sets(A_full, comm)
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Index sets created"
    endif

    ! --- Step 1b: Assemble simplified diagonal blocks if requested ---
    if (use_reassembled) then
      call petsc_assemble_pc_diagonal_matrices(A_full, comm, my_id)
    endif

    ! --- Step 2: Extract sub-blocks from full system ---
    ! Diagonal blocks of the constraint equations
    call extract_sub_block(A_full, var_zj, var_zj, g_ctx%B_33, first_time)
    call extract_sub_block(A_full, var_w,  var_w,  g_ctx%B_44, first_time)

    ! Constraint operator blocks (always needed for back-sub and Schur correction)
    call extract_sub_block(A_full, var_zj,  var_psi, g_ctx%B_31, first_time)
    call extract_sub_block(A_full, var_w,   var_u,   g_ctx%B_42, first_time)

    ! Coupling blocks TO j,w (always needed for RHS correction)
    call extract_sub_block(A_full, var_psi, var_zj, g_ctx%B_13, first_time)
    call extract_sub_block(A_full, var_u,   var_zj, g_ctx%B_23, first_time)
    call extract_sub_block(A_full, var_u,   var_w,  g_ctx%B_24, first_time)
    call extract_sub_block(A_full, var_T,   var_zj, g_ctx%B_63, first_time)

    ! Diagonal blocks of the 4x4 system (only extract if not reassembling)
    if (.not. use_reassembled) then
      call extract_sub_block(A_full, var_psi, var_psi, g_ctx%B_11, first_time)
      call extract_sub_block(A_full, var_u,   var_u,   g_ctx%B_22, first_time)
      call extract_sub_block(A_full, var_rho, var_rho, g_ctx%B_55, first_time)
      call extract_sub_block(A_full, var_T,   var_T,   g_ctx%B_66, first_time)
    else if (debug_physics_pc) then
      ! In debug mode, extract anyway to compare norms
      call extract_sub_block(A_full, var_psi, var_psi, g_ctx%B_11, first_time)
      call extract_sub_block(A_full, var_u,   var_u,   g_ctx%B_22, first_time)
      call extract_sub_block(A_full, var_rho, var_rho, g_ctx%B_55, first_time)
      call extract_sub_block(A_full, var_T,   var_T,   g_ctx%B_66, first_time)
    endif

    ! Off-diagonal blocks of the 4x4 system (always needed for coupled mode)
    call extract_sub_block(A_full, var_psi, var_u,   g_ctx%B_12, first_time)
    call extract_sub_block(A_full, var_psi, var_T,   g_ctx%B_16, first_time)
    call extract_sub_block(A_full, var_u,   var_psi, g_ctx%B_21, first_time)
    call extract_sub_block(A_full, var_u,   var_rho, g_ctx%B_25, first_time)
    call extract_sub_block(A_full, var_u,   var_T,   g_ctx%B_26, first_time)
    call extract_sub_block(A_full, var_rho, var_psi, g_ctx%B_51, first_time)
    call extract_sub_block(A_full, var_rho, var_u,   g_ctx%B_52, first_time)
    call extract_sub_block(A_full, var_T,   var_psi, g_ctx%B_61, first_time)
    call extract_sub_block(A_full, var_T,   var_u,   g_ctx%B_62, first_time)

    if (use_reassembled) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Sub-blocks extracted (17 off-diag/constraint)"
    else
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Sub-blocks extracted (21 blocks)"
    endif

    ! --- Step 3: Compute block diagonal inverse (n_tor x n_tor blocks) ---
    ! bs_ntor = n_tor  ! TODO: n_degrees*n_tor would be better but axis node breaks uniform blocking
    ! call compute_block_diagonal_inverse(g_ctx%B_33, g_ctx%Dinv_Mj, bs_ntor, first_time)
    ! call compute_block_diagonal_inverse(g_ctx%B_44, g_ctx%Dinv_Mw, bs_ntor, first_time)
    ! g_ctx%dinv_created = .true.
    ! if (my_id == 0) write(*,'(A)') "[Physics PC]   Computed block diagonal inverse (block_size = n_tor)"

    ! call MatNorm(g_ctx%Dinv_Mj, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Dinv_Mj||_F = ", norm_val
    ! call MatNorm(g_ctx%Dinv_Mw, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Dinv_Mw||_F = ", norm_val

    call compute_diag_mass_inverse(g_ctx%B_33, g_ctx%diag_Mj_inv, first_time)
    call compute_diag_mass_inverse(g_ctx%B_44, g_ctx%diag_Mw_inv, first_time)
    call VecNorm(g_ctx%diag_Mj_inv, NORM_2, norm_val, ierr)
    if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||D_j^{-1} (diag)||_2 = ", norm_val
    call VecNorm(g_ctx%diag_Mw_inv, NORM_2, norm_val, ierr)
    if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||D_w^{-1} (diag)||_2 = ", norm_val

    ! --- Debug: compare reassembled vs extracted norms ---
    if (use_reassembled .and. debug_physics_pc) then
      call MatNorm(g_ctx%B_11, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_11 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_11, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_11 (reassembled)||_F = ", norm_val
      call MatNorm(g_ctx%B_22, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_22 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_22, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_22 (reassembled)||_F = ", norm_val
      call MatNorm(g_ctx%B_55, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_55 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_55, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_55 (reassembled)||_F = ", norm_val
      call MatNorm(g_ctx%B_66, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_66 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_66, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_66 (reassembled)||_F = ", norm_val
    endif

    ! --- Step 4a: Set up KSPs for elliptic constraint mass matrices ---
    ! (Must be done before Schur correction so MUMPS factorization is available)
    call setup_block_ksp(g_ctx%ksp_Mj, g_ctx%B_33, comm, first_time)
    call setup_block_ksp(g_ctx%ksp_Mw, g_ctx%B_44, comm, first_time)
    g_ctx%ksp_elliptic_created = .true.
    if (my_id == 0) write(*,'(A)') "[Physics PC]   Elliptic KSPs set up (PREONLY+LU+MUMPS)"

    ! --- Step 4b: Form Schur-corrected blocks using exact M^{-1} via MatMatSolve ---
    ! Choose diagonal blocks: reassembled (R_*) or extracted (B_*)
    if (use_reassembled) then
      ! Ã_11 = R_11 - B_13 * B_33^{-1} * B_31
      call compute_schur_corrected_block_exact(g_ctx%R_11, g_ctx%B_13, g_ctx%B_31, &
                                          g_ctx%ksp_Mj, g_ctx%Atilde_11, first_time)
      ! call compute_schur_corrected_block_diag(g_ctx%R_11, g_ctx%B_13, g_ctx%B_31, &
      !                                     g_ctx%diag_Mj_inv, g_ctx%Atilde_11, first_time)
      ! Ã_22 = R_22 - B_24 * B_44^{-1} * B_42
      call compute_schur_corrected_block_exact(g_ctx%R_22, g_ctx%B_24, g_ctx%B_42, &
                                          g_ctx%ksp_Mw, g_ctx%Atilde_22, first_time)
      ! call compute_schur_corrected_block_diag(g_ctx%R_22, g_ctx%B_24, g_ctx%B_42, &
      !                                     g_ctx%diag_Mw_inv, g_ctx%Atilde_22, first_time)
    else
      ! Ã_11 = B_11 - B_13 * B_33^{-1} * B_31
      call compute_schur_corrected_block_exact(g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, &
                                          g_ctx%ksp_Mj, g_ctx%Atilde_11, first_time)
      ! call compute_schur_corrected_block_diag(g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, &
      !                                     g_ctx%diag_Mj_inv, g_ctx%Atilde_11, first_time)
      ! Ã_22 = B_22 - B_24 * B_44^{-1} * B_42
      call compute_schur_corrected_block_exact(g_ctx%B_22, g_ctx%B_24, g_ctx%B_42, &
                                          g_ctx%ksp_Mw, g_ctx%Atilde_22, first_time)
      ! call compute_schur_corrected_block_diag(g_ctx%B_22, g_ctx%B_24, g_ctx%B_42, &
      !                                     g_ctx%diag_Mw_inv, g_ctx%Atilde_22, first_time)
    endif

    ! Off-diagonal Schur corrections (always from extracted blocks)
    ! Ã_21 = B_21 - B_23 * B_33^{-1} * B_31
    call compute_schur_corrected_block_exact(g_ctx%B_21, g_ctx%B_23, g_ctx%B_31, &
                                        g_ctx%ksp_Mj, g_ctx%Atilde_21, first_time)
    ! call compute_schur_corrected_block_diag(g_ctx%B_21, g_ctx%B_23, g_ctx%B_31, &
    !                                     g_ctx%diag_Mj_inv, g_ctx%Atilde_21, first_time)
    ! Ã_61 = B_61 - B_63 * B_33^{-1} * B_31
    call compute_schur_corrected_block_exact(g_ctx%B_61, g_ctx%B_63, g_ctx%B_31, &
                                        g_ctx%ksp_Mj, g_ctx%Atilde_61, first_time)
    ! call compute_schur_corrected_block_diag(g_ctx%B_61, g_ctx%B_63, g_ctx%B_31, &
    !                                     g_ctx%diag_Mj_inv, g_ctx%Atilde_61, first_time)
    

    if (my_id == 0) write(*,'(A)') "[Physics PC]   Computed Schur-corrected blocks (exact M^{-1})"

    call MatNorm(g_ctx%Atilde_11, NORM_FROBENIUS, norm_val, ierr)
    if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_11||_F = ", norm_val
    call MatNorm(g_ctx%Atilde_22, NORM_FROBENIUS, norm_val, ierr)
    if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_22||_F = ", norm_val
    call MatNorm(g_ctx%Atilde_21, NORM_FROBENIUS, norm_val, ierr)
    if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_21||_F = ", norm_val
    call MatNorm(g_ctx%Atilde_61, NORM_FROBENIUS, norm_val, ierr)
    if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_61||_F = ", norm_val

    ! --- Step 5: Set up solver(s) ---
    if (physics_pc_monolithic) then
      ! Monolithic mode: assemble full 4x4 reduced matrix and single KSP
      call assemble_monolithic_4x4(use_reassembled, comm, first_time, my_id)
    else
      ! Block-diagonal mode: 4 separate sub-KSPs
      call setup_block_ksp(g_ctx%ksp_psi, g_ctx%Atilde_11, comm, first_time)
      call setup_block_ksp(g_ctx%ksp_u,   g_ctx%Atilde_22, comm, first_time)
      if (use_reassembled) then
        call setup_block_ksp(g_ctx%ksp_rho, g_ctx%R_55, comm, first_time)
        call setup_block_ksp(g_ctx%ksp_T,   g_ctx%R_66, comm, first_time)
      else
        call setup_block_ksp(g_ctx%ksp_rho, g_ctx%B_55, comm, first_time)
        call setup_block_ksp(g_ctx%ksp_T,   g_ctx%B_66, comm, first_time)
      endif
      g_ctx%ksp_created = .true.
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Sub-KSPs set up (PREONLY+LU+MUMPS)"
    endif

    ! --- Step 6: Allocate work vectors (first time only) ---
    if (first_time) then
      if (use_reassembled) then
        call MatCreateVecs(g_ctx%R_11, g_ctx%work_1, PETSC_NULL_VEC, ierr)
      else
        call MatCreateVecs(g_ctx%B_11, g_ctx%work_1, PETSC_NULL_VEC, ierr)
      endif
      call VecDuplicate(g_ctx%work_1, g_ctx%work_2, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_3, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_4, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_5, ierr)
    endif

    g_ctx%reduced_ready = .true.
    g_ctx%comm = comm

    if (my_id == 0) write(*,'(A)') "[Physics PC] Reduced system ready."
  end subroutine petsc_physics_pc_build_reduced


  !--------------------------------------------------------------------
  !> PCSHELL apply callback: compute y = P^{-1} x.
  !!
  !! Algorithm:
  !!   1. Extract variable sub-vectors from x and y
  !!   2. Compute Mass-scaled constraint residuals
  !!   3. Schur-correct the RHS for the 4x4 reduced system
  !!   4. Solve diagonal blocks of the reduced system
  !!   5. Back-substitute for j and w
  !!   6. Restore sub-vectors
  !--------------------------------------------------------------------
  subroutine physics_pc_apply(pc_obj, x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    use phys_module, only: physics_pc_coupled, physics_pc_monolithic

    PC :: pc_obj
    Vec :: x, y
    PetscErrorCode :: ierr

    ! Sub-vectors (views into x and y)
    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T

    ! Monolithic mode sub-vector views
    Vec :: rhs_psi, rhs_u, rhs_rho, rhs_T
    Vec :: sol_psi, sol_u, sol_rho, sol_T

    if (.not. g_ctx%reduced_ready) then
      ierr = 1
      return
    endif

    ! --- Step 1: Extract variable sub-vectors ---
    call VecGetSubVector(x, g_ctx%is_var(var_psi), x_psi, ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_u),   x_u,   ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_zj),  x_j,   ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_w),   x_w,   ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_rho), x_rho, ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_T),   x_T,   ierr)

    call VecGetSubVector(y, g_ctx%is_var(var_psi), y_psi, ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_u),   y_u,   ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_zj),  y_j,   ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_w),   y_w,   ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_rho), y_rho, ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_T),   y_T,   ierr)

    ! --- Step 2: Apply inverse of elliptic constraint mass matrices ---
    ! work_1 = A_33^{-1} * x_j  (temp_j)
    ! call VecPointwiseMult(g_ctx%work_1, g_ctx%diag_Mj_inv, x_j, ierr)  ! mass approx
    call KSPSolve(g_ctx%ksp_Mj, x_j, g_ctx%work_1, ierr)
    ! work_2 = A_44^{-1} * x_w  (temp_w)
    ! call VecPointwiseMult(g_ctx%work_2, g_ctx%diag_Mw_inv, x_w, ierr)  ! mass approx
    call KSPSolve(g_ctx%ksp_Mw, x_w, g_ctx%work_2, ierr)

    ! --- Step 3: Schur-correct the RHS and solve ---
    if (physics_pc_monolithic) then
      ! ---- Monolithic 4x4 solve ----
      ! Compute Schur-corrected RHS into work vectors, then scatter into monolithic vector

      ! b_psi = x_psi - B_13 * temp_j  → store in work_3
      call MatMult(g_ctx%B_13, g_ctx%work_1, g_ctx%work_3, ierr)
      call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_psi, ierr)
      call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(1), rhs_psi, ierr)
      call VecCopy(g_ctx%work_4, rhs_psi, ierr)
      call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(1), rhs_psi, ierr)

      ! b_u = x_u - B_23 * temp_j - B_24 * temp_w  → store in work_5
      call MatMult(g_ctx%B_23, g_ctx%work_1, g_ctx%work_3, ierr)
      call MatMult(g_ctx%B_24, g_ctx%work_2, g_ctx%work_4, ierr)
      call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, x_u, ierr)
      call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_4, ierr)
      call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(2), rhs_u, ierr)
      call VecCopy(g_ctx%work_5, rhs_u, ierr)
      call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(2), rhs_u, ierr)

      ! b_rho = x_rho (no Schur correction)
      call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(3), rhs_rho, ierr)
      call VecCopy(x_rho, rhs_rho, ierr)
      call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(3), rhs_rho, ierr)

      ! b_T = x_T - B_63 * temp_j  → store in work_4
      call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_3, ierr)
      call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_T, ierr)
      call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(4), rhs_T, ierr)
      call VecCopy(g_ctx%work_4, rhs_T, ierr)
      call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(4), rhs_T, ierr)

      ! Single monolithic solve
      call KSPSolve(g_ctx%ksp_reduced, g_ctx%work_rhs_4v, g_ctx%work_sol_4v, ierr)

      ! Scatter solution back to per-variable output vectors
      call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(1), sol_psi, ierr)
      call VecCopy(sol_psi, y_psi, ierr)
      call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(1), sol_psi, ierr)

      call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(2), sol_u, ierr)
      call VecCopy(sol_u, y_u, ierr)
      call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(2), sol_u, ierr)

      call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(3), sol_rho, ierr)
      call VecCopy(sol_rho, y_rho, ierr)
      call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(3), sol_rho, ierr)

      call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(4), sol_T, ierr)
      call VecCopy(sol_T, y_T, ierr)
      call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(4), sol_T, ierr)

    else
      ! ---- Block-diagonal / block-triangular solve ----
      ! b_psi = x_psi - B_13 * temp_j
      call MatMult(g_ctx%B_13, g_ctx%work_1, g_ctx%work_3, ierr)
      call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_psi, ierr)
      ! Solve psi block: Ã_11 * y_psi = b_psi
      call KSPSolve(g_ctx%ksp_psi, g_ctx%work_4, y_psi, ierr)

      ! b_u = x_u - B_23 * temp_j - B_24 * temp_w
      call MatMult(g_ctx%B_23, g_ctx%work_1, g_ctx%work_3, ierr)
      call MatMult(g_ctx%B_24, g_ctx%work_2, g_ctx%work_4, ierr)
      call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, x_u, ierr)
      call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_4, ierr)
      if (physics_pc_coupled) then
        ! Lower-triangular correction: b_u -= Ã_21 * y_psi
        call MatMult(g_ctx%Atilde_21, y_psi, g_ctx%work_3, ierr)
        call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
      endif
      ! Solve u block: Ã_22 * y_u = b_u
      call KSPSolve(g_ctx%ksp_u, g_ctx%work_5, y_u, ierr)

      ! b_rho = x_rho
      if (physics_pc_coupled) then
        ! Lower-triangular correction: b_rho -= B_51 * y_psi + B_52 * y_u
        call VecCopy(x_rho, g_ctx%work_5, ierr)
        call MatMult(g_ctx%B_51, y_psi, g_ctx%work_3, ierr)
        call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
        call MatMult(g_ctx%B_52, y_u, g_ctx%work_3, ierr)
        call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
        call KSPSolve(g_ctx%ksp_rho, g_ctx%work_5, y_rho, ierr)
      else
        call KSPSolve(g_ctx%ksp_rho, x_rho, y_rho, ierr)
      endif

      ! b_T = x_T - B_63 * temp_j
      call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_3, ierr)
      call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_T, ierr)
      if (physics_pc_coupled) then
        ! Lower-triangular correction: b_T -= Ã_61 * y_psi + B_62 * y_u
        call MatMult(g_ctx%Atilde_61, y_psi, g_ctx%work_3, ierr)
        call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
        call MatMult(g_ctx%B_62, y_u, g_ctx%work_3, ierr)
        call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
      endif
      ! Solve T block: B_66 * y_T = b_T
      call KSPSolve(g_ctx%ksp_T, g_ctx%work_4, y_T, ierr)
    endif

    ! --- Step 4: Back-substitute for j and w ---
    ! y_j = A_33^{-1} * (x_j - B_31 * y_psi)
    call MatMult(g_ctx%B_31, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_j, ierr)
    ! call VecPointwiseMult(y_j, g_ctx%diag_Mj_inv, g_ctx%work_4, ierr)  ! mass approx
    call KSPSolve(g_ctx%ksp_Mj, g_ctx%work_4, y_j, ierr)

    ! y_w = A_44^{-1} * (x_w - B_42 * y_u)
    call MatMult(g_ctx%B_42, y_u, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_w, ierr)
    ! call VecPointwiseMult(y_w, g_ctx%diag_Mw_inv, g_ctx%work_4, ierr)  ! mass approx
    call KSPSolve(g_ctx%ksp_Mw, g_ctx%work_4, y_w, ierr)

    ! --- Step 5: Restore sub-vectors ---
    call VecRestoreSubVector(x, g_ctx%is_var(var_psi), x_psi, ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_u),   x_u,   ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_zj),  x_j,   ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_w),   x_w,   ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_rho), x_rho, ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_T),   x_T,   ierr)

    call VecRestoreSubVector(y, g_ctx%is_var(var_psi), y_psi, ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_u),   y_u,   ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_zj),  y_j,   ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_w),   y_w,   ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_rho), y_rho, ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_T),   y_T,   ierr)

    ierr = 0
  end subroutine physics_pc_apply


  !--------------------------------------------------------------------
  !> Create an n_vars-variable MPIBAIJ PC matrix (sparsity only).
  !--------------------------------------------------------------------
  subroutine petsc_create_pc_matrix(petsc_A, a_mat, n_vars)
    use data_structure,  only: type_SP_MATRIX
    use mod_parameters,  only: n_var

    Mat,                   intent(out) :: petsc_A
    type(type_SP_MATRIX),  intent(in)  :: a_mat
    integer,               intent(in)  :: n_vars

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


  !> Create the four PC sub-matrices (sparsity allocation only).
  subroutine petsc_create_pc_matrices(a_mat)
    use data_structure,  only: type_SP_MATRIX

    type(type_SP_MATRIX), intent(in) :: a_mat

    g_ctx%comm = a_mat%comm
    call petsc_create_pc_matrix(g_ctx%A_j,    a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_w,    a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_jpsi, a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_wu,   a_mat, 1)
  end subroutine petsc_create_pc_matrices


  !> Assemble the four elliptic PC sub-matrices from element-level data.
  subroutine petsc_assemble_pc_matrices(my_id, local_elms, n_local_elms, a_mat)
    use construct_pc_matrix_mod
    use data_structure,  only: type_SP_MATRIX
    use phys_module,     only: debug_physics_pc

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    PetscErrorCode :: ierr
    logical        :: first_assembly


    first_assembly = .not. g_ctx%matrices_ready

    if (first_assembly) then
      call petsc_create_pc_matrices(a_mat)
    else
      PetscCallA(MatZeroEntries(g_ctx%A_j,    ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_w,    ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_jpsi, ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_wu,   ierr))
    endif

    call construct_pc_elliptic_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                        g_ctx%A_j, g_ctx%A_w, g_ctx%A_jpsi, g_ctx%A_wu)
    g_ctx%matrices_ready = .true.

    if (first_assembly .and. debug_physics_pc) then
      call petsc_analyze_pc_matrices(my_id)
      call petsc_test_pc_matrices(my_id)
    endif
  end subroutine petsc_assemble_pc_matrices


  !> Assemble the four simplified diagonal PC sub-matrices (R_11, R_22, R_55, R_66).
  !! Self-contained: derives ownership from A_full and local elements from global mesh data.
  !! Called from petsc_physics_pc_build_reduced when physics_pc_reassemble = .true.
  subroutine petsc_assemble_pc_diagonal_matrices(A_full, comm, my_id)
    use construct_pc_matrix_mod, only: construct_pc_diagonal_matrices
    use mod_parameters, only: n_var, n_tor, n_degrees, n_vertex_max
    use nodes_elements

    Mat, intent(in) :: A_full
    integer, intent(in) :: comm, my_id

    PetscErrorCode :: ierr
    PetscInt :: rstart, rend, block_size, n_block_local
    PetscInt :: n_local_1v, n_global_1v
    integer :: my_ind_min, my_ind_max
    integer :: ielm, iv, inode, i_order, idx
    logical :: first_assembly

    ! Local element list (computed from mesh + ownership)
    integer, allocatable :: local_elms(:)
    integer :: n_local_elms, n_elements

    first_assembly = .not. g_ctx%reassembled_ready

    ! --- Derive node ownership from PETSc matrix ---
    call MatGetOwnershipRange(A_full, rstart, rend, ierr)
    block_size    = n_var * n_tor
    n_block_local = (rend - rstart) / block_size
    my_ind_min    = rstart / block_size + 1   ! 1-based node index
    my_ind_max    = my_ind_min + n_block_local - 1

    ! --- Compute local element list ---
    n_elements = element_list%n_elements
    allocate(local_elms(n_elements))
    n_local_elms = 0
    element_loop: do ielm = 1, n_elements
      do iv = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(iv)
        do i_order = 1, n_degrees
          idx = node_list%node(inode)%index(i_order)
          if (idx >= my_ind_min .and. idx <= my_ind_max) then
            n_local_elms = n_local_elms + 1
            local_elms(n_local_elms) = ielm
            cycle element_loop 
          endif
        enddo
      enddo
    enddo element_loop

    ! --- Destroy old matrices on rebuild (they were converted to AIJ after first assembly,
    !     so we must recreate as BAIJ for MatSetValuesBlocked in the assembly loop) ---
    if (.not. first_assembly) then
      call MatDestroy(g_ctx%R_11, ierr)
      call MatDestroy(g_ctx%R_22, ierr)
      call MatDestroy(g_ctx%R_55, ierr)
      call MatDestroy(g_ctx%R_66, ierr)
    endif

    ! --- Create fresh 1-var BAIJ matrices ---
    n_local_1v  = n_block_local * n_tor
    n_global_1v = PETSC_DETERMINE
    call MatCreate(comm, g_ctx%R_11, ierr)
    call MatSetSizes(g_ctx%R_11, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_11, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_11, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_11, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    call MatCreate(comm, g_ctx%R_22, ierr)
    call MatSetSizes(g_ctx%R_22, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_22, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_22, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_22, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    call MatCreate(comm, g_ctx%R_55, ierr)
    call MatSetSizes(g_ctx%R_55, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_55, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_55, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_55, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    call MatCreate(comm, g_ctx%R_66, ierr)
    call MatSetSizes(g_ctx%R_66, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_66, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_66, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_66, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    ! Allow new nonzero entries (conservative pre-allocation may undercount)
    call MatSetOption(g_ctx%R_11, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(g_ctx%R_22, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(g_ctx%R_55, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(g_ctx%R_66, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

    ! --- Element-level assembly ---
    call construct_pc_diagonal_matrices(my_id, local_elms(1:n_local_elms), n_local_elms, &
                                         my_ind_min, my_ind_max, &
                                         g_ctx%R_11, g_ctx%R_22, g_ctx%R_55, g_ctx%R_66)

    deallocate(local_elms)

    ! Convert BAIJ to AIJ for compatibility with extracted sub-blocks in Schur correction
    call MatConvert(g_ctx%R_11, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_11, ierr)
    call MatConvert(g_ctx%R_22, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_22, ierr)
    call MatConvert(g_ctx%R_55, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_55, ierr)
    call MatConvert(g_ctx%R_66, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_66, ierr)

    g_ctx%reassembled_ready = .true.

    if (my_id == 0) write(*,'(A)') "[Physics PC]   Diagonal blocks reassembled (R_11, R_22, R_55, R_66)"
  end subroutine petsc_assemble_pc_diagonal_matrices


  !> Refresh the module-level context after a matrix rebuild.
  subroutine petsc_update_physics_pc_ctx()
    ! Reserved for future use
  end subroutine petsc_update_physics_pc_ctx


  !> Print structural info and norms for the elliptic PC sub-matrices.
  subroutine petsc_analyze_pc_matrices(my_id)
    use mod_petsc_matrix_analysis

    integer, intent(in) :: my_id
    PetscReal :: diff_norm
#ifdef USE_SLEPC
    PetscReal :: kappa
#endif

    if (my_id == 0) write(*,'(A)') &
      "=== PC matrix analysis ================================="

    call petsc_mat_print_info(g_ctx%A_j,    "A_j")
    call petsc_mat_print_info(g_ctx%A_w,    "A_w")
    call petsc_mat_print_info(g_ctx%A_jpsi, "A_jpsi")
    call petsc_mat_print_info(g_ctx%A_wu,   "A_wu")

    call petsc_mat_norms(g_ctx%A_j,    "A_j")
    call petsc_mat_norms(g_ctx%A_w,    "A_w")
    call petsc_mat_norms(g_ctx%A_jpsi, "A_jpsi")
    call petsc_mat_norms(g_ctx%A_wu,   "A_wu")

    call petsc_mat_diff_norm(g_ctx%A_j, g_ctx%A_w, "A_j vs A_w", diff_norm)

#ifdef USE_SLEPC
    !call petsc_mat_cond_estimate(g_ctx%A_j, kappa)
    !if (my_id == 0 .and. kappa > 0.0d0) write(*,'(A,ES12.4)') "[PC] cond(A_j) = ", kappa
    !call petsc_mat_cond_estimate(g_ctx%A_w, kappa)
    !if (my_id == 0 .and. kappa > 0.0d0) write(*,'(A,ES12.4)') "[PC] cond(A_w) = ", kappa

    !call petsc_mat_full_spectrum(g_ctx%A_j,    "A_j",    0, symmetric=.true.)
    !call petsc_mat_full_spectrum(g_ctx%A_w,    "A_w",    0, symmetric=.true.)
    !call petsc_mat_full_spectrum(g_ctx%A_jpsi, "A_jpsi", 0, symmetric=.false.)
    !call petsc_mat_full_spectrum(g_ctx%A_wu,   "A_wu",   0, symmetric=.false.)
#endif

    if (my_id == 0) write(*,'(A)') &
      "========================================================"
  end subroutine petsc_analyze_pc_matrices


  !> Run manufactured-solution solver tests on the elliptic PC sub-matrices.
  subroutine petsc_test_pc_matrices(my_id)
    use mod_petsc_matrix_tests

    integer, intent(in) :: my_id

    call petsc_run_matrix_tests(my_id, g_ctx%comm, &
                                 g_ctx%A_j, g_ctx%A_w, g_ctx%A_jpsi, g_ctx%A_wu)
  end subroutine petsc_test_pc_matrices

#endif
end module mod_petsc_pc_physics
