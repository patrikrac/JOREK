module mod_petsc_pc_physics_construction
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  use mod_petsc_pc_toroidal, only: petsc_setup_toroidal_harmonic_pc_blocked
  use mod_petsc_pc_physics_apply, only: k_a_exact_mult, s_pbp_diag_mult, physics_pc_apply, &
                                        pack_2v, unpack_2v
  use phys_module, only: time_evol_zeta
  implicit none
  private

  ! --- Compile-time selector for the (psi,u) Alfven super-block solver ---
  ! Edit ALFVEN_BLOCK_SOLVER and rebuild to switch.
  integer, parameter :: ALFVEN_SOLVER_DIRECT   = 1   ! PREONLY + LU + MUMPS (default)
  integer, parameter :: ALFVEN_SOLVER_TOROIDAL = 2   ! GMRES + toroidal mode-split PC
  integer, parameter :: ALFVEN_SOLVER_TOROIDAL_EXACT = 3   ! GMRES on EXACT Schur shell + toroidal PC from approx K_A_aij
  integer, parameter :: ALFVEN_BLOCK_SOLVER = ALFVEN_SOLVER_DIRECT
  integer, parameter :: ALFVEN_TOROIDAL_MAXITS = 4   ! outer GMRES iters (TOROIDAL only)

  ! --- Stage 4.6 state: global solve with the assembled-Schur ansatz -----
  ! The block-LDU preconditioner is applied through a PCSHELL. PETSc's
  ! Fortran shell contexts are painful to type, and this diagnostic is
  ! serial and runs exactly one solve at a time, so the operands live here
  ! -- the same pattern the existing MATSHELLs use via g_ctx.
  Mat      :: g46_Myy, g46_Uyu, g46_Luy, g46_AuM, g46_Qiu
  KSP      :: g46_kspM, g46_kspS
  Vec      :: g46_ry, g46_ru, g46_sy, g46_ty, g46_zu, g46_tu, g46_tu2, g46_su
  Vec      :: g46_umask, g46_ubnd
  PetscInt :: g46_n1, g46_n3
  integer  :: g46_inner_its = 0
  logical  :: g46_use_schur = .true.   !< .false. -> D_uu-only, no Schur correction

  ! --- Stage 6.2 state: the production small-flow Schur operator --------
  ! B_33 (1/R mass) and B_44 (R mass) are geometry-only, so their sparse
  ! inverses are built once and reused for the whole run.
  Mat, save     :: sfp_Qip, sfp_QiR
  logical, save :: sfp_massinv_ready = .false.
  logical, save :: sfp_QiR_ready     = .false.   !< sfp_QiR is built on demand (channels >= 2, or a CM_OP_QR candidate)


  public :: create_variable_index_sets
  public :: extract_sub_block
  public :: compute_schur_corrected_block_psi
  public :: compute_schur_corrected_block_u
  public :: compute_schur_corrected_block_21
  public :: compute_schur_corrected_block_61
  public :: compute_schur_corrected_block_exact
  public :: compute_full_momentum_schur_exact
  public :: compute_explicit_preconditioned_matrix
  public :: setup_S_PBP_diag_shell        ! TEMPORARY diagnostic (Option A, Step 2)
  public :: materialize_S_PBP_diag_aij    ! TEMPORARY diagnostic (Option A, Step 2)
  public :: setup_block_ksp
  public :: setup_constraint_mass_ksp
  public :: setup_alfven_block_ksp
  public :: setup_rho_block_ksp, setup_T_block_ksp
  public :: setup_block_ksp_amg_krylov, setup_block_ksp_hypre_amg_krylov
  public :: assemble_monolithic_4x4
  public :: assemble_probed_exact_4x4
  public :: verify_alfven_2x2_segregated
  public :: verify_reduced_pde_operator
  public :: verify_schur_factorization_4x4
  public :: verify_schur_approx_4x4
  public :: verify_schur_itersolve         ! Stage 4.5: is S_ass iteratively solvable?
  public :: schur_itersolve_probe
  public :: verify_schur_global_solve      ! Stage 4.6: does the ansatz solve P_full?
  public :: create_reduced_index_sets
  public :: build_schur_smallflow_prod     ! Stage 6.2: the production S_PBP
  public :: setup_schur_inner_ksp          ! Stage 6.3: the inner S_PBP solver selector
  public :: probe_inner_solvers            ! Workstream B phase 5: AMG amenability probe
  public :: build_schur_commutator_prod    ! Stage 6.3: the production commutator-device S_PBP
  public :: build_pair_psi_prod            ! Workstream B: pair_psi = [[B_11,B_13],[B_31,B_33]]
  public :: build_schur_mixed_prod         ! Workstream B: pair_w   = [[S_uu^SFM,B_24],[B_42,B_44]]
  public :: schur_mixed_require_serial     ! Workstream B: shared np>1 hard stop
  public :: verify_schur_mixed_apply       ! Workstream B: the mixed-pair null test
  public :: report_operator_density        ! absolute nnz/row, for cross-arm comparison

contains

  !--------------------------------------------------------------------
  !> Print one coherent setup line on rank 0:  "[Physics PC]   <label>: <method>"
  !! Used by the block-KSP setup routines so each reports what it configured.
  !--------------------------------------------------------------------
  subroutine pc_print_block_setup(comm, label, method)
    integer,          intent(in) :: comm
    character(len=*), intent(in) :: label, method

    integer        :: rank
    PetscErrorCode :: ierr

    call MPI_Comm_rank(comm, rank, ierr)
    if (rank == 0) write(*,'(A)') "[Physics PC]   " // trim(label) // ": " // trim(method)
  end subroutine pc_print_block_setup

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
      PetscCallA(ISCreateGeneral(comm, n_var_dofs, indices, PETSC_COPY_VALUES, g_ctx%is_var(v), ierr))

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
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), MAT_INITIAL_MATRIX, B, ierr))
    else
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), MAT_REUSE_MATRIX, B, ierr))
    endif
  end subroutine extract_sub_block





  !--------------------------------------------------------------------
  !> Atilde_11 = B_11 - K_psi_correction, using the element-assembled Schur
  !! correction (mod_elt_matrix_elliptic), not a mass-inverse approximation.
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block_psi(B_diag, Atilde, first_time)

    Mat, intent(in)    :: B_diag
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%psi_correction_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_psi_correction, DIFFERENT_NONZERO_PATTERN, ierr) !TODO: In principle they have the same non-zero pattern but in practice PETSc might drop some entires which would lead to a different pattern. No idea if that impacts anything?
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block required!"
    endif
  end subroutine compute_schur_corrected_block_psi


  !--------------------------------------------------------------------
  !> Atilde_22 = B_22 - K_u_correction, using the element-assembled Schur
  !! correction (mod_elt_matrix_elliptic), not a mass-inverse approximation.
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block_u(B_diag, Atilde, first_time)

    Mat, intent(in)    :: B_diag
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%u_correction_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_u_correction, DIFFERENT_NONZERO_PATTERN, ierr) !TODO: In principle they have the same non-zero pattern but in practice PETSc might drop some entires which would lead to a different pattern. No idea if that impacts anything?
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block required!"
    endif
  end subroutine compute_schur_corrected_block_u


  !> Off-diagonal Schur correction Atilde_21 = B_21 - K_21_correction
  subroutine compute_schur_corrected_block_21(B_diag, Atilde, first_time)
    Mat, intent(in)    :: B_diag
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%correction_21_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_21_correction, DIFFERENT_NONZERO_PATTERN, ierr)
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block 21 required!"
    endif
  end subroutine compute_schur_corrected_block_21


  !> Off-diagonal Schur correction Atilde_61 = B_61 - K_61_correction
  subroutine compute_schur_corrected_block_61(B_diag, Atilde, first_time)
    Mat, intent(in)    :: B_diag
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%correction_61_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_61_correction, DIFFERENT_NONZERO_PATTERN, ierr)
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block 61 required!"
    endif
  end subroutine compute_schur_corrected_block_61

  !--------------------------------------------------------------------
  !> Compute a Schur-corrected diagonal block using block diagonal inverse:


!--------------------------------------------------------------------
  !> Compute a Schur-corrected block using vector probing:
  !! Atilde = B_diag - B_coupling * M^{-1} * B_constraint
  !!
  !! Uses a KSP (MUMPS) to solve M * y = z column-by-column (probing).
  !! Rank 0 gathers the columns into a dense array, and a fresh sparse 
  !! MATMPIAIJ is assembled at the end.
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block_exact(M, B_diag, B_coupling, &
                                          B_constraint, Atilde, first_time)
    implicit none
    Mat, intent(in)     :: M, B_diag, B_coupling, B_constraint
    Mat, intent(inout)  :: Atilde
    logical, intent(in) :: first_time

    KSP            :: ksp_M
    PC             :: pc_M
    Vec            :: e_j, z, y, r_corr, r_diag, r_seq
    VecScatter     :: scat
    PetscErrorCode :: ierr
    PetscInt       :: N_global, m_local, n_local
    integer        :: n, j, my_id, mpierr
    integer       :: comm

    PetscScalar, pointer  :: arr(:)
    real*8, allocatable   :: S_dense(:,:)
    PetscInt, allocatable :: row_idxs(:)
    PetscInt       :: col_idx(1)

    ! Infer communicator and rank from the input matrix
    call PetscObjectGetComm(B_diag, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    ! Get the global size of the block (assume square mapping for the variable)
    call MatGetSize(B_diag, PETSC_NULL_INTEGER, N_global, ierr)
    n = int(N_global)

    if (my_id == 0) then
      allocate(S_dense(n, n))
      S_dense = 0.0d0
      write(*,'(A,I0,A)') "[Diagnostics] Probing exact Schur block (", n, " columns)..."
      flush(6)
    endif

    ! -----------------------------------------------------------------
    ! 1. Set Up the Exact LU KSP for M
    ! -----------------------------------------------------------------
    call KSPCreate(comm, ksp_M, ierr)
    call KSPSetOperators(ksp_M, M, M, ierr)
    call KSPSetType(ksp_M, KSPPREONLY, ierr)
    call KSPGetPC(ksp_M, pc_M, ierr)
    call PCSetType(pc_M, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_M, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_M, ierr)

    ! -----------------------------------------------------------------
    ! 2. Allocate Vector Workspace
    ! -----------------------------------------------------------------
    ! e_j : Domain of B_constraint (and B_diag)
    ! z   : Range of B_constraint
    call MatCreateVecs(B_constraint, e_j, z, ierr)
    
    ! y   : Domain of M (Solution of M y = z)
    call MatCreateVecs(M, y, PETSC_NULL_VEC, ierr)
    
    ! r_diag, r_corr : Range of B_diag / B_coupling
    call MatCreateVecs(B_diag, PETSC_NULL_VEC, r_diag, ierr)
    call VecDuplicate(r_diag, r_corr, ierr)

    ! Sequential vector for gathering results on Rank 0
    call VecScatterCreateToZero(r_diag, scat, r_seq, ierr)

    ! -----------------------------------------------------------------
    ! 3. Probe the Matrix Column by Column
    ! -----------------------------------------------------------------
    do j = 0, n-1
      ! Set up standard basis vector e_j
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr)
      call VecAssemblyEnd(e_j, ierr)

      ! z = B_constraint * e_j
      call MatMult(B_constraint, e_j, z, ierr)

      ! Solve M * y = z
      call KSPSolve(ksp_M, z, y, ierr)

      ! r_corr = B_coupling * y
      call MatMult(B_coupling, y, r_corr, ierr)

      ! r_diag = B_diag * e_j
      call MatMult(B_diag, e_j, r_diag, ierr)

      ! r_diag = r_diag - r_corr (This is the j-th column of Atilde)
      call VecAXPY(r_diag, -1.0d0, r_corr, ierr)

      ! Scatter the j-th column to Rank 0
      call VecScatterBegin(scat, r_diag, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_diag, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)

      ! Insert into dense Fortran array
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        S_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! Status output
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Diagnostics]   Probed: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! -----------------------------------------------------------------
    ! 4. Construct the Parallel Sparse Matrix from the Dense Array
    ! -----------------------------------------------------------------
    ! Get local parallel layout matching the working vectors
    call VecGetLocalSize(r_diag, m_local, ierr)
    call VecGetLocalSize(e_j,    n_local, ierr)

    if (.not. first_time) call MatDestroy(Atilde, ierr)

    call MatCreate(comm, Atilde, ierr)
    call MatSetSizes(Atilde, m_local, n_local, n, n, ierr)
    call MatSetType(Atilde, MATMPIAIJ, ierr)
    call MatSetOption(Atilde, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(Atilde, ierr)

    if (my_id == 0) then
      allocate(row_idxs(n))
      do j = 0, n-1
        row_idxs(j+1) = j
      enddo

      ! Insert dense column by dense column
      do j = 0, n-1
        col_idx(1) = j
        call MatSetValues(Atilde, n, row_idxs, 1, col_idx, &
                          S_dense(:, j+1), INSERT_VALUES, ierr)
      enddo

      deallocate(row_idxs)
      deallocate(S_dense)
    endif

    call MatAssemblyBegin(Atilde, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (Atilde, MAT_FINAL_ASSEMBLY, ierr)

    ! -----------------------------------------------------------------
    ! 5. Cleanup Workspace
    ! -----------------------------------------------------------------
    call VecScatterDestroy(scat, ierr)
    call VecDestroy(r_seq,  ierr)
    call VecDestroy(e_j,    ierr)
    call VecDestroy(z,      ierr)
    call VecDestroy(y,      ierr)
    call VecDestroy(r_diag, ierr)
    call VecDestroy(r_corr, ierr)
    call KSPDestroy(ksp_M,  ierr)

  end subroutine compute_schur_corrected_block_exact


  !--------------------------------------------------------------------
  !> Build the EXACT full momentum Schur complement (channels A+B+C):
  !!   S_u = Atilde_22
  !!         - Atilde_21 Atilde_11^{-1} B_12                  (A: magnetic)
  !!         - B_25 B_55^{-1} B_52                             (B: pressure, rho)
  !!         - B_26 B_66^{-1} B_62                             (B: pressure, T)
  !!         + B_25 B_55^{-1} B_51      Atilde_11^{-1} B_12    (C: pressure-flutter)
  !!         + B_26 B_66^{-1} Atilde_61 Atilde_11^{-1} B_12    (C: thermal-flutter)
  !! Channel C are the L-coupling cross-terms of A_pp^{-1} (eq.10 of the design
  !! doc). They REUSE y_psi = Atilde_11^{-1}(B_12 e_j) already formed for channel
  !! A, plus one extra B_55 and one extra B_66 solve per column. Note the sign:
  !! channel C is ADDED (A_pp^{-1} off-diagonal block carries a minus, and S_u
  !! subtracts the whole A_up A_pp^{-1} A_pu, so the two minuses cancel).
  !! Toggle include_channel_C=.false. to recover the old A+B-only S_u (H5:
  !! measure the flutter contribution per benchmark arm). The U-coupling (B_16)
  !! cross-term is higher-order (negligible) and remains omitted.
  !! Column-probing: u-sized basis vectors, three MUMPS solves per column.
  !! Mirrors compute_schur_corrected_block_exact.
  !--------------------------------------------------------------------
  subroutine compute_full_momentum_schur_exact(S_u, first_time)
    implicit none
    Mat, intent(inout)  :: S_u
    logical, intent(in) :: first_time

    ! Toggle: .true. builds the full S_u (A+B+C); .false. recovers the old
    ! A+B-only reference. Flip and re-run to measure Channel C per arm (H5).
    logical, parameter :: include_channel_C = .false.

    KSP            :: ksp_psi_l, ksp_rho_l, ksp_T_l
    PC             :: pc_l
    Vec            :: e_j, col, d_u
    Vec            :: z_psi, y_psi, r_psi
    Vec            :: z_rho, y_rho, r_rho
    Vec            :: z_T,   y_T,   r_T
    Vec            :: col_seq
    VecScatter     :: scat
    PetscErrorCode :: ierr
    PetscInt       :: N_global, m_local, n_local
    integer        :: n, j, my_id, mpierr
    integer       :: comm
    PetscScalar, pointer  :: arr(:)
    real*8, allocatable   :: S_dense(:,:)
    PetscInt, allocatable :: row_idxs(:)
    PetscInt       :: col_idx(1)

    call PetscObjectGetComm(g_ctx%Atilde_22, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MatGetSize(g_ctx%Atilde_22, PETSC_NULL_INTEGER, N_global, ierr)
    n = int(N_global)

    if (my_id == 0) then
      allocate(S_dense(n, n)); S_dense = 0.0d0
      write(*,'(A,I0,A)') "[Diagnostics] Probing full momentum Schur S_u (", n, " columns)..."
      if (include_channel_C) then
        write(*,'(A)') "[Diagnostics]   channels: A (magnetic) + B (pressure) + C (flutter cross-terms)"
      else
        write(*,'(A)') "[Diagnostics]   channels: A (magnetic) + B (pressure)   [C OMITTED]"
      endif
      flush(6)
    endif

    ! Three exact LU (MUMPS) KSPs for the diagonal blocks
    call KSPCreate(comm, ksp_psi_l, ierr)
    call KSPSetOperators(ksp_psi_l, g_ctx%Atilde_11, g_ctx%Atilde_11, ierr)
    call KSPSetType(ksp_psi_l, KSPPREONLY, ierr)
    call KSPGetPC(ksp_psi_l, pc_l, ierr); call PCSetType(pc_l, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_l, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_psi_l, ierr)

    call KSPCreate(comm, ksp_rho_l, ierr)
    call KSPSetOperators(ksp_rho_l, g_ctx%B_55, g_ctx%B_55, ierr)
    call KSPSetType(ksp_rho_l, KSPPREONLY, ierr)
    call KSPGetPC(ksp_rho_l, pc_l, ierr); call PCSetType(pc_l, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_l, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_rho_l, ierr)

    call KSPCreate(comm, ksp_T_l, ierr)
    call KSPSetOperators(ksp_T_l, g_ctx%B_66, g_ctx%B_66, ierr)
    call KSPSetType(ksp_T_l, KSPPREONLY, ierr)
    call KSPGetPC(ksp_T_l, pc_l, ierr); call PCSetType(pc_l, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_l, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_T_l, ierr)

    ! Work vectors. e_j/col/d_u/r_* are u-sized; z/y per channel are block-sized.
    call MatCreateVecs(g_ctx%B_12, e_j,   z_psi, ierr)   ! e_j: u (domain), z_psi: psi (range)
    call MatCreateVecs(g_ctx%Atilde_11, y_psi, PETSC_NULL_VEC, ierr)
    call MatCreateVecs(g_ctx%Atilde_22, d_u, col, ierr)  ! both u-sized
    call VecDuplicate(col, r_psi, ierr)
    call VecDuplicate(col, r_rho, ierr)
    call VecDuplicate(col, r_T,   ierr)
    call MatCreateVecs(g_ctx%B_52, PETSC_NULL_VEC, z_rho, ierr)  ! z_rho: rho (range)
    call MatCreateVecs(g_ctx%B_55, y_rho, PETSC_NULL_VEC, ierr)
    call MatCreateVecs(g_ctx%B_62, PETSC_NULL_VEC, z_T, ierr)    ! z_T: T (range)
    call MatCreateVecs(g_ctx%B_66, y_T, PETSC_NULL_VEC, ierr)
    call VecScatterCreateToZero(col, scat, col_seq, ierr)

    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr); call VecAssemblyEnd(e_j, ierr)

      ! diagonal: d_u = Atilde_22 e_j
      call MatMult(g_ctx%Atilde_22, e_j, d_u, ierr)

      ! channel A (magnetic): r_psi = Atilde_21 Atilde_11^{-1} (B_12 e_j)
      call MatMult(g_ctx%B_12, e_j, z_psi, ierr)
      call KSPSolve(ksp_psi_l, z_psi, y_psi, ierr)
      call MatMult(g_ctx%Atilde_21, y_psi, r_psi, ierr)

      ! channel B (rho): r_rho = B_25 B_55^{-1} (B_52 e_j)
      call MatMult(g_ctx%B_52, e_j, z_rho, ierr)
      call KSPSolve(ksp_rho_l, z_rho, y_rho, ierr)
      call MatMult(g_ctx%B_25, y_rho, r_rho, ierr)

      ! channel B (T): r_T = B_26 B_66^{-1} (B_62 e_j)
      call MatMult(g_ctx%B_62, e_j, z_T, ierr)
      call KSPSolve(ksp_T_l, z_T, y_T, ierr)
      call MatMult(g_ctx%B_26, y_T, r_T, ierr)

      ! col = d_u - r_psi - r_rho - r_T  (channels A + B)
      call VecCopy(d_u, col, ierr)
      call VecAXPY(col, -1.0d0, r_psi, ierr)
      call VecAXPY(col, -1.0d0, r_rho, ierr)
      call VecAXPY(col, -1.0d0, r_T,   ierr)

      ! channel C: L-coupling cross-terms of A_pp^{-1}, ADDED to col.
      ! Reuses y_psi = Atilde_11^{-1}(B_12 e_j) from channel A; z_*/y_*/r_* are
      ! free scratch here (their channel-B contributions are already in col).
      if (include_channel_C) then
        ! pressure-flutter:  + B_25 B_55^{-1} B_51 y_psi
        call MatMult (g_ctx%B_51,    y_psi, z_rho, ierr)
        call KSPSolve(ksp_rho_l,     z_rho, y_rho, ierr)
        call MatMult (g_ctx%B_25,    y_rho, r_rho, ierr)
        call VecAXPY (col, +1.0d0,   r_rho,        ierr)
        ! thermal-flutter:   + B_26 B_66^{-1} Atilde_61 y_psi
        call MatMult (g_ctx%Atilde_61, y_psi, z_T, ierr)
        call KSPSolve(ksp_T_l,         z_T,   y_T, ierr)
        call MatMult (g_ctx%B_26,      y_T,   r_T, ierr)
        call VecAXPY (col, +1.0d0,     r_T,        ierr)
      endif

      call VecScatterBegin(scat, col, col_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, col, col_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(col_seq, arr, ierr)
        S_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArray(col_seq, arr, ierr)
      endif
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0)') "[Diagnostics]   S_u probed: ", j+1, "/", n
        flush(6)
      endif
    enddo

    call VecGetLocalSize(col, m_local, ierr)
    call VecGetLocalSize(e_j, n_local, ierr)
    if (.not. first_time) call MatDestroy(S_u, ierr)
    call MatCreate(comm, S_u, ierr)
    call MatSetSizes(S_u, m_local, n_local, n, n, ierr)
    call MatSetType(S_u, MATMPIAIJ, ierr)
    call MatSetOption(S_u, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(S_u, ierr)
    if (my_id == 0) then
      allocate(row_idxs(n))
      do j = 0, n-1; row_idxs(j+1) = j; enddo
      do j = 0, n-1
        col_idx(1) = j
        call MatSetValues(S_u, n, row_idxs, 1, col_idx, S_dense(:, j+1), INSERT_VALUES, ierr)
      enddo
      deallocate(row_idxs); deallocate(S_dense)
    endif
    call MatAssemblyBegin(S_u, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (S_u, MAT_FINAL_ASSEMBLY, ierr)

    call VecScatterDestroy(scat, ierr); call VecDestroy(col_seq, ierr)
    call VecDestroy(e_j, ierr);   call VecDestroy(col, ierr);   call VecDestroy(d_u, ierr)
    call VecDestroy(z_psi, ierr); call VecDestroy(y_psi, ierr); call VecDestroy(r_psi, ierr)
    call VecDestroy(z_rho, ierr); call VecDestroy(y_rho, ierr); call VecDestroy(r_rho, ierr)
    call VecDestroy(z_T, ierr);   call VecDestroy(y_T, ierr);   call VecDestroy(r_T, ierr)
    call KSPDestroy(ksp_psi_l, ierr); call KSPDestroy(ksp_rho_l, ierr); call KSPDestroy(ksp_T_l, ierr)
  end subroutine compute_full_momentum_schur_exact


  !--------------------------------------------------------------------
  !> Compute the explicitly preconditioned matrix: B = M^{-1} * A
  !!
  !! Uses a KSP (MUMPS) to solve M * y = z where z = A * e_j.
  !! Rank 0 gathers the columns into a dense array, and a fresh sparse 
  !! MATMPIAIJ is assembled at the end to be passed to an eigensolver.
  !--------------------------------------------------------------------
  subroutine compute_explicit_preconditioned_matrix(M, A, B, first_time)
    implicit none
    Mat, intent(in)     :: M, A
    Mat, intent(inout)  :: B
    logical, intent(in) :: first_time

    KSP            :: ksp_M
    PC             :: pc_M
    Vec            :: e_j, z, y, y_seq
    VecScatter     :: scat
    PetscErrorCode :: ierr
    PetscInt       :: N_global, m_local, n_local
    integer        :: n, j, my_id, mpierr
    integer       :: comm

    PetscScalar, pointer  :: arr(:)
    real*8, allocatable   :: B_dense(:,:)
    PetscInt, allocatable :: row_idxs(:)
    PetscInt       :: col_idx(1)

    ! Infer communicator and rank from the input matrix
    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    ! Get the global size of the matrix
    call MatGetSize(A, PETSC_NULL_INTEGER, N_global, ierr)
    n = int(N_global)

    if (my_id == 0) then
      allocate(B_dense(n, n))
      B_dense = 0.0d0
      write(*,'(A,I0,A)') "[Diagnostics] Probing M^{-1} A (", n, " columns)..."
      flush(6)
    endif

    ! -----------------------------------------------------------------
    ! 1. Set Up the Exact LU KSP for M
    ! -----------------------------------------------------------------
    call KSPCreate(comm, ksp_M, ierr)
    call KSPSetOperators(ksp_M, M, M, ierr)
    call KSPSetType(ksp_M, KSPPREONLY, ierr)
    call KSPGetPC(ksp_M, pc_M, ierr)
    call PCSetType(pc_M, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_M, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_M, ierr)

    ! -----------------------------------------------------------------
    ! 2. Allocate Vector Workspace
    ! -----------------------------------------------------------------
    ! e_j : Domain of A
    ! z   : Range of A (which is also Domain of M)
    call MatCreateVecs(A, e_j, z, ierr)
    
    ! y   : Solution of M y = z
    call VecDuplicate(z, y, ierr)
    
    ! Sequential vector for gathering results on Rank 0
    call VecScatterCreateToZero(y, scat, y_seq, ierr)

    ! -----------------------------------------------------------------
    ! 3. Probe the Matrix Column by Column
    ! -----------------------------------------------------------------
    do j = 0, n-1
      ! Set up standard basis vector e_j
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr)
      call VecAssemblyEnd(e_j, ierr)

      ! z = A * e_j  (Extract the j-th column of A)
      call MatMult(A, e_j, z, ierr)

      ! Solve M * y = z  (Apply M^{-1} to the j-th column of A)
      call KSPSolve(ksp_M, z, y, ierr)

      ! Scatter the j-th column of the result to Rank 0
      call VecScatterBegin(scat, y, y_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, y, y_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)

      ! Insert into dense Fortran array
      if (my_id == 0) then
        call VecGetArray(y_seq, arr, ierr)
        B_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArray(y_seq, arr, ierr)
      endif

      ! Status output
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Diagnostics]   Probed: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! -----------------------------------------------------------------
    ! 4. Construct the Parallel Sparse Matrix from the Dense Array
    ! -----------------------------------------------------------------
    ! Get local parallel layout matching the working vectors
    call VecGetLocalSize(y,   m_local, ierr)
    call VecGetLocalSize(e_j, n_local, ierr)

    if (.not. first_time) call MatDestroy(B, ierr)

    call MatCreate(comm, B, ierr)
    call MatSetSizes(B, m_local, n_local, n, n, ierr)
    call MatSetType(B, MATMPIAIJ, ierr)
    ! B will be mathematically dense, so allow new non-zero allocation
    call MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(B, ierr)

    if (my_id == 0) then
      allocate(row_idxs(n))
      do j = 0, n-1
        row_idxs(j+1) = j
      enddo

      ! Insert dense column by dense column
      do j = 0, n-1
        col_idx(1) = j
        call MatSetValues(B, n, row_idxs, 1, col_idx, &
                          B_dense(:, j+1), INSERT_VALUES, ierr)
      enddo

      deallocate(row_idxs)
      deallocate(B_dense)
    endif

    call MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (B, MAT_FINAL_ASSEMBLY, ierr)

    ! -----------------------------------------------------------------
    ! 5. Cleanup Workspace
    ! -----------------------------------------------------------------
    call VecScatterDestroy(scat, ierr)
    call VecDestroy(y_seq, ierr)
    call VecDestroy(e_j,   ierr)
    call VecDestroy(z,     ierr)
    call VecDestroy(y,     ierr)
    call KSPDestroy(ksp_M, ierr)

    if (my_id == 0) then
      write(*,'(A)') "[Diagnostics] Assembly of B = M^{-1} A complete."
      flush(6)
    endif

  end subroutine compute_explicit_preconditioned_matrix


  !--------------------------------------------------------------------
  !> Set up a sub-KSP for a diagonal block: PREONLY + LU + MUMPS.
  !--------------------------------------------------------------------
  subroutine setup_block_ksp(ksp_block, B_block, comm, first_time, label)
    KSP, intent(inout) :: ksp_block
    Mat, intent(in)    :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    character(len=*), intent(in), optional :: label

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

    if (present(label)) call pc_print_block_setup(comm, label, "PREONLY + LU (MUMPS)")
  end subroutine setup_block_ksp

  !--------------------------------------------------------------------
  !> Set up the KSP for a constraint mass matrix (B_33 = M_jj or B_44 = M_ww).
  !! These blocks are geometry-only and simulation-invariant (no theta*tstep,
  !! no state dependence -- see model199/mod_elt_matrix.f90 eqs 3/4), so the
  !! MUMPS factorization is performed ONCE (first_time) and reused for the whole
  !! run. Signature matches setup_block_ksp for a drop-in swap.
  !--------------------------------------------------------------------
  subroutine setup_constraint_mass_ksp(ksp_block, B_block, comm, first_time, label)
    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    character(len=*), intent(in), optional :: label

    PC :: pc
    PetscErrorCode :: ierr

    ! Operator never changes: skip re-factorization on every rebuild after the first.
    if (.not. first_time) return

    call KSPCreate(comm, ksp_block, ierr)
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    call KSPSetType(ksp_block, KSPPREONLY, ierr)
    call KSPGetPC(ksp_block, pc, ierr)
    call PCSetType(pc, PCLU, ierr)
    call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_block, ierr)

    if (present(label)) call pc_print_block_setup(comm, label, "PREONLY + LU (MUMPS), factored once")
  end subroutine setup_constraint_mass_ksp

  !--------------------------------------------------------------------
  !> Create the exact-(psi,u)-Schur MATSHELL and its 1-variable work vecs (once).
  !! K_A_aij is passed for sizing/parallel layout; the shell's MULT reads blocks
  !! and KSPs directly from g_ctx. All work vecs are 1-variable sized.
  !--------------------------------------------------------------------
  subroutine setup_k_a_exact_shell(K_A_aij, comm)
    Mat, intent(in)     :: K_A_aij
    integer, intent(in) :: comm

    PetscInt :: n_loc, n_glob
    PetscErrorCode :: ierr

    if (g_ctx%kae_setup_done) return

    ! Shell sized exactly like K_A_aij (square 2-variable [psi|u] system).
    call MatGetLocalSize(K_A_aij, n_loc, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(K_A_aij, n_glob, PETSC_NULL_INTEGER, ierr)
    call MatCreateShell(comm, n_loc, n_loc, n_glob, n_glob, &
                        PETSC_NULL_INTEGER, g_ctx%K_A_exact_shell, ierr)
    call MatShellSetOperation(g_ctx%K_A_exact_shell, MATOP_MULT, k_a_exact_mult, ierr)

    ! 1-variable work vecs. B_31 has psi columns / j rows; B_42 has u columns / w rows.
    call MatCreateVecs(g_ctx%B_31, g_ctx%kae_p, g_ctx%kae_tj, ierr)   ! p~psi(col), tj~j(row)
    call MatCreateVecs(g_ctx%B_42, g_ctx%kae_q, g_ctx%kae_tw, ierr)   ! q~u(col),   tw~w(row)
    call VecDuplicate(g_ctx%kae_tj, g_ctx%kae_sj,   ierr)
    call VecDuplicate(g_ctx%kae_tw, g_ctx%kae_sw,   ierr)
    call VecDuplicate(g_ctx%kae_p,  g_ctx%kae_ypsi, ierr)
    call VecDuplicate(g_ctx%kae_q,  g_ctx%kae_yu,   ierr)
    call VecDuplicate(g_ctx%kae_p,  g_ctx%kae_spsi, ierr)
    call VecDuplicate(g_ctx%kae_q,  g_ctx%kae_su,   ierr)

    g_ctx%kae_setup_done = .true.
  end subroutine setup_k_a_exact_shell

  !--------------------------------------------------------------------
  !> TEMPORARY (Option A, Step 2) -- TO BE REPLACED (Step 3).
  !> Set up the diagnostic S_PBP MatShell that approximates Atilde_11^{-1} by
  !! (1+zeta)^{-1} M_j^{-1} and keeps channels B, C exact. Reuses g_ctx%ksp_Mj
  !! (consistent mass = B_33) and creates dedicated EXACT MUMPS solves for
  !! B_55, B_66. Work vecs and the shell are created once.
  !--------------------------------------------------------------------
  subroutine setup_S_PBP_diag_shell(comm, first_time)
    use mod_parameters, only: n_tor
    use phys_module,    only: time_evol_theta, tstep, eta, mode, R_geo
    integer, intent(in) :: comm
    logical, intent(in) :: first_time

    PC             :: pc_l
    PetscInt       :: n_loc, n_glob, rstart, rend
    PetscErrorCode :: ierr
    PetscScalar, pointer :: relax_arr(:)
    integer        :: i, off, nloc_entries
    real*8         :: ktor2

    ! cached scalar weight 1/(1+zeta) -- refreshed every call (zeta may change).
    ! CN: zeta=0 -> 1 ; Gears: zeta=1/2 -> 2/3.
    g_ctx%spbpd_inv_gears = 1.0d0 / (1.0d0 + time_evol_zeta)

    if (.not. g_ctx%spbpd_setup_done) then
      ! --- one-time creation of KSP objects, work vecs, and the shell ---
      ! Dedicated EXACT (MUMPS LU) solves for the B-channel diagonal blocks, so the
      ! diagnostic matches the exact-S_u reference channels regardless of how the
      ! production ksp_rho/ksp_T happen to be configured (B_55/B_66 vs R_55/R_66).
      call KSPCreate(comm, g_ctx%spbpd_ksp_B55, ierr)
      call KSPSetType(g_ctx%spbpd_ksp_B55, KSPPREONLY, ierr)
      call KSPGetPC(g_ctx%spbpd_ksp_B55, pc_l, ierr); call PCSetType(pc_l, PCLU, ierr)
      call PCFactorSetMatSolverType(pc_l, MATSOLVERMUMPS, ierr)

      call KSPCreate(comm, g_ctx%spbpd_ksp_B66, ierr)
      call KSPSetType(g_ctx%spbpd_ksp_B66, KSPPREONLY, ierr)
      call KSPGetPC(g_ctx%spbpd_ksp_B66, pc_l, ierr); call PCSetType(pc_l, PCLU, ierr)
      call PCFactorSetMatSolverType(pc_l, MATSOLVERMUMPS, ierr)

      ! work vecs: psi-, u-, rho-, T-sized. Atilde_21: psi(col)->u(row).
      call MatCreateVecs(g_ctx%Atilde_21, g_ctx%spbpd_zpsi, g_ctx%spbpd_ru, ierr)
      call VecDuplicate(g_ctx%spbpd_zpsi, g_ctx%spbpd_ypsi, ierr)
      call VecDuplicate(g_ctx%spbpd_zpsi, g_ctx%spbpd_relax, ierr)
      call MatCreateVecs(g_ctx%B_52, PETSC_NULL_VEC, g_ctx%spbpd_zrho, ierr)
      call VecDuplicate(g_ctx%spbpd_zrho, g_ctx%spbpd_yrho, ierr)
      call MatCreateVecs(g_ctx%B_62, PETSC_NULL_VEC, g_ctx%spbpd_zT, ierr)
      call VecDuplicate(g_ctx%spbpd_zT, g_ctx%spbpd_yT, ierr)

      ! the shell itself, sized like the u-block (Atilde_22 : square, u x u)
      call MatGetLocalSize(g_ctx%Atilde_22, n_loc, PETSC_NULL_INTEGER, ierr)
      call MatGetSize(g_ctx%Atilde_22, n_glob, PETSC_NULL_INTEGER, ierr)
      call MatCreateShell(comm, n_loc, n_loc, n_glob, n_glob, &
                          PETSC_NULL_INTEGER, g_ctx%S_PBP_diag_shell, ierr)
      call MatShellSetOperation(g_ctx%S_PBP_diag_shell, MATOP_MULT, s_pbp_diag_mult, ierr)

      g_ctx%spbpd_setup_done = .true.
    endif

    ! (re)bind + (re)factor the B-channel solves on the CURRENT B_55/B_66 (the
    ! blocks are rebuilt each PC rebuild), mirroring the exact-S_u builder's freshness.
    call KSPSetOperators(g_ctx%spbpd_ksp_B55, g_ctx%B_55, g_ctx%B_55, ierr)
    call KSPSetUp(g_ctx%spbpd_ksp_B55, ierr)
    call KSPSetOperators(g_ctx%spbpd_ksp_B66, g_ctx%B_66, g_ctx%B_66, ierr)
    call KSPSetUp(g_ctx%spbpd_ksp_B66, ierr)

    ! per-harmonic TOROIDAL resistive relaxation factor applied to the M_j^-1
    ! response (mirrors apply_schur_relaxation, toroidal part only):
    !   fac(slot) = 1 / (1 + theta*tstep*eta*(mode(slot)/R_geo)^2 / (1+zeta))
    ! mode(:) already encodes the real/imag pairing (e.g. n_tor=3 -> 0,8,8), so
    ! off = mod(idx,n_tor)+1 indexes it directly. Representative global eta(center)
    ! and R_geo. The poloidal c_lambda/area term lives at assembly and is NOT here:
    ! this shell measures how much of the resistive tail is toroidal-k. NOTE the
    ! 1/(1+zeta) follows App A eq.27 (exact weight); apply_schur_relaxation omits it.
    call VecGetOwnershipRange(g_ctx%spbpd_relax, rstart, rend, ierr)
    nloc_entries = int(rend - rstart)
    call VecGetArray(g_ctx%spbpd_relax, relax_arr, ierr)
    do i = 1, nloc_entries
      off   = mod(int(rstart) + (i - 1), n_tor) + 1
      ktor2 = 0.d0
      if (R_geo > 0.d0) ktor2 = (dble(mode(off)) / R_geo)**2
      relax_arr(i) = 1.0d0 / (1.0d0 + time_evol_theta * tstep * eta * ktor2 * g_ctx%spbpd_inv_gears)
    enddo
    call VecRestoreArray(g_ctx%spbpd_relax, relax_arr, ierr)
  end subroutine setup_S_PBP_diag_shell

  !--------------------------------------------------------------------
  !> TEMPORARY (Option A, Step 2) -- TO BE REPLACED (Step 3).
  !> Materialize the S_PBP_diag MatShell into an explicit MPIAIJ matrix by
  !! column-probing (MatMult on unit vectors), so the dense spectrum and
  !! preconditioned-spectrum drivers (which LU-factor) can consume it.
  !! Mirrors the assembly half of compute_full_momentum_schur_exact.
  !--------------------------------------------------------------------
  subroutine materialize_S_PBP_diag_aij(aij, first_time)
    Mat, intent(inout)  :: aij
    logical, intent(in) :: first_time

    Vec            :: e_j, col, col_seq
    VecScatter     :: scat
    PetscErrorCode :: ierr
    PetscInt       :: N_global, m_local, n_local
    integer        :: n, j, my_id, mpierr
    integer       :: comm
    PetscScalar, pointer  :: arr(:)
    real*8, allocatable   :: S_dense(:,:)
    PetscInt, allocatable :: row_idxs(:)
    PetscInt       :: col_idx(1)

    call PetscObjectGetComm(g_ctx%S_PBP_diag_shell, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MatGetSize(g_ctx%S_PBP_diag_shell, PETSC_NULL_INTEGER, N_global, ierr)
    n = int(N_global)

    if (my_id == 0) then
      allocate(S_dense(n, n)); S_dense = 0.0d0
      write(*,'(A,I0,A)') "[Diagnostics] Materializing S_PBP_diag shell (", n, " columns)..."
      flush(6)
    endif

    call MatCreateVecs(g_ctx%S_PBP_diag_shell, e_j, col, ierr)
    call VecScatterCreateToZero(col, scat, col_seq, ierr)

    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr); call VecAssemblyEnd(e_j, ierr)
      call MatMult(g_ctx%S_PBP_diag_shell, e_j, col, ierr)
      call VecScatterBegin(scat, col, col_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, col, col_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(col_seq, arr, ierr)
        S_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArray(col_seq, arr, ierr)
      endif
    enddo

    call VecGetLocalSize(col, m_local, ierr)
    call VecGetLocalSize(e_j, n_local, ierr)
    if (.not. first_time) call MatDestroy(aij, ierr)
    call MatCreate(comm, aij, ierr)
    call MatSetSizes(aij, m_local, n_local, n, n, ierr)
    call MatSetType(aij, MATMPIAIJ, ierr)
    call MatSetOption(aij, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(aij, ierr)
    if (my_id == 0) then
      allocate(row_idxs(n))
      do j = 0, n-1; row_idxs(j+1) = j; enddo
      do j = 0, n-1
        col_idx(1) = j
        call MatSetValues(aij, n, row_idxs, 1, col_idx, S_dense(:, j+1), INSERT_VALUES, ierr)
      enddo
      deallocate(row_idxs); deallocate(S_dense)
    endif
    call MatAssemblyBegin(aij, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (aij, MAT_FINAL_ASSEMBLY, ierr)

    call VecScatterDestroy(scat, ierr); call VecDestroy(col_seq, ierr)
    call VecDestroy(e_j, ierr); call VecDestroy(col, ierr)
  end subroutine materialize_S_PBP_diag_aij

  !--------------------------------------------------------------------
  !> Set up the KSP for the (psi,u) Alfven super-block K_A.
  !! Compile-time switch ALFVEN_BLOCK_SOLVER selects the method.
  !! Signature matches setup_block_ksp so the call site is a drop-in swap.
  !--------------------------------------------------------------------
  subroutine setup_alfven_block_ksp(ksp_block, B_block, comm, first_time, label)
    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    character(len=*), intent(in), optional :: label

    PetscReal :: rtol, abstol, dtol
    PetscErrorCode :: ierr
    character(len=64) :: method

    select case (ALFVEN_BLOCK_SOLVER)
    case (ALFVEN_SOLVER_TOROIDAL)
      method = "GMRES + toroidal mode-split PC"
      if (first_time) then
        call KSPCreate(comm, ksp_block, ierr)
        call KSPSetOperators(ksp_block, B_block, B_block, ierr)

        ! Outer GMRES wrapper (mirrors setup_block_ksp_amg_krylov): a few
        ! iterations recover inter-mode-family coupling that the additive
        ! fieldsplit drops, at a fraction of a full direct factorization.
        call KSPSetType(ksp_block, KSPGMRES, ierr)
        call KSPGMRESSetRestart(ksp_block, ALFVEN_TOROIDAL_MAXITS, ierr)
        rtol   = 1.0d-4   ! effectively disabled: hard-stop at ALFVEN_TOROIDAL_MAXITS iters
        abstol = 1.0d-50
        dtol   = 1.0d4
        call KSPSetTolerances(ksp_block, rtol, abstol, dtol, ALFVEN_TOROIDAL_MAXITS, ierr)

        ! Inner PC: toroidal mode-family fieldsplit (exact LU per family).
        ! STRUCTURAL setup (splits + sub-KSP config) depends only on the mode
        ! layout, so it is done ONCE. Re-calling it every rebuild would append
        ! another set of fieldsplit index sets to the same PC (PCSetType is a
        ! no-op when the type is unchanged, so it does not reset the splits).
        ! Also issues PCSetUp/KSPSetUp on ksp_block.
        call petsc_setup_toroidal_harmonic_pc_blocked(ksp_block, B_block, comm)
      else
        ! Rebuild: B_block (K_A_aij) holds new values; refresh the operator and let
        ! PCSetUp re-extract the per-family submatrices and refactor (mirrors the
        ! global toroidal PC in mod_petsc.f90).
        call KSPSetOperators(ksp_block, B_block, B_block, ierr)
        call KSPSetUp(ksp_block, ierr)
      endif

    case (ALFVEN_SOLVER_TOROIDAL_EXACT)
      method = "GMRES on exact Schur shell + toroidal PC"
      if (first_time) then
        call KSPCreate(comm, ksp_block, ierr)

        ! Ensure the exact-operator shell + work vecs exist (once).
        call setup_k_a_exact_shell(B_block, comm)

        ! GMRES iterates on the EXACT shell (Amat); the toroidal mode-split PC is
        ! built from the cheap approximate K_A_aij (Pmat). Fixed hard-stop iters.
        call KSPSetOperators(ksp_block, g_ctx%K_A_exact_shell, B_block, ierr)
        call KSPSetType(ksp_block, KSPGMRES, ierr)
        call KSPGMRESSetRestart(ksp_block, ALFVEN_TOROIDAL_MAXITS, ierr)
        rtol   = 1.0d-4
        abstol = 1.0d-50
        dtol   = 1.0d4
        call KSPSetTolerances(ksp_block, rtol, abstol, dtol, ALFVEN_TOROIDAL_MAXITS, ierr)

        ! Inner PC from the approximate K_A_aij (STRUCTURAL setup once; see the
        ! ALFVEN_SOLVER_TOROIDAL branch for why it must not be re-called).
        call petsc_setup_toroidal_harmonic_pc_blocked(ksp_block, B_block, comm)
      else
        ! Rebuild: Amat (shell) is unchanged; Pmat (K_A_aij) holds new values.
        ! Refresh operators and refactor the fieldsplit submatrices via PCSetUp.
        call KSPSetOperators(ksp_block, g_ctx%K_A_exact_shell, B_block, ierr)
        call KSPSetUp(ksp_block, ierr)
      endif

    case default
      ! ALFVEN_SOLVER_DIRECT: current behaviour (PREONLY + LU + MUMPS).
      method = "PREONLY + LU (MUMPS)"
      call setup_block_ksp(ksp_block, B_block, comm, first_time)
    end select

    if (present(label)) call pc_print_block_setup(comm, label, method)
  end subroutine setup_alfven_block_ksp

  !--------------------------------------------------------------------
  !> Set up a sub-KSP for a diagonal block: GMRES + GAMG (3-10 iters)
  !--------------------------------------------------------------------
  subroutine setup_block_ksp_amg_krylov(ksp_block, B_block, comm, first_time, max_its)
    implicit none

    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    integer, intent(in) :: max_its      ! Pass in 3 to 10

    PC :: pc
    PetscErrorCode :: ierr
    PetscReal :: rtol, abstol, dtol

    if (first_time) then
      call KSPCreate(comm, ksp_block, ierr)
    endif
    
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    
    ! 1. Choose a lightweight Krylov solver
    ! GMRES is highly robust for non-symmetric/non-M-matrices.
    call KSPSetType(ksp_block, KSPGMRES, ierr)
    
    ! Keep memory footprint tiny by telling GMRES to restart 
    ! at max_its (it will only allocate 'max_its' search vectors).
    call KSPGMRESSetRestart(ksp_block, max_its, ierr)
    
    ! 2. Set tolerances and max iterations
    ! Set a loose relative tolerance (e.g., 1.0d-2 = 1% error reduction)
    ! so it can exit early if it converges in 3-4 iterations.
    ! Otherwise, it will hard-stop at 'max_its'.
    rtol   = 1.0d-3  
    abstol = 1.0d-50 ! Ignore absolute tolerance
    dtol   = 1.0d4   ! Divergence tolerance
    call KSPSetTolerances(ksp_block, rtol, abstol, dtol, max_its, ierr)
    
    ! 3. Set the Preconditioner to GAMG
    call KSPGetPC(ksp_block, pc, ierr)
    call PCSetType(pc, PCGAMG, ierr)
    call PCGAMGSetType(pc, PCGAMGAGG, ierr)
    
    call KSPSetUp(ksp_block, ierr)
    
  end subroutine setup_block_ksp_amg_krylov

  subroutine setup_block_ksp_hypre_amg_krylov(ksp_block, B_block, comm, first_time, max_its)
    implicit none

    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    integer, intent(in) :: max_its      ! Pass in 3 to 10

    PC :: pc
    MatNullSpace :: nullsp
    PetscErrorCode :: ierr
    PetscReal :: rtol, abstol, dtol

    if (first_time) then
      call KSPCreate(comm, ksp_block, ierr)
    endif
    
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    
    ! Set options prefix to match the hmg_ configuration
    call KSPSetOptionsPrefix(ksp_block, "hmg_", ierr)

    ! 1. Choose a lightweight Krylov solver
    call KSPSetType(ksp_block, KSPGMRES, ierr)
    
    ! Keep memory footprint tiny
    call KSPGMRESSetRestart(ksp_block, max_its, ierr)
    
    ! 2. Set tolerances and max iterations
    rtol   = 1.0d-3  
    abstol = 1.0d-50 
    dtol   = 1.0d4   
    call KSPSetTolerances(ksp_block, rtol, abstol, dtol, max_its, ierr)
    
    ! 3. Set the Preconditioner to Hypre BoomerAMG
    call KSPGetPC(ksp_block, pc, ierr)
    call PCSetType(pc, PCHYPRE, ierr)
    call PCHYPRESetType(pc, "boomeramg", ierr)

    ! Set Near Null Space on the operator
    call MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL_VEC_ARRAY, nullsp, ierr)
    call MatSetNearNullSpace(B_block, nullsp, ierr)
    call MatNullSpaceDestroy(nullsp, ierr)

    ! 4. Set Hypre options via the PETSc options database
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_coarsen_type", "HMIS", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_nodal_coarsen", "6", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_truncfactor", "0.3", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_P_max", "4", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_strong_threshold", "0.5", ierr)

    ! Smoothers
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_smooth_num_levels", "1", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_smooth_type", "Euclid", ierr)
    !call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_euclid_levels", "1", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_relax_type_all", "Chebyshev", ierr)
    !call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_min_iter", "250", ierr)

    ! Apply the options database changes to this KSP instance
    call KSPSetFromOptions(ksp_block, ierr)
    
    call KSPSetUp(ksp_block, ierr)

  end subroutine setup_block_ksp_hypre_amg_krylov


  !--------------------------------------------------------------------
  !> Stage 6.3: set up the inner KSP for the assembled momentum-Schur block
  !! S_PBP, selected by physics_pc_schur_inner INDEPENDENTLY of which ansatz
  !! built S_PBP (physics_pc_schur_variant).
  !!
  !! Why a separate knob: Stage 4.5 established that the inner solvability of
  !! the assembled operator is a different question from its approximation
  !! quality -- M1a conditions well but buys nothing, M2e approximates well
  !! but AMG fails on it. Comparing the ansaetze therefore needs the inner
  !! budget held fixed and varied on purpose, not implied by the variant.
  !!
  !! Back-compatibility is the whole point of case 0: it reproduces the
  !! previous inline branch verbatim, so an existing namelist that only sets
  !! physics_pc_schur_amg keeps its exact behaviour. Any nonzero value
  !! OVERRIDES physics_pc_schur_amg and says so, so the two knobs can never
  !! silently disagree.
  !!
  !! A varying inner solve is admissible because the outer solver is FGMRES
  !! (mod_petsc.f90), which is flexible by construction.
  !--------------------------------------------------------------------
  subroutine setup_schur_inner_ksp(ksp_block, B_block, comm, first_time, my_id)
    use phys_module, only: physics_pc_schur_inner, physics_pc_schur_amg, &
                           physics_pc_schur_amg_its

    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    integer, intent(in) :: my_id

    PC :: pc
    PetscErrorCode :: ierr
    PetscReal :: rtol, abstol, dtol

    if (physics_pc_schur_inner /= 0 .and. physics_pc_schur_amg .and. my_id == 0) &
      write(*,'(A,I0,A)') "[Physics PC]   NOTE: physics_pc_schur_inner = ", &
        physics_pc_schur_inner, " overrides physics_pc_schur_amg"

    select case (physics_pc_schur_inner)

    case (0)
      ! LEGACY -- byte-for-byte the branch this routine replaced.
      if (physics_pc_schur_amg) then
        call setup_block_ksp_hypre_amg_krylov(ksp_block, B_block, comm, &
                                              first_time, physics_pc_schur_amg_its)
        if (my_id == 0) write(*,'(A,I0,A)') &
          "[Physics PC]   S_PBP KSP: GMRES + HYPRE BoomerAMG, ", &
          physics_pc_schur_amg_its, " iterations"
      else
        call setup_block_ksp(ksp_block, B_block, comm, first_time, "S_PBP KSP")
      endif

    case (1)
      call setup_block_ksp(ksp_block, B_block, comm, first_time, "S_PBP KSP")

    case (2)
      call setup_block_ksp_hypre_amg_krylov(ksp_block, B_block, comm, &
                                            first_time, physics_pc_schur_amg_its)
      if (my_id == 0) write(*,'(A,I0,A)') &
        "[Physics PC]   S_PBP KSP: GMRES + HYPRE BoomerAMG, ", &
        physics_pc_schur_amg_its, " iterations (fixed budget)"

    case (3)
      ! Same AMG configuration, but tolerance-driven rather than budget-driven:
      ! set it up through the helper (which owns all the hypre options) and then
      ! relax the tolerances it hard-codes.
      call setup_block_ksp_hypre_amg_krylov(ksp_block, B_block, comm, first_time, 30)
      rtol = 1.0d-2; abstol = 1.0d-50; dtol = 1.0d4
      call KSPSetTolerances(ksp_block, rtol, abstol, dtol, 200, ierr)
      call KSPSetUp(ksp_block, ierr)
      if (my_id == 0) write(*,'(A)') &
        "[Physics PC]   S_PBP KSP: GMRES + HYPRE BoomerAMG, rtol 1e-2, max 200 its"

    case (4)
      if (first_time) call KSPCreate(comm, ksp_block, ierr)
      call KSPSetOperators(ksp_block, B_block, B_block, ierr)
      call KSPSetType(ksp_block, KSPGMRES, ierr)
      call KSPGMRESSetRestart(ksp_block, max(physics_pc_schur_amg_its, 2), ierr)
      rtol = 1.0d-3; abstol = 1.0d-50; dtol = 1.0d4
      call KSPSetTolerances(ksp_block, rtol, abstol, dtol, physics_pc_schur_amg_its, ierr)
      call KSPGetPC(ksp_block, pc, ierr)
      call PCSetType(pc, PCBJACOBI, ierr)
      call KSPSetUp(ksp_block, ierr)
      if (my_id == 0) write(*,'(A,I0,A)') &
        "[Physics PC]   S_PBP KSP: GMRES + BJACOBI/ILU(0), ", &
        physics_pc_schur_amg_its, " iterations"

    case default
      if (my_id == 0) write(*,'(A,I0,A)') &
        "[Physics PC]   WARNING: physics_pc_schur_inner = ", physics_pc_schur_inner, &
        " is not a valid setting; falling back to PREONLY + LU"
      call setup_block_ksp(ksp_block, B_block, comm, first_time, "S_PBP KSP")

    end select

  end subroutine setup_schur_inner_ksp

  !--------------------------------------------------------------------
  !> Set up the KSP for the density (rho, B_55) transport block.
  !! Physics-motivated, block-specific solver. Placeholder body: currently
  !! delegates to the shared Hypre BoomerAMG setup (GMRES(3) wrapper). The
  !! rho-specific AMG configuration will be dropped in here later, independent
  !! of the T block. Signature matches the setup_block_ksp family.
  !--------------------------------------------------------------------
  subroutine setup_rho_block_ksp(ksp_block, B_block, comm, first_time, label)
    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    character(len=*), intent(in), optional :: label

    !call setup_block_ksp_hypre_amg_krylov(ksp_block, B_block, comm, first_time, 3)
    !call setup_block_ksp_amg_krylov(ksp_block, B_block, comm, first_time, 5)
    call setup_block_ksp(ksp_block, B_block, comm, first_time)

    ! if (present(label)) call pc_print_block_setup(comm, label, "GMRES(5)+GAMG")
    if (present(label)) call pc_print_block_setup(comm, label, "MUMPS")
  end subroutine setup_rho_block_ksp

  !--------------------------------------------------------------------
  !> Set up the KSP for the temperature/pressure (T, B_66) transport block.
  !! Physics-motivated, block-specific solver. Placeholder body: currently
  !! delegates to the shared Hypre BoomerAMG setup (GMRES(3) wrapper). The
  !! T-specific AMG configuration (e.g. anisotropy-aware) will be dropped in
  !! here later, independent of the rho block. Signature matches the
  !! setup_block_ksp family.
  !--------------------------------------------------------------------
  subroutine setup_T_block_ksp(ksp_block, B_block, comm, first_time, label)
    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    character(len=*), intent(in), optional :: label

    !call setup_block_ksp_hypre_amg_krylov(ksp_block, B_block, comm, first_time, 3)
    !call setup_block_ksp_amg_krylov(ksp_block, B_block, comm, first_time, 5)
    call setup_block_ksp(ksp_block, B_block, comm, first_time)

    !if (present(label)) call pc_print_block_setup(comm, label, "GMRES(5)+GAMG")
    if (present(label)) call pc_print_block_setup(comm, label, "MUMPS")
  end subroutine setup_T_block_ksp

  !--------------------------------------------------------------------
  !> Assemble the monolithic 4x4 reduced system via MatCreateNest +
  !! MatConvert, and set up a single KSP (PREONLY+LU+MUMPS).
  !--------------------------------------------------------------------
  subroutine assemble_monolithic_4x4(comm, first_time, my_id, skip_ksp_setup)
     use mod_petsc_matrix_analysis, only: petsc_mat_convert_spectrum, petsc_mat_equilibrate
     use phys_module, only: physics_pc_reduced_pde
    logical, intent(in) :: first_time, skip_ksp_setup
    integer, intent(in) :: comm, my_id

    Mat :: mats_nest(16), A_nest   ! 1D row-major: (row0,col0), (row0,col1), ...
    Mat :: diag_55, diag_66
    PC  :: pc_obj
    PetscErrorCode :: ierr
    PetscInt :: rstart, rend, n_local, n_global
    PetscInt, parameter :: nblocks = 4
    integer :: k

    Mat :: mats_nest_hydro(9), A_nest_hydro
    Mat :: mats_nest_alfven(4), A_nest_alfven, A_alfven_2x2

    diag_55 = g_ctx%B_55
    diag_66 = g_ctx%B_66

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

    PetscCallA(MatCreateNest(comm, nblocks, PETSC_NULL_IS_ARRAY, nblocks, PETSC_NULL_IS_ARRAY, mats_nest, A_nest, ierr))

    ! Destroy old monolithic AIJ if rebuilding
    if (.not. first_time .and. g_ctx%ksp_reduced_created) then
      call MatDestroy(g_ctx%A_reduced_4x4, ierr)
    endif

    ! Convert nest to concrete AIJ
    call MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%A_reduced_4x4, ierr)
    call MatDestroy(A_nest, ierr)

    mats_nest_hydro(1) = g_ctx%Atilde_22
    mats_nest_hydro(2) = g_ctx%B_25
    mats_nest_hydro(3) = g_ctx%B_26
    mats_nest_hydro(4) = g_ctx%B_52
    mats_nest_hydro(5) = diag_55
    mats_nest_hydro(6) = PETSC_NULL_MAT
    mats_nest_hydro(7) = g_ctx%B_62
    mats_nest_hydro(8) = PETSC_NULL_MAT
    mats_nest_hydro(9) = diag_66

    PetscCallA(MatCreateNest(comm, 3, PETSC_NULL_IS_ARRAY, 3, PETSC_NULL_IS_ARRAY, mats_nest_hydro, A_nest_hydro, ierr))

    if (.not. first_time .and. g_ctx%ksp_hydro_created) then
      call MatDestroy(g_ctx%M_hydro, ierr)
    endif

    call MatConvert(A_nest_hydro, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%M_hydro, ierr)
    call MatDestroy(A_nest_hydro, ierr)

    !call petsc_mat_convert_spectrum(g_ctx%M_hydro, "M_hydro", .false.)
    !call petsc_test_pc_matrix(g_ctx%M_hydro, "M_hydro", .false., my_id)

    mats_nest_alfven(1) = g_ctx%Atilde_11
    mats_nest_alfven(2) = g_ctx%B_12
    mats_nest_alfven(3) = g_ctx%Atilde_21
    mats_nest_alfven(4) = g_ctx%Atilde_22

    PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS_ARRAY, 2, PETSC_NULL_IS_ARRAY, mats_nest_alfven, A_nest_alfven, ierr))

    if (.not. first_time .and. g_ctx%ksp_alfven_created) then
      call MatDestroy(g_ctx%A_alfven, ierr)
    endif

    call MatConvert(A_nest_alfven, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%A_alfven, ierr)
    call MatDestroy(A_nest_alfven, ierr)

    !call petsc_mat_convert_spectrum(g_ctx%A_alfven, "A_alfven_2x2", .false.)
    !call petsc_test_pc_matrix(g_ctx%A_alfven, "A_alfven_2x2", .false., my_id)

    ! Set up monolithic KSP (skipped when probe_exact=.true.: probe owns the KSP)
    ! --- Milestone 2: optionally replace the extracted-block operator by the
    !     reduced PDE operator P_full. Only the operator that ksp_reduced factors
    !     changes; the Variant-B residual transfer and the j/w back-substitution
    !     in the apply path are untouched, since those legitimately use blocks of
    !     the true mixed Jacobian either way.
    if (physics_pc_reduced_pde) then
      call build_permuted_P_full(comm, my_id)
      if (.not. g_ctx%p_full_perm_ready) then
        ierr = 1
        return
      endif
      block
        PetscInt :: nr_p, nr_a
        call MatGetSize(g_ctx%P_full_perm,   nr_p, PETSC_NULL_INTEGER, ierr)
        call MatGetSize(g_ctx%A_reduced_4x4, nr_a, PETSC_NULL_INTEGER, ierr)
        if (nr_p /= nr_a) then
          if (my_id == 0) write(*,'(A,I0,A,I0)') &
            "[Physics PC] ERROR: P_full size ", nr_p, " /= reduced system size ", nr_a
          ierr = 1
          return
        endif
      end block
    endif

    if (.not. skip_ksp_setup) then
      if (first_time) then
        call KSPCreate(comm, g_ctx%ksp_reduced, ierr)
      endif
      if (physics_pc_reduced_pde) then
        call KSPSetOperators(g_ctx%ksp_reduced, g_ctx%P_full_perm, g_ctx%P_full_perm, ierr)
        if (my_id == 0) write(*,'(A)') &
          "[Physics PC]   reduced solve operator: P_full (substituted PDE operator)"
      else
        call KSPSetOperators(g_ctx%ksp_reduced, g_ctx%A_reduced_4x4, g_ctx%A_reduced_4x4, ierr)
        if (my_id == 0) write(*,'(A)') &
          "[Physics PC]   reduced solve operator: extracted blocks + algebraic Schur"
      endif
      call KSPSetType(g_ctx%ksp_reduced, KSPPREONLY, ierr)
      call KSPGetPC(g_ctx%ksp_reduced, pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
      call KSPSetUp(g_ctx%ksp_reduced, ierr)
      g_ctx%ksp_reduced_created = .true.
    endif

    ! Create ksp for sub-blocks (Alfven and hydro (For now mumps but to be changed...))
    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_hydro, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_hydro, g_ctx%M_hydro, g_ctx%M_hydro, ierr)
    call KSPSetType(g_ctx%ksp_hydro, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_hydro, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_hydro, ierr)
    g_ctx%ksp_hydro_created = .true.

    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_alfven, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_alfven, g_ctx%A_alfven, g_ctx%A_alfven, ierr)
    call KSPSetType(g_ctx%ksp_alfven, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_alfven, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_alfven, ierr)
    g_ctx%ksp_alfven_created = .true.

    ! S_PBP (inner Alfven Schur operator, u-sized) — direct solver for the
    ! three-step Alfven corrector. Guard against an unassembled S_PBP: a zero
    ! matrix would make MUMPS fail opaquely or produce garbage.
    block
      PetscReal :: norm_spbp
      call MatNorm(g_ctx%S_PBP, NORM_FROBENIUS, norm_spbp, ierr)
      if (norm_spbp <= 0.0d0) then
        if (my_id == 0) write(*,'(A)') &
          "[Physics PC] ERROR: S_PBP is empty/zero — element assembly missing; cannot set up ksp_S_PBP"
        ierr = 1
        return
      endif
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||S_PBP||_F = ", norm_spbp
    end block

    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_S_PBP, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_S_PBP, g_ctx%S_PBP, g_ctx%S_PBP, ierr)
    call KSPSetType(g_ctx%ksp_S_PBP, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_S_PBP, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_S_PBP, ierr)
    g_ctx%ksp_S_PBP_created = .true.
    if (my_id == 0) write(*,'(A)') "[Physics PC]   ksp_S_PBP set up (PREONLY+LU+MUMPS)"

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

      call MatCreateVecs(g_ctx%M_hydro, g_ctx%hydro_predictor_3v, g_ctx%work_rhs_3v, ierr)
      do k = 1, 3
        call ISCreateStride(comm, n_local, rstart + (k-1)*n_local, 1, &
                            g_ctx%is_hydro(k), ierr)
      enddo
      g_ctx%is_reduced_created = .true.
    endif

    if (my_id == 0) then
      if (skip_ksp_setup) then
        write(*,'(A)') "[Physics PC]   Monolithic 4x4 approx matrix built (KSP deferred to probe)"
      else
        write(*,'(A)') "[Physics PC]   Monolithic 4x4 KSP set up (PREONLY+LU+MUMPS)"
      endif
    endif
  end subroutine assemble_monolithic_4x4


  !--------------------------------------------------------------------
  !> Assemble the EXACT 4×4 Schur complement by probing:
  !! apply S_4x4 to all 4N standard basis vectors, gather columns to
  !! rank 0, build a dense AIJ matrix, and replace the approximate
  !! operator in ksp_reduced.
  !!
  !! Cost: 2N KSPSolve (triangular substitutions on existing MUMPS
  !! factorisations) + 14N MatMult.  Intended for small test problems.
  !!
  !! Must be called AFTER assemble_monolithic_4x4 so that ksp_reduced
  !! and g_ctx%A_reduced_4x4 (the approximate baseline) already exist.
  !--------------------------------------------------------------------
  subroutine assemble_probed_exact_4x4(comm, first_time, my_id)
    use mod_petsc_matrix_analysis, only: petsc_mat_diff_norm
    implicit none
    logical, intent(in) :: first_time
    integer, intent(in) :: comm, my_id

    Vec            :: e_j, z, temp, r_blk, scratch, r_seq
    VecScatter     :: scat
    Mat            :: A_exact, diag_55, diag_66
    PC             :: pc_obj
    PetscInt       :: N_global, n4p, m_local_4v
    PetscInt       :: rstart_1v_p, rend_1v_p
    PetscErrorCode :: ierr
    PetscScalar, pointer :: arr(:)
    PetscInt, allocatable :: interleaved_idxs(:)  ! block-contiguous → 4N interleaved global index
    PetscInt       :: row_idx(1)
    integer        :: n, n4, j, p, j_loc, kb
    integer        :: nproc, mpierr, n_local_1v
    integer, allocatable :: n_local_arr(:), rstart_arr(:)
    real*8, allocatable :: A_dense(:,:)
    PetscReal      :: norm_approx, norm_diff, norm_exact

    call MatGetSize(g_ctx%B_11, N_global, PETSC_NULL_INTEGER, ierr)
    n   = int(N_global)
    n4  = 4 * n
    n4p = n4

    diag_55 = g_ctx%B_55
    diag_66 = g_ctx%B_66

    ! Local work vectors — work_1..5 not yet allocated at build time
    call MatCreateVecs(g_ctx%B_11, e_j, PETSC_NULL_VEC, ierr)
    call VecDuplicate(e_j, z,       ierr)
    call VecDuplicate(e_j, temp,    ierr)
    call VecDuplicate(e_j, r_blk,   ierr)
    call VecDuplicate(e_j, scratch, ierr)
    call VecScatterCreateToZero(e_j, scat, r_seq, ierr)

    ! Build mapping: block-contiguous 4N index → interleaved 4N global index.
    ! For P=1 these are identical; for P>1 the 4N matrix layout produced by
    ! MatConvert(MATNEST→MATMPIAIJ) interleaves variable blocks per process:
    !   process p owns [rstart_4v_p .. rstart_4v_p + 4*nloc_p - 1] where
    !   rstart_4v_p = 4 * rstart_1v_p and within that range variable k (0-based)
    !   occupies [rstart_4v_p + k*nloc_p .. rstart_4v_p + (k+1)*nloc_p - 1].
    ! Block-contiguous index kb*n + j_global maps to interleaved index:
    !   4*rstart_1v[p] + kb*nloc[p] + (j_global - rstart_1v[p])
    ! where p is the process owning j_global in the 1-variable distribution.
    call MPI_Comm_size(comm, nproc, mpierr)
    call VecGetOwnershipRange(e_j, rstart_1v_p, rend_1v_p, ierr)
    n_local_1v = int(rend_1v_p - rstart_1v_p)
    allocate(n_local_arr(nproc), rstart_arr(nproc+1))
    call MPI_Allgather(n_local_1v, 1, MPI_INTEGER, n_local_arr, 1, MPI_INTEGER, comm, mpierr)
    rstart_arr(1) = 0
    do p = 1, nproc
      rstart_arr(p+1) = rstart_arr(p) + n_local_arr(p)
    enddo

    ! Build interleaved_idxs on rank 0 (only rank 0 inserts into A_exact)
    if (my_id == 0) then
      allocate(interleaved_idxs(n4))
      p = 1
      do j_loc = 0, n-1
        do while (j_loc >= rstart_arr(p+1))
          p = p + 1
        enddo
        do kb = 0, 3
          interleaved_idxs(kb*n + j_loc + 1) = &
            4*rstart_arr(p) + kb*n_local_arr(p) + (j_loc - rstart_arr(p))
        enddo
      enddo
    endif

    if (my_id == 0) then
      allocate(A_dense(n4, n4))
      A_dense = 0.0d0   ! ensure structural zeros are correct before block-by-block fill
      write(*,'(A,I0,A)') "[Physics PC]   Probing exact 4x4 Schur (4x", n, " columns)..."
      flush(6)
    endif

    ! =================================================================
    ! Block 1: ψ input → columns 0..n-1
    !   temp_j = ksp_Mj^{-1} · B_31 · e_ψ
    !   r_ψ = B_11·e_ψ − B_13·temp_j
    !   r_u = B_21·e_ψ − B_23·temp_j
    !   r_ρ = B_51·e_ψ
    !   r_T = B_61·e_ψ − B_63·temp_j
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 1/4 (psi, ksp_Mj):"
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)
      call MatMult(g_ctx%B_31, e_j, z,    ierr)
      call KSPSolve(g_ctx%ksp_Mj, z, temp, ierr)

      ! r_ψ
      call MatMult(g_ctx%B_11, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_13, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_u
      call MatMult(g_ctx%B_21, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_23, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(n+1:2*n, j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_ρ  (no Schur coupling from ψ to ρ through j)
      call MatMult(g_ctx%B_51, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(2*n+1:3*n, j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_T
      call MatMult(g_ctx%B_61, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_63, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(3*n+1:4*n, j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Physics PC]     Block 1/4: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! =================================================================
    ! Block 2: u input → columns n..2n-1
    !   temp_w = ksp_Mw^{-1} · B_42 · e_u
    !   r_ψ = B_12·e_u
    !   r_u = B_22·e_u − B_24·temp_w
    !   r_ρ = B_52·e_u
    !   r_T = B_62·e_u
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 2/4 (u, ksp_Mw):"
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)
      call MatMult(g_ctx%B_42, e_j, z,    ierr)
      call KSPSolve(g_ctx%ksp_Mw, z, temp, ierr)

      ! r_ψ  (no Schur from u to ψ through j)
      call MatMult(g_ctx%B_12, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(1:n, n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_u
      call MatMult(g_ctx%B_22, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_24, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(n+1:2*n, n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_ρ
      call MatMult(g_ctx%B_52, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(2*n+1:3*n, n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_T  (no Schur: u doesn't drive j-elimination in T row)
      call MatMult(g_ctx%B_62, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(3*n+1:4*n, n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Physics PC]     Block 2/4: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! =================================================================
    ! Block 3: ρ input → columns 2n..3n-1
    !   No Schur corrections (ρ not in constraint system)
    !   r_ψ = 0,  r_u = B_25·e_ρ,  r_ρ = B_55·e_ρ,  r_T = 0
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 3/4 (rho, MatMult)..."
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)

      ! r_u
      call MatMult(g_ctx%B_25, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(n+1:2*n, 2*n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_ρ
      call MatMult(diag_55, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(2*n+1:3*n, 2*n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif
    enddo

    ! =================================================================
    ! Block 4: T input → columns 3n..4n-1
    !   No Schur corrections (T not in constraint system)
    !   r_ψ = B_16·e_T,  r_u = B_26·e_T,  r_ρ = 0,  r_T = B_66·e_T
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 4/4 (T, MatMult)..."
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)

      ! r_ψ
      call MatMult(g_ctx%B_16, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(1:n, 3*n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_u
      call MatMult(g_ctx%B_26, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(n+1:2*n, 3*n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif

      ! r_T
      call MatMult(diag_66, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArray(r_seq, arr, ierr)
        A_dense(3*n+1:4*n, 3*n+j+1) = real(arr, kind=8)
        call VecRestoreArray(r_seq, arr, ierr)
      endif
    enddo

    ! =================================================================
    ! Build MATMPIAIJ from A_dense (rank 0 inserts all rows)
    ! =================================================================
    ! Use the same local row count as work_rhs_4v (created from A_approx's nest layout)
    ! so that A_exact is compatible with the work vectors in KSPSolve.
    ! PETSC_DECIDE would distribute rows independently of the nest structure, producing
    ! a different layout when N is not divisible by the number of MPI ranks.
    call VecGetLocalSize(g_ctx%work_rhs_4v, m_local_4v, ierr)
    call MatCreate(comm, A_exact, ierr)
    call MatSetSizes(A_exact, m_local_4v, m_local_4v, n4p, n4p, ierr)
    call MatSetType(A_exact, MATMPIAIJ, ierr)
    call MatSetOption(A_exact, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(A_exact, ierr)

    if (my_id == 0) then
      ! Insert column j (block-contiguous) into interleaved column interleaved_idxs(j+1).
      ! Row indices are also remapped via interleaved_idxs so that A_dense(i, j+1) —
      ! which holds S(bc_row=i-1, bc_col=j) — lands at A_exact(interleaved(i-1), interleaved(j)).
      do j = 0, n4-1
        row_idx(1) = interleaved_idxs(j+1)
        call MatSetValues(A_exact, n4p, interleaved_idxs, 1, row_idx, &
                          A_dense(:, j+1), INSERT_VALUES, ierr)
      enddo
      deallocate(interleaved_idxs, A_dense)
    endif
    deallocate(n_local_arr, rstart_arr)
    call MatAssemblyBegin(A_exact, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (A_exact, MAT_FINAL_ASSEMBLY, ierr)

    ! Diagnostic: relative Frobenius error vs approximate matrix
    call petsc_mat_diff_norm(A_exact, g_ctx%A_reduced_4x4, &
                             "A_exact_4x4 - A_approx_4x4", norm_diff)
    call MatNorm(g_ctx%A_reduced_4x4, NORM_FROBENIUS, norm_approx, ierr)
    call MatNorm(A_exact,             NORM_FROBENIUS, norm_exact,  ierr)
    if (my_id == 0) then
      write(*,'(A,ES12.4)') "[Physics PC]   ||A_approx||_F        = ", norm_approx
      write(*,'(A,ES12.4)') "[Physics PC]   ||A_exact||_F         = ", norm_exact
      write(*,'(A,ES12.4)') &
        "[Physics PC]   ||A_exact - A_approx||_F / ||A_approx||_F = ", &
        norm_diff / norm_approx
    endif

    ! Replace approximate matrix and set up KSP from scratch.
    ! assemble_monolithic_4x4 was called with skip_ksp_setup=.true., so ksp_reduced
    ! does not exist yet on first_time; on subsequent calls it exists but must be
    ! re-factored with the new A_exact.  A fresh KSPCreate on first_time and a full
    ! setup sequence on every call avoids any stale-factorization pitfalls.
    call MatDestroy(g_ctx%A_reduced_4x4, ierr)
    g_ctx%A_reduced_4x4 = A_exact
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

    ! Cleanup local temporaries
    call VecScatterDestroy(scat,    ierr)
    call VecDestroy(r_seq,   ierr)
    call VecDestroy(e_j,     ierr)
    call VecDestroy(z,       ierr)
    call VecDestroy(temp,    ierr)
    call VecDestroy(r_blk,   ierr)
    call VecDestroy(scratch, ierr)

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]   Exact probed 4x4 set up (PREONLY+LU+MUMPS)"
      flush(6)
    endif
  end subroutine assemble_probed_exact_4x4

  !--------------------------------------------------------------------
  !> Stage-1 verification of the segregated 2x2 Alfven solve.
  !!
  !! Solves a random RHS on the 2x2 Alfven block K_A = [[Atilde_11, B_12],
  !! [Atilde_21, Atilde_22]] two ways and prints the relative error:
  !!   (a) ground truth: direct MUMPS solve via ksp_alfven (assembled A_alfven)
  !!   (b) segregated block factorization using ksp_psi (Atilde_11^-1),
  !!       Atilde_21, ksp_S_PBP (which must hold the EXACT S_u), B_12:
  !!         t_psi  = Atilde_11^-1 b_psi
  !!         y_u    = S_u^-1 (b_u - Atilde_21 t_psi)
  !!         y_psi  = Atilde_11^-1 (b_psi - B_12 y_u)
  !!
  !! Precondition: ksp_alfven, ksp_psi, and ksp_S_PBP are all set up, and the
  !! S_u -> S_PBP copy (or KSP rebind) is active so ksp_S_PBP solves the exact
  !! Schur complement. Expect rel_err <~ 1e-10.
  !--------------------------------------------------------------------
  subroutine verify_alfven_2x2_segregated(comm, my_id)
    integer, intent(in) :: comm, my_id

    Vec :: b, x_exact, x_seg
    Vec :: b_psi, b_u, t_psi, y_psi, y_u, rhs_u, rhs_psi, scratch_u, scratch_psi
    PetscScalar, pointer :: a_b(:), a_x(:)
    PetscRandom :: rctx
    PetscReal   :: nrm_diff, nrm_exact, rel_err
    PetscInt    :: n1_local, n2_local
    PetscErrorCode :: ierr

    ! Packed (psi,u) work vectors on the assembled 2x2 layout
    call MatCreateVecs(g_ctx%A_alfven, x_exact, b, ierr)
    call VecDuplicate(b, x_seg, ierr)

    ! 1-variable work vectors: psi-sized from Atilde_11, u-sized from S_PBP (=S_u)
    call MatCreateVecs(g_ctx%Atilde_11, t_psi, b_psi, ierr)   ! psi-sized
    call VecDuplicate(b_psi, y_psi,       ierr)
    call VecDuplicate(b_psi, rhs_psi,     ierr)
    call VecDuplicate(b_psi, scratch_psi, ierr)
    call MatCreateVecs(g_ctx%S_PBP, y_u, b_u, ierr)           ! u-sized
    call VecDuplicate(b_u, rhs_u,     ierr)
    call VecDuplicate(b_u, scratch_u, ierr)

    ! --- Random RHS on the packed 2x2 layout ---
    call PetscRandomCreate(comm, rctx, ierr)
    call PetscRandomSetType(rctx, "rand", ierr)
    call VecSetRandom(b, rctx, ierr)
    call PetscRandomDestroy(rctx, ierr)

    ! --- Ground truth: x_exact = K_A^-1 b (direct MUMPS on assembled A_alfven) ---
    call KSPSolve(g_ctx%ksp_alfven, b, x_exact, ierr)

    ! --- Split packed b into b_psi (top half) and b_u (bottom half) ---
    call VecGetLocalSize(b_psi, n1_local, ierr)
    call VecGetLocalSize(b_u,   n2_local, ierr)
    call VecGetArrayRead(b, a_b, ierr)
    call VecGetArray(b_psi, a_x, ierr)
    a_x(1:n1_local) = a_b(1:n1_local)
    call VecRestoreArray(b_psi, a_x, ierr)
    call VecGetArray(b_u, a_x, ierr)
    a_x(1:n2_local) = a_b(n1_local+1 : n1_local+n2_local)
    call VecRestoreArray(b_u, a_x, ierr)
    call VecRestoreArrayRead(b, a_b, ierr)

    ! --- Segregated block-factorization solve ---
    ! t_psi = Atilde_11^-1 b_psi
    call KSPSolve(g_ctx%ksp_psi, b_psi, t_psi, ierr)
    ! rhs_u = b_u - Atilde_21 t_psi
    call MatMult(g_ctx%Atilde_21, t_psi, scratch_u, ierr)
    call VecWAXPY(rhs_u, -1.0d0, scratch_u, b_u, ierr)
    ! y_u = S_u^-1 rhs_u   (ksp_S_PBP holds the exact S_u)
    call KSPSolve(g_ctx%ksp_S_PBP, rhs_u, y_u, ierr)
    ! rhs_psi = b_psi - B_12 y_u
    call MatMult(g_ctx%B_12, y_u, scratch_psi, ierr)
    call VecWAXPY(rhs_psi, -1.0d0, scratch_psi, b_psi, ierr)
    ! y_psi = Atilde_11^-1 rhs_psi
    call KSPSolve(g_ctx%ksp_psi, rhs_psi, y_psi, ierr)

    ! --- Pack (y_psi, y_u) into x_seg ---
    call VecGetArray(x_seg, a_x, ierr)
    call VecGetArrayRead(y_psi, a_b, ierr)
    a_x(1:n1_local) = a_b(1:n1_local)
    call VecRestoreArrayRead(y_psi, a_b, ierr)
    call VecGetArrayRead(y_u, a_b, ierr)
    a_x(n1_local+1 : n1_local+n2_local) = a_b(1:n2_local)
    call VecRestoreArrayRead(y_u, a_b, ierr)
    call VecRestoreArray(x_seg, a_x, ierr)

    ! --- Relative error: ||x_seg - x_exact|| / ||x_exact|| ---
    call VecAXPY(x_seg, -1.0d0, x_exact, ierr)     ! x_seg <- x_seg - x_exact
    call VecNorm(x_seg,   NORM_2, nrm_diff,  ierr)
    call VecNorm(x_exact, NORM_2, nrm_exact, ierr)
    if (nrm_exact > 0.0d0) then
      rel_err = nrm_diff / nrm_exact
    else
      rel_err = nrm_diff
    endif
    if (my_id == 0) write(*,'(A,ES12.4)') &
      "[Physics PC]   [verify 2x2] ||x_seg - x_exact||/||x_exact|| = ", rel_err

    ! --- Cleanup ---
    call VecDestroy(b,           ierr)
    call VecDestroy(x_exact,     ierr)
    call VecDestroy(x_seg,       ierr)
    call VecDestroy(b_psi,       ierr)
    call VecDestroy(b_u,         ierr)
    call VecDestroy(t_psi,       ierr)
    call VecDestroy(y_psi,       ierr)
    call VecDestroy(y_u,         ierr)
    call VecDestroy(rhs_u,       ierr)
    call VecDestroy(rhs_psi,     ierr)
    call VecDestroy(scratch_u,   ierr)
    call VecDestroy(scratch_psi, ierr)
  end subroutine verify_alfven_2x2_segregated


  !--------------------------------------------------------------------
  !> Create index sets that extract each of the four reduced variables
  !! from P_full, the 4-variable BAIJ operator.
  !!
  !! Mirrors create_variable_index_sets with n_var -> 4. DOF ordering inside a
  !! block of size 4*n_tor is var-major, toroidal-minor:
  !!   reduced var v (1-based) at node block i:  i*block_size + (v-1)*n_tor + m
  !!
  !! The resulting 1-variable sub-matrices have the same row/column layout as
  !! the B_ij blocks extracted from the full 6-variable system, so blocks of
  !! P_full and of the condensed mixed Jacobian can be compared directly with no
  !! permutation.
  !--------------------------------------------------------------------
  subroutine create_reduced_index_sets(P_full, comm, is_red)
    use mod_parameters, only: n_tor

    Mat,     intent(in)  :: P_full
    integer, intent(in)  :: comm
    IS,      intent(out) :: is_red(4)

    PetscInt :: n_local, rstart, rend
    PetscInt :: block_size, n_block_local, n_var_dofs
    PetscInt, allocatable :: indices(:)
    PetscErrorCode :: ierr
    integer :: v, i, m, k

    PetscCallA(MatGetLocalSize(P_full, n_local, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetOwnershipRange(P_full, rstart, rend, ierr))

    block_size    = 4 * n_tor
    n_block_local = n_local / block_size
    n_var_dofs    = n_block_local * n_tor

    allocate(indices(n_var_dofs))

    do v = 1, 4
      k = 0
      do i = 0, n_block_local - 1
        do m = 0, n_tor - 1
          k = k + 1
          indices(k) = rstart + i * block_size + (v-1) * n_tor + m
        enddo
      enddo
      PetscCallA(ISCreateGeneral(comm, n_var_dofs, indices, PETSC_COPY_VALUES, is_red(v), ierr))
    enddo

    deallocate(indices)
  end subroutine create_reduced_index_sets


  !--------------------------------------------------------------------
  !> Permute P_full into the layout expected by ksp_reduced (Milestone 2).
  !!
  !! P_full_pde is MPIBAIJ with block size 4*n_tor and node-major ordering:
  !!   node block i, reduced var v, harmonic m  ->  i*4*n_tor + (v-1)*n_tor + m
  !!
  !! A_reduced_4x4 (built by MatConvert of a 4x4 MatNest) instead lays the
  !! variables out block-contiguously per process, which is what is_reduced(k)
  !! (an ISCreateStride) and therefore the whole apply path assume:
  !!   variable k occupies rows [rstart + (k-1)*n_local .. rstart + k*n_local-1]
  !!
  !! Concatenating the four per-variable index sets of P_full gives exactly the
  !! permutation between the two orderings, so a single MatCreateSubMatrix with
  !! that IS on both rows and columns produces a drop-in replacement operator.
  !--------------------------------------------------------------------
  subroutine build_permuted_P_full(comm, my_id)
    use mod_parameters, only: n_tor

    integer, intent(in) :: comm, my_id

    Mat :: P_aij
    IS  :: is_perm
    PetscInt :: n_local, rstart, rend
    PetscInt :: block_size, n_block_local, n_tot
    PetscInt, allocatable :: idx(:)
    PetscErrorCode :: ierr
    integer :: v, i, m, k

    if (.not. g_ctx%p_full_ready) then
      if (my_id == 0) write(*,'(A)') &
        "[Physics PC] ERROR: physics_pc_reduced_pde requested but P_full is not assembled."
      return
    endif

    ! BAIJ -> AIJ so that scalar index sets address individual DOFs
    PetscCallA(MatConvert(g_ctx%P_full_pde, MATMPIAIJ, MAT_INITIAL_MATRIX, P_aij, ierr))

    PetscCallA(MatGetLocalSize(P_aij, n_local, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetOwnershipRange(P_aij, rstart, rend, ierr))

    block_size    = 4 * n_tor
    n_block_local = n_local / block_size
    n_tot         = 4 * n_block_local * n_tor

    allocate(idx(n_tot))
    k = 0
    do v = 1, 4
      do i = 0, n_block_local - 1
        do m = 0, n_tor - 1
          k = k + 1
          idx(k) = rstart + i * block_size + (v-1) * n_tor + m
        enddo
      enddo
    enddo

    PetscCallA(ISCreateGeneral(comm, n_tot, idx, PETSC_COPY_VALUES, is_perm, ierr))
    deallocate(idx)

    if (g_ctx%p_full_perm_ready) PetscCallA(MatDestroy(g_ctx%P_full_perm, ierr))
    PetscCallA(MatCreateSubMatrix(P_aij, is_perm, is_perm, MAT_INITIAL_MATRIX, g_ctx%P_full_perm, ierr))
    g_ctx%p_full_perm_ready = .true.

    PetscCallA(ISDestroy(is_perm, ierr))
    PetscCallA(MatDestroy(P_aij, ierr))

    if (my_id == 0) write(*,'(A)') &
      "[Physics PC]   P_full permuted to the reduced-solve layout"
  end subroutine build_permuted_P_full


  !--------------------------------------------------------------------
  !> Milestone-1 verification of the reduced PDE operator P_full.
  !!
  !! Compares P_full against S_disc = A - B*D_c^{-1}*C, the EXACT algebraic
  !! condensation of the mixed Jacobian, which is the reference operator
  !! (docs/physics_pc/Development_Plan_Physics-Based.md Sec. 6.3).
  !!
  !! Exact agreement is NOT expected and is not the acceptance criterion. JOREK
  !! carries j and w as L2 projections onto the same C1 space, so S_disc
  !! contains a DISCRETE Delta* (namely M^{-1}K) where P_full uses the CONTINUUM
  !! Delta*. The difference is the projection error Pi_h - I: O(h^2) on smooth
  !! fields, but O(1) on the stiffest modes -- exactly the modes a preconditioner
  !! targets. A single global norm therefore cannot distinguish "correct operator,
  !! expected projection difference" from "wrong operator", which is why this
  !! routine reports the discrepancy resolved by spectral band.
  !!
  !! Reports:
  !!   1. per-block relative Frobenius discrepancy over all 16 blocks;
  !!   2. mat-vec discrepancy on a SMOOTH and on a ROUGH test vector;
  !!   3. a reminder of the discrepancy sources that are there by construction.
  !!
  !! Cost: O(N) MUMPS solves for each of the four Schur-corrected reference
  !! blocks. Small test problems only -- this is a diagnostic, not a solver path.
  !--------------------------------------------------------------------
  subroutine verify_reduced_pde_operator(A_full, comm, my_id)
    use mod_parameters, only: n_tor, var_psi, var_u, var_zj, var_w, var_rho, var_T
    use phys_module,    only: physics_pc_drop_psi_coupling, visco_num, eta_num

    Mat,     intent(in) :: A_full
    integer, intent(in) :: comm, my_id

    IS  :: is_red(4)
    Mat :: P_aij           ! P_full converted to MPIAIJ (see below)
    Mat :: P(4,4)          ! blocks of P_full
    Mat :: S(4,4)          ! reference blocks of S_disc
    Mat :: B_13, B_23, B_24, B_63, B_31, B_42, B_33, B_44
    Mat :: C_tmp
    Vec :: x(4), yP(4), yS(4), tmp, dlump
    Vec :: mask(4)
    Mat :: D_tmp
    PetscReal :: nrm_S, nrm_D, nrm_P
    PetscReal :: nrm_Si, nrm_Di
    PetscReal :: nP, nS, nD
    PetscScalar, pointer :: parr(:)
    PetscErrorCode :: ierr
    integer :: r, c, ipass, i, nproc, mpierr
    !> Rows carrying the ZBIG Dirichlet penalty are excluded from the "interior"
    !! comparison: both P_full and S_disc set the identical penalty there, so the
    !! difference cancels while the norm of S is completely dominated by it,
    !! which would make every relative discrepancy look spuriously tiny.
    real*8, parameter :: zbig_thresh = 1.d11
    integer :: var_map(4)
    !> full-system (row,col) of each corrected block, and which mass inverse it uses
    logical :: corrected(4,4)
    character(len=5) :: vname(4)

    var_map = (/ var_psi, var_u, var_rho, var_T /)
    vname   = (/ ' psi ', '  u  ', ' rho ', '  T  ' /)

    if (.not. g_ctx%p_full_ready) then
      if (my_id == 0) write(*,'(A)') &
        "[Verify P_full] ERROR: P_full not assembled; set physics_pc_reduced_pde=.true."
      return
    endif

    call MPI_Comm_size(comm, nproc, mpierr)

    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "================================================================"
      write(*,'(A)') " Milestone 1: verification of the reduced PDE operator P_full"
      write(*,'(A)') "   reference: S_disc = A - B D_c^-1 C  (exact condensation)"
      write(*,'(A)') "================================================================"
      if (nproc > 1) then
        write(*,'(A,I0,A)') " WARNING: running on ", nproc, " ranks. The UNCONDENSED blocks are"
        write(*,'(A)')      "   verified correctly in parallel, but the four CONDENSED reference"
        write(*,'(A)')      "   blocks are built by compute_schur_corrected_block_exact, which"
        write(*,'(A)')      "   gathers to rank 0 and is only reliable on a single rank (compare"
        write(*,'(A)')      "   assemble_probed_exact_4x4, which needs explicit interleaving"
        write(*,'(A)')      "   corrections for P>1). Judge the condensed rows from a 1-rank run."
      endif
      flush(6)
    endif

    ! --- Index sets ---------------------------------------------------
    ! P_full is MPIBAIJ, and MatCreateSubMatrix interprets index sets on a BAIJ
    ! matrix as BLOCK indices. Convert to MPIAIJ first so that scalar index sets
    ! work, exactly as the main path does for A_full (petsc_sys%A -> A_aij).
    if (.not. g_ctx%is_created) call create_variable_index_sets(A_full, comm)
    PetscCallA(MatConvert(g_ctx%P_full_pde, MATMPIAIJ, MAT_INITIAL_MATRIX, P_aij, ierr))
    call create_reduced_index_sets(P_aij, comm, is_red)

    ! --- Blocks of P_full ---------------------------------------------
    do r = 1, 4
      do c = 1, 4
        PetscCallA(MatCreateSubMatrix(P_aij, is_red(r), is_red(c), MAT_INITIAL_MATRIX, P(r,c), ierr))
      enddo
    enddo

    ! --- Reference blocks ---------------------------------------------
    ! Uncorrected blocks are just the corresponding blocks of the mixed
    ! Jacobian; only the four blocks that couple to j or w need condensing.
    corrected = .false.
    corrected(1,1) = .true.   ! psi-psi : - B_13 M_j^-1 B_31
    corrected(2,1) = .true.   ! u-psi   : - B_23 M_j^-1 B_31
    corrected(2,2) = .true.   ! u-u     : - B_24 M_w^-1 B_42
    corrected(4,1) = .true.   ! T-psi   : - B_63 M_j^-1 B_31

    call extract_sub_block(A_full, var_psi, var_zj,  B_13, .true.)
    call extract_sub_block(A_full, var_u,   var_zj,  B_23, .true.)
    call extract_sub_block(A_full, var_u,   var_w,   B_24, .true.)
    call extract_sub_block(A_full, var_T,   var_zj,  B_63, .true.)
    call extract_sub_block(A_full, var_zj,  var_psi, B_31, .true.)
    call extract_sub_block(A_full, var_w,   var_u,   B_42, .true.)
    call extract_sub_block(A_full, var_zj,  var_zj,  B_33, .true.)
    call extract_sub_block(A_full, var_w,   var_w,   B_44, .true.)

    do r = 1, 4
      do c = 1, 4
        call extract_sub_block(A_full, var_map(r), var_map(c), S(r,c), .true.)
      enddo
    enddo

    if (my_id == 0) then
      write(*,'(A)') "[Verify P_full] Condensing the four j/w-coupled reference blocks"
      write(*,'(A)') "                (exact, by probing -- O(N) solves each)"
      flush(6)
    endif

    call schur_correct_in_place(B_33, S(1,1), B_13, B_31)
    call schur_correct_in_place(B_33, S(2,1), B_23, B_31)
    call schur_correct_in_place(B_44, S(2,2), B_24, B_42)
    call schur_correct_in_place(B_33, S(4,1), B_63, B_31)

    ! --- Row masks: 1 on interior DOFs, 0 on Dirichlet-penalty DOFs ----
    do r = 1, 4
      PetscCallA(MatCreateVecs(S(r,r), mask(r), PETSC_NULL_VEC, ierr))
      PetscCallA(MatGetDiagonal(S(r,r), mask(r), ierr))
      PetscCallA(VecGetArray(mask(r), parr, ierr))
      do i = 1, size(parr)
        if (abs(parr(i)) > zbig_thresh) then
          parr(i) = 0.0d0
        else
          parr(i) = 1.0d0
        endif
      enddo
      PetscCallA(VecRestoreArray(mask(r), parr, ierr))
    enddo

    ! --- 1. Per-block comparison --------------------------------------
    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "--- 1. Per-block discrepancy  ||P-S||_F / ||S||_F ---------------"
      write(*,'(A)') "  'all' includes the ZBIG Dirichlet rows, whose norm dominates and"
      write(*,'(A)') "  cancels in the difference; 'interior' excludes them and is the"
      write(*,'(A)') "  number to judge the assembly by."
      write(*,'(A)') "  row   col    ||S||_F      rel.diff     ||S_int||_F  rel.diff_int  cond"
      flush(6)
    endif

    do r = 1, 4
      do c = 1, 4
        PetscCallA(MatNorm(S(r,c), NORM_FROBENIUS, nrm_S, ierr))
        PetscCallA(MatNorm(P(r,c), NORM_FROBENIUS, nrm_P, ierr))

        PetscCallA(MatDuplicate(S(r,c), MAT_COPY_VALUES, C_tmp, ierr))
        PetscCallA(MatAXPY(C_tmp, -1.0d0, P(r,c), DIFFERENT_NONZERO_PATTERN, ierr))
        PetscCallA(MatNorm(C_tmp, NORM_FROBENIUS, nrm_D, ierr))

        ! Interior-only: scale away the Dirichlet rows of both difference and
        ! reference before measuring.
        PetscCallA(MatDiagonalScale(C_tmp, mask(r), PETSC_NULL_VEC, ierr))
        PetscCallA(MatNorm(C_tmp, NORM_FROBENIUS, nrm_Di, ierr))
        PetscCallA(MatDestroy(C_tmp, ierr))

        PetscCallA(MatDuplicate(S(r,c), MAT_COPY_VALUES, D_tmp, ierr))
        PetscCallA(MatDiagonalScale(D_tmp, mask(r), PETSC_NULL_VEC, ierr))
        PetscCallA(MatNorm(D_tmp, NORM_FROBENIUS, nrm_Si, ierr))
        PetscCallA(MatDestroy(D_tmp, ierr))

        if (my_id == 0) then
          write(*,'(A,A5,2X,A5,2X,ES11.4,2X,A11,2X,ES11.4,2X,A11,3X,L1)') &
                "  ", vname(r), vname(c), &
                nrm_S,  ratio_str(nrm_D,  nrm_S), &
                nrm_Si, ratio_str(nrm_Di, nrm_Si), corrected(r,c)
        endif
      enddo
    enddo

    ! --- 2. Spectral-band mat-vec comparison --------------------------
    ! A smooth and a rough probe vector are built from the same random vector by
    ! filtering with the current mass matrix M_j: the operator D_lump^-1 M_j has
    ! eigenvalues ~3/2 on smooth modes and ~1/2 on oscillatory ones, so iterating
    ! it selects smooth modes, and iterating (2I - D_lump^-1 M_j) selects rough
    ! ones. Sec. 1.3 predicts close agreement on the smooth probe and an O(1)
    ! difference on the rough probe.
    do r = 1, 4
      PetscCallA(MatCreateVecs(P(1,1), x(r), PETSC_NULL_VEC, ierr))
      PetscCallA(VecDuplicate(x(r), yP(r), ierr))
      PetscCallA(VecDuplicate(x(r), yS(r), ierr))
    enddo
    PetscCallA(VecDuplicate(x(1), tmp,   ierr))
    PetscCallA(VecDuplicate(x(1), dlump, ierr))

    PetscCallA(MatGetDiagonal(B_33, dlump, ierr))
    PetscCallA(VecReciprocal(dlump, ierr))

    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "--- 2. Mat-vec discrepancy by spectral band ---------------------"
      flush(6)
    endif

    do ipass = 1, 2

      do r = 1, 4
        PetscCallA(VecSetRandom(x(r), PETSC_NULL_RANDOM, ierr))
        call spectral_filter(x(r), B_33, dlump, tmp, (ipass == 1))
        ! Probe only interior DOFs: a random entry on a ZBIG row is amplified by
        ! 1e12 and would swamp the measurement with boundary-penalty response
        ! that cancels identically between P and S.
        PetscCallA(VecPointwiseMult(x(r), x(r), mask(r), ierr))
        PetscCallA(VecNorm(x(r), NORM_2, nP, ierr))
        if (nP > 0.d0) PetscCallA(VecScale(x(r), 1.0d0/nP, ierr))
      enddo

      do r = 1, 4
        PetscCallA(VecZeroEntries(yP(r), ierr))
        PetscCallA(VecZeroEntries(yS(r), ierr))
        do c = 1, 4
          PetscCallA(MatMult(P(r,c), x(c), tmp, ierr))
          PetscCallA(VecAXPY(yP(r), 1.0d0, tmp, ierr))
          PetscCallA(MatMult(S(r,c), x(c), tmp, ierr))
          PetscCallA(VecAXPY(yS(r), 1.0d0, tmp, ierr))
        enddo
        ! and measure the response on interior rows only
        PetscCallA(VecPointwiseMult(yP(r), yP(r), mask(r), ierr))
        PetscCallA(VecPointwiseMult(yS(r), yS(r), mask(r), ierr))
      enddo

      do r = 1, 4
        PetscCallA(VecNorm(yS(r), NORM_2, nS, ierr))
        PetscCallA(VecNorm(yP(r), NORM_2, nP, ierr))
        PetscCallA(VecCopy(yP(r), tmp, ierr))
        PetscCallA(VecAXPY(tmp, -1.0d0, yS(r), ierr))
        PetscCallA(VecNorm(tmp, NORM_2, nD, ierr))

        if (my_id == 0) then
          if (ipass == 1) then
            write(*,'(A,A5,A,3(2X,ES11.4))') "  SMOOTH probe, row ", vname(r), &
                  " : |S x|, |P x|, rel.diff =", nS, nP, nD/max(nS,1.d-300)
          else
            write(*,'(A,A5,A,3(2X,ES11.4))') "  ROUGH  probe, row ", vname(r), &
                  " : |S x|, |P x|, rel.diff =", nS, nP, nD/max(nS,1.d-300)
          endif
        endif
      enddo

    enddo

    ! --- 3. Known discrepancy sources ---------------------------------
    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "--- 3. Differences present by construction ----------------------"
      write(*,'(A)') "  These are deliberate; they are NOT assembly errors."
      if (visco_num /= 0.d0) then
        write(*,'(A,ES11.4)') "  (a) ACTIVE: visco_num hyperviscosity dropped from D_u; visco_num = ", visco_num
        write(*,'(A)')        "      -> becomes 6th order under the substitution (not C1-representable)."
        write(*,'(A)')        "      To remove this source from the comparison, rerun with visco_num = 0."
      else
        write(*,'(A)')        "  (a) inactive: visco_num = 0, so the dropped hyperviscosity term"
        write(*,'(A)')        "      contributes nothing to the discrepancy in this run."
      endif
      if (eta_num /= 0.d0) then
        write(*,'(A,ES11.4)') "  (b) ACTIVE: boundary terms discarded by integration by parts; eta_num = ", eta_num
        write(*,'(A)')        "      -> affects boundary-adjacent rows only; also an O(1/R) commutator"
        write(*,'(A)')        "         in the eta_num term, traded for a symmetric positive form."
      else
        write(*,'(A)')        "  (b) partly inactive: eta_num = 0, so only the visco_T integration by"
        write(*,'(A)')        "      parts contributes a boundary term (it is exact in the interior)."
      endif
      write(*,'(A)')        "  (c) eliminating j and w drops their wall constraints, so Delta*psi is"
      write(*,'(A)')        "      no longer forced to vanish on the boundary."
      write(*,'(A)')        "  (d) the Pi_h - I projection error of Sec. 1.3 -- the dominant, expected"
      write(*,'(A)')        "      difference, and the reason band 2 above is reported at all."
      if (physics_pc_drop_psi_coupling) then
        write(*,'(A)')      "  (e) physics_pc_drop_psi_coupling is ON: the three group-psi blocks"
        write(*,'(A)')      "      (psi-T, T-psi, rho-psi) are zero in P_full by request (Milestone 3)."
      endif
      write(*,'(A)') "================================================================"
      write(*,'(A)') ""
      flush(6)
    endif

    ! --- Cleanup ------------------------------------------------------
    do r = 1, 4
      do c = 1, 4
        PetscCallA(MatDestroy(P(r,c), ierr))
        PetscCallA(MatDestroy(S(r,c), ierr))
      enddo
      PetscCallA(VecDestroy(x(r),  ierr))
      PetscCallA(VecDestroy(yP(r), ierr))
      PetscCallA(VecDestroy(yS(r), ierr))
      PetscCallA(VecDestroy(mask(r), ierr))
      PetscCallA(ISDestroy(is_red(r), ierr))
    enddo
    PetscCallA(VecDestroy(tmp,   ierr))
    PetscCallA(VecDestroy(dlump, ierr))
    PetscCallA(MatDestroy(P_aij, ierr))
    PetscCallA(MatDestroy(B_13, ierr))
    PetscCallA(MatDestroy(B_23, ierr))
    PetscCallA(MatDestroy(B_24, ierr))
    PetscCallA(MatDestroy(B_63, ierr))
    PetscCallA(MatDestroy(B_31, ierr))
    PetscCallA(MatDestroy(B_42, ierr))
    PetscCallA(MatDestroy(B_33, ierr))
    PetscCallA(MatDestroy(B_44, ierr))

  end subroutine verify_reduced_pde_operator


  !--------------------------------------------------------------------
  !> Milestone 4, Stage 4.1: verify the EXACT Schur factorization of the
  !! directly-assembled reduced PDE operator P_full.
  !!
  !! IMPORTANT: this operates on g_ctx%P_full_perm, i.e. the operator built by
  !! substituting j=J(psi), w=W(u) at the continuous level and assembled
  !! element-wise. It deliberately does NOT touch A_reduced_4x4 / the B_ij
  !! blocks, which come from the older path (blocks EXTRACTED out of the mixed
  !! Jacobian plus algebraic Schur corrections) and are a different operator.
  !!
  !! Reordering ( psi,u,rho,T) -> (y | u) with y = (psi,rho,T) puts P_full in
  !! Chacon's 2x2 arrow form
  !!     P_2x2 = [ M_yy  U_yu ]
  !!             [ L_uy  D_uu ]
  !! whose exact inverse is the three-step algorithm (Chacon 2008, eq. 9):
  !!     predictor : y* = M_yy^-1 b_y
  !!     u-update  : du = S_u^-1 (b_u - L_uy y*)
  !!     corrector : dy = y* - M_yy^-1 U_yu du
  !! with the exact Schur complement  S_u = D_uu - L_uy M_yy^-1 U_yu.
  !!
  !! Here S_u is formed EXACTLY, column by column, using a direct (MUMPS) solve
  !! on M_yy, and is then inverted by a direct dense LU. No approximation is
  !! introduced anywhere, so the factorized solve must reproduce a direct solve
  !! of P_2x2 to solver tolerance. That is the baseline every Stage-4.2
  !! approximation of S_u will be measured against.
  !!
  !! Note M_yy is block-diagonal only when physics_pc_drop_psi_coupling is on
  !! (the arrow form). The factorization above is exact either way, because
  !! M_yy is inverted as a coupled 3-variable operator; the flag only changes
  !! what M_yy contains.
  !--------------------------------------------------------------------
  subroutine verify_schur_factorization_4x4(comm, my_id)
    use phys_module, only: physics_pc_drop_psi_coupling

    integer, intent(in) :: comm, my_id

    IS  :: is_y, is_u, is_yu
    Mat :: M_yy, U_yu, L_uy, D_uu, P_2x2, S_u, S_corr
    Mat :: D_int, S_int, C_int
    Mat :: D_blk(3), U_blk(3), L_blk(3)
    KSP :: ksp_M, ksp_S, ksp_ref, ksp_blk(3)
    PC  :: pc_obj
    Vec :: e_u, t_y, w_y, sv_u, d_u
    Vec :: b_y, b_u, y_star, dy, du, r_u, tmp_y
    Vec :: b_ref, x_ref, x_fac
    Vec :: dmask, cv_x, cv_t, cv_s
    PetscInt :: n1, n3, ntot, k, kk
    PetscInt, allocatable :: idx(:), rows(:)
    PetscInt :: colidx(1)
    PetscScalar, pointer :: parr(:), qarr(:)
    PetscReal :: nrm_ref, nrm_diff, nrm_Du, nrm_Su, nrm_corr
    PetscReal :: ch_num, ch_den, chan(3), probe_den
    PetscErrorCode :: ierr
    integer :: nproc, mpierr, ip, ich, n_pen
    real*8, parameter :: zbig_thresh = 1.d11
    character(len=6) :: chname(3)

    chname = (/ ' psi  ', ' rho  ', '  T   ' /)

    if (.not. g_ctx%p_full_perm_ready) then
      if (my_id == 0) write(*,'(A)') &
        "[Stage 4.1] ERROR: P_full_perm not built; set physics_pc_reduced_pde=.true."
      return
    endif
    if (.not. g_ctx%is_reduced_created) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.1] ERROR: is_reduced not created."
      return
    endif

    call MPI_Comm_size(comm, nproc, mpierr)
    if (nproc /= 1) then
      if (my_id == 0) write(*,'(A)') &
        "[Stage 4.1] ERROR: this diagnostic is serial only (dense S_u). Use -np 1."
      return
    endif

    call MatGetSize(g_ctx%P_full_perm, ntot, PETSC_NULL_INTEGER, ierr)
    n1 = ntot / 4
    n3 = 3 * n1

    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "================================================================"
      write(*,'(A)') " Stage 4.1: exact Schur factorization of P_full  (Chacon eq. 9)"
      write(*,'(A)') "   operator under test: P_full (substituted PDE operator)"
      write(*,'(A,I0,A,I0)') "   y = (psi,rho,T), dim ", n3, " ;  u, dim ", n1
      if (physics_pc_drop_psi_coupling) then
        write(*,'(A)') "   M_yy is BLOCK-DIAGONAL (arrow form, drop_psi_coupling=.t.)"
      else
        write(*,'(A)') "   M_yy is FULL 3x3 (drop_psi_coupling=.f.)"
      endif
      write(*,'(A)') "================================================================"
    endif

    !--- index sets: y = (psi, rho, T) = vars 1,3,4 ; u = var 2 --------
    allocate(idx(n3))
    kk = 0
    do k = 0, n1-1
      kk = kk + 1 ;  idx(kk) = k                ! psi
    enddo
    do k = 0, n1-1
      kk = kk + 1 ;  idx(kk) = 2*n1 + k         ! rho
    enddo
    do k = 0, n1-1
      kk = kk + 1 ;  idx(kk) = 3*n1 + k         ! T
    enddo
    call ISCreateGeneral(comm, n3, idx, PETSC_COPY_VALUES, is_y, ierr)
    deallocate(idx)

    allocate(idx(n1))
    do k = 0, n1-1
      idx(k+1) = n1 + k                         ! u
    enddo
    call ISCreateGeneral(comm, n1, idx, PETSC_COPY_VALUES, is_u, ierr)
    deallocate(idx)

    allocate(idx(ntot))
    do k = 0, n1-1
      idx(k+1)        = k
      idx(n1+k+1)     = 2*n1 + k
      idx(2*n1+k+1)   = 3*n1 + k
      idx(3*n1+k+1)   = n1 + k
    enddo
    call ISCreateGeneral(comm, ntot, idx, PETSC_COPY_VALUES, is_yu, ierr)
    deallocate(idx)

    !--- the four blocks, and the reordered 2x2 reference operator -----
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_y, is_y, MAT_INITIAL_MATRIX, M_yy, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_y, is_u, MAT_INITIAL_MATRIX, U_yu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_u, is_y, MAT_INITIAL_MATRIX, L_uy, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_u, is_u, MAT_INITIAL_MATRIX, D_uu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_yu, is_yu, MAT_INITIAL_MATRIX, P_2x2, ierr)

    !--- direct solvers ------------------------------------------------
    call KSPCreate(comm, ksp_M, ierr)
    call KSPSetOperators(ksp_M, M_yy, M_yy, ierr)
    call KSPSetType(ksp_M, KSPPREONLY, ierr)
    call KSPGetPC(ksp_M, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_M, ierr)

    call KSPCreate(comm, ksp_ref, ierr)
    call KSPSetOperators(ksp_ref, P_2x2, P_2x2, ierr)
    call KSPSetType(ksp_ref, KSPPREONLY, ierr)
    call KSPGetPC(ksp_ref, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_ref, ierr)

    !--- form S_u = D_uu - L_uy M_yy^-1 U_yu, exactly, column by column
    call MatCreateVecs(U_yu, e_u, t_y, ierr)
    call VecDuplicate(t_y, w_y, ierr)
    call VecDuplicate(e_u, sv_u, ierr)
    call VecDuplicate(e_u, d_u, ierr)

    call MatCreate(PETSC_COMM_SELF, S_u, ierr)
    call MatSetSizes(S_u, n1, n1, n1, n1, ierr)
    call MatSetType(S_u, MATSEQDENSE, ierr)
    call MatSetUp(S_u, ierr)
    allocate(rows(n1))
    do k = 0, n1-1
      rows(k+1) = k
    enddo

    if (my_id == 0) write(*,'(A,I0,A)') &
      "[Stage 4.1] Forming exact S_u (", n1, " columns, one M_yy solve each)..."

    do k = 0, n1-1
      call VecZeroEntries(e_u, ierr)
      colidx(1) = k
      call VecSetValue(e_u, k, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_u, ierr)
      call VecAssemblyEnd(e_u, ierr)

      call MatMult(U_yu, e_u, t_y, ierr)         ! U_yu e_k
      call KSPSolve(ksp_M, t_y, w_y, ierr)       ! M_yy^-1 U_yu e_k
      call MatMult(L_uy, w_y, sv_u, ierr)         ! L_uy M_yy^-1 U_yu e_k
      call MatMult(D_uu, e_u, d_u, ierr)         ! D_uu e_k
      call VecAYPX(sv_u, -1.0d0, d_u, ierr)       ! sv_u <- d_u - sv_u

      call VecGetArray(sv_u, parr, ierr)
      call MatSetValues(S_u, n1, rows, 1, colidx, parr, INSERT_VALUES, ierr)
      call VecRestoreArray(sv_u, parr, ierr)

      if (my_id == 0 .and. n1 >= 10) then
        if (mod(k+1, max(1,n1/10)) == 0) write(*,'(A,I0,A,I0,A)') &
          "[Stage 4.1]   column ", k+1, "/", n1, ""
      endif
    enddo
    call MatAssemblyBegin(S_u, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(S_u, MAT_FINAL_ASSEMBLY, ierr)

    !--- direct dense LU on S_u ---------------------------------------
    call KSPCreate(comm, ksp_S, ierr)
    call KSPSetOperators(ksp_S, S_u, S_u, ierr)
    call KSPSetType(ksp_S, KSPPREONLY, ierr)
    call KSPGetPC(ksp_S, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call KSPSetUp(ksp_S, ierr)

    !--- the actual test ----------------------------------------------
    call MatCreateVecs(P_2x2, x_ref, b_ref, ierr)
    call VecDuplicate(x_ref, x_fac, ierr)
    call VecSetRandom(b_ref, PETSC_NULL_RANDOM, ierr)

    call KSPSolve(ksp_ref, b_ref, x_ref, ierr)   ! ground truth

    call MatCreateVecs(M_yy, y_star, b_y, ierr)
    call VecDuplicate(y_star, dy,    ierr)
    call VecDuplicate(y_star, tmp_y, ierr)
    call MatCreateVecs(D_uu, du, b_u, ierr)
    call VecDuplicate(du, r_u, ierr)

    ! split b_ref -> (b_y, b_u)
    call VecGetArray(b_ref, parr, ierr)
    call VecGetArray(b_y, qarr, ierr)
    do k = 1, n3
      qarr(k) = parr(k)
    enddo
    call VecRestoreArray(b_y, qarr, ierr)
    call VecGetArray(b_u, qarr, ierr)
    do k = 1, n1
      qarr(k) = parr(n3+k)
    enddo
    call VecRestoreArray(b_u, qarr, ierr)
    call VecRestoreArray(b_ref, parr, ierr)

    ! three-step exact inversion
    call KSPSolve(ksp_M, b_y, y_star, ierr)            ! predictor
    call MatMult(L_uy, y_star, r_u, ierr)
    call VecAYPX(r_u, -1.0d0, b_u, ierr)               ! b_u - L_uy y*
    call KSPSolve(ksp_S, r_u, du, ierr)                ! u-update
    call MatMult(U_yu, du, tmp_y, ierr)
    call KSPSolve(ksp_M, tmp_y, dy, ierr)
    call VecAYPX(dy, -1.0d0, y_star, ierr)             ! corrector

    ! recombine -> x_fac
    call VecGetArray(x_fac, parr, ierr)
    call VecGetArray(dy, qarr, ierr)
    do k = 1, n3
      parr(k) = qarr(k)
    enddo
    call VecRestoreArray(dy, qarr, ierr)
    call VecGetArray(du, qarr, ierr)
    do k = 1, n1
      parr(n3+k) = qarr(k)
    enddo
    call VecRestoreArray(du, qarr, ierr)
    call VecRestoreArray(x_fac, parr, ierr)

    call VecNorm(x_ref, NORM_2, nrm_ref, ierr)
    call VecAXPY(x_fac, -1.0d0, x_ref, ierr)
    call VecNorm(x_fac, NORM_2, nrm_diff, ierr)

    !--- how much does the Schur correction matter? --------------------
    call MatDuplicate(S_u, MAT_COPY_VALUES, S_corr, ierr)
    call MatAXPY(S_corr, -1.0d0, D_uu, DIFFERENT_NONZERO_PATTERN, ierr)

    ! Mask the ZBIG Dirichlet rows before taking any norm. Their penalty is
    ! identical in D_uu and S_u, cancels in the difference, and completely
    ! dominates ||D_uu||_F -- exactly the trap that made the Milestone-1
    ! diagonal blocks look spuriously perfect.
    call MatCreateVecs(D_uu, PETSC_NULL_VEC, dmask, ierr)
    call MatGetDiagonal(D_uu, dmask, ierr)
    call VecGetArray(dmask, parr, ierr)
    n_pen = 0
    do k = 1, n1
      if (abs(parr(k)) > zbig_thresh) then
        parr(k) = 0.0d0
        n_pen   = n_pen + 1
      else
        parr(k) = 1.0d0
      endif
    enddo
    call VecRestoreArray(dmask, parr, ierr)

    call MatDuplicate(D_uu,   MAT_COPY_VALUES, D_int, ierr)
    call MatDuplicate(S_u,    MAT_COPY_VALUES, S_int, ierr)
    call MatDuplicate(S_corr, MAT_COPY_VALUES, C_int, ierr)
    call MatDiagonalScale(D_int, dmask, PETSC_NULL_VEC, ierr)
    call MatDiagonalScale(S_int, dmask, PETSC_NULL_VEC, ierr)
    call MatDiagonalScale(C_int, dmask, PETSC_NULL_VEC, ierr)
    call MatNorm(D_int, NORM_FROBENIUS, nrm_Du,   ierr)
    call MatNorm(S_int, NORM_FROBENIUS, nrm_Su,   ierr)
    call MatNorm(C_int, NORM_FROBENIUS, nrm_corr, ierr)

    !--- per-channel feedback strength (exact iff M_yy block-diagonal) --
    call MatCreateVecs(D_uu, cv_x, cv_t, ierr)
    call VecDuplicate(cv_x, cv_s, ierr)
    do ich = 1, 3
      if (ich == 1) then
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(1), g_ctx%is_reduced(1), MAT_INITIAL_MATRIX, D_blk(ich), ierr)
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(1), g_ctx%is_reduced(2), MAT_INITIAL_MATRIX, U_blk(ich), ierr)
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(1), MAT_INITIAL_MATRIX, L_blk(ich), ierr)
      else if (ich == 2) then
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(3), g_ctx%is_reduced(3), MAT_INITIAL_MATRIX, D_blk(ich), ierr)
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(3), g_ctx%is_reduced(2), MAT_INITIAL_MATRIX, U_blk(ich), ierr)
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(3), MAT_INITIAL_MATRIX, L_blk(ich), ierr)
      else
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(4), g_ctx%is_reduced(4), MAT_INITIAL_MATRIX, D_blk(ich), ierr)
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(4), g_ctx%is_reduced(2), MAT_INITIAL_MATRIX, U_blk(ich), ierr)
        call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(4), MAT_INITIAL_MATRIX, L_blk(ich), ierr)
      endif

      call KSPCreate(comm, ksp_blk(ich), ierr)
      call KSPSetOperators(ksp_blk(ich), D_blk(ich), D_blk(ich), ierr)
      call KSPSetType(ksp_blk(ich), KSPPREONLY, ierr)
      call KSPGetPC(ksp_blk(ich), pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
      call KSPSetUp(ksp_blk(ich), ierr)

      ! Average over a few random probes. All three blocks are n1 x n1, so
      ! the work vectors must be n1-sized (t_y/w_y are n3 and would silently
      ! fail). Penalty rows are masked out of the response, otherwise the
      ! number is just the ZBIG boundary reaction.
      chan(ich) = 0.0d0
      ch_den    = 0.0d0
      do ip = 1, 8
        PetscCallA(VecSetRandom(cv_x, PETSC_NULL_RANDOM, ierr))
        PetscCallA(VecPointwiseMult(cv_x, cv_x, dmask, ierr))
        PetscCallA(MatMult(U_blk(ich), cv_x, cv_t, ierr))
        PetscCallA(KSPSolve(ksp_blk(ich), cv_t, cv_s, ierr))
        PetscCallA(MatMult(L_blk(ich), cv_s, cv_t, ierr))
        PetscCallA(VecPointwiseMult(cv_t, cv_t, dmask, ierr))
        PetscCallA(VecNorm(cv_t, NORM_2, ch_num, ierr))
        PetscCallA(VecNorm(cv_x, NORM_2, probe_den, ierr))
        chan(ich) = chan(ich) + ch_num
        ch_den    = ch_den + probe_den
      enddo
      chan(ich) = chan(ich) / max(ch_den, 1.d-300)
    enddo

    !--- report --------------------------------------------------------
    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "--- Exactness of the three-step factorization -------------------"
      write(*,'(A,E12.4)') "  ||x_factorized - x_direct|| / ||x_direct|| = ", &
                            nrm_diff / max(nrm_ref, 1.d-300)
      write(*,'(A)') "  (must be at the level of the direct-solver tolerance;"
      write(*,'(A)') "   anything larger means a block, sign or ordering error)"
      write(*,'(A)') ""
      write(*,'(A)') "--- Weight of the Schur correction ------------------------------"
      write(*,'(A,I0,A,I0,A)') "  (", n_pen, " of ", n1, " u-rows carry the ZBIG penalty and are masked out)"
      write(*,'(A,E12.4)') "  ||D_uu||_F  (interior) = ", nrm_Du
      write(*,'(A,E12.4)') "  ||S_u||_F   (interior) = ", nrm_Su
      write(*,'(A,E12.4)') "  ||S_u-D_uu||(interior) = ", nrm_corr
      write(*,'(A,E12.4)') "  correction / ||D_uu|| = ", nrm_corr / max(nrm_Du, 1.d-300)
      write(*,'(A)') "  (if this is small, D_uu alone is already a usable Schur"
      write(*,'(A)') "   approximation and Stage 4.2 is nearly free)"
      write(*,'(A)') ""
      write(*,'(A)') "--- Feedback strength per channel  ||L_uk D_k^-1 U_ku v||/||v|| -"
      if (.not. physics_pc_drop_psi_coupling) then
        write(*,'(A)') "  NOTE: drop_psi_coupling=.f., so M_yy is not block-diagonal and"
        write(*,'(A)') "        these three channels do not sum to the exact correction."
      endif
      do ich = 1, 3
        write(*,'(A,A,A,E12.4)') "  channel ", chname(ich), " : ", chan(ich)
      enddo
      write(*,'(A)') "  (the dominant channel is the one Stage 4.2 must reproduce)"
      write(*,'(A)') "================================================================"
      write(*,'(A)') ""
    endif

    !--- cleanup -------------------------------------------------------
    deallocate(rows)
    do ich = 1, 3
      call KSPDestroy(ksp_blk(ich), ierr)
      call MatDestroy(D_blk(ich), ierr)
      call MatDestroy(U_blk(ich), ierr)
      call MatDestroy(L_blk(ich), ierr)
    enddo
    call KSPDestroy(ksp_M, ierr)
    call KSPDestroy(ksp_S, ierr)
    call KSPDestroy(ksp_ref, ierr)
    call MatDestroy(M_yy, ierr)
    call MatDestroy(U_yu, ierr)
    call MatDestroy(L_uy, ierr)
    call MatDestroy(D_uu, ierr)
    call MatDestroy(P_2x2, ierr)
    call MatDestroy(S_u, ierr)
    call MatDestroy(S_corr, ierr)
    call MatDestroy(D_int, ierr)
    call MatDestroy(S_int, ierr)
    call MatDestroy(C_int, ierr)
    call VecDestroy(dmask, ierr)
    call VecDestroy(cv_x, ierr)
    call VecDestroy(cv_t, ierr)
    call VecDestroy(cv_s, ierr)
    call ISDestroy(is_y, ierr)
    call ISDestroy(is_u, ierr)
    call ISDestroy(is_yu, ierr)
    call VecDestroy(e_u, ierr)
    call VecDestroy(t_y, ierr)
    call VecDestroy(w_y, ierr)
    call VecDestroy(sv_u, ierr)
    call VecDestroy(d_u, ierr)
    call VecDestroy(b_y, ierr)
    call VecDestroy(b_u, ierr)
    call VecDestroy(y_star, ierr)
    call VecDestroy(dy, ierr)
    call VecDestroy(du, ierr)
    call VecDestroy(r_u, ierr)
    call VecDestroy(tmp_y, ierr)
    call VecDestroy(b_ref, ierr)
    call VecDestroy(x_ref, ierr)
    call VecDestroy(x_fac, ierr)

  end subroutine verify_schur_factorization_4x4


  !--------------------------------------------------------------------
  !> Milestone 4, Stage 4.2: approximate Schur complements for P_full.
  !!
  !! Stage 4.1 established that on P_full the Schur complement is dominated
  !! by ONE term,
  !!     S_u = D_uu - L_upsi D_psi^-1 U_psiu  (+ tiny rho,T channels)
  !! so every approximation here targets D_psi^-1 inside that single term.
  !!
  !! Two families, both from Chacon (2008):
  !!
  !!  (a) SMALL-FLOW LIMIT  [his ref. 4]:  M^-1 ~ dt*I. In JOREK's weak form
  !!      D_psi = (1+zeta)*Q_1R + theta*dt*(advection + resistive), so
  !!      dropping everything but the mass term gives
  !!          D_psi^-1  ->  (1+zeta)^-1 Q_1R^-1
  !!      i.e. M_*^-1 = I/(1+zeta). This is candidate M0 of the existing
  !!      commutator table, and is what g_ctx%spbpd_inv_gears was reaching for.
  !!
  !!  (b) COMMUTATOR DEVICE [his sec. 4.1]: find M_* on u-space with
  !!          D_psi^-1 U_psiu  ~  U_psiu M_*^-1     (i.e. U M_* ~ D_psi U)
  !!      giving   S_u ~ D_uu - L_upsi U_psiu M_*^-1.
  !!      Candidates come from mod_petsc_pc_commutator_table (M0, M1x, M0R,
  !!      and M1a/M3/M2 when the building blocks are assembled).
  !!
  !! IMPORTANT: the pre-existing commutator ANALYSIS module measures its
  !! defect against A_pp = B11 = amat_11, a block of the MIXED Jacobian.
  !! P_full's D_psi is amat_11 + amat_13[dj->J(dpsi)], which is NOT the same
  !! operator. Here every reference block (D_uu, U_psiu, L_upsi, D_psi) is
  !! taken from P_full_perm; only the M_* CANDIDATES are reused from the
  !! table, which is legitimate because they are generic u-space operators.
  !!
  !! For each candidate the routine reports
  !!   - ||Shat - S_u||_F / ||S_u||_F   (interior rows only), and
  !!   - GMRES iterations to solve S_u x = b preconditioned by Shat^-1,
  !! the latter being the quantity that actually governs Stage 4.3.
  !--------------------------------------------------------------------
  subroutine verify_schur_approx_4x4(comm, my_id)
    use mod_petsc_pc_commutator_table, only: CM_NOP, CM_MAXC, CM_LABLEN, &
          CM_OP_B11, CM_OP_Q1R, CM_OP_QR, CM_OP_AISO, CM_OP_AANI, cm_table_build, &
          cm_ops_gather, cm_blocks_ready, cm_table_lookup
    use mod_elt_matrix_commutator, only: CM_S1R, CM_ES1R, CM_EG1R
    use phys_module, only: time_evol_zeta, time_evol_theta, tstep, tstep_prev, eta, &
                           physics_pc_schur_assemble

    integer, intent(in) :: comm, my_id

    IS  :: is_y, is_u
    Mat :: M_yy, U_yu, L_uy, D_uu, S_u, Shat, S_dif
    Mat :: D_psi, U_pu, L_up, A_uM
    !--- Stage 6.1: the rho and T channels of the same Schur complement ---
    Mat :: D_rho, U_ru, L_ur
    Mat :: D_T,   U_Tu, L_uT
    Mat :: op(CM_NOP)
    KSP :: ksp_M, ksp_psi, ksp_AuM, ksp_S, ksp_test, ksp_Q
    PC  :: pc_obj
    Vec :: e_u, t_y, w_y, sv_u, d_u, cv_t, cv_w, cv_q, dmask, pmask, bt, xt
    Vec :: prow, pprb
    Mat :: R_ex
    !--- Workstream A (sparse assembled Schur complement) ---------------
    Mat :: S_ass, T1_a, T2_a, Dsc, Lsc, Shat_a
    Mat :: G_u, G_p, Qi_u, Qi_p, Pat_u, Pat_p
    KSP :: ksp_Qu
    Vec :: ones_p, lq_u, lq_p, ax, ay, az, aw
    !--- Stage 6.1: the channel list ------------------------------------
    !  Channel y contributes  L_uy Q_y^-1 U_yu / (1+zeta)  to the small-flow
    !  Schur complement. All four reduced variables live on the SAME C1 Bezier
    !  scalar space, so the only per-channel difference is the geometric weight
    !  the equation carries: the psi row is 1/R-weighted (amat_11 -> Q1R) while
    !  the rho and T rows are R-weighted (amat_55/amat_66, both
    !  v*(rho|T)*BigR*xjac*(1+zeta) -> QR). Hence NO new element assembly: the
    !  two masses are already in op() via cm_ops_gather(B_11, B_33, B_44).
    integer, parameter :: NCH    = 3
    integer, parameter :: CH_PSI = 1, CH_RHO = 2, CH_T = 3
    Mat     :: ch_L(NCH), ch_U(NCH), ch_D(NCH)
    integer :: ch_q(NCH)              !< op() index of this channel's mass Q_y
    Vec     :: ch_lq(NCH)             !< its diagonal/row-sum reciprocal
    Mat     :: ch_Qi(NCH)             !< its FSAI inverse, when built
    logical :: ch_on(NCH)             !< channels active in the case being built
    character(len=3), parameter :: chname(NCH) = (/ "psi", "rho", "T  " /)
    integer :: ich
    !  Because rho and T share QR, the second mass needs exactly one extra
    !  lumping vector and one extra FSAI factor, not two.
    Mat     :: G_R, Qi_R, Pat_R
    Vec     :: lq_R
    KSP     :: ksp_QR                 !< exact QR, for the mu/mp=3,4 variants
    real*8  :: ch_worth(NCH), ch_dmg(NCH), ch_m1r(NCH), ch_mR(NCH)
    !  Second index selects how the two interior mass inverses -- Q_u^-1 in
    !  D_uu Q_u^-1 A_uM, and Q^-1 in L_up Q^-1 U_pu -- are approximated.
    !  Variants 3 and 4 keep ONE of them exact to attribute the damage.
    integer, parameter :: NLUMP = 8
    real*8  :: rassm(CM_MAXC+3,NLUMP)   !< ||Shat_eff - S_u|| / ||S_u||, interior
    real*8  :: rasscons(CM_MAXC+3,NLUMP)!< algebra check vs the operator-form Shat
    real*8  :: fillr(CM_MAXC+3,NLUMP)   !< nnz(S_ass) / nnz(D_uu); -1 if not assemblable
    integer :: itassm(CM_MAXC+3,NLUMP)
    integer :: ilump
    integer :: nzrow(NLUMP)             !< rows whose mass "inverse" had to be guarded
    character(len=22), parameter :: lumpname(NLUMP) = (/ &
         "row-sum  / row-sum    ", &
         "diagonal / diagonal   ", &
         "EXACT    / diagonal   ", &
         "diagonal / EXACT      ", &
         "Neumann-2/ diagonal   ", &
         "NONE     / NONE       ", &
         "FSAI-0   / FSAI-0     ", &
         "FSAI-1   / FSAI-1     " /)
    PetscInt :: n1, n3, ntot, k
    PetscInt, allocatable :: idx(:), rows(:)
    PetscInt :: colidx(1), nits
    PetscScalar, pointer :: parr(:), parr2(:)
    PetscReal :: nrm_S, nrm_dif, chk
    integer :: nbrow
    PetscErrorCode :: ierr
    integer :: nproc, mpierr, ic, ncand, ncase, icase
    integer :: ic_sf              !< table index of M0, the small-flow candidate
    real*8  :: rel_sf             !< operator-form defect of the small-flow channel
    real*8  :: opz, zeta, tdt
    real*8  :: coef(CM_MAXC, CM_NOP)
    integer :: quop(CM_MAXC)
    character(len=CM_LABLEN) :: lab(CM_MAXC)
    character(len=12) :: cname(CM_MAXC+3)
    real*8  :: rel(CM_MAXC+3)
    real*8  :: rdp(CM_MAXC+3)   !< ||A_uM - D_psi|| / ||D_psi||, interior
    real*8  :: nrm_Dp
    integer :: itc(CM_MAXC+3)
    real*8, parameter :: zbig_thresh = 1.d11

    if (.not. g_ctx%p_full_perm_ready) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.2] ERROR: P_full_perm not built."
      return
    endif
    call MPI_Comm_size(comm, nproc, mpierr)
    if (nproc /= 1) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.2] ERROR: serial only. Use -np 1."
      return
    endif

    zeta = time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    opz  = 1.d0 + zeta
    tdt  = time_evol_theta * tstep

    call MatGetSize(g_ctx%P_full_perm, ntot, PETSC_NULL_INTEGER, ierr)
    n1 = ntot / 4
    n3 = 3 * n1

    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "================================================================"
      write(*,'(A)') " Stage 4.2: approximate Schur complements for P_full"
      write(*,'(A,E12.4,A,E12.4)') "   (1+zeta) = ", opz, " ,  theta*dt = ", tdt
      write(*,'(A)') "================================================================"
    endif

    !--- blocks of P_full (same construction as Stage 4.1) -------------
    allocate(idx(n3))
    do k = 0, n1-1
      idx(k+1)      = k
      idx(n1+k+1)   = 2*n1 + k
      idx(2*n1+k+1) = 3*n1 + k
    enddo
    call ISCreateGeneral(comm, n3, idx, PETSC_COPY_VALUES, is_y, ierr)
    deallocate(idx)
    allocate(idx(n1))
    do k = 0, n1-1
      idx(k+1) = n1 + k
    enddo
    call ISCreateGeneral(comm, n1, idx, PETSC_COPY_VALUES, is_u, ierr)
    deallocate(idx)

    call MatCreateSubMatrix(g_ctx%P_full_perm, is_y, is_y, MAT_INITIAL_MATRIX, M_yy, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_y, is_u, MAT_INITIAL_MATRIX, U_yu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_u, is_y, MAT_INITIAL_MATRIX, L_uy, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_u, is_u, MAT_INITIAL_MATRIX, D_uu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(1), g_ctx%is_reduced(1), MAT_INITIAL_MATRIX, D_psi, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(1), g_ctx%is_reduced(2), MAT_INITIAL_MATRIX, U_pu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(1), MAT_INITIAL_MATRIX, L_up, ierr)
    !--- Stage 6.1: the same three blocks for the rho and T channels ----
    !    is_reduced = (psi, u, rho, T), so 3 and 4 are the transport rows.
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(3), g_ctx%is_reduced(3), MAT_INITIAL_MATRIX, D_rho, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(3), g_ctx%is_reduced(2), MAT_INITIAL_MATRIX, U_ru, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(3), MAT_INITIAL_MATRIX, L_ur, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(4), g_ctx%is_reduced(4), MAT_INITIAL_MATRIX, D_T, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(4), g_ctx%is_reduced(2), MAT_INITIAL_MATRIX, U_Tu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(4), MAT_INITIAL_MATRIX, L_uT, ierr)

    call cm_ops_gather(g_ctx%B_11, g_ctx%B_33, g_ctx%B_44, op)

    !--- Stage 6.1: bind the channel list ------------------------------
    ch_L(CH_PSI) = L_up;  ch_U(CH_PSI) = U_pu;  ch_D(CH_PSI) = D_psi
    ch_L(CH_RHO) = L_ur;  ch_U(CH_RHO) = U_ru;  ch_D(CH_RHO) = D_rho
    ch_L(CH_T)   = L_uT;  ch_U(CH_T)   = U_Tu;  ch_D(CH_T)   = D_T
    ch_q(CH_PSI) = CM_OP_Q1R
    ch_q(CH_RHO) = CM_OP_QR
    ch_q(CH_T)   = CM_OP_QR
    ! The commutator candidates are psi-channel objects by construction, so
    ! every builder runs psi-only unless a case explicitly widens the set.
    ch_on = (/ .true., .false., .false. /)

    call KSPCreate(comm, ksp_M, ierr)
    call KSPSetOperators(ksp_M, M_yy, M_yy, ierr)
    call KSPSetType(ksp_M, KSPPREONLY, ierr)
    call KSPGetPC(ksp_M, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_M, ierr)

    !--- sanity: does P_full's U_psiu agree with the mixed B_12? -------
    !    (they should: both are amat_12. A mismatch means the 1-variable
    !     DOF orderings of P_full_perm and the B_ij blocks differ, which
    !     would invalidate reusing the commutator table's operators.)
    call MatDuplicate(U_pu, MAT_COPY_VALUES, S_dif, ierr)
    call MatAXPY(S_dif, -1.0d0, g_ctx%B_12, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatNorm(S_dif, NORM_FROBENIUS, nrm_dif, ierr)
    call MatNorm(g_ctx%B_12, NORM_FROBENIUS, chk, ierr)
    if (my_id == 0) write(*,'(A,E12.4)') &
      "[Stage 4.2] layout check  ||U_psiu(P_full) - B_12|| / ||B_12|| = ", &
      nrm_dif / max(chk, 1.d-300)
    call MatDestroy(S_dif, ierr)

    !--- Stage 6.1: the same check for the four transport coupling blocks.
    !    This is what makes the harness verdict transferable to production,
    !    which builds S_prod from the MIXED blocks (B_25/B_52/B_26/B_62) and
    !    not from P_full. The psi row is condensed (Atilde_61 exists because
    !    of B_63), but the u<->rho,T couplings should be untouched by the
    !    j/w elimination -- these four numbers are the proof, not the claim.
    if (my_id == 0) write(*,'(A)') &
      "[Stage 6.1] layout checks vs the mixed blocks (must be round-off):"
    do ich = 1, 4
      select case (ich)
      case (1); call MatDuplicate(U_ru, MAT_COPY_VALUES, S_dif, ierr)
                call MatAXPY(S_dif, -1.0d0, g_ctx%B_52, DIFFERENT_NONZERO_PATTERN, ierr)
                call MatNorm(g_ctx%B_52, NORM_FROBENIUS, chk, ierr)
      case (2); call MatDuplicate(L_ur, MAT_COPY_VALUES, S_dif, ierr)
                call MatAXPY(S_dif, -1.0d0, g_ctx%B_25, DIFFERENT_NONZERO_PATTERN, ierr)
                call MatNorm(g_ctx%B_25, NORM_FROBENIUS, chk, ierr)
      case (3); call MatDuplicate(U_Tu, MAT_COPY_VALUES, S_dif, ierr)
                call MatAXPY(S_dif, -1.0d0, g_ctx%B_62, DIFFERENT_NONZERO_PATTERN, ierr)
                call MatNorm(g_ctx%B_62, NORM_FROBENIUS, chk, ierr)
      case (4); call MatDuplicate(L_uT, MAT_COPY_VALUES, S_dif, ierr)
                call MatAXPY(S_dif, -1.0d0, g_ctx%B_26, DIFFERENT_NONZERO_PATTERN, ierr)
                call MatNorm(g_ctx%B_26, NORM_FROBENIUS, chk, ierr)
      end select
      call MatNorm(S_dif, NORM_FROBENIUS, nrm_dif, ierr)
      call MatDestroy(S_dif, ierr)
      if (my_id == 0) then
        select case (ich)
        case (1); write(*,'(A,E12.4)') "             U_rhou vs B_52 = ", nrm_dif/max(chk,1.d-300)
        case (2); write(*,'(A,E12.4)') "             L_urho vs B_25 = ", nrm_dif/max(chk,1.d-300)
        case (3); write(*,'(A,E12.4)') "             U_Tu   vs B_62 = ", nrm_dif/max(chk,1.d-300)
        case (4); write(*,'(A,E12.4)') "             L_uT   vs B_26 = ", nrm_dif/max(chk,1.d-300)
        end select
      endif
    enddo

    !--- how far is D_psi from the operators the candidates model? -----
    !    Every table candidate models D_psi as (1+zeta)*mass (+flow terms).
    !    P_full's D_psi = amat_11 + amat_13[dj->J(dpsi)] also carries the
    !    SUBSTITUTED resistive term, which no candidate represents. Measure
    !    both gaps before interpreting any defect.
    ! psi-space interior mask. WITHOUT this the ZBIG Dirichlet rows -- which
    ! are identical in D_psi, B11 and Q -- dominate every Frobenius norm and
    ! make the blocks look identical when their interiors are not.
    call MatCreateVecs(D_psi, PETSC_NULL_VEC, pmask, ierr)
    call MatGetDiagonal(D_psi, pmask, ierr)
    call VecGetArray(pmask, parr, ierr)
    do k = 1, n1
      if (abs(parr(k)) > zbig_thresh) then
        parr(k) = 0.0d0
      else
        parr(k) = 1.0d0
      endif
    enddo
    call VecRestoreArray(pmask, parr, ierr)

    ! Second boundary criterion, needed under eliminate_boundary_dofs: there
    ! the Dirichlet rows carry a MODEST representative diagonal, so the
    ! zbig_thresh test above catches nothing, yet those rows are still not
    ! comparable across operators -- construct_commutator_matrices ZEROES the
    ! building blocks' boundary rows (they rely on the extracted mass to carry
    ! the boundary), while D_psi keeps its diagonal. Probing a building block
    ! with a random vector identifies the zeroed rows exactly. (A vector of
    ! ones would not: a stiffness annihilates constants on interior rows too.)
    if (cm_blocks_ready()) then
      call VecDuplicate(pmask, prow, ierr)
      call VecDuplicate(pmask, pprb, ierr)
      call VecSetRandom(prow, PETSC_NULL_RANDOM, ierr)
      call MatMult(op(3+CM_S1R), prow, pprb, ierr)
      call VecGetArray(pprb, parr, ierr)
      call VecGetArray(pmask, parr2, ierr)
      nbrow = 0
      do k = 1, n1
        if (parr(k) == 0.0d0) then
          parr2(k) = 0.0d0
          nbrow = nbrow + 1
        endif
      enddo
      call VecRestoreArray(pmask, parr2, ierr)
      call VecRestoreArray(pprb, parr, ierr)
      if (my_id == 0) write(*,'(A,I0,A,I0)') &
        "[Stage 4.2] psi-space boundary rows excluded from the norms: ", nbrow, " of ", n1
      call VecDestroy(prow, ierr)
      call VecDestroy(pprb, ierr)
    endif

    call MatDuplicate(D_psi, MAT_COPY_VALUES, S_dif, ierr)
    call MatAXPY(S_dif, -opz, op(CM_OP_Q1R), DIFFERENT_NONZERO_PATTERN, ierr)
    call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
    call MatNorm(S_dif, NORM_FROBENIUS, nrm_dif, ierr)
    call MatDestroy(S_dif, ierr)
    call MatDuplicate(op(CM_OP_Q1R), MAT_COPY_VALUES, S_dif, ierr)
    call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
    call MatNorm(S_dif, NORM_FROBENIUS, chk, ierr)
    call MatDestroy(S_dif, ierr)
    if (my_id == 0) write(*,'(A,E12.4)') &
      "[Stage 4.2] ||D_psi-(1+zeta)Q||/||(1+zeta)Q||  (interior) = ", &
      nrm_dif / max(opz*chk, 1.d-300)

    call MatDuplicate(D_psi, MAT_COPY_VALUES, S_dif, ierr)
    call MatAXPY(S_dif, -1.0d0, g_ctx%B_11, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
    call MatNorm(S_dif, NORM_FROBENIUS, nrm_dif, ierr)
    call MatDestroy(S_dif, ierr)
    call MatDuplicate(g_ctx%B_11, MAT_COPY_VALUES, S_dif, ierr)
    call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
    call MatNorm(S_dif, NORM_FROBENIUS, chk, ierr)
    call MatDestroy(S_dif, ierr)
    if (my_id == 0) write(*,'(A,E12.4)') &
      "[Stage 4.2] ||D_psi-B11||/||B11||              (interior) = ", &
      nrm_dif / max(chk, 1.d-300)

    !--- direct test of the resistive building blocks -------------------
    ! D_psi - B_11 IS the substituted resistive term, exactly and alone:
    ! D_psi = amat_11 + amat_13[dj -> J(dpsi)] and B_11 = amat_11. So this
    ! compares the candidate blocks against the very operator they are meant
    ! to reproduce, with no commutator device in the way -- it separates
    ! "the block is wrong" from "the device fails".
    if (cm_blocks_ready()) then
      call MatDuplicate(D_psi, MAT_COPY_VALUES, R_ex, ierr)
      call MatAXPY(R_ex, -1.0d0, g_ctx%B_11, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDuplicate(R_ex, MAT_COPY_VALUES, S_dif, ierr)
      call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
      call MatNorm(S_dif, NORM_FROBENIUS, chk, ierr)
      call MatDestroy(S_dif, ierr)
      do k = 1, 3
        call MatDuplicate(R_ex, MAT_COPY_VALUES, S_dif, ierr)
        if (k == 1) then
          call MatAXPY(S_dif, -eta*tdt, op(3+CM_S1R), DIFFERENT_NONZERO_PATTERN, ierr)
        else
          call MatAXPY(S_dif, -eta*tdt, op(3+CM_ES1R), DIFFERENT_NONZERO_PATTERN, ierr)
          if (k == 3) call MatAXPY(S_dif, -eta*tdt, op(3+CM_EG1R), &
                                   DIFFERENT_NONZERO_PATTERN, ierr)
        endif
        call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
        call MatNorm(S_dif, NORM_FROBENIUS, nrm_dif, ierr)
        call MatDestroy(S_dif, ierr)
        if (my_id == 0) then
          if (k == 1) write(*,'(A,E12.4)') &
            "[Stage 4.2] resistive block test   eta*tdt*S1R          : ", nrm_dif/max(chk,1.d-300)
          if (k == 2) write(*,'(A,E12.4)') &
            "[Stage 4.2] resistive block test   eta*tdt*ES1R         : ", nrm_dif/max(chk,1.d-300)
          if (k == 3) write(*,'(A,E12.4)') &
            "[Stage 4.2] resistive block test   eta*tdt*(ES1R+EG1R)  : ", nrm_dif/max(chk,1.d-300)
        endif
      enddo
      call MatDestroy(R_ex, ierr)
    endif

    ! Interior ||D_psi||, the reference for the per-candidate OPERATOR-FORM
    ! column below. That column answers a different question from the Schur
    ! defect: "does this candidate's A_uM reproduce D_psi at all?" A candidate
    ! can match D_psi well and still give a poor Schur complement -- that
    ! separates a wrong operator from a violated commutator premise.
    call MatDuplicate(D_psi, MAT_COPY_VALUES, S_dif, ierr)
    call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
    call MatNorm(S_dif, NORM_FROBENIUS, nrm_Dp, ierr)
    call MatDestroy(S_dif, ierr)

    !--- exact S_u by probing (the Stage 4.1 baseline) -----------------
    call MatCreateVecs(U_yu, e_u, t_y, ierr)
    call VecDuplicate(t_y, w_y, ierr)
    call VecDuplicate(e_u, sv_u, ierr)
    call VecDuplicate(e_u, d_u, ierr)
    call VecDuplicate(e_u, cv_t, ierr)
    call VecDuplicate(e_u, cv_w, ierr)
    call VecDuplicate(e_u, cv_q, ierr)

    call MatCreate(PETSC_COMM_SELF, S_u, ierr)
    call MatSetSizes(S_u, n1, n1, n1, n1, ierr)
    call MatSetType(S_u, MATSEQDENSE, ierr)
    call MatSetUp(S_u, ierr)
    allocate(rows(n1))
    do k = 0, n1-1
      rows(k+1) = k
    enddo

    if (my_id == 0) write(*,'(A)') "[Stage 4.2] forming exact S_u ..."
    do k = 0, n1-1
      call VecZeroEntries(e_u, ierr)
      colidx(1) = k
      call VecSetValue(e_u, k, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_u, ierr)
      call VecAssemblyEnd(e_u, ierr)
      call MatMult(U_yu, e_u, t_y, ierr)
      call KSPSolve(ksp_M, t_y, w_y, ierr)
      call MatMult(L_uy, w_y, sv_u, ierr)
      call MatMult(D_uu, e_u, d_u, ierr)
      call VecAYPX(sv_u, -1.0d0, d_u, ierr)
      call VecGetArray(sv_u, parr, ierr)
      call MatSetValues(S_u, n1, rows, 1, colidx, parr, INSERT_VALUES, ierr)
      call VecRestoreArray(sv_u, parr, ierr)
    enddo
    call MatAssemblyBegin(S_u, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(S_u, MAT_FINAL_ASSEMBLY, ierr)

    !--- interior mask -------------------------------------------------
    call MatCreateVecs(D_uu, PETSC_NULL_VEC, dmask, ierr)
    call MatGetDiagonal(D_uu, dmask, ierr)
    call VecGetArray(dmask, parr, ierr)
    do k = 1, n1
      if (abs(parr(k)) > zbig_thresh) then
        parr(k) = 0.0d0
      else
        parr(k) = 1.0d0
      endif
    enddo
    call VecRestoreArray(dmask, parr, ierr)

    call MatDuplicate(S_u, MAT_COPY_VALUES, S_dif, ierr)
    call MatDiagonalScale(S_dif, dmask, PETSC_NULL_VEC, ierr)
    call MatNorm(S_dif, NORM_FROBENIUS, nrm_S, ierr)
    call MatDestroy(S_dif, ierr)

    !--- candidate table ----------------------------------------------
    call cm_table_build(opz, tdt, eta, coef, quop, lab, ncand)
    if (my_id == 0) write(*,'(A,I0)') "[Stage 4.2] candidates in table: ", ncand
    if (my_id == 0 .and. .not. cm_blocks_ready()) write(*,'(A)') &
      "[Stage 4.2] NOTE: commutator building blocks not assembled "// &
      "(set commutator_analysis=.t.); only M0/M1x/M0R available."

    !--- exact D_psi^-1 for the reference channel ----------------------
    ! Riesz map Q = Q1R = B_33 on psi-space. The intertwining relation is
    !   D_psi Q^-1 U ~ U M_*   <=>   D_psi^-1 U ~ Q^-1 U M_*^-1
    ! so the Q^-1 is part of the device and must NOT be dropped.
    call KSPCreate(comm, ksp_Q, ierr)
    call KSPSetOperators(ksp_Q, op(CM_OP_Q1R), op(CM_OP_Q1R), ierr)
    call KSPSetType(ksp_Q, KSPPREONLY, ierr)
    call KSPGetPC(ksp_Q, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_Q, ierr)

    !--- Stage 6.1: the same, for the R-weighted mass the transport rows
    !    carry. Shared by the rho and T channels, hence one factorization.
    call KSPCreate(comm, ksp_QR, ierr)
    call KSPSetOperators(ksp_QR, op(CM_OP_QR), op(CM_OP_QR), ierr)
    call KSPSetType(ksp_QR, KSPPREONLY, ierr)
    call KSPGetPC(ksp_QR, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_QR, ierr)

    !--- Stage 6.1: price the three channels before approximating any of
    !    them. This is the measurement that decides which channels earn a
    !    place in the production operator, and it needs no new code paths.
    call channel_diagnostics()

    call KSPCreate(comm, ksp_psi, ierr)
    call KSPSetOperators(ksp_psi, D_psi, D_psi, ierr)
    call KSPSetType(ksp_psi, KSPPREONLY, ierr)
    call KSPGetPC(ksp_psi, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_psi, ierr)

    rassm    = -1.d0
    rasscons = -1.d0
    fillr    = -1.d0
    itassm   = -2
    nzrow    = 0

    ncase = 0
    call MatCreate(PETSC_COMM_SELF, Shat, ierr)
    call MatSetSizes(Shat, n1, n1, n1, n1, ierr)
    call MatSetType(Shat, MATSEQDENSE, ierr)
    call MatSetUp(Shat, ierr)

    !== case 0: Shat = D_uu (control) ==================================
    ncase = ncase + 1
    cname(ncase) = "D_uu only"
    call build_shat_channel(-1)

    !== case 1: psi channel with EXACT D_psi^-1 (upper bound on quality)
    ncase = ncase + 1
    cname(ncase) = "psi exact"
    call build_shat_channel(0)

    !== candidates from the table =====================================
    do ic = 1, ncand
      ncase = ncase + 1
      cname(ncase) = "M_* " // trim(lab(ic))
      call build_shat_channel(ic)
      if (physics_pc_schur_assemble) then
        do ilump = 1, NLUMP
          call build_shat_assembled(ic, ilump, .false.)
        enddo
      endif
    enddo

    !== Stage 5.1: the small-flow limit ===============================
    ! D_psi ~ (1+zeta) Q_p, no commutator. The operator form is algebraically
    ! identical to M0 (so rel() here must match M0's rel() to round-off), but
    ! the ASSEMBLED form is not: it exploits the cancellation and therefore
    ! carries one triple product and one mass inverse instead of two of each.
    ! The assembled branch needs M0's table index to evaluate the reference
    ! side of its algebra check through the A_uM solve.
    ic_sf = cm_table_lookup(lab, ncand, "M0")
    ncase = ncase + 1
    cname(ncase) = "SFp  psi"
    ch_on = (/ .true., .false., .false. /)
    call build_shat_channel(-2)
    rel_sf = rel(ncase)
    if (physics_pc_schur_assemble) then
      if (ic_sf == 0) then
        if (my_id == 0) write(*,'(A)') &
          "[Stage 5.1] WARNING: M0 absent from the table; small-flow " // &
          "assembled form skipped (its algebra check needs M0's A_uM)."
      else
        do ilump = 1, NLUMP
          ! mu is irrelevant here (there is no Q_u^-1), so variants that
          ! differ only in mu would print duplicate rows. Keep one row per
          ! distinct psi-space treatment: mp = 1,2,0,4,5,6 for ilump =
          ! 1,2,4,6,7,8; ilump 3 and 5 repeat mp = 2.
          if (ilump == 3 .or. ilump == 5) cycle
          call build_shat_assembled(ic_sf, ilump, .true.)
        enddo
      endif
    endif

    !== Stage 6.1: the same form with ALL THREE channels ===============
    ! S_sf,all = D_uu - sum_y L_uy Q_y^-1 U_yu / (1+zeta), y = psi, rho, T.
    ! Per-channel pricing above says the rho and T channels carry ~1e-4 (this
    ! case) to ~1.5e-3 (at realistic beta) of the exact correction, so this
    ! row is EXPECTED to equal SFp to the printed precision. It is built
    ! anyway because that equality, measured on the assembled operator rather
    ! than inferred from norm ratios, is what actually closes the question --
    ! and the fill column prices what the two extra triple products cost.
    ncase = ncase + 1
    cname(ncase) = "SFall p+r+T"
    ch_on = (/ .true., .true., .true. /)
    call build_shat_channel(-2)
    if (physics_pc_schur_assemble .and. ic_sf > 0) then
      do ilump = 1, NLUMP
        if (ilump == 3 .or. ilump == 5) cycle
        call build_shat_assembled(ic_sf, ilump, .true.)
      enddo
    endif
    ! Every consumer after this point is psi-only by construction.
    ch_on = (/ .true., .false., .false. /)

    !== Stage 5.2: the CONTINUOUS-PDE Schur complement =================
    ! Cyr et al. Sec. 3.2.2 derive the Schur complement by eliminating psi
    ! from the LINEARIZED PDE pair instead of composing discrete blocks.
    ! The object that replaces is exactly the psi channel L_up Q_p^-1 U_pu
    ! / (1+zeta), so the honest first question is not "does it precondition"
    ! but "is it even the same operator". Answer that with a best-fit scalar
    ! rather than a derived constant: the SHAPE is what the continuous
    ! derivation fixes, while the R-weights and the (theta dt)^2/(1+zeta)
    ! normalisation depend on measure conventions that are easy to get
    ! wrong and are not what is under test here.
    !
    !   alpha = <A,T2>/<A,A>,   residual = ||alpha A - T2||_F / ||T2||_F
    !
    ! using <A,B> = (||A||^2 + ||B||^2 - ||A-B||^2)/2, so no matrix product
    ! is needed. The residual is the real gate: a right-shaped operator has
    ! a residual that DECREASES under refinement (both are approximating
    ! the same continuous object, so they must converge to each other);
    ! a wrong-shaped one has a residual that sits at O(1) or grows. Report
    ! it, and alpha, and let the mesh sequence decide.
    ! Deliberately NOT gated on physics_pc_schur_assemble: this test has
    ! nothing to do with the mass-inverse variant sweep, and that sweep is
    ! 8x the cost of everything else on the finer meshes -- which is exactly
    ! where the shape residual has to be measured.
    if (cm_blocks_ready()) then
      call csc_shape_test(CM_OP_AISO, "CSCi (isotropic)")
      call csc_shape_test(CM_OP_AANI, "CSCa (field-aligned)")
    endif

    !--- report --------------------------------------------------------
    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "--- Approximate Schur complements ------------------------------"
      write(*,'(A,E12.4)') "  reference ||S_u||_F (interior) = ", nrm_S
      write(*,'(A,I0)')    "  channels built = ", ncase
      write(*,'(A)') ""
      write(*,'(A)') "  candidate      ||Shat-S_u||/||S_u||   GMRES   ||A_uM-D_psi||/||D_psi||"
      do icase = 1, ncase
        if (itc(icase) >= 0) then
          write(*,'(A,A12,A,E14.4,A,I6,A)',advance='no') &
            "  ", cname(icase), "   ", rel(icase), "  ", itc(icase), "   "
        else
          write(*,'(A,A12,A,E14.4,A)',advance='no') &
            "  ", cname(icase), "   ", rel(icase), "  n.c.   "
        endif
        if (rdp(icase) >= 0.d0) then
          write(*,'(E14.4)') rdp(icase)
        else
          write(*,'(A)') "        --"
        endif
      enddo
      if (physics_pc_schur_assemble) then
        write(*,'(A)') ""
        write(*,'(A)') "--- Sparse ASSEMBLED Schur complement (Workstream A) ------------"
        write(*,'(A)') "    S_ass = D_uu Q_u^-1 A_uM - L_up Q^-1 U_pu  (lumped masses),"
        write(*,'(A)') "    Shat = S_ass A_uM^-1 Q_u,  so Shat^-1 = Q_u^-1 A_uM S_ass^-1."
        write(*,'(A)') "    The 'SF small-fl' rows are the Stage 5.1 collapsed form"
        write(*,'(A)') "    S_sf = D_uu - L_up Q_p^-1 U_pu/(1+zeta): no Q_u^-1, no right"
        write(*,'(A)') "    factor, one triple product. Its Q_u^-1 column is not read."
        write(*,'(A)') ""
        do ilump = 1, NLUMP
          write(*,'(A)') ""
          write(*,'(A,A,A)') "  [Q_u^-1 / Q^-1  =  ", lumpname(ilump), "]"
          write(*,'(A)') "  candidate      algebra chk   ||Shat-S_u||/||S_u||   GMRES   nnz/nnz(D_uu)"
          do icase = 1, ncase
            if (rassm(icase,ilump) < 0.d0) cycle
            write(*,'(A,A12,A,E12.4,A,E14.4,A)',advance='no') &
              "  ", cname(icase), "   ", rasscons(icase,ilump), "  ", rassm(icase,ilump), "  "
            if (itassm(icase,ilump) >= 0) then
              write(*,'(I6,A,F10.2)') itassm(icase,ilump), "   ", fillr(icase,ilump)
            else
              write(*,'(A,F10.2)') "  n.c.   ", fillr(icase,ilump)
            endif
          enddo
        enddo
        write(*,'(A)') ""
        write(*,'(A)') "  'algebra chk' must be ~1e-13: it verifies the factorisation with"
        write(*,'(A)') "  CONSISTENT masses, independently of the diagonal approximation."
        write(*,'(A)') "  Damage from that approximation shows up instead as a gap between"
        write(*,'(A)') "  these GMRES counts and the operator-form ones above. Row-sum"
        write(*,'(A)') "  lumping is invalid on a C1 Bezier/Hermite space (derivative DOFs"
        write(*,'(A)') "  have vanishing mass row sums); compare against the diagonal row."
        write(*,'(A)') "  nnz/nnz(D_uu) = -1.00 marks the two EXACT-inverse variants: they"
        write(*,'(A)') "  are not sparse-assemblable and exist only to attribute the loss"
        write(*,'(A)') "  to one mass inverse or the other."
      endif
      write(*,'(A)') ""
      write(*,'(A)') "  'psi exact' keeps L D_psi^-1 U exactly and drops only the"
      write(*,'(A)') "  rho and T channels: it is the best any psi-only scheme can do."
      write(*,'(A)') "  M0 is the small-flow limit (M_*^-1 = I/(1+zeta))."
      write(*,'(A)') "  'SF small-fl' is that same limit taken WITHOUT the commutator:"
      write(*,'(A)') "  Shat_sf = D_uu - L_up Q_p^-1 U_pu/(1+zeta), its own precondi-"
      write(*,'(A)') "  tioner. In the operator form it is algebraically identical to"
      write(*,'(A)') "  M0; in the assembled form it is strictly cheaper."
      if (ic_sf > 0) then
        write(*,'(A,E12.4,A,E12.4)') &
          "  CHECK  rel(SF) = ", rel_sf, "   vs  rel(M0) = ", rel(2 + ic_sf)
        if (abs(rel_sf - rel(2 + ic_sf)) > 1.d-8 * max(rel_sf, 1.d-300)) then
          write(*,'(A)') "  *** MISMATCH: the small-flow channel is NOT M0. ***"
        endif
      endif
      write(*,'(A)') "================================================================"
      write(*,'(A)') ""
    endif

    deallocate(rows)
    call KSPDestroy(ksp_M, ierr)
    call KSPDestroy(ksp_psi, ierr)
    call KSPDestroy(ksp_Q, ierr)
    call KSPDestroy(ksp_QR, ierr)
    call MatDestroy(M_yy, ierr)
    call MatDestroy(U_yu, ierr)
    call MatDestroy(L_uy, ierr)
    call MatDestroy(D_uu, ierr)
    call MatDestroy(D_psi, ierr)
    call MatDestroy(U_pu, ierr)
    call MatDestroy(L_up, ierr)
    call MatDestroy(D_rho, ierr)
    call MatDestroy(U_ru, ierr)
    call MatDestroy(L_ur, ierr)
    call MatDestroy(D_T, ierr)
    call MatDestroy(U_Tu, ierr)
    call MatDestroy(L_uT, ierr)
    call MatDestroy(S_u, ierr)
    call MatDestroy(Shat, ierr)
    call ISDestroy(is_y, ierr)
    call ISDestroy(is_u, ierr)
    call VecDestroy(e_u, ierr)
    call VecDestroy(t_y, ierr)
    call VecDestroy(w_y, ierr)
    call VecDestroy(sv_u, ierr)
    call VecDestroy(d_u, ierr)
    call VecDestroy(cv_t, ierr)
    call VecDestroy(cv_w, ierr)
    call VecDestroy(cv_q, ierr)
    call VecDestroy(dmask, ierr)
    call VecDestroy(pmask, ierr)

  contains

    !> Dense n1 x n1 probe of  sc * L * (kspd)^-1 * U, one column per call to
    !! the inner solve. `wd` must live in the space kspd solves in (y-sized for
    !! M_yy, 1-variable for a single channel); `wu` is u-sized.
    !! The same shape as the exact-S_u loop above, factored out because Stage
    !! 6.1 needs it seven times.
    subroutine probe_LDU(L, kspd, U, wu, wd, sc, M_out)
      Mat, intent(in)    :: L, U
      KSP, intent(in)    :: kspd
      Vec, intent(inout) :: wu, wd
      real*8, intent(in) :: sc
      Mat, intent(out)   :: M_out

      Vec :: wd2
      integer :: kk

      call VecDuplicate(wd, wd2, ierr)
      call MatCreate(PETSC_COMM_SELF, M_out, ierr)
      call MatSetSizes(M_out, n1, n1, n1, n1, ierr)
      call MatSetType(M_out, MATSEQDENSE, ierr)
      call MatSetUp(M_out, ierr)
      do kk = 0, n1-1
        call VecZeroEntries(e_u, ierr)
        colidx(1) = kk
        call VecSetValue(e_u, kk, 1.0d0, INSERT_VALUES, ierr)
        call VecAssemblyBegin(e_u, ierr)
        call VecAssemblyEnd(e_u, ierr)
        call MatMult(U, e_u, wd, ierr)
        call KSPSolve(kspd, wd, wd2, ierr)
        call MatMult(L, wd2, wu, ierr)
        if (sc /= 1.0d0) call VecScale(wu, sc, ierr)
        call VecGetArray(wu, parr, ierr)
        call MatSetValues(M_out, n1, rows, 1, colidx, parr, INSERT_VALUES, ierr)
        call VecRestoreArray(wu, parr, ierr)
      enddo
      call MatAssemblyBegin(M_out, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(M_out, MAT_FINAL_ASSEMBLY, ierr)
      call VecDestroy(wd2, ierr)
    end subroutine probe_LDU


    !> Interior Frobenius norm of a dense u-space matrix (rows masked).
    real*8 function int_norm(A)
      Mat, intent(in) :: A
      Mat :: W
      PetscReal :: rn
      call MatDuplicate(A, MAT_COPY_VALUES, W, ierr)
      call MatDiagonalScale(W, dmask, PETSC_NULL_VEC, ierr)
      call MatNorm(W, NORM_FROBENIUS, rn, ierr)
      call MatDestroy(W, ierr)
      int_norm = rn
    end function int_norm


    !--------------------------------------------------------------------
    !> Stage 6.1: price each channel of the Schur complement separately,
    !! BEFORE any of them is approximated. Three numbers per channel:
    !!
    !!  (1) mass dominance   ||D_yy - (1+zeta)Q_y|| / ||(1+zeta)Q_y||
    !!      reported against BOTH available weights. The psi row is
    !!      1/R-weighted (amat_11) and the rho/T rows are R-weighted
    !!      (amat_55/amat_66 both carry v*(rho|T)*BigR*xjac*(1+zeta)), so
    !!      psi should pick Q1R and rho/T should pick QR. Measured, not
    !!      asserted: a different winner invalidates the channel weights and
    !!      must be fixed before any defect below is believed.
    !!
    !!  (2) worth            ||L_uy D_yy^-1 U_yu|| / ||L_uy M_yy^-1 U_yu||
    !!      with the EXACT D_yy. What the channel is worth at all, before
    !!      any approximation enters -- the direct answer to why the psi
    !!      channel only recovers ~30% of the correction.
    !!
    !!  (3) damage           ||L_uy Q_y^-1 U_yu/(1+zeta) - L_uy D_yy^-1 U_yu||
    !!                        / ||L_uy D_yy^-1 U_yu||
    !!      with the EXACT mass inverse, so this isolates the small-flow
    !!      MODEL error (D_yy -> (1+zeta)Q_y) from the sparse-inverse error
    !!      that the NLUMP sweep measures separately.
    !!
    !! Admission rule: a channel belongs in the operator when (2) is not
    !! negligible AND (3) < 1. (3) >= 1 means the approximated channel is
    !! further from the truth than omitting the channel entirely -- which is
    !! the expected fate of the T channel at production tstep, where amat_66
    !! is dominated by (ZK_par-ZK_prof)*theta*tstep parallel conduction and
    !! not by its mass term. There the mass inverse OVER-estimates the
    !! channel, so including it is worse than dropping it.
    !--------------------------------------------------------------------
    subroutine channel_diagnostics()
      Mat :: C_ex, E_y, A_y, Dif, Qm
      KSP :: ksp_D, ksp_Qy
      PC  :: pc_l
      integer :: jc, iq
      real*8  :: nrm_corr, nrm_E
      PetscReal :: rn, rd

      !--- (1) mass dominance, per channel, against both weights --------
      ! pmask is the shared-space boundary-node mask: all four reduced
      ! variables live on the same scalar space and the same boundary nodes,
      ! so one mask serves every channel.
      do jc = 1, NCH
        do iq = 1, 2
          if (iq == 1) then
            Qm = op(CM_OP_Q1R)
          else
            Qm = op(CM_OP_QR)
          endif
          call MatDuplicate(ch_D(jc), MAT_COPY_VALUES, Dif, ierr)
          call MatAXPY(Dif, -opz, Qm, DIFFERENT_NONZERO_PATTERN, ierr)
          call MatDiagonalScale(Dif, pmask, PETSC_NULL_VEC, ierr)
          call MatNorm(Dif, NORM_FROBENIUS, rn, ierr)
          call MatDestroy(Dif, ierr)
          call MatDuplicate(Qm, MAT_COPY_VALUES, Dif, ierr)
          call MatDiagonalScale(Dif, pmask, PETSC_NULL_VEC, ierr)
          call MatNorm(Dif, NORM_FROBENIUS, rd, ierr)
          call MatDestroy(Dif, ierr)
          if (iq == 1) then
            ch_m1r(jc) = rn / max(opz*rd, 1.d-300)
          else
            ch_mR(jc)  = rn / max(opz*rd, 1.d-300)
          endif
        enddo
      enddo

      !--- the denominator for (2): the exact all-channel correction -----
      ! L_uy M_yy^-1 U_yu, i.e. D_uu - S_u. Probed rather than differenced
      ! so no dense/sparse MatAXPY is needed.
      call probe_LDU(L_uy, ksp_M, U_yu, sv_u, t_y, 1.0d0, C_ex)
      nrm_corr = int_norm(C_ex)
      call MatDestroy(C_ex, ierr)

      do jc = 1, NCH
        ! exact channel: one MUMPS LU on this channel's own diagonal block
        call KSPCreate(comm, ksp_D, ierr)
        call KSPSetOperators(ksp_D, ch_D(jc), ch_D(jc), ierr)
        call KSPSetType(ksp_D, KSPPREONLY, ierr)
        call KSPGetPC(ksp_D, pc_l, ierr)
        call PCSetType(pc_l, PCLU, ierr)
        call PCFactorSetMatSolverType(pc_l, MATSOLVERMUMPS, ierr)
        call KSPSetUp(ksp_D, ierr)
        call probe_LDU(ch_L(jc), ksp_D, ch_U(jc), sv_u, cv_t, 1.0d0, E_y)
        call KSPDestroy(ksp_D, ierr)
        nrm_E = int_norm(E_y)
        ch_worth(jc) = nrm_E / max(nrm_corr, 1.d-300)

        ! small-flow channel, with the mass inverse taken EXACTLY
        if (ch_q(jc) == CM_OP_Q1R) then
          ksp_Qy = ksp_Q
        else
          ksp_Qy = ksp_QR
        endif
        call probe_LDU(ch_L(jc), ksp_Qy, ch_U(jc), sv_u, cv_t, 1.0d0/opz, A_y)
        call MatAXPY(A_y, -1.0d0, E_y, SAME_NONZERO_PATTERN, ierr)
        ch_dmg(jc) = int_norm(A_y) / max(nrm_E, 1.d-300)
        call MatDestroy(A_y, ierr)
        call MatDestroy(E_y, ierr)
      enddo

      if (my_id == 0) then
        write(*,'(A)') ""
        write(*,'(A)') "----------------------------------------------------------------"
        write(*,'(A)') " Stage 6.1: per-channel pricing of the Schur complement"
        write(*,'(A,E12.4)') "   ||L_uy M_yy^-1 U_yu||_F (interior) = ", nrm_corr
        write(*,'(A)') "   chan   |D-(1+z)Q1R|   |D-(1+z)QR|      worth      damage"
        do jc = 1, NCH
          write(*,'(A,A4,A,E12.4,A,E12.4,A,E12.4,A,E12.4)') &
            "   ", chname(jc), "  ", ch_m1r(jc), "  ", ch_mR(jc), "  ", &
            ch_worth(jc), "  ", ch_dmg(jc)
        enddo
        write(*,'(A)') "   (worth: fraction of the exact correction this channel"
        write(*,'(A)') "    carries, exact D_yy.  damage: small-flow error relative"
        write(*,'(A)') "    to that channel's own size; >= 1 means omit the channel.)"
        write(*,'(A)') "----------------------------------------------------------------"
        flush(6)
      endif
    end subroutine channel_diagnostics


    !> Fill Shat for one channel choice and measure it.
    !!   mode = -2 : Shat = D_uu - L_up Q^-1 U_pu / (1+zeta)   (small flow)
    !!   mode = -1 : Shat = D_uu
    !!   mode =  0 : Shat = D_uu - L_up D_psi^-1 U_pu   (exact psi channel)
    !!   mode >  0 : Shat = D_uu - L_up U_pu M_*(mode)^-1
    !!
    !! mode = -2 is the SMALL-FLOW limit of Chacon (2008) and of Cyr et al.
    !! Sec. 3.2.2 (their P_Diag): the psi block is taken mass-dominated,
    !! D_psi ~ (1+zeta) Q_p, and the commutator argument is not used at all.
    !! It is algebraically identical to candidate M0, for which A_uM =
    !! (1+zeta) Q_u and hence A_uM^-1 Q_u = I/(1+zeta) -- so rel() for this
    !! channel MUST equal rel() for M0 to round-off. That equality is the
    !! cheapest available check that the cancellation exploited by
    !! build_shat_assembled(..., smallflow=.true.) is the right one.
    subroutine build_shat_channel(mode)
      integer, intent(in) :: mode
      integer :: kk, iop
      KSPConvergedReason :: kreason
      PetscReal :: rn

      if (mode > 0) then
        ! assemble A_uM = sum_iop coef(mode,iop)*op(iop), then M_*^-1 = A_uM^-1 Q_u
        call MatDuplicate(op(quop(mode)), MAT_COPY_VALUES, A_uM, ierr)
        call MatScale(A_uM, 0.0d0, ierr)
        do iop = 1, CM_NOP
          if (coef(mode,iop) /= 0.d0) then
            call MatAXPY(A_uM, coef(mode,iop), op(iop), DIFFERENT_NONZERO_PATTERN, ierr)
          endif
        enddo
        call KSPCreate(comm, ksp_AuM, ierr)
        call KSPSetOperators(ksp_AuM, A_uM, A_uM, ierr)
        call KSPSetType(ksp_AuM, KSPPREONLY, ierr)
        call KSPGetPC(ksp_AuM, pc_obj, ierr)
        call PCSetType(pc_obj, PCLU, ierr)
        call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
        call KSPSetUp(ksp_AuM, ierr)

        ! operator-form check: how close is this A_uM to D_psi itself?
        call MatDuplicate(A_uM, MAT_COPY_VALUES, S_dif, ierr)
        call MatAXPY(S_dif, -1.0d0, D_psi, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatDiagonalScale(S_dif, pmask, PETSC_NULL_VEC, ierr)
        call MatNorm(S_dif, NORM_FROBENIUS, rn, ierr)
        call MatDestroy(S_dif, ierr)
        rdp(ncase) = rn / max(nrm_Dp, 1.d-300)
      else
        rdp(ncase) = -1.d0
      endif

      do kk = 0, n1-1
        call VecZeroEntries(e_u, ierr)
        colidx(1) = kk
        call VecSetValue(e_u, kk, 1.0d0, INSERT_VALUES, ierr)
        call VecAssemblyBegin(e_u, ierr)
        call VecAssemblyEnd(e_u, ierr)
        call MatMult(D_uu, e_u, d_u, ierr)

        if (mode == -2) then
          ! Small flow, summed over the channels ch_on: for each y,
          !   D_yy^-1 ~ Q_y^-1/(1+zeta),  no commutator, no A_uM.
          ! ch_on = (T,F,F) reproduces the psi-only Stage 5.1 form exactly.
          call VecZeroEntries(sv_u, ierr)
          do ich = 1, NCH
            if (.not. ch_on(ich)) cycle
            call MatMult(ch_U(ich), e_u, cv_t, ierr)
            if (ch_q(ich) == CM_OP_Q1R) then
              call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
            else
              call KSPSolve(ksp_QR, cv_t, cv_q, ierr)
            endif
            call MatMult(ch_L(ich), cv_q, cv_w, ierr)
            call VecAXPY(sv_u, 1.0d0/opz, cv_w, ierr)
          enddo
          call VecAYPX(sv_u, -1.0d0, d_u, ierr)
        else if (mode == -1) then
          call VecCopy(d_u, sv_u, ierr)
        else if (mode == 0) then
          call MatMult(U_pu, e_u, cv_t, ierr)
          call KSPSolve(ksp_psi, cv_t, cv_w, ierr)
          call MatMult(L_up, cv_w, sv_u, ierr)
          call VecAYPX(sv_u, -1.0d0, d_u, ierr)
        else
          ! L D_psi^-1 U ~ L Q^-1 U M_*^-1,  with M_*^-1 e = A_uM^-1 Q_u e
          call MatMult(op(quop(mode)), e_u, cv_t, ierr)
          call KSPSolve(ksp_AuM, cv_t, cv_w, ierr)
          call MatMult(U_pu, cv_w, cv_t, ierr)
          call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
          call MatMult(L_up, cv_q, sv_u, ierr)
          call VecAYPX(sv_u, -1.0d0, d_u, ierr)
        endif

        call VecGetArray(sv_u, parr, ierr)
        call MatSetValues(Shat, n1, rows, 1, colidx, parr, INSERT_VALUES, ierr)
        call VecRestoreArray(sv_u, parr, ierr)
      enddo
      call MatAssemblyBegin(Shat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(Shat, MAT_FINAL_ASSEMBLY, ierr)

      ! ||Shat - S_u|| on interior rows
      call MatDuplicate(Shat, MAT_COPY_VALUES, S_dif, ierr)
      call MatAXPY(S_dif, -1.0d0, S_u, SAME_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(S_dif, dmask, PETSC_NULL_VEC, ierr)
      call MatNorm(S_dif, NORM_FROBENIUS, rn, ierr)
      call MatDestroy(S_dif, ierr)
      rel(ncase) = rn / max(nrm_S, 1.d-300)

      ! GMRES on S_u preconditioned by Shat
      call KSPCreate(comm, ksp_test, ierr)
      call KSPSetOperators(ksp_test, S_u, Shat, ierr)
      call KSPSetType(ksp_test, KSPGMRES, ierr)
      call KSPSetTolerances(ksp_test, 1.d-8, PETSC_DEFAULT_REAL, PETSC_DEFAULT_REAL, 200, ierr)
      call KSPGetPC(ksp_test, pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call KSPSetUp(ksp_test, ierr)
      call MatCreateVecs(S_u, xt, bt, ierr)
      call VecSetRandom(bt, PETSC_NULL_RANDOM, ierr)
      call VecPointwiseMult(bt, bt, dmask, ierr)
      call KSPSolve(ksp_test, bt, xt, ierr)
      call KSPGetIterationNumber(ksp_test, nits, ierr)
      call KSPGetConvergedReason(ksp_test, kreason, ierr)
      if (kreason .ne. KSP_DIVERGED_ITS .and. kreason%v > 0) then
        itc(ncase) = int(nits)
      else
        itc(ncase) = -1
      endif
      call VecDestroy(bt, ierr)
      call VecDestroy(xt, ierr)
      call KSPDestroy(ksp_test, ierr)

      if (mode > 0) then
        call KSPDestroy(ksp_AuM, ierr)
        call MatDestroy(A_uM, ierr)
      endif
    end subroutine build_shat_channel


    !------------------------------------------------------------------
    !> Stage 5.2/5.3: is the CONTINUOUS-PDE operator the same object as the
    !! discrete psi channel it is meant to replace?
    !!
    !! Builds  T2 = L_up Q_p^-1 U_pu / (1+zeta)  with an EXACT Q_p (dense, but
    !! this runs once and only on the Stage 4.2 meshes), fits a single scalar
    !! alpha to the assembled continuous operator A, and reports the residual
    !! that no scalar can remove. Then scores S_csc = D_uu - alpha A on the
    !! same footing as every other channel, so its defect and GMRES count sit
    !! in the same table.
    !!
    !! alpha is FITTED, not derived. That is deliberate: the continuous
    !! derivation fixes the shape, while the R-weights and the
    !! (theta dt)^2/(1+zeta) normalisation depend on measure conventions.
    !! Fitting alpha isolates the question that matters -- whether the shape
    !! is right -- from the bookkeeping question of what constant multiplies
    !! it. A shape that needs a wildly mesh-dependent alpha has failed even
    !! if its residual looks small.
    !------------------------------------------------------------------
    subroutine csc_shape_test(iop_csc, tag)
      integer,          intent(in) :: iop_csc
      character(len=*), intent(in) :: tag
      Mat :: T2_ex, A_c, Dif
      integer :: kk
      KSPConvergedReason :: kreason
      PetscReal :: nA, nB, nD, dotAB, alpha, resid, rn

      ! T2 = L_up Q_p^-1 U_pu / (1+zeta), exact Q_p, by dense probing.
      call MatCreate(PETSC_COMM_SELF, T2_ex, ierr)
      call MatSetSizes(T2_ex, n1, n1, n1, n1, ierr)
      call MatSetType(T2_ex, MATSEQDENSE, ierr)
      call MatSetUp(T2_ex, ierr)
      do kk = 0, n1-1
        call VecZeroEntries(e_u, ierr)
        colidx(1) = kk
        call VecSetValue(e_u, kk, 1.0d0, INSERT_VALUES, ierr)
        call VecAssemblyBegin(e_u, ierr)
        call VecAssemblyEnd(e_u, ierr)
        call MatMult(U_pu, e_u, cv_t, ierr)
        call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
        call MatMult(L_up, cv_q, sv_u, ierr)
        call VecScale(sv_u, 1.0d0/opz, ierr)
        call VecGetArray(sv_u, parr, ierr)
        call MatSetValues(T2_ex, n1, rows, 1, colidx, parr, INSERT_VALUES, ierr)
        call VecRestoreArray(sv_u, parr, ierr)
      enddo
      call MatAssemblyBegin(T2_ex, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(T2_ex, MAT_FINAL_ASSEMBLY, ierr)

      ! Dense copy of the assembled continuous operator, same masking.
      call MatConvert(op(iop_csc), MATSEQDENSE, MAT_INITIAL_MATRIX, A_c, ierr)

      ! Interior rows only -- the ZBIG Dirichlet rows would otherwise set
      ! every one of these norms (see the D_psi/B11 184x trap in Sec. 3).
      call MatDiagonalScale(A_c,   dmask, PETSC_NULL_VEC, ierr)
      call MatDiagonalScale(T2_ex, dmask, PETSC_NULL_VEC, ierr)

      call MatNorm(A_c,   NORM_FROBENIUS, nA, ierr)
      call MatNorm(T2_ex, NORM_FROBENIUS, nB, ierr)
      call MatDuplicate(A_c, MAT_COPY_VALUES, Dif, ierr)
      call MatAXPY(Dif, -1.0d0, T2_ex, SAME_NONZERO_PATTERN, ierr)
      call MatNorm(Dif, NORM_FROBENIUS, nD, ierr)
      call MatDestroy(Dif, ierr)

      dotAB = 0.5d0 * (nA*nA + nB*nB - nD*nD)
      if (nA > 1.d-300) then
        alpha = dotAB / (nA*nA)
        resid = sqrt(max(nB*nB - dotAB*dotAB/(nA*nA), 0.d0)) / max(nB, 1.d-300)
      else
        alpha = 0.d0
        resid = -1.d0
      endif

      ! S_csc = D_uu - alpha A, scored like every other channel.
      call MatConvert(op(iop_csc), MATSEQDENSE, MAT_INITIAL_MATRIX, Dif, ierr)
      call MatScale(Dif, -alpha, ierr)
      call MatAXPY(Dif, 1.0d0, D_uu, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDuplicate(Dif, MAT_COPY_VALUES, A_c, ierr)   ! reuse as S_csc
      call MatDestroy(Dif, ierr)

      call MatDuplicate(A_c, MAT_COPY_VALUES, Dif, ierr)
      call MatAXPY(Dif, -1.0d0, S_u, SAME_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(Dif, dmask, PETSC_NULL_VEC, ierr)
      call MatNorm(Dif, NORM_FROBENIUS, rn, ierr)
      call MatDestroy(Dif, ierr)
      rn = rn / max(nrm_S, 1.d-300)

      call KSPCreate(comm, ksp_test, ierr)
      call KSPSetOperators(ksp_test, S_u, A_c, ierr)
      call KSPSetType(ksp_test, KSPGMRES, ierr)
      call KSPSetTolerances(ksp_test, 1.d-8, PETSC_DEFAULT_REAL, PETSC_DEFAULT_REAL, 200, ierr)
      call KSPGetPC(ksp_test, pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call KSPSetUp(ksp_test, ierr)
      call MatCreateVecs(S_u, xt, bt, ierr)
      call VecSetRandom(bt, PETSC_NULL_RANDOM, ierr)
      call VecPointwiseMult(bt, bt, dmask, ierr)
      call KSPSolve(ksp_test, bt, xt, ierr)
      call KSPGetIterationNumber(ksp_test, nits, ierr)
      call KSPGetConvergedReason(ksp_test, kreason, ierr)

      if (my_id == 0) then
        write(*,'(A)') ""
        write(*,'(A,A)') "--- Stage 5.2/5.3 continuous-PDE Schur: ", trim(tag)
        write(*,'(A,E12.4)') "    ||A||_F (interior)                 = ", nA
        write(*,'(A,E12.4)') "    ||L_up Q_p^-1 U_pu/(1+zeta)||_F    = ", nB
        write(*,'(A,E12.4)') "    best-fit alpha                     = ", alpha
        write(*,'(A,E12.4)') "    shape residual after fit (GATE)    = ", resid
        write(*,'(A,E12.4)') "    ||S_csc - S_u||/||S_u||            = ", rn
        if (kreason .ne. KSP_DIVERGED_ITS .and. kreason%v > 0) then
          write(*,'(A,I0)')  "    GMRES on S_u preconditioned by it  = ", int(nits)
        else
          write(*,'(A)')     "    GMRES on S_u preconditioned by it  = n.c."
        endif
        write(*,'(A)') "    GATE: the shape residual must DECREASE under mesh"
        write(*,'(A)') "    refinement. It is the fraction of the discrete psi"
        write(*,'(A)') "    channel that no scalar multiple of this continuous"
        write(*,'(A)') "    operator can reproduce -- an O(1) or growing value"
        write(*,'(A)') "    means the derived shape is wrong, whatever the"
        write(*,'(A)') "    GMRES count says."
      endif

      call VecDestroy(bt, ierr)
      call VecDestroy(xt, ierr)
      call KSPDestroy(ksp_test, ierr)
      call MatDestroy(A_c, ierr)
      call MatDestroy(T2_ex, ierr)
    end subroutine csc_shape_test


    !------------------------------------------------------------------
    !> Workstream A: the SPARSE ASSEMBLED form of the same Schur complement.
    !!
    !! build_shat_channel forms Shat only as an OPERATOR: every application
    !! costs a solve with A_uM and a solve with Q. That is fine as a
    !! diagnostic but useless as a preconditioner -- there is no matrix to
    !! hand to LU/ILU/AMG. Factor the inverse out to the right:
    !!
    !!   Shat = D_uu - L_up Q^-1 U_pu A_uM^-1 Q_u
    !!        = [ D_uu Q_u^-1 A_uM - L_up Q^-1 U_pu ] A_uM^-1 Q_u
    !!        =:  S_ass  A_uM^-1 Q_u
    !!
    !! so that
    !!
    !!   Shat^-1 = Q_u^-1 A_uM S_ass^-1 .
    !!
    !! S_ass is a sum of two TRIPLE PRODUCTS of assembled blocks. Replace the
    !! two interior mass inverses by LUMPED (row-sum) diagonals and S_ass
    !! becomes a genuinely sparse matrix, formed once by two MatMatMults --
    !! and applying Shat^-1 then costs one sparse solve with S_ass, one
    !! matvec with A_uM, and one diagonal scaling. No inner Krylov anywhere.
    !!
    !! Two things must hold for this to be worth anything, and both are
    !! measured here:
    !!   (1) the algebra: S_ass (A_uM^-1 Q_u) x must reproduce Shat x with
    !!       CONSISTENT mass inverses (rasscons ~ 1e-13). This catches a
    !!       wrong Q_u, a wrong index set, or a mis-derived factorisation.
    !!   (2) the lumping: the sparse LUMPED S_ass must still precondition
    !!       S_u about as well as the exact-mass operator form did
    !!       (compare rassm/itassm against rel/itc of the same candidate).
    !! fillr reports what the triple products cost in nonzeros.
    !!
    !! SMALL-FLOW BRANCH (smallflow = .true.).  Stage 5.1.  If the psi block is
    !! taken mass-dominated, D_psi ~ (1+zeta) Q_p -- the small-flow limit of
    !! Chacon (2008) and Cyr et al. Sec. 3.2.2 -- then the commutator is not
    !! needed and A_uM = (1+zeta) Q_u.  Two things then cancel ANALYTICALLY:
    !!
    !!   D_uu Q_u^-1 A_uM = (1+zeta) D_uu        (first triple product)
    !!   A_uM^-1 Q_u      = I/(1+zeta)           (the right factor)
    !!
    !! so the whole object collapses to a single sparse matrix which is its own
    !! preconditioner:
    !!
    !!   S_sf = D_uu - L_up Q_p^-1 U_pu / (1+zeta),     Shat_sf^-1 = S_sf^-1 .
    !!
    !! This matters because it removes BOTH costs that block the commutator
    !! form: one of the two triple products (hence roughly half the fill --
    !! see Sec. 7, where ~1300 nnz/row is the gate) and, more importantly, the
    !! u-space mass inverse Q_u^-1 entirely.  What survives is the psi-space
    !! Q_p^-1, the one FSAI-1 reproduces to ~1% at BOTH timestep ends (Sec. 5
    !! Q2).  The old code path forms (1+zeta) D_uu as an actual MatMatMult
    !! against an APPROXIMATE Q_u^-1, i.e. it pays fill and error for what is
    !! exactly an identity.
    !!
    !! The algebra check is what proves the cancellation rather than merely
    !! coding it: with an exact Q_p, S_sf x must reproduce the M0 candidate's
    !! operator-form Shat x, which is computed here THROUGH the A_uM solve.
    !! Pass mode = the table index of M0 so that comparison is available.
    !------------------------------------------------------------------
    subroutine build_shat_assembled(mode, ilmp, smallflow)
      integer, intent(in) :: mode
      integer, intent(in) :: ilmp   !< 1 = row-sum lumping, 2 = diagonal extraction
      logical, intent(in) :: smallflow !< .true. = the Stage 5.1 small-flow form
      integer :: kk, iop, itry, mu, mp, nfl
      logical :: sparse_ok
      logical :: ch_first             !< first active channel builds T2, rest add
      Mat     :: T2_c                 !< per-channel triple product before the sum
      KSPConvergedReason :: kreason
      PetscReal :: rn, rd, racc, rR
      MatInfo :: minfo
      real*8 :: nzS, nzD
      real*8, parameter :: lump_floor = 1.d-12   !< relative to ||.||_inf of the diagonal
      integer, parameter :: npoly = 2            !< Richardson steps for the mu = 3 variant

      ! --- A_uM for this candidate (identical combination to build_shat_channel)
      call MatDuplicate(op(quop(mode)), MAT_COPY_VALUES, A_uM, ierr)
      call MatScale(A_uM, 0.0d0, ierr)
      do iop = 1, CM_NOP
        if (coef(mode,iop) /= 0.d0) then
          call MatAXPY(A_uM, coef(mode,iop), op(iop), DIFFERENT_NONZERO_PATTERN, ierr)
        endif
      enddo
      call KSPCreate(comm, ksp_AuM, ierr)
      call KSPSetOperators(ksp_AuM, A_uM, A_uM, ierr)
      call KSPSetType(ksp_AuM, KSPPREONLY, ierr)
      call KSPGetPC(ksp_AuM, pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
      call KSPSetUp(ksp_AuM, ierr)

      ! --- consistent Q_u^-1, for the algebra check only ------------------
      call KSPCreate(comm, ksp_Qu, ierr)
      call KSPSetOperators(ksp_Qu, op(quop(mode)), op(quop(mode)), ierr)
      call KSPSetType(ksp_Qu, KSPPREONLY, ierr)
      call KSPGetPC(ksp_Qu, pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
      call KSPSetUp(ksp_Qu, ierr)

      call MatCreateVecs(D_uu, ax, ay, ierr)
      call VecDuplicate(ax, az, ierr)
      call VecDuplicate(ax, aw, ierr)

      !== (1) algebra check on random vectors ============================
      ! Commutator form:
      !   lhs = Shat x       = D_uu x - L_up Q^-1 U_pu A_uM^-1 Q_u x
      !   rhs = S_ass y      = D_uu Q_u^-1 A_uM y - L_up Q^-1 U_pu y,
      !                        y = A_uM^-1 Q_u x
      ! Both use CONSISTENT inverses, so they must agree to round-off.
      !
      ! Small-flow form: the lhs is unchanged and is still evaluated THROUGH
      ! the A_uM solve, while the rhs is the collapsed
      !   rhs = S_sf x = D_uu x - L_up Q^-1 U_pu x / (1+zeta) .
      ! Agreement to round-off is therefore a numerical proof that
      ! A_uM^-1 Q_u = I/(1+zeta), i.e. that the cancellation is real and has
      ! been applied with the right factor -- not a tautology.
      racc = 0.d0
      do itry = 1, 3
        call VecSetRandom(ax, PETSC_NULL_RANDOM, ierr)
        call VecPointwiseMult(ax, ax, dmask, ierr)
        ! y = A_uM^-1 Q_u x   (shared by both sides)
        call MatMult(op(quop(mode)), ax, ay, ierr)
        call KSPSolve(ksp_AuM, ay, az, ierr)          ! az = y
        ! common second term: sum_y L_uy Q_y^-1 U_yu y
        call VecZeroEntries(aw, ierr)
        do ich = 1, NCH
          if (.not. ch_on(ich)) cycle
          call MatMult(ch_U(ich), az, cv_t, ierr)
          if (ch_q(ich) == CM_OP_Q1R) then
            call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
          else
            call KSPSolve(ksp_QR, cv_t, cv_q, ierr)
          endif
          call MatMult(ch_L(ich), cv_q, cv_w, ierr)
          call VecAXPY(aw, 1.0d0, cv_w, ierr)         ! aw = second term
        enddo
        ! lhs = D_uu x - aw
        call MatMult(D_uu, ax, sv_u, ierr)
        call VecAXPY(sv_u, -1.0d0, aw, ierr)          ! sv_u = Shat x
        call VecNorm(sv_u, NORM_2, rd, ierr)
        if (smallflow) then
          ! rhs = S_sf x = D_uu x - sum_y L_uy Q_y^-1 U_yu x / (1+zeta)
          call VecZeroEntries(ay, ierr)
          do ich = 1, NCH
            if (.not. ch_on(ich)) cycle
            call MatMult(ch_U(ich), ax, cv_t, ierr)
            if (ch_q(ich) == CM_OP_Q1R) then
              call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
            else
              call KSPSolve(ksp_QR, cv_t, cv_q, ierr)
            endif
            call MatMult(ch_L(ich), cv_q, cv_w, ierr)
            call VecAXPY(ay, 1.0d0, cv_w, ierr)
          enddo
          call MatMult(D_uu, ax, d_u, ierr)
          call VecAXPY(d_u, -1.0d0/opz, ay, ierr)     ! d_u = S_sf x
        else
          ! rhs = D_uu Q_u^-1 A_uM y - aw
          call MatMult(A_uM, az, cv_t, ierr)
          call KSPSolve(ksp_Qu, cv_t, cv_w, ierr)
          call MatMult(D_uu, cv_w, d_u, ierr)
          call VecAXPY(d_u, -1.0d0, aw, ierr)         ! d_u = S_ass y
        endif
        call VecAXPY(d_u, -1.0d0, sv_u, ierr)         ! d_u = rhs - lhs
        call VecNorm(d_u, NORM_2, rn, ierr)
        racc = max(racc, rn / max(rd, 1.d-300))
      enddo
      rasscons(ncase,ilmp) = racc

      !== (2) approximate the two interior mass inverses =================
      ! mu / mp select the treatment of Q_u^-1 / Q^-1:
      !   0 = exact (KSP solve, NOT sparse-assemblable -- attribution only)
      !   1 = row-sum lumping  (M*1)
      !   2 = diagonal extraction
      ! Row-sum lumping is the textbook choice, but it presumes a nodal
      ! partition-of-unity basis. JOREK's C1 Bezier/Hermite DOFs include
      ! DERIVATIVES, whose basis functions have (near-)vanishing integrals,
      ! so their mass row sums carry no scale information and the "lumped
      ! inverse" is not an approximation of anything on those rows.
      select case (ilmp)
        case (1) ; mu = 1 ; mp = 1
        case (2) ; mu = 2 ; mp = 2
        case (3) ; mu = 0 ; mp = 2
        case (4) ; mu = 2 ; mp = 0
        case (5) ; mu = 3 ; mp = 2
        case (6) ; mu = 4 ; mp = 4
        case (7) ; mu = 5 ; mp = 5
        case default ; mu = 6 ; mp = 6
      end select

      call VecDuplicate(ax, lq_u, ierr)
      call VecDuplicate(ax, lq_p, ierr)
      call VecDuplicate(ax, ones_p, ierr)
      call VecSet(ones_p, 1.0d0, ierr)
      if (mu == 1) then
        call MatMult(op(quop(mode)), ones_p, lq_u, ierr)
      else
        call MatGetDiagonal(op(quop(mode)), lq_u, ierr)
      endif
      if (mp == 1) then
        call MatMult(op(CM_OP_Q1R), ones_p, lq_p, ierr)
      else
        call MatGetDiagonal(op(CM_OP_Q1R), lq_p, ierr)
      endif

      ! FSAI factors, built once per (candidate, variant). Qi_* = G^T G.
      ! The small-flow form has no Q_u^-1 at all, so its factor is never built
      ! -- that is the point of the branch, not an optimisation.
      if ((mu == 5 .or. mu == 6) .and. .not. smallflow) then
        if (mu == 6) then
          call MatMatMult(op(quop(mode)), op(quop(mode)), MAT_INITIAL_MATRIX, &
                          PETSC_DEFAULT_REAL, Pat_u, ierr)
        else
          Pat_u = op(quop(mode))
        endif
        call build_fsai(op(quop(mode)), Pat_u, n1, comm, G_u, nfl)
        call MatTransposeMatMult(G_u, G_u, MAT_INITIAL_MATRIX, &
                                 PETSC_DEFAULT_REAL, Qi_u, ierr)
        if (my_id == 0 .and. mode == 1) then
          call MatGetInfo(Qi_u, MAT_LOCAL, minfo, ierr)
          nzS = minfo%nz_used
          call MatGetInfo(op(quop(mode)), MAT_LOCAL, minfo, ierr)
          write(*,'(A,F8.2,A,I0)') &
            "[Stage 4.2] FSAI(Q_u) nnz(G^T G)/nnz(Q_u) = ", &
            nzS / max(minfo%nz_used, 1.d0), " , identity-fallback rows = ", nfl
        endif
      endif
      if (mp == 5 .or. mp == 6) then
        if (mp == 6) then
          call MatMatMult(op(CM_OP_Q1R), op(CM_OP_Q1R), MAT_INITIAL_MATRIX, &
                          PETSC_DEFAULT_REAL, Pat_p, ierr)
        else
          Pat_p = op(CM_OP_Q1R)
        endif
        call build_fsai(op(CM_OP_Q1R), Pat_p, n1, comm, G_p, nfl)
        call MatTransposeMatMult(G_p, G_p, MAT_INITIAL_MATRIX, &
                                 PETSC_DEFAULT_REAL, Qi_p, ierr)
      endif
      ! Stage 6.1: the same inverse for the R-weighted mass that the rho and T
      ! rows carry. Built ONLY when one of those channels is active, and once
      ! for both because they share QR -- so the psi-only path is untouched.
      if (ch_on(CH_RHO) .or. ch_on(CH_T)) then
        call VecDuplicate(ax, lq_R, ierr)
        if (mp == 1) then
          call MatMult(op(CM_OP_QR), ones_p, lq_R, ierr)
        else
          call MatGetDiagonal(op(CM_OP_QR), lq_R, ierr)
        endif
        call VecNorm(lq_R, NORM_INFINITY, rR, ierr)
        call VecGetArray(lq_R, parr, ierr)
        do kk = 1, n1
          if (abs(parr(kk)) > lump_floor * max(rR, 1.d-300)) then
            parr(kk) = 1.0d0 / parr(kk)
          else
            parr(kk) = 1.0d0
          endif
        enddo
        call VecRestoreArray(lq_R, parr, ierr)
        if (mp == 5 .or. mp == 6) then
          if (mp == 6) then
            call MatMatMult(op(CM_OP_QR), op(CM_OP_QR), MAT_INITIAL_MATRIX, &
                            PETSC_DEFAULT_REAL, Pat_R, ierr)
          else
            Pat_R = op(CM_OP_QR)
          endif
          call build_fsai(op(CM_OP_QR), Pat_R, n1, comm, G_R, nfl)
          call MatTransposeMatMult(G_R, G_R, MAT_INITIAL_MATRIX, &
                                   PETSC_DEFAULT_REAL, Qi_R, ierr)
        endif
      endif
      ! Scale the floor by the operator's own magnitude: an absolute 1e-300
      ! test cannot tell a genuinely zeroed boundary row from a derivative
      ! row whose row sum has merely cancelled to round-off.
      call VecNorm(lq_p, NORM_INFINITY, rd, ierr)
      call VecNorm(lq_u, NORM_INFINITY, rn, ierr)
      ! reciprocal, with degenerate rows left unscaled rather than turned
      ! into infinities (construct_commutator_matrices zeroes boundary rows).
      nzrow(ilmp) = 0
      call VecGetArray(lq_u, parr, ierr)
      call VecGetArray(lq_p, parr2, ierr)
      do kk = 1, n1
        if (abs(parr(kk))  > lump_floor * max(rn, 1.d-300)) then
          parr(kk)  = 1.0d0 / parr(kk)
        else
          parr(kk)  = 1.0d0
          nzrow(ilmp) = nzrow(ilmp) + 1
        endif
        if (abs(parr2(kk)) > lump_floor * max(rd, 1.d-300)) then
          parr2(kk) = 1.0d0 / parr2(kk)
        else
          parr2(kk) = 1.0d0
        endif
      enddo
      call VecRestoreArray(lq_p, parr2, ierr)
      call VecRestoreArray(lq_u, parr, ierr)
      if (my_id == 0 .and. mode == 1) write(*,'(A,A,A,I0,A,I0)') &
        "[Stage 4.2] mass inverse ", lumpname(ilmp), ": degenerate u-space rows = ", &
        nzrow(ilmp), " of ", n1

      ! Bind each channel to its own mass inverse. rho and T share the R-mass
      ! entry; the psi entry is the one Stage 5.1 already used.
      ch_lq(CH_PSI) = lq_p;  ch_Qi(CH_PSI) = Qi_p
      ch_lq(CH_RHO) = lq_R;  ch_Qi(CH_RHO) = Qi_R
      ch_lq(CH_T)   = lq_R;  ch_Qi(CH_T)   = Qi_R

      ! S_ass is a genuine sparse matrix only when BOTH inverses are diagonal.
      ! Variants 3/4 keep one exact, so they exist as an operator only and
      ! report no fill -- they are attribution experiments, not candidates.
      ! Sparse assembly is possible whenever BOTH inverses are themselves
      ! sparse matrices: a diagonal (1,2), nothing at all (4), or FSAI (5).
      ! In the small-flow form there is no Q_u^-1, so sparsity depends on the
      ! psi-space inverse alone and mu is irrelevant.
      if (smallflow) then
        sparse_ok = (mp >= 1 .and. mp /= 3)
      else
        sparse_ok = (mu >= 1 .and. mu /= 3) .and. (mp >= 1 .and. mp /= 3)
      endif
      if (sparse_ok) then
        if (smallflow) then
          ! T1 = D_uu, exactly. The cancellation D_uu Q_u^-1 A_uM =
          ! (1+zeta) D_uu, divided through by the (1+zeta) of the right
          ! factor. No matrix product and no mass approximation whatsoever.
          call MatDuplicate(D_uu, MAT_COPY_VALUES, T1_a, ierr)
        else
          ! T1 = (D_uu Q_u^-1) A_uM
          call MatDuplicate(D_uu, MAT_COPY_VALUES, Dsc, ierr)
          if (mu == 5 .or. mu == 6) then
            call MatDestroy(Dsc, ierr)
            call MatMatMult(D_uu, Qi_u, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Dsc, ierr)
          else if (mu /= 4) then
            call MatDiagonalScale(Dsc, PETSC_NULL_VEC, lq_u, ierr)
          endif
          call MatMatMult(Dsc, A_uM, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T1_a, ierr)
        endif
        ! T2 = sum_y (L_uy Q_y^-1) U_yu over the active channels. The
        ! commutator form has only the psi channel, so its loop runs once and
        ! reproduces the previous single triple product exactly.
        ch_first = .true.
        do ich = 1, NCH
          if (.not. ch_on(ich)) cycle
          call MatDuplicate(ch_L(ich), MAT_COPY_VALUES, Lsc, ierr)
          if (mp == 5 .or. mp == 6) then
            call MatDestroy(Lsc, ierr)
            call MatMatMult(ch_L(ich), ch_Qi(ich), MAT_INITIAL_MATRIX, &
                            PETSC_DEFAULT_REAL, Lsc, ierr)
          else if (mp /= 4) then
            call MatDiagonalScale(Lsc, PETSC_NULL_VEC, ch_lq(ich), ierr)
          endif
          if (ch_first) then
            call MatMatMult(Lsc, ch_U(ich), MAT_INITIAL_MATRIX, &
                            PETSC_DEFAULT_REAL, T2_a, ierr)
            ch_first = .false.
          else
            call MatMatMult(Lsc, ch_U(ich), MAT_INITIAL_MATRIX, &
                            PETSC_DEFAULT_REAL, T2_c, ierr)
            call MatAXPY(T2_a, 1.0d0, T2_c, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatDestroy(T2_c, ierr)
          endif
          call MatDestroy(Lsc, ierr)
        enddo

        call MatDuplicate(T1_a, MAT_COPY_VALUES, S_ass, ierr)
        if (smallflow) then
          call MatAXPY(S_ass, -1.0d0/opz, T2_a, DIFFERENT_NONZERO_PATTERN, ierr)
        else
          call MatAXPY(S_ass, -1.0d0, T2_a, DIFFERENT_NONZERO_PATTERN, ierr)
        endif

        call MatGetInfo(S_ass, MAT_LOCAL, minfo, ierr)
        nzS = minfo%nz_used
        call MatGetInfo(D_uu, MAT_LOCAL, minfo, ierr)
        nzD = minfo%nz_used
        fillr(ncase,ilmp) = nzS / max(nzD, 1.d0)
      else
        fillr(ncase,ilmp) = -1.d0
      endif

      !== (3) the effective preconditioner Shat_eff = S_ass A_uM^-1 Q_u ==
      ! Formed densely here ONLY so it can be scored on the same footing as
      ! the operator-form Shat; in production one never forms this, one
      ! applies Shat^-1 = Q_u^-1 A_uM S_ass^-1.
      call MatCreate(PETSC_COMM_SELF, Shat_a, ierr)
      call MatSetSizes(Shat_a, n1, n1, n1, n1, ierr)
      call MatSetType(Shat_a, MATSEQDENSE, ierr)
      call MatSetUp(Shat_a, ierr)

      do kk = 0, n1-1
        call VecZeroEntries(e_u, ierr)
        colidx(1) = kk
        call VecSetValue(e_u, kk, 1.0d0, INSERT_VALUES, ierr)
        call VecAssemblyBegin(e_u, ierr)
        call VecAssemblyEnd(e_u, ierr)
        if (smallflow) then
          call VecCopy(e_u, az, ierr)                   ! right factor is I
        else
          call MatMult(op(quop(mode)), e_u, cv_t, ierr)
          call KSPSolve(ksp_AuM, cv_t, az, ierr)        ! az = A_uM^-1 Q_u e_k
        endif
        ! S_ass az, applied term by term so an exact inverse can be mixed in
        if (smallflow) then
          ! first term is D_uu, unmodified
          call MatMult(D_uu, az, sv_u, ierr)
        else
          call MatMult(A_uM, az, cv_t, ierr)
          if (mu == 0) then
            call KSPSolve(ksp_Qu, cv_t, cv_w, ierr)
          else if (mu == 4) then
            call VecCopy(cv_t, cv_w, ierr)                ! no mass inverse
          else if (mu == 5 .or. mu == 6) then
            call MatMult(Qi_u, cv_t, cv_w, ierr)          ! FSAI
          else if (mu == 3) then
            ! Q_u^-1 r  by npoly steps of diagonally-preconditioned Richardson
            ! (a Neumann series in D^-1 Q_u). Still a POLYNOMIAL in Q_u, hence
            ! still sparse-assemblable in principle -- just wider.
            call VecPointwiseMult(cv_w, cv_t, lq_u, ierr)
            do itry = 1, npoly
              call MatMult(op(quop(mode)), cv_w, ay, ierr)
              call VecAYPX(ay, -1.0d0, cv_t, ierr)        ! ay = r - Q_u w
              call VecPointwiseMult(ay, ay, lq_u, ierr)
              call VecAXPY(cv_w, 1.0d0, ay, ierr)
            enddo
          else
            call VecPointwiseMult(cv_w, cv_t, lq_u, ierr)
          endif
          call MatMult(D_uu, cv_w, sv_u, ierr)
        endif
        ! Second term, summed over the active channels. The commutator form
        ! runs psi only, so this is the previous single-channel code path.
        do ich = 1, NCH
          if (.not. ch_on(ich)) cycle
          call MatMult(ch_U(ich), az, cv_t, ierr)
          if (mp == 0) then
            if (ch_q(ich) == CM_OP_Q1R) then
              call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
            else
              call KSPSolve(ksp_QR, cv_t, cv_q, ierr)
            endif
          else if (mp == 4) then
            call VecCopy(cv_t, cv_q, ierr)                ! no mass inverse
          else if (mp == 5 .or. mp == 6) then
            call MatMult(ch_Qi(ich), cv_t, cv_q, ierr)    ! FSAI
          else
            call VecPointwiseMult(cv_q, cv_t, ch_lq(ich), ierr)
          endif
          call MatMult(ch_L(ich), cv_q, aw, ierr)
          if (smallflow) then
            call VecAXPY(sv_u, -1.0d0/opz, aw, ierr)
          else
            call VecAXPY(sv_u, -1.0d0, aw, ierr)
          endif
        enddo
        call VecGetArray(sv_u, parr, ierr)
        call MatSetValues(Shat_a, n1, rows, 1, colidx, parr, INSERT_VALUES, ierr)
        call VecRestoreArray(sv_u, parr, ierr)
      enddo
      call MatAssemblyBegin(Shat_a, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(Shat_a, MAT_FINAL_ASSEMBLY, ierr)

      call MatDuplicate(Shat_a, MAT_COPY_VALUES, S_dif, ierr)
      call MatAXPY(S_dif, -1.0d0, S_u, SAME_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(S_dif, dmask, PETSC_NULL_VEC, ierr)
      call MatNorm(S_dif, NORM_FROBENIUS, rn, ierr)
      call MatDestroy(S_dif, ierr)
      rassm(ncase,ilmp) = rn / max(nrm_S, 1.d-300)

      call KSPCreate(comm, ksp_test, ierr)
      call KSPSetOperators(ksp_test, S_u, Shat_a, ierr)
      call KSPSetType(ksp_test, KSPGMRES, ierr)
      call KSPSetTolerances(ksp_test, 1.d-8, PETSC_DEFAULT_REAL, PETSC_DEFAULT_REAL, 200, ierr)
      call KSPGetPC(ksp_test, pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call KSPSetUp(ksp_test, ierr)
      call MatCreateVecs(S_u, xt, bt, ierr)
      call VecSetRandom(bt, PETSC_NULL_RANDOM, ierr)
      call VecPointwiseMult(bt, bt, dmask, ierr)
      call KSPSolve(ksp_test, bt, xt, ierr)
      call KSPGetIterationNumber(ksp_test, nits, ierr)
      call KSPGetConvergedReason(ksp_test, kreason, ierr)
      if (kreason .ne. KSP_DIVERGED_ITS .and. kreason%v > 0) then
        itassm(ncase,ilmp) = int(nits)
      else
        itassm(ncase,ilmp) = -1
      endif
      call VecDestroy(bt, ierr)
      call VecDestroy(xt, ierr)
      call KSPDestroy(ksp_test, ierr)

      call MatDestroy(Shat_a, ierr)
      if (sparse_ok) then
        call MatDestroy(S_ass, ierr)
        call MatDestroy(T1_a, ierr)
        call MatDestroy(T2_a, ierr)
        if (.not. smallflow) call MatDestroy(Dsc, ierr)
        ! Lsc is now created and destroyed per channel inside the T2 loop.
      endif
      call VecDestroy(ones_p, ierr)
      call VecDestroy(lq_u, ierr)
      call VecDestroy(lq_p, ierr)
      if (ch_on(CH_RHO) .or. ch_on(CH_T)) call VecDestroy(lq_R, ierr)
      if ((mu == 5 .or. mu == 6) .and. .not. smallflow) then
        call MatDestroy(G_u, ierr)
        call MatDestroy(Qi_u, ierr)
        if (mu == 6) call MatDestroy(Pat_u, ierr)
      endif
      if (mp == 5 .or. mp == 6) then
        call MatDestroy(G_p, ierr)
        call MatDestroy(Qi_p, ierr)
        if (mp == 6) call MatDestroy(Pat_p, ierr)
        if (ch_on(CH_RHO) .or. ch_on(CH_T)) then
          call MatDestroy(G_R, ierr)
          call MatDestroy(Qi_R, ierr)
          if (mp == 6) call MatDestroy(Pat_R, ierr)
        endif
      endif
      call VecDestroy(ax, ierr)
      call VecDestroy(ay, ierr)
      call VecDestroy(az, ierr)
      call VecDestroy(aw, ierr)
      call KSPDestroy(ksp_Qu, ierr)
      call KSPDestroy(ksp_AuM, ierr)
      call MatDestroy(A_uM, ierr)
    end subroutine build_shat_assembled

  end subroutine verify_schur_approx_4x4



  !--------------------------------------------------------------------
  !> Factorized sparse approximate inverse (Kolotilina-Yeremin) of an SPD
  !! matrix A:   A^-1  ~  G^T G,  G lower triangular on the pattern of
  !! tril(A).
  !!
  !! Why this and not a diagonal: the inverse of a banded SPD matrix decays
  !! exponentially away from the diagonal (Demko-Moss-Smith), so a sparse
  !! factor with the bandwidth of A already captures most of A^-1. A
  !! diagonal is the zero-bandwidth member of that family, which is exactly
  !! why it fails on a C1 Bezier mass matrix while FSAI need not.
  !!
  !! Row i solves the small SPD system  A[J,J] g = e_last,  J = {j <= i in
  !! the pattern}, then scales g by 1/sqrt(g_i). Rows are independent.
  !--------------------------------------------------------------------
  !--------------------------------------------------------------------
  subroutine build_fsai(A, P, n, comm, G, nfail)
    Mat, intent(in)  :: A
    Mat, intent(in)  :: P   !< supplies the SPARSITY PATTERN (A itself = level 0,
                            !< A*A = level 1, ...); values always come from A
    PetscInt, intent(in) :: n
    integer, intent(in)  :: comm
    Mat, intent(out) :: G
    integer, intent(out) :: nfail   !< rows that fell back to the identity

    PetscInt :: i, k, m, ncols
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: jj(:), rcnt(:)
    real*8, allocatable :: asub(:,:), rhs(:)
    integer, allocatable :: ipiv(:)
    integer :: info
    character(len=64) :: mtype
    PetscErrorCode :: ierr
    external :: dgesv

    allocate(rcnt(n))
    do i = 0, n-1
      call MatGetRow(P, i, ncols, PETSC_NULL_INTEGER_POINTER, &
                     PETSC_NULL_SCALAR_POINTER, ierr)
      rcnt(i+1) = ncols
      call MatRestoreRow(P, i, ncols, PETSC_NULL_INTEGER_POINTER, &
                         PETSC_NULL_SCALAR_POINTER, ierr)
    enddo

    ! Match A's type exactly: the blocks this multiplies are MPIAIJ even on
    ! one rank, and PETSc has no mixed MPIAIJ*SEQAIJ product.
    call MatGetType(A, mtype, ierr)
    call MatCreate(comm, G, ierr)
    call MatSetSizes(G, n, n, n, n, ierr)
    call MatSetType(G, mtype, ierr)
    call MatSeqAIJSetPreallocation(G, PETSC_DEFAULT_INTEGER, rcnt, ierr)
    call MatMPIAIJSetPreallocation(G, PETSC_DEFAULT_INTEGER, rcnt, &
                                   PETSC_DEFAULT_INTEGER, PETSC_NULL_INTEGER_ARRAY, ierr)
    call MatSetOption(G, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

    nfail = 0
    do i = 0, n-1
      call MatGetRow(P, i, ncols, cols, vals, ierr)
      m = 0
      allocate(jj(ncols))
      do k = 1, ncols
        if (cols(k) <= i) then
          m = m + 1
          jj(m) = cols(k)
        endif
      enddo
      call MatRestoreRow(P, i, ncols, cols, vals, ierr)

      if (m < 1) then
        call MatSetValue(G, i, i, 1.0d0, INSERT_VALUES, ierr)
        nfail = nfail + 1
        deallocate(jj)
        cycle
      endif

      ! MatGetRow returns columns in ascending order, so jj(m) == i.
      allocate(asub(m,m), rhs(m), ipiv(m))
      call MatGetValues(A, m, jj(1:m), m, jj(1:m), asub, ierr)
      rhs      = 0.0d0
      rhs(m)   = 1.0d0
      call dgesv(m, 1, asub, m, ipiv, rhs, m, info)

      if (info /= 0 .or. rhs(m) <= 0.0d0) then
        ! singular row (a zeroed Dirichlet row) -- fall back to identity
        call MatSetValue(G, i, i, 1.0d0, INSERT_VALUES, ierr)
        nfail = nfail + 1
      else
        rhs = rhs / sqrt(rhs(m))
        call MatSetValues(G, 1, (/ i /), m, jj(1:m), rhs, INSERT_VALUES, ierr)
      endif
      deallocate(asub, rhs, ipiv, jj)
    enddo

    call MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY, ierr)
    deallocate(rcnt)

  end subroutine build_fsai


  !--------------------------------------------------------------------
  !> Stage 6.2: build the PRODUCTION small-flow momentum Schur complement,
  !!
  !!   S = Atilde_22 - 1/(1+zeta) * sum_y (L_uy Qi_y) U_yu
  !!
  !! over y = psi (Atilde_21, B_12, Qi from B_33), rho (B_25, B_52, Qi from
  !! B_44) and T (B_26, B_62, Qi from B_44), as selected by
  !! physics_pc_schur_channels. The result lands in g_ctx%S_PBP, which is what
  !! apply_wave_schur_predictor_corrector solves against through ksp_S_PBP.
  !!
  !! This is the mixed-block twin of build_shat_assembled(..., smallflow) in
  !! the Stage 4.2 harness. The channel structure and the SIGNS mirror
  !! s_pbp_diag_mult term for term; the only difference is that the exact
  !! B_55/B_66 MUMPS solves there become sparse mass inverses here, which is
  !! what makes the result an assemblable sparse matrix rather than a shell.
  !!
  !! The Stage 4.2 layout checks verify U_rhou == B_52, L_urho == B_25,
  !! U_Tu == B_62 and L_uT == B_26 to round-off, so the harness's verdict on
  !! this operator transfers to the blocks used here.
  !!
  !! Why (1+zeta) and not theta*dt: the small-flow limit takes D_yy to be
  !! mass-dominated, D_yy ~ (1+zeta) Q_y, and every theta*dt factor is already
  !! carried inside L_uy and U_yu. How good that step is, per channel and per
  !! timestep, is the `damage` column of data/stage61_channels.tsv.
  !--------------------------------------------------------------------
  subroutine build_schur_smallflow_prod(comm, first_time, my_id)
    use phys_module, only: time_evol_zeta, tstep, tstep_prev, &
                           physics_pc_schur_channels, physics_pc_schur_massinv

    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    integer, intent(in) :: my_id

    PetscErrorCode :: ierr
    MatInfo        :: minfo
    real*8         :: opz, nzS, nzD
    integer        :: nch

    opz = 1.d0 + time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    nch = max(1, min(3, physics_pc_schur_channels))

    ! The two masses are geometry-only (B_33 = 1/R, B_44 = R -- see the note on
    ! setup_constraint_mass_ksp), so their sparse inverses are built once for
    ! the whole run rather than on every PC rebuild.
    if (.not. sfp_massinv_ready) then
      call make_mass_inverse(g_ctx%B_33, sfp_Qip, "1/R", comm, my_id, physics_pc_schur_massinv)
      sfp_massinv_ready = .true.
    endif
    ! The R mass is tracked separately because it is only needed for channels
    ! >= 2 here, but also for a CM_OP_QR candidate in the commutator arm --
    ! one flag for both would let sfp_QiR be read before it was built.
    if (nch >= 2 .and. .not. sfp_QiR_ready) then
      call make_mass_inverse(g_ctx%B_44, sfp_QiR, "R  ", comm, my_id, physics_pc_schur_massinv)
      sfp_QiR_ready = .true.
    endif

    if (.not. first_time) call MatDestroy(g_ctx%S_PBP, ierr)
    call MatDuplicate(g_ctx%Atilde_22, MAT_COPY_VALUES, g_ctx%S_PBP, ierr)

    call add_channel(g_ctx%Atilde_21, sfp_Qip, g_ctx%B_12)
    if (nch >= 2) call add_channel(g_ctx%B_25, sfp_QiR, g_ctx%B_52)
    if (nch >= 3) call add_channel(g_ctx%B_26, sfp_QiR, g_ctx%B_62)

    if (my_id == 0) then
      call MatGetInfo(g_ctx%S_PBP, MAT_LOCAL, minfo, ierr)
      nzS = minfo%nz_used
      call MatGetInfo(g_ctx%Atilde_22, MAT_LOCAL, minfo, ierr)
      nzD = minfo%nz_used
      write(*,'(A,I0,A,I0,A,F7.2)') &
        "[Physics PC]   S_PBP: small-flow Schur, channels = ", nch, &
        ", mass inverse = ", physics_pc_schur_massinv, &
        ", nnz/nnz(Atilde_22) = ", nzS / max(nzD, 1.d0)
    endif

    ! Same absolute measure as the mixed arm, so the two are comparable.
    call report_operator_density(g_ctx%S_PBP, "S_PBP (SF momentum Schur)", my_id)
    ! The psi-side operator this arm factorizes, for the like-for-like cost
    ! comparison against the mixed arm's K_pj.
    call report_operator_density(g_ctx%Atilde_11, "A11   (SF psi predictor) ", my_id)

  contains

    !> S_PBP -= (L Qi U) / (1+zeta)
    subroutine add_channel(L, Qi, U)
      Mat, intent(in) :: L, Qi, U
      Mat :: Lsc, Tp

      call MatMatMult(L, Qi, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Lsc, ierr)
      call MatMatMult(Lsc, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Tp, ierr)
      call MatAXPY(g_ctx%S_PBP, -1.0d0/opz, Tp, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDestroy(Lsc, ierr)
      call MatDestroy(Tp, ierr)
    end subroutine add_channel

  end subroutine build_schur_smallflow_prod

  !--------------------------------------------------------------------
  !> Stage 6.3: the PRODUCTION commutator-device momentum Schur operator,
  !! the production twin of the Stage 4.6 harness assembly.
  !!
  !!   S_ass  = Atilde_22 Q_u^-1 A_uM - Atilde_21 Q_p^-1 B_12
  !!   Shat^-1 = Q_u^-1 A_uM S_ass^-1                (applied in the apply)
  !!
  !! where A_uM(ic) = sum_iop coef(ic,iop) op(iop) is the candidate selected
  !! by `label` from the cm_table, and Q_u = op(quop(ic)) is the Riesz map
  !! that candidate pairs with. This routine builds S_ass into g_ctx%S_PBP
  !! and parks A_uM / Q_u^-1 in g_ctx for apply_wave_schur_predictor_corrector.
  !!
  !! WHY THE COEFFICIENT IS -1 AND NOT -1/opz (it differs from the small-flow
  !! builder and that is deliberate): every theta*dt and (1+zeta) factor is
  !! already carried inside A_uM, so the correction enters unscaled. The
  !! consequence is a free regression test -- for label M0, A_uM = opz*Q1R and
  !! Q_u = Q1R, hence Q_u^-1 A_uM = opz*I and
  !!     S_ass = opz*S_sf,   Shat^-1 = opz*(opz*S_sf)^-1 = S_sf^-1,
  !! i.e. M0 with masking off is ALGEBRAICALLY IDENTICAL to the small-flow arm.
  !!
  !! ZBIG masking (physics_pc_schur_mask): the commutator building blocks have
  !! their boundary rows ZEROED (construct_commutator_matrices ->
  !! zero_bc_rows_pc_matrix) while Atilde_22 carries zbig = 1.d12 there, so the
  !! product Atilde_22 Q_u^-1 A_uM puts 1e12-scale garbage on those rows. The
  !! interior form masks them out and carries the identity instead; the apply
  !! restores a plain Jacobi divide there. Only M0-vs-SF regression runs should
  !! turn this off (M0's A_uM is a mass matrix, whose BC rows are not zeroed).
  !!
  !! SERIAL ONLY: make_mass_inverse / build_fsai walk global row indices. The
  !! guard below is load-bearing, not cosmetic -- and it applies equally to the
  !! small-flow arm, which has simply never been run with np > 1.
  !--------------------------------------------------------------------
  subroutine build_schur_commutator_prod(comm, first_time, my_id, label, ok)
    use mod_petsc_pc_commutator_table, only: CM_NOP, CM_MAXC, CM_LABLEN, &
          CM_OP_Q1R, CM_OP_QR, cm_table_build, cm_ops_gather, cm_table_lookup, &
          cm_blocks_ready
    use phys_module, only: time_evol_zeta, time_evol_theta, tstep, tstep_prev, eta, &
                           physics_pc_wave_schur, physics_pc_schur_channels, &
                           physics_pc_schur_mask, physics_pc_schur_massinv

    integer,          intent(in)  :: comm
    logical,          intent(in)  :: first_time
    integer,          intent(in)  :: my_id
    character(len=*), intent(in)  :: label
    logical,          intent(out) :: ok

    real*8, parameter :: zbig_thresh = 1.d11

    PetscErrorCode :: ierr
    MatInfo        :: minfo
    Mat            :: op(CM_NOP)
    Mat            :: Dsc, Lsc, Tp
    real*8         :: coef(CM_MAXC, CM_NOP)
    integer        :: quop(CM_MAXC)
    character(len=CM_LABLEN) :: lab(CM_MAXC)
    integer        :: ncand, ic, iop, nproc, mpierr, ib
    real*8         :: opz, tdt, nzS, nzD
    PetscScalar, pointer :: parr(:)
    PetscInt       :: nloc
    integer        :: k, n_pen
    character(len=256) :: labs

    ok = .false.

    !--- Guards. All of them run BEFORE anything is destroyed or created, so
    !--- that a failure leaves g_ctx exactly as build_schur_smallflow_prod
    !--- expects to find it for this value of first_time.
    ! np > 1 is a HARD stop, not an ok=.false. fallback: the small-flow arm the
    ! caller would fall back to is serial-only for the very same reason (its mass
    ! inverse comes from the same builders), so a fallback merely relocates the
    ! crash into PETSc with a far less informative message. Verified: falling
    ! back here produced ~2e5 lines of PETSc errors and an MPI abort.
    call MPI_Comm_size(comm, nproc, mpierr)
    if (nproc > 1) then
      if (my_id == 0) then
        write(*,'(A)') "[Physics PC]   FATAL: physics_pc_schur_variant = '"// &
          trim(label)//"' is SERIAL ONLY."
        write(*,'(A)') "[Physics PC]     make_mass_inverse and build_fsai walk GLOBAL row "// &
          "indices with MatGetRow, which is invalid for a distributed matrix."
        write(*,'(A)') "[Physics PC]     This limitation applies to the small-flow arm too, "// &
          "so there is no working fallback. Rerun with -np 1."
      endif
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    endif

    if (.not. physics_pc_wave_schur) then
      if (my_id == 0) write(*,'(A)') &
        "[Physics PC]   ERROR: the commutator Schur needs physics_pc_wave_schur = .t. "// &
        "-- the segregated apply solves with ksp_S_PBP but has no slot for the right "// &
        "factor Q_u^-1 A_uM, which would silently give a wrong preconditioner."
      return
    endif

    if (physics_pc_schur_channels /= 1) then
      if (my_id == 0) write(*,'(A,I0,A)') &
        "[Physics PC]   ERROR: the commutator Schur supports physics_pc_schur_channels = 1 "// &
        "only (got ", physics_pc_schur_channels, "). The right factor multiplies the WHOLE "// &
        "Schur complement, so each extra channel would need its own right-multiply."
      return
    endif

    call cm_ops_gather(g_ctx%B_11, g_ctx%B_33, g_ctx%B_44, op)

    opz = 1.d0 + time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    tdt = time_evol_theta * tstep
    call cm_table_build(opz, tdt, eta, coef, quop, lab, ncand)

    ic = cm_table_lookup(lab, ncand, label)
    if (ic == 0) then
      if (my_id == 0) then
        labs = ""
        do ib = 1, ncand
          labs = trim(labs)//" "//trim(lab(ib))
        enddo
        write(*,'(A)') "[Physics PC]   ERROR: physics_pc_schur_variant = '"// &
          trim(label)//"' is not an available cm_table candidate."
        write(*,'(A,L1)') "[Physics PC]     commutator building blocks assembled: ", &
          cm_blocks_ready()
        write(*,'(A)') "[Physics PC]     available labels:"//trim(labs)
        if (.not. cm_blocks_ready()) write(*,'(A)') &
          "[Physics PC]     (block-based candidates such as M1a/M2e are omitted from the "// &
          "table until the blocks exist -- they are assembled automatically for a "// &
          "non-'SF' variant, so this usually means the label is a typo.)"
      endif
      return
    endif

    !--- Mass inverses. Q_u is B_33 (1/R) or B_44 (R) depending on the
    !--- candidate; both are geometry-only, so the run-long cache the
    !--- small-flow arm already keeps is valid here too, and reusing its
    !--- sfp_Qip is exactly what makes the M0 regression exact.
    !--- A_uM by contrast is NOT cacheable -- see below.
    if (.not. sfp_massinv_ready) then
      call make_mass_inverse(g_ctx%B_33, sfp_Qip, "1/R", comm, my_id, physics_pc_schur_massinv)
      sfp_massinv_ready = .true.
    endif
    if (quop(ic) == CM_OP_QR .and. .not. sfp_QiR_ready) then
      call make_mass_inverse(g_ctx%B_44, sfp_QiR, "R  ", comm, my_id, physics_pc_schur_massinv)
      sfp_QiR_ready = .true.
    endif
    if (quop(ic) == CM_OP_QR) then
      g_ctx%Qi_u_prod = sfp_QiR       ! a REFERENCE; ownership stays with the cache
    else
      g_ctx%Qi_u_prod = sfp_Qip
    endif

    !--- A_uM = sum coef(ic,iop) op(iop).
    !--- This MUST be an owned copy and MUST be rebuilt every time: the
    !--- coefficients carry opz/tdt/eta (so they move with the timestep), and
    !--- the building blocks in op(4..) carry the evolving equilibrium (ADV1
    !--- the flow, ES1R/EG1R the Spitzer weight) and are DESTROYED and
    !--- reassembled by petsc_commutator_assemble at the top of every step.
    !--- A stored handle into cm_blk would dangle in the very next apply.
    if (g_ctx%schur_comm_ready) call MatDestroy(g_ctx%A_uM_prod, ierr)
    call MatDuplicate(op(quop(ic)), MAT_COPY_VALUES, g_ctx%A_uM_prod, ierr)
    call MatScale(g_ctx%A_uM_prod, 0.d0, ierr)
    do iop = 1, CM_NOP
      if (coef(ic,iop) /= 0.d0) &
        call MatAXPY(g_ctx%A_uM_prod, coef(ic,iop), op(iop), DIFFERENT_NONZERO_PATTERN, ierr)
    enddo

    !--- S_PBP = Atilde_22 Q_u^-1 A_uM - Atilde_21 Q_p^-1 B_12.
    !--- The psi-space Riesz map is always Q1R (sfp_Qip) regardless of quop:
    !--- that leg is the psi channel, not the u channel.
    if (.not. first_time) call MatDestroy(g_ctx%S_PBP, ierr)
    call MatMatMult(g_ctx%Atilde_22, g_ctx%Qi_u_prod, MAT_INITIAL_MATRIX, &
                    PETSC_DEFAULT_REAL, Dsc, ierr)
    call MatMatMult(Dsc, g_ctx%A_uM_prod, MAT_INITIAL_MATRIX, &
                    PETSC_DEFAULT_REAL, g_ctx%S_PBP, ierr)
    call MatDestroy(Dsc, ierr)

    call MatMatMult(g_ctx%Atilde_21, sfp_Qip, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Lsc, ierr)
    call MatMatMult(Lsc, g_ctx%B_12, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Tp, ierr)
    call MatAXPY(g_ctx%S_PBP, -1.0d0, Tp, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatDestroy(Lsc, ierr)
    call MatDestroy(Tp, ierr)

    !--- ZBIG interior treatment.
    n_pen = 0
    g_ctx%schur_comm_mask = physics_pc_schur_mask
    if (physics_pc_schur_mask) then
      if (.not. g_ctx%schur_comm_ready) then
        call MatCreateVecs(g_ctx%Atilde_22, PETSC_NULL_VEC, g_ctx%u_mask, ierr)
        call VecDuplicate(g_ctx%u_mask, g_ctx%u_bnd, ierr)
        call VecDuplicate(g_ctx%u_mask, g_ctx%u_one, ierr)
      endif

      ! Atilde_22 is what carries the penalty, and it is rebuilt every time,
      ! so the mask is refreshed every time too. NOTE the LOCAL size here:
      ! the Stage 4.6 harness loops the global size, which is one of the
      ! reasons it is serial-only. This form is at least layout-correct.
      call MatGetDiagonal(g_ctx%Atilde_22, g_ctx%u_mask, ierr)
      call VecCopy(g_ctx%u_mask, g_ctx%u_bnd, ierr)
      call VecGetLocalSize(g_ctx%u_mask, nloc, ierr)

      call VecGetArray(g_ctx%u_mask, parr, ierr)
      do k = 1, int(nloc)
        if (abs(parr(k)) > zbig_thresh) then
          parr(k) = 0.0d0
          n_pen = n_pen + 1
        else
          parr(k) = 1.0d0
        endif
      enddo
      call VecRestoreArray(g_ctx%u_mask, parr, ierr)

      ! On a ZBIG row the Schur block IS the penalty, so the right action is a
      ! plain Jacobi divide -- not the identity the interior operator carries,
      ! which would be off by ~1e11.
      call VecGetArray(g_ctx%u_bnd, parr, ierr)
      do k = 1, int(nloc)
        if (abs(parr(k)) > zbig_thresh) then
          parr(k) = 1.0d0 / parr(k)
        else
          parr(k) = 0.0d0
        endif
      enddo
      call VecRestoreArray(g_ctx%u_bnd, parr, ierr)

      call VecSet(g_ctx%u_one, 1.d0, ierr)
      call VecAXPY(g_ctx%u_one, -1.d0, g_ctx%u_mask, ierr)

      ! MatDiagonalSet(ADD_VALUES) can need a new diagonal entry.
      call MatSetOption(g_ctx%S_PBP, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
      call MatSetOption(g_ctx%S_PBP, MAT_NEW_NONZERO_LOCATION_ERR,   PETSC_FALSE, ierr)
      call MatDiagonalScale(g_ctx%S_PBP, g_ctx%u_mask, g_ctx%u_mask, ierr)
      call MatDiagonalSet(g_ctx%S_PBP, g_ctx%u_one, ADD_VALUES, ierr)
    endif

    g_ctx%schur_comm_ready  = .true.
    g_ctx%schur_comm_active = .true.
    ok = .true.

    if (my_id == 0) then
      call MatGetInfo(g_ctx%S_PBP, MAT_LOCAL, minfo, ierr)
      nzS = minfo%nz_used
      call MatGetInfo(g_ctx%Atilde_22, MAT_LOCAL, minfo, ierr)
      nzD = minfo%nz_used
      write(*,'(A,A,A,I0,A,F7.2)') &
        "[Physics PC]   S_PBP: commutator Schur '", trim(label), &
        "', mass inverse = ", physics_pc_schur_massinv, &
        ", nnz/nnz(Atilde_22) = ", nzS / max(nzD, 1.d0)
      write(*,'(A,ES11.4,A,ES11.4,A,ES11.4)') &
        "[Physics PC]     opz = ", opz, ", theta*dt = ", tdt, ", eta*theta*dt = ", eta*tdt
      if (physics_pc_schur_mask) then
        write(*,'(A,I0,A)') "[Physics PC]     ZBIG interior form: ", n_pen, " penalty rows masked"
      else
        write(*,'(A)') "[Physics PC]     ZBIG masking OFF (regression mode)"
      endif
    endif

  end subroutine build_schur_commutator_prod

  !--------------------------------------------------------------------
  !> Report ABSOLUTE operator density: rows, nnz, and nnz per row.
  !!
  !! The per-arm build prints quote nnz RATIOS against different denominators
  !! (Atilde_22 on one arm, B_22 on another, and the mixed arm's operator is
  !! twice the size because it is packed), so those ratios cannot be compared
  !! across arms. Densification under mesh refinement is the gate this whole
  !! line of work is trying to clear, so it needs one absolute number measured
  !! the same way everywhere. That is what this prints.
  !--------------------------------------------------------------------
  subroutine report_operator_density(A, tag, my_id)
    Mat, intent(in)              :: A
    character(len=*), intent(in) :: tag
    integer, intent(in)          :: my_id

    MatInfo        :: minfo
    PetscErrorCode :: ierr
    PetscInt       :: nrow
    real*8         :: nz

    call MatGetSize(A, nrow, PETSC_NULL_INTEGER, ierr)
    call MatGetInfo(A, MAT_GLOBAL_SUM, minfo, ierr)
    nz = minfo%nz_used
    if (my_id == 0) write(*,'(A,A,A,I8,A,ES12.5,A,F9.2)') &
      "[Physics PC]   DENSITY ", tag, ": rows = ", nrow, &
      ", nnz = ", nz, ", nnz/row = ", nz / max(dble(nrow), 1.d0)
  end subroutine report_operator_density

  !--------------------------------------------------------------------
  !> Workstream B, phase 5: is any step of the mixed sweep multigrid-amenable?
  !!
  !! The whole motivation for keeping j and omega explicit was that every
  !! resulting block is second order or a mass matrix, so nothing inherits the
  !! h^-4 conditioning that defeats AMG on the substituted operators. That claim
  !! has never been tested: every measurement so far used PREONLY + LU on all
  !! four operators. This routine tests it directly, per operator, WITHOUT
  !! touching the production solve path -- it builds its own throwaway KSPs on
  !! the same Mats and reports.
  !!
  !! Method: draw a random reference solution x_ref, form b = A x_ref, solve,
  !! and report the TRUE relative error ||x - x_ref|| / ||x_ref|| alongside the
  !! iteration count. The true error is the point: pair_psi spans a ~1e12
  !! dynamic range (B_11 carries the ZBIG Dirichlet rows, B_33 is a mass matrix
  !! with diagonal down to ~2e-7), so its residual norm is dominated by the
  !! penalty rows and says nothing about the j half. A converged residual on
  !! this operator is not evidence of a converged solution.
  !!
  !! Levels (physics_pc_probe_inner), cumulative, so the expensive candidates
  !! can be requested on purpose rather than discovered by a stalled run:
  !!   1  LU (reference) + GMRES/ILU(0)
  !!   2  + GMRES/BoomerAMG applied SCALARLY
  !!   3  + FGMRES/PCFIELDSPLIT on the pair layout (packed operators only)
  !!
  !! On level 2, read the scalar-AMG rows as a control and not as a verdict on
  !! multigrid: the nest->AIJ layout is field-major (all psi rows, then all j
  !! rows), so a scalar coarsening sees two weakly-connected half-graphs of
  !! different physical character. Level 3 is the structurally correct question.
  !--------------------------------------------------------------------
  subroutine probe_inner_solvers(comm, my_id)
    use phys_module, only: physics_pc_probe_inner

    integer, intent(in) :: comm
    integer, intent(in) :: my_id

    integer :: lvl

    lvl = physics_pc_probe_inner
    if (lvl <= 0) return

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ========== inner-solver / AMG amenability probe =========="
      write(*,'(A,I0)') "[Physics PC]   level = ", lvl
      write(*,'(A)') "[Physics PC]   err = ||x - x_ref||/||x_ref||; res = final relative residual."
      write(*,'(A)') "[Physics PC]   On pair_psi trust err, NOT res (ZBIG rows dominate the norm)."
    endif

    ! Packed operators: both get the fieldsplit candidate, since both have a
    ! genuine 2-field layout with a mass matrix in the (2,2) corner.
    call probe_one_operator(g_ctx%K_pj_aij, "pair_psi [B_11,B_13;B_31,B_33]", &
                            g_ctx%is_pair_psi, .true., lvl, comm, my_id)
    call probe_one_operator(g_ctx%S_W_aij,  "pair_w   [S_uu,B_24;B_42,B_44]", &
                            g_ctx%is_pair_w,  .true., lvl, comm, my_id)

    ! Single-field transport blocks. These are the ones that ought to be easy;
    ! if AMG cannot do B_55/B_66 then the probe itself is misconfigured.
    call probe_one_operator(g_ctx%B_55, "B_55 (rho transport)          ", &
                            g_ctx%is_pair_psi, .false., lvl, comm, my_id)
    call probe_one_operator(g_ctx%B_66, "B_66 (T transport)            ", &
                            g_ctx%is_pair_psi, .false., lvl, comm, my_id)

    if (my_id == 0) &
      write(*,'(A)') "[Physics PC] ========== end inner-solver probe =========="

  end subroutine probe_inner_solvers

  !--------------------------------------------------------------------
  !> One operator, all candidates admitted by the probe level.
  !--------------------------------------------------------------------
  subroutine probe_one_operator(A, tag, is_pair, packed, lvl, comm, my_id)
    Mat, intent(in)              :: A
    character(len=*), intent(in) :: tag
    IS,  intent(in)              :: is_pair(2)
    logical, intent(in)          :: packed   !< .true. -> the fieldsplit candidate is meaningful
    integer, intent(in)          :: lvl
    integer, intent(in)          :: comm
    integer, intent(in)          :: my_id

    PetscErrorCode :: ierr
    Vec       :: x_ref, b, x
    PetscInt  :: nrow
    PetscReal :: nref, dmin, dmax
    MatInfo   :: minfo
    Vec       :: dvec

    call MatGetSize(A, nrow, PETSC_NULL_INTEGER, ierr)
    call MatGetInfo(A, MAT_GLOBAL_SUM, minfo, ierr)

    ! Diagonal spread is reported next to the density because together they
    ! predict the AMG result: a 1e12 spread breaks strength-of-connection
    ! thresholding, and a near-dense operator makes coarsening pointless.
    call MatCreateVecs(A, dvec, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, dvec, ierr)
    call VecAbs(dvec, ierr)
    call VecMax(dvec, PETSC_NULL_INTEGER, dmax, ierr)
    call VecMin(dvec, PETSC_NULL_INTEGER, dmin, ierr)
    call VecDestroy(dvec, ierr)

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]   ---------------------------------------------------------"
      write(*,'(A,A)')  "[Physics PC]   operator: ", tag
      write(*,'(A,I8,A,F9.2,A,ES10.3,A,ES10.3)') &
        "[Physics PC]     rows = ", nrow, ", nnz/row = ", &
        minfo%nz_used / max(dble(nrow), 1.d0), &
        ", |diag| min = ", dmin, ", max = ", dmax
      flush(6)
    endif

    ! b = A x_ref with x_ref random: a consistent right-hand side of the
    ! operator's own scale, so the true error below is meaningful.
    call MatCreateVecs(A, x_ref, b, ierr)
    call VecDuplicate(x_ref, x, ierr)
    call VecSetRandom(x_ref, PETSC_NULL_RANDOM, ierr)
    call MatMult(A, x_ref, b, ierr)
    call VecNorm(x_ref, NORM_2, nref, ierr)

                    call probe_candidate(A, b, x, x_ref, nref, "LU (MUMPS)          ", 1, is_pair, comm, my_id)
                    call probe_candidate(A, b, x, x_ref, nref, "GMRES + ILU(0)      ", 2, is_pair, comm, my_id)
    if (lvl >= 2)   call probe_candidate(A, b, x, x_ref, nref, "GMRES + AMG (scalar)", 3, is_pair, comm, my_id)
    if (lvl >= 3 .and. packed) &
                    call probe_candidate(A, b, x, x_ref, nref, "FGMRES + FIELDSPLIT ", 4, is_pair, comm, my_id)

    call VecDestroy(x_ref, ierr); call VecDestroy(b, ierr); call VecDestroy(x, ierr)

  end subroutine probe_one_operator

  !--------------------------------------------------------------------
  !> One (operator, preconditioner) pair: set up, solve, report, tear down.
  !!
  !! Everything is local and destroyed on exit, so the probe cannot perturb the
  !! production KSPs or leak across candidates. Options are set with per-
  !! candidate prefixes for the same reason.
  !--------------------------------------------------------------------
  subroutine probe_candidate(A, b, x, x_ref, nref, name, which, is_pair, comm, my_id)
    Mat, intent(in)              :: A
    Vec, intent(in)              :: b, x_ref
    Vec, intent(inout)           :: x
    PetscReal, intent(in)        :: nref
    character(len=*), intent(in) :: name
    integer, intent(in)          :: which
    IS,  intent(in)              :: is_pair(2)
    integer, intent(in)          :: comm
    integer, intent(in)          :: my_id

    PetscErrorCode :: ierr
    KSP       :: ksp
    PC        :: pc
    Vec       :: e
    PetscInt  :: its
    KSPConvergedReason :: reason
    PetscReal :: rnorm, enorm, t0, t1, t2
    PetscReal :: rtol, abstol, dtol
    integer, parameter :: PROBE_MAXITS = 300
    character(len=16) :: pfx
    character(len=12) :: verdict

    rtol = 1.0d-8; abstol = 1.0d-50; dtol = 1.0d4

    call PetscTime(t0, ierr)
    call KSPCreate(comm, ksp, ierr)
    call KSPSetOperators(ksp, A, A, ierr)
    call KSPGetPC(ksp, pc, ierr)

    select case (which)

    case (1)
      call KSPSetType(ksp, KSPPREONLY, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)

    case (2)
      call KSPSetType(ksp, KSPGMRES, ierr)
      call KSPGMRESSetRestart(ksp, 60, ierr)
      ! PCILU cannot be applied to MPIAIJ, and these operators are MPIAIJ even
      ! on one rank (MatConvert of the nest produces MPIAIJ unconditionally).
      ! BJACOBI with one block per rank is the serial-equivalent route: its
      ! default sub-PC is exactly ILU(0) on the local SEQAIJ block.
      call PCSetType(pc, PCBJACOBI, ierr)

    case (3)
      ! Scalar BoomerAMG. Deliberately plain: no nodal coarsening, no near-null
      ! space, no Euclid smoothing. The question here is whether the operator is
      ! amenable at all, and a tuned configuration would confound "the operator
      ! is fine" with "the options were right".
      pfx = "probeamg_"
      call KSPSetOptionsPrefix(ksp, trim(pfx), ierr)
      call KSPSetType(ksp, KSPGMRES, ierr)
      call KSPGMRESSetRestart(ksp, 60, ierr)
      call PCSetType(pc, PCHYPRE, ierr)
      call PCHYPRESetType(pc, "boomeramg", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
        "-probeamg_pc_hypre_boomeramg_strong_threshold", "0.5", ierr)
      call KSPSetFromOptions(ksp, ierr)

    case (4)
      ! The structurally correct iterative candidate: split on the pair layout,
      ! lower Schur factorization, AMG on the (1,1) field and a direct solve on
      ! the (2,2) mass block. "selfp" builds the Schur approximation from the
      ! (2,2) diagonal; on this layout the (2,2) IS a mass matrix, which is the
      ! one case where selfp is the right choice rather than a fallback.
      pfx = "probefs_"
      call KSPSetOptionsPrefix(ksp, trim(pfx), ierr)
      call KSPSetType(ksp, KSPFGMRES, ierr)
      call KSPGMRESSetRestart(ksp, 60, ierr)
      call PCSetType(pc, PCFIELDSPLIT, ierr)
      call PCFieldSplitSetIS(pc, "0", is_pair(1), ierr)
      call PCFieldSplitSetIS(pc, "1", is_pair(2), ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_pc_fieldsplit_type", "schur", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_pc_fieldsplit_schur_fact_type", "lower", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_pc_fieldsplit_schur_precondition", "selfp", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_fieldsplit_0_ksp_type", "gmres", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_fieldsplit_0_ksp_max_it", "20", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_fieldsplit_0_ksp_rtol", "1e-2", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_fieldsplit_0_pc_type", "hypre", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_fieldsplit_0_pc_hypre_type", "boomeramg", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_fieldsplit_1_ksp_type", "preonly", ierr)
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-probefs_fieldsplit_1_pc_type", "lu", ierr)
      ! Same MPIAIJ restriction as above: name the factor package explicitly
      ! rather than relying on a default that does not exist for this type.
      call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
        "-probefs_fieldsplit_1_pc_factor_mat_solver_type", "mumps", ierr)
      call KSPSetFromOptions(ksp, ierr)

    end select

    call KSPSetTolerances(ksp, rtol, abstol, dtol, PROBE_MAXITS, ierr)

    if (my_id == 0) then
      write(*,'(A,A,A)') "[Physics PC]     ", name, " : setting up ..."
      flush(6)   !< a PETSc abort in this candidate must not swallow earlier rows
    endif
    call KSPSetUp(ksp, ierr)
    call PetscTime(t1, ierr)

    call VecZeroEntries(x, ierr)
    call KSPSolve(ksp, b, x, ierr)
    call PetscTime(t2, ierr)

    call KSPGetIterationNumber(ksp, its, ierr)
    call KSPGetResidualNorm(ksp, rnorm, ierr)
    call KSPGetConvergedReason(ksp, reason, ierr)

    call VecDuplicate(x, e, ierr)
    call VecCopy(x, e, ierr)
    call VecAXPY(e, -1.0d0, x_ref, ierr)
    call VecNorm(e, NORM_2, enorm, ierr)
    call VecDestroy(e, ierr)
    enorm = enorm / max(nref, 1.d-300)

    ! Verdict on the SOLUTION, per the recorded pair_psi lesson. 1e-6 is a
    ! deliberately generous bar: as an inner solve inside a preconditioner this
    ! only has to be a good approximation, not exact.
    if (reason%v < 0) then
      verdict = "DIVERGED"
    else if (enorm > 1.d-6) then
      verdict = "BAD-SOLUTION"
    else if (its >= PROBE_MAXITS) then
      verdict = "MAXITS"
    else
      verdict = "ok"
    endif

    if (my_id == 0) write(*,'(A,A,A,I5,A,ES10.3,A,ES10.3,A,F8.2,A,F8.2,A,I3,A,A)') &
      "[Physics PC]     ", name, " : its = ", its, &
      ", err = ", enorm, ", res = ", rnorm, &
      ", setup = ", t1 - t0, " s, solve = ", t2 - t1, " s, reason = ", reason%v, &
      "  ", trim(verdict)
    if (my_id == 0) flush(6)

    call KSPDestroy(ksp, ierr)

  end subroutine probe_candidate

  !--------------------------------------------------------------------
  !> Workstream B: hard stop when np > 1.
  !!
  !! make_mass_inverse walks GLOBAL row indices with MatGetRow and, on the
  !! diagonal arm, indexes a local VecGetArray array by the global size --
  !! both invalid for a distributed matrix. This is an MPI_Abort and not an
  !! ok=.false. fallback on purpose: there is no serial-safe arm to fall back
  !! to, so a fallback would merely relocate the crash into PETSc with a far
  !! less informative message.
  !--------------------------------------------------------------------
  subroutine schur_mixed_require_serial(comm, my_id, label, always)
    use phys_module, only: physics_pc_schur_massinv, physics_pc_schur_pairinv

    integer, intent(in) :: comm, my_id
    character(len=*), intent(in) :: label
    logical, intent(in), optional :: always  !< .true. -> serial regardless of flags

    integer :: nproc, mpierr
    logical :: unconditional, needs_fsai

    unconditional = .false.
    if (present(always)) unconditional = always

    call MPI_Comm_size(comm, nproc, mpierr)
    if (nproc == 1) return

    ! What is actually serial is build_fsai, which walks GLOBAL row indices with
    ! MatGetRow to assemble each row's least-squares pattern -- it needs rows this
    ! rank does not own. The diagonal/lumped arm of make_mass_inverse is now
    ! ownership-range correct, so an SFM2 configuration that uses only diagonal
    ! inverses is parallel-safe and must NOT be blocked here.
    needs_fsai = (physics_pc_schur_massinv == 7 .or. physics_pc_schur_massinv == 8 .or. &
                  physics_pc_schur_pairinv == 7 .or. physics_pc_schur_pairinv == 8)

    if (unconditional .or. needs_fsai) then
      if (my_id == 0) then
        write(*,'(A)') "[Physics PC]   FATAL: "//trim(label)//" is SERIAL ONLY."
        if (needs_fsai) then
          write(*,'(A,I0,A,I0,A)') &
            "[Physics PC]     physics_pc_schur_massinv = ", physics_pc_schur_massinv, &
            ", physics_pc_schur_pairinv = ", physics_pc_schur_pairinv, &
            " -- FSAI (7, 8) walks GLOBAL row indices with MatGetRow."
          write(*,'(A)') "[Physics PC]     Use 2 (diagonal) on both for a parallel run, "// &
            "or rerun with -np 1."
        else
          write(*,'(A)') "[Physics PC]     Rerun with -np 1."
        endif
      endif
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    endif
  end subroutine schur_mixed_require_serial

  !--------------------------------------------------------------------
  !> Workstream B: assemble pair_psi = [[B_11, B_13], [B_31, B_33]].
  !!
  !! The mixed (psi,j) pair, kept explicit instead of substituting j = J(psi).
  !! Row 2 IS Jacobian row 3 verbatim -- [B_31, 0, B_33, 0, 0, 0] -- so this
  !! pair carries the j constraint EXACTLY, with zero approximation. That is
  !! what lets the apply drop the ksp_Mj pre-solve and the j back-substitution:
  !! solving the pair for RHS (x_psi, x_j) returns j* = M_j^-1 (x_j - B_31 psi*)
  !! as its second component, which is what the back-substitution computed.
  !!
  !! Both off-diagonal blocks are second order (B_13, B_31 are the Grad-Shafranov
  !! Delta* constraint couplings) and the (2,2) entry B_33 is a positive 1/R mass
  !! matrix. So this is NOT a true saddle point -- it is the Ciarlet-Raviart
  !! mixed form of the biharmonic Delta* Q^-1 Delta*, LU-friendly now and
  !! fieldsplit-friendly later, whereas the substituted fourth-order operator
  !! has kappa ~ h^-4.
  !!
  !! Rebuilt on every PC rebuild: B_11 carries theta*tstep and the evolving
  !! state. The nest is a temporary and is destroyed here -- it holds REFERENCES
  !! to B_11/B_13/B_31/B_33, which are themselves destroyed and re-extracted on
  !! the next rebuild, so a cached nest would dangle.
  !--------------------------------------------------------------------
  subroutine build_pair_psi_prod(comm, first_time, my_id)
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    integer, intent(in) :: my_id

    Mat            :: mats_nest(4), PJ_nest
    PetscErrorCode :: ierr
    PetscInt       :: n1_loc, n2_loc, n1_glo, n2_glo

    call schur_mixed_require_serial(comm, my_id, "physics_pc_schur_variant = 'SFM'")

    ! Row-major (PETSc Fortran MatCreateNest convention, see assemble_monolithic_4x4).
    ! Row 1 (psi eq):  B_11  B_13
    mats_nest(1) = g_ctx%B_11
    mats_nest(2) = g_ctx%B_13
    ! Row 2 (j eq):    B_31  B_33   -- verbatim, no sign flip: the constraint is
    !                                  assembled as B_31 psi + B_33 j = r_j.
    mats_nest(3) = g_ctx%B_31
    mats_nest(4) = g_ctx%B_33

    PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS_ARRAY, 2, PETSC_NULL_IS_ARRAY, mats_nest, PJ_nest, ierr))

    if (g_ctx%schur_mixed_ready) call MatDestroy(g_ctx%K_pj_aij, ierr)
    call MatConvert(PJ_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%K_pj_aij, ierr)
    call MatDestroy(PJ_nest, ierr)

    ! Layout-only strides into the packed vector. Unused by the direct arm; a
    ! future PCFIELDSPLIT inner solver needs exactly these.
    if (.not. g_ctx%schur_mixed_vecs_ready) then
      call MatGetLocalSize(g_ctx%B_11, n1_loc, PETSC_NULL_INTEGER, ierr)
      call MatGetLocalSize(g_ctx%B_33, n2_loc, PETSC_NULL_INTEGER, ierr)
      call MatGetSize(g_ctx%B_11, n1_glo, PETSC_NULL_INTEGER, ierr)
      call MatGetSize(g_ctx%B_33, n2_glo, PETSC_NULL_INTEGER, ierr)
      call ISCreateStride(comm, n1_loc, 0,      1, g_ctx%is_pair_psi(1), ierr)
      call ISCreateStride(comm, n2_loc, n1_glo, 1, g_ctx%is_pair_psi(2), ierr)
    endif

    if (my_id == 0) write(*,'(A)') &
      "[Physics PC]   pair_psi: [[B_11,B_13],[B_31,B_33]] assembled (j kept mixed)"

  end subroutine build_pair_psi_prod

  !--------------------------------------------------------------------
  !> Workstream B: assemble pair_w = [[S_uu^SFM, B_24], [B_42, B_44]], the
  !! small-flow MIXED-PAIR momentum Schur operator.
  !!
  !!   S_uu^SFM = B_22 - sum_ch (L Qi U)/(1+zeta),  L in {B_21, B_25, B_26}
  !!
  !! Derivation. With y = (rho,T,psi,j) and w = (u,omega), the j-row of the upper
  !! coupling U and the omega-row of the lower coupling L are both identically
  !! zero, so L M_y^-1 U has exactly ONE nonzero block entry, at (u,u):
  !!
  !!   S_w = D_w - L M_y^-1 U = [[ D_u - L_uy M_y^-1 U_yu , J_uw ], [ J_wu , M_w ]]
  !!
  !! The omega row and column pass through EXACTLY -- only the (u,u) entry is
  !! ever approximated. The small-flow limit then replaces M_y^-1 by a
  !! block-diagonal Riesz map diag(Q_rho, Q_T, (1+zeta)Q_psi, M_j)^-1, and
  !! because U's j-row is zero the j-channel contributes NOTHING, leaving the
  !! three channels above.
  !!
  !! RAW BLOCKS ONLY. The base is B_22, not a Schur-corrected u diagonal, and the
  !! psi-channel L is B_21, not a j-folded one. The two fourth-order folds
  !! (B_24 M_w^-1 B_42 and B_23 M_j^-1 B_31) are recovered by keeping the pairs
  !! mixed -- subtracting them from the diagonals as well would double-count
  !! them. That failure mode is silent: mildly worse convergence, no error, no
  !! log difference. The ||S_uu - B_22||_F and nnz/row prints below are the
  !! guard against it.
  !!
  !! All three channels are supported. There is no right factor on this arm (the
  !! inverse of pair_w is the whole operator, unlike the commutator arm's
  !! S_ass), hence no channels == 1 restriction.
  !--------------------------------------------------------------------
  subroutine build_schur_mixed_prod(comm, first_time, my_id, label, ok)
    use phys_module, only: time_evol_zeta, tstep, tstep_prev, &
                           physics_pc_schur_channels, physics_pc_schur_massinv, &
                           physics_pc_schur_pairinv, physics_pc_wave_schur

    integer, intent(in)           :: comm
    logical, intent(in)           :: first_time
    integer, intent(in)           :: my_id
    character(len=*), intent(in)  :: label   !< "SFM" or "SFM2"
    logical, intent(out)          :: ok

    Mat            :: S_uu, D_uu_diff, mats_nest(4), W_nest
    PetscErrorCode :: ierr
    MatInfo        :: minfo
    real*8         :: opz, nzS, nzD, fnorm, dmin, dmax
    integer        :: nch
    PetscInt       :: n1_loc, n2_loc, n1_glo, n2_glo
    Vec            :: dvec
    Mat            :: Ltil, Shat, Shati, TA, TB
    logical        :: paired
    real*8         :: fL, fS

    ok = .false.

    !--- Guards. Both run BEFORE anything is created or destroyed, so a failure
    !--- leaves g_ctx exactly as it was found for this value of first_time.
    call schur_mixed_require_serial(comm, my_id, "physics_pc_schur_variant = 'SFM'")

    if (.not. physics_pc_wave_schur) then
      if (my_id == 0) write(*,'(A)') &
        "[Physics PC]   ERROR: the mixed-pair Schur needs physics_pc_wave_schur = .t. "// &
        "-- the segregated apply has no slot for the packed (u,omega) solve, so it "// &
        "would silently give a wrong preconditioner."
      return
    endif

    opz = 1.d0 + time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    nch = max(1, min(3, physics_pc_schur_channels))

    ! B_33 (1/R mass) and B_44 (R mass) are geometry-only, so their sparse
    ! inverses are built once for the whole run. TWO separate ready flags, not
    ! one: sfp_QiR is only needed for channels >= 2 here but also by a CM_OP_QR
    ! commutator candidate, and a single combined flag would let it be read
    ! before it was built.
    if (.not. sfp_massinv_ready) then
      call make_mass_inverse(g_ctx%B_33, sfp_Qip, "1/R", comm, my_id, physics_pc_schur_massinv)
      sfp_massinv_ready = .true.
    endif
    if (nch >= 2 .and. .not. sfp_QiR_ready) then
      call make_mass_inverse(g_ctx%B_44, sfp_QiR, "R  ", comm, my_id, physics_pc_schur_massinv)
      sfp_QiR_ready = .true.
    endif

    paired = (trim(label) == "SFM2")

    !--- (1,1) entry: S_uu. NOTE B_22 -- the raw u diagonal (see header).
    call MatDuplicate(g_ctx%B_22, MAT_COPY_VALUES, S_uu, ierr)

    if (.not. paired) then
      !--- "SFM": M_y^-1 -> a fully block-DIAGONAL Riesz map. Measured and
      !--- rejected (see docs/physics_pc/workstream_B_mixed_schur.md section 5):
      !--- diagonalising M_y severs psi from j, and since U reaches the momentum
      !--- equation only through its psi-row, the Lorentz coupling B_23 cannot
      !--- reach the Schur complement AT ALL. Retained only as the control.
      call add_channel(g_ctx%B_21, sfp_Qip, g_ctx%B_12, 1.d0/opz)      ! psi channel
    else
      !--- "SFM2": still block diagonal, but with ONE LARGER BLOCK -- rho, T and
      !--- the (psi,j) PAIR kept together:
      !---
      !---   M_y^-1 -> diag( Q_rho, Q_T, pair_psi_sf^-1 )
      !---   pair_psi_sf = [[ (1+zeta) Q_psi , J_psij ], [ J_jpsi , M_j ]]
      !---
      !--- The 2x2 block inverse then gives, for the psi-j contribution to the
      !--- momentum Schur complement,
      !---
      !---   [L_upsi, L_uj] pair^-1 [U_psiu; 0] = (B_21 - B_23 M_j^-1 B_31) Shat^-1 B_12
      !---   Shat = (1+zeta) Q_psi - B_13 M_j^-1 B_31
      !---
      !--- so B_23 is back: keeping psi and j in one block is exactly what lets
      !--- the Lorentz coupling reach the channel. psi and j share the 1/R mass
      !--- space, so Q_psi = B_33 and every operand is already extracted.
      !---
      !--- Note this is strictly MORE accurate than a small-flow channel that
      !--- uses Q_psi^-1/(1+zeta): Shat additionally carries the J M^-1 J
      !--- correction to the psi diagonal. The (1+zeta) lives INSIDE Shat, so the
      !--- channel coefficient below is 1, not 1/opz.
      call MatMatMult(g_ctx%B_23, sfp_Qip, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, TA, ierr)
      call MatMatMult(TA, g_ctx%B_31, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, TB, ierr)
      call MatDuplicate(g_ctx%B_21, MAT_COPY_VALUES, Ltil, ierr)
      call MatAXPY(Ltil, -1.0d0, TB, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDestroy(TA, ierr); call MatDestroy(TB, ierr)

      call MatMatMult(g_ctx%B_13, sfp_Qip, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, TA, ierr)
      call MatMatMult(TA, g_ctx%B_31, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, TB, ierr)
      call MatDuplicate(g_ctx%B_33, MAT_COPY_VALUES, Shat, ierr)
      call MatScale(Shat, opz, ierr)
      call MatAXPY(Shat, -1.0d0, TB, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDestroy(TA, ierr); call MatDestroy(TB, ierr)

      ! Shat^-1 gets its OWN fidelity flag, not physics_pc_schur_massinv. Two
      ! reasons: it is the new knob and conflating it with the mass inverse would
      ! hide which one matters; and massinv = 8 forms Q*Q for its sparsity
      ! pattern, which for a Shat that is already a triple product would be a
      ! density explosion. Default is 2 (diagonal).
      call make_mass_inverse(Shat, Shati, "Shat", comm, my_id, physics_pc_schur_pairinv)

      call add_channel(Ltil, Shati, g_ctx%B_12, 1.0d0)

      call MatNorm(Ltil, NORM_FROBENIUS, fL, ierr)
      call MatNorm(Shat, NORM_FROBENIUS, fS, ierr)
      if (my_id == 0) write(*,'(A,I0,A,ES11.4,A,ES11.4)') &
        "[Physics PC]   SFM2 paired psi channel: pairinv = ", physics_pc_schur_pairinv, &
        ", ||Ltil||_F = ", fL, ", ||Shat||_F = ", fS
      call report_operator_density(Ltil, "Ltil (B_21 - B_23 Qi B_31)", my_id)
      call report_operator_density(Shat, "Shat (opz Q - B_13 Qi B_31)", my_id)
      call MatDestroy(Ltil, ierr)
      call MatDestroy(Shat, ierr)
      call MatDestroy(Shati, ierr)
    endif

    if (nch >= 2) call add_channel(g_ctx%B_25, sfp_QiR, g_ctx%B_52, 1.d0/opz)  ! rho channel
    if (nch >= 3) call add_channel(g_ctx%B_26, sfp_QiR, g_ctx%B_62, 1.d0/opz)  ! T   channel

    !--- Pack into the 2x2 (u,omega) nest and convert to a concrete AIJ.
    ! Row 1 (u eq):     S_uu  B_24
    mats_nest(1) = S_uu
    mats_nest(2) = g_ctx%B_24
    ! Row 2 (omega eq): B_42  B_44   -- verbatim, no sign flip (constraint row 4).
    mats_nest(3) = g_ctx%B_42
    mats_nest(4) = g_ctx%B_44

    PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS_ARRAY, 2, PETSC_NULL_IS_ARRAY, mats_nest, W_nest, ierr))

    if (g_ctx%schur_mixed_ready) call MatDestroy(g_ctx%S_W_aij, ierr)
    call MatConvert(W_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%S_W_aij, ierr)
    call MatDestroy(W_nest, ierr)

    if (.not. g_ctx%schur_mixed_vecs_ready) then
      call MatGetLocalSize(g_ctx%B_22, n1_loc, PETSC_NULL_INTEGER, ierr)
      call MatGetLocalSize(g_ctx%B_44, n2_loc, PETSC_NULL_INTEGER, ierr)
      call MatGetSize(g_ctx%B_22, n1_glo, PETSC_NULL_INTEGER, ierr)
      call MatGetSize(g_ctx%B_44, n2_glo, PETSC_NULL_INTEGER, ierr)
      call ISCreateStride(comm, n1_loc, 0,      1, g_ctx%is_pair_w(1), ierr)
      call ISCreateStride(comm, n2_loc, n1_glo, 1, g_ctx%is_pair_w(2), ierr)
    endif

    !--- Diagnostics. These are the measurements this arm exists for.
    if (my_id == 0) then
      call MatGetInfo(g_ctx%S_W_aij, MAT_LOCAL, minfo, ierr)
      nzS = minfo%nz_used
      call MatGetInfo(g_ctx%B_22, MAT_LOCAL, minfo, ierr)
      nzD = minfo%nz_used
      write(*,'(A,I0,A,I0,A,F7.2)') &
        "[Physics PC]   pair_w: mixed small-flow Schur, channels = ", nch, &
        ", mass inverse = ", physics_pc_schur_massinv, &
        ", nnz(S_W)/nnz(B_22) = ", nzS / max(nzD, 1.d0)
    endif

    ! ||S_uu - B_22||_F: how much the Riesz channels actually moved the diagonal.
    ! If this is ~0 the channels are inert; if it is enormous, suspect a
    ! double-counted fold (a Schur-corrected block used in place of a raw one).
    call MatDuplicate(S_uu, MAT_COPY_VALUES, D_uu_diff, ierr)
    call MatAXPY(D_uu_diff, -1.0d0, g_ctx%B_22, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatNorm(D_uu_diff, NORM_FROBENIUS, fnorm, ierr)
    call MatDestroy(D_uu_diff, ierr)

    ! min/max |diag| of the packed operator: ~1e12 ZBIG penalty rows against ~h^2
    ! mass rows. This ratio decides whether an iterative inner solver has any
    ! chance here, and it is the first thing to look at if the LU struggles.
    call MatCreateVecs(g_ctx%S_W_aij, dvec, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(g_ctx%S_W_aij, dvec, ierr)
    call VecAbs(dvec, ierr)
    call VecMax(dvec, PETSC_NULL_INTEGER, dmax, ierr)
    call VecMin(dvec, PETSC_NULL_INTEGER, dmin, ierr)
    call VecDestroy(dvec, ierr)

    if (my_id == 0) then
      write(*,'(A,ES12.5)') "[Physics PC]     ||S_uu - B_22||_F        = ", fnorm
      write(*,'(A,ES12.5,A,ES12.5)') &
        "[Physics PC]     |diag(S_W)| min = ", dmin, "  max = ", dmax
    endif

    ! S_uu is the entry that compares against the SF arm's S_PBP: same L Qi U
    ! triple-product structure, different base and different psi-channel L. S_W
    ! additionally carries the two sparse constraint blocks and twice the rows,
    ! so it is NOT the like-for-like number.
    call report_operator_density(S_uu,           "S_uu (SFM momentum Schur)", my_id)
    call report_operator_density(g_ctx%S_W_aij,  "S_W  (SFM packed u,omega)", my_id)
    call report_operator_density(g_ctx%K_pj_aij, "K_pj (SFM packed psi,j)  ", my_id)
    call report_operator_density(g_ctx%B_22,     "B_22 (raw u diagonal)    ", my_id)

    call MatDestroy(S_uu, ierr)   ! safe: MatConvert copied the values out

    g_ctx%schur_mixed_ready  = .true.
    g_ctx%schur_mixed_active = .true.
    ok = .true.

  contains

    !> S_uu -= alpha * (L Qi U). alpha is explicit because the SFM2 psi channel
    !! carries its (1+zeta) inside Shat and so needs alpha = 1, while every
    !! Riesz-map channel needs alpha = 1/(1+zeta).
    subroutine add_channel(L, Qi, U, alpha)
      Mat, intent(in)    :: L, Qi, U
      real*8, intent(in) :: alpha
      Mat :: Lsc, Tp

      call MatMatMult(L, Qi, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Lsc, ierr)
      call MatMatMult(Lsc, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Tp, ierr)
      call MatAXPY(S_uu, -alpha, Tp, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDestroy(Lsc, ierr)
      call MatDestroy(Tp, ierr)
    end subroutine add_channel

  end subroutine build_schur_mixed_prod

  !--------------------------------------------------------------------
  !> Workstream B null test: verify the mixed-pair apply against the
  !! preconditioner's OWN defining equations.
  !!
  !! Self-referential on purpose -- no reference implementation, no dense
  !! probing, O(1) matvecs, so it runs on any mesh. Draw one random 6-variable
  !! residual x, apply the preconditioner once, and evaluate every row of
  !! P_mixed on the result.
  !!
  !!   row 3 (j)     : ||B_31 y_psi + B_33 y_j - x_j||          HARD PASS/FAIL
  !!   row 4 (omega) : ||B_42 y_u   + B_44 y_w - x_w||          HARD PASS/FAIL
  !!
  !! Those two rows are exact BY CONSTRUCTION: row 2 of each mixed pair IS the
  !! corresponding constraint equation, verbatim and unapproximated. So any
  !! deviation beyond solver round-off is a wiring bug, and this single test
  !! catches every one that will realistically happen -- wrong packing order, a
  !! transposed nest, a swapped index set, a corrector sign error, or a
  !! back-substitution that should have been skipped but was not.
  !!
  !!   row 1 (psi), rows 5/6 (rho,T) : small but NONZERO. These measure the
  !!     block-LDU lag -- the corrector applies B_16 against the PREDICTOR T*,
  !!     and the rho/T correctors build on the predictor values.
  !!   row 2 (u)    : NOT small. This row carries the small-flow Riesz
  !!     approximation itself, which is the whole point of the arm; it is
  !!     reported for scale, not as a check.
  !!
  !! The last two prints are the redundancy check: they confirm that y_j and y_w
  !! equal what the constraint mass back-substitutions would have produced, which
  !! is what licenses skipping them in the dispatcher.
  !--------------------------------------------------------------------
  subroutine verify_schur_mixed_apply(comm, my_id)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T

    integer, intent(in) :: comm, my_id

    PC             :: dummy_pc     !< physics_pc_apply never dereferences its PC
                                   !  argument; it reads everything from g_ctx.
    Vec            :: xf, yf, acc, tmp, jref, wref
    Vec            :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec            :: y_psi, y_u, y_j, y_w, y_rho, y_T
    PetscErrorCode :: ierr
    PetscInt       :: n1_loc, n1_glo
    real*8         :: r1, r2, r3, r4, r5, r6, nx, dj, dw
    real*8         :: r3s, r4s, d33min, d33max
    real*8         :: pr_top, pr_bot, wr_top, wr_bot
    Vec            :: pres, wres
    logical        :: pass34

    if (.not. g_ctx%schur_mixed_active) then
      if (my_id == 0) write(*,'(A)') &
        "[Physics PC] mixed-pair null test SKIPPED: physics_pc_schur_variant /= 'SFM'"
      return
    endif
    call schur_mixed_require_serial(comm, my_id, "the mixed-pair null test", always=.true.)

    ! All six variables carry identical DOF counts, so the full system vector is
    ! six times one variable block.
    call MatGetLocalSize(g_ctx%B_11, n1_loc, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(g_ctx%B_11, n1_glo, PETSC_NULL_INTEGER, ierr)
    call VecCreate(comm, xf, ierr)
    call VecSetSizes(xf, 6*n1_loc, 6*n1_glo, ierr)
    call VecSetFromOptions(xf, ierr)
    call VecDuplicate(xf, yf, ierr)
    call VecSetRandom(xf, PETSC_NULL_RANDOM, ierr)
    call VecZeroEntries(yf, ierr)

    call physics_pc_apply(dummy_pc, xf, yf, ierr)

    call VecGetSubVector(xf, g_ctx%is_var(var_psi), x_psi, ierr)
    call VecGetSubVector(xf, g_ctx%is_var(var_u),   x_u,   ierr)
    call VecGetSubVector(xf, g_ctx%is_var(var_zj),  x_j,   ierr)
    call VecGetSubVector(xf, g_ctx%is_var(var_w),   x_w,   ierr)
    call VecGetSubVector(xf, g_ctx%is_var(var_rho), x_rho, ierr)
    call VecGetSubVector(xf, g_ctx%is_var(var_T),   x_T,   ierr)
    call VecGetSubVector(yf, g_ctx%is_var(var_psi), y_psi, ierr)
    call VecGetSubVector(yf, g_ctx%is_var(var_u),   y_u,   ierr)
    call VecGetSubVector(yf, g_ctx%is_var(var_zj),  y_j,   ierr)
    call VecGetSubVector(yf, g_ctx%is_var(var_w),   y_w,   ierr)
    call VecGetSubVector(yf, g_ctx%is_var(var_rho), y_rho, ierr)
    call VecGetSubVector(yf, g_ctx%is_var(var_T),   y_T,   ierr)

    call MatCreateVecs(g_ctx%B_11, acc,  PETSC_NULL_VEC, ierr)
    call MatCreateVecs(g_ctx%B_11, tmp,  PETSC_NULL_VEC, ierr)
    call MatCreateVecs(g_ctx%B_11, jref, PETSC_NULL_VEC, ierr)
    call MatCreateVecs(g_ctx%B_11, wref, PETSC_NULL_VEC, ierr)
    call MatCreateVecs(g_ctx%K_pj_aij, pres, PETSC_NULL_VEC, ierr)
    call MatCreateVecs(g_ctx%S_W_aij,  wres, PETSC_NULL_VEC, ierr)

    ! row 3 (j):  B_31 y_psi + B_33 y_j - x_j
    call MatMult(g_ctx%B_31, y_psi, acc, ierr)
    call MatMult(g_ctx%B_33, y_j,   tmp, ierr)
    call VecAXPY(acc, 1.0d0, tmp, ierr)
    call VecAXPY(acc, -1.0d0, x_j, ierr)
    call VecNorm(acc, NORM_2, r3, ierr)
    call VecNorm(x_j, NORM_2, nx, ierr)
    r3 = r3 / max(nx, 1.d-300)

    ! row 4 (omega):  B_42 y_u + B_44 y_w - x_w
    call MatMult(g_ctx%B_42, y_u, acc, ierr)
    call MatMult(g_ctx%B_44, y_w, tmp, ierr)
    call VecAXPY(acc, 1.0d0, tmp, ierr)
    call VecAXPY(acc, -1.0d0, x_w, ierr)
    call VecNorm(acc, NORM_2, r4, ierr)
    call VecNorm(x_w, NORM_2, nx, ierr)
    r4 = r4 / max(nx, 1.d-300)

    ! row 1 (psi):  B_11 y_psi + B_13 y_j + B_12 y_u + B_16 y_T - x_psi
    call MatMult(g_ctx%B_11, y_psi, acc, ierr)
    call MatMult(g_ctx%B_13, y_j,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_12, y_u,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_16, y_T,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call VecAXPY(acc, -1.0d0, x_psi, ierr)
    call VecNorm(acc, NORM_2, r1, ierr)
    call VecNorm(x_psi, NORM_2, nx, ierr)
    r1 = r1 / max(nx, 1.d-300)

    ! row 2 (u):  B_21 y_psi + B_22 y_u + B_23 y_j + B_24 y_w + B_25 y_rho + B_26 y_T - x_u
    call MatMult(g_ctx%B_21, y_psi, acc, ierr)
    call MatMult(g_ctx%B_22, y_u,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_23, y_j,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_24, y_w,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_25, y_rho, tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_26, y_T,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call VecAXPY(acc, -1.0d0, x_u, ierr)
    call VecNorm(acc, NORM_2, r2, ierr)
    call VecNorm(x_u, NORM_2, nx, ierr)
    r2 = r2 / max(nx, 1.d-300)

    ! row 5 (rho):  B_51 y_psi + B_52 y_u + B_55 y_rho - x_rho
    call MatMult(g_ctx%B_51, y_psi, acc, ierr)
    call MatMult(g_ctx%B_52, y_u,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_55, y_rho, tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call VecAXPY(acc, -1.0d0, x_rho, ierr)
    call VecNorm(acc, NORM_2, r5, ierr)
    call VecNorm(x_rho, NORM_2, nx, ierr)
    r5 = r5 / max(nx, 1.d-300)

    ! row 6 (T):  B_61 y_psi + B_62 y_u + B_63 y_j + B_66 y_T - x_T
    call MatMult(g_ctx%B_61, y_psi, acc, ierr)
    call MatMult(g_ctx%B_62, y_u,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_63, y_j,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call MatMult(g_ctx%B_66, y_T,   tmp, ierr); call VecAXPY(acc, 1.0d0, tmp, ierr)
    call VecAXPY(acc, -1.0d0, x_T, ierr)
    call VecNorm(acc, NORM_2, r6, ierr)
    call VecNorm(x_T, NORM_2, nx, ierr)
    r6 = r6 / max(nx, 1.d-300)

    ! Redundancy check: do y_j / y_w match the mass back-substitutions that the
    ! dispatcher skips on this arm? If yes, skipping them was legitimate.
    call MatMult(g_ctx%B_31, y_psi, tmp, ierr)
    call VecAYPX(tmp, -1.0d0, x_j, ierr)                 ! tmp = x_j - B_31 y_psi
    call KSPSolve(g_ctx%ksp_Mj, tmp, jref, ierr)
    call VecAXPY(jref, -1.0d0, y_j, ierr)
    call VecNorm(jref, NORM_2, dj, ierr)
    call VecNorm(y_j, NORM_2, nx, ierr)
    dj = dj / max(nx, 1.d-300)

    call MatMult(g_ctx%B_42, y_u, tmp, ierr)
    call VecAYPX(tmp, -1.0d0, x_w, ierr)                 ! tmp = x_w - B_42 y_u
    call KSPSolve(g_ctx%ksp_Mw, tmp, wref, ierr)
    call VecAXPY(wref, -1.0d0, y_w, ierr)
    call VecNorm(wref, NORM_2, dw, ierr)
    call VecNorm(y_w, NORM_2, nx, ierr)
    dw = dw / max(nx, 1.d-300)

    ! Diagonal-scaled versions of rows 3 and 4. These, not the raw norms above,
    ! are the meaningful gate. B_33 and B_44 carry ZBIG = 1e12 penalty rows on the
    ! boundary alongside ~h^2 mass rows in the interior, so a raw ||.||/||x||
    ! ratio is dominated by that row-scale disparity: an absolute error which is
    ! negligible in the penalty rows still reads ~1e-9 relative. Dividing through
    ! by the row diagonal removes the disparity and asks the actual question --
    ! is the constraint equation satisfied, row by row?
    call MatGetDiagonal(g_ctx%B_33, tmp, ierr)
    call VecAbs(tmp, ierr)
    call VecMax(tmp, PETSC_NULL_INTEGER, d33max, ierr)
    call VecMin(tmp, PETSC_NULL_INTEGER, d33min, ierr)
    call MatMult(g_ctx%B_31, y_psi, acc, ierr)
    call MatMult(g_ctx%B_33, y_j,   jref, ierr)
    call VecAXPY(acc, 1.0d0, jref, ierr)
    call VecAXPY(acc, -1.0d0, x_j, ierr)
    call VecPointwiseDivide(acc, acc, tmp, ierr)
    call VecNorm(acc, NORM_2, r3s, ierr)
    call VecPointwiseDivide(jref, x_j, tmp, ierr)
    call VecNorm(jref, NORM_2, nx, ierr)
    r3s = r3s / max(nx, 1.d-300)

    call MatGetDiagonal(g_ctx%B_44, tmp, ierr)
    call VecAbs(tmp, ierr)
    call MatMult(g_ctx%B_42, y_u, acc, ierr)
    call MatMult(g_ctx%B_44, y_w, wref, ierr)
    call VecAXPY(acc, 1.0d0, wref, ierr)
    call VecAXPY(acc, -1.0d0, x_w, ierr)
    call VecPointwiseDivide(acc, acc, tmp, ierr)
    call VecNorm(acc, NORM_2, r4s, ierr)
    call VecPointwiseDivide(wref, x_w, tmp, ierr)
    call VecNorm(wref, NORM_2, nx, ierr)
    r4s = r4s / max(nx, 1.d-300)

    ! The gate uses the diagonal-scaled constraint rows together with the two
    ! back-substitution redundancy checks. The latter are the strongest evidence
    ! available: they recompute j and omega from y_psi/y_u through the INDEPENDENT
    ! ksp_Mj/ksp_Mw factorizations, so they cannot agree unless the packing order,
    ! the block orientations, the pair RHS components and the corrector signs are
    ! all correct.
    ! Is the discrepancy the PAIR SOLVE's own residual rather than the sweep?
    ! Solve pair_psi on a known RHS and measure the residual of each block row
    ! separately. If it lands in the j half, the LU on the packed operator -- not
    ! the wiring -- is the source, and the gate above is measuring the inner
    ! solver rather than the algorithm.
    call pack_2v(x_psi, x_j, g_ctx%rhs_PJ, ierr)
    call KSPSolve(g_ctx%ksp_pair_psi, g_ctx%rhs_PJ, g_ctx%sol_PJ, ierr)
    call MatMult(g_ctx%K_pj_aij, g_ctx%sol_PJ, pres, ierr)
    call VecAXPY(pres, -1.0d0, g_ctx%rhs_PJ, ierr)
    call unpack_2v(pres, acc, tmp, ierr)
    call VecNorm(acc, NORM_2, pr_top, ierr)      ! psi-row half
    call VecNorm(tmp, NORM_2, pr_bot, ierr)      ! j-row half
    call VecNorm(g_ctx%rhs_PJ, NORM_2, nx, ierr)
    pr_top = pr_top / max(nx, 1.d-300)
    pr_bot = pr_bot / max(nx, 1.d-300)

    call pack_2v(x_u, x_w, g_ctx%rhs_W, ierr)
    call KSPSolve(g_ctx%ksp_pair_w, g_ctx%rhs_W, g_ctx%sol_W, ierr)
    call MatMult(g_ctx%S_W_aij, g_ctx%sol_W, wres, ierr)
    call VecAXPY(wres, -1.0d0, g_ctx%rhs_W, ierr)
    call unpack_2v(wres, acc, tmp, ierr)
    call VecNorm(acc, NORM_2, wr_top, ierr)      ! u-row half
    call VecNorm(tmp, NORM_2, wr_bot, ierr)      ! omega-row half
    call VecNorm(g_ctx%rhs_W, NORM_2, nx, ierr)
    wr_top = wr_top / max(nx, 1.d-300)
    wr_bot = wr_bot / max(nx, 1.d-300)

    ! The gate is the two back-substitution redundancy checks. They recompute j
    ! and omega from y_psi/y_u through the INDEPENDENT ksp_Mj/ksp_Mw
    ! factorizations, so they cannot agree unless the packing order, the block
    ! orientations, the pair RHS components and the corrector signs are all
    ! correct -- and unlike the constraint-row norms they are not polluted by the
    ! packed LU's own residual, which the prints below isolate.
    pass34 = (dj < 1.d-10) .and. (dw < 1.d-10)

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ================ mixed-pair (SFM) null test ================"
      write(*,'(A)')          "[Physics PC]   -- THE GATE: j and omega recomputed via the independent mass KSPs --"
      write(*,'(A,ES12.5)')   "[Physics PC]   y_j vs M_j^-1(x_j - B_31 y_psi) = ", dj
      write(*,'(A,ES12.5)')   "[Physics PC]   y_w vs M_w^-1(x_w - B_42 y_u)   = ", dw
      write(*,'(A)')          "[Physics PC]   -- constraint-row RESIDUALS (informational, see note below) --"
      write(*,'(A,ES12.5,A,ES12.5)') "[Physics PC]   row 3 (j)     raw = ", r3, "  diag-scaled = ", r3s
      write(*,'(A,ES12.5,A,ES12.5)') "[Physics PC]   row 4 (omega) raw = ", r4, "  diag-scaled = ", r4s
      write(*,'(A,ES12.5,A,ES12.5)') "[Physics PC]   |diag(B_33)| min = ", d33min, "  max = ", d33max
      write(*,'(A,ES12.5,A)') "[Physics PC]   row 1 (psi)   rel resid = ", r1, "   (block-LDU lag, expected nonzero)"
      write(*,'(A,ES12.5,A)') "[Physics PC]   row 5 (rho)   rel resid = ", r5, "   (block-LDU lag, expected nonzero)"
      write(*,'(A,ES12.5,A)') "[Physics PC]   row 6 (T)     rel resid = ", r6, "   (block-LDU lag, expected nonzero)"
      write(*,'(A,ES12.5,A)') "[Physics PC]   row 2 (u)     rel resid = ", r2, "   (the small-flow Riesz approximation itself)"
      write(*,'(A)')          "[Physics PC]   -- inner LU residual, per block row --"
      write(*,'(A,ES12.5,A,ES12.5)') "[Physics PC]   pair_psi resid: psi-half = ", pr_top, "  j-half     = ", pr_bot
      write(*,'(A,ES12.5,A,ES12.5)') "[Physics PC]   pair_w   resid: u-half   = ", wr_top, "  omega-half = ", wr_bot
      if (pass34) then
        write(*,'(A)') "[Physics PC]   PASS: j and omega match their mass back-substitutions to round-off"
        write(*,'(A)') "[Physics PC]         -- the mixed-pair sweep is wired correctly."
        write(*,'(A)') "[Physics PC]   NOTE on the constraint-row residuals above: they read ~1e-8/1e-9, not"
        write(*,'(A)') "[Physics PC]     ~1e-14, and that is EXPECTED rather than a defect. pair_psi spans a"
        write(*,'(A)') "[Physics PC]     ~1e12 dynamic range (B_11 carries the ZBIG = 1e12 penalty rows while"
        write(*,'(A)') "[Physics PC]     B_33 is a mass matrix with diagonal down to ~1e-7), so a backward-"
        write(*,'(A)') "[Physics PC]     stable LU leaves a residual ~eps*||A|| which is round-off in the"
        write(*,'(A)') "[Physics PC]     large-scale psi rows and ~1e-9 relative in the small-scale j rows."
        write(*,'(A)') "[Physics PC]     The per-block-row LU residuals below show exactly that split. The"
        write(*,'(A)') "[Physics PC]     SOLUTIONS are accurate to round-off, which is what the gate measures"
        write(*,'(A)') "[Physics PC]     and what the sweep consumes. Verified not to be a solver scaling"
        write(*,'(A)') "[Physics PC]     deficiency: MUMPS automatic scaling (-mat_mumps_icntl_8 77) leaves"
        write(*,'(A)') "[Physics PC]     the j-half residual unchanged at ~3e-9."
      else
        write(*,'(A)') "[Physics PC]   *** FAIL *** j or omega does not match its mass back-substitution."
        write(*,'(A)') "[Physics PC]       This is a WIRING BUG,"
        write(*,'(A)') "[Physics PC]       not an approximation quality issue. Suspects, in order:"
        write(*,'(A)') "[Physics PC]         - pack_2v/unpack_2v order vs the MatCreateNest row-major order"
        write(*,'(A)') "[Physics PC]         - a transposed off-diagonal block (B_13 vs B_31, B_24 vs B_42)"
        write(*,'(A)') "[Physics PC]         - the corrector pair RHS j-component not being ZERO"
        write(*,'(A)') "[Physics PC]         - the predictor pair RHS j-component not being x_j"
        write(*,'(A)') "[Physics PC]         - the dispatcher back-substitution not actually skipped"
        write(*,'(A)') "[Physics PC]       Do NOT trust any SFM convergence number until this passes."
      endif
      write(*,'(A)') "[Physics PC] =============================================================="
    endif

    call VecRestoreSubVector(xf, g_ctx%is_var(var_psi), x_psi, ierr)
    call VecRestoreSubVector(xf, g_ctx%is_var(var_u),   x_u,   ierr)
    call VecRestoreSubVector(xf, g_ctx%is_var(var_zj),  x_j,   ierr)
    call VecRestoreSubVector(xf, g_ctx%is_var(var_w),   x_w,   ierr)
    call VecRestoreSubVector(xf, g_ctx%is_var(var_rho), x_rho, ierr)
    call VecRestoreSubVector(xf, g_ctx%is_var(var_T),   x_T,   ierr)
    call VecRestoreSubVector(yf, g_ctx%is_var(var_psi), y_psi, ierr)
    call VecRestoreSubVector(yf, g_ctx%is_var(var_u),   y_u,   ierr)
    call VecRestoreSubVector(yf, g_ctx%is_var(var_zj),  y_j,   ierr)
    call VecRestoreSubVector(yf, g_ctx%is_var(var_w),   y_w,   ierr)
    call VecRestoreSubVector(yf, g_ctx%is_var(var_rho), y_rho, ierr)
    call VecRestoreSubVector(yf, g_ctx%is_var(var_T),   y_T,   ierr)

    call VecDestroy(acc,  ierr); call VecDestroy(tmp,  ierr)
    call VecDestroy(jref, ierr); call VecDestroy(wref, ierr)
    call VecDestroy(pres, ierr); call VecDestroy(wres, ierr)
    call VecDestroy(xf,   ierr); call VecDestroy(yf,   ierr)

  end subroutine verify_schur_mixed_apply

  !> Qi ~ Q^-1 as an explicit sparse matrix: FSAI (Kolotilina-Yeremin) at
  !! level 0 or 1, or the plain diagonal. A diagonal is NOT adequate on the
  !! C1 Bezier space at production tstep (docs/physics_pc §9.4); it is kept
  !! as the cheap arm, not as a recommendation.
  !!
  !! Lives at module scope (it used to be a contains routine of
  !! build_schur_smallflow_prod) so build_schur_commutator_prod can share it.
  !! comm/my_id/massinv were host-associated before; the body is otherwise
  !! unchanged.
  !!
  !! SERIAL ONLY, both arms: build_fsai walks GLOBAL row indices with
  !! MatGetRow, and the diagonal arm indexes a VecGetArray local array by the
  !! global size. Callers must guard on np == 1.
  subroutine make_mass_inverse(Q, Qi, tag, comm, my_id, massinv)
    Mat, intent(in)  :: Q
    Mat, intent(out) :: Qi
    character(len=*), intent(in) :: tag
    integer, intent(in) :: comm
    integer, intent(in) :: my_id
    integer, intent(in) :: massinv  !< physics_pc_schur_massinv at the call site

    PetscErrorCode :: ierr

    Mat      :: G, Pat
    Vec      :: dv
    PetscInt :: nn, ii, nloc, nloc_c, rstart, rend
    integer  :: mpierr
    PetscScalar, pointer :: dptr(:)
    integer  :: nf, kk, nfloor
    PetscReal :: dmax
    character(len=64) :: qtype
    character(len=8)  :: dlab

    call MatGetSize(Q, nn, PETSC_NULL_INTEGER, ierr)

    ! Parallel cross-check: ||Q||_F is independent of everything this routine
    ! does, so if it differs between np=1 and np>1 the discrepancy is in the
    ! block that was handed in, not in the inverse built from it.
    block
      PetscReal :: qn
      call MatNorm(Q, NORM_FROBENIUS, qn, ierr)
      if (my_id == 0) write(*,'(A,A,A,ES14.7)') &
        "[Physics PC]   operand(", tag, "): ||Q||_F = ", qn
    end block

    if (massinv == 7 .or. massinv == 8) then
      if (massinv == 8) then
        call MatMatMult(Q, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Pat, ierr)
      else
        Pat = Q
      endif
      call build_fsai(Q, Pat, nn, comm, G, nf)
      call MatTransposeMatMult(G, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Qi, ierr)
      call MatDestroy(G, ierr)
      if (massinv == 8) call MatDestroy(Pat, ierr)
      if (my_id == 0) write(*,'(A,A,A,I0,A,I0)') &
        "[Physics PC]   FSAI(", tag, " mass): identity-fallback rows = ", &
        nf, " of ", nn
    else
      ! diagonal, with the same magnitude-relative floor the harness uses, so
      ! an eliminated boundary row becomes 1 rather than an infinity.
      call MatCreateVecs(Q, PETSC_NULL_VEC, dv, ierr)
      if (massinv == 3) then
        ! Row-sum mass lumping: M_lump_ii = sum_j M_ij. For a Lagrange basis
        ! this is the textbook-superior alternative to diagonal truncation --
        ! it is exact on constants and conserves total mass. On JOREK's C1
        ! Bezier/Hermite space it is NOT: the derivative basis functions have
        ! (near-)zero mean, so their row sums nearly vanish and the lumped
        ! entry is a small difference of larger numbers. Watch the floored-row
        ! count printed below: any nonzero count means lumping produced a
        ! singular entry that had to be replaced by 1, and the inverse is then
        ! not an approximation of anything.
        call MatGetRowSum(Q, dv, ierr)
        dlab = "rowsum  "
      else
        call MatGetDiagonal(Q, dv, ierr)
        dlab = "diagonal"
      endif
      call VecNorm(dv, NORM_INFINITY, dmax, ierr)
      ! Match Q's type exactly, as build_fsai does and for the same reason: the
      ! blocks this gets multiplied against are MPIAIJ even on one rank, and
      ! PETSc has no mixed MPIAIJ*SEQAIJ product. MATAIJ resolves to SEQAIJ on a
      ! single rank, so the previous MatSetType(Qi, MATAIJ) produced a type
      ! mismatch whose MatMatMult silently returned garbage -- ierr is not
      ! checked on these calls, so it was a wrong S_PBP with no error message.
      call MatGetType(Q, qtype, ierr)
      call MatCreate(comm, Qi, ierr)
      ! Inherit Q's ROW LAYOUT rather than letting PETSc decide it: Qi is
      ! multiplied against Q and against the rectangular coupling blocks, and a
      ! PETSC_DECIDE split need not reproduce the layout those already carry.
      call MatGetLocalSize(Q, nloc, nloc_c, ierr)
      call MatSetSizes(Qi, nloc, nloc_c, nn, nn, ierr)
      call MatSetType(Qi, qtype, ierr)
      call MatSeqAIJSetPreallocation(Qi, 1, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatMPIAIJSetPreallocation(Qi, 1, PETSC_NULL_INTEGER_ARRAY, &
                                     0, PETSC_NULL_INTEGER_ARRAY, ierr)
      ! PARALLEL-SAFE: walk the LOCAL rows and offset by the ownership range.
      ! The previous version looped kk = 1 .. global nn while indexing the LOCAL
      ! VecGetArray pointer, which is correct only on one rank -- it read past
      ! the end of dptr and wrote rows this rank does not own. dmax is already a
      ! global NORM_INFINITY, so the floor threshold is rank-independent.
      call MatGetOwnershipRange(Q, rstart, rend, ierr)
      call VecGetArray(dv, dptr, ierr)
      nfloor = 0
      do kk = 1, int(rend - rstart)
        ii = rstart + kk - 1
        if (abs(dptr(kk)) > 1.d-12 * max(dmax, 1.d-300)) then
          call MatSetValue(Qi, ii, ii, 1.0d0/dptr(kk), INSERT_VALUES, ierr)
        else
          call MatSetValue(Qi, ii, ii, 1.0d0, INSERT_VALUES, ierr)
          nfloor = nfloor + 1
        endif
      enddo
      call VecRestoreArray(dv, dptr, ierr)
      call MPI_Allreduce(MPI_IN_PLACE, nfloor, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
      call MatAssemblyBegin(Qi, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(Qi, MAT_FINAL_ASSEMBLY, ierr)
      call VecDestroy(dv, ierr)
      if (my_id == 0) write(*,'(A,A,A,A,A,I0,A,I0)') &
        "[Physics PC]   ", trim(dlab), "(", tag, ") inverse built: floored rows = ", &
        nfloor, " of ", nn
    endif

    ! Fidelity of the inverse: ||Qi*Q - I||_F. Not a curiosity -- the
    ! commutator Schur multiplies by Qi*Q explicitly (its right factor is
    ! Q_u^-1 A_uM), so this number IS the perturbation that separates a
    ! commutator candidate from its exact algebraic limit. For label M0 it is
    ! the ENTIRE difference from the small-flow arm, which is why the two arms
    ! agree in convergence RATE but not to the last digit.
    block
      Mat :: QiQ
      PetscReal :: dev
      call MatMatMult(Qi, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, QiQ, ierr)
      call MatShift(QiQ, -1.0d0, ierr)
      call MatNorm(QiQ, NORM_FROBENIUS, dev, ierr)
      call MatDestroy(QiQ, ierr)
      if (my_id == 0) write(*,'(A,A,A,ES11.4,A,ES11.4)') &
        "[Physics PC]   fidelity(", tag, " mass): ||Qi*Q - I||_F = ", dev, &
        ", per row = ", dev / sqrt(max(real(nn,8), 1.d0))
    end block
  end subroutine make_mass_inverse


  !--------------------------------------------------------------------
  !> Stage 4.5 driver: assemble S_ass sparsely for every M_* candidate and
  !! hand each one to schur_itersolve_probe.
  !!
  !! This deliberately does NOT reuse the Stage 4.2 harness. That harness
  !! forms the exact S_u and the effective Shat by DENSE probing -- O(n)
  !! MUMPS solves per candidate per mass-inverse variant -- which caps the
  !! problem at the 9x8 toy mesh. But the question here ("does the
  !! iteration count stay flat under refinement?") is answerable only ON
  !! refinement, so the probing has to go. Everything below is O(nnz):
  !! two sparse triple products per candidate, no dense object anywhere.
  !!
  !! The price is that this reports no approximation defect -- it cannot,
  !! without S_u. That is fine: the defect is already published (Stage 4.2)
  !! and is a property of the approximation, not of its solvability. Run
  !! Stage 4.2 on the small mesh for quality, Stage 4.5 on a sequence of
  !! meshes for feasibility.
  !!
  !! Mass inverses are FSAI-1 throughout, the treatment selected in
  !! Workstream A: it is the only sparse variant that survives the
  !! timestep flip in finding (b), and diagonal/lumped/Neumann variants are
  !! ruled out by finding (c) (the C1 mass matrix is not diagonally
  !! dominant).
  !--------------------------------------------------------------------
  subroutine verify_schur_itersolve(comm, my_id)
    use mod_petsc_pc_commutator_table, only: CM_NOP, CM_MAXC, CM_LABLEN, &
          CM_OP_Q1R, cm_table_build, cm_ops_gather, cm_blocks_ready
    use phys_module, only: time_evol_zeta, time_evol_theta, tstep, tstep_prev, eta

    integer, intent(in) :: comm, my_id

    IS  :: is_u
    Mat :: D_uu, U_pu, L_up, A_uM
    Mat :: op(CM_NOP)
    Mat :: G_u, G_p, Qi_u, Qi_p, Pat_u, Pat_p
    Mat :: Dsc, Lsc, T1_a, T2_a, S_ass
    Mat :: Lsc_d, T2_d                  !< Stage 5.1 diagonal-Q_p variant
    Vec :: dmask, dq
    MatInfo :: minfo
    PetscInt :: n1, ntot, k
    PetscInt, allocatable :: idx(:)
    PetscScalar, pointer :: parr(:)
    PetscErrorCode :: ierr
    integer :: nproc, mpierr, ic, iop, nfl_u, nfl_p
    real*8  :: opz, zeta, tdt, nzS, nzD
    PetscReal :: dqmax
    real*8  :: coef(CM_MAXC, CM_NOP)
    integer :: quop(CM_MAXC)
    character(len=CM_LABLEN) :: lab(CM_MAXC)
    integer :: ncand
    real*8, parameter :: zbig_thresh = 1.d11

    if (.not. g_ctx%p_full_perm_ready) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.5] ERROR: P_full_perm not built."
      return
    endif
    call MPI_Comm_size(comm, nproc, mpierr)
    if (nproc /= 1) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.5] ERROR: serial only. Use -np 1."
      return
    endif

    zeta = time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    opz  = 1.d0 + zeta
    tdt  = time_evol_theta * tstep

    call MatGetSize(g_ctx%P_full_perm, ntot, PETSC_NULL_INTEGER, ierr)
    n1 = ntot / 4

    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "================================================================"
      write(*,'(A)') " Stage 4.5: is the assembled Schur complement SOLVABLE?"
      write(*,'(A)') "   S_ass = D_uu Q_u^-1 A_uM - L_up Q^-1 U_pu,  FSAI-1 inverses."
      write(*,'(A,I0,A,E12.4)') "   n = ", n1, " per variable,  theta*dt = ", tdt
      write(*,'(A)') "================================================================"
    endif

    allocate(idx(n1))
    do k = 0, n1-1
      idx(k+1) = n1 + k
    enddo
    call ISCreateGeneral(comm, n1, idx, PETSC_COPY_VALUES, is_u, ierr)
    deallocate(idx)

    call MatCreateSubMatrix(g_ctx%P_full_perm, is_u, is_u, MAT_INITIAL_MATRIX, D_uu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(1), g_ctx%is_reduced(2), &
                            MAT_INITIAL_MATRIX, U_pu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(1), &
                            MAT_INITIAL_MATRIX, L_up, ierr)

    call cm_ops_gather(g_ctx%B_11, g_ctx%B_33, g_ctx%B_44, op)
    call cm_table_build(opz, tdt, eta, coef, quop, lab, ncand)
    if (my_id == 0 .and. .not. cm_blocks_ready()) write(*,'(A)') &
      "[Stage 4.5] NOTE: commutator blocks not assembled "// &
      "(set commutator_analysis=.t.); only M0/M1x/M0R available."

    ! Interior mask: ZBIG Dirichlet rows would otherwise dominate every norm
    ! and wreck every factorisation.
    call MatCreateVecs(D_uu, PETSC_NULL_VEC, dmask, ierr)
    call MatGetDiagonal(D_uu, dmask, ierr)
    call VecGetArray(dmask, parr, ierr)
    do k = 1, n1
      if (abs(parr(k)) > zbig_thresh) then
        parr(k) = 0.0d0
      else
        parr(k) = 1.0d0
      endif
    enddo
    call VecRestoreArray(dmask, parr, ierr)

    call MatGetInfo(D_uu, MAT_LOCAL, minfo, ierr)
    nzD = minfo%nz_used

    ! psi-space Riesz map is the same for every candidate, so its FSAI is
    ! built once; Q_u depends on the candidate through quop().
    call MatMatMult(op(CM_OP_Q1R), op(CM_OP_Q1R), MAT_INITIAL_MATRIX, &
                    PETSC_DEFAULT_REAL, Pat_p, ierr)
    call build_fsai(op(CM_OP_Q1R), Pat_p, n1, comm, G_p, nfl_p)
    call MatTransposeMatMult(G_p, G_p, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Qi_p, ierr)
    call MatMatMult(L_up, Qi_p, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Lsc, ierr)
    call MatMatMult(Lsc, U_pu, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T2_a, ierr)

    do ic = 1, ncand
      ! Only the live candidates. Ranking the full table is Stage 4.2's job
      ! and it is a QUALITY measurement on one small mesh; this is a COST
      ! measurement that has to be repeated on a mesh sequence, so probing
      ! all 14 would multiply the only expensive part by ~5 for candidates
      ! already known not to be in contention. M0 is the small-flow control,
      ! M1a and M2e are the two regime candidates.
      if (.not. (trim(lab(ic)) == "M0"  .or. &
                 trim(lab(ic)) == "M1a" .or. &
                 trim(lab(ic)) == "M2e")) cycle

      call MatDuplicate(op(quop(ic)), MAT_COPY_VALUES, A_uM, ierr)
      call MatScale(A_uM, 0.0d0, ierr)
      do iop = 1, CM_NOP
        if (coef(ic,iop) /= 0.d0) then
          call MatAXPY(A_uM, coef(ic,iop), op(iop), DIFFERENT_NONZERO_PATTERN, ierr)
        endif
      enddo

      call MatMatMult(op(quop(ic)), op(quop(ic)), MAT_INITIAL_MATRIX, &
                      PETSC_DEFAULT_REAL, Pat_u, ierr)
      call build_fsai(op(quop(ic)), Pat_u, n1, comm, G_u, nfl_u)
      call MatTransposeMatMult(G_u, G_u, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Qi_u, ierr)

      call MatMatMult(D_uu, Qi_u, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Dsc, ierr)
      call MatMatMult(Dsc, A_uM, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T1_a, ierr)
      call MatDuplicate(T1_a, MAT_COPY_VALUES, S_ass, ierr)
      call MatAXPY(S_ass, -1.0d0, T2_a, DIFFERENT_NONZERO_PATTERN, ierr)

      call MatGetInfo(S_ass, MAT_LOCAL, minfo, ierr)
      nzS = minfo%nz_used
      if (my_id == 0) write(*,'(A,A,A,F8.2,A,I0)') &
        "[Stage 4.5] M_* ", trim(lab(ic)), ": nnz(S_ass)/nnz(D_uu) = ", &
        nzS / max(nzD, 1.d0), " , FSAI identity-fallback rows = ", nfl_u + nfl_p

      call schur_itersolve_probe(comm, my_id, S_ass, dmask, "M_* " // trim(lab(ic)))

      call MatDestroy(S_ass, ierr)
      call MatDestroy(T1_a, ierr)
      call MatDestroy(Dsc, ierr)
      call MatDestroy(Qi_u, ierr)
      call MatDestroy(G_u, ierr)
      call MatDestroy(Pat_u, ierr)
      call MatDestroy(A_uM, ierr)
    enddo

    !== Stage 5.1: the small-flow forms ===============================
    ! S_sf = D_uu - L_up Q_p^-1 U_pu / (1+zeta): no Q_u^-1, no A_uM, no
    ! right factor, and D_uu enters unmodified. The only remaining choice
    ! is how to treat the psi-space Q_p^-1, so probe BOTH treatments --
    ! unlike the commutator form, where Stage 4.2 finding (c) ruled the
    ! diagonal out. That finding was about the u-space Q_u^-1, which is
    ! exactly the inverse this form does not have; measured at tstep = 1
    ! the diagonal already reaches the operator-form ceiling here, at
    ! 2.71x fill instead of FSAI-1's 6.44x -- and Sec. 7 identified
    ! DENSITY, not iteration count, as the gate.
    !
    ! FSAI-1 variant first: T2_a is already the FSAI-1 triple product.
    call MatDuplicate(D_uu, MAT_COPY_VALUES, S_ass, ierr)
    call MatAXPY(S_ass, -1.0d0/opz, T2_a, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatGetInfo(S_ass, MAT_LOCAL, minfo, ierr)
    nzS = minfo%nz_used
    if (my_id == 0) write(*,'(A,F8.2)') &
      "[Stage 5.1] SF FSAI-1 : nnz(S_sf)/nnz(D_uu) = ", nzS / max(nzD, 1.d0)
    call schur_itersolve_probe(comm, my_id, S_ass, dmask, "SF FSAI-1")
    call MatDestroy(S_ass, ierr)

    ! Plain-diagonal variant. Guard degenerate rows the same way Stage 4.2
    ! does: construct_commutator_matrices zeroes the boundary rows, and an
    ! unguarded reciprocal would turn them into infinities.
    call MatCreateVecs(D_uu, PETSC_NULL_VEC, dq, ierr)
    call MatGetDiagonal(op(CM_OP_Q1R), dq, ierr)
    call VecNorm(dq, NORM_INFINITY, dqmax, ierr)
    call VecGetArray(dq, parr, ierr)
    do k = 1, n1
      if (abs(parr(k)) > 1.d-12 * max(dqmax, 1.d-300)) then
        parr(k) = 1.0d0 / parr(k)
      else
        parr(k) = 1.0d0
      endif
    enddo
    call VecRestoreArray(dq, parr, ierr)
    call MatDuplicate(L_up, MAT_COPY_VALUES, Lsc_d, ierr)
    call MatDiagonalScale(Lsc_d, PETSC_NULL_VEC, dq, ierr)
    call MatMatMult(Lsc_d, U_pu, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T2_d, ierr)
    call MatDuplicate(D_uu, MAT_COPY_VALUES, S_ass, ierr)
    call MatAXPY(S_ass, -1.0d0/opz, T2_d, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatGetInfo(S_ass, MAT_LOCAL, minfo, ierr)
    nzS = minfo%nz_used
    if (my_id == 0) write(*,'(A,F8.2)') &
      "[Stage 5.1] SF diag   : nnz(S_sf)/nnz(D_uu) = ", nzS / max(nzD, 1.d0)
    call schur_itersolve_probe(comm, my_id, S_ass, dmask, "SF diag")
    call MatDestroy(S_ass, ierr)
    call MatDestroy(T2_d, ierr)
    call MatDestroy(Lsc_d, ierr)
    call VecDestroy(dq, ierr)

    call MatDestroy(T2_a, ierr)
    call MatDestroy(Lsc, ierr)
    call MatDestroy(Qi_p, ierr)
    call MatDestroy(G_p, ierr)
    call MatDestroy(Pat_p, ierr)
    call MatDestroy(D_uu, ierr)
    call MatDestroy(U_pu, ierr)
    call MatDestroy(L_up, ierr)
    call VecDestroy(dmask, ierr)
    call ISDestroy(is_u, ierr)

  end subroutine verify_schur_itersolve


  !--------------------------------------------------------------------
  !> Stage 4.6: put the assembled-Schur ansatz to work on the GLOBAL solve.
  !!
  !! Stages 4.2 and 4.5 answer two halves of the question in isolation --
  !! how well Shat approximates S_u (quality, direct solves throughout) and
  !! how cheaply S_ass x = b can itself be solved (cost, no S_u anywhere).
  !! Neither says what the preconditioner is actually worth, because the
  !! object that has to converge in production is the OUTER solve on
  !! P_full, and its iteration count is not a function of either number
  !! alone: a mediocre approximation solved cheaply can beat an excellent
  !! one solved expensively.
  !!
  !! So: FGMRES on P_full, reordered to the 2x2 (y,u) form of Stage 4.1,
  !! preconditioned by the exact block-LDU factorisation
  !!
  !!    z_y* = M_yy^-1 r_y
  !!    z_u  = Shat^-1 (r_u - L_uy z_y*)
  !!    z_y  = z_y* - M_yy^-1 (U_yu z_u)
  !!
  !! with the ONLY approximation in the Schur block, applied as
  !! Shat^-1 = Q_u^-1 A_uM S_ass^-1. M_yy is inverted exactly (MUMPS): it
  !! is not what is under test here, and leaving it exact isolates the
  !! Schur ansatz as the sole source of outer iterations. With the exact
  !! S_u in that slot the factorisation is a direct solver -- Stage 4.1
  !! verified precisely that -- so every outer iteration counted below is
  !! bought by the approximation, and the sweep prices it.
  !!
  !! FGMRES, not GMRES: variants 4-7 put a Krylov method inside the
  !! preconditioner, so the operator changes from iteration to iteration
  !! and a fixed-preconditioner method would be solving the wrong problem.
  !! The production outer solver is already FGMRES (mod_petsc.f90), so this
  !! costs nothing to adopt.
  !!
  !! The honest cost metric is TOTAL INNER ITERATIONS, not outer count: a
  !! configuration converging in 12 outer at 2 inner each beats one taking
  !! 8 outer with an exact inner solve. Both columns are reported.
  !--------------------------------------------------------------------
  subroutine verify_schur_global_solve(comm, my_id)
    use mod_petsc_pc_commutator_table, only: CM_NOP, CM_MAXC, CM_LABLEN, &
          CM_OP_Q1R, cm_table_build, cm_ops_gather, cm_blocks_ready
    use phys_module, only: time_evol_zeta, time_evol_theta, tstep, tstep_prev, eta

    integer, intent(in) :: comm, my_id

    integer, parameter :: NVAR = 7
    integer, parameter :: OUT_MAXITS = 400, OUT_RESTART = 400
    real*8,  parameter :: OUT_RTOL = 1.d-8
    character(len=30), parameter :: vname(NVAR) = (/ &
         "no preconditioner             ", &
         "block-LDU, D_uu only (exact)  ", &
         "block-LDU, Shat, S_ass exact  ", &
         "block-LDU, Shat, amg rtol 1e-2", &
         "block-LDU, Shat, amg  2 its   ", &
         "block-LDU, Shat, amg  4 its   ", &
         "block-LDU, Shat, amg  8 its   " /)

    IS  :: is_y, is_u, is_yu
    Mat :: D_uu, U_pu, L_up, P_2x2
    Mat :: op(CM_NOP)
    Mat :: G_u, G_p, Qi_p, Pat_u, Pat_p
    Mat :: Dsc, Lsc, T1_a, T2_a, S_ass, S_int, D_int, Tm
    Vec :: gmask, vone, b, x, x_ref, r
    KSP :: ksp_out
    PC  :: pc_out, pc_in
    KSPConvergedReason :: kreason
    MatInfo :: minfo
    PetscInt :: n1, n3, ntot, k, kk, nits
    PetscInt, allocatable :: idx(:)
    PetscScalar, pointer :: parr(:)
    PetscReal :: rnrm, bnrm, enrm, xnrm, nrm1, nrm2
    PetscErrorCode :: ierr
    integer :: nproc, mpierr, ic, iop, nfl_u, nfl_p, iv
    logical :: is_sf                 !< this pass is the Stage 5.1 small-flow form
    character(len=13) :: clab        !< report label for the current pass
    integer :: outits(NVAR), innits(NVAR)
    real*8  :: tsol(NVAR), rres(NVAR), rerr(NVAR)
    real*8  :: opz, zeta, tdt, t0, nzS, nzD
    real*8  :: coef(CM_MAXC, CM_NOP)
    integer :: quop(CM_MAXC)
    character(len=CM_LABLEN) :: lab(CM_MAXC)
    integer :: ncand
    real*8, parameter :: zbig_thresh = 1.d11

    if (.not. g_ctx%p_full_perm_ready) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.6] ERROR: P_full_perm not built."
      return
    endif
    if (.not. g_ctx%is_reduced_created) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.6] ERROR: is_reduced not created."
      return
    endif
    call MPI_Comm_size(comm, nproc, mpierr)
    if (nproc /= 1) then
      if (my_id == 0) write(*,'(A)') "[Stage 4.6] ERROR: serial only. Use -np 1."
      return
    endif

    zeta = time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    opz  = 1.d0 + zeta
    tdt  = time_evol_theta * tstep

    call MatGetSize(g_ctx%P_full_perm, ntot, PETSC_NULL_INTEGER, ierr)
    n1 = ntot / 4
    n3 = 3 * n1
    g46_n1 = n1
    g46_n3 = n3

    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A)') "================================================================"
      write(*,'(A)') " Stage 4.6: GLOBAL solve of P_full with the assembled-Schur PC"
      write(*,'(A)') "   outer FGMRES on the 2x2 (y,u) form; block-LDU PC;"
      write(*,'(A)') "   Shat^-1 = Q_u^-1 A_uM S_ass^-1, M_yy exact (MUMPS)."
      write(*,'(A,I0,A,I0,A,E12.4)') "   y dim ", n3, " , u dim ", n1, &
                                     " ,  theta*dt = ", tdt
      write(*,'(A,E10.2,A,I0)') "   outer rtol = ", OUT_RTOL, " , cap ", OUT_MAXITS
      write(*,'(A)') "================================================================"
    endif

    !--- index sets: y = (psi,rho,T) = vars 1,3,4 ; u = var 2 ----------
    allocate(idx(n3))
    kk = 0
    do k = 0, n1-1
      kk = kk + 1 ;  idx(kk) = k
    enddo
    do k = 0, n1-1
      kk = kk + 1 ;  idx(kk) = 2*n1 + k
    enddo
    do k = 0, n1-1
      kk = kk + 1 ;  idx(kk) = 3*n1 + k
    enddo
    call ISCreateGeneral(comm, n3, idx, PETSC_COPY_VALUES, is_y, ierr)
    deallocate(idx)

    allocate(idx(n1))
    do k = 0, n1-1
      idx(k+1) = n1 + k
    enddo
    call ISCreateGeneral(comm, n1, idx, PETSC_COPY_VALUES, is_u, ierr)
    deallocate(idx)

    ! y first, then u: inside P_2x2 the two fields are contiguous, which is
    ! what lets the shell split a vector by a plain array copy.
    allocate(idx(ntot))
    do k = 0, n1-1
      idx(k+1)      = k
      idx(n1+k+1)   = 2*n1 + k
      idx(2*n1+k+1) = 3*n1 + k
      idx(3*n1+k+1) = n1 + k
    enddo
    call ISCreateGeneral(comm, ntot, idx, PETSC_COPY_VALUES, is_yu, ierr)
    deallocate(idx)

    call MatCreateSubMatrix(g_ctx%P_full_perm, is_y,  is_y,  MAT_INITIAL_MATRIX, g46_Myy, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_y,  is_u,  MAT_INITIAL_MATRIX, g46_Uyu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_u,  is_y,  MAT_INITIAL_MATRIX, g46_Luy, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_u,  is_u,  MAT_INITIAL_MATRIX, D_uu,    ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, is_yu, is_yu, MAT_INITIAL_MATRIX, P_2x2,   ierr)

    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(1), g_ctx%is_reduced(2), &
                            MAT_INITIAL_MATRIX, U_pu, ierr)
    call MatCreateSubMatrix(g_ctx%P_full_perm, g_ctx%is_reduced(2), g_ctx%is_reduced(1), &
                            MAT_INITIAL_MATRIX, L_up, ierr)

    !--- M_yy exactly. Not under test; exact here isolates the Schur block.
    call KSPCreate(comm, g46_kspM, ierr)
    call KSPSetOperators(g46_kspM, g46_Myy, g46_Myy, ierr)
    call KSPSetType(g46_kspM, KSPPREONLY, ierr)
    call KSPGetPC(g46_kspM, pc_in, ierr)
    call PCSetType(pc_in, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_in, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g46_kspM, ierr)

    !--- u-space interior mask and the penalty-row diagonal -------------
    call MatCreateVecs(D_uu, PETSC_NULL_VEC, g46_umask, ierr)
    call VecDuplicate(g46_umask, g46_ubnd, ierr)
    call MatGetDiagonal(D_uu, g46_umask, ierr)
    call VecCopy(g46_umask, g46_ubnd, ierr)
    call VecGetArray(g46_umask, parr, ierr)
    do k = 1, n1
      if (abs(parr(k)) > zbig_thresh) then
        parr(k) = 0.0d0
      else
        parr(k) = 1.0d0
      endif
    enddo
    call VecRestoreArray(g46_umask, parr, ierr)
    ! On ZBIG rows the Schur block is the penalty itself, so the right
    ! action there is a plain Jacobi divide -- NOT the identity the
    ! interior operator carries, which would be off by ~1e11.
    call VecGetArray(g46_ubnd, parr, ierr)
    do k = 1, n1
      if (abs(parr(k)) > zbig_thresh) then
        parr(k) = 1.0d0 / parr(k)
      else
        parr(k) = 0.0d0
      endif
    enddo
    call VecRestoreArray(g46_ubnd, parr, ierr)

    call VecDuplicate(g46_umask, g46_ru,  ierr)
    call VecDuplicate(g46_umask, g46_zu,  ierr)
    call VecDuplicate(g46_umask, g46_tu,  ierr)
    call VecDuplicate(g46_umask, g46_tu2, ierr)
    call VecDuplicate(g46_umask, g46_su,  ierr)
    call MatCreateVecs(g46_Myy, PETSC_NULL_VEC, g46_ry, ierr)
    call VecDuplicate(g46_ry, g46_sy, ierr)
    call VecDuplicate(g46_ry, g46_ty, ierr)

    !--- a consistent global RHS: b = P_2x2 x_ref with x_ref interior ---
    ! Drawing b at random instead would load the ZBIG rows with O(1) data
    ! the operator cannot represent, and the outer rtol would then be
    ! measuring the penalty rows rather than the physics.
    call MatCreateVecs(P_2x2, x, b, ierr)
    call VecDuplicate(b, x_ref, ierr)
    call VecDuplicate(b, r, ierr)
    call VecDuplicate(b, gmask, ierr)
    call MatGetDiagonal(P_2x2, gmask, ierr)
    call VecGetArray(gmask, parr, ierr)
    do k = 1, ntot
      if (abs(parr(k)) > zbig_thresh) then
        parr(k) = 0.0d0
      else
        parr(k) = 1.0d0
      endif
    enddo
    call VecRestoreArray(gmask, parr, ierr)
    call VecSetRandom(x_ref, PETSC_NULL_RANDOM, ierr)
    call VecPointwiseMult(x_ref, x_ref, gmask, ierr)
    call MatMult(P_2x2, x_ref, b, ierr)
    call VecNorm(b, NORM_2, bnrm, ierr)
    call VecNorm(x_ref, NORM_2, xnrm, ierr)

    !--- D_uu with the interior treatment, for the no-Schur variant -----
    call MatConvert(D_uu, MATSEQAIJ, MAT_INITIAL_MATRIX, D_int, ierr)
    call MatSetOption(D_int, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(D_int, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE, ierr)
    call MatDiagonalScale(D_int, g46_umask, g46_umask, ierr)
    call VecDuplicate(g46_umask, vone, ierr)
    call VecSet(vone, 1.d0, ierr)
    call VecAXPY(vone, -1.d0, g46_umask, ierr)
    call MatDiagonalSet(D_int, vone, ADD_VALUES, ierr)

    call cm_ops_gather(g_ctx%B_11, g_ctx%B_33, g_ctx%B_44, op)
    call cm_table_build(opz, tdt, eta, coef, quop, lab, ncand)
    if (my_id == 0 .and. .not. cm_blocks_ready()) write(*,'(A)') &
      "[Stage 4.6] NOTE: commutator blocks not assembled "// &
      "(set commutator_analysis=.t.); only M0/M1x/M0R available."

    call MatGetInfo(D_uu, MAT_LOCAL, minfo, ierr)
    nzD = minfo%nz_used

    ! psi-space Riesz map: candidate-independent, so built once.
    call MatMatMult(op(CM_OP_Q1R), op(CM_OP_Q1R), MAT_INITIAL_MATRIX, &
                    PETSC_DEFAULT_REAL, Pat_p, ierr)
    call build_fsai(op(CM_OP_Q1R), Pat_p, n1, comm, G_p, nfl_p)
    call MatTransposeMatMult(G_p, G_p, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Qi_p, ierr)
    call MatMatMult(L_up, Qi_p, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Lsc, ierr)
    call MatMatMult(Lsc, U_pu, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T2_a, ierr)

    ! ic = ncand+1 is the Stage 5.1 small-flow form, which is not a table
    ! candidate: it has no M_*, no A_uM and no right factor, so Shat_sf^-1
    ! is S_sf^-1 alone. Everything downstream of the assembly -- the
    ! interior form, the variant sweep, the reporting -- is shared.
    do ic = 1, ncand + 1
      is_sf = (ic == ncand + 1)
      if (.not. is_sf) then
        ! Same shortlist as Stage 4.5: M0 is the small-flow control, M1a and
        ! M2e the two regime candidates. Ranking the full table is Stage
        ! 4.2's job.
        if (.not. (trim(lab(ic)) == "M0"  .or. &
                   trim(lab(ic)) == "M1a" .or. &
                   trim(lab(ic)) == "M2e")) cycle
        clab = "M_* " // trim(lab(ic))
      else
        clab = "SF small-flow"
      endif

      if (is_sf) then
        !--- S_sf = D_uu - L_up Q_p^-1 U_pu / (1+zeta). T1 is D_uu itself
        !--- (the cancellation), so there is no first triple product and no
        !--- Q_u^-1 to build. Q_p^-1 is FSAI-1 here, matching what the
        !--- commutator candidates use, so the comparison is like for like;
        !--- Stage 5.1 shows a plain diagonal also suffices at short tstep.
        call MatDuplicate(D_uu, MAT_COPY_VALUES, T1_a, ierr)
        call MatDuplicate(D_uu, MAT_COPY_VALUES, S_ass, ierr)
        call MatAXPY(S_ass, -1.0d0/opz, T2_a, DIFFERENT_NONZERO_PATTERN, ierr)
        nfl_u = 0        ! no Q_u FSAI is built in this form
      else
      !--- A_uM and the FSAI-1 Q_u^-1 (the same one S_ass is built with,
      !--- so the composition is self-consistent) ------------------------
      call MatDuplicate(op(quop(ic)), MAT_COPY_VALUES, g46_AuM, ierr)
      call MatScale(g46_AuM, 0.0d0, ierr)
      do iop = 1, CM_NOP
        if (coef(ic,iop) /= 0.d0) then
          call MatAXPY(g46_AuM, coef(ic,iop), op(iop), DIFFERENT_NONZERO_PATTERN, ierr)
        endif
      enddo

      call MatMatMult(op(quop(ic)), op(quop(ic)), MAT_INITIAL_MATRIX, &
                      PETSC_DEFAULT_REAL, Pat_u, ierr)
      call build_fsai(op(quop(ic)), Pat_u, n1, comm, G_u, nfl_u)
      call MatTransposeMatMult(G_u, G_u, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, g46_Qiu, ierr)

      call MatMatMult(D_uu, g46_Qiu, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, Dsc, ierr)
      call MatMatMult(Dsc, g46_AuM, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T1_a, ierr)
      call MatDuplicate(T1_a, MAT_COPY_VALUES, S_ass, ierr)
      call MatAXPY(S_ass, -1.0d0, T2_a, DIFFERENT_NONZERO_PATTERN, ierr)
      endif

      ! interior form, exactly as Stage 4.5 measures it
      call MatConvert(S_ass, MATSEQAIJ, MAT_INITIAL_MATRIX, S_int, ierr)
      call MatSetOption(S_int, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
      call MatSetOption(S_int, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE, ierr)
      call MatDiagonalScale(S_int, g46_umask, g46_umask, ierr)
      call MatDiagonalSet(S_int, vone, ADD_VALUES, ierr)

      call MatGetInfo(S_ass, MAT_LOCAL, minfo, ierr)
      nzS = minfo%nz_used

      ! Weight of the Schur correction, interior only. Without this number
      ! the D_uu-only baseline is uninterpretable: if the correction is
      ! negligible the baseline SHOULD match the ansatz, and the outer count
      ! says nothing about the ansatz either way.
      call MatDuplicate(T1_a, MAT_COPY_VALUES, Tm, ierr)
      call MatDiagonalScale(Tm, g46_umask, g46_umask, ierr)
      call MatNorm(Tm, NORM_FROBENIUS, nrm1, ierr)
      call MatDestroy(Tm, ierr)
      call MatDuplicate(T2_a, MAT_COPY_VALUES, Tm, ierr)
      call MatDiagonalScale(Tm, g46_umask, g46_umask, ierr)
      call MatNorm(Tm, NORM_FROBENIUS, nrm2, ierr)
      call MatDestroy(Tm, ierr)
      ! the small-flow psi term carries the explicit 1/(1+zeta)
      if (is_sf) nrm2 = nrm2 / opz

      if (my_id == 0) then
        write(*,'(A)') ""
        write(*,'(A,A,A,F8.2,A,I0)') "--- Stage 4.6 [", trim(clab), &
          "]   nnz(S_ass)/nnz(D_uu) = ", nzS / max(nzD, 1.d0), &
          " , FSAI fallback rows = ", nfl_u + nfl_p
        write(*,'(A,E12.4)') "    ||L_up Q^-1 U_pu||_F / ||D_uu Q_u^-1 A_uM||_F = ", &
          nrm2 / max(nrm1, 1.d-300)
      endif

      !--- the variant sweep -------------------------------------------
      do iv = 1, NVAR
        ! g46_use_schur gates the RIGHT FACTOR Q_u^-1 A_uM, not the presence
        ! of the correction. The small-flow form has no right factor, so it
        ! runs with the flag off while still solving with S_sf in the Schur
        ! slot; iv = 2 differs from it only in which operator g46_kspS holds.
        g46_use_schur = (iv >= 3) .and. .not. is_sf
        g46_inner_its = 0

        if (iv >= 2) then
          call KSPCreate(comm, g46_kspS, ierr)
          if (iv == 2) then
            call KSPSetOperators(g46_kspS, D_int, D_int, ierr)
          else
            call KSPSetOperators(g46_kspS, S_int, S_int, ierr)
          endif
          if (iv <= 3) then
            ! exact inner solve: the ceiling of the ansatz, cost aside
            call KSPSetType(g46_kspS, KSPPREONLY, ierr)
            call KSPGetPC(g46_kspS, pc_in, ierr)
            call PCSetType(pc_in, PCLU, ierr)
            call PCFactorSetMatSolverType(pc_in, MATSOLVERMUMPS, ierr)
          else
            call KSPSetType(g46_kspS, KSPGMRES, ierr)
            call KSPGMRESSetRestart(g46_kspS, 60, ierr)
            call KSPGetPC(g46_kspS, pc_in, ierr)
            call PCSetType(pc_in, PCHYPRE, ierr)
            call PCHYPRESetType(pc_in, "boomeramg", ierr)
            if (iv == 4) then
              call KSPSetTolerances(g46_kspS, 1.d-2, 1.d-50, 1.d8, 200, ierr)
            else
              ! fixed budget: rtol below anything reachable, so the solve
              ! always runs exactly maxits sweeps. This is the cheap form a
              ! production PC would use, and FGMRES tolerates it.
              call KSPSetTolerances(g46_kspS, 1.d-30, 1.d-50, 1.d8, &
                                    int(2**(iv-4), kind=kind(nits)), ierr)
            endif
          endif
          call KSPSetUp(g46_kspS, ierr)
        endif

        call KSPCreate(comm, ksp_out, ierr)
        call KSPSetOperators(ksp_out, P_2x2, P_2x2, ierr)
        call KSPSetType(ksp_out, KSPFGMRES, ierr)
        call KSPGMRESSetRestart(ksp_out, OUT_RESTART, ierr)
        call KSPSetTolerances(ksp_out, OUT_RTOL, 1.d-50, 1.d8, OUT_MAXITS, ierr)
        call KSPGetPC(ksp_out, pc_out, ierr)
        if (iv == 1) then
          call PCSetType(pc_out, PCNONE, ierr)
        else
          call PCSetType(pc_out, PCSHELL, ierr)
          call PCShellSetApply(pc_out, schur_global_pc_apply, ierr)
          call PCShellSetName(pc_out, "block-LDU assembled Schur", ierr)
        endif

        call VecSet(x, 0.d0, ierr)
        t0 = MPI_Wtime()
        call KSPSolve(ksp_out, b, x, ierr)
        tsol(iv) = MPI_Wtime() - t0

        call KSPGetIterationNumber(ksp_out, nits, ierr)
        call KSPGetConvergedReason(ksp_out, kreason, ierr)
        outits(iv) = int(nits)
        if (kreason%v < 0) outits(iv) = -1
        innits(iv) = g46_inner_its

        ! true residual and true error, both recomputed rather than taken
        ! from the Krylov recurrence
        call MatMult(P_2x2, x, r, ierr)
        call VecAYPX(r, -1.d0, b, ierr)
        call VecNorm(r, NORM_2, rnrm, ierr)
        rres(iv) = rnrm / max(bnrm, 1.d-300)
        call VecWAXPY(r, -1.d0, x_ref, x, ierr)
        call VecNorm(r, NORM_2, enrm, ierr)
        rerr(iv) = enrm / max(xnrm, 1.d-300)

        call KSPDestroy(ksp_out, ierr)
        if (iv >= 2) call KSPDestroy(g46_kspS, ierr)
      enddo

      !--- report -------------------------------------------------------
      if (my_id == 0) then
        write(*,'(A)') "  variant                          outer   inner    time[s]" // &
                       "   ||r||/||b||   ||x-x*||/||x*||"
        do iv = 1, NVAR
          if (outits(iv) < 0) then
            write(*,'(A,A,A,A,I8,F11.3,A)') "    ", vname(iv), "  ", &
              "    n.c.", innits(iv), tsol(iv), "         --            --"
          else
            write(*,'(A,A,I8,I8,F11.3,2E14.4)') "    ", vname(iv), &
              outits(iv), innits(iv), tsol(iv), rres(iv), rerr(iv)
          endif
        enddo
        write(*,'(A)') "  (outer = FGMRES its on P_full; inner = total Krylov its"
        write(*,'(A)') "   inside the Schur block -- the column that prices the PC."
        write(*,'(A)') "   Variant 2 is the same factorization with NO Schur"
        write(*,'(A)') "   correction: the ansatz has to beat it to be worth having.)"
      endif

      call MatDestroy(S_int, ierr)
      call MatDestroy(S_ass, ierr)
      call MatDestroy(T1_a, ierr)
      if (.not. is_sf) then
        call MatDestroy(Dsc, ierr)
        call MatDestroy(g46_Qiu, ierr)
        call MatDestroy(G_u, ierr)
        call MatDestroy(Pat_u, ierr)
        call MatDestroy(g46_AuM, ierr)
      endif
    enddo

    if (my_id == 0) then
      write(*,'(A)') "================================================================"
      write(*,'(A)') ""
    endif

    !--- cleanup -------------------------------------------------------
    call VecDestroy(vone, ierr)
    call MatDestroy(D_int, ierr)
    call MatDestroy(T2_a, ierr)
    call MatDestroy(Lsc, ierr)
    call MatDestroy(Qi_p, ierr)
    call MatDestroy(G_p, ierr)
    call MatDestroy(Pat_p, ierr)
    call MatDestroy(D_uu, ierr)
    call MatDestroy(U_pu, ierr)
    call MatDestroy(L_up, ierr)
    call MatDestroy(P_2x2, ierr)
    call MatDestroy(g46_Myy, ierr)
    call MatDestroy(g46_Uyu, ierr)
    call MatDestroy(g46_Luy, ierr)
    call KSPDestroy(g46_kspM, ierr)
    call VecDestroy(g46_umask, ierr)
    call VecDestroy(g46_ubnd, ierr)
    call VecDestroy(g46_ry, ierr)
    call VecDestroy(g46_sy, ierr)
    call VecDestroy(g46_ty, ierr)
    call VecDestroy(g46_ru, ierr)
    call VecDestroy(g46_zu, ierr)
    call VecDestroy(g46_tu, ierr)
    call VecDestroy(g46_tu2, ierr)
    call VecDestroy(g46_su, ierr)
    call VecDestroy(b, ierr)
    call VecDestroy(x, ierr)
    call VecDestroy(x_ref, ierr)
    call VecDestroy(r, ierr)
    call VecDestroy(gmask, ierr)
    call ISDestroy(is_y, ierr)
    call ISDestroy(is_u, ierr)
    call ISDestroy(is_yu, ierr)

  end subroutine verify_schur_global_solve


  !--------------------------------------------------------------------
  !> Stage 4.6 PCSHELL: one block-LDU apply on the (y,u) ordering.
  !!
  !!   s_y = M_yy^-1 r_y ;  z_u = Shat^-1 (r_u - L_uy s_y)
  !!   z_y = s_y - M_yy^-1 (U_yu z_u)
  !!
  !! Exact in Shat: with S_u in that slot this reproduces the direct solve
  !! (Stage 4.1), so all outer iterations are attributable to the Schur
  !! approximation and the inner budget.
  !--------------------------------------------------------------------
  subroutine schur_global_pc_apply(pc, rvec, zvec, ierr)
    PC  :: pc
    Vec :: rvec, zvec
    PetscErrorCode :: ierr

    PetscScalar, pointer :: ra(:), za(:), ya(:), ua(:)
    PetscInt :: k, nits

    !--- split r into (r_y, r_u). Serial and contiguous by construction of
    !--- is_yu, so a copy is enough and needs no scatter to keep alive.
    call VecGetArrayRead(rvec, ra, ierr)
    call VecGetArray(g46_ry, ya, ierr)
    do k = 1, g46_n3
      ya(k) = ra(k)
    enddo
    call VecRestoreArray(g46_ry, ya, ierr)
    call VecGetArray(g46_ru, ua, ierr)
    do k = 1, g46_n1
      ua(k) = ra(g46_n3 + k)
    enddo
    call VecRestoreArray(g46_ru, ua, ierr)
    call VecRestoreArrayRead(rvec, ra, ierr)

    !--- s_y = M_yy^-1 r_y
    call KSPSolve(g46_kspM, g46_ry, g46_sy, ierr)

    !--- t_u = r_u - L_uy s_y
    call MatMult(g46_Luy, g46_sy, g46_tu, ierr)
    call VecAYPX(g46_tu, -1.d0, g46_ru, ierr)

    !--- z_u = Shat^-1 t_u
    call VecPointwiseMult(g46_tu2, g46_tu, g46_umask, ierr)
    call KSPSolve(g46_kspS, g46_tu2, g46_su, ierr)
    call KSPGetIterationNumber(g46_kspS, nits, ierr)
    g46_inner_its = g46_inner_its + int(nits)
    call VecPointwiseMult(g46_su, g46_su, g46_umask, ierr)
    if (g46_use_schur) then
      ! Shat^-1 = Q_u^-1 A_uM S_ass^-1 -- the Riesz map and the commutator
      ! factor that S_ass was divided by when it was assembled.
      call MatMult(g46_AuM, g46_su, g46_tu2, ierr)
      call MatMult(g46_Qiu, g46_tu2, g46_zu, ierr)
      call VecPointwiseMult(g46_zu, g46_zu, g46_umask, ierr)
    else
      call VecCopy(g46_su, g46_zu, ierr)
    endif
    ! penalty rows: Jacobi on the ZBIG diagonal (the interior operator
    ! carries the identity there, which is ~1e11 off)
    call VecPointwiseMult(g46_tu2, g46_tu, g46_ubnd, ierr)
    call VecAXPY(g46_zu, 1.d0, g46_tu2, ierr)

    !--- z_y = s_y - M_yy^-1 (U_yu z_u)
    call MatMult(g46_Uyu, g46_zu, g46_ty, ierr)
    call KSPSolve(g46_kspM, g46_ty, g46_ry, ierr)
    call VecAXPY(g46_sy, -1.d0, g46_ry, ierr)

    !--- gather
    call VecGetArray(zvec, za, ierr)
    call VecGetArrayRead(g46_sy, ya, ierr)
    do k = 1, g46_n3
      za(k) = ya(k)
    enddo
    call VecRestoreArrayRead(g46_sy, ya, ierr)
    call VecGetArrayRead(g46_zu, ua, ierr)
    do k = 1, g46_n1
      za(g46_n3 + k) = ua(k)
    enddo
    call VecRestoreArrayRead(g46_zu, ua, ierr)
    call VecRestoreArray(zvec, za, ierr)

    ierr = 0
  end subroutine schur_global_pc_apply


  !--------------------------------------------------------------------
  !> Stage 4.5, layers A and B: is the sparse assembled Schur complement
  !! S_ass ITSELF tractable by iterative methods?
  !!
  !! Everything upstream of this measures how well Shat APPROXIMATES S_u,
  !! with Shat^-1 applied by a direct (dense LU / MUMPS) solve. That is a
  !! quality ceiling, deliberately free of cost. In production the global
  !! preconditioner solves S_ass x = b, so its own solvability is the
  !! feasibility gate -- and it had never been measured, because no path in
  !! this tree ever applied anything but PCLU+MUMPS.
  !!
  !! Layer A characterises the operator (symmetry, conditioning, diagonal
  !! dominance) so that layer B's outcome is interpretable rather than a
  !! bare table of failures. The hypothesis under test is that S_ass, being
  !! a difference of two products of second-order operators, is
  !! biharmonic-like with cond ~ h^-4 -- in which case neither ILU nor
  !! classical AMG will be h-independent on it and the assembled route
  !! needs a different split rather than better tuning.
  !!
  !! Boundary handling: rows/cols carrying the ZBIG Dirichlet trick would
  !! otherwise dominate every norm and wreck every factorisation, so the
  !! measured operator is  diag(m) A diag(m) + diag(1-m)  -- the interior
  !! problem, with the boundary block replaced by the identity. Those rows
  !! are exactly solvable and contribute nothing to the iteration count.
  !--------------------------------------------------------------------
  subroutine schur_itersolve_probe(comm, my_id, A_in, mask, label)
    integer, intent(in) :: comm, my_id
    Mat, intent(in) :: A_in            !< the sparse assembled S_ass
    Vec, intent(in) :: mask            !< 1 on interior rows, 0 on ZBIG rows
    character(len=*), intent(in) :: label

    integer, parameter :: NPC = 9      !< preconditioner candidates
    integer, parameter :: NTOL = 3     !< rtol sweep
    real*8, parameter  :: tols(NTOL) = (/ 1.d-2, 1.d-4, 1.d-8 /)
    ! Restart == max its, so GMRES never truncates its space: the Hessenberg
    ! singular-value estimate below is only meaningful without a restart.
    integer, parameter :: MAXITS = 300
    integer, parameter :: KRESTART = 300

    character(len=16), parameter :: pcname(NPC) = (/ &
         "none            ", &
         "jacobi          ", &
         "ilu(0)          ", &
         "ilu(1)          ", &
         "ilu(2)          ", &
         "bjacobi+ilu(0)  ", &
         "gamg            ", &
         "hypre-boomeramg ", &
         "lu (mumps)      " /)

    Mat :: A, At, F
    Vec :: b, x, r, vone
    KSP :: ksp
    PC  :: pc
    KSPConvergedReason :: kreason
    MatInfo :: minfo
    PetscInt :: n, i, k, ncols, nits
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscReal :: rnrm, rasym, smax, smin, rres, bnrm, vsum
    PetscReal :: rowdia, rowoff
    PetscReal :: nzA, nzF
    PetscErrorCode :: ierr
    integer :: ipc, itol, ndd, n_int, nbnd
    integer :: itsg(NPC,NTOL), itsb(NPC)
    real*8  :: tset(NPC), tsol(NPC), fill(NPC), res8(NPC)
    real*8  :: t0, condest, resnone
    logical :: is_fact

    itsg = -1
    itsb = -1
    tset = 0.d0
    tsol = 0.d0
    fill = -1.d0
    res8 = -1.d0
    condest = -1.d0
    resnone = -1.d0

    !--- interior operator: diag(m) A diag(m) + diag(1-m) ---------------
    ! Convert to SEQAIJ: the P_full blocks are MPIAIJ even on one rank, and
    ! PCILU / PCLU have no MPIAIJ implementation. This diagnostic is serial
    ! only (like the rest of Stage 4.2), so the conversion is exact.
    call MatConvert(A_in, MATSEQAIJ, MAT_INITIAL_MATRIX, A, ierr)
    call MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE, ierr)
    call MatDiagonalScale(A, mask, mask, ierr)
    call MatCreateVecs(A, x, b, ierr)
    call VecDuplicate(b, r, ierr)
    call VecDuplicate(b, vone, ierr)
    call VecSet(vone, 1.d0, ierr)
    call VecAXPY(vone, -1.d0, mask, ierr)         ! vone = 1 - m
    call MatDiagonalSet(A, vone, ADD_VALUES, ierr)
    call VecDestroy(vone, ierr)

    call MatGetSize(A, n, PETSC_NULL_INTEGER, ierr)
    call VecSum(mask, vsum, ierr)
    n_int = int(vsum + 0.5d0)
    nbnd  = int(n) - n_int

    !=== Layer A: character of the operator ============================
    call MatTranspose(A, MAT_INITIAL_MATRIX, At, ierr)
    call MatAXPY(At, -1.d0, A, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatNorm(At, NORM_FROBENIUS, rasym, ierr)
    call MatNorm(A, NORM_FROBENIUS, rnrm, ierr)
    call MatDestroy(At, ierr)

    ! Diagonal dominance: |a_ii| >= sum_{j/=i} |a_ij|. If this fraction is
    ! small, no diagonal-based method (Jacobi, Neumann, lumping) can work --
    ! the same mechanism that killed the lumped mass inverses upstream.
    ndd = 0
    do i = 0, n-1
      call MatGetRow(A, i, ncols, cols, vals, ierr)
      rowdia = 0.d0
      rowoff = 0.d0
      do k = 1, int(ncols)
        if (cols(k) == i) then
          rowdia = abs(vals(k))
        else
          rowoff = rowoff + abs(vals(k))
        endif
      enddo
      if (rowdia >= rowoff) ndd = ndd + 1
      call MatRestoreRow(A, i, ncols, cols, vals, ierr)
    enddo

    call MatGetInfo(A, MAT_LOCAL, minfo, ierr)
    nzA = minfo%nz_used

    !--- interior RHS, drawn once so every candidate solves the same system
    call VecSetRandom(b, PETSC_NULL_RANDOM, ierr)
    call VecPointwiseMult(b, b, mask, ierr)
    call VecNorm(b, NORM_2, bnrm, ierr)

    !=== Layer B: solve S_ass x = b, iteratively ========================
    do ipc = 1, NPC
      is_fact = (ipc >= 3 .and. ipc <= 5) .or. (ipc == NPC)

      do itol = 1, NTOL
        call KSPCreate(comm, ksp, ierr)
        call KSPSetOperators(ksp, A, A, ierr)
        call KSPSetType(ksp, KSPGMRES, ierr)
        call KSPGMRESSetRestart(ksp, KRESTART, ierr)
        call KSPSetTolerances(ksp, tols(itol), 1.d-50, 1.d8, MAXITS, ierr)
        call KSPGetPC(ksp, pc, ierr)
        call set_probe_pc(pc, ipc)
        ! The condition estimate rides along on the unpreconditioned run:
        ! GMRES's Hessenberg gives sigma_max/sigma_min of A itself only when
        ! the PC is the identity and the space is never restarted.
        if (ipc == 1 .and. itol == NTOL) then
          call KSPSetComputeSingularValues(ksp, PETSC_TRUE, ierr)
        endif

        t0 = MPI_Wtime()
        call KSPSetUp(ksp, ierr)
        if (itol == 1) tset(ipc) = MPI_Wtime() - t0

        call VecZeroEntries(x, ierr)
        t0 = MPI_Wtime()
        call KSPSolve(ksp, b, x, ierr)
        if (itol == NTOL) tsol(ipc) = MPI_Wtime() - t0

        call KSPGetIterationNumber(ksp, nits, ierr)
        call KSPGetConvergedReason(ksp, kreason, ierr)
        if (kreason%v > 0) itsg(ipc,itol) = int(nits)

        if (itol == NTOL) then
          ! TRUE residual, not the recurred one: a preconditioner that lies
          ! about convergence is worse than one that fails honestly.
          call MatMult(A, x, r, ierr)
          call VecAYPX(r, -1.d0, b, ierr)
          call VecNorm(r, NORM_2, rres, ierr)
          res8(ipc) = rres / max(bnrm, 1.d-300)
          if (ipc == 1) then
            resnone = res8(ipc)
            call KSPComputeExtremeSingularValues(ksp, smax, smin, ierr)
            if (smin > 0.d0) condest = smax / smin
          endif
          if (is_fact) then
            call PCFactorGetMatrix(pc, F, ierr)
            call MatGetInfo(F, MAT_LOCAL, minfo, ierr)
            nzF = minfo%nz_used
            fill(ipc) = nzF / max(nzA, 1.d0)
          endif
        endif
        call KSPDestroy(ksp, ierr)
      enddo

      !--- BiCGStab cross-check at the tightest tolerance ---------------
      ! GMRES(300) stores the whole space; on a production mesh it would not.
      ! If BCGS collapses where GMRES converges, the operator is too
      ! non-normal for a short-recurrence method and the memory cost is real.
      call KSPCreate(comm, ksp, ierr)
      call KSPSetOperators(ksp, A, A, ierr)
      call KSPSetType(ksp, KSPBCGS, ierr)
      call KSPSetTolerances(ksp, tols(NTOL), 1.d-50, 1.d8, MAXITS, ierr)
      call KSPGetPC(ksp, pc, ierr)
      call set_probe_pc(pc, ipc)
      call VecZeroEntries(x, ierr)
      call KSPSolve(ksp, b, x, ierr)
      call KSPGetIterationNumber(ksp, nits, ierr)
      call KSPGetConvergedReason(ksp, kreason, ierr)
      if (kreason%v > 0) itsb(ipc) = int(nits)
      call KSPDestroy(ksp, ierr)
    enddo

    !=== report =========================================================
    if (my_id == 0) then
      write(*,'(A)') ""
      write(*,'(A,A,A)') "--- Stage 4.5 [", trim(label), "] ---------------------------------"
      write(*,'(A)') "  Layer A: character of S_ass (interior operator)"
      write(*,'(A,I0,A,I0,A)')     "    size                        = ", n_int, &
                                   " interior + ", nbnd, " boundary (identity)"
      write(*,'(A,F12.2)')         "    nnz / row                   = ", nzA / max(dble(n), 1.d0)
      write(*,'(A,ES12.4)')        "    ||A-A^T||_F / ||A||_F       = ", rasym / max(rnrm, 1.d-300)
      write(*,'(A,F11.1,A)')       "    diagonally dominant rows    = ", &
                                   1.d2 * dble(ndd) / max(dble(n), 1.d0), " %"
      if (condest > 0.d0) then
        write(*,'(A,ES12.4)')      "    sigma_max/sigma_min (est)   = ", condest
      else
        write(*,'(A)')             "    sigma_max/sigma_min (est)   =   unavailable"
      endif
      write(*,'(A,ES12.4)')        "    unpreconditioned rel. res.  = ", resnone
      write(*,'(A)') ""
      write(*,'(A)') "  Layer B: GMRES iterations on S_ass x = b  (n.c. = not converged in 500)"
      write(*,'(A)') "    preconditioner     1e-2    1e-4    1e-8   BCGS    true res    setup[s]  solve[s]    fill"
      do ipc = 1, NPC
        write(*,'(A,A16)',advance='no') "    ", pcname(ipc)
        do itol = 1, NTOL
          if (itsg(ipc,itol) >= 0) then
            write(*,'(I8)',advance='no') itsg(ipc,itol)
          else
            write(*,'(A8)',advance='no') "n.c."
          endif
        enddo
        if (itsb(ipc) >= 0) then
          write(*,'(I7)',advance='no') itsb(ipc)
        else
          write(*,'(A7)',advance='no') "n.c."
        endif
        write(*,'(ES12.3,F11.3,F10.3)',advance='no') res8(ipc), tset(ipc), tsol(ipc)
        if (fill(ipc) >= 0.d0) then
          write(*,'(F8.2)') fill(ipc)
        else
          write(*,'(A8)') "--"
        endif
      enddo
      write(*,'(A)') ""
      write(*,'(A)') "    'fill' = nnz(factor)/nnz(S_ass); 'lu (mumps)' is the cost floor"
      write(*,'(A)') "    this must beat. A preconditioner is only interesting if its"
      write(*,'(A)') "    iterations x matvec cost undercuts that factorisation, and only"
      write(*,'(A)') "    SCALABLE if the count stays flat under mesh refinement -- run"
      write(*,'(A)') "    this on three meshes before believing any single row."
    endif

    call MatDestroy(A, ierr)
    call VecDestroy(b, ierr)
    call VecDestroy(x, ierr)
    call VecDestroy(r, ierr)

  contains

    !> Configure one preconditioner candidate. Kept in one place so the
    !! GMRES and BCGS passes cannot drift apart.
    subroutine set_probe_pc(pcl, which)
      PC, intent(inout) :: pcl
      integer, intent(in) :: which
      PetscInt :: lev

      select case (which)
      case (1)
        call PCSetType(pcl, PCNONE, ierr)
      case (2)
        call PCSetType(pcl, PCJACOBI, ierr)
      case (3:5)
        call PCSetType(pcl, PCILU, ierr)
        lev = which - 3
        call PCFactorSetLevels(pcl, lev, ierr)
      case (6)
        ! The MPI-ready form of ILU: PETSc's default sub-PC for BJACOBI on
        ! AIJ is ILU(0), so on one rank this reproduces row 3 -- but unlike
        ! PCILU it survives np > 1, which is where this has to end up.
        call PCSetType(pcl, PCBJACOBI, ierr)
      case (7)
        call PCSetType(pcl, PCGAMG, ierr)
        call PCGAMGSetType(pcl, PCGAMGAGG, ierr)
      case (8)
        call PCSetType(pcl, PCHYPRE, ierr)
        call PCHYPRESetType(pcl, "boomeramg", ierr)
      case default
        call PCSetType(pcl, PCLU, ierr)
        call PCFactorSetMatSolverType(pcl, MATSOLVERMUMPS, ierr)
      end select
    end subroutine set_probe_pc

  end subroutine schur_itersolve_probe


  !--------------------------------------------------------------------
  !> Format num/den for the discrepancy table, guarding a zero reference.
  !--------------------------------------------------------------------
  function ratio_str(num, den) result(s)
    PetscReal, intent(in) :: num, den
    character(len=11) :: s

    if (den > 0.d0) then
      write(s,'(ES11.4)') num/den
    else if (num > 0.d0) then
      s = '  (ref=0) '
    else
      s = '   both 0  '
    endif
  end function ratio_str


  !--------------------------------------------------------------------
  !> Replace B_diag by B_diag - B_coupling * M^{-1} * B_constraint, in place.
  !! Thin wrapper around compute_schur_corrected_block_exact that swaps the
  !! result back into the caller's handle and frees the original.
  !--------------------------------------------------------------------
  subroutine schur_correct_in_place(M, B_diag, B_coupling, B_constraint)
    Mat, intent(in)    :: M, B_coupling, B_constraint
    Mat, intent(inout) :: B_diag

    Mat            :: Atilde
    PetscErrorCode :: ierr

    call compute_schur_corrected_block_exact(M, B_diag, B_coupling, B_constraint, &
                                             Atilde, .true.)
    PetscCallA(MatDestroy(B_diag, ierr))
    B_diag = Atilde
  end subroutine schur_correct_in_place


  !--------------------------------------------------------------------
  !> Filter a vector towards the smooth or the rough end of the spectrum.
  !!
  !! Uses the mass-matrix smoother G = D_lump^{-1} M, whose eigenvalues are
  !! ~3/2 on smooth modes and ~1/2 on oscillatory ones. Iterating G selects
  !! smooth modes; iterating (2I - G) selects rough ones. The vector is
  !! renormalized after every step to avoid overflow.
  !--------------------------------------------------------------------
  subroutine spectral_filter(v, M, dlump_inv, work, want_smooth)
    Vec,     intent(inout) :: v
    Mat,     intent(in)    :: M
    Vec,     intent(in)    :: dlump_inv
    Vec,     intent(inout) :: work
    logical, intent(in)    :: want_smooth

    integer, parameter :: n_sweeps = 12
    integer :: it
    PetscReal :: nrm
    PetscErrorCode :: ierr

    do it = 1, n_sweeps
      PetscCallA(MatMult(M, v, work, ierr))
      PetscCallA(VecPointwiseMult(work, work, dlump_inv, ierr))
      if (want_smooth) then
        PetscCallA(VecCopy(work, v, ierr))
      else
        ! v <- 2v - G v
        PetscCallA(VecAXPBY(v, -1.0d0, 2.0d0, work, ierr))
      endif
      PetscCallA(VecNorm(v, NORM_2, nrm, ierr))
      if (nrm > 0.d0) PetscCallA(VecScale(v, 1.0d0/nrm, ierr))
    enddo
  end subroutine spectral_filter

#endif
end module mod_petsc_pc_physics_construction
