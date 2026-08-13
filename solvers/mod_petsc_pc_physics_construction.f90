module mod_petsc_pc_physics_construction
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  use mod_petsc_pc_toroidal, only: petsc_setup_toroidal_harmonic_pc_blocked
  use mod_petsc_pc_physics_apply, only: k_a_exact_mult, s_pbp_diag_mult
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
  public :: create_reduced_index_sets

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
          CM_OP_B11, CM_OP_Q1R, CM_OP_QR, cm_table_build, cm_ops_gather, &
          cm_blocks_ready
    use mod_elt_matrix_commutator, only: CM_S1R, CM_ES1R, CM_EG1R
    use phys_module, only: time_evol_zeta, time_evol_theta, tstep, tstep_prev, eta, &
                           physics_pc_schur_assemble

    integer, intent(in) :: comm, my_id

    IS  :: is_y, is_u
    Mat :: M_yy, U_yu, L_uy, D_uu, S_u, Shat, S_dif
    Mat :: D_psi, U_pu, L_up, A_uM
    Mat :: op(CM_NOP)
    KSP :: ksp_M, ksp_psi, ksp_AuM, ksp_S, ksp_test, ksp_Q
    PC  :: pc_obj
    Vec :: e_u, t_y, w_y, sv_u, d_u, cv_t, cv_w, cv_q, dmask, pmask, bt, xt
    Vec :: prow, pprb
    Mat :: R_ex
    !--- Workstream A (sparse assembled Schur complement) ---------------
    Mat :: S_ass, T1_a, T2_a, Dsc, Lsc, Shat_a
    KSP :: ksp_Qu
    Vec :: ones_p, lq_u, lq_p, ax, ay, az, aw
    !  Second index selects how the two interior mass inverses -- Q_u^-1 in
    !  D_uu Q_u^-1 A_uM, and Q^-1 in L_up Q^-1 U_pu -- are approximated.
    !  Variants 3 and 4 keep ONE of them exact to attribute the damage.
    integer, parameter :: NLUMP = 5
    real*8  :: rassm(CM_MAXC+2,NLUMP)   !< ||Shat_eff - S_u|| / ||S_u||, interior
    real*8  :: rasscons(CM_MAXC+2,NLUMP)!< algebra check vs the operator-form Shat
    real*8  :: fillr(CM_MAXC+2,NLUMP)   !< nnz(S_ass) / nnz(D_uu); -1 if not assemblable
    integer :: itassm(CM_MAXC+2,NLUMP)
    integer :: ilump
    integer :: nzrow(NLUMP)             !< rows whose mass "inverse" had to be guarded
    character(len=22), parameter :: lumpname(NLUMP) = (/ &
         "row-sum  / row-sum    ", &
         "diagonal / diagonal   ", &
         "EXACT    / diagonal   ", &
         "diagonal / EXACT      ", &
         "Neumann-2/ diagonal   " /)
    PetscInt :: n1, n3, ntot, k
    PetscInt, allocatable :: idx(:), rows(:)
    PetscInt :: colidx(1), nits
    PetscScalar, pointer :: parr(:), parr2(:)
    PetscReal :: nrm_S, nrm_dif, chk
    integer :: nbrow
    PetscErrorCode :: ierr
    integer :: nproc, mpierr, ic, ncand, ncase, icase
    real*8  :: opz, zeta, tdt
    real*8  :: coef(CM_MAXC, CM_NOP)
    integer :: quop(CM_MAXC)
    character(len=CM_LABLEN) :: lab(CM_MAXC)
    character(len=12) :: cname(CM_MAXC+2)
    real*8  :: rel(CM_MAXC+2)
    real*8  :: rdp(CM_MAXC+2)   !< ||A_uM - D_psi|| / ||D_psi||, interior
    real*8  :: nrm_Dp
    integer :: itc(CM_MAXC+2)
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

    call cm_ops_gather(g_ctx%B_11, g_ctx%B_33, g_ctx%B_44, op)

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
          call build_shat_assembled(ic, ilump)
        enddo
      endif
    enddo

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
      write(*,'(A)') "================================================================"
      write(*,'(A)') ""
    endif

    deallocate(rows)
    call KSPDestroy(ksp_M, ierr)
    call KSPDestroy(ksp_psi, ierr)
    call KSPDestroy(ksp_Q, ierr)
    call MatDestroy(M_yy, ierr)
    call MatDestroy(U_yu, ierr)
    call MatDestroy(L_uy, ierr)
    call MatDestroy(D_uu, ierr)
    call MatDestroy(D_psi, ierr)
    call MatDestroy(U_pu, ierr)
    call MatDestroy(L_up, ierr)
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

    !> Fill Shat for one channel choice and measure it.
    !!   mode = -1 : Shat = D_uu
    !!   mode =  0 : Shat = D_uu - L_up D_psi^-1 U_pu   (exact psi channel)
    !!   mode >  0 : Shat = D_uu - L_up U_pu M_*(mode)^-1
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

        if (mode == -1) then
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
    !------------------------------------------------------------------
    subroutine build_shat_assembled(mode, ilmp)
      integer, intent(in) :: mode
      integer, intent(in) :: ilmp   !< 1 = row-sum lumping, 2 = diagonal extraction
      integer :: kk, iop, itry, mu, mp
      logical :: sparse_ok
      KSPConvergedReason :: kreason
      PetscReal :: rn, rd, racc
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
      !   lhs = Shat x       = D_uu x - L_up Q^-1 U_pu A_uM^-1 Q_u x
      !   rhs = S_ass y      = D_uu Q_u^-1 A_uM y - L_up Q^-1 U_pu y,
      !                        y = A_uM^-1 Q_u x
      ! Both use CONSISTENT inverses, so they must agree to round-off.
      racc = 0.d0
      do itry = 1, 3
        call VecSetRandom(ax, PETSC_NULL_RANDOM, ierr)
        call VecPointwiseMult(ax, ax, dmask, ierr)
        ! y = A_uM^-1 Q_u x   (shared by both sides)
        call MatMult(op(quop(mode)), ax, ay, ierr)
        call KSPSolve(ksp_AuM, ay, az, ierr)          ! az = y
        ! common second term: L_up Q^-1 U_pu y
        call MatMult(U_pu, az, cv_t, ierr)
        call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
        call MatMult(L_up, cv_q, aw, ierr)            ! aw = second term
        ! lhs = D_uu x - aw
        call MatMult(D_uu, ax, sv_u, ierr)
        call VecAXPY(sv_u, -1.0d0, aw, ierr)          ! sv_u = Shat x
        call VecNorm(sv_u, NORM_2, rd, ierr)
        ! rhs = D_uu Q_u^-1 A_uM y - aw
        call MatMult(A_uM, az, cv_t, ierr)
        call KSPSolve(ksp_Qu, cv_t, cv_w, ierr)
        call MatMult(D_uu, cv_w, d_u, ierr)
        call VecAXPY(d_u, -1.0d0, aw, ierr)           ! d_u = S_ass y
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
        case default ; mu = 3 ; mp = 2
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

      ! S_ass is a genuine sparse matrix only when BOTH inverses are diagonal.
      ! Variants 3/4 keep one exact, so they exist as an operator only and
      ! report no fill -- they are attribution experiments, not candidates.
      sparse_ok = ((mu == 1 .or. mu == 2) .and. mp > 0)
      if (sparse_ok) then
        ! T1 = (D_uu diag(1/mass(Q_u))) A_uM
        call MatDuplicate(D_uu, MAT_COPY_VALUES, Dsc, ierr)
        call MatDiagonalScale(Dsc, PETSC_NULL_VEC, lq_u, ierr)
        call MatMatMult(Dsc, A_uM, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T1_a, ierr)
        ! T2 = (L_up diag(1/mass(Q))) U_pu
        call MatDuplicate(L_up, MAT_COPY_VALUES, Lsc, ierr)
        call MatDiagonalScale(Lsc, PETSC_NULL_VEC, lq_p, ierr)
        call MatMatMult(Lsc, U_pu, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, T2_a, ierr)

        call MatDuplicate(T1_a, MAT_COPY_VALUES, S_ass, ierr)
        call MatAXPY(S_ass, -1.0d0, T2_a, DIFFERENT_NONZERO_PATTERN, ierr)

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
        call MatMult(op(quop(mode)), e_u, cv_t, ierr)
        call KSPSolve(ksp_AuM, cv_t, az, ierr)          ! az = A_uM^-1 Q_u e_k
        ! S_ass az, applied term by term so an exact inverse can be mixed in
        call MatMult(A_uM, az, cv_t, ierr)
        if (mu == 0) then
          call KSPSolve(ksp_Qu, cv_t, cv_w, ierr)
        else if (mu == 3) then
          ! Q_u^-1 r  by npoly steps of diagonally-preconditioned Richardson
          ! (a Neumann series in D^-1 Q_u). Still a POLYNOMIAL in Q_u, hence
          ! still sparse-assemblable in principle -- just wider.
          call VecPointwiseMult(cv_w, cv_t, lq_u, ierr)
          do itry = 1, npoly
            call MatMult(op(quop(mode)), cv_w, ay, ierr)
            call VecAYPX(ay, -1.0d0, cv_t, ierr)          ! ay = r - Q_u w
            call VecPointwiseMult(ay, ay, lq_u, ierr)
            call VecAXPY(cv_w, 1.0d0, ay, ierr)
          enddo
        else
          call VecPointwiseMult(cv_w, cv_t, lq_u, ierr)
        endif
        call MatMult(D_uu, cv_w, sv_u, ierr)
        call MatMult(U_pu, az, cv_t, ierr)
        if (mp == 0) then
          call KSPSolve(ksp_Q, cv_t, cv_q, ierr)
        else
          call VecPointwiseMult(cv_q, cv_t, lq_p, ierr)
        endif
        call MatMult(L_up, cv_q, aw, ierr)
        call VecAXPY(sv_u, -1.0d0, aw, ierr)
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
        call MatDestroy(Dsc, ierr)
        call MatDestroy(Lsc, ierr)
      endif
      call VecDestroy(ones_p, ierr)
      call VecDestroy(lq_u, ierr)
      call VecDestroy(lq_p, ierr)
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
