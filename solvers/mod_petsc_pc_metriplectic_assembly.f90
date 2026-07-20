module mod_petsc_pc_metriplectic_assembly
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_metriplectic_ctx, only: type_metriplectic_ctx, g_mctx
  implicit none
  private

  public :: metriplectic_create_matrices
  public :: metriplectic_assemble
  public :: metriplectic_compose_P_u
  ! --- shared helpers (used by the sweep and the analysis module) ---
  public :: create_index_sets
  public :: extract_block
  public :: setup_lu_ksp
  public :: setup_mass_ksp
  ! --- stage B/C sweep build (spec Sec. 6; note Sec. "sweep") ---
  public :: metriplectic_build_sweep
  public :: metriplectic_refresh_Pu

  ! --- TEMPORARY diagnostic: per-block stiffness-norm log (note Sec.
  ! "Coefficient table"). Flip to .false. to disable; delete this block
  ! and its call site in metriplectic_build_sweep once no longer needed. ---
  logical, parameter :: LOG_STIFFNESS_METRICS = .true.
  character(len=*), parameter :: STIFFNESS_LOG_FILE = 'metriplectic_stiffness.dat'
  logical, save :: g_stiffness_header_written = .false.

contains

  !--------------------------------------------------------------------
  !> Create one 1-var MPIBAIJ matrix with sparsity derived from a_mat
  !! (same logic as mod_petsc_pc_physics_element::petsc_create_pc_matrix,
  !!  n_vars = 1; replicated privately to keep the families independent).
  !--------------------------------------------------------------------
  subroutine create_1v_matrix(petsc_A, a_mat)
    use data_structure,  only: type_SP_MATRIX
    use mod_parameters,  only: n_var

    Mat,                   intent(out) :: petsc_A
    type(type_SP_MATRIX),  intent(in)  :: a_mat

    integer :: i, j
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, col_block
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size    = a_mat%block_size / n_var          ! = n_tor (1-var)
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local       = n_block_local * block_size
    n_global      = a_mat%ng / n_var

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
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: metriplectic create_1v_matrix ierr=", ierr
    ! BC handling uses MatZeroRowsColumns; allow the resulting fill changes.
    call MatSetOption(petsc_A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(petsc_A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    deallocate(d_nnz, o_nnz)
  end subroutine create_1v_matrix


  !> Create the six element-assembled operator matrices (sparsity only).
  subroutine metriplectic_create_matrices(a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_SP_MATRIX), intent(in) :: a_mat

    g_mctx%comm = a_mat%comm
    call create_1v_matrix(g_mctx%M_psi,     a_mat)
    call create_1v_matrix(g_mctx%D_op,      a_mat)
    call create_1v_matrix(g_mctx%Dp_op,     a_mat)
    call create_1v_matrix(g_mctx%Dp_struct, a_mat)
    call create_1v_matrix(g_mctx%L_rho,     a_mat)
    call create_1v_matrix(g_mctx%W_para,    a_mat)
  end subroutine metriplectic_create_matrices


  !--------------------------------------------------------------------
  !> Assemble (or re-assemble) the six operators from element data and
  !! cache the Gears-normalized time factor tau = theta*dt/(1+zeta)
  !! (note, Sec. "Elimination": tau replaces theta*dt everywhere).
  !--------------------------------------------------------------------
  subroutine metriplectic_assemble(my_id, local_elms, n_local_elms, a_mat)
    use construct_metriplectic_matrix_mod, only: construct_metriplectic_matrices
    use data_structure, only: type_SP_MATRIX
    use phys_module,    only: time_evol_theta, time_evol_zeta, tstep

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat

    PetscErrorCode :: ierr

    if (g_mctx%matrices_ready) then
      PetscCallA(MatDestroy(g_mctx%M_psi,     ierr))
      PetscCallA(MatDestroy(g_mctx%D_op,      ierr))
      PetscCallA(MatDestroy(g_mctx%Dp_op,     ierr))
      PetscCallA(MatDestroy(g_mctx%Dp_struct, ierr))
      PetscCallA(MatDestroy(g_mctx%L_rho,     ierr))
      PetscCallA(MatDestroy(g_mctx%W_para,    ierr))
    endif
    call metriplectic_create_matrices(a_mat)

    call construct_metriplectic_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                         g_mctx%M_psi, g_mctx%D_op, g_mctx%Dp_op, &
                                         g_mctx%Dp_struct, g_mctx%L_rho, g_mctx%W_para)

    g_mctx%dt_theta = time_evol_theta * tstep / (1.d0 + time_evol_zeta)

    ! Work vectors (1-var layout), created once from the assembled matrices
    if (.not. g_mctx%work_created) then
      PetscCallA(MatCreateVecs(g_mctx%M_psi, g_mctx%wv_psi_1, g_mctx%wv_psi_2, ierr))
      PetscCallA(MatCreateVecs(g_mctx%L_rho, g_mctx%wv_u_1,   g_mctx%wv_u_2,   ierr))
      g_mctx%work_created = .true.
    endif

    g_mctx%matrices_ready = .true.
    if (my_id == 0) write(*,'(A,ES12.4)') &
      "[Metriplectic] operators assembled; tau = theta*dt/(1+zeta) = ", g_mctx%dt_theta

    ! TEMPORARY diagnostic: fires here (not in metriplectic_build_sweep) so
    ! it also covers the analysis-only path (use_metriplectic_pc=.false.,
    ! metriplectic_analysis=.true.), which never calls build_sweep.
    if (LOG_STIFFNESS_METRICS) &
      call log_stiffness_metrics(my_id, 1.d0 + time_evol_zeta, &
                                 g_mctx%dt_theta * (1.d0 + time_evol_zeta))
  end subroutine metriplectic_assemble


  !--------------------------------------------------------------------
  !> Compose P_u = L_rho + tau^2 * W_para without re-assembly.
  !! Fresh copy each call so a dt-sweep can call this repeatedly.
  !! BC rows: L_rho carries diag 1, W_para diag 0 (symmetric elimination
  !! with diag_value=0) — P_u holds exactly unit Dirichlet rows at any tau.
  !--------------------------------------------------------------------
  subroutine metriplectic_compose_P_u(dt_theta_in)
    real*8, intent(in) :: dt_theta_in
    PetscErrorCode :: ierr

    if (g_mctx%P_u_created) then
      PetscCallA(MatDestroy(g_mctx%P_u, ierr))
    endif
    PetscCallA(MatDuplicate(g_mctx%L_rho, MAT_COPY_VALUES, g_mctx%P_u, ierr))
    PetscCallA(MatAXPY(g_mctx%P_u, dt_theta_in**2, g_mctx%W_para, &
                       DIFFERENT_NONZERO_PATTERN, ierr))
    g_mctx%P_u_created = .true.
  end subroutine metriplectic_compose_P_u


  !====================================================================
  ! Shared helpers (moved from the analysis module so the sweep and the
  ! analysis checks use one implementation; signatures unchanged).
  !====================================================================

  !> Variable index sets on A_full's layout (rank-local strided blocks:
  !! node-block i, variable v, mode m -> rstart + i*block_size + (v-1)*n_tor + m).
  subroutine create_index_sets(A_full, comm)
    use mod_parameters, only: n_var, n_tor
    Mat,     intent(in) :: A_full
    integer, intent(in) :: comm

    PetscInt :: n_local, rstart, rend
    PetscInt :: block_size, n_block_local, n_var_dofs
    PetscInt, allocatable :: indices(:)
    PetscErrorCode :: ierr
    integer :: v, i, m, k

    PetscCallA(MatGetLocalSize(A_full, n_local, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetOwnershipRange(A_full, rstart, rend, ierr))

    block_size    = n_var * n_tor
    n_block_local = n_local / block_size
    n_var_dofs    = n_block_local * n_tor

    allocate(indices(n_var_dofs))
    do v = 1, 6
      k = 0
      do i = 0, n_block_local - 1
        do m = 0, n_tor - 1
          k = k + 1
          indices(k) = rstart + i * block_size + (v-1) * n_tor + m
        enddo
      enddo
      PetscCallA(ISCreateGeneral(comm, n_var_dofs, indices, PETSC_COPY_VALUES, &
                                 g_mctx%is_var(v), ierr))
    enddo
    deallocate(indices)
    g_mctx%is_created = .true.
  end subroutine create_index_sets


  subroutine extract_block(A_full, eq_row, var_col, B)
    Mat, intent(in)    :: A_full
    integer, intent(in) :: eq_row, var_col
    Mat, intent(out)    :: B
    PetscErrorCode :: ierr
    PetscCallA(MatCreateSubMatrix(A_full, g_mctx%is_var(eq_row), g_mctx%is_var(var_col), &
                                  MAT_INITIAL_MATRIX, B, ierr))
  end subroutine extract_block


  !> Direct MUMPS solver (LU, or Cholesky when symmetric=.true.).
  subroutine setup_lu_ksp(A, ksp, comm, symmetric)
    Mat,     intent(in)  :: A
    KSP,     intent(out) :: ksp
    integer, intent(in)  :: comm
    logical, intent(in)  :: symmetric
    PC :: pc
    PetscErrorCode :: ierr

    PetscCallA(KSPCreate(comm, ksp, ierr))
    PetscCallA(KSPSetOperators(ksp, A, A, ierr))
    PetscCallA(KSPSetType(ksp, KSPPREONLY, ierr))
    PetscCallA(KSPGetPC(ksp, pc, ierr))
    if (symmetric) then
      PetscCallA(PCSetType(pc, PCCHOLESKY, ierr))
    else
      PetscCallA(PCSetType(pc, PCLU, ierr))
    endif
    PetscCallA(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr))
    PetscCallA(KSPSetUp(ksp, ierr))
  end subroutine setup_lu_ksp


  !> Consistent-mass solve: CG + Jacobi, tight tolerance.
  subroutine setup_mass_ksp(A, ksp, comm)
    Mat,     intent(in)  :: A
    KSP,     intent(out) :: ksp
    integer, intent(in)  :: comm
    PC :: pc
    PetscErrorCode :: ierr

    PetscCallA(KSPCreate(comm, ksp, ierr))
    PetscCallA(KSPSetOperators(ksp, A, A, ierr))
    PetscCallA(KSPSetType(ksp, KSPCG, ierr))
    PetscCallA(KSPSetTolerances(ksp, 1.d-12, 1.d-50, PETSC_CURRENT_REAL, 500, ierr))
    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCJACOBI, ierr))
    PetscCallA(KSPSetUp(ksp, ierr))
  end subroutine setup_mass_ksp


  !====================================================================
  ! Stage B/C: sweep build/refresh (spec Sec. 6.1; plan Task 3).
  !
  ! S-half = true-Jacobian constraint pairs (mixed/Ciarlet-Raviart form):
  !   A_pair_psij = [B11 B13; B31 B33],  A_pair_uw = [B22 B24; B42 B44].
  ! HARD CONSTRAINT (spec 6.1.2): the folds B13*B33^-1*B31 etc. are never
  ! formed; the pairs are solved coupled (MUMPS LU).
  ! Factored ONCE across rebuilds: B33/B44 (constant R-weighted constraint
  ! masses). Refreshed each build: pairs, B55/B66, ksp_Mpsi, P_u.
  !====================================================================
  subroutine metriplectic_build_sweep(A_full, my_id)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    use phys_module,    only: time_evol_zeta, metriplectic_analysis, &
                              metriplectic_khalf, metriplectic_ps_inner_it, &
                              metriplectic_ps_inner_tol
    use mod_petsc_pc_metriplectic_apply, only: metriplectic_build_ps_shell

    Mat,     intent(in) :: A_full
    integer, intent(in) :: my_id

    Mat :: B11, B13, B22, B24
    Mat :: Mps, Ds, Dps, Ls
    Mat :: B22c, Wp_aij
    Mat :: mats2(4), mats4(16), A_nest
    PetscErrorCode :: ierr
    integer :: comm
    real*8 :: tdt, opz
    logical :: build_k2, build_k4, build_ps, build_uw_lu

    call PetscObjectGetComm(A_full, comm, ierr)
    if (.not. g_mctx%is_created) call create_index_sets(A_full, comm)
    g_mctx%khalf_mode   = metriplectic_khalf
    g_mctx%ps_inner_it  = metriplectic_ps_inner_it
    g_mctx%ps_inner_tol = metriplectic_ps_inner_tol
    opz = 1.d0 + time_evol_zeta
    tdt = g_mctx%dt_theta * opz

    ! Build set (spec Sec. 7.3): production builds only what the selected
    ! mode applies; analysis builds everything so one run compares modes.
    ! The (psi,j) pair LU is unconditional (PS pivot / S-half / analysis);
    ! the (u,w) pair MATRIX is unconditional (P_uw true blocks + S_uw
    ! mult), its LU only where diss_half_solve runs.
    build_k2    = (g_mctx%khalf_mode == 'K2') .or. metriplectic_analysis
    build_k4    = (g_mctx%khalf_mode == 'K4') .or. metriplectic_analysis
    build_ps    = (g_mctx%khalf_mode == 'PS') .or. metriplectic_analysis
    build_uw_lu = (g_mctx%khalf_mode == 'K2') .or. &
                  (g_mctx%khalf_mode == 'PU') .or. metriplectic_analysis

    ! --- destroy per-build objects from a previous build (guards mirror
    !     the build set, which is constant within a run) ---
    if (g_mctx%sweep_ready) then
      PetscCallA(KSPDestroy(g_mctx%ksp_pair_psij, ierr))
      if (build_uw_lu) then
        PetscCallA(KSPDestroy(g_mctx%ksp_pair_uw, ierr))
      endif
      if (build_k2) then
        PetscCallA(KSPDestroy(g_mctx%ksp_kideal, ierr))
        PetscCallA(MatDestroy(g_mctx%A_kideal,   ierr))
      endif
      if (build_k4) then
        PetscCallA(KSPDestroy(g_mctx%ksp_k4, ierr))
        PetscCallA(MatDestroy(g_mctx%A_k4,   ierr))
      endif
      if (build_ps) then
        PetscCallA(KSPDestroy(g_mctx%ksp_Puw, ierr))
        PetscCallA(MatDestroy(g_mctx%P_uw,    ierr))
      endif
      PetscCallA(KSPDestroy(g_mctx%ksp_B55,       ierr))
      PetscCallA(KSPDestroy(g_mctx%ksp_B66,       ierr))
      PetscCallA(KSPDestroy(g_mctx%ksp_Mpsi,      ierr))
      PetscCallA(MatDestroy(g_mctx%A_pair_psij, ierr))
      PetscCallA(MatDestroy(g_mctx%A_pair_uw,   ierr))
      PetscCallA(MatDestroy(g_mctx%B_31s, ierr))
      PetscCallA(MatDestroy(g_mctx%B_42s, ierr))
      PetscCallA(MatDestroy(g_mctx%B_52s, ierr))
      PetscCallA(MatDestroy(g_mctx%B_55s, ierr))
      PetscCallA(MatDestroy(g_mctx%B_62s, ierr))
      PetscCallA(MatDestroy(g_mctx%B_66s, ierr))
      g_mctx%sweep_ready = .false.
    endif

    ! --- extract true sub-blocks (sweep-owned copies) ---
    call extract_block(A_full, var_psi, var_psi, B11)
    call extract_block(A_full, var_psi, var_zj,  B13)
    call extract_block(A_full, var_u,   var_u,   B22)
    call extract_block(A_full, var_u,   var_w,   B24)
    call extract_block(A_full, var_zj,  var_psi, g_mctx%B_31s)
    call extract_block(A_full, var_w,   var_u,   g_mctx%B_42s)
    call extract_block(A_full, var_rho, var_u,   g_mctx%B_52s)
    call extract_block(A_full, var_rho, var_rho, g_mctx%B_55s)
    call extract_block(A_full, var_T,   var_u,   g_mctx%B_62s)
    call extract_block(A_full, var_T,   var_T,   g_mctx%B_66s)
    if (.not. g_mctx%sweep_once_done) then
      call extract_block(A_full, var_zj, var_zj, g_mctx%B_33s)
      call extract_block(A_full, var_w,  var_w,  g_mctx%B_44s)
    endif

    ! --- coupled pair matrices (2x2 nest -> AIJ; the mixed form) ---
    mats2 = PETSC_NULL_MAT
    mats2(1) = B11;           mats2(2) = B13
    mats2(3) = g_mctx%B_31s;  mats2(4) = g_mctx%B_33s
    PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS, 2, PETSC_NULL_IS, mats2, A_nest, ierr))
    PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_mctx%A_pair_psij, ierr))
    PetscCallA(MatDestroy(A_nest, ierr))

    mats2(1) = B22;           mats2(2) = B24
    mats2(3) = g_mctx%B_42s;  mats2(4) = g_mctx%B_44s
    PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS, 2, PETSC_NULL_IS, mats2, A_nest, ierr))
    PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_mctx%A_pair_uw, ierr))
    PetscCallA(MatDestroy(A_nest, ierr))
    ! (B11/B13/B22/B24 stay alive: A_k4 and P_uw below reuse them)

    ! --- scaled model-wave blocks tdt*D, tdt*Dp (shared by K2 and K4) ---
    if (build_k2 .or. build_k4) then
      PetscCallA(MatConvert(g_mctx%D_op, MATMPIAIJ, MAT_INITIAL_MATRIX, Ds, ierr))
      PetscCallA(MatScale(Ds, tdt, ierr))
      PetscCallA(MatConvert(g_mctx%Dp_op, MATMPIAIJ, MAT_INITIAL_MATRIX, Dps, ierr))
      PetscCallA(MatScale(Dps, tdt, ierr))
    endif

    ! --- coupled model-Alfven K-half [(1+z)M_psi, tdt D; tdt Dp, -(1+z)L_rho]
    !     (mode 'K2'; same construction as the T3b ideal-reference upper block) ---
    if (build_k2) then
      PetscCallA(MatConvert(g_mctx%M_psi, MATMPIAIJ, MAT_INITIAL_MATRIX, Mps, ierr))
      PetscCallA(MatScale(Mps, opz, ierr))
      PetscCallA(MatConvert(g_mctx%L_rho, MATMPIAIJ, MAT_INITIAL_MATRIX, Ls, ierr))
      PetscCallA(MatScale(Ls, -opz, ierr))
      mats2(1) = Mps;  mats2(2) = Ds
      mats2(3) = Dps;  mats2(4) = Ls
      PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS, 2, PETSC_NULL_IS, mats2, A_nest, ierr))
      PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_mctx%A_kideal, ierr))
      PetscCallA(MatDestroy(A_nest, ierr))
      PetscCallA(MatDestroy(Mps, ierr)); PetscCallA(MatDestroy(Ls, ierr))
    endif

    ! --- 4-field model K (mode 'K4' reference/fallback; spec Sec. 7.2 table,
    !     note eq. (AK4)): true pair blocks + model wave coupling, nest in
    !     the (psi,u,j,w) pack_4v order; PETSC_NULL_MAT slots = zero blocks ---
    if (build_k4) then
      mats4 = PETSC_NULL_MAT
      mats4( 1) = B11;           mats4( 2) = Ds
      mats4( 3) = B13
      mats4( 5) = Dps;           mats4( 6) = B22
      mats4( 8) = B24
      mats4( 9) = g_mctx%B_31s;  mats4(11) = g_mctx%B_33s
      mats4(14) = g_mctx%B_42s;  mats4(16) = g_mctx%B_44s
      PetscCallA(MatCreateNest(comm, 4, PETSC_NULL_IS, 4, PETSC_NULL_IS, mats4, A_nest, ierr))
      PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_mctx%A_k4, ierr))
      PetscCallA(MatDestroy(A_nest, ierr))
    endif

    ! --- sparse pair-Schur preconditioner P_uw (mode 'PS'; note eq. (Puw)):
    !     [B22 - (1+z)tau^2 W_para, B24; B42, B44]. MINUS: the JOREK u-row is
    !     negatively oriented, so the SPD-convention +tau^2 W_para of P_u
    !     enters subtracted (note Sec. Puw). W_para BC rows are zero
    !     (diag_value=0), so B22's BC rows pass through unchanged. ---
    if (build_ps) then
      PetscCallA(MatDuplicate(B22, MAT_COPY_VALUES, B22c, ierr))
      PetscCallA(MatConvert(g_mctx%W_para, MATMPIAIJ, MAT_INITIAL_MATRIX, Wp_aij, ierr))
      PetscCallA(MatAXPY(B22c, -opz*g_mctx%dt_theta**2, Wp_aij, &
                         DIFFERENT_NONZERO_PATTERN, ierr))
      PetscCallA(MatDestroy(Wp_aij, ierr))
      mats2(1) = B22c;          mats2(2) = B24
      mats2(3) = g_mctx%B_42s;  mats2(4) = g_mctx%B_44s
      PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS, 2, PETSC_NULL_IS, mats2, A_nest, ierr))
      PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_mctx%P_uw, ierr))
      PetscCallA(MatDestroy(A_nest, ierr))
      PetscCallA(MatDestroy(B22c, ierr))
    endif

    if (build_k2 .or. build_k4) then
      PetscCallA(MatDestroy(Ds, ierr)); PetscCallA(MatDestroy(Dps, ierr))
    endif
    PetscCallA(MatDestroy(B11, ierr)); PetscCallA(MatDestroy(B13, ierr))
    PetscCallA(MatDestroy(B22, ierr)); PetscCallA(MatDestroy(B24, ierr))

    ! --- solvers ---
    call setup_lu_ksp(g_mctx%A_pair_psij, g_mctx%ksp_pair_psij, comm, symmetric=.false.)
    if (build_uw_lu) &
      call setup_lu_ksp(g_mctx%A_pair_uw, g_mctx%ksp_pair_uw, comm, symmetric=.false.)
    if (build_k2) &
      call setup_lu_ksp(g_mctx%A_kideal, g_mctx%ksp_kideal, comm, symmetric=.false.)
    if (build_k4) &
      call setup_lu_ksp(g_mctx%A_k4, g_mctx%ksp_k4, comm, symmetric=.false.)
    if (build_ps) &
      call setup_lu_ksp(g_mctx%P_uw, g_mctx%ksp_Puw, comm, symmetric=.false.)
    call setup_lu_ksp(g_mctx%B_55s, g_mctx%ksp_B55, comm, symmetric=.false.)
    call setup_lu_ksp(g_mctx%B_66s, g_mctx%ksp_B66, comm, symmetric=.false.)
    call setup_mass_ksp(g_mctx%M_psi, g_mctx%ksp_Mpsi, comm)
    if (.not. g_mctx%sweep_once_done) then
      call setup_lu_ksp(g_mctx%B_33s, g_mctx%ksp_B33, comm, symmetric=.false.)
      call setup_lu_ksp(g_mctx%B_44s, g_mctx%ksp_B44, comm, symmetric=.false.)
      PetscCallA(MatCreateVecs(g_mctx%A_pair_psij, g_mctx%wv_pair_psij_1, &
                               g_mctx%wv_pair_psij_2, ierr))
      PetscCallA(MatCreateVecs(g_mctx%A_pair_uw,   g_mctx%wv_pair_uw_1, &
                               g_mctx%wv_pair_uw_2, ierr))
      PetscCallA(MatCreateVecs(g_mctx%A_pair_uw,   g_mctx%wv_uw_1, &
                               g_mctx%wv_uw_2, ierr))
      if (build_k2) then
        PetscCallA(MatCreateVecs(g_mctx%A_kideal, g_mctx%wv_kid_1, g_mctx%wv_kid_2, ierr))
      endif
      if (build_k4) then
        PetscCallA(MatCreateVecs(g_mctx%A_k4, g_mctx%wv_k4_1, g_mctx%wv_k4_2, ierr))
      endif
      PetscCallA(MatCreateVecs(g_mctx%B_55s, g_mctx%wv_rho_1, g_mctx%wv_rho_2, ierr))
      PetscCallA(MatCreateVecs(g_mctx%B_66s, g_mctx%wv_T_1,   g_mctx%wv_T_2,   ierr))
      g_mctx%sweep_once_done = .true.
    endif

    ! --- stage-D shell + inner Schur solver (created once; dereferences
    !     the current g_mctx handles at call time, so no rebuild needed) ---
    if (build_ps) call metriplectic_build_ps_shell(comm)

    ! --- P_u composed and factored at the run's tau (needed by the Schur
    !     K-half path and the analysis checks; skipped in pure coupled runs) ---
    if ((g_mctx%khalf_mode == 'PU') .or. metriplectic_analysis) then
      call metriplectic_refresh_Pu(g_mctx%dt_theta)
    endif

    g_mctx%sweep_ready = .true.
    if (my_id == 0) then
      select case (g_mctx%khalf_mode)
      case ('PS')
        write(*,'(A,ES12.4)') &
          "[Metriplectic] built: pair-Schur LDU K-half (P_uw); tau = ", g_mctx%dt_theta
      case ('K4')
        write(*,'(A,ES12.4)') &
          "[Metriplectic] built: 4-field LU reference K-half; tau = ", g_mctx%dt_theta
      case ('K2')
        write(*,'(A,ES12.4)') &
          "[Metriplectic] sweep built (pair solves + coupled K-half); tau = ", g_mctx%dt_theta
      case default   ! 'PU'
        write(*,'(A,ES12.4)') &
          "[Metriplectic] sweep built (pair solves + P_u Schur K-half); tau = ", g_mctx%dt_theta
      end select
    endif
  end subroutine metriplectic_build_sweep


  !--------------------------------------------------------------------
  !> (Re)compose P_u at tau_in and (re)factor the sweep's ksp_Pu.
  !! Used by the build (run tau) and by T5a/T5b compose-lag sweeps.
  !--------------------------------------------------------------------
  subroutine metriplectic_refresh_Pu(tau_in)
    real*8, intent(in) :: tau_in
    PetscErrorCode :: ierr

    if (g_mctx%tau_Pu >= 0.d0) then
      PetscCallA(KSPDestroy(g_mctx%ksp_Pu, ierr))
      PetscCallA(MatDestroy(g_mctx%Pu_aij, ierr))
    endif
    call metriplectic_compose_P_u(tau_in)
    PetscCallA(MatConvert(g_mctx%P_u, MATMPIAIJ, MAT_INITIAL_MATRIX, g_mctx%Pu_aij, ierr))
    PetscCallA(MatSetOption(g_mctx%Pu_aij, MAT_SYMMETRIC, PETSC_TRUE, ierr))
    call setup_lu_ksp(g_mctx%Pu_aij, g_mctx%ksp_Pu, g_mctx%comm, symmetric=.true.)
    g_mctx%tau_Pu = tau_in
  end subroutine metriplectic_refresh_Pu


  !--------------------------------------------------------------------
  !> TEMPORARY diagnostic. Per-block Frobenius-norm proxy for how much
  !! each element-assembled operator (note Sec. "Coefficient table":
  !! M_psi, D, D', L_rho, W_para) contributes to the K-half/S-half
  !! stiffness, appended every metriplectic_assemble call (i.e. whenever
  !! metriplectic_analysis .or. use_metriplectic_pc, jorek2_main.f90) so
  !! the balance can be tracked over a run in either mode. Scalars match
  !! the coefficients A_kideal / P_u actually assemble in
  !! metriplectic_build_sweep (note Sec. "halves"): (1+zeta) on
  !! M_psi/L_rho, theta*dt on D/D', tau^2 on W_para.
  !!
  !! Raw Frobenius norms alone are not comparable across blocks of
  !! different differential order (M_psi 0th, D/Dp coupling, L_rho 2nd,
  !! W_para 4th-ish parabolized) since they scale differently with mesh
  !! h; ratio_* divides the coefficient-scaled norms by the raw M_psi
  !! norm (mass-normalized, generalized-eigenvalue-style) before frac_*
  !! is formed, so frac_* reflects genuine stiffness balance rather than
  !! an order-driven bias toward W_para.
  !--------------------------------------------------------------------
  subroutine log_stiffness_metrics(my_id, opz, tdt)
    use phys_module, only: index_now, t_now

    integer, intent(in) :: my_id
    real*8,  intent(in) :: opz, tdt

    Mat :: A_aij
    PetscReal :: nrm_Mpsi, nrm_D, nrm_Dp, nrm_Lrho, nrm_Wpara
    real*8    :: nrm_Mpsi_raw
    real*8    :: ratio_Mpsi, ratio_D, ratio_Dp, ratio_Lrho, ratio_Wpara, ratio_tot
    real*8    :: f_Mpsi, f_D, f_Dp, f_Lrho, f_Wpara
    integer :: u, ios
    PetscErrorCode :: ierr

    PetscCallA(MatConvert(g_mctx%M_psi, MATMPIAIJ, MAT_INITIAL_MATRIX, A_aij, ierr))
    PetscCallA(MatNorm(A_aij, NORM_FROBENIUS, nrm_Mpsi, ierr))
    PetscCallA(MatDestroy(A_aij, ierr))
    nrm_Mpsi_raw = max(nrm_Mpsi, tiny(1.d0))

    PetscCallA(MatConvert(g_mctx%D_op, MATMPIAIJ, MAT_INITIAL_MATRIX, A_aij, ierr))
    PetscCallA(MatNorm(A_aij, NORM_FROBENIUS, nrm_D, ierr))
    PetscCallA(MatDestroy(A_aij, ierr))

    PetscCallA(MatConvert(g_mctx%Dp_op, MATMPIAIJ, MAT_INITIAL_MATRIX, A_aij, ierr))
    PetscCallA(MatNorm(A_aij, NORM_FROBENIUS, nrm_Dp, ierr))
    PetscCallA(MatDestroy(A_aij, ierr))

    PetscCallA(MatConvert(g_mctx%L_rho, MATMPIAIJ, MAT_INITIAL_MATRIX, A_aij, ierr))
    PetscCallA(MatNorm(A_aij, NORM_FROBENIUS, nrm_Lrho, ierr))
    PetscCallA(MatDestroy(A_aij, ierr))

    PetscCallA(MatConvert(g_mctx%W_para, MATMPIAIJ, MAT_INITIAL_MATRIX, A_aij, ierr))
    PetscCallA(MatNorm(A_aij, NORM_FROBENIUS, nrm_Wpara, ierr))
    PetscCallA(MatDestroy(A_aij, ierr))

    ! Coefficient-scaled norms, as actually assembled into A_kideal / P_u.
    nrm_Mpsi  = opz               * nrm_Mpsi
    nrm_D     = tdt               * nrm_D
    nrm_Dp    = tdt               * nrm_Dp
    nrm_Lrho  = opz               * nrm_Lrho
    nrm_Wpara = g_mctx%dt_theta**2 * nrm_Wpara

    ! Mass-normalized stiffness ratios: divide by the *raw* (unscaled)
    ! M_psi norm so blocks of different differential order (M_psi 0th,
    ! D/Dp 1st-ish coupling, L_rho 2nd, W_para 4th-ish parabolized) become
    ! comparable, the way a generalized eigenvalue problem A x = lambda M x
    ! normalizes stiffness against mass. Raw Frobenius norms alone are
    ! biased toward higher-order operators purely by mesh-h scaling.
    ratio_Mpsi  = nrm_Mpsi  / nrm_Mpsi_raw
    ratio_D     = nrm_D     / nrm_Mpsi_raw
    ratio_Dp    = nrm_Dp    / nrm_Mpsi_raw
    ratio_Lrho  = nrm_Lrho  / nrm_Mpsi_raw
    ratio_Wpara = nrm_Wpara / nrm_Mpsi_raw

    ratio_tot = ratio_Mpsi + ratio_D + ratio_Dp + ratio_Lrho + ratio_Wpara
    if (ratio_tot > 0.d0) then
      f_Mpsi  = ratio_Mpsi  / ratio_tot
      f_D     = ratio_D     / ratio_tot
      f_Dp    = ratio_Dp    / ratio_tot
      f_Lrho  = ratio_Lrho  / ratio_tot
      f_Wpara = ratio_Wpara / ratio_tot
    else
      f_Mpsi = 0.d0; f_D = 0.d0; f_Dp = 0.d0; f_Lrho = 0.d0; f_Wpara = 0.d0
    endif

    if (my_id /= 0) return

    if (.not. g_stiffness_header_written) then
      open(newunit=u, file=STIFFNESS_LOG_FILE, status='replace', action='write', iostat=ios)
      if (ios == 0) then
        write(u,'(A)') '# step  time  tau  '// &
                        'nrm_Mpsi  nrm_D  nrm_Dp  nrm_Lrho  nrm_Wpara  '// &
                        'ratio_Mpsi  ratio_D  ratio_Dp  ratio_Lrho  ratio_Wpara  '// &
                        'frac_Mpsi  frac_D  frac_Dp  frac_Lrho  frac_Wpara'
        close(u)
      endif
      g_stiffness_header_written = .true.
    endif

    open(newunit=u, file=STIFFNESS_LOG_FILE, status='old', position='append', action='write', iostat=ios)
    if (ios == 0) then
      write(u,'(I8,1X,ES14.6,1X,ES14.6,5(1X,ES14.6),5(1X,ES14.6),5(1X,F10.6))') &
        index_now, t_now, g_mctx%dt_theta, &
        nrm_Mpsi, nrm_D, nrm_Dp, nrm_Lrho, nrm_Wpara, &
        ratio_Mpsi, ratio_D, ratio_Dp, ratio_Lrho, ratio_Wpara, &
        f_Mpsi, f_D, f_Dp, f_Lrho, f_Wpara
      close(u)
    endif
  end subroutine log_stiffness_metrics

#endif
end module mod_petsc_pc_metriplectic_assembly
