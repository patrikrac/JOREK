module mod_petsc_pc_metriplectic_analysis
!----------------------------------------------------------------
! Slice-A analysis checks for the metriplectic HSS PC
! (spec docs/superpowers/specs/2026-07-09-metriplectic-hss-pc-design.md,
!  Sec. 2.2; coefficients docs/notes/metriplectic_parabolization_note.tex).
!
!   T1  : transpose-pair / integration-by-parts consistency
!         (Dp_op skew-moved form vs Dp_struct by-parts form) +
!         exact-symmetry checks of M_psi, L_rho, W_para
!   T2  : SPD of P_u + extreme singular values across a tau sweep
!   T3  : segregated ideal-half solve vs direct MUMPS solve of the
!         4-var (psi,u,j,w) reference system with constraint rows kept
!   T3b : same apply vs the IDEAL-ONLY reference built from our own
!         operators (isolates the P_u-vs-discrete-Schur gap in norm)
!   T3c : Ritz values of P_u^-1 (L_rho + tau^2 Dp M^-1 D) — spectral
!         equivalence of the continuous parabolization with the exact
!         discrete Schur complement (MATSHELL, the decisive u-block gate)
!   T4  : harmonic n-n' coupling table of D_op (Remark 6 measurement)
!   T5a : Ritz values of the ideal-half-preconditioned 4-var system
!         (PCSHELL delegates to the production ideal-half apply)
!   T6  : sweep constraint exactness (both orders) + T6b mass-model gap
!   T5b : Ritz values of the FULL-SWEEP-preconditioned 4-var system (gate G2)
!   T5c : Ritz values of the production full-system apply (rho/T included)
!
! Reference system (constraints as rows, NEVER folded):
!   A4 = [ B11 B12 B13  0  ;
!          B21 B22 B23 B24 ;
!          B31  0  B33  0  ;
!           0  B42  0  B44 ]      (blocks extracted from the Jacobian)
!
! All checks print rank-0 verdict lines prefixed [Metriplectic].
!----------------------------------------------------------------
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_metriplectic_ctx, only: g_mctx
  use mod_petsc_pc_metriplectic_assembly, only: create_index_sets, extract_block, &
        setup_lu_ksp, setup_mass_ksp, metriplectic_build_sweep, metriplectic_refresh_Pu
  use mod_petsc_pc_metriplectic_apply, only: metriplectic_ideal_half_solve, &
        metriplectic_recover_jw, metriplectic_sweep_apply_4v, &
        metriplectic_sweep_apply_full, mpc_sweep_order
  implicit none
  private

  public :: petsc_metriplectic_run_analysis, petsc_metriplectic_track_energies

  ! --- Module state consumed by the PCSHELL apply callback ---
  Mat :: m_A4                          !< 4-var reference system (AIJ)
  Mat :: m_B31, m_B42                  !< constraint coupling blocks
  Mat :: m_B33, m_B44                  !< constraint diagonal blocks (T3b, T6)
  Mat :: m_B11, m_B22                  !< true diagonal blocks (T6b mass-model gap)
  KSP :: m_ksp_Mpsi                    !< consistent-mass solve on M_psi (T3c shell)
  KSP :: m_ksp_Pu                      !< direct solve on composed P_u (T2/T3c refs)
  Mat :: m_Pu_aij                      !< AIJ copy of the composed P_u
  logical :: m_pu_solver_ready = .false.
  real*8 :: m_tau  = 0.d0              !< tau used in the apply
  real*8 :: m_opz  = 1.d0              !< (1+zeta) Gears factor
  ! 1-var work vectors of the apply (all psi-sized = u-sized = j-sized)
  Vec :: m_rpsi, m_ru, m_rj, m_rw
  Vec :: m_h, m_dpsi, m_du, m_dj, m_dw, m_t1, m_t2

contains

  !====================================================================
  ! Entry point
  !====================================================================
  subroutine petsc_metriplectic_run_analysis(A_full, my_id)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w
    use phys_module,    only: time_evol_zeta, metriplectic_analysis_nsweep

    Mat,     intent(in) :: A_full
    integer, intent(in) :: my_id

    Mat :: B11, B12, B13, B21, B22, B23, B24, B31, B33, B42, B44
    Mat :: mats_nest(16), A_nest
    PetscErrorCode :: ierr
    integer :: comm

    if (.not. g_mctx%matrices_ready) then
      if (my_id == 0) write(*,'(A)') &
        "[Metriplectic] WARNING: analysis requested but operators not assembled — skipping"
      return
    endif

    call PetscObjectGetComm(A_full, comm, ierr)
    m_opz = 1.d0 + time_evol_zeta
    m_tau = g_mctx%dt_theta

    if (my_id == 0) then
      write(*,'(A)') "[Metriplectic] ================ Slice-A analysis ================"
      write(*,'(A,ES12.4,A,F8.4)') "[Metriplectic] tau = ", m_tau, ",  (1+zeta) = ", m_opz
    endif

    ! --- Variable index sets (shared with the sweep; persistent) ---
    if (.not. g_mctx%is_created) then
      call create_index_sets(A_full, comm)
    endif

    ! --- Build the production sweep (pair solves, ctx solvers, P_u at run tau);
    !     the delegated ideal-half apply and T6/T5b/T5c consume it ---
    call metriplectic_build_sweep(A_full, my_id)

    ! --- Extract the Jacobian blocks of the 4-var reference system ---
    call extract_block(A_full, var_psi, var_psi, B11)
    call extract_block(A_full, var_psi, var_u,   B12)
    call extract_block(A_full, var_psi, var_zj,  B13)
    call extract_block(A_full, var_u,   var_psi, B21)
    call extract_block(A_full, var_u,   var_u,   B22)
    call extract_block(A_full, var_u,   var_zj,  B23)
    call extract_block(A_full, var_u,   var_w,   B24)
    call extract_block(A_full, var_zj,  var_psi, B31)
    call extract_block(A_full, var_zj,  var_zj,  B33)
    call extract_block(A_full, var_w,   var_u,   B42)
    call extract_block(A_full, var_w,   var_w,   B44)
    m_B31 = B31
    m_B42 = B42
    m_B33 = B33
    m_B44 = B44
    m_B11 = B11
    m_B22 = B22

    ! --- 4-var nest -> AIJ (row-major block order) ---
    mats_nest       = PETSC_NULL_MAT
    mats_nest( 1) = B11;  mats_nest( 2) = B12;  mats_nest( 3) = B13
    mats_nest( 5) = B21;  mats_nest( 6) = B22;  mats_nest( 7) = B23;  mats_nest( 8) = B24
    mats_nest( 9) = B31;  mats_nest(11) = B33
    mats_nest(14) = B42;  mats_nest(16) = B44
    PetscCallA(MatCreateNest(comm, 4, PETSC_NULL_IS, 4, PETSC_NULL_IS, mats_nest, A_nest, ierr))
    PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, m_A4, ierr))
    PetscCallA(MatDestroy(A_nest, ierr))

    ! --- Work vectors and reference-solver setup ---
    call create_work_vecs(ierr)
    call setup_mass_ksp(g_mctx%M_psi, m_ksp_Mpsi, comm)

    ! --- Checks ---
    call run_T1(comm, my_id)
    call run_T2(comm, my_id, metriplectic_analysis_nsweep)

    call setup_pu_solver(m_tau, comm)   ! analysis-local P_u factor (T3c shell PC)
    call run_T3(comm, my_id)
    call run_T3b(comm, my_id)
    call run_T3c(comm, my_id)
    call run_T6(comm, my_id)
    call run_T4(comm, my_id)
    call run_T5a(comm, my_id)
    call run_T5b(comm, my_id)
    call run_T5c(comm, my_id, A_full)

    ! --- Cleanup (keep g_mctx operators, sweep objects, and index sets —
    !     the production apply owns them; destroy analysis-local objects) ---
    PetscCallA(KSPDestroy(m_ksp_Mpsi, ierr))
    if (m_pu_solver_ready) then
      PetscCallA(KSPDestroy(m_ksp_Pu, ierr))
      PetscCallA(MatDestroy(m_Pu_aij, ierr))
      m_pu_solver_ready = .false.
    endif
    call destroy_work_vecs(ierr)
    PetscCallA(MatDestroy(m_A4, ierr))
    PetscCallA(MatDestroy(B11, ierr)); PetscCallA(MatDestroy(B12, ierr))
    PetscCallA(MatDestroy(B13, ierr)); PetscCallA(MatDestroy(B21, ierr))
    PetscCallA(MatDestroy(B22, ierr)); PetscCallA(MatDestroy(B23, ierr))
    PetscCallA(MatDestroy(B24, ierr)); PetscCallA(MatDestroy(B31, ierr))
    PetscCallA(MatDestroy(B33, ierr)); PetscCallA(MatDestroy(B42, ierr))
    PetscCallA(MatDestroy(B44, ierr))

    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] ================ analysis complete ================"
  end subroutine petsc_metriplectic_run_analysis


  !====================================================================
  ! Setup helpers (create_index_sets / extract_block / setup_lu_ksp /
  ! setup_mass_ksp now live in the assembly module — shared with the sweep)
  !====================================================================

  !> (Re)compose P_u at tau_in, convert to AIJ, factor with MUMPS Cholesky
  !! (analysis-local copy bound to m_ksp_Pu; the T3c shell PC uses it).
  subroutine setup_pu_solver(tau_in, comm)
    use mod_petsc_pc_metriplectic_assembly, only: metriplectic_compose_P_u
    real*8,  intent(in) :: tau_in
    integer, intent(in) :: comm
    PetscErrorCode :: ierr

    if (m_pu_solver_ready) then
      PetscCallA(KSPDestroy(m_ksp_Pu, ierr))
      PetscCallA(MatDestroy(m_Pu_aij, ierr))
    endif
    call metriplectic_compose_P_u(tau_in)
    PetscCallA(MatConvert(g_mctx%P_u, MATMPIAIJ, MAT_INITIAL_MATRIX, m_Pu_aij, ierr))
    PetscCallA(MatSetOption(m_Pu_aij, MAT_SYMMETRIC, PETSC_TRUE, ierr))
    call setup_lu_ksp(m_Pu_aij, m_ksp_Pu, comm, symmetric=.true.)
    m_pu_solver_ready = .true.
    m_tau = tau_in
  end subroutine setup_pu_solver


  subroutine create_work_vecs(ierr)
    PetscErrorCode :: ierr
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_rpsi, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_ru,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_rj,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_rw,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_h,    ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_dpsi, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_du,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_dj,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_dw,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_t1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, m_t2,   ierr))
  end subroutine create_work_vecs

  subroutine destroy_work_vecs(ierr)
    PetscErrorCode :: ierr
    PetscCallA(VecDestroy(m_rpsi, ierr)); PetscCallA(VecDestroy(m_ru, ierr))
    PetscCallA(VecDestroy(m_rj,   ierr)); PetscCallA(VecDestroy(m_rw, ierr))
    PetscCallA(VecDestroy(m_h,    ierr)); PetscCallA(VecDestroy(m_dpsi, ierr))
    PetscCallA(VecDestroy(m_du,   ierr)); PetscCallA(VecDestroy(m_dj, ierr))
    PetscCallA(VecDestroy(m_dw,   ierr)); PetscCallA(VecDestroy(m_t1, ierr))
    PetscCallA(VecDestroy(m_t2,   ierr))
  end subroutine destroy_work_vecs


  !> Pack four 1-var vecs into a 4v packed vec (rank-local concatenation,
  !! matching the MatNest->AIJ layout; pattern: physics apply pack_2v).
  subroutine pack_4v(x1, x2, x3, x4, y, ierr)
    Vec :: x1, x2, x3, x4, y
    PetscErrorCode :: ierr
    PetscScalar, pointer :: a1(:), a2(:), a3(:), a4(:), ay(:)
    PetscInt :: n1, n2, n3, n4

    call VecGetLocalSize(x1, n1, ierr); call VecGetLocalSize(x2, n2, ierr)
    call VecGetLocalSize(x3, n3, ierr); call VecGetLocalSize(x4, n4, ierr)
    call VecGetArrayReadF90(x1, a1, ierr); call VecGetArrayReadF90(x2, a2, ierr)
    call VecGetArrayReadF90(x3, a3, ierr); call VecGetArrayReadF90(x4, a4, ierr)
    call VecGetArrayF90(y, ay, ierr)
    ay(1:n1)                   = a1(1:n1)
    ay(n1+1:n1+n2)             = a2(1:n2)
    ay(n1+n2+1:n1+n2+n3)       = a3(1:n3)
    ay(n1+n2+n3+1:n1+n2+n3+n4) = a4(1:n4)
    call VecRestoreArrayReadF90(x1, a1, ierr); call VecRestoreArrayReadF90(x2, a2, ierr)
    call VecRestoreArrayReadF90(x3, a3, ierr); call VecRestoreArrayReadF90(x4, a4, ierr)
    call VecRestoreArrayF90(y, ay, ierr)
  end subroutine pack_4v

  subroutine unpack_4v(x, y1, y2, y3, y4, ierr)
    Vec :: x, y1, y2, y3, y4
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ax(:), a1(:), a2(:), a3(:), a4(:)
    PetscInt :: n1, n2, n3, n4
    call VecGetLocalSize(y1, n1, ierr); call VecGetLocalSize(y2, n2, ierr)
    call VecGetLocalSize(y3, n3, ierr); call VecGetLocalSize(y4, n4, ierr)
    call VecGetArrayReadF90(x, ax, ierr)
    call VecGetArrayF90(y1, a1, ierr); call VecGetArrayF90(y2, a2, ierr)
    call VecGetArrayF90(y3, a3, ierr); call VecGetArrayF90(y4, a4, ierr)
    a1(1:n1) = ax(1:n1)
    a2(1:n2) = ax(n1+1:n1+n2)
    a3(1:n3) = ax(n1+n2+1:n1+n2+n3)
    a4(1:n4) = ax(n1+n2+n3+1:n1+n2+n3+n4)
    call VecRestoreArrayReadF90(x, ax, ierr)
    call VecRestoreArrayF90(y1, a1, ierr); call VecRestoreArrayF90(y2, a2, ierr)
    call VecRestoreArrayF90(y3, a3, ierr); call VecRestoreArrayF90(y4, a4, ierr)
  end subroutine unpack_4v


  !====================================================================
  ! Diagnostic PCSHELL apply: DELEGATES to the production ideal-half
  ! routines (mod_petsc_pc_metriplectic_apply; ctx-owned solvers, P_u
  ! factored by build_sweep/refresh_Pu). Keeping this thin wrapper makes
  ! T3/T3b/T5a regression tests of the production code path.
  !====================================================================
  subroutine metriplectic_halfpc_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    call unpack_4v(x, m_rpsi, m_ru, m_rj, m_rw, ierr)
    call metriplectic_ideal_half_solve(m_rpsi, m_ru, m_dpsi, m_du)
    call metriplectic_recover_jw(m_rj, m_rw, m_dpsi, m_du, m_dj, m_dw)
    call pack_4v(m_dpsi, m_du, m_dj, m_dw, y, ierr)
    ierr = 0
  end subroutine metriplectic_halfpc_apply


  !====================================================================
  ! T1: transpose-pair / IBP consistency + symmetry
  !====================================================================
  subroutine run_T1(comm, my_id)
    integer, intent(in) :: comm, my_id

    Mat :: Dp_aij, Dps_aij, diffM
    PetscReal :: nrm_diff, nrm_ref
    PetscErrorCode :: ierr

    PetscCallA(MatConvert(g_mctx%Dp_op,     MATMPIAIJ, MAT_INITIAL_MATRIX, Dp_aij,  ierr))
    PetscCallA(MatConvert(g_mctx%Dp_struct, MATMPIAIJ, MAT_INITIAL_MATRIX, Dps_aij, ierr))

    PetscCallA(MatDuplicate(Dp_aij, MAT_COPY_VALUES, diffM, ierr))
    PetscCallA(MatAXPY(diffM, -1.d0, Dps_aij, DIFFERENT_NONZERO_PATTERN, ierr))
    PetscCallA(MatNorm(diffM, NORM_FROBENIUS, nrm_diff, ierr))
    PetscCallA(MatNorm(Dp_aij, NORM_FROBENIUS, nrm_ref, ierr))
    if (my_id == 0) write(*,'(A,ES12.4)') &
      "[Metriplectic] T1 IBP consistency |Dp - Dp_struct|/|Dp| = ", &
      nrm_diff / max(nrm_ref, tiny(1.d0))
    PetscCallA(MatDestroy(diffM,   ierr))
    PetscCallA(MatDestroy(Dp_aij,  ierr))
    PetscCallA(MatDestroy(Dps_aij, ierr))

    call check_symmetry(g_mctx%W_para, "W_para", my_id)
    call check_symmetry(g_mctx%M_psi,  "M_psi ", my_id)
    call check_symmetry(g_mctx%L_rho,  "L_rho ", my_id)
  end subroutine run_T1


  subroutine check_symmetry(A, label, my_id)
    Mat,              intent(in) :: A
    character(len=*), intent(in) :: label
    integer,          intent(in) :: my_id

    Mat :: A_aij, At
    PetscReal :: nrm_diff, nrm_ref
    PetscErrorCode :: ierr

    PetscCallA(MatConvert(A, MATMPIAIJ, MAT_INITIAL_MATRIX, A_aij, ierr))
    PetscCallA(MatTranspose(A_aij, MAT_INITIAL_MATRIX, At, ierr))
    PetscCallA(MatAXPY(At, -1.d0, A_aij, DIFFERENT_NONZERO_PATTERN, ierr))
    PetscCallA(MatNorm(At, NORM_FROBENIUS, nrm_diff, ierr))
    PetscCallA(MatNorm(A_aij, NORM_FROBENIUS, nrm_ref, ierr))
    if (my_id == 0) write(*,'(A,A,A,ES12.4)') &
      "[Metriplectic] T1 symmetry |A - A^T|/|A| (", label, ") = ", &
      nrm_diff / max(nrm_ref, tiny(1.d0))
    PetscCallA(MatDestroy(At,    ierr))
    PetscCallA(MatDestroy(A_aij, ierr))
  end subroutine check_symmetry


  !====================================================================
  ! T2: SPD + conditioning sweep of P_u(tau)
  !====================================================================
  subroutine run_T2(comm, my_id, nsweep)
    use mod_petsc_pc_metriplectic_assembly, only: metriplectic_compose_P_u
    integer, intent(in) :: comm, my_id, nsweep

    Mat :: P_aij
    KSP :: ksp_chol, ksp_sv
    PC  :: pc
    Vec :: b, x
    PetscRandom :: rnd
    KSPConvergedReason :: reason
    PetscReal :: smax, smin
    PetscErrorCode :: ierr
    integer :: is
    real*8  :: tau_s
    character(len=8) :: spd_verdict

    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] T2:      tau           SPD       sigma_min     sigma_max     cond"

    do is = 1, nsweep
      tau_s = m_tau * 4.d0**(is - (nsweep+1)/2)
      call metriplectic_compose_P_u(tau_s)
      PetscCallA(MatConvert(g_mctx%P_u, MATMPIAIJ, MAT_INITIAL_MATRIX, P_aij, ierr))
      PetscCallA(MatCreateVecs(P_aij, x, b, ierr))
      PetscCallA(PetscRandomCreate(comm, rnd, ierr))
      PetscCallA(VecSetRandom(b, rnd, ierr))

      ! --- SPD probe: MUMPS Cholesky (LDL^T); an indefinite pivot surfaces
      !     as KSP_DIVERGED_PC_FAILED at solve time (plain calls, no abort) ---
      PetscCallA(MatSetOption(P_aij, MAT_SPD, PETSC_TRUE, ierr))
      call KSPCreate(comm, ksp_chol, ierr)
      call KSPSetOperators(ksp_chol, P_aij, P_aij, ierr)
      call KSPSetType(ksp_chol, KSPPREONLY, ierr)
      call KSPGetPC(ksp_chol, pc, ierr)
      call PCSetType(pc, PCCHOLESKY, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
      call KSPSolve(ksp_chol, b, x, ierr)
      spd_verdict = "FAIL"
      if (ierr == 0) then
        call KSPGetConvergedReason(ksp_chol, reason, ierr)
        if (ierr == 0) then
          if (reason > 0) spd_verdict = "PASS"
        endif
      endif
      call KSPDestroy(ksp_chol, ierr)

      ! --- Extreme singular values via GMRES Krylov estimates ---
      PetscCallA(KSPCreate(comm, ksp_sv, ierr))
      PetscCallA(KSPSetOperators(ksp_sv, P_aij, P_aij, ierr))
      PetscCallA(KSPSetType(ksp_sv, KSPGMRES, ierr))
      PetscCallA(KSPGMRESSetRestart(ksp_sv, 60, ierr))
      PetscCallA(KSPSetTolerances(ksp_sv, 1.d-30, 1.d-50, PETSC_CURRENT_REAL, 50, ierr))
      PetscCallA(KSPSetComputeSingularValues(ksp_sv, PETSC_TRUE, ierr))
      PetscCallA(KSPGetPC(ksp_sv, pc, ierr))
      PetscCallA(PCSetType(pc, PCNONE, ierr))
      call KSPSolve(ksp_sv, b, x, ierr)
      PetscCallA(KSPComputeExtremeSingularValues(ksp_sv, smax, smin, ierr))
      PetscCallA(KSPDestroy(ksp_sv, ierr))

      if (my_id == 0) write(*,'(A,ES12.4,A,A,2X,3ES14.4)') &
        "[Metriplectic] T2: ", tau_s, "  ", spd_verdict, smin, smax, &
        smax / max(smin, tiny(1.d0))

      PetscCallA(PetscRandomDestroy(rnd, ierr))
      PetscCallA(VecDestroy(b, ierr))
      PetscCallA(VecDestroy(x, ierr))
      PetscCallA(MatDestroy(P_aij, ierr))
    enddo
  end subroutine run_T2


  !====================================================================
  ! T3: segregated ideal-half vs direct 4-var reference solve
  !====================================================================
  subroutine run_T3(comm, my_id)
    integer, intent(in) :: comm, my_id

    KSP :: ksp_ref
    Vec :: b4, y4, z4
    Vec :: yp, yu, yj, yw
    PetscRandom :: rnd
    PetscReal :: e_psi, e_u, e_j, e_w, n_psi, n_u, n_j, n_w
    PetscErrorCode :: ierr
    integer :: trial
    PC :: dummy_pc

    call setup_lu_ksp(m_A4, ksp_ref, comm, symmetric=.false.)
    PetscCallA(MatCreateVecs(m_A4, y4, b4, ierr))
    PetscCallA(VecDuplicate(y4, z4, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yp, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yu, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yj, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yw, ierr))
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))

    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] T3:  trial   err_psi       err_u         err_j         err_w"

    do trial = 1, 3
      PetscCallA(VecSetRandom(b4, rnd, ierr))

      ! Reference: direct MUMPS on the 4-var constrained system
      PetscCallA(KSPSolve(ksp_ref, b4, y4, ierr))

      ! Segregated ideal-half apply (same routine T5a uses as PCSHELL)
      call metriplectic_halfpc_apply(dummy_pc, b4, z4, ierr)

      ! Component-wise relative errors
      call unpack_4v(y4, yp, yu, yj, yw, ierr)
      PetscCallA(VecNorm(yp, NORM_2, n_psi, ierr))
      PetscCallA(VecNorm(yu, NORM_2, n_u,   ierr))
      PetscCallA(VecNorm(yj, NORM_2, n_j,   ierr))
      PetscCallA(VecNorm(yw, NORM_2, n_w,   ierr))

      PetscCallA(VecAXPY(z4, -1.d0, y4, ierr))
      call unpack_4v(z4, yp, yu, yj, yw, ierr)
      PetscCallA(VecNorm(yp, NORM_2, e_psi, ierr))
      PetscCallA(VecNorm(yu, NORM_2, e_u,   ierr))
      PetscCallA(VecNorm(yj, NORM_2, e_j,   ierr))
      PetscCallA(VecNorm(yw, NORM_2, e_w,   ierr))

      if (my_id == 0) write(*,'(A,I5,2X,4ES14.4)') "[Metriplectic] T3: ", trial, &
        e_psi/max(n_psi,tiny(1.d0)), e_u/max(n_u,tiny(1.d0)), &
        e_j/max(n_j,tiny(1.d0)),     e_w/max(n_w,tiny(1.d0))
    enddo

    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(KSPDestroy(ksp_ref, ierr))
    PetscCallA(VecDestroy(b4, ierr)); PetscCallA(VecDestroy(y4, ierr))
    PetscCallA(VecDestroy(z4, ierr))
    PetscCallA(VecDestroy(yp, ierr)); PetscCallA(VecDestroy(yu, ierr))
    PetscCallA(VecDestroy(yj, ierr)); PetscCallA(VecDestroy(yw, ierr))
  end subroutine run_T3


  !====================================================================
  ! T3b: segregated apply vs direct solve of the IDEAL-ONLY reference
  !   A_id = [ (1+z)M_psi   tdt*D      0    0  ;
  !            tdt*Dp      -(1+z)L_rho 0    0  ;
  !            B31          0          B33  0  ;
  !            0            B42        0    B44 ]   (tdt = tau*(1+z))
  ! assembled from OUR operators in JOREK sign convention. Against this,
  ! the segregated apply is an exact block elimination except P_u vs the
  ! discrete Schur (L_rho + tau^2 Dp M^-1 D): T3b isolates precisely the
  ! continuous-vs-discrete Schur consistency gap (spec Sec. 2.2, T3).
  !====================================================================
  subroutine run_T3b(comm, my_id)
    integer, intent(in) :: comm, my_id

    Mat :: Mpsi_s, D_s, Dp_s, Lrho_s
    Mat :: mats_nest(16), A_nest, A_id
    KSP :: ksp_ref
    Vec :: b4, y4, z4
    Vec :: yp, yu, yj, yw
    PetscRandom :: rnd
    PetscReal :: e_psi, e_u, e_j, e_w, n_psi, n_u, n_j, n_w
    PetscErrorCode :: ierr
    integer :: trial
    real*8 :: tdt
    PC :: dummy_pc

    tdt = m_tau * m_opz

    ! Scaled AIJ copies of the element-assembled operators (JOREK convention)
    PetscCallA(MatConvert(g_mctx%M_psi, MATMPIAIJ, MAT_INITIAL_MATRIX, Mpsi_s, ierr))
    PetscCallA(MatScale(Mpsi_s, m_opz, ierr))
    PetscCallA(MatConvert(g_mctx%D_op, MATMPIAIJ, MAT_INITIAL_MATRIX, D_s, ierr))
    PetscCallA(MatScale(D_s, tdt, ierr))
    PetscCallA(MatConvert(g_mctx%Dp_op, MATMPIAIJ, MAT_INITIAL_MATRIX, Dp_s, ierr))
    PetscCallA(MatScale(Dp_s, tdt, ierr))
    PetscCallA(MatConvert(g_mctx%L_rho, MATMPIAIJ, MAT_INITIAL_MATRIX, Lrho_s, ierr))
    PetscCallA(MatScale(Lrho_s, -m_opz, ierr))

    mats_nest     = PETSC_NULL_MAT
    mats_nest( 1) = Mpsi_s;  mats_nest( 2) = D_s
    mats_nest( 5) = Dp_s;    mats_nest( 6) = Lrho_s
    mats_nest( 9) = m_B31;   mats_nest(11) = m_B33
    mats_nest(14) = m_B42;   mats_nest(16) = m_B44
    PetscCallA(MatCreateNest(comm, 4, PETSC_NULL_IS, 4, PETSC_NULL_IS, mats_nest, A_nest, ierr))
    PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, A_id, ierr))
    PetscCallA(MatDestroy(A_nest, ierr))

    call setup_lu_ksp(A_id, ksp_ref, comm, symmetric=.false.)
    PetscCallA(MatCreateVecs(A_id, y4, b4, ierr))
    PetscCallA(VecDuplicate(y4, z4, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yp, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yu, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yj, ierr))
    PetscCallA(VecDuplicate(m_rpsi, yw, ierr))
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))

    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] T3b: ideal-only reference (isolates Schur consistency gap)"
    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] T3b: trial   err_psi       err_u         err_j         err_w"

    do trial = 1, 3
      PetscCallA(VecSetRandom(b4, rnd, ierr))
      PetscCallA(KSPSolve(ksp_ref, b4, y4, ierr))
      call metriplectic_halfpc_apply(dummy_pc, b4, z4, ierr)

      call unpack_4v(y4, yp, yu, yj, yw, ierr)
      PetscCallA(VecNorm(yp, NORM_2, n_psi, ierr))
      PetscCallA(VecNorm(yu, NORM_2, n_u,   ierr))
      PetscCallA(VecNorm(yj, NORM_2, n_j,   ierr))
      PetscCallA(VecNorm(yw, NORM_2, n_w,   ierr))

      PetscCallA(VecAXPY(z4, -1.d0, y4, ierr))
      call unpack_4v(z4, yp, yu, yj, yw, ierr)
      PetscCallA(VecNorm(yp, NORM_2, e_psi, ierr))
      PetscCallA(VecNorm(yu, NORM_2, e_u,   ierr))
      PetscCallA(VecNorm(yj, NORM_2, e_j,   ierr))
      PetscCallA(VecNorm(yw, NORM_2, e_w,   ierr))

      if (my_id == 0) write(*,'(A,I5,2X,4ES14.4)') "[Metriplectic] T3b:", trial, &
        e_psi/max(n_psi,tiny(1.d0)), e_u/max(n_u,tiny(1.d0)), &
        e_j/max(n_j,tiny(1.d0)),     e_w/max(n_w,tiny(1.d0))
    enddo

    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(KSPDestroy(ksp_ref, ierr))
    PetscCallA(VecDestroy(b4, ierr)); PetscCallA(VecDestroy(y4, ierr))
    PetscCallA(VecDestroy(z4, ierr))
    PetscCallA(VecDestroy(yp, ierr)); PetscCallA(VecDestroy(yu, ierr))
    PetscCallA(VecDestroy(yj, ierr)); PetscCallA(VecDestroy(yw, ierr))
    PetscCallA(MatDestroy(A_id,   ierr))
    PetscCallA(MatDestroy(Mpsi_s, ierr)); PetscCallA(MatDestroy(D_s,    ierr))
    PetscCallA(MatDestroy(Dp_s,   ierr)); PetscCallA(MatDestroy(Lrho_s, ierr))
  end subroutine run_T3b


  !====================================================================
  ! T3c: spectral equivalence of P_u with the exact discrete Schur
  !   S = L_rho + tau^2 * Dp M_psi^-1 D          (MATSHELL, matrix-free)
  ! Ritz values of P_u^-1 S from preconditioned GMRES. This is the
  ! decisive u-block gate: T3b measures the NORM gap between W_para and
  ! Dp M^-1 D, which is O(1) at grid scale by construction (the mass
  ! inverse inserts an L2 projection between the factors, and random
  ! vectors are grid-scale dominated). What the Krylov method needs is
  ! a bounded spectrum of P_u^-1 S — measured here in isolation from
  ! the kink/dissipative content that pollutes T5a.
  ! Requires m_ksp_Pu factored at m_tau (call after setup_pu_solver).
  !====================================================================
  subroutine run_T3c(comm, my_id)
    integer, intent(in) :: comm, my_id

    Mat :: S_shell
    KSP :: ksp
    PC  :: pc
    Vec :: b, x
    PetscRandom :: rnd
    PetscReal :: r_eig(60), c_eig(60)
    PetscInt  :: neig, its, n_loc, n_glob
    KSPConvergedReason :: reason
    PetscErrorCode :: ierr
    integer :: i, jmin
    real*8  :: tmp, re_min, re_max, im_max
    character(len=16) :: converged_txt

    call MatGetLocalSize(m_Pu_aij, n_loc, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(m_Pu_aij, n_glob, PETSC_NULL_INTEGER, ierr)
    call MatCreateShell(comm, n_loc, n_loc, n_glob, n_glob, &
                        PETSC_NULL_INTEGER, S_shell, ierr)
    call MatShellSetOperation(S_shell, MATOP_MULT, schur_discrete_mult, ierr)

    PetscCallA(MatCreateVecs(m_Pu_aij, x, b, ierr))
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))
    PetscCallA(VecSetRandom(b, rnd, ierr))

    PetscCallA(KSPCreate(comm, ksp, ierr))
    PetscCallA(KSPSetOperators(ksp, S_shell, S_shell, ierr))
    PetscCallA(KSPSetType(ksp, KSPGMRES, ierr))
    PetscCallA(KSPGMRESSetRestart(ksp, 60, ierr))
    PetscCallA(KSPSetTolerances(ksp, 1.d-10, 1.d-50, PETSC_CURRENT_REAL, 60, ierr))
    PetscCallA(KSPSetComputeEigenvalues(ksp, PETSC_TRUE, ierr))
    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(pc, pu_pc_apply, ierr))

    call KSPSolve(ksp, b, x, ierr)
    PetscCallA(KSPGetIterationNumber(ksp, its, ierr))
    PetscCallA(KSPGetConvergedReason(ksp, reason, ierr))
    converged_txt = "not converged"
    if (reason > 0) converged_txt = "converged"
    PetscCallA(KSPComputeEigenvalues(ksp, 60, r_eig, c_eig, neig, ierr))

    do i = 1, neig-1
      jmin = i + minloc(r_eig(i:neig), 1) - 1
      if (jmin /= i) then
        tmp = r_eig(i); r_eig(i) = r_eig(jmin); r_eig(jmin) = tmp
        tmp = c_eig(i); c_eig(i) = c_eig(jmin); c_eig(jmin) = tmp
      endif
    enddo
    re_min = r_eig(1); re_max = r_eig(1); im_max = 0.d0
    do i = 1, neig
      re_max = max(re_max, r_eig(i))
      im_max = max(im_max, abs(c_eig(i)))
    enddo

    if (my_id == 0) then
      write(*,'(A,ES11.4)') "[Metriplectic] T3c: spec(P_u^-1 S_discrete) at tau =", m_tau
      write(*,'(A,I5,A,A)') "[Metriplectic] T3c: GMRES iterations = ", its, &
                            ", ", trim(converged_txt)
      write(*,'(A,3ES13.4)') "[Metriplectic] T3c: min Re, max Re, max |Im| = ", &
                             re_min, re_max, im_max
      write(*,'(A,I4,A)')   "[Metriplectic] T3c: ", neig, " Ritz values (Re, Im):"
      do i = 1, neig
        write(*,'(A,2ES14.5)') "[Metriplectic] T3c:   ", r_eig(i), c_eig(i)
      enddo
    endif

    PetscCallA(KSPDestroy(ksp, ierr))
    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(VecDestroy(b, ierr))
    PetscCallA(VecDestroy(x, ierr))
    PetscCallA(MatDestroy(S_shell, ierr))
  end subroutine run_T3c


  !> MATSHELL MULT: y = L_rho x + tau^2 * Dp M_psi^-1 D x  (discrete Schur).
  !! BC rows: L_rho carries unit diag, D/Dp rows are zero -> identity rows,
  !! matching P_u exactly (W_para BC diag is 0), so BC dofs contribute a
  !! trivial eigenvalue-1 cluster to T3c.
  subroutine schur_discrete_mult(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr

    call MatMult(g_mctx%D_op, x, m_t1, ierr)
    call KSPSolve(m_ksp_Mpsi, m_t1, m_t2, ierr)
    call MatMult(g_mctx%Dp_op, m_t2, y, ierr)
    call MatMult(g_mctx%L_rho, x, m_t1, ierr)
    call VecAXPBY(y, 1.d0, m_tau**2, m_t1, ierr)
    ierr = 0
  end subroutine schur_discrete_mult


  !> PCSHELL apply for T3c: y = P_u^-1 x via the factored MUMPS solve.
  subroutine pu_pc_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    call KSPSolve(m_ksp_Pu, x, y, ierr)
    ierr = 0
  end subroutine pu_pc_apply


  !====================================================================
  ! T4: harmonic n-n' coupling table of D_op (Remark 6 measurement)
  !====================================================================
  subroutine run_T4(comm, my_id)
    use mod_parameters, only: n_tor
    use phys_module,    only: mode
    integer, intent(in) :: comm, my_id

    Mat :: D_aij, Dsub
    IS  :: is_h(n_tor)
    PetscInt :: n_local, rstart, rend, n_block_local, n_h
    PetscInt, allocatable :: idx(:)
    PetscReal :: nrm(n_tor, n_tor), dmax
    PetscErrorCode :: ierr
    integer :: h, hp, i, k
    character(len=512) :: line

    PetscCallA(MatConvert(g_mctx%D_op, MATMPIAIJ, MAT_INITIAL_MATRIX, D_aij, ierr))
    PetscCallA(MatGetLocalSize(D_aij, n_local, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetOwnershipRange(D_aij, rstart, rend, ierr))
    n_block_local = n_local / n_tor
    n_h = n_block_local

    allocate(idx(n_h))
    do h = 1, n_tor
      k = 0
      do i = 0, n_block_local - 1
        k = k + 1
        idx(k) = rstart + i * n_tor + (h-1)
      enddo
      PetscCallA(ISCreateGeneral(comm, n_h, idx, PETSC_COPY_VALUES, is_h(h), ierr))
    enddo
    deallocate(idx)

    dmax = 0.d0
    do h = 1, n_tor
      do hp = 1, n_tor
        PetscCallA(MatCreateSubMatrix(D_aij, is_h(h), is_h(hp), MAT_INITIAL_MATRIX, Dsub, ierr))
        PetscCallA(MatNorm(Dsub, NORM_FROBENIUS, nrm(h,hp), ierr))
        PetscCallA(MatDestroy(Dsub, ierr))
        if (h == hp .and. nrm(h,hp) > dmax) dmax = nrm(h,hp)
      enddo
    enddo

    if (my_id == 0) then
      write(*,'(A)') "[Metriplectic] T4: harmonic coupling |D_(h,h')|_F / max_h |D_(h,h)|_F"
      write(line,'(A)') "[Metriplectic] T4:  h\h' "
      do hp = 1, n_tor
        write(line(len_trim(line)+1:),'(I12)') mode(hp)
      enddo
      write(*,'(A)') trim(line)
      do h = 1, n_tor
        write(line,'(A,I4,A)') "[Metriplectic] T4: ", mode(h), " "
        do hp = 1, n_tor
          write(line(len_trim(line)+1:),'(ES12.3)') nrm(h,hp) / max(dmax, tiny(1.d0))
        enddo
        write(*,'(A)') trim(line)
      enddo
    endif

    do h = 1, n_tor
      PetscCallA(ISDestroy(is_h(h), ierr))
    enddo
    PetscCallA(MatDestroy(D_aij, ierr))
  end subroutine run_T4


  !====================================================================
  ! T5a: Ritz values of the ideal-half-preconditioned 4-var system.
  ! The PRODUCTION P_u factorization (ctx ksp_Pu) is refreshed at
  ! tau/4, tau, 4*tau while the system and the apply's tau scalars stay
  ! at the run tau — the factorization-lag model of a lagged PC. (Slice A
  ! lagged the tau scalars too; the matched-tau row is identical, the
  ! off-tau rows changed meaning on 2026-07-16 — see the gate record.)
  !====================================================================
  subroutine run_T5a(comm, my_id)
    integer, intent(in) :: comm, my_id

    KSP :: ksp
    PC  :: pc
    Vec :: b4, x4
    PetscRandom :: rnd
    PetscReal :: r_eig(60), c_eig(60)
    PetscInt  :: neig, its
    KSPConvergedReason :: reason
    PetscErrorCode :: ierr
    integer :: it3, i, jmin
    real*8  :: tau_run, tau_s, tmp
    character(len=64) :: head
    character(len=16) :: converged_txt

    tau_run = m_tau

    do it3 = 1, 3
      tau_s = tau_run * 4.d0**(it3 - 2)
      call metriplectic_refresh_Pu(tau_s)

      PetscCallA(MatCreateVecs(m_A4, x4, b4, ierr))
      PetscCallA(PetscRandomCreate(comm, rnd, ierr))
      PetscCallA(VecSetRandom(b4, rnd, ierr))

      PetscCallA(KSPCreate(comm, ksp, ierr))
      PetscCallA(KSPSetOperators(ksp, m_A4, m_A4, ierr))
      PetscCallA(KSPSetType(ksp, KSPGMRES, ierr))
      PetscCallA(KSPGMRESSetRestart(ksp, 60, ierr))
      PetscCallA(KSPSetTolerances(ksp, 1.d-10, 1.d-50, PETSC_CURRENT_REAL, 60, ierr))
      PetscCallA(KSPSetComputeEigenvalues(ksp, PETSC_TRUE, ierr))
      PetscCallA(KSPGetPC(ksp, pc, ierr))
      PetscCallA(PCSetType(pc, PCSHELL, ierr))
      PetscCallA(PCShellSetApply(pc, metriplectic_halfpc_apply, ierr))

      call KSPSolve(ksp, b4, x4, ierr)
      PetscCallA(KSPGetIterationNumber(ksp, its, ierr))
      PetscCallA(KSPGetConvergedReason(ksp, reason, ierr))
      converged_txt = "not converged"
      if (reason > 0) converged_txt = "converged"
      PetscCallA(KSPComputeEigenvalues(ksp, 60, r_eig, c_eig, neig, ierr))

      ! Sort Ritz values by real part (selection sort; neig <= 60)
      do i = 1, neig-1
        jmin = i + minloc(r_eig(i:neig), 1) - 1
        if (jmin /= i) then
          tmp = r_eig(i); r_eig(i) = r_eig(jmin); r_eig(jmin) = tmp
          tmp = c_eig(i); c_eig(i) = c_eig(jmin); c_eig(jmin) = tmp
        endif
      enddo

      if (my_id == 0) then
        write(head,'(A,ES11.4,A,ES11.4)') " P_u at tau_s=", tau_s, ", system tau=", tau_run
        write(*,'(A,A)')        "[Metriplectic] T5a:", trim(head)
        write(*,'(A,I5,A,A)')   "[Metriplectic] T5a: GMRES iterations = ", its, &
                                ", ", trim(converged_txt)
        write(*,'(A,I4,A)')     "[Metriplectic] T5a: ", neig, " Ritz values (Re, Im):"
        do i = 1, neig
          write(*,'(A,2ES14.5)') "[Metriplectic] T5a:   ", r_eig(i), c_eig(i)
        enddo
      endif

      PetscCallA(KSPDestroy(ksp, ierr))
      PetscCallA(PetscRandomDestroy(rnd, ierr))
      PetscCallA(VecDestroy(b4, ierr))
      PetscCallA(VecDestroy(x4, ierr))
    enddo

    ! Restore the production P_u at the run's tau
    call metriplectic_refresh_Pu(tau_run)
  end subroutine run_T5a


  !====================================================================
  ! T6: sweep bookkeeping checks (gate G2, part 1).
  !  (a) Constraint exactness, both orders: the sweep output must satisfy
  !      B31 dpsi + B33 dj = r_j and B42 du + B44 dw = r_w to roundoff
  !      (note Sec. sweep, "Constraint exactness of the PC") — this pins
  !      the single-consumption rule and the pack/unpack plumbing.
  !  (b) T6b, mass-model gap: |(1+z)M_psi z - B11 z|/|B11 z| and
  !      |-(1+z)L_rho z - B22 z|/|B22 z| — the dissipative + model content
  !      of the true diagonal blocks beyond the middle mass M_red
  !      (bounded model choice; recorded, not gated).
  !====================================================================
  subroutine run_T6(comm, my_id)
    use phys_module, only: time_evol_zeta
    integer, intent(in) :: comm, my_id

    Vec :: x4, y4
    PetscRandom :: rnd
    PetscReal :: cj, cw, nj, nw, gpsi, gu, npsi_r, nu_r
    PetscErrorCode :: ierr
    integer :: io
    character(len=2) :: orders(2)
    PC :: dummy_pc
    real*8 :: opz

    opz = 1.d0 + time_evol_zeta
    orders(1) = 'SK'; orders(2) = 'KS'
    PetscCallA(MatCreateVecs(m_A4, y4, x4, ierr))
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))

    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] T6: sweep constraint exactness (relative residual of the constraint rows)"
    do io = 1, 2
      mpc_sweep_order = orders(io)
      PetscCallA(VecSetRandom(x4, rnd, ierr))
      call metriplectic_sweep_apply_4v(dummy_pc, x4, y4, ierr)

      call unpack_4v(x4, m_rpsi, m_ru, m_rj, m_rw, ierr)
      call unpack_4v(y4, m_dpsi, m_du, m_dj, m_dw, ierr)

      ! |B31 dpsi + B33 dj - rj| / |rj|
      call MatMult(m_B31, m_dpsi, m_t1, ierr)
      call MatMultAdd(m_B33, m_dj, m_t1, m_t1, ierr)
      call VecAXPY(m_t1, -1.d0, m_rj, ierr)
      PetscCallA(VecNorm(m_t1, NORM_2, cj, ierr))
      PetscCallA(VecNorm(m_rj, NORM_2, nj, ierr))

      ! |B42 du + B44 dw - rw| / |rw|
      call MatMult(m_B42, m_du, m_t1, ierr)
      call MatMultAdd(m_B44, m_dw, m_t1, m_t1, ierr)
      call VecAXPY(m_t1, -1.d0, m_rw, ierr)
      PetscCallA(VecNorm(m_t1, NORM_2, cw, ierr))
      PetscCallA(VecNorm(m_rw, NORM_2, nw, ierr))

      if (my_id == 0) write(*,'(A,A,A,2ES14.4)') &
        "[Metriplectic] T6:  order ", orders(io), "  (j, w) = ", &
        cj/max(nj,tiny(1.d0)), cw/max(nw,tiny(1.d0))
    enddo
    mpc_sweep_order = 'SK'

    ! --- T6b: mass-model content of the true diagonal blocks ---
    PetscCallA(VecSetRandom(m_h, rnd, ierr))
    call MatMult(m_B11, m_h, m_t1, ierr)
    call MatMult(g_mctx%M_psi, m_h, m_t2, ierr)
    call VecAXPBY(m_t2, -1.d0, opz, m_t1, ierr)      ! t2 = opz*Mpsi z - B11 z
    PetscCallA(VecNorm(m_t2, NORM_2, gpsi, ierr))
    PetscCallA(VecNorm(m_t1, NORM_2, npsi_r, ierr))

    call MatMult(m_B22, m_h, m_t1, ierr)
    call MatMult(g_mctx%L_rho, m_h, m_t2, ierr)
    call VecAXPBY(m_t2, -1.d0, -opz, m_t1, ierr)     ! t2 = -opz*Lrho z - B22 z
    PetscCallA(VecNorm(m_t2, NORM_2, gu, ierr))
    PetscCallA(VecNorm(m_t1, NORM_2, nu_r, ierr))

    if (my_id == 0) write(*,'(A,2ES14.4)') &
      "[Metriplectic] T6b: mass-model gap |M_red z - B_diag z|/|B_diag z| (psi, u) = ", &
      gpsi/max(npsi_r,tiny(1.d0)), gu/max(nu_r,tiny(1.d0))

    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(VecDestroy(x4, ierr))
    PetscCallA(VecDestroy(y4, ierr))
  end subroutine run_T6


  !====================================================================
  ! T5b: Ritz values of the FULL-SWEEP-preconditioned 4-var system
  ! (gate G2, part 2). Same factorization-lag protocol as T5a.
  ! Expectation: the T5a dissipative band and the sub-unit band collapse
  ! into the treated spectrum; residual outliers = kink + model mismatch.
  !====================================================================
  subroutine run_T5b(comm, my_id)
    integer, intent(in) :: comm, my_id

    KSP :: ksp
    PC  :: pc
    Vec :: b4, x4
    PetscRandom :: rnd
    PetscReal :: r_eig(60), c_eig(60)
    PetscInt  :: neig, its
    KSPConvergedReason :: reason
    PetscErrorCode :: ierr
    integer :: it3, i, jmin
    real*8  :: tau_run, tau_s, tmp, re_min, re_max, im_max
    character(len=64) :: head
    character(len=16) :: converged_txt

    tau_run = m_tau

    do it3 = 1, 3
      tau_s = tau_run * 4.d0**(it3 - 2)
      call metriplectic_refresh_Pu(tau_s)

      PetscCallA(MatCreateVecs(m_A4, x4, b4, ierr))
      PetscCallA(PetscRandomCreate(comm, rnd, ierr))
      PetscCallA(VecSetRandom(b4, rnd, ierr))

      PetscCallA(KSPCreate(comm, ksp, ierr))
      PetscCallA(KSPSetOperators(ksp, m_A4, m_A4, ierr))
      PetscCallA(KSPSetType(ksp, KSPGMRES, ierr))
      PetscCallA(KSPGMRESSetRestart(ksp, 60, ierr))
      PetscCallA(KSPSetTolerances(ksp, 1.d-10, 1.d-50, PETSC_CURRENT_REAL, 60, ierr))
      PetscCallA(KSPSetComputeEigenvalues(ksp, PETSC_TRUE, ierr))
      PetscCallA(KSPGetPC(ksp, pc, ierr))
      PetscCallA(PCSetType(pc, PCSHELL, ierr))
      PetscCallA(PCShellSetApply(pc, metriplectic_sweep_apply_4v, ierr))

      call KSPSolve(ksp, b4, x4, ierr)
      PetscCallA(KSPGetIterationNumber(ksp, its, ierr))
      PetscCallA(KSPGetConvergedReason(ksp, reason, ierr))
      converged_txt = "not converged"
      if (reason > 0) converged_txt = "converged"
      PetscCallA(KSPComputeEigenvalues(ksp, 60, r_eig, c_eig, neig, ierr))

      do i = 1, neig-1
        jmin = i + minloc(r_eig(i:neig), 1) - 1
        if (jmin /= i) then
          tmp = r_eig(i); r_eig(i) = r_eig(jmin); r_eig(jmin) = tmp
          tmp = c_eig(i); c_eig(i) = c_eig(jmin); c_eig(jmin) = tmp
        endif
      enddo
      re_min = r_eig(1); re_max = r_eig(1); im_max = 0.d0
      do i = 1, neig
        re_max = max(re_max, r_eig(i))
        im_max = max(im_max, abs(c_eig(i)))
      enddo

      if (my_id == 0) then
        write(head,'(A,ES11.4,A,ES11.4)') " P_u at tau_s=", tau_s, ", system tau=", tau_run
        write(*,'(A,A,A,A)')    "[Metriplectic] T5b:", trim(head), &
                                ", order ", mpc_sweep_order
        write(*,'(A,I5,A,A)')   "[Metriplectic] T5b: GMRES iterations = ", its, &
                                ", ", trim(converged_txt)
        write(*,'(A,3ES13.4)')  "[Metriplectic] T5b: min Re, max Re, max |Im| = ", &
                                re_min, re_max, im_max
        write(*,'(A,I4,A)')     "[Metriplectic] T5b: ", neig, " Ritz values (Re, Im):"
        do i = 1, neig
          write(*,'(A,2ES14.5)') "[Metriplectic] T5b:   ", r_eig(i), c_eig(i)
        enddo
      endif

      PetscCallA(KSPDestroy(ksp, ierr))
      PetscCallA(PetscRandomDestroy(rnd, ierr))
      PetscCallA(VecDestroy(b4, ierr))
      PetscCallA(VecDestroy(x4, ierr))
    enddo

    call metriplectic_refresh_Pu(tau_run)
  end subroutine run_T5b


  !====================================================================
  ! T5c: Ritz values of the production full-system apply on the FULL
  ! Jacobian (rho/T recovery included) — what the outer FGMRES will see.
  ! First measurement of the untreated pressure back-coupling band.
  !====================================================================
  subroutine run_T5c(comm, my_id, A_full)
    integer, intent(in) :: comm, my_id
    Mat,     intent(in) :: A_full

    KSP :: ksp
    PC  :: pc
    Vec :: b, x
    PetscRandom :: rnd
    PetscReal :: r_eig(60), c_eig(60)
    PetscInt  :: neig, its
    KSPConvergedReason :: reason
    PetscErrorCode :: ierr
    integer :: i, jmin
    real*8  :: tmp, re_min, re_max, im_max
    character(len=16) :: converged_txt

    PetscCallA(MatCreateVecs(A_full, x, b, ierr))
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))
    PetscCallA(VecSetRandom(b, rnd, ierr))

    PetscCallA(KSPCreate(comm, ksp, ierr))
    PetscCallA(KSPSetOperators(ksp, A_full, A_full, ierr))
    PetscCallA(KSPSetType(ksp, KSPGMRES, ierr))
    PetscCallA(KSPGMRESSetRestart(ksp, 60, ierr))
    PetscCallA(KSPSetTolerances(ksp, 1.d-10, 1.d-50, PETSC_CURRENT_REAL, 60, ierr))
    PetscCallA(KSPSetComputeEigenvalues(ksp, PETSC_TRUE, ierr))
    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(pc, metriplectic_sweep_apply_full, ierr))

    call KSPSolve(ksp, b, x, ierr)
    PetscCallA(KSPGetIterationNumber(ksp, its, ierr))
    PetscCallA(KSPGetConvergedReason(ksp, reason, ierr))
    converged_txt = "not converged"
    if (reason > 0) converged_txt = "converged"
    PetscCallA(KSPComputeEigenvalues(ksp, 60, r_eig, c_eig, neig, ierr))

    do i = 1, neig-1
      jmin = i + minloc(r_eig(i:neig), 1) - 1
      if (jmin /= i) then
        tmp = r_eig(i); r_eig(i) = r_eig(jmin); r_eig(jmin) = tmp
        tmp = c_eig(i); c_eig(i) = c_eig(jmin); c_eig(jmin) = tmp
      endif
    enddo
    re_min = r_eig(1); re_max = r_eig(1); im_max = 0.d0
    do i = 1, neig
      re_max = max(re_max, r_eig(i))
      im_max = max(im_max, abs(c_eig(i)))
    enddo

    if (my_id == 0) then
      write(*,'(A,A)')       "[Metriplectic] T5c: production apply on the full system, order ", &
                             mpc_sweep_order
      write(*,'(A,I5,A,A)')  "[Metriplectic] T5c: GMRES iterations = ", its, &
                             ", ", trim(converged_txt)
      write(*,'(A,3ES13.4)') "[Metriplectic] T5c: min Re, max Re, max |Im| = ", &
                             re_min, re_max, im_max
      write(*,'(A,I4,A)')    "[Metriplectic] T5c: ", neig, " Ritz values (Re, Im):"
      do i = 1, neig
        write(*,'(A,2ES14.5)') "[Metriplectic] T5c:   ", r_eig(i), c_eig(i)
      enddo
    endif

    PetscCallA(KSPDestroy(ksp, ierr))
    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(VecDestroy(b, ierr))
    PetscCallA(VecDestroy(x, ierr))
  end subroutine run_T5c

  !====================================================================
  ! Dynamically track physical and continuous-Schur energies of vectors
  !====================================================================
  subroutine petsc_metriplectic_track_energies(time, istep, X_global, dX_global, my_id)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w
    use phys_module,    only: time_evol_zeta
    
    real*8, intent(in)  :: time
    integer, intent(in) :: istep
    Vec, intent(in)     :: X_global   !< The global state vector
    Vec, intent(in)     :: dX_global  !< The Newton increment vector (solution of GMRES)
    integer, intent(in) :: my_id      !< MPI rank

    ! Local PETSc variables
    Vec :: vec_psi, vec_u, vec_j, vec_w
    Vec :: dvec_psi, dvec_u
    Vec :: tmp_1, tmp_2
    PetscErrorCode :: ierr
    real*8 :: E_M, E_K, E_A, E_P
    real*8 :: dE_M, dE_K, dE_A, dE_P
    real*8 :: E_j_viol, E_w_viol

    if (.not. g_mctx%matrices_ready .or. .not. g_mctx%is_created) return

    ! Extract state variables
    PetscCallA(VecGetSubVector(X_global, g_mctx%is_var(var_psi), vec_psi, ierr))
    PetscCallA(VecGetSubVector(X_global, g_mctx%is_var(var_u),   vec_u,   ierr))
    PetscCallA(VecGetSubVector(X_global, g_mctx%is_var(var_zj),  vec_j,   ierr))
    PetscCallA(VecGetSubVector(X_global, g_mctx%is_var(var_w),   vec_w,   ierr))

    ! Extract increment variables
    PetscCallA(VecGetSubVector(dX_global, g_mctx%is_var(var_psi), dvec_psi, ierr))
    PetscCallA(VecGetSubVector(dX_global, g_mctx%is_var(var_u),   dvec_u,   ierr))

    PetscCallA(VecDuplicate(vec_psi, tmp_1, ierr))
    PetscCallA(VecDuplicate(vec_psi, tmp_2, ierr))

    ! 1. Magnetic Flux Energy (E_M)
    PetscCallA(MatMult(g_mctx%M_psi, vec_psi, tmp_1, ierr))
    PetscCallA(VecDot(vec_psi, tmp_1, E_M, ierr))

    ! 2. Kinetic Energy (E_K)
    PetscCallA(MatMult(g_mctx%L_rho, vec_u, tmp_1, ierr))
    PetscCallA(VecDot(vec_u, tmp_1, E_K, ierr))

    ! 3. Alfvenic Perturbation Energy (E_A)
    PetscCallA(MatMult(g_mctx%W_para, vec_u, tmp_1, ierr))
    PetscCallA(VecDot(vec_u, tmp_1, E_A, ierr))

    ! 4. Total Preconditioner Schur Energy (E_P)
    PetscCallA(MatMult(g_mctx%Pu_aij, vec_u, tmp_1, ierr))
    PetscCallA(VecDot(vec_u, tmp_1, E_P, ierr))

    ! 5. Operator Energies on the Increment (\delta X)
    PetscCallA(MatMult(g_mctx%M_psi, dvec_psi, tmp_1, ierr))
    PetscCallA(VecDot(dvec_psi, tmp_1, dE_M, ierr))

    PetscCallA(MatMult(g_mctx%L_rho, dvec_u, tmp_1, ierr))
    PetscCallA(VecDot(dvec_u, tmp_1, dE_K, ierr))

    PetscCallA(MatMult(g_mctx%W_para, dvec_u, tmp_1, ierr))
    PetscCallA(VecDot(dvec_u, tmp_1, dE_A, ierr))

    PetscCallA(MatMult(g_mctx%Pu_aij, dvec_u, tmp_1, ierr))
    PetscCallA(VecDot(dvec_u, tmp_1, dE_P, ierr))

    ! 6. Current constraint violation (j): || B31*psi + B33*j ||^2
    PetscCallA(MatMult(g_mctx%B_31s, vec_psi, tmp_1, ierr))
    PetscCallA(MatMultAdd(g_mctx%B_33s, vec_j, tmp_1, tmp_1, ierr))
    PetscCallA(VecNorm(tmp_1, NORM_2, E_j_viol, ierr))
    E_j_viol = E_j_viol**2.d0

    ! 7. Vorticity constraint violation (w): || B42*u + B44*w ||^2
    PetscCallA(MatMult(g_mctx%B_42s, vec_u, tmp_1, ierr))
    PetscCallA(MatMultAdd(g_mctx%B_44s, vec_w, tmp_1, tmp_1, ierr))
    PetscCallA(VecNorm(tmp_1, NORM_2, E_w_viol, ierr))
    E_w_viol = E_w_viol**2.d0

    PetscCallA(VecDestroy(tmp_1, ierr))
    PetscCallA(VecDestroy(tmp_2, ierr))

    PetscCallA(VecRestoreSubVector(X_global,  g_mctx%is_var(var_psi), vec_psi,  ierr))
    PetscCallA(VecRestoreSubVector(X_global,  g_mctx%is_var(var_u),   vec_u,    ierr))
    PetscCallA(VecRestoreSubVector(X_global,  g_mctx%is_var(var_zj),  vec_j,    ierr))
    PetscCallA(VecRestoreSubVector(X_global,  g_mctx%is_var(var_w),   vec_w,    ierr))
    PetscCallA(VecRestoreSubVector(dX_global, g_mctx%is_var(var_psi), dvec_psi, ierr))
    PetscCallA(VecRestoreSubVector(dX_global, g_mctx%is_var(var_u),   dvec_u,   ierr))

    if (my_id == 0) then
      write(*,'(A)') "[Metriplectic] ---- Operator Energies (State) ----"
      write(*,'(A,ES14.6)') "[Metriplectic] E_M (Flux Mass)     = ", E_M
      write(*,'(A,ES14.6)') "[Metriplectic] E_K (Kinetic Flow)  = ", E_K
      write(*,'(A,ES14.6)') "[Metriplectic] E_A (Alfvenic Pert) = ", E_A
      write(*,'(A,ES14.6)') "[Metriplectic] E_P (Total Schur)   = ", E_P
      write(*,'(A)') "[Metriplectic] ---- Constraint Violations ----"
      write(*,'(A,ES14.6)') "[Metriplectic] E_j_viol            = ", E_j_viol
      write(*,'(A,ES14.6)') "[Metriplectic] E_w_viol            = ", E_w_viol
      write(*,'(A)') "[Metriplectic] ---- Operator Energies (Delta) ----"
      write(*,'(A,ES14.6)') "[Metriplectic] dE_M (Flux Mass)    = ", dE_M
      write(*,'(A,ES14.6)') "[Metriplectic] dE_K (Kinetic Flow) = ", dE_K
      write(*,'(A,ES14.6)') "[Metriplectic] dE_A (Alfvenic Pert)= ", dE_A
      write(*,'(A,ES14.6)') "[Metriplectic] dE_P (Total Schur)  = ", dE_P

      open(unit=123, file='metriplectic_energies.csv', position='append', status='unknown')
      if (istep == 1) then
        write(123, '(A)') "# Metriplectic Operator Energies"
        write(123, '(A)') "# E_M = Int( (1/R) * psi^2 ) dV"
        write(123, '(A)') "# E_K = Int( rho_hat * R * |Grad(u)|^2 ) dV"
        write(123, '(A)') "# E_A = Int( (1/R) * |Grad(R^2 B.Grad u)|^2 ) dV"
        write(123, '(A)') "# E_P = E_K + tau^2 * E_A"
        write(123, '(A)') "# E_j_viol = || B31*psi + B33*j ||^2"
        write(123, '(A)') "# E_w_viol = || B42*u + B44*w ||^2"
        write(123, '(A)') "time,step,E_M,E_K,E_A,E_P,E_j_viol,E_w_viol,dE_M,dE_K,dE_A,dE_P"
      endif
      write(123, '(ES14.6,I8,10ES16.8)') time, istep, E_M, E_K, E_A, E_P, E_j_viol, E_w_viol, dE_M, dE_K, dE_A, dE_P
      close(123)
    endif
  end subroutine petsc_metriplectic_track_energies

#endif
end module mod_petsc_pc_metriplectic_analysis
