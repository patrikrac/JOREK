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
!   T4  : harmonic n-n' coupling table of D_op (Remark 6 measurement)
!   T5a : Ritz values of the ideal-half-preconditioned 4-var system
!         (diagnostic PCSHELL = prototype of the Slice-B apply)
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
  implicit none
  private

  public :: petsc_metriplectic_run_analysis

  ! --- Module state consumed by the PCSHELL apply callback ---
  Mat :: m_A4                          !< 4-var reference system (AIJ)
  Mat :: m_B31, m_B42                  !< constraint coupling blocks
  KSP :: m_ksp_Mpsi                    !< consistent-mass solve on M_psi
  KSP :: m_ksp_Pu                      !< direct solve on composed P_u
  KSP :: m_ksp_B33, m_ksp_B44          !< constraint-mass solves
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
    integer :: comm, v

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

    ! --- Variable index sets (independent copy; layout as in
    !     mod_petsc_pc_physics_construction::create_variable_index_sets) ---
    if (.not. g_mctx%is_created) then
      call create_index_sets(A_full, comm)
    endif

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

    ! --- 4-var nest -> AIJ (row-major block order) ---
    mats_nest       = PETSC_NULL_MAT
    mats_nest( 1) = B11;  mats_nest( 2) = B12;  mats_nest( 3) = B13
    mats_nest( 5) = B21;  mats_nest( 6) = B22;  mats_nest( 7) = B23;  mats_nest( 8) = B24
    mats_nest( 9) = B31;  mats_nest(11) = B33
    mats_nest(14) = B42;  mats_nest(16) = B44
    PetscCallA(MatCreateNest(comm, 4, PETSC_NULL_IS, 4, PETSC_NULL_IS, mats_nest, A_nest, ierr))
    PetscCallA(MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, m_A4, ierr))
    PetscCallA(MatDestroy(A_nest, ierr))

    ! --- Work vectors and constraint/mass solvers ---
    call create_work_vecs(ierr)
    call setup_mass_ksp(g_mctx%M_psi, m_ksp_Mpsi, comm)
    call setup_lu_ksp(B33, m_ksp_B33, comm, symmetric=.false.)
    call setup_lu_ksp(B44, m_ksp_B44, comm, symmetric=.false.)

    ! --- Checks ---
    call run_T1(comm, my_id)
    call run_T2(comm, my_id, metriplectic_analysis_nsweep)

    call setup_pu_solver(m_tau, comm)   ! compose + factor P_u at the run's tau
    call run_T3(comm, my_id)
    call run_T4(comm, my_id)
    call run_T5a(comm, my_id)

    ! --- Cleanup (keep g_mctx element operators; destroy analysis objects) ---
    PetscCallA(KSPDestroy(m_ksp_Mpsi, ierr))
    PetscCallA(KSPDestroy(m_ksp_B33, ierr))
    PetscCallA(KSPDestroy(m_ksp_B44, ierr))
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
    do v = 1, 6
      PetscCallA(ISDestroy(g_mctx%is_var(v), ierr))
    enddo
    g_mctx%is_created = .false.

    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] ================ analysis complete ================"
  end subroutine petsc_metriplectic_run_analysis


  !====================================================================
  ! Setup helpers
  !====================================================================

  !> Variable index sets on A_full's layout (independent of the old PC ctx).
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


  !> (Re)compose P_u at tau_in, convert to AIJ, factor with MUMPS Cholesky.
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
  ! Diagnostic PCSHELL apply: segregated ideal-half solve with
  ! post-hoc constraint recovery (prototype of the Slice-B apply;
  ! spec Sec. 4, note Sec. "Elimination" for the tau/(1+zeta) scaling).
  !
  !   h      = M_psi^-1 r_psi / (1+zeta)
  !   rhs_u  = -r_u/(1+zeta) - tau * Dp h        (u-row negation: note Sec. 6)
  !   du     = P_u^-1 rhs_u
  !   dpsi   = h - tau * M_psi^-1 (D du)
  !   dj     = B33^-1 (r_j - B31 dpsi)           (constraint recovery)
  !   dw     = B44^-1 (r_w - B42 du)
  !====================================================================
  subroutine metriplectic_halfpc_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    call unpack_4v(x, m_rpsi, m_ru, m_rj, m_rw, ierr)

    ! h = M^-1 r_psi / (1+zeta)
    call KSPSolve(m_ksp_Mpsi, m_rpsi, m_h, ierr)
    call VecScale(m_h, 1.d0/m_opz, ierr)

    ! rhs_u = -r_u/(1+zeta) - tau * Dp h   (accumulated in m_t2)
    call MatMult(g_mctx%Dp_op, m_h, m_t2, ierr)
    call VecAXPBY(m_t2, -1.d0/m_opz, -m_tau, m_ru, ierr)

    ! du = P_u^-1 rhs_u
    call KSPSolve(m_ksp_Pu, m_t2, m_du, ierr)

    ! dpsi = h - tau * M^-1 (D du)
    call MatMult(g_mctx%D_op, m_du, m_t1, ierr)
    call KSPSolve(m_ksp_Mpsi, m_t1, m_dpsi, ierr)
    call VecAYPX(m_dpsi, -m_tau, m_h, ierr)

    ! dj = B33^-1 (r_j - B31 dpsi)
    call MatMult(m_B31, m_dpsi, m_t1, ierr)
    call VecAYPX(m_t1, -1.d0, m_rj, ierr)
    call KSPSolve(m_ksp_B33, m_t1, m_dj, ierr)

    ! dw = B44^-1 (r_w - B42 du)
    call MatMult(m_B42, m_du, m_t2, ierr)
    call VecAYPX(m_t2, -1.d0, m_rw, ierr)
    call KSPSolve(m_ksp_B44, m_t2, m_dw, ierr)

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
  ! P_u is re-composed at tau/4, tau, 4*tau (system fixed at tau):
  ! measures spectrum quality and its sensitivity to compose-lag.
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
      call setup_pu_solver(tau_s, comm)

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

    ! Restore P_u at the run's tau
    call setup_pu_solver(tau_run, comm)
  end subroutine run_T5a

#endif
end module mod_petsc_pc_metriplectic_analysis
