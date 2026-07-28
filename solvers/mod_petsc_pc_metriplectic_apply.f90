module mod_petsc_pc_metriplectic_apply
!----------------------------------------------------------------
! Metriplectic HSS sweep apply (stage B/C; spec Sec. 6, note Sec.
! "The constrained HSS sweep", plan 2026-07-16 Task 4).
!
! One alternating sweep in the reduced (psi,u) space:
!   order 'SK':  pair solves -> M_red mat-vec -> ideal half -> j/w recovery
!   order 'KS':  ideal half -> M_red mat-vec -> pair solves (final incl. j/w)
! Constraint residuals r_j, r_w are consumed with the FINAL dpsi/du by
! whichever half runs last (single-consumption rule, note Sec. sweep).
! rho/T are recovered block-triangularly outside the sweep.
!
! Sign discipline (note Sec. 6 / eq. (Mred)): JOREK's u-row carries
! negative inertia; every scalar below is annotated with its note origin.
!----------------------------------------------------------------
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_metriplectic_ctx, only: g_mctx
  use mod_petsc_pc_commutator_table, only: CM_NOP, CM_MAXC, CM_LABLEN, &
        CM_OP_QR, cm_mstar_mult
  implicit none
  private

  public :: metriplectic_ideal_half_solve
  public :: metriplectic_diss_half_solve
  public :: metriplectic_mass_apply
  public :: metriplectic_recover_jw
  public :: metriplectic_recover_rhoT
  public :: metriplectic_sweep_apply_4v
  public :: metriplectic_sweep_apply_full
  public :: metriplectic_build_ps_shell
  public :: metriplectic_ps_ldu_solve
  public :: metriplectic_build_cm_shell, metriplectic_cm_bind, cm_set_variant
  public :: cm_schur_variant, cm_ab_active, cm_counters_reset, cm_counters_get
  public :: pack_2v, unpack_2v, pack_4v, unpack_4v

  !> Active sweep order ('SK' default; 'KS' = ideal half first). Set from
  !! the namelist flag at PC setup; the analysis flips it for T6.
  character(len=2), public, save :: mpc_sweep_order = 'SK'

  !> Which operator step 3 of the PS-LDU uses:
  !!   'LEGACY' (default) -- untouched pre-existing behaviour, i.e. ksp_Suw
  !!                         when ps_inner_it > 0 else ksp_Puw. Nothing that
  !!                         does not opt in may ever leave this value.
  !!   'M0D'              -- assembled P_uw, single pass (the incumbent)
  !!   'EXACT'            -- exact Schur shell (the ceiling)
  !!   any other label    -- candidate resolved against the shared table,
  !!                         i.e. the commutator device
  !! Never assign directly -- use cm_set_variant, which re-resolves cm_cand.
  character(len=8), save :: cm_schur_variant = 'LEGACY'

  !> Set by the A/B harness only: force the PS-LDU path irrespective of
  !! khalf_mode, so the diagnostic can run alongside any production PC.
  logical, save :: cm_ab_active = .false.

  ! --- module work vecs (1-var sized), created on first use ---
  logical :: a_work_ready = .false.
  Vec :: a_h, a_t1, a_t2                      ! ideal-half internals
  Vec :: a_p1, a_u1, a_j1, a_w1               ! first-half outputs
  Vec :: a_yp, a_yu                           ! middle mass outputs
  Vec :: a_rpsi, a_ru, a_rj, a_rw             ! unpacked residuals
  Vec :: a_dpsi, a_du, a_dj, a_dw             ! solution components
  Vec :: a_rrho, a_rT, a_drho, a_dT           ! rho/T components (full apply)

  ! --- stage-D pair-Schur shell internals (spec Sec. 7.3) ---
  ! Dedicated vecs: the S_uw shell mult runs inside KSPSolve(ksp_Suw),
  ! which the LDU apply calls between its two pivot solves -- it must not
  ! touch the a_* pool the LDU is using. The pair packed vecs
  ! wv_pair_psij_1/2 ARE safe here (no pivot solve is in flight during
  ! the inner Schur iteration).
  logical :: ps_shell_ready = .false.
  Vec :: s_xu, s_xw, s_yu, s_yw               ! (u,w) components of x / y
  Vec :: s_t1, s_zj, s_h, s_hj                ! psi/j-space intermediates
  Vec :: s_cu                                 ! u-space wave correction

  ! --- commutator-device M_* Schur internals (Slice 1) ---
  ! Same dedicated-vec discipline as the s_* pool above: the S_cm mult runs
  ! inside MatComputeOperator, called from the LDU's variant setup.
  logical :: cm_shell_ready = .false.
  Vec :: c_xu, c_xw, c_mu                     ! (u,w) split of x, M_* image
  Vec :: c_t1, c_t2                           ! matvec scratch / A_M* image
  Vec :: c_zero                               ! permanent zero (w-slot padding)
  Vec :: c_pk1, c_pk2                         ! packed (u,w) work for the shell mult
  !> Candidate binding: coefficients + operator handles for the whole table,
  !! gathered once per PC build. cm_cand indexes the ACTIVE row.
  real*8,  save :: cm_coef(CM_MAXC, CM_NOP) = 0.d0
  integer, save :: cm_quop(CM_MAXC) = 0
  character(len=CM_LABLEN), save :: cm_lab(CM_MAXC) = ' '
  integer, save :: cm_ncand = 0
  integer, save :: cm_cand  = 0               ! active row; 0 = unresolved
  Mat,     save :: cm_op(CM_NOP)
  logical, save :: cm_bound = .false.
  !> Label the explicit T_cm currently corresponds to ('' = none/invalid).
  character(len=8), save :: cm_exp_label = ' '
  !> Hard cap on the (u,w) pair dimension for the explicit path. T_cm is
  !! dense, so this bounds memory at ~ (8 * MAXDIM^2) bytes.
  integer, parameter :: CM_EXP_MAXDIM = 20000
  !> Inner-iteration accounting for the A/B report. With the explicit path
  !! there is no inner solve, so these stay zero and the report prints '--'.
  integer, save :: cm_inner_tot = 0, cm_inner_calls = 0

contains

  subroutine ensure_work(ierr)
    PetscErrorCode :: ierr
    if (a_work_ready) return
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_h,    ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_t1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_t2,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_p1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_u1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_j1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_w1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_yp,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_yu,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rpsi, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_ru,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rj,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rw,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dpsi, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_du,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dj,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dw,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rrho, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rT,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_drho, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dT,   ierr))
    a_work_ready = .true.
  end subroutine ensure_work


  !====================================================================
  ! Stage-D pair-Schur infrastructure (spec Sec. 7.3; note Sec. pschur).
  !
  ! S_uw_shell: matrix-free exact Schur on the (u,w) pair,
  !   S_uw = A_uw - tdt^2 [Dp (A_psij^-1)_psipsi D]_uu   (note eq. (Suw))
  ! with tdt = theta*dt = (1+zeta)*tau; one (psi,j) pair solve per mult.
  !
  ! ksp_Suw: FGMRES on the shell, preconditioned by P_uw through a
  ! PCSHELL that applies ksp_Puw (single factorization, owned by
  ! ksp_Puw; same sharing pattern as the analysis module's pu_pc_apply).
  !
  ! Created ONCE; all g_mctx handles (A_pair_uw, D_op, Dp_op,
  ! ksp_pair_psij, ksp_Puw) are dereferenced at call time, so PC rebuilds
  ! need no shell rebuild.
  !====================================================================
  subroutine metriplectic_build_ps_shell(comm)
    integer, intent(in) :: comm
    PetscErrorCode :: ierr
    PetscInt :: n_loc, n_glob
    PC  :: pc

    if (ps_shell_ready) return

    ! dedicated work vecs (all 1-var layouts coincide, cf. ensure_work)
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_xu, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_xw, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_yu, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_yw, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_t1, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_zj, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_h,  ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_hj, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, s_cu, ierr))

    ! shell with the (u,w) pair layout
    call MatGetLocalSize(g_mctx%A_pair_uw, n_loc, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(g_mctx%A_pair_uw, n_glob, PETSC_NULL_INTEGER, ierr)
    PetscCallA(MatCreateShell(comm, n_loc, n_loc, n_glob, n_glob, &
                              PETSC_NULL_INTEGER, g_mctx%S_uw_shell, ierr))
    PetscCallA(MatShellSetOperation(g_mctx%S_uw_shell, MATOP_MULT, &
                                    metriplectic_Suw_mult, ierr))

    ! inner Schur solver: FGMRES(S_uw) with P_uw^-1 (via ksp_Puw) as PC
    PetscCallA(KSPCreate(comm, g_mctx%ksp_Suw, ierr))
    PetscCallA(KSPSetOperators(g_mctx%ksp_Suw, g_mctx%S_uw_shell, &
                               g_mctx%S_uw_shell, ierr))
    PetscCallA(KSPSetType(g_mctx%ksp_Suw, KSPFGMRES, ierr))
    PetscCallA(KSPGetPC(g_mctx%ksp_Suw, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(pc, metriplectic_puw_pc_apply, ierr))
    if (g_mctx%ps_inner_it > 0) then
      call KSPSetTolerances(g_mctx%ksp_Suw, g_mctx%ps_inner_tol, &
                            PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, &
                            g_mctx%ps_inner_it, ierr)
    else
      ! single-pass production never calls ksp_Suw; this default serves
      ! the analysis uses (PS1 tightens it explicitly)
      call KSPSetTolerances(g_mctx%ksp_Suw, g_mctx%ps_inner_tol, &
                            PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, &
                            30, ierr)
    endif
    PetscCallA(KSPSetInitialGuessNonzero(g_mctx%ksp_Suw, PETSC_FALSE, ierr))

    ps_shell_ready = .true.
  end subroutine metriplectic_build_ps_shell


  !--------------------------------------------------------------------
  !> MATOP_MULT of S_uw_shell:  y = A_uw x - tdt^2 [Dp R^-1 D x_u]_u
  !! (note eq. (Suw); R^-1 action = psi-component of one (psi,j) pair
  !!  solve with RHS (D x_u, 0)).
  !--------------------------------------------------------------------
  subroutine metriplectic_Suw_mult(A, x, y, ierr)
    use phys_module, only: time_evol_zeta
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    real*8 :: tdt

    tdt = g_mctx%dt_theta * (1.d0 + time_evol_zeta)     ! = theta*dt

    call MatMult(g_mctx%A_pair_uw, x, y, ierr)          ! y = A_uw x
    call unpack_2v(x, s_xu, s_xw, ierr)
    call MatMult(g_mctx%D_op, s_xu, s_t1, ierr)         ! D x_u (psi space)
    call VecZeroEntries(s_zj, ierr)
    call pack_2v(s_t1, s_zj, g_mctx%wv_pair_psij_1, ierr)
    call KSPSolve(g_mctx%ksp_pair_psij, g_mctx%wv_pair_psij_1, &
                  g_mctx%wv_pair_psij_2, ierr)          ! (R^-1 D x_u, *)
    call unpack_2v(g_mctx%wv_pair_psij_2, s_h, s_hj, ierr)
    call MatMult(g_mctx%Dp_op, s_h, s_cu, ierr)         ! Dp R^-1 D x_u (u space)
    call unpack_2v(y, s_yu, s_yw, ierr)
    call VecAXPY(s_yu, -tdt*tdt, s_cu, ierr)            ! minus: note eq. (Suw)
    call pack_2v(s_yu, s_yw, y, ierr)
    ierr = 0
  end subroutine metriplectic_Suw_mult


  !--------------------------------------------------------------------
  !> PCSHELL apply for ksp_Suw: one P_uw^-1 application via ksp_Puw
  !! (shares the single MUMPS factorization owned by ksp_Puw).
  !--------------------------------------------------------------------
  subroutine metriplectic_puw_pc_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    call KSPSolve(g_mctx%ksp_Puw, x, y, ierr)
    ierr = 0
  end subroutine metriplectic_puw_pc_apply


  !====================================================================
  ! Commutator-device M_* Schur (Slice 1; note Eq. (39), Sec. 6.2).
  !
  ! The exact Schur term tdt^2 A_Dp B11^-1 A_D is replaced using the
  ! intertwining relation B11^-1 A_D ~ Q_psi^-1 A_D A_M*^-1 Q_u, which
  ! turns it into tdt^2 W_para A_M*^-1 Q_u -- no M^-1 anywhere. Folding
  ! M_* = Q_u^-1 A_M* out to the right gives the pair operator
  !
  !   T_pair = [ B22 M_* - tdt^2 W_para   B24 ]
  !            [ B42 M_*                  B44 ]
  !
  !   T_pair (chi_u, dw) = (g_u, r_w),      du = M_* chi_u
  !
  ! W_para is the continuum assembly of A_Dp Q_psi^-1 A_D, so the psi-side
  ! Riesz map has already cancelled; Q_u^-1 survives ONLY inside B22 M_*
  ! and B42 M_*, and is applied here as an exact MUMPS mass solve.
  !
  ! CONSISTENCY: at M0 (A_M* = opz*Q_u) this is exactly (1+zeta)*P_uw with
  ! du = opz*chi_u, so the M0 candidate must reproduce the 'M0D' path to
  ! round-off -- now via a completely different route (explicit assembly +
  ! LU rather than a one-iteration Krylov solve), which makes it a much
  ! stronger check of the T_pair algebra.
  !
  ! The shell exists only to be materialized by cm_build_explicit; it is
  ! never handed to a Krylov method.
  !====================================================================
  subroutine metriplectic_build_cm_shell(comm)
    integer, intent(in) :: comm
    PetscErrorCode :: ierr
    PetscInt :: n_loc, n_glob

    if (cm_shell_ready) return

    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, c_xu,  ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, c_xw,  ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, c_mu,  ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, c_t1,  ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, c_t2,  ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, c_zero, ierr))
    PetscCallA(VecZeroEntries(c_zero, ierr))
    PetscCallA(MatCreateVecs(g_mctx%A_pair_uw, c_pk1, c_pk2, ierr))

    call MatGetLocalSize(g_mctx%A_pair_uw, n_loc, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(g_mctx%A_pair_uw, n_glob, PETSC_NULL_INTEGER, ierr)
    PetscCallA(MatCreateShell(comm, n_loc, n_loc, n_glob, n_glob, &
                              PETSC_NULL_INTEGER, g_mctx%S_cm_shell, ierr))
    PetscCallA(MatShellSetOperation(g_mctx%S_cm_shell, MATOP_MULT, &
                                    metriplectic_Scm_mult, ierr))

    cm_shell_ready = .true.
  end subroutine metriplectic_build_cm_shell


  !--------------------------------------------------------------------
  !> Materialize T_pair for the active candidate and LU-factor it, so
  !! step 3 solves it EXACTLY and the measured outer counts carry no
  !! inner-solver artifact.
  !!
  !! Cost: one shell application per column (each carrying a B33 MUMPS
  !! back-solve) and a dense n x n result, because M_* = B33^-1 A_M* is
  !! dense. Diagnostic only -- hence the hard size guard.
  !!
  !! Rebuilt when the active label changes (the A/B cycles variants) or
  !! when metriplectic_cm_bind invalidates the cache on a PC rebuild.
  !! MATDENSE first, then convert: assembling a dense operator straight
  !! into MATAIJ would reallocate on every row.
  !--------------------------------------------------------------------
  subroutine cm_build_explicit(my_id)
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr
    PetscInt :: n_glob
    integer :: comm
    PC :: pc

    if (cm_cand <= 0) return                              ! M0D / EXACT / LEGACY
    if (trim(cm_exp_label) == trim(cm_schur_variant)) return

    PetscCallA(MatGetSize(g_mctx%A_pair_uw, n_glob, PETSC_NULL_INTEGER, ierr))
    if (n_glob > CM_EXP_MAXDIM) then
      if (my_id == 0) then
        write(*,'(A,I0,A,I0,A)') "[CommPC] ERROR: explicit T_pair would be a dense ", &
          n_glob, " x ", n_glob, " operator"
        write(*,'(A,I0,A)') "[CommPC] the commutator Schur path is DIAGNOSTIC ONLY " // &
          "(small meshes); CM_EXP_MAXDIM = ", CM_EXP_MAXDIM, &
          ". Use a coarser grid, or run only the 'EXACT'/'M0D' variants."
      endif
      error stop "commutator explicit T_pair too large"
    endif

    if (g_mctx%Tcm_ready) then
      PetscCallA(KSPDestroy(g_mctx%ksp_Tcm, ierr))
      PetscCallA(MatDestroy(g_mctx%T_cm,    ierr))
      g_mctx%Tcm_ready = .false.
    endif

    PetscCallA(MatComputeOperator(g_mctx%S_cm_shell, MATDENSE, g_mctx%T_cm, ierr))
    PetscCallA(MatConvert(g_mctx%T_cm, MATMPIAIJ, MAT_INPLACE_MATRIX, g_mctx%T_cm, ierr))

    ! MUMPS LU. Cannot reuse the assembly module's setup_lu_ksp: that module
    ! already uses THIS one (metriplectic_build_ps_shell), so the dependency
    ! may not be reversed.
    call PetscObjectGetComm(g_mctx%T_cm, comm, ierr)
    PetscCallA(KSPCreate(comm, g_mctx%ksp_Tcm, ierr))
    PetscCallA(KSPSetOperators(g_mctx%ksp_Tcm, g_mctx%T_cm, g_mctx%T_cm, ierr))
    PetscCallA(KSPSetType(g_mctx%ksp_Tcm, KSPPREONLY, ierr))
    PetscCallA(KSPGetPC(g_mctx%ksp_Tcm, pc, ierr))
    PetscCallA(PCSetType(pc, PCLU, ierr))
    PetscCallA(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr))
    PetscCallA(KSPSetUp(g_mctx%ksp_Tcm, ierr))

    g_mctx%Tcm_ready = .true.
    cm_exp_label = cm_schur_variant
    if (my_id == 0) write(*,'(A,A,A,I0,A)') &
      "[CommPC] explicit T_pair(", trim(cm_schur_variant), ") built and factored (n = ", &
      n_glob, ")"
  end subroutine cm_build_explicit


  !--------------------------------------------------------------------
  !> Bind the shared candidate table to this PC build: gather the operator
  !! handles from the caller-supplied blocks and resolve the active label.
  !! Called from metriplectic_build_sweep once the sub-blocks exist.
  !--------------------------------------------------------------------
  subroutine metriplectic_cm_bind(B11, Q1R, QR, opz, tdt, eta, my_id)
    use mod_petsc_pc_commutator_table, only: cm_table_build, cm_ops_gather
    Mat,     intent(in) :: B11, Q1R, QR
    real*8,  intent(in) :: opz, tdt, eta
    integer, intent(in) :: my_id

    call cm_ops_gather(B11, Q1R, QR, cm_op)
    call cm_table_build(opz, tdt, eta, cm_coef, cm_quop, cm_lab, cm_ncand)
    cm_bound = .true.
    ! The blocks are fresh copies and opz/tdt/eta have moved with the time
    ! step, so any previously materialized T_cm is stale.
    cm_exp_label = ' '
    ! Re-resolve the currently selected variant against the fresh table.
    call cm_set_variant(cm_schur_variant, my_id)
  end subroutine metriplectic_cm_bind


  !--------------------------------------------------------------------
  !> Select the step-3 operator by label and resolve it against the table.
  !! 'M0D' and 'EXACT' are not table candidates and leave cm_cand = 0,
  !! which is exactly what the LDU dispatch keys off.
  !--------------------------------------------------------------------
  subroutine cm_set_variant(label, my_id)
    use mod_petsc_pc_commutator_table, only: cm_table_lookup
    character(len=*), intent(in) :: label
    integer,          intent(in) :: my_id

    cm_schur_variant = label
    cm_cand = 0
    if (trim(label) == 'M0D' .or. trim(label) == 'EXACT' &
        .or. trim(label) == 'LEGACY') return
    if (.not. cm_bound) return

    cm_cand = cm_table_lookup(cm_lab, cm_ncand, label)
    if (cm_cand == 0 .and. my_id == 0) then
      write(*,'(A)') "[CommPC] WARNING: candidate '" // trim(label) // &
        "' not in the table (building blocks not assembled?) -- using P_uw instead"
      return
    endif
    ! Materialize + factor T_pair for this candidate (no-op if already built
    ! for this label and this PC build).
    call cm_build_explicit(my_id)
  end subroutine cm_set_variant


  !--------------------------------------------------------------------
  !> MATOP_MULT of S_cm_shell:  y = A_pair_uw (M_* x_u, x_w)
  !!                                - tdt^2 (W_para x_u, 0)
  !--------------------------------------------------------------------
  subroutine metriplectic_Scm_mult(A, x, y, ierr)
    use phys_module, only: time_evol_zeta
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    real*8 :: tdt

    tdt = g_mctx%dt_theta * (1.d0 + time_evol_zeta)     ! = theta*dt

    call unpack_2v(x, c_xu, c_xw, ierr)
    call cm_mstar_apply(c_xu, c_mu, ierr)               ! c_mu = M_* x_u
    call pack_2v(c_mu, c_xw, c_pk1, ierr)
    call MatMult(g_mctx%A_pair_uw, c_pk1, y, ierr)      ! A_uw (M_* x_u, x_w)

    ! u-row wave term: -tdt^2 W_para x_u  (sign as in P_uw, note Sec. Puw)
    call MatMult(g_mctx%W_para, c_xu, c_t1, ierr)
    call pack_2v(c_t1, c_zero, c_pk2, ierr)
    call VecAXPY(y, -tdt**2, c_pk2, ierr)
    ierr = 0
  end subroutine metriplectic_Scm_mult


  !--------------------------------------------------------------------
  !> M_* x = Q_u^-1 A_M* x, with Q_u the candidate's own mass (B33 for the
  !! 1/R family, B44 for the R family -- Caution 5 of the note). Both are
  !! factored once under the sweep_once_done guard, so this is one MUMPS
  !! back-solve plus the table matvec.
  !! c_t2 receives A_M* x; c_t1 is the table matvec scratch.
  !--------------------------------------------------------------------
  subroutine cm_mstar_apply(x, y, ierr)
    Vec :: x, y
    PetscErrorCode :: ierr

    call cm_mstar_mult(cm_op, cm_coef, cm_cand, x, c_t2, c_t1, ierr)
    if (cm_quop(cm_cand) == CM_OP_QR) then
      call KSPSolve(g_mctx%ksp_B44, c_t2, y, ierr)
    else
      call KSPSolve(g_mctx%ksp_B33, c_t2, y, ierr)
    endif
  end subroutine cm_mstar_apply


  !> Reset / read the inner-iteration counters (A/B reporting).
  subroutine cm_counters_reset()
    cm_inner_tot   = 0
    cm_inner_calls = 0
  end subroutine cm_counters_reset

  subroutine cm_counters_get(tot, calls)
    integer, intent(out) :: tot, calls
    tot   = cm_inner_tot
    calls = cm_inner_calls
  end subroutine cm_counters_get


  !====================================================================
  ! Reduced K-half solve.
  ! Mode 'K2': exact coupled 2x2 MUMPS solve of the model Alfven system
  ! A_kideal — no Schur substitution, no T3c grid-scale tail (this fixed
  ! the intear FGMRES stall of 2026-07-16).
  ! Mode 'PU' (segregated Schur path, historical):
  !   h     = M_psi^-1 rpsi / (1+z)             [note SK step 3]
  !   RHS_u = -ru/(1+z) + tau * Dp h            [note Sec. 6: u-row negation
  !                                              flips BOTH terms -> +tau Dp h]
  !   du    = P_u^-1 RHS_u
  !   dpsi  = h - tau * M_psi^-1 (D du)
  !====================================================================
  subroutine metriplectic_ideal_half_solve(rpsi, ru, dpsi, du)
    use phys_module, only: time_evol_zeta
    Vec :: rpsi, ru, dpsi, du
    PetscErrorCode :: ierr
    real*8 :: opz, tau

    if (g_mctx%khalf_mode == 'K2') then
      call pack_2v(rpsi, ru, g_mctx%wv_kid_1, ierr)
      call KSPSolve(g_mctx%ksp_kideal, g_mctx%wv_kid_1, g_mctx%wv_kid_2, ierr)
      call unpack_2v(g_mctx%wv_kid_2, dpsi, du, ierr)
      return
    endif

    opz = 1.d0 + time_evol_zeta
    tau = g_mctx%dt_theta
    call ensure_work(ierr)

    call KSPSolve(g_mctx%ksp_Mpsi, rpsi, a_h, ierr)
    call VecScale(a_h, 1.d0/opz, ierr)

    call MatMult(g_mctx%Dp_op, a_h, a_t2, ierr)
    call VecAXPBY(a_t2, -1.d0/opz, +tau, ru, ierr)   ! t2 = -ru/(1+z) + tau*t2

    call KSPSolve(g_mctx%ksp_Pu, a_t2, du, ierr)

    call MatMult(g_mctx%D_op, du, a_t1, ierr)
    call KSPSolve(g_mctx%ksp_Mpsi, a_t1, dpsi, ierr)
    call VecAYPX(dpsi, -tau, a_h, ierr)              ! dpsi = h - tau*dpsi
  end subroutine metriplectic_ideal_half_solve


  !====================================================================
  ! S-half: coupled constraint-pair solves (mixed form, note eq. (pair)).
  !   [B11 B13; B31 B33](dpsi,dj) = (rpsi,rj)
  !   [B22 B24; B42 B44](du, dw ) = (ru,  rw)
  !====================================================================
  subroutine metriplectic_diss_half_solve(rpsi, ru, rj, rw, dpsi, du, dj, dw)
    Vec :: rpsi, ru, rj, rw, dpsi, du, dj, dw
    PetscErrorCode :: ierr

    call pack_2v(rpsi, rj, g_mctx%wv_pair_psij_1, ierr)
    call KSPSolve(g_mctx%ksp_pair_psij, g_mctx%wv_pair_psij_1, &
                  g_mctx%wv_pair_psij_2, ierr)
    call unpack_2v(g_mctx%wv_pair_psij_2, dpsi, dj, ierr)

    call pack_2v(ru, rw, g_mctx%wv_pair_uw_1, ierr)
    call KSPSolve(g_mctx%ksp_pair_uw, g_mctx%wv_pair_uw_1, &
                  g_mctx%wv_pair_uw_2, ierr)
    call unpack_2v(g_mctx%wv_pair_uw_2, du, dw, ierr)
  end subroutine metriplectic_diss_half_solve


  !====================================================================
  ! Middle operator M_red (note eq. (Mred)):
  !   y_psi = +(1+z) M_psi  z_psi
  !   y_u   = -(1+z) L_rho  z_u     <- MINUS: JOREK u-row negative inertia
  !====================================================================
  subroutine metriplectic_mass_apply(zpsi, zu, ypsi, yu)
    use phys_module, only: time_evol_zeta
    Vec :: zpsi, zu, ypsi, yu
    PetscErrorCode :: ierr
    real*8 :: opz

    opz = 1.d0 + time_evol_zeta
    call MatMult(g_mctx%M_psi, zpsi, ypsi, ierr)
    call VecScale(ypsi, +opz, ierr)
    call MatMult(g_mctx%L_rho, zu, yu, ierr)
    call VecScale(yu, -opz, ierr)
  end subroutine metriplectic_mass_apply


  !====================================================================
  ! Constraint recovery with the final dpsi/du (note SK step 4):
  !   dj = B33^-1 (rj - B31 dpsi),  dw = B44^-1 (rw - B42 du)
  !====================================================================
  subroutine metriplectic_recover_jw(rj, rw, dpsi, du, dj, dw)
    Vec :: rj, rw, dpsi, du, dj, dw
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    call MatMult(g_mctx%B_31s, dpsi, a_t1, ierr)
    call VecAYPX(a_t1, -1.d0, rj, ierr)              ! t1 = rj - B31 dpsi
    call KSPSolve(g_mctx%ksp_B33, a_t1, dj, ierr)

    call MatMult(g_mctx%B_42s, du, a_t1, ierr)
    call VecAYPX(a_t1, -1.d0, rw, ierr)              ! t1 = rw - B42 du
    call KSPSolve(g_mctx%ksp_B44, a_t1, dw, ierr)
  end subroutine metriplectic_recover_jw


  !====================================================================
  ! rho/T block-triangular recovery (outside the sweep; spec 6.1.5):
  !   drho = B55^-1 (rrho - B52 du),  dT = B66^-1 (rT - B62 du)
  !====================================================================
  subroutine metriplectic_recover_rhoT(rrho, rT, du, drho, dT)
    Vec :: rrho, rT, du, drho, dT
    PetscErrorCode :: ierr

    call MatMult(g_mctx%B_52s, du, g_mctx%wv_rho_2, ierr)
    call VecAYPX(g_mctx%wv_rho_2, -1.d0, rrho, ierr)
    call KSPSolve(g_mctx%ksp_B55, g_mctx%wv_rho_2, drho, ierr)

    call MatMult(g_mctx%B_62s, du, g_mctx%wv_T_2, ierr)
    call VecAYPX(g_mctx%wv_T_2, -1.d0, rT, ierr)
    call KSPSolve(g_mctx%ksp_B66, g_mctx%wv_T_2, dT, ierr)
  end subroutine metriplectic_recover_rhoT


  !====================================================================
  ! Stage-D pair-Schur block-LDU solve of the 4-field model K
  ! (mode 'PS'; spec Sec. 7.3, note eq. (ldu)). Exact inverse of A_k4
  ! when the Schur solve is exact (note Prop. ldu); the j-constraint row
  ! is structurally exact for ANY du, and single-pass also satisfies the
  ! w-constraint row exactly (note Rem. ldu-constraints).
  !
  !   1. pivot:      (dpsi0, dj0) = A_psij^-1 (rpsi, rj)
  !   2. Schur RHS:  gu = ru - tdt*Dp*dpsi0 ; gw = rw      [tdt = theta*dt]
  !   3. Schur:      (du, dw) = S_uw^-1 (gu, gw)
  !                  [ps_inner_it = 0: one P_uw^-1; else FGMRES(ksp_Suw)]
  !   4. back-subst: (dpsi, dj) = A_psij^-1 (rpsi - tdt*D*du, rj)
  !
  ! Signs: JOREK psi-row  B11 dpsi + tdt*D du + B13 dj = rpsi  and
  ! u-row  tdt*Dp dpsi + B22 du + B24 dw = ru, matching the A_k4 nest
  ! (assembly, spec Sec. 7.2 table) -- both correction terms enter with
  ! MINUS on the right-hand side.
  !====================================================================
  subroutine metriplectic_ps_ldu_solve(rpsi, ru, rj, rw, dpsi, du, dj, dw)
    use phys_module, only: time_evol_zeta
    Vec :: rpsi, ru, rj, rw, dpsi, du, dj, dw
    PetscErrorCode :: ierr
    real*8 :: tdt

    tdt = g_mctx%dt_theta * (1.d0 + time_evol_zeta)     ! = theta*dt
    call ensure_work(ierr)

    ! (1) pivot pair solve; dj0 (a_j1) is discarded
    call pack_2v(rpsi, rj, g_mctx%wv_pair_psij_1, ierr)
    call KSPSolve(g_mctx%ksp_pair_psij, g_mctx%wv_pair_psij_1, &
                  g_mctx%wv_pair_psij_2, ierr)
    call unpack_2v(g_mctx%wv_pair_psij_2, a_p1, a_j1, ierr)

    ! (2) Schur RHS: gu = ru - tdt*(Dp dpsi0)            [note eq. (ldu) 2]
    call MatMult(g_mctx%Dp_op, a_p1, a_t2, ierr)
    call VecAYPX(a_t2, -tdt, ru, ierr)
    call pack_2v(a_t2, rw, g_mctx%wv_uw_1, ierr)

    ! (3) Schur solve                                    [note eq. (ldu) 3]
    !     Variant dispatch. 'LEGACY' reproduces the pre-commutator
    !     behaviour exactly, so runs that do not opt in are unaffected.
    !     Only the commutator branch has to undo the M_* fold afterwards.
    if (cm_cand > 0) then
      ! T_pair solved EXACTLY (explicit + LU), so no inner-solver artifact
      call KSPSolve(g_mctx%ksp_Tcm, g_mctx%wv_uw_1, g_mctx%wv_uw_2, ierr)
      call unpack_2v(g_mctx%wv_uw_2, a_t1, dw, ierr)    ! a_t1 = chi_u
      call cm_mstar_apply(a_t1, du, ierr)               ! du = M_* chi_u
    else if (trim(cm_schur_variant) == 'EXACT') then
      call KSPSolve(g_mctx%ksp_Suw, g_mctx%wv_uw_1, g_mctx%wv_uw_2, ierr)
      call unpack_2v(g_mctx%wv_uw_2, du, dw, ierr)
    else if (trim(cm_schur_variant) == 'M0D') then
      call KSPSolve(g_mctx%ksp_Puw, g_mctx%wv_uw_1, g_mctx%wv_uw_2, ierr)
      call unpack_2v(g_mctx%wv_uw_2, du, dw, ierr)
    else if (g_mctx%ps_inner_it > 0) then               ! 'LEGACY'
      call KSPSolve(g_mctx%ksp_Suw, g_mctx%wv_uw_1, g_mctx%wv_uw_2, ierr)
      call unpack_2v(g_mctx%wv_uw_2, du, dw, ierr)
    else                                                ! 'LEGACY'
      call KSPSolve(g_mctx%ksp_Puw, g_mctx%wv_uw_1, g_mctx%wv_uw_2, ierr)
      call unpack_2v(g_mctx%wv_uw_2, du, dw, ierr)
    endif

    ! (4) back-substitution: dpsi from rpsi - tdt*(D du) [note eq. (ldu) 4]
    call MatMult(g_mctx%D_op, du, a_t1, ierr)
    call VecAYPX(a_t1, -tdt, rpsi, ierr)
    call pack_2v(a_t1, rj, g_mctx%wv_pair_psij_1, ierr)
    call KSPSolve(g_mctx%ksp_pair_psij, g_mctx%wv_pair_psij_1, &
                  g_mctx%wv_pair_psij_2, ierr)
    call unpack_2v(g_mctx%wv_pair_psij_2, dpsi, dj, ierr)
  end subroutine metriplectic_ps_ldu_solve


  !====================================================================
  ! Sweep core on 1-var components. Mode dispatch (spec Sec. 7.3):
  ! 'PS' pair-Schur LDU / 'K4' coupled LU reference / 'K2', 'PU' the
  ! stage-B/C alternating sweep (both orders).
  !====================================================================
  subroutine metriplectic_sweep_core(rpsi, ru, rj, rw, dpsi, du, dj, dw)
    Vec :: rpsi, ru, rj, rw, dpsi, du, dj, dw
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    ! The A/B diagnostic always exercises the PS-LDU path, whatever the
    ! production khalf_mode is -- that is what lets it run alongside any PC.
    if (cm_ab_active .or. g_mctx%khalf_mode == 'PS') then
      call metriplectic_ps_ldu_solve(rpsi, ru, rj, rw, dpsi, du, dj, dw)
      return
    endif
    if (g_mctx%khalf_mode == 'K4') then
      ! P = A_k4 (+ rho/T recovery): one coupled reference solve
      call pack_4v(rpsi, ru, rj, rw, g_mctx%wv_k4_1, ierr)
      call KSPSolve(g_mctx%ksp_k4, g_mctx%wv_k4_1, g_mctx%wv_k4_2, ierr)
      call unpack_4v(g_mctx%wv_k4_2, dpsi, du, dj, dw, ierr)
      return
    endif
    if (mpc_sweep_order == 'KS') then
      ! P^-1 = A1^-1 M_red A2^-1: ideal first, pairs last (consume rj, rw)
      call metriplectic_ideal_half_solve(rpsi, ru, a_p1, a_u1)
      call metriplectic_mass_apply(a_p1, a_u1, a_yp, a_yu)
      call metriplectic_diss_half_solve(a_yp, a_yu, rj, rw, dpsi, du, dj, dw)
    else
      ! 'SK' (default): P^-1 = A2^-1 M_red A1^-1: pairs first (their j/w
      ! outputs are discarded), ideal last, recovery consumes rj, rw
      call metriplectic_diss_half_solve(rpsi, ru, rj, rw, a_p1, a_u1, a_j1, a_w1)
      call metriplectic_mass_apply(a_p1, a_u1, a_yp, a_yu)
      call metriplectic_ideal_half_solve(a_yp, a_yu, dpsi, du)
      call metriplectic_recover_jw(rj, rw, dpsi, du, dj, dw)
    endif
  end subroutine metriplectic_sweep_core


  !====================================================================
  ! PCSHELL callback on the packed 4-var (psi,u,j,w) vector (T5b/T6)
  !====================================================================
  subroutine metriplectic_sweep_apply_4v(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    call unpack_4v(x, a_rpsi, a_ru, a_rj, a_rw, ierr)
    call metriplectic_sweep_core(a_rpsi, a_ru, a_rj, a_rw, a_dpsi, a_du, a_dj, a_dw)
    call pack_4v(a_dpsi, a_du, a_dj, a_dw, y, ierr)
    ierr = 0
  end subroutine metriplectic_sweep_apply_4v


  !====================================================================
  ! PCSHELL callback on the FULL system vector (production, T5c).
  ! Handles (psi,u,j,w) via the sweep + (rho,T) via recovery; any
  ! further variables pass through as identity (VecCopy first).
  !====================================================================
  subroutine metriplectic_sweep_apply_full(pc, x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    call copy_var_out(x, var_psi, a_rpsi, ierr)
    call copy_var_out(x, var_u,   a_ru,   ierr)
    call copy_var_out(x, var_zj,  a_rj,   ierr)
    call copy_var_out(x, var_w,   a_rw,   ierr)
    call copy_var_out(x, var_rho, a_rrho, ierr)
    call copy_var_out(x, var_T,   a_rT,   ierr)

    call metriplectic_sweep_core(a_rpsi, a_ru, a_rj, a_rw, a_dpsi, a_du, a_dj, a_dw)
    call metriplectic_recover_rhoT(a_rrho, a_rT, a_du, a_drho, a_dT)

    call VecCopy(x, y, ierr)     ! identity pass-through for unhandled vars
    call copy_var_in(y, var_psi, a_dpsi, ierr)
    call copy_var_in(y, var_u,   a_du,   ierr)
    call copy_var_in(y, var_zj,  a_dj,   ierr)
    call copy_var_in(y, var_w,   a_dw,   ierr)
    call copy_var_in(y, var_rho, a_drho, ierr)
    call copy_var_in(y, var_T,   a_dT,   ierr)
    ierr = 0
  end subroutine metriplectic_sweep_apply_full


  !====================================================================
  ! Layout helpers.
  ! Full-system rank-local layout: node-block i, variable v, mode m
  !   -> local index i*(n_var*n_tor) + (v-1)*n_tor + m   (1-based m).
  ! 1-var layout: i*n_tor + m. Ownership is aligned (same rank holds the
  ! corresponding rows of both layouts), so the copy is purely local —
  ! same assumption the pack_4v/nest->AIJ path validated in Slice A.
  !====================================================================
  subroutine copy_var_out(x_full, v, x_sub, ierr)
    use mod_parameters, only: n_var, n_tor
    Vec :: x_full, x_sub
    integer, intent(in) :: v
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ax(:), as(:)
    PetscInt :: n_local
    integer :: i, m, n_block_local, bs

    call VecGetLocalSize(x_full, n_local, ierr)
    bs = n_var * n_tor
    n_block_local = n_local / bs
    call VecGetArrayReadF90(x_full, ax, ierr)
    call VecGetArrayF90(x_sub, as, ierr)
    do i = 0, n_block_local - 1
      do m = 1, n_tor
        as(i*n_tor + m) = ax(i*bs + (v-1)*n_tor + m)
      enddo
    enddo
    call VecRestoreArrayReadF90(x_full, ax, ierr)
    call VecRestoreArrayF90(x_sub, as, ierr)
  end subroutine copy_var_out

  subroutine copy_var_in(y_full, v, y_sub, ierr)
    use mod_parameters, only: n_var, n_tor
    Vec :: y_full, y_sub
    integer, intent(in) :: v
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ay(:), as(:)
    PetscInt :: n_local
    integer :: i, m, n_block_local, bs

    call VecGetLocalSize(y_full, n_local, ierr)
    bs = n_var * n_tor
    n_block_local = n_local / bs
    call VecGetArrayF90(y_full, ay, ierr)
    call VecGetArrayReadF90(y_sub, as, ierr)
    do i = 0, n_block_local - 1
      do m = 1, n_tor
        ay(i*bs + (v-1)*n_tor + m) = as(i*n_tor + m)
      enddo
    enddo
    call VecRestoreArrayF90(y_full, ay, ierr)
    call VecRestoreArrayReadF90(y_sub, as, ierr)
  end subroutine copy_var_in


  !====================================================================
  ! Pack/unpack for nest->AIJ packed vectors (rank-local concatenation;
  ! same convention as the Slice-A analysis helpers).
  !====================================================================
  subroutine pack_2v(x1, x2, y, ierr)
    Vec :: x1, x2, y
    PetscErrorCode :: ierr
    PetscScalar, pointer :: a1(:), a2(:), ay(:)
    PetscInt :: n1, n2

    call VecGetLocalSize(x1, n1, ierr); call VecGetLocalSize(x2, n2, ierr)
    call VecGetArrayReadF90(x1, a1, ierr); call VecGetArrayReadF90(x2, a2, ierr)
    call VecGetArrayF90(y, ay, ierr)
    ay(1:n1)       = a1(1:n1)
    ay(n1+1:n1+n2) = a2(1:n2)
    call VecRestoreArrayReadF90(x1, a1, ierr); call VecRestoreArrayReadF90(x2, a2, ierr)
    call VecRestoreArrayF90(y, ay, ierr)
  end subroutine pack_2v

  subroutine unpack_2v(x, y1, y2, ierr)
    Vec :: x, y1, y2
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ax(:), a1(:), a2(:)
    PetscInt :: n1, n2

    call VecGetLocalSize(y1, n1, ierr); call VecGetLocalSize(y2, n2, ierr)
    call VecGetArrayReadF90(x, ax, ierr)
    call VecGetArrayF90(y1, a1, ierr); call VecGetArrayF90(y2, a2, ierr)
    a1(1:n1) = ax(1:n1)
    a2(1:n2) = ax(n1+1:n1+n2)
    call VecRestoreArrayReadF90(x, ax, ierr)
    call VecRestoreArrayF90(y1, a1, ierr); call VecRestoreArrayF90(y2, a2, ierr)
  end subroutine unpack_2v

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

#endif
end module mod_petsc_pc_metriplectic_apply
