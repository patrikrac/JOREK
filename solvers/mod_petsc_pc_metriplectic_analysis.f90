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
! Stage-D pair-Schur battery (spec Sec. 7.3; runs INSTEAD of T3..T5c when
! metriplectic_khalf == 'PS'; T1/T2 operator certification still run):
!   PS1 : block-LDU (tight inner Schur) vs A_k4 MUMPS LU — sign/layout gate
!   PS2 : Ritz of P_uw^-1 S_uw — the parabolization gate (screening: min
!         positive Ritz rises with eta_num across the namelist arms)
!   PS3 : 4-var Ritz, single-pass 'PS' apply vs 'K4' LU reference apply
!         (separates Schur-approx cost from the additive model gap)
!   PS4 : inner FGMRES(P_uw) iteration counts on S_uw to fixed r/r0
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
        metriplectic_sweep_apply_full, metriplectic_ps_ldu_solve, mpc_sweep_order, &
        pack_2v, unpack_2v
  use mod_elt_matrix_commutator, only: CM_NB, CM_ADV1, CM_ADVR, CM_COMP, CM_S1R, CM_SR
  implicit none
  private

  public :: petsc_metriplectic_run_analysis, petsc_metriplectic_track_energies
  public :: petsc_commutator_run_analysis, petsc_commutator_assemble

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

  ! --- Commutator-analysis assembled building-block operators (Sec. 8.5) ---
  Mat, save     :: cm_blk(CM_NB)  !< pure integrand blocks (ADV1/ADVR/COMP/S1R/SR)
  logical, save :: cm_ops_ready = .false.

contains

  !====================================================================
  ! Assemble the M2/M3 candidate operators (S1r 1/R stiffness, M3
  ! conservative rho-form). Called from jorek2_main (gated by
  ! commutator_analysis) where the element list is available; mirrors
  ! metriplectic_assemble's create/destroy lifecycle. run_CM reads the
  ! module-saved handle array cm_blk once cm_ops_ready is set.
  !====================================================================
  subroutine petsc_commutator_assemble(my_id, local_elms, n_local_elms, a_mat)
    use construct_commutator_matrix_mod, only: commutator_create_matrices, &
                                               construct_commutator_matrices
    use data_structure, only: type_SP_MATRIX

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat

    PetscErrorCode :: ierr
    integer :: ib

    if (cm_ops_ready) then
      do ib = 1, CM_NB
        PetscCallA(MatDestroy(cm_blk(ib), ierr))
      enddo
    endif
    call commutator_create_matrices(a_mat, cm_blk)
    call construct_commutator_matrices(my_id, local_elms, n_local_elms, a_mat, cm_blk)
    ! Convert BAIJ -> AIJ so MatMult interoperates with the AIJ-extracted
    ! A_full sub-blocks / probe vectors used in run_CM (same size layout).
    do ib = 1, CM_NB
      PetscCallA(MatConvert(cm_blk(ib), MATMPIAIJ, MAT_INPLACE_MATRIX, cm_blk(ib), ierr))
    enddo
    cm_ops_ready = .true.
    if (my_id == 0) write(*,'(A,I0,A)') "[Commutator] ", CM_NB, " building-block operators assembled"
  end subroutine petsc_commutator_assemble


  !====================================================================
  ! Commutator-operator (M_*) intertwining-defect analysis
  !   (note: JOREK_commutator_preconditioner_baseline, Eqs. 43/48).
  !
  ! Chacon's commutator device replaces the dense M^-1 inside the Schur
  ! complement by an operator M_* on velocity (flow-potential u) space
  ! obeying U M_* ~ M U.  This routine MEASURES how well candidate M_*
  ! satisfy that intertwining relation, on the stiff (psi,u) sub-block:
  !
  !   D(M_*) du = A_pp Q^-1 A_pu du  -  A_pu Q^-1 A_uM du
  !   eps(M_*)  = ||D du||_{Q^-1} / ||A_pp Q^-1 A_pu du||_{Q^-1}
  !
  ! with  A_pp = amat_11 (=B11, = a I + theta A, inertia + ExB advection),
  !       A_pu = amat_12 (=B12, = theta G, the induction coupling), and
  !       Q    = amat_33 (=B33), the 1/R scalar mass (note Sec. 8.3; psi,
  !       u, j share the Bezier basis so Q_psi = Q_u = Q).  Q^-1 is a FULL
  !       MUMPS factorization -- NO mass lumping (crude on C1 Bezier).
  !
  ! eps is estimated PER TOROIDAL HARMONIC n by a Rademacher probe set
  ! restricted to that harmonic (Hutchinson relative-Frobenius estimate);
  ! the same probes are used for every candidate for a fair comparison.
  !
  ! Candidates are rows of a coefficient TABLE over the operator set
  !   {B11(=A_pp), Q1R(=B33,1/R mass), QR(=B44,R mass), ADV1, ADVR, COMP,
  !    S1R, SR},  applied matrix-free:  A_uM du = sum_iop coef(iop)*op(iop) du.
  ! Adding a candidate is ONE table row -- no reassembly. Default table
  ! (tdt = theta*dt; eta = central resistivity):
  !   M0  = opz*Q1R                       (Q_u=Q1R)  mass only (incumbent)
  !   M1x = B11 (exact A_pp)              (Q_u=Q1R)  flow op, reference (Eq.35)
  !   M0R = opz*QR                        (Q_u=QR)   R-mass only
  !   M1a = opz*Q1R - tdt*ADV1            (Q_u=Q1R)  assembled amat_11 (~M1x)
  !   M3  = opz*QR - tdt*ADVR - tdt*COMP  (Q_u=QR)   conservative rho-form
  !   M2  = M1a + eta*tdt*S1R             (Q_u=Q1R)  + resistive diffusion
  ! M1a/M3/M2 need the assembled building blocks (cm_ops_ready); M0/M1x/M0R
  ! use only extracted A_full blocks. The reference P always uses Q_psi=B33.
  !
  ! EPS_MIN -- fundamental lower bound on the defect over ALL M_*
  ! ------------------------------------------------------------
  !   eps_min = ||(I-P) C||_{Q_psi^-1} / ||C||_{Q_psi^-1},
  !   C = A_pp Q_psi^-1 A_pu S  (S = the probe set, columns = test vectors),
  !   P = the Q_psi^-1-orthogonal projector onto range(A_pu).  Equivalently,
  !   as implemented (column by column, Y unconstrained):
  !       eps_min^2 = min_Y ||C - A_pu Y||^2_{Q_psi^-1} / ||C||^2_{Q_psi^-1}.
  !   Y plays the role of Q_u^-1 A_uM S, so NO structure whatsoever is imposed
  !   on M_*: eps(M_*) >= eps_min for EVERY candidate, local or nonlocal.
  !   eps_min = 0 iff range(A_pp Q^-1 A_pu S) is contained in range(A_pu); it
  !   measures the non-surjectivity of the parallel gradient G = R^2 B.grad,
  !   i.e. the part of M U that U = theta G cannot produce at all.
  !     eps_min ~ eps(M1)   -> M1 is essentially optimal; stop searching forms
  !     eps_min << eps(M1)  -> headroom exists; a better M_* is worth hunting
  !   NORM: Q_psi^-1, the SAME norm as eps(M_*) above (not the Q_psi norm of
  !   the abstract statement) -- that is what makes eps_min <= eps(M_*) an
  !   exact checkable identity instead of a norm-dependent near-miss.  The
  !   per-probe LS is warm-started from the best candidate's y, and CG
  !   minimises exactly that LS objective, so the bound holds by construction
  !   (not by luck) even if the CG is stopped early.
  !   READING IT: an under-converged CG can only OVERstate the residual, so the
  !   printed eps_min is an UPPER bound on the true lower bound.  Hence
  !   "eps_min << eps(M1) -> headroom" is a rigorous verdict at any iteration
  !   count, whereas "eps_min ~ eps(M1) -> M1 optimal" is only trustworthy if
  !   the reported mean CG its/probe sits well below CM_LS_MAXIT.
  !
  ! Reads A_full sub-blocks + module-saved cm_blk (no metriplectic_assemble).
  ! NOTE on the zero-flow check: at u0=0, B11 = (1+zeta) Q holds in the
  ! interior, so eps(M1)=0 there to machine precision; boundary rows of
  ! B11 (psi Dirichlet) and B33 (j aux) differ, leaving a small boundary
  ! residual.  For an EXACT zero-flow check, use g_mctx%M_psi (the
  ! psi-row mass) as Q instead -- at the cost of requiring the
  ! metriplectic assembly to have run.
  !====================================================================
  subroutine petsc_commutator_run_analysis(A_full, my_id)
    use mod_parameters, only: n_tor, var_psi, var_u, var_zj, var_w
    use phys_module,    only: mode, time_evol_theta, time_evol_zeta, tstep, tstep_prev, eta

    Mat,     intent(in) :: A_full
    integer, intent(in) :: my_id

    integer, parameter :: CM_NPROBE = 32          ! Hutchinson probes per harmonic
    integer, parameter :: NOP    = 3 + CM_NB      ! op set: B11,Q1R,QR + blocks
    integer, parameter :: OP_B11 = 1, OP_Q1R = 2, OP_QR = 3
    integer, parameter :: MAXC   = 16             ! candidate-table capacity
    integer, parameter :: CM_LS_MAXIT = 300       ! CG cap for the eps_min LS
    real*8,  parameter :: CM_LS_RTOL  = 1.d-8     ! relative rz drop for that CG

    Mat :: B11, B12, Qpsi, QuR, op(NOP)
    KSP :: ksp_Qpsi, ksp_QuR, ksp_Qu
    Vec :: z, va, vb, vPz, vc, vt, vd, vRz, vDz, vg
    Vec :: vy, wd, ws, wq, wr, wzu, wp, wtu       ! eps_min least-squares work
    PetscErrorCode :: ierr
    integer :: comm, h, kp, ic, iop, ib, ncand
    integer :: nit, nit_tot, nit_max
    real*8  :: opz, zeta, tdt, nd, npp, sumP2, eps_val
    real*8  :: sumE2, ndbest, fls, eps_min, eps_best
    real*8  :: coef(MAXC, NOP)
    integer :: quop(MAXC)
    character(len=4) :: lab(MAXC)
    real*8, allocatable :: sumD2(:)

    call PetscObjectGetComm(A_full, comm, ierr)
    zeta = time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    opz  = 1.d0 + zeta
    tdt  = time_evol_theta * tstep

    ! --- Index sets + sub-blocks: A_pp, A_pu, 1/R mass (B33), R mass (B44) ---
    if (.not. g_mctx%is_created) call create_index_sets(A_full, comm)
    call extract_block(A_full, var_psi, var_psi, B11)
    call extract_block(A_full, var_psi, var_u,   B12)
    call extract_block(A_full, var_zj,  var_zj,  Qpsi)
    call extract_block(A_full, var_w,   var_w,   QuR)

    ! --- Full mass factorizations (MUMPS LU; no lumping) ---
    call setup_lu_ksp(Qpsi, ksp_Qpsi, comm, .false.)
    call setup_lu_ksp(QuR,  ksp_QuR,  comm, .false.)

    ! --- Operator handle table op(1..NOP) ---
    op(OP_B11) = B11
    op(OP_Q1R) = Qpsi
    op(OP_QR)  = QuR
    if (cm_ops_ready) then
      do ib = 1, CM_NB
        op(3+ib) = cm_blk(ib)
      enddo
    endif

    ! ============ CANDIDATE TABLE (edit rows here to try candidates) ======
    coef = 0.d0
    ncand = 0
    ncand=ncand+1; lab(ncand)="M0" ; quop(ncand)=OP_Q1R; coef(ncand,OP_Q1R)=opz
    ncand=ncand+1; lab(ncand)="M1x"; quop(ncand)=OP_Q1R; coef(ncand,OP_B11)=1.d0
    ncand=ncand+1; lab(ncand)="M0R"; quop(ncand)=OP_QR ; coef(ncand,OP_QR )=opz
    if (cm_ops_ready) then
      ncand=ncand+1; lab(ncand)="M1a"; quop(ncand)=OP_Q1R
        coef(ncand,OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt
      ncand=ncand+1; lab(ncand)="M3" ; quop(ncand)=OP_QR
        coef(ncand,OP_QR)=opz;  coef(ncand,3+CM_ADVR)=-tdt; coef(ncand,3+CM_COMP)=-tdt
      ncand=ncand+1; lab(ncand)="M2" ; quop(ncand)=OP_Q1R
        coef(ncand,OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt; coef(ncand,3+CM_S1R)=eta*tdt
    endif
    ! ======================================================================

    allocate(sumD2(ncand))

    if (my_id == 0) then
      write(*,'(A)') "[Commutator] ========= M_* intertwining-defect analysis ========="
      write(*,'(A,I0,A,F8.4,A,I0)') "[Commutator] probes/harmonic = ", CM_NPROBE, &
                               ",  (1+zeta) = ", opz, ",  candidates = ", ncand
      if (.not. cm_ops_ready) write(*,'(A)') &
        "[Commutator] NOTE: building blocks not assembled -> extracted-only candidates"
      write(*,'(A)') &
        "[Commutator] eps(n) = ||D du||_{Q_psi^-1} / ||A_pp Q_psi^-1 A_pu du||_{Q_psi^-1}"
      write(*,'(A,I0,A,ES8.1,A)') &
        "[Commutator] EPSMIN = lower bound over ALL M_* (unconstrained LS onto " // &
        "range(A_pu); CG maxit = ", CM_LS_MAXIT, ", rtol = ", CM_LS_RTOL, ")"
    endif

    ! --- Work vectors (all size n_var_dofs) ---
    PetscCallA(MatCreateVecs(B12, z, va, ierr))
    PetscCallA(VecDuplicate(va, vb,  ierr))
    PetscCallA(VecDuplicate(va, vPz, ierr))
    PetscCallA(VecDuplicate(va, vc,  ierr))
    PetscCallA(VecDuplicate(va, vt,  ierr))
    PetscCallA(VecDuplicate(va, vd,  ierr))
    PetscCallA(VecDuplicate(va, vRz, ierr))
    PetscCallA(VecDuplicate(va, vDz, ierr))
    PetscCallA(VecDuplicate(va, vg,  ierr))
    PetscCallA(VecDuplicate(va, vy,  ierr));  PetscCallA(VecDuplicate(va, wd,  ierr))
    PetscCallA(VecDuplicate(va, ws,  ierr));  PetscCallA(VecDuplicate(va, wq,  ierr))
    PetscCallA(VecDuplicate(va, wr,  ierr));  PetscCallA(VecDuplicate(va, wzu, ierr))
    PetscCallA(VecDuplicate(va, wp,  ierr));  PetscCallA(VecDuplicate(va, wtu, ierr))
    PetscCallA(VecSet(vy, 0.d0, ierr))        ! defined even if no candidate wins

    ! --- eps(n) per toroidal harmonic; shared probes across candidates ---
    do h = 1, n_tor
      sumP2   = 0.d0
      sumD2   = 0.d0
      sumE2   = 0.d0
      nit_tot = 0
      nit_max = 0
      do kp = 1, CM_NPROBE
        call cm_probe_fill(z, h, n_tor)
        ! P du = A_pp Q_psi^-1 (A_pu du)   [candidate-independent, shared]
        PetscCallA(MatMult(B12, z, va, ierr))
        PetscCallA(KSPSolve(ksp_Qpsi, va, vb, ierr))
        PetscCallA(MatMult(B11, vb, vPz, ierr))
        PetscCallA(KSPSolve(ksp_Qpsi, vPz, vg, ierr))
        PetscCallA(VecDot(vPz, vg, npp, ierr))
        sumP2  = sumP2 + npp
        ndbest = huge(1.d0)
        do ic = 1, ncand
          ! A_uM du = sum_iop coef(ic,iop) * op(iop) du   (matrix-free)
          PetscCallA(VecSet(vc, 0.d0, ierr))
          do iop = 1, NOP
            if (coef(ic,iop) /= 0.d0) then
              PetscCallA(MatMult(op(iop), z, vt, ierr))
              PetscCallA(VecAXPY(vc, coef(ic,iop), vt, ierr))
            endif
          enddo
          ksp_Qu = ksp_Qpsi
          if (quop(ic) == OP_QR) ksp_Qu = ksp_QuR
          ! R du = A_pu Q_u^-1 (A_uM du) ; D = P - R ; ||D||^2_{Q_psi^-1}
          PetscCallA(KSPSolve(ksp_Qu, vc, vd, ierr))
          PetscCallA(MatMult(B12, vd, vRz, ierr))
          PetscCallA(VecCopy(vPz, vDz, ierr))
          PetscCallA(VecAXPY(vDz, -1.d0, vRz, ierr))
          PetscCallA(KSPSolve(ksp_Qpsi, vDz, vg, ierr))
          PetscCallA(VecDot(vDz, vg, nd, ierr))
          sumD2(ic) = sumD2(ic) + nd
          ! vd IS the y of the LS parametrisation D = P du - A_pu y; keep the
          ! best candidate's as the warm start so eps_min <= min_ic eps(ic)
          ! holds per probe by construction.
          if (nd < ndbest) then
            ndbest = nd
            PetscCallA(VecCopy(vd, vy, ierr))
          endif
        enddo
        ! --- eps_min contribution: min over ALL y of ||P du - A_pu y||^2 ---
        call cm_ls_min(B12, ksp_Qpsi, ksp_Qpsi, vPz, vy, CM_LS_MAXIT, CM_LS_RTOL, &
                       wd, ws, wq, wr, wzu, wp, wtu, fls, nit)
        sumE2   = sumE2 + fls
        nit_tot = nit_tot + nit
        nit_max = max(nit_max, nit)
      enddo
      if (my_id == 0) then
        eps_best = huge(1.d0)
        do ic = 1, ncand
          eps_val = sqrt(max(sumD2(ic), 0.d0) / max(sumP2, tiny(1.d0)))
          eps_best = min(eps_best, eps_val)
          write(*,'(A,A,A,I4,A,ES13.5)') "[Commutator] ", trim(lab(ic)), &
            ":   n = ", mode(h), "   eps = ", eps_val
        enddo
        eps_min = sqrt(max(sumE2, 0.d0) / max(sumP2, tiny(1.d0)))
        write(*,'(A,I4,A,ES13.5,A,F7.3)') "[Commutator] EPSMIN:   n = ", mode(h), &
          "   eps = ", eps_min, "   eps_min/eps_best = ", &
          eps_min / max(eps_best, tiny(1.d0))
        write(*,'(A,I0,A,I0)') "[Commutator]           LS CG its: mean/probe = ", &
          nint(dble(nit_tot) / dble(CM_NPROBE)), " ,  max = ", nit_max
      endif
    enddo

    ! --- Cleanup (cm_blk are module-saved; not destroyed here) ---
    PetscCallA(VecDestroy(z,   ierr));  PetscCallA(VecDestroy(va,  ierr))
    PetscCallA(VecDestroy(vb,  ierr));  PetscCallA(VecDestroy(vPz, ierr))
    PetscCallA(VecDestroy(vc,  ierr));  PetscCallA(VecDestroy(vt,  ierr))
    PetscCallA(VecDestroy(vd,  ierr));  PetscCallA(VecDestroy(vRz, ierr))
    PetscCallA(VecDestroy(vDz, ierr));  PetscCallA(VecDestroy(vg,  ierr))
    PetscCallA(VecDestroy(vy,  ierr));  PetscCallA(VecDestroy(wd,  ierr))
    PetscCallA(VecDestroy(ws,  ierr));  PetscCallA(VecDestroy(wq,  ierr))
    PetscCallA(VecDestroy(wr,  ierr));  PetscCallA(VecDestroy(wzu, ierr))
    PetscCallA(VecDestroy(wp,  ierr));  PetscCallA(VecDestroy(wtu, ierr))
    PetscCallA(KSPDestroy(ksp_Qpsi, ierr))
    PetscCallA(KSPDestroy(ksp_QuR,  ierr))
    PetscCallA(MatDestroy(B11,  ierr))
    PetscCallA(MatDestroy(B12,  ierr))
    PetscCallA(MatDestroy(Qpsi, ierr))
    PetscCallA(MatDestroy(QuR,  ierr))
    deallocate(sumD2)
    if (my_id == 0) write(*,'(A)') "[Commutator] ================ analysis complete ================"
  end subroutine petsc_commutator_run_analysis


  !--------------------------------------------------------------------
  !> One column of the eps_min least-squares problem:
  !!     f = min_y || c - A_pu y ||^2_{Qp^-1},   y UNCONSTRAINED.
  !!
  !! Solved by mass-preconditioned CG on the normal equations
  !!     N y = b,   N = A_pu^T Qp^-1 A_pu,   b = A_pu^T Qp^-1 c,
  !! preconditioned by the u-space mass (ksp_Qu).  CG minimises exactly the
  !! LS objective f(y) over the Krylov space, so with y entering as the best
  !! candidate's y (warm start) the returned f is monotonically <= that
  !! candidate's defect -- this is what makes eps_min <= eps(M_*) hold by
  !! construction rather than by luck, even under an early CG stop.  Chosen
  !! over LSQR/LSMR for exactly that warm-start property; the squared
  !! conditioning is harmless here because we only need ~3 digits of a
  !! residual NORM, and f is re-measured honestly at exit (below).
  !!
  !! N is singular -- null(N) = null(A_pu) = the field-line-constant modes
  !! that G = R^2 B.grad annihilates -- but the system is consistent, since
  !! b lies in range(A_pu^T) by construction.  Null-space growth in y is
  !! harmless for the objective (A_pu kills it), and the final f is recomputed
  !! from d = c - A_pu y rather than from a recursion, so no drift enters the
  !! measurement.  The warm-start value is kept if CG failed to improve on it.
  !!
  !! Work vectors come from the caller (psi-space wd/ws/wq, u-space wr/wz/wp/wt)
  !! to avoid create/destroy churn inside the probe loop.  Costs 2 mass
  !! backsolves + 1 MatMult + 1 MatMultTranspose per iteration.
  !--------------------------------------------------------------------
  subroutine cm_ls_min(A_pu, ksp_Qp, ksp_Qu, c, y, maxit, rtol, &
                       wd, ws, wq, wr, wz, wp, wt, f, nit)
    Mat,     intent(in)    :: A_pu
    KSP,     intent(in)    :: ksp_Qp, ksp_Qu
    Vec,     intent(in)    :: c
    Vec,     intent(inout) :: y
    integer, intent(in)    :: maxit
    real*8,  intent(in)    :: rtol
    Vec,     intent(inout) :: wd, ws, wq, wr, wz, wp, wt
    real*8,  intent(out)   :: f
    integer, intent(out)   :: nit

    PetscErrorCode :: ierr
    integer :: it
    real*8  :: f0, rz, rz0, rznew, pNp, alpha, beta

    ! d = c - A_pu y ;  s = Qp^-1 d ;  f0 = ||d||^2_{Qp^-1}  (warm-start value)
    PetscCallA(MatMult(A_pu, y, wd, ierr))
    PetscCallA(VecAYPX(wd, -1.d0, c, ierr))
    PetscCallA(KSPSolve(ksp_Qp, wd, ws, ierr))
    PetscCallA(VecDot(wd, ws, f0, ierr))

    ! r = A_pu^T Qp^-1 d ;  z = Qu^-1 r ;  p = z
    PetscCallA(MatMultTranspose(A_pu, ws, wr, ierr))
    PetscCallA(KSPSolve(ksp_Qu, wr, wz, ierr))
    PetscCallA(VecCopy(wz, wp, ierr))
    PetscCallA(VecDot(wr, wz, rz, ierr))
    rz0 = rz
    nit = 0

    do it = 1, maxit
      if (rz <= 0.d0 .or. rz <= rtol*rtol*rz0) exit
      PetscCallA(MatMult(A_pu, wp, wq, ierr))
      PetscCallA(KSPSolve(ksp_Qp, wq, ws, ierr))
      PetscCallA(VecDot(wq, ws, pNp, ierr))
      if (pNp <= 0.d0) exit                     ! p hit null(A_pu)
      alpha = rz / pNp
      PetscCallA(VecAXPY(y, alpha, wp, ierr))
      PetscCallA(MatMultTranspose(A_pu, ws, wt, ierr))
      PetscCallA(VecAXPY(wr, -alpha, wt, ierr))
      PetscCallA(KSPSolve(ksp_Qu, wr, wz, ierr))
      PetscCallA(VecDot(wr, wz, rznew, ierr))
      beta = rznew / rz
      PetscCallA(VecAYPX(wp, beta, wz, ierr))
      rz  = rznew
      nit = it
    enddo

    ! Honest re-measurement of the objective at the final y.
    PetscCallA(MatMult(A_pu, y, wd, ierr))
    PetscCallA(VecAYPX(wd, -1.d0, c, ierr))
    PetscCallA(KSPSolve(ksp_Qp, wd, ws, ierr))
    PetscCallA(VecDot(wd, ws, f, ierr))
    if (.not. (f <= f0)) f = f0                 ! also traps NaN
  end subroutine cm_ls_min


  !--------------------------------------------------------------------
  !> M4 coefficient fit: least-squares over probes of the basis
  !! {Qpsi, B11-opz*Qpsi, S1r} minimizing ||P - sum_k d_k g_k||_{Qpsi^-1},
  !! g_k = A_pu Qpsi^-1 (basis_k du), P = A_pp Qpsi^-1 A_pu du.
  !! Returns d = c4(0:2) via 3x3 normal equations (symmetric, small ridge).
  !--------------------------------------------------------------------
  subroutine cm_fit_m4(B11, B12, Qpsi, S1r, ksp_Qpsi, opz, n_tor, nprobe, my_id, c4)
    Mat, intent(in) :: B11, B12, Qpsi, S1r
    KSP, intent(in) :: ksp_Qpsi
    real*8,  intent(in)  :: opz
    integer, intent(in)  :: n_tor, nprobe, my_id
    real*8,  intent(out) :: c4(0:2)

    Vec :: z, t1, t2, g0, g1, g2, pP, s0, s1, s2, sp
    PetscErrorCode :: ierr
    integer :: h, kp, a, b
    real*8  :: Gram(0:2,0:2), rhs(0:2), val, gmax, det
    real*8  :: Gn(0:2,0:2), rn(0:2), dn(0:2), nrm(0:2)

    PetscCallA(MatCreateVecs(B12, z, t1, ierr))
    PetscCallA(VecDuplicate(t1, t2, ierr))
    PetscCallA(VecDuplicate(t1, g0, ierr))
    PetscCallA(VecDuplicate(t1, g1, ierr))
    PetscCallA(VecDuplicate(t1, g2, ierr))
    PetscCallA(VecDuplicate(t1, pP, ierr))
    PetscCallA(VecDuplicate(t1, s0, ierr))
    PetscCallA(VecDuplicate(t1, s1, ierr))
    PetscCallA(VecDuplicate(t1, s2, ierr))
    PetscCallA(VecDuplicate(t1, sp, ierr))

    Gram = 0.d0
    rhs  = 0.d0
    do h = 1, n_tor
      do kp = 1, nprobe
        call cm_probe_fill(z, h, n_tor)
        ! g0 = A_pu du
        PetscCallA(MatMult(B12, z, g0, ierr))
        ! g1 = A_pu Qpsi^-1 (B11 du) - opz g0
        PetscCallA(MatMult(B11, z, t1, ierr))
        PetscCallA(KSPSolve(ksp_Qpsi, t1, t2, ierr))
        PetscCallA(MatMult(B12, t2, g1, ierr))
        PetscCallA(VecAXPY(g1, -opz, g0, ierr))
        ! g2 = A_pu Qpsi^-1 (S1r du)
        PetscCallA(MatMult(S1r, z, t1, ierr))
        PetscCallA(KSPSolve(ksp_Qpsi, t1, t2, ierr))
        PetscCallA(MatMult(B12, t2, g2, ierr))
        ! P = A_pp Qpsi^-1 (A_pu du) = B11 Qpsi^-1 g0
        PetscCallA(KSPSolve(ksp_Qpsi, g0, t2, ierr))
        PetscCallA(MatMult(B11, t2, pP, ierr))
        ! Qpsi^-1 g_k and Qpsi^-1 P
        PetscCallA(KSPSolve(ksp_Qpsi, g0, s0, ierr))
        PetscCallA(KSPSolve(ksp_Qpsi, g1, s1, ierr))
        PetscCallA(KSPSolve(ksp_Qpsi, g2, s2, ierr))
        PetscCallA(KSPSolve(ksp_Qpsi, pP, sp, ierr))
        ! Gram(a,b) = <g_a, Qpsi^-1 g_b> ; rhs(a) = <g_a, Qpsi^-1 P>
        PetscCallA(VecDot(g0, s0, val, ierr)); Gram(0,0) = Gram(0,0) + val
        PetscCallA(VecDot(g0, s1, val, ierr)); Gram(0,1) = Gram(0,1) + val
        PetscCallA(VecDot(g0, s2, val, ierr)); Gram(0,2) = Gram(0,2) + val
        PetscCallA(VecDot(g1, s1, val, ierr)); Gram(1,1) = Gram(1,1) + val
        PetscCallA(VecDot(g1, s2, val, ierr)); Gram(1,2) = Gram(1,2) + val
        PetscCallA(VecDot(g2, s2, val, ierr)); Gram(2,2) = Gram(2,2) + val
        PetscCallA(VecDot(g0, sp, val, ierr)); rhs(0) = rhs(0) + val
        PetscCallA(VecDot(g1, sp, val, ierr)); rhs(1) = rhs(1) + val
        PetscCallA(VecDot(g2, sp, val, ierr)); rhs(2) = rhs(2) + val
      enddo
    enddo

    ! Symmetrize.
    Gram(1,0) = Gram(0,1); Gram(2,0) = Gram(0,2); Gram(2,1) = Gram(1,2)
    ! Column-scale by ||g_k||_{Qpsi^-1} = sqrt(Gram(k,k)) before solving:
    ! the basis operators span ~1/h^2 in magnitude (mass vs stiffness S1r),
    ! so the raw normal equations are catastrophically ill-conditioned. A
    ! degenerate column (e.g. the B11-opz*Qpsi advection basis at near-zero
    ! flow) is dropped (coefficient 0). d solves the normalized system;
    ! c_k = d_k / nrm(k).
    do a = 0, 2
      nrm(a) = sqrt(max(Gram(a,a), 0.d0))
    enddo
    gmax = max(nrm(0), max(nrm(1), nrm(2)))
    do a = 0, 2
      if (nrm(a) < 1.d-13 * max(gmax, tiny(1.d0))) nrm(a) = 0.d0
    enddo
    do a = 0, 2
      do b = 0, 2
        if (nrm(a) > 0.d0 .and. nrm(b) > 0.d0) then
          Gn(a,b) = Gram(a,b) / (nrm(a)*nrm(b))
        else
          Gn(a,b) = merge(1.d0, 0.d0, a == b)
        endif
      enddo
      if (nrm(a) > 0.d0) then
        rn(a) = rhs(a) / nrm(a)
      else
        rn(a) = 0.d0
      endif
    enddo
    do a = 0, 2
      Gn(a,a) = Gn(a,a) + 1.d-12       ! ridge on the O(1) normalized Gram
    enddo
    call cm_solve3(Gn, rn, dn, det)
    do a = 0, 2
      if (nrm(a) > 0.d0) then
        c4(a) = dn(a) / nrm(a)
      else
        c4(a) = 0.d0
      endif
    enddo
    if (my_id == 0) write(*,'(A,ES12.4)') "[Commutator] M4 fit: normalized Gram det = ", det

    PetscCallA(VecDestroy(z,  ierr));  PetscCallA(VecDestroy(t1, ierr))
    PetscCallA(VecDestroy(t2, ierr));  PetscCallA(VecDestroy(g0, ierr))
    PetscCallA(VecDestroy(g1, ierr));  PetscCallA(VecDestroy(g2, ierr))
    PetscCallA(VecDestroy(pP, ierr));  PetscCallA(VecDestroy(s0, ierr))
    PetscCallA(VecDestroy(s1, ierr));  PetscCallA(VecDestroy(s2, ierr))
    PetscCallA(VecDestroy(sp, ierr))
  end subroutine cm_fit_m4


  !> Solve the 3x3 system A x = b by cofactor expansion (adjugate/det).
  subroutine cm_solve3(A, b, x, det)
    real*8, intent(in)  :: A(0:2,0:2), b(0:2)
    real*8, intent(out) :: x(0:2), det
    real*8 :: c00,c01,c02,c10,c11,c12,c20,c21,c22

    c00 =  (A(1,1)*A(2,2) - A(1,2)*A(2,1))
    c01 = -(A(1,0)*A(2,2) - A(1,2)*A(2,0))
    c02 =  (A(1,0)*A(2,1) - A(1,1)*A(2,0))
    det = A(0,0)*c00 + A(0,1)*c01 + A(0,2)*c02
    if (abs(det) < tiny(1.d0)) then
      x = 0.d0
      return
    endif
    c10 = -(A(0,1)*A(2,2) - A(0,2)*A(2,1))
    c11 =  (A(0,0)*A(2,2) - A(0,2)*A(2,0))
    c12 = -(A(0,0)*A(2,1) - A(0,1)*A(2,0))
    c20 =  (A(0,1)*A(1,2) - A(0,2)*A(1,1))
    c21 = -(A(0,0)*A(1,2) - A(0,2)*A(1,0))
    c22 =  (A(0,0)*A(1,1) - A(0,1)*A(1,0))
    x(0) = (c00*b(0) + c10*b(1) + c20*b(2)) / det
    x(1) = (c01*b(0) + c11*b(1) + c21*b(2)) / det
    x(2) = (c02*b(0) + c12*b(1) + c22*b(2)) / det
  end subroutine cm_solve3


  !> Fill z with Rademacher (+/-1) entries on toroidal harmonic h only
  !! (1-var layout: local index i*n_tor + (h-1)); zero elsewhere.
  subroutine cm_probe_fill(z, h, n_tor)
    Vec, intent(inout) :: z
    integer, intent(in) :: h, n_tor
    PetscScalar, pointer :: arr(:)
    PetscInt :: lo, hi, nloc, n_block_local
    PetscErrorCode :: ierr
    integer :: i
    real*8  :: r

    PetscCallA(VecSet(z, 0.d0, ierr))
    PetscCallA(VecGetOwnershipRange(z, lo, hi, ierr))
    nloc = hi - lo
    n_block_local = nloc / n_tor
    call VecGetArrayF90(z, arr, ierr)
    do i = 0, n_block_local - 1
      call random_number(r)
      if (r >= 0.5d0) then
        arr(i*n_tor + (h-1) + 1) =  1.d0
      else
        arr(i*n_tor + (h-1) + 1) = -1.d0
      endif
    enddo
    call VecRestoreArrayF90(z, arr, ierr)
  end subroutine cm_probe_fill


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

    if (g_mctx%khalf_mode == 'PS') then
      ! Stage-D pair-Schur battery (spec Sec. 7.3, note Sec. pschecks).
      ! The T-battery is retired from active PS runs; set metriplectic_khalf
      ! to 'K2'/'PU'/'K4' to exercise it for regression.
      call run_PS1(comm, my_id)
      call run_PS2(comm, my_id)
      call run_PS3(comm, my_id)
      call run_PS4(comm, my_id)
    else
      call setup_pu_solver(m_tau, comm)   ! analysis-local P_u factor (T3c shell PC)
      call run_T3(comm, my_id)
      call run_T3b(comm, my_id)
      call run_T3c(comm, my_id)
      call run_T6(comm, my_id)
      call run_T4(comm, my_id)
      call run_T5a(comm, my_id)
      call run_T5b(comm, my_id)
      call run_T5c(comm, my_id, A_full)
    endif

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
  ! PS1: sign/layout gate (spec Sec. 7.3, note Prop. ldu). Two exact
  ! checks that do NOT depend on the inner Schur solve converging (unlike
  ! comparing the whole LDU to A_k4^-1, which needs a tight inner solve
  ! and is contaminated by the P_uw tail at eta_num = 0):
  !
  !  PS1a  static-condensation identity. For random v on (u,w), build
  !        Z = ( -A_psij^-1 (tdt D v_u, 0) ; v ). Then EXACTLY
  !          (A_k4 Z)_(psi,j) = 0                  [pivot + C coupling]
  !          (A_k4 Z)_(u,w)   = S_uw v             [Chat coupling + shell]
  !        Validates the (psi,j) pivot, both wave couplings tdt*D / tdt*Dp,
  !        and that the S_uw shell is the true Schur of A_k4.
  !
  !  PS1b  end-to-end LDU residual split. res = b - A_k4 * LDU(b). The
  !        (psi,j) rows are satisfied by steps 1&4 for ANY du, so
  !          |res_(psi,j)|/|b|  is the WIRING gate (expect roundoff;
  !            an O(1) value = an r_j routing or a coupling-sign error),
  !          |res_(u,w)|/|b|    is the inner-solve residual (single-pass at
  !            default ps_inner_it; cross-reference PS4, not a sign gate).
  !====================================================================
  subroutine run_PS1(comm, my_id)
    integer, intent(in) :: comm, my_id

    Vec :: v_uw, cv_psij, w_psij, z4, y4, y_uw, s_uw_v
    Vec :: b4, x4, res4
    PetscRandom :: rnd
    PetscReal :: tdt, ncv, r_cond, r_shell, nb, ns, r_wire, r_inner
    PetscErrorCode :: ierr
    integer :: k

    tdt = g_mctx%dt_theta * m_opz
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))

    ! ---- PS1a: static-condensation exactness (no iterative solve) ----
    PetscCallA(MatCreateVecs(g_mctx%A_pair_uw,   v_uw,   y_uw,   ierr))
    PetscCallA(VecDuplicate(v_uw, s_uw_v, ierr))
    PetscCallA(MatCreateVecs(g_mctx%A_pair_psij, cv_psij, w_psij, ierr))
    PetscCallA(MatCreateVecs(g_mctx%A_k4, z4, y4, ierr))

    PetscCallA(VecSetRandom(v_uw, rnd, ierr))
    call unpack_2v(v_uw, m_du, m_dw, ierr)              ! v = (v_u, v_w)
    call MatMult(g_mctx%D_op, m_du, m_t1, ierr)         ! D v_u
    call VecScale(m_t1, tdt, ierr)                      ! tdt D v_u = (C v)_psi
    call VecZeroEntries(m_t2, ierr)
    call pack_2v(m_t1, m_t2, cv_psij, ierr)            ! C v = (tdt D v_u, 0)
    PetscCallA(VecNorm(cv_psij, NORM_2, ncv, ierr))
    call KSPSolve(g_mctx%ksp_pair_psij, cv_psij, w_psij, ierr)  ! A_psij^-1 C v

    call unpack_2v(w_psij, m_dpsi, m_dj, ierr)
    call VecScale(m_dpsi, -1.d0, ierr)                  ! Z_psi = -(A_psij^-1 C v)_psi
    call VecScale(m_dj,   -1.d0, ierr)
    call pack_4v(m_dpsi, m_du, m_dj, m_dw, z4, ierr)    ! Z = (Z_psi, v_u, Z_j, v_w)
    call MatMult(g_mctx%A_k4, z4, y4, ierr)
    call unpack_4v(y4, m_rpsi, m_ru, m_rj, m_rw, ierr)

    ! (psi,j) block should vanish; (u,w) block should equal S_uw v
    call pack_2v(m_rpsi, m_rj, cv_psij, ierr)           ! reuse cv_psij as (y_psi,y_j)
    PetscCallA(VecNorm(cv_psij, NORM_2, r_cond, ierr))
    call MatMult(g_mctx%S_uw_shell, v_uw, s_uw_v, ierr)
    call pack_2v(m_ru, m_rw, y_uw, ierr)
    call VecAXPY(y_uw, -1.d0, s_uw_v, ierr)
    PetscCallA(VecNorm(y_uw,   NORM_2, r_shell, ierr))
    PetscCallA(VecNorm(s_uw_v, NORM_2, ns, ierr))

    if (my_id == 0) then
      write(*,'(A)') "[Metriplectic] PS1a: static-condensation exactness (no inner solve)"
      write(*,'(A,ES14.4)') "[Metriplectic] PS1a:  |(A_k4 Z)_(psi,j)| / |C v|      = ", &
        r_cond/max(ncv,tiny(1.d0))
      write(*,'(A,ES14.4)') "[Metriplectic] PS1a:  |(A_k4 Z)_(u,w) - S_uw v| / |S_uw v| = ", &
        r_shell/max(ns,tiny(1.d0))
    endif

    PetscCallA(VecDestroy(v_uw, ierr));    PetscCallA(VecDestroy(y_uw, ierr))
    PetscCallA(VecDestroy(s_uw_v, ierr));  PetscCallA(VecDestroy(cv_psij, ierr))
    PetscCallA(VecDestroy(w_psij, ierr));  PetscCallA(VecDestroy(z4, ierr))
    PetscCallA(VecDestroy(y4, ierr))

    ! ---- PS1b: end-to-end LDU residual split (wiring vs inner solve) ----
    PetscCallA(MatCreateVecs(g_mctx%A_k4, x4, b4, ierr))
    PetscCallA(VecDuplicate(b4, res4, ierr))
    if (my_id == 0) write(*,'(A)') &
      "[Metriplectic] PS1b: LDU residual split, |res_(psi,j)|/|b| (WIRING) , |res_(u,w)|/|b| (inner)"
    do k = 1, 3
      PetscCallA(VecSetRandom(b4, rnd, ierr))
      call unpack_4v(b4, m_rpsi, m_ru, m_rj, m_rw, ierr)
      call metriplectic_ps_ldu_solve(m_rpsi, m_ru, m_rj, m_rw, &
                                     m_dpsi, m_du, m_dj, m_dw)
      call pack_4v(m_dpsi, m_du, m_dj, m_dw, x4, ierr)
      call MatMult(g_mctx%A_k4, x4, res4, ierr)
      call VecAYPX(res4, -1.d0, b4, ierr)               ! res = b - A_k4 x
      PetscCallA(VecNorm(b4, NORM_2, nb, ierr))

      call unpack_4v(res4, m_rpsi, m_ru, m_rj, m_rw, ierr)
      call block_norm2(m_rpsi, m_rj, r_wire, ierr)
      call block_norm2(m_ru,   m_rw, r_inner, ierr)
      if (my_id == 0) write(*,'(A,I2,A,2ES14.4)') &
        "[Metriplectic] PS1b:  RHS ", k, "  = ", &
        r_wire/max(nb,tiny(1.d0)), r_inner/max(nb,tiny(1.d0))
    enddo

    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(VecDestroy(b4, ierr))
    PetscCallA(VecDestroy(x4, ierr))
    PetscCallA(VecDestroy(res4, ierr))
  end subroutine run_PS1


  !> 2-norm of the concatenation of two 1-var vecs: sqrt(|a|^2 + |b|^2).
  subroutine block_norm2(a, b, nrm, ierr)
    Vec :: a, b
    PetscReal :: nrm
    PetscErrorCode :: ierr
    PetscReal :: na, nb
    call VecNorm(a, NORM_2, na, ierr)
    call VecNorm(b, NORM_2, nb, ierr)
    nrm = sqrt(na*na + nb*nb)
  end subroutine block_norm2


  !====================================================================
  ! PS2: the parabolization gate (spec Sec. 7.3, note Sec. pschecks).
  ! Ritz values of P_uw^-1 S_uw (matrix-free exact Schur S_uw vs the
  ! sparse P_uw). Generalizes T3c to the screened pair setting. At
  ! eta_num = 0: the T3c picture in pair form (cluster at 1, tail -> 0);
  ! the SCREENING claim is that the min positive Ritz RISES with eta_num
  ! (compared across the eta_num namelist arms). If it does not rise, the
  ! two-level correction moves to the critical path.
  !====================================================================
  subroutine run_PS2(comm, my_id)
    integer, intent(in) :: comm, my_id

    KSP :: ksp
    PC  :: pc
    Vec :: b, x
    PetscRandom :: rnd
    PetscReal :: r_eig(60), c_eig(60)
    PetscInt  :: neig, its
    KSPConvergedReason :: reason
    PetscErrorCode :: ierr
    integer :: i, jmin, ncluster
    real*8  :: tmp, re_min, re_max, im_max, re_minpos
    character(len=16) :: converged_txt

    PetscCallA(MatCreateVecs(g_mctx%A_pair_uw, x, b, ierr))
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))
    PetscCallA(VecSetRandom(b, rnd, ierr))

    PetscCallA(KSPCreate(comm, ksp, ierr))
    PetscCallA(KSPSetOperators(ksp, g_mctx%S_uw_shell, g_mctx%S_uw_shell, ierr))
    PetscCallA(KSPSetType(ksp, KSPGMRES, ierr))
    PetscCallA(KSPGMRESSetRestart(ksp, 60, ierr))
    PetscCallA(KSPSetTolerances(ksp, 1.d-10, 1.d-50, PETSC_CURRENT_REAL, 60, ierr))
    PetscCallA(KSPSetComputeEigenvalues(ksp, PETSC_TRUE, ierr))
    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(pc, ps_puw_pc_apply, ierr))

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
    re_minpos = huge(1.d0); ncluster = 0
    do i = 1, neig
      re_max = max(re_max, r_eig(i))
      im_max = max(im_max, abs(c_eig(i)))
      if (r_eig(i) > 0.d0) re_minpos = min(re_minpos, r_eig(i))
      if (r_eig(i) >= 0.9d0 .and. r_eig(i) <= 1.1d0) ncluster = ncluster + 1
    enddo
    if (re_minpos == huge(1.d0)) re_minpos = 0.d0

    if (my_id == 0) then
      write(*,'(A,ES11.4)') "[Metriplectic] PS2: spec(P_uw^-1 S_uw) at tau =", m_tau
      write(*,'(A,I5,A,A)') "[Metriplectic] PS2: GMRES iterations = ", its, &
                            ", ", trim(converged_txt)
      write(*,'(A,4ES13.4)') &
        "[Metriplectic] PS2: min Re, min Re>0, max Re, max |Im| = ", &
        re_min, re_minpos, re_max, im_max
      write(*,'(A,I4,A)')   "[Metriplectic] PS2: ", ncluster, " Ritz in [0.9,1.1]"
      write(*,'(A,I4,A)')   "[Metriplectic] PS2: ", neig, " Ritz values (Re, Im):"
      do i = 1, neig
        write(*,'(A,2ES14.5)') "[Metriplectic] PS2:   ", r_eig(i), c_eig(i)
      enddo
    endif

    PetscCallA(KSPDestroy(ksp, ierr))
    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(VecDestroy(b, ierr))
    PetscCallA(VecDestroy(x, ierr))
  end subroutine run_PS2


  !> PCSHELL apply for PS2: y = P_uw^-1 x via the factored MUMPS solve
  !! owned by ksp_Puw (single factorization; shared, not re-factored).
  subroutine ps_puw_pc_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    call KSPSolve(g_mctx%ksp_Puw, x, y, ierr)
    ierr = 0
  end subroutine ps_puw_pc_apply


  !====================================================================
  ! PS3: production 4-var spectrum (spec Sec. 7.3, note Sec. pschecks).
  ! Ritz of the TRUE 4-var system m_A4 under the single-pass 'PS' apply
  ! vs the 'K4' MUMPS-LU reference apply. The K4 row's residual from 1 is
  ! the additive model gap (kink + model mismatch, A - A_k4); the PS-minus-
  ! K4 difference is the pure single-pass Schur-approximation cost.
  !====================================================================
  subroutine run_PS3(comm, my_id)
    integer, intent(in) :: comm, my_id

    KSP :: ksp
    PC  :: pc
    Vec :: b4, x4
    PetscRandom :: rnd
    PetscReal :: r_eig(60), c_eig(60)
    PetscInt  :: neig, its
    KSPConvergedReason :: reason
    PetscErrorCode :: ierr
    integer :: im, i, jmin, ps_it_save
    real*8  :: tmp, re_min, re_max, im_max
    character(len=2)  :: modes(2), khalf_save
    character(len=16) :: converged_txt
    character(len=20) :: apply_tag

    modes(1) = 'PS'; modes(2) = 'K4'
    khalf_save = g_mctx%khalf_mode
    ps_it_save = g_mctx%ps_inner_it
    g_mctx%ps_inner_it = 0            ! PS row = single-pass P_uw^-1

    do im = 1, 2
      g_mctx%khalf_mode = modes(im)

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

      if (modes(im) == 'PS') then
        apply_tag = "single-pass"
      else
        apply_tag = "MUMPS-LU reference"
      endif
      if (my_id == 0) then
        write(*,'(A,A,A,A,A)') "[Metriplectic] PS3: 4-var Ritz, apply = ", modes(im), &
                           " (", trim(apply_tag), ")"
        write(*,'(A,I5,A,A)')  "[Metriplectic] PS3: GMRES iterations = ", its, &
                               ", ", trim(converged_txt)
        write(*,'(A,3ES13.4)') "[Metriplectic] PS3: min Re, max Re, max |Im| = ", &
                               re_min, re_max, im_max
        write(*,'(A,I4,A)')    "[Metriplectic] PS3: ", neig, " Ritz values (Re, Im):"
        do i = 1, neig
          write(*,'(A,2ES14.5)') "[Metriplectic] PS3:   ", r_eig(i), c_eig(i)
        enddo
      endif

      PetscCallA(KSPDestroy(ksp, ierr))
      PetscCallA(PetscRandomDestroy(rnd, ierr))
      PetscCallA(VecDestroy(b4, ierr))
      PetscCallA(VecDestroy(x4, ierr))
    enddo

    g_mctx%khalf_mode  = khalf_save
    g_mctx%ps_inner_it = ps_it_save
  end subroutine run_PS3


  !====================================================================
  ! PS4: inner-solve cost curve (spec Sec. 7.3, note Sec. pschecks).
  ! FGMRES(P_uw) iterations on the matrix-free S_uw to fixed relative
  ! reductions {1e-2, 1e-4, 1e-6}, worst of 3 random RHS. Decides
  ! single-pass vs inner-iterated production and whether the two-level
  ! correction is critical path (esp. at eta_num = 0).
  !====================================================================
  subroutine run_PS4(comm, my_id)
    integer, intent(in) :: comm, my_id

    Vec :: b, x
    PetscRandom :: rnd
    PetscInt  :: its
    PetscErrorCode :: ierr
    integer :: k, it, its_worst(3), ps_it_save
    real*8  :: ps_tol_save, thr(3)

    thr(1) = 1.d-2; thr(2) = 1.d-4; thr(3) = 1.d-6
    its_worst = 0
    ps_it_save  = g_mctx%ps_inner_it
    ps_tol_save = g_mctx%ps_inner_tol

    PetscCallA(MatCreateVecs(g_mctx%A_pair_uw, x, b, ierr))
    PetscCallA(PetscRandomCreate(comm, rnd, ierr))

    do k = 1, 3
      PetscCallA(VecSetRandom(b, rnd, ierr))
      do it = 1, 3
        call KSPSetTolerances(g_mctx%ksp_Suw, thr(it), 1.d-50, &
                              PETSC_CURRENT_REAL, 200, ierr)
        call KSPSolve(g_mctx%ksp_Suw, b, x, ierr)
        PetscCallA(KSPGetIterationNumber(g_mctx%ksp_Suw, its, ierr))
        its_worst(it) = max(its_worst(it), int(its))
      enddo
    enddo

    if (my_id == 0) then
      write(*,'(A,ES11.4)') "[Metriplectic] PS4: FGMRES(P_uw) on S_uw at tau =", m_tau
      write(*,'(A,3(A,I4))') "[Metriplectic] PS4: worst its to r/r0 =", &
        "  1e-2: ", its_worst(1), "   1e-4: ", its_worst(2), "   1e-6: ", its_worst(3)
    endif

    PetscCallA(PetscRandomDestroy(rnd, ierr))
    PetscCallA(VecDestroy(b, ierr))
    PetscCallA(VecDestroy(x, ierr))

    ! restore the as-built inner-solver configuration
    if (ps_it_save > 0) then
      call KSPSetTolerances(g_mctx%ksp_Suw, ps_tol_save, PETSC_CURRENT_REAL, &
                            PETSC_CURRENT_REAL, ps_it_save, ierr)
    else
      call KSPSetTolerances(g_mctx%ksp_Suw, ps_tol_save, PETSC_CURRENT_REAL, &
                            PETSC_CURRENT_REAL, 30, ierr)
    endif
  end subroutine run_PS4


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
