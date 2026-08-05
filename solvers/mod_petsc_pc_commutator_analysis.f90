module mod_petsc_pc_commutator_analysis
!----------------------------------------------------------------
! Commutator-device (M_*) intertwining-defect analysis.
!
! Offline diagnostic, gated by the namelist flag commutator_analysis.
! Reports eps per toroidal harmonic for every candidate M_* in the
! table owned by mod_petsc_pc_commutator_table, plus the fundamental
! lower bound EPSMIN over ALL M_*.
!
! This module is SELF-CONTAINED: it extracts its own sub-blocks from the
! Jacobian and builds its own mass factorizations. Its only dependency is
! mod_petsc_pc_commutator_table (candidate table + building blocks).
!
! The block-extraction helpers at the bottom (create_index_sets,
! extract_block, setup_lu_ksp) were previously shared with the
! metriplectic PC; they are generic and are kept private here. Note that
! mod_petsc_pc_physics_ctx maintains its own equivalent index sets --
! that duplication predates this module and is deliberately left alone.
!
! All output lines are rank-0 and prefixed [Commutator].
!----------------------------------------------------------------
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_commutator_table, only: CM_NOP, CM_MAXC, CM_LABLEN, &
        CM_OP_QR, cm_table_build, cm_ops_gather, cm_mstar_mult, cm_blocks_ready
  implicit none
  private

  public :: petsc_commutator_run_analysis

  !> Per-variable index sets over the Jacobian rows, built once on first use.
  IS      :: cm_is_var(6)
  logical :: cm_is_created = .false.

contains

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
  ! NOTE ON REPRODUCIBILITY: the probes come from RANDOM_NUMBER without an
  ! explicit seed, so eps values move by ~1% between runs and EPSMIN by
  ! ~10%. Comparisons across runs must be read with that spread in mind.
  !
  ! Candidates are rows of a coefficient TABLE over the operator set
  !   {B11(=A_pp), Q1R(=B33,1/R mass), QR(=B44,R mass), ADV1, ADVR, COMP,
  !    S1R, SR},  applied matrix-free:  A_uM du = sum_iop coef(iop)*op(iop) du.
  ! The table itself lives in mod_petsc_pc_commutator_table (cm_table_build).
  ! Adding a candidate is ONE table row there -- no reassembly. The
  ! reference P always uses Q_psi = B33 regardless of the candidate's own Q_u.
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
  ! Reads A_full sub-blocks + the shared building blocks owned by
  ! mod_petsc_pc_commutator_table.
  ! NOTE on the zero-flow check: at u0=0, B11 = (1+zeta) Q holds in the
  ! interior, so eps(M1)=0 there to machine precision; boundary rows of
  ! B11 (psi Dirichlet) and B33 (j aux) differ, leaving a small boundary
  ! residual.
  !====================================================================
  subroutine petsc_commutator_run_analysis(A_full, my_id)
    use mod_parameters, only: n_tor, var_psi, var_u, var_zj, var_w
    use phys_module,    only: mode, time_evol_theta, time_evol_zeta, tstep, tstep_prev, eta

    Mat,     intent(in) :: A_full
    integer, intent(in) :: my_id

    integer, parameter :: CM_NPROBE = 32          ! Hutchinson probes per harmonic
    integer, parameter :: CM_LS_MAXIT = 300       ! CG cap for the eps_min LS
    real*8,  parameter :: CM_LS_RTOL  = 1.d-8     ! relative rz drop for that CG

    Mat :: B11, B12, Qpsi, QuR, op(CM_NOP)
    KSP :: ksp_Qpsi, ksp_QuR, ksp_Qu
    Vec :: z, va, vb, vPz, vc, vt, vd, vRz, vDz, vg
    Vec :: vy, wd, ws, wq, wr, wzu, wp, wtu       ! eps_min least-squares work
    PetscErrorCode :: ierr
    integer :: comm, h, kp, ic, ncand
    integer :: nit, nit_tot, nit_max
    real*8  :: opz, zeta, tdt, nd, npp, sumP2, eps_val
    real*8  :: sumE2, ndbest, fls, eps_min, eps_best
    real*8  :: coef(CM_MAXC, CM_NOP)
    integer :: quop(CM_MAXC)
    character(len=CM_LABLEN) :: lab(CM_MAXC)
    real*8, allocatable :: sumD2(:)

    call PetscObjectGetComm(A_full, comm, ierr)
    zeta = time_evol_zeta * 2.d0 * tstep / (tstep + tstep_prev)
    opz  = 1.d0 + zeta
    tdt  = time_evol_theta * tstep

    ! --- Index sets + sub-blocks: A_pp, A_pu, 1/R mass (B33), R mass (B44) ---
    if (.not. cm_is_created) call create_index_sets(A_full, comm)
    call extract_block(A_full, var_psi, var_psi, B11)
    call extract_block(A_full, var_psi, var_u,   B12)
    call extract_block(A_full, var_zj,  var_zj,  Qpsi)
    call extract_block(A_full, var_w,   var_w,   QuR)

    ! --- Full mass factorizations (MUMPS LU; no lumping) ---
    call setup_lu_ksp(Qpsi, ksp_Qpsi, comm, .false.)
    call setup_lu_ksp(QuR,  ksp_QuR,  comm, .false.)

    ! --- Operator handles + candidate table (the table itself lives in
    !     mod_petsc_pc_commutator_table) ---
    call cm_ops_gather(B11, Qpsi, QuR, op)
    call cm_table_build(opz, tdt, eta, coef, quop, lab, ncand)

    allocate(sumD2(ncand))

    if (my_id == 0) then
      write(*,'(A)') "[Commutator] ========= M_* intertwining-defect analysis ========="
      write(*,'(A,I0,A,F8.4,A,I0)') "[Commutator] probes/harmonic = ", CM_NPROBE, &
                               ",  (1+zeta) = ", opz, ",  candidates = ", ncand
      if (.not. cm_blocks_ready()) write(*,'(A)') &
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
          call cm_mstar_mult(op, coef, ic, z, vc, vt, ierr)
          ksp_Qu = ksp_Qpsi
          if (quop(ic) == CM_OP_QR) ksp_Qu = ksp_QuR
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

    ! --- Cleanup (the building blocks are owned by the table module;
    !     not destroyed here) ---
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
    call VecGetArray(z, arr, ierr)
    do i = 0, n_block_local - 1
      call random_number(r)
      if (r >= 0.5d0) then
        arr(i*n_tor + (h-1) + 1) =  1.d0
      else
        arr(i*n_tor + (h-1) + 1) = -1.d0
      endif
    enddo
    call VecRestoreArray(z, arr, ierr)
  end subroutine cm_probe_fill


  !====================================================================
  ! Generic block-extraction helpers (previously shared with the
  ! metriplectic PC; private to this module now).
  !====================================================================

  !> Build the per-variable index sets over the rows of the Jacobian.
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
      PetscCallA(ISCreateGeneral(comm, n_var_dofs, indices, PETSC_COPY_VALUES, cm_is_var(v), ierr))
    enddo
    deallocate(indices)
    cm_is_created = .true.
  end subroutine create_index_sets


  !> Extract the (eq_row, var_col) sub-block of the Jacobian.
  subroutine extract_block(A_full, eq_row, var_col, B)
    Mat, intent(in)    :: A_full
    integer, intent(in) :: eq_row, var_col
    Mat, intent(out)    :: B
    PetscErrorCode :: ierr
    PetscCallA(MatCreateSubMatrix(A_full, cm_is_var(eq_row), cm_is_var(var_col), MAT_INITIAL_MATRIX, B, ierr))
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

#endif
end module mod_petsc_pc_commutator_analysis
