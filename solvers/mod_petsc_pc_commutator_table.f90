module mod_petsc_pc_commutator_table
!----------------------------------------------------------------
! Candidate table for the commutator-device operator M_*
! (note JOREK_commutator_preconditioner_baseline, Sec. 8.5, Table 1).
!
! SINGLE SOURCE OF TRUTH for the candidates: the offline intertwining-defect
! analysis (petsc_commutator_run_analysis in
! mod_petsc_pc_commutator_analysis, eps per toroidal harmonic) reads them
! from here, so adding a candidate is ONE table row.
!
! A candidate is a linear combination over the operator set
!   1 = B11  (amat_11, = a I + theta A, inertia + ExB advection)
!   2 = Q1R  (amat_33, 1/R scalar mass)   <- the psi/u/j Riesz map
!   3 = QR   (amat_44, R scalar mass)
!   3+ib    (ib = 1..CM_NB) the assembled building blocks of
!           mod_elt_matrix_commutator: ADV1, ADVR, COMP, S1R, SR,
!                                      ES1R, EG1R
! together with a choice quop(ic) of WHICH mass plays the u-space Riesz
! map Q_u for that candidate, so that
!
!   A_uM(ic) = sum_iop coef(ic,iop) * op(iop)      (assembled form)
!   M_*(ic)  = Q_u(ic)^-1 A_uM(ic)                 (the operator)
!
! Caution 5 of the note: A_uM and Q_u must carry the SAME geometric
! weight. The 1/R rows (B11, Q1R, ADV1, S1R) pair with quop = CM_OP_Q1R;
! the R rows (QR, ADVR, COMP, SR) pair with quop = CM_OP_QR. Mixing them
! reports a large, entirely self-inflicted defect.
!
! This module is deliberately STATELESS with respect to the operator
! handles: cm_table_build only fills coefficients, and cm_mstar_mult takes
! the op array explicitly. The analysis extracts its own blocks and mass
! factorizations; the preconditioner uses g_mctx's. Both can be active in
! one run without clobbering each other. The only module state is the
! assembled building-block array cm_blk, which is genuinely global (one
! assembly per matrix construction, shared by every consumer).
!
! TIME-FACTOR CONVENTION: opz and tdt are ARGUMENTS, not read from
! phys_module, so each caller supplies its own convention.
! petsc_commutator_run_analysis passes the variable-step Gears value
! zeta = time_evol_zeta*2*tstep/(tstep+tstep_prev).
!----------------------------------------------------------------
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_elt_matrix_commutator, only: CM_NB, CM_ADV1, CM_ADVR, CM_COMP, &
                                       CM_S1R, CM_SR, CM_ES1R, CM_EG1R, &
                                       CM_AISO, CM_AANI
  implicit none
  private

  public :: CM_NOP, CM_OP_B11, CM_OP_Q1R, CM_OP_QR, CM_MAXC, CM_LABLEN
  public :: CM_OP_AISO, CM_OP_AANI
  public :: cm_table_build, cm_table_lookup, cm_ops_gather, cm_mstar_mult
  public :: petsc_commutator_assemble, cm_blocks_ready

  integer, parameter :: CM_OP_B11 = 1
  integer, parameter :: CM_OP_Q1R = 2
  integer, parameter :: CM_OP_QR  = 3
  integer, parameter :: CM_NOP    = 3 + CM_NB   !< B11, Q1R, QR + building blocks
  !> Handle of the Stage 5.2 continuous-PDE (isotropic) Schur operator. It is
  !! NOT a candidate building block -- no table row references it -- but it is
  !! assembled and gathered by the same machinery, so it is addressed the same
  !! way. Kept in sync with CM_AISO's position among the blocks.
  integer, parameter :: CM_OP_AISO = 3 + CM_AISO
  !> Stage 5.3a: the same operator with the field-alignment restored.
  integer, parameter :: CM_OP_AANI = 3 + CM_AANI
  integer, parameter :: CM_MAXC   = 16          !< candidate-table capacity
  integer, parameter :: CM_LABLEN = 4           !< candidate label length

  !> Assembled building-block operators (pure integrands; all time/opz/eta
  !! factors live in the coefficient table). Module-global by design.
  Mat, save     :: cm_blk(CM_NB)
  logical, save :: cm_ops_ready = .false.

contains

  !> .true. once the element-assembled building blocks exist. Candidates
  !! that use only B11/Q1R/QR are available either way.
  logical function cm_blocks_ready()
    cm_blocks_ready = cm_ops_ready
  end function cm_blocks_ready


  !====================================================================
  ! Assemble the building-block operators. Called from jorek2_main (where
  ! the element list is available), gated by commutator_analysis.
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
    ! A_full sub-blocks / probe vectors (same size layout).
    do ib = 1, CM_NB
      PetscCallA(MatConvert(cm_blk(ib), MATMPIAIJ, MAT_INPLACE_MATRIX, cm_blk(ib), ierr))
    enddo
    cm_ops_ready = .true.
    if (my_id == 0) write(*,'(A,I0,A)') "[Commutator] ", CM_NB, " building-block operators assembled"
  end subroutine petsc_commutator_assemble


  !====================================================================
  ! Fill the operator handle table from the three caller-owned blocks
  ! plus the module-global building blocks. Entries 4..CM_NOP are left
  ! untouched when the blocks are not assembled -- cm_table_build then
  ! emits only the candidates that do not reference them.
  !====================================================================
  subroutine cm_ops_gather(B11, Q1R, QR, op)
    Mat, intent(in)  :: B11, Q1R, QR
    Mat, intent(out) :: op(CM_NOP)
    integer :: ib

    op(CM_OP_B11) = B11
    op(CM_OP_Q1R) = Q1R
    op(CM_OP_QR)  = QR
    if (cm_ops_ready) then
      do ib = 1, CM_NB
        op(3+ib) = cm_blk(ib)
      enddo
    endif
  end subroutine cm_ops_gather


  !====================================================================
  ! THE CANDIDATE TABLE -- edit rows here to try new candidates.
  !
  !   M0  = opz*Q1R                        (Q_u=Q1R)  mass only; the
  !                                        small-flow incumbent M^-1 ~ dt I
  !   M1x = B11 (the true amat_11)         (Q_u=Q1R)  Eq. (35) exactly, no
  !                                        building blocks needed
  !   M0R = opz*QR                         (Q_u=QR)   R-mass only
  !   M1a = opz*Q1R - tdt*ADV1             (Q_u=Q1R)  assembled clone of
  !                                        amat_11 (~M1x, exact at u0=0)
  !   M3  = opz*QR - tdt*ADVR - tdt*COMP   (Q_u=QR)   conservative rho-form
  !   M2  = M1a + eta*tdt*S1R              (Q_u=Q1R)  + resistive diffusion
  !   M2e = M1a + eta*tdt*ES1R             (Q_u=Q1R)  + Spitzer-SHAPED
  !                                        resistive diffusion (= M2 when
  !                                        eta_T_dependent is off)
  !
  ! M1a/M3/M2 need the assembled blocks; M0/M1x/M0R do not.
  !
  ! NOTE on M1x vs M1a: both realise Eq. (35). M1x is free (it is the true
  ! Jacobian block) but its psi-Dirichlet rows differ from Q1R's j-aux
  ! rows, so it does NOT reduce to M0 at u0 = 0. M1a does, exactly, which
  ! makes it the candidate to use for the zero-flow regression check.
  !====================================================================
  subroutine cm_table_build(opz, tdt, eta, coef, quop, lab, ncand)
    real*8,  intent(in)  :: opz, tdt, eta
    real*8,  intent(out) :: coef(CM_MAXC, CM_NOP)
    integer, intent(out) :: quop(CM_MAXC)
    character(len=CM_LABLEN), intent(out) :: lab(CM_MAXC)
    integer, intent(out) :: ncand

    coef = 0.d0
    quop = CM_OP_Q1R
    lab  = ' '
    ncand = 0

    ncand=ncand+1; lab(ncand)="M0" ; quop(ncand)=CM_OP_Q1R; coef(ncand,CM_OP_Q1R)=opz
    ncand=ncand+1; lab(ncand)="M1x"; quop(ncand)=CM_OP_Q1R; coef(ncand,CM_OP_B11)=1.d0
    ncand=ncand+1; lab(ncand)="M0R"; quop(ncand)=CM_OP_QR ; coef(ncand,CM_OP_QR )=opz
    if (cm_ops_ready) then
      ncand=ncand+1; lab(ncand)="M1a"; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt
      ncand=ncand+1; lab(ncand)="M3" ; quop(ncand)=CM_OP_QR
        coef(ncand,CM_OP_QR)=opz;  coef(ncand,3+CM_ADVR)=-tdt; coef(ncand,3+CM_COMP)=-tdt
      ncand=ncand+1; lab(ncand)="M2" ; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt; coef(ncand,3+CM_S1R)=eta*tdt
      ! M2 uses the CONSTANT eta, but the substituted resistive term in
      ! P_full's D_psi carries the Spitzer eta_T(T0) = eta*(T0/T_0)^-1.5,
      ! which on a pedestal case varies by ~1e3 between axis and cold edge.
      ! These scaled variants separate "wrong constant" from "wrong operator
      ! form": if some alpha collapses the defect, a scalar suffices; if the
      ! defect plateaus, the spatial weight is required.
      ncand=ncand+1; lab(ncand)="M2b"; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt; coef(ncand,3+CM_S1R)=1.d1*eta*tdt
      ncand=ncand+1; lab(ncand)="M2c"; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt; coef(ncand,3+CM_S1R)=1.d2*eta*tdt
      ncand=ncand+1; lab(ncand)="M2d"; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt; coef(ncand,3+CM_S1R)=1.d3*eta*tdt
      ! --- Spitzer-shaped resistive diffusion --------------------------
      ! The substituted resistive term of P_full's D_psi integrates by parts
      ! into eta_T*stiffness PLUS a grad(eta_T) first-order term (see the
      ! header of mod_elt_matrix_commutator). All three of the following take
      ! the same coefficient eta*tdt as M2 and reduce to M2 exactly when
      ! eta_T_dependent is off (ES1R -> S1R, EG1R -> 0).
      !   M2e = stiffness half only     -- diagnostic: expected to be WORSE
      !         than M2 (measured 3.90 vs 2.70), because the omitted
      !         grad(eta_T) piece is comparable in a pedestal layer.
      !   M2g = BOTH halves             -- the actual target operator.
      !   M2h = grad(eta_T) half only   -- diagnostic: isolates its weight.
      ncand=ncand+1; lab(ncand)="M2e" ; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt; coef(ncand,3+CM_ES1R)=eta*tdt
      ncand=ncand+1; lab(ncand)="M2g" ; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt
        coef(ncand,3+CM_ES1R)=eta*tdt; coef(ncand,3+CM_EG1R)=eta*tdt
      ncand=ncand+1; lab(ncand)="M2h" ; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt; coef(ncand,3+CM_EG1R)=eta*tdt
      ! alpha bracket on the full pair, to confirm alpha = 1 is optimal once
      ! the operator FORM is right (with S1R alone the sweep was flat).
      ncand=ncand+1; lab(ncand)="M2ga"; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt
        coef(ncand,3+CM_ES1R)=0.5d0*eta*tdt; coef(ncand,3+CM_EG1R)=0.5d0*eta*tdt
      ncand=ncand+1; lab(ncand)="M2gb"; quop(ncand)=CM_OP_Q1R
        coef(ncand,CM_OP_Q1R)=opz; coef(ncand,3+CM_ADV1)=-tdt
        coef(ncand,3+CM_ES1R)=2.d0*eta*tdt; coef(ncand,3+CM_EG1R)=2.d0*eta*tdt
    endif
  end subroutine cm_table_build


  !> Index of the candidate labelled `label`, or 0 if absent (e.g. a
  !! block-based candidate requested without the blocks assembled).
  integer function cm_table_lookup(lab, ncand, label)
    character(len=CM_LABLEN), intent(in) :: lab(CM_MAXC)
    integer,                  intent(in) :: ncand
    character(len=*),         intent(in) :: label
    integer :: ic

    cm_table_lookup = 0
    do ic = 1, ncand
      if (trim(lab(ic)) == trim(label)) then
        cm_table_lookup = ic
        return
      endif
    enddo
  end function cm_table_lookup


  !====================================================================
  ! Matrix-free assembled-form apply:  y = A_uM(ic) x
  !   y = sum_iop coef(ic,iop) * op(iop) * x
  ! `t` is a caller-owned work vector of the same layout as y.
  !====================================================================
  subroutine cm_mstar_mult(op, coef, ic, x, y, t, ierr)
    Mat,    intent(in)    :: op(CM_NOP)
    real*8, intent(in)    :: coef(CM_MAXC, CM_NOP)
    integer, intent(in)   :: ic
    Vec,    intent(in)    :: x
    Vec,    intent(inout) :: y, t
    PetscErrorCode, intent(inout) :: ierr
    integer :: iop

    PetscCallA(VecSet(y, 0.d0, ierr))
    do iop = 1, CM_NOP
      if (coef(ic,iop) /= 0.d0) then
        PetscCallA(MatMult(op(iop), x, t, ierr))
        PetscCallA(VecAXPY(y, coef(ic,iop), t, ierr))
      endif
    enddo
  end subroutine cm_mstar_mult

#endif
end module mod_petsc_pc_commutator_table
