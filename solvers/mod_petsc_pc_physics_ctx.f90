module mod_petsc_pc_physics_ctx
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  public

  ! This module holds the shared context type (type_physics_pc_ctx) and the
  ! module-level singleton instance (g_ctx) used by the physics PC family of
  ! modules.  All other modules in the family `use` this one to access the
  ! type definition and the singleton without circular dependencies.

  type :: type_physics_pc_ctx
    logical :: initialized    = .false.
    logical :: matrices_ready = .false.
    logical :: reduced_ready  = .false.
    integer :: comm           = -1
    !> 1-var BAIJ matrices from separate element-level assembly
    ! TODO: The following matrices are unused and depriciated!
    Mat :: A_j, A_w, A_jpsi, A_wu

    !> Index sets for each variable in the full system vector (1=psi..6=T)
    IS :: is_var(6)
    logical :: is_created = .false.

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

    !> Schur-corrected blocks:
    !! Diagonal:
    !! Atilde_11 = B_11 - B_13 * D_j^{-1} * B_31
    !! Atilde_22 = B_22 - B_24 * D_w^{-1} * B_42
    Mat :: Atilde_11, Atilde_22
    Mat :: K_psi_correction
    logical :: psi_correction_ready = .false.
    Mat :: K_u_correction
    logical :: u_correction_ready = .false.
    !! Off-diagonal (psi-column and u-column get corrections):
    !! Atilde_21 = B_21 - B_23 * D_j^{-1} * B_31
    !! Atilde_61 = B_61 - B_63 * D_j^{-1} * B_31
    Mat :: Atilde_21, Atilde_61
    !! Element-assembled approximations for the off-diagonal Schur corrections
    !! (alternative to the diagonal-mass-inverse path; integrand TBD)
    Mat :: K_21_correction, K_61_correction
    logical :: correction_21_ready = .false.
    logical :: correction_61_ready = .false.

    !> KSP for each diagonal block of the reduced 4x4 system
    KSP :: ksp_psi, ksp_u, ksp_rho, ksp_T
    logical :: ksp_created = .false.

    !> KSP for elliptic constraint mass matrices (replaces approx mass inverse)
    KSP :: ksp_Mj, ksp_Mw
    logical :: ksp_elliptic_created = .false.

    !> Work vectors (1-var size) — allocated once, reused in every apply
    Vec :: work_1, work_2, work_3, work_4, work_5

    !> P_full: the reduced 4-variable PDE operator assembled from
    !! pc_elt_matrix_reduced_fft by substituting j = J(psi), w = W(u) at the
    !! continuous level (Milestone 1 of docs/physics_pc). Distinct from
    !! A_reduced_4x4, which is built from blocks EXTRACTED out of the assembled
    !! mixed Jacobian plus algebraic Schur corrections.
    Mat :: P_full_pde
    logical :: p_full_ready = .false.

    !> P_full_pde permuted into the variable-block-contiguous layout that
    !! A_reduced_4x4 and is_reduced use, so it can be dropped into ksp_reduced
    !! as a direct replacement (Milestone 2). P_full_pde itself is BAIJ with
    !! node-major/variable/toroidal interleaving; the two orderings differ.
    Mat :: P_full_perm
    logical :: p_full_perm_ready = .false.

    !> Monolithic 4x4 reduced system (stage-one test mode)
    Mat :: A_reduced_4x4
    KSP :: ksp_reduced
    logical :: ksp_reduced_created = .false.
    Vec :: work_rhs_4v, work_sol_4v
    logical :: work_4v_created = .false.
    IS :: is_reduced(4)
    logical :: is_reduced_created = .false.

    Mat :: M_hydro
    KSP :: ksp_hydro
    logical :: ksp_hydro_created = .false.
    IS :: is_hydro(3)
    Vec :: work_rhs_3v
    Vec :: hydro_predictor_3v

    Mat :: A_alfven
    KSP :: ksp_alfven
    logical :: ksp_alfven_created = .false.
    Vec :: work_rhs_2v

    Mat :: S_u ! Exact schur complement, to be removed ?
    Mat :: S_PBP ! Approximation (Physics Based) of the Schur complement S_u
    KSP :: ksp_S_PBP
    logical :: ksp_S_PBP_created = .false.

    ! --- Stage 6.3: commutator-device momentum Schur (physics_pc_schur_variant /= "SF") ---
    !! S_PBP then holds S_ass = Atilde_22 Q_u^-1 A_uM - Atilde_21 Q_p^-1 B_12, whose
    !! inverse is only half the operator: the apply must finish with the RIGHT factor
    !! Shat^-1 = Q_u^-1 A_uM S_ass^-1. Hence A_uM and Q_u^-1 have to survive the build.
    Mat :: A_uM_prod                     !< assembled A_uM(ic); OWNED here, rebuilt every PC rebuild
    Mat :: Qi_u_prod                     !< REFERENCE to sfp_Qip/sfp_QiR; NOT owned, never destroyed here
    Vec :: u_mask                        !< 1 on interior u-rows, 0 on the ZBIG Dirichlet rows
    Vec :: u_bnd                         !< 1/diag on the ZBIG rows, 0 elsewhere (the Jacobi add-back)
    Vec :: u_one                         !< 1 - u_mask, the identity put on the masked rows
    logical :: schur_comm_ready  = .false. !< A_uM_prod / u_* exist and must be destroyed before reuse
    logical :: schur_comm_active = .false. !< the apply must use the right factor
    logical :: schur_comm_mask   = .false. !< ... and the ZBIG interior treatment

    ! --- Workstream B: small-flow MIXED-PAIR arm (physics_pc_schur_variant = "SFM") ---
    !! Neither constraint variable is eliminated. Instead of substituting
    !! j = J(psi) and w = W(u) -- which raises the differential order of the psi
    !! and u diagonals to fourth -- the sweep keeps both constraints explicit and
    !! solves two 2x2 pairs:
    !!   pair_psi = [[B_11, B_13], [B_31, B_33]]              (psi,j kept mixed)
    !!   pair_w   = [[S_uu^SFM, B_24], [B_42, B_44]]          (u,omega kept mixed)
    !! with the small-flow Schur channel
    !!   S_uu^SFM = B_22 - sum_ch (L Qi U)/(1+zeta),  L in {B_21, B_25, B_26}
    !! and Riesz maps Qi = B_33^-1 (psi channel), B_44^-1 (rho, T channels).
    !!
    !! RAW blocks only: the base is B_22 and the psi-channel L is B_21, NOT their
    !! Schur-corrected counterparts. The fourth-order couplings B_24 M_w^-1 B_42
    !! and B_23 M_j^-1 B_31 are carried by the mixed pairs themselves, so
    !! subtracting them from the diagonals as well would double-count them --
    !! silently, with no error and no log difference. Every block of pair_psi and
    !! pair_w is therefore second order or a mass matrix, which is the point.
    !!
    !! The j-channel is absent from S_uu^SFM because the j-row of the upper
    !! coupling U is identically zero (the j constraint carries no u-coupling), so
    !! a block-diagonal Riesz M_y contributes nothing through it. That is a
    !! statement about the SCHUR CHANNEL only -- at the sweep level j is fully
    !! present, via the explicit B_23 j* and B_63 j* terms in the apply.
    !!
    !! CONSEQUENCE FOR THE APPLY: Jacobian rows 3 and 4 are [B_31,0,B_33,0,0,0]
    !! and [0,B_42,0,B_44,0,0], so both pairs represent their constraint equation
    !! EXACTLY. This arm therefore needs NO constraint mass pre-solve and NO
    !! back-substitution -- j and omega come out of the pair solves. The
    !! dispatcher must skip both, and must not fold -B_24 M_w^-1 x_w into the u
    !! residual: that coupling lives in the (1,2) entry of pair_w.
    Mat :: K_pj_aij          !< OWNED MPIAIJ of pair_psi; rebuilt every PC rebuild
                             !  (B_11 carries theta*tstep and the evolving state)
    Mat :: S_W_aij           !< OWNED MPIAIJ of pair_w;   rebuilt every PC rebuild
    KSP :: ksp_pair_psi      !< PREONLY + LU on K_pj_aij
    KSP :: ksp_pair_w        !< PREONLY + LU on S_W_aij
    Vec :: rhs_PJ, sol_PJ    !< packed (psi,j)-sized work vecs;   created ONCE
    Vec :: rhs_W,  sol_W     !< packed (u,omega)-sized work vecs; created ONCE
                             !  The Mat handles change on every rebuild but their
                             !  SIZES never do, so create-once is correct.
    IS  :: is_pair_psi(2), is_pair_w(2)  !< layout-only strides into the packed
                             !  vectors. Unused by the direct arm; they are what a
                             !  future PCFIELDSPLIT inner solver needs.
    logical :: schur_mixed_ready      = .false. !< K_pj_aij/S_W_aij exist and must be
                                                !  destroyed before being rebuilt
    logical :: schur_mixed_vecs_ready = .false. !< rhs_PJ/sol_PJ/rhs_W/sol_W + the ISs exist
    logical :: schur_mixed_active     = .false. !< the apply must take the mixed-pair arm.
                                                !  Set by the BUILDER on success, never read
                                                !  from the namelist by the apply -- so a
                                                !  failed build can never be mistaken for a
                                                !  successful one.

    ! --- Sub-blocks PC: (psi,u) Alfven super-block + independent rho, T blocks ---
    Mat :: K_A_block                     !< 2x2 MatNest of the (psi,u) Alfven super-system
    Mat :: K_A_aij                       !< MPIAIJ conversion of K_A_block for the block solver
    KSP :: ksp_block_A                   !< (psi,u) Alfven super-block solver
    Vec :: rhs_A, sol_A                  !< (psi,u)-sized packed work vectors
    Vec :: tmp_rho, tmp_T                !< 1-variable rho/T residual vectors (GS + direct solves)
    logical :: sub_blocks_setup_done = .false.

    ! --- Part 2a: exact Alfven operator MATSHELL (ALFVEN_SOLVER_TOROIDAL_EXACT) ---
    !! Shell applies K_A_exact*(p,q):
    !!   y_psi = B_11 p + B_12 q - B_13 (M_jj^-1 B_31 p)
    !!   y_u   = B_21 p + B_22 q - B_23 (M_jj^-1 B_31 p) - B_24 (M_ww^-1 B_42 q)
    !! All work vecs are 1-variable sized; created once.
    Mat :: K_A_exact_shell
    Vec :: kae_p,  kae_q              !< unpacked inputs (psi-, u-sized)
    Vec :: kae_tj, kae_tw             !< M_jj^-1 B_31 p (j-), M_ww^-1 B_42 q (w-)
    Vec :: kae_sj, kae_sw             !< scratch for B_31 p (j-), B_42 q (w-)
    Vec :: kae_ypsi, kae_yu           !< accumulated outputs (psi-, u-sized)
    Vec :: kae_spsi, kae_su           !< scratch for the -B_1x/-B_2x terms
    logical :: kae_setup_done = .false.

    ! ================================================================
    ! TEMPORARY diagnostic (Option A, Step 2) -- TO BE REPLACED (Step 3).
    ! S_PBP MatShell using the consistent-mass approximation
    !   Atilde_11^{-1} -> (1+zeta)^{-1} M_j^{-1}     (M_j = B_33 = ksp_Mj)
    ! with channels B and C kept EXACT (dedicated B_55/B_66 MUMPS solves).
    ! Built only for offline spectrum analysis vs the exact S_u; isolates the
    ! quality of the Atilde_11^{-1} -> (1+zeta)^{-1} M_j^{-1} substitution.
    ! Delete these fields + setup_S_PBP_diag_shell/materialize_S_PBP_diag_aij
    ! once the production S_PBP operator exists.
    ! ================================================================
    Mat     :: S_PBP_diag_shell
    KSP     :: spbpd_ksp_B55, spbpd_ksp_B66   !< dedicated EXACT (MUMPS LU) B_55/B_66 solves
    real*8  :: spbpd_inv_gears = 1.0d0        !< cached 1/(1+zeta)
    logical :: spbpd_setup_done = .false.
    Vec     :: spbpd_zpsi, spbpd_ypsi         !< psi-sized: B_12 x ; (1+zeta)^-1 M_j^-1 B_12 x
    Vec     :: spbpd_zrho, spbpd_yrho         !< rho-sized scratch (channels B/C)
    Vec     :: spbpd_zT,   spbpd_yT           !< T-sized scratch   (channels B/C)
    Vec     :: spbpd_ru                       !< u-sized channel accumulator scratch
    Vec     :: spbpd_relax                    !< psi-sized per-harmonic TOROIDAL resistive relaxation factor
    logical :: spbpd_use_toroidal_relax = .false.  !< toggle: apply spbpd_relax to ypsi
  end type type_physics_pc_ctx

  type(type_physics_pc_ctx), save :: g_ctx

  ! ================================================================
  ! PetscLogEvents for the physics PC.
  !
  ! These exist to answer one question that nothing in the code answers
  ! today: of the wall time the physics PC costs, how much is the rebuild
  ! (element assembly, block extraction, triple products, factorizations)
  ! and how much is the apply (the back-solves, once per GMRES iteration)?
  ! Every optimization decision downstream depends on that split, and it
  ! must be measured rather than guessed.
  !
  ! PetscLogEvent rather than hand-rolled MPI_Wtime counters because
  ! mod_petsc.f90:377 already registers PetscLogStages for KSP setup/solve,
  ! so this is the established pattern here; because -log_view then reports
  ! call counts, times and flops with no new output format to parse; and
  ! because the events nest correctly inside those existing stages.
  !
  ! Read them with:  -log_view
  ! The event names all share the "PhysPC_" prefix so they sort together.
  ! ================================================================
  PetscLogEvent, save :: pcev_elem_asm   = -1  !< construct_schur_correction_matrices
  PetscLogEvent, save :: pcev_extract    = -1  !< the extract_sub_block group
  PetscLogEvent, save :: pcev_build_suu  = -1  !< build_schur_mixed_prod (triple products)
  PetscLogEvent, save :: pcev_fact_pj    = -1  !< MUMPS LU of K_pj_aij
  PetscLogEvent, save :: pcev_fact_w     = -1  !< MUMPS LU of S_W_aij
  PetscLogEvent, save :: pcev_fact_rhot  = -1  !< MUMPS LU of B_55 and B_66
  PetscLogEvent, save :: pcev_apply      = -1  !< one whole physics_pc_apply
  PetscLogEvent, save :: pcev_solve_pj   = -1  !< pair_psi back-solves (2 per apply)
  PetscLogEvent, save :: pcev_solve_w    = -1  !< pair_w back-solve    (1 per apply)
  PetscLogEvent, save :: pcev_solve_rhot = -1  !< rho/T back-solves    (4 per apply)

  ! Sub-events inside pcev_build_suu. The first measurement showed that bucket
  ! costs more than all three factorizations put together, so it needs to be
  ! broken down before anything inside it is optimized.
  PetscLogEvent, save :: pcev_massinv    = -1  !< make_mass_inverse (all call sites)
  PetscLogEvent, save :: pcev_shat       = -1  !< the Shat and Ltil triple-product chains
  PetscLogEvent, save :: pcev_channel    = -1  !< add_channel (L Qi U) products
  PetscLogEvent, save :: pcev_convert    = -1  !< MatConvert nest -> MPIAIJ, both pairs
  PetscLogEvent, save :: pcev_builddiag  = -1  !< the norm/density diagnostics in the build

  logical, save, private :: pcev_registered = .false.

contains

  !--------------------------------------------------------------------
  !> .true. on the mixed-pair arms ("SFM", "SFM2"), which keep j and omega
  !! explicit and solve two 2x2 pairs.
  !!
  !! These arms read NONE of the element-assembled Schur corrections
  !! (K_psi/u/21/61_correction), none of the Atilde_* blocks derived from
  !! them, and never call ksp_psi -- apply_wave_schur_mixed_pairs works
  !! entirely from the raw B_ij blocks. Everything guarded by this predicate
  !! is therefore work whose result is computed and then discarded: a whole
  !! extra element-loop assembly over the mesh plus a MUMPS factorization,
  !! per Newton step.
  !!
  !! It lives here rather than at each call site so that the assembly gate
  !! (mod_petsc_pc_physics_element) and the consumer gates
  !! (mod_petsc_pc_physics) cannot drift apart. They must agree: skipping the
  !! assembly while still running compute_schur_corrected_block_* would build
  !! Atilde from an unassembled correction.
  !--------------------------------------------------------------------
  logical function physics_pc_mixed_arm()
    use phys_module, only: physics_pc_schur_variant

    physics_pc_mixed_arm = (trim(physics_pc_schur_variant) == "SFM" .or. &
                            trim(physics_pc_schur_variant) == "SFM2")
  end function physics_pc_mixed_arm

  !--------------------------------------------------------------------
  !> Register the physics-PC log events. Idempotent: safe to call from any
  !! entry point without the caller having to know whether it ran already.
  !! Registering twice would give two distinct events with the same name and
  !! split the timings between them, which is why the guard is here rather
  !! than at the call site.
  !--------------------------------------------------------------------
  subroutine physics_pc_log_events_register()
    PetscErrorCode :: ierr

    if (pcev_registered) return

    call PetscLogEventRegister("PhysPC_ElemAsm",   0, pcev_elem_asm,   ierr)
    call PetscLogEventRegister("PhysPC_Extract",   0, pcev_extract,    ierr)
    call PetscLogEventRegister("PhysPC_BuildSuu",  0, pcev_build_suu,  ierr)
    call PetscLogEventRegister("PhysPC_FactPJ",    0, pcev_fact_pj,    ierr)
    call PetscLogEventRegister("PhysPC_FactW",     0, pcev_fact_w,     ierr)
    call PetscLogEventRegister("PhysPC_FactRhoT",  0, pcev_fact_rhot,  ierr)
    call PetscLogEventRegister("PhysPC_Apply",     0, pcev_apply,      ierr)
    call PetscLogEventRegister("PhysPC_SolvePJ",   0, pcev_solve_pj,   ierr)
    call PetscLogEventRegister("PhysPC_SolveW",    0, pcev_solve_w,    ierr)
    call PetscLogEventRegister("PhysPC_SolveRhoT", 0, pcev_solve_rhot, ierr)

    call PetscLogEventRegister("PhysPC_MassInv",   0, pcev_massinv,    ierr)
    call PetscLogEventRegister("PhysPC_Shat",      0, pcev_shat,       ierr)
    call PetscLogEventRegister("PhysPC_Channel",   0, pcev_channel,    ierr)
    call PetscLogEventRegister("PhysPC_Convert",   0, pcev_convert,    ierr)
    call PetscLogEventRegister("PhysPC_BuildDiag", 0, pcev_builddiag,  ierr)

    pcev_registered = .true.
  end subroutine physics_pc_log_events_register

#endif
end module mod_petsc_pc_physics_ctx
