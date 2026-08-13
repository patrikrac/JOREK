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

#endif
end module mod_petsc_pc_physics_ctx
