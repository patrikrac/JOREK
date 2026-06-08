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

    !> Lumped mass inverse vectors (1-var layout from extracted sub-blocks)
    Vec :: diag_Mj_inv, diag_Mw_inv

    !> Block diagonal inverse matrices (BAIJ, block_size = n_tor)
    !! Computed via MatInvertBlockDiagonalMat from B_33/B_44
    Mat :: Dinv_Mj, Dinv_Mw
    logical :: dinv_created   = .false.
    logical :: dinv_created_w = .false.

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

    !> Reassembled diagonal blocks (simplified integrands, used when physics_pc_reassemble = .true.)
    Mat :: R_11, R_22, R_55, R_66
    logical :: reassembled_ready = .false.

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

    ! --- Sub-blocks PC: 2x2 super-blocks (psi,u) and (rho,T) ---
    Mat :: K_A_block, K_B_block          !< 2x2 MatNests of (psi,u) and (rho,T) sub-systems
    Mat :: K_A_aij,   K_B_aij            !< MPIAIJ conversions of the MatNests for MUMPS
    KSP :: ksp_block_A, ksp_block_B      !< PREONLY + LU + MUMPS solvers, one per super-block
    Vec :: rhs_A, sol_A                  !< (psi,u)-sized packed work vectors
    Vec :: rhs_B, sol_B                  !< (rho,T)-sized packed work vectors
    Vec :: tmp_rho, tmp_T                !< 1-variable scratch for GS residual updates
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
  end type type_physics_pc_ctx

  type(type_physics_pc_ctx), save :: g_ctx

#endif
end module mod_petsc_pc_physics_ctx
