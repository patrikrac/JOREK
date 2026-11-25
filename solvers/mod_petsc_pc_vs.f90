!> The vorticity-space Schur (VS) physics-based preconditioner
!! (use_physics_pc, physics_pc_method = "vs_ideal" | "vs_resistive" |
!! "vs_lumped"): a block
!! factorization of the Jacobian over (psi, u, j, omega, rho, T) with u as the
!! last unknown, beside the split-field sweep of mod_petsc_pc_sf. With
!! B_ij = A(equation i, variable j), 1 psi, 2 u, 3 j, 4 omega, 5 rho, 6 T and
!! opz = 1 + zeta:
!!
!! The rows of j and omega are definitions with mass matrices on the diagonal,
!!   B_33 j + B_31 psi = r_j,   B_44 omega + B_42 u = r_omega.
!! With j and omega eliminated, and psi by its mass, the Schur complement of u
!! is approximated as a product of an operator on omega and the omega row's
!! Laplacian:
!!   S_u ~ -P B_44^-1 B_42,    P = M_rho + B_24 + L
!!   M_rho = opz (v, rho0 R^3 w)        the inertia: B_22 ~ -M_rho B_44^-1 B_42
!!   B_24                               viscosity and advection of omega
!!   L = -(1/opz) B_23 X B_12           the line bending as a parallel diffusion
!!                                      on omega, with
!!   X = B_33^-1                        (vs_ideal) or
!!   X = (B_33 - B_31 B_11^-1 B_13)^-1  (vs_resistive).
!! The terms of lower order of S_u (B_21, B_25, B_26: kink and pressure) and
!! B_16 (eta'(T)) are left out.
!!
!! P is never formed: P w = f is the Schur complement for w of the sparse
!! vorticity system, whose blocks are blocks of the Jacobian:
!!   vs_ideal       [M_rho + B_24  B_23    ] [w]   [f]
!!                  [B_12          opz B_33] [y] = [0]
!!   vs_resistive   [M_rho + B_24  B_23      0   ] [w]   [f]
!!                  [B_12          opz B_33  B_31] [y] = [0]
!!                  [0             opz B_13  B_11] [z]   [0]
!!
!! One application y of the preconditioner to x, steps 2 and 3 being
!! S_u^-1 = -B_42^-1 B_44 P^-1:
!!   1. m_j = B_33^-1 x_j, m_omega = B_44^-1 x_omega      Chebyshev mass solves
!!   2. w from the vorticity system with f = x_u - B_23 m_j - B_24 m_omega
!!   3. omega = m_omega + w;  B_42 u = x_omega - B_44 omega  (the Laplacian)
!!   4. pair_psi (psi, j) = (x_psi - B_12 u, x_j)
!!   5. B_55 rho = x_rho - B_51 psi - B_52 u
!!   6. B_66 T = x_T - B_61 psi - B_63 j - B_62 u - B_65 rho
!!   7. the full Jacobian's boundary rows by back-substitution
!! Step 3's right-hand side x_omega - B_44 omega, not -B_44 w, holds the
!! omega row of the Jacobian to the accuracy of the Laplacian solve, whatever
!! the accuracy of the mass solve.
!!
!! vs_lumped: P assembled as one sparse scalar operator, X replaced by a
!! sparse symmetric matrix from the local dual basis (mod_petsc_pc_dual), and
!! solved in the place of the vorticity system; every other step as in
!! vs_ideal:
!!   P_lumped = M_rho + B_24 - (1/opz) B_23 X B_12
!! P_lumped is symmetric and positive definite where B_24 is (no flow):
!! B_23 = -B_12' up to the quadrature, so the line bending is the congruence
!! (1/opz) B_12' X B_12 with a symmetric positive definite X. The local dual
!! basis Z of a symmetric operator S is not symmetric; X is a symmetric form
!! of it (dual_symmetrize, -vs_dual_form):
!!   plain  X = Z                   (P_lumped then not symmetric)
!!   avg    X = (Z + Z') / 2        default; Z's pattern, couplings up to three
!!                                  nodes apart in P_lumped
!!   sym    X = Z + Z' - Z' S Z     X <= S^-1; five nodes apart in P_lumped
!! The line bending (-vs_bending) selects S:
!!   ideal      S = B_33, Z once (default; X ~ B_33^-1 as in vs_ideal)
!!   resistive  S = S_s, the symmetric resistive operator, at every build:
!!                S_s = B_33 + K_eta / opz + sym(B_31 Z_11 K_hyp)
!!              with sym(A) = (A + A') / 2, Z_11 the local dual basis of
!!              B_11 (3 x 3 node patches) and
!!                M_eta = theta dt (eta_T v, w / R)            the eta part of -B_13
!!                K_hyp = -B_13 - M_eta                         its hyper-resistive part
!!                K_eta = theta dt (eta_T grad v, grad w / R)   the self-adjoint form of
!!                                                              the eta part of
!!                                                              -B_31 B_11^-1 B_13
!!              M_eta and K_eta are assembled with the model's eta_T(T0) at
!!              every PC build (petsc_assemble_pc_matrices, the model's
!!              pc_elt_matrix_eta: model199 and model600; for model600
!!              eta_T(T0, rho0) of resistivity() with the Coulomb
!!              logarithm).
!!              S = B_33 - B_31 B_11^-1 B_13 (vs_resistive) is not symmetric
!!              with eta(T) (eta Delta is not self-adjoint) and sym(S) is then
!!              not positive definite; S_s is symmetric positive definite for
!!              any eta >= 0 and equals sym(B_33 - B_31 Z_11 B_13) for a
!!              constant eta up to Z_11. Its inverse is not local at the axis,
!!              where the hyper-resistivity is much larger than the mass:
!!              Z is the dual basis with the core patch at the axis
!!              (-vs_dual_core, default true; without it X and P_lumped are
!!              indefinite on fine grids or at large dt). The rows and columns
!!              of psi's and j's constrained DOFs (B_11's and B_33's
!!              Dirichlet or penalty rows, both boundary modes) leave K_hyp
!!              and K_eta, so that S_s has B_33's boundary rows. The first
!!              build checks the assembly against the Jacobian: rank 0
!!              prints |B_13 + M_eta + K_1|_F / |M_eta|_F on the interior
!!              rows, K_1 = theta dt (eta_num_T grad v, grad w) the
!!              hyper-resistive part of -B_13 assembled once with the
!!              model's coefficient (model199: the constant eta_num;
!!              model600: eta_num_T(T0, psi_N) of hyper_resistivity()),
!!              which catches any difference between the eta_T or eta_num_T
!!              of the PC and the model's (a WARNING above 1e-8). K_hyp
!!              itself stays -B_13 - M_eta, from the Jacobian.
!!              The dual basis reads the slot-diagonal part of S_s only (a
!!              state whose T0 has harmonics couples the slots in K_eta).
!! P_lumped's boundary rows lose their columns in both boundary modes (M_rho
!! keeps them), so that the stored matrix is the symmetric one; the first
!! build reports its skew part.
!!
!! THE AXIS SCHUR COMPLEMENT FORM AS(k) (-vs_dual_axis schur, resistive
!! bending; the study studies/spd_multigrid/axis). The core patch puts a dense
!! block of some 190 (n_tht ns)^2 entries into P_lumped, built in O(n_tht^3) on
!! the axis rank. AS(k) instead: A = the indices of S_s's variable on the
!! rings 0..k (mod_petsc_pc_dual: the shared axis value included), R the rest,
!!   X_R = the symmetric form of Z_R, Z_R the local dual basis of S_s(R, R)
!!         (the patches lose A's indices, A's blocks give no rows),
!!   W   = (1 - X_R S_d) E_A = [I_A; -X_R S_RA],    K = W' S_d W,
!!   X   = X_R + W K^-1 W',
!!   P   = P_s + L K^-1 R,   P_s = M_rho + B_24 - B_23 X_R B_12 / opz,
!!         L = -B_23 W / opz,  R = W' B_12,
!! S_d the slot-diagonal part of S_s (as the dual basis reads it: W, K, L and
!! R do not couple the slots of K). The block LDL' factorization gives S^-1 =
!! E_R' S_RR^-1 E_R + W S'^-1 W' with the exact W and S' = W' S W: AS replaces
!! S_RR^-1 by X_R and keeps the rest, so X >= 0 wherever X_R >= 0 (measured,
!! not proven, as the core's), K >= the exact Schur complement, P >= M_rho +
!! B_24 > 0, and nothing is approximated on A. P_s is P_lumped of X_R
!! (bounded rows), P is applied (mod_petsc_pc_axis_schur: a MATSHELL), never
!! assembled: K and its LU live on one rank, the owner of K's rows (the K
!! rank: the owner of A's smallest index, the axis), K's order slot-major,
!! inside a slot poloidal-major with the shared axis value last (no fill by
!! its dense row with the natural ordering, the default). Z_11 inside S_s then has a block per axis
!! node and no row of the axis value (-vs_dual_axis_z11 split, default):
!! S_s has no dense block either. L's rows and R's columns at P's
!! constrained rows are zero (P_s's are decoupled by the block solver). P's
!! solvers: the SPD multigrid (-vs_p_solver spdmg; levels P_g = P_s,g + L_g
!! K^-1 R_g, the axis block through its augmented matrix and over the rings
!! 0..k at least, the radial lines from ring k + 1 at the earliest (P_s has no
!! bending on A), the coarse level by the augmented LU; mod_petsc_pc_spdmg
!! spdmg_set_axis), or with
!! physics_pc_sf_pair_w = "lu" the exact solve: MUMPS on the augmented
!! matrix [[P_s, L], [R, -K]]. The ring GMG cannot take it (FATAL). k: the
!! study's rule k + 1 >= n_tht / 42 (tested up to n_tht 96), k >= 1 (AS(0) is
!! indefinite).
!!
!! V_PAR (vs_lumped on model600 with v_par: n_var 7, 7 = V). V's row couples
!! to psi (B_71, delta B . grad p0), u (B_72, with flow), rho, T and V; V's
!! column enters the u (B_27, with flow), rho (B_57) and T (B_67) rows only:
!! B_17, B_37, B_47, B_73 and B_74 are zeros of model600's element matrix. V
!! is solved after the Alfven group and before rho and T (the order of
!! vs_alfven's b0), on a symmetric positive definite operator of its own, the
!! sound wave as a congruence (the v_par study, studies/spd_multigrid/vpar):
!!   P_V = sym(B_77) + B_57' X B_57,   X = (Z + Z') / 2,
!!   Z the local dual basis of Q = W sym(B_55) W   (3 x 3 node patches, the
!!     core patch at the axis as the resistive bending's, -vs_vpar_core),
!!   W = diag((rho0 / T0)^1/2), the node's n = 0 values of the state for all
!!     DOFs and slots of the node.
!! At rest B_57 = B_76 and B_76 = -B_57' up to the quadrature: the (rho, T,
!! V) coupling is skew-adjoint in the energy inner product, and Q^-1 is the
!! isothermal compression's (rho0 / T0 the weight of rho), within the factor
!! gamma of the adiabatic one. P_V is symmetric by construction and positive
!! definite where sym(B_77) is and X >= 0, which is measured, not proven (the
!! core patch keeps X definite on fine grids). The rows of B_57 at rho's
!! constrained DOFs and its columns at V's leave the congruence, Q's
!! constrained rows and columns are decoupled (unit diagonal), so that P_V's
!! interior is the free system's operator; P_V's constrained rows are B_77's
!! diagonal alone, Dirichlet rows decoupled by the block solver like the
!! other blocks', the full Jacobian's boundary rows of V set by step 7. One
!! application has the step
!!   4b. P_V V = x_V - B_71 psi - B_72 u
!! between steps 4 and 5, and -B_57 V in step 5, -B_67 V in step 6. Left
!! out: V's feedback on the Alfven group (B_27, with flow) and the upper
!! couplings B_75, B_76. With flow B_77 and B_55 hold advection;
!! -vs_vpar_flow adv solves
!!   P_V^A = B_77 + B_57' Z_A B_57,   Z_A the plain dual basis of W B_55 W,
!! by FGMRES to the inner rtol with P_V's V-cycle (LU: P_V^A's own LU). The
!! first build reports |B_57 - B_76|_F and |B_76 + B_57'|_F against |B_57|_F
!! on the interior rows and columns, and P_V's skew part.
!! P_V's solver (-vs_pv_solver): the backend of physics_pc_sf_vpar (lu, or
!! gmg: the ring GMG, instance 6, flux-surface rings, as vs_alfven's b0), or
!! the second instance of the SPD multigrid (mod_petsc_pc_spdmg, options
!! -vs_pvmg_*, radial lines and wedges by default), or BoomerAMG
!! (mod_petsc_pc_hypre, options -sf_amg6_*).
!!
!! PETSc options of vs_lumped (read once, at the first assembly):
!!   -vs_bending ideal | resistive     the line bending (default ideal)
!!   -vs_dual_form plain | avg | sym   the form of X (default avg)
!!   -vs_dual_core <bool>              the core patch of the resistive bending
!!                                     (default true)
!!   -vs_dual_axis core | schur        the resistive bending's axis treatment:
!!                                     the core patch (default, -vs_dual_core)
!!                                     or the axis Schur complement form AS(k)
!!   -vs_dual_axis_k <k>               AS(k)'s rings 0..k (default
!!                                     max(1, ceil(n_tht / 42) - 1))
!!   -vs_dual_axis_z11 split | node    AS: Z_11 with a block per axis node and
!!                                     no row of the axis value (default), or
!!                                     today's axis node block
!!   -vs_dual_axis_ordering <ord>      AS: the ordering of K's LU (natural = K's
!!                                     own order, the default: m4 405 k
!!                                     entries against nd's 453 k; nd, rcm,
!!                                     qmd, 1wd)
!!   -vs_axis_view <prefix>            AS: the first build's K, L, R, W and A's
!!                                     indices in K's order (<prefix>_K.petsc,
!!                                     _L, _R, _W, _idx: a Vec), PETSc binary
!!   -vs_pl_view <file>                P_lumped of every build, PETSc binary (the
!!                                     first build creates the file, later
!!                                     builds append)
!!   -vs_ss_view <file>                S_s of every build, likewise
!!   -vs_blk_view <prefix>             the solved blocks pair_psi, B_55, B_66 of every
!!                                     build and the Laplacian (first build) to
!!                                     <prefix>_{pj,rho,T,lap}.petsc, likewise
!!   -vs_mem_report                  the first build reports every vs_lumped
!!                                     matrix (entries, rows, longest row, exact
!!                                     zeros, entries between |n| families) and
!!                                     the RSS after it, and the SPD-MG levels
!!                                     ([VS mem] lines; collective, off)
!!   -vs_dual_mat_view                 every dual basis (mod_petsc_pc_dual)
!!   -vs_p_solver default | spdmg | hypre | gmg
!!                                     P_lumped's solver: the backend of
!!                                     physics_pc_sf_pair_w (default: the ring
!!                                     GMG with FGMRES, or LU), or the ring-free
!!                                     SPD multigrid of mod_petsc_pc_spdmg (its
!!                                     options -vs_pmg_*), or BoomerAMG
!!                                     (mod_petsc_pc_hypre, -sf_amg1_*), or the
!!                                     ring GMG whatever physics_pc_sf_pair_w
!!                                     (gmg: that one then serves the Laplacian)
!! and with v_par:
!!   -vs_pv_solver default | spdmg | hypre
!!                                     P_V's solver: the backend of
!!                                     physics_pc_sf_vpar (default), or the SPD
!!                                     multigrid's P_V instance (-vs_pvmg_*), or
!!                                     BoomerAMG (-sf_amg6_*)
!!   -vs_vpar_flow sym | adv           the solve on P_V (default), or on P_V^A
!!                                     with P_V's V-cycle
!!   -vs_vpar_core <bool>              Z with the core patch (default true)
!!   -vs_pv_view <file>                P_V of every build, as -vs_pl_view
!!
!! Boundary rows: M_rho and the Laplacian's boundary diagonal D_bc carry
!! Dirichlet rows at u's constrained DOFs (1, or the penalty with penalty
!! rows), so that the vorticity system and the Laplacian B_42 + D_bc have
!! Dirichlet rows wherever the Jacobian has them. Steps 2 and 3 do not give
!! the boundary values of u and omega the Jacobian's (u_b follows from
!! omega's and u's residuals together): step 7 sets them, for penalty and
!! for eliminated boundary rows alike.
module mod_petsc_pc_vs
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var, var_Vpar
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, physics_pc_method_id, PC_METHOD_VS_RESISTIVE, &
       PC_METHOD_VS_LUMPED, pcev_extract, pcev_fact_pj, pcev_fact_rhot, pcev_fact_vort, pcev_fact_lap, pcev_apply, &
       pcev_solve_pj, pcev_solve_rhot, pcev_solve_vort, pcev_solve_lap, pcev_solve_mass, pcev_asm_pl, pcev_asm_pv, &
       pcev_fact_pv, pcev_solve_pv
  use mod_petsc_pc_blocks, only: penalty_rows_t, full_penalty_rows, penalty_backsub, penalty_row_mask
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_ops, only: sf_op_t, sfo_layout, sfo_build, sfo_fill, sfo_vars_in, sfo_vars_out, &
       sfo_comm, sfo_define, sfo_b65_used, sfo_destroy
  use mod_petsc_raw_csr, only: blockmv_attach
  use mod_petsc_pc_mass_cheb, only: mass_cheb_t, mass_cheb_setup, mass_cheb_solve
  use mod_petsc_pc_dual, only: dual_basis, dual_symmetrize, DUAL_PLAIN, DUAL_AVG, DUAL_SYM
  use mod_petsc_pc_spdmg, only: SPDMG_PV, SPDMG_P, spdmg_set_axis
  use mod_petsc_pc_axis_schur, only: axk_set, axk_lu_nnz, axk_rank, axs_shell, axs_axis_mult, axa_new, axa_setup, &
       axa_set_pc, axa_factor_nnz
  implicit none
  private

  logical, save :: vs_init_done = .false.
  logical, save :: vs_first     = .true.
  logical, save :: resistive    = .false.     !< the resistive line bending
  logical, save :: lumped       = .false.     !< P as the lumped scalar operator
  integer, save :: dual_form    = DUAL_AVG    !< vs_lumped: the form of X (-vs_dual_form)
  logical, save :: res_dual     = .false.     !< vs_lumped: the resistive bending S_s (-vs_bending)
  logical, save :: res_core     = .true.      !< and its core patch at the axis (-vs_dual_core)
  character(len=512), save :: pl_view = "", ss_view = ""   !< -vs_pl_view, -vs_ss_view
  character(len=512), save :: blk_view = ""                !< -vs_blk_view <prefix>: the solved blocks
  logical, save :: famode       = .false.     !< mode split
  logical, save :: harm_all     = .true.      !< physics_pc_sf_harm_couple
  integer, save :: comm_g = MPI_COMM_NULL     !< the full system's communicator
  !> this rank's wall time in vs_apply and in its gather / scatter since the
  !! last report (the mode split's per-family report)
  real*8, save  :: t_apply = 0.d0, t_scat = 0.d0

  type(block_solver_t), save :: slv_vort, slv_lap, slv_pj, slv_rho, slv_T
  integer, save :: bk_vort = SF_GMG, bk_pj = SF_GMG, bk_rho = SF_GMG, bk_T = SF_GMG
  !> vs_lumped: the backend of P_lumped's solve (-vs_p_solver): bk_vort
  !! (physics_pc_sf_pair_w, default), SF_SPDMG (mod_petsc_pc_spdmg),
  !! SF_HYPRE (mod_petsc_pc_hypre) or SF_GMG
  integer, save :: bk_pl = SF_GMG

  !> B-norm error bound of the Chebyshev mass solves
  real*8, parameter :: MASS_TARGET = 1.d-2
  !> the self-check of the resistive bending: largest |B_13 + M_eta + K_1| / |M_eta| without a warning
  real*8, parameter :: ETA_CHECK_TOL = 1.d-8

  !> The operators: the vorticity system, the Laplacian B_42 + D_bc, pair_psi,
  !! B_55, B_66, the masses B_33 and B_44, and the coupling blocks o_cb(CB_ij) = B_ij.
  type(sf_op_t), save :: o_vort, o_lap, o_pj, o_rho, o_T, o_mj, o_mw
  integer, parameter :: CB_12 = 1, CB_23 = 2, CB_24 = 3, CB_51 = 4, CB_52 = 5, CB_61 = 6, CB_62 = 7, &
                        CB_63 = 8, CB_65 = 9
  integer, parameter :: NCB = 9
  type(sf_op_t), save :: o_cb(NCB)
  !> vs_lumped: X (z_j; once, or at every build with the resistive bending),
  !! the product B_23 X B_12 and P_lumped (o_pl%mat, its pattern fixed by the
  !! first build)
  Mat, save :: z_j, t_bzb
  type(sf_op_t), save :: o_pl
  !> P_lumped is t_bzb itself (assemble_lumped)
  logical, save :: pl_merged = .false.
  !> vs_lumped with the resistive bending: o_rb(RB_ij) = B_ij; M_eta and K_eta
  !! in the operators' layout (W sources alone); Z_11, K_hyp, the product
  !! T = B_31 Z_11 K_hyp and its transpose, S_s; the masks of psi's and j's
  !! constrained DOFs (0 there, 1 elsewhere)
  integer, parameter :: RB_11 = 1, RB_13 = 2, RB_31 = 3
  integer, parameter :: NRB = 3
  type(sf_op_t), save :: o_rb(NRB), o_meta, o_keta
  Mat, save :: z_11, k_hyp, t_s, t_st, s_res
  Vec, save :: mask_psi, mask_j
  !> the axis treatment of the resistive bending (-vs_dual_axis; module
  !! header): the core patch, or the axis Schur complement form AS(k) with
  !! its options; A's indices in K's order (ax_idx); the slot-diagonal part of
  !! S_s (s_dg), E_A, S_d E_A, X_R S_d E_A, W, W' S_d W and its transpose, K,
  !! L, R, P as a shell; the mask of P's constrained rows (0 there); K's
  !! object (mod_petsc_pc_axis_schur) and the augmented matrix of the exact
  !! P solve
  integer, parameter :: AX_CORE = 1, AX_SCHUR = 2
  integer, save :: res_axis = AX_CORE
  logical, save :: ax_on = .false.            !< lumped, resistive and AX_SCHUR
  integer, save :: ax_k = -1                  !< -vs_dual_axis_k
  logical, save :: z11_split = .false.        !< -vs_dual_axis_z11 split
  character(len=16), save :: ax_order = "natural"  !< -vs_dual_axis_ordering
  character(len=512), save :: ax_view = ""    !< -vs_axis_view <prefix>
  integer, allocatable, save :: ax_idx(:)
  Mat, save :: s_dg, e_a, ax_sa, ax_xsa, ax_w, ax_k0, ax_kt, ax_kk, ax_l, ax_r, p_shell
  Vec, save :: mask_u
  integer, parameter :: AX_IK = 1             !< P_lumped's K object
  integer, save :: ax_ia = 0                  !< the augmented matrix of the exact P solve
  !> B_65 (T row, rho column) vanishes in model199 and is applied only where it does not
  logical, save :: use_b65 = .false.
  type(mass_cheb_t), save :: mc_j, mc_w
  type(penalty_rows_t), save :: pen           !< the full Jacobian's penalty rows

  !> v_par (vs_lumped, model600 with v_par; module header): P_V's solver and
  !! options, B_77 and the couplings o_vc(VC_ij) = B_ij, P_V (o_pv%mat) and
  !! with -vs_vpar_flow adv P_V^A (pva); the pieces kept for the refills: the
  !! masks of rho's and V's constrained DOFs (1 - mask: cmask), W times rho's
  !! mask, Q = W sym(B_55) W and B_55', X, B_57 masked, the congruence
  !! B_57' X B_57, B_77'; for adv Q_A = W B_55 W, Z_A and B_57' Z_A B_57
  logical, save :: vpar     = .false.
  logical, save :: pv_adv   = .false.         !< -vs_vpar_flow adv
  logical, save :: pv_core  = .true.          !< -vs_vpar_core
  integer, save :: bk_pv    = SF_GMG          !< physics_pc_sf_vpar, or SF_SPDMG (-vs_pv_solver)
  character(len=512), save :: pv_view = ""    !< -vs_pv_view
  type(block_solver_t), save :: slv_pv
  integer, parameter :: VC_57 = 1, VC_67 = 2, VC_71 = 3, VC_72 = 4
  integer, parameter :: NVC = 4
  type(sf_op_t), save :: o_v77, o_vc(NVC), o_pv
  Vec, save :: mask_rho, cmask_rho, mask_v, cmask_v, w_rho
  Mat, save :: q_pv, b55t, x_pv, b57m, t_pv, b77t, pva, qa_pv, za_pv, ta_pv
  !> floors of rho0 and T0 in W against their maxima over the grid
  real*8, parameter :: W_FLOOR = 1.d-6

  !> the variables' pieces of x and y: (psi, u, j, omega, rho, T[, V]); V's
  !! is NSV with v_par (vs_init checks var_Vpar), T's otherwise, never read
  integer, parameter :: NSV = merge(7, 6, var_Vpar > 0)
  integer, parameter :: SV_V = NSV
  Vec, save :: sv_x(NSV), sv_y(NSV)
  Vec, save :: rhs_V, sol_V, rhs_PJ, sol_PJ
  Vec, save :: m_j, w1, w2, w3
  Vec, save :: zv                              !< zero field
  logical, save :: vecs_ready = .false.

  !> -log_view: the stages of the first build and of the later builds
  !! (refills), and events of vs_lumped's assembly (the resistive bending's
  !! S_s and X, the product B_23 X B_12 and P_lumped, the first build's
  !! checks)
  PetscLogStage, save :: vst_build1 = -1, vst_buildn = -1
  PetscLogEvent, save :: vev_resbend = -1, vev_plprod = -1, vev_check = -1
  !> AS(k): W and K (VS_AxisK), L and R (VS_AxisLR)
  PetscLogEvent, save :: vev_axk = -1, vev_axlr = -1
  logical, save :: vlog_ready = .false.
  !> -vs_mem_report: the first build reports each vs_lumped matrix (entries,
  !! rows, longest row; sum, largest rank; exact zeros, entries between |n|
  !! families) and the RSS after it
  logical, save :: mem_report = .false.

  public :: vs_init, vs_build, vs_apply, vs_report, vs_wants_eta

contains

  !> The stages and events above, once.
  subroutine vs_register_log()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    if (vlog_ready) return
    call PetscLogStageRegister("VS build first", vst_build1, ierr)
    call PetscLogStageRegister("VS build refill", vst_buildn, ierr)
    call PetscLogEventRegister("VS_ResBend", cid, vev_resbend, ierr)
    call PetscLogEventRegister("VS_PLProd",  cid, vev_plprod,  ierr)
    call PetscLogEventRegister("VS_Check",   cid, vev_check,   ierr)
    call PetscLogEventRegister("VS_AxisK",   cid, vev_axk,     ierr)
    call PetscLogEventRegister("VS_AxisLR",  cid, vev_axlr,    ierr)
    vlog_ready = .true.
  end subroutine vs_register_log

  !> Resolve the namelist and the PETSc options once and refuse
  !! configurations the method cannot run. Called before the first assembly
  !! of M_rho.
  subroutine vs_init(my_id)
    use phys_module, only: physics_pc_sf_pair_psi, physics_pc_sf_pair_w, physics_pc_sf_rho, physics_pc_sf_T, &
                           physics_pc_sf_rtol, physics_pc_sf_harm_couple, physics_pc_sf_mode_split, &
                           physics_pc_sf_vpar, physics_pc_sf_vpar_stage, n_tht
    use mod_parameters, only: n_tor, var_T
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr
    PetscBool :: is_set, flag
    PetscInt :: iv
    character(len=16) :: form
    character(len=48) :: pname
    character(len=*), parameter :: P = "[Physics PC]   "
    character(len=*), parameter :: form_text(0:2) = [character(len=24) :: "Z", "(Z + Z') / 2", "Z + Z' - Z' S Z"]

    if (vs_init_done) return

    resistive = (physics_pc_method_id() == PC_METHOD_VS_RESISTIVE)
    lumped    = (physics_pc_method_id() == PC_METHOD_VS_LUMPED)
    if (lumped) then
      form = "avg"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_form", form, is_set, ierr)
      select case (trim(form))
      case ("plain"); dual_form = DUAL_PLAIN
      case ("avg");   dual_form = DUAL_AVG
      case ("sym");   dual_form = DUAL_SYM
      case default
        call fatal("-vs_dual_form must be plain | avg | sym, got '"//trim(form)//"'")
      end select
      form = "ideal"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_bending", form, is_set, ierr)
      select case (trim(form))
      case ("ideal");     res_dual = .false.
      case ("resistive"); res_dual = .true.
      case default
        call fatal("-vs_bending must be ideal | resistive, got '"//trim(form)//"'")
      end select
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_core", flag, is_set, ierr)
      if (is_set) res_core = flag
      form = "core"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_axis", form, is_set, ierr)
      select case (trim(form))
      case ("core");  res_axis = AX_CORE
      case ("schur"); res_axis = AX_SCHUR
      case default
        call fatal("-vs_dual_axis must be core | schur, got '"//trim(form)//"'")
      end select
      if (res_axis == AX_SCHUR) then
        if (.not. res_dual) call fatal("-vs_dual_axis schur needs -vs_bending resistive")
        ax_on = .true.
        ax_k = max(1, (n_tht + 41) / 42 - 1)
        call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_axis_k", iv, is_set, ierr)
        if (is_set) ax_k = int(iv)
        if (ax_k < 0) call fatal("-vs_dual_axis_k must be >= 0")
        form = "split"
        call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_axis_z11", form, is_set, ierr)
        select case (trim(form))
        case ("split"); z11_split = .true.
        case ("node");  z11_split = .false.
        case default
          call fatal("-vs_dual_axis_z11 must be split | node, got '"//trim(form)//"'")
        end select
        call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_axis_ordering", ax_order, &
                                   is_set, ierr)
        select case (trim(ax_order))
        case ("natural", "nd", "rcm", "qmd", "1wd")
        case default
          call fatal("-vs_dual_axis_ordering must be natural | nd | rcm | qmd | 1wd, got '"//trim(ax_order)//"'")
        end select
        call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_axis_view", ax_view, is_set, ierr)
        if (.not. is_set) ax_view = ""
      endif
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pl_view", pl_view, is_set, ierr)
      if (.not. is_set) pl_view = ""
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_ss_view", ss_view, is_set, ierr)
      if (.not. is_set) ss_view = ""
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_blk_view", blk_view, is_set, ierr)
      if (.not. is_set) blk_view = ""
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mem_report", flag, is_set, ierr)
      if (is_set) mem_report = flag
    endif
    bk_pj   = sf_backend(physics_pc_sf_pair_psi, "physics_pc_sf_pair_psi", my_id)
    bk_vort = sf_backend(physics_pc_sf_pair_w,   "physics_pc_sf_pair_w",   my_id)
    bk_rho  = sf_backend(physics_pc_sf_rho,      "physics_pc_sf_rho",      my_id)
    bk_T    = sf_backend(physics_pc_sf_T,        "physics_pc_sf_T",        my_id)
    bk_pl   = bk_vort
    if (lumped) then
      form = "default"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_p_solver", form, is_set, ierr)
      select case (trim(form))
      case ("default");  bk_pl = bk_vort
      case ("spdmg");    bk_pl = SF_SPDMG
      case ("hypre");    bk_pl = SF_HYPRE
      case ("gmg");      bk_pl = SF_GMG
      case default
        call fatal("-vs_p_solver must be default | spdmg | hypre | gmg, got '"//trim(form)//"'")
      end select
      if (ax_on .and. (bk_pl == SF_GMG .or. bk_pl == SF_HYPRE)) call fatal("-vs_dual_axis schur: P = P_s + "// &
        "L K^-1 R is applied, not assembled, and the ring GMG and BoomerAMG cannot take it: use -vs_p_solver "// &
        "spdmg, or physics_pc_sf_pair_w = lu (the exact solve)")
    endif
    pname = physics_pc_sf_pair_w
    if (bk_pl == SF_SPDMG) pname = "spdmg (ring-free SPD multigrid)"
    if (bk_pl == SF_HYPRE) pname = "hypre (BoomerAMG, -sf_amg1_*)"
    if (bk_pl == SF_GMG .and. bk_vort /= SF_GMG) pname = "gmg (-vs_p_solver gmg)"

    if (jorek_model /= 199 .and. jorek_model /= 600) &
      call fatal("physics_pc_method = vs_ideal | vs_resistive | vs_lumped supports model199 and model600 "// &
                 "only (model183's psi row has the mass B_11, not opz B_33)")
    ! v_par: vs_lumped on model600 (V = 7 after (psi, u, zj, w, rho, T))
    vpar = lumped .and. jorek_model == 600 .and. n_var == 7 .and. var_Vpar == 7
    if (.not. (n_var == 6 .or. vpar) .or. var_T /= 6) &
      call fatal("physics_pc_method = vs_ideal | vs_resistive | vs_lumped supports (psi, u, zj, w, rho, T), "// &
                 "and vs_lumped also with v_par on model600, only: build without with_TiTe, with_neutrals and "// &
                 "with_impurities (and without v_par for vs_ideal and vs_resistive)")
    if (vpar) then
      bk_pv = sf_backend(physics_pc_sf_vpar, "physics_pc_sf_vpar", my_id)
      form = "default"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pv_solver", form, is_set, ierr)
      select case (trim(form))
      case ("default")
      case ("spdmg");    bk_pv = SF_SPDMG
      case ("hypre");    bk_pv = SF_HYPRE
      case default
        call fatal("-vs_pv_solver must be default | spdmg | hypre, got '"//trim(form)//"'")
      end select
      form = "sym"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_vpar_flow", form, is_set, ierr)
      select case (trim(form))
      case ("sym");  pv_adv = .false.
      case ("adv");  pv_adv = .true.
      case default
        call fatal("-vs_vpar_flow must be sym | adv, got '"//trim(form)//"'")
      end select
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_vpar_core", flag, is_set, ierr)
      if (is_set) pv_core = flag
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pv_view", pv_view, is_set, ierr)
      if (.not. is_set) pv_view = ""
    endif
    if (mod(n_tor, 2) /= 1) call fatal("physics_pc_method = vs_ideal | vs_resistive | vs_lumped needs n_tor "// &
                                       "odd (n = 0 plus cos/sin pairs)")
    ! -vs_bending resistive: the models above assemble M_eta, K_eta and K_1
    ! with their own eta_T and eta_num_T (pc_elt_matrix_eta); model600's
    ! without its extensions, which n_var 6 excludes

    harm_all    = physics_pc_sf_harm_couple
    sf_gmg_harm = merge(SF_GMG_HARM_ALL, SF_GMG_HARM_PAIR, harm_all)
    famode      = physics_pc_sf_mode_split .and. .not. harm_all

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ========= vorticity-space Schur (VS) preconditioner ========="
      if (resistive) then
        write(*,'(A)') P//"line bending: resistive, (B_33 - B_31 B_11^-1 B_13)^-1"
        write(*,'(A,A)') P//"vorticity: ", trim(physics_pc_sf_pair_w)//" (omega, y, z; M_rho = inertia)"
      else if (lumped .and. res_dual) then
        write(*,'(A)') P//"line bending: resistive, S_s^-1 by its local dual basis Z as X = "// &
                       trim(form_text(dual_form))//", S_s = B_33 + K_eta / opz + sym(B_31 Z_11 K_hyp)"
        if (ax_on) then
          pname = "with the axis node block"
          if (z11_split) pname = "split at the axis"
          write(*,'(A,I0,A,A,A)') P//"  axis Schur complement form AS(", ax_k, ") (-vs_dual_axis schur): X = X_R + "// &
            "W K^-1 W' on the rings 0..k, P = P_s + L K^-1 R applied; Z_11 ", trim(pname), &
            ", K's LU ordering "//trim(ax_order)
          pname = "spdmg (ring-free SPD multigrid)"
          if (bk_pl /= SF_SPDMG) pname = "lu (MUMPS on [[P_s, L], [R, -K]])"
        else if (res_core) then
          write(*,'(A)') P//"  Z with one core patch at the axis (-vs_dual_core)"
        else
          write(*,'(A)') P//"  Z without the core patch at the axis (-vs_dual_core false)"
        endif
        write(*,'(A,A)') P//"vorticity: ", trim(pname)//" (P_lumped = M_rho + B_24 - B_23 X B_12 / opz)"
      else if (lumped) then
        write(*,'(A)') P//"line bending: ideal, B_33^-1 by its local dual basis Z as X = "// &
                       trim(form_text(dual_form))
        write(*,'(A,A)') P//"vorticity: ", trim(pname)//" (P_lumped = M_rho + B_24 - B_23 X B_12 / opz)"
      else
        write(*,'(A)') P//"line bending: ideal, B_33^-1"
        write(*,'(A,A)') P//"vorticity: ", trim(physics_pc_sf_pair_w)//" (omega, y; M_rho = inertia)"
      endif
      write(*,'(A,A)') P//"Laplacian: ", trim(physics_pc_sf_pair_w)//" (B_42, once)"
      write(*,'(A,A)') P//"pair_psi : ", trim(physics_pc_sf_pair_psi)
      write(*,'(A,A)') P//"rho / T  : ", trim(physics_pc_sf_rho)//" / "//trim(physics_pc_sf_T)
      if (vpar) then
        if (bk_pv == SF_SPDMG) then
          write(*,'(A)') P//"v_par    : spdmg (ring-free SPD multigrid, -vs_pvmg_*), after pair_psi, before rho and T"
        else if (bk_pv == SF_HYPRE) then
          write(*,'(A)') P//"v_par    : hypre (BoomerAMG, -sf_amg6_*), after pair_psi, before rho and T"
        else
          write(*,'(A)') P//"v_par    : "//trim(physics_pc_sf_vpar)//", after pair_psi, before rho and T"
        endif
        if (pv_core) then
          form = "with"
        else
          form = "without"
        endif
        write(*,'(A)') P//"  P_V = sym(B_77) + B_57' X B_57, X = (Z + Z') / 2, Z the local dual basis of "// &
                       "W sym(B_55) W, W = (rho0 / T0)^1/2, "//trim(form)//" the core patch (-vs_vpar_core)"
        if (pv_adv) write(*,'(A)') P//"  -vs_vpar_flow adv: FGMRES on P_V^A = B_77 + B_57' Z_A B_57 (Z_A of "// &
                                   "W B_55 W) with P_V's preconditioner"
        write(*,'(A)') P//"  P_V V = x_V - B_71 psi - B_72 u, then B_57 V, B_67 V in the rho, T rows; left out: "// &
                       "B_27, B_75, B_76 (physics_pc_sf_vpar_stage = "//trim(physics_pc_sf_vpar_stage)// &
                       " has no effect)"
      endif
      write(*,'(A,ES9.2)') P//"masses B_33, B_44: Chebyshev, B-norm error bound ", MASS_TARGET
      write(*,'(A,ES9.2)') P//"inner rtol: ", physics_pc_sf_rtol
      if (harm_all) then
        write(*,'(A)') P//"cross-|n|: all kept, all slots per GMG smoother block"
      else
        write(*,'(A)') P//"cross-|n|: none, |n|-diagonal blocks, cos / sin per GMG smoother block"
      endif
      if (famode) then
        write(*,'(A)') P//"layout   : one |n| family per rank group (physics_pc_sf_mode_split)"
      else
        write(*,'(A)') P//"layout   : global operators"
      endif
    endif

    vs_init_done = .true.

  contains

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') P//"FATAL: ", trim(msg)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine vs_init

  !> Does the method need M_eta and K_eta from the element assembly
  !! (vs_lumped with -vs_bending resistive)? After vs_init.
  logical function vs_wants_eta()
    vs_wants_eta = vs_init_done .and. lumped .and. res_dual
  end function vs_wants_eta

  !> Build the preconditioner from the assembled Jacobian A_full, at every PC
  !! rebuild. The operators' patterns are fixed by the first build; later
  !! builds refill the operators whose values change. The masses and the
  !! Laplacian depend on the grid only: extracted and set up once.
  subroutine vs_build(A_full, comm, my_id)
    use phys_module, only: physics_pc_sf_rtol, eliminate_boundary_dofs
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr
    logical :: first
    real*8  :: rtol
    integer :: k, nno, comm_s
    Mat :: wm(1), wd(1)

    call vs_init(my_id)
    first  = vs_first
    comm_g = comm
    rtol   = physics_pc_sf_rtol
    call vs_register_log()
    if (first) then
      call PetscLogStagePush(vst_build1, ierr)
    else
      call PetscLogStagePush(vst_buildn, ierr)
    endif

    if (first) then
      if (.not. (g_ctx%m_rho_ready .and. g_ctx%d_bc_ready)) &
        call build_fatal("M_rho / D_bc were never assembled (petsc_assemble_pc_matrices)")
      if (res_dual .and. .not. g_ctx%eta_ready) &
        call build_fatal("M_eta / K_eta were never assembled (petsc_assemble_pc_matrices)")
      call sfo_layout(A_full, comm, famode, harm_all, my_id)
      call define_operators()
    endif
    comm_s = sfo_comm

    ! the W sources: M_rho in the vorticity system (unread elsewhere), D_bc in the Laplacian
    wm = g_ctx%M_rho
    wd = g_ctx%D_bc
    call PetscLogEventBegin(pcev_extract, ierr)
    call build_or_fill(o_vort, wm); call build_or_fill(o_pj, wm)
    call build_or_fill(o_rho, wm);  call build_or_fill(o_T, wm)
    do k = 1, NCB
      call build_or_fill(o_cb(k), wm)
    enddo
    if (res_dual) then
      do k = 1, NRB
        call build_or_fill(o_rb(k), wm)
      enddo
      call build_or_fill(o_meta, [g_ctx%M_eta])
      call build_or_fill(o_keta, [g_ctx%K_eta])
    endif
    if (vpar) then
      call build_or_fill(o_v77, wm)
      do k = 1, NVC
        call build_or_fill(o_vc(k), wm)
      enddo
    endif
    if (first) then
      call sfo_build(o_lap, A_full, wd, sf_opz(), my_id)
      call sfo_build(o_mj, A_full, wm, sf_opz(), my_id)
      call sfo_build(o_mw, A_full, wm, sf_opz(), my_id)
    endif
    call PetscLogEventEnd(pcev_extract, ierr)
    if (lumped .and. mem_report .and. first) then
      call mem_line("M_rho + B_24", o_vort%mat)
      call mem_line("B_23", o_cb(CB_23)%mat)
      call mem_line("B_12", o_cb(CB_12)%mat)
      call mem_line("B_33", o_mj%mat)
      if (res_dual) then
        call mem_line("B_11", o_rb(RB_11)%mat)
        call mem_line("B_13", o_rb(RB_13)%mat)
        call mem_line("B_31", o_rb(RB_31)%mat)
        call mem_line("M_eta", o_meta%mat)
        call mem_line("K_eta", o_keta%mat)
      endif
      if (vpar) then
        call mem_line("B_55", o_rho%mat)
        call mem_line("B_57", o_vc(VC_57)%mat)
        call mem_line("B_77", o_v77%mat)
      endif
      call mem_line("the operators extracted")
    endif
    if (first) then
      ! on the operators' communicator: B_65 is refilled at every build, the
      ! flag only skips its product, so the families need not agree
      use_b65 = sfo_b65_used(o_cb(CB_65), comm_s)
      nno = 0
      do k = 1, NCB
        if (.not. blockmv_attach(o_cb(k)%mat, int(pc_ntor))) nno = nno + 1
      enddo
      if (vpar) then
        do k = 1, NVC
          if (.not. blockmv_attach(o_vc(k)%mat, int(pc_ntor))) nno = nno + 1
        enddo
      endif
      if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   VS: WARNING ", nno, &
        " coupling block(s) not AIJ, left on PETSc's matvec"
    endif
    if (lumped) then
      call PetscLogEventBegin(pcev_asm_pl, ierr)
      call assemble_lumped()
      call PetscLogEventEnd(pcev_asm_pl, ierr)
    endif
    ! P_V from B_55 before its penalty rows are decoupled for rho's solve
    if (vpar) then
      call PetscLogEventBegin(pcev_asm_pv, ierr)
      call assemble_pv()
      call PetscLogEventEnd(pcev_asm_pv, ierr)
    endif
    call physics_pc_mem("VS build: operators filled", my_id)

    ! the Jacobian's boundary rows, penalty or eliminated (diagonal only), are
    ! back-substituted after every apply in both modes: steps 2 and 3 give u
    ! and omega there from each other's residual, which would put an
    ! eigenvalue -a_uu a_ww of A P^-1 on the eliminated rows
    call full_penalty_rows(pen, A_full, comm, my_id, first)
    ! penalty rows as Dirichlet rows of the solved blocks, before the scaling.
    ! P_lumped in both boundary modes: M_rho's Dirichlet rows keep their
    ! columns, which meet zero boundary values with eliminated boundary rows
    ! but leave the matrix not symmetric
    if (lumped) then
      call sf_solver_penalty_rows(slv_vort, o_pl, comm_s, my_id)
      if (first) then
        call PetscLogEventBegin(vev_check, ierr)
        if (ax_on) then
          call report_skew(o_pl%mat, "P_s")
          call report_skew_applied()
        else
          call report_skew(o_pl%mat, "P_lumped")
        endif
        call PetscLogEventEnd(vev_check, ierr)
      endif
      if (ax_on) then
        call view_build(o_pl%mat, pl_view, "P_s")
        if (first .and. len_trim(ax_view) > 0) call axis_view()
      else
        call view_build(o_pl%mat, pl_view, "P_lumped")
      endif
    endif
    ! P_V likewise: its constrained rows are B_77's diagonal alone (P_V^A's
    ! take P_V's decoupled ones)
    if (vpar) then
      call sf_solver_penalty_rows(slv_pv, o_pv, comm_s, my_id)
      if (pv_adv) call pva_boundary_rows()
      if (first) then
        call PetscLogEventBegin(vev_check, ierr)
        call report_skew(o_pv%mat, "P_V")
        if (pv_adv) call report_skew(pva, "P_V^A")
        call PetscLogEventEnd(vev_check, ierr)
      endif
      call view_build(o_pv%mat, pv_view, "P_V")
    endif
    if (.not. eliminate_boundary_dofs) then
      if (.not. lumped) call sf_solver_penalty_rows(slv_vort, o_vort, comm_s, my_id)
      call sf_solver_penalty_rows(slv_pj, o_pj, comm_s, my_id)
      call sf_solver_penalty_rows(slv_rho, o_rho, comm_s, my_id)
      call sf_solver_penalty_rows(slv_T, o_T, comm_s, my_id)
      if (first) call sf_solver_penalty_rows(slv_lap, o_lap, comm_s, my_id)
    endif
    if (.not. lumped) call sf_solver_block_scale(slv_vort, o_vort, comm_s, my_id)
    call sf_solver_block_scale(slv_pj, o_pj, comm_s, my_id)
    ! -vs_blk_view <prefix>: the solved blocks as their solvers get them
    if (len_trim(blk_view) > 0) then
      call view_build(o_pj%mat, trim(blk_view)//"_pj.petsc", "pair_psi")
      call view_build(o_rho%mat, trim(blk_view)//"_rho.petsc", "B_55")
      call view_build(o_T%mat, trim(blk_view)//"_T.petsc", "B_66")
      if (first) call view_build(o_lap%mat, trim(blk_view)//"_lap.petsc", "Laplacian")
    endif

    call PetscLogEventBegin(pcev_fact_pj, ierr)
    call sf_solver_setup(slv_pj, o_pj%mat, bk_pj, "pair_psi", comm_s, my_id, rtol, gmg_inst=2, nfields=2, &
                         smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS, &
                         pre0=SF_PJ_PRE0, post0=SF_PJ_POST0, nsmooth_c=SF_PJ_NSC, &
                         axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    call PetscLogEventEnd(pcev_fact_pj, ierr)

    ! the vorticity system as pair_w: line bending along the flux surfaces;
    ! P_lumped, one field, as T: a diffusion along B
    call PetscLogEventBegin(pcev_fact_vort, ierr)
    if (lumped .and. ax_on) then
      call axis_p_solver()
    else if (lumped) then
      call sf_solver_setup(slv_vort, o_pl%mat, bk_pl, "P_lumped", comm_s, my_id, rtol, gmg_inst=1, &
                           nfields=1, smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS)
    else
      call sf_solver_setup(slv_vort, o_vort%mat, bk_vort, "vorticity", comm_s, my_id, rtol, gmg_inst=1, &
                           nfields=o_vort%nf, smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                           pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC, &
                           axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    endif
    call PetscLogEventEnd(pcev_fact_vort, ierr)

    ! P_V, one field, the sound wave as a diffusion along B: T's rings (as
    ! vs_alfven's b0) or the SPD multigrid's P_V instance
    if (vpar) then
      call PetscLogEventBegin(pcev_fact_pv, ierr)
      if (pv_adv) then
        call sf_solver_setup(slv_pv, o_pv%mat, bk_pv, "P_V", comm_s, my_id, rtol, gmg_inst=6, nfields=1, &
                             smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, spdmg_inst=SPDMG_PV, amat=pva)
      else
        call sf_solver_setup(slv_pv, o_pv%mat, bk_pv, "P_V", comm_s, my_id, rtol, gmg_inst=6, nfields=1, &
                             smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, spdmg_inst=SPDMG_PV)
      endif
      call PetscLogEventEnd(pcev_fact_pv, ierr)
    endif

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    call sf_solver_setup(slv_rho, o_rho%mat, bk_rho, "rho", comm_s, my_id, rtol, gmg_inst=3, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    call sf_solver_setup(slv_T, o_T%mat, bk_T, "T", comm_s, my_id, rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS_RHOT)
    call PetscLogEventEnd(pcev_fact_rhot, ierr)

    ! the isotropic Laplacian on the anisotropic polar grid: zebra rings
    ! inside, zebra radial lines outside, as pair_psi's Laplacian rows
    if (first) then
      call PetscLogEventBegin(pcev_fact_lap, ierr)
      call sf_solver_setup(slv_lap, o_lap%mat, bk_vort, "Laplacian", comm_s, my_id, rtol, gmg_inst=5, &
                           nfields=1, smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS)
      call mass_cheb_setup(mc_j, o_mj%mat, comm_s, "j mass B_33", int(pc_ntor), target=MASS_TARGET)
      call mass_cheb_setup(mc_w, o_mw%mat, comm_s, "omega mass B_44", int(pc_ntor), target=MASS_TARGET)
      call PetscLogEventEnd(pcev_fact_lap, ierr)
    endif
    call physics_pc_mem("VS build: solvers set up", my_id)
    if (lumped) call mem_line("the solvers set up")

    if (.not. vecs_ready) call create_work_vectors()
    g_ctx%reduced_ready = .true.
    vs_first = .false.
    ! the families finish their setups at different times: wait here, so that
    ! the setup timer holds the slowest one rather than the first apply
    if (famode) call MPI_Barrier(comm, ierr)
    call PetscLogStagePop(ierr)

  contains

    !> P_lumped = (M_rho + B_24) - (1/opz) B_23 X B_12: X and the product's
    !! symbolic phase once, the union pattern fixed by the first build, the
    !! numeric product at every build; X at every build for the resistive
    !! bending. Where the pattern of M_rho + B_24 lies inside the product's
    !! (checked at the first build; the usual case, the product reaching
    !! further), P_lumped is the product matrix itself, scaled by -1/opz and
    !! M_rho + B_24 added in place (the same arithmetic as the two additions
    !! into a zeroed copy): no second matrix of P's size.
    subroutine assemble_lumped()
      if (res_dual) then
        call PetscLogEventBegin(vev_resbend, ierr)
        call resistive_bending()
        call PetscLogEventEnd(vev_resbend, ierr)
      endif
      call PetscLogEventBegin(vev_plprod, ierr)
      if (first) then
        if (.not. res_dual) then
          call dual_basis(o_mj%mat, int(pc_ntor), z_j, my_id, "B_33")
          call dual_symmetrize(z_j, o_mj%mat, dual_form)
        endif
        call MatMatMatMult(o_cb(CB_23)%mat, z_j, o_cb(CB_12)%mat, MAT_INITIAL_MATRIX, 2.0d0, t_bzb, ierr)
        call mem_line("B_23 X B_12", t_bzb)
        pl_merged = pattern_within(o_vort%mat, t_bzb)
        if (pl_merged) then
          o_pl%mat = t_bzb
          if (my_id == 0) write(*,'(A)') "[Physics PC]   P_lumped: the product B_23 X B_12 holds M_rho + B_24's "// &
            "pattern, assembled in place"
        else
          call MatDuplicate(t_bzb, MAT_COPY_VALUES, o_pl%mat, ierr)
          call MatAXPY(o_pl%mat, 1.0d0, o_vort%mat, DIFFERENT_NONZERO_PATTERN, ierr)
        endif
        o_pl%label = "P_lumped"; o_pl%nf = 1
      else
        call MatMatMatMult(o_cb(CB_23)%mat, z_j, o_cb(CB_12)%mat, MAT_REUSE_MATRIX, 2.0d0, t_bzb, ierr)
      endif
      if (pl_merged) then
        call MatScale(t_bzb, -1.0d0 / sf_opz(), ierr)
      else
        call MatZeroEntries(o_pl%mat, ierr)
        call MatAXPY(o_pl%mat, -1.0d0 / sf_opz(), t_bzb, SUBSET_NONZERO_PATTERN, ierr)
      endif
      call MatAXPY(o_pl%mat, 1.0d0, o_vort%mat, SUBSET_NONZERO_PATTERN, ierr)
      if (first .and. ax_on) then
        call mem_line("P_s (P_lumped of X_R)", o_pl%mat)
      else if (first) then
        call mem_line("P_lumped", o_pl%mat)
      endif
      call PetscLogEventEnd(vev_plprod, ierr)
      if (ax_on) then
        call PetscLogEventBegin(vev_axlr, ierr)
        call axis_lr()
        call PetscLogEventEnd(vev_axlr, ierr)
      endif
    end subroutine assemble_lumped

    !> AS(k): L = -B_23 W / opz and R = W' B_12, both zero at P's
    !! constrained rows (the rows of P_s that the block solver decouples:
    !! the first build's mask), K's LU (mod_petsc_pc_axis_schur), and P = P_s
    !! + L K^-1 R as a shell (first build). Every build.
    subroutine axis_lr()
      integer :: nmu
      real*8 :: nlu
      if (first) then
        call penalty_row_mask(o_pl%mat, mask_u, nmu)
        call MatMatMult(o_cb(CB_23)%mat, ax_w, MAT_INITIAL_MATRIX, 2.0d0, ax_l, ierr)
        call MatTransposeMatMult(ax_w, o_cb(CB_12)%mat, MAT_INITIAL_MATRIX, 2.0d0, ax_r, ierr)
      else
        call MatMatMult(o_cb(CB_23)%mat, ax_w, MAT_REUSE_MATRIX, 2.0d0, ax_l, ierr)
        call MatTransposeMatMult(ax_w, o_cb(CB_12)%mat, MAT_REUSE_MATRIX, 2.0d0, ax_r, ierr)
      endif
      call MatScale(ax_l, -1.0d0 / sf_opz(), ierr)
      call MatDiagonalScale(ax_l, mask_u, PETSC_NULL_VEC, ierr)
      call MatDiagonalScale(ax_r, PETSC_NULL_VEC, mask_u, ierr)
      call axk_set(AX_IK, ax_kk, ax_order)
      if (first) then
        call mem_line("L = -B_23 W / opz", ax_l, kspace=.true.)
        call mem_line("R = W' B_12", ax_r, kspace=.true.)
        call axs_shell(o_pl%mat, ax_l, ax_r, AX_IK, p_shell)
        nlu = axk_lu_nnz(AX_IK)
        if (my_id == 0) write(*,'(A,ES9.2,A,I0,A)') "[Physics PC]   AS: LU(K) ", nlu, " entries on rank ", &
          axk_rank(AX_IK), " (ordering "//trim(ax_order)//")"
      endif
    end subroutine axis_lr

    !> Is the pattern of a (its stored entries) inside that of b, on every
    !! rank of the operators' communicator? Row by row, both sorted.
    logical function pattern_within(a, b)
      Mat, intent(in) :: a, b
      PetscInt :: rs_, re_, r_, na_, nb_, i_, j_
      PetscInt, pointer :: ca_(:), cb_(:)
      integer :: mpe
      logical :: ok_
      call MatGetOwnershipRange(a, rs_, re_, ierr)
      ok_ = .true.
      do r_ = rs_, re_ - 1
        call MatGetRow(a, r_, na_, ca_, PETSC_NULL_SCALAR_POINTER, ierr)
        call MatGetRow(b, r_, nb_, cb_, PETSC_NULL_SCALAR_POINTER, ierr)
        j_ = 1
        do i_ = 1, na_
          do while (j_ <= nb_)
            if (cb_(j_) >= ca_(i_)) exit
            j_ = j_ + 1
          enddo
          if (j_ > nb_) then
            ok_ = .false.
          else if (cb_(j_) /= ca_(i_)) then
            ok_ = .false.
          endif
          if (.not. ok_) exit
        enddo
        call MatRestoreRow(b, r_, nb_, cb_, PETSC_NULL_SCALAR_POINTER, ierr)
        call MatRestoreRow(a, r_, na_, ca_, PETSC_NULL_SCALAR_POINTER, ierr)
        if (.not. ok_) exit
      enddo
      call MPI_Allreduce(MPI_IN_PLACE, ok_, 1, MPI_LOGICAL, MPI_LAND, comm_s, mpe)
      pattern_within = ok_
    end function pattern_within

    !> X (z_j) of the resistive bending, at every build (module header):
    !!   K_hyp = -(B_13 + M_eta),  T = B_31 Z_11 K_hyp,
    !!   S_s = B_33 + K_eta / opz + (T + T') / 2,
    !! X the form dual_form of the local dual basis of S_s with the core patch.
    !! K_hyp and K_eta are masked to psi's and j's interior DOFs. The patterns
    !! are those of the first build (the grid's patches), and z_11, k_hyp,
    !! t_s, t_st, s_res and z_j stay the same objects for the reuse of the
    !! products.
    subroutine resistive_bending()
      Mat :: zn
      integer :: nm_psi, nm_j
      if (first) then
        call penalty_row_mask(o_rb(RB_11)%mat, mask_psi, nm_psi)
        call penalty_row_mask(o_mj%mat, mask_j, nm_j)
        if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   resistive bending: ", nm_psi, &
          " constrained rows of psi (B_11), ", nm_j, " of j (B_33) leave K_hyp and K_eta"
        call MatDuplicate(o_rb(RB_13)%mat, MAT_COPY_VALUES, k_hyp, ierr)
        call MatAXPY(k_hyp, 1.0d0, o_meta%mat, DIFFERENT_NONZERO_PATTERN, ierr)
        call mem_line("K_hyp", k_hyp)
      else
        call MatZeroEntries(k_hyp, ierr)
        call MatAXPY(k_hyp, 1.0d0, o_rb(RB_13)%mat, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(k_hyp, 1.0d0, o_meta%mat, SUBSET_NONZERO_PATTERN, ierr)
      endif
      call MatScale(k_hyp, -1.0d0, ierr)
      call MatDiagonalScale(k_hyp, mask_psi, mask_j, ierr)
      call MatDiagonalScale(o_keta%mat, mask_j, mask_j, ierr)
      if (first) then
        call PetscLogEventBegin(vev_check, ierr)
        call eta_self_check()
        call PetscLogEventEnd(vev_check, ierr)
      endif

      ! T = B_31 Z_11 K_hyp (AS: Z_11 split at the axis by default)
      if (first) then
        if (ax_on .and. z11_split) then
          call dual_basis(o_rb(RB_11)%mat, int(pc_ntor), z_11, my_id, "B_11", axis_split=.true.)
        else
          call dual_basis(o_rb(RB_11)%mat, int(pc_ntor), z_11, my_id, "B_11")
        endif
        call mem_line("Z_11", z_11)
        call MatMatMatMult(o_rb(RB_31)%mat, z_11, k_hyp, MAT_INITIAL_MATRIX, 2.0d0, t_s, ierr)
        call mem_line("T = B_31 Z_11 K_hyp", t_s)
        call MatTranspose(t_s, MAT_INITIAL_MATRIX, t_st, ierr)
        call mem_line("T'", t_st)
        ! S_s's pattern: B_33, K_eta, T and T'
        call MatDuplicate(o_mj%mat, MAT_COPY_VALUES, s_res, ierr)
        call MatAXPY(s_res, 1.0d0, o_keta%mat, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatAXPY(s_res, 1.0d0, t_s, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatAXPY(s_res, 1.0d0, t_st, DIFFERENT_NONZERO_PATTERN, ierr)
        call mem_line("S_s", s_res)
      else
        if (ax_on .and. z11_split) then
          call dual_basis(o_rb(RB_11)%mat, int(pc_ntor), zn, my_id, axis_split=.true.)
        else
          call dual_basis(o_rb(RB_11)%mat, int(pc_ntor), zn, my_id)
        endif
        call MatCopy(zn, z_11, SAME_NONZERO_PATTERN, ierr)
        call MatDestroy(zn, ierr)
        call MatMatMatMult(o_rb(RB_31)%mat, z_11, k_hyp, MAT_REUSE_MATRIX, 2.0d0, t_s, ierr)
        call MatTranspose(t_s, MAT_REUSE_MATRIX, t_st, ierr)
      endif
      ! S_s = B_33 + K_eta / opz + (T + T') / 2
      call MatZeroEntries(s_res, ierr)
      call MatAXPY(s_res, 1.0d0, o_mj%mat, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(s_res, 1.0d0 / sf_opz(), o_keta%mat, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(s_res, 0.5d0, t_s, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(s_res, 0.5d0, t_st, SUBSET_NONZERO_PATTERN, ierr)
      call view_build(s_res, ss_view, "S_s")

      ! X (AS: X_R, the dual basis on the rings > k)
      if (first) then
        if (ax_on) then
          call dual_basis(s_res, int(pc_ntor), z_j, my_id, "S_s", axis_cut=ax_k, axis_idx=ax_idx)
        else
          call dual_basis(s_res, int(pc_ntor), z_j, my_id, "S_s", core=res_core)
        endif
        call mem_line("Z of S_s", z_j)
        call dual_symmetrize(z_j, s_res, dual_form)
        call mem_line("X = sym. form of Z", z_j)
      else
        if (ax_on) then
          call dual_basis(s_res, int(pc_ntor), zn, my_id, axis_cut=ax_k)
        else
          call dual_basis(s_res, int(pc_ntor), zn, my_id, core=res_core)
        endif
        call dual_symmetrize(zn, s_res, dual_form)
        call MatCopy(zn, z_j, SAME_NONZERO_PATTERN, ierr)
        call MatDestroy(zn, ierr)
      endif
      if (ax_on) then
        call PetscLogEventBegin(vev_axk, ierr)
        call axis_wk()
        call PetscLogEventEnd(vev_axk, ierr)
      endif
    end subroutine resistive_bending

    !> AS(k) (module header), at every build after X_R: S_d, the
    !! slot-diagonal part of S_s; E_A (first build: K's layout, all of it on
    !! the K rank, the owner of A's smallest index; K's index s nA + p for
    !! slot s and A's p-th index in ax_idx's order); W = E_A - X_R (S_d E_A);
    !! K = (W' S_d W + its transpose) / 2. Patterns and symbolic products of
    !! the first build.
    subroutine axis_wk()
      PetscInt, parameter :: zero_ = 0
      PetscInt :: rs_, re_, nka, nkl, r_, c_
      PetscInt, allocatable :: dnz(:), onz(:)
      integer :: ns_, na_, p_, s_, kr_, me_, mpe
      ns_ = int(pc_ntor)
      call slot_diagonal(s_res, s_dg)
      if (first) then
        call mem_line("S_d (slot-diagonal S_s)", s_dg)
        na_ = size(ax_idx)
        ! the K rank: the owner of A's smallest index (the axis)
        call MPI_Comm_rank(comm_s, me_, mpe)
        call MatGetOwnershipRange(s_res, rs_, re_, ierr)
        kr_ = -1
        r_ = int(minval(ax_idx), kind(rs_)) * ns_
        if (r_ >= rs_ .and. r_ < re_) kr_ = me_
        call MPI_Allreduce(MPI_IN_PLACE, kr_, 1, MPI_INTEGER, MPI_MAX, comm_s, mpe)
        nka = int(na_ * ns_, kind(rs_))
        nkl = 0
        if (me_ == kr_) nkl = nka
        allocate(dnz(re_ - rs_), onz(re_ - rs_)); dnz = 0; onz = 0
        do p_ = 1, na_
          r_ = int(ax_idx(p_), kind(rs_)) * ns_
          if (r_ < rs_ .or. r_ >= re_) cycle
          do s_ = 0, ns_ - 1
            if (me_ == kr_) then
              dnz(r_ - rs_ + s_ + 1) = 1
            else
              onz(r_ - rs_ + s_ + 1) = 1
            endif
          enddo
        enddo
        call MatCreate(comm_s, e_a, ierr)
        call MatSetSizes(e_a, re_ - rs_, nkl, PETSC_DETERMINE, nka, ierr)
        call MatSetType(e_a, MATMPIAIJ, ierr)
        call MatMPIAIJSetPreallocation(e_a, zero_, dnz, zero_, onz, ierr)
        do p_ = 1, na_
          r_ = int(ax_idx(p_), kind(rs_)) * ns_
          if (r_ < rs_ .or. r_ >= re_) cycle
          do s_ = 0, ns_ - 1
            c_ = int(s_ * na_ + p_ - 1, kind(rs_))
            call MatSetValue(e_a, r_ + s_, c_, 1.0d0, INSERT_VALUES, ierr)
          enddo
        enddo
        call MatAssemblyBegin(e_a, MAT_FINAL_ASSEMBLY, ierr)
        call MatAssemblyEnd(e_a, MAT_FINAL_ASSEMBLY, ierr)
        call MatMatMult(s_dg, e_a, MAT_INITIAL_MATRIX, 2.0d0, ax_sa, ierr)
        call MatMatMult(z_j, ax_sa, MAT_INITIAL_MATRIX, 2.0d0, ax_xsa, ierr)
        ! W = E_A - X_R S_d E_A: X_R has no rows on A, E_A no rows on R
        call MatDuplicate(ax_xsa, MAT_COPY_VALUES, ax_w, ierr)
        call MatScale(ax_w, -1.0d0, ierr)
        call MatAXPY(ax_w, 1.0d0, e_a, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatPtAP(s_dg, ax_w, MAT_INITIAL_MATRIX, 2.0d0, ax_k0, ierr)
        call MatTranspose(ax_k0, MAT_INITIAL_MATRIX, ax_kt, ierr)
        call MatDuplicate(ax_k0, MAT_COPY_VALUES, ax_kk, ierr)
        call MatAXPY(ax_kk, 1.0d0, ax_kt, DIFFERENT_NONZERO_PATTERN, ierr)
        call mem_line("W", ax_w, kspace=.true.)
        if (my_id == 0) write(*,'(A,I0,A,I0,A,I0,A,I0,A)') "[Physics PC]   AS(", ax_k, "): A = ", na_, &
          " indices on the rings 0..k, K ", nka, " rows on rank ", kr_, " of the operators' communicator"
      else
        call MatMatMult(s_dg, e_a, MAT_REUSE_MATRIX, 2.0d0, ax_sa, ierr)
        call MatMatMult(z_j, ax_sa, MAT_REUSE_MATRIX, 2.0d0, ax_xsa, ierr)
        call MatZeroEntries(ax_w, ierr)
        call MatAXPY(ax_w, -1.0d0, ax_xsa, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(ax_w, 1.0d0, e_a, SUBSET_NONZERO_PATTERN, ierr)
        call MatPtAP(s_dg, ax_w, MAT_REUSE_MATRIX, 2.0d0, ax_k0, ierr)
        call MatTranspose(ax_k0, MAT_REUSE_MATRIX, ax_kt, ierr)
        call MatZeroEntries(ax_kk, ierr)
        call MatAXPY(ax_kk, 1.0d0, ax_k0, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(ax_kk, 1.0d0, ax_kt, SUBSET_NONZERO_PATTERN, ierr)
      endif
      call MatScale(ax_kk, 0.5d0, ierr)
      if (first) call mem_line("K = W' S_d W", ax_kk, kspace=.true.)
    end subroutine axis_wk

    !> sd = the slot-diagonal part of the one-field operator s (the entries
    !! whose row and column have the same slot), s's layout; the pattern of
    !! the first call, refilled later.
    subroutine slot_diagonal(s, sd)
      Mat, intent(in) :: s
      Mat, intent(inout) :: sd
      PetscInt, parameter :: zero_ = 0, one_ = 1
      PetscInt :: rs_, re_, r_, nc_, cs_, ce_
      PetscInt, pointer :: cl_(:)
      PetscScalar, pointer :: vl_(:)
      PetscInt, allocatable :: dnz(:), onz(:), cc(:)
      PetscScalar, allocatable :: cv(:)
      integer :: ns_, q_, n_, c_
      ns_ = int(pc_ntor)
      call MatGetOwnershipRange(s, rs_, re_, ierr)
      call MatGetOwnershipRangeColumn(s, cs_, ce_, ierr)
      if (first) then
        allocate(dnz(re_ - rs_), onz(re_ - rs_)); dnz = 0; onz = 0
        do r_ = rs_, re_ - 1
          call MatGetRow(s, r_, nc_, cl_, vl_, ierr)
          q_ = int(mod(r_, int(ns_, kind(r_))))
          dnz(r_ - rs_ + 1) = count(mod(cl_(1:nc_), int(ns_, kind(r_))) == q_ .and. cl_(1:nc_) >= cs_ .and. &
                                    cl_(1:nc_) < ce_)
          onz(r_ - rs_ + 1) = count(mod(cl_(1:nc_), int(ns_, kind(r_))) == q_) - dnz(r_ - rs_ + 1)
          call MatRestoreRow(s, r_, nc_, cl_, vl_, ierr)
        enddo
        call MatCreate(comm_s, sd, ierr)
        call MatSetSizes(sd, re_ - rs_, ce_ - cs_, PETSC_DETERMINE, PETSC_DETERMINE, ierr)
        call MatSetType(sd, MATMPIAIJ, ierr)
        call MatMPIAIJSetPreallocation(sd, zero_, dnz, zero_, onz, ierr)
        call MatSetOption(sd, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE, ierr)
      else
        call MatZeroEntries(sd, ierr)
      endif
      do r_ = rs_, re_ - 1
        call MatGetRow(s, r_, nc_, cl_, vl_, ierr)
        q_ = int(mod(r_, int(ns_, kind(r_))))
        allocate(cc(nc_), cv(nc_))
        n_ = 0
        do c_ = 1, int(nc_)
          if (int(mod(cl_(c_), int(ns_, kind(r_)))) /= q_) cycle
          n_ = n_ + 1; cc(n_) = cl_(c_); cv(n_) = vl_(c_)
        enddo
        call MatRestoreRow(s, r_, nc_, cl_, vl_, ierr)
        if (n_ > 0) call MatSetValues(sd, one_, [r_], int(n_, kind(r_)), cc(1:n_), cv(1:n_), INSERT_VALUES, ierr)
        deallocate(cc, cv)
      enddo
      call MatAssemblyBegin(sd, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(sd, MAT_FINAL_ASSEMBLY, ierr)
    end subroutine slot_diagonal

    !> The first build's check of M_eta against the Jacobian: K_hyp - K_1 =
    !! -(B_13 + M_eta + K_1) on the interior rows and columns, against M_eta
    !! and B_13 there (Frobenius norms), and K_hyp's skew part (its
    !! coefficient eta_num_T is a function of the state in model600). Then
    !! K_1 is freed.
    subroutine eta_self_check()
      type(sf_op_t) :: o_k1
      Mat :: dm, bm
      real*8 :: nd, nb, nm, nref, nks, nkh
      o_k1%label = "K_1"; o_k1%nf = 1; o_k1%tag = o_keta%tag + 1
      o_k1%se(1, 1) = 0; o_k1%sv(1, 1) = 0; o_k1%wid(1, 1) = 1
      if (.not. g_ctx%k1_ready) call build_fatal("K_1 was never assembled (petsc_assemble_pc_matrices)")
      call sfo_build(o_k1, A_full, [g_ctx%K_1], sf_opz(), my_id)
      call MatDuplicate(k_hyp, MAT_COPY_VALUES, dm, ierr)
      call MatAXPY(dm, -1.0d0, o_k1%mat, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(dm, mask_psi, mask_j, ierr)
      call MatNorm(dm, NORM_FROBENIUS, nd, ierr)
      call MatDestroy(dm, ierr)
      call MatDuplicate(o_rb(RB_13)%mat, MAT_COPY_VALUES, bm, ierr)
      call MatDiagonalScale(bm, mask_psi, mask_j, ierr)
      call MatNorm(bm, NORM_FROBENIUS, nb, ierr)
      call MatDestroy(bm, ierr)
      call MatDuplicate(o_meta%mat, MAT_COPY_VALUES, bm, ierr)
      call MatDiagonalScale(bm, mask_psi, mask_j, ierr)
      call MatNorm(bm, NORM_FROBENIUS, nm, ierr)
      call MatDestroy(bm, ierr)
      ! K_hyp' - K_hyp on the rows and columns interior to psi and j both
      call MatTranspose(k_hyp, MAT_INITIAL_MATRIX, dm, ierr)
      call MatAXPY(dm, -1.0d0, k_hyp, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(dm, mask_psi, mask_psi, ierr)
      call MatDiagonalScale(dm, mask_j, mask_j, ierr)
      call MatNorm(dm, NORM_FROBENIUS, nks, ierr)
      call MatDestroy(dm, ierr)
      call MatNorm(k_hyp, NORM_FROBENIUS, nkh, ierr)
      call sfo_destroy(o_k1)
      call MatDestroy(g_ctx%K_1, ierr)
      g_ctx%k1_ready = .false.; g_ctx%k1_done = .true.
      if (my_id == 0) then
        write(*,'(A,ES10.3,A,ES10.3,A,ES10.3,A)') "[Physics PC]   resistive bending check: |B_13 + M_eta + K_1|_F = ", &
          nd, " of |M_eta|_F = ", nm, ", of |B_13|_F = ", nb, &
          " (K_1 = theta dt (eta_num_T grad v, grad w), interior rows)"
        ! against M_eta, or B_13 where eta = 0
        nref = nm
        if (nref <= 0.d0) nref = nb
        if (nref > 0.d0 .and. nd > ETA_CHECK_TOL * nref) write(*,'(A,ES10.3,A)') &
          "[Physics PC]   WARNING: -B_13 - M_eta differs from the hyper-resistive part by ", nd / nref, &
          " of |M_eta| (|B_13| if eta = 0): the resistive bending does not use the model's eta_T and eta_num_T"
        if (nkh > 0.d0) write(*,'(A,ES9.2,A)') "[Physics PC]   K_hyp = -B_13 - M_eta: skew part ", &
          0.5d0 * nks / nkh, " of its Frobenius norm (interior rows)"
      endif
    end subroutine eta_self_check

    !> -vs_mem_report, first build: one line for mat (if given) with its
    !! stored entries (sum over the ranks and the largest rank), its rows
    !! and its longest row (largest rank), and a second one with its entries
    !! that are exactly zero, those between two |n| families (rows and
    !! columns of different families; JOREK index * slots + slot layout) and
    !! the exact zeros among them; then the current RSS (rank 0 and the
    !! largest rank) and the RSS peak (the largest rank). Collective on the
    !! operators' communicator. kspace: rows or columns in K's index space
    !! (AS(k)), whose families the layout does not give: none counted.
    subroutine mem_line(tag, mat, kspace)
      use mod_petsc_pc_physics_ctx, only: physics_pc_peak_rss
      use mod_petsc_pc_harm, only: pc_global_slot, pc_ntor
      character(len=*), intent(in) :: tag
      Mat, intent(in), optional :: mat
      logical, intent(in), optional :: kspace
      MatInfo :: mi
      PetscLogDouble :: cur
      PetscInt :: rs_, re_, r_, nc_, c_
      PetscInt, pointer :: cl_(:)
      PetscScalar, pointer :: vl_(:)
      real*8 :: v(9), vs(9), vm(9), rss(3), rssm(3)
      integer :: mpe, ns_, fr_, fc_
      logical :: fam
      if (.not. (mem_report .and. first)) return
      fam = .true.
      if (present(kspace)) fam = .not. kspace
      if (present(mat)) then
        call MatGetInfo(mat, MAT_LOCAL, mi, ierr)
        call MatGetOwnershipRange(mat, rs_, re_, ierr)
        ns_ = pc_ntor
        v = 0.d0
        v(1) = mi%nz_allocated; v(2) = mi%memory; v(3) = dble(re_ - rs_)
        do r_ = rs_, re_ - 1
          call MatGetRow(mat, r_, nc_, cl_, vl_, ierr)
          v(4) = max(v(4), dble(nc_))
          fr_ = (pc_global_slot(int(mod(r_, int(ns_, kind(r_))))) + 1) / 2
          do c_ = 1, nc_
            fc_ = (pc_global_slot(int(mod(cl_(c_), int(ns_, kind(r_))))) + 1) / 2
            if (.not. fam) fc_ = fr_
            if (vl_(c_) == 0.d0) v(5) = v(5) + 1.d0
            v(8) = max(v(8), abs(vl_(c_)))
            if (fc_ /= fr_) then
              v(6) = v(6) + 1.d0
              if (vl_(c_) == 0.d0) v(7) = v(7) + 1.d0
              v(9) = max(v(9), abs(vl_(c_)))
            endif
          enddo
          call MatRestoreRow(mat, r_, nc_, cl_, vl_, ierr)
        enddo
        call MPI_Allreduce(v, vs, 9, MPI_DOUBLE_PRECISION, MPI_SUM, comm_s, mpe)
        call MPI_Allreduce(v, vm, 9, MPI_DOUBLE_PRECISION, MPI_MAX, comm_s, mpe)
        ! (MatGetInfo's memory is 0 without PETSc's malloc tracing: entries only)
        if (my_id == 0) write(*,'(A,A,T44,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A,F7.0)') &
          "[VS mem] ", tag, "nnz ", vs(1), " (max rank ", vm(1), "), rows ", vs(3), " (max rank ", vm(3), &
          "), longest row ", vm(4)
        if (my_id == 0) write(*,'(A,A,T44,A,ES9.2,A,F5.1,A,ES9.2,A,F5.1,A,ES9.2,A,ES9.2)') &
          "[VS mem] ", "  "//tag, "exact zeros ", vs(5), " (", 100.d0 * vs(5) / max(vs(1), 1.d0), &
          " %), between |n| families ", vs(6), " (", 100.d0 * vs(6) / max(vs(1), 1.d0), " %), of them zero ", &
          vs(7), ", their largest |a| / largest |a| ", vm(9) / max(vm(8), tiny(1.d0))
      endif
      call PetscMemoryGetCurrentUsage(cur, ierr)
      rss = [dble(cur), physics_pc_peak_rss(), 0.d0]
      call MPI_Allreduce(rss, rssm, 3, MPI_DOUBLE_PRECISION, MPI_MAX, comm_s, mpe)
      if (my_id == 0) write(*,'(A,A,T44,A,F8.3,A,F8.3,A,F8.3,A)') "[VS mem] ", "  RSS after "//tag, &
        "rank 0 ", rss(1) / 1.d9, " GB, max rank ", rssm(1) / 1.d9, " GB, peak max rank ", rssm(2) / 1.d9, " GB"
    end subroutine mem_line

    !> The skew part of the solved operator mat against the matrix, in the
    !! Frobenius norm. P_lumped: the quadrature error of B_23 = -B_12' with a
    !! symmetric X and no flow; P_V: rounding (symmetric by construction);
    !! P_V^A: the advection.
    subroutine report_skew(mat, name)
      Mat, intent(in) :: mat
      character(len=*), intent(in) :: name
      Mat :: pt
      real*8 :: nskew, np_
      call MatTranspose(mat, MAT_INITIAL_MATRIX, pt, ierr)
      call MatAXPY(pt, -1.0d0, mat, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatNorm(pt, NORM_FROBENIUS, nskew, ierr)
      call MatNorm(mat, NORM_FROBENIUS, np_, ierr)
      call MatDestroy(pt, ierr)
      if (my_id == 0) write(*,'(A,ES9.2,A)') "[Physics PC]   "//name//": skew part ", 0.5d0 * nskew / np_, &
        " of its Frobenius norm"
    end subroutine report_skew

    !> AS(k): the applied P = P_s + L K^-1 R on two random vectors x, y in
    !! [-1, 1] on the rows that L reaches (the axis region, zero elsewhere):
    !! (x'P y - y'P x) / (|x'P y| + |y'P x|) for P and for its axis part
    !! L K^-1 R alone (L = R' / opz up to the quadrature of B_23 = -B_12'),
    !! and x'L K^-1 R x / x'P x (>= 0 when the axis part is definite).
    subroutine report_skew_applied()
      Vec :: x1, x2, y1, rm
      PetscRandom :: rnd
      PetscInt :: rs_, re_, r_, nc_
      PetscInt, pointer :: cl_(:)
      PetscScalar, pointer :: vl_(:), ra(:)
      real*8 :: a12, a21, b12, b21, b11, a11
      call MatCreateVecs(o_pl%mat, x1, y1, ierr)
      call VecDuplicate(x1, x2, ierr)
      call VecDuplicate(x1, rm, ierr)
      call MatGetOwnershipRange(ax_l, rs_, re_, ierr)
      call VecGetArray(rm, ra, ierr)
      do r_ = rs_, re_ - 1
        call MatGetRow(ax_l, r_, nc_, cl_, vl_, ierr)
        ra(r_ - rs_ + 1) = merge(1.d0, 0.d0, nc_ > 0)
        call MatRestoreRow(ax_l, r_, nc_, cl_, vl_, ierr)
      enddo
      call VecRestoreArray(rm, ra, ierr)
      call PetscRandomCreate(comm_s, rnd, ierr)
      call PetscRandomSetInterval(rnd, -1.d0, 1.d0, ierr)
      call VecSetRandom(x1, rnd, ierr)
      call VecSetRandom(x2, rnd, ierr)
      call VecPointwiseMult(x1, x1, rm, ierr)
      call VecPointwiseMult(x2, x2, rm, ierr)
      call MatMult(p_shell, x2, y1, ierr);   call VecDot(x1, y1, a12, ierr)
      call MatMult(p_shell, x1, y1, ierr);   call VecDot(x2, y1, a21, ierr)
      call VecDot(x1, y1, a11, ierr)
      call axs_axis_mult(p_shell, x2, y1);   call VecDot(x1, y1, b12, ierr)
      call axs_axis_mult(p_shell, x1, y1);   call VecDot(x2, y1, b21, ierr)
      call VecDot(x1, y1, b11, ierr)
      if (my_id == 0) write(*,'(A,ES9.2,A,ES9.2,A,ES10.3)') "[Physics PC]   P = P_s + L K^-1 R applied, random "// &
        "x, y on L's rows: skew (x'Py - y'Px) / (|x'Py| + |y'Px|) ", (a12 - a21) / max(abs(a12) + abs(a21), &
        tiny(1.d0)), ", of L K^-1 R ", (b12 - b21) / max(abs(b12) + abs(b21), tiny(1.d0)), &
        "; x'L K^-1 R x / x'Px ", b11 / a11
      call PetscRandomDestroy(rnd, ierr)
      call VecDestroy(x1, ierr); call VecDestroy(x2, ierr); call VecDestroy(y1, ierr); call VecDestroy(rm, ierr)
    end subroutine report_skew_applied

    !> -vs_axis_view <prefix>, first build: K, L, R, W (<prefix>_K.petsc,
    !! _L, _R, _W) and A's indices in K's order (<prefix>_idx.petsc, a Vec of
    !! the 0-based JOREK indices) in PETSc's binary format. Global layout only
    !! (the mode split's families would write the same files).
    subroutine axis_view()
      PetscViewer :: vw
      Vec :: vi
      PetscScalar, pointer :: va(:)
      PetscInt :: nvi
      if (famode) then
        if (my_id == 0) write(*,'(A)') "[Physics PC]   WARNING: -vs_axis_view ignored under the mode split"
        return
      endif
      call view_mat(ax_kk, trim(ax_view)//"_K.petsc"); call view_mat(ax_l, trim(ax_view)//"_L.petsc")
      call view_mat(ax_r, trim(ax_view)//"_R.petsc");  call view_mat(ax_w, trim(ax_view)//"_W.petsc")
      if (my_id == 0) then
        nvi = size(ax_idx)
        call VecCreateSeq(PETSC_COMM_SELF, nvi, vi, ierr)
        call VecGetArray(vi, va, ierr)
        va = dble(ax_idx)
        call VecRestoreArray(vi, va, ierr)
        call PetscViewerBinaryOpen(PETSC_COMM_SELF, trim(ax_view)//"_idx.petsc", FILE_MODE_WRITE, vw, ierr)
        call VecView(vi, vw, ierr)
        call PetscViewerDestroy(vw, ierr)
        call VecDestroy(vi, ierr)
        write(*,'(A)') "[Physics PC]   AS: K, L, R, W and A's indices written to "//trim(ax_view)//"_*.petsc"
      endif
    end subroutine axis_view

    !> m in PETSc's binary format to the file fname (the operators' communicator).
    subroutine view_mat(m, fname)
      Mat, intent(in) :: m
      character(len=*), intent(in) :: fname
      PetscViewer :: vw
      call PetscViewerBinaryOpen(comm_s, fname, FILE_MODE_WRITE, vw, ierr)
      call MatView(m, vw, ierr)
      call PetscViewerDestroy(vw, ierr)
    end subroutine view_mat

    !> AS(k)'s P solver: the SPD multigrid with the axis part
    !! (spdmg_set_axis; Krylov on the shell P), or the exact solve, a PREONLY
    !! KSP on the shell P whose PCSHELL solves the augmented matrix [[P_s, L],
    !! [R, -K]] by MUMPS (mod_petsc_pc_axis_schur, refactored at every
    !! build).
    subroutine axis_p_solver()
      use mod_petsc_pc_blocks, only: pc_print_block_setup
      PC :: pcp
      character(len=24) :: tstr
      if (bk_pl == SF_SPDMG) then
        call spdmg_set_axis(SPDMG_P, ax_l, ax_r, AX_IK, ax_k)
        call sf_solver_setup(slv_vort, o_pl%mat, bk_pl, "P_lumped", comm_s, my_id, rtol, gmg_inst=1, &
                             nfields=1, smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, amat=p_shell)
        return
      endif
      if (first) ax_ia = axa_new()
      call axa_setup(ax_ia, o_pl%mat, ax_l, ax_r, AX_IK, "p")
      if (.not. slv_vort%created) then
        call KSPCreate(comm_s, slv_vort%ksp, ierr)
        call KSPSetType(slv_vort%ksp, KSPPREONLY, ierr)
        call KSPGetPC(slv_vort%ksp, pcp, ierr)
        call PCSetType(pcp, PCSHELL, ierr)
        call PCShellSetName(pcp, "P_lumped: MUMPS LU of [[P_s, L], [R, -K]]", ierr)
        call axa_set_pc(ax_ia, pcp)
        call KSPSetOperators(slv_vort%ksp, p_shell, p_shell, ierr)
        slv_vort%created = .true.; slv_vort%backend = SF_LU; slv_vort%label = "P_lumped"
        write(tstr, '(ES9.2)') axa_factor_nnz(ax_ia)
        call pc_print_block_setup(comm_s, "P_lumped", "PREONLY, MUMPS LU of the augmented matrix [[P_s, L], "// &
                                  "[R, -K]] (factor entries "//trim(adjustl(tstr))//")")
      endif
      call KSPSetUp(slv_vort%ksp, ierr)
    end subroutine axis_p_solver

    !> P_V = Mv (B_77 + B_77') Mv / 2 + (1 - Mv) diag(B_77) + B_57m' X B_57m
    !! (module header), B_57m = Mr B_57 Mv, Mr and Mv the masks of rho's and
    !! V's constrained DOFs (0 there), X = (Z + Z') / 2, Z the local dual basis
    !! of Q = W sym(B_55) W with Q's constrained rows and columns decoupled
    !! (unit diagonal), W at every build from the state; with -vs_vpar_flow
    !! adv also P_V^A = Mv B_77 Mv + B_57m' Z_A B_57m, Z_A the dual basis of
    !! W B_55 W (pva_boundary_rows sets its constrained rows). The patterns
    !! are the first build's (the grid's patches), the products' symbolic
    !! phases are reused.
    subroutine assemble_pv()
      Mat :: zn
      Vec :: dv
      integer :: nm_r, nm_v
      if (first) then
        call penalty_row_mask(o_rho%mat, mask_rho, nm_r)
        call penalty_row_mask(o_v77%mat, mask_v, nm_v)
        call VecDuplicate(mask_rho, cmask_rho, ierr)
        call VecSet(cmask_rho, 1.0d0, ierr)
        call VecAXPY(cmask_rho, -1.0d0, mask_rho, ierr)
        call VecDuplicate(mask_v, cmask_v, ierr)
        call VecSet(cmask_v, 1.0d0, ierr)
        call VecAXPY(cmask_v, -1.0d0, mask_v, ierr)
        call VecDuplicate(mask_rho, w_rho, ierr)
        if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   v_par: ", nm_r, &
          " constrained rows of rho (B_55), ", nm_v, " of V (B_77) leave the congruence B_57' X B_57"
        call PetscLogEventBegin(vev_check, ierr)
        call vpar_self_check()
        call PetscLogEventEnd(vev_check, ierr)
      endif
      call node_weights(w_rho)

      ! Q = W sym(B_55) W, Dirichlet rows at rho's constrained DOFs
      if (first) then
        call MatTranspose(o_rho%mat, MAT_INITIAL_MATRIX, b55t, ierr)
        call MatDuplicate(o_rho%mat, MAT_COPY_VALUES, q_pv, ierr)
        call MatAXPY(q_pv, 1.0d0, b55t, DIFFERENT_NONZERO_PATTERN, ierr)
      else
        call MatTranspose(o_rho%mat, MAT_REUSE_MATRIX, b55t, ierr)
        call MatZeroEntries(q_pv, ierr)
        call MatAXPY(q_pv, 1.0d0, o_rho%mat, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(q_pv, 1.0d0, b55t, SUBSET_NONZERO_PATTERN, ierr)
      endif
      call MatScale(q_pv, 0.5d0, ierr)
      call MatDiagonalScale(q_pv, w_rho, w_rho, ierr)
      call MatDiagonalSet(q_pv, cmask_rho, ADD_VALUES, ierr)

      ! X
      if (first) then
        call mem_line("Q = W sym(B_55) W", q_pv)
        call dual_basis(q_pv, int(pc_ntor), x_pv, my_id, "W sym(B_55) W", core=pv_core)
        call dual_symmetrize(x_pv, q_pv, DUAL_AVG)
        call mem_line("X of Q", x_pv)
      else
        call dual_basis(q_pv, int(pc_ntor), zn, my_id, core=pv_core)
        call dual_symmetrize(zn, q_pv, DUAL_AVG)
        call MatCopy(zn, x_pv, SAME_NONZERO_PATTERN, ierr)
        call MatDestroy(zn, ierr)
      endif

      ! B_57m, then the congruence T = B_57m' X B_57m
      if (first) then
        call MatDuplicate(o_vc(VC_57)%mat, MAT_COPY_VALUES, b57m, ierr)
      else
        call MatCopy(o_vc(VC_57)%mat, b57m, SAME_NONZERO_PATTERN, ierr)
      endif
      call MatDiagonalScale(b57m, mask_rho, mask_v, ierr)
      if (first) then
        call MatPtAP(x_pv, b57m, MAT_INITIAL_MATRIX, 2.0d0, t_pv, ierr)
        call mem_line("B_57' X B_57", t_pv)
      else
        call MatPtAP(x_pv, b57m, MAT_REUSE_MATRIX, 2.0d0, t_pv, ierr)
      endif

      ! P_V; its pattern: B_77, B_77' and T
      if (first) then
        call MatTranspose(o_v77%mat, MAT_INITIAL_MATRIX, b77t, ierr)
        call MatDuplicate(t_pv, MAT_COPY_VALUES, o_pv%mat, ierr)
        call MatAXPY(o_pv%mat, 1.0d0, o_v77%mat, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatAXPY(o_pv%mat, 1.0d0, b77t, DIFFERENT_NONZERO_PATTERN, ierr)
        o_pv%label = "P_V"; o_pv%nf = 1
      else
        call MatTranspose(o_v77%mat, MAT_REUSE_MATRIX, b77t, ierr)
      endif
      call MatZeroEntries(o_pv%mat, ierr)
      call MatAXPY(o_pv%mat, 0.5d0, o_v77%mat, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(o_pv%mat, 0.5d0, b77t, SUBSET_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(o_pv%mat, mask_v, mask_v, ierr)
      call MatCreateVecs(o_v77%mat, dv, PETSC_NULL_VEC, ierr)
      call MatGetDiagonal(o_v77%mat, dv, ierr)
      call VecPointwiseMult(dv, dv, cmask_v, ierr)
      call MatDiagonalSet(o_pv%mat, dv, ADD_VALUES, ierr)
      call VecDestroy(dv, ierr)
      call MatAXPY(o_pv%mat, 1.0d0, t_pv, SUBSET_NONZERO_PATTERN, ierr)
      if (first) call mem_line("P_V", o_pv%mat)

      if (.not. pv_adv) return
      ! Q_A = W B_55 W, Z_A, T_A = B_57m' Z_A B_57m, P_V^A = Mv B_77 Mv + T_A;
      ! dual_basis returns the rows of the patch matrices' inverse transposes
      ! (the dual basis of M' for a non-symmetric M), so it gets Q_A' = W B_55' W
      if (first) then
        call MatDuplicate(b55t, MAT_COPY_VALUES, qa_pv, ierr)
      else
        call MatCopy(b55t, qa_pv, SAME_NONZERO_PATTERN, ierr)
      endif
      call MatDiagonalScale(qa_pv, w_rho, w_rho, ierr)
      call MatDiagonalSet(qa_pv, cmask_rho, ADD_VALUES, ierr)
      if (first) then
        call dual_basis(qa_pv, int(pc_ntor), za_pv, my_id, "W B_55' W (P_V^A)", core=pv_core)
        call MatPtAP(za_pv, b57m, MAT_INITIAL_MATRIX, 2.0d0, ta_pv, ierr)
        call MatDuplicate(ta_pv, MAT_COPY_VALUES, pva, ierr)
        call MatAXPY(pva, 1.0d0, o_v77%mat, DIFFERENT_NONZERO_PATTERN, ierr)
      else
        call dual_basis(qa_pv, int(pc_ntor), zn, my_id, core=pv_core)
        call MatCopy(zn, za_pv, SAME_NONZERO_PATTERN, ierr)
        call MatDestroy(zn, ierr)
        call MatPtAP(za_pv, b57m, MAT_REUSE_MATRIX, 2.0d0, ta_pv, ierr)
      endif
      call MatZeroEntries(pva, ierr)
      call MatAXPY(pva, 1.0d0, o_v77%mat, SUBSET_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(pva, mask_v, mask_v, ierr)
      call MatAXPY(pva, 1.0d0, ta_pv, SUBSET_NONZERO_PATTERN, ierr)
    end subroutine assemble_pv

    !> P_V^A's constrained rows: P_V's decoupled diagonal (after
    !! sf_solver_penalty_rows), so that both have the same Dirichlet rows and
    !! the penalty rows' right-hand-side factor of slv_pv fits P_V^A.
    subroutine pva_boundary_rows()
      Vec :: dv
      call MatCreateVecs(o_pv%mat, dv, PETSC_NULL_VEC, ierr)
      call MatGetDiagonal(o_pv%mat, dv, ierr)
      call VecPointwiseMult(dv, dv, cmask_v, ierr)
      call MatDiagonalSet(pva, dv, ADD_VALUES, ierr)
      call VecDestroy(dv, ierr)
    end subroutine pva_boundary_rows

    !> w = (rho0 / T0)^1/2 at the node of each row's JOREK index (the n = 0
    !! value DOF of the state, node_list), times rho's mask, in o_rho's
    !! layout. rho0 and T0 are floored at W_FLOOR times their maxima over
    !! the grid (a WARNING with the count where the floor acts).
    subroutine node_weights(w)
      use nodes_elements, only: node_list
      use mod_parameters, only: n_degrees, var_rho, var_T
      Vec, intent(inout) :: w
      PetscScalar, pointer :: wa(:), ma(:)
      PetscInt :: rs, re, ng
      real*8, allocatable :: wi(:)
      real*8 :: rmax, tmax, r0, t0
      integer :: n, kd, i, ns, nfl
      ns = int(pc_ntor)
      call MatGetSize(o_rho%mat, ng, PETSC_NULL_INTEGER, ierr)
      allocate(wi(0:int(ng) / ns - 1))
      wi = 0.d0
      rmax = 0.d0; tmax = 0.d0
      do n = 1, node_list%n_nodes
        rmax = max(rmax, node_list%node(n)%values(1, 1, var_rho))
        tmax = max(tmax, node_list%node(n)%values(1, 1, var_T))
      enddo
      nfl = 0
      do n = 1, node_list%n_nodes
        r0 = node_list%node(n)%values(1, 1, var_rho)
        t0 = node_list%node(n)%values(1, 1, var_T)
        if (r0 < W_FLOOR * rmax .or. t0 < W_FLOOR * tmax) nfl = nfl + 1
        r0 = max(r0, W_FLOOR * rmax); t0 = max(t0, W_FLOOR * tmax)
        do kd = 1, n_degrees
          wi(node_list%node(n)%index(kd) - 1) = sqrt(r0 / t0)
        enddo
      enddo
      if (nfl > 0 .and. my_id == 0) write(*,'(A,I0,A,ES8.1,A)') "[Physics PC]   v_par: WARNING rho0 or T0 "// &
        "floored at ", nfl, " nodes (", W_FLOOR, " of the maximum) in W = (rho0 / T0)^1/2"
      call MatGetOwnershipRange(o_rho%mat, rs, re, ierr)
      call VecGetArray(w, wa, ierr)
      call VecGetArrayRead(mask_rho, ma, ierr)
      do i = int(rs), int(re) - 1
        wa(i - int(rs) + 1) = wi(i / ns) * ma(i - int(rs) + 1)
      enddo
      call VecRestoreArrayRead(mask_rho, ma, ierr)
      call VecRestoreArray(w, wa, ierr)
      deallocate(wi)
    end subroutine node_weights

    !> The first build's check of the structure P_V rests on, on the
    !! interior rows and columns (B_57m = Mr B_57 Mv, B_76m = Mv B_76 MT):
    !! |B_57m - B_76m|_F and |B_76m + B_57m'|_F against |B_57m|_F (the study:
    !! 1e-16 and 1e-5 at rest). B_76 is extracted for the check alone.
    subroutine vpar_self_check()
      use mod_parameters, only: var_T
      type(sf_op_t) :: o_76
      Mat :: b57i, b76i, dm
      Vec :: mask_t
      real*8 :: n57, nd, ns_
      o_76%label = "B_76"; o_76%nf = 1; o_76%tag = o_vc(NVC)%tag + 1
      o_76%se(1, 1) = var_Vpar; o_76%sv(1, 1) = var_T
      call sfo_build(o_76, A_full, wm, sf_opz(), my_id)
      call penalty_row_mask(o_T%mat, mask_t)
      call MatDuplicate(o_vc(VC_57)%mat, MAT_COPY_VALUES, b57i, ierr)
      call MatDiagonalScale(b57i, mask_rho, mask_v, ierr)
      call MatDuplicate(o_76%mat, MAT_COPY_VALUES, b76i, ierr)
      call MatDiagonalScale(b76i, mask_v, mask_t, ierr)
      call MatNorm(b57i, NORM_FROBENIUS, n57, ierr)
      call MatDuplicate(b57i, MAT_COPY_VALUES, dm, ierr)
      call MatAXPY(dm, -1.0d0, b76i, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatNorm(dm, NORM_FROBENIUS, nd, ierr)
      call MatDestroy(dm, ierr)
      call MatTranspose(b57i, MAT_INITIAL_MATRIX, dm, ierr)
      call MatAXPY(dm, 1.0d0, b76i, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatNorm(dm, NORM_FROBENIUS, ns_, ierr)
      call MatDestroy(dm, ierr)
      call MatDestroy(b57i, ierr)
      call MatDestroy(b76i, ierr)
      call VecDestroy(mask_t, ierr)
      call sfo_destroy(o_76)
      if (my_id == 0 .and. n57 > 0.d0) write(*,'(A,ES10.3,A,ES10.3,A,ES10.3,A)') "[Physics PC]   v_par check: "// &
        "|B_57 - B_76|_F = ", nd / n57, ", |B_76 + B_57'|_F = ", ns_ / n57, " of |B_57|_F = ", n57, &
        " (interior rows and columns)"
    end subroutine vpar_self_check

    !> mat in PETSc's binary format to file fname, if fname is set (the
    !! operators' layout: JOREK index, slot fastest): the first build creates
    !! the file, every later build appends its matrix. Collective.
    subroutine view_build(mat, fname, what)
      Mat, intent(in) :: mat
      character(len=*), intent(in) :: fname, what
      PetscViewer :: vw
      if (len_trim(fname) == 0) return
      if (first) then
        call PetscViewerBinaryOpen(comm_s, trim(fname), FILE_MODE_WRITE, vw, ierr)
      else
        call PetscViewerBinaryOpen(comm_s, trim(fname), FILE_MODE_APPEND, vw, ierr)
      endif
      call MatView(mat, vw, ierr)
      call PetscViewerDestroy(vw, ierr)
      if (my_id == 0) write(*,'(A)') "[Physics PC]   "//what//" written to "//trim(fname)
    end subroutine view_build

    subroutine build_or_fill(op, ws)
      type(sf_op_t), intent(inout) :: op
      Mat, intent(in) :: ws(:)
      if (first) then
        call sfo_build(op, A_full, ws, sf_opz(), my_id)
      else
        call sfo_fill(op, A_full, ws, sf_opz())
      endif
    end subroutine build_or_fill

    !> The operators' block layouts.
    subroutine define_operators()
      use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
      integer, parameter :: ce(NCB) = [var_psi, var_u, var_u, var_rho, var_rho, var_T, var_T, var_T, var_T]
      integer, parameter :: cv(NCB) = [var_u, var_zj, var_w, var_psi, var_u, var_psi, var_u, var_zj, var_rho]
      character(len=4), parameter :: cn(NCB) = ["B_12", "B_23", "B_24", "B_51", "B_52", "B_61", "B_62", &
                                                "B_63", "B_65"]
      integer, parameter :: re(NRB) = [var_psi, var_psi, var_zj]
      integer, parameter :: rv(NRB) = [var_psi, var_zj, var_psi]
      character(len=4), parameter :: rn(NRB) = ["B_11", "B_13", "B_31"]
      integer, parameter :: ve(NVC) = [var_rho, var_T, var_Vpar, var_Vpar]
      integer, parameter :: vv(NVC) = [var_Vpar, var_Vpar, var_psi, var_u]
      character(len=4), parameter :: vn(NVC) = ["B_57", "B_67", "B_71", "B_72"]
      integer, parameter :: NDIAG = 7
      integer :: k_
      ! fields (w, y[, z]): the columns of omega, j[, psi]; the rows of u, then
      ! of psi and j (y), [j and psi (z)], as in the module header
      o_vort%label = "vorticity"; o_vort%tag = 1
      o_vort%nf = merge(3, 2, resistive)
      if (lumped) then                                                             ! M_rho + B_24 alone
        o_vort%label = "M_rho + B_24"; o_vort%nf = 1
      endif
      o_vort%se(1, 1) = var_u;   o_vort%sv(1, 1) = var_w;  o_vort%wid(1, 1) = 1     ! B_24 + M_rho
      o_vort%se(1, 2) = var_u;   o_vort%sv(1, 2) = var_zj                          ! B_23
      o_vort%se(2, 1) = var_psi; o_vort%sv(2, 1) = var_u                           ! B_12
      o_vort%se(2, 2) = var_zj;  o_vort%sv(2, 2) = var_zj; o_vort%scl(2, 2) = 1     ! opz B_33
      if (resistive) then
        o_vort%se(2, 3) = var_zj;  o_vort%sv(2, 3) = var_psi                       ! B_31
        o_vort%se(3, 2) = var_psi; o_vort%sv(3, 2) = var_zj; o_vort%scl(3, 2) = 1   ! opz B_13
        o_vort%se(3, 3) = var_psi; o_vort%sv(3, 3) = var_psi                       ! B_11
      endif
      call sfo_define(o_lap, "Laplacian B_42", 2, [var_w])
      o_lap%sv(1, 1) = var_u; o_lap%wid(1, 1) = 1                                  ! B_42 + D_bc
      call sfo_define(o_pj,  "pair_psi", 3, [var_psi, var_zj])
      call sfo_define(o_rho, "B_55", 4, [var_rho])
      call sfo_define(o_T,   "B_66", 5, [var_T])
      call sfo_define(o_mj,  "B_33", 6, [var_zj])
      call sfo_define(o_mw,  "B_44", NDIAG, [var_w])
      do k_ = 1, NCB
        o_cb(k_)%label = cn(k_); o_cb(k_)%nf = 1; o_cb(k_)%tag = NDIAG + k_
        o_cb(k_)%se(1, 1) = ce(k_); o_cb(k_)%sv(1, 1) = cv(k_)
      enddo
      ! the resistive bending: B_11, B_13, B_31 and the W sources M_eta, K_eta alone
      do k_ = 1, NRB
        o_rb(k_)%label = rn(k_); o_rb(k_)%nf = 1; o_rb(k_)%tag = NDIAG + NCB + k_
        o_rb(k_)%se(1, 1) = re(k_); o_rb(k_)%sv(1, 1) = rv(k_)
      enddo
      o_meta%label = "M_eta"; o_meta%nf = 1; o_meta%tag = NDIAG + NCB + NRB + 1; o_meta%wid(1, 1) = 1
      o_keta%label = "K_eta"; o_keta%nf = 1; o_keta%tag = NDIAG + NCB + NRB + 2; o_keta%wid(1, 1) = 1
      ! v_par: B_77 and the couplings B_57, B_67 (V's column in the rho and
      ! T rows), B_71, B_72 (V's row); tags after K_1's (K_eta's + 1)
      if (vpar) then
        call sfo_define(o_v77, "B_77", NDIAG + NCB + NRB + 4, [var_Vpar])
        do k_ = 1, NVC
          o_vc(k_)%label = vn(k_); o_vc(k_)%nf = 1; o_vc(k_)%tag = NDIAG + NCB + NRB + 4 + k_
          o_vc(k_)%se(1, 1) = ve(k_); o_vc(k_)%sv(1, 1) = vv(k_)
        enddo
      endif
    end subroutine define_operators

    subroutine create_work_vectors()
      integer :: k_
      call MatCreateVecs(o_vort%mat, rhs_V, sol_V, ierr)
      call MatCreateVecs(o_pj%mat, rhs_PJ, sol_PJ, ierr)
      call MatCreateVecs(o_rho%mat, sv_x(1), PETSC_NULL_VEC, ierr)
      do k_ = 2, NSV
        call VecDuplicate(sv_x(1), sv_x(k_), ierr)
      enddo
      do k_ = 1, NSV
        call VecDuplicate(sv_x(1), sv_y(k_), ierr)
      enddo
      call VecDuplicate(sv_x(1), m_j, ierr)
      call VecDuplicate(sv_x(1), w1, ierr)
      call VecDuplicate(sv_x(1), w2, ierr)
      call VecDuplicate(sv_x(1), w3, ierr)
      call VecDuplicate(sv_x(1), zv, ierr)
      call VecSet(zv, 0.0d0, ierr)
      vecs_ready = .true.
    end subroutine create_work_vectors

    subroutine build_fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: ", msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine build_fatal

  end subroutine vs_build

  !> y = the preconditioner applied to x, steps 1 to 7 of the module header.
  subroutine vs_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T
    real*8 :: t0, t1

    ierr = 0
    call PetscLogEventBegin(pcev_apply, ierr)
    t0 = MPI_Wtime()
    call sfo_vars_in(x, sv_x)
    t_scat = t_scat + (MPI_Wtime() - t0)
    x_psi = sv_x(var_psi); x_u   = sv_x(var_u);   x_j = sv_x(var_zj)
    x_w   = sv_x(var_w);   x_rho = sv_x(var_rho); x_T = sv_x(var_T)
    y_psi = sv_y(var_psi); y_u   = sv_y(var_u);   y_j = sv_y(var_zj)
    y_w   = sv_y(var_w);   y_rho = sv_y(var_rho); y_T = sv_y(var_T)

    ! 1. m_j = B_33^-1 x_j, m_omega = B_44^-1 x_omega (held in y_w)
    call PetscLogEventBegin(pcev_solve_mass, ierr)
    call mass_cheb_solve(mc_j, x_j, m_j)
    call mass_cheb_solve(mc_w, x_w, y_w)
    call PetscLogEventEnd(pcev_solve_mass, ierr)

    ! 2. the vorticity system (P_lumped) on (x_u - B_23 m_j - B_24 m_omega, 0[, 0])
    call VecCopy(x_u, w1, ierr)
    call sub_mult(CB_23, m_j, w1)
    call sub_mult(CB_24, y_w, w1)
    select case (o_vort%nf)
    case (1); call sf_split_parts(rhs_V, [w1], .true.)
    case (2); call sf_split_parts(rhs_V, [w1, zv], .true.)
    case (3); call sf_split_parts(rhs_V, [w1, zv, zv], .true.)
    end select
    call sf_solver_solve(slv_vort, rhs_V, sol_V, pcev_solve_vort)
    call sf_split_parts(sol_V, [w2], .false.)

    ! 3. omega = m_omega + w; B_42 u = x_omega - B_44 omega
    call VecAXPY(y_w, 1.0d0, w2, ierr)
    call MatMult(o_mw%mat, y_w, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_w, ierr)
    call sf_solver_solve(slv_lap, w1, y_u, pcev_solve_lap)

    ! 4. pair_psi (psi, j) = (x_psi - B_12 u, x_j)
    call MatMult(o_cb(CB_12)%mat, y_u, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_psi, ierr)
    call sf_split_parts(rhs_PJ, [w1, x_j], .true.)
    call sf_solver_solve(slv_pj, rhs_PJ, sol_PJ, pcev_solve_pj)
    call sf_split_parts(sol_PJ, [y_psi, y_j], .false.)

    ! 4b. v_par: P_V V = x_V - B_71 psi - B_72 u
    if (vpar) then
      call MatMult(o_vc(VC_71)%mat, y_psi, w1, ierr)
      call VecAYPX(w1, -1.0d0, sv_x(SV_V), ierr)
      call sub_vmult(VC_72, y_u, w1)
      call sf_solver_solve(slv_pv, w1, sv_y(SV_V), pcev_solve_pv)
    endif

    ! 5. B_55 rho = x_rho - B_51 psi - B_52 u [- B_57 V]
    call MatMult(o_cb(CB_51)%mat, y_psi, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_rho, ierr)
    call sub_mult(CB_52, y_u, w1)
    if (vpar) call sub_vmult(VC_57, sv_y(SV_V), w1)
    call sf_solver_solve(slv_rho, w1, y_rho, pcev_solve_rhot)

    ! 6. B_66 T = x_T - B_61 psi - B_63 j - B_62 u - B_65 rho [- B_67 V]
    call MatMult(o_cb(CB_61)%mat, y_psi, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_T, ierr)
    call sub_mult(CB_63, y_j, w1)
    call sub_mult(CB_62, y_u, w1)
    if (use_b65) call sub_mult(CB_65, y_rho, w1)
    if (vpar) call sub_vmult(VC_67, sv_y(SV_V), w1)
    call sf_solver_solve(slv_T, w1, y_T, pcev_solve_rhot)

    ! 7.
    t1 = MPI_Wtime()
    call sfo_vars_out(sv_y, y)
    t_scat  = t_scat + (MPI_Wtime() - t1)
    if (pen%n > 0) call penalty_backsub(pen, x, y)
    t_apply = t_apply + (MPI_Wtime() - t0)
    call PetscLogEventEnd(pcev_apply, ierr)
    ierr = 0

  contains

    !> r -= B_k v
    subroutine sub_mult(k, v, r)
      integer, intent(in) :: k
      Vec :: v, r
      call MatMult(o_cb(k)%mat, v, w3, ierr)
      call VecAXPY(r, -1.0d0, w3, ierr)
    end subroutine sub_mult

    !> r -= B_k v, v_par's couplings
    subroutine sub_vmult(k, v, r)
      integer, intent(in) :: k
      Vec :: v, r
      call MatMult(o_vc(k)%mat, v, w3, ierr)
      call VecAXPY(r, -1.0d0, w3, ierr)
    end subroutine sub_vmult

  end subroutine vs_apply

  !> Inner-iteration summary, one line per block (mode split: per family),
  !! then reset. Collective.
  subroutine vs_report(my_id)
    integer, intent(in) :: my_id
    if (famode .and. vpar) then
      call sf_solver_family_report("VS", [slv_vort, slv_lap, slv_pj, slv_pv, slv_rho, slv_T], t_apply, t_scat, &
                                   comm_g)
    else if (famode) then
      call sf_solver_family_report("VS", [slv_vort, slv_lap, slv_pj, slv_rho, slv_T], t_apply, t_scat, comm_g)
    else
      call sf_solver_report(slv_vort, my_id)
      call sf_solver_report(slv_lap,  my_id)
      call sf_solver_report(slv_pj,   my_id)
      if (vpar) call sf_solver_report(slv_pv, my_id)
      call sf_solver_report(slv_rho,  my_id)
      call sf_solver_report(slv_T,    my_id)
    endif
    call sf_solver_reset_counters(slv_vort)
    call sf_solver_reset_counters(slv_lap)
    call sf_solver_reset_counters(slv_pj)
    call sf_solver_reset_counters(slv_pv)
    call sf_solver_reset_counters(slv_rho)
    call sf_solver_reset_counters(slv_T)
    t_apply = 0.d0;  t_scat = 0.d0
  end subroutine vs_report

#endif
end module mod_petsc_pc_vs
