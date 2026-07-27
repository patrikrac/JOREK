module mod_petsc_pc_metriplectic_ctx
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  public

  ! Shared context for the metriplectic HSS PC family (design doc
  ! docs/superpowers/specs/2026-07-09-metriplectic-hss-pc-design.md).
  ! Slice A holds only the analysis operators; apply-path fields arrive
  ! with Slices B/C.
  !
  ! Coefficient conventions (R-exponents, signs, Gears normalization):
  ! docs/notes/metriplectic_parabolization_note.tex.

  type :: type_metriplectic_ctx
    logical :: initialized     = .false.
    logical :: matrices_ready  = .false.
    integer :: comm            = -1

    !> Element-assembled operators (1-var BAIJ, block_size = n_tor).
    !! Continuous-Schur ingredients, constraints substituted (spec 1.2):
    Mat :: M_psi     !< psi-row mass, weight 1/R (note, table row 1)
    Mat :: D_op      !< weak R^2(B.grad), 1/R-tested : psi-row, u-column
    Mat :: Dp_op     !< weak R(B.grad)Delta* : u-row, psi-column (direct form,
                     !! note eq. (Dhat-direct); needs 2nd derivs of trial psi)
    Mat :: Dp_struct !< by-parts form of the same operator (note eq.
                     !! (Dhat-parts) interior part); T1 comparison partner
    Mat :: L_rho     !< A_rho = rho-weighted vorticity stiffness (SPD part of P_u)
    Mat :: W_para    !< parabolized field-aligned form (B.grad)^T W Delta* (B.grad)
    Mat :: P_u       !< L_rho + tau^2 * W_para  (composed, not re-assembled)
    logical :: P_u_created = .false.

    !> Variable index sets on the full Jacobian's layout (shared by the
    !! analysis checks and the production sweep; persistent once created).
    IS  :: is_var(6)
    logical :: is_created = .false.

    !> --- Sweep operators (stage B/C, spec Sec. 6; note Sec. "sweep") ---
    !! True-Jacobian sub-blocks, extracted copies owned by the sweep
    !! (suffix s; independent of the analysis module's m_* handles).
    Mat :: A_pair_psij        !< 2x2 [B11 B13; B31 B33] MatNest->AIJ (S-half, psi/j pair)
    Mat :: A_pair_uw          !< 2x2 [B22 B24; B42 B44] MatNest->AIJ (S-half, u/w pair)
    Mat :: B_11s              !< psi-row diagonal block, sweep-owned copy (commutator
                              !! candidate M1x = the true amat_11; only when build_cm)
    Mat :: B_31s, B_33s       !< constraint row blocks (j recovery; B_33s factored once)
    Mat :: B_42s, B_44s       !< constraint row blocks (w recovery; B_44s factored once)
    Mat :: B_52s, B_55s       !< rho recovery: drho = B55^-1 (r_rho - B52 du)
    Mat :: B_62s, B_66s       !< T   recovery: dT   = B66^-1 (r_T   - B62 du)
    Mat :: Pu_aij             !< AIJ copy of composed P_u bound to ksp_Pu
    Mat :: A_kideal           !< coupled model-Alfven K-half
                              !! [(1+z)M_psi, tdt*D; tdt*Dp, -(1+z)L_rho] (tdt = theta*dt)

    !> --- Stage-D pair-Schur objects (spec Sec. 7.3; note Sec. "pschur") ---
    Mat :: A_k4               !< 4-field model K (psi,u,j,w pack_4v order; true pair
                              !! blocks + tdt*D/tdt*Dp coupling) -- reference/fallback LU
    Mat :: P_uw               !< sparse Schur PC on the (u,w) pair:
                              !! [B22 - (1+z)tau^2 W_para, B24; B42, B44] (note eq. (Puw))
    Mat :: S_uw_shell         !< matrix-free exact Schur S_uw = A_uw - tdt^2 [Dp R^-1 D]_uu
                              !! (one (psi,j) pair solve per mult; note eq. (Suw))

    !> Sweep solvers (created/refreshed by metriplectic_build_sweep)
    KSP :: ksp_pair_psij      !< MUMPS LU on A_pair_psij
    KSP :: ksp_pair_uw        !< MUMPS LU on A_pair_uw
    KSP :: ksp_kideal         !< MUMPS LU on A_kideal (exact coupled K-half solve)
    KSP :: ksp_Pu             !< MUMPS Cholesky on Pu_aij (refresh via refresh_Pu)
    KSP :: ksp_Mpsi           !< CG+Jacobi consistent-mass solve on M_psi
    KSP :: ksp_B33, ksp_B44   !< MUMPS LU, factored ONCE (constant constraint masses)
    KSP :: ksp_B55, ksp_B66   !< MUMPS LU on rho/T diagonal blocks (refreshed)
    KSP :: ksp_k4             !< MUMPS LU on A_k4 (reference/fallback, mode 'K4')
    KSP :: ksp_Puw            !< MUMPS LU on P_uw (mode 'PS')
    KSP :: ksp_Suw            !< FGMRES on S_uw_shell, Pmat = P_uw (inner Schur solve)
    logical :: sweep_ready       = .false.
    logical :: sweep_once_done   = .false.  !< B33/B44 factored-once guard
    real*8  :: tau_Pu            = -1.d0    !< tau at which ksp_Pu is factored
    character(len=2) :: khalf_mode = 'PS'   !< K-half solve mode (spec Sec. 7.3):
                                            !! 'PS' pair-Schur LDU (default), 'K4'
                                            !! coupled 4-field LU (reference/fallback),
                                            !! 'K2' coupled 2x2 model-Alfven LU,
                                            !! 'PU' segregated P_u Schur path
    integer :: ps_inner_it       = 0        !< 'PS' inner Schur iterations (0 = single
                                            !! pass P_uw^-1)
    real*8  :: ps_inner_tol      = 1.d-2    !< 'PS' inner Schur relative tolerance

    !> --- Commutator-device M_* Schur (Slice 1; note Sec. 6.2, Eq. (39)) ---
    !! Replaces step 3 of the PS-LDU by
    !!   [B22 M_* - tdt^2 W_para, B24; B42 M_*, B44] (chi_u, dw) = (g_u, r_w)
    !!   du = M_* chi_u,     M_* = Q_u^-1 A_M*
    !! At M0 (A_M* = opz*Q_u) this is (1+zeta)*P_uw and du = opz*chi_u, i.e.
    !! bit-identical to the ps_inner_it=0 path. The candidate binding
    !! (coefficients, operator handles, counters) lives in the apply module
    !! next to the shell internals, as ps_shell_ready/s_* do.
    Mat :: S_cm_shell         !< matrix-free T_pair on the (u,w) layout
    KSP :: ksp_Scm            !< FGMRES on S_cm_shell, PC = ksp_Puw / (1+zeta)
    integer :: cm_inner_it    = 20       !< inner FGMRES cap
    real*8  :: cm_inner_tol   = 1.d-6    !< inner FGMRES relative tolerance

    !> Sweep work vectors: 2-var packed pair vecs + rho/T sized 1-var vecs
    Vec :: wv_pair_psij_1, wv_pair_psij_2
    Vec :: wv_pair_uw_1,   wv_pair_uw_2
    Vec :: wv_kid_1, wv_kid_2
    Vec :: wv_k4_1, wv_k4_2   !< 4-var packed (mode 'K4' reference solve)
    Vec :: wv_uw_1, wv_uw_2   !< 2-var (u,w) packed ('PS' Schur RHS/solution)
    Vec :: wv_rho_1, wv_rho_2, wv_T_1, wv_T_2

    !> Work vectors (1-var sized), created with the matrices
    Vec :: wv_psi_1, wv_psi_2, wv_u_1, wv_u_2
    logical :: work_created = .false.

    !> Gears-normalized time factor tau = theta*dt/(1+zeta) at assembly
    !! (note, Sec. "Elimination": tau replaces theta*dt everywhere).
    real*8 :: dt_theta = 0.d0
  end type type_metriplectic_ctx

  type(type_metriplectic_ctx), save :: g_mctx

#endif
end module mod_petsc_pc_metriplectic_ctx
