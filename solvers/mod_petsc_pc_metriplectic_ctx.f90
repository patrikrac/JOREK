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
    Mat :: B_31s, B_33s       !< constraint row blocks (j recovery; B_33s factored once)
    Mat :: B_42s, B_44s       !< constraint row blocks (w recovery; B_44s factored once)
    Mat :: B_52s, B_55s       !< rho recovery: drho = B55^-1 (r_rho - B52 du)
    Mat :: B_62s, B_66s       !< T   recovery: dT   = B66^-1 (r_T   - B62 du)
    Mat :: Pu_aij             !< AIJ copy of composed P_u bound to ksp_Pu
    Mat :: A_kideal           !< coupled model-Alfven K-half
                              !! [(1+z)M_psi, tdt*D; tdt*Dp, -(1+z)L_rho] (tdt = theta*dt)

    !> Sweep solvers (created/refreshed by metriplectic_build_sweep)
    KSP :: ksp_pair_psij      !< MUMPS LU on A_pair_psij
    KSP :: ksp_pair_uw        !< MUMPS LU on A_pair_uw
    KSP :: ksp_kideal         !< MUMPS LU on A_kideal (exact coupled K-half solve)
    KSP :: ksp_Pu             !< MUMPS Cholesky on Pu_aij (refresh via refresh_Pu)
    KSP :: ksp_Mpsi           !< CG+Jacobi consistent-mass solve on M_psi
    KSP :: ksp_B33, ksp_B44   !< MUMPS LU, factored ONCE (constant constraint masses)
    KSP :: ksp_B55, ksp_B66   !< MUMPS LU on rho/T diagonal blocks (refreshed)
    logical :: sweep_ready       = .false.
    logical :: sweep_once_done   = .false.  !< B33/B44 factored-once guard
    real*8  :: tau_Pu            = -1.d0    !< tau at which ksp_Pu is factored
    logical :: ideal_coupled     = .true.   !< K-half solve: coupled 2x2 MUMPS (exact,
                                            !! default) vs P_u Schur path (T3c tail;
                                            !! iterative upgrade route)

    !> Sweep work vectors: 2-var packed pair vecs + rho/T sized 1-var vecs
    Vec :: wv_pair_psij_1, wv_pair_psij_2
    Vec :: wv_pair_uw_1,   wv_pair_uw_2
    Vec :: wv_kid_1, wv_kid_2
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
