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

    !> True sub-blocks of the full system for T3/T5a reference solves
    !! (extracted from the full Jacobian; independent copies — nothing
    !!  shared with g_ctx of the old physics PC).
    Mat :: A_alfven_true      !< 2x2 (psi,u) MatNest->AIJ from the full Jacobian
    IS  :: is_var(6)
    logical :: is_created = .false.

    !> Solvers used by the analysis (created on demand)
    KSP :: ksp_P_u            !< Cholesky/MUMPS on P_u (T2, T3, T5a)
    KSP :: ksp_M_psi          !< consistent-mass solve on M_psi (T3, T5a)
    KSP :: ksp_alfven_ref     !< MUMPS LU on A_alfven_true (T3 reference)
    logical :: ksp_created = .false.

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
