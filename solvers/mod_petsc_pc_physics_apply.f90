module mod_petsc_pc_physics_apply
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  implicit none
  private

  public :: physics_pc_apply
  public :: k_a_exact_mult
  public :: s_pbp_diag_mult   ! TEMPORARY diagnostic (Option A, Step 2) -- remove with Step 3
  public :: pack_2v, unpack_2v   ! Workstream B: also used by the mixed-pair null test

contains

  !> Pack two 1-variable PETSc vecs into one 2v-sized packed vec.
  !!
  !! y_packed must already exist with size = size(x1) + size(x2).
  !! Layout: top half = x1, bottom half = x2.
  subroutine pack_2v(x1, x2, y_packed, ierr)
    Vec            :: x1, x2, y_packed
    PetscErrorCode :: ierr

    PetscScalar, pointer :: a_x1(:), a_x2(:), a_y(:)
    PetscInt :: n1, n2

    call VecGetLocalSize(x1, n1, ierr)
    call VecGetLocalSize(x2, n2, ierr)

    call VecGetArrayRead(x1, a_x1, ierr)
    call VecGetArrayRead(x2, a_x2, ierr)
    call VecGetArray    (y_packed, a_y, ierr)

    a_y(1     : n1     ) = a_x1(1:n1)
    a_y(n1+1  : n1+n2  ) = a_x2(1:n2)

    call VecRestoreArrayRead(x1, a_x1, ierr)
    call VecRestoreArrayRead(x2, a_x2, ierr)
    call VecRestoreArray    (y_packed, a_y, ierr)
  end subroutine pack_2v

  !> Inverse of pack_2v: unpack a 2v-sized packed vec into two 1v vecs.
  subroutine unpack_2v(x_packed, y1, y2, ierr)
    Vec            :: x_packed, y1, y2
    PetscErrorCode :: ierr

    PetscScalar, pointer :: a_x(:), a_y1(:), a_y2(:)
    PetscInt :: n1, n2

    call VecGetLocalSize(y1, n1, ierr)
    call VecGetLocalSize(y2, n2, ierr)

    call VecGetArrayRead(x_packed, a_x, ierr)
    call VecGetArray    (y1, a_y1, ierr)
    call VecGetArray    (y2, a_y2, ierr)

    a_y1(1:n1) = a_x(1     : n1     )
    a_y2(1:n2) = a_x(n1+1  : n1+n2  )

    call VecRestoreArrayRead(x_packed, a_x, ierr)
    call VecRestoreArray    (y1, a_y1, ierr)
    call VecRestoreArray    (y2, a_y2, ierr)
  end subroutine unpack_2v

  !> MATSHELL MATOP_MULT callback: y = K_A_exact * x  (exact (psi,u) Schur operator).
  !!
  !!   unpack x -> (p, q)
  !!   tj = M_jj^-1 (B_31 p);   tw = M_ww^-1 (B_42 q)
  !!   y_psi = B_11 p + B_12 q - B_13 tj
  !!   y_u   = B_21 p + B_22 q - B_23 tj - B_24 tw
  !!   pack (y_psi, y_u) -> y
  !!
  !! Work vecs live in g_ctx (created once by setup_k_a_exact_shell). The shell
  !! context is unused; all state is read from the g_ctx singleton.
  subroutine k_a_exact_mult(A_shell, x, y, ierr)
    Mat            :: A_shell
    Vec            :: x, y
    PetscErrorCode :: ierr

    ! 1. unpack the packed 2v input into (p, q)
    call unpack_2v(x, g_ctx%kae_p, g_ctx%kae_q, ierr)

    ! 2. exact constraint eliminations
    call MatMult(g_ctx%B_31, g_ctx%kae_p, g_ctx%kae_sj, ierr)      ! sj = B_31 p
    call KSPSolve(g_ctx%ksp_Mj, g_ctx%kae_sj, g_ctx%kae_tj, ierr)  ! tj = M_jj^-1 sj
    call MatMult(g_ctx%B_42, g_ctx%kae_q, g_ctx%kae_sw, ierr)      ! sw = B_42 q
    call KSPSolve(g_ctx%ksp_Mw, g_ctx%kae_sw, g_ctx%kae_tw, ierr)  ! tw = M_ww^-1 sw

    ! 3. y_psi = B_11 p + B_12 q - B_13 tj
    call MatMult(g_ctx%B_11, g_ctx%kae_p, g_ctx%kae_ypsi, ierr)
    call MatMultAdd(g_ctx%B_12, g_ctx%kae_q, g_ctx%kae_ypsi, g_ctx%kae_ypsi, ierr)
    call MatMult(g_ctx%B_13, g_ctx%kae_tj, g_ctx%kae_spsi, ierr)
    call VecAXPY(g_ctx%kae_ypsi, -1.0d0, g_ctx%kae_spsi, ierr)

    ! 4. y_u = B_21 p + B_22 q - B_23 tj - B_24 tw
    call MatMult(g_ctx%B_21, g_ctx%kae_p, g_ctx%kae_yu, ierr)
    call MatMultAdd(g_ctx%B_22, g_ctx%kae_q, g_ctx%kae_yu, g_ctx%kae_yu, ierr)
    call MatMult(g_ctx%B_23, g_ctx%kae_tj, g_ctx%kae_su, ierr)
    call VecAXPY(g_ctx%kae_yu, -1.0d0, g_ctx%kae_su, ierr)
    call MatMult(g_ctx%B_24, g_ctx%kae_tw, g_ctx%kae_su, ierr)
    call VecAXPY(g_ctx%kae_yu, -1.0d0, g_ctx%kae_su, ierr)

    ! 5. pack outputs
    call pack_2v(g_ctx%kae_ypsi, g_ctx%kae_yu, y, ierr)

    ierr = 0
  end subroutine k_a_exact_mult

  !> MATSHELL MATOP_MULT callback: y = S_PBP_diag * x   (TEMPORARY, Option A).
  !!
  !! Diagnostic momentum-Schur approximation that replaces ONLY the magnetic-
  !! channel inner inverse Atilde_11^{-1} by the consistent-mass weight
  !! (1+zeta)^{-1} M_j^{-1} (M_j = B_33, already factored as ksp_Mj); channels
  !! B and C are kept EXACT via dedicated B_55/B_66 MUMPS solves. Mirrors
  !! compute_full_momentum_schur_exact term-by-term, so spec(S_PBP_diag^{-1} S_u)
  !! isolates purely the quality of Atilde_11^{-1} -> (1+zeta)^{-1} M_j^{-1}.
  !!
  !!   ypsi = (1+zeta)^{-1} M_j^{-1} (B_12 x)
  !!   y    = Atilde_22 x
  !!        - Atilde_21 ypsi                                      (A: magnetic)
  !!        - B_25 B_55^{-1}(B_52 x) - B_26 B_66^{-1}(B_62 x)     (B: pressure)
  !!        + B_25 B_55^{-1}(B_51 ypsi)                           (C: pressure-flutter)
  !!        + B_26 B_66^{-1}(Atilde_61 ypsi)                      (C: thermal-flutter)
  !!
  !! Work vecs live in g_ctx (created once by setup_S_PBP_diag_shell). The shell
  !! context is unused; all state is read from the g_ctx singleton.
  !! TO BE REPLACED by the production S_PBP operator (Step 3).
  subroutine s_pbp_diag_mult(A_shell, x, y, ierr)
    Mat            :: A_shell
    Vec            :: x, y
    PetscErrorCode :: ierr

    ! diagonal: y = Atilde_22 x
    call MatMult(g_ctx%Atilde_22, x, y, ierr)

    ! consistent-mass psi response: ypsi = (1+zeta)^-1 M_j^-1 (B_12 x)
    call MatMult (g_ctx%B_12,  x,               g_ctx%spbpd_zpsi, ierr)
    call KSPSolve(g_ctx%ksp_Mj, g_ctx%spbpd_zpsi, g_ctx%spbpd_ypsi, ierr)
    call VecScale(g_ctx%spbpd_ypsi, g_ctx%spbpd_inv_gears, ierr)

    ! per-harmonic toroidal resistive relaxation: approximates Atilde_11^-1's
    ! resistive skin WITHOUT a solve. Applied to ypsi so BOTH channel A and the
    ! reused channel C see the relaxed response. Toggle: spbpd_use_toroidal_relax.
    if (g_ctx%spbpd_use_toroidal_relax) &
      call VecPointwiseMult(g_ctx%spbpd_ypsi, g_ctx%spbpd_ypsi, g_ctx%spbpd_relax, ierr)

    ! channel A (magnetic): y -= Atilde_21 ypsi
    call MatMult(g_ctx%Atilde_21, g_ctx%spbpd_ypsi, g_ctx%spbpd_ru, ierr)
    call VecAXPY(y, -1.0d0, g_ctx%spbpd_ru, ierr)

    ! channel B (rho): y -= B_25 B_55^-1 (B_52 x)
    call MatMult (g_ctx%B_52, x,                  g_ctx%spbpd_zrho, ierr)
    call KSPSolve(g_ctx%spbpd_ksp_B55, g_ctx%spbpd_zrho, g_ctx%spbpd_yrho, ierr)
    call MatMult (g_ctx%B_25, g_ctx%spbpd_yrho,   g_ctx%spbpd_ru,   ierr)
    call VecAXPY(y, -1.0d0, g_ctx%spbpd_ru, ierr)

    ! channel B (T): y -= B_26 B_66^-1 (B_62 x)
    call MatMult (g_ctx%B_62, x,                  g_ctx%spbpd_zT, ierr)
    call KSPSolve(g_ctx%spbpd_ksp_B66, g_ctx%spbpd_zT, g_ctx%spbpd_yT, ierr)
    call MatMult (g_ctx%B_26, g_ctx%spbpd_yT,     g_ctx%spbpd_ru, ierr)
    call VecAXPY(y, -1.0d0, g_ctx%spbpd_ru, ierr)

    ! channel C (pressure-flutter): y += B_25 B_55^-1 (B_51 ypsi)
    call MatMult (g_ctx%B_51, g_ctx%spbpd_ypsi,   g_ctx%spbpd_zrho, ierr)
    call KSPSolve(g_ctx%spbpd_ksp_B55, g_ctx%spbpd_zrho, g_ctx%spbpd_yrho, ierr)
    call MatMult (g_ctx%B_25, g_ctx%spbpd_yrho,   g_ctx%spbpd_ru,   ierr)
    call VecAXPY(y, +1.0d0, g_ctx%spbpd_ru, ierr)

    ! channel C (thermal-flutter): y += B_26 B_66^-1 (Atilde_61 ypsi)
    call MatMult (g_ctx%Atilde_61, g_ctx%spbpd_ypsi, g_ctx%spbpd_zT, ierr)
    call KSPSolve(g_ctx%spbpd_ksp_B66, g_ctx%spbpd_zT, g_ctx%spbpd_yT, ierr)
    call MatMult (g_ctx%B_26, g_ctx%spbpd_yT,     g_ctx%spbpd_ru, ierr)
    call VecAXPY(y, +1.0d0, g_ctx%spbpd_ru, ierr)

    ierr = 0
  end subroutine s_pbp_diag_mult

  !> Compute the j,w-folded Alfven-row residuals.
  !!   r_psi = x_psi - B_13 * temp_j
  !!   r_u   = x_u   - B_23 * temp_j - B_24 * temp_w
  !! where temp_j = g_ctx%work_1, temp_w = g_ctx%work_2 (set by the dispatcher).
  !! `scratch` must be a work vec distinct from r_psi, r_u, work_1, work_2.
  !! (The rho row needs no fold: r_rho = x_rho. The T row fold, r_T = x_T - B_63*temp_j,
  !!  is a single MatMult and is done inline by callers.)
  subroutine fold_alfven_residual(x_psi, x_u, r_psi, r_u, scratch, ierr)
    Vec            :: x_psi, x_u, r_psi, r_u, scratch
    PetscErrorCode :: ierr

    ! r_psi = x_psi - B_13 * temp_j
    call MatMult(g_ctx%B_13, g_ctx%work_1, scratch, ierr)
    call VecWAXPY(r_psi, -1.0d0, scratch, x_psi, ierr)

    ! r_u = x_u - B_23 * temp_j - B_24 * temp_w
    call MatMult(g_ctx%B_23, g_ctx%work_1, scratch, ierr)
    call VecWAXPY(r_u, -1.0d0, scratch, x_u, ierr)
    call MatMult(g_ctx%B_24, g_ctx%work_2, scratch, ierr)
    call VecAXPY(r_u, -1.0d0, scratch, ierr)
  end subroutine fold_alfven_residual

  !> Block-Jacobi apply: y_A = K_A^-1 x_A, y_rho = B_55^-1 x_rho, y_T = B_66^-1 x_T
  !! (parallel, no coupling).
  !!
  !! Mode 1 of the sub-blocks PC. Independent solves on the (psi,u) Alfven block
  !! and the separate rho (B_55) and T (B_66) transport blocks — cross-block
  !! coupling is ignored.
  subroutine apply_block_jacobi(x_psi, x_u, x_rho, x_T, &
                                y_psi, y_u, y_rho, y_T, ierr)
    Vec            :: x_psi, x_u, x_rho, x_T
    Vec            :: y_psi, y_u, y_rho, y_T
    PetscErrorCode :: ierr

    ! Solve the Alfven super-block (packed psi,u).
    call pack_2v(x_psi, x_u, g_ctx%rhs_A, ierr)
    call KSPSolve(g_ctx%ksp_block_A, g_ctx%rhs_A, g_ctx%sol_A, ierr)
    call unpack_2v(g_ctx%sol_A, y_psi, y_u, ierr)

    ! Independent transport-block solves (block-Jacobi: no cross coupling).
    call KSPSolve(g_ctx%ksp_rho, x_rho, y_rho, ierr)
    call KSPSolve(g_ctx%ksp_T,   x_T,   y_T,   ierr)
  end subroutine apply_block_jacobi

  !> Block-GS-forward apply: solve K_A, propagate to (rho,T), solve K_B.
  !!
  !! Mode 2 of the sub-blocks PC. Forward Gauss-Seidel sweep:
  !!   1. y_A = K_A^-1 x_A
  !!   2. r_B = x_B - C_BA * y_A      where C_BA = [B_51 B_52; Atilde_61 B_62]
  !!                                  (RHS uses j,w-folded residuals)
  !!   3. y_rho = B_55^-1 r_rho, y_T = B_66^-1 r_T   (B is block-diagonal in rho,T)
  !!
  !! Captures the (psi,u) -> (rho,T) coupling via the C_BA submatrices.
  subroutine apply_block_gs_forward(x_psi, x_u, x_rho, x_T, &
                                    y_psi, y_u, y_rho, y_T, ierr)
    Vec            :: x_psi, x_u, x_rho, x_T
    Vec            :: y_psi, y_u, y_rho, y_T
    PetscErrorCode :: ierr

    ! 1. Forward sweep on the Alfven super-block, using the j,w-folded RHS.
    !    work_4 = r_psi, work_5 = r_u  (work_3 = scratch; work_1/work_2 = temp_j/temp_w preserved)
    call fold_alfven_residual(x_psi, x_u, g_ctx%work_4, g_ctx%work_5, g_ctx%work_3, ierr)
    call pack_2v(g_ctx%work_4, g_ctx%work_5, g_ctx%rhs_A, ierr)
    call KSPSolve(g_ctx%ksp_block_A, g_ctx%rhs_A, g_ctx%sol_A, ierr)
    call unpack_2v(g_ctx%sol_A, y_psi, y_u, ierr)

    ! 2a. rho residual (no j,w fold): tmp_rho = x_rho - B_51*y_psi - B_52*y_u
    call VecCopy(x_rho, g_ctx%tmp_rho, ierr)
    call MatMult(g_ctx%B_51, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_rho, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_52, y_u,   g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_rho, -1.0d0, g_ctx%work_3, ierr)

    ! 2b. T residual with j,w fold and Schur-corrected Atilde_61:
    !     tmp_T = (x_T - B_63*temp_j) - Atilde_61*y_psi - B_62*y_u
    call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%tmp_T, -1.0d0, g_ctx%work_3, x_T, ierr)
    call MatMult(g_ctx%Atilde_61, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_T, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_62, y_u, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_T, -1.0d0, g_ctx%work_3, ierr)

    ! 3. Solve the independent transport blocks on their folded residuals.
    call KSPSolve(g_ctx%ksp_rho, g_ctx%tmp_rho, y_rho, ierr)
    call KSPSolve(g_ctx%ksp_T,   g_ctx%tmp_T,   y_T,   ierr)
  end subroutine apply_block_gs_forward

  !> Block-GS-symmetric apply: forward sweep + backward sweep.
  !!
  !! Mode 3 of the sub-blocks PC. Symmetric Gauss-Seidel — three block solves total.
  !!
  !!   Forward sweep (same as mode 2):
  !!     y_A = K_A^-1 x_A
  !!     r_B = x_B - C_BA * y_A
  !!     y_B = K_B^-1 r_B
  !!
  !!   Backward sweep:
  !!     r_A = x_A - C_AB * y_B    where C_AB = [0 B_16; B_25 B_26]
  !!                                            (B_15 is structurally zero in model199)
  !!     y_A = K_A^-1 r_A
  !!
  subroutine apply_block_gs_symmetric(x_psi, x_u, x_rho, x_T, &
                                      y_psi, y_u, y_rho, y_T, ierr)
    Vec            :: x_psi, x_u, x_rho, x_T
    Vec            :: y_psi, y_u, y_rho, y_T
    PetscErrorCode :: ierr

    ! --- Forward sweep (identical to apply_block_gs_forward) ---
    call apply_block_gs_forward(x_psi, x_u, x_rho, x_T, &
                                y_psi, y_u, y_rho, y_T, ierr)

    ! --- Backward sweep ---
    ! Re-solve the Alfven block with the dropped upper coupling K_alpha,beta applied as a
    ! corrector, on the SAME j,w-folded residual as the forward sweep.
    !   work_4 = r_psi - B_16*y_T
    !   work_5 = r_u   - B_25*y_rho - B_26*y_T
    ! NOTE: must NOT reuse work_1/work_2 here (they hold temp_j/temp_w needed by the fold).
    call fold_alfven_residual(x_psi, x_u, g_ctx%work_4, g_ctx%work_5, g_ctx%work_3, ierr)

    call MatMult(g_ctx%B_16, y_T, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)

    call MatMult(g_ctx%B_25, y_rho, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_26, y_T,   g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)

    call pack_2v(g_ctx%work_4, g_ctx%work_5, g_ctx%rhs_A, ierr)
    call KSPSolve(g_ctx%ksp_block_A, g_ctx%rhs_A, g_ctx%sol_A, ierr)
    call unpack_2v(g_ctx%sol_A, y_psi, y_u, ierr)
  end subroutine apply_block_gs_symmetric

  !--------------------------------------------------------------------
  !> Apply path: monolithic 4x4 reduced solve.
  !>
  !> Precondition: physics_pc_apply (the dispatcher) has already populated
  !>   g_ctx%work_1 = M_j^{-1} x_j   (temp_j)
  !>   g_ctx%work_2 = M_w^{-1} x_w   (temp_w)
  !--------------------------------------------------------------------
  subroutine apply_monolithic_4x4(x_psi, x_u, x_rho, x_T, &
                                  y_psi, y_u, y_rho, y_T, ierr)
    Vec, intent(in)    :: x_psi, x_u, x_rho, x_T
    Vec, intent(inout) :: y_psi, y_u, y_rho, y_T
    PetscErrorCode, intent(out) :: ierr

    Vec :: rhs_psi, rhs_u, rhs_rho, rhs_T
    Vec :: sol_psi, sol_u, sol_rho, sol_T

    ! b_psi = x_psi - B_13 * temp_j  → store in work_3
    call MatMult(g_ctx%B_13, g_ctx%work_1, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_psi, ierr)
    call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(1), rhs_psi, ierr)
    call VecCopy(g_ctx%work_4, rhs_psi, ierr)
    call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(1), rhs_psi, ierr)

    ! b_u = x_u - B_23 * temp_j - B_24 * temp_w  → store in work_5
    call MatMult(g_ctx%B_23, g_ctx%work_1, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_24, g_ctx%work_2, g_ctx%work_4, ierr)
    call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, x_u, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_4, ierr)
    call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(2), rhs_u, ierr)
    call VecCopy(g_ctx%work_5, rhs_u, ierr)
    call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(2), rhs_u, ierr)

    ! b_rho = x_rho (no Schur correction)
    call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(3), rhs_rho, ierr)
    call VecCopy(x_rho, rhs_rho, ierr)
    call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(3), rhs_rho, ierr)

    ! b_T = x_T - B_63 * temp_j  → store in work_4
    call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_T, ierr)
    call VecGetSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(4), rhs_T, ierr)
    call VecCopy(g_ctx%work_4, rhs_T, ierr)
    call VecRestoreSubVector(g_ctx%work_rhs_4v, g_ctx%is_reduced(4), rhs_T, ierr)

    ! Single monolithic solve
    call KSPSolve(g_ctx%ksp_reduced, g_ctx%work_rhs_4v, g_ctx%work_sol_4v, ierr)

    ! Scatter solution back to per-variable output vectors
    call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(1), sol_psi, ierr)
    call VecCopy(sol_psi, y_psi, ierr)
    call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(1), sol_psi, ierr)

    call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(2), sol_u, ierr)
    call VecCopy(sol_u, y_u, ierr)
    call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(2), sol_u, ierr)

    call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(3), sol_rho, ierr)
    call VecCopy(sol_rho, y_rho, ierr)
    call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(3), sol_rho, ierr)

    call VecGetSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(4), sol_T, ierr)
    call VecCopy(sol_T, y_T, ierr)
    call VecRestoreSubVector(g_ctx%work_sol_4v, g_ctx%is_reduced(4), sol_T, ierr)

    ierr = 0
  end subroutine apply_monolithic_4x4

  !--------------------------------------------------------------------
  !> Apply path: segregated-Schur (Chacon parabolization) predictor-corrector.
  !> Inverts the reduced 4x4 (psi,u,rho,T) WITHOUT factoring the coupled (psi,u)
  !> 2x2, via two separable solves of the Alfven pair:
  !>   Predictor: t_psi = Atilde_11^-1 r_psi ; y_u = S_PBP^-1 (r_u - Atilde_21 t_psi)
  !>   Corrector: y_psi = Atilde_11^-1 (r_psi - B_12 y_u)
  !>   Transport: y_rho = B_55^-1(r_rho - B_51 y_psi - B_52 y_u)
  !>              y_T   = B_66^-1(r_T   - Atilde_61 y_psi - B_62 y_u)
  !> If physics_pc_multi_step_symmetric, a backward corrector applies the dropped
  !> weak feedback K_alpha,beta = [[0,B_16],[B_25,B_26]] via the same segregated solves.
  !> S_PBP = Atilde_22 - Atilde_21 Atilde_11^-1 B_12 is the element-assembled
  !> parabolized momentum operator (ksp_S_PBP); all inner solves are MUMPS.
  !>
  !> Precondition: physics_pc_apply (the dispatcher) has already populated
  !>   g_ctx%work_1 = M_j^{-1} x_j   (temp_j)
  !>   g_ctx%work_2 = M_w^{-1} x_w   (temp_w)
  !--------------------------------------------------------------------
  subroutine apply_block_predictor_corrector(x_psi, x_u, x_rho, x_T, &
                                             y_psi, y_u, y_rho, y_T, ierr)
    use phys_module, only: physics_pc_multi_step_symmetric
    Vec, intent(in)    :: x_psi, x_u, x_rho, x_T
    Vec, intent(inout) :: y_psi, y_u, y_rho, y_T
    PetscErrorCode, intent(out) :: ierr

    ! Inputs from dispatcher: work_1 = temp_j = M_j^-1 x_j, work_2 = temp_w = M_w^-1 x_w.
    ! work_1/work_2 are preserved until the last fold (the B_63*temp_j term in the T row).

    ! --- Folded Alfven residuals: work_3 = r_psi, work_4 = r_u (scratch work_5) ---
    call fold_alfven_residual(x_psi, x_u, g_ctx%work_3, g_ctx%work_4, g_ctx%work_5, ierr)

    ! --- Predictor: velocity from the parabolized momentum operator S_PBP ---
    !   t_psi = Atilde_11^-1 * r_psi
    call KSPSolve(g_ctx%ksp_psi, g_ctx%work_3, g_ctx%work_5, ierr)        ! work_5 = t_psi
    !   rhs_u = r_u - Atilde_21 * t_psi   (use y_psi as scratch for Atilde_21*t_psi)
    call MatMult(g_ctx%Atilde_21, g_ctx%work_5, y_psi, ierr)             ! y_psi = Atilde_21 t_psi
    call VecWAXPY(y_u, -1.0d0, y_psi, g_ctx%work_4, ierr)                ! y_u = r_u - Atilde_21 t_psi
    !   y_u = S_PBP^-1 rhs_u   (solve out-of-place into work_5, then copy)
    call KSPSolve(g_ctx%ksp_S_PBP, y_u, g_ctx%work_5, ierr)             ! work_5 = velocity
    call VecCopy(g_ctx%work_5, y_u, ierr)                               ! y_u = velocity

    ! --- Corrector: flux back-substitution  y_psi = Atilde_11^-1 (r_psi - B_12 y_u) ---
    call MatMult(g_ctx%B_12, y_u, g_ctx%work_4, ierr)                   ! work_4 = B_12 y_u
    call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_4, g_ctx%work_3, ierr) ! work_5 = r_psi - B_12 y_u
    call KSPSolve(g_ctx%ksp_psi, g_ctx%work_5, y_psi, ierr)             ! y_psi = corrected flux

    ! --- Transport rho: y_rho = B_55^-1 (x_rho - B_51 y_psi - B_52 y_u) ---
    call MatMult(g_ctx%B_51, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_rho, ierr)      ! work_4 = x_rho - B_51 y_psi
    call MatMult(g_ctx%B_52, y_u, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_rho, g_ctx%work_4, y_rho, ierr)

    ! --- Transport T: r_T = x_T - B_63 temp_j; y_T = B_66^-1 (r_T - Atilde_61 y_psi - B_62 y_u) ---
    call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_T, ierr)        ! work_4 = r_T
    call MatMult(g_ctx%Atilde_61, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_62, y_u, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_T, g_ctx%work_4, y_T, ierr)

    ! --- Optional backward corrector for the dropped weak feedback K_alpha,beta = [[0,B16],[B25,B26]] ---
    ! Applied via the SAME segregated (Atilde_11, S_PBP) solves, NOT a joint 2x2.
    ! Here temp_j/temp_w (work_1/work_2) are no longer needed, so they are reused as scratch.
    if (physics_pc_multi_step_symmetric) then
      ! c_psi = B_16 y_T   (work_3) ;  c_u = B_25 y_rho + B_26 y_T  (work_4)
      call MatMult(g_ctx%B_16, y_T, g_ctx%work_3, ierr)
      call MatMult(g_ctx%B_25, y_rho, g_ctx%work_4, ierr)
      call MatMult(g_ctx%B_26, y_T,   g_ctx%work_1, ierr)
      call VecAXPY(g_ctx%work_4, 1.0d0, g_ctx%work_1, ierr)            ! work_4 = c_u

      ! d_u = S_PBP^-1 ( c_u - Atilde_21 Atilde_11^-1 c_psi )
      call KSPSolve(g_ctx%ksp_psi, g_ctx%work_3, g_ctx%work_5, ierr)   ! work_5 = Atilde_11^-1 c_psi
      call MatMult(g_ctx%Atilde_21, g_ctx%work_5, g_ctx%work_1, ierr)  ! work_1 = Atilde_21 (...)
      call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_1, ierr)           ! work_4 = c_u - Atilde_21 (...)
      call KSPSolve(g_ctx%ksp_S_PBP, g_ctx%work_4, g_ctx%work_1, ierr) ! work_1 = d_u
      call VecAXPY(y_u, -1.0d0, g_ctx%work_1, ierr)                    ! y_u -= d_u

      ! d_psi = Atilde_11^-1 ( c_psi - B_12 d_u )
      call MatMult(g_ctx%B_12, g_ctx%work_1, g_ctx%work_2, ierr)       ! work_2 = B_12 d_u  (work_1 = d_u)
      call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_2, g_ctx%work_3, ierr) ! work_4 = c_psi - B_12 d_u
      call KSPSolve(g_ctx%ksp_psi, g_ctx%work_4, g_ctx%work_2, ierr)   ! work_2 = d_psi
      call VecAXPY(y_psi, -1.0d0, g_ctx%work_2, ierr)                  ! y_psi -= d_psi
    endif

    ierr = 0
  end subroutine apply_block_predictor_corrector

  !--------------------------------------------------------------------
  !> Apply path: wave-Schur block-LDU predictor-corrector (new 4-variable scheme).
  !>
  !> Implements the single block-LDU sweep of Sec. 5.4, eq. (algorithm) in
  !>   docs/superpowers/research/2026-06-22-toroidal-wave-schur-progress.
  !> Unlike apply_block_predictor_corrector (the older segregated-Schur variant),
  !> the predictor carries the full (psi*, rho*, T*) into the momentum wave solve,
  !> and the corrector applies the single weak upper coupling A_16 against T*. The
  !> pressure feedback (channel B) and the psi<-T coupling are thus captured in ONE
  !> forward sweep, so no separate symmetric backward corrector is needed.
  !>
  !>   0. Fold (exact):  r~_psi = x_psi - B_13 temp_j
  !>                     r~_u   = x_u   - B_23 temp_j - B_24 temp_w
  !>                     r~_T   = x_T   - B_63 temp_j
  !>   1. Predictor:     psi* = Atilde_11^-1 r~_psi
  !>                     rho* = B_55^-1 (x_rho - B_51 psi*)
  !>                     T*   = B_66^-1 (r~_T  - Atilde_61 psi*)
  !>   2. Wave solve:    u    = S_PBP^-1 (r~_u - Atilde_21 psi* - B_25 rho* - B_26 T*)
  !>   3. Corrector:     psi  = psi* - Atilde_11^-1 (B_12 u + B_16 T*)
  !>                     rho  = rho* - B_55^-1 (B_52 u)
  !>                     T    = T*   - B_66^-1 (B_62 u)
  !>   4. Constraints (delta_j, delta_w) are restored by the dispatcher.
  !>
  !> NOTE on the fold: eq. (algorithm) abbreviates r~_u = r_2 - A_24 M_w^-1 r_w, but
  !> the reduced row-2 operator Atilde_21 = A_21 - A_23 M_j^-1 A_31 also carries the
  !> A_23 path, so the EXACT residual fold must include -A_23 M_j^-1 r_j as well. We
  !> therefore reuse fold_alfven_residual, which folds both temp_j and temp_w.
  !>
  !> The predictor freezes the lower-triangular blocks B_51, Atilde_61 against the
  !> PREDICTOR flux psi* (not the corrected psi), exactly as the LDU sweep prescribes;
  !> the rho/T correctors then build on the predictor values rho*/T*.
  !>
  !> All inner solves (Atilde_11=ksp_psi, B_55=ksp_rho, B_66=ksp_T, S_PBP=ksp_S_PBP)
  !> are the MUMPS KSPs already set up for the multi_step path.
  !>
  !> Work-vector roles:
  !>   work_1 = temp_j, work_2 = temp_w   (inputs; read by the folds, then free)
  !>   work_3 = scratch
  !>   work_5 = scratch; in step 2 it holds the wave RHS and, on the commutator
  !>            arm, must SURVIVE the S_PBP solve -- the ZBIG Jacobi add-back
  !>            reads it after y_u has been formed
  !>   work_4 = r~_u (held from the fold until the wave RHS is formed)
  !>   y_psi  = psi*  (predictor flux, corrected in place in step 3)
  !>   tmp_rho = rho*, tmp_T = T*  (predictor pressure, corrected into y_rho / y_T)
  !>   y_u    = u
  !>
  !> Precondition: physics_pc_apply (the dispatcher) has already populated
  !>   g_ctx%work_1 = M_j^{-1} x_j   (temp_j)
  !>   g_ctx%work_2 = M_w^{-1} x_w   (temp_w)
  !--------------------------------------------------------------------
  subroutine apply_wave_schur_predictor_corrector(x_psi, x_u, x_rho, x_T, &
                                                  y_psi, y_u, y_rho, y_T, ierr)
    Vec, intent(in)    :: x_psi, x_u, x_rho, x_T
    Vec, intent(inout) :: y_psi, y_u, y_rho, y_T
    PetscErrorCode, intent(out) :: ierr

    ! --- Step 0: exact j,w fold of the (psi,u) rows -> work_3 = r~_psi, work_4 = r~_u ---
    call fold_alfven_residual(x_psi, x_u, g_ctx%work_3, g_ctx%work_4, g_ctx%work_5, ierr)

    ! --- Step 0: fold of the T row -> tmp_T = r~_T = x_T - B_63 temp_j ---
    call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_5, ierr)
    call VecWAXPY(g_ctx%tmp_T, -1.0d0, g_ctx%work_5, x_T, ierr)
    ! temp_j/temp_w (work_1/work_2) are no longer needed below.

    ! --- Step 1: predictor flux  psi* = Atilde_11^-1 r~_psi   (work_3 = r~_psi) ---
    call KSPSolve(g_ctx%ksp_psi, g_ctx%work_3, y_psi, ierr)              ! y_psi = psi*

    ! --- Step 1: predictor density  rho* = B_55^-1 (x_rho - B_51 psi*) ---
    call MatMult(g_ctx%B_51, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, x_rho, ierr)       ! work_5 = x_rho - B_51 psi*
    call KSPSolve(g_ctx%ksp_rho, g_ctx%work_5, g_ctx%tmp_rho, ierr)      ! tmp_rho = rho*

    ! --- Step 1: predictor temperature  T* = B_66^-1 (r~_T - Atilde_61 psi*) ---
    call MatMult(g_ctx%Atilde_61, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, g_ctx%tmp_T, ierr) ! work_5 = r~_T - Atilde_61 psi*
    call KSPSolve(g_ctx%ksp_T, g_ctx%work_5, g_ctx%tmp_T, ierr)          ! tmp_T = T*

    ! --- Step 2: wave solve  u = S_PBP^-1 (r~_u - Atilde_21 psi* - B_25 rho* - B_26 T*) ---
    call VecCopy(g_ctx%work_4, g_ctx%work_5, ierr)                       ! work_5 = r~_u
    call MatMult(g_ctx%Atilde_21, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_25, g_ctx%tmp_rho, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_26, g_ctx%tmp_T, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    if (.not. g_ctx%schur_comm_active) then
      call KSPSolve(g_ctx%ksp_S_PBP, g_ctx%work_5, y_u, ierr)           ! y_u = u
    else
      ! Stage 6.3, commutator arm: S_PBP holds only S_ass, so the wave solve
      ! is Shat^-1 = Q_u^-1 A_uM S_ass^-1 -- the Riesz map and the commutator
      ! factor that S_ass was multiplied by when it was assembled. Same
      ! composition as the Stage 4.6 harness (schur_global_pc_apply).
      ! work_5 must survive to the boundary add-back below, so the masked
      ! right-hand side goes into work_3 (dead from the line above).
      if (g_ctx%schur_comm_mask) then
        call VecPointwiseMult(g_ctx%work_3, g_ctx%work_5, g_ctx%u_mask, ierr)
      else
        call VecCopy(g_ctx%work_5, g_ctx%work_3, ierr)
      endif
      call KSPSolve(g_ctx%ksp_S_PBP, g_ctx%work_3, y_u, ierr)
      if (g_ctx%schur_comm_mask) call VecPointwiseMult(y_u, y_u, g_ctx%u_mask, ierr)
      call MatMult(g_ctx%A_uM_prod, y_u,          g_ctx%work_3, ierr)
      call MatMult(g_ctx%Qi_u_prod, g_ctx%work_3, y_u,          ierr)   ! y_u = u
      if (g_ctx%schur_comm_mask) then
        ! Penalty rows: Jacobi on the ZBIG diagonal. The interior operator
        ! carries the identity there, which is ~1e11 off.
        call VecPointwiseMult(y_u, y_u, g_ctx%u_mask, ierr)
        call VecPointwiseMult(g_ctx%work_3, g_ctx%work_5, g_ctx%u_bnd, ierr)
        call VecAXPY(y_u, 1.0d0, g_ctx%work_3, ierr)
      endif
    endif

    ! --- Step 3: flux corrector  psi = psi* - Atilde_11^-1 (B_12 u + B_16 T*) ---
    call MatMult(g_ctx%B_12, y_u, g_ctx%work_3, ierr)                   ! work_3 = B_12 u
    call MatMult(g_ctx%B_16, g_ctx%tmp_T, g_ctx%work_5, ierr)           ! work_5 = B_16 T*
    call VecAXPY(g_ctx%work_3, 1.0d0, g_ctx%work_5, ierr)               ! work_3 = B_12 u + B_16 T*
    call KSPSolve(g_ctx%ksp_psi, g_ctx%work_3, g_ctx%work_5, ierr)      ! work_5 = correction
    call VecAXPY(y_psi, -1.0d0, g_ctx%work_5, ierr)                     ! y_psi = psi

    ! --- Step 3: density corrector  rho = rho* - B_55^-1 (B_52 u) ---
    call MatMult(g_ctx%B_52, y_u, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_rho, g_ctx%work_3, g_ctx%work_5, ierr)      ! work_5 = B_55^-1 B_52 u
    call VecWAXPY(y_rho, -1.0d0, g_ctx%work_5, g_ctx%tmp_rho, ierr)     ! y_rho = rho

    ! --- Step 3: temperature corrector  T = T* - B_66^-1 (B_62 u) ---
    call MatMult(g_ctx%B_62, y_u, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_T, g_ctx%work_3, g_ctx%work_5, ierr)        ! work_5 = B_66^-1 B_62 u
    call VecWAXPY(y_T, -1.0d0, g_ctx%work_5, g_ctx%tmp_T, ierr)         ! y_T = T

    ierr = 0
  end subroutine apply_wave_schur_predictor_corrector

  !--------------------------------------------------------------------
  !> Apply path: MIXED-PAIR block-LDU sweep (Workstream B, "SFM").
  !>
  !> Neither constraint variable is eliminated. Where the substituted schemes
  !> fold j and omega away with exact mass solves -- raising the differential
  !> order of the psi and u diagonals to fourth in the process -- this sweep
  !> keeps both explicit and solves two mixed 2x2 pairs:
  !>
  !>   pair_psi = [[B_11, B_13], [B_31, B_33]]
  !>   pair_w   = [[S_uu^SFM, B_24], [B_42, B_44]]
  !>
  !> Sequence (no mass pre-solves, no residual folds, no back-substitution):
  !>
  !>   1. Predictor psi-pair: pair_psi (psi*, j*) = (x_psi, x_j)      [1 packed solve]
  !>   1. Predictor rho:      rho* = B_55^-1 (x_rho - B_51 psi*)
  !>   1. Predictor T:        T*   = B_66^-1 (x_T - B_61 psi* - B_63 j*)
  !>   2. Wave solve:         RHS_u  = x_u - B_21 psi* - B_23 j* - B_25 rho* - B_26 T*
  !>                          RHS_om = x_w
  !>                          pair_w (u, omega) = (RHS_u, RHS_om)     [1 packed solve]
  !>   3. Corrector psi-pair: pair_psi (dpsi, dj) = (B_12 u + B_16 T*, 0)
  !>                          y_psi = psi* - dpsi ,  y_j = j* - dj
  !>   3. Corrector rho:      y_rho = rho* - B_55^-1 (B_52 u)
  !>   3. Corrector T:        y_T   = T*   - B_66^-1 (B_62 u)
  !>
  !> Three zero structures carry the whole scheme, and each is commented at its
  !> site below because getting any of them wrong is silent:
  !>
  !>   - L's omega-row is zero  -> RHS_om is the RAW input residual x_w, with no
  !>     fold and no correction. In particular there is NO -B_24 M_w^-1 x_w term
  !>     on RHS_u: that coupling lives in the (1,2) entry of pair_w, and folding
  !>     it here as well would double-count it.
  !>   - U's j-row is zero      -> the corrector pair RHS has a zero j-component.
  !>     (The PREDICTOR pair RHS j-component is x_j, not zero.)
  !>   - U's omega-column is zero -> step 3 never touches y_w; omega is final
  !>     straight out of the wave solve.
  !>
  !> Jacobian rows 3 and 4 are [B_31,0,B_33,0,0,0] and [0,B_42,0,B_44,0,0], i.e.
  !> row 2 of each pair IS its constraint equation verbatim. So the pairs
  !> reproduce exactly what a mass pre-solve plus back-substitution would have
  !> computed, and the dispatcher skips both on this arm.
  !>
  !> PRECONDITION: none. Unlike every other arm, this one neither reads nor
  !> requires work_1/work_2 (temp_j/temp_w) -- it never folds.
  !>
  !> Work-vector roles:
  !>   work_3, work_4, work_5 = 1-variable scratch
  !>   tmp_rho = rho*, tmp_T = T*   (predictor values, needed until step 3)
  !>   rhs_PJ/sol_PJ, rhs_W/sol_W   = the packed pair vectors
  !>   y_psi = psi* then psi ; y_j = j* then j ; y_u = u ; y_w = omega (final)
  !>
  !> All six output components are written exactly once, so this routine does not
  !> depend on y being zeroed on entry.
  !--------------------------------------------------------------------
  subroutine apply_wave_schur_mixed_pairs(x_psi, x_u, x_j, x_w, x_rho, x_T, &
                                          y_psi, y_u, y_j, y_w, y_rho, y_T, ierr)
    Vec, intent(in)    :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec, intent(inout) :: y_psi, y_u, y_j, y_w, y_rho, y_T
    PetscErrorCode, intent(out) :: ierr

    ! --- Step 1: predictor psi-pair.  pair_psi (psi*, j*) = (x_psi, x_j) ---
    ! The j-component of the RHS is x_j, NOT zero: this is the constraint
    ! equation's own residual, and it is what makes j* equal the mass
    ! back-substitution M_j^-1 (x_j - B_31 psi*).
    call pack_2v(x_psi, x_j, g_ctx%rhs_PJ, ierr)
    call KSPSolve(g_ctx%ksp_pair_psi, g_ctx%rhs_PJ, g_ctx%sol_PJ, ierr)
    call unpack_2v(g_ctx%sol_PJ, y_psi, y_j, ierr)          ! y_psi = psi*, y_j = j*

    ! --- Step 1: predictor density  rho* = B_55^-1 (x_rho - B_51 psi*) ---
    call MatMult(g_ctx%B_51, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_rho, ierr)
    call KSPSolve(g_ctx%ksp_rho, g_ctx%work_4, g_ctx%tmp_rho, ierr)   ! tmp_rho = rho*

    ! --- Step 1: predictor temperature  T* = B_66^-1 (x_T - B_61 psi* - B_63 j*) ---
    ! B_61 and B_63 act against the EXPLICIT predictor pair (psi*, j*). No
    ! j-folded lower-triangular block is needed or wanted here.
    call MatMult(g_ctx%B_61, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_T, ierr)
    call MatMult(g_ctx%B_63, y_j, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_T, g_ctx%work_4, g_ctx%tmp_T, ierr)       ! tmp_T = T*

    ! --- Step 2: the ONE packed wave solve ---
    !   RHS_u = x_u - B_21 psi* - B_23 j* - B_25 rho* - B_26 T*
    ! B_21 is the RAW lower coupling and B_23 j* carries the Lorentz path
    ! explicitly. There is deliberately NO -B_24 M_w^-1 x_w term: the u-omega
    ! coupling is the (1,2) entry of pair_w (see the header).
    call VecCopy(x_u, g_ctx%work_5, ierr)
    call MatMult(g_ctx%B_21, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_23, y_j, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_25, g_ctx%tmp_rho, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_26, g_ctx%tmp_T, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    !   RHS_om = x_w VERBATIM. The omega-row of the lower coupling L is
    !   identically zero, so the omega component of (r_w - L y*) is the raw
    !   input residual -- no fold, no correction term.
    call pack_2v(g_ctx%work_5, x_w, g_ctx%rhs_W, ierr)
    call KSPSolve(g_ctx%ksp_pair_w, g_ctx%rhs_W, g_ctx%sol_W, ierr)
    call unpack_2v(g_ctx%sol_W, y_u, y_w, ierr)      ! BOTH final; y_w is DONE

    ! --- Step 3: corrector psi-pair.  pair_psi (dpsi, dj) = (B_12 u + B_16 T*, 0) ---
    ! Here the j-component of the RHS IS zero, because the j-row of the upper
    ! coupling U is identically zero. (Contrast the predictor above.)
    call MatMult(g_ctx%B_12, y_u, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_16, g_ctx%tmp_T, g_ctx%work_4, ierr)
    call VecAXPY(g_ctx%work_3, 1.0d0, g_ctx%work_4, ierr)      ! B_12 u + B_16 T*
    call VecZeroEntries(g_ctx%work_4, ierr)
    call pack_2v(g_ctx%work_3, g_ctx%work_4, g_ctx%rhs_PJ, ierr)
    call KSPSolve(g_ctx%ksp_pair_psi, g_ctx%rhs_PJ, g_ctx%sol_PJ, ierr)
    call unpack_2v(g_ctx%sol_PJ, g_ctx%work_3, g_ctx%work_4, ierr)  ! dpsi, dj
    call VecAXPY(y_psi, -1.0d0, g_ctx%work_3, ierr)            ! y_psi = psi* - dpsi
    call VecAXPY(y_j,   -1.0d0, g_ctx%work_4, ierr)            ! y_j   = j*   - dj
                                                               ! (replaces the j
                                                               !  back-substitution)

    ! --- Step 3: rho / T correctors. Only u enters -- U's omega COLUMN is zero. ---
    call MatMult(g_ctx%B_52, y_u, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_rho, g_ctx%work_3, g_ctx%work_5, ierr)
    call VecWAXPY(y_rho, -1.0d0, g_ctx%work_5, g_ctx%tmp_rho, ierr)

    call MatMult(g_ctx%B_62, y_u, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_T, g_ctx%work_3, g_ctx%work_5, ierr)
    call VecWAXPY(y_T, -1.0d0, g_ctx%work_5, g_ctx%tmp_T, ierr)

    ierr = 0
  end subroutine apply_wave_schur_mixed_pairs

  !--------------------------------------------------------------------
  !> PCSHELL apply callback: compute y = P^{-1} x.
  !!
  !! Algorithm:
  !!   1. Extract variable sub-vectors from x and y
  !!   2. Compute Mass-scaled constraint residuals
  !!   3. Schur-correct the RHS for the 4x4 reduced system
  !!   4. Solve diagonal blocks of the reduced system
  !!   5. Back-substitute for j and w
  !!   6. Restore sub-vectors
  !--------------------------------------------------------------------
  subroutine physics_pc_apply(pc_obj, x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    use phys_module, only: physics_pc_monolithic, physics_pc_multi_step, &
                           physics_pc_sub_blocks, physics_pc_sub_blocks_mode, &
                           physics_pc_multi_step_symmetric, physics_pc_wave_schur

    PC :: pc_obj
    Vec :: x, y
    PetscErrorCode :: ierr

    ! Sub-vectors (views into x and y)
    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T

    if (.not. g_ctx%reduced_ready) then
      ierr = 1
      return
    endif

    ! --- Step 1: Extract variable sub-vectors ---
    call VecGetSubVector(x, g_ctx%is_var(var_psi), x_psi, ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_u),   x_u,   ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_zj),  x_j,   ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_w),   x_w,   ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_rho), x_rho, ierr)
    call VecGetSubVector(x, g_ctx%is_var(var_T),   x_T,   ierr)

    call VecGetSubVector(y, g_ctx%is_var(var_psi), y_psi, ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_u),   y_u,   ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_zj),  y_j,   ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_w),   y_w,   ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_rho), y_rho, ierr)
    call VecGetSubVector(y, g_ctx%is_var(var_T),   y_T,   ierr)

    ! --- Step 2: Apply inverse of elliptic constraint mass matrices ---
    ! The mixed-pair arm keeps j and omega explicit, so it needs neither of
    ! these: both constraint equations are solved inside their pair.
    !
    ! The zeroing is deliberate belt-and-braces rather than a bare skip. If any
    ! fold path were ever reached on this arm, a ZERO makes that fold a no-op,
    ! which is the correct behaviour here -- whereas values left stale from the
    ! previous apply would instead produce plausible-looking wrong numbers. Two
    ! VecZeroEntries in exchange for turning the nastiest silent-failure mode on
    ! this arm into a harmless one, and it still saves the two solves.
    if (.not. g_ctx%schur_mixed_active) then
      ! work_1 = A_33^{-1} * x_j  (temp_j)
      call KSPSolve(g_ctx%ksp_Mj, x_j, g_ctx%work_1, ierr)
      ! work_2 = A_44^{-1} * x_w  (temp_w)
      call KSPSolve(g_ctx%ksp_Mw, x_w, g_ctx%work_2, ierr)
    else
      call VecZeroEntries(g_ctx%work_1, ierr)
      call VecZeroEntries(g_ctx%work_2, ierr)
    endif

    ! --- Step 3: Schur-correct the RHS and solve ---
    if (physics_pc_multi_step) then
      if (g_ctx%schur_mixed_active) then
        ! Workstream B: mixed-pair sweep. j and omega are outputs of the two
        ! packed pair solves, so this routine takes and returns all six
        ! components and the dispatcher's folds/back-substitutions are skipped.
        call apply_wave_schur_mixed_pairs(x_psi, x_u, x_j, x_w, x_rho, x_T, &
                                         y_psi, y_u, y_j, y_w, y_rho, y_T, ierr)
      else if (physics_pc_wave_schur) then
        ! New wave-Schur block-LDU sweep (Sec. 5.4): full predictor + A_16 corrector.
        call apply_wave_schur_predictor_corrector(x_psi, x_u, x_rho, x_T, &
                                                  y_psi, y_u, y_rho, y_T, ierr)
      else
        ! Older segregated-Schur predictor-corrector (optionally symmetric).
        call apply_block_predictor_corrector(x_psi, x_u, x_rho, x_T, &
                                             y_psi, y_u, y_rho, y_T, ierr)
      endif

    else if (physics_pc_monolithic) then
      call apply_monolithic_4x4(x_psi, x_u, x_rho, x_T, &
                                y_psi, y_u, y_rho, y_T, ierr)

    else if (physics_pc_sub_blocks) then
      ! ===== Sub-blocks PC: (psi,u) Alfven + (rho,T) transport =====
      select case (physics_pc_sub_blocks_mode)
      case (1)
        call apply_block_jacobi(x_psi, x_u, x_rho, x_T, &
                                y_psi, y_u, y_rho, y_T, ierr)
      case (2)
        call apply_block_gs_forward(x_psi, x_u, x_rho, x_T, &
                                    y_psi, y_u, y_rho, y_T, ierr)
      case (3)
        call apply_block_gs_symmetric(x_psi, x_u, x_rho, x_T, &
                                      y_psi, y_u, y_rho, y_T, ierr)
      case default
        write(*,'(A,I0)') &
          "[Physics PC] ERROR: invalid physics_pc_sub_blocks_mode = ", physics_pc_sub_blocks_mode
        ierr = 1
        return
      end select

    else
      write(*,'(A)') "[Physics PC] ERROR: no apply mode selected"
      ierr = 1
      return
    endif

    ! --- Step 4: Back-substitute for j and w ---
    ! Skipped on the mixed-pair arm: y_j and y_w are already final, having come
    ! out of the two pair solves. Row 2 of each pair IS the constraint equation
    ! being back-substituted here, so the values would be identical -- but y_j is
    ! a VecGetSubVector view that this block would overwrite, and running it
    ! would cost two needless solves.
    if (.not. g_ctx%schur_mixed_active) then
      ! y_j = A_33^{-1} * (x_j - B_31 * y_psi)
      call MatMult(g_ctx%B_31, y_psi, g_ctx%work_3, ierr)
      call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_j, ierr)
      call KSPSolve(g_ctx%ksp_Mj, g_ctx%work_4, y_j, ierr)

      ! y_w = A_44^{-1} * (x_w - B_42 * y_u)
      call MatMult(g_ctx%B_42, y_u, g_ctx%work_3, ierr)
      call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_w, ierr)
      call KSPSolve(g_ctx%ksp_Mw, g_ctx%work_4, y_w, ierr)
    endif

    ! --- Step 5: Restore sub-vectors ---
    call VecRestoreSubVector(x, g_ctx%is_var(var_psi), x_psi, ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_u),   x_u,   ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_zj),  x_j,   ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_w),   x_w,   ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_rho), x_rho, ierr)
    call VecRestoreSubVector(x, g_ctx%is_var(var_T),   x_T,   ierr)

    call VecRestoreSubVector(y, g_ctx%is_var(var_psi), y_psi, ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_u),   y_u,   ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_zj),  y_j,   ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_w),   y_w,   ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_rho), y_rho, ierr)
    call VecRestoreSubVector(y, g_ctx%is_var(var_T),   y_T,   ierr)

    ierr = 0
  end subroutine physics_pc_apply

#endif
end module mod_petsc_pc_physics_apply
