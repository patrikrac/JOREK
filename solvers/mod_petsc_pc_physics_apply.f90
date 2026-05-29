module mod_petsc_pc_physics_apply
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  implicit none
  private

  public :: physics_pc_apply

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

    call VecGetArrayReadF90(x1, a_x1, ierr)
    call VecGetArrayReadF90(x2, a_x2, ierr)
    call VecGetArrayF90    (y_packed, a_y, ierr)

    a_y(1     : n1     ) = a_x1(1:n1)
    a_y(n1+1  : n1+n2  ) = a_x2(1:n2)

    call VecRestoreArrayReadF90(x1, a_x1, ierr)
    call VecRestoreArrayReadF90(x2, a_x2, ierr)
    call VecRestoreArrayF90    (y_packed, a_y, ierr)
  end subroutine pack_2v

  !> Inverse of pack_2v: unpack a 2v-sized packed vec into two 1v vecs.
  subroutine unpack_2v(x_packed, y1, y2, ierr)
    Vec            :: x_packed, y1, y2
    PetscErrorCode :: ierr

    PetscScalar, pointer :: a_x(:), a_y1(:), a_y2(:)
    PetscInt :: n1, n2

    call VecGetLocalSize(y1, n1, ierr)
    call VecGetLocalSize(y2, n2, ierr)

    call VecGetArrayReadF90(x_packed, a_x, ierr)
    call VecGetArrayF90    (y1, a_y1, ierr)
    call VecGetArrayF90    (y2, a_y2, ierr)

    a_y1(1:n1) = a_x(1     : n1     )
    a_y2(1:n2) = a_x(n1+1  : n1+n2  )

    call VecRestoreArrayReadF90(x_packed, a_x, ierr)
    call VecRestoreArrayF90    (y1, a_y1, ierr)
    call VecRestoreArrayF90    (y2, a_y2, ierr)
  end subroutine unpack_2v

  !> Block-Jacobi apply: y_A = K_A^-1 x_A, y_B = K_B^-1 x_B (parallel, no coupling).
  !!
  !! Mode 1 of the sub-blocks PC. Independent solves on the (psi,u) Alfven block
  !! and the (rho,T) transport block — cross-block coupling is ignored.
  subroutine apply_block_jacobi(x_psi, x_u, x_rho, x_T, &
                                y_psi, y_u, y_rho, y_T, ierr)
    Vec            :: x_psi, x_u, x_rho, x_T
    Vec            :: y_psi, y_u, y_rho, y_T
    PetscErrorCode :: ierr

    ! Pack inputs into 2v rhs vectors
    call pack_2v(x_psi, x_u,   g_ctx%rhs_A, ierr)
    call pack_2v(x_rho, x_T,   g_ctx%rhs_B, ierr)

    ! Solve each super-block (PREONLY + LU + MUMPS)
    call KSPSolve(g_ctx%ksp_block_A, g_ctx%rhs_A, g_ctx%sol_A, ierr)
    call KSPSolve(g_ctx%ksp_block_B, g_ctx%rhs_B, g_ctx%sol_B, ierr)

    ! Unpack into outputs
    call unpack_2v(g_ctx%sol_A, y_psi, y_u,   ierr)
    call unpack_2v(g_ctx%sol_B, y_rho, y_T,   ierr)
  end subroutine apply_block_jacobi

  !> Block-GS-forward apply: solve K_A, propagate to (rho,T), solve K_B.
  !!
  !! Mode 2 of the sub-blocks PC. Forward Gauss-Seidel sweep:
  !!   1. y_A = K_A^-1 x_A
  !!   2. r_B = x_B - C_BA * y_A      where C_BA = [B_51 B_52; B_61 B_62]
  !!   3. y_B = K_B^-1 r_B
  !!
  !! Captures the (psi,u) -> (rho,T) coupling via the C_BA submatrices.
  subroutine apply_block_gs_forward(x_psi, x_u, x_rho, x_T, &
                                    y_psi, y_u, y_rho, y_T, ierr)
    Vec            :: x_psi, x_u, x_rho, x_T
    Vec            :: y_psi, y_u, y_rho, y_T
    PetscErrorCode :: ierr

    ! 1. Forward sweep on Alfven super-block
    call pack_2v(x_psi, x_u, g_ctx%rhs_A, ierr)
    call KSPSolve(g_ctx%ksp_block_A, g_ctx%rhs_A, g_ctx%sol_A, ierr)
    call unpack_2v(g_ctx%sol_A, y_psi, y_u, ierr)

    ! 2. Residual on transport super-block:
    !    r_rho = x_rho - B_51 * y_psi - B_52 * y_u
    !    r_T   = x_T   - B_61 * y_psi - B_62 * y_u
    call VecCopy(x_rho, g_ctx%tmp_rho, ierr)
    call MatMult(g_ctx%B_51, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_rho, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_52, y_u,   g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_rho, -1.0d0, g_ctx%work_3, ierr)

    call VecCopy(x_T, g_ctx%tmp_T, ierr)
    call MatMult(g_ctx%B_61, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_T, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_62, y_u,   g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%tmp_T, -1.0d0, g_ctx%work_3, ierr)

    ! 3. Pack residuals and solve K_B
    call pack_2v(g_ctx%tmp_rho, g_ctx%tmp_T, g_ctx%rhs_B, ierr)
    call KSPSolve(g_ctx%ksp_block_B, g_ctx%rhs_B, g_ctx%sol_B, ierr)
    call unpack_2v(g_ctx%sol_B, y_rho, y_T, ierr)
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
    ! r_psi = x_psi - B_16 * y_T          (no B_15: psi-rho coupling is zero in model199)
    call VecCopy(x_psi, g_ctx%work_1, ierr)
    call MatMult(g_ctx%B_16, y_T, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_1, -1.0d0, g_ctx%work_3, ierr)

    ! r_u = x_u - B_25 * y_rho - B_26 * y_T
    call VecCopy(x_u, g_ctx%work_2, ierr)
    call MatMult(g_ctx%B_25, y_rho, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_2, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_26, y_T,   g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_2, -1.0d0, g_ctx%work_3, ierr)

    ! Pack residuals and re-solve K_A
    call pack_2v(g_ctx%work_1, g_ctx%work_2, g_ctx%rhs_A, ierr)
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
  !> Apply path: three-step predictor-corrector.
  !> Stages: hydro predictor (M_hydro joint 3x3 solve for u,rho,T) ->
  !>         magnetic predictor (ksp_psi) ->
  !>         Alfven corrector (ksp_S_PBP, with Atilde_21 coupling) ->
  !>         transport correction (ksp_rho, ksp_T).
  !>
  !> Precondition: physics_pc_apply (the dispatcher) has already populated
  !>   g_ctx%work_1 = M_j^{-1} x_j   (temp_j)
  !>   g_ctx%work_2 = M_w^{-1} x_w   (temp_w)
  !--------------------------------------------------------------------
  subroutine apply_block_predictor_corrector(x_psi, x_u, x_rho, x_T, &
                                             y_psi, y_u, y_rho, y_T, ierr)
    Vec, intent(in)    :: x_psi, x_u, x_rho, x_T
    Vec, intent(inout) :: y_psi, y_u, y_rho, y_T
    PetscErrorCode, intent(out) :: ierr

    Vec :: rhs_u, rhs_rho, rhs_T
    Vec :: sol_u, sol_T

    ! --- Step 1: Hydro predictor — joint 3x3 M_hydro solve (u,rho,T) ---
    ! b_u  = x_u - B_23*temp_j - B_24*temp_w
    call MatMult(g_ctx%B_23, g_ctx%work_1, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_24, g_ctx%work_2, g_ctx%work_4, ierr)
    call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, x_u, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_4, ierr)
    call VecGetSubVector(g_ctx%work_rhs_3v, g_ctx%is_hydro(1), rhs_u, ierr)
    call VecCopy(g_ctx%work_5, rhs_u, ierr)
    call VecRestoreSubVector(g_ctx%work_rhs_3v, g_ctx%is_hydro(1), rhs_u, ierr)

    ! b_rho = x_rho
    call VecGetSubVector(g_ctx%work_rhs_3v, g_ctx%is_hydro(2), rhs_rho, ierr)
    call VecCopy(x_rho, rhs_rho, ierr)
    call VecRestoreSubVector(g_ctx%work_rhs_3v, g_ctx%is_hydro(2), rhs_rho, ierr)

    ! b_T = x_T - B_63*temp_j
    call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_T, ierr)
    call VecGetSubVector(g_ctx%work_rhs_3v, g_ctx%is_hydro(3), rhs_T, ierr)
    call VecCopy(g_ctx%work_4, rhs_T, ierr)
    call VecRestoreSubVector(g_ctx%work_rhs_3v, g_ctx%is_hydro(3), rhs_T, ierr)

    call KSPSolve(g_ctx%ksp_hydro, g_ctx%work_rhs_3v, g_ctx%hydro_predictor_3v, ierr)

    ! Extract predicted velocity and temperature (rho_h unused: rho comes
    ! from the Step 4 transport correction).
    ! work_5 := u_pred, work_4 := T_pred (kept across Steps 2-3 as noted).
    call VecGetSubVector(g_ctx%hydro_predictor_3v, g_ctx%is_hydro(1), sol_u, ierr)
    call VecCopy(sol_u, g_ctx%work_5, ierr)
    call VecRestoreSubVector(g_ctx%hydro_predictor_3v, g_ctx%is_hydro(1), sol_u, ierr)
    call VecGetSubVector(g_ctx%hydro_predictor_3v, g_ctx%is_hydro(3), sol_T, ierr)
    call VecCopy(sol_T, g_ctx%work_4, ierr)
    call VecRestoreSubVector(g_ctx%hydro_predictor_3v, g_ctx%is_hydro(3), sol_T, ierr)

    ! --- Step 2: Magnetic predictor — advect flux with predicted u, T ---
    ! b_psi = x_psi - B_13*temp_j - B_12*u_pred - B_16*T_pred
    call MatMult(g_ctx%B_13, g_ctx%work_1, g_ctx%work_3, ierr)
    call VecWAXPY(y_psi, -1.0d0, g_ctx%work_3, x_psi, ierr)
    call MatMult(g_ctx%B_12, g_ctx%work_5, g_ctx%work_3, ierr)
    call VecAXPY(y_psi, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_16, g_ctx%work_4, g_ctx%work_3, ierr)
    call VecAXPY(y_psi, -1.0d0, g_ctx%work_3, ierr)
    call VecCopy(y_psi, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_psi, g_ctx%work_3, y_psi, ierr)

    ! --- Step 3: Alfven corrector — full block solve of S_PBP, bare RHS ---
    ! r_u = x_u - B_23*temp_j - B_24*temp_w
    call MatMult(g_ctx%B_23, g_ctx%work_1, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_24, g_ctx%work_2, g_ctx%work_4, ierr)
    call VecWAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_3, x_u, ierr)
    call VecAXPY(g_ctx%work_5, -1.0d0, g_ctx%work_4, ierr)
    ! Add psi -> u Lorentz coupling so the corrector residual matches
    ! the coupled (psi,u) block: r_u -= Atilde_21 * y_psi
    call MatMult(g_ctx%Atilde_21, y_psi, g_ctx%work_3, ierr)
    call VecAXPY (g_ctx%work_5, -1.0d0, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_S_PBP, g_ctx%work_5, y_u, ierr)

    ! --- Step 4: Transport correction for rho and T ---
    ! b_rho = x_rho - B_51*y_psi - B_52*y_u
    call MatMult(g_ctx%B_51, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_rho, ierr)
    call MatMult(g_ctx%B_52, y_u, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_rho, g_ctx%work_4, y_rho, ierr)

    ! b_T = x_T - B_63*temp_j - B_61*y_psi - B_62*y_u
    call MatMult(g_ctx%B_63, g_ctx%work_1, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_T, ierr)
    call MatMult(g_ctx%B_61, y_psi, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
    call MatMult(g_ctx%B_62, y_u, g_ctx%work_3, ierr)
    call VecAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, ierr)
    call KSPSolve(g_ctx%ksp_T, g_ctx%work_4, y_T, ierr)

    ! y_psi, y_u, y_rho, y_T are now set; shared back-substitution below
    ! recovers y_j, y_w from y_psi, y_u.

    ierr = 0
  end subroutine apply_block_predictor_corrector

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
                           physics_pc_sub_blocks, physics_pc_sub_blocks_mode

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
    ! work_1 = A_33^{-1} * x_j  (temp_j)
    call KSPSolve(g_ctx%ksp_Mj, x_j, g_ctx%work_1, ierr)
    ! work_2 = A_44^{-1} * x_w  (temp_w)
    call KSPSolve(g_ctx%ksp_Mw, x_w, g_ctx%work_2, ierr)

    ! --- Step 3: Schur-correct the RHS and solve ---
    if (physics_pc_multi_step) then
      call apply_block_predictor_corrector(x_psi, x_u, x_rho, x_T, &
                                           y_psi, y_u, y_rho, y_T, ierr)

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
    ! y_j = A_33^{-1} * (x_j - B_31 * y_psi)
    call MatMult(g_ctx%B_31, y_psi, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_j, ierr)
    call KSPSolve(g_ctx%ksp_Mj, g_ctx%work_4, y_j, ierr)

    ! y_w = A_44^{-1} * (x_w - B_42 * y_u)
    call MatMult(g_ctx%B_42, y_u, g_ctx%work_3, ierr)
    call VecWAXPY(g_ctx%work_4, -1.0d0, g_ctx%work_3, x_w, ierr)
    call KSPSolve(g_ctx%ksp_Mw, g_ctx%work_4, y_w, ierr)

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
