module mod_petsc_matrix_tests
!----------------------------------------------------------------------
! Solver tests for PETSc Mat objects using a manufactured solution.
!
! A_j and A_w (SPD mass matrices) get a full manufactured-solution test:
!   1. Generate random x_exact, compute b = A * x_exact.
!   2. Solve A * x = b with MUMPS, CG+BJacobi, and CG+GAMG.
!   3. Report residual norm, relative error, iteration count, and timing.
!
! A_jpsi and A_wu (off-diagonal coupling, singular without BCs) are used
! only as MatMult operators in physics_pc_apply and are never inverted.
! They receive a MatMult sanity check only: b = A * x, report ||b||/||x||.
!
! Intended for diagnostic/debug use, not production runs.
!
! Public interface (USE_PETSC):
!   petsc_run_matrix_tests(my_id, comm, A_j, A_w, A_jpsi, A_wu)
!----------------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_run_matrix_tests

contains

  !--------------------------------------------------------------------
  !> Run manufactured-solution solver tests on all four PC sub-matrices.
  !--------------------------------------------------------------------
  subroutine petsc_run_matrix_tests(my_id, comm, A_j, A_w, A_jpsi, A_wu)
    integer, intent(in) :: my_id, comm
    Mat,     intent(in) :: A_j, A_w, A_jpsi, A_wu

    if (my_id == 0) write(*,'(A)') &
      "=== PC matrix solver tests ================================="

    ! Diagonal blocks: SPD mass matrices — full solver test
    call petsc_mat_solve_test(my_id, comm, A_j, "A_j", symmetric=.true.)
    call petsc_mat_solve_test(my_id, comm, A_w, "A_w", symmetric=.true.)

    ! Off-diagonal blocks: singular without BCs, used as MatMult only in apply
    call petsc_mat_matvec_test(my_id, comm, A_jpsi, "A_jpsi")
    call petsc_mat_matvec_test(my_id, comm, A_wu,   "A_wu")

    if (my_id == 0) write(*,'(A)') &
      "==========================================================="
  end subroutine petsc_run_matrix_tests


  !--------------------------------------------------------------------
  !> MatMult sanity check for an operator-only matrix (not inverted).
  !! Verifies the matrix is assembled and non-trivial by computing
  !! b = A * x for a random x and reporting ||b||_2 / ||x||_2.
  !--------------------------------------------------------------------
  subroutine petsc_mat_matvec_test(my_id, comm, A, label)
    integer,          intent(in) :: my_id, comm
    Mat,              intent(in) :: A
    character(len=*), intent(in) :: label

    Vec            :: x, b
    PetscRandom    :: rctx
    PetscReal      :: norm_x, norm_b
    PetscErrorCode :: ierr

    call MatCreateVecs(A, x, b, ierr)

    call PetscRandomCreate(comm, rctx, ierr)
    call PetscRandomSetType(rctx, "rand", ierr)
    call VecSetRandom(x, rctx, ierr)
    call PetscRandomDestroy(rctx, ierr)

    call MatMult(A, x, b, ierr)

    call VecNorm(x, NORM_2, norm_x, ierr)
    call VecNorm(b, NORM_2, norm_b, ierr)

    if (my_id == 0) then
      write(*,'(3A)') "  --- ", trim(label), " (MatMult only) ---"
      write(*,'(A,ES10.3,A,ES10.3,A,ES10.3)') &
        "  ||A*x||/||x|| = ", norm_b/norm_x, &
        "  (||x||=", norm_x, "  ||A*x||=", norm_b, ")"
    endif

    call VecDestroy(x, ierr)
    call VecDestroy(b, ierr)
  end subroutine petsc_mat_matvec_test


  !--------------------------------------------------------------------
  !> Manufactured-solution test for a single Mat:
  !!  x_exact = VecSetRandom,  b = A * x_exact
  !!  Test 1: MUMPS direct solver (verifies matrix is non-singular).
  !!  Test 2: CG (symmetric) or GMRES (general) + block Jacobi.
  !--------------------------------------------------------------------
  subroutine petsc_mat_solve_test(my_id, comm, A, label, symmetric)
    integer,          intent(in) :: my_id, comm
    Mat,              intent(in) :: A
    character(len=*), intent(in) :: label
    logical,          intent(in) :: symmetric

    Vec            :: x_exact, b
    PetscRandom    :: rctx
    PetscErrorCode :: ierr

    ! Allocate vectors with A's parallel layout
    call MatCreateVecs(A, x_exact, b, ierr)

    ! Random manufactured solution
    call PetscRandomCreate(comm, rctx, ierr)
    call PetscRandomSetType(rctx, "rand", ierr)
    call VecSetRandom(x_exact, rctx, ierr)
    call PetscRandomDestroy(rctx, ierr)

    call MatMult(A, x_exact, b, ierr)

    if (my_id == 0) write(*,'(3A)') "  --- ", trim(label), " ---"

    call run_one_solve(my_id, comm, A, b, x_exact, &
                       "MUMPS direct    ", use_direct=.true., use_cg=.false.)

    call run_one_solve(my_id, comm, A, b, x_exact, &
                       merge("CG + BJacobi    ", "GMRES + BJacobi ", symmetric), &
                       use_direct=.false., use_cg=symmetric)

    call run_one_solve(my_id, comm, A, b, x_exact, &
                       merge("CG + GAMG       ", "GMRES + GAMG    ", symmetric), &
                       use_direct=.false., use_cg=symmetric, use_amg=.true.)

    call VecDestroy(x_exact, ierr)
    call VecDestroy(b, ierr)
  end subroutine petsc_mat_solve_test


  !--------------------------------------------------------------------
  !> Solve A*x=b, measure relative error against x_exact, and print.
  !!
  !! @param use_direct  .true. → MUMPS (PREONLY+LU); .false. → iterative
  !! @param use_cg      (iterative only) .true. → CG; .false. → GMRES
  !--------------------------------------------------------------------
  subroutine run_one_solve(my_id, comm, A, b, x_exact, solver_name, &
                            use_direct, use_cg, use_amg)
    integer,          intent(in)           :: my_id, comm
    Mat,              intent(in)           :: A
    Vec,              intent(in)           :: b, x_exact
    character(len=*), intent(in)           :: solver_name
    logical,          intent(in)           :: use_direct, use_cg
    logical,          intent(in), optional :: use_amg

    KSP  :: ksp
    PC   :: pc
    Mat  :: A_op   ! may be a converted copy for AMG
    Vec  :: x_sol
    KSPConvergedReason :: reason
    PetscInt   :: its
    PetscReal  :: rnorm, err_norm, ref_norm
    PetscErrorCode :: ierr
    integer :: cc0, cc1, cr
    real    :: t_elapsed
    logical :: do_amg, converted

    do_amg = .false.
    if (present(use_amg)) do_amg = use_amg

    ! PCGAMG requires a scalar (AIJ) matrix for its coarsening algorithm.
    ! MPIBAIJ block matrices confuse the aggregation, so convert first.
    converted = .false.
    if (do_amg) then
      call MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, A_op, ierr)
      converted = .true.
    else
      A_op = A
    endif

    call VecDuplicate(b, x_sol, ierr)
    call VecSet(x_sol, 0.0d0, ierr)

    call KSPCreate(comm, ksp, ierr)
    call KSPSetOperators(ksp, A_op, A_op, ierr)

    if (use_direct) then
      call KSPSetType(ksp, KSPPREONLY, ierr)
      call KSPGetPC(ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
    else
      if (use_cg) then
        call KSPSetType(ksp, KSPCG, ierr)
      else
        call KSPSetType(ksp, KSPGMRES, ierr)
      endif
      call KSPGetPC(ksp, pc, ierr)
      if (do_amg) then
        ! PCGAMG: PETSc smoothed-aggregation AMG (no external library needed).
        ! Override at runtime with -pc_type hypre -pc_hypre_type boomeramg for HYPRE.
        call PCSetType(pc, PCGAMG, ierr)
      else
        call PCSetType(pc, PCBJACOBI, ierr)
      endif
      call KSPSetTolerances(ksp, 1.0d-10, PETSC_DEFAULT_REAL, &
                             PETSC_DEFAULT_REAL, 10000, ierr)
    endif

    ! Allow runtime override via -ksp_type, -pc_type, etc.
    call KSPSetFromOptions(ksp, ierr)

    call system_clock(count=cc0, count_rate=cr)
    call KSPSolve(ksp, b, x_sol, ierr)
    call system_clock(count=cc1)
    t_elapsed = real(cc1 - cc0) / real(cr)

    call KSPGetConvergedReason(ksp, reason, ierr)
    call KSPGetIterationNumber(ksp, its, ierr)
    call KSPGetResidualNorm(ksp, rnorm, ierr)

    ! Relative error: ||x_sol - x_exact||_2 / ||x_exact||_2
    call VecAXPY(x_sol, -1.0d0, x_exact, ierr)
    call VecNorm(x_sol,   NORM_2, err_norm, ierr)
    call VecNorm(x_exact, NORM_2, ref_norm, ierr)

    if (my_id == 0) then
      write(*,'(3A,I6,2(A,ES10.3),A,F8.3,A)') &
        "  ", trim(solver_name), "  its=", its, &
        "  |r|=", rnorm, "  rel_err=", err_norm/ref_norm, &
        "  t=", t_elapsed, "s"
      if (reason < 0) &
        write(*,'(A,I0)') "    WARNING: KSP diverged, reason=", reason
    endif

    call VecDestroy(x_sol, ierr)
    call KSPDestroy(ksp, ierr)
    if (converted) call MatDestroy(A_op, ierr)
  end subroutine run_one_solve

#endif
end module mod_petsc_matrix_tests
