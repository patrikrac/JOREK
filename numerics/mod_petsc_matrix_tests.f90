module mod_petsc_matrix_tests
!----------------------------------------------------------------------
! Solver tests for PETSc Mat objects using a manufactured solution.
!
! For each matrix the test:
!   1. Generates a random solution vector x_exact.
!   2. Computes the right-hand side b = A * x_exact.
!   3. Solves A * x = b with (a) MUMPS direct and (b) iterative Krylov.
!   4. Reports residual norm, relative error, iteration count, and timing.
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

    call petsc_mat_solve_test(my_id, comm, A_j,    "A_j",    symmetric=.true.)
    call petsc_mat_solve_test(my_id, comm, A_w,    "A_w",    symmetric=.true.)
    call petsc_mat_solve_test(my_id, comm, A_jpsi, "A_jpsi", symmetric=.false.)
    call petsc_mat_solve_test(my_id, comm, A_wu,   "A_wu",   symmetric=.false.)

    if (my_id == 0) write(*,'(A)') &
      "==========================================================="
  end subroutine petsc_run_matrix_tests


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
    call PetscRandomSetType(rctx, PETSCRAND48, ierr)
    call VecSetRandom(x_exact, rctx, ierr)
    call PetscRandomDestroy(rctx, ierr)

    call MatMult(A, x_exact, b, ierr)

    if (my_id == 0) write(*,'(3A)') "  --- ", trim(label), " ---"

    call run_one_solve(my_id, comm, A, b, x_exact, &
                       "MUMPS direct    ", use_direct=.true., use_cg=.false.)

    call run_one_solve(my_id, comm, A, b, x_exact, &
                       merge("CG + BJacobi    ", "GMRES + BJacobi ", symmetric), &
                       use_direct=.false., use_cg=symmetric)

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
                            use_direct, use_cg)
    integer,          intent(in) :: my_id, comm
    Mat,              intent(in) :: A
    Vec,              intent(in) :: b, x_exact
    character(len=*), intent(in) :: solver_name
    logical,          intent(in) :: use_direct, use_cg

    KSP  :: ksp
    PC   :: pc
    Vec  :: x_sol
    KSPConvergedReason :: reason
    PetscInt   :: its
    PetscReal  :: rnorm, err_norm, ref_norm
    PetscErrorCode :: ierr
    integer :: cc0, cc1, cr
    real    :: t_elapsed

    call VecDuplicate(b, x_sol, ierr)
    call VecSet(x_sol, 0.0d0, ierr)

    call KSPCreate(comm, ksp, ierr)
    call KSPSetOperators(ksp, A, A, ierr)

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
      call PCSetType(pc, PCBJACOBI, ierr)
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
  end subroutine run_one_solve

#endif
end module mod_petsc_matrix_tests
