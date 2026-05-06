module mod_petsc_matrix_tests
!----------------------------------------------------------------------
! Solver tests for PETSc Mat objects using a manufactured solution.
!
! A_j and A_w (SPD mass matrices) get a full manufactured-solution test:
!   1. Generate random x_exact, compute b = A * x_exact.
!   2. Solve A * x = b with MUMPS, CG+BJacobi, CG+GAMG, CG+BoomerAMG,
!      and (when n_axis_dofs > 0) fieldsplit variants where the axis DOF
!      block is solved with MUMPS and the bulk block with GAMG/BoomerAMG.
!   3. Report residual norm, relative error, iteration count, and timing.
!
! A_jpsi and A_wu (off-diagonal coupling, singular without BCs) are used
! only as MatMult operators in physics_pc_apply and are never inverted.
! They receive a MatMult sanity check only: b = A * x, report ||b||/||x||.
!
! Intended for diagnostic/debug use, not production runs.
!
! Public interface (USE_PETSC):
!   petsc_run_matrix_tests(my_id, comm, A_j, A_w, A_jpsi, A_wu[, n_axis_dofs])
!----------------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_settings, only: n_degrees
  implicit none
  private
  public :: petsc_run_matrix_tests

contains

  !--------------------------------------------------------------------
  !> Run manufactured-solution solver tests on all four PC sub-matrices.
  !!
  !! @param n_axis_dofs  Number of axis DOF rows at the start of the global
  !!                     index space (from JOREK axis nodes).  When > 0,
  !!                     additional fieldsplit tests are run: axis block
  !!                     solved with MUMPS, bulk block with GAMG/BoomerAMG.
  !--------------------------------------------------------------------
  subroutine petsc_run_matrix_tests(my_id, comm, A_j, A_w, A_jpsi, A_wu, n_axis_dofs)
    integer, intent(in)           :: my_id, comm
    Mat,     intent(in)           :: A_j, A_w, A_jpsi, A_wu
    integer, intent(in), optional :: n_axis_dofs

    if (my_id == 0) write(*,'(A)') &
      "=== PC matrix solver tests ================================="

    ! Diagonal blocks: SPD mass matrices — full solver test
    call petsc_mat_solve_test(my_id, comm, A_j, "A_j", symmetric=.true., &
                              n_axis_dofs=n_axis_dofs)
    call petsc_mat_solve_test(my_id, comm, A_w, "A_w", symmetric=.true., &
                              n_axis_dofs=n_axis_dofs)

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
  !!  Test 3: CG/GMRES + GAMG (smoothed aggregation AMG).
  !!  Test 4: CG/GMRES + Hypre BoomerAMG (requires HYPRE-enabled PETSc).
  !!  Tests 5-6 (when n_axis_dofs > 0): fieldsplit variants — axis block
  !!  solved with MUMPS, bulk block with GAMG or BoomerAMG respectively.
  !--------------------------------------------------------------------
  subroutine petsc_mat_solve_test(my_id, comm, A, label, symmetric, n_axis_dofs)
    integer,          intent(in)           :: my_id, comm
    Mat,              intent(in)           :: A
    character(len=*), intent(in)           :: label
    logical,          intent(in)           :: symmetric
    integer,          intent(in), optional :: n_axis_dofs

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

    call run_one_solve(my_id, comm, A, b, x_exact, &
                       merge("CG + BoomerAMG   ", "GMRES + BoomerAMG", symmetric), &
                       use_direct=.false., use_cg=symmetric, use_hypre_amg=.true.)

    ! Fieldsplit variants: axis block → MUMPS, bulk block → AMG.
    ! Only run when the caller provides the axis DOF count.
    if (present(n_axis_dofs)) then
      if (n_axis_dofs > 0) then
        call run_one_solve(my_id, comm, A, b, x_exact, &
                           merge("CG+GAMG+Split    ", "GMRES+GAMG+Split ", symmetric), &
                           use_direct=.false., use_cg=symmetric, use_amg=.true., &
                           use_fieldsplit=.true., n_axis_dofs=n_axis_dofs)
        call run_one_solve(my_id, comm, A, b, x_exact, &
                           merge("CG+Boom+Split    ", "GMRES+Boom+Split ", symmetric), &
                           use_direct=.false., use_cg=symmetric, use_hypre_amg=.true., &
                           use_fieldsplit=.true., n_axis_dofs=n_axis_dofs)
      endif
    endif

    call VecDestroy(x_exact, ierr)
    call VecDestroy(b, ierr)
  end subroutine petsc_mat_solve_test


  !--------------------------------------------------------------------
  !> Solve A*x=b, measure relative error against x_exact, and print.
  !!
  !! @param use_direct     .true. → MUMPS (PREONLY+LU); .false. → iterative
  !! @param use_cg         (iterative only) .true. → CG; .false. → GMRES
  !! @param use_amg        (iterative only) .true. → PCGAMG
  !! @param use_hypre_amg  (iterative only) .true. → PCHYPRE boomeramg.
  !!                       Requires HYPRE-enabled PETSc.
  !! @param use_fieldsplit .true. → PCFIELDSPLIT additive: IS "axis" solved
  !!                       with MUMPS, IS "bulk" solved with GAMG or
  !!                       BoomerAMG.  Requires n_axis_dofs > 0.
  !! @param n_axis_dofs    Number of axis DOF rows at the front of the
  !!                       global index range.  Used only when
  !!                       use_fieldsplit = .true.
  !--------------------------------------------------------------------
  subroutine run_one_solve(my_id, comm, A, b, x_exact, solver_name, &
                            use_direct, use_cg, use_amg, use_hypre_amg, &
                            use_fieldsplit, n_axis_dofs)
    integer,          intent(in)           :: my_id, comm
    Mat,              intent(in)           :: A
    Vec,              intent(in)           :: b, x_exact
    character(len=*), intent(in)           :: solver_name
    logical,          intent(in)           :: use_direct, use_cg
    logical,          intent(in), optional :: use_amg
    logical,          intent(in), optional :: use_hypre_amg
    logical,          intent(in), optional :: use_fieldsplit
    integer,          intent(in), optional :: n_axis_dofs

    KSP  :: ksp
    PC   :: pc, sub_pc
    KSP, pointer :: sub_ksp(:)
    Mat  :: A_op, sub_A
    IS   :: is_axis, is_bulk
    Vec  :: x_sol
    MatNullSpace       :: nullsp
    KSPConvergedReason :: reason
    PetscInt   :: its, bs, n_total, n_ax, n_bk, nsplit
    PetscReal  :: rnorm, err_norm, ref_norm
    PetscErrorCode :: ierr
    integer :: cc0, cc1, cr
    real    :: t_elapsed
    logical :: do_amg, do_hypre_amg, do_fieldsplit, converted

    do_amg        = .false.
    do_hypre_amg  = .false.
    do_fieldsplit = .false.
    if (present(use_amg))        do_amg        = use_amg
    if (present(use_hypre_amg))  do_hypre_amg  = use_hypre_amg
    if (present(use_fieldsplit)) do_fieldsplit  = use_fieldsplit
    n_ax = 0
    if (present(n_axis_dofs))    n_ax          = n_axis_dofs
    if (do_fieldsplit .and. n_ax == 0) do_fieldsplit = .false.

    ! PCGAMG, PCHYPRE boomeramg, and the bulk sub-PC inside PCFIELDSPLIT
    ! all require a scalar (AIJ) matrix.  Read the MPIBAIJ block size at
    ! runtime (= n_tor_local, not the compile-time n_tor constant which can
    ! differ).  The true mesh-node block is n_tor_local * n_degrees; only
    ! set it when bs > 1 so that a matrix that arrived as scalar AIJ is
    ! handled safely.
    converted = .false.
    if (do_amg .or. do_hypre_amg) then
      call MatGetBlockSize(A, bs, ierr)
      call MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, A_op, ierr)
      if (bs > 1) call MatSetBlockSize(A_op, bs * n_degrees, ierr)
      converted = .true.
    else
      A_op = A
    endif

    call VecDuplicate(b, x_sol, ierr)
    call VecSet(x_sol, 0.0d0, ierr)

    call KSPCreate(comm, ksp, ierr)
    ! Use a unique prefix for non-fieldsplit BoomerAMG so its options don't
    ! affect other KSPs.  Override at runtime via -hmg_pc_hypre_boomeramg_*
    ! or -hmg_ksp_* flags.
    if (do_hypre_amg .and. .not. do_fieldsplit) &
      call KSPSetOptionsPrefix(ksp, "hmg_", ierr)
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

      if (do_fieldsplit) then
        ! Additive PCFIELDSPLIT: IS "axis" (first n_ax rows) solved with
        ! MUMPS; IS "bulk" (remaining rows) solved with GAMG or BoomerAMG
        ! with the correct nodal block size.  This handles the magnetic
        ! axis where degenerate element DOFs break divisibility by the full
        ! n_tor_local*n_degrees block, preventing AMG from running on the
        ! full matrix.
        call PCSetType(pc, PCFIELDSPLIT, ierr)
        call PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE, ierr)

        call MatGetSize(A_op, n_total, PETSC_NULL_INTEGER, ierr)
        n_bk = n_total - n_ax

        call ISCreateStride(comm, n_ax, 0, 1, is_axis, ierr)
        call ISCreateStride(comm, n_bk, n_ax, 1, is_bulk, ierr)
        call PCFieldSplitSetIS(pc, "axis", is_axis, ierr)
        call PCFieldSplitSetIS(pc, "bulk", is_bulk, ierr)
        call ISDestroy(is_axis, ierr)
        call ISDestroy(is_bulk, ierr)

        ! Force sub-matrix extraction so sub-KSPs can be configured before
        ! the actual solve.
        call KSPSetUp(ksp, ierr)
        call PCFieldSplitGetSubKSP(pc, nsplit, sub_ksp, ierr)

        ! Axis sub-KSP: PREONLY + MUMPS direct
        call KSPSetType(sub_ksp(1), KSPPREONLY, ierr)
        call KSPGetPC(sub_ksp(1), sub_pc, ierr)
        call PCSetType(sub_pc, PCLU, ierr)
        call PCFactorSetMatSolverType(sub_pc, MATSOLVERMUMPS, ierr)

        ! Bulk sub-KSP: PREONLY + AMG
        call KSPSetType(sub_ksp(2), KSPPREONLY, ierr)
        call KSPGetPC(sub_ksp(2), sub_pc, ierr)

        ! Set nodal block size on the extracted bulk sub-matrix.  The
        ! sub-matrix does not automatically inherit the block size set on
        ! A_op; setting it here drives nodal coarsening in AMG.
        call KSPGetOperators(sub_ksp(2), sub_A, PETSC_NULL_MAT, ierr)
        if (bs > 1) call MatSetBlockSize(sub_A, bs * n_degrees, ierr)

        if (do_amg) then
          call PCSetType(sub_pc, PCGAMG, ierr)
          call MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL_VEC, nullsp, ierr)
          call MatSetNearNullSpace(sub_A, nullsp, ierr)
          call MatNullSpaceDestroy(nullsp, ierr)
        else if (do_hypre_amg) then
          call PCSetType(sub_pc, PCHYPRE, ierr)
          call PCHYPRESetType(sub_pc, "boomeramg", ierr)
          call MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL_VEC, nullsp, ierr)
          call MatSetNearNullSpace(sub_A, nullsp, ierr)
          call MatNullSpaceDestroy(nullsp, ierr)
          ! PCFIELDSPLIT sub-KSPs are prefixed "fieldsplit_<name>_"; set
          ! BoomerAMG options under "fieldsplit_bulk_" accordingly.
          if (bs > 1) then
            call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
                "-fieldsplit_bulk_pc_hypre_boomeramg_nodal_coarsen", "6", ierr)
          endif
          call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
              "-fieldsplit_bulk_pc_hypre_boomeramg_coarsen_type", "Falgout", ierr)
          call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
              "-fieldsplit_bulk_pc_hypre_boomeramg_relax_type_all", &
              "symmetric-SOR/Jacobi", ierr)
        endif

      else if (do_amg) then
        call PCSetType(pc, PCGAMG, ierr)
        ! Constant near-null space improves GAMG aggregation on the poloidal mesh.
        call MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL_VEC, nullsp, ierr)
        call MatSetNearNullSpace(A_op, nullsp, ierr)
        call MatNullSpaceDestroy(nullsp, ierr)
      else if (do_hypre_amg) then
        call PCSetType(pc, PCHYPRE, ierr)
        call PCHYPRESetType(pc, "boomeramg", ierr)
        call MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL_VEC, nullsp, ierr)
        call MatSetNearNullSpace(A_op, nullsp, ierr)
        call MatNullSpaceDestroy(nullsp, ierr)
        ! Nodal coarsening (criterion 6 = measured strength): groups all
        ! n_tor_local*n_degrees DOFs of one FEM mesh node onto the same coarse
        ! point. Criterion 6 preserves the SPD property through the Galerkin
        ! product; criterion 1 (Frobenius norm) does not and causes
        ! KSP_DIVERGED_INDEFINITE_PC with CG. Skipped if A arrived as scalar
        ! AIJ (bs=1) since the nodal block structure would be unknown.
        if (bs > 1) then
          call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
              "-hmg_pc_hypre_boomeramg_nodal_coarsen", "6", ierr)
        endif
        ! Falgout coarsening: proven stable with nodal_coarsen=6 on 2D FEM
        ! systems (PETSc ex49 elasticity). Override at runtime with
        ! -hmg_pc_hypre_boomeramg_coarsen_type HMIS to test parallel scaling.
        call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
            "-hmg_pc_hypre_boomeramg_coarsen_type", "Falgout", ierr)
        ! Symmetric SOR/Jacobi smoother for SPD mass matrices.
        call PetscOptionsSetValue(PETSC_NULL_OPTIONS, &
            "-hmg_pc_hypre_boomeramg_relax_type_all", "symmetric-SOR/Jacobi", ierr)
      else
        call PCSetType(pc, PCBJACOBI, ierr)
      endif
      call KSPSetTolerances(ksp, 1.0d-8, PETSC_DEFAULT_REAL, &
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
    ! PetscOptionsSetValue writes to the process-global database and persists
    ! after this call returns; clear the scoped entries to avoid poisoning
    ! any later KSP that happens to use the same prefix.
    if (do_hypre_amg .and. .not. do_fieldsplit) then
      call PetscOptionsClearValue(PETSC_NULL_OPTIONS, &
          "-hmg_pc_hypre_boomeramg_nodal_coarsen", ierr)
      call PetscOptionsClearValue(PETSC_NULL_OPTIONS, &
          "-hmg_pc_hypre_boomeramg_coarsen_type", ierr)
      call PetscOptionsClearValue(PETSC_NULL_OPTIONS, &
          "-hmg_pc_hypre_boomeramg_relax_type_all", ierr)
    endif
    if (do_hypre_amg .and. do_fieldsplit) then
      call PetscOptionsClearValue(PETSC_NULL_OPTIONS, &
          "-fieldsplit_bulk_pc_hypre_boomeramg_nodal_coarsen", ierr)
      call PetscOptionsClearValue(PETSC_NULL_OPTIONS, &
          "-fieldsplit_bulk_pc_hypre_boomeramg_coarsen_type", ierr)
      call PetscOptionsClearValue(PETSC_NULL_OPTIONS, &
          "-fieldsplit_bulk_pc_hypre_boomeramg_relax_type_all", ierr)
    endif
  end subroutine run_one_solve

#endif
end module mod_petsc_matrix_tests
