module mod_petsc_pc_physics
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_setup_physics_pc, petsc_create_pc_matrices, &
            petsc_assemble_pc_matrices, petsc_update_physics_pc_ctx, &
            petsc_analyze_pc_matrices

  type :: type_physics_pc_ctx
    logical :: initialized    = .false.
    logical :: matrices_ready = .false.
    !> 1-var BAIJ matrix for j equation,  block_size = n_tor_local
    Mat :: A_j
    !> 1-var BAIJ matrix for w equation,  block_size = n_tor_local  (= A_j)
    Mat :: A_w
    !> 1-var BAIJ matrix for psi->j off-diagonal coupling, block_size = n_tor_local
    Mat :: A_jpsi
    !> 1-var BAIJ matrix for u->w off-diagonal coupling,   block_size = n_tor_local
    Mat :: A_wu
  end type type_physics_pc_ctx

  type(type_physics_pc_ctx), save :: g_ctx

contains

  !> Register the physics-based PCSHELL on an existing KSP.
  !! A is provided for any matrix-structure queries during setup.
  subroutine petsc_setup_physics_pc(ksp, A)
    KSP, intent(inout) :: ksp
    Mat, intent(in)    :: A
    PC :: pc
    PetscErrorCode :: ierr

    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(pc, physics_pc_apply, ierr))
    g_ctx%initialized = .true.
  end subroutine petsc_setup_physics_pc


  !> Create the four PC sub-matrices (sparsity allocation only).
  !! Must be called once before petsc_assemble_pc_matrices.
  subroutine petsc_create_pc_matrices(a_mat)
    use mod_petsc,       only: petsc_create_pc_matrix
    use data_structure,  only: type_SP_MATRIX

    type(type_SP_MATRIX), intent(in) :: a_mat

    call petsc_create_pc_matrix(g_ctx%A_j,    a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_w,    a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_jpsi, a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_wu,   a_mat, 1)
  end subroutine petsc_create_pc_matrices


  !> Create and assemble the four PC sub-matrices.
  !! On the first call, allocates the matrices (sparsity) and runs analysis
  !! if debug_physics_pc is set.
  !! On subsequent calls, zeros and re-fills them.
  !! Must be called after construct_matrix has been called for a_mat.
  subroutine petsc_assemble_pc_matrices(my_id, local_elms, n_local_elms, a_mat)
    use construct_pc_matrix_mod
    use data_structure,  only: type_SP_MATRIX
    use phys_module,     only: debug_physics_pc

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    PetscErrorCode :: ierr
    logical        :: first_assembly

    first_assembly = .not. g_ctx%matrices_ready

    ! Create matrices on first call
    if (first_assembly) then
      call petsc_create_pc_matrices(a_mat)
    else
      ! Zero existing entries before re-assembly
      PetscCallA(MatZeroEntries(g_ctx%A_j,    ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_w,    ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_jpsi, ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_wu,   ierr))
    endif

    call construct_pc_elliptic_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                        g_ctx%A_j, g_ctx%A_w, g_ctx%A_jpsi, g_ctx%A_wu)
    g_ctx%matrices_ready = .true.

    ! Run analysis once after the first assembly
    if (first_assembly .and. debug_physics_pc) &
      call petsc_analyze_pc_matrices(my_id)
  end subroutine petsc_assemble_pc_matrices


  !> Refresh the module-level context after a matrix rebuild.
  !! Extend the argument list as the physics PC is developed.
  subroutine petsc_update_physics_pc_ctx()
    ! TODO: update g_ctx fields from current matrix data
  end subroutine petsc_update_physics_pc_ctx


  !> Print structural info, norms, and (with SLEPc) eigenvalue analysis
  !! for all four PC sub-matrices.  Gated by debug_physics_pc in the
  !! namelist; can also be called directly for one-off diagnostics.
  !!
  !! With SLEPc: computes condition number for the symmetric 1-var matrices
  !! (A_j, A_w) and writes full spectra for all four matrices to
  !! {label}_spectrum.dat files.
  subroutine petsc_analyze_pc_matrices(my_id)
    use mod_petsc_matrix_analysis

    integer, intent(in) :: my_id
    PetscReal :: diff_norm
#ifdef USE_SLEPC
    PetscReal :: kappa
#endif

    if (my_id == 0) write(*,'(A)') &
      "=== PC matrix analysis ================================="

    call petsc_mat_print_info(g_ctx%A_j,    "A_j")
    call petsc_mat_print_info(g_ctx%A_w,    "A_w")
    call petsc_mat_print_info(g_ctx%A_jpsi, "A_jpsi")
    call petsc_mat_print_info(g_ctx%A_wu,   "A_wu")

    call petsc_mat_norms(g_ctx%A_j,    "A_j")
    call petsc_mat_norms(g_ctx%A_w,    "A_w")
    call petsc_mat_norms(g_ctx%A_jpsi, "A_jpsi")
    call petsc_mat_norms(g_ctx%A_wu,   "A_wu")

    call petsc_mat_diff_norm(g_ctx%A_j, g_ctx%A_w, "A_j vs A_w", diff_norm)

#ifdef USE_SLEPC
    ! Condition number (symmetric 1-var matrices)
    call petsc_mat_cond_estimate(g_ctx%A_j, kappa)
    if (my_id == 0) write(*,'(A,ES12.4)') "[PC] cond(A_j) = ", kappa
    call petsc_mat_cond_estimate(g_ctx%A_w, kappa)
    if (my_id == 0) write(*,'(A,ES12.4)') "[PC] cond(A_w) = ", kappa

    ! Full spectra — 0 requests all eigenvalues
    call petsc_mat_full_spectrum(g_ctx%A_j,    "A_j",    0, symmetric=.true.)
    call petsc_mat_full_spectrum(g_ctx%A_w,    "A_w",    0, symmetric=.true.)
    call petsc_mat_full_spectrum(g_ctx%A_jpsi, "A_jpsi", 0, symmetric=.false.)
    call petsc_mat_full_spectrum(g_ctx%A_wu,   "A_wu",   0, symmetric=.false.)
#endif

    if (my_id == 0) write(*,'(A)') &
      "========================================================"
  end subroutine petsc_analyze_pc_matrices


  !> PCSHELL apply callback: compute y = M^{-1} x.
  !! Accesses g_ctx directly via module scope.
  subroutine physics_pc_apply(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    ! TODO: implement physics-based apply using g_ctx
  end subroutine physics_pc_apply

#endif
end module mod_petsc_pc_physics
