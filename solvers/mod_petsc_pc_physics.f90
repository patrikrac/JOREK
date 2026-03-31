module mod_petsc_pc_physics
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_setup_physics_pc, petsc_update_physics_pc_ctx

  type :: type_physics_pc_ctx
    logical :: initialized = .false.
    ! TODO: add physics-specific fields here
    !   e.g. sub-matrices, communicators, scaling factors
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

  !> Refresh the module-level context after a matrix rebuild.
  !! Extend the argument list as the physics PC is developed.
  subroutine petsc_update_physics_pc_ctx()
    ! TODO: update g_ctx fields from current matrix data
  end subroutine petsc_update_physics_pc_ctx

  !> PCSHELL apply callback: compute y = M^{-1} x.
  !! Accesses g_ctx directly via module scope.
  subroutine physics_pc_apply(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    ! TODO: implement physics-based apply using g_ctx
  end subroutine physics_pc_apply

end module mod_petsc_pc_physics
