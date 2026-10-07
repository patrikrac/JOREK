module mod_petsc_pc_physics
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_log_events_register
  use mod_petsc_pc_physics_element, only: petsc_assemble_pc_matrices
  implicit none
  private
  public :: petsc_setup_physics_pc, petsc_assemble_pc_matrices, &
            petsc_physics_pc_build_reduced

  ! The physics-based preconditioner (use_physics_pc): the split-field (SF)
  ! block-LDU sweep of mod_petsc_pc_sf. This module is its entry point:
  ! jorek2_main assembles W (petsc_assemble_pc_matrices), mod_petsc registers
  ! the PCSHELL and (re)builds the blocks at every PC rebuild.

contains

  !--------------------------------------------------------------------
  !> Register the physics-based PCSHELL on an existing KSP.
  !--------------------------------------------------------------------
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

  !--------------------------------------------------------------------
  !> (Re)build the preconditioner's blocks and solvers from the full
  !! system matrix. Collective.
  !--------------------------------------------------------------------
  subroutine petsc_physics_pc_build_reduced(A_full)
    use mod_petsc_pc_sf, only: sf_build
    Mat, intent(in) :: A_full

    PetscErrorCode :: ierr
    integer :: comm, my_id, mpierr

    call PetscObjectGetComm(A_full, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)
    call physics_pc_log_events_register()
    call sf_build(A_full, comm, my_id)
  end subroutine petsc_physics_pc_build_reduced

  !--------------------------------------------------------------------
  !> PCSHELL apply callback: y = P^{-1} x.
  !--------------------------------------------------------------------
  subroutine physics_pc_apply(pc_obj, x, y, ierr)
    use mod_petsc_pc_sf, only: sf_apply
    PC :: pc_obj
    Vec :: x, y
    PetscErrorCode :: ierr

    if (.not. g_ctx%reduced_ready) then
      ierr = 1
      return
    endif
    call sf_apply(x, y, ierr)
  end subroutine physics_pc_apply

#endif
end module mod_petsc_pc_physics
