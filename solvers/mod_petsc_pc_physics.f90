!> Entry point of the physics-based preconditioner (use_physics_pc): the SF
!! sweep of mod_petsc_pc_sf or the vorticity-space Schur PC of
!! mod_petsc_pc_vs, by physics_pc_method. jorek2_main assembles their element
!! matrices through petsc_assemble_pc_matrices, mod_petsc registers the
!! PCSHELL, rebuilds the preconditioner at every PC rebuild and reports it.
module mod_petsc_pc_physics
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_log_events_register, physics_pc_method_id, PC_METHOD_SF
  use mod_petsc_pc_physics_element, only: petsc_assemble_pc_matrices
  implicit none
  private
  public :: petsc_setup_physics_pc, petsc_assemble_pc_matrices, &
            petsc_physics_pc_build_reduced, petsc_physics_pc_report

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
  end subroutine petsc_setup_physics_pc

  !--------------------------------------------------------------------
  !> (Re)build the preconditioner's blocks and solvers from the full
  !! system matrix. Collective.
  !--------------------------------------------------------------------
  subroutine petsc_physics_pc_build_reduced(A_full)
    use mod_petsc_pc_sf, only: sf_build
    use mod_petsc_pc_vs, only: vs_build
    Mat, intent(in) :: A_full

    PetscErrorCode :: ierr
    integer :: comm, my_id, mpierr

    call PetscObjectGetComm(A_full, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)
    call physics_pc_log_events_register()
    if (physics_pc_method_id() == PC_METHOD_SF) then
      call sf_build(A_full, comm, my_id)
    else
      call vs_build(A_full, comm, my_id)
    endif
  end subroutine petsc_physics_pc_build_reduced

  !--------------------------------------------------------------------
  !> The inner-iteration summary since the last report. Collective.
  !--------------------------------------------------------------------
  subroutine petsc_physics_pc_report(my_id)
    use mod_petsc_pc_sf, only: sf_report
    use mod_petsc_pc_vs, only: vs_report
    integer, intent(in) :: my_id

    if (physics_pc_method_id() == PC_METHOD_SF) then
      call sf_report(my_id)
    else
      call vs_report(my_id)
    endif
  end subroutine petsc_physics_pc_report

  !--------------------------------------------------------------------
  !> PCSHELL apply callback: y = P^{-1} x.
  !--------------------------------------------------------------------
  subroutine physics_pc_apply(pc_obj, x, y, ierr)
    use mod_petsc_pc_sf, only: sf_apply
    use mod_petsc_pc_vs, only: vs_apply
    PC :: pc_obj
    Vec :: x, y
    PetscErrorCode :: ierr

    if (.not. g_ctx%reduced_ready) then
      ierr = 1
      return
    endif
    if (physics_pc_method_id() == PC_METHOD_SF) then
      call sf_apply(x, y, ierr)
    else
      call vs_apply(x, y, ierr)
    endif
  end subroutine physics_pc_apply

#endif
end module mod_petsc_pc_physics
