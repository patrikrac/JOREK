!> Stub for the physics-based block preconditioner.
!!
!! The working implementation lives on the numerics_develop_pbpc branch, where
!! the physics-based preconditioners are being developed. On numerics_develop
!! only the verified toroidal-harmonic preconditioner
!! (mod_petsc_pc_toroidal) is a live option; this module keeps the PCSHELL
!! registration and context skeleton so the dispatch in mod_petsc_pc stays
!! intact, but applies the identity.
module mod_petsc_pc_physics
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_setup_physics_pc

  type :: type_physics_pc_ctx
    logical :: initialized    = .false.
    logical :: matrices_ready = .false.
    integer :: comm           = -1    !< MPI communicator shared by the PC matrices
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


  !> PCSHELL apply callback: y = M^{-1} x.  Stub: applies the identity.
  subroutine physics_pc_apply(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    PetscCallA(VecCopy(x, y, ierr))
  end subroutine physics_pc_apply

#endif
end module mod_petsc_pc_physics
