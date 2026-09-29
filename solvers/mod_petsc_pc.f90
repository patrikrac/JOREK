module mod_petsc_pc
#ifdef USE_PETSC
  use mod_petsc_pc_toroidal
  use mod_petsc_pc_physics
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  public

  integer, parameter :: PETSC_PC_TOROIDAL_HARMONIC = 1
  integer, parameter :: PETSC_PC_PHYSICS           = 2
  integer, parameter :: PETSC_PC_FULL_LU           = 3

contains

  !> One MUMPS LU of the whole coupled system (all harmonics, all variables):
  !! the direct-solve reference for the scaling study. Same MUMPS controls as
  !! the per-harmonic fieldsplit blocks, except in-core (ICNTL(22) = 0), so the
  !! factor is not written to disk inside the timed region.
  subroutine petsc_setup_full_lu_pc(ksp)
    KSP, intent(inout) :: ksp

    PC :: pc
    Mat :: F
    PetscErrorCode :: ierr

    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCLU, ierr))
    PetscCallA(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr))
    PetscCallA(PCFactorSetUpMatSolverType(pc, ierr))
    PetscCallA(PCFactorGetMatrix(pc, F, ierr))
    PetscCallA(MatMumpsSetIcntl(F, 7,  7,  ierr))   ! fill-reducing ordering (automatic)
    PetscCallA(MatMumpsSetIcntl(F, 14, 50, ierr))   ! workspace expansion %
    PetscCallA(MatMumpsSetIcntl(F, 8,  77, ierr))   ! numerical scaling (auto)
    PetscCallA(MatMumpsSetIcntl(F, 22, 0,  ierr))   ! in-core
  end subroutine petsc_setup_full_lu_pc

  !> Dispatch to the requested preconditioner setup routine.
  subroutine petsc_setup_pc(ksp, A, pc_type)
    KSP, intent(inout) :: ksp
    Mat, intent(in)    :: A
    integer, intent(in) :: pc_type

    select case (pc_type)
      case (PETSC_PC_TOROIDAL_HARMONIC)
        call petsc_setup_toroidal_harmonic_pc(ksp, A)
      case (PETSC_PC_PHYSICS)
        call petsc_setup_physics_pc(ksp, A)
      case (PETSC_PC_FULL_LU)
        call petsc_setup_full_lu_pc(ksp)
    end select
  end subroutine petsc_setup_pc
#endif
end module mod_petsc_pc
