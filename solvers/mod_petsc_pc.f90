module mod_petsc_pc
#ifdef USE_PETSC
  use mod_petsc_pc_toroidal
  use mod_petsc_pc_physics
  use mod_petsc_pc_modesplit
  use mod_petsc_direct_solver, only: petsc_setup_wrapped_solver
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  public

  integer, parameter :: PETSC_PC_TOROIDAL_HARMONIC = 1
  integer, parameter :: PETSC_PC_PHYSICS           = 2
  !> Same block-diagonal-in-mode-families operator as PETSC_PC_TOROIDAL_HARMONIC,
  !! but with each family on its own disjoint sub-communicator so the blocks are
  !! factorized and solved concurrently. Selected with -jorek_pc_mode_split.
  integer, parameter :: PETSC_PC_MODE_SPLIT        = 3

contains

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
      case (PETSC_PC_MODE_SPLIT)
        call petsc_setup_modesplit_pc(ksp, A)
    end select
  end subroutine petsc_setup_pc


  !> Finish setting up the block solvers, after KSPSetUpOnBlocks has done what it can.
  !!
  !! KSPSetUpOnBlocks stops at any PCTELESCOPE, which implements no setuponblocks,
  !! leaving the factorization it wraps to be triggered by the first PCApply inside
  !! the GMRES solve. Walking the blocks here keeps that cost in the setup stage
  !! where the rest of the factorization work is already accounted.
  !!
  !! Independent of which preconditioner petsc_setup_pc selected, and a no-op when
  !! no telescope is in use.
  subroutine petsc_setup_pc_blocks(ksp)
    KSP, intent(inout) :: ksp

    PC       :: pc, subpc
    PCType   :: ptype
    integer  :: i
    PetscInt :: n_split
    KSP, pointer, dimension(:) :: subksp_array
    PetscErrorCode :: ierr

    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCGetType(pc, ptype, ierr))

    ! The mode-split PC factorizes its blocks inside its own PCSetUp callback, on
    ! the sub-communicator that owns them; there is nothing here to reach into.
    if (ptype == PCSHELL) return

    if (ptype /= PCFIELDSPLIT) then
      call petsc_setup_wrapped_solver(pc)
      return
    endif

    PetscCallA(PCFieldSplitGetSubKSP(pc, n_split, subksp_array, ierr))
    do i = 1, n_split
      PetscCallA(KSPGetPC(subksp_array(i), subpc, ierr))
      call petsc_setup_wrapped_solver(subpc)
    enddo
    PetscCallA(PCFieldSplitRestoreSubKSP(pc, n_split, subksp_array, ierr))
  end subroutine petsc_setup_pc_blocks
#endif
end module mod_petsc_pc
