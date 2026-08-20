!> Configuration of the PETSc direct (factorization) solvers used inside the
!! preconditioner blocks and on the direct-solve path.
!!
!! JOREK sets its own defaults programmatically (LU via MUMPS, plus the MUMPS
!! controls it has been tuned for) and then calls PCSetFromOptions, so every one
!! of those choices can be overridden at run time through the PETSc options
!! database without recompiling. See petsc_initialize in mod_petsc for how the
!! options file is picked up, and namelist/jorek.petsc.example for the syntax.
!!
!! The factor package is queried back with PCFactorGetMatSolverType *after* the
!! options have been applied, so package-specific tuning is only ever applied to
!! the package actually in use.
module mod_petsc_direct_solver
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_configure_direct_solver

contains

  !> Configure `pc` as a direct solver.
  !!
  !! Order matters: JOREK's defaults go in first, PCSetFromOptions then lets the
  !! user override them, and only afterwards do we ask what we actually ended up
  !! with. Calling a package-specific routine before that query would abort as
  !! soon as anybody selected a different factor package.
  subroutine petsc_configure_direct_solver(pc)
    PC, intent(inout) :: pc

    PCType         :: ptype
    MatSolverType  :: stype
    PetscErrorCode :: ierr

    ! --- JOREK defaults, overridable below
    PetscCallA(PCSetType(pc, PCLU, ierr))
    PetscCallA(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr))
    PetscCallA(PCSetFromOptions(pc, ierr))

    ! Everything below is specific to a factorization preconditioner. The user is
    ! free to select something else entirely (-..._pc_type hypre, jacobi, ...), in
    ! which case there is no factor matrix to configure.
    PetscCallA(PCGetType(pc, ptype, ierr))
    if (ptype /= PCLU .and. ptype /= PCCHOLESKY .and. ptype /= PCILU) return

    PetscCallA(PCFactorGetMatSolverType(pc, stype, ierr))

    select case (trim(stype))
      case ('mumps')
        call tune_mumps(pc)
      case ('strumpack')
        call tune_strumpack(pc)
      case ('petsc')
        call tune_petsc_builtin(pc)
      case default
        ! superlu_dist, cudss, ... : PETSc defaults, tunable through
        ! -<prefix>mat_superlu_dist_* / -<prefix>mat_cudss_* etc.
    end select

    ! Create the factor matrix so its numerical factorization is done during
    ! setup rather than inside the first solve.
    PetscCallA(PCFactorSetUpMatSolverType(pc, ierr))
  end subroutine petsc_configure_direct_solver


  !> MUMPS controls JOREK has been tuned for.
  !!
  !! These are seeded into the options database rather than applied with
  !! MatMumpsSetIcntl. PETSc rescans -<prefix>mat_mumps_icntl_* during the
  !! symbolic factorization and overwrites whatever was set programmatically with
  !! its own defaults, which silently reverted ICNTL(14) from 50 to 20. Going
  !! through the options database puts JOREK's values in the same place PETSc
  !! reads from, and set_default_opt leaves any value the user supplied alone.
  subroutine tune_mumps(pc)
    PC, intent(in) :: pc
    call set_default_opt(pc, 'mat_mumps_icntl_7',  '7')    ! fill-reducing ordering (METIS)
    call set_default_opt(pc, 'mat_mumps_icntl_14', '50')   ! workspace expansion %
    call set_default_opt(pc, 'mat_mumps_icntl_8',  '77')   ! numerical scaling (auto)
    call set_default_opt(pc, 'mat_mumps_icntl_22', '0')    ! 0 = in-core factorization
  end subroutine tune_mumps


  !> STRUMPACK controls. The reordering mirrors the MUMPS choice (METIS).
  !! Compression is deliberately left at the PETSc default so STRUMPACK stays an
  !! exact factorization rather than an inexact preconditioner; enable it with
  !! -<prefix>mat_strumpack_compression if that is what you want.
  subroutine tune_strumpack(pc)
    PC, intent(in) :: pc
    call set_default_opt(pc, 'mat_strumpack_reordering', 'METIS')
  end subroutine tune_strumpack


  !> Seed one option for `pc` unless the user already supplied it, so that
  !! JOREK's defaults never override what came from jorek.petsc.
  !!
  !! Going through the options database rather than the package-specific setters
  !! also means this module references no symbol from MUMPS, STRUMPACK or cuDSS,
  !! so it links against any PETSc build regardless of which packages are in it.
  subroutine set_default_opt(pc, name, val)
    PC,               intent(in) :: pc
    character(len=*), intent(in) :: name, val

    character(len=256) :: prefix
    PetscBool          :: is_set
    PetscErrorCode     :: ierr

    PetscCallA(PCGetOptionsPrefix(pc, prefix, ierr))
    PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS, trim(prefix), '-'//name, is_set, ierr))
    if (is_set) return
    PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, '-'//trim(prefix)//name, val, ierr))
  end subroutine set_default_opt


  !> PETSc's own built-in factorization. Unlike the external packages, this one
  !! honours the PC-level ordering type.
  subroutine tune_petsc_builtin(pc)
    PC, intent(inout) :: pc
    PetscErrorCode    :: ierr
    PetscCallA(PCFactorSetMatOrderingType(pc, MATORDERINGND, ierr))
  end subroutine tune_petsc_builtin

#endif
end module mod_petsc_direct_solver
