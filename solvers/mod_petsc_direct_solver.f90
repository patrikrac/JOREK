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
  public :: petsc_configure_direct_solver, petsc_solver_option, petsc_setup_wrapped_solver

  !> Suffix PCTELESCOPE appends to its own options prefix for the PC it wraps
  !! (src/ksp/pc/impls/telescope/telescope.c, KSPAppendOptionsPrefix).
  character(len=*), parameter :: TELESCOPE_SUFFIX = 'telescope_'

  !> How many nested telescopes to walk through. Bounded so that a malformed
  !! options file cannot drive the walk indefinitely.
  integer, parameter :: MAX_TELESCOPE_DEPTH = 4

  !> Value PETSc default-initializes a Fortran object handle to, and writes back
  !! when the C routine it wraps yields no object (PETSC_FORTRAN_TYPE_INITIALIZE
  !! in include/petsc/private/ftnimpl.h). PETSc exposes no predicate for an absent
  !! Fortran object, so comparing against it is the only way to distinguish "this
  !! rank was not reduced onto" from a KSP that really is there.
  PetscFortranAddr, parameter :: ABSENT_HANDLE = -2

contains

  !> Configure `pc` as a direct solver.
  !!
  !! Order matters: JOREK's defaults go in first, PCSetFromOptions then lets the
  !! user override them, and only afterwards do we ask what we actually ended up
  !! with. Calling a package-specific routine before that query would abort as
  !! soon as anybody selected a different factor package.
  subroutine petsc_configure_direct_solver(pc)
    PC, intent(inout) :: pc

    PCType             :: ptype
    MatSolverType      :: stype
    character(len=256) :: prefix
    PetscErrorCode     :: ierr

    ! --- JOREK defaults, overridable below
    PetscCallA(PCSetType(pc, PCLU, ierr))
    PetscCallA(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr))
    PetscCallA(PCSetFromOptions(pc, ierr))

    PetscCallA(PCGetType(pc, ptype, ierr))
    PetscCallA(PCGetOptionsPrefix(pc, prefix, ierr))

    ! PCTELESCOPE is not itself a factorization: it reduces the operator onto a
    ! sub-communicator and wraps a second PC, and that inner PC is the one doing
    ! the factorizing. Handing it off rather than returning here is what keeps
    ! JOREK's package tuning (notably the MUMPS ICNTLs) from being silently
    ! dropped the moment somebody selects telescope.
    if (ptype == PCTELESCOPE) then
      call configure_wrapped_solver(trim(prefix)//TELESCOPE_SUFFIX)
      return
    endif

    ! Everything below is specific to a factorization preconditioner. The user is
    ! free to select something else entirely (-..._pc_type hypre, jacobi, ...), in
    ! which case there is no factor matrix to configure.
    if (ptype /= PCLU .and. ptype /= PCCHOLESKY .and. ptype /= PCILU) return

    PetscCallA(PCFactorGetMatSolverType(pc, stype, ierr))

    select case (trim(stype))
      case ('mumps')
        call tune_mumps(trim(prefix))
      case ('strumpack')
        call tune_strumpack(trim(prefix))
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


  !> Apply JOREK's defaults and package tuning to a PC that is addressed only by
  !! its options prefix, because no object exists to configure.
  !!
  !! PCSetUp_Telescope creates the KSP it wraps lazily and only on the ranks it
  !! reduced onto, during the KSPSetUp that happens after this module has had its
  !! say. There is therefore nothing to call PCSetType or PCFactorSetMatSolverType
  !! on. Everything instead goes into the options database, which is exactly where
  !! that inner KSP reads from once it is built -- the same mechanism tune_mumps
  !! already relies on, applied one level deeper.
  !!
  !! Seeding LU via MUMPS here means selecting telescope alone reproduces the
  !! solver JOREK would have used without it, rather than falling back to PETSc's
  !! own default PC for a reduced communicator.
  subroutine configure_wrapped_solver(prefix)
    character(len=*), intent(in) :: prefix

    character(len=256) :: pfx
    character(len=128) :: ptype, stype
    integer            :: depth

    pfx = prefix
    ! Telescopes can be nested; walk down to the PC that actually factorizes.
    do depth = 1, MAX_TELESCOPE_DEPTH
      call set_default_opt(trim(pfx), 'pc_type', PCLU)
      call petsc_solver_option(trim(pfx), 'pc_type', ptype)
      if (trim(ptype) /= PCTELESCOPE) exit
      pfx = trim(pfx)//TELESCOPE_SUFFIX
    enddo

    if (trim(ptype) /= PCLU .and. trim(ptype) /= PCCHOLESKY .and. trim(ptype) /= PCILU) return

    call set_default_opt(trim(pfx), 'pc_factor_mat_solver_type', MATSOLVERMUMPS)
    call petsc_solver_option(trim(pfx), 'pc_factor_mat_solver_type', stype)

    select case (trim(stype))
      case ('mumps')
        call tune_mumps(trim(pfx))
      case ('strumpack')
        call tune_strumpack(trim(pfx))
      case ('petsc')
        ! PCFactorSetMatOrderingType has no object to act on here. Seeding the
        ! option works in this direction and not in the direct one, because the
        ! wrapped PC has not run PCSetFromOptions yet whereas `pc` already has.
        call set_default_opt(trim(pfx), 'pc_factor_mat_ordering_type', MATORDERINGND)
      case default
        ! superlu_dist, cudss, ... : PETSc defaults, tunable through
        ! -<prefix>mat_superlu_dist_* / -<prefix>mat_cudss_* etc.
    end select
  end subroutine configure_wrapped_solver


  !> Run the factorizations that KSPSetUpOnBlocks cannot reach.
  !!
  !! PCFIELDSPLIT forwards KSPSetUpOnBlocks to each of its blocks, but PCTELESCOPE
  !! implements no setuponblocks of its own, so the chain stops at it and the
  !! factorization inside is left to the first PCApply -- charged, in other words,
  !! to the GMRES solve rather than to setup. Doing it here restores the accounting
  !! the untelescoped path already gets from the KSPSetUpOnBlocks call in mod_petsc.
  !!
  !! Call it after that KSPSetUpOnBlocks, on every solve rather than only the first:
  !! PCSetUp_Telescope calls KSPSetOperators on the PC it wraps each time it runs,
  !! which is what marks the factorization stale again after a matrix update.
  !!
  !! Safe on any PC -- anything that is not a telescope returns immediately.
  subroutine petsc_setup_wrapped_solver(pc)
    PC, intent(in) :: pc

    PC             :: cur
    KSP            :: inner
    PCType         :: ptype
    integer        :: depth
    PetscErrorCode :: ierr

    cur = pc
    do depth = 1, MAX_TELESCOPE_DEPTH
      PetscCallA(PCGetType(cur, ptype, ierr))
      if (ptype /= PCTELESCOPE) return

      ! `inner` is deliberately left at its default initialization. Assigning
      ! PETSC_NULL_KSP first would be read by CHKFORTRANNULLOBJECT in the generated
      ! stub as "the caller does not want this output", and the handle would come
      ! back unset on every rank, including the ones that do hold a KSP.
      PetscCallA(PCTelescopeGetKSP(cur, inner, ierr))
      if (inner%v == 0 .or. inner%v == ABSENT_HANDLE) return  ! not reduced onto this rank

      PetscCallA(KSPSetUp(inner, ierr))
      PetscCallA(KSPGetPC(inner, cur, ierr))
    enddo
  end subroutine petsc_setup_wrapped_solver


  !> MUMPS controls JOREK has been tuned for.
  !!
  !! These are seeded into the options database rather than applied with
  !! MatMumpsSetIcntl. PETSc rescans -<prefix>mat_mumps_icntl_* during the
  !! symbolic factorization and overwrites whatever was set programmatically with
  !! its own defaults, which silently reverted ICNTL(14) from 50 to 20. Going
  !! through the options database puts JOREK's values in the same place PETSc
  !! reads from, and set_default_opt leaves any value the user supplied alone.
  subroutine tune_mumps(prefix)
    character(len=*), intent(in) :: prefix
    call set_default_opt(prefix, 'mat_mumps_icntl_7',  '7')    ! fill-reducing ordering (METIS)
    call set_default_opt(prefix, 'mat_mumps_icntl_14', '50')   ! workspace expansion %
    call set_default_opt(prefix, 'mat_mumps_icntl_8',  '77')   ! numerical scaling (auto)
    call set_default_opt(prefix, 'mat_mumps_icntl_22', '0')    ! 0 = in-core factorization
  end subroutine tune_mumps


  !> STRUMPACK controls. The reordering mirrors the MUMPS choice (METIS).
  !! Compression is deliberately left at the PETSc default so STRUMPACK stays an
  !! exact factorization rather than an inexact preconditioner; enable it with
  !! -<prefix>mat_strumpack_compression if that is what you want.
  subroutine tune_strumpack(prefix)
    character(len=*), intent(in) :: prefix
    call set_default_opt(prefix, 'mat_strumpack_reordering', 'METIS')
  end subroutine tune_strumpack


  !> Seed one option for `prefix` unless the user already supplied it, so that
  !! JOREK's defaults never override what came from jorek.petsc.
  !!
  !! Going through the options database rather than the package-specific setters
  !! also means this module references no symbol from MUMPS, STRUMPACK or cuDSS,
  !! so it links against any PETSc build regardless of which packages are in it.
  subroutine set_default_opt(prefix, name, val)
    character(len=*), intent(in) :: prefix, name, val

    PetscBool      :: is_set
    PetscErrorCode :: ierr

    PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS, prefix, '-'//name, is_set, ierr))
    if (is_set) return
    PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, '-'//prefix//name, val, ierr))
  end subroutine set_default_opt


  !> Read one option for `prefix` back out of the options database, blank if unset.
  !! Public so that callers can report the solver a telescoped block resolved to,
  !! which cannot be queried from a PC that has not been created yet.
  subroutine petsc_solver_option(prefix, name, val)
    character(len=*), intent(in)  :: prefix, name
    character(len=*), intent(out) :: val

    PetscBool      :: is_set
    PetscErrorCode :: ierr

    val = ''
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, prefix, '-'//name, val, is_set, ierr))
    if (.not. is_set) val = ''
  end subroutine petsc_solver_option


  !> PETSc's own built-in factorization. Unlike the external packages, this one
  !! honours the PC-level ordering type.
  subroutine tune_petsc_builtin(pc)
    PC, intent(inout) :: pc
    PetscErrorCode    :: ierr
    PetscCallA(PCFactorSetMatOrderingType(pc, MATORDERINGND, ierr))
  end subroutine tune_petsc_builtin

#endif
end module mod_petsc_direct_solver
