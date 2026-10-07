module mod_petsc_pc_physics_ctx
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  public

  ! This module holds the shared context type (type_physics_pc_ctx) and the
  ! module-level singleton instance (g_ctx) used by the physics PC family of
  ! modules.  All other modules in the family `use` this one to access the
  ! type definition and the singleton without circular dependencies.

  type :: type_physics_pc_ctx
    logical :: initialized    = .false.
    logical :: reduced_ready  = .false.

    !> Index sets for each variable in the full system vector (1=psi..6=T)
    IS :: is_var(6)
    logical :: is_created = .false.

    !> Sub-blocks extracted from the full system matrix (the global SF path).
    !! Naming: B_ij = block at (equation i, variable j), variables
    !! 1..6 = (psi, u, j, w, rho, T).
    Mat :: B_11, B_12, B_13, B_16
    Mat :: B_21, B_22, B_23, B_24, B_25, B_26
    Mat :: B_31, B_33
    Mat :: B_42, B_44
    Mat :: B_51, B_52, B_55
    Mat :: B_61, B_62, B_63, B_66

    !> The assembled pairs of the global SF path (MPIAIJ, rebuilt every PC rebuild)
    Mat :: K_pj_aij          !< pair_psi = [[B_11, B_13], [B_31, B_33]]
    Mat :: S_W_aij           !< pair_w

    !> The momentum-Schur force operator W, assembled directly from the
    !! analytic composition of the two off-diagonal couplings (Chacon JCP 526
    !! (2025) Eqs. 17-19), so that S_uu = B_22 + W. One variable (u), block
    !! size n_tor. See models/model199/mod_pc_elt_matrix_force_fft.f90.
    Mat :: W_force
    logical :: w_force_ready = .false.
  end type type_physics_pc_ctx

  type(type_physics_pc_ctx), save :: g_ctx

  ! PetscLogEvents for the physics PC: the rebuild/apply split of its wall
  ! time. Read them with -log_view; the names share the "PhysPC_" prefix.
  PetscLogEvent, save :: pcev_elem_asm   = -1  !< the force-operator element assembly
  PetscLogEvent, save :: pcev_extract    = -1  !< block extraction / family operator fill
  PetscLogEvent, save :: pcev_build_suu  = -1  !< pair_w assembly
  PetscLogEvent, save :: pcev_convert    = -1  !< MatConvert to the solvers' format
  PetscLogEvent, save :: pcev_fact_pj    = -1  !< pair_psi solver setup
  PetscLogEvent, save :: pcev_fact_w     = -1  !< pair_w solver setup
  PetscLogEvent, save :: pcev_fact_rhot  = -1  !< rho and T solver setup
  PetscLogEvent, save :: pcev_apply      = -1  !< one whole PC apply
  PetscLogEvent, save :: pcev_solve_pj   = -1  !< pair_psi solves
  PetscLogEvent, save :: pcev_solve_w    = -1  !< pair_w solves
  PetscLogEvent, save :: pcev_solve_rhot = -1  !< rho/T solves
  PetscLogEvent, save :: pcev_shellmult  = -1  !< one matrix-free pair_w matvec (suu = schur)
  PetscLogEvent, save :: pcev_mjsolve    = -1  !< the B_33^-1 solve inside it

  logical, save, private :: pcev_registered = .false.

contains

  !--------------------------------------------------------------------
  !> Register the physics-PC log events. Idempotent: safe to call from any
  !! entry point without the caller having to know whether it ran already.
  !! Registering twice would give two distinct events with the same name and
  !! split the timings between them, which is why the guard is here rather
  !! than at the call site.
  !--------------------------------------------------------------------
  subroutine physics_pc_log_events_register()
    PetscErrorCode :: ierr

    if (pcev_registered) return

    call PetscLogEventRegister("PhysPC_ElemAsm",   0, pcev_elem_asm,   ierr)
    call PetscLogEventRegister("PhysPC_Extract",   0, pcev_extract,    ierr)
    call PetscLogEventRegister("PhysPC_BuildSuu",  0, pcev_build_suu,  ierr)
    call PetscLogEventRegister("PhysPC_Convert",   0, pcev_convert,    ierr)
    call PetscLogEventRegister("PhysPC_FactPJ",    0, pcev_fact_pj,    ierr)
    call PetscLogEventRegister("PhysPC_FactW",     0, pcev_fact_w,     ierr)
    call PetscLogEventRegister("PhysPC_FactRhoT",  0, pcev_fact_rhot,  ierr)
    call PetscLogEventRegister("PhysPC_Apply",     0, pcev_apply,      ierr)
    call PetscLogEventRegister("PhysPC_SolvePJ",   0, pcev_solve_pj,   ierr)
    call PetscLogEventRegister("PhysPC_SolveW",    0, pcev_solve_w,    ierr)
    call PetscLogEventRegister("PhysPC_SolveRhoT", 0, pcev_solve_rhot, ierr)
    call PetscLogEventRegister("PhysPC_ShellMult", 0, pcev_shellmult,  ierr)
    call PetscLogEventRegister("PhysPC_MjSolve",   0, pcev_mjsolve,    ierr)

    pcev_registered = .true.
  end subroutine physics_pc_log_events_register

  !> Peak resident set size of the process so far, in bytes (getrusage).
  !! ru_maxrss is the fifth long of struct rusage (after two 16-byte timevals)
  !! and is in bytes on macOS but in KiB on Linux. gfortran's preprocessor
  !! defines no OS macro, so the unit is decided at run time: a peak below the
  !! current RSS can only be KiB.
  real*8 function physics_pc_peak_rss()
    use iso_c_binding, only: c_int, c_long
    interface
      integer(c_int) function c_getrusage(who, buf) bind(C, name="getrusage")
        import :: c_int, c_long
        integer(c_int), value :: who
        integer(c_long)       :: buf(18)
      end function c_getrusage
    end interface
    integer(c_long) :: buf(18)
    integer(c_int)  :: rc
    PetscLogDouble  :: cur
    PetscErrorCode  :: ierr
    buf = 0
    rc  = c_getrusage(0_c_int, buf)
    call PetscMemoryGetCurrentUsage(cur, ierr)
    physics_pc_peak_rss = dble(buf(5))
    if (physics_pc_peak_rss < 0.5d0 * cur) physics_pc_peak_rss = physics_pc_peak_rss * 1024.d0
  end function physics_pc_peak_rss

  !> Memory audit checkpoint, in GB: the live PETSc heap and its high-water
  !! mark (tracked only under -log_view_memory or -malloc_debug, else 0), and
  !! the process peak RSS. On macOS the RSS is load-dependent (the compressor
  !! evicts pages under pressure), so the heap figures are the attribution;
  !! MUMPS-internal memory is not on the PETSc heap (see physics_pc_mumps_mem).
  subroutine physics_pc_mem(tag, my_id)
    character(len=*), intent(in) :: tag
    integer, intent(in)          :: my_id
    PetscLogDouble :: hcur, hmax
    PetscErrorCode :: ierr
    call PetscMallocGetCurrentUsage(hcur, ierr)
    call PetscMallocGetMaximumUsage(hmax, ierr)
    if (my_id == 0) write(*,'(A,A,T52,A,F7.3,A,F7.3,A,F7.3,A)') "[Mem] ", tag, &
      "heap ", hcur / 1.d9, " GB, heap peak ", hmax / 1.d9, " GB, RSS peak ", &
      physics_pc_peak_rss() / 1.d9, " GB"
  end subroutine physics_pc_mem

  !> Memory audit: MUMPS memory and factor size of a factored PREONLY KSP.
  !! INFOG(22) = MB effectively used during factorisation, INFOG(29) = entries
  !! in the factors (negative means millions). Silent if the PC is not MUMPS.
  subroutine physics_pc_mumps_mem(ksp, label, my_id)
    KSP, intent(in)              :: ksp
    character(len=*), intent(in) :: label
    integer, intent(in)          :: my_id
    PC             :: pc
    Mat            :: F
    MatSolverType  :: stype
    PetscInt       :: mb, nent
    PetscErrorCode :: ierr
    real*8         :: ent
    call KSPGetPC(ksp, pc, ierr)
    call PCFactorGetMatSolverType(pc, stype, ierr)
    if (ierr /= 0 .or. trim(stype) /= "mumps") return
    call PCFactorGetMatrix(pc, F, ierr)
    call MatMumpsGetInfog(F, 22_4, mb, ierr)
    call MatMumpsGetInfog(F, 29_4, nent, ierr)
    ent = dble(nent)
    if (nent < 0) ent = -dble(nent) * 1.d6
    if (my_id == 0) write(*,'(A,A,T52,A,I7,A,ES10.3)') "[Mem] MUMPS ", label, &
      "MB ", mb, ", factor entries ", ent
  end subroutine physics_pc_mumps_mem

#endif
end module mod_petsc_pc_physics_ctx
