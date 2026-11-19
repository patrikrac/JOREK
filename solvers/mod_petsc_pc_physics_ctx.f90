!> The physics PC's shared state (g_ctx), the method selection, its log
!! events and memory audit.
module mod_petsc_pc_physics_ctx
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  public

  type :: type_physics_pc_ctx
    logical :: reduced_ready = .false.
    !> The curvature part of the composed force operator, one variable (u),
    !! block size n_tor (models/model*/mod_pc_elt_matrix_force_fft.f90).
    Mat :: W_force
    logical :: w_force_ready = .false.
    !> v_par's sound-wave Schur terms (model600, stages b0 / b1): W_vv on
    !! B_77, W_vu (v_par row, u column) on B_72, W_uv (u row, v_par column) on B_27.
    Mat :: W_vv, W_vu, W_uv
    logical :: w_v_ready = .false.
    !> The vorticity-space Schur PC's two one-variable matrices, block size
    !! n_tor: the inertia opz (rho0 R^3 v, w) with Dirichlet rows at u's
    !! constrained DOFs, and the diagonal of those DOFs (built once).
    Mat :: M_rho, D_bc
    logical :: m_rho_ready = .false., d_bc_ready = .false.
  end type type_physics_pc_ctx

  type(type_physics_pc_ctx), save :: g_ctx

  !> physics_pc_method: the split-field sweep, or the vorticity-space Schur
  !! PC with the ideal or the resistive line bending, or the ideal one as a
  !! lumped scalar operator (an experiment), or the Alfven-block triangular
  !! PC (mod_petsc_pc_vs_alfven)
  integer, parameter :: PC_METHOD_SF = 1, PC_METHOD_VS_IDEAL = 2, PC_METHOD_VS_RESISTIVE = 3, &
                        PC_METHOD_VS_LUMPED = 4, PC_METHOD_VS_ALFVEN = 5

  !> The element-level matrices a method builds on (petsc_assemble_pc_matrices).
  type :: physics_pc_elements_t
    !> W_force: the SF sweep (pair_w) and vs_alfven (K_W)
    logical :: w      = .false.
    !> v_par's W_vv, W_vu, W_uv: the SF sweep and vs_alfven, and there only
    !! where the v_par stage needs them (sf_vpar_w_assembled)
    logical :: w_vpar = .false.
    !> M_rho, and D_bc once: vs_ideal, vs_resistive, vs_lumped
    logical :: m_rho  = .false.
  end type physics_pc_elements_t

  ! -log_view events of the physics PC ("PhysPC_*")
  PetscLogEvent, save :: pcev_elem_asm   = -1  !< the element assembly (W, or M_rho and D_bc)
  PetscLogEvent, save :: pcev_extract    = -1  !< operator extraction / refill
  PetscLogEvent, save :: pcev_fact_pj    = -1  !< pair_psi solver setup
  PetscLogEvent, save :: pcev_fact_w     = -1  !< pair_w solver setup (also K_W of vs_alfven)
  PetscLogEvent, save :: pcev_fact_rhot  = -1  !< rho and T solver setup
  PetscLogEvent, save :: pcev_apply      = -1  !< one whole PC apply
  PetscLogEvent, save :: pcev_solve_pj   = -1  !< pair_psi solves
  PetscLogEvent, save :: pcev_solve_w    = -1  !< pair_w solves (also K_W of vs_alfven)
  PetscLogEvent, save :: pcev_solve_rhot = -1  !< rho/T solves
  PetscLogEvent, save :: pcev_fact_vort  = -1  !< vorticity-system solver setup
  PetscLogEvent, save :: pcev_fact_lap   = -1  !< Laplacian and mass solver setup (once)
  PetscLogEvent, save :: pcev_solve_vort = -1  !< vorticity-system solves
  PetscLogEvent, save :: pcev_solve_lap  = -1  !< Laplacian solves
  PetscLogEvent, save :: pcev_solve_mass = -1  !< mass solves of j and omega

  logical, save, private :: pcev_registered = .false.

contains

  !> physics_pc_method as PC_METHOD_*; an unknown name stops the run.
  integer function physics_pc_method_id()
    use phys_module, only: physics_pc_method
    integer :: me, mpierr
    select case (trim(adjustl(physics_pc_method)))
    case ("sf");           physics_pc_method_id = PC_METHOD_SF
    case ("vs_ideal");     physics_pc_method_id = PC_METHOD_VS_IDEAL
    case ("vs_resistive"); physics_pc_method_id = PC_METHOD_VS_RESISTIVE
    case ("vs_lumped");    physics_pc_method_id = PC_METHOD_VS_LUMPED
    case ("vs_alfven");    physics_pc_method_id = PC_METHOD_VS_ALFVEN
    case default
      physics_pc_method_id = 0
      call MPI_Comm_rank(MPI_COMM_WORLD, me, mpierr)
      if (me == 0) write(*,'(A)') "[Physics PC]   FATAL: physics_pc_method must be sf | vs_ideal | "// &
        "vs_resistive | vs_lumped | vs_alfven, got '"//trim(physics_pc_method)//"'"
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    end select
  end function physics_pc_method_id

  !> The method's name for the solver report.
  function physics_pc_method_label() result(s)
    character(len=16) :: s
    select case (physics_pc_method_id())
    case (PC_METHOD_SF);           s = "SF"
    case (PC_METHOD_VS_IDEAL);     s = "VS, ideal"
    case (PC_METHOD_VS_RESISTIVE); s = "VS, resistive"
    case (PC_METHOD_VS_LUMPED);    s = "VS, lumped"
    case (PC_METHOD_VS_ALFVEN);    s = "Alfven block"
    case default;                  s = "unknown"
    end select
  end function physics_pc_method_label

  !> The element-level matrices of the method: W, and v_par's W terms, for
  !! the SF sweep and the Alfven block, M_rho and D_bc for the
  !! vorticity-space Schur variants.
  function physics_pc_elements() result(e)
    type(physics_pc_elements_t) :: e
    select case (physics_pc_method_id())
    case (PC_METHOD_SF, PC_METHOD_VS_ALFVEN)
      e%w = .true.; e%w_vpar = .true.
    case (PC_METHOD_VS_IDEAL, PC_METHOD_VS_RESISTIVE, PC_METHOD_VS_LUMPED)
      e%m_rho = .true.
    end select
  end function physics_pc_elements

  !> Register the physics-PC log events, once (a second registration would
  !! split the timings between two events of the same name).
  subroutine physics_pc_log_events_register()
    PetscErrorCode :: ierr

    if (pcev_registered) return

    call PetscLogEventRegister("PhysPC_ElemAsm",   0, pcev_elem_asm,   ierr)
    call PetscLogEventRegister("PhysPC_Extract",   0, pcev_extract,    ierr)
    call PetscLogEventRegister("PhysPC_FactPJ",    0, pcev_fact_pj,    ierr)
    call PetscLogEventRegister("PhysPC_FactW",     0, pcev_fact_w,     ierr)
    call PetscLogEventRegister("PhysPC_FactRhoT",  0, pcev_fact_rhot,  ierr)
    call PetscLogEventRegister("PhysPC_Apply",     0, pcev_apply,      ierr)
    call PetscLogEventRegister("PhysPC_SolvePJ",   0, pcev_solve_pj,   ierr)
    call PetscLogEventRegister("PhysPC_SolveW",    0, pcev_solve_w,    ierr)
    call PetscLogEventRegister("PhysPC_SolveRhoT", 0, pcev_solve_rhot, ierr)
    call PetscLogEventRegister("PhysPC_FactVort",  0, pcev_fact_vort,  ierr)
    call PetscLogEventRegister("PhysPC_FactLap",   0, pcev_fact_lap,   ierr)
    call PetscLogEventRegister("PhysPC_SolveVort", 0, pcev_solve_vort, ierr)
    call PetscLogEventRegister("PhysPC_SolveLap",  0, pcev_solve_lap,  ierr)
    call PetscLogEventRegister("PhysPC_SolveMass", 0, pcev_solve_mass, ierr)

    pcev_registered = .true.
  end subroutine physics_pc_log_events_register

  !> Peak resident set size of the process so far, in bytes: ru_maxrss of
  !! getrusage, the fifth long of struct rusage, in KiB on Linux and bytes on
  !! macOS; a peak below the current RSS can only be KiB.
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

  !> Memory checkpoint in GB: the PETSc heap and its peak (tracked only with
  !! -log_view_memory or -malloc_debug) and the process peak RSS.
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

  !> MUMPS memory (INFOG(22), MB) and factor entries (INFOG(29), negative =
  !! millions) of a factored KSP; silent if its PC is not MUMPS.
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
