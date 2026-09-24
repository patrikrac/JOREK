module mod_petsc_pc_sf_solver
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: physics_pc_mumps_mem
  use mod_petsc_pc_blocks,      only: pc_print_block_setup
  implicit none
  private

  !--------------------------------------------------------------------
  !> Block -> solver abstraction for the production SFM2 path.
  !!
  !! Every diagonal block of the LDU sweep is solved through ONE type, so
  !! "which solver does this block use" is a value rather than a code path.
  !!
  !! Backends -- exactly two, by design: one production method and the exact
  !! reference it is gated against.
  !! --------
  !!   SF_GMG  FGMRES + PCSHELL on one V-cycle of the C1 geometric multigrid
  !!           (mod_petsc_pc_gmg), hierarchy instance gmg_inst. On a packed
  !!           pair (nfields = 2) the smoother's blocks hold both fields, so
  !!           the pair is smoothed collectively rather than split.
  !!   SF_LU   PREONLY + LU (MUMPS). The reference, and the only exact one.
  !!
  !! The GMG smoother of each block, the axis-ring extent and the Krylov
  !! budgets are compile-time constants below rather than namelist entries:
  !! each is the value a recorded measurement was taken at, and the
  !! production path does not offer them as knobs.
  !--------------------------------------------------------------------

  integer, parameter, public :: SF_LU  = 1
  integer, parameter, public :: SF_GMG = 2

  !--- GMG smoothers (codes of mod_petsc_pc_gmg) ----------------------------
  !> Collective radial-line block Jacobi: pair_w, rho, T. Workstream D S12:
  !! the winner on pair_w; workstream H: at least as good as the point and
  !! node smoothers on rho and T at 41x64 and tstep 0.1 / 1 / 10.
  integer, parameter, public :: SF_GMG_SMOOTHER_LINES = 5
  !> Zebra (red-black) radial-line block Gauss-Seidel on the split (psi, j)
  !! pair: Chacon's collective smoothing of the split system (JCP 526 (2025)
  !! S4.1) extended along lines, because on C1 Hermite elements a per-node
  !! block does not dominate the operator (node blocks: 10-21 cycles and
  !! capped; point Jacobi: diverges). Workstream H: half the V-cycles of
  !! SF_GMG_SMOOTHER_LINES on pair_psi (3.3 -> 2.0) at unchanged outer counts.
  integer, parameter, public :: SF_GMG_SMOOTHER_ZEBRA = 7
  integer, parameter, public :: SF_GMG_AXIS_RINGS = 3   !< rings folded into the axis block
  integer, parameter, public :: SF_GMG_NSMOOTH    = 0   !< 0 = the smoother's own default (4)
  !> Line smoothers across rank boundaries: each local radial-line segment is
  !! extended by this many nodes into the neighbouring ranks' rows
  !! (restricted additive Schwarz, Cai & Sarkis, SISC 21 (1999) 792). JOREK
  !! partitions ring by ring, so without it every rank boundary cuts every
  !! line. Workstream H2, 41x64, tstep 1, mean V-cycles per solve at np 8
  !! (~5 rings per rank): pair_psi 5.3-6.5 without overlap, 2.3-2.4 with 1,
  !! 2.0-2.2 with 2 or 3 (np 1: 2.0-2.2); pair_w 3.1-3.4 -> 2.1. 2 is the
  !! smallest overlap that keeps the counts flat in np.
  integer, parameter, public :: SF_GMG_LINE_OVERLAP = 2
  !> Axis blocks solved over J-sector ranks (mod_petsc_pc_gmg_axis: an exact
  !! one-level nested dissection in J, gated against the LU on the first
  !! build); -1 = the cost model's sector count, 0 = the sequential LU on the
  !! ranks owning the block. OFF until the cluster decides. Workstream H2 on
  !! the laptop: exact (1e-12..1e-15 against the LU, identical counts) and it
  !! halves the axis LU time on the critical rank (np 4 / 8 at 21x64: 11.8 ->
  !! 6.5 s, 24.7 -> 11.0 s), but the extra synchronisation cancels that: wall
  !! 44 -> 48 s, 93 -> 106 s, 138 -> 143 s at 41x64 np 4. The LU's rank only
  !! becomes the bottleneck once the axis block no longer fits in one rank's
  !! rows (161x64 from np ~43), where every owner repeats the whole LU.
  integer, parameter, public :: SF_GMG_AXIS_SECTORS = 0
  !> FGMRES budget around a V-cycle, per block. These are not free parameters:
  !! they are the budgets the workstream D/G measurements were taken at
  !! (physics_pc_pair_maxits = 30 for the packed pairs, physics_pc_rhot_gmg =
  !! 10 for the scalar transport blocks), so the production path reproduces
  !! those runs rather than approximating them.
  integer, parameter, public :: SF_GMG_MAXITS      = 30  !< default: the packed pairs
  integer, parameter, public :: SF_GMG_MAXITS_RHOT = 10  !< the scalar rho / T blocks

  type, public :: block_solver_t
    KSP     :: ksp
    integer :: backend  = SF_LU
    Vec     :: dscale                      !< symmetric block scaling D, or unset
    logical :: scaled   = .false.
    logical :: created  = .false.
    integer :: gmg_inst = 0                !< hierarchy id when backend == SF_GMG
    integer :: its_sum  = 0, its_max = 0, nsolve = 0
    character(len=56) :: label = ""
  end type block_solver_t

  public :: sf_solver_setup, sf_solver_apply, sf_solver_destroy
  public :: sf_solver_reset_counters, sf_solver_report
  public :: sf_backend_name, sf_split_halves

contains

  !> Human-readable backend name, for the one setup line each block prints.
  function sf_backend_name(backend) result(s)
    integer, intent(in) :: backend
    character(len=64)   :: s
    select case (backend)
    case (SF_LU);  s = "PREONLY + LU (MUMPS)"
    case (SF_GMG); s = "FGMRES + SHELL[C1 GMG V-cycle]"
    case default;  s = "UNKNOWN"
    end select
  end function sf_backend_name

  !--------------------------------------------------------------------
  !> Configure slv to solve A. Idempotent across PC rebuilds: the KSP is
  !! created once and re-pointed at the (refilled, pattern-frozen) operator,
  !! so MUMPS and the GMG both reuse their symbolic phases.
  !--------------------------------------------------------------------
  subroutine sf_solver_setup(slv, A, backend, label, comm, my_id, rtol, gmg_inst, &
                             nfields, smoother, maxits)
    use mod_petsc_pc_gmg, only: gmg_select, gmg_is_ready, gmg_build_prolongations, &
                                gmg_setup_operator, gmg_pc_apply_1, gmg_pc_apply_2, &
                                gmg_pc_apply_3, gmg_pc_apply_4, gmg_opts_t
    type(block_solver_t), intent(inout) :: slv
    Mat, intent(in)              :: A
    integer, intent(in)          :: backend, comm, my_id
    character(len=*), intent(in) :: label
    real*8, intent(in)           :: rtol
    integer, intent(in)          :: gmg_inst  !< SF_GMG: hierarchy instance (1..4)
    integer, intent(in)          :: nfields   !< SF_GMG: fields packed per node in
                                              !< A -- 2 for the mixed pairs, 1 for a
                                              !< scalar block
    integer, intent(in)          :: smoother  !< SF_GMG: SF_GMG_SMOOTHER_*
    integer, intent(in)          :: maxits    !< SF_GMG: FGMRES budget

    PC :: pc
    PetscErrorCode :: ierr
    logical :: ok, fresh
    character(len=24) :: tstr
    type(gmg_opts_t) :: o

    fresh        = .not. slv%created
    slv%backend  = backend
    slv%label    = label
    slv%gmg_inst = gmg_inst

    select case (backend)

    case (SF_LU)
      if (fresh) call KSPCreate(comm, slv%ksp, ierr)
      call KSPSetOperators(slv%ksp, A, A, ierr)
      call KSPSetType(slv%ksp, KSPPREONLY, ierr)
      call KSPGetPC(slv%ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
      call KSPSetUp(slv%ksp, ierr)
      call pc_print_block_setup(comm, label, trim(sf_backend_name(backend)))
      if (fresh) call physics_pc_mumps_mem(slv%ksp, label, my_id)

    case (SF_GMG)
      !--- the complete hierarchy configuration, from this module's constants
      !--- only: no physics_pc_gmg_* namelist entry reaches this path.
      o%smoother     = smoother
      o%nsmooth      = SF_GMG_NSMOOTH
      o%axis_rings   = SF_GMG_AXIS_RINGS
      o%line_overlap = SF_GMG_LINE_OVERLAP
      o%axis_sectors = SF_GMG_AXIS_SECTORS
      o%bnd_drop     = 1               ! Dirichlet DOFs out of the coarse spaces
      o%harm_split   = 1               ! the extracted blocks are |n|-diagonal
      o%axis_mult    = 0;  o%axis_split = 0;  o%smooth_op = 0;  o%ring_diag = 0
      o%omega        = 0.7d0;  o%axis_droptol = 0.d0;  o%ring_aspect = 1.d0

      !--- the hierarchy: built once per instance, then refilled per rebuild
      call gmg_select(slv%gmg_inst)
      if (.not. gmg_is_ready()) then
        call gmg_build_prolongations(A, comm, my_id, nfields, ok, opts=o)
        if (.not. ok) then
          if (my_id == 0) write(*,'(A,A,A)') &
            "[Physics PC]   FATAL: the GMG backend for ", trim(label), &
            " needs the structured flux-surface grid."
          call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
      endif
      call gmg_setup_operator(A, comm, my_id, tag=trim(label), opts=o)
      call gmg_select(1)

      !--- the Krylov wrapper: created once, re-pointed at every rebuild
      if (fresh) then
        call KSPCreate(comm, slv%ksp, ierr)
        call KSPSetType(slv%ksp, KSPFGMRES, ierr)
        call KSPGMRESSetRestart(slv%ksp, max(maxits, 2), ierr)
        call KSPSetTolerances(slv%ksp, rtol, 1.d-50, 1.d6, maxits, ierr)
        call KSPGetPC(slv%ksp, pc, ierr)
        call PCSetType(pc, PCSHELL, ierr)
        select case (slv%gmg_inst)
        case (1); call PCShellSetApply(pc, gmg_pc_apply_1, ierr)
        case (2); call PCShellSetApply(pc, gmg_pc_apply_2, ierr)
        case (3); call PCShellSetApply(pc, gmg_pc_apply_3, ierr)
        case (4); call PCShellSetApply(pc, gmg_pc_apply_4, ierr)
        end select
        call PCShellSetName(pc, trim(label)//" C1 GMG V-cycle", ierr)
      endif
      call KSPSetOperators(slv%ksp, A, A, ierr)
      call KSPSetUp(slv%ksp, ierr)
      write(tstr,'(ES9.2)') rtol
      call pc_print_block_setup(comm, label, &
        trim(sf_backend_name(backend))//", rtol "//trim(adjustl(tstr)))

    case default
      if (my_id == 0) write(*,'(A,I0)') &
        "[Physics PC]   FATAL: sf_solver_setup got an unknown backend ", backend
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end select

    slv%created = .true.
  end subroutine sf_solver_setup

  !--------------------------------------------------------------------
  !> Solve A z = r through slv, applying the block scaling as an exact
  !! similarity and accumulating the iteration counts.
  !!
  !! rhs is scaled IN PLACE, which is what the research path's pair_ksp_solve
  !! does and is safe because the caller owns rhs and refills it every apply.
  !--------------------------------------------------------------------
  subroutine sf_solver_apply(slv, rhs, sol, ierr)
    type(block_solver_t), intent(inout) :: slv
    Vec, intent(in) :: rhs, sol
    PetscErrorCode, intent(inout) :: ierr
    PetscInt :: its

    if (slv%scaled) call VecPointwiseMult(rhs, rhs, slv%dscale, ierr)
    call KSPSolve(slv%ksp, rhs, sol, ierr)
    if (slv%scaled) call VecPointwiseMult(sol, sol, slv%dscale, ierr)

    call KSPGetIterationNumber(slv%ksp, its, ierr)
    slv%its_sum = slv%its_sum + int(its)
    slv%its_max = max(slv%its_max, int(its))
    slv%nsolve  = slv%nsolve + 1
  end subroutine sf_solver_apply

  subroutine sf_solver_reset_counters(slv)
    type(block_solver_t), intent(inout) :: slv
    slv%its_sum = 0; slv%its_max = 0; slv%nsolve = 0
  end subroutine sf_solver_reset_counters

  !> One line: mean and max inner iterations since the last reset. An exact
  !! backend reports 1/1 and is worth printing anyway -- it is how a silently
  !! mis-selected backend shows up.
  subroutine sf_solver_report(slv, my_id)
    type(block_solver_t), intent(in) :: slv
    integer, intent(in) :: my_id
    if (my_id /= 0 .or. slv%nsolve == 0) return
    write(*,'(A,A,A,F7.2,A,I0,A,I0,A)') "[Physics PC]   inner ", trim(slv%label), &
      ": mean ", dble(slv%its_sum) / dble(slv%nsolve), " its, max ", slv%its_max, &
      " (", slv%nsolve, " solves)"
  end subroutine sf_solver_report

  subroutine sf_solver_destroy(slv)
    type(block_solver_t), intent(inout) :: slv
    PetscErrorCode :: ierr
    if (.not. slv%created) return
    call KSPDestroy(slv%ksp, ierr)
    if (slv%scaled) call VecDestroy(slv%dscale, ierr)
    slv%created = .false.; slv%scaled = .false.
  end subroutine sf_solver_destroy

  !--------------------------------------------------------------------
  !> Move between a packed pair vector and its two field halves. The pack is
  !! rank-contiguous ([field-1 local | field-2 local]), so this is a local
  !! copy with no communication -- the reason pack_pair_aij exists.
  !--------------------------------------------------------------------
  subroutine sf_split_halves(p, h1, h2, to_p)
    Vec :: p, h1, h2
    logical, intent(in) :: to_p
    PetscScalar, pointer :: pa(:), a1(:), a2(:)
    PetscErrorCode :: ierr
    PetscInt :: n1, n2

    call VecGetLocalSize(h1, n1, ierr)
    call VecGetLocalSize(h2, n2, ierr)
    if (to_p) then
      call VecGetArray(p, pa, ierr)
      call VecGetArrayRead(h1, a1, ierr)
      call VecGetArrayRead(h2, a2, ierr)
      pa(1:n1)           = a1(1:n1)
      pa(n1 + 1:n1 + n2) = a2(1:n2)
      call VecRestoreArrayRead(h1, a1, ierr)
      call VecRestoreArrayRead(h2, a2, ierr)
      call VecRestoreArray(p, pa, ierr)
    else
      call VecGetArrayRead(p, pa, ierr)
      call VecGetArray(h1, a1, ierr)
      call VecGetArray(h2, a2, ierr)
      a1(1:n1) = pa(1:n1)
      a2(1:n2) = pa(n1 + 1:n1 + n2)
      call VecRestoreArray(h1, a1, ierr)
      call VecRestoreArray(h2, a2, ierr)
      call VecRestoreArrayRead(p, pa, ierr)
    endif
  end subroutine sf_split_halves

#endif
end module mod_petsc_pc_sf_solver
