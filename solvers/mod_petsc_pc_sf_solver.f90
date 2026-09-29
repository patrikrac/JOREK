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
  !> Collective radial-line block Jacobi: rho, T. Workstream H: at least as
  !! good as the point and node smoothers on rho and T at 41x64 and tstep
  !! 0.1 / 1 / 10.
  integer, parameter, public :: SF_GMG_SMOOTHER_LINES = 5
  !> Zebra (red-black) radial-line block Gauss-Seidel on the split (psi, j)
  !! pair: Chacon's collective smoothing of the split system (JCP 526 (2025)
  !! S4.1) extended along lines, because on C1 Hermite elements a per-node
  !! block does not dominate the operator (node blocks: 10-21 cycles and
  !! capped; point Jacobi: diverges). Workstream H: half the V-cycles of
  !! SF_GMG_SMOOTHER_LINES on pair_psi (3.3 -> 2.0) at unchanged outer counts.
  !! On pair_w (exact-mass S_uu, blocks from the diagonal-mass channel) flat
  !! at ~2.0 V-cycles from 81x32 to 121x48 where lines take 3.1-3.4.
  integer, parameter, public :: SF_GMG_SMOOTHER_ZEBRA = 7
  !> Zebra over rings between the axis block and the switch ring I_s (median
  !! r*dtheta/dr = 1, about n_tht/(2 pi) rings), zebra radial lines outside:
  !! GMGPolar's circle/radial split (Bourne et al., JCP 488 (2023)) with its
  !! colouring. The fixed axis block (rings 0..3) leaves rings 4..I_s-1 to
  !! radial lines, the weak direction there; solving rings 0..I_s-1 exactly
  !! instead (axis_rings = -1) bounds the gain: 321x128 w arm np 64, pair_w
  !! 3.17 -> 2.16 V-cycles. Smoother 8 reaches 2.23 at O(N) cost (solve 89.1
  !! -> 78.3 s, setup unchanged); pair_psi 1.58 -> 1.49 (its bound 1.49).
  !! At 161x64 (I_s = 10) it is neutral (pair_w 2.27 -> 2.21, bound 2.27).
  !! schur arm's pair_w, 321x128 np 64, 5 runs each: neutral within the
  !! run-to-run spread (outer its 91-115 vs 93-109, mean solve 216 vs 230 s;
  !! pair_w hits its 30-iteration cap at tstep 1, which makes that arm's
  !! outer count vary by +-10% between identical runs). Kept for uniformity.
  !! Not on rho / T (line Jacobi there; smoother 8: T 2.0 -> 8.8 V-cycles).
  integer, parameter, public :: SF_GMG_SMOOTHER_ZEBRA_RINGS = 8
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
  !! On the cluster the ghosts cost more than the counts save: a rank always
  !! holds n_tht lines, so only their length shrinks with np, and at 2-5
  !! rings per rank overlap 2 adds 150% / 60% ghost rows. 2026-09-28
  !! (sf_runs/sf_iter, np 64 x 8 on 2 nodes, coarse Richardson on): w arm
  !! 161x64 solve 28.1 -> 23.6 s, 321x128 107.2 -> 85.0 s, outer its +1-2%;
  !! overlap 0 does not converge (rho / T 5-10 V-cycles). The schur arm's
  !! pair_w (Dh-channel level 0, couplings to J +- 2) keeps 2.
  integer, parameter, public :: SF_GMG_LINE_OVERLAP = 1
  integer, parameter, public :: SF_GMG_LINE_OVERLAP_SCHUR_W = 2
  !> Axis blocks solved over J-sector ranks (mod_petsc_pc_gmg_axis: an exact
  !! one-level nested dissection in J, gated against the LU on the first
  !! build); -1 = the cost model's sector count, 0 = the sequential LU on the
  !! ranks owning the block. ON: the first cluster runs of the scaling kit
  !! (2026-09-24) call for it, since there the sequential axis LU on rank 0
  !! is the strong-scaling critical path. On the laptop it does NOT pay, and
  !! those numbers are no argument against it: exact (1e-12..1e-15 against
  !! the LU, identical counts), it halves the axis LU time on the critical
  !! rank (np 4 / 8 at 21x64: 11.8 -> 6.5 s, 24.7 -> 11.0 s), but the extra
  !! synchronisation cancels that there: wall 44 -> 48 s, 93 -> 106 s,
  !! 138 -> 143 s at 41x64 np 4. The first build's exactness gate still keeps
  !! the LU for any level whose sector solve disagrees with it.
  integer, parameter, public :: SF_GMG_AXIS_SECTORS = -1
  !> ... except the schur arm's pair_w: its level-0 axis block carries the Dh
  !! channel's J +- 2 couplings, so half its rows become separator / border
  !! and the reduced system is ~3.3k rows (161x64) -- more work than the
  !! sequential LU it replaces. 161x64 np 64 x 8 (2026-09-28): setup 17.8 ->
  !! 7.2 s, solve 17.8 -> 14.1 s with the sequential LU, outer its 45 -> 44.
  integer, parameter, public :: SF_GMG_AXIS_SECTORS_SCHUR_W = 0
  !> Matvec kernel of the SF operators, the GMG levels and the SFM2 coupling
  !! blocks: 0 = PETSc's (one thread per rank; AIJMKL where PETSc has MKL
  !! sparse), 1 = jorek_blockmv_attach.c (OpenMP, reads each n_tor harmonic block's
  !! column indices once; exact, gated against MatMult on first attach).
  integer, parameter, public :: SF_BLOCKMV = 1
  !--- the two pair_w operators (physics_pc_sf_suu) -------------------------
  !> "w": S_uu = B_22 + W, assembled; pair_w's GMG and Krylov both on it.
  !> "schur": S_uu = the psi-channel Schur complement, matrix-free
  !! (mod_petsc_pc_sf_pairw); its multigrid on B_22 + W with a Dh-channel
  !! fine level.
  integer, parameter, public :: SF_SUU_W     = 1
  integer, parameter, public :: SF_SUU_SCHUR = 2

  !--- V-cycle shapes (gmg_opts_t pre0 / post0 / nsmooth_c) -----------------
  !> pair_psi: no level-0 pre-smoothing, 12 (schur) / 10 (w) post-smoothing
  !! steps, 2 steps per side on the coarse levels. From a psi-row
  !! right-hand side (the corrector's [B_12 u; 0], every nested solve of the
  !! schur shell) the coarse correction leaves a large j-row residual at the
  !! edge (20-60x ||b|| after one V(4,4), rings 113-118 of 121x48) that only
  !! level-0 POST-smoothing removes: W/F-cycles, a wider line overlap and
  !! pre-smoothing (V(6,0): still 23-63x) leave it. Measured at 121x48,
  !! np 4, shaped pcbench, tstep 1, 3 steps:
  !!   schur: V(4,4) 2-3 cycles per solve (the first one stalls), 93.7 s;
  !!          V(0,12) one cycle in every solve, 69.9 s -- which the nested
  !!          solve's single fixed cycle needs; V(0,8) 2 cycles, V(0,10)
  !!          1.05, V(0,16) / V(2,12) / V(12,12) one at more cost.
  !!   w:     V(4,4) 1.48 cycles, 99.4 s -> V(0,10) 1.00, 94.4 s; V(0,8)
  !!          1.16 at the same time.
  !! Coarse levels at 2 steps instead of 4: same counts, ~2 s less.
  integer, parameter, public :: SF_PJ_PRE0 = 0, SF_PJ_NSC = 2
  integer, parameter, public :: SF_PJ_POST0_SCHUR = 12, SF_PJ_POST0_W = 10
  !> pair_w on both arms: the zebra smoother, V(0,6). schur: 4.8 -> 4.05
  !! pair_w its per solve against V(4,4) (121x48 np 4, V-cycle time 20.8 ->
  !! 13.0 s); V(0,8) the same counts at more cost, V(0,4) 4.9 its, V(1,5)
  !! 4.4. w (B_22 + W): radial lines V(4,4) 1.48 its, 17.4 s -> zebra V(0,6)
  !! 1.07 its, 14.6 s; zebra V(4,4) 1.20 / 21.2 s, zebra V(0,4) and lines
  !! V(0,4) / V(0,6) slower. The coarse levels keep 4 steps (2: no gain).
  !! (rho / T keep radial lines V(4,4): V(0,4) / V(0,6) save ~1.5 s of 12
  !! but raise T's cycles 1.43 -> 2.28 / 1.84.)
  integer, parameter, public :: SF_W_PRE0 = 0, SF_W_POST0 = 6, SF_W_NSC = 0
  !> Coarse levels (1 and below) smooth with Richardson on the same zebra /
  !! line blocks instead of GMRES: no norms or inner products, so no global
  !! reduction below level 0. GMRES there cost two allreduces over all ranks
  !! per step on levels with a few hundred rows per rank (none on many), and
  !! made the coarse smoothing flat in np: at 321x128 (w arm, v3) it went
  !! 51 -> 34 -> 44 -> 43 s from np 8 to np 128, 62% of the solve at np 128.
  !! 161x64 w arm, shaped pcbench, 10 steps (2026-09-28, sf_runs/sf_iter):
  !!   np 16 x 16: solve 31.8 -> 25.9 s; np 64 x 8 (2 nodes): 35.3 -> 28.1 s;
  !!   outer its 241-243 in all; pair_w inner its 2.5 -> 2.0.
  !! The zebra sweep takes the full step; the rho / T line-Jacobi blocks need
  !! damping (scale 1: T inner its 1.42 -> 1.85; 0.8: 1.47).
  !! At 321x128 np 64 x 8 it pays only together with SF_GMG_LINE_OVERLAP = 1
  !! (solve: reference 106.1 s; Richardson alone 107.2; overlap 1 alone
  !! 111.6; both 85.0 / 88.3 in two runs; Richardson from level 2 97.0): the
  !! pairs take a few more V-cycles (pair_w 2.5 -> 3.0), which the reduction-
  !! free coarse levels and the shorter lines pay for only jointly.
  !! Not on the schur arm's pair_w: its V-cycle needs GMRES on the coarse
  !! levels (161x64 np 64: 4.5 -> 16 V-cycles per solve with Richardson).
  !! Nor on its pair_psi, which also runs nested in the S_uu shell: with
  !! overlap 1 + Richardson there, 321x128 np 64 went from 16/12/10 outer its
  !! at tstep 1 to 71/38/52 (pair_w 25.6 V-cycles per solve); 161x64 held.
  integer, parameter, public :: SF_GMG_RICH_FROM = 1, SF_GMG_RICH_NONE = -1
  real*8,  parameter, public :: SF_GMG_RICH_OMEGA_ZEBRA = 1.0d0, SF_GMG_RICH_OMEGA_LINES = 0.8d0
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
                             nfields, smoother, maxits, Aop, Ablk, Amg, pre0, post0, nsmooth_c, &
                             line_overlap, axis_sectors, rich_from)
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
    integer, intent(in)          :: maxits    !< FGMRES budget (SF_GMG, or SF_LU with Aop)
    Mat, intent(in), optional    :: Aop       !< the operator to solve when it is not A
                                              !< (a MATSHELL); A then only preconditions:
                                              !< its LU (SF_LU) or its Galerkin chain (SF_GMG)
    Mat, intent(in), optional    :: Ablk      !< SF_GMG: level-0 smoother blocks from Ablk
    Mat, intent(in), optional    :: Amg       !< SF_GMG: the V-cycle's level-0 operator, when
                                              !< it is neither A nor Aop (default: Aop, else A)
    integer, intent(in), optional :: pre0, post0, nsmooth_c   !< SF_GMG: V-cycle shape
                                              !< (gmg_opts_t); absent = SF_GMG_NSMOOTH everywhere
    integer, intent(in), optional :: line_overlap, axis_sectors, rich_from  !< SF_GMG: per-block
                                              !< overrides of SF_GMG_LINE_OVERLAP, SF_GMG_AXIS_SECTORS,
                                              !< SF_GMG_RICH_FROM

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
      if (present(Aop)) then
        ! the exact reference for an operator that is not assembled: Krylov
        ! on it, preconditioned by the exact LU of the assembled A
        call KSPSetOperators(slv%ksp, Aop, A, ierr)
        call KSPSetType(slv%ksp, KSPFGMRES, ierr)
        call KSPGMRESSetRestart(slv%ksp, max(maxits, 2), ierr)
        call KSPSetTolerances(slv%ksp, rtol, 1.d-50, 1.d6, maxits, ierr)
      else
        call KSPSetOperators(slv%ksp, A, A, ierr)
        call KSPSetType(slv%ksp, KSPPREONLY, ierr)
      endif
      call KSPGetPC(slv%ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
      call KSPSetUp(slv%ksp, ierr)
      if (present(Aop)) then
        write(tstr,'(ES9.2)') rtol
        call pc_print_block_setup(comm, label, "FGMRES + LU (MUMPS) of the assembled surrogate, rtol "// &
                                  trim(adjustl(tstr)))
      else
        call pc_print_block_setup(comm, label, trim(sf_backend_name(backend)))
      endif
      if (fresh) call physics_pc_mumps_mem(slv%ksp, label, my_id)

    case (SF_GMG)
      !--- the complete hierarchy configuration, from this module's constants
      !--- only: no physics_pc_gmg_* namelist entry reaches this path.
      o%smoother     = smoother
      o%nsmooth      = SF_GMG_NSMOOTH
      o%axis_rings   = SF_GMG_AXIS_RINGS
      o%line_overlap = SF_GMG_LINE_OVERLAP
      if (present(pre0))      o%pre0      = pre0
      if (present(post0))     o%post0     = post0
      if (present(nsmooth_c)) o%nsmooth_c = nsmooth_c
      o%axis_sectors = SF_GMG_AXIS_SECTORS
      o%blockmv      = SF_BLOCKMV
      o%bnd_drop     = 1               ! Dirichlet DOFs out of the coarse spaces
      o%harm_split   = 1               ! the extracted blocks are |n|-diagonal
      o%axis_mult    = 0;  o%axis_split = 0;  o%smooth_op = 0;  o%ring_diag = 0
      o%omega        = 0.7d0;  o%axis_droptol = 0.d0;  o%ring_aspect = 1.d0
      o%rich_from    = SF_GMG_RICH_FROM
      o%rich_omega   = SF_GMG_RICH_OMEGA_ZEBRA
      if (smoother == SF_GMG_SMOOTHER_LINES) o%rich_omega = SF_GMG_RICH_OMEGA_LINES
      if (present(line_overlap)) o%line_overlap = line_overlap
      if (present(axis_sectors)) o%axis_sectors = axis_sectors
      if (present(rich_from))    o%rich_from    = rich_from

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
      if (present(Amg)) then
        call gmg_setup_operator(A, comm, my_id, Afine=Amg, tag=trim(label), opts=o, Ablk=Ablk)
      else
        call gmg_setup_operator(A, comm, my_id, Afine=Aop, tag=trim(label), opts=o, Ablk=Ablk)
      endif
      call gmg_select(1)

      !--- the Krylov wrapper: created once, re-pointed at every rebuild
      if (fresh) then
        ! -sf_gmg<k>_maxits: the FGMRES budget of hierarchy k, for experiments
        block
          PetscInt :: mx
          PetscBool :: set
          character(len=32) :: nm
          mx = maxits
          write(nm, '(A,I0,A)') "-sf_gmg", slv%gmg_inst, "_maxits"
          call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(nm), mx, set, ierr)
          call KSPCreate(comm, slv%ksp, ierr)
          call KSPSetType(slv%ksp, KSPFGMRES, ierr)
          call KSPGMRESSetRestart(slv%ksp, max(int(mx), 2), ierr)
          call KSPSetTolerances(slv%ksp, rtol, 1.d-50, 1.d6, mx, ierr)
        end block
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
      if (present(Aop)) then
        call KSPSetOperators(slv%ksp, Aop, A, ierr)
      else
        call KSPSetOperators(slv%ksp, A, A, ierr)
      endif
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
