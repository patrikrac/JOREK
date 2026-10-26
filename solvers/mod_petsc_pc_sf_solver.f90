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
  !> One type for every diagonal block of the SF sweep (mod_petsc_pc_sf),
  !! with two backends:
  !!   SF_GMG  FGMRES around one V-cycle of the C1 geometric multigrid
  !!           (mod_petsc_pc_gmg), hierarchy instance gmg_inst; all packed
  !!           fields of a node share the smoother blocks.
  !!   SF_LU   PREONLY + MUMPS LU, the exact reference.
  !! The GMG configuration is fixed below rather than in the namelist: each
  !! constant is the value the recorded measurements were taken at.
  !--------------------------------------------------------------------

  integer, parameter, public :: SF_LU  = 1
  integer, parameter, public :: SF_GMG = 2
  !> pair_w only: FGMRES on pair_w around the peeled Schur block
  !! preconditioner (mod_petsc_pc_sf_peel), whose scalar blocks use the two above
  integer, parameter, public :: SF_PEEL = 3

  !--- GMG smoothers (codes of mod_petsc_pc_gmg) ----------------------------
  !> Radial-line block Jacobi: rho.
  integer, parameter, public :: SF_GMG_SMOOTHER_LINES = 5
  !> Zebra rings inside the switch radius (median r dtheta/dr = 1), zebra
  !! radial lines outside (GMGPolar, Bourne et al. JCP 488 (2023)): pair_psi,
  !! (psi, j) smoothed collectively (Chacon JCP 526 (2025) S4.1).
  integer, parameter, public :: SF_GMG_SMOOTHER_ZEBRA_RINGS = 8
  !> Flux-surface ring blocks: pair_w, whose psi - u coupling theta dt Bpar
  !! runs along the field lines at large dt, and T (chi_par diffusion along
  !! B; ZK_par 1-1000 at one V-cycle where radial lines took up to 5). Ring
  !! smoothers also take SF_GMG_RING_OVERLAP and SF_GMG_SEMI_R.
  integer, parameter, public :: SF_GMG_SMOOTHER_RINGS = 4
  !> Rings cut by a rank boundary are completed from the neighbours (RAS
  !! along J); 2 already fetches every cut ring whole.
  integer, parameter, public :: SF_GMG_RING_OVERLAP = 2
  !> Radial-only coarsening until the median cell aspect r dtheta/dr is 2
  !! (-1 = that level count from the grid): ring blocks leave the radial
  !! couplings to the coarse grid, which a 2:1 full coarsening cannot take.
  integer, parameter, public :: SF_GMG_SEMI_R = -1
  !> Axis block rings: 0 on the pairs (the shared axis value only), 0..1 on
  !! the scalar blocks. Larger axis blocks were the strong-scaling floor.
  integer, parameter, public :: SF_GMG_AXIS_RINGS_PAIRS = 0
  integer, parameter, public :: SF_GMG_AXIS_RINGS_RHOT  = 1
  !> Radial lines extended by one node into the neighbouring ranks (RAS, Cai
  !! & Sarkis SISC 21 (1999)): JOREK partitions ring by ring, so every rank
  !! boundary cuts every line.
  integer, parameter, public :: SF_GMG_LINE_OVERLAP = 1
  !> Axis blocks over J-sector ranks (mod_petsc_pc_gmg_axis); -1 = the cost
  !! model's count. Exact, gated against the LU on the first build.
  integer, parameter, public :: SF_GMG_AXIS_SECTORS = -1
  !> Level and coupling-block matvecs on the OpenMP block kernel
  !! (jorek_blockmv_attach.c).
  integer, parameter, public :: SF_BLOCKMV = 1
  !> Coarse levels (>= 1) smooth with Richardson on the smoother blocks: no
  !! global reductions below level 0, the V-cycle's strong-scaling floor.
  integer, parameter, public :: SF_GMG_RICH_FROM = 1
  !> Richardson scale: zebra sweeps take the full step, the additive
  !! line / ring Jacobi blocks are damped (undamped ring Jacobi is no
  !! smoother where the mass is small against the radial coupling).
  real*8,  parameter, public :: SF_GMG_RICH_OMEGA_ZEBRA = 1.0d0, SF_GMG_RICH_OMEGA_ADDITIVE = 0.8d0

  !--- V-cycle shapes (gmg_opts_t pre0 / post0 / nsmooth_c) -----------------
  !> pair_psi V(0,10), 2 steps per side below: from a psi-row right-hand
  !! side the coarse correction leaves a j-row residual at the edge that only
  !! level-0 post-smoothing removes.
  integer, parameter, public :: SF_PJ_PRE0 = 0, SF_PJ_POST0 = 10, SF_PJ_NSC = 2
  !> pair_w V(0,6), 2 steps per side below (each coarse step is a neighbour
  !! exchange on levels with few rows per rank).
  integer, parameter, public :: SF_W_PRE0 = 0, SF_W_POST0 = 6, SF_W_NSC = 2
  !> The toroidal slots per GMG smoother block, set by
  !! physics_pc_sf_harm_couple: all slots of a node (cross-|n| coupling), or
  !! a harmonic's cos and sin slots (d_phi maps one to the other).
  integer, parameter, public :: SF_GMG_HARM_PAIR = 1, SF_GMG_HARM_ALL = 2
  integer, save, public :: sf_gmg_harm = SF_GMG_HARM_ALL
  !> FGMRES budget around a V-cycle: the packed pairs, the scalar blocks
  integer, parameter, public :: SF_GMG_MAXITS      = 30
  integer, parameter, public :: SF_GMG_MAXITS_RHOT = 10

  !--- v_par stages (physics_pc_sf_vpar_stage) -------------------------------
  integer, parameter, public :: SF_VPAR_V0 = 0   !< B_77 last
  integer, parameter, public :: SF_VPAR_B0 = 1   !< B_77 + W_vv after pair_w
  integer, parameter, public :: SF_VPAR_B1 = 2   !< v_par in pair_w (mode split)

  type, public :: block_solver_t
    KSP     :: ksp
    integer :: backend  = SF_LU
    Vec     :: dscale                      !< symmetric block scaling D, if scaled
    logical :: scaled   = .false.
    logical :: created  = .false.
    integer :: gmg_inst = 0
    integer :: its_sum  = 0, its_max = 0, nsolve = 0
    integer :: nfail    = 0                !< unconverged solves (its cap, failed PC)
    real*8  :: t_sum    = 0.d0             !< wall time in its solves (this rank)
    character(len=56) :: label = ""
  end type block_solver_t

  public :: sf_solver_setup, sf_solver_apply, sf_solver_reset_counters, sf_solver_report
  public :: sf_split_parts, sf_opz
  public :: sf_vpar_stage, sf_vpar_wmask, sf_vpar_gamma_w, sf_vpar_w_assembled, sf_vpar_wvu_full

contains

  !> physics_pc_sf_vpar_stage -> SF_VPAR_*; -1 for an unknown string.
  integer function sf_vpar_stage()
    use phys_module, only: physics_pc_sf_vpar_stage
    select case (trim(adjustl(physics_pc_sf_vpar_stage)))
    case ("v0");  sf_vpar_stage = SF_VPAR_V0
    case ("b0");  sf_vpar_stage = SF_VPAR_B0
    case ("b1");  sf_vpar_stage = SF_VPAR_B1
    case default; sf_vpar_stage = -1
    end select
  end function sf_vpar_stage

  !> -sf_vpar_w: v_par's W terms in b0 / b1, bit 0 W_vv on B_77, bit 1 W_vu
  !! on B_72, bit 2 W_uv on B_27 (b1 only). Default all; 0 = the stage
  !! without the sound-wave Schur terms.
  integer function sf_vpar_wmask()
    PetscInt :: iv
    PetscBool :: set
    PetscErrorCode :: ierr
    iv = 3
    if (sf_vpar_stage() == SF_VPAR_B1) iv = 7
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_vpar_w", iv, set, ierr)
    sf_vpar_wmask = int(iv)
  end function sf_vpar_wmask

  !> -sf_vpar_gamma_w: the adiabatic index of v_par's W terms (default
  !! GAMMA; strong parallel conduction is closer to isothermal, 1).
  real*8 function sf_vpar_gamma_w()
    use phys_module, only: gamma
    PetscReal :: rv
    PetscBool :: set
    PetscErrorCode :: ierr
    rv = gamma
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_vpar_gamma_w", rv, set, ierr)
    sf_vpar_gamma_w = rv
  end function sf_vpar_gamma_w

  !> -sf_vpar_wvu_full 1: W_vu with the whole p channel, psi_w in v_par's row.
  logical function sf_vpar_wvu_full()
    PetscInt :: iv
    PetscBool :: set
    PetscErrorCode :: ierr
    iv = 0
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_vpar_wvu_full", iv, set, ierr)
    sf_vpar_wvu_full = (iv /= 0)
  end function sf_vpar_wvu_full

  !> Are v_par's W terms assembled? With v_par, the SF path and stage b0 / b1.
  logical function sf_vpar_w_assembled()
    use mod_parameters, only: var_Vpar
    use phys_module,    only: use_physics_pc
    sf_vpar_w_assembled = (var_Vpar > 0) .and. use_physics_pc .and. &
                          (sf_vpar_stage() == SF_VPAR_B0 .or. sf_vpar_stage() == SF_VPAR_B1)
  end function sf_vpar_w_assembled

  !--------------------------------------------------------------------
  !> Configure slv to solve A. The KSP is created once and re-pointed at the
  !! refilled (pattern-frozen) operator on every rebuild, so MUMPS and the
  !! GMG reuse their symbolic phases. pre0 / post0 / nsmooth_c: the V-cycle
  !! shape (absent: the smoother's default everywhere); axis_rings: default
  !! SF_GMG_AXIS_RINGS_RHOT.
  !--------------------------------------------------------------------
  subroutine sf_solver_setup(slv, A, backend, label, comm, my_id, rtol, gmg_inst, nfields, &
                             smoother, maxits, pre0, post0, nsmooth_c, axis_rings, ncyc)
    use mod_petsc_pc_gmg, only: gmg_select, gmg_is_ready, gmg_build_prolongations, &
                                gmg_setup_operator, gmg_pc_apply_1, gmg_pc_apply_2, &
                                gmg_pc_apply_3, gmg_pc_apply_4, gmg_pc_apply_5, &
                                gmg_pc_apply_6, gmg_pc_apply_7, gmg_pc_apply_8, gmg_pc_apply_9, &
                                gmg_pc_apply_10, gmg_opts_t
    type(block_solver_t), intent(inout) :: slv
    Mat, intent(in)              :: A
    integer, intent(in)          :: backend, comm, my_id
    character(len=*), intent(in) :: label
    real*8, intent(in)           :: rtol
    integer, intent(in)          :: gmg_inst   !< hierarchy instance (1..10)
    integer, intent(in)          :: nfields    !< fields packed per node in A
    integer, intent(in)          :: smoother   !< SF_GMG_SMOOTHER_*
    integer, intent(in)          :: maxits     !< FGMRES budget
    integer, intent(in), optional :: pre0, post0, nsmooth_c, axis_rings
    !> GMG: > 0 = exactly ncyc V-cycles (Richardson, no norms: a fixed linear
    !! operator) instead of FGMRES to rtol
    integer, intent(in), optional :: ncyc

    PC :: pc
    PetscErrorCode :: ierr
    logical :: ok, fresh, fixed
    character(len=24) :: tstr
    type(gmg_opts_t) :: o

    fresh        = .not. slv%created
    fixed        = .false.
    if (present(ncyc)) fixed = (ncyc > 0)
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
      call pc_print_block_setup(comm, label, "PREONLY + LU (MUMPS)")
      if (fresh) call physics_pc_mumps_mem(slv%ksp, label, my_id)

    case (SF_GMG)
      o%smoother     = smoother
      o%axis_rings   = SF_GMG_AXIS_RINGS_RHOT
      o%line_overlap = SF_GMG_LINE_OVERLAP
      o%axis_sectors = SF_GMG_AXIS_SECTORS
      o%blockmv      = SF_BLOCKMV
      o%bnd_drop     = 1               ! Dirichlet DOFs out of the coarse spaces
      o%rich_from    = SF_GMG_RICH_FROM
      o%rich_omega   = SF_GMG_RICH_OMEGA_ZEBRA
      o%harm_pair    = sf_gmg_harm
      if (smoother == SF_GMG_SMOOTHER_LINES .or. smoother == SF_GMG_SMOOTHER_RINGS) &
        o%rich_omega = SF_GMG_RICH_OMEGA_ADDITIVE
      if (smoother == SF_GMG_SMOOTHER_RINGS) then
        o%ring_overlap = SF_GMG_RING_OVERLAP
        o%semi_r       = SF_GMG_SEMI_R
      endif
      if (present(pre0))       o%pre0       = pre0
      if (present(post0))      o%post0      = post0
      if (present(nsmooth_c))  o%nsmooth_c  = nsmooth_c
      if (present(axis_rings)) o%axis_rings = axis_rings

      !--- the hierarchy: built once per instance, refilled per rebuild
      call gmg_select(slv%gmg_inst)
      if (.not. gmg_is_ready()) then
        call gmg_build_prolongations(A, comm, my_id, nfields, ok, opts=o)
        if (.not. ok) then
          if (my_id == 0) write(*,'(A,A,A)') "[Physics PC]   FATAL: the GMG backend for ", trim(label), &
            " needs the structured flux-surface grid."
          call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
      endif
      call gmg_setup_operator(A, comm, my_id, tag=trim(label), opts=o)

      if (fresh) then
        call KSPCreate(comm, slv%ksp, ierr)
        call KSPSetType(slv%ksp, KSPFGMRES, ierr)
        if (fixed) then
          call KSPSetType(slv%ksp, KSPRICHARDSON, ierr)
          call KSPSetNormType(slv%ksp, KSP_NORM_NONE, ierr)
        endif
        block
          ! -sf_gmg<k>_maxits, and -sf_gmg<k>_ksp_* through the prefix
          PetscInt :: mx
          PetscBool :: set
          character(len=32) :: nm
          mx = maxits
          write(nm, '(A,I0,A)') "-sf_gmg", slv%gmg_inst, "_maxits"
          call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(nm), mx, set, ierr)
          if (fixed) mx = ncyc
          call KSPGMRESSetRestart(slv%ksp, max(int(mx), 2), ierr)
          call KSPSetTolerances(slv%ksp, rtol, 1.d-50, 1.d6, mx, ierr)
          write(nm, '(A,I0,A)') "sf_gmg", slv%gmg_inst, "_"
          call KSPSetOptionsPrefix(slv%ksp, trim(nm), ierr)
        end block
        call KSPSetFromOptions(slv%ksp, ierr)
        call KSPGetPC(slv%ksp, pc, ierr)
        call PCSetType(pc, PCSHELL, ierr)
        select case (slv%gmg_inst)
        case (1); call PCShellSetApply(pc, gmg_pc_apply_1, ierr)
        case (2); call PCShellSetApply(pc, gmg_pc_apply_2, ierr)
        case (3); call PCShellSetApply(pc, gmg_pc_apply_3, ierr)
        case (4); call PCShellSetApply(pc, gmg_pc_apply_4, ierr)
        case (5); call PCShellSetApply(pc, gmg_pc_apply_5, ierr)
        case (6); call PCShellSetApply(pc, gmg_pc_apply_6, ierr)
        case (7); call PCShellSetApply(pc, gmg_pc_apply_7, ierr)
        case (8); call PCShellSetApply(pc, gmg_pc_apply_8, ierr)
        case (9); call PCShellSetApply(pc, gmg_pc_apply_9, ierr)
        case (10); call PCShellSetApply(pc, gmg_pc_apply_10, ierr)
        end select
        call PCShellSetName(pc, trim(label)//" C1 GMG V-cycle", ierr)
      endif
      call KSPSetOperators(slv%ksp, A, A, ierr)
      call KSPSetUp(slv%ksp, ierr)
      if (fixed) then
        write(tstr,'(I0)') ncyc
        call pc_print_block_setup(comm, label, "RICHARDSON + SHELL[C1 GMG V-cycle], "//trim(tstr)//" cycle(s)")
      else
        write(tstr,'(ES9.2)') rtol
        call pc_print_block_setup(comm, label, "FGMRES + SHELL[C1 GMG V-cycle], rtol "//trim(adjustl(tstr)))
      endif

    case default
      if (my_id == 0) write(*,'(A,I0)') "[Physics PC]   FATAL: sf_solver_setup got an unknown backend ", backend
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end select

    slv%created = .true.
  end subroutine sf_solver_setup

  !> Solve A z = r through slv, with the block scaling as an exact similarity
  !! (rhs is scaled in place: the caller refills it every apply), and count.
  subroutine sf_solver_apply(slv, rhs, sol, ierr)
    type(block_solver_t), intent(inout) :: slv
    Vec, intent(in) :: rhs, sol
    PetscErrorCode, intent(inout) :: ierr
    PetscInt :: its
    KSPConvergedReason :: reason
    real*8 :: t0

    t0 = MPI_Wtime()
    if (slv%scaled) call VecPointwiseMult(rhs, rhs, slv%dscale, ierr)
    call KSPSolve(slv%ksp, rhs, sol, ierr)
    if (slv%scaled) call VecPointwiseMult(sol, sol, slv%dscale, ierr)
    slv%t_sum = slv%t_sum + (MPI_Wtime() - t0)

    call KSPGetIterationNumber(slv%ksp, its, ierr)
    slv%its_sum = slv%its_sum + int(its)
    slv%its_max = max(slv%its_max, int(its))
    slv%nsolve  = slv%nsolve + 1
    ! an unconverged inner solve is still a usable preconditioner
    ! application: counted, not raised
    call KSPGetConvergedReason(slv%ksp, reason, ierr)
    if (reason%v < 0) slv%nfail = slv%nfail + 1
  end subroutine sf_solver_apply

  subroutine sf_solver_reset_counters(slv)
    type(block_solver_t), intent(inout) :: slv
    slv%its_sum = 0; slv%its_max = 0; slv%nsolve = 0; slv%nfail = 0; slv%t_sum = 0.d0
  end subroutine sf_solver_reset_counters

  !> One line: mean and max inner iterations since the last reset.
  subroutine sf_solver_report(slv, my_id)
    type(block_solver_t), intent(in) :: slv
    integer, intent(in) :: my_id
    if (my_id /= 0 .or. slv%nsolve == 0) return
    write(*,'(A,A,A,F7.2,A,I0,A,I0,A)', advance="no") "[Physics PC]   inner ", trim(slv%label), &
      ": mean ", dble(slv%its_sum) / dble(slv%nsolve), " its, max ", slv%its_max, &
      " (", slv%nsolve, " solves)"
    if (slv%nfail > 0) then
      write(*,'(A,I0,A)') ", WARNING ", slv%nfail, " not converged"
    else
      write(*,*)
    endif
  end subroutine sf_solver_report

  !> Between a packed vector p = [h(1) local | h(2) local | ...] and its
  !! fields: a rank-local copy, no communication.
  subroutine sf_split_parts(p, h, to_p)
    Vec :: p
    Vec, intent(in) :: h(:)
    logical, intent(in) :: to_p
    PetscScalar, pointer :: pa(:), a(:)
    PetscErrorCode :: ierr
    PetscInt :: n, o
    integer :: k

    if (to_p) then
      call VecGetArray(p, pa, ierr)
    else
      call VecGetArrayRead(p, pa, ierr)
    endif
    o = 0
    do k = 1, size(h)
      call VecGetLocalSize(h(k), n, ierr)
      if (to_p) then
        call VecGetArrayRead(h(k), a, ierr)
        pa(o + 1:o + n) = a(1:n)
        call VecRestoreArrayRead(h(k), a, ierr)
      else
        call VecGetArray(h(k), a, ierr)
        a(1:n) = pa(o + 1:o + n)
        call VecRestoreArray(h(k), a, ierr)
      endif
      o = o + n
    enddo
    if (to_p) then
      call VecRestoreArray(p, pa, ierr)
    else
      call VecRestoreArrayRead(p, pa, ierr)
    endif
  end subroutine sf_split_parts

#endif

  !> 1 + zeta, as the Jacobian forms it (zeta rescaled for variable dt)
  real*8 function sf_opz()
    use phys_module, only: time_evol_zeta, tstep, tstep_prev
    sf_opz = 1.0d0 + time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  end function sf_opz

end module mod_petsc_pc_sf_solver
