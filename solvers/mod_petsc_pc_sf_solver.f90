!> The solver of one diagonal block of the SF sweep (mod_petsc_pc_sf), with
!! two backends: FGMRES around one V-cycle of the geometric multigrid
!! (mod_petsc_pc_gmg), or one V-cycle alone (one_cycle, vs_mono), or
!! PREONLY + MUMPS LU as the exact reference. The GMG configuration is fixed
!! here, per block, rather than in the namelist. A third backend, SF_SPDMG,
!! is the ring-free multigrid of mod_petsc_pc_spdmg for a scalar symmetric
!! positive definite block (vs_lumped's P, -vs_p_solver spdmg, and its P_V,
!! -vs_pv_solver spdmg). A fourth, SF_HYPRE ("hypre"), is BoomerAMG for a
!! scalar block (mod_petsc_pc_hypre).
module mod_petsc_pc_sf_solver
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: physics_pc_mumps_mem
  use mod_petsc_pc_blocks,      only: pc_print_block_setup, make_field_block_scale, decouple_penalty_rows
  use mod_petsc_pc_sf_ops,      only: sf_op_t, sfo_n1
  use mod_petsc_pc_gmg,         only: GMG_SM_RINGS, GMG_SM_LINES, GMG_SM_ZEBRA_RINGS
  implicit none
  private

  integer, parameter, public :: SF_LU  = 1
  integer, parameter, public :: SF_GMG = 2
  !> The ring-free multigrid for a scalar SPD block (mod_petsc_pc_spdmg):
  !! vs_lumped's P with -vs_p_solver spdmg, P_V with -vs_pv_solver spdmg;
  !! not a namelist backend.
  integer, parameter, public :: SF_SPDMG = 3
  !> BoomerAMG (hypre via PCHYPRE) for a scalar block (mod_petsc_pc_hypre):
  !! the namelist value "hypre", -vs_p_solver hypre and -vs_pv_solver hypre.
  integer, parameter, public :: SF_HYPRE = 4

  !> Radial-line blocks: rho, v_par v0.
  integer, parameter, public :: SF_GMG_SMOOTHER_LINES = GMG_SM_LINES
  !> Zebra rings inside the radius where the median cell aspect r dtheta/dr
  !! reaches 1, zebra radial lines outside (Bourne et al., JCP 488 (2023)):
  !! pair_psi, with (psi, j) in one block (Chacon, JCP 526 (2025) S4.1).
  integer, parameter, public :: SF_GMG_SMOOTHER_ZEBRA_RINGS = GMG_SM_ZEBRA_RINGS
  !> Flux-surface ring blocks: pair_w, whose psi - u coupling runs along the
  !! field lines at large dt, and T and v_par b0 (diffusion along B).
  integer, parameter, public :: SF_GMG_SMOOTHER_RINGS = GMG_SM_RINGS
  !> Ring smoothers coarsen radially only until the median cell aspect is 2
  !! (-1: the level count follows from the grid).
  integer, parameter, public :: SF_GMG_SEMI_R = -1
  !> Axis block extent in rings: -2 = automatic on the pairs, 1 on the scalars.
  integer, parameter, public :: SF_GMG_AXIS_RINGS_PAIRS = -2
  integer, parameter, public :: SF_GMG_AXIS_RINGS_RHOT  = 1
  !> Coarse-level Richardson scale: full steps for zebra sweeps, damped for
  !! the additive ring / line blocks.
  real*8,  parameter, public :: SF_GMG_RICH_OMEGA_ZEBRA = 1.0d0, SF_GMG_RICH_OMEGA_ADDITIVE = 0.8d0

  !> V-cycle shapes: level-0 pre / post steps, steps per side below. pair_psi
  !! needs the long level-0 post-smoothing: from a psi-row right-hand side the
  !! coarse correction leaves a j-row residual at the edge.
  integer, parameter, public :: SF_PJ_PRE0 = 0, SF_PJ_POST0 = 10, SF_PJ_NSC = 2
  integer, parameter, public :: SF_W_PRE0  = 0, SF_W_POST0  = 6,  SF_W_NSC  = 2
  !> Toroidal slots per smoother block: a harmonic's cos and sin, or all.
  integer, parameter, public :: SF_GMG_HARM_PAIR = 1, SF_GMG_HARM_ALL = 2
  integer, save, public :: sf_gmg_harm = SF_GMG_HARM_ALL
  !> FGMRES iteration caps around a V-cycle: the pairs, the scalar blocks.
  integer, parameter, public :: SF_GMG_MAXITS      = 30
  integer, parameter, public :: SF_GMG_MAXITS_RHOT = 10

  !> v_par stages (physics_pc_sf_vpar_stage): B_77 last; B_77 + W_vv after
  !! pair_w; v_par in pair_w (mode split).
  integer, parameter, public :: SF_VPAR_V0 = 0, SF_VPAR_B0 = 1, SF_VPAR_B1 = 2

  type, public :: block_solver_t
    KSP     :: ksp
    integer :: backend  = SF_LU
    Vec     :: dscale                      !< symmetric block scaling D, if scaled
    logical :: scaled   = .false.
    Vec     :: pfac                        !< rhs factor of the decoupled penalty rows, if penal
    logical :: penal    = .false.
    logical :: created  = .false.
    integer :: its_sum  = 0, its_max = 0, nsolve = 0
    integer :: nfail    = 0                !< unconverged solves
    real*8  :: t_sum    = 0.d0             !< wall time in its solves
    character(len=32) :: label = ""
  end type block_solver_t

  public :: sf_solver_setup, sf_solver_apply, sf_solver_solve, sf_solver_reset_counters, sf_solver_report
  public :: sf_solver_family_report, sf_solver_block_scale, sf_solver_penalty_rows, sf_backend
  public :: sf_split_parts, sf_opz, sf_vpar_stage, sf_vpar_w_assembled

contains

  !> A backend name of the namelist ("lu" | "gmg" | "hypre") as SF_LU /
  !! SF_GMG / SF_HYPRE; any other name stops the run. nm is the namelist
  !! variable, for the message.
  integer function sf_backend(s, nm, my_id)
    character(len=*), intent(in) :: s, nm
    integer, intent(in) :: my_id
    integer :: mpierr
    select case (trim(adjustl(s)))
    case ("lu");    sf_backend = SF_LU
    case ("gmg");   sf_backend = SF_GMG
    case ("hypre"); sf_backend = SF_HYPRE
    case default
      sf_backend = SF_LU
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: ", trim(nm)//" must be lu | gmg | hypre, got '"// &
                                       trim(s)//"'"
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    end select
  end function sf_backend

  !> physics_pc_sf_vpar_stage as SF_VPAR_*, -1 for an unknown string.
  integer function sf_vpar_stage()
    use phys_module, only: physics_pc_sf_vpar_stage
    select case (trim(adjustl(physics_pc_sf_vpar_stage)))
    case ("v0");  sf_vpar_stage = SF_VPAR_V0
    case ("b0");  sf_vpar_stage = SF_VPAR_B0
    case ("b1");  sf_vpar_stage = SF_VPAR_B1
    case default; sf_vpar_stage = -1
    end select
  end function sf_vpar_stage

  !> Does the v_par stage need v_par's W terms (b0 / b1, the SF sweep and
  !! vs_alfven)?
  logical function sf_vpar_w_assembled()
    use mod_parameters, only: var_Vpar
    use phys_module,    only: use_physics_pc
    sf_vpar_w_assembled = (var_Vpar > 0) .and. use_physics_pc .and. &
                          (sf_vpar_stage() == SF_VPAR_B0 .or. sf_vpar_stage() == SF_VPAR_B1)
  end function sf_vpar_w_assembled

  !> Configure slv to solve A. The KSP is created once and re-pointed at the
  !! refilled operator on every rebuild, so MUMPS and the GMG reuse their
  !! symbolic phases. The GMG KSP takes PETSc options with the prefix
  !! sf_gmg<gmg_inst>_. one_cycle (GMG backend, default .false.): PREONLY
  !! around the V-cycle, one cycle per solve, instead of FGMRES to rtol; set
  !! before the options, so -sf_gmg<gmg_inst>_ksp_type still overrides it.
  !! amat (default A): the operator of the solve when A only builds its
  !! preconditioner: the Krylov method runs on amat with the V-cycle of A
  !! (GMG, SPD-MG), LU factors amat itself.
  subroutine sf_solver_setup(slv, A, backend, label, comm, my_id, rtol, gmg_inst, nfields, &
                             smoother, maxits, pre0, post0, nsmooth_c, axis_rings, one_cycle, &
                             spdmg_inst, amat)
    use mod_petsc_pc_gmg, only: gmg_select, gmg_is_ready, gmg_build_prolongations, &
                                gmg_setup_operator, gmg_pc_apply_1, gmg_pc_apply_2, &
                                gmg_pc_apply_3, gmg_pc_apply_4, gmg_pc_apply_5, gmg_pc_apply_6, &
                                gmg_opts_t
    use mod_petsc_pc_spdmg, only: spdmg_ksp_setup, SPDMG_P
    use mod_petsc_pc_hypre, only: hypre_ksp_setup
    use mod_petsc_pc_harm, only: pc_ntor
    type(block_solver_t), intent(inout) :: slv
    Mat, intent(in)              :: A
    integer, intent(in)          :: backend, comm, my_id
    character(len=*), intent(in) :: label
    real*8, intent(in)           :: rtol
    integer, intent(in)          :: gmg_inst   !< hierarchy instance (1..6)
    integer, intent(in)          :: nfields    !< fields packed per node in A
    integer, intent(in)          :: smoother   !< SF_GMG_SMOOTHER_*
    integer, intent(in)          :: maxits     !< FGMRES iteration cap
    integer, intent(in), optional :: pre0, post0, nsmooth_c, axis_rings
    logical, intent(in), optional :: one_cycle
    integer, intent(in), optional :: spdmg_inst !< SF_SPDMG's instance (default SPDMG_P)
    Mat, intent(in), optional    :: amat
    PC :: pc
    PetscErrorCode :: ierr
    logical :: ok, fresh, one
    character(len=24) :: tstr
    character(len=16) :: prefix
    KSPType :: ktyp
    type(gmg_opts_t) :: o
    integer :: kinst

    fresh       = .not. slv%created
    slv%backend = backend
    slv%label   = label
    one         = .false.
    if (present(one_cycle)) one = one_cycle
    kinst       = SPDMG_P
    if (present(spdmg_inst)) kinst = spdmg_inst

    select case (backend)

    case (SF_LU)
      if (fresh) call KSPCreate(comm, slv%ksp, ierr)
      if (present(amat)) then
        call KSPSetOperators(slv%ksp, amat, amat, ierr)
      else
        call KSPSetOperators(slv%ksp, A, A, ierr)
      endif
      call KSPSetType(slv%ksp, KSPPREONLY, ierr)
      call KSPGetPC(slv%ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
      call KSPSetUp(slv%ksp, ierr)
      call pc_print_block_setup(comm, label, "PREONLY + LU (MUMPS)")
      if (fresh) call physics_pc_mumps_mem(slv%ksp, label, my_id)

    case (SF_GMG)
      o%smoother   = smoother
      o%axis_rings = SF_GMG_AXIS_RINGS_RHOT
      o%harm_pair  = sf_gmg_harm
      o%rich_omega = SF_GMG_RICH_OMEGA_ZEBRA
      if (smoother /= SF_GMG_SMOOTHER_ZEBRA_RINGS) o%rich_omega = SF_GMG_RICH_OMEGA_ADDITIVE
      if (smoother == SF_GMG_SMOOTHER_RINGS) o%semi_r = SF_GMG_SEMI_R
      if (present(pre0))       o%pre0       = pre0
      if (present(post0))      o%post0      = post0
      if (present(nsmooth_c))  o%nsmooth_c  = nsmooth_c
      if (present(axis_rings)) o%axis_rings = axis_rings

      ! the hierarchy is built once per instance and refilled per rebuild
      call gmg_select(gmg_inst)
      if (.not. gmg_is_ready()) then
        call gmg_build_prolongations(A, comm, my_id, nfields, ok, opts=o)
        if (.not. ok) then
          if (my_id == 0) write(*,'(A,A,A)') "[Physics PC]   FATAL: no GMG hierarchy for ", trim(label), &
            " on this grid (see the refusal above)."
          call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
      endif
      call gmg_setup_operator(A, comm, my_id, tag=trim(label), opts=o)

      if (fresh) then
        call KSPCreate(comm, slv%ksp, ierr)
        if (one) then
          call KSPSetType(slv%ksp, KSPPREONLY, ierr)
        else
          call KSPSetType(slv%ksp, KSPFGMRES, ierr)
          call KSPGMRESSetRestart(slv%ksp, maxits, ierr)
        endif
        call KSPSetTolerances(slv%ksp, rtol, 1.d-50, 1.d6, maxits, ierr)
        write(prefix, '(A,I0,A)') "sf_gmg", gmg_inst, "_"
        call KSPSetOptionsPrefix(slv%ksp, trim(prefix), ierr)
        call KSPSetFromOptions(slv%ksp, ierr)
        call KSPGetPC(slv%ksp, pc, ierr)
        call PCSetType(pc, PCSHELL, ierr)
        select case (gmg_inst)
        case (1); call PCShellSetApply(pc, gmg_pc_apply_1, ierr)
        case (2); call PCShellSetApply(pc, gmg_pc_apply_2, ierr)
        case (3); call PCShellSetApply(pc, gmg_pc_apply_3, ierr)
        case (4); call PCShellSetApply(pc, gmg_pc_apply_4, ierr)
        case (5); call PCShellSetApply(pc, gmg_pc_apply_5, ierr)
        case (6); call PCShellSetApply(pc, gmg_pc_apply_6, ierr)
        end select
        call PCShellSetName(pc, trim(label)//" GMG V-cycle", ierr)
      endif
      if (present(amat)) then
        call KSPSetOperators(slv%ksp, amat, A, ierr)
      else
        call KSPSetOperators(slv%ksp, A, A, ierr)
      endif
      call KSPSetUp(slv%ksp, ierr)
      write(tstr,'(ES9.2)') rtol
      if (.not. one) then
        call pc_print_block_setup(comm, label, "FGMRES + GMG V-cycle, rtol "//trim(adjustl(tstr)))
      else
        ! the KSP type the options left
        call KSPGetType(slv%ksp, ktyp, ierr)
        if (trim(ktyp) == KSPPREONLY) then
          call pc_print_block_setup(comm, label, "PREONLY + GMG V-cycle (one cycle per solve)")
        else
          call pc_print_block_setup(comm, label, trim(ktyp)//" + GMG V-cycle (by the options), rtol "// &
                                    trim(adjustl(tstr)))
        endif
      endif

    case (SF_SPDMG)
      ! one field, all toroidal slots of the operator per node block
      if (nfields /= 1) then
        if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: the SPD multigrid takes one field, got ", trim(label)
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      endif
      if (fresh) call KSPCreate(comm, slv%ksp, ierr)
      call spdmg_ksp_setup(slv%ksp, A, int(pc_ntor), comm, my_id, rtol, maxits, label, fresh, kinst, amat)
      if (fresh) then
        call KSPGetType(slv%ksp, ktyp, ierr)
        write(tstr,'(ES9.2)') rtol
        call pc_print_block_setup(comm, label, trim(ktyp)//" + SPD-MG V-cycle (ring-free), rtol "// &
                                  trim(adjustl(tstr)))
      endif

    case (SF_HYPRE)
      ! BoomerAMG with the axis block and the flip of mod_petsc_pc_hypre,
      ! options -sf_amg<gmg_inst>_*
      if (fresh) call KSPCreate(comm, slv%ksp, ierr)
      call hypre_ksp_setup(slv%ksp, A, int(pc_ntor), gmg_inst, nfields, label, comm, my_id, rtol, maxits, fresh, &
                           amat)

    case default
      if (my_id == 0) write(*,'(A,I0)') "[Physics PC]   FATAL: sf_solver_setup got an unknown backend ", backend
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end select

    slv%created = .true.
  end subroutine sf_solver_setup

  !> Solve A z = r through slv, with the block scaling as an exact similarity
  !! (rhs is scaled in place: the caller refills it on every apply).
  !! An unconverged inner solve is still a usable preconditioner application:
  !! it is counted, not raised.
  subroutine sf_solver_apply(slv, rhs, sol, ierr)
    type(block_solver_t), intent(inout) :: slv
    Vec, intent(in) :: rhs, sol
    PetscErrorCode, intent(inout) :: ierr
    PetscInt :: its
    KSPConvergedReason :: reason
    real*8 :: t0

    t0 = MPI_Wtime()
    if (slv%penal)  call VecPointwiseMult(rhs, rhs, slv%pfac, ierr)
    if (slv%scaled) call VecPointwiseMult(rhs, rhs, slv%dscale, ierr)
    call KSPSolve(slv%ksp, rhs, sol, ierr)
    if (slv%scaled) call VecPointwiseMult(sol, sol, slv%dscale, ierr)
    slv%t_sum = slv%t_sum + (MPI_Wtime() - t0)

    call KSPGetIterationNumber(slv%ksp, its, ierr)
    slv%its_sum = slv%its_sum + int(its)
    slv%its_max = max(slv%its_max, int(its))
    slv%nsolve  = slv%nsolve + 1
    call KSPGetConvergedReason(slv%ksp, reason, ierr)
    if (reason%v < 0) slv%nfail = slv%nfail + 1
  end subroutine sf_solver_apply

  !> sf_solver_apply inside the log event ev.
  subroutine sf_solver_solve(slv, rhs, sol, ev)
    type(block_solver_t), intent(inout) :: slv
    Vec :: rhs, sol
    PetscLogEvent, intent(in) :: ev
    PetscErrorCode :: ierr
    ierr = 0
    call PetscLogEventBegin(ev, ierr)
    call sf_solver_apply(slv, rhs, sol, ierr)
    call PetscLogEventEnd(ev, ierr)
  end subroutine sf_solver_solve

  !> The symmetric field scaling of op (make_field_block_scale), kept in slv
  !! for its solves; at every rebuild, after the refill.
  subroutine sf_solver_block_scale(slv, op, comm, my_id)
    type(block_solver_t), intent(inout) :: slv
    type(sf_op_t), intent(inout) :: op
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr
    if (slv%scaled) call VecDestroy(slv%dscale, ierr)
    call make_field_block_scale(op%mat, spread(sfo_n1, 1, op%nf), slv%dscale, comm, my_id, op%label)
    slv%scaled = .true.
  end subroutine sf_solver_block_scale

  !> op's penalty rows as Dirichlet rows (decouple_penalty_rows), with their
  !! right-hand-side factor kept in slv; before the scaling.
  subroutine sf_solver_penalty_rows(slv, op, comm, my_id)
    type(block_solver_t), intent(inout) :: slv
    type(sf_op_t), intent(inout) :: op
    integer, intent(in) :: comm, my_id
    call decouple_penalty_rows(op%mat, spread(sfo_n1, 1, op%nf), slv%pfac, slv%penal, comm, my_id, op%label)
  end subroutine sf_solver_penalty_rows

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
      ": mean ", dble(slv%its_sum) / dble(slv%nsolve), " its, max ", slv%its_max, " (", slv%nsolve, " solves)"
    if (slv%nfail > 0) then
      write(*,'(A,I0,A)') ", WARNING ", slv%nfail, " not converged"
    else
      write(*,*)
    endif
  end subroutine sf_solver_report

  !> Mode split: a heading line, then one line per family on the rank 0 of
  !! comm (the full system's communicator): each block's mean inner
  !! iterations and solve time, the rest of the application and the
  !! scatters, the maxima over the family's ranks. The scatter back waits for
  !! every family, so the critical family is the one with the smallest
  !! scatter time. Collective on comm. (mod_petsc_pc_sf keeps its own
  !! family_report, whose log must not change.)
  subroutine sf_solver_family_report(name, slv, t_apply, t_scat, comm)
    use mod_petsc_pc_sf_ops, only: sfo_comm, sfo_fam, sfo_nfam
    character(len=*), intent(in)     :: name      !< the method, for the heading
    type(block_solver_t), intent(in) :: slv(:)    !< the blocks, in report order
    real*8, intent(in)  :: t_apply, t_scat        !< this rank's apply and scatter times
    integer, intent(in) :: comm
    real*8, allocatable :: v(:), tab(:, :), tab_g(:, :)
    integer :: nb, nv, i_rest, i_scat, i_fail, i_nrk, b, f, me, fme, fnp, mpierr

    ! v = (its(1:nb), solve s(1:nb), rest, scatter, unconverged solves, ranks)
    nb = size(slv)
    i_rest = 2 * nb + 1; i_scat = i_rest + 1; i_fail = i_scat + 1; i_nrk = i_fail + 1
    nv = i_nrk
    allocate(v(nv), tab(nv, sfo_nfam), tab_g(nv, sfo_nfam))
    v = 0.d0
    do b = 1, nb
      if (slv(b)%nsolve > 0) v(b) = dble(slv(b)%its_sum) / dble(slv(b)%nsolve)
      v(nb + b) = slv(b)%t_sum
    enddo
    v(i_rest) = t_apply - t_scat - sum(v(nb + 1:2 * nb))
    v(i_scat) = t_scat
    v(i_fail) = dble(sum(slv%nfail))
    ! the maxima over the family; the rank count is the family's size, not a maximum
    call MPI_Allreduce(MPI_IN_PLACE, v, i_nrk - 1, MPI_DOUBLE_PRECISION, MPI_MAX, sfo_comm, mpierr)
    call MPI_Comm_rank(sfo_comm, fme, mpierr)
    call MPI_Comm_size(sfo_comm, fnp, mpierr)
    v(i_nrk) = dble(fnp)
    ! one column per family, from its first rank
    tab = 0.d0
    if (fme == 0) tab(:, sfo_fam) = v
    call MPI_Reduce(tab, tab_g, nv * sfo_nfam, MPI_DOUBLE_PRECISION, MPI_SUM, 0, comm, mpierr)
    call MPI_Comm_rank(comm, me, mpierr)
    if (me /= 0) return

    write(*,'(A)') "[Physics PC]   "//name//" mode split per family: inner its / solve s (max over its "// &
      "ranks), rest of the application, scatters incl. the wait for the slowest family"
    do f = 1, sfo_nfam
      write(*,'(A,I0,A,I0,A)', advance="no") "[Physics PC]     family ", f, " (", nint(tab_g(i_nrk, f)), " rk):"
      do b = 1, nb
        write(*,'(A,F5.2,A,F7.3)', advance="no") "  "//trim(slv(b)%label)//" ", tab_g(b, f), " /", tab_g(nb + b, f)
      enddo
      write(*,'(A,F7.3,A,F7.3,A)', advance="no") "  rest ", tab_g(i_rest, f), "  scatter ", tab_g(i_scat, f), " s"
      if (tab_g(i_fail, f) > 0.d0) then
        write(*,'(A,I0,A)') ", WARNING ", nint(tab_g(i_fail, f)), " not converged"
      else
        write(*,*)
      endif
    enddo
  end subroutine sf_solver_family_report

  !> Between a packed vector p = [h(1) local | h(2) local | ...] and its
  !! fields h(:), rank-local.
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

  !> 1 + zeta as the Jacobian forms it (zeta rescaled for a variable dt).
  real*8 function sf_opz()
    use phys_module, only: time_evol_zeta, tstep, tstep_prev
    sf_opz = 1.0d0 + time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  end function sf_opz

end module mod_petsc_pc_sf_solver
