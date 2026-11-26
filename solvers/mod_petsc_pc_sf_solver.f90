!> The solver of one diagonal block of the SF sweep (mod_petsc_pc_sf), with
!! two backends: FGMRES around one V-cycle of the geometric multigrid
!! (mod_petsc_pc_gmg), or PREONLY + MUMPS LU as the exact reference. The GMG
!! configuration is fixed here, per block, rather than in the namelist.
module mod_petsc_pc_sf_solver
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx,  only: physics_pc_mumps_mem
  use mod_petsc_pc_blocks,       only: pc_print_block_setup
  use mod_petsc_direct_solver,   only: petsc_configure_direct_solver, petsc_setup_wrapped_solver
  use mod_petsc_pc_gmg,          only: GMG_SM_RINGS, GMG_SM_LINES, GMG_SM_ZEBRA_RINGS
#include "petsc_f90_compat.h"
  implicit none
  private

  integer, parameter, public :: SF_LU  = 1
  integer, parameter, public :: SF_GMG = 2

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

  public :: sf_solver_setup, sf_solver_apply, sf_solver_reset_counters, sf_solver_report
  public :: sf_split_parts, sf_opz, sf_vpar_stage, sf_vpar_w_assembled

contains

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

  !> Does the SF sweep need v_par's W terms (stages b0 / b1)?
  logical function sf_vpar_w_assembled()
    use mod_parameters, only: var_Vpar
    use phys_module,    only: use_physics_pc
    sf_vpar_w_assembled = (var_Vpar > 0) .and. use_physics_pc .and. &
                          (sf_vpar_stage() == SF_VPAR_B0 .or. sf_vpar_stage() == SF_VPAR_B1)
  end function sf_vpar_w_assembled

  !> Configure slv to solve A. The KSP is created once and re-pointed at the
  !! refilled operator on every rebuild, so MUMPS and the GMG reuse their
  !! symbolic phases. The GMG KSP takes PETSc options with the prefix
  !! sf_gmg<gmg_inst>_.
  subroutine sf_solver_setup(slv, A, backend, label, comm, my_id, rtol, gmg_inst, nfields, &
                             smoother, maxits, pre0, post0, nsmooth_c, axis_rings)
    use mod_petsc_pc_gmg, only: gmg_select, gmg_is_ready, gmg_build_prolongations, &
                                gmg_setup_operator, gmg_pc_apply_1, gmg_pc_apply_2, &
                                gmg_pc_apply_3, gmg_pc_apply_4, gmg_pc_apply_5, gmg_opts_t
    type(block_solver_t), intent(inout) :: slv
    Mat, intent(in)              :: A
    integer, intent(in)          :: backend, comm, my_id
    character(len=*), intent(in) :: label
    real*8, intent(in)           :: rtol
    integer, intent(in)          :: gmg_inst   !< hierarchy instance (1..5)
    integer, intent(in)          :: nfields    !< fields packed per node in A
    integer, intent(in)          :: smoother   !< SF_GMG_SMOOTHER_*
    integer, intent(in)          :: maxits     !< FGMRES iteration cap
    integer, intent(in), optional :: pre0, post0, nsmooth_c, axis_rings
    PC :: pc
    PetscErrorCode :: ierr
    logical :: ok, fresh
    character(len=24) :: tstr
    character(len=16) :: prefix
    type(gmg_opts_t) :: o

    fresh       = .not. slv%created
    slv%backend = backend
    slv%label   = label

    select case (backend)

    case (SF_LU)
      write(prefix, '(A,I0,A)') "sf_lu", gmg_inst, "_"
      if (fresh) then
        call KSPCreate(comm, slv%ksp, ierr)
        call KSPSetType(slv%ksp, KSPPREONLY, ierr)
        call KSPSetOptionsPrefix(slv%ksp, trim(prefix), ierr)
      endif
      call KSPSetOperators(slv%ksp, A, A, ierr)
      call KSPGetPC(slv%ksp, pc, ierr)
#if defined(PETSC_HAVE_MUMPS)
      block
        PetscBool        :: has_icntl14
        PCFailedReason   :: pcreason
        if (fresh) then
          call PetscOptionsHasName(PETSC_NULL_OPTIONS, trim(prefix), '-mat_mumps_icntl_14', has_icntl14, ierr)
          if (.not. has_icntl14) then
            call PetscOptionsSetValue(PETSC_NULL_OPTIONS, '-'//trim(prefix)//'mat_mumps_icntl_14', '200', ierr)
          endif
          call petsc_configure_direct_solver(pc)
          call KSPSetFromOptions(slv%ksp, ierr)
        endif
        call KSPSetUp(slv%ksp, ierr)
        call petsc_setup_wrapped_solver(pc)
        call PCGetFailedReason(pc, pcreason, ierr)
        if (pcreason == PC_FACTOR_OUTMEMORY) then
          if (my_id == 0) write(*,'(A,A,A)') "[Physics PC]   ", trim(label), &
            ": MUMPS out of static workspace; expanding ICNTL(14)=500 and retrying."
          call PetscOptionsSetValue(PETSC_NULL_OPTIONS, '-'//trim(prefix)//'mat_mumps_icntl_14', '500', ierr)
          call PCReset(pc, ierr)
          call KSPSetOperators(slv%ksp, A, A, ierr)
          call petsc_configure_direct_solver(pc)
          call KSPSetUp(slv%ksp, ierr)
          call petsc_setup_wrapped_solver(pc)
        endif
        call pc_print_block_setup(comm, label, "PREONLY + LU (MUMPS)")
        if (fresh) call physics_pc_mumps_mem(slv%ksp, label, my_id)
      end block
#else
      block
        integer :: np, mpierr
        KSP :: inner_ksp
        PC  :: inner_pc
        call MPI_Comm_size(comm, np, mpierr)
        if (np == 1) then
          call PCSetType(pc, PCREDUNDANT, ierr)
          call KSPSetUp(slv%ksp, ierr)
          call PCRedundantGetKSP(pc, inner_ksp, ierr)
          call KSPSetType(inner_ksp, KSPPREONLY, ierr)
          call KSPGetPC(inner_ksp, inner_pc, ierr)
          call PCSetType(inner_pc, PCLU, ierr)
          call PCFactorSetMatSolverType(inner_pc, MATSOLVERPETSC, ierr)
          call PCFactorSetMatOrderingType(inner_pc, MATORDERINGND, ierr)
          call PCFactorSetShiftType(inner_pc, MAT_SHIFT_NONZERO, ierr)
          call KSPSetUp(inner_ksp, ierr)
          call pc_print_block_setup(comm, label, "PREONLY + LU (PETSc)")
        else
          call PCSetType(pc, PCTELESCOPE, ierr)
          call PCTelescopeSetReductionFactor(pc, np, ierr)
          call KSPSetUp(slv%ksp, ierr)
          call PCTelescopeGetKSP(pc, inner_ksp, ierr)
          if (inner_ksp%v /= 0 .and. inner_ksp%v /= -2) then
            call KSPSetType(inner_ksp, KSPPREONLY, ierr)
            call KSPGetPC(inner_ksp, inner_pc, ierr)
            call PCSetType(inner_pc, PCLU, ierr)
            call PCFactorSetMatSolverType(inner_pc, MATSOLVERPETSC, ierr)
            call PCFactorSetMatOrderingType(inner_pc, MATORDERINGND, ierr)
            call PCFactorSetShiftType(inner_pc, MAT_SHIFT_NONZERO, ierr)
            call KSPSetUp(inner_ksp, ierr)
          endif
          call pc_print_block_setup(comm, label, "PREONLY + Telescope LU (PETSc)")
        endif
      end block
#endif

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
        call KSPSetType(slv%ksp, KSPFGMRES, ierr)
        call KSPGMRESSetRestart(slv%ksp, maxits, ierr)
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
        end select
        call PCShellSetName(pc, trim(label)//" GMG V-cycle", ierr)
      endif
      call KSPSetOperators(slv%ksp, A, A, ierr)
      call KSPSetUp(slv%ksp, ierr)
      write(tstr,'(ES9.2)') rtol
      call pc_print_block_setup(comm, label, "FGMRES + GMG V-cycle, rtol "//trim(adjustl(tstr)))

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
#if PETSC_VERSION_LT(3,23,0)
    if (reason < 0) slv%nfail = slv%nfail + 1
#else
    if (reason%v < 0) slv%nfail = slv%nfail + 1
#endif
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
      ": mean ", dble(slv%its_sum) / dble(slv%nsolve), " its, max ", slv%its_max, " (", slv%nsolve, " solves)"
    if (slv%nfail > 0) then
      write(*,'(A,I0,A)') ", WARNING ", slv%nfail, " not converged"
    else
      write(*,*)
    endif
  end subroutine sf_solver_report

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
    if (tstep + tstep_prev > 0.d0) then
      sf_opz = 1.0d0 + time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
    else
      sf_opz = 1.0d0
    endif
  end function sf_opz

end module mod_petsc_pc_sf_solver
