module mod_petsc_pc_sf_solver
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mumps_mem, pcev_psipc
  use mod_petsc_pc_blocks,      only: pc_print_block_setup, report_operator_density
  implicit none
  private

  !--------------------------------------------------------------------
  !> Block -> solver abstraction for the production SFM2 path.
  !!
  !! Every diagonal block of the LDU sweep is solved through ONE type, so
  !! "which solver does this block use" is a value rather than a code path.
  !! That is what makes extending the M blocks to multigrid a configuration
  !! change later: pick a different backend for that block, nothing else moves.
  !!
  !! Backends
  !! --------
  !!   SF_LU        PREONLY + LU (MUMPS). The reference, and the only backend
  !!                that is exact.
  !!   SF_GMG       FGMRES + PCSHELL on one V-cycle of the C1 geometric
  !!                multigrid (mod_petsc_pc_gmg), hierarchy instance gmg_inst.
  !!   SF_ETASCHUR  pair_psi only: one application of the j-first lower block
  !!                factorisation with the eta-scaled Schur approximation
  !!
  !!                  z_j   = B_33^-1 r_j
  !!                  z_psi = Shat^-1 (r_psi - B_13 z_j),
  !!                  Shat  = B_11 - diag(B_13/B_33) B_31,
  !!
  !!                optionally wrapped in FGMRES on the raw pair. Shat is
  !!                itself a block_solver_t, so LU-on-Shat and GMG-on-Shat are
  !!                the same two backend values rather than two special cases.
  !!                Requires eta_num = 0: with hyper-resistivity B_13 gains
  !!                eta_num K_1, whose diagonal dominates the row ratio, and
  !!                the true correction is a discrete biharmonic that no row
  !!                scaling represents (workstream C S3.6).
  !!
  !! The GMG smoother, axis-ring extent and boundary-drop are compile-time
  !! constants below rather than namelist entries. They are the values the
  !! workstream D/G measurements were taken at; the production path does not
  !! offer them as knobs.
  !--------------------------------------------------------------------

  integer, parameter, public :: SF_LU       = 1
  integer, parameter, public :: SF_GMG      = 2
  integer, parameter, public :: SF_ETASCHUR = 3

  !--- Fixed GMG configuration for this path (workstream D S12, G S3) -------
  integer, parameter, public :: SF_GMG_SMOOTHER   = 5   !< radial-line block Jacobi
  integer, parameter, public :: SF_GMG_AXIS_RINGS = 3   !< rings folded into the axis block
  integer, parameter, public :: SF_GMG_NSMOOTH    = 0   !< 0 = the smoother's own default
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
  public :: sf_etaschur_setup, sf_backend_name, sf_split_halves

  !--- SF_ETASCHUR state ---------------------------------------------------
  ! A PCSHELL callback cannot carry a Fortran-typed context, so the operands
  ! live here. Only pair_psi uses this backend, so one instance is enough --
  ! the same reason the research path keeps its psc_* state at module scope.
  Mat, save  :: es_Shat                    !< B_11 - diag(B_13/B_33) B_31
  Vec, save  :: es_zpsi, es_zj, es_t, es_rp, es_rj
  type(block_solver_t), save :: es_Sslv    !< the solver for Shat (LU or GMG)
  type(block_solver_t), save :: es_Mj      !< the LU of B_33
  logical, save :: es_ready = .false., es_vecs_ready = .false.

contains

  !> Human-readable backend name, for the one setup line each block prints.
  function sf_backend_name(backend) result(s)
    integer, intent(in) :: backend
    character(len=64)   :: s
    select case (backend)
    case (SF_LU);       s = "PREONLY + LU (MUMPS)"
    case (SF_GMG);      s = "FGMRES + SHELL[C1 GMG V-cycle]"
    case (SF_ETASCHUR); s = "SHELL[j-first eta-scaled Schur]"
    case default;       s = "UNKNOWN"
    end select
  end function sf_backend_name

  !--------------------------------------------------------------------
  !> Configure slv to solve A. Idempotent across PC rebuilds: the KSP is
  !! created once and re-pointed at the (refilled, pattern-frozen) operator,
  !! so MUMPS and the GMG both reuse their symbolic phases.
  !--------------------------------------------------------------------
  subroutine sf_solver_setup(slv, A, backend, label, comm, my_id, rtol, gmg_inst, maxits)
    use mod_petsc_pc_gmg, only: gmg_select, gmg_is_ready, gmg_build_prolongations, &
                                gmg_setup_operator, gmg_pc_apply_1, gmg_pc_apply_2, &
                                gmg_pc_apply_3, gmg_pc_apply_4
    type(block_solver_t), intent(inout) :: slv
    Mat, intent(in)              :: A
    integer, intent(in)          :: backend, comm, my_id
    character(len=*), intent(in) :: label
    real*8, intent(in)           :: rtol
    integer, intent(in), optional :: gmg_inst
    integer, intent(in), optional :: maxits   !< SF_GMG only. 0 = PREONLY, i.e.
                                              !< exactly one V-cycle with no
                                              !< Krylov around it, which is what
                                              !< Shat gets inside SF_ETASCHUR.

    PC :: pc
    PetscErrorCode :: ierr
    logical :: ok, fresh
    integer :: mits
    character(len=24) :: tstr

    fresh        = .not. slv%created
    mits         = SF_GMG_MAXITS
    if (present(maxits)) mits = maxits
    slv%backend  = backend
    slv%label    = label
    if (present(gmg_inst)) slv%gmg_inst = gmg_inst

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
      !--- the hierarchy: built once per instance, then refilled per rebuild
      call gmg_select(slv%gmg_inst)
      if (.not. gmg_is_ready()) then
        call gmg_build_prolongations(A, comm, my_id, nfields(slv%gmg_inst), ok)
        if (.not. ok) then
          if (my_id == 0) write(*,'(A,A,A)') &
            "[Physics PC]   FATAL: the GMG backend for ", trim(label), &
            " needs the structured flux-surface grid."
          call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
      endif
      call gmg_setup_operator(A, comm, my_id, tag=trim(label), &
                              smoother=SF_GMG_SMOOTHER, nsmooth=SF_GMG_NSMOOTH)
      call gmg_select(1)

      !--- the Krylov wrapper. Rebuilt rather than re-pointed: the PCSHELL
      !--- holds no operator reference, so there is nothing to keep.
      if (.not. fresh) call KSPDestroy(slv%ksp, ierr)
      call KSPCreate(comm, slv%ksp, ierr)
      call KSPSetOperators(slv%ksp, A, A, ierr)
      if (mits > 0) then
        call KSPSetType(slv%ksp, KSPFGMRES, ierr)
        call KSPGMRESSetRestart(slv%ksp, max(mits, 2), ierr)
        call KSPSetTolerances(slv%ksp, rtol, 1.d-50, 1.d6, mits, ierr)
      else
        call KSPSetType(slv%ksp, KSPPREONLY, ierr)
      endif
      call KSPGetPC(slv%ksp, pc, ierr)
      call PCSetType(pc, PCSHELL, ierr)
      select case (slv%gmg_inst)
      case (1); call PCShellSetApply(pc, gmg_pc_apply_1, ierr)
      case (2); call PCShellSetApply(pc, gmg_pc_apply_2, ierr)
      case (3); call PCShellSetApply(pc, gmg_pc_apply_3, ierr)
      case (4); call PCShellSetApply(pc, gmg_pc_apply_4, ierr)
      end select
      call PCShellSetName(pc, trim(label)//" C1 GMG V-cycle", ierr)
      call KSPSetUp(slv%ksp, ierr)
      if (mits > 0) then
        write(tstr,'(ES9.2)') rtol
        call pc_print_block_setup(comm, label, &
          trim(sf_backend_name(backend))//", rtol "//trim(adjustl(tstr)))
      else
        call pc_print_block_setup(comm, label, "PREONLY + SHELL[one C1 GMG V-cycle]")
      endif

    case default
      if (my_id == 0) write(*,'(A,I0)') &
        "[Physics PC]   FATAL: sf_solver_setup got an unknown backend ", backend
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end select

    slv%created = .true.

  contains
    !> Fields per node in hierarchy k's operator: the packed pairs carry two,
    !! the scalar blocks one. Drives the prolongation's block size.
    integer function nfields(k)
      integer, intent(in) :: k
      nfields = merge(2, 1, k == 1)
    end function nfields
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
  !> Build the SF_ETASCHUR shell for pair_psi and point slv at it.
  !!
  !! K_pj is the shell KSP's nominal operator: FGMRES(psi_outer > 0) iterates
  !! on the raw packed pair while the shell supplies the preconditioner. The
  !! shell reads the UNSCALED g_ctx blocks, so the caller must not block-scale
  !! K_pj when this backend is selected.
  !--------------------------------------------------------------------
  subroutine sf_etaschur_setup(slv, K_pj, shat_backend, comm, my_id, rtol, outer)
    use phys_module, only: eta_num, tstep
    type(block_solver_t), intent(inout) :: slv
    Mat, intent(in)     :: K_pj
    integer, intent(in) :: shat_backend, comm, my_id
    real*8, intent(in)  :: rtol
    integer, intent(in) :: outer          !< FGMRES budget on the pair; 0 = PREONLY

    Vec :: d13, d33
    Mat :: T31
    PC  :: pc
    PetscErrorCode :: ierr
    PetscScalar, pointer :: a13(:), a33(:)
    PetscInt :: n, k
    real*8   :: dmax, emin, emax, ebuf(2), erbuf(2)
    integer  :: mpierr
    character(len=24) :: tstr

    if (eta_num /= 0.d0) then
      ! Not a fallback: refuse rather than mis-measure (workstream C S3.6).
      if (my_id == 0) write(*,'(A)') &
        "[Physics PC]   FATAL: the eta-Schur pair_psi backend needs eta_num = 0 "// &
        "(it is invalid with hyper-resistivity)."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif

    !--- B_33^-1, exact. Its error re-enters through B_13 ~ dt, so it must be.
    call sf_solver_setup(es_Mj, g_ctx%B_33, SF_LU, "pair_psi B_33 (1/R mass)", &
                         comm, my_id, rtol)

    !--- r = diag(B_13)/diag(B_33), zero on rows with no mass (boundary rows)
    call MatGetLocalSize(g_ctx%B_33, n, PETSC_NULL_INTEGER, ierr)
    call MatCreateVecs(g_ctx%B_33, d33, PETSC_NULL_VEC, ierr)
    call VecDuplicate(d33, d13, ierr)
    call MatGetDiagonal(g_ctx%B_33, d33, ierr)
    call MatGetDiagonal(g_ctx%B_13, d13, ierr)
    call VecNorm(d33, NORM_INFINITY, dmax, ierr)
    call VecGetArray(d13, a13, ierr)
    call VecGetArrayRead(d33, a33, ierr)
    emin = huge(1.d0); emax = -huge(1.d0)
    do k = 1, n
      if (abs(a33(k)) > 1.d-12 * dmax) then
        a13(k) = a13(k) / a33(k)
        emin = min(emin, -a13(k) / tstep); emax = max(emax, -a13(k) / tstep)
      else
        a13(k) = 0.d0
      endif
    enddo
    call VecRestoreArrayRead(d33, a33, ierr)
    call VecRestoreArray(d13, a13, ierr)
    ebuf = [-emin, emax]
    call MPI_Allreduce(ebuf, erbuf, 2, MPI_DOUBLE_PRECISION, MPI_MAX, comm, mpierr)
    emin = -erbuf(1); emax = erbuf(2)

    !--- Shat = B_11 - diag(r) B_31. es_Shat lives for the run: the first build
    !--- fixes the union pattern, later builds refill it in place, so the GMG
    !--- sees the same Mat and reuses its PtAP symbolic phase.
    call MatDuplicate(g_ctx%B_31, MAT_COPY_VALUES, T31, ierr)
    call MatDiagonalScale(T31, d13, PETSC_NULL_VEC, ierr)
    if (es_ready) then
      call MatZeroEntries(es_Shat, ierr)
      call MatAXPY(es_Shat, 1.0d0, g_ctx%B_11, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(es_Shat, -1.0d0, T31, SUBSET_NONZERO_PATTERN, ierr)
    else
      call MatDuplicate(g_ctx%B_11, MAT_COPY_VALUES, es_Shat, ierr)
      call MatAXPY(es_Shat, -1.0d0, T31, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatSetOption(es_Shat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
    endif
    call MatDestroy(T31, ierr)
    call VecDestroy(d13, ierr)
    call VecDestroy(d33, ierr)

    if (my_id == 0) write(*,'(A,ES10.3,A,ES10.3,A)') &
      "[Physics PC]   pair_psi eta-Schur: theta*eta_T = -r/tstep in [", emin, ", ", emax, "]"
    if (.not. es_ready) call report_operator_density(es_Shat, "Shat (eta-scaled psi Schur)", my_id)

    if (.not. es_vecs_ready) then
      call MatCreateVecs(es_Shat, es_zpsi, es_t, ierr)
      call VecDuplicate(es_zpsi, es_zj, ierr)
      call VecDuplicate(es_zpsi, es_rp, ierr)
      call VecDuplicate(es_zpsi, es_rj, ierr)
      es_vecs_ready = .true.
    endif

    !--- Shat's own solver: this is the whole point of the abstraction. LU and
    !--- one GMG V-cycle differ only in this one value.
    !--- maxits = 0: Shat gets exactly ONE V-cycle, with no Krylov of its own.
    !--- The Krylov that matters is the FGMRES on the raw pair below; nesting a
    !--- second one here would change the preconditioner, not just its cost.
    call sf_solver_setup(es_Sslv, es_Shat, shat_backend, "pair_psi Shat", &
                         comm, my_id, rtol, gmg_inst=2, maxits=0)

    !--- the pair_psi KSP: FGMRES on the raw pair around the shell. One
    !--- application of the factorisation alone loses ~10 outer its at tstep 10
    !--- even with Shat exact, so outer > 0 is the production setting.
    if (slv%created) call KSPDestroy(slv%ksp, ierr)
    call KSPCreate(comm, slv%ksp, ierr)
    call KSPSetOperators(slv%ksp, K_pj, K_pj, ierr)
    if (outer > 0) then
      call KSPSetType(slv%ksp, KSPFGMRES, ierr)
      call KSPGMRESSetRestart(slv%ksp, max(outer, 2), ierr)
      call KSPSetTolerances(slv%ksp, rtol, 1.d-50, 1.d6, outer, ierr)
    else
      call KSPSetType(slv%ksp, KSPPREONLY, ierr)
    endif
    call KSPGetPC(slv%ksp, pc, ierr)
    call PCSetType(pc, PCSHELL, ierr)
    call PCShellSetApply(pc, es_apply, ierr)
    call PCShellSetName(pc, "pair_psi eta-scaled j-first Schur", ierr)
    call KSPSetUp(slv%ksp, ierr)

    slv%backend = SF_ETASCHUR
    slv%created = .true.
    slv%scaled  = .false.
    es_ready    = .true.

    write(tstr,'(ES9.2)') rtol
    call pc_print_block_setup(comm, trim(slv%label), &
      "FGMRES + "//trim(sf_backend_name(SF_ETASCHUR))//", Shat by "// &
      trim(sf_backend_name(shat_backend))//", rtol "//trim(adjustl(tstr)))
  end subroutine sf_etaschur_setup

  !--------------------------------------------------------------------
  !> PCSHELL apply for SF_ETASCHUR:  z = [Shat^-1 (r_psi - B_13 B_33^-1 r_j);
  !!                                      B_33^-1 r_j]
  !--------------------------------------------------------------------
  subroutine es_apply(pc, rvec, zvec, ierr)
    PC  :: pc
    Vec :: rvec, zvec
    PetscErrorCode :: ierr

    call PetscLogEventBegin(pcev_psipc, ierr)
    call sf_split_halves(rvec, es_rp, es_rj, .false.)
    call sf_solver_apply(es_Mj, es_rj, es_zj, ierr)        ! z_j = B_33^-1 r_j

    call MatMult(g_ctx%B_13, es_zj, es_t, ierr)
    call VecAYPX(es_t, -1.0d0, es_rp, ierr)                ! t = r_psi - B_13 z_j

    call sf_solver_apply(es_Sslv, es_t, es_zpsi, ierr)     ! z_psi = Shat^-1 t

    call sf_split_halves(zvec, es_zpsi, es_zj, .true.)     ! z = [z_psi; z_j]
    call PetscLogEventEnd(pcev_psipc, ierr)
    ierr = 0
  end subroutine es_apply

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
