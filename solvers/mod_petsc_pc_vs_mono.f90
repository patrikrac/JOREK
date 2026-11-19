!> The monolithic multigrid preconditioner (use_physics_pc,
!! physics_pc_method = "vs_mono"): one V-cycle of the geometric multigrid of
!! mod_petsc_pc_gmg on the whole Jacobian, all variables in one operator and
!! in every smoother block, beside the split-field sweep of mod_petsc_pc_sf,
!! the vorticity-space Schur PC of mod_petsc_pc_vs and the Alfven-block PC of
!! mod_petsc_pc_vs_alfven. The name groups it with the VS variants; it forms
!! no Schur complement and has no force operator W, no predictor and no
!! corrector. A_p is the Jacobian in the packed layout of mod_petsc_pc_sf_ops,
!! fields in the order (u, omega, psi, j, rho, T [, v_par]); u, the first, is
!! the reference of the symmetric field scaling (sf_solver_block_scale).
!!
!! One application y of the preconditioner to x:
!!   1. A_p y = x by one V-cycle (physics_pc_sf_pair_w = "gmg": PREONLY around
!!      the GMG's shell PC, hierarchy instance 1, options prefix sf_gmg1_), or
!!      exactly ("lu": MUMPS on A_p, the reference)
!!   2. the full Jacobian's penalty rows by back-substitution
!! sf_solver_setup sets the one V-cycle (one_cycle) before the options:
!! -sf_gmg1_ksp_type fgmres puts FGMRES to physics_pc_sf_rtol around it again.
!!
!! With physics_pc_sf_harm_couple = .t. (the default, global layout) A_p is
!! the Jacobian, reordered, and the V-cycle the only approximation. With .f.
!! A_p has no entries between different |n| and a harmonic's cos and sin per
!! smoother block, and physics_pc_sf_mode_split (default .t.) puts one |n|
!! family per rank group, as in the SF sweep.
!!
!! The hierarchy is configured as the SF sweep's pair_w: flux-surface ring
!! blocks (the psi - u coupling runs along the field lines at large dt),
!! V(SF_W_PRE0, SF_W_POST0) on level 0, SF_W_NSC steps per side below, the
!! automatic axis block. PETSc options for experiments:
!!   -vs_mono_smoother rings | lines | zebra   the block smoother
!!   -vs_mono_pre0 k, -vs_mono_post0 k         level-0 smoothing steps
!!   -vs_mono_nsc k                            steps per side below level 0
!!   -vs_mono_axis_rings k                     axis block rings 0..k, -2 automatic
!!   -vs_mono_harm_pair 1                      a harmonic's cos and sin per smoother
!!                                             block on the exact (cross-|n|) A_p
!! cbm18 17x36 (n = 0, 6, model199, np 1): 14 outer iterations at dt 3
!! against 18 for the SF sweep; 40 at dt 30, where the SF sweep does not
!! converge in 200 at its default inner rtol (148 at 1e-2). One V-cycle costs
!! about one SF application, mostly the 7 matvecs of the level-0 GMRES
!! smoother.
!!
!! MEMORY. A ring block holds 4 n_f s rows per node (n_f = 6 or 7 fields, s
!! slots per block: n_tor with all slots, 2 with cos / sin blocks) and its
!! factor about five dense node blocks per node in single precision
!! (mod_gmg_ring_btd): 5 (4 n_f s)^2 n_tor / s words per node, quadratic in
!! n_tor with all slots, linear (constant per harmonic) with cos / sin
!! blocks, and n_f^2 = 36 or 49 against pair_w's 16. cbm18 17x36, n_tor 3:
!! 74 MB with all slots, 41 MB with cos / sin, 44 MB for all SF blocks
!! together. With many harmonics use physics_pc_sf_harm_couple = .f. (A_p and
!! its levels |n|-diagonal too) or -vs_mono_harm_pair 1 (the smoother blocks
!! only: A_p and its Galerkin levels keep every cross-|n| entry, (4 n_f
!! n_tor)^2 per node pair as the Jacobian). A_p is a copy of the whole
!! Jacobian (AIJ plus the refill's value array and map). The axis block and
!! the coarsest level have 1.5 - 1.75x the rows of pair_w's, and the
!! automatic axis block counts n_f in its ring-0 rows, so ring 1 can join
!! where pair_w's stays at ring 0. Their LUs are redundant on the ranks
!! owning their rows, unless the J-sector solve splits the axis block across
!! ranks or the coarsest level exceeds CRS_DIST_ROWS rows (one distributed
!! MUMPS LU).
!!
!! PARALLEL. Global reductions per application: those of the level-0 GMRES
!! smoother, two per step plus one per smoothing call (13 for V(0, 6)); the
!! coarser levels smooth by Richardson without any (multi-field ring
!! hierarchies on X-point grids keep GMRES on every level). PREONLY adds
!! none. The redundant LUs gather among their members only; the J-sector
!! axis solve's scatters and a distributed coarse LU involve every rank of
!! the hierarchy. A ring block cut by a rank boundary is completed by ghost
!! rows on every rank holding a part (restricted additive Schwarz), which
!! copies their rows of A_p and factors the whole ring there: with fewer
!! nodes per rank than a ring has, rings are factored several times. In the
!! mode split the exchanges on the full system's layout are the gather, the
!! scatter and step 2's product, which holds the penalty rows alone
!! (full_penalty_rows' compact option).
!!
!! Penalty rows: with eliminated boundary rows A_p holds the Jacobian's
!! diagonal-only rows there, and there is no step 2. Otherwise A_p's penalty
!! rows are decoupled as Dirichlet rows (decouple_penalty_rows, a test of the
!! whole row) and step 2 restores them. Limit: model600's Mach-1 rows of v_par
!! have penalty-size entries in the T and u columns, both inside A_p, which
!! the whole-row test can miss; such rows stay in A_p (its v_par scale, the
!! coarse spaces), and step 2 back-substitutes them as whole rows.
!!
!! The variables' pieces of x and y are host-array views
!! (VecCreateMPIWithArray) into the storage of A_p's packed right-hand side
!! and solution: the gather fills step 1's right-hand side, which step 1
!! scales in place, and step 1 writes y's pieces. A device Vec type would
!! break the aliasing.
module mod_petsc_pc_vs_mono
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var, var_psi, var_u, var_zj, var_w, var_rho, var_T, var_Vpar
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, pcev_extract, pcev_fact_w, pcev_solve_w, pcev_apply
  use mod_petsc_pc_blocks, only: penalty_rows_t, full_penalty_rows, penalty_backsub
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_ops, only: sf_op_t, sfo_layout, sfo_build, sfo_fill, sfo_vars_in, sfo_vars_out, &
       sfo_comm, sfo_define, sfo_n1, SFO_NFMAX
  implicit none
  private

  interface
    !> PETSc's Fortran stub (src/sys/ftn-custom/zsys.c); the petsc module
    !! declares no interface for it
    subroutine PetscObjectStateIncrease(obj, ierr)
      import
      Vec :: obj
      PetscErrorCode :: ierr
    end subroutine PetscObjectStateIncrease
  end interface

  logical, save :: vsm_init_done = .false.     !< the namelist and the options resolved and checked (vsm_init)
  logical, save :: vsm_first     = .true.      !< the next build is the first: pattern, layout, vectors
  logical, save :: famode        = .false.     !< mode split
  logical, save :: harm_all      = .true.      !< physics_pc_sf_harm_couple
  integer, save :: comm_g = MPI_COMM_NULL      !< the full system's communicator

  type(block_solver_t), save :: slv
  integer, save :: bk = SF_GMG

  !> A_p's fields in their packed order, v_par seventh; nfm (= n_var) of them
  integer, parameter :: M_VAR(7) = [var_u, var_w, var_psi, var_zj, var_rho, var_T, var_Vpar]
  integer, save :: nfm = 6

  type(sf_op_t), save :: o_a                   !< A_p
  type(penalty_rows_t), save :: pen            !< the full Jacobian's penalty rows

  !> the hierarchy's configuration, the SF sweep's pair_w unless the -vs_mono_*
  !! options change it (vsm_init)
  integer, save :: m_smoother = SF_GMG_SMOOTHER_RINGS
  character(len=8), save :: m_smoother_name = "rings"
  integer, save :: m_pre0 = SF_W_PRE0, m_post0 = SF_W_POST0, m_nsc = SF_W_NSC
  integer, save :: m_axis = SF_GMG_AXIS_RINGS_PAIRS

  !> this rank's wall time in vsm_apply (step 2 included) and in its gather /
  !! scatter since the last report (the mode split's per-family report)
  real*8, save :: t_apply = 0.d0, t_scat = 0.d0

  !> A_p's packed right-hand side and solution (fields in M_VAR order, each
  !! sfo_n1 long) and their storage, which sv_x / sv_y (by JOREK variable) share
  PetscScalar, allocatable, target, save :: rhs_store(:), sol_store(:)
  Vec, save :: rhs_p, sol_p
  Vec, save :: sv_x(SFO_NFMAX), sv_y(SFO_NFMAX)
  logical, save :: vecs_ready = .false.

  public :: vsm_init, vsm_build, vsm_apply, vsm_report

contains

  !> Resolve the namelist and the -vs_mono_* options once and refuse
  !! configurations the method cannot run. Called before the first PC build
  !! (petsc_assemble_pc_matrices).
  subroutine vsm_init(my_id)
    use phys_module, only: physics_pc_sf_pair_w, physics_pc_sf_rtol, physics_pc_sf_harm_couple, &
                           physics_pc_sf_mode_split
    use mod_parameters, only: n_tor
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr
    PetscBool :: set
    PetscInt :: iv
    character(len=32) :: s
    character(len=*), parameter :: P = "[Physics PC]   "

    if (vsm_init_done) return

    bk = sf_backend(physics_pc_sf_pair_w, "physics_pc_sf_pair_w", my_id)

    if (jorek_model /= 199 .and. jorek_model /= 600) &
      call fatal("physics_pc_method = vs_mono supports model199 and model600 only: A_p's field order and "// &
                 "its handling of the boundary rows (v_par's Mach-1 rows) follow their reduced MHD")
    ! A_p packs the variables by these indices, at most SFO_NFMAX of them
    if (var_psi /= 1 .or. var_u /= 2 .or. var_zj /= 3 .or. var_w /= 4 .or. var_rho /= 5 .or. var_T /= 6 .or. &
        .not. (n_var == 6 .or. (jorek_model == 600 .and. n_var == 7 .and. var_Vpar == 7))) &
      call fatal("physics_pc_method = vs_mono supports (psi, u, zj, w, rho, T), plus v_par on model600 only: "// &
                 "A_p packs no further variables: build without with_TiTe, with_neutrals and with_impurities")
    if (mod(n_tor, 2) /= 1) &
      call fatal("physics_pc_method = vs_mono needs n_tor odd: the operator's toroidal slots are n = 0 "// &
                 "plus cos/sin pairs")
    nfm = n_var

    harm_all    = physics_pc_sf_harm_couple
    sf_gmg_harm = merge(SF_GMG_HARM_ALL, SF_GMG_HARM_PAIR, harm_all)
    famode      = physics_pc_sf_mode_split .and. .not. harm_all

    ! the hierarchy's configuration (module header); the GMG checks the values
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mono_smoother", s, set, ierr)
    if (set) then
      select case (trim(s))
      case ("rings"); m_smoother = SF_GMG_SMOOTHER_RINGS
      case ("lines"); m_smoother = SF_GMG_SMOOTHER_LINES
      case ("zebra"); m_smoother = SF_GMG_SMOOTHER_ZEBRA_RINGS
      case default;   call fatal("-vs_mono_smoother must be rings | lines | zebra, got '"//trim(s)//"'")
      end select
      m_smoother_name = trim(s)
    endif
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mono_pre0", iv, set, ierr)
    if (set) m_pre0 = int(iv)
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mono_post0", iv, set, ierr)
    if (set) m_post0 = int(iv)
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mono_nsc", iv, set, ierr)
    if (set) m_nsc = int(iv)
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mono_axis_rings", iv, set, ierr)
    if (set) m_axis = int(iv)
    ! a harmonic's cos and sin per smoother block on the exact (cross-|n|) A_p
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mono_harm_pair", iv, set, ierr)
    if (set .and. iv /= 0) sf_gmg_harm = SF_GMG_HARM_PAIR

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ========= monolithic multigrid (vs_mono) preconditioner ========="
      write(*,'(A,A,I0,A)') P//"A_p      : ", trim(physics_pc_sf_pair_w)//", the Jacobian's ", nfm, &
                            " variables in one operator, one solve per application, no W / predictor / corrector"
      if (bk == SF_GMG) then
        write(*,'(A,A,A,I0,A,I0,A,I0,A,I0)') P//"GMG      : one V-cycle (PREONLY), smoother ", &
          trim(m_smoother_name), ", pre0 ", m_pre0, ", post0 ", m_post0, ", nsc ", m_nsc, ", axis_rings ", m_axis
        write(*,'(A,ES9.2,A)') P//"inner rtol: ", physics_pc_sf_rtol, " (used only if -sf_gmg1_ksp_type "// &
                               "replaces the one V-cycle)"
      endif
      write(*,'(A)') P//"not used : physics_pc_sf_pair_psi, _rho, _T, _vpar, _vpar_stage, _corrector"
      if (harm_all .and. sf_gmg_harm == SF_GMG_HARM_ALL) then
        write(*,'(A)') P//"cross-|n|: all kept, all slots per GMG smoother block"
      else if (harm_all) then
        write(*,'(A)') P//"cross-|n|: all kept in A_p, cos / sin per GMG smoother block (-vs_mono_harm_pair)"
      else
        write(*,'(A)') P//"cross-|n|: none, |n|-diagonal A_p, cos / sin per GMG smoother block"
      endif
      if (famode) then
        write(*,'(A)') P//"layout   : one |n| family per rank group (physics_pc_sf_mode_split)"
      else
        write(*,'(A)') P//"layout   : global operator"
      endif
    endif

    vsm_init_done = .true.

  contains

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') P//"FATAL: ", trim(msg)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine vsm_init

  !> Build the preconditioner from the assembled Jacobian A_full, at every PC
  !! rebuild. A_p's pattern, the layout and the vectors are fixed by the first
  !! build; later builds refill A_p, so MUMPS and the GMG keep their setups.
  subroutine vsm_build(A_full, comm, my_id)
    use phys_module, only: physics_pc_sf_rtol, eliminate_boundary_dofs
    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr
    logical :: first
    integer :: comm_s
    Mat :: ws(1)

    call vsm_init(my_id)
    first  = vsm_first
    comm_g = comm

    if (first) then
      call sfo_layout(A_full, comm, famode, harm_all, my_id)
      call sfo_define(o_a, "A_p", 1, M_VAR(1:nfm))
    endif
    comm_s = sfo_comm

    ! A_p has no W source and no block scaled by opz: ws stands in, unread,
    ! for sfo_build / sfo_fill's W sources
    ws(1) = A_full
    call PetscLogEventBegin(pcev_extract, ierr)
    if (first) then
      call sfo_build(o_a, A_full, ws, sf_opz(), my_id)
    else
      call sfo_fill(o_a, A_full, ws, sf_opz())
    endif
    call PetscLogEventEnd(pcev_extract, ierr)
    call physics_pc_mem("vs_mono build: operator filled", my_id)

    ! Penalty rows (eliminated rows are diagonal-only rows of A_p and need
    ! nothing here): the full Jacobian's for step 2, and A_p's as Dirichlet
    ! rows, after the refill and before the scaling.
    if (.not. eliminate_boundary_dofs) then
      call full_penalty_rows(pen, A_full, comm, my_id, first, compact=.true.)
      call sf_solver_penalty_rows(slv, o_a, comm_s, my_id)
    endif
    call sf_solver_block_scale(slv, o_a, comm_s, my_id)

    ! GMG hierarchy instance 1, as the SF sweep's pair_w and vs_alfven's K_W
    ! (one method is active), one V-cycle per solve; rtol and maxits bind only
    ! where -sf_gmg1_ksp_type replaces PREONLY
    call PetscLogEventBegin(pcev_fact_w, ierr)
    call sf_solver_setup(slv, o_a%mat, bk, trim(o_a%label), comm_s, my_id, physics_pc_sf_rtol, gmg_inst=1, &
                         nfields=nfm, smoother=m_smoother, maxits=SF_GMG_MAXITS, &
                         pre0=m_pre0, post0=m_post0, nsmooth_c=m_nsc, axis_rings=m_axis, one_cycle=.true.)
    call PetscLogEventEnd(pcev_fact_w, ierr)
    call physics_pc_mem("vs_mono build: solver set up", my_id)

    if (.not. vecs_ready) call create_vectors()
    g_ctx%reduced_ready = .true.
    vsm_first = .false.
    ! the families finish their setups at different times: wait here, so that
    ! the setup timer holds the slowest one rather than the first apply
    if (famode) call MPI_Barrier(comm, ierr)

  contains

    !> rhs_p, sol_p and the variables' pieces on one storage each. A first
    !! gather (of a zero vector) creates the mode split's scatter here rather
    !! than in the first apply.
    subroutine create_vectors()
      PetscInt, parameter :: ONE = 1
      PetscInt :: n1, o
      integer :: f
      Vec :: x0

      n1 = sfo_n1
      ! at least one entry, so that *_store(o) below exists on a rank without rows
      allocate(rhs_store(max(nfm * n1, ONE)), sol_store(max(nfm * n1, ONE)))
      call VecCreateMPIWithArray(sfo_comm, ONE, nfm * n1, PETSC_DETERMINE, rhs_store, rhs_p, ierr)
      call VecCreateMPIWithArray(sfo_comm, ONE, nfm * n1, PETSC_DETERMINE, sol_store, sol_p, ierr)
      do f = 1, nfm
        o = (f - 1) * n1 + 1
        call VecCreateMPIWithArray(sfo_comm, ONE, n1, PETSC_DETERMINE, rhs_store(o), sv_x(M_VAR(f)), ierr)
        call VecCreateMPIWithArray(sfo_comm, ONE, n1, PETSC_DETERMINE, sol_store(o), sv_y(M_VAR(f)), ierr)
      enddo
      call MatCreateVecs(A_full, x0, PETSC_NULL_VEC, ierr)
      call VecSet(x0, 0.0d0, ierr)
      call sfo_vars_in(x0, sv_x(1:nfm))
      call VecDestroy(x0, ierr)
      vecs_ready = .true.
    end subroutine create_vectors

  end subroutine vsm_build

  !> y = P^-1 x, steps 1 and 2 of the module header.
  subroutine vsm_apply(x, y, ierr)
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    real*8 :: t0, t1

    ierr = 0
    call PetscLogEventBegin(pcev_apply, ierr)
    t0 = MPI_Wtime()
    call sfo_vars_in(x, sv_x(1:nfm))
    t_scat = t_scat + (MPI_Wtime() - t0)
    ! the gather wrote rhs_p's storage through its pieces, which leaves
    ! rhs_p's object state (and any norm cached on it) behind
    call PetscObjectStateIncrease(rhs_p, ierr)

    ! 1. A_p y = x into y's pieces; rhs_p is scaled in place, its pieces of x
    ! are not read again
    call sf_solver_solve(slv, rhs_p, sol_p, pcev_solve_w)

    t1 = MPI_Wtime()
    call sfo_vars_out(sv_y(1:nfm), y)
    t_scat = t_scat + (MPI_Wtime() - t1)
    ! 2. penalty rows of the full Jacobian (pen%n = 0 with eliminated
    ! boundary rows: nothing to do)
    if (pen%n > 0) call penalty_backsub(pen, x, y)
    t_apply = t_apply + (MPI_Wtime() - t0)
    call PetscLogEventEnd(pcev_apply, ierr)
    ierr = 0
  end subroutine vsm_apply

  !> Inner-iteration summary (mode split: per family), then reset. Collective.
  subroutine vsm_report(my_id)
    integer, intent(in) :: my_id
    if (famode) then
      call sf_solver_family_report("monolithic GMG", [slv], t_apply, t_scat, comm_g)
    else
      call sf_solver_report(slv, my_id)
    endif
    call sf_solver_reset_counters(slv)
    t_apply = 0.d0;  t_scat = 0.d0
  end subroutine vsm_report

#endif
end module mod_petsc_pc_vs_mono
