module mod_petsc_pc_sf
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, &
       pcev_extract, pcev_convert, pcev_build_suu, pcev_fact_pj, pcev_fact_w, &
       pcev_fact_rhot, pcev_solve_pj, pcev_solve_w, pcev_solve_rhot, pcev_apply
  use mod_petsc_pc_blocks, only: create_variable_index_sets, extract_sub_blocks_h, &
       pack_pair_aij, make_pair_block_scale, report_operator_density, &
       split_vars, merge_vars
  use mod_petsc_pc_sf_solver
  implicit none
  private

  !--------------------------------------------------------------------
  !> The production SFM2 preconditioner: a clean path, free of the research
  !! arms and of their per-rebuild cost.
  !!
  !! WHAT IT IS
  !! ----------
  !! One block-LDU sweep over the six variables, with j and omega kept
  !! EXPLICIT inside two mixed 2x2 pairs rather than substituted out:
  !!
  !!   pair_psi = [[B_11, B_13], [B_31, B_33]]      (psi, j)
  !!   pair_w   = [[S_uu, B_24], [B_42, B_44]]      (u, omega)
  !!
  !! pair_psi is solved SPLIT: the C1 GMG runs on the (psi, j) pair itself,
  !! psi and j kept separate through the whole cycle and smoothed together
  !! (Chacon JCP 526 (2025) S4.1). No Schur approximation and no B_33 solve
  !! enter, and eta_num > 0 is allowed: hyper-resistivity only adds
  !! eta_num K_1 to B_13, so both rows stay second order.
  !!
  !! and the momentum block taken in its COMPOSED form (workstream E; Chacon
  !! JCP 526 (2025) 113789, Eq. 17-19):
  !!
  !!   S_uu := B_22 + (theta dt)^2/opz * W(psi_0, p_0; n)
  !!
  !! rather than as the triple product B_22 - Ltil Shat^-1 B_12. W is assembled
  !! at element level (construct_force_operator_matrix) already carrying its
  !! prefactor, its toroidal channels and its zeroed Dirichlet rows, so forming
  !! S_uu here is one MatAXPY.
  !!
  !! WHAT THAT BUYS, AND WHY THIS PATH EXISTS
  !! ----------------------------------------
  !! The composed form has NO mass inverse. The triple product needs B_33^-1
  !! inside it, and PhysPC_MjSolve is the measured cluster blocker (90 s at
  !! np 1 -> 312 s at np 32 at 161x64). Dropping it also drops, for this path:
  !! the Ltil/Shat/channel chain, the sparse mass inverses and their FSAI
  !! machinery, the matrix-free pair_w MATSHELL, and -- because the SFM2 apply
  !! folds neither constraint -- the LU factorisations of B_33 and B_44 as
  !! well: neither mass matrix is ever factored on this path.
  !!
  !! WHERE IT IS VALID
  !! -----------------
  !! The composed operator is a SMALL-dt method. Measured on the shaped-limiter
  !! ballooning case: a 1.66x win at tstep 0.1, a wash at tstep 1, and NO
  !! convergence at tstep 10, where the exact-mass shell still converges. Do
  !! not read a failure at large dt as a bug in this module.
  !!
  !! WHAT IS DELIBERATELY ABSENT
  !! ---------------------------
  !! Every verify_/probe_/report_/dump_ routine, every rejected arm, and every
  !! knob that was a measurement variable rather than a design choice, and every
  !! superseded method: each block has ONE production solver (GMG) plus the
  !! exact LU it is gated against, nothing else. The GMG smoothers, axis rings
  !! and boundary drop are fixed at their audited values in
  !! mod_petsc_pc_sf_solver. Four namelist flags pick gmg | lu per block and
  !! one sets the shared inner tolerance; nothing else is configurable.
  !!
  !! Of the ~61 research physics_pc_* flags, this path READS exactly one --
  !! physics_pc_force_operator, which must be 1 because W is assembled at
  !! element level -- and FORCES one, physics_pc_harm_split = 1, which the
  !! block extraction reads. Every other one is ignored: the GMG receives its
  !! whole configuration explicitly (gmg_opts_t, from the constants in
  !! mod_petsc_pc_sf_solver), so a production deck cannot inherit a research
  !! setting.
  !!
  !! It is also excluded, by name rather than by a flag it happens to leave at
  !! a default, from three costs elsewhere in the solver: the commutator
  !! element blocks (physics_pc_needs_commutator_blocks), the Schur-correction
  !! element assembly (physics_pc_mixed_arm) and the full AIJ copy of the
  !! Jacobian (no_aij in mod_petsc). It reads none of the three.
  !!
  !! The self-check runs on the FIRST BUILD ONLY and has no flag.
  !--------------------------------------------------------------------

  logical, save :: sf_init_done = .false.
  logical, save :: sf_first     = .true.

  !--- the four block solvers ---------------------------------------------
  type(block_solver_t), save :: slv_pj     !< pair_psi
  type(block_solver_t), save :: slv_w      !< pair_w
  type(block_solver_t), save :: slv_rho
  type(block_solver_t), save :: slv_T

  !--- backends, resolved once from the namelist strings ------------------
  integer, save :: bk_pj = SF_GMG, bk_w = SF_GMG
  integer, save :: bk_rho = SF_LU, bk_T = SF_LU

  !--- work state owned by this path --------------------------------------
  Vec, save :: sv_x(6), sv_y(6)
  Vec, save :: rhs_PJ, sol_PJ, rhs_W, sol_W
  Vec, save :: w3, w4, w5, t_rho, t_T
  logical, save :: vecs_ready = .false.
  logical, save :: kpj_packed = .false., sw_packed = .false.

  public :: sf_enabled, sf_build, sf_apply, sf_report

contains

  !> Is the production path selected? Read by the two dispatch points in
  !! mod_petsc_pc_physics / mod_petsc_pc_physics_apply.
  logical function sf_enabled()
    use phys_module, only: physics_pc_sf
    sf_enabled = physics_pc_sf
  end function sf_enabled

  !--------------------------------------------------------------------
  !> Resolve the namelist into backends, and force the settings this path
  !! IMPLIES rather than offers. Forcing is loud, and follows the convention
  !! already used for suu_form = 1 forcing suu_shell = 0: a configuration that
  !! silently ran something other than what was asked for would mean measuring
  !! the wrong thing while believing otherwise.
  !--------------------------------------------------------------------
  subroutine sf_init(my_id)
    use phys_module, only: physics_pc_sf_pair_psi, physics_pc_sf_pair_w, &
                           physics_pc_sf_rho, physics_pc_sf_T, physics_pc_sf_rtol, &
                           physics_pc_force_operator, physics_pc_harm_split
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr

    if (sf_init_done) return

    bk_pj  = backend_of(physics_pc_sf_pair_psi, "physics_pc_sf_pair_psi")
    bk_w   = backend_of(physics_pc_sf_pair_w, "physics_pc_sf_pair_w")
    bk_rho = backend_of(physics_pc_sf_rho,    "physics_pc_sf_rho")
    bk_T   = backend_of(physics_pc_sf_T,      "physics_pc_sf_T")

    !--- the one hard precondition: W must have been assembled.
    if (physics_pc_force_operator /= 1) &
      call fatal("physics_pc_sf needs physics_pc_force_operator = 1 (the FULL Eq. (19) W); "// &
                 "without it S_uu = B_22 + W has no W to add.")

    !--- settings this path implies. Forced, not offered.
    call force_int(physics_pc_harm_split,      1, "physics_pc_harm_split")

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ================ production SFM2 path ================"
      write(*,'(A)') "[Physics PC]   S_uu = B_22 + W (composed force operator)"
      write(*,'(A,A)') "[Physics PC]   pair_psi : ", trim(physics_pc_sf_pair_psi)
      write(*,'(A,A)') "[Physics PC]   pair_w   : ", trim(physics_pc_sf_pair_w)
      write(*,'(A,A,A,A)') "[Physics PC]   rho / T  : ", trim(physics_pc_sf_rho), " / ", &
                           trim(physics_pc_sf_T)
      write(*,'(A,ES9.2)') "[Physics PC]   inner rtol: ", physics_pc_sf_rtol
    endif

    sf_init_done = .true.

  contains

    integer function backend_of(s, nm)
      character(len=*), intent(in) :: s, nm
      select case (trim(adjustl(s)))
      case ("lu");  backend_of = SF_LU
      case ("gmg"); backend_of = SF_GMG
      case default
        backend_of = SF_LU
        call fatal(trim(nm)//" must be lu | gmg, got '"//trim(s)//"'")
      end select
    end function backend_of

    subroutine force_int(v, want, nm)
      integer, intent(inout)       :: v
      integer, intent(in)          :: want
      character(len=*), intent(in) :: nm
      if (v == want) return
      if (my_id == 0) write(*,'(A,A,A,I0,A,I0,A)') &
        "[Physics PC]   production path: forcing ", trim(nm), " = ", want, &
        " (was ", v, "); this path implies it rather than offering it."
      v = want
    end subroutine force_int

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: ", trim(msg)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine sf_init

  !--------------------------------------------------------------------
  !> Build the preconditioner from the assembled Jacobian A_full. Called on
  !! every PC rebuild; everything whose pattern is frozen is refilled in
  !! place, so MUMPS and the GMG reuse their symbolic phases.
  !--------------------------------------------------------------------
  subroutine sf_build(A_full, comm, my_id)
    use phys_module,    only: physics_pc_sf_rtol
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T

    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id

    PetscErrorCode :: ierr
    Mat      :: S_uu
    PetscInt :: n1_loc
    logical  :: first

    call sf_init(my_id)
    first = sf_first

    !--- index sets ------------------------------------------------------
    if (.not. g_ctx%is_created) call create_variable_index_sets(A_full, comm)

    !--- the 21 blocks this path reads, in ONE pass over A_full's rows.
    !--- MatGetRow on JOREK's BAIJ matrix expands the whole n_var*n_tor-wide
    !--- block row, so reading each equation row once and dispatching its
    !--- entries costs a fraction of one extraction per block.
    call PetscLogEventBegin(pcev_extract, ierr)
    block
      integer, parameter :: NBLK = 21
      integer :: eqs(NBLK), vrs(NBLK)
      Mat :: M(NBLK)
      eqs = [var_zj, var_w, var_zj, var_w, &
             var_psi, var_u, var_u, var_T, &
             var_psi, var_u, var_rho, var_T, &
             var_psi, var_psi, var_u, var_u, var_u, var_rho, var_rho, var_T, var_T]
      vrs = [var_zj, var_w, var_psi, var_u, &
             var_zj, var_zj, var_w, var_zj, &
             var_psi, var_u, var_rho, var_T, &
             var_u, var_T, var_psi, var_rho, var_T, var_psi, var_u, var_psi, var_u]
      if (.not. first) then
        M = [g_ctx%B_33, g_ctx%B_44, g_ctx%B_31, g_ctx%B_42, &
             g_ctx%B_13, g_ctx%B_23, g_ctx%B_24, g_ctx%B_63, &
             g_ctx%B_11, g_ctx%B_22, g_ctx%B_55, g_ctx%B_66, &
             g_ctx%B_12, g_ctx%B_16, g_ctx%B_21, g_ctx%B_25, g_ctx%B_26, &
             g_ctx%B_51, g_ctx%B_52, g_ctx%B_61, g_ctx%B_62]
      endif
      call extract_sub_blocks_h(A_full, eqs, vrs, M, first)
      g_ctx%B_33 = M(1);  g_ctx%B_44 = M(2);  g_ctx%B_31 = M(3);  g_ctx%B_42 = M(4)
      g_ctx%B_13 = M(5);  g_ctx%B_23 = M(6);  g_ctx%B_24 = M(7);  g_ctx%B_63 = M(8)
      g_ctx%B_11 = M(9);  g_ctx%B_22 = M(10); g_ctx%B_55 = M(11); g_ctx%B_66 = M(12)
      g_ctx%B_12 = M(13); g_ctx%B_16 = M(14); g_ctx%B_21 = M(15); g_ctx%B_25 = M(16)
      g_ctx%B_26 = M(17); g_ctx%B_51 = M(18); g_ctx%B_52 = M(19); g_ctx%B_61 = M(20)
      g_ctx%B_62 = M(21)
    end block
    call PetscLogEventEnd(pcev_extract, ierr)
    call physics_pc_mem("SF build: blocks extracted", my_id)

    !--- pair_psi = [[B_11, B_13], [B_31, B_33]] -------------------------
    ! Row 2 IS Jacobian row 3 verbatim, so this pair carries the j constraint
    ! EXACTLY. That is what lets the apply drop the M_j pre-solve and the j
    ! back-substitution: the pair's second component already IS
    ! j* = M_j^-1 (x_j - B_31 psi*).
    call PetscLogEventBegin(pcev_convert, ierr)
    call pack_pair_aij(g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, g_ctx%B_33, &
                       g_ctx%K_pj_aij, kpj_packed, comm)
    call PetscLogEventEnd(pcev_convert, ierr)

    !--- S_uu = B_22 + W, then pair_w = [[S_uu, B_24], [B_42, B_44]] ------
    call PetscLogEventBegin(pcev_build_suu, ierr)
    if (.not. g_ctx%w_force_ready) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: W_force was never assembled "// &
        "(petsc_assemble_pc_matrices did not run?)."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    ! Duplicated rather than refilled in place: W's pattern is not a subset of
    ! B_22's, and one MatDuplicate per rebuild is negligible against the whole
    ! channel chain this form exists to avoid.
    call MatDuplicate(g_ctx%B_22, MAT_COPY_VALUES, S_uu, ierr)
    call MatAXPY(S_uu, 1.0d0, g_ctx%W_force, DIFFERENT_NONZERO_PATTERN, ierr)
    call pack_pair_aij(S_uu, g_ctx%B_24, g_ctx%B_42, g_ctx%B_44, &
                       g_ctx%S_W_aij, sw_packed, comm)
    call MatDestroy(S_uu, ierr)          ! safe: pack_pair_aij copied the values out
    call PetscLogEventEnd(pcev_build_suu, ierr)
    call physics_pc_mem("SF build: S_uu / pair_w assembled", my_id)

    !--- symmetric block scaling, an exact similarity applied to the STORED
    !--- operator, on both pairs.
    call MatGetLocalSize(g_ctx%B_11, n1_loc, PETSC_NULL_INTEGER, ierr)
    if (slv_pj%scaled) call VecDestroy(slv_pj%dscale, ierr)
    call make_pair_block_scale(g_ctx%K_pj_aij, n1_loc, slv_pj%dscale, comm, my_id, "pair_psi")
    slv_pj%scaled = .true.
    call MatGetLocalSize(g_ctx%B_22, n1_loc, PETSC_NULL_INTEGER, ierr)
    if (slv_w%scaled) call VecDestroy(slv_w%dscale, ierr)
    call make_pair_block_scale(g_ctx%S_W_aij, n1_loc, slv_w%dscale, comm, my_id, "pair_w")
    slv_w%scaled = .true.

    !--- the four block solvers -------------------------------------------
    ! pair_psi is solved SPLIT (see the module header): the GMG runs on the
    ! packed (psi, j) pair, both fields in every smoother block.
    call PetscLogEventBegin(pcev_fact_pj, ierr)
    call sf_solver_setup(slv_pj, g_ctx%K_pj_aij, bk_pj, "pair_psi KSP ([B_11,B_13;B_31,B_33])", &
                         comm, my_id, physics_pc_sf_rtol, gmg_inst=2, nfields=2, &
                         smoother=SF_GMG_SMOOTHER_ZEBRA, maxits=SF_GMG_MAXITS)
    call PetscLogEventEnd(pcev_fact_pj, ierr)

    call PetscLogEventBegin(pcev_fact_w, ierr)
    call sf_solver_setup(slv_w, g_ctx%S_W_aij, bk_w, "pair_w KSP ([B_22+W,B_24;B_42,B_44])", &
                         comm, my_id, physics_pc_sf_rtol, gmg_inst=1, nfields=2, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS)
    call PetscLogEventEnd(pcev_fact_w, ierr)

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    call sf_solver_setup(slv_rho, g_ctx%B_55, bk_rho, "rho-block KSP", &
                         comm, my_id, physics_pc_sf_rtol, gmg_inst=3, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    call sf_solver_setup(slv_T,   g_ctx%B_66, bk_T,   "T-block KSP", &
                         comm, my_id, physics_pc_sf_rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    call PetscLogEventEnd(pcev_fact_rhot, ierr)
    call physics_pc_mem("SF build: solvers set up", my_id)

    !--- work vectors: the operators keep their layout for the run.
    if (.not. vecs_ready) then
      call MatCreateVecs(g_ctx%K_pj_aij, rhs_PJ, sol_PJ, ierr)
      call MatCreateVecs(g_ctx%S_W_aij,  rhs_W,  sol_W,  ierr)
      call MatCreateVecs(g_ctx%B_11, sv_x(1), PETSC_NULL_VEC, ierr)
      block
        integer :: k
        do k = 2, 6
          call VecDuplicate(sv_x(1), sv_x(k), ierr)
        enddo
        do k = 1, 6
          call VecDuplicate(sv_x(1), sv_y(k), ierr)
        enddo
      end block
      call VecDuplicate(sv_x(1), w3, ierr)
      call VecDuplicate(sv_x(1), w4, ierr)
      call VecDuplicate(sv_x(1), w5, ierr)
      call MatCreateVecs(g_ctx%B_55, t_rho, PETSC_NULL_VEC, ierr)
      call MatCreateVecs(g_ctx%B_66, t_T,   PETSC_NULL_VEC, ierr)
      vecs_ready = .true.
    endif

    if (first) call sf_selfcheck(my_id)

    g_ctx%reduced_ready = .true.
    sf_first = .false.
  end subroutine sf_build

  !--------------------------------------------------------------------
  !> First-build structural check. No flag: once per run it costs a fraction
  !! of a second, and it is the only thing that catches a mis-scaled or
  !! mis-assembled operator -- both of which are invisible to every norm the
  !! build already prints. The GMG's own coarse-solve and boundary-row gates
  !! run on its first build for the same reason.
  !--------------------------------------------------------------------
  subroutine sf_selfcheck(my_id)
    integer, intent(in) :: my_id
    call report_operator_density(g_ctx%B_22,     "B_22 (bare momentum)", my_id)
    call report_operator_density(g_ctx%W_force,  "W    (force operator)", my_id)
    call report_operator_density(g_ctx%S_W_aij,  "pair_w (packed u,omega)", my_id)
    call report_operator_density(g_ctx%K_pj_aij, "pair_psi (packed psi,j)", my_id)
  end subroutine sf_selfcheck

  !--------------------------------------------------------------------
  !> y = P^-1 x: the block-LDU sweep.
  !!
  !!   1. predictor  pair_psi (psi*, j*) = (x_psi, x_j)
  !!      then       rho* , T*  against that explicit predictor
  !!   2. the ONE packed wave solve, pair_w (u, omega)
  !!   3. corrector  pair_psi (dpsi, dj) = (B_12 u + B_16 T*, 0);  psi -= dpsi
  !--------------------------------------------------------------------
  subroutine sf_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr

    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T

    ierr = 0
    call PetscLogEventBegin(pcev_apply, ierr)

    call split_vars(x, sv_x)
    x_psi = sv_x(var_psi); x_u   = sv_x(var_u);   x_j = sv_x(var_zj)
    x_w   = sv_x(var_w);   x_rho = sv_x(var_rho); x_T = sv_x(var_T)
    y_psi = sv_y(var_psi); y_u   = sv_y(var_u);   y_j = sv_y(var_zj)
    y_w   = sv_y(var_w);   y_rho = sv_y(var_rho); y_T = sv_y(var_T)

    !--- Step 1: predictor psi-pair -------------------------------------
    ! The j-component of the RHS is x_j, NOT zero: it is the constraint
    ! equation's own residual, and it is what makes j* equal the mass
    ! back-substitution M_j^-1 (x_j - B_31 psi*).
    call sf_split_halves(rhs_PJ, x_psi, x_j, .true.)
    call PetscLogEventBegin(pcev_solve_pj, ierr)
    call sf_solver_apply(slv_pj, rhs_PJ, sol_PJ, ierr)
    call PetscLogEventEnd(pcev_solve_pj, ierr)
    call sf_split_halves(sol_PJ, y_psi, y_j, .false.)     ! psi*, j*

    !--- Step 1: predictor density   rho* = B_55^-1 (x_rho - B_51 psi*) ---
    call MatMult(g_ctx%B_51, y_psi, w3, ierr)
    call VecWAXPY(w4, -1.0d0, w3, x_rho, ierr)
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call sf_solver_apply(slv_rho, w4, t_rho, ierr)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)

    !--- Step 1: predictor temperature  T* = B_66^-1 (x_T - B_61 psi* - B_63 j*)
    ! B_61 and B_63 act against the EXPLICIT predictor pair. No j-folded
    ! lower-triangular block is needed or wanted here.
    call MatMult(g_ctx%B_61, y_psi, w3, ierr)
    call VecWAXPY(w4, -1.0d0, w3, x_T, ierr)
    call MatMult(g_ctx%B_63, y_j, w3, ierr)
    call VecAXPY(w4, -1.0d0, w3, ierr)
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call sf_solver_apply(slv_T, w4, t_T, ierr)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)

    !--- Step 2: the ONE packed wave solve -------------------------------
    !   RHS_u  = x_u - B_21 psi* - B_23 j* - B_25 rho* - B_26 T*
    ! B_21 is the RAW lower coupling and B_23 j* carries the Lorentz path
    ! explicitly. There is deliberately NO -B_24 M_w^-1 x_w term: the
    ! u-omega coupling is the (1,2) entry of pair_w.
    !   RHS_om = x_w VERBATIM -- the omega row of the lower coupling is
    ! identically zero, so no fold and no correction term.
    call VecCopy(x_u, w5, ierr)
    call MatMult(g_ctx%B_21, y_psi, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call MatMult(g_ctx%B_23, y_j, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call MatMult(g_ctx%B_25, t_rho, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call MatMult(g_ctx%B_26, t_T, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call sf_split_halves(rhs_W, w5, x_w, .true.)
    call PetscLogEventBegin(pcev_solve_w, ierr)
    call sf_solver_apply(slv_w, rhs_W, sol_W, ierr)
    call PetscLogEventEnd(pcev_solve_w, ierr)
    call sf_split_halves(sol_W, y_u, y_w, .false.)        ! BOTH final; y_w is DONE

    !--- Step 3: corrector psi-pair --------------------------------------
    ! Here the j-component of the RHS IS zero, because the j-row of the upper
    ! coupling U is identically zero. (Contrast the predictor above.)
    call MatMult(g_ctx%B_12, y_u, w3, ierr)
    call MatMult(g_ctx%B_16, t_T, w4, ierr)
    call VecAXPY(w3, 1.0d0, w4, ierr)                     ! B_12 u + B_16 T*
    call VecZeroEntries(w4, ierr)
    call sf_split_halves(rhs_PJ, w3, w4, .true.)
    call PetscLogEventBegin(pcev_solve_pj, ierr)
    call sf_solver_apply(slv_pj, rhs_PJ, sol_PJ, ierr)
    call PetscLogEventEnd(pcev_solve_pj, ierr)
    call sf_split_halves(sol_PJ, w3, w4, .false.)         ! dpsi, dj

    call VecAXPY(y_psi, -1.0d0, w3, ierr)
    call VecAXPY(y_j,   -1.0d0, w4, ierr)

    !--- Step 3: rho / T correctors. Only u enters -- U's omega COLUMN is zero.
    call MatMult(g_ctx%B_52, y_u, w3, ierr)
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call sf_solver_apply(slv_rho, w3, w5, ierr)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)
    call VecWAXPY(y_rho, -1.0d0, w5, t_rho, ierr)

    call MatMult(g_ctx%B_62, y_u, w3, ierr)
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call sf_solver_apply(slv_T, w3, w5, ierr)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)
    call VecWAXPY(y_T, -1.0d0, w5, t_T, ierr)

    call merge_vars(sv_y, y)
    call PetscLogEventEnd(pcev_apply, ierr)
    ierr = 0
  end subroutine sf_apply

  !> Inner-iteration summary, one line per block, then reset.
  subroutine sf_report(my_id)
    integer, intent(in) :: my_id
    call sf_solver_report(slv_pj,  my_id)
    call sf_solver_report(slv_w,   my_id)
    call sf_solver_report(slv_rho, my_id)
    call sf_solver_report(slv_T,   my_id)
    call sf_solver_reset_counters(slv_pj)
    call sf_solver_reset_counters(slv_w)
    call sf_solver_reset_counters(slv_rho)
    call sf_solver_reset_counters(slv_T)
  end subroutine sf_report

#endif
end module mod_petsc_pc_sf
