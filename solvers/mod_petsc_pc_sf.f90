module mod_petsc_pc_sf
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, &
       pcev_extract, pcev_convert, pcev_build_suu, pcev_fact_pj, pcev_fact_w, &
       pcev_fact_rhot, pcev_solve_pj, pcev_solve_w, pcev_solve_rhot, pcev_apply
  use mod_petsc_pc_blocks, only: create_variable_index_sets, extract_sub_blocks_h, &
       pack_pair_aij, make_pair_block_scale, make_field_block_scale, report_operator_density, &
       split_vars, merge_vars, harm_band
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_gather, only: sfg_build, sfg_gather, sfg_cross_weights
  use mod_petsc_pc_sf_pairw, only: sfw_structure, sfw_numeric, sfw_shell, sfw_dh, sfw_lines
  use mod_petsc_pc_sf_mixed, only: sfm_build, sfm_refill, sfm_op, sfm_nf
  use mod_petsc_raw_csr, only: blockmv_attach
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
  !! pair_w's (1,1) block S_uu has two production forms, one namelist entry
  !! apart (physics_pc_sf_suu); everything else in the sweep is shared.
  !!
  !! "w": the COMPOSED operator (workstream E; Chacon JCP 526 (2025) 113789,
  !! Eq. 17-19), assembled,
  !!
  !!   S_uu = B_22 + (theta dt)^2/opz * W(psi_0, p_0; n)          (S_W_aij)
  !!
  !! W is assembled at element level (construct_force_operator_matrix)
  !! already carrying its prefactor, its toroidal channels and its zeroed
  !! Dirichlet rows. No mass inverse anywhere, cheapest per outer iteration.
  !! But W is the continuum operator: it lacks the discrete projection M_j^-1
  !! of the Jacobian's Schur complement, an O(1) error at grid scale that
  !! grows as (dt v_A / h)^2, so its outer count grows with the mesh (shaped
  !! pcbench, np 4: 112 -> 203 its from 41x32 to 161x64). A SMALL-dt method:
  !! no convergence at tstep 10 on the ballooning case.
  !!
  !! "schur" (default): the psi-channel Schur complement of the Jacobian,
  !! matrix-free (mod_petsc_pc_sf_pairw),
  !!
  !!   S_uu u = B_22 u - B_21 psi - B_23 j,   [psi; j] = pair_psi^-1 [B_12 u; 0],
  !!
  !! pair_psi^-1 being one application of pair_psi's own solver. Its
  !! multigrid keeps B_22 + W for the Galerkin chain and runs level 0 on the
  !! same channel with a diagonal psi-row inverse and a Chebyshev constraint
  !! mass. Outer count flat in the mesh (21/20/19/21/22 over 41x32 .. 161x64,
  !! np 4, against 112 .. 201 for "w"); at 161x64 ~148 s against ~250 s.
  !!
  !! "wj" / "wpj": pair_w MIXED, assembled (mod_petsc_pc_sf_mixed). W with
  !! its bending term ("wj") or its whole psi channel ("wpj") taken back out,
  !! and the channel put back through explicit fields, the way pair_psi keeps
  !! j explicit: (u, omega, j) with psi eliminated by the node-lumped psi mass,
  !! or (u, omega, psi, j) with the small-flow psi row. Restores the discrete
  !! projections and the resistive damping of the psi response that W lacks,
  !! with every block sparse and every row second order. Its multigrid
  !! smooths with flux-surface ring blocks, a harmonic's cos and sin slots in
  !! one block (SF_GMG_SMOOTHER_RINGS, SF_GMG_HARM_PAIR_MIXED): at large dt
  !! the psi - u coupling runs along the field lines, toroidally as well.
  !! The rings leave the radial couplings, strongest on cells long in theta,
  !! to the coarse grid, so its first levels coarsen radially only
  !! (SF_GMG_SEMI_R_MIXED): pair_w then converges at the same rate from
  !! 41x32 to 121x48 and from tstep 0.1 to 10 (rho 0.043-0.045 per cycle).
  !!
  !! Both pairs smooth with zebra lines and run asymmetric V-cycles, all
  !! smoothing after the coarse correction (SF_PJ_*, SF_W_*): pair_psi
  !! converges in one cycle per solve, pair_w in ~1 (w) / ~4 (schur).
  !!
  !! WHERE IT IS VALID
  !! -----------------
  !! Both arms are measured up to tstep 1 (the shaped pcbench ramp). The
  !! schur arm's multigrid builds its coarse levels from the small-dt
  !! composed operator, not yet measured at tstep 10. The zebra smoother
  !! degrades on poloidally heavy meshes (n_flux well below n_tht); the
  !! production meshes are radially heavy.
  !!
  !! WHAT IS DELIBERATELY ABSENT
  !! ---------------------------
  !! Every verify_/probe_/report_/dump_ routine, every rejected arm, and every
  !! knob that was a measurement variable rather than a design choice, and every
  !! superseded method: each block has ONE production solver (GMG) plus the
  !! exact LU it is gated against, nothing else. The GMG smoothers, axis rings
  !! and boundary drop are fixed at their audited values in
  !! mod_petsc_pc_sf_solver. One namelist entry picks S_uu, four pick gmg | lu
  !! per block and one sets the shared inner tolerance; nothing else is
  !! configurable.
  !!
  !! Of the ~61 research physics_pc_* flags, this path READS exactly one --
  !! physics_pc_force_operator, which must be 1 because W is assembled at
  !! element level (the mixed arms then assemble only W's terms they do not
  !! carry through explicit fields: sf_force_terms) -- and FORCES one,
  !! physics_pc_harm_split = 1, which the
  !! block extraction reads. Its filter keeps the |n| groups within
  !! physics_pc_sf_harm_couple of each other (0, the default: the same |n|
  !! only; harm_band in mod_petsc_pc_blocks), fixed for the run because the
  !! band fixes every pattern. Every other one is ignored: the GMG receives its
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
  integer, save :: suu = SF_SUU_SCHUR          !< pair_w's S_uu (physics_pc_sf_suu)
  type(suu_form_t), save :: sform              !< ... with its variant (the mixed forms)
  logical, save :: mixed = .false.             !< pair_w mixed: "wj" / "wpj"
  integer, save :: corr = 0                    !< step-3 corrector (physics_pc_sf_corrector)

  !--- work state owned by this path --------------------------------------
  Vec, save :: sv_x(6), sv_y(6)
  Vec, save :: rhs_PJ, sol_PJ, rhs_W, sol_W
  Vec, save :: w3, w4, w5, t_rho, t_T
  Vec, save :: zv                      !< mixed: a zero field (the psi / j right-hand sides)
  logical, save :: vecs_ready = .false.
  logical, save :: kpj_packed = .false., sw_packed = .false., pw0_packed = .false.
  Mat, save     :: pw0                 !< schur: pair_w without W, [[B_22, B_24], [B_42, B_44]]

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
    use phys_module, only: physics_pc_sf_suu, physics_pc_sf_pair_psi, physics_pc_sf_pair_w, &
                           physics_pc_sf_rho, physics_pc_sf_T, physics_pc_sf_rtol, &
                           physics_pc_force_operator, physics_pc_harm_split, &
                           physics_pc_sf_harm_couple, physics_pc_sf_corrector
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr

    if (sf_init_done) return

    block
      logical :: ok
      call sf_suu_parse(physics_pc_sf_suu, sform, ok)
      if (.not. ok) call fatal("physics_pc_sf_suu must be schur | w | wj | wpj, got '"// &
                               trim(physics_pc_sf_suu)//"'")
    end block
    suu   = sform%form
    mixed = (suu == SF_SUU_WJ .or. suu == SF_SUU_WPJ)
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

    !--- the cross-|n| band of that filter: one value for the run
    if (physics_pc_sf_harm_couple < -1) &
      call fatal("physics_pc_sf_harm_couple must be -1 (all), 0 or a band k > 0")
    harm_band = physics_pc_sf_harm_couple

    !--- the corrector: Eq. (17) reads (psi, j) off pair_w, so it needs the
    !--- one form that carries both explicitly
    corr = physics_pc_sf_corrector
    if (corr < -1 .or. corr > 2) call fatal("physics_pc_sf_corrector must be -1 (auto), 0, 1 or 2")
    if (corr == -1) then
      corr = 0
      if (suu == SF_SUU_WPJ) corr = 1
    endif
    if (corr /= 0 .and. suu /= SF_SUU_WPJ) &
      call fatal("physics_pc_sf_corrector = 1 | 2 needs physics_pc_sf_suu = wpj (pair_w must carry psi and j)")

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ================ production SFM2 path ================"
      select case (suu)
      case (SF_SUU_SCHUR)
        write(*,'(A)') "[Physics PC]   S_uu     : schur (psi-channel Schur complement, matrix-free; GMG chain on B_22 + W)"
      case (SF_SUU_W)
        write(*,'(A)') "[Physics PC]   S_uu     : w (B_22 + W, assembled)"
      case (SF_SUU_WJ)
        write(*,'(A,A,A)') "[Physics PC]   S_uu     : ", trim(physics_pc_sf_suu), &
          " (pair_w mixed (u, omega, j): B_22 + W's kink and curvature, bending through j)"
      case (SF_SUU_WPJ)
        write(*,'(A,A,A)') "[Physics PC]   S_uu     : ", trim(physics_pc_sf_suu), &
          " (pair_w mixed (u, omega, psi, j): B_22 + W's curvature, psi channel through psi, j)"
      end select
      write(*,'(A,A)') "[Physics PC]   pair_psi : ", trim(physics_pc_sf_pair_psi)
      write(*,'(A,A)') "[Physics PC]   pair_w   : ", trim(physics_pc_sf_pair_w)
      write(*,'(A,A,A,A)') "[Physics PC]   rho / T  : ", trim(physics_pc_sf_rho), " / ", &
                           trim(physics_pc_sf_T)
      write(*,'(A,ES9.2)') "[Physics PC]   inner rtol: ", physics_pc_sf_rtol
      select case (harm_band)
      case (-1)
        write(*,'(A)') "[Physics PC]   cross-|n|: all kept (physics_pc_sf_harm_couple = -1)"
      case (0)
        write(*,'(A)') "[Physics PC]   cross-|n|: none, |n|-diagonal blocks (physics_pc_sf_harm_couple = 0)"
      case default
        write(*,'(A,I0,A)') "[Physics PC]   cross-|n|: |n| groups at most ", harm_band, &
          " apart kept (physics_pc_sf_harm_couple)"
      end select
      select case (corr)
      case (0)
        write(*,'(A)') "[Physics PC]   corrector: Eq. (16), second pair_psi solve"
      case (1)
        write(*,'(A)') "[Physics PC]   corrector: Eq. (17), (psi, j) from pair_w, B_16 T* on its psi row"
      case (2)
        write(*,'(A)') "[Physics PC]   corrector: Eq. (17), (psi, j) from pair_w, B_16 T* dropped"
      end select
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
    PetscInt :: n1_loc
    logical  :: first
    integer  :: pj_post0, pj_ovl, pj_rich, pj_sm, ax_pair, ax_rhot

    call sf_init(my_id)
    first = sf_first
    pj_post0 = SF_PJ_POST0_SCHUR
    if (suu /= SF_SUU_SCHUR) pj_post0 = SF_PJ_POST0_W
    ! schur: pair_psi also runs nested inside pair_w's S_uu shell, one fixed
    ! cycle per matvec, so it keeps the stronger V-cycle (see SF_GMG_RICH_FROM)
    pj_ovl = SF_GMG_LINE_OVERLAP;  pj_rich = SF_GMG_RICH_FROM;  pj_sm = SF_GMG_SMOOTHER_ZEBRA_RINGS
    if (suu == SF_SUU_SCHUR) then
      pj_ovl = SF_GMG_LINE_OVERLAP_SCHUR_W;  pj_rich = SF_GMG_RICH_NONE;  pj_sm = SF_GMG_SMOOTHER_ZEBRA
    endif
    ! exact axis extent per block (see SF_GMG_AXIS_RINGS_MIXED_PAIRS)
    ax_pair = SF_GMG_AXIS_RINGS;  ax_rhot = SF_GMG_AXIS_RINGS
    if (mixed) then
      ax_pair = SF_GMG_AXIS_RINGS_MIXED_PAIRS;  ax_rhot = SF_GMG_AXIS_RINGS_MIXED_RHOT
    endif

    !--- index sets ------------------------------------------------------
    if (.not. g_ctx%is_created) call create_variable_index_sets(A_full, comm)

    !--- how much cross-|n| coupling this Jacobian carries (one line)
    call sfg_cross_weights(A_full, comm, my_id)

    !--- the operators. Their patterns are fixed for the run, so the first
    !--- build constructs them and precomputes a VALUE MAP from JOREK's BAIJ
    !--- matrix (and W) into them; every later rebuild is one gather.
    if (first) then
      call first_build_operators()
    else
      call PetscLogEventBegin(pcev_extract, ierr)
      call sfg_gather(A_full, g_ctx%W_force, my_id)
      call PetscLogEventEnd(pcev_extract, ierr)
      if (mixed) then
        call PetscLogEventBegin(pcev_build_suu, ierr)
        call sfm_refill(comm, my_id)
        call PetscLogEventEnd(pcev_build_suu, ierr)
      endif
    endif
    call physics_pc_mem("SF build: operators filled", my_id)

    !--- symmetric block scaling, an exact similarity applied to the STORED
    !--- operator, on both pairs.
    call MatGetLocalSize(g_ctx%B_55, n1_loc, PETSC_NULL_INTEGER, ierr)
    if (slv_pj%scaled) call VecDestroy(slv_pj%dscale, ierr)
    call make_pair_block_scale(g_ctx%K_pj_aij, n1_loc, slv_pj%dscale, comm, my_id, "pair_psi")
    slv_pj%scaled = .true.

    !--- pair_psi is solved SPLIT (see the module header): the GMG runs on the
    !--- packed (psi, j) pair, both fields in every smoother block. Set up
    !--- before pair_w, whose schur shell applies it.
    call PetscLogEventBegin(pcev_fact_pj, ierr)
    call sf_solver_setup(slv_pj, g_ctx%K_pj_aij, bk_pj, "pair_psi KSP ([B_11,B_13;B_31,B_33])", &
                         comm, my_id, physics_pc_sf_rtol, gmg_inst=2, nfields=2, &
                         smoother=pj_sm, maxits=SF_GMG_MAXITS, &
                         pre0=SF_PJ_PRE0, post0=pj_post0, nsmooth_c=SF_PJ_NSC, &
                         line_overlap=pj_ovl, rich_from=pj_rich, axis_rings=ax_pair)
    call PetscLogEventEnd(pcev_fact_pj, ierr)

    !--- pair_w: the scaling balances the operator pair_w SOLVES -- schur on
    !--- the GMG backend: the Dh channel's sfw_lines, whose diagonal is the
    !--- shell's; otherwise the assembled S_W_aij
    if (slv_w%scaled) call VecDestroy(slv_w%dscale, ierr)
    if (suu == SF_SUU_SCHUR) then
      call PetscLogEventBegin(pcev_build_suu, ierr)
      call sfw_numeric(comm, my_id, slv_pj%dscale, bk_pj == SF_GMG, slv_pj%ksp)
      call PetscLogEventEnd(pcev_build_suu, ierr)
      if (bk_w == SF_GMG) then
        call make_pair_block_scale(sfw_lines, n1_loc, slv_w%dscale, comm, my_id, "pair_w")
        call MatDiagonalScale(g_ctx%S_W_aij, slv_w%dscale, slv_w%dscale, ierr)
      else
        call make_pair_block_scale(g_ctx%S_W_aij, n1_loc, slv_w%dscale, comm, my_id, "pair_w")
      endif
      call MatDiagonalScale(pw0, slv_w%dscale, slv_w%dscale, ierr)
    else if (mixed) then
      call make_field_block_scale(sfm_op, spread(n1_loc, 1, sfm_nf), slv_w%dscale, comm, my_id, "pair_w")
    else
      call make_pair_block_scale(g_ctx%S_W_aij, n1_loc, slv_w%dscale, comm, my_id, "pair_w")
    endif
    slv_w%scaled = .true.

    call PetscLogEventBegin(pcev_fact_w, ierr)
    if (suu == SF_SUU_SCHUR) then
      ! Krylov on the shell; the V-cycle on the Dh channel, its Galerkin
      ! chain on B_22 + W, its level-0 blocks from sfw_lines
      if (bk_w == SF_GMG) then
        call sf_solver_setup(slv_w, g_ctx%S_W_aij, bk_w, "pair_w KSP (schur S_uu)", &
                             comm, my_id, physics_pc_sf_rtol, gmg_inst=1, nfields=2, &
                             smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS, &
                             Aop=sfw_shell, Ablk=sfw_lines, Amg=sfw_dh, &
                             pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC, &
                             line_overlap=SF_GMG_LINE_OVERLAP_SCHUR_W, rich_from=SF_GMG_RICH_NONE, &
                             axis_sectors=SF_GMG_AXIS_SECTORS_SCHUR_W)
      else
        call sf_solver_setup(slv_w, g_ctx%S_W_aij, bk_w, "pair_w KSP (schur S_uu)", &
                             comm, my_id, physics_pc_sf_rtol, gmg_inst=1, nfields=2, &
                             smoother=SF_GMG_SMOOTHER_ZEBRA, maxits=SF_GMG_MAXITS, Aop=sfw_shell)
      endif
    else if (mixed) then
      call sf_solver_setup(slv_w, sfm_op, bk_w, "pair_w KSP (mixed)", &
                           comm, my_id, physics_pc_sf_rtol, gmg_inst=1, nfields=sfm_nf, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                           pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC_MIXED, &
                           harm_pair=SF_GMG_HARM_PAIR_MIXED, ring_overlap=SF_GMG_RING_OVERLAP, &
                           semi_r=SF_GMG_SEMI_R_MIXED, axis_rings=ax_pair)
    else
      call sf_solver_setup(slv_w, g_ctx%S_W_aij, bk_w, "pair_w KSP ([B_22+W,B_24;B_42,B_44])", &
                           comm, my_id, physics_pc_sf_rtol, gmg_inst=1, nfields=2, &
                           smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS, &
                           pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC)
    endif
    call PetscLogEventEnd(pcev_fact_w, ierr)

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    call sf_solver_setup(slv_rho, g_ctx%B_55, bk_rho, "rho-block KSP", &
                         comm, my_id, physics_pc_sf_rtol, gmg_inst=3, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT, axis_rings=ax_rhot)
    call sf_solver_setup(slv_T,   g_ctx%B_66, bk_T,   "T-block KSP", &
                         comm, my_id, physics_pc_sf_rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT, axis_rings=ax_rhot)
    call PetscLogEventEnd(pcev_fact_rhot, ierr)
    call physics_pc_mem("SF build: solvers set up", my_id)

    !--- work vectors: the operators keep their layout for the run.
    if (.not. vecs_ready) then
      call MatCreateVecs(g_ctx%K_pj_aij, rhs_PJ, sol_PJ, ierr)
      if (mixed) then
        call MatCreateVecs(sfm_op, rhs_W, sol_W, ierr)
      else
        call MatCreateVecs(g_ctx%S_W_aij, rhs_W, sol_W, ierr)
      endif
      call MatCreateVecs(g_ctx%B_55, sv_x(1), PETSC_NULL_VEC, ierr)
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
      call VecDuplicate(sv_x(1), zv, ierr)
      call VecSet(zv, 0.0d0, ierr)
      call MatCreateVecs(g_ctx%B_55, t_rho, PETSC_NULL_VEC, ierr)
      call MatCreateVecs(g_ctx%B_66, t_T,   PETSC_NULL_VEC, ierr)
      vecs_ready = .true.
    endif

    g_ctx%reduced_ready = .true.
    sf_first = .false.

  contains

    !> First build: the operators with their frozen patterns, the value maps
    !! into them, and the release of the blocks that only fed the packing.
    subroutine first_build_operators()
      integer, parameter :: NBLK = 21
      integer :: eqs(NBLK), vrs(NBLK), nkeep
      Mat :: M(NBLK), S_uu, none(0)

      if (.not. g_ctx%w_force_ready) then
        if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: W_force was never assembled "// &
          "(petsc_assemble_pc_matrices did not run?)."
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      endif

      !--- the 21 blocks in ONE pass over A_full's rows. The first 13 are the
      !--- ones the apply and the rho/T solvers read, the next 3 (B_13, B_31,
      !--- B_33) the ones the schur shell reads; the rest only feed the packed
      !--- pairs and are released once the maps exist.
      nkeep = 13
      if (suu == SF_SUU_SCHUR) nkeep = 16
      if (mixed) nkeep = NBLK                 ! the mixed pair_w repacks from all of them
      call PetscLogEventBegin(pcev_extract, ierr)
      eqs = [var_psi, var_psi, var_u, var_u, var_u, var_u, var_rho, var_rho, var_rho, &
             var_T, var_T, var_T, var_T, var_psi, var_zj, var_zj, &
             var_psi, var_u, var_u, var_w, var_w]
      vrs = [var_u, var_T, var_psi, var_zj, var_rho, var_T, var_psi, var_u, var_rho, &
             var_psi, var_u, var_zj, var_T, var_zj, var_psi, var_zj, &
             var_psi, var_u, var_w, var_u, var_w]
      call extract_sub_blocks_h(A_full, eqs, vrs, M, .true.)
      g_ctx%B_12 = M(1);  g_ctx%B_16 = M(2);  g_ctx%B_21 = M(3);  g_ctx%B_23 = M(4)
      g_ctx%B_25 = M(5);  g_ctx%B_26 = M(6);  g_ctx%B_51 = M(7);  g_ctx%B_52 = M(8)
      g_ctx%B_55 = M(9);  g_ctx%B_61 = M(10); g_ctx%B_62 = M(11); g_ctx%B_63 = M(12)
      g_ctx%B_66 = M(13)
      g_ctx%B_13 = M(14); g_ctx%B_31 = M(15); g_ctx%B_33 = M(16); g_ctx%B_11 = M(17)
      g_ctx%B_22 = M(18); g_ctx%B_24 = M(19); g_ctx%B_42 = M(20); g_ctx%B_44 = M(21)
      call PetscLogEventEnd(pcev_extract, ierr)

      !--- pair_psi = [[B_11, B_13], [B_31, B_33]]. Row 2 IS Jacobian row 3
      !--- verbatim, so the pair carries the j constraint EXACTLY: its second
      !--- component already is j* = M_j^-1 (x_j - B_31 psi*).
      call PetscLogEventBegin(pcev_convert, ierr)
      call pack_pair_aij(g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, g_ctx%B_33, &
                         g_ctx%K_pj_aij, kpj_packed, comm)
      call PetscLogEventEnd(pcev_convert, ierr)

      !--- S_uu = B_22 + W, then pair_w = [[S_uu, B_24], [B_42, B_44]]. W's
      !--- pattern is not a subset of B_22's; the union is fixed here, once.
      call PetscLogEventBegin(pcev_build_suu, ierr)
      call MatDuplicate(g_ctx%B_22, MAT_COPY_VALUES, S_uu, ierr)
      call MatAXPY(S_uu, 1.0d0, g_ctx%W_force, DIFFERENT_NONZERO_PATTERN, ierr)
      call pack_pair_aij(S_uu, g_ctx%B_24, g_ctx%B_42, g_ctx%B_44, &
                         g_ctx%S_W_aij, sw_packed, comm)
      call MatDestroy(S_uu, ierr)
      if (mixed) call sfm_build(sform, comm, my_id)
      if (suu == SF_SUU_SCHUR) then
        call pack_pair_aij(g_ctx%B_22, g_ctx%B_24, g_ctx%B_42, g_ctx%B_44, pw0, pw0_packed, comm)
        call sfw_structure(pw0, bk_w == SF_GMG, comm, my_id)
      endif
      call PetscLogEventEnd(pcev_build_suu, ierr)

      call sf_selfcheck(my_id)

      !--- the value maps, gated against what was just extracted (schur:
      !--- pw0 and sfw_lines, the pair_w smoother operator, are refilled like
      !--- S_W_aij; sfw_numeric adds the channel on top of sfw_lines)
      if (suu /= SF_SUU_SCHUR) then
        call sfg_build(A_full, g_ctx%W_force, M(1:nkeep), eqs(1:nkeep), vrs(1:nkeep), &
                       g_ctx%K_pj_aij, [var_psi, var_zj], [var_psi, var_zj], &
                       g_ctx%S_W_aij, none, [var_u, var_w], [var_u, var_w], comm, my_id)
      else if (bk_w == SF_GMG) then
        call sfg_build(A_full, g_ctx%W_force, M(1:nkeep), eqs(1:nkeep), vrs(1:nkeep), &
                       g_ctx%K_pj_aij, [var_psi, var_zj], [var_psi, var_zj], &
                       g_ctx%S_W_aij, [pw0, sfw_lines], [var_u, var_w], [var_u, var_w], comm, my_id)
      else
        call sfg_build(A_full, g_ctx%W_force, M(1:nkeep), eqs(1:nkeep), vrs(1:nkeep), &
                       g_ctx%K_pj_aij, [var_psi, var_zj], [var_psi, var_zj], &
                       g_ctx%S_W_aij, [pw0], [var_u, var_w], [var_u, var_w], comm, my_id)
      endif
      block
        integer :: k
        do k = nkeep + 1, NBLK
          call MatDestroy(M(k), ierr)
        enddo
      end block
      ! g_ctx held copies of the released handles: null them, so nothing can
      ! reach a freed Mat through them
      if (nkeep < 16) then
        g_ctx%B_13 = PETSC_NULL_MAT; g_ctx%B_31 = PETSC_NULL_MAT; g_ctx%B_33 = PETSC_NULL_MAT
      endif
      if (nkeep < NBLK) then
        g_ctx%B_11 = PETSC_NULL_MAT; g_ctx%B_22 = PETSC_NULL_MAT; g_ctx%B_24 = PETSC_NULL_MAT
        g_ctx%B_42 = PETSC_NULL_MAT; g_ctx%B_44 = PETSC_NULL_MAT
      endif

      !--- the SFM2 apply's coupling blocks: fixed patterns, refilled in place
      !--- by the value map, so one attach holds for the run
      if (SF_BLOCKMV == 1) then
        block
          use mod_parameters, only: n_tor
          integer :: nno, ncb, k
          Mat :: cb(13)
          ! a refused attach (not AIJ) keeps PETSc's own matvec: correct, but
          ! single-threaded, so it is reported
          cb(1:11) = [g_ctx%B_12, g_ctx%B_16, g_ctx%B_21, g_ctx%B_23, g_ctx%B_25, g_ctx%B_26, &
                      g_ctx%B_51, g_ctx%B_52, g_ctx%B_61, g_ctx%B_62, g_ctx%B_63]
          ncb = 11
          if (suu == SF_SUU_SCHUR) then
            cb(12:13) = [g_ctx%B_31, pw0]
            ncb = 13
          endif
          nno = 0
          do k = 1, ncb
            if (.not. blockmv_attach(cb(k), int(n_tor))) nno = nno + 1
          enddo
          if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   SF: WARNING ", nno, &
            " coupling block(s) not AIJ, left on PETSc's single-threaded matvec"
        end block
      endif
#if defined(PETSC_HAVE_MKL_SPARSE)
      !--- threaded SpMV: PETSc's AIJ matvec is single-threaded per rank, and
      !--- the level-0 matvecs are most of a V-cycle, so the operators the
      !--- multigrid multiplies with become MKL's inspector-executor type (in
      !--- place: same CSR arrays, so the value maps stay valid; the gather's
      !--- assembly refreshes MKL's handle). LU backends keep plain AIJ.
      !--- (SF_BLOCKMV = 1 does this with its own kernel, in the GMG setup)
      if (SF_BLOCKMV == 0) then
        if (bk_pj  == SF_GMG) call MatConvert(g_ctx%K_pj_aij, MATAIJMKL, MAT_INPLACE_MATRIX, g_ctx%K_pj_aij, ierr)
        if (bk_w   == SF_GMG) call MatConvert(g_ctx%S_W_aij,  MATAIJMKL, MAT_INPLACE_MATRIX, g_ctx%S_W_aij,  ierr)
        if (bk_rho == SF_GMG) call MatConvert(g_ctx%B_55,     MATAIJMKL, MAT_INPLACE_MATRIX, g_ctx%B_55,     ierr)
        if (bk_T   == SF_GMG) call MatConvert(g_ctx%B_66,     MATAIJMKL, MAT_INPLACE_MATRIX, g_ctx%B_66,     ierr)
        if (my_id == 0) write(*,'(A)') "[Physics PC]   SF: GMG operators converted to AIJMKL (threaded SpMV)"
      endif
#endif
    end subroutine first_build_operators

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
  !!      (Eq. (16)), or on "wpj" (default) psi += psi_w, j += j_w from
  !!      pair_w's own solution (Eq. (17), physics_pc_sf_corrector)
  !--------------------------------------------------------------------
  subroutine sf_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr

    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T
    Vec :: parts(4)
    logical, parameter :: keep_uw(4) = [.true., .true., .false., .false.]

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
    ! mixed: the psi / j rows' right-hand side is zero (the predictor has
    ! consumed x_psi, x_j). With the Eq. (16) corrector their solution is
    ! discarded -- step 3 recomputes psi and j by a pair_psi solve. With
    ! Eq. (17) (wpj) it IS the correction: rows 3-4 of pair_w read
    ! M~ (psi_w, j_w) = -(B_12 u + b, 0), M~ the small-flow pair_psi, so
    ! (psi_w, j_w) = -M~^-1 U (u, T*) with b = B_16 T* (corr 1) or 0 (corr 2).
    if (mixed) then
      if (corr == 1) then
        call MatMult(g_ctx%B_16, t_T, w4, ierr)
        call VecScale(w4, -1.0d0, ierr)
        parts = [w5, x_w, w4, zv]
      else
        parts = [w5, x_w, zv, zv]
      endif
      call sf_split_parts(rhs_W, parts(1:sfm_nf), .true.)
    else
      call sf_split_halves(rhs_W, w5, x_w, .true.)
    endif
    call PetscLogEventBegin(pcev_solve_w, ierr)
    call sf_solver_apply(slv_w, rhs_W, sol_W, ierr)
    call PetscLogEventEnd(pcev_solve_w, ierr)
    if (corr /= 0) then
      parts = [y_u, y_w, w3, w4]                          ! w3, w4 = psi_w, j_w
      call sf_split_parts(sol_W, parts, .false.)
    else if (mixed) then
      parts = [y_u, y_w, zv, zv]
      call sf_split_parts(sol_W, parts(1:sfm_nf), .false., keep=keep_uw(1:sfm_nf))
    else
      call sf_split_halves(sol_W, y_u, y_w, .false.)      ! BOTH final; y_w is DONE
    endif

    !--- Step 3: corrector psi-pair --------------------------------------
    if (corr /= 0) then
      ! Eq. (17): psi = psi* + psi_w, j = j* + j_w, no solve
      call VecAXPY(y_psi, 1.0d0, w3, ierr)
      call VecAXPY(y_j,   1.0d0, w4, ierr)
    else
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
    endif

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
