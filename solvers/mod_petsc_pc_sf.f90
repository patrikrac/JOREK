module mod_petsc_pc_sf
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, &
       pcev_extract, pcev_convert, pcev_build_suu, pcev_fact_pj, pcev_fact_w, &
       pcev_fact_rhot, pcev_solve_pj, pcev_solve_w, pcev_solve_rhot, pcev_apply
  use mod_petsc_pc_blocks, only: create_variable_index_sets, extract_sub_blocks_h, &
       pack_pair_aij, make_pair_block_scale, make_field_block_scale, &
       split_vars, merge_vars, harm_band
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_gather, only: sfg_build, sfg_gather, sfg_cross_weights
  use mod_petsc_pc_sf_pairw, only: sfw_structure, sfw_numeric, sfw_shell, sfw_dh, sfw_lines
  use mod_petsc_pc_sf_mixed, only: sfm_build, sfm_refill, sfm_op, sfm_nf
  use mod_petsc_raw_csr, only: blockmv_attach
  use mod_petsc_pc_mass_cheb, only: mass_cheb_t, mass_cheb_setup, mass_cheb_solve
  use mod_petsc_pc_sf_fam, only: fam_op_t, sff_comm, sff_decompose, sff_op_build, &
       sff_op_fill, sff_vec_setup, sff_scatter_in, sff_scatter_out, &
       sff_cross_addmult, sff_cross_scale, sff_copy_local, sff_add_local
  implicit none
  private

  !--------------------------------------------------------------------
  !> The physics-based preconditioner (use_physics_pc): the split-field (SF)
  !! block-LDU sweep over the six variables (psi, u, j, omega, rho, T).
  !!
  !! PRODUCTION CONFIGURATION (the namelist defaults)
  !! ------------------------------------------------
  !!   physics_pc_sf_suu        = "wpj"   pair_w mixed (u, omega, psi, j)
  !!   physics_pc_sf_corrector  = -1      auto: strict Eq. (17) for wpj (3;
  !!                                      1 on model600)
  !!   physics_pc_sf_mode_split = .t.     one |n| family per rank group
  !!   physics_pc_sf_pair_psi / _pair_w / _rho / _T = "gmg"
  !!   physics_pc_sf_harm_couple = 0      |n|-diagonal blocks
  !! Everything else -- the S_uu forms schur / w / wj, the global (not mode
  !! split) operators, the cross-|n| band, the LU backends, the Eq. (16)
  !! corrector -- is a reference arm, kept for comparison.
  !!
  !! THE SWEEP
  !! ---------
  !! j and omega stay EXPLICIT inside two mixed pairs rather than being
  !! substituted out:
  !!
  !!   pair_psi = [[B_11, B_13], [B_31, B_33]]      (psi, j)
  !!   pair_w   = [[S_uu, B_24], [B_42, B_44]]      (u, omega)
  !!
  !! pair_psi is solved SPLIT: the C1 GMG runs on the (psi, j) pair itself,
  !! psi and j smoothed together (Chacon JCP 526 (2025) S4.1). No Schur
  !! approximation and no B_33 solve enter.
  !!
  !! pair_w's S_uu (physics_pc_sf_suu) builds on the composed force operator
  !! W (Chacon JCP 526 (2025) Eqs. 17-19), assembled at element level
  !! (construct_force_operator_matrix) with its prefactor and zeroed
  !! Dirichlet rows:
  !!
  !!   "wpj" / "wj": pair_w MIXED, assembled (mod_petsc_pc_sf_mixed): W with
  !!     its whole psi channel ("wpj") or its bending term ("wj") taken back
  !!     out, and the channel put back through explicit fields, the way
  !!     pair_psi keeps j explicit: (u, omega, psi, j) with the small-flow psi
  !!     row, or (u, omega, j) with psi eliminated by the node-lumped psi mass.
  !!     Restores the discrete projections and the resistive damping of the
  !!     psi response that W lacks, with every block sparse and every row
  !!     second order. Its multigrid smooths with flux-surface ring blocks, a
  !!     harmonic's cos and sin slots in one block (SF_GMG_SMOOTHER_RINGS,
  !!     SF_GMG_HARM_PAIR_MIXED), and coarsens radially only on its first
  !!     levels (SF_GMG_SEMI_R_MIXED). With wpj, step 3 reads (psi, j) off
  !!     pair_w's own solution (Eq. (17)) instead of a second pair_psi solve.
  !!   "w": S_uu = B_22 + W, assembled. W is the continuum operator: it lacks
  !!     the discrete projection M_j^-1, so its outer count grows with the mesh.
  !!   "schur": the psi-channel Schur complement of the Jacobian, matrix-free
  !!     (mod_petsc_pc_sf_pairw), S_uu u = B_22 u - B_21 psi - B_23 j with
  !!     [psi; j] = pair_psi^-1 [B_12 u; 0]; its multigrid keeps B_22 + W.
  !!
  !! THE MODE SPLIT
  !! --------------
  !! With physics_pc_sf_harm_couple = 0 every block is |n|-diagonal, so each
  !! |n| family (n = 0, then each cos/sin pair) is solved on its own rank
  !! group, all families concurrently (mod_petsc_pc_sf_fam). The operators
  !! are extracted straight from A and W into the family layout; no global
  !! operator exists. It needs at least one rank per family. A band k /= 0
  !! keeps cross-|n| couplings: each block solve then becomes FGMRES over all
  !! families on D + C, the family solvers as block-Jacobi preconditioner.
  !!
  !! CONFIGURATION
  !! -------------
  !! The GMG smoothers, V-cycle shapes, axis rings and boundary drop are fixed
  !! in mod_petsc_pc_sf_solver; the GMG receives its whole configuration
  !! explicitly (gmg_opts_t). The namelist picks S_uu, gmg | lu per block, the
  !! shared inner tolerance, the corrector, the band and the mode split.
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
  integer, save :: bk_rho = SF_GMG, bk_T = SF_GMG
  integer, save :: suu = SF_SUU_WPJ            !< pair_w's S_uu (physics_pc_sf_suu)
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

  !--- the operators the solvers and the sweep work on, and their communicator:
  !--- the global ones, or under physics_pc_sf_mode_split this rank's |n|
  !--- family's, extracted straight from A and W (mod_petsc_pc_sf_fam) at
  !--- every build; no global operator exists then
  logical, save :: famode = .false.
  integer, save :: comm_s = MPI_COMM_NULL
  integer, save :: comm_g = MPI_COMM_NULL      !< the full system's communicator
  real*8, save  :: t_apply = 0.d0, t_scat = 0.d0   !< mode split: apply / scatter wall time since the report
  integer, save :: nf_w = 0                    !< pair_w's packed fields
  Mat, save :: a_pj, a_w, a_rho, a_T
  Mat, save :: a_12, a_16, a_21, a_23, a_25, a_26, a_51, a_52, a_61, a_62, a_63, a_65
  !> The sweep's coupling blocks: B_12 B_16 B_21 B_23 B_25 B_26 B_51 B_52 B_61
  !! B_62 B_63, and B_65, the T row's rho column -- zero in model199 (a
  !! temperature equation), the O(1) mass (1+zeta) T0 drho in model600 (a
  !! pressure equation). It is applied only where it is not zero (use_b65), so
  !! model199 runs are unchanged.
  integer, parameter :: NCB = 12
  logical, save :: use_b65 = .false.
  Mat, save :: gB_65 = PETSC_NULL_MAT          !< global path's B_65 (g_ctx has no slot)
  type(fam_op_t), save :: o_pj, o_w, o_rho, o_T, o_cb(NCB)
  !> corr 3: the rho / T time-derivative mass. In model199 both rows carry
  !! (1 + zeta) v phi R, which is the omega mass B_44 times opz; B_44 is
  !! geometry-only, so its Chebyshev inverse is set up once and opz divided out
  type(fam_op_t), save :: o_m
  Mat, save :: a_m
  type(mass_cheb_t), save :: mcm

  !--- ... with a cross-|n| band (physics_pc_sf_harm_couple /= 0): each
  !--- operator's cross-family part C (global comm, mod_petsc_pc_sf_fam). The
  !--- sweep's coupling blocks apply D + C; each block solve becomes FGMRES on
  !--- the global communicator over D + C, preconditioned by the families'
  !--- own solvers (their V-cycles, all families at once): block Jacobi over
  !--- the |n| families inside a Krylov method that sees the coupling.
  logical, save :: banded = .false.
  type(fam_op_t), save :: x_pj, x_w, x_rho, x_T, x_cb(NCB)
  type :: band_ksp_t
    logical :: ready = .false.
    KSP :: ksp
    Mat :: op                          !< MATSHELL, D + C
    Vec :: bx, by                      !< global comm
    Vec :: fx, fy                      !< family comm, D's layout
  end type band_ksp_t
  type(band_ksp_t), save :: bks(4)     !< by GMG instance: 1 pair_w, 2 pair_psi, 3 rho, 4 T

  public :: sf_build, sf_apply, sf_report

contains

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
                           physics_pc_sf_harm_couple, physics_pc_sf_corrector, &
                           physics_pc_sf_mode_split
    use mod_parameters, only: n_tor
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

    !--- the variables the sweep knows: (psi, u, zj, w, rho, T), in the slots
    !--- of model199 -- which model600 has with its extensions off. The slot
    !--- layout (n = 0, then cos/sin pairs) needs n_tor odd -- an even n_tor
    !--- would leave the last slot in no family (fslots).
    if (jorek_model /= 199 .and. jorek_model /= 600) &
      call fatal("physics_pc_sf supports model199 and model600 only")
    if (n_var /= 6) &
      call fatal("physics_pc_sf supports (psi, u, zj, w, rho, T) only: build model600 with "// &
                 "with_vpar, with_TiTe, with_neutrals and with_impurities off")
    if (mod(n_tor, 2) /= 1) call fatal("physics_pc_sf needs n_tor odd (n = 0 plus cos/sin pairs)")

    !--- the cross-|n| band of that filter: one value for the run
    if (physics_pc_sf_harm_couple < -1) &
      call fatal("physics_pc_sf_harm_couple must be -1 (all), 0 or a band k > 0")
    harm_band = physics_pc_sf_harm_couple

    !--- the corrector: Eq. (17) reads (psi, j) off pair_w, so it needs the
    !--- one form that carries both explicitly
    corr = physics_pc_sf_corrector
    if (corr < -1 .or. corr > 3) call fatal("physics_pc_sf_corrector must be -1 (auto), 0, 1, 2 or 3")
    if (corr == -1) then
      corr = 0
      if (suu == SF_SUU_WPJ) corr = merge(3, 1, jorek_model == 199)
    endif
    if (corr /= 0 .and. suu /= SF_SUU_WPJ) &
      call fatal("physics_pc_sf_corrector = 1 | 2 | 3 needs physics_pc_sf_suu = wpj (pair_w must carry psi and j)")
    ! corr 3 takes the rho / T mass as opz B_44: model199's rows; model600's T
    ! row is a pressure equation (mass rho0 v phi R, plus a rho column)
    if (corr == 3 .and. jorek_model /= 199) &
      call fatal("physics_pc_sf_corrector = 3 supports model199 only (rho / T mass = opz B_44)")

    !--- one |n| family per rank: only where every block is |n|-diagonal, and
    !--- only for the assembled pair_w forms (the schur shell is not split)
    famode = physics_pc_sf_mode_split
    banded = famode .and. harm_band /= 0
    if (famode .and. (suu == SF_SUU_SCHUR .or. suu == SF_SUU_WJ)) &
      call fatal("physics_pc_sf_mode_split needs pair_w assembled entry by entry from A and W: "// &
                 "physics_pc_sf_suu = w | wpj")

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ================ split-field (SF) preconditioner ================"
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
      if (famode) write(*,'(A)') "[Physics PC]   layout   : one |n| family per rank, families concurrent "// &
                                 "(physics_pc_sf_mode_split)"
      if (banded) write(*,'(A)') "[Physics PC]   band     : block solves = FGMRES over all families on "// &
                                 "D + C (W too), the family solvers as block-Jacobi preconditioner"
      select case (corr)
      case (0)
        write(*,'(A)') "[Physics PC]   corrector: Eq. (16), second pair_psi solve"
      case (1)
        write(*,'(A)') "[Physics PC]   corrector: Eq. (17), (psi, j) from pair_w, B_16 T* on its psi row"
      case (2)
        write(*,'(A)') "[Physics PC]   corrector: Eq. (17), (psi, j) from pair_w, B_16 T* dropped"
      case (3)
        write(*,'(A)') "[Physics PC]   corrector: Eq. (17) strict, (psi, j) from pair_w, B_16 T* dropped, "// &
                       "rho / T by their mass (opz B_44)^-1, no block solve"
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
    use phys_module,    only: physics_pc_sf_rtol, physics_pc_sf_cross_weights
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T

    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id

    PetscErrorCode :: ierr
    PetscInt :: n1_loc
    logical  :: first
    integer  :: pj_post0, pj_ovl, pj_rich, pj_sm, ax_pair, ax_rhot

    call sf_init(my_id)
    first = sf_first
    comm_g = comm
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

    !--- how much cross-|n| coupling this Jacobian carries (diagnostic, off by default)
    if (physics_pc_sf_cross_weights) call sfg_cross_weights(A_full, comm, my_id)

    !--- the operators. Their patterns are fixed for the run, so the first
    !--- build constructs them and precomputes a VALUE MAP from JOREK's BAIJ
    !--- matrix (and W) into them; every later rebuild is one gather. Under
    !--- the mode split they are extracted straight into the family layout
    !--- instead, and no global operator exists.
    if (famode) then
      call family_operators(first)
      comm_s = sff_comm
    else
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
      a_pj = g_ctx%K_pj_aij;  a_rho = g_ctx%B_55;  a_T = g_ctx%B_66
      if (corr == 3) a_m = g_ctx%B_44
      a_w  = g_ctx%S_W_aij
      if (mixed) a_w = sfm_op
      nf_w = 2
      if (mixed) nf_w = sfm_nf
      a_12 = g_ctx%B_12; a_16 = g_ctx%B_16; a_21 = g_ctx%B_21; a_23 = g_ctx%B_23
      a_25 = g_ctx%B_25; a_26 = g_ctx%B_26; a_51 = g_ctx%B_51; a_52 = g_ctx%B_52
      a_61 = g_ctx%B_61; a_62 = g_ctx%B_62; a_63 = g_ctx%B_63; a_65 = gB_65
      comm_s = comm
    endif
    call physics_pc_mem("SF build: operators filled", my_id)

    !--- symmetric block scaling, an exact similarity applied to the STORED
    !--- operator, on both pairs.
    call MatGetLocalSize(a_rho, n1_loc, PETSC_NULL_INTEGER, ierr)
    if (slv_pj%scaled) call VecDestroy(slv_pj%dscale, ierr)
    call make_pair_block_scale(a_pj, n1_loc, slv_pj%dscale, comm_s, my_id, "pair_psi")
    slv_pj%scaled = .true.
    if (banded) call sff_cross_scale(x_pj, slv_pj%dscale)

    !--- pair_psi is solved SPLIT (see the module header): the GMG runs on the
    !--- packed (psi, j) pair, both fields in every smoother block. Set up
    !--- before pair_w, whose schur shell applies it.
    call PetscLogEventBegin(pcev_fact_pj, ierr)
    call sf_solver_setup(slv_pj, a_pj, bk_pj, "pair_psi KSP ([B_11,B_13;B_31,B_33])", &
                         comm_s, my_id, physics_pc_sf_rtol, gmg_inst=2, nfields=2, &
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
      call make_field_block_scale(a_w, spread(n1_loc, 1, nf_w), slv_w%dscale, comm_s, my_id, "pair_w")
    else
      call make_pair_block_scale(a_w, n1_loc, slv_w%dscale, comm_s, my_id, "pair_w")
    endif
    slv_w%scaled = .true.
    if (banded) call sff_cross_scale(x_w, slv_w%dscale)

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
      call sf_solver_setup(slv_w, a_w, bk_w, "pair_w KSP (mixed)", &
                           comm_s, my_id, physics_pc_sf_rtol, gmg_inst=1, nfields=nf_w, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                           pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC_MIXED, &
                           harm_pair=SF_GMG_HARM_PAIR_MIXED, ring_overlap=SF_GMG_RING_OVERLAP, &
                           semi_r=SF_GMG_SEMI_R_MIXED, axis_rings=ax_pair)
    else
      call sf_solver_setup(slv_w, a_w, bk_w, "pair_w KSP ([B_22+W,B_24;B_42,B_44])", &
                           comm_s, my_id, physics_pc_sf_rtol, gmg_inst=1, nfields=2, &
                           smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS, &
                           pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC)
    endif
    call PetscLogEventEnd(pcev_fact_w, ierr)

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    call sf_solver_setup(slv_rho, a_rho, bk_rho, "rho-block KSP", &
                         comm_s, my_id, physics_pc_sf_rtol, gmg_inst=3, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT, axis_rings=ax_rhot)
    call sf_solver_setup(slv_T,   a_T,   bk_T,   "T-block KSP", &
                         comm_s, my_id, physics_pc_sf_rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT, axis_rings=ax_rhot, &
                         harm_pair=SF_GMG_HARM_PAIR_T)
    if (corr == 3 .and. first) then
      block
        use mod_petsc_pc_harm, only: pc_ntor
        use mod_parameters,   only: n_tor
        integer :: nh
        nh = n_tor
        if (famode) nh = int(pc_ntor)
        call mass_cheb_setup(mcm, a_m, comm_s, "rho / T mass B_44", nh)
      end block
    endif
    call PetscLogEventEnd(pcev_fact_rhot, ierr)
    if (banded .and. first) then
      call band_setup(1, o_w,   SF_GMG_MAXITS)
      call band_setup(2, o_pj,  SF_GMG_MAXITS)
      call band_setup(3, o_rho, SF_GMG_MAXITS_RHOT)
      call band_setup(4, o_T,   SF_GMG_MAXITS_RHOT)
    endif
    call physics_pc_mem("SF build: solvers set up", my_id)

    !--- work vectors: the operators keep their layout for the run.
    if (.not. vecs_ready) then
      call MatCreateVecs(a_pj, rhs_PJ, sol_PJ, ierr)
      call MatCreateVecs(a_w, rhs_W, sol_W, ierr)
      call MatCreateVecs(a_rho, sv_x(1), PETSC_NULL_VEC, ierr)
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
      call MatCreateVecs(a_rho, t_rho, PETSC_NULL_VEC, ierr)
      call MatCreateVecs(a_T,   t_T,   PETSC_NULL_VEC, ierr)
      if (famode) then
        block
          Vec :: xt
          call MatCreateVecs(A_full, xt, PETSC_NULL_VEC, ierr)
          call sff_vec_setup(xt)
          call VecDestroy(xt, ierr)
        end block
      endif
      vecs_ready = .true.
    endif

    g_ctx%reduced_ready = .true.
    sf_first = .false.
    ! the families finish their setups at different times; wait for the
    ! slowest here, so the setup timer holds all of them rather than the
    ! first apply's scatter
    if (famode) call MPI_Barrier(comm, ierr)

  contains

    !> physics_pc_sf_mode_split: this rank's |n| family's operators, extracted
    !! straight from A and W. The first call splits the ranks, defines the
    !! operators and builds them (structure, plans, values); later calls
    !! refill their values (fixed patterns).
    subroutine family_operators(first_)
      logical, intent(in) :: first_
      integer :: k
      if (first_) then
        if (.not. g_ctx%w_force_ready) then
          if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: W_force was never assembled "// &
            "(petsc_assemble_pc_matrices did not run?)."
          call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
        call sff_decompose(A_full, comm, my_id)
        call define_family_operators()
        call PetscLogEventBegin(pcev_extract, ierr)
        call sff_op_build(o_pj,  A_full, g_ctx%W_force, sf_opz(), my_id)
        call sff_op_build(o_w,   A_full, g_ctx%W_force, sf_opz(), my_id)
        call sff_op_build(o_rho, A_full, g_ctx%W_force, sf_opz(), my_id)
        call sff_op_build(o_T,   A_full, g_ctx%W_force, sf_opz(), my_id)
        if (corr == 3) call sff_op_build(o_m, A_full, g_ctx%W_force, sf_opz(), my_id)
        do k = 1, NCB
          call sff_op_build(o_cb(k), A_full, g_ctx%W_force, sf_opz(), my_id)
        enddo
        if (banded) then
          call sff_op_build(x_pj,  A_full, g_ctx%W_force, sf_opz(), my_id)
          call sff_op_build(x_w,   A_full, g_ctx%W_force, sf_opz(), my_id)
          call sff_op_build(x_rho, A_full, g_ctx%W_force, sf_opz(), my_id)
          call sff_op_build(x_T,   A_full, g_ctx%W_force, sf_opz(), my_id)
          do k = 1, NCB
            call sff_op_build(x_cb(k), A_full, g_ctx%W_force, sf_opz(), my_id)
          enddo
        endif
        call PetscLogEventEnd(pcev_extract, ierr)
      else
        call PetscLogEventBegin(pcev_extract, ierr)
        call sff_op_fill(o_pj,  A_full, g_ctx%W_force, sf_opz())
        call sff_op_fill(o_w,   A_full, g_ctx%W_force, sf_opz())
        call sff_op_fill(o_rho, A_full, g_ctx%W_force, sf_opz())
        call sff_op_fill(o_T,   A_full, g_ctx%W_force, sf_opz())
        do k = 1, NCB
          call sff_op_fill(o_cb(k), A_full, g_ctx%W_force, sf_opz())
        enddo
        if (banded) then
          call sff_op_fill(x_pj,  A_full, g_ctx%W_force, sf_opz())
          call sff_op_fill(x_w,   A_full, g_ctx%W_force, sf_opz())
          call sff_op_fill(x_rho, A_full, g_ctx%W_force, sf_opz())
          call sff_op_fill(x_T,   A_full, g_ctx%W_force, sf_opz())
          do k = 1, NCB
            call sff_op_fill(x_cb(k), A_full, g_ctx%W_force, sf_opz())
          enddo
        endif
        call PetscLogEventEnd(pcev_extract, ierr)
      endif
      a_pj = o_pj%fam;  a_w = o_w%fam;  a_rho = o_rho%fam;  a_T = o_T%fam
      if (corr == 3) a_m = o_m%fam
      nf_w = o_w%nf
      a_12 = o_cb(1)%fam; a_16 = o_cb(2)%fam; a_21 = o_cb(3)%fam;  a_23 = o_cb(4)%fam
      a_25 = o_cb(5)%fam; a_26 = o_cb(6)%fam; a_51 = o_cb(7)%fam;  a_52 = o_cb(8)%fam
      a_61 = o_cb(9)%fam; a_62 = o_cb(10)%fam; a_63 = o_cb(11)%fam; a_65 = o_cb(12)%fam
      if (first_) call set_use_b65(a_65, comm)
      !--- the sweep's coupling blocks on the threaded block kernel, as on the
      !--- global path; the kernel reads the CSR arrays in place, so one
      !--- attach holds across the refills
      if (first_) then
        block
          use mod_petsc_pc_harm, only: pc_ntor
          integer :: nno
          nno = 0
          do k = 1, NCB
            if (.not. blockmv_attach(o_cb(k)%fam, int(pc_ntor))) nno = nno + 1
          enddo
          ! the cross parts: rows of one family, columns of several, so no
          ! common harmonic block -- the kernel row by row
          if (banded) then
            if (.not. blockmv_attach(x_pj%fam, 1))  nno = nno + 1
            if (.not. blockmv_attach(x_w%fam, 1))   nno = nno + 1
            if (.not. blockmv_attach(x_rho%fam, 1)) nno = nno + 1
            if (.not. blockmv_attach(x_T%fam, 1))   nno = nno + 1
            do k = 1, NCB
              if (.not. blockmv_attach(x_cb(k)%fam, 1)) nno = nno + 1
            enddo
          endif
          if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   SF: WARNING ", nno, &
            " family coupling block(s) not AIJ, left on PETSc's single-threaded matvec"
        end block
      endif
    end subroutine family_operators

    !> The family operators' blocks, as the global path packs them: pair_psi
    !! [[B_11, B_13], [B_31, B_33]]; pair_w [[B_22 + W, B_24], [B_42, B_44]]
    !! or, wpj, [[B_22 + W, B_24, B_21, B_23], [B_42, B_44, 0, 0],
    !! [B_12, 0, opz B_33, B_13], [0, 0, B_31, B_33]] (mod_petsc_pc_sf_mixed);
    !! rho, T and the sweep's NCB coupling blocks.
    subroutine define_family_operators()
      integer, parameter :: ce(NCB) = [var_psi, var_psi, var_u, var_u, var_u, var_u, &
                                       var_rho, var_rho, var_T, var_T, var_T, var_T]
      integer, parameter :: cv(NCB) = [var_u, var_T, var_psi, var_zj, var_rho, var_T, &
                                       var_psi, var_u, var_psi, var_u, var_zj, var_rho]
      character(len=4), parameter :: cn(NCB) = ["B_12", "B_16", "B_21", "B_23", "B_25", "B_26", &
                                                 "B_51", "B_52", "B_61", "B_62", "B_63", "B_65"]
      integer :: k
      call full_op(o_pj, "pair_psi", 1, [var_psi, var_zj])
      if (suu == SF_SUU_WPJ) then
        call full_op(o_w, "pair_w (wpj)", 2, [var_u, var_w, var_psi, var_zj])
        o_w%se(2, 3:4) = 0; o_w%se(3, 2) = 0; o_w%se(4, 1:2) = 0
        o_w%se(3, 3) = var_zj; o_w%sv(3, 3) = var_zj; o_w%scl(3, 3) = 1     ! opz M_psi = opz B_33
      else
        call full_op(o_w, "pair_w", 2, [var_u, var_w])
      endif
      o_w%has_w = .true.
      call full_op(o_rho, "B_55", 3, [var_rho])
      call full_op(o_T,   "B_66", 4, [var_T])
      ! tags 1 .. 2 (4 + NCB) are the family and cross parts'
      if (corr == 3) call full_op(o_m, "B_44", 2 * (4 + NCB) + 1, [var_w])
      do k = 1, NCB
        o_cb(k)%label = cn(k); o_cb(k)%nf = 1; o_cb(k)%tagb = 4 + k
        o_cb(k)%se(1, 1) = ce(k); o_cb(k)%sv(1, 1) = cv(k)
      enddo
      ! the cross-family parts: the same blocks, part 1, own message tags
      if (banded) then
        call cross_of(o_pj, x_pj);  call cross_of(o_w, x_w)
        call cross_of(o_rho, x_rho);  call cross_of(o_T, x_T)
        do k = 1, NCB
          call cross_of(o_cb(k), x_cb(k))
        enddo
      endif
    end subroutine define_family_operators

    subroutine cross_of(d, c)
      type(fam_op_t), intent(in)    :: d
      type(fam_op_t), intent(inout) :: c
      ! tags 1..4 + NCB are the family parts', so the cross parts start above
      c%label = d%label; c%nf = d%nf; c%tagb = d%tagb + 4 + NCB; c%part = 1
      c%se = d%se; c%sv = d%sv; c%scl = d%scl; c%has_w = d%has_w
    end subroutine cross_of

    subroutine full_op(op, label, tagb, vars)
      type(fam_op_t), intent(inout) :: op
      character(len=*), intent(in) :: label
      integer, intent(in) :: tagb, vars(:)
      integer :: a, b
      op%label = label; op%nf = size(vars); op%tagb = tagb
      do a = 1, size(vars)
        do b = 1, size(vars)
          op%se(a, b) = vars(a); op%sv(a, b) = vars(b)
        enddo
      enddo
    end subroutine full_op

    !> First build: the operators with their frozen patterns, the value maps
    !! into them, and the release of the blocks that only fed the packing.
    subroutine first_build_operators()
      integer, parameter :: NBLK = 22
      integer :: eqs(NBLK), vrs(NBLK), nkeep
      Mat :: M(NBLK), S_uu, none(0)

      if (.not. g_ctx%w_force_ready) then
        if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: W_force was never assembled "// &
          "(petsc_assemble_pc_matrices did not run?)."
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      endif

      !--- the 22 blocks in ONE pass over A_full's rows. The first 14 are the
      !--- ones the apply and the rho/T solvers read (the 14th is B_65), the
      !--- next 3 (B_13, B_31, B_33) the ones the schur shell reads; the rest
      !--- only feed the packed pairs and are released once the maps exist.
      nkeep = 14
      if (suu == SF_SUU_SCHUR) nkeep = 17
      if (mixed) nkeep = NBLK                 ! the mixed pair_w repacks from all of them
      call PetscLogEventBegin(pcev_extract, ierr)
      eqs(1:14) = [var_psi, var_psi, var_u, var_u, var_u, var_u, var_rho, var_rho, var_rho, &
                   var_T, var_T, var_T, var_T, var_T]
      vrs(1:14) = [var_u, var_T, var_psi, var_zj, var_rho, var_T, var_psi, var_u, var_rho, &
                   var_psi, var_u, var_zj, var_T, var_rho]
      eqs(15:NBLK) = [var_psi, var_zj, var_zj, var_psi, var_u, var_u, var_w, var_w]
      vrs(15:NBLK) = [var_zj, var_psi, var_zj, var_psi, var_u, var_w, var_u, var_w]
      call extract_sub_blocks_h(A_full, eqs, vrs, M, .true.)
      g_ctx%B_12 = M(1);  g_ctx%B_16 = M(2);  g_ctx%B_21 = M(3);  g_ctx%B_23 = M(4)
      g_ctx%B_25 = M(5);  g_ctx%B_26 = M(6);  g_ctx%B_51 = M(7);  g_ctx%B_52 = M(8)
      g_ctx%B_55 = M(9);  g_ctx%B_61 = M(10); g_ctx%B_62 = M(11); g_ctx%B_63 = M(12)
      g_ctx%B_66 = M(13); gB_65 = M(14)
      g_ctx%B_13 = M(15); g_ctx%B_31 = M(16); g_ctx%B_33 = M(17); g_ctx%B_11 = M(18)
      g_ctx%B_22 = M(19); g_ctx%B_24 = M(20); g_ctx%B_42 = M(21); g_ctx%B_44 = M(22)
      call set_use_b65(gB_65, comm)
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
      if (nkeep < 17) then
        g_ctx%B_13 = PETSC_NULL_MAT; g_ctx%B_31 = PETSC_NULL_MAT; g_ctx%B_33 = PETSC_NULL_MAT
      endif
      if (nkeep < NBLK) then
        g_ctx%B_11 = PETSC_NULL_MAT; g_ctx%B_22 = PETSC_NULL_MAT; g_ctx%B_24 = PETSC_NULL_MAT
        g_ctx%B_42 = PETSC_NULL_MAT; g_ctx%B_44 = PETSC_NULL_MAT
      endif

      !--- the sweep's coupling blocks: fixed patterns, refilled in place by
      !--- the value map, so one attach holds for the run
      block
        use mod_parameters, only: n_tor
        integer :: nno, ncb, k
        Mat :: cb(14)
        ! a refused attach (not AIJ) keeps PETSc's own matvec: correct, but
        ! single-threaded, so it is reported
        cb(1:12) = [g_ctx%B_12, g_ctx%B_16, g_ctx%B_21, g_ctx%B_23, g_ctx%B_25, g_ctx%B_26, &
                    g_ctx%B_51, g_ctx%B_52, g_ctx%B_61, g_ctx%B_62, g_ctx%B_63, gB_65]
        ncb = 12
        if (suu == SF_SUU_SCHUR) then
          cb(13:14) = [g_ctx%B_31, pw0]
          ncb = 14
        endif
        nno = 0
        do k = 1, ncb
          if (.not. blockmv_attach(cb(k), int(n_tor))) nno = nno + 1
        enddo
        if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   SF: WARNING ", nno, &
          " coupling block(s) not AIJ, left on PETSc's single-threaded matvec"
      end block
    end subroutine first_build_operators

  end subroutine sf_build

  !====================================================================
  ! the cross-|n| band under the mode split
  !====================================================================

  !> The banded solve of block k (GMG instance numbering): FGMRES on the
  !! global communicator over the shell D + C, preconditioned by one
  !! application of the family solver's own preconditioner (its V-cycle).
  !! Once: the shell reads the operators in place and the family solvers are
  !! re-pointed at every rebuild.
  subroutine band_setup(k, d, maxits)
    use phys_module, only: physics_pc_sf_rtol
    integer, intent(in)        :: k, maxits
    type(fam_op_t), intent(in) :: d
    PetscInt :: nloc
    PC :: pc
    PetscErrorCode :: ierr
    character(len=16) :: pfx
    associate (b => bks(k))
      call MatGetLocalSize(d%fam, nloc, PETSC_NULL_INTEGER, ierr)
      call MatCreateShell(comm_g, nloc, nloc, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_NULL_INTEGER, b%op, ierr)
      select case (k)
      case (1); call MatShellSetOperation(b%op, MATOP_MULT, band_mult_1, ierr)
      case (2); call MatShellSetOperation(b%op, MATOP_MULT, band_mult_2, ierr)
      case (3); call MatShellSetOperation(b%op, MATOP_MULT, band_mult_3, ierr)
      case (4); call MatShellSetOperation(b%op, MATOP_MULT, band_mult_4, ierr)
      end select
      call MatCreateVecs(d%fam, b%fx, b%fy, ierr)
      call MatCreateVecs(b%op, b%bx, b%by, ierr)
      call KSPCreate(comm_g, b%ksp, ierr)
      call KSPSetType(b%ksp, KSPFGMRES, ierr)
      call KSPGMRESSetRestart(b%ksp, max(maxits, 2), ierr)
      call KSPSetTolerances(b%ksp, physics_pc_sf_rtol, 1.d-50, 1.d6, maxits, ierr)
      write(pfx, '(A,I0,A)') "sf_band", k, "_"            ! -sf_band<k>_ksp_monitor etc.
      call KSPSetOptionsPrefix(b%ksp, trim(pfx), ierr)
      call KSPSetFromOptions(b%ksp, ierr)
      call KSPSetOperators(b%ksp, b%op, b%op, ierr)
      call KSPGetPC(b%ksp, pc, ierr)
      call PCSetType(pc, PCSHELL, ierr)
      select case (k)
      case (1); call PCShellSetApply(pc, band_pc_1, ierr)
      case (2); call PCShellSetApply(pc, band_pc_2, ierr)
      case (3); call PCShellSetApply(pc, band_pc_3, ierr)
      case (4); call PCShellSetApply(pc, band_pc_4, ierr)
      end select
      call PCShellSetName(pc, "family solvers, block Jacobi over |n|", ierr)
      call KSPSetUp(b%ksp, ierr)
      b%ready = .true.
    end associate
  end subroutine band_setup

  !> y = (D + C) x for block k, x and y on the global communicator.
  subroutine band_mult(k, x, y)
    integer, intent(in) :: k
    Vec :: x, y
    select case (k)
    case (1); call mult_dc(o_w,   x_w)
    case (2); call mult_dc(o_pj,  x_pj)
    case (3); call mult_dc(o_rho, x_rho)
    case (4); call mult_dc(o_T,   x_T)
    end select
  contains
    subroutine mult_dc(d, c)
      type(fam_op_t), intent(inout) :: d, c
      PetscErrorCode :: ierr
      call MatMult(c%fam, x, y, ierr)
      call sff_copy_local(x, bks(k)%fx)
      call MatMult(d%fam, bks(k)%fx, bks(k)%fy, ierr)
      call sff_add_local(y, bks(k)%fy)
    end subroutine mult_dc
  end subroutine band_mult

  !> y = P_k^-1 x: the family solver's preconditioner (all families at once).
  subroutine band_pc(k, x, y)
    integer, intent(in) :: k
    Vec :: x, y
    PC :: pc
    PetscErrorCode :: ierr
    select case (k)
    case (1); call KSPGetPC(slv_w%ksp, pc, ierr)
    case (2); call KSPGetPC(slv_pj%ksp, pc, ierr)
    case (3); call KSPGetPC(slv_rho%ksp, pc, ierr)
    case (4); call KSPGetPC(slv_T%ksp, pc, ierr)
    end select
    call sff_copy_local(x, bks(k)%fx)
    call PCApply(pc, bks(k)%fx, bks(k)%fy, ierr)
    call sff_copy_local(bks(k)%fy, y)
  end subroutine band_pc

  subroutine band_mult_1(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_mult(1, x, y);  ierr = 0
  end subroutine band_mult_1
  subroutine band_mult_2(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_mult(2, x, y);  ierr = 0
  end subroutine band_mult_2
  subroutine band_mult_3(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_mult(3, x, y);  ierr = 0
  end subroutine band_mult_3
  subroutine band_mult_4(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_mult(4, x, y);  ierr = 0
  end subroutine band_mult_4
  subroutine band_pc_1(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_pc(1, x, y);  ierr = 0
  end subroutine band_pc_1
  subroutine band_pc_2(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_pc(2, x, y);  ierr = 0
  end subroutine band_pc_2
  subroutine band_pc_3(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_pc(3, x, y);  ierr = 0
  end subroutine band_pc_3
  subroutine band_pc_4(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    call band_pc(4, x, y);  ierr = 0
  end subroutine band_pc_4

  !> A block solve of the sweep: the solver's own, or with a band the
  !! FGMRES over D + C (the same scaling similarity, the same counters).
  subroutine bsolve(slv, rhs, sol)
    type(block_solver_t), intent(inout) :: slv
    Vec :: rhs, sol
    PetscErrorCode :: ierr
    PetscInt :: its
    KSPConvergedReason :: reason
    real*8 :: t0
    ierr = 0
    if (.not. banded) then
      call sf_solver_apply(slv, rhs, sol, ierr)
      return
    endif
    associate (b => bks(slv%gmg_inst))
      t0 = MPI_Wtime()
      if (slv%scaled) call VecPointwiseMult(rhs, rhs, slv%dscale, ierr)
      call sff_copy_local(rhs, b%bx)
      call KSPSolve(b%ksp, b%bx, b%by, ierr)
      call sff_copy_local(b%by, sol)
      if (slv%scaled) call VecPointwiseMult(sol, sol, slv%dscale, ierr)
      slv%t_sum = slv%t_sum + (MPI_Wtime() - t0)
      call KSPGetIterationNumber(b%ksp, its, ierr)
      slv%its_sum = slv%its_sum + int(its)
      slv%its_max = max(slv%its_max, int(its))
      slv%nsolve  = slv%nsolve + 1
      call KSPGetConvergedReason(b%ksp, reason, ierr)
      if (reason%v < 0) slv%nfail = slv%nfail + 1
    end associate
  end subroutine bsolve

  !> use_b65 = B_65 has a nonzero entry anywhere (comm: the operators'
  !! communicator; on the mode split the families' union, i.e. comm_g).
  subroutine set_use_b65(a, comm)
    Mat :: a
    integer, intent(in) :: comm
    real*8 :: nrm
    integer :: mpierr, me
    PetscErrorCode :: ierr
    call MatNorm(a, NORM_FROBENIUS, nrm, ierr)
    nrm = nrm**2
    call MPI_Allreduce(MPI_IN_PLACE, nrm, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, mpierr)
    use_b65 = (nrm > 0.d0)
    call MPI_Comm_rank(comm, me, mpierr)
    if (me == 0 .and. use_b65) write(*,'(A,ES10.3,A)') "[Physics PC]   SF: B_65 (T row, rho column) |.|_F = ", &
      sqrt(nrm), " (max over ranks): applied in the rho -> T sweep"
  end subroutine set_use_b65

  !> y = B x for a coupling block of the sweep (k: its index in o_cb), with
  !! a band plus its cross-family part.
  subroutine cmult(a, k, x, y)
    Mat :: a
    integer, intent(in) :: k
    Vec :: x, y
    PetscErrorCode :: ierr
    call MatMult(a, x, y, ierr)
    if (banded) call sff_cross_addmult(x_cb(k), x, y)
  end subroutine cmult

  !--------------------------------------------------------------------
  !> y = P^-1 x: the block-LDU sweep.
  !!
  !!   1. predictor  pair_psi (psi*, j*) = (x_psi, x_j)
  !!      then       rho* , T*  against that explicit predictor
  !!   2. the ONE packed wave solve, pair_w (u, omega)
  !!   3. corrector  pair_psi (dpsi, dj) = (B_12 u + B_16 T*, 0);  psi -= dpsi
  !!      (Eq. (16)), or on "wpj" (default) psi += psi_w, j += j_w from
  !!      pair_w's own solution (Eq. (17), physics_pc_sf_corrector)
  !!      then rho / T: a B_55 / B_66 solve on (B_52 u, B_62 u) (Eq. (16)),
  !!      or with corrector 3 their mass inverse (opz B_44)^-1 (Eq. (17))
  !--------------------------------------------------------------------
  subroutine sf_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr

    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T
    Vec :: parts(4)
    logical, parameter :: keep_uw(4) = [.true., .true., .false., .false.]

    real*8 :: t0, t1

    ierr = 0
    call PetscLogEventBegin(pcev_apply, ierr)

    t0 = MPI_Wtime()
    if (famode) then
      call sff_scatter_in(x, sv_x)
      t_scat = t_scat + (MPI_Wtime() - t0)
    else
      call split_vars(x, sv_x)
    endif
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
    call bsolve(slv_pj, rhs_PJ, sol_PJ)
    call PetscLogEventEnd(pcev_solve_pj, ierr)
    call sf_split_halves(sol_PJ, y_psi, y_j, .false.)     ! psi*, j*

    !--- Step 1: predictor density   rho* = B_55^-1 (x_rho - B_51 psi*) ---
    call cmult(a_51, 7, y_psi, w3)
    call VecWAXPY(w4, -1.0d0, w3, x_rho, ierr)
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call bsolve(slv_rho, w4, t_rho)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)

    !--- Step 1: predictor temperature  T* = B_66^-1 (x_T - B_61 psi* - B_63 j*)
    ! B_61 and B_63 act against the EXPLICIT predictor pair. No j-folded
    ! lower-triangular block is needed or wanted here.
    call cmult(a_61, 9, y_psi, w3)
    call VecWAXPY(w4, -1.0d0, w3, x_T, ierr)
    call cmult(a_63, 11, y_j, w3)
    call VecAXPY(w4, -1.0d0, w3, ierr)
    if (use_b65) then                                   ! - B_65 rho*
      call cmult(a_65, 12, t_rho, w3)
      call VecAXPY(w4, -1.0d0, w3, ierr)
    endif
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call bsolve(slv_T, w4, t_T)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)

    !--- Step 2: the ONE packed wave solve -------------------------------
    !   RHS_u  = x_u - B_21 psi* - B_23 j* - B_25 rho* - B_26 T*
    ! B_21 is the RAW lower coupling and B_23 j* carries the Lorentz path
    ! explicitly. There is deliberately NO -B_24 M_w^-1 x_w term: the
    ! u-omega coupling is the (1,2) entry of pair_w.
    !   RHS_om = x_w VERBATIM -- the omega row of the lower coupling is
    ! identically zero, so no fold and no correction term.
    call VecCopy(x_u, w5, ierr)
    call cmult(a_21, 3, y_psi, w3)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call cmult(a_23, 4, y_j, w3)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call cmult(a_25, 5, t_rho, w3)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call cmult(a_26, 6, t_T, w3)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    ! mixed: the psi / j rows' right-hand side is zero (the predictor has
    ! consumed x_psi, x_j). With the Eq. (16) corrector their solution is
    ! discarded -- step 3 recomputes psi and j by a pair_psi solve. With
    ! Eq. (17) (wpj) it IS the correction: rows 3-4 of pair_w read
    ! M~ (psi_w, j_w) = -(B_12 u + b, 0), M~ the small-flow pair_psi, so
    ! (psi_w, j_w) = -M~^-1 U (u, T*) with b = B_16 T* (corr 1) or 0 (corr 2).
    if (mixed) then
      if (corr == 1) then
        call cmult(a_16, 2, t_T, w4)
        call VecScale(w4, -1.0d0, ierr)
        parts = [w5, x_w, w4, zv]
      else
        parts = [w5, x_w, zv, zv]
      endif
      call sf_split_parts(rhs_W, parts(1:nf_w), .true.)
    else
      call sf_split_halves(rhs_W, w5, x_w, .true.)
    endif
    call PetscLogEventBegin(pcev_solve_w, ierr)
    call bsolve(slv_w, rhs_W, sol_W)
    call PetscLogEventEnd(pcev_solve_w, ierr)
    if (corr /= 0) then
      parts = [y_u, y_w, w3, w4]                          ! w3, w4 = psi_w, j_w
      call sf_split_parts(sol_W, parts, .false.)
    else if (mixed) then
      parts = [y_u, y_w, zv, zv]
      call sf_split_parts(sol_W, parts(1:nf_w), .false., keep=keep_uw(1:nf_w))
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
      call cmult(a_12, 1, y_u, w3)
      call cmult(a_16, 2, t_T, w4)
      call VecAXPY(w3, 1.0d0, w4, ierr)                     ! B_12 u + B_16 T*
      call VecZeroEntries(w4, ierr)
      call sf_split_halves(rhs_PJ, w3, w4, .true.)
      call PetscLogEventBegin(pcev_solve_pj, ierr)
      call bsolve(slv_pj, rhs_PJ, sol_PJ)
      call PetscLogEventEnd(pcev_solve_pj, ierr)
      call sf_split_halves(sol_PJ, w3, w4, .false.)         ! dpsi, dj

      call VecAXPY(y_psi, -1.0d0, w3, ierr)
      call VecAXPY(y_j,   -1.0d0, w4, ierr)
    endif

    !--- Step 3: rho / T correctors. Only u enters -- U's omega COLUMN is zero.
    if (corr == 3) then
      ! Eq. (17) strict: M^-1 ~ the time derivative's, (opz B_44)^-1; no B_65
      ! (zero in model199), no block solve
      call cmult(a_52, 8, y_u, w3)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_rho, -1.0d0 / sf_opz(), w5, t_rho, ierr)
      call cmult(a_62, 10, y_u, w3)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_T, -1.0d0 / sf_opz(), w5, t_T, ierr)
    else
      call cmult(a_52, 8, y_u, w3)
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call bsolve(slv_rho, w3, w5)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
      call VecWAXPY(y_rho, -1.0d0, w5, t_rho, ierr)

      ! [[B_55, 0], [B_65, B_66]] (drho, dT) = (B_52 u, B_62 u): with B_65 the
      ! T correction sees the rho correction just computed (w5)
      call cmult(a_62, 10, y_u, w3)
      if (use_b65) then
        call cmult(a_65, 12, w5, w4)
        call VecAXPY(w3, -1.0d0, w4, ierr)
      endif
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call bsolve(slv_T, w3, w5)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
      call VecWAXPY(y_T, -1.0d0, w5, t_T, ierr)
    endif

    if (famode) then
      t1 = MPI_Wtime()
      call sff_scatter_out(sv_y, y)
      t_scat = t_scat + (MPI_Wtime() - t1)
      t_apply = t_apply + (MPI_Wtime() - t0)
    else
      call merge_vars(sv_y, y)
    endif
    call PetscLogEventEnd(pcev_apply, ierr)
    ierr = 0
  end subroutine sf_apply

  !> Inner-iteration summary, one line per block (under the mode split, one
  !! line per |n| family), then reset. Collective.
  subroutine sf_report(my_id)
    integer, intent(in) :: my_id
    if (famode) then
      call family_report(my_id)
    else
      call sf_solver_report(slv_pj,  my_id)
      call sf_solver_report(slv_w,   my_id)
      call sf_solver_report(slv_rho, my_id)
      call sf_solver_report(slv_T,   my_id)
    endif
    call sf_solver_reset_counters(slv_pj)
    call sf_solver_reset_counters(slv_w)
    call sf_solver_reset_counters(slv_rho)
    call sf_solver_reset_counters(slv_T)
    t_apply = 0.d0;  t_scat = 0.d0
  end subroutine sf_report

  !> Mode split: per |n| family, each block's mean inner its and solve time,
  !! the rest of the sweep (coupling matvecs, vector work) and the two
  !! scatters, all since the last report. Times are the maximum over the
  !! family's ranks. The scatter back to the full system completes only when
  !! every family has finished, so "scatter" holds a family's wait for the
  !! slowest one: the critical family is the one with the smallest.
  subroutine family_report(my_id)
    use mod_petsc_pc_sf_fam, only: sff_fam, sff_nfam
    integer, intent(in) :: my_id
    integer, parameter :: NV = 12
    real*8 :: v(NV), vf(NV)
    real*8, allocatable :: g(:, :)
    integer :: np, r, f, nrk, mpierr
    logical :: done
    v(1:4)  = [its_mean(slv_pj), its_mean(slv_w), its_mean(slv_rho), its_mean(slv_T)]
    v(5:8)  = [slv_pj%t_sum, slv_w%t_sum, slv_rho%t_sum, slv_T%t_sum]
    v(9)    = t_apply - t_scat - sum(v(5:8))
    v(10)   = t_scat
    v(11)   = dble(slv_pj%nfail + slv_w%nfail + slv_rho%nfail + slv_T%nfail)
    v(12)   = dble(sff_fam)
    call MPI_Allreduce(v, vf, NV, MPI_DOUBLE_PRECISION, MPI_MAX, comm_s, mpierr)
    call MPI_Comm_size(comm_g, np, mpierr)
    allocate(g(NV, 0:merge(np - 1, 0, my_id == 0)))
    call MPI_Gather(vf, NV, MPI_DOUBLE_PRECISION, g, NV, MPI_DOUBLE_PRECISION, 0, comm_g, mpierr)
    if (my_id /= 0) return
    write(*,'(A)') "[Physics PC]   SF mode split per family: inner its / solve s (max over its ranks), "// &
      "rest of the sweep, scatters incl. the wait for the slowest family"
    do f = 1, sff_nfam
      nrk = count(nint(g(12, :)) == f)
      done = .false.
      do r = 0, np - 1
        if (nint(g(12, r)) /= f .or. done) cycle
        done = .true.
        write(*,'(A,I0,A,I0,A,4(A,F5.2,A,F7.3),A,F7.3,A,F7.3,A)', advance="no") &
          "[Physics PC]     family ", f, " (", nrk, " rk):", &
          " pair_psi ", g(1, r), " /", g(5, r), &
          "  pair_w ", g(2, r), " /", g(6, r), &
          "  rho ", g(3, r), " /", g(7, r), &
          "  T ", g(4, r), " /", g(8, r), &
          "  rest ", g(9, r), "  scatter ", g(10, r), " s"
        if (g(11, r) > 0.d0) then
          write(*,'(A,I0,A)') ", WARNING ", nint(g(11, r)), " not converged"
        else
          write(*,*)
        endif
      enddo
    enddo
  end subroutine family_report

  real*8 function its_mean(slv)
    type(block_solver_t), intent(in) :: slv
    its_mean = 0.d0
    if (slv%nsolve > 0) its_mean = dble(slv%its_sum) / dble(slv%nsolve)
  end function its_mean

#endif
end module mod_petsc_pc_sf
