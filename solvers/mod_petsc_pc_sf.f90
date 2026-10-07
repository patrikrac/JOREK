module mod_petsc_pc_sf
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var, var_Vpar
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, physics_pc_mumps_mem, &
       pcev_extract, pcev_convert, pcev_build_suu, pcev_fact_pj, pcev_fact_w, &
       pcev_fact_rhot, pcev_solve_pj, pcev_solve_w, pcev_solve_rhot, pcev_apply
  use mod_petsc_pc_blocks, only: create_variable_index_sets, extract_sub_blocks_h, &
       pack_pair_aij, make_pair_block_scale, make_field_block_scale, &
       split_vars, merge_vars, harm_all
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_gather, only: sfg_build, sfg_gather, sfg_cross_weights
  use mod_petsc_pc_sf_pairw, only: sfw_structure, sfw_numeric, sfw_shell, sfw_dh, sfw_lines
  use mod_petsc_pc_sf_mixed, only: sfm_build, sfm_refill, sfm_op, sfm_nf
  use mod_petsc_raw_csr, only: blockmv_attach
  use mod_petsc_pc_mass_cheb, only: mass_cheb_t, mass_cheb_setup, mass_cheb_solve
  use mod_petsc_pc_sf_fam, only: fam_op_t, sff_comm, sff_decompose, sff_op_build, &
       sff_op_fill, sff_vec_setup, sff_scatter_in, sff_scatter_out
  implicit none
  private

  !--------------------------------------------------------------------
  !> The physics-based preconditioner (use_physics_pc): the split-field (SF)
  !! block-LDU sweep over the six variables (psi, u, j, omega, rho, T), and
  !! model600's v_par (var 7, with_vpar) as a seventh block (see V_PAR).
  !!
  !! PRODUCTION CONFIGURATION (the namelist defaults)
  !! ------------------------------------------------
  !!   physics_pc_sf_suu        = "wpj"   pair_w mixed (u, omega, psi, j)
  !!   physics_pc_sf_corrector  = -1      auto: the consistent corrector 4
  !!                                      for wpj
  !!   physics_pc_sf_pair_psi / _pair_w / _rho / _T = "gmg"
  !!   physics_pc_sf_harm_couple = .t.    cross-|n| coupling kept: in the
  !!                                      blocks and in the GMG smoother
  !!                                      blocks (all slots of a node in one
  !!                                      block), on the global operators
  !! Everything else -- the S_uu forms schur / w / wj, the |n|-diagonal
  !! blocks (harm_couple .f., with or without the mode split), the LU
  !! backends, the Eq. (16) corrector -- is a reference arm, kept for
  !! comparison.
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
  !!     second order. Its multigrid smooths with flux-surface ring blocks
  !!     (SF_GMG_SMOOTHER_RINGS), and coarsens radially only on its first
  !!     levels (SF_GMG_SEMI_R_MIXED). With wpj, step 3 reads (psi, j) off
  !!     pair_w's own solution (Eq. (17)) instead of a second pair_psi solve.
  !!   "w": S_uu = B_22 + W, assembled. W is the continuum operator: it lacks
  !!     the discrete projection M_j^-1, so its outer count grows with the mesh.
  !!   "schur": the psi-channel Schur complement of the Jacobian, matrix-free
  !!     (mod_petsc_pc_sf_pairw), S_uu u = B_22 u - B_21 psi - B_23 j with
  !!     [psi; j] = pair_psi^-1 [B_12 u; 0]; its multigrid keeps B_22 + W.
  !!
  !! HARMONIC COUPLING AND THE MODE SPLIT
  !! ------------------------------------
  !! physics_pc_sf_harm_couple is one switch for both places the toroidal
  !! harmonics meet: the block extraction (harm_kept) and the GMG smoother
  !! blocks (gmg_opts_t%harm_pair, sf_gmg_harm). .t.: every cross-|n| entry
  !! is kept and a node's slots all sit in one smoother block, so the
  !! coupling a nonlinear or 3D equilibrium carries at fine radial scales
  !! is smoothed where no coarse level resolves it; the ring blocks are
  !! then solved by the block Thomas / hybrid solvers (mod_gmg_ring_btd).
  !! .f.: every block is |n|-diagonal and a harmonic's cos and sin slots
  !! share the smoother blocks (d_phi maps one to the other). Then each |n|
  !! family (n = 0, then each cos/sin pair) is solved on its own rank group,
  !! all families concurrently (physics_pc_sf_mode_split,
  !! mod_petsc_pc_sf_fam). The operators are extracted straight from A and W
  !! into the family layout; no global operator exists. It needs at least
  !! one rank per family.
  !!
  !! V_PAR
  !! -----
  !! model600 built with_vpar (TiTe, neutrals, impurities off): v_par is its
  !! own block, B_77, on GMG hierarchy 5 or LU (physics_pc_sf_vpar). Its row
  !! couples to psi, u, rho, T (B_71, B_72, B_75 + B_76 = theta dt v D_B dp);
  !! only the u, rho and T rows have a v_par column (B_27 ~ 0 at small flow,
  !! not used). physics_pc_sf_vpar_stage:
  !!   "v0": solved LAST, B_77 V = x_V - B_71 psi - B_72 u - B_75 rho - B_76 T
  !!         against the final psi, u, rho, T; nothing else sees V.
  !!   "b0": the sequential velocity stage: right after pair_w,
  !!         (B_77 + W_vv) V = x_V - B_71 psi* - B_75 rho* - B_76 T* - (B_72 + W_vu) u,
  !!         then the rho / T correctors with B_57 V and B_67 V. W_vv, W_vu
  !!         are the sound-wave Schur terms (mod_pc_elt_matrix_force_fft,
  !!         V_PAR): -sf_vpar_w picks them (bit 0 W_vv, bit 1 W_vu; 0 = b0
  !!         without them), -sf_vpar_gamma_w their gamma. psi* (not psi_w)
  !!         on v_par's right-hand side: W_vu leaves out the part of the p
  !!         channel that cancels against the psi channel.
  !!   "b1": v_par bundled with u: pair_w (wpj) gets v_par as its 5th field,
  !!         [[B_22 + W, B_24, B_21, B_23, B_27 + W_uv], [B_42, B_44, 0, 0, 0],
  !!          [B_12, 0, opz B_33, B_13, 0], [0, 0, B_31, B_33, 0],
  !!          [B_72 + W_vu, 0, 0, 0, B_77 + W_vv]],
  !!         right-hand side x_V - B_71 psi* - B_75 rho* - B_76 T*; V read off
  !!         pair_w's solution with (psi_w, j_w), then the correctors as in b0.
  !!         -sf_vpar_w bit 2 adds W_uv. Mode split and Eq. (17) only.
  !!
  !! CONFIGURATION
  !! -------------
  !! The GMG smoothers, V-cycle shapes, axis rings and boundary drop are fixed
  !! in mod_petsc_pc_sf_solver; the GMG receives its whole configuration
  !! explicitly (gmg_opts_t). The namelist picks S_uu, gmg | lu per block, the
  !! shared inner tolerance, the corrector, the harmonic coupling and the
  !! mode split.
  !--------------------------------------------------------------------

  logical, save :: sf_init_done = .false.
  logical, save :: sf_first     = .true.

  !--- the block solvers (v_par's: slv_v, below) -------------------------
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
  !> EXPERIMENT (-sf_seq, see sf_core_sweep): 0 the LDU sweep below; 1 core
  !! first, block Gauss--Seidel on {(u, omega, psi, j) | (rho, T)}; 2 block
  !! LDU on that split. Both use pair_w (wpj) as the whole core: no pair_psi
  !! solve.
  integer, save :: seq = 0

  !--- v_par (model600 with_vpar, var 7; see V_PAR) ------------------------
  logical, parameter :: with_v = (var_Vpar > 0)
  integer, parameter :: NSV = merge(7, 6, with_v)     !< the variables the sweep splits
  type(block_solver_t), save :: slv_v
  integer, save :: bk_v   = SF_GMG
  integer, save :: vstage = SF_VPAR_V0          !< physics_pc_sf_vpar_stage
  integer, save :: vwmask = 0                   !< "b0": bit 0 W_vv on B_77, bit 1 W_vu on B_72
  logical, save :: vfull = .false.              !< -sf_vpar_wvu_full: whole-p-channel W_vu, psi_w in v_par's row
  Mat, save :: a_v                              !< B_77 (+ W_vv)
  !> v_par's coupling blocks, a_vc(VC_ij) = B_ij; a_vc(VC72) = B_72 (+ W_vu).
  !! Under the mode split coupling block o_cb(12 + VC_ij).
  integer, parameter :: VC71 = 1, VC72 = 2, VC75 = 3, VC76 = 4, VC57 = 5, VC67 = 6
  Mat, save :: a_vc(6)
  !> the W sources of the mode split's operators (fam_op_t wid): W, W_vv,
  !! W_vu, W_uv
  integer, parameter :: WS_C = 1, WS_VV = 2, WS_VU = 3, WS_UV = 4, WS_N = 4
  !> the global path's extracted v_par blocks, and "b0"'s sums with W
  Mat, save :: gB_71 = PETSC_NULL_MAT, gB_72 = PETSC_NULL_MAT, gB_75 = PETSC_NULL_MAT, &
               gB_76 = PETSC_NULL_MAT, gB_77 = PETSC_NULL_MAT, gB_57 = PETSC_NULL_MAT, &
               gB_67 = PETSC_NULL_MAT
  !> (not PETSC_NULL_MAT-initialised: PETSc's Fortran stubs read a null
  !! output argument as "not wanted" and MatDuplicate would get NULL)
  Mat, save :: vop_g, c72_g
  logical, save :: vop_on = .false., c72_on = .false.
  Vec, save :: wv1, wv2
  logical, save :: vcheck_done = .false.         !< -sf_vpar_wcheck: once, at the first apply

  !--- work state owned by this path --------------------------------------
  Vec, save :: sv_x(NSV), sv_y(NSV)
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
  !! temperature equation), the O(1) mass (1+zeta) T0 drho in model600 and
  !! model183 (a pressure equation). It is applied only where it is not zero (use_b65), so
  !! model199 runs are unchanged.
  !! With v_par six more: 13 B_71, 14 B_72 (+ W_vu), 15 B_75, 16 B_76, and
  !! for "b0" 17 B_57, 18 B_67; ncb_on of them are in use.
  integer, parameter :: NCB = 12 + merge(6, 0, with_v)
  integer, save :: ncb_on = 12
  !> the diagonal operators' message tags 1 .. NDIAG (pair_w, pair_psi, rho,
  !! T, v_par); the coupling blocks' follow
  integer, parameter :: NDIAG = 5
  logical, save :: use_b65 = .false.
  Mat, save :: gB_65 = PETSC_NULL_MAT          !< global path's B_65 (g_ctx has no slot)
  type(fam_op_t), save :: o_pj, o_w, o_rho, o_T, o_v, o_cb(NCB)
  !> corr 3: the rho / T time-derivative mass. In model199 both rows carry
  !! (1 + zeta) v phi R, which is the omega mass B_44 times opz; B_44 is
  !! geometry-only, so its Chebyshev inverse is set up once and opz divided out
  type(fam_op_t), save :: o_m
  Mat, save :: a_m
  type(mass_cheb_t), save :: mcm

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
                           physics_pc_sf_mode_split, physics_pc_sf_vpar, physics_pc_sf_vpar_stage
    use mod_parameters, only: n_tor, var_T
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
    !--- of model199 -- which model600 has with its extensions off -- and
    !--- v_par as var 7 (model600, and the stellarator model183). The slot
    !--- layout (n = 0, then cos/sin pairs) needs n_tor odd -- an even n_tor
    !--- would leave the last slot in no family (fslots).
    if (jorek_model /= 199 .and. jorek_model /= 600 .and. jorek_model /= 183) &
      call fatal("physics_pc_sf supports model199, model600 and model183 only")
    if (n_var /= 6 .and. .not. ((jorek_model == 600 .or. jorek_model == 183) .and. n_var == 7 .and. &
                                var_Vpar == 7 .and. var_T == 6)) &
      call fatal("physics_pc_sf supports (psi, u, zj, w, rho, T), plus v_par on model600 / model183: build "// &
                 "with with_TiTe, with_neutrals and with_impurities off")
    if (mod(n_tor, 2) /= 1) call fatal("physics_pc_sf needs n_tor odd (n = 0 plus cos/sin pairs)")

    !--- the cross-|n| coupling, one switch for the run: the extraction keeps
    !--- every slot pair (harm_kept) and every GMG smoother block holds all
    !--- of a node's slots, or both stay |n|-diagonal / cos-sin paired
    harm_all = physics_pc_sf_harm_couple
    sf_gmg_harm = merge(SF_GMG_HARM_ALL, SF_GMG_HARM_PAIR, harm_all)

    !--- the corrector: Eq. (17) reads (psi, j) off pair_w, so it needs the
    !--- one form that carries both explicitly
    corr = physics_pc_sf_corrector
    if (corr < -1 .or. corr > 4) call fatal("physics_pc_sf_corrector must be -1 (auto), 0, 1, 2, 3 or 4")
    if (corr == -1) then
      corr = 0
      if (suu == SF_SUU_WPJ) corr = 4
    endif
    if (corr /= 0 .and. suu /= SF_SUU_WPJ) &
      call fatal("physics_pc_sf_corrector = 1 | 2 | 3 | 4 needs physics_pc_sf_suu = wpj (pair_w must carry psi and j)")
    ! corr 3 takes the rho / T mass as opz B_44: model199's rows; model600's T
    ! row is a pressure equation (mass rho0 v phi R, plus a rho column)
    if (corr == 3 .and. jorek_model /= 199) &
      call fatal("physics_pc_sf_corrector = 3 supports model199 only (rho / T mass = opz B_44)")

    !--- EXPERIMENT: the core-first sweeps (-sf_seq 1 | 2, sf_core_sweep)
    block
      PetscInt :: iv
      PetscBool :: set
      iv = 0
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_seq", iv, set, ierr)
      seq = int(iv)
    end block
    if (seq < 0 .or. seq > 2) call fatal("-sf_seq must be 0, 1 or 2")
    if (seq /= 0 .and. (suu /= SF_SUU_WPJ .or. with_v)) &
      call fatal("-sf_seq 1 | 2 needs physics_pc_sf_suu = wpj and no v_par")

    !--- one |n| family per rank: only where every block is |n|-diagonal
    !--- (physics_pc_sf_mode_split is read only then), and only for the
    !--- assembled pair_w forms (the schur shell is not split)
    famode = physics_pc_sf_mode_split .and. .not. harm_all
    if (famode .and. (suu == SF_SUU_SCHUR .or. suu == SF_SUU_WJ)) &
      call fatal("physics_pc_sf_mode_split needs pair_w assembled entry by entry from A and W: "// &
                 "physics_pc_sf_suu = w | wpj")

    !--- v_par: its backend, its place in the sweep, and the W terms of "b0"
    if (with_v) then
      bk_v   = backend_of(physics_pc_sf_vpar, "physics_pc_sf_vpar")
      vstage = sf_vpar_stage()
      if (vstage < 0) call fatal("physics_pc_sf_vpar_stage must be v0 | b0 | b1, got '"// &
                                 trim(physics_pc_sf_vpar_stage)//"'")
      ! b0 / b1 need the sound-wave Schur terms W_vv, W_vu, W_uv, assembled
      ! only by model600's pc_elt_matrix_force_fft
      if (vstage /= SF_VPAR_V0 .and. jorek_model /= 600) &
        call fatal("physics_pc_sf_vpar_stage = b0 | b1 needs model600 (W_vv): run v0")
      ncb_on = 16
      if (vstage /= SF_VPAR_V0) then
        ncb_on = 18
        vwmask = sf_vpar_wmask()
        if (vwmask < 0 .or. vwmask > 7) call fatal("-sf_vpar_w must be 0 .. 7 (bit 0 W_vv, bit 1 W_vu, "// &
                                                   "bit 2 W_uv)")
      endif
      ! -sf_vpar_wvu_full: psi_w enters v_par's row, so it must exist (Eq. (17))
      if (vstage /= SF_VPAR_V0) vfull = sf_vpar_wvu_full()
      if (vfull .and. corr == 0) &
        call fatal("-sf_vpar_wvu_full needs the Eq. (17) corrector (pair_w's psi_w)")
      ! b1 packs v_par into the mixed pair_w, which only the mode split builds
      ! with v_par, and reads it off pair_w's solution like (psi_w, j_w)
      if (vstage == SF_VPAR_B1) then
        if (.not. famode) call fatal("physics_pc_sf_vpar_stage = b1 needs the mode split "// &
                                     "(physics_pc_sf_harm_couple = .f., physics_pc_sf_mode_split = .t.)")
        if (suu /= SF_SUU_WPJ) call fatal("physics_pc_sf_vpar_stage = b1 needs physics_pc_sf_suu = wpj")
        if (corr /= 1 .and. corr /= 2) &
          call fatal("physics_pc_sf_vpar_stage = b1 needs the Eq. (17) corrector (physics_pc_sf_corrector 1 | 2)")
      endif
    endif

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
      if (harm_all) then
        write(*,'(A)') "[Physics PC]   cross-|n|: all kept, all slots per GMG smoother block "// &
                       "(physics_pc_sf_harm_couple = .t.)"
      else
        write(*,'(A)') "[Physics PC]   cross-|n|: none, |n|-diagonal blocks, cos / sin per GMG smoother block "// &
                       "(physics_pc_sf_harm_couple = .f.)"
      endif
      if (famode) then
        write(*,'(A)') "[Physics PC]   layout   : one |n| family per rank, families concurrent "// &
                       "(physics_pc_sf_mode_split)"
      else
        write(*,'(A)') "[Physics PC]   layout   : global operators, every block on every rank"
      endif
      if (with_v) then
        write(*,'(A,A)') "[Physics PC]   v_par    : ", trim(physics_pc_sf_vpar)
        if (vstage == SF_VPAR_V0) then
          write(*,'(A)') "[Physics PC]   v_par    : v0, B_77 solved last against the final psi, u, rho, T"
        else if (vstage == SF_VPAR_B1) then
          write(*,'(A,L1,A,L1,A,L1,A,F6.3,A)') "[Physics PC]   v_par    : b1, in pair_w (5 fields), "// &
            "rho / T correctors with B_57 V, B_67 V; W_vv ", btest(vwmask, 0), ", W_vu ", btest(vwmask, 1), &
            ", W_uv ", btest(vwmask, 2), ", gamma_w ", sf_vpar_gamma_w(), " (-sf_vpar_w, -sf_vpar_gamma_w)"
          if (vfull) write(*,'(A)') "[Physics PC]   v_par    : W_vu with the whole p channel, B_71 psi_w in "// &
            "v_par's row (-sf_vpar_wvu_full)"
        else
          write(*,'(A,L1,A,L1,A,F6.3,A)') "[Physics PC]   v_par    : b0, (B_77 + W_vv) after pair_w, "// &
            "rho / T correctors with B_57 V, B_67 V; W_vv ", btest(vwmask, 0), ", W_vu ", btest(vwmask, 1), &
            ", gamma_w ", sf_vpar_gamma_w(), " (-sf_vpar_w, -sf_vpar_gamma_w)"
          if (vfull) write(*,'(A)') "[Physics PC]   v_par    : W_vu with the whole p channel, B_71 psi_w in "// &
            "v_par's right-hand side (-sf_vpar_wvu_full)"
        endif
      endif
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
      case (4)
        write(*,'(A)') "[Physics PC]   corrector: consistent (Alg. 1), (psi, j) from pair_w, B_16 T* on its psi row, "// &
                       "rho / T corrections with B_51 psi_w, B_61 psi_w + B_63 j_w"
      end select
      select case (seq)
      case (1)
        write(*,'(A)') "[Physics PC]   SEQUENCE : EXPERIMENT -sf_seq 1, core first: pair_w on (x_u, x_w, x_psi, x_j), "// &
                       "then rho / T against (psi, j, u); no pair_psi solve, the corrector setting unused"
      case (2)
        write(*,'(A)') "[Physics PC]   SEQUENCE : EXPERIMENT -sf_seq 2, block LDU on core | (rho, T): rho*, T*, "// &
                       "pair_w, rho / T correctors; no pair_psi solve, the corrector setting unused"
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
        if (with_v) call vpar_global_sums(.false.)
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
      if (with_v) then
        a_vc = [gB_71, gB_72, gB_75, gB_76, gB_57, gB_67]
        a_v  = gB_77
        if (vop_on) a_v = vop_g
        if (c72_on) a_vc(VC72) = c72_g
      endif
      comm_s = comm
    endif
    if (first .and. with_v .and. vstage /= SF_VPAR_V0) call vpar_w_report()
    call physics_pc_mem("SF build: operators filled", my_id)

    !--- symmetric block scaling, an exact similarity applied to the STORED
    !--- operator, on both pairs.
    call MatGetLocalSize(a_rho, n1_loc, PETSC_NULL_INTEGER, ierr)
    if (slv_pj%scaled) call VecDestroy(slv_pj%dscale, ierr)
    call make_pair_block_scale(a_pj, n1_loc, slv_pj%dscale, comm_s, my_id, "pair_psi")
    slv_pj%scaled = .true.

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
                           ring_overlap=SF_GMG_RING_OVERLAP, &
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
    ! T: chi_par diffuses along B, around the flux surfaces -- pair_w's ring
    ! blocks and radial semi-coarsening (see SF_GMG_SMOOTHER_RINGS)
    call sf_solver_setup(slv_T,   a_T,   bk_T,   "T-block KSP", &
                         comm_s, my_id, physics_pc_sf_rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS_RHOT, axis_rings=ax_rhot, &
                         ring_overlap=SF_GMG_RING_OVERLAP, semi_r=SF_GMG_SEMI_R_MIXED)
    !--- v_par (GMG hierarchy 5). v0: B_77 is mass-dominated and nearly
    !--- isotropic (mass + poloidal visco_par), so the rho setup. b0: the
    !--- W_vv sound term is diffusion ALONG B -- along the flux surfaces, and
    !--- cos <-> sin through d_phi -- so pair_w's ring blocks and its radial
    !--- semi-coarsening; Richardson damped
    !--- as on the lines. Both untuned starting points (-sf_gmg5_* override).
    if (with_v .and. vstage /= SF_VPAR_B1) then         ! b1: v_par is in pair_w
      if (vstage == SF_VPAR_V0) then
        call sf_solver_setup(slv_v, a_v, bk_v, "v_par-block KSP (B_77)", &
                             comm_s, my_id, physics_pc_sf_rtol, gmg_inst=5, nfields=1, &
                             smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT, axis_rings=ax_rhot)
      else
        call sf_solver_setup(slv_v, a_v, bk_v, "v_par-block KSP (B_77 + W_vv)", &
                             comm_s, my_id, physics_pc_sf_rtol, gmg_inst=5, nfields=1, &
                             smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, axis_rings=ax_rhot, &
                             ring_overlap=SF_GMG_RING_OVERLAP, &
                             semi_r=SF_GMG_SEMI_R_MIXED, rich_omega=SF_GMG_RICH_OMEGA_ADDITIVE)
      endif
    endif
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
    call physics_pc_mem("SF build: solvers set up", my_id)

    !--- work vectors: the operators keep their layout for the run.
    if (.not. vecs_ready) then
      call MatCreateVecs(a_pj, rhs_PJ, sol_PJ, ierr)
      call MatCreateVecs(a_w, rhs_W, sol_W, ierr)
      call MatCreateVecs(a_rho, sv_x(1), PETSC_NULL_VEC, ierr)
      block
        integer :: k
        do k = 2, NSV
          call VecDuplicate(sv_x(1), sv_x(k), ierr)
        enddo
        do k = 1, NSV
          call VecDuplicate(sv_x(1), sv_y(k), ierr)
        enddo
      end block
      if (with_v) then
        call VecDuplicate(sv_x(1), wv1, ierr)
        call VecDuplicate(sv_x(1), wv2, ierr)
      endif
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
          call sff_vec_setup(xt, NSV)
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
      Mat :: ws(WS_N)
      ! the W sources the operators read (fam_op_t wid): W, and v_par's three
      ! where assembled (W stands in otherwise; no operator reads them then)
      ws = g_ctx%W_force
      if (g_ctx%w_v_ready) then
        ws(WS_VV) = g_ctx%W_vv; ws(WS_VU) = g_ctx%W_vu; ws(WS_UV) = g_ctx%W_uv
      endif
      if (first_) then
        if (.not. g_ctx%w_force_ready) then
          if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: W_force was never assembled "// &
            "(petsc_assemble_pc_matrices did not run?)."
          call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
        call sff_decompose(A_full, comm, my_id)
        call define_family_operators()
        call PetscLogEventBegin(pcev_extract, ierr)
        call sff_op_build(o_pj,  A_full, ws, sf_opz(), my_id)
        call sff_op_build(o_w,   A_full, ws, sf_opz(), my_id)
        call sff_op_build(o_rho, A_full, ws, sf_opz(), my_id)
        call sff_op_build(o_T,   A_full, ws, sf_opz(), my_id)
        if (with_v .and. vstage /= SF_VPAR_B1) call sff_op_build(o_v, A_full, ws, sf_opz(), my_id)
        if (corr == 3) call sff_op_build(o_m, A_full, ws, sf_opz(), my_id)
        do k = 1, ncb_on
          call sff_op_build(o_cb(k), A_full, ws, sf_opz(), my_id)
        enddo
        call PetscLogEventEnd(pcev_extract, ierr)
      else
        call PetscLogEventBegin(pcev_extract, ierr)
        call sff_op_fill(o_pj,  A_full, ws, sf_opz())
        call sff_op_fill(o_w,   A_full, ws, sf_opz())
        call sff_op_fill(o_rho, A_full, ws, sf_opz())
        call sff_op_fill(o_T,   A_full, ws, sf_opz())
        if (with_v .and. vstage /= SF_VPAR_B1) call sff_op_fill(o_v, A_full, ws, sf_opz())
        do k = 1, ncb_on
          call sff_op_fill(o_cb(k), A_full, ws, sf_opz())
        enddo
        call PetscLogEventEnd(pcev_extract, ierr)
      endif
      a_pj = o_pj%fam;  a_w = o_w%fam;  a_rho = o_rho%fam;  a_T = o_T%fam
      if (corr == 3) a_m = o_m%fam
      nf_w = o_w%nf
      a_12 = o_cb(1)%fam; a_16 = o_cb(2)%fam; a_21 = o_cb(3)%fam;  a_23 = o_cb(4)%fam
      a_25 = o_cb(5)%fam; a_26 = o_cb(6)%fam; a_51 = o_cb(7)%fam;  a_52 = o_cb(8)%fam
      a_61 = o_cb(9)%fam; a_62 = o_cb(10)%fam; a_63 = o_cb(11)%fam; a_65 = o_cb(12)%fam
      if (with_v) then
        if (vstage /= SF_VPAR_B1) a_v = o_v%fam
        do k = 13, ncb_on
          a_vc(k - 12) = o_cb(k)%fam
        enddo
      endif
      if (first_) call set_use_b65(a_65, comm)
      !--- the sweep's coupling blocks on the threaded block kernel, as on the
      !--- global path; the kernel reads the CSR arrays in place, so one
      !--- attach holds across the refills
      if (first_) then
        block
          use mod_petsc_pc_harm, only: pc_ntor
          integer :: nno
          nno = 0
          do k = 1, ncb_on
            if (.not. blockmv_attach(o_cb(k)%fam, int(pc_ntor))) nno = nno + 1
          enddo
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
      integer, parameter :: ce(12) = [var_psi, var_psi, var_u, var_u, var_u, var_u, &
                                      var_rho, var_rho, var_T, var_T, var_T, var_T]
      integer, parameter :: cv(12) = [var_u, var_T, var_psi, var_zj, var_rho, var_T, &
                                      var_psi, var_u, var_psi, var_u, var_zj, var_rho]
      character(len=4), parameter :: cn(12) = ["B_12", "B_16", "B_21", "B_23", "B_25", "B_26", &
                                                "B_51", "B_52", "B_61", "B_62", "B_63", "B_65"]
      !> v_par's: B_71 B_72 B_75 B_76, and for "b0" B_57 B_67 (o_cb 13 .. 18)
      integer, parameter :: cev(6) = [var_Vpar, var_Vpar, var_Vpar, var_Vpar, var_rho, var_T]
      integer, parameter :: cvv(6) = [var_psi, var_u, var_rho, var_T, var_Vpar, var_Vpar]
      character(len=4), parameter :: cnv(6) = ["B_71", "B_72", "B_75", "B_76", "B_57", "B_67"]  ! VC_ij order
      integer :: k
      call full_op(o_pj, "pair_psi", 1, [var_psi, var_zj])
      if (suu == SF_SUU_WPJ .and. with_v .and. vstage == SF_VPAR_B1) then
        ! b1: v_par bundled with u -- [[B_22 + W, B_24, B_21, B_23, B_27 + W_uv],
        ! [B_42, B_44, 0, 0, 0], [B_12, 0, opz B_33, B_13, 0], [0, 0, B_31, B_33, 0],
        ! [B_72 + W_vu, 0, 0, 0, B_77 + W_vv]]: v_par's row and column carry
        ! only u (psi_w must not enter v_par's row, see V_PAR)
        call full_op(o_w, "pair_w (wpj + v_par)", 2, [var_u, var_w, var_psi, var_zj, var_Vpar])
        o_w%se(2, 3:5) = 0; o_w%se(3, 2) = 0; o_w%se(3, 5) = 0; o_w%se(4, 1:2) = 0; o_w%se(4, 5) = 0
        o_w%se(5, 2:4) = 0
        o_w%se(3, 3) = var_zj; o_w%sv(3, 3) = var_zj; o_w%scl(3, 3) = 1     ! opz M_psi = opz B_33
        if (btest(vwmask, 0)) o_w%wid(5, 5) = WS_VV
        if (btest(vwmask, 1)) o_w%wid(5, 1) = WS_VU
        if (btest(vwmask, 2)) o_w%wid(1, 5) = WS_UV
        ! -sf_vpar_wvu_full: the psi channel explicit in v_par's row, B_71 on psi_w
        if (vfull) then
          o_w%se(5, 3) = var_Vpar; o_w%sv(5, 3) = var_psi
        endif
      else if (suu == SF_SUU_WPJ) then
        call full_op(o_w, "pair_w (wpj)", 2, [var_u, var_w, var_psi, var_zj])
        o_w%se(2, 3:4) = 0; o_w%se(3, 2) = 0; o_w%se(4, 1:2) = 0
        o_w%se(3, 3) = var_zj; o_w%sv(3, 3) = var_zj; o_w%scl(3, 3) = 1     ! opz M_psi = opz B_33
        if (jorek_model == 183) then                     ! B_11 (mod_petsc_pc_sf_mixed, psi_mass_b11)
          o_w%se(3, 3) = var_psi; o_w%sv(3, 3) = var_psi; o_w%scl(3, 3) = 0
        endif
      else
        call full_op(o_w, "pair_w", 2, [var_u, var_w])
      endif
      o_w%wid(1, 1) = WS_C
      call full_op(o_rho, "B_55", 3, [var_rho])
      call full_op(o_T,   "B_66", 4, [var_T])
      if (with_v .and. vstage /= SF_VPAR_B1) then
        call full_op(o_v, "B_77", 5, [var_Vpar])
        if (vstage == SF_VPAR_B0 .and. btest(vwmask, 0)) then
          o_v%wid(1, 1) = WS_VV; o_v%label = "B_77 + W_vv"
        endif
      endif
      ! tags 1 .. NDIAG + NCB are the diagonal and coupling operators'
      if (corr == 3) call full_op(o_m, "B_44", NDIAG + NCB + 1, [var_w])
      do k = 1, ncb_on
        o_cb(k)%nf = 1; o_cb(k)%tagb = NDIAG + k
        if (k <= 12) then
          o_cb(k)%label = cn(k); o_cb(k)%se(1, 1) = ce(k); o_cb(k)%sv(1, 1) = cv(k)
        else
          o_cb(k)%label = cnv(k - 12); o_cb(k)%se(1, 1) = cev(k - 12); o_cb(k)%sv(1, 1) = cvv(k - 12)
          ! "b0": B_72 carries W_vu
          if (k - 12 == VC72 .and. vstage == SF_VPAR_B0 .and. btest(vwmask, 1)) then
            o_cb(k)%wid(1, 1) = WS_VU; o_cb(k)%label = "B_72 + W_vu"
          endif
        endif
      enddo
    end subroutine define_family_operators

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
      integer, parameter :: NVP = merge(7, 0, with_v)      !< v_par's blocks
      integer, parameter :: NBLK = 22 + NVP
      integer :: eqs(NBLK), vrs(NBLK), nkeep
      Mat :: M(NBLK), S_uu, none(0)

      if (.not. g_ctx%w_force_ready) then
        if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: W_force was never assembled "// &
          "(petsc_assemble_pc_matrices did not run?)."
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      endif

      !--- the 22 blocks (29 with v_par) in ONE pass over A_full's rows. The
      !--- first 14 are the ones the apply and the rho/T solvers read (the 14th
      !--- is B_65), then v_par's 7, the next 3 (B_13, B_31, B_33) the ones the
      !--- schur shell reads; the rest only feed the packed pairs and are
      !--- released once the maps exist.
      nkeep = 14 + NVP
      if (suu == SF_SUU_SCHUR) nkeep = 17 + NVP
      if (mixed) nkeep = NBLK                 ! the mixed pair_w repacks from all of them
      call PetscLogEventBegin(pcev_extract, ierr)
      eqs(1:14) = [var_psi, var_psi, var_u, var_u, var_u, var_u, var_rho, var_rho, var_rho, &
                   var_T, var_T, var_T, var_T, var_T]
      vrs(1:14) = [var_u, var_T, var_psi, var_zj, var_rho, var_T, var_psi, var_u, var_rho, &
                   var_psi, var_u, var_zj, var_T, var_rho]
      if (with_v) then                                     ! B_71 B_72 B_75 B_76 B_77 B_57 B_67
        eqs(15:21) = [var_Vpar, var_Vpar, var_Vpar, var_Vpar, var_Vpar, var_rho, var_T]
        vrs(15:21) = [var_psi, var_u, var_rho, var_T, var_Vpar, var_Vpar, var_Vpar]
      endif
      eqs(15+NVP:NBLK) = [var_psi, var_zj, var_zj, var_psi, var_u, var_u, var_w, var_w]
      vrs(15+NVP:NBLK) = [var_zj, var_psi, var_zj, var_psi, var_u, var_w, var_u, var_w]
      call extract_sub_blocks_h(A_full, eqs, vrs, M, .true.)
      g_ctx%B_12 = M(1);  g_ctx%B_16 = M(2);  g_ctx%B_21 = M(3);  g_ctx%B_23 = M(4)
      g_ctx%B_25 = M(5);  g_ctx%B_26 = M(6);  g_ctx%B_51 = M(7);  g_ctx%B_52 = M(8)
      g_ctx%B_55 = M(9);  g_ctx%B_61 = M(10); g_ctx%B_62 = M(11); g_ctx%B_63 = M(12)
      g_ctx%B_66 = M(13); gB_65 = M(14)
      if (with_v) then
        gB_71 = M(15); gB_72 = M(16); gB_75 = M(17); gB_76 = M(18); gB_77 = M(19)
        gB_57 = M(20); gB_67 = M(21)
      endif
      g_ctx%B_13 = M(15+NVP); g_ctx%B_31 = M(16+NVP); g_ctx%B_33 = M(17+NVP); g_ctx%B_11 = M(18+NVP)
      g_ctx%B_22 = M(19+NVP); g_ctx%B_24 = M(20+NVP); g_ctx%B_42 = M(21+NVP); g_ctx%B_44 = M(22+NVP)
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
      if (nkeep < 17 + NVP) then
        g_ctx%B_13 = PETSC_NULL_MAT; g_ctx%B_31 = PETSC_NULL_MAT; g_ctx%B_33 = PETSC_NULL_MAT
      endif
      ! "b0": B_77 + W_vv and B_72 + W_vu, refilled from the gathered blocks
      if (with_v) call vpar_global_sums(.true.)
      if (nkeep < NBLK) then
        g_ctx%B_11 = PETSC_NULL_MAT; g_ctx%B_22 = PETSC_NULL_MAT; g_ctx%B_24 = PETSC_NULL_MAT
        g_ctx%B_42 = PETSC_NULL_MAT; g_ctx%B_44 = PETSC_NULL_MAT
      endif

      !--- the sweep's coupling blocks: fixed patterns, refilled in place by
      !--- the value map, so one attach holds for the run
      block
        use mod_parameters, only: n_tor
        integer :: nno, ncb, k
        Mat :: cb(20)
        ! a refused attach (not AIJ) keeps PETSc's own matvec: correct, but
        ! single-threaded, so it is reported
        cb(1:12) = [g_ctx%B_12, g_ctx%B_16, g_ctx%B_21, g_ctx%B_23, g_ctx%B_25, g_ctx%B_26, &
                    g_ctx%B_51, g_ctx%B_52, g_ctx%B_61, g_ctx%B_62, g_ctx%B_63, gB_65]
        ncb = 12
        if (suu == SF_SUU_SCHUR) then
          cb(13:14) = [g_ctx%B_31, pw0]
          ncb = 14
        endif
        if (with_v) then
          cb(ncb + 1:ncb + 4) = [gB_71, gB_75, gB_76, gB_72]
          ncb = ncb + 4
          if (c72_on) then
            ncb = ncb + 1;  cb(ncb) = c72_g
          endif
          if (vstage == SF_VPAR_B0) then
            cb(ncb + 1:ncb + 2) = [gB_57, gB_67]
            ncb = ncb + 2
          endif
        endif
        nno = 0
        do k = 1, ncb
          if (.not. blockmv_attach(cb(k), int(n_tor))) nno = nno + 1
        enddo
        if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   SF: WARNING ", nno, &
          " coupling block(s) not AIJ, left on PETSc's single-threaded matvec"
      end block
    end subroutine first_build_operators

    !> The global path's "b0" operators of v_par: vop_g = B_77 + W_vv,
    !! c72_g = B_72 + W_vu (as -sf_vpar_w selects), the union pattern fixed at
    !! the first build (as S_uu = B_22 + W), refilled from the gathered blocks
    !! at every later one.
    subroutine vpar_global_sums(first_)
      logical, intent(in) :: first_
      if (vstage /= SF_VPAR_B0) return
      if (.not. g_ctx%w_v_ready) then
        if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: v_par's W_vv / W_vu were never "// &
          "assembled (physics_pc_sf_vpar_stage = b0 needs them)."
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      endif
      if (btest(vwmask, 0)) call vpar_one_sum(vop_g, gB_77, g_ctx%W_vv, first_)
      if (btest(vwmask, 1)) call vpar_one_sum(c72_g, gB_72, g_ctx%W_vu, first_)
      vop_on = btest(vwmask, 0);  c72_on = btest(vwmask, 1)
    end subroutine vpar_global_sums

    !> S = B + Wm: the union pattern created at the first build, refilled after.
    subroutine vpar_one_sum(S, B, Wm, first_)
      Mat :: S, B, Wm
      logical, intent(in) :: first_
      if (first_) then
        call MatDuplicate(B, MAT_COPY_VALUES, S, ierr)
        call MatAXPY(S, 1.0d0, Wm, DIFFERENT_NONZERO_PATTERN, ierr)
      else
        call MatCopy(B, S, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(S, 1.0d0, Wm, SUBSET_NONZERO_PATTERN, ierr)
      endif
    end subroutine vpar_one_sum

    !> "b0", first build: v_par's W terms, their size against the blocks they
    !! are added to, and the (u, v_par) symmetry defect
    !! |W_vu + W_uv^T|_F / |W_vu|_F -- zero up to the P_B = D_B p0 terms,
    !! i.e. for an axisymmetric p0 = p0(psi0) (mod_pc_elt_matrix_force_fft).
    subroutine vpar_w_report()
      Mat :: D
      real*8 :: nvv, nvu, nuv, nd
      call MatNorm(g_ctx%W_vv, NORM_FROBENIUS, nvv, ierr)
      call MatNorm(g_ctx%W_vu, NORM_FROBENIUS, nvu, ierr)
      call MatNorm(g_ctx%W_uv, NORM_FROBENIUS, nuv, ierr)
      call MatTranspose(g_ctx%W_uv, MAT_INITIAL_MATRIX, D, ierr)
      call MatAXPY(D, 1.0d0, g_ctx%W_vu, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatNorm(D, NORM_FROBENIUS, nd, ierr)
      call MatDestroy(D, ierr)
      if (my_id == 0) write(*,'(A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3)') &
        "[Physics PC]   SF: v_par W |W_vv|_F ", nvv, ", |W_vu|_F ", nvu, ", |W_uv|_F ", nuv, &
        "; symmetry defect |W_vu + W_uv^T|_F / |W_vu|_F ", nd / max(nvu, 1.d-300)
    end subroutine vpar_w_report

  end subroutine sf_build

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

  !> First-build diagnostics of W: its symmetry defect, every run; -sf_w_check
  !! and -sf_dump (w_symmetry_check, dump_blocks).
  subroutine sf_selfcheck(my_id)
    integer, intent(in) :: my_id
    call w_symmetry_check(my_id)
    call dump_blocks()
  contains
    !> -sf_dump 1: the extracted blocks and W as PETSc binaries sfdump_<op>.petsc
    !! in the run directory, for offline checks of W against the discrete
    !! Schur correction (first build, global path only).
    subroutine dump_blocks()
      PetscInt :: iv
      PetscBool :: set
      PetscErrorCode :: ierr
      iv = 0
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_dump", iv, set, ierr)
      if (iv == 0 .or. famode) return
      call one(g_ctx%B_11, "B_11"); call one(g_ctx%B_12, "B_12"); call one(g_ctx%B_13, "B_13")
      call one(g_ctx%B_16, "B_16"); call one(g_ctx%B_21, "B_21"); call one(g_ctx%B_22, "B_22")
      call one(g_ctx%B_23, "B_23"); call one(g_ctx%B_24, "B_24"); call one(g_ctx%B_25, "B_25")
      call one(g_ctx%B_26, "B_26"); call one(g_ctx%B_31, "B_31"); call one(g_ctx%B_33, "B_33")
      call one(g_ctx%B_42, "B_42"); call one(g_ctx%B_44, "B_44"); call one(g_ctx%B_51, "B_51")
      call one(g_ctx%B_52, "B_52"); call one(g_ctx%B_55, "B_55"); call one(g_ctx%B_61, "B_61")
      call one(g_ctx%B_62, "B_62"); call one(g_ctx%B_63, "B_63"); call one(gB_65, "B_65")
      call one(g_ctx%B_66, "B_66"); call one(g_ctx%W_force, "W")
      if (my_id == 0) write(*,'(A)') "[Physics PC]   SF: blocks dumped to sfdump_*.petsc"
    end subroutine dump_blocks
    subroutine one(A, nm)
      Mat :: A
      character(len=*), intent(in) :: nm
      PetscViewer :: vw
      PetscErrorCode :: ierr
      if (A == PETSC_NULL_MAT) return
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD, "sfdump_"//nm//".petsc", FILE_MODE_WRITE, vw, ierr)
      call MatView(A, vw, ierr)
      call PetscViewerDestroy(vw, ierr)
    end subroutine one
  end subroutine sf_selfcheck

  !> W's symmetry defect, every run (one transpose at the first build). The
  !! bending and the gamma p0 part are symmetric by construction; the kink and
  !! the grad p0 part only together, at force balance -- so on a real
  !! equilibrium the defect measures how far it is from (reduced) force
  !! balance, and an assembly error shows up as a defect at the bending-only
  !! term set (-sf_w_terms 3), which must be at round-off.
  !! -sf_w_check 1 adds the unit tests of the W assembly (model183):
  !!   -sf_w_terms 4: W must equal A's (w, u) block, B_42, to round-off
  !!   -sf_w_terms 7 / 8: |W|_F of int (v_pp u + v_p u_p) against int v_p u_p
  !! (needs physics_pc_sf_harm_couple = .t., so B_42 keeps every harmonic).
  subroutine w_symmetry_check(my_id)
    integer, intent(in) :: my_id
    Mat :: Wt, D
    real*8 :: nw, nd, nb
    PetscInt :: iv
    PetscBool :: set
    PetscErrorCode :: ierr
    call MatNorm(g_ctx%W_force, NORM_FROBENIUS, nw, ierr)
    call MatTranspose(g_ctx%W_force, MAT_INITIAL_MATRIX, Wt, ierr)
    call MatAYPX(Wt, -1.0d0, g_ctx%W_force, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatNorm(Wt, NORM_FROBENIUS, nd, ierr)
    call MatDestroy(Wt, ierr)
    if (my_id == 0) write(*,'(A,I0,A,ES10.3,A,ES10.3)') "[Physics PC]   SF: W (terms ", sf_force_terms(), &
      ") |W|_F ", nw, ", symmetry defect |W - W^T|_F / |W|_F ", nd / max(nw, 1.d-300)
    iv = 0
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_w_check", iv, set, ierr)
    if (iv == 0) return
    select case (sf_force_terms())
    case (4);  call against(g_ctx%B_42, "B_42")
    case (9);  call against(g_ctx%B_12, "B_12")
    case (10); call against(g_ctx%B_23, "B_23")
    case (11); call against(g_ctx%B_31, "B_31")
    end select
    ! 7 / 8: d_phi at fixed (R, Z) integrates to zero only where the
    ! cross-section does not move with phi -- on a stellarator the boundary
    ! does, so restrict to DOFs of non-boundary nodes (v = u = 0 there)
    if (sf_force_terms() == 7 .or. sf_force_terms() == 8) then
      block
        use nodes_elements, only: node_list
        use mod_parameters, only: n_tor, n_degrees
        Vec :: msk
        PetscInt :: r0, r1, row
        integer :: n, dg, m
        call MatCreateVecs(g_ctx%W_force, msk, PETSC_NULL_VEC, ierr)
        call VecSet(msk, 1.0d0, ierr)
        call VecGetOwnershipRange(msk, r0, r1, ierr)
        do n = 1, node_list%n_nodes
          if (node_list%node(n)%boundary == 0) cycle
          do dg = 1, n_degrees
            do m = 1, n_tor
              row = int(node_list%node(n)%index(dg) - 1, kind(row)) * n_tor + (m - 1)
              if (row >= r0 .and. row < r1) call VecSetValue(msk, row, 0.0d0, INSERT_VALUES, ierr)
            enddo
          enddo
        enddo
        call VecAssemblyBegin(msk, ierr); call VecAssemblyEnd(msk, ierr)
        call MatDuplicate(g_ctx%W_force, MAT_COPY_VALUES, D, ierr)
        call MatDiagonalScale(D, msk, msk, ierr)
        call MatNorm(D, NORM_FROBENIUS, nd, ierr)
        call MatDestroy(D, ierr)
        call VecDestroy(msk, ierr)
        if (my_id == 0) write(*,'(A,I0,A,ES10.3)') "[Physics PC]   SF: W unit test ", sf_force_terms(), &
          ": |W|_F on interior DOFs ", nd
      end block
    endif
  contains
    !> W's unit-test term sets reproduce one Jacobian block: compare
    subroutine against(Bk, nm)
      Mat :: Bk
      character(len=*), intent(in) :: nm
      if (Bk == PETSC_NULL_MAT) then
        if (my_id == 0) write(*,'(A)') "[Physics PC]   SF: -sf_w_check needs "//nm//" (global path)"
        return
      endif
      call MatDuplicate(g_ctx%W_force, MAT_COPY_VALUES, D, ierr)
      call MatAXPY(D, -1.0d0, Bk, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatNorm(D, NORM_FROBENIUS, nd, ierr)
      call MatNorm(Bk, NORM_FROBENIUS, nb, ierr)
      call MatDestroy(D, ierr)
      if (my_id == 0) write(*,'(A,I0,A,A,A,A,A,ES10.3,A,ES10.3)') "[Physics PC]   SF: W unit test ", &
        sf_force_terms(), ": |W - ", nm, "|_F / |", nm, "|_F ", nd / max(nb, 1.d-300), ", norm ", nb
    end subroutine against
  end subroutine w_symmetry_check

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
  !! v_par (V_PAR in the module header): "b0" solves it between 2 and 3, "b1"
  !! inside 2 (pair_w's 5th field); both add B_57 V, B_67 V to the rho / T
  !! correctors. "v0" solves it after 3.
  !--------------------------------------------------------------------
  subroutine sf_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr

    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T, x_v
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T, y_v
    Vec :: parts(5)
    logical :: vb0, vb1, vcor
    logical, parameter :: keep_uw(5) = [.true., .true., .false., .false., .false.]

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
    vb0 = .false.; vb1 = .false.
    x_v = zv; y_v = zv                                   ! placeholders without v_par (never read)
    if (with_v) then
      x_v = sv_x(NSV); y_v = sv_y(NSV)                   ! NSV = var_Vpar = 7 (sf_init)
      vb0 = (vstage == SF_VPAR_B0)
      vb1 = (vstage == SF_VPAR_B1)
      if (vb0 .and. .not. vcheck_done) call vpar_w_check(x_v, x_u)
    endif
    vcor = vb0 .or. vb1                                  ! v_par in the rho / T correctors

    ! EXPERIMENT (-sf_seq): the core-first sweeps replace steps 1-3
    sweep: block
    if (seq /= 0) then
      call sf_core_sweep(x_psi, x_u, x_j, x_w, x_rho, x_T, y_psi, y_u, y_j, y_w, y_rho, y_T)
      exit sweep
    endif

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
    call MatMult(a_51, y_psi, w3, ierr)
    call VecWAXPY(w4, -1.0d0, w3, x_rho, ierr)
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call sf_solver_apply(slv_rho, w4, t_rho, ierr)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)

    !--- Step 1: predictor temperature  T* = B_66^-1 (x_T - B_61 psi* - B_63 j*)
    ! B_61 and B_63 act against the EXPLICIT predictor pair. No j-folded
    ! lower-triangular block is needed or wanted here.
    call MatMult(a_61, y_psi, w3, ierr)
    call VecWAXPY(w4, -1.0d0, w3, x_T, ierr)
    call MatMult(a_63, y_j, w3, ierr)
    call VecAXPY(w4, -1.0d0, w3, ierr)
    if (use_b65) then                                   ! - B_65 rho*
      call MatMult(a_65, t_rho, w3, ierr)
      call VecAXPY(w4, -1.0d0, w3, ierr)
    endif
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
    call MatMult(a_21, y_psi, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call MatMult(a_23, y_j, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call MatMult(a_25, t_rho, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    call MatMult(a_26, t_T, w3, ierr)
    call VecAXPY(w5, -1.0d0, w3, ierr)
    ! mixed: the psi / j rows' right-hand side is zero (the predictor has
    ! consumed x_psi, x_j). With the Eq. (16) corrector their solution is
    ! discarded -- step 3 recomputes psi and j by a pair_psi solve. With
    ! Eq. (17) (wpj) it IS the correction: rows 3-4 of pair_w read
    ! M~ (psi_w, j_w) = -(B_12 u + b, 0), M~ the small-flow pair_psi, so
    ! (psi_w, j_w) = -M~^-1 U (u, T*) with b = B_16 T* (corr 1) or 0 (corr 2).
    ! b1: v_par is pair_w's 5th field, its right-hand side
    ! x_V - B_71 psi* - B_75 rho* - B_76 T* (psi*, not psi_w: see V_PAR)
    if (vb1) then
      call MatMult(a_vc(VC71), y_psi, wv1, ierr)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      call MatMult(a_vc(VC75), t_rho, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
      call MatMult(a_vc(VC76), t_T, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
    endif
    if (mixed) then
      if (corr == 1 .or. corr == 4) then
        call MatMult(a_16, t_T, w4, ierr)
        call VecScale(w4, -1.0d0, ierr)
        parts = [w5, x_w, w4, zv, wv2]
      else
        parts = [w5, x_w, zv, zv, wv2]
      endif
      call sf_split_parts(rhs_W, parts(1:nf_w), .true.)
    else
      call sf_split_halves(rhs_W, w5, x_w, .true.)
    endif
    call PetscLogEventBegin(pcev_solve_w, ierr)
    call sf_solver_apply(slv_w, rhs_W, sol_W, ierr)
    call PetscLogEventEnd(pcev_solve_w, ierr)
    if (corr /= 0) then
      parts = [y_u, y_w, w3, w4, y_v]                     ! w3, w4 = psi_w, j_w; b1: y_v = V
      call sf_split_parts(sol_W, parts(1:nf_w), .false.)
    else if (mixed) then
      parts = [y_u, y_w, zv, zv, zv]
      call sf_split_parts(sol_W, parts(1:nf_w), .false., keep=keep_uw(1:nf_w))
    else
      call sf_split_halves(sol_W, y_u, y_w, .false.)      ! BOTH final; y_w is DONE
    endif

    !--- v_par, "b0": the sequential velocity stage, right after pair_w ----
    !   (B_77 + W_vv) V = x_V - B_71 psi* - B_75 rho* - B_76 T* - (B_72 + W_vu) u
    ! y_psi is still psi* here (step 3 adds psi_w): W_vu leaves out the p
    ! channel's part that cancels against the psi channel, so psi_w does not
    ! enter (mod_pc_elt_matrix_force_fft, V_PAR) -- unless -sf_vpar_wvu_full
    ! keeps that part in W_vu, and then psi_w enters explicitly.
    if (vb0) then
      call MatMult(a_vc(VC71), y_psi, wv1, ierr)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      if (vfull) then                                   ! - B_71 psi_w (w3, corr /= 0)
        call MatMult(a_vc(VC71), w3, wv1, ierr)
        call VecAXPY(wv2, -1.0d0, wv1, ierr)
      endif
      call MatMult(a_vc(VC75), t_rho, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
      call MatMult(a_vc(VC76), t_T, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
      call MatMult(a_vc(VC72), y_u, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call sf_solver_apply(slv_v, wv2, y_v, ierr)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
    endif

    !--- corrector 4, the consistent one (Rac, GMG for the SF preconditioner,
    !--- Alg. 1): the rho / T corrections below also carry pair_w's change of
    !--- (psi, j), B_51 psi_w and B_61 psi_w + B_63 j_w, which corrector 1
    !--- drops. Kept in y_rho / y_T (free until the correctors) while w3, w4
    !--- still hold psi_w, j_w. The same block solves as corrector 1.
    if (corr == 4) then
      call MatMult(a_51, w3, y_rho, ierr)
      call MatMult(a_61, w3, y_T, ierr)
      call MatMult(a_63, w4, w5, ierr)
      call VecAXPY(y_T, 1.0d0, w5, ierr)
    endif

    !--- Step 3: corrector psi-pair --------------------------------------
    if (corr /= 0) then
      ! Eq. (17): psi = psi* + psi_w, j = j* + j_w, no solve
      call VecAXPY(y_psi, 1.0d0, w3, ierr)
      call VecAXPY(y_j,   1.0d0, w4, ierr)
    else
      ! Here the j-component of the RHS IS zero, because the j-row of the upper
      ! coupling U is identically zero. (Contrast the predictor above.)
      call MatMult(a_12, y_u, w3, ierr)
      call MatMult(a_16, t_T, w4, ierr)
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
    if (corr == 3) then
      ! Eq. (17) strict: M^-1 ~ the time derivative's, (opz B_44)^-1; no B_65
      ! (zero in model199), no block solve
      call MatMult(a_52, y_u, w3, ierr)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_rho, -1.0d0 / sf_opz(), w5, t_rho, ierr)
      call MatMult(a_62, y_u, w3, ierr)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_T, -1.0d0 / sf_opz(), w5, t_T, ierr)
    else
      ! "b0" / "b1": the upper couplings of v_par too, (B_52 u + B_57 V, B_62 u + B_67 V)
      call MatMult(a_52, y_u, w3, ierr)
      if (corr == 4) call VecAXPY(w3, 1.0d0, y_rho, ierr)  ! + B_51 psi_w
      if (vcor) then
        call MatMult(a_vc(VC57), y_v, w4, ierr)
        call VecAXPY(w3, 1.0d0, w4, ierr)
      endif
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call sf_solver_apply(slv_rho, w3, w5, ierr)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
      call VecWAXPY(y_rho, -1.0d0, w5, t_rho, ierr)

      ! [[B_55, 0], [B_65, B_66]] (drho, dT) = (B_52 u, B_62 u): with B_65 the
      ! T correction sees the rho correction just computed (w5)
      call MatMult(a_62, y_u, w3, ierr)
      if (corr == 4) call VecAXPY(w3, 1.0d0, y_T, ierr)    ! + B_61 psi_w + B_63 j_w
      if (vcor) then
        call MatMult(a_vc(VC67), y_v, w4, ierr)
        call VecAXPY(w3, 1.0d0, w4, ierr)
      endif
      if (use_b65) then
        call MatMult(a_65, w5, w4, ierr)
        call VecAXPY(w3, -1.0d0, w4, ierr)
      endif
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call sf_solver_apply(slv_T, w3, w5, ierr)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
      call VecWAXPY(y_T, -1.0d0, w5, t_T, ierr)
    endif

    !--- v_par, "v0": last, against the final psi, u, rho, T ---------------
    !   B_77 V = x_V - B_71 psi - B_72 u - B_75 rho - B_76 T
    if (with_v .and. .not. vcor) then
      call MatMult(a_vc(VC71), y_psi, wv1, ierr)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      call MatMult(a_vc(VC72), y_u, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
      call MatMult(a_vc(VC75), y_rho, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
      call MatMult(a_vc(VC76), y_T, wv1, ierr)
      call VecAXPY(wv2, -1.0d0, wv1, ierr)
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call sf_solver_apply(slv_v, wv2, y_v, ierr)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
    endif
    end block sweep

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

  !--------------------------------------------------------------------
  !> EXPERIMENT (-sf_seq 1 | 2): pair_w (wpj) taken as the whole core
  !! (u, omega, psi, j), with x_psi and x_j on its psi / j rows, so that its
  !! (psi, j) are the final ones and no pair_psi solve is needed. pair_w's
  !! psi row is the small-flow one (opz B_33 for B_11); with exact solves this
  !! is the predictor / corrector of steps 1-3 with that pair_psi. What is
  !! left is the split core | (rho, T):
  !!
  !!   1 (block Gauss--Seidel): pair_w (u, w, psi, j) = (x_u, x_w, x_psi, x_j),
  !!     then rho = B_55^-1 (x_rho - B_51 psi - B_52 u),
  !!          T   = B_66^-1 (x_T - B_61 psi - B_63 j - B_62 u - B_65 rho).
  !!     Drops B_25, B_26, B_16 (rho / T into the core).
  !!   2 (block LDU): rho* = B_55^-1 x_rho, T* = B_66^-1 (x_T - B_65 rho*),
  !!     pair_w on (x_u - B_25 rho* - B_26 T*, x_w, x_psi - B_16 T*, x_j),
  !!     then rho / T corrected by B_55^-1, B_66^-1 of (B_51 psi + B_52 u,
  !!     B_61 psi + B_63 j + B_62 u - B_65 drho). Drops the transport Schur
  !!     terms (the pressure channels) from the core.
  !--------------------------------------------------------------------
  subroutine sf_core_sweep(x_psi, x_u, x_j, x_w, x_rho, x_T, y_psi, y_u, y_j, y_w, y_rho, y_T)
    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T
    Vec :: parts(4)
    PetscErrorCode :: ierr

    !--- seq 2: transport predictor, without the core
    if (seq == 2) then
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call sf_solver_apply(slv_rho, x_rho, t_rho, ierr)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
      call VecCopy(x_T, w4, ierr)
      if (use_b65) then
        call MatMult(a_65, t_rho, w3, ierr)
        call VecAXPY(w4, -1.0d0, w3, ierr)
      endif
      call PetscLogEventBegin(pcev_solve_rhot, ierr)
      call sf_solver_apply(slv_T, w4, t_T, ierr)
      call PetscLogEventEnd(pcev_solve_rhot, ierr)
    endif

    !--- the core: pair_w with the psi / j rows' own right-hand sides
    call VecCopy(x_u, w5, ierr)
    call VecCopy(x_psi, w4, ierr)
    if (seq == 2) then
      call MatMult(a_25, t_rho, w3, ierr)
      call VecAXPY(w5, -1.0d0, w3, ierr)
      call MatMult(a_26, t_T, w3, ierr)
      call VecAXPY(w5, -1.0d0, w3, ierr)
      call MatMult(a_16, t_T, w3, ierr)
      call VecAXPY(w4, -1.0d0, w3, ierr)
    endif
    parts = [w5, x_w, w4, x_j]
    call sf_split_parts(rhs_W, parts, .true.)
    call PetscLogEventBegin(pcev_solve_w, ierr)
    call sf_solver_apply(slv_w, rhs_W, sol_W, ierr)
    call PetscLogEventEnd(pcev_solve_w, ierr)
    parts = [y_u, y_w, y_psi, y_j]
    call sf_split_parts(sol_W, parts, .false.)

    !--- transport: B_55 drho = B_51 psi + B_52 u (seq 2: the correction;
    !--- seq 1: rho = B_55^-1 (x_rho - that))
    call MatMult(a_51, y_psi, w3, ierr)
    call MatMult(a_52, y_u, w4, ierr)
    call VecAXPY(w3, 1.0d0, w4, ierr)
    if (seq == 1) call VecAYPX(w3, -1.0d0, x_rho, ierr)     ! x_rho - (B_51 psi + B_52 u)
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call sf_solver_apply(slv_rho, w3, w5, ierr)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)
    if (seq == 1) then
      call VecCopy(w5, y_rho, ierr)
    else
      call VecWAXPY(y_rho, -1.0d0, w5, t_rho, ierr)
    endif

    ! B_66 dT = B_61 psi + B_63 j + B_62 u - B_65 drho
    call MatMult(a_61, y_psi, w3, ierr)
    call MatMult(a_63, y_j, w4, ierr)
    call VecAXPY(w3, 1.0d0, w4, ierr)
    call MatMult(a_62, y_u, w4, ierr)
    call VecAXPY(w3, 1.0d0, w4, ierr)
    if (seq == 1) then
      call VecAYPX(w3, -1.0d0, x_T, ierr)                   ! x_T - (...)
      if (use_b65) then
        call MatMult(a_65, y_rho, w4, ierr)
        call VecAXPY(w3, -1.0d0, w4, ierr)
      endif
    else if (use_b65) then
      call MatMult(a_65, w5, w4, ierr)
      call VecAXPY(w3, -1.0d0, w4, ierr)
    endif
    call PetscLogEventBegin(pcev_solve_rhot, ierr)
    call sf_solver_apply(slv_T, w3, w5, ierr)
    call PetscLogEventEnd(pcev_solve_rhot, ierr)
    if (seq == 1) then
      call VecCopy(w5, y_T, ierr)
    else
      call VecWAXPY(y_T, -1.0d0, w5, t_T, ierr)
    endif
  end subroutine sf_core_sweep

  !> -sf_vpar_wcheck 1 ("b0", global path, first apply): v_par's W terms
  !! against the discrete Schur terms they stand for, on x (the first v_par
  !! right-hand side, a smooth field):
  !!   W_vv x  vs  -(B_75 drho + B_76 dT)
  !!   W_uv x  vs  -(B_25 drho + B_26 dT),   [[B_55, 0], [B_65, B_66]] (drho, dT) = (B_57 x, B_67 x)
  !! with the rho / T solvers of the run (exact on lu). They differ by the
  !! projection defect and by what W leaves out of B_55 / B_66 (diffusion,
  !! parallel conduction), both small at small dt: there a sign or factor
  !! error of the element terms shows as cos far from 1 or a ratio off 1.
  subroutine vpar_w_check(x, xu)
    Vec :: x, xu
    PetscInt :: iv
    PetscBool :: set
    PetscErrorCode :: ierr
    integer :: my_id, mpierr
    vcheck_done = .true.
    iv = 0
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_vpar_wcheck", iv, set, ierr)
    if (iv == 0) return
    call MPI_Comm_rank(comm_g, my_id, mpierr)
    if (famode) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   SF: -sf_vpar_wcheck needs the global path (mode_split .f.)"
      return
    endif
    ! (drho, dT) from (B_57 x, B_67 x): drho in wv1, dT in wv2
    call MatMult(a_vc(VC57), x, w3, ierr)
    call sf_solver_apply(slv_rho, w3, wv1, ierr)                         ! drho
    call MatMult(a_vc(VC67), x, w3, ierr)
    if (use_b65) then
      call MatMult(a_65, wv1, w4, ierr)
      call VecAXPY(w3, -1.0d0, w4, ierr)
    endif
    call sf_solver_apply(slv_T, w3, wv2, ierr)                           ! dT
    ! v_par row: W_vv x against -(B_75 drho + B_76 dT)
    call MatMult(a_vc(VC75), wv1, w3, ierr)
    call MatMult(a_vc(VC76), wv2, w4, ierr)
    call VecAXPY(w3, 1.0d0, w4, ierr)
    call VecScale(w3, -1.0d0, ierr)
    call MatMult(g_ctx%W_vv, x, w5, ierr)
    call compare("W_vv x vs -(B_75 drho + B_76 dT)", w5, w3)
    ! u row: W_uv x against -(B_25 drho + B_26 dT)
    call MatMult(a_25, wv1, w3, ierr)
    call MatMult(a_26, wv2, w4, ierr)
    call VecAXPY(w3, 1.0d0, w4, ierr)
    call VecScale(w3, -1.0d0, ierr)
    call MatMult(g_ctx%W_uv, x, w5, ierr)
    call compare("W_uv x vs -(B_25 drho + B_26 dT)", w5, w3)
    ! -sf_vpar_wvu_full: v_par row, u column, on xu (the first u right-hand
    ! side): W_vu xu against -(B_75 drho + B_76 dT), (drho, dT) from (B_52 xu, B_62 xu)
    if (vfull) then
      call MatMult(a_52, xu, w3, ierr)
      call sf_solver_apply(slv_rho, w3, wv1, ierr)
      call MatMult(a_62, xu, w3, ierr)
      if (use_b65) then
        call MatMult(a_65, wv1, w4, ierr)
        call VecAXPY(w3, -1.0d0, w4, ierr)
      endif
      call sf_solver_apply(slv_T, w3, wv2, ierr)
      call MatMult(a_vc(VC75), wv1, w3, ierr)
      call MatMult(a_vc(VC76), wv2, w4, ierr)
      call VecAXPY(w3, 1.0d0, w4, ierr)
      call VecScale(w3, -1.0d0, ierr)
      call MatMult(g_ctx%W_vu, xu, w5, ierr)
      call compare("W_vu u vs -(B_75 drho + B_76 dT)", w5, w3)
    endif
  contains
    subroutine compare(label, a, b)
      character(len=*), intent(in) :: label
      Vec :: a, b
      real*8 :: na, nb, ab, nd
      call VecNorm(a, NORM_2, na, ierr)
      call VecNorm(b, NORM_2, nb, ierr)
      call VecDot(a, b, ab, ierr)
      call VecAXPY(a, -1.0d0, b, ierr)
      call VecNorm(a, NORM_2, nd, ierr)
      if (my_id == 0) write(*,'(A,A,A,ES10.3,A,ES10.3,A,F8.5,A,ES10.3)') "[Physics PC]   SF: v_par W check ", &
        label, ": |W|/|S| ", na / max(nb, 1.d-300), ", |W - S|/|S| ", nd / max(nb, 1.d-300), &
        ", cos ", ab / max(na * nb, 1.d-300), ", |S| ", nb
    end subroutine compare
  end subroutine vpar_w_check

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
      if (with_v) call sf_solver_report(slv_v, my_id)
    endif
    call sf_solver_reset_counters(slv_pj)
    call sf_solver_reset_counters(slv_w)
    call sf_solver_reset_counters(slv_rho)
    call sf_solver_reset_counters(slv_T)
    if (with_v) call sf_solver_reset_counters(slv_v)
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
    integer, parameter :: NV = 14
    real*8 :: v(NV), vf(NV)
    real*8, allocatable :: g(:, :)
    integer :: np, r, f, nrk, mpierr
    logical :: done
    v(1:4)  = [its_mean(slv_pj), its_mean(slv_w), its_mean(slv_rho), its_mean(slv_T)]
    v(5:8)  = [slv_pj%t_sum, slv_w%t_sum, slv_rho%t_sum, slv_T%t_sum]
    v(13:14) = 0.d0
    if (with_v) v(13:14) = [its_mean(slv_v), slv_v%t_sum]
    v(9)    = t_apply - t_scat - sum(v(5:8)) - v(14)
    v(10)   = t_scat
    v(11)   = dble(slv_pj%nfail + slv_w%nfail + slv_rho%nfail + slv_T%nfail + slv_v%nfail)
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
        if (with_v) write(*,'(A,F5.2,A,F7.3)', advance="no") ", v_par ", g(13, r), " /", g(14, r)
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
