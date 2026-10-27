module mod_petsc_pc_sf
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var, var_Vpar
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, &
       pcev_extract, pcev_convert, pcev_build_suu, pcev_fact_pj, pcev_fact_w, &
       pcev_fact_rhot, pcev_solve_pj, pcev_solve_w, pcev_solve_rhot, pcev_apply
  use mod_petsc_pc_blocks, only: extract_sub_blocks_h, pack_pair_aij, make_field_block_scale, &
       decouple_penalty_rows, split_vars, merge_vars, harm_all
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_gather, only: sfg_build, sfg_gather, sfg_drop, sfg_dev_setup
  use mod_petsc_pc_sf_mixed, only: sfm_build, sfm_refill, sfm_op, SFM_NF, sfm_map_spec, sfm_release_parts
  use mod_petsc_blk_dev, only: vec_make_dev
  use mod_petsc_raw_csr, only: blockmv_attach
  use mod_petsc_pc_mass_cheb, only: mass_cheb_t, mass_cheb_setup, mass_cheb_solve
  use mod_petsc_pc_sf_fam, only: fam_op_t, sff_comm, sff_decompose, sff_op_build, sff_op_dev, &
       sff_op_fill, sff_vec_setup, sff_scatter_in, sff_scatter_out
  implicit none
  private

  !--------------------------------------------------------------------
  !> The physics-based preconditioner (use_physics_pc): the split-field (SF)
  !! block-LDU sweep over (psi, u, j, omega, rho, T), plus v_par (var 7) as a
  !! seventh block where the model has it.
  !!
  !! The two mixed pairs keep j and omega explicit:
  !!   pair_psi = [[B_11, B_13], [B_31, B_33]]                  (psi, j)
  !!   pair_w   = [[B_22 + W, B_24, B_21,      B_23],           (u, omega, psi, j)
  !!               [B_42,     B_44, 0,         0   ],
  !!               [B_12,     0,    opz M_psi, B_13],
  !!               [0,        0,    B_31,      B_33]]
  !! with W the curvature part of the composed force operator (Chacon JCP 526
  !! (2025) Eqs. 17-19, mod_pc_elt_matrix_force_fft) and the psi channel
  !! carried by the explicit (psi, j) rows (mod_petsc_pc_sf_mixed).
  !!
  !! physics_pc_sf_harm_couple .t. (default) keeps every cross-|n| entry and
  !! puts all slots of a node into one GMG smoother block, on the global
  !! operators. .f. makes every block |n|-diagonal; with
  !! physics_pc_sf_mode_split each |n| family is then solved on its own rank
  !! group (mod_petsc_pc_sf_fam).
  !!
  !! v_par (physics_pc_sf_vpar_stage):
  !!   v0: B_77 V = x_V - B_71 psi - B_72 u - B_75 rho - B_76 T, last.
  !!   b0: after pair_w, (B_77 + W_vv) V = x_V - B_71 psi* - B_75 rho* - B_76 T*
  !!       - (B_72 + W_vu) u, then B_57 V, B_67 V in the rho / T correctors.
  !!   b1: v_par as pair_w's 5th field (mode split, Eq. (17) corrector only).
  !!   -sf_vpar_w selects the W terms (bit 0 W_vv, 1 W_vu, 2 W_uv), and
  !!   -sf_vpar_gamma_w their gamma.
  !--------------------------------------------------------------------

  logical, save :: sf_init_done = .false.
  logical, save :: sf_first     = .true.

  type(block_solver_t), save :: slv_pj, slv_w, slv_rho, slv_T
  integer, save :: bk_pj = SF_GMG, bk_w = SF_GMG, bk_rho = SF_GMG, bk_T = SF_GMG
  integer, save :: corr = 4                    !< step-3 corrector (physics_pc_sf_corrector)

  !--- v_par
  logical, parameter :: with_v = (var_Vpar > 0)
  integer, parameter :: NSV = merge(7, 6, with_v)     !< the variables the sweep splits
  type(block_solver_t), save :: slv_v
  integer, save :: bk_v   = SF_GMG
  integer, save :: vstage = SF_VPAR_V0
  integer, save :: vwmask = 0                   !< bit 0 W_vv on B_77, 1 W_vu on B_72, 2 W_uv on B_27
  logical, save :: vfull = .false.              !< -sf_vpar_wvu_full: whole-p-channel W_vu, psi_w in v_par's row
  Mat, save :: a_v                              !< B_77 (+ W_vv)
  !> v_par's coupling blocks a_vc(VC_ij) = B_ij (VC72: B_72 + W_vu with b0)
  integer, parameter :: VC71 = 1, VC72 = 2, VC75 = 3, VC76 = 4, VC57 = 5, VC67 = 6
  Mat, save :: a_vc(6)
  !> the W sources of the mode split's operators (fam_op_t wid)
  integer, parameter :: WS_C = 1, WS_VV = 2, WS_VU = 3, WS_UV = 4, WS_N = 4
  Mat, save :: gB_71 = PETSC_NULL_MAT, gB_72 = PETSC_NULL_MAT, gB_75 = PETSC_NULL_MAT, &
               gB_76 = PETSC_NULL_MAT, gB_77 = PETSC_NULL_MAT, gB_57 = PETSC_NULL_MAT, &
               gB_67 = PETSC_NULL_MAT
  !> b0 sums on the global path (not PETSC_NULL_MAT-initialised: PETSc's
  !! Fortran stubs read a null output argument as "not wanted")
  Mat, save :: vop_g, c72_g
  logical, save :: vop_on = .false., c72_on = .false.
  Vec, save :: wv1, wv2

  !--- work vectors
  Vec, save :: sv_x(NSV), sv_y(NSV)
  Vec, save :: rhs_PJ, sol_PJ, rhs_W, sol_W
  Vec, save :: w3, w4, w5, t_rho, t_T
  Vec, save :: zv                      !< zero field
  logical, save :: vecs_ready = .false., kpj_packed = .false.
  logical, save :: sfm_gathered = .false.      !< pair_w refilled by the value map (no sfm_refill)
  ! The SF apply on the device (-sf_gpu 1 with every block solver on the GMG
  ! device path; -sf_gpu_apply 0 keeps it on the host): its work vectors are
  ! device vectors and cmult multiplies with device copies ad(k) of the
  ! couplings, refilled lazily once per build (ad_gen(k) /= sf_gen).
  logical, save :: sfdev = .false.
  integer, save :: sf_gen = 0
  integer, parameter :: NAD = 18
  Mat, save     :: ad(NAD)
  integer, save :: ad_gen(NAD) = -1

  !--- the operators the sweep uses: the global ones, or under the mode split
  !--- this rank's family's (no global operator exists then)
  logical, save :: famode = .false.
  integer, save :: comm_s = MPI_COMM_NULL      !< the operators' communicator
  integer, save :: comm_g = MPI_COMM_NULL      !< the full system's
  real*8, save  :: t_apply = 0.d0, t_scat = 0.d0   !< mode split: apply / scatter time since the report
  integer, save :: nf_w = SFM_NF               !< pair_w's packed fields
  Mat, save :: a_pj, a_w, a_rho, a_T
  !> The sweep's coupling blocks. B_65 (T row, rho column) is zero in model199
  !! (temperature equation) and is applied only where it is not (use_b65).
  Mat, save :: a_12, a_16, a_21, a_23, a_25, a_26, a_51, a_52, a_61, a_62, a_63, a_65
  !> mode split coupling blocks: the 12 above, then 13 B_71, 14 B_72, 15 B_75,
  !! 16 B_76, 17 B_57, 18 B_67; ncb_on of them in use
  integer, parameter :: NCB = 12 + merge(6, 0, with_v)
  integer, save :: ncb_on = 12
  integer, parameter :: NDIAG = 5              !< message tags of the diagonal operators
  logical, save :: use_b65 = .false.
  Mat, save :: gB_65 = PETSC_NULL_MAT
  type(fam_op_t), save :: o_pj, o_w, o_rho, o_T, o_v, o_cb(NCB)
  !> corr 3: the rho / T mass, opz B_44 in model199, inverted by Chebyshev
  type(fam_op_t), save :: o_m
  Mat, save :: a_m
  type(mass_cheb_t), save :: mcm

  !--- penalty rows of the full Jacobian (eliminate_boundary_dofs = .f.):
  !--- local offsets and diagonals, refreshed on every build
  Mat, save :: a_full_s
  integer, save :: npen_full = 0             !< global count; 0 = no back-substitution
  integer, allocatable, save :: pen_loc(:)
  real*8, allocatable, save  :: pen_d(:)
  Vec, save :: pen_w
  logical, save :: pen_w_made = .false.

  public :: sf_build, sf_apply, sf_report

contains

  !> Resolve the namelist once, and refuse configurations the path cannot run.
  subroutine sf_init(my_id)
    use phys_module, only: physics_pc_sf_pair_psi, physics_pc_sf_pair_w, &
                           physics_pc_sf_rho, physics_pc_sf_T, physics_pc_sf_rtol, &
                           physics_pc_sf_harm_couple, physics_pc_sf_corrector, &
                           physics_pc_sf_mode_split, physics_pc_sf_vpar, physics_pc_sf_vpar_stage
    use mod_parameters, only: n_tor, var_T
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr
    character(len=*), parameter :: P = "[Physics PC]   "

    if (sf_init_done) return

    bk_pj  = backend_of(physics_pc_sf_pair_psi, "physics_pc_sf_pair_psi")
    bk_w   = backend_of(physics_pc_sf_pair_w,   "physics_pc_sf_pair_w")
    bk_rho = backend_of(physics_pc_sf_rho,      "physics_pc_sf_rho")
    bk_T   = backend_of(physics_pc_sf_T,        "physics_pc_sf_T")

    ! the variables (psi, u, zj, w, rho, T) in model199's slots, v_par as var 7;
    ! n_tor odd: n = 0 plus cos/sin pairs
    if (jorek_model /= 199 .and. jorek_model /= 600 .and. jorek_model /= 183) &
      call fatal("physics_pc_sf supports model199, model600 and model183 only")
    if (n_var /= 6 .and. .not. ((jorek_model == 600 .or. jorek_model == 183) .and. n_var == 7 .and. &
                                var_Vpar == 7 .and. var_T == 6)) &
      call fatal("physics_pc_sf supports (psi, u, zj, w, rho, T), plus v_par on model600 / model183: build "// &
                 "with with_TiTe, with_neutrals and with_impurities off")
    if (mod(n_tor, 2) /= 1) call fatal("physics_pc_sf needs n_tor odd (n = 0 plus cos/sin pairs)")

    harm_all    = physics_pc_sf_harm_couple
    sf_gmg_harm = merge(SF_GMG_HARM_ALL, SF_GMG_HARM_PAIR, harm_all)
    famode      = physics_pc_sf_mode_split .and. .not. harm_all

    corr = physics_pc_sf_corrector
    if (corr == -1) corr = 4
    if (corr < 0 .or. corr > 4) call fatal("physics_pc_sf_corrector must be -1 (= 4), 0, 1, 2, 3 or 4")
    ! corr 3 takes the rho / T mass as opz B_44, which holds in model199 only
    if (corr == 3 .and. jorek_model /= 199) &
      call fatal("physics_pc_sf_corrector = 3 supports model199 only (rho / T mass = opz B_44)")

    if (with_v) then
      bk_v   = backend_of(physics_pc_sf_vpar, "physics_pc_sf_vpar")
      vstage = sf_vpar_stage()
      if (vstage < 0) call fatal("physics_pc_sf_vpar_stage must be v0 | b0 | b1, got '"// &
                                 trim(physics_pc_sf_vpar_stage)//"'")
      ! W_vv, W_vu, W_uv are assembled by model600's element routine only
      if (vstage /= SF_VPAR_V0 .and. jorek_model /= 600) &
        call fatal("physics_pc_sf_vpar_stage = b0 | b1 needs model600 (W_vv): run v0")
      ncb_on = 16
      if (vstage /= SF_VPAR_V0) then
        ncb_on = 18
        vwmask = sf_vpar_wmask()
        if (vwmask < 0 .or. vwmask > 7) call fatal("-sf_vpar_w must be 0 .. 7 (bit 0 W_vv, bit 1 W_vu, "// &
                                                   "bit 2 W_uv)")
        vfull = sf_vpar_wvu_full()
      endif
      if (vfull .and. corr == 0) &
        call fatal("-sf_vpar_wvu_full needs the Eq. (17) corrector (pair_w's psi_w)")
      if (vstage == SF_VPAR_B1) then
        if (.not. famode) call fatal("physics_pc_sf_vpar_stage = b1 needs the mode split "// &
                                     "(physics_pc_sf_harm_couple = .f., physics_pc_sf_mode_split = .t.)")
        if (corr /= 1 .and. corr /= 2) &
          call fatal("physics_pc_sf_vpar_stage = b1 needs the Eq. (17) corrector (physics_pc_sf_corrector 1 | 2)")
      endif
    endif

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ================ split-field (SF) preconditioner ================"
      write(*,'(A,A)') P//"pair_psi : ", trim(physics_pc_sf_pair_psi)
      write(*,'(A,A)') P//"pair_w   : ", trim(physics_pc_sf_pair_w)//" (u, omega, psi, j; W = curvature)"
      write(*,'(A,A)') P//"rho / T  : ", trim(physics_pc_sf_rho)//" / "//trim(physics_pc_sf_T)
      write(*,'(A,ES9.2)') P//"inner rtol: ", physics_pc_sf_rtol
      if (harm_all) then
        write(*,'(A)') P//"cross-|n|: all kept, all slots per GMG smoother block"
      else
        write(*,'(A)') P//"cross-|n|: none, |n|-diagonal blocks, cos / sin per GMG smoother block"
      endif
      if (famode) then
        write(*,'(A)') P//"layout   : one |n| family per rank group (physics_pc_sf_mode_split)"
      else
        write(*,'(A)') P//"layout   : global operators"
      endif
      if (with_v) then
        select case (vstage)
        case (SF_VPAR_V0)
          write(*,'(A)') P//"v_par    : "//trim(physics_pc_sf_vpar)//", v0 (B_77 last)"
        case default
          write(*,'(A,A,A,3L2,A,F6.3,A,L1)') P//"v_par    : ", trim(physics_pc_sf_vpar), &
            merge(", b0 (after pair_w)", ", b1 (in pair_w)   ", vstage == SF_VPAR_B0), &
            btest(vwmask, 0), btest(vwmask, 1), btest(vwmask, 2), &
            " (W_vv W_vu W_uv), gamma_w ", sf_vpar_gamma_w(), ", wvu_full ", vfull
        end select
      endif
      select case (corr)
      case (0); write(*,'(A)') P//"corrector: 0, Eq. (16), second pair_psi solve"
      case (1); write(*,'(A)') P//"corrector: 1, Eq. (17), (psi, j) from pair_w, B_16 T* on its psi row"
      case (2); write(*,'(A)') P//"corrector: 2, Eq. (17), (psi, j) from pair_w, B_16 T* dropped"
      case (3); write(*,'(A)') P//"corrector: 3, Eq. (17) strict, rho / T by their mass (opz B_44)^-1"
      case (4); write(*,'(A)') P//"corrector: 4, consistent (Alg. 1), B_51 psi_w, B_61 psi_w + B_63 j_w"
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
      if (my_id == 0) write(*,'(A,A)') P//"FATAL: ", trim(msg)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine sf_init

  !--------------------------------------------------------------------
  !> Build the preconditioner from the assembled Jacobian A_full, on every
  !! PC rebuild. Patterns are frozen after the first build, so later builds
  !! refill values in place and MUMPS / the GMG keep their symbolic phases.
  !--------------------------------------------------------------------
  subroutine sf_build(A_full, comm, my_id)
    use phys_module,    only: physics_pc_sf_rtol
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T

    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id

    PetscErrorCode :: ierr
    PetscInt :: n1_loc
    logical  :: first
    real*8   :: rtol

    call sf_init(my_id)
    first  = sf_first
    sf_gen = sf_gen + 1
    comm_g = comm
    rtol   = physics_pc_sf_rtol

    if (famode) then
      call family_operators(first)
      comm_s = sff_comm
    else
      if (first) then
        call first_build_operators()
      else
        call PetscLogEventBegin(pcev_extract, ierr)
        call sfg_gather(A_full, g_ctx%W_force, my_id, sf_opz())
        if (with_v) call vpar_global_sums(.false.)
        call PetscLogEventEnd(pcev_extract, ierr)
        if (.not. sfm_gathered) then
          call PetscLogEventBegin(pcev_build_suu, ierr)
          call sfm_refill()
          call PetscLogEventEnd(pcev_build_suu, ierr)
        endif
      endif
      a_pj = g_ctx%K_pj_aij;  a_w = sfm_op;  a_rho = g_ctx%B_55;  a_T = g_ctx%B_66
      if (corr == 3) a_m = g_ctx%B_44
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
    call physics_pc_mem("SF build: operators filled", my_id)

    call MatGetLocalSize(a_rho, n1_loc, PETSC_NULL_INTEGER, ierr)

    !--- penalty BC rows (eliminate_boundary_dofs = .f.) as Dirichlet rows
    !--- of the diagonal blocks, before their scaling; -sf_penalty_rows 0: off
    if (penalty_rows_on()) then
      call full_penalty_rows(A_full)
      call penalty_rows(slv_pj, a_pj, 2, "pair_psi")
      call penalty_rows(slv_w, a_w, nf_w, "pair_w")
      call penalty_rows(slv_rho, a_rho, 1, "rho-block")
      call penalty_rows(slv_T, a_T, 1, "T-block")
      if (with_v .and. vstage /= SF_VPAR_B1) call penalty_rows(slv_v, a_v, 1, "v_par-block")
    endif

    !--- symmetric block scaling of both pairs: an exact similarity on the
    !--- stored operator, undone in sf_solver_apply
    call block_scale(slv_pj, a_pj, 2, "pair_psi")
    call block_scale(slv_w, a_w, nf_w, "pair_w")

    call PetscLogEventBegin(pcev_fact_pj, ierr)
    call sf_solver_setup(slv_pj, a_pj, bk_pj, "pair_psi KSP ([B_11,B_13;B_31,B_33])", &
                         comm_s, my_id, rtol, gmg_inst=2, nfields=2, &
                         smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS, &
                         pre0=SF_PJ_PRE0, post0=SF_PJ_POST0, nsmooth_c=SF_PJ_NSC, &
                         axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    call PetscLogEventEnd(pcev_fact_pj, ierr)

    call PetscLogEventBegin(pcev_fact_w, ierr)
    call sf_solver_setup(slv_w, a_w, bk_w, "pair_w KSP (mixed)", &
                         comm_s, my_id, rtol, gmg_inst=1, nfields=nf_w, &
                         smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                         pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC, &
                         axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    call PetscLogEventEnd(pcev_fact_w, ierr)

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    call sf_solver_setup(slv_rho, a_rho, bk_rho, "rho-block KSP", &
                         comm_s, my_id, rtol, gmg_inst=3, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    ! chi_par diffuses along B, i.e. around the rings
    call sf_solver_setup(slv_T, a_T, bk_T, "T-block KSP", &
                         comm_s, my_id, rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS_RHOT)
    ! v_par: v0's B_77 is mass-dominated (rho's setup); b0's W_vv is
    ! diffusion along B (T's rings). b1 solves it inside pair_w.
    if (with_v .and. vstage == SF_VPAR_V0) then
      call sf_solver_setup(slv_v, a_v, bk_v, "v_par-block KSP (B_77)", &
                           comm_s, my_id, rtol, gmg_inst=5, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    else if (with_v .and. vstage == SF_VPAR_B0) then
      call sf_solver_setup(slv_v, a_v, bk_v, "v_par-block KSP (B_77 + W_vv)", &
                           comm_s, my_id, rtol, gmg_inst=5, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS)
    endif
    if (corr == 3 .and. first) then
      block
        use mod_petsc_pc_harm, only: pc_ntor
        call mass_cheb_setup(mcm, a_m, comm_s, "rho / T mass B_44", int(pc_ntor))
      end block
    endif
    call PetscLogEventEnd(pcev_fact_rhot, ierr)
    call physics_pc_mem("SF build: solvers set up", my_id)

    if (.not. vecs_ready) call create_work_vectors()

    g_ctx%reduced_ready = .true.
    sf_first = .false.
    ! the families finish their setups at different times: wait here, so the
    ! setup timer holds the slowest one rather than the first apply
    if (famode) call MPI_Barrier(comm, ierr)

  contains

    subroutine block_scale(slv, A, nf, label)
      type(block_solver_t), intent(inout) :: slv
      Mat :: A
      integer, intent(in) :: nf
      character(len=*), intent(in) :: label
      if (slv%scaled) call VecDestroy(slv%dscale, ierr)
      call make_field_block_scale(A, spread(n1_loc, 1, nf), slv%dscale, comm_s, my_id, label)
      slv%scaled = .true.
    end subroutine block_scale

    subroutine penalty_rows(slv, A, nf, label)
      type(block_solver_t), intent(inout) :: slv
      Mat :: A
      integer, intent(in) :: nf
      character(len=*), intent(in) :: label
      call decouple_penalty_rows(A, spread(n1_loc, 1, nf), slv%pfac, slv%penal, comm_s, my_id, label)
    end subroutine penalty_rows

    !> The full Jacobian's penalty rows, for the back-substitution closing
    !! sf_apply: |a_ii| > 1e6 max |a_ij| over the row's own variable (the
    !! Mach-1 rows also hold zbig couplings to T and u).
    subroutine full_penalty_rows(A)
      use mod_parameters, only: n_tor
      Mat :: A
      PetscInt :: rs, re, row, ncols, bs
      PetscInt, pointer :: cols(:)
      PetscScalar, pointer :: vals(:)
      integer :: jc, n, iv, mpierr
      real*8 :: dg, off
      integer, allocatable :: tl(:)
      real*8, allocatable :: td(:)
      a_full_s = A
      call MatGetBlockSize(A, bs, ierr)
      call MatGetOwnershipRange(A, rs, re, ierr)
      allocate(tl(re - rs), td(re - rs))
      n = 0
      do row = rs, re - 1
        iv = int(mod(row, bs)) / n_tor
        call MatGetRow(A, row, ncols, cols, vals, ierr)
        dg = 0.d0; off = 0.d0
        do jc = 1, int(ncols)
          if (cols(jc) == row) then
            dg = vals(jc)
          else if (int(mod(cols(jc), bs)) / n_tor == iv) then
            off = max(off, abs(vals(jc)))
          endif
        enddo
        call MatRestoreRow(A, row, ncols, cols, vals, ierr)
        if (abs(dg) > 0.d0 .and. abs(dg) > 1.d6 * off) then
          n = n + 1; tl(n) = int(row - rs) + 1; td(n) = dg
        endif
      enddo
      if (allocated(pen_loc)) deallocate(pen_loc, pen_d)
      pen_loc = tl(1:n); pen_d = td(1:n)
      call MPI_Allreduce(n, npen_full, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
      if (.not. pen_w_made) call MatCreateVecs(A, pen_w, PETSC_NULL_VEC, ierr)
      pen_w_made = .true.
      if (my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   full Jacobian: ", npen_full, &
        " penalty rows, back-substituted after every sweep"
    end subroutine full_penalty_rows

    logical function penalty_rows_on()
      use phys_module, only: eliminate_boundary_dofs
      PetscInt :: iv
      PetscBool :: set
      iv = merge(0, 1, eliminate_boundary_dofs)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_penalty_rows", iv, set, ierr)
      penalty_rows_on = (iv /= 0)
    end function penalty_rows_on

    subroutine create_work_vectors()
      use mod_petsc_pc_gmg, only: gmg_dev_on
      integer :: k
      PetscInt :: pv
      PetscBool :: set
      pv = 1
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gpu_apply", pv, set, ierr)
      sfdev = gmg_dev_on() .and. pv > 0 .and. corr /= 3 .and. &
              slv_pj%dev .and. slv_w%dev .and. slv_rho%dev .and. slv_T%dev
      if (with_v .and. sfdev) sfdev = slv_v%dev
      if (my_id == 0 .and. gmg_dev_on()) write(*,'(A,A)') "[Physics PC]   SF apply: ", &
        merge("on the device (work vectors, couplings, packs)", "on the host                                   ", sfdev)
      call MatCreateVecs(a_pj, rhs_PJ, sol_PJ, ierr)
      call MatCreateVecs(a_w, rhs_W, sol_W, ierr)
      call MatCreateVecs(a_rho, sv_x(1), PETSC_NULL_VEC, ierr)
      if (sfdev) then
        call vec_make_dev(rhs_PJ); call vec_make_dev(sol_PJ)
        call vec_make_dev(rhs_W);  call vec_make_dev(sol_W)
        call vec_make_dev(sv_x(1))
      endif
      do k = 2, NSV
        call VecDuplicate(sv_x(1), sv_x(k), ierr)
      enddo
      do k = 1, NSV
        call VecDuplicate(sv_x(1), sv_y(k), ierr)
      enddo
      if (with_v) then
        call VecDuplicate(sv_x(1), wv1, ierr)
        call VecDuplicate(sv_x(1), wv2, ierr)
      endif
      call VecDuplicate(sv_x(1), w3, ierr)
      call VecDuplicate(sv_x(1), w4, ierr)
      call VecDuplicate(sv_x(1), w5, ierr)
      call VecDuplicate(sv_x(1), t_rho, ierr)
      call VecDuplicate(sv_x(1), t_T, ierr)
      call VecDuplicate(sv_x(1), zv, ierr)
      call VecSet(zv, 0.0d0, ierr)
      if (famode) then
        block
          Vec :: xt
          call MatCreateVecs(A_full, xt, PETSC_NULL_VEC, ierr)
          call sff_vec_setup(xt, NSV, dev=sfdev)
          call VecDestroy(xt, ierr)
        end block
      endif
      vecs_ready = .true.
    end subroutine create_work_vectors

    !> Mode split: this rank's family operators, straight from A and W. The
    !! first call splits the ranks and builds them; later calls refill values.
    subroutine family_operators(first_)
      use mod_petsc_pc_harm, only: pc_ntor
      logical, intent(in) :: first_
      integer :: k, nno
      Mat :: ws(WS_N)
      ! W, and v_par's three where assembled (W stands in, unread, otherwise)
      ws = g_ctx%W_force
      if (g_ctx%w_v_ready) then
        ws(WS_VV) = g_ctx%W_vv; ws(WS_VU) = g_ctx%W_vu; ws(WS_UV) = g_ctx%W_uv
      endif
      if (first_) then
        call require_w()
        call sff_decompose(A_full, comm, my_id)
        call define_family_operators()
      endif
      call PetscLogEventBegin(pcev_extract, ierr)
      call build_or_fill(o_pj, ws, first_); call build_or_fill(o_w, ws, first_)
      call build_or_fill(o_rho, ws, first_); call build_or_fill(o_T, ws, first_)
      if (with_v .and. vstage /= SF_VPAR_B1) call build_or_fill(o_v, ws, first_)
      if (corr == 3 .and. first_) call build_or_fill(o_m, ws, first_)    ! geometry only
      do k = 1, ncb_on
        call build_or_fill(o_cb(k), ws, first_)
      enddo
      call PetscLogEventEnd(pcev_extract, ierr)
      !--- device path: the operators become device matrices, filled there
      !--- from the next build on (-sf_gpu_gather 0: host fills + copies)
      if (first_ .and. sf_dev_gather()) then
        block
          use mod_petsc_pc_gmg, only: gmg_dev_mtype
          character(len=64) :: mt
          mt = gmg_dev_mtype()
          call sff_op_dev(o_pj, mt);  call sff_op_dev(o_w, mt)
          call sff_op_dev(o_rho, mt); call sff_op_dev(o_T, mt)
          if (with_v .and. vstage /= SF_VPAR_B1) call sff_op_dev(o_v, mt)
          if (corr == 3) call sff_op_dev(o_m, mt)
          do k = 1, ncb_on
            call sff_op_dev(o_cb(k), mt)
          enddo
        end block
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
      if (.not. first_) return
      call set_use_b65(a_65, comm)
      ! the coupling blocks on the threaded block kernel; it reads the CSR
      ! arrays in place, so one attach holds across the refills
      nno = 0
      do k = 1, ncb_on
        if (.not. blockmv_attach(o_cb(k)%fam, int(pc_ntor))) nno = nno + 1
      enddo
      if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   SF: WARNING ", nno, &
        " family coupling block(s) not AIJ, left on PETSc's single-threaded matvec"
    end subroutine family_operators

    subroutine build_or_fill(op, ws, first_)
      type(fam_op_t), intent(inout) :: op
      Mat, intent(in) :: ws(:)
      logical, intent(in) :: first_
      if (first_) then
        call sff_op_build(op, A_full, ws, sf_opz(), my_id)
      else
        call sff_op_fill(op, A_full, ws, sf_opz())
      endif
    end subroutine build_or_fill

    !> The family operators' block layouts, as the global path packs them.
    subroutine define_family_operators()
      integer, parameter :: ce(12) = [var_psi, var_psi, var_u, var_u, var_u, var_u, &
                                      var_rho, var_rho, var_T, var_T, var_T, var_T]
      integer, parameter :: cv(12) = [var_u, var_T, var_psi, var_zj, var_rho, var_T, &
                                      var_psi, var_u, var_psi, var_u, var_zj, var_rho]
      character(len=4), parameter :: cn(12) = ["B_12", "B_16", "B_21", "B_23", "B_25", "B_26", &
                                                "B_51", "B_52", "B_61", "B_62", "B_63", "B_65"]
      !> v_par's, in VC_ij order (o_cb 13 .. 18)
      integer, parameter :: cev(6) = [var_Vpar, var_Vpar, var_Vpar, var_Vpar, var_rho, var_T]
      integer, parameter :: cvv(6) = [var_psi, var_u, var_rho, var_T, var_Vpar, var_Vpar]
      character(len=4), parameter :: cnv(6) = ["B_71", "B_72", "B_75", "B_76", "B_57", "B_67"]
      integer :: k
      call full_op(o_pj, "pair_psi", 1, [var_psi, var_zj])
      if (with_v .and. vstage == SF_VPAR_B1) then
        ! v_par's row and column carry only u (psi_w must not enter its row)
        call full_op(o_w, "pair_w (wpj + v_par)", 2, [var_u, var_w, var_psi, var_zj, var_Vpar])
        o_w%se(2, 3:5) = 0; o_w%se(3, 2) = 0; o_w%se(3, 5) = 0; o_w%se(4, 1:2) = 0; o_w%se(4, 5) = 0
        o_w%se(5, 2:4) = 0
        if (btest(vwmask, 0)) o_w%wid(5, 5) = WS_VV
        if (btest(vwmask, 1)) o_w%wid(5, 1) = WS_VU
        if (btest(vwmask, 2)) o_w%wid(1, 5) = WS_UV
        if (vfull) then                                  ! B_71 on psi_w
          o_w%se(5, 3) = var_Vpar; o_w%sv(5, 3) = var_psi
        endif
      else
        call full_op(o_w, "pair_w (wpj)", 2, [var_u, var_w, var_psi, var_zj])
        o_w%se(2, 3:4) = 0; o_w%se(3, 2) = 0; o_w%se(4, 1:2) = 0
      endif
      ! the psi row's mass: opz B_33, in model183 B_11 (mod_petsc_pc_sf_mixed)
      if (jorek_model == 183) then
        o_w%se(3, 3) = var_psi; o_w%sv(3, 3) = var_psi; o_w%scl(3, 3) = 0
      else
        o_w%se(3, 3) = var_zj;  o_w%sv(3, 3) = var_zj;  o_w%scl(3, 3) = 1
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
      if (corr == 3) call full_op(o_m, "B_44", NDIAG + NCB + 1, [var_w])
      do k = 1, ncb_on
        o_cb(k)%nf = 1; o_cb(k)%tagb = NDIAG + k
        if (k <= 12) then
          o_cb(k)%label = cn(k); o_cb(k)%se(1, 1) = ce(k); o_cb(k)%sv(1, 1) = cv(k)
        else
          o_cb(k)%label = cnv(k - 12); o_cb(k)%se(1, 1) = cev(k - 12); o_cb(k)%sv(1, 1) = cvv(k - 12)
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
      integer :: a
      op%label = label; op%nf = size(vars); op%tagb = tagb
      do a = 1, size(vars)
        op%se(a, 1:size(vars)) = vars(a); op%sv(1:size(vars), a) = vars(a)
      enddo
    end subroutine full_op

    !> First build on the global path: every block in one pass over A, the
    !! packed pairs, and the value map that refills the blocks later.
    subroutine first_build_operators()
      use mod_parameters, only: n_tor
      integer, parameter :: NVP = merge(7, 0, with_v)      !< v_par's blocks
      integer, parameter :: NBLK = 22 + NVP
      integer :: eqs(NBLK), vrs(NBLK), nno, k
      Mat :: M(NBLK)
      Mat, allocatable :: cb(:)

      call require_w()
      call PetscLogEventBegin(pcev_extract, ierr)
      ! the sweep's (1-14), v_par's (B_71 B_72 B_75 B_76 B_77 B_57 B_67), and
      ! the blocks only the pairs are packed from
      eqs(1:14) = [var_psi, var_psi, var_u, var_u, var_u, var_u, var_rho, var_rho, var_rho, &
                   var_T, var_T, var_T, var_T, var_T]
      vrs(1:14) = [var_u, var_T, var_psi, var_zj, var_rho, var_T, var_psi, var_u, var_rho, &
                   var_psi, var_u, var_zj, var_T, var_rho]
      if (with_v) then
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

      ! pair_psi's row 2 IS Jacobian row 3, so its j component already is
      ! j* = M_j^-1 (x_j - B_31 psi*)
      call PetscLogEventBegin(pcev_convert, ierr)
      call pack_pair_aij(g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, g_ctx%B_33, &
                         g_ctx%K_pj_aij, kpj_packed, comm)
      call PetscLogEventEnd(pcev_convert, ierr)
      call PetscLogEventBegin(pcev_build_suu, ierr)
      call sfm_build()
      call PetscLogEventEnd(pcev_build_suu, ierr)

      ! pair_w straight from A and W by the value map; then the blocks only
      ! its re-assembly read are dropped (corrector 3 keeps B_44: a_m)
      block
        integer :: mse(SFM_NF, SFM_NF), msv(SFM_NF, SFM_NF)
        logical :: mzs(SFM_NF, SFM_NF), mok
        call sfm_map_spec(mse, msv, mzs, mok)
        call sfg_build(A_full, g_ctx%W_force, M, eqs, vrs, g_ctx%K_pj_aij, [var_psi, var_zj], &
                       comm, my_id, Mix=sfm_op, mix_se=mse, mix_sv=msv, mix_zs=mzs, &
                       opz=sf_opz(), mix_ok=sfm_gathered)
        if (sfm_gathered .and. corr /= 3) then
          call sfg_drop(M(15+NVP:NBLK))
          call sfm_release_parts()
          do k = 15 + NVP, NBLK
            call MatDestroy(M(k), ierr)
          enddo
          ! g_ctx held copies of the released handles: null them, so nothing
          ! can reach a freed Mat through them
          g_ctx%B_13 = PETSC_NULL_MAT; g_ctx%B_31 = PETSC_NULL_MAT; g_ctx%B_33 = PETSC_NULL_MAT
          g_ctx%B_11 = PETSC_NULL_MAT; g_ctx%B_22 = PETSC_NULL_MAT; g_ctx%B_24 = PETSC_NULL_MAT
          g_ctx%B_42 = PETSC_NULL_MAT; g_ctx%B_44 = PETSC_NULL_MAT
        endif
      end block
      if (with_v) call vpar_global_sums(.true.)

      ! the coupling blocks on the threaded block kernel (fixed patterns, so
      ! one attach holds for the run); a refused attach is reported
      cb = [g_ctx%B_12, g_ctx%B_16, g_ctx%B_21, g_ctx%B_23, g_ctx%B_25, g_ctx%B_26, &
            g_ctx%B_51, g_ctx%B_52, g_ctx%B_61, g_ctx%B_62, g_ctx%B_63, gB_65]
      if (with_v) then
        cb = [cb, gB_71, gB_75, gB_76, gB_72]
        if (c72_on) cb = [cb, c72_g]
        if (vstage == SF_VPAR_B0) cb = [cb, gB_57, gB_67]
      endif
      nno = 0
      do k = 1, size(cb)
        if (.not. blockmv_attach(cb(k), int(n_tor))) nno = nno + 1
      enddo
      if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   SF: WARNING ", nno, &
        " coupling block(s) not AIJ, left on PETSc's single-threaded matvec"

      !--- device path: the operators become device matrices, gathered there
      !--- on every rebuild (-sf_gpu_gather 0: host gather + device copies)
      if (sf_dev_gather()) then
        block
          use mod_petsc_pc_gmg, only: gmg_dev_mtype
          logical :: dok
          call sfg_dev_setup(A_full, g_ctx%W_force, gmg_dev_mtype(), sf_opz(), comm, my_id, dok)
        end block
      endif
    end subroutine first_build_operators

    !> b0 on the global path: vop_g = B_77 + W_vv, c72_g = B_72 + W_vu (as
    !! -sf_vpar_w selects), union patterns fixed at the first build.
    subroutine vpar_global_sums(first_)
      logical, intent(in) :: first_
      if (vstage /= SF_VPAR_B0) return
      if (.not. g_ctx%w_v_ready) then
        if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: v_par's W_vv / W_vu were never "// &
          "assembled (physics_pc_sf_vpar_stage = b0 needs them)."
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      endif
      vop_on = btest(vwmask, 0);  c72_on = btest(vwmask, 1)
      if (vop_on) call add_into(vop_g, gB_77, g_ctx%W_vv, first_)
      if (c72_on) call add_into(c72_g, gB_72, g_ctx%W_vu, first_)
    end subroutine vpar_global_sums

    !> S = B + Wm (S created on the first build)
    subroutine add_into(S, B, Wm, first_)
      Mat :: S, B, Wm
      logical, intent(in) :: first_
      if (first_) then
        call MatDuplicate(B, MAT_COPY_VALUES, S, ierr)
        call MatAXPY(S, 1.0d0, Wm, DIFFERENT_NONZERO_PATTERN, ierr)
      else
        call MatCopy(B, S, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(S, 1.0d0, Wm, SUBSET_NONZERO_PATTERN, ierr)
      endif
    end subroutine add_into

    subroutine require_w()
      if (g_ctx%w_force_ready) return
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: W_force was never assembled "// &
        "(petsc_assemble_pc_matrices did not run?)."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine require_w

  end subroutine sf_build

  !> The SF operators gathered on the device (-sf_gpu 1 and -sf_gpu_gather 1,
  !! the default with it, in a Kokkos build). Not with decoupled penalty rows:
  !! those are edited on the host after every fill, so the host gathers and
  !! the GMG copies the result up.
  logical function sf_dev_gather()
    use mod_petsc_raw_csr, only: sfg_dev_available
    use phys_module, only: eliminate_boundary_dofs
    PetscInt :: pg, pv, pp
    PetscBool :: set
    PetscErrorCode :: ierr
    pg = 0; pv = 1; pp = merge(0, 1, eliminate_boundary_dofs)
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gpu", pg, set, ierr)
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gpu_gather", pv, set, ierr)
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_penalty_rows", pp, set, ierr)
    sf_dev_gather = pg > 0 .and. pv > 0 .and. pp == 0 .and. sfg_dev_available()
  end function sf_dev_gather

  !> use_b65 = B_65 has a nonzero entry anywhere on comm.
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

  !--------------------------------------------------------------------
  !> y = P^-1 x: the block-LDU sweep.
  !!   1. predictor  pair_psi (psi*, j*) = (x_psi, x_j); rho*, T* against it
  !!   2. pair_w (u, omega, psi_w, j_w) on the rest
  !!   3. corrector  psi, j: Eq. (16) second pair_psi solve (corrector 0), or
  !!                 psi* + psi_w, j* + j_w (Eq. (17), correctors 1-4);
  !!                 rho, T: block solves on (B_52 u, B_62 u) (+ pair_w's
  !!                 (psi_w, j_w) with corrector 4), or their mass inverse (3)
  !! v_par: b0 between 2 and 3, b1 inside 2, v0 after 3.
  !--------------------------------------------------------------------
  subroutine sf_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr

    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T, x_v
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T, y_v
    logical :: vb0, vb1, vcor
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
    x_v = zv; y_v = zv                                   ! without v_par: placeholders, never read
    vb0 = .false.; vb1 = .false.
    if (with_v) then
      x_v = sv_x(NSV); y_v = sv_y(NSV)
      vb0 = (vstage == SF_VPAR_B0)
      vb1 = (vstage == SF_VPAR_B1)
    endif
    vcor = vb0 .or. vb1                                  ! v_par in the rho / T correctors

    !--- 1. predictor: (psi*, j*) = pair_psi^-1 (x_psi, x_j). x_j, not 0: it
    !--- makes j* the mass back-substitution M_j^-1 (x_j - B_31 psi*).
    call sf_split_parts(rhs_PJ, [x_psi, x_j], .true.)
    call solve(slv_pj, rhs_PJ, sol_PJ, pcev_solve_pj)
    call sf_split_parts(sol_PJ, [y_psi, y_j], .false.)

    ! rho* = B_55^-1 (x_rho - B_51 psi*)
    call cmult(a_51, y_psi, w3)
    call VecWAXPY(w4, -1.0d0, w3, x_rho, ierr)
    call solve(slv_rho, w4, t_rho, pcev_solve_rhot)

    ! T* = B_66^-1 (x_T - B_61 psi* - B_63 j* - B_65 rho*)
    call cmult(a_61, y_psi, w3)
    call VecWAXPY(w4, -1.0d0, w3, x_T, ierr)
    call sub_mult(a_63, y_j, w4)
    if (use_b65) call sub_mult(a_65, t_rho, w4)
    call solve(slv_T, w4, t_T, pcev_solve_rhot)

    !--- 2. pair_w on (x_u - B_21 psi* - B_23 j* - B_25 rho* - B_26 T*, x_w,
    !--- -B_16 T* or 0, 0). The omega row's lower coupling is zero, so x_w
    !--- enters verbatim; the psi / j rows' x_psi, x_j went to the predictor.
    call VecCopy(x_u, w5, ierr)
    call sub_mult(a_21, y_psi, w5)
    call sub_mult(a_23, y_j, w5)
    call sub_mult(a_25, t_rho, w5)
    call sub_mult(a_26, t_T, w5)
    if (corr == 1 .or. corr == 4) then
      call cmult(a_16, t_T, w4)
      call VecScale(w4, -1.0d0, ierr)
    else
      call VecSet(w4, 0.0d0, ierr)
    endif
    if (vb1) then                                        ! x_V - B_71 psi* - B_75 rho* - B_76 T*
      call cmult(a_vc(VC71), y_psi, wv1)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      call sub_mult(a_vc(VC75), t_rho, wv2)
      call sub_mult(a_vc(VC76), t_T, wv2)
      call sf_split_parts(rhs_W, [w5, x_w, w4, zv, wv2], .true.)
    else
      call sf_split_parts(rhs_W, [w5, x_w, w4, zv], .true.)
    endif
    call solve(slv_w, rhs_W, sol_W, pcev_solve_w)
    ! w3, w4 = psi_w, j_w (discarded by corrector 0); b1: y_v = V
    if (vb1) then
      call sf_split_parts(sol_W, [y_u, y_w, w3, w4, y_v], .false.)
    else
      call sf_split_parts(sol_W, [y_u, y_w, w3, w4], .false.)
    endif

    !--- v_par b0: (B_77 + W_vv) V = x_V - B_71 psi* - B_75 rho* - B_76 T*
    !--- - (B_72 + W_vu) u. psi* rather than psi_w: W_vu leaves out the part
    !--- of the p channel that cancels against the psi channel, unless vfull.
    if (vb0) then
      call cmult(a_vc(VC71), y_psi, wv1)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      if (vfull) call sub_mult(a_vc(VC71), w3, wv2)
      call sub_mult(a_vc(VC75), t_rho, wv2)
      call sub_mult(a_vc(VC76), t_T, wv2)
      call sub_mult(a_vc(VC72), y_u, wv2)
      call solve(slv_v, wv2, y_v, pcev_solve_rhot)
    endif

    ! corrector 4: pair_w's change of (psi, j) in the rho / T corrections,
    ! held in y_rho / y_T until step 3
    if (corr == 4) then
      call cmult(a_51, w3, y_rho)
      call cmult(a_61, w3, y_T)
      call cmult(a_63, w4, w5)
      call VecAXPY(y_T, 1.0d0, w5, ierr)
    endif

    !--- 3. corrector: psi, j
    if (corr /= 0) then
      call VecAXPY(y_psi, 1.0d0, w3, ierr)
      call VecAXPY(y_j,   1.0d0, w4, ierr)
    else
      ! pair_psi (dpsi, dj) = (B_12 u + B_16 T*, 0): U's j row is zero
      call cmult(a_12, y_u, w3)
      call cmult(a_16, t_T, w4)
      call VecAXPY(w3, 1.0d0, w4, ierr)
      call VecZeroEntries(w4, ierr)
      call sf_split_parts(rhs_PJ, [w3, w4], .true.)
      call solve(slv_pj, rhs_PJ, sol_PJ, pcev_solve_pj)
      call sf_split_parts(sol_PJ, [w3, w4], .false.)
      call VecAXPY(y_psi, -1.0d0, w3, ierr)
      call VecAXPY(y_j,   -1.0d0, w4, ierr)
    endif

    !--- 3. corrector: rho, T (only u enters: U's omega column is zero)
    if (corr == 3) then
      ! mass inverse (opz B_44)^-1, no block solve (B_65 is zero in model199)
      call cmult(a_52, y_u, w3)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_rho, -1.0d0 / sf_opz(), w5, t_rho, ierr)
      call cmult(a_62, y_u, w3)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_T, -1.0d0 / sf_opz(), w5, t_T, ierr)
    else
      ! [[B_55, 0], [B_65, B_66]] (drho, dT) = (B_52 u [+ B_51 psi_w] [+ B_57 V],
      ! B_62 u [+ B_61 psi_w + B_63 j_w] [+ B_67 V])
      call cmult(a_52, y_u, w3)
      if (corr == 4) call VecAXPY(w3, 1.0d0, y_rho, ierr)
      if (vcor) call add_mult(a_vc(VC57), y_v, w3)
      call solve(slv_rho, w3, w5, pcev_solve_rhot)
      call VecWAXPY(y_rho, -1.0d0, w5, t_rho, ierr)

      call cmult(a_62, y_u, w3)
      if (corr == 4) call VecAXPY(w3, 1.0d0, y_T, ierr)
      if (vcor) call add_mult(a_vc(VC67), y_v, w3)
      if (use_b65) call sub_mult(a_65, w5, w3)
      call solve(slv_T, w3, w5, pcev_solve_rhot)
      call VecWAXPY(y_T, -1.0d0, w5, t_T, ierr)
    endif

    !--- v_par v0: B_77 V = x_V - B_71 psi - B_72 u - B_75 rho - B_76 T
    if (with_v .and. .not. vcor) then
      call cmult(a_vc(VC71), y_psi, wv1)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      call sub_mult(a_vc(VC72), y_u, wv2)
      call sub_mult(a_vc(VC75), y_rho, wv2)
      call sub_mult(a_vc(VC76), y_T, wv2)
      call solve(slv_v, wv2, y_v, pcev_solve_rhot)
    endif

    if (famode) then
      t1 = MPI_Wtime()
      call sff_scatter_out(sv_y, y)
      t_scat = t_scat + (MPI_Wtime() - t1)
      t_apply = t_apply + (MPI_Wtime() - t0)
    else
      call merge_vars(sv_y, y)
    endif
    if (npen_full > 0) call penalty_backsub(x, y)
    call PetscLogEventEnd(pcev_apply, ierr)
    ierr = 0

  contains

    subroutine solve(slv, b, s, ev)
      type(block_solver_t), intent(inout) :: slv
      Vec :: b, s
      PetscLogEvent, intent(in) :: ev
      call PetscLogEventBegin(ev, ierr)
      call sf_solver_apply(slv, b, s, ierr)
      call PetscLogEventEnd(ev, ierr)
    end subroutine solve

    !> r -= A v (w3 / w4 / wv1 free as scratch: the caller's r is another one)
    subroutine sub_mult(A, v, r)
      Mat :: A
      Vec :: v, r
      call cmult(A, v, scratch(r))
      call VecAXPY(r, -1.0d0, scratch(r), ierr)
    end subroutine sub_mult

    subroutine add_mult(A, v, r)
      Mat :: A
      Vec :: v, r
      call cmult(A, v, scratch(r))
      call VecAXPY(r, 1.0d0, scratch(r), ierr)
    end subroutine add_mult

    !> The penalty rows' x_b = (r_b - sum_j a_bj y_j) / a_bb with the
    !! sweep's y: a penalty row's residual is zbig times x_b's error, so the
    !! sweep's x_b = r_b / a_bb (decoupled rows) leaves it at the size of the
    !! whole outer residual. Changes the rest of y's residual by O(1 / zbig).
    subroutine penalty_backsub(r, z)
      Vec :: r, z
      PetscScalar, pointer :: zp(:), rp(:), wp(:)
      integer :: q
      call MatMult(a_full_s, z, pen_w, ierr)
      call VecGetArray(z, zp, ierr)
      call VecGetArrayRead(r, rp, ierr)
      call VecGetArrayRead(pen_w, wp, ierr)
      do q = 1, size(pen_loc)
        zp(pen_loc(q)) = zp(pen_loc(q)) + (rp(pen_loc(q)) - wp(pen_loc(q))) / pen_d(q)
      enddo
      call VecRestoreArrayRead(pen_w, wp, ierr)
      call VecRestoreArrayRead(r, rp, ierr)
      call VecRestoreArray(z, zp, ierr)
    end subroutine penalty_backsub

    !> y = B x for a coupling block of the sweep; on the device path with B's
    !! device copy, refilled once per build. The copy's slot is B's place in
    !! the sweep's list (B_12 .. B_65, then v_par's a_vc).
    subroutine cmult(B, xx, yy)
      use mod_petsc_pc_gmg, only: gmg_mat_dev
      Mat :: B
      Vec :: xx, yy
      Mat :: lst(NAD)
      integer :: k, n
      if (.not. sfdev) then
        call MatMult(B, xx, yy, ierr)
        return
      endif
      lst(1:12) = [a_12, a_16, a_21, a_23, a_25, a_26, a_51, a_52, a_61, a_62, a_63, a_65]
      n = 12
      if (with_v) then
        lst(13:18) = a_vc; n = 18
      endif
      k = 0
      do while (k < n)
        k = k + 1
        if (lst(k) == B) exit
      enddo
      if (.not. (lst(k) == B)) stop "cmult: not a coupling block of the sweep"
      if (ad_gen(k) /= sf_gen) then
        call gmg_mat_dev(B, ad(k), ad_gen(k) < 0)
        ad_gen(k) = sf_gen
      endif
      call MatMult(ad(k), xx, yy, ierr)
    end subroutine cmult

    !> a work vector distinct from r and from every operand still live
    function scratch(r) result(s)
      Vec, intent(in) :: r
      Vec :: s
      if (with_v .and. (r == wv2)) then
        s = wv1
      else if (r == w3) then
        s = w4
      else
        s = w3
      endif
    end function scratch

  end subroutine sf_apply

  !> Inner-iteration summary, one line per block (mode split: per family),
  !! then reset. Collective.
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
  !! the rest of the sweep and the scatters (max over the family's ranks).
  !! The scatter back completes only when every family has finished, so the
  !! critical family is the one with the smallest scatter time.
  subroutine family_report(my_id)
    use mod_petsc_pc_sf_fam, only: sff_fam, sff_nfam
    integer, intent(in) :: my_id
    integer, parameter :: NV = 14
    real*8 :: v(NV), vf(NV)
    real*8, allocatable :: g(:, :)
    integer :: np, r, f, nrk, mpierr
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
      r = findloc(nint(g(12, :)), f, 1) - 1
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
  end subroutine family_report

  real*8 function its_mean(slv)
    type(block_solver_t), intent(in) :: slv
    its_mean = 0.d0
    if (slv%nsolve > 0) its_mean = dble(slv%its_sum) / dble(slv%nsolve)
  end function its_mean

#endif
end module mod_petsc_pc_sf
