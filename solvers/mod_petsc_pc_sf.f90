!> The split-field (SF) physics-based preconditioner (use_physics_pc): a
!! block-LDU sweep over (psi, u, j, omega, rho, T), plus v_par where the model
!! has it, after Chacon, JCP 526 (2025). The two mixed pairs keep j and omega
!! explicit, with B_ij = A(equation i, variable j):
!!   pair_psi = [[B_11, B_13], [B_31, B_33]]                  (psi, j)
!!   pair_w   = [[B_22 + W, B_24, B_21,      B_23],           (u, omega, psi, j)
!!               [B_42,     B_44, 0,         0   ],
!!               [B_12,     0,    opz M_psi, B_13],
!!               [0,        0,    B_31,      B_33]]
!! W is the curvature part of the composed force operator (Chacon Eqs. 17-19)
!! and M_psi = B_33 (B_11 in model183). physics_pc_sf_harm_couple .f. makes
!! every block |n|-diagonal; physics_pc_sf_mode_split then solves each |n|
!! family on its own rank group.
module mod_petsc_pc_sf
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var, var_Vpar, var_T, var_Ti, var_Te
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, pcev_extract, pcev_fact_pj, pcev_fact_w, &
       pcev_fact_rhot, pcev_solve_pj, pcev_solve_w, pcev_solve_rhot, pcev_apply
  use mod_petsc_pc_blocks, only: make_field_block_scale, decouple_penalty_rows
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_ops, only: sf_op_t, sfo_layout, sfo_build, sfo_fill, sfo_vars_in, sfo_vars_out, &
       sfo_comm, sfo_n1
  use mod_petsc_raw_csr, only: blockmv_attach
#if PETSC_VERSION_LT(3,23,0)
  use mod_petsc_raw_csr, only: mat_get_row, mat_restore_row
#endif
  use mod_petsc_pc_mass_cheb, only: mass_cheb_t, mass_cheb_setup, mass_cheb_solve
#include "petsc_f90_compat.h"
  implicit none
  private

  logical, save :: sf_init_done = .false.
  logical, save :: sf_first     = .true.

  type(block_solver_t), save :: slv_pj, slv_w, slv_rho, slv_T, slv_v
  integer, save :: bk_pj = SF_GMG, bk_w = SF_GMG, bk_rho = SF_GMG, bk_T = SF_GMG, bk_v = SF_GMG
  integer, save :: corr = 4                    !< physics_pc_sf_corrector
  logical, save :: famode = .false.            !< mode split
  logical, save :: harm_all = .true.           !< physics_pc_sf_harm_couple

  logical, parameter :: with_v    = (var_Vpar > 0)
  logical, parameter :: with_tite = (var_Ti > 0 .and. var_Te > 0)
  integer, parameter :: var_T1    = max(var_T, var_Ti)
  integer, parameter :: var_T2    = max(var_Te, 1)
  integer, parameter :: NSV       = n_var              !< variables of the sweep
  integer, save :: vstage = SF_VPAR_V0

  !> The operators: the diagonal blocks, corrector 3's mass (opz B_44) and
  !! the coupling blocks o_cb(CB_ij) = B_ij (CB_72: + W_vu in stage b0).
  type(sf_op_t), save :: o_pj, o_w, o_rho, o_T, o_v, o_m
  integer, parameter :: CB_12 = 1, CB_16 = 2, CB_21 = 3, CB_23 = 4, CB_25 = 5, CB_26 = 6, &
                        CB_51 = 7, CB_52 = 8, CB_61 = 9, CB_62 = 10, CB_63 = 11, CB_65 = 12, &
                        CB_71 = 13, CB_72 = 14, CB_75 = 15, CB_76 = 16, CB_57 = 17, CB_67 = 18, &
                        CB_18 = 19, CB_28 = 20, CB_81 = 21, CB_82 = 22, CB_83 = 23, CB_85 = 24, &
                        CB_78 = 25, CB_87 = 26
  integer, parameter :: NCB = 26
  type(sf_op_t), save :: o_cb(NCB)
  logical, save :: cb_active(NCB) = .false.
  !> the W sources (sf_op_t wid)
  integer, parameter :: WS_C = 1, WS_VV = 2, WS_VU = 3, WS_UV = 4, WS_N = 4
  !> B_65 / B_85 (T row, rho column) vanish in model199 and are applied only where nonzero
  logical, save :: use_b65 = .false., use_b85 = .false.
  type(mass_cheb_t), save :: mcm
  integer, save :: comm_g = MPI_COMM_NULL      !< the full system's communicator
  real*8, save  :: t_apply = 0.d0, t_scat = 0.d0

  Vec, save :: sv_x(NSV), sv_y(NSV)
  Vec, save :: rhs_PJ, sol_PJ, rhs_W, sol_W, rhs_T, sol_T
  Vec, save :: w3, w4, w5, w6, w_scr, t_rho, t_T, t_Te, wv1, wv2
  Vec, save :: zv                              !< zero field
  logical, save :: vecs_ready = .false.

  !> penalty rows of the full Jacobian (eliminate_boundary_dofs = .f.): local
  !! offsets and diagonals, refreshed on every build
  Mat, save :: a_full_s
  integer, save :: npen_full = 0
  integer, allocatable, save :: pen_loc(:)
  real*8, allocatable, save  :: pen_d(:)
  Vec, save :: pen_w
  logical, save :: pen_w_made = .false.

  public :: sf_build, sf_apply, sf_report, sf_refresh_operator

contains

  !> Resolve the namelist once and refuse configurations the sweep cannot run.
  subroutine sf_init(my_id)
    use phys_module, only: physics_pc_sf_pair_psi, physics_pc_sf_pair_w, physics_pc_sf_rho, physics_pc_sf_T, &
                           physics_pc_sf_rtol, physics_pc_sf_harm_couple, physics_pc_sf_corrector, &
                           physics_pc_sf_mode_split, physics_pc_sf_vpar, physics_pc_sf_vpar_stage, use_pfafem
    use mod_parameters, only: n_tor, n_period, n_coord_period
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr
    logical :: ok_vars, floquet_split
    character(len=*), parameter :: P = "[Physics PC]   "

    if (sf_init_done) return

    bk_pj  = backend_of(physics_pc_sf_pair_psi, "physics_pc_sf_pair_psi")
    bk_w   = backend_of(physics_pc_sf_pair_w,   "physics_pc_sf_pair_w")
    bk_rho = backend_of(physics_pc_sf_rho,      "physics_pc_sf_rho")
    bk_T   = backend_of(physics_pc_sf_T,        "physics_pc_sf_T")

    if (jorek_model /= 199 .and. jorek_model /= 600 .and. jorek_model /= 183 .and. jorek_model /= 180) &
      call fatal("physics_pc_sf supports model199, model600, model180 and model183 only")
    ok_vars = (n_var == 6 .and. var_T == 6) .or. &
              ((jorek_model == 600 .or. jorek_model == 183 .or. jorek_model == 180) .and. &
               ((n_var == 7 .and. var_Vpar == 7 .and. var_T == 6) .or. &
                (n_var == 7 .and. var_Ti == 6 .and. var_Te == 7 .and. var_Vpar == 0) .or. &
                (n_var == 8 .and. var_Ti == 6 .and. var_Vpar == 7 .and. var_Te == 8)))
    if (.not. ok_vars) &
      call fatal("physics_pc_sf supports (psi, u, zj, w, rho, T or Ti/Te) [+ v_par] with neutrals/impurities off")
    if (.not. use_pfafem .and. mod(n_tor, 2) /= 1) &
      call fatal("physics_pc_sf needs n_tor odd (n = 0 plus cos/sin pairs)")

    harm_all      = physics_pc_sf_harm_couple
    sf_gmg_harm   = merge(SF_GMG_HARM_ALL, SF_GMG_HARM_PAIR, harm_all)
    floquet_split = physics_pc_sf_mode_split .and. harm_all .and. (n_coord_period > n_period)
    famode        = physics_pc_sf_mode_split .and. ((.not. harm_all) .or. floquet_split)

    corr = physics_pc_sf_corrector
    if (corr == -1) corr = 4
    if (corr < 0 .or. corr > 4) call fatal("physics_pc_sf_corrector must be -1 (= 4), 0, 1, 2, 3 or 4")
    if (corr == 3 .and. jorek_model /= 199) &
      call fatal("physics_pc_sf_corrector = 3 supports model199 only (rho / T mass = opz B_44)")

    cb_active = .false.
    cb_active(1:12) = .true.
    if (with_v) then
      bk_v   = backend_of(physics_pc_sf_vpar, "physics_pc_sf_vpar")
      vstage = sf_vpar_stage()
      if (vstage < 0) call fatal("physics_pc_sf_vpar_stage must be v0 | b0 | b1, got '"// &
                                 trim(physics_pc_sf_vpar_stage)//"'")
      if (vstage /= SF_VPAR_V0 .and. jorek_model /= 600) &
        call fatal("physics_pc_sf_vpar_stage = b0 | b1 needs model600 (W_vv): run v0")
      cb_active(13:16) = .true.
      if (vstage /= SF_VPAR_V0) cb_active(17:18) = .true.
      if (vstage == SF_VPAR_B1) then
        if (.not. famode .or. harm_all) call fatal("physics_pc_sf_vpar_stage = b1 needs the mode split "// &
                                                   "(physics_pc_sf_harm_couple = .f., physics_pc_sf_mode_split = .t.)")
        if (corr /= 1 .and. corr /= 2) &
          call fatal("physics_pc_sf_vpar_stage = b1 needs the Eq. (17) corrector (physics_pc_sf_corrector 1 | 2)")
      endif
    endif
    if (with_tite) then
      cb_active(19:24) = .true.
      if (with_v) then
        cb_active(25) = .true.
        if (vstage /= SF_VPAR_V0) cb_active(26) = .true.
      endif
    endif

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ================ split-field (SF) preconditioner ================"
      write(*,'(A,A)') P//"pair_psi : ", trim(physics_pc_sf_pair_psi)
      write(*,'(A,A)') P//"pair_w   : ", trim(physics_pc_sf_pair_w)//" (u, omega, psi, j; W = curvature)"
      if (with_tite) then
        write(*,'(A,A)') P//"rho / TiTe: ", trim(physics_pc_sf_rho)//" / "//trim(physics_pc_sf_T)//" (coupled Ti, Te)"
      else
        write(*,'(A,A)') P//"rho / T  : ", trim(physics_pc_sf_rho)//" / "//trim(physics_pc_sf_T)
      endif
      write(*,'(A,ES9.2)') P//"inner rtol: ", physics_pc_sf_rtol
      if (use_pfafem) &
        write(*,'(A,I0,A)') P//"toroidal : PFAFEM B-spline (n_tor = ", n_tor, ") -> orthogonal Real-DFT (Q^T A Q)"
      if (floquet_split) then
        write(*,'(A,I0,A)') P//"cross-|n|: Floquet family-coupled (N_fp = ", n_coord_period / max(1, n_period), &
          "), all family slots per GMG smoother block"
      else if (harm_all) then
        write(*,'(A)') P//"cross-|n|: all kept, all slots per GMG smoother block"
      else
        write(*,'(A)') P//"cross-|n|: none, |n|-diagonal blocks, cos / sin per GMG smoother block"
      endif
      if (floquet_split) then
        write(*,'(A)') P//"layout   : one stellarator Floquet mode family per rank group (mode_split + harm_couple)"
      else if (famode) then
        write(*,'(A)') P//"layout   : one |n| family per rank group (physics_pc_sf_mode_split)"
      else
        write(*,'(A)') P//"layout   : global operators"
      endif
      if (with_v) then
        select case (vstage)
        case (SF_VPAR_V0); write(*,'(A)') P//"v_par    : "//trim(physics_pc_sf_vpar)//", v0 (B_77 last)"
        case (SF_VPAR_B0); write(*,'(A)') P//"v_par    : "//trim(physics_pc_sf_vpar)//", b0 (after pair_w)"
        case (SF_VPAR_B1); write(*,'(A)') P//"v_par    : "//trim(physics_pc_sf_vpar)//", b1 (in pair_w)"
        end select
      endif
      select case (corr)
      case (0); write(*,'(A)') P//"corrector: 0, Eq. (16), second pair_psi solve"
      case (1); write(*,'(A)') P//"corrector: 1, Eq. (17), (psi, j) from pair_w, B_16 T* on its psi row"
      case (2); write(*,'(A)') P//"corrector: 2, Eq. (17), (psi, j) from pair_w, B_16 T* dropped"
      case (3); write(*,'(A)') P//"corrector: 3, Eq. (17) strict, rho / T by their mass (opz B_44)^-1"
      case (4); write(*,'(A)') P//"corrector: 4, consistent, B_51 psi_w, B_61 psi_w + B_63 j_w"
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

  !> Build the preconditioner from the assembled Jacobian A_full, at every PC
  !! rebuild. The operators' patterns are fixed by the first build; later
  !! builds refill their values, so MUMPS and the GMG keep their setups.
  subroutine sf_build(A_full, comm, my_id)
    use phys_module, only: physics_pc_sf_rtol, use_pfafem
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr
    logical :: first
    real*8  :: rtol
    integer :: k, nno
    Mat :: ws(WS_N)
    integer :: comm_s

    call sf_init(my_id)
    first  = sf_first
    comm_g = comm
    rtol   = physics_pc_sf_rtol

    if (first) then
      if (.not. g_ctx%w_force_ready) call build_fatal("W_force was never assembled (petsc_assemble_pc_matrices)")
      if (with_v .and. vstage /= SF_VPAR_V0 .and. .not. g_ctx%w_v_ready) &
        call build_fatal("v_par's W_vv / W_vu / W_uv were never assembled (stage b0 / b1)")
      call sfo_layout(A_full, comm, famode, harm_all, my_id)
      call define_operators()
    endif
    comm_s = sfo_comm

    ! W, and v_par's three where assembled (W stands in, unread, otherwise)
    ws = g_ctx%W_force
    if (g_ctx%w_v_ready) then
      ws(WS_VV) = g_ctx%W_vv; ws(WS_VU) = g_ctx%W_vu; ws(WS_UV) = g_ctx%W_uv
    endif
    call PetscLogEventBegin(pcev_extract, ierr)
    call build_or_fill(o_pj); call build_or_fill(o_w); call build_or_fill(o_rho); call build_or_fill(o_T)
    if (with_v .and. vstage /= SF_VPAR_B1) call build_or_fill(o_v)
    if (corr == 3 .and. first) call build_or_fill(o_m)    ! geometry only
    do k = 1, NCB
      if (cb_active(k)) call build_or_fill(o_cb(k))
    enddo
    call PetscLogEventEnd(pcev_extract, ierr)
    if (first) then
      call set_use_b65(o_cb(CB_65)%mat, comm_s, "B_65", use_b65)
      if (with_tite) call set_use_b65(o_cb(CB_85)%mat, comm_s, "B_85", use_b85)
      nno = 0
      do k = 1, NCB
        if (cb_active(k)) then
          if (.not. blockmv_attach(o_cb(k)%mat, int(pc_ntor))) nno = nno + 1
        endif
      enddo
      if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   SF: WARNING ", nno, &
        " coupling block(s) not AIJ, left on PETSc's matvec"
    endif
    call physics_pc_mem("SF build: operators filled", my_id)

    ! penalty rows as Dirichlet rows of the diagonal blocks, before the scaling
    ! (covers both eliminate_boundary_dofs = .f. and fix_axis_nodes = .t.)
    call full_penalty_rows(A_full)
    if (npen_full > 0) then
      call penalty_rows(slv_pj, o_pj)
      call penalty_rows(slv_w, o_w)
      call penalty_rows(slv_rho, o_rho)
      call penalty_rows(slv_T, o_T)
      if (with_v .and. vstage /= SF_VPAR_B1) call penalty_rows(slv_v, o_v)
    endif
    call block_scale(slv_pj, o_pj)
    call block_scale(slv_w, o_w)
    if (with_tite) call block_scale(slv_T, o_T)

    call PetscLogEventBegin(pcev_fact_pj, ierr)
    call sf_solver_setup(slv_pj, o_pj%mat, bk_pj, "pair_psi", comm_s, my_id, rtol, gmg_inst=2, nfields=2, &
                         smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS, &
                         pre0=SF_PJ_PRE0, post0=SF_PJ_POST0, nsmooth_c=SF_PJ_NSC, &
                         axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    call PetscLogEventEnd(pcev_fact_pj, ierr)

    call PetscLogEventBegin(pcev_fact_w, ierr)
    call sf_solver_setup(slv_w, o_w%mat, bk_w, "pair_w", comm_s, my_id, rtol, gmg_inst=1, nfields=o_w%nf, &
                         smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                         pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC, &
                         axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    call PetscLogEventEnd(pcev_fact_w, ierr)

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    if (use_pfafem) then
      call sf_solver_setup(slv_rho, o_rho%mat, bk_rho, "rho", comm_s, my_id, rtol, gmg_inst=3, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                           axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    else
      call sf_solver_setup(slv_rho, o_rho%mat, bk_rho, "rho", comm_s, my_id, rtol, gmg_inst=3, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    endif
    if (with_tite) then
      call sf_solver_setup(slv_T, o_T%mat, bk_T, "pair_T", comm_s, my_id, rtol, gmg_inst=4, nfields=2, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS_RHOT, &
                           axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    else
      call sf_solver_setup(slv_T, o_T%mat, bk_T, "T", comm_s, my_id, rtol, gmg_inst=4, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS_RHOT, &
                           axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    endif
    ! v0's B_77 is mass-dominated (rho's setup); b0's W_vv diffuses along B (T's)
    if (with_v .and. vstage == SF_VPAR_V0) then
      if (use_pfafem) then
        call sf_solver_setup(slv_v, o_v%mat, bk_v, "v_par", comm_s, my_id, rtol, gmg_inst=5, nfields=1, &
                             smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                             axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
      else
        call sf_solver_setup(slv_v, o_v%mat, bk_v, "v_par", comm_s, my_id, rtol, gmg_inst=5, nfields=1, &
                             smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
      endif
    else if (with_v .and. vstage == SF_VPAR_B0) then
      call sf_solver_setup(slv_v, o_v%mat, bk_v, "v_par", comm_s, my_id, rtol, gmg_inst=5, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                           axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    endif
    if (corr == 3 .and. first) call mass_cheb_setup(mcm, o_m%mat, comm_s, "rho / T mass B_44", int(pc_ntor))
    call PetscLogEventEnd(pcev_fact_rhot, ierr)
    call physics_pc_mem("SF build: solvers set up", my_id)

    if (.not. vecs_ready) call create_work_vectors()
    g_ctx%reduced_ready = .true.
    sf_first = .false.
    ! the families finish their setups at different times: wait here, so that
    ! the setup timer holds the slowest one rather than the first apply
    if (famode) call MPI_Barrier(comm, ierr)

  contains

    subroutine build_or_fill(op)
      type(sf_op_t), intent(inout) :: op
      if (first) then
        call sfo_build(op, A_full, ws, sf_opz(), my_id)
      else
        call sfo_fill(op, A_full, ws, sf_opz())
      endif
    end subroutine build_or_fill

    !> The operators' block layouts.
    subroutine define_operators()
      use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho
      integer, parameter :: NDIAG = 5
      integer, parameter :: ce(NCB) = [var_psi, var_psi, var_u, var_u, var_u, var_u, var_rho, var_rho, &
                                       var_T1, var_T1, var_T1, var_T1, max(var_Vpar, 1), max(var_Vpar, 1), &
                                       max(var_Vpar, 1), max(var_Vpar, 1), var_rho, var_T1, &
                                       var_psi, var_u, var_T2, var_T2, var_T2, var_T2, &
                                       max(var_Vpar, 1), var_T2]
      integer, parameter :: cv(NCB) = [var_u, var_T1, var_psi, var_zj, var_rho, var_T1, var_psi, var_u, &
                                       var_psi, var_u, var_zj, var_rho, var_psi, var_u, var_rho, var_T1, &
                                       max(var_Vpar, 1), max(var_Vpar, 1), &
                                       var_T2, var_T2, var_psi, var_u, var_zj, var_rho, &
                                       var_T2, max(var_Vpar, 1)]
      character(len=4), parameter :: cn(NCB) = ["B_12", "B_16", "B_21", "B_23", "B_25", "B_26", "B_51", "B_52", &
                                                "B_61", "B_62", "B_63", "B_65", "B_71", "B_72", "B_75", "B_76", &
                                                "B_57", "B_67", &
                                                "B_18", "B_28", "B_81", "B_82", "B_83", "B_85", &
                                                "B_78", "B_87"]
      integer :: k
      call full_op(o_pj, "pair_psi", 1, [var_psi, var_zj])
      if (with_v .and. vstage == SF_VPAR_B1) then
        ! v_par's row and column carry only u: psi_w must not enter its row
        call full_op(o_w, "pair_w (wpj + v_par)", 2, [var_u, var_w, var_psi, var_zj, var_Vpar])
        o_w%se(2, 3:5) = 0; o_w%se(3, 2) = 0; o_w%se(3, 5) = 0; o_w%se(4, 1:2) = 0; o_w%se(4, 5) = 0
        o_w%se(5, 2:4) = 0
        o_w%wid(5, 5) = WS_VV; o_w%wid(5, 1) = WS_VU; o_w%wid(1, 5) = WS_UV
      else
        call full_op(o_w, "pair_w (wpj)", 2, [var_u, var_w, var_psi, var_zj])
        o_w%se(2, 3:4) = 0; o_w%se(3, 2) = 0; o_w%se(4, 1:2) = 0
      endif
      ! the psi row's mass: opz B_33, in model180 / model183 B_11
      if (jorek_model == 183 .or. jorek_model == 180) then
        o_w%se(3, 3) = var_psi; o_w%sv(3, 3) = var_psi; o_w%scl(3, 3) = 0
      else
        o_w%se(3, 3) = var_zj;  o_w%sv(3, 3) = var_zj;  o_w%scl(3, 3) = 1
      endif
      o_w%wid(1, 1) = WS_C
      call full_op(o_rho, "B_55", 3, [var_rho])
      if (with_tite) then
        call full_op(o_T, "pair_T (Ti, Te)", 4, [var_T1, var_T2])
      else
        call full_op(o_T, "B_66", 4, [var_T1])
      endif
      if (with_v .and. vstage /= SF_VPAR_B1) then
        call full_op(o_v, "B_77", 5, [var_Vpar])
        if (vstage == SF_VPAR_B0) then
          o_v%wid(1, 1) = WS_VV; o_v%label = "B_77 + W_vv"
        endif
      endif
      if (corr == 3) call full_op(o_m, "B_44", NDIAG + NCB + 1, [var_w])
      do k = 1, NCB
        if (.not. cb_active(k)) cycle
        o_cb(k)%label = cn(k); o_cb(k)%nf = 1; o_cb(k)%tag = NDIAG + k
        o_cb(k)%se(1, 1) = ce(k); o_cb(k)%sv(1, 1) = cv(k)
      enddo
      if (with_v .and. vstage == SF_VPAR_B0) then
        o_cb(CB_72)%wid(1, 1) = WS_VU; o_cb(CB_72)%label = "B_72 + W_vu"
      endif
    end subroutine define_operators

    subroutine full_op(op, label, tag, vars)
      type(sf_op_t), intent(inout) :: op
      character(len=*), intent(in) :: label
      integer, intent(in) :: tag, vars(:)
      integer :: a
      op%label = label; op%nf = size(vars); op%tag = tag
      do a = 1, size(vars)
        op%se(a, 1:size(vars)) = vars(a); op%sv(1:size(vars), a) = vars(a)
      enddo
    end subroutine full_op

    subroutine block_scale(slv, op)
      type(block_solver_t), intent(inout) :: slv
      type(sf_op_t), intent(inout) :: op
      if (slv%scaled) call VecDestroy(slv%dscale, ierr)
      call make_field_block_scale(op%mat, spread(sfo_n1, 1, op%nf), slv%dscale, comm_s, my_id, op%label)
      slv%scaled = .true.
    end subroutine block_scale

    subroutine penalty_rows(slv, op)
      type(block_solver_t), intent(inout) :: slv
      type(sf_op_t), intent(inout) :: op
      call decouple_penalty_rows(op%mat, spread(sfo_n1, 1, op%nf), slv%pfac, slv%penal, comm_s, my_id, op%label)
    end subroutine penalty_rows

    !> The full Jacobian's penalty rows, for the back-substitution closing
    !! sf_apply: |a_ii| > 1e6 max |a_ij| over the row's own variable (the
    !! Mach-1 rows also hold zbig couplings to other variables).
    subroutine full_penalty_rows(A)
      use mod_parameters, only: n_tor
      Mat :: A
      PetscInt :: rs, re, row, ncols, bs
      PetscInt, pointer :: cols(:)
      PetscScalar, pointer :: vals(:), wp(:)
      integer :: jc, n, iv, mpierr
      real*8 :: dg, off
      integer, allocatable :: tl(:)
      real*8, allocatable :: td(:)
      a_full_s = A
      call MatGetBlockSize(A, bs, ierr)
      call MatGetOwnershipRange(A, rs, re, ierr)
      if (.not. pen_w_made) then
        call MatCreateVecs(A, pen_w, PETSC_NULL_VEC, ierr)
        pen_w_made = .true.
      endif
      call MatGetDiagonal(A, pen_w, ierr)
      call VecGetArrayRead(pen_w, wp, ierr)
      allocate(tl(re - rs), td(re - rs))
      n = 0
      do row = rs, re - 1
        dg = wp(int(row - rs) + 1)
        if (abs(dg) < 1.d7) cycle
        iv = int(mod(row, bs)) / n_tor
        call MatGetRow(A, row, ncols, cols, vals, ierr)
        off = 0.d0
        do jc = 1, int(ncols)
          if (cols(jc) /= row .and. int(mod(cols(jc), bs)) / n_tor == iv) then
            off = max(off, abs(vals(jc)))
          endif
        enddo
        call MatRestoreRow(A, row, ncols, cols, vals, ierr)
        if (abs(dg) > 1.d6 * off) then
          n = n + 1; tl(n) = int(row - rs) + 1; td(n) = dg
        endif
      enddo
      call VecRestoreArrayRead(pen_w, wp, ierr)
      pen_loc = tl(1:n); pen_d = td(1:n)
      call MPI_Allreduce(n, npen_full, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
      if (npen_full > 0) then
        if (my_id == 0 .and. first) write(*,'(A,I0,A)') "[Physics PC]   full Jacobian: ", npen_full, &
          " penalty rows, back-substituted after every sweep"
      endif
    end subroutine full_penalty_rows

    subroutine create_work_vectors()
      integer :: k_
      call MatCreateVecs(o_pj%mat, rhs_PJ, sol_PJ, ierr)
      call MatCreateVecs(o_w%mat, rhs_W, sol_W, ierr)
      if (with_tite) call MatCreateVecs(o_T%mat, rhs_T, sol_T, ierr)
      call MatCreateVecs(o_rho%mat, sv_x(1), PETSC_NULL_VEC, ierr)
      do k_ = 2, NSV
        call VecDuplicate(sv_x(1), sv_x(k_), ierr)
      enddo
      do k_ = 1, NSV
        call VecDuplicate(sv_x(1), sv_y(k_), ierr)
      enddo
      call VecDuplicate(sv_x(1), w3, ierr)
      call VecDuplicate(sv_x(1), w4, ierr)
      call VecDuplicate(sv_x(1), w5, ierr)
      call VecDuplicate(sv_x(1), w6, ierr)
      call VecDuplicate(sv_x(1), w_scr, ierr)
      call VecDuplicate(sv_x(1), t_rho, ierr)
      call VecDuplicate(sv_x(1), t_T, ierr)
      call VecDuplicate(sv_x(1), t_Te, ierr)
      call VecDuplicate(sv_x(1), wv1, ierr)
      call VecDuplicate(sv_x(1), wv2, ierr)
      call VecDuplicate(sv_x(1), zv, ierr)
      call VecSet(zv, 0.0d0, ierr)
      vecs_ready = .true.
    end subroutine create_work_vectors

    subroutine build_fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: ", msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine build_fatal

  end subroutine sf_build

  !> flag = matrix a has a nonzero entry anywhere on comm.
  subroutine set_use_b65(a, comm, nm, flag)
    Mat :: a
    integer, intent(in) :: comm
    character(len=*), intent(in) :: nm
    logical, intent(out) :: flag
    real*8 :: nrm
    integer :: mpierr, me
    PetscErrorCode :: ierr
    call MatNorm(a, NORM_FROBENIUS, nrm, ierr)
    nrm = nrm**2
    call MPI_Allreduce(MPI_IN_PLACE, nrm, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, mpierr)
    flag = (nrm > 0.d0)
    call MPI_Comm_rank(comm, me, mpierr)
    if (me == 0 .and. flag) write(*,'(A,A,A,ES10.3,A)') "[Physics PC]   SF: |", trim(nm), "|_F = ", sqrt(nrm), &
      " (max over ranks), applied in the rho -> T sweep"
  end subroutine set_use_b65

  !> y = P^-1 x, the block-LDU sweep:
  !!   1. predictor: (psi*, j*) by pair_psi; rho*, T* (or coupled Ti*, Te*) against them
  !!   2. pair_w (u, omega, psi_w, j_w) on the rest
  !!   3. corrector: psi, j by a second pair_psi solve (corrector 0) or as
  !!      psi* + psi_w, j* + j_w (1-4); rho, T / (Ti, Te) by block solves on B_52 u,
  !!      B_62 u / B_82 u (+ pair_w's psi_w, j_w with corrector 4) or by their mass
  !!      inverse (3).
  !! v_par: stage b0 between 2 and 3, b1 inside 2, v0 after 3.
  subroutine sf_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T, x_Te, x_v
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T, y_Te, y_v
    logical :: vb0, vb1
    real*8 :: t0, t1

    ierr = 0
    call PetscLogEventBegin(pcev_apply, ierr)
    t0 = MPI_Wtime()
    call sfo_vars_in(x, sv_x)
    t_scat = t_scat + (MPI_Wtime() - t0)
    x_psi = sv_x(var_psi); x_u   = sv_x(var_u);   x_j = sv_x(var_zj)
    x_w   = sv_x(var_w);   x_rho = sv_x(var_rho); x_T = sv_x(var_T1)
    y_psi = sv_y(var_psi); y_u   = sv_y(var_u);   y_j = sv_y(var_zj)
    y_w   = sv_y(var_w);   y_rho = sv_y(var_rho); y_T = sv_y(var_T1)
    x_Te = zv; y_Te = zv
    if (with_tite) then
      x_Te = sv_x(var_T2); y_Te = sv_y(var_T2)
    endif
    x_v = zv; y_v = zv                                   ! placeholders without v_par, never read
    vb0 = .false.; vb1 = .false.
    if (with_v) then
      x_v = sv_x(max(var_Vpar, 1)); y_v = sv_y(max(var_Vpar, 1))
      vb0 = (vstage == SF_VPAR_B0)
      vb1 = (vstage == SF_VPAR_B1)
    endif

    ! 1. (psi*, j*) = pair_psi^-1 (x_psi, x_j); x_j, not 0, makes j* the mass
    ! back-substitution M_j^-1 (x_j - B_31 psi*)
    call sf_split_parts(rhs_PJ, [x_psi, x_j], .true.)
    call solve(slv_pj, rhs_PJ, sol_PJ, pcev_solve_pj)
    call sf_split_parts(sol_PJ, [y_psi, y_j], .false.)

    ! rho* = B_55^-1 (x_rho - B_51 psi*)
    call MatMult(o_cb(CB_51)%mat, y_psi, w3, ierr)
    call VecWAXPY(w4, -1.0d0, w3, x_rho, ierr)
    call solve(slv_rho, w4, t_rho, pcev_solve_rhot)

    ! T* = B_66^-1 (x_T - B_61 psi* - B_63 j* - B_65 rho*), or coupled pair_T (Ti*, Te*)
    call MatMult(o_cb(CB_61)%mat, y_psi, w3, ierr)
    call VecWAXPY(w4, -1.0d0, w3, x_T, ierr)
    call sub_mult(CB_63, y_j, w4)
    if (use_b65) call sub_mult(CB_65, t_rho, w4)
    if (with_tite) then
      call MatMult(o_cb(CB_81)%mat, y_psi, w3, ierr)
      call VecWAXPY(w6, -1.0d0, w3, x_Te, ierr)
      call sub_mult(CB_83, y_j, w6)
      if (use_b85) call sub_mult(CB_85, t_rho, w6)
      call sf_split_parts(rhs_T, [w4, w6], .true.)
      call solve(slv_T, rhs_T, sol_T, pcev_solve_rhot)
      call sf_split_parts(sol_T, [t_T, t_Te], .false.)
    else
      call solve(slv_T, w4, t_T, pcev_solve_rhot)
    endif

    ! 2. pair_w on (x_u - B_21 psi* - B_23 j* - B_25 rho* - B_26 T* [- B_28 Te*], x_w,
    ! -B_16 T* [- B_18 Te*] or 0, 0): the omega row has no lower coupling, and x_psi, x_j
    ! went to the predictor
    call VecCopy(x_u, w5, ierr)
    call sub_mult(CB_21, y_psi, w5)
    call sub_mult(CB_23, y_j, w5)
    call sub_mult(CB_25, t_rho, w5)
    call sub_mult(CB_26, t_T, w5)
    if (with_tite) call sub_mult(CB_28, t_Te, w5)
    if (corr == 1 .or. corr == 4) then
      call MatMult(o_cb(CB_16)%mat, t_T, w4, ierr)
      call VecScale(w4, -1.0d0, ierr)
      if (with_tite) call sub_mult(CB_18, t_Te, w4)
    else
      call VecSet(w4, 0.0d0, ierr)
    endif
    if (vb1) then                                        ! x_V - B_71 psi* - B_75 rho* - B_76 T* [- B_78 Te*]
      call MatMult(o_cb(CB_71)%mat, y_psi, wv1, ierr)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      call sub_mult(CB_75, t_rho, wv2)
      call sub_mult(CB_76, t_T, wv2)
      if (with_tite) call sub_mult(CB_78, t_Te, wv2)
      call sf_split_parts(rhs_W, [w5, x_w, w4, zv, wv2], .true.)
    else
      call sf_split_parts(rhs_W, [w5, x_w, w4, zv], .true.)
    endif
    call solve(slv_w, rhs_W, sol_W, pcev_solve_w)
    ! w3, w4 = psi_w, j_w (dropped by corrector 0)
    if (vb1) then
      call sf_split_parts(sol_W, [y_u, y_w, w3, w4, y_v], .false.)
    else
      call sf_split_parts(sol_W, [y_u, y_w, w3, w4], .false.)
    endif

    ! v_par b0: (B_77 + W_vv) V = x_V - B_71 psi* - B_75 rho* - B_76 T* [- B_78 Te*] - (B_72 + W_vu) u
    if (vb0) then
      call MatMult(o_cb(CB_71)%mat, y_psi, wv1, ierr)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      call sub_mult(CB_75, t_rho, wv2)
      call sub_mult(CB_76, t_T, wv2)
      if (with_tite) call sub_mult(CB_78, t_Te, wv2)
      call sub_mult(CB_72, y_u, wv2)
      call solve(slv_v, wv2, y_v, pcev_solve_rhot)
    endif

    ! corrector 4: pair_w's change of (psi, j) in the rho / T corrections,
    ! held in y_rho / y_T / y_Te until step 3
    if (corr == 4) then
      call MatMult(o_cb(CB_51)%mat, w3, y_rho, ierr)
      call MatMult(o_cb(CB_61)%mat, w3, y_T, ierr)
      call add_mult(CB_63, w4, y_T)
      if (with_tite) then
        call MatMult(o_cb(CB_81)%mat, w3, y_Te, ierr)
        call add_mult(CB_83, w4, y_Te)
      endif
    endif

    ! 3. psi, j
    if (corr /= 0) then
      call VecAXPY(y_psi, 1.0d0, w3, ierr)
      call VecAXPY(y_j,   1.0d0, w4, ierr)
    else
      ! pair_psi (dpsi, dj) = (B_12 u + B_16 T* [+ B_18 Te*], 0): the j row of U is zero
      call MatMult(o_cb(CB_12)%mat, y_u, w3, ierr)
      call add_mult(CB_16, t_T, w3)
      if (with_tite) call add_mult(CB_18, t_Te, w3)
      call VecZeroEntries(w4, ierr)
      call sf_split_parts(rhs_PJ, [w3, w4], .true.)
      call solve(slv_pj, rhs_PJ, sol_PJ, pcev_solve_pj)
      call sf_split_parts(sol_PJ, [w3, w4], .false.)
      call VecAXPY(y_psi, -1.0d0, w3, ierr)
      call VecAXPY(y_j,   -1.0d0, w4, ierr)
    endif

    ! 3. rho, T / (Ti, Te) (only u enters: the omega column of U is zero)
    if (corr == 3) then
      ! (opz B_44)^-1, no block solve (B_65 vanishes in model199)
      call MatMult(o_cb(CB_52)%mat, y_u, w3, ierr)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_rho, -1.0d0 / sf_opz(), w5, t_rho, ierr)
      call MatMult(o_cb(CB_62)%mat, y_u, w3, ierr)
      call mass_cheb_solve(mcm, w3, w5)
      call VecWAXPY(y_T, -1.0d0 / sf_opz(), w5, t_T, ierr)
    else
      ! [[B_55, 0], [B_65, B_66]] (drho, dT) = (B_52 u [+ B_51 psi_w] [+ B_57 V],
      !                                         B_62 u [+ B_61 psi_w + B_63 j_w] [+ B_67 V])
      call MatMult(o_cb(CB_52)%mat, y_u, w3, ierr)
      if (corr == 4) call VecAXPY(w3, 1.0d0, y_rho, ierr)
      if (vb0 .or. vb1) call add_mult(CB_57, y_v, w3)
      call solve(slv_rho, w3, w5, pcev_solve_rhot)
      call VecWAXPY(y_rho, -1.0d0, w5, t_rho, ierr)

      call MatMult(o_cb(CB_62)%mat, y_u, w3, ierr)
      if (corr == 4) call VecAXPY(w3, 1.0d0, y_T, ierr)
      if (vb0 .or. vb1) call add_mult(CB_67, y_v, w3)
      if (use_b65) call sub_mult(CB_65, w5, w3)
      if (with_tite) then
        call MatMult(o_cb(CB_82)%mat, y_u, w6, ierr)
        if (corr == 4) call VecAXPY(w6, 1.0d0, y_Te, ierr)
        if (vb0 .or. vb1) call add_mult(CB_87, y_v, w6)
        if (use_b85) call sub_mult(CB_85, w5, w6)
        call sf_split_parts(rhs_T, [w3, w6], .true.)
        call solve(slv_T, rhs_T, sol_T, pcev_solve_rhot)
        call sf_split_parts(sol_T, [w4, w6], .false.)
        call VecWAXPY(y_T,  -1.0d0, w4, t_T,  ierr)
        call VecWAXPY(y_Te, -1.0d0, w6, t_Te, ierr)
      else
        call solve(slv_T, w3, w5, pcev_solve_rhot)
        call VecWAXPY(y_T, -1.0d0, w5, t_T, ierr)
      endif
    endif

    ! v_par v0: B_77 V = x_V - B_71 psi - B_72 u - B_75 rho - B_76 T [- B_78 Te]
    if (with_v .and. .not. (vb0 .or. vb1)) then
      call MatMult(o_cb(CB_71)%mat, y_psi, wv1, ierr)
      call VecWAXPY(wv2, -1.0d0, wv1, x_v, ierr)
      call sub_mult(CB_72, y_u, wv2)
      call sub_mult(CB_75, y_rho, wv2)
      call sub_mult(CB_76, y_T, wv2)
      if (with_tite) call sub_mult(CB_78, y_Te, wv2)
      call solve(slv_v, wv2, y_v, pcev_solve_rhot)
    endif

    t1 = MPI_Wtime()
    call sfo_vars_out(sv_y, y)
    t_scat  = t_scat + (MPI_Wtime() - t1)
    t_apply = t_apply + (MPI_Wtime() - t0)
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

    !> r -= B_k v
    subroutine sub_mult(k, v, r)
      integer, intent(in) :: k
      Vec :: v, r
      call MatMult(o_cb(k)%mat, v, w_scr, ierr)
      call VecAXPY(r, -1.0d0, w_scr, ierr)
    end subroutine sub_mult

    !> r += B_k v
    subroutine add_mult(k, v, r)
      integer, intent(in) :: k
      Vec :: v, r
      call MatMult(o_cb(k)%mat, v, w_scr, ierr)
      call VecAXPY(r, 1.0d0, w_scr, ierr)
    end subroutine add_mult

    !> The penalty rows' x_b = (r_b - sum_j a_bj y_j) / a_bb with the sweep's
    !! y: their residual is zbig times x_b's error, which the decoupled block
    !! rows leave at the size of the whole outer residual. Changes the other
    !! rows' residual by O(1 / zbig).
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

  end subroutine sf_apply

  !> Refresh the full-Jacobian handle and penalty-row diagonals on solve_only
  !! calls (e.g. when solve_newton hands a fresh temporary Mat to KSPSolve
  !! without rebuilding the preconditioner).
  subroutine sf_refresh_operator(A_full)
    Mat, intent(in) :: A_full
    PetscScalar, pointer :: wp(:)
    PetscErrorCode :: ierr
    integer :: q
    if (.not. g_ctx%reduced_ready) return
    a_full_s = A_full
    if (npen_full > 0 .and. pen_w_made .and. allocated(pen_loc)) then
      call MatGetDiagonal(A_full, pen_w, ierr)
      call VecGetArrayRead(pen_w, wp, ierr)
      do q = 1, size(pen_loc)
        if (abs(wp(pen_loc(q))) > 0.d0) pen_d(q) = wp(pen_loc(q))
      enddo
      call VecRestoreArrayRead(pen_w, wp, ierr)
    endif
  end subroutine sf_refresh_operator

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
      if (with_v .and. vstage /= SF_VPAR_B1) call sf_solver_report(slv_v, my_id)
    endif
    call sf_solver_reset_counters(slv_pj)
    call sf_solver_reset_counters(slv_w)
    call sf_solver_reset_counters(slv_rho)
    call sf_solver_reset_counters(slv_T)
    call sf_solver_reset_counters(slv_v)
    t_apply = 0.d0;  t_scat = 0.d0
  end subroutine sf_report

  !> Mode split: per family, each block's mean inner iterations and solve
  !! time, the rest of the sweep and the scatters (max over the family's
  !! ranks). The scatter back waits for every family, so the critical family
  !! is the one with the smallest scatter time.
  subroutine family_report(my_id)
    use mod_petsc_pc_sf_ops, only: sfo_fam, sfo_nfam
    integer, intent(in) :: my_id
    integer, parameter :: NV = 14
    real*8 :: v(NV), vf(NV)
    real*8, allocatable :: g(:, :)
    integer :: np, r, f, nrk, mpierr
    v(1:4)   = [its_mean(slv_pj), its_mean(slv_w), its_mean(slv_rho), its_mean(slv_T)]
    v(5:8)   = [slv_pj%t_sum, slv_w%t_sum, slv_rho%t_sum, slv_T%t_sum]
    v(13:14) = [its_mean(slv_v), slv_v%t_sum]
    v(9)     = t_apply - t_scat - sum(v(5:8)) - v(14)
    v(10)    = t_scat
    v(11)    = dble(slv_pj%nfail + slv_w%nfail + slv_rho%nfail + slv_T%nfail + slv_v%nfail)
    v(12)    = dble(sfo_fam)
    call MPI_Allreduce(v, vf, NV, MPI_DOUBLE_PRECISION, MPI_MAX, sfo_comm, mpierr)
    call MPI_Comm_size(comm_g, np, mpierr)
    allocate(g(NV, 0:merge(np - 1, 0, my_id == 0)))
    call MPI_Gather(vf, NV, MPI_DOUBLE_PRECISION, g, NV, MPI_DOUBLE_PRECISION, 0, comm_g, mpierr)
    if (my_id /= 0) return
    write(*,'(A)') "[Physics PC]   SF mode split per family: inner its / solve s (max over its ranks), "// &
      "rest of the sweep, scatters incl. the wait for the slowest family"
    do f = 1, sfo_nfam
      nrk = count(nint(g(12, :)) == f)
      r = findloc(nint(g(12, :)), f, 1) - 1
      write(*,'(A,I0,A,I0,A,4(A,F5.2,A,F7.3),A,F7.3,A,F7.3,A)', advance="no") &
        "[Physics PC]     family ", f, " (", nrk, " rk):", &
        " pair_psi ", g(1, r), " /", g(5, r), "  pair_w ", g(2, r), " /", g(6, r), &
        "  rho ", g(3, r), " /", g(7, r), "  T ", g(4, r), " /", g(8, r), &
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
