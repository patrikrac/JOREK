!> The Alfven-block physics-based preconditioner (use_physics_pc,
!! physics_pc_method = "vs_alfven"): a block lower-triangular factorization
!! of the Jacobian with the Alfven group a = (u, omega, psi, j) first and the
!! fields f = (rho, T) after it, beside the split-field sweep of
!! mod_petsc_pc_sf and the vorticity-space Schur PC of mod_petsc_pc_vs. The
!! name vs_alfven groups it with the VS variants; no vorticity-space Schur
!! complement is formed. With B_ij = A(equation i, variable j), 1 psi, 2 u,
!! 3 j, 4 omega, 5 rho, 6 T:
!!
!!   A = [[K, C], [L, D]],    K = A(a, a),   D = [[B_55, 0], [B_65, B_66]]
!!   P = [[K_W, 0], [L, D]]
!!
!!   K_W = [[B_22 + W, B_24, B_21, B_23],     (u, omega, psi, j)
!!          [B_42,     B_44, 0,    0   ],
!!          [B_12,     0,    B_11, B_13],
!!          [0,        0,    B_31, B_33]]
!!
!! W is the SF sweep's force operator (Chacon Eqs. 17-19, as mod_petsc_pc_sf):
!! it stands in for the Schur term -C D^-1 L of rho and T. K_W is the SF
!! sweep's pair_w with the psi row's own B_11 in the place of opz B_33. Its
!! zero blocks are zeros of the Jacobian in model199 and model600 (their
!! element matrices assign none of B_14, B_32, B_34, B_41, B_43), so K_W is K
!! plus W. With W = -C D^-1 L exactly, A P^-1 = [[I, C D^-1], [0, I]] has all
!! eigenvalues 1 and a minimal polynomial of degree 2: no predictor and no
!! corrector, the outer count is set by W and by the inner solves. With
!! physics_pc_sf_harm_couple = .f. K_W, W and L are |n|-diagonal, so this
!! statement holds for the coupled layout only. The upper part C (B_15,
!! B_16 = eta'(T), B_25, B_26) does not enter the application.
!!
!! L and D as used: B_51, B_52 and B_61, B_62, B_63 always; B_64 (model600,
!! viscous heating) and B_65 where they are nonzero; B_53, B_54 vanish in
!! both models. B_56 (model600: tauIC, the sheath term of the open boundary)
!! would make D non-triangular: it is left out, with a warning at the first
!! build.
!!
!! One application y of the preconditioner to x:
!!   1. K_W (u, omega, psi, j) = (x_u, x_omega, x_psi, x_j)
!!   2. B_55 rho = x_rho - B_51 psi - B_52 u
!!   3. B_66 T   = x_T - B_61 psi - B_62 u - B_63 j - B_64 omega - B_65 rho
!!   4. the full Jacobian's penalty rows by back-substitution
!! With eliminated boundary rows there is no step 4: K_W, B_55 and B_66 hold
!! the Jacobian's own diagonal rows there and W's rows vanish, so those rows
!! are solved exactly. Step 4's product holds the penalty rows alone
!! (full_penalty_rows' compact option): in mode split it is the one exchange
!! on the full system's layout inside an application besides the gather and
!! the scatter, and it is of boundary size.
!!
!! V_PAR. model600 built with v_par (n_var 7, 7 = V). Its V row couples to
!! psi, u, rho, T and V, V's column enters the u, rho and T rows: B_73, B_74,
!! B_17, B_37 and B_47 are zeros of its element matrix and of its boundary
!! rows. B_72 and B_27 are the conservative-form terms, proportional to the
!! flow (V0, u0): zero on a state at rest, so they are never dropped by a
!! test of their values. W_vv, W_vu and W_uv are the sound-wave terms of
!! model600's mod_pc_elt_matrix_force_fft (stages b0, b1, shared with the SF
!! sweep), the pressure channel of the Schur complement of (rho, T)
!! eliminated by their mass: W_vv the V - V part, W_uv the V column of the u
!! row, W_vu the u column of the V row. A block lower-triangular P is exact
!! when each diagonal block is the Schur complement with respect to the
!! unknowns solved after it and each lower coupling is the one after that
!! elimination.
!! W_vu's convention: it leaves out the advection half R [p0, u] of u's
!! pressure response, which for an axisymmetric p0(psi0) cancels against
!! the psi channel -B_71 (opz B_33)^-1 B_12 u. W_vu is thus u's coupling
!! after the elimination of (rho, T) and of the u-driven part of psi, and
!! the lower coupling consistent with it is
!!   x_V - B_71 psi' - (B_72 + W_vu) u,   psi' = psi + (opz B_33)^-1 B_12 u,
!! psi' the flux change that u does not drive; B_71 on the whole psi would
!! count the psi channel twice. psi' is left out (B_71 not applied): psi'
!! by one Chebyshev B_33 solve per application was no better on
!! inxflow_shaped_vpar_pcbench (b0, dt 1 to 1000), and the form without it
!! is the cheapest consistent one. The SF sweep zeroes B_71 in its b1 block
!! for the same convention; it applies B_71 to its predictor's psi* alone.
!! physics_pc_sf_vpar_stage, shared with the SF sweep, places V:
!!   "b0": a, V, rho, T. Between steps 1 and 2
!!          (B_77 + W_vv) V = x_V - (B_72 + W_vu) u,
!!        then -B_57 V in step 2 and -B_67 V in step 3. B_77 + W_vv stands
!!        in for V's Schur complement after rho and T (the sound wave as a
!!        diffusion along B), B_72 + W_vu for u's coupling into the V row
!!        after that elimination. Left out: B_71 psi', V's feedback on the
!!        Alfven group, (B_27 + W_uv) (B_77 + W_vv)^-1 (B_72 + W_vu), and the
!!        upper couplings B_75, B_76, B_27.
!!   "v0": a, rho, T, V. Steps 2 and 3 without V, then
!!          B_77 V = x_V - B_71 psi - B_72 u - B_75 rho - B_76 T,
!!        the V row exact (no W term, so B_71 on the whole psi). Left out:
!!        B_57, B_67, B_27, so the sound wave is not in P.
!!   "b1": (a, V), rho, T. Step 1 solves the five fields together,
!!          K_W5 = [[B_22 + W,    B_24, B_21, B_23, B_27 + W_uv],   (u, omega, psi, j, V)
!!                  [B_42,        B_44, 0,    0,    0          ],
!!                  [B_12,        0,    B_11, B_13, 0          ],
!!                  [0,           0,    B_31, B_33, 0          ],
!!                  [B_72 + W_vu, 0,    0,    0,    B_77 + W_vv]],
!!        then steps 2 and 3 as in b0. Its zero blocks are zeros of the
!!        Jacobian except B_71, left out as in b0; the upper couplings B_75
!!        and B_76 are left out.
!! V's penalty rows are decoupled in its block like the others'. The Mach-1
!! sheath rows of V (bcs%mach1: a penalty diagonal with penalty-size entries
!! in the T and u columns) are found by full_penalty_rows from V's own
!! columns and back-substituted in step 4 as whole rows. Before step 4 they
!! are approximate in b0 and b1, which solve them without their T column (T
!! comes after V), and B_57 V, B_67 V carry that into rho and T. Limit of
!! b1: the u column's penalty-size entry is inside K_W5, where
!! decouple_penalty_rows' test of the whole row can miss those rows.
!!
!! Backends: K_W (K_W5) physics_pc_sf_pair_w, B_55 physics_pc_sf_rho, B_66
!! physics_pc_sf_T, V's own block physics_pc_sf_vpar; physics_pc_sf_pair_psi
!! is not used.
!!
!! The Alfven group's pieces of x and y (with V in b1) are host-array views
!! (VecCreateMPIWithArray) into the storage of K_W's packed right-hand side
!! and solution: the gather of x fills step 1's right-hand side and step 1
!! writes y's pieces. Step 1 consumes x's pieces (the right-hand side is
!! scaled in place), the rho, T and V solves consume x_rho, x_T and x_V, in
!! which the steps before them accumulate their right-hand sides. A device
!! Vec type would break the aliasing.
module mod_petsc_pc_vs_alfven
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var, var_psi, var_u, var_zj, var_w, var_rho, var_T, var_Vpar
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, pcev_extract, pcev_fact_w, pcev_fact_rhot, &
       pcev_solve_w, pcev_solve_rhot, pcev_apply
  use mod_petsc_pc_blocks, only: penalty_rows_t, full_penalty_rows, penalty_backsub
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_ops, only: sf_op_t, sfo_layout, sfo_build, sfo_fill, sfo_vars_in, sfo_vars_out, &
       sfo_comm, sfo_define, sfo_block_used, sfo_destroy, sfo_n1
  use mod_petsc_raw_csr, only: blockmv_attach
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

  logical, save :: vsa_init_done = .false.     !< the namelist resolved and checked (vsa_init)
  logical, save :: vsa_first     = .true.      !< the next build is the first: patterns, layout, vectors
  logical, save :: famode        = .false.     !< mode split
  logical, save :: harm_all      = .true.      !< physics_pc_sf_harm_couple
  integer, save :: comm_g = MPI_COMM_NULL      !< the full system's communicator

  !> v_par's stage: SF_VPAR_V0 | _B0 | _B1, VS_NOV without v_par
  integer, parameter :: VS_NOV = -1
  integer, save :: vstage = VS_NOV
  !> V solved as a block of its own (v0, b0), not in K_W5 (b1)
  logical, save :: v_own = .false.

  type(block_solver_t), save :: slv_kw, slv_rho, slv_T, slv_v
  integer, save :: bk_kw = SF_GMG, bk_rho = SF_GMG, bk_T = SF_GMG, bk_v = SF_GMG

  !> K_W's fields in their packed order, V fifth in K_W5 (b1); nkw of them
  integer, parameter :: KW_VAR(5) = [var_u, var_w, var_psi, var_zj, var_Vpar]
  integer, save :: nkw = 4

  !> The operators K_W, B_55, B_66, V's block (B_77, or B_77 + W_vv in b0) and
  !! the coupling blocks: B_51, B_52, B_61, B_62, B_63, B_64 of L, B_65 of D,
  !! and v_par's B_57, B_67, B_71, B_72 (+ W_vu in b0), B_75, B_76. Unlike in
  !! mod_petsc_pc_sf and _vs the coupling blocks are stored NEGATED,
  !! o_cb(CB_ij) = -B_ij (negated again after every refill, see vsa_build),
  !! so that each coupling product is one in-place MatMultAdd:
  !! rhs = rhs + (-B_ij) v.
  type(sf_op_t), save :: o_kw, o_rho, o_T, o_v
  integer, parameter :: CB_51 = 1, CB_52 = 2, CB_61 = 3, CB_62 = 4, CB_63 = 5, CB_64 = 6, CB_65 = 7, &
                        CB_57 = 8, CB_67 = 9, CB_71 = 10, CB_72 = 11, CB_75 = 12, CB_76 = 13
  integer, parameter :: NCB = 13
  type(sf_op_t), save :: o_cb(NCB)
  !> the coupling blocks in use: v_par's by its stage (vsa_init); B_64 and
  !! B_65 vanish in model199 (B_64 also in model600 without viscous heating)
  !! and are kept, refilled and applied only where they do not, decided once
  !! on all ranks alike
  logical, save :: cb_on(NCB) = .true.
  !> the W sources of the operators (sf_op_t wid): W, then v_par's three
  !! where assembled
  integer, parameter :: WS_C = 1, WS_VV = 2, WS_VU = 3, WS_UV = 4, WS_N = 4
  type(penalty_rows_t), save :: pen           !< the full Jacobian's penalty rows
  !> this rank's wall time in vsa_apply (step 4 included) and in its gather /
  !! scatter since the last report (the mode split's per-family report)
  real*8, save :: t_apply = 0.d0, t_scat = 0.d0

  !> K_W's packed right-hand side and solution (fields in KW_VAR order, each
  !! sfo_n1 long) and their storage, which the Alfven group's pieces of
  !! sv_x / sv_y (by JOREK variable) share
  PetscScalar, allocatable, target, save :: kw_rhs_store(:), kw_sol_store(:)
  Vec, save :: rhs_kw, sol_kw
  integer, parameter :: NSV = merge(7, 6, var_Vpar > 0)
  !> V's piece of sv_x / sv_y: var_Vpar = NSV with v_par (vsa_init checks
  !! it); without v_par T's, never read
  integer, parameter :: SV_V = NSV
  Vec, save :: sv_x(NSV), sv_y(NSV)
  logical, save :: vecs_ready = .false.

  public :: vsa_init, vsa_build, vsa_apply, vsa_report

contains

  !> Resolve the namelist once and refuse configurations the method cannot
  !! run. Called before the first assembly of W.
  subroutine vsa_init(my_id)
    use phys_module, only: physics_pc_sf_pair_psi, physics_pc_sf_pair_w, physics_pc_sf_rho, physics_pc_sf_T, &
                           physics_pc_sf_rtol, physics_pc_sf_harm_couple, physics_pc_sf_mode_split, &
                           physics_pc_sf_vpar, physics_pc_sf_vpar_stage
    use mod_parameters, only: n_tor
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr
    character(len=*), parameter :: P = "[Physics PC]   "

    if (vsa_init_done) return

    bk_kw  = sf_backend(physics_pc_sf_pair_w, "physics_pc_sf_pair_w", my_id)
    bk_rho = sf_backend(physics_pc_sf_rho,    "physics_pc_sf_rho",    my_id)
    bk_T   = sf_backend(physics_pc_sf_T,      "physics_pc_sf_T",      my_id)

    if (jorek_model /= 199 .and. jorek_model /= 600) &
      call fatal("physics_pc_method = vs_alfven supports model199 and model600 only: the blocks of K_W and "// &
                 "the forward substitution are those of their reduced MHD (psi, u, zj, w, rho, T)")
    ! the blocks are addressed by these indices, V by NSV
    if (var_psi /= 1 .or. var_u /= 2 .or. var_zj /= 3 .or. var_w /= 4 .or. var_rho /= 5 .or. var_T /= 6 .or. &
        .not. (n_var == 6 .or. (jorek_model == 600 .and. n_var == 7 .and. var_Vpar == 7))) &
      call fatal("physics_pc_method = vs_alfven supports (psi, u, zj, w, rho, T), plus v_par on model600 only: "// &
                 "the forward substitution has no block for further variables: build without with_TiTe, "// &
                 "with_neutrals and with_impurities")
    if (mod(n_tor, 2) /= 1) &
      call fatal("physics_pc_method = vs_alfven needs n_tor odd: the operators' toroidal slots are n = 0 "// &
                 "plus cos/sin pairs")

    harm_all    = physics_pc_sf_harm_couple
    sf_gmg_harm = merge(SF_GMG_HARM_ALL, SF_GMG_HARM_PAIR, harm_all)
    famode      = physics_pc_sf_mode_split .and. .not. harm_all

    ! v_par's coupling blocks by stage: B_57, B_67 behind V (b0, b1), B_72
    ! (+ W_vu in b0) on V's own right-hand side (v0, b0), B_71, B_75, B_76
    ! with V last (v0); B_71 not in b0 and b1, W_vu holds its psi channel
    ! (module header)
    cb_on(CB_57:NCB) = .false.
    if (var_Vpar > 0) then
      vstage = sf_vpar_stage()
      if (vstage < 0) call fatal("physics_pc_sf_vpar_stage must be v0 | b0 | b1, got '"// &
                                 trim(physics_pc_sf_vpar_stage)//"'")
      v_own = (vstage /= SF_VPAR_B1)
      if (v_own) bk_v = sf_backend(physics_pc_sf_vpar, "physics_pc_sf_vpar", my_id)
      if (vstage == SF_VPAR_B1) nkw = 5
      cb_on(CB_57) = (vstage /= SF_VPAR_V0);  cb_on(CB_67) = cb_on(CB_57)
      cb_on(CB_72) = v_own
      cb_on(CB_71) = (vstage == SF_VPAR_V0);  cb_on(CB_75) = cb_on(CB_71);  cb_on(CB_76) = cb_on(CB_71)
    endif

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] =========== Alfven-block (vs_alfven) preconditioner ==========="
      if (nkw == 5) then
        write(*,'(A,A)') P//"K_W5     : ", trim(physics_pc_sf_pair_w)//" (u, omega, psi, j, v_par; B_11 exact, "// &
                         "W = SF force operator, W_uv, W_vu, W_vv), one solve per application, no predictor / corrector"
      else
        write(*,'(A,A)') P//"K_W      : ", trim(physics_pc_sf_pair_w)//" (u, omega, psi, j; B_11 exact, "// &
                         "W = SF force operator), one solve per application, no predictor / corrector"
      endif
      write(*,'(A,A)') P//"rho / T  : ", trim(physics_pc_sf_rho)//" / "//trim(physics_pc_sf_T)// &
                       " (forward substitution)"
      select case (vstage)
      case (SF_VPAR_V0)
        write(*,'(A)') P//"v_par    : "//trim(physics_pc_sf_vpar)//", v0: B_77 V last, against psi, u, rho, T; "// &
                       "left out: B_57, B_67, B_27 (no sound wave)"
      case (SF_VPAR_B0)
        write(*,'(A)') P//"v_par    : "//trim(physics_pc_sf_vpar)//", b0: (B_77 + W_vv) V after K_W, against "// &
                       "u (B_72 + W_vu), then B_57 V, B_67 V; left out: B_71 psi' (W_vu holds its psi channel), "// &
                       "V's feedback on K_W, B_75, B_76, B_27"
      case (SF_VPAR_B1)
        write(*,'(A)') P//"v_par    : b1: in K_W5 (B_71 zeroed: W_vu holds its psi channel), then B_57 V, "// &
                       "B_67 V; left out: B_75, B_76 (physics_pc_sf_vpar = "//trim(physics_pc_sf_vpar)//" has no effect)"
      end select
      write(*,'(A,A,A)') P//"pair_psi : not used (physics_pc_sf_pair_psi = ", trim(physics_pc_sf_pair_psi), &
                         " has no effect)"
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
    endif

    vsa_init_done = .true.

  contains

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') P//"FATAL: ", trim(msg)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine vsa_init

  !> Build the preconditioner from the assembled Jacobian A_full, at every PC
  !! rebuild. The operators' patterns, the layout and the vectors are fixed by
  !! the first build; later builds refill the operators, so MUMPS and the GMG
  !! keep their setups.
  subroutine vsa_build(A_full, comm, my_id)
    use phys_module, only: physics_pc_sf_rtol, eliminate_boundary_dofs
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id
    !> operator tags (MPI message tags of sfo_build / sfo_fill): K_W, B_55,
    !! B_66, V's block, then the coupling blocks, then B_56's test
    integer, parameter :: NDIAG = 4
    PetscErrorCode :: ierr
    logical :: first
    real*8  :: rtol
    integer :: k, nno, comm_s
    Mat :: ws(WS_N)

    call vsa_init(my_id)
    first  = vsa_first
    comm_g = comm
    rtol   = physics_pc_sf_rtol

    if (first) then
      if (.not. g_ctx%w_force_ready) call build_fatal("W_force was never assembled (petsc_assemble_pc_matrices)")
      if ((vstage == SF_VPAR_B0 .or. vstage == SF_VPAR_B1) .and. .not. g_ctx%w_v_ready) &
        call build_fatal("v_par's W_vv / W_vu / W_uv were never assembled (stage b0 / b1)")
      call sfo_layout(A_full, comm, famode, harm_all, my_id)
      call define_operators()
    endif
    comm_s = sfo_comm

    ! W, read by K_W's (u, u) block, and v_par's three where assembled (W
    ! stands in, unread, otherwise)
    ws = g_ctx%W_force
    if (g_ctx%w_v_ready) then
      ws(WS_VV) = g_ctx%W_vv; ws(WS_VU) = g_ctx%W_vu; ws(WS_UV) = g_ctx%W_uv
    endif
    call PetscLogEventBegin(pcev_extract, ierr)
    call build_or_fill(o_kw); call build_or_fill(o_rho); call build_or_fill(o_T)
    if (v_own) call build_or_fill(o_v)
    do k = 1, NCB
      if (.not. cb_on(k)) cycle
      call build_or_fill(o_cb(k))
      ! the coupling blocks are kept as -B_ij (vsa_apply adds them to the rhs);
      ! sfo_fill restores +B_ij, so the sign is applied after every refill
      call MatScale(o_cb(k)%mat, -1.0d0, ierr)
    enddo
    if (first) call check_b56()
    call PetscLogEventEnd(pcev_extract, ierr)
    if (first) then
      ! the optional blocks on the full system's communicator: their refill
      ! is collective there, so every family must take the same decision
      call drop_if_zero(CB_64, "B_64", "omega in the T row")
      call drop_if_zero(CB_65, "B_65", "rho in the T row")
      nno = 0
      do k = 1, NCB
        if (.not. cb_on(k)) cycle
        if (.not. blockmv_attach(o_cb(k)%mat, int(pc_ntor))) nno = nno + 1
      enddo
      if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   vs_alfven: WARNING ", nno, &
        " coupling block(s) not AIJ, left on PETSc's matvec"
    endif
    call physics_pc_mem("vs_alfven build: operators filled", my_id)

    ! Penalty rows (eliminated rows are diagonal-only rows of K_W, B_55,
    ! B_66 and V's block, solved exactly, and need nothing here): the full
    ! Jacobian's for step 4, and the blocks' as Dirichlet rows, after the
    ! refill and before the scaling.
    if (.not. eliminate_boundary_dofs) then
      call full_penalty_rows(pen, A_full, comm, my_id, first, compact=.true.)
      call sf_solver_penalty_rows(slv_kw, o_kw, comm_s, my_id)
      call sf_solver_penalty_rows(slv_rho, o_rho, comm_s, my_id)
      call sf_solver_penalty_rows(slv_T, o_T, comm_s, my_id)
      if (v_own) call sf_solver_penalty_rows(slv_v, o_v, comm_s, my_id)
    endif
    call sf_solver_block_scale(slv_kw, o_kw, comm_s, my_id)

    ! The GMG hierarchies are global state per instance, one method being
    ! active: 1 = K_W (configured as the SF sweep's pair_w, also with V as
    ! K_W5's fifth field), 3 = rho, 4 = T, 5 = V, as in the SF sweep and the
    ! VS variants. K_W's psi - u coupling runs along the field lines at large
    ! dt, hence flux-surface rings.
    call PetscLogEventBegin(pcev_fact_w, ierr)
    call sf_solver_setup(slv_kw, o_kw%mat, bk_kw, trim(o_kw%label), comm_s, my_id, rtol, gmg_inst=1, &
                         nfields=o_kw%nf, smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                         pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC, &
                         axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    call PetscLogEventEnd(pcev_fact_w, ierr)

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    call sf_solver_setup(slv_rho, o_rho%mat, bk_rho, "rho", comm_s, my_id, rtol, gmg_inst=3, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    call sf_solver_setup(slv_T, o_T%mat, bk_T, "T", comm_s, my_id, rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS_RHOT)
    ! as the SF sweep's: v0's B_77 is mass-dominated (rho's setup), b0's
    ! W_vv diffuses along B (T's rings, the pairs' iteration cap)
    if (vstage == SF_VPAR_V0) then
      call sf_solver_setup(slv_v, o_v%mat, bk_v, "v_par", comm_s, my_id, rtol, gmg_inst=5, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    else if (vstage == SF_VPAR_B0) then
      call sf_solver_setup(slv_v, o_v%mat, bk_v, "v_par", comm_s, my_id, rtol, gmg_inst=5, nfields=1, &
                           smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS)
    endif
    call PetscLogEventEnd(pcev_fact_rhot, ierr)
    call physics_pc_mem("vs_alfven build: solvers set up", my_id)

    if (.not. vecs_ready) call create_vectors()
    g_ctx%reduced_ready = .true.
    vsa_first = .false.
    ! the families finish their setups at different times: wait here, so that
    ! the setup timer holds the slowest one rather than the first apply
    if (famode) call MPI_Barrier(comm, ierr)

  contains

    !> sf_opz(): sfo_build / sfo_fill take opz for the blocks they scale; no
    !! block of these operators is scaled.
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
      integer, parameter :: ce(NCB) = [var_rho, var_rho, var_T, var_T, var_T, var_T, var_T, &
                                       var_rho, var_T, var_Vpar, var_Vpar, var_Vpar, var_Vpar]
      integer, parameter :: cv(NCB) = [var_psi, var_u, var_psi, var_u, var_zj, var_w, var_rho, &
                                       var_Vpar, var_Vpar, var_psi, var_u, var_rho, var_T]
      character(len=5), parameter :: cn(NCB) = ["-B_51", "-B_52", "-B_61", "-B_62", "-B_63", "-B_64", "-B_65", &
                                                "-B_57", "-B_67", "-B_71", "-B_72", "-B_75", "-B_76"]
      integer :: k_
      ! the zero blocks of the SF sweep's pair_w (zeros of the Jacobian too):
      ! omega's row has no psi and j, psi's row no omega, j's row no u and
      ! omega; with V (zeros of model600's Jacobian): V's column only in the
      ! u and V rows, V's row no omega and j; and V's row without B_71, whose
      ! psi channel W_vu holds
      if (nkw == 5) then
        call sfo_define(o_kw, "K_W5", 1, KW_VAR)
        o_kw%se(2, 3:5) = 0; o_kw%se(3, 2) = 0; o_kw%se(3, 5) = 0; o_kw%se(4, 1:2) = 0; o_kw%se(4, 5) = 0
        o_kw%se(5, 2:4) = 0
        o_kw%wid(1, 5) = WS_UV; o_kw%wid(5, 1) = WS_VU; o_kw%wid(5, 5) = WS_VV
      else
        call sfo_define(o_kw, "K_W", 1, KW_VAR(1:4))
        o_kw%se(2, 3:4) = 0; o_kw%se(3, 2) = 0; o_kw%se(4, 1:2) = 0
      endif
      o_kw%wid(1, 1) = WS_C                             ! + W in the (u, u) block
      call sfo_define(o_rho, "B_55", 2, [var_rho])
      call sfo_define(o_T,   "B_66", 3, [var_T])
      if (v_own) then
        call sfo_define(o_v, "B_77", NDIAG, [var_Vpar])
        if (vstage == SF_VPAR_B0) then
          o_v%wid(1, 1) = WS_VV; o_v%label = "B_77 + W_vv"
        endif
      endif
      do k_ = 1, NCB
        o_cb(k_)%label = cn(k_); o_cb(k_)%nf = 1; o_cb(k_)%tag = NDIAG + k_
        o_cb(k_)%se(1, 1) = ce(k_); o_cb(k_)%sv(1, 1) = cv(k_)
      enddo
      if (vstage == SF_VPAR_B0) then
        o_cb(CB_72)%wid(1, 1) = WS_VU; o_cb(CB_72)%label = "-(B_72 + W_vu)"
      endif
    end subroutine define_operators

    !> Keep coupling block k only if it is nonzero somewhere (decided on comm,
    !! the full system's communicator), and say so where it is.
    subroutine drop_if_zero(k, name, what)
      integer, intent(in) :: k
      character(len=*), intent(in) :: name, what
      real*8 :: nrm
      cb_on(k) = sfo_block_used(o_cb(k), comm, nrm)
      if (cb_on(k)) then
        if (my_id == 0) write(*,'(A,ES10.3,A)') "[Physics PC]   vs_alfven: |"//name//"|_F = ", nrm, &
          " (max over ranks), "//what//", applied in the T right-hand side"
      else
        call sfo_destroy(o_cb(k))
      endif
    end subroutine drop_if_zero

    !> B_56 (rho row, T column) cannot enter the forward substitution: warn
    !! if it is nonzero, then free it.
    subroutine check_b56()
      type(sf_op_t) :: o_56
      real*8 :: nrm
      o_56%label = "B_56"; o_56%nf = 1; o_56%tag = NDIAG + NCB + 1
      o_56%se(1, 1) = var_rho; o_56%sv(1, 1) = var_T
      call sfo_build(o_56, A_full, ws, sf_opz(), my_id)
      if (sfo_block_used(o_56, comm, nrm) .and. my_id == 0) write(*,'(A,ES10.3,A)') &
        "[Physics PC]   vs_alfven: WARNING B_56 (T in the rho row) is nonzero, |B_56|_F = ", nrm, &
        " (max over ranks): it would make D non-triangular and is left out"
      call sfo_destroy(o_56)
    end subroutine check_b56

    !> rhs_kw, sol_kw and the Alfven group's pieces (with V in b1) on one
    !! storage each; rho's, T's and V's own block's pieces of their own. A
    !! first gather (of a zero vector) creates the mode split's scatter here
    !! rather than in the first apply.
    subroutine create_vectors()
      PetscInt, parameter :: ONE = 1
      PetscInt :: n1, o
      integer :: f
      Vec :: x0

      n1 = sfo_n1
      ! at least one entry, so that kw_*_store(o) below exists on a rank without rows
      allocate(kw_rhs_store(max(nkw * n1, ONE)), kw_sol_store(max(nkw * n1, ONE)))
      call VecCreateMPIWithArray(sfo_comm, ONE, nkw * n1, PETSC_DETERMINE, kw_rhs_store, rhs_kw, ierr)
      call VecCreateMPIWithArray(sfo_comm, ONE, nkw * n1, PETSC_DETERMINE, kw_sol_store, sol_kw, ierr)
      do f = 1, nkw
        o = (f - 1) * n1 + 1
        call VecCreateMPIWithArray(sfo_comm, ONE, n1, PETSC_DETERMINE, kw_rhs_store(o), sv_x(KW_VAR(f)), ierr)
        call VecCreateMPIWithArray(sfo_comm, ONE, n1, PETSC_DETERMINE, kw_sol_store(o), sv_y(KW_VAR(f)), ierr)
      enddo
      call MatCreateVecs(o_rho%mat, sv_x(var_rho), sv_y(var_rho), ierr)
      call MatCreateVecs(o_T%mat, sv_x(var_T), sv_y(var_T), ierr)
      if (v_own) call MatCreateVecs(o_v%mat, sv_x(SV_V), sv_y(SV_V), ierr)
      call MatCreateVecs(A_full, x0, PETSC_NULL_VEC, ierr)
      call VecSet(x0, 0.0d0, ierr)
      call sfo_vars_in(x0, sv_x)
      call VecDestroy(x0, ierr)
      vecs_ready = .true.
    end subroutine create_vectors

    subroutine build_fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: ", msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine build_fatal

  end subroutine vsa_build

  !> y = P^-1 x, steps 1 to 4 of the module header, with V's block where
  !! its stage puts it.
  subroutine vsa_apply(x, y, ierr)
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    real*8 :: t0, t1

    ierr = 0
    call PetscLogEventBegin(pcev_apply, ierr)
    t0 = MPI_Wtime()
    call sfo_vars_in(x, sv_x)
    t_scat = t_scat + (MPI_Wtime() - t0)
    ! the gather wrote rhs_kw's storage through its pieces, which leaves
    ! rhs_kw's object state (and any norm cached on it) behind
    call PetscObjectStateIncrease(rhs_kw, ierr)

    ! 1. (u, omega, psi, j), with V in b1, into y's pieces; rhs_kw is scaled
    ! in place, its pieces of x are not read again
    call sf_solver_solve(slv_kw, rhs_kw, sol_kw, pcev_solve_w)

    ! b0: (B_77 + W_vv) V = x_V - (B_72 + W_vu) u    (o_cb = -B, accumulated
    ! in place in x_V; no B_71 psi, see the module header)
    if (vstage == SF_VPAR_B0) then
      call MatMultAdd(o_cb(CB_72)%mat, sv_y(var_u), sv_x(SV_V), sv_x(SV_V), ierr)
      call sf_solver_solve(slv_v, sv_x(SV_V), sv_y(SV_V), pcev_solve_rhot)
    endif

    ! 2. B_55 rho = x_rho - B_51 psi - B_52 u [- B_57 V]    (o_cb = -B,
    ! accumulated in place in x_rho; v3 = v2 is allowed)
    call MatMultAdd(o_cb(CB_51)%mat, sv_y(var_psi), sv_x(var_rho), sv_x(var_rho), ierr)
    call MatMultAdd(o_cb(CB_52)%mat, sv_y(var_u),   sv_x(var_rho), sv_x(var_rho), ierr)
    if (cb_on(CB_57)) call MatMultAdd(o_cb(CB_57)%mat, sv_y(SV_V), sv_x(var_rho), sv_x(var_rho), ierr)
    call sf_solver_solve(slv_rho, sv_x(var_rho), sv_y(var_rho), pcev_solve_rhot)

    ! 3. B_66 T = x_T - B_61 psi - B_62 u - B_63 j - B_64 omega - B_65 rho
    ! [- B_67 V]    (o_cb = -B, accumulated in place in x_T)
    call MatMultAdd(o_cb(CB_61)%mat, sv_y(var_psi), sv_x(var_T), sv_x(var_T), ierr)
    call MatMultAdd(o_cb(CB_62)%mat, sv_y(var_u),   sv_x(var_T), sv_x(var_T), ierr)
    call MatMultAdd(o_cb(CB_63)%mat, sv_y(var_zj),  sv_x(var_T), sv_x(var_T), ierr)
    if (cb_on(CB_64)) call MatMultAdd(o_cb(CB_64)%mat, sv_y(var_w),   sv_x(var_T), sv_x(var_T), ierr)
    if (cb_on(CB_65)) call MatMultAdd(o_cb(CB_65)%mat, sv_y(var_rho), sv_x(var_T), sv_x(var_T), ierr)
    if (cb_on(CB_67)) call MatMultAdd(o_cb(CB_67)%mat, sv_y(SV_V),    sv_x(var_T), sv_x(var_T), ierr)
    call sf_solver_solve(slv_T, sv_x(var_T), sv_y(var_T), pcev_solve_rhot)

    ! v0: B_77 V = x_V - B_71 psi - B_72 u - B_75 rho - B_76 T    (o_cb = -B,
    ! accumulated in place in x_V)
    if (vstage == SF_VPAR_V0) then
      call MatMultAdd(o_cb(CB_71)%mat, sv_y(var_psi), sv_x(SV_V), sv_x(SV_V), ierr)
      call MatMultAdd(o_cb(CB_72)%mat, sv_y(var_u),   sv_x(SV_V), sv_x(SV_V), ierr)
      call MatMultAdd(o_cb(CB_75)%mat, sv_y(var_rho), sv_x(SV_V), sv_x(SV_V), ierr)
      call MatMultAdd(o_cb(CB_76)%mat, sv_y(var_T),   sv_x(SV_V), sv_x(SV_V), ierr)
      call sf_solver_solve(slv_v, sv_x(SV_V), sv_y(SV_V), pcev_solve_rhot)
    endif

    t1 = MPI_Wtime()
    call sfo_vars_out(sv_y, y)
    t_scat = t_scat + (MPI_Wtime() - t1)
    ! 4. penalty rows of the full Jacobian (pen%n = 0 with eliminated
    ! boundary rows: nothing to do)
    if (pen%n > 0) call penalty_backsub(pen, x, y)
    t_apply = t_apply + (MPI_Wtime() - t0)
    call PetscLogEventEnd(pcev_apply, ierr)
    ierr = 0
  end subroutine vsa_apply

  !> Inner-iteration summary, one line per block in the order of the
  !! application (mode split: per family), then reset. Collective.
  subroutine vsa_report(my_id)
    integer, intent(in) :: my_id
    if (famode) then
      select case (vstage)
      case (SF_VPAR_B0)
        call sf_solver_family_report("Alfven block", [slv_kw, slv_v, slv_rho, slv_T], t_apply, t_scat, comm_g)
      case (SF_VPAR_V0)
        call sf_solver_family_report("Alfven block", [slv_kw, slv_rho, slv_T, slv_v], t_apply, t_scat, comm_g)
      case default
        call sf_solver_family_report("Alfven block", [slv_kw, slv_rho, slv_T], t_apply, t_scat, comm_g)
      end select
    else
      call sf_solver_report(slv_kw,  my_id)
      if (vstage == SF_VPAR_B0) call sf_solver_report(slv_v, my_id)
      call sf_solver_report(slv_rho, my_id)
      call sf_solver_report(slv_T,   my_id)
      if (vstage == SF_VPAR_V0) call sf_solver_report(slv_v, my_id)
    endif
    call sf_solver_reset_counters(slv_kw)
    call sf_solver_reset_counters(slv_rho)
    call sf_solver_reset_counters(slv_T)
    call sf_solver_reset_counters(slv_v)
    t_apply = 0.d0;  t_scat = 0.d0
  end subroutine vsa_report

#endif
end module mod_petsc_pc_vs_alfven
