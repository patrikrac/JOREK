!> The vorticity-space Schur (VS) physics-based preconditioner
!! (use_physics_pc, physics_pc_method = "vs_ideal" | "vs_resistive" |
!! "vs_lumped"): a block
!! factorization of the Jacobian over (psi, u, j, omega, rho, T) with u as the
!! last unknown, beside the split-field sweep of mod_petsc_pc_sf. With
!! B_ij = A(equation i, variable j), 1 psi, 2 u, 3 j, 4 omega, 5 rho, 6 T and
!! opz = 1 + zeta:
!!
!! The rows of j and omega are definitions with mass matrices on the diagonal,
!!   B_33 j + B_31 psi = r_j,   B_44 omega + B_42 u = r_omega.
!! With j and omega eliminated, and psi by its mass, the Schur complement of u
!! is approximated as a product of an operator on omega and the omega row's
!! Laplacian:
!!   S_u ~ -P B_44^-1 B_42,    P = M_rho + B_24 + L
!!   M_rho = opz (v, rho0 R^3 w)        the inertia: B_22 ~ -M_rho B_44^-1 B_42
!!   B_24                               viscosity and advection of omega
!!   L = -(1/opz) B_23 X B_12           the line bending as a parallel diffusion
!!                                      on omega, with
!!   X = B_33^-1                        (vs_ideal) or
!!   X = (B_33 - B_31 B_11^-1 B_13)^-1  (vs_resistive).
!! The terms of lower order of S_u (B_21, B_25, B_26: kink and pressure) and
!! B_16 (eta'(T)) are left out.
!!
!! P is never formed: P w = f is the Schur complement for w of the sparse
!! vorticity system, whose blocks are blocks of the Jacobian:
!!   vs_ideal       [M_rho + B_24  B_23    ] [w]   [f]
!!                  [B_12          opz B_33] [y] = [0]
!!   vs_resistive   [M_rho + B_24  B_23      0   ] [w]   [f]
!!                  [B_12          opz B_33  B_31] [y] = [0]
!!                  [0             opz B_13  B_11] [z]   [0]
!!
!! One application y of the preconditioner to x, steps 2 and 3 being
!! S_u^-1 = -B_42^-1 B_44 P^-1:
!!   1. m_j = B_33^-1 x_j, m_omega = B_44^-1 x_omega      Chebyshev mass solves
!!   2. w from the vorticity system with f = x_u - B_23 m_j - B_24 m_omega
!!   3. omega = m_omega + w;  B_42 u = x_omega - B_44 omega  (the Laplacian)
!!   4. pair_psi (psi, j) = (x_psi - B_12 u, x_j)
!!   5. B_55 rho = x_rho - B_51 psi - B_52 u
!!   6. B_66 T = x_T - B_61 psi - B_63 j - B_62 u - B_65 rho
!!   7. the full Jacobian's boundary rows by back-substitution
!! Step 3's right-hand side x_omega - B_44 omega, not -B_44 w, holds the
!! omega row of the Jacobian to the accuracy of the Laplacian solve, whatever
!! the accuracy of the mass solve.
!!
!! vs_lumped: P assembled as one sparse scalar operator, X replaced by a
!! sparse symmetric matrix from the local dual basis (mod_petsc_pc_dual), and
!! solved in the place of the vorticity system; every other step as in
!! vs_ideal:
!!   P_lumped = M_rho + B_24 - (1/opz) B_23 X B_12
!! P_lumped is symmetric and positive definite where B_24 is (no flow):
!! B_23 = -B_12' up to the quadrature, so the line bending is the congruence
!! (1/opz) B_12' X B_12 with a symmetric positive definite X. The local dual
!! basis Z of a symmetric operator S is not symmetric; X is a symmetric form
!! of it (dual_symmetrize, -vs_dual_form):
!!   plain  X = Z                   (P_lumped then not symmetric)
!!   avg    X = (Z + Z') / 2        default; Z's pattern, couplings up to three
!!                                  nodes apart in P_lumped
!!   sym    X = Z + Z' - Z' S Z     X <= S^-1; five nodes apart in P_lumped
!! The line bending (-vs_bending) selects S:
!!   ideal      S = B_33, Z once (default; X ~ B_33^-1 as in vs_ideal)
!!   resistive  S = S_s, the symmetric resistive operator, at every build:
!!                S_s = B_33 + K_eta / opz + sym(B_31 Z_11 K_hyp)
!!              with sym(A) = (A + A') / 2, Z_11 the local dual basis of
!!              B_11 (3 x 3 node patches) and
!!                M_eta = theta dt (eta_T v, w / R)            the eta part of -B_13
!!                K_hyp = -B_13 - M_eta                         its hyper-resistive part
!!                K_eta = theta dt (eta_T grad v, grad w / R)   the self-adjoint form of
!!                                                              the eta part of
!!                                                              -B_31 B_11^-1 B_13
!!              M_eta and K_eta are assembled with the model's eta_T(T0) at
!!              every PC build (petsc_assemble_pc_matrices, the model's
!!              pc_elt_matrix_eta: model199 and model600; for model600
!!              eta_T(T0, rho0) of resistivity() with the Coulomb
!!              logarithm).
!!              S = B_33 - B_31 B_11^-1 B_13 (vs_resistive) is not symmetric
!!              with eta(T) (eta Delta is not self-adjoint) and sym(S) is then
!!              not positive definite; S_s is symmetric positive definite for
!!              any eta >= 0 and equals sym(B_33 - B_31 Z_11 B_13) for a
!!              constant eta up to Z_11. Its inverse is not local at the axis,
!!              where the hyper-resistivity is much larger than the mass:
!!              Z is the dual basis with the core patch at the axis
!!              (-vs_dual_core, default true; without it X and P_lumped are
!!              indefinite on fine grids or at large dt). The rows and columns
!!              of psi's and j's constrained DOFs (B_11's and B_33's
!!              Dirichlet or penalty rows, both boundary modes) leave K_hyp
!!              and K_eta, so that S_s has B_33's boundary rows. The first
!!              build checks the assembly against the Jacobian: rank 0
!!              prints |B_13 + M_eta + K_1|_F / |M_eta|_F on the interior
!!              rows, K_1 = theta dt (eta_num_T grad v, grad w) the
!!              hyper-resistive part of -B_13 assembled once with the
!!              model's coefficient (model199: the constant eta_num;
!!              model600: eta_num_T(T0, psi_N) of hyper_resistivity()),
!!              which catches any difference between the eta_T or eta_num_T
!!              of the PC and the model's (a WARNING above 1e-8). K_hyp
!!              itself stays -B_13 - M_eta, from the Jacobian.
!!              The dual basis reads the slot-diagonal part of S_s only (a
!!              state whose T0 has harmonics couples the slots in K_eta).
!! P_lumped's boundary rows lose their columns in both boundary modes (M_rho
!! keeps them), so that the stored matrix is the symmetric one; the first
!! build reports its skew part.
!!
!! PETSc options of vs_lumped (read once, at the first assembly):
!!   -vs_bending ideal | resistive     the line bending (default ideal)
!!   -vs_dual_form plain | avg | sym   the form of X (default avg)
!!   -vs_dual_core <bool>              the core patch of the resistive bending
!!                                     (default true)
!!   -vs_pl_view <file>                P_lumped of every build, PETSc binary (the
!!                                     first build creates the file, later
!!                                     builds append)
!!   -vs_ss_view <file>                S_s of every build, likewise
!!   -vs_dual_mat_view                 every dual basis (mod_petsc_pc_dual)
!!   -vs_p_solver default | spdmg      P_lumped's solver: the backend of
!!                                     physics_pc_sf_pair_w (default: the ring
!!                                     GMG with FGMRES, or LU), or the ring-free
!!                                     SPD multigrid of mod_petsc_pc_spdmg (its
!!                                     options -vs_pmg_*)
!!
!! Boundary rows: M_rho and the Laplacian's boundary diagonal D_bc carry
!! Dirichlet rows at u's constrained DOFs (1, or the penalty with penalty
!! rows), so that the vorticity system and the Laplacian B_42 + D_bc have
!! Dirichlet rows wherever the Jacobian has them. Steps 2 and 3 do not give
!! the boundary values of u and omega the Jacobian's (u_b follows from
!! omega's and u's residuals together): step 7 sets them, for penalty and
!! for eliminated boundary rows alike.
module mod_petsc_pc_vs
#ifdef USE_PETSC
  use mpi_mod
  use mod_parameters, only: n_var
  use mod_model_settings, only: jorek_model
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_mem, physics_pc_method_id, PC_METHOD_VS_RESISTIVE, &
       PC_METHOD_VS_LUMPED, pcev_extract, pcev_fact_pj, pcev_fact_rhot, pcev_fact_vort, pcev_fact_lap, pcev_apply, &
       pcev_solve_pj, pcev_solve_rhot, pcev_solve_vort, pcev_solve_lap, pcev_solve_mass
  use mod_petsc_pc_blocks, only: penalty_rows_t, full_penalty_rows, penalty_backsub, penalty_row_mask
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_sf_ops, only: sf_op_t, sfo_layout, sfo_build, sfo_fill, sfo_vars_in, sfo_vars_out, &
       sfo_comm, sfo_define, sfo_b65_used, sfo_destroy
  use mod_petsc_raw_csr, only: blockmv_attach
  use mod_petsc_pc_mass_cheb, only: mass_cheb_t, mass_cheb_setup, mass_cheb_solve
  use mod_petsc_pc_dual, only: dual_basis, dual_symmetrize, DUAL_PLAIN, DUAL_AVG, DUAL_SYM
  implicit none
  private

  logical, save :: vs_init_done = .false.
  logical, save :: vs_first     = .true.
  logical, save :: resistive    = .false.     !< the resistive line bending
  logical, save :: lumped       = .false.     !< P as the lumped scalar operator
  integer, save :: dual_form    = DUAL_AVG    !< vs_lumped: the form of X (-vs_dual_form)
  logical, save :: res_dual     = .false.     !< vs_lumped: the resistive bending S_s (-vs_bending)
  logical, save :: res_core     = .true.      !< and its core patch at the axis (-vs_dual_core)
  character(len=512), save :: pl_view = "", ss_view = ""   !< -vs_pl_view, -vs_ss_view
  logical, save :: famode       = .false.     !< mode split
  logical, save :: harm_all     = .true.      !< physics_pc_sf_harm_couple
  integer, save :: comm_g = MPI_COMM_NULL     !< the full system's communicator
  !> this rank's wall time in vs_apply and in its gather / scatter since the
  !! last report (the mode split's per-family report)
  real*8, save  :: t_apply = 0.d0, t_scat = 0.d0

  type(block_solver_t), save :: slv_vort, slv_lap, slv_pj, slv_rho, slv_T
  integer, save :: bk_vort = SF_GMG, bk_pj = SF_GMG, bk_rho = SF_GMG, bk_T = SF_GMG
  !> vs_lumped: the backend of P_lumped's solve (-vs_p_solver): bk_vort
  !! (physics_pc_sf_pair_w, default) or SF_SPDMG (mod_petsc_pc_spdmg)
  integer, save :: bk_pl = SF_GMG

  !> B-norm error bound of the Chebyshev mass solves
  real*8, parameter :: MASS_TARGET = 1.d-2
  !> the self-check of the resistive bending: largest |B_13 + M_eta + K_1| / |M_eta| without a warning
  real*8, parameter :: ETA_CHECK_TOL = 1.d-8

  !> The operators: the vorticity system, the Laplacian B_42 + D_bc, pair_psi,
  !! B_55, B_66, the masses B_33 and B_44, and the coupling blocks o_cb(CB_ij) = B_ij.
  type(sf_op_t), save :: o_vort, o_lap, o_pj, o_rho, o_T, o_mj, o_mw
  integer, parameter :: CB_12 = 1, CB_23 = 2, CB_24 = 3, CB_51 = 4, CB_52 = 5, CB_61 = 6, CB_62 = 7, &
                        CB_63 = 8, CB_65 = 9
  integer, parameter :: NCB = 9
  type(sf_op_t), save :: o_cb(NCB)
  !> vs_lumped: X (z_j; once, or at every build with the resistive bending),
  !! the product B_23 X B_12 and P_lumped (o_pl%mat, its pattern fixed by the
  !! first build)
  Mat, save :: z_j, t_bzb
  type(sf_op_t), save :: o_pl
  !> vs_lumped with the resistive bending: o_rb(RB_ij) = B_ij; M_eta and K_eta
  !! in the operators' layout (W sources alone); Z_11, K_hyp, the product
  !! T = B_31 Z_11 K_hyp and its transpose, S_s; the masks of psi's and j's
  !! constrained DOFs (0 there, 1 elsewhere)
  integer, parameter :: RB_11 = 1, RB_13 = 2, RB_31 = 3
  integer, parameter :: NRB = 3
  type(sf_op_t), save :: o_rb(NRB), o_meta, o_keta
  Mat, save :: z_11, k_hyp, t_s, t_st, s_res
  Vec, save :: mask_psi, mask_j
  !> B_65 (T row, rho column) vanishes in model199 and is applied only where it does not
  logical, save :: use_b65 = .false.
  type(mass_cheb_t), save :: mc_j, mc_w
  type(penalty_rows_t), save :: pen           !< the full Jacobian's penalty rows

  integer, parameter :: NSV = 6
  Vec, save :: sv_x(NSV), sv_y(NSV)
  Vec, save :: rhs_V, sol_V, rhs_PJ, sol_PJ
  Vec, save :: m_j, w1, w2, w3
  Vec, save :: zv                              !< zero field
  logical, save :: vecs_ready = .false.

  public :: vs_init, vs_build, vs_apply, vs_report, vs_wants_eta

contains

  !> Resolve the namelist and the PETSc options once and refuse
  !! configurations the method cannot run. Called before the first assembly
  !! of M_rho.
  subroutine vs_init(my_id)
    use phys_module, only: physics_pc_sf_pair_psi, physics_pc_sf_pair_w, physics_pc_sf_rho, physics_pc_sf_T, &
                           physics_pc_sf_rtol, physics_pc_sf_harm_couple, physics_pc_sf_mode_split
    use mod_parameters, only: n_tor, var_T
    integer, intent(in) :: my_id
    PetscErrorCode :: ierr
    PetscBool :: is_set, flag
    character(len=16) :: form
    character(len=48) :: pname
    character(len=*), parameter :: P = "[Physics PC]   "
    character(len=*), parameter :: form_text(0:2) = [character(len=24) :: "Z", "(Z + Z') / 2", "Z + Z' - Z' S Z"]

    if (vs_init_done) return

    resistive = (physics_pc_method_id() == PC_METHOD_VS_RESISTIVE)
    lumped    = (physics_pc_method_id() == PC_METHOD_VS_LUMPED)
    if (lumped) then
      form = "avg"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_form", form, is_set, ierr)
      select case (trim(form))
      case ("plain"); dual_form = DUAL_PLAIN
      case ("avg");   dual_form = DUAL_AVG
      case ("sym");   dual_form = DUAL_SYM
      case default
        call fatal("-vs_dual_form must be plain | avg | sym, got '"//trim(form)//"'")
      end select
      form = "ideal"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_bending", form, is_set, ierr)
      select case (trim(form))
      case ("ideal");     res_dual = .false.
      case ("resistive"); res_dual = .true.
      case default
        call fatal("-vs_bending must be ideal | resistive, got '"//trim(form)//"'")
      end select
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_dual_core", flag, is_set, ierr)
      if (is_set) res_core = flag
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pl_view", pl_view, is_set, ierr)
      if (.not. is_set) pl_view = ""
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_ss_view", ss_view, is_set, ierr)
      if (.not. is_set) ss_view = ""
    endif
    bk_pj   = sf_backend(physics_pc_sf_pair_psi, "physics_pc_sf_pair_psi", my_id)
    bk_vort = sf_backend(physics_pc_sf_pair_w,   "physics_pc_sf_pair_w",   my_id)
    bk_rho  = sf_backend(physics_pc_sf_rho,      "physics_pc_sf_rho",      my_id)
    bk_T    = sf_backend(physics_pc_sf_T,        "physics_pc_sf_T",        my_id)
    bk_pl   = bk_vort
    if (lumped) then
      form = "default"
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_p_solver", form, is_set, ierr)
      select case (trim(form))
      case ("default");  bk_pl = bk_vort
      case ("spdmg");    bk_pl = SF_SPDMG
      case default
        call fatal("-vs_p_solver must be default | spdmg, got '"//trim(form)//"'")
      end select
    endif
    pname = physics_pc_sf_pair_w
    if (bk_pl == SF_SPDMG) pname = "spdmg (ring-free SPD multigrid)"

    if (jorek_model /= 199 .and. jorek_model /= 600) &
      call fatal("physics_pc_method = vs_ideal | vs_resistive | vs_lumped supports model199 and model600 "// &
                 "only (model183's psi row has the mass B_11, not opz B_33)")
    if (n_var /= 6 .or. var_T /= 6) &
      call fatal("physics_pc_method = vs_ideal | vs_resistive | vs_lumped supports (psi, u, zj, w, rho, T) "// &
                 "only: build without v_par, with_TiTe, with_neutrals and with_impurities")
    if (mod(n_tor, 2) /= 1) call fatal("physics_pc_method = vs_ideal | vs_resistive | vs_lumped needs n_tor "// &
                                       "odd (n = 0 plus cos/sin pairs)")
    ! -vs_bending resistive: the models above assemble M_eta, K_eta and K_1
    ! with their own eta_T and eta_num_T (pc_elt_matrix_eta); model600's
    ! without its extensions, which n_var 6 excludes

    harm_all    = physics_pc_sf_harm_couple
    sf_gmg_harm = merge(SF_GMG_HARM_ALL, SF_GMG_HARM_PAIR, harm_all)
    famode      = physics_pc_sf_mode_split .and. .not. harm_all

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] ========= vorticity-space Schur (VS) preconditioner ========="
      if (resistive) then
        write(*,'(A)') P//"line bending: resistive, (B_33 - B_31 B_11^-1 B_13)^-1"
        write(*,'(A,A)') P//"vorticity: ", trim(physics_pc_sf_pair_w)//" (omega, y, z; M_rho = inertia)"
      else if (lumped .and. res_dual) then
        write(*,'(A)') P//"line bending: resistive, S_s^-1 by its local dual basis Z as X = "// &
                       trim(form_text(dual_form))//", S_s = B_33 + K_eta / opz + sym(B_31 Z_11 K_hyp)"
        if (res_core) then
          write(*,'(A)') P//"  Z with one core patch at the axis (-vs_dual_core)"
        else
          write(*,'(A)') P//"  Z without the core patch at the axis (-vs_dual_core false)"
        endif
        write(*,'(A,A)') P//"vorticity: ", trim(pname)//" (P_lumped = M_rho + B_24 - B_23 X B_12 / opz)"
      else if (lumped) then
        write(*,'(A)') P//"line bending: ideal, B_33^-1 by its local dual basis Z as X = "// &
                       trim(form_text(dual_form))
        write(*,'(A,A)') P//"vorticity: ", trim(pname)//" (P_lumped = M_rho + B_24 - B_23 X B_12 / opz)"
      else
        write(*,'(A)') P//"line bending: ideal, B_33^-1"
        write(*,'(A,A)') P//"vorticity: ", trim(physics_pc_sf_pair_w)//" (omega, y; M_rho = inertia)"
      endif
      write(*,'(A,A)') P//"Laplacian: ", trim(physics_pc_sf_pair_w)//" (B_42, once)"
      write(*,'(A,A)') P//"pair_psi : ", trim(physics_pc_sf_pair_psi)
      write(*,'(A,A)') P//"rho / T  : ", trim(physics_pc_sf_rho)//" / "//trim(physics_pc_sf_T)
      write(*,'(A,ES9.2)') P//"masses B_33, B_44: Chebyshev, B-norm error bound ", MASS_TARGET
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

    vs_init_done = .true.

  contains

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') P//"FATAL: ", trim(msg)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine vs_init

  !> Does the method need M_eta and K_eta from the element assembly
  !! (vs_lumped with -vs_bending resistive)? After vs_init.
  logical function vs_wants_eta()
    vs_wants_eta = vs_init_done .and. lumped .and. res_dual
  end function vs_wants_eta

  !> Build the preconditioner from the assembled Jacobian A_full, at every PC
  !! rebuild. The operators' patterns are fixed by the first build; later
  !! builds refill the operators whose values change. The masses and the
  !! Laplacian depend on the grid only: extracted and set up once.
  subroutine vs_build(A_full, comm, my_id)
    use phys_module, only: physics_pc_sf_rtol, eliminate_boundary_dofs
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr
    logical :: first
    real*8  :: rtol
    integer :: k, nno, comm_s
    Mat :: wm(1), wd(1)

    call vs_init(my_id)
    first  = vs_first
    comm_g = comm
    rtol   = physics_pc_sf_rtol

    if (first) then
      if (.not. (g_ctx%m_rho_ready .and. g_ctx%d_bc_ready)) &
        call build_fatal("M_rho / D_bc were never assembled (petsc_assemble_pc_matrices)")
      if (res_dual .and. .not. g_ctx%eta_ready) &
        call build_fatal("M_eta / K_eta were never assembled (petsc_assemble_pc_matrices)")
      call sfo_layout(A_full, comm, famode, harm_all, my_id)
      call define_operators()
    endif
    comm_s = sfo_comm

    ! the W sources: M_rho in the vorticity system (unread elsewhere), D_bc in the Laplacian
    wm = g_ctx%M_rho
    wd = g_ctx%D_bc
    call PetscLogEventBegin(pcev_extract, ierr)
    call build_or_fill(o_vort, wm); call build_or_fill(o_pj, wm)
    call build_or_fill(o_rho, wm);  call build_or_fill(o_T, wm)
    do k = 1, NCB
      call build_or_fill(o_cb(k), wm)
    enddo
    if (res_dual) then
      do k = 1, NRB
        call build_or_fill(o_rb(k), wm)
      enddo
      call build_or_fill(o_meta, [g_ctx%M_eta])
      call build_or_fill(o_keta, [g_ctx%K_eta])
    endif
    if (first) then
      call sfo_build(o_lap, A_full, wd, sf_opz(), my_id)
      call sfo_build(o_mj, A_full, wm, sf_opz(), my_id)
      call sfo_build(o_mw, A_full, wm, sf_opz(), my_id)
    endif
    call PetscLogEventEnd(pcev_extract, ierr)
    if (first) then
      ! on the operators' communicator: B_65 is refilled at every build, the
      ! flag only skips its product, so the families need not agree
      use_b65 = sfo_b65_used(o_cb(CB_65), comm_s)
      nno = 0
      do k = 1, NCB
        if (.not. blockmv_attach(o_cb(k)%mat, int(pc_ntor))) nno = nno + 1
      enddo
      if (nno > 0 .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   VS: WARNING ", nno, &
        " coupling block(s) not AIJ, left on PETSc's matvec"
    endif
    if (lumped) call assemble_lumped()
    call physics_pc_mem("VS build: operators filled", my_id)

    ! the Jacobian's boundary rows, penalty or eliminated (diagonal only), are
    ! back-substituted after every apply in both modes: steps 2 and 3 give u
    ! and omega there from each other's residual, which would put an
    ! eigenvalue -a_uu a_ww of A P^-1 on the eliminated rows
    call full_penalty_rows(pen, A_full, comm, my_id, first)
    ! penalty rows as Dirichlet rows of the solved blocks, before the scaling.
    ! P_lumped in both boundary modes: M_rho's Dirichlet rows keep their
    ! columns, which meet zero boundary values with eliminated boundary rows
    ! but leave the matrix not symmetric
    if (lumped) then
      call sf_solver_penalty_rows(slv_vort, o_pl, comm_s, my_id)
      if (first) call report_skew()
      call view_build(o_pl%mat, pl_view, "P_lumped")
    endif
    if (.not. eliminate_boundary_dofs) then
      if (.not. lumped) call sf_solver_penalty_rows(slv_vort, o_vort, comm_s, my_id)
      call sf_solver_penalty_rows(slv_pj, o_pj, comm_s, my_id)
      call sf_solver_penalty_rows(slv_rho, o_rho, comm_s, my_id)
      call sf_solver_penalty_rows(slv_T, o_T, comm_s, my_id)
      if (first) call sf_solver_penalty_rows(slv_lap, o_lap, comm_s, my_id)
    endif
    if (.not. lumped) call sf_solver_block_scale(slv_vort, o_vort, comm_s, my_id)
    call sf_solver_block_scale(slv_pj, o_pj, comm_s, my_id)

    call PetscLogEventBegin(pcev_fact_pj, ierr)
    call sf_solver_setup(slv_pj, o_pj%mat, bk_pj, "pair_psi", comm_s, my_id, rtol, gmg_inst=2, nfields=2, &
                         smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS, &
                         pre0=SF_PJ_PRE0, post0=SF_PJ_POST0, nsmooth_c=SF_PJ_NSC, &
                         axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    call PetscLogEventEnd(pcev_fact_pj, ierr)

    ! the vorticity system as pair_w: line bending along the flux surfaces;
    ! P_lumped, one field, as T: a diffusion along B
    call PetscLogEventBegin(pcev_fact_vort, ierr)
    if (lumped) then
      call sf_solver_setup(slv_vort, o_pl%mat, bk_pl, "P_lumped", comm_s, my_id, rtol, gmg_inst=1, &
                           nfields=1, smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS)
    else
      call sf_solver_setup(slv_vort, o_vort%mat, bk_vort, "vorticity", comm_s, my_id, rtol, gmg_inst=1, &
                           nfields=o_vort%nf, smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS, &
                           pre0=SF_W_PRE0, post0=SF_W_POST0, nsmooth_c=SF_W_NSC, &
                           axis_rings=SF_GMG_AXIS_RINGS_PAIRS)
    endif
    call PetscLogEventEnd(pcev_fact_vort, ierr)

    call PetscLogEventBegin(pcev_fact_rhot, ierr)
    call sf_solver_setup(slv_rho, o_rho%mat, bk_rho, "rho", comm_s, my_id, rtol, gmg_inst=3, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_LINES, maxits=SF_GMG_MAXITS_RHOT)
    call sf_solver_setup(slv_T, o_T%mat, bk_T, "T", comm_s, my_id, rtol, gmg_inst=4, nfields=1, &
                         smoother=SF_GMG_SMOOTHER_RINGS, maxits=SF_GMG_MAXITS_RHOT)
    call PetscLogEventEnd(pcev_fact_rhot, ierr)

    ! the isotropic Laplacian on the anisotropic polar grid: zebra rings
    ! inside, zebra radial lines outside, as pair_psi's Laplacian rows
    if (first) then
      call PetscLogEventBegin(pcev_fact_lap, ierr)
      call sf_solver_setup(slv_lap, o_lap%mat, bk_vort, "Laplacian", comm_s, my_id, rtol, gmg_inst=5, &
                           nfields=1, smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, maxits=SF_GMG_MAXITS)
      call mass_cheb_setup(mc_j, o_mj%mat, comm_s, "j mass B_33", int(pc_ntor), target=MASS_TARGET)
      call mass_cheb_setup(mc_w, o_mw%mat, comm_s, "omega mass B_44", int(pc_ntor), target=MASS_TARGET)
      call PetscLogEventEnd(pcev_fact_lap, ierr)
    endif
    call physics_pc_mem("VS build: solvers set up", my_id)

    if (.not. vecs_ready) call create_work_vectors()
    g_ctx%reduced_ready = .true.
    vs_first = .false.
    ! the families finish their setups at different times: wait here, so that
    ! the setup timer holds the slowest one rather than the first apply
    if (famode) call MPI_Barrier(comm, ierr)

  contains

    !> P_lumped = (M_rho + B_24) - (1/opz) B_23 X B_12: X and the product's
    !! symbolic phase once, the union pattern fixed by the first build, the
    !! numeric product at every build; X at every build for the resistive
    !! bending.
    subroutine assemble_lumped()
      if (res_dual) call resistive_bending()
      if (first) then
        if (.not. res_dual) then
          call dual_basis(o_mj%mat, int(pc_ntor), z_j, my_id, "B_33")
          call dual_symmetrize(z_j, o_mj%mat, dual_form)
        endif
        call MatMatMatMult(o_cb(CB_23)%mat, z_j, o_cb(CB_12)%mat, MAT_INITIAL_MATRIX, 2.0d0, t_bzb, ierr)
        call MatDuplicate(t_bzb, MAT_COPY_VALUES, o_pl%mat, ierr)
        call MatAXPY(o_pl%mat, 1.0d0, o_vort%mat, DIFFERENT_NONZERO_PATTERN, ierr)
        o_pl%label = "P_lumped"; o_pl%nf = 1
      else
        call MatMatMatMult(o_cb(CB_23)%mat, z_j, o_cb(CB_12)%mat, MAT_REUSE_MATRIX, 2.0d0, t_bzb, ierr)
      endif
      call MatZeroEntries(o_pl%mat, ierr)
      call MatAXPY(o_pl%mat, -1.0d0 / sf_opz(), t_bzb, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(o_pl%mat, 1.0d0, o_vort%mat, SUBSET_NONZERO_PATTERN, ierr)
    end subroutine assemble_lumped

    !> X (z_j) of the resistive bending, at every build (module header):
    !!   K_hyp = -(B_13 + M_eta),  T = B_31 Z_11 K_hyp,
    !!   S_s = B_33 + K_eta / opz + (T + T') / 2,
    !! X the form dual_form of the local dual basis of S_s with the core patch.
    !! K_hyp and K_eta are masked to psi's and j's interior DOFs. The patterns
    !! are those of the first build (the grid's patches), and z_11, k_hyp,
    !! t_s, t_st, s_res and z_j stay the same objects for the reuse of the
    !! products.
    subroutine resistive_bending()
      Mat :: zn
      integer :: nm_psi, nm_j
      if (first) then
        call penalty_row_mask(o_rb(RB_11)%mat, mask_psi, nm_psi)
        call penalty_row_mask(o_mj%mat, mask_j, nm_j)
        if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   resistive bending: ", nm_psi, &
          " constrained rows of psi (B_11), ", nm_j, " of j (B_33) leave K_hyp and K_eta"
        call MatDuplicate(o_rb(RB_13)%mat, MAT_COPY_VALUES, k_hyp, ierr)
        call MatAXPY(k_hyp, 1.0d0, o_meta%mat, DIFFERENT_NONZERO_PATTERN, ierr)
      else
        call MatZeroEntries(k_hyp, ierr)
        call MatAXPY(k_hyp, 1.0d0, o_rb(RB_13)%mat, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(k_hyp, 1.0d0, o_meta%mat, SUBSET_NONZERO_PATTERN, ierr)
      endif
      call MatScale(k_hyp, -1.0d0, ierr)
      call MatDiagonalScale(k_hyp, mask_psi, mask_j, ierr)
      call MatDiagonalScale(o_keta%mat, mask_j, mask_j, ierr)
      if (first) call eta_self_check()

      ! T = B_31 Z_11 K_hyp
      if (first) then
        call dual_basis(o_rb(RB_11)%mat, int(pc_ntor), z_11, my_id, "B_11")
        call MatMatMatMult(o_rb(RB_31)%mat, z_11, k_hyp, MAT_INITIAL_MATRIX, 2.0d0, t_s, ierr)
        call MatTranspose(t_s, MAT_INITIAL_MATRIX, t_st, ierr)
        ! S_s's pattern: B_33, K_eta, T and T'
        call MatDuplicate(o_mj%mat, MAT_COPY_VALUES, s_res, ierr)
        call MatAXPY(s_res, 1.0d0, o_keta%mat, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatAXPY(s_res, 1.0d0, t_s, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatAXPY(s_res, 1.0d0, t_st, DIFFERENT_NONZERO_PATTERN, ierr)
      else
        call dual_basis(o_rb(RB_11)%mat, int(pc_ntor), zn, my_id)
        call MatCopy(zn, z_11, SAME_NONZERO_PATTERN, ierr)
        call MatDestroy(zn, ierr)
        call MatMatMatMult(o_rb(RB_31)%mat, z_11, k_hyp, MAT_REUSE_MATRIX, 2.0d0, t_s, ierr)
        call MatTranspose(t_s, MAT_REUSE_MATRIX, t_st, ierr)
      endif
      ! S_s = B_33 + K_eta / opz + (T + T') / 2
      call MatZeroEntries(s_res, ierr)
      call MatAXPY(s_res, 1.0d0, o_mj%mat, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(s_res, 1.0d0 / sf_opz(), o_keta%mat, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(s_res, 0.5d0, t_s, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(s_res, 0.5d0, t_st, SUBSET_NONZERO_PATTERN, ierr)
      call view_build(s_res, ss_view, "S_s")

      ! X
      if (first) then
        call dual_basis(s_res, int(pc_ntor), z_j, my_id, "S_s", core=res_core)
        call dual_symmetrize(z_j, s_res, dual_form)
      else
        call dual_basis(s_res, int(pc_ntor), zn, my_id, core=res_core)
        call dual_symmetrize(zn, s_res, dual_form)
        call MatCopy(zn, z_j, SAME_NONZERO_PATTERN, ierr)
        call MatDestroy(zn, ierr)
      endif
    end subroutine resistive_bending

    !> The first build's check of M_eta against the Jacobian: K_hyp - K_1 =
    !! -(B_13 + M_eta + K_1) on the interior rows and columns, against M_eta
    !! and B_13 there (Frobenius norms), and K_hyp's skew part (its
    !! coefficient eta_num_T is a function of the state in model600). Then
    !! K_1 is freed.
    subroutine eta_self_check()
      type(sf_op_t) :: o_k1
      Mat :: dm, bm
      real*8 :: nd, nb, nm, nref, nks, nkh
      o_k1%label = "K_1"; o_k1%nf = 1; o_k1%tag = o_keta%tag + 1
      o_k1%se(1, 1) = 0; o_k1%sv(1, 1) = 0; o_k1%wid(1, 1) = 1
      if (.not. g_ctx%k1_ready) call build_fatal("K_1 was never assembled (petsc_assemble_pc_matrices)")
      call sfo_build(o_k1, A_full, [g_ctx%K_1], sf_opz(), my_id)
      call MatDuplicate(k_hyp, MAT_COPY_VALUES, dm, ierr)
      call MatAXPY(dm, -1.0d0, o_k1%mat, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(dm, mask_psi, mask_j, ierr)
      call MatNorm(dm, NORM_FROBENIUS, nd, ierr)
      call MatDestroy(dm, ierr)
      call MatDuplicate(o_rb(RB_13)%mat, MAT_COPY_VALUES, bm, ierr)
      call MatDiagonalScale(bm, mask_psi, mask_j, ierr)
      call MatNorm(bm, NORM_FROBENIUS, nb, ierr)
      call MatDestroy(bm, ierr)
      call MatDuplicate(o_meta%mat, MAT_COPY_VALUES, bm, ierr)
      call MatDiagonalScale(bm, mask_psi, mask_j, ierr)
      call MatNorm(bm, NORM_FROBENIUS, nm, ierr)
      call MatDestroy(bm, ierr)
      ! K_hyp' - K_hyp on the rows and columns interior to psi and j both
      call MatTranspose(k_hyp, MAT_INITIAL_MATRIX, dm, ierr)
      call MatAXPY(dm, -1.0d0, k_hyp, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDiagonalScale(dm, mask_psi, mask_psi, ierr)
      call MatDiagonalScale(dm, mask_j, mask_j, ierr)
      call MatNorm(dm, NORM_FROBENIUS, nks, ierr)
      call MatDestroy(dm, ierr)
      call MatNorm(k_hyp, NORM_FROBENIUS, nkh, ierr)
      call sfo_destroy(o_k1)
      call MatDestroy(g_ctx%K_1, ierr)
      g_ctx%k1_ready = .false.; g_ctx%k1_done = .true.
      if (my_id == 0) then
        write(*,'(A,ES10.3,A,ES10.3,A,ES10.3,A)') "[Physics PC]   resistive bending check: |B_13 + M_eta + K_1|_F = ", &
          nd, " of |M_eta|_F = ", nm, ", of |B_13|_F = ", nb, &
          " (K_1 = theta dt (eta_num_T grad v, grad w), interior rows)"
        ! against M_eta, or B_13 where eta = 0
        nref = nm
        if (nref <= 0.d0) nref = nb
        if (nref > 0.d0 .and. nd > ETA_CHECK_TOL * nref) write(*,'(A,ES10.3,A)') &
          "[Physics PC]   WARNING: -B_13 - M_eta differs from the hyper-resistive part by ", nd / nref, &
          " of |M_eta| (|B_13| if eta = 0): the resistive bending does not use the model's eta_T and eta_num_T"
        if (nkh > 0.d0) write(*,'(A,ES9.2,A)') "[Physics PC]   K_hyp = -B_13 - M_eta: skew part ", &
          0.5d0 * nks / nkh, " of its Frobenius norm (interior rows)"
      endif
    end subroutine eta_self_check

    !> The skew part of the solved P_lumped against the matrix, in the
    !! Frobenius norm: the quadrature error of B_23 = -B_12' with a symmetric
    !! X and no flow.
    subroutine report_skew()
      Mat :: pt
      real*8 :: nskew, np_
      call MatTranspose(o_pl%mat, MAT_INITIAL_MATRIX, pt, ierr)
      call MatAXPY(pt, -1.0d0, o_pl%mat, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatNorm(pt, NORM_FROBENIUS, nskew, ierr)
      call MatNorm(o_pl%mat, NORM_FROBENIUS, np_, ierr)
      call MatDestroy(pt, ierr)
      if (my_id == 0) write(*,'(A,ES9.2,A)') "[Physics PC]   P_lumped: skew part ", 0.5d0 * nskew / np_, &
        " of its Frobenius norm"
    end subroutine report_skew

    !> mat in PETSc's binary format to file fname, if fname is set (the
    !! operators' layout: JOREK index, slot fastest): the first build creates
    !! the file, every later build appends its matrix. Collective.
    subroutine view_build(mat, fname, what)
      Mat, intent(in) :: mat
      character(len=*), intent(in) :: fname, what
      PetscViewer :: vw
      if (len_trim(fname) == 0) return
      if (first) then
        call PetscViewerBinaryOpen(comm_s, trim(fname), FILE_MODE_WRITE, vw, ierr)
      else
        call PetscViewerBinaryOpen(comm_s, trim(fname), FILE_MODE_APPEND, vw, ierr)
      endif
      call MatView(mat, vw, ierr)
      call PetscViewerDestroy(vw, ierr)
      if (my_id == 0) write(*,'(A)') "[Physics PC]   "//what//" written to "//trim(fname)
    end subroutine view_build

    subroutine build_or_fill(op, ws)
      type(sf_op_t), intent(inout) :: op
      Mat, intent(in) :: ws(:)
      if (first) then
        call sfo_build(op, A_full, ws, sf_opz(), my_id)
      else
        call sfo_fill(op, A_full, ws, sf_opz())
      endif
    end subroutine build_or_fill

    !> The operators' block layouts.
    subroutine define_operators()
      use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
      integer, parameter :: ce(NCB) = [var_psi, var_u, var_u, var_rho, var_rho, var_T, var_T, var_T, var_T]
      integer, parameter :: cv(NCB) = [var_u, var_zj, var_w, var_psi, var_u, var_psi, var_u, var_zj, var_rho]
      character(len=4), parameter :: cn(NCB) = ["B_12", "B_23", "B_24", "B_51", "B_52", "B_61", "B_62", &
                                                "B_63", "B_65"]
      integer, parameter :: re(NRB) = [var_psi, var_psi, var_zj]
      integer, parameter :: rv(NRB) = [var_psi, var_zj, var_psi]
      character(len=4), parameter :: rn(NRB) = ["B_11", "B_13", "B_31"]
      integer, parameter :: NDIAG = 7
      integer :: k_
      ! fields (w, y[, z]): the columns of omega, j[, psi]; the rows of u, then
      ! of psi and j (y), [j and psi (z)], as in the module header
      o_vort%label = "vorticity"; o_vort%tag = 1
      o_vort%nf = merge(3, 2, resistive)
      if (lumped) then                                                             ! M_rho + B_24 alone
        o_vort%label = "M_rho + B_24"; o_vort%nf = 1
      endif
      o_vort%se(1, 1) = var_u;   o_vort%sv(1, 1) = var_w;  o_vort%wid(1, 1) = 1     ! B_24 + M_rho
      o_vort%se(1, 2) = var_u;   o_vort%sv(1, 2) = var_zj                          ! B_23
      o_vort%se(2, 1) = var_psi; o_vort%sv(2, 1) = var_u                           ! B_12
      o_vort%se(2, 2) = var_zj;  o_vort%sv(2, 2) = var_zj; o_vort%scl(2, 2) = 1     ! opz B_33
      if (resistive) then
        o_vort%se(2, 3) = var_zj;  o_vort%sv(2, 3) = var_psi                       ! B_31
        o_vort%se(3, 2) = var_psi; o_vort%sv(3, 2) = var_zj; o_vort%scl(3, 2) = 1   ! opz B_13
        o_vort%se(3, 3) = var_psi; o_vort%sv(3, 3) = var_psi                       ! B_11
      endif
      call sfo_define(o_lap, "Laplacian B_42", 2, [var_w])
      o_lap%sv(1, 1) = var_u; o_lap%wid(1, 1) = 1                                  ! B_42 + D_bc
      call sfo_define(o_pj,  "pair_psi", 3, [var_psi, var_zj])
      call sfo_define(o_rho, "B_55", 4, [var_rho])
      call sfo_define(o_T,   "B_66", 5, [var_T])
      call sfo_define(o_mj,  "B_33", 6, [var_zj])
      call sfo_define(o_mw,  "B_44", NDIAG, [var_w])
      do k_ = 1, NCB
        o_cb(k_)%label = cn(k_); o_cb(k_)%nf = 1; o_cb(k_)%tag = NDIAG + k_
        o_cb(k_)%se(1, 1) = ce(k_); o_cb(k_)%sv(1, 1) = cv(k_)
      enddo
      ! the resistive bending: B_11, B_13, B_31 and the W sources M_eta, K_eta alone
      do k_ = 1, NRB
        o_rb(k_)%label = rn(k_); o_rb(k_)%nf = 1; o_rb(k_)%tag = NDIAG + NCB + k_
        o_rb(k_)%se(1, 1) = re(k_); o_rb(k_)%sv(1, 1) = rv(k_)
      enddo
      o_meta%label = "M_eta"; o_meta%nf = 1; o_meta%tag = NDIAG + NCB + NRB + 1; o_meta%wid(1, 1) = 1
      o_keta%label = "K_eta"; o_keta%nf = 1; o_keta%tag = NDIAG + NCB + NRB + 2; o_keta%wid(1, 1) = 1
    end subroutine define_operators

    subroutine create_work_vectors()
      integer :: k_
      call MatCreateVecs(o_vort%mat, rhs_V, sol_V, ierr)
      call MatCreateVecs(o_pj%mat, rhs_PJ, sol_PJ, ierr)
      call MatCreateVecs(o_rho%mat, sv_x(1), PETSC_NULL_VEC, ierr)
      do k_ = 2, NSV
        call VecDuplicate(sv_x(1), sv_x(k_), ierr)
      enddo
      do k_ = 1, NSV
        call VecDuplicate(sv_x(1), sv_y(k_), ierr)
      enddo
      call VecDuplicate(sv_x(1), m_j, ierr)
      call VecDuplicate(sv_x(1), w1, ierr)
      call VecDuplicate(sv_x(1), w2, ierr)
      call VecDuplicate(sv_x(1), w3, ierr)
      call VecDuplicate(sv_x(1), zv, ierr)
      call VecSet(zv, 0.0d0, ierr)
      vecs_ready = .true.
    end subroutine create_work_vectors

    subroutine build_fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: ", msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine build_fatal

  end subroutine vs_build

  !> y = the preconditioner applied to x, steps 1 to 7 of the module header.
  subroutine vs_apply(x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    Vec :: x_psi, x_u, x_j, x_w, x_rho, x_T
    Vec :: y_psi, y_u, y_j, y_w, y_rho, y_T
    real*8 :: t0, t1

    ierr = 0
    call PetscLogEventBegin(pcev_apply, ierr)
    t0 = MPI_Wtime()
    call sfo_vars_in(x, sv_x)
    t_scat = t_scat + (MPI_Wtime() - t0)
    x_psi = sv_x(var_psi); x_u   = sv_x(var_u);   x_j = sv_x(var_zj)
    x_w   = sv_x(var_w);   x_rho = sv_x(var_rho); x_T = sv_x(var_T)
    y_psi = sv_y(var_psi); y_u   = sv_y(var_u);   y_j = sv_y(var_zj)
    y_w   = sv_y(var_w);   y_rho = sv_y(var_rho); y_T = sv_y(var_T)

    ! 1. m_j = B_33^-1 x_j, m_omega = B_44^-1 x_omega (held in y_w)
    call PetscLogEventBegin(pcev_solve_mass, ierr)
    call mass_cheb_solve(mc_j, x_j, m_j)
    call mass_cheb_solve(mc_w, x_w, y_w)
    call PetscLogEventEnd(pcev_solve_mass, ierr)

    ! 2. the vorticity system (P_lumped) on (x_u - B_23 m_j - B_24 m_omega, 0[, 0])
    call VecCopy(x_u, w1, ierr)
    call sub_mult(CB_23, m_j, w1)
    call sub_mult(CB_24, y_w, w1)
    select case (o_vort%nf)
    case (1); call sf_split_parts(rhs_V, [w1], .true.)
    case (2); call sf_split_parts(rhs_V, [w1, zv], .true.)
    case (3); call sf_split_parts(rhs_V, [w1, zv, zv], .true.)
    end select
    call sf_solver_solve(slv_vort, rhs_V, sol_V, pcev_solve_vort)
    call sf_split_parts(sol_V, [w2], .false.)

    ! 3. omega = m_omega + w; B_42 u = x_omega - B_44 omega
    call VecAXPY(y_w, 1.0d0, w2, ierr)
    call MatMult(o_mw%mat, y_w, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_w, ierr)
    call sf_solver_solve(slv_lap, w1, y_u, pcev_solve_lap)

    ! 4. pair_psi (psi, j) = (x_psi - B_12 u, x_j)
    call MatMult(o_cb(CB_12)%mat, y_u, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_psi, ierr)
    call sf_split_parts(rhs_PJ, [w1, x_j], .true.)
    call sf_solver_solve(slv_pj, rhs_PJ, sol_PJ, pcev_solve_pj)
    call sf_split_parts(sol_PJ, [y_psi, y_j], .false.)

    ! 5. B_55 rho = x_rho - B_51 psi - B_52 u
    call MatMult(o_cb(CB_51)%mat, y_psi, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_rho, ierr)
    call sub_mult(CB_52, y_u, w1)
    call sf_solver_solve(slv_rho, w1, y_rho, pcev_solve_rhot)

    ! 6. B_66 T = x_T - B_61 psi - B_63 j - B_62 u - B_65 rho
    call MatMult(o_cb(CB_61)%mat, y_psi, w1, ierr)
    call VecAYPX(w1, -1.0d0, x_T, ierr)
    call sub_mult(CB_63, y_j, w1)
    call sub_mult(CB_62, y_u, w1)
    if (use_b65) call sub_mult(CB_65, y_rho, w1)
    call sf_solver_solve(slv_T, w1, y_T, pcev_solve_rhot)

    ! 7.
    t1 = MPI_Wtime()
    call sfo_vars_out(sv_y, y)
    t_scat  = t_scat + (MPI_Wtime() - t1)
    if (pen%n > 0) call penalty_backsub(pen, x, y)
    t_apply = t_apply + (MPI_Wtime() - t0)
    call PetscLogEventEnd(pcev_apply, ierr)
    ierr = 0

  contains

    !> r -= B_k v
    subroutine sub_mult(k, v, r)
      integer, intent(in) :: k
      Vec :: v, r
      call MatMult(o_cb(k)%mat, v, w3, ierr)
      call VecAXPY(r, -1.0d0, w3, ierr)
    end subroutine sub_mult

  end subroutine vs_apply

  !> Inner-iteration summary, one line per block (mode split: per family),
  !! then reset. Collective.
  subroutine vs_report(my_id)
    integer, intent(in) :: my_id
    if (famode) then
      call sf_solver_family_report("VS", [slv_vort, slv_lap, slv_pj, slv_rho, slv_T], t_apply, t_scat, comm_g)
    else
      call sf_solver_report(slv_vort, my_id)
      call sf_solver_report(slv_lap,  my_id)
      call sf_solver_report(slv_pj,   my_id)
      call sf_solver_report(slv_rho,  my_id)
      call sf_solver_report(slv_T,    my_id)
    endif
    call sf_solver_reset_counters(slv_vort)
    call sf_solver_reset_counters(slv_lap)
    call sf_solver_reset_counters(slv_pj)
    call sf_solver_reset_counters(slv_rho)
    call sf_solver_reset_counters(slv_T)
    t_apply = 0.d0;  t_scat = 0.d0
  end subroutine vs_report

#endif
end module mod_petsc_pc_vs
