module mod_petsc_pc_sf_peel
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx
  use mod_petsc_pc_blocks,      only: pack_pair_aij, make_field_block_scale, pc_print_block_setup
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_mass_cheb,   only: mass_cheb_t, mass_cheb_setup, mass_cheb_solve
  implicit none
  private

  !--------------------------------------------------------------------
  !> pair_w by the peeled Schur preconditioner (physics_pc_sf_pair_w =
  !! "peel"; design note pc_design_22_10_26/peeled_schur_preconditioner.md):
  !! FGMRES on the packed pair_w (u, omega, psi, j) = [[A_uu, A_ua], [A_au,
  !! A_aa]], right-preconditioned by the block LDU
  !!   y_a* = A_aa^-1 r_a,  y_u = -P^-1 (r_u - A_ua y_a*),
  !!   y_a  = A_aa^-1 (r_a - A_au y_u),
  !! A_aa = diag(B_44, [[opz B_33, B_13], [B_31, B_33]]), and the positive
  !! reduced velocity operator S = -(A_uu - A_ua A_aa^-1 A_au) replaced by P.
  !!
  !! v2 (default; pc_design_fix/peeled_schur_preconditioner_v2.md):
  !!   P = Q (opz R_hat)^-1 N,  P^-1 = N^-1 (opz R_hat) Q^-1,
  !!   Q     = opz K_rho + W_comp,
  !!   R_hat = opz M_rho + theta dt K_eta + theta dt K_w X K_w,
  !!   N     = opz R_hat + B_12^T X B_12 + theta dt sym(R_hat X_AS(M_rho) K_nu),
  !! X = X_AS, the element additive Schwarz inverse of B_33 (axis elements one
  !! patch). R_hat carries the psi row's resistive response (eta_T, and the
  !! hyper-resistive eta_num k^4 by the sandwich, w^2 = eta_num rho0 R^3, which
  !! the design leaves out) and is only multiplied, so X only ever
  !! approximates a mass inverse. The viscous term carries R_hat's resistive
  !! factor too: the design's opz theta dt K_nu is divided by it in P and
  !! fell short of S by up to ~600x on viscosity-dominated modes (tstep 10).
  !! v1 (-sf_peel_v 1): P = Q M^-1 T, T = M_rho + opz^-2 B_12^T X B_12 +
  !! theta dt / opz K_nu, X ~ B_33^-1 diagonal (optionally resistive) or
  !! Chebyshev.
  !! Left out of P (the FGMRES matvec keeps them): the curvature half of W,
  !! the kink path B_21, flow and hyper-viscosity.
  !!
  !! Measured on the 41x32 shaped case (util/peeled_schur/README.md):
  !! cond(X_AS B_33) = 9 (diag 474); with exact inner solves S P^-1 in
  !! [0.60, 2.0] (exact X) and [0.14, 1.27] (X_AS) at tstep 10.
  !!
  !! Q: real, isotropic up to W_comp's dZ^2 (GMG instance 6, zebra rings as
  !! pair_psi). T / N: anisotropic along B (instance 7, rings + radial
  !! semi-coarsening as pair_w). (psi, j): instance 8, as pair_psi. omega:
  !! B_44 by Chebyshev (mod_petsc_pc_mass_cheb). Dirichlet and axis-penalty
  !! u rows (rows of pair_w's u block with nothing off the diagonal) are
  !! solved exactly from pair_w's diagonal; Q, M_rho, T, R_hat, N are the
  !! identity there.
  !!
  !! Options (all -sf_peel_*):
  !!   v 1|2         the peeling (default 2)
  !!   xas 0|1       v2: X = X_AS (default) or diag(B_33)^-1; as_axis 0|1 the
  !!                 merged axis patch (default 1)
  !!   hyper 0|1     v2: the hyper-resistive sandwich in R_hat (default 1)
  !!   eta_w <w>     v2: K_eta's weight (diagnostic, default 1)
  !!   visc 0..3     0 none, 1 the design's K_nu, v2 2 (default) / 3 with R_hat's
  !!                 resistive factor (3: without its hyper part)
  !!   loc_its n     check, v2: power steps locating X_AS B_33's bottom mode
  !!   x 0|1         v1: X_0 = diag(B_33)^-1 | X_1 = degree-1 Chebyshev on
  !!                 D^-1 B_33 (Gershgorin top, kappa x1_kappa); default 0
  !!   xres 0|1      v1: the resistive diagonal in X_0 (default 1)
  !!   wcomp <w>     W_comp's weight in Q (default 1; 0 leaves it out)
  !!   q, t, aa      lu | gmg: Q, T and A_aa by MUMPS or the GMG (default gmg)
  !!   qcyc, tcyc, pjcyc   V-cycles per solve (default 1; 0 = FGMRES to rtol)
  !!   tri 0|1       lower block-triangular (no y_a*), one A_aa solve less
  !!   restart, maxits     the outer FGMRES (default 10, 30)
  !!   check 1       Stage-0 checks: B_23 + B_12^T, Rayleigh quotients,
  !!                 the compression and resistive weights
  !!   spec n        Stage-1 spectrum: n Arnoldi steps on S P^-1 with
  !!                 A_aa^-1 by MUMPS, so only P (and Q / T's inner solves)
  !!                 is approximate; spec_every 1 repeats it on every build;
  !!                 spec_drop: terms removed from S to attribute the
  !!                 outliers, bit 0 W (curvature and W_comp), bit 1 the kink
  !!                 path B_21 psi, bit 2 W's curvature half only
  !!   texact 1      T (v1) / N (v2) with the exact X (MUMPS; v1 the resistive
  !!                 psi response, v2 B_33^-1 and M_rho^-1), solved to 1e-10
  !!                 by GMRES around its sparse version's solver: separates
  !!                 the commutation error from X's
  !!   check 1, v2   also M1: the spectrum of X_AS B_33 against the colouring
  !!                 bound, and where its extreme eigenvectors sit
  !--------------------------------------------------------------------

  public :: peel_build, peel_solve, peel_report

  !--- options
  integer, save :: o_x = 0, o_xres = 1, o_visc = 1, o_tri = 0, o_check = 0, o_spec = 0, o_spec_every = 0
  integer, save :: o_spec_drop = 0, o_texact = 0
  integer, save :: o_v = 2, o_xas = 1, o_hyper = 1, o_as_axis = 1, o_loc_its = 3000
  integer, save :: o_visc_set = -1             !< -sf_peel_visc as given (v2 default 2)
  real*8,  save :: o_eta_w = 1.d0              !< v2: K_eta's weight in R_hat (diagnostic)
  integer, save :: o_restart = 10, o_maxits = 30
  integer, save :: bk_q = SF_GMG, bk_t = SF_GMG, bk_aa = SF_GMG
  integer, save :: ncyc_q = 1, ncyc_t = 1, ncyc_pj = 1
  real*8,  save :: o_wcomp = 1.d0, o_x1_kappa = 50.d0

  logical, save :: ready = .false.
  integer, save :: comm_p = MPI_COMM_NULL, me_p = 0
  real*8,  save :: opz = 1.d0, thdt = 1.d0

  !--- operators
  Mat, save :: Wop                         !< the packed pair_w (unscaled)
  Mat, save :: Pc(6)                       !< AIJ copies of g_ctx%P_peel
  Mat, save :: Qm, Mm, Tm, TB              !< Q, M_rho, T, B_12^T X B_12
  Mat, save :: B12s, XmB12, Xm             !< X_0: diag(sqrt x) B_12; X_1: X, X B_12
  Mat, save :: RB31, Cres                  !< xres: D^-1 B_31, B_13 D^-1 B_31
  Mat, save :: mpsi, PJm                   !< opz B_33, [[opz B_33, B_13], [B_31, B_33]]
  !> v2: X_AS (element additive Schwarz of B_33, once per mesh), X_AS B_12,
  !! K_w X K_w, R_hat
  Mat, save :: Xas, XasB12, KwX, Hm, Rh
  !> v2 viscosity (visc >= 2): X_AS(M_rho) per rebuild, X_AS(M_rho) K_nu,
  !! R_v X_AS(M_rho) K_nu and its transpose, R_v = R_hat or opz M_rho +
  !! theta dt K_eta
  Mat, save :: Xmr, XmK, Vc, Vt, Rv
  logical, save :: vis_built = .false.
  integer, save :: as_ncol = 0, as_npatch = 0
  type(block_solver_t), save :: lu33          !< texact v2: B_33 by MUMPS
  type(block_solver_t), save :: luM           !< texact v2, visc >= 2: M_rho by MUMPS
  logical, save :: pj_packed = .false.
  Vec, save :: xd                          !< X_0's diagonal
  Vec, save :: binv, imask                 !< u Dirichlet rows: mask / diag, 1 - mask
  PetscInt, allocatable, save :: brows(:)  !< their global rows (u layout)

  !--- solvers
  type(block_solver_t), save :: sq, st, spj, sw
  type(block_solver_t), save :: lw, lpj      !< exact A_aa (MUMPS): spec, texact
  logical, save :: exact_aa = .false.
  KSP, save :: ktex                         !< texact: GMRES on T with the exact X
  Mat, save :: Tex
  type(mass_cheb_t), save :: mw
  KSP, save :: kout

  !--- work vectors (u layout; the (psi, j) pair packed)
  Vec, save :: ru, rw, rp, rj, yu, yw, yp, yj, sv, t1, t2, rpj, ypj

  !--- outer counters since the last report
  integer, save :: n_solve = 0, its_sum = 0, its_max = 0, n_fail = 0
  real*8,  save :: t_sum = 0.d0

contains

  !--------------------------------------------------------------------
  !> Build (first call) or refill the arm from pair_w's packed operator W
  !! and the blocks in g_ctx. Collective on comm.
  !--------------------------------------------------------------------
  subroutine peel_build(W, comm, my_id, rtol)
    use mod_pc_elt_matrix_force_fft, only: pc_peel_supported
    use phys_module, only: time_evol_theta, tstep
    Mat, intent(in)     :: W
    integer, intent(in) :: comm, my_id
    real*8, intent(in)  :: rtol
    PetscErrorCode :: ierr
    PetscInt :: n1
    logical :: first
    PC :: pc
    real*8 :: tw(6)
    character(len=*), parameter :: PP = "[Physics PC]   peel: "

    first  = .not. ready
    comm_p = comm; me_p = my_id
    exact_aa = .false.                       ! the MUMPS A_aa (spec, texact) on these values
    Wop    = W
    opz    = sf_opz()
    thdt   = time_evol_theta * tstep

    if (first) then
      if (.not. pc_peel_supported) call fatal("physics_pc_sf_pair_w = peel supports model199 only")
      if (.not. g_ctx%peel_ready) call fatal("the peel operators were never assembled")
      call read_options()
    endif

    !--- the operators
    tw(1) = MPI_Wtime()
    call to_aij(first)
    if (first) call find_dirichlet_u()
    call set_dirichlet_diag()
    call build_q(first)
    call build_m(first)
    tw(2) = MPI_Wtime()
    if (o_v == 1) then
      call build_x(first)
      call build_t(first)
      tw(3:4) = MPI_Wtime()
    else
      if (first .and. o_xas /= 0) call build_xas(g_ctx%B_33, Xas, "B_33", .true.)
      tw(3) = MPI_Wtime()
      call build_rhat(first)
      call build_n(first)
      tw(4) = MPI_Wtime()
    endif
    call build_pj(first)
    call MatGetLocalSize(g_ctx%B_33, n1, PETSC_NULL_INTEGER, ierr)
    if (spj%scaled) call VecDestroy(spj%dscale, ierr)
    call make_field_block_scale(PJm, [n1, n1], spj%dscale, comm, my_id, "peel (psi, j)")
    spj%scaled = .true.

    !--- the scalar solvers
    tw(5) = MPI_Wtime()
    call sf_solver_setup(sq, Qm, bk_q, "peel Q = opz K_rho + W_comp", comm, my_id, rtol, &
                         gmg_inst=6, nfields=1, smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, &
                         maxits=SF_GMG_MAXITS, ncyc=ncyc_q)
    call sf_solver_setup(st, Tm, bk_t, merge("peel T = M_rho + B_12^T X B_12 / opz^2 + K_nu", &
                                             "peel N = opz R_hat + B_12^T X B_12 + K_nu    ", o_v == 1), &
                         comm, my_id, rtol, &
                         gmg_inst=7, nfields=1, smoother=SF_GMG_SMOOTHER_RINGS, &
                         maxits=SF_GMG_MAXITS, ncyc=ncyc_t)
    call sf_solver_setup(spj, PJm, bk_aa, "peel A_aa (psi, j)", comm, my_id, rtol, &
                         gmg_inst=8, nfields=2, smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, &
                         maxits=SF_GMG_MAXITS, pre0=SF_PJ_PRE0, post0=SF_PJ_POST0, &
                         nsmooth_c=SF_PJ_NSC, axis_rings=SF_GMG_AXIS_RINGS_PAIRS, ncyc=ncyc_pj)
    if (bk_aa == SF_LU) then
      call sf_solver_setup(sw, g_ctx%B_44, SF_LU, "peel A_aa omega (B_44)", comm, my_id, rtol, &
                           gmg_inst=0, nfields=1, smoother=0, maxits=1)
    else if (first) then
      call mass_cheb_setup(mw, g_ctx%B_44, comm, "peel A_aa omega (B_44)")
    endif

    tw(6) = MPI_Wtime()
    if (o_texact /= 0) call setup_texact(first)
    block
      PetscInt :: rs_, re_, i_, nc_
      real*8 :: nz(2)
      integer :: mpierr_
      call MatGetOwnershipRange(Tm, rs_, re_, ierr)
      nz = [0.d0, dble(re_ - rs_)]
      do i_ = rs_, re_ - 1
        call MatGetRow(Tm, i_, nc_, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
        nz(1) = nz(1) + nc_
        call MatRestoreRow(Tm, i_, nc_, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
      enddo
      call MPI_Allreduce(MPI_IN_PLACE, nz, 2, MPI_DOUBLE_PRECISION, MPI_SUM, comm, mpierr_)
      if (my_id == 0) write(*,'(A,4F8.3,A,F7.1)') "[Physics PC]   peel setup s: Q / M, X_AS, R_hat + N (T), "// &
        "solvers:", tw(2) - tw(1), tw(3) - tw(2), tw(4) - tw(3), tw(6) - tw(5), &
        "; N (T) nonzeros per row ", nz(1) / nz(2)
    end block

    !--- the outer FGMRES on pair_w
    if (first) then
      call KSPCreate(comm, kout, ierr)
      call KSPSetType(kout, KSPFGMRES, ierr)
      call KSPGMRESSetRestart(kout, o_restart, ierr)
      call KSPSetTolerances(kout, rtol, 1.d-50, 1.d6, o_maxits, ierr)
      call KSPSetOptionsPrefix(kout, "sf_peel_", ierr)
      call KSPGetPC(kout, pc, ierr)
      call PCSetType(pc, PCSHELL, ierr)
      call PCShellSetApply(pc, peel_pc_apply, ierr)
      call PCShellSetName(pc, "peeled Schur block LDU", ierr)
      call KSPSetFromOptions(kout, ierr)
      call create_work_vectors()
    endif
    call KSPSetOperators(kout, Wop, Wop, ierr)
    call KSPSetUp(kout, ierr)
    block
      character(len=24) :: tstr
      write(tstr, '(ES9.2)') rtol
      call pc_print_block_setup(comm, "pair_w KSP (peel)", "FGMRES + SHELL[peeled Schur block " // &
                                merge("L  ", "LDU", o_tri /= 0) // "], rtol " // trim(adjustl(tstr)))
    end block
    ready = .true.

    if (o_check /= 0 .and. first) call stage0_check()
    if (o_spec > 0 .and. (first .or. o_spec_every /= 0)) call spectrum()

  contains

    subroutine read_options()
      call opt_i("-sf_peel_x", o_x);         call opt_i("-sf_peel_xres", o_xres)
      call opt_i("-sf_peel_visc", o_visc);   call opt_i("-sf_peel_tri", o_tri)
      call opt_i("-sf_peel_check", o_check); call opt_i("-sf_peel_spec", o_spec)
      call opt_i("-sf_peel_spec_every", o_spec_every); call opt_i("-sf_peel_spec_drop", o_spec_drop)
      call opt_i("-sf_peel_texact", o_texact)
      call opt_i("-sf_peel_v", o_v);         call opt_i("-sf_peel_xas", o_xas)
      call opt_i("-sf_peel_hyper", o_hyper); call opt_i("-sf_peel_as_axis", o_as_axis)
      call opt_i("-sf_peel_loc_its", o_loc_its); call opt_r("-sf_peel_eta_w", o_eta_w)
      if (o_v /= 1 .and. o_v /= 2) call fatal("-sf_peel_v must be 1 or 2")
      if (o_v == 2) then                         ! v2 default: the resistive factor on K_nu
        o_visc_set = 2
        call opt_i("-sf_peel_visc", o_visc_set)
        o_visc = o_visc_set
      endif
      if (o_visc < 0 .or. o_visc > 3 .or. (o_v == 1 .and. o_visc > 1)) &
        call fatal("-sf_peel_visc must be 0 | 1 (v1, v2), 2 | 3 (v2)")
      call opt_i("-sf_peel_restart", o_restart); call opt_i("-sf_peel_maxits", o_maxits)
      call opt_i("-sf_peel_qcyc", ncyc_q);   call opt_i("-sf_peel_tcyc", ncyc_t)
      call opt_i("-sf_peel_pjcyc", ncyc_pj)
      call opt_r("-sf_peel_wcomp", o_wcomp); call opt_r("-sf_peel_x1_kappa", o_x1_kappa)
      call opt_b("-sf_peel_q", bk_q); call opt_b("-sf_peel_t", bk_t); call opt_b("-sf_peel_aa", bk_aa)
      if (o_x /= 0 .and. o_x /= 1) call fatal("-sf_peel_x must be 0 or 1")
      if (o_x == 1 .and. o_xres /= 0) then
        if (my_id == 0) write(*,'(A)') PP//"-sf_peel_x 1 takes diag(B_33) alone: xres ignored"
        o_xres = 0
      endif
      if (my_id == 0 .and. o_v == 2) then
        write(*,'(A,A,A,I0,A,I0,A,F6.3,A,I0,A,I0)') PP//"v2, X = ", merge("X_AS     ", "diag B_33", o_xas /= 0), &
          ", hyper-resistive sandwich ", o_hyper, ", axis patch ", o_as_axis, ", W_comp weight ", o_wcomp, &
          ", K_nu ", o_visc, ", lower-triangular ", o_tri
      else if (my_id == 0) then
        write(*,'(A,I0,A,I0,A,F6.3,A,I0,A,I0)') PP//"v1, X_", o_x, ", xres ", o_xres, ", W_comp weight ", &
          o_wcomp, ", K_nu ", o_visc, ", lower-triangular ", o_tri
        write(*,'(A,3A4,A,3I3,A,I0,A,I0)') PP//"Q / T / A_aa:", bname(bk_q), bname(bk_t), bname(bk_aa), &
          ", V-cycles", ncyc_q, ncyc_t, ncyc_pj, ", outer FGMRES(", o_restart, ") max ", o_maxits
      endif
    end subroutine read_options

  end subroutine peel_build

  !--------------------------------------------------------------------
  !> sol = pair_w^-1 rhs (packed), FGMRES to the build's rtol.
  !--------------------------------------------------------------------
  subroutine peel_solve(rhs, sol)
    Vec :: rhs, sol
    PetscErrorCode :: ierr
    PetscInt :: its
    KSPConvergedReason :: reason
    real*8 :: t0
    t0 = MPI_Wtime()
    call KSPSolve(kout, rhs, sol, ierr)
    t_sum = t_sum + (MPI_Wtime() - t0)
    call KSPGetIterationNumber(kout, its, ierr)
    its_sum = its_sum + int(its); its_max = max(its_max, int(its)); n_solve = n_solve + 1
    call KSPGetConvergedReason(kout, reason, ierr)
    if (reason%v < 0) n_fail = n_fail + 1
  end subroutine peel_solve

  !> One line for the outer pair_w solves, then Q, T and A_aa's; reset.
  subroutine peel_report(my_id)
    integer, intent(in) :: my_id
    if (my_id == 0 .and. n_solve > 0) then
      write(*,'(A,F7.2,A,I0,A,I0,A,F8.3,A)', advance="no") "[Physics PC]   inner pair_w KSP (peel): mean ", &
        dble(its_sum) / dble(n_solve), " its, max ", its_max, " (", n_solve, " solves, ", t_sum, " s)"
      if (n_fail > 0) then
        write(*,'(A,I0,A)') ", WARNING ", n_fail, " not converged"
      else
        write(*,*)
      endif
    endif
    call sf_solver_report(sq, my_id); call sf_solver_report(st, my_id); call sf_solver_report(spj, my_id)
    call sf_solver_reset_counters(sq); call sf_solver_reset_counters(st)
    call sf_solver_reset_counters(spj); call sf_solver_reset_counters(sw)
    n_solve = 0; its_sum = 0; its_max = 0; n_fail = 0; t_sum = 0.d0
  end subroutine peel_report

  !--------------------------------------------------------------------
  !> The block LDU on a packed pair_w vector r (u, omega, psi, j).
  !--------------------------------------------------------------------
  subroutine peel_pc_apply(pc, r, y, ierr)
    PC  :: pc
    Vec :: r, y
    PetscErrorCode :: ierr
    call sf_split_parts(r, [ru, rw, rp, rj], .false.)
    if (o_tri == 0) then
      ! y_a* = A_aa^-1 r_a;  s = r_u - B_24 omega* - B_21 psi* - B_23 j*
      call solve_aa(rw, rp, rj, yw, yp, yj)
      call VecCopy(ru, sv, ierr)
      call sub_mult(g_ctx%B_24, yw, sv)
      call sub_mult(g_ctx%B_21, yp, sv)
      call sub_mult(g_ctx%B_23, yj, sv)
    else
      call VecCopy(ru, sv, ierr)
    endif
    call apply_pinv(sv, yu)
    ! y_a = A_aa^-1 (r_a - A_au y_u), A_au = (B_42; B_12; 0)
    call sub_mult(g_ctx%B_42, yu, rw)
    call sub_mult(g_ctx%B_12, yu, rp)
    call solve_aa(rw, rp, rj, yw, yp, yj)
    call sf_split_parts(y, [yu, yw, yp, yj], .true.)
    ierr = 0
  end subroutine peel_pc_apply

  !> y = -P^-1 s = -T^-1 M_rho Q^-1 s (v1), -N^-1 opz R_hat Q^-1 s (v2), with
  !! the Dirichlet u rows exact.
  subroutine apply_pinv(s, y)
    Vec :: s, y
    PetscErrorCode :: ierr
    call VecCopy(s, t1, ierr)
    call sf_solver_apply(sq, t1, t2, ierr)
    if (o_v == 1) then
      call MatMult(Mm, t2, t1, ierr)
    else if (o_texact /= 0) then                ! v2: opz R_hat, R_hat exact with texact
      call rhat_exact(t2, t1)
      call VecScale(t1, opz, ierr)
    else
      call MatMult(Rh, t2, t1, ierr)
      call VecScale(t1, opz, ierr)
    endif
    if (o_texact /= 0) then
      call KSPSolve(ktex, t1, y, ierr)
    else
      call sf_solver_apply(st, t1, y, ierr)
    endif
    call VecScale(y, -1.d0, ierr)
    call VecPointwiseMult(y, y, imask, ierr)
    call VecPointwiseMult(t1, s, binv, ierr)
    call VecAXPY(y, 1.d0, t1, ierr)
  end subroutine apply_pinv

  !> (omega, psi, j) = A_aa^-1 (a_w, a_p, a_j); a_p, a_j are overwritten
  subroutine solve_aa(aw, ap, aj, ow, op, oj)
    Vec :: aw, ap, aj, ow, op, oj
    PetscErrorCode :: ierr
    if (bk_aa == SF_LU) then
      call sf_solver_apply(sw, aw, ow, ierr)
    else
      call mass_cheb_solve(mw, aw, ow)
    endif
    call sf_split_parts(rpj, [ap, aj], .true.)
    call sf_solver_apply(spj, rpj, ypj, ierr)
    call sf_split_parts(ypj, [op, oj], .false.)
  end subroutine solve_aa

  !> v -= A x (t2 as scratch: never an operand here)
  subroutine sub_mult(A, x, v)
    Mat :: A
    Vec :: x, v
    PetscErrorCode :: ierr
    call MatMult(A, x, t2, ierr)
    call VecAXPY(v, -1.d0, t2, ierr)
  end subroutine sub_mult

  !====================================================================
  ! Operators
  !====================================================================

  !> AIJ copies of the element-assembled BAIJ operators (the GMG reads CSR)
  subroutine to_aij(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    integer :: k
    do k = 1, size(Pc)
      if (first) then
        call MatConvert(g_ctx%P_peel(k), MATMPIAIJ, MAT_INITIAL_MATRIX, Pc(k), ierr)
      else
        call MatConvert(g_ctx%P_peel(k), MATMPIAIJ, MAT_REUSE_MATRIX, Pc(k), ierr)
      endif
    enddo
  end subroutine to_aij

  !> u rows of pair_w with nothing off the diagonal in B_22, B_24, B_21, B_23:
  !! eliminated Dirichlet DOFs and fix_axis_nodes' penalty rows.
  subroutine find_dirichlet_u()
    PetscErrorCode :: ierr
    PetscInt :: rs, re, i, nc, k, nb
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    logical, allocatable :: iso(:)
    integer :: b, ntot, mpierr
    Mat :: bl(4)
    bl = [g_ctx%B_22, g_ctx%B_24, g_ctx%B_21, g_ctx%B_23]
    call MatGetOwnershipRange(g_ctx%B_22, rs, re, ierr)
    allocate(iso(rs:re - 1))
    iso = .true.
    do b = 1, 4
      do i = rs, re - 1
        if (.not. iso(i)) cycle
        call MatGetRow(bl(b), i, nc, cols, vals, ierr)
        do k = 1, nc
          if (vals(k) /= 0.d0 .and. (b > 1 .or. cols(k) /= i)) iso(i) = .false.
        enddo
        call MatRestoreRow(bl(b), i, nc, cols, vals, ierr)
      enddo
    enddo
    nb = count(iso)
    allocate(brows(nb))
    nb = 0
    do i = rs, re - 1
      if (iso(i)) then
        nb = nb + 1; brows(nb) = i
      endif
    enddo
    call MPI_Allreduce(int(nb), ntot, 1, MPI_INTEGER, MPI_SUM, comm_p, mpierr)
    if (me_p == 0) write(*,'(A,I0,A)') "[Physics PC]   peel: ", ntot, &
      " u rows decoupled in pair_w (Dirichlet / axis), solved from its diagonal"
  end subroutine find_dirichlet_u

  !> binv = mask / diag(pair_w's u block), imask = 1 - mask (every build)
  subroutine set_dirichlet_diag()
    PetscErrorCode :: ierr
    Vec :: dW
    PetscScalar, pointer :: dw_(:), bi(:), im(:)
    PetscInt :: rs, k
    if (.not. ready) then
      call MatCreateVecs(g_ctx%B_22, binv, PETSC_NULL_VEC, ierr)
      call VecDuplicate(binv, imask, ierr)
    endif
    call MatCreateVecs(Wop, dW, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(Wop, dW, ierr)           ! packed: the u rows come first
    call MatGetOwnershipRange(g_ctx%B_22, rs, PETSC_NULL_INTEGER, ierr)
    call VecSet(binv, 0.d0, ierr); call VecSet(imask, 1.d0, ierr)
    call VecGetArrayRead(dW, dw_, ierr)
    call VecGetArray(binv, bi, ierr); call VecGetArray(imask, im, ierr)
    do k = 1, size(brows)
      bi(brows(k) - rs + 1) = 1.d0 / dw_(brows(k) - rs + 1)
      im(brows(k) - rs + 1) = 0.d0
    enddo
    call VecRestoreArray(binv, bi, ierr); call VecRestoreArray(imask, im, ierr)
    call VecRestoreArrayRead(dW, dw_, ierr)
    call VecDestroy(dW, ierr)
  end subroutine set_dirichlet_diag

  !> Q = opz K_rho + w W_comp, the Dirichlet rows the identity
  subroutine build_q(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    if (first) then
      call MatDuplicate(Pc(1), MAT_COPY_VALUES, Qm, ierr)
      call MatSetOption(Qm, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatCopy(Pc(1), Qm, SAME_NONZERO_PATTERN, ierr)
    endif
    if (o_wcomp /= 0.d0) call MatAXPY(Qm, o_wcomp, Pc(2), SAME_NONZERO_PATTERN, ierr)
    call dirichlet(Qm)
  end subroutine build_q

  subroutine build_m(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    if (first) then
      call MatDuplicate(Pc(3), MAT_COPY_VALUES, Mm, ierr)
      call MatSetOption(Mm, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatCopy(Pc(3), Mm, SAME_NONZERO_PATTERN, ierr)
    endif
    call dirichlet(Mm)
  end subroutine build_m

  !> TB = B_12^T X B_12. X_0 = diag(x), x = 1 / (diag B_33 - diag(B_13
  !! D^-1 B_31) / opz) with xres (the psi row's resistive Schur diagonal),
  !! zero where that is not positive (fix_axis_nodes' zero-size DOFs).
  !! X_1 = p(D^-1 B_33) D^-1 with p(l) = (4 th - 2 l) / (2 th^2 - de^2) on
  !! [b / kappa, b], b the Gershgorin top: p > 0 on the whole spectrum, so
  !! X_1 is SPD whatever kappa.
  subroutine build_x(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    Vec :: d, c
    PetscScalar, pointer :: dp(:), cp(:)
    PetscInt :: k, n
    real*8 :: dmax, a, b, th, de, den
    integer :: mpierr

    if (first) call MatCreateVecs(g_ctx%B_33, xd, PETSC_NULL_VEC, ierr)
    call VecDuplicate(xd, d, ierr)
    call MatGetDiagonal(g_ctx%B_33, d, ierr)
    call VecGetLocalSize(d, n, ierr)

    if (o_x == 0) then
      call VecCopy(d, xd, ierr)
      if (o_xres /= 0) then
        ! RB31 = D^-1 B_31, Cres = B_13 RB31
        call VecDuplicate(d, c, ierr)
        call VecCopy(d, c, ierr)
        call guarded_reciprocal(c)
        if (first) then
          call MatDuplicate(g_ctx%B_31, MAT_COPY_VALUES, RB31, ierr)
        else
          call MatCopy(g_ctx%B_31, RB31, SAME_NONZERO_PATTERN, ierr)
        endif
        call MatDiagonalScale(RB31, c, PETSC_NULL_VEC, ierr)
        if (first) then
          call MatMatMult(g_ctx%B_13, RB31, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, Cres, ierr)
        else
          call MatMatMult(g_ctx%B_13, RB31, MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, Cres, ierr)
        endif
        call MatGetDiagonal(Cres, c, ierr)
        call VecAXPY(xd, -1.d0 / opz, c, ierr)
        call VecDestroy(c, ierr)
      endif
      call guarded_reciprocal(xd)
      ! B12s = diag(sqrt x) B_12
      call VecSqrtAbs(xd, ierr)
      if (first) then
        call MatDuplicate(g_ctx%B_12, MAT_COPY_VALUES, B12s, ierr)
      else
        call MatCopy(g_ctx%B_12, B12s, SAME_NONZERO_PATTERN, ierr)
      endif
      call MatDiagonalScale(B12s, xd, PETSC_NULL_VEC, ierr)
      if (first) then
        call MatTransposeMatMult(B12s, B12s, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, TB, ierr)
      else
        call MatTransposeMatMult(B12s, B12s, MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, TB, ierr)
      endif
      call VecPointwiseMult(xd, xd, xd, ierr)        ! back to x for the checks
    else
      ! Gershgorin top of D^-1 B_33 (rows of |B_33| over the diagonal)
      call VecCopy(d, xd, ierr)
      call guarded_reciprocal(xd)
      b = gershgorin_top(g_ctx%B_33, xd)
      a = b / o_x1_kappa
      th = 0.5d0 * (a + b); de = 0.5d0 * (b - a); den = 2.d0 * th**2 - de**2
      ! Xm = (4 th D^-1 - 2 D^-1 B_33 D^-1) / den
      if (first) then
        call MatDuplicate(g_ctx%B_33, MAT_COPY_VALUES, Xm, ierr)
      else
        call MatCopy(g_ctx%B_33, Xm, SAME_NONZERO_PATTERN, ierr)
      endif
      call MatDiagonalScale(Xm, xd, xd, ierr)
      call MatScale(Xm, -2.d0 / den, ierr)
      call VecDuplicate(xd, c, ierr)
      call VecCopy(xd, c, ierr); call VecScale(c, 4.d0 * th / den, ierr)
      call MatDiagonalSet(Xm, c, ADD_VALUES, ierr)
      call VecDestroy(c, ierr)
      if (first) then
        call MatMatMult(Xm, g_ctx%B_12, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, XmB12, ierr)
        call MatTransposeMatMult(g_ctx%B_12, XmB12, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, TB, ierr)
      else
        call MatMatMult(Xm, g_ctx%B_12, MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, XmB12, ierr)
        call MatTransposeMatMult(g_ctx%B_12, XmB12, MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, TB, ierr)
      endif
      if (me_p == 0 .and. first) write(*,'(A,ES10.3,A,F6.1)') &
        "[Physics PC]   peel: X_1 on [b / kappa, b], Gershgorin b = ", b, ", kappa ", o_x1_kappa
    endif
    call VecDestroy(d, ierr)

  contains

    !> v <- 1 / v where v > 1e-14 max v, else 0
    subroutine guarded_reciprocal(v)
      Vec :: v
      PetscScalar, pointer :: p(:)
      call VecGetArray(v, p, ierr)
      dmax = 0.d0
      if (n > 0) dmax = maxval(p)
      call MPI_Allreduce(MPI_IN_PLACE, dmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm_p, mpierr)
      do k = 1, n
        if (p(k) > 1.d-14 * dmax) then
          p(k) = 1.d0 / p(k)
        else
          p(k) = 0.d0
        endif
      enddo
      call VecRestoreArray(v, p, ierr)
    end subroutine guarded_reciprocal

  end subroutine build_x

  real*8 function gershgorin_top(A, dinv)
    Mat :: A
    Vec :: dinv
    PetscErrorCode :: ierr
    PetscInt :: rs, re, i, nc
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:), di(:)
    integer :: mpierr
    real*8 :: g
    g = 0.d0
    call MatGetOwnershipRange(A, rs, re, ierr)
    call VecGetArrayRead(dinv, di, ierr)
    do i = rs, re - 1
      call MatGetRow(A, i, nc, cols, vals, ierr)
      g = max(g, di(i - rs + 1) * sum(abs(vals(1:nc))))
      call MatRestoreRow(A, i, nc, cols, vals, ierr)
    enddo
    call VecRestoreArrayRead(dinv, di, ierr)
    call MPI_Allreduce(g, gershgorin_top, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm_p, mpierr)
  end function gershgorin_top

  !--------------------------------------------------------------------
  !> v2's X_AS = sum_p R_p^T (R_p B R_p^T)^-1 R_p, once per mesh (B is
  !! geometry-only). Patch p = the DOFs (all slots) of one element without
  !! B's decoupled rows (Dirichlet / penalty rows get 1 / diag, a zero
  !! diagonal 0); with as_axis every element touching the axis is one patch.
  !! R_p B R_p^T is the GLOBAL B restricted to the patch. A patch belongs to
  !! the rank owning its first row; its block comes from one sequential
  !! submatrix over the rank's patch rows and is inverted by Cholesky.
  !! Also N_c, the greedy colouring of the patch graph (patches adjacent
  !! where B couples any of their DOFs): lambda_max(X_AS B) <= N_c.
  !--------------------------------------------------------------------
  subroutine build_xas(B, X, label, verbose)
    use mod_parameters, only: n_tor, n_degrees, n_vertex_max
    use nodes_elements, only: node_list, element_list
    Mat :: B, X
    character(len=*), intent(in) :: label
    logical, intent(in) :: verbose
    PetscErrorCode :: ierr
    PetscInt :: rs, re, nloc, ng, i, nc, k, n, nu
    PetscInt, parameter :: one = 1
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: dnz(:), onz(:), urows(:), prow(:), lp(:)
    integer, allocatable :: flag(:), pel(:), pst(:), plen(:), pdof(:), rstart(:), cnt(:)
    integer, allocatable :: vst(:), vel(:), colour(:), mark(:)
    real*8, allocatable :: dg(:), blk(:, :)
    integer :: ne, nn, e, e2, v, w, kd, m, np, me, mpierr, p, q, axp, info, a, j, npat, nfail, r, c, naxd
    IS :: isu
    Mat, pointer :: sub(:)
    external :: dpotrf, dpotri

    call MPI_Comm_size(comm_p, np, mpierr); call MPI_Comm_rank(comm_p, me, mpierr)
    call MatGetOwnershipRange(B, rs, re, ierr)
    call MatGetSize(B, ng, PETSC_NULL_INTEGER, ierr)
    nloc = re - rs
    allocate(rstart(0:np))
    call MPI_Allgather(int(rs), 1, MPI_INTEGER, rstart, 1, MPI_INTEGER, comm_p, mpierr)
    rstart(np) = int(ng)

    !--- decoupled rows: 1 = nothing off the diagonal, 2 = and a zero diagonal
    allocate(flag(0:ng - 1), dg(rs:re - 1))
    flag = 0; dg = 0.d0
    do i = rs, re - 1
      call MatGetRow(B, i, nc, cols, vals, ierr)
      a = 1
      do k = 1, nc
        if (cols(k) == i) then
          dg(i) = vals(k)
        else if (vals(k) /= 0.d0) then
          a = 0
        endif
      enddo
      call MatRestoreRow(B, i, nc, cols, vals, ierr)
      if (a == 1) flag(i) = merge(2, 1, dg(i) == 0.d0)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, flag, int(ng), MPI_INTEGER, MPI_SUM, comm_p, mpierr)

    !--- patches: one per element, the axis elements merged into patch 1
    ne = element_list%n_elements
    allocate(pel(ne))
    axp = 0
    if (o_as_axis /= 0) then
      do e = 1, ne
        if (any([(node_list%node(element_list%element(e)%vertex(v))%axis_node, v = 1, n_vertex_max)])) axp = 1
      enddo
    endif
    npat = axp
    do e = 1, ne
      if (axp == 1 .and. any([(node_list%node(element_list%element(e)%vertex(v))%axis_node, &
                               v = 1, n_vertex_max)])) then
        pel(e) = 1
      else
        npat = npat + 1; pel(e) = npat
      endif
    enddo
    allocate(cnt(npat), pst(npat + 1), plen(npat))
    cnt = 0
    do e = 1, ne
      cnt(pel(e)) = cnt(pel(e)) + n_vertex_max * n_degrees * n_tor
    enddo
    pst(1) = 1
    do p = 1, npat
      pst(p + 1) = pst(p) + cnt(p)
    enddo
    allocate(pdof(pst(npat + 1) - 1))
    plen = 0
    do e = 1, ne
      p = pel(e)
      do v = 1, n_vertex_max
        do kd = 1, n_degrees
          do m = 0, n_tor - 1
            r = (node_list%node(element_list%element(e)%vertex(v))%index(kd) - 1) * n_tor + m
            if (flag(r) /= 0) cycle
            pdof(pst(p) + plen(p)) = r; plen(p) = plen(p) + 1
          enddo
        enddo
      enddo
    enddo
    do p = 1, npat                                ! sort, unique
      call sort_unique(pdof(pst(p):pst(p) + plen(p) - 1), plen(p))
    enddo
    naxd = 0
    if (axp == 1) naxd = plen(1)

    !--- X: preallocation from the patch sums (an upper bound)
    allocate(dnz(nloc), onz(nloc))
    dnz = 1; onz = 0
    do p = 1, npat
      do j = pst(p), pst(p) + plen(p) - 1
        r = pdof(j)
        if (r < rs .or. r >= re) cycle
        do k = pst(p), pst(p) + plen(p) - 1
          if (pdof(k) >= rs .and. pdof(k) < re) then
            dnz(r - rs + 1) = dnz(r - rs + 1) + 1
          else
            onz(r - rs + 1) = onz(r - rs + 1) + 1
          endif
        enddo
      enddo
    enddo
    dnz = min(dnz, nloc); onz = min(onz, ng - nloc)
    call MatCreate(comm_p, X, ierr)
    call MatSetSizes(X, nloc, nloc, ng, ng, ierr)
    call MatSetType(X, MATMPIAIJ, ierr)
    call MatMPIAIJSetPreallocation(X, PETSC_DEFAULT_INTEGER, dnz, PETSC_DEFAULT_INTEGER, onz, ierr)
    call MatSetOption(X, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

    !--- this rank's patches (owner of the first row): their rows, one submatrix
    allocate(mark(npat))
    mark = 0; nu = 0
    do p = 1, npat
      if (plen(p) == 0) cycle
      if (owner_of(pdof(pst(p))) == me) then
        mark(p) = 1; nu = nu + plen(p)
      endif
    enddo
    allocate(urows(max(nu, 1)))
    nu = 0
    do p = 1, npat
      if (mark(p) == 0) cycle
      urows(nu + 1:nu + plen(p)) = pdof(pst(p):pst(p) + plen(p) - 1); nu = nu + plen(p)
    enddo
    block
      integer, allocatable :: tmp(:)
      tmp = int(urows(1:nu))
      call sort_unique(tmp, a); nu = a
      urows(1:nu) = tmp(1:nu)
    end block
    call ISCreateGeneral(PETSC_COMM_SELF, nu, urows, PETSC_COPY_VALUES, isu, ierr)
    call MatCreateSubMatrices(B, one, [isu], [isu], MAT_INITIAL_MATRIX, sub, ierr)
    nfail = 0
    do p = 1, npat
      if (mark(p) == 0) cycle
      n = plen(p)
      allocate(prow(n), lp(n), blk(n, n))
      prow = pdof(pst(p):pst(p) + n - 1)
      do j = 1, int(n)
        lp(j) = findpos(urows(1:nu), prow(j))
      enddo
      call MatGetValues(sub(1), n, lp, n, lp, blk, ierr)
      call dpotrf('U', int(n), blk, int(n), info)
      if (info == 0) then
        call dpotri('U', int(n), blk, int(n), info)
        do c = 1, int(n)
          do r = c + 1, int(n)
            blk(r, c) = blk(c, r)
          enddo
        enddo
        call MatSetValues(X, n, prow, n, prow, blk, ADD_VALUES, ierr)
      else
        nfail = nfail + 1
      endif
      deallocate(prow, lp, blk)
    enddo
    do i = rs, re - 1
      if (flag(i) == 1) call MatSetValue(X, i, i, 1.d0 / dg(i), ADD_VALUES, ierr)
      if (flag(i) == 2) call MatSetValue(X, i, i, 0.d0, ADD_VALUES, ierr)
    enddo
    call MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY, ierr)
    call MatDestroySubMatrices(one, sub, ierr)
    call ISDestroy(isu, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, comm_p, mpierr)

    !--- N_c: greedy colouring, patches adjacent through a shared element of
    !--- their vertices (B's couplings), computed redundantly on every rank
    nn = node_list%n_nodes
    allocate(vst(nn + 1))
    vst = 0
    do e = 1, ne
      do v = 1, n_vertex_max
        a = element_list%element(e)%vertex(v)
        vst(a + 1) = vst(a + 1) + 1
      enddo
    enddo
    vst(1) = 1
    do a = 1, nn
      vst(a + 1) = vst(a + 1) + vst(a)
    enddo
    allocate(vel(vst(nn + 1) - 1))
    cnt = 0
    deallocate(cnt); allocate(cnt(nn)); cnt = 0
    do e = 1, ne
      do v = 1, n_vertex_max
        a = element_list%element(e)%vertex(v)
        vel(vst(a) + cnt(a)) = e; cnt(a) = cnt(a) + 1
      enddo
    enddo
    allocate(colour(npat))
    colour = 0; mark = 0
    as_ncol = 0
    do e = 1, ne                       ! patches in element order
      p = pel(e)
      if (colour(p) /= 0) cycle
      ! mark the neighbours' colours: elements at the vertices coupled to p's
      ! (the axis patch: all its elements, the others: e itself)
      do e2 = merge(1, e, p == axp), merge(ne, e, p == axp)
        if (pel(e2) /= p) cycle
        do v = 1, n_vertex_max
          a = element_list%element(e2)%vertex(v)
          do j = vst(a), vst(a + 1) - 1
            do w = 1, n_vertex_max
              r = element_list%element(vel(j))%vertex(w)
              do k = vst(r), vst(r + 1) - 1
                q = pel(vel(k))
                if (colour(q) > 0) mark(colour(q)) = p
              enddo
            enddo
          enddo
        enddo
      enddo
      c = 1
      do while (c <= npat)
        if (mark(c) /= p) exit
        c = c + 1
      enddo
      colour(p) = c
      as_ncol = max(as_ncol, c)
    enddo
    as_npatch = npat
    if (me_p == 0 .and. verbose) write(*,'(A,A,A,I0,A,I0,A,I0,A,I0,A)') "[Physics PC]   peel: X_AS of ", label, ", ", npat, &
      " patches (axis patch ", naxd, " DOFs), colouring bound N_c = ", as_ncol, ", ", nfail, " Cholesky failures"

  contains

    integer function owner_of(row)
      integer, intent(in) :: row
      integer :: lo, hi, mid
      lo = 0; hi = np - 1
      do while (lo < hi)
        mid = (lo + hi + 1) / 2
        if (rstart(mid) <= row) then
          lo = mid
        else
          hi = mid - 1
        endif
      enddo
      owner_of = lo
    end function owner_of

    !> 0-based position of x in the sorted list l
    PetscInt function findpos(l, x)
      PetscInt, intent(in) :: l(:), x
      integer :: lo, hi, mid
      lo = 1; hi = size(l)
      do while (lo < hi)
        mid = (lo + hi) / 2
        if (l(mid) < x) then
          lo = mid + 1
        else
          hi = mid
        endif
      enddo
      findpos = lo - 1
    end function findpos

  end subroutine build_xas

  !> l(1:n) sorted ascending without repeats, n updated (insertion sort: the
  !! lists are element patches, the axis patch at most a few thousand)
  subroutine sort_unique(l, n)
    integer, intent(inout) :: l(:)
    integer, intent(inout) :: n
    integer :: i, j, t, k
    n = size(l)
    do i = 2, n
      t = l(i); j = i - 1
      do while (j >= 1)
        if (l(j) <= t) exit
        l(j + 1) = l(j); j = j - 1
      enddo
      l(j + 1) = t
    enddo
    k = min(n, 1)
    do i = 2, n
      if (l(i) /= l(k)) then
        k = k + 1; l(k) = l(i)
      endif
    enddo
    n = k
  end subroutine sort_unique

  !> v2: R_hat = opz M_rho + theta dt K_eta [+ theta dt K_w X K_w], the
  !! Dirichlet rows the identity. Only ever multiplied.
  subroutine build_rhat(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    if (o_hyper /= 0) then
      if (first) then
        call MatMatMult(Pc(6), xmat(), MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, KwX, ierr)
        call MatMatMult(KwX, Pc(6), MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, Hm, ierr)
      else
        call MatMatMult(Pc(6), xmat(), MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, KwX, ierr)
        call MatMatMult(KwX, Pc(6), MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, Hm, ierr)
      endif
    endif
    if (first) then
      call MatDuplicate(Pc(3), MAT_COPY_VALUES, Rh, ierr)
      call MatScale(Rh, opz, ierr)
      if (o_hyper /= 0) call MatAXPY(Rh, thdt, Hm, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatSetOption(Rh, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatZeroEntries(Rh, ierr)
      call MatAXPY(Rh, opz, Pc(3), SUBSET_NONZERO_PATTERN, ierr)
      if (o_hyper /= 0) call MatAXPY(Rh, thdt, Hm, SUBSET_NONZERO_PATTERN, ierr)
    endif
    call MatAXPY(Rh, o_eta_w * thdt, Pc(5), SUBSET_NONZERO_PATTERN, ierr)
    call dirichlet(Rh)
  end subroutine build_rhat

  !> v2: N = opz R_hat + B_12^T X B_12 + opz theta dt K_nu (TB by the X_AS
  !! products, or X_0 = diag(B_33)^-1 with -sf_peel_xas 0)
  subroutine build_n(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    if (o_xas /= 0) then
      if (first) then
        call MatMatMult(Xas, g_ctx%B_12, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, XasB12, ierr)
        call MatTransposeMatMult(g_ctx%B_12, XasB12, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, TB, ierr)
      else
        call MatMatMult(Xas, g_ctx%B_12, MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, XasB12, ierr)
        call MatTransposeMatMult(g_ctx%B_12, XasB12, MAT_REUSE_MATRIX, PETSC_DETERMINE_REAL, TB, ierr)
      endif
    else
      o_xres = 0
      call build_x(first)                       ! X_0, TB = B_12^T X_0 B_12
    endif
    if (o_visc >= 2) call build_visc()
    if (first) then
      call MatDuplicate(TB, MAT_COPY_VALUES, Tm, ierr)
      call MatAXPY(Tm, opz, Rh, DIFFERENT_NONZERO_PATTERN, ierr)
      if (o_visc == 1) call MatAXPY(Tm, opz * thdt, Pc(4), DIFFERENT_NONZERO_PATTERN, ierr)
      if (o_visc >= 2) then
        call MatAXPY(Tm, 0.5d0 * thdt, Vc, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatAXPY(Tm, 0.5d0 * thdt, Vt, DIFFERENT_NONZERO_PATTERN, ierr)
      endif
      call MatSetOption(Tm, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatZeroEntries(Tm, ierr)
      call MatAXPY(Tm, 1.d0, TB, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(Tm, opz, Rh, SUBSET_NONZERO_PATTERN, ierr)
      if (o_visc == 1) call MatAXPY(Tm, opz * thdt, Pc(4), SUBSET_NONZERO_PATTERN, ierr)
      if (o_visc >= 2) then
        call MatAXPY(Tm, 0.5d0 * thdt, Vc, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(Tm, 0.5d0 * thdt, Vt, SUBSET_NONZERO_PATTERN, ierr)
      endif
    endif
    call dirichlet(Tm)
  end subroutine build_n

  !> v2 viscosity, resistive factor included: theta dt sym(R_v M_rho^-1 K_nu),
  !! M_rho^-1 by X_AS(M_rho) (rebuilt with rho0), R_v = R_hat (visc 2) or
  !! opz M_rho + theta dt K_eta (visc 3, no hyper-resistive k^6 part), so that
  !! Q (opz R_hat)^-1 N carries theta dt K_rho M_rho^-1 K_nu as S does. The
  !! design's opz theta dt K_nu (visc 1) is divided by R_hat's resistive
  !! factor, up to ~600x on viscosity-dominated modes here.
  subroutine build_visc()
    PetscErrorCode :: ierr
    if (vis_built) then
      call MatDestroy(Xmr, ierr); call MatDestroy(XmK, ierr)
      call MatDestroy(Vc, ierr); call MatDestroy(Vt, ierr)
    endif
    call build_xas(Mm, Xmr, "M_rho", .not. vis_built)
    call MatMatMult(Xmr, Pc(4), MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, XmK, ierr)
    if (o_visc == 3) then
      if (vis_built) then
        call MatZeroEntries(Rv, ierr)
        call MatAXPY(Rv, opz, Pc(3), SAME_NONZERO_PATTERN, ierr)
      else
        call MatDuplicate(Pc(3), MAT_COPY_VALUES, Rv, ierr)
        call MatScale(Rv, opz, ierr)
        call MatSetOption(Rv, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
      endif
      call MatAXPY(Rv, o_eta_w * thdt, Pc(5), SAME_NONZERO_PATTERN, ierr)
      call dirichlet(Rv)
      call MatMatMult(Rv, XmK, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, Vc, ierr)
    else
      call MatMatMult(Rh, XmK, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, Vc, ierr)
    endif
    call MatTranspose(Vc, MAT_INITIAL_MATRIX, Vt, ierr)
    vis_built = .true.
  end subroutine build_visc

  !> The X of the hyper-resistive sandwich: X_AS, or diag(B_33)^-1 as a matrix
  function xmat() result(A)
    Mat :: A
    PetscErrorCode :: ierr
    Vec :: d
    Mat, save :: Xd
    logical, save :: have = .false.
    if (o_xas /= 0) then
      A = Xas
      return
    endif
    if (.not. have) then
      call MatCreateVecs(g_ctx%B_33, d, PETSC_NULL_VEC, ierr)
      call MatGetDiagonal(g_ctx%B_33, d, ierr)
      call VecReciprocal(d, ierr)
      call MatDuplicate(g_ctx%B_33, MAT_DO_NOT_COPY_VALUES, Xd, ierr)
      call MatZeroEntries(Xd, ierr)
      call MatDiagonalSet(Xd, d, INSERT_VALUES, ierr)
      call VecDestroy(d, ierr)
      have = .true.
    endif
    A = Xd
  end function xmat

  !> texact v2: y = R_hat x with the exact X = B_33^-1 (MUMPS) in the sandwich
  subroutine rhat_exact(x, y)
    Vec :: x, y
    PetscErrorCode :: ierr
    Vec :: va, vb
    call VecDuplicate(x, va, ierr); call VecDuplicate(x, vb, ierr)
    call MatMult(Pc(3), x, y, ierr)
    call VecScale(y, opz, ierr)
    call MatMult(Pc(5), x, va, ierr)
    call VecAXPY(y, o_eta_w * thdt, va, ierr)
    if (o_hyper /= 0) then
      call MatMult(Pc(6), x, va, ierr)
      call sf_solver_apply(lu33, va, vb, ierr)
      call MatMult(Pc(6), vb, va, ierr)
      call VecAXPY(y, thdt, va, ierr)
    endif
    call VecPointwiseMult(y, y, imask, ierr)
    call VecDestroy(va, ierr); call VecDestroy(vb, ierr)
  end subroutine rhat_exact

  !> T = M_rho + opz^-2 TB + theta dt / opz K_nu, patterns united on the first build
  subroutine build_t(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    if (first) then
      call MatDuplicate(Pc(3), MAT_COPY_VALUES, Tm, ierr)
      call MatAXPY(Tm, 1.d0 / opz**2, TB, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatSetOption(Tm, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatZeroEntries(Tm, ierr)
      call MatAXPY(Tm, 1.d0, Pc(3), SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(Tm, 1.d0 / opz**2, TB, SUBSET_NONZERO_PATTERN, ierr)
    endif
    if (o_visc /= 0) call MatAXPY(Tm, thdt / opz, Pc(4), SUBSET_NONZERO_PATTERN, ierr)
    call dirichlet(Tm)
  end subroutine build_t

  !> A_aa's (psi, j) block, pair_w's lower right: [[opz B_33, B_13], [B_31, B_33]]
  subroutine build_pj(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    if (first) then
      call MatDuplicate(g_ctx%B_33, MAT_COPY_VALUES, mpsi, ierr)
    else
      call MatCopy(g_ctx%B_33, mpsi, SAME_NONZERO_PATTERN, ierr)
    endif
    call MatScale(mpsi, opz, ierr)
    call pack_pair_aij(mpsi, g_ctx%B_13, g_ctx%B_31, g_ctx%B_33, PJm, pj_packed, comm_p)
  end subroutine build_pj

  !> Rows and columns of the decoupled u DOFs: the identity
  subroutine dirichlet(A)
    Mat :: A
    PetscErrorCode :: ierr
    call MatZeroRowsColumns(A, size(brows, kind=kind(brows)), brows, 1.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
  end subroutine dirichlet

  subroutine create_work_vectors()
    PetscErrorCode :: ierr
    call MatCreateVecs(g_ctx%B_22, ru, PETSC_NULL_VEC, ierr)
    call VecDuplicate(ru, rw, ierr); call VecDuplicate(ru, rp, ierr); call VecDuplicate(ru, rj, ierr)
    call VecDuplicate(ru, yu, ierr); call VecDuplicate(ru, yw, ierr); call VecDuplicate(ru, yp, ierr)
    call VecDuplicate(ru, yj, ierr); call VecDuplicate(ru, sv, ierr)
    call VecDuplicate(ru, t1, ierr); call VecDuplicate(ru, t2, ierr)
    call MatCreateVecs(PJm, rpj, ypj, ierr)
  end subroutine create_work_vectors

  !> MUMPS on A_aa's blocks (once per build): B_44 and the scaled (psi, j)
  subroutine ensure_exact_aa()
    if (exact_aa) return
    call sf_solver_setup(lw, g_ctx%B_44, SF_LU, "peel exact A_aa omega", comm_p, me_p, 1.d-12, &
                         gmg_inst=0, nfields=1, smoother=0, maxits=1)
    call sf_solver_setup(lpj, PJm, SF_LU, "peel exact A_aa (psi, j)", comm_p, me_p, 1.d-12, &
                         gmg_inst=0, nfields=2, smoother=0, maxits=1)
    lpj%scaled = .true.; lpj%dscale = spj%dscale
    exact_aa = .true.
  end subroutine ensure_exact_aa

  !> texact: T_ex = M_rho + opz^-2 B_12^T X_ex B_12 + theta dt / opz K_nu,
  !! X_ex z = opz [(psi, j) block^-1 (z, 0)]_psi, matrix-free, solved by
  !! GMRES to 1e-10 around T_X's own solver
  subroutine setup_texact(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    PetscInt :: nl, ng
    PC :: pc
    call ensure_exact_aa()
    if (o_v == 2) call sf_solver_setup(lu33, g_ctx%B_33, SF_LU, "peel exact X = B_33^-1", comm_p, me_p, &
                                       1.d-12, gmg_inst=0, nfields=1, smoother=0, maxits=1)
    if (o_v == 2 .and. o_visc >= 2) call sf_solver_setup(luM, Mm, SF_LU, "peel exact M_rho^-1", comm_p, me_p, &
                                                        1.d-12, gmg_inst=0, nfields=1, smoother=0, maxits=1)
    if (.not. first) return
    call MatGetLocalSize(g_ctx%B_22, nl, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(g_ctx%B_22, ng, PETSC_NULL_INTEGER, ierr)
    call MatCreateShell(comm_p, nl, nl, ng, ng, PETSC_NULL_INTEGER, Tex, ierr)
    call MatShellSetOperation(Tex, MATOP_MULT, tex_mult, ierr)
    call KSPCreate(comm_p, ktex, ierr)
    call KSPSetType(ktex, KSPGMRES, ierr)
    call KSPGMRESSetRestart(ktex, 100, ierr)
    call KSPSetPCSide(ktex, PC_RIGHT, ierr)
    call KSPSetTolerances(ktex, 1.d-10, 1.d-50, 1.d10, 300, ierr)
    call KSPSetOperators(ktex, Tex, Tex, ierr)
    call KSPGetPC(ktex, pc, ierr)
    call PCSetType(pc, PCSHELL, ierr)
    call PCShellSetApply(pc, tex_pc, ierr)
    if (me_p == 0) write(*,'(A)') "[Physics PC]   peel: texact, "//merge("T", "N", o_v == 1)// &
      " with the exact X (MUMPS), GMRES to 1e-10"
  end subroutine setup_texact

  subroutine tex_mult(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    Vec :: xm, va, vb
    call VecDuplicate(x, xm, ierr); call VecDuplicate(x, va, ierr); call VecDuplicate(x, vb, ierr)
    call VecPointwiseMult(xm, x, imask, ierr)
    if (o_v == 2) then
      ! N_ex xm = opz R_hat_ex xm + B_12^T B_33^-1 B_12 xm + opz theta dt K_nu xm
      call rhat_exact(xm, y)
      call VecScale(y, opz, ierr)
      call MatMult(g_ctx%B_12, xm, va, ierr)
      call sf_solver_apply(lu33, va, vb, ierr)
      call MatMultTranspose(g_ctx%B_12, vb, va, ierr)
      call VecAXPY(y, 1.d0, va, ierr)
      if (o_visc == 1) then
        call MatMult(Pc(4), xm, va, ierr)
        call VecAXPY(y, opz * thdt, va, ierr)
      else if (o_visc >= 2) then
        ! theta dt sym(R_ex M_rho^-1 K_nu), M_rho^-1 by MUMPS
        call MatMult(Pc(4), xm, va, ierr)
        call sf_solver_apply(luM, va, vb, ierr)
        call rhat_exact(vb, va)
        call VecAXPY(y, 0.5d0 * thdt, va, ierr)
        call rhat_exact(xm, va)
        call sf_solver_apply(luM, va, vb, ierr)
        call MatMult(Pc(4), vb, va, ierr)
        call VecAXPY(y, 0.5d0 * thdt, va, ierr)
      endif
      call VecPointwiseMult(y, y, imask, ierr)
      call VecAXPY(y, 1.d0, x, ierr)
      call VecPointwiseMult(va, x, imask, ierr)
      call VecAXPY(y, -1.d0, va, ierr)
      call VecDestroy(xm, ierr); call VecDestroy(va, ierr); call VecDestroy(vb, ierr)
      ierr = 0
      return
    endif
    ! vb = B_12^T X_ex B_12 xm
    call MatMult(g_ctx%B_12, xm, va, ierr)
    call VecSet(vb, 0.d0, ierr)
    call sf_split_parts(rpj, [va, vb], .true.)
    call sf_solver_apply(lpj, rpj, ypj, ierr)
    call sf_split_parts(ypj, [va, vb], .false.)
    call MatMultTranspose(g_ctx%B_12, va, vb, ierr)
    call VecScale(vb, 1.d0 / opz, ierr)            ! opz X_ex / opz^2
    call MatMult(Pc(3), xm, y, ierr)
    call VecAXPY(y, 1.d0, vb, ierr)
    if (o_visc /= 0) then
      call MatMult(Pc(4), xm, va, ierr)
      call VecAXPY(y, thdt / opz, va, ierr)
    endif
    call VecPointwiseMult(y, y, imask, ierr)
    call VecAXPY(y, 1.d0, x, ierr)                ! Dirichlet rows: the identity
    call VecPointwiseMult(va, x, imask, ierr)
    call VecAXPY(y, -1.d0, va, ierr)
    call VecDestroy(xm, ierr); call VecDestroy(va, ierr); call VecDestroy(vb, ierr)
    ierr = 0
  end subroutine tex_mult

  subroutine tex_pc(pc, r, y, ierr)
    PC  :: pc
    Vec :: r, y
    PetscErrorCode :: ierr
    Vec :: rr
    call VecDuplicate(r, rr, ierr)
    call VecCopy(r, rr, ierr)
    call sf_solver_apply(st, rr, y, ierr)
    call VecDestroy(rr, ierr)
    ierr = 0
  end subroutine tex_pc

  !====================================================================
  ! Validation (Stage 0 / Stage 1 of the design note)
  !====================================================================

  !> Stage 0: is B_12^T X B_12 the bending's sign (B_23 = -B_12^T away from
  !! the boundary), are Q, M_rho, T positive, and how large are the terms
  !! the design treats as small (W_comp against opz K_rho; the resistive
  !! change of X).
  subroutine stage0_check()
    PetscErrorCode :: ierr
    Mat :: S
    real*8 :: n23, nd, rq(3), rqmin(3), r
    Vec :: v, av, d1, d2
    integer :: k, j
    PetscScalar, pointer :: p1(:), p2(:)
    real*8 :: wmax, wmean, cnt
    integer :: mpierr

    call MatTranspose(g_ctx%B_12, MAT_INITIAL_MATRIX, S, ierr)
    call MatAXPY(S, 1.d0, g_ctx%B_23, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatNorm(S, NORM_FROBENIUS, nd, ierr)
    call MatNorm(g_ctx%B_23, NORM_FROBENIUS, n23, ierr)
    call MatDestroy(S, ierr)
    if (me_p == 0) write(*,'(A,ES10.3)') "[Physics PC]   peel check: |B_23 + B_12^T|_F / |B_23|_F = ", &
      nd / max(n23, 1.d-300)

    ! the spectrum of D^-1 B_33 (CG + Jacobi, Lanczos extremes): the design's
    ! bound on X_0, [0.0625, 3.0625] on a uniform grid
    block
      KSP :: kc
      PC  :: pcj
      Vec :: bb, xx
      PetscReal :: emax, emin
      call KSPCreate(comm_p, kc, ierr)
      call KSPSetType(kc, KSPCG, ierr)
      call KSPSetOperators(kc, g_ctx%B_33, g_ctx%B_33, ierr)
      call KSPGetPC(kc, pcj, ierr)
      call PCSetType(pcj, PCJACOBI, ierr)
      call KSPSetComputeSingularValues(kc, PETSC_TRUE, ierr)
      call KSPSetTolerances(kc, 1.d-30, 1.d-50, 1.d10, 200, ierr)
      call KSPSetNormType(kc, KSP_NORM_UNPRECONDITIONED, ierr)
      call MatCreateVecs(g_ctx%B_33, xx, bb, ierr)
      call VecSetRandom(bb, PETSC_NULL_RANDOM, ierr)
      call KSPSolve(kc, bb, xx, ierr)
      call KSPComputeExtremeSingularValues(kc, emax, emin, ierr)
      if (me_p == 0) write(*,'(A,2ES11.3,A,ES10.3)') "[Physics PC]   peel check: spectrum of D^-1 B_33 ~ [", &
        emin, emax, "], cond ", emax / max(emin, 1.d-300)
      call KSPDestroy(kc, ierr); call VecDestroy(bb, ierr); call VecDestroy(xx, ierr)
    end block

    ! Rayleigh quotients on random vectors, min over 5 draws
    call VecDuplicate(ru, v, ierr); call VecDuplicate(ru, av, ierr)
    rqmin = huge(1.d0)
    do k = 1, 5
      call VecSetRandom(v, PETSC_NULL_RANDOM, ierr)
      call VecShift(v, -0.5d0, ierr)
      call VecDot(v, v, r, ierr)
      call MatMult(Qm, v, av, ierr); call VecDot(v, av, rq(1), ierr)
      call MatMult(Mm, v, av, ierr); call VecDot(v, av, rq(2), ierr)
      call MatMult(Tm, v, av, ierr); call VecDot(v, av, rq(3), ierr)
      do j = 1, 3
        rqmin(j) = min(rqmin(j), rq(j) / r)
      enddo
    enddo
    if (me_p == 0) write(*,'(A,3ES11.3)') "[Physics PC]   peel check: min Rayleigh quotient Q, M_rho, T: ", rqmin
    call VecDestroy(v, ierr); call VecDestroy(av, ierr)

    ! W_comp / (opz K_rho) and B_12^T X B_12 / (opz^2 M_rho) on the diagonals
    call VecDuplicate(ru, d1, ierr); call VecDuplicate(ru, d2, ierr)
    call ratio_report(Pc(2), Pc(1), 1.d0, "diag W_comp / diag(opz K_rho)")
    call ratio_report(TB, Pc(3), 1.d0 / opz**2, "diag(B_12^T X B_12) / (opz^2 diag M_rho)")
    call VecDestroy(d1, ierr); call VecDestroy(d2, ierr)

    ! v2: the resistive terms of R_hat against its mass (~ 6.5 E, 6.5 E_4 in
    ! the LFA's grid numbers), and M1, the spectrum of X_AS B_33
    if (o_v == 2) then
      call VecDuplicate(ru, d1, ierr); call VecDuplicate(ru, d2, ierr)
      call ratio_report(Pc(5), Pc(3), thdt / opz, "theta dt diag K_eta / (opz diag M_rho)")
      if (o_hyper /= 0) call ratio_report(Hm, Pc(3), thdt / opz, "theta dt diag(K_w X K_w) / (opz diag M_rho)")
      call VecDestroy(d1, ierr); call VecDestroy(d2, ierr)
      if (o_xas /= 0) call m1_report()
    endif

    ! the resistive change of X: 1 - x_res / x_0 (xres only)
    if (o_v == 1 .and. o_x == 0 .and. o_xres /= 0) then
      call VecDuplicate(xd, d1, ierr)
      call MatGetDiagonal(g_ctx%B_33, d1, ierr)
      call VecGetArray(d1, p1, ierr); call VecGetArrayRead(xd, p2, ierr)
      wmax = 0.d0
      do k = 1, size(p1)
        if (p2(k) > 0.d0 .and. p1(k) > 0.d0) wmax = max(wmax, 1.d0 / (p1(k) * p2(k)) - 1.d0)
      enddo
      call VecRestoreArray(d1, p1, ierr); call VecRestoreArrayRead(xd, p2, ierr)
      call MPI_Allreduce(MPI_IN_PLACE, wmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm_p, mpierr)
      if (me_p == 0) write(*,'(A,ES10.3)') "[Physics PC]   peel check: resistive X, max (x_0 / x_res - 1) = ", wmax
      call VecDestroy(d1, ierr)
    endif

  contains

    subroutine ratio_report(A, B, sc, label)
      Mat :: A, B
      real*8, intent(in) :: sc
      character(len=*), intent(in) :: label
      call MatGetDiagonal(A, d1, ierr); call MatGetDiagonal(B, d2, ierr)
      call VecGetArray(d1, p1, ierr); call VecGetArray(d2, p2, ierr)
      wmax = 0.d0; wmean = 0.d0; cnt = 0.d0
      do k = 1, size(p1)
        if (abs(p2(k)) > 0.d0) then
          r = sc * p1(k) / p2(k)
          wmax = max(wmax, r); wmean = wmean + r; cnt = cnt + 1.d0
        endif
      enddo
      call VecRestoreArray(d1, p1, ierr); call VecRestoreArray(d2, p2, ierr)
      call MPI_Allreduce(MPI_IN_PLACE, wmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm_p, mpierr)
      call MPI_Allreduce(MPI_IN_PLACE, wmean, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm_p, mpierr)
      call MPI_Allreduce(MPI_IN_PLACE, cnt, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm_p, mpierr)
      if (me_p == 0) write(*,'(A,A,A,ES10.3,A,ES10.3)') "[Physics PC]   peel check: ", label, &
        ": max ", wmax, ", mean ", wmean / max(cnt, 1.d0)
    end subroutine ratio_report

  end subroutine stage0_check

  !> M1 of the v2 decision gate: the Lanczos extremes of X_AS B_33 (CG on
  !! B_33 with X_AS as the preconditioning matrix) against the colouring
  !! bound, and where the extreme eigenvectors sit (power iterations on X_AS
  !! B_33 and on s I - X_AS B_33, loc_its steps).
  subroutine m1_report()
    PetscErrorCode :: ierr
    KSP :: kc
    PC  :: pcm
    Vec :: bb, xx
    PetscReal :: emax, emin
    call KSPCreate(comm_p, kc, ierr)
    call KSPSetType(kc, KSPCG, ierr)
    call KSPSetOperators(kc, g_ctx%B_33, Xas, ierr)
    call KSPGetPC(kc, pcm, ierr)
    call PCSetType(pcm, PCMAT, ierr)
    call KSPSetComputeSingularValues(kc, PETSC_TRUE, ierr)
    call KSPSetTolerances(kc, 1.d-30, 1.d-50, 1.d10, 200, ierr)
    call KSPSetNormType(kc, KSP_NORM_UNPRECONDITIONED, ierr)
    call MatCreateVecs(g_ctx%B_33, xx, bb, ierr)
    call VecSetRandom(bb, PETSC_NULL_RANDOM, ierr)
    call KSPSolve(kc, bb, xx, ierr)
    call KSPComputeExtremeSingularValues(kc, emax, emin, ierr)
    call KSPDestroy(kc, ierr); call VecDestroy(bb, ierr); call VecDestroy(xx, ierr)
    if (me_p == 0) write(*,'(A,2ES11.3,A,F8.2,A,I0,A,F7.3)') "[Physics PC]   peel M1: spectrum of X_AS B_33 ~ [", &
      emin, emax, "], cond ", emax / max(emin, 1.d-300), "; colouring bound N_c = ", as_ncol, &
      ", Chebyshev top min(N_c, 1.25 lambda_max) = ", min(dble(as_ncol), 1.25d0 * emax)
    call locate(.true., emax)
    call locate(.false., 1.05d0 * emax)

  contains

    !> power iteration for the top (top) or the bottom (s I - X B, s = shift)
    !! eigenvector; prints its eigenvalue and the node of its largest entry
    subroutine locate(top, shift)
      use mod_parameters, only: n_tor, n_degrees
      use nodes_elements, only: node_list
      logical, intent(in) :: top
      real*8, intent(in) :: shift
      Vec :: v, w, z
      PetscInt :: imax
      PetscReal :: vmax, nrm, lam
      integer :: it, idx, nd, kd, m
      call MatCreateVecs(g_ctx%B_33, v, w, ierr)
      call VecDuplicate(v, z, ierr)
      call VecSetRandom(v, PETSC_NULL_RANDOM, ierr)
      call VecNormalize(v, nrm, ierr)
      do it = 1, merge(200, o_loc_its, top)
        call MatMult(g_ctx%B_33, v, w, ierr)
        call MatMult(Xas, w, z, ierr)
        if (.not. top) then                                      ! z = s v - X B v
          call VecScale(z, -1.d0, ierr)
          call VecAXPY(z, shift, v, ierr)
        endif
        call VecCopy(z, v, ierr)
        call VecNormalize(v, nrm, ierr)
      enddo
      call MatMult(g_ctx%B_33, v, w, ierr)
      call MatMult(Xas, w, z, ierr)
      call VecDot(v, z, lam, ierr)
      call VecCopy(v, w, ierr); call VecAbs(w, ierr)
      call VecMax(w, imax, vmax, ierr)
      if (me_p == 0) then
        idx = int(imax) / n_tor + 1; m = mod(int(imax), n_tor)
        do nd = 1, node_list%n_nodes
          do kd = 1, n_degrees
            if (node_list%node(nd)%index(kd) /= idx) cycle
            write(*,'(A,A,F8.4,A,I0,A,I0,A,I0,A,2F8.4,A,L1,A,I0,A,F6.3)') "[Physics PC]   peel M1: ", &
              merge("top    lambda ", "bottom lambda ", top), lam, ": max entry at node ", nd, &
              " (Hermite DOF ", kd, ", slot ", m, "), R Z", node_list%node(nd)%x(1, 1, 1:2), &
              ", axis ", node_list%node(nd)%axis_node, ", boundary ", node_list%node(nd)%boundary, &
              ", |v|_max ", vmax
          enddo
        enddo
      endif
      call VecDestroy(v, ierr); call VecDestroy(w, ierr); call VecDestroy(z, ierr)
    end subroutine locate

  end subroutine m1_report

  !--------------------------------------------------------------------
  !> Stage 1: Arnoldi on S P^-1 (right-preconditioned GMRES on the reduced
  !! operator -S, A_aa^-1 by MUMPS), from a random right-hand side. With
  !! -sf_peel_q lu -sf_peel_t lu only the Schur approximation P is left;
  !! with the GMG (fixed V-cycles, a linear operator) also its inner error.
  !! Reports the Ritz values' modulus range and argument, the outliers, and
  !! the iterations to 1e-1 and 1e-6.
  !--------------------------------------------------------------------
  subroutine spectrum()
    PetscErrorCode :: ierr
    PetscInt :: nl, ng, neig, its1, its6
    Mat :: Sop
    KSP :: ks
    PC  :: pcs
    Vec :: b, x
    real*8, allocatable :: er(:), ei(:), am(:)
    integer :: nout, k
    call ensure_exact_aa()

    call MatGetLocalSize(g_ctx%B_22, nl, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(g_ctx%B_22, ng, PETSC_NULL_INTEGER, ierr)
    call MatCreateShell(comm_p, nl, nl, ng, ng, PETSC_NULL_INTEGER, Sop, ierr)
    call MatShellSetOperation(Sop, MATOP_MULT, s_mult, ierr)

    call KSPCreate(comm_p, ks, ierr)
    call KSPSetType(ks, KSPGMRES, ierr)
    call KSPSetOperators(ks, Sop, Sop, ierr)
    call KSPGMRESSetRestart(ks, o_spec, ierr)
    call KSPSetPCSide(ks, PC_RIGHT, ierr)
    call KSPGetPC(ks, pcs, ierr)
    call PCSetType(pcs, PCSHELL, ierr)
    call PCShellSetApply(pcs, p_apply, ierr)
    call KSPSetComputeEigenvalues(ks, PETSC_TRUE, ierr)
    call VecDuplicate(ru, b, ierr); call VecDuplicate(ru, x, ierr)
    call VecSetRandom(b, PETSC_NULL_RANDOM, ierr)
    call VecPointwiseMult(b, b, imask, ierr)

    call KSPSetTolerances(ks, 1.d-1, 1.d-50, 1.d10, o_spec, ierr)
    call KSPSolve(ks, b, x, ierr)
    call KSPGetIterationNumber(ks, its1, ierr)
    call KSPSetTolerances(ks, 1.d-6, 1.d-50, 1.d10, o_spec, ierr)
    call KSPSolve(ks, b, x, ierr)
    call KSPGetIterationNumber(ks, its6, ierr)

    allocate(er(o_spec), ei(o_spec))
    call KSPComputeEigenvalues(ks, int(o_spec, kind(neig)), er, ei, neig, ierr)
    if (me_p == 0 .and. neig > 0) then
      am = sqrt(er(1:neig)**2 + ei(1:neig)**2)
      nout = count(am < 0.5d0 .or. am > 2.d0)
      write(*,'(A,I0,A,I0,A,I0,A,I0)') "[Physics PC]   peel spec: GMRES on S P^-1: ", its1, " its to 1e-1, ", &
        its6, " to 1e-6 (", neig, " Ritz values), S without bits ", o_spec_drop
      write(*,'(A,2ES11.3,A,F8.2,A,F7.4,A,I0,A)') "[Physics PC]   peel spec: |lambda| in [", minval(am), &
        maxval(am), "], ratio ", maxval(am) / max(minval(am), 1.d-300), ", max |arg| ", &
        maxval(abs(atan2(ei(1:neig), er(1:neig)))), ", ", nout, " outside [0.5, 2]"
      write(*,'(A)', advance="no") "[Physics PC]   peel spec: smallest |lambda|:"
      do k = 1, min(6, int(neig))
        write(*,'(A,F8.4,SP,F8.4,SS,A)', advance="no") " (", er(minloc(am, 1)), ei(minloc(am, 1)), "i)"
        am(minloc(am, 1)) = huge(1.d0)
      enddo
      write(*,*)
      am = sqrt(er(1:neig)**2 + ei(1:neig)**2)
      write(*,'(A)', advance="no") "[Physics PC]   peel spec: largest |lambda|: "
      do k = 1, min(6, int(neig))
        write(*,'(A,F8.4,SP,F8.4,SS,A)', advance="no") " (", er(maxloc(am, 1)), ei(maxloc(am, 1)), "i)"
        am(maxloc(am, 1)) = -1.d0
      enddo
      write(*,*)
    endif
    call KSPDestroy(ks, ierr); call MatDestroy(Sop, ierr)
    call VecDestroy(b, ierr); call VecDestroy(x, ierr)

  contains

    !> y = A_uu u - A_ua A_aa^-1 A_au u (= -S u), A_aa^-1 exact: pair_w
    !! applied to (u, -A_aa^-1 A_au u), its u rows
    subroutine s_mult(A, u, y, ierr_)
      Mat :: A
      Vec :: u, y
      PetscErrorCode :: ierr_
      Vec :: wf, wy
      call MatMult(g_ctx%B_42, u, rw, ierr_)
      call MatMult(g_ctx%B_12, u, rp, ierr_)
      call VecSet(rj, 0.d0, ierr_)
      call sf_solver_apply(lw, rw, yw, ierr_)
      call sf_split_parts(rpj, [rp, rj], .true.)
      call sf_solver_apply(lpj, rpj, ypj, ierr_)
      call sf_split_parts(ypj, [yp, yj], .false.)
      call VecScale(yw, -1.d0, ierr_); call VecScale(yp, -1.d0, ierr_); call VecScale(yj, -1.d0, ierr_)
      call MatCreateVecs(Wop, wf, wy, ierr_)
      call sf_split_parts(wf, [u, yw, yp, yj], .true.)
      call MatMult(Wop, wf, wy, ierr_)
      call sf_split_parts(wy, [y, rw, rp, rj], .false.)
      call VecDestroy(wf, ierr_); call VecDestroy(wy, ierr_)
      if (btest(o_spec_drop, 0)) then
        call MatMult(g_ctx%W_force, u, t2, ierr_)
        call VecAXPY(y, -1.d0, t2, ierr_)
      endif
      if (btest(o_spec_drop, 2)) then             ! the curvature half: W + W_comp
        call MatMult(g_ctx%W_force, u, t2, ierr_)
        call VecAXPY(y, -1.d0, t2, ierr_)
        call MatMult(Pc(2), u, t2, ierr_)
        call VecPointwiseMult(t2, t2, imask, ierr_)
        call VecAXPY(y, -1.d0, t2, ierr_)
      endif
      if (btest(o_spec_drop, 1)) then             ! yp = -psi
        call MatMult(g_ctx%B_21, yp, t2, ierr_)
        call VecAXPY(y, -1.d0, t2, ierr_)
      endif
      ierr_ = 0
    end subroutine s_mult

    subroutine p_apply(pc_, r, y, ierr_)
      PC  :: pc_
      Vec :: r, y
      PetscErrorCode :: ierr_
      call VecCopy(r, sv, ierr_)
      call apply_pinv(sv, y)
      ierr_ = 0
    end subroutine p_apply

  end subroutine spectrum

  !====================================================================
  ! Small helpers
  !====================================================================

  subroutine opt_i(name, v)
    character(len=*), intent(in) :: name
    integer, intent(inout) :: v
    PetscInt :: iv
    PetscBool :: set
    PetscErrorCode :: ierr
    iv = v
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, name, iv, set, ierr)
    v = int(iv)
  end subroutine opt_i

  subroutine opt_r(name, v)
    character(len=*), intent(in) :: name
    real*8, intent(inout) :: v
    PetscReal :: rv
    PetscBool :: set
    PetscErrorCode :: ierr
    rv = v
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, name, rv, set, ierr)
    v = rv
  end subroutine opt_r

  !> lu | gmg
  subroutine opt_b(name, v)
    character(len=*), intent(in) :: name
    integer, intent(inout) :: v
    character(len=16) :: s
    PetscBool :: set
    PetscErrorCode :: ierr
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, name, s, set, ierr)
    if (.not. set) return
    select case (trim(s))
    case ("lu");  v = SF_LU
    case ("gmg"); v = SF_GMG
    case default; call fatal(name//" must be lu | gmg")
    end select
  end subroutine opt_b

  character(len=4) function bname(b)
    integer, intent(in) :: b
    bname = merge(" lu ", " gmg", b == SF_LU)
  end function bname

  subroutine fatal(msg)
    character(len=*), intent(in) :: msg
    PetscErrorCode :: ierr
    if (me_p == 0) write(*,'(A,A)') "[Physics PC]   FATAL: peel: ", trim(msg)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end subroutine fatal

#endif
end module mod_petsc_pc_sf_peel
