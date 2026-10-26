module mod_petsc_pc_sf_peel
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx
  use mod_petsc_pc_blocks,      only: pack_pair_aij, make_field_block_scale, pc_print_block_setup
  use mod_petsc_pc_sf_solver
  use mod_petsc_pc_mass_cheb,   only: mass_cheb_t, mass_cheb_setup, mass_cheb_refresh, mass_cheb_solve
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
  !! v3 (-sf_peel_v 3; v3/peeled_schur_preconditioner_v3.md): viscosity moved
  !! out of N into the factor with the inertia,
  !!   P = Q M_rho^-1 H_nu (opz R_hat)^-1 N,
  !!   P^-1 = N^-1 (opz R_hat) H_nu^-1 M_rho Q^-1,
  !!   H_nu = M_rho + theta dt / opz K_nu,  N = opz R_hat + B_12^T Y B_12,
  !!   Y = X_AS(B_33 + theta dt / opz K_nu^psi),
  !! K_nu^psi = K_nu's symbol over M_rho's times B_33's (slot 13), so Q M_rho^-1
  !! H_nu carries S's theta dt K_rho M_rho^-1 K_nu exactly and the bending's
  !! viscous factor cancels in Y. N has reach 3 and is SPD by construction.
  !! (The note writes the same with R_psi in B_33's weights and Y = H_nu^-1;
  !! here R_hat keeps v2's M_rho weights, which moves B_33 / M_rho into Y.)
  !! H_nu^-1 by Chebyshev + ASM(ICC) (mod_petsc_pc_mass_cheb), refitted per build.
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
  !!   ebe bits      v2: terms as sums of element-local products instead of the
  !!                 global X_AS ones (pc_elt_matrix_peel_fft's slots 9..11, the
  !!                 X_AS patch of one element as X and as M_rho^-1), keeping
  !!                 the element stencil: bit 0 the bending B_12^T X B_12, bit 1
  !!                 R_hat's hyper-resistive sandwich, bit 2 the viscous term
  !!                 (7: N without a matrix product); bit 3 the sandwich in
  !!                 strong form, int R div(w grad v) div(w grad u) (slot 12).
  !!                 Measured (41x32, tstep 10): the element-local bending
  !!                 breaks B_12's kernel (S P^-1 down to 1e-3), the
  !!                 element-local sandwich overweights smooth modes (up to 5e3)
  !!   ninner k      v2: P^-1's N^-1 by k inner iterations on the matrix-free
  !!                 N with the global X_AS products (R_hat too), preconditioned
  !!                 by N's solver (with ebe the element-local N's V-cycle); 0 off
  !!   ninner_type   0 Richardson (defect correction, default), 1 GMRES
  !!   tvisc 0..3    ninner: the viscous term of the assembled N that N's
  !!                 solver is built on (default: visc); the matrix-free N keeps visc
  !!   patch 0|1     v2 / v3: the triple products B_12^T X B_12 and K_w X K_w
  !!                 assembled patch by patch, sum_p (R_p C)^T (R_p H R_p^T)^-1
  !!                 (R_p C) with R_p C C's assembled rows, exactly the global
  !!                 X_AS product without MatMatMult or intermediates
  !!   pjfac 0|1     A_aa's (psi, j) by elimination with the factorised psi-Schur,
  !!                 R_psi ~ opz (B_33 + K_alpha) B_33^-1 (B_33 + K_gamma) (slots
  !!                 14, 15): B_33^-1 by Chebyshev + ASM(ICC), the two Helmholtz
  !!                 operators by scalar GMG V-cycles (instances 9, 10, default
  !!                 V(1,1), -sf_gmg9_* / -sf_gmg10_*), no constraint rows
  !!   pjcheck 1     first build: the (psi, j) solve's error against MUMPS on
  !!                 random right-hand sides, the GMG V-cycle and pjfac, and their
  !!                 times
  !--------------------------------------------------------------------

  public :: peel_build, peel_solve, peel_report

  !--- options
  integer, save :: o_x = 0, o_xres = 1, o_visc = 1, o_tri = 0, o_check = 0, o_spec = 0, o_spec_every = 0
  integer, save :: o_spec_drop = 0, o_texact = 0
  integer, save :: o_v = 2, o_xas = 1, o_hyper = 1, o_as_axis = 1, o_loc_its = 3000
  integer, save :: o_ebe = 0, o_ninner = 0, o_ninner_type = 0
  integer, save :: o_patch = 0, o_pjfac = 0, o_pjcheck = 0, o_hcyc = 1
  integer, save :: o_tvisc = -1                !< ninner: the viscous term of N's assembled stand-in (-1 = visc)
  integer, save :: o_visc_set = -1             !< -sf_peel_visc as given (v2 default 2)
  real*8,  save :: o_eta_w = 1.d0              !< v2: K_eta's weight in R_hat (diagnostic)
  integer, save :: o_restart = 10, o_maxits = 30
  integer, save :: bk_q = SF_GMG, bk_t = SF_GMG, bk_aa = SF_GMG
  integer, save :: ncyc_q = 1, ncyc_t = 1, ncyc_pj = 1
  real*8,  save :: o_wcomp = 1.d0, o_x1_kappa = 50.d0
  integer, save :: o_qv = 0

  logical, save :: ready = .false.
  integer, save :: comm_p = MPI_COMM_NULL, me_p = 0
  real*8,  save :: opz = 1.d0, thdt = 1.d0

  !--- operators
  Mat, save :: Wop                         !< the packed pair_w (unscaled)
  Mat, save :: Pc(15)                      !< AIJ copies of g_ctx%P_peel
  Mat, save :: Qm, Mm, Tm, TB              !< Q, M_rho, T, B_12^T X B_12
  Mat, save :: Mv                          !< qv: B_44 / opz, in M_rho's place in P^-1
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
  !> v3: H_nu = M_rho + theta dt / opz K_nu (u), HY = B_33 + theta dt / opz
  !! K_nu^psi and Y = X_AS(HY) (both per build)
  Mat, save :: Hn, HY, Yas, YasB12
  Mat, save :: B33a                           !< B_33 as AIJ (v3, pjfac)
  logical, save :: b33a_built = .false.
  logical, save :: yas_built = .false.
  type(mass_cheb_t), save :: mhn              !< v3: H_nu^-1
  type(block_solver_t), save :: luHn, luY     !< texact v3: H_nu, HY by MUMPS
  !> pjfac: B_33 + K_alpha, B_33 + K_gamma, B_33^-1, the psi rows without B_13
  Mat, save :: Ha, Hg
  type(mass_cheb_t), save :: m33
  type(block_solver_t), save :: sha, shg
  PetscInt, allocatable, save :: pbrows(:)
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
  KSP, save :: kin                          !< ninner: inner iterations on the matrix-free N
  Mat, save :: Nmf
  logical, save :: xmr_built = .false.
  type(mass_cheb_t), save :: mw
  KSP, save :: kout

  !--- work vectors (u layout; the (psi, j) pair packed)
  Vec, save :: ru, rw, rp, rj, yu, yw, yp, yj, sv, t1, t2, rpj, ypj, p1, p2, p3

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
      ! X_AS(B_33) as a matrix: the global products (not patch), the checks, ninner
      if (first .and. (iand(o_ebe, 7) /= 7 .or. o_ninner > 0) .and. o_xas /= 0 .and. &
          (o_patch == 0 .or. o_check /= 0 .or. o_ninner > 0)) &
        call build_xas(g_ctx%B_33, Xas, "B_33", .true.)
      if (o_ninner > 0) call build_xmr()
      if (first .and. o_ebe /= 0) call ebe_checks()
      if (o_v == 3) call build_v3(first)
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
    if (o_pjfac == 0 .or. o_pjcheck /= 0) &
      call sf_solver_setup(spj, PJm, bk_aa, "peel A_aa (psi, j)", comm, my_id, rtol, &
                           gmg_inst=8, nfields=2, smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, &
                           maxits=SF_GMG_MAXITS, pre0=SF_PJ_PRE0, post0=SF_PJ_POST0, &
                           nsmooth_c=SF_PJ_NSC, axis_rings=SF_GMG_AXIS_RINGS_PAIRS, ncyc=ncyc_pj)
    if (o_pjfac /= 0) call build_pjfac(first)
    if (bk_aa == SF_LU) then
      call sf_solver_setup(sw, g_ctx%B_44, SF_LU, "peel A_aa omega (B_44)", comm, my_id, rtol, &
                           gmg_inst=0, nfields=1, smoother=0, maxits=1)
    else if (first) then
      call mass_cheb_setup(mw, g_ctx%B_44, comm, "peel A_aa omega (B_44)")
    endif

    tw(6) = MPI_Wtime()
    if (o_texact /= 0) call setup_texact(first)
    if (o_ninner > 0 .and. first) call setup_ninner()
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
    if (o_pjcheck /= 0 .and. first .and. o_pjfac /= 0) call pj_check()
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
      call opt_i("-sf_peel_ebe", o_ebe);     call opt_i("-sf_peel_ninner", o_ninner)
      call opt_i("-sf_peel_ninner_type", o_ninner_type); call opt_i("-sf_peel_tvisc", o_tvisc)
      call opt_i("-sf_peel_patch", o_patch); call opt_i("-sf_peel_pjfac", o_pjfac)
      call opt_i("-sf_peel_pjcheck", o_pjcheck); call opt_i("-sf_peel_hcyc", o_hcyc)
      call opt_i("-sf_peel_qv", o_qv)
      if (o_qv /= 0 .and. o_v /= 3) call fatal("-sf_peel_qv needs v3")
      if ((o_ebe /= 0 .or. o_ninner > 0) .and. o_v /= 2) call fatal("-sf_peel_ebe / ninner need v2")
      if (o_ninner > 0 .and. o_xas == 0) call fatal("-sf_peel_ninner needs X_AS (xas 1)")
      if (o_v < 1 .or. o_v > 3) call fatal("-sf_peel_v must be 1, 2 or 3")
      if (o_patch /= 0 .and. (o_v == 1 .or. o_xas == 0 .or. o_ebe /= 0)) &
        call fatal("-sf_peel_patch needs v2 / v3 with X_AS and no ebe")
      if (o_v == 2) then                         ! v2 default: the resistive factor on K_nu
        o_visc_set = 2
        call opt_i("-sf_peel_visc", o_visc_set)
        o_visc = o_visc_set
      else if (o_v == 3) then                    ! v3: viscosity in H_nu and Y, none in N
        o_visc = 0
      endif
      if (o_visc < 0 .or. o_visc > 3 .or. (o_v == 1 .and. o_visc > 1)) &
        call fatal("-sf_peel_visc must be 0 | 1 (v1, v2), 2 | 3 (v2)")
      call opt_i("-sf_peel_restart", o_restart); call opt_i("-sf_peel_maxits", o_maxits)
      call opt_i("-sf_peel_qcyc", ncyc_q);   call opt_i("-sf_peel_tcyc", ncyc_t)
      call opt_i("-sf_peel_pjcyc", ncyc_pj)
      call opt_r("-sf_peel_wcomp", o_wcomp); call opt_r("-sf_peel_x1_kappa", o_x1_kappa)
      call opt_b("-sf_peel_q", bk_q); call opt_b("-sf_peel_t", bk_t); call opt_b("-sf_peel_aa", bk_aa)
      if (o_pjfac /= 0 .and. bk_aa == SF_LU) call fatal("-sf_peel_pjfac replaces the GMG A_aa (aa gmg)")
      if (o_x /= 0 .and. o_x /= 1) call fatal("-sf_peel_x must be 0 or 1")
      if (o_x == 1 .and. o_xres /= 0) then
        if (my_id == 0) write(*,'(A)') PP//"-sf_peel_x 1 takes diag(B_33) alone: xres ignored"
        o_xres = 0
      endif
      if (my_id == 0 .and. o_v >= 2) then
        write(*,'(A,I0,A,A,A,I0,A,I0,A,F6.3,A,I0,A,I0)') PP//"v", o_v, ", X = ", merge("X_AS     ", "diag B_33", o_xas /= 0), &
          ", hyper-resistive sandwich ", o_hyper, ", axis patch ", o_as_axis, ", W_comp weight ", o_wcomp, &
          ", K_nu ", o_visc, ", lower-triangular ", o_tri
        write(*,'(A,I0,A,I0,A,A)') PP//"element-local R_hat, N ", o_ebe, ", inner iterations on N ", o_ninner, &
          ", ", merge("Richardson", "GMRES     ", o_ninner_type == 0)
        write(*,'(A,I0,A,I0)') PP//"patch-wise triple products ", o_patch, ", factorised (psi, j) ", o_pjfac
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
    call sf_solver_report(sq, my_id); call sf_solver_report(st, my_id)
    if (o_pjfac == 0) call sf_solver_report(spj, my_id)
    if (o_pjfac /= 0) then
      call sf_solver_report(sha, my_id); call sf_solver_report(shg, my_id)
      call sf_solver_reset_counters(sha); call sf_solver_reset_counters(shg)
    endif
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

  !> y = -P^-1 s = -T^-1 M_rho Q^-1 s (v1), -N^-1 opz R_hat Q^-1 s (v2),
  !! -N^-1 opz R_hat H_nu^-1 M_rho Q^-1 s (v3), with the Dirichlet u rows exact.
  subroutine apply_pinv(s, y)
    Vec :: s, y
    PetscErrorCode :: ierr
    call VecCopy(s, t1, ierr)
    call sf_solver_apply(sq, t1, t2, ierr)
    if (o_v == 3) then                          ! t2 = H_nu^-1 M_rho t2 (qv: B_44 / opz)
      if (o_qv /= 0) then
        call MatMult(Mv, t2, t1, ierr)
      else
        call MatMult(Mm, t2, t1, ierr)
      endif
      if (o_texact /= 0) then
        call sf_solver_apply(luHn, t1, t2, ierr)
      else
        call mass_cheb_solve(mhn, t1, t2)
      endif
    endif
    if (o_v == 1) then
      call MatMult(Mm, t2, t1, ierr)
    else if (o_texact /= 0) then                ! v2: opz R_hat, R_hat exact with texact
      call rhat_exact(t2, t1)
      call VecScale(t1, opz, ierr)
    else if (o_ninner > 0) then                 ! v2: R_hat by the global X_AS products
      call rhat_as(t2, t1)
      call VecScale(t1, opz, ierr)
    else
      call MatMult(Rh, t2, t1, ierr)
      call VecScale(t1, opz, ierr)
    endif
    if (o_texact /= 0) then
      call KSPSolve(ktex, t1, y, ierr)
    else if (o_ninner > 0) then
      call KSPSolve(kin, t1, y, ierr)
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
    if (o_pjfac /= 0) then
      call pj_fac_solve(ap, aj, op, oj)
      return
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

  !> Q = opz K_rho + w W_comp, the Dirichlet rows the identity. qv: Q = L, the
  !! vorticity row's Laplacian B_42 on the u rows' test functions (B_42^T, sign
  !! as K_rho), so P^-1's B_44 L^-1 / opz commutes the Lorentz term through the
  !! operator that defines omega and the density stays in H_nu (no W_comp)
  subroutine build_q(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    if (o_qv /= 0) then
      if (first) then
        call MatTranspose(g_ctx%B_42, MAT_INITIAL_MATRIX, Qm, ierr)
      else
        call MatTranspose(g_ctx%B_42, MAT_REUSE_MATRIX, Qm, ierr)
      endif
      call MatScale(Qm, same_sign(Qm, Pc(1)), ierr)
      call dirichlet(Qm)
      return
    endif
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
    if (o_qv /= 0) then
      if (first) then
        call MatDuplicate(g_ctx%B_44, MAT_COPY_VALUES, Mv, ierr)
      else
        call MatCopy(g_ctx%B_44, Mv, SAME_NONZERO_PATTERN, ierr)
      endif
      call MatScale(Mv, same_sign(Mv, Pc(3)) / opz, ierr)
      call dirichlet(Mv)
    endif
  end subroutine build_m

  !> +-1: the sign that gives A's diagonal the sign of B's (summed)
  real*8 function same_sign(A, B)
    Mat :: A, B
    PetscErrorCode :: ierr
    Vec :: d
    PetscScalar :: sa, sb
    call MatCreateVecs(A, d, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, d, ierr); call VecSum(d, sa, ierr)
    call MatGetDiagonal(B, d, ierr); call VecSum(d, sb, ierr)
    call VecDestroy(d, ierr)
    same_sign = merge(1.d0, -1.d0, sa * sb > 0.d0)
  end function same_sign

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
  !! With Cm (patch): X = Cm^T X_AS Cm instead, assembled patch by patch as
  !! sum_p (R_p C)^T (R_p B R_p^T)^-1 (R_p C) (+ the decoupled rows' c_i^T c_i
  !! / b_ii), R_p C the assembled rows of C = Cm: the global product exactly, by
  !! one Cholesky solve and one rank update per patch, without MatMatMult or
  !! intermediates; X's pattern on firstc (MATPREALLOCATOR), refilled after.
  !--------------------------------------------------------------------
  subroutine build_xas(B, X, label, verbose, Cm, firstc)
    use mod_parameters, only: n_tor, n_degrees, n_vertex_max
    use nodes_elements, only: node_list, element_list
    Mat :: B, X
    character(len=*), intent(in) :: label
    logical, intent(in) :: verbose
    Mat, optional :: Cm
    logical, intent(in), optional :: firstc
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
    external :: dpotrf, dpotri, dtrsm, dsyrk

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
    if (.not. present(Cm)) then
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
    endif

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
    if (present(Cm)) then
      call c_products(sub(1))
      call MatDestroySubMatrices(one, sub, ierr)
      call ISDestroy(isu, ierr)
      call MPI_Allreduce(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, comm_p, mpierr)
      if (me_p == 0 .and. firstc) write(*,'(A,A,A,I0,A,I0,A,I0,A)') "[Physics PC]   peel: ", label, ", ", npat, &
        " patches (axis patch ", naxd, " DOFs), ", nfail, " Cholesky failures"
      return
    endif
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

    !> X = sum_p (R_p C)^T (R_p B R_p^T)^-1 (R_p C) + the decoupled rows'
    !! outer products, sb = B on this rank's patch rows
    subroutine c_products(sb)
      Mat :: sb
      Mat :: Cc, pre, tgt
      Mat, pointer :: subc(:)
      IS :: isall
      PetscInt :: ngc, nlc, ncol, nr
      PetscInt, allocatable :: cl(:)
      integer, allocatable :: cpos(:)
      real*8, allocatable :: cp(:, :), g(:, :)
      MatType :: mt
      logical :: conv, vals_pass
      integer :: pass, npass, jj, ic
      call MatGetType(Cm, mt, ierr)
      conv = (mt /= MATMPIAIJ .and. mt /= MATSEQAIJ)
      if (conv) then
        call MatConvert(Cm, MATMPIAIJ, MAT_INITIAL_MATRIX, Cc, ierr)
      else
        Cc = Cm
      endif
      call MatGetSize(Cc, PETSC_NULL_INTEGER, ngc, ierr)
      call MatGetLocalSize(Cc, PETSC_NULL_INTEGER, nlc, ierr)
      call ISCreateStride(PETSC_COMM_SELF, ngc, 0, one, isall, ierr)
      call MatCreateSubMatrices(Cc, one, [isu], [isall], MAT_INITIAL_MATRIX, subc, ierr)
      allocate(cpos(0:ngc - 1), cl(ngc))
      cpos = 0
      if (firstc) then
        call MatCreate(comm_p, pre, ierr)
        call MatSetSizes(pre, nlc, nlc, ngc, ngc, ierr)
        call MatSetType(pre, MATPREALLOCATOR, ierr)
        call MatSetUp(pre, ierr)
        npass = 2
      else
        call MatZeroEntries(X, ierr)
        npass = 1
      endif
      do pass = 1, npass
        vals_pass = (pass == npass)
        if (vals_pass) then
          tgt = X
        else
          tgt = pre
        endif
        do p = 1, npat
          if (mark(p) == 0) cycle
          n = plen(p)
          allocate(prow(n), lp(n))
          prow = pdof(pst(p):pst(p) + n - 1)
          do j = 1, int(n)
            lp(j) = findpos(urows(1:nu), prow(j))
          enddo
          ! the columns of C's patch rows
          ncol = 0
          do j = 1, int(n)
            call MatGetRow(subc(1), lp(j), nc, cols, vals, ierr)
            do k = 1, nc
              if (cpos(cols(k)) == 0) then
                ncol = ncol + 1; cl(ncol) = cols(k); cpos(cols(k)) = int(ncol)
              endif
            enddo
            call MatRestoreRow(subc(1), lp(j), nc, cols, vals, ierr)
          enddo
          allocate(g(ncol, ncol))
          g = 0.d0
          if (vals_pass) then
            allocate(cp(n, ncol), blk(n, n))
            cp = 0.d0
            do j = 1, int(n)
              call MatGetRow(subc(1), lp(j), nc, cols, vals, ierr)
              do k = 1, nc
                cp(j, cpos(cols(k))) = vals(k)
              enddo
              call MatRestoreRow(subc(1), lp(j), nc, cols, vals, ierr)
            enddo
            call MatGetValues(sb, n, lp, n, lp, blk, ierr)
            call dpotrf('U', int(n), blk, int(n), info)
            if (info == 0) then
              ! U^-T R_p C, then its Gram matrix: (R_p C)^T (U^T U)^-1 (R_p C)
              call dtrsm('L', 'U', 'T', 'N', int(n), int(ncol), 1.d0, blk, int(n), cp, int(n))
              call dsyrk('U', 'T', int(ncol), int(n), 1.d0, cp, int(n), 0.d0, g, int(ncol))
              do ic = 1, int(ncol)
                do jj = ic + 1, int(ncol)
                  g(jj, ic) = g(ic, jj)
                enddo
              enddo
            else
              nfail = nfail + 1
            endif
            deallocate(cp, blk)
          endif
          call MatSetValues(tgt, ncol, cl, ncol, cl, g, ADD_VALUES, ierr)
          cpos(cl(1:ncol)) = 0
          deallocate(g, prow, lp)
        enddo
        ! B's decoupled rows (owned here): c_i^T c_i / b_ii
        do i = rs, re - 1
          if (flag(i) /= 1) cycle
          call MatGetRow(Cc, i, nc, cols, vals, ierr)
          nr = nc
          allocate(g(nr, nr))
          g = 0.d0
          if (vals_pass) then
            do ic = 1, int(nr)
              g(:, ic) = vals(1:nr) * vals(ic) / dg(i)
            enddo
          endif
          cl(1:nr) = cols(1:nr)
          call MatRestoreRow(Cc, i, nc, cols, vals, ierr)
          call MatSetValues(tgt, nr, cl, nr, cl, g, ADD_VALUES, ierr)
          deallocate(g)
        enddo
        call MatAssemblyBegin(tgt, MAT_FINAL_ASSEMBLY, ierr)
        call MatAssemblyEnd(tgt, MAT_FINAL_ASSEMBLY, ierr)
        if (.not. vals_pass) then
          call MatCreate(comm_p, X, ierr)
          call MatSetSizes(X, nlc, nlc, ngc, ngc, ierr)
          call MatSetType(X, MATMPIAIJ, ierr)
          call MatPreallocatorPreallocate(pre, PETSC_TRUE, X, ierr)
          call MatDestroy(pre, ierr)
          call MatSetOption(X, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
        endif
      enddo
      call MatDestroySubMatrices(one, subc, ierr)
      call ISDestroy(isall, ierr)
      if (conv) call MatDestroy(Cc, ierr)
    end subroutine c_products

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
    if (o_hyper /= 0 .and. btest(o_ebe, 3)) then
      Hm = Pc(12)
    else if (o_hyper /= 0 .and. btest(o_ebe, 1)) then
      Hm = Pc(9)
    else if (o_hyper /= 0 .and. o_patch /= 0) then
      call build_xas(g_ctx%B_33, Hm, "B_33 (K_w X K_w by patches)", .false., Pc(6), first)
    else if (o_hyper /= 0) then
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
    integer :: vs
    vs = o_visc
    if (o_tvisc >= 0) vs = o_tvisc
    if (btest(o_ebe, 0)) then
      TB = Pc(10)
    else if (o_v == 3 .and. o_patch /= 0) then     ! v3: Y = X_AS(HY)
      call build_xas(HY, TB, "HY (B_12^T Y B_12 by patches)", .false., g_ctx%B_12, first)
    else if (o_v == 3) then                          ! Yas is a new matrix per build
      if (.not. first) then
        call MatDestroy(YasB12, ierr); call MatDestroy(TB, ierr)
      endif
      call MatMatMult(Yas, g_ctx%B_12, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, YasB12, ierr)
      call MatTransposeMatMult(g_ctx%B_12, YasB12, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, TB, ierr)
    else if (o_xas /= 0 .and. o_patch /= 0) then
      call build_xas(g_ctx%B_33, TB, "B_33 (B_12^T X B_12 by patches)", .false., g_ctx%B_12, first)
    else if (o_xas /= 0) then
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
    if (vs >= 2 .and. .not. btest(o_ebe, 2)) call build_visc(vs)
    if (first) then
      call MatDuplicate(TB, MAT_COPY_VALUES, Tm, ierr)
      call MatAXPY(Tm, opz, Rh, DIFFERENT_NONZERO_PATTERN, ierr)
      if (vs == 1) call MatAXPY(Tm, opz * thdt, Pc(4), DIFFERENT_NONZERO_PATTERN, ierr)
      if (vs >= 2 .and. btest(o_ebe, 2)) then
        call MatAXPY(Tm, 1.d0, Pc(11), DIFFERENT_NONZERO_PATTERN, ierr)
      else if (vs >= 2) then
        call MatAXPY(Tm, 0.5d0 * thdt, Vc, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatAXPY(Tm, 0.5d0 * thdt, Vt, DIFFERENT_NONZERO_PATTERN, ierr)
      endif
      call MatSetOption(Tm, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatZeroEntries(Tm, ierr)
      call MatAXPY(Tm, 1.d0, TB, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(Tm, opz, Rh, SUBSET_NONZERO_PATTERN, ierr)
      if (vs == 1) call MatAXPY(Tm, opz * thdt, Pc(4), SUBSET_NONZERO_PATTERN, ierr)
      if (vs >= 2 .and. btest(o_ebe, 2)) then
        call MatAXPY(Tm, 1.d0, Pc(11), SUBSET_NONZERO_PATTERN, ierr)
      else if (vs >= 2) then
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
  subroutine build_visc(vs)
    integer, intent(in) :: vs
    PetscErrorCode :: ierr
    if (vis_built) then
      call MatDestroy(Xmr, ierr); call MatDestroy(XmK, ierr)
      call MatDestroy(Vc, ierr); call MatDestroy(Vt, ierr)
    endif
    call build_xas(Mm, Xmr, "M_rho", .not. vis_built)
    call MatMatMult(Xmr, Pc(4), MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, XmK, ierr)
    if (vs == 3) then
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

  !> B_33 as AIJ, once (geometry-only)
  subroutine ensure_b33a()
    PetscErrorCode :: ierr
    if (b33a_built) return
    call MatConvert(g_ctx%B_33, MATMPIAIJ, MAT_INITIAL_MATRIX, B33a, ierr)
    b33a_built = .true.
  end subroutine ensure_b33a

  !> v3: H_nu = M_rho + theta dt / opz K_nu (u; the Dirichlet rows the
  !! identity) and its Chebyshev; HY = B_33 + theta dt / opz K_nu^psi and,
  !! for the global products, Y = X_AS(HY). Per build (nu(T), rho0). Prints
  !! the grid viscous numbers diag(theta dt / opz K_nu) / diag(M_rho) (and
  !! HY's): the element / rank Schwarz of H_nu degrades once they are >> 1
  !! (v3 note: kappa(X_AS H) 91 at 30).
  subroutine build_v3(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    call ensure_b33a()
    if (first) then
      call MatDuplicate(Pc(3), MAT_COPY_VALUES, Hn, ierr)
      call MatAXPY(Hn, thdt / opz, Pc(4), DIFFERENT_NONZERO_PATTERN, ierr)
      call MatSetOption(Hn, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
      call MatDuplicate(Pc(13), MAT_COPY_VALUES, HY, ierr)
      call MatScale(HY, thdt / opz, ierr)
      call MatAXPY(HY, 1.d0, B33a, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatSetOption(HY, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatZeroEntries(Hn, ierr)
      call MatAXPY(Hn, 1.d0, Pc(3), SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(Hn, thdt / opz, Pc(4), SUBSET_NONZERO_PATTERN, ierr)
      call MatZeroEntries(HY, ierr)
      call MatAXPY(HY, thdt / opz, Pc(13), SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(HY, 1.d0, B33a, SUBSET_NONZERO_PATTERN, ierr)
    endif
    call dirichlet(Hn)
    call diag_ratio(Pc(4), Pc(3), thdt / opz, "v3 grid viscous number, theta dt / opz diag K_nu / diag M_rho")
    call diag_ratio(Pc(13), B33a, thdt / opz, "v3 Y's, theta dt / opz diag K_nu^psi / diag B_33")
    if (o_patch == 0) then
      if (yas_built) call MatDestroy(Yas, ierr)
      call build_xas(HY, Yas, "Y = B_33 + theta dt / opz K_nu^psi", .not. yas_built)
      yas_built = .true.
    endif
    if (first) then
      call mass_cheb_setup(mhn, Hn, comm_p, "peel v3 H_nu = M_rho + theta dt / opz K_nu")
    else
      call mass_cheb_refresh(mhn, Hn, comm_p, "peel v3 H_nu")
    endif
  end subroutine build_v3

  !> max and mean of sc diag(A) / diag(B) over B's nonzero diagonal
  subroutine diag_ratio(A, B, sc, label)
    Mat :: A, B
    real*8, intent(in) :: sc
    character(len=*), intent(in) :: label
    PetscErrorCode :: ierr
    Vec :: d1, d2
    PetscScalar, pointer :: p1(:), p2(:)
    real*8 :: w(3)
    integer :: k, mpierr
    call MatCreateVecs(B, d1, PETSC_NULL_VEC, ierr)
    call VecDuplicate(d1, d2, ierr)
    call MatGetDiagonal(A, d1, ierr); call MatGetDiagonal(B, d2, ierr)
    call VecGetArrayRead(d1, p1, ierr); call VecGetArrayRead(d2, p2, ierr)
    w = 0.d0
    do k = 1, size(p1)
      if (abs(p2(k)) > 0.d0) then
        w(1) = max(w(1), sc * p1(k) / p2(k)); w(2) = w(2) + sc * p1(k) / p2(k); w(3) = w(3) + 1.d0
      endif
    enddo
    call VecRestoreArrayRead(d1, p1, ierr); call VecRestoreArrayRead(d2, p2, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, w(1), 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm_p, mpierr)
    call MPI_Allreduce(MPI_IN_PLACE, w(2:3), 2, MPI_DOUBLE_PRECISION, MPI_SUM, comm_p, mpierr)
    if (me_p == 0) write(*,'(A,A,A,ES10.3,A,ES10.3)') "[Physics PC]   peel: ", label, ": max ", w(1), &
      ", mean ", w(2) / max(w(3), 1.d0)
    call VecDestroy(d1, ierr); call VecDestroy(d2, ierr)
  end subroutine diag_ratio

  !> pjfac: H_alpha = B_33 + K_alpha, H_gamma = B_33 + K_gamma, with K's rows
  !! and columns of the psi rows without B_13 zeroed (R_psi = opz B_33 there),
  !! their GMG V-cycles (instances 9, 10), and B_33^-1's Chebyshev once
  subroutine build_pjfac(first)
    logical, intent(in) :: first
    PetscErrorCode :: ierr
    call ensure_b33a()
    if (first) then
      call find_pb_rows()
      call MatDuplicate(Pc(14), MAT_COPY_VALUES, Ha, ierr)
      call MatDuplicate(Pc(15), MAT_COPY_VALUES, Hg, ierr)
    else
      call MatCopy(Pc(14), Ha, SUBSET_NONZERO_PATTERN, ierr)
      call MatCopy(Pc(15), Hg, SUBSET_NONZERO_PATTERN, ierr)
    endif
    call MatZeroRowsColumns(Ha, size(pbrows, kind=kind(pbrows)), pbrows, 0.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
    call MatZeroRowsColumns(Hg, size(pbrows, kind=kind(pbrows)), pbrows, 0.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
    if (first) then
      call MatAXPY(Ha, 1.d0, B33a, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatAXPY(Hg, 1.d0, B33a, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatSetOption(Ha, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
      call MatSetOption(Hg, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    else
      call MatAXPY(Ha, 1.d0, B33a, SUBSET_NONZERO_PATTERN, ierr)
      call MatAXPY(Hg, 1.d0, B33a, SUBSET_NONZERO_PATTERN, ierr)
    endif
    call diag_ratio(Pc(14), B33a, 1.d0, "pjfac diag K_alpha / diag B_33")
    call diag_ratio(Pc(15), B33a, 1.d0, "pjfac diag K_gamma / diag B_33")
    call sf_solver_setup(sha, Ha, SF_GMG, "peel (psi, j) H_alpha = B_33 + K_alpha", comm_p, me_p, 1.d-2, &
                         gmg_inst=9, nfields=1, smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, &
                         maxits=SF_GMG_MAXITS, pre0=1, post0=1, nsmooth_c=1, ncyc=o_hcyc)
    call sf_solver_setup(shg, Hg, SF_GMG, "peel (psi, j) H_gamma = B_33 + K_gamma", comm_p, me_p, 1.d-2, &
                         gmg_inst=10, nfields=1, smoother=SF_GMG_SMOOTHER_ZEBRA_RINGS, &
                         maxits=SF_GMG_MAXITS, pre0=1, post0=1, nsmooth_c=1, ncyc=o_hcyc)
    if (first) call mass_cheb_setup(m33, B33a, comm_p, "peel (psi, j) B_33")
  end subroutine build_pjfac

  !> pjfac: the psi rows with nothing in B_13 (the eliminated psi DOFs)
  subroutine find_pb_rows()
    PetscErrorCode :: ierr
    PetscInt :: rs, re, i, nc, k, nb
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    logical, allocatable :: em(:)
    integer :: ntot, mpierr
    call MatGetOwnershipRange(g_ctx%B_13, rs, re, ierr)
    allocate(em(rs:re - 1))
    do i = rs, re - 1
      em(i) = .true.
      call MatGetRow(g_ctx%B_13, i, nc, cols, vals, ierr)
      do k = 1, nc
        if (vals(k) /= 0.d0) em(i) = .false.
      enddo
      call MatRestoreRow(g_ctx%B_13, i, nc, cols, vals, ierr)
    enddo
    nb = count(em)
    allocate(pbrows(nb))
    nb = 0
    do i = rs, re - 1
      if (em(i)) then
        nb = nb + 1; pbrows(nb) = i
      endif
    enddo
    call MPI_Allreduce(int(nb), ntot, 1, MPI_INTEGER, MPI_SUM, comm_p, mpierr)
    if (me_p == 0) write(*,'(A,I0,A)') "[Physics PC]   peel pjfac: ", ntot, " psi rows without B_13"
  end subroutine find_pb_rows

  !> (psi, j) = [[opz B_33, B_13], [B_31, B_33]]^-1 (a_p, a_j) by elimination,
  !! R_psi^-1 ~ opz^-1 H_gamma^-1 B_33 H_alpha^-1 (pjfac)
  subroutine pj_fac_solve(ap, aj, op, oj)
    Vec :: ap, aj, op, oj
    PetscErrorCode :: ierr
    call mass_cheb_solve(m33, aj, p1)              ! z = B_33^-1 a_j
    call MatMult(g_ctx%B_13, p1, p2, ierr)
    call VecAYPX(p2, -1.d0, ap, ierr)              ! a_p - B_13 z
    call sf_solver_apply(sha, p2, p3, ierr)
    call MatMult(B33a, p3, p2, ierr)
    call sf_solver_apply(shg, p2, op, ierr)
    call VecScale(op, 1.d0 / opz, ierr)
    call MatMult(g_ctx%B_31, op, p2, ierr)
    call VecAYPX(p2, -1.d0, aj, ierr)              ! a_j - B_31 psi
    call mass_cheb_solve(m33, p2, oj)
  end subroutine pj_fac_solve

  !> pjcheck: the (psi, j) solve against MUMPS on 3 random right-hand sides,
  !! the GMG V-cycle (spj) and pjfac: relative 2-norm errors of psi and j, ms
  subroutine pj_check()
    PetscErrorCode :: ierr
    Vec :: ap, aj, ep, ej, xp, xj
    real*8 :: e(4), t(2), nn(2), d, t0
    integer :: k
    call ensure_exact_aa()
    call VecDuplicate(ru, ap, ierr); call VecDuplicate(ru, aj, ierr)
    call VecDuplicate(ru, ep, ierr); call VecDuplicate(ru, ej, ierr)
    call VecDuplicate(ru, xp, ierr); call VecDuplicate(ru, xj, ierr)
    e = 0.d0; t = 0.d0
    do k = 1, 3
      call VecSetRandom(ap, PETSC_NULL_RANDOM, ierr); call VecSetRandom(aj, PETSC_NULL_RANDOM, ierr)
      call sf_split_parts(rpj, [ap, aj], .true.)
      call sf_solver_apply(lpj, rpj, ypj, ierr)
      call sf_split_parts(ypj, [ep, ej], .false.)
      call VecNorm(ep, NORM_2, nn(1), ierr); call VecNorm(ej, NORM_2, nn(2), ierr)
      call sf_split_parts(rpj, [ap, aj], .true.)
      t0 = MPI_Wtime()
      call sf_solver_apply(spj, rpj, ypj, ierr)
      t(1) = t(1) + MPI_Wtime() - t0
      call sf_split_parts(ypj, [xp, xj], .false.)
      call VecAXPY(xp, -1.d0, ep, ierr); call VecNorm(xp, NORM_2, d, ierr); e(1) = max(e(1), d / nn(1))
      call VecAXPY(xj, -1.d0, ej, ierr); call VecNorm(xj, NORM_2, d, ierr); e(2) = max(e(2), d / nn(2))
      t0 = MPI_Wtime()
      call pj_fac_solve(ap, aj, xp, xj)
      t(2) = t(2) + MPI_Wtime() - t0
      call VecAXPY(xp, -1.d0, ep, ierr); call VecNorm(xp, NORM_2, d, ierr); e(3) = max(e(3), d / nn(1))
      call VecAXPY(xj, -1.d0, ej, ierr); call VecNorm(xj, NORM_2, d, ierr); e(4) = max(e(4), d / nn(2))
    enddo
    if (me_p == 0) write(*,'(A,2ES10.2,A,F8.2,A,2ES10.2,A,F8.2,A)') &
      "[Physics PC]   peel pjcheck: (psi, j) error vs MUMPS, GMG V-cycle psi / j", e(1:2), " (", &
      1.d3 * t(1) / 3.d0, " ms); pjfac", e(3:4), " (", 1.d3 * t(2) / 3.d0, " ms)"
    call VecDestroy(ap, ierr); call VecDestroy(aj, ierr); call VecDestroy(ep, ierr)
    call VecDestroy(ej, ierr); call VecDestroy(xp, ierr); call VecDestroy(xj, ierr)
  end subroutine pj_check

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
    call VecDuplicate(ru, p1, ierr); call VecDuplicate(ru, p2, ierr); call VecDuplicate(ru, p3, ierr)
  end subroutine create_work_vectors

  !> ebe, first build: B_12, B_33's element sums against pair_w's blocks
  !! (the element products' inputs)
  subroutine ebe_checks()
    call ebe_check(Pc(7), g_ctx%B_12, "B_12")
    call ebe_check(Pc(8), g_ctx%B_33, "B_33")
  end subroutine ebe_checks

  !> |E - B|_F / |B|_F with the Dirichlet rows and columns of both removed
  subroutine ebe_check(E, B, label)
    Mat :: E, B
    character(len=*), intent(in) :: label
    PetscErrorCode :: ierr
    Mat :: D, Bc
    real*8 :: nd, nb
    call MatDuplicate(E, MAT_COPY_VALUES, D, ierr)
    call MatConvert(B, MATMPIAIJ, MAT_INITIAL_MATRIX, Bc, ierr)
    call MatAXPY(D, -1.d0, Bc, DIFFERENT_NONZERO_PATTERN, ierr)
    call MatZeroRowsColumns(D, size(brows, kind=kind(brows)), brows, 0.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
    call MatZeroRowsColumns(Bc, size(brows, kind=kind(brows)), brows, 0.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
    call MatNorm(D, NORM_FROBENIUS, nd, ierr)
    call MatNorm(Bc, NORM_FROBENIUS, nb, ierr)
    if (me_p == 0) write(*,'(A,A,A,ES10.3)') "[Physics PC]   peel: element sum of ", label, &
      " vs pair_w's block off the Dirichlet rows, relative difference ", nd / max(nb, tiny(1.d0))
    call MatDestroy(D, ierr); call MatDestroy(Bc, ierr)
  end subroutine ebe_check

  !> ninner: X_AS(M_rho) for the viscous term, per rebuild (rho0 changes)
  subroutine build_xmr()
    PetscErrorCode :: ierr
    if (xmr_built) call MatDestroy(Xmr, ierr)
    call build_xas(Mm, Xmr, "M_rho", .not. xmr_built)
    xmr_built = .true.
  end subroutine build_xmr

  !> y = R_hat x with the global X_AS sandwich, on the interior u DOFs (x is
  !! masked, y masked)
  subroutine rhat_as(x, y)
    Vec :: x, y
    PetscErrorCode :: ierr
    Vec :: xm, va, vb
    call VecDuplicate(x, xm, ierr); call VecDuplicate(x, va, ierr); call VecDuplicate(x, vb, ierr)
    call VecPointwiseMult(xm, x, imask, ierr)
    call MatMult(Pc(3), xm, y, ierr)
    call VecScale(y, opz, ierr)
    call MatMult(Pc(5), xm, va, ierr)
    call VecAXPY(y, o_eta_w * thdt, va, ierr)
    if (o_hyper /= 0) then
      call MatMult(Pc(6), xm, va, ierr)
      call MatMult(Xas, va, vb, ierr)
      call MatMult(Pc(6), vb, va, ierr)
      call VecAXPY(y, thdt, va, ierr)
    endif
    call VecPointwiseMult(y, y, imask, ierr)
    call VecDestroy(xm, ierr); call VecDestroy(va, ierr); call VecDestroy(vb, ierr)
  end subroutine rhat_as

  !> ninner: N x = opz R_hat x + B_12^T X_AS B_12 x + theta dt sym(R_hat
  !! X_AS(M_rho) K_nu) x, all by products of the element-assembled operators
  !! (v2's assembled N without forming it), the identity on the Dirichlet rows
  subroutine nmf_mult(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    Vec :: xm, va, vb
    call VecDuplicate(x, xm, ierr); call VecDuplicate(x, va, ierr); call VecDuplicate(x, vb, ierr)
    call VecPointwiseMult(xm, x, imask, ierr)
    call rhat_as(xm, y)
    call VecScale(y, opz, ierr)
    call MatMult(g_ctx%B_12, xm, va, ierr)
    call MatMult(Xas, va, vb, ierr)
    call MatMultTranspose(g_ctx%B_12, vb, va, ierr)
    call VecAXPY(y, 1.d0, va, ierr)
    if (o_visc == 1) then
      call MatMult(Pc(4), xm, va, ierr)
      call VecAXPY(y, opz * thdt, va, ierr)
    else if (o_visc >= 2) then
      call MatMult(Pc(4), xm, va, ierr)
      call MatMult(Xmr, va, vb, ierr)
      call rhat_as(vb, va)
      call VecAXPY(y, 0.5d0 * thdt, va, ierr)
      call rhat_as(xm, va)
      call MatMult(Xmr, va, vb, ierr)
      call MatMult(Pc(4), vb, va, ierr)
      call VecAXPY(y, 0.5d0 * thdt, va, ierr)
    endif
    call VecPointwiseMult(y, y, imask, ierr)
    call VecAXPY(y, 1.d0, x, ierr)                ! Dirichlet rows: the identity
    call VecPointwiseMult(va, x, imask, ierr)
    call VecAXPY(y, -1.d0, va, ierr)
    call VecDestroy(xm, ierr); call VecDestroy(va, ierr); call VecDestroy(vb, ierr)
    ierr = 0
  end subroutine nmf_mult

  !> ninner: k Richardson or GMRES steps on Nmf around N's solver (st)
  subroutine setup_ninner()
    PetscErrorCode :: ierr
    PetscInt :: nl, ng
    PC :: pc
    call MatGetLocalSize(g_ctx%B_22, nl, PETSC_NULL_INTEGER, ierr)
    call MatGetSize(g_ctx%B_22, ng, PETSC_NULL_INTEGER, ierr)
    call MatCreateShell(comm_p, nl, nl, ng, ng, PETSC_NULL_INTEGER, Nmf, ierr)
    call MatShellSetOperation(Nmf, MATOP_MULT, nmf_mult, ierr)
    call KSPCreate(comm_p, kin, ierr)
    if (o_ninner_type == 0) then
      call KSPSetType(kin, KSPRICHARDSON, ierr)
      call KSPSetNormType(kin, KSP_NORM_NONE, ierr)
    else
      call KSPSetType(kin, KSPGMRES, ierr)
      call KSPGMRESSetRestart(kin, o_ninner, ierr)
      call KSPSetPCSide(kin, PC_RIGHT, ierr)
    endif
    call KSPSetTolerances(kin, 0.d0, 0.d0, 1.d10, o_ninner, ierr)
    call KSPSetOperators(kin, Nmf, Nmf, ierr)
    call KSPGetPC(kin, pc, ierr)
    call PCSetType(pc, PCSHELL, ierr)
    call PCShellSetApply(pc, tex_pc, ierr)
  end subroutine setup_ninner

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
    if (o_v >= 2) call sf_solver_setup(lu33, g_ctx%B_33, SF_LU, "peel exact X = B_33^-1", comm_p, me_p, &
                                       1.d-12, gmg_inst=0, nfields=1, smoother=0, maxits=1)
    if (o_v == 3) then
      call sf_solver_setup(luY, HY, SF_LU, "peel exact Y = HY^-1", comm_p, me_p, &
                           1.d-12, gmg_inst=0, nfields=1, smoother=0, maxits=1)
      call sf_solver_setup(luHn, Hn, SF_LU, "peel exact H_nu", comm_p, me_p, &
                           1.d-12, gmg_inst=0, nfields=1, smoother=0, maxits=1)
    endif
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
    if (o_v >= 2) then
      ! N_ex xm = opz R_hat_ex xm + B_12^T X_ex B_12 xm + the viscous term (v2),
      ! X_ex = B_33^-1 (v2), HY^-1 (v3)
      call rhat_exact(xm, y)
      call VecScale(y, opz, ierr)
      call MatMult(g_ctx%B_12, xm, va, ierr)
      if (o_v == 3) then
        call sf_solver_apply(luY, va, vb, ierr)
      else
        call sf_solver_apply(lu33, va, vb, ierr)
      endif
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
    if (o_v >= 2) then
      call VecDuplicate(ru, d1, ierr); call VecDuplicate(ru, d2, ierr)
      call ratio_report(Pc(5), Pc(3), thdt / opz, "theta dt diag K_eta / (opz diag M_rho)")
      if (o_hyper /= 0) call ratio_report(Hm, Pc(3), thdt / opz, "theta dt diag(K_w X K_w) / (opz diag M_rho)")
      call VecDestroy(d1, ierr); call VecDestroy(d2, ierr)
      if (o_xas /= 0) call m1_report()
      call n_lanczos()
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

  !> v2 / v3: is N SPD? Lanczos (CG, Jacobi) on N: its Ritz values of D^-1 N
  subroutine n_lanczos()
    PetscErrorCode :: ierr
    KSP :: kc
    PC  :: pcj
    Vec :: bb, xx
    PetscInt :: ne
    real*8 :: er(200), ei(200)
    call KSPCreate(comm_p, kc, ierr)
    call KSPSetType(kc, KSPCG, ierr)
    call KSPSetOperators(kc, Tm, Tm, ierr)
    call KSPGetPC(kc, pcj, ierr)
    call PCSetType(pcj, PCJACOBI, ierr)
    call KSPSetComputeEigenvalues(kc, PETSC_TRUE, ierr)
    call KSPSetTolerances(kc, 1.d-30, 1.d-50, 1.d10, 200, ierr)
    call KSPSetNormType(kc, KSP_NORM_UNPRECONDITIONED, ierr)
    call MatCreateVecs(Tm, xx, bb, ierr)
    call VecSetRandom(bb, PETSC_NULL_RANDOM, ierr)
    call KSPSolve(kc, bb, xx, ierr)
    call KSPComputeEigenvalues(kc, 200, er, ei, ne, ierr)
    if (me_p == 0 .and. ne > 0) write(*,'(A,2ES11.3,A,I0,A)') "[Physics PC]   peel check: Lanczos on D^-1 N, Ritz in [", &
      minval(er(1:ne)), maxval(er(1:ne)), "] (", ne, " values; negative = N indefinite)"
    call KSPDestroy(kc, ierr); call VecDestroy(bb, ierr); call VecDestroy(xx, ierr)
  end subroutine n_lanczos

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
