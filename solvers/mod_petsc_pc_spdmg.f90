!> A ring-free multigrid for a scalar symmetric positive definite operator on
!! JOREK's polar flux-surface grid: the P solver of vs_lumped
!! (mod_petsc_pc_vs, -vs_p_solver spdmg) and its P_V solver for v_par
!! (-vs_pv_solver spdmg), from the study studies/spd_multigrid (Python:
!! spdmg/hierarchy.py prolongation_kinds and level_shapes mode "theta",
!! spdmg/multigrid.py). PETSc's PCMG with
!!
!!  - levels that halve the nodes along the flux surface only (J, the ring
!!    position), no radial coarsening: n_tht -> n_tht/2 -> ... while the count
!!    stays even and the next one has at least min_nj positions;
!!  - each Hermite DOF kind k (value, d/dI, d/dJ, d2/dIdJ) as a field of its
!!    own (study section 3): a DOF (I, J, k) of the finer level takes the
!!    coarse DOF (I, J/2, k) for an even J and the mean of (I, (J-1)/2, k) and
!!    (I, (J+1)/2, k) for an odd J (-vs_pmg_transfer linear), or
!!    (I, floor(J/2), k) alone (const); the weights of a row sum to 1 over the
!!    coarse DOFs that exist; on the finest level times the DOF's orientation
!!    (its element size factor, +-1). The shared value of the axis takes the
!!    coarse axis value. A DOF of the finest level whose rows are Dirichlet
!!    rows (penalty or eliminated rows, mod_petsc_pc_blocks penalty_row_mask)
!!    or that no basis function uses has no coarse part, and the coarse levels
!!    have no DOF there. A vector with one value per ring in one kind stays one
!!    on every level (the kernel of the line bending at n = 0);
!!  - or the ring-harmonic transfer from the operator (study section 7,
!!    -vs_pmg_transfer ringharm): on a ring I >= 1 the odd positions (F) take
!!    X_f Pc, X ~ -(A_FF^ring)^-1 A_FC^ring from rh_sweeps node-block Jacobi
!!    steps with the entries within rh_reach positions along the ring, Pc the
!!    linear rows of the even positions (C), which keep them, as do the axis
!!    ring and the axis value (ringharm_rows). It holds the low modes of each
!!    ring of a harmonic (the linear transfer: kappa 24 for n = 6 on the
!!    11 x 24 grid, this one 1.5) and needs no equilibrium data. Its P(g)
!!    depend on the operator of level g-1 and are recomputed at every build;
!!  - the default, -vs_pmg_transfer mixed (the study's choice): the linear
!!    transfer on the n = 0 slot, whose space holds the kernel of n = 0 (dt 30,
!!    m2, linewedge: kappa 10 with the ring-harmonic transfer on n = 0),
!!    the ring-harmonic one on the harmonics' slots;
!!  - all toroidal slots of the operator together (the linear P = P_s x I_slots;
!!    the ring-harmonic X couples the slots where the operator does);
!!  - Galerkin coarse operators P(g)' A(g-1) P(g) (galerkin_chain, refilled at
!!    every build), MUMPS LU on the coarsest level;
!!  - on every other level the same smoother before and after the coarse
!!    correction (a symmetric V-cycle): a Chebyshev polynomial of degree d of
!!    an additive Schwarz preconditioner M on [safety lmax / ratio, safety
!!    lmax], lmax from esteig_its CG (Lanczos) steps on M^-1 A (PETSc's
!!    KSPCHEBYSHEV with KSPChebyshevEstEigSet(0, safety / ratio, 0, safety));
!!  - M = sum over blocks of R_b' A_bb^-1 R_b, every block solved exactly
!!    (LAPACK LU up to dense_max rows, PETSc's sparse LU above). Block layouts
!!    (-vs_pmg_blocks):
!!      node   all DOFs and slots of a node; the rings 0..axis_rings as one
!!             block (study: node blocks with rings 0..2 at the axis)
!!      line   all rings line_from.. at one ring position J (a radial line);
!!             line_from defaults to axis_rings + 1, a smaller one makes the
!!             lines overlap the axis block (study: the resistive P on fine
!!             grids)
!!      linewedge  the lines, and wedges: blocks of wedge_lines neighbouring
!!             lines (J0..J0 + wedge_lines - 1, periodic, J0 every
!!             wedge_stride lines) on the rings line_from..t, t the last ring
!!             whose poloidal / radial cell aspect on the level (the finest
!!             level's ring median times n_tht / nj) is below wedge_aspect
!!             (none on a level without such a ring), with the weight
!!             wedge_weight in the sum (smoother study, smoothers/smlib.py
!!             block_sets kind "line" with wedge; spec
!!             cheb1:line:4:a3:o2:w4s2q50:v50 = the recommended set
!!             -vs_pmg_blocks linewedge -vs_pmg_axis_rings 3
!!             -vs_pmg_line_from 2, the wedge defaults)
!!      point  every DOF with its slots alone, the axis block as above
!!    M = sum_b w_b R_b' A_bb^-1 R_b, w_b = 1 except the wedges;
!!    A block may cross ranks (its rows are gathered by a scatter, its
!!    solution added back) and blocks may overlap (then M is additive
!!    Schwarz), so another layout, e.g. overlapping patches, is one more
!!    labelling routine (block_members), nothing else;
!!  - the outer Krylov method of the P solve: FGMRES (restart and iteration cap
!!    of the caller, rtol physics_pc_sf_rtol); CG with -vs_pmg_ksp_type cg.
!!
!! Instances: one hierarchy per operator, each with its own options under its
!! own prefix and its own defaults:
!!   SPDMG_P   P_lumped, prefix vs_pmg_   (the defaults below)
!!   SPDMG_PV  P_V,      prefix vs_pvmg_  (blocks linewedge, axis_rings 3,
!!                                         line_from 2: the study's R2a, the
!!                                         cycle of the v_par study)
!! The routines work on the active instance H, which spdmg_ksp_setup and the
!! smoothers' shell PCs (by their handle) select.
!!
!! PETSc options (shown for vs_pmg_, likewise for vs_pvmg_; defaults = the
!! study's cycle):
!!   -vs_pmg_blocks node | line | linewedge | point   block layout        (node)
!!   -vs_pmg_axis_rings <k>                axis block = rings 0..k        (2)
!!   -vs_pmg_line_from <i>                 first ring of a radial line    (k + 1)
!!   -vs_pmg_wedge_lines <w>               linewedge: lines in a wedge    (4)
!!   -vs_pmg_wedge_stride <s>              a wedge every s lines          (2)
!!   -vs_pmg_wedge_aspect <q>              wedge rings: cell aspect < q   (0.5)
!!   -vs_pmg_wedge_weight <v>              a wedge's weight               (0.5)
!!   -vs_pmg_degree <d>                    Chebyshev degree               (4)
!!   -vs_pmg_cheb_ratio <r>                interval [safety lmax / r, ..] (10)
!!   -vs_pmg_cheb_safety <s>               lmax factor                    (1.1)
!!   -vs_pmg_esteig_its <n>                CG steps of the lmax estimate  (12)
!!   -vs_pmg_esteig_reuse                  the first build's intervals on every
!!                                         later build: no estimate there (the
!!                                         refilled operator's lmax drifts with
!!                                         the state; off: estimated per build)
!!   -vs_pmg_transfer mixed | ringharm | linear | const   the transfer along
!!                                         the ring                       (mixed)
!!   -vs_pmg_rh_sweeps <k>                 ring-harmonic Jacobi steps     (4)
!!   -vs_pmg_rh_reach <r>                  its reach along the ring       (3)
!!   -vs_pmg_rh_omega <w>                  its damping                    (1)
!!   -vs_pmg_rh_family                     with physics_pc_sf_harm_couple = .f.
!!                                         (no couplings between |n| families in
!!                                         the operator, so none in X either):
!!                                         P(g) leaves out the exact zeros
!!                                         between families that it would store,
!!                                         and the Galerkin levels lose them (m3:
!!                                         C_op 2.46 -> 1.76); the levels'
!!                                         patterns change, hence the rounding
!!                                         (no effect with .t.)          (off)
!!   -vs_pmg_levels <n>                    at most n levels               (all)
!!   -vs_pmg_min_nj <n>                    fewest ring positions          (3)
!!   -vs_pmg_dense_max <n>                 largest block with a dense LU  (300)
!!   -vs_pmg_blockmv                       the levels' operators (the operator
!!                                         and the Galerkin levels) multiply on
!!                                         JOREK's OpenMP block kernel
!!                                         (jorek_blockmv_attach.c, as the ring
!!                                         GMG's levels) instead of PETSc's
!!                                         MatMult: threads in the matvecs,
!!                                         another summation order      (off)
!!   -vs_pmg_band                          a block whose band in its row order is
!!                                         narrow (2 kl + ku + 1 <= rows / 2:
!!                                         the radial lines), whatever its size,
!!                                         as a LAPACK band LU (dgbtrf / dgbtrs)
!!                                         instead of a dense LU or a PETSc KSP
!!                                         with a sparse LU (off): about the
!!                                         flops of the sparse LU (m4: +16 % per
!!                                         line solve against nd), no KSP per
!!                                         block, and solved on the OpenMP
!!                                         threads as the dense blocks are.
!!                                         With -vs_pmg_dense_max above the
!!                                         axis block's rows, that block is a
!!                                         dense LAPACK LU (MKL's threads in its
!!                                         factorisation) and the lines band
!!   -vs_pmg_sparse_ordering nd | natural | rcm | qmd | 1wd   the ordering of
!!                                         the sparse block LUs, all of them
!!                                         (m4: natural takes 24 % fewer solve
!!                                         flops on the radial lines, banded
!!                                         in their ring-major order, but 15 %
!!                                         more factor flops on the axis
!!                                         block)                       (nd)
!!   -vs_pmg_block_owner first | balanced  the rank that solves a block: the
!!                                         owner of its smallest DOF, or for a
!!                                         block across ranks (a radial line,
!!                                         a wedge, the axis block) the least
!!                                         loaded of its ranks (first: every
!!                                         radial line on the rank of ring
!!                                         line_from). Bitwise the same at np 1;
!!                                         on more ranks the additive sum of
!!                                         overlapping blocks is added in
!!                                         another order               (first)
!!   -vs_pmg_test                          on the first build: CG with one cycle
!!                                         on a random right-hand side to 1e-10,
!!                                         its to 1e-1 / 1e-6 / 1e-10 and the
!!                                         Lanczos kappa (as the study's tables),
!!                                         for all slots and family by family
!!   -vs_pmg_view_p <prefix>               P(g) of the first build to
!!                                         <prefix>_P<g>.petsc (PETSc binary)
!!   and every PETSc option under the prefix vs_pmg_ (KSPSetFromOptions after
!!   the defaults): -vs_pmg_ksp_type cg, -vs_pmg_mg_levels_ksp_max_it,
!!   -vs_pmg_mg_levels_ksp_chebyshev_esteig a,b,c,d, -vs_pmg_mg_levels_pc_type
!!   (a different smoother), -vs_pmg_ksp_view, ...
!!
!! The smoother is the one piece the options do not reach beyond the block
!! layout and the polynomial: another smoother is another PC on the levels'
!! KSPs (-vs_pmg_mg_levels_pc_type, or a shell like as_apply).
!!
!! Limits: polar grids of n_flux x n_tht nodes only (the gates of
!! mod_petsc_pc_gmg); one hierarchy (one operator) per instance and run; the
!! operator's pattern must stay that of the first build (refills only). The
!! Chebyshev interval is estimated on the whole operator: with several toroidal
!! families the interval is the union's, not each family's (on m1, n = 0 then
!! gets kappa 1.5 to 1.6 where its own interval gives 1.3).
module mod_petsc_pc_spdmg
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: spdmg_ksp_setup

  !> the instances: P_lumped's and P_V's hierarchies
  integer, parameter, public :: SPDMG_P = 1, SPDMG_PV = 2
  integer, parameter :: NINST = 2

  integer, parameter :: MAXL = 12
  integer, parameter :: BL_NODE = 1, BL_LINE = 2, BL_POINT = 3, BL_LINEWEDGE = 4
  integer, parameter :: TR_LINEAR = 1, TR_CONST = 2, TR_RINGHARM = 3, TR_MIXED = 4
  integer, parameter :: OWN_FIRST = 1, OWN_BALANCED = 2

  !> the configuration (PETSc options, module header)
  type :: spdmg_opts_t
    integer :: blocks     = BL_NODE
    integer :: axis_rings = 2
    integer :: line_from  = -1         !< the first ring of a radial line (-1: axis_rings + 1)
    integer :: wedge_lines  = 4        !< linewedge: lines in a wedge
    integer :: wedge_stride = 2        !< a wedge every wedge_stride lines
    real*8  :: wedge_aspect = 0.5d0    !< the rings of a wedge: cell aspect below this
    real*8  :: wedge_weight = 0.5d0    !< a wedge's weight in the additive sum
    integer :: degree     = 4
    real*8  :: ratio      = 10.d0
    real*8  :: safety     = 1.1d0
    integer :: esteig_its = 12
    logical :: esteig_reuse = .false.  !< the first build's Chebyshev intervals for every later build
    integer :: transfer   = TR_MIXED
    integer :: rh_sweeps  = 4          !< ring-harmonic transfer: Jacobi steps
    integer :: rh_reach   = 3          !< and the reach of X along the ring
    real*8  :: rh_omega   = 1.d0       !< and the step's damping
    logical :: rh_family  = .false.    !< -vs_pmg_rh_family: no entries between |n| families in P(g) (harm_couple .f.)
    integer :: max_levels = MAXL
    integer :: min_nj     = 3
    integer :: dense_max  = 300
    character(len=16) :: sp_order = "nd" !< the ordering of the sparse block LUs (-vs_pmg_sparse_ordering)
    logical :: band = .false.          !< -vs_pmg_band: LAPACK band LUs for the blocks with a narrow band
    logical :: blockmv = .false.       !< -vs_pmg_blockmv: the levels' matvecs on the OpenMP block kernel
    integer :: block_owner = OWN_FIRST !< the rank solving a block (block_members)
    logical :: test       = .false.
    character(len=256) :: view_p = ""  !< -vs_pmg_view_p <prefix>: P(g) of the first build to <prefix>_P<g>.petsc
  end type spdmg_opts_t

  !> The ring-harmonic transfer of one level g (P(g) from the operator of
  !! level g-1): the rows of the F nodes of the rings this rank needs, all
  !! columns of those rings, gathered once (isr, isc, sub; reused).
  type :: rh_t
    logical :: ready = .false.
    PetscInt, allocatable :: rrow(:), rcol(:)      !< the gathered global rows / columns, ascending
    IS  :: isr, isc
    Mat, pointer :: sub(:) => null()
  end type rh_t

  !> The DOFs of one level, known to every rank. Level 0 is the operator's own
  !! index space (JOREK indices, DOF = index); a coarse level numbers its DOFs
  !! in (I, J, k) order, the shared axis value once.
  type :: lev_t
    integer :: nj = 0, n = 0                       !< ring positions, DOFs
    integer, allocatable :: d(:,:,:)               !< (0:ni-1, 0:nj-1, 0:3) -> DOF (0-based), -1 none
    integer, allocatable :: dI(:), dJ(:), dK(:)    !< DOF -> its first lattice position (I, J, k)
    integer, allocatable :: own(:)                 !< DOF -> owner rank
    PetscInt, allocatable :: row0(:)               !< DOF -> global row of its slot 0
    logical, allocatable :: ok(:)                  !< DOF has a coarse part (level 0: free and used)
    integer, allocatable :: osgn(:)                !< DOF orientation (+-1; level 0 only, else 1)
    PetscInt :: nloc = 0                           !< this rank's rows
  end type lev_t

  !> The additive Schwarz preconditioner of one level (the Chebyshev
  !! smoother's PC): this rank's blocks, their rows gathered into w.
  type :: asm_t
    logical :: ready = .false.
    integer :: nb = 0, nw = 0, nu = 0
    integer, allocatable :: off(:), sz(:)          !< block -> offset in w (0-based), rows
    real*8, allocatable  :: bw(:)                  !< block -> its weight in the sum
    PetscInt, allocatable :: gidx(:)               !< w position -> global row
    integer, allocatable :: upos(:)                !< w position -> position in the union (0-based)
    IS  :: isu                                     !< the union of the rows, ascending (COMM_SELF)
    Mat, pointer :: sub(:) => null()               !< A(union, union)
    VecScatter :: sct
    Vec :: w
    logical, allocatable :: dense(:)
    logical, allocatable :: band(:)                !< -vs_pmg_band: a LAPACK band LU in lu (kl, ku)
    integer, allocatable :: kl(:), ku(:)           !< its band widths
    integer(8), allocatable :: loff(:)             !< dense or band block -> offset of its LU in lu
    real*8, allocatable :: lu(:)
    integer, allocatable :: piv(:)                 !< pivots, at off
    KSP, allocatable :: kb(:)                      !< sparse block -> its LU
    Mat, allocatable :: mb(:)                      !< its matrix
    IS,  allocatable :: ib(:)                      !< its rows in the union
    Vec, allocatable :: vb(:), vx(:)               !< its right-hand side and solution
    integer :: nsparse = 0
  end type asm_t

  !> One hierarchy (an instance) with its configuration and state.
  type :: spdmg_inst_t
    type(spdmg_opts_t) :: o
    character(len=16) :: prefix = ""                !< the PETSc option prefix (vs_pmg_, vs_pvmg_)
    character(len=16) :: tag = ""                   !< the log label (SPD-MG, SPD-MG P_V)
    logical :: built = .false.
    integer :: nlev = 0, comm = MPI_COMM_NULL, me = 0, np = 1, ns = 1
    type(lev_t) :: lv(0:MAXL-1)
    Mat :: P(1:MAXL-1)                              !< P(g): level g -> level g-1
    Mat :: Ag(1:MAXL-1)                             !< the Galerkin operators P(g)' A(g-1) P(g)
    type(rh_t) :: rh(1:MAXL-1)
    type(asm_t) :: as(0:MAXL-1)
    PC :: as_pc(0:MAXL-1)                           !< the shell PC of level g's smoother
    real*8, allocatable :: hist(:)                  !< the test's residual history
    real*8, allocatable :: ring_asp(:)              !< (0:n_flux-1) the finest level's ring cell aspect (wedge_last)
    logical, allocatable :: rh_slot(:)              !< (0:ns-1) the slot's F rows are ring-harmonic
    real*8 :: rh_ab(2, 1:MAXL-1) = 0.d0             !< bytes of the ring-harmonic Ab blocks zeroed, summed over
                                                    !! the rank's rings and builds: as sized, as nF x nF (before)
    real*8 :: cheb_iv(2, 0:MAXL-1) = 0.d0           !< -vs_pmg_esteig_reuse: level g's [emax, emin]
    logical :: cheb_frozen = .false.                !< the intervals are saved (and set on later builds)
  end type spdmg_inst_t

  type(spdmg_inst_t), target, save :: inst(NINST)
  type(spdmg_inst_t), pointer, save :: H => null()  !< the active instance

  !> -log_view events (SPDMG_*, shared by the instances): the levels (first
  !! build), the transfers and Galerkin products, the KSP setup (smoother
  !! factors, Chebyshev estimates, coarse LU), the additive Schwarz setup and
  !! apply (its block solves alone: ASMBlk), the test. LAPACK flops are logged
  !! with PetscLogFlops.
  PetscLogEvent, save :: sev_levels = -1, sev_transfer = -1, sev_ptap = -1, sev_kspsetup = -1, &
                         sev_asmsetup = -1, sev_asmapply = -1, sev_asmblk = -1, sev_test = -1
  logical, save :: sev_ready = .false.

contains

  !> The -log_view events of the module, once.
  subroutine spdmg_register_events()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    if (sev_ready) return
    call PetscLogEventRegister("SPDMG_Levels",   cid, sev_levels,   ierr)
    call PetscLogEventRegister("SPDMG_Transfer", cid, sev_transfer, ierr)
    call PetscLogEventRegister("SPDMG_PtAP",     cid, sev_ptap,     ierr)
    call PetscLogEventRegister("SPDMG_KSPSetUp", cid, sev_kspsetup, ierr)
    call PetscLogEventRegister("SPDMG_ASMSetup", cid, sev_asmsetup, ierr)
    call PetscLogEventRegister("SPDMG_ASMApply", cid, sev_asmapply, ierr)
    call PetscLogEventRegister("SPDMG_ASMBlk",   cid, sev_asmblk,   ierr)
    call PetscLogEventRegister("SPDMG_Test",     cid, sev_test,     ierr)
    sev_ready = .true.
  end subroutine spdmg_register_events

  !> Configure ksp as the solver of operator A (one field, ns slots per JOREK
  !! index, rows index * ns + slot; mod_petsc_pc_sf_ops) with instance k's
  !! hierarchy. fresh: ksp was just created (the hierarchy is built on the
  !! instance's first call of the run); later calls take the refilled A (same
  !! pattern). amat: the Krylov method's operator, if not A (the cycle stays
  !! A's). Collective on comm.
  subroutine spdmg_ksp_setup(ksp, A, ns, comm, my_id, rtol, maxits, label, fresh, k, amat)
    KSP, intent(inout)           :: ksp
    Mat, intent(in)              :: A
    integer, intent(in)          :: ns, comm, my_id, maxits
    real*8, intent(in)           :: rtol
    character(len=*), intent(in) :: label
    logical, intent(in)          :: fresh
    integer, intent(in)          :: k              !< SPDMG_P | SPDMG_PV
    Mat, intent(in), optional    :: amat
    PC :: pc
    PetscErrorCode :: ierr
    logical :: first_chain

    H => inst(k)
    first_chain = .not. H%built
    call spdmg_register_events()
    if (.not. H%built) then
      call PetscLogEventBegin(sev_levels, ierr)
      H%comm = comm; H%ns = ns
      call MPI_Comm_rank(comm, H%me, ierr)
      call MPI_Comm_size(comm, H%np, ierr)
      call instance_defaults(k)
      call read_options()
      call build_levels(A, my_id)
      call create_prolongations()
      H%built = .true.
      call PetscLogEventEnd(sev_levels, ierr)
    endif
    ! the levels' transfers (ring-harmonic: from the operators) and Galerkin operators
    call galerkin_chain(A, first_chain)
    ! -vs_pmg_blockmv: the levels' matvecs on JOREK's OpenMP block kernel (as
    ! the ring GMG's), attached once (the kernel follows the refills); the
    ! cycle's operators only, not amat
    if (first_chain .and. H%o%blockmv) then
      block
        use mod_petsc_raw_csr, only: blockmv_attach
        integer :: g_, nok
        nok = 0
        if (blockmv_attach(A, H%ns)) nok = nok + 1
        do g_ = 1, H%nlev - 1
          if (blockmv_attach(H%Ag(g_), H%ns)) nok = nok + 1
        enddo
        if (H%me == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   "//trim(H%tag)//": ", nok, " of ", H%nlev, &
          " level operators on the OpenMP block matvec kernel (-"//trim(H%prefix)//"blockmv)"
      end block
    endif
    if (fresh) then
      call KSPSetType(ksp, KSPFGMRES, ierr)
      call KSPGMRESSetRestart(ksp, maxits, ierr)
      call KSPSetTolerances(ksp, rtol, 1.d-50, 1.d6, maxits, ierr)
      call KSPSetOptionsPrefix(ksp, trim(H%prefix), ierr)
      call KSPGetPC(ksp, pc, ierr)
      call configure_pcmg(pc, A)
      call set_operators()
      call KSPSetFromOptions(ksp, ierr)
    else
      call set_operators()
      if (H%cheb_frozen) call cheb_freeze(ksp)
    endif
    call PetscLogEventBegin(sev_kspsetup, ierr)
    call KSPSetUp(ksp, ierr)
    call PetscLogEventEnd(sev_kspsetup, ierr)
    if (H%o%esteig_reuse .and. .not. H%cheb_frozen) call cheb_save(ksp)
    if (fresh) then
      call report(ksp, tagged(), my_id)
      call zero_report(A)
      if (len_trim(H%o%view_p) > 0) call view_transfers()
      if (H%o%test) then
        call PetscLogEventBegin(sev_test, ierr)
        call run_test(ksp, A, tagged(), my_id)
        call PetscLogEventEnd(sev_test, ierr)
      endif
    endif
  contains
    subroutine set_operators()
      if (present(amat)) then
        call KSPSetOperators(ksp, amat, A, ierr)
      else
        call KSPSetOperators(ksp, A, A, ierr)
      endif
    end subroutine set_operators
    !> the label with the operator's global slots where it holds a part of them (mode split)
    function tagged() result(s)
      use mod_parameters,    only: n_tor
      use mod_petsc_pc_harm, only: pc_global_slot
      character(len=64) :: s
      s = label
      if (H%ns /= n_tor) write(s, '(A,A,I0,A,I0)') trim(label), ", slots ", pc_global_slot(0), "-", &
                                                    pc_global_slot(H%ns - 1)
    end function tagged
  end subroutine spdmg_ksp_setup

  !> Instance k's option prefix, log label and the defaults that differ from
  !! spdmg_opts_t's (module header).
  subroutine instance_defaults(k)
    integer, intent(in) :: k
    select case (k)
    case (SPDMG_P)
      H%prefix = "vs_pmg_";  H%tag = "SPD-MG"
    case (SPDMG_PV)
      H%prefix = "vs_pvmg_"; H%tag = "SPD-MG P_V"
      H%o%blocks = BL_LINEWEDGE; H%o%axis_rings = 3; H%o%line_from = 2
    case default
      call fatal("an unknown instance")
    end select
  end subroutine instance_defaults

  !> -vs_mem_report (first build): each level operator's stored entries,
  !! those exactly zero, those between two |n| families (row and column
  !! slots of different families) and the exact zeros among them, summed
  !! over the ranks. Collective.
  subroutine zero_report(A0)
    use mod_petsc_pc_harm, only: pc_global_slot
    Mat, intent(in) :: A0
    PetscBool :: flg
    PetscErrorCode :: ierr
    PetscInt :: rs, re, r, nc, c
    PetscInt, pointer :: cl(:)
    PetscScalar, pointer :: vl(:)
    real*8 :: v(6), vs(6)
    integer :: g, fr, fc
    Mat :: M
    call PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_mem_report", flg, ierr)
    if (.not. flg) return
    do g = 0, H%nlev - 1
      M = A0
      if (g > 0) M = H%Ag(g)
      call MatGetOwnershipRange(M, rs, re, ierr)
      v = 0.d0
      do r = rs, re - 1
        call MatGetRow(M, r, nc, cl, vl, ierr)
        fr = (pc_global_slot(int(mod(r, int(H%ns, kind(r))))) + 1) / 2
        v(1) = v(1) + dble(nc)
        do c = 1, nc
          fc = (pc_global_slot(int(mod(cl(c), int(H%ns, kind(r))))) + 1) / 2
          if (vl(c) == 0.d0) v(2) = v(2) + 1.d0
          v(5) = max(v(5), abs(vl(c)))
          if (fc /= fr) then
            v(3) = v(3) + 1.d0
            if (vl(c) == 0.d0) v(4) = v(4) + 1.d0
            v(6) = max(v(6), abs(vl(c)))
          endif
        enddo
        call MatRestoreRow(M, r, nc, cl, vl, ierr)
      enddo
      call MPI_Allreduce(v(1:4), vs(1:4), 4, MPI_DOUBLE_PRECISION, MPI_SUM, H%comm, ierr)
      call MPI_Allreduce(v(5:6), vs(5:6), 2, MPI_DOUBLE_PRECISION, MPI_MAX, H%comm, ierr)
      if (H%me == 0) write(*,'(A,A,I0,A,ES9.2,A,ES9.2,A,F5.1,A,ES9.2,A,F5.1,A,ES9.2,A,ES9.2)') "[VS mem] ", &
        trim(H%tag)//" level ", g, ": stored ", vs(1), ", exact zeros ", vs(2), " (", 100.d0 * vs(2) / max(vs(1), 1.d0), &
        " %), between |n| families ", vs(3), " (", 100.d0 * vs(3) / max(vs(1), 1.d0), " %), of them zero ", vs(4), &
        ", their largest |a| / largest |a| ", vs(6) / max(vs(5), tiny(1.d0))
    enddo
    do g = 1, H%nlev - 1
      call MPI_Allreduce(H%rh_ab(:, g), v(1:2), 2, MPI_DOUBLE_PRECISION, MPI_MAX, H%comm, ierr)
      if (H%me == 0 .and. v(2) > 0.d0) write(*,'(A,A,I0,A,F10.2,A,F10.2,A)') "[VS mem] ", trim(H%tag)//" transfer ", &
        g, ": ring-harmonic neighbour blocks zeroed per build, largest rank ", v(1) / 1.d6, " MB (as nF x nF: ", &
        v(2) / 1.d6, " MB)"
    enddo
  end subroutine zero_report

  !> -vs_pmg_esteig_reuse: the Chebyshev intervals of the levels' smoothers
  !! as the first build's estimates set them, [b lmax, d lmax] with lmax the
  !! largest Lanczos eigenvalue of the estimate and (b, d) = (safety / ratio,
  !! safety) (PETSc's transform with a = c = 0), saved after that build.
  subroutine cheb_save(ksp)
    KSP, intent(in) :: ksp
    PC :: pc
    KSP :: ks, kest
    PetscInt :: L, n, neig
    PetscErrorCode :: ierr
    real*8, allocatable :: re(:), im(:)
    real*8 :: lmax
    integer :: g
    call KSPGetPC(ksp, pc, ierr)
    do g = 0, H%nlev - 2
      L = H%nlev - 1 - g
      call PCMGGetSmoother(pc, L, ks, ierr)
      call KSPChebyshevEstEigGetKSP(ks, kest, ierr)
      call KSPGetIterationNumber(kest, n, ierr)
      allocate(re(max(n, 1)), im(max(n, 1)))
      neig = 0
      if (n > 0) call KSPComputeEigenvalues(kest, n, re, im, neig, ierr)
      if (neig < 1) call fatal("-"//trim(H%prefix)//"esteig_reuse: no eigenvalue estimate on a level")
      lmax = maxval(re(1:neig))
      H%cheb_iv(1, g) = H%o%safety * lmax
      H%cheb_iv(2, g) = (H%o%safety / H%o%ratio) * lmax
      deallocate(re, im)
    enddo
    H%cheb_frozen = .true.
    if (H%me == 0) write(*,'(A,I0,A)') "[Physics PC]   "//trim(H%tag)//": the Chebyshev intervals of the first "// &
      "build kept for the later builds (-"//trim(H%prefix)//"esteig_reuse), ", H%nlev - 1, " levels"
  end subroutine cheb_save

  !> The saved intervals on the levels' smoothers, estimation off (later builds).
  subroutine cheb_freeze(ksp)
    KSP, intent(in) :: ksp
    PC :: pc
    KSP :: ks
    PetscInt :: L
    PetscErrorCode :: ierr
    integer :: g
    call KSPGetPC(ksp, pc, ierr)
    do g = 0, H%nlev - 2
      L = H%nlev - 1 - g
      call PCMGGetSmoother(pc, L, ks, ierr)
      call KSPChebyshevEstEigSet(ks, 0.d0, 0.d0, 0.d0, 0.d0, ierr)
      call KSPChebyshevSetEigenvalues(ks, H%cheb_iv(1, g), H%cheb_iv(2, g), ierr)
    enddo
  end subroutine cheb_freeze

  !> P(g) in PETSc's binary format to <view_p>_P<g>.petsc (rows of level
  !! g-1, columns of level g, the row numbering of lv). Collective.
  subroutine view_transfers()
    PetscViewer :: vw
    PetscErrorCode :: ierr
    integer :: g
    character(len=300) :: fn
    do g = 1, H%nlev - 1
      write(fn, '(A,A,I0,A)') trim(H%o%view_p), "_P", g, ".petsc"
      call PetscViewerBinaryOpen(H%comm, trim(fn), FILE_MODE_WRITE, vw, ierr)
      call MatView(H%P(g), vw, ierr)
      call PetscViewerDestroy(vw, ierr)
    enddo
    if (H%me == 0) write(*,'(A)') "[Physics PC]   "//trim(H%tag)//" transfers written to "//trim(H%o%view_p)// &
                                  "_P<g>.petsc"
  end subroutine view_transfers

  !> The options of the module header, under the instance's prefix.
  subroutine read_options()
    PetscErrorCode :: ierr
    PetscBool :: is_set, flag
    PetscInt :: iv
    PetscReal :: rv
    character(len=16) :: s
    character(len=24) :: pf
    pf = "-"//trim(H%prefix)
    s = blocks_name(H%o%blocks)
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"blocks", s, is_set, ierr)
    select case (trim(s))
    case ("node");      H%o%blocks = BL_NODE
    case ("line");      H%o%blocks = BL_LINE
    case ("point");     H%o%blocks = BL_POINT
    case ("linewedge"); H%o%blocks = BL_LINEWEDGE
    case default;   call fatal(trim(pf)//"blocks must be node | line | point | linewedge, got '"//trim(s)//"'")
    end select
    call get_int("line_from", H%o%line_from)
    call get_int("wedge_lines", H%o%wedge_lines)
    call get_int("wedge_stride", H%o%wedge_stride)
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"wedge_aspect", rv, is_set, ierr)
    if (is_set) H%o%wedge_aspect = rv
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"wedge_weight", rv, is_set, ierr)
    if (is_set) H%o%wedge_weight = rv
    if (H%o%wedge_lines < 1 .or. H%o%wedge_stride < 1) &
      call fatal(trim(pf)//"wedge_lines, "//trim(pf)//"wedge_stride must be >= 1")
    s = "mixed"
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"transfer", s, is_set, ierr)
    select case (trim(s))
    case ("linear");   H%o%transfer = TR_LINEAR
    case ("const");    H%o%transfer = TR_CONST
    case ("ringharm"); H%o%transfer = TR_RINGHARM
    case ("mixed");    H%o%transfer = TR_MIXED
    case default;      call fatal(trim(pf)//"transfer must be mixed | ringharm | linear | const, got '"//trim(s)//"'")
    end select
    s = "first"
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"block_owner", s, is_set, ierr)
    select case (trim(s))
    case ("first");    H%o%block_owner = OWN_FIRST
    case ("balanced"); H%o%block_owner = OWN_BALANCED
    case default;      call fatal(trim(pf)//"block_owner must be first | balanced, got '"//trim(s)//"'")
    end select
    call get_int("rh_sweeps", H%o%rh_sweeps)
    call get_int("rh_reach", H%o%rh_reach)
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"rh_omega", rv, is_set, ierr)
    if (is_set) H%o%rh_omega = rv
    if (H%o%rh_sweeps < 1 .or. H%o%rh_reach < 1) &
      call fatal(trim(pf)//"rh_sweeps and "//trim(pf)//"rh_reach must be >= 1")
    call get_int("axis_rings", H%o%axis_rings)
    call get_int("degree", H%o%degree)
    call get_int("esteig_its", H%o%esteig_its)
    call get_int("levels", H%o%max_levels)
    call get_int("min_nj", H%o%min_nj)
    call get_int("dense_max", H%o%dense_max)
    s = "nd"
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"sparse_ordering", s, is_set, ierr)
    select case (trim(s))
    case ("nd", "natural", "rcm", "qmd", "1wd"); H%o%sp_order = trim(s)
    case default; call fatal(trim(pf)//"sparse_ordering must be nd | natural | rcm | qmd | 1wd, got '"//trim(s)//"'")
    end select
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"cheb_ratio", rv, is_set, ierr)
    if (is_set) H%o%ratio = rv
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"cheb_safety", rv, is_set, ierr)
    if (is_set) H%o%safety = rv
    call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"test", flag, is_set, ierr)
    if (is_set) H%o%test = flag
    call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"esteig_reuse", flag, is_set, ierr)
    if (is_set) H%o%esteig_reuse = flag
    call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"band", flag, is_set, ierr)
    if (is_set) H%o%band = flag
    call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"blockmv", flag, is_set, ierr)
    if (is_set) H%o%blockmv = flag
    call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"rh_family", flag, is_set, ierr)
    if (is_set) H%o%rh_family = flag
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//"view_p", H%o%view_p, is_set, ierr)
    if (.not. is_set) H%o%view_p = ""
    H%o%max_levels = min(max(H%o%max_levels, 1), MAXL)
    if (H%o%axis_rings < 0) call fatal(trim(pf)//"axis_rings must be >= 0")
    if (H%o%degree < 1 .or. H%o%esteig_its < 1 .or. H%o%ratio <= 1.d0 .or. H%o%safety <= 0.d0) &
      call fatal(trim(pf)//"degree, "//trim(pf)//"esteig_its >= 1, "//trim(pf)//"cheb_ratio > 1, "// &
                 trim(pf)//"cheb_safety > 0")
  contains
    subroutine get_int(nm, v)
      character(len=*), intent(in) :: nm
      integer, intent(inout) :: v
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(pf)//nm, iv, is_set, ierr)
      if (is_set) v = int(iv)
    end subroutine get_int
  end subroutine read_options

  !> The name of a block layout (-vs_pmg_blocks).
  function blocks_name(b) result(s)
    integer, intent(in) :: b
    character(len=16) :: s
    select case (b)
    case (BL_LINE);      s = "line"
    case (BL_POINT);     s = "point"
    case (BL_LINEWEDGE); s = "linewedge"
    case default;        s = "node"
    end select
  end function blocks_name

  !> The levels' DOFs, owners and rows (lv), from the polar grid and A's row
  !! layout and Dirichlet rows.
  subroutine build_levels(A, my_id)
    use nodes_elements, only: node_list, element_list
    use phys_module,    only: n_flux, n_tht
    use mod_parameters, only: n_degrees
    use mod_petsc_pc_blocks, only: penalty_row_mask
    Mat, intent(in)     :: A
    integer, intent(in) :: my_id
    integer, allocatable :: orient(:,:), i0s(:), cnt(:)
    logical, allocatable :: oset(:,:), freeb(:)
    integer :: n_idx, e, iv, n, k, i, j, g, c, f, ni, nj, mpierr, r, nfree
    integer :: vi(4), vj(4), eps_s(4), eps_t(4), cand
    real*8  :: sz(4)
    PetscInt :: rs, re, ng
    PetscErrorCode :: ierr
    PetscScalar, pointer :: mp(:)
    Vec :: mask
    integer, allocatable :: fl(:)

    eps_s = [1, -1, -1, 1]
    eps_t = [1, 1, -1, -1]
    if (n_degrees /= 4) call fatal("needs the bicubic Hermite element (n_degrees = 4)")
    ! gate 1: the (i-1)*n_tht + j polar layout of n_flux rings (mod_petsc_pc_gmg)
    if (node_list%n_nodes /= n_flux * n_tht .or. n_flux < 3 .or. &
        element_list%n_elements /= (n_flux - 1) * n_tht) &
      call fatal("needs a periodic flux-surface (polar) grid of n_flux x n_tht nodes; X-point grids are not "// &
                 "supported, use the ring GMG (-vs_p_solver ring_gmg)")
    do e = 1, element_list%n_elements
      do iv = 1, 4
        n = element_list%element(e)%vertex(iv)
        vi(iv) = (n - 1) / n_tht; vj(iv) = mod(n - 1, n_tht)
      enddo
      if (.not. (vi(2) == vi(1) + 1 .and. vj(2) == vj(1) .and. vi(3) == vi(1) + 1 .and. &
                 vj(3) == mod(vj(1) + 1, n_tht) .and. vi(4) == vi(1) .and. vj(4) == mod(vj(1) + 1, n_tht))) &
        call fatal("element vertices are not the structured (i, j) layout of a polar grid")
    enddo
    ! gate 2: the element size factors are a per-node orientation +-1 (0: an
    ! angular axis DOF no basis function uses)
    allocate(orient(node_list%n_nodes, 0:3), oset(node_list%n_nodes, 0:3))
    orient = 0; oset = .false.
    do e = 1, element_list%n_elements
      do iv = 1, 4
        n = element_list%element(e)%vertex(iv)
        sz = [element_list%element(e)%size(iv, 1), eps_s(iv) * element_list%element(e)%size(iv, 2), &
              eps_t(iv) * element_list%element(e)%size(iv, 3), eps_s(iv) * eps_t(iv) * element_list%element(e)%size(iv, 4)]
        do k = 0, 3
          if (abs(abs(sz(k + 1)) - 1.d0) < 1.d-12) then
            cand = nint(sign(1.d0, sz(k + 1)))
          else if (sz(k + 1) == 0.d0 .and. k >= 2 .and. node_list%node(n)%axis_node) then
            cand = 0
          else
            call fatal("an element size factor is not +-1")
          endif
          if (.not. oset(n, k)) then
            orient(n, k) = cand; oset(n, k) = .true.
          else if (orient(n, k) /= cand) then
            call fatal("inconsistent DOF orientation between elements")
          endif
        enddo
      enddo
    enddo

    ! the ring cell aspects r dtheta / dr for the wedges (smoothers/smlib.py ring_aspect):
    ! ring i = 1 .. n_flux - 2, the median (numpy's) over the cells between rings i and i + 1
    allocate(H%ring_asp(0:n_flux - 1))
    H%ring_asp = 0.d0
    block
      real*8 :: cell(n_tht), xa(2), xb(2), xc(2)
      integer :: jj
      do i = 1, n_flux - 2
        do jj = 0, n_tht - 1
          xa = node_list%node(i * n_tht + jj + 1)%x(1, 1, 1:2)
          xb = node_list%node((i + 1) * n_tht + jj + 1)%x(1, 1, 1:2)
          xc = node_list%node(i * n_tht + mod(jj + 1, n_tht) + 1)%x(1, 1, 1:2)
          cell(jj + 1) = norm2(xc - xa) / max(norm2(xb - xa), 1.d-300)
        enddo
        call sort_real(cell)
        if (mod(n_tht, 2) == 1) then
          H%ring_asp(i) = cell((n_tht + 1) / 2)
        else
          H%ring_asp(i) = 0.5d0 * (cell(n_tht / 2) + cell(n_tht / 2 + 1))
        endif
      enddo
      H%ring_asp(n_flux - 1) = H%ring_asp(n_flux - 2)
    end block

    ! level 0: the operator's index space and its row layout
    call MatGetOwnershipRange(A, rs, re, ierr)
    call MatGetSize(A, ng, PETSC_NULL_INTEGER, ierr)
    if (mod(int(rs), H%ns) /= 0 .or. mod(int(re), H%ns) /= 0 .or. mod(int(ng), H%ns) /= 0) &
      call fatal("the operator's rows are not whole (index, slots) groups")
    n_idx = int(ng) / H%ns
    allocate(i0s(0:H%np))
    call MPI_Allgather(int(rs) / H%ns, 1, MPI_INTEGER, i0s, 1, MPI_INTEGER, H%comm, mpierr)
    i0s(H%np) = n_idx
    ! the free indices: a row that is not a Dirichlet row on some slot
    call penalty_row_mask(A, mask)
    allocate(fl(0:n_idx - 1)); fl = 0
    call VecGetArray(mask, mp, ierr)
    do i = int(rs) / H%ns, int(re) / H%ns - 1
      if (any(mp((i - int(rs) / H%ns) * H%ns + 1 : (i - int(rs) / H%ns + 1) * H%ns) > 0.5d0)) fl(i) = 1
    enddo
    call VecRestoreArray(mask, mp, ierr)
    call VecDestroy(mask, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, fl, n_idx, MPI_INTEGER, MPI_MAX, H%comm, mpierr)
    allocate(freeb(0:n_idx - 1)); freeb = fl > 0
    nfree = count(freeb)

    ni = n_flux; nj = n_tht
    associate (L => H%lv(0))
      L%nj = nj; L%n = n_idx
      allocate(L%d(0:ni - 1, 0:nj - 1, 0:3), L%dI(0:n_idx - 1), L%dJ(0:n_idx - 1), L%dK(0:n_idx - 1), &
               L%own(0:n_idx - 1), L%row0(0:n_idx - 1), L%ok(0:n_idx - 1), L%osgn(0:n_idx - 1))
      L%dI = -1
      do n = 1, node_list%n_nodes
        i = (n - 1) / n_tht; j = mod(n - 1, n_tht)
        do k = 0, 3
          f = node_list%node(n)%index(k + 1) - 1
          if (f < 0 .or. f >= n_idx) call fatal("a node index outside the operator")
          L%d(i, j, k) = f
          if (L%dI(f) >= 0) cycle                       ! a shared DOF: its first position
          L%dI(f) = i; L%dJ(f) = j; L%dK(f) = k
          L%osgn(f) = orient(n, k)
          L%ok(f) = freeb(f) .and. orient(n, k) /= 0
        enddo
      enddo
      if (any(L%dI < 0)) call fatal("an index without a node")
      r = 0
      do f = 0, n_idx - 1
        do while (f >= i0s(r + 1))
          r = r + 1
        enddo
        L%own(f) = r
        L%row0(f) = int(f, kind(rs)) * H%ns
      enddo
      L%nloc = re - rs
    end associate

    ! coarse levels: J halved, a DOF where the finer level's DOF at (I, 2J, k) has a coarse part
    H%nlev = 1
    do g = 1, H%o%max_levels - 1
      if (mod(H%lv(g - 1)%nj, 2) /= 0 .or. H%lv(g - 1)%nj / 2 < H%o%min_nj) exit
      nj = H%lv(g - 1)%nj / 2
      associate (L => H%lv(g), LF => H%lv(g - 1))
        L%nj = nj
        allocate(L%d(0:ni - 1, 0:nj - 1, 0:3))
        L%d = -1
        c = 0
        do i = 0, ni - 1
          do j = 0, nj - 1
            do k = 0, 3
              f = LF%d(i, 2 * j, k)
              if (f < 0) cycle
              if (.not. LF%ok(f)) cycle
              if (i == 0 .and. k == 0 .and. j > 0) then     ! the shared axis value
                L%d(i, j, k) = L%d(0, 0, 0)
                cycle
              endif
              L%d(i, j, k) = c; c = c + 1
            enddo
          enddo
        enddo
        L%n = c
        allocate(L%dI(0:c - 1), L%dJ(0:c - 1), L%dK(0:c - 1), L%own(0:c - 1), L%row0(0:c - 1), &
                 L%ok(0:c - 1), L%osgn(0:c - 1))
        L%dI = -1; L%ok = .true.; L%osgn = 1
        do i = 0, ni - 1
          do j = 0, nj - 1
            do k = 0, 3
              f = L%d(i, j, k)
              if (f < 0) cycle
              if (L%dI(f) >= 0) cycle
              L%dI(f) = i; L%dJ(f) = j; L%dK(f) = k
              L%own(f) = LF%own(LF%d(i, 2 * j, k))
            enddo
          enddo
        enddo
        ! rows: each rank's DOFs in DOF order, its block of rows after the lower ranks'
        allocate(cnt(0:H%np))
        cnt = 0
        do f = 0, c - 1
          cnt(L%own(f) + 1) = cnt(L%own(f) + 1) + 1
        enddo
        do r = 1, H%np
          cnt(r) = cnt(r) + cnt(r - 1)                   ! cnt(r): DOFs of the ranks below r
        enddo
        do f = 0, c - 1
          L%row0(f) = int(cnt(L%own(f)), kind(rs)) * H%ns
          cnt(L%own(f)) = cnt(L%own(f)) + 1
        enddo
        deallocate(cnt)
        L%nloc = int(count(L%own == H%me), kind(rs)) * H%ns
      end associate
      H%nlev = H%nlev + 1
    enddo
    if (H%nlev < 2) call fatal("the grid has no coarse level along the ring (n_tht must be even, n_tht/2 >= min_nj)")
    if (H%me == 0) then                                   ! the first rank of the operator's communicator
      write(*,'(A,A,I0,A,I0,A,I0,A)', advance="no") "[Physics PC]   "//trim(H%tag)//": ", "", H%nlev, &
        " levels along the ring, ", nfree, " of ", n_idx, " indices free; lattice "
      do g = 0, H%nlev - 1
        write(*,'(I0,A,I0,A)', advance="no") n_flux, "x", H%lv(g)%nj, merge(" -> ", "    ", g < H%nlev - 1)
      enddo
      write(*,*)
    endif
  end subroutine build_levels

  !> P(g), level g -> level g-1, created (MPIAIJ, the preallocation of the
  !! transfer); the linear and constant transfers filled here, once.
  subroutine create_prolongations()
    use mod_petsc_pc_harm, only: pc_global_slot
    integer :: g, cap
    PetscInt :: nl_r, nl_c
    PetscInt, allocatable :: dnz(:), onz(:)
    PetscInt :: zero
    PetscErrorCode :: ierr
    zero = 0
    ! entries of a row: 2 (linear), the coarse nodes within the reach times kinds and slots (ring-harmonic)
    ! the slots whose odd ring positions take the ring-harmonic rows: all (ringharm), the
    ! harmonics' (mixed: n = 0 keeps the linear transfer, whose space holds its kernel)
    allocate(H%rh_slot(0:H%ns - 1))
    do g = 0, H%ns - 1
      H%rh_slot(g) = H%o%transfer == TR_RINGHARM .or. (H%o%transfer == TR_MIXED .and. pc_global_slot(g) /= 0)
    enddo
    cap = 2
    if (any(H%rh_slot)) cap = max(2, 2 * ((H%o%rh_reach + 1) / 2) * 4 * H%ns)
    do g = 1, H%nlev - 1
      nl_r = H%lv(g - 1)%nloc; nl_c = H%lv(g)%nloc
      allocate(dnz(nl_r), onz(nl_r))
      dnz = min(int(cap, 8), int(nl_c, 8)); onz = min(int(cap, 8), int(H%lv(g)%n, 8) * H%ns - int(nl_c, 8))
      ! MPIAIJ also on one rank, as the operators: their products need it
      call MatCreate(H%comm, H%P(g), ierr)
      call MatSetSizes(H%P(g), nl_r, nl_c, PETSC_DETERMINE, PETSC_DETERMINE, ierr)
      call MatSetType(H%P(g), MATMPIAIJ, ierr)
      call MatMPIAIJSetPreallocation(H%P(g), zero, dnz, zero, onz, ierr)
      deallocate(dnz, onz)
      call MatSetOption(H%P(g), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
      if (.not. any(H%rh_slot)) call fill_p(g)
    enddo
  end subroutine create_prolongations

  !> The transfers that come from the operators (ring-harmonic) and the
  !! Galerkin operators Ag(g) = P(g)' A(g-1) P(g), at every build. first: the
  !! products' symbolic phases (the patterns stay those of the first build).
  subroutine galerkin_chain(A0, first)
    Mat, intent(in) :: A0
    logical, intent(in) :: first
    Mat :: af
    integer :: g
    PetscErrorCode :: ierr
    af = A0
    do g = 1, H%nlev - 1
      call PetscLogEventBegin(sev_transfer, ierr)
      if (any(H%rh_slot)) call fill_p(g, af)
      call PetscLogEventEnd(sev_transfer, ierr)
      call PetscLogEventBegin(sev_ptap, ierr)
      if (first) then
        call MatPtAP(af, H%P(g), MAT_INITIAL_MATRIX, 2.0d0, H%Ag(g), ierr)
      else
        call MatPtAP(af, H%P(g), MAT_REUSE_MATRIX, 2.0d0, H%Ag(g), ierr)
      endif
      call PetscLogEventEnd(sev_ptap, ierr)
      af = H%Ag(g)
    enddo
  end subroutine galerkin_chain

  !> The rows of P(g) this rank owns (module header): the shared axis value
  !! and every DOF of a C node (even J) or of the axis ring by the linear (or
  !! constant) rule of the kinds; with the ring-harmonic transfer the DOFs of
  !! the F nodes (I >= 1, odd J) as X_f Pc from the operator af of level g-1
  !! (ringharm_rows).
  subroutine fill_p(g, af)
    integer, intent(in) :: g
    Mat, intent(in), optional :: af
    integer :: f, s, i, j, k, njc, nc, q, cc(2)
    real*8 :: w(2), tot
    logical :: rhrow
    PetscInt :: row, cols(2), one, ncol
    PetscScalar :: vals(2)
    PetscErrorCode :: ierr
    one = 1
    associate (LF => H%lv(g - 1), LC => H%lv(g))
      njc = LC%nj
      do f = 0, LF%n - 1
        if (LF%own(f) /= H%me) cycle
        if (.not. LF%ok(f)) cycle
        i = LF%dI(f); j = LF%dJ(f); k = LF%dK(f)
        rhrow = i >= 1 .and. mod(j, 2) == 1                 ! an F node: ringharm_rows on the rh_slot slots
        nc = 0
        if (i == 0 .and. k == 0) then
          call add(LC%d(0, 0, 0), 1.d0)
        else if (mod(j, 2) == 0) then
          call add(LC%d(i, j / 2, k), 1.d0)
        else if (H%o%transfer == TR_CONST) then
          call add(LC%d(i, (j - 1) / 2, k), 1.d0)
        else
          call add(LC%d(i, (j - 1) / 2, k), 0.5d0)
          call add(LC%d(i, mod((j + 1) / 2, njc), k), 0.5d0)
        endif
        if (nc == 0) cycle
        tot = sum(w(1:nc))
        do s = 0, H%ns - 1
          if (rhrow .and. H%rh_slot(s)) cycle
          row = LF%row0(f) + s
          ncol = int(nc, kind(row))
          do q = 1, nc
            cols(q) = LC%row0(cc(q)) + s
            vals(q) = LF%osgn(f) * w(q) / tot
          enddo
          call MatSetValues(H%P(g), one, [row], ncol, cols(1:nc), vals(1:nc), INSERT_VALUES, ierr)
        enddo
      enddo
    end associate
    if (present(af)) call ringharm_rows(g, af)
    call MatAssemblyBegin(H%P(g), MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(H%P(g), MAT_FINAL_ASSEMBLY, ierr)
  contains
    subroutine add(cdof, wt)
      integer, intent(in) :: cdof
      real*8, intent(in)  :: wt
      if (cdof < 0) return
      nc = nc + 1; cc(nc) = cdof; w(nc) = wt
    end subroutine add
  end subroutine fill_p

  !> The ring-harmonic rows of P(g) (study section 7, spdmg/ringharm_sparse.py):
  !! on a ring I >= 1 of level g-1 the even positions J are the C nodes, the
  !! odd ones the F nodes. A low mode z has z_F = -(A_FF^ring)^-1 A_FC^ring z_C
  !! with the couplings inside the ring; X ~ -(A_FF^ring)^-1 A_FC^ring from
  !! rh_sweeps steps X <- mask(X - omega D^-1 (A_FF^ring X + A_FC^ring)) from
  !! X = 0, D the node blocks of A_FF^ring (all kinds and slots of a node), the
  !! mask the C nodes within rh_reach positions along the ring. An F node's
  !! rows of P are X_f Pc, Pc the C nodes' rows (the kind to the coarse DOF of
  !! the same kind at J/2, times the orientation). Every rank computes the
  !! whole rings that hold an F row of its own (the rows gathered from af).
  subroutine ringharm_rows(g, af)
    use phys_module, only: n_flux, physics_pc_sf_harm_couple
    use mod_petsc_pc_harm, only: pc_global_slot
    integer, intent(in) :: g
    Mat, intent(in) :: af
    external :: dgetrf, dgetri
    integer :: nj, ni, m, nF, i, j, k, s, f, ff, q, qq, c, jc, dd, t, info, nnbmax, ncmax, a, ll, kc, sc, nent
    integer :: jdist, nb_
    integer(8) :: nmm
    logical :: fam_only
    logical, allocatable :: myring(:)
    integer, allocatable :: colJ(:), colL(:), nnb(:), nbF(:,:), ncq(:), nearJ(:,:), piv(:)
    real*8, allocatable :: Dinv(:,:,:), Ab(:,:,:,:), Afc(:,:,:,:), X(:,:,:,:), R(:,:,:,:), wk(:)
    PetscInt, allocatable :: rlist(:), clist(:), ecol(:)
    PetscScalar, allocatable :: evals(:)
    PetscInt :: ncols, rpos, one, nr_, nc_, row
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    one = 1
    fam_only = H%o%rh_family .and. .not. physics_pc_sf_harm_couple
    associate (LF => H%lv(g - 1), LC => H%lv(g), HR => H%rh(g))
      nj = LF%nj; ni = n_flux; m = 4 * H%ns; nF = nj / 2
      ! the rings with an F row of mine, and their gathered rows (F nodes) and columns (all nodes)
      if (.not. HR%ready) then
        allocate(myring(0:ni - 1)); myring = .false.
        do i = 1, ni - 1
          do j = 1, nj - 1, 2
            do k = 0, 3
              dd = LF%d(i, j, k)
              if (dd < 0) cycle
              if (LF%own(dd) == H%me) myring(i) = .true.
            enddo
          enddo
        enddo
        nr_ = 0; nc_ = 0
        do t = 1, 2                                        ! count, fill
          if (t == 2) allocate(rlist(nr_), clist(nc_))
          nr_ = 0; nc_ = 0
          do i = 1, ni - 1
            if (.not. myring(i)) cycle
            do j = 0, nj - 1
              do k = 0, 3
                dd = LF%d(i, j, k)
                if (dd < 0) cycle
                do s = 0, H%ns - 1
                  nc_ = nc_ + 1
                  if (t == 2) clist(nc_) = LF%row0(dd) + s
                  if (mod(j, 2) == 1) then
                    nr_ = nr_ + 1
                    if (t == 2) rlist(nr_) = LF%row0(dd) + s
                  endif
                enddo
              enddo
            enddo
          enddo
        enddo
        call sort_pint(rlist); call sort_pint(clist)
        HR%rrow = rlist; HR%rcol = clist
        nr_ = int(size(rlist), kind(one)); nc_ = int(size(clist), kind(one))
        call ISCreateGeneral(PETSC_COMM_SELF, nr_, rlist, PETSC_COPY_VALUES, HR%isr, ierr)
        call ISCreateGeneral(PETSC_COMM_SELF, nc_, clist, PETSC_COPY_VALUES, HR%isc, ierr)
        call MatCreateSubMatrices(af, one, [HR%isr], [HR%isc], MAT_INITIAL_MATRIX, HR%sub, ierr)
        HR%ready = .true.
      else
        call MatCreateSubMatrices(af, one, [HR%isr], [HR%isc], MAT_REUSE_MATRIX, HR%sub, ierr)
      endif

      allocate(colJ(0:max(size(HR%rcol), 1) - 1), colL(0:max(size(HR%rcol), 1) - 1))
      colJ = -1; colL = 0
      ! C nodes (even positions) within the reach of an odd position: at most 2 ceil(reach / 2)
      ncmax = max(1, min(nF, 2 * ((H%o%rh_reach + 1) / 2)))
      allocate(ncq(nF), nearJ(ncmax, nF), Dinv(m, m, nF), Afc(m, m, ncmax, nF), X(m, m, ncmax, nF), &
               R(m, m, ncmax, nF), piv(m), wk(m * m), nnb(nF))
      do i = 1, ni - 1
        if (.not. ring_gathered(i)) cycle
        ! this ring's columns: position -> (J, local index k * ns + s + 1)
        do j = 0, nj - 1
          do k = 0, 3
            dd = LF%d(i, j, k)
            if (dd < 0) cycle
            do s = 0, H%ns - 1
              c = bsearch(HR%rcol, LF%row0(dd) + s)
              colJ(c) = j; colL(c) = k * H%ns + s + 1
            enddo
          enddo
        enddo
        ! the near C nodes of every F node
        do ff = 1, nF
          j = 2 * ff - 1
          ncq(ff) = 0
          do jc = 0, nj - 2, 2
            if (cdist(j, jc) > H%o%rh_reach) cycle
            ncq(ff) = ncq(ff) + 1; nearJ(ncq(ff), ff) = jc
          enddo
        enddo
        ! the blocks: D, the neighbour F nodes, the near C nodes. Ab holds
        ! nnbmax neighbour blocks per F node, the largest count of distinct
        ! neighbour F nodes on the ring (a pre-pass over the rows): a few
        ! along a sparse ring, nF - 1 only where the ring is coupled densely
        ! (the axis core of vs_lumped's P), rather than nF everywhere (memory
        ! and zeroing quadratic in the ring length)
        nnbmax = max(1, ring_nnb_max(i))
        H%rh_ab(1, g) = H%rh_ab(1, g) + 8.d0 * dble(m)**2 * dble(nnbmax) * dble(nF)
        H%rh_ab(2, g) = H%rh_ab(2, g) + 8.d0 * dble(m)**2 * dble(nF) * dble(nF)
        allocate(nbF(nnbmax, nF), Ab(m, m, nnbmax, nF))
        nnb = 0; Ab = 0.d0; Afc = 0.d0
        Dinv = 0.d0
        do ff = 1, nF
          j = 2 * ff - 1
          do k = 0, 3
            dd = LF%d(i, j, k)
            do s = 0, H%ns - 1
              ll = k * H%ns + s + 1
              if (dd < 0) then
                Dinv(ll, ll, ff) = 1.d0                    ! no DOF: an identity row, decoupled
                cycle
              endif
              rpos = int(bsearch(HR%rrow, LF%row0(dd) + s), kind(one))
              call MatGetRow(HR%sub(1), rpos, ncols, cols, vals, ierr)
              do a = 1, int(ncols)
                c = int(cols(a))
                if (colJ(c) < 0) cycle                     ! another ring
                if (colJ(c) == j) then
                  Dinv(ll, colL(c), ff) = vals(a)
                else if (mod(colJ(c), 2) == 1) then        ! an F node of the ring
                  nb_ = 0
                  do qq = 1, nnb(ff)
                    if (nbF(qq, ff) == (colJ(c) + 1) / 2) nb_ = qq
                  enddo
                  if (nb_ == 0) then
                    if (nnb(ff) >= nnbmax) call fatal("ring-harmonic transfer: more neighbour F nodes than counted")
                    nnb(ff) = nnb(ff) + 1; nb_ = nnb(ff); nbF(nb_, ff) = (colJ(c) + 1) / 2
                  endif
                  Ab(ll, colL(c), nb_, ff) = vals(a)
                else                                       ! a C node: within the reach
                  do qq = 1, ncq(ff)
                    if (nearJ(qq, ff) == colJ(c)) Afc(ll, colL(c), qq, ff) = vals(a)
                  enddo
                endif
              enddo
              call MatRestoreRow(HR%sub(1), rpos, ncols, cols, vals, ierr)
            enddo
          enddo
          ! D^-1
          call dgetrf(m, m, Dinv(:, :, ff), m, piv, info)
          if (info /= 0) call fatal("a singular node block in the ring-harmonic transfer")
          call dgetri(m, Dinv(:, :, ff), m, piv, wk, m * m, info)
        enddo
        call PetscLogFlops(2.d0 * dble(m)**3 * dble(nF), ierr)
        ! the sweeps (Jacobi: all R from the last X)
        X = 0.d0
        nmm = 0
        do t = 1, H%o%rh_sweeps
          do ff = 1, nF
            do q = 1, ncq(ff)
              R(:, :, q, ff) = Afc(:, :, q, ff)
              do qq = 1, nnb(ff)
                f = nbF(qq, ff)
                do c = 1, ncq(f)
                  if (nearJ(c, f) == nearJ(q, ff)) then
                    R(:, :, q, ff) = R(:, :, q, ff) + matmul(Ab(:, :, qq, ff), X(:, :, c, f))
                    nmm = nmm + 1
                  endif
                enddo
              enddo
            enddo
          enddo
          ! X - omega D^-1 (A_FF X + A_FC) with A_FF = D + the neighbours (R holds the latter and A_FC)
          do ff = 1, nF
            do q = 1, ncq(ff)
              X(:, :, q, ff) = (1.d0 - H%o%rh_omega) * X(:, :, q, ff) - H%o%rh_omega * matmul(Dinv(:, :, ff), &
                                                                                          R(:, :, q, ff))
              nmm = nmm + 1
            enddo
          enddo
        enddo
        call PetscLogFlops(2.d0 * dble(m)**3 * dble(nmm), ierr)
        ! the rows of P of my F DOFs: X_f Pc
        allocate(ecol(ncmax * m), evals(ncmax * m))
        do ff = 1, nF
          j = 2 * ff - 1
          do k = 0, 3
            dd = LF%d(i, j, k)
            if (dd < 0) cycle
            if (LF%own(dd) /= H%me .or. .not. LF%ok(dd)) cycle
            do s = 0, H%ns - 1
              if (.not. H%rh_slot(s)) cycle                ! mixed: n = 0 keeps the linear row (fill_p)
              ll = k * H%ns + s + 1
              row = LF%row0(dd) + s
              nent = 0
              do q = 1, ncq(ff)
                jc = nearJ(q, ff)
                do kc = 0, 3
                  c = LF%d(i, jc, kc)
                  if (c < 0) cycle
                  if (.not. LF%ok(c)) cycle
                  do sc = 0, H%ns - 1
                    ! -vs_pmg_rh_family: without the couplings between |n| families
                    ! (physics_pc_sf_harm_couple = .f.) the operator, its node blocks and
                    ! so X have none: their exact zeros stay out of P(g) and of the
                    ! Galerkin levels' patterns
                    if (fam_only) then
                      if ((pc_global_slot(sc) + 1) / 2 /= (pc_global_slot(s) + 1) / 2) cycle
                    endif
                    nent = nent + 1
                    ecol(nent) = LC%row0(LC%d(i, jc / 2, kc)) + sc
                    evals(nent) = X(ll, kc * H%ns + sc + 1, q, ff) * LF%osgn(c)
                  enddo
                enddo
              enddo
              if (nent > 0) call MatSetValues(H%P(g), one, [row], int(nent, kind(one)), ecol(1:nent), &
                                              evals(1:nent), INSERT_VALUES, ierr)
            enddo
          enddo
        enddo
        deallocate(ecol, evals, nbF, Ab)
        ! reset this ring's column map
        do j = 0, nj - 1
          do k = 0, 3
            dd = LF%d(i, j, k)
            if (dd < 0) cycle
            do s = 0, H%ns - 1
              c = bsearch(HR%rcol, LF%row0(dd) + s)
              colJ(c) = -1
            enddo
          enddo
        enddo
      enddo
    end associate
  contains
    !> the largest number of distinct F nodes (odd positions other than its
    !! own) that an F row of ring ii couples to, from the gathered rows (the
    !! ring's column map colJ set)
    integer function ring_nnb_max(ii)
      integer, intent(in) :: ii
      integer :: ff_, k_, s_, a_, dd_, j_, cnt_, rp_, c_
      integer, allocatable :: seen_(:)
      PetscInt :: nc_r
      PetscInt, pointer :: cl_(:)
      PetscScalar, pointer :: vl_(:)
      PetscErrorCode :: ie_
      allocate(seen_(0:nj - 1)); seen_ = 0
      ring_nnb_max = 0
      do ff_ = 1, nF
        j_ = 2 * ff_ - 1
        cnt_ = 0
        do k_ = 0, 3
          dd_ = H%lv(g - 1)%d(ii, j_, k_)
          if (dd_ < 0) cycle
          do s_ = 0, H%ns - 1
            rp_ = bsearch(H%rh(g)%rrow, H%lv(g - 1)%row0(dd_) + s_)
            call MatGetRow(H%rh(g)%sub(1), int(rp_, kind(nc_r)), nc_r, cl_, vl_, ie_)
            do a_ = 1, int(nc_r)
              c_ = int(cl_(a_))
              if (colJ(c_) < 0 .or. colJ(c_) == j_ .or. mod(colJ(c_), 2) /= 1) cycle
              if (seen_(colJ(c_)) == ff_) cycle
              seen_(colJ(c_)) = ff_; cnt_ = cnt_ + 1
            enddo
            call MatRestoreRow(H%rh(g)%sub(1), int(rp_, kind(nc_r)), nc_r, cl_, vl_, ie_)
          enddo
        enddo
        ring_nnb_max = max(ring_nnb_max, cnt_)
      enddo
      deallocate(seen_)
    end function ring_nnb_max
    !> distance of the ring positions a and b (periodic, nj positions)
    integer function cdist(a_, b_)
      integer, intent(in) :: a_, b_
      cdist = modulo(a_ - b_, nj)
      cdist = min(cdist, nj - cdist)
    end function cdist
    !> is ring ii among the gathered rings (its first F row in rrow)?
    logical function ring_gathered(ii)
      integer, intent(in) :: ii
      integer :: kk
      ring_gathered = .false.
      do kk = 0, 3
        if (H%lv(g - 1)%d(ii, 1, kk) < 0) cycle
        ring_gathered = bsearch(H%rh(g)%rrow, H%lv(g - 1)%row0(H%lv(g - 1)%d(ii, 1, kk))) >= 0
        return
      enddo
    end function ring_gathered
  end subroutine ringharm_rows

  !> PCMG on pc: the levels, P, the level operators (A0 and the Galerkin
  !! chain), the Chebyshev smoothers with the additive Schwarz shell PCs,
  !! MUMPS on the coarsest level. PETSc numbers the levels from the coarsest
  !! (0); level g here is PETSc's nlev - 1 - g.
  subroutine configure_pcmg(pc, A0)
    PC, intent(inout) :: pc
    Mat, intent(in) :: A0
    KSP :: ks, kest, kc
    PC :: pcs, pcc
    PetscErrorCode :: ierr
    PetscInt :: L
    integer :: g
    character(len=48) :: nm
    call PCSetType(pc, PCMG, ierr)
    call PCMGSetLevels(pc, int(H%nlev, kind(L)), PETSC_NULL_MPI_COMM, ierr)
    call PCMGSetType(pc, PC_MG_MULTIPLICATIVE, ierr)
    call PCMGSetCycleType(pc, PC_MG_CYCLE_V, ierr)
    ! the Galerkin operators are galerkin_chain's (the ring-harmonic transfer needs them level by level)
    call PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE, ierr)
    do g = 1, H%nlev - 1
      L = H%nlev - g
      call PCMGSetInterpolation(pc, L, H%P(g), ierr)
    enddo
    do g = 0, H%nlev - 2
      L = H%nlev - 1 - g
      call PCMGGetSmoother(pc, L, ks, ierr)
      if (g == 0) then
        call KSPSetOperators(ks, A0, A0, ierr)
      else
        call KSPSetOperators(ks, H%Ag(g), H%Ag(g), ierr)
      endif
      call KSPSetType(ks, KSPCHEBYSHEV, ierr)
      call KSPChebyshevEstEigSet(ks, 0.d0, H%o%safety / H%o%ratio, 0.d0, H%o%safety, ierr)
      call KSPChebyshevEstEigGetKSP(ks, kest, ierr)
      call KSPSetType(kest, KSPCG, ierr)
      call KSPSetTolerances(kest, 1.d-50, 1.d-50, 1.d10, int(H%o%esteig_its, kind(L)), ierr)
      call KSPSetTolerances(ks, 1.d-50, 1.d-50, 1.d10, int(H%o%degree, kind(L)), ierr)
      call KSPGetPC(ks, pcs, ierr)
      call PCSetType(pcs, PCSHELL, ierr)
      call PCShellSetApply(pcs, as_apply, ierr)
      call PCShellSetSetUp(pcs, as_setup, ierr)
      write(nm, '(A,A,I0)') trim(H%tag), " additive Schwarz, level ", g
      call PCShellSetName(pcs, trim(nm), ierr)
      H%as_pc(g) = pcs
    enddo
    call PCMGGetCoarseSolve(pc, kc, ierr)
    call KSPSetOperators(kc, H%Ag(H%nlev - 1), H%Ag(H%nlev - 1), ierr)
    call KSPSetType(kc, KSPPREONLY, ierr)
    call KSPGetPC(kc, pcc, ierr)
    call PCSetType(pcc, PCLU, ierr)
    call PCFactorSetMatSolverType(pcc, MATSOLVERMUMPS, ierr)
  end subroutine configure_pcmg

  !> The level of the shell PC pc (its handle among the instances' as_pc);
  !! makes its instance the active one.
  integer function level_of(pc)
    PC, intent(in) :: pc
    integer :: g, k
    level_of = -1
    do k = 1, NINST
      do g = 0, inst(k)%nlev - 2
        if (inst(k)%as_pc(g)%v == pc%v) then
          H => inst(k)
          level_of = g; return
        endif
      enddo
    enddo
    call fatal("a shell PC that is not one of the levels' smoothers")
  end function level_of

  !> The rows of level g's blocks that this rank solves, block by block, and
  !! their weights in the additive sum. All ranks list all blocks of the
  !! layout (o%blocks) as DOF sets, which may overlap: the axis block (rings
  !! 0..axis_rings), then the nodes, points or radial lines, then the wedges
  !! (linewedge). A block's rows are all slots of its DOFs, ascending; it
  !! belongs to the owner of its smallest DOF, or (-vs_pmg_block_owner
  !! balanced) a block across ranks to the least loaded of its ranks. A new
  !! layout is a new case here.
  subroutine block_members(g, nb, off, gidx, bw)
    integer, intent(in) :: g
    integer, intent(out) :: nb
    integer, allocatable, intent(out) :: off(:)
    PetscInt, allocatable, intent(out) :: gidx(:)
    real*8, allocatable, intent(out) :: bw(:)
    integer, allocatable :: ab0(:), abd(:), sel(:), key0(:), keyd(:), bown(:)
    real*8, allocatable :: aw(:)
    logical, allocatable :: jin(:)
    integer :: nab, nabd, f, j, j0, k, ax, lf, t, b, s, nsel, p, nmine, nw
    ax = H%o%axis_rings
    lf = H%o%line_from
    if (lf < 0) lf = ax + 1
    associate (L => H%lv(g))
      allocate(ab0(1), abd(max(2 * L%n, 16)), aw(0), sel(L%n))
      ab0(1) = 0; nab = 0; nabd = 0
      ! the axis block
      nsel = 0
      do f = 0, L%n - 1
        if (L%dI(f) <= ax) call put(f)
      enddo
      call push(1.d0)
      select case (H%o%blocks)
      case (BL_NODE)                                        ! the DOFs of a node outside the axis block
        call buckets(.true.)
        do k = 0, size(key0) - 2
          nsel = 0
          do p = key0(k) + 1, key0(k + 1)
            if (L%dI(keyd(p)) > ax) call put(keyd(p))
          enddo
          call push(1.d0)
        enddo
      case (BL_POINT)
        do f = 0, L%n - 1
          if (L%dI(f) <= ax) cycle
          nsel = 0; call put(f); call push(1.d0)
        enddo
      case (BL_LINE, BL_LINEWEDGE)                          ! the rings lf.. at one position J
        call buckets(.false.)
        do j = 0, L%nj - 1
          nsel = 0
          do p = key0(j) + 1, key0(j + 1)
            if (L%dI(keyd(p)) >= lf) call put(keyd(p))
          enddo
          call push(1.d0)
        enddo
      end select
      if (H%o%blocks == BL_LINEWEDGE) then                  ! wedge_lines lines every wedge_stride, rings lf..t
        t = wedge_last(g, lf)
        nw = 0
        if (t >= lf) then
          allocate(jin(0:L%nj - 1))
          do j0 = 0, L%nj - 1, H%o%wedge_stride
            jin = .false.
            do k = 0, H%o%wedge_lines - 1
              jin(modulo(j0 + k, L%nj)) = .true.
            enddo
            nsel = 0
            do f = 0, L%n - 1
              if (L%dI(f) >= lf .and. L%dI(f) <= t .and. jin(L%dJ(f))) call put(f)
            enddo
            if (nsel > 0) nw = nw + 1
            call push(H%o%wedge_weight)
          enddo
        endif
        if (H%me == 0 .and. .not. H%as(g)%ready) then
          if (nw > 0) then
            write(*,'(A,I0,A,I0,A,I0,A,I0,A,F5.2,A)') "[Physics PC]   "//trim(H%tag)//" level ", g, ": ", nw, &
              " wedges on the rings ", lf, "..", t, " (weight ", H%o%wedge_weight, ")"
          else
            write(*,'(A,I0,A)') "[Physics PC]   "//trim(H%tag)//" level ", g, &
              ": no wedge (no ring below the cell aspect)"
          endif
        endif
      endif

      ! the owner of every block: the owner of its smallest DOF (-vs_pmg_block_owner
      ! first), or with -vs_pmg_block_owner balanced, for a block whose DOFs
      ! more than one rank owns, the least loaded of these ranks (load: the
      ! sum of rows^2 of its blocks, the dense solve's cost), the largest such
      ! blocks first; the same on every rank. A radial line spans every rank
      ! of an onion (ring-major) layout and would otherwise go to the owner of
      ! its innermost ring, all lines of a level to one rank.
      allocate(bown(nab))
      do b = 1, nab
        bown(b) = L%own(minval(abd(ab0(b) + 1 : ab0(b + 1))))
      enddo
      if (H%o%block_owner == OWN_BALANCED .and. H%np > 1) then
        block
          real*8, allocatable :: load(:), cost(:)
          integer, allocatable :: ord(:)
          logical, allocatable :: multi(:)
          integer :: q, r, best, nmulti, tmp_
          allocate(load(0:H%np - 1), cost(nab), multi(nab))
          load = 0.d0
          do b = 1, nab
            cost(b) = (dble(ab0(b + 1) - ab0(b)) * dble(H%ns))**2
            multi(b) = any(L%own(abd(ab0(b) + 1 : ab0(b + 1))) /= bown(b))
            if (.not. multi(b)) load(bown(b)) = load(bown(b)) + cost(b)
          enddo
          ord = pack([(b, b = 1, nab)], multi)
          nmulti = size(ord)
          do q = 2, nmulti                                  ! stable insertion sort, cost descending
            tmp_ = ord(q); r = q - 1
            do while (r >= 1)
              if (cost(ord(r)) >= cost(tmp_)) exit
              ord(r + 1) = ord(r); r = r - 1
            enddo
            ord(r + 1) = tmp_
          enddo
          do q = 1, nmulti
            b = ord(q)
            best = -1
            do p = ab0(b) + 1, ab0(b + 1)
              r = L%own(abd(p))
              if (best < 0) then
                best = r
              else if (load(r) < load(best) .or. (load(r) == load(best) .and. r < best)) then
                best = r
              endif
            enddo
            bown(b) = best
            load(best) = load(best) + cost(b)
          enddo
          if (H%me == 0 .and. .not. H%as(g)%ready) write(*,'(A,I0,A,I0,A,I0,A,F7.2)') "[Physics PC]   "// &
            trim(H%tag)//" level ", g, ": ", nmulti, " of ", nab, " blocks across ranks placed by load; "// &
            "largest rank load / mean ", maxval(load) / max(sum(load) / H%np, tiny(1.d0))
        end block
      endif

      ! mine: rows (all slots) in w order, sorted inside each block
      nmine = 0
      do b = 1, nab
        if (bown(b) == H%me) nmine = nmine + 1
      enddo
      nb = nmine
      allocate(off(0:nb), bw(0:max(nb - 1, 0)))
      off(0) = 0; k = 0
      do b = 1, nab
        if (bown(b) /= H%me) cycle
        k = k + 1
        off(k) = off(k - 1) + (ab0(b + 1) - ab0(b)) * H%ns
        bw(k - 1) = aw(b)
      enddo
      allocate(gidx(max(off(nb), 1)))
      k = 0
      do b = 1, nab
        if (bown(b) /= H%me) cycle
        p = off(k)
        do j = ab0(b) + 1, ab0(b + 1)
          do s = 0, H%ns - 1
            p = p + 1
            gidx(p) = L%row0(abd(j)) + s
          enddo
        enddo
        if (off(k + 1) - off(k) > 1) call sort_pint(gidx(off(k) + 1 : off(k + 1)))
        k = k + 1
      enddo
    end associate
  contains
    subroutine put(ff)
      integer, intent(in) :: ff
      nsel = nsel + 1; sel(nsel) = ff
    end subroutine put
    !> the selection as a block of weight wt (none if empty)
    subroutine push(wt)
      real*8, intent(in) :: wt
      integer, allocatable :: tmp(:)
      if (nsel == 0) return
      if (nabd + nsel > size(abd)) then
        allocate(tmp(max(2 * size(abd), nabd + nsel)))
        tmp(1:nabd) = abd(1:nabd)
        call move_alloc(tmp, abd)
      endif
      abd(nabd + 1 : nabd + nsel) = sel(1:nsel)
      nabd = nabd + nsel
      nab = nab + 1
      ab0 = [ab0, nabd]
      aw = [aw, wt]
    end subroutine push
    !> the DOFs grouped by node (I nj + J) or by ring position J (CSR key0, keyd)
    subroutine buckets(by_node)
      logical, intent(in) :: by_node
      integer :: nk, ff, kk
      integer, allocatable :: c(:)
      associate (L => H%lv(g))
        nk = merge(size(L%d, 1) * L%nj, L%nj, by_node)
        allocate(key0(0:nk), c(0:nk - 1), keyd(L%n))
        c = 0
        do ff = 0, L%n - 1
          kk = merge(L%dI(ff) * L%nj + L%dJ(ff), L%dJ(ff), by_node)
          c(kk) = c(kk) + 1
        enddo
        key0(0) = 0
        do kk = 0, nk - 1
          key0(kk + 1) = key0(kk) + c(kk)
        enddo
        c = 0
        do ff = 0, L%n - 1
          kk = merge(L%dI(ff) * L%nj + L%dJ(ff), L%dJ(ff), by_node)
          c(kk) = c(kk) + 1
          keyd(key0(kk) + c(kk)) = ff
        enddo
      end associate
    end subroutine buckets
  end subroutine block_members

  !> The last ring of level g's wedges: the largest ring I >= max(lf, 1) whose
  !! poloidal / radial cell aspect on level g is below wedge_aspect (-1: none).
  !! The aspect of ring I is the median over the ring of the cells between the
  !! rings I and I + 1 on the finest level (the last ring: the one before),
  !! times n_tht / nj(g): the poloidal edge doubles with every level
  !! (smoothers/smlib.py ring_aspect).
  integer function wedge_last(g, lf)
    use phys_module, only: n_tht
    integer, intent(in) :: g, lf
    integer :: i
    real*8 :: a
    wedge_last = -1
    do i = max(lf, 1), size(H%ring_asp) - 1
      a = H%ring_asp(i) * dble(n_tht) / dble(H%lv(g)%nj)
      if (a < H%o%wedge_aspect) wedge_last = i
    enddo
  end function wedge_last

  !> The shell PC's setup: level g's blocks (first call) and their factors (every call).
  subroutine as_setup(pc, ierr)
    PC :: pc
    PetscErrorCode, intent(out) :: ierr
    Mat :: amat, pmat
    integer :: g
    ierr = 0
    g = level_of(pc)
    call PCGetOperators(pc, amat, pmat, ierr)
    call asm_build(H%as(g), g, pmat)
    ierr = 0
  end subroutine as_setup

  !> y = M^-1 x on level g (module header).
  subroutine as_apply(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    integer :: g
    ierr = 0
    g = level_of(pc)
    call asm_apply(H%as(g), x, y)
    ierr = 0
  end subroutine as_apply

  !> Blocks, scatter and factors of level g's additive Schwarz PC for A.
  subroutine asm_build(S, g, Aop)
    type(asm_t), intent(inout) :: S
    integer, intent(in) :: g
    Mat, intent(in) :: Aop
    external :: dgetrf, dgbtrf
    PetscErrorCode :: ierr
    PetscInt :: ncols, nwp, one
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: uniq(:), isb(:)
    integer, allocatable :: bpos(:)
    integer :: b, n, a, c, jj, info, k, nsp
    integer(8) :: tot
    real*8, allocatable :: m(:, :)
    IS :: isf
    Vec :: xg
    one = 1

    call PetscLogEventBegin(sev_asmsetup, ierr)
    if (.not. S%ready) then
      call block_members(g, S%nb, S%off, S%gidx, S%bw)
      S%nw = S%off(S%nb)
      allocate(S%sz(0:S%nb - 1))
      do b = 0, S%nb - 1
        S%sz(b) = S%off(b + 1) - S%off(b)
      enddo
      ! the union of the rows, ascending, and every w position's place in it
      uniq = S%gidx(1:S%nw)
      call sort_pint(uniq)
      n = 0
      do a = 1, S%nw
        if (n > 0) then
          if (uniq(a) == uniq(n)) cycle
        endif
        n = n + 1; uniq(n) = uniq(a)
      enddo
      S%nu = n
      allocate(S%upos(S%nw))
      do a = 1, S%nw
        S%upos(a) = bsearch(uniq(1:n), S%gidx(a))
      enddo
      call ISCreateGeneral(PETSC_COMM_SELF, int(S%nu, kind(one)), uniq(1:n), PETSC_COPY_VALUES, S%isu, ierr)
      ! the gather of w from the level vector, and back
      call MatCreateVecs(Aop, xg, PETSC_NULL_VEC, ierr)
      nwp = int(S%nw, kind(one))
      call VecCreateSeq(PETSC_COMM_SELF, nwp, S%w, ierr)
      call ISCreateGeneral(PETSC_COMM_SELF, nwp, S%gidx(1:max(S%nw, 1)), PETSC_COPY_VALUES, isf, ierr)
      call VecScatterCreate(xg, isf, S%w, PETSC_NULL_IS, S%sct, ierr)
      call ISDestroy(isf, ierr)
      call VecDestroy(xg, ierr)
      call MatCreateSubMatrices(Aop, one, [S%isu], [S%isu], MAT_INITIAL_MATRIX, S%sub, ierr)
      ! the factor storage: dense blocks in lu, with -vs_pmg_band the blocks
      ! whose band (in their ascending row order) is narrow (2 kl + ku + 1 <=
      ! n / 2) as LAPACK band LUs in lu, whatever dense_max, the others sparse
      allocate(S%dense(0:S%nb - 1), S%band(0:S%nb - 1), S%kl(0:S%nb - 1), S%ku(0:S%nb - 1), &
               S%loff(0:S%nb - 1), S%piv(max(S%nw, 1)))
      S%band = .false.; S%kl = 0; S%ku = 0
      if (H%o%band) call band_widths()
      tot = 0; nsp = 0
      do b = 0, S%nb - 1
        if (H%o%band) S%band(b) = 2 * S%kl(b) + S%ku(b) + 1 <= S%sz(b) / 2
        S%dense(b) = S%sz(b) <= H%o%dense_max .and. .not. S%band(b)
        S%loff(b) = tot
        if (S%dense(b)) then
          tot = tot + int(S%sz(b), 8)**2
        else if (S%band(b)) then
          tot = tot + int(2 * S%kl(b) + S%ku(b) + 1, 8) * S%sz(b)
        else
          nsp = nsp + 1
        endif
      enddo
      allocate(S%lu(max(tot, 1_8)))
      S%nsparse = nsp
      if (nsp > 0) allocate(S%kb(0:S%nb - 1), S%mb(0:S%nb - 1), S%ib(0:S%nb - 1), S%vb(0:S%nb - 1), &
                            S%vx(0:S%nb - 1))
    else
      call MatCreateSubMatrices(Aop, one, [S%isu], [S%isu], MAT_REUSE_MATRIX, S%sub, ierr)
    endif

    ! the blocks' matrices and factors
    allocate(bpos(0:S%nu - 1)); bpos = -1
    do b = 0, S%nb - 1
      n = S%sz(b)
      do a = 1, n
        bpos(S%upos(S%off(b) + a)) = a
      enddo
      if (S%dense(b)) then
        allocate(m(n, n)); m = 0.d0
        do a = 1, n
          call MatGetRow(S%sub(1), int(S%upos(S%off(b) + a), kind(one)), ncols, cols, vals, ierr)
          do c = 1, int(ncols)
            jj = bpos(int(cols(c)))
            if (jj > 0) m(a, jj) = vals(c)
          enddo
          call MatRestoreRow(S%sub(1), int(S%upos(S%off(b) + a), kind(one)), ncols, cols, vals, ierr)
        enddo
        call dgetrf(n, n, m, n, S%piv(S%off(b) + 1), info)
        if (info /= 0) call fatal("a singular smoother block")
        call PetscLogFlops(2.d0 / 3.d0 * dble(n)**3, ierr)
        S%lu(S%loff(b) + 1 : S%loff(b) + int(n, 8)**2) = reshape(m, [n * n])
        deallocate(m)
      else if (S%band(b)) then
        ! LAPACK general band storage: A(i, j) in ab(kl + ku + 1 + i - j, j), kl rows of fill above
        block
          integer :: ldab, kl_, ku_
          integer(8) :: o_
          kl_ = S%kl(b); ku_ = S%ku(b); ldab = 2 * kl_ + ku_ + 1
          o_ = S%loff(b)
          S%lu(o_ + 1 : o_ + int(ldab, 8) * n) = 0.d0
          do a = 1, n
            call MatGetRow(S%sub(1), int(S%upos(S%off(b) + a), kind(one)), ncols, cols, vals, ierr)
            do c = 1, int(ncols)
              jj = bpos(int(cols(c)))
              if (jj > 0) S%lu(o_ + int(jj - 1, 8) * ldab + kl_ + ku_ + 1 + a - jj) = vals(c)
            enddo
            call MatRestoreRow(S%sub(1), int(S%upos(S%off(b) + a), kind(one)), ncols, cols, vals, ierr)
          enddo
          call dgbtrf(n, n, kl_, ku_, S%lu(o_ + 1), ldab, S%piv(S%off(b) + 1), info)
          if (info /= 0) call fatal("a singular smoother block (band)")
          call PetscLogFlops(2.d0 * dble(n) * dble(kl_) * dble(kl_ + ku_ + 1), ierr)
        end block
      else
        if (.not. S%ready) then
          allocate(isb(n))
          do a = 1, n
            isb(a) = int(S%upos(S%off(b) + a), kind(one))
          enddo
          call ISCreateGeneral(PETSC_COMM_SELF, int(n, kind(one)), isb, PETSC_COPY_VALUES, S%ib(b), ierr)
          deallocate(isb)
          call MatCreateSubMatrix(S%sub(1), S%ib(b), S%ib(b), MAT_INITIAL_MATRIX, S%mb(b), ierr)
          call KSPCreate(PETSC_COMM_SELF, S%kb(b), ierr)
          call KSPSetType(S%kb(b), KSPPREONLY, ierr)
          block
            PC :: pcb
            call KSPGetPC(S%kb(b), pcb, ierr)
            call PCSetType(pcb, PCLU, ierr)
            call PCFactorSetMatOrderingType(pcb, trim(H%o%sp_order), ierr)
          end block
          call KSPSetOperators(S%kb(b), S%mb(b), S%mb(b), ierr)
          call MatCreateVecs(S%mb(b), S%vx(b), S%vb(b), ierr)
        else
          call MatCreateSubMatrix(S%sub(1), S%ib(b), S%ib(b), MAT_REUSE_MATRIX, S%mb(b), ierr)
        endif
        call KSPSetUp(S%kb(b), ierr)
      endif
      do a = 1, n
        bpos(S%upos(S%off(b) + a)) = -1
      enddo
    enddo
    deallocate(bpos)
    S%ready = .true.
    call PetscLogEventEnd(sev_asmsetup, ierr)
  contains
    !> kl, ku of every block in its ascending row order, from the pattern of
    !! sub (first build)
    subroutine band_widths()
      integer, allocatable :: bp(:)
      integer :: bb, nn, aa, cc, j_
      allocate(bp(0:S%nu - 1)); bp = -1
      do bb = 0, S%nb - 1
        nn = S%sz(bb)
        do aa = 1, nn
          bp(S%upos(S%off(bb) + aa)) = aa
        enddo
        do aa = 1, nn
          call MatGetRow(S%sub(1), int(S%upos(S%off(bb) + aa), kind(one)), ncols, cols, vals, ierr)
          do cc = 1, int(ncols)
            j_ = bp(int(cols(cc)))
            if (j_ <= 0) cycle
            S%kl(bb) = max(S%kl(bb), aa - j_)
            S%ku(bb) = max(S%ku(bb), j_ - aa)
          enddo
          call MatRestoreRow(S%sub(1), int(S%upos(S%off(bb) + aa), kind(one)), ncols, cols, vals, ierr)
        enddo
        do aa = 1, nn
          bp(S%upos(S%off(bb) + aa)) = -1
        enddo
      enddo
    end subroutine band_widths
  end subroutine asm_build

  !> y = sum_b R_b' A_bb^-1 R_b x. The LAPACK blocks (dense, band) solve on
  !! the rank's OpenMP threads, each in its own part of w (the same arithmetic
  !! per block as serially); the sparse ones (a PETSc KSP each) one after the
  !! other.
  subroutine asm_apply(S, x, y)
    type(asm_t), intent(inout) :: S
    Vec :: x, y
    external :: dgetrs, dgbtrs
    PetscErrorCode :: ierr
    PetscScalar, pointer :: wa(:), va(:)
    integer :: b, n, info
    real*8 :: fl
    call PetscLogEventBegin(sev_asmapply, ierr)
    call VecScatterBegin(S%sct, x, S%w, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(S%sct, x, S%w, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call PetscLogEventBegin(sev_asmblk, ierr)
    call VecGetArray(S%w, wa, ierr)
    fl = 0.d0
    !$omp parallel do schedule(dynamic, 1) private(n, info) reduction(+:fl) if (S%nb > 1)
    do b = 0, S%nb - 1
      n = S%sz(b)
      if (S%dense(b)) then
        call dgetrs("N", n, 1, S%lu(S%loff(b) + 1), n, S%piv(S%off(b) + 1), wa(S%off(b) + 1), n, info)
        fl = fl + 2.d0 * dble(n)**2
      else if (S%band(b)) then
        call dgbtrs("N", n, S%kl(b), S%ku(b), 1, S%lu(S%loff(b) + 1), 2 * S%kl(b) + S%ku(b) + 1, &
                    S%piv(S%off(b) + 1), wa(S%off(b) + 1), n, info)
        fl = fl + 2.d0 * dble(n) * dble(2 * S%kl(b) + S%ku(b) + 1)
      else
        cycle
      endif
      if (S%bw(b) /= 1.d0) wa(S%off(b) + 1 : S%off(b) + n) = S%bw(b) * wa(S%off(b) + 1 : S%off(b) + n)
    enddo
    !$omp end parallel do
    do b = 0, S%nb - 1
      if (S%dense(b) .or. S%band(b)) cycle
      n = S%sz(b)
      call VecGetArray(S%vb(b), va, ierr)
      va(1:n) = wa(S%off(b) + 1 : S%off(b) + n)
      call VecRestoreArray(S%vb(b), va, ierr)
      call KSPSolve(S%kb(b), S%vb(b), S%vx(b), ierr)
      call VecGetArray(S%vx(b), va, ierr)
      wa(S%off(b) + 1 : S%off(b) + n) = va(1:n)
      call VecRestoreArray(S%vx(b), va, ierr)
      if (S%bw(b) /= 1.d0) wa(S%off(b) + 1 : S%off(b) + n) = S%bw(b) * wa(S%off(b) + 1 : S%off(b) + n)
    enddo
    call VecRestoreArray(S%w, wa, ierr)
    call PetscLogFlops(fl, ierr)
    call PetscLogEventEnd(sev_asmblk, ierr)
    call VecSet(y, 0.d0, ierr)
    call VecScatterBegin(S%sct, S%w, y, ADD_VALUES, SCATTER_REVERSE, ierr)
    call VecScatterEnd(S%sct, S%w, y, ADD_VALUES, SCATTER_REVERSE, ierr)
    call PetscLogEventEnd(sev_asmapply, ierr)
  end subroutine asm_apply

  !> One report after the first setup: the levels' rows and entries, the
  !! operator complexity, the smoother.
  subroutine report(ksp, label, my_id)
    KSP, intent(in) :: ksp
    character(len=*), intent(in) :: label
    integer, intent(in) :: my_id
    PC :: pc
    KSP :: kl
    Mat :: al, pl
    MatInfo :: info
    PetscInt :: L
    PetscErrorCode :: ierr
    integer :: g, nbk(2), nbk_g(2), nsp_g, nbd(2)
    real*8 :: nz(0:MAXL-1), c_op
    character(len=9), parameter :: bn(4) = ["node     ", "line     ", "point    ", "linewedge"]
    character(len=40) :: tg
    tg = "[Physics PC]   "//trim(H%tag)
    call KSPGetPC(ksp, pc, ierr)
    do g = 0, H%nlev - 1
      L = H%nlev - 1 - g
      if (g == H%nlev - 1) then
        call PCMGGetCoarseSolve(pc, kl, ierr)
      else
        call PCMGGetSmoother(pc, L, kl, ierr)
      endif
      call KSPGetOperators(kl, al, pl, ierr)
      call MatGetInfo(al, MAT_GLOBAL_SUM, info, ierr)
      nz(g) = info%nz_used
    enddo
    c_op = sum(nz(0:H%nlev - 1)) / nz(0)
    nbk = 0
    do g = 0, H%nlev - 2
      nbk(1) = nbk(1) + H%as(g)%nb
      nbk(2) = max(nbk(2), maxval(H%as(g)%sz))
    enddo
    call MPI_Allreduce(nbk(1), nbk_g(1), 1, MPI_INTEGER, MPI_SUM, H%comm, ierr)
    call MPI_Allreduce(nbk(2), nbk_g(2), 1, MPI_INTEGER, MPI_MAX, H%comm, ierr)
    call MPI_Allreduce(H%as(0)%nsparse, nsp_g, 1, MPI_INTEGER, MPI_SUM, H%comm, ierr)
    nbd = [count(H%as(0)%band), maxval(H%as(0)%kl, mask=H%as(0)%band, dim=1)]
    nbd(2) = max(nbd(2), 0)
    call MPI_Allreduce(MPI_IN_PLACE, nbd(1), 1, MPI_INTEGER, MPI_SUM, H%comm, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, nbd(2), 1, MPI_INTEGER, MPI_MAX, H%comm, ierr)
    if (H%me == 0 .and. nbd(1) > 0) write(*,'(A,I0,A,I0)') trim(tg)//" level 0: ", nbd(1), &
      " block(s) on LAPACK band LU (-"//trim(H%prefix)//"band), largest kl ", nbd(2)
    if (H%me /= 0) return                                 ! the first rank of the operator's communicator
    write(*,'(A,A,A)', advance="no") trim(tg)//" (", trim(label), ") levels rows (nnz/row):"
    do g = 0, H%nlev - 1
      write(*,'(A,I0,A,F7.1,A)', advance="no") " ", H%lv(g)%n * H%ns, " (", nz(g) / max(H%lv(g)%n * H%ns, 1), ")"
    enddo
    write(*,'(A,F6.3)') "   C_op = ", c_op
    write(*,'(A,A,A,I0,A,I0,A,I0,A)') trim(tg)//" smoother: Chebyshev(", trim(itoa(H%o%degree)), &
      ") of additive Schwarz, "//trim(bn(H%o%blocks))//" blocks, axis block rings 0..", H%o%axis_rings, &
      "; ", nbk_g(1), " blocks over the levels, largest ", nbk_g(2), " rows"
    if (H%o%blocks == BL_LINE .or. H%o%blocks == BL_LINEWEDGE) write(*,'(A,I0,A)') &
      trim(tg)//" radial lines from ring ", merge(H%o%line_from, H%o%axis_rings + 1, H%o%line_from >= 0), &
      merge(" (overlapping the axis block)", "                             ", &
            H%o%line_from >= 0 .and. H%o%line_from <= H%o%axis_rings)
    if (H%o%blocks == BL_LINEWEDGE) write(*,'(A,I0,A,I0,A,F5.2,A,F5.2)') trim(tg)//" wedges of ", &
      H%o%wedge_lines, " lines every ", H%o%wedge_stride, " lines, rings with a cell aspect below ", &
      H%o%wedge_aspect, ", weight ", H%o%wedge_weight
    write(*,'(A,F6.2,A,F6.3,A,I0,A,A)') trim(tg)//" interval [lmax * ", H%o%safety / H%o%ratio, &
      ", lmax * ", H%o%safety, "], lmax from ", H%o%esteig_its, " CG steps; transfer ", trim(trname())
    if (nsp_g > 0) write(*,'(A,I0,A,I0,A)') trim(tg)//" level 0: ", nsp_g, &
      " block(s) above ", H%o%dense_max, " rows on sparse LU"
  end subroutine report

  !> -vs_pmg_test: CG with one V-cycle as preconditioner, the true residual
  !! norm, a random right-hand side in [-1, 1] and a zero start, to 1e-10;
  !! the iterations to 1e-1, 1e-6, 1e-10 and kappa from the Lanczos matrix
  !! (KSPComputeExtremeSingularValues), as the study's tables. With all
  !! toroidal slots in the operator: for all slots, then for the right-hand
  !! side on one family's slots alone (n = 0: slot 0; the k-th harmonic: the
  !! slots 2k - 1, 2k), which in a state whose families do not couple is the
  !! family's own CG (the study's per-family numbers).
  subroutine run_test(ksp, A, label, my_id)
    use mod_parameters, only: n_tor
    KSP, intent(in) :: ksp
    Mat, intent(in) :: A
    character(len=*), intent(in) :: label
    integer, intent(in) :: my_id
    KSP :: kt
    PC :: pc
    Vec :: b, x
    PetscRandom :: rnd
    PetscInt :: its, rs, re, r
    PetscCount :: nh
    PetscReal :: emax, emin
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ba(:)
    KSPConvergedReason :: reason
    integer :: t(3), k, q, fam, nfam, s
    character(len=24) :: what
    real*8, parameter :: tols(3) = [1.d-1, 1.d-6, 1.d-10]
    integer, parameter :: MAXIT = 500
    if (.not. allocated(H%hist)) allocate(H%hist(MAXIT + 1))
    nh = MAXIT + 1
    nfam = 0
    if (H%ns == n_tor .and. n_tor > 1) nfam = (n_tor + 1) / 2
    call KSPGetPC(ksp, pc, ierr)
    call MatCreateVecs(A, x, b, ierr)
    call MatGetOwnershipRange(A, rs, re, ierr)
    call PetscRandomCreate(H%comm, rnd, ierr)
    call PetscRandomSetInterval(rnd, -1.d0, 1.d0, ierr)
    call PetscRandomSetFromOptions(rnd, ierr)
    do fam = -1, nfam - 1
      call KSPCreate(H%comm, kt, ierr)
      call KSPSetType(kt, KSPCG, ierr)
      call KSPSetNormType(kt, KSP_NORM_UNPRECONDITIONED, ierr)
      call KSPSetOperators(kt, A, A, ierr)
      call KSPSetPC(kt, pc, ierr)
      call KSPSetTolerances(kt, 1.d-10, 1.d-50, 1.d10, int(MAXIT, kind(its)), ierr)
      call KSPSetComputeSingularValues(kt, PETSC_TRUE, ierr)
      call KSPSetResidualHistory(kt, H%hist, nh, PETSC_TRUE, ierr)
      call VecSetRandom(b, rnd, ierr)
      what = "all slots"
      if (fam >= 0) then
        call VecGetArray(b, ba, ierr)
        do r = rs, re - 1
          s = int(mod(r, int(H%ns, kind(r))))
          if ((s + 1) / 2 /= fam) ba(r - rs + 1) = 0.d0
        enddo
        call VecRestoreArray(b, ba, ierr)
        write(what, '(A,I0)') "family ", fam
      endif
      call VecSet(x, 0.d0, ierr)
      call KSPSolve(kt, b, x, ierr)
      call KSPGetIterationNumber(kt, its, ierr)
      call KSPGetConvergedReason(kt, reason, ierr)
      call KSPComputeExtremeSingularValues(kt, emax, emin, ierr)
      t = -1
      do q = 1, 3
        do k = 1, int(its) + 1
          if (H%hist(k) <= tols(q) * H%hist(1)) then
            t(q) = k - 1; exit
          endif
        enddo
      enddo
      if (H%me == 0) write(*,'(A,A,A,A,A,3(I0,A),ES10.3,A,ES10.3,A,F9.2,A,I0)') "[Physics PC]   "//trim(H%tag)// &
        " test (", trim(label), ", ", trim(what), "): CG its to 1e-1 / 1e-6 / 1e-10: ", t(1), " / ", t(2), " / ", t(3), &
        ";  Lanczos lambda in [", emin, ", ", emax, "], kappa ", emax / max(emin, tiny(1.d0)), ", reason ", reason%v
      call KSPDestroy(kt, ierr)
    enddo
    call PetscRandomDestroy(rnd, ierr)
    call VecDestroy(b, ierr)
    call VecDestroy(x, ierr)
  end subroutine run_test

  !> binary search of v in the ascending a(1:), 0-based position (-1 if absent)
  integer function bsearch(a, v)
    PetscInt, intent(in) :: a(:), v
    integer :: lo, hi, mid
    lo = 1; hi = size(a); bsearch = -1
    do while (lo <= hi)
      mid = (lo + hi) / 2
      if (a(mid) == v) then
        bsearch = mid - 1; return
      else if (a(mid) < v) then
        lo = mid + 1
      else
        hi = mid - 1
      endif
    enddo
  end function bsearch

  !> insertion sort of a short real array, ascending
  subroutine sort_real(a)
    real*8, intent(inout) :: a(:)
    integer :: i, j
    real*8 :: x
    do i = 2, size(a)
      x = a(i); j = i - 1
      do while (j >= 1)
        if (a(j) <= x) exit
        a(j + 1) = a(j); j = j - 1
      enddo
      a(j + 1) = x
    enddo
  end subroutine sort_real

  !> heap sort, ascending
  subroutine sort_pint(a)
    PetscInt, intent(inout) :: a(:)
    integer :: n, i, last
    PetscInt :: t
    n = size(a)
    do i = n / 2, 1, -1
      call sift(i, n)
    enddo
    do last = n, 2, -1
      t = a(1); a(1) = a(last); a(last) = t
      call sift(1, last - 1)
    enddo
  contains
    subroutine sift(start, nn)
      integer, intent(in) :: start, nn
      integer :: root, child
      PetscInt :: tt
      root = start
      do while (2 * root <= nn)
        child = 2 * root
        if (child < nn) then
          if (a(child) < a(child + 1)) child = child + 1
        endif
        if (a(root) < a(child)) then
          tt = a(root); a(root) = a(child); a(child) = tt
          root = child
        else
          return
        endif
      enddo
    end subroutine sift
  end subroutine sort_pint

  !> the transfer's name for the report
  function trname() result(s)
    character(len=96) :: s
    select case (H%o%transfer)
    case (TR_LINEAR);   s = "linear (kinds as fields)"
    case (TR_CONST);    s = "const (kinds as fields)"
    case (TR_RINGHARM)
      write(s, '(A,I0,A,I0,A,F5.2,A)') "ring-harmonic (", H%o%rh_sweeps, " steps, reach ", H%o%rh_reach, &
        ", omega ", H%o%rh_omega, ")"
    case (TR_MIXED)
      write(s, '(A,I0,A,I0,A,F5.2,A)') "mixed: n = 0 linear, harmonics ring-harmonic (", H%o%rh_sweeps, &
        " steps, reach ", H%o%rh_reach, ", omega ", H%o%rh_omega, ")"
    case default;       s = "?"
    end select
  end function trname

  function itoa(i) result(s)
    integer, intent(in) :: i
    character(len=12) :: s
    write(s, '(I0)') i
  end function itoa

  subroutine fatal(msg)
    character(len=*), intent(in) :: msg
    integer :: mpierr, r
    r = 0
    if (associated(H)) r = H%me
    if (r == 0) write(*,'(A,A)') "[Physics PC]   FATAL: SPD-MG: ", trim(msg)
    call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
  end subroutine fatal

#endif
end module mod_petsc_pc_spdmg
