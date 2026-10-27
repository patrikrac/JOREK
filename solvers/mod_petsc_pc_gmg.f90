!> Geometric multigrid on the exact nested C1 coarse space of JOREK's
!! logically structured polar grid: the GMG backend of the SF preconditioner's
!! blocks (mod_petsc_pc_sf_solver), up to four independent hierarchies.
!!
!! THE COARSE SPACE. JOREK's cubic element is Bernstein in disguise
!! (elements/mod_basisfunctions.f90): along an edge the control points are
!! c0 = u, c1 = u + size*d, so a node's derivative DOF is d = o * f_x/3 with x
!! the parametric index coordinate and o = +-1 (element%size is a pure sign on
!! flux-surface grids, grid_flux_surface.f90:628-643). The level-L space is
!! the set of fine functions that are bicubic Hermite on 2^L x 2^L PARAMETRIC
!! macro-elements -- a subspace by construction, so P is exact subdivision:
!! the tensor product of the 1D cubic-Hermite midpoint rule. No toroidal
!! coarsening (P acts identically on every harmonic) and the same P on both
!! fields of the pair.
!!
!! THE AXIS. force_central_node shares the axis value DOF. A coarse function
!! with a non-zero angular slope b on the axis would take values between
!! coarse nodes other than the shared one, which the fine space cannot
!! represent, so coarse axis b is dropped. The fine axis angular DOFs are then
!! outside range(P) and near-null (X(:,3) = 0 on the axis), so the block
!! smoothers solve the axis ring (rings 0..axis_k) as one block of its own.
!!
!! PARALLEL LAYOUT. The operator is packed rank by rank, [field 1 local |
!! field 2 local], each field in JOREK index order (the Nest->AIJ pack). The
!! fine layout is read off the operator (gmg_build_prolongations' Aref); a
!! coarse DOF belongs to the rank owning the fine node it is injected into, so
!! P is nearly block-diagonal across ranks. On one rank every map reduces to the
!! serial numbering. Smoother blocks are built from the OWNED rows, extended by
!! gmg_opts_t%line_overlap nodes into the ranks that own a radial line's
!! continuation (restricted additive Schwarz, see blk_t); without overlap a
!! line cut by the partition becomes one block per rank (block Jacobi, no
!! communication in the smoother).
module mod_petsc_pc_gmg
#ifdef USE_PETSC
  use mpi_mod
  !$ use omp_lib
#include "petsc/finclude/petsc.h"
  use petsc
  use iso_c_binding, only: c_ptr, c_double
  use mod_petsc_raw_csr, only: aij_parts, get_ij, put_ij, aij_vals_read, aij_vals_done, blockmv_attach
  use mod_petsc_pc_gmg_axis, only: axd_t, axd_nsec, axd_setup, axd_refill, axd_factor, axd_solve
  use mod_gmg_ring_btd
  implicit none
  private

  public :: gmg_build_prolongations, gmg_setup_operator, gmg_is_ready, gmg_select
  public :: gmg_pc_apply_1, gmg_pc_apply_2, gmg_pc_apply_3, gmg_pc_apply_4, gmg_pc_apply_5
  public :: gmg_opts_t

  !> Everything that configures a hierarchy, passed EXPLICITLY by the caller
  !! (mod_petsc_pc_sf_solver); no namelist entry reaches the GMG.
  type :: gmg_opts_t
    integer :: smoother     = 0      !< level smoother: 4 rings, 5 radial lines, 7 zebra lines,
                                     !< 8 zebra rings inside / zebra lines outside
    integer :: nsmooth      = 0      !< smoothing steps; <= 0 = the smoother's default
    integer :: pre0         = -1     !< level-0 pre-smoothing steps; < 0 = nsmooth (0 = none)
    integer :: post0        = -1     !< level-0 post-smoothing steps; < 0 = nsmooth (0 = none)
    integer :: nsmooth_c    = -1     !< pre and post steps on the coarse levels; <= 0 = nsmooth
    integer :: axis_rings   = 0      !< axis block extent: rings 0..k; -1 = by physical radius
    integer :: bnd_drop     = 0      !< drop Dirichlet DOFs from the coarse spaces
    integer :: ring_overlap = 0      !< smoother 4 (and 9's rings): a ring cut by a rank
                                     !< boundary is extended by up to this many nodes along
                                     !< J into the ranks owning the rest (RAS, as the lines)
    integer :: harm_pair    = 1      !< 1 = a harmonic's cos and sin slots share every
                                     !< smoother block (first-order d_phi couplings, which
                                     !< map cos <-> sin, then lie inside the blocks);
                                     !< 2 = every slot in one block (cross-|n| coupling).
                                     !< Set by the caller only (physics_pc_sf_harm_couple)
    integer :: ring_solver  = -1     !< ring blocks (smoothers 4, 8): 0 LAPACK band / dense LU,
                                     !< 1 periodic block Thomas, 2 Thomas / reduction hybrid
                                     !< threaded across the rank's rings and their segments
                                     !< (mod_gmg_ring_btd); -1 = 2 with harm_pair 2, else 0
    integer :: ring_segs    = 0      !< hybrid: segments per ring; 0 = ceil(threads / rings)
                                     !< when a rank has fewer rings than threads, else 1
    integer :: ring_btd_min = 0      !< ring blocks with fewer rows keep the LAPACK LU
    integer :: ring_check   = 0      !< 1 = residual of every ring solve after the first factorisation,
                                     !< 2 = and the Thomas fill's decay along the ring
    integer :: ring_single  = -1     !< 1 = ring solver factors kept and applied in single precision
                                     !< (factored in double; the smoothing solves are bandwidth-bound);
                                     !< -1 = 1 with a ring solver and harm_pair 2, else 0
    integer :: line_overlap = 0      !< smoothers 5/7: radial lines extended by this many
                                     !< nodes into the ranks owning their continuation
                                     !< (restricted additive Schwarz); 0 = local segments
    integer :: axis_sectors = 0      !< axis blocks solved over this many J-sector ranks
                                     !< (mod_petsc_pc_gmg_axis); -1 = its cost optimum,
                                     !< 0 = the sequential LU on the owning ranks
    integer :: blockmv      = 0      !< 1 = level operators and prolongations multiply
                                     !< with jorek_blockmv_attach.c's OpenMP kernel
    integer :: rich_from    = -1     !< levels >= rich_from smooth with Richardson on the
                                     !< smoother's blocks (no reductions); < 0 = none
    real*8  :: rich_omega   = 0.d0   !< Richardson scale on levels >= rich_from; <= 0 = 1
    real*8  :: ring_aspect  = 1.d0   !< smoother 8 / axis_rings -1 switch radius
    integer :: semi_r       = 0      !< levels 1..semi_r coarsen in I (radially) only:
                                     !< they keep the finer level's nj and P is the
                                     !< identity along J; the levels below coarsen both.
                                     !< < 0: as many as bring the median cell aspect
                                     !< r dtheta/dr to SEMI_ASPECT
  end type gmg_opts_t

  integer, parameter :: MAX_LEV   = 8
  !> the level cap of a full-coarsening hierarchy (as all tuned arms ran);
  !! semi-coarsened ones keep more rows per level and may use MAX_LEV
  integer, parameter :: MAX_LEV_FULL = 6
  integer, parameter :: CRS_MUMPS_MIN = 2000  !< coarsest LU rows from which MUMPS (SCOTCH) factors
  integer, parameter :: SM_STEPS  = 4      !< default smoothing steps
  integer, parameter :: MAX_ENT   = 16     !< max nonzeros per scalar row of P (2x2 sources x 4 DOFs)
  !> semi_r < 0: levels coarsen radially only while the median r dtheta/dr of
  !! their cells exceeds this; each such level halves it, a full one keeps it
  real*8, parameter :: SEMI_ASPECT = 2.d0

  type :: lvl_t
    integer :: ni = 0, nj = 0, n = 0
    integer, allocatable :: d(:,:,:)       !< (0:ni-1, 0:nj-1, 0:3) -> canonical DOF, -1 = dropped
    integer, allocatable :: rnode(:)       !< packed LOCAL row -> I*nj + J (block maps)
    integer, allocatable :: rharm(:)       !< packed LOCAL row -> toroidal slot m
    integer, allocatable :: rcol(:)        !< X-point grids: packed LOCAL row -> radial column
  end type lvl_t

  !> X-point grids: the smoothers' virtual lattice of one level
  !! (gmg_build_xpoint_prolongations). Ring I (the row's I = rnode / nj) is
  !! one flux line, nn(I) nodes long, closed (per) or open; the row's J is
  !! its position along it, and rcol / fine_col its radial column.
  type :: xring_t
    integer, allocatable :: nn(:)          !< (0:nring-1) positions along ring I
    logical, allocatable :: per(:)         !< ring I closes on itself
  end type xring_t

  !> Parallel layout of one level while the prolongations are built: scalar
  !! DOF d (0-based; the JOREK index-1 on the fine level) -> owner rank and
  !! position among that rank's DOFs. Packed global row of (f, d, m) is
  !! ps(own) + f*nl(own)*n_tor + lpos(d)*n_tor + m.
  type :: lay_t
    integer :: n = 0
    integer, allocatable :: own(:), lpos(:), nl(:), ps(:)   !< own/lpos(0:n-1), nl(0:np-1), ps(0:np)
  end type lay_t

  ! Fine level is 0. gP(g) maps level g -> level g-1 (g = 1..nlev-1).
  integer, save     :: nlev = 0
  Mat, save         :: gP(1:MAX_LEV-1)
  Mat, save         :: gA(0:MAX_LEV-1)      !< gA(0) borrowed (the fine operator), gA(g>0) owned
  KSP, save         :: gSm(0:MAX_LEV-1)
  Vec, save         :: gx(0:MAX_LEV-1), gb(0:MAX_LEV-1), gr(0:MAX_LEV-1)
  logical, save     :: p_ready = .false., op_ready = .false., vec_ready = .false.

  !> A direct solve of A(rows, rows) done REDUNDANTLY on the ranks that own
  !! some of those rows (sub-communicator comm; mostly one rank, since JOREK
  !! partitions ring by ring). Each member holds the whole submatrix as a
  !! sequential AIJ (sub(1)) with its own LU, so a solve costs no collective on
  !! the hierarchy's communicator -- only an Allgatherv among the members when
  !! the rows span ranks -- and non-members skip it. loc = the rank's rows
  !! (local, 0-based, ascending; set before the first rds_setup), cnt/dsp =
  !! the members' row counts/offsets. Used for the axis blocks and the
  !! coarsest level, which both live on a few ranks at scale.
  type :: rds_t
    logical :: ready = .false., member = .false.
    integer :: comm = MPI_COMM_NULL, np = 0, me = 0
    integer, allocatable :: loc(:), cnt(:), dsp(:)
    real*8, allocatable  :: send(:)
    IS  :: isq
    Mat, pointer :: sub(:) => null()
    KSP :: ksp
    Vec :: b, x
  end type rds_t

  ! The block smoothers (Chacon 2025 S4.1, extended along rings and radial
  ! lines): one block per line or ring and toroidal harmonic, holding every
  ! field and all four C1 DOFs, and the axis ring as one block per harmonic.
  ! bid(row) -> block (1-based);
  ! rows(off(b)+1 : off(b)+sz(b)) are the block's 0-based rows; the LU factors
  ! (column-major) start at lu(loff(b)+1), the pivots at piv(off(b)+1).
  type :: blk_t
    integer :: nb = 0, nrow = 0, nsing = 0
    integer, allocatable :: bid(:), off(:), sz(:), rows(:), pos(:), piv(:), kl(:), ku(:)
    logical, allocatable :: band(:)
    integer(8), allocatable :: loff(:)
    real*8, allocatable  :: lu(:)
    ! With axis_rings /= 0 the axis block (rings 0..axis_lim(g), every slot)
    ! is a sparse direct solve on its submatrix (rds_t, only the ranks owning
    ! axis rows take part): a dense LU of a fixed fraction of the level's rows
    ! would cost O(N^3). axblk(b) marks its blocks.
    logical :: axsparse = .false.
    logical, allocatable :: axblk(:)
    IS  :: axis_is
    ! its direct solve (one group, all slots; size 1)
    type(rds_t), allocatable :: axg(:)
    ! Smoother 7 (zebra radial lines): colour of each block (0 = even J lines,
    ! the axis block and the I = 0 block; 1 = odd J lines), and the coupling of
    ! the odd lines' rows to colour-0 columns, A(odd, even+axis), as local CSR
    ! over the rank's rows (zp 1-based pointers, zc 0-based local columns).
    integer, allocatable :: bcol(:), zp(:), zc(:)
    real*8, allocatable  :: zv(:)
    ! Overlapping line segments (restricted additive Schwarz, Cai & Sarkis,
    ! SISC 21 (1999) 792). The partition is ring-major, so every rank boundary
    ! cuts every radial line; with lovl > 0 each local line segment is
    ! extended by up to lovl nodes (all DOFs, fields, same slot) into the
    ! ranks that own the line's continuation, solved whole, and only the
    ! rank's own rows are written back. Row indices of the blocks live in a
    ! COMBINED space: 0..nrow-1 the rank's own rows, nrow..nrow+ngh-1 its
    ! ghost rows, in ascending global index gidx. xg receives the input's
    ! ghost values (scatter sct, neighbour ranks only); yg keeps the zebra's
    ! colour-0 ghost solutions, which the colour-1 lines couple to; sg(1) =
    ! A(ghost rows, all columns). With lovl = 0 nothing of this exists.
    logical :: ovl = .false.
    integer :: ngh = 0
    PetscInt, allocatable :: gidx(:)
    IS  :: isg, isall
    Vec :: xg
    VecScatter :: sct
    real*8, allocatable :: yg(:)
    Mat, pointer :: sg(:) => null()
    ! axis groups solved over J-sectors (axsec /= 0 and past the first-build
    ! gate against their LU): one axd_t per group, all or none
    logical :: axdon = .false.
    logical :: axpend = .false.            !< sector couplings refilled, factorisation pending (one setup)
    logical :: axnew = .false.             !< first build: factorisation and gate pending
    Mat :: axA                             !< the operator of the pending first build
    type(axd_t), allocatable :: axd(:)
    ! Value maps, built once per operator pattern (pat_id/pat_nz): the k-th
    ! entry of the source goes to lu(dst) (dst > 0) or zv(-dst) (dst < 0).
    ! Sources: vs = +k A's diagonal part, -k its off-diagonal part; gs = k in
    ! sg(1). A rebuild is then one gather per level, no MatGetRow.
    integer, allocatable    :: vs(:), gs(:)
    integer(8), allocatable :: vd(:), gd(:)
    integer(8) :: nvd = 0, nvo = 0, nvg = 0          !< source value-array lengths
    integer(8) :: pat_id = -1, pat_nz = -1
    ! ring_solver > 0: ring blocks solved as periodic block-tridiagonal
    ! chains of node blocks (mod_gmg_ring_btd). btd(b) marks them; their rows
    ! are in natural J order (node by node), qn(off(b)+q) = the node (0..nn-1)
    ! of block position q, rg(b) the layout, factor and pivots in lu / piv
    ! as the LAPACK blocks'. jac(b): a singular node block, point Jacobi.
    ! ghJ: J of each ghost row (ovl_extend). Hybrid on smoother 4: the
    ! (block lb, segment lk) work items of all the rank's rings, and the work
    ! vector wk (block order) the threaded solve runs on.
    logical, allocatable :: btd(:), jac(:)
    integer, allocatable :: qn(:), ghJ(:)
    type(btd_ring), allocatable :: rg(:)
    integer, allocatable :: lb(:), lk(:)
    real*8, allocatable :: wk(:)
    ! ring_single: the ring blocks' factors rounded to single (lu4, block b at
    ! loff4(b)), which the solves read instead of lu; wk4 = wk's single copy
    real*4, allocatable :: lu4(:), wk4(:)
    ! ring_single: the rings' equilibration S (btd_equil), block order as wk;
    ! 1 on every other block
    real*8, allocatable :: eqs(:)
    integer(8), allocatable :: loff4(:)
  end type blk_t
  type(blk_t), save, target :: gBk(0:MAX_LEV-1)
  type(rds_t), save :: gcrs                  !< the coarsest level's direct solve
  type(lvl_t), allocatable, save :: glv(:)   !< coarse DOF numberings, kept for the block maps
  integer, allocatable, save :: fine_node(:) !< level-0 row -> node-1 (i*n_tht + j), from node%index
  integer, allocatable, save :: fine_harm(:) !< level-0 row -> toroidal slot m
  integer, allocatable, save :: fine_col(:)  !< X-point grids: level-0 row -> radial column
  logical, save :: xgrid = .false.           !< the hierarchy is an X-point grid's (virtual lattice)
  type(xring_t), save :: xrg(0:MAX_LEV-1)    !< X-point grids: ring tables per level
  ! X-point grids on several ranks: JOREK numbers the divertor legs column by
  ! column (every private and SOL ring at one poloidal position), so a rank
  ! there holds short pieces of all leg rings and the ring overlap completes
  ! each of them (xg65 at 64 ranks: 572% ghost rows, closed grid 152%). The
  ! hierarchy then works in its own layout: the DOFs in ring-major order,
  ! cut after the same rows per rank (xsplit 0, the default) or on ring
  ! boundaries (xsplit 1, ring_split: no ring shared, so no ring ghosts).
  ! xp_is: old global row of every new local row;
  ! xp_A = A(xp_is, xp_is); xp_sc scatters the PC's vectors in and out.
  logical, save :: xperm = .false.
  IS, save :: xp_is
  Mat, save :: xp_A
  Vec, save :: xp_b, xp_x
  VecScatter, save :: xp_sc
  integer, save :: nth0 = 0                  !< n_tht (level-0 nj)
  integer, save :: nf_s = 0, cur_lev = 0
  integer, save :: sm_type = 0, sm_nstep = 4
  real*8, allocatable, save :: blk_t_work(:,:)   !< (max block size, 0:threads-1)
  integer, parameter :: LINES_OMP_MIN = 4000     !< rows below which lines_solve stays serial
  ! Axis treatment (Bourne et al., JCP 488 (2023) 112249 S2-S3). Fine rings
  ! I < ring_is have median r*dtheta/dr below gmg_opts_t%ring_aspect: the
  ! circle couplings dominate there, so smoother 8 uses ring blocks inside and
  ! radial lines outside (GMGPolar). Rings 0..axis_k form the axis block.
  integer, save :: ring_is = 1, axis_k = 0
  integer, save :: lovl = 0                 !< line overlap in nodes (gmg_opts_t%line_overlap)
  integer, save :: hpair = 1                !< gmg_opts_t%harm_pair
  integer, save :: rovl = 0                 !< gmg_opts_t%ring_overlap (ring blocks)
  integer, save :: rsolv = 0, rbmin = 0, rchk = 0   !< gmg_opts_t%ring_solver / ring_btd_min / ring_check
  integer, save :: rseg = 0                         !< gmg_opts_t%ring_segs
  logical, save :: rsp = .false.                    !< gmg_opts_t%ring_single
  integer, save :: semi_r = 0               !< gmg_opts_t%semi_r, of the hierarchy being built
  integer, save :: axsec = 0                !< axis J-sectors (gmg_opts_t%axis_sectors)
  integer, save :: gcomm = 0, gme = 0       !< communicator and rank of the hierarchy
  ! Galerkin chain reuse: the fine operand's sparsity pattern is fixed (the
  ! physics PC even keeps the same Mat for the run), so the PtAP symbolic phase
  ! (80% of GMG_PtAP) is done once and later rebuilds refill values
  ! (MAT_REUSE_MATRIX). a0_id/a0_nzst identify the last operand; for another
  ! Mat, a0_sig (a hash of the pattern) decides, and a mismatch falls back to
  ! a fresh product.
  integer(8), save :: a0_sig(2) = 0
  integer(8), save :: a0_id = 0, a0_nzst = -1   !< the last fine operand and its nonzero state
  ! Set for one setup when a rebuild's fine pattern differs from the last one's:
  ! the redundant LUs (rds_setup) then extract and analyse anew, and a level's
  ! J-sector axis solve, whose submatrices are pattern-bound, gives way to the LU.
  logical, save :: pat_fresh = .false.

  ! Several independent hierarchies: the routines work on the module-level
  ! state, and gmg_select(k) parks the active instance's state in
  ! inst(cur_inst) and loads instance k (handles copied, arrays moved, O(1)).
  ! Instance 1 = pair_w, 2 = pair_psi, 3 = rho, 4 = T, 5 = v_par.
  integer, parameter :: MAX_INST = 5
  type :: gmg_inst_t
    integer :: nlev = 0, nth0 = 0, nf_s = 0, sm_type = 0, sm_nstep = 4
    integer :: ring_is = 1, axis_k = 0, lovl = 0, axsec = 0, hpair = 1, rovl = 0
    integer :: rsolv = 0, rbmin = 0, rchk = 0, rseg = 0
    logical :: rsp = .false.
    integer(8) :: a0_sig(2) = 0, a0_id = 0, a0_nzst = -1
    logical :: p_ready = .false., op_ready = .false., vec_ready = .false.
    Mat :: gP(1:MAX_LEV-1), gA(0:MAX_LEV-1)
    KSP :: gSm(0:MAX_LEV-1)
    type(rds_t) :: gcrs
    Vec :: gx(0:MAX_LEV-1), gb(0:MAX_LEV-1), gr(0:MAX_LEV-1)
    type(blk_t) :: gBk(0:MAX_LEV-1)
    type(lvl_t), allocatable :: glv(:)
    integer, allocatable :: fine_node(:), fine_harm(:), fine_col(:)
    logical :: xgrid = .false.
    type(xring_t) :: xrg(0:MAX_LEV-1)
    logical :: xperm = .false.
    IS :: xp_is
    Mat :: xp_A
    Vec :: xp_b, xp_x
    VecScatter :: xp_sc
  end type gmg_inst_t
  type(gmg_inst_t), save :: inst(MAX_INST)
  integer, save :: cur_inst = 1
  !> The V-cycle's shape per instance (gmg_opts_t pre0 / post0 / nsmooth_c,
  !! resolved against nsmooth at setup): level-0 pre and post steps, and the
  !! steps of every coarser level. Kept by instance, like the hierarchy.
  integer, save :: st_pre0(MAX_INST) = SM_STEPS, st_post0(MAX_INST) = SM_STEPS
  integer, save :: st_crs(MAX_INST) = SM_STEPS

  ! -log_view events; the apply-side ones per instance (GMG_*, GMG<k>_*)
  PetscLogEvent, save :: gev_ptap = -1, gev_smsetup = -1, gev_coarselu = -1
  PetscLogEvent, save :: gev_prolong = -1
  PetscLogEvent, save :: gev_blkmesh = -1, gev_blkpat = -1, gev_blknum = -1, gev_axnum = -1, gev_axtry = -1
  PetscLogEvent, save :: gev_lines(MAX_INST) = -1, gev_axsolve(MAX_INST) = -1   !< the block smoother's two halves
  PetscLogEvent, save :: gev_vcycle(MAX_INST) = -1, gev_smooth0(MAX_INST) = -1, gev_smooth(MAX_INST) = -1
  PetscLogEvent, save :: gev_coarse(MAX_INST) = -1
  logical, save       :: gev_ready = .false.

  ! 1D cubic-Hermite subdivision, level-local canonical units (u, f_x/3):
  ! W(p+1, q+1) maps coarse quantity q to fine quantity p.
  real*8, parameter :: W_CO(2,2) = reshape([1.0d0, 0.0d0, 0.0d0, 0.5d0],      [2,2])
  real*8, parameter :: W_L(2,2)  = reshape([0.5d0, -0.25d0, 0.375d0, -0.125d0], [2,2])
  real*8, parameter :: W_R(2,2)  = reshape([0.5d0, 0.25d0, -0.375d0, -0.125d0], [2,2])
  real*8, parameter :: W_ID(2,2) = reshape([1.0d0, 0.0d0, 0.0d0, 1.0d0],      [2,2])  !< not coarsened
  integer, parameter :: PQ_I(0:3) = [0, 1, 0, 1]   !< canonical k -> derivative order in i (u, a, b, c)
  integer, parameter :: PQ_J(0:3) = [0, 0, 1, 1]   !< ... and in j
  ! X-point grids (gmg_build_xpoint_prolongations): element slot k -> its
  ! vertex's (dA, dB) in the element lattice, and its corner of the Bezier
  ! net (i along s / across, j along t / along) with the inward directions
  integer, parameter :: SLOT_A_(4) = [0, 0, 1, 1], SLOT_B_(4) = [0, 1, 1, 0]
  integer, parameter :: NET_I(4) = [0, 3, 3, 0], NET_J(4) = [0, 0, 3, 3]
  integer, parameter :: NET_DI(4) = [1, -1, -1, 1], NET_DJ(4) = [1, 1, -1, -1]

contains

  !> Make hierarchy k the active one (see gmg_inst_t).
  subroutine gmg_select(k)
    integer, intent(in) :: k
    integer :: g
    if (k == cur_inst) return
    ! park the active state
    associate (S => inst(cur_inst))
      S%nlev = nlev; S%nth0 = nth0; S%nf_s = nf_s; S%sm_type = sm_type; S%sm_nstep = sm_nstep
      S%ring_is = ring_is; S%axis_k = axis_k
      S%lovl = lovl; S%axsec = axsec; S%hpair = hpair; S%rovl = rovl
      S%rsolv = rsolv; S%rbmin = rbmin; S%rchk = rchk; S%rsp = rsp; S%rseg = rseg
      S%a0_sig = a0_sig; S%a0_id = a0_id; S%a0_nzst = a0_nzst
      S%p_ready = p_ready; S%op_ready = op_ready; S%vec_ready = vec_ready
      S%gP = gP; S%gA = gA; S%gSm = gSm
      S%gx = gx; S%gb = gb; S%gr = gr
      do g = 0, MAX_LEV - 1
        call move_blk(gBk(g), S%gBk(g))
      enddo
      call move_rds(gcrs, S%gcrs)
      if (allocated(glv)) call move_alloc(glv, S%glv)
      if (allocated(fine_node)) call move_alloc(fine_node, S%fine_node)
      if (allocated(fine_harm)) call move_alloc(fine_harm, S%fine_harm)
      if (allocated(fine_col)) call move_alloc(fine_col, S%fine_col)
      S%xgrid = xgrid
      S%xperm = xperm; S%xp_is = xp_is; S%xp_A = xp_A; S%xp_b = xp_b; S%xp_x = xp_x; S%xp_sc = xp_sc
      do g = 0, MAX_LEV - 1
        if (allocated(xrg(g)%nn)) call move_alloc(xrg(g)%nn, S%xrg(g)%nn)
        if (allocated(xrg(g)%per)) call move_alloc(xrg(g)%per, S%xrg(g)%per)
      enddo
    end associate
    ! load instance k
    associate (S => inst(k))
      nlev = S%nlev; nth0 = S%nth0; nf_s = S%nf_s; sm_type = S%sm_type; sm_nstep = S%sm_nstep
      ring_is = S%ring_is; axis_k = S%axis_k
      lovl = S%lovl; axsec = S%axsec; hpair = S%hpair; rovl = S%rovl
      rsolv = S%rsolv; rbmin = S%rbmin; rchk = S%rchk; rsp = S%rsp; rseg = S%rseg
      a0_sig = S%a0_sig; a0_id = S%a0_id; a0_nzst = S%a0_nzst
      p_ready = S%p_ready; op_ready = S%op_ready; vec_ready = S%vec_ready
      gP = S%gP; gA = S%gA; gSm = S%gSm
      gx = S%gx; gb = S%gb; gr = S%gr
      do g = 0, MAX_LEV - 1
        call move_blk(S%gBk(g), gBk(g))
      enddo
      call move_rds(S%gcrs, gcrs)
      if (allocated(S%glv)) call move_alloc(S%glv, glv)
      if (allocated(S%fine_node)) call move_alloc(S%fine_node, fine_node)
      if (allocated(S%fine_harm)) call move_alloc(S%fine_harm, fine_harm)
      if (allocated(S%fine_col)) call move_alloc(S%fine_col, fine_col)
      xgrid = S%xgrid
      xperm = S%xperm; xp_is = S%xp_is; xp_A = S%xp_A; xp_b = S%xp_b; xp_x = S%xp_x; xp_sc = S%xp_sc
      do g = 0, MAX_LEV - 1
        if (allocated(S%xrg(g)%nn)) call move_alloc(S%xrg(g)%nn, xrg(g)%nn)
        if (allocated(S%xrg(g)%per)) call move_alloc(S%xrg(g)%per, xrg(g)%per)
      enddo
    end associate
    cur_inst = k
  end subroutine gmg_select

  subroutine move_blk(a, b)
    type(blk_t), intent(inout) :: a, b
    b%nb = a%nb; b%nrow = a%nrow; b%nsing = a%nsing
    a%nb = 0; a%nrow = 0; a%nsing = 0
    b%axsparse = a%axsparse; b%axis_is = a%axis_is
    a%axsparse = .false.
    if (allocated(a%axg)) call move_alloc(a%axg, b%axg)
    if (allocated(a%axblk)) call move_alloc(a%axblk, b%axblk)
    if (allocated(a%bid))  call move_alloc(a%bid,  b%bid)
    if (allocated(a%off))  call move_alloc(a%off,  b%off)
    if (allocated(a%sz))   call move_alloc(a%sz,   b%sz)
    if (allocated(a%rows)) call move_alloc(a%rows, b%rows)
    if (allocated(a%pos))  call move_alloc(a%pos,  b%pos)
    if (allocated(a%piv))  call move_alloc(a%piv,  b%piv)
    if (allocated(a%kl))   call move_alloc(a%kl,   b%kl)
    if (allocated(a%ku))   call move_alloc(a%ku,   b%ku)
    if (allocated(a%band)) call move_alloc(a%band, b%band)
    if (allocated(a%loff)) call move_alloc(a%loff, b%loff)
    if (allocated(a%lu))   call move_alloc(a%lu,   b%lu)
    if (allocated(a%bcol)) call move_alloc(a%bcol, b%bcol)
    if (allocated(a%zp))   call move_alloc(a%zp,   b%zp)
    if (allocated(a%zc))   call move_alloc(a%zc,   b%zc)
    if (allocated(a%zv))   call move_alloc(a%zv,   b%zv)
    b%ovl = a%ovl; b%ngh = a%ngh; a%ovl = .false.; a%ngh = 0
    b%isg = a%isg; b%isall = a%isall; b%xg = a%xg; b%sct = a%sct
    b%sg => a%sg; a%sg => null()
    if (allocated(a%gidx)) call move_alloc(a%gidx, b%gidx)
    if (allocated(a%yg))   call move_alloc(a%yg,   b%yg)
    if (allocated(a%vs))   call move_alloc(a%vs,   b%vs)
    if (allocated(a%gs))   call move_alloc(a%gs,   b%gs)
    if (allocated(a%vd))   call move_alloc(a%vd,   b%vd)
    if (allocated(a%gd))   call move_alloc(a%gd,   b%gd)
    b%nvd = a%nvd; b%nvo = a%nvo; b%nvg = a%nvg
    b%axdon = a%axdon; a%axdon = .false.
    if (allocated(a%axd))  call move_alloc(a%axd,  b%axd)
    b%pat_id = a%pat_id; b%pat_nz = a%pat_nz; a%pat_id = -1; a%pat_nz = -1
    if (allocated(a%btd))  call move_alloc(a%btd,  b%btd)
    if (allocated(a%jac))  call move_alloc(a%jac,  b%jac)
    if (allocated(a%qn))   call move_alloc(a%qn,   b%qn)
    if (allocated(a%ghJ))  call move_alloc(a%ghJ,  b%ghJ)
    if (allocated(a%rg))   call move_alloc(a%rg,   b%rg)
    if (allocated(a%lb))   call move_alloc(a%lb,   b%lb)
    if (allocated(a%lk))   call move_alloc(a%lk,   b%lk)
    if (allocated(a%wk))   call move_alloc(a%wk,   b%wk)
    if (allocated(a%lu4))  call move_alloc(a%lu4,  b%lu4)
    if (allocated(a%wk4))  call move_alloc(a%wk4,  b%wk4)
    if (allocated(a%loff4)) call move_alloc(a%loff4, b%loff4)
    if (allocated(a%eqs))  call move_alloc(a%eqs,  b%eqs)
  end subroutine move_blk

  subroutine move_rds(a, b)
    type(rds_t), intent(inout) :: a, b
    b%ready = a%ready; b%member = a%member
    b%comm = a%comm; b%np = a%np; b%me = a%me
    b%isq = a%isq; b%ksp = a%ksp; b%b = a%b; b%x = a%x
    b%sub => a%sub; a%sub => null()
    a%ready = .false.; a%member = .false.; a%comm = MPI_COMM_NULL; a%np = 0; a%me = 0
    if (allocated(a%loc))  call move_alloc(a%loc,  b%loc)
    if (allocated(a%cnt))  call move_alloc(a%cnt,  b%cnt)
    if (allocated(a%dsp))  call move_alloc(a%dsp,  b%dsp)
    if (allocated(a%send)) call move_alloc(a%send, b%send)
  end subroutine move_rds

  !> (Re)build R for A(rows, rows), rows = R%loc on each rank. what = options
  !! prefix part (gmg<k>_<what>_); tag = label of the first build's
  !! factor-size print. Every owner of some of the rows solves.
  subroutine rds_setup(R, A, what, tag)
    type(rds_t), intent(inout) :: R
    Mat, intent(in)     :: A
    character(len=*), intent(in) :: what, tag
    PetscInt, parameter :: one = 1
    PetscInt :: rst, ren
    PetscErrorCode :: ierr
    PC :: pc
    integer :: color, mpierr, k, nl, ntot
    integer, allocatable :: gl(:)
    character(len=64) :: pre
    logical :: fresh

    call MatGetOwnershipRange(A, rst, ren, ierr)
    ! a new operator pattern: the submatrix and the factorisation's symbolic
    ! phase are pattern-bound, the members and the row set are not
    fresh = .not. R%ready .or. pat_fresh
    if (R%ready .and. pat_fresh) then
      call MatDestroySubMatrices(one, R%sub, ierr)
      if (R%member) then
        call KSPDestroy(R%ksp, ierr)
        call VecDestroy(R%x, ierr)
        call VecDestroy(R%b, ierr)
      endif
    endif
    if (.not. R%ready) then
      ! members, and the whole row set (global, ascending = rank order) on each
      nl = size(R%loc)
      color = MPI_UNDEFINED
      if (nl > 0) color = 1
      call MPI_Comm_split(gcomm, color, gme, R%comm, mpierr)
      R%member = (color == 1)
      ntot = 0
      if (R%member) then
        call MPI_Comm_size(R%comm, R%np, mpierr)
        call MPI_Comm_rank(R%comm, R%me, mpierr)
        allocate(R%cnt(0:R%np - 1), R%dsp(0:R%np), R%send(nl))
        call MPI_Allgather(nl, 1, MPI_INTEGER, R%cnt, 1, MPI_INTEGER, R%comm, mpierr)
        R%dsp(0) = 0
        do k = 0, R%np - 1
          R%dsp(k + 1) = R%dsp(k) + R%cnt(k)
        enddo
        ntot = R%dsp(R%np)
        allocate(gl(ntot))
        call MPI_Allgatherv(int(rst) + R%loc, nl, MPI_INTEGER, gl, R%cnt, R%dsp(0:R%np - 1), &
                            MPI_INTEGER, R%comm, mpierr)
      else
        allocate(gl(0))
      endif
      call ISCreateGeneral(PETSC_COMM_SELF, int(ntot, kind(rst)), int(gl, kind(rst)), &
                           PETSC_COPY_VALUES, R%isq, ierr)
      deallocate(gl)
    endif
    ! collective on A's communicator (non-members pass an empty IS)
    ! Extracted and analysed ONCE; later rebuilds refill the values in place
    ! and refactor numerically only. The assembly after the refill is not
    ! optional: at np > 1 MAT_REUSE_MATRIX refills the values WITHOUT advancing
    ! the submatrix's object state, so PCSetUp would silently keep the old
    ! factor (measured: coarse self-check error 2e-3 from the second build on,
    ! and two numeric factorisations missing per rebuild). That missing state
    ! bump was the whole of the old "stale at np > 1" bug, previously worked
    ! around by re-extracting and re-analysing at every rebuild.
    if (fresh) then
      call MatCreateSubMatrices(A, one, [R%isq], [R%isq], MAT_INITIAL_MATRIX, R%sub, ierr)
    else
      call MatCreateSubMatrices(A, one, [R%isq], [R%isq], MAT_REUSE_MATRIX, R%sub, ierr)
      call MatAssemblyBegin(R%sub(1), MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(R%sub(1), MAT_FINAL_ASSEMBLY, ierr)
    endif
    if (R%member) then
      if (fresh) then
        ! the coarsest level from CRS_MUMPS_MIN rows on: MUMPS with SCOTCH
        ! ordering (ICNTL(7) = 3), threaded and with half the fill of PETSc's
        ! ND LU. 161x64 n_tor 7, pair_w 4508 rows (2026-10-01, sf_runs/sf_ss,
        ! 10 steps): factor entries 7.2e6 -> 3.8e6, setup np 64 x 16 6.23 ->
        ! 4.21 s, np 128 x 8 5.56 -> 3.24 s, solve unchanged or -2%. MUMPS'
        ! default ordering instead: 1.7e7 entries, solve +4-8%. The smaller
        ! coarsest levels (287-574 rows) cost < 0.1 s in all and keep PETSc's.
        block
          PetscInt :: nsub
          call MatGetSize(R%sub(1), nsub, PETSC_NULL_INTEGER, ierr)
          if (what == "coarse" .and. nsub >= CRS_MUMPS_MIN) then
            call rds_make_ksp(MATSOLVERMUMPS)
          else
            call rds_make_ksp(MATSOLVERPETSC)
          endif
        end block
        call MatCreateVecs(R%sub(1), R%x, R%b, ierr)
      else
        call KSPSetOperators(R%ksp, R%sub(1), R%sub(1), ierr)
      endif
    endif
    if (R%member) then
      call KSPSetUp(R%ksp, ierr)
      ! PETSc's LU does not pivot: on a failed factorisation, MUMPS instead
      block
        PCFailedReason :: why
        call KSPGetPC(R%ksp, pc, ierr)
        call PCGetFailedReason(pc, why, ierr)
        if (why /= PC_NOERROR) then
          write(*,'(A,I0,A,A,A)') "[Physics PC]   GMG", cur_inst, " ", tag, &
            ": PETSc LU failed (zero pivot?), refactoring with MUMPS"
          call KSPDestroy(R%ksp, ierr)
          call rds_make_ksp(MATSOLVERMUMPS)
          call KSPSetUp(R%ksp, ierr)
        endif
      end block
    endif
    if (.not. R%ready .and. R%member) then      ! factor size
      if (R%me == 0) then
        block
          Mat :: F
          MatInfo :: finfo
          PetscInt :: nax
          call KSPGetPC(R%ksp, pc, ierr)
          call PCFactorGetMatrix(pc, F, ierr)
          call MatGetInfo(F, MAT_LOCAL, finfo, ierr)
          call ISGetSize(R%isq, nax, ierr)
          write(*,'(A,I0,A,A,A,I0,A,I0,A,ES10.3)') "[Mem] GMG", cur_inst, " ", tag, " LU: ", &
            nax, " rows on ", R%np, " rank(s), factor entries ", finfo%nz_used
        end block
      endif
    endif
    R%ready = .true.

  contains

    !> Sequential LU KSP of R%sub(1). Default PETSc's own LU (ordering:
    !! nested dissection, quotient minimum degree for the axis blocks): on
    !! these few-thousand-row blocks its
    !! triangular solve is ~2.5x cheaper per call than MUMPS'. Options under
    !! gmg<k>_<what>_ override both choices.
    subroutine rds_make_ksp(stype)
      character(len=*), intent(in) :: stype   ! not MatSolverType (len=80): ifort rejects the short constants
      call KSPCreate(PETSC_COMM_SELF, R%ksp, ierr)
      call KSPSetOperators(R%ksp, R%sub(1), R%sub(1), ierr)
      call KSPSetType(R%ksp, KSPPREONLY, ierr)
      call KSPGetPC(R%ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, stype, ierr)
      if (stype == MATSOLVERPETSC) then
        ! axis blocks: quotient minimum degree. On the mixed pair_w block
        ! (161x64, 11532 rows) it leaves 6.2e6 factor entries where nested
        ! dissection leaves 1.7e7: the numeric factorisation is 9x cheaper
        ! and the triangular solve 2x
        if (what(1:2) == "ax") then
          call PCFactorSetMatOrderingType(pc, MATORDERINGQMD, ierr)
        else
          call PCFactorSetMatOrderingType(pc, MATORDERINGND, ierr)
        endif
      endif
      write(pre, '(A,I0,A,A,A)') "gmg", cur_inst, "_", what, "_"   ! sequential: no ICNTL(20) needed
      if (stype == MATSOLVERMUMPS) then
        ! SCOTCH ordering unless the user chose one (see CRS_MUMPS_MIN)
        block
          character(len=80) :: nm
          PetscBool :: has
          nm = "-"//trim(pre)//"mat_mumps_icntl_7"
          call PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(nm), has, ierr)
          if (.not. has) call PetscOptionsSetValue(PETSC_NULL_OPTIONS, trim(nm), "3", ierr)
        end block
      endif
      call KSPSetOptionsPrefix(R%ksp, trim(pre), ierr)
      call KSPSetFromOptions(R%ksp, ierr)
    end subroutine rds_make_ksp
  end subroutine rds_setup

  !> yy(R%loc) = A(rows, rows)^-1 xx(R%loc) on the members; others return.
  subroutine rds_solve(R, xx, yy)
    type(rds_t), intent(inout) :: R
    Vec :: xx, yy
    PetscScalar, pointer :: xp(:), yp(:)
    PetscErrorCode :: ie
    if (.not. R%member) return
    call VecGetArrayRead(xx, xp, ie)
    call VecGetArray(yy, yp, ie)
    call rds_solve_arr(R, xp, yp)
    call VecRestoreArray(yy, yp, ie)
    call VecRestoreArrayRead(xx, xp, ie)
  end subroutine rds_solve

  !> rds_solve on the arrays of xx and yy (blk_apply holds them already).
  subroutine rds_solve_arr(R, xp, yp)
    type(rds_t), intent(inout) :: R
    PetscScalar, intent(in)    :: xp(:)
    PetscScalar, intent(inout) :: yp(:)
    PetscScalar, pointer :: bp(:)
    PetscErrorCode :: ie
    integer :: q, nl, o, mpierr
    if (.not. R%member) return
    nl = size(R%loc)
    o = R%dsp(R%me)
    call VecGetArray(R%b, bp, ie)
    if (R%np == 1) then
      do q = 1, nl
        bp(q) = xp(R%loc(q) + 1)
      enddo
    else
      do q = 1, nl
        R%send(q) = xp(R%loc(q) + 1)
      enddo
      call MPI_Allgatherv(R%send, nl, MPI_DOUBLE_PRECISION, bp, R%cnt, R%dsp(0:R%np - 1), &
                          MPI_DOUBLE_PRECISION, R%comm, mpierr)
    endif
    call VecRestoreArray(R%b, bp, ie)
    call KSPSolve(R%ksp, R%b, R%x, ie)
    call VecGetArrayRead(R%x, bp, ie)
    do q = 1, nl
      yp(R%loc(q) + 1) = bp(o + q)
    enddo
    call VecRestoreArrayRead(R%x, bp, ie)
  end subroutine rds_solve_arr

  !> PC-shell entry points, one per hierarchy instance: x = one V-cycle on b.
  subroutine gmg_pc_apply_1(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call vcycle_of(1, b, x, ierr)
  end subroutine gmg_pc_apply_1

  subroutine gmg_pc_apply_2(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call vcycle_of(2, b, x, ierr)
  end subroutine gmg_pc_apply_2

  subroutine gmg_pc_apply_3(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call vcycle_of(3, b, x, ierr)
  end subroutine gmg_pc_apply_3

  subroutine gmg_pc_apply_4(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call vcycle_of(4, b, x, ierr)
  end subroutine gmg_pc_apply_4

  subroutine gmg_pc_apply_5(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call vcycle_of(5, b, x, ierr)
  end subroutine gmg_pc_apply_5

  subroutine vcycle_of(k, b, x, ierr)
    integer, intent(in) :: k
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_select(k)
    call PetscLogEventBegin(gev_vcycle(cur_inst), ierr)
    if (xperm) then
      call VecScatterBegin(xp_sc, b, xp_b, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd(xp_sc, b, xp_b, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call vcycle(0, xp_b, xp_x)
      call VecScatterBegin(xp_sc, xp_x, x, INSERT_VALUES, SCATTER_REVERSE, ierr)
      call VecScatterEnd(xp_sc, xp_x, x, INSERT_VALUES, SCATTER_REVERSE, ierr)
    else
      call vcycle(0, b, x)
    endif
    call PetscLogEventEnd(gev_vcycle(cur_inst), ierr)
    ierr = 0
  end subroutine vcycle_of

  logical function gmg_is_ready()
    gmg_is_ready = p_ready
  end function gmg_is_ready

  !--------------------------------------------------------------------
  !> Build the prolongations once per run (they depend on the grid only).
  !! Aref: an operator with the packed layout the hierarchy will serve (its
  !! row ownership fixes every level's parallel layout; its values are unused).
  !! n_fields: number of identically indexed fields packed per rank in the
  !! operator (2 for pair_w). ok = .false. if the grid is not the structured
  !! flux-surface layout this construction needs.
  !--------------------------------------------------------------------
  subroutine gmg_build_prolongations(Aref, comm, my_id, n_fields, ok, opts)
    use nodes_elements, only: node_list, element_list
    use phys_module,    only: n_flux, n_tht
    use mod_parameters, only: n_degrees
    use mod_petsc_pc_harm, only: pc_ntor
    use mod_petsc_pc_xpatch, only: xpatch_is_xgrid

    Mat, intent(in)      :: Aref
    integer, intent(in)  :: comm, my_id, n_fields
    logical, intent(out) :: ok
    type(gmg_opts_t), intent(in) :: opts
    type(gmg_opts_t) :: gopt

    type(lvl_t), allocatable :: lv(:)
    integer, allocatable :: o(:,:)
    logical, allocatable :: oset(:,:)
    integer :: n_inact
    ! (lv is handed to the module-level glv at the end, for the block maps)
    integer :: n_idx, e, iv, n, k, i, j, ni, nj, g, cand(4), nent, nrow_s
    integer :: eps_s(4), eps_t(4), cols(MAX_ENT), vi(4), vj(4), idx
    real*8  :: w(MAX_ENT)
    integer, allocatable :: rs(:), cs(:)
    real*8,  allocatable :: ws(:)
    logical, allocatable :: done(:)
    integer :: ns, nlev_cap
    real*8, allocatable :: asp(:,:), asp_sorted(:)
    type(lay_t), allocatable :: ly(:)
    integer, allocatable :: rstarts(:)
    integer :: me, np, mpierr, r, first, ci, cj, dd, own_, ff, mm, lr, nloc
    PetscInt :: rs0, re0, nglob
    PetscErrorCode :: ierr

    ok = .false.
    gcomm = comm; gme = my_id
    call gmg_register_events()
    call PetscLogEventBegin(gev_prolong, ierr)
    gopt = opts
    call opt_real("ring_aspect", gopt%ring_aspect)
    semi_r = max(gopt%semi_r, -1)
    call opt_int("semi_r", semi_r)
    eps_s = [1, -1, -1, 1]
    eps_t = [1, 1, -1, -1]

    !--- gate 1: the npnew*(i-1)+j polar layout; an X-point grid takes its
    !--- own builder (one hierarchy over core, SOL and private region)
    xgrid = .false.
    if (node_list%n_nodes /= n_flux * n_tht) then
      if (xpatch_is_xgrid()) then
        call PetscLogEventEnd(gev_prolong, ierr)
        call gmg_build_xpoint_prolongations(Aref, comm, my_id, n_fields, ok, gopt)
        return
      endif
      call fail("not a flux-surface grid (n_nodes /= n_flux*n_tht)")
      return
    endif
    do e = 1, element_list%n_elements
      do iv = 1, 4
        n = element_list%element(e)%vertex(iv)
        vi(iv) = (n - 1) / n_tht
        vj(iv) = mod(n - 1, n_tht)
      enddo
      if (.not. (vi(2) == vi(1) + 1 .and. vj(2) == vj(1) .and. &
                 vi(3) == vi(1) + 1 .and. vj(3) == mod(vj(1) + 1, n_tht) .and. &
                 vi(4) == vi(1)     .and. vj(4) == mod(vj(1) + 1, n_tht))) then
        call fail("element vertices are not the structured (i,j) layout")
        return
      endif
    enddo

    !--- gate 2: per-node orientation o = +-1, consistent across elements.
    !--- fix_axis_nodes zeroes the angular (t, st) size factors of the axis
    !--- nodes (grid_flux_surface.f90:644, mod_import_gvec.f90:471) and pins
    !--- those DOFs (fix_nodes_on_axis): no basis function uses them, so they
    !--- get o = 0 -- an INACTIVE DOF that P never writes. The coarse axis
    !--- angular slope is dropped anyway (header, THE AXIS).
    allocate(o(node_list%n_nodes, 0:3), oset(node_list%n_nodes, 0:3))
    o = 0; oset = .false.; n_inact = 0
    do e = 1, element_list%n_elements
      do iv = 1, 4
        n = element_list%element(e)%vertex(iv)
        cand = nint([element_list%element(e)%size(iv, 1), &
                     eps_s(iv) * element_list%element(e)%size(iv, 2), &
                     eps_t(iv) * element_list%element(e)%size(iv, 3), &
                     eps_s(iv) * eps_t(iv) * element_list%element(e)%size(iv, 4)])
        do k = 0, 3
          if (abs(cand(k + 1)) /= 1) then
            if (.not. (cand(k + 1) == 0 .and. k >= 2 .and. node_list%node(n)%axis_node)) then
              call fail("element size factor is not +-1")
              return
            endif
          endif
          if (.not. oset(n, k)) then
            o(n, k) = cand(k + 1); oset(n, k) = .true.
            if (cand(k + 1) == 0) n_inact = n_inact + 1
          else if (o(n, k) /= cand(k + 1)) then
            call fail("inconsistent DOF orientation between elements")
            return
          endif
        enddo
      enddo
    enddo
    if (my_id == 0 .and. n_inact > 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, &
      ": ", n_inact, " axis DOFs inactive (fix_axis_nodes), P leaves them out"

    !--- cell aspect r dtheta/dr (> 1: long in theta) of every cell off the
    !--- axis, ring by ring, and sorted for the medians
    allocate(asp(n_tht, n_flux - 2))
    do i = 1, n_flux - 2
      do j = 0, n_tht - 1
        asp(j + 1, i) = cell_aspect(i, j)
      enddo
    enddo
    asp_sorted = reshape(asp, [size(asp)])
    call sort_real(asp_sorted)

    !--- coarse levels. Radial semi-coarsening (semi_r): on cells long in theta
    !--- the couplings are strongest radially, which the ring blocks do not
    !--- smooth and a full-coarsening coarse grid cannot take; levels that
    !--- coarsen in I alone can.
    if (semi_r < 0) then
      block
        real*8 :: a_
        a_ = asp_sorted((size(asp_sorted) + 1) / 2)
        semi_r = 0
        do while (a_ > SEMI_ASPECT .and. semi_r < MAX_LEV - 1)
          semi_r = semi_r + 1; a_ = a_ / 2
        enddo
        if (my_id == 0) write(*,'(A,I0,A,I0,A,F6.2,A)') "[Physics PC]   GMG", cur_inst, &
          ": radial semi-coarsening on levels 1..", semi_r, " (median r dtheta/dr ", &
          asp_sorted((size(asp_sorted) + 1) / 2), ")"
      end block
    endif
    allocate(lv(1:MAX_LEV - 1))
    ni = n_flux; nj = n_tht; nlev = 1
    nlev_cap = MAX_LEV_FULL
    if (semi_r > 0) nlev_cap = MAX_LEV
    call opt_int("nlev", nlev_cap)
    do g = 1, min(MAX_LEV, max(nlev_cap, 2)) - 1
      if (mod(ni - 1, 2) /= 0) exit
      if (g > semi_r .and. (mod(nj, 2) /= 0 .or. nj < 4)) exit
      ni = (ni - 1) / 2 + 1
      if (g > semi_r) nj = nj / 2
      call make_level(lv(g), ni, nj, gopt%bnd_drop > 0)
      nlev = nlev + 1
    enddo
    if (nlev < 2) then
      call fail("mesh cannot be coarsened (need (n_flux-1) even and n_tht divisible by 4)")
      return
    endif

    n_idx = 0
    do n = 1, node_list%n_nodes
      n_idx = max(n_idx, maxval(node_list%node(n)%index(1:n_degrees)))
    enddo

    !--- parallel layouts. Fine: read off Aref (rank r owns packed rows
    !--- [rstarts(r), rstarts(r+1)), n_fields x n_tor rows per JOREK index).
    !--- Coarse: DOF (I,J,k) goes to the owner of the fine node (I 2^g, J 2^g).
    call MPI_Comm_rank(comm, me, mpierr)
    call MPI_Comm_size(comm, np, mpierr)
    call MatGetOwnershipRange(Aref, rs0, re0, ierr)
    call MatGetSize(Aref, nglob, PETSC_NULL_INTEGER, ierr)
    if (nglob /= int(n_fields, 8) * n_idx * pc_ntor) then
      call fail("operator size /= n_fields * n_index * n_tor")
      return
    endif
    allocate(rstarts(0:np))
    call MPI_Allgather(int(rs0), 1, MPI_INTEGER, rstarts, 1, MPI_INTEGER, comm, mpierr)
    rstarts(np) = int(nglob)
    allocate(ly(0:nlev - 1))
    ly(0)%n = n_idx
    allocate(ly(0)%own(0:n_idx - 1), ly(0)%lpos(0:n_idx - 1), ly(0)%nl(0:np - 1), ly(0)%ps(0:np))
    do r = 0, np - 1
      if (mod(rstarts(r + 1) - rstarts(r), n_fields * pc_ntor) /= 0) then
        call fail("a rank's rows are not whole (field x toroidal slot) groups")
        return
      endif
      ly(0)%nl(r) = (rstarts(r + 1) - rstarts(r)) / (n_fields * pc_ntor)
      ly(0)%ps(r) = rstarts(r)
    enddo
    ly(0)%ps(np) = rstarts(np)
    r = 0; first = 0
    do idx = 0, n_idx - 1
      do while (idx >= first + ly(0)%nl(r))
        first = first + ly(0)%nl(r); r = r + 1
      enddo
      ly(0)%own(idx) = r; ly(0)%lpos(idx) = idx - first
    enddo
    do g = 1, nlev - 1
      ly(g)%n = lv(g)%n
      allocate(ly(g)%own(0:lv(g)%n - 1), ly(g)%lpos(0:lv(g)%n - 1), ly(g)%nl(0:np - 1), ly(g)%ps(0:np))
      ly(g)%own = -1
      do ci = 0, lv(g)%ni - 1
        do cj = 0, lv(g)%nj - 1
          n = (ci * 2**g) * n_tht + cj * (n_tht / lv(g)%nj) + 1
          own_ = ly(0)%own(node_list%node(n)%index(1) - 1)
          do k = 0, 3
            dd = lv(g)%d(ci, cj, k)
            if (dd < 0) cycle
            if (ly(g)%own(dd) < 0) ly(g)%own(dd) = own_     ! shared axis value: first visit
          enddo
        enddo
      enddo
      ! crebal 1: whole rings (ci) per rank, as on the X-point grids (off by
      ! default, see the X-point builder)
      block
        integer, allocatable :: wr(:), pr(:), own_o(:), rng(:)
        integer :: q_
        q_ = 0
        call opt_int("crebal", q_)
        if (q_ > 0 .and. np > 1) then
          allocate(wr(lv(g)%ni), pr(lv(g)%ni), rng(0:lv(g)%n - 1))
          rng = -1
          do ci = 0, lv(g)%ni - 1
            do cj = 0, lv(g)%nj - 1
              do k = 0, 3
                dd = lv(g)%d(ci, cj, k)
                if (dd < 0) cycle
                if (rng(dd) < 0) rng(dd) = ci
              enddo
            enddo
          enddo
          wr = 0
          do dd = 0, lv(g)%n - 1
            wr(rng(dd) + 1) = wr(rng(dd) + 1) + 1
          enddo
          call ring_parts(wr, np, pr)
          own_o = ly(g)%own
          do dd = 0, lv(g)%n - 1
            ly(g)%own(dd) = pr(rng(dd) + 1)
          enddo
          call crebal_pick(g, np, rng, own_o, ly(g)%own)
        endif
      end block
      ly(g)%nl = 0
      do dd = 0, lv(g)%n - 1
        ly(g)%lpos(dd) = ly(g)%nl(ly(g)%own(dd))
        ly(g)%nl(ly(g)%own(dd)) = ly(g)%nl(ly(g)%own(dd)) + 1
      enddo
      ly(g)%ps(0) = 0
      do r = 0, np - 1
        ly(g)%ps(r + 1) = ly(g)%ps(r) + n_fields * pc_ntor * ly(g)%nl(r)
      enddo
    enddo

    !--- P(1): level 1 canonical -> JOREK index space
    allocate(rs(MAX_ENT * 4 * node_list%n_nodes), cs(MAX_ENT * 4 * node_list%n_nodes), &
             ws(MAX_ENT * 4 * node_list%n_nodes), done(n_idx))
    done = .false.; ns = 0
    do n = 1, node_list%n_nodes
      i = (n - 1) / n_tht; j = mod(n - 1, n_tht)
      do k = 0, 3
        idx = node_list%node(n)%index(k + 1) - 1
        if (done(idx + 1)) cycle             ! the shared axis value: one row
        done(idx + 1) = .true.
        if (o(n, k) == 0) cycle              ! inactive (fixed) axis DOF: empty row
        call interp_row(i, j, k, lv(1), lv(1)%nj == n_tht, cols, w, nent)
        rs(ns + 1:ns + nent) = idx
        cs(ns + 1:ns + nent) = cols(1:nent)
        ws(ns + 1:ns + nent) = o(n, k) * w(1:nent)
        ns = ns + nent
      enddo
    enddo
    call make_expanded(gP(1), 0, 1, rs(1:ns), cs(1:ns), ws(1:ns))
    deallocate(rs, cs, ws, done)

    !--- P(g), g >= 2: level g canonical -> level g-1 canonical
    do g = 2, nlev - 1
      nrow_s = lv(g - 1)%n
      allocate(rs(MAX_ENT * nrow_s), cs(MAX_ENT * nrow_s), ws(MAX_ENT * nrow_s))
      ns = 0
      do i = 0, lv(g - 1)%ni - 1
        do j = 0, lv(g - 1)%nj - 1
          do k = 0, 3
            idx = lv(g - 1)%d(i, j, k)
            if (idx < 0) cycle
            if (i == 0 .and. k == 0 .and. j > 0) cycle   ! shared axis value
            call interp_row(i, j, k, lv(g), lv(g)%nj == lv(g - 1)%nj, cols, w, nent)
            rs(ns + 1:ns + nent) = idx
            cs(ns + 1:ns + nent) = cols(1:nent)
            ws(ns + 1:ns + nent) = w(1:nent)
            ns = ns + nent
          enddo
        enddo
      enddo
      call make_expanded(gP(g), g - 1, g, rs(1:ns), cs(1:ns), ws(1:ns))
      deallocate(rs, cs, ws)
    enddo

    !--- packed LOCAL row -> (node, harmonic) maps for the block smoothers,
    !--- fine level (fine_node/fine_harm) and every coarse level
    nf_s = n_fields
    nth0 = n_tht
    nloc = n_fields * pc_ntor * ly(0)%nl(me)
    allocate(fine_node(nloc), fine_harm(nloc))
    fine_node = -1
    do n = 1, node_list%n_nodes
      do k = 1, 4
        idx = node_list%node(n)%index(k) - 1
        if (ly(0)%own(idx) /= me) cycle
        do ff = 0, n_fields - 1
          do mm = 0, pc_ntor - 1
            lr = (ff * ly(0)%nl(me) + ly(0)%lpos(idx)) * pc_ntor + mm + 1
            fine_node(lr) = n - 1
            fine_harm(lr) = mm
          enddo
        enddo
      enddo
    enddo
    ! rank-local tests: every rank must refuse together, or the others run on
    ! into the collectives below
    if (any_rank(any(fine_node < 0))) then
      call fail("row -> node map does not cover every row")
      return
    endif
    do g = 1, nlev - 1
      nloc = n_fields * pc_ntor * ly(g)%nl(me)
      allocate(lv(g)%rnode(nloc), lv(g)%rharm(nloc))
      lv(g)%rnode = -1
      do ci = 0, lv(g)%ni - 1
        do cj = 0, lv(g)%nj - 1
          do k = 0, 3
            dd = lv(g)%d(ci, cj, k)
            if (dd < 0) cycle
            if (ly(g)%own(dd) /= me) cycle
            do ff = 0, n_fields - 1
              do mm = 0, pc_ntor - 1
                lr = (ff * ly(g)%nl(me) + ly(g)%lpos(dd)) * pc_ntor + mm + 1
                lv(g)%rnode(lr) = ci * lv(g)%nj + cj
                lv(g)%rharm(lr) = mm
              enddo
            enddo
          enddo
        enddo
      enddo
      if (any_rank(any(lv(g)%rnode < 0))) then
        call fail("coarse row -> node map does not cover every row")
        return
      endif
    enddo
    if (my_id == 0 .and. np > 1) then
      write(*,'(A,I0,A)', advance="no") "[Physics PC]   GMG parallel layout on ", np, " ranks, rows/rank min-max per level:"
      do g = 0, nlev - 1
        write(*,'(A,I0,A,I0)', advance="no") " ", n_fields * pc_ntor * minval(ly(g)%nl), "-", &
                                             n_fields * pc_ntor * maxval(ly(g)%nl)
      enddo
      write(*,*)
    endif
    deallocate(ly, rstarts)
    call move_alloc(lv, glv)

    !--- the switch ring I_s of smoother 8 / axis_rings -1: the first ring
    !--- whose median aspect reaches ring_aspect (GMGPolar's criterion); ring
    !--- ii holds the cells between flux surfaces ii and ii+1
    block
      real*8 :: rm(n_flux - 2), row(n_tht)
      integer :: ii, na
      na = size(asp_sorted)
      if (my_id == 0) write(*,'(A,F8.2,A,F8.2,A,F8.2)') &
        "[Physics PC]   GMG grid cells, rdtheta/dr: median ", asp_sorted((na + 1) / 2), &
        ", max ", asp_sorted(na), ", min ", asp_sorted(1)
      do ii = 1, n_flux - 2
        row = asp(:, ii)
        call sort_real(row)
        rm(ii) = row((n_tht + 1) / 2)
      enddo
      ring_is = n_flux - 1
      do ii = 1, n_flux - 2
        if (rm(ii) >= gopt%ring_aspect) then
          ring_is = ii; exit
        endif
      enddo
      if (my_id == 0) then
        write(*,'(A,F5.2,A,I0,A)', advance="no") "[Physics PC]   GMG ring medians rdtheta/dr (switch at ", &
          gopt%ring_aspect, ": rings I < ", ring_is, "):"
        do ii = 1, min(n_flux - 2, ring_is + 2)
          write(*,'(A,F6.2)', advance="no") " ", rm(ii)
        enddo
        write(*,*)
      endif
    end block
    deallocate(asp, asp_sorted)

    if (my_id == 0) then
      write(*,'(A,I0,A)', advance="no") "[Physics PC]   GMG hierarchy: ", nlev, " levels, grid "
      write(*,'(I0,A,I0)', advance="no") n_flux, "x", n_tht
      do g = 1, nlev - 1
        write(*,'(A,I0,A,I0)', advance="no") " -> ", glv(g)%ni, "x", glv(g)%nj
      enddo
      write(*,*)
    endif
    p_ready = .true.
    ok = .true.
    call PetscLogEventEnd(gev_prolong, ierr)

  contains

    !> Aspect r dtheta/dr of cell (ring ii, angle jj) from its vertex
    !! positions. On a stellarator grid R, Z depend on phi, and node%x(1,...)
    !! is only the phi-averaged harmonic, rounder than any real cross-section:
    !! take the median over the n_plane toroidal planes instead.
    real*8 function cell_aspect(ii, jj)
#ifdef STELLARATOR_MODEL
      use basis_at_gaussian, only: HZ_coord
      use mod_parameters,    only: n_plane, n_coord_tor
#endif
      integer, intent(in) :: ii, jj
      integer :: na_, nb_, nc_
      real*8  :: xa_(2), xb_(2), xc_(2)
#ifdef STELLARATOR_MODEL
      integer :: mp_, d_
      real*8  :: r_(n_plane)
#endif
      na_ = ii * n_tht + jj + 1
      nb_ = (ii + 1) * n_tht + jj + 1
      nc_ = ii * n_tht + mod(jj + 1, n_tht) + 1
#ifdef STELLARATOR_MODEL
      do mp_ = 1, n_plane
        do d_ = 1, 2
          xa_(d_) = sum(HZ_coord(1:n_coord_tor, mp_) * node_list%node(na_)%x(1:n_coord_tor, 1, d_))
          xb_(d_) = sum(HZ_coord(1:n_coord_tor, mp_) * node_list%node(nb_)%x(1:n_coord_tor, 1, d_))
          xc_(d_) = sum(HZ_coord(1:n_coord_tor, mp_) * node_list%node(nc_)%x(1:n_coord_tor, 1, d_))
        enddo
        r_(mp_) = norm2(xc_ - xa_) / max(norm2(xb_ - xa_), 1.d-300)
      enddo
      call sort_real(r_)
      cell_aspect = r_((n_plane + 1) / 2)
#else
      xa_ = node_list%node(na_)%x(1, 1, 1:2)
      xb_ = node_list%node(nb_)%x(1, 1, 1:2)
      xc_ = node_list%node(nc_)%x(1, 1, 1:2)
      cell_aspect = norm2(xc_ - xa_) / max(norm2(xb_ - xa_), 1.d-300)
#endif
    end function cell_aspect

    subroutine fail(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   GMG hierarchy REFUSED: ", msg
      call PetscLogEventEnd(gev_prolong, ierr)
    end subroutine fail

    logical function any_rank(bad)
      logical, intent(in) :: bad
      any_rank = bad
      call MPI_Allreduce(MPI_IN_PLACE, any_rank, 1, MPI_LOGICAL, MPI_LOR, comm, mpierr)
    end function any_rank

    !> Packed global row of (field f, scalar DOF d, slot m) on layout level L.
    integer function prow(L, f, d, m)
      integer, intent(in) :: L, f, d, m
      integer :: ow
      ow = ly(L)%own(d)
      prow = ly(L)%ps(ow) + (f * ly(L)%nl(ow) + ly(L)%lpos(d)) * pc_ntor + m
    end function prow

    !> Expand a scalar (one field, one harmonic) P into the packed operators'
    !! layouts: rows on level lr_, columns on level lc_. Each rank inserts the
    !! rows it owns; on one rank row = f*nrow*n_tor + r*n_tor + m as before.
    subroutine make_expanded(P, lr_, lc_, r_, c, v)
      Mat, intent(out)    :: P
      integer, intent(in) :: lr_, lc_, r_(:), c(:)
      real*8, intent(in)  :: v(:)
      PetscInt :: nrl, ncl, nrg, ncg, row, col
      PetscErrorCode :: ierr_
      integer :: f, m, q

      nrl = n_fields * pc_ntor * ly(lr_)%nl(me); ncl = n_fields * pc_ntor * ly(lc_)%nl(me)
      nrg = n_fields * pc_ntor * ly(lr_)%n;      ncg = n_fields * pc_ntor * ly(lc_)%n
      call MatCreate(comm, P, ierr_)
      call MatSetSizes(P, nrl, ncl, nrg, ncg, ierr_)
      call MatSetType(P, MATMPIAIJ, ierr_)       ! the packed pairs are MPIAIJ even on one rank
      call MatSeqAIJSetPreallocation(P, MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr_)
      call MatMPIAIJSetPreallocation(P, MAX_ENT, PETSC_NULL_INTEGER_ARRAY, &
                                     MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr_)
      do q = 1, size(r_)
        if (ly(lr_)%own(r_(q)) /= me) cycle
        do f = 0, n_fields - 1
          do m = 0, pc_ntor - 1
            row = prow(lr_, f, r_(q), m)
            col = prow(lc_, f, c(q), m)
            call MatSetValue(P, row, col, v(q), ADD_VALUES, ierr_)
          enddo
        enddo
      enddo
      call MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY, ierr_)
      call MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY, ierr_)
    end subroutine make_expanded

  end subroutine gmg_build_prolongations

  !> X-POINT GRIDS (lower single null), the counterpart of
  !! gmg_build_prolongations. The grid is three logically rectangular patches
  !! (mod_petsc_pc_xpatch: core, SOL, private region) glued at the
  !! separatrix; the X-point is four co-located nodes sharing 9 DOFs. ONE
  !! hierarchy covers all three -- no interface, no Schwarz:
  !!  - every level coarsens every patch at once and stays conforming across
  !!    the separatrices. Along the flux surface the core's X-to-X stretch
  !!    pairs from both X ends (a single cell in the middle when odd), the
  !!    SOL's middle stretch takes the core's cuts through their shared
  !!    separatrix nodes, the legs pair from the X-point (a single at the
  !!    target when odd) and the private region takes the SOL legs' cuts.
  !!    Across, each patch pairs from its separatrix side. Levels 1..semi_r
  !!    coarsen across only, one count for all patches.
  !!  - P: JOREK's element as a Bezier net (element%size), subdivided by de
  !!    Casteljau at the split the fine sizes imply, element by element; a
  !!    fine DOF seen by several elements (the separatrix, the X-point) gets
  !!    their average. Coarse DOFs are fine node DOFs, so the X-point's 9
  !!    carry through every level. The flux-aligned grid is not exactly the
  !!    subdivision of a coarser one, so the hierarchy is nested only
  !!    approximately; the offline probe (util/physics_pc_scaling/
  !!    xpoint_probe, xglobal.py) found smooth fields reproduced to ~1e-3
  !!    near the X-point as away from it.
  !!  - the smoothers see a VIRTUAL lattice (xrg, fine_col / rcol): ring I is
  !!    one flux line -- the core rings from the axis (I = 0) to the
  !!    separatrix, then the private rows from the wall, then the SOL rows
  !!    from the separatrix, last the X-point's DOFs as a block of their own
  !!    -- J the position along it, and each row's radial column follows the
  !!    SOL's columns through the shared separatrix nodes, so radial lines
  !!    run from the axis to the wall and across the private region. A DOF
  !!    takes its place from the first of core, SOL, private that has it.
  subroutine gmg_build_xpoint_prolongations(Aref, comm, my_id, n_fields, ok, opts)
    use nodes_elements,      only: node_list, element_list
    use mod_parameters,      only: n_degrees
    use mod_petsc_pc_harm,   only: pc_ntor
    use mod_petsc_pc_xpatch, only: xpatch_t, xpatch_build, xpatch_report, XP_NPATCH, XP_CORE, &
                                   XP_SOL, XP_PRIV
    Mat, intent(in)      :: Aref
    integer, intent(in)  :: comm, my_id, n_fields
    logical, intent(out) :: ok
    type(gmg_opts_t), intent(in) :: opts

    type :: xpl_t                                !< one patch on one level
      integer :: na = 0, nb = 0, bsep = 0        !< bsep: the separatrix's node row (0 or nb)
      integer :: nx = 0, xa(2) = -1              !< node columns of the X-point (ascending)
      integer, allocatable :: vt(:,:,:)          !< (4, 0:na-1, 0:nb-1) node of each slot
      real*8,  allocatable :: sz(:,:,:,:)        !< (3, 4, 0:na-1, 0:nb-1) s2, s3, s4
      !> coarse column / row -> its first fine one and width (1, 2); the
      !! de Casteljau splits of each coarse cell across (ls) and along (lt)
      integer, allocatable :: a0(:), wa(:), b0(:), wb(:)
      real*8,  allocatable :: ls(:,:), lt(:,:)
    end type xpl_t
    type :: xlv_t                                !< one level
      integer :: n = 0, nring = 0, ncol = 0
      type(xpl_t) :: p(XP_NPATCH)
      integer, allocatable :: dof(:)             !< level DOF 1..n -> JOREK index (1-based)
      integer, allocatable :: lmap(:)            !< JOREK index -> level DOF, 0 = absent
      integer, allocatable :: rI(:), rJ(:), rC(:) !< level DOF -> ring, position along it, column
    end type xlv_t

    type(xpatch_t) :: xp(XP_NPATCH)
    type(xlv_t), allocatable :: xl(:)
    type(lay_t), allocatable :: ly(:)
    type(lvl_t), allocatable :: lv(:)
    integer, allocatable :: rstarts(:), rr(:), cc(:), cnt(:)
    real*8,  allocatable :: ww(:)
    logical, allocatable :: drop(:), dropc(:), xdof(:)
    integer :: me, np, mpierr, r, first, n_idx, g, c, k, kk, n, idx, ntrip, nloc, lr, ff, mm, q
    integer :: semi, nlev_cap
    character(len=256) :: msg
    character(len=64) :: shape
    PetscInt :: rs0, re0, nglob
    PetscErrorCode :: ierr

    ok = .false.
    gcomm = comm; gme = my_id
    call MPI_Comm_rank(comm, me, mpierr)
    call MPI_Comm_size(comm, np, mpierr)
    call PetscLogEventBegin(gev_prolong, ierr)

    !--- the patches
    call xpatch_build(xp, 0, ok, msg)
    if (.not. ok) then
      call fail(trim(msg)); return
    endif
    ok = .false.
    call xpatch_report(xp, my_id)
    do c = 1, XP_NPATCH
      if (xp(c)%nxp < 1 .or. any(xp(c)%xpI(1:xp(c)%nxp) /= xp(c)%xpI(1)) .or. &
          (xp(c)%xpI(1) /= 0 .and. xp(c)%xpI(1) /= xp(c)%ni - 1)) then
        call fail("patch "//trim(xp(c)%name)//": the X-point is not on one edge row"); return
      endif
    enddo
    if (.not. xp(XP_CORE)%axis .or. xp(XP_CORE)%xpI(1) /= xp(XP_CORE)%ni - 1) then
      call fail("core patch: the axis is not on row 0 opposite the separatrix"); return
    endif

    n_idx = 0
    do n = 1, node_list%n_nodes
      n_idx = max(n_idx, maxval(node_list%node(n)%index(1:n_degrees)))
    enddo

    !--- the fine layout, read off Aref (as gmg_build_prolongations)
    call MatGetOwnershipRange(Aref, rs0, re0, ierr)
    call MatGetSize(Aref, nglob, PETSC_NULL_INTEGER, ierr)
    if (nglob /= int(n_fields, 8) * n_idx * pc_ntor) then
      call fail("operator size /= n_fields * n_index * n_tor"); return
    endif
    allocate(rstarts(0:np))
    call MPI_Allgather(int(rs0), 1, MPI_INTEGER, rstarts, 1, MPI_INTEGER, comm, mpierr)
    rstarts(np) = int(nglob)
    allocate(ly(0:MAX_LEV - 1))
    ly(0)%n = n_idx
    allocate(ly(0)%own(0:n_idx - 1), ly(0)%lpos(0:n_idx - 1), ly(0)%nl(0:np - 1), ly(0)%ps(0:np))
    do r = 0, np - 1
      if (mod(rstarts(r + 1) - rstarts(r), n_fields * pc_ntor) /= 0) then
        call fail("a rank's rows are not whole (field x toroidal slot) groups"); return
      endif
      ly(0)%nl(r) = (rstarts(r + 1) - rstarts(r)) / (n_fields * pc_ntor)
      ly(0)%ps(r) = rstarts(r)
    enddo
    ly(0)%ps(np) = rstarts(np)
    r = 0; first = 0
    do idx = 0, n_idx - 1
      do while (idx >= first + ly(0)%nl(r))
        first = first + ly(0)%nl(r); r = r + 1
      enddo
      ly(0)%own(idx) = r; ly(0)%lpos(idx) = idx - first
    enddo

    !--- Dirichlet DOFs (rows of field 1, slot 0 with only a diagonal): empty
    !--- rows of P, and out of every coarse space; the axis's angular slope
    !--- (index 3 of an axis node) too, as on the polar grid (THE AXIS)
    allocate(drop(n_idx), dropc(n_idx))
    drop = .false.
    block
      PetscInt :: ncols, row
      PetscInt, pointer :: cols(:)
      PetscScalar, pointer :: vals(:)
      integer :: jc
      logical :: diag_only
      do idx = 0, n_idx - 1
        if (ly(0)%own(idx) /= me) cycle
        row = ly(0)%ps(me) + ly(0)%lpos(idx) * pc_ntor
        call MatGetRow(Aref, row, ncols, cols, vals, ierr)
        diag_only = .true.
        do jc = 1, int(ncols)
          if (cols(jc) /= row .and. vals(jc) /= 0.0d0) diag_only = .false.
        enddo
        call MatRestoreRow(Aref, row, ncols, cols, vals, ierr)
        drop(idx + 1) = diag_only
      enddo
    end block
    call MPI_Allreduce(MPI_IN_PLACE, drop, n_idx, MPI_LOGICAL, MPI_LOR, comm, mpierr)
    dropc = drop
    do n = 1, node_list%n_nodes
      if (node_list%node(n)%axis_node) dropc(node_list%node(n)%index(3)) = .true.
    enddo

    !--- level 0: the patches' elements; the X-point's DOFs
    allocate(xl(0:MAX_LEV - 1), xdof(n_idx))
    xdof = .false.
    do c = 1, XP_NPATCH
      associate (L => xl(0)%p(c), P => xp(c))
      L%na = P%na; L%nb = P%nb
      L%bsep = P%xpI(1)
      L%nx = 0
      do q = 1, P%nxp
        if (L%nx < 2 .and. .not. any(L%xa(1:L%nx) == P%xpJ(q))) then
          L%nx = L%nx + 1; L%xa(L%nx) = P%xpJ(q)
        endif
      enddo
      if (L%nx == 2 .and. L%xa(1) > L%xa(2)) L%xa = L%xa([2, 1])
      allocate(L%vt(4, 0:P%na - 1, 0:P%nb - 1), L%sz(3, 4, 0:P%na - 1, 0:P%nb - 1))
      do k = 0, P%nb - 1
        do kk = 0, P%na - 1
          do q = 1, 4
            L%vt(q, kk, k) = element_list%element(P%eat(kk, k))%vertex(q)
            L%sz(:, q, kk, k) = element_list%element(P%eat(kk, k))%size(q, 2:4)
            if (any(P%xpI(1:P%nxp) == k + SLOT_B_(q) .and. &
                    P%xpJ(1:P%nxp) == modulo(kk + SLOT_A_(q), merge(P%na, P%na + 1, P%periodic)))) &
              xdof(node_list%node(L%vt(q, kk, k))%index(1:4)) = .true.
          enddo
        enddo
      enddo
      end associate
    enddo
    if (count(xdof) /= 9) then
      write(msg, '(A,I0,A)') "the X-point holds ", count(xdof), " DOFs, not 9"
      call fail(trim(msg)); return
    endif
    if (xl(0)%p(XP_CORE)%nx /= 1 .or. xl(0)%p(XP_CORE)%xa(1) /= 0 .or. xl(0)%p(XP_SOL)%nx /= 2 &
        .or. xl(0)%p(XP_PRIV)%nx /= 1) then
      call fail("unexpected X-point columns (core: one at J = 0, SOL: two, private: one)"); return
    endif

    !--- radial-only levels: one count for every patch (semi_r < 0: by the
    !--- median cell aspect over all patches, the core's axis row excluded)
    semi = max(opts%semi_r, -1)
    call opt_int("semi_r", semi)
    if (semi < 0) then
      block
        real*8 :: a_
        ! down to half the polar grid's SEMI_ASPECT: one radial level more
        ! (21x33 deck, pair_w at tstep 10: 2 levels 4.3 inner its, 3 levels 3.2,
        ! with GMRES on every level)
        a_ = median_aspect(-1)
        semi = 0
        do while (a_ > 0.5d0 * SEMI_ASPECT .and. semi < MAX_LEV - 1)
          semi = semi + 1; a_ = a_ / 2
        enddo
        if (my_id == 0) write(*,'(A,I0,A,I0,A,F6.2,A)') "[Physics PC]   GMG", cur_inst, &
          ": radial semi-coarsening on levels 1..", semi, " (median r dtheta/dr ", median_aspect(-1), ")"
      end block
    endif
    semi_r = semi
    nlev_cap = MAX_LEV_FULL
    if (semi > 0) nlev_cap = MAX_LEV
    call opt_int("nlev", nlev_cap)

    !--- the levels
    call level_dofs(0)
    if (len_trim(msg) > 0) then
      call fail(trim(msg)); return
    endif
    !--- several ranks: the hierarchy's own layout, ring-major (see xperm)
    xperm = .false.
    q = 1
    call opt_int("xperm", q)
    if (q > 0 .and. np > 1) then
      block
        integer, allocatable :: ord(:), okey(:), own0(:), lpos0(:), nl0(:), ps0(:)
        PetscInt, allocatable :: ix(:)
        integer :: nj1, rk_, fill_
        own0 = ly(0)%own; lpos0 = ly(0)%lpos
        nj1 = maxval(xl(0)%rJ(1:xl(0)%n)) + 1
        allocate(ord(n_idx), okey(n_idx))
        do idx = 0, n_idx - 1
          ord(idx + 1) = idx
          k = xl(0)%lmap(idx + 1)
          okey(idx + 1) = huge(1)                 ! outside the patches (refused below)
          if (k > 0) okey(idx + 1) = xl(0)%rI(k) * nj1 + xl(0)%rJ(k)
        enddo
        call sort_rows_by_key(ord, okey)          ! stable: a node's DOFs stay together
        nl0 = ly(0)%nl; ps0 = ly(0)%ps
        ! xsplit 1: the rank boundaries on ring boundaries, so that no ring
        ! block is cut and none is solved twice (no ring ghosts). Off by
        ! default: on tok (xg65 / xg129, 16-64 ranks) the V-cycle gained only
        ! at xg65 64 ranks (-20%), lost 12% at xg129 64x4, and PtAP rose
        ! 1.1-1.6x from the busiest rank's 1.15-1.39x rows
        q = 0
        call opt_int("xsplit", q)
        if (q > 0) call ring_split(ord, okey / nj1)
        rk_ = 0; fill_ = 0
        do k = 1, n_idx
          do while (fill_ >= ly(0)%nl(rk_))
            rk_ = rk_ + 1; fill_ = 0
          enddo
          ly(0)%own(ord(k)) = rk_; ly(0)%lpos(ord(k)) = fill_
          fill_ = fill_ + 1
        enddo
        nloc = n_fields * pc_ntor * ly(0)%nl(me)
        allocate(ix(nloc))
        do idx = 0, n_idx - 1
          if (ly(0)%own(idx) /= me) cycle
          do ff = 0, n_fields - 1
            do mm = 0, pc_ntor - 1
              lr = (ff * ly(0)%nl(me) + ly(0)%lpos(idx)) * pc_ntor + mm + 1
              ix(lr) = ps0(own0(idx)) + (ff * nl0(own0(idx)) + lpos0(idx)) * pc_ntor + mm
            enddo
          enddo
        enddo
        call ISCreateGeneral(comm, int(nloc, kind(rs0)), ix, PETSC_COPY_VALUES, xp_is, ierr)
      end block
      xperm = .true.
    endif

    nlev = 1
    shape = ""
    do g = 1, min(MAX_LEV, max(nlev_cap, 2)) - 1
      if (.not. coarsen(g - 1, g <= semi)) exit
      if (len_trim(msg) > 0) then
        call fail(trim(msg)); return
      endif
      call level_dofs(g)
      if (len_trim(msg) > 0) then
        call fail(trim(msg)); return
      endif
      nlev = nlev + 1
      write(shape(len_trim(shape) + 1:), '(A,A)') " ", merge("r", "b", g <= semi)
    enddo
    if (nlev < 2) then
      call fail("the X-point grid cannot be coarsened"); return
    endif

    !--- coarse layouts: a coarse DOF is a fine node DOF and goes with it
    do g = 1, nlev - 1
      ly(g)%n = xl(g)%n
      allocate(ly(g)%own(0:xl(g)%n - 1), ly(g)%lpos(0:xl(g)%n - 1), ly(g)%nl(0:np - 1), ly(g)%ps(0:np))
      ly(g)%nl = 0
      do k = 1, xl(g)%n
        ly(g)%own(k - 1) = ly(0)%own(xl(g)%dof(k) - 1)
        ly(g)%nl(ly(g)%own(k - 1)) = ly(g)%nl(ly(g)%own(k - 1)) + 1
      enddo
      ! crebal 1: whole rings per rank on this level, the busiest rank's
      ! share balanced (the inherited owners leave the ranks whose rings a
      ! radial level drops idle and the others with their full rings). Off
      ! by default: no V-cycle gain on tok, PtAP up to 1.3x (P across ranks)
      q = 0
      call opt_int("crebal", q)
      if (q > 0 .and. np > 1) then
        block
          integer, allocatable :: wr(:), pr(:), own_o(:)
          allocate(wr(maxval(xl(g)%rI(1:xl(g)%n)) + 1))
          allocate(pr(size(wr)))
          wr = 0
          do k = 1, xl(g)%n
            wr(xl(g)%rI(k) + 1) = wr(xl(g)%rI(k) + 1) + 1
          enddo
          call ring_parts(wr, np, pr)
          own_o = ly(g)%own
          do k = 1, xl(g)%n
            ly(g)%own(k - 1) = pr(xl(g)%rI(k) + 1)
          enddo
          call crebal_pick(g, np, xl(g)%rI(1:xl(g)%n), own_o, ly(g)%own)
        end block
      endif
      ly(g)%nl = 0
      do k = 1, xl(g)%n
        ly(g)%lpos(k - 1) = ly(g)%nl(ly(g)%own(k - 1))
        ly(g)%nl(ly(g)%own(k - 1)) = ly(g)%nl(ly(g)%own(k - 1)) + 1
      enddo
      ly(g)%ps(0) = 0
      do r = 0, np - 1
        ly(g)%ps(r + 1) = ly(g)%ps(r) + n_fields * pc_ntor * ly(g)%nl(r)
      enddo
    enddo

    !--- the prolongations, expanded to fields and slots
    do g = 1, nlev - 1
      call make_p(g)
      call make_expanded(gP(g), g - 1, g)
    enddo

    !--- the block maps: ring, position, column of every local row
    nf_s = n_fields
    nth0 = max(maxval(xrg(0)%nn), 1)
    nloc = n_fields * pc_ntor * ly(0)%nl(me)
    if (allocated(fine_node)) deallocate(fine_node, fine_harm)
    if (allocated(fine_col)) deallocate(fine_col)
    allocate(fine_node(nloc), fine_harm(nloc), fine_col(nloc))
    fine_node = -1
    do k = 1, xl(0)%n
      idx = xl(0)%dof(k) - 1
      if (ly(0)%own(idx) /= me) cycle
      do ff = 0, n_fields - 1
        do mm = 0, pc_ntor - 1
          lr = (ff * ly(0)%nl(me) + ly(0)%lpos(idx)) * pc_ntor + mm + 1
          fine_node(lr) = xl(0)%rI(k) * nth0 + xl(0)%rJ(k)
          fine_harm(lr) = mm
          fine_col(lr) = xl(0)%rC(k)
        enddo
      enddo
    enddo
    if (any_rank(any(fine_node < 0))) then
      call fail("row -> ring map does not cover every row (a DOF outside the patches)"); return
    endif
    allocate(lv(1:MAX_LEV - 1))
    do g = 1, nlev - 1
      lv(g)%ni = xl(g)%nring; lv(g)%nj = max(maxval(xrg(g)%nn), 1); lv(g)%n = xl(g)%n
      nloc = n_fields * pc_ntor * ly(g)%nl(me)
      allocate(lv(g)%rnode(nloc), lv(g)%rharm(nloc), lv(g)%rcol(nloc))
      lv(g)%rnode = -1
      do k = 1, xl(g)%n
        if (ly(g)%own(k - 1) /= me) cycle
        do ff = 0, n_fields - 1
          do mm = 0, pc_ntor - 1
            lr = (ff * ly(g)%nl(me) + ly(g)%lpos(k - 1)) * pc_ntor + mm + 1
            lv(g)%rnode(lr) = xl(g)%rI(k) * lv(g)%nj + xl(g)%rJ(k)
            lv(g)%rharm(lr) = mm
            lv(g)%rcol(lr) = xl(g)%rC(k)
          enddo
        enddo
      enddo
    enddo
    if (allocated(glv)) deallocate(glv)
    call move_alloc(lv, glv)

    !--- the switch ring of smoother 8 / axis_rings -1: the first core ring
    !--- whose median aspect reaches ring_aspect (core rings are I = 0.. from the axis)
    ring_is = xl(0)%p(XP_CORE)%nb
    do k = 1, xl(0)%p(XP_CORE)%nb - 1
      if (median_aspect(k) >= opts%ring_aspect) then
        ring_is = k; exit
      endif
    enddo

    if (my_id == 0) then
      write(*,'(A,I0,A,I0,A)', advance="no") "[Physics PC]   GMG", cur_inst, " X-point hierarchy: ", nlev, &
        " levels, elements core | SOL | private"
      do g = 0, nlev - 1
        write(*,'(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)', advance="no") "  ", xl(g)%p(1)%na, "x", xl(g)%p(1)%nb, &
          "|", xl(g)%p(2)%na, "x", xl(g)%p(2)%nb, "|", xl(g)%p(3)%na, "x", xl(g)%p(3)%nb
      enddo
      write(*,'(A,A,A,I0,A,I0,A,I0,A,I0)') " (", trim(adjustl(shape)), "), ", count(drop), &
        " Dirichlet DOFs dropped, rings ", xl(0)%nring, ", switch ring ", ring_is, ", coarsest DOFs ", &
        xl(nlev - 1)%n
    endif
    xgrid = .true.
    p_ready = .true.
    ok = .true.
    call PetscLogEventEnd(gev_prolong, ierr)

  contains

    subroutine fail(m_)
      character(len=*), intent(in) :: m_
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   GMG X-point hierarchy REFUSED: ", m_
      ok = .false.
      call PetscLogEventEnd(gev_prolong, ierr)
    end subroutine fail

    logical function any_rank(bad)
      logical, intent(in) :: bad
      any_rank = bad
      call MPI_Allreduce(MPI_IN_PLACE, any_rank, 1, MPI_LOGICAL, MPI_LOR, comm, mpierr)
    end function any_rank

    !> xsplit: new ly(0)%nl / ps, the ring-major order ord cut on ring
    !! boundaries (ring of DOF idx: rid(idx + 1)) into np contiguous parts
    !! with the least possible maximum, every rank at least one ring. Too
    !! few rings or a part above 1.5 x the even split: nl kept (equal counts).
    subroutine ring_split(ord, rid)
      integer, intent(in) :: ord(:), rid(:)
      integer, allocatable :: w(:), pr(:), nl_(:)
      integer :: nr, k, r_
      allocate(w(n_idx))
      w = 0; nr = 0
      do k = 1, n_idx
        if (k == 1) then
          nr = 1
        else if (rid(ord(k) + 1) /= rid(ord(k - 1) + 1)) then
          nr = nr + 1
        endif
        w(nr) = w(nr) + 1
      enddo
      if (nr < np) then
        if (me == 0) write(*,'(A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, ": xsplit off (", nr, &
          " rings on ", np, " ranks), equal counts"
        return
      endif
      allocate(pr(nr), nl_(0:np - 1))
      call ring_parts(w(1:nr), np, pr)
      nl_ = 0
      do k = 1, nr
        nl_(pr(k)) = nl_(pr(k)) + w(k)
      enddo
      if (maxval(nl_) > 1.5d0 * n_idx / np) then
        if (me == 0) write(*,'(A,I0,A,F5.2,A)') "[Physics PC]   GMG", cur_inst, &
          ": xsplit off (whole rings: max rank ", maxval(nl_) * np / dble(n_idx), " x even), equal counts"
        return
      endif
      ly(0)%nl = nl_
      ly(0)%ps(0) = 0
      do r_ = 0, np - 1
        ly(0)%ps(r_ + 1) = ly(0)%ps(r_) + n_fields * pc_ntor * nl_(r_)
      enddo
      if (me == 0) write(*,'(A,I0,A,I0,A,I0,A,F5.2,A)') "[Physics PC]   GMG", cur_inst, &
        ": xsplit, whole rings per rank (", nr, " rings on ", np, " ranks, max rank ", &
        maxval(nl_) * np / dble(n_idx), " x even)"
    end subroutine ring_split

    !> Median cell aspect along / across (|x(A+1) - x(A)| / |x(B+1) - x(B)|)
    !! of the fine core row b (b >= 1), or (b = -1) of every patch's cells
    !! but the core's axis row.
    real*8 function median_aspect(b)
      integer, intent(in) :: b
      real*8, allocatable :: ar(:)
      real*8 :: x1(2), x2(2), x4(2)
      integer :: cp, aa, bb, na_
      allocate(ar(sum([(xl(0)%p(cp)%na * xl(0)%p(cp)%nb, cp = 1, XP_NPATCH)])))
      na_ = 0
      do cp = 1, XP_NPATCH
        if (b >= 0 .and. cp /= XP_CORE) cycle
        do bb = 0, xl(0)%p(cp)%nb - 1
          if (b >= 0 .and. bb /= b) cycle
          if (cp == XP_CORE .and. bb == 0) cycle
          do aa = 0, xl(0)%p(cp)%na - 1
            x1 = node_list%node(xl(0)%p(cp)%vt(1, aa, bb))%x(1, 1, 1:2)
            x2 = node_list%node(xl(0)%p(cp)%vt(2, aa, bb))%x(1, 1, 1:2)
            x4 = node_list%node(xl(0)%p(cp)%vt(4, aa, bb))%x(1, 1, 1:2)
            na_ = na_ + 1
            ar(na_) = norm2(x4 - x1) / max(norm2(x2 - x1), 1.d-300)
          enddo
        enddo
      enddo
      if (na_ == 0) then
        median_aspect = 1.d0; return
      endif
      call sort_real(ar(1:na_))
      median_aspect = ar((na_ + 1) / 2)
    end function median_aspect

    !> Node of patch level L at node column jj of its separatrix row.
    integer function sepnode(L, jj)
      type(xpl_t), intent(in) :: L
      integer, intent(in) :: jj
      if (L%bsep == 0) then
        if (jj < L%na) then
          sepnode = L%vt(1, jj, 0)
        else
          sepnode = L%vt(4, L%na - 1, 0)
        endif
      else
        if (jj < L%na) then
          sepnode = L%vt(2, jj, L%nb - 1)
        else
          sepnode = L%vt(3, L%na - 1, L%nb - 1)
        endif
      endif
    end function sepnode

    !> Group sizes of a stretch of ln cells: pairs from the X-point end(s), a
    !! single at the far end (one X end) or in the middle (both) when odd.
    subroutine group_stretch(ln, xlo, xhi, gs, ngs)
      integer, intent(in)  :: ln
      logical, intent(in)  :: xlo, xhi
      integer, intent(out) :: gs(:), ngs
      integer :: npair, ii
      npair = ln / 2
      ngs = 0
      if (mod(ln, 2) == 0) then
        gs(1:npair) = 2; ngs = npair; return
      endif
      if (xlo .and. xhi) then
        do ii = 1, npair / 2
          ngs = ngs + 1; gs(ngs) = 2
        enddo
        ngs = ngs + 1; gs(ngs) = 1
        do ii = 1, npair - npair / 2
          ngs = ngs + 1; gs(ngs) = 2
        enddo
      else if (xhi) then
        ngs = 1; gs(1) = 1
        gs(2:npair + 1) = 2; ngs = npair + 1
      else
        gs(1:npair) = 2; gs(npair + 1) = 1; ngs = npair + 1
      endif
    end subroutine group_stretch

    !> Cut positions (node columns, ascending, 0 and na included) -> the
    !! coarse columns' first fine column and width; .false. if a gap > 2.
    logical function cuts_to_groups(cut, ncut, a0_, wa_)
      integer, intent(in) :: cut(:), ncut
      integer, allocatable, intent(out) :: a0_(:), wa_(:)
      integer :: ii
      allocate(a0_(0:ncut - 2), wa_(0:ncut - 2))
      cuts_to_groups = .true.
      do ii = 1, ncut - 1
        a0_(ii - 1) = cut(ii); wa_(ii - 1) = cut(ii + 1) - cut(ii)
        if (wa_(ii - 1) < 1 .or. wa_(ii - 1) > 2) cuts_to_groups = .false.
      enddo
    end function cuts_to_groups

    !> Level gf + 1 from level gf: .false. when nothing coarsens any more
    !! (msg set on an inconsistency).
    logical function coarsen(gf, radial_only)
      integer, intent(in) :: gf
      logical, intent(in) :: radial_only
      integer, allocatable :: cutc(:), cuts(:), cutp(:), gs(:)
      logical, allocatable :: isc(:)
      integer :: ncc, ncs, ncp, ngs, cp, ii, jj, s0, s1, nnode
      logical :: any2
      coarsen = .false.
      msg = ""
      associate (Fc => xl(gf)%p(XP_CORE), Fs => xl(gf)%p(XP_SOL), Fp => xl(gf)%p(XP_PRIV))
      nnode = node_list%n_nodes
      allocate(gs(max(Fc%na, Fs%na, Fp%na, Fc%nb, Fs%nb, Fp%nb) + 2))
      allocate(cutc(Fc%na + 1), cuts(Fs%na + 1), cutp(Fp%na + 1), isc(nnode))
      ! along-surface coarsening stops at 4 core cells (the polar grid's nj >= 4)
      if (.not. radial_only .and. Fc%na < 8) return
      if (radial_only) then
        ncc = Fc%na + 1; cutc(1:ncc) = [(ii, ii = 0, Fc%na)]
        ncs = Fs%na + 1; cuts(1:ncs) = [(ii, ii = 0, Fs%na)]
        ncp = Fp%na + 1; cutp(1:ncp) = [(ii, ii = 0, Fp%na)]
      else
        ! core: X (J = 0) to X (J = na)
        call group_stretch(Fc%na, .true., .true., gs, ngs)
        ncc = 1; cutc(1) = 0
        do ii = 1, ngs
          ncc = ncc + 1; cutc(ncc) = cutc(ncc - 1) + gs(ii)
        enddo
        ! SOL: the legs by the rule, the middle through the core's cut nodes
        isc = .false.
        do ii = 1, ncc
          if (cutc(ii) < Fc%na) isc(sepnode(Fc, cutc(ii))) = .true.
        enddo
        ncs = 1; cuts(1) = 0
        call group_stretch(Fs%xa(1), .false., .true., gs, ngs)
        do ii = 1, ngs
          ncs = ncs + 1; cuts(ncs) = cuts(ncs - 1) + gs(ii)
        enddo
        do jj = Fs%xa(1) + 1, Fs%xa(2)
          if (jj == Fs%xa(2) .or. isc(sepnode(Fs, jj))) then
            ncs = ncs + 1; cuts(ncs) = jj
          endif
        enddo
        call group_stretch(Fs%na - Fs%xa(2), .true., .false., gs, ngs)
        do ii = 1, ngs
          ncs = ncs + 1; cuts(ncs) = cuts(ncs - 1) + gs(ii)
        enddo
        ! private: through the SOL legs' cut nodes
        isc = .false.
        do ii = 1, ncs
          if (cuts(ii) < Fs%xa(1) .or. cuts(ii) > Fs%xa(2)) isc(sepnode(Fs, cuts(ii))) = .true.
        enddo
        ncp = 0
        do jj = 0, Fp%na
          if (jj == 0 .or. jj == Fp%na .or. jj == Fp%xa(1) .or. isc(sepnode(Fp, jj))) then
            ncp = ncp + 1; cutp(ncp) = jj
          endif
        enddo
      endif
      any2 = .false.
      do cp = 1, XP_NPATCH
        associate (F => xl(gf)%p(cp), C_ => xl(gf + 1)%p(cp))
        select case (cp)
        case (XP_CORE)
          if (.not. cuts_to_groups(cutc, ncc, C_%a0, C_%wa)) msg = "core: along-surface cuts more than 2 apart"
        case (XP_SOL)
          if (.not. cuts_to_groups(cuts, ncs, C_%a0, C_%wa)) msg = "SOL: along-surface cuts more than 2 apart"
        case (XP_PRIV)
          if (.not. cuts_to_groups(cutp, ncp, C_%a0, C_%wa)) msg = "private: along-surface cuts more than 2 apart"
        end select
        if (len_trim(msg) > 0) return
        ! across: pairs from the separatrix side
        call group_stretch(F%nb, F%bsep == 0, F%bsep /= 0, gs, ngs)
        allocate(C_%b0(0:ngs - 1), C_%wb(0:ngs - 1))
        s0 = 0
        do ii = 1, ngs
          C_%b0(ii - 1) = s0; C_%wb(ii - 1) = gs(ii); s0 = s0 + gs(ii)
        enddo
        any2 = any2 .or. any(C_%wa == 2) .or. any(C_%wb == 2)
        end associate
      enddo
      if (.not. any2) return
      ! the coarse elements: corner nodes and sizes, the splits (sizes rule)
      do cp = 1, XP_NPATCH
        associate (F => xl(gf)%p(cp), C_ => xl(gf + 1)%p(cp))
        C_%na = size(C_%a0); C_%nb = size(C_%b0)
        C_%bsep = merge(0, C_%nb, F%bsep == 0)
        C_%nx = F%nx
        do ii = 1, F%nx
          C_%xa(ii) = count(C_%a0 < F%xa(ii))
        enddo
        allocate(C_%vt(4, 0:C_%na - 1, 0:C_%nb - 1), C_%sz(3, 4, 0:C_%na - 1, 0:C_%nb - 1), &
                 C_%ls(0:C_%na - 1, 0:C_%nb - 1), C_%lt(0:C_%na - 1, 0:C_%nb - 1))
        do jj = 0, C_%nb - 1
          do ii = 0, C_%na - 1
            call coarse_cell(F, C_, ii, jj)
          enddo
        enddo
        end associate
      enddo
      end associate
      coarsen = .true.
    end function coarsen

    !> Coarse cell (ia, jb) of patch level C_ over F: lambda across from the
    !! s2 sizes at the shared node of its two fine rows, along from the s3
    !! sizes, each the mean of the cell's two edges; corner nodes from the
    !! corner fine elements, sizes S = s / lambda (or s / (1 - lambda)).
    subroutine coarse_cell(F, C_, ia, jb)
      type(xpl_t), intent(in)    :: F
      type(xpl_t), intent(inout) :: C_
      integer, intent(in) :: ia, jb
      integer :: fa, fb, ca_, cb_, ks, fa_k, fb_k
      real*8 :: x0, x1, ls_, lt_, fs, ft
      fa = C_%a0(ia); fb = C_%b0(jb); ca_ = C_%wa(ia); cb_ = C_%wb(jb)
      ls_ = 0.5d0; lt_ = 0.5d0
      if (cb_ == 2) then
        x0 = abs(F%sz(1, 2, fa, fb)); x1 = abs(F%sz(1, 1, fa, fb + 1))
        ls_ = x0 / max(x0 + x1, 1.d-300)
        x0 = abs(F%sz(1, 3, fa + ca_ - 1, fb)); x1 = abs(F%sz(1, 4, fa + ca_ - 1, fb + 1))
        ls_ = 0.5d0 * (ls_ + x0 / max(x0 + x1, 1.d-300))
      endif
      if (ca_ == 2) then
        x0 = abs(F%sz(2, 4, fa, fb)); x1 = abs(F%sz(2, 1, fa + 1, fb))
        lt_ = x0 / max(x0 + x1, 1.d-300)
        x0 = abs(F%sz(2, 3, fa, fb + cb_ - 1)); x1 = abs(F%sz(2, 2, fa + 1, fb + cb_ - 1))
        lt_ = 0.5d0 * (lt_ + x0 / max(x0 + x1, 1.d-300))
      endif
      if (ls_ <= 0.d0 .or. ls_ >= 1.d0) ls_ = 0.5d0          ! zero sizes (the axis)
      if (lt_ <= 0.d0 .or. lt_ >= 1.d0) lt_ = 0.5d0
      C_%ls(ia, jb) = ls_; C_%lt(ia, jb) = lt_
      do ks = 1, 4
        fa_k = fa + SLOT_A_(ks) * (ca_ - 1)
        fb_k = fb + SLOT_B_(ks) * (cb_ - 1)
        fs = 1.d0; ft = 1.d0
        if (cb_ == 2) fs = merge(ls_, 1.d0 - ls_, SLOT_B_(ks) == 0)
        if (ca_ == 2) ft = merge(lt_, 1.d0 - lt_, SLOT_A_(ks) == 0)
        C_%vt(ks, ia, jb) = F%vt(ks, fa_k, fb_k)
        C_%sz(1, ks, ia, jb) = F%sz(1, ks, fa_k, fb_k) / fs
        C_%sz(2, ks, ia, jb) = F%sz(2, ks, fa_k, fb_k) / ft
        C_%sz(3, ks, ia, jb) = F%sz(3, ks, fa_k, fb_k) / (fs * ft)
      enddo
    end subroutine coarse_cell

    !> The DOFs of level g (the vertex nodes' indices; coarse: minus dropc),
    !! JOREK index order, and their ring / position / column.
    subroutine level_dofs(g_)
      integer, intent(in) :: g_
      integer, allocatable :: pr(:), rawj(:), colc(:), colp(:)
      integer :: cp, aa, bb, ks, k4, id, ii, jj, nr_, rx, nbc, nbp, nbs, cnt_, rg_
      logical, allocatable :: seen(:)
      msg = ""
      associate (X => xl(g_), Lc => xl(g_)%p(XP_CORE), Ls => xl(g_)%p(XP_SOL), Lp => xl(g_)%p(XP_PRIV))
      nbc = Lc%nb; nbs = Ls%nb; nbp = Lp%nb
      rx = nbc + nbp + nbs + 3                       ! the X-point's ring
      X%nring = rx + 1
      ! columns: the SOL's; the core's and the private's through their shared separatrix nodes
      allocate(pr(node_list%n_nodes))
      pr = -1
      do jj = 0, Ls%na
        pr(sepnode(Ls, jj)) = jj
      enddo
      allocate(colc(0:Lc%na - 1), colp(0:Lp%na))
      do jj = 0, Lc%na - 1
        colc(jj) = pr(sepnode(Lc, jj))
      enddo
      ! the core's X-point column (its own X-point nodes): the SOL's X column
      ! next to the core's first
      if (colc(0) < 0 .and. Lc%na >= 2) then
        colc(0) = Ls%xa(1)
        if (abs(colc(1) - Ls%xa(2)) < abs(colc(1) - Ls%xa(1))) colc(0) = Ls%xa(2)
      endif
      do jj = 0, Lp%na
        colp(jj) = pr(sepnode(Lp, jj))
        if (jj == Lp%xa(1)) colp(jj) = Ls%na + 1            ! the private seam: a column of its own
      enddo
      if (any(colc < 0) .or. any(colp < 0)) then
        write(msg, '(A,I0,A,I0,A,I0,A,I0)') "separatrix nodes not shared between the patches: core ", &
          count(colc < 0), " of ", size(colc), ", private ", count(colp < 0), " of ", size(colp)
        return
      endif
      X%ncol = Ls%na + 3
      ! DOFs, first patch wins
      allocate(X%lmap(n_idx), X%rI(n_idx), X%rJ(n_idx), X%rC(n_idx), seen(n_idx))
      X%lmap = 0; seen = .false.
      do cp = 1, XP_NPATCH
        associate (L => X%p(cp))
        do bb = 0, L%nb - 1
          do aa = 0, L%na - 1
            do ks = 1, 4
              ii = bb + SLOT_B_(ks)
              jj = aa + SLOT_A_(ks)
              if (cp == XP_CORE) jj = modulo(jj, L%na)
              do k4 = 1, 4
                id = node_list%node(L%vt(ks, aa, bb))%index(k4)
                if (seen(id)) cycle
                if (g_ > 0 .and. dropc(id)) cycle
                seen(id) = .true.
                if (xdof(id)) then
                  X%rI(id) = rx; X%rJ(id) = 0; X%rC(id) = Ls%na + 2
                else if (cp == XP_CORE) then
                  X%rI(id) = ii; X%rJ(id) = jj; X%rC(id) = colc(jj)
                else if (cp == XP_PRIV) then
                  X%rI(id) = nbc + 1 + nbp - abs(ii - L%bsep); X%rJ(id) = jj; X%rC(id) = colp(jj)
                else
                  X%rI(id) = nbc + nbp + 2 + abs(ii - L%bsep); X%rJ(id) = jj; X%rC(id) = jj
                endif
              enddo
            enddo
          enddo
        enddo
        end associate
      enddo
      X%n = count(seen)
      allocate(X%dof(X%n))
      nr_ = 0
      do id = 1, n_idx
        if (.not. seen(id)) cycle
        nr_ = nr_ + 1
        X%dof(nr_) = id; X%lmap(id) = nr_
      enddo
      X%rI(1:X%n) = X%rI(X%dof); X%rJ(1:X%n) = X%rJ(X%dof); X%rC(1:X%n) = X%rC(X%dof)
      ! each ring's positions compacted to 0..nn-1 in their order; closed when
      ! it is a whole core ring
      if (allocated(xrg(g_)%nn)) deallocate(xrg(g_)%nn, xrg(g_)%per)
      allocate(xrg(g_)%nn(0:X%nring - 1), xrg(g_)%per(0:X%nring - 1))
      allocate(rawj(0:max(Ls%na, Lc%na, Lp%na) + 1))
      do rg_ = 0, X%nring - 1
        rawj = -1
        do ii = 1, X%n
          if (X%rI(ii) == rg_) rawj(X%rJ(ii)) = 0
        enddo
        cnt_ = 0
        do jj = 0, size(rawj) - 1
          if (rawj(jj) < 0) cycle
          rawj(jj) = cnt_; cnt_ = cnt_ + 1
        enddo
        do ii = 1, X%n
          if (X%rI(ii) == rg_) X%rJ(ii) = rawj(X%rJ(ii))
        enddo
        xrg(g_)%nn(rg_) = cnt_
        xrg(g_)%per(rg_) = (rg_ >= 1 .and. rg_ < nbc .and. cnt_ == Lc%na)
      enddo
      end associate
    end subroutine level_dofs

    !> P(gg) as scalar triplets (rr: level gg-1 DOF, cc: level gg DOF,
    !! 0-based) in rr/cc/ww(1:ntrip) of the rank's own fine rows, every
    !! patch's elements; a fine row seen by several (coarse cell, fine
    !! element, slot) instances gets their average.
    subroutine make_p(gg)
      integer, intent(in) :: gg
      integer :: cp, aa, bb, kc, dc, da, db, kf, dq, fr, gc, t, cap, i0, j0, e, u, v, nrow
      real*8 :: Cn(0:3, 0:3), Qn(0:6, 0:6), Sc(3, 4), Dm(4, 4), sub(0:3, 0:3)
      integer, allocatable :: rcnt(:), roff(:), fillp(:), cc2(:)
      real*8, allocatable :: ww2(:)
      nrow = xl(gg - 1)%n
      if (allocated(cnt)) deallocate(cnt)
      allocate(cnt(nrow))
      cnt = 0
      do cp = 1, XP_NPATCH
        associate (Lf => xl(gg - 1)%p(cp), Lc => xl(gg)%p(cp), Xf => xl(gg - 1), Xc => xl(gg))
        do bb = 0, Lc%nb - 1
          do aa = 0, Lc%na - 1
            Sc = Lc%sz(:, :, aa, bb)
            do db = 0, Lc%wb(bb) - 1
              do da = 0, Lc%wa(aa) - 1
                do kf = 1, 4
                  do dq = 1, 4
                    fr = Xf%lmap(node_list%node(Lf%vt(kf, Lc%a0(aa) + da, Lc%b0(bb) + db))%index(dq))
                    if (fr == 0) cycle
                    if (drop(Xf%dof(fr))) cycle
                    if (ly(gg - 1)%own(fr - 1) /= me) cycle
                    cnt(fr) = cnt(fr) + 1
                  enddo
                enddo
              enddo
            enddo
          enddo
        enddo
        end associate
      enddo
      ! at most one triplet per (instance, coarse DOF of its cell)
      cap = 16 * sum(cnt)
      if (allocated(rr)) deallocate(rr, cc, ww)
      allocate(rr(max(cap, 1)), cc(max(cap, 1)), ww(max(cap, 1)))
      ntrip = 0
      do cp = 1, XP_NPATCH
        associate (Lf => xl(gg - 1)%p(cp), Lc => xl(gg)%p(cp), Xf => xl(gg - 1), Xc => xl(gg))
        do bb = 0, Lc%nb - 1
          do aa = 0, Lc%na - 1
            Sc = Lc%sz(:, :, aa, bb)
            do kc = 1, 4
              do dc = 1, 4
                gc = Xc%lmap(node_list%node(Lc%vt(kc, aa, bb))%index(dc))
                if (gc == 0) cycle
                Dm = 0.d0; Dm(dc, kc) = 1.d0
                call net_from_dofs(Sc, Dm, Cn)
                Qn = 0.d0
                Qn(0:3, 0:3) = Cn
                if (Lc%wb(bb) == 2) call split_rows(Qn, 4, Lc%ls(aa, bb))
                if (Lc%wa(aa) == 2) call split_cols(Qn, merge(7, 4, Lc%wb(bb) == 2), Lc%lt(aa, bb))
                do db = 0, Lc%wb(bb) - 1
                  do da = 0, Lc%wa(aa) - 1
                    i0 = 3 * db; j0 = 3 * da
                    sub = Qn(i0:i0 + 3, j0:j0 + 3)
                    call dofs_from_net(Lf%sz(:, :, Lc%a0(aa) + da, Lc%b0(bb) + db), sub, Dm)
                    do kf = 1, 4
                      do dq = 1, 4
                        fr = Xf%lmap(node_list%node(Lf%vt(kf, Lc%a0(aa) + da, Lc%b0(bb) + db))%index(dq))
                        if (fr == 0) cycle
                        if (drop(Xf%dof(fr))) cycle
                        if (ly(gg - 1)%own(fr - 1) /= me) cycle
                        if (Dm(dq, kf) == 0.d0) cycle
                        ntrip = ntrip + 1
                        rr(ntrip) = fr; cc(ntrip) = gc; ww(ntrip) = Dm(dq, kf)
                      enddo
                    enddo
                  enddo
                enddo
              enddo
            enddo
          enddo
        enddo
        end associate
      enddo
      ! merge duplicates row by row, average over the instances
      allocate(rcnt(nrow), roff(nrow + 1))
      rcnt = 0
      do t = 1, ntrip
        rcnt(rr(t)) = rcnt(rr(t)) + 1
      enddo
      roff(1) = 0
      do t = 1, nrow
        roff(t + 1) = roff(t) + rcnt(t)
      enddo
      allocate(fillp(nrow), cc2(ntrip), ww2(ntrip))
      fillp = 0
      do t = 1, ntrip
        e = roff(rr(t)) + fillp(rr(t)) + 1
        fillp(rr(t)) = fillp(rr(t)) + 1
        cc2(e) = cc(t); ww2(e) = ww(t)
      enddo
      ntrip = 0
      do t = 1, nrow
        u = ntrip
        do e = roff(t) + 1, roff(t + 1)
          do v = u + 1, ntrip
            if (cc(v) == cc2(e)) exit
          enddo
          if (v <= ntrip) then
            ww(v) = ww(v) + ww2(e)
          else
            ntrip = ntrip + 1
            rr(ntrip) = t - 1; cc(ntrip) = cc2(e); ww(ntrip) = ww2(e)
          endif
        enddo
        do v = u + 1, ntrip
          ww(v) = ww(v) / max(cnt(t), 1)
        enddo
      enddo
      ! columns 0-based, tiny weights dropped
      e = 0
      do t = 1, ntrip
        if (abs(ww(t)) < 1.d-14) cycle
        e = e + 1
        rr(e) = rr(t); cc(e) = cc(t) - 1; ww(e) = ww(t)
      enddo
      ntrip = e
    end subroutine make_p

    integer function prow(L, f, d, m)
      integer, intent(in) :: L, f, d, m
      integer :: ow
      ow = ly(L)%own(d)
      prow = ly(L)%ps(ow) + (f * ly(L)%nl(ow) + ly(L)%lpos(d)) * pc_ntor + m
    end function prow

    !> The scalar triplets of P (rr, cc, ww) expanded to every field and slot
    !! (rows on level lr_, columns on lc_); each rank inserts its own rows.
    subroutine make_expanded(P, lr_, lc_)
      Mat, intent(out)    :: P
      integer, intent(in) :: lr_, lc_
      PetscInt :: nrl, ncl, nrg, ncg, row, col
      PetscErrorCode :: ierr_
      integer :: f, m, t
      nrl = n_fields * pc_ntor * ly(lr_)%nl(me); ncl = n_fields * pc_ntor * ly(lc_)%nl(me)
      nrg = n_fields * pc_ntor * ly(lr_)%n;      ncg = n_fields * pc_ntor * ly(lc_)%n
      call MatCreate(comm, P, ierr_)
      call MatSetSizes(P, nrl, ncl, nrg, ncg, ierr_)
      call MatSetType(P, MATMPIAIJ, ierr_)
      call MatSeqAIJSetPreallocation(P, 2 * MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr_)
      call MatMPIAIJSetPreallocation(P, 2 * MAX_ENT, PETSC_NULL_INTEGER_ARRAY, &
                                     2 * MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr_)
      call MatSetOption(P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr_)
      do t = 1, ntrip
        if (ly(lr_)%own(rr(t)) /= me) cycle
        do f = 0, n_fields - 1
          do m = 0, pc_ntor - 1
            row = prow(lr_, f, rr(t), m)
            col = prow(lc_, f, cc(t), m)
            call MatSetValue(P, row, col, ww(t), ADD_VALUES, ierr_)
          enddo
        enddo
      enddo
      call MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY, ierr_)
      call MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY, ierr_)
    end subroutine make_expanded

  end subroutine gmg_build_xpoint_prolongations

  !> JOREK's element as a Bezier net (i along s / across, j along t / along):
  !! P00 = u, P10 = u + s2 d_s, P01 = u + s3 d_t, P11 = P10 + P01 - P00 + s4 d_st,
  !! mirrored at the other corners. S(1:3, slot) = s2, s3, s4; D(dof, slot).
  subroutine net_from_dofs(S, D, C)
    real*8, intent(in)  :: S(3, 4), D(4, 4)
    real*8, intent(out) :: C(0:3, 0:3)
    integer :: k, i, j, di, dj
    do k = 1, 4
      i = NET_I(k); j = NET_J(k); di = NET_DI(k); dj = NET_DJ(k)
      C(i, j) = D(1, k)
      C(i + di, j) = D(1, k) + S(1, k) * D(2, k)
      C(i, j + dj) = D(1, k) + S(2, k) * D(3, k)
      C(i + di, j + dj) = C(i + di, j) + C(i, j + dj) - D(1, k) + S(3, k) * D(4, k)
    enddo
  end subroutine net_from_dofs

  !> The inverse; a zero size (fix_axis_nodes' angular sizes on the axis)
  !! gives a zero DOF: no basis function uses it.
  subroutine dofs_from_net(S, C, D)
    real*8, intent(in)  :: S(3, 4), C(0:3, 0:3)
    real*8, intent(out) :: D(4, 4)
    integer :: k, i, j, di, dj
    real*8 :: u
    do k = 1, 4
      i = NET_I(k); j = NET_J(k); di = NET_DI(k); dj = NET_DJ(k)
      u = C(i, j)
      D(1, k) = u
      D(2, k) = sdiv(C(i + di, j) - u, S(1, k))
      D(3, k) = sdiv(C(i, j + dj) - u, S(2, k))
      D(4, k) = sdiv(C(i + di, j + dj) - C(i + di, j) - C(i, j + dj) + u, S(3, k))
    enddo
  contains
    pure real*8 function sdiv(a, b)
      real*8, intent(in) :: a, b
      sdiv = 0.d0
      if (b /= 0.d0) sdiv = a / b
    end function sdiv
  end subroutine dofs_from_net

  !> de Casteljau at lam along axis 0 of Q(0:3, 0:nc-1) -> Q(0:6, 0:nc-1)
  subroutine split_rows(Q, nc, lam)
    real*8, intent(inout) :: Q(0:6, 0:6)
    integer, intent(in) :: nc
    real*8, intent(in) :: lam
    real*8 :: b0(0:3), b1(0:2), b2(0:1), b3
    integer :: j
    do j = 0, nc - 1
      b0 = Q(0:3, j)
      b1 = (1.d0 - lam) * b0(0:2) + lam * b0(1:3)
      b2 = (1.d0 - lam) * b1(0:1) + lam * b1(1:2)
      b3 = (1.d0 - lam) * b2(0) + lam * b2(1)
      Q(0:6, j) = [b0(0), b1(0), b2(0), b3, b2(1), b1(2), b0(3)]
    enddo
  end subroutine split_rows

  !> ... along axis 1 of Q(0:nr-1, 0:3) -> Q(0:nr-1, 0:6)
  subroutine split_cols(Q, nr, lam)
    real*8, intent(inout) :: Q(0:6, 0:6)
    integer, intent(in) :: nr
    real*8, intent(in) :: lam
    real*8 :: b0(0:3), b1(0:2), b2(0:1), b3
    integer :: i
    do i = 0, nr - 1
      b0 = Q(i, 0:3)
      b1 = (1.d0 - lam) * b0(0:2) + lam * b0(1:3)
      b2 = (1.d0 - lam) * b1(0:1) + lam * b1(1:2)
      b3 = (1.d0 - lam) * b2(0) + lam * b2(1)
      Q(i, 0:6) = [b0(0), b1(0), b2(0), b3, b2(1), b1(2), b0(3)]
    enddo
  end subroutine split_cols

  !> Canonical DOF numbering of one coarse level. The axis keeps the shared
  !! value and per-node a and c; its angular slope b is dropped (see header).
  !! bnd_drop: the boundary ring also drops u and the angular
  !! slope b, the two DOFs JOREK's Dirichlet condition fixes. A coarse function
  !! with u = b = 0 on the boundary prolongs to a fine one with u = b = 0 there
  !! (C1 subdivision along the boundary edge), so the constrained spaces stay
  !! exactly nested, P has empty rows for the fine Dirichlet DOFs, and the
  !! coarse correction never writes into them.
  subroutine make_level(L, ni, nj, bnd_drop)
    type(lvl_t), intent(inout) :: L
    integer, intent(in) :: ni, nj
    logical, intent(in) :: bnd_drop
    integer :: I, J, k, cnt

    L%ni = ni; L%nj = nj
    allocate(L%d(0:ni - 1, 0:nj - 1, 0:3))
    L%d = -1
    cnt = 1                                       ! 0 = the shared axis value
    do J = 0, nj - 1
      L%d(0, J, 0) = 0
      L%d(0, J, 1) = cnt; cnt = cnt + 1
      L%d(0, J, 3) = cnt; cnt = cnt + 1
    enddo
    do I = 1, ni - 1
      do J = 0, nj - 1
        do k = 0, 3
          if (bnd_drop .and. I == ni - 1 .and. (k == 0 .or. k == 2)) cycle
          L%d(I, J, k) = cnt; cnt = cnt + 1
        enddo
      enddo
    enddo
    L%n = cnt
  end subroutine make_level

  !> 1D sources of fine level-local index ip on a coarse line of nc nodes.
  subroutine sources_1d(ip, nc, periodic, s, W, ns)
    integer, intent(in)  :: ip, nc
    logical, intent(in)  :: periodic
    integer, intent(out) :: s(2), ns
    real*8, intent(out)  :: W(2, 2, 2)
    if (mod(ip, 2) == 0) then
      ns = 1; s(1) = ip / 2; W(:, :, 1) = W_CO
    else
      ns = 2
      s(1) = (ip - 1) / 2
      s(2) = s(1) + 1
      if (periodic) s(2) = mod(s(2), nc)
      W(:, :, 1) = W_L; W(:, :, 2) = W_R
    endif
  end subroutine sources_1d

  !> Canonical fine quantity k at (fi, fj) as a combination of coarse DOFs.
  subroutine interp_row(fi, fj, k, c, jsame, cols, w, nent)
    integer, intent(in)      :: fi, fj, k
    type(lvl_t), intent(in)  :: c
    logical, intent(in)      :: jsame           !< J not coarsened (semi_r): fine node = coarse node
    integer, intent(out)     :: cols(MAX_ENT), nent
    real*8, intent(out)      :: w(MAX_ENT)
    integer :: si(2), sj(2), nsi, nsj, a, b, kc, d, q
    real*8  :: Wi(2, 2, 2), Wj(2, 2, 2), wt
    logical :: found

    call sources_1d(fi, c%ni, .false., si, Wi, nsi)
    if (jsame) then
      nsj = 1; sj(1) = fj; Wj(:, :, 1) = W_ID
    else
      call sources_1d(fj, c%nj, .true.,  sj, Wj, nsj)
    endif
    nent = 0
    do a = 1, nsi
      do b = 1, nsj
        do kc = 0, 3
          wt = Wi(PQ_I(k) + 1, PQ_I(kc) + 1, a) * Wj(PQ_J(k) + 1, PQ_J(kc) + 1, b)
          if (wt == 0.0d0) cycle
          d = c%d(si(a), sj(b), kc)
          if (d < 0) cycle                       ! dropped axis b
          found = .false.
          do q = 1, nent
            if (cols(q) == d) then
              w(q) = w(q) + wt; found = .true.; exit
            endif
          enddo
          if (.not. found) then
            nent = nent + 1; cols(nent) = d; w(nent) = wt
          endif
        enddo
      enddo
    enddo
  end subroutine interp_row

  !--------------------------------------------------------------------
  !> Galerkin chain, smoothers, coarse and axis solves for operator A. Called
  !! at every PC rebuild with new values in A; P is reused, and with an
  !! unchanged pattern so are the coarse operators and the LUs' symbolic phase.
  !--------------------------------------------------------------------
  subroutine gmg_setup_operator(A, comm, my_id, tag, opts)
    Mat, intent(in)     :: A
    integer, intent(in) :: comm, my_id
    character(len=*), intent(in), optional :: tag   !< label for the prints
    type(gmg_opts_t), intent(in) :: opts    !< the whole configuration
    type(gmg_opts_t) :: o
    PetscErrorCode :: ierr
    PC   :: pc
    MatInfo :: minfo
    PetscInt :: nr
    integer :: g, rich_from
    real*8 :: nz0, nzt, nzg, rich_omega
    integer(8) :: sig(2)
    logical :: reuse

    ! Same fine pattern as the last build: keep the coarse operators (and the
    ! coarse LU's symbolic factorisation) and refill them in place. The same
    ! Mat (PETSc object ids are never reused) with an unchanged nonzero state
    ! -- the physics PC keeps its operands for the run, patterns frozen --
    ! needs no check; anything else is hashed.
    block
      PetscObjectState :: nzst
      PetscInt64 :: aid
      logical :: same_obj
      integer :: mpierr
      call MatGetNonzeroState(A, nzst, ierr)
      call PetscObjectGetId(A, aid, ierr)
      same_obj = op_ready .and. int(aid, 8) == a0_id .and. int(nzst, 8) == a0_nzst
      call MPI_Allreduce(MPI_IN_PLACE, same_obj, 1, MPI_LOGICAL, MPI_LAND, comm, mpierr)
      if (same_obj) then
        reuse = .true.
      else
        sig = pattern_sig(A, comm)
        reuse = op_ready .and. all(sig == a0_sig)
        a0_sig = sig
      endif
      a0_id = int(aid, 8); a0_nzst = int(nzst, 8)
    end block
    pat_fresh = op_ready .and. .not. reuse
    if (pat_fresh .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   GMG", cur_inst, &
      ": operator pattern changed, coarse operators and LUs rebuilt from scratch"

    if (op_ready) then
      if (.not. reuse) then
        do g = 1, nlev - 1
          call MatDestroy(gA(g), ierr)
        enddo
      endif
      ! the level smoothers' KSPs are kept (created once, re-pointed below)
    endif

    call gmg_register_events()
    gA(0) = A
    if (xperm) then
      ! the hierarchy's ring-major copy of A (see xperm); the scatter once
      if (op_ready .and. reuse) then
        call MatCreateSubMatrix(A, xp_is, xp_is, MAT_REUSE_MATRIX, xp_A, ierr)
      else
        if (op_ready) call MatDestroy(xp_A, ierr)
        call MatCreateSubMatrix(A, xp_is, xp_is, MAT_INITIAL_MATRIX, xp_A, ierr)
      endif
      if (.not. op_ready) then
        call MatCreateVecs(xp_A, xp_x, xp_b, ierr)
        block
          Vec :: va
          call MatCreateVecs(A, va, PETSC_NULL_VEC, ierr)
          call VecScatterCreate(va, xp_is, xp_b, PETSC_NULL_IS, xp_sc, ierr)
          call VecDestroy(va, ierr)
        end block
      endif
      gA(0) = xp_A
    endif
    call PetscLogEventBegin(gev_ptap, ierr)
    do g = 1, nlev - 1
      if (reuse) then
        call MatPtAP(gA(g - 1), gP(g), MAT_REUSE_MATRIX, 2.0d0, gA(g), ierr)
      else
        call MatPtAP(gA(g - 1), gP(g), MAT_INITIAL_MATRIX, 2.0d0, gA(g), ierr)
      endif
    enddo
    call PetscLogEventEnd(gev_ptap, ierr)
    ! the multiply kernel each level's matvecs actually run: MatPtAP decides
    ! the coarse types, whatever the fine operator was converted to
    if (.not. op_ready .and. my_id == 0) then
      block
        character(len=80) :: ta, tp
        integer :: gg
        write(*,'(A,I0,A)', advance="no") "[Physics PC]   GMG", cur_inst, " level types (A/P):"
        do gg = 0, nlev - 1
          call MatGetType(gA(gg), ta, ierr)
          tp = "-"
          if (gg > 0) call MatGetType(gP(gg), tp, ierr)
          write(*,'(A,I0,A,A,A,A)', advance="no") "  ", gg, "=", trim(ta), "/", trim(tp)
        enddo
        write(*,*)
      end block
    endif

    call PetscLogEventBegin(gev_smsetup, ierr)
    o = opts
    sm_type = o%smoother
    sm_nstep = o%nsmooth
    ! threaded matvecs: after the Galerkin chain, on every rebuild (the attach
    ! is idempotent and only re-plans on a new pattern). gA(0) is the caller's
    ! operator; P serves the V-cycle's prolongation (MatMultAdd).
    if (o%blockmv == 1) then
      block
        use mod_petsc_pc_harm, only: pc_ntor
        integer :: gg
        logical :: okb
        do gg = 0, nlev - 1
          okb = blockmv_attach(gA(gg), int(pc_ntor))
          if (gg > 0) okb = blockmv_attach(gP(gg), 1)
        enddo
        if (.not. op_ready .and. my_id == 0) write(*,'(A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, &
          ": level matvecs on the OpenMP block kernel (bs ", pc_ntor, ", ", nlev, " levels)"
      end block
    endif
    call opt_int("smoother", sm_type)
    if (sm_type /= 4 .and. sm_type /= 5 .and. sm_type /= 7 .and. sm_type /= 8) then
      if (my_id == 0) write(*,'(A,I0)') "[Physics PC]   FATAL: GMG smoother must be 4, 5, 7 or 8, got ", sm_type
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    if (sm_nstep <= 0) sm_nstep = SM_STEPS
    st_pre0(cur_inst) = sm_nstep; st_post0(cur_inst) = sm_nstep; st_crs(cur_inst) = sm_nstep
    if (o%pre0 >= 0)     st_pre0(cur_inst)  = o%pre0
    if (o%post0 >= 0)    st_post0(cur_inst) = o%post0
    if (o%nsmooth_c > 0) st_crs(cur_inst)   = o%nsmooth_c
    call opt_int("pre0", st_pre0(cur_inst))
    call opt_int("post0", st_post0(cur_inst))
    call opt_int("nsc", st_crs(cur_inst))
    if (st_pre0(cur_inst) + st_post0(cur_inst) == 0) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: GMG level 0 needs pre0 + post0 > 0"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    axis_k = max(o%axis_rings, -1)
    call opt_int("axis_rings", axis_k)
    lovl = 0
    if (sm_type /= 4) lovl = max(o%line_overlap, 0)
    if (sm_type /= 4) call opt_int("line_overlap", lovl)
    hpair = o%harm_pair
    if (hpair /= 1 .and. hpair /= 2) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: GMG harm_pair must be 1 or 2"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    rovl = 0
    if (sm_type == 4) rovl = max(o%ring_overlap, 0)
    if (sm_type == 4) call opt_int("ring_overlap", rovl)
    rsolv = o%ring_solver
    call opt_int("ring_solver", rsolv)
    if (rsolv < 0) rsolv = merge(RS_HYB, RS_LAPACK, hpair == 2)
    rseg = max(o%ring_segs, 0)
    call opt_int("ring_segs", rseg)
    rbmin = max(o%ring_btd_min, 0)
    call opt_int("ring_btd_min", rbmin)
    rchk = max(o%ring_check, 0)
    call opt_int("ring_check", rchk)
    block
      integer :: isp
      isp = o%ring_single
      call opt_int("ring_single", isp)
      if (isp < 0) isp = merge(1, 0, hpair == 2)
      rsp = (isp > 0 .and. rsolv > 0)
    end block
    if (rsolv > RS_HYB) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: GMG ring_solver must be -1..2"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    axsec = 0
    if (axis_k /= 0) axsec = o%axis_sectors
    if (axis_k /= 0) call opt_int("axis_sectors", axsec)
    ! X-point grids: the axis J-sectors split the axis block by the polar J,
    ! not yet on the virtual lattice (its LU instead); the line and ring
    ! overlaps walk the virtual lattice (rings, columns)
    if (xgrid .and. axsec /= 0) then
      if (my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   GMG", cur_inst, &
        ": X-point grid, axis J-sectors off"
      axsec = 0
    endif
    ! Levels g >= rich_from smooth with Richardson on the same blocks instead
    ! of GMRES: no norm, no inner product, so no global reduction below that
    ! level. GMRES costs two allreduces over the whole communicator per step,
    ! on coarse levels with a few hundred rows per rank and on ranks owning
    ! none -- the V-cycle's strong-scaling floor at np >= 32.
    rich_from = nlev
    if (o%rich_from >= 0) rich_from = o%rich_from
    ! X-point grids, the mixed pair (ring blocks, several fields): GMRES on
    ! every level. Richardson on its ring blocks lets the coarse levels
    ! stall at large dt there (21x33 deck, tstep 10: 13 inner its and 4.9 s
    ! per solve with Richardson from level 1, 17 from level 2, 4.3 and 1.9 s
    ! with GMRES throughout); the scalar blocks keep Richardson
    if (xgrid .and. sm_type == 4 .and. nf_s > 1) rich_from = nlev
    rich_omega = 1.d0
    if (o%rich_omega > 0.d0) rich_omega = o%rich_omega
    call opt_int("rich_from", rich_from)
    call opt_real("rich_omega", rich_omega)
    do g = 0, nlev - 2
      ! Created and configured once per hierarchy; a rebuild only re-points
      ! the operators and refactors the smoother blocks.
      if (.not. op_ready) call KSPCreate(comm, gSm(g), ierr)
      call KSPSetOperators(gSm(g), gA(g), gA(g), ierr)
      if (.not. op_ready) then
        if (g >= rich_from) then
          call KSPSetType(gSm(g), KSPRICHARDSON, ierr)
          call KSPRichardsonSetScale(gSm(g), rich_omega, ierr)
          call KSPSetNormType(gSm(g), KSP_NORM_NONE, ierr)
          call KSPSetTolerances(gSm(g), 1.d-30, 1.d-50, 1.d30, sm_nstep, ierr)
        else
          call KSPSetType(gSm(g), KSPGMRES, ierr)
          if (g == 0) then
            call KSPGMRESSetRestart(gSm(g), max(st_pre0(cur_inst), st_post0(cur_inst)), ierr)
          else
            call KSPGMRESSetRestart(gSm(g), st_crs(cur_inst), ierr)
          endif
          call KSPSetTolerances(gSm(g), 1.d-30, 1.d-50, 1.d30, sm_nstep, ierr)
          call KSPSetPCSide(gSm(g), PC_RIGHT, ierr)
        endif
        call KSPSetInitialGuessNonzero(gSm(g), PETSC_TRUE, ierr)
        call KSPGetPC(gSm(g), pc, ierr)
        call PCSetType(pc, PCSHELL, ierr)
        call PCShellSetApply(pc, blk_apply, ierr)
        call PCShellSetName(pc, "C1 block smoother", ierr)
      endif
      call build_blocks(g, gA(g))
      call KSPSetUp(gSm(g), ierr)
    enddo
    call axd_factor_pending()
    call PetscLogEventEnd(gev_smsetup, ierr)
    block
      integer :: ib(3), ibg(3), mpierr, nbtd, nseg, bb
      real*8 :: st(2)
      nbtd = 0; nseg = 0
      do g = 0, nlev - 2
        if (allocated(gBk(g)%btd)) nbtd = nbtd + count(gBk(g)%btd)
      enddo
      if (allocated(gBk(0)%btd)) then             ! the hybrid's segments per ring, level 0
        do bb = 1, gBk(0)%nb
          if (gBk(0)%btd(bb)) nseg = max(nseg, gBk(0)%rg(bb)%ns)
        enddo
      endif
      call MPI_Allreduce(MPI_IN_PLACE, nbtd, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
      call MPI_Allreduce(MPI_IN_PLACE, nseg, 1, MPI_INTEGER, MPI_MAX, comm, mpierr)
      ib = 0
      ib = [gBk(0)%nb, count(gBk(0)%band), sum(gBk(0:nlev-2)%nsing)]
      call MPI_Allreduce(ib, ibg, 3, MPI_INTEGER, MPI_SUM, comm, mpierr)
      ib(1) = 0
      if (gBk(0)%nb > 0) ib(1) = maxval(gBk(0)%kl)
      call MPI_Allreduce(MPI_IN_PLACE, ib(1), 1, MPI_INTEGER, MPI_MAX, comm, mpierr)
      ! the smoothers' factor storage over all levels (entries): sum and the
      ! largest rank
      st(1) = smoother_storage()
      st(2) = st(1)
      call MPI_Allreduce(MPI_IN_PLACE, st(1), 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, mpierr)
      call MPI_Allreduce(MPI_IN_PLACE, st(2), 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, mpierr)
      if (my_id == 0) then
        if (present(tag)) write(*,'(A,A,A)', advance="no") "[Physics PC]   GMG (", tag, ")"
        write(*,'(A,I0,A,I0,A,I0,A,I0,A)', advance="no") "[Physics PC]   GMG smoother ", sm_type, &
          ": V(", st_pre0(cur_inst), ",", st_post0(cur_inst), ") on level 0, ", st_crs(cur_inst), &
          " steps per side below"
        write(*,'(A,I0,A,I0,A,I0,A,I0,A)', advance="no") ", blocks ", ibg(1), " on level 0 (", &
          ibg(2), " banded, max kl ", ib(1), "; ", ibg(3), " singular -> point)"
        if (hpair == 2) write(*,'(A)', advance="no") ", all slots per block"
        ! the ring solver's line only where it has blocks (not on line smoothers)
        if (nbtd > 0) then
          select case (rsolv)
          case (RS_THOMAS)
            write(*,'(A)', advance="no") ", ring blocks by block Thomas"
          case (RS_HYB)
            write(*,'(A,I0,A)', advance="no") ", ring blocks by the Thomas / reduction hybrid (", nseg, &
              " segments on level 0)"
          end select
          write(*,'(A,I0,A)', advance="no") " (", nbtd, " over all levels)"
          if (rsp) write(*,'(A)', advance="no") ", single precision"
        endif
        write(*,'(A,F9.2,A,F9.2,A)') "; factors ", st(1) * 8.d-6, " MB (max rank ", st(2) * 8.d-6, " MB)"
        if (sm_type == 8) write(*,'(A,I0,A)', advance="no") &
          "[Physics PC]   GMG hybrid smoother: ring blocks on fine rings I < ", ring_is, &
          ", radial lines outside"
        if (axis_k > 0) write(*,'(A,I0,A)', advance="no") &
          "; axis block = rings 0..", axis_k, " on every level"
        if (axis_k < 0) write(*,'(A,I0,A)', advance="no") &
          "[Physics PC]   GMG axis block = rings 0..I_s-1 of every level (fine: 0..", axis_lim(0), ")"
        if (sm_type == 8 .or. axis_k /= 0) write(*,*)
        if (rich_from < nlev - 1) write(*,'(A,I0,A,I0,A,F5.2,A)') &
          "[Physics PC]   GMG", cur_inst, ": Richardson smoothing on levels >= ", rich_from, &
          " (scale ", rich_omega, ", no reductions)"
      endif
    end block

    ! Coarse and axis LUs carry an options prefix (gmg<k>_coarse_, gmg<k>_axis_,
    ! gmg<k>_axblk<g>_) for MUMPS options without a rebuild.
    ! Coarsest level: an exact LU done redundantly on the few ranks owning its
    ! rows (rds_t), not a MUMPS solve over the whole communicator per V-cycle.
    call PetscLogEventBegin(gev_coarselu, ierr)
    if (.not. gcrs%ready) then
      call MatGetLocalSize(gA(nlev - 1), nr, PETSC_NULL_INTEGER, ierr)
      allocate(gcrs%loc(nr))
      do g = 1, int(nr)
        gcrs%loc(g) = g - 1
      enddo
    endif
    call rds_setup(gcrs, gA(nlev - 1), "coarse", "coarse level")
    call PetscLogEventEnd(gev_coarselu, ierr)
    if (.not. vec_ready) then
      do g = 0, nlev - 1
        call MatCreateVecs(gA(g), gx(g), gb(g), ierr)
        call VecDuplicate(gx(g), gr(g), ierr)
      enddo
      vec_ready = .true.
    endif
    ! The C_op report describes the hierarchy's structure, frozen after the
    ! first build: first build only.
    pat_fresh = .false.
    if (op_ready) return
    op_ready = .true.

    ! operator complexity
    call MatGetInfo(gA(0), MAT_GLOBAL_SUM, minfo, ierr)
    nz0 = minfo%nz_used; nzt = nz0
    if (my_id == 0) write(*,'(A)', advance="no") "[Physics PC]   GMG levels n (nnz/row):"
    do g = 0, nlev - 1
      call MatGetSize(gA(g), nr, PETSC_NULL_INTEGER, ierr)
      call MatGetInfo(gA(g), MAT_GLOBAL_SUM, minfo, ierr)
      nzg = minfo%nz_used
      if (g > 0) nzt = nzt + nzg
      if (my_id == 0) write(*,'(A,I0,A,F7.1,A)', advance="no") " ", nr, " (", nzg / max(dble(nr), 1.d0), ")"
    enddo
    if (my_id == 0) write(*,'(A,F6.3)') "   C_op = ", nzt / max(nz0, 1.d0)
  end subroutine gmg_setup_operator

  !> Entries of the active hierarchy's smoother factors on this rank, all
  !! smoothing levels.
  real*8 function smoother_storage()
    integer :: g
    smoother_storage = 0.d0
    do g = 0, nlev - 2
      if (allocated(gBk(g)%lu4)) smoother_storage = smoother_storage + 0.5d0 * size(gBk(g)%lu4, kind=8)
      if (allocated(gBk(g)%lu)) then
        smoother_storage = smoother_storage + dble(size(gBk(g)%lu, kind=8))
      endif
    enddo
  end function smoother_storage

  !> Two hashes of A's sparsity pattern (row lengths and global column indices
  !! of the owned rows), summed over the ranks. Equal signatures between two
  !! rebuilds let gmg_setup_operator reuse the PtAP symbolic phase; any change
  !! of the pattern changes them.
  function pattern_sig(A, comm) result(sig)
    Mat, intent(in)     :: A
    integer, intent(in) :: comm
    integer(8) :: sig(2)
    integer(8), parameter :: PM = 2147483647_8
    PetscInt :: rst, ren, r, ncols, c
    PetscInt, pointer :: cols(:)
    PetscErrorCode :: ierr
    integer :: mpierr
    integer(8) :: h1, h2, v
    call MatGetOwnershipRange(A, rst, ren, ierr)
    h1 = mod(int(rst, 8) + 1_8, PM); h2 = mod(int(ren, 8) + 7_8, PM)
    do r = rst, ren - 1
      call MatGetRow(A, r, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      h1 = mod(h1 * 131_8 + int(ncols, 8) + 1_8, PM)
      do c = 1, ncols
        v = mod(int(cols(c), 8), PM)
        h1 = mod(h1 * 131_8 + v + 1_8, PM)
        h2 = mod(h2 * 1000003_8 + v + 3_8, PM)
      enddo
      call MatRestoreRow(A, r, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
    enddo
    sig = [h1, h2]
    call MPI_Allreduce(MPI_IN_PLACE, sig, 2, MPI_INTEGER8, MPI_SUM, comm, mpierr)
  end function pattern_sig

  !> Runtime overrides of the V-cycle shape for scaling experiments:
  !! -sf_gmg_<name>, then -sf_gmg<k>_<name> for hierarchy instance k
  !! (1 pair_w, 2 pair_psi, 3 rho, 4 T). Absent = v unchanged.
  subroutine opt_int(name, v)
    character(len=*), intent(in) :: name
    integer, intent(inout) :: v
    PetscInt :: pv
    PetscBool :: set
    PetscErrorCode :: ierr
    character(len=64) :: nm
    pv = v
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gmg_"//name, pv, set, ierr)
    write(nm, '(A,I0,A,A)') "-sf_gmg", cur_inst, "_", name
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(nm), pv, set, ierr)
    v = int(pv)
  end subroutine opt_int

  subroutine opt_real(name, v)
    character(len=*), intent(in) :: name
    real*8, intent(inout) :: v
    PetscReal :: pv
    PetscBool :: set
    PetscErrorCode :: ierr
    character(len=64) :: nm
    pv = v
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gmg_"//name, pv, set, ierr)
    write(nm, '(A,I0,A,A)') "-sf_gmg", cur_inst, "_", name
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(nm), pv, set, ierr)
    v = pv
  end subroutine opt_real

  subroutine gmg_register_events()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    if (gev_ready) return
    call PetscLogEventRegister("GMG_PtAP",     cid, gev_ptap,     ierr)
    call PetscLogEventRegister("GMG_Prolong",  cid, gev_prolong,  ierr)
    call PetscLogEventRegister("GMG_SmSetup",  cid, gev_smsetup,  ierr)
    call PetscLogEventRegister("GMG_CoarseLU", cid, gev_coarselu, ierr)
    call PetscLogEventRegister("GMG_BlkMesh",  cid, gev_blkmesh,  ierr)
    call PetscLogEventRegister("GMG_BlkPat",   cid, gev_blkpat,   ierr)
    call PetscLogEventRegister("GMG_BlkNum",   cid, gev_blknum,   ierr)
    call PetscLogEventRegister("GMG_AxNum",    cid, gev_axnum,    ierr)
    call PetscLogEventRegister("GMG_AxTry",    cid, gev_axtry,    ierr)
    call PetscLogEventRegister("GMG_VCycle",   cid, gev_vcycle(1),  ierr)
    call PetscLogEventRegister("GMG_Smooth0",  cid, gev_smooth0(1), ierr)
    call PetscLogEventRegister("GMG_SmoothC",  cid, gev_smooth(1),  ierr)
    call PetscLogEventRegister("GMG_Coarse",   cid, gev_coarse(1),  ierr)
    call PetscLogEventRegister("GMG2_VCycle",  cid, gev_vcycle(2),  ierr)
    call PetscLogEventRegister("GMG2_Smooth0", cid, gev_smooth0(2), ierr)
    call PetscLogEventRegister("GMG2_SmoothC", cid, gev_smooth(2),  ierr)
    call PetscLogEventRegister("GMG2_Coarse",  cid, gev_coarse(2),  ierr)
    call PetscLogEventRegister("GMG3_VCycle",  cid, gev_vcycle(3),  ierr)
    call PetscLogEventRegister("GMG3_Smooth0", cid, gev_smooth0(3), ierr)
    call PetscLogEventRegister("GMG3_SmoothC", cid, gev_smooth(3),  ierr)
    call PetscLogEventRegister("GMG3_Coarse",  cid, gev_coarse(3),  ierr)
    call PetscLogEventRegister("GMG4_VCycle",  cid, gev_vcycle(4),  ierr)
    call PetscLogEventRegister("GMG4_Smooth0", cid, gev_smooth0(4), ierr)
    call PetscLogEventRegister("GMG4_SmoothC", cid, gev_smooth(4),  ierr)
    call PetscLogEventRegister("GMG4_Coarse",  cid, gev_coarse(4),  ierr)
    call PetscLogEventRegister("GMG5_VCycle",  cid, gev_vcycle(5),  ierr)
    call PetscLogEventRegister("GMG5_Smooth0", cid, gev_smooth0(5), ierr)
    call PetscLogEventRegister("GMG5_SmoothC", cid, gev_smooth(5),  ierr)
    call PetscLogEventRegister("GMG5_Coarse",  cid, gev_coarse(5),  ierr)
    call PetscLogEventRegister("GMG_Lines",    cid, gev_lines(1),   ierr)
    call PetscLogEventRegister("GMG_AxSolve",  cid, gev_axsolve(1), ierr)
    call PetscLogEventRegister("GMG2_Lines",   cid, gev_lines(2),   ierr)
    call PetscLogEventRegister("GMG2_AxSolve", cid, gev_axsolve(2), ierr)
    call PetscLogEventRegister("GMG3_Lines",   cid, gev_lines(3),   ierr)
    call PetscLogEventRegister("GMG3_AxSolve", cid, gev_axsolve(3), ierr)
    call PetscLogEventRegister("GMG4_Lines",   cid, gev_lines(4),   ierr)
    call PetscLogEventRegister("GMG4_AxSolve", cid, gev_axsolve(4), ierr)
    call PetscLogEventRegister("GMG5_Lines",   cid, gev_lines(5),   ierr)
    call PetscLogEventRegister("GMG5_AxSolve", cid, gev_axsolve(5), ierr)
    gev_ready = .true.
  end subroutine gmg_register_events

  !> One V-cycle from level g (x out, zero initial guess). Level 0 takes
  !! st_pre0 / st_post0 steps, the coarse levels st_crs each side. Without
  !! pre-smoothing x is still 0 at the restriction, so r = b needs no matvec.
  recursive subroutine vcycle(g, b, x)
    integer, intent(in) :: g
    Vec :: b, x
    PetscErrorCode :: ierr
    integer :: npre, npost

    if (g == nlev - 1) then
      call PetscLogEventBegin(gev_coarse(cur_inst), ierr)
      call rds_solve(gcrs, b, x)
      call PetscLogEventEnd(gev_coarse(cur_inst), ierr)
      return
    endif
    call VecZeroEntries(x, ierr)
    if (g == 0) then
      npre = st_pre0(cur_inst); npost = st_post0(cur_inst)
    else
      npre = st_crs(cur_inst);  npost = st_crs(cur_inst)
    endif
    if (npre > 0) call smooth(g, b, x, .true., npre)     ! pre-smooth (zero guess: no b - A*0)
    if (npre == 0) then
      call VecCopy(b, gr(g), ierr)                        ! x is still 0: r = b
    else
      call MatMult(gA(g), x, gr(g), ierr)
      call VecAYPX(gr(g), -1.0d0, b, ierr)                ! r = b - A x
    endif
    call MatMultTranspose(gP(g + 1), gr(g), gb(g + 1), ierr)
    call vcycle(g + 1, gb(g + 1), gx(g + 1))
    call MatMultAdd(gP(g + 1), gx(g + 1), x, x, ierr)     ! x += P e_c
    if (npost > 0) call smooth(g, b, x, .false., npost)  ! post-smooth
  end subroutine vcycle

  !> nstep smoother iterations on level g. The level's GMRES restart is the
  !! larger of its two step counts (set once, at setup), so switching between
  !! the pre and post counts only moves the iteration cap.
  subroutine smooth(g, b, x, zero_guess, nstep)
    integer, intent(in) :: g
    Vec :: b, x
    logical, intent(in) :: zero_guess
    integer, intent(in) :: nstep
    PetscErrorCode :: ierr
    PetscInt :: ns
    cur_lev = g
    ns = nstep
    call KSPSetTolerances(gSm(g), 1.d-30, 1.d-50, 1.d30, ns, ierr)
    if (zero_guess) then
      call KSPSetInitialGuessNonzero(gSm(g), PETSC_FALSE, ierr)
    else
      call KSPSetInitialGuessNonzero(gSm(g), PETSC_TRUE, ierr)
    endif
    if (g == 0) then
      call PetscLogEventBegin(gev_smooth0(cur_inst), ierr)
    else
      call PetscLogEventBegin(gev_smooth(cur_inst), ierr)
    endif
    call KSPSolve(gSm(g), b, x, ierr)
    if (g == 0) then
      call PetscLogEventEnd(gev_smooth0(cur_inst), ierr)
    else
      call PetscLogEventEnd(gev_smooth(cur_inst), ierr)
    endif
  end subroutine smooth

  !> The smoother blocks of level g and their LU factors, in three parts:
  !!  - mesh part, first build only: the block map (0 = fine, from
  !!    fine_node/fine_harm; g > 0 from glv(g)%rnode/rharm) over the rank's
  !!    rows, the axis groups, and with a line overlap the ghost rows
  !!    (ovl_extend). Rows inside a block are ordered by their coordinate along
  !!    the line (I for radial lines, J for rings), so a radial line is banded.
  !!  - pattern part, once per operator pattern (blk_pattern): band widths,
  !!    storage (banded dgbtrf where the band is narrow, else dense dgetrf),
  !!    the zebra coupling CSR, and the value maps from A's CSR.
  !!  - numeric part, every rebuild: gather through the maps (blk_fill), then
  !!    factor. A singular block falls back to its diagonal (nsing).
  subroutine build_blocks(g, A)
    integer, intent(in) :: g
    Mat, intent(in)     :: A
    type(blk_t), pointer :: B
    PetscInt :: nr, r, ncols, rst, ren
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    PetscInt, parameter :: one = 1
    integer :: nc, I, J, m, bb, q, pcn, info, n, ldab, kk, nthr, mpierr, Jc
    integer, allocatable :: cnt(:), key(:)
    integer(8) :: ix
    PetscInt, allocatable :: axr(:)
    integer, allocatable :: binfo(:)
    real*8, allocatable :: a0chk(:)
    external :: dgetrf, dgbtrf

    B => gBk(g)
    call MatGetLocalSize(A, nr, PETSC_NULL_INTEGER, ierr)
    call MatGetOwnershipRange(A, rst, ren, ierr)
    if (.not. allocated(B%bid)) then
      call PetscLogEventBegin(gev_blkmesh, ierr)
      B%nrow = int(nr)
      allocate(B%bid(B%nrow), key(B%nrow))
      key = 0
      do r = 1, B%nrow
        if (g == 0) then
          I = fine_node(r) / nth0; J = mod(fine_node(r), nth0); m = fine_harm(r); nc = nth0
        else
          nc = glv(g)%nj
          I = glv(g)%rnode(r) / nc; J = mod(glv(g)%rnode(r), nc); m = glv(g)%rharm(r)
        endif
        Jc = row_col(g, int(r), J)
        B%bid(r) = blk_id(g, I, Jc, m)
        if (sm_type == 5 .or. sm_type == 7) key(r) = I
        if (sm_type == 4) key(r) = rkey(g, I, J, nc)
        if (sm_type == 8) then
          if (I < ring_lim(g)) then
            key(r) = rkey(g, I, J, nc)
          else
            key(r) = I
          endif
        endif
        if (I <= axis_lim(g) .and. axis_k /= 0) key(r) = I * nc + J
      enddo
      ! compress to contiguous ids (the fine map leaves the axis nodes' slots unused)
      allocate(cnt(maxval(B%bid)))
      cnt = 0
      do r = 1, B%nrow
        cnt(B%bid(r)) = 1
      enddo
      bb = 0
      do q = 1, size(cnt)
        if (cnt(q) > 0) then
          bb = bb + 1; cnt(q) = bb
        endif
      enddo
      do r = 1, B%nrow
        B%bid(r) = cnt(B%bid(r))
      enddo
      deallocate(cnt)
      B%nb = bb
      allocate(cnt(B%nb), B%off(B%nb), B%sz(B%nb), B%rows(B%nrow), B%pos(B%nrow), &
               B%loff(B%nb), B%piv(B%nrow), B%kl(B%nb), B%ku(B%nb), B%band(B%nb))
      cnt = 0
      do r = 1, B%nrow
        cnt(B%bid(r)) = cnt(B%bid(r)) + 1
      enddo
      B%sz = cnt
      if (B%nb > 0) B%off(1) = 0                ! a rank may own no rows on a coarse level
      do bb = 2, B%nb
        B%off(bb) = B%off(bb - 1) + B%sz(bb - 1)
      enddo
      cnt = 0
      do r = 1, B%nrow
        bb = B%bid(r)
        cnt(bb) = cnt(bb) + 1
        B%rows(B%off(bb) + cnt(bb)) = int(r) - 1
      enddo
      ! stable sort of each block's rows by key (line coordinate), then positions
      do bb = 1, B%nb
        call sort_rows_by_key(B%rows(B%off(bb) + 1:B%off(bb) + B%sz(bb)), key)
        do q = 1, B%sz(bb)
          B%pos(B%rows(B%off(bb) + q) + 1) = q
        enddo
      enddo
      deallocate(cnt, key)
      ! the axis block(s) go to a sparse direct solve
      allocate(B%axblk(B%nb))
      B%axblk = .false.
      if (axis_k /= 0) then
        do bb = 1, B%nb
          r = B%rows(B%off(bb) + 1) + 1
          if (g == 0) then
            I = fine_node(r) / nth0
          else
            I = glv(g)%rnode(r) / glv(g)%nj
          endif
          B%axblk(bb) = (I <= axis_lim(g))
        enddo
      endif
      B%axsparse = (axis_k /= 0)
      allocate(B%bcol(B%nb))
      B%bcol = 0
      if (sm_type == 7 .or. sm_type == 8) then
        do bb = 1, B%nb
          r = B%rows(B%off(bb) + 1) + 1
          if (g == 0) then
            I = fine_node(r) / nth0; J = mod(fine_node(r), nth0)
          else
            I = glv(g)%rnode(r) / glv(g)%nj; J = mod(glv(g)%rnode(r), glv(g)%nj)
          endif
          if (I /= 0 .and. .not. B%axblk(bb)) B%bcol(bb) = mod(row_col(g, int(r), J), 2)
          ! smoother 8: ring blocks alternate in I (zebra over rings)
          if (sm_type == 8 .and. I < ring_lim(g) .and. .not. B%axblk(bb)) B%bcol(bb) = mod(I, 2)
        enddo
      endif
      if (B%axsparse) then
        allocate(axr(sum(B%sz, mask=B%axblk)))
        n = 0
        do bb = 1, B%nb
          if (.not. B%axblk(bb)) cycle
          do q = 1, B%sz(bb)
            n = n + 1
            axr(n) = rst + B%rows(B%off(bb) + q)
          enddo
        enddo
        call ISCreateGeneral(gcomm, int(n, kind(nr)), axr, PETSC_COPY_VALUES, B%axis_is, ierr)
        call ISSort(B%axis_is, ierr)
        ! the same rows, local and ascending, for its direct solve
        allocate(B%axg(1), key(B%nrow))
        key = 0
        do q = 1, n
          key(axr(q) - rst + 1) = 1
        enddo
        allocate(B%axg(1)%loc(count(key == 1)))
        n = 0
        do r = 1, B%nrow
          if (key(r) /= 1) cycle
          n = n + 1; B%axg(1)%loc(n) = int(r) - 1
        enddo
        deallocate(axr, key)
      endif
      if (lovl > 0 .and. sm_type /= 4) call ovl_extend(g, A, B, rst, ren, lovl, .false.)
      if (rovl > 0 .and. sm_type == 4) call ovl_extend(g, A, B, rst, ren, rovl, .true.)
      call btd_mark(g, B)
      nthr = 1
      !$ nthr = omp_get_max_threads()
      if (allocated(blk_t_work)) then
        if (size(blk_t_work, 1) < maxval(B%sz) .or. size(blk_t_work, 2) < nthr) deallocate(blk_t_work)
      endif
      if (.not. allocated(blk_t_work)) allocate(blk_t_work(maxval(B%sz), 0:nthr - 1))
      call PetscLogEventEnd(gev_blkmesh, ierr)
    endif

    ! Pattern part, once per operator pattern: band widths and storage of the
    ! blocks, the zebra coupling CSR and the value maps. The fine operand and
    ! the Galerkin chain keep their Mats and patterns for the run, so this
    ! runs on the first build only; a rebuild refills the ghost rows and
    ! gathers (numeric part).
    block
      PetscObjectState :: nzst
      PetscInt64 :: aid
      logical :: newpat
      call MatGetNonzeroState(A, nzst, ierr)
      call PetscObjectGetId(A, aid, ierr)
      newpat = (int(aid, 8) /= B%pat_id .or. int(nzst, 8) /= B%pat_nz)
      call MPI_Allreduce(MPI_IN_PLACE, newpat, 1, MPI_LOGICAL, MPI_LOR, gcomm, mpierr)
      if (newpat) then
        call PetscLogEventBegin(gev_blkpat, ierr)
        call blk_pattern(B, A, rst, ren)
        call PetscLogEventEnd(gev_blkpat, ierr)
        B%pat_id = int(aid, 8); B%pat_nz = int(nzst, 8)
      else if (B%ovl) then
        call MatCreateSubMatrices(A, one, [B%isg], [B%isall], MAT_REUSE_MATRIX, B%sg, ierr)
      endif
    end block
    call PetscLogEventBegin(gev_blknum, ierr)
    call blk_fill(B, A)

    ! The blocks are independent: factor them on the rank's OpenMP threads
    ! (hybrid runs leave them idle in the PETSc parts). The singular-block
    ! fallback reads A (MatGetRow is not thread-safe), so it runs afterwards.
    allocate(binfo(B%nb))
    binfo = 0
    if (rchk > 0) a0chk = B%lu
    if (rsp) then
      if (allocated(B%eqs)) then
        if (size(B%eqs) /= size(B%rows)) deallocate(B%eqs)
      endif
      if (.not. allocated(B%eqs)) allocate(B%eqs(size(B%rows)))
      B%eqs = 1.d0
    endif
    !$omp parallel do schedule(dynamic, 1) private(bb, n, ldab)
    do bb = 1, B%nb
      if (B%axblk(bb)) cycle
      n = B%sz(bb)
      if (B%btd(bb)) then
        B%jac(bb) = .false.
        if (rsp) call btd_equil(B%rg(bb), B%lu(B%loff(bb) + 1), B%eqs(B%off(bb) + 1))
        if (.not. allocated(B%wk)) call btd_factor(B%rg(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), binfo(bb))
      else if (B%band(bb)) then
        ldab = 2 * B%kl(bb) + B%ku(bb) + 1
        call dgbtrf(n, n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), ldab, B%piv(B%off(bb) + 1), binfo(bb))
      else
        call dgetrf(n, n, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), binfo(bb))
      endif
    enddo
    !$omp end parallel do
    if (allocated(B%wk)) call hyb_par_factor(B, binfo)
    B%nsing = 0
    do bb = 1, B%nb
      n = B%sz(bb)
      if (B%axblk(bb)) cycle
      info = binfo(bb)
      if (info /= 0) then
        ! singular block: keep only its diagonal, from the Pmat (point Jacobi)
        B%nsing = B%nsing + 1
        ! a ring solver's block: a singular node block (the block
        ! elimination pivots inside the nodes only)
        if (B%btd(bb)) B%jac(bb) = .true.
        if (rsp) B%eqs(B%off(bb) + 1 : B%off(bb) + n) = 1.d0
        B%lu(B%loff(bb) + 1 : B%loff(bb) + blk_size(B, bb)) = 0.0d0
        do q = 1, n
          r = B%rows(B%off(bb) + q)
          ix = lu_index(B, bb, q, q)
          if (r < B%nrow) then
            call MatGetRow(A, rst + r, ncols, cols, vals, ierr)
            do pcn = 1, int(ncols)
              if (cols(pcn) == rst + r) B%lu(ix) = vals(pcn)
            enddo
            call MatRestoreRow(A, rst + r, ncols, cols, vals, ierr)
          else                                          ! ghost row: from sg(1), global columns
            call MatGetRow(B%sg(1), r - B%nrow, ncols, cols, vals, ierr)
            do pcn = 1, int(ncols)
              if (cols(pcn) == B%gidx(r - B%nrow + 1)) B%lu(ix) = vals(pcn)
            enddo
            call MatRestoreRow(B%sg(1), r - B%nrow, ncols, cols, vals, ierr)
          endif
          if (abs(B%lu(ix)) < tiny(1.0d0)) B%lu(ix) = 1.0d0
          B%piv(B%off(bb) + q) = q
        enddo
      endif
    enddo
    if (rsp) call btd_single(B)
    if (rchk > 0) call btd_check(g, B, a0chk)
    if (rchk > 1) call btd_decay(g, B)

    call PetscLogEventEnd(gev_blknum, ierr)
    if (B%axsparse .and. B%axdon .and. pat_fresh) then
      ! the sector solve's submatrices hold the old pattern: the LU takes over
      B%axdon = .false.
      if (gme == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, " level ", g, &
        ": axis J-sector solve dropped for the LU after a pattern change"
    endif
    if (B%axsparse .and. B%axdon) then
      ! refill only: the factorisations of all levels run together after the
      ! level loop (axd_factor_pending), on each level's own sector ranks
      call PetscLogEventBegin(gev_axnum, ierr)
      do kk = 1, size(B%axd)
        call axd_refill(B%axd(kk), A)
      enddo
      B%axpend = .true.
      B%axA = A                                  ! for the LU, should a factorisation fail
      call PetscLogEventEnd(gev_axnum, ierr)
    else if (B%axsparse) then
      ! first build: the J-sector solve is set up before any LU of the whole
      ! block; its factorisation and gate follow the level loop
      ! (axd_factor_pending), and the LU is only built where it is not in use
      if (axsec /= 0 .and. .not. allocated(B%axd)) then
        call PetscLogEventBegin(gev_axtry, ierr)
        call axd_prepare(g, A, B)
        call PetscLogEventEnd(gev_axtry, ierr)
      endif
    endif
    if (B%axsparse .and. .not. B%axdon .and. .not. B%axnew) call ax_lu(g, A, B)
  end subroutine build_blocks

  !> ring_solver > 0, mesh part: the ring blocks solved as periodic
  !! block-tridiagonal chains (mod_gmg_ring_btd). A block qualifies when it is
  !! a ring (smoother 4, or 8 inside ring_lim) other than the axis, spans all
  !! nc nodes of its ring (the cut-ring ghosts included) and has at least
  !! ring_btd_min rows. Its rows are re-sorted to natural J order, stably, so
  !! a node's rows stay in their packed order. The rest keep the LAPACK LU.
  !! The hybrid's segments (ring_segs 0): ceil(threads / rings) when the
  !! rank's rings of this level are fewer than its threads, else 1. Splitting
  !! rings only pays then: at 16 rings on 12 threads, 3 segments cost 15% in
  !! the ring solves (tok s540, bandwidth-bound) and 1 equals block Thomas.
  subroutine btd_mark(g, B)
    integer, intent(in) :: g
    type(blk_t), intent(inout) :: B
    integer :: nc, nall, r, bb, q, I, ns, nthr, n, k
    integer, allocatable :: key(:), cnt(:,:), nnb(:)
    nall = B%nrow + B%ngh
    if (allocated(B%btd)) deallocate(B%btd, B%jac, B%rg, B%qn)
    allocate(B%btd(B%nb), B%jac(B%nb), B%rg(B%nb), B%qn(nall))
    B%btd = .false.; B%jac = .false.; B%qn = 0
    if (rsolv == RS_LAPACK .or. (sm_type /= 4 .and. sm_type /= 8)) return
    if (g == 0) then
      nc = nth0
    else
      nc = glv(g)%nj
    endif
    if (nc < 4) return
    allocate(key(nall), cnt(0:nc - 1, B%nb), nnb(B%nb))
    nnb = nc
    do r = 1, nall
      if (r <= B%nrow) then
        if (g == 0) then
          key(r) = mod(fine_node(r), nth0)
        else
          key(r) = mod(glv(g)%rnode(r), nc)
        endif
      else
        key(r) = B%ghJ(r - B%nrow)
      endif
    enddo
    do bb = 1, B%nb
      if (B%axblk(bb) .or. B%sz(bb) < max(rbmin, 1)) cycle
      ! the ring's I, from one of its own rows
      I = -1
      do q = 1, B%sz(bb)
        r = B%rows(B%off(bb) + q) + 1
        if (r > B%nrow) cycle
        if (g == 0) then
          I = fine_node(r) / nth0
        else
          I = glv(g)%rnode(r) / nc
        endif
        exit
      enddo
      if (I <= 0) cycle
      if (sm_type == 8) then
        if (I >= ring_lim(g)) cycle
      endif
      ! an X-point grid's ring: its own length (an open line is a ring
      ! without the wrap couplings)
      if (xgrid) nnb(bb) = xrg(g)%nn(I)
      if (nnb(bb) < 4) cycle
      cnt(:, bb) = 0
      do q = 1, B%sz(bb)
        r = B%rows(B%off(bb) + q) + 1
        cnt(key(r), bb) = cnt(key(r), bb) + 1
      enddo
      if (any(cnt(0:nnb(bb) - 1, bb) == 0)) cycle   ! still cut by a rank boundary
      call sort_rows_by_key(B%rows(B%off(bb) + 1:B%off(bb) + B%sz(bb)), key)
      do q = 1, B%sz(bb)
        r = B%rows(B%off(bb) + q)
        B%pos(r + 1) = q
        B%qn(B%off(bb) + q) = key(r + 1)
      enddo
      B%btd(bb) = .true.
    enddo
    ns = rseg
    if (ns == 0) then
      nthr = 1
      !$ nthr = omp_get_max_threads()
      n = max(count(B%btd), 1)
      ns = 1
      if (n < nthr) ns = (nthr + n - 1) / n
    endif
    do bb = 1, B%nb
      if (B%btd(bb)) call btd_plan(B%rg(bb), rsolv, nnb(bb), cnt(0:nnb(bb) - 1, bb), ns)
    enddo
    ! smoother 4 with segmented rings: the hybrid threaded over (block,
    ! segment) work items, one item for every other non-axis block
    if (allocated(B%wk)) deallocate(B%wk, B%lb, B%lk)
    if (rsolv == RS_HYB .and. sm_type == 4 .and. ns > 1 .and. any(B%btd)) then
      allocate(B%wk(nall))
      do q = 1, 2
        n = 0
        do bb = 1, B%nb
          if (B%axblk(bb)) cycle
          do k = 0, merge(B%rg(bb)%ns, 1, B%btd(bb)) - 1
            n = n + 1
            if (q == 2) then
              B%lb(n) = bb; B%lk(n) = k
            endif
          enddo
        enddo
        if (q == 1) allocate(B%lb(n), B%lk(n))
      enddo
    endif
  end subroutine btd_mark

  !> hybrid on smoother 4: the segments of all the rank's rings on the
  !! threads, then every ring's separators. One parallel region.
  subroutine hyb_par_factor(B, binfo)
    type(blk_t), intent(inout) :: B
    integer, intent(inout) :: binfo(:)
    integer :: p, bb, info
    !$omp parallel private(p, bb, info)
    !$omp do schedule(dynamic, 1)
    do p = 1, size(B%lb)
      bb = B%lb(p)
      if (.not. B%btd(bb)) cycle
      call hyb_fac_seg(B%rg(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), B%lk(p), info)
      if (info /= 0) then
        !$omp atomic write
        binfo(bb) = info
      endif
    enddo
    !$omp end do
    !$omp do schedule(dynamic, 1)
    do bb = 1, B%nb
      if (.not. B%btd(bb)) cycle
      if (binfo(bb) /= 0) cycle
      call hyb_fac_sep(B%rg(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), binfo(bb))
    enddo
    !$omp end do
    !$omp end parallel
  end subroutine hyb_par_factor

  !> ring_single: the ring blocks' factors rounded to single, into lu4 (its
  !! layout is set again on every build; the pattern rarely changes)
  subroutine btd_single(B)
    type(blk_t), intent(inout) :: B
    integer :: bb
    integer(8) :: t
    if (allocated(B%loff4)) deallocate(B%loff4)
    allocate(B%loff4(B%nb))
    t = 0
    do bb = 1, B%nb
      B%loff4(bb) = t
      if (B%btd(bb)) t = t + B%rg(bb)%tot
    enddo
    if (allocated(B%lu4)) then
      if (size(B%lu4, kind=8) /= t) deallocate(B%lu4)
    endif
    if (.not. allocated(B%lu4)) allocate(B%lu4(t))
    if (allocated(B%wk) .and. .not. allocated(B%wk4)) allocate(B%wk4(size(B%wk)))
    !$omp parallel do schedule(dynamic, 1)
    do bb = 1, B%nb
      if (B%btd(bb)) call btd_to_sp(B%rg(bb), B%lu(B%loff(bb) + 1), B%lu4(B%loff4(bb) + 1))
    enddo
    !$omp end parallel do
  end subroutine btd_single

  !> ring_check: the relative residual |b - A x| / |b| of every ring solve
  !! (random b), A from the unfactored values a0; the largest per level
  subroutine btd_check(g, B, a0)
    integer, intent(in) :: g
    type(blk_t), intent(in) :: B
    real*8, intent(in) :: a0(*)
    integer :: bb, n, mpierr, nchk
    real*8 :: res
    real*8, allocatable :: rhs(:), x(:), y(:)
    res = 0.d0; nchk = 0
    do bb = 1, B%nb
      if (.not. B%btd(bb)) cycle
      if (B%jac(bb)) cycle
      n = B%sz(bb)
      allocate(rhs(n), x(n), y(n))
      call random_number(rhs)
      x = rhs
      if (rsp) then                                ! the solve the smoother uses
        block
          real*4 :: x4(n)
          x4 = real(x * B%eqs(B%off(bb) + 1 : B%off(bb) + n), 4)
          call btd_solve_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%piv(B%off(bb) + 1), x4)
          x = x4 * B%eqs(B%off(bb) + 1 : B%off(bb) + n)
        end block
      else
        call btd_solve(B%rg(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), x)
      endif
      call btd_matvec(B%rg(bb), a0(B%loff(bb) + 1), x, y)
      res = max(res, maxval(abs(rhs - y)) / maxval(abs(rhs)))
      nchk = nchk + 1
      deallocate(rhs, x, y)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, res, 1, MPI_DOUBLE_PRECISION, MPI_MAX, gcomm, mpierr)
    call MPI_Allreduce(MPI_IN_PLACE, nchk, 1, MPI_INTEGER, MPI_SUM, gcomm, mpierr)
    if (gme == 0) write(*,'(A,I0,A,I0,A,I0,A,ES9.2)') "[Physics PC]   GMG", cur_inst, " level ", g, &
      ": ring solver check, ", nchk, " blocks, max |b - A x| / |b| ", res
  end subroutine btd_check

  !> ring_check 2, block Thomas rings: how the fill (the periodic wrap's
  !! spikes, H_J = D~_J^-1 C~_J and R~_J) decays with the distance d of node
  !! J from the ring ends where it starts (J = 0 and c-1). Prints per d the
  !! largest |H_J|_F / max(|H_0|_F, |H_c-1|_F) over the rings (R likewise),
  !! and the share of the fill (entries) in blocks below 1e-3 and 1e-6.
  subroutine btd_decay(g, B)
    integer, intent(in) :: g
    type(blk_t), intent(in) :: B
    integer, parameter :: DMAX = 16
    integer :: bb, J, c, d, mpierr
    real*8 :: hd(0:DMAX), rd(0:DMAX), h0, r0, sm(3)
    real*8, allocatable :: hn(:), rn(:)
    hd = 0.d0; rd = 0.d0; sm = 0.d0
    do bb = 1, B%nb
      if (.not. B%btd(bb) .or. B%jac(bb)) cycle
      if (B%rg(bb)%meth /= RS_THOMAS) cycle
      associate (rg => B%rg(bb))
        c = rg%nn - 1
        allocate(hn(0:c - 1), rn(0:c - 1))
        do J = 0, c - 1
          hn(J) = fro(rg%oc(J), int(rg%bs(J), 8) * rg%bs(c))
          rn(J) = fro(rg%orr(J), int(rg%bs(J), 8) * rg%bs(c))
        enddo
        h0 = max(hn(0), hn(c - 1), tiny(1.d0)); r0 = max(rn(0), rn(c - 1), tiny(1.d0))
        do J = 0, c - 1
          d = min(J, c - 1 - J, DMAX)
          hd(d) = max(hd(d), hn(J) / h0); rd(d) = max(rd(d), rn(J) / r0)
          sm(1) = sm(1) + 2.d0 * rg%bs(J) * rg%bs(c)
          if (hn(J) < 1.d-3 * h0) sm(2) = sm(2) + rg%bs(J) * rg%bs(c)
          if (rn(J) < 1.d-3 * r0) sm(2) = sm(2) + rg%bs(J) * rg%bs(c)
          if (hn(J) < 1.d-6 * h0) sm(3) = sm(3) + rg%bs(J) * rg%bs(c)
          if (rn(J) < 1.d-6 * r0) sm(3) = sm(3) + rg%bs(J) * rg%bs(c)
        enddo
        deallocate(hn, rn)
      end associate
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, hd, DMAX + 1, MPI_DOUBLE_PRECISION, MPI_MAX, gcomm, mpierr)
    call MPI_Allreduce(MPI_IN_PLACE, rd, DMAX + 1, MPI_DOUBLE_PRECISION, MPI_MAX, gcomm, mpierr)
    call MPI_Allreduce(MPI_IN_PLACE, sm, 3, MPI_DOUBLE_PRECISION, MPI_SUM, gcomm, mpierr)
    if (gme == 0) then
      write(*,'(A,I0,A,I0,A,17ES9.1)') "[Physics PC]   GMG", cur_inst, " level ", g, ": ring fill decay H(d) ", hd
      write(*,'(A,I0,A,I0,A,17ES9.1)') "[Physics PC]   GMG", cur_inst, " level ", g, ": ring fill decay R(d) ", rd
      if (sm(1) > 0) write(*,'(A,I0,A,I0,A,F6.3,A,F6.3)') "[Physics PC]   GMG", cur_inst, " level ", g, &
        ": ring fill below 1e-3 ", sm(2) / sm(1), ", below 1e-6 ", sm(3) / sm(1)
    endif

  contains

    real*8 function fro(o, n)
      integer(8), intent(in) :: o, n
      fro = sqrt(sum(B%lu(B%loff(bb) + o + 1 : B%loff(bb) + o + n)**2))
    end function fro
  end subroutine btd_decay

  !> Pattern part of build_blocks: band widths and storage layout of the
  !! blocks, the zebra's coupling CSR, and the value maps from A's CSR (and
  !! the ghost rows' sg(1)) into them. Two passes over the combined rows:
  !! count, then place. Row entries are visited in CSR order (ascending
  !! column), so the zebra coupling sums run in the same order as before.
  subroutine blk_pattern(B, A, rst, ren)
    type(blk_t), intent(inout) :: B
    Mat, intent(in)      :: A
    PetscInt, intent(in) :: rst, ren
    PetscInt, parameter :: one = 1
    Mat :: Ad, Ao
    PetscInt, pointer :: garr(:), dia(:), dja(:), oia(:), oja(:), sia(:), sja(:)
    PetscInt :: nd, no, ns
    logical :: has_o
    integer :: pass, R, nall, bb, n, kz, nva, nvg, nbad
    PetscInt :: k
    integer(8) :: tot
    PetscErrorCode :: ierr

    if (B%ovl) then
      if (associated(B%sg)) call MatDestroySubMatrices(one, B%sg, ierr)
      call MatCreateSubMatrices(A, one, [B%isg], [B%isall], MAT_INITIAL_MATRIX, B%sg, ierr)
    endif
    call aij_parts(A, Ad, Ao, garr, has_o)
    call get_ij(Ad, .false., nd, dia, dja)
    B%nvd = dia(nd + 1); B%nvo = 0; B%nvg = 0
    if (has_o) then
      call get_ij(Ao, .false., no, oia, oja)
      B%nvo = oia(no + 1)
    endif
    if (B%ovl) then
      call get_ij(B%sg(1), .false., ns, sia, sja)
      B%nvg = sia(ns + 1)
    endif
    if (max(B%nvd, B%nvo, B%nvg) >= int(huge(0), 8)) then
      write(*,'(A)') "[Physics PC]   FATAL: a GMG level's value array exceeds the 32-bit map index."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    nall = B%nrow + B%ngh
    nbad = 0
    B%kl = 0; B%ku = 0
    if (allocated(B%zp)) deallocate(B%zp)
    allocate(B%zp(nall + 1))
    B%zp = 0
    do pass = 1, 2
      nva = 0; nvg = 0
      do R = 0, nall - 1
        bb = B%bid(R + 1)
        kz = 0
        if (R < B%nrow) then
          do k = dia(R + 1), dia(R + 2) - 1
            call visit(int(dja(k + 1)), int(k) + 1, .false.)
          enddo
          if (has_o) then
            do k = oia(R + 1), oia(R + 2) - 1
              call visit(comb(garr(oja(k + 1) + 1)), -(int(k) + 1), .false.)
            enddo
          endif
        else
          do k = sia(R - B%nrow + 1), sia(R - B%nrow + 2) - 1
            call visit(comb(sja(k + 1)), int(k) + 1, .true.)
          enddo
        endif
      enddo
      if (pass == 2) exit
      if (nbad > 0) write(*,'(A,I0,A)') "[Physics PC]   GMG WARNING: ", nbad, &
        " ring block(s) couple beyond J +- 1: dense LU instead of the ring solver"
      ! storage: banded where the band is narrow, dense otherwise
      tot = 0
      do bb = 1, B%nb
        n = B%sz(bb)
        B%band(bb) = (2 * (2 * B%kl(bb) + B%ku(bb) + 1) < n)
        B%loff(bb) = tot
        if (B%axblk(bb) .or. B%btd(bb)) B%band(bb) = .false.
        tot = tot + blk_size(B, bb)
      enddo
      if (allocated(B%lu)) deallocate(B%lu)
      allocate(B%lu(tot))
      B%zp(1) = 1
      do R = 1, nall
        B%zp(R + 1) = B%zp(R + 1) + B%zp(R)
      enddo
      if (allocated(B%zc)) deallocate(B%zc, B%zv)
      allocate(B%zc(B%zp(nall + 1) - 1), B%zv(B%zp(nall + 1) - 1))
      if (allocated(B%vs)) deallocate(B%vs, B%vd, B%gs, B%gd)
      allocate(B%vs(nva), B%vd(nva), B%gs(nvg), B%gd(nvg))
    enddo
    call put_ij(Ad, .false., nd, dia, dja)
    if (has_o) call put_ij(Ao, .false., no, oia, oja)
    if (B%ovl) call put_ij(B%sg(1), .false., ns, sia, sja)

  contains

    !> entry (R, C) with source src: into the zebra CSR, into the block, or nowhere
    subroutine visit(C, src, ghost)
      integer, intent(in) :: C, src
      logical, intent(in) :: ghost
      integer :: cb, pr, pc
      if (C < 0) return                           ! outside the rank's rows and ghosts
      cb = B%bid(C + 1)
      if ((sm_type == 7 .or. sm_type == 8) .and. B%bcol(bb) == 1 .and. B%bcol(cb) == 0) then
        if (pass == 1) then
          B%zp(R + 2) = B%zp(R + 2) + 1
        else
          B%zc(B%zp(R + 1) + kz) = C
        endif
        call put(src, -int(B%zp(R + 1) + kz, 8), ghost)
        kz = kz + 1
      else if (cb == bb) then
        pr = B%pos(R + 1); pc = B%pos(C + 1)
        if (pass == 1) then
          B%kl(bb) = max(B%kl(bb), pr - pc)
          B%ku(bb) = max(B%ku(bb), pc - pr)
          if (B%btd(bb)) then
            ! a ring solved as a block-tridiagonal chain: J +- 1 couplings only
            if (btd_index(B%rg(bb), B%qn(B%off(bb) + pr), 1, B%qn(B%off(bb) + pc), 1) == 0) then
              B%btd(bb) = .false.
              nbad = nbad + 1
            endif
          endif
        endif
        if (.not. B%axblk(bb)) then
          if (pass == 1) then
            call put(src, 0_8, ghost)
          else
            call put(src, lu_index(B, bb, pr, pc), ghost)
          endif
        endif
      endif
    end subroutine visit

    subroutine put(src, dst, ghost)
      integer, intent(in)    :: src
      integer(8), intent(in) :: dst
      logical, intent(in)    :: ghost
      if (ghost) then
        nvg = nvg + 1
        if (pass == 2) then
          B%gs(nvg) = src; B%gd(nvg) = dst
        endif
      else
        nva = nva + 1
        if (pass == 2) then
          B%vs(nva) = src; B%vd(nva) = dst
        endif
      endif
    end subroutine put

    !> combined row index of global column gc, or -1
    integer function comb(gc)
      PetscInt, intent(in) :: gc
      integer :: lo, hi, mid
      comb = -1
      if (gc >= rst .and. gc < ren) then
        comb = int(gc - rst)
        return
      endif
      lo = 1; hi = B%ngh
      do while (lo <= hi)
        mid = (lo + hi) / 2
        if (B%gidx(mid) == gc) then
          comb = B%nrow + mid - 1
          return
        else if (B%gidx(mid) < gc) then
          lo = mid + 1
        else
          hi = mid - 1
        endif
      enddo
    end function comb
  end subroutine blk_pattern

  !> Numeric part of build_blocks: the blocks' values (and the zebra
  !! couplings) gathered from A and sg(1) through the value maps.
  subroutine blk_fill(B, A)
    type(blk_t), intent(inout) :: B
    Mat, intent(in) :: A
    Mat :: Ad, Ao
    PetscInt, pointer :: garr(:)
    logical :: has_o
    type(c_ptr) :: pd, po, ps
    real(c_double), pointer :: vd_(:), vo_(:), vg_(:)
    integer :: k
    integer(8) :: d
    real*8 :: v

    call aij_parts(A, Ad, Ao, garr, has_o)
    call aij_vals_read(Ad, B%nvd, pd, vd_)
    if (has_o) then
      call aij_vals_read(Ao, B%nvo, po, vo_)
    else
      vo_ => vd_(1:0)
    endif
    if (B%ovl) then
      call aij_vals_read(B%sg(1), B%nvg, ps, vg_)
    else
      vg_ => vd_(1:0)
    endif
    ! zeroed block by block on the threads: GBs with all slots per block, and
    ! the pages' first touch then spreads over the threads' memory
    !$omp parallel do schedule(static)
    do k = 1, B%nb
      B%lu(B%loff(k) + 1 : B%loff(k) + blk_size(B, k)) = 0.0d0
    enddo
    !$omp end parallel do
    !$omp parallel do private(v, d) schedule(static)
    do k = 1, size(B%vs)
      if (B%vs(k) > 0) then
        v = vd_(B%vs(k))
      else
        v = vo_(-B%vs(k))
      endif
      d = B%vd(k)
      if (d > 0) then
        B%lu(d) = v
      else
        B%zv(-d) = v
      endif
    enddo
    !$omp end parallel do
    !$omp parallel do private(v, d) schedule(static)
    do k = 1, size(B%gs)
      v = vg_(B%gs(k))
      d = B%gd(k)
      if (d > 0) then
        B%lu(d) = v
      else
        B%zv(-d) = v
      endif
    enddo
    !$omp end parallel do
    call aij_vals_done(Ad, pd)
    if (has_o) call aij_vals_done(Ao, po)
    if (B%ovl) call aij_vals_done(B%sg(1), ps)
  end subroutine blk_fill

  !> Mesh part of the line overlap (first build only; see blk_t). The ghost
  !! rows of a local line segment (J, slot m, rings Ia..Ib) are the rows of
  !! the nodes (I, J), I in Ia-lovl..Ia-1 or Ib+1..Ib+lovl outside the axis
  !! block, that other ranks own. They are found by a breadth-first walk of
  !! A's graph from the rank's off-diagonal columns, lovl steps deep, each
  !! step reading the candidates' (I, J, m) from their owners through a
  !! scatter of a code vector; then they are inserted into their blocks in
  !! line order (rings, then global index, the owners' own row order), and
  !! the ghost scatter is built. Collective on A's communicator.
  subroutine ovl_extend(g, A, B, rst, ren, depth, rings)
    use mod_petsc_pc_harm, only: pc_ntor
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    PetscInt, intent(in) :: rst, ren
    integer, intent(in) :: depth        !< nodes to extend by (lovl, or rovl for rings)
    logical, intent(in) :: rings        !< ring blocks: extend along J within the ring
    PetscInt, parameter :: one = 1, zero = 0
    Mat :: Ad, Ao
    Mat, pointer :: sub(:)
    PetscInt, pointer :: garr(:), sia(:), sja(:)
    PetscInt :: nglob, ns, nn, k
    PetscScalar, pointer :: kp(:), cp(:)
    Vec :: kv, cv
    IS  :: isc, isn
    VecScatter :: sc
    PetscErrorCode :: ierr
    logical :: has_o
    integer :: nc, nr, r, bb, I, J, m, ilim, d, q, t, ngh, nlo, mpierr, code, nkey, qtot
    integer :: gtot(2), gloc(2), dmax, ninc, ncd
    integer, allocatable :: Ia(:), Ib(:), blkof(:), gb(:), gI(:), nsz(:), noff(:), nrows(:), fill(:)
    integer, allocatable :: tb(:), tI(:), tJ(:), gJ(:), lst(:), rk(:)
    PetscInt, allocatable :: cand(:), gl(:), tl(:), perm(:)

    nr = B%nrow
    if (g == 0) then
      nc = nth0
    else
      nc = glv(g)%nj
    endif
    ! an X-point grid's lines are its virtual lattice's columns (node_of
    ! returns the column as J there), which can outnumber the ring positions
    ncd = nc
    if (xgrid .and. .not. rings) then
      do r = 1, nr
        ncd = max(ncd, row_col(g, r, 0) + 1)
      enddo
      call MPI_Allreduce(MPI_IN_PLACE, ncd, 1, MPI_INTEGER, MPI_MAX, gcomm, mpierr)
    endif
    ilim = 0
    if (axis_k /= 0) ilim = axis_lim(g)
    ! the local segments, and the block of every (line or ring, slot): a line
    ! is keyed by J, a ring by I; every slot of the block is registered (a
    ! block holds a harmonic's cos and sin slots, with harm_pair 2 all slots)
    nkey = ncd
    if (rings) then
      nkey = 1
      do r = 1, nr
        call node_of(r, I, J, m)
        nkey = max(nkey, I + 1)
      enddo
    endif
    allocate(Ia(B%nb), Ib(B%nb), blkof(0:nkey * pc_ntor - 1))
    Ia = -1; Ib = -1; blkof = 0
    do bb = 1, B%nb
      if (B%axblk(bb)) cycle
      call node_of(B%rows(B%off(bb) + 1) + 1, I, J, m)
      if (I <= ilim) cycle                           ! the per-slot I = 0 blocks
      if (sm_type == 8 .and. I < ring_lim(g)) cycle  ! ring blocks: whole rings, on one rank
      Ia(bb) = I
      if (rings) Ia(bb) = huge(1)                    ! ring ghosts all go before the own rows
      do q = 1, B%sz(bb)
        call node_of(B%rows(B%off(bb) + q) + 1, I, J, m)
        if (.not. rings) Ib(bb) = max(Ib(bb), I)
        if (rings) then
          blkof(I * pc_ntor + m) = bb
        else
          blkof(J * pc_ntor + m) = bb
        endif
      enddo
    enddo
    ! (I, J, m) of every owned row, as a vector the walk can read remotely
    call MatCreateVecs(A, kv, PETSC_NULL_VEC, ierr)
    call VecGetArray(kv, kp, ierr)
    do r = 1, nr
      call node_of(r, I, J, m)
      kp(r) = dble((I * ncd + J) * pc_ntor + m)
    enddo
    call VecRestoreArray(kv, kp, ierr)
    call MatGetSize(A, nglob, PETSC_NULL_INTEGER, ierr)
    call ISCreateStride(PETSC_COMM_SELF, nglob, zero, one, B%isall, ierr)

    ! step 1: the off-diagonal columns of the rank's rows (sorted, unique)
    call aij_parts(A, Ad, Ao, garr, has_o)
    nn = 0
    if (has_o .and. associated(garr)) nn = size(garr)
    allocate(cand(nn))
    if (nn > 0) cand = garr
    allocate(gl(0), gb(0), gI(0), gJ(0))
    ! A ring is completed whole: a rank holding part of a ring and no whole
    ! neighbouring ring (a coarse level, whose rings lie 2^g fine rings apart,
    ! or < 2 rings per rank) reaches the rest only along J, one node per step.
    dmax = depth
    if (rings) dmax = max(depth, nc)
    do d = 1, dmax
      ! drop the candidates already accepted
      allocate(tl(nn))
      ns = 0
      do k = 1, nn
        if (find(gl, cand(k)) > 0) cycle
        ns = ns + 1; tl(ns) = cand(k)
      enddo
      ! their (I, J, m) from the owners
      call ISCreateGeneral(PETSC_COMM_SELF, ns, tl(1:ns), PETSC_COPY_VALUES, isc, ierr)
      call VecCreateSeq(PETSC_COMM_SELF, ns, cv, ierr)
      call VecScatterCreate(kv, isc, cv, PETSC_NULL_IS, sc, ierr)
      call VecScatterBegin(sc, kv, cv, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd(sc, kv, cv, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecGetArrayRead(cv, cp, ierr)
      allocate(tb(ns), tI(ns), tJ(ns))
      q = 0
      do k = 1, ns
        code = nint(cp(k))
        m = mod(code, pc_ntor); J = mod(code / pc_ntor, ncd); I = code / pc_ntor / ncd
        if (I <= ilim) cycle
        if (rings) then
          ! same ring and slot as a local ring block; the walk bounds the reach
          if (I >= nkey) cycle
          bb = blkof(I * pc_ntor + m)
          if (bb == 0) cycle
        else
          bb = blkof(J * pc_ntor + m)
          if (bb == 0) cycle
          ! X-point grid: a column's ring indices jump at the separatrix (core
          ! | private | SOL numbering), so the walk depth alone bounds the reach
          if (.not. xgrid .and. .not. ((I < Ia(bb) .and. Ia(bb) - I <= lovl) .or. (I > Ib(bb) .and. I - Ib(bb) <= lovl))) cycle
        endif
        q = q + 1
        tl(q) = tl(k); tb(q) = bb; tI(q) = I; tJ(q) = J
      enddo
      call VecRestoreArrayRead(cv, cp, ierr)
      call VecScatterDestroy(sc, ierr); call VecDestroy(cv, ierr); call ISDestroy(isc, ierr)
      gl = [gl, tl(1:q)]; gb = [gb, tb(1:q)]; gI = [gI, tI(1:q)]; gJ = [gJ, tJ(1:q)]
      call sort_ghosts()
      ! no rank found a new ghost: the walk is complete (all ranks agree)
      call MPI_Allreduce(q, qtot, 1, MPI_INTEGER, MPI_SUM, gcomm, mpierr)
      if (qtot == 0) then
        deallocate(tl, tb, tI, tJ)
        exit
      endif
      ! next step's candidates: the off-rank columns of the rows just accepted
      if (d < dmax) then
        call ISCreateGeneral(PETSC_COMM_SELF, int(q, kind(nn)), tl(1:q), PETSC_COPY_VALUES, isn, ierr)
        call MatCreateSubMatrices(A, one, [isn], [B%isall], MAT_INITIAL_MATRIX, sub, ierr)
        call get_ij(sub(1), .false., ns, sia, sja)
        deallocate(cand)
        allocate(cand(sia(ns + 1)))
        nn = 0
        do k = 1, sia(ns + 1)
          if (sja(k) >= rst .and. sja(k) < ren) cycle
          nn = nn + 1; cand(nn) = sja(k)
        enddo
        call put_ij(sub(1), .false., ns, sia, sja)
        call MatDestroySubMatrices(one, sub, ierr)
        call ISDestroy(isn, ierr)
        call PetscSortRemoveDupsInt(nn, cand, ierr)
      endif
      deallocate(tl, tb, tI, tJ)
    enddo
    ngh = size(gl)

    ! insert the ghosts into their blocks: below the segment, own rows, above;
    ! each side by ring, then global index
    allocate(nsz(B%nb), noff(B%nb), fill(B%nb), nrows(nr + ngh))
    nsz = B%sz
    do k = 1, ngh
      nsz(gb(k)) = nsz(gb(k)) + 1
    enddo
    if (B%nb > 0) noff(1) = 0
    do bb = 2, B%nb
      noff(bb) = noff(bb - 1) + nsz(bb - 1)
    enddo
    allocate(lst(ngh))
    fill = 0
    do bb = 1, B%nb                                     ! ghosts of block bb, ascending k
      if (nsz(bb) == B%sz(bb)) cycle
      nlo = 0
      q = 0
      do k = 1, ngh
        if (gb(k) /= bb) cycle
        q = q + 1; lst(q) = int(k)
      enddo
      do t = 2, q                                       ! stable by ring
        r = lst(t); m = t - 1
        do while (m >= 1)
          if (gI(lst(m)) <= gI(r)) exit
          lst(m + 1) = lst(m); m = m - 1
        enddo
        lst(m + 1) = r
      enddo
      do t = 1, q
        if (gI(lst(t)) < Ia(bb)) nlo = nlo + 1
      enddo
      do t = 1, nlo
        nrows(noff(bb) + t) = nr + lst(t) - 1
      enddo
      nrows(noff(bb) + nlo + 1 : noff(bb) + nlo + B%sz(bb)) = B%rows(B%off(bb) + 1 : B%off(bb) + B%sz(bb))
      do t = nlo + 1, q
        nrows(noff(bb) + B%sz(bb) + t) = nr + lst(t) - 1
      enddo
    enddo
    do bb = 1, B%nb
      if (nsz(bb) == B%sz(bb)) nrows(noff(bb) + 1 : noff(bb) + nsz(bb)) = B%rows(B%off(bb) + 1 : B%off(bb) + B%sz(bb))
    enddo
    ! A completed ring back in build_blocks' periodic order (J = 0, nc-1, 1,
    ! nc-2, ...), own rows and ghosts together: in the order above (ghosts
    ! first) its J +- 1 couplings span the ring, and a ring cut by a rank
    ! boundary was factored dense -- at 321x128, 4096 rows per cos/sin block.
    if (rings) then
      allocate(rk(nr + ngh))
      do r = 1, nr + ngh
        if (r <= nr) then
          call node_of(r, I, J, m)
        else
          I = gI(r - nr); J = gJ(r - nr)
        endif
        rk(r) = rkey(g, I, J, nc)
      enddo
      do bb = 1, B%nb
        if (nsz(bb) == B%sz(bb)) cycle
        call sort_rows_by_key(nrows(noff(bb) + 1 : noff(bb) + nsz(bb)), rk)
      enddo
      ! every ring block must now span its ring (a cut ring leaves the
      ! poloidal Alfven coupling across the cut outside the block)
      ninc = 0
      block
        logical :: seen(0:nc - 1)
        integer :: nring
        do bb = 1, B%nb
          if (B%axblk(bb) .or. Ia(bb) < 0) cycle
          seen = .false.
          nring = nc
          do q = 1, nsz(bb)
            r = nrows(noff(bb) + q) + 1
            if (r <= nr) then
              call node_of(r, I, J, m)
            else
              I = gI(r - nr); J = gJ(r - nr)
            endif
            if (xgrid) nring = xrg(g)%nn(I)               ! an X-point grid's ring: its own length
            seen(J) = .true.
          enddo
          if (.not. all(seen(0:nring - 1))) ninc = ninc + 1
        enddo
      end block
      call MPI_Allreduce(MPI_IN_PLACE, ninc, 1, MPI_INTEGER, MPI_SUM, gcomm, mpierr)
      if (gme == 0 .and. ninc > 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG WARNING: ", ninc, &
        " ring block(s) on level ", g, " still cut by a rank boundary"
      deallocate(rk)
    endif
    call move_alloc(nrows, B%rows)
    B%sz = nsz; B%off = noff
    deallocate(B%pos, B%piv)
    allocate(B%pos(nr + ngh), B%piv(nr + ngh))
    B%bid = [B%bid(1:nr), gb]
    do bb = 1, B%nb
      do q = 1, B%sz(bb)
        B%pos(B%rows(B%off(bb) + q) + 1) = q
      enddo
    enddo

    ! the ghost scatter (neighbour ranks only)
    B%ngh = ngh
    allocate(B%gidx(ngh), B%yg(ngh))
    B%gidx = gl
    if (allocated(B%ghJ)) deallocate(B%ghJ)
    allocate(B%ghJ(ngh))
    B%ghJ = gJ
    call ISCreateGeneral(PETSC_COMM_SELF, int(ngh, kind(nn)), gl, PETSC_COPY_VALUES, B%isg, ierr)
    call VecCreateSeq(PETSC_COMM_SELF, int(ngh, kind(nn)), B%xg, ierr)
    call VecScatterCreate(kv, B%isg, B%xg, PETSC_NULL_IS, B%sct, ierr)
    call VecDestroy(kv, ierr)
    B%ovl = .true.
    gloc = [ngh, nr]
    call MPI_Allreduce(gloc, gtot, 2, MPI_INTEGER, MPI_SUM, gcomm, mpierr)
    if (gme == 0 .and. g == 0 .and. rings) write(*,'(A,I0,A,F5.1,A)') &
      "[Physics PC]   GMG rings completed across rank cuts: ", gtot(1), &
      " ghost rows on level 0 (", 100.d0 * gtot(1) / max(gtot(2), 1), "% of the rows)"
    if (gme == 0 .and. g == 0 .and. .not. rings) write(*,'(A,I0,A,I0,A,F5.1,A)') &
      "[Physics PC]   GMG line overlap ", depth, &
      " node(s): ", gtot(1), " ghost rows on level 0 (", 100.d0 * gtot(1) / max(gtot(2), 1), "% of the rows)"

  contains

    subroutine node_of(r_, I_, J_, m_)
      integer, intent(in)  :: r_
      integer, intent(out) :: I_, J_, m_
      if (g == 0) then
        I_ = fine_node(r_) / nth0; J_ = mod(fine_node(r_), nth0); m_ = fine_harm(r_)
      else
        I_ = glv(g)%rnode(r_) / nc; J_ = mod(glv(g)%rnode(r_), nc); m_ = glv(g)%rharm(r_)
      endif
      if (xgrid .and. .not. rings) J_ = row_col(g, r_, J_)
    end subroutine node_of

    !> position of x in the sorted list l (1-based), or 0
    integer function find(l, x)
      PetscInt, intent(in) :: l(:), x
      integer :: lo, hi, mid
      find = 0
      lo = 1; hi = size(l)
      do while (lo <= hi)
        mid = (lo + hi) / 2
        if (l(mid) == x) then
          find = mid
          return
        else if (l(mid) < x) then
          lo = mid + 1
        else
          hi = mid - 1
        endif
      enddo
    end function find

    !> gl ascending, gb/gI permuted along
    subroutine sort_ghosts()
      integer :: n_
      PetscCount :: nn_
      n_ = size(gl)
      nn_ = n_
      if (allocated(perm)) deallocate(perm)
      allocate(perm(n_))
      do k = 1, n_
        perm(k) = k
      enddo
      call PetscSortIntWithArray(nn_, gl, perm, ierr)
      gb = gb(perm); gI = gI(perm); gJ = gJ(perm)
    end subroutine sort_ghosts
  end subroutine ovl_extend

  !> The sector factorisations the level loop left pending (rebuilds): every
  !! level's sector ranks work at the same time. Levels in ascending order on
  !! every rank, so the gathers among a level's sector ranks cannot deadlock.
  subroutine axd_factor_pending()
    integer :: g, kk
    PetscErrorCode :: ierr
    call PetscLogEventBegin(gev_axnum, ierr)
    do g = 0, nlev - 2
      if (gBk(g)%axpend) then
        do kk = 1, size(gBk(g)%axd)
          call axd_factor(gBk(g)%axd(kk))
        enddo
      endif
    enddo
    call PetscLogEventEnd(gev_axnum, ierr)
    ! a failed sector factorisation (zero pivot) on any rank: that level's
    ! block goes to the LU. Every rank walks the same levels (axpend agrees).
    do g = 0, nlev - 2
      if (gBk(g)%axpend) call check(gBk(g))
    enddo
    ! first build: the gates, and the LU where a sector solve failed its gate
    do g = 0, nlev - 2
      if (gBk(g)%axnew) call finish(gBk(g))
    enddo

  contains

    subroutine check(B)
      type(blk_t), intent(inout) :: B
      logical :: bad
      integer :: mpierr
      B%axpend = .false.
      bad = any(B%axd(:)%failed)
      call MPI_Allreduce(MPI_IN_PLACE, bad, 1, MPI_LOGICAL, MPI_LOR, gcomm, mpierr)
      if (.not. bad) return
      if (gme == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, " level ", g, &
        ": axis J-sector factorisation failed (zero pivot), LU used"
      B%axdon = .false.
      B%axnew = .false.                          ! first build: nothing left to gate
      call ax_lu(g, B%axA, B)
    end subroutine check

    subroutine finish(B)
      type(blk_t), intent(inout) :: B
      B%axnew = .false.
      call axd_gate(g, B%axA, B)
      if (.not. B%axdon) call ax_lu(g, B%axA, B)
    end subroutine finish
  end subroutine axd_factor_pending

  !> The LU of the whole axis block of level g (rds_t): where no sector
  !! solve is in use. First build and every rebuild.
  subroutine ax_lu(g, A, B)
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    character(len=24) :: axname
    character(len=64) :: axtag
    write(axname, '(A,I0)') "axblk", g
    write(axtag, '(A,I0)') "axis block level ", g
    call rds_setup(B%axg(1), A, trim(axname), trim(axtag))
  end subroutine ax_lu

  !> First sector rank of level g: the levels' sector sets follow each other
  !! over the ranks (wrapping at np), so their factorisations overlap in time.
  integer function axd_offset(g, gnp)
    integer, intent(in) :: g, gnp
    integer :: h, nc, ns
    axd_offset = 0
    do h = 0, g - 1
      if (h == 0) then
        nc = nth0
      else
        nc = glv(h)%nj
      endif
      ns = axsec
      if (ns < 0) ns = axd_nsec(nc, gnp)
      ns = min(ns, gnp, nc / 4)
      if (ns >= 2) axd_offset = axd_offset + ns
    enddo
    axd_offset = modulo(axd_offset, gnp)
  end function axd_offset

  !> First build: the structure of the J-sector solve of level g's axis
  !! groups (axd_setup). If every group is a chain the method applies to, the
  !! factorisation and the gate are left pending (B%axnew); otherwise the
  !! caller builds the LU. Collective on gcomm.
  subroutine axd_prepare(g, A, B)
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    integer :: kk, q, r, nsec, gnp, mpierr, nc
    integer, allocatable :: jl(:)
    character(len=64) :: axtag

    call MPI_Comm_size(gcomm, gnp, mpierr)
    if (g == 0) then
      nc = nth0
    else
      nc = glv(g)%nj
    endif
    nsec = axsec
    if (nsec < 0) nsec = axd_nsec(nc, gnp)
    nsec = min(nsec, gnp, nc / 4)
    if (nsec < 2) return
    allocate(B%axd(size(B%axg)))
    do kk = 1, size(B%axg)
      allocate(jl(size(B%axg(kk)%loc)))
      do q = 1, size(jl)
        r = B%axg(kk)%loc(q) + 1
        if (g == 0) then
          jl(q) = mod(fine_node(r), nth0)
        else
          jl(q) = mod(glv(g)%rnode(r), nc)
        endif
      enddo
      write(axtag, '(A,I0,A,I0,A,I0)') "GMG", cur_inst, " axis block level ", g, " group ", kk - 1
      call axd_setup(B%axd(kk), A, B%axg(kk)%loc, jl, nc, nsec, gcomm, trim(axtag), axd_offset(g, gnp))
      deallocate(jl)
      if (.not. B%axd(kk)%on) return
    enddo
    B%axnew = .true.; B%axpend = .true.
    B%axA = A
  end subroutine axd_prepare

  !> First build, after the factorisations: the gate of level g's sector
  !! solves, their row-wise backward error for a random right-hand side on
  !! each group's rows. The sector solve is used (for all groups, or none) if
  !! that is below 1e-8. Not the error: the blocks are ill-conditioned (a
  !! manufactured solution comes back to 1e-4 only). No LU of the whole
  !! block is needed for the gate: that factorisation (sequential, on the
  !! ranks owning the axis rows) was the first build's critical path.
  !! Collective on gcomm.
  subroutine axd_gate(g, A, B)
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    integer :: kk, q, mpierr
    logical, allocatable :: ing(:)
    character(len=64) :: axtag
    Vec :: x, y1, y2
    PetscScalar, pointer :: yp(:), xp(:), vals(:)
    PetscInt :: rst, ren, ncols
    real*8 :: dn, yn, sc
    logical :: pass
    PetscErrorCode :: ierr

    pass = .true.
    do kk = 1, size(B%axg)
      write(axtag, '(A,I0,A,I0,A,I0)') "GMG", cur_inst, " axis block level ", g, " group ", kk - 1
      ! random b on the group's rows, y = sector solve, residual b - A y there
      ! (y is zero outside the group, so A y on its rows is the block times y)
      call MatCreateVecs(A, y2, x, ierr)
      call VecDuplicate(x, y1, ierr)
      call VecSetRandom(x, PETSC_NULL_RANDOM, ierr)
      allocate(ing(B%nrow))
      ing = .false.
      ing(B%axg(kk)%loc + 1) = .true.
      call VecGetArray(x, yp, ierr)
      do q = 1, B%nrow
        if (.not. ing(q)) yp(q) = 0.0d0
      enddo
      call VecRestoreArray(x, yp, ierr)
      call VecZeroEntries(y2, ierr)
      call axd_solve(B%axd(kk), x, y2)
      call MatMult(A, y2, y1, ierr)
      call VecNorm(y2, NORM_INFINITY, yn, ierr)
      ! row-wise backward error max_i |r_i| / (max_j |a_ij| |y|_inf + |b_i|):
      ! the rows' scales differ by decades (boundary rows), so a plain
      ! residual norm would measure the largest rows' round-off only
      call MatGetOwnershipRange(A, rst, ren, ierr)
      call VecGetArrayRead(y1, yp, ierr)
      call VecGetArrayRead(x, xp, ierr)
      dn = 0.0d0
      do q = 1, B%nrow
        if (.not. ing(q)) cycle
        call MatGetRow(A, rst + q - 1, ncols, PETSC_NULL_INTEGER_POINTER, vals, ierr)
        sc = maxval(abs(vals(1:ncols)))
        call MatRestoreRow(A, rst + q - 1, ncols, PETSC_NULL_INTEGER_POINTER, vals, ierr)
        dn = max(dn, abs(xp(q) - yp(q)) / max(sc * yn + abs(xp(q)), 1.d-300))
      enddo
      call VecRestoreArrayRead(x, xp, ierr)
      call VecRestoreArrayRead(y1, yp, ierr)
      deallocate(ing)
      call MPI_Allreduce(MPI_IN_PLACE, dn, 1, MPI_DOUBLE_PRECISION, MPI_MAX, gcomm, mpierr)
      call VecDestroy(x, ierr); call VecDestroy(y1, ierr); call VecDestroy(y2, ierr)
      if (gme == 0) write(*,'(A,A,A,ES9.2,A)') "[Physics PC]   ", trim(axtag), &
        ": J-sector solve backward error ", dn, merge(" -> in use ", " -> LU used", dn <= 1.d-8)
      if (.not. (dn <= 1.d-8)) pass = .false.        ! a NaN fails too
    enddo
    B%axdon = pass
  end subroutine axd_gate

  !> Whole rings per rank: ring i of w(i) DOFs (rings in radial order, empty
  !! ones skipped) -> rank part(i). At least np rings: np contiguous parts
  !! with the least possible maximum, every rank at least one ring (binary
  !! search on the cap, then greedy). Fewer rings: one per rank, spread
  !! evenly over the ranks (and so over the nodes' memory bandwidth).
  subroutine ring_parts(w, np, part)
    integer, intent(in)  :: w(:), np
    integer, intent(out) :: part(:)
    integer, allocatable :: wc(:), ic(:)
    integer :: nr, k, lo, hi, cap, r, s, first, nparts
    nr = count(w > 0)
    allocate(wc(nr), ic(nr))
    nr = 0
    do k = 1, size(w)
      if (w(k) <= 0) cycle
      nr = nr + 1; wc(nr) = w(k); ic(nr) = k
    enddo
    part = 0
    if (nr == 0) return
    if (nr < np) then
      do k = 1, nr
        part(ic(k)) = int((int(k - 1, 8) * np) / nr)
      enddo
    else
      lo = maxval(wc); hi = sum(wc)
      do while (lo < hi)
        cap = (lo + hi) / 2
        nparts = 1; s = 0
        do k = 1, nr
          if (s + wc(k) > cap) then
            nparts = nparts + 1; s = wc(k)
          else
            s = s + wc(k)
          endif
        enddo
        if (nparts <= np) then
          hi = cap
        else
          lo = cap + 1
        endif
      enddo
      cap = lo
      k = 1
      do r = 0, np - 1
        s = 0; first = k
        do while (k <= nr)
          if (r < np - 1) then
            if (k > first .and. s + wc(k) > cap) exit
            if (k > first .and. nr - k < np - 1 - r) exit   ! one ring left for each later rank
          endif
          part(ic(k)) = r
          s = s + wc(k); k = k + 1
        enddo
      enddo
    endif
    do k = 2, size(w)                    ! empty rings: no rows, any rank
      if (w(k) <= 0) part(k) = part(k - 1)
    enddo
  end subroutine ring_parts

  !> Coarse layout rebalancing (crebal): keep the whole-ring owners own_new
  !! (else back to own_old, the owners inherited from the fine nodes) only if
  !! they lower the busiest rank's ring-solve load: the DOFs of every ring
  !! it holds a part of, since each rank sharing a ring solves all of it;
  !! on a tie, if they lower its row count (the matvec). DOF d's ring: rid(d).
  subroutine crebal_pick(g, np, rid, own_old, own_new)
    integer, intent(in) :: g, np, rid(0:), own_old(0:)
    integer, intent(inout) :: own_new(0:)
    integer :: lo, ln, ro, rn
    logical :: take
    call ring_load(own_old, lo, ro)
    call ring_load(own_new, ln, rn)
    take = (ln < lo) .or. (ln == lo .and. rn < ro)
    if (.not. take) own_new = own_old
    if (gme == 0) write(*,'(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,A)') "[Physics PC]   GMG", cur_inst, &
      ": level ", g, " whole rings per rank: busiest ring-solve load ", lo, " -> ", ln, &
      " DOFs, busiest rows ", ro, " -> ", rn, merge(" (applied)", " (kept)   ", take)
  contains
    subroutine ring_load(own, ld, rows)
      integer, intent(in) :: own(0:)
      integer, intent(out) :: ld, rows
      integer, allocatable :: w(:), cnt(:)
      logical, allocatable :: hit(:,:)
      integer :: d, r
      allocate(w(0:maxval(rid)), cnt(0:np - 1), hit(0:maxval(rid), 0:np - 1))
      w = 0; cnt = 0; hit = .false.
      do d = 0, size(rid) - 1
        w(rid(d)) = w(rid(d)) + 1
        cnt(own(d)) = cnt(own(d)) + 1
        hit(rid(d), own(d)) = .true.
      enddo
      ld = 0
      do r = 0, np - 1
        ld = max(ld, sum(w, mask=hit(:, r)))
      enddo
      rows = maxval(cnt)
    end subroutine ring_load
  end subroutine crebal_pick

  !> In-place ascending heapsort, O(n log n) (the setup's medians; an
  !! insertion sort there was O(n^2) in the cell count).
  subroutine sort_real(a)
    real*8, intent(inout) :: a(:)
    integer :: n, i, e
    real*8  :: t
    n = size(a)
    do i = n / 2, 1, -1
      call sift(i, n)
    enddo
    do e = n, 2, -1
      t = a(1); a(1) = a(e); a(e) = t
      call sift(1, e - 1)
    enddo
  contains
    subroutine sift(i0, m)
      integer, intent(in) :: i0, m
      integer :: r, c
      real*8  :: v
      r = i0; v = a(r)
      do
        c = 2 * r
        if (c > m) exit
        if (c < m) then
          if (a(c + 1) > a(c)) c = c + 1
        endif
        if (a(c) <= v) exit
        a(r) = a(c); r = c
      enddo
      a(r) = v
    end subroutine sift
  end subroutine sort_real

  !> Stable sort of the 0-based rows r(:) by key(r + 1): a bottom-up merge
  !! sort, O(n log n) (a radial line has n_flux x 4 x fields rows; the
  !! insertion sort it replaces was O(n^2) per line).
  subroutine sort_rows_by_key(r, key)
    integer, intent(inout) :: r(:)
    integer, intent(in)    :: key(:)
    integer, allocatable :: t(:)
    integer :: n, w, lo, mid, hi, i, j, k
    n = size(r)
    if (n < 2) return
    allocate(t(n))
    w = 1
    do while (w < n)
      lo = 1
      do while (lo <= n)
        mid = min(lo + w - 1, n); hi = min(lo + 2 * w - 1, n)
        i = lo; j = mid + 1; k = lo
        do while (i <= mid .and. j <= hi)
          if (key(r(j) + 1) < key(r(i) + 1)) then
            t(k) = r(j); j = j + 1
          else
            t(k) = r(i); i = i + 1
          endif
          k = k + 1
        enddo
        do while (i <= mid)
          t(k) = r(i); i = i + 1; k = k + 1
        enddo
        do while (j <= hi)
          t(k) = r(j); j = j + 1; k = k + 1
        enddo
        lo = lo + 2 * w
      enddo
      r = t
      w = 2 * w
    enddo
  end subroutine sort_rows_by_key

  !> x = A^-1 x for one right-hand side, A = the dgbtrf factors in LAPACK band
  !! storage (ldab = 2 kl + ku + 1, pivots ipiv): dgbtrs('N') written out.
  !! dgbtrs does one level-2 BLAS call (dger / dtbsv column step) per column on
  !! vectors of length ~kl, so for the GMG's radial lines (n ~ 10^2-10^3,
  !! kl ~ 15) the call overhead was most of GMG_Lines. Same operations in the
  !! same order: L with the row interchanges, then U by columns.
  subroutine band_solve(n, kl, ku, ab, ipiv, x)
    integer, intent(in)   :: n, kl, ku, ipiv(n)
    real*8, intent(in)    :: ab(2 * kl + ku + 1, n)
    real*8, intent(inout) :: x(n)
    integer :: j, i, l, lm, kd
    real*8  :: t
    kd = kl + ku + 1
    if (kl > 0) then
      do j = 1, n - 1
        lm = min(kl, n - j)
        l = ipiv(j)
        if (l /= j) then
          t = x(l); x(l) = x(j); x(j) = t
        endif
        t = x(j)
        do i = 1, lm
          x(j + i) = x(j + i) - ab(kd + i, j) * t
        enddo
      enddo
    endif
    do j = n, 1, -1
      if (x(j) /= 0.0d0) then
        x(j) = x(j) / ab(kd, j)
        t = x(j)
        do i = j - 1, max(1, j - kl - ku), -1
          x(i) = x(i) - t * ab(kd + i - j, j)
        enddo
      endif
    enddo
  end subroutine band_solve

  !> Factor storage of block bb (entries of lu): none for the sparse axis
  !! solve, else the ring solver's, LAPACK band (with dgbtrf's fill rows) or
  !! dense
  integer(8) function blk_size(B, bb)
    type(blk_t), intent(in) :: B
    integer, intent(in) :: bb
    if (B%axblk(bb)) then
      blk_size = 0
    else if (B%btd(bb)) then
      blk_size = B%rg(bb)%tot
    else if (B%band(bb)) then
      blk_size = int(2 * B%kl(bb) + B%ku(bb) + 1, 8) * B%sz(bb)
    else
      blk_size = int(B%sz(bb), 8)**2
    endif
  end function blk_size

  !> Storage index of entry (i, j) of block bb (dense column-major, LAPACK
  !! band storage with the kl extra rows dgbtrf needs for fill, or a ring
  !! solver's node blocks).
  integer(8) function lu_index(B, bb, i, j)
    type(blk_t), intent(in) :: B
    integer, intent(in) :: bb, i, j
    integer :: jr, jc
    if (B%btd(bb)) then
      jr = B%qn(B%off(bb) + i); jc = B%qn(B%off(bb) + j)
      lu_index = B%loff(bb) + btd_index(B%rg(bb), jr, i - B%rg(bb)%rp(jr), jc, j - B%rg(bb)%rp(jc))
    else if (B%band(bb)) then
      lu_index = B%loff(bb) + int(j - 1, 8) * (2 * B%kl(bb) + B%ku(bb) + 1) &
                 + (B%kl(bb) + B%ku(bb) + 1 + i - j)
    else
      lu_index = B%loff(bb) + int(j - 1, 8) * B%sz(bb) + i
    endif
  end function lu_index

  !> Block of the DOFs at grid point (I, J) of level g, toroidal slot m, for
  !! the active smoother. The axis ring (I = 0, one shared value DOF) is one
  !! block per slot; with axis_k /= 0 rings 0..axis_lim(g) are. 4 flux-surface
  !! ring (all J at one I), 5 / 7 radial line (all I > axis_k at one J), 8
  !! rings for I < ring_lim(g) and radial lines outside. Ids have gaps;
  !! build_blocks compresses them.
  integer function blk_id(g, I, J, m_)
    use mod_petsc_pc_harm, only: pc_ntor, pc_grp
    integer, intent(in) :: g, I, J, m_
    integer :: is_g, m
    m = m_
    ! harm_pair 1: slots 0 | 1, 2 | 3, 4 | ...: n = 0, then cos/sin pairs.
    ! harm_pair 2: every slot in one block. Cross-|n| couplings (a nonlinear
    ! or 3D equilibrium couples n to n' at O(1); a stellarator n to n +-
    ! k*n_coord_period through the geometry and grad chi in B.grad, at fine
    ! scales that no coarse level resolves) then lie inside the smoother's blocks.
    m = pc_grp(m_)
    if (hpair == 2) m = 0
    if (I == 0 .or. I <= axis_lim(g)) then
      blk_id = m + 1
    else if (sm_type == 4) then
      blk_id = pc_ntor + (I - 1) * pc_ntor + m + 1
    else if (sm_type == 5 .or. sm_type == 7) then
      blk_id = pc_ntor + J * pc_ntor + m + 1
    else                                        ! 8
      is_g = ring_lim(g)
      if (I < is_g) then
        blk_id = pc_ntor + (I - 1) * pc_ntor + m + 1
      else
        blk_id = pc_ntor + max(is_g - 1, 0) * pc_ntor + J * pc_ntor + m + 1
      endif
    endif
  end function blk_id

  !> Radial column of local row r (1-based) of level g: its J on the polar
  !! grid, the virtual lattice's column on an X-point grid.
  integer function row_col(g, r, J)
    integer, intent(in) :: g, r, J
    row_col = J
    if (.not. xgrid) return
    if (g == 0) then
      row_col = fine_col(r)
    else
      row_col = glv(g)%rcol(r)
    endif
  end function row_col

  !> Order key of position J on ring I of level g (nc: its nj): the band
  !! order of a closed ring (ring_key), the natural order on an X-point
  !! grid's open lines.
  integer function rkey(g, I, J, nc)
    integer, intent(in) :: g, I, J, nc
    rkey = ring_key(J, nc)
    if (.not. xgrid) return
    rkey = J
    if (xrg(g)%per(I)) rkey = ring_key(J, xrg(g)%nn(I))
  end function rkey

  !> Position of node J in a periodic ring's band order 0, nc-1, 1, nc-2, ...:
  !! the J +- 1 couplings, the wrap included, then lie within two nodes, a
  !! narrow band instead of a dense block.
  pure integer function ring_key(J, nc)
    integer, intent(in) :: J, nc
    if (J < nc - J) then
      ring_key = 2 * J
    else
      ring_key = 2 * (nc - 1 - J) + 1
    endif
  end function ring_key

  !> Switch ring of level g: coarse ring I sits at fine ring I*2^g, so the
  !! same physical radius as the fine ring_is (the ratio r*dtheta/dr does not
  !! change under standard coarsening).
  integer function ring_lim(g)
    integer, intent(in) :: g
    ring_lim = (ring_is + 2**g - 1) / 2**g
  end function ring_lim

  !> Last ring of the axis block on level g: axis_k (>= 0) on every level, or
  !! with axis_k < 0 the rings below the switch radius, 0..ring_lim(g)-1 (at
  !! least ring 1). The circle-dominated rings only help when they
  !! are solved TOGETHER with the pole; as separate ring blocks they hurt.
  integer function axis_lim(g)
    integer, intent(in) :: g
    if (axis_k >= 0) then
      axis_lim = axis_k
    else
      axis_lim = max(ring_lim(g) - 1, 1)
    endif
  end function axis_lim

  subroutine blk_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    type(blk_t), pointer :: B

    B => gBk(cur_lev)
    if (sm_type == 7 .or. sm_type == 8) then
      call zebra_solve(x, y)
    else
      ! axis first: it depends on x only, and the ranks enter here in step
      ! (GMRES's reductions just synchronised them), which a J-sector solve
      ! needs; after the lines it would wait for the slowest rank's lines
      if (B%axsparse) call ax_solve(x, y)
      if (allocated(B%wk)) then
        call lines_solve_par(x, y)
      else
        call lines_solve(x, y)
      endif
    endif
    ierr = 0

  contains

    !> w = block bb's solve of w (block order): LAPACK band / dense, or the
    !! ring solver (point Jacobi for a ring with a singular node block)
    subroutine one_solve(bb, w)
      integer, intent(in)   :: bb
      real*8, intent(inout) :: w(*)
      integer :: n, q, info
      external :: dgetrs
      n = B%sz(bb)
      if (B%btd(bb)) then
        if (B%jac(bb)) then
          do q = 1, n
            w(q) = w(q) / B%lu(lu_index(B, bb, q, q))
          enddo
        else if (rsp) then
          block
            real*4 :: w4(n)
            w4 = real(w(1:n) * B%eqs(B%off(bb) + 1 : B%off(bb) + n), 4)
            call btd_solve_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%piv(B%off(bb) + 1), w4)
            w(1:n) = w4 * B%eqs(B%off(bb) + 1 : B%off(bb) + n)
          end block
        else
          call btd_solve(B%rg(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), w)
        endif
      else if (B%band(bb)) then
        call band_solve(n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), w)
      else
        call dgetrs('N', n, 1, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), w, n, info)
      endif
    end subroutine one_solve

    !> hybrid on smoother 4: lines_solve with the rings' segments threaded
    !! across all the rank's rings (work items B%lb, B%lk), on the work vector
    !! B%wk in block order (its single copy wk4 with ring_single). A segment
    !! gathers its nodes (left separator and interior) before its forward
    !! sweep and scatters its interior after its backward sweep; a ring's
    !! separators are solved and scattered in between. Other blocks are
    !! solved whole in the first loop. One parallel region, three barriers.
    subroutine lines_solve_par(xx, yy)
      Vec :: xx, yy
      PetscScalar, pointer :: xp(:), yp(:), gp(:)
      PetscErrorCode :: ie
      integer :: bb, k, p
      ! B%rg(bb) is spelled out: an associate inside these OpenMP loops
      ! crashes ifx (internal compiler error)
      call PetscLogEventBegin(gev_lines(cur_inst), ie)
      call ghosts_in(xx, gp)
      call VecGetArrayRead(xx, xp, ie)
      call VecGetArray(yy, yp, ie)
      !$omp parallel private(bb, k, p) if (B%nrow >= LINES_OMP_MIN)
      !$omp do schedule(dynamic, 1)
      do p = 1, size(B%lb)
        bb = B%lb(p); k = B%lk(p)
        if (.not. reduced(bb)) then
          if (k > 0) cycle
          call gath(bb, 1, B%sz(bb), .false., xp, gp)
          call one_solve(bb, B%wk(B%off(bb) + 1))
          call scat(bb, 1, B%sz(bb), .false., yp)
          cycle
        endif
        call gath(bb, B%rg(bb)%rp(B%rg(bb)%sep(k)) + 1, B%rg(bb)%rp(hyb_end(B%rg(bb), k) + 1), rsp, xp, gp)
        if (rsp) then
          call hyb_sol_fwd_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%piv(B%off(bb) + 1), B%wk4(B%off(bb) + 1), k)
        else
          call hyb_sol_fwd(B%rg(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), B%wk(B%off(bb) + 1), k)
        endif
      enddo
      !$omp end do
      !$omp do schedule(dynamic, 1)
      do bb = 1, B%nb
        if (.not. reduced(bb)) cycle
        if (rsp) then
          call hyb_sol_sep_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%piv(B%off(bb) + 1), B%wk4(B%off(bb) + 1))
        else
          call hyb_sol_sep(B%rg(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), B%wk(B%off(bb) + 1))
        endif
        do k = 0, B%rg(bb)%ns - 1
          call scat(bb, B%rg(bb)%rp(B%rg(bb)%sep(k)) + 1, B%rg(bb)%rp(B%rg(bb)%sep(k) + 1), rsp, yp)
        enddo
      enddo
      !$omp end do
      !$omp do schedule(dynamic, 1)
      do p = 1, size(B%lb)
        bb = B%lb(p); k = B%lk(p)
        if (.not. reduced(bb)) cycle
        if (rsp) then
          call hyb_sol_bwd_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%wk4(B%off(bb) + 1), k)
        else
          call hyb_sol_bwd(B%rg(bb), B%lu(B%loff(bb) + 1), B%wk(B%off(bb) + 1), k)
        endif
        call scat(bb, B%rg(bb)%rp(B%rg(bb)%sep(k) + 1) + 1, B%rg(bb)%rp(hyb_end(B%rg(bb), k) + 1), rsp, yp)
      enddo
      !$omp end do
      !$omp end parallel
      call VecRestoreArrayRead(xx, xp, ie)
      call VecRestoreArray(yy, yp, ie)
      call ghosts_done(gp)
      call PetscLogEventEnd(gev_lines(cur_inst), ie)
    end subroutine lines_solve_par

    !> lines_solve_par: block bb's positions q1..q2 of x (owned rows xp,
    !! ghosts gp) -> B%wk, or B%wk4 (sp)
    subroutine gath(bb, q1, q2, sp, xp, gp)
      integer, intent(in) :: bb, q1, q2
      logical, intent(in) :: sp
      PetscScalar, intent(in) :: xp(:)
      PetscScalar, pointer, intent(in) :: gp(:)            !< null without overlap
      integer :: q, r
      real*8 :: v
      do q = q1, q2
        r = B%rows(B%off(bb) + q)
        if (r < B%nrow) then
          v = xp(r + 1)
        else
          v = gp(r - B%nrow + 1)
        endif
        if (sp) then
          B%wk4(B%off(bb) + q) = real(v * B%eqs(B%off(bb) + q), 4)
        else
          B%wk(B%off(bb) + q) = v
        endif
      enddo
    end subroutine gath

    !> lines_solve_par: B%wk (B%wk4) at block bb's positions q1..q2 -> yp on
    !! the owned rows
    subroutine scat(bb, q1, q2, sp, yp)
      integer, intent(in) :: bb, q1, q2
      logical, intent(in) :: sp
      PetscScalar, intent(inout) :: yp(:)
      integer :: q, r
      do q = q1, q2
        r = B%rows(B%off(bb) + q)
        if (r >= B%nrow) cycle
        if (sp) then
          yp(r + 1) = B%wk4(B%off(bb) + q) * B%eqs(B%off(bb) + q)
        else
          yp(r + 1) = B%wk(B%off(bb) + q)
        endif
      enddo
    end subroutine scat

    !> is block bb solved by the threaded hybrid (a ring, not singular)?
    logical function reduced(bb)
      integer, intent(in) :: bb
      reduced = .false.
      if (B%btd(bb)) reduced = .not. B%jac(bb)
    end function reduced

    !> y(rows outside the axis block) = dense/banded block solves of x. The
    !! blocks own disjoint rows, so they run on the rank's OpenMP threads.
    !! With the line overlap the ghost rows' input comes from the neighbours
    !! (one scatter) and only the rank's own rows are written back.
    subroutine lines_solve(xx, yy)
      Vec :: xx, yy
      PetscScalar, pointer :: xp(:), yp(:), gp(:)
      PetscErrorCode :: ie
      integer :: bb, q, n, t, r, nr, ck
      call PetscLogEventBegin(gev_lines(cur_inst), ie)
      call ghosts_in(xx, gp)
      call VecGetArrayRead(xx, xp, ie)
      call VecGetArray(yy, yp, ie)
      nr = B%nrow
      ! chunks of 4 lines, but single blocks where a rank has few (rings: a
      ! handful per rank, each a whole flux surface)
      ck = 4
      !$ ck = max(1, min(4, B%nb / (8 * omp_get_max_threads())))
      ! small (coarse) levels stay serial: fork/join would cost more than the work
      !$omp parallel do schedule(dynamic, ck) private(bb, q, n, t, r) if (B%nrow >= LINES_OMP_MIN)
      do bb = 1, B%nb
        if (B%axblk(bb)) cycle
        t = 0
        !$ t = omp_get_thread_num()
        n = B%sz(bb)
        do q = 1, n
          r = B%rows(B%off(bb) + q)
          if (r < nr) then
            blk_t_work(q, t) = xp(r + 1)
          else
            blk_t_work(q, t) = gp(r - nr + 1)
          endif
        enddo
        call one_solve(bb, blk_t_work(1, t))
        do q = 1, n
          r = B%rows(B%off(bb) + q)
          if (r < nr) yp(r + 1) = blk_t_work(q, t)
        enddo
      enddo
      !$omp end parallel do
      call VecRestoreArrayRead(xx, xp, ie)
      call VecRestoreArray(yy, yp, ie)
      call ghosts_done(gp)
      call PetscLogEventEnd(gev_lines(cur_inst), ie)
    end subroutine lines_solve

    !> Smoother 7: one forward block Gauss-Seidel sweep over two colours of
    !! radial lines (zebra line relaxation, Trottenberg-Oosterlee-Schueller
    !! S5.1). Colour 0 (the axis block, then even J) is solved from x; colour 1
    !! (odd J) from x - A(odd, colour 0) y. A radial line couples only to the
    !! lines at J +- 1, so odd lines never couple to each other and the sweep
    !! is exact block Gauss-Seidel on the rank's rows. Same line factors and one
    !! line pass as smoother 5, plus half a local matvec. With the line overlap
    !! the colour-0 solutions on the ghost rows are kept (yg), and the
    !! extended colour-1 lines couple to them: no second communication.
    subroutine zebra_solve(xx, yy)
      Vec :: xx, yy
      PetscScalar, pointer :: xp(:), yp(:), gp(:)
      PetscErrorCode :: ie
      integer :: bb, q, n, t, c, k, r, nr, zc_
      nr = B%nrow
      ! the axis block belongs to colour 0 and depends on x only: solved
      ! first, while the ranks are in step (see blk_apply)
      if (B%axsparse) call ax_solve(xx, yy)
      call ghosts_in(xx, gp)
      do c = 0, 1
        call PetscLogEventBegin(gev_lines(cur_inst), ie)
        call VecGetArrayRead(xx, xp, ie)
        call VecGetArray(yy, yp, ie)
        !$omp parallel do schedule(dynamic, 4) private(bb, q, n, t, k, r, zc_) if (B%nrow >= LINES_OMP_MIN)
        do bb = 1, B%nb
          if (B%axblk(bb) .or. B%bcol(bb) /= c) cycle
          t = 0
          !$ t = omp_get_thread_num()
          n = B%sz(bb)
          do q = 1, n
            r = B%rows(B%off(bb) + q)
            if (r < nr) then
              blk_t_work(q, t) = xp(r + 1)
            else
              blk_t_work(q, t) = gp(r - nr + 1)
            endif
            ! colour 1: minus the couplings to the colour-0 solution
            do k = B%zp(r + 1), B%zp(r + 2) - 1
              zc_ = B%zc(k)
              if (zc_ < nr) then
                blk_t_work(q, t) = blk_t_work(q, t) - B%zv(k) * yp(zc_ + 1)
              else
                blk_t_work(q, t) = blk_t_work(q, t) - B%zv(k) * B%yg(zc_ - nr + 1)
              endif
            enddo
          enddo
          call one_solve(bb, blk_t_work(1, t))
          do q = 1, n
            r = B%rows(B%off(bb) + q)
            if (r < nr) then
              yp(r + 1) = blk_t_work(q, t)
            else if (c == 0) then
              B%yg(r - nr + 1) = blk_t_work(q, t)
            endif
          enddo
        enddo
        !$omp end parallel do
        call VecRestoreArrayRead(xx, xp, ie)
        call VecRestoreArray(yy, yp, ie)
        call PetscLogEventEnd(gev_lines(cur_inst), ie)
      enddo
      call ghosts_done(gp)
    end subroutine zebra_solve

    !> gp = the input's values on the ghost rows (empty without overlap)
    subroutine ghosts_in(xx, gp)
      Vec :: xx
      PetscScalar, pointer :: gp(:)
      PetscErrorCode :: ie
      gp => null()
      if (.not. B%ovl) return
      call VecScatterBegin(B%sct, xx, B%xg, INSERT_VALUES, SCATTER_FORWARD, ie)
      call VecScatterEnd(B%sct, xx, B%xg, INSERT_VALUES, SCATTER_FORWARD, ie)
      call VecGetArrayRead(B%xg, gp, ie)
    end subroutine ghosts_in

    subroutine ghosts_done(gp)
      PetscScalar, pointer :: gp(:)
      PetscErrorCode :: ie
      if (B%ovl) call VecRestoreArrayRead(B%xg, gp, ie)
    end subroutine ghosts_done

    !> y(axis rows) = axis-block solve of x(axis rows), on the axis ranks only
    subroutine ax_solve(xx, yy)
      Vec :: xx, yy
      PetscErrorCode :: ie
      PetscScalar, pointer :: xp(:), yp(:)
      integer :: kk
      if (B%axdon) then                         ! collective: every rank enters the scatters
        call PetscLogEventBegin(gev_axsolve(cur_inst), ie)
        do kk = 1, size(B%axd)
          call axd_solve(B%axd(kk), xx, yy)
        enddo
        call PetscLogEventEnd(gev_axsolve(cur_inst), ie)
        return
      endif
      if (.not. B%axg(1)%member) return
      call PetscLogEventBegin(gev_axsolve(cur_inst), ie)
      call VecGetArrayRead(xx, xp, ie)
      call VecGetArray(yy, yp, ie)
      call rds_solve_arr(B%axg(1), xp, yp)
      call VecRestoreArray(yy, yp, ie)
      call VecRestoreArrayRead(xx, xp, ie)
      call PetscLogEventEnd(gev_axsolve(cur_inst), ie)
    end subroutine ax_solve

  end subroutine blk_apply

#endif
end module mod_petsc_pc_gmg
