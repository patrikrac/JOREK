!> Workstream C: geometric multigrid on the exact nested C1 coarse space of
!! JOREK's logically structured polar grid, applied to the SFM2 packed pair_w.
!!
!! Port of docs/physics_pc/tools/gmg_probe.py, validated offline in stage C1
!! (docs/physics_pc/workstream_C_gmg_probe.md S2, S3.2): on pair_w a V-cycle
!! with GMRES(4)+Jacobi smoothing and an exact axis patch needs 15/16/44 FGMRES
!! its at tstep 0.1/1/10 on 41x16 and 21/21/56 on the nested 81x32, against
!! 33/37/77 -> 62/83/n.c. for the same smoother without a coarse grid.
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
!! outside range(P) and near-null (X(:,3) = 0 on the axis), so the fine level
!! gets an exact solve on all axis-node DOFs after each smoothing. Without it
!! the error floor is ~50x higher (stage C1 gate 3).
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
  use iso_c_binding, only: c_ptr, c_double, c_null_ptr, c_associated
  use mod_petsc_raw_csr, only: aij_parts, get_ij, put_ij, aij_vals_read, aij_vals_done, blockmv_attach, &
                               aij_vals_write, aij_vals_written
  use mod_petsc_blk_dev, only: blk_dev_available, blk_dev_pattern, blk_dev_values, blk_dev_apply, blk_dev_free
  use mod_petsc_pc_gmg_axis, only: axd_t, axd_nsec, axd_setup, axd_refill, axd_factor, axd_solve
  implicit none
  private

  public :: gmg_build_prolongations, gmg_setup_operator, gmg_vcycle_apply, gmg_is_ready
  public :: gmg_select, gmg_vcycle, gmg_pc_apply_3, gmg_pc_apply_4
  public :: gmg_pc_apply_1, gmg_pc_apply_2
  public :: gmg_push, gmg_pop
  public :: gmg_opts_t, gmg_opts_from_namelist

  !> Everything that configures a hierarchy, passed EXPLICITLY. A caller that
  !! passes it (the production SF path) is independent of every
  !! physics_pc_gmg_* namelist entry; a caller that does not gets
  !! gmg_opts_from_namelist(), i.e. exactly the research path's behaviour.
  type :: gmg_opts_t
    integer :: smoother     = 0      !< level smoother code (see physics_pc_gmg_smoother)
    integer :: nsmooth      = 0      !< smoothing steps; <= 0 = the smoother's default
    integer :: pre0         = -1     !< level-0 pre-smoothing steps; < 0 = nsmooth (0 = none)
    integer :: post0        = -1     !< level-0 post-smoothing steps; < 0 = nsmooth (0 = none)
    integer :: nsmooth_c    = -1     !< pre and post steps on the coarse levels; <= 0 = nsmooth
    integer :: axis_rings   = 0      !< axis block extent: rings 0..k; -1 = by physical radius
    integer :: axis_mult    = 0      !< axis/lines coupling (0 = additive)
    integer :: axis_split   = 0      !< one axis solve per |n| group, on distinct ranks
    integer :: smooth_op    = 0      !< 1 = fine smoother on the assembled surrogate
    integer :: ring_diag    = 0      !< diagnostic samples per rebuild (0 = off)
    integer :: bnd_drop     = 0      !< drop Dirichlet DOFs from the coarse spaces
    integer :: harm_split   = 0      !< operator is block-diagonal in |n|
    integer :: ring_overlap = 0      !< smoother 4 (and 9's rings): a ring cut by a rank
                                     !< boundary is extended by up to this many nodes along
                                     !< J into the ranks owning the rest (RAS, as the lines)
    integer :: harm_pair    = 0      !< 1 = a harmonic's cos and sin slots share every
                                     !< smoother block (first-order d_phi couplings, which
                                     !< map cos <-> sin, then lie inside the blocks)
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
    real*8  :: omega        = 0.7d0  !< Richardson damping (smoothers 1, 2)
    real*8  :: rich_omega   = 0.d0   !< Richardson scale on levels >= rich_from;
                                     !< <= 0 = 1 (smoothers 1, 2: omega)
    real*8  :: axis_droptol = 0.d0   !< relative drop tolerance of the axis block
    real*8  :: ring_aspect  = 1.d0   !< smoother 6 / axis_rings -1 switch radius
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
  integer, parameter :: SM_STEPS  = 4      !< GMRES iterations per smoothing (the paper's, and stage C1's)
  integer, parameter :: MAX_ENT   = 16     !< max nonzeros per scalar row of P (2x2 sources x 4 DOFs)
  !> semi_r < 0: levels coarsen radially only while the median r dtheta/dr of
  !! their cells exceeds this; each such level halves it, a full one keeps it
  real*8, parameter :: SEMI_ASPECT = 2.d0

  type :: lvl_t
    integer :: ni = 0, nj = 0, n = 0
    integer, allocatable :: d(:,:,:)       !< (0:ni-1, 0:nj-1, 0:3) -> canonical DOF, -1 = dropped
    integer, allocatable :: rnode(:)       !< packed LOCAL row -> I*nj + J (block maps)
    integer, allocatable :: rharm(:)       !< packed LOCAL row -> toroidal slot m
  end type lvl_t

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
  KSP, save         :: gSm(0:MAX_LEV-1), gAxis
  Vec, save         :: gx(0:MAX_LEV-1), gb(0:MAX_LEV-1), gr(0:MAX_LEV-1), gzax
  IS, save          :: gisAxis
  Mat, save         :: gF                   !< fine-level matvec operator: gA(0), or a
                                            !< matrix-free equivalent (Workstream D)
  logical, save     :: p_ready = .false., op_ready = .false., vec_ready = .false.
  ! Device path (-sf_gpu 1, dev_operators): the V-cycle multiplies with device
  ! copies of the level operators and prolongations and keeps its vectors on
  ! the device. The host matrices above stay what the setup works on (Galerkin
  ! chain, smoother blocks, value maps), so the setup is the CPU path's.
  logical, save     :: dev = .false.
  logical, save     :: dev_blocks = .true.    !< -sf_gpu_blocks: block smoothers on the device too
  logical, save     :: blk_host = .false.     !< blk_apply on the host factors (the gate's reference)
  Mat, save         :: gAd(0:MAX_LEV-1), gPd(1:MAX_LEV-1), gFd

  !> A direct solve of A(rows, rows) done REDUNDANTLY on the ranks that own
  !! some of those rows (sub-communicator comm; mostly one rank, since JOREK
  !! partitions ring by ring). Each member holds the whole submatrix as a
  !! sequential AIJ (sub(1)) with its own LU, so a solve costs no collective on
  !! the hierarchy's communicator -- only an Allgatherv among the members when
  !! the rows span ranks -- and non-members skip it. loc = the rank's rows
  !! (local, 0-based, ascending; set before the first rds_setup), cnt/dsp =
  !! the members' row counts/offsets. Used for the stage-D13 axis blocks and
  !! the coarsest level, which both live on a few ranks at scale.
  ! root >= 0 (stage Q, physics_pc_gmg_axis_split): the member of rank root
  ! alone factors and solves; the others gather their rows to it and get the
  ! solution back. root = -1: every member solves the same system redundantly.
  type :: rds_t
    logical :: ready = .false., member = .false., solver = .false.
    integer :: comm = MPI_COMM_NULL, np = 0, me = 0, root = -1
    integer, allocatable :: loc(:), cnt(:), dsp(:)
    real*8, allocatable  :: send(:)
    IS  :: isq
    Mat, pointer :: sub(:) => null()
    Mat :: flt                        !< sub(1) with small entries dropped (axis_droptol > 0)
    logical :: filtered = .false.
    KSP :: ksp
    Vec :: b, x
  end type rds_t

  ! Workstream D: node-block Jacobi smoother (Chacon 2025 S4.1). One dense block
  ! per (node, toroidal harmonic) holding both fields and all four C1 DOFs, and
  ! the axis ring as one block per harmonic. bid(row) -> block (1-based);
  ! rows(off(b)+1 : off(b)+sz(b)) are the block's 0-based rows; the LU factors
  ! (column-major) start at lu(loff(b)+1), the pivots at piv(off(b)+1).
  type :: blk_t
    integer :: nb = 0, nrow = 0, nsing = 0
    integer, allocatable :: bid(:), off(:), sz(:), rows(:), pos(:), piv(:), kl(:), ku(:)
    logical, allocatable :: band(:)
    integer(8), allocatable :: loff(:)
    real*8, allocatable  :: lu(:)
    ! Stage D13: with physics_pc_gmg_axis_rings /= 0 the axis block (rings
    ! 0..axis_lim(g), every slot) is not factored dense but by MUMPS on its
    ! submatrix: it holds a fixed fraction of the level's rows (~I_s/n_flux),
    ! so a dense LU would cost O(N^3). axblk(b) marks its blocks.
    ! The solve is an rds_t: only the ranks owning axis rows take part.
    logical :: axsparse = .false.
    logical, allocatable :: axblk(:)
    IS  :: axis_is
    ! one solve per |n| group with physics_pc_gmg_axis_split (the block is
    ! block-diagonal in |n| once harm_split drops the cross-|n| entries),
    ! group k solved on rank mod(k, np); otherwise one group, all slots
    type(rds_t), allocatable :: axg(:)
    ! physics_pc_gmg_axis_mult > 0: Gauss-Seidel coupling between the axis
    ! block and the lines through the interface blocks A(rest,ax), A(ax,rest)
    logical :: gsready = .false., gsvec = .false.
    IS  :: rest_is
    Mat :: Bra, Bar
    Vec :: xw, xw2, tr, ta
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
    ! Device path: the factors' copy on the device (mod_petsc_blk_dev), which
    ! then does the solves of every block but the axis blocks; null = host.
    ! dstat = largest block, widest band, multiply-adds of one pass.
    type(c_ptr) :: dh = c_null_ptr
    real*8 :: dstat(3) = 0.d0
  end type blk_t
  type(blk_t), save, target :: gBk(0:MAX_LEV-1)
  !> Smoother 9 (alternating): the radial-line blocks; gBk holds the rings
  type(blk_t), save, target :: gBk2(0:MAX_LEV-1)
  logical, save :: blk_set2 = .false.        !< blk_apply works on gBk2
  type(rds_t), save :: gcrs                  !< the coarsest level's direct solve
  type(lvl_t), allocatable, save :: glv(:)   !< coarse DOF numberings, kept for the block maps
  integer, allocatable, save :: fine_node(:) !< level-0 row -> node-1 (i*n_tht + j), from node%index
  integer, allocatable, save :: fine_kf(:)   !< level-0 row -> canonical DOF k + 4*field (ring diagnostics)
  integer, allocatable, save :: fine_harm(:) !< level-0 row -> toroidal slot m
  integer, save :: nth0 = 0                  !< n_tht (level-0 nj)
  integer, save :: nf_s = 0, cur_lev = 0
  integer, save :: stk_inst(8), stk_lev(8), nstk = 0
  integer, save :: sm_type = 0, sm_nstep = 4
  logical, save :: sm_blocks = .false.
  real*8, allocatable, save :: blk_t_work(:,:)   !< (max block size, 0:threads-1)
  integer, parameter :: LINES_OMP_MIN = 4000     !< rows below which lines_solve stays serial
  ! Axis treatment (Bourne et al., JCP 488 (2023) 112249 S2-S3). Fine rings
  ! I < ring_is have median r*dtheta/dr below physics_pc_gmg_ring_aspect: the
  ! circle couplings dominate there, so smoother 6 uses ring blocks inside and
  ! radial lines outside (GMGPolar). Rings 0..axis_k form the axis block.
  integer, save :: ring_is = 1, axis_k = 0, axis_mult = 0
  integer, save :: lovl = 0                 !< line overlap in nodes (gmg_opts_t%line_overlap)
  integer, save :: hpair = 0                !< gmg_opts_t%harm_pair
  integer, save :: rovl = 0                 !< gmg_opts_t%ring_overlap (ring blocks)
  integer, save :: semi_r = 0               !< gmg_opts_t%semi_r, of the hierarchy being built
  integer, save :: axsec = 0                !< axis J-sectors (gmg_opts_t%axis_sectors)
  logical, save :: axis_split = .false.     !< stage Q: per-|n| axis solves on distinct ranks
  real*8, save  :: axis_droptol = 0.d0      !< stage Q: relative drop tolerance of the axis block
  integer, save :: diag_left = 0            !< ring-diag samples left in this rebuild
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

  ! Workstream D (pair_psi): several independent hierarchies in one module.
  ! The routines below work on the module-level state; gmg_select(k) parks
  ! the active instance's state in inst(cur_inst) and loads instance k.
  ! PETSc handles are pointers and the arrays move with move_alloc, so a
  ! switch is O(1). Instance 1 = pair_w (2 fields), 2 = the pair_psi Schur
  ! Shat, 3 = B_55 (rho), 4 = B_66 (T), all one field. V-cycles of different
  ! instances never nest.
  integer, parameter :: MAX_INST = 4
  type :: gmg_inst_t
    integer :: nlev = 0, nth0 = 0, nf_s = 0, sm_type = 0, sm_nstep = 4
    integer :: ring_is = 1, axis_k = 0, axis_mult = 0, diag_left = 0, lovl = 0, axsec = 0, hpair = 0, rovl = 0
    logical :: axis_split = .false.
    real*8  :: axis_droptol = 0.d0
    integer(8) :: a0_sig(2) = 0, a0_id = 0, a0_nzst = -1
    logical :: p_ready = .false., op_ready = .false., vec_ready = .false., sm_blocks = .false.
    Mat :: gP(1:MAX_LEV-1), gA(0:MAX_LEV-1), gF
    logical :: dev = .false.
    Mat :: gAd(0:MAX_LEV-1), gPd(1:MAX_LEV-1), gFd
    KSP :: gSm(0:MAX_LEV-1), gAxis
    type(rds_t) :: gcrs
    Vec :: gx(0:MAX_LEV-1), gb(0:MAX_LEV-1), gr(0:MAX_LEV-1), gzax
    IS  :: gisAxis
    type(blk_t) :: gBk(0:MAX_LEV-1), gBk2(0:MAX_LEV-1)
    type(lvl_t), allocatable :: glv(:)
    integer, allocatable :: fine_node(:), fine_harm(:), fine_kf(:)
  end type gmg_inst_t
  type(gmg_inst_t), save :: inst(MAX_INST)
  integer, save :: cur_inst = 1
  !> The V-cycle's shape per instance (gmg_opts_t pre0 / post0 / nsmooth_c,
  !! resolved against nsmooth at setup): level-0 pre and post steps, and the
  !! steps of every coarser level. Kept by instance, like the hierarchy.
  integer, save :: st_pre0(MAX_INST) = SM_STEPS, st_post0(MAX_INST) = SM_STEPS
  integer, save :: st_crs(MAX_INST) = SM_STEPS

  ! -log_view events (Workstream D cost audit). Registered here, not in the
  ! physics-PC ctx, so this module keeps no dependency on it.
  PetscLogEvent, save :: gev_ptap = -1, gev_smsetup = -1, gev_coarselu = -1, gev_axislu = -1
  PetscLogEvent, save :: gev_axis = -1, gev_prolong = -1, gev_dev = -1
  PetscLogEvent, save :: gev_blkmesh = -1, gev_blkpat = -1, gev_blknum = -1, gev_axnum = -1, gev_axtry = -1
  ! Workstream G, finding D3: these are per INSTANCE, like gev_vcycle. While
  ! they were shared, no -log_view number could attribute line or axis time to
  ! pair_w rather than to Shat / rho / T, which is what left C1's rank soft.
  PetscLogEvent, save :: gev_lines(4) = -1, gev_axsolve(4) = -1   !< the block smoother's two halves
  ! per hierarchy instance (1 = pair_w keeps the original names, 2 = GMG2_*)
  PetscLogEvent, save :: gev_vcycle(4) = -1, gev_smooth0(4) = -1, gev_smooth(4) = -1, gev_coarse(4) = -1
  logical, save       :: gev_ready = .false.

  ! 1D cubic-Hermite subdivision, level-local canonical units (u, f_x/3):
  ! W(p+1, q+1) maps coarse quantity q to fine quantity p.
  real*8, parameter :: W_CO(2,2) = reshape([1.0d0, 0.0d0, 0.0d0, 0.5d0],      [2,2])
  real*8, parameter :: W_L(2,2)  = reshape([0.5d0, -0.25d0, 0.375d0, -0.125d0], [2,2])
  real*8, parameter :: W_R(2,2)  = reshape([0.5d0, 0.25d0, -0.375d0, -0.125d0], [2,2])
  real*8, parameter :: W_ID(2,2) = reshape([1.0d0, 0.0d0, 0.0d0, 1.0d0],      [2,2])  !< not coarsened
  integer, parameter :: PQ_I(0:3) = [0, 1, 0, 1]   !< canonical k -> derivative order in i (u, a, b, c)
  integer, parameter :: PQ_J(0:3) = [0, 0, 1, 1]   !< ... and in j

contains

  !> The research path's configuration: every field from its namelist entry.
  function gmg_opts_from_namelist() result(o)
    use phys_module, only: physics_pc_gmg_smoother, physics_pc_gmg_nsmooth, physics_pc_gmg_omega, &
                           physics_pc_gmg_axis_rings, physics_pc_gmg_ring_diag, physics_pc_gmg_axis_mult, &
                           physics_pc_gmg_bnd_drop, physics_pc_gmg_smooth_op, physics_pc_gmg_axis_split, &
                           physics_pc_harm_split, physics_pc_gmg_axis_droptol, physics_pc_gmg_ring_aspect
    type(gmg_opts_t) :: o
    o%smoother = physics_pc_gmg_smoother;    o%nsmooth = physics_pc_gmg_nsmooth
    o%axis_rings = physics_pc_gmg_axis_rings; o%axis_mult = physics_pc_gmg_axis_mult
    o%axis_split = physics_pc_gmg_axis_split; o%smooth_op = physics_pc_gmg_smooth_op
    o%ring_diag = physics_pc_gmg_ring_diag;   o%bnd_drop = physics_pc_gmg_bnd_drop
    o%harm_split = physics_pc_harm_split;     o%omega = physics_pc_gmg_omega
    o%axis_droptol = physics_pc_gmg_axis_droptol; o%ring_aspect = physics_pc_gmg_ring_aspect
  end function gmg_opts_from_namelist

  !> Save / restore the active hierarchy and level around a NESTED use of
  !! another hierarchy -- the pair_w shell applying pair_psi's V-cycle
  !! (instance 2) while instance 1 is active.
  subroutine gmg_push()
    nstk = nstk + 1
    if (nstk > size(stk_inst)) stop "gmg_push: stack overflow"
    stk_inst(nstk) = cur_inst; stk_lev(nstk) = cur_lev
  end subroutine gmg_push

  subroutine gmg_pop()
    if (nstk < 1) stop "gmg_pop: empty stack"
    call gmg_select(stk_inst(nstk))
    cur_lev = stk_lev(nstk)
    nstk = nstk - 1
  end subroutine gmg_pop

  !> Make hierarchy k the active one (see gmg_inst_t).
  subroutine gmg_select(k)
    integer, intent(in) :: k
    integer :: g
    if (k == cur_inst) return
    ! park the active state
    associate (S => inst(cur_inst))
      S%nlev = nlev; S%nth0 = nth0; S%nf_s = nf_s; S%sm_type = sm_type; S%sm_nstep = sm_nstep
      S%ring_is = ring_is; S%axis_k = axis_k; S%axis_mult = axis_mult; S%diag_left = diag_left
      S%lovl = lovl; S%axsec = axsec; S%hpair = hpair; S%rovl = rovl
      S%axis_split = axis_split; S%axis_droptol = axis_droptol
      S%a0_sig = a0_sig; S%a0_id = a0_id; S%a0_nzst = a0_nzst
      S%p_ready = p_ready; S%op_ready = op_ready; S%vec_ready = vec_ready; S%sm_blocks = sm_blocks
      S%gP = gP; S%gA = gA; S%gF = gF; S%gSm = gSm; S%gAxis = gAxis
      S%dev = dev; S%gAd = gAd; S%gPd = gPd; S%gFd = gFd
      S%gx = gx; S%gb = gb; S%gr = gr; S%gzax = gzax; S%gisAxis = gisAxis
      do g = 0, MAX_LEV - 1
        call move_blk(gBk(g), S%gBk(g))
        call move_blk(gBk2(g), S%gBk2(g))
      enddo
      call move_rds(gcrs, S%gcrs)
      if (allocated(glv)) call move_alloc(glv, S%glv)
      if (allocated(fine_node)) call move_alloc(fine_node, S%fine_node)
      if (allocated(fine_kf)) call move_alloc(fine_kf, S%fine_kf)
      if (allocated(fine_harm)) call move_alloc(fine_harm, S%fine_harm)
    end associate
    ! load instance k
    associate (S => inst(k))
      nlev = S%nlev; nth0 = S%nth0; nf_s = S%nf_s; sm_type = S%sm_type; sm_nstep = S%sm_nstep
      ring_is = S%ring_is; axis_k = S%axis_k; axis_mult = S%axis_mult; diag_left = S%diag_left
      lovl = S%lovl; axsec = S%axsec; hpair = S%hpair; rovl = S%rovl
      axis_split = S%axis_split; axis_droptol = S%axis_droptol
      a0_sig = S%a0_sig; a0_id = S%a0_id; a0_nzst = S%a0_nzst
      p_ready = S%p_ready; op_ready = S%op_ready; vec_ready = S%vec_ready; sm_blocks = S%sm_blocks
      gP = S%gP; gA = S%gA; gF = S%gF; gSm = S%gSm; gAxis = S%gAxis
      dev = S%dev; gAd = S%gAd; gPd = S%gPd; gFd = S%gFd
      gx = S%gx; gb = S%gb; gr = S%gr; gzax = S%gzax; gisAxis = S%gisAxis
      do g = 0, MAX_LEV - 1
        call move_blk(S%gBk(g), gBk(g))
        call move_blk(S%gBk2(g), gBk2(g))
      enddo
      call move_rds(S%gcrs, gcrs)
      if (allocated(S%glv)) call move_alloc(S%glv, glv)
      if (allocated(S%fine_node)) call move_alloc(S%fine_node, fine_node)
      if (allocated(S%fine_kf)) call move_alloc(S%fine_kf, fine_kf)
      if (allocated(S%fine_harm)) call move_alloc(S%fine_harm, fine_harm)
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
    b%gsready = a%gsready; b%gsvec = a%gsvec
    b%rest_is = a%rest_is; b%Bra = a%Bra; b%Bar = a%Bar
    b%xw = a%xw; b%xw2 = a%xw2; b%tr = a%tr; b%ta = a%ta
    a%gsready = .false.; a%gsvec = .false.
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
    b%dh = a%dh; a%dh = c_null_ptr; b%dstat = a%dstat
  end subroutine move_blk

  subroutine move_rds(a, b)
    type(rds_t), intent(inout) :: a, b
    b%ready = a%ready; b%member = a%member; b%solver = a%solver
    b%comm = a%comm; b%np = a%np; b%me = a%me; b%root = a%root
    b%isq = a%isq; b%ksp = a%ksp; b%b = a%b; b%x = a%x
    b%flt = a%flt; b%filtered = a%filtered; a%filtered = .false.
    b%sub => a%sub; a%sub => null()
    a%ready = .false.; a%member = .false.; a%solver = .false.; a%comm = MPI_COMM_NULL; a%np = 0; a%me = 0
    a%root = -1
    if (allocated(a%loc))  call move_alloc(a%loc,  b%loc)
    if (allocated(a%cnt))  call move_alloc(a%cnt,  b%cnt)
    if (allocated(a%dsp))  call move_alloc(a%dsp,  b%dsp)
    if (allocated(a%send)) call move_alloc(a%send, b%send)
  end subroutine move_rds

  !> (Re)build R for A(rows, rows), rows = R%loc on each rank. what = options
  !! prefix part (gmg<k>_<what>_); tag = label of the first build's
  !! factor-size print. solver_rank (rank in gcomm): that rank alone solves,
  !! whether or not it owns any of the rows; absent: all owners solve.
  subroutine rds_setup(R, A, what, tag, solver_rank)
    type(rds_t), intent(inout) :: R
    Mat, intent(in)     :: A
    character(len=*), intent(in) :: what, tag
    integer, intent(in), optional :: solver_rank
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
      if (R%filtered) call MatDestroy(R%flt, ierr)
      R%filtered = .false.
      if (R%solver) then
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
      if (present(solver_rank)) then
        if (gme == solver_rank) color = 1
      endif
      call MPI_Comm_split(gcomm, color, gme, R%comm, mpierr)
      R%member = (color == 1)
      ntot = 0
      if (R%member) then
        call MPI_Comm_size(R%comm, R%np, mpierr)
        call MPI_Comm_rank(R%comm, R%me, mpierr)
        R%root = -1
        if (present(solver_rank)) then
          k = -1
          if (gme == solver_rank) k = R%me
          call MPI_Allreduce(k, R%root, 1, MPI_INTEGER, MPI_MAX, R%comm, mpierr)
        endif
        R%solver = (R%root < 0 .or. R%me == R%root)
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
        if (.not. R%solver) then            ! the root alone extracts and factors
          deallocate(gl); allocate(gl(0)); ntot = 0
        endif
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
    if (R%solver .and. axis_droptol > 0.d0 .and. what(1:2) == "ax") then
      if (R%filtered) call MatDestroy(R%flt, ierr)
      call rds_filter(R, tag, fresh)
      R%filtered = .true.
    endif
    if (R%solver) then
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
          call MatGetSize(rds_op(R), nsub, PETSC_NULL_INTEGER, ierr)
          if (what == "coarse" .and. nsub >= CRS_MUMPS_MIN) then
            call rds_make_ksp(MATSOLVERMUMPS)
          else
            call rds_make_ksp(MATSOLVERPETSC)
          endif
        end block
        call MatCreateVecs(rds_op(R), R%x, R%b, ierr)
      else
        call KSPSetOperators(R%ksp, rds_op(R), rds_op(R), ierr)
      endif
    endif
    if (R%solver) then
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
    if (.not. R%ready .and. R%solver) then      ! factor size
      if (R%me == max(R%root, 0)) then
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
      call KSPSetOperators(R%ksp, rds_op(R), rds_op(R), ierr)
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

  !> The matrix the KSP factors: the filtered copy where there is one.
  function rds_op(R) result(M_)
    type(rds_t), intent(in) :: R
    Mat :: M_
    M_ = R%sub(1)
    if (R%filtered) M_ = R%flt
  end function rds_op

  !> Stage Q (physics_pc_gmg_axis_droptol): R%flt = R%sub(1) without the
  !! entries below tol*sqrt(|a_ii a_jj|). The scaling is per entry because
  !! JOREK's boundary rows carry a diagonal about 7 decades above the bulk,
  !! so one absolute threshold would empty the small-scale rows. The block
  !! is a smoother component, so an approximate factor is allowed; what it
  !! buys is a smaller factor and a cheaper triangular solve per call.
  subroutine rds_filter(R, tag, verbose)
    type(rds_t), intent(inout) :: R
    character(len=*), intent(in) :: tag
    logical, intent(in) :: verbose
    Vec :: dv
    MatInfo :: i0, i1
    PetscScalar, pointer :: dp(:)
    PetscErrorCode :: ierr
    PetscInt :: n, q

    call MatDuplicate(R%sub(1), MAT_COPY_VALUES, R%flt, ierr)
    call MatCreateVecs(R%flt, dv, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(R%flt, dv, ierr)
    call VecGetLocalSize(dv, n, ierr)
    call VecGetArray(dv, dp, ierr)
    do q = 1, n
      if (abs(dp(q)) > 0.d0) then
        dp(q) = 1.d0 / sqrt(abs(dp(q)))
      else
        dp(q) = 1.d0
      endif
    enddo
    call VecRestoreArray(dv, dp, ierr)
    call MatGetInfo(R%flt, MAT_LOCAL, i0, ierr)
    ! symmetric diagonal scaling, absolute filter, scaling undone: drops the
    ! entries with |a_ij| < tol*sqrt(|a_ii a_jj|)
    call MatDiagonalScale(R%flt, dv, dv, ierr)
    call MatFilter(R%flt, axis_droptol, PETSC_TRUE, PETSC_TRUE, ierr)
    call VecGetArray(dv, dp, ierr)
    do q = 1, n
      dp(q) = 1.d0 / dp(q)
    enddo
    call VecRestoreArray(dv, dp, ierr)
    call MatDiagonalScale(R%flt, dv, dv, ierr)
    ! compression can leave a row without a stored diagonal, which the LU
    ! needs: put the (zero) diagonals back
    call MatSetOption(R%flt, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call VecSet(dv, 0.d0, ierr)
    call MatDiagonalSet(R%flt, dv, ADD_VALUES, ierr)
    call MatGetInfo(R%flt, MAT_LOCAL, i1, ierr)
    call VecDestroy(dv, ierr)
    if (verbose) write(*,'(A,I0,A,A,A,F5.1,A,ES8.1)') "[Mem] GMG", cur_inst, " ", trim(tag), &
      ": axis drop keeps ", 100.d0 * i1%nz_used / max(i0%nz_used, 1.d0), "% of the entries, tol ", axis_droptol
  end subroutine rds_filter

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
    if (R%root >= 0) then
      call rds_gather(R, xp)
      call rds_local(R)
      call rds_return(R, yp)
      return
    endif
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

  !> Several rooted solves at once (the per-|n| axis groups): every group's
  !! rows go to its root first, then each root solves its groups, then the
  !! solutions come back, so groups on different ranks are solved at the
  !! same time. All ranks walk the groups in the same order (no deadlock).
  subroutine rds_solve_groups(Rs, xp, yp)
    type(rds_t), intent(inout) :: Rs(:)
    PetscScalar, intent(in)    :: xp(:)
    PetscScalar, intent(inout) :: yp(:)
    integer :: k
    if (size(Rs) == 1) then
      call rds_solve_arr(Rs(1), xp, yp)
      return
    endif
    do k = 1, size(Rs)
      if (Rs(k)%member) call rds_gather(Rs(k), xp)
    enddo
    do k = 1, size(Rs)
      if (Rs(k)%member) call rds_local(Rs(k))
    enddo
    do k = 1, size(Rs)
      if (Rs(k)%member) call rds_return(Rs(k), yp)
    enddo
  end subroutine rds_solve_groups

  !> rooted R: the members' rows of xp into the root's R%b
  subroutine rds_gather(R, xp)
    type(rds_t), intent(inout) :: R
    PetscScalar, intent(in) :: xp(:)
    PetscScalar, pointer :: bp(:)
    PetscErrorCode :: ie
    integer :: q, nl, mpierr
    nl = size(R%loc)
    do q = 1, nl
      R%send(q) = xp(R%loc(q) + 1)
    enddo
    if (R%solver) then
      call VecGetArray(R%b, bp, ie)
      call MPI_Gatherv(R%send, nl, MPI_DOUBLE_PRECISION, bp, R%cnt, R%dsp(0:R%np - 1), &
                       MPI_DOUBLE_PRECISION, R%root, R%comm, mpierr)
      call VecRestoreArray(R%b, bp, ie)
    else
      call MPI_Gatherv(R%send, nl, MPI_DOUBLE_PRECISION, R%send, R%cnt, R%dsp(0:R%np - 1), &
                       MPI_DOUBLE_PRECISION, R%root, R%comm, mpierr)
    endif
  end subroutine rds_gather

  subroutine rds_local(R)
    type(rds_t), intent(inout) :: R
    PetscErrorCode :: ie
    if (R%solver) call KSPSolve(R%ksp, R%b, R%x, ie)
  end subroutine rds_local

  !> rooted R: the root's R%x back to the members' rows of yp
  subroutine rds_return(R, yp)
    type(rds_t), intent(inout) :: R
    PetscScalar, intent(inout) :: yp(:)
    PetscScalar, pointer :: bp(:)
    PetscErrorCode :: ie
    integer :: q, nl, mpierr
    nl = size(R%loc)
    if (R%solver) then
      call VecGetArrayRead(R%x, bp, ie)
      call MPI_Scatterv(bp, R%cnt, R%dsp(0:R%np - 1), MPI_DOUBLE_PRECISION, R%send, nl, &
                        MPI_DOUBLE_PRECISION, R%root, R%comm, mpierr)
      call VecRestoreArrayRead(R%x, bp, ie)
    else
      call MPI_Scatterv(R%send, R%cnt, R%dsp(0:R%np - 1), MPI_DOUBLE_PRECISION, R%send, nl, &
                        MPI_DOUBLE_PRECISION, R%root, R%comm, mpierr)
    endif
    do q = 1, nl
      yp(R%loc(q) + 1) = R%send(q)
    enddo
  end subroutine rds_return

  !> One V-cycle of hierarchy k on b (x out), outside any PC: the pair_psi
  !! eta-Schur applies it to Shat.
  subroutine gmg_vcycle(k, b, x)
    integer, intent(in) :: k
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_select(k)
    if (diag_left > 0 .and. cur_inst <= 2) call ring_diag(b)
    call PetscLogEventBegin(gev_vcycle(cur_inst), ierr)
    call vcycle_top(b, x)
    call PetscLogEventEnd(gev_vcycle(cur_inst), ierr)
  end subroutine gmg_vcycle

  !> The V-cycle on the caller's vectors. On the device path they are host
  !! vectors (the block's Krylov solver owns them): level 0's own pair, unused
  !! by the cycle otherwise, carries them to the device and back.
  subroutine vcycle_top(b, x)
    Vec :: b, x
    PetscErrorCode :: ierr
    if (dev) then
      call VecCopy(b, gb(0), ierr)
      call vcycle(0, gb(0), gx(0))
      call VecCopy(gx(0), x, ierr)
    else
      call vcycle(0, b, x)
    endif
  end subroutine vcycle_top

  !> PC-shell entry points of the pair_w (1), Shat (2), rho (3) and T (4)
  !! hierarchies. gmg_vcycle_apply is instance 1's historical name and is kept
  !! so the research path's PCShellSetApply is unchanged; gmg_pc_apply_1 is the
  !! same callback under the uniform name the production path selects by index.
  subroutine gmg_pc_apply_1(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_vcycle_apply(pc, b, x, ierr)
  end subroutine gmg_pc_apply_1

  subroutine gmg_pc_apply_2(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_vcycle(2, b, x)
    ierr = 0
  end subroutine gmg_pc_apply_2

  subroutine gmg_pc_apply_3(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_vcycle(3, b, x)
    ierr = 0
  end subroutine gmg_pc_apply_3

  subroutine gmg_pc_apply_4(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_vcycle(4, b, x)
    ierr = 0
  end subroutine gmg_pc_apply_4

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

    Mat, intent(in)      :: Aref
    integer, intent(in)  :: comm, my_id, n_fields
    logical, intent(out) :: ok
    type(gmg_opts_t), intent(in), optional :: opts
    type(gmg_opts_t) :: gopt

    type(lvl_t), allocatable :: lv(:)
    integer, allocatable :: o(:,:)
    ! (lv is handed to the module-level glv at the end, for the block maps)
    integer :: n_idx, e, iv, n, k, i, j, ni, nj, g, cand(4), nent, nrow_s, ncol_s
    integer :: eps_s(4), eps_t(4), cols(MAX_ENT), vi(4), vj(4), idx
    real*8  :: w(MAX_ENT)
    integer, allocatable :: rs(:), cs(:)
    real*8,  allocatable :: ws(:)
    logical, allocatable :: done(:)
    integer :: ns, nlev_cap
    type(lay_t), allocatable :: ly(:)
    integer, allocatable :: rstarts(:)
    integer :: me, np, mpierr, r, first, ci, cj, dd, own_, ff, mm, lr, nloc
    PetscInt :: rs0, re0, nglob
    PetscErrorCode :: ierr

    ok = .false.
    gcomm = comm; gme = my_id
    call gmg_register_events()
    call PetscLogEventBegin(gev_prolong, ierr)
    if (present(opts)) then
      gopt = opts
    else
      gopt = gmg_opts_from_namelist()
    endif
    call opt_real("ring_aspect", gopt%ring_aspect)
    semi_r = max(gopt%semi_r, -1)
    call opt_int("semi_r", semi_r)
    eps_s = [1, -1, -1, 1]
    eps_t = [1, 1, -1, -1]

    !--- gate 1: the npnew*(i-1)+j polar layout
    if (node_list%n_nodes /= n_flux * n_tht) then
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

    !--- gate 2: per-node orientation o = +-1, consistent across elements
    allocate(o(node_list%n_nodes, 0:3))
    o = 0
    do e = 1, element_list%n_elements
      do iv = 1, 4
        n = element_list%element(e)%vertex(iv)
        cand = nint([element_list%element(e)%size(iv, 1), &
                     eps_s(iv) * element_list%element(e)%size(iv, 2), &
                     eps_t(iv) * element_list%element(e)%size(iv, 3), &
                     eps_s(iv) * eps_t(iv) * element_list%element(e)%size(iv, 4)])
        do k = 0, 3
          if (abs(cand(k + 1)) /= 1) then
            call fail("element size factor is not +-1 (fix_axis_nodes?)")
            return
          endif
          if (o(n, k) == 0) then
            o(n, k) = cand(k + 1)
          else if (o(n, k) /= cand(k + 1)) then
            call fail("inconsistent DOF orientation between elements")
            return
          endif
        enddo
      enddo
    enddo

    !--- coarse levels. Radial semi-coarsening (semi_r): on cells long in theta
    !--- the elliptic couplings are strongest radially, which the ring blocks do
    !--- not smooth, and a full-coarsening coarse grid cannot take the error
    !--- they leave; the levels that coarsen in I alone can (line relaxation
    !--- with semi-coarsening across the lines, robust for either anisotropy).
    if (semi_r < 0) then
      block
        real*8, allocatable :: ar(:)
        real*8 :: asp, xa(2), xb(2), xc(2)
        integer :: na, ii, jj
        allocate(ar((n_flux - 2) * n_tht))
        na = 0
        do ii = 1, n_flux - 2
          do jj = 0, n_tht - 1
            xa = node_list%node(ii * n_tht + jj + 1)%x(1, 1, 1:2)
            xb = node_list%node((ii + 1) * n_tht + jj + 1)%x(1, 1, 1:2)
            xc = node_list%node(ii * n_tht + mod(jj + 1, n_tht) + 1)%x(1, 1, 1:2)
            na = na + 1
            ar(na) = norm2(xc - xa) / max(norm2(xb - xa), 1.d-300)
          enddo
        enddo
        call sort_real(ar(1:na))
        asp = ar((na + 1) / 2)
        semi_r = 0
        do while (asp > SEMI_ASPECT .and. semi_r < MAX_LEV - 1)
          semi_r = semi_r + 1; asp = asp / 2
        enddo
        if (my_id == 0) write(*,'(A,I0,A,I0,A,F6.2,A)') "[Physics PC]   GMG", cur_inst, &
          ": radial semi-coarsening on levels 1..", semi_r, " (median r dtheta/dr ", ar((na + 1) / 2), ")"
        deallocate(ar)
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

    !--- the fine-level axis patch: every DOF of every axis-ring node (owned rows)
    call make_axis_is(n_idx)

    !--- Workstream D: packed LOCAL row -> (node, harmonic) maps for the block
    !--- smoothers, fine level (fine_node/fine_harm) and every coarse level
    nf_s = n_fields
    nth0 = n_tht
    nloc = n_fields * pc_ntor * ly(0)%nl(me)
    allocate(fine_node(nloc), fine_harm(nloc), fine_kf(nloc))
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
            fine_kf(lr) = (k - 1) + 4 * ff
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

    !--- Workstream D (flux-geometry robustness): cell aspect ratio, from the
    !--- vertex positions, skipping the axis ring where the angular side vanishes
    block
      real*8, allocatable :: ar(:)
      real*8 :: dr, dt, xa(2), xb(2), xc(2), tmp
      integer :: na, ii, jj, kk
      allocate(ar((n_flux - 2) * n_tht))
      na = 0
      do ii = 1, n_flux - 2
        do jj = 0, n_tht - 1
          xa = node_list%node(ii * n_tht + jj + 1)%x(1, 1, 1:2)
          xb = node_list%node((ii + 1) * n_tht + jj + 1)%x(1, 1, 1:2)
          xc = node_list%node(ii * n_tht + mod(jj + 1, n_tht) + 1)%x(1, 1, 1:2)
          dr = norm2(xb - xa); dt = norm2(xc - xa)
          na = na + 1
          ar(na) = dt / max(dr, 1.d-300)        ! > 1: cell long in theta
        enddo
      enddo
      call sort_real(ar(1:na))                   ! na = (n_flux-2) n_tht cells
      if (my_id == 0) write(*,'(A,F8.2,A,F8.2,A,F8.2)') &
        "[Physics PC]   GMG grid cells, rdtheta/dr: median ", ar((na + 1) / 2), &
        ", max ", ar(na), ", min ", ar(1)
      deallocate(ar)

      ! Per-ring median and the switch ring I_s: the first ring whose median
      ! reaches physics_pc_gmg_ring_aspect (GMGPolar's circle/radial criterion).
      ! Ring ii holds the cells between flux surfaces ii and ii+1.
      block
        real*8, allocatable :: rm(:), row(:)
        integer :: nshow
        allocate(rm(n_flux - 2), row(n_tht))
        do ii = 1, n_flux - 2
          do jj = 0, n_tht - 1
            xa = node_list%node(ii * n_tht + jj + 1)%x(1, 1, 1:2)
            xb = node_list%node((ii + 1) * n_tht + jj + 1)%x(1, 1, 1:2)
            xc = node_list%node(ii * n_tht + mod(jj + 1, n_tht) + 1)%x(1, 1, 1:2)
            row(jj + 1) = norm2(xc - xa) / max(norm2(xb - xa), 1.d-300)
          enddo
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
          nshow = min(n_flux - 2, ring_is + 2)
          write(*,'(A,F5.2,A,I0,A)', advance="no") "[Physics PC]   GMG ring medians rdtheta/dr (switch at ", &
            gopt%ring_aspect, ": rings I < ", ring_is, "):"
          do ii = 1, nshow
            write(*,'(A,F6.2)', advance="no") " ", rm(ii)
          enddo
          write(*,*)
        endif
        deallocate(rm, row)
      end block
    end block

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

    subroutine make_axis_is(n_idx_)
      integer, intent(in) :: n_idx_
      PetscInt, allocatable :: ax(:)
      logical, allocatable  :: seen(:)
      PetscErrorCode :: ierr_
      integer :: nn, kk, f, m, nax, id

      allocate(seen(n_idx_), ax(n_fields * pc_ntor * 4 * n_tht * 2))
      seen = .false.; nax = 0
      do nn = 1, node_list%n_nodes
        if ((nn - 1) / n_tht /= 0) cycle          ! first ring = the axis
        do kk = 1, 4
          id = node_list%node(nn)%index(kk)
          if (seen(id)) cycle
          seen(id) = .true.
          if (ly(0)%own(id - 1) /= me) cycle      ! owned rows only: no duplicates at np > 1
          do f = 0, n_fields - 1
            do m = 0, pc_ntor - 1
              nax = nax + 1
              ax(nax) = prow(0, f, id - 1, m)
            enddo
          enddo
        enddo
      enddo
      call ISCreateGeneral(comm, nax, ax(1:nax), PETSC_COPY_VALUES, gisAxis, ierr_)
      deallocate(seen, ax)
    end subroutine make_axis_is

  end subroutine gmg_build_prolongations

  !> Canonical DOF numbering of one coarse level. The axis keeps the shared
  !! value and per-node a and c; its angular slope b is dropped (see header).
  !! bnd_drop (stage D13): the boundary ring also drops u and the angular
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
  subroutine gmg_setup_operator(A, comm, my_id, Afine, tag, smoother, nsmooth, opts, Ablk)
    Mat, intent(in)     :: A
    integer, intent(in) :: comm, my_id
    Mat, intent(in), optional :: Afine  !< Workstream D: applies the fine operator
                                        !< for every level-0 matvec; A still gives the
                                        !< Galerkin chain, Jacobi diagonal and axis block
    Mat, intent(in), optional :: Ablk   !< the level-0 smoother blocks (lines, zebra
                                        !< coupling, axis block) from Ablk instead of A;
                                        !< same layout as A, frozen pattern
    character(len=*), intent(in), optional :: tag   !< label for the prints
    integer, intent(in), optional :: smoother, nsmooth   !< override the physics_pc_gmg_* knobs
    type(gmg_opts_t), intent(in), optional :: opts    !< the whole configuration; absent =
                                                      !< gmg_opts_from_namelist()
    type(gmg_opts_t) :: o
    PetscErrorCode :: ierr
    PC   :: pc
    Mat  :: Aax
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
      if (.not. sm_blocks) call KSPDestroy(gAxis, ierr)   ! sm_blocks: still last setup's
    endif

    call gmg_register_events()
    gA(0) = A
    gF = A
    if (present(Afine)) gF = Afine
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
    if (present(opts)) then
      o = opts
    else
      o = gmg_opts_from_namelist()
    endif
    sm_type = o%smoother
    sm_nstep = o%nsmooth
    call dev_operators(my_id, reuse)
    ! threaded matvecs: after the Galerkin chain, on every rebuild (the attach
    ! is idempotent and only re-plans on a new pattern). gA(0) is the caller's
    ! operator; P serves the V-cycle's prolongation (MatMultAdd). Not on the
    ! device path, whose V-cycle multiplies with the device copies.
    if (o%blockmv == 1 .and. .not. dev) then
      block
        use mod_petsc_pc_harm, only: pc_ntor
        integer :: gg
        logical :: okb
        do gg = 0, nlev - 1
          okb = blockmv_attach(gA(gg), int(pc_ntor))
          if (gg > 0) okb = blockmv_attach(gP(gg), 1)
        enddo
        if (gF /= gA(0)) okb = blockmv_attach(gF, int(pc_ntor))   ! a shell: left alone
        if (.not. op_ready .and. my_id == 0) write(*,'(A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, &
          ": level matvecs on the OpenMP block kernel (bs ", pc_ntor, ", ", nlev, " levels)"
      end block
    endif
    if (present(smoother)) then
      if (smoother >= 0) sm_type = smoother
    endif
    call opt_int("smoother", sm_type)
    if (present(nsmooth)) then
      if (nsmooth > 0) sm_nstep = nsmooth
    endif
    if (sm_nstep <= 0) then
      sm_nstep = SM_STEPS
      if (sm_type == 1 .or. sm_type == 2) sm_nstep = 3
    endif
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
    sm_blocks = (sm_type >= 2)
    axis_k = 0
    if (sm_type >= 4) axis_k = max(o%axis_rings, -1)
    if (sm_type >= 4) call opt_int("axis_rings", axis_k)
    axis_mult = 0
    if (axis_k /= 0) axis_mult = min(max(o%axis_mult, 0), 3)
    axis_droptol = max(o%axis_droptol, 0.d0)
    lovl = 0
    if (sm_type == 5 .or. sm_type >= 7) lovl = max(o%line_overlap, 0)
    if (sm_type == 5 .or. sm_type >= 7) call opt_int("line_overlap", lovl)
    hpair = max(o%harm_pair, 0)
    call opt_int("harm_pair", hpair)
    rovl = 0
    if (sm_type == 4 .or. sm_type == 9) rovl = max(o%ring_overlap, 0)
    if (sm_type == 4 .or. sm_type == 9) call opt_int("ring_overlap", rovl)
    axsec = 0
    if (axis_k /= 0 .and. axis_mult == 0) axsec = o%axis_sectors
    if (axis_k /= 0 .and. axis_mult == 0) call opt_int("axis_sectors", axsec)
    ! per-|n| axis solves need the block to be block-diagonal in |n|
    axis_split = (axis_k /= 0 .and. o%axis_split > 0 .and. o%harm_split > 0)
    if (axis_k /= 0 .and. o%axis_split > 0 .and. o%harm_split == 0 .and. my_id == 0) &
      write(*,'(A)') "[Physics PC]   GMG: physics_pc_gmg_axis_split needs physics_pc_harm_split = 1; ignored"
    diag_left = max(o%ring_diag, 0)
    ! Levels g >= rich_from smooth with Richardson on the same blocks instead
    ! of GMRES: no norm, no inner product, so no global reduction below that
    ! level. GMRES costs two allreduces over the whole communicator per step,
    ! on coarse levels with a few hundred rows per rank and on ranks owning
    ! none -- the V-cycle's strong-scaling floor at np >= 32.
    rich_from = nlev
    if (o%rich_from >= 0) rich_from = o%rich_from
    rich_omega = 1.d0
    if (sm_type == 1 .or. sm_type == 2) rich_omega = o%omega
    if (o%rich_omega > 0.d0) rich_omega = o%rich_omega
    call opt_int("rich_from", rich_from)
    call opt_real("rich_omega", rich_omega)
    do g = 0, nlev - 2
      ! Created and configured once per hierarchy; a rebuild only re-points
      ! the operators and refactors the smoother blocks.
      if (.not. op_ready) call KSPCreate(comm, gSm(g), ierr)
      if (g == 0 .and. o%smooth_op == 0) then
        call KSPSetOperators(gSm(g), gFd, gAd(g), ierr)   ! Jacobi reads the Pmat
      else
        ! smooth_op 1: the fine smoother iterates on the assembled surrogate;
        ! only the V-cycle residual and the outer Krylov apply gF (the shell)
        ! -- one exact-mass solve per matvec there instead of per GMRES step
        call KSPSetOperators(gSm(g), gAd(g), gAd(g), ierr)
      endif
      if (.not. op_ready) then
        if (sm_type == 1 .or. sm_type == 2 .or. g >= rich_from) then
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
        if (sm_type == 9) then
          ! alternating: ring blocks, then radial lines on the updated residual,
          !   y = R^-1 x;  y = y + L^-1 (x - A y)
          call PCSetType(pc, PCCOMPOSITE, ierr)
          call PCCompositeSetType(pc, PC_COMPOSITE_MULTIPLICATIVE, ierr)
          call PCCompositeAddPCType(pc, PCSHELL, ierr)
          call PCCompositeAddPCType(pc, PCSHELL, ierr)
          block
            PC :: sub
            call PCCompositeGetPC(pc, 0, sub, ierr)
            call PCShellSetApply(sub, blk_apply, ierr)
            call PCShellSetName(sub, "C1 ring blocks", ierr)
            call PCCompositeGetPC(pc, 1, sub, ierr)
            call PCShellSetApply(sub, blk_apply_set2, ierr)
            call PCShellSetName(sub, "C1 radial-line blocks", ierr)
          end block
        else if (sm_blocks) then
          call PCSetType(pc, PCSHELL, ierr)
          call PCShellSetApply(pc, blk_apply, ierr)
          call PCShellSetName(pc, "C1 node-block Jacobi", ierr)
        else
          call PCSetType(pc, PCJACOBI, ierr)
        endif
      endif
      if (sm_type == 9) then
        ! the two partitions, each built as its own smoother kind: rings (4,
        ! no line overlap), then radial lines (5, with it) into gBk2
        block
          integer :: lovl_
          lovl_ = lovl
          sm_type = 4; lovl = 0
          if (g == 0 .and. present(Ablk)) then
            call build_blocks(g, Ablk)
          else
            call build_blocks(g, gA(g))
          endif
          sm_type = 5; lovl = lovl_
          if (g == 0 .and. present(Ablk)) then
            call build_blocks(g, Ablk, set2=.true.)
          else
            call build_blocks(g, gA(g), set2=.true.)
          endif
          sm_type = 9
        end block
      else if (sm_blocks) then
        if (g == 0 .and. present(Ablk)) then
          call build_blocks(g, Ablk)
        else
          call build_blocks(g, gA(g))
        endif
      endif
      call KSPSetUp(gSm(g), ierr)
    enddo
    call axd_factor_pending()
    if (dev .and. .not. op_ready) call blk_dev_gate(my_id)
    call PetscLogEventEnd(gev_smsetup, ierr)
    block
      integer :: ib(3), ibg(3), mpierr
      ib = 0
      if (sm_blocks) ib = [gBk(0)%nb, count(gBk(0)%band), sum(gBk(0:nlev-2)%nsing)]
      call MPI_Allreduce(ib, ibg, 3, MPI_INTEGER, MPI_SUM, comm, mpierr)
      ib(1) = 0
      if (sm_blocks) ib(1) = maxval(gBk(0)%kl)
      call MPI_Allreduce(MPI_IN_PLACE, ib(1), 1, MPI_INTEGER, MPI_MAX, comm, mpierr)
      if (my_id == 0) then
        if (present(tag)) write(*,'(A,A,A)', advance="no") "[Physics PC]   GMG (", tag, ")"
        write(*,'(A,I0,A,I0,A,I0,A,I0,A,F5.2)', advance="no") "[Physics PC]   GMG smoother ", sm_type, &
          ": V(", st_pre0(cur_inst), ",", st_post0(cur_inst), ") on level 0, ", st_crs(cur_inst), &
          " steps per side below, omega = ", o%omega
        if (sm_blocks) then
          write(*,'(A,I0,A,I0,A,I0,A,I0,A)') ", blocks ", ibg(1), " on level 0 (", &
            ibg(2), " banded, max kl ", ib(1), "; ", ibg(3), " singular -> point)"
        else
          write(*,*)
        endif
        if (sm_type == 6 .or. sm_type == 8) write(*,'(A,I0,A)', advance="no") &
          "[Physics PC]   GMG hybrid smoother: ring blocks on fine rings I < ", ring_is, &
          ", radial lines outside"
        if (sm_type >= 4 .and. axis_k > 0) write(*,'(A,I0,A)', advance="no") &
          "; axis block = rings 0..", axis_k, " on every level"
        if (sm_type >= 4 .and. axis_k < 0) write(*,'(A,I0,A)', advance="no") &
          "[Physics PC]   GMG axis block = rings 0..I_s-1 of every level (fine: 0..", axis_lim(0), ")"
        if (axis_mult > 0) write(*,'(A,I0)', advance="no") ", axis/lines Gauss-Seidel mode ", axis_mult
        if (sm_type == 6 .or. sm_type == 8 .or. (sm_type >= 4 .and. axis_k /= 0)) write(*,*)
        if (rich_from < nlev - 1 .and. sm_type /= 1 .and. sm_type /= 2) write(*,'(A,I0,A,I0,A,F5.2,A)') &
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
    ! Wiring gate (one coarse solve): the redundant solve must be exact.
    ! Workstream G, finding A1: FIRST BUILD ONLY. It used to run on every PC
    ! rebuild -- a random vector, two MatMults, a full coarse solve and four
    ! VecNorms (four collectives) -- while printing only on the first build or
    ! on failure. What it gates is the WIRING of rds_t against the coarse
    ! operator, and the pattern is frozen after the first build, so a later
    ! rebuild cannot break the wiring without also changing the pattern (which
    ! pattern_sig catches on its own). Apply-neutral either way.
    if (.not. op_ready) then
    block
        Vec :: cb, cx, cr
        real*8 :: rn, bn, xn, en
        call MatCreateVecs(gA(nlev - 1), cx, cb, ierr)
        call VecDuplicate(cb, cr, ierr)
        ! manufactured solution: b = A xt, so both the residual and the
        ! error of the solve are known (an ill-conditioned coarse operator
        ! shows as a small residual with a large error)
        call VecSetRandom(cr, PETSC_NULL_RANDOM, ierr)               ! xt
        call MatMult(gA(nlev - 1), cr, cb, ierr)
        call VecZeroEntries(cx, ierr)
        call rds_solve(gcrs, cb, cx)
        call VecNorm(cr, NORM_2, en, ierr)
        call VecAXPY(cr, -1.0d0, cx, ierr)                           ! xt - x
        call VecNorm(cr, NORM_2, xn, ierr)
        call MatMult(gA(nlev - 1), cx, cr, ierr)
        call VecAXPY(cr, -1.0d0, cb, ierr)
        call VecNorm(cr, NORM_2, rn, ierr)
        call VecNorm(cb, NORM_2, bn, ierr)
        if (my_id == 0 .and. (.not. op_ready .or. rn > 1.d-8 * bn)) write(*,'(A,A,ES10.3,A,ES10.3)') &
          "[Physics PC]   GMG coarse solve self-check", merge(": WARNING, inexact! ", ":                   ", &
          rn > 1.d-8 * bn), rn / max(bn, 1.d-300), " (residual), ", xn / max(en, 1.d-300)
        call VecDestroy(cb, ierr); call VecDestroy(cx, ierr); call VecDestroy(cr, ierr)
    end block
    endif

    ! The exact axis patch is only applied with the point-Jacobi smoothers; the
    ! block smoothers carry the axis ring as one block of their own.
    if (.not. sm_blocks) then
      call PetscLogEventBegin(gev_axislu, ierr)
      call MatCreateSubMatrix(gA(0), gisAxis, gisAxis, MAT_INITIAL_MATRIX, Aax, ierr)
      call KSPCreate(comm, gAxis, ierr)
      call KSPSetOperators(gAxis, Aax, Aax, ierr)
      call KSPSetType(gAxis, KSPPREONLY, ierr)
      call KSPGetPC(gAxis, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
      call set_prefix_mumps(gAxis, "axis")
      call KSPSetUp(gAxis, ierr)
      call PetscLogEventEnd(gev_axislu, ierr)
    endif

    if (.not. vec_ready) then
      do g = 0, nlev - 1
        call MatCreateVecs(gAd(g), gx(g), gb(g), ierr)
        call VecDuplicate(gx(g), gr(g), ierr)
      enddo
      if (.not. sm_blocks) call MatCreateVecs(Aax, gzax, PETSC_NULL_VEC, ierr)
      vec_ready = .true.
    endif
    if (.not. sm_blocks) call MatDestroy(Aax, ierr)        ! gAxis holds its own reference
    ! Workstream G, findings A2/A3: the boundary-row scans and the C_op report
    ! are FIRST BUILD ONLY. Both describe the hierarchy's STRUCTURE -- which is
    ! frozen after the first build -- yet re-ran every rebuild, the C_op loop
    ! costing 2*nlev collectives each time. Apply-neutral.
    pat_fresh = .false.
    if (op_ready) return
    op_ready = .true.
    if (diag_left > 0) call report_bnd_rows(gA(0))
    if (o%bnd_drop > 0) call check_bnd_rows(gA(0))

    ! Operator complexity: the whole point against GAMG's C_op ~ 1.00 (T2).
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

  !> The operators the V-cycle multiplies with: gAd / gPd / gFd. Host path:
  !! the level operators themselves. Device path (-sf_gpu 1, or -sf_gmg<k>_gpu
  !! for one hierarchy): copies of type -sf_gpu_mat_type (default aijkokkos,
  !! PETSc's portable backend), created at the first build or for a new
  !! pattern and refilled from the host values at every other rebuild. A
  !! matrix-free gF stays as it is.
  subroutine dev_operators(my_id, reuse)
    integer, intent(in) :: my_id
    logical, intent(in) :: reuse
    character(len=64), save :: mtype = "aijkokkos"
    character(len=80) :: tf
    PetscErrorCode :: ierr
    PetscInt :: pv
    PetscBool :: set
    integer :: g, idev
    logical :: fresh, own_f

    if (.not. op_ready) then
      pv = 0
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gpu", pv, set, ierr)
      idev = int(pv)
      call opt_int("gpu", idev)
      dev = (idev > 0)
      call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gpu_mat_type", mtype, set, ierr)
      pv = 1
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_gpu_blocks", pv, set, ierr)
      dev_blocks = (pv > 0 .and. blk_dev_available())
    endif
    if (.not. dev) then
      gAd = gA; gPd = gP; gFd = gF
      return
    endif
    call PetscLogEventBegin(gev_dev, ierr)
    fresh = .not. (op_ready .and. reuse)
    call MatGetType(gF, tf, ierr)
    own_f = (gF /= gA(0) .and. index(tf, "aij") > 0)
    do g = 0, nlev - 1
      if (fresh) then
        if (op_ready) call MatDestroy(gAd(g), ierr)
        call dev_copy(gA(g), gAd(g))
      else
        call dev_refill(gA(g), gAd(g))
      endif
    enddo
    if (.not. op_ready) then
      do g = 1, nlev - 1
        call dev_copy(gP(g), gPd(g))
      enddo
    endif
    if (own_f) then
      if (fresh) then
        if (op_ready) call MatDestroy(gFd, ierr)
        call dev_copy(gF, gFd)
      else
        call dev_refill(gF, gFd)
      endif
    else if (gF == gA(0)) then
      gFd = gAd(0)
    else
      gFd = gF
    endif
    call PetscLogEventEnd(gev_dev, ierr)
    if (.not. op_ready .and. my_id == 0) then
      call MatGetType(gAd(0), tf, ierr)
      write(*,'(A,I0,A,A,A,I0,A)') "[Physics PC]   GMG", cur_inst, ": V-cycle operators on the device path (", &
        trim(tf), ", ", nlev, " levels)"
    endif

  contains

    !> D = a device-type copy of S. Not MatConvert(S, ..., MAT_INITIAL_MATRIX):
    !! PETSc 3.25.3's MPIAIJ -> MPIAIJKokkos conversion then retypes the
    !! SOURCE's diagonal/off-diagonal parts and ghost vector instead of the
    !! copy's (mpiaijkok.kokkos.cxx, MatConvert_MPIAIJ_MPIAIJKokkos), which
    !! breaks the host operator the setup keeps using. In place it is right.
    subroutine dev_copy(S, D)
      Mat, intent(in)  :: S
      Mat, intent(out) :: D
      call MatDuplicate(S, MAT_COPY_VALUES, D, ierr)
      call MatConvert(D, trim(mtype), MAT_INPLACE_MATRIX, D, ierr)
    end subroutine dev_copy
  end subroutine dev_operators

  !> First build: the device block solves of every level against the host
  !! factors they were copied from, on a random vector. Same algorithm, so the
  !! difference is round-off; a level that misses DEV_GATE_TOL goes back to the
  !! host solves (on every rank).
  subroutine blk_dev_gate(my_id)
    integer, intent(in) :: my_id
    real*8, parameter :: DEV_GATE_TOL = 1.d-10
    PC  :: nopc
    Vec :: x, y1, y2
    PetscErrorCode :: ierr
    real*8 :: dn, yn, st(3), sm
    logical :: has
    integer :: g, lev0, mpierr

    lev0 = cur_lev
    do g = 0, nlev - 2
      has = c_associated(gBk(g)%dh)
      call MPI_Allreduce(MPI_IN_PLACE, has, 1, MPI_LOGICAL, MPI_LOR, gcomm, mpierr)
      if (.not. has) cycle
      call MatCreateVecs(gAd(g), x, y1, ierr)
      call VecDuplicate(y1, y2, ierr)
      call VecSetRandom(x, PETSC_NULL_RANDOM, ierr)
      call VecZeroEntries(y1, ierr)
      call VecZeroEntries(y2, ierr)
      cur_lev = g
      blk_host = .true.
      call blk_apply(nopc, x, y1, ierr)
      blk_host = .false.
      call blk_apply(nopc, x, y2, ierr)
      call VecNorm(y1, NORM_2, yn, ierr)
      call VecAXPY(y1, -1.0d0, y2, ierr)
      call VecNorm(y1, NORM_2, dn, ierr)
      call VecDestroy(x, ierr); call VecDestroy(y1, ierr); call VecDestroy(y2, ierr)
      dn = dn / max(yn, 1.d-300)
      st = 0.d0
      if (c_associated(gBk(g)%dh)) st = gBk(g)%dstat
      sm = st(3)
      call MPI_Allreduce(MPI_IN_PLACE, st(1:2), 2, MPI_DOUBLE_PRECISION, MPI_MAX, gcomm, mpierr)
      call MPI_Allreduce(MPI_IN_PLACE, sm, 1, MPI_DOUBLE_PRECISION, MPI_SUM, gcomm, mpierr)
      if (.not. (dn <= DEV_GATE_TOL)) call blk_dev_free(gBk(g)%dh)
      if (my_id == 0) write(*,'(A,I0,A,I0,A,ES9.2,A,A,I0,A,I0,A,ES9.2,A)') "[Physics PC]   GMG", cur_inst, &
        " level ", g, ": device block solves vs host ", dn, &
        merge(" -> in use       ", " -> WARNING: host", dn <= DEV_GATE_TOL), &
        " (largest block ", nint(st(1)), ", widest band ", nint(st(2)), ", ", sm, " mult-adds per pass)"
    enddo
    cur_lev = lev0
  end subroutine blk_dev_gate

  !> D's values = S's, D a converted copy of S (same CSR, entry by entry).
  subroutine dev_refill(S, D)
    Mat, intent(in) :: S, D
    Mat :: Sd, So, Dd, Do_
    PetscInt, pointer :: garr(:)
    logical :: has_o

    call aij_parts(S, Sd, So, garr, has_o)
    call aij_parts(D, Dd, Do_, garr, has_o)
    call copy_part(Sd, Dd)
    if (has_o) call copy_part(So, Do_)

  contains

    subroutine copy_part(Ms, Md)
      Mat, intent(in) :: Ms, Md
      PetscInt :: n
      PetscInt, pointer :: ia(:), ja(:)
      integer(8) :: nz
      type(c_ptr) :: ps, pd
      real(c_double), pointer :: vs(:), vd(:)
      call get_ij(Ms, .false., n, ia, ja)
      nz = int(ia(n + 1), 8)
      call put_ij(Ms, .false., n, ia, ja)
      if (nz == 0) return
      call aij_vals_read(Ms, nz, ps, vs)
      call aij_vals_write(Md, nz, pd, vd)
      vd(1:nz) = vs(1:nz)
      call aij_vals_written(Md, pd)
      call aij_vals_done(Ms, ps)
    end subroutine copy_part
  end subroutine dev_refill

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

  !> Options prefix gmg<k>_<what>_ for a direct-solve KSP of hierarchy k, and
  !! MUMPS' centralized RHS under that prefix unless the user set it: PETSc's
  !! np > 1 default (distributed RHS) corrupts the heap on our builds and is
  !! read only from the options database (see mod_petsc petsc_initialize).
  subroutine set_prefix_mumps(ksp, what)
    KSP :: ksp
    character(len=*), intent(in) :: what
    character(len=64) :: pre, nm
    PetscBool :: has
    PetscErrorCode :: ierr
    write(pre, '(A,I0,A,A,A)') "gmg", cur_inst, "_", what, "_"
    call KSPSetOptionsPrefix(ksp, trim(pre), ierr)
    nm = "-"//trim(pre)//"mat_mumps_icntl_20"
    call PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, trim(nm), has, ierr)
    if (.not. has) call PetscOptionsSetValue(PETSC_NULL_OPTIONS, trim(nm), "0", ierr)
    call KSPSetFromOptions(ksp, ierr)
  end subroutine set_prefix_mumps

  !> PCSHELL apply: x = one V-cycle applied to b.
  subroutine gmg_vcycle_apply(pc, b, x, ierr)
    PC  :: pc
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_select(1)                  ! the PC-shell entry point is pair_w's
    if (diag_left > 0) call ring_diag(b)
    call PetscLogEventBegin(gev_vcycle(cur_inst), ierr)
    call vcycle_top(b, x)
    call PetscLogEventEnd(gev_vcycle(cur_inst), ierr)
    ierr = 0
  end subroutine gmg_vcycle_apply

  subroutine gmg_register_events()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    if (gev_ready) return
    call PetscLogEventRegister("GMG_PtAP",     cid, gev_ptap,     ierr)
    call PetscLogEventRegister("GMG_Prolong",  cid, gev_prolong,  ierr)
    call PetscLogEventRegister("GMG_DevOps",   cid, gev_dev,      ierr)
    call PetscLogEventRegister("GMG_SmSetup",  cid, gev_smsetup,  ierr)
    call PetscLogEventRegister("GMG_CoarseLU", cid, gev_coarselu, ierr)
    call PetscLogEventRegister("GMG_BlkMesh",  cid, gev_blkmesh,  ierr)
    call PetscLogEventRegister("GMG_BlkPat",   cid, gev_blkpat,   ierr)
    call PetscLogEventRegister("GMG_BlkNum",   cid, gev_blknum,   ierr)
    call PetscLogEventRegister("GMG_AxNum",    cid, gev_axnum,    ierr)
    call PetscLogEventRegister("GMG_AxTry",    cid, gev_axtry,    ierr)
    call PetscLogEventRegister("GMG_AxisLU",   cid, gev_axislu,   ierr)
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
    call PetscLogEventRegister("GMG_Axis",     cid, gev_axis,     ierr)
    call PetscLogEventRegister("GMG_Lines",    cid, gev_lines(1),   ierr)
    call PetscLogEventRegister("GMG_AxSolve",  cid, gev_axsolve(1), ierr)
    call PetscLogEventRegister("GMG2_Lines",   cid, gev_lines(2),   ierr)
    call PetscLogEventRegister("GMG2_AxSolve", cid, gev_axsolve(2), ierr)
    call PetscLogEventRegister("GMG3_Lines",   cid, gev_lines(3),   ierr)
    call PetscLogEventRegister("GMG3_AxSolve", cid, gev_axsolve(3), ierr)
    call PetscLogEventRegister("GMG4_Lines",   cid, gev_lines(4),   ierr)
    call PetscLogEventRegister("GMG4_AxSolve", cid, gev_axsolve(4), ierr)
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
    if (g == 0 .and. .not. sm_blocks) call axis_patch(b, x)
    if (npre == 0 .and. .not. (g == 0 .and. .not. sm_blocks)) then
      call VecCopy(b, gr(g), ierr)                        ! x is still 0: r = b
    else
      if (g == 0) then
        call MatMult(gFd, x, gr(g), ierr)
      else
        call MatMult(gAd(g), x, gr(g), ierr)
      endif
      call VecAYPX(gr(g), -1.0d0, b, ierr)                ! r = b - A x
    endif
    call MatMultTranspose(gPd(g + 1), gr(g), gb(g + 1), ierr)
    call vcycle(g + 1, gb(g + 1), gx(g + 1))
    call MatMultAdd(gPd(g + 1), gx(g + 1), x, x, ierr)    ! x += P e_c
    if (npost > 0) call smooth(g, b, x, .false., npost)  ! post-smooth
    if (g == 0 .and. .not. sm_blocks) call axis_patch(b, x)
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
  subroutine build_blocks(g, A, set2)
    use mod_petsc_pc_harm, only: pc_ntor, pc_ngrp, pc_grp
    integer, intent(in) :: g
    Mat, intent(in)     :: A
    logical, intent(in), optional :: set2   !< build into gBk2 (smoother 9's lines)
    type(blk_t), pointer :: B
    PetscInt :: nr, r, ncols, rst, ren
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    logical :: devpat
    PetscErrorCode :: ierr
    PetscInt, parameter :: one = 1
    integer :: nc, I, J, m, bb, q, pcn, info, n, ldab, kk, tmp, nthr, ngrp, mpierr
    integer, allocatable :: cnt(:), key(:)
    integer(8) :: ix
    PetscInt, allocatable :: axr(:)
    integer, allocatable :: binfo(:)
    external :: dgetrf, dgbtrf

    B => gBk(g)
    if (present(set2)) then
      if (set2) B => gBk2(g)
    endif
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
        B%bid(r) = blk_id(g, I, J, m, nc)
        if (sm_type == 5 .or. sm_type == 7) key(r) = I
        if (sm_type == 4) then
          ! a ring is periodic in J: ordered 0, nc-1, 1, nc-2, ... (as smoother
          ! 8's rings) its J +- 1 couplings, the wrap included, lie within two
          ! nodes -- a band instead of a dense block
          if (J < nc - J) then
            key(r) = 2 * J
          else
            key(r) = 2 * (nc - 1 - J) + 1
          endif
        endif
        if (sm_type == 6) then                     ! rings run along J, radial lines along I
          if (I < ring_lim(g)) then
            key(r) = J
          else
            key(r) = I
          endif
        endif
        if (sm_type == 8) then
          ! a ring is periodic in J: ordered 0, nc-1, 1, nc-2, ... its J +- 1
          ! couplings, the wrap included, lie within two nodes -- a narrow band
          ! instead of a dense block
          if (I < ring_lim(g)) then
            if (J < nc - J) then
              key(r) = 2 * J
            else
              key(r) = 2 * (nc - 1 - J) + 1
            endif
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
      ! Stage D13: the axis block(s) go to a sparse direct solve
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
          if (I /= 0 .and. .not. B%axblk(bb)) B%bcol(bb) = mod(J, 2)
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
        ! the same rows, local and ascending, per axis group (key = group + 1)
        ngrp = 1
        if (axis_split) ngrp = pc_ngrp()
        allocate(B%axg(ngrp), key(B%nrow))
        key = 0
        do q = 1, n
          r = axr(q) - rst + 1
          key(r) = 1
          if (axis_split) then
            if (g == 0) then
              m = fine_harm(r)
            else
              m = glv(g)%rharm(r)
            endif
            key(r) = pc_grp(m) + 1                   ! |n| group of slot m (cos/sin together)
          endif
        enddo
        do kk = 1, ngrp
          allocate(B%axg(kk)%loc(count(key == kk)))
          n = 0
          do r = 1, B%nrow
            if (key(r) /= kk) cycle
            n = n + 1; B%axg(kk)%loc(n) = int(r) - 1
          enddo
        enddo
        deallocate(axr, key)
      endif
      if (lovl > 0 .and. sm_type /= 4) call ovl_extend(g, A, B, rst, ren, lovl, .false.)
      if (rovl > 0 .and. sm_type == 4) call ovl_extend(g, A, B, rst, ren, rovl, .true.)
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
      devpat = newpat
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
    !$omp parallel do schedule(dynamic, 1) private(bb, n, ldab)
    do bb = 1, B%nb
      if (B%axblk(bb)) cycle
      n = B%sz(bb)
      if (B%band(bb)) then
        ldab = 2 * B%kl(bb) + B%ku(bb) + 1
        call dgbtrf(n, n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), ldab, B%piv(B%off(bb) + 1), binfo(bb))
      else
        call dgetrf(n, n, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), binfo(bb))
      endif
    enddo
    !$omp end parallel do
    B%nsing = 0
    do bb = 1, B%nb
      n = B%sz(bb)
      if (B%axblk(bb)) cycle
      info = binfo(bb)
      if (info /= 0) then
        ! singular block: keep only its diagonal, from the Pmat (point Jacobi)
        B%nsing = B%nsing + 1
        if (B%band(bb)) then
          B%lu(B%loff(bb) + 1 : B%loff(bb) + int(2 * B%kl(bb) + B%ku(bb) + 1, 8) * n) = 0.0d0
        else
          B%lu(B%loff(bb) + 1 : B%loff(bb) + int(n, 8)**2) = 0.0d0
        endif
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

    ! device path: the factors' copy (smoother 9's two block sets stay on the host)
    if (dev .and. dev_blocks .and. sm_type /= 9 .and. count(.not. B%axblk) > 0) then
      if (devpat .or. .not. c_associated(B%dh)) call blk_dev_pattern(B%dh, B%nrow, B%ngh, B%off, B%sz, B%kl, B%ku, &
        B%band, B%loff, B%axblk, B%rows, B%bcol, B%zp, B%zc, B%dstat)
      call blk_dev_values(B%dh, B%lu, B%piv, B%zv)
    endif

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

    if (B%axsparse .and. axis_mult > 0) then
      if (B%gsready) then
        call MatDestroy(B%Bra, ierr)
        call MatDestroy(B%Bar, ierr)
      else
        call ISComplement(B%axis_is, rst, ren, B%rest_is, ierr)
      endif
      call MatCreateSubMatrix(A, B%rest_is, B%axis_is, MAT_INITIAL_MATRIX, B%Bra, ierr)
      call MatCreateSubMatrix(A, B%axis_is, B%rest_is, MAT_INITIAL_MATRIX, B%Bar, ierr)
      if (.not. B%gsvec) then
        call MatCreateVecs(A, B%xw, PETSC_NULL_VEC, ierr)
        call VecDuplicate(B%xw, B%xw2, ierr)
        call MatCreateVecs(B%Bra, PETSC_NULL_VEC, B%tr, ierr)
        call MatCreateVecs(B%Bar, PETSC_NULL_VEC, B%ta, ierr)
        B%gsvec = .true.
      endif
      B%gsready = .true.
    endif
  end subroutine build_blocks

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
    integer :: pass, R, nall, bb, n, kz, nva, nvg
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
      ! storage: banded where the band is narrow, dense otherwise
      tot = 0
      do bb = 1, B%nb
        n = B%sz(bb)
        B%band(bb) = (2 * (2 * B%kl(bb) + B%ku(bb) + 1) < n)
        B%loff(bb) = tot
        if (B%axblk(bb)) then
          B%band(bb) = .false.                   ! sparse axis solve: no dense storage
        else if (B%band(bb)) then
          tot = tot + int(2 * B%kl(bb) + B%ku(bb) + 1, 8) * n
        else
          tot = tot + int(n, 8)**2
        endif
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
      if (sm_type >= 7 .and. B%bcol(bb) == 1 .and. B%bcol(cb) == 0) then
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
    B%lu = 0.0d0
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
    integer :: gtot(2), gloc(2), dmax, ninc
    integer, allocatable :: Ia(:), Ib(:), blkof(:), gb(:), gI(:), nsz(:), noff(:), nrows(:), fill(:)
    integer, allocatable :: tb(:), tI(:), tJ(:), gJ(:), lst(:), rkey(:)
    PetscInt, allocatable :: cand(:), gl(:), tl(:), perm(:)

    nr = B%nrow
    if (g == 0) then
      nc = nth0
    else
      nc = glv(g)%nj
    endif
    ilim = 0
    if (axis_k /= 0) ilim = axis_lim(g)
    ! the local segments, and the block of every (line or ring, slot): a line
    ! is keyed by J, a ring by I; every slot of the block is registered (with
    ! harm_pair a block holds a harmonic's cos and sin slots)
    nkey = nc
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
      kp(r) = dble((I * nc + J) * pc_ntor + m)
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
        m = mod(code, pc_ntor); J = mod(code / pc_ntor, nc); I = code / pc_ntor / nc
        if (I <= ilim) cycle
        if (rings) then
          ! same ring and slot as a local ring block; the walk bounds the reach
          if (I >= nkey) cycle
          bb = blkof(I * pc_ntor + m)
          if (bb == 0) cycle
        else
          bb = blkof(J * pc_ntor + m)
          if (bb == 0) cycle
          if (.not. ((I < Ia(bb) .and. Ia(bb) - I <= lovl) .or. (I > Ib(bb) .and. I - Ib(bb) <= lovl))) cycle
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
      allocate(rkey(nr + ngh))
      do r = 1, nr + ngh
        if (r <= nr) then
          call node_of(r, I, J, m)
        else
          J = gJ(r - nr)
        endif
        if (J < nc - J) then
          rkey(r) = 2 * J
        else
          rkey(r) = 2 * (nc - 1 - J) + 1
        endif
      enddo
      do bb = 1, B%nb
        if (nsz(bb) == B%sz(bb)) cycle
        call sort_rows_by_key(nrows(noff(bb) + 1 : noff(bb) + nsz(bb)), rkey)
      enddo
      ! every ring block must now span its ring (a cut ring leaves the
      ! poloidal Alfven coupling across the cut outside the block)
      ninc = 0
      block
        logical :: seen(0:nc - 1)
        do bb = 1, B%nb
          if (B%axblk(bb) .or. Ia(bb) < 0) cycle
          seen = .false.
          do q = 1, nsz(bb)
            r = nrows(noff(bb) + q) + 1
            if (r <= nr) then
              call node_of(r, I, J, m)
            else
              J = gJ(r - nr)
            endif
            seen(J) = .true.
          enddo
          if (.not. all(seen)) ninc = ninc + 1
        enddo
      end block
      call MPI_Allreduce(MPI_IN_PLACE, ninc, 1, MPI_INTEGER, MPI_SUM, gcomm, mpierr)
      if (gme == 0 .and. ninc > 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG WARNING: ", ninc, &
        " ring block(s) on level ", g, " still cut by a rank boundary"
      deallocate(rkey)
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
    call ISCreateGeneral(PETSC_COMM_SELF, int(ngh, kind(nn)), gl, PETSC_COPY_VALUES, B%isg, ierr)
    call VecCreate(PETSC_COMM_SELF, B%xg, ierr)
    call VecSetSizes(B%xg, int(ngh, kind(nn)), int(ngh, kind(nn)), ierr)
    if (dev) then                                 ! the ghost values stay on the device
      block
        character(len=64) :: vt
        call MatGetVecType(gAd(g), vt, ierr)
        call VecSetType(B%xg, trim(vt), ierr)
      end block
    else
      call VecSetType(B%xg, VECSEQ, ierr)
    endif
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
      if (gBk2(g)%axpend) then
        do kk = 1, size(gBk2(g)%axd)
          call axd_factor(gBk2(g)%axd(kk))
        enddo
      endif
    enddo
    call PetscLogEventEnd(gev_axnum, ierr)
    ! a failed sector factorisation (zero pivot) on any rank: that level's
    ! block goes to the LU. Every rank walks the same levels (axpend agrees).
    do g = 0, nlev - 2
      if (gBk(g)%axpend) call check(gBk(g))
      if (gBk2(g)%axpend) call check(gBk2(g))
    enddo
    ! first build: the gates, and the LU where a sector solve failed its gate
    do g = 0, nlev - 2
      if (gBk(g)%axnew) call finish(gBk(g))
      if (gBk2(g)%axnew) call finish(gBk2(g))
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

  !> The LU of the whole axis block(s) of level g (rds_t): where no sector
  !! solve is in use. First build and every rebuild.
  subroutine ax_lu(g, A, B)
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    integer :: kk, gnp, mpierr
    character(len=24) :: axname
    character(len=64) :: axtag
    if (size(B%axg) == 1) then
      write(axname, '(A,I0)') "axblk", g
      write(axtag, '(A,I0)') "axis block level ", g
      call rds_setup(B%axg(1), A, trim(axname), trim(axtag))
    else
      call MPI_Comm_size(gcomm, gnp, mpierr)
      do kk = 1, size(B%axg)
        write(axname, '(A,I0,A,I0)') "axblk", g, "n", kk - 1
        write(axtag, '(A,I0,A,I0,A,I0)') "axis block level ", g, " |n|-group ", kk - 1, &
                                         " on rank ", mod(kk - 1, gnp)
        call rds_setup(B%axg(kk), A, trim(axname), trim(axtag), mod(kk - 1, gnp))
      enddo
    endif
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

  !> Storage index of entry (i, j) of block bb (dense column-major, or LAPACK
  !! band storage with the kl extra rows dgbtrf needs for fill).
  integer(8) function lu_index(B, bb, i, j)
    type(blk_t), intent(in) :: B
    integer, intent(in) :: bb, i, j
    if (B%band(bb)) then
      lu_index = B%loff(bb) + int(j - 1, 8) * (2 * B%kl(bb) + B%ku(bb) + 1) &
                 + (B%kl(bb) + B%ku(bb) + 1 + i - j)
    else
      lu_index = B%loff(bb) + int(j - 1, 8) * B%sz(bb) + i
    endif
  end function lu_index

  !> Block of the DOFs at grid point (I, J) of level g, toroidal slot m, for
  !! the active smoother. The axis ring (I = 0, one shared value DOF) is one
  !! block per slot for every kind; with axis_k > 0 (kinds 4-6) rings 0..axis_k
  !! are. 2/3 node, 4 flux-surface ring (all J at one I), 5 radial line (all
  !! I > axis_k at one J), 6 rings for I < ring_lim(g) and radial lines
  !! outside. Ids have gaps; build_blocks compresses them.
  integer function blk_id(g, I, J, m_, nj)
    use mod_petsc_pc_harm, only: pc_ntor, pc_grp
    integer, intent(in) :: g, I, J, m_, nj
    integer :: is_g, m
    m = m_
    if (hpair > 0) m = pc_grp(m_)          ! slots 0 | 1, 2 | 3, 4 | ...: n = 0, then cos/sin pairs
    if (I == 0 .or. (sm_type >= 4 .and. I <= axis_lim(g))) then
      blk_id = m + 1
    else if (sm_type == 4) then
      blk_id = pc_ntor + (I - 1) * pc_ntor + m + 1
    else if (sm_type == 5 .or. sm_type == 7) then
      blk_id = pc_ntor + J * pc_ntor + m + 1
    else if (sm_type == 6 .or. sm_type == 8) then
      is_g = ring_lim(g)
      if (I < is_g) then
        blk_id = pc_ntor + (I - 1) * pc_ntor + m + 1
      else
        blk_id = pc_ntor + max(is_g - 1, 0) * pc_ntor + J * pc_ntor + m + 1
      endif
    else
      blk_id = pc_ntor + ((I - 1) * nj + J) * pc_ntor + m + 1
    endif
  end function blk_id

  !> Switch ring of level g: coarse ring I sits at fine ring I*2^g, so the
  !! same physical radius as the fine ring_is (the ratio r*dtheta/dr does not
  !! change under standard coarsening).
  integer function ring_lim(g)
    integer, intent(in) :: g
    ring_lim = (ring_is + 2**g - 1) / 2**g
  end function ring_lim

  !> Last ring of the axis block on level g: axis_k (>= 0) on every level, or
  !! with axis_k < 0 the rings below the switch radius, 0..ring_lim(g)-1 (at
  !! least ring 1). Stage D13: the circle-dominated rings only help when they
  !! are solved TOGETHER with the pole; as separate ring blocks they hurt.
  integer function axis_lim(g)
    integer, intent(in) :: g
    if (axis_k >= 0) then
      axis_lim = axis_k
    else
      axis_lim = max(ring_lim(g) - 1, 1)
    endif
  end function axis_lim

  !> PCSHELL apply on level cur_lev: y = blockdiag(A)^-1 x. With a sparse
  !! axis block and axis_mult > 0 the axis block and the lines are coupled
  !! Gauss-Seidel style instead of Jacobi (stage D13: the residual collects
  !! on both sides of the axis-block boundary): 1 = axis block first, then
  !! the lines on x - A(rest,ax) y_ax; 2 = lines first, then the axis block on
  !! x - A(ax,rest) y_rest; 3 = axis, lines, axis (symmetric).
  !> Smoother 9's second stage: blk_apply on the radial-line blocks.
  subroutine blk_apply_set2(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    blk_set2 = .true.
    call blk_apply(pc, x, y, ierr)
    blk_set2 = .false.
  end subroutine blk_apply_set2

  subroutine blk_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    type(blk_t), pointer :: B

    B => gBk(cur_lev)
    if (blk_set2) B => gBk2(cur_lev)
    if (c_associated(B%dh) .and. .not. blk_host .and. &
        (sm_type == 7 .or. sm_type == 8 .or. .not. B%axsparse .or. axis_mult == 0)) then
      ! device factors: the axis block first (on the host), as below
      if (B%axsparse) call ax_solve(x, y)
      call PetscLogEventBegin(gev_lines(cur_inst), ierr)
      if (B%ovl) then
        call VecScatterBegin(B%sct, x, B%xg, INSERT_VALUES, SCATTER_FORWARD, ierr)
        call VecScatterEnd(B%sct, x, B%xg, INSERT_VALUES, SCATTER_FORWARD, ierr)
      endif
      call blk_dev_apply(B%dh, x, y, B%xg)
      call PetscLogEventEnd(gev_lines(cur_inst), ierr)
    else if (sm_type == 7 .or. sm_type == 8) then
      call zebra_solve(x, y)
    else if (.not. B%axsparse .or. axis_mult == 0) then
      ! axis first: it depends on x only, and the ranks enter here in step
      ! (GMRES's reductions just synchronised them), which a J-sector solve
      ! needs; after the lines it would wait for the slowest rank's lines
      if (B%axsparse) call ax_solve(x, y)
      call lines_solve(x, y)
    else if (axis_mult == 2) then
      call lines_solve(x, y)
      call update_rhs(x, y, B%xw, B%axis_is, B%rest_is, B%Bar, B%ta)
      call ax_solve(B%xw, y)
    else
      call ax_solve(x, y)
      call update_rhs(x, y, B%xw, B%rest_is, B%axis_is, B%Bra, B%tr)
      call lines_solve(B%xw, y)
      if (axis_mult == 3) then
        call update_rhs(x, y, B%xw2, B%axis_is, B%rest_is, B%Bar, B%ta)
        call ax_solve(B%xw2, y)
      endif
    endif
    ierr = 0

  contains

    !> y(rows outside the axis block) = dense/banded block solves of x. The
    !! blocks own disjoint rows, so they run on the rank's OpenMP threads.
    !! With the line overlap the ghost rows' input comes from the neighbours
    !! (one scatter) and only the rank's own rows are written back.
    subroutine lines_solve(xx, yy)
      Vec :: xx, yy
      PetscScalar, pointer :: xp(:), yp(:), gp(:)
      PetscErrorCode :: ie
      integer :: bb, q, n, info, t, r, nr, ck
      external :: dgetrs
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
      !$omp parallel do schedule(dynamic, ck) private(bb, q, n, info, t, r) if (B%nrow >= LINES_OMP_MIN)
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
        if (B%band(bb)) then
          call band_solve(n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), &
                          blk_t_work(1, t))
        else
          call dgetrs('N', n, 1, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), blk_t_work(1, t), n, info)
        endif
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
      integer :: bb, q, n, info, t, c, k, r, nr, zc_
      external :: dgetrs
      nr = B%nrow
      ! the axis block belongs to colour 0 and depends on x only: solved
      ! first, while the ranks are in step (see blk_apply)
      if (B%axsparse) call ax_solve(xx, yy)
      call ghosts_in(xx, gp)
      do c = 0, 1
        call PetscLogEventBegin(gev_lines(cur_inst), ie)
        call VecGetArrayRead(xx, xp, ie)
        call VecGetArray(yy, yp, ie)
        !$omp parallel do schedule(dynamic, 4) private(bb, q, n, info, t, k, r, zc_) if (B%nrow >= LINES_OMP_MIN)
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
            if (c == 1) then
              do k = B%zp(r + 1), B%zp(r + 2) - 1
                zc_ = B%zc(k)
                if (zc_ < nr) then
                  blk_t_work(q, t) = blk_t_work(q, t) - B%zv(k) * yp(zc_ + 1)
                else
                  blk_t_work(q, t) = blk_t_work(q, t) - B%zv(k) * B%yg(zc_ - nr + 1)
                endif
              enddo
            endif
          enddo
          if (B%band(bb)) then
            call band_solve(n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), &
                            blk_t_work(1, t))
          else
            call dgetrs('N', n, 1, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), blk_t_work(1, t), n, info)
          endif
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
      if (.not. any(B%axg(:)%member)) return
      call PetscLogEventBegin(gev_axsolve(cur_inst), ie)
      call VecGetArrayRead(xx, xp, ie)
      call VecGetArray(yy, yp, ie)
      call rds_solve_groups(B%axg, xp, yp)
      call VecRestoreArray(yy, yp, ie)
      call VecRestoreArrayRead(xx, xp, ie)
      call PetscLogEventEnd(gev_axsolve(cur_inst), ie)
    end subroutine ax_solve

    !> w = x, then w(to rows) -= C y(from rows), C = A(to, from)
    subroutine update_rhs(xx, yy, w, to_is, from_is, C, t)
      Vec :: xx, yy, w, t
      IS  :: to_is, from_is
      Mat :: C
      Vec :: ys, ws
      PetscErrorCode :: ie
      call VecCopy(xx, w, ie)
      call VecGetSubVector(yy, from_is, ys, ie)
      call MatMult(C, ys, t, ie)
      call VecRestoreSubVector(yy, from_is, ys, ie)
      call VecGetSubVector(w, to_is, ws, ie)
      call VecAXPY(ws, -1.0d0, t, ie)
      call VecRestoreSubVector(w, to_is, ws, ie)
    end subroutine update_rhs
  end subroutine blk_apply

  !> physics_pc_gmg_ring_diag: WHERE the residual of a real right-hand side
  !! lives while stationary V-cycles reduce it. Runs NCYC cycles x <- x +
  !! V(b - F x) from x = 0 on a copy and prints ||r||^2 by ring zone -- the
  !! axis ring, rings 1..I_s-1 (r*dtheta/dr below the switch), I_s..2 I_s-1,
  !! the rest -- next to each zone's share of the rows, plus the three rings
  !! holding most of ||r||^2. The caller's solve is untouched.
  subroutine ring_diag(b)
    use phys_module, only: n_flux
    Vec :: b
    integer, parameter :: NCYC = 6
    Vec :: x, r, e
    PetscScalar, pointer :: rp(:)
    PetscErrorCode :: ierr
    PetscInt :: nl
    integer :: c, l, q, t, mpierr, top(3)
    real*8 :: zs(4), zr(4), rn, bn
    real*8, allocatable :: pr(:)

    diag_left = diag_left - 1
    call VecDuplicate(b, x, ierr)
    call VecDuplicate(b, r, ierr)
    call VecDuplicate(b, e, ierr)
    call VecZeroEntries(x, ierr)
    call VecCopy(b, r, ierr)
    call VecGetLocalSize(b, nl, ierr)
    allocate(pr(0:n_flux - 1))
    zr = 0.0d0
    do l = 1, int(nl)
      q = zone(fine_node(l) / nth0)
      zr(q) = zr(q) + 1.0d0
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, zr, 4, MPI_DOUBLE_PRECISION, MPI_SUM, gcomm, mpierr)
    if (gme == 0) write(*,'(A,I0,A,I0,A,I0,A,4F6.1)') "[GMG ring-diag] inst ", cur_inst, &
      ": zones axis | 1..", ring_is - 1, " | ", ring_is, "..2Is-1 | rest, rows %", 100.0d0 * zr / sum(zr)
    bn = 0.0d0
    do c = 0, NCYC
      if (c > 0) then
        call vcycle(0, r, e)
        call VecAXPY(x, 1.0d0, e, ierr)
        call MatMult(gF, x, r, ierr)
        call VecAYPX(r, -1.0d0, b, ierr)
      endif
      pr = 0.0d0
      call VecGetArrayRead(r, rp, ierr)
      do l = 1, int(nl)
        t = fine_node(l) / nth0
        pr(t) = pr(t) + rp(l)**2
      enddo
      call VecRestoreArrayRead(r, rp, ierr)
      call MPI_Allreduce(MPI_IN_PLACE, pr, n_flux, MPI_DOUBLE_PRECISION, MPI_SUM, gcomm, mpierr)
      zs = 0.0d0
      do t = 0, n_flux - 1
        zs(zone(t)) = zs(zone(t)) + pr(t)
      enddo
      rn = sqrt(sum(pr))
      if (c == 0) bn = max(rn, 1.d-300)
      do q = 1, 3
        top(q) = maxloc(pr, 1) - 1
        pr(top(q)) = -pr(top(q)) - 1.d-300           ! mark taken (restored below)
      enddo
      pr = abs(pr)
      if (gme == 0) write(*,'(A,I0,A,ES9.2,A,4F6.1,A,3(1X,I0,A,F4.1,A))') &
        "[GMG ring-diag]   cyc ", c, " |r|/|b| ", rn / bn, "  share %", 100.0d0 * zs / max(sum(zs), 1.d-300), &
        "  hot rings", (top(q), "(", 100.0d0 * pr(top(q)) / max(sum(pr), 1.d-300), "%)", q = 1, 3)
    enddo
    deallocate(pr)
    call VecDestroy(x, ierr)
    call VecDestroy(r, ierr)
    call VecDestroy(e, ierr)

  contains

    integer function zone(ring)
      integer, intent(in) :: ring
      if (ring == 0) then
        zone = 1
      else if (ring < ring_is) then
        zone = 2
      else if (ring < 2 * ring_is) then
        zone = 3
      else
        zone = 4
      endif
    end function zone
  end subroutine ring_diag

  !> physics_pc_gmg_ring_diag: which DOFs of the two outermost rings are pure
  !! Dirichlet rows (only a diagonal entry) in the level-0 Pmat, by field and
  !! canonical C1 DOF (u, a = radial, b = angular, c = cross derivative).
  subroutine report_bnd_rows(A)
    use phys_module, only: n_flux
    Mat :: A
    PetscInt :: nr, rst, ren, r, ncols
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    integer :: cnt(0:7, 2, 2), q, ring, kf, mpierr, c
    logical :: diag_only
    character(len=1), parameter :: kn(0:3) = ['u', 'a', 'b', 'c']

    call MatGetLocalSize(A, nr, PETSC_NULL_INTEGER, ierr)
    call MatGetOwnershipRange(A, rst, ren, ierr)
    cnt = 0
    do r = 0, nr - 1
      ring = fine_node(r + 1) / nth0
      if (ring < n_flux - 2) cycle
      q = ring - (n_flux - 2) + 1                 ! 1 = ring n_flux-2, 2 = boundary ring
      kf = fine_kf(r + 1)
      call MatGetRow(A, rst + r, ncols, cols, vals, ierr)
      diag_only = .true.
      do c = 1, int(ncols)
        if (cols(c) /= rst + r .and. vals(c) /= 0.0d0) diag_only = .false.
      enddo
      call MatRestoreRow(A, rst + r, ncols, cols, vals, ierr)
      cnt(kf, q, 1) = cnt(kf, q, 1) + 1
      if (diag_only) cnt(kf, q, 2) = cnt(kf, q, 2) + 1
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, cnt, size(cnt), MPI_INTEGER, MPI_SUM, gcomm, mpierr)
    if (gme == 0) then
      do q = 1, 2
        write(*,'(A,I0,A)', advance="no") "[GMG ring-diag] Dirichlet rows on ring ", n_flux - 3 + q, " (field.dof diag-only/total):"
        do kf = 0, 4 * nf_s - 1
          write(*,'(1X,I0,A,A,A,I0,A,I0)', advance="no") kf / 4 + 1, ".", kn(mod(kf, 4)), " ", cnt(kf, q, 2), "/", cnt(kf, q, 1)
        enddo
        write(*,*)
      enddo
    endif
  end subroutine report_bnd_rows

  !> physics_pc_gmg_bnd_drop: every boundary-ring u and b row of the level-0
  !! Pmat must be a pure Dirichlet row, or the dropped coarse DOFs remove a
  !! part of the space the operator acts on. Warns (all ranks' count) if not.
  subroutine check_bnd_rows(A)
    use phys_module, only: n_flux
    Mat :: A
    PetscInt :: nr, rst, ren, r, ncols
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    integer :: nbad, k, c, mpierr

    call MatGetLocalSize(A, nr, PETSC_NULL_INTEGER, ierr)
    call MatGetOwnershipRange(A, rst, ren, ierr)
    nbad = 0
    do r = 0, nr - 1
      if (fine_node(r + 1) / nth0 /= n_flux - 1) cycle
      k = mod(fine_kf(r + 1), 4)
      if (k /= 0 .and. k /= 2) cycle
      call MatGetRow(A, rst + r, ncols, cols, vals, ierr)
      do c = 1, int(ncols)
        if (cols(c) /= rst + r .and. vals(c) /= 0.0d0) then
          nbad = nbad + 1; exit
        endif
      enddo
      call MatRestoreRow(A, rst + r, ncols, cols, vals, ierr)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, gcomm, mpierr)
    if (gme == 0 .and. nbad > 0) write(*,'(A,I0,A)') "[Physics PC]   GMG WARNING: bnd_drop on, but ", nbad, &
      " boundary u/b rows are not pure Dirichlet rows; the coarse space misses them"
  end subroutine check_bnd_rows

  subroutine axis_patch(b, x)
    Vec :: b, x
    Vec :: rs, xs
    PetscErrorCode :: ierr
    call PetscLogEventBegin(gev_axis, ierr)
    call MatMult(gF, x, gr(0), ierr)
    call VecAYPX(gr(0), -1.0d0, b, ierr)
    call VecGetSubVector(gr(0), gisAxis, rs, ierr)
    call KSPSolve(gAxis, rs, gzax, ierr)
    call VecRestoreSubVector(gr(0), gisAxis, rs, ierr)
    call VecGetSubVector(x, gisAxis, xs, ierr)
    call VecAXPY(xs, 1.0d0, gzax, ierr)
    call VecRestoreSubVector(x, gisAxis, xs, ierr)
    call PetscLogEventEnd(gev_axis, ierr)
  end subroutine axis_patch

#endif
end module mod_petsc_pc_gmg
