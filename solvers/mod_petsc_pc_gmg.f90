!> Geometric multigrid for the diagonal blocks of the SF preconditioner
!! (mod_petsc_pc_sf_solver), several independent hierarchies (gmg_select). The
!! coarse spaces are nested C1 spaces of JOREK's logically structured grids: on
!! the polar grid the bicubic Hermite functions on 2^L x 2^L parametric macro-
!! elements (P = exact subdivision, the 1D cubic-Hermite midpoint rule), on
!! X-point grids de Casteljau subdivision of the elements' Bezier nets over core,
!! SOL and private region (nested approximately). No toroidal coarsening; the
!! coarse operators are Galerkin products. Smoothers: block relaxation on flux-
!! surface rings, radial lines, or zebra rings inside / zebra lines outside a
!! switch radius (GMGPolar, Bourne et al., JCP 488 (2023) 112249), each block
!! holding all fields and C1 DOFs of its nodes (Chacon, JCP 526 (2025)), an exact
!! solve of the axis block, restricted additive Schwarz across ranks (Cai &
!! Sarkis, SISC 21 (1999) 792).
module mod_petsc_pc_gmg
#ifdef USE_PETSC
  use mpi_mod
  !$ use omp_lib
#include "petsc/finclude/petsc.h"
  use petsc
  use iso_c_binding, only: c_ptr, c_double
  use mod_petsc_raw_csr, only: aij_parts, get_ij, put_ij, aij_vals_read, aij_vals_done, blockmv_attach, &
                               galerkin_blk
  use mod_petsc_pc_gmg_axis, only: axd_t, axd_nsec, axd_setup, axd_refill, axd_factor, axd_solve, axd_destroy
  use mod_gmg_ring_btd
  implicit none
  private

  public :: gmg_build_prolongations, gmg_setup_operator, gmg_is_ready, gmg_select
  public :: gmg_pc_apply_1, gmg_pc_apply_2, gmg_pc_apply_3, gmg_pc_apply_4, gmg_pc_apply_5
  public :: gmg_opts_t

  integer, parameter, public :: GMG_SM_RINGS       = 4  !< flux-surface ring blocks
  integer, parameter, public :: GMG_SM_LINES       = 5  !< radial-line blocks
  integer, parameter, public :: GMG_SM_ZEBRA_RINGS = 8  !< zebra rings inside the switch radius, zebra lines outside

  integer, parameter :: SM_STEPS = 4                    !< default smoothing steps

  !> A hierarchy's configuration, set by the caller.
  type :: gmg_opts_t
    integer :: smoother   = 0         !< GMG_SM_*
    integer :: pre0       = SM_STEPS  !< level-0 pre-smoothing steps (0 = none)
    integer :: post0      = SM_STEPS  !< level-0 post-smoothing steps (0 = none)
    integer :: nsmooth_c  = SM_STEPS  !< pre- and post-smoothing steps on every coarser level
    integer :: axis_rings = 0         !< axis block = rings 0..axis_rings; -2 = automatic (0 or 1)
    integer :: harm_pair  = 1         !< smoother block slots: 1 = a harmonic's cos and sin, 2 = all
    real*8  :: rich_omega = 1.d0      !< Richardson scale of the coarse-level smoothing
    integer :: semi_r     = 0         !< levels 1..semi_r coarsen radially only; -1 = by the cell aspect
  end type gmg_opts_t

  integer, parameter :: MAX_LEV         = 8
  integer, parameter :: MAX_LEV_FULL    = 6         !< level cap without radial semi-coarsening
  integer, parameter :: MAX_ENT         = 16        !< max nonzeros per scalar row of P (2x2 sources x 4 DOFs)
  integer, parameter :: CRS_MUMPS_MIN   = 2000      !< coarsest rows from which MUMPS factors its LU
  integer, parameter :: CRS_DIST_ROWS   = 20000     !< coarsest rows from which its LU is distributed
  real*8,  parameter :: SEMI_ASPECT     = 2.d0      !< semi_r -1: radial levels while median r dtheta/dr > this
  real*8,  parameter :: RING_ASPECT     = 1.d0      !< switch ring: the first with median r dtheta/dr >= this
  integer, parameter :: LINE_OVERLAP    = 1         !< radial lines extended by this many nodes across ranks
  integer, parameter :: RICH_FROM       = 1         !< levels >= this smooth with Richardson (no reductions)
  integer, parameter :: LINES_OMP_MIN   = 4000      !< rows below which the block solves stay serial
  integer, parameter :: AXIS_RING1_ROWS = 4096      !< axis_rings -2: ring 1 joins above this many ring-0 rows
  integer, parameter :: RING_GATHER_MAX = 16000000  !< level rows up to which the ring ghosts come from one gather

  !> One coarse level's DOFs and the block maps of its rows.
  type :: lvl_t
    integer :: ni = 0, nj = 0, n = 0
    integer, allocatable :: d(:,:,:)       !< (0:ni-1, 0:nj-1, 0:3) -> canonical DOF, -1 = dropped
    integer, allocatable :: rnode(:)       !< packed local row -> I*nj + J
    integer, allocatable :: rharm(:)       !< packed local row -> toroidal slot
    integer, allocatable :: rcol(:)        !< X-point grids: packed local row -> radial column
  end type lvl_t

  !> X-point grids: the smoothers' virtual lattice of one level. Ring I is one
  !! flux line of nn(I) positions J, closed (per) or open.
  type :: xring_t
    integer, allocatable :: nn(:)          !< (0:nring-1) positions along ring I
    logical, allocatable :: per(:)         !< ring I closes on itself
  end type xring_t

  !> Parallel layout of one level while the prolongations are built: scalar
  !! DOF d (0-based) -> owner rank and position among its DOFs. The packed
  !! global row of (field f, d, slot m) is ps(own) + (f*nl(own) + lpos(d))*n_tor + m.
  type :: lay_t
    integer :: n = 0
    integer, allocatable :: own(:), lpos(:), nl(:), ps(:)   !< own/lpos(0:n-1), nl(0:np-1), ps(0:np)
  end type lay_t

  !> A direct solve of A(rows, rows), redundant on the ranks owning some of the
  !! rows (sub-communicator comm), each with the whole submatrix and its LU: a
  !! solve needs no collective on the hierarchy's communicator, only an
  !! Allgatherv among the members. The coarsest level above CRS_DIST_ROWS rows
  !! instead takes one MUMPS LU over the hierarchy's ranks (dist, dksp).
  type :: rds_t
    logical :: ready = .false., member = .false.
    integer :: comm = MPI_COMM_NULL, np = 0, me = 0
    integer, allocatable :: loc(:), cnt(:), dsp(:)   !< the rank's local rows (0-based, ascending); members' counts, offsets
    real*8, allocatable  :: send(:)
    IS  :: isq
    Mat, pointer :: sub(:) => null()
    KSP :: ksp
    Vec :: b, x
    logical :: dist = .false.
    KSP :: dksp
  end type rds_t

  !> The block smoother of one level: blocks of the rank's rows and of ghost
  !! rows that neighbouring ranks own, each the axis block (rings 0..axis_k,
  !! one block per slot group, solved together), a ring or a radial line. Rows
  !! are numbered in a combined space: 0..nrow-1 the rank's own, nrow.. its
  !! ghosts (ascending global index); only own rows are written back (RAS).
  type :: blk_t
    integer :: nb = 0, nrow = 0, nsing = 0                 !< blocks, own rows, singular blocks
    integer, allocatable :: bid(:)                         !< combined row -> block (1-based)
    integer, allocatable :: off(:), sz(:)                  !< block -> offset of its rows, their count
    integer, allocatable :: rows(:)                        !< block rows (0-based combined), in block order
    integer, allocatable :: pos(:)                         !< combined row -> position in its block
    integer, allocatable :: piv(:)                         !< pivots, block order
    integer, allocatable :: kl(:), ku(:)                   !< band widths
    logical, allocatable :: band(:)                        !< LAPACK band storage, else dense
    integer(8), allocatable :: loff(:), loff4(:)           !< factor offset in lu; < 0: ring solver, in lu4 at loff4
    real*8, allocatable  :: lu(:)                          !< LAPACK factors
    real(rk_ring), allocatable :: lu4(:)                   !< ring-solver factors (kind rk_ring: single unless -DSF_RING_DP)
    integer(8) :: nlu = 0                                  !< size(lu): lu_index codes above address lu4
    real*8, allocatable  :: eqs(:)                         !< ring blocks' equilibration (btd_equil), else 1
    logical, allocatable :: axblk(:)                       !< the axis block's blocks
    type(rds_t), allocatable :: axr                        !< the axis block's LU
    type(axd_t), allocatable :: axd                        !< its J-sector solve (allocated: tried)
    logical :: axdon = .false.                             !< axd in use
    logical :: axpend = .false.                            !< axd refilled, factorisation pending
    logical :: axnew = .false.                             !< first build: axd factorisation and gate pending
    Mat :: axA                                             !< the operator of the pending factorisation
    integer, allocatable :: bcol(:)                        !< zebra colour of each block
    integer, allocatable :: zp(:), zc(:)                   !< zebra: A(colour 1, colour 0) as CSR, combined columns
    real*8, allocatable  :: zv(:)
    logical :: ovl = .false.                               !< ghost rows set up (ovl_extend)
    integer :: ngh = 0
    PetscInt, allocatable :: gidx(:)                       !< ghost -> global row, ascending
    integer, allocatable :: ghI(:), ghJ(:)                 !< ghost -> I, position J along its ring
    IS  :: isg, isall
    Vec :: xg                                              !< the input's ghost values (scatter sct)
    VecScatter :: sct
    real*8, allocatable :: yg(:)                           !< zebra: colour-0 solution on the ghosts
    Mat, pointer :: sg(:) => null()                        !< A(ghost rows, all columns)
    integer, allocatable    :: vs(:), gs(:)                !< value maps, source: +k / -k A's diag / off-diag part, k sg(1)
    integer(8), allocatable :: vd(:), gd(:)                !< value maps, destination: > 0 lu_index code, < 0 zv
    integer(8) :: nvd = 0, nvo = 0, nvg = 0                !< source value-array lengths
    integer(8) :: pat_id = -1, pat_nz = -1                 !< the operator the maps were built for
    logical, allocatable :: btd(:), jac(:)                 !< ring-solver block; its point-Jacobi fallback
    integer, allocatable :: qn(:)                          !< ring-solver block position -> node
    type(btd_ring), allocatable :: rg(:)
    integer, allocatable :: lb(:), lk(:)                   !< threaded hybrid: (block, segment) work items
    real*8, allocatable :: wk(:)                           !< threaded hybrid: work vector, block order
    real(rk_ring), allocatable :: wk4(:)                   !< its rk_ring copy
  end type blk_t

  !> One hierarchy (an instance of gmg_select). Level 0 is the fine level;
  !! P(g) maps level g to level g-1 (g = 1..nlev-1).
  type :: gmg_hier_t
    integer :: comm = MPI_COMM_NULL                 !< communicator the hierarchy was built on
    integer :: my_id = 0                            !< the builder's my_id (rank-0 prints, split keys)
    logical :: p_ready = .false.                    !< levels and prolongations built
    logical :: op_ready = .false.                   !< operators, smoothers and coarse solve set up once
    logical :: vec_ready = .false.                  !< level vectors x, b, r created
    integer :: nlev = 0                             !< levels
    integer :: nfields = 0                          !< identically indexed fields packed per rank
    integer :: nth0 = 0                             !< level-0 nj
    logical :: xgrid = .false.                      !< an X-point grid's hierarchy (virtual lattice)
    integer :: smoother = 0                         !< GMG_SM_*
    integer :: pre0 = SM_STEPS                      !< level-0 pre-smoothing steps
    integer :: post0 = SM_STEPS                     !< level-0 post-smoothing steps
    integer :: nsmooth_c = SM_STEPS                 !< steps per side on the coarser levels
    integer :: harm_pair = 1                        !< gmg_opts_t%harm_pair
    integer :: axis_k = 0                           !< the axis block: rings 0..axis_k on every level
    integer :: ring_is = 1                          !< smoother 8: fine rings I < ring_is take ring blocks
    integer :: sm_lev = 0                           !< the level smooth() works on (blk_apply's blocks)
    Mat :: P(1:MAX_LEV-1)                           !< prolongations, I_fields x P_s x I_slots
    Mat :: Ps(1:MAX_LEV-1)                          !< I_fields x P_s (Galerkin product on slot blocks)
    Mat :: A(0:MAX_LEV-1)                           !< operators: A(0) borrowed (or xp_A), A(g>0) owned
    logical :: ptap_blk(1:MAX_LEV-1) = .false.      !< A(g) made by galerkin_blk
    KSP :: sm_ksp(0:MAX_LEV-1)                      !< smoothers (shell PC blk_apply)
    Vec :: x(0:MAX_LEV-1)                           !< level solutions
    Vec :: b(0:MAX_LEV-1)                           !< level right-hand sides
    Vec :: r(0:MAX_LEV-1)                           !< level residuals
    type(blk_t) :: blk(0:MAX_LEV-1)                 !< the smoothers' blocks
    type(rds_t) :: coarse                           !< the coarsest level's direct solve
    type(lvl_t), allocatable :: lvl(:)              !< coarse DOF numberings and block maps (1:nlev-1)
    integer, allocatable :: fine_node(:)            !< level-0 row -> I*nth0 + J
    integer, allocatable :: fine_harm(:)            !< level-0 row -> toroidal slot
    integer, allocatable :: fine_col(:)             !< X-point grids: level-0 row -> radial column
    type(xring_t) :: xring(0:MAX_LEV-1)             !< X-point grids: each level's virtual rings
    ! X-point grids on several ranks: the hierarchy's own ring-major layout
    ! (JOREK numbers the divertor legs column by column)
    logical :: xperm = .false.                      !< the layout is in use
    IS :: xp_is                                     !< the old global row of every new local row
    Mat :: xp_A                                     !< A(xp_is, xp_is)
    Vec :: xp_b, xp_x                               !< the V-cycle's input and output in that layout
    VecScatter :: xp_sc                             !< the caller's vectors -> xp_b
    ! the last fine operand (object id, nonzero state, pattern hash): the same
    ! pattern lets the Galerkin chain and the LUs keep their symbolic phase
    integer(8) :: a0_sig(2) = 0, a0_id = 0, a0_nzst = -1
    logical :: pat_fresh = .false.                  !< this setup's fine pattern differs from the last one's
  end type gmg_hier_t

  integer, parameter :: MAX_INST = 5
  type(gmg_hier_t), target, save :: hier(MAX_INST)
  type(gmg_hier_t), pointer, save :: H => null()    !< the active hierarchy, hier(cur_inst)
  integer, save :: cur_inst = 1
  ! block-solve work arrays (max block size, 0:threads-1), shared by all
  ! levels and hierarchies (their solves never overlap), grow-only
  real*8, allocatable, save :: wbuf(:,:)
  real(rk_ring), allocatable, save :: wbuf4(:,:)

  ! -log_view events; the apply-side ones per instance (GMG_*, GMG<k>_*)
  PetscLogEvent, save :: gev_ptap = -1, gev_smsetup = -1, gev_coarselu = -1
  PetscLogEvent, save :: gev_prolong = -1
  PetscLogEvent, save :: gev_blkmesh = -1, gev_blkpat = -1, gev_blknum = -1, gev_axnum = -1, gev_axtry = -1
  PetscLogEvent, save :: gev_ovl = -1
  PetscLogEvent, save :: gev_lines(MAX_INST) = -1, gev_axsolve(MAX_INST) = -1
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

  !> Make hierarchy k the active one (H); every routine below works on H.
  subroutine gmg_select(k)
    integer, intent(in) :: k
    cur_inst = k
    H => hier(k)
  end subroutine gmg_select

  !> Is this rank the first of the active hierarchy's communicator?
  logical function hier_root()
    integer :: r, mpierr
    call MPI_Comm_rank(H%comm, r, mpierr)
    hier_root = (r == 0)
  end function hier_root

  !> (Re)build R for A(rows, rows), rows = R%loc on each rank; collective on
  !! H%comm. what = "coarse" or "axis" (options prefix sf_gmg<k>_<what>_);
  !! tag = label of the first build's factor-size print.
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
    ! a new pattern renews the submatrix and the symbolic factorisation, not the members
    fresh = .not. R%ready .or. H%pat_fresh
    if (R%ready .and. H%pat_fresh) then
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
      call MPI_Comm_split(H%comm, color, H%my_id, R%comm, mpierr)
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
    ! collective (non-members pass an empty IS). MAT_REUSE_MATRIX does not
    ! advance the submatrix's state: without the assembly PCSetUp would keep
    ! the old factor
    if (fresh) then
      call MatCreateSubMatrices(A, one, [R%isq], [R%isq], MAT_INITIAL_MATRIX, R%sub, ierr)
    else
      call MatCreateSubMatrices(A, one, [R%isq], [R%isq], MAT_REUSE_MATRIX, R%sub, ierr)
      call MatAssemblyBegin(R%sub(1), MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(R%sub(1), MAT_FINAL_ASSEMBLY, ierr)
    endif
    if (R%member) then
      if (fresh) then
        ! a large coarsest level: MUMPS (SCOTCH ordering, half the fill of PETSc's ND LU)
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

    !> Sequential LU KSP of R%sub(1): PETSc's own LU (cheaper triangular solves
    !! than MUMPS' on these blocks), nested dissection, quotient minimum degree
    !! (less fill) on the axis block; options under sf_gmg<k>_<what>_ override.
    subroutine rds_make_ksp(stype)
      character(len=*), intent(in) :: stype   ! not MatSolverType (len=80): ifort rejects the short constants
      call KSPCreate(PETSC_COMM_SELF, R%ksp, ierr)
      call KSPSetOperators(R%ksp, R%sub(1), R%sub(1), ierr)
      call KSPSetType(R%ksp, KSPPREONLY, ierr)
      call KSPGetPC(R%ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, stype, ierr)
      if (stype == MATSOLVERPETSC) then
        if (what == "axis") then
          call PCFactorSetMatOrderingType(pc, MATORDERINGQMD, ierr)
        else
          call PCFactorSetMatOrderingType(pc, MATORDERINGND, ierr)
        endif
      endif
      write(pre, '(A,I0,A,A,A)') "sf_gmg", cur_inst, "_", what, "_"
      if (stype == MATSOLVERMUMPS) then
        ! SCOTCH ordering unless the user chose one
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
    if (H%xperm) then
      call VecScatterBegin(H%xp_sc, b, H%xp_b, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd(H%xp_sc, b, H%xp_b, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call vcycle(0, H%xp_b, H%xp_x)
      call VecScatterBegin(H%xp_sc, H%xp_x, x, INSERT_VALUES, SCATTER_REVERSE, ierr)
      call VecScatterEnd(H%xp_sc, H%xp_x, x, INSERT_VALUES, SCATTER_REVERSE, ierr)
    else
      call vcycle(0, b, x)
    endif
    call PetscLogEventEnd(gev_vcycle(cur_inst), ierr)
    ierr = 0
  end subroutine vcycle_of

  !> The active hierarchy's levels and prolongations are built.
  logical function gmg_is_ready()
    gmg_is_ready = H%p_ready
  end function gmg_is_ready

  !> Build the active hierarchy's levels and prolongations, once per run
  !! (collective on comm, which becomes H%comm: all its later collectives and
  !! objects use it). Aref: an operator with the packed layout the hierarchy
  !! will serve (its row ownership fixes every level's layout; on X-point grids
  !! its diagonal-only rows mark the Dirichlet DOFs). n_fields: identically
  !! indexed fields packed per rank. ok = .false. (on every rank) if the grid is
  !! not one this construction handles.
  subroutine gmg_build_prolongations(Aref, comm, my_id, n_fields, ok, opts)
    use nodes_elements, only: node_list, element_list
    use phys_module,    only: n_flux, n_tht, treat_axis
    use mod_parameters, only: n_degrees
    use mod_petsc_pc_harm, only: pc_ntor
    use mod_petsc_pc_xpatch, only: xpatch_is_xgrid

    Mat, intent(in)      :: Aref
    integer, intent(in)  :: comm, my_id, n_fields
    logical, intent(out) :: ok
    type(gmg_opts_t), intent(in) :: opts

    type(lvl_t), allocatable :: lv(:)
    integer, allocatable :: o(:,:)
    logical, allocatable :: oset(:,:)
    integer :: n_inact
    integer :: n_idx, e, iv, n, k, i, j, ni, nj, g, cand(4), nent, nrow_s, semi_r
    integer :: eps_s(4), eps_t(4), cols(MAX_ENT), vi(4), vj(4), idx
    real*8  :: w(MAX_ENT), sz(4)
    integer, allocatable :: rs(:), cs(:)
    real*8,  allocatable :: ws(:)
    logical, allocatable :: done(:)
    integer :: ns, nlev_cap
    real*8, allocatable :: asp(:,:), asp_sorted(:)
    type(lay_t), allocatable :: ly(:)
    integer :: me, np, mpierr, r, ci, cj, dd, own_, ff, mm, lr, nloc
    character(len=256) :: msg
    PetscErrorCode :: ierr

    ok = .false.
    H%comm = comm; H%my_id = my_id
    call gmg_register_events()
    call PetscLogEventBegin(gev_prolong, ierr)
    semi_r = max(opts%semi_r, -1)
    eps_s = [1, -1, -1, 1]
    eps_t = [1, 1, -1, -1]
    if (n_degrees /= 4) then
      call fail("needs the bicubic Hermite element (n_degrees = 4)")
      return
    endif
    if (treat_axis .and. hier_root()) write(*,'(A)') "[Physics PC]   GMG WARNING: treat_axis shares "// &
      "Cartesian axis DOFs, which the coarse spaces do not represent (slower V-cycles near the axis)"

    !--- gate 1: the (i-1)*n_tht + j polar layout; an X-point grid takes its
    !--- own builder
    H%xgrid = .false.
    if (node_list%n_nodes /= n_flux * n_tht) then
      if (xpatch_is_xgrid()) then
        call PetscLogEventEnd(gev_prolong, ierr)
        call gmg_build_xpoint_prolongations(Aref, comm, my_id, n_fields, ok, semi_r)
        return
      endif
      call fail("not a flux-surface grid (n_nodes /= n_flux*n_tht)")
      return
    endif
    if (n_flux < 3 .or. element_list%n_elements /= (n_flux - 1) * n_tht) then
      call fail("not a periodic flux-surface grid of (n_flux-1)*n_tht elements, n_flux >= 3")
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

    !--- gate 2: the element size factors are a per-node orientation o = +-1
    !--- (a derivative DOF is o * f_x/3), consistent across elements. On the
    !--- axis fix_axis_nodes zeroes the angular ones: no basis function uses
    !--- those DOFs, o = 0, and P leaves them out
    allocate(o(node_list%n_nodes, 0:3), oset(node_list%n_nodes, 0:3))
    o = 0; oset = .false.; n_inact = 0
    do e = 1, element_list%n_elements
      do iv = 1, 4
        n = element_list%element(e)%vertex(iv)
        sz = [element_list%element(e)%size(iv, 1), eps_s(iv) * element_list%element(e)%size(iv, 2), &
              eps_t(iv) * element_list%element(e)%size(iv, 3), &
              eps_s(iv) * eps_t(iv) * element_list%element(e)%size(iv, 4)]
        do k = 0, 3
          if (abs(abs(sz(k + 1)) - 1.d0) < 1.d-12) then
            cand(k + 1) = nint(sign(1.d0, sz(k + 1)))
          else if (sz(k + 1) == 0.d0 .and. k >= 2 .and. node_list%node(n)%axis_node) then
            cand(k + 1) = 0
          else
            call fail("element size factor is not +-1")
            return
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

    !--- coarse levels. On cells long in theta the couplings are strongest
    !--- radially, which ring blocks do not smooth and a full coarsening cannot
    !--- represent: levels 1..semi_r coarsen in I only
    if (semi_r < 0) then
      block
        real*8 :: a_
        a_ = asp_sorted((size(asp_sorted) + 1) / 2)
        semi_r = 0
        do while (a_ > SEMI_ASPECT .and. semi_r < MAX_LEV - 1)
          semi_r = semi_r + 1; a_ = a_ / 2
        enddo
        if (my_id == 0 .and. semi_r > 0) write(*,'(A,I0,A,I0,A,F6.2,A)') "[Physics PC]   GMG", cur_inst, &
          ": radial semi-coarsening on levels 1..", semi_r, " (median r dtheta/dr ", &
          asp_sorted((size(asp_sorted) + 1) / 2), ")"
      end block
    endif
    allocate(lv(1:MAX_LEV - 1))
    ni = n_flux; nj = n_tht; H%nlev = 1
    nlev_cap = MAX_LEV_FULL
    if (semi_r > 0) nlev_cap = MAX_LEV
    do g = 1, nlev_cap - 1
      if (mod(ni - 1, 2) /= 0) exit
      if (g > semi_r .and. (mod(nj, 2) /= 0 .or. nj < 4)) exit
      ni = (ni - 1) / 2 + 1
      if (g > semi_r) nj = nj / 2
      call make_level(lv(g), ni, nj)
      H%nlev = H%nlev + 1
    enddo
    if (H%nlev < 2) then
      call fail("mesh cannot be coarsened (need (n_flux-1) even and n_tht divisible by 4)")
      return
    endif

    !--- parallel layouts. Fine: read off Aref (fine_layout). Coarse: DOF
    !--- (I,J,k) goes to the owner of the fine node (I 2^g, J 2^g).
    call MPI_Comm_rank(comm, me, mpierr)
    call MPI_Comm_size(comm, np, mpierr)
    allocate(ly(0:H%nlev - 1))
    call fine_layout(Aref, comm, n_fields, n_idx, ly(0), msg)
    if (len_trim(msg) > 0) then
      call fail(trim(msg))
      return
    endif
    do g = 1, H%nlev - 1
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
        if (o(n, k) == 0) cycle              ! inactive (fixed) axis DOF: empty row
        call interp_row(i, j, k, lv(1), lv(1)%nj == n_tht, cols, w, nent)
        rs(ns + 1:ns + nent) = idx
        cs(ns + 1:ns + nent) = cols(1:nent)
        ws(ns + 1:ns + nent) = o(n, k) * w(1:nent)
        ns = ns + nent
      enddo
    enddo
    call expand_p(H%P(1), H%Ps(1), comm, ly, me, n_fields, 0, 1, rs(1:ns), cs(1:ns), ws(1:ns), MAX_ENT, .false.)
    deallocate(rs, cs, ws, done)

    !--- P(g), g >= 2: level g canonical -> level g-1 canonical
    do g = 2, H%nlev - 1
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
      call expand_p(H%P(g), H%Ps(g), comm, ly, me, n_fields, g - 1, g, rs(1:ns), cs(1:ns), ws(1:ns), &
                    MAX_ENT, .false.)
      deallocate(rs, cs, ws)
    enddo

    !--- packed LOCAL row -> (node, harmonic) maps for the block smoothers,
    !--- fine level (fine_node/fine_harm) and every coarse level
    H%nfields = n_fields
    H%nth0 = n_tht
    nloc = n_fields * pc_ntor * ly(0)%nl(me)
    if (allocated(H%fine_node)) deallocate(H%fine_node, H%fine_harm)
    allocate(H%fine_node(nloc), H%fine_harm(nloc))
    H%fine_node = -1
    do n = 1, node_list%n_nodes
      do k = 1, 4
        idx = node_list%node(n)%index(k) - 1
        if (ly(0)%own(idx) /= me) cycle
        do ff = 0, n_fields - 1
          do mm = 0, pc_ntor - 1
            lr = (ff * ly(0)%nl(me) + ly(0)%lpos(idx)) * pc_ntor + mm + 1
            H%fine_node(lr) = n - 1
            H%fine_harm(lr) = mm
          enddo
        enddo
      enddo
    enddo
    if (any_rank(any(H%fine_node < 0), comm)) then
      call fail("row -> node map does not cover every row")
      return
    endif
    do g = 1, H%nlev - 1
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
      if (any_rank(any(lv(g)%rnode < 0), comm)) then
        call fail("coarse row -> node map does not cover every row")
        return
      endif
    enddo
    if (my_id == 0 .and. np > 1) then
      write(*,'(A,I0,A)', advance="no") "[Physics PC]   GMG parallel layout on ", np, " ranks, rows/rank min-max per level:"
      do g = 0, H%nlev - 1
        write(*,'(A,I0,A,I0)', advance="no") " ", n_fields * pc_ntor * minval(ly(g)%nl), "-", &
                                             n_fields * pc_ntor * maxval(ly(g)%nl)
      enddo
      write(*,*)
    endif
    deallocate(ly)
    call move_alloc(lv, H%lvl)

    !--- the switch ring of smoother 8: the first ring whose median aspect
    !--- reaches RING_ASPECT (GMGPolar's criterion); ring i holds the cells
    !--- between flux surfaces i and i+1
    if (my_id == 0) write(*,'(A,F8.2,A,F8.2,A,F8.2)') &
      "[Physics PC]   GMG grid cells, rdtheta/dr: median ", asp_sorted((size(asp_sorted) + 1) / 2), &
      ", max ", asp_sorted(size(asp_sorted)), ", min ", asp_sorted(1)
    deallocate(asp_sorted)
    allocate(ws(n_flux - 2), asp_sorted(n_tht))
    do i = 1, n_flux - 2
      asp_sorted = asp(:, i)
      call sort_real(asp_sorted)
      ws(i) = asp_sorted((n_tht + 1) / 2)
    enddo
    H%ring_is = n_flux - 1
    do i = 1, n_flux - 2
      if (ws(i) >= RING_ASPECT) then
        H%ring_is = i; exit
      endif
    enddo
    if (my_id == 0) then
      write(*,'(A,F5.2,A,I0,A)', advance="no") "[Physics PC]   GMG ring medians rdtheta/dr (switch at ", &
        RING_ASPECT, ": rings I < ", H%ring_is, "):"
      do i = 1, min(n_flux - 2, H%ring_is + 2)
        write(*,'(A,F6.2)', advance="no") " ", ws(i)
      enddo
      write(*,*)
    endif
    deallocate(asp, asp_sorted, ws)

    if (my_id == 0) then
      write(*,'(A,I0,A)', advance="no") "[Physics PC]   GMG hierarchy: ", H%nlev, " levels, grid "
      write(*,'(I0,A,I0)', advance="no") n_flux, "x", n_tht
      do g = 1, H%nlev - 1
        write(*,'(A,I0,A,I0)', advance="no") " -> ", H%lvl(g)%ni, "x", H%lvl(g)%nj
      enddo
      write(*,*)
    endif
    H%p_ready = .true.
    ok = .true.
    call PetscLogEventEnd(gev_prolong, ierr)

  contains

    !> Aspect r dtheta/dr of cell (ring ii, angle jj) from its vertices; on a
    !! stellarator grid the median over the n_plane toroidal planes.
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

    subroutine fail(m_)
      character(len=*), intent(in) :: m_
      if (hier_root()) write(*,'(A,A)') "[Physics PC]   GMG hierarchy REFUSED: ", m_
      call PetscLogEventEnd(gev_prolong, ierr)
    end subroutine fail

  end subroutine gmg_build_prolongations

  !> gmg_build_prolongations on a lower single-null X-point grid: one hierarchy
  !! over the patches of mod_petsc_pc_xpatch. Every level coarsens every patch
  !! conformingly across the separatrices (cells paired from the X-point, cuts
  !! shared through the separatrix nodes; levels 1..semi_req across only); P is
  !! de Casteljau subdivision of each element's Bezier net, a fine DOF seen by
  !! several elements taking their average; coarse DOFs are fine node DOFs. The
  !! smoothers see a virtual lattice (xring, fine_col / rcol): ring I a flux line
  !! (core rings from the axis, private rows from the wall, SOL rows from the
  !! separatrix, last the X-point's 9 DOFs), J the position along it.
  subroutine gmg_build_xpoint_prolongations(Aref, comm, my_id, n_fields, ok, semi_req)
    use nodes_elements,      only: node_list, element_list
    use mod_petsc_pc_harm,   only: pc_ntor
    use mod_petsc_pc_xpatch, only: xpatch_t, xpatch_build, xpatch_report, XP_NPATCH, XP_CORE, &
                                   XP_SOL, XP_PRIV
    Mat, intent(in)      :: Aref
    integer, intent(in)  :: comm, my_id, n_fields
    logical, intent(out) :: ok
    integer, intent(in)  :: semi_req          !< gmg_opts_t%semi_r

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
    integer, allocatable :: rr(:), cc(:), cnt(:)
    real*8,  allocatable :: ww(:)
    logical, allocatable :: drop(:), dropc(:), xdof(:)
    integer :: me, np, mpierr, r, n_idx, g, c, k, kk, n, idx, ntrip, nloc, lr, ff, mm, q
    integer :: semi, nlev_cap
    logical :: more
    character(len=256) :: msg
    character(len=64) :: shape
    PetscErrorCode :: ierr

    ok = .false.
    H%comm = comm; H%my_id = my_id
    call MPI_Comm_rank(comm, me, mpierr)
    call MPI_Comm_size(comm, np, mpierr)
    call PetscLogEventBegin(gev_prolong, ierr)

    !--- the patches
    call xpatch_build(xp, ok, msg)
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

    !--- the fine layout, read off Aref
    allocate(ly(0:MAX_LEV - 1))
    call fine_layout(Aref, comm, n_fields, n_idx, ly(0), msg)
    if (len_trim(msg) > 0) then
      call fail(trim(msg)); return
    endif

    !--- Dirichlet DOFs (rows of field 1, slot 0 with only a diagonal): empty
    !--- rows of P, and out of every coarse space; the axis's angular slope
    !--- (index 3 of an axis node) too, as on the polar grid (make_level)
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

    !--- radial-only levels: one count for every patch (semi_req < 0: by the
    !--- median cell aspect over all patches but the core's axis row, down to
    !--- half the polar grid's SEMI_ASPECT)
    semi = semi_req
    if (semi < 0) then
      block
        real*8 :: a_
        a_ = median_aspect(-1)
        semi = 0
        do while (a_ > 0.5d0 * SEMI_ASPECT .and. semi < MAX_LEV - 1)
          semi = semi + 1; a_ = a_ / 2
        enddo
        if (my_id == 0 .and. semi > 0) write(*,'(A,I0,A,I0,A,F6.2,A)') "[Physics PC]   GMG", cur_inst, &
          ": radial semi-coarsening on levels 1..", semi, " (median r dtheta/dr ", median_aspect(-1), ")"
      end block
    endif
    nlev_cap = MAX_LEV_FULL
    if (semi > 0) nlev_cap = MAX_LEV

    !--- the levels
    call level_dofs(0)
    if (len_trim(msg) > 0) then
      call fail(trim(msg)); return
    endif
    !--- several ranks: the hierarchy's own layout, ring-major (see gmg_hier_t)
    H%xperm = .false.
    if (np > 1) then
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
        call ISCreateGeneral(comm, int(nloc, kind(ix)), ix, PETSC_COPY_VALUES, H%xp_is, ierr)
      end block
      H%xperm = .true.
    endif

    H%nlev = 1
    shape = ""
    do g = 1, nlev_cap - 1
      more = coarsen(g - 1, g <= semi)
      if (len_trim(msg) > 0) then
        call fail(trim(msg)); return
      endif
      if (.not. more) exit
      call level_dofs(g)
      if (len_trim(msg) > 0) then
        call fail(trim(msg)); return
      endif
      H%nlev = H%nlev + 1
      write(shape(len_trim(shape) + 1:), '(A,A)') " ", merge("r", "b", g <= semi)
    enddo
    if (H%nlev < 2) then
      call fail("the X-point grid cannot be coarsened"); return
    endif

    !--- coarse layouts: a coarse DOF is a fine node DOF and goes with it
    do g = 1, H%nlev - 1
      ly(g)%n = xl(g)%n
      allocate(ly(g)%own(0:xl(g)%n - 1), ly(g)%lpos(0:xl(g)%n - 1), ly(g)%nl(0:np - 1), ly(g)%ps(0:np))
      ly(g)%nl = 0
      do k = 1, xl(g)%n
        ly(g)%own(k - 1) = ly(0)%own(xl(g)%dof(k) - 1)
        ly(g)%lpos(k - 1) = ly(g)%nl(ly(g)%own(k - 1))
        ly(g)%nl(ly(g)%own(k - 1)) = ly(g)%nl(ly(g)%own(k - 1)) + 1
      enddo
      ly(g)%ps(0) = 0
      do r = 0, np - 1
        ly(g)%ps(r + 1) = ly(g)%ps(r) + n_fields * pc_ntor * ly(g)%nl(r)
      enddo
    enddo

    !--- the prolongations, expanded to fields and slots
    do g = 1, H%nlev - 1
      call make_p(g)
      call expand_p(H%P(g), H%Ps(g), comm, ly, me, n_fields, g - 1, g, rr(1:ntrip), cc(1:ntrip), ww(1:ntrip), &
                    2 * MAX_ENT, .true.)
    enddo

    !--- the block maps: ring, position, column of every local row
    H%nfields = n_fields
    H%nth0 = max(maxval(H%xring(0)%nn), 1)
    nloc = n_fields * pc_ntor * ly(0)%nl(me)
    if (allocated(H%fine_node)) deallocate(H%fine_node, H%fine_harm)
    if (allocated(H%fine_col)) deallocate(H%fine_col)
    allocate(H%fine_node(nloc), H%fine_harm(nloc), H%fine_col(nloc))
    H%fine_node = -1
    do k = 1, xl(0)%n
      idx = xl(0)%dof(k) - 1
      if (ly(0)%own(idx) /= me) cycle
      do ff = 0, n_fields - 1
        do mm = 0, pc_ntor - 1
          lr = (ff * ly(0)%nl(me) + ly(0)%lpos(idx)) * pc_ntor + mm + 1
          H%fine_node(lr) = xl(0)%rI(k) * H%nth0 + xl(0)%rJ(k)
          H%fine_harm(lr) = mm
          H%fine_col(lr) = xl(0)%rC(k)
        enddo
      enddo
    enddo
    if (any_rank(any(H%fine_node < 0), comm)) then
      call fail("row -> ring map does not cover every row (a DOF outside the patches)"); return
    endif
    allocate(lv(1:MAX_LEV - 1))
    do g = 1, H%nlev - 1
      lv(g)%ni = xl(g)%nring; lv(g)%nj = max(maxval(H%xring(g)%nn), 1); lv(g)%n = xl(g)%n
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
    if (allocated(H%lvl)) deallocate(H%lvl)
    call move_alloc(lv, H%lvl)

    !--- the switch ring of smoother 8: the first core ring whose median
    !--- aspect reaches RING_ASPECT
    H%ring_is = xl(0)%p(XP_CORE)%nb
    do k = 1, xl(0)%p(XP_CORE)%nb - 1
      if (median_aspect(k) >= RING_ASPECT) then
        H%ring_is = k; exit
      endif
    enddo

    if (my_id == 0) then
      write(*,'(A,I0,A,I0,A)', advance="no") "[Physics PC]   GMG", cur_inst, " X-point hierarchy: ", H%nlev, &
        " levels, elements core | SOL | private"
      do g = 0, H%nlev - 1
        write(*,'(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)', advance="no") "  ", xl(g)%p(1)%na, "x", xl(g)%p(1)%nb, &
          "|", xl(g)%p(2)%na, "x", xl(g)%p(2)%nb, "|", xl(g)%p(3)%na, "x", xl(g)%p(3)%nb
      enddo
      write(*,'(A,A,A,I0,A,I0,A,I0,A,I0)') " (", trim(adjustl(shape)), "), ", count(drop), &
        " Dirichlet DOFs dropped, rings ", xl(0)%nring, ", switch ring ", H%ring_is, ", coarsest DOFs ", &
        xl(H%nlev - 1)%n
    endif
    H%xgrid = .true.
    H%p_ready = .true.
    ok = .true.
    call PetscLogEventEnd(gev_prolong, ierr)

  contains

    subroutine fail(m_)
      character(len=*), intent(in) :: m_
      if (hier_root()) write(*,'(A,A)') "[Physics PC]   GMG X-point hierarchy REFUSED: ", m_
      ok = .false.
      call PetscLogEventEnd(gev_prolong, ierr)
    end subroutine fail

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

    !> Coarse cell (ia, jb) of patch level C_ over F: the splits lambda from
    !! the fine s2 / s3 sizes (mean of the cell's two edges), corner nodes from
    !! the corner fine elements, sizes S = s / lambda (or s / (1 - lambda)).
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
      if (allocated(H%xring(g_)%nn)) deallocate(H%xring(g_)%nn, H%xring(g_)%per)
      allocate(H%xring(g_)%nn(0:X%nring - 1), H%xring(g_)%per(0:X%nring - 1))
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
        H%xring(g_)%nn(rg_) = cnt_
        H%xring(g_)%per(rg_) = (rg_ >= 1 .and. rg_ < nbc .and. cnt_ == Lc%na)
      enddo
      end associate
    end subroutine level_dofs

    !> P(gg) as scalar triplets rr / cc / ww(1:ntrip) (level gg-1 and gg DOFs,
    !! 0-based) of the rank's own fine rows; a fine row seen by several
    !! (coarse cell, fine element, slot) instances gets their average.
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

  !> Canonical DOF numbering of one coarse level: the axis keeps the shared
  !! value and per-node a, c (a coarse angular slope b there is not in the fine
  !! space); the boundary drops the Dirichlet DOFs u, b (subdivision keeps them 0).
  subroutine make_level(L, ni, nj)
    type(lvl_t), intent(inout) :: L
    integer, intent(in) :: ni, nj
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
          if (I == ni - 1 .and. (k == 0 .or. k == 2)) cycle
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

  !> Galerkin chain, smoothers, coarse and axis solves for operator A, at
  !! every PC rebuild (collective on H%comm; comm, the caller's, is the one the
  !! hierarchy was built on). P is reused, and with an unchanged pattern so
  !! are the coarse operators and the LUs' symbolic phases.
  subroutine gmg_setup_operator(A, comm, my_id, tag, opts)
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(in)     :: A
    integer, intent(in) :: comm, my_id
    character(len=*), intent(in), optional :: tag   !< label for the prints
    type(gmg_opts_t), intent(in) :: opts
    PetscErrorCode :: ierr
    PC   :: pc
    MatInfo :: minfo
    PetscInt :: nr
    integer :: g, rich_lev, mpierr, nlev
    real*8 :: nz0, nzt, nzg
    integer(8) :: sig(2)
    logical :: reuse, okb

    nlev = H%nlev
    ! the same Mat (object ids are never reused) with an unchanged nonzero
    ! state needs no check; anything else is hashed
    block
      PetscObjectState :: nzst
      PetscInt64 :: aid
      logical :: same_obj
      call MatGetNonzeroState(A, nzst, ierr)
      call PetscObjectGetId(A, aid, ierr)
      same_obj = H%op_ready .and. int(aid, 8) == H%a0_id .and. int(nzst, 8) == H%a0_nzst
      call MPI_Allreduce(MPI_IN_PLACE, same_obj, 1, MPI_LOGICAL, MPI_LAND, H%comm, mpierr)
      if (same_obj) then
        reuse = .true.
      else
        sig = pattern_sig(A, H%comm)
        reuse = H%op_ready .and. all(sig == H%a0_sig)
        H%a0_sig = sig
      endif
      H%a0_id = int(aid, 8); H%a0_nzst = int(nzst, 8)
    end block
    H%pat_fresh = H%op_ready .and. .not. reuse
    if (H%pat_fresh .and. my_id == 0) write(*,'(A,I0,A)') "[Physics PC]   GMG", cur_inst, &
      ": operator pattern changed, coarse operators and LUs rebuilt from scratch"

    if (H%op_ready .and. .not. reuse) then
      do g = 1, nlev - 1
        call MatDestroy(H%A(g), ierr)
      enddo
    endif

    call gmg_register_events()
    H%A(0) = A
    if (H%xperm) then
      ! the hierarchy's ring-major copy of A; the scatter once
      if (H%op_ready .and. reuse) then
        call MatCreateSubMatrix(A, H%xp_is, H%xp_is, MAT_REUSE_MATRIX, H%xp_A, ierr)
      else
        if (H%op_ready) call MatDestroy(H%xp_A, ierr)
        call MatCreateSubMatrix(A, H%xp_is, H%xp_is, MAT_INITIAL_MATRIX, H%xp_A, ierr)
      endif
      if (.not. H%op_ready) then
        call MatCreateVecs(H%xp_A, H%xp_x, H%xp_b, ierr)
        block
          Vec :: va
          call MatCreateVecs(A, va, PETSC_NULL_VEC, ierr)
          call VecScatterCreate(va, H%xp_is, H%xp_b, PETSC_NULL_IS, H%xp_sc, ierr)
          call VecDestroy(va, ierr)
        end block
      endif
      H%A(0) = H%xp_A
    endif
    ! Galerkin products on dense slot blocks (jorek_galerkin_blk.c) with
    ! several slots, else PETSc's MatPtAP; the choice of a fresh product is
    ! kept for its refills
    call PetscLogEventBegin(gev_ptap, ierr)
    do g = 1, nlev - 1
      if (.not. reuse) then
        H%ptap_blk(g) = .false.
        if (pc_ntor > 1) H%ptap_blk(g) = galerkin_blk(H%A(g - 1), H%Ps(g), int(pc_ntor), .false., H%A(g))
        if (.not. H%ptap_blk(g)) call MatPtAP(H%A(g - 1), H%P(g), MAT_INITIAL_MATRIX, 2.0d0, H%A(g), ierr)
      else if (H%ptap_blk(g)) then
        if (.not. galerkin_blk(H%A(g - 1), H%Ps(g), int(pc_ntor), .true., H%A(g))) then
          if (my_id == 0) write(*,'(A,I0,A,I0)') "[Physics PC]   FATAL: GMG", cur_inst, &
            " Galerkin block product refused a refill on level ", g
          call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
      else
        call MatPtAP(H%A(g - 1), H%P(g), MAT_REUSE_MATRIX, 2.0d0, H%A(g), ierr)
      endif
    enddo
    call PetscLogEventEnd(gev_ptap, ierr)
    call gmg_mem("Galerkin levels")
    ! the multiply kernel each level's matvecs run
    if (.not. H%op_ready .and. my_id == 0) then
      block
        character(len=80) :: ta, tp
        integer :: gg
        write(*,'(A,I0,A)', advance="no") "[Physics PC]   GMG", cur_inst, " level types (A/P):"
        do gg = 0, nlev - 1
          call MatGetType(H%A(gg), ta, ierr)
          tp = "-"
          if (gg > 0) call MatGetType(H%P(gg), tp, ierr)
          write(*,'(A,I0,A,A,A,A)', advance="no") "  ", gg, "=", trim(ta), "/", trim(tp)
        enddo
        write(*,*)
      end block
    endif

    call PetscLogEventBegin(gev_smsetup, ierr)
    ! the level operators' and prolongations' matvecs on the OpenMP block
    ! kernel (idempotent, re-plans on a new pattern only)
    do g = 0, nlev - 1
      okb = blockmv_attach(H%A(g), int(pc_ntor))
      if (g > 0) okb = blockmv_attach(H%P(g), 1)
    enddo
    if (.not. H%op_ready .and. my_id == 0) write(*,'(A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, &
      ": level matvecs on the OpenMP block kernel (bs ", pc_ntor, ", ", nlev, " levels)"
    H%smoother = opts%smoother
    if (H%smoother /= GMG_SM_RINGS .and. H%smoother /= GMG_SM_LINES .and. H%smoother /= GMG_SM_ZEBRA_RINGS) &
      call fatal("smoother must be GMG_SM_RINGS, GMG_SM_LINES or GMG_SM_ZEBRA_RINGS")
    H%pre0 = opts%pre0; H%post0 = opts%post0; H%nsmooth_c = opts%nsmooth_c
    if (opts%pre0 < 0 .or. opts%post0 < 0 .or. opts%pre0 + opts%post0 == 0 .or. opts%nsmooth_c < 1) &
      call fatal("needs pre0, post0 >= 0, pre0 + post0 > 0 and nsmooth_c > 0")
    H%harm_pair = opts%harm_pair
    if (H%harm_pair /= 1 .and. H%harm_pair /= 2) call fatal("harm_pair must be 1 or 2")
    if (opts%rich_omega <= 0.d0) call fatal("rich_omega must be positive")
    H%axis_k = opts%axis_rings
    if (H%axis_k == -2) then
      ! automatic: ring 1 joins the axis block where ring 0 is a large block
      block
        integer :: nrow0
        nrow0 = H%nth0
        if (H%xgrid) nrow0 = H%xring(0)%nn(0)
        nrow0 = nrow0 * H%nfields * 4 * merge(pc_ntor, min(2, pc_ntor), H%harm_pair == 2)
        H%axis_k = merge(1, 0, nrow0 > AXIS_RING1_ROWS)
      end block
    endif
    if (H%axis_k < 0) call fatal("axis_rings must be >= 0 or -2")
    ! levels >= RICH_FROM smooth with Richardson on the same blocks: no global
    ! reductions there. The mixed pair on X-point grids keeps GMRES (its coarse
    ! levels stall under Richardson at large dt)
    rich_lev = RICH_FROM
    if (H%xgrid .and. H%smoother == GMG_SM_RINGS .and. H%nfields > 1) rich_lev = nlev
    do g = 0, nlev - 2
      ! created once per hierarchy; a rebuild re-points the operators and
      ! refactors the smoother blocks
      if (.not. H%op_ready) call KSPCreate(H%comm, H%sm_ksp(g), ierr)
      call KSPSetOperators(H%sm_ksp(g), H%A(g), H%A(g), ierr)
      if (.not. H%op_ready) then
        if (g >= rich_lev) then
          call KSPSetType(H%sm_ksp(g), KSPRICHARDSON, ierr)
          call KSPRichardsonSetScale(H%sm_ksp(g), opts%rich_omega, ierr)
          call KSPSetNormType(H%sm_ksp(g), KSP_NORM_NONE, ierr)
          call KSPSetTolerances(H%sm_ksp(g), 1.d-30, 1.d-50, 1.d30, SM_STEPS, ierr)
        else
          call KSPSetType(H%sm_ksp(g), KSPGMRES, ierr)
          if (g == 0) then
            call KSPGMRESSetRestart(H%sm_ksp(g), max(H%pre0, H%post0), ierr)
          else
            call KSPGMRESSetRestart(H%sm_ksp(g), H%nsmooth_c, ierr)
          endif
          call KSPSetTolerances(H%sm_ksp(g), 1.d-30, 1.d-50, 1.d30, SM_STEPS, ierr)
          call KSPSetPCSide(H%sm_ksp(g), PC_RIGHT, ierr)
        endif
        call KSPSetInitialGuessNonzero(H%sm_ksp(g), PETSC_TRUE, ierr)
        call KSPGetPC(H%sm_ksp(g), pc, ierr)
        call PCSetType(pc, PCSHELL, ierr)
        call PCShellSetApply(pc, blk_apply, ierr)
        call PCShellSetName(pc, "C1 block smoother", ierr)
      endif
      call build_blocks(g, H%A(g))
      call KSPSetUp(H%sm_ksp(g), ierr)
    enddo
    call axd_factor_pending()
    call PetscLogEventEnd(gev_smsetup, ierr)
    call gmg_mem("smoothers")
    block
      integer :: ib(4), im(2), bb
      real*8 :: st(2)
      character(len=96) :: lbl
      ib = [H%blk(0)%nb, count(H%blk(0)%band), sum(H%blk(0:nlev-2)%nsing), 0]
      do g = 0, nlev - 2
        ib(4) = ib(4) + count(H%blk(g)%btd)
      enddo
      im = 0                                      ! max kl of the LAPACK blocks, max segments, level 0
      do bb = 1, H%blk(0)%nb
        if (.not. H%blk(0)%axblk(bb) .and. .not. H%blk(0)%btd(bb)) im(1) = max(im(1), H%blk(0)%kl(bb))
        if (H%blk(0)%btd(bb)) im(2) = max(im(2), H%blk(0)%rg(bb)%ns)
      enddo
      call MPI_Allreduce(MPI_IN_PLACE, ib, 4, MPI_INTEGER, MPI_SUM, H%comm, mpierr)
      call MPI_Allreduce(MPI_IN_PLACE, im, 2, MPI_INTEGER, MPI_MAX, H%comm, mpierr)
      ! the smoothers' factor storage over all levels (entries): sum, largest rank
      st(1) = smoother_storage()
      st(2) = st(1)
      call MPI_Allreduce(MPI_IN_PLACE, st(1), 1, MPI_DOUBLE_PRECISION, MPI_SUM, H%comm, mpierr)
      call MPI_Allreduce(MPI_IN_PLACE, st(2), 1, MPI_DOUBLE_PRECISION, MPI_MAX, H%comm, mpierr)
      if (my_id == 0) then
        lbl = ""
        if (present(tag)) lbl = " ("//trim(tag)//")"
        write(*,'(A,I0,A,A,I0,A,I0,A,I0,A,I0,A)', advance="no") "[Physics PC]   GMG", cur_inst, trim(lbl), &
          " smoother ", H%smoother, ": V(", H%pre0, ",", H%post0, ") on level 0, ", &
          H%nsmooth_c, " steps per side below"
        write(*,'(A,I0,A,I0,A,I0,A,I0,A)', advance="no") ", blocks ", ib(1), " on level 0 (", &
          ib(2), " banded, max kl ", im(1), "; ", ib(3), " singular -> point)"
        if (H%harm_pair == 2) write(*,'(A)', advance="no") ", all slots per block"
        if (ib(4) > 0) write(*,'(A,I0,A,I0,A)', advance="no") ", ring solver on ", ib(4), &
          " blocks over all levels ("//merge("double", "single", rk_ring == 8)//" precision, up to ", im(2), " segments on level 0)"
        write(*,'(A,F9.2,A,F9.2,A)') "; factors ", st(1) * 8.d-6, " MB (max rank ", st(2) * 8.d-6, " MB)"
        if (H%smoother == GMG_SM_ZEBRA_RINGS) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, &
          ": zebra ring blocks on fine rings I < ", H%ring_is, ", zebra radial lines outside"
        write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, ": axis block = rings 0..", H%axis_k, &
          " on every level"
        if (rich_lev < nlev - 1) write(*,'(A,I0,A,I0,A,F5.2,A)') &
          "[Physics PC]   GMG", cur_inst, ": Richardson smoothing on levels >= ", rich_lev, &
          " (scale ", opts%rich_omega, ", no reductions)"
      endif
    end block

    ! the coarsest level: an LU done redundantly on the few ranks owning its
    ! rows (rds_t), or above CRS_DIST_ROWS rows one distributed MUMPS LU
    call PetscLogEventBegin(gev_coarselu, ierr)
    if (.not. H%coarse%ready) then
      call MatGetLocalSize(H%A(nlev - 1), nr, PETSC_NULL_INTEGER, ierr)
      allocate(H%coarse%loc(nr))
      do g = 1, int(nr)
        H%coarse%loc(g) = g - 1
      enddo
      block
        PetscInt :: nglob
        call MatGetSize(H%A(nlev - 1), nglob, PETSC_NULL_INTEGER, ierr)
        H%coarse%dist = (nglob > CRS_DIST_ROWS)
      end block
    endif
    call gmg_mem("before the coarse LU")
    if (H%coarse%dist) then
      call crs_dist_setup()
    else
      call rds_setup(H%coarse, H%A(nlev - 1), "coarse", "coarse level")
    endif
    call PetscLogEventEnd(gev_coarselu, ierr)
    if (.not. H%vec_ready) then
      do g = 0, nlev - 1
        call MatCreateVecs(H%A(g), H%x(g), H%b(g), ierr)
        call VecDuplicate(H%x(g), H%r(g), ierr)
      enddo
      H%vec_ready = .true.
    endif
    H%pat_fresh = .false.
    if (H%op_ready) return
    H%op_ready = .true.

    ! operator complexity, first build only
    call MatGetInfo(H%A(0), MAT_GLOBAL_SUM, minfo, ierr)
    nz0 = minfo%nz_used; nzt = nz0
    if (my_id == 0) write(*,'(A)', advance="no") "[Physics PC]   GMG levels n (nnz/row):"
    do g = 0, nlev - 1
      call MatGetSize(H%A(g), nr, PETSC_NULL_INTEGER, ierr)
      call MatGetInfo(H%A(g), MAT_GLOBAL_SUM, minfo, ierr)
      nzg = minfo%nz_used
      if (g > 0) nzt = nzt + nzg
      if (my_id == 0) write(*,'(A,I0,A,F7.1,A)', advance="no") " ", nr, " (", nzg / max(dble(nr), 1.d0), ")"
    enddo
    if (my_id == 0) write(*,'(A,F6.3)') "   C_op = ", nzt / max(nz0, 1.d0)

  contains

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (hier_root()) write(*,'(A,I0,A,A)') "[Physics PC]   FATAL: GMG", cur_inst, ": ", msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

    !> The coarsest level's LU as one MUMPS factorisation over the hierarchy's
    !! communicator; rebuilds refactor numerically
    subroutine crs_dist_setup()
      PC :: cpc
      PetscInt :: nglob
      integer :: np
      character(len=64) :: pre
      if (.not. H%coarse%ready) then
        call KSPCreate(H%comm, H%coarse%dksp, ierr)
        call KSPSetType(H%coarse%dksp, KSPPREONLY, ierr)
        call KSPGetPC(H%coarse%dksp, cpc, ierr)
        call PCSetType(cpc, PCLU, ierr)
        call PCFactorSetMatSolverType(cpc, MATSOLVERMUMPS, ierr)
        write(pre, '(A,I0,A)') "sf_gmg", cur_inst, "_coarse_"
        call KSPSetOptionsPrefix(H%coarse%dksp, trim(pre), ierr)
        call KSPSetFromOptions(H%coarse%dksp, ierr)
      endif
      call KSPSetOperators(H%coarse%dksp, H%A(nlev - 1), H%A(nlev - 1), ierr)
      call KSPSetUp(H%coarse%dksp, ierr)
      if (.not. H%coarse%ready) then
        call MatGetSize(H%A(nlev - 1), nglob, PETSC_NULL_INTEGER, ierr)
        call MPI_Comm_size(H%comm, np, mpierr)
        if (my_id == 0) write(*,'(A,I0,A,I0,A,I0,A,I0,A)') "[Mem] GMG", cur_inst, " coarse level LU: ", nglob, &
          " rows, one MUMPS factorisation over ", np, " ranks (> ", CRS_DIST_ROWS, " rows)"
        call gmg_mem("coarse LU")
      endif
      H%coarse%ready = .true.
    end subroutine crs_dist_setup
  end subroutine gmg_setup_operator

  !> bad on any rank of comm. The builders' rank-local tests go through it:
  !! every rank must refuse together, or the others run on into the
  !! collectives that follow.
  logical function any_rank(bad, comm)
    logical, intent(in) :: bad
    integer, intent(in) :: comm
    integer :: mpierr
    any_rank = bad
    call MPI_Allreduce(MPI_IN_PLACE, any_rank, 1, MPI_LOGICAL, MPI_LOR, comm, mpierr)
  end function any_rank

  !> The builders' fine layout, read off Aref: rank r owns the packed rows
  !! [rstarts(r), rstarts(r+1)), n_fields x n_tor rows per JOREK index
  !! 0..n_idx-1 (n_idx from the node list). msg /= "" (on every rank) if Aref
  !! does not have this layout.
  subroutine fine_layout(Aref, comm, n_fields, n_idx, ly0, msg)
    use nodes_elements,    only: node_list
    use mod_parameters,    only: n_degrees
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(in)               :: Aref
    integer, intent(in)           :: comm, n_fields
    integer, intent(out)          :: n_idx
    type(lay_t), intent(out)      :: ly0
    character(len=*), intent(out) :: msg
    integer, allocatable :: rstarts(:)
    integer :: n, np, r, first, idx, mpierr
    PetscInt :: rs0, re0, nglob
    PetscErrorCode :: ierr

    msg = ""
    n_idx = 0
    do n = 1, node_list%n_nodes
      n_idx = max(n_idx, maxval(node_list%node(n)%index(1:n_degrees)))
    enddo
    call MPI_Comm_size(comm, np, mpierr)
    call MatGetOwnershipRange(Aref, rs0, re0, ierr)
    call MatGetSize(Aref, nglob, PETSC_NULL_INTEGER, ierr)
    if (nglob /= int(n_fields, 8) * n_idx * pc_ntor) then
      msg = "operator size /= n_fields * n_index * n_tor"
      return
    endif
    allocate(rstarts(0:np))
    call MPI_Allgather(int(rs0), 1, MPI_INTEGER, rstarts, 1, MPI_INTEGER, comm, mpierr)
    rstarts(np) = int(nglob)
    ly0%n = n_idx
    allocate(ly0%own(0:n_idx - 1), ly0%lpos(0:n_idx - 1), ly0%nl(0:np - 1), ly0%ps(0:np))
    do r = 0, np - 1
      if (mod(rstarts(r + 1) - rstarts(r), n_fields * pc_ntor) /= 0) then
        msg = "a rank's rows are not whole (field x toroidal slot) groups"
        return
      endif
      ly0%nl(r) = (rstarts(r + 1) - rstarts(r)) / (n_fields * pc_ntor)
      ly0%ps(r) = rstarts(r)
    enddo
    ly0%ps(np) = rstarts(np)
    r = 0; first = 0
    do idx = 0, n_idx - 1
      do while (idx >= first + ly0%nl(r))
        first = first + ly0%nl(r); r = r + 1
      enddo
      ly0%own(idx) = r; ly0%lpos(idx) = idx - first
    enddo
  end subroutine fine_layout

  !> Expand a scalar (one field, one harmonic) prolongation, triplets r_ / c
  !! / v of 0-based DOFs with rows on layout level lr_ and columns on lc_,
  !! into the packed operators' layouts: P = I_fields x P_s x I_slots and
  !! Ps = I_fields x P_s. Each rank inserts the rows it owns. nz: entries
  !! preallocated per row in each part; grow: further entries may allocate.
  subroutine expand_p(P, Ps, comm, ly, me, nf, lr_, lc_, r_, c, v, nz, grow)
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(out)        :: P, Ps
    integer, intent(in)     :: comm, me, nf, lr_, lc_, r_(:), c(:), nz
    type(lay_t), intent(in) :: ly(0:)
    real*8, intent(in)      :: v(:)
    logical, intent(in)     :: grow
    PetscInt :: nrl, ncl, nrg, ncg, row, col, nzp
    PetscErrorCode :: ierr
    integer :: f, m, q

    nrl = nf * pc_ntor * ly(lr_)%nl(me); ncl = nf * pc_ntor * ly(lc_)%nl(me)
    nrg = nf * pc_ntor * ly(lr_)%n;      ncg = nf * pc_ntor * ly(lc_)%n
    nzp = nz
    call MatCreate(comm, P, ierr)
    call MatSetSizes(P, nrl, ncl, nrg, ncg, ierr)
    call MatSetType(P, MATMPIAIJ, ierr)       ! the packed pairs are MPIAIJ even on one rank
    call MatSeqAIJSetPreallocation(P, nzp, PETSC_NULL_INTEGER_ARRAY, ierr)
    call MatMPIAIJSetPreallocation(P, nzp, PETSC_NULL_INTEGER_ARRAY, nzp, PETSC_NULL_INTEGER_ARRAY, ierr)
    if (grow) call MatSetOption(P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    do q = 1, size(r_)
      if (ly(lr_)%own(r_(q)) /= me) cycle
      do f = 0, nf - 1
        do m = 0, pc_ntor - 1
          row = prow(lr_, f, r_(q), m)
          col = prow(lc_, f, c(q), m)
          call MatSetValue(P, row, col, v(q), ADD_VALUES, ierr)
        enddo
      enddo
    enddo
    call MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY, ierr)
    call make_slot_free(Ps, comm, nrl, ncl, nrg, ncg, size(r_), &
                        [(ly(lr_)%own(r_(q)) == me, q = 1, size(r_))], &
                        [(prow(lr_, 0, r_(q), 0), q = 1, size(r_))], [(prow(lc_, 0, c(q), 0), q = 1, size(r_))], &
                        [(prow(lr_, 1, r_(q), 0) - prow(lr_, 0, r_(q), 0), q = 1, size(r_))], &
                        [(prow(lc_, 1, c(q), 0) - prow(lc_, 0, c(q), 0), q = 1, size(r_))], v, nf)

  contains

    !> Packed global row of (field f_, scalar DOF d, slot m_) on layout level L.
    integer function prow(L, f_, d, m_)
      integer, intent(in) :: L, f_, d, m_
      integer :: ow
      ow = ly(L)%own(d)
      prow = ly(L)%ps(ow) + (f_ * ly(L)%nl(ow) + ly(L)%lpos(d)) * pc_ntor + m_
    end function prow
  end subroutine expand_p

  !> Ps = I_fields x P_s (one row per pc_ntor packed rows). Triplet t (owned
  !! when own(t)): field-0 packed row r0 / column c0 at slot 0, the packed
  !! offsets dr / dc from one field to the next.
  subroutine make_slot_free(Ps, comm, nrl, ncl, nrg, ncg, n, own, r0, c0, dr, dc, v, nf)
    use mod_petsc_pc_harm, only: pc_ntor
    Mat, intent(out) :: Ps
    integer, intent(in) :: comm, n, nf
    PetscInt, intent(in) :: nrl, ncl, nrg, ncg
    logical, intent(in) :: own(n)
    integer, intent(in) :: r0(n), c0(n), dr(n), dc(n)
    real*8, intent(in) :: v(n)
    PetscErrorCode :: ierr
    PetscInt :: row, col
    integer :: t, f
    call MatCreate(comm, Ps, ierr)
    call MatSetSizes(Ps, nrl / pc_ntor, ncl / pc_ntor, nrg / pc_ntor, ncg / pc_ntor, ierr)
    call MatSetType(Ps, MATMPIAIJ, ierr)
    call MatSeqAIJSetPreallocation(Ps, 2 * MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr)
    call MatMPIAIJSetPreallocation(Ps, 2 * MAX_ENT, PETSC_NULL_INTEGER_ARRAY, &
                                   2 * MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr)
    call MatSetOption(Ps, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    do t = 1, n
      if (.not. own(t)) cycle
      do f = 0, nf - 1
        row = (r0(t) + f * dr(t)) / pc_ntor
        col = (c0(t) + f * dc(t)) / pc_ntor
        call MatSetValue(Ps, row, col, v(t), ADD_VALUES, ierr)
      enddo
    enddo
    call MatAssemblyBegin(Ps, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(Ps, MAT_FINAL_ASSEMBLY, ierr)
  end subroutine make_slot_free

  !> [Mem] checkpoint of the GMG setup (RSS peak: rank 0, max and sum over the
  !! hierarchy's ranks; first build only)
  subroutine gmg_mem(what)
    use mod_petsc_pc_physics_ctx, only: physics_pc_peak_rss
    character(len=*), intent(in) :: what
    real*8 :: r, rmax, rsum
    integer :: mpierr, imax, loc
    if (H%op_ready) return
    r = physics_pc_peak_rss() / 1.d9
    call MPI_Allreduce(r, rmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, H%comm, mpierr)
    call MPI_Allreduce(r, rsum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, H%comm, mpierr)
    imax = merge(H%my_id, -1, r >= rmax)
    call MPI_Allreduce(imax, loc, 1, MPI_INTEGER, MPI_MAX, H%comm, mpierr)
    if (H%my_id == 0) write(*,'(A,I0,A,A,T52,A,F7.3,A,F7.3,A,I0,A,F8.1,A)') "[Mem] GMG", cur_inst, ": ", what, &
      "RSS peak rank 0 ", r, " GB, max ", rmax, " GB (rank ", loc, "), sum ", rsum, " GB"
  end subroutine gmg_mem

  !> Entries of the active hierarchy's smoother factors on this rank, all
  !! smoothing levels.
  real*8 function smoother_storage()
    integer :: g
    smoother_storage = 0.d0
    do g = 0, H%nlev - 2
      if (allocated(H%blk(g)%lu4)) smoother_storage = smoother_storage + (rk_ring / 8.d0) * size(H%blk(g)%lu4, kind=8)
      if (allocated(H%blk(g)%lu)) then
        smoother_storage = smoother_storage + dble(size(H%blk(g)%lu, kind=8))
      endif
    enddo
  end function smoother_storage

  !> Two hashes of A's sparsity pattern (row lengths, global columns of the
  !! owned rows), summed over the ranks.
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

  !> The -log_view events: setup ones shared, apply ones per instance
  !! (GMG_<what> for instance 1, GMG<k>_<what> for k > 1).
  subroutine gmg_register_events()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    character(len=8) :: p
    integer :: k
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
    call PetscLogEventRegister("GMG_Ovl",      cid, gev_ovl,      ierr)
    do k = 1, MAX_INST
      p = "GMG_"
      if (k > 1) write(p, '(A,I0,A)') "GMG", k, "_"
      call PetscLogEventRegister(trim(p)//"VCycle",  cid, gev_vcycle(k),  ierr)
      call PetscLogEventRegister(trim(p)//"Smooth0", cid, gev_smooth0(k), ierr)
      call PetscLogEventRegister(trim(p)//"SmoothC", cid, gev_smooth(k),  ierr)
      call PetscLogEventRegister(trim(p)//"Coarse",  cid, gev_coarse(k),  ierr)
      call PetscLogEventRegister(trim(p)//"Lines",   cid, gev_lines(k),   ierr)
      call PetscLogEventRegister(trim(p)//"AxSolve", cid, gev_axsolve(k), ierr)
    enddo
    gev_ready = .true.
  end subroutine gmg_register_events

  !> One V-cycle of H from level g (x out, zero initial guess). Level 0 takes
  !! pre0 / post0 steps, the coarse levels nsmooth_c each side. Without
  !! pre-smoothing x is still 0 at the restriction, so r = b needs no matvec.
  recursive subroutine vcycle(g, b, x)
    integer, intent(in) :: g
    Vec :: b, x
    PetscErrorCode :: ierr
    integer :: npre, npost

    if (g == H%nlev - 1) then
      call PetscLogEventBegin(gev_coarse(cur_inst), ierr)
      if (H%coarse%dist) then
        call KSPSolve(H%coarse%dksp, b, x, ierr)
      else
        call rds_solve(H%coarse, b, x)
      endif
      call PetscLogEventEnd(gev_coarse(cur_inst), ierr)
      return
    endif
    call VecZeroEntries(x, ierr)
    if (g == 0) then
      npre = H%pre0; npost = H%post0
    else
      npre = H%nsmooth_c; npost = H%nsmooth_c
    endif
    if (npre > 0) call smooth(g, b, x, .true., npre)     ! pre-smooth (zero guess: no b - A*0)
    if (npre == 0) then
      call VecCopy(b, H%r(g), ierr)                       ! x is still 0: r = b
    else
      call MatMult(H%A(g), x, H%r(g), ierr)
      call VecAYPX(H%r(g), -1.0d0, b, ierr)               ! r = b - A x
    endif
    call MatMultTranspose(H%P(g + 1), H%r(g), H%b(g + 1), ierr)
    call vcycle(g + 1, H%b(g + 1), H%x(g + 1))
    call MatMultAdd(H%P(g + 1), H%x(g + 1), x, x, ierr)  ! x += P e_c
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
    H%sm_lev = g
    ns = nstep
    call KSPSetTolerances(H%sm_ksp(g), 1.d-30, 1.d-50, 1.d30, ns, ierr)
    if (zero_guess) then
      call KSPSetInitialGuessNonzero(H%sm_ksp(g), PETSC_FALSE, ierr)
    else
      call KSPSetInitialGuessNonzero(H%sm_ksp(g), PETSC_TRUE, ierr)
    endif
    if (g == 0) then
      call PetscLogEventBegin(gev_smooth0(cur_inst), ierr)
    else
      call PetscLogEventBegin(gev_smooth(cur_inst), ierr)
    endif
    call KSPSolve(H%sm_ksp(g), b, x, ierr)
    if (g == 0) then
      call PetscLogEventEnd(gev_smooth0(cur_inst), ierr)
    else
      call PetscLogEventEnd(gev_smooth(cur_inst), ierr)
    endif
  end subroutine smooth

  !> The smoother blocks of level g and their factors, collective on H%comm:
  !! the mesh part on the first build (blocks, ghosts, ring-solver layouts),
  !! the pattern part once per operator pattern (blk_pattern), the numeric
  !! part every rebuild (gather, factor; a singular block falls back to its diagonal).
  subroutine build_blocks(g, A)
    integer, intent(in) :: g
    Mat, intent(in)     :: A
    type(blk_t), pointer :: B
    PetscInt :: nr, r, ncols, rst, ren
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    PetscInt, parameter :: one = 1
    integer :: nc, I, J, m, bb, q, pcn, n, ldab, mpierr
    integer, allocatable :: cnt(:), key(:), binfo(:)
    integer(8) :: ix
    real*8, allocatable :: wsc(:)
    real*8 :: dv
    external :: dgetrf, dgbtrf

    B => H%blk(g)
    call MatGetLocalSize(A, nr, PETSC_NULL_INTEGER, ierr)
    call MatGetOwnershipRange(A, rst, ren, ierr)
    if (.not. allocated(B%bid)) then
      call PetscLogEventBegin(gev_blkmesh, ierr)
      B%nrow = int(nr)
      nc = level_nj(g)
      allocate(B%bid(B%nrow), key(B%nrow))
      do r = 1, B%nrow
        call row_node(g, int(r), I, J, m)
        B%bid(r) = blk_id(g, I, row_col(g, int(r), J), m)
        if (I <= H%axis_k) then
          key(r) = I * nc + J
        else if (ring_block(g, I)) then
          key(r) = rkey(g, I, J, nc)
        else
          key(r) = I
        endif
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
      ! the axis block's blocks (a sparse direct solve) and the zebra colours:
      ! rings alternate in I, radial lines in their column
      allocate(B%axblk(B%nb), B%bcol(B%nb))
      do bb = 1, B%nb
        r = B%rows(B%off(bb) + 1) + 1
        call row_node(g, int(r), I, J, m)
        B%axblk(bb) = (I <= H%axis_k)
        B%bcol(bb) = 0
        if (H%smoother == GMG_SM_ZEBRA_RINGS .and. .not. B%axblk(bb)) then
          if (ring_block(g, I)) then
            B%bcol(bb) = mod(I, 2)
          else
            B%bcol(bb) = mod(row_col(g, int(r), J), 2)
          endif
        endif
      enddo
      allocate(B%axr)
      allocate(B%axr%loc(count(B%axblk(B%bid))))
      n = 0
      do r = 1, B%nrow
        if (.not. B%axblk(B%bid(r))) cycle
        n = n + 1; B%axr%loc(n) = int(r) - 1
      enddo
      call PetscLogEventBegin(gev_ovl, ierr)
      call ovl_extend(g, A, B, rst, ren)
      call PetscLogEventEnd(gev_ovl, ierr)
      call btd_mark(g, B)
      call work_reserve(max(1, maxval(B%sz, mask=.not. B%axblk)))
      call PetscLogEventEnd(gev_blkmesh, ierr)
    endif

    ! Pattern part, once per operator pattern (the fine operand and the
    ! Galerkin chain keep their Mats for the run); a rebuild refills the ghost
    ! rows' submatrix
    block
      PetscObjectState :: nzst
      PetscInt64 :: aid
      logical :: newpat
      call MatGetNonzeroState(A, nzst, ierr)
      call PetscObjectGetId(A, aid, ierr)
      newpat = (int(aid, 8) /= B%pat_id .or. int(nzst, 8) /= B%pat_nz)
      call MPI_Allreduce(MPI_IN_PLACE, newpat, 1, MPI_LOGICAL, MPI_LOR, H%comm, mpierr)
      if (newpat) then
        call PetscLogEventBegin(gev_blkpat, ierr)
        call blk_pattern(g, B, A, rst, ren)
        call PetscLogEventEnd(gev_blkpat, ierr)
        B%pat_id = int(aid, 8); B%pat_nz = int(nzst, 8)
      else if (B%ovl) then
        call MatCreateSubMatrices(A, one, [B%isg], [B%isall], MAT_REUSE_MATRIX, B%sg, ierr)
      endif
    end block
    call PetscLogEventBegin(gev_blknum, ierr)
    call blk_fill(B, A)

    ! The blocks are independent: factored on the rank's threads. A ring
    ! block is factored in a double scratch (equilibrated, pivot blocks
    ! inverted) and rounded into lu4; with the threaded hybrid (B%wk) the rings
    ! follow one at a time, their segments on the threads. The singular-block
    ! fallback reads A (MatGetRow is not thread-safe), so it runs afterwards.
    allocate(binfo(B%nb))
    binfo = 0
    if (allocated(B%eqs)) then
      if (size(B%eqs) /= size(B%rows)) deallocate(B%eqs)
    endif
    if (.not. allocated(B%eqs)) allocate(B%eqs(size(B%rows)))
    B%eqs = 1.d0
    !$omp parallel do schedule(dynamic, 1) private(bb, n, ldab, wsc)
    do bb = 1, B%nb
      if (B%axblk(bb)) cycle
      n = B%sz(bb)
      if (B%btd(bb)) then
        B%jac(bb) = .false.
        if (allocated(B%wk)) cycle
        call scratch_in(bb, wsc)
        call btd_equil(B%rg(bb), wsc, B%eqs(B%off(bb) + 1))
        call btd_factor(B%rg(bb), wsc, B%piv(B%off(bb) + 1), binfo(bb))
        call ring_out(bb, wsc)
      else if (B%band(bb)) then
        ldab = 2 * B%kl(bb) + B%ku(bb) + 1
        call dgbtrf(n, n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), ldab, B%piv(B%off(bb) + 1), binfo(bb))
      else
        call dgetrf(n, n, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), binfo(bb))
      endif
    enddo
    !$omp end parallel do
    if (allocated(B%wk)) then
      do bb = 1, B%nb
        if (.not. B%btd(bb)) cycle
        call scratch_in(bb, wsc)
        call btd_equil(B%rg(bb), wsc, B%eqs(B%off(bb) + 1))
        call seg_factor(bb, wsc)
        call ring_out(bb, wsc)
      enddo
    endif
    B%nsing = 0
    do bb = 1, B%nb
      n = B%sz(bb)
      if (B%axblk(bb)) cycle
      if (binfo(bb) /= 0) then
        ! singular block (for the ring solver: a singular node block): its
        ! diagonal from A, point Jacobi
        B%nsing = B%nsing + 1
        if (B%btd(bb)) B%jac(bb) = .true.
        B%eqs(B%off(bb) + 1 : B%off(bb) + n) = 1.d0
        if (B%loff(bb) < 0) then
          B%lu4(B%loff4(bb) + 1 : B%loff4(bb) + blk_size(B, bb)) = 0.0
        else
          B%lu(B%loff(bb) + 1 : B%loff(bb) + blk_size(B, bb)) = 0.0d0
        endif
        do q = 1, n
          r = B%rows(B%off(bb) + q)
          ix = lu_index(B, bb, q, q)
          dv = 0.d0
          if (r < B%nrow) then
            call MatGetRow(A, rst + r, ncols, cols, vals, ierr)
            do pcn = 1, int(ncols)
              if (cols(pcn) == rst + r) dv = vals(pcn)
            enddo
            call MatRestoreRow(A, rst + r, ncols, cols, vals, ierr)
          else                                          ! ghost row: from sg(1), global columns
            call MatGetRow(B%sg(1), r - B%nrow, ncols, cols, vals, ierr)
            do pcn = 1, int(ncols)
              if (cols(pcn) == B%gidx(r - B%nrow + 1)) dv = vals(pcn)
            enddo
            call MatRestoreRow(B%sg(1), r - B%nrow, ncols, cols, vals, ierr)
          endif
          if (abs(dv) < tiny(1.0d0)) dv = 1.0d0
          if (ix > B%nlu) then
            B%lu4(ix - B%nlu) = real(dv, rk_ring)
          else
            B%lu(ix) = dv
          endif
          B%piv(B%off(bb) + q) = q
        enddo
      endif
    enddo
    if (allocated(B%wk)) then
      if (allocated(B%wk4)) then
        if (size(B%wk4) /= size(B%wk)) deallocate(B%wk4)
      endif
      if (.not. allocated(B%wk4)) allocate(B%wk4(size(B%wk)))
    endif
    call PetscLogEventEnd(gev_blknum, ierr)

    ! the axis block: its J-sector solve (axis_k > 0) where it passed the
    ! first build's gate, else the LU. The sector factorisations of all levels
    ! run together after the level loop (axd_factor_pending)
    if (B%axdon .and. H%pat_fresh) then
      ! the sector solve's submatrices hold the old pattern
      call axd_destroy(B%axd)
      B%axdon = .false.
      if (H%my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, " level ", g, &
        ": axis J-sector solve dropped for the LU after a pattern change"
    endif
    if (B%axdon) then
      call PetscLogEventBegin(gev_axnum, ierr)
      call axd_refill(B%axd, A)
      B%axpend = .true.
      B%axA = A
      call PetscLogEventEnd(gev_axnum, ierr)
    else if (H%axis_k > 0 .and. .not. allocated(B%axd)) then
      call PetscLogEventBegin(gev_axtry, ierr)
      call axd_prepare(g, A, B)
      call PetscLogEventEnd(gev_axtry, ierr)
    endif
    if (.not. B%axdon .and. .not. B%axnew) call ax_lu(g, A, B)

  contains

    !> ring block bb's single entries (lu4) into the double scratch w
    subroutine scratch_in(bb_, w)
      integer, intent(in) :: bb_
      real*8, allocatable, intent(inout) :: w(:)
      integer(8) :: t
      t = B%rg(bb_)%tot
      if (allocated(w)) then
        if (size(w, kind=8) < t) deallocate(w)
      endif
      if (.not. allocated(w)) allocate(w(t))
      w(1:t) = real(B%lu4(B%loff4(bb_) + 1 : B%loff4(bb_) + t), 8)
    end subroutine scratch_in

    !> a factored ring bb in w: pivot blocks inverted, rounded into lu4
    subroutine ring_out(bb_, w)
      integer, intent(in) :: bb_
      real*8, intent(inout) :: w(:)
      if (binfo(bb_) /= 0) return
      call btd_invert(B%rg(bb_), w, B%piv(B%off(bb_) + 1))
      B%lu4(B%loff4(bb_) + 1 : B%loff4(bb_) + B%rg(bb_)%tot) = real(w(1:B%rg(bb_)%tot), rk_ring)
    end subroutine ring_out

    !> hybrid ring bb in w: its segments on the threads, then the separators
    subroutine seg_factor(bb_, w)
      integer, intent(in) :: bb_
      real*8, intent(inout) :: w(:)
      integer :: k, inf
      !$omp parallel do schedule(dynamic, 1) private(inf)
      do k = 0, B%rg(bb_)%ns - 1
        call hyb_fac_seg(B%rg(bb_), w, B%piv(B%off(bb_) + 1), k, inf)
        if (inf /= 0) then
          !$omp atomic write
          binfo(bb_) = inf
        endif
      enddo
      !$omp end parallel do
      if (binfo(bb_) == 0) call hyb_fac_sep(B%rg(bb_), w, B%piv(B%off(bb_) + 1), binfo(bb_))
    end subroutine seg_factor
  end subroutine build_blocks

  !> Mesh part: the blocks the ring solver takes, rows re-sorted stably to node
  !! order (qn): whole rings or one contiguous arc of >= 4 nodes, and radial
  !! lines on closed grids. Hybrid segments per ring: on smoother 4 (threaded,
  !! lines_solve_par) ceil(threads / rings), at most sqrt(nodes); else 1.
  subroutine btd_mark(g, B)
    integer, intent(in) :: g
    type(blk_t), intent(inout) :: B
    integer :: nc, nall, r, bb, q, I, J, m, ns, nthr, n, k, nimax, j0, p, nst
    integer, allocatable :: key(:), cnt(:,:), nnb(:), lcn(:,:)
    logical, allocatable :: lchain(:)
    nall = B%nrow + B%ngh
    if (allocated(B%btd)) deallocate(B%btd, B%jac, B%rg, B%qn)
    allocate(B%btd(B%nb), B%jac(B%nb), B%rg(B%nb), B%qn(nall))
    B%btd = .false.; B%jac = .false.; B%qn = 0
    if (allocated(B%wk)) deallocate(B%wk, B%lb, B%lk)
    nc = level_nj(g)
    if (nc < 4) return
    allocate(key(nall), cnt(0:nc - 1, B%nb), nnb(B%nb), lchain(B%nb))
    nnb = nc; lchain = .false.
    nimax = 1
    if (H%smoother /= GMG_SM_RINGS) then             ! the radial lines' node counts
      do r = 1, nall
        nimax = max(nimax, row_I(r) + 1)
      enddo
      allocate(lcn(0:nimax - 1, B%nb))
    endif
    do r = 1, nall
      if (r <= B%nrow) then
        call row_node(g, r, I, key(r), m)
      else
        key(r) = B%ghJ(r - B%nrow)
      endif
    enddo
    do bb = 1, B%nb
      if (B%axblk(bb)) cycle
      I = -1                                         ! the block's I, from one of its own rows
      do q = 1, B%sz(bb)
        r = B%rows(B%off(bb) + q) + 1
        if (r > B%nrow) cycle
        call row_node(g, r, I, J, m)
        exit
      enddo
      if (I < 0) cycle
      if (.not. ring_block(g, I)) then
        if (.not. H%xgrid) call line_chain(bb)
        cycle
      endif
      if (H%xgrid) nnb(bb) = H%xring(g)%nn(I)         ! an X-point grid's ring, maybe open
      if (nnb(bb) < 4) cycle
      cnt(:, bb) = 0
      do q = 1, B%sz(bb)
        r = B%rows(B%off(bb) + q) + 1
        cnt(key(r), bb) = cnt(key(r), bb) + 1
      enddo
      if (any(cnt(0:nnb(bb) - 1, bb) == 0)) then
        ! cut: one contiguous arc of >= 4 nodes is a ring of the arc's length
        ! with zero wrap couplings
        nst = 0; j0 = -1; m = count(cnt(0:nnb(bb) - 1, bb) > 0)
        do p = 0, nnb(bb) - 1
          if (cnt(p, bb) > 0 .and. cnt(modulo(p - 1, nnb(bb)), bb) == 0) then
            nst = nst + 1; j0 = p
          endif
        enddo
        if (nst /= 1 .or. m < 4) cycle
        do q = 1, B%sz(bb)
          r = B%rows(B%off(bb) + q) + 1
          key(r) = modulo(key(r) - j0, nnb(bb))
        enddo
        cnt(0:nnb(bb) - 1, bb) = cshift(cnt(0:nnb(bb) - 1, bb), j0)
        nnb(bb) = m
      endif
      call sort_rows_by_key(B%rows(B%off(bb) + 1:B%off(bb) + B%sz(bb)), key)
      do q = 1, B%sz(bb)
        r = B%rows(B%off(bb) + q)
        B%pos(r + 1) = q
        B%qn(B%off(bb) + q) = key(r + 1)
      enddo
      B%btd(bb) = .true.
    enddo
    ns = 1
    if (H%smoother == GMG_SM_RINGS) then
      nthr = 1
      !$ nthr = omp_get_max_threads()
      n = max(count(B%btd), 1)
      if (n < nthr) ns = (nthr + n - 1) / n
    endif
    do bb = 1, B%nb
      if (.not. B%btd(bb)) cycle
      if (lchain(bb)) then
        call btd_plan(B%rg(bb), RS_THOMAS, nnb(bb), lcn(0:nnb(bb) - 1, bb), 1)
      else
        call btd_plan(B%rg(bb), RS_HYB, nnb(bb), cnt(0:nnb(bb) - 1, bb), &
                      min(ns, max(1, int(sqrt(dble(nnb(bb)))))))
      endif
    enddo
    ! smoother 4 with segmented rings: the hybrid threaded over (block,
    ! segment) work items, one item for every other non-axis block
    if (ns > 1 .and. any(B%btd)) then
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

  contains

    !> I of combined row r
    integer function row_I(r_)
      integer, intent(in) :: r_
      integer :: J_, m_
      if (r_ <= B%nrow) then
        call row_node(g, r_, row_I, J_, m_)
      else
        row_I = B%ghI(r_ - B%nrow)
      endif
    end function row_I

    !> block bb as a radial line: its rows' I must cover one contiguous range
    !! of >= 4 nodes; key = I - I_min, rows sorted by it
    subroutine line_chain(bb_)
      integer, intent(in) :: bb_
      integer :: q_, r_, i0, i1, m
      i0 = huge(1); i1 = -1
      do q_ = 1, B%sz(bb_)
        r_ = row_I(B%rows(B%off(bb_) + q_) + 1)
        if (r_ < 0) return
        i0 = min(i0, r_); i1 = max(i1, r_)
      enddo
      m = i1 - i0 + 1
      if (m < 4) return
      lcn(0:m - 1, bb_) = 0
      do q_ = 1, B%sz(bb_)
        r_ = B%rows(B%off(bb_) + q_) + 1
        key(r_) = row_I(r_) - i0
        lcn(key(r_), bb_) = lcn(key(r_), bb_) + 1
      enddo
      if (any(lcn(0:m - 1, bb_) == 0)) return
      call sort_rows_by_key(B%rows(B%off(bb_) + 1:B%off(bb_) + B%sz(bb_)), key)
      do q_ = 1, B%sz(bb_)
        r_ = B%rows(B%off(bb_) + q_)
        B%pos(r_ + 1) = q_
        B%qn(B%off(bb_) + q_) = key(r_ + 1)
      enddo
      nnb(bb_) = m; lchain(bb_) = .true.; B%btd(bb_) = .true.
    end subroutine line_chain
  end subroutine btd_mark

  !> Pattern part of build_blocks (collective on H%comm): block storage, the
  !! zebra coupling CSR and the value maps from A (and sg(1)), in two passes.
  !! A ring-solver block coupling beyond J +- 1 goes back to the LAPACK band LU.
  subroutine blk_pattern(g, B, A, rst, ren)
    integer, intent(in) :: g
    type(blk_t), intent(inout) :: B
    Mat, intent(in)      :: A
    PetscInt, intent(in) :: rst, ren
    PetscInt, parameter :: one = 1
    Mat :: Ad, Ao
    PetscInt, pointer :: garr(:), dia(:), dja(:), oia(:), oja(:), sia(:), sja(:)
    PetscInt :: nd, no, ns
    logical :: has_o
    integer :: pass, R, nall, bb, n, nva, nvg, nbad, mpierr
    integer, allocatable :: cva(:), cvg(:)
    logical, allocatable :: bad(:)
    !> one row's scan (scan_row): its block, zebra entries, ghost search
    !! hint (comb), map positions (own, ghost), band below / above the
    !! diagonal, a coupling the ring solver cannot take
    type :: row_scan_t
      integer :: bb = 0, kz = 0, chint = 0, nva = 0, nvg = 0, kl = 0, ku = 0
      logical :: bad = .false.
    end type row_scan_t
    PetscInt :: k
    integer(8) :: tot, tot4
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
    ! pass 1 counts each row's entries and gathers the band widths, the ring
    ! check (bad) and the zebra counts; pass 2 writes the maps
    allocate(cva(0:nall), cvg(0:nall), bad(B%nb))
    bad = .false.
    do pass = 1, 2
      !$omp parallel do schedule(dynamic, 16)
      do R = 0, nall - 1
        call scan_row(R, pass)
      enddo
      !$omp end parallel do
      if (pass == 2) exit
      nbad = count(bad .and. B%btd)
      if (nbad > 0) then
        ! demoted ring blocks: back to the band order, pass 1 again for the band widths
        call band_order()
        B%kl = 0; B%ku = 0
        !$omp parallel do schedule(dynamic, 16)
        do R = 0, nall - 1
          call scan_row(R, 1)
        enddo
        !$omp end parallel do
      endif
      call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, H%comm, mpierr)
      if (H%my_id == 0 .and. nbad > 0) write(*,'(A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, " level ", g, &
        ": ", nbad, " ring block(s) couple beyond J +- 1, LAPACK band LU instead of the ring solver"
      nva = 0; nvg = 0
      do R = 0, nall - 1
        n = cva(R); cva(R) = nva; nva = nva + n
        n = cvg(R); cvg(R) = nvg; nvg = nvg + n
      enddo
      ! storage: banded where the band is narrow, dense otherwise; the ring
      ! solver's blocks in lu4 only (4 instead of 12 bytes per entry)
      tot = 0; tot4 = 0
      if (allocated(B%loff4)) deallocate(B%loff4)
      allocate(B%loff4(B%nb))
      B%loff4 = 0
      do bb = 1, B%nb
        n = B%sz(bb)
        B%band(bb) = (2 * (2 * B%kl(bb) + B%ku(bb) + 1) < n)
        if (B%axblk(bb) .or. B%btd(bb)) B%band(bb) = .false.
        if (B%btd(bb)) then
          B%loff(bb) = -1; B%loff4(bb) = tot4
          tot4 = tot4 + blk_size(B, bb)
        else
          B%loff(bb) = tot
          tot = tot + blk_size(B, bb)
        endif
      enddo
      if (allocated(B%lu)) deallocate(B%lu)
      allocate(B%lu(tot))
      B%nlu = tot
      if (allocated(B%lu4)) deallocate(B%lu4)
      allocate(B%lu4(tot4))
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

    !> The ring-solver blocks found bad: btd off, rows in the ring's band order
    !! rkey (a radial line, one J, keeps its order)
    subroutine band_order()
      integer, allocatable :: rk(:)
      integer :: r_, q_, b_, I_, J_, m_
      allocate(rk(nall))
      do r_ = 1, nall
        if (r_ <= B%nrow) then
          call row_node(g, r_, I_, J_, m_)
        else
          I_ = B%ghI(r_ - B%nrow); J_ = B%ghJ(r_ - B%nrow)
        endif
        rk(r_) = rkey(g, I_, J_, level_nj(g))
      enddo
      do b_ = 1, B%nb
        if (.not. (bad(b_) .and. B%btd(b_))) cycle
        B%btd(b_) = .false.
        call sort_rows_by_key(B%rows(B%off(b_) + 1:B%off(b_) + B%sz(b_)), rk)
        do q_ = 1, B%sz(b_)
          B%pos(B%rows(B%off(b_) + q_) + 1) = q_
        enddo
      enddo
      bad = .false.
    end subroutine band_order

    !> Row R's entries, pass 1 (count, band, ring check, zebra count) or 2
    !! (the maps, from the row's offsets). Thread-safe: row-private state,
    !! per-block results by atomic updates.
    subroutine scan_row(R_, pass_)
      integer, intent(in) :: R_, pass_
      type(row_scan_t) :: st
      PetscInt :: k_
      st = row_scan_t(bb = B%bid(R_ + 1))
      if (pass_ == 2) then
        st%nva = cva(R_); st%nvg = cvg(R_)
      endif
      if (R_ < B%nrow) then
        do k_ = dia(R_ + 1), dia(R_ + 2) - 1
          call visit(R_, pass_, st, int(dja(k_ + 1)), int(k_) + 1, .false.)
        enddo
        if (has_o) then
          do k_ = oia(R_ + 1), oia(R_ + 2) - 1
            call visit(R_, pass_, st, comb(garr(oja(k_ + 1) + 1), st%chint), -(int(k_) + 1), .false.)
          enddo
        endif
      else
        do k_ = sia(R_ - B%nrow + 1), sia(R_ - B%nrow + 2) - 1
          call visit(R_, pass_, st, comb(sja(k_ + 1), st%chint), int(k_) + 1, .true.)
        enddo
      endif
      if (pass_ == 1) then
        cva(R_) = st%nva; cvg(R_) = st%nvg
        B%zp(R_ + 2) = st%kz
        if (st%kl > 0) then
          !$omp atomic
          B%kl(st%bb) = max(B%kl(st%bb), st%kl)
        endif
        if (st%ku > 0) then
          !$omp atomic
          B%ku(st%bb) = max(B%ku(st%bb), st%ku)
        endif
        if (st%bad) then
          !$omp atomic write
          bad(st%bb) = .true.
        endif
      endif
    end subroutine scan_row

    !> entry (R, C) with source src: into the zebra CSR, into the block, or
    !! nowhere
    subroutine visit(R_, pass_, st, C, src, ghost)
      integer, intent(in) :: R_, pass_, C, src
      type(row_scan_t), intent(inout) :: st
      logical, intent(in) :: ghost
      integer :: cb, pr, pc
      if (C < 0) return                           ! outside the rank's rows and ghosts
      cb = B%bid(C + 1)
      if (H%smoother == GMG_SM_ZEBRA_RINGS .and. B%bcol(st%bb) == 1 .and. B%bcol(cb) == 0) then
        if (pass_ == 2) B%zc(B%zp(R_ + 1) + st%kz) = C
        call put(pass_, st, src, -int(B%zp(R_ + 1) + st%kz, 8), ghost)
        st%kz = st%kz + 1
      else if (cb == st%bb) then
        pr = B%pos(R_ + 1); pc = B%pos(C + 1)
        if (pass_ == 1) then
          st%kl = max(st%kl, pr - pc)
          st%ku = max(st%ku, pc - pr)
          if (B%btd(st%bb) .and. .not. st%bad) then
            ! a ring solved as a block-tridiagonal chain: J +- 1 couplings only
            st%bad = btd_index(B%rg(st%bb), B%qn(B%off(st%bb) + pr), 1, B%qn(B%off(st%bb) + pc), 1) == 0
          endif
        endif
        if (.not. B%axblk(st%bb)) then
          if (pass_ == 1) then
            call put(pass_, st, src, 0_8, ghost)
          else
            call put(pass_, st, src, lu_index(B, st%bb, pr, pc), ghost)
          endif
        endif
      endif
    end subroutine visit

    subroutine put(pass_, st, src, dst, ghost)
      integer, intent(in)    :: pass_, src
      type(row_scan_t), intent(inout) :: st
      integer(8), intent(in) :: dst
      logical, intent(in)    :: ghost
      if (ghost) then
        st%nvg = st%nvg + 1
        if (pass_ == 2) then
          B%gs(st%nvg) = src; B%gd(st%nvg) = dst
        endif
      else
        st%nva = st%nva + 1
        if (pass_ == 2) then
          B%vs(st%nva) = src; B%vd(st%nva) = dst
        endif
      endif
    end subroutine put

    !> combined row index of global column gc, or -1. A row's ghost columns
    !! ascend in runs (a node's slots and C1 DOFs): the ghost after the last
    !! one found (chint) is tried first.
    integer function comb(gc, chint)
      PetscInt, intent(in) :: gc
      integer, intent(inout) :: chint
      integer :: lo, hi, mid
      comb = -1
      if (gc >= rst .and. gc < ren) then
        comb = int(gc - rst)
        return
      endif
      if (chint > 0 .and. chint < B%ngh) then
        if (B%gidx(chint + 1) == gc) then
          chint = chint + 1
          comb = B%nrow + chint - 1
          return
        endif
      endif
      lo = 1; hi = B%ngh
      do while (lo <= hi)
        mid = (lo + hi) / 2
        if (B%gidx(mid) == gc) then
          comb = B%nrow + mid - 1
          chint = mid
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
      if (B%loff(k) < 0) then
        B%lu4(B%loff4(k) + 1 : B%loff4(k) + blk_size(B, k)) = 0.0
      else
        B%lu(B%loff(k) + 1 : B%loff(k) + blk_size(B, k)) = 0.0d0
      endif
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
      if (d > B%nlu) then
        B%lu4(d - B%nlu) = real(v, rk_ring)
      else if (d > 0) then
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
      if (d > B%nlu) then
        B%lu4(d - B%nlu) = real(v, rk_ring)
      else if (d > 0) then
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

  !> Mesh part of the overlap (first build only): ghost rows that other ranks
  !! own, joined to the rank's blocks (restricted additive Schwarz). A ring
  !! block (smoother 4; 8 inside the switch ring) is completed to its whole
  !! ring and slot group, from one gather of all rows' (I, J, m) codes (above
  !! RING_GATHER_MAX rows by the walk below); a radial-line segment is extended
  !! by LINE_OVERLAP nodes along its line, found by a breadth-first walk of A's
  !! graph from the rank's off-diagonal columns, each step reading the
  !! candidates' codes from their owners. The ghosts join their blocks in line
  !! order (rings: the band order rkey), and the ghost scatter is built.
  !! Collective on H%comm.
  subroutine ovl_extend(g, A, B, rst, ren)
    use mod_petsc_pc_harm, only: pc_ntor
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    PetscInt, intent(in) :: rst, ren
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
    logical :: has_o, rings, lines
    integer :: nc, nr, r, bb, I, J, Jc, m, d, q, t, ngh, nlo, mpierr, nkey, qtot, nring
    integer :: gtot(2), gloc(2), dmax, ninc, ncd
    integer, allocatable :: Ia(:), Ib(:), blkr(:), blkl(:), gb(:), gI(:), nsz(:), noff(:), nrows(:), fill(:)
    integer, allocatable :: tb(:), tI(:), tJ(:), gJ(:), lst(:), rk(:), gst(:)
    logical, allocatable :: isring(:), seen(:)
    PetscInt, allocatable :: cand(:), gl(:), tl(:), perm(:)

    nr = B%nrow
    nc = level_nj(g)
    rings = (H%smoother /= GMG_SM_LINES)       ! this level's smoother has ring blocks
    lines = (H%smoother /= GMG_SM_RINGS)       ! ... and radial-line blocks
    ! the row codes carry the radial column, which on an X-point grid can
    ! outnumber the ring positions
    ncd = nc
    if (H%xgrid) then
      do r = 1, nr
        call node_of(r, I, J, Jc, m)
        ncd = max(ncd, Jc + 1)
      enddo
      call MPI_Allreduce(MPI_IN_PLACE, ncd, 1, MPI_INTEGER, MPI_MAX, H%comm, mpierr)
    endif
    ! the block of every (ring, slot) and (line, slot), and the local line
    ! segments' ends Ia..Ib (a block holds all slots of its slot group)
    nkey = 1
    do r = 1, nr
      call node_of(r, I, J, Jc, m)
      nkey = max(nkey, I + 1)
    enddo
    allocate(Ia(B%nb), Ib(B%nb), isring(B%nb), blkr(0:nkey * pc_ntor - 1), blkl(0:ncd * pc_ntor - 1))
    Ia = -1; Ib = -1; isring = .false.; blkr = 0; blkl = 0
    do bb = 1, B%nb
      if (B%axblk(bb)) cycle
      call node_of(B%rows(B%off(bb) + 1) + 1, I, J, Jc, m)
      isring(bb) = ring_block(g, I)
      Ia(bb) = I
      if (isring(bb)) Ia(bb) = huge(1)               ! ring ghosts all go before the own rows
      do q = 1, B%sz(bb)
        call node_of(B%rows(B%off(bb) + q) + 1, I, J, Jc, m)
        if (isring(bb)) then
          blkr(I * pc_ntor + m) = bb
        else
          Ib(bb) = max(Ib(bb), I)
          blkl(Jc * pc_ntor + m) = bb
        endif
      enddo
    enddo
    ! the code of every owned row, as a vector the walk can read remotely
    call MatCreateVecs(A, kv, PETSC_NULL_VEC, ierr)
    call VecGetArray(kv, kp, ierr)
    do r = 1, nr
      call node_of(r, I, J, Jc, m)
      kp(r) = dble(enc(I, J, Jc, m))
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
    ! the walk: LINE_OVERLAP steps for the lines; without the ring gather the
    ! rings' too, which reach the rest of a ring only along J, a node per step
    dmax = 0
    if (lines) dmax = LINE_OVERLAP
    if (rings) then
      if (.not. ring_gather()) dmax = max(dmax, nc)
    endif
    do d = 1, dmax
      ! drop the candidates already accepted
      allocate(tl(nn))
      ns = 0
      do k = 1, nn
        if (find(gl, cand(k)) > 0) cycle
        ns = ns + 1; tl(ns) = cand(k)
      enddo
      ! their codes from the owners
      call ISCreateGeneral(PETSC_COMM_SELF, ns, tl(1:ns), PETSC_COPY_VALUES, isc, ierr)
      call VecCreateSeq(PETSC_COMM_SELF, ns, cv, ierr)
      call VecScatterCreate(kv, isc, cv, PETSC_NULL_IS, sc, ierr)
      call VecScatterBegin(sc, kv, cv, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd(sc, kv, cv, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecGetArrayRead(cv, cp, ierr)
      allocate(tb(ns), tI(ns), tJ(ns))
      q = 0
      do k = 1, ns
        call dec(nint(cp(k), 8), I, J, Jc, m)
        if (I <= H%axis_k) cycle
        ! the same ring and slot as a local ring block, else within reach of a
        ! local line segment: LINE_OVERLAP nodes beyond its ends (on an X-point
        ! grid a column's ring indices jump at the separatrix, so the walk
        ! depth alone bounds the reach)
        bb = 0
        if (I < nkey) bb = blkr(I * pc_ntor + m)
        if (bb == 0 .and. d <= LINE_OVERLAP) then
          bb = blkl(Jc * pc_ntor + m)
          if (bb > 0 .and. .not. H%xgrid) then
            if (.not. ((I < Ia(bb) .and. Ia(bb) - I <= LINE_OVERLAP) .or. &
                       (I > Ib(bb) .and. I - Ib(bb) <= LINE_OVERLAP))) bb = 0
          endif
        endif
        if (bb == 0) cycle
        q = q + 1
        tl(q) = tl(k); tb(q) = bb; tI(q) = I; tJ(q) = J
      enddo
      call VecRestoreArrayRead(cv, cp, ierr)
      call VecScatterDestroy(sc, ierr); call VecDestroy(cv, ierr); call ISDestroy(isc, ierr)
      gl = [gl, tl(1:q)]; gb = [gb, tb(1:q)]; gI = [gI, tI(1:q)]; gJ = [gJ, tJ(1:q)]
      call sort_ghosts()
      ! no rank found a new ghost: the walk is complete (all ranks agree)
      call MPI_Allreduce(q, qtot, 1, MPI_INTEGER, MPI_SUM, H%comm, mpierr)
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
    ! the ghosts bucketed by block, ascending k within each (gst(bb)+1 ..)
    allocate(lst(ngh), gst(B%nb + 1))
    gst = 0
    do k = 1, ngh
      gst(gb(k) + 1) = gst(gb(k) + 1) + 1
    enddo
    gst(1) = 0
    do bb = 2, B%nb + 1
      gst(bb) = gst(bb) + gst(bb - 1)
    enddo
    fill = gst(1:B%nb)
    do k = 1, ngh
      fill(gb(k)) = fill(gb(k)) + 1
      lst(fill(gb(k))) = int(k)
    enddo
    do bb = 1, B%nb                                     ! ghosts of block bb, ascending k
      if (nsz(bb) == B%sz(bb)) cycle
      nlo = 0
      q = gst(bb + 1) - gst(bb)
      lst(1:q) = lst(gst(bb) + 1:gst(bb + 1))
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
    ! a completed ring in the band order (J = 0, nc-1, 1, nc-2, ...), own rows
    ! and ghosts together
    if (rings) then
      allocate(rk(nr + ngh))
      do r = 1, nr + ngh
        if (r <= nr) then
          call node_of(r, I, J, Jc, m)
        else
          I = gI(r - nr); J = gJ(r - nr)
        endif
        rk(r) = rkey(g, I, J, nc)
      enddo
      do bb = 1, B%nb
        if (.not. isring(bb) .or. nsz(bb) == B%sz(bb)) cycle
        call sort_rows_by_key(nrows(noff(bb) + 1 : noff(bb) + nsz(bb)), rk)
      enddo
      ! every ring block must now span its ring
      ninc = 0
      allocate(seen(0:nc - 1))
      do bb = 1, B%nb
        if (.not. isring(bb)) cycle
        seen = .false.
        nring = nc
        do q = 1, nsz(bb)
          r = nrows(noff(bb) + q) + 1
          if (r <= nr) then
            call node_of(r, I, J, Jc, m)
          else
            I = gI(r - nr); J = gJ(r - nr)
          endif
          if (H%xgrid) nring = H%xring(g)%nn(I)
          seen(J) = .true.
        enddo
        if (.not. all(seen(0:nring - 1))) ninc = ninc + 1
      enddo
      call MPI_Allreduce(MPI_IN_PLACE, ninc, 1, MPI_INTEGER, MPI_SUM, H%comm, mpierr)
      if (H%my_id == 0 .and. ninc > 0) write(*,'(A,I0,A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, " WARNING: ", &
        ninc, " ring block(s) on level ", g, " still cut by a rank boundary"
      deallocate(rk, seen)
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
    if (allocated(B%ghI)) deallocate(B%ghI)
    allocate(B%ghI(ngh))
    B%ghI = gI
    call ISCreateGeneral(PETSC_COMM_SELF, int(ngh, kind(nn)), gl, PETSC_COPY_VALUES, B%isg, ierr)
    call VecCreateSeq(PETSC_COMM_SELF, int(ngh, kind(nn)), B%xg, ierr)
    call VecScatterCreate(kv, B%isg, B%xg, PETSC_NULL_IS, B%sct, ierr)
    call VecDestroy(kv, ierr)
    B%ovl = .true.
    gloc = [ngh, nr]
    call MPI_Allreduce(gloc, gtot, 2, MPI_INTEGER, MPI_SUM, H%comm, mpierr)
    if (H%my_id == 0 .and. g == 0) write(*,'(A,I0,A,I0,A,F5.1,A)') "[Physics PC]   GMG", cur_inst, &
      " overlap (cut rings completed, radial lines extended by LINE_OVERLAP nodes): ", gtot(1), &
      " ghost rows on level 0 (", 100.d0 * gtot(1) / max(gtot(2), 1), "% of the rows)"

  contains

    !> own row r_: ring I_, position J_ along it, radial column Jc_, slot m_
    subroutine node_of(r_, I_, J_, Jc_, m_)
      integer, intent(in)  :: r_
      integer, intent(out) :: I_, J_, Jc_, m_
      call row_node(g, r_, I_, J_, m_)
      Jc_ = row_col(g, r_, J_)
    end subroutine node_of

    !> the walk's code of a row
    integer(8) function enc(I_, J_, Jc_, m_)
      integer, intent(in) :: I_, J_, Jc_, m_
      enc = ((int(I_, 8) * ncd + Jc_) * nc + J_) * pc_ntor + m_
    end function enc

    subroutine dec(c_, I_, J_, Jc_, m_)
      integer(8), intent(in) :: c_
      integer, intent(out)   :: I_, J_, Jc_, m_
      integer(8) :: t_
      m_ = int(mod(c_, int(pc_ntor, 8))); t_ = c_ / pc_ntor
      J_ = int(mod(t_, int(nc, 8)));      t_ = t_ / nc
      Jc_ = int(mod(t_, int(ncd, 8)));    I_ = int(t_ / ncd)
    end subroutine dec

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

    !> The ring ghosts from a gather of every rank's ring codes; .false. (the
    !! walk follows) above RING_GATHER_MAX rows.
    logical function ring_gather()
      integer :: np_, kk, ntot_
      integer, allocatable :: cnt_(:), dsp_(:), code_(:), grow_(:), myc(:), myg(:), bk(:)
      ring_gather = .false.
      call MPI_Comm_size(H%comm, np_, mpierr)
      allocate(cnt_(0:np_ - 1), dsp_(0:np_))
      call MPI_Allgather(nr, 1, MPI_INTEGER, cnt_, 1, MPI_INTEGER, H%comm, mpierr)
      dsp_(0) = 0
      do kk = 0, np_ - 1
        dsp_(kk + 1) = dsp_(kk) + cnt_(kk)
      enddo
      ntot_ = dsp_(np_)
      if (ntot_ > RING_GATHER_MAX) return
      allocate(myc(nr), myg(nr), code_(ntot_), grow_(ntot_), bk(ntot_))
      do r = 1, nr
        call node_of(r, I, J, Jc, m)
        myc(r) = (I * ncd + J) * pc_ntor + m
        myg(r) = int(rst) + r - 1
      enddo
      call MPI_Allgatherv(myc, nr, MPI_INTEGER, code_, cnt_, dsp_(0:np_ - 1), MPI_INTEGER, H%comm, mpierr)
      call MPI_Allgatherv(myg, nr, MPI_INTEGER, grow_, cnt_, dsp_(0:np_ - 1), MPI_INTEGER, H%comm, mpierr)
      ! bk: the local ring block (same ring and slot) of each off-rank row, or 0
      bk = 0
      do kk = 1, ntot_
        if (grow_(kk) >= rst .and. grow_(kk) < ren) cycle
        m = mod(code_(kk), pc_ntor); I = code_(kk) / pc_ntor / ncd
        if (I > H%axis_k .and. I < nkey) bk(kk) = blkr(I * pc_ntor + m)
      enddo
      code_ = pack(code_, bk > 0)
      gl = int(pack(grow_, bk > 0), kind(nn))
      gb = pack(bk, bk > 0)
      gI = code_ / pc_ntor / ncd
      gJ = mod(code_ / pc_ntor, ncd)
      call sort_ghosts()
      ring_gather = .true.
    end function ring_gather

    !> gl ascending, gb / gI / gJ permuted along
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

  !> The sector factorisations the level loop left pending: every level's
  !! sector ranks work at the same time (levels in ascending order on every
  !! rank, so the gathers cannot deadlock). A failed factorisation, and on the
  !! first build a failed gate, hands the level's axis block to its LU.
  subroutine axd_factor_pending()
    integer :: g
    PetscErrorCode :: ierr
    call PetscLogEventBegin(gev_axnum, ierr)
    do g = 0, H%nlev - 2
      if (H%blk(g)%axpend) call axd_factor(H%blk(g)%axd)
    enddo
    call PetscLogEventEnd(gev_axnum, ierr)
    do g = 0, H%nlev - 2
      if (H%blk(g)%axpend) call check(H%blk(g))
    enddo
    do g = 0, H%nlev - 2
      if (H%blk(g)%axnew) call finish(H%blk(g))
    enddo

  contains

    subroutine check(B)
      type(blk_t), intent(inout) :: B
      logical :: bad
      integer :: mpierr
      B%axpend = .false.
      bad = B%axd%failed
      call MPI_Allreduce(MPI_IN_PLACE, bad, 1, MPI_LOGICAL, MPI_LOR, H%comm, mpierr)
      if (.not. bad) return
      if (H%my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   GMG", cur_inst, " level ", g, &
        ": axis J-sector factorisation failed (zero pivot), LU used"
      call axd_destroy(B%axd)
      B%axdon = .false.
      B%axnew = .false.
      call ax_lu(g, B%axA, B)
    end subroutine check

    subroutine finish(B)
      type(blk_t), intent(inout) :: B
      B%axnew = .false.
      call axd_gate(g, B%axA, B)
      if (B%axdon) return
      call axd_destroy(B%axd)
      call ax_lu(g, B%axA, B)
    end subroutine finish
  end subroutine axd_factor_pending

  !> The LU of level g's axis block (rds_t), where no sector solve is in use.
  subroutine ax_lu(g, A, B)
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    character(len=64) :: axtag
    write(axtag, '(A,I0)') "axis block level ", g
    call rds_setup(B%axr, A, "axis", trim(axtag))
  end subroutine ax_lu

  !> First sector rank of level g: the levels' sector sets follow each other
  !! over the ranks (wrapping at np), so their factorisations overlap in time.
  integer function axd_offset(g, gnp)
    integer, intent(in) :: g, gnp
    integer :: gf, ns
    axd_offset = 0
    do gf = 0, g - 1
      ns = axd_nsec(ax_ring_len(gf), gnp)
      if (ns >= 2) axd_offset = axd_offset + ns
    enddo
    axd_offset = modulo(axd_offset, gnp)
  end function axd_offset

  !> First build: the structure of level g's J-sector axis solve (axd_setup,
  !! the cost model's sector count). Where it applies, its factorisation and
  !! gate are left pending (B%axnew); else the caller builds the LU.
  !! Collective on H%comm.
  subroutine axd_prepare(g, A, B)
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    integer :: q, I, m, nsec, gnp, mpierr, nc
    integer, allocatable :: jl(:)
    character(len=64) :: axtag, pre

    call MPI_Comm_size(H%comm, gnp, mpierr)
    nc = ax_ring_len(g)
    nsec = axd_nsec(nc, gnp)
    if (nsec < 2) return
    allocate(B%axd)
    allocate(jl(size(B%axr%loc)))
    do q = 1, size(jl)
      call row_node(g, B%axr%loc(q) + 1, I, jl(q), m)
    enddo
    write(axtag, '(A,I0,A,I0)') "GMG", cur_inst, " axis block level ", g
    write(pre, '(A,I0,A)') "sf_gmg", cur_inst, "_axis_"
    call axd_setup(B%axd, A, B%axr%loc, jl, nc, nsec, H%comm, trim(axtag), trim(pre), axd_offset(g, gnp))
    if (.not. B%axd%on) return
    B%axnew = .true.; B%axpend = .true.
    B%axA = A
  end subroutine axd_prepare

  !> First build, after the factorisations: the gate of level g's sector
  !! solve. For a random b on the axis rows, y = the sector solve, it is used
  !! if the componentwise backward error (Oettli-Prager) max_i |b - A y|_i /
  !! (|A| |y| + |b|)_i is at most GATE_TOL; invariant under the diagonal
  !! scalings of the C1 DOFs, unlike a normwise one. Collective on H%comm.
  subroutine axd_gate(g, A, B)
    integer, intent(in) :: g
    Mat, intent(in) :: A
    type(blk_t), intent(inout) :: B
    real*8, parameter :: GATE_TOL = 1.d-8
    integer :: q, c, t, nl, np, mpierr
    integer, allocatable :: cnt(:), dsp(:), gax(:)
    real*8, allocatable :: yl(:), yax(:)
    logical, allocatable :: ing(:)
    Vec :: x, y1, y2
    PetscScalar, pointer :: yp(:), xp(:), vals(:)
    PetscInt, pointer :: cols(:)
    PetscInt :: rst, ren, ncols
    real*8 :: dn, s
    PetscErrorCode :: ierr

    call MatGetOwnershipRange(A, rst, ren, ierr)
    call MatCreateVecs(A, y2, x, ierr)
    call VecDuplicate(x, y1, ierr)
    call VecSetRandom(x, PETSC_NULL_RANDOM, ierr)
    nl = size(B%axr%loc)
    allocate(ing(B%nrow))
    ing = .false.
    ing(B%axr%loc + 1) = .true.
    call VecGetArray(x, xp, ierr)
    do q = 1, B%nrow
      if (.not. ing(q)) xp(q) = 0.0d0
    enddo
    call VecRestoreArray(x, xp, ierr)
    call VecZeroEntries(y2, ierr)
    call axd_solve(B%axd, x, y2)
    call MatMult(A, y2, y1, ierr)
    ! y is zero off the axis rows: their global indices (ascending over the
    ! ranks) and values on every rank give |A| |y| on the local ones
    call MPI_Comm_size(H%comm, np, mpierr)
    allocate(cnt(0:np - 1), dsp(0:np), yl(nl))
    call MPI_Allgather(nl, 1, MPI_INTEGER, cnt, 1, MPI_INTEGER, H%comm, mpierr)
    dsp(0) = 0
    do q = 0, np - 1
      dsp(q + 1) = dsp(q) + cnt(q)
    enddo
    allocate(gax(dsp(np)), yax(dsp(np)))
    call VecGetArrayRead(y2, yp, ierr)
    yl = yp(B%axr%loc + 1)
    call VecRestoreArrayRead(y2, yp, ierr)
    call MPI_Allgatherv(int(rst) + B%axr%loc, nl, MPI_INTEGER, gax, cnt, dsp(0:np - 1), MPI_INTEGER, H%comm, mpierr)
    call MPI_Allgatherv(yl, nl, MPI_DOUBLE_PRECISION, yax, cnt, dsp(0:np - 1), MPI_DOUBLE_PRECISION, H%comm, mpierr)
    call VecGetArrayRead(y1, yp, ierr)
    call VecGetArrayRead(x, xp, ierr)
    dn = 0.0d0
    do q = 1, B%nrow
      if (.not. ing(q)) cycle
      call MatGetRow(A, rst + q - 1, ncols, cols, vals, ierr)
      s = 0.d0
      t = 0
      do c = 1, int(ncols)
        t = find_hint(gax, int(cols(c)), t)
        if (t > 0) s = s + abs(vals(c)) * abs(yax(t))
      enddo
      call MatRestoreRow(A, rst + q - 1, ncols, cols, vals, ierr)
      dn = max(dn, abs(xp(q) - yp(q)) / max(s + abs(xp(q)), 1.d-300))
    enddo
    call VecRestoreArrayRead(x, xp, ierr)
    call VecRestoreArrayRead(y1, yp, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, dn, 1, MPI_DOUBLE_PRECISION, MPI_MAX, H%comm, mpierr)
    call VecDestroy(x, ierr); call VecDestroy(y1, ierr); call VecDestroy(y2, ierr)
    B%axdon = (dn <= GATE_TOL)                     ! a NaN fails too
    if (H%my_id == 0) write(*,'(A,I0,A,I0,A,ES9.2,A)') "[Physics PC]   GMG", cur_inst, " axis block level ", g, &
      ": J-sector solve backward error ", dn, merge(" -> in use ", " -> LU used", B%axdon)

  contains

    !> position of x in the ascending list l, or 0; the position after hint first
    integer function find_hint(l, x, hint)
      integer, intent(in) :: l(:), x, hint
      integer :: lo, hi, mid
      find_hint = 0
      if (hint > 0 .and. hint < size(l)) then
        if (l(hint + 1) == x) then
          find_hint = hint + 1
          return
        endif
      endif
      lo = 1; hi = size(l)
      do while (lo <= hi)
        mid = (lo + hi) / 2
        if (l(mid) == x) then
          find_hint = mid
          return
        else if (l(mid) < x) then
          lo = mid + 1
        else
          hi = mid - 1
        endif
      enddo
    end function find_hint
  end subroutine axd_gate

  !> In-place ascending heapsort.
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

  !> Stable sort of the 0-based rows r(:) by key(r + 1) (bottom-up merge sort).
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
  !! storage (ldab = 2 kl + ku + 1, pivots ipiv): dgbtrs('N') written out, the
  !! same operations in the same order without a BLAS call per column.
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
      lu_index = btd_index(B%rg(bb), jr, i - B%rg(bb)%rp(jr), jc, j - B%rg(bb)%rp(jc))
      if (B%loff(bb) < 0) then
        lu_index = B%nlu + B%loff4(bb) + lu_index
      else
        lu_index = B%loff(bb) + lu_index
      endif
    else if (B%band(bb)) then
      lu_index = B%loff(bb) + int(j - 1, 8) * (2 * B%kl(bb) + B%ku(bb) + 1) &
                 + (B%kl(bb) + B%ku(bb) + 1 + i - j)
    else
      lu_index = B%loff(bb) + int(j - 1, 8) * B%sz(bb) + i
    endif
  end function lu_index

  !> Block id (with gaps) of grid point (I, radial column J) of level g, slot
  !! m_: the axis block, a ring or a radial line, per slot group (harm_pair 1:
  !! n = 0, then each harmonic's cos and sin; 2: all slots).
  integer function blk_id(g, I, J, m_)
    use mod_petsc_pc_harm, only: pc_ntor, pc_grp
    integer, intent(in) :: g, I, J, m_
    integer :: m
    m = pc_grp(m_)
    if (H%harm_pair == 2) m = 0
    if (I <= H%axis_k) then
      blk_id = m + 1
    else if (ring_block(g, I)) then
      blk_id = pc_ntor + (I - 1) * pc_ntor + m + 1
    else if (H%smoother == GMG_SM_LINES) then
      blk_id = pc_ntor + J * pc_ntor + m + 1
    else
      blk_id = pc_ntor + max(ring_lim(g) - 1, 0) * pc_ntor + J * pc_ntor + m + 1
    endif
  end function blk_id

  !> Do the blocks of ring I > axis_k of level g hold rings (else radial lines)?
  logical function ring_block(g, I)
    integer, intent(in) :: g, I
    ring_block = (H%smoother == GMG_SM_RINGS .or. (H%smoother == GMG_SM_ZEBRA_RINGS .and. I < ring_lim(g)))
  end function ring_block

  !> Level g's nj: the positions along its rings (X-point grids: the longest ring)
  integer function level_nj(g)
    integer, intent(in) :: g
    if (g == 0) then
      level_nj = H%nth0
    else
      level_nj = H%lvl(g)%nj
    endif
  end function level_nj

  !> Own row r (1-based) of level g: ring I, position J along it, toroidal slot m
  subroutine row_node(g, r, I, J, m)
    integer, intent(in)  :: g, r
    integer, intent(out) :: I, J, m
    if (g == 0) then
      I = H%fine_node(r) / H%nth0; J = mod(H%fine_node(r), H%nth0); m = H%fine_harm(r)
    else
      I = H%lvl(g)%rnode(r) / H%lvl(g)%nj; J = mod(H%lvl(g)%rnode(r), H%lvl(g)%nj); m = H%lvl(g)%rharm(r)
    endif
  end subroutine row_node

  !> The block-solve work arrays hold at least n rows per thread.
  subroutine work_reserve(n)
    integer, intent(in) :: n
    integer :: nthr, n0, t0
    nthr = 1
    !$ nthr = omp_get_max_threads()
    n0 = 0; t0 = 0
    if (allocated(wbuf)) then
      n0 = size(wbuf, 1); t0 = size(wbuf, 2)
      if (n0 >= n .and. t0 >= nthr) return
      deallocate(wbuf, wbuf4)
    endif
    allocate(wbuf(max(n, n0), 0:max(nthr, t0) - 1), wbuf4(max(n, n0), 0:max(nthr, t0) - 1))
  end subroutine work_reserve

  !> Radial column of local row r (1-based) of level g: its J on the polar
  !! grid, the virtual lattice's column on an X-point grid.
  integer function row_col(g, r, J)
    integer, intent(in) :: g, r, J
    row_col = J
    if (.not. H%xgrid) return
    if (g == 0) then
      row_col = H%fine_col(r)
    else
      row_col = H%lvl(g)%rcol(r)
    endif
  end function row_col

  !> Order key of position J on ring I of level g (nc: its nj): the band
  !! order of a closed ring (ring_key), the natural order on an X-point
  !! grid's open lines.
  integer function rkey(g, I, J, nc)
    integer, intent(in) :: g, I, J, nc
    rkey = ring_key(J, nc)
    if (.not. H%xgrid) return
    rkey = J
    if (H%xring(g)%per(I)) rkey = ring_key(J, H%xring(g)%nn(I))
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

  !> Switch ring of level g: coarse ring I sits at fine ring I*2^g, the
  !! same physical radius as the fine ring_is.
  integer function ring_lim(g)
    integer, intent(in) :: g
    ring_lim = (H%ring_is + 2**g - 1) / 2**g
  end function ring_lim

  !> The J period of level g's axis sectors: nj; on an X-point grid the axis
  !! rings' length if rings 1..axis_k are closed core rings of one length (and
  !! ring 0 has it or is a single position), else 0 (no sectors).
  integer function ax_ring_len(g)
    integer, intent(in) :: g
    integer :: I
    if (.not. H%xgrid) then
      ax_ring_len = level_nj(g)
      return
    endif
    ax_ring_len = 0
    if (size(H%xring(g)%nn) < 2 .or. H%axis_k < 1) return
    ax_ring_len = H%xring(g)%nn(1)
    do I = 1, min(H%axis_k, size(H%xring(g)%nn) - 1)
      if (.not. H%xring(g)%per(I) .or. H%xring(g)%nn(I) /= ax_ring_len) ax_ring_len = 0
    enddo
    if (H%xring(g)%nn(0) /= ax_ring_len .and. H%xring(g)%nn(0) /= 1) ax_ring_len = 0
  end function ax_ring_len

  !> PC-shell apply of the smoothers: y = the block solve of x on level
  !! H%sm_lev of the active hierarchy (smooth sets both).
  subroutine blk_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    type(blk_t), pointer :: B

    B => H%blk(H%sm_lev)
    if (H%smoother == GMG_SM_ZEBRA_RINGS) then
      call zebra_solve(x, y)
    else
      ! axis first: it depends on x only, and the ranks enter here in step,
      ! which a J-sector solve needs
      call ax_solve(x, y)
      if (allocated(B%wk)) then
        call lines_solve_par(x, y)
      else
        call lines_solve(x, y)
      endif
    endif
    ierr = 0

  contains

    !> w = block bb's solve of w (block order) on thread t: LAPACK band /
    !! dense, or the ring solver (point Jacobi for a singular node block)
    subroutine one_solve(bb, w, t)
      integer, intent(in)   :: bb, t
      real*8, intent(inout) :: w(*)
      integer :: n, q, info
      external :: dgetrs
      n = B%sz(bb)
      if (B%btd(bb)) then
        if (B%jac(bb)) then
          do q = 1, n
            w(q) = w(q) / B%lu4(lu_index(B, bb, q, q) - B%nlu)
          enddo
        else
          wbuf4(1:n, t) = real(w(1:n) * B%eqs(B%off(bb) + 1 : B%off(bb) + n), rk_ring)
          call btd_solve_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%piv(B%off(bb) + 1), wbuf4(1, t))
          w(1:n) = wbuf4(1:n, t) * B%eqs(B%off(bb) + 1 : B%off(bb) + n)
        endif
      else if (B%band(bb)) then
        call band_solve(n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), B%piv(B%off(bb) + 1), w)
      else
        call dgetrs('N', n, 1, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), w, n, info)
      endif
    end subroutine one_solve

    !> lines_solve with the rings' hybrid segments threaded across all the
    !! rank's rings (work items B%lb, B%lk) on B%wk4: forward sweeps, the
    !! separators, backward sweeps. One parallel region, three barriers.
    subroutine lines_solve_par(xx, yy)
      Vec :: xx, yy
      PetscScalar, pointer :: xp(:), yp(:), gp(:)
      PetscErrorCode :: ie
      integer :: bb, k, p, t
      ! B%rg(bb) is spelled out: an associate inside these OpenMP loops
      ! crashes ifx (internal compiler error)
      call PetscLogEventBegin(gev_lines(cur_inst), ie)
      call ghosts_in(xx, gp)
      call VecGetArrayRead(xx, xp, ie)
      call VecGetArray(yy, yp, ie)
      !$omp parallel private(bb, k, p, t) if (B%nrow >= LINES_OMP_MIN)
      t = 0
      !$ t = omp_get_thread_num()
      !$omp do schedule(dynamic, 1)
      do p = 1, size(B%lb)
        bb = B%lb(p); k = B%lk(p)
        if (.not. reduced(bb)) then
          if (k > 0) cycle
          call gath(bb, 1, B%sz(bb), .false., xp, gp)
          call one_solve(bb, B%wk(B%off(bb) + 1), t)
          call scat(bb, 1, B%sz(bb), .false., yp)
          cycle
        endif
        call gath(bb, B%rg(bb)%rp(B%rg(bb)%sep(k)) + 1, B%rg(bb)%rp(hyb_end(B%rg(bb), k) + 1), .true., xp, gp)
        call hyb_sol_fwd_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%piv(B%off(bb) + 1), B%wk4(B%off(bb) + 1), k)
      enddo
      !$omp end do
      !$omp do schedule(dynamic, 1)
      do bb = 1, B%nb
        if (.not. reduced(bb)) cycle
        call hyb_sol_sep_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%piv(B%off(bb) + 1), B%wk4(B%off(bb) + 1))
        do k = 0, B%rg(bb)%ns - 1
          call scat(bb, B%rg(bb)%rp(B%rg(bb)%sep(k)) + 1, B%rg(bb)%rp(B%rg(bb)%sep(k) + 1), .true., yp)
        enddo
      enddo
      !$omp end do
      !$omp do schedule(dynamic, 1)
      do p = 1, size(B%lb)
        bb = B%lb(p); k = B%lk(p)
        if (.not. reduced(bb)) cycle
        call hyb_sol_bwd_sp(B%rg(bb), B%lu4(B%loff4(bb) + 1), B%wk4(B%off(bb) + 1), k)
        call scat(bb, B%rg(bb)%rp(B%rg(bb)%sep(k) + 1) + 1, B%rg(bb)%rp(hyb_end(B%rg(bb), k) + 1), .true., yp)
      enddo
      !$omp end do
      !$omp end parallel
      call VecRestoreArrayRead(xx, xp, ie)
      call VecRestoreArray(yy, yp, ie)
      call ghosts_done(gp)
      call PetscLogEventEnd(gev_lines(cur_inst), ie)
    end subroutine lines_solve_par

    !> lines_solve_par: block bb's positions q1..q2 of x (owned rows xp,
    !! ghosts gp) -> B%wk, or equilibrated to B%wk4 (sp)
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
          B%wk4(B%off(bb) + q) = real(v * B%eqs(B%off(bb) + q), rk_ring)
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

    !> y(rows outside the axis block) = the block solves of x, on the rank's
    !! threads; the ghost rows' input comes from the neighbours (one scatter),
    !! only the rank's own rows are written back.
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
            wbuf(q, t) = xp(r + 1)
          else
            wbuf(q, t) = gp(r - nr + 1)
          endif
        enddo
        call one_solve(bb, wbuf(1, t), t)
        do q = 1, n
          r = B%rows(B%off(bb) + q)
          if (r < nr) yp(r + 1) = wbuf(q, t)
        enddo
      enddo
      !$omp end parallel do
      call VecRestoreArrayRead(xx, xp, ie)
      call VecRestoreArray(yy, yp, ie)
      call ghosts_done(gp)
      call PetscLogEventEnd(gev_lines(cur_inst), ie)
    end subroutine lines_solve

    !> Smoother 8: one forward block Gauss-Seidel sweep over two colours
    !! (zebra relaxation, Trottenberg, Oosterlee & Schuller, Multigrid (2001)
    !! S5.1): colour 0 (the axis block, even rings, even radial lines) from x,
    !! colour 1 from x - A(colour 1, colour 0) y. Rings couple to I +- 1 and
    !! lines to J +- 1 only, so the sweep is exact block Gauss-Seidel on the
    !! rank's rows. The colour-0 solutions on the ghost rows are kept (yg) for
    !! the colour-1 blocks' ghosts: no second communication.
    subroutine zebra_solve(xx, yy)
      Vec :: xx, yy
      PetscScalar, pointer :: xp(:), yp(:), gp(:)
      PetscErrorCode :: ie
      integer :: bb, q, n, t, c, k, r, nr, zc_
      nr = B%nrow
      ! the axis block (colour 0) depends on x only: first, while the ranks are in step
      call ax_solve(xx, yy)
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
              wbuf(q, t) = xp(r + 1)
            else
              wbuf(q, t) = gp(r - nr + 1)
            endif
            ! colour 1: minus the couplings to the colour-0 solution
            do k = B%zp(r + 1), B%zp(r + 2) - 1
              zc_ = B%zc(k)
              if (zc_ < nr) then
                wbuf(q, t) = wbuf(q, t) - B%zv(k) * yp(zc_ + 1)
              else
                wbuf(q, t) = wbuf(q, t) - B%zv(k) * B%yg(zc_ - nr + 1)
              endif
            enddo
          enddo
          call one_solve(bb, wbuf(1, t), t)
          do q = 1, n
            r = B%rows(B%off(bb) + q)
            if (r < nr) then
              yp(r + 1) = wbuf(q, t)
            else if (c == 0) then
              B%yg(r - nr + 1) = wbuf(q, t)
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

    !> y(axis rows) = the axis-block solve of x(axis rows): the J-sector solve
    !! (collective: every rank enters its scatters) or the LU on its members
    subroutine ax_solve(xx, yy)
      Vec :: xx, yy
      PetscErrorCode :: ie
      PetscScalar, pointer :: xp(:), yp(:)
      if (B%axdon) then
        call PetscLogEventBegin(gev_axsolve(cur_inst), ie)
        call axd_solve(B%axd, xx, yy)
        call PetscLogEventEnd(gev_axsolve(cur_inst), ie)
        return
      endif
      if (.not. B%axr%member) return
      call PetscLogEventBegin(gev_axsolve(cur_inst), ie)
      call VecGetArrayRead(xx, xp, ie)
      call VecGetArray(yy, yp, ie)
      call rds_solve_arr(B%axr, xp, yp)
      call VecRestoreArray(yy, yp, ie)
      call VecRestoreArrayRead(xx, xp, ie)
      call PetscLogEventEnd(gev_axsolve(cur_inst), ie)
    end subroutine ax_solve

  end subroutine blk_apply

#endif
end module mod_petsc_pc_gmg
