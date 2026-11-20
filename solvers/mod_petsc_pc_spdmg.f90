!> A ring-free multigrid for a scalar symmetric positive definite operator on
!! JOREK's polar flux-surface grid: the P solver of vs_lumped
!! (mod_petsc_pc_vs, -vs_p_solver spdmg), from the study
!! studies/spd_multigrid (Python: spdmg/hierarchy.py prolongation_kinds and
!! level_shapes mode "theta", spdmg/multigrid.py). PETSc's PCMG with
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
!! PETSc options (defaults = the study's cycle):
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
!!   -vs_pmg_transfer mixed | ringharm | linear | const   the transfer along
!!                                         the ring                       (mixed)
!!   -vs_pmg_rh_sweeps <k>                 ring-harmonic Jacobi steps     (4)
!!   -vs_pmg_rh_reach <r>                  its reach along the ring       (3)
!!   -vs_pmg_rh_omega <w>                  its damping                    (1)
!!   -vs_pmg_levels <n>                    at most n levels               (all)
!!   -vs_pmg_min_nj <n>                    fewest ring positions          (3)
!!   -vs_pmg_dense_max <n>                 largest block with a dense LU  (300)
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
!! mod_petsc_pc_gmg); one hierarchy (one operator) per run; the operator's
!! pattern must stay that of the first build (refills only). The Chebyshev
!! interval is estimated on the whole operator: with several toroidal families
!! the interval is the union's, not each family's (on m1, n = 0 then gets
!! kappa 1.5 to 1.6 where its own interval gives 1.3).
module mod_petsc_pc_spdmg
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: spdmg_ksp_setup

  integer, parameter :: MAXL = 12
  integer, parameter :: BL_NODE = 1, BL_LINE = 2, BL_POINT = 3, BL_LINEWEDGE = 4
  integer, parameter :: TR_LINEAR = 1, TR_CONST = 2, TR_RINGHARM = 3, TR_MIXED = 4

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
    integer :: transfer   = TR_MIXED
    integer :: rh_sweeps  = 4          !< ring-harmonic transfer: Jacobi steps
    integer :: rh_reach   = 3          !< and the reach of X along the ring
    real*8  :: rh_omega   = 1.d0       !< and the step's damping
    integer :: max_levels = MAXL
    integer :: min_nj     = 3
    integer :: dense_max  = 300
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
    integer(8), allocatable :: loff(:)             !< dense block -> offset of its LU in lu
    real*8, allocatable :: lu(:)
    integer, allocatable :: piv(:)                 !< pivots, at off
    KSP, allocatable :: kb(:)                      !< sparse block -> its LU
    Mat, allocatable :: mb(:)                      !< its matrix
    IS,  allocatable :: ib(:)                      !< its rows in the union
    Vec, allocatable :: vb(:), vx(:)               !< its right-hand side and solution
    integer :: nsparse = 0
  end type asm_t

  type(spdmg_opts_t), save :: o
  logical, save :: built = .false.
  integer, save :: nlev = 0, comm_h = MPI_COMM_NULL, me = 0, np_h = 1, ns_h = 1
  type(lev_t), save :: lv(0:MAXL-1)
  Mat, save :: P(1:MAXL-1)                          !< P(g): level g -> level g-1
  Mat, save :: Ag(1:MAXL-1)                         !< the Galerkin operators P(g)' A(g-1) P(g)
  type(rh_t), save :: rh(1:MAXL-1)
  type(asm_t), save :: as(0:MAXL-1)
  PC, save :: as_pc(0:MAXL-1)                       !< the shell PC of level g's smoother
  real*8, allocatable, save :: hist(:)              !< the test's residual history
  real*8, allocatable, save :: ring_asp(:)          !< (0:n_flux-1) the finest level's ring cell aspect (wedge_last)
  logical, allocatable, save :: rh_slot(:)          !< (0:ns-1) the slot's F rows are ring-harmonic

contains

  !> Configure ksp as the P solver of operator A (one field, ns slots per
  !! JOREK index, rows index * ns + slot; mod_petsc_pc_sf_ops). fresh: ksp
  !! was just created (the hierarchy is built on the first call of the run);
  !! later calls take the refilled A (same pattern). Collective on comm.
  subroutine spdmg_ksp_setup(ksp, A, ns, comm, my_id, rtol, maxits, label, fresh)
    KSP, intent(inout)           :: ksp
    Mat, intent(in)              :: A
    integer, intent(in)          :: ns, comm, my_id, maxits
    real*8, intent(in)           :: rtol
    character(len=*), intent(in) :: label
    logical, intent(in)          :: fresh
    PC :: pc
    PetscErrorCode :: ierr
    logical :: first_chain

    first_chain = .not. built
    if (.not. built) then
      comm_h = comm; ns_h = ns
      call MPI_Comm_rank(comm, me, ierr)
      call MPI_Comm_size(comm, np_h, ierr)
      call read_options()
      call build_levels(A, my_id)
      call create_prolongations()
      built = .true.
    endif
    ! the levels' transfers (ring-harmonic: from the operators) and Galerkin operators
    call galerkin_chain(A, first_chain)
    if (fresh) then
      call KSPSetType(ksp, KSPFGMRES, ierr)
      call KSPGMRESSetRestart(ksp, maxits, ierr)
      call KSPSetTolerances(ksp, rtol, 1.d-50, 1.d6, maxits, ierr)
      call KSPSetOptionsPrefix(ksp, "vs_pmg_", ierr)
      call KSPGetPC(ksp, pc, ierr)
      call configure_pcmg(pc, A)
      call KSPSetOperators(ksp, A, A, ierr)
      call KSPSetFromOptions(ksp, ierr)
    else
      call KSPSetOperators(ksp, A, A, ierr)
    endif
    call KSPSetUp(ksp, ierr)
    if (fresh) then
      call report(ksp, tagged(), my_id)
      if (len_trim(o%view_p) > 0) call view_transfers()
      if (o%test) call run_test(ksp, A, tagged(), my_id)
    endif
  contains
    !> the label with the operator's global slots where it holds a part of them (mode split)
    function tagged() result(s)
      use mod_parameters,    only: n_tor
      use mod_petsc_pc_harm, only: pc_global_slot
      character(len=64) :: s
      s = label
      if (ns_h /= n_tor) write(s, '(A,A,I0,A,I0)') trim(label), ", slots ", pc_global_slot(0), "-", &
                                                    pc_global_slot(ns_h - 1)
    end function tagged
  end subroutine spdmg_ksp_setup

  !> P(g) in PETSc's binary format to <view_p>_P<g>.petsc (rows of level
  !! g-1, columns of level g, the row numbering of lv). Collective.
  subroutine view_transfers()
    PetscViewer :: vw
    PetscErrorCode :: ierr
    integer :: g
    character(len=300) :: fn
    do g = 1, nlev - 1
      write(fn, '(A,A,I0,A)') trim(o%view_p), "_P", g, ".petsc"
      call PetscViewerBinaryOpen(comm_h, trim(fn), FILE_MODE_WRITE, vw, ierr)
      call MatView(P(g), vw, ierr)
      call PetscViewerDestroy(vw, ierr)
    enddo
    if (me == 0) write(*,'(A)') "[Physics PC]   SPD-MG transfers written to "//trim(o%view_p)//"_P<g>.petsc"
  end subroutine view_transfers

  !> The options of the module header.
  subroutine read_options()
    PetscErrorCode :: ierr
    PetscBool :: is_set, flag
    PetscInt :: iv
    PetscReal :: rv
    character(len=16) :: s
    s = "node"
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_blocks", s, is_set, ierr)
    select case (trim(s))
    case ("node");      o%blocks = BL_NODE
    case ("line");      o%blocks = BL_LINE
    case ("point");     o%blocks = BL_POINT
    case ("linewedge"); o%blocks = BL_LINEWEDGE
    case default;   call fatal("-vs_pmg_blocks must be node | line | point | linewedge, got '"//trim(s)//"'")
    end select
    call get_int("-vs_pmg_line_from", o%line_from)
    call get_int("-vs_pmg_wedge_lines", o%wedge_lines)
    call get_int("-vs_pmg_wedge_stride", o%wedge_stride)
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_wedge_aspect", rv, is_set, ierr)
    if (is_set) o%wedge_aspect = rv
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_wedge_weight", rv, is_set, ierr)
    if (is_set) o%wedge_weight = rv
    if (o%wedge_lines < 1 .or. o%wedge_stride < 1) call fatal("-vs_pmg_wedge_lines, -vs_pmg_wedge_stride must be >= 1")
    s = "mixed"
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_transfer", s, is_set, ierr)
    select case (trim(s))
    case ("linear");   o%transfer = TR_LINEAR
    case ("const");    o%transfer = TR_CONST
    case ("ringharm"); o%transfer = TR_RINGHARM
    case ("mixed");    o%transfer = TR_MIXED
    case default;      call fatal("-vs_pmg_transfer must be mixed | ringharm | linear | const, got '"//trim(s)//"'")
    end select
    call get_int("-vs_pmg_rh_sweeps", o%rh_sweeps)
    call get_int("-vs_pmg_rh_reach", o%rh_reach)
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_rh_omega", rv, is_set, ierr)
    if (is_set) o%rh_omega = rv
    if (o%rh_sweeps < 1 .or. o%rh_reach < 1) call fatal("-vs_pmg_rh_sweeps and -vs_pmg_rh_reach must be >= 1")
    call get_int("-vs_pmg_axis_rings", o%axis_rings)
    call get_int("-vs_pmg_degree", o%degree)
    call get_int("-vs_pmg_esteig_its", o%esteig_its)
    call get_int("-vs_pmg_levels", o%max_levels)
    call get_int("-vs_pmg_min_nj", o%min_nj)
    call get_int("-vs_pmg_dense_max", o%dense_max)
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_cheb_ratio", rv, is_set, ierr)
    if (is_set) o%ratio = rv
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_cheb_safety", rv, is_set, ierr)
    if (is_set) o%safety = rv
    call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_test", flag, is_set, ierr)
    if (is_set) o%test = flag
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-vs_pmg_view_p", o%view_p, is_set, ierr)
    if (.not. is_set) o%view_p = ""
    o%max_levels = min(max(o%max_levels, 1), MAXL)
    if (o%axis_rings < 0) call fatal("-vs_pmg_axis_rings must be >= 0")
    if (o%degree < 1 .or. o%esteig_its < 1 .or. o%ratio <= 1.d0 .or. o%safety <= 0.d0) &
      call fatal("-vs_pmg_degree, -vs_pmg_esteig_its >= 1, -vs_pmg_cheb_ratio > 1, -vs_pmg_cheb_safety > 0")
  contains
    subroutine get_int(nm, v)
      character(len=*), intent(in) :: nm
      integer, intent(inout) :: v
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, nm, iv, is_set, ierr)
      if (is_set) v = int(iv)
    end subroutine get_int
  end subroutine read_options

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
    allocate(ring_asp(0:n_flux - 1))
    ring_asp = 0.d0
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
          ring_asp(i) = cell((n_tht + 1) / 2)
        else
          ring_asp(i) = 0.5d0 * (cell(n_tht / 2) + cell(n_tht / 2 + 1))
        endif
      enddo
      ring_asp(n_flux - 1) = ring_asp(n_flux - 2)
    end block

    ! level 0: the operator's index space and its row layout
    call MatGetOwnershipRange(A, rs, re, ierr)
    call MatGetSize(A, ng, PETSC_NULL_INTEGER, ierr)
    if (mod(int(rs), ns_h) /= 0 .or. mod(int(re), ns_h) /= 0 .or. mod(int(ng), ns_h) /= 0) &
      call fatal("the operator's rows are not whole (index, slots) groups")
    n_idx = int(ng) / ns_h
    allocate(i0s(0:np_h))
    call MPI_Allgather(int(rs) / ns_h, 1, MPI_INTEGER, i0s, 1, MPI_INTEGER, comm_h, mpierr)
    i0s(np_h) = n_idx
    ! the free indices: a row that is not a Dirichlet row on some slot
    call penalty_row_mask(A, mask)
    allocate(fl(0:n_idx - 1)); fl = 0
    call VecGetArray(mask, mp, ierr)
    do i = int(rs) / ns_h, int(re) / ns_h - 1
      if (any(mp((i - int(rs) / ns_h) * ns_h + 1 : (i - int(rs) / ns_h + 1) * ns_h) > 0.5d0)) fl(i) = 1
    enddo
    call VecRestoreArray(mask, mp, ierr)
    call VecDestroy(mask, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, fl, n_idx, MPI_INTEGER, MPI_MAX, comm_h, mpierr)
    allocate(freeb(0:n_idx - 1)); freeb = fl > 0
    nfree = count(freeb)

    ni = n_flux; nj = n_tht
    associate (L => lv(0))
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
        L%row0(f) = int(f, kind(rs)) * ns_h
      enddo
      L%nloc = re - rs
    end associate

    ! coarse levels: J halved, a DOF where the finer level's DOF at (I, 2J, k) has a coarse part
    nlev = 1
    do g = 1, o%max_levels - 1
      if (mod(lv(g - 1)%nj, 2) /= 0 .or. lv(g - 1)%nj / 2 < o%min_nj) exit
      nj = lv(g - 1)%nj / 2
      associate (L => lv(g), LF => lv(g - 1))
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
        allocate(cnt(0:np_h))
        cnt = 0
        do f = 0, c - 1
          cnt(L%own(f) + 1) = cnt(L%own(f) + 1) + 1
        enddo
        do r = 1, np_h
          cnt(r) = cnt(r) + cnt(r - 1)                   ! cnt(r): DOFs of the ranks below r
        enddo
        do f = 0, c - 1
          L%row0(f) = int(cnt(L%own(f)), kind(rs)) * ns_h
          cnt(L%own(f)) = cnt(L%own(f)) + 1
        enddo
        deallocate(cnt)
        L%nloc = int(count(L%own == me), kind(rs)) * ns_h
      end associate
      nlev = nlev + 1
    enddo
    if (nlev < 2) call fatal("the grid has no coarse level along the ring (n_tht must be even, n_tht/2 >= min_nj)")
    if (me == 0) then                                     ! the first rank of the operator's communicator
      write(*,'(A,I0,A,I0,A,I0,A)', advance="no") "[Physics PC]   SPD-MG: ", nlev, " levels along the ring, ", &
        nfree, " of ", n_idx, " indices free; lattice "
      do g = 0, nlev - 1
        write(*,'(I0,A,I0,A)', advance="no") n_flux, "x", lv(g)%nj, merge(" -> ", "    ", g < nlev - 1)
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
    allocate(rh_slot(0:ns_h - 1))
    do g = 0, ns_h - 1
      rh_slot(g) = o%transfer == TR_RINGHARM .or. (o%transfer == TR_MIXED .and. pc_global_slot(g) /= 0)
    enddo
    cap = 2
    if (any(rh_slot)) cap = max(2, 2 * ((o%rh_reach + 1) / 2) * 4 * ns_h)
    do g = 1, nlev - 1
      nl_r = lv(g - 1)%nloc; nl_c = lv(g)%nloc
      allocate(dnz(nl_r), onz(nl_r))
      dnz = min(int(cap, 8), int(nl_c, 8)); onz = min(int(cap, 8), int(lv(g)%n, 8) * ns_h - int(nl_c, 8))
      ! MPIAIJ also on one rank, as the operators: their products need it
      call MatCreate(comm_h, P(g), ierr)
      call MatSetSizes(P(g), nl_r, nl_c, PETSC_DETERMINE, PETSC_DETERMINE, ierr)
      call MatSetType(P(g), MATMPIAIJ, ierr)
      call MatMPIAIJSetPreallocation(P(g), zero, dnz, zero, onz, ierr)
      deallocate(dnz, onz)
      call MatSetOption(P(g), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
      if (.not. any(rh_slot)) call fill_p(g)
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
    do g = 1, nlev - 1
      if (any(rh_slot)) call fill_p(g, af)
      if (first) then
        call MatPtAP(af, P(g), MAT_INITIAL_MATRIX, 2.0d0, Ag(g), ierr)
      else
        call MatPtAP(af, P(g), MAT_REUSE_MATRIX, 2.0d0, Ag(g), ierr)
      endif
      af = Ag(g)
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
    associate (LF => lv(g - 1), LC => lv(g))
      njc = LC%nj
      do f = 0, LF%n - 1
        if (LF%own(f) /= me) cycle
        if (.not. LF%ok(f)) cycle
        i = LF%dI(f); j = LF%dJ(f); k = LF%dK(f)
        rhrow = i >= 1 .and. mod(j, 2) == 1                 ! an F node: ringharm_rows on the rh_slot slots
        nc = 0
        if (i == 0 .and. k == 0) then
          call add(LC%d(0, 0, 0), 1.d0)
        else if (mod(j, 2) == 0) then
          call add(LC%d(i, j / 2, k), 1.d0)
        else if (o%transfer == TR_CONST) then
          call add(LC%d(i, (j - 1) / 2, k), 1.d0)
        else
          call add(LC%d(i, (j - 1) / 2, k), 0.5d0)
          call add(LC%d(i, mod((j + 1) / 2, njc), k), 0.5d0)
        endif
        if (nc == 0) cycle
        tot = sum(w(1:nc))
        do s = 0, ns_h - 1
          if (rhrow .and. rh_slot(s)) cycle
          row = LF%row0(f) + s
          ncol = int(nc, kind(row))
          do q = 1, nc
            cols(q) = LC%row0(cc(q)) + s
            vals(q) = LF%osgn(f) * w(q) / tot
          enddo
          call MatSetValues(P(g), one, [row], ncol, cols(1:nc), vals(1:nc), INSERT_VALUES, ierr)
        enddo
      enddo
    end associate
    if (present(af)) call ringharm_rows(g, af)
    call MatAssemblyBegin(P(g), MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(P(g), MAT_FINAL_ASSEMBLY, ierr)
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
    use phys_module, only: n_flux
    integer, intent(in) :: g
    Mat, intent(in) :: af
    external :: dgetrf, dgetri
    integer :: nj, ni, m, nF, i, j, k, s, f, ff, q, qq, c, jc, dd, t, info, nnbmax, ncmax, a, ll, kc, sc, nent
    integer :: jdist, nb_
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
    associate (LF => lv(g - 1), LC => lv(g), H => rh(g))
      nj = LF%nj; ni = n_flux; m = 4 * ns_h; nF = nj / 2
      ! the rings with an F row of mine, and their gathered rows (F nodes) and columns (all nodes)
      if (.not. H%ready) then
        allocate(myring(0:ni - 1)); myring = .false.
        do i = 1, ni - 1
          do j = 1, nj - 1, 2
            do k = 0, 3
              dd = LF%d(i, j, k)
              if (dd < 0) cycle
              if (LF%own(dd) == me) myring(i) = .true.
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
                do s = 0, ns_h - 1
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
        H%rrow = rlist; H%rcol = clist
        nr_ = int(size(rlist), kind(one)); nc_ = int(size(clist), kind(one))
        call ISCreateGeneral(PETSC_COMM_SELF, nr_, rlist, PETSC_COPY_VALUES, H%isr, ierr)
        call ISCreateGeneral(PETSC_COMM_SELF, nc_, clist, PETSC_COPY_VALUES, H%isc, ierr)
        call MatCreateSubMatrices(af, one, [H%isr], [H%isc], MAT_INITIAL_MATRIX, H%sub, ierr)
        H%ready = .true.
      else
        call MatCreateSubMatrices(af, one, [H%isr], [H%isc], MAT_REUSE_MATRIX, H%sub, ierr)
      endif

      allocate(colJ(0:max(size(H%rcol), 1) - 1), colL(0:max(size(H%rcol), 1) - 1))
      colJ = -1; colL = 0
      ! C nodes (even positions) within the reach of an odd position: at most 2 ceil(reach / 2)
      ncmax = max(1, min(nF, 2 * ((o%rh_reach + 1) / 2)))
      allocate(ncq(nF), nearJ(ncmax, nF), Dinv(m, m, nF), Afc(m, m, ncmax, nF), X(m, m, ncmax, nF), &
               R(m, m, ncmax, nF), piv(m), wk(m * m), nnb(nF))
      do i = 1, ni - 1
        if (.not. ring_gathered(i)) cycle
        ! this ring's columns: position -> (J, local index k * ns + s + 1)
        do j = 0, nj - 1
          do k = 0, 3
            dd = LF%d(i, j, k)
            if (dd < 0) cycle
            do s = 0, ns_h - 1
              c = bsearch(H%rcol, LF%row0(dd) + s)
              colJ(c) = j; colL(c) = k * ns_h + s + 1
            enddo
          enddo
        enddo
        ! the near C nodes of every F node
        do ff = 1, nF
          j = 2 * ff - 1
          ncq(ff) = 0
          do jc = 0, nj - 2, 2
            if (cdist(j, jc) > o%rh_reach) cycle
            ncq(ff) = ncq(ff) + 1; nearJ(ncq(ff), ff) = jc
          enddo
        enddo
        ! the blocks: D, the neighbour F nodes, the near C nodes
        nnbmax = nF
        allocate(nbF(nnbmax, nF), Ab(m, m, nnbmax, nF))
        nnb = 0; Ab = 0.d0; Afc = 0.d0
        Dinv = 0.d0
        do ff = 1, nF
          j = 2 * ff - 1
          do k = 0, 3
            dd = LF%d(i, j, k)
            do s = 0, ns_h - 1
              ll = k * ns_h + s + 1
              if (dd < 0) then
                Dinv(ll, ll, ff) = 1.d0                    ! no DOF: an identity row, decoupled
                cycle
              endif
              rpos = int(bsearch(H%rrow, LF%row0(dd) + s), kind(one))
              call MatGetRow(H%sub(1), rpos, ncols, cols, vals, ierr)
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
                    nnb(ff) = nnb(ff) + 1; nb_ = nnb(ff); nbF(nb_, ff) = (colJ(c) + 1) / 2
                  endif
                  Ab(ll, colL(c), nb_, ff) = vals(a)
                else                                       ! a C node: within the reach
                  do qq = 1, ncq(ff)
                    if (nearJ(qq, ff) == colJ(c)) Afc(ll, colL(c), qq, ff) = vals(a)
                  enddo
                endif
              enddo
              call MatRestoreRow(H%sub(1), rpos, ncols, cols, vals, ierr)
            enddo
          enddo
          ! D^-1
          call dgetrf(m, m, Dinv(:, :, ff), m, piv, info)
          if (info /= 0) call fatal("a singular node block in the ring-harmonic transfer")
          call dgetri(m, Dinv(:, :, ff), m, piv, wk, m * m, info)
        enddo
        ! the sweeps (Jacobi: all R from the last X)
        X = 0.d0
        do t = 1, o%rh_sweeps
          do ff = 1, nF
            do q = 1, ncq(ff)
              R(:, :, q, ff) = Afc(:, :, q, ff)
              do qq = 1, nnb(ff)
                f = nbF(qq, ff)
                do c = 1, ncq(f)
                  if (nearJ(c, f) == nearJ(q, ff)) &
                    R(:, :, q, ff) = R(:, :, q, ff) + matmul(Ab(:, :, qq, ff), X(:, :, c, f))
                enddo
              enddo
            enddo
          enddo
          ! X - omega D^-1 (A_FF X + A_FC) with A_FF = D + the neighbours (R holds the latter and A_FC)
          do ff = 1, nF
            do q = 1, ncq(ff)
              X(:, :, q, ff) = (1.d0 - o%rh_omega) * X(:, :, q, ff) - o%rh_omega * matmul(Dinv(:, :, ff), R(:, :, q, ff))
            enddo
          enddo
        enddo
        ! the rows of P of my F DOFs: X_f Pc
        allocate(ecol(ncmax * m), evals(ncmax * m))
        do ff = 1, nF
          j = 2 * ff - 1
          do k = 0, 3
            dd = LF%d(i, j, k)
            if (dd < 0) cycle
            if (LF%own(dd) /= me .or. .not. LF%ok(dd)) cycle
            do s = 0, ns_h - 1
              if (.not. rh_slot(s)) cycle                  ! mixed: n = 0 keeps the linear row (fill_p)
              ll = k * ns_h + s + 1
              row = LF%row0(dd) + s
              nent = 0
              do q = 1, ncq(ff)
                jc = nearJ(q, ff)
                do kc = 0, 3
                  c = LF%d(i, jc, kc)
                  if (c < 0) cycle
                  if (.not. LF%ok(c)) cycle
                  do sc = 0, ns_h - 1
                    nent = nent + 1
                    ecol(nent) = LC%row0(LC%d(i, jc / 2, kc)) + sc
                    evals(nent) = X(ll, kc * ns_h + sc + 1, q, ff) * LF%osgn(c)
                  enddo
                enddo
              enddo
              if (nent > 0) call MatSetValues(P(g), one, [row], int(nent, kind(one)), ecol(1:nent), &
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
            do s = 0, ns_h - 1
              c = bsearch(H%rcol, LF%row0(dd) + s)
              colJ(c) = -1
            enddo
          enddo
        enddo
      enddo
    end associate
  contains
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
        if (lv(g - 1)%d(ii, 1, kk) < 0) cycle
        ring_gathered = bsearch(rh(g)%rrow, lv(g - 1)%row0(lv(g - 1)%d(ii, 1, kk))) >= 0
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
    call PCMGSetLevels(pc, int(nlev, kind(L)), PETSC_NULL_MPI_COMM, ierr)
    call PCMGSetType(pc, PC_MG_MULTIPLICATIVE, ierr)
    call PCMGSetCycleType(pc, PC_MG_CYCLE_V, ierr)
    ! the Galerkin operators are galerkin_chain's (the ring-harmonic transfer needs them level by level)
    call PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE, ierr)
    do g = 1, nlev - 1
      L = nlev - g
      call PCMGSetInterpolation(pc, L, P(g), ierr)
    enddo
    do g = 0, nlev - 2
      L = nlev - 1 - g
      call PCMGGetSmoother(pc, L, ks, ierr)
      if (g == 0) then
        call KSPSetOperators(ks, A0, A0, ierr)
      else
        call KSPSetOperators(ks, Ag(g), Ag(g), ierr)
      endif
      call KSPSetType(ks, KSPCHEBYSHEV, ierr)
      call KSPChebyshevEstEigSet(ks, 0.d0, o%safety / o%ratio, 0.d0, o%safety, ierr)
      call KSPChebyshevEstEigGetKSP(ks, kest, ierr)
      call KSPSetType(kest, KSPCG, ierr)
      call KSPSetTolerances(kest, 1.d-50, 1.d-50, 1.d10, int(o%esteig_its, kind(L)), ierr)
      call KSPSetTolerances(ks, 1.d-50, 1.d-50, 1.d10, int(o%degree, kind(L)), ierr)
      call KSPGetPC(ks, pcs, ierr)
      call PCSetType(pcs, PCSHELL, ierr)
      call PCShellSetApply(pcs, as_apply, ierr)
      call PCShellSetSetUp(pcs, as_setup, ierr)
      write(nm, '(A,I0)') "SPD-MG additive Schwarz, level ", g
      call PCShellSetName(pcs, trim(nm), ierr)
      as_pc(g) = pcs
    enddo
    call PCMGGetCoarseSolve(pc, kc, ierr)
    call KSPSetOperators(kc, Ag(nlev - 1), Ag(nlev - 1), ierr)
    call KSPSetType(kc, KSPPREONLY, ierr)
    call KSPGetPC(kc, pcc, ierr)
    call PCSetType(pcc, PCLU, ierr)
    call PCFactorSetMatSolverType(pcc, MATSOLVERMUMPS, ierr)
  end subroutine configure_pcmg

  !> The level g of the shell PC pc (its handle among as_pc).
  integer function level_of(pc)
    PC, intent(in) :: pc
    integer :: g
    level_of = -1
    do g = 0, nlev - 2
      if (as_pc(g)%v == pc%v) then
        level_of = g; return
      endif
    enddo
    call fatal("a shell PC that is not one of the levels' smoothers")
  end function level_of

  !> The rows of level g's blocks that this rank solves, block by block, and
  !! their weights in the additive sum. All ranks list all blocks of the
  !! layout (o%blocks) as DOF sets, which may overlap: the axis block (rings
  !! 0..axis_rings), then the nodes, points or radial lines, then the wedges
  !! (linewedge). A block's rows are all slots of its DOFs, ascending; it
  !! belongs to the owner of its smallest DOF. A new layout is a new case
  !! here.
  subroutine block_members(g, nb, off, gidx, bw)
    integer, intent(in) :: g
    integer, intent(out) :: nb
    integer, allocatable, intent(out) :: off(:)
    PetscInt, allocatable, intent(out) :: gidx(:)
    real*8, allocatable, intent(out) :: bw(:)
    integer, allocatable :: ab0(:), abd(:), sel(:), key0(:), keyd(:)
    real*8, allocatable :: aw(:)
    logical, allocatable :: jin(:)
    integer :: nab, nabd, f, j, j0, k, ax, lf, t, b, s, nsel, p, nmine, nw
    ax = o%axis_rings
    lf = o%line_from
    if (lf < 0) lf = ax + 1
    associate (L => lv(g))
      allocate(ab0(1), abd(max(2 * L%n, 16)), aw(0), sel(L%n))
      ab0(1) = 0; nab = 0; nabd = 0
      ! the axis block
      nsel = 0
      do f = 0, L%n - 1
        if (L%dI(f) <= ax) call put(f)
      enddo
      call push(1.d0)
      select case (o%blocks)
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
      if (o%blocks == BL_LINEWEDGE) then                    ! wedge_lines lines every wedge_stride, rings lf..t
        t = wedge_last(g, lf)
        nw = 0
        if (t >= lf) then
          allocate(jin(0:L%nj - 1))
          do j0 = 0, L%nj - 1, o%wedge_stride
            jin = .false.
            do k = 0, o%wedge_lines - 1
              jin(modulo(j0 + k, L%nj)) = .true.
            enddo
            nsel = 0
            do f = 0, L%n - 1
              if (L%dI(f) >= lf .and. L%dI(f) <= t .and. jin(L%dJ(f))) call put(f)
            enddo
            if (nsel > 0) nw = nw + 1
            call push(o%wedge_weight)
          enddo
        endif
        if (me == 0 .and. .not. as(g)%ready) then
          if (nw > 0) then
            write(*,'(A,I0,A,I0,A,I0,A,I0,A,F5.2,A)') "[Physics PC]   SPD-MG level ", g, ": ", nw, &
              " wedges on the rings ", lf, "..", t, " (weight ", o%wedge_weight, ")"
          else
            write(*,'(A,I0,A)') "[Physics PC]   SPD-MG level ", g, ": no wedge (no ring below the cell aspect)"
          endif
        endif
      endif

      ! mine: rows (all slots) in w order, sorted inside each block
      nmine = 0
      do b = 1, nab
        if (L%own(minval(abd(ab0(b) + 1 : ab0(b + 1)))) == me) nmine = nmine + 1
      enddo
      nb = nmine
      allocate(off(0:nb), bw(0:max(nb - 1, 0)))
      off(0) = 0; k = 0
      do b = 1, nab
        if (L%own(minval(abd(ab0(b) + 1 : ab0(b + 1)))) /= me) cycle
        k = k + 1
        off(k) = off(k - 1) + (ab0(b + 1) - ab0(b)) * ns_h
        bw(k - 1) = aw(b)
      enddo
      allocate(gidx(max(off(nb), 1)))
      k = 0
      do b = 1, nab
        if (L%own(minval(abd(ab0(b) + 1 : ab0(b + 1)))) /= me) cycle
        p = off(k)
        do j = ab0(b) + 1, ab0(b + 1)
          do s = 0, ns_h - 1
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
      associate (L => lv(g))
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
    do i = max(lf, 1), size(ring_asp) - 1
      a = ring_asp(i) * dble(n_tht) / dble(lv(g)%nj)
      if (a < o%wedge_aspect) wedge_last = i
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
    call asm_build(as(g), g, pmat)
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
    call asm_apply(as(g), x, y)
    ierr = 0
  end subroutine as_apply

  !> Blocks, scatter and factors of level g's additive Schwarz PC for A.
  subroutine asm_build(S, g, Aop)
    type(asm_t), intent(inout) :: S
    integer, intent(in) :: g
    Mat, intent(in) :: Aop
    external :: dgetrf
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
      ! the factor storage: dense blocks in lu, the others sparse
      allocate(S%dense(0:S%nb - 1), S%loff(0:S%nb - 1), S%piv(max(S%nw, 1)))
      tot = 0; nsp = 0
      do b = 0, S%nb - 1
        S%dense(b) = S%sz(b) <= o%dense_max
        S%loff(b) = tot
        if (S%dense(b)) then
          tot = tot + int(S%sz(b), 8)**2
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
        S%lu(S%loff(b) + 1 : S%loff(b) + int(n, 8)**2) = reshape(m, [n * n])
        deallocate(m)
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
            call PCFactorSetMatOrderingType(pcb, MATORDERINGND, ierr)
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
  end subroutine asm_build

  !> y = sum_b R_b' A_bb^-1 R_b x.
  subroutine asm_apply(S, x, y)
    type(asm_t), intent(inout) :: S
    Vec :: x, y
    external :: dgetrs
    PetscErrorCode :: ierr
    PetscScalar, pointer :: wa(:), va(:)
    integer :: b, n, info
    call VecScatterBegin(S%sct, x, S%w, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(S%sct, x, S%w, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecGetArray(S%w, wa, ierr)
    do b = 0, S%nb - 1
      n = S%sz(b)
      if (S%dense(b)) then
        call dgetrs("N", n, 1, S%lu(S%loff(b) + 1), n, S%piv(S%off(b) + 1), wa(S%off(b) + 1), n, info)
      else
        call VecGetArray(S%vb(b), va, ierr)
        va(1:n) = wa(S%off(b) + 1 : S%off(b) + n)
        call VecRestoreArray(S%vb(b), va, ierr)
        call KSPSolve(S%kb(b), S%vb(b), S%vx(b), ierr)
        call VecGetArray(S%vx(b), va, ierr)
        wa(S%off(b) + 1 : S%off(b) + n) = va(1:n)
        call VecRestoreArray(S%vx(b), va, ierr)
      endif
      if (S%bw(b) /= 1.d0) wa(S%off(b) + 1 : S%off(b) + n) = S%bw(b) * wa(S%off(b) + 1 : S%off(b) + n)
    enddo
    call VecRestoreArray(S%w, wa, ierr)
    call VecSet(y, 0.d0, ierr)
    call VecScatterBegin(S%sct, S%w, y, ADD_VALUES, SCATTER_REVERSE, ierr)
    call VecScatterEnd(S%sct, S%w, y, ADD_VALUES, SCATTER_REVERSE, ierr)
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
    integer :: g, nbk(2), nbk_g(2), nsp_g
    real*8 :: nz(0:MAXL-1), c_op
    character(len=9), parameter :: bn(4) = ["node     ", "line     ", "point    ", "linewedge"]
    call KSPGetPC(ksp, pc, ierr)
    do g = 0, nlev - 1
      L = nlev - 1 - g
      if (g == nlev - 1) then
        call PCMGGetCoarseSolve(pc, kl, ierr)
      else
        call PCMGGetSmoother(pc, L, kl, ierr)
      endif
      call KSPGetOperators(kl, al, pl, ierr)
      call MatGetInfo(al, MAT_GLOBAL_SUM, info, ierr)
      nz(g) = info%nz_used
    enddo
    c_op = sum(nz(0:nlev - 1)) / nz(0)
    nbk = 0
    do g = 0, nlev - 2
      nbk(1) = nbk(1) + as(g)%nb
      nbk(2) = max(nbk(2), maxval(as(g)%sz))
    enddo
    call MPI_Allreduce(nbk(1), nbk_g(1), 1, MPI_INTEGER, MPI_SUM, comm_h, ierr)
    call MPI_Allreduce(nbk(2), nbk_g(2), 1, MPI_INTEGER, MPI_MAX, comm_h, ierr)
    call MPI_Allreduce(as(0)%nsparse, nsp_g, 1, MPI_INTEGER, MPI_SUM, comm_h, ierr)
    if (me /= 0) return                                   ! the first rank of the operator's communicator
    write(*,'(A,A,A)', advance="no") "[Physics PC]   SPD-MG (", trim(label), ") levels rows (nnz/row):"
    do g = 0, nlev - 1
      write(*,'(A,I0,A,F7.1,A)', advance="no") " ", lv(g)%n * ns_h, " (", nz(g) / max(lv(g)%n * ns_h, 1), ")"
    enddo
    write(*,'(A,F6.3)') "   C_op = ", c_op
    write(*,'(A,A,A,I0,A,I0,A,I0,A)') "[Physics PC]   SPD-MG smoother: Chebyshev(", trim(itoa(o%degree)), &
      ") of additive Schwarz, "//trim(bn(o%blocks))//" blocks, axis block rings 0..", o%axis_rings, &
      "; ", nbk_g(1), " blocks over the levels, largest ", nbk_g(2), " rows"
    if (o%blocks == BL_LINE .or. o%blocks == BL_LINEWEDGE) write(*,'(A,I0,A)') &
      "[Physics PC]   SPD-MG radial lines from ring ", merge(o%line_from, o%axis_rings + 1, o%line_from >= 0), &
      merge(" (overlapping the axis block)", "                             ", &
            o%line_from >= 0 .and. o%line_from <= o%axis_rings)
    if (o%blocks == BL_LINEWEDGE) write(*,'(A,I0,A,I0,A,F5.2,A,F5.2)') "[Physics PC]   SPD-MG wedges of ", &
      o%wedge_lines, " lines every ", o%wedge_stride, " lines, rings with a cell aspect below ", o%wedge_aspect, &
      ", weight ", o%wedge_weight
    write(*,'(A,F6.2,A,F6.3,A,I0,A,A)') "[Physics PC]   SPD-MG interval [lmax * ", o%safety / o%ratio, &
      ", lmax * ", o%safety, "], lmax from ", o%esteig_its, " CG steps; transfer ", trim(trname())
    if (nsp_g > 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   SPD-MG level 0: ", nsp_g, &
      " block(s) above ", o%dense_max, " rows on sparse LU"
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
    if (.not. allocated(hist)) allocate(hist(MAXIT + 1))
    nh = MAXIT + 1
    nfam = 0
    if (ns_h == n_tor .and. n_tor > 1) nfam = (n_tor + 1) / 2
    call KSPGetPC(ksp, pc, ierr)
    call MatCreateVecs(A, x, b, ierr)
    call MatGetOwnershipRange(A, rs, re, ierr)
    call PetscRandomCreate(comm_h, rnd, ierr)
    call PetscRandomSetInterval(rnd, -1.d0, 1.d0, ierr)
    call PetscRandomSetFromOptions(rnd, ierr)
    do fam = -1, nfam - 1
      call KSPCreate(comm_h, kt, ierr)
      call KSPSetType(kt, KSPCG, ierr)
      call KSPSetNormType(kt, KSP_NORM_UNPRECONDITIONED, ierr)
      call KSPSetOperators(kt, A, A, ierr)
      call KSPSetPC(kt, pc, ierr)
      call KSPSetTolerances(kt, 1.d-10, 1.d-50, 1.d10, int(MAXIT, kind(its)), ierr)
      call KSPSetComputeSingularValues(kt, PETSC_TRUE, ierr)
      call KSPSetResidualHistory(kt, hist, nh, PETSC_TRUE, ierr)
      call VecSetRandom(b, rnd, ierr)
      what = "all slots"
      if (fam >= 0) then
        call VecGetArray(b, ba, ierr)
        do r = rs, re - 1
          s = int(mod(r, int(ns_h, kind(r))))
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
          if (hist(k) <= tols(q) * hist(1)) then
            t(q) = k - 1; exit
          endif
        enddo
      enddo
      if (me == 0) write(*,'(A,A,A,A,A,3(I0,A),ES10.3,A,ES10.3,A,F9.2,A,I0)') "[Physics PC]   SPD-MG test (", &
        trim(label), ", ", trim(what), "): CG its to 1e-1 / 1e-6 / 1e-10: ", t(1), " / ", t(2), " / ", t(3), &
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
    select case (o%transfer)
    case (TR_LINEAR);   s = "linear (kinds as fields)"
    case (TR_CONST);    s = "const (kinds as fields)"
    case (TR_RINGHARM)
      write(s, '(A,I0,A,I0,A,F5.2,A)') "ring-harmonic (", o%rh_sweeps, " steps, reach ", o%rh_reach, &
        ", omega ", o%rh_omega, ")"
    case (TR_MIXED)
      write(s, '(A,I0,A,I0,A,F5.2,A)') "mixed: n = 0 linear, harmonics ring-harmonic (", o%rh_sweeps, &
        " steps, reach ", o%rh_reach, ", omega ", o%rh_omega, ")"
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
    integer :: mpierr
    if (me == 0) write(*,'(A,A)') "[Physics PC]   FATAL: SPD-MG: ", trim(msg)
    call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
  end subroutine fatal

#endif
end module mod_petsc_pc_spdmg
