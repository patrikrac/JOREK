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
!! serial numbering. Smoother blocks are built from OWNED rows only: a radial
!! line cut by the partition becomes one block per rank (block Jacobi, no
!! communication in the smoother).
module mod_petsc_pc_gmg
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: gmg_build_prolongations, gmg_setup_operator, gmg_vcycle_apply, gmg_is_ready
  public :: gmg_select, gmg_vcycle, gmg_pc_apply_3, gmg_pc_apply_4

  integer, parameter :: MAX_LEV   = 6
  integer, parameter :: SM_STEPS  = 4      !< GMRES iterations per smoothing (the paper's, and stage C1's)
  integer, parameter :: MAX_ENT   = 16     !< max nonzeros per scalar row of P (2x2 sources x 4 DOFs)

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
  KSP, save         :: gSm(0:MAX_LEV-1), gCoarse, gAxis
  Vec, save         :: gx(0:MAX_LEV-1), gb(0:MAX_LEV-1), gr(0:MAX_LEV-1), gzax
  IS, save          :: gisAxis
  Mat, save         :: gF                   !< fine-level matvec operator: gA(0), or a
                                            !< matrix-free equivalent (Workstream D)
  logical, save     :: p_ready = .false., op_ready = .false., vec_ready = .false.

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
    logical :: axsparse = .false., axready = .false.
    logical, allocatable :: axblk(:)
    IS  :: axis_is
    Mat :: axmat
    KSP :: axksp
    ! physics_pc_gmg_axis_mult > 0: Gauss-Seidel coupling between the axis
    ! block and the lines through the interface blocks A(rest,ax), A(ax,rest)
    logical :: gsready = .false., gsvec = .false.
    IS  :: rest_is
    Mat :: Bra, Bar
    Vec :: xw, xw2, tr, ta
  end type blk_t
  type(blk_t), save, target :: gBk(0:MAX_LEV-1)
  type(lvl_t), allocatable, save :: glv(:)   !< coarse DOF numberings, kept for the block maps
  integer, allocatable, save :: fine_node(:) !< level-0 row -> node-1 (i*n_tht + j), from node%index
  integer, allocatable, save :: fine_kf(:)   !< level-0 row -> canonical DOF k + 4*field (ring diagnostics)
  integer, allocatable, save :: fine_harm(:) !< level-0 row -> toroidal slot m
  integer, save :: nth0 = 0                  !< n_tht (level-0 nj)
  integer, save :: nf_s = 0, cur_lev = 0
  integer, save :: sm_type = 0, sm_nstep = 4
  logical, save :: sm_blocks = .false.
  real*8, allocatable, save :: blk_t_work(:)
  ! Axis treatment (Bourne et al., JCP 488 (2023) 112249 S2-S3). Fine rings
  ! I < ring_is have median r*dtheta/dr below physics_pc_gmg_ring_aspect: the
  ! circle couplings dominate there, so smoother 6 uses ring blocks inside and
  ! radial lines outside (GMGPolar). Rings 0..axis_k form the axis block.
  integer, save :: ring_is = 1, axis_k = 0, axis_mult = 0
  integer, save :: diag_left = 0            !< ring-diag samples left in this rebuild
  integer, save :: gcomm = 0, gme = 0       !< communicator and rank of the hierarchy

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
    integer :: ring_is = 1, axis_k = 0, axis_mult = 0, diag_left = 0
    logical :: p_ready = .false., op_ready = .false., vec_ready = .false., sm_blocks = .false.
    Mat :: gP(1:MAX_LEV-1), gA(0:MAX_LEV-1), gF
    KSP :: gSm(0:MAX_LEV-1), gCoarse, gAxis
    Vec :: gx(0:MAX_LEV-1), gb(0:MAX_LEV-1), gr(0:MAX_LEV-1), gzax
    IS  :: gisAxis
    type(blk_t) :: gBk(0:MAX_LEV-1)
    type(lvl_t), allocatable :: glv(:)
    integer, allocatable :: fine_node(:), fine_harm(:), fine_kf(:)
  end type gmg_inst_t
  type(gmg_inst_t), save :: inst(MAX_INST)
  integer, save :: cur_inst = 1

  ! -log_view events (Workstream D cost audit). Registered here, not in the
  ! physics-PC ctx, so this module keeps no dependency on it.
  PetscLogEvent, save :: gev_ptap = -1, gev_smsetup = -1, gev_coarselu = -1, gev_axislu = -1
  PetscLogEvent, save :: gev_axis = -1
  ! per hierarchy instance (1 = pair_w keeps the original names, 2 = GMG2_*)
  PetscLogEvent, save :: gev_vcycle(4) = -1, gev_smooth0(4) = -1, gev_smooth(4) = -1, gev_coarse(4) = -1
  logical, save       :: gev_ready = .false.

  ! 1D cubic-Hermite subdivision, level-local canonical units (u, f_x/3):
  ! W(p+1, q+1) maps coarse quantity q to fine quantity p.
  real*8, parameter :: W_CO(2,2) = reshape([1.0d0, 0.0d0, 0.0d0, 0.5d0],      [2,2])
  real*8, parameter :: W_L(2,2)  = reshape([0.5d0, -0.25d0, 0.375d0, -0.125d0], [2,2])
  real*8, parameter :: W_R(2,2)  = reshape([0.5d0, 0.25d0, -0.375d0, -0.125d0], [2,2])
  integer, parameter :: PQ_I(0:3) = [0, 1, 0, 1]   !< canonical k -> derivative order in i (u, a, b, c)
  integer, parameter :: PQ_J(0:3) = [0, 0, 1, 1]   !< ... and in j

contains

  !> Make hierarchy k the active one (see gmg_inst_t).
  subroutine gmg_select(k)
    integer, intent(in) :: k
    integer :: g
    if (k == cur_inst) return
    ! park the active state
    associate (S => inst(cur_inst))
      S%nlev = nlev; S%nth0 = nth0; S%nf_s = nf_s; S%sm_type = sm_type; S%sm_nstep = sm_nstep
      S%ring_is = ring_is; S%axis_k = axis_k; S%axis_mult = axis_mult; S%diag_left = diag_left
      S%p_ready = p_ready; S%op_ready = op_ready; S%vec_ready = vec_ready; S%sm_blocks = sm_blocks
      S%gP = gP; S%gA = gA; S%gF = gF; S%gSm = gSm; S%gCoarse = gCoarse; S%gAxis = gAxis
      S%gx = gx; S%gb = gb; S%gr = gr; S%gzax = gzax; S%gisAxis = gisAxis
      do g = 0, MAX_LEV - 1
        call move_blk(gBk(g), S%gBk(g))
      enddo
      if (allocated(glv)) call move_alloc(glv, S%glv)
      if (allocated(fine_node)) call move_alloc(fine_node, S%fine_node)
      if (allocated(fine_kf)) call move_alloc(fine_kf, S%fine_kf)
      if (allocated(fine_harm)) call move_alloc(fine_harm, S%fine_harm)
    end associate
    ! load instance k
    associate (S => inst(k))
      nlev = S%nlev; nth0 = S%nth0; nf_s = S%nf_s; sm_type = S%sm_type; sm_nstep = S%sm_nstep
      ring_is = S%ring_is; axis_k = S%axis_k; axis_mult = S%axis_mult; diag_left = S%diag_left
      p_ready = S%p_ready; op_ready = S%op_ready; vec_ready = S%vec_ready; sm_blocks = S%sm_blocks
      gP = S%gP; gA = S%gA; gF = S%gF; gSm = S%gSm; gCoarse = S%gCoarse; gAxis = S%gAxis
      gx = S%gx; gb = S%gb; gr = S%gr; gzax = S%gzax; gisAxis = S%gisAxis
      do g = 0, MAX_LEV - 1
        call move_blk(S%gBk(g), gBk(g))
      enddo
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
    b%axsparse = a%axsparse; b%axready = a%axready
    b%axis_is = a%axis_is; b%axmat = a%axmat; b%axksp = a%axksp
    a%axsparse = .false.; a%axready = .false.
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
  end subroutine move_blk

  !> One V-cycle of hierarchy k on b (x out), outside any PC: the pair_psi
  !! eta-Schur applies it to Shat.
  subroutine gmg_vcycle(k, b, x)
    integer, intent(in) :: k
    Vec :: b, x
    PetscErrorCode :: ierr
    call gmg_select(k)
    if (diag_left > 0 .and. cur_inst <= 2) call ring_diag(b)
    call PetscLogEventBegin(gev_vcycle(cur_inst), ierr)
    call vcycle(0, b, x)
    call PetscLogEventEnd(gev_vcycle(cur_inst), ierr)
  end subroutine gmg_vcycle

  !> PC-shell entry points of the rho (3) and T (4) hierarchies.
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
  subroutine gmg_build_prolongations(Aref, comm, my_id, n_fields, ok)
    use nodes_elements, only: node_list, element_list
    use phys_module,    only: n_flux, n_tht, physics_pc_gmg_ring_aspect, physics_pc_gmg_bnd_drop
    use mod_parameters, only: n_tor, n_degrees

    Mat, intent(in)      :: Aref
    integer, intent(in)  :: comm, my_id, n_fields
    logical, intent(out) :: ok

    type(lvl_t), allocatable :: lv(:)
    integer, allocatable :: o(:,:)
    ! (lv is handed to the module-level glv at the end, for the block maps)
    integer :: n_idx, e, iv, n, k, i, j, ni, nj, g, cand(4), nent, nrow_s, ncol_s
    integer :: eps_s(4), eps_t(4), cols(MAX_ENT), vi(4), vj(4), idx
    real*8  :: w(MAX_ENT)
    integer, allocatable :: rs(:), cs(:)
    real*8,  allocatable :: ws(:)
    logical, allocatable :: done(:)
    integer :: ns
    type(lay_t), allocatable :: ly(:)
    integer, allocatable :: rstarts(:)
    integer :: me, np, mpierr, r, first, ci, cj, dd, own_, ff, mm, lr, nloc
    PetscInt :: rs0, re0, nglob
    PetscErrorCode :: ierr

    ok = .false.
    gcomm = comm; gme = my_id
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

    !--- coarse levels
    allocate(lv(1:MAX_LEV - 1))
    ni = n_flux; nj = n_tht; nlev = 1
    do g = 1, MAX_LEV - 1
      if (mod(ni - 1, 2) /= 0 .or. mod(nj, 2) /= 0 .or. nj < 4) exit
      ni = (ni - 1) / 2 + 1
      nj = nj / 2
      call make_level(lv(g), ni, nj, physics_pc_gmg_bnd_drop > 0)
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
    if (nglob /= int(n_fields, 8) * n_idx * n_tor) then
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
      if (mod(rstarts(r + 1) - rstarts(r), n_fields * n_tor) /= 0) then
        call fail("a rank's rows are not whole (field x toroidal slot) groups")
        return
      endif
      ly(0)%nl(r) = (rstarts(r + 1) - rstarts(r)) / (n_fields * n_tor)
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
          n = (ci * 2**g) * n_tht + cj * 2**g + 1
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
        ly(g)%ps(r + 1) = ly(g)%ps(r) + n_fields * n_tor * ly(g)%nl(r)
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
        call interp_row(i, j, k, lv(1), cols, w, nent)
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
            call interp_row(i, j, k, lv(g), cols, w, nent)
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
    nloc = n_fields * n_tor * ly(0)%nl(me)
    allocate(fine_node(nloc), fine_harm(nloc), fine_kf(nloc))
    fine_node = -1
    do n = 1, node_list%n_nodes
      do k = 1, 4
        idx = node_list%node(n)%index(k) - 1
        if (ly(0)%own(idx) /= me) cycle
        do ff = 0, n_fields - 1
          do mm = 0, n_tor - 1
            lr = (ff * ly(0)%nl(me) + ly(0)%lpos(idx)) * n_tor + mm + 1
            fine_node(lr) = n - 1
            fine_harm(lr) = mm
            fine_kf(lr) = (k - 1) + 4 * ff
          enddo
        enddo
      enddo
    enddo
    if (any(fine_node < 0)) then
      call fail("row -> node map does not cover every row")
      return
    endif
    do g = 1, nlev - 1
      nloc = n_fields * n_tor * ly(g)%nl(me)
      allocate(lv(g)%rnode(nloc), lv(g)%rharm(nloc))
      lv(g)%rnode = -1
      do ci = 0, lv(g)%ni - 1
        do cj = 0, lv(g)%nj - 1
          do k = 0, 3
            dd = lv(g)%d(ci, cj, k)
            if (dd < 0) cycle
            if (ly(g)%own(dd) /= me) cycle
            do ff = 0, n_fields - 1
              do mm = 0, n_tor - 1
                lr = (ff * ly(g)%nl(me) + ly(g)%lpos(dd)) * n_tor + mm + 1
                lv(g)%rnode(lr) = ci * lv(g)%nj + cj
                lv(g)%rharm(lr) = mm
              enddo
            enddo
          enddo
        enddo
      enddo
      if (any(lv(g)%rnode < 0)) then
        call fail("coarse row -> node map does not cover every row")
        return
      endif
    enddo
    if (my_id == 0 .and. np > 1) then
      write(*,'(A,I0,A)', advance="no") "[Physics PC]   GMG parallel layout on ", np, " ranks, rows/rank min-max per level:"
      do g = 0, nlev - 1
        write(*,'(A,I0,A,I0)', advance="no") " ", n_fields * n_tor * minval(ly(g)%nl), "-", &
                                             n_fields * n_tor * maxval(ly(g)%nl)
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
      do ii = 2, na                              ! insertion sort, na ~ 1e3-1e4
        tmp = ar(ii); kk = ii - 1
        do while (kk >= 1)
          if (ar(kk) <= tmp) exit
          ar(kk + 1) = ar(kk); kk = kk - 1
        enddo
        ar(kk + 1) = tmp
      enddo
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
          do jj = 2, n_tht
            tmp = row(jj); kk = jj - 1
            do while (kk >= 1)
              if (row(kk) <= tmp) exit
              row(kk + 1) = row(kk); kk = kk - 1
            enddo
            row(kk + 1) = tmp
          enddo
          rm(ii) = row((n_tht + 1) / 2)
        enddo
        ring_is = n_flux - 1
        do ii = 1, n_flux - 2
          if (rm(ii) >= physics_pc_gmg_ring_aspect) then
            ring_is = ii; exit
          endif
        enddo
        if (my_id == 0) then
          nshow = min(n_flux - 2, ring_is + 2)
          write(*,'(A,F5.2,A,I0,A)', advance="no") "[Physics PC]   GMG ring medians rdtheta/dr (switch at ", &
            physics_pc_gmg_ring_aspect, ": rings I < ", ring_is, "):"
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

  contains

    subroutine fail(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   GMG hierarchy REFUSED: ", msg
    end subroutine fail

    !> Packed global row of (field f, scalar DOF d, slot m) on layout level L.
    integer function prow(L, f, d, m)
      integer, intent(in) :: L, f, d, m
      integer :: ow
      ow = ly(L)%own(d)
      prow = ly(L)%ps(ow) + (f * ly(L)%nl(ow) + ly(L)%lpos(d)) * n_tor + m
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

      nrl = n_fields * n_tor * ly(lr_)%nl(me); ncl = n_fields * n_tor * ly(lc_)%nl(me)
      nrg = n_fields * n_tor * ly(lr_)%n;      ncg = n_fields * n_tor * ly(lc_)%n
      call MatCreate(comm, P, ierr_)
      call MatSetSizes(P, nrl, ncl, nrg, ncg, ierr_)
      call MatSetType(P, MATMPIAIJ, ierr_)       ! the packed pairs are MPIAIJ even on one rank
      call MatSeqAIJSetPreallocation(P, MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr_)
      call MatMPIAIJSetPreallocation(P, MAX_ENT, PETSC_NULL_INTEGER_ARRAY, &
                                     MAX_ENT, PETSC_NULL_INTEGER_ARRAY, ierr_)
      do q = 1, size(r_)
        if (ly(lr_)%own(r_(q)) /= me) cycle
        do f = 0, n_fields - 1
          do m = 0, n_tor - 1
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

      allocate(seen(n_idx_), ax(n_fields * n_tor * 4 * n_tht * 2))
      seen = .false.; nax = 0
      do nn = 1, node_list%n_nodes
        if ((nn - 1) / n_tht /= 0) cycle          ! first ring = the axis
        do kk = 1, 4
          id = node_list%node(nn)%index(kk)
          if (seen(id)) cycle
          seen(id) = .true.
          if (ly(0)%own(id - 1) /= me) cycle      ! owned rows only: no duplicates at np > 1
          do f = 0, n_fields - 1
            do m = 0, n_tor - 1
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
  subroutine interp_row(fi, fj, k, c, cols, w, nent)
    integer, intent(in)      :: fi, fj, k
    type(lvl_t), intent(in)  :: c
    integer, intent(out)     :: cols(MAX_ENT), nent
    real*8, intent(out)      :: w(MAX_ENT)
    integer :: si(2), sj(2), nsi, nsj, a, b, kc, d, q
    real*8  :: Wi(2, 2, 2), Wj(2, 2, 2), wt
    logical :: found

    call sources_1d(fi, c%ni, .false., si, Wi, nsi)
    call sources_1d(fj, c%nj, .true.,  sj, Wj, nsj)
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
  !! at every PC rebuild (A is a new Mat each time); P is reused.
  !--------------------------------------------------------------------
  subroutine gmg_setup_operator(A, comm, my_id, Afine, tag, smoother, nsmooth)
    use phys_module, only: physics_pc_gmg_smoother, physics_pc_gmg_nsmooth, physics_pc_gmg_omega, &
                           physics_pc_gmg_axis_rings, physics_pc_gmg_ring_diag, physics_pc_gmg_axis_mult, &
                           physics_pc_gmg_bnd_drop
    Mat, intent(in)     :: A
    integer, intent(in) :: comm, my_id
    Mat, intent(in), optional :: Afine  !< Workstream D: applies the fine operator
                                        !< for every level-0 matvec; A still gives the
                                        !< Galerkin chain, Jacobi diagonal and axis block
    character(len=*), intent(in), optional :: tag   !< label for the prints
    integer, intent(in), optional :: smoother, nsmooth   !< override the physics_pc_gmg_* knobs
    PetscErrorCode :: ierr
    PC   :: pc
    Mat  :: Aax
    MatInfo :: minfo
    PetscInt :: nr
    integer :: g
    real*8 :: nz0, nzt, nzg

    if (op_ready) then
      do g = 1, nlev - 1
        call MatDestroy(gA(g), ierr)
      enddo
      do g = 0, nlev - 2
        call KSPDestroy(gSm(g), ierr)
      enddo
      call KSPDestroy(gCoarse, ierr)
      if (.not. sm_blocks) call KSPDestroy(gAxis, ierr)   ! sm_blocks: still last setup's
    endif

    call gmg_register_events()
    gA(0) = A
    gF = A
    if (present(Afine)) gF = Afine
    call PetscLogEventBegin(gev_ptap, ierr)
    do g = 1, nlev - 1
      call MatPtAP(gA(g - 1), gP(g), MAT_INITIAL_MATRIX, 2.0d0, gA(g), ierr)
    enddo
    call PetscLogEventEnd(gev_ptap, ierr)

    call PetscLogEventBegin(gev_smsetup, ierr)
    sm_type = physics_pc_gmg_smoother
    sm_nstep = physics_pc_gmg_nsmooth
    if (present(smoother)) then
      if (smoother >= 0) sm_type = smoother
    endif
    if (present(nsmooth)) then
      if (nsmooth > 0) sm_nstep = nsmooth
    endif
    if (sm_nstep <= 0) then
      sm_nstep = SM_STEPS
      if (sm_type == 1 .or. sm_type == 2) sm_nstep = 3
    endif
    sm_blocks = (sm_type >= 2)
    axis_k = 0
    if (sm_type >= 4) axis_k = max(physics_pc_gmg_axis_rings, -1)
    axis_mult = 0
    if (axis_k /= 0) axis_mult = min(max(physics_pc_gmg_axis_mult, 0), 3)
    diag_left = max(physics_pc_gmg_ring_diag, 0)
    do g = 0, nlev - 2
      call KSPCreate(comm, gSm(g), ierr)
      if (g == 0) then
        call KSPSetOperators(gSm(g), gF, gA(g), ierr)     ! Jacobi reads the Pmat
      else
        call KSPSetOperators(gSm(g), gA(g), gA(g), ierr)
      endif
      if (sm_type == 1 .or. sm_type == 2) then
        call KSPSetType(gSm(g), KSPRICHARDSON, ierr)
        call KSPRichardsonSetScale(gSm(g), physics_pc_gmg_omega, ierr)
        call KSPSetNormType(gSm(g), KSP_NORM_NONE, ierr)
        call KSPSetTolerances(gSm(g), 1.d-30, 1.d-50, 1.d30, sm_nstep, ierr)
      else
        call KSPSetType(gSm(g), KSPGMRES, ierr)
        call KSPGMRESSetRestart(gSm(g), sm_nstep, ierr)
        call KSPSetTolerances(gSm(g), 1.d-30, 1.d-50, 1.d30, sm_nstep, ierr)
        call KSPSetPCSide(gSm(g), PC_RIGHT, ierr)
      endif
      call KSPSetInitialGuessNonzero(gSm(g), PETSC_TRUE, ierr)
      call KSPGetPC(gSm(g), pc, ierr)
      if (sm_blocks) then
        call build_blocks(g, gA(g))
        call PCSetType(pc, PCSHELL, ierr)
        call PCShellSetApply(pc, blk_apply, ierr)
        call PCShellSetName(pc, "C1 node-block Jacobi", ierr)
      else
        call PCSetType(pc, PCJACOBI, ierr)
      endif
      call KSPSetUp(gSm(g), ierr)
    enddo
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
        write(*,'(A,I0,A,I0,A,F5.2)', advance="no") "[Physics PC]   GMG smoother ", sm_type, &
          ": steps = ", sm_nstep, ", omega = ", physics_pc_gmg_omega
        if (sm_blocks) then
          write(*,'(A,I0,A,I0,A,I0,A,I0,A)') ", blocks ", ibg(1), " on level 0 (", &
            ibg(2), " banded, max kl ", ib(1), "; ", ibg(3), " singular -> point)"
        else
          write(*,*)
        endif
        if (sm_type == 6) write(*,'(A,I0,A)', advance="no") &
          "[Physics PC]   GMG hybrid smoother: ring blocks on fine rings I < ", ring_is, &
          ", radial lines outside"
        if (sm_type >= 4 .and. axis_k > 0) write(*,'(A,I0,A)', advance="no") &
          "; axis block = rings 0..", axis_k, " on every level"
        if (sm_type >= 4 .and. axis_k < 0) write(*,'(A,I0,A)', advance="no") &
          "[Physics PC]   GMG axis block = rings 0..I_s-1 of every level (fine: 0..", axis_lim(0), ")"
        if (axis_mult > 0) write(*,'(A,I0)', advance="no") ", axis/lines Gauss-Seidel mode ", axis_mult
        if (sm_type == 6 .or. (sm_type >= 4 .and. axis_k /= 0)) write(*,*)
      endif
    end block

    ! Coarse and axis LUs carry an options prefix (gmg<k>_coarse_, gmg<k>_axis_)
    ! so e.g. -gmg1_coarse_pc_type telescope can be tried without a rebuild.
    call PetscLogEventBegin(gev_coarselu, ierr)
    call KSPCreate(comm, gCoarse, ierr)
    call KSPSetOperators(gCoarse, gA(nlev - 1), gA(nlev - 1), ierr)
    call KSPSetType(gCoarse, KSPPREONLY, ierr)
    call KSPGetPC(gCoarse, pc, ierr)
    call PCSetType(pc, PCLU, ierr)
    call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
    call set_prefix_mumps(gCoarse, "coarse")
    call KSPSetUp(gCoarse, ierr)
    call PetscLogEventEnd(gev_coarselu, ierr)

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
        call MatCreateVecs(gA(g), gx(g), gb(g), ierr)
        call VecDuplicate(gx(g), gr(g), ierr)
      enddo
      if (.not. sm_blocks) call MatCreateVecs(Aax, gzax, PETSC_NULL_VEC, ierr)
      vec_ready = .true.
    endif
    if (.not. sm_blocks) call MatDestroy(Aax, ierr)        ! gAxis holds its own reference
    op_ready = .true.
    if (diag_left > 0) call report_bnd_rows(gA(0))
    if (physics_pc_gmg_bnd_drop > 0) call check_bnd_rows(gA(0))

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
    call vcycle(0, b, x)
    call PetscLogEventEnd(gev_vcycle(cur_inst), ierr)
    ierr = 0
  end subroutine gmg_vcycle_apply

  subroutine gmg_register_events()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    if (gev_ready) return
    call PetscLogEventRegister("GMG_PtAP",     cid, gev_ptap,     ierr)
    call PetscLogEventRegister("GMG_SmSetup",  cid, gev_smsetup,  ierr)
    call PetscLogEventRegister("GMG_CoarseLU", cid, gev_coarselu, ierr)
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
    gev_ready = .true.
  end subroutine gmg_register_events

  recursive subroutine vcycle(g, b, x)
    integer, intent(in) :: g
    Vec :: b, x
    PetscErrorCode :: ierr

    if (g == nlev - 1) then
      call PetscLogEventBegin(gev_coarse(cur_inst), ierr)
      call KSPSolve(gCoarse, b, x, ierr)
      call PetscLogEventEnd(gev_coarse(cur_inst), ierr)
      return
    endif
    call VecZeroEntries(x, ierr)
    call smooth(g, b, x, .true.)                         ! pre-smooth (zero guess: no b - A*0)
    if (g == 0 .and. .not. sm_blocks) call axis_patch(b, x)
    if (g == 0) then
      call MatMult(gF, x, gr(g), ierr)
    else
      call MatMult(gA(g), x, gr(g), ierr)
    endif
    call VecAYPX(gr(g), -1.0d0, b, ierr)                  ! r = b - A x
    call MatMultTranspose(gP(g + 1), gr(g), gb(g + 1), ierr)
    call vcycle(g + 1, gb(g + 1), gx(g + 1))
    call MatMultAdd(gP(g + 1), gx(g + 1), x, x, ierr)     ! x += P e_c
    call smooth(g, b, x, .false.)                        ! post-smooth
    if (g == 0 .and. .not. sm_blocks) call axis_patch(b, x)
  end subroutine vcycle

  subroutine smooth(g, b, x, zero_guess)
    integer, intent(in) :: g
    Vec :: b, x
    logical, intent(in) :: zero_guess
    PetscErrorCode :: ierr
    cur_lev = g
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

  !> Block map of level g (0 = fine, from fine_node/fine_harm; g > 0 from
  !! glv(g)%rnode/rharm), then LU of every block of the Pmat A. Only the rows
  !! this rank owns take part (rows/bid/pos are LOCAL, 0-based in rows), so a
  !! block cut by the partition is solved as its local part. Rows inside a block are ordered by
  !! their coordinate along the line (I for radial lines, J for rings), so a
  !! radial line is banded; a block whose band is narrow is factored banded
  !! (dgbtrf), otherwise dense (dgetrf). A singular block falls back to its
  !! diagonal (counted in nsing).
  subroutine build_blocks(g, A)
    use mod_parameters, only: n_tor
    integer, intent(in) :: g
    Mat, intent(in)     :: A
    type(blk_t), pointer :: B
    PetscInt :: nr, r, ncols, rst, ren, lc
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscErrorCode :: ierr
    integer :: nc, I, J, m, bb, q, pr, pcn, info, n, ldab, kk, tmp
    integer, allocatable :: cnt(:), key(:)
    integer(8) :: tot, ix
    PetscInt, allocatable :: axr(:)
    PC :: axpc
    character(len=16) :: axname
    external :: dgetrf, dgbtrf

    B => gBk(g)
    call MatGetLocalSize(A, nr, PETSC_NULL_INTEGER, ierr)
    call MatGetOwnershipRange(A, rst, ren, ierr)
    if (.not. allocated(B%bid)) then
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
        if (sm_type == 5) key(r) = I
        if (sm_type == 4) key(r) = J
        if (sm_type == 6) then                     ! rings run along J, radial lines along I
          if (I < ring_lim(g)) then
            key(r) = J
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
      B%off(1) = 0
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
        do q = 2, B%sz(bb)
          tmp = B%rows(B%off(bb) + q); kk = q - 1
          do while (kk >= 1)
            if (key(B%rows(B%off(bb) + kk) + 1) <= key(tmp + 1)) exit
            B%rows(B%off(bb) + kk + 1) = B%rows(B%off(bb) + kk); kk = kk - 1
          enddo
          B%rows(B%off(bb) + kk + 1) = tmp
        enddo
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
        deallocate(axr)
      endif
      if (allocated(blk_t_work)) then
        if (size(blk_t_work) < maxval(B%sz)) deallocate(blk_t_work)
      endif
      if (.not. allocated(blk_t_work)) allocate(blk_t_work(maxval(B%sz)))
    endif

    ! band widths of the blocks as stored in A (pattern fixed across rebuilds,
    ! but recomputed: cheap next to the factorizations)
    B%kl = 0; B%ku = 0
    do r = 0, nr - 1
      bb = B%bid(r + 1); pr = B%pos(r + 1)
      call MatGetRow(A, rst + r, ncols, cols, vals, ierr)
      do q = 1, int(ncols)
        lc = cols(q) - rst
        if (lc < 0 .or. lc >= nr) cycle              ! off-rank column: not in any local block
        if (B%bid(lc + 1) /= bb) cycle
        pcn = B%pos(lc + 1)
        B%kl(bb) = max(B%kl(bb), pr - pcn)
        B%ku(bb) = max(B%ku(bb), pcn - pr)
      enddo
      call MatRestoreRow(A, rst + r, ncols, cols, vals, ierr)
    enddo
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
    if (allocated(B%lu)) then
      if (size(B%lu, kind=8) /= tot) deallocate(B%lu)
    endif
    if (.not. allocated(B%lu)) allocate(B%lu(tot))

    B%lu = 0.0d0
    do r = 0, nr - 1
      bb = B%bid(r + 1)
      if (B%axblk(bb)) cycle
      pr = B%pos(r + 1)
      call MatGetRow(A, rst + r, ncols, cols, vals, ierr)
      do q = 1, int(ncols)
        lc = cols(q) - rst
        if (lc < 0 .or. lc >= nr) cycle
        if (B%bid(lc + 1) /= bb) cycle
        pcn = B%pos(lc + 1)
        B%lu(lu_index(B, bb, pr, pcn)) = vals(q)
      enddo
      call MatRestoreRow(A, rst + r, ncols, cols, vals, ierr)
    enddo
    B%nsing = 0
    do bb = 1, B%nb
      n = B%sz(bb)
      if (B%axblk(bb)) cycle
      if (B%band(bb)) then
        ldab = 2 * B%kl(bb) + B%ku(bb) + 1
        call dgbtrf(n, n, B%kl(bb), B%ku(bb), B%lu(B%loff(bb) + 1), ldab, B%piv(B%off(bb) + 1), info)
      else
        call dgetrf(n, n, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), info)
      endif
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
          call MatGetRow(A, rst + r, ncols, cols, vals, ierr)
          do pcn = 1, int(ncols)
            if (cols(pcn) == rst + r) B%lu(ix) = vals(pcn)
          enddo
          call MatRestoreRow(A, rst + r, ncols, cols, vals, ierr)
          if (abs(B%lu(ix)) < tiny(1.0d0)) B%lu(ix) = 1.0d0
          B%piv(B%off(bb) + q) = q
        enddo
      endif
    enddo

    if (B%axsparse) then
      if (B%axready) then
        call KSPDestroy(B%axksp, ierr)
        call MatDestroy(B%axmat, ierr)
      endif
      call MatCreateSubMatrix(A, B%axis_is, B%axis_is, MAT_INITIAL_MATRIX, B%axmat, ierr)
      call KSPCreate(gcomm, B%axksp, ierr)
      call KSPSetOperators(B%axksp, B%axmat, B%axmat, ierr)
      call KSPSetType(B%axksp, KSPPREONLY, ierr)
      call KSPGetPC(B%axksp, axpc, ierr)
      call PCSetType(axpc, PCLU, ierr)
      call PCFactorSetMatSolverType(axpc, MATSOLVERMUMPS, ierr)
      write(axname, '(A,I0)') "axblk", g
      call set_prefix_mumps(B%axksp, trim(axname))
      call KSPSetUp(B%axksp, ierr)
      if (.not. B%axready) then                ! first build: factor size (INFOG 22 = MB, 29 = entries)
        block
          Mat :: Fax
          PetscInt :: mb, nent, nax
          call PCFactorGetMatrix(axpc, Fax, ierr)
          call MatMumpsGetInfog(Fax, 22_4, mb, ierr)
          call MatMumpsGetInfog(Fax, 29_4, nent, ierr)
          call ISGetSize(B%axis_is, nax, ierr)
          if (gme == 0) write(*,'(A,I0,A,I0,A,I0,A,I0,A,ES10.3)') "[Mem] MUMPS GMG", cur_inst, &
            " axis block level ", g, ": ", nax, " rows, MB ", mb, ", factor entries ", &
            merge(-dble(nent) * 1.d6, dble(nent), nent < 0)
        end block
      endif
      B%axready = .true.
    endif

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
  integer function blk_id(g, I, J, m, nj)
    use mod_parameters, only: n_tor
    integer, intent(in) :: g, I, J, m, nj
    integer :: is_g
    if (I == 0 .or. (sm_type >= 4 .and. I <= axis_lim(g))) then
      blk_id = m + 1
    else if (sm_type == 4) then
      blk_id = n_tor + (I - 1) * n_tor + m + 1
    else if (sm_type == 5) then
      blk_id = n_tor + J * n_tor + m + 1
    else if (sm_type == 6) then
      is_g = ring_lim(g)
      if (I < is_g) then
        blk_id = n_tor + (I - 1) * n_tor + m + 1
      else
        blk_id = n_tor + max(is_g - 1, 0) * n_tor + J * n_tor + m + 1
      endif
    else
      blk_id = n_tor + ((I - 1) * nj + J) * n_tor + m + 1
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
  subroutine blk_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    type(blk_t), pointer :: B

    B => gBk(cur_lev)
    if (.not. B%axsparse .or. axis_mult == 0) then
      call lines_solve(x, y)
      if (B%axsparse) call ax_solve(x, y)
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

    !> y(rows outside the axis block) = dense/banded block solves of x
    subroutine lines_solve(xx, yy)
      Vec :: xx, yy
      PetscScalar, pointer :: xp(:), yp(:)
      PetscErrorCode :: ie
      integer :: bb, q, n, info
      external :: dgetrs, dgbtrs
      call VecGetArrayRead(xx, xp, ie)
      call VecGetArray(yy, yp, ie)
      do bb = 1, B%nb
        if (B%axblk(bb)) cycle
        n = B%sz(bb)
        do q = 1, n
          blk_t_work(q) = xp(B%rows(B%off(bb) + q) + 1)
        enddo
        if (B%band(bb)) then
          call dgbtrs('N', n, B%kl(bb), B%ku(bb), 1, B%lu(B%loff(bb) + 1), 2 * B%kl(bb) + B%ku(bb) + 1, &
                      B%piv(B%off(bb) + 1), blk_t_work, n, info)
        else
          call dgetrs('N', n, 1, B%lu(B%loff(bb) + 1), n, B%piv(B%off(bb) + 1), blk_t_work, n, info)
        endif
        do q = 1, n
          yp(B%rows(B%off(bb) + q) + 1) = blk_t_work(q)
        enddo
      enddo
      call VecRestoreArrayRead(xx, xp, ie)
      call VecRestoreArray(yy, yp, ie)
    end subroutine lines_solve

    !> y(axis rows) = axis-block solve of x(axis rows)
    subroutine ax_solve(xx, yy)
      Vec :: xx, yy
      Vec :: xs, ys
      PetscErrorCode :: ie
      call VecGetSubVector(xx, B%axis_is, xs, ie)
      call VecGetSubVector(yy, B%axis_is, ys, ie)
      call KSPSolve(B%axksp, xs, ys, ie)
      call VecRestoreSubVector(yy, B%axis_is, ys, ie)
      call VecRestoreSubVector(xx, B%axis_is, xs, ie)
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
