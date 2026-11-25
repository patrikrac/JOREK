!> The local dual basis Z ~ M^-1 of a mass operator M: sparse, explicit and
!! not symmetric, one small dense solve per node block and toroidal slot.
!!
!! A node block is the set of all DOFs of a grid node (nodes that share DOFs,
!! as at the polar axis, form one block); its patch is the block and every
!! block of a node that shares an element with it (the blocks M couples to
!! it). For a block with the DOFs own and the patch DOFs J:
!!   Y = M(J, J)^-1 E_own,   Z(own, J) = Y',   Z(own, not J) = 0,
!! E_own the columns of the identity on J at own. Then (Z M)(i, j) = delta_ij
!! for i in own and j in J. The mass matrix does not couple the toroidal
!! slots: the patch solves run per slot. For a non-symmetric M, Y' holds the
!! rows of M(J, J)^-T: Z is the local dual basis of M', and a caller that
!! wants M's passes M' (vs_lumped's P_V^A).
!!
!! M is a one-field operator of mod_petsc_pc_sf_ops: global row i ns + q for
!! JOREK index i (0-based) and the operator's slot q, ns slots per index, in
!! either layout. A patch can cross rank boundaries: each rank fetches the
!! rows and columns of its patches once (MatCreateSubMatrices) and computes
!! the rows of Z it owns. Z holds M's values of the call: a changed M needs a
!! new call.
!!
!! The patches come from the grid, not from M's pattern: M can be any operator
!! of this layout that the patch matrices invert well, as the mass plus a
!! resistive term (the resistive line bending of vs_lumped). Entries of M that
!! couple two slots or leave the patch are not read.
!!
!! The core (option core of dual_basis): the blocks of more than one node (the
!! polar axis) and the blocks of their patches share one patch, the union of
!! their patches (one solve for all of them). For an operator much stiffer
!! than a mass on the small elements at the axis, whose inverse is not local
!! there (the resistive line bending: without the core its Z has negative
!! eigenvalues at the axis on fine grids).
!!
!! The rings: ring 0 is the blocks of more than one node (the polar axis), ring
!! d + 1 the blocks in the patches of ring d's that no lower ring holds (on a
!! polar grid the flux surfaces, counted from the axis). Two options of
!! dual_basis use them, for the axis Schur complement form of the resistive
!! bending (mod_petsc_pc_vs, -vs_dual_axis schur):
!!   axis_cut k   the blocks on the rings 0..k give no rows of Z and their
!!                indices leave every patch: Z is the local dual basis of
!!                M(R, R), R the indices on the rings > k, and zero on the rows
!!                and columns of A, the indices on the rings 0..k; axis_idx
!!                returns A, ordered poloidal-major (the position along the
!!                ring, then the ring, then the DOF kind), the shared indices
!!                (the axis value) last
!!   axis_split   every node is a block of its own indices; an index of more
!!                than one node (the shared value of the axis) is in no block
!!                and in no patch, so its row and column of Z are zero: no
!!                block of Z spans the axis
!! The core is not taken with either.
!!
!! Symmetric forms of Z (dual_symmetrize), for a product U' Z U that must be
!! symmetric:
!!   DUAL_AVG   (Z + Z') / 2        the symmetric part, on Z's pattern. U' Z U
!!                                  and U' ((Z + Z') / 2) U have the same
!!                                  quadratic form: the average takes the skew
!!                                  part out and leaves the definiteness
!!   DUAL_SYM   Z + Z' - Z' M Z     1 - Z_sym M = (1 - Z' M)(1 - Z M), so
!!                                  Z_sym <= M^-1, and positive definite where
!!                                  the M-norm of 1 - Z M is below 1; couplings
!!                                  up to three nodes apart
module mod_petsc_pc_dual
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  integer, parameter, public :: DUAL_PLAIN = 0, DUAL_AVG = 1, DUAL_SYM = 2

  public :: dual_basis, dual_symmetrize

  !> -log_view events (DUAL_*): the whole dual basis, its plan (blocks,
  !! patches, halo and its submatrix), the patch solves (the core patch also
  !! alone: DUAL_Core), Z's assembly, the symmetric form. LAPACK flops are
  !! logged with PetscLogFlops.
  PetscLogEvent, save :: dev_basis = -1, dev_plan = -1, dev_solve = -1, dev_core = -1, dev_z = -1, dev_sym = -1
  logical, save :: dev_ready = .false.

contains

  !> The -log_view events of the module, once.
  subroutine dual_register_events()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    if (dev_ready) return
    call PetscLogEventRegister("DUAL_Basis", cid, dev_basis, ierr)
    call PetscLogEventRegister("DUAL_Plan",  cid, dev_plan,  ierr)
    call PetscLogEventRegister("DUAL_Solve", cid, dev_solve, ierr)
    call PetscLogEventRegister("DUAL_Core",  cid, dev_core,  ierr)
    call PetscLogEventRegister("DUAL_Z",     cid, dev_z,     ierr)
    call PetscLogEventRegister("DUAL_Sym",   cid, dev_sym,   ierr)
    dev_ready = .true.
  end subroutine dual_register_events

  !> Z := the symmetric form `form` (DUAL_AVG | DUAL_SYM) of the local dual
  !! basis Z of the symmetric M; DUAL_PLAIN leaves Z. Collective.
  subroutine dual_symmetrize(Z, M, form)
    Mat, intent(inout)  :: Z
    Mat, intent(in)     :: M
    integer, intent(in) :: form
    Mat :: zt, ztmz
    PetscErrorCode :: ierr

    if (form == DUAL_PLAIN) return
    call dual_register_events()
    call PetscLogEventBegin(dev_sym, ierr)
    call MatTranspose(Z, MAT_INITIAL_MATRIX, zt, ierr)
    if (form == DUAL_SYM) then
      call MatPtAP(M, Z, MAT_INITIAL_MATRIX, 2.0d0, ztmz, ierr)
      call MatAXPY(Z, 1.0d0, zt, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatAXPY(Z, -1.0d0, ztmz, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatDestroy(ztmz, ierr)
    else
      call MatAXPY(Z, 1.0d0, zt, DIFFERENT_NONZERO_PATTERN, ierr)
      call MatScale(Z, 0.5d0, ierr)
    endif
    call MatDestroy(zt, ierr)
    call PetscLogEventEnd(dev_sym, ierr)
  end subroutine dual_symmetrize

  !> Z, the local dual basis of M (MPIAIJ, M's layout and communicator, the
  !! options prefix vs_dual_: -vs_dual_mat_view shows it). Collective. With
  !! `name` (the operator's name) rank 0 reports the patches; core: the core
  !! patch at the axis; axis_cut, axis_idx, axis_split: the axis Schur
  !! complement form's options (module header).
  subroutine dual_basis(M, ns, Z, my_id, name, core, axis_cut, axis_idx, axis_split)
    use nodes_elements, only: node_list, element_list
    use mod_parameters, only: n_degrees, n_vertex_max
    Mat, intent(in)     :: M
    integer, intent(in) :: ns           !< toroidal slots per index in M's layout
    Mat, intent(out)    :: Z
    integer, intent(in) :: my_id
    character(len=*), intent(in), optional :: name
    logical, intent(in), optional :: core
    integer, intent(in), optional :: axis_cut     !< the blocks on the rings 0..axis_cut give no rows and leave the patches
    integer, allocatable, intent(out), optional :: axis_idx(:)  !< with axis_cut: the indices on those rings, K's order
    logical, intent(in), optional :: axis_split   !< a block per node, the shared indices in no block
    external :: dgesv
    PetscInt, parameter :: zero = 0, one = 1
    PetscInt :: rs, re, ng, ncols, nloc
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    Mat, pointer :: sub(:)
    IS :: isr
    PetscErrorCode :: ierr
    integer :: comm, n_idx, i0, i1, nn, ne, n, k, i, e, iv, b, c, p, q, a, j, nj, no, info, mpierr
    integer :: nmy, nh, ent, stamp, nsum, njc, kcut, ncut
    logical :: core_done, split
    logical, allocatable :: incore(:), shared(:), bcut(:)
    integer, allocatable :: jcore(:), nring(:)
    integer, allocatable :: parent(:), owner(:), root(:), blk(:)
    integer, allocatable :: bn0(:), bnod(:), bm0(:), bmem(:), ne0(:), nel(:), cnt(:)
    integer, allocatable :: mine(:), hal(:), posh(:), posj(:), jl(:), ol(:), piv(:), mark(:), seen(:)
    integer, allocatable :: zr0(:), d_nnz(:), o_nnz(:)
    PetscInt, allocatable :: zc(:), gl(:)
    real*8, allocatable :: mjj(:, :), y(:, :), zv(:)

    call dual_register_events()
    call PetscLogEventBegin(dev_basis, ierr)
    call PetscLogEventBegin(dev_plan, ierr)
    call PetscObjectGetComm(M, comm, ierr)
    call MatGetOwnershipRange(M, rs, re, ierr)
    call MatGetSize(M, ng, PETSC_NULL_INTEGER, ierr)
    if (mod(int(rs), ns) /= 0 .or. mod(int(re), ns) /= 0 .or. mod(int(ng), ns) /= 0) &
      call fail("the operator's rows are not whole (index, slots) groups")
    n_idx = int(ng) / ns; i0 = int(rs) / ns; i1 = int(re) / ns
    nn = node_list%n_nodes; ne = element_list%n_elements
    kcut = -1
    if (present(axis_cut)) kcut = axis_cut
    split = .false.
    if (present(axis_split)) split = axis_split
    if ((kcut >= 0 .or. split) .and. present(core)) then
      if (core) call fail("the core patch with axis_cut or axis_split")
    endif

    ! node blocks: nodes that share an index are merged (union-find on nodes)
    allocate(parent(nn), owner(0:n_idx - 1), root(nn), blk(0:n_idx - 1))
    parent = [(n, n = 1, nn)]
    owner = 0
    allocate(shared(0:n_idx - 1)); shared = .false.
    do n = 1, nn
      do k = 1, n_degrees
        i = node_list%node(n)%index(k) - 1
        if (i < 0 .or. i >= n_idx) call fail("a node index outside the operator")
        if (owner(i) == 0) then
          owner(i) = n
        else
          parent(find(n)) = find(owner(i))
          shared(i) = .true.
        endif
      enddo
    enddo
    do n = 1, nn
      root(n) = find(n)
    enddo
    if (any(owner == 0)) call fail("an index without a node")
    do i = 0, n_idx - 1
      blk(i) = root(owner(i))
    enddo

    ! CSR: the nodes of each block, the indices of each block, the elements of each node
    call csr_count(root, nn, nn, bn0, bnod)
    allocate(cnt(nn + 1)); cnt = 0
    do e = 1, ne
      do iv = 1, n_vertex_max
        n = element_list%element(e)%vertex(iv)
        cnt(n + 1) = cnt(n + 1) + 1
      enddo
    enddo
    allocate(ne0(nn + 1)); ne0(1) = 1
    do n = 1, nn
      ne0(n + 1) = ne0(n) + cnt(n + 1)
    enddo
    allocate(nel(ne0(nn + 1) - 1)); cnt = 0
    do e = 1, ne
      do iv = 1, n_vertex_max
        n = element_list%element(e)%vertex(iv)
        nel(ne0(n) + cnt(n)) = e; cnt(n) = cnt(n) + 1
      enddo
    enddo
    ! the rings of the merged blocks (axis_cut): nring(node)
    if (kcut >= 0) call make_rings()
    ! axis_split: a block per node, the shared indices in the extra block nn + 1 (no rows, no patch)
    if (split) then
      root = [(n, n = 1, nn)]
      deallocate(bn0, bnod)
      call csr_count(root, nn, nn, bn0, bnod)
      do i = 0, n_idx - 1
        blk(i) = root(owner(i))
        if (shared(i)) blk(i) = nn + 1
      enddo
      call csr_count(blk, n_idx, nn + 1, bm0, bmem)
    else
      call csr_count(blk, n_idx, nn, bm0, bmem)
    endif
    bmem = bmem - 1                                   ! indices are 0-based
    allocate(bcut(nn)); bcut = .false.
    if (kcut >= 0) then
      do n = 1, nn
        if (root(n) == n) bcut(n) = nring(n) <= kcut  ! a merged block's nodes share the ring
      enddo
      if (present(axis_idx)) call axis_order()
    endif

    ! my blocks (those holding an index of mine) and the halo of their patches
    allocate(mark(nn), seen(nn), posh(0:n_idx - 1), posj(0:n_idx - 1))
    mark = 0; seen = 0; stamp = 0; nmy = 0
    allocate(incore(nn)); incore = .false.; njc = 0
    if (present(core)) then
      if (core) call make_core()
    endif
    allocate(mine(nn))
    do i = i0, i1 - 1
      if (blk(i) > nn) cycle                          ! axis_split: a shared index, no row
      if (bcut(blk(i))) cycle                         ! axis_cut: no rows on the rings 0..k
      if (mark(blk(i)) == 0) then
        mark(blk(i)) = 1; nmy = nmy + 1; mine(nmy) = blk(i)
      endif
    enddo
    posh = -1
    do b = 1, nmy
      call patch(mine(b), jl, nj)
      do a = 1, nj
        posh(jl(a)) = 0
      enddo
    enddo
    nh = count(posh == 0)
    allocate(hal(nh)); p = 0
    do i = 0, n_idx - 1
      if (posh(i) == 0) then
        p = p + 1; hal(p) = i; posh(i) = p - 1
      endif
    enddo

    ! the halo's rows and columns of M, on every slot
    allocate(gl(nh * ns))
    do p = 1, nh
      do q = 0, ns - 1
        gl((p - 1) * ns + q + 1) = int(hal(p), kind(rs)) * ns + q
      enddo
    enddo
    call ISCreateGeneral(PETSC_COMM_SELF, int(nh * ns, kind(rs)), gl, PETSC_COPY_VALUES, isr, ierr)
    call MatCreateSubMatrices(M, one, [isr], [isr], MAT_INITIAL_MATRIX, sub, ierr)
    call ISDestroy(isr, ierr)
    call PetscLogEventEnd(dev_plan, ierr)

    ! the patch solves, per block and slot; Z's rows of mine in CSR (zr0, zc, zv)
    call PetscLogEventBegin(dev_solve, ierr)
    nloc = re - rs
    allocate(zr0(nloc + 1), d_nnz(nloc), o_nnz(nloc))
    zr0 = 0
    do b = 1, nmy                                       ! row lengths: the patch size
      call patch(mine(b), jl, nj)
      do k = bm0(mine(b)), bm0(mine(b) + 1) - 1
        i = bmem(k)
        if (i < i0 .or. i >= i1) cycle
        do q = 0, ns - 1
          zr0((i - i0) * ns + q + 2) = nj
        enddo
      enddo
    enddo
    zr0(1) = 0
    do k = 1, int(nloc)
      zr0(k + 1) = zr0(k) + zr0(k + 1)
    enddo
    allocate(zc(max(zr0(nloc + 1), 1)), zv(max(zr0(nloc + 1), 1)))
    posj = 0
    core_done = .false.
    do b = 1, nmy
      if (incore(mine(b))) then                         ! the core: one solve for all its blocks
        if (core_done) cycle
        core_done = .true.
        call PetscLogEventBegin(dev_core, ierr)
        ol = pack(jcore, incore(blk(jcore)))
      else
        ol = bmem(bm0(mine(b)) : bm0(mine(b) + 1) - 1)
      endif
      no = size(ol)
      call patch(mine(b), jl, nj)
      do a = 1, nj
        posj(jl(a)) = a
      enddo
      allocate(mjj(nj, nj), y(nj, no), piv(nj))
      do q = 0, ns - 1
        mjj = 0.d0
        do a = 1, nj
          call MatGetRow(sub(1), int(posh(jl(a)) * ns + q, kind(rs)), ncols, cols, vals, ierr)
          do c = 1, int(ncols)
            if (mod(int(cols(c)), ns) /= q) cycle
            j = posj(hal(int(cols(c)) / ns + 1))
            if (j > 0) mjj(a, j) = vals(c)
          enddo
          call MatRestoreRow(sub(1), int(posh(jl(a)) * ns + q, kind(rs)), ncols, cols, vals, ierr)
        enddo
        y = 0.d0
        do k = 1, no
          y(posj(ol(k)), k) = 1.d0
        enddo
        call dgesv(nj, no, mjj, nj, piv, y, nj, info)
        if (info /= 0) call fail("a singular patch matrix")
        call PetscLogFlops(2.d0 / 3.d0 * dble(nj)**3 + 2.d0 * dble(nj)**2 * dble(no), ierr)
        do k = 1, no
          i = ol(k)
          if (i < i0 .or. i >= i1) cycle
          ent = zr0((i - i0) * ns + q + 1)
          do a = 1, nj
            zc(ent + a) = int(jl(a), kind(rs)) * ns + q
            zv(ent + a) = y(a, k)
          enddo
        enddo
      enddo
      deallocate(mjj, y, piv)
      do a = 1, nj
        posj(jl(a)) = 0
      enddo
      if (incore(mine(b))) call PetscLogEventEnd(dev_core, ierr)
    enddo
    call MatDestroySubMatrices(one, sub, ierr)
    call PetscLogEventEnd(dev_solve, ierr)

    ! Z, with M's layout
    call PetscLogEventBegin(dev_z, ierr)
    do k = 1, int(nloc)
      d_nnz(k) = count(zc(zr0(k) + 1 : zr0(k + 1)) >= rs .and. zc(zr0(k) + 1 : zr0(k + 1)) < re)
      o_nnz(k) = zr0(k + 1) - zr0(k) - d_nnz(k)
    enddo
    call MatCreate(comm, Z, ierr)
    call MatSetSizes(Z, nloc, nloc, ng, ng, ierr)
    call MatSetType(Z, MATMPIAIJ, ierr)                ! as M, also on one rank: their products need it
    call MatSetOptionsPrefix(Z, "vs_dual_", ierr)
    call MatMPIAIJSetPreallocation(Z, zero, int(d_nnz, kind(rs)), zero, int(o_nnz, kind(rs)), ierr)
    do k = 1, int(nloc)
      if (zr0(k + 1) == zr0(k)) cycle
      call MatSetValues(Z, one, [rs + k - 1], int(zr0(k + 1) - zr0(k), kind(rs)), zc(zr0(k) + 1 : zr0(k + 1)), &
                        zv(zr0(k) + 1 : zr0(k + 1)), INSERT_VALUES, ierr)
    enddo
    call MatAssemblyBegin(Z, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(Z, MAT_FINAL_ASSEMBLY, ierr)
    call PetscLogEventEnd(dev_z, ierr)
    call PetscLogEventEnd(dev_basis, ierr)
    if (present(name)) then
      call MPI_Allreduce(nmy, nsum, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
      if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   local dual basis of "//name//": ", nsum, &
        " block patches (summed over ranks), ", ns, " slot(s) each"
      if (my_id == 0 .and. njc > 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]     core at the axis: ", &
        count(incore), " blocks, one patch of ", njc, " indices"
      if (my_id == 0 .and. kcut >= 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]     axis cut: the ", ncut, &
        " indices on the rings 0..", kcut, " give no rows and leave every patch"
      if (my_id == 0 .and. split) write(*,'(A,I0,A)') "[Physics PC]     axis split: a block per node, ", &
        count(shared), " shared index (the axis value) in no patch, its row zero"
    endif

  contains

    recursive integer function find(x) result(r)
      integer, intent(in) :: x
      r = x
      if (parent(r) /= r) then
        r = find(parent(r)); parent(x) = r
      endif
    end function find

    !> CSR (p0, v) of the items 1..nitem grouped by key(item) in 1..nkey;
    !! keys and items are those of the arrays' indexing (lbound of key).
    subroutine csr_count(key, nitem, nkey, p0, v)
      integer, intent(in) :: key(:), nitem, nkey
      integer, allocatable, intent(out) :: p0(:), v(:)
      integer :: t, kk
      integer, allocatable :: c_(:)
      allocate(p0(nkey + 1), c_(nkey))
      c_ = 0
      do t = 1, nitem
        c_(key(t)) = c_(key(t)) + 1
      enddo
      p0(1) = 1
      do kk = 1, nkey
        p0(kk + 1) = p0(kk) + c_(kk)
      enddo
      allocate(v(nitem)); c_ = 0
      do t = 1, nitem
        v(p0(key(t)) + c_(key(t))) = t
        c_(key(t)) = c_(key(t)) + 1
      enddo
    end subroutine csr_count

    !> The core (incore, jcore, njc): the blocks of more than one node, the
    !! blocks of their patches, and the union of these blocks' patches.
    subroutine make_core()
      integer :: t, s, x
      logical, allocatable :: inc(:), inj(:)
      allocate(inc(nn), inj(0:n_idx - 1)); inc = .false.; inj = .false.
      do t = 1, nn
        if (root(t) /= t .or. bn0(t + 1) - bn0(t) < 2) cycle
        inc(t) = .true.
        call patch(t, jl, nj)
        do s = 1, nj
          inc(blk(jl(s))) = .true.
        enddo
      enddo
      do t = 1, nn
        if (.not. inc(t)) cycle
        call patch(t, jl, nj)
        do s = 1, nj
          inj(jl(s)) = .true.
        enddo
      enddo
      jcore = pack([(x, x = 0, n_idx - 1)], inj)
      njc = size(jcore)
      incore = inc
    end subroutine make_core

    !> nring(node): the ring of the node's merged block (module header), by a
    !! breadth-first search over the blocks that share an element; ncut, the
    !! indices on the rings 0..kcut. Before the blocks are split.
    subroutine make_rings()
      integer :: t, s, x, w, d, nf, nnext
      integer, allocatable :: front(:), nxt(:)
      allocate(nring(nn)); nring = -1
      allocate(front(nn), nxt(nn))
      nf = 0
      do t = 1, nn
        if (root(t) == t .and. bn0(t + 1) - bn0(t) >= 2) then
          nf = nf + 1; front(nf) = t; nring(t) = 0
        endif
      enddo
      if (nf == 0) call fail("axis_cut: no block of more than one node (no polar axis)")
      d = 0
      do while (nf > 0)
        nnext = 0
        do s = 1, nf
          do t = bn0(front(s)), bn0(front(s) + 1) - 1
            do x = ne0(bnod(t)), ne0(bnod(t) + 1) - 1
              do iv = 1, n_vertex_max
                w = root(element_list%element(nel(x))%vertex(iv))
                if (nring(w) >= 0) cycle
                nring(w) = d + 1; nnext = nnext + 1; nxt(nnext) = w
              enddo
            enddo
          enddo
        enddo
        front(1:nnext) = nxt(1:nnext); nf = nnext; d = d + 1
      enddo
      do t = 1, nn
        nring(t) = nring(root(t))
      enddo
      if (any(nring < 0)) call fail("axis_cut: a node the rings do not reach")
      ncut = 0
      do i = 0, n_idx - 1
        if (nring(owner(i)) <= kcut) ncut = ncut + 1
      enddo
    end subroutine make_rings

    !> axis_idx: the indices on the rings 0..kcut, poloidal-major (the node's
    !! position along its ring in node order, as a fraction of the ring, then
    !! the ring, then the DOF kind), the shared indices last.
    subroutine axis_order()
      integer :: t, s, x, kd, nr
      integer, allocatable :: jpos(:), rcnt(:), key(:)
      integer(8), allocatable :: kk(:)
      integer(8) :: tk
      nr = maxval(nring) + 1
      allocate(jpos(nn), rcnt(0:nr - 1)); rcnt = 0
      do t = 1, nn                                      ! node order runs along the ring
        jpos(t) = rcnt(nring(t)); rcnt(nring(t)) = rcnt(nring(t)) + 1
      enddo
      allocate(axis_idx(ncut), kk(ncut), key(ncut))
      s = 0
      do i = 0, n_idx - 1
        t = owner(i)
        if (nring(t) > kcut) cycle
        kd = 0
        do x = 1, n_degrees
          if (node_list%node(t)%index(x) - 1 == i) kd = x
        enddo
        s = s + 1
        axis_idx(s) = i
        if (shared(i)) then
          kk(s) = huge(1_8)
        else
          kk(s) = (int(nint(1.d6 * dble(jpos(t)) / dble(rcnt(nring(t)))), 8) * 64_8 + int(nring(t), 8)) * 8_8 + kd
        endif
      enddo
      do t = 2, ncut                                   ! insertion sort by key, stable
        tk = kk(t); x = axis_idx(t); s = t - 1
        do while (s >= 1)
          if (kk(s) <= tk) exit
          kk(s + 1) = kk(s); axis_idx(s + 1) = axis_idx(s); s = s - 1
        enddo
        kk(s + 1) = tk; axis_idx(s + 1) = x
      enddo
    end subroutine axis_order

    !> The patch of block bb: the sorted indices of every block of a node that
    !! shares an element with a node of bb; the core's patch for a core block;
    !! with axis_cut without the blocks on the rings 0..kcut.
    subroutine patch(bb, jl_, nj_)
      integer, intent(in) :: bb
      integer, allocatable, intent(out) :: jl_(:)
      integer, intent(out) :: nj_
      integer :: t, s, x, w, nb_
      integer, allocatable :: bl(:)
      if (incore(bb)) then
        jl_ = jcore; nj_ = njc
        return
      endif
      allocate(bl(0)); nb_ = 0
      stamp = stamp + 1
      do t = bn0(bb), bn0(bb + 1) - 1
        do s = ne0(bnod(t)), ne0(bnod(t) + 1) - 1
          do x = 1, n_vertex_max
            w = root(element_list%element(nel(s))%vertex(x))
            if (bcut(w)) cycle
            if (seen(w) /= stamp) then
              seen(w) = stamp; nb_ = nb_ + 1; bl = [bl, w]
            endif
          enddo
        enddo
      enddo
      nj_ = 0
      do t = 1, nb_
        nj_ = nj_ + bm0(bl(t) + 1) - bm0(bl(t))
      enddo
      allocate(jl_(nj_)); nj_ = 0
      do t = 1, nb_
        do s = bm0(bl(t)), bm0(bl(t) + 1) - 1
          nj_ = nj_ + 1; jl_(nj_) = bmem(s)
        enddo
      enddo
      call sort_int(jl_)
    end subroutine patch

    subroutine sort_int(v)
      integer, intent(inout) :: v(:)
      integer :: t, s, x
      do t = 2, size(v)                                 ! insertion sort: patches hold some 40 indices
        x = v(t); s = t - 1
        do while (s >= 1)
          if (v(s) <= x) exit
          v(s + 1) = v(s); s = s - 1
        enddo
        v(s + 1) = x
      enddo
    end subroutine sort_int

    subroutine fail(msg)
      character(len=*), intent(in) :: msg
      write(*,'(A,A,A,I0)') "[Physics PC]   FATAL: local dual basis: ", msg, " on rank ", my_id
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    end subroutine fail

  end subroutine dual_basis

#endif
end module mod_petsc_pc_dual
