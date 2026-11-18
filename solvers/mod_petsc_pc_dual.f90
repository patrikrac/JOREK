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
!! slots: the patch solves run per slot.
!!
!! M is a one-field operator of mod_petsc_pc_sf_ops: global row i ns + q for
!! JOREK index i (0-based) and the operator's slot q, ns slots per index, in
!! either layout. A patch can cross rank boundaries: each rank fetches the
!! rows and columns of its patches once (MatCreateSubMatrices) and computes
!! the rows of Z it owns. M's values must not change afterwards.
module mod_petsc_pc_dual
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: dual_basis

contains

  !> Z, the local dual basis of M (MPIAIJ, M's layout and communicator, the
  !! options prefix vs_dual_: -vs_dual_mat_view shows it). Collective.
  subroutine dual_basis(M, ns, Z, my_id)
    use nodes_elements, only: node_list, element_list
    use mod_parameters, only: n_degrees, n_vertex_max
    Mat, intent(in)     :: M
    integer, intent(in) :: ns           !< toroidal slots per index in M's layout
    Mat, intent(out)    :: Z
    integer, intent(in) :: my_id
    external :: dgesv
    PetscInt, parameter :: zero = 0, one = 1
    PetscInt :: rs, re, ng, ncols, nloc
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    Mat, pointer :: sub(:)
    IS :: isr
    PetscErrorCode :: ierr
    integer :: comm, n_idx, i0, i1, nn, ne, n, k, i, e, iv, b, c, p, q, a, j, nj, no, info, mpierr
    integer :: nmy, nh, ent, stamp, nsum
    integer, allocatable :: parent(:), owner(:), root(:), blk(:)
    integer, allocatable :: bn0(:), bnod(:), bm0(:), bmem(:), ne0(:), nel(:), cnt(:)
    integer, allocatable :: mine(:), hal(:), posh(:), posj(:), jl(:), ol(:), piv(:), mark(:), seen(:)
    integer, allocatable :: zr0(:), d_nnz(:), o_nnz(:)
    PetscInt, allocatable :: zc(:), gl(:)
    real*8, allocatable :: mjj(:, :), y(:, :), zv(:)

    call PetscObjectGetComm(M, comm, ierr)
    call MatGetOwnershipRange(M, rs, re, ierr)
    call MatGetSize(M, ng, PETSC_NULL_INTEGER, ierr)
    if (mod(int(rs), ns) /= 0 .or. mod(int(re), ns) /= 0 .or. mod(int(ng), ns) /= 0) &
      call fail("the operator's rows are not whole (index, slots) groups")
    n_idx = int(ng) / ns; i0 = int(rs) / ns; i1 = int(re) / ns
    nn = node_list%n_nodes; ne = element_list%n_elements

    ! node blocks: nodes that share an index are merged (union-find on nodes)
    allocate(parent(nn), owner(0:n_idx - 1), root(nn), blk(0:n_idx - 1))
    parent = [(n, n = 1, nn)]
    owner = 0
    do n = 1, nn
      do k = 1, n_degrees
        i = node_list%node(n)%index(k) - 1
        if (i < 0 .or. i >= n_idx) call fail("a node index outside the operator")
        if (owner(i) == 0) then
          owner(i) = n
        else
          parent(find(n)) = find(owner(i))
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
    call csr_count(blk, n_idx, nn, bm0, bmem)
    bmem = bmem - 1                                   ! indices are 0-based
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

    ! my blocks (those holding an index of mine) and the halo of their patches
    allocate(mark(nn), seen(nn), posh(0:n_idx - 1), posj(0:n_idx - 1))
    mark = 0; seen = 0; stamp = 0; nmy = 0
    allocate(mine(nn))
    do i = i0, i1 - 1
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

    ! the patch solves, per block and slot; Z's rows of mine in CSR (zr0, zc, zv)
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
    do b = 1, nmy
      call patch(mine(b), jl, nj)
      no = bm0(mine(b) + 1) - bm0(mine(b))
      ol = bmem(bm0(mine(b)) : bm0(mine(b) + 1) - 1)
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
        if (info /= 0) call fail("a singular patch mass matrix")
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
    enddo
    call MatDestroySubMatrices(one, sub, ierr)

    ! Z, with M's layout
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
    call MPI_Allreduce(nmy, nsum, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
    if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   local dual basis of B_33: ", nsum, &
      " block patches (summed over ranks), ", ns, " slot(s) each"

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

    !> The patch of block bb: the sorted indices of every block of a node that
    !! shares an element with a node of bb.
    subroutine patch(bb, jl_, nj_)
      integer, intent(in) :: bb
      integer, allocatable, intent(out) :: jl_(:)
      integer, intent(out) :: nj_
      integer :: t, s, x, w, nb_
      integer, allocatable :: bl(:)
      allocate(bl(0)); nb_ = 0
      stamp = stamp + 1
      do t = bn0(bb), bn0(bb + 1) - 1
        do s = ne0(bnod(t)), ne0(bnod(t) + 1) - 1
          do x = 1, n_vertex_max
            w = root(element_list%element(nel(s))%vertex(x))
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
