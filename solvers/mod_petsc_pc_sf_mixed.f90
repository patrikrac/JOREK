module mod_petsc_pc_sf_mixed
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx
  use mod_petsc_pc_blocks,      only: pack_blocks_aij, report_operator_density
  use mod_petsc_pc_sf_solver,   only: suu_form_t, SF_SUU_WJ, SF_SUU_WPJ, sf_split_parts
  implicit none
  private

  !--------------------------------------------------------------------
  !> pair_w of the mixed arms (physics_pc_sf_suu = "wj" / "wpj"): B_22 + W
  !! with W's psi-channel terms taken back out and the psi channel put back
  !! through explicit fields, ASSEMBLED.
  !!
  !! WHY. W composes the psi channel at the continuum level: it assumes the
  !! ideal response psi = -(theta dt / opz) Bpar u and j = Delta* psi. The
  !! Jacobian's Schur complement instead carries
  !!   (a) the projection of Bpar u (only C0) onto the C1 space, M_psi^-1,
  !!   (b) the projection of the Laplacian, M_j^-1,
  !!   (c) the resistive and hyper-resistive damping of the psi response
  !!       (B_13 = theta dt (eta_num K + eta_T M)),
  !! and all three grow with dt -- (a), (b) as (dt v_A / h)^2, (c) as
  !! theta dt eta_num / h^4 against opz -- which is where the "w" arm's outer
  !! count goes at large tstep. The mass inverses are dense, so no single-field
  !! sparse correction of W is exact; and (c) has no h-uniform sparse
  !! single-field surrogate at all (the Dh channel: 60 -> 103 its). Keeping j
  !! explicit -- the pair_psi idea -- makes every row second order and every
  !! block sparse.
  !!
  !! THE TWO FORMS (u, omega first; the solution's psi / j parts are
  !! discarded, the sweep's corrector recomputes them exactly)
  !!
  !!  "wj"  (u, omega, j), psi eliminated by D = (opz M_psi)^-1 lumped to its
  !!        node blocks (small flow: B_11 ~ opz M_psi),
  !!          [[B_22 + W_kc,   B_24,  B_23              ],
  !!           [B_42,          B_44,  0                 ],
  !!           [-B_31 D B_12,  0,     B_33 - B_31 D B_13]]
  !!        W_kc = W's kink + curvature terms. Eliminating j gives the Schur
  !!        complement with B_11 -> D^-1: (b), (c) exact, (a) up to the
  !!        h-uniform equivalence of the node-block mass to M (kappa ~ 58 for
  !!        bicubic Hermite, against ~850 for the point diagonal).
  !!  "wpj" (u, omega, psi, j), the small-flow psi row kept,
  !!          [[B_22 + W_c,  B_24,  B_21,        B_23],
  !!           [B_42,        B_44,  0,           0   ],
  !!           [B_12,        0,     opz M_psi,   B_13],
  !!           [0,           0,     B_31,        B_33]]
  !!        W_c = W's curvature term. Exact up to the small-flow psi row;
  !!        "wpj_flow" keeps B_11 there, i.e. the schur arm's S_uu plus W_c.
  !!
  !! M_psi is B_33: the psi row's mass (amat_11) and the constraint mass
  !! (amat_33) are the same integrand, and every variable carries the same
  !! boundary rows.
  !!
  !! Variants for the gate (physics_pc_sf_suu): wj_diag (D the point diagonal),
  !! wj_nores (no B_31 D B_13: sizes (c)), wj_kd (the kink discrete as well:
  !! B_22 + W_c - B_21 D B_12 and B_23 - B_21 D B_13 in the u row, i.e. exactly
  !! "wpj" with psi eliminated by D).
  !!
  !! STRUCTURE ONCE, NUMBERS PER REBUILD. sfm_build (first build) fixes D's
  !! node groups and every pattern; sfm_refill (every later rebuild, after the
  !! SF gather refilled the B_ij and the element assembly W) recomputes the
  !! values in place, so the packed operator keeps its identity and MUMPS its
  !! symbolic phase.
  !--------------------------------------------------------------------

  Mat, save, public     :: sfm_op          !< the packed mixed pair_w
  integer, save, public :: sfm_nf = 0      !< its fields: 3 (wj) or 4 (wpj)

  type(suu_form_t), save :: fm
  Mat, save :: suu                         !< B_22 + W (W without its psi-channel terms)
  Mat, save :: dpsi                        !< wj: the lumped (opz M_psi)^-1
  Mat, save :: p12, p13                    !< wj: B_31 D B_12, B_31 D B_13
  Mat, save :: jju, jjj                    !< wj: -B_31 D B_12, B_33 - B_31 D B_13
  Mat, save :: q12, q13, juj               !< wj_kd: B_21 D B_12, B_21 D B_13, B_23 - B_21 D B_13
  Mat, save :: mpsi                        !< wpj: opz M_psi
  logical, save :: packed = .false.

  !--- D's groups: the local rows (0-based, rank-local) of each diagonal block
  integer, allocatable, save :: gptr(:), grow(:)

  public :: sfm_build, sfm_refill

contains

  !--------------------------------------------------------------------
  !> First build: D's groups, the blocks, the packed operator and its gate.
  !--------------------------------------------------------------------
  subroutine sfm_build(form, comm, my_id)
    type(suu_form_t), intent(in) :: form
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr

    fm = form
    call MatDuplicate(g_ctx%B_22, MAT_COPY_VALUES, suu, ierr)
    call MatAXPY(suu, 1.0d0, g_ctx%W_force, DIFFERENT_NONZERO_PATTERN, ierr)

    select case (fm%form)
    case (SF_SUU_WJ)
      sfm_nf = 3
      call d_groups(comm, my_id)
      call d_create(comm)
      call d_fill(comm, my_id)
      call MatMatMatMult(g_ctx%B_31, dpsi, g_ctx%B_12, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, p12, ierr)
      call MatDuplicate(p12, MAT_COPY_VALUES, jju, ierr)
      call MatScale(jju, -1.0d0, ierr)
      if (fm%no_res) then
        call MatDuplicate(g_ctx%B_33, MAT_COPY_VALUES, jjj, ierr)
      else
        call MatMatMatMult(g_ctx%B_31, dpsi, g_ctx%B_13, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, p13, ierr)
        call MatDuplicate(p13, MAT_COPY_VALUES, jjj, ierr)
        call MatScale(jjj, -1.0d0, ierr)
        call MatAXPY(jjj, 1.0d0, g_ctx%B_33, DIFFERENT_NONZERO_PATTERN, ierr)
      endif
      if (fm%kink_disc) then
        call MatMatMatMult(g_ctx%B_21, dpsi, g_ctx%B_12, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, q12, ierr)
        call MatMatMatMult(g_ctx%B_21, dpsi, g_ctx%B_13, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, q13, ierr)
        call MatAXPY(suu, -1.0d0, q12, DIFFERENT_NONZERO_PATTERN, ierr)
        call MatDuplicate(q13, MAT_COPY_VALUES, juj, ierr)
        call MatScale(juj, -1.0d0, ierr)
        call MatAXPY(juj, 1.0d0, g_ctx%B_23, DIFFERENT_NONZERO_PATTERN, ierr)
      endif
    case (SF_SUU_WPJ)
      sfm_nf = 4
      if (.not. fm%flow) then
        call MatDuplicate(g_ctx%B_33, MAT_COPY_VALUES, mpsi, ierr)
        call MatScale(mpsi, opz(), ierr)
      endif
    end select

    call pack_op()
    call gate(my_id)
    call report_operator_density(sfm_op, trim(label()), my_id)
  end subroutine sfm_build

  !--------------------------------------------------------------------
  !> Every later rebuild: the B_ij and W hold the new values; recompute the
  !! products and refill the packed operator in place.
  !--------------------------------------------------------------------
  subroutine sfm_refill(comm, my_id)
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr

    call MatZeroEntries(suu, ierr)
    call MatAXPY(suu, 1.0d0, g_ctx%B_22, SUBSET_NONZERO_PATTERN, ierr)
    call MatAXPY(suu, 1.0d0, g_ctx%W_force, SUBSET_NONZERO_PATTERN, ierr)

    select case (fm%form)
    case (SF_SUU_WJ)
      call d_fill(comm, my_id)
      call MatMatMatMult(g_ctx%B_31, dpsi, g_ctx%B_12, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, p12, ierr)
      call MatCopy(p12, jju, SAME_NONZERO_PATTERN, ierr)
      call MatScale(jju, -1.0d0, ierr)
      call MatZeroEntries(jjj, ierr)
      if (.not. fm%no_res) then
        call MatMatMatMult(g_ctx%B_31, dpsi, g_ctx%B_13, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, p13, ierr)
        call MatAXPY(jjj, -1.0d0, p13, SUBSET_NONZERO_PATTERN, ierr)
      endif
      call MatAXPY(jjj, 1.0d0, g_ctx%B_33, SUBSET_NONZERO_PATTERN, ierr)
      if (fm%kink_disc) then
        call MatMatMatMult(g_ctx%B_21, dpsi, g_ctx%B_12, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, q12, ierr)
        call MatMatMatMult(g_ctx%B_21, dpsi, g_ctx%B_13, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, q13, ierr)
        call MatAXPY(suu, -1.0d0, q12, SUBSET_NONZERO_PATTERN, ierr)
        call MatZeroEntries(juj, ierr)
        call MatAXPY(juj, -1.0d0, q13, SUBSET_NONZERO_PATTERN, ierr)
        call MatAXPY(juj, 1.0d0, g_ctx%B_23, SUBSET_NONZERO_PATTERN, ierr)
      endif
    case (SF_SUU_WPJ)
      if (.not. fm%flow) then
        call MatCopy(g_ctx%B_33, mpsi, SAME_NONZERO_PATTERN, ierr)
        call MatScale(mpsi, opz(), ierr)
      endif
    end select

    call pack_op()
  end subroutine sfm_refill

  !> The packed operator from the current blocks (first call: its pattern).
  subroutine pack_op()
    Mat :: blk(4, 4)
    logical :: have(4, 4)
    integer :: nf, comm_
    PetscErrorCode :: ierr

    nf = sfm_nf
    have = .false.
    blk(1, 1) = suu;         have(1, 1) = .true.
    blk(1, 2) = g_ctx%B_24;  have(1, 2) = .true.
    blk(2, 1) = g_ctx%B_42;  have(2, 1) = .true.
    blk(2, 2) = g_ctx%B_44;  have(2, 2) = .true.
    if (fm%form == SF_SUU_WJ) then
      blk(1, 3) = g_ctx%B_23;  have(1, 3) = .true.
      if (fm%kink_disc) blk(1, 3) = juj
      blk(3, 1) = jju;         have(3, 1) = .true.
      blk(3, 3) = jjj;         have(3, 3) = .true.
    else
      blk(1, 3) = g_ctx%B_21;  have(1, 3) = .true.
      blk(1, 4) = g_ctx%B_23;  have(1, 4) = .true.
      blk(3, 1) = g_ctx%B_12;  have(3, 1) = .true.
      if (fm%flow) then
        blk(3, 3) = g_ctx%B_11
      else
        blk(3, 3) = mpsi
      endif
      have(3, 3) = .true.
      blk(3, 4) = g_ctx%B_13;  have(3, 4) = .true.
      blk(4, 3) = g_ctx%B_31;  have(4, 3) = .true.
      blk(4, 4) = g_ctx%B_33;  have(4, 4) = .true.
    endif
    call PetscObjectGetComm(g_ctx%B_22, comm_, ierr)
    call pack_blocks_aij(blk(1:nf, 1:nf), have(1:nf, 1:nf), sfm_op, packed, comm_)
  end subroutine pack_op

  !> 1 + zeta, as the Jacobian's element matrix forms it (mod_elt_matrix_fft)
  real*8 function opz()
    use phys_module, only: time_evol_zeta, tstep, tstep_prev
    opz = 1.0d0 + time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  end function opz

  !--------------------------------------------------------------------
  !> D's groups: the rank's rows of each node, all harmonics (an axis node
  !! shares its indices with the other axis nodes, so the axis is one group).
  !! A node whose indices straddle a rank boundary keeps its local part.
  !--------------------------------------------------------------------
  subroutine d_groups(comm, my_id)
    use nodes_elements, only: node_list
    use mod_parameters, only: n_tor, n_degrees
    integer, intent(in) :: comm, my_id
    integer, allocatable :: gmin(:), cnt(:), key(:), ord(:)
    PetscInt :: rs, re
    integer :: i, k, nidx, nrow, ng, r, g
    PetscErrorCode :: ierr

    call MatGetOwnershipRange(g_ctx%B_33, rs, re, ierr)
    nidx = 0
    do i = 1, node_list%n_nodes
      nidx = max(nidx, maxval(node_list%node(i)%index(1:n_degrees)))
    enddo
    allocate(gmin(nidx))
    gmin = huge(1)
    do i = 1, node_list%n_nodes
      k = minval(node_list%node(i)%index(1:n_degrees))
      gmin(node_list%node(i)%index(1:n_degrees)) = min(gmin(node_list%node(i)%index(1:n_degrees)), k)
    enddo

    !--- local rows, keyed by their index's group, stable within a group
    nrow = int(re - rs)
    allocate(key(nrow), ord(nrow))
    do r = 1, nrow
      key(r) = gmin(int((rs + r - 1) / n_tor) + 1)
      ord(r) = r
    enddo
    call sort_by_key()

    ng = 0
    allocate(cnt(nrow + 1))
    do r = 1, nrow
      if (r == 1) then
        ng = 1; cnt(1) = 1
      else if (key(ord(r)) /= key(ord(r - 1))) then
        ng = ng + 1; cnt(ng) = 1
      else
        cnt(ng) = cnt(ng) + 1
      endif
    enddo
    allocate(gptr(ng + 1), grow(nrow))
    gptr(1) = 0
    do g = 1, ng
      gptr(g + 1) = gptr(g) + cnt(g)
    enddo
    grow = ord - 1

    block
      integer :: gs(2)
      gs = [ng, maxval(cnt(1:ng))]
      call MPI_Allreduce(MPI_IN_PLACE, gs(1), 1, MPI_INTEGER, MPI_SUM, comm, ierr)
      call MPI_Allreduce(MPI_IN_PLACE, gs(2), 1, MPI_INTEGER, MPI_MAX, comm, ierr)
      if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   pair_w mixed: psi mass lumped to ", &
        gs(1), " node blocks (largest ", gs(2), " rows)"
    end block

  contains

    !> insertion-free merge sort of ord by key (stable)
    subroutine sort_by_key()
      integer, allocatable :: tmp(:)
      integer :: w, lo, mid, hi, a, b, t
      allocate(tmp(nrow))
      w = 1
      do while (w < nrow)
        lo = 1
        do while (lo <= nrow)
          mid = min(lo + w - 1, nrow); hi = min(lo + 2 * w - 1, nrow)
          a = lo; b = mid + 1; t = lo
          do while (a <= mid .and. b <= hi)
            if (key(ord(b)) < key(ord(a))) then
              tmp(t) = ord(b); b = b + 1
            else
              tmp(t) = ord(a); a = a + 1
            endif
            t = t + 1
          enddo
          do while (a <= mid)
            tmp(t) = ord(a); a = a + 1; t = t + 1
          enddo
          do while (b <= hi)
            tmp(t) = ord(b); b = b + 1; t = t + 1
          enddo
          lo = lo + 2 * w
        enddo
        ord = tmp
        w = 2 * w
      enddo
    end subroutine sort_by_key

  end subroutine d_groups

  !> D's pattern: each group's block, restricted to the entries its inverse
  !! can have -- the connected components of M_psi's block (the harmonics do
  !! not couple in the mass, so D stays |n|-diagonal like every SF block).
  subroutine d_create(comm)
    integer, intent(in) :: comm
    PetscInt, allocatable :: dnz(:), onz(:)
    real*8, allocatable :: a(:, :)
    integer, allocatable :: comp(:)
    PetscInt :: rs, re, n
    integer :: g, nb, i, j
    PetscErrorCode :: ierr

    call MatGetOwnershipRange(g_ctx%B_33, rs, re, ierr)
    n = re - rs
    allocate(dnz(n), onz(n))
    dnz = 0; onz = 0
    do g = 1, size(gptr) - 1
      nb = gptr(g + 1) - gptr(g)
      call block_of(g, a, comp)
      do i = 1, nb
        if (fm%diag_mass) then
          dnz(grow(gptr(g) + i) + 1) = 1
        else
          dnz(grow(gptr(g) + i) + 1) = count([(comp(j) == comp(i), j = 1, nb)])
        endif
      enddo
    enddo
    call MatCreate(comm, dpsi, ierr)
    call MatSetSizes(dpsi, n, n, PETSC_DETERMINE, PETSC_DETERMINE, ierr)
    call MatSetType(dpsi, MATMPIAIJ, ierr)
    call MatMPIAIJSetPreallocation(dpsi, 0, dnz, 0, onz, ierr)
    call MatSetOption(dpsi, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE, ierr)
  end subroutine d_create

  !--------------------------------------------------------------------
  !> D's values: each group's block of opz M_psi (a component at a time)
  !! inverted by LU, every inverse checked (||A X - I||_max <= 1e-8).
  !--------------------------------------------------------------------
  subroutine d_fill(comm, my_id)
    integer, intent(in) :: comm, my_id
    real*8, allocatable :: a(:, :), x(:, :), ac(:, :), xc(:, :)
    integer, allocatable :: comp(:), piv(:), sel(:)
    PetscInt :: rs, re
    PetscInt, allocatable :: gr(:)
    integer :: g, nb, c, m, info, i, nbad
    real*8 :: err, emax, s
    PetscErrorCode :: ierr
    external :: dgesv

    call MatGetOwnershipRange(g_ctx%B_33, rs, re, ierr)
    s = opz()
    nbad = 0; emax = 0.d0
    do g = 1, size(gptr) - 1
      nb = gptr(g + 1) - gptr(g)
      gr = [(rs + grow(gptr(g) + i), i = 1, nb)]
      call block_of(g, a, comp)
      a = s * a
      if (fm%diag_mass) then
        do i = 1, nb
          call MatSetValue(dpsi, gr(i), gr(i), 1.0d0 / a(i, i), INSERT_VALUES, ierr)
        enddo
        cycle
      endif
      do c = 1, maxval(comp)
        sel = pack([(i, i = 1, nb)], comp == c)
        m = size(sel)
        if (m == 0) cycle
        ac = a(sel, sel)
        xc = 0.d0 * ac
        do i = 1, m
          xc(i, i) = 1.d0
        enddo
        if (allocated(piv)) deallocate(piv)
        allocate(piv(m))
        x = ac
        call dgesv(m, m, x, m, piv, xc, m, info)
        if (info /= 0) then
          nbad = nbad + 1
          cycle
        endif
        ! the ZBIG boundary rows make A badly scaled, but they are diagonally
        ! dominant and pivoted first, so A X = I holds to round-off anyway
        err = maxval(abs(matmul(ac, xc) - ident(m)))
        emax = max(emax, err)
        ! PETSc reads values row-major
        call MatSetValues(dpsi, int(m, kind(rs)), gr(sel), int(m, kind(rs)), gr(sel), &
                          reshape(transpose(xc), [m * m]), &
                          INSERT_VALUES, ierr)
      enddo
    enddo
    call MatAssemblyBegin(dpsi, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(dpsi, MAT_FINAL_ASSEMBLY, ierr)

    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, comm, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, emax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, ierr)
    if (nbad > 0 .or. emax > 1.d-8) then
      if (my_id == 0) write(*,'(A,I0,A,ES9.2)') "[Physics PC]   FATAL: pair_w mixed: psi mass blocks: ", &
        nbad, " singular, max ||A X - I|| = ", emax
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif

  contains

    function ident(k) result(e)
      integer, intent(in) :: k
      real*8 :: e(k, k)
      integer :: q
      e = 0.d0
      do q = 1, k
        e(q, q) = 1.d0
      enddo
    end function ident

  end subroutine d_fill

  !> Group g's dense block of M_psi (= B_33) and the connected components
  !! of its nonzero graph (comp(i) = component of the group's i-th row).
  subroutine block_of(g, a, comp)
    integer, intent(in) :: g
    real*8, allocatable, intent(out) :: a(:, :)
    integer, allocatable, intent(out) :: comp(:)
    PetscInt, allocatable :: gr(:)
    PetscInt :: rs, re
    integer :: nb, i, j, c
    logical :: changed
    PetscErrorCode :: ierr

    call MatGetOwnershipRange(g_ctx%B_33, rs, re, ierr)
    nb = gptr(g + 1) - gptr(g)
    gr = [(rs + grow(gptr(g) + i), i = 1, nb)]
    allocate(a(nb, nb), comp(nb))
    call MatGetValues(g_ctx%B_33, int(nb, kind(rs)), gr, int(nb, kind(rs)), gr, a, ierr)
    a = transpose(a)                         ! PETSc returns row-major
    comp = [(i, i = 1, nb)]
    changed = .true.
    do while (changed)                       ! label propagation, nb <= 4 n_tor
      changed = .false.
      do i = 1, nb
        do j = 1, nb
          if (a(i, j) /= 0.d0 .or. a(j, i) /= 0.d0) then
            c = min(comp(i), comp(j))
            if (comp(i) /= c .or. comp(j) /= c) then
              comp(i) = c; comp(j) = c; changed = .true.
            endif
          endif
        enddo
      enddo
    enddo
    ! renumber 1..ncomp
    block
      integer, allocatable :: lab(:)
      integer :: nc
      allocate(lab(nb)); lab = 0; nc = 0
      do i = 1, nb
        if (lab(comp(i)) == 0) then
          nc = nc + 1; lab(comp(i)) = nc
        endif
      enddo
      comp = lab(comp)
    end block
  end subroutine block_of

  !--------------------------------------------------------------------
  !> First-build gate: the packed operator against its definition, block by
  !! block through MatMult on the separate blocks, for a random x. Checks
  !! the pack, the products and every sign. No flag: once per run.
  !--------------------------------------------------------------------
  subroutine gate(my_id)
    integer, intent(in) :: my_id
    Vec :: x, y, xs(4), ys(4), t1, t2
    integer :: k, nf
    real*8 :: e(2), d
    PetscErrorCode :: ierr

    nf = sfm_nf
    call MatCreateVecs(sfm_op, x, y, ierr)
    call VecSetRandom(x, PETSC_NULL_RANDOM, ierr)
    call MatMult(sfm_op, x, y, ierr)
    do k = 1, nf
      call MatCreateVecs(g_ctx%B_33, xs(k), ys(k), ierr)
    enddo
    call MatCreateVecs(g_ctx%B_33, t1, t2, ierr)
    call sf_split_parts(x, xs(1:nf), .false.)

    ! rows 1, 2: (u, omega); row 1 from B_22 and W, not from the stored suu
    call MatMult(g_ctx%B_22, xs(1), ys(1), ierr)
    call MatMultAdd(g_ctx%W_force, xs(1), ys(1), ys(1), ierr)
    call MatMultAdd(g_ctx%B_24, xs(2), ys(1), ys(1), ierr)
    call MatMult(g_ctx%B_42, xs(1), ys(2), ierr)
    call MatMultAdd(g_ctx%B_44, xs(2), ys(2), ys(2), ierr)
    if (fm%form == SF_SUU_WJ) then
      call MatMultAdd(g_ctx%B_23, xs(3), ys(1), ys(1), ierr)
      ! j row: -B_31 D (B_12 x_u + B_13 x_j) + B_33 x_j
      call MatMult(g_ctx%B_12, xs(1), t1, ierr)
      if (.not. fm%no_res) call MatMultAdd(g_ctx%B_13, xs(3), t1, t1, ierr)
      call MatMult(dpsi, t1, t2, ierr)
      call MatMult(g_ctx%B_31, t2, ys(3), ierr)
      if (fm%kink_disc) then                  ! u row: - B_21 D (B_12 x_u + B_13 x_j)
        call MatMult(g_ctx%B_21, t2, t1, ierr)
        call VecAXPY(ys(1), -1.0d0, t1, ierr)
      endif
      call VecScale(ys(3), -1.0d0, ierr)
      call MatMultAdd(g_ctx%B_33, xs(3), ys(3), ys(3), ierr)
    else
      call MatMultAdd(g_ctx%B_21, xs(3), ys(1), ys(1), ierr)
      call MatMultAdd(g_ctx%B_23, xs(4), ys(1), ys(1), ierr)
      call MatMult(g_ctx%B_12, xs(1), ys(3), ierr)
      if (fm%flow) then
        call MatMultAdd(g_ctx%B_11, xs(3), ys(3), ys(3), ierr)
      else
        call MatMult(g_ctx%B_33, xs(3), t1, ierr)
        call VecAXPY(ys(3), opz(), t1, ierr)
      endif
      call MatMultAdd(g_ctx%B_13, xs(4), ys(3), ys(3), ierr)
      call MatMult(g_ctx%B_31, xs(3), ys(4), ierr)
      call MatMultAdd(g_ctx%B_33, xs(4), ys(4), ys(4), ierr)
    endif

    call sf_split_parts(x, ys(1:nf), .true.)   ! x <- the reference, packed
    call VecAXPY(x, -1.0d0, y, ierr)
    call VecNorm(x, NORM_2, e(1), ierr)
    call VecNorm(y, NORM_2, e(2), ierr)
    d = e(1) / max(e(2), 1.d-300)
    if (my_id == 0) write(*,'(A,ES10.3)') &
      "[Physics PC]   pair_w mixed gate (packed vs block matvecs): ", d
    if (d > 1.d-12) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: the packed mixed pair_w is wrong."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    call VecDestroy(x, ierr); call VecDestroy(y, ierr)
    call VecDestroy(t1, ierr); call VecDestroy(t2, ierr)
    do k = 1, nf
      call VecDestroy(xs(k), ierr); call VecDestroy(ys(k), ierr)
    enddo

  end subroutine gate

  !> What the packed operator carries, for the log.
  character(len=64) function label()
    select case (fm%form)
    case (SF_SUU_WJ)
      label = "pair_w mixed (u,omega,j), psi by lumped mass"
      if (fm%diag_mass) label = "pair_w mixed (u,omega,j), psi by diagonal mass"
      if (fm%no_res)    label = "pair_w mixed (u,omega,j), no resistive psi"
      if (fm%kink_disc) label = "pair_w mixed (u,omega,j), kink discrete"
    case default
      label = "pair_w mixed (u,omega,psi,j), small-flow psi"
      if (fm%flow) label = "pair_w mixed (u,omega,psi,j), psi row B_11"
    end select
  end function label

#endif
end module mod_petsc_pc_sf_mixed
