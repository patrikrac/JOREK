module mod_petsc_pc_blocks
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  !--------------------------------------------------------------------
  !> Block primitives of the SF preconditioner: extraction, packing, scaling
  !! and the variable split of vectors. The extraction keeps every slot pair
  !! with harm_all, else only those of one |n| group (harm_kept).
  !--------------------------------------------------------------------

  public :: pc_print_block_setup
  public :: extract_sub_blocks_h
  public :: pack_pair_aij, pack_blocks_aij
  public :: make_field_block_scale
  public :: decouple_penalty_rows
  public :: split_vars, merge_vars
  public :: harm_all, harm_kept

  !> Cross-|n| coupling of the block extraction: .true. = every slot pair,
  !! .false. = one |n| group only (cos and sin of one n together).
  logical, save :: harm_all = .false.

contains

  !> Does the block extraction keep the entry between toroidal slots m and q?
  !! Slot m belongs to |n| group (m+1)/2 (slot 0 = n = 0, then cos/sin pairs).
  pure logical function harm_kept(m, q)
    integer, intent(in) :: m, q
    harm_kept = harm_all .or. ((m + 1) / 2 == (q + 1) / 2)
  end function harm_kept

  !--------------------------------------------------------------------
  !> Print one coherent setup line on rank 0:  "[Physics PC]   <label>: <method>"
  !! Used by the block-KSP setup routines so each reports what it configured.
  !--------------------------------------------------------------------
  subroutine pc_print_block_setup(comm, label, method)
    integer,          intent(in) :: comm
    character(len=*), intent(in) :: label, method

    integer        :: rank
    PetscErrorCode :: ierr

    call MPI_Comm_rank(comm, rank, ierr)
    if (rank == 0) write(*,'(A)') "[Physics PC]   " // trim(label) // ": " // trim(method)
  end subroutine pc_print_block_setup

  !--------------------------------------------------------------------
  !> Extract the blocks Mb(k) = A(eqs(k), vrs(k)) of the full system matrix in
  !! ONE pass over A_full's rows: each equation row is read once (MatGetRow on
  !! JOREK's BAIJ matrix expands the whole n_var*n_tor-wide block row) and its
  !! entries are dispatched to every requested block of that equation. Every
  !! entry is kept with harm_all, else only those of one |n| group
  !! (harm_kept; slot m is in group (m+1)/2, cos and sin of one n together).
  !!
  !! The blocks are read straight out of A_full's rows (MatGetRow), so no
  !! unfiltered copy is ever held, and A_full may be JOREK's own BAIJ matrix.
  !! Layout (create_variable_index_sets): full row node*bs + (v-1)*n_tor + m,
  !! sub-block row node*n_tor + m, bs = n_var*n_tor.
  !!
  !! Each Mb(k) keeps its identity across rebuilds: it is refilled into the
  !! pattern fixed on the first build.
  !--------------------------------------------------------------------
  subroutine extract_sub_blocks_h(A_full, eqs, vrs, Mb, first_time)
    use mod_parameters, only: n_tor, n_var
    Mat, intent(in)     :: A_full
    integer, intent(in) :: eqs(:), vrs(:)
    Mat, intent(inout)  :: Mb(:)
    logical, intent(in) :: first_time
    PetscErrorCode :: ierr
    PetscInt :: i, ncols, rstart, rend, m, k, q, bs, nloc, nsub, nglob
    PetscInt :: sr, sc, cstart, cend, node, cb
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: dcnt(:,:), ocnt(:,:), cc(:,:)
    PetscScalar, allocatable :: vv(:,:)
    integer, allocatable :: tgt(:,:), c(:)
    integer :: comm, nb, e, v, kb, pass

    nb = size(eqs)

    bs = n_var * n_tor
    call MatGetOwnershipRange(A_full, rstart, rend, ierr)
    call MatGetSize(A_full, nglob, PETSC_NULL_INTEGER, ierr)
    nloc   = ((rend - rstart) / bs) * n_tor
    nsub   = (nglob / bs) * n_tor
    cstart = (rstart / bs) * n_tor
    cend   = cstart + nloc
    allocate(tgt(n_var, n_var), c(nb))
    tgt = 0
    do kb = 1, nb
      tgt(eqs(kb), vrs(kb)) = kb
    enddo

    if (first_time) then
      allocate(dcnt(nloc, nb), ocnt(nloc, nb))
      dcnt = 0; ocnt = 0
    else
      do kb = 1, nb
        call MatZeroEntries(Mb(kb), ierr)
      enddo
    endif
    allocate(cc(bs * 64, nb), vv(bs * 64, nb))

    do pass = merge(1, 2, first_time), 2          ! 1 = count (first build), 2 = fill
      do node = rstart / bs, rend / bs - 1
        do e = 1, n_var
          if (all(tgt(e, :) == 0)) cycle
          do m = 0, n_tor - 1
            i  = node * bs + (e - 1) * n_tor + m
            sr = node * n_tor + m
            call MatGetRow(A_full, i, ncols, cols, vals, ierr)
            if (ncols > size(cc, 1)) then
              deallocate(cc, vv); allocate(cc(ncols, nb), vv(ncols, nb))
            endif
            c = 0
            do k = 1, ncols
              cb = mod(cols(k), bs)
              v  = int(cb / n_tor) + 1
              kb = tgt(e, v)
              if (kb == 0) cycle
              q = cb - (v - 1) * n_tor
              if (.not. harm_kept(int(m), int(q))) cycle
              sc = (cols(k) / bs) * n_tor + q
              if (pass == 1) then
                if (sc >= cstart .and. sc < cend) then
                  dcnt(sr - cstart + 1, kb) = dcnt(sr - cstart + 1, kb) + 1
                else
                  ocnt(sr - cstart + 1, kb) = ocnt(sr - cstart + 1, kb) + 1
                endif
              else
                c(kb) = c(kb) + 1; cc(c(kb), kb) = sc; vv(c(kb), kb) = vals(k)
              endif
            enddo
            call MatRestoreRow(A_full, i, ncols, cols, vals, ierr)
            if (pass == 2) then
              do kb = 1, nb
                if (c(kb) > 0) call MatSetValues(Mb(kb), 1_4, [sr], c(kb), cc(1:c(kb), kb), &
                                                 vv(1:c(kb), kb), INSERT_VALUES, ierr)
              enddo
            endif
          enddo
        enddo
      enddo
      if (pass == 1) then
        call PetscObjectGetComm(A_full, comm, ierr)
        do kb = 1, nb
          call MatCreate(comm, Mb(kb), ierr)
          call MatSetSizes(Mb(kb), nloc, nloc, nsub, nsub, ierr)
          call MatSetType(Mb(kb), MATMPIAIJ, ierr)
          call MatMPIAIJSetPreallocation(Mb(kb), PETSC_DEFAULT_INTEGER, dcnt(:, kb), &
                                         PETSC_DEFAULT_INTEGER, ocnt(:, kb), ierr)
        enddo
        deallocate(dcnt, ocnt)
      endif
    enddo
    deallocate(cc, vv, tgt, c)
    do kb = 1, nb
      call MatAssemblyBegin(Mb(kb), MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(Mb(kb), MAT_FINAL_ASSEMBLY, ierr)
      if (first_time) call MatSetOption(Mb(kb), MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
    enddo
  end subroutine extract_sub_blocks_h

  !--------------------------------------------------------------------
  !> C = [[A11, A12], [A21, A22]] as one MPIAIJ in the packed layout of
  !! MatConvert(MatNest -> MPIAIJ); pack_blocks_aij with two fields.
  !--------------------------------------------------------------------
  subroutine pack_pair_aij(A11, A12, A21, A22, C, ready, comm)
    Mat, intent(in)        :: A11, A12, A21, A22
    Mat, intent(inout)     :: C
    logical, intent(inout) :: ready
    integer, intent(in)    :: comm
    Mat :: blk(2, 2)
    blk(1, 1) = A11; blk(1, 2) = A12; blk(2, 1) = A21; blk(2, 2) = A22
    call pack_blocks_aij(blk, reshape([.true., .true., .true., .true.], [2, 2]), C, ready, comm)
  end subroutine pack_pair_aij

  !--------------------------------------------------------------------
  !> C = [A_fg], f, g = 1..nf, as one MPIAIJ in the packed layout of
  !! MatConvert(MatNest -> MPIAIJ): rank r owns [its field-1 rows | ... | its
  !! field-nf rows], and a field-g column c owned by rank p sits at packed
  !! column c + sum_{h<g} rg_h(p+1) + sum_{h>g} rg_h(p), rg_h = field h's
  !! ownership starts. Block (f, g) is read only where have(f, g); an absent
  !! block is zero. Replaces that MatConvert, whose parallel path
  !! ISAllGathers every global column index onto every rank -- O(N) memory and
  !! traffic per rank, twice per PC rebuild.
  !!
  !! First call (ready = .false.): exact preallocation, and the pattern is
  !! frozen (MAT_NEW_NONZERO_LOCATION_ERR). Later calls refill the values in
  !! place, so C keeps its identity across rebuilds and every consumer (the
  !! GMG's PtAP, the coarse/axis LUs, MUMPS) reuses its symbolic phase. The
  !! stored values and each row's column order are those of the MatConvert.
  !! Field f's row and column layout is taken from its diagonal block
  !! blk(f, f), which must therefore be present.
  !--------------------------------------------------------------------
  subroutine pack_blocks_aij(blk, have, C, ready, comm)
    Mat, intent(in)        :: blk(:, :)
    logical, intent(in)    :: have(:, :)
    Mat, intent(inout)     :: C
    logical, intent(inout) :: ready
    integer, intent(in)    :: comm
    PetscErrorCode :: ierr
    PetscInt :: ps, r, ncols, k, pr, nloc, ng
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: dnz(:), onz(:), pc_(:), s(:), n(:)
    integer, allocatable :: rg(:, :)
    PetscInt, allocatable :: coff(:, :)   !< packed column offset of field g's columns on rank p
    integer :: np, me, mpierr, pass, nf, f, g, p

    nf = size(blk, 1)
    call MPI_Comm_size(comm, np, mpierr)
    call MPI_Comm_rank(comm, me, mpierr)
    allocate(s(nf), n(nf), rg(0:np, nf), coff(0:np - 1, nf))
    do f = 1, nf
      call MatGetOwnershipRange(blk(f, f), s(f), k, ierr)
      n(f) = k - s(f)
      call MPI_Allgather(int(s(f)), 1, MPI_INTEGER, rg(0:np - 1, f), 1, MPI_INTEGER, comm, mpierr)
      call MatGetSize(blk(f, f), k, PETSC_NULL_INTEGER, ierr); rg(np, f) = int(k)
    enddo
    do p = 0, np - 1
      do g = 1, nf
        coff(p, g) = 0
        do f = 1, nf
          if (f < g) coff(p, g) = coff(p, g) + rg(p + 1, f)
          if (f > g) coff(p, g) = coff(p, g) + rg(p, f)
        enddo
      enddo
    enddo
    ps   = sum(s)                                   ! first packed row of this rank
    nloc = sum(n)
    ng   = sum(rg(np, :))

    if (.not. ready) then
      allocate(dnz(nloc), onz(nloc))
      dnz = 0; onz = 0
    else
      call MatZeroEntries(C, ierr)
    endif
    allocate(pc_(64))
    ! pass 1 counts (first call only), pass 2 inserts
    do pass = merge(2, 1, ready), 2
      pr = ps
      do f = 1, nf
        do r = 0, n(f) - 1
          do g = 1, nf
            if (have(f, g)) call put_row(blk(f, g), s(f) + r, g, pr, pass)
          enddo
          pr = pr + 1
        enddo
      enddo
      if (pass == 1) then
        call MatCreate(comm, C, ierr)
        call MatSetSizes(C, nloc, nloc, ng, ng, ierr)
        call MatSetType(C, MATMPIAIJ, ierr)
        call MatMPIAIJSetPreallocation(C, PETSC_DEFAULT_INTEGER, dnz, PETSC_DEFAULT_INTEGER, onz, ierr)
        ! every packed row is the rank's own: assembly skips the stash exchange
        call MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
        deallocate(dnz, onz)
      endif
    enddo
    call MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY, ierr)
    if (.not. ready) call MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
    ready = .true.
    deallocate(s, n, rg, coff, pc_)

  contains

    !> Row `row` of block Ab (columns in field fc) into packed row prow:
    !! pass 1 counts diagonal/off-diagonal entries, pass 2 inserts.
    subroutine put_row(Ab, row, fc, prow, pass_)
      Mat, intent(in)      :: Ab
      PetscInt, intent(in) :: row, prow
      integer, intent(in)  :: fc, pass_
      PetscInt :: q, cg
      integer :: p_
      call MatGetRow(Ab, row, ncols, cols, vals, ierr)
      if (ncols > size(pc_)) then
        deallocate(pc_); allocate(pc_(2 * ncols))
      endif
      p_ = me
      do q = 1, ncols
        cg = cols(q)
        if (cg < rg(p_, fc) .or. cg >= rg(p_ + 1, fc)) p_ = owner(rg(:, fc), int(cg))
        pc_(q) = cg + coff(p_, fc)
      enddo
      if (pass_ == 1) then
        do q = 1, ncols
          if (pc_(q) >= ps .and. pc_(q) < ps + nloc) then
            dnz(prow - ps + 1) = dnz(prow - ps + 1) + 1
          else
            onz(prow - ps + 1) = onz(prow - ps + 1) + 1
          endif
        enddo
      else if (ncols > 0) then
        call MatSetValues(C, 1_4, [prow], ncols, pc_(1:ncols), vals(1:ncols), INSERT_VALUES, ierr)
      endif
      call MatRestoreRow(Ab, row, ncols, cols, vals, ierr)
    end subroutine put_row

    !> Rank owning index ix of a field with ownership starts rg_(0:np).
    integer function owner(rg_, ix)
      integer, intent(in) :: rg_(0:), ix
      integer :: lo, hi, mid
      lo = 0; hi = np - 1
      do while (lo < hi)
        mid = (lo + hi + 1) / 2
        if (rg_(mid) <= ix) then
          lo = mid
        else
          hi = mid - 1
        endif
      enddo
      owner = lo
    end function owner

  end subroutine pack_blocks_aij

  !--------------------------------------------------------------------
  !> Symmetric block scaling of a packed operator, A <- D A D with
  !! D = diag(I, s_2 I, ..., s_nf I), s_f = sqrt(mean|diag| of field 1 /
  !! mean|diag| of field f). The caller solves (D A D) z = D b and takes
  !! x = D z: an exact similarity that only evens out the scales of the
  !! operator and mass blocks (pair_w's spread grows with dt). One scalar per
  !! field keeps the block structure the smoothers rely on.
  !--------------------------------------------------------------------
  subroutine make_field_block_scale(A, nloc, dvec, comm, my_id, label)
    Mat, intent(inout)           :: A
    PetscInt, intent(in)         :: nloc(:)  !< LOCAL rows of each field, in pack order
    Vec, intent(inout)           :: dvec     !< out: D, kept for the apply
    integer, intent(in)          :: comm
    integer, intent(in)          :: my_id
    character(len=*), intent(in) :: label

    PetscErrorCode :: ierr
    integer   :: kk, f, nf, mpierr
    PetscScalar, pointer :: dptr(:)
    real*8, allocatable :: acc(:), m(:), sc(:)
    integer, allocatable :: fld(:)
    PetscReal :: dmin, dmax
    Vec       :: chk
    character(len=256) :: ss, sm
    character(len=16)  :: t1

    nf = size(nloc)
    allocate(acc(2 * nf), m(nf), sc(nf), fld(sum(nloc)))
    kk = 0
    do f = 1, nf
      fld(kk + 1:kk + nloc(f)) = f
      kk = kk + int(nloc(f))
    enddo

    call MatCreateVecs(A, dvec, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, dvec, ierr)

    !--- mean |diag| of each field, over owned rows, then reduced.
    acc = 0.d0
    call VecGetArray(dvec, dptr, ierr)
    do kk = 1, size(fld)
      acc(fld(kk))      = acc(fld(kk)) + abs(dptr(kk))
      acc(nf + fld(kk)) = acc(nf + fld(kk)) + 1.d0
    enddo
    call VecRestoreArray(dvec, dptr, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, acc, 2 * nf, MPI_DOUBLE_PRECISION, MPI_SUM, comm, mpierr)
    m = acc(1:nf) / max(acc(nf + 1:2 * nf), 1.d0)

    ! a field with zero mean |diag| is not the shape assumed here: identity
    if (all(m > 0.d0)) then
      sc = sqrt(m(1) / m)
    else
      sc = 1.d0
      if (my_id == 0) write(*,'(A,A,A)') &
        "[Physics PC]   WARNING: ", trim(label), &
        " block scaling SKIPPED (a field has zero mean |diag|); D = I"
    endif

    !--- D itself: s_f on field f (s_1 = 1).
    call VecGetArray(dvec, dptr, ierr)
    do kk = 1, size(fld)
      dptr(kk) = sc(fld(kk))
    enddo
    call VecRestoreArray(dvec, dptr, ierr)

    call MatDiagonalScale(A, dvec, dvec, ierr)
    call report_tiny_diag(A, nf, m, comm, my_id, label)

    !--- the |diag| spread achieved
    call MatCreateVecs(A, chk, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, chk, ierr)
    call VecAbs(chk, ierr)
    call VecMax(chk, PETSC_NULL_INTEGER, dmax, ierr)
    call VecMin(chk, PETSC_NULL_INTEGER, dmin, ierr)
    call VecDestroy(chk, ierr)

    if (my_id == 0) then
      ss = ""; sm = ""
      do f = 1, nf
        write(t1, '(ES11.4)') m(f)
        sm = trim(sm)//merge("   ", " / ", f == 1)//t1
        if (f > 1) then
          write(t1, '(ES11.4)') sc(f)
          ss = trim(ss)//merge("   ", " / ", f == 2)//t1
        endif
      enddo
      write(*,'(A,A,A,A,A,A,A,ES11.4)') "[Physics PC]   ", trim(label), " block scale: s = ", &
        trim(adjustl(ss)), ", mean|diag| ", trim(adjustl(sm)), " -> scaled |diag| spread = ", &
        dmax / max(dmin, 1.d-300)
    endif

  end subroutine make_field_block_scale

  !--------------------------------------------------------------------
  !> -sf_diag_report: the rows of a packed block whose |diag| is below
  !! 1e-10 of their field's mean (m, unscaled), with their node, DOF order
  !! and toroidal slot, up to 20 per rank. A is the scaled operator, so its
  !! diagonal is divided back by D before the test.
  !--------------------------------------------------------------------
  subroutine report_tiny_diag(A, nf, m, comm, my_id, label)
    use mod_parameters, only: n_tor, n_degrees
    use nodes_elements, only: node_list
    Mat, intent(in)              :: A
    integer, intent(in)          :: nf
    real*8, intent(in)           :: m(:)
    integer, intent(in)          :: comm, my_id
    character(len=*), intent(in) :: label

    PetscErrorCode :: ierr
    PetscInt :: rs, re, iv
    PetscBool :: set
    PetscScalar, pointer :: dptr(:)
    Vec :: d
    integer :: nloc, kk, f, lf, g, idx, slot, n, k, nfound, ntot, mpierr

    iv = 0
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_diag_report", iv, set, ierr)
    if (iv == 0) return

    call MatGetOwnershipRange(A, rs, re, ierr)
    nloc = int(re - rs) / nf
    call MatCreateVecs(A, d, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, d, ierr)
    call VecGetArrayRead(d, dptr, ierr)
    nfound = 0
    do kk = 1, int(re - rs)
      f  = (kk - 1) / nloc + 1
      lf = mod(kk - 1, nloc)
      ! the scaling multiplied the diagonal by s_f^2 = m(1) / m(f)
      if (abs(dptr(kk)) * m(f) / m(1) >= 1.d-10 * m(f)) cycle
      nfound = nfound + 1
      if (nfound > 20) cycle
      g = int(rs) / nf + lf
      idx = g / n_tor + 1; slot = mod(g, n_tor)
      do n = 1, node_list%n_nodes
        do k = 1, n_degrees
          if (node_list%node(n)%index(k) == idx) then
            write(*,'(A,A,A,I0,A,I0,A,I0,A,I0,A,I0,A,ES10.3,A,2F9.4,A,I0,A,L1)') "[Physics PC]   ", &
              trim(label), " tiny diag: rank ", my_id, " field ", f, " slot ", slot, " node ", n, &
              " dof ", k, " |d|/mean ", abs(dptr(kk)) / m(1), " RZ", node_list%node(n)%x(1,1,1), &
              node_list%node(n)%x(1,1,2), " bnd ", node_list%node(n)%boundary, " axis ", &
              node_list%node(n)%axis_node
          endif
        enddo
      enddo
    enddo
    call VecRestoreArrayRead(d, dptr, ierr)
    call VecDestroy(d, ierr)
    call MPI_Allreduce(nfound, ntot, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
    if (my_id == 0) write(*,'(A,A,A,I0)') "[Physics PC]   ", trim(label), " tiny-diag rows: ", ntot
  end subroutine report_tiny_diag

  !--------------------------------------------------------------------
  !> The penalty rows of a diagonal block, made the Dirichlet rows
  !! eliminate_boundary_dofs builds. With eliminate_boundary_dofs = .f.,
  !! model600's boundary conditions (and fix_axis_nodes) add zbig = 1e12
  !! (1e15 with keep_n0_const) to the diagonal of the whole FEM row, so on
  !! that row x_b = r_b / a_bb up to O(max|a_bj| / zbig). Left in, those
  !! rows set every field's mean |diag| (make_field_block_scale), dominate
  !! the inner FGMRES norms, and stay in the GMG coarse spaces, which drop
  !! only diagonal-only rows. A row with |a_ii| > PEN_TAU max_j |a_ij| has
  !! its row and column zeroed and its field's bulk mean |diag| put on the
  !! diagonal; pfac = bulk / a_ii there (1 elsewhere), so a caller scaling
  !! the right-hand side by pfac still gets x_b = r_b / a_ii. The pattern
  !! is kept (frozen-pattern refills, MUMPS / GMG symbolic reuse).
  !--------------------------------------------------------------------
  subroutine decouple_penalty_rows(A, nloc, pfac, made, comm, my_id, label)
    Mat, intent(inout)           :: A
    PetscInt, intent(in)         :: nloc(:)  !< LOCAL rows of each field, in pack order
    Vec, intent(inout)           :: pfac     !< out: the right-hand-side factor
    logical, intent(inout)       :: made     !< pfac exists (refilled in place)
    integer, intent(in)          :: comm
    integer, intent(in)          :: my_id
    character(len=*), intent(in) :: label

    real*8, parameter :: PEN_TAU = 1.d6
    PetscErrorCode :: ierr
    PetscInt :: rs, re, row, ncols, npen
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:), dptr(:), pptr(:)
    PetscInt, allocatable :: prow(:)
    logical, allocatable :: pen(:)
    integer, allocatable :: fld(:)
    real*8, allocatable :: acc(:), m(:)
    integer :: nf, f, kk, jc, mpierr
    real*8 :: dg, off
    Vec :: dv
    character(len=256) :: sn, sm
    character(len=16)  :: t1

    nf = size(nloc)
    call MatGetOwnershipRange(A, rs, re, ierr)
    allocate(pen(re - rs), fld(re - rs), prow(re - rs), acc(3 * nf), m(nf))
    kk = 0
    do f = 1, nf
      fld(kk + 1:kk + nloc(f)) = f
      kk = kk + int(nloc(f))
    enddo

    !--- detect, and the bulk mean |diag| of each field (owned rows, reduced)
    acc = 0.d0
    do row = rs, re - 1
      kk = int(row - rs) + 1
      call MatGetRow(A, row, ncols, cols, vals, ierr)
      dg = 0.d0; off = 0.d0
      do jc = 1, int(ncols)
        if (cols(jc) == row) then
          dg = abs(vals(jc))
        else
          off = max(off, abs(vals(jc)))
        endif
      enddo
      call MatRestoreRow(A, row, ncols, cols, vals, ierr)
      pen(kk) = (dg > 0.d0) .and. (dg > PEN_TAU * off)
      if (pen(kk)) then
        acc(2 * nf + fld(kk)) = acc(2 * nf + fld(kk)) + 1.d0
      else
        acc(fld(kk))      = acc(fld(kk)) + dg
        acc(nf + fld(kk)) = acc(nf + fld(kk)) + 1.d0
      endif
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, acc, 3 * nf, MPI_DOUBLE_PRECISION, MPI_SUM, comm, mpierr)
    m = acc(1:nf) / max(acc(nf + 1:2 * nf), 1.d0)
    where (.not. (m > 0.d0)) m = 1.d0

    !--- pfac, and the diagonal the decoupled rows get
    if (.not. made) call MatCreateVecs(A, pfac, PETSC_NULL_VEC, ierr)
    made = .true.
    call MatCreateVecs(A, dv, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, dv, ierr)
    call VecGetArray(dv, dptr, ierr)
    call VecGetArray(pfac, pptr, ierr)
    npen = 0
    do kk = 1, size(pen)
      if (pen(kk)) then
        npen = npen + 1
        prow(npen) = rs + kk - 1
        pptr(kk) = m(fld(kk)) / dptr(kk)
        dptr(kk) = m(fld(kk))
      else
        pptr(kk) = 1.d0
      endif
    enddo
    call VecRestoreArray(pfac, pptr, ierr)
    call VecRestoreArray(dv, dptr, ierr)

    call MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    call MatZeroRowsColumns(A, npen, prow(1:npen), 1.d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr)
    call MatDiagonalSet(A, dv, INSERT_VALUES, ierr)
    call VecDestroy(dv, ierr)

    if (my_id == 0) then
      sn = ""; sm = ""
      do f = 1, nf
        write(t1, '(I0)') nint(acc(2 * nf + f))
        sn = trim(sn)//merge(" ", "/", f == 1)//trim(t1)
        write(t1, '(ES10.3)') m(f)
        sm = trim(sm)//merge("  ", " /", f == 1)//t1
      enddo
      write(*,'(A,A,A,A,A,A)') "[Physics PC]   ", trim(label), " penalty rows decoupled:", &
        trim(sn), ", bulk mean|diag|", trim(sm)
    endif
  end subroutine decouple_penalty_rows

  !> v(k) = variable k of the full-system vector x (k = 1..size(v)): local rows
  !! node*bs + (k-1)*n_tor + m -> node*n_tor + m, bs = n_var*n_tor, exactly the
  !! map of create_variable_index_sets. Rank-local, no communication.
  subroutine split_vars(x, v)
    use mod_parameters, only: n_var, n_tor
    Vec :: x
    Vec :: v(:)                         !< variables 1..size(v)
    PetscScalar, pointer :: xa(:), va(:)
    PetscErrorCode :: ierr
    PetscInt :: nl
    integer :: k, i, bs, nn
    call VecGetLocalSize(x, nl, ierr)
    bs = n_var * n_tor
    nn = int(nl) / bs
    call VecGetArrayRead(x, xa, ierr)
    do k = 1, size(v)
      call VecGetArray(v(k), va, ierr)
      do i = 0, nn - 1
        va(i * n_tor + 1 : (i + 1) * n_tor) = xa(i * bs + (k - 1) * n_tor + 1 : i * bs + k * n_tor)
      enddo
      call VecRestoreArray(v(k), va, ierr)
    enddo
    call VecRestoreArrayRead(x, xa, ierr)
  end subroutine split_vars

  !> Inverse of split_vars: variables 1..size(v) of y from v(k); any further
  !! variables of y are left untouched.
  subroutine merge_vars(v, y)
    use mod_parameters, only: n_var, n_tor
    Vec :: v(:)
    Vec :: y
    PetscScalar, pointer :: ya(:), va(:)
    PetscErrorCode :: ierr
    PetscInt :: nl
    integer :: k, i, bs, nn
    call VecGetLocalSize(y, nl, ierr)
    bs = n_var * n_tor
    nn = int(nl) / bs
    call VecGetArray(y, ya, ierr)
    do k = 1, size(v)
      call VecGetArrayRead(v(k), va, ierr)
      do i = 0, nn - 1
        ya(i * bs + (k - 1) * n_tor + 1 : i * bs + k * n_tor) = va(i * n_tor + 1 : (i + 1) * n_tor)
      enddo
      call VecRestoreArrayRead(v(k), va, ierr)
    enddo
    call VecRestoreArray(y, ya, ierr)
  end subroutine merge_vars

#endif
end module mod_petsc_pc_blocks
