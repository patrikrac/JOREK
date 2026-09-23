module mod_petsc_pc_blocks
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx
  implicit none
  private

  !--------------------------------------------------------------------
  !> Leaf primitives shared by the physics preconditioner paths.
  !!
  !! These routines were factored out of mod_petsc_pc_physics_construction so
  !! that the production path (mod_petsc_pc_sf) and the research path can use
  !! ONE implementation rather than a copy each. Nothing here decides anything:
  !! every routine is a pure operation on PETSc objects plus g_ctx%is_var, and
  !! none of them reads a physics_pc_* flag except physics_pc_harm_split, which
  !! selects the harmonic-block filter inside the extraction.
  !!
  !! Moving them was value-neutral by construction: the bodies are unchanged.
  !--------------------------------------------------------------------

  public :: pc_print_block_setup
  public :: create_variable_index_sets
  public :: extract_sub_block
  public :: extract_sub_block_h, extract_sub_blocks_h
  public :: pack_pair_aij
  public :: make_pair_block_scale
  public :: report_operator_density
  public :: split_vars, merge_vars

contains

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
  !> Create variable index sets for extracting sub-vectors from the
  !! full 6-variable system vector.
  !!
  !! DOF ordering within each BAIJ block (block_size = n_var*n_tor):
  !!   var v (0-based) at node block i: i*block_size + v*n_tor + m
  !!   for m = 0..n_tor-1
  !--------------------------------------------------------------------
  subroutine create_variable_index_sets(A_full, comm)
    use mod_parameters, only: n_var, n_tor

    Mat, intent(in) :: A_full
    integer, intent(in) :: comm

    PetscInt :: n_local, n_global, rstart, rend
    PetscInt :: block_size, n_block_local, n_var_dofs, n_block_global
    !PetscInt :: out_local, out_global, out_start, out_end
    PetscInt, allocatable :: indices(:)
    PetscErrorCode :: ierr
    integer :: v, i, m, k

    ! Get parallel layout from the full system matrix
    PetscCallA(MatGetLocalSize(A_full, n_local, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetSize(A_full, n_global, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetOwnershipRange(A_full, rstart, rend, ierr))

    !write(*,'(A,I8,A,I8)') "[Physics PC]   Creating variable index sets: local DOFs ", n_local, " [", rstart, "-", rend-1, "]"

    block_size    = n_var * n_tor
    n_block_local = n_local / block_size
    n_block_global = n_global / block_size
    n_var_dofs    = n_block_local * n_tor

    allocate(indices(n_var_dofs))

    do v = 1, 6
      k = 0
      do i = 0, n_block_local - 1
        do m = 0, n_tor - 1
          k = k + 1
          !k = i*(n_tor-1) + m + 1
          indices(k) = rstart + i * block_size + (v-1) * n_tor + m
        enddo
      enddo
      PetscCallA(ISCreateGeneral(comm, n_var_dofs, indices, PETSC_COPY_VALUES, g_ctx%is_var(v), ierr))

      ! Print information about the created IS for debugging
      !PetscCallA(ISGetSize(g_ctx%is_var(v), out_global, ierr))
      !PetscCallA(ISGetLocalSize(g_ctx%is_var(v), out_local, ierr))
      !PetscCallA(ISGetMinMax(g_ctx%is_var(v), out_start, out_end, ierr))
      !write(*,*) "[Physics PC]     Variable ", v, ": local DOFs : ", out_local," global DOFs ", out_global, " [", out_start, "-", out_end, "]"
    enddo

    deallocate(indices)
    g_ctx%is_created = .true.
  end subroutine create_variable_index_sets


  !--------------------------------------------------------------------
  !> Extract a sub-block A_ij from the full system matrix.
  !! A_ij has rows corresponding to equation eq_row and columns
  !! corresponding to variable var_col.
  !--------------------------------------------------------------------
  subroutine extract_sub_block(A_full, eq_row, var_col, B, first_time)
    Mat, intent(in)    :: A_full
    integer, intent(in) :: eq_row, var_col
    Mat, intent(inout)  :: B
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (first_time) then
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), MAT_INITIAL_MATRIX, B, ierr))
    else
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), MAT_REUSE_MATRIX, B, ierr))
    endif
  end subroutine extract_sub_block

  !--------------------------------------------------------------------
  !> Audit A8: extract_sub_block, optionally keeping only the entries between
  !! the SAME toroidal mode number |n| (group (m+1)/2 of slot m; cos and sin of
  !! one n stay coupled) (physics_pc_harm_split = 1). JOREK assembles n_tor x n_tor blocks,
  !! so every extracted block stores the cross-harmonic couplings, which are
  !! zero for an axisymmetric linearisation and O(perturbation) otherwise. The
  !! standard JOREK preconditioner is already harmonic-block-diagonal; this
  !! makes the physics PC's operands so too, which removes ~2/3 of every
  !! product's and matvec's work and splits each pair LU into n_tor
  !! independent factorizations.
  !!
  !! The filtered block is read straight out of A_full's rows (MatGetRow), so
  !! no unfiltered copy is ever held (memory audit: those copies of the 21
  !! blocks were ~0.8 GB at 81x32), and A_full may be the AIJ copy or JOREK's
  !! own BAIJ matrix (physics_pc_lean_setup >= 3 drops the AIJ copy).
  !! Layout (create_variable_index_sets): full row node*bs + (v-1)*n_tor + m,
  !! sub-block row node*n_tor + m, bs = n_var*n_tor.
  !!
  !! B keeps its identity across rebuilds (the product caches depend on it):
  !! it is refilled into the pattern fixed on the first build.
  !--------------------------------------------------------------------
  subroutine extract_sub_block_h(A_full, eq_row, var_col, B, first_time)
    use phys_module,    only: physics_pc_harm_split
    use mod_parameters, only: n_tor, n_var
    Mat, intent(in)     :: A_full
    integer, intent(in) :: eq_row, var_col
    Mat, intent(inout)  :: B
    logical, intent(in) :: first_time
    PetscErrorCode :: ierr
    PetscInt :: i, ncols, rstart, rend, m, k, q, c, bs, nloc, nsub, nglob
    PetscInt :: sr, sc, cstart, cend, node, col0
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: dcnt(:), ocnt(:), cc(:)
    PetscScalar, allocatable :: vv(:)
    integer :: comm

    if (physics_pc_harm_split == 0) then
      call extract_sub_block(A_full, eq_row, var_col, B, first_time)
      return
    endif

    bs = n_var * n_tor
    call MatGetOwnershipRange(A_full, rstart, rend, ierr)
    call MatGetSize(A_full, nglob, PETSC_NULL_INTEGER, ierr)
    nloc   = ((rend - rstart) / bs) * n_tor
    nsub   = (nglob / bs) * n_tor
    cstart = (rstart / bs) * n_tor            ! owned sub-block columns [cstart, cend)
    cend   = cstart + nloc
    col0   = (var_col - 1) * n_tor            ! first slot of var_col inside a block

    if (first_time) then
      allocate(dcnt(nloc), ocnt(nloc))
      dcnt = 0; ocnt = 0
      do node = rstart / bs, rend / bs - 1
        do m = 0, n_tor - 1
          i = node * bs + (eq_row - 1) * n_tor + m
          sr = node * n_tor + m - cstart + 1
          call MatGetRow(A_full, i, ncols, cols, vals, ierr)
          do k = 1, ncols
            q = mod(cols(k), bs) - col0
            if (q < 0 .or. q >= n_tor) cycle
            if ((q + 1) / 2 /= (m + 1) / 2) cycle
            sc = (cols(k) / bs) * n_tor + q
            if (sc >= cstart .and. sc < cend) then
              dcnt(sr) = dcnt(sr) + 1
            else
              ocnt(sr) = ocnt(sr) + 1
            endif
          enddo
          call MatRestoreRow(A_full, i, ncols, cols, vals, ierr)
        enddo
      enddo
      call PetscObjectGetComm(A_full, comm, ierr)
      call MatCreate(comm, B, ierr)
      call MatSetSizes(B, nloc, nloc, nsub, nsub, ierr)
      call MatSetType(B, MATMPIAIJ, ierr)
      call MatMPIAIJSetPreallocation(B, PETSC_DEFAULT_INTEGER, dcnt, &
                                     PETSC_DEFAULT_INTEGER, ocnt, ierr)
      deallocate(dcnt, ocnt)
    else
      call MatZeroEntries(B, ierr)
    endif

    allocate(cc(bs * 64), vv(bs * 64))
    do node = rstart / bs, rend / bs - 1
      do m = 0, n_tor - 1
        i = node * bs + (eq_row - 1) * n_tor + m
        sr = node * n_tor + m
        call MatGetRow(A_full, i, ncols, cols, vals, ierr)
        if (ncols > size(cc)) then
          deallocate(cc, vv); allocate(cc(ncols), vv(ncols))
        endif
        c = 0
        do k = 1, ncols
          q = mod(cols(k), bs) - col0
          if (q < 0 .or. q >= n_tor) cycle
          if ((q + 1) / 2 /= (m + 1) / 2) cycle
          c = c + 1; cc(c) = (cols(k) / bs) * n_tor + q; vv(c) = vals(k)
        enddo
        call MatRestoreRow(A_full, i, ncols, cols, vals, ierr)
        if (c > 0) call MatSetValues(B, 1_4, [sr], c, cc(1:c), vv(1:c), INSERT_VALUES, ierr)
      enddo
    enddo
    deallocate(cc, vv)
    call MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY, ierr)
    if (first_time) call MatSetOption(B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
  end subroutine extract_sub_block_h

  !--------------------------------------------------------------------
  !> extract_sub_block_h for a list of blocks Mb(k) = A(eqs(k), vrs(k)) in ONE
  !! pass over A_full's rows: each equation row is read once (MatGetRow on
  !! JOREK's BAIJ matrix expands the whole n_var*n_tor-wide block row) and its
  !! entries are dispatched to every requested block of that equation. The
  !! 21-block SFM2 extraction read each row 2-6 times. Same entries, same
  !! per-row insertion order, same preallocation as extract_sub_block_h.
  !--------------------------------------------------------------------
  subroutine extract_sub_blocks_h(A_full, eqs, vrs, Mb, first_time)
    use phys_module,    only: physics_pc_harm_split
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
    if (physics_pc_harm_split == 0) then
      do kb = 1, nb
        call extract_sub_block(A_full, eqs(kb), vrs(kb), Mb(kb), first_time)
      enddo
      return
    endif

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
              if ((q + 1) / 2 /= (m + 1) / 2) cycle
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
  !> Report ABSOLUTE operator density: rows, nnz, and nnz per row.
  !!
  !! The per-arm build prints quote nnz RATIOS against different denominators
  !! (Atilde_22 on one arm, B_22 on another, and the mixed arm's operator is
  !! twice the size because it is packed), so those ratios cannot be compared
  !! across arms. Densification under mesh refinement is the gate this whole
  !! line of work is trying to clear, so it needs one absolute number measured
  !! the same way everywhere. That is what this prints.
  !--------------------------------------------------------------------
  subroutine report_operator_density(A, tag, my_id)
    Mat, intent(in)              :: A
    character(len=*), intent(in) :: tag
    integer, intent(in)          :: my_id

    MatInfo        :: minfo
    PetscErrorCode :: ierr
    PetscInt       :: nrow
    real*8         :: nz

    call MatGetSize(A, nrow, PETSC_NULL_INTEGER, ierr)
    call MatGetInfo(A, MAT_GLOBAL_SUM, minfo, ierr)
    nz = minfo%nz_used
    if (my_id == 0) write(*,'(A,A,A,I8,A,ES12.5,A,F9.2)') &
      "[Physics PC]   DENSITY ", tag, ": rows = ", nrow, &
      ", nnz = ", nz, ", nnz/row = ", nz / max(dble(nrow), 1.d0)
  end subroutine report_operator_density

  !--------------------------------------------------------------------
  !> C = [[A11, A12], [A21, A22]] as one MPIAIJ in the packed layout of
  !! MatConvert(MatNest -> MPIAIJ): rank r owns [its field-1 rows | its field-2
  !! rows], and a field-j column c owned by rank p sits at packed column
  !! c + rg2(p) (field 1) or c + rg1(p+1) (field 2), rg = the fields'
  !! ownership starts. Replaces that MatConvert, whose parallel path
  !! ISAllGathers every global column index onto every rank -- O(N) memory and
  !! traffic per rank, twice per PC rebuild.
  !!
  !! First call (ready = .false.): exact preallocation, and the pattern is
  !! frozen (MAT_NEW_NONZERO_LOCATION_ERR). Later calls refill the values in
  !! place, so C keeps its identity across rebuilds and every consumer (the
  !! GMG's PtAP, the coarse/axis LUs, MUMPS) reuses its symbolic phase. The
  !! stored values and each row's column order are those of the MatConvert.
  !--------------------------------------------------------------------
  subroutine pack_pair_aij(A11, A12, A21, A22, C, ready, comm)
    Mat, intent(in)        :: A11, A12, A21, A22
    Mat, intent(inout)     :: C
    logical, intent(inout) :: ready
    integer, intent(in)    :: comm
    PetscErrorCode :: ierr
    PetscInt :: s1, e1, s2, e2, n1, n2, ps, r, ncols, k, pr
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: dnz(:), onz(:), pc_(:)
    integer, allocatable :: rg1(:), rg2(:)
    integer :: np, me, mpierr, pass

    call MPI_Comm_size(comm, np, mpierr)
    call MPI_Comm_rank(comm, me, mpierr)
    call MatGetOwnershipRange(A11, s1, e1, ierr)
    call MatGetOwnershipRange(A22, s2, e2, ierr)
    n1 = e1 - s1; n2 = e2 - s2
    allocate(rg1(0:np), rg2(0:np))
    call MPI_Allgather(int(s1), 1, MPI_INTEGER, rg1, 1, MPI_INTEGER, comm, mpierr)
    call MPI_Allgather(int(s2), 1, MPI_INTEGER, rg2, 1, MPI_INTEGER, comm, mpierr)
    call MatGetSize(A11, k, PETSC_NULL_INTEGER, ierr); rg1(np) = int(k)
    call MatGetSize(A22, k, PETSC_NULL_INTEGER, ierr); rg2(np) = int(k)
    ps = s1 + s2                                    ! first packed row of this rank

    if (.not. ready) then
      allocate(dnz(n1 + n2), onz(n1 + n2))
      dnz = 0; onz = 0
    else
      call MatZeroEntries(C, ierr)
    endif
    allocate(pc_(64))
    ! pass 1 counts (first call only), pass 2 inserts
    do pass = merge(2, 1, ready), 2
      do r = 0, n1 - 1
        pr = ps + r
        call put_row(A11, s1 + r, 1, pr, pass)
        call put_row(A12, s1 + r, 2, pr, pass)
      enddo
      do r = 0, n2 - 1
        pr = ps + n1 + r
        call put_row(A21, s2 + r, 1, pr, pass)
        call put_row(A22, s2 + r, 2, pr, pass)
      enddo
      if (pass == 1) then
        call MatCreate(comm, C, ierr)
        call MatSetSizes(C, n1 + n2, n1 + n2, int(rg1(np) + rg2(np), kind(n1)), &
                         int(rg1(np) + rg2(np), kind(n1)), ierr)
        call MatSetType(C, MATMPIAIJ, ierr)
        call MatMPIAIJSetPreallocation(C, PETSC_DEFAULT_INTEGER, dnz, PETSC_DEFAULT_INTEGER, onz, ierr)
        deallocate(dnz, onz)
      endif
    enddo
    call MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY, ierr)
    if (.not. ready) call MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
    ready = .true.
    deallocate(rg1, rg2, pc_)

  contains

    !> Row `row` of block Ab (columns in field fc) into packed row prow:
    !! pass 1 counts diagonal/off-diagonal entries, pass 2 inserts.
    subroutine put_row(Ab, row, fc, prow, pass_)
      Mat, intent(in)      :: Ab
      PetscInt, intent(in) :: row, prow
      integer, intent(in)  :: fc, pass_
      PetscInt :: q, cg
      integer :: p
      call MatGetRow(Ab, row, ncols, cols, vals, ierr)
      if (ncols > size(pc_)) then
        deallocate(pc_); allocate(pc_(2 * ncols))
      endif
      p = me
      do q = 1, ncols
        cg = cols(q)
        if (fc == 1) then
          if (cg < rg1(p) .or. cg >= rg1(p + 1)) p = owner(rg1, int(cg))
          pc_(q) = cg + rg2(p)
        else
          if (cg < rg2(p) .or. cg >= rg2(p + 1)) p = owner(rg2, int(cg))
          pc_(q) = cg + rg1(p + 1)
        endif
      enddo
      if (pass_ == 1) then
        do q = 1, ncols
          if (pc_(q) >= ps .and. pc_(q) < ps + n1 + n2) then
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

    !> Rank owning index ix of a field with ownership starts rg(0:np).
    integer function owner(rg, ix)
      integer, intent(in) :: rg(0:), ix
      integer :: lo, hi, mid
      lo = 0; hi = np - 1
      do while (lo < hi)
        mid = (lo + hi + 1) / 2
        if (rg(mid) <= ix) then
          lo = mid
        else
          hi = mid - 1
        endif
      enddo
      owner = lo
    end function owner

  end subroutine pack_pair_aij

  !--------------------------------------------------------------------
  !> Workstream B, Step 2: symmetric BLOCK scaling of a packed 2-field pair.
  !!
  !!   A <- D A D,   D = diag(I, s I)
  !!
  !! The caller then solves (D A D) z = D b and recovers x = D z, so this is an
  !! exact similarity: it changes the conditioning the inner solver sees and
  !! NOTHING else. physics_pc_pair_scale = 0 skips it entirely and so reproduces
  !! the unscaled results bit-for-bit.
  !!
  !! WHY. Both packed pairs pit an operator block against a mass block, and in
  !! both the two carry very different scale. Measured over the full ramp with
  !! physics_pc_probe_inner = 3 (meas_B/pr_ramp_m8), the two behave DIFFERENTLY
  !! and it matters:
  !!
  !!   pair_w   |diag| spread 5.3e9 -> 2.2e10 -> 1.2e12 -> 3.1e13 at tstep
  !!            1 / 10 / 100 / 1000. min is pinned at 1.858e-5 (the dt-independent
  !!            B_44 mass rows) while max tracks dt. Genuinely dt-driven.
  !!   pair_psi |diag| spread CONSTANT at 1.78e10 across the whole ramp. Its
  !!            mismatch is static, not dt-driven.
  !!
  !! So the dt story holds for pair_w only. It is NOT the ZBIG penalty rows in
  !! either case -- eliminate_boundary_dofs has already removed those, and no
  !! measured diagonal reaches 1e12 until pair_w does so on its own at tstep=100.
  !!
  !! Note also that spread alone does NOT predict solvability: pair_psi holds a
  !! constant 1.78e10 spread while GMRES+ILU(0) on it improves from err 7.3e+1 at
  !! tstep=1 to 3.5e-5 at tstep=1000. Scaling is therefore expected to pay on
  !! pair_w, where the spread grows four decades and every candidate (INCLUDING
  !! the MUMPS LU, err 1.7e-6 at tstep=1000) degrades with it. It is applied to
  !! both because it is an exact similarity and costs one MatDiagonalScale.
  !!
  !! s is MEASURED from the two block diagonals, not assumed from a power of dt.
  !! pair_w's max does not follow a clean power of dt (ratios 1.9 / 37 / 26 across
  !! the ramp's decades) because S_uu carries both a mass part and the dt^2
  !! channel correction, so an assumed exponent would be wrong; and pair_psi has
  !! no dt dependence to assume in the first place.
  !!
  !! One scalar per block, deliberately: the block structure that a PCFIELDSPLIT
  !! or a point-block smoother needs is preserved exactly. Per-row equilibration
  !! would flatten the diagonal further but destroy that structure.
  !--------------------------------------------------------------------
  subroutine make_pair_block_scale(A, n1_loc, dvec, comm, my_id, label)
    Mat, intent(inout)           :: A
    PetscInt, intent(in)         :: n1_loc   !< LOCAL rows in the FIRST field: the
                                             !< Nest->AIJ pack is [f1 local | f2 local] per rank
    Vec, intent(inout)           :: dvec     !< out: D, kept for the apply
    integer, intent(in)          :: comm
    integer, intent(in)          :: my_id
    character(len=*), intent(in) :: label

    PetscErrorCode :: ierr
    PetscInt  :: rstart, rend, ii
    integer   :: kk, mpierr
    PetscScalar, pointer :: dptr(:)
    real*8    :: acc(4), m1, m2, s
    PetscReal :: dmin, dmax
    Vec       :: chk

    call MatCreateVecs(A, dvec, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, dvec, ierr)
    call MatGetOwnershipRange(A, rstart, rend, ierr)

    !--- mean |diag| of each field, over owned rows, then reduced.
    acc = 0.d0
    call VecGetArray(dvec, dptr, ierr)
    do kk = 1, int(rend - rstart)
      if (kk <= n1_loc) then
        acc(1) = acc(1) + abs(dptr(kk))
        acc(3) = acc(3) + 1.d0
      else
        acc(2) = acc(2) + abs(dptr(kk))
        acc(4) = acc(4) + 1.d0
      endif
    enddo
    call VecRestoreArray(dvec, dptr, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, acc, 4, MPI_DOUBLE_PRECISION, MPI_SUM, comm, mpierr)

    m1 = acc(1) / max(acc(3), 1.d0)
    m2 = acc(2) / max(acc(4), 1.d0)

    ! Fall back to the identity rather than guessing. A zero mean means the block
    ! is not the shape this routine assumes, and a silently wrong scaling would be
    ! indistinguishable from a bad preconditioner in every downstream number.
    if (m1 > 0.d0 .and. m2 > 0.d0) then
      s = sqrt(m1 / m2)
    else
      s = 1.d0
      if (my_id == 0) write(*,'(A,A,A)') &
        "[Physics PC]   WARNING: ", trim(label), &
        " block scaling SKIPPED (a field has zero mean |diag|); D = I"
    endif

    !--- D itself: 1 on the first field, s on the second.
    call VecGetArray(dvec, dptr, ierr)
    do kk = 1, int(rend - rstart)
      if (kk <= n1_loc) then
        dptr(kk) = 1.d0
      else
        dptr(kk) = s
      endif
    enddo
    call VecRestoreArray(dvec, dptr, ierr)

    call MatDiagonalScale(A, dvec, dvec, ierr)

    !--- The spread actually achieved. This is the number the scaling exists to
    !--- move, so it belongs in the log next to the unscaled one.
    call MatCreateVecs(A, chk, PETSC_NULL_VEC, ierr)
    call MatGetDiagonal(A, chk, ierr)
    call VecAbs(chk, ierr)
    call VecMax(chk, PETSC_NULL_INTEGER, dmax, ierr)
    call VecMin(chk, PETSC_NULL_INTEGER, dmin, ierr)
    call VecDestroy(chk, ierr)

    if (my_id == 0) write(*,'(A,A,A,ES11.4,A,ES11.4,A,ES11.4,A,ES11.4)') &
      "[Physics PC]   ", trim(label), " block scale: s = ", s, &
      ", mean|diag| ", m1, " / ", m2, " -> scaled |diag| spread = ", dmax/max(dmin, 1.d-300)

  end subroutine make_pair_block_scale

  !> v(k) = variable k of the full-system vector x (k = 1..6): local rows
  !! node*bs + (k-1)*n_tor + m -> node*n_tor + m, bs = n_var*n_tor, exactly the
  !! map of create_variable_index_sets. Rank-local, no communication.
  subroutine split_vars(x, v)
    use mod_parameters, only: n_var, n_tor
    Vec :: x
    Vec :: v(6)
    PetscScalar, pointer :: xa(:), va(:)
    PetscErrorCode :: ierr
    PetscInt :: nl
    integer :: k, i, bs, nn
    call VecGetLocalSize(x, nl, ierr)
    bs = n_var * n_tor
    nn = int(nl) / bs
    call VecGetArrayRead(x, xa, ierr)
    do k = 1, 6
      call VecGetArray(v(k), va, ierr)
      do i = 0, nn - 1
        va(i * n_tor + 1 : (i + 1) * n_tor) = xa(i * bs + (k - 1) * n_tor + 1 : i * bs + k * n_tor)
      enddo
      call VecRestoreArray(v(k), va, ierr)
    enddo
    call VecRestoreArrayRead(x, xa, ierr)
  end subroutine split_vars

  !> Inverse of split_vars: variables 1..6 of y from v(k); any further
  !! variables of y (n_var > 6) are left untouched, as before.
  subroutine merge_vars(v, y)
    use mod_parameters, only: n_var, n_tor
    Vec :: v(6)
    Vec :: y
    PetscScalar, pointer :: ya(:), va(:)
    PetscErrorCode :: ierr
    PetscInt :: nl
    integer :: k, i, bs, nn
    call VecGetLocalSize(y, nl, ierr)
    bs = n_var * n_tor
    nn = int(nl) / bs
    call VecGetArray(y, ya, ierr)
    do k = 1, 6
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
