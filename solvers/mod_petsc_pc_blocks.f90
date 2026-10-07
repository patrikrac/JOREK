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
  !! Nothing here decides anything: every routine is a pure operation on
  !! PETSc objects plus g_ctx%is_var. The block extraction keeps the |n|
  !! groups at most harm_band apart (harm_kept); mod_petsc_pc_sf sets
  !! harm_band from physics_pc_sf_harm_couple.
  !--------------------------------------------------------------------

  public :: pc_print_block_setup
  public :: create_variable_index_sets
  public :: extract_sub_blocks_h
  public :: pack_pair_aij, pack_blocks_aij
  public :: make_pair_block_scale, make_field_block_scale
  public :: split_vars, merge_vars
  public :: harm_band, harm_kept

  !> Cross-|n| band of the block extraction: 0 = same |n| group only (cos and
  !! sin of one n together), k > 0 = groups at most k apart, < 0 = all.
  integer, save :: harm_band = 0

contains

  !> Does the block extraction keep the entry between toroidal slots m and q?
  !! Slot m belongs to |n| group (m+1)/2 (slot 0 = n = 0, then cos/sin pairs).
  pure logical function harm_kept(m, q)
    integer, intent(in) :: m, q
    harm_kept = (harm_band < 0) .or. (abs((m + 1) / 2 - (q + 1) / 2) <= harm_band)
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
  !> Extract the blocks Mb(k) = A(eqs(k), vrs(k)) of the full system matrix in
  !! ONE pass over A_full's rows: each equation row is read once (MatGetRow on
  !! JOREK's BAIJ matrix expands the whole n_var*n_tor-wide block row) and its
  !! entries are dispatched to every requested block of that equation. Only
  !! the entries between |n| groups at most harm_band apart are kept
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
  !> Workstream B, Step 2: symmetric BLOCK scaling of a packed 2-field pair.
  !!
  !!   A <- D A D,   D = diag(I, s I)
  !!
  !! The caller then solves (D A D) z = D b and recovers x = D z, so this is an
  !! exact similarity: it changes the conditioning the inner solver sees and
  !! NOTHING else.
  !!
  !! WHY. Both packed pairs pit an operator block against a mass block, and in
  !! both the two carry very different scale. Measured over the full ramp (the
  !! former inner-solver probe, meas_B/pr_ramp_m8), the two behave DIFFERENTLY
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
    PetscInt :: rstart, rend
    PetscErrorCode :: ierr
    call MatGetOwnershipRange(A, rstart, rend, ierr)
    call make_field_block_scale(A, [n1_loc, rend - rstart - n1_loc], dvec, comm, my_id, label)
  end subroutine make_pair_block_scale

  !--------------------------------------------------------------------
  !> make_pair_block_scale for nf packed fields: D = diag(I, s_2 I, ..., s_nf I),
  !! s_f = sqrt(mean|diag| of field 1 / mean|diag| of field f).
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

    ! Fall back to the identity rather than guessing. A zero mean means the block
    ! is not the shape this routine assumes, and a silently wrong scaling would be
    ! indistinguishable from a bad preconditioner in every downstream number.
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

    !--- The spread actually achieved. This is the number the scaling exists to
    !--- move, so it belongs in the log next to the unscaled one.
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
