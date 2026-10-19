module mod_petsc_pc_sf_gather
#ifdef USE_PETSC
  use mpi_mod
  use iso_c_binding
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_blocks, only: harm_kept
  use mod_petsc_raw_csr, only: split_parts, get_ij, put_ij, &
                               c_baij_get, c_baij_restore, c_aij_get, c_aij_restore
  implicit none
  private

  !--------------------------------------------------------------------
  !> Rebuild-time refill of the global SF path's blocks by a value map.
  !!
  !! Every target's pattern is fixed by JOREK's BAIJ Jacobian A, which is
  !! fixed for the run. So for every stored entry of every target, the
  !! position of its source value in A's BAIJ value arrays is computed once
  !! (sfg_build); a rebuild is then one indexed copy per entry (sfg_gather),
  !! with no MatGetRow, MatSetValues or packing. sfg_gather stops if A changes
  !! object or pattern.
  !--------------------------------------------------------------------

  !> One target and its map: md / mo index the target's diagonal and
  !! off-diagonal CSR values; each entry is +k (k-th value of A's diagonal
  !! BAIJ part), -k (off-diagonal part) or 0 (not from A).
  type :: tgt_t
    Mat     :: T
    integer :: e(2) = 0, v(2) = 0      !< row / column variables (fields 1, 2)
    logical :: pair = .false.
    integer(c_int32_t), allocatable :: md(:), mo(:)
  end type tgt_t

  type(tgt_t), allocatable, save :: tg(:)
  integer(8), save :: a_nzst = -1, a_id = -1
  integer(8), save :: na_d = 0, na_o = 0     !< A's value-array lengths

  public :: sfg_build, sfg_gather

contains

  !> Build the maps (first build): blocks(k) = A(eqs(k), vrs(k)) in the
  !! sub-block layout, and the packed pair Kpj on the fields pj.
  subroutine sfg_build(A, blocks, eqs, vrs, Kpj, pj, comm, my_id)
    Mat, intent(in)     :: A
    Mat, intent(in)     :: blocks(:), Kpj
    integer, intent(in) :: eqs(:), vrs(:), pj(2)
    integer, intent(in) :: comm, my_id
    integer :: k, nb

    call require_baij(A, my_id)
    nb = size(blocks)
    allocate(tg(nb + 1))
    do k = 1, nb
      tg(k)%T = blocks(k); tg(k)%e = [eqs(k), 0]; tg(k)%v = [vrs(k), 0]
    enddo
    tg(nb + 1)%T = Kpj; tg(nb + 1)%e = pj; tg(nb + 1)%v = pj; tg(nb + 1)%pair = .true.
    do k = 1, size(tg)
      call build_map(tg(k), A, comm)
    enddo
    call source_state(A, a_id, a_nzst)
  end subroutine sfg_build

  !> Refill every target from A (every rebuild after the first).
  subroutine sfg_gather(A, my_id)
    Mat, intent(in)     :: A
    integer, intent(in) :: my_id
    integer(8) :: id, nz
    integer :: k
    PetscErrorCode :: ierr

    call source_state(A, id, nz)
    if (id /= a_id .or. nz /= a_nzst) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: the Jacobian changed object or pattern; "// &
        "the SF value maps assume both frozen."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    do k = 1, size(tg)
      call gather_one(tg(k), A)
      ! bump the object state: MUMPS and the GMG decide "new values" by it
      call MatAssemblyBegin(tg(k)%T, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(tg(k)%T, MAT_FINAL_ASSEMBLY, ierr)
    enddo
  end subroutine sfg_gather

  subroutine require_baij(M, my_id)
    Mat, intent(in) :: M
    integer, intent(in) :: my_id
    MatType :: mt
    PetscErrorCode :: ierr
    call MatGetType(M, mt, ierr)
    if (trim(mt) /= MATMPIBAIJ) then
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: the SF value maps need the Jacobian "// &
        "as MATMPIBAIJ, got ", trim(mt)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
  end subroutine require_baij

  subroutine source_state(M, id, nz)
    Mat, intent(in)         :: M
    integer(8), intent(out) :: id, nz
    PetscInt64 :: pid
    PetscObjectState :: st
    PetscErrorCode :: ierr
    call PetscObjectGetId(M, pid, ierr)
    call MatGetNonzeroState(M, st, ierr)
    id = int(pid, 8); nz = int(st, 8)
  end subroutine source_state

  !--------------------------------------------------------------------
  !> The map of one target. Coordinates: a sub-block index s = node*n_tor + m
  !! (the layout of every extracted block); a packed pair holds, on rank p,
  !! [field-1 rows | field-2 rows] of that rank's sub-block range.
  !--------------------------------------------------------------------
  subroutine build_map(t, A, comm)
    use mod_parameters, only: n_tor, n_var
    type(tgt_t), intent(inout) :: t
    Mat, intent(in)     :: A
    integer, intent(in) :: comm

    Mat :: Ad, Ao, Td, To
    PetscInt, pointer :: ga(:), gt(:)
    PetscInt, pointer :: aia(:), aja(:), oia(:), oja(:)
    PetscInt, pointer :: tia(:), tja(:), toia(:), toja(:)
    PetscInt :: nad, nao, ntd, nto, rs, re, cs, ce, bsa
    integer, allocatable :: sub0(:), nsub(:)
    integer :: np, me, mpierr, lr, k, f_r, f_c, node0
    integer(8) :: sr, sc
    PetscErrorCode :: ierr

    call MPI_Comm_size(comm, np, mpierr)
    call MPI_Comm_rank(comm, me, mpierr)
    bsa = n_var * n_tor

    call split_parts(A, .true., Ad, Ao, ga)
    call get_ij(Ad, .true., nad, aia, aja)
    call get_ij(Ao, .true., nao, oia, oja)
    na_d = int(aia(nad + 1), 8) * bsa * bsa
    na_o = int(oia(nao + 1), 8) * bsa * bsa
    if (max(na_d, na_o) >= int(huge(0_c_int32_t), 8)) then
      write(*,'(A)') "[Physics PC]   FATAL: a BAIJ value array exceeds the 32-bit map index."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif

    ! the sub-block ranges of every rank (packed-pair column decoding)
    call MatGetOwnershipRange(t%T, rs, re, ierr)
    call MatGetOwnershipRangeColumn(t%T, cs, ce, ierr)
    allocate(sub0(0:np), nsub(0:np - 1))
    if (t%pair) then
      call MPI_Allgather(int(rs) / 2, 1, MPI_INTEGER, sub0, 1, MPI_INTEGER, comm, mpierr)
      call MPI_Allgather(int(re - rs) / 2, 1, MPI_INTEGER, nsub, 1, MPI_INTEGER, comm, mpierr)
    else
      call MPI_Allgather(int(rs), 1, MPI_INTEGER, sub0, 1, MPI_INTEGER, comm, mpierr)
      call MPI_Allgather(int(re - rs), 1, MPI_INTEGER, nsub, 1, MPI_INTEGER, comm, mpierr)
    endif
    sub0(np) = sub0(np - 1) + nsub(np - 1)
    node0 = sub0(me) / n_tor                       ! this rank's first block row of A

    call split_parts(t%T, .false., Td, To, gt)
    call get_ij(Td, .false., ntd, tia, tja)
    call get_ij(To, .false., nto, toia, toja)
    if (allocated(t%md)) deallocate(t%md, t%mo)
    allocate(t%md(tia(ntd + 1)), t%mo(toia(nto + 1)))
    t%md = 0; t%mo = 0

    do lr = 0, int(ntd) - 1
      call decode(int(rs, 8) + lr, .true., sr, f_r)
      do k = int(tia(lr + 1)) + 1, int(tia(lr + 2))
        call decode(int(cs, 8) + tja(k), .false., sc, f_c)
        t%md(k) = locate(sr, sc, f_r, f_c)
      enddo
      do k = int(toia(lr + 1)) + 1, int(toia(lr + 2))
        call decode(int(gt(toja(k) + 1), 8), .false., sc, f_c)
        t%mo(k) = locate(sr, sc, f_r, f_c)
      enddo
    enddo

    call put_ij(Td, .false., ntd, tia, tja); call put_ij(To, .false., nto, toia, toja)
    call put_ij(Ad, .true., nad, aia, aja);  call put_ij(Ao, .true., nao, oia, oja)
    deallocate(sub0, nsub)

  contains

    !> Target global index -> (sub-block index, field 1 | 2).
    subroutine decode(g, is_row, s, f)
      integer(8), intent(in)  :: g
      logical, intent(in)     :: is_row
      integer(8), intent(out) :: s
      integer, intent(out)    :: f
      integer :: p, lo, hi, mid
      integer(8) :: off
      if (.not. t%pair) then
        s = g; f = 1
        return
      endif
      if (is_row) then
        p = me
      else
        lo = 0; hi = np - 1                     ! rank whose packed range holds g
        do while (lo < hi)
          mid = (lo + hi + 1) / 2
          if (2_8 * sub0(mid) <= g) then
            lo = mid
          else
            hi = mid - 1
          endif
        enddo
        p = lo
      endif
      off = g - 2_8 * sub0(p)
      if (off < nsub(p)) then
        f = 1; s = sub0(p) + off
      else
        f = 2; s = sub0(p) + off - nsub(p)
      endif
    end subroutine decode

    !> Source in A of target entry (sub row sr_, field fr; sub col sc_, field
    !! fc); only between the slots the extraction keeps (harm_kept).
    integer(c_int32_t) function locate(sr_, sc_, fr, fc)
      integer(8), intent(in) :: sr_, sc_
      integer, intent(in)    :: fr, fc
      integer :: m, q
      integer(8) :: arow, acol
      m = int(mod(sr_, int(n_tor, 8))); q = int(mod(sc_, int(n_tor, 8)))
      locate = 0
      if (.not. harm_kept(m, q)) return
      arow = (sr_ / n_tor) * bsa + (t%e(fr) - 1) * n_tor + m
      acol = (sc_ / n_tor) * bsa + (t%v(fc) - 1) * n_tor + q
      locate = find_baij(arow, acol, int(bsa), node0, aia, aja, oia, oja, ga)
    end function locate

  end subroutine build_map

  !> Position (+ diagonal part, - off-diagonal part, 0 absent) of global
  !! scalar entry (row, col) in a BAIJ matrix of block size bs whose local
  !! block rows start at node0. Blocks are stored column-major.
  integer(c_int32_t) function find_baij(row, col, bs, node0, dia, dja, oia, oja, garr) result(pos)
    integer(8), intent(in) :: row, col
    integer, intent(in)    :: bs, node0
    PetscInt, pointer      :: dia(:), dja(:), oia(:), oja(:), garr(:)
    integer :: br, bc, rw, cw, k, nloc
    br = int(row / bs) - node0; rw = int(mod(row, int(bs, 8)))
    bc = int(col / bs);         cw = int(mod(col, int(bs, 8)))
    nloc = size(dia) - 1
    pos = 0
    if (bc >= node0 .and. bc < node0 + nloc) then
      do k = int(dia(br + 1)) + 1, int(dia(br + 2))
        if (dja(k) == bc - node0) then
          pos = int((k - 1) * bs * bs + cw * bs + rw + 1, c_int32_t)
          return
        endif
      enddo
    else
      do k = int(oia(br + 1)) + 1, int(oia(br + 2))
        if (garr(oja(k) + 1) == bc) then
          pos = -int((k - 1) * bs * bs + cw * bs + rw + 1, c_int32_t)
          return
        endif
      enddo
    endif
  end function find_baij

  !> Gather one target's values from A, straight into its CSR arrays.
  subroutine gather_one(t, A)
    type(tgt_t), intent(inout) :: t
    Mat, intent(in)     :: A
    Mat :: Ad, Ao, Td, To
    PetscInt, pointer :: ga(:), gt(:)
    type(c_ptr) :: pad, pao, ptd, pto
    real(c_double), pointer :: va_d(:), va_o(:), vt_d(:), vt_o(:)
    integer(c_int) :: rc
    integer :: k

    call split_parts(A, .true., Ad, Ao, ga)
    call split_parts(t%T, .false., Td, To, gt)
    rc = c_baij_get(transfer(Ad%v, 0_c_intptr_t), pad); call c_f_pointer(pad, va_d, [max(na_d, 1_8)])
    rc = c_baij_get(transfer(Ao%v, 0_c_intptr_t), pao); call c_f_pointer(pao, va_o, [max(na_o, 1_8)])
    rc = c_aij_get(transfer(Td%v, 0_c_intptr_t), ptd);  call c_f_pointer(ptd, vt_d, [max(size(t%md), 1)])
    rc = c_aij_get(transfer(To%v, 0_c_intptr_t), pto);  call c_f_pointer(pto, vt_o, [max(size(t%mo), 1)])

    !$omp parallel do schedule(static)
    do k = 1, size(t%md)
      vt_d(k) = src(t%md(k), va_d, va_o)
    enddo
    !$omp end parallel do
    !$omp parallel do schedule(static)
    do k = 1, size(t%mo)
      vt_o(k) = src(t%mo(k), va_d, va_o)
    enddo
    !$omp end parallel do

    rc = c_aij_restore(transfer(To%v, 0_c_intptr_t), pto)
    rc = c_aij_restore(transfer(Td%v, 0_c_intptr_t), ptd)
    rc = c_baij_restore(transfer(Ao%v, 0_c_intptr_t), pao)
    rc = c_baij_restore(transfer(Ad%v, 0_c_intptr_t), pad)
  end subroutine gather_one

  pure real*8 function src(m, d, o)
    integer(c_int32_t), intent(in) :: m
    real(c_double), intent(in)     :: d(:), o(:)
    if (m > 0) then
      src = d(m)
    else if (m < 0) then
      src = o(-m)
    else
      src = 0.d0
    endif
  end function src

#endif
end module mod_petsc_pc_sf_gather
