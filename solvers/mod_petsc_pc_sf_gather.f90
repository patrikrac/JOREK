module mod_petsc_pc_sf_gather
#ifdef USE_PETSC
  use mpi_mod
  use iso_c_binding
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_raw_csr, only: split_parts, get_ij, put_ij, &
                               c_baij_get, c_baij_restore, c_aij_get, c_aij_restore
  implicit none
  private

  !--------------------------------------------------------------------
  !> Rebuild-time refill of the SF path's operators by a precomputed VALUE MAP.
  !!
  !! The operators the SF path reads -- the two packed pairs and the coupling
  !! blocks -- all have patterns fixed by JOREK's BAIJ Jacobian, which is fixed
  !! for the run. So everything about WHERE an entry comes from is structure,
  !! computed once: for every stored entry of every target, the position of its
  !! source value in the BAIJ value arrays of A (and, for pair_w's S_uu part,
  !! of W). A PC rebuild then only gathers: one indexed copy per entry, straight
  !! into the targets' CSR value arrays, with no MatGetRow expansion of the BAIJ
  !! block rows, no MatSetValues search, no pattern union and no packing.
  !!
  !! sfg_build   first build only: the targets already hold their values from
  !!             the conventional extraction; build the maps, then gather once
  !!             into scratch and require the result to be IDENTICAL (a wrong
  !!             map is otherwise invisible to every norm the build prints).
  !! sfg_gather  every later rebuild.
  !!
  !! The maps assume the frozen pattern. If A's or W's nonzero state or object
  !! changes, sfg_gather stops loudly rather than silently dropping entries.
  !!
  !! The raw CSR access (value arrays, diagonal/off-diagonal split) is in
  !! mod_petsc_raw_csr.
  !--------------------------------------------------------------------

  !> One target operator and its map. md/mo index the target's diagonal and
  !! off-diagonal CSR values; each entry is +k (k-th value of A's diagonal
  !! BAIJ part), -k (of A's off-diagonal part) or 0 (no A contribution).
  !! wd/wo: the same into W, for pair_w's S_uu entries only.
  type :: tgt_t
    Mat     :: T
    integer :: e(2) = 0, v(2) = 0      !< equations (row fields) / variables (column fields)
    logical :: pair = .false., with_w = .false.
    integer(c_int32_t), allocatable :: md(:), mo(:), wd(:), wo(:)
  end type tgt_t

  integer, parameter :: MAXT = 16
  type(tgt_t), save :: tg(MAXT)
  integer, save     :: ntg = 0
  integer(8), save  :: a_nzst = -1, w_nzst = -1, a_id = -1, w_id = -1
  integer(8), save  :: na_d = 0, na_o = 0, nw_d = 0, nw_o = 0   !< value-array lengths
  logical, save     :: ready = .false.

  public :: sfg_build, sfg_gather, sfg_ready

contains

  logical function sfg_ready()
    sfg_ready = ready
  end function sfg_ready

  !--------------------------------------------------------------------
  !> Build the maps (first build only). blocks(k) = A(eqs(k), vrs(k)) in the
  !! sub-block layout; Kpj / Sw the packed pairs (pair_psi, pair_w). All of
  !! them already hold the conventionally extracted values.
  !--------------------------------------------------------------------
  subroutine sfg_build(A, W, blocks, eqs, vrs, Kpj, pj_e, pj_v, Sw, w_e, w_v, comm, my_id)
    Mat, intent(in)     :: A, W
    Mat, intent(in)     :: blocks(:), Kpj, Sw
    integer, intent(in) :: eqs(:), vrs(:), pj_e(2), pj_v(2), w_e(2), w_v(2)
    integer, intent(in) :: comm, my_id
    integer :: k
    real*8  :: dmax, dglob
    integer :: mpierr
    PetscErrorCode :: ierr

    call require_baij(A, "A"); call require_baij(W, "W")
    ntg = 0
    do k = 1, size(blocks)
      ntg = ntg + 1
      tg(ntg)%T = blocks(k); tg(ntg)%e = [eqs(k), 0]; tg(ntg)%v = [vrs(k), 0]
    enddo
    ntg = ntg + 1
    tg(ntg)%T = Kpj; tg(ntg)%e = pj_e; tg(ntg)%v = pj_v; tg(ntg)%pair = .true.
    ntg = ntg + 1
    tg(ntg)%T = Sw;  tg(ntg)%e = w_e;  tg(ntg)%v = w_v;  tg(ntg)%pair = .true.
    tg(ntg)%with_w = .true.

    do k = 1, ntg
      call build_map(tg(k), A, W, comm)
    enddo
    call source_state(A, a_id, a_nzst)
    call source_state(W, w_id, w_nzst)
    ready = .true.

    ! the gate: a gather must reproduce what the conventional path stored
    dmax = 0.d0
    do k = 1, ntg
      dmax = max(dmax, gather_one(tg(k), A, W, .true.))
    enddo
    call MPI_Allreduce(dmax, dglob, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, mpierr)
    if (my_id == 0) write(*,'(A,I0,A,ES9.2)') "[Physics PC]   value maps: ", ntg, &
      " operators, gather vs extraction max |diff| = ", dglob
    if (dglob /= 0.d0) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: the value-map gather does not "// &
        "reproduce the extracted operators."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
  end subroutine sfg_build

  !--------------------------------------------------------------------
  !> Refill every target from A and W (every rebuild after the first).
  !--------------------------------------------------------------------
  subroutine sfg_gather(A, W, my_id)
    Mat, intent(in)     :: A, W
    integer, intent(in) :: my_id
    integer(8) :: id, nz
    real*8 :: d
    integer :: k
    PetscErrorCode :: ierr

    call source_state(A, id, nz)
    if (id /= a_id .or. nz /= a_nzst) call frozen_fail("A")
    call source_state(W, id, nz)
    if (id /= w_id .or. nz /= w_nzst) call frozen_fail("W")
    do k = 1, ntg
      d = gather_one(tg(k), A, W, .false.)
      ! bump the object state: MUMPS and the GMG decide "new values" by it
      call MatAssemblyBegin(tg(k)%T, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(tg(k)%T, MAT_FINAL_ASSEMBLY, ierr)
    enddo
  contains
    subroutine frozen_fail(what)
      character(len=*), intent(in) :: what
      if (my_id == 0) write(*,'(A,A,A)') "[Physics PC]   FATAL: the ", what, &
        " operator changed object or pattern; the SF value maps assume both frozen."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine frozen_fail
  end subroutine sfg_gather

  !====================================================================
  ! internals
  !====================================================================

  subroutine require_baij(M, what)
    Mat, intent(in) :: M
    character(len=*), intent(in) :: what
    MatType :: mt
    PetscErrorCode :: ierr
    call MatGetType(M, mt, ierr)
    if (trim(mt) /= MATMPIBAIJ) then
      write(*,'(A,A,A,A)') "[Physics PC]   FATAL: the SF value maps need ", what, &
        " as MATMPIBAIJ, got ", trim(mt)
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
  subroutine build_map(t, A, W, comm)
    use mod_parameters, only: n_tor, n_var
    type(tgt_t), intent(inout) :: t
    Mat, intent(in)     :: A, W
    integer, intent(in) :: comm

    Mat :: Ad, Ao, Wd, Wo, Td, To
    PetscInt, pointer :: ga(:), gw(:), gt(:)
    PetscInt, pointer :: aia(:), aja(:), oia(:), oja(:), wia(:), wja(:), woia(:), woja(:)
    PetscInt, pointer :: tia(:), tja(:), toia(:), toja(:)
    PetscInt :: nad, nao, nwd, nwo, ntd, nto, rs, re, cs, ce, bsa
    integer, allocatable :: sub0(:), nsub(:)
    integer :: np, me, mpierr, part, lr, k, f_r, f_c
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
    if (t%with_w) then
      call split_parts(W, .true., Wd, Wo, gw)
      call get_ij(Wd, .true., nwd, wia, wja)
      call get_ij(Wo, .true., nwo, woia, woja)
      nw_d = int(wia(nwd + 1), 8) * n_tor * n_tor
      nw_o = int(woia(nwo + 1), 8) * n_tor * n_tor
    endif
    if (max(na_d, na_o, nw_d, nw_o) >= int(huge(0_c_int32_t), 8)) then
      write(*,'(A)') "[Physics PC]   FATAL: a BAIJ value array exceeds the 32-bit map index."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif

    ! the sub-block ranges of every rank (packed-pair column decoding)
    call MatGetOwnershipRange(t%T, rs, re, ierr)
    call MatGetOwnershipRangeColumn(t%T, cs, ce, ierr)
    allocate(sub0(0:np), nsub(0:np - 1))
    block
      integer :: mysub0, mynsub
      if (t%pair) then
        mynsub = int(re - rs) / 2; mysub0 = int(rs) / 2
      else
        mynsub = int(re - rs); mysub0 = int(rs)
      endif
      call MPI_Allgather(mysub0, 1, MPI_INTEGER, sub0, 1, MPI_INTEGER, comm, mpierr)
      call MPI_Allgather(mynsub, 1, MPI_INTEGER, nsub, 1, MPI_INTEGER, comm, mpierr)
      sub0(np) = sub0(np - 1) + nsub(np - 1)
    end block

    call split_parts(t%T, .false., Td, To, gt)
    call get_ij(Td, .false., ntd, tia, tja)
    call get_ij(To, .false., nto, toia, toja)
    if (allocated(t%md)) deallocate(t%md, t%mo)
    allocate(t%md(tia(ntd + 1)), t%mo(toia(nto + 1)))
    t%md = 0; t%mo = 0
    if (t%with_w) then
      if (allocated(t%wd)) deallocate(t%wd, t%wo)
      allocate(t%wd(tia(ntd + 1)), t%wo(toia(nto + 1)))
      t%wd = 0; t%wo = 0
    endif

    do part = 1, 2                                   ! 1 = diagonal, 2 = off-diagonal
      do lr = 0, int(ntd) - 1
        call decode(int(rs, 8) + lr, .true., sr, f_r)
        if (part == 1) then
          do k = int(tia(lr + 1)) + 1, int(tia(lr + 2))
            call decode(int(cs, 8) + tja(k), .false., sc, f_c)
            call locate(t%md(k), t%wd, k, sr, sc, f_r, f_c)
          enddo
        else
          do k = int(toia(lr + 1)) + 1, int(toia(lr + 2))
            call decode(int(gt(toja(k) + 1), 8), .false., sc, f_c)
            call locate(t%mo(k), t%wo, k, sr, sc, f_r, f_c)
          enddo
        endif
      enddo
    enddo

    call put_ij(Td, .false., ntd, tia, tja); call put_ij(To, .false., nto, toia, toja)
    call put_ij(Ad, .true., nad, aia, aja);  call put_ij(Ao, .true., nao, oia, oja)
    if (t%with_w) then
      call put_ij(Wd, .true., nwd, wia, wja); call put_ij(Wo, .true., nwo, woia, woja)
    endif
    deallocate(sub0, nsub)

  contains

    !> Target global index -> (sub-block index, field 1|2).
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

    !> Map entry k: the source of target (sub row sr, field f_r; sub col sc,
    !! field f_c) in A, and for pair_w's (1,1) block also in W.
    subroutine locate(mk, wmap, k, sr_, sc_, fr, fc)
      integer(c_int32_t), intent(out) :: mk
      integer(c_int32_t), allocatable, intent(inout) :: wmap(:)
      integer, intent(in)    :: k, fr, fc
      integer(8), intent(in) :: sr_, sc_
      integer :: e, v, m, q
      integer(8) :: arow, acol
      e = t%e(fr); v = t%v(fc)
      m = int(mod(sr_, int(n_tor, 8))); q = int(mod(sc_, int(n_tor, 8)))
      arow = (sr_ / n_tor) * bsa + (e - 1) * n_tor + m
      acol = (sc_ / n_tor) * bsa + (v - 1) * n_tor + q
      ! A contributes only between slots of the same |n| (cos and sin of one n
      ! together): the extraction's harm_split filter. W is NOT filtered, so
      ! pair_w's S_uu holds cross-|n| entries that come from W alone.
      mk = 0
      if ((q + 1) / 2 == (m + 1) / 2) mk = find_baij(arow, acol, bsa, rs_node(), aia, aja, oia, oja, ga)
      if (t%with_w .and. fr == 1 .and. fc == 1) &
        wmap(k) = find_baij(sr_, sc_, int(n_tor), rs_node(), wia, wja, woia, woja, gw)
    end subroutine locate

    !> This rank's first JOREK node (= first block row of A and of W).
    integer function rs_node()
      if (t%pair) then
        rs_node = int(rs) / 2 / n_tor
      else
        rs_node = int(rs) / n_tor
      endif
    end function rs_node

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

  !--------------------------------------------------------------------
  !> Gather one target from A (and W). check = .true.: compare against the
  !! values the target holds instead of writing, and return max |diff|.
  !--------------------------------------------------------------------
  real*8 function gather_one(t, A, W, check) result(dmax)
    type(tgt_t), intent(inout) :: t
    Mat, intent(in)     :: A, W
    logical, intent(in) :: check
    Mat :: Ad, Ao, Wd, Wo, Td, To
    PetscInt, pointer :: ga(:), gw(:), gt(:)
    type(c_ptr) :: pad, pao, pwd, pwo, ptd, pto
    real(c_double), pointer :: va_d(:), va_o(:), vw_d(:), vw_o(:), vt_d(:), vt_o(:)
    integer(c_int) :: rc
    integer :: k
    real*8 :: v

    dmax = 0.d0
    call split_parts(A, .true., Ad, Ao, ga)
    call split_parts(t%T, .false., Td, To, gt)
    rc = c_baij_get(transfer(Ad%v, 0_c_intptr_t), pad); call c_f_pointer(pad, va_d, [max(na_d, 1_8)])
    rc = c_baij_get(transfer(Ao%v, 0_c_intptr_t), pao); call c_f_pointer(pao, va_o, [max(na_o, 1_8)])
    rc = c_aij_get(transfer(Td%v, 0_c_intptr_t), ptd);  call c_f_pointer(ptd, vt_d, [max(size(t%md), 1)])
    rc = c_aij_get(transfer(To%v, 0_c_intptr_t), pto);  call c_f_pointer(pto, vt_o, [max(size(t%mo), 1)])
    if (t%with_w) then
      call split_parts(W, .true., Wd, Wo, gw)
      rc = c_baij_get(transfer(Wd%v, 0_c_intptr_t), pwd); call c_f_pointer(pwd, vw_d, [max(nw_d, 1_8)])
      rc = c_baij_get(transfer(Wo%v, 0_c_intptr_t), pwo); call c_f_pointer(pwo, vw_o, [max(nw_o, 1_8)])
    endif

    !$omp parallel do private(v) reduction(max:dmax) schedule(static)
    do k = 1, size(t%md)
      v = src(t%md(k), va_d, va_o)
      if (t%with_w) v = v + src(t%wd(k), vw_d, vw_o)
      if (check) then
        dmax = max(dmax, abs(v - vt_d(k)))
      else
        vt_d(k) = v
      endif
    enddo
    !$omp end parallel do
    !$omp parallel do private(v) reduction(max:dmax) schedule(static)
    do k = 1, size(t%mo)
      v = src(t%mo(k), va_d, va_o)
      if (t%with_w) v = v + src(t%wo(k), vw_d, vw_o)
      if (check) then
        dmax = max(dmax, abs(v - vt_o(k)))
      else
        vt_o(k) = v
      endif
    enddo
    !$omp end parallel do

    rc = c_aij_restore(transfer(To%v, 0_c_intptr_t), pto)
    rc = c_aij_restore(transfer(Td%v, 0_c_intptr_t), ptd)
    rc = c_baij_restore(transfer(Ao%v, 0_c_intptr_t), pao)
    rc = c_baij_restore(transfer(Ad%v, 0_c_intptr_t), pad)
    if (t%with_w) then
      rc = c_baij_restore(transfer(Wo%v, 0_c_intptr_t), pwo)
      rc = c_baij_restore(transfer(Wd%v, 0_c_intptr_t), pwd)
    endif
  end function gather_one

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
