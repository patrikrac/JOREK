!> The SF preconditioner's operators, extracted straight from JOREK's BAIJ
!! Jacobian A and the force operator W into packed MPIAIJ matrices.
!!
!! Operator rows are (field, JOREK index, toroidal slot), the slot fastest and
!! the fields of a rank one after the other, as the GMG expects. On the global
!! layout every rank keeps A's indices and all slots. Under the mode split the
!! ranks form one group per |n| family (the n = 0 slot, then the cos and sin
!! slots of one n), sized in proportion to its slots; each group holds its
!! family's operators on an equal share of the indices.
!!
!! The first build sends each operator row's structure from the rank owning
!! A's rows of that index to the row's owner and records where every entry's
!! value sits in A's (and W's) value arrays; a refill is then one gather and
!! one message per (source, owner, field).
module mod_petsc_pc_sf_ops
#ifdef USE_PETSC
  use mpi_mod
  use iso_c_binding
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_raw_csr, only: split_parts, get_ij, put_ij, c_baij_get, c_baij_restore
  implicit none
  private

  !> fields per operator (vs_mono's whole Jacobian with v_par); the MPI tags
  !! of an operator's fields, TAG_* + field - 1, stay apart up to 8
  integer, parameter, public :: SFO_NFMAX = 7

  !> Operator of nf x nf fields: block (a, b) is A(se(a,b), sv(a,b)) (JOREK
  !! variables, 0 = none), times opz where scl(a,b) = 1, plus W source wid(a,b)
  !! (0 = none). The W sources of one operator must share a block pattern.
  type, public :: sf_op_t
    character(len=24) :: label = ""
    integer :: nf = 0, tag = 0
    integer :: se(SFO_NFMAX, SFO_NFMAX) = 0, sv(SFO_NFMAX, SFO_NFMAX) = 0
    integer :: scl(SFO_NFMAX, SFO_NFMAX) = 0, wid(SFO_NFMAX, SFO_NFMAX) = 0
    logical :: has_w = .false.
    Mat     :: mat
    real(c_double), allocatable :: val(:)              !< values in CSR order
    ! owner side: one message per (source rank, field) into val
    integer :: nrecv = 0
    integer, allocatable :: rsrc(:), rtag(:), roff(:), rcnt(:)
    ! source side: one message per (owner rank, field)
    integer :: nsend = 0
    integer, allocatable :: sdst(:), stag(:), soff(:), scnt(:)
    integer(c_int32_t), allocatable :: sa(:), sw(:)    !< +k / -k: k-th value of A's (W's) diagonal / off-diagonal part
    integer(c_int8_t), allocatable :: ss(:), sc(:)     !< W source of sw; 1 = times opz
  end type sf_op_t

  logical, save, public  :: sfo_split = .false.         !< mode split layout
  integer, save, public  :: sfo_comm = MPI_COMM_NULL   !< the operators' communicator
  integer, save, public  :: sfo_fam = 0                 !< this rank's family (1-based)
  integer, save, public  :: sfo_nfam = 0
  PetscInt, save, public :: sfo_n1 = 0                  !< local rows per variable

  logical, save :: ready = .false., keep_cross = .true.
  integer, save :: gcomm = MPI_COMM_NULL, gme = -1, gnp = 0
  integer, save :: n_idx = 0                            !< JOREK indices (A's block rows)
  integer, save :: i0 = 0, i1 = 0                       !< my operator indices [i0, i1)
  integer, allocatable, save :: fr0(:)                  !< first global rank of each family, fr0(nfam+1) = gnp
  integer, allocatable, save :: ix0(:)                  !< first operator index of each global rank
  integer, allocatable, save :: idx0(:)                 !< first index of A's rows on each global rank, idx0(gnp) = n_idx
  PetscInt, allocatable, save :: rsx(:)                 !< first full-system row of each global rank
  integer, allocatable, save :: myslots(:)
  VecScatter, save :: scv
  Vec, save :: stagev
  integer, save :: nsv = 0
  logical, save :: vec_ok = .false.

  integer, parameter :: TAG_LEN = 0, TAG_COL = 8, TAG_VAL = 16, TAG_STRIDE = 32

  public :: sfo_layout, sfo_build, sfo_fill, sfo_vars_in, sfo_vars_out, sfo_define, sfo_b65_used
  public :: sfo_block_used, sfo_destroy

contains

  !> op = the full nf x nf operator of the JOREK variables vars: block (a, b)
  !! is A(vars(a), vars(b)), nf = size(vars).
  subroutine sfo_define(op, label, tag, vars)
    type(sf_op_t), intent(inout) :: op
    character(len=*), intent(in) :: label
    integer, intent(in) :: tag, vars(:)
    integer :: a
    op%label = label; op%nf = size(vars); op%tag = tag
    do a = 1, size(vars)
      op%se(a, 1:size(vars)) = vars(a); op%sv(1:size(vars), a) = vars(a)
    enddo
  end subroutine sfo_define

  !> Is B_65 (T row, rho column) nonzero anywhere on comm? It vanishes in
  !! model199, and the sweeps apply it only where it does not.
  logical function sfo_b65_used(b65, comm)
    type(sf_op_t), intent(in) :: b65
    integer, intent(in) :: comm
    real*8 :: nrm
    integer :: mpierr, me
    sfo_b65_used = sfo_block_used(b65, comm, nrm)
    call MPI_Comm_rank(comm, me, mpierr)
    if (me == 0 .and. sfo_b65_used) write(*,'(A,ES10.3,A)') "[Physics PC]   SF: |B_65|_F = ", nrm, &
      " (max over ranks), applied in the rho -> T sweep"
  end function sfo_b65_used

  !> Is op nonzero anywhere on comm? nrm: its Frobenius norm, the maximum
  !! over comm (op's communicator may be a part of comm). Collective on comm,
  !! silent.
  logical function sfo_block_used(op, comm, nrm)
    type(sf_op_t), intent(in) :: op
    integer, intent(in) :: comm
    real*8, intent(out), optional :: nrm
    real*8 :: s
    integer :: mpierr
    PetscErrorCode :: ierr
    call MatNorm(op%mat, NORM_FROBENIUS, s, ierr)
    s = s**2
    call MPI_Allreduce(MPI_IN_PLACE, s, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm, mpierr)
    sfo_block_used = (s > 0.d0)
    if (present(nrm)) nrm = sqrt(s)
  end function sfo_block_used

  !> Free op's matrix, values and message plans (an operator no longer
  !! needed); its layout definition stays. Rank-local.
  subroutine sfo_destroy(op)
    type(sf_op_t), intent(inout) :: op
    PetscErrorCode :: ierr
    call MatDestroy(op%mat, ierr)
    if (allocated(op%val)) deallocate(op%val)
    if (allocated(op%rsrc)) deallocate(op%rsrc, op%rtag, op%roff, op%rcnt)
    if (allocated(op%sdst)) deallocate(op%sdst, op%stag, op%soff, op%scnt)
    if (allocated(op%sa)) deallocate(op%sa)
    if (allocated(op%sw)) deallocate(op%sw, op%ss)
    if (allocated(op%sc)) deallocate(op%sc)
    op%nrecv = 0; op%nsend = 0; op%has_w = .false.
  end subroutine sfo_destroy

  !> Fix the layout, once: split = the mode split; cross = keep the entries
  !! between different |n| (global layout only).
  subroutine sfo_layout(A, comm, split, cross, my_id)
    use mod_parameters,    only: n_tor, n_var
    use mod_petsc_pc_harm, only: pc_harm_set
    Mat, intent(in)     :: A
    integer, intent(in) :: comm, my_id
    logical, intent(in) :: split, cross
    integer, allocatable :: nrk(:), nl(:)
    real*8, allocatable :: ideal(:)
    integer :: f, r, k, mpierr
    PetscInt :: rs, re
    PetscErrorCode :: ierr

    if (ready) return
    sfo_split = split
    keep_cross = cross .and. .not. split
    gcomm = comm
    call MPI_Comm_rank(comm, gme, mpierr)
    call MPI_Comm_size(comm, gnp, mpierr)

    allocate(nl(0:gnp - 1), idx0(0:gnp), rsx(0:gnp - 1), ix0(0:gnp - 1))
    call MatGetOwnershipRange(A, rs, re, ierr)
    call MPI_Allgather(int((re - rs) / (n_var * n_tor)), 1, MPI_INTEGER, nl, 1, MPI_INTEGER, comm, mpierr)
    call MPI_Allgather(rs, 1, MPIU_INTEGER, rsx, 1, MPIU_INTEGER, comm, mpierr)
    idx0(0) = 0
    do r = 0, gnp - 1
      idx0(r + 1) = idx0(r) + nl(r)
    enddo
    n_idx = idx0(gnp)

    ! ranks per family in proportion to its slots, largest remainder first
    sfo_nfam = merge((n_tor + 1) / 2, 1, split)
    if (gnp < sfo_nfam) then
      if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   FATAL: physics_pc_sf_mode_split needs at "// &
        "least one rank per |n| family: ", sfo_nfam, " families, ", gnp, " ranks"
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    endif
    allocate(nrk(sfo_nfam), ideal(sfo_nfam), fr0(sfo_nfam + 1))
    ideal = 2.d0; ideal(1) = 1.d0
    ideal = gnp * ideal / sum(ideal)
    nrk = max(1, int(ideal))
    do while (sum(nrk) < gnp)
      f = maxloc(ideal - nrk, 1); nrk(f) = nrk(f) + 1
    enddo
    do while (sum(nrk) > gnp)
      f = maxloc(merge(dble(nrk) - ideal, -huge(1.d0), nrk > 1), 1); nrk(f) = nrk(f) - 1
    enddo
    fr0(1) = 0
    do f = 1, sfo_nfam
      fr0(f + 1) = fr0(f) + nrk(f)
      do r = fr0(f), fr0(f + 1) - 1
        if (split) then
          ix0(r) = int(int(r - fr0(f), 8) * n_idx / nrk(f))
        else
          ix0(r) = idx0(r)
        endif
      enddo
      if (gme >= fr0(f) .and. gme < fr0(f + 1)) sfo_fam = f
    enddo

    myslots = fslots(sfo_fam)
    if (split) then
      call MPI_Comm_split(comm, sfo_fam, gme, sfo_comm, mpierr)
      call pc_harm_set(myslots)
    else
      sfo_comm = comm
    endif
    k = gme - fr0(sfo_fam)
    i0 = frange(sfo_fam, k)
    i1 = frange(sfo_fam, k + 1)
    sfo_n1 = int(i1 - i0, kind(sfo_n1)) * size(myslots)
    ready = .true.
    if (split) call report_layout(my_id, nrk)
  end subroutine sfo_layout

  !> First index of family f's rank q (q = its rank count: n_idx).
  integer function frange(f, q)
    integer, intent(in) :: f, q
    if (q >= fr0(f + 1) - fr0(f)) then
      frange = n_idx
    else
      frange = ix0(fr0(f) + q)
    endif
  end function frange

  !> Family f's rank owning index i.
  integer function fowner(f, i)
    integer, intent(in) :: f, i
    integer :: lo, hi, mid
    lo = 0; hi = fr0(f + 1) - fr0(f) - 1
    do while (lo < hi)
      mid = (lo + hi + 1) / 2
      if (frange(f, mid) <= i) then
        lo = mid
      else
        hi = mid - 1
      endif
    enddo
    fowner = lo
  end function fowner

  !> Global rank holding A's rows of index i.
  integer function sowner(i)
    integer, intent(in) :: i
    integer :: lo, hi, mid
    lo = 0; hi = gnp - 1
    do while (lo < hi)
      mid = (lo + hi + 1) / 2
      if (idx0(mid) <= i) then
        lo = mid
      else
        hi = mid - 1
      endif
    enddo
    sowner = lo
  end function sowner

  !> Family f's global slots.
  function fslots(f) result(s)
    use mod_parameters, only: n_tor
    integer, intent(in) :: f
    integer, allocatable :: s(:)
    integer :: m
    if (.not. sfo_split) then
      s = [(m, m = 0, n_tor - 1)]
    else if (f == 1) then
      s = [0]
    else
      s = [2 * f - 3, 2 * f - 2]
    endif
  end function fslots

  !> Is the entry between slots m and q kept? Slot m belongs to |n| group (m+1)/2.
  pure logical function kept(m, q)
    integer, intent(in) :: m, q
    kept = keep_cross .or. (m + 1) / 2 == (q + 1) / 2
  end function kept

  subroutine report_layout(my_id, nrk)
    integer, intent(in) :: my_id, nrk(:)
    integer :: f, q, nmin, nmax
    if (my_id /= 0) return
    write(*,'(A,I0,A,I0,A)') "[Physics PC]   SF mode split: ", sfo_nfam, " |n| families on ", gnp, " ranks"
    do f = 1, sfo_nfam
      nmin = huge(1); nmax = 0
      do q = 0, nrk(f) - 1
        nmin = min(nmin, frange(f, q + 1) - frange(f, q)); nmax = max(nmax, frange(f, q + 1) - frange(f, q))
      enddo
      if (f == 1) then
        write(*,'(A,I0,A,I0,A,I0,A,I0)') "[Physics PC]     family ", f, ": ", nrk(f), &
          " rank(s), slot 0 (n = 0), indices/rank ", nmin, "-", nmax
      else
        write(*,'(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') "[Physics PC]     family ", f, ": ", nrk(f), &
          " rank(s), slots ", 2 * f - 3, ", ", 2 * f - 2, " (|n| group ", f - 1, "), indices/rank ", &
          nmin, "-", nmax
      endif
    enddo
  end subroutine report_layout

  !> First build of op: row structure from the source ranks, the matrix on the
  !! owner, the message plans, and the first values. Collective on the global
  !! communicator.
  subroutine sfo_build(op, Aj, W, opz, my_id)
    use mod_parameters, only: n_tor, n_var
    type(sf_op_t), intent(inout) :: op
    Mat, intent(in)     :: Aj, W(:)
    real*8, intent(in)  :: opz
    integer, intent(in) :: my_id

    Mat :: Ad, Ao, Wd, Wo
    PetscInt, pointer :: ga(:), gw(:), aia(:), aja(:), oia(:), oja(:)
    PetscInt, pointer :: wia(:), wja(:), woia(:), woja(:)
    PetscInt :: nad, nao, nwd, nwo, bsx
    integer :: bsa, nf, nloc_i, me0, pass, f, q, a, b, i, l, s, sq, m, qs, iw0, iw
    integer :: nmsg, ns, ent, row, nrow, e, v, u, k, jq, qj, nq, nu, nkeep
    integer(8) :: na_d, na_o, nw_d, nw_o, ndrop, nd_glob
    integer, allocatable :: sl(:), uj(:), ka(:), kw(:)        ! one index's block columns: A / W block (+-k, 0)
    integer, allocatable :: srow(:), mrow0(:), ment0(:)
    PetscInt, allocatable :: scol(:)
    integer :: nq_me, ns_me, nloc, r, ra, rb, nreq, mpierr
    PetscInt, allocatable :: fi(:), fj(:)
    integer, allocatable :: rlen(:), rrow0(:), rnrow(:), reqs(:)
    PetscErrorCode :: ierr

    nf = op%nf
    if (nf > SFO_NFMAX) call fail("more fields than SFO_NFMAX")
    if (any(op%wid(1:nf, 1:nf) < 0) .or. any(op%wid(1:nf, 1:nf) > size(W))) call fail("W source outside W(:)")
    op%has_w = any(op%wid(1:nf, 1:nf) > 0)
    bsa = n_var * n_tor
    call MatGetBlockSize(Aj, bsx, ierr)
    if (bsx /= bsa) call fail("A's block size is not n_var * n_tor")
    call split_parts(Aj, .true., Ad, Ao, ga)
    call get_ij(Ad, .true., nad, aia, aja)
    call get_ij(Ao, .true., nao, oia, oja)
    na_d = int(aia(nad + 1), 8) * bsa * bsa
    na_o = int(oia(nao + 1), 8) * bsa * bsa
    nw_d = 0; nw_o = 0
    iw0 = 0
    if (op%has_w) then
      iw0 = minval(op%wid(1:nf, 1:nf), mask=op%wid(1:nf, 1:nf) > 0)
      call MatGetBlockSize(W(iw0), bsx, ierr)
      if (bsx /= n_tor) call fail("W's block size is not n_tor")
      call split_parts(W(iw0), .true., Wd, Wo, gw)
      call get_ij(Wd, .true., nwd, wia, wja)
      call get_ij(Wo, .true., nwo, woia, woja)
      if (nwd /= nad) call fail("W's block rows are not A's")
      nw_d = int(wia(nwd + 1), 8) * n_tor * n_tor
      nw_o = int(woia(nwo + 1), 8) * n_tor * n_tor
      do iw = 1, size(W)
        if (iw /= iw0 .and. any(op%wid(1:nf, 1:nf) == iw)) call same_pattern(W(iw))
      enddo
    endif
    if (max(na_d, na_o, nw_d, nw_o) >= int(huge(0_c_int32_t), 8)) &
      call fail("a BAIJ value array exceeds the 32-bit position index")
    nloc_i = int(nad)
    me0 = idx0(gme)
    if (nloc_i /= idx0(gme + 1) - me0) call fail("A's row layout changed since sfo_layout")

    ! Source side, two passes (count, fill) over my rows of every family. One
    ! message per (family owner, field); rows in (index, slot) order, entries
    ! in the owner's ascending column order: owner rank, field, index, slot.
    allocate(uj(64), ka(64), kw(64))
    ndrop = 0
    do pass = 1, 2
      nmsg = 0; ent = 0; nrow = 0
      do f = 1, sfo_nfam
        if (nloc_i == 0) cycle
        sl = fslots(f); ns = size(sl)
        do q = fowner(f, me0), fowner(f, me0 + nloc_i - 1)
          if (min(me0 + nloc_i, frange(f, q + 1)) <= max(me0, frange(f, q))) cycle
          do a = 1, nf
            nmsg = nmsg + 1
            if (pass == 2) then
              op%sdst(nmsg) = fr0(f) + q
              op%stag(nmsg) = TAG_STRIDE * op%tag + TAG_VAL + a - 1
              op%soff(nmsg) = ent
              mrow0(nmsg) = nrow
              ment0(nmsg) = ent
            endif
            do i = max(me0, frange(f, q)), min(me0 + nloc_i, frange(f, q + 1)) - 1
              l = i - me0
              call block_cols(l)
              do s = 1, ns
                m = sl(s)
                nrow = nrow + 1
                row = 0
                u = 1
                do while (u <= nu)
                  qj = fowner(f, uj(u))
                  k = u
                  do while (k < nu)
                    if (fowner(f, uj(k + 1)) /= qj) exit
                    k = k + 1
                  enddo
                  nq = frange(f, qj + 1) - frange(f, qj)
                  do b = 1, nf
                    e = op%se(a, b); v = op%sv(a, b)
                    do jq = u, k
                      if (.not. ((e > 0 .and. ka(jq) /= 0) .or. (op%wid(a, b) > 0 .and. kw(jq) /= 0))) cycle
                      do sq = 1, ns
                        qs = sl(sq)
                        if (.not. kept(m, qs)) cycle
                        row = row + 1
                        ent = ent + 1
                        if (pass == 1) cycle
                        scol(ent) = nf * frange(f, qj) * ns + (b - 1) * nq * ns + (uj(jq) - frange(f, qj)) * ns + sq - 1
                        op%sa(ent) = 0
                        if (e > 0 .and. ka(jq) /= 0) &
                          op%sa(ent) = apos(ka(jq), (e - 1) * n_tor + m, (v - 1) * n_tor + qs, bsa)
                        if (op%has_w) then
                          op%sw(ent) = 0; op%ss(ent) = 0
                          if (op%wid(a, b) > 0 .and. kw(jq) /= 0) then
                            op%sw(ent) = apos(kw(jq), m, qs, n_tor)
                            op%ss(ent) = int(op%wid(a, b), c_int8_t)
                          endif
                        endif
                        if (allocated(op%sc)) op%sc(ent) = int(op%scl(a, b), c_int8_t)
                      enddo
                    enddo
                  enddo
                  u = k + 1
                enddo
                if (pass == 2) srow(nrow) = row
                ! W's entries of this row that the layout drops
                if (pass == 1 .and. any(op%wid(a, 1:nf) > 0)) then
                  nkeep = count([(kept(m, sl(sq)), sq = 1, ns)])
                  ndrop = ndrop + int(count(op%wid(a, 1:nf) > 0) * count(kw(1:nu) /= 0), 8) * (n_tor - nkeep)
                endif
              enddo
            enddo
            if (pass == 2) op%scnt(nmsg) = ent - op%soff(nmsg)
          enddo
        enddo
      enddo
      if (pass == 1) then
        op%nsend = nmsg
        allocate(op%sdst(nmsg), op%stag(nmsg), op%soff(nmsg), op%scnt(nmsg), mrow0(nmsg + 1), ment0(nmsg + 1))
        allocate(op%sa(ent), srow(nrow), scol(ent))
        if (op%has_w) allocate(op%sw(ent), op%ss(ent))
        if (any(op%scl(1:nf, 1:nf) /= 0)) allocate(op%sc(ent))
      else
        mrow0(nmsg + 1) = nrow; ment0(nmsg + 1) = ent
      endif
    enddo
    deallocate(uj, ka, kw)
    call put_ij(Ad, .true., nad, aia, aja); call put_ij(Ao, .true., nao, oia, oja)
    if (op%has_w) then
      call put_ij(Wd, .true., nwd, wia, wja); call put_ij(Wo, .true., nwo, woia, woja)
      call MPI_Allreduce(ndrop, nd_glob, 1, MPI_INTEGER8, MPI_SUM, gcomm, mpierr)
      if (my_id == 0 .and. nd_glob > 0) write(*,'(A,A,A,I0,A)') "[Physics PC]   SF: ", trim(op%label), &
        ": ", nd_glob, " cross-|n| W entries dropped"
    endif

    ! Owner side: my rows of every field come from the source ranks of my
    ! indices, one message per (source, field).
    ns_me = size(myslots)
    nq_me = i1 - i0
    nloc = nf * nq_me * ns_me
    ra = 0; rb = -1
    if (nq_me > 0) then
      ra = sowner(i0); rb = sowner(i1 - 1)
    endif
    op%nrecv = 0
    do a = 1, nf
      do r = ra, rb
        if (min(i1, idx0(r + 1)) > max(i0, idx0(r))) op%nrecv = op%nrecv + 1
      enddo
    enddo
    allocate(op%rsrc(op%nrecv), op%rtag(op%nrecv), op%roff(op%nrecv), op%rcnt(op%nrecv), &
             rrow0(op%nrecv), rnrow(op%nrecv), rlen(nloc), fi(nloc + 1))
    k = 0
    do a = 1, nf
      do r = ra, rb
        if (min(i1, idx0(r + 1)) <= max(i0, idx0(r))) cycle
        k = k + 1
        op%rsrc(k) = r
        op%rtag(k) = TAG_STRIDE * op%tag + TAG_VAL + a - 1
        rrow0(k) = (a - 1) * nq_me * ns_me + (max(i0, idx0(r)) - i0) * ns_me
        rnrow(k) = (min(i1, idx0(r + 1)) - max(i0, idx0(r))) * ns_me
      enddo
    enddo

    ! the row lengths, then the columns
    allocate(reqs(op%nrecv + op%nsend))
    nreq = 0
    do k = 1, op%nrecv
      nreq = nreq + 1
      call MPI_Irecv(rlen(rrow0(k) + 1), rnrow(k), MPI_INTEGER, op%rsrc(k), &
                     op%rtag(k) - TAG_VAL + TAG_LEN, gcomm, reqs(nreq), mpierr)
    enddo
    do k = 1, op%nsend
      nreq = nreq + 1
      call MPI_Isend(srow(mrow0(k) + 1), mrow0(k + 1) - mrow0(k), MPI_INTEGER, op%sdst(k), &
                     op%stag(k) - TAG_VAL + TAG_LEN, gcomm, reqs(nreq), mpierr)
    enddo
    call MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE, mpierr)
    fi(1) = 0
    do row = 1, nloc
      fi(row + 1) = fi(row) + rlen(row)
    enddo
    do k = 1, op%nrecv
      op%roff(k) = int(fi(rrow0(k) + 1))
      op%rcnt(k) = int(fi(rrow0(k) + rnrow(k) + 1) - fi(rrow0(k) + 1))
    enddo
    allocate(fj(max(int(fi(nloc + 1)), 1)))
    nreq = 0
    do k = 1, op%nrecv
      nreq = nreq + 1
      call MPI_Irecv(fj(op%roff(k) + 1), op%rcnt(k), MPIU_INTEGER, op%rsrc(k), &
                     op%rtag(k) - TAG_VAL + TAG_COL, gcomm, reqs(nreq), mpierr)
    enddo
    do k = 1, op%nsend
      nreq = nreq + 1
      call MPI_Isend(scol(ment0(k) + 1), ment0(k + 1) - ment0(k), MPIU_INTEGER, op%sdst(k), &
                     op%stag(k) - TAG_VAL + TAG_COL, gcomm, reqs(nreq), mpierr)
    enddo
    call MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE, mpierr)
    deallocate(srow, scol, mrow0, ment0, rlen, rrow0, rnrow, reqs)

    allocate(op%val(max(int(fi(nloc + 1)), 1)))
    op%val = 0.d0
    call MatCreateMPIAIJWithArrays(sfo_comm, int(nloc, kind(fi)), int(nloc, kind(fi)), PETSC_DETERMINE, &
                                   PETSC_DETERMINE, fi, fj, op%val, op%mat, ierr)
    deallocate(fi, fj)
    call sfo_fill(op, Aj, W, opz)

  contains

    !> Fail unless W source Wm has the block pattern of W(iw0).
    subroutine same_pattern(Wm)
      Mat, intent(in) :: Wm
      Mat :: Xd, Xo
      PetscInt, pointer :: gx(:), xia(:), xja(:), xoia(:), xoja(:)
      PetscInt :: nxd, nxo
      logical :: same
      call split_parts(Wm, .true., Xd, Xo, gx)
      call get_ij(Xd, .true., nxd, xia, xja)
      call get_ij(Xo, .true., nxo, xoia, xoja)
      ! a rank without off-process columns has no column map (split_parts)
      same = nxd == nwd .and. nxo == nwo .and. (associated(gx) .eqv. associated(gw))
      if (same .and. associated(gx)) same = size(gx) == size(gw)
      if (same .and. associated(gx)) same = all(gx == gw)
      if (same) same = all(xia == wia) .and. all(xoia == woia)
      if (same) same = all(xja(1:xia(nxd + 1)) == wja(1:wia(nwd + 1)))
      if (same) same = all(xoja(1:xoia(nxo + 1)) == woja(1:woia(nwo + 1)))
      call put_ij(Xd, .true., nxd, xia, xja); call put_ij(Xo, .true., nxo, xoia, xoja)
      if (.not. same) call fail("the W sources of one operator have different block patterns")
    end subroutine same_pattern

    !> Block columns of local block row l_ of A and W, merged and ascending:
    !! uj(1:nu), with the A / W block (+k diagonal part, -k off-diagonal, 0 none).
    subroutine block_cols(l_)
      integer, intent(in) :: l_
      integer :: pa, pa_end, po, po_end, pw, pw_end, pv, pv_end, jd, jo, jwd, jwo, jmin
      nu = 0
      pa = int(aia(l_ + 1)) + 1; pa_end = int(aia(l_ + 2))
      po = int(oia(l_ + 1)) + 1; po_end = int(oia(l_ + 2))
      pw = 1; pw_end = 0; pv = 1; pv_end = 0
      if (op%has_w) then
        pw = int(wia(l_ + 1)) + 1; pw_end = int(wia(l_ + 2))
        pv = int(woia(l_ + 1)) + 1; pv_end = int(woia(l_ + 2))
      endif
      do
        jd = huge(1); jo = huge(1); jwd = huge(1); jwo = huge(1)
        if (pa <= pa_end) jd = me0 + int(aja(pa))
        if (po <= po_end) jo = int(ga(oja(po) + 1))
        if (pw <= pw_end) jwd = me0 + int(wja(pw))
        if (pv <= pv_end) jwo = int(gw(woja(pv) + 1))
        jmin = min(jd, jo, jwd, jwo)
        if (jmin == huge(1)) exit
        nu = nu + 1
        if (nu > size(uj)) then
          uj = [uj, 0 * uj]; ka = [ka, 0 * ka]; kw = [kw, 0 * kw]
        endif
        uj(nu) = jmin; ka(nu) = 0; kw(nu) = 0
        if (jd == jmin) then
          ka(nu) = pa; pa = pa + 1
        else if (jo == jmin) then
          ka(nu) = -po; po = po + 1
        endif
        if (jwd == jmin) then
          kw(nu) = pw; pw = pw + 1
        else if (jwo == jmin) then
          kw(nu) = -pv; pv = pv + 1
        endif
      enddo
    end subroutine block_cols

  end subroutine sfo_build

  !> Position of scalar (rw, cw) of the column-major bs x bs block +-k, signed.
  integer(c_int32_t) function apos(k, rw, cw, bs)
    integer, intent(in) :: k, rw, cw, bs
    apos = int((int(abs(k), 8) - 1) * bs * bs + cw * bs + rw + 1, c_int32_t)
    if (k < 0) apos = -apos
  end function apos

  !> Refill op from A and W. Collective on the global communicator.
  subroutine sfo_fill(op, A, W, opz)
    type(sf_op_t), intent(inout) :: op
    Mat, intent(in)    :: A, W(:)
    real*8, intent(in) :: opz
    type :: wvals_t
      Mat :: d, o
      type(c_ptr) :: pd, po
      real(c_double), pointer :: vd(:) => null(), vo(:) => null()
      logical :: on = .false.
    end type wvals_t
    Mat :: Ad, Ao
    PetscInt, pointer :: ga(:), gw(:)
    type(c_ptr) :: pad, pao
    real(c_double), pointer :: va_d(:), va_o(:)
    type(wvals_t), allocatable :: wv(:)
    real(c_double), allocatable :: sbuf(:)
    integer(c_int) :: rc
    integer :: k, nreq, mpierr, iw
    integer, allocatable :: reqs(:)
    real*8 :: x
    PetscErrorCode :: ierr

    call split_parts(A, .true., Ad, Ao, ga)
    rc = c_baij_get(transfer(Ad%v, 0_c_intptr_t), pad); call chk(rc)
    rc = c_baij_get(transfer(Ao%v, 0_c_intptr_t), pao); call chk(rc)
    call c_f_pointer(pad, va_d, [huge(1)])
    call c_f_pointer(pao, va_o, [huge(1)])
    allocate(wv(size(W)))
    if (op%has_w) then
      do iw = 1, size(W)
        if (.not. any(op%wid(1:op%nf, 1:op%nf) == iw)) cycle
        wv(iw)%on = .true.
        call split_parts(W(iw), .true., wv(iw)%d, wv(iw)%o, gw)
        rc = c_baij_get(transfer(wv(iw)%d%v, 0_c_intptr_t), wv(iw)%pd); call chk(rc)
        rc = c_baij_get(transfer(wv(iw)%o%v, 0_c_intptr_t), wv(iw)%po); call chk(rc)
        call c_f_pointer(wv(iw)%pd, wv(iw)%vd, [huge(1)])
        call c_f_pointer(wv(iw)%po, wv(iw)%vo, [huge(1)])
      enddo
    endif

    allocate(sbuf(max(size(op%sa), 1)))
    !$omp parallel do private(x) schedule(static)
    do k = 1, size(op%sa)
      x = 0.d0
      if (op%sa(k) > 0) then
        x = va_d(op%sa(k))
      else if (op%sa(k) < 0) then
        x = va_o(-op%sa(k))
      endif
      if (allocated(op%sc)) then
        if (op%sc(k) == 1) x = x * opz
      endif
      if (op%has_w) then
        if (op%sw(k) > 0) then
          x = x + wv(op%ss(k))%vd(op%sw(k))
        else if (op%sw(k) < 0) then
          x = x + wv(op%ss(k))%vo(-op%sw(k))
        endif
      endif
      sbuf(k) = x
    enddo
    !$omp end parallel do

    rc = c_baij_restore(transfer(Ao%v, 0_c_intptr_t), pao); call chk(rc)
    rc = c_baij_restore(transfer(Ad%v, 0_c_intptr_t), pad); call chk(rc)
    do iw = 1, size(W)
      if (.not. wv(iw)%on) cycle
      rc = c_baij_restore(transfer(wv(iw)%o%v, 0_c_intptr_t), wv(iw)%po); call chk(rc)
      rc = c_baij_restore(transfer(wv(iw)%d%v, 0_c_intptr_t), wv(iw)%pd); call chk(rc)
    enddo

    allocate(reqs(op%nrecv + op%nsend))
    nreq = 0
    do k = 1, op%nrecv
      nreq = nreq + 1
      call MPI_Irecv(op%val(op%roff(k) + 1), op%rcnt(k), MPI_DOUBLE_PRECISION, op%rsrc(k), op%rtag(k), &
                     gcomm, reqs(nreq), mpierr)
    enddo
    do k = 1, op%nsend
      nreq = nreq + 1
      call MPI_Isend(sbuf(op%soff(k) + 1), op%scnt(k), MPI_DOUBLE_PRECISION, op%sdst(k), op%stag(k), &
                     gcomm, reqs(nreq), mpierr)
    enddo
    call MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE, mpierr)
    ! in place: same pattern and arrays, a new object state for MUMPS and the GMG
    call MatUpdateMPIAIJWithArray(op%mat, op%val, ierr)

  contains

    subroutine chk(code)
      integer(c_int), intent(in) :: code
      if (code /= 0) call fail("access to a BAIJ value array failed")
    end subroutine chk

  end subroutine sfo_fill

  subroutine fail(msg)
    character(len=*), intent(in) :: msg
    integer :: mpierr
    write(*,'(A,A,A,I0)') "[Physics PC]   FATAL: SF operators: ", msg, " on rank ", gme
    call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
  end subroutine fail

  !> v(k) = variable k of the full-system vector x in the operators' layout,
  !! k = 1 .. size(v). Rank-local on the global layout.
  subroutine sfo_vars_in(x, v)
    use mod_parameters, only: n_var, n_tor
    Vec :: x
    Vec :: v(:)
    PetscScalar, pointer :: xa(:), va(:)
    PetscErrorCode :: ierr
    PetscInt :: nl
    integer :: k, i, bs

    if (sfo_split) then
      call vec_setup(x, size(v))
      call VecScatterBegin(scv, x, stagev, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd(scv, x, stagev, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecGetArrayRead(stagev, xa, ierr)
      do k = 1, nsv
        call VecGetArray(v(k), va, ierr)
        va = xa((k - 1) * sfo_n1 + 1 : k * sfo_n1)
        call VecRestoreArray(v(k), va, ierr)
      enddo
      call VecRestoreArrayRead(stagev, xa, ierr)
      return
    endif
    call VecGetLocalSize(x, nl, ierr)
    bs = n_var * n_tor
    call VecGetArrayRead(x, xa, ierr)
    do k = 1, size(v)
      call VecGetArray(v(k), va, ierr)
      do i = 0, int(nl) / bs - 1
        va(i * n_tor + 1 : (i + 1) * n_tor) = xa(i * bs + (k - 1) * n_tor + 1 : i * bs + k * n_tor)
      enddo
      call VecRestoreArray(v(k), va, ierr)
    enddo
    call VecRestoreArrayRead(x, xa, ierr)
  end subroutine sfo_vars_in

  !> Variables 1 .. size(v) of y from v(k), the inverse of sfo_vars_in; any
  !! further variables of y are left untouched.
  subroutine sfo_vars_out(v, y)
    use mod_parameters, only: n_var, n_tor
    Vec :: v(:)
    Vec :: y
    PetscScalar, pointer :: ya(:), va(:)
    PetscErrorCode :: ierr
    PetscInt :: nl
    integer :: k, i, bs

    if (sfo_split) then
      call VecGetArray(stagev, ya, ierr)
      do k = 1, nsv
        call VecGetArrayRead(v(k), va, ierr)
        ya((k - 1) * sfo_n1 + 1 : k * sfo_n1) = va
        call VecRestoreArrayRead(v(k), va, ierr)
      enddo
      call VecRestoreArray(stagev, ya, ierr)
      call VecScatterBegin(scv, stagev, y, INSERT_VALUES, SCATTER_REVERSE, ierr)
      call VecScatterEnd(scv, stagev, y, INSERT_VALUES, SCATTER_REVERSE, ierr)
      return
    endif
    call VecGetLocalSize(y, nl, ierr)
    bs = n_var * n_tor
    call VecGetArray(y, ya, ierr)
    do k = 1, size(v)
      call VecGetArrayRead(v(k), va, ierr)
      do i = 0, int(nl) / bs - 1
        ya(i * bs + (k - 1) * n_tor + 1 : i * bs + k * n_tor) = va(i * n_tor + 1 : (i + 1) * n_tor)
      enddo
      call VecRestoreArrayRead(v(k), va, ierr)
    enddo
    call VecRestoreArray(y, ya, ierr)
  end subroutine sfo_vars_out

  !> Mode split: the scatter between the full-system vector and the family
  !! layout of variables 1 .. nvars, once.
  subroutine vec_setup(x, nvars)
    use mod_parameters, only: n_tor, n_var
    Vec, intent(in) :: x
    integer, intent(in) :: nvars
    PetscInt, allocatable :: idx(:)
    PetscInt :: k
    IS :: isv
    integer :: v, r, i, s
    PetscErrorCode :: ierr

    if (vec_ok) then
      if (nvars /= nsv) call fail("vector variables differ from the first call")
      return
    endif
    nsv = nvars
    allocate(idx(nsv * sfo_n1))
    k = 0
    do v = 1, nsv
      do i = i0, i1 - 1
        r = sowner(i)
        do s = 1, size(myslots)
          k = k + 1
          idx(k) = rsx(r) + int(i - idx0(r), kind(k)) * (n_var * n_tor) + (v - 1) * n_tor + myslots(s)
        enddo
      enddo
    enddo
    call ISCreateGeneral(gcomm, k, idx, PETSC_COPY_VALUES, isv, ierr)
    call VecCreateMPI(gcomm, k, PETSC_DETERMINE, stagev, ierr)
    call VecScatterCreate(x, isv, stagev, PETSC_NULL_IS, scv, ierr)
    call ISDestroy(isv, ierr)
    vec_ok = .true.
  end subroutine vec_setup

#endif
end module mod_petsc_pc_sf_ops
