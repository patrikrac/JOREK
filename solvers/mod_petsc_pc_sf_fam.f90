module mod_petsc_pc_sf_fam
#ifdef USE_PETSC
  use mpi_mod
  use iso_c_binding
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_raw_csr, only: split_parts, get_ij, put_ij, c_baij_get, c_baij_restore
  implicit none
  private

  !--------------------------------------------------------------------
  !> The SF path on one |n| family per rank (physics_pc_sf_mode_split).
  !!
  !! With physics_pc_sf_harm_couple = .f. every SF block is |n|-diagonal, so the
  !! whole SF sweep falls apart into independent problems, one per |n|
  !! family: the n = 0 slot, then the cos and sin slots of each n. The global
  !! path still solves them together, every block on every rank, one block
  !! after another -- the regime in which the GMG's coarse levels, the
  !! scalar rho / T solves and the inner Krylov reductions stop scaling.
  !!
  !! Here the ranks are split into one sub-communicator per family (as
  !! JOREK's legacy solver and its PETSc mode-split PC do), and each rank
  !! runs the SF solvers on its own family only, concurrently with the other
  !! families: every reduction and halo exchange spans the family's ranks.
  !!
  !! Ranks per family. A family's work scales with its slots (1 for n = 0, 2
  !! for every other n), so the ranks are split in proportion to them
  !! (largest remainder, at least one each; -sf_ms_w0 sets the n = 0 weight,
  !! 2 gives the even split). Within a family, each rank owns a contiguous,
  !! equal share of the JOREK indices.
  !!
  !! Family layout. The family operators are packed as the GMG expects: row =
  !! (field, index, local slot), the local slot fastest, field-major per rank.
  !! Their slots are the family's global slots (mod_petsc_pc_harm).
  !!
  !! DIRECT EXTRACTION. The family operators are built straight from JOREK's
  !! BAIJ Jacobian A (and the force operator W): no global block, packed pair
  !! or copy of them exists. The rank that owns A's rows of an index is the
  !! SOURCE of every family operator row of that index. On the first build it
  !! works out each such row's columns (family numbering, sorted) and, for
  !! every entry, where its value sits in its own A / W value arrays and with
  !! which coefficient (1, or opz for wpj's opz M_psi block); it sends the row
  !! structure to the row's family owner, which creates the operator. Every
  !! build then costs one message per (source, owner, field): the source
  !! computes the final values (A, opz A, or the B_22 + W sum) and sends them
  !! straight into the slice of the owner's value array they belong to.
  !! Entries outside the family (W's cross-|n| part; A's are already
  !! filtered out) are dropped and counted.
  !--------------------------------------------------------------------

  !> A family operator extracted directly. Block (a, b) of its nf x nf fields
  !! is A(se(a,b), sv(a,b)) (variables; 0 = none), times opz where scl = 1,
  !! plus W source wid(a,b) of the Wf(:) the build / fill get (0 = none;
  !! pair_w's B_22 + W: wid(1,1) = W's index). All W sources must share one
  !! block pattern (they come from one element pass); the build checks it.
  integer, parameter, public :: SFF_NFMAX = 5          !< fields per operator (pair_w with v_par)
  type, public :: fam_op_t
    character(len=24) :: label = ""
    integer :: nf = 0, tagb = 0
    integer :: se(SFF_NFMAX, SFF_NFMAX) = 0, sv(SFF_NFMAX, SFF_NFMAX) = 0, scl(SFF_NFMAX, SFF_NFMAX) = 0
    integer :: wid(SFF_NFMAX, SFF_NFMAX) = 0
    logical :: has_w = .false.                       !< any wid > 0 (set by the build)
    logical :: ready = .false.
    Mat     :: fam                                   !< the operator, family comm
    real(c_double), allocatable :: val(:)            !< its values, CSR order
    !--- owner side: one message per (source rank, field), into val
    integer :: nrecv = 0
    integer, allocatable :: rsrc(:), rtag(:), roff(:), rcnt(:)
    !--- source side: one message per (owner rank, field), values in send order
    integer :: nsend = 0
    integer, allocatable :: sdst(:), stag(:), soff(:), scnt(:)
    integer(c_int32_t), allocatable :: sa(:), sw(:)  !< +k / -k: A's (W's) diagonal / off-diagonal part
    integer(c_int8_t), allocatable :: ss(:)          !< the W source of sw (wid), 0 = none
    integer(c_int8_t), allocatable :: sc(:)          !< 1: times opz
  end type fam_op_t

  logical, save, public :: sff_on = .false.   !< decomposed and active
  integer, save, public :: sff_comm = MPI_COMM_NULL   !< this rank's family communicator
  integer, save, public :: sff_fam  = 0       !< this rank's family (1-based)
  integer, save, public :: sff_nfam = 0
  PetscInt, save, public :: sff_n1 = 0        !< family per-variable local rows

  integer, save :: gcomm = MPI_COMM_NULL, gme = -1, gnp = 0
  integer, save :: fme = -1, fnp = 0          !< rank / size in the family
  integer, allocatable, save :: fmodes(:)     !< this family's global slots (0-based)
  integer, allocatable, save :: fr0(:)        !< first global rank of each family; fr0(nfam+1) = gnp
  integer, allocatable, save :: idx0(:)       !< first JOREK index of each global rank; idx0(gnp) = n_idx
  PetscInt, allocatable, save :: rsx(:)       !< first full-system row of each global rank
  integer, save :: n_idx = 0                  !< JOREK indices (A's block rows)
  integer, save :: i0 = 0, i1 = 0             !< my family indices [i0, i1)
  VecScatter, save :: scv                     !< x <-> stagev
  Vec, save :: stagev                         !< global comm, local nsv n1: variables in turn
  integer, save :: nsv = 0                    !< the variables the sweep splits (sff_vec_setup)
  logical, save :: vec_ok = .false.

  integer, parameter :: TAG_LEN = 0, TAG_COL = 8, TAG_VAL = 16  !< + field - 1 (< 8), + TAG_STRIDE tagb
  integer, parameter :: TAG_STRIDE = 32

  public :: sff_decompose, sff_op_build, sff_op_fill, &
            sff_vec_setup, sff_scatter_in, sff_scatter_out

contains

  !--------------------------------------------------------------------
  !> Split the ranks into |n| families and set the PC's slot layout. Once.
  !! A_full: the full-system Jacobian (BAIJ, bs = n_var n_tor), whose row
  !! layout fixes where every index's rows come from.
  !--------------------------------------------------------------------
  subroutine sff_decompose(A_full, comm, my_id)
    use mod_parameters,    only: n_tor, n_var
    use mod_petsc_pc_harm, only: pc_harm_set
    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id

    integer :: f, mpierr, r
    integer, allocatable :: nrk(:), nl(:)
    real*8, allocatable :: wf(:), ideal(:)
    real*8 :: w0
    PetscInt :: rs, re
    PetscReal :: rw
    PetscBool :: set
    PetscErrorCode :: ierr

    if (sff_on) return
    gcomm = comm
    call MPI_Comm_rank(comm, gme, mpierr)
    call MPI_Comm_size(comm, gnp, mpierr)
    sff_nfam = (n_tor + 1) / 2
    if (gnp < sff_nfam) then
      if (my_id == 0) write(*,'(A,I0,A,I0,A)') "[Physics PC]   FATAL: physics_pc_sf_mode_split needs at "// &
        "least one rank per |n| family: ", sff_nfam, " families, ", gnp, " ranks"
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    endif

    !--- ranks per family in proportion to its slots (n = 0: weight w0)
    w0 = 1.d0
    call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_ms_w0", rw, set, ierr)
    if (set) w0 = rw
    allocate(nrk(sff_nfam), wf(sff_nfam), ideal(sff_nfam))
    wf = 2.d0; wf(1) = w0
    ideal = gnp * wf / sum(wf)
    nrk = max(1, int(ideal))
    do while (sum(nrk) < gnp)                     ! largest remainder first
      f = maxloc(ideal - nrk, 1)
      nrk(f) = nrk(f) + 1
    enddo
    do while (sum(nrk) > gnp)                     ! the at-least-one floor overshot
      f = maxloc(merge(dble(nrk) - ideal, -huge(1.d0), nrk > 1), 1)
      nrk(f) = nrk(f) - 1
    enddo
    allocate(fr0(sff_nfam + 1))
    fr0(1) = 0
    do f = 1, sff_nfam
      fr0(f + 1) = fr0(f) + nrk(f)
    enddo
    do f = 1, sff_nfam
      if (gme >= fr0(f) .and. gme < fr0(f + 1)) sff_fam = f
    enddo
    call MPI_Comm_split(comm, sff_fam, gme, sff_comm, mpierr)
    call MPI_Comm_rank(sff_comm, fme, mpierr)
    call MPI_Comm_size(sff_comm, fnp, mpierr)

    !--- the family's slots: 0 | 1, 2 | 3, 4 | ...
    if (sff_fam == 1) then
      fmodes = [0]
    else
      fmodes = [2 * sff_fam - 3, 2 * sff_fam - 2]
    endif
    call pc_harm_set(fmodes)

    !--- where every index's rows are: A's block rows per global rank
    allocate(nl(0:gnp - 1), idx0(0:gnp), rsx(0:gnp - 1))
    call MatGetOwnershipRange(A_full, rs, re, ierr)
    call MPI_Allgather(int((re - rs) / (n_var * n_tor)), 1, MPI_INTEGER, nl, 1, MPI_INTEGER, comm, mpierr)
    call MPI_Allgather(rs, 1, MPIU_INTEGER, rsx, 1, MPIU_INTEGER, comm, mpierr)
    idx0(0) = 0
    do r = 0, gnp - 1
      idx0(r + 1) = idx0(r) + nl(r)
    enddo
    n_idx = idx0(gnp)

    !--- my family indices: an equal share
    i0 = frange(sff_fam, fme)
    i1 = frange(sff_fam, fme + 1)
    sff_n1 = int(i1 - i0, kind(sff_n1)) * size(fmodes)

    sff_on = .true.
    call report_layout(my_id, nrk, w0)
  end subroutine sff_decompose

  !> First index of family f's rank q (q = its rank count: n_idx).
  integer function frange(f, q)
    integer, intent(in) :: f, q
    frange = int(int(q, 8) * n_idx / (fr0(f + 1) - fr0(f)))
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
    integer, intent(in) :: f
    integer, allocatable :: s(:)
    if (f == 1) then
      s = [0]
    else
      s = [2 * f - 3, 2 * f - 2]
    endif
  end function fslots

  subroutine report_layout(my_id, nrk, w0)
    integer, intent(in) :: my_id, nrk(:)
    real*8, intent(in)  :: w0
    integer :: f, nmin, nmax, q
    if (my_id /= 0) return
    write(*,'(A,I0,A,I0,A,F4.2,A)') "[Physics PC]   SF mode split: ", sff_nfam, " |n| families on ", gnp, &
      " ranks (n = 0 weight ", w0, ", others 2)"
    do f = 1, sff_nfam
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

  !--------------------------------------------------------------------
  !> First build of a directly extracted family operator: the row structure
  !! from the source ranks, the operator on the owner, the send / receive
  !! plans, and the first values. Collective on the global communicator.
  !--------------------------------------------------------------------
  subroutine sff_op_build(op, Aj, Wf, opz, my_id)
    use mod_parameters, only: n_tor, n_var
    type(fam_op_t), intent(inout) :: op
    Mat, intent(in)     :: Aj, Wf(:)
    real*8, intent(in)  :: opz
    integer, intent(in) :: my_id

    !--- source side
    Mat :: Ad, Ao, Wd, Wo
    PetscInt, pointer :: ga(:), gw(:), aia(:), aja(:), oia(:), oja(:)
    PetscInt, pointer :: wia(:), wja(:), woia(:), woja(:)
    PetscInt :: nad, nao, nwd, nwo, bsx
    integer :: bsa, nloc_i, me0, pass, f, q, qa, qb, a, b, l, i, ia_, ib_, s, sq, m, qslot, k, u
    integer :: nmsg, ns_f, ent, row, e, v, nq, jq, qj, nu
    integer :: ns_g
    integer(8) :: na_d, na_o, nw_d, nw_o, ndrop, nd_glob
    integer, allocatable :: sl(:)
    integer :: iw0, iw
    integer, allocatable :: uj(:), ka(:), kw(:)          ! one index's block columns: A / W block (+-k, 0)
    integer, allocatable :: srow(:), scol(:), smrow(:), smcol(:)
    integer :: nrow_tot
    !--- owner side
    integer :: nf, nq_me, ns_me, nloc, r, ra, rb, nreq, mpierr
    PetscInt, allocatable :: fi(:), fj(:)
    integer, allocatable :: rlen(:), rrow0(:), rnrow(:), reqs(:)
    PetscErrorCode :: ierr

    nf = op%nf
    if (nf > SFF_NFMAX) call fail("more fields than SFF_NFMAX")
    if (any(op%wid(1:nf, 1:nf) < 0) .or. any(op%wid(1:nf, 1:nf) > size(Wf))) &
      call fail("a W source index outside Wf")
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
    ! W's block pattern from the first source this operator reads; every
    ! other source it reads must have the same (one element pass)
    iw0 = 0
    if (op%has_w) then
      iw0 = minval(op%wid(1:nf, 1:nf), mask=op%wid(1:nf, 1:nf) > 0)
      call MatGetBlockSize(Wf(iw0), bsx, ierr)
      if (bsx /= n_tor) call fail("W's block size is not n_tor")
      call split_parts(Wf(iw0), .true., Wd, Wo, gw)
      call get_ij(Wd, .true., nwd, wia, wja)
      call get_ij(Wo, .true., nwo, woia, woja)
      if (nwd /= nad) call fail("W's block rows are not A's")
      nw_d = int(wia(nwd + 1), 8) * n_tor * n_tor
      nw_o = int(woia(nwo + 1), 8) * n_tor * n_tor
      do iw = 1, size(Wf)
        if (iw == iw0 .or. .not. any(op%wid(1:nf, 1:nf) == iw)) cycle
        call same_pattern(Wf(iw))
      enddo
    endif
    if (max(na_d, na_o, nw_d, nw_o) >= int(huge(0_c_int32_t), 8)) &
      call fail("a BAIJ value array exceeds the 32-bit position index")
    nloc_i = int(nad)
    me0 = idx0(gme)
    if (nloc_i /= idx0(gme + 1) - me0) call fail("A's row layout changed since the decomposition")

    !--- source side: two passes (count, then fill) over my rows of every
    !--- family. Messages go out per (family owner, field), rows ordered
    !--- (index, slot), entries in the owner's column order.
    allocate(uj(64), ka(64), kw(64))
    ndrop = 0
    do pass = 1, 2
      nmsg = 0; ent = 0; nrow_tot = 0
      do f = 1, sff_nfam
        sl = fslots(f); ns_f = size(sl)
        if (nloc_i == 0) cycle
        qa = fowner(f, me0); qb = fowner(f, me0 + nloc_i - 1)
        do q = qa, qb
          ia_ = max(me0, frange(f, q)); ib_ = min(me0 + nloc_i, frange(f, q + 1))
          if (ib_ <= ia_) cycle
          do a = 1, nf
            nmsg = nmsg + 1
            if (pass == 2) then
              op%sdst(nmsg) = fr0(f) + q
              op%stag(nmsg) = TAG_STRIDE * op%tagb + TAG_VAL + a - 1
              op%soff(nmsg) = ent
              smrow(nmsg) = nrow_tot
              smcol(nmsg) = ent
            endif
            do i = ia_, ib_ - 1
              l = i - me0
              call block_cols(l)
              do s = 1, ns_f
                m = sl(s)
                nrow_tot = nrow_tot + 1
                row = 0
                ! entries: owner groups of the block columns in ascending
                ! order; inside a group b, then index, then slot -- the
                ! owner's column order
                ns_g = ns_f
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
                      if (.not. ((e > 0 .and. ka(jq) /= 0) .or. &
                                 (op%wid(a, b) > 0 .and. kw(jq) /= 0))) cycle
                      do sq = 1, ns_g
                        qslot = sl(sq)
                        row = row + 1
                        ent = ent + 1
                        if (pass == 2) then
                          scol(ent) = nf * frange(f, qj) * ns_g + (b - 1) * nq * ns_g &
                                      + (uj(jq) - frange(f, qj)) * ns_g + sq - 1
                          op%sa(ent) = 0
                          if (e > 0 .and. ka(jq) /= 0) &
                            op%sa(ent) = apos(ka(jq), (e - 1) * n_tor + m, (v - 1) * n_tor + qslot, bsa)
                          if (op%has_w) then
                            op%sw(ent) = 0; op%ss(ent) = 0
                            if (op%wid(a, b) > 0 .and. kw(jq) /= 0) then
                              op%sw(ent) = apos(kw(jq), m, qslot, n_tor)
                              op%ss(ent) = int(op%wid(a, b), c_int8_t)
                            endif
                          endif
                          if (allocated(op%sc)) op%sc(ent) = int(op%scl(a, b), c_int8_t)
                        endif
                      enddo
                    enddo
                  enddo
                  u = k + 1
                enddo
                if (pass == 2) srow(nrow_tot) = row
                ! W's entries of this row outside the family
                if (pass == 1 .and. any(op%wid(a, 1:nf) > 0)) then
                  do jq = 1, nu
                    if (kw(jq) /= 0) ndrop = ndrop + (n_tor - ns_f)
                  enddo
                endif
              enddo
            enddo
            if (pass == 2) op%scnt(nmsg) = ent - op%soff(nmsg)
          enddo
        enddo
      enddo
      if (pass == 1) then
        op%nsend = nmsg
        allocate(op%sdst(nmsg), op%stag(nmsg), op%soff(nmsg), op%scnt(nmsg), smrow(nmsg + 1), smcol(nmsg + 1))
        allocate(op%sa(ent), srow(nrow_tot), scol(ent))
        if (op%has_w) allocate(op%sw(ent), op%ss(ent))
        if (any(op%scl(1:nf, 1:nf) /= 0)) allocate(op%sc(ent))
      else
        smrow(nmsg + 1) = nrow_tot; smcol(nmsg + 1) = ent
      endif
    enddo
    deallocate(uj, ka, kw)
    call put_ij(Ad, .true., nad, aia, aja); call put_ij(Ao, .true., nao, oia, oja)
    if (op%has_w) then
      call put_ij(Wd, .true., nwd, wia, wja); call put_ij(Wo, .true., nwo, woia, woja)
    endif
    if (op%has_w) then
      call MPI_Allreduce(ndrop, nd_glob, 1, MPI_INTEGER8, MPI_SUM, gcomm, mpierr)
      if (my_id == 0 .and. nd_glob > 0) write(*,'(A,A,A,I0,A)') "[Physics PC]   SF mode split: ", &
        trim(op%label), ": ", nd_glob, " cross-|n| W entries dropped"
    endif

    !--- owner side: my rows of every field come from the source ranks of
    !--- my indices; one message per (source, field)
    ns_me = size(fmodes)
    nq_me = i1 - i0
    nloc = nf * nq_me * ns_me
    if (nq_me > 0) then
      ra = sowner(i0); rb = sowner(i1 - 1)
    else
      ra = 0; rb = -1
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
        ia_ = max(i0, idx0(r)); ib_ = min(i1, idx0(r + 1))
        if (ib_ <= ia_) cycle
        k = k + 1
        op%rsrc(k) = r
        op%rtag(k) = TAG_STRIDE * op%tagb + TAG_VAL + a - 1
        rrow0(k) = (a - 1) * nq_me * ns_me + (ia_ - i0) * ns_me
        rnrow(k) = (ib_ - ia_) * ns_me
      enddo
    enddo

    !--- round 1: row lengths
    nreq = 0
    allocate(reqs(op%nrecv + op%nsend))
    do k = 1, op%nrecv
      nreq = nreq + 1
      call MPI_Irecv(rlen(rrow0(k) + 1), rnrow(k), MPI_INTEGER, op%rsrc(k), &
                     op%rtag(k) - TAG_VAL + TAG_LEN, gcomm, reqs(nreq), mpierr)
    enddo
    do k = 1, op%nsend
      nreq = nreq + 1
      call MPI_Isend(srow(smrow(k) + 1), smrow(k + 1) - smrow(k), MPI_INTEGER, op%sdst(k), &
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

    !--- round 2: the columns
    allocate(fj(max(int(fi(nloc + 1)), 1)))
    nreq = 0
    do k = 1, op%nrecv
      nreq = nreq + 1
      call MPI_Irecv(fj(op%roff(k) + 1), op%rcnt(k), MPIU_INTEGER, op%rsrc(k), &
                     op%rtag(k) - TAG_VAL + TAG_COL, gcomm, reqs(nreq), mpierr)
    enddo
    block
      PetscInt, allocatable :: scol_p(:)
      allocate(scol_p(max(size(scol), 1)))
      if (size(scol) > 0) scol_p(1:size(scol)) = int(scol, kind(scol_p))
      do k = 1, op%nsend
        nreq = nreq + 1
        call MPI_Isend(scol_p(smcol(k) + 1), smcol(k + 1) - smcol(k), MPIU_INTEGER, op%sdst(k), &
                       op%stag(k) - TAG_VAL + TAG_COL, gcomm, reqs(nreq), mpierr)
      enddo
      call MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE, mpierr)
      deallocate(scol_p)
    end block
    deallocate(srow, scol, smrow, smcol, rlen, rrow0, rnrow, reqs)

    !--- the operator, on the family communicator
    allocate(op%val(max(int(fi(nloc + 1)), 1)))
    op%val = 0.d0
    call MatCreateMPIAIJWithArrays(sff_comm, int(nloc, kind(fi)), int(nloc, kind(fi)), PETSC_DETERMINE, &
                                   PETSC_DETERMINE, fi, fj, op%val, op%fam, ierr)
    deallocate(fi, fj)
    op%ready = .true.
    call sff_op_fill(op, Aj, Wf, opz)

  contains

    !> Fail unless W source Wm has the block pattern of Wf(iw0) read above.
    subroutine same_pattern(Wm)
      Mat, intent(in) :: Wm
      Mat :: Xd, Xo
      PetscInt, pointer :: gx(:), xia(:), xja(:), xoia(:), xoja(:)
      PetscInt :: nxd, nxo
      logical :: same
      call split_parts(Wm, .true., Xd, Xo, gx)
      call get_ij(Xd, .true., nxd, xia, xja)
      call get_ij(Xo, .true., nxo, xoia, xoja)
      same = (nxd == nwd .and. nxo == nwo)
      if (same) same = all(xia == wia) .and. all(xoia == woia)
      if (same) same = all(xja(1:xia(nxd + 1)) == wja(1:wia(nwd + 1)))
      if (same) same = all(xoja(1:xoia(nxo + 1)) == woja(1:woia(nwo + 1)))
      if (same .and. size(gx) == size(gw)) then
        same = all(gx == gw)
      else
        same = .false.
      endif
      call put_ij(Xd, .true., nxd, xia, xja); call put_ij(Xo, .true., nxo, xoia, xoja)
      if (.not. same) call fail("the W sources of one operator have different block patterns")
    end subroutine same_pattern

    !> Block columns of local block row l of A and of W, merged, ascending:
    !! uj(1:nu), with the A / W block (+k diagonal part, -k off-diagonal, 0).
    subroutine block_cols(l_)
      integer, intent(in) :: l_
      integer :: pa, pa_end, po, po_end, pw, pw_end, pv, pv_end, jmin, jn
      integer :: jd, jo, jwd, jwo, kk
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
          jn = 2 * size(uj)
          uj = [uj, (0, kk = 1, jn - size(uj))]
          ka = [ka, (0, kk = 1, jn - size(ka))]
          kw = [kw, (0, kk = 1, jn - size(kw))]
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

  end subroutine sff_op_build

  !> Position of scalar (rw, cw) of block +-k (column-major, size bs), signed.
  integer(c_int32_t) function apos(k, rw, cw, bs)
    integer, intent(in) :: k, rw, cw, bs
    apos = int((int(abs(k), 8) - 1) * bs * bs + cw * bs + rw + 1, c_int32_t)
    if (k < 0) apos = -apos
  end function apos

  !--------------------------------------------------------------------
  !> Refill a family operator from A and W (every build). Collective on the
  !! global communicator.
  !--------------------------------------------------------------------
  subroutine sff_op_fill(op, Aj, Wf, opz)
    type(fam_op_t), intent(inout) :: op
    Mat, intent(in)    :: Aj, Wf(:)
    real*8, intent(in) :: opz
    !> one W source's value arrays
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
    integer :: k, nreq, mpierr, n, iw
    integer, allocatable :: reqs(:)
    real*8 :: x
    PetscErrorCode :: ierr

    n = size(op%sa)
    call split_parts(Aj, .true., Ad, Ao, ga)
    rc = c_baij_get(transfer(Ad%v, 0_c_intptr_t), pad); call c_f_pointer(pad, va_d, [huge(1)])
    rc = c_baij_get(transfer(Ao%v, 0_c_intptr_t), pao); call c_f_pointer(pao, va_o, [huge(1)])
    allocate(wv(size(Wf)))
    if (op%has_w) then
      do iw = 1, size(Wf)
        if (.not. any(op%wid(1:op%nf, 1:op%nf) == iw)) cycle
        wv(iw)%on = .true.
        call split_parts(Wf(iw), .true., wv(iw)%d, wv(iw)%o, gw)
        rc = c_baij_get(transfer(wv(iw)%d%v, 0_c_intptr_t), wv(iw)%pd)
        call c_f_pointer(wv(iw)%pd, wv(iw)%vd, [huge(1)])
        rc = c_baij_get(transfer(wv(iw)%o%v, 0_c_intptr_t), wv(iw)%po)
        call c_f_pointer(wv(iw)%po, wv(iw)%vo, [huge(1)])
      enddo
    endif

    allocate(sbuf(max(n, 1)))
    !$omp parallel do private(x) schedule(static)
    do k = 1, n
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

    rc = c_baij_restore(transfer(Ao%v, 0_c_intptr_t), pao)
    rc = c_baij_restore(transfer(Ad%v, 0_c_intptr_t), pad)
    do iw = 1, size(Wf)
      if (.not. wv(iw)%on) cycle
      rc = c_baij_restore(transfer(wv(iw)%o%v, 0_c_intptr_t), wv(iw)%po)
      rc = c_baij_restore(transfer(wv(iw)%d%v, 0_c_intptr_t), wv(iw)%pd)
    enddo
    deallocate(wv)

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
    deallocate(sbuf, reqs)
    ! new values in place: same pattern, same arrays (the block kernel and
    ! the GMG keep their handles); the object state tells MUMPS and the GMG
    call MatUpdateMPIAIJWithArray(op%fam, op%val, ierr)
  end subroutine sff_op_fill

  subroutine fail(msg)
    character(len=*), intent(in) :: msg
    integer :: mpierr
    write(*,'(A,A,A,I0)') "[Physics PC]   FATAL: SF mode split: ", msg, " on rank ", gme
    call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
  end subroutine fail

  !====================================================================
  ! vectors
  !====================================================================

  !--------------------------------------------------------------------
  !> The scatter between the full-system vector and the family layout of
  !! variables 1..nvars (the sweep's: 6, or 7 with v_par). x_tmpl: a vector of
  !! the full system's layout. Once.
  !--------------------------------------------------------------------
  subroutine sff_vec_setup(x_tmpl, nvars)
    use mod_parameters, only: n_tor, n_var
    Vec, intent(in) :: x_tmpl
    integer, intent(in) :: nvars
    PetscInt, allocatable :: idx(:)
    PetscInt :: k
    IS :: isv
    integer :: v, r, i, s
    PetscErrorCode :: ierr

    if (vec_ok) return
    if (nvars < 1 .or. nvars > n_var) call fail("sff_vec_setup: nvars outside 1 .. n_var")
    nsv = nvars
    allocate(idx(nsv * sff_n1))
    k = 0
    do v = 1, nsv
      do i = i0, i1 - 1
        r = sowner(i)
        do s = 1, size(fmodes)
          k = k + 1
          ! create_variable_index_sets: node block i of rank r, variable v, slot m
          idx(k) = rsx(r) + int(i - idx0(r), kind(k)) * (n_var * n_tor) + (v - 1) * n_tor + fmodes(s)
        enddo
      enddo
    enddo
    call ISCreateGeneral(gcomm, k, idx, PETSC_COPY_VALUES, isv, ierr)
    call VecCreateMPI(gcomm, k, PETSC_DETERMINE, stagev, ierr)
    call VecScatterCreate(x_tmpl, isv, stagev, PETSC_NULL_IS, scv, ierr)
    call ISDestroy(isv, ierr)
    vec_ok = .true.
  end subroutine sff_vec_setup

  !> v(k) = variable k of x, in the family layout (v on the family comm),
  !! k = 1 .. nsv.
  subroutine sff_scatter_in(x, v)
    Vec :: x
    Vec :: v(:)
    PetscScalar, pointer :: sa(:), va(:)
    PetscErrorCode :: ierr
    integer :: k
    if (size(v) /= nsv) call fail("sff_scatter_in: not the variables of sff_vec_setup")
    call VecScatterBegin(scv, x, stagev, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(scv, x, stagev, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecGetArrayRead(stagev, sa, ierr)
    do k = 1, nsv
      call VecGetArray(v(k), va, ierr)
      va = sa((k - 1) * sff_n1 + 1 : k * sff_n1)
      call VecRestoreArray(v(k), va, ierr)
    enddo
    call VecRestoreArrayRead(stagev, sa, ierr)
  end subroutine sff_scatter_in

  !> y = the family-layout variables v(k), k = 1 .. nsv, back in the full
  !! system. Every row of y belongs to exactly one family, so variables
  !! 1 .. nsv of y are fully overwritten (any further ones are not touched).
  subroutine sff_scatter_out(v, y)
    Vec :: v(:)
    Vec :: y
    PetscScalar, pointer :: sa(:), va(:)
    PetscErrorCode :: ierr
    integer :: k
    if (size(v) /= nsv) call fail("sff_scatter_out: not the variables of sff_vec_setup")
    call VecGetArray(stagev, sa, ierr)
    do k = 1, nsv
      call VecGetArrayRead(v(k), va, ierr)
      sa((k - 1) * sff_n1 + 1 : k * sff_n1) = va
      call VecRestoreArrayRead(v(k), va, ierr)
    enddo
    call VecRestoreArray(stagev, sa, ierr)
    call VecScatterBegin(scv, stagev, y, INSERT_VALUES, SCATTER_REVERSE, ierr)
    call VecScatterEnd(scv, stagev, y, INSERT_VALUES, SCATTER_REVERSE, ierr)
  end subroutine sff_scatter_out

#endif
end module mod_petsc_pc_sf_fam
