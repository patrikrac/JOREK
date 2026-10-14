!> Fortran side of the device block smoother (jorek_blk_kokkos.cpp): the
!! factors of one GMG level's smoother blocks, copied to the device after each
!! host factorisation, and their application to PETSc Kokkos vectors.
!!
!! Built with USE_GPU_PC against a PETSc with Kokkos; without it
!! blk_dev_available() is .false. and the block smoothers stay on the host
!! (mod_petsc_pc_gmg's device path then still runs, with a host round trip per
!! smoothing step).
module mod_petsc_blk_dev
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
  use petsc
  use iso_c_binding
  implicit none
  private

  public :: blk_dev_available, blk_dev_pattern, blk_dev_spike, blk_dev_values, blk_dev_apply, blk_dev_free
  public :: ring_dev_pattern, ring_dev_values, ring_dev_factor, ring_dev_fillmap, ring_dev_apply
  public :: blk_dev_fillmap, blk_dev_fill
  public :: crs_dev_has, crs_dev_map, crs_dev_setup, crs_dev_apply, crs_dev_free
  public :: vec_copy_strided, vec_is_dev, vec_make_dev, vec_touch

#ifdef USE_GPU_PC
  interface
    ! (this declaration's form is what util/makedepend links the C++ file by)
    function jorek_blk_kokkos(h, nb, nrow, ngh, off, sz, kl, ku, band, loff, axblk, rows, bcol, nzp, zp, nz, zc, stats) bind(C)
      use, intrinsic :: iso_c_binding
      integer(c_int)            :: jorek_blk_kokkos
      type(c_ptr)               :: h
      integer(c_int), value     :: nb, nrow, ngh, nzp, nz
      integer(c_int)            :: off(*), sz(*), kl(*), ku(*), band(*), axblk(*), rows(*), zp(*), zc(*)
      integer(c_long_long)      :: loff(*)
      type(c_ptr), value        :: bcol
      real(c_double)            :: stats(3)
    end function jorek_blk_kokkos
    integer(c_int) function c_blk_spike(h, nlu, lu, stats) bind(C, name="jorek_blk_kokkos_spike")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: nlu
      real(c_double)              :: lu(*), stats(3)
    end function c_blk_spike
    integer(c_int) function c_blk_values(h, nlu, lu, npiv, piv, nz, zv, bad, draw, dbad, ndbad, stats) &
        bind(C, name="jorek_blk_kokkos_values")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: nlu
      integer(c_int), value       :: npiv, nz, draw
      real(c_double)              :: lu(*), zv(*), stats(3)
      integer(c_int)              :: piv(*), bad(*), dbad(*), ndbad
    end function c_blk_values
    integer(c_int) function c_blk_apply(h, x, y, xg, c) bind(C, name="jorek_blk_kokkos_apply")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value         :: h
      integer(c_intptr_t), value :: x, y, xg
      integer(c_int), value      :: c
    end function c_blk_apply
    integer(c_int) function c_blk_free(h) bind(C, name="jorek_blk_kokkos_free")
      use, intrinsic :: iso_c_binding
      type(c_ptr) :: h
    end function c_blk_free
    ! the device ring solver (jorek_ring_kokkos.cpp); this declaration's form
    ! links the C++ file, as above
    function jorek_ring_kokkos(h, nr, nrow, ngh, hdr, hdr8, ni, n8, si, s8, l8, rows, stats) bind(C)
      use, intrinsic :: iso_c_binding
      integer(c_int)        :: jorek_ring_kokkos
      type(c_ptr)           :: h
      integer(c_int), value :: nr, nrow, ngh
      integer(c_int)        :: hdr(*), ni(*), si(*), rows(*)
      integer(c_long_long)  :: hdr8(*), n8(*), s8(*), l8(*)
      real(c_double)        :: stats(4)
    end function jorek_ring_kokkos
    integer(c_int) function c_ring_factor(h, nfac, raw, ng8, g8, newpat, sp, devfill, ad, ao, nvg, vg, fail, ok, stats) &
        bind(C, name="jorek_ring_kokkos_factor")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: nfac, nvg
      real(c_double)              :: raw(*), vg(*), stats(4)
      integer(c_int), value       :: ng8, newpat, sp, devfill
      integer(c_intptr_t), value  :: ad, ao
      integer(c_long_long)        :: g8(*)
      integer(c_int)              :: fail(*), ok
    end function c_ring_factor
    integer(c_int) function c_blk_fillmap(h, n1, src, dst, n2, gsrc, gdst) bind(C, name="jorek_blk_kokkos_fillmap")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: n1, n2
      integer(c_int)              :: src(*), gsrc(*)
      integer(c_long_long)        :: dst(*), gdst(*)
    end function c_blk_fillmap
    integer(c_int) function c_blk_fill(h, nlu, nz, ad, ao, nvg, vg, dbad, ndbad) bind(C, name="jorek_blk_kokkos_fill")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: nlu, nvg
      integer(c_int), value       :: nz
      integer(c_intptr_t), value  :: ad, ao
      real(c_double)              :: vg(*)
      integer(c_int)              :: dbad(*), ndbad
    end function c_blk_fill
    integer(c_int) function c_ring_fillmap(h, n1, src, dst, n2, gsrc, gdst) bind(C, name="jorek_ring_kokkos_fillmap")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: n1, n2
      integer(c_int)              :: src(*), gsrc(*)
      integer(c_long_long)        :: dst(*), gdst(*)
    end function c_ring_fillmap
    integer(c_int) function c_ring_values(h, nfac, fac, jac, sp, stats) bind(C, name="jorek_ring_kokkos_values")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: nfac
      real(c_double)              :: fac(*), stats(4)
      integer(c_int)              :: jac(*)
      integer(c_int), value       :: sp
    end function c_ring_values
    integer(c_int) function c_ring_apply(h, x, y, xg, c, bh) bind(C, name="jorek_ring_kokkos_apply")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value         :: h, bh
      integer(c_intptr_t), value :: x, y, xg
      integer(c_int), value      :: c
    end function c_ring_apply
    integer(c_int) function c_ring_free(h) bind(C, name="jorek_ring_kokkos_free")
      use, intrinsic :: iso_c_binding
      type(c_ptr) :: h
    end function c_ring_free
    ! strided local copies between device vectors (jorek_vec_kokkos.cpp)
    function jorek_vec_kokkos(dst, doff, dstr, src, soff, sstr, blk, nblk) bind(C)
      use, intrinsic :: iso_c_binding
      integer(c_int)             :: jorek_vec_kokkos
      integer(c_intptr_t), value :: dst, src
      integer(c_int), value      :: doff, dstr, soff, sstr, blk, nblk
    end function jorek_vec_kokkos
    integer(c_int) function c_vec_touch(v) bind(C, name="jorek_vec_touch")
      use, intrinsic :: iso_c_binding
      integer(c_intptr_t), value :: v
    end function c_vec_touch
    ! the device coarse solve (jorek_crs_kokkos.cpp)
    function jorek_crs_kokkos(h, mode, fresh, n, rp, ci, va, stats) bind(C)
      use, intrinsic :: iso_c_binding
      integer(c_int)        :: jorek_crs_kokkos
      type(c_ptr)           :: h
      integer(c_int), value :: mode, fresh, n
      integer(c_int)        :: rp(*), ci(*)
      real(c_double)        :: va(*), stats(1)
    end function jorek_crs_kokkos
    integer(c_int) function c_crs_has(mode) bind(C, name="jorek_crs_kokkos_has")
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: mode
    end function c_crs_has
    integer(c_int) function c_crs_apply(h, b, x) bind(C, name="jorek_crs_kokkos_apply")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value         :: h
      integer(c_intptr_t), value :: b, x
    end function c_crs_apply
    integer(c_int) function c_crs_map(h, ntot, nloc, lidx, loff, fcomm, np, cnt, dsp) bind(C, name="jorek_crs_kokkos_map")
      use, intrinsic :: iso_c_binding
      type(c_ptr)           :: h
      integer(c_int), value :: ntot, nloc, loff, fcomm, np
      integer(c_int)        :: lidx(*), cnt(*), dsp(*)
    end function c_crs_map
    integer(c_int) function c_crs_free(h) bind(C, name="jorek_crs_kokkos_free")
      use, intrinsic :: iso_c_binding
      type(c_ptr) :: h
    end function c_crs_free
  end interface
#endif

contains

  logical function blk_dev_available()
#ifdef USE_GPU_PC
    blk_dev_available = .true.
#else
    blk_dev_available = .false.
#endif
  end function blk_dev_available

  !> Structure of a level's blocks (blk_t's arrays), once per operator
  !! pattern; h = c_null_ptr creates the handle. stats = largest block, widest
  !! band, multiply-adds of one pass over the blocks.
  subroutine blk_dev_pattern(h, nrow, ngh, off, sz, kl, ku, band, loff, axblk, rows, bcol, zp, zc, stats)
    type(c_ptr), intent(inout) :: h
    integer, intent(in) :: nrow, ngh
    integer, intent(in) :: off(:), sz(:), kl(:), ku(:), rows(:)
    logical, intent(in) :: band(:), axblk(:)
    integer(8), intent(in) :: loff(:)
    integer, intent(in), allocatable, target :: bcol(:), zp(:), zc(:)
    real*8, intent(out) :: stats(3)
#ifdef USE_GPU_PC
    integer(c_int), allocatable :: ib(:), ia(:)
    integer(c_int), target :: none(1)
    integer(c_int), pointer :: pzp(:), pzc(:)
    type(c_ptr) :: pcol
    integer :: nb, nzp, nz
    nb = size(sz)
    allocate(ib(nb), ia(nb))
    ib = merge(1, 0, band(1:nb)); ia = merge(1, 0, axblk(1:nb))
    none = 0
    pcol = c_null_ptr; pzp => none; pzc => none; nzp = 0; nz = 0
    if (allocated(bcol)) then
      pcol = c_loc(bcol)
      pzp => zp; pzc => zc; nzp = size(zp); nz = size(zc)
    endif
    if (jorek_blk_kokkos(h, int(nb, c_int), int(nrow, c_int), int(ngh, c_int), off, sz, kl, ku, ib, &
                         loff, ia, rows, pcol, int(nzp, c_int), pzp, int(nz, c_int), pzc, stats) /= 0) &
      stop "blk_dev_pattern: PETSc/Kokkos error"
#else
    stats = 0.d0
#endif
  end subroutine blk_dev_pattern

  !> The partitioned (SPIKE) form of the band blocks, at every rebuild, from
  !! the blocks' UNFACTORED band storage (before the host LU overwrites it).
  subroutine blk_dev_spike(h, lu)
    type(c_ptr), intent(in) :: h
    real*8, intent(in) :: lu(:)
#ifdef USE_GPU_PC
    real(c_double) :: st(3)
    if (c_blk_spike(h, int(size(lu, kind=8), c_long_long), lu, st) /= 0) stop "blk_dev_spike: PETSc/Kokkos error"
#endif
  end subroutine blk_dev_spike

  !> The factors (lu, LAPACK pivots) and the zebra coupling's values, at every
  !! rebuild; bad(b) /= 0: block b's host LU failed (point Jacobi), so it is
  !! not partitioned. stats = partitioned blocks, interiors, their device MB.
  !> raw: the dense blocks in lu are unfactored, inverted on the device
  !! (batched); nbad: how many were singular (then nothing is done for them:
  !! the caller factors on the host and calls again without raw)
  subroutine blk_dev_values(h, lu, piv, zv, bad, stats, raw, nbad)
    type(c_ptr), intent(in) :: h
    real*8, intent(in)  :: lu(:)
    integer, intent(in) :: piv(:), bad(:)
    real*8, intent(in), allocatable, target :: zv(:)
    real*8, intent(out) :: stats(3)
    logical, intent(in), optional :: raw
    integer, intent(out), optional :: nbad
#ifdef USE_GPU_PC
    real(c_double), target  :: none(1)
    real(c_double), pointer :: pzv(:)
    integer :: nz
    integer(c_int) :: draw, dbad(max(size(bad), 1)), ndbad
    none = 0.d0
    pzv => none; nz = 0
    if (allocated(zv)) then
      pzv => zv; nz = size(zv)
    endif
    draw = 0
    if (present(raw)) draw = merge(1_c_int, 0_c_int, raw)
    if (c_blk_values(h, int(size(lu, kind=8), c_long_long), lu, int(size(piv), c_int), piv, int(nz, c_int), pzv, &
                     bad, draw, dbad, ndbad, stats) /= 0) stop "blk_dev_values: PETSc/Kokkos error"
    if (present(nbad)) nbad = int(ndbad)
#else
    stats = 0.d0
    if (present(nbad)) nbad = 0
#endif
  end subroutine blk_dev_values

  !> y(block rows) = block solves of x on the device; xg = x on the ghost rows
  !! (read only when the level has any). colour: only that zebra colour's
  !! blocks (default: both, in order)
  subroutine blk_dev_apply(h, x, y, xg, colour)
    type(c_ptr), intent(in) :: h
    Vec, intent(in) :: x, y, xg
    integer, intent(in), optional :: colour
#ifdef USE_GPU_PC
    integer(c_int) :: c
    c = -1
    if (present(colour)) c = int(colour, c_int)
    if (c_blk_apply(h, transfer(x%v, 0_c_intptr_t), transfer(y%v, 0_c_intptr_t), transfer(xg%v, 0_c_intptr_t), c) /= 0) &
      stop "blk_dev_apply: PETSc/Kokkos error"
#endif
  end subroutine blk_dev_apply

  !> free a blk_dev handle, or (ring) a ring solver's
  subroutine blk_dev_free(h, ring)
    type(c_ptr), intent(inout) :: h
    logical, intent(in), optional :: ring
#ifdef USE_GPU_PC
    integer(c_int) :: rc
    logical :: rg
    rg = .false.
    if (present(ring)) rg = ring
    if (c_associated(h)) then
      if (rg) then
        rc = c_ring_free(h)
      else
        rc = c_blk_free(h)
      endif
    endif
#endif
    h = c_null_ptr
  end subroutine blk_dev_free

  !> v is a Kokkos (device path) vector
  logical function vec_is_dev(v)
    Vec, intent(in) :: v
    VecType :: vt
    PetscErrorCode :: ierr
    call VecGetType(v, vt, ierr)
    vec_is_dev = (index(vt, "kokkos") > 0)
  end function vec_is_dev

  !> v (just created, values not needed) becomes a Kokkos vector, same layout
  subroutine vec_make_dev(v)
    Vec, intent(inout) :: v
    PetscErrorCode :: ierr
#ifdef USE_GPU_PC
    call VecSetType(v, VECKOKKOS, ierr)
#endif
  end subroutine vec_make_dev

  !> v's values changed on some ranks only: a new state on this rank too, so
  !! that no rank answers a VecNorm from its cache (a skipped reduction)
  subroutine vec_touch(v)
    Vec, intent(in) :: v
#ifdef USE_GPU_PC
    if (c_vec_touch(transfer(v%v, 0_c_intptr_t)) /= 0) stop "vec_touch: PETSc error"
#endif
  end subroutine vec_touch

  !> dst(doff + i dstr + j) = src(soff + i sstr + j), j = 1..blk, i = 0..nblk-1
  !! (0-based offsets into the local arrays): on the device when both are
  !! device vectors, else on the host arrays
  subroutine vec_copy_strided(dst, doff, dstr, src, soff, sstr, blk, nblk)
    Vec, intent(in) :: dst, src
    integer, intent(in) :: doff, dstr, soff, sstr, blk, nblk
    PetscScalar, pointer :: da(:), sa(:)
    PetscErrorCode :: ierr
    integer :: i
#ifdef USE_GPU_PC
    if (vec_is_dev(dst) .and. vec_is_dev(src)) then
      if (jorek_vec_kokkos(transfer(dst%v, 0_c_intptr_t), int(doff, c_int), int(dstr, c_int), &
                           transfer(src%v, 0_c_intptr_t), int(soff, c_int), int(sstr, c_int), &
                           int(blk, c_int), int(nblk, c_int)) /= 0) stop "vec_copy_strided: PETSc/Kokkos error"
      return
    endif
#endif
    if (blk <= 0 .or. nblk <= 0) return
    call VecGetArrayRead(src, sa, ierr)
    call VecGetArray(dst, da, ierr)
    do i = 0, nblk - 1
      da(doff + i * dstr + 1 : doff + i * dstr + blk) = sa(soff + i * sstr + 1 : soff + i * sstr + blk)
    enddo
    call VecRestoreArray(dst, da, ierr)
    call VecRestoreArrayRead(src, sa, ierr)
  end subroutine vec_copy_strided

  !> the device coarse solve's mode (1 dense, 2 cuDSS) is built in
  logical function crs_dev_has(mode)
    integer, intent(in) :: mode
#ifdef USE_GPU_PC
    crs_dev_has = (c_crs_has(int(mode, c_int)) /= 0)
#else
    crs_dev_has = .false.
#endif
  end function crs_dev_has

  !> (Re)factor the coarsest matrix M (sequential AIJ) on the device; fresh:
  !! a new pattern. mb = its device MB.
  subroutine crs_dev_setup(h, mode, fresh, M, mb)
    type(c_ptr), intent(inout) :: h
    integer, intent(in) :: mode
    logical, intent(in) :: fresh
    Mat, intent(in) :: M
    real*8, intent(out) :: mb
#ifdef USE_GPU_PC
    PetscInt, pointer :: ia(:), ja(:)
    PetscBool :: done
    PetscErrorCode :: ierr
    PetscInt :: n
    PetscInt, parameter :: zero = 0
    PetscScalar, pointer :: va(:)
    real(c_double) :: st(1)
    integer(c_int), allocatable :: rp(:), ci(:)
    call MatGetRowIJ(M, zero, PETSC_FALSE, PETSC_FALSE, n, ia, ja, done, ierr)
    allocate(rp(n + 1), ci(ia(n + 1)))
    rp = int(ia(1:n + 1), c_int); ci = int(ja(1:ia(n + 1)), c_int)
    call MatSeqAIJGetArrayRead(M, va, ierr)
    if (jorek_crs_kokkos(h, int(mode, c_int), merge(1_c_int, 0_c_int, fresh), int(n, c_int), rp, ci, va, st) /= 0) &
      stop "crs_dev_setup: PETSc/Kokkos error"
    call MatSeqAIJRestoreArrayRead(M, va, ierr)
    call MatRestoreRowIJ(M, zero, PETSC_FALSE, PETSC_FALSE, n, ia, ja, done, ierr)
    mb = st(1)
#else
    mb = 0.d0
#endif
  end subroutine crs_dev_setup

  !> A member's map of a redundant solve: lidx = its own block rows (local,
  !! 0-based) at loff of the block's ntot rows; comm, cnt, dsp = the members'
  !! communicator, row counts and offsets (rds_t)
  subroutine crs_dev_map(h, ntot, lidx, loff, comm, cnt, dsp)
    type(c_ptr), intent(inout) :: h
    integer, intent(in) :: ntot, lidx(:), loff, comm, cnt(:), dsp(:)
#ifdef USE_GPU_PC
    integer(c_int) :: l1(max(size(lidx), 1)), c1(size(cnt)), d1(size(cnt))
    l1 = 0; l1(1:size(lidx)) = lidx
    c1 = cnt; d1 = dsp(1:size(cnt))
    if (c_crs_map(h, int(ntot, c_int), int(size(lidx), c_int), l1, int(loff, c_int), int(comm, c_int), &
                  int(size(cnt), c_int), c1, d1) /= 0) stop "crs_dev_map: PETSc/Kokkos error"
#endif
  end subroutine crs_dev_map

  !> x(own block rows) = A^-1 b(block rows), level vectors; members only
  subroutine crs_dev_apply(h, b, x)
    type(c_ptr), intent(in) :: h
    Vec, intent(in) :: b, x
#ifdef USE_GPU_PC
    if (c_crs_apply(h, transfer(b%v, 0_c_intptr_t), transfer(x%v, 0_c_intptr_t)) /= 0) &
      stop "crs_dev_apply: PETSc/Kokkos error"
#endif
  end subroutine crs_dev_apply

  subroutine crs_dev_free(h)
    type(c_ptr), intent(inout) :: h
#ifdef USE_GPU_PC
    integer(c_int) :: rc
    if (c_associated(h)) rc = c_crs_free(h)
#endif
    h = c_null_ptr
  end subroutine crs_dev_free

  !> Layout of a level's ring-solver blocks (mod_petsc_pc_gmg ring_dev_upload
  !! flattens them), once per pattern; h = c_null_ptr creates the handle.
  !! stats = rings, segments per ring, largest node block (MB: at values).
  subroutine ring_dev_pattern(h, nr, nrow, ngh, hdr, hdr8, ni, n8, si, s8, l8, rows, stats)
    type(c_ptr), intent(inout) :: h
    integer, intent(in) :: nr, nrow, ngh
    integer, intent(in) :: hdr(:), ni(:), si(:), rows(:)
    integer(8), intent(in) :: hdr8(:), n8(:), s8(:), l8(:)
    real*8, intent(inout) :: stats(4)
#ifdef USE_GPU_PC
    if (jorek_ring_kokkos(h, int(nr, c_int), int(nrow, c_int), int(ngh, c_int), hdr, hdr8, ni, n8, si, s8, l8, &
                          rows, stats) /= 0) stop "ring_dev_pattern: PETSc/Kokkos error"
#else
    stats = 0.d0
#endif
  end subroutine ring_dev_pattern

  !> The rings' factors in their device form (btd_dev_pack), every rebuild;
  !! jac(r) /= 0: ring r is point Jacobi; sp: keep them in single precision;
  !! stats(4) = their device MB
  subroutine ring_dev_values(h, fac, jac, sp, stats)
    type(c_ptr), intent(in) :: h
    real*8, intent(in)  :: fac(:)
    integer, intent(in) :: jac(:)
    logical, intent(in) :: sp
    real*8, intent(inout) :: stats(4)
#ifdef USE_GPU_PC
    if (c_ring_values(h, int(size(fac, kind=8), c_long_long), fac, jac, merge(1_c_int, 0_c_int, sp), stats) /= 0) &
      stop "ring_dev_values: PETSc/Kokkos error"
#endif
  end subroutine ring_dev_values

  !> The rings factored on the device, from their unfactored storage raw
  !! (ring_dev_upload's packing); g8: per separator its segment's S_ab, S_ba,
  !! S_bb offsets; newpat: a new layout. fail(r) /= 0: ring r singular;
  !! ok = .false.: no device factorisation for this layout.
  !! Device fill (Ad present): the values from the level matrix's device parts
  !! Ad / Ao (ao_ok: Ao exists) and the ghost rows' values vg through the map
  !! of ring_dev_fillmap, raw then only sizes the factors (nfac).
  subroutine ring_dev_factor(h, raw, g8, newpat, sp, fail, ok, stats, nfac, Ad, Ao, ao_ok, vg)
    type(c_ptr), intent(in) :: h
    real*8, intent(in)  :: raw(:)
    integer(8), intent(in) :: g8(:)
    logical, intent(in) :: newpat, sp
    integer, intent(out) :: fail(:)
    logical, intent(out) :: ok
    real*8, intent(inout) :: stats(4)
    integer(8), intent(in) :: nfac
    Mat, intent(in), optional :: Ad, Ao
    logical, intent(in), optional :: ao_ok
    real*8, intent(in), optional :: vg(:)
#ifdef USE_GPU_PC
    integer(c_int) :: f(max(size(fail), 1)), okc, dfill
    integer(c_long_long) :: g(max(size(g8), 1)), nvg
    integer(c_intptr_t) :: pa, po
    real(c_double) :: none(1)
    g = 0; g(1:size(g8)) = g8
    dfill = 0; pa = 0; po = 0; nvg = 0; none = 0.d0
    if (present(Ad)) then
      dfill = 1
      pa = transfer(Ad%v, 0_c_intptr_t)
      if (ao_ok) po = transfer(Ao%v, 0_c_intptr_t)
      nvg = size(vg)
    endif
    if (dfill == 1 .and. nvg > 0) then
      if (c_ring_factor(h, int(nfac, c_long_long), raw, int(size(g8), c_int), g, merge(1_c_int, 0_c_int, newpat), &
                        merge(1_c_int, 0_c_int, sp), dfill, pa, po, nvg, vg, f, okc, stats) /= 0) &
        stop "ring_dev_factor: PETSc/Kokkos error"
    else
      if (c_ring_factor(h, int(nfac, c_long_long), raw, int(size(g8), c_int), g, merge(1_c_int, 0_c_int, newpat), &
                        merge(1_c_int, 0_c_int, sp), dfill, pa, po, nvg, none, f, okc, stats) /= 0) &
        stop "ring_dev_factor: PETSc/Kokkos error"
    endif
    fail = f(1:size(fail)); ok = (okc /= 0)
#else
    fail = 0; ok = .false.
#endif
  end subroutine ring_dev_factor

  !> The device fill's map: entry k of the level matrix's values (src(k) = +i
  !! diagonal part, -i off-diagonal part) to dst(k) of the rings' packed
  !! factors; gsrc / gdst the same from the ghost rows' values
  subroutine ring_dev_fillmap(h, src, dst, gsrc, gdst)
    type(c_ptr), intent(in) :: h
    integer, intent(in) :: src(:), gsrc(:)
    integer(8), intent(in) :: dst(:), gdst(:)
#ifdef USE_GPU_PC
    integer(c_int) :: s1(max(size(src), 1)), s2(max(size(gsrc), 1))
    integer(c_long_long) :: d1(max(size(dst), 1)), d2(max(size(gdst), 1))
    s1 = 0; s2 = 0; d1 = 0; d2 = 0
    s1(1:size(src)) = src; d1(1:size(dst)) = dst; s2(1:size(gsrc)) = gsrc; d2(1:size(gdst)) = gdst
    if (c_ring_fillmap(h, int(size(src), c_long_long), s1, d1, int(size(gsrc), c_long_long), s2, d2) /= 0) &
      stop "ring_dev_fillmap: PETSc/Kokkos error"
#endif
  end subroutine ring_dev_fillmap

  !> The block smoother's device fill map (the level's host value-map share:
  !! src(k) = +i / -i entry of the diagonal / off-diagonal part's values, dst
  !! = 1-based position in lu, or -position in the zebra coupling's values;
  !! gsrc / gdst the same from the ghost rows' values), once per pattern
  subroutine blk_dev_fillmap(h, src, dst, gsrc, gdst)
    type(c_ptr), intent(in) :: h
    integer, intent(in) :: src(:), gsrc(:)
    integer(8), intent(in) :: dst(:), gdst(:)
#ifdef USE_GPU_PC
    integer(c_int) :: z4(1)
    integer(c_long_long) :: z8(1)
    integer(c_int) :: rc
    z4 = 0; z8 = 0
    if (size(gsrc) > 0) then
      rc = c_blk_fillmap(h, size(src, kind=c_long_long), src, dst, size(gsrc, kind=c_long_long), gsrc, gdst)
    else
      rc = c_blk_fillmap(h, size(src, kind=c_long_long), src, dst, 0_c_long_long, z4, z8)
    endif
    if (rc /= 0) stop "blk_dev_fillmap: PETSc/Kokkos error"
#endif
  end subroutine blk_dev_fillmap

  !> Every rebuild, in place of the host fill + blk_dev_values: the dense
  !! blocks' and the zebra coupling's values from the level's device matrix
  !! parts (Ad, Ao when ao_ok) and the ghost rows' values vg, the dense
  !! blocks inverted there (nlu / nz: the host arrays' lengths). bad(b) = 1:
  !! dense block b singular; nbad > 0: the caller fills and factors on the host.
  subroutine blk_dev_fill(h, nlu, nz, Ad, Ao, ao_ok, vg, bad, nbad)
    type(c_ptr), intent(in) :: h
    integer(8), intent(in) :: nlu
    integer, intent(in) :: nz
    Mat, intent(in) :: Ad, Ao
    logical, intent(in) :: ao_ok
    real*8, intent(in) :: vg(:)
    integer, intent(out) :: bad(:), nbad
#ifdef USE_GPU_PC
    integer(c_intptr_t) :: po
    integer(c_int) :: b(max(size(bad), 1)), nb
    real(c_double) :: none(1)
    integer(c_int) :: rc
    po = 0
    if (ao_ok) po = transfer(Ao%v, 0_c_intptr_t)
    none = 0.d0
    if (size(vg) > 0) then
      rc = c_blk_fill(h, int(nlu, c_long_long), int(nz, c_int), transfer(Ad%v, 0_c_intptr_t), po, &
                      size(vg, kind=c_long_long), vg, b, nb)
    else
      rc = c_blk_fill(h, int(nlu, c_long_long), int(nz, c_int), transfer(Ad%v, 0_c_intptr_t), po, &
                      0_c_long_long, none, b, nb)
    endif
    if (rc /= 0) stop "blk_dev_fill: PETSc/Kokkos error"
    bad = b(1:size(bad)); nbad = nb
#else
    bad = 0; nbad = 1
#endif
  end subroutine blk_dev_fill

  !> y(ring rows) = ring solves of x on the device; xg = x on the ghost rows.
  !! colour, bh: zebra, only that colour's rings / lines; colour 1 subtracts
  !! the coupling to colour 0's solution, held by blk_dev handle bh
  subroutine ring_dev_apply(h, x, y, xg, colour, bh)
    type(c_ptr), intent(in) :: h
    Vec, intent(in) :: x, y, xg
    integer, intent(in), optional :: colour
    type(c_ptr), intent(in), optional :: bh
#ifdef USE_GPU_PC
    integer(c_int) :: c
    type(c_ptr) :: b
    c = -1; b = c_null_ptr
    if (present(colour)) c = int(colour, c_int)
    if (present(bh)) b = bh
    if (c_ring_apply(h, transfer(x%v, 0_c_intptr_t), transfer(y%v, 0_c_intptr_t), transfer(xg%v, 0_c_intptr_t), c, b) /= 0) &
      stop "ring_dev_apply: PETSc/Kokkos error"
#endif
  end subroutine ring_dev_apply

#endif
end module mod_petsc_blk_dev
