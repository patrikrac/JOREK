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
    integer(c_int) function c_blk_values(h, nlu, lu, npiv, piv, nz, zv, bad, stats) bind(C, name="jorek_blk_kokkos_values")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value          :: h
      integer(c_long_long), value :: nlu
      integer(c_int), value       :: npiv, nz
      real(c_double)              :: lu(*), zv(*), stats(3)
      integer(c_int)              :: piv(*), bad(*)
    end function c_blk_values
    integer(c_int) function c_blk_apply(h, x, y, xg) bind(C, name="jorek_blk_kokkos_apply")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value         :: h
      integer(c_intptr_t), value :: x, y, xg
    end function c_blk_apply
    integer(c_int) function c_blk_free(h) bind(C, name="jorek_blk_kokkos_free")
      use, intrinsic :: iso_c_binding
      type(c_ptr) :: h
    end function c_blk_free
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
  subroutine blk_dev_values(h, lu, piv, zv, bad, stats)
    type(c_ptr), intent(in) :: h
    real*8, intent(in)  :: lu(:)
    integer, intent(in) :: piv(:), bad(:)
    real*8, intent(in), allocatable, target :: zv(:)
    real*8, intent(out) :: stats(3)
#ifdef USE_GPU_PC
    real(c_double), target  :: none(1)
    real(c_double), pointer :: pzv(:)
    integer :: nz
    none = 0.d0
    pzv => none; nz = 0
    if (allocated(zv)) then
      pzv => zv; nz = size(zv)
    endif
    if (c_blk_values(h, int(size(lu, kind=8), c_long_long), lu, int(size(piv), c_int), piv, int(nz, c_int), pzv, &
                     bad, stats) /= 0) stop "blk_dev_values: PETSc/Kokkos error"
#else
    stats = 0.d0
#endif
  end subroutine blk_dev_values

  !> y(block rows) = block solves of x on the device; xg = x on the ghost rows
  !! (read only when the level has any).
  subroutine blk_dev_apply(h, x, y, xg)
    type(c_ptr), intent(in) :: h
    Vec, intent(in) :: x, y, xg
#ifdef USE_GPU_PC
    if (c_blk_apply(h, transfer(x%v, 0_c_intptr_t), transfer(y%v, 0_c_intptr_t), transfer(xg%v, 0_c_intptr_t)) /= 0) &
      stop "blk_dev_apply: PETSc/Kokkos error"
#endif
  end subroutine blk_dev_apply

  subroutine blk_dev_free(h)
    type(c_ptr), intent(inout) :: h
#ifdef USE_GPU_PC
    integer(c_int) :: rc
    if (c_associated(h)) rc = c_blk_free(h)
#endif
    h = c_null_ptr
  end subroutine blk_dev_free

#endif
end module mod_petsc_blk_dev
