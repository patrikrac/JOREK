module mod_petsc_raw_csr
#ifdef USE_PETSC
  use iso_c_binding
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  !--------------------------------------------------------------------
  !> Raw CSR access to PETSc AIJ/BAIJ matrices, for the value maps of the
  !! physics PC (mod_petsc_pc_sf_gather, the GMG smoother blocks).
  !!
  !! A value map records, once per run, where each stored entry of a target
  !! comes from in a source's value array; a rebuild is then one indexed copy.
  !! That needs the diagonal/off-diagonal split of an MPI matrix and the
  !! value arrays of its sequential parts, which PETSc exposes only in C (the
  !! Fortran binding of MatMPIBAIJGetSeqBAIJ mishandles its colmap), so those
  !! calls go through iso_c_binding on the handles' addresses.
  !--------------------------------------------------------------------

  interface
    integer(c_int) function c_mpibaij_seq(A, Ad, Ao, cmap) bind(C, name="MatMPIBAIJGetSeqBAIJ")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: A
      integer(c_intptr_t)        :: Ad, Ao
      type(c_ptr)                :: cmap
    end function c_mpibaij_seq
    integer(c_int) function c_mpiaij_seq(A, Ad, Ao, cmap) bind(C, name="MatMPIAIJGetSeqAIJ")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: A
      integer(c_intptr_t)        :: Ad, Ao
      type(c_ptr)                :: cmap
    end function c_mpiaij_seq
    integer(c_int) function c_baij_get(M, arr) bind(C, name="MatSeqBAIJGetArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_baij_get
    integer(c_int) function c_baij_restore(M, arr) bind(C, name="MatSeqBAIJRestoreArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_baij_restore
    integer(c_int) function c_aij_get(M, arr) bind(C, name="MatSeqAIJGetArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_get
    integer(c_int) function c_aij_restore(M, arr) bind(C, name="MatSeqAIJRestoreArray")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_restore
    ! read-only access: does not advance the matrix's object state
    integer(c_int) function c_aij_get_read(M, arr) bind(C, name="MatSeqAIJGetArrayRead")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_get_read
    integer(c_int) function c_aij_restore_read(M, arr) bind(C, name="MatSeqAIJRestoreArrayRead")
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t, c_ptr
      integer(c_intptr_t), value :: M
      type(c_ptr)                :: arr
    end function c_aij_restore_read
    ! the threaded block matvec of jorek_blockmv_attach.c (this declaration's
    ! form is what util/makedepend links the C file by)
    function jorek_blockmv_attach(M, bs, ok) bind(C)
      use, intrinsic :: iso_c_binding, only: c_int, c_intptr_t
      integer(c_int)             :: jorek_blockmv_attach
      integer(c_intptr_t), value :: M
      integer(c_int), value      :: bs
      integer(c_int)             :: ok
    end function jorek_blockmv_attach
#ifdef USE_GPU_PC
    ! the device gathers of jorek_sfg_kokkos.cpp
    ! 1 when the gathers run here (this declaration's form is what
    ! util/makedepend links jorek_sfg_kokkos.cpp by)
    function jorek_sfg_kokkos() bind(C)
      import :: c_int
      integer(c_int) :: jorek_sfg_kokkos
    end function jorek_sfg_kokkos
    integer(c_int) function c_sfg_source(which, Bd, Bo, nd, vd, no, vo) bind(C, name="jorek_sfg_source")
      import :: c_int, c_intptr_t, c_long_long, c_ptr
      integer(c_int), value :: which
      integer(c_intptr_t), value :: Bd, Bo
      integer(c_long_long), value :: nd, no
      type(c_ptr), value :: vd, vo
    end function c_sfg_source
    integer(c_int) function c_sfg_expand_setup(T, Ad, Ao, bs, nbr, bia_d, bia_o) bind(C, name="jorek_sfg_expand_setup")
      import :: c_int, c_intptr_t
      integer(c_intptr_t), value :: T, Ad, Ao
      integer(c_int), value :: bs, nbr
      integer(c_int) :: bia_d(*), bia_o(*)
    end function c_sfg_expand_setup
    integer(c_int) function c_sfg_expand() bind(C, name="jorek_sfg_expand")
      import :: c_int
    end function c_sfg_expand
    integer(c_int) function c_sfg_target(k, T, Td, To, nd, md, wd, zd, no, mo, wo, zo, hw, hz) &
        bind(C, name="jorek_sfg_target")
      import :: c_int, c_intptr_t, c_long_long, c_int32_t, c_int8_t
      integer(c_int), value :: k, hw, hz
      integer(c_intptr_t), value :: T, Td, To
      integer(c_long_long), value :: nd, no
      integer(c_int32_t) :: md(*), wd(*), mo(*), wo(*)
      integer(c_int8_t) :: zd(*), zo(*)
    end function c_sfg_target
    integer(c_int) function c_sfg_target_clear() bind(C, name="jorek_sfg_target_clear")
      import :: c_int
    end function c_sfg_target_clear
    integer(c_int) function c_sfg_fam_set(k, n, sa, sw, ss, sc, hw, hz, T, Td, To, nval, perm) &
        bind(C, name="jorek_sfg_fam_set")
      import :: c_int, c_intptr_t, c_long_long, c_int32_t, c_int8_t
      integer(c_int), value :: k, hw, hz
      integer(c_long_long), value :: n, nval
      integer(c_int32_t) :: sa(*), sw(*)
      integer(c_int8_t) :: ss(*), sc(*)
      integer(c_intptr_t), value :: T, Td, To
      integer(c_long_long) :: perm(*)
    end function c_sfg_fam_set
    integer(c_int) function c_sfg_fam_values(k, opz, sbuf) bind(C, name="jorek_sfg_fam_values")
      import :: c_int, c_double
      integer(c_int), value :: k
      real(c_double), value :: opz
      real(c_double) :: sbuf(*)
    end function c_sfg_fam_values
    integer(c_int) function c_sfg_fam_install(k, val) bind(C, name="jorek_sfg_fam_install")
      import :: c_int, c_double
      integer(c_int), value :: k
      real(c_double) :: val(*)
    end function c_sfg_fam_install
    integer(c_int) function c_sfg_release() bind(C, name="jorek_sfg_release")
      import :: c_int
    end function c_sfg_release
    integer(c_int) function c_sfg_gather(k, opz) bind(C, name="jorek_sfg_gather")
      import :: c_int, c_double
      integer(c_int), value :: k
      real(c_double), value :: opz
    end function c_sfg_gather
#endif
  end interface

  public :: c_baij_get, c_baij_restore, c_aij_get, c_aij_restore
  public :: baij_aij_map, baij_aij_fill
  public :: split_parts, get_ij, put_ij, aij_parts, aij_vals_read, aij_vals_done
  public :: aij_vals_write, aij_vals_written
  public :: blockmv_attach
  public :: sfg_dev_available, sfg_dev_source, sfg_dev_expand_setup, sfg_dev_expand
  public :: sfg_dev_target, sfg_dev_target_clear, sfg_dev_gather
  public :: sfg_dev_fam_set, sfg_dev_fam_values, sfg_dev_fam_install, sfg_dev_release

  !> Where each entry of an AIJ copy of an MPIBAIJ matrix (same pattern, every
  !! block entry stored, as MatConvert makes it) reads from the BAIJ value
  !! arrays: md / mo per diagonal / off-diagonal AIJ entry, the 1-based index
  !! into the BAIJ part's values. ok: the patterns matched entry by entry.
  type, public :: baij_map_t
    integer(8), allocatable :: md(:), mo(:)
    logical :: ok = .false.
  end type baij_map_t

contains

  !> Replace M's MatMult / MatMultAdd by the OpenMP kernel of jorek_blockmv_attach.c
  !! (bs consecutive rows sharing one column pattern read it once). Only for
  !! (MPI/Seq)AIJ; returns .false. and leaves M alone otherwise. Collective;
  !! idempotent, so it can follow every refill of M.
  logical function blockmv_attach(M, bs)
    Mat, intent(in)     :: M
    integer, intent(in) :: bs
    integer(c_int) :: ok
    ok = 0
    if (jorek_blockmv_attach(transfer(M%v, 0_c_intptr_t), int(bs, c_int), ok) /= 0) &
      stop "blockmv_attach: PETSc error"
    blockmv_attach = (ok /= 0)
  end function blockmv_attach

  !> Diagonal/off-diagonal sequential parts of an MPI(B)AIJ matrix, and the
  !! off-diagonal part's global (block) column of each local column.
  subroutine split_parts(M, is_baij, Md, Mo, garr)
    Mat, intent(in)  :: M
    logical, intent(in) :: is_baij
    Mat, intent(out) :: Md, Mo
    PetscInt, pointer, intent(out) :: garr(:)
    integer(c_intptr_t) :: pd, po
    type(c_ptr) :: cm
    integer(c_int) :: rc
    PetscInt :: nr, nc, bs
    PetscErrorCode :: ierr
    if (is_baij) then
      rc = c_mpibaij_seq(transfer(M%v, 0_c_intptr_t), pd, po, cm)
    else
      rc = c_mpiaij_seq(transfer(M%v, 0_c_intptr_t), pd, po, cm)
    endif
    Md%v = transfer(pd, Md%v); Mo%v = transfer(po, Mo%v)
    call MatGetSize(Mo, nr, nc, ierr)
    bs = 1
    if (is_baij) call MatGetBlockSize(Mo, bs, ierr)
    if (nc / bs > 0) then
      call c_f_pointer(cm, garr, [nc / bs])
    else
      garr => null()
    endif
  end subroutine split_parts

  !> split_parts for an AIJ matrix that may also be sequential (a one-rank
  !! communicator can give either): has_o = .false. means M is its own
  !! diagonal part and there is no off-diagonal part.
  subroutine aij_parts(M, Md, Mo, garr, has_o)
    Mat, intent(in)  :: M
    Mat, intent(out) :: Md, Mo
    PetscInt, pointer, intent(out) :: garr(:)
    logical, intent(out) :: has_o
    MatType :: mt
    PetscErrorCode :: ierr
    call MatGetType(M, mt, ierr)
    has_o = (index(mt, "mpi") == 1)
    if (has_o) then
      call split_parts(M, .false., Md, Mo, garr)
    else if (index(mt, "seqaij") == 1) then
      Md = M; garr => null()
    else
      write(*,'(A,A)') "[Physics PC]   FATAL: raw CSR access needs an AIJ matrix, got ", trim(mt)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
  end subroutine aij_parts

  !> Block CSR (BAIJ) or CSR (AIJ) of a sequential part, 0-based.
  subroutine get_ij(M, blocked, n, ia, ja)
    Mat, intent(in) :: M
    logical, intent(in) :: blocked
    PetscInt, intent(out) :: n
    PetscInt, pointer :: ia(:), ja(:)
    PetscBool :: done, bc
    PetscErrorCode :: ierr
    PetscInt, parameter :: zero = 0
    bc = PETSC_FALSE
    if (blocked) bc = PETSC_TRUE
    call MatGetRowIJ(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
  end subroutine get_ij

  subroutine put_ij(M, blocked, n, ia, ja)
    Mat, intent(in) :: M
    logical, intent(in) :: blocked
    PetscInt, intent(inout) :: n
    PetscInt, pointer :: ia(:), ja(:)
    PetscBool :: done, bc
    PetscErrorCode :: ierr
    PetscInt, parameter :: zero = 0
    bc = PETSC_FALSE
    if (blocked) bc = PETSC_TRUE
    call MatRestoreRowIJ(M, zero, PETSC_FALSE, bc, n, ia, ja, done, ierr)
  end subroutine put_ij

  !> Read-only value array (length n) of a sequential AIJ matrix; p is the
  !! handle aij_vals_done needs.
  subroutine aij_vals_read(M, n, p, v)
    Mat, intent(in) :: M
    integer(8), intent(in) :: n
    type(c_ptr), intent(out) :: p
    real(c_double), pointer, intent(out) :: v(:)
    integer(c_int) :: rc
    rc = c_aij_get_read(transfer(M%v, 0_c_intptr_t), p)
    call c_f_pointer(p, v, [max(n, 1_8)])
  end subroutine aij_vals_read

  subroutine aij_vals_done(M, p)
    Mat, intent(in) :: M
    type(c_ptr), intent(inout) :: p
    integer(c_int) :: rc
    rc = c_aij_restore_read(transfer(M%v, 0_c_intptr_t), p)
  end subroutine aij_vals_done

  !> Writable value array (length n) of a sequential AIJ matrix. Through
  !! PETSc's accessor, so a device matrix type knows its host values changed.
  subroutine aij_vals_write(M, n, p, v)
    Mat, intent(in) :: M
    integer(8), intent(in) :: n
    type(c_ptr), intent(out) :: p
    real(c_double), pointer, intent(out) :: v(:)
    integer(c_int) :: rc
    rc = c_aij_get(transfer(M%v, 0_c_intptr_t), p)
    call c_f_pointer(p, v, [max(n, 1_8)])
  end subroutine aij_vals_write

  subroutine aij_vals_written(M, p)
    Mat, intent(in) :: M
    type(c_ptr), intent(inout) :: p
    integer(c_int) :: rc
    rc = c_aij_restore(transfer(M%v, 0_c_intptr_t), p)
  end subroutine aij_vals_written

  !> The value map of A (AIJ, any device subtype) from B (MPIBAIJ): BAIJ blocks
  !! are column-major, block k's entry (i, j) at k bs^2 + j bs + i; each AIJ
  !! row lists its blocks' columns in order. Checked column by column.
  subroutine baij_aij_map(B, A, map)
    Mat, intent(in) :: B, A
    type(baij_map_t), intent(inout) :: map
    Mat :: Bd, Bo, Ad, Ao
    PetscInt, pointer :: bgar(:), agar(:)
    PetscInt :: bs
    PetscErrorCode :: ierr
    logical :: has_o, ok1, ok2
    call split_parts(B, .true., Bd, Bo, bgar)
    call aij_parts(A, Ad, Ao, agar, has_o)
    call MatGetBlockSize(B, bs, ierr)
    call one(Bd, Ad, .false., map%md, ok1)
    ok2 = .true.
    if (has_o) call one(Bo, Ao, .true., map%mo, ok2)
    map%ok = ok1 .and. ok2 .and. has_o
  contains
    subroutine one(Bp, Ap, offd, m, ok)
      Mat, intent(in) :: Bp, Ap
      logical, intent(in) :: offd
      integer(8), allocatable, intent(inout) :: m(:)
      logical, intent(out) :: ok
      PetscInt :: nb, n
      PetscInt, pointer :: bia(:), bja(:), aia(:), aja(:)
      integer :: ib, i, j
      integer(8) :: k, p, colx, cola
      call get_ij(Bp, .true., nb, bia, bja)
      call get_ij(Ap, .false., n, aia, aja)
      ok = (n == nb * bs)
      if (allocated(m)) deallocate(m)
      allocate(m(max(int(aia(n + 1), 8), 1_8)))
      if (ok) then
        do ib = 0, int(nb) - 1
          do i = 0, int(bs) - 1
            p = aia(ib * bs + i + 1)
            do k = bia(ib + 1), bia(ib + 2) - 1
              do j = 0, int(bs) - 1
                if (p >= aia(ib * bs + i + 2)) then
                  ok = .false.; exit
                endif
                if (offd) then
                  colx = int(bgar(bja(k + 1) + 1), 8) * bs + j
                  cola = agar(aja(p + 1) + 1)
                else
                  colx = int(bja(k + 1), 8) * bs + j
                  cola = aja(p + 1)
                endif
                if (colx /= cola) ok = .false.
                m(p + 1) = k * bs * bs + int(j, 8) * bs + i + 1
                p = p + 1
              enddo
            enddo
            if (p /= aia(ib * bs + i + 2)) ok = .false.
          enddo
        enddo
      endif
      call put_ij(Ap, .false., n, aia, aja)
      call put_ij(Bp, .true., nb, bia, bja)
    end subroutine one
  end subroutine baij_aij_map

  !> A's values = B's through the map (baij_aij_map); host arrays, the device
  !! copy of a Kokkos A follows at its next use
  subroutine baij_aij_fill(B, A, map)
    Mat, intent(in) :: B, A
    type(baij_map_t), intent(in) :: map
    Mat :: Bd, Bo, Ad, Ao
    PetscInt, pointer :: bgar(:), agar(:)
    logical :: has_o
    call split_parts(B, .true., Bd, Bo, bgar)
    call aij_parts(A, Ad, Ao, agar, has_o)
    call one(Bd, Ad, map%md)
    if (has_o) call one(Bo, Ao, map%mo)
  contains
    subroutine one(Bp, Ap, m)
      Mat, intent(in) :: Bp, Ap
      integer(8), intent(in) :: m(:)
      type(c_ptr) :: pb, pa
      real(c_double), pointer :: vb(:), va(:)
      integer(c_int) :: rc
      integer(8) :: q, n
      n = size(m, kind=8)
      rc = c_baij_get(transfer(Bp%v, 0_c_intptr_t), pb); call c_f_pointer(pb, vb, [huge(1)])
      call aij_vals_write(Ap, n, pa, va)
      !$omp parallel do schedule(static)
      do q = 1, n
        va(q) = vb(m(q))
      enddo
      !$omp end parallel do
      call aij_vals_written(Ap, pa)
      rc = c_baij_restore(transfer(Bp%v, 0_c_intptr_t), pb)
    end subroutine one
  end subroutine baij_aij_fill

  !> The device gathers (jorek_sfg_kokkos.cpp) can run in this build
  logical function sfg_dev_available()
#ifdef USE_GPU_PC
    sfg_dev_available = (jorek_sfg_kokkos() /= 0)
#else
    sfg_dev_available = .false.
#endif
  end function sfg_dev_available

  !> Source which (0: A, 1 .. 4: the W's) on the device from the BAIJ matrix
  !! B's value arrays; uploaded only when B's parts changed since the last
  !! upload.
  subroutine sfg_dev_source(which, B)
    integer, intent(in) :: which
    Mat, intent(in) :: B
#ifdef USE_GPU_PC
    Mat :: Bd, Bo
    PetscInt, pointer :: garr(:), ia(:), ja(:)
    PetscInt :: bs, n
    PetscErrorCode :: ierr
    type(c_ptr) :: pd, po
    integer(c_long_long) :: nd, no
    integer(c_int) :: rc
    integer(c_intptr_t) :: ho
    call split_parts(B, .true., Bd, Bo, garr)
    call MatGetBlockSize(B, bs, ierr)
    call get_ij(Bd, .true., n, ia, ja); nd = int(ia(n + 1), c_long_long) * bs * bs; call put_ij(Bd, .true., n, ia, ja)
    call get_ij(Bo, .true., n, ia, ja); no = int(ia(n + 1), c_long_long) * bs * bs; call put_ij(Bo, .true., n, ia, ja)
    ho = transfer(Bo%v, 0_c_intptr_t)
    rc = c_baij_get(transfer(Bd%v, 0_c_intptr_t), pd)
    rc = c_baij_get(ho, po)
    rc = c_sfg_source(int(which, c_int), transfer(Bd%v, 0_c_intptr_t), ho, nd, pd, no, po)
    if (rc /= 0) stop "sfg_dev_source: PETSc/Kokkos error"
    rc = c_baij_restore(ho, po)
    rc = c_baij_restore(transfer(Bd%v, 0_c_intptr_t), pd)
#endif
  end subroutine sfg_dev_source

  !> T (an aijkokkos copy of the BAIJ matrix B, every block entry stored, in
  !! baij_aij_map's order) filled on the device from source 0: its layout
  subroutine sfg_dev_expand_setup(T, B)
    Mat, intent(in) :: T, B
#ifdef USE_GPU_PC
    Mat :: Bd, Bo, Td, To
    PetscInt, pointer :: bgar(:), tgar(:), ia(:), ja(:), oa(:), oja(:)
    PetscInt :: bs, n, no_
    logical :: has_o
    integer(c_int) :: rc
    integer(c_intptr_t) :: ho
    PetscErrorCode :: ierr
    call split_parts(B, .true., Bd, Bo, bgar)
    call aij_parts(T, Td, To, tgar, has_o)
    call MatGetBlockSize(B, bs, ierr)
    call get_ij(Bd, .true., n, ia, ja)
    call get_ij(Bo, .true., no_, oa, oja)
    ho = 0
    if (has_o) ho = transfer(To%v, 0_c_intptr_t)
    rc = c_sfg_expand_setup(transfer(T%v, 0_c_intptr_t), transfer(Td%v, 0_c_intptr_t), ho, int(bs, c_int), &
                            int(n, c_int), int(ia(1:n + 1), c_int), int(oa(1:no_ + 1), c_int))
    call put_ij(Bo, .true., no_, oa, oja)
    call put_ij(Bd, .true., n, ia, ja)
    if (rc /= 0) stop "sfg_dev_expand_setup: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_expand_setup

  subroutine sfg_dev_expand()
#ifdef USE_GPU_PC
    if (c_sfg_expand() /= 0) stop "sfg_dev_expand: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_expand

  !> SF target k (an aijkokkos matrix T) and its value maps (mod_petsc_pc_sf_gather)
  subroutine sfg_dev_target(k, T, md, mo, wd, wo, zd, zo)
    integer, intent(in) :: k
    Mat, intent(in) :: T
    integer(c_int32_t), intent(in) :: md(:), mo(:)
    integer(c_int32_t), intent(in), optional :: wd(:), wo(:)
    integer(c_int8_t), intent(in), optional :: zd(:), zo(:)
#ifdef USE_GPU_PC
    Mat :: Td, To
    PetscInt, pointer :: garr(:)
    logical :: has_o
    integer(c_int32_t) :: d32(1)
    integer(c_int8_t) :: d8(1)
    integer(c_intptr_t) :: ho
    integer(c_int) :: rc, hw, hz
    call aij_parts(T, Td, To, garr, has_o)
    ho = 0
    if (has_o) ho = transfer(To%v, 0_c_intptr_t)
    d32 = 0; d8 = 0
    hw = merge(1, 0, present(wd)); hz = merge(1, 0, present(zd))
    if (present(wd) .and. present(zd)) then
      rc = c_sfg_target(int(k, c_int), transfer(T%v, 0_c_intptr_t), transfer(Td%v, 0_c_intptr_t), ho, &
                        size(md, kind=c_long_long), md, wd, zd, size(mo, kind=c_long_long), mo, wo, zo, hw, hz)
    else if (present(wd)) then
      rc = c_sfg_target(int(k, c_int), transfer(T%v, 0_c_intptr_t), transfer(Td%v, 0_c_intptr_t), ho, &
                        size(md, kind=c_long_long), md, wd, d8, size(mo, kind=c_long_long), mo, wo, d8, hw, hz)
    else if (present(zd)) then
      rc = c_sfg_target(int(k, c_int), transfer(T%v, 0_c_intptr_t), transfer(Td%v, 0_c_intptr_t), ho, &
                        size(md, kind=c_long_long), md, d32, zd, size(mo, kind=c_long_long), mo, d32, zo, hw, hz)
    else
      rc = c_sfg_target(int(k, c_int), transfer(T%v, 0_c_intptr_t), transfer(Td%v, 0_c_intptr_t), ho, &
                        size(md, kind=c_long_long), md, d32, d8, size(mo, kind=c_long_long), mo, d32, d8, hw, hz)
    endif
    if (rc /= 0) stop "sfg_dev_target: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_target

  subroutine sfg_dev_target_clear()
#ifdef USE_GPU_PC
    if (c_sfg_target_clear() /= 0) stop "sfg_dev_target_clear: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_target_clear

  subroutine sfg_dev_gather(k, opz)
    integer, intent(in) :: k
    real*8, intent(in) :: opz
#ifdef USE_GPU_PC
    if (c_sfg_gather(int(k, c_int), real(opz, c_double)) /= 0) stop "sfg_dev_gather: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_gather

  !> The device gathers' memory, before PetscFinalize finalizes Kokkos
  subroutine sfg_dev_release()
#ifdef USE_GPU_PC
    if (c_sfg_release() /= 0) stop "sfg_dev_release: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_release

  !> Family operator k (mod_petsc_pc_sf_fam) on the device: the source side's
  !! maps (sa; sw / ss when W enters, sc when opz does) and the owner side's
  !! operator T (aijkokkos) with the received values' destinations perm
  !! (>= 0: diagonal part, < 0: -(off-diagonal position) - 1, 0-based)
  subroutine sfg_dev_fam_set(k, sa, sw, ss, sc, T, perm)
    integer, intent(in) :: k
    integer(c_int32_t), intent(in) :: sa(:)
    integer(c_int32_t), intent(in), optional :: sw(:)
    integer(c_int8_t), intent(in), optional :: ss(:), sc(:)
    Mat, intent(in) :: T
    integer(c_long_long), intent(in) :: perm(:)
#ifdef USE_GPU_PC
    Mat :: Td, To
    PetscInt, pointer :: garr(:)
    logical :: has_o
    integer(c_intptr_t) :: ho
    integer(c_int32_t) :: d32(1)
    integer(c_int8_t) :: d8(1)
    integer(c_int) :: rc, hw, hz
    call aij_parts(T, Td, To, garr, has_o)
    ho = 0
    if (has_o) ho = transfer(To%v, 0_c_intptr_t)
    d32 = 0; d8 = 0
    hw = merge(1, 0, present(sw)); hz = merge(1, 0, present(sc))
    ! zero-size arrays pass as they are (never read); absent ones as dummies
    if (present(sw) .and. present(sc)) then
      rc = c_sfg_fam_set(int(k, c_int), size(sa, kind=c_long_long), sa, sw, ss, sc, hw, hz, transfer(T%v, 0_c_intptr_t), &
                         transfer(Td%v, 0_c_intptr_t), ho, size(perm, kind=c_long_long), perm)
    else if (present(sw)) then
      rc = c_sfg_fam_set(int(k, c_int), size(sa, kind=c_long_long), sa, sw, ss, d8, hw, hz, transfer(T%v, 0_c_intptr_t), &
                         transfer(Td%v, 0_c_intptr_t), ho, size(perm, kind=c_long_long), perm)
    else if (present(sc)) then
      rc = c_sfg_fam_set(int(k, c_int), size(sa, kind=c_long_long), sa, d32, d8, sc, hw, hz, transfer(T%v, 0_c_intptr_t), &
                         transfer(Td%v, 0_c_intptr_t), ho, size(perm, kind=c_long_long), perm)
    else
      rc = c_sfg_fam_set(int(k, c_int), size(sa, kind=c_long_long), sa, d32, d8, d8, hw, hz, transfer(T%v, 0_c_intptr_t), &
                         transfer(Td%v, 0_c_intptr_t), ho, size(perm, kind=c_long_long), perm)
    endif
    if (rc /= 0) stop "sfg_dev_fam_set: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_fam_set

  !> Family operator k, source side: the send buffer, gathered on the device
  subroutine sfg_dev_fam_values(k, opz, sbuf)
    integer, intent(in) :: k
    real*8, intent(in) :: opz
    real(c_double), intent(inout) :: sbuf(:)
#ifdef USE_GPU_PC
    if (c_sfg_fam_values(int(k, c_int), real(opz, c_double), sbuf) /= 0) stop "sfg_dev_fam_values: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_fam_values

  !> Family operator k, owner side: the received values into the device operator
  subroutine sfg_dev_fam_install(k, val)
    integer, intent(in) :: k
    real(c_double), intent(in) :: val(:)
#ifdef USE_GPU_PC
    real(c_double) :: v1(1)
    integer(c_int) :: rc
    v1 = 0.d0
    if (size(val) > 0) then
      rc = c_sfg_fam_install(int(k, c_int), val)
    else
      rc = c_sfg_fam_install(int(k, c_int), v1)
    endif
    if (rc /= 0) stop "sfg_dev_fam_install: PETSc/Kokkos error"
#endif
  end subroutine sfg_dev_fam_install

#endif
end module mod_petsc_raw_csr
