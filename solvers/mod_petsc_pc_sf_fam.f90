module mod_petsc_pc_sf_fam
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  !--------------------------------------------------------------------
  !> The SF path on one |n| family per rank (physics_pc_sf_mode_split).
  !!
  !! At physics_pc_sf_harm_couple = 0 every SF block is |n|-diagonal, so the
  !! whole SFM2 sweep falls apart into independent problems, one per |n|
  !! family: the n = 0 slot, then the cos and sin slots of each n. The global
  !! path still solves them together, every block on every rank, one block
  !! after another -- the regime in which the GMG's coarse levels, the
  !! scalar rho / T solves and the inner Krylov reductions stop scaling.
  !!
  !! Here the ranks are split into one sub-communicator per family (as
  !! JOREK's legacy solver and its PETSc mode-split PC do), and each rank
  !! runs the SF solvers on its own family only, concurrently with the other
  !! families: every reduction and halo exchange spans np / n_fam ranks, and
  !! every rank does n_fam times the work per collective.
  !!
  !! What stays global: the operator build (extraction, value-map gathers,
  !! packing), which already scales. What this module adds: after each build,
  !! every operator the solvers and the sweep use is redistributed into a
  !! family copy (one MatCreateSubMatrix per operator into family-major order,
  !! then the rank's rows re-created on the family communicator); per apply,
  !! one scatter of x into the family layout and one back.
  !!
  !! Family layout. Family rank q of family f owns the JOREK indices of a
  !! contiguous chunk of global ranks, field-major as the GMG expects:
  !! row = (field, index, local slot), the local slot fastest. Its slots are
  !! family f's global slots (mod_petsc_pc_harm).
  !--------------------------------------------------------------------

  !> One operator's family copy.
  type, public :: fam_mat_t
    logical :: ready = .false.
    integer :: nf = 0                     !< packed fields per JOREK index
    Mat     :: perm                       !< family-major copy, global comm
    Mat     :: loc                        !< its local rows, sequential
    Mat     :: fam                        !< the family operator, family comm
    PetscInt, allocatable :: keep(:)      !< positions of loc's entries kept in fam
    PetscScalar, allocatable :: val(:)    !< fam's values, in its CSR order
  end type fam_mat_t

  logical, save, public :: sff_on = .false.   !< decomposed and active
  integer, save, public :: sff_comm = MPI_COMM_NULL   !< this rank's family communicator
  integer, save, public :: sff_fam  = 0       !< this rank's family (1-based)
  integer, save, public :: sff_nfam = 0
  PetscInt, save, public :: sff_n1 = 0        !< family per-variable local rows

  integer, save :: gcomm = MPI_COMM_NULL, gme = -1, gnp = 0
  integer, save :: fme = -1, fnp = 0          !< rank / size in the family
  integer, allocatable, save :: fmodes(:)     !< this family's global slots (0-based)
  integer, allocatable, save :: nl(:)         !< JOREK indices per global rank
  PetscInt, allocatable, save :: rsx(:)       !< first full-system row per global rank
  integer, save :: c0 = 0, c1 = 0             !< my chunk of global ranks [c0, c1)
  IS, save :: is_f(4)                         !< family rows of a packed nf-field operator
  logical, save :: is_ok(4) = .false.
  VecScatter, save :: sc6                     !< x <-> stage6
  Vec, save :: stage6                         !< global comm, local 6 n1: variables in turn
  logical, save :: vec_ok = .false.
  integer, save :: ndrop_tot = 0              !< out-of-family entries dropped (first build)

  public :: sff_decompose, sff_mat_refresh, sff_scatter_in, sff_scatter_out, sff_vec_setup, &
            sff_report_layout

contains

  !--------------------------------------------------------------------
  !> Split the ranks into |n| families and set the PC's slot layout. Once.
  !! A_full: the full-system Jacobian (BAIJ, bs = n_var n_tor), whose row
  !! layout fixes every per-variable and packed operator's layout.
  !--------------------------------------------------------------------
  subroutine sff_decompose(A_full, comm, my_id)
    use mod_parameters,    only: n_tor, n_var
    use mod_petsc_pc_harm, only: pc_harm_set
    Mat, intent(in)     :: A_full
    integer, intent(in) :: comm, my_id

    integer :: f, r, mpierr, q, nr_f, rem, first
    integer, allocatable :: fam_of(:), nrk(:)
    PetscInt :: rs, re
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

    !--- ranks per family: even, the remainder to the first families (as the
    !--- legacy solver's autodistribute_ranks); contiguous global ranks
    allocate(nrk(sff_nfam), fam_of(0:gnp - 1))
    nrk = gnp / sff_nfam
    rem = mod(gnp, sff_nfam)
    nrk(1:rem) = nrk(1:rem) + 1
    r = 0
    do f = 1, sff_nfam
      fam_of(r:r + nrk(f) - 1) = f
      r = r + nrk(f)
    enddo
    sff_fam = fam_of(gme)
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

    !--- the global layout: JOREK indices and first row per global rank
    allocate(nl(0:gnp - 1), rsx(0:gnp - 1))
    call MatGetOwnershipRange(A_full, rs, re, ierr)
    call MPI_Allgather(int((re - rs) / (n_var * n_tor)), 1, MPI_INTEGER, nl, 1, MPI_INTEGER, comm, mpierr)
    call MPI_Allgather(rs, 1, MPIU_INTEGER, rsx, 1, MPIU_INTEGER, comm, mpierr)

    !--- my chunk of global ranks: the gnp ranks in fnp contiguous chunks
    nr_f = gnp / fnp
    rem  = mod(gnp, fnp)
    first = 0
    do q = 0, fme - 1
      first = first + nr_f
      if (q < rem) first = first + 1
    enddo
    c0 = first
    c1 = c0 + nr_f
    if (fme < rem) c1 = c1 + 1
    sff_n1 = int(sum(nl(c0:c1 - 1)), kind(sff_n1)) * size(fmodes)

    sff_on = .true.
    call sff_report_layout(my_id, nrk)
  end subroutine sff_decompose

  subroutine sff_report_layout(my_id, nrk)
    integer, intent(in) :: my_id
    integer, intent(in), optional :: nrk(:)
    integer :: f
    if (my_id /= 0 .or. .not. present(nrk)) return
    write(*,'(A,I0,A,I0,A)') "[Physics PC]   SF mode split: ", sff_nfam, " |n| families on ", gnp, " ranks"
    do f = 1, sff_nfam
      if (f == 1) then
        write(*,'(A,I0,A,I0,A)') "[Physics PC]     family ", f, ": ", nrk(f), " rank(s), slot 0 (n = 0)"
      else
        write(*,'(A,I0,A,I0,A,I0,A,I0,A)') "[Physics PC]     family ", f, ": ", nrk(f), " rank(s), slots ", &
          2 * f - 3, ", ", 2 * f - 2, " (|n| group ", f - 1, ")"
      endif
    enddo
  end subroutine sff_report_layout

  !--------------------------------------------------------------------
  !> My family rows of a packed nf-field operator src (global comm), in
  !! family order: field, then index (my chunk of global ranks, in order),
  !! then the family's slots. src's own layout: global rank r holds
  !! [field 0 | ... | field nf-1], each nl(r) indices x n_tor slots.
  !--------------------------------------------------------------------
  subroutine family_rows(nf, src, is)
    use mod_parameters, only: n_tor
    integer, intent(in) :: nf
    Mat, intent(in)     :: src
    IS, intent(out)     :: is
    PetscInt, allocatable :: ps(:), rows(:)
    PetscInt :: rs, re, k
    integer :: r, ff, l, s, mpierr
    PetscErrorCode :: ierr

    allocate(ps(0:gnp - 1))
    call MatGetOwnershipRange(src, rs, re, ierr)
    call MPI_Allgather(rs, 1, MPIU_INTEGER, ps, 1, MPIU_INTEGER, gcomm, mpierr)
    if (re - rs /= int(nf, kind(rs)) * nl(gme) * n_tor) then
      write(*,'(A,I0,A,I0)') "[Physics PC]   FATAL: SF mode split: an operator's rows are not "// &
        "nf x indices x n_tor on rank ", gme, ", nf = ", nf
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    endif
    allocate(rows(int(nf, kind(k)) * sff_n1))
    k = 0
    do ff = 0, nf - 1
      do r = c0, c1 - 1
        do l = 0, nl(r) - 1
          do s = 1, size(fmodes)
            k = k + 1
            rows(k) = ps(r) + (int(ff, kind(k)) * nl(r) + l) * n_tor + fmodes(s)
          enddo
        enddo
      enddo
    enddo
    call ISCreateGeneral(gcomm, k, rows, PETSC_COPY_VALUES, is, ierr)
  end subroutine family_rows

  !--------------------------------------------------------------------
  !> Create (first call) or refresh t%fam, the family copy of src. Collective
  !! on the global communicator. src keeps its nonzero pattern for the run.
  !! Entries outside the family (none at harm_couple = 0, where every block is
  !! |n|-diagonal) are dropped and counted on the first build.
  !--------------------------------------------------------------------
  subroutine sff_mat_refresh(t, src, nf, label, my_id)
    type(fam_mat_t), intent(inout) :: t
    Mat, intent(in)                :: src
    integer, intent(in)            :: nf, my_id
    character(len=*), intent(in)   :: label

    PetscInt, pointer :: ia(:), ja(:)
    PetscScalar, pointer :: a(:)
    PetscInt :: n, nloc, off, k, j, nk
    PetscInt, allocatable :: fi(:), fj(:)
    PetscBool :: done
    PetscErrorCode :: ierr
    integer :: mpierr, ndrop, nd_glob
    PetscInt, parameter :: zero = 0

    if (.not. is_ok(nf)) then
      call family_rows(nf, src, is_f(nf))
      is_ok(nf) = .true.
    endif

    if (.not. t%ready) then
      t%nf = nf
      call MatCreateSubMatrix(src, is_f(nf), is_f(nf), MAT_INITIAL_MATRIX, t%perm, ierr)
      call MatMPIAIJGetLocalMat(t%perm, MAT_INITIAL_MATRIX, t%loc, ierr)
    else
      call MatCreateSubMatrix(src, is_f(nf), is_f(nf), MAT_REUSE_MATRIX, t%perm, ierr)
      call MatMPIAIJGetLocalMat(t%perm, MAT_REUSE_MATRIX, t%loc, ierr)
    endif

    call MatGetRowIJ(t%loc, zero, PETSC_FALSE, PETSC_FALSE, n, ia, ja, done, ierr)
    call MatSeqAIJGetArrayRead(t%loc, a, ierr)

    if (.not. t%ready) then
      !--- the family's columns in perm's numbering: [off, off + family size)
      call ISGetLocalSize(is_f(nf), nloc, ierr)
      call MPI_Exscan(nloc, off, 1, MPIU_INTEGER, MPI_SUM, gcomm, mpierr)
      if (gme == 0) off = 0
      call MPI_Allreduce(MPI_IN_PLACE, off, 1, MPIU_INTEGER, MPI_MIN, sff_comm, mpierr)
      block
        PetscInt :: fsize
        call MPI_Allreduce(nloc, fsize, 1, MPIU_INTEGER, MPI_SUM, sff_comm, mpierr)
        allocate(fi(n + 1), fj(ia(n + 1)), t%keep(ia(n + 1)))
        nk = 0
        fi(1) = 0
        do k = 1, n
          do j = ia(k) + 1, ia(k + 1)
            if (ja(j) >= off .and. ja(j) < off + fsize) then
              nk = nk + 1
              fj(nk) = ja(j) - off
              t%keep(nk) = j
            endif
          enddo
          fi(k + 1) = nk
        enddo
      end block
      ndrop = int(ia(n + 1) - nk)
      allocate(t%val(nk))
      t%val = a(t%keep(1:nk))
      call MatCreateMPIAIJWithArrays(sff_comm, n, n, PETSC_DETERMINE, PETSC_DETERMINE, &
                                     fi, fj(1:nk), t%val, t%fam, ierr)
      call MPI_Allreduce(ndrop, nd_glob, 1, MPI_INTEGER, MPI_SUM, gcomm, mpierr)
      ndrop_tot = ndrop_tot + nd_glob
      if (my_id == 0 .and. nd_glob > 0) write(*,'(A,A,A,I0,A)') "[Physics PC]   SF mode split: ", &
        trim(label), ": ", nd_glob, " entries outside the |n| family dropped"
      deallocate(fi, fj)
      t%ready = .true.
    else
      t%val = a(t%keep(1:size(t%val)))
      call MatUpdateMPIAIJWithArray(t%fam, t%val, ierr)
    endif

    call MatSeqAIJRestoreArrayRead(t%loc, a, ierr)
    call MatRestoreRowIJ(t%loc, zero, PETSC_FALSE, PETSC_FALSE, n, ia, ja, done, ierr)
  end subroutine sff_mat_refresh

  !--------------------------------------------------------------------
  !> The scatter between the full-system vector and the family layout of all
  !! six variables. x_tmpl: a vector of the full system's layout. Once.
  !--------------------------------------------------------------------
  subroutine sff_vec_setup(x_tmpl)
    use mod_parameters, only: n_tor, n_var
    Vec, intent(in) :: x_tmpl
    PetscInt, allocatable :: idx(:)
    PetscInt :: k
    IS :: is6
    integer :: v, r, l, s
    PetscErrorCode :: ierr

    if (vec_ok) return
    allocate(idx(6 * sff_n1))
    k = 0
    do v = 1, 6
      do r = c0, c1 - 1
        do l = 0, nl(r) - 1
          do s = 1, size(fmodes)
            k = k + 1
            ! create_variable_index_sets: node block l of rank r, variable v, slot m
            idx(k) = rsx(r) + int(l, kind(k)) * (n_var * n_tor) + (v - 1) * n_tor + fmodes(s)
          enddo
        enddo
      enddo
    enddo
    call ISCreateGeneral(gcomm, k, idx, PETSC_COPY_VALUES, is6, ierr)
    call VecCreateMPI(gcomm, k, PETSC_DETERMINE, stage6, ierr)
    call VecScatterCreate(x_tmpl, is6, stage6, PETSC_NULL_IS, sc6, ierr)
    call ISDestroy(is6, ierr)
    vec_ok = .true.
  end subroutine sff_vec_setup

  !> v(k) = variable k of x, in the family layout (v on the family comm).
  subroutine sff_scatter_in(x, v)
    Vec :: x
    Vec :: v(6)
    PetscScalar, pointer :: sa(:), va(:)
    PetscErrorCode :: ierr
    integer :: k
    call VecScatterBegin(sc6, x, stage6, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(sc6, x, stage6, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecGetArrayRead(stage6, sa, ierr)
    do k = 1, 6
      call VecGetArray(v(k), va, ierr)
      va = sa((k - 1) * sff_n1 + 1 : k * sff_n1)
      call VecRestoreArray(v(k), va, ierr)
    enddo
    call VecRestoreArrayRead(stage6, sa, ierr)
  end subroutine sff_scatter_in

  !> y = the six family-layout variables v(k), back in the full system.
  !! Every row of y belongs to exactly one family, so y is fully overwritten.
  subroutine sff_scatter_out(v, y)
    Vec :: v(6)
    Vec :: y
    PetscScalar, pointer :: sa(:), va(:)
    PetscErrorCode :: ierr
    integer :: k
    call VecGetArray(stage6, sa, ierr)
    do k = 1, 6
      call VecGetArrayRead(v(k), va, ierr)
      sa((k - 1) * sff_n1 + 1 : k * sff_n1) = va
      call VecRestoreArrayRead(v(k), va, ierr)
    enddo
    call VecRestoreArray(stage6, sa, ierr)
    call VecScatterBegin(sc6, stage6, y, INSERT_VALUES, SCATTER_REVERSE, ierr)
    call VecScatterEnd(sc6, stage6, y, INSERT_VALUES, SCATTER_REVERSE, ierr)
  end subroutine sff_scatter_out

#endif
end module mod_petsc_pc_sf_fam
