module mod_petsc_pc_mass_slot
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: physics_pc_mumps_mem
  implicit none
  private

  !--------------------------------------------------------------------
  !> A constraint mass B (B_33 or B_44) factored once per DISTINCT toroidal
  !! slot (workstream D1).
  !!
  !! JOREK's mass matrices are exactly slot-block-diagonal (the stored
  !! cross-slot entries are 0, even between cos and sin of one n), and the
  !! cos/sin slots of every n >= 1 carry the SAME matrix (n = 0 differs: 2x on
  !! the interior, not on the boundary ring). So one factor per distinct slot
  !! matrix serves every slot, and a group of k identical slots is solved as
  !! one k-column dense RHS, reading the factor once. Factor memory: 2 slot
  !! factors for any n_tor.
  !!
  !! Both structural facts are GATED, not assumed: a mass with cross-slot
  !! content aborts (not a fallback -- silently factoring a different operator
  !! would be measuring the wrong thing), and slot identity is MatEqual (exact).
  !! Rows are index-major (local entry i*n_tor + m is row i of slot m), the
  !! order extract_sub_blocks_h keeps.
  !--------------------------------------------------------------------
  type, public :: mass_slot_t
    logical :: ready = .false.
    integer :: ngrp = 0
    integer, allocatable :: gn(:)        !< slots in group g
    integer, allocatable :: gm(:,:)      !< gm(g, c) = c-th slot (0-based) of group g
    KSP, allocatable     :: ksp(:)       !< factor of group g's slot matrix
    Mat, allocatable     :: rhs(:), sol(:)
    PetscInt :: nloc_s = 0
  end type mass_slot_t

  public :: mass_slot_setup, mass_slot_solve

contains

  !> Split B into its slot matrices, gate the structure, and factor each
  !! distinct slot matrix once (Cholesky when symmetric). Collective on comm.
  subroutine mass_slot_setup(MS, B, comm, label)
    use mod_parameters, only: n_tor
    type(mass_slot_t), intent(inout) :: MS
    Mat, intent(in)              :: B
    integer, intent(in)          :: comm
    character(len=*), intent(in) :: label
    Mat, allocatable :: S(:)
    IS             :: ism
    PC             :: pc
    PetscErrorCode :: ierr
    PetscInt       :: rstart, rend, nloc, nglo, i, ncols
    PetscInt, pointer    :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscBool      :: eq, symm
    real*8         :: fB, fs, cross
    integer        :: m, g, k, rank
    character(len=160) :: line

    call MPI_Comm_rank(comm, rank, ierr)
    call MatGetOwnershipRange(B, rstart, rend, ierr)
    call MatGetSize(B, nglo, PETSC_NULL_INTEGER, ierr)
    nloc = rend - rstart
    if (mod(rstart, n_tor) /= 0 .or. mod(nloc, n_tor) /= 0) &
      call fatal("the per-slot mass needs slot-aligned row ownership.")
    MS%nloc_s = nloc / n_tor

    !--- cross-slot gate, summed entry by entry (a difference of Frobenius
    !--- norms would cancel to ~sqrt(eps) and could not tell 0 from round-off)
    fs = 0.d0
    do i = rstart, rend - 1
      call MatGetRow(B, i, ncols, cols, vals, ierr)
      do k = 1, ncols
        if (mod(cols(k), n_tor) /= mod(i, n_tor)) fs = fs + abs(vals(k))**2
      enddo
      call MatRestoreRow(B, i, ncols, cols, vals, ierr)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, fs, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
    call MatNorm(B, NORM_FROBENIUS, fB, ierr)
    cross = sqrt(fs) / fB
    if (cross > 1.d-13) then
      write(line, '(A,ES10.3,A)') "the mass has cross-slot content (||cross||/||B|| = ", cross, ")."
      call fatal(trim(line))
    endif

    !--- slot matrices
    allocate(S(0:n_tor - 1))
    do m = 0, n_tor - 1
      call ISCreateStride(comm, MS%nloc_s, rstart + m, int(n_tor, kind=kind(rstart)), ism, ierr)
      call MatCreateSubMatrix(B, ism, ism, MAT_INITIAL_MATRIX, S(m), ierr)
      call ISDestroy(ism, ierr)
    enddo

    !--- group identical slots (exact MatEqual against each group's first slot)
    MS%ngrp = 0
    allocate(MS%gn(n_tor), MS%gm(n_tor, n_tor))
    MS%gn = 0; MS%gm = -1
    do m = 0, n_tor - 1
      do g = 1, MS%ngrp
        call MatEqual(S(m), S(MS%gm(g, 1)), eq, ierr)
        if (eq) exit
      enddo
      if (g > MS%ngrp) then
        MS%ngrp = MS%ngrp + 1; g = MS%ngrp
      endif
      MS%gn(g) = MS%gn(g) + 1
      MS%gm(g, MS%gn(g)) = m
    enddo

    !--- one factor per group
    allocate(MS%ksp(MS%ngrp), MS%rhs(MS%ngrp), MS%sol(MS%ngrp))
    do g = 1, MS%ngrp
      call KSPCreate(comm, MS%ksp(g), ierr)
      call KSPSetOperators(MS%ksp(g), S(MS%gm(g, 1)), S(MS%gm(g, 1)), ierr)
      call KSPSetType(MS%ksp(g), KSPPREONLY, ierr)
      call KSPGetPC(MS%ksp(g), pc, ierr)
      call MatIsSymmetric(S(MS%gm(g, 1)), 1.d-12, symm, ierr)
      if (symm) then
        call MatSetOption(S(MS%gm(g, 1)), MAT_SPD, PETSC_TRUE, ierr)
        call PCSetType(pc, PCCHOLESKY, ierr)
      else
        call PCSetType(pc, PCLU, ierr)
      endif
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
      call KSPSetUp(MS%ksp(g), ierr)
      call MatCreateDense(comm, MS%nloc_s, PETSC_DECIDE, nglo / n_tor, int(MS%gn(g), kind=kind(nglo)), &
                          PETSC_NULL_SCALAR_ARRAY, MS%rhs(g), ierr)
      call MatDuplicate(MS%rhs(g), MAT_DO_NOT_COPY_VALUES, MS%sol(g), ierr)
      write(line,'(A,I0,A,I0,A)') "slot group ", g, " (", MS%gn(g), " slots)"
      call physics_pc_mumps_mem(MS%ksp(g), trim(label)//" "//trim(line), rank)
    enddo
    do m = 0, n_tor - 1
      call MatDestroy(S(m), ierr)       ! each group KSP holds its own reference
    enddo
    MS%ready = .true.

    if (rank == 0) write(*,'(A,A,A,I0,A,I0,A,ES9.2,A)') "[Physics PC]   ", trim(label), &
      ": PREONLY + per-slot MUMPS factors, ", MS%ngrp, " distinct of ", n_tor, &
      " slots (cross-slot ||.||/||B|| = ", cross, "), factored once"

  contains

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (rank == 0) write(*,'(A,A,A,A)') "[Physics PC]   FATAL: ", trim(label), ": ", trim(msg)
      flush(6)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine mass_slot_setup

  !> y = B^-1 x slot by slot: gather each group's slots into its dense RHS,
  !! one multi-RHS solve per group, scatter back. Rank-local apart from the
  !! factor's own solve.
  subroutine mass_slot_solve(MS, x, y)
    use mod_parameters, only: n_tor
    type(mass_slot_t), intent(inout) :: MS
    Vec :: x, y
    PetscScalar, pointer :: xa(:), ya(:), a2(:,:)
    PetscErrorCode :: ierr
    integer :: g, c, m
    PetscInt :: i

    call VecGetArrayRead(x, xa, ierr)
    call VecGetArray(y, ya, ierr)
    do g = 1, MS%ngrp
      call MatDenseGetArray(MS%rhs(g), a2, ierr)
      do c = 1, MS%gn(g)
        m = MS%gm(g, c)
        do i = 1, MS%nloc_s
          a2(i, c) = xa((i - 1) * n_tor + m + 1)
        enddo
      enddo
      call MatDenseRestoreArray(MS%rhs(g), a2, ierr)
      call KSPMatSolve(MS%ksp(g), MS%rhs(g), MS%sol(g), ierr)
      call MatDenseGetArrayRead(MS%sol(g), a2, ierr)
      do c = 1, MS%gn(g)
        m = MS%gm(g, c)
        do i = 1, MS%nloc_s
          ya((i - 1) * n_tor + m + 1) = a2(i, c)
        enddo
      enddo
      call MatDenseRestoreArrayRead(MS%sol(g), a2, ierr)
    enddo
    call VecRestoreArrayRead(x, xa, ierr)
    call VecRestoreArray(y, ya, ierr)
  end subroutine mass_slot_solve

#endif
end module mod_petsc_pc_mass_slot
