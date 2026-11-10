module mod_petsc_pc_toroidal
#ifdef USE_PETSC
  use mpi_mod
  use mod_petsc_direct_solver, only: petsc_configure_direct_solver, petsc_report_solver
  use mod_petsc_dump,          only: petsc_dump_operator
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_setup_toroidal_harmonic_pc

  !> Shared options prefix for every PCFIELDSPLIT block, e.g.
  !! -jorek_pcblock_pc_factor_mat_solver_type superlu_dist
  character(len=*), parameter :: PC_BLOCK_PREFIX = 'jorek_pcblock_'

contains

  !> Set up PCFIELDSPLIT preconditioner grouped by toroidal mode families.
  !! A is the BAIJ matrix (used only for MatGetBlockSize); ksp already holds the AIJ operator.
  subroutine petsc_setup_toroidal_harmonic_pc(ksp, A)
    use mod_parameters, only: n_tor
    use phys_module,    only: autodistribute_modes, n_mode_families, &
                              modes_per_family, mode_families_modes

    KSP, intent(inout) :: ksp
    Mat, intent(in)    :: A

    integer :: i, j, k, n_split, split_size, field_size, n_modes_in_fam, idx
    Mat :: block_A
    character(len=32) :: dump_tag
    PetscInt :: block_size
    PetscCount :: field_count
    PetscInt, allocatable :: fields(:)
    integer, allocatable :: fam_modes(:)
    PC :: pc, subpc
    KSP, pointer, dimension(:) :: subksp_array
    integer :: comm, my_id, mpierr
    PetscErrorCode :: ierr

    PetscCallA(PetscObjectGetComm(ksp, comm, ierr))
    PetscCallA(MatGetBlockSize(A, block_size, ierr))
    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCFIELDSPLIT, ierr))
    PetscCallA(PCFieldSplitSetBlockSize(pc, block_size, ierr))
    if (autodistribute_modes) then
      n_split = (n_tor + 1)/2
    else
      n_split = n_mode_families
    endif
    split_size = block_size/n_tor
    do i = 1, n_split
      if (autodistribute_modes) then
        if (i == 1) then
          n_modes_in_fam = 1
          allocate(fam_modes(1))
          fam_modes(1) = 1
        else
          n_modes_in_fam = 2
          allocate(fam_modes(2))
          fam_modes(1) = 2*(i-1)
          fam_modes(2) = 2*(i-1) + 1
        endif
      else
        n_modes_in_fam = modes_per_family(i)
        allocate(fam_modes(n_modes_in_fam))
        fam_modes(1:n_modes_in_fam) = mode_families_modes(i, 1:n_modes_in_fam)
      endif

      field_size = split_size * n_modes_in_fam
      allocate(fields(field_size))
      idx = 0
      do j = 1, split_size
        do k = 1, n_modes_in_fam
          idx = idx + 1
          fields(idx) = (j-1)*n_tor + (fam_modes(k) - 1)
        enddo
      enddo
      field_count = field_size
      PetscCallA(PetscSortInt(field_count, fields, ierr))
      PetscCallA(PCFieldSplitSetFields(pc, PETSC_NULL_CHARACTER, field_size, fields, fields, ierr))
      deallocate(fields, fam_modes)
    enddo

    PetscCallA(PCSetUp(pc, ierr))
    PetscCallA(KSPSetUp(ksp, ierr))
    PetscCallA(PCFieldSplitGetSubKSP(pc, n_split, subksp_array, ierr))
    do i = 1, n_split
      ! All blocks share one options prefix rather than PETSc's per-split
      ! fieldsplit_<N>_, so that a single option configures every toroidal mode
      ! family instead of having to be repeated once per split.
      PetscCallA(KSPSetOptionsPrefix(subksp_array(i), PC_BLOCK_PREFIX, ierr))
      PetscCallA(KSPSetType(subksp_array(i), KSPPREONLY, ierr))
      ! Establish JOREK's default block PC *before* any option is processed.
      ! KSPSetFromOptions below configures whatever PC the sub-KSP currently has,
      ! and an untyped PC falls back to PETSc's own default - PCBJACOBI when the
      ! block is parallel, but PCILU when it is sequential. PCILU is a factor PC,
      ! so it consumes -jorek_pcblock_pc_factor_mat_solver_type and aborts on any
      ! package with no ILU (MUMPS: "does not support factorization type ILU").
      ! petsc_configure_direct_solver sets PCLU too, but it runs afterwards, which
      ! is too late to protect that first pass. Users can still override the type
      ! here or there - PCSetFromOptions is applied after this in both places.
      PetscCallA(KSPGetPC(subksp_array(i), subpc, ierr))
      PetscCallA(PCSetType(subpc, PCLU, ierr))
      PetscCallA(PCFactorSetMatSolverType(subpc, MATSOLVERMUMPS, ierr))
      PetscCallA(KSPSetFromOptions(subksp_array(i), ierr))
      call petsc_configure_direct_solver(subpc)
      PetscCallA(KSPSetUp(subksp_array(i), ierr))
      ! Opt-in, inert unless -jorek_dump_mat is set: write this block out so it
      ! can be replayed offline against other factorization packages.
      PetscCallA(KSPGetOperators(subksp_array(i), block_A, PETSC_NULL_MAT, ierr))
      write(dump_tag,'(A,I0)') 'pcblock', i
      call petsc_dump_operator(block_A, trim(dump_tag))
    enddo
    ! One line describes every block: they share PC_BLOCK_PREFIX. The splits live
    ! on the parent communicator, so its rank 0 is the right one to print from.
    call MPI_COMM_RANK(comm, my_id, mpierr)
    if (my_id == 0) call petsc_report_solver(subksp_array(1), PC_BLOCK_PREFIX)
    PetscCallA(PCFieldSplitRestoreSubKSP(pc, n_split, subksp_array, ierr))
  end subroutine petsc_setup_toroidal_harmonic_pc


#endif
end module mod_petsc_pc_toroidal
