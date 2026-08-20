module mod_petsc_pc_toroidal
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_setup_toroidal_harmonic_pc

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
    PetscInt :: block_size
    PetscCount :: field_count
    PetscInt, allocatable :: fields(:)
    integer, allocatable :: fam_modes(:)
    Mat :: F
    PC :: pc, subpc
    KSP, pointer, dimension(:) :: subksp_array
    PetscErrorCode :: ierr

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
    allocate(subksp_array(n_split))
    PetscCallA(PCFieldSplitGetSubKSP(pc, n_split, subksp_array, ierr))
    do i = 1, n_split
      PetscCallA(KSPSetType(subksp_array(i), KSPPREONLY, ierr))
      PetscCallA(KSPGetPC(subksp_array(i), subpc, ierr))
      PetscCallA(PCSetType(subpc, PCLU, ierr))
      PetscCallA(PCFactorSetMatSolverType(subpc, MATSOLVERMUMPS, ierr))
      PetscCallA(KSPSetUp(subksp_array(i), ierr))
      PetscCallA(PCFactorGetMatrix(subpc, F, ierr))
      PetscCallA(MatMumpsSetIcntl(F, 7,  7,  ierr))   ! fill-reducing ordering (METIS)
      PetscCallA(MatMumpsSetIcntl(F, 14, 50, ierr))   ! workspace expansion %
      PetscCallA(MatMumpsSetIcntl(F, 8,  77, ierr))   ! numerical scaling (auto)
      PetscCallA(MatMumpsSetIcntl(F, 21, 1,  ierr))   ! out-of-core processing
    enddo
    deallocate(subksp_array)
  end subroutine petsc_setup_toroidal_harmonic_pc

end module mod_petsc_pc_toroidal
