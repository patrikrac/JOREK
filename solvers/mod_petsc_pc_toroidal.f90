module mod_petsc_pc_toroidal
#ifdef USE_PETSC
  use mpi_mod
  use mod_petsc_direct_solver, only: petsc_configure_direct_solver
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
    PetscCallA(PCFieldSplitGetSubKSP(pc, n_split, subksp_array, ierr))
    do i = 1, n_split
      ! All blocks share one options prefix rather than PETSc's per-split
      ! fieldsplit_<N>_, so that a single option configures every toroidal mode
      ! family instead of having to be repeated once per split.
      PetscCallA(KSPSetOptionsPrefix(subksp_array(i), PC_BLOCK_PREFIX, ierr))
      PetscCallA(KSPSetType(subksp_array(i), KSPPREONLY, ierr))
      PetscCallA(KSPSetFromOptions(subksp_array(i), ierr))
      PetscCallA(KSPGetPC(subksp_array(i), subpc, ierr))
      call petsc_configure_direct_solver(subpc)
      PetscCallA(KSPSetUp(subksp_array(i), ierr))
      ! Opt-in, inert unless -jorek_dump_mat is set: write this block out so it
      ! can be replayed offline against other factorization packages.
      PetscCallA(KSPGetOperators(subksp_array(i), block_A, PETSC_NULL_MAT, ierr))
      write(dump_tag,'(A,I0)') 'pcblock', i
      call petsc_dump_operator(block_A, trim(dump_tag))
    enddo
    call report_block_solver(subksp_array(1))
    PetscCallA(PCFieldSplitRestoreSubKSP(pc, n_split, subksp_array, ierr))
  end subroutine petsc_setup_toroidal_harmonic_pc


  !> Report the solver configuration the blocks actually resolved to. All blocks
  !! share one options prefix, so reporting the first one describes them all.
  subroutine report_block_solver(subksp)
    KSP, intent(in) :: subksp

    PC             :: subpc
    PCType         :: ptype
    KSPType        :: ktype
    MatSolverType  :: stype
    integer        :: comm, my_id, mpierr
    PetscErrorCode :: ierr

    PetscCallA(PetscObjectGetComm(subksp, comm, ierr))
    call MPI_COMM_RANK(comm, my_id, mpierr)
    if (my_id /= 0) return

    PetscCallA(KSPGetType(subksp, ktype, ierr))
    PetscCallA(KSPGetPC(subksp, subpc, ierr))
    PetscCallA(PCGetType(subpc, ptype, ierr))
    if (ptype == PCLU .or. ptype == PCCHOLESKY .or. ptype == PCILU) then
      PetscCallA(PCFactorGetMatSolverType(subpc, stype, ierr))
      write(*,*) '[PETSc] PC blocks (-'//PC_BLOCK_PREFIX//'...): ' &
                 //trim(ktype)//' + '//trim(ptype)//' via '//trim(stype)
    else
      write(*,*) '[PETSc] PC blocks (-'//PC_BLOCK_PREFIX//'...): ' &
                 //trim(ktype)//' + '//trim(ptype)
    endif
  end subroutine report_block_solver

#endif
end module mod_petsc_pc_toroidal
