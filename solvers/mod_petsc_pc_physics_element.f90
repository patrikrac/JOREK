module mod_petsc_pc_physics_element
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, physics_pc_log_events_register, pcev_elem_asm
  implicit none
  private

  public :: petsc_assemble_pc_matrices

contains

  !--------------------------------------------------------------------
  !> Create an n_vars-variable MPIBAIJ PC matrix (sparsity only).
  !--------------------------------------------------------------------
  subroutine petsc_create_pc_matrix(petsc_A, a_mat, n_vars)
    use data_structure,  only: type_SP_MATRIX
    use mod_parameters,  only: n_var

    Mat,                   intent(out) :: petsc_A
    type(type_SP_MATRIX),  intent(in)  :: a_mat
    integer,               intent(in)  :: n_vars

    integer :: i, j
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, col_block
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size    = n_vars * a_mat%block_size / n_var
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local       = n_block_local * block_size
    n_global      = n_vars * a_mat%ng / n_var

    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0
    o_nnz = 0
    do i = 1, n_block_local
      do j = 1, a_mat%ijA_size(i)
        col_block = a_mat%irn_jcn(i, j)
        if (col_block >= a_mat%my_ind_min .and. col_block <= a_mat%my_ind_max) then
          d_nnz(i) = d_nnz(i) + 1
        else
          o_nnz(i) = o_nnz(i) + 1
        endif
      enddo
    enddo

    call MatCreate(comm, petsc_A, ierr)
    call MatSetSizes(petsc_A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_A, block_size, ierr)
    call MatMPIBAIJSetPreallocation(petsc_A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: petsc_create_pc_matrix ierr=", ierr
    deallocate(d_nnz, o_nnz)

    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0,A,I0)') &
      "[PETSc] create_pc_matrix (", n_vars, "-var): BAIJ ", n_global, "x", n_global, &
      ", block_size=", block_size
  end subroutine petsc_create_pc_matrix


  !> Assemble the element-level operators of the physics PC: the composed
  !! force operator W (models/model*/mod_pc_elt_matrix_force_fft.f90). It
  !! needs mhd_sim (the linearisation state), which is why it is assembled
  !! here and not in the PC build.
  subroutine petsc_assemble_pc_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim)
    use construct_pc_matrix_mod, only: construct_force_operator_matrix
    use data_structure,  only: type_SP_MATRIX
    use mod_simulation_data, only: type_MHD_SIM
    use mod_petsc_pc_sf_solver, only: sf_force_terms

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    type(type_MHD_SIM),   intent(in) :: mhd_sim
    PetscErrorCode :: ierr

    ! This routine runs before the PC build on the first step, so the events
    ! are registered here too (idempotent).
    call physics_pc_log_events_register()
    call PetscLogEventBegin(pcev_elem_asm, ierr)

    ! Created once and zeroed before every refill: the element routine only
    ! ADDs, so this is the same operator as a fresh matrix, and W keeps its
    ! identity and pattern for the run -- which the SF path's value maps rely on.
    if (g_ctx%w_force_ready) then
      PetscCallA(MatZeroEntries(g_ctx%W_force, ierr))
    else
      call petsc_create_pc_matrix(g_ctx%W_force, a_mat, 1)
      ! The boundary rows are zeroed by MatZeroRows after every assembly;
      ! without this it also DELETES their pattern, so the next assembly
      ! would insert outside the (compressed) structure.
      PetscCallA(MatSetOption(g_ctx%W_force, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr))
    endif

    ! the mixed pair_w forms carry the bending term (or the whole psi
    ! channel) through explicit fields, and take it out of W here
    call construct_force_operator_matrix(my_id, local_elms, n_local_elms, a_mat, &
                                         mhd_sim, g_ctx%W_force, terms=sf_force_terms())
    g_ctx%w_force_ready = .true.
    call PetscLogEventEnd(pcev_elem_asm, ierr)

    if (my_id .eq. 0) write(*,'(A)') &
      "[Physics PC]   W (composed force operator) assembled"
  end subroutine petsc_assemble_pc_matrices



#endif
end module mod_petsc_pc_physics_element
