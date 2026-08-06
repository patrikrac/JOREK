module mod_petsc_pc_physics_element
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  implicit none
  private

  public :: petsc_create_pc_matrices
  public :: petsc_assemble_pc_matrices
  public :: petsc_update_physics_pc_ctx 
  public :: petsc_test_pc_matrix

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


  !> Create the four PC sub-matrices (sparsity allocation only).
  subroutine petsc_create_pc_matrices(a_mat)
    use data_structure,  only: type_SP_MATRIX

    type(type_SP_MATRIX), intent(in) :: a_mat

    g_ctx%comm = a_mat%comm
    !call petsc_create_pc_matrix(g_ctx%A_j,    a_mat, 1)
    !call petsc_create_pc_matrix(g_ctx%A_w,    a_mat, 1)
    !call petsc_create_pc_matrix(g_ctx%A_jpsi, a_mat, 1)
    !call petsc_create_pc_matrix(g_ctx%A_wu,   a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%K_psi_correction, a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%K_u_correction, a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%K_21_correction,  a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%K_61_correction,  a_mat, 1)

    call petsc_create_pc_matrix(g_ctx%S_PBP, a_mat, 1)
  end subroutine petsc_create_pc_matrices


  !> Assemble the four elliptic PC sub-matrices from element-level data.
  subroutine petsc_assemble_pc_matrices(my_id, local_elms, n_local_elms, a_mat)
    use construct_pc_matrix_mod
    use data_structure,  only: type_SP_MATRIX
    use phys_module,     only: debug_physics_pc

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    PetscErrorCode :: ierr
    logical        :: first_assembly


    first_assembly = .not. g_ctx%matrices_ready

    if (first_assembly) then
      call petsc_create_pc_matrices(a_mat)
    else
      ! PetscCallA(MatDestroy(g_ctx%A_j,    ierr))
      ! PetscCallA(MatDestroy(g_ctx%A_w,    ierr))
      ! PetscCallA(MatDestroy(g_ctx%A_jpsi, ierr))
      ! PetscCallA(MatDestroy(g_ctx%A_wu,   ierr))
      PetscCallA(MatDestroy(g_ctx%K_psi_correction, ierr))
      PetscCallA(MatDestroy(g_ctx%K_u_correction, ierr))
      PetscCallA(MatDestroy(g_ctx%K_21_correction,  ierr))
      PetscCallA(MatDestroy(g_ctx%K_61_correction,  ierr))
      PetscCallA(MatDestroy(g_ctx%S_PBP,  ierr))
      call petsc_create_pc_matrices(a_mat)
    endif

    !call construct_pc_elliptic_matrices(my_id, local_elms, n_local_elms, a_mat, &
    !                                    g_ctx%A_j, g_ctx%A_w, g_ctx%A_jpsi, g_ctx%A_wu)

    call construct_schur_correction_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                        g_ctx%K_psi_correction, g_ctx%K_u_correction, &
                                        g_ctx%K_21_correction,  g_ctx%K_61_correction, g_ctx%S_PBP)
    g_ctx%psi_correction_ready  = .true.
    g_ctx%u_correction_ready    = .true.
    g_ctx%correction_21_ready   = .true.
    g_ctx%correction_61_ready   = .true.
    g_ctx%matrices_ready = .true.
  end subroutine petsc_assemble_pc_matrices



  !> Refresh the module-level context after a matrix rebuild.
  subroutine petsc_update_physics_pc_ctx()
    ! Reserved for future use
  end subroutine petsc_update_physics_pc_ctx



    subroutine petsc_test_pc_matrix(A, mat_name, symmetric, my_id)
    use mod_petsc_matrix_tests
    use mod_parameters, only: n_tor, n_degrees
    use nodes_elements

    Mat,     intent(in) :: A
    character(len=*), intent(in) :: mat_name
    logical, intent(in) :: symmetric
    integer, intent(in) :: my_id

    integer :: inode, i_order, mpierr
    integer :: max_axis_blk_local, max_axis_blk_global, n_axis_dofs

    ! Find the largest block index assigned to any axis node on this rank.
    max_axis_blk_local = 0
    do inode = 1, node_list%n_nodes
      if (.not. node_list%node(inode)%axis_node) cycle
      do i_order = 1, n_degrees
        if (node_list%node(inode)%index(i_order) > max_axis_blk_local) &
          max_axis_blk_local = node_list%node(inode)%index(i_order)
      end do
    end do
    call MPI_Allreduce(max_axis_blk_local, max_axis_blk_global, 1, &
                       MPI_INTEGER, MPI_MAX, g_ctx%comm, mpierr)
    ! Each block index corresponds to n_tor scalar rows in A_j
    ! (construct_pc_matrix_mod.f90 uses bs1 = n_tor, assuming n_tor_local = n_tor).
    n_axis_dofs = max_axis_blk_global * n_tor

    call petsc_run_matrix_test(my_id, g_ctx%comm, A, mat_name, symmetric, n_axis_dofs)
  end subroutine petsc_test_pc_matrix

#endif
end module mod_petsc_pc_physics_element
