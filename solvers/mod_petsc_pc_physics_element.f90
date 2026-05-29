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
  public :: petsc_assemble_pc_diagonal_matrices
  public :: petsc_update_physics_pc_ctx
  public :: petsc_analyze_pc_matrices
  public :: petsc_test_pc_matrices
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
    call petsc_create_pc_matrix(g_ctx%A_j,    a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_w,    a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_jpsi, a_mat, 1)
    call petsc_create_pc_matrix(g_ctx%A_wu,   a_mat, 1)
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
      PetscCallA(MatZeroEntries(g_ctx%A_j,    ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_w,    ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_jpsi, ierr))
      PetscCallA(MatZeroEntries(g_ctx%A_wu,   ierr))
      PetscCallA(MatZeroEntries(g_ctx%K_psi_correction, ierr))
      PetscCallA(MatZeroEntries(g_ctx%K_u_correction, ierr))
      PetscCallA(MatZeroEntries(g_ctx%K_21_correction,  ierr))
      PetscCallA(MatZeroEntries(g_ctx%K_61_correction,  ierr))
      PetscCallA(MatZeroEntries(g_ctx%S_PBP,  ierr))
    endif

    call construct_pc_elliptic_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                        g_ctx%A_j, g_ctx%A_w, g_ctx%A_jpsi, g_ctx%A_wu)
    g_ctx%matrices_ready = .true.

    call construct_schur_correction_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                        g_ctx%K_psi_correction, g_ctx%K_u_correction, &
                                        g_ctx%K_21_correction,  g_ctx%K_61_correction, g_ctx%S_PBP)
    g_ctx%psi_correction_ready  = .true.
    g_ctx%u_correction_ready    = .true.
    g_ctx%correction_21_ready   = .true.
    g_ctx%correction_61_ready   = .true.

    if (first_assembly .and. debug_physics_pc) then
      !call petsc_analyze_pc_matrices(my_id)
      !call petsc_test_pc_matrices(my_id)
    endif
  end subroutine petsc_assemble_pc_matrices


  !> Assemble the four simplified diagonal PC sub-matrices (R_11, R_22, R_55, R_66).
  !! Self-contained: derives ownership from A_full and local elements from global mesh data.
  !! Called from petsc_physics_pc_build_reduced when physics_pc_reassemble = .true.
  subroutine petsc_assemble_pc_diagonal_matrices(A_full, comm, my_id)
    use construct_pc_matrix_mod, only: construct_pc_diagonal_matrices
    use mod_parameters, only: n_var, n_tor, n_degrees, n_vertex_max
    use nodes_elements

    Mat, intent(in) :: A_full
    integer, intent(in) :: comm, my_id

    PetscErrorCode :: ierr
    PetscInt :: rstart, rend, block_size, n_block_local
    PetscInt :: n_local_1v, n_global_1v
    integer :: my_ind_min, my_ind_max
    integer :: ielm, iv, inode, i_order, idx
    logical :: first_assembly

    ! Local element list (computed from mesh + ownership)
    integer, allocatable :: local_elms(:)
    integer :: n_local_elms, n_elements

    first_assembly = .not. g_ctx%reassembled_ready

    ! --- Derive node ownership from PETSc matrix ---
    call MatGetOwnershipRange(A_full, rstart, rend, ierr)
    block_size    = n_var * n_tor
    n_block_local = (rend - rstart) / block_size
    my_ind_min    = rstart / block_size + 1   ! 1-based node index
    my_ind_max    = my_ind_min + n_block_local - 1

    ! --- Compute local element list ---
    n_elements = element_list%n_elements
    allocate(local_elms(n_elements))
    n_local_elms = 0
    element_loop: do ielm = 1, n_elements
      do iv = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(iv)
        do i_order = 1, n_degrees
          idx = node_list%node(inode)%index(i_order)
          if (idx >= my_ind_min .and. idx <= my_ind_max) then
            n_local_elms = n_local_elms + 1
            local_elms(n_local_elms) = ielm
            cycle element_loop
          endif
        enddo
      enddo
    enddo element_loop

    ! --- Destroy old matrices on rebuild (they were converted to AIJ after first assembly,
    !     so we must recreate as BAIJ for MatSetValuesBlocked in the assembly loop) ---
    if (.not. first_assembly) then
      call MatDestroy(g_ctx%R_11, ierr)
      call MatDestroy(g_ctx%R_22, ierr)
      call MatDestroy(g_ctx%R_55, ierr)
      call MatDestroy(g_ctx%R_66, ierr)
    endif

    ! --- Create fresh 1-var BAIJ matrices ---
    n_local_1v  = n_block_local * n_tor
    n_global_1v = PETSC_DETERMINE
    call MatCreate(comm, g_ctx%R_11, ierr)
    call MatSetSizes(g_ctx%R_11, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_11, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_11, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_11, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    call MatCreate(comm, g_ctx%R_22, ierr)
    call MatSetSizes(g_ctx%R_22, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_22, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_22, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_22, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    call MatCreate(comm, g_ctx%R_55, ierr)
    call MatSetSizes(g_ctx%R_55, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_55, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_55, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_55, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    call MatCreate(comm, g_ctx%R_66, ierr)
    call MatSetSizes(g_ctx%R_66, n_local_1v, n_local_1v, n_global_1v, n_global_1v, ierr)
    call MatSetType(g_ctx%R_66, MATMPIBAIJ, ierr)
    call MatSetBlockSize(g_ctx%R_66, n_tor, ierr)
    call MatMPIBAIJSetPreallocation(g_ctx%R_66, n_tor, 20, PETSC_NULL_INTEGER_ARRAY, 20, PETSC_NULL_INTEGER_ARRAY, ierr)

    ! Allow new nonzero entries (conservative pre-allocation may undercount)
    call MatSetOption(g_ctx%R_11, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(g_ctx%R_22, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(g_ctx%R_55, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(g_ctx%R_66, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

    ! --- Element-level assembly ---
    call construct_pc_diagonal_matrices(my_id, local_elms(1:n_local_elms), n_local_elms, &
                                         my_ind_min, my_ind_max, &
                                         g_ctx%R_11, g_ctx%R_22, g_ctx%R_55, g_ctx%R_66)

    deallocate(local_elms)

    ! Convert BAIJ to AIJ for compatibility with extracted sub-blocks in Schur correction
    call MatConvert(g_ctx%R_11, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_11, ierr)
    call MatConvert(g_ctx%R_22, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_22, ierr)
    call MatConvert(g_ctx%R_55, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_55, ierr)
    call MatConvert(g_ctx%R_66, MATMPIAIJ, MAT_INPLACE_MATRIX, g_ctx%R_66, ierr)

    g_ctx%reassembled_ready = .true.

    if (my_id == 0) write(*,'(A)') "[Physics PC]   Diagonal blocks reassembled (R_11, R_22, R_55, R_66)"
  end subroutine petsc_assemble_pc_diagonal_matrices


  !> Refresh the module-level context after a matrix rebuild.
  subroutine petsc_update_physics_pc_ctx()
    ! Reserved for future use
  end subroutine petsc_update_physics_pc_ctx


  !> Print structural info and norms for the elliptic PC sub-matrices.
  subroutine petsc_analyze_pc_matrices(my_id)
    use mod_petsc_matrix_analysis

    integer, intent(in) :: my_id
    PetscReal :: diff_norm
#ifdef USE_SLEPC
    PetscReal :: kappa
#endif

    if (my_id == 0) write(*,'(A)') &
      "=== PC matrix analysis ================================="

    call petsc_mat_print_info(g_ctx%A_j,    "A_j")
    call petsc_mat_print_info(g_ctx%A_w,    "A_w")
    call petsc_mat_print_info(g_ctx%A_jpsi, "A_jpsi")
    call petsc_mat_print_info(g_ctx%A_wu,   "A_wu")

    call petsc_mat_norms(g_ctx%A_j,    "A_j")
    call petsc_mat_norms(g_ctx%A_w,    "A_w")
    call petsc_mat_norms(g_ctx%A_jpsi, "A_jpsi")
    call petsc_mat_norms(g_ctx%A_wu,   "A_wu")

    call petsc_mat_diff_norm(g_ctx%A_j, g_ctx%A_w, "A_j vs A_w", diff_norm)

#ifdef USE_SLEPC
    !call petsc_mat_cond_estimate(g_ctx%A_j, kappa)
    !if (my_id == 0 .and. kappa > 0.0d0) write(*,'(A,ES12.4)') "[PC] cond(A_j) = ", kappa
    !call petsc_mat_cond_estimate(g_ctx%A_w, kappa)
    !if (my_id == 0 .and. kappa > 0.0d0) write(*,'(A,ES12.4)') "[PC] cond(A_w) = ", kappa

    !call petsc_mat_full_spectrum(g_ctx%A_j,    "A_j",    0, symmetric=.true.)
    !call petsc_mat_full_spectrum(g_ctx%A_w,    "A_w",    0, symmetric=.true.)
    !call petsc_mat_full_spectrum(g_ctx%A_jpsi, "A_jpsi", 0, symmetric=.false.)
    !call petsc_mat_full_spectrum(g_ctx%A_wu,   "A_wu",   0, symmetric=.false.)
#endif

    if (my_id == 0) write(*,'(A)') &
      "========================================================"
  end subroutine petsc_analyze_pc_matrices


  !> Run manufactured-solution solver tests on the elliptic PC sub-matrices.
  !!
  !! Computes n_axis_dofs by scanning node_list for axis nodes and finding
  !! the highest block index they own.  Axis nodes sit at the front of the
  !! global DOF ordering; the scalar row boundary is max_axis_block * n_tor.
  !! Fieldsplit tests are added when n_axis_dofs > 0.
  subroutine petsc_test_pc_matrices(my_id)
    use mod_petsc_matrix_tests
    use mod_parameters, only: n_tor, n_degrees
    use nodes_elements

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

    call petsc_run_matrix_tests(my_id, g_ctx%comm, &
                                 g_ctx%A_j, g_ctx%A_w, g_ctx%A_jpsi, g_ctx%A_wu, &
                                 n_axis_dofs)
  end subroutine petsc_test_pc_matrices

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
