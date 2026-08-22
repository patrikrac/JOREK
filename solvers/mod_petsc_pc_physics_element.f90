module mod_petsc_pc_physics_element
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx, &
       physics_pc_log_events_register, pcev_elem_asm, physics_pc_mixed_arm
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
  !!
  !! When physics_pc_reduced_pde is set, also assembles P_full -- the reduced
  !! 4-variable PDE operator of Milestone 1 (docs/physics_pc). P_full needs the
  !! equilibrium state, which is why mhd_sim is threaded down to here.
  subroutine petsc_assemble_pc_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim)
    use construct_pc_matrix_mod
    use data_structure,  only: type_SP_MATRIX
    use mod_simulation_data, only: type_MHD_SIM
    use phys_module,     only: debug_physics_pc, physics_pc_reduced_pde

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    type(type_MHD_SIM),   intent(in) :: mhd_sim
    PetscErrorCode :: ierr
    logical        :: first_assembly


    first_assembly = .not. g_ctx%matrices_ready

    ! Registered here as well as in petsc_physics_pc_build_reduced: this routine
    ! runs FIRST (jorek2_main.f90 assembles before mod_petsc.f90 builds the
    ! reduced system), so relying on the other call site would push an
    ! unregistered event id on the first step. The register call is idempotent.
    call physics_pc_log_events_register()
    call PetscLogEventBegin(pcev_elem_asm, ierr)

    ! The mixed-pair arms read none of what this block produces (see
    ! physics_pc_mixed_arm). Skipping it removes a full element-loop assembly
    ! over the mesh, plus five BAIJ create/destroy pairs, from every Newton
    ! step. The *_ready flags are deliberately left .false. so that any future
    ! reader of Atilde_* on this arm hits the existing "Schur correction block
    ! required!" error rather than silently using an unassembled matrix.
    if (.not. physics_pc_mixed_arm()) then
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
    endif
    ! Ends here, not at the routine's end: P_full below is an optional
    ! diagnostic path (physics_pc_reduced_pde) and folding it in would make the
    ! event mean different things in different configurations.
    call PetscLogEventEnd(pcev_elem_asm, ierr)

    ! --- Milestone 1: the reduced PDE operator P_full ---
    if (physics_pc_reduced_pde) then
      if (.not. g_ctx%p_full_ready) then
        call petsc_create_pc_matrix(g_ctx%P_full_pde, a_mat, 4)
      else
        PetscCallA(MatDestroy(g_ctx%P_full_pde, ierr))
        call petsc_create_pc_matrix(g_ctx%P_full_pde, a_mat, 4)
      endif

      call construct_reduced_pde_matrix(my_id, local_elms, n_local_elms, a_mat, &
                                        mhd_sim, g_ctx%P_full_pde)
      g_ctx%p_full_ready = .true.

      if (my_id .eq. 0) write(*,'(A)') "[Physics PC]   P_full (reduced PDE operator) assembled"
    endif
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
