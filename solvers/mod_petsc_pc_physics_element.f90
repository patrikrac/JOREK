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


  !> Assemble the physics PC's element-level operators, those the method
  !! needs (physics_pc_elements): W for the SF sweep and for the Alfven
  !! block, with v_par's W terms where the v_par stage needs them, M_rho
  !! (and once D_bc) for the vorticity-space Schur PC, with M_eta and K_eta
  !! for vs_lumped's resistive line bending (vs_wants_eta), none for the
  !! monolithic GMG. They need mhd_sim, the linearisation state, so they are
  !! assembled here rather than in the PC build.
  subroutine petsc_assemble_pc_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim)
    use data_structure,  only: type_SP_MATRIX
    use mod_simulation_data, only: type_MHD_SIM
    use mod_petsc_pc_physics_ctx, only: physics_pc_method_id, PC_METHOD_SF, PC_METHOD_VS_IDEAL, &
                                        PC_METHOD_VS_RESISTIVE, PC_METHOD_VS_LUMPED, PC_METHOD_VS_ALFVEN, &
                                        PC_METHOD_VS_MONO, physics_pc_elements_t, physics_pc_elements
    use mod_petsc_pc_vs, only: vs_init, vs_wants_eta
    use mod_petsc_pc_vs_alfven, only: vsa_init
    use mod_petsc_pc_vs_mono, only: vsm_init

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    type(type_MHD_SIM),   intent(in) :: mhd_sim
    type(physics_pc_elements_t) :: need

    call physics_pc_log_events_register()       ! runs before the first PC build
    ! the methods' refusals of unsupported models come before any assembly
    select case (physics_pc_method_id())
    case (PC_METHOD_SF)
      ! its checks run in sf_build
    case (PC_METHOD_VS_IDEAL, PC_METHOD_VS_RESISTIVE, PC_METHOD_VS_LUMPED)
      call vs_init(my_id)
    case (PC_METHOD_VS_ALFVEN)
      call vsa_init(my_id)
    case (PC_METHOD_VS_MONO)
      call vsm_init(my_id)
    end select
    need = physics_pc_elements()
    if (need%w)     call assemble_w(my_id, local_elms, n_local_elms, a_mat, mhd_sim, need%w_vpar)
    if (need%m_rho) call assemble_vs_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim, vs_wants_eta())
  end subroutine petsc_assemble_pc_matrices


  !> W, the composed force operator, and v_par's W terms where want_vpar_w
  !! and the v_par stage needs them.
  subroutine assemble_w(my_id, local_elms, n_local_elms, a_mat, mhd_sim, want_vpar_w)
    use construct_pc_matrix_mod, only: construct_pc_element_matrices
    use data_structure,  only: type_SP_MATRIX
    use mod_simulation_data, only: type_MHD_SIM
    use mod_petsc_pc_sf_solver, only: sf_vpar_w_assembled

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    type(type_MHD_SIM),   intent(in) :: mhd_sim
    logical,              intent(in) :: want_vpar_w
    PetscErrorCode :: ierr

    call PetscLogEventBegin(pcev_elem_asm, ierr)

    ! created once, zeroed before every refill: W keeps its identity and
    ! pattern for the run, which the SF path relies on
    if (g_ctx%w_force_ready) then
      PetscCallA(MatZeroEntries(g_ctx%W_force, ierr))
    else
      call petsc_create_pc_matrix(g_ctx%W_force, a_mat, 1)
      ! MatZeroRows on the boundary rows must keep their pattern
      PetscCallA(MatSetOption(g_ctx%W_force, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr))
    endif

    if (want_vpar_w .and. sf_vpar_w_assembled()) then
      ! v_par's sound-wave Schur terms, the same lifecycle as W
      if (g_ctx%w_v_ready) then
        PetscCallA(MatZeroEntries(g_ctx%W_vv, ierr))
        PetscCallA(MatZeroEntries(g_ctx%W_vu, ierr))
        PetscCallA(MatZeroEntries(g_ctx%W_uv, ierr))
      else
        call petsc_create_pc_matrix(g_ctx%W_vv, a_mat, 1)
        call petsc_create_pc_matrix(g_ctx%W_vu, a_mat, 1)
        call petsc_create_pc_matrix(g_ctx%W_uv, a_mat, 1)
        PetscCallA(MatSetOption(g_ctx%W_vv, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr))
        PetscCallA(MatSetOption(g_ctx%W_vu, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr))
        PetscCallA(MatSetOption(g_ctx%W_uv, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr))
      endif
      call construct_pc_element_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                         mhd_sim, W_force=g_ctx%W_force, &
                                         W_vv=g_ctx%W_vv, W_vu=g_ctx%W_vu, W_uv=g_ctx%W_uv)
      g_ctx%w_v_ready = .true.
    else
      call construct_pc_element_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                         mhd_sim, W_force=g_ctx%W_force)
    endif
    g_ctx%w_force_ready = .true.
    call PetscLogEventEnd(pcev_elem_asm, ierr)

    if (my_id .eq. 0) then
      if (g_ctx%w_v_ready) then
        write(*,'(A)') "[Physics PC]   W (composed force operator) and v_par's W_vv, W_vu, W_uv assembled"
      else
        write(*,'(A)') "[Physics PC]   W (composed force operator) assembled"
      endif
    endif
  end subroutine assemble_w


  !> The vorticity-space Schur PC's matrices: M_rho on every call, D_bc once.
  !! Their Dirichlet rows at u's constrained DOFs get 1 with eliminated
  !! boundary DOFs; with penalty rows they get the penalty, so that the
  !! operators built on them carry penalty rows where the Jacobian does.
  !! With eta (vs_lumped's resistive line bending) also M_eta and K_eta on
  !! every call, and K_1 on the first (the self-check of the first build), in
  !! the same element pass.
  subroutine assemble_vs_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim, eta)
    use construct_pc_matrix_mod, only: construct_pc_element_matrices, construct_bc_diagonal
    use data_structure,  only: type_SP_MATRIX
    use mod_simulation_data, only: type_MHD_SIM
    use phys_module, only: eliminate_boundary_dofs
    !> the penalty of the Jacobian's boundary rows (mod_boundary_conditions)
    real*8, parameter :: D_PENALTY = 1.d12

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
    type(type_MHD_SIM),   intent(in) :: mhd_sim
    logical,              intent(in) :: eta
    PetscErrorCode :: ierr
    real*8 :: d
    logical :: want_k1

    call PetscLogEventBegin(pcev_elem_asm, ierr)
    d = merge(1.d0, D_PENALTY, eliminate_boundary_dofs)
    if (g_ctx%m_rho_ready) then
      PetscCallA(MatZeroEntries(g_ctx%M_rho, ierr))
    else
      call petsc_create_pc_matrix(g_ctx%M_rho, a_mat, 1)
      PetscCallA(MatSetOption(g_ctx%M_rho, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr))
    endif
    if (eta) then
      ! created once, refilled: their patterns are those of every build
      if (g_ctx%eta_ready) then
        PetscCallA(MatZeroEntries(g_ctx%M_eta, ierr))
        PetscCallA(MatZeroEntries(g_ctx%K_eta, ierr))
      else
        call petsc_create_pc_matrix(g_ctx%M_eta, a_mat, 1)
        call petsc_create_pc_matrix(g_ctx%K_eta, a_mat, 1)
      endif
      want_k1 = .not. (g_ctx%k1_ready .or. g_ctx%k1_done)
      if (want_k1) then
        call petsc_create_pc_matrix(g_ctx%K_1, a_mat, 1)
        call construct_pc_element_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim, &
                                           M_rho=g_ctx%M_rho, d_bc=d, M_eta=g_ctx%M_eta, &
                                           K_eta=g_ctx%K_eta, K_1=g_ctx%K_1)
        g_ctx%k1_ready = .true.
      else
        call construct_pc_element_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim, &
                                           M_rho=g_ctx%M_rho, d_bc=d, M_eta=g_ctx%M_eta, &
                                           K_eta=g_ctx%K_eta)
      endif
      g_ctx%eta_ready = .true.
    else
      call construct_pc_element_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim, &
                                         M_rho=g_ctx%M_rho, d_bc=d)
    endif
    g_ctx%m_rho_ready = .true.
    if (.not. g_ctx%d_bc_ready) then
      call petsc_create_pc_matrix(g_ctx%D_bc, a_mat, 1)
      call construct_bc_diagonal(local_elms, n_local_elms, a_mat, g_ctx%D_bc, d)
      g_ctx%d_bc_ready = .true.
    endif
    call PetscLogEventEnd(pcev_elem_asm, ierr)
    if (my_id == 0) then
      if (eta) then
        write(*,'(A)') "[Physics PC]   M_rho (inertia), M_eta and K_eta (resistive line bending) assembled"
      else
        write(*,'(A)') "[Physics PC]   M_rho (inertia) assembled"
      endif
    endif
  end subroutine assemble_vs_matrices

#endif
end module mod_petsc_pc_physics_element
