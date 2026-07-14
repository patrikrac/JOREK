module mod_petsc_pc_metriplectic_assembly
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_metriplectic_ctx, only: type_metriplectic_ctx, g_mctx
  implicit none
  private

  public :: metriplectic_create_matrices
  public :: metriplectic_assemble
  public :: metriplectic_compose_P_u

contains

  !--------------------------------------------------------------------
  !> Create one 1-var MPIBAIJ matrix with sparsity derived from a_mat
  !! (same logic as mod_petsc_pc_physics_element::petsc_create_pc_matrix,
  !!  n_vars = 1; replicated privately to keep the families independent).
  !--------------------------------------------------------------------
  subroutine create_1v_matrix(petsc_A, a_mat)
    use data_structure,  only: type_SP_MATRIX
    use mod_parameters,  only: n_var

    Mat,                   intent(out) :: petsc_A
    type(type_SP_MATRIX),  intent(in)  :: a_mat

    integer :: i, j
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, col_block
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size    = a_mat%block_size / n_var          ! = n_tor (1-var)
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local       = n_block_local * block_size
    n_global      = a_mat%ng / n_var

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
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: metriplectic create_1v_matrix ierr=", ierr
    ! BC handling uses MatZeroRowsColumns; allow the resulting fill changes.
    call MatSetOption(petsc_A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(petsc_A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    deallocate(d_nnz, o_nnz)
  end subroutine create_1v_matrix


  !> Create the six element-assembled operator matrices (sparsity only).
  subroutine metriplectic_create_matrices(a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_SP_MATRIX), intent(in) :: a_mat

    g_mctx%comm = a_mat%comm
    call create_1v_matrix(g_mctx%M_psi,     a_mat)
    call create_1v_matrix(g_mctx%D_op,      a_mat)
    call create_1v_matrix(g_mctx%Dp_op,     a_mat)
    call create_1v_matrix(g_mctx%Dp_struct, a_mat)
    call create_1v_matrix(g_mctx%L_rho,     a_mat)
    call create_1v_matrix(g_mctx%W_para,    a_mat)
  end subroutine metriplectic_create_matrices


  !--------------------------------------------------------------------
  !> Assemble (or re-assemble) the six operators from element data and
  !! cache the Gears-normalized time factor tau = theta*dt/(1+zeta)
  !! (note, Sec. "Elimination": tau replaces theta*dt everywhere).
  !--------------------------------------------------------------------
  subroutine metriplectic_assemble(my_id, local_elms, n_local_elms, a_mat)
    use construct_metriplectic_matrix_mod, only: construct_metriplectic_matrices
    use data_structure, only: type_SP_MATRIX
    use phys_module,    only: time_evol_theta, time_evol_zeta, tstep

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat

    PetscErrorCode :: ierr

    if (g_mctx%matrices_ready) then
      PetscCallA(MatDestroy(g_mctx%M_psi,     ierr))
      PetscCallA(MatDestroy(g_mctx%D_op,      ierr))
      PetscCallA(MatDestroy(g_mctx%Dp_op,     ierr))
      PetscCallA(MatDestroy(g_mctx%Dp_struct, ierr))
      PetscCallA(MatDestroy(g_mctx%L_rho,     ierr))
      PetscCallA(MatDestroy(g_mctx%W_para,    ierr))
    endif
    call metriplectic_create_matrices(a_mat)

    call construct_metriplectic_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                         g_mctx%M_psi, g_mctx%D_op, g_mctx%Dp_op, &
                                         g_mctx%Dp_struct, g_mctx%L_rho, g_mctx%W_para)

    g_mctx%dt_theta = time_evol_theta * tstep / (1.d0 + time_evol_zeta)

    ! Work vectors (1-var layout), created once from the assembled matrices
    if (.not. g_mctx%work_created) then
      PetscCallA(MatCreateVecs(g_mctx%M_psi, g_mctx%wv_psi_1, g_mctx%wv_psi_2, ierr))
      PetscCallA(MatCreateVecs(g_mctx%L_rho, g_mctx%wv_u_1,   g_mctx%wv_u_2,   ierr))
      g_mctx%work_created = .true.
    endif

    g_mctx%matrices_ready = .true.
    if (my_id == 0) write(*,'(A,ES12.4)') &
      "[Metriplectic] operators assembled; tau = theta*dt/(1+zeta) = ", g_mctx%dt_theta
  end subroutine metriplectic_assemble


  !--------------------------------------------------------------------
  !> Compose P_u = L_rho + tau^2 * W_para without re-assembly.
  !! Fresh copy each call so a dt-sweep can call this repeatedly.
  !! Note: BC rows carry diag (1 + tau^2) — both SPD parts hold unit
  !! Dirichlet diagonals; harmless (decoupled identity rows).
  !--------------------------------------------------------------------
  subroutine metriplectic_compose_P_u(dt_theta_in)
    real*8, intent(in) :: dt_theta_in
    PetscErrorCode :: ierr

    if (g_mctx%P_u_created) then
      PetscCallA(MatDestroy(g_mctx%P_u, ierr))
    endif
    PetscCallA(MatDuplicate(g_mctx%L_rho, MAT_COPY_VALUES, g_mctx%P_u, ierr))
    PetscCallA(MatAXPY(g_mctx%P_u, dt_theta_in**2, g_mctx%W_para, &
                       DIFFERENT_NONZERO_PATTERN, ierr))
    g_mctx%P_u_created = .true.
  end subroutine metriplectic_compose_P_u

#endif
end module mod_petsc_pc_metriplectic_assembly
