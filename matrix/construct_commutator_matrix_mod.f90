module construct_commutator_matrix_mod
!----------------------------------------------------------------
! Element-loop driver for the commutator-preconditioner analysis
! operators S1r (1/R poloidal stiffness) and M3 (conservative rho-form).
! Structure mirrors construct_metriplectic_matrix_mod; the 1-var matrix
! creation is replicated privately to keep the families independent.
!
! BC policy: both are treated as the metriplectic SPD operators are --
! Dirichlet identity rows when eliminate_boundary_dofs is set -- so they
! are boundary-consistent with the extracted A_full blocks they are
! combined with in the defect (note Sec. 8.3).
!----------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none
private

public :: commutator_create_matrices
public :: construct_commutator_matrices

contains

#ifdef USE_PETSC
  !> Create one 1-var MPIBAIJ matrix with sparsity derived from a_mat
  !! (replicated from mod_petsc_pc_metriplectic_assembly::create_1v_matrix).
  subroutine create_1v_matrix(petsc_A, a_mat)
    use data_structure, only: type_SP_MATRIX
    use mod_parameters, only: n_var

    Mat,                  intent(out) :: petsc_A
    type(type_SP_MATRIX), intent(in)  :: a_mat

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
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: commutator create_1v_matrix ierr=", ierr
    call MatSetOption(petsc_A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetOption(petsc_A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE, ierr)
    deallocate(d_nnz, o_nnz)
  end subroutine create_1v_matrix
#endif


  !> Create the two 1-var operator matrices (sparsity only).
  subroutine commutator_create_matrices(a_mat, S1r, M3)
    use data_structure, only: type_SP_MATRIX
    type(type_SP_MATRIX), intent(in) :: a_mat
#ifdef USE_PETSC
    Mat, intent(out) :: S1r, M3
    call create_1v_matrix(S1r, a_mat)
    call create_1v_matrix(M3,  a_mat)
#else
    integer, intent(out) :: S1r, M3
    S1r = 0; M3 = 0
#endif
  end subroutine commutator_create_matrices


  subroutine construct_commutator_matrices(my_id, local_elms, n_local_elms, a_mat, S1r, M3)

    use mod_elt_matrix_commutator
    use mod_parameters,  only: n_tor, n_degrees, n_vertex_max, var_u
    use data_structure,  only: type_SP_MATRIX, type_element, type_node
    use nodes_elements
    use phys_module,     only: eliminate_boundary_dofs
    use construct_pc_matrix_mod, only: apply_dirichlet_bnd
    use omp_lib

    implicit none

    integer,              intent(in) :: my_id
    integer, pointer,     intent(in) :: local_elms(:)
    integer,              intent(in) :: n_local_elms
    type(type_SP_MATRIX), intent(in) :: a_mat
#ifdef USE_PETSC
    Mat, intent(inout) :: S1r, M3
#else
    integer, intent(inout) :: S1r, M3
#endif

    integer :: my_ind_min, my_ind_max

#define D1V_CM (n_tor*n_vertex_max*n_degrees)

    type(type_element), allocatable :: element_thr(:)
    type(type_node),    allocatable :: nodes_thr(:,:)
    integer,            allocatable :: node_out_thr(:,:)

    real*8, allocatable :: ELM_S1r_thr(:,:,:)
    real*8, allocatable :: ELM_M3_thr (:,:,:)
    real*8, allocatable :: buf1v_thr(:,:)

    integer :: nthreads, omp_tid
    integer :: ife, ielm, iv, inode
    integer :: i, i_order, k, k_order
    integer :: index_node1, index_node2
    integer :: j, l, idx_ij, idx_kl
    integer :: bs1
#ifdef USE_PETSC
    PetscErrorCode :: ierr
    PetscInt :: idxm(1), idxn(1)
#endif

    my_ind_min = a_mat%index_min(my_id+1)
    my_ind_max = a_mat%index_max(my_id+1)
    bs1 = n_tor

    if (my_id .eq. 0) then
      write(*,*) '*********************************************'
      write(*,*) '*   construct commutator PC matrices        *'
      write(*,*) '*********************************************'
    endif

    !$omp parallel
    !$omp master
#ifdef _OPENMP
    nthreads = omp_get_num_threads()
#else
    nthreads = 1
#endif
    !$omp end master
    !$omp end parallel

    allocate(element_thr  (nthreads))
    allocate(nodes_thr    (n_vertex_max, nthreads))
    allocate(node_out_thr (n_vertex_max, nthreads))
    allocate(ELM_S1r_thr (D1V_CM, D1V_CM, nthreads))
    allocate(ELM_M3_thr  (D1V_CM, D1V_CM, nthreads))
    allocate(buf1v_thr   (bs1*bs1, nthreads))

    !$omp parallel &
    !$omp   default(shared) &
    !$omp   shared(n_local_elms, local_elms, element_list, node_list, a_mat, my_ind_min, my_ind_max, bs1, &
    !$omp          element_thr, nodes_thr, node_out_thr, ELM_S1r_thr, ELM_M3_thr, buf1v_thr &
#ifdef USE_PETSC
    !$omp          , S1r, M3 &
#endif
    !$omp         ) &
    !$omp   private(ife, ielm, iv, inode, omp_tid, &
    !$omp           i, i_order, k, k_order, index_node1, index_node2, &
    !$omp           j, l, idx_ij, idx_kl &
#ifdef USE_PETSC
    !$omp           ,ierr, idxm, idxn &
#endif
    !$omp          )

#ifdef _OPENMP
    omp_tid = 1 + omp_get_thread_num()
#else
    omp_tid = 1
#endif

    !$omp do schedule(runtime)
    do ife = 1, n_local_elms
      ielm = local_elms(ife)

      element_thr(omp_tid) = element_list%element(ielm)
      do iv = 1, n_vertex_max
        inode = element_thr(omp_tid)%vertex(iv)
        nodes_thr(iv, omp_tid)    = node_list%node(inode)
        node_out_thr(iv, omp_tid) = inode
      enddo

      call element_matrix_commutator(element_thr(omp_tid), nodes_thr(:,omp_tid), &
                                     ELM_S1r_thr(:,:,omp_tid), ELM_M3_thr(:,:,omp_tid))

#ifdef USE_PETSC
      do i = 1, n_vertex_max
        do i_order = 1, n_degrees

          index_node1 = node_list%node(node_out_thr(i,omp_tid))%index(i_order)
          if ((index_node1 .lt. my_ind_min) .or. (index_node1 .gt. my_ind_max)) cycle
          idxm(1) = index_node1 - 1

          do k = 1, n_vertex_max
            do k_order = 1, n_degrees

              index_node2 = node_list%node(node_out_thr(k,omp_tid))%index(k_order)
              idxn(1) = index_node2 - 1

              ! --- S1r ---
              buf1v_thr(:,omp_tid) = 0.d0
              do j = 1, bs1
                idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
                do l = 1, bs1
                  idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                  buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_S1r_thr(idx_ij, idx_kl, omp_tid)
                enddo
              enddo
              !$omp critical
              PetscCallA(MatSetValuesBlocked(S1r, 1, idxm, 1, idxn, &
                                       buf1v_thr(:,omp_tid), ADD_VALUES, ierr))
              !$omp end critical

              ! --- M3 ---
              buf1v_thr(:,omp_tid) = 0.d0
              do j = 1, bs1
                idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
                do l = 1, bs1
                  idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                  buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_M3_thr(idx_ij, idx_kl, omp_tid)
                enddo
              enddo
              !$omp critical
              PetscCallA(MatSetValuesBlocked(M3, 1, idxm, 1, idxn, &
                                       buf1v_thr(:,omp_tid), ADD_VALUES, ierr))
              !$omp end critical

            enddo  ! k_order
          enddo    ! k

        enddo  ! i_order
      enddo    ! i
#endif

    enddo  ! ife
    !$omp end do
    !$omp end parallel

    deallocate(element_thr, nodes_thr, node_out_thr)
    deallocate(ELM_S1r_thr, ELM_M3_thr, buf1v_thr)

#ifdef USE_PETSC
    PetscCallA(MatAssemblyBegin(S1r, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyBegin(M3,  MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd  (S1r, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd  (M3,  MAT_FINAL_ASSEMBLY, ierr))

    ! Dirichlet identity rows on BC dofs (boundary-consistent with A_full).
    if (eliminate_boundary_dofs) then
      call apply_dirichlet_bnd(S1r, var_u, local_elms, n_local_elms, my_ind_min, my_ind_max, symmetric=.true.)
      call apply_dirichlet_bnd(M3,  var_u, local_elms, n_local_elms, my_ind_min, my_ind_max, symmetric=.true.)
    endif
#endif

  end subroutine construct_commutator_matrices

end module construct_commutator_matrix_mod
