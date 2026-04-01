module construct_pc_matrix_mod
!----------------------------------------------------------------
! Assembly loop for the four elliptic PC sub-matrices.
!
! Reuses the mesh traversal infrastructure of construct_matrix_mod
! but calls the lightweight element_matrix_elliptic from modelPC/
! instead of the full element_matrix_fft, and inserts directly
! into the four provided PETSc BAIJ matrices.
!
! Assumptions:
!   - No mesh refinement (PC is an approximation).
!   - n_tor_local = n_tor  (all toroidal modes on every process).
!   - Boundary conditions are NOT applied here; the PC matrices
!     are used as raw operators inside the PCSHELL.
!----------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none
public :: construct_pc_elliptic_matrices

contains

!> Assemble the four elliptic PC sub-matrices.
!!
!! @param my_id      MPI rank of this process (0-based)
!! @param local_elms Pointer to the list of local element indices
!! @param n_local_elms Number of local elements
!! @param a_mat      Sparse matrix descriptor (ownership ranges + node indices)
!! @param A_j        1-var PETSc matrix for j equation (mass)
!! @param A_w        1-var PETSc matrix for w equation (= A_j by symmetry)
!! @param A_jpsi     1-var PETSc matrix for psi->j off-diagonal coupling
!! @param A_wu       1-var PETSc matrix for u->w off-diagonal coupling
subroutine construct_pc_elliptic_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                          A_j, A_w, A_jpsi, A_wu)

  use mod_elt_matrix_elliptic
  use mod_parameters, only: n_tor, n_degrees, n_vertex_max
  use data_structure,  only: type_SP_MATRIX, type_element, type_node
  use nodes_elements
  use omp_lib

  implicit none

  integer,              intent(in) :: my_id
  integer, pointer,     intent(in) :: local_elms(:)
  integer,              intent(in) :: n_local_elms
  type(type_SP_MATRIX), intent(in) :: a_mat
#ifdef USE_PETSC
  Mat, intent(inout) :: A_j, A_w, A_jpsi, A_wu
#endif

  ! --- Ownership range for this process ---
  integer :: my_ind_min, my_ind_max

#define D1V_CM (n_tor*n_vertex_max*n_degrees)

  ! --- Thread-local variables (OMP private) ---
  type(type_element), allocatable :: element_thr(:)      ! one per OMP thread
  type(type_node),    allocatable :: nodes_thr(:,:)      ! (n_vertex_max, n_threads)
  integer,            allocatable :: node_out_thr(:,:)   ! (n_vertex_max, n_threads)

  ! Thread-private ELM arrays (allocated once per thread)
  real*8, allocatable :: ELM_j_thr   (:,:,:)
  real*8, allocatable :: ELM_w_thr   (:,:,:)
  real*8, allocatable :: ELM_jpsi_thr(:,:,:)
  real*8, allocatable :: ELM_wu_thr  (:,:,:)
  real*8, allocatable :: buf1v_thr(:,:)

  integer :: nthreads, omp_tid
  integer :: ife, ielm, iv, inode
  integer :: i, i_order, k, k_order
  integer :: index_node1, index_node2
  integer :: j, l, idx_ij, idx_kl
  integer :: bs1
#ifdef USE_PETSC
  PetscErrorCode :: petsc_ierr
  PetscInt :: idxm(1), idxn(1)
#endif

  my_ind_min = a_mat%index_min(my_id+1)
  my_ind_max = a_mat%index_max(my_id+1)
  bs1 = n_tor

  ! --- Determine thread count and pre-allocate per-thread buffers ---
  !$omp parallel
  !$omp master
#ifdef _OPENMP
  nthreads = omp_get_num_threads()
#else
  nthreads = 1
#endif
  !$omp end master
  !$omp end parallel

  allocate(element_thr   (nthreads))
  allocate(nodes_thr     (n_vertex_max, nthreads))
  allocate(node_out_thr  (n_vertex_max, nthreads))
  allocate(ELM_j_thr    (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_w_thr    (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_jpsi_thr (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_wu_thr   (D1V_CM, D1V_CM, nthreads))
  allocate(buf1v_thr    (bs1*bs1, nthreads))

  !$omp parallel &
  !$omp   default(none) &
  !$omp   shared(n_local_elms, local_elms, a_mat, my_ind_min, my_ind_max, bs1, &
  !$omp          element_thr, nodes_thr, node_out_thr, &
  !$omp          ELM_j_thr, ELM_w_thr, ELM_jpsi_thr, ELM_wu_thr, &
  !$omp          buf1v_thr &
#ifdef USE_PETSC
  !$omp          ,A_j, A_w, A_jpsi, A_wu &
#endif
  !$omp         ) &
  !$omp   private(ife, ielm, iv, inode, omp_tid, &
  !$omp           i, i_order, k, k_order, &
  !$omp           index_node1, index_node2, &
  !$omp           j, l, idx_ij, idx_kl &
#ifdef USE_PETSC
  !$omp           ,petsc_ierr, idxm, idxn &
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

    ! --- Fetch element and nodes (no refinement support needed for PC) ---
    element_thr(omp_tid) = element_list%element(ielm)
    do iv = 1, n_vertex_max
      inode = element_thr(omp_tid)%vertex(iv)
      nodes_thr(iv, omp_tid)   = node_list%node(inode)
      node_out_thr(iv, omp_tid) = inode
    enddo

    ! --- Compute element matrices ---
    call element_matrix_elliptic(element_thr(omp_tid), nodes_thr(:,omp_tid), &
                                  ELM_j_thr(:,:,omp_tid),    ELM_w_thr(:,:,omp_tid), &
                                  ELM_jpsi_thr(:,:,omp_tid), ELM_wu_thr(:,:,omp_tid))

#ifdef USE_PETSC
    ! --- Insert element blocks into PETSc matrices (all four are 1-var) ---
    do i = 1, n_vertex_max
      do i_order = 1, n_degrees

        index_node1 = node_list%node(node_out_thr(i,omp_tid))%index(i_order)

        ! Only assemble rows owned by this process
        if ((index_node1 .lt. my_ind_min) .or. (index_node1 .gt. my_ind_max)) cycle

        idxm(1) = index_node1 - 1   ! 0-based block row

        do k = 1, n_vertex_max
          do k_order = 1, n_degrees

            index_node2 = node_list%node(node_out_thr(k,omp_tid))%index(k_order)
            idxn(1) = index_node2 - 1   ! 0-based block col

            ! --- Extract and insert 1-var block for A_j ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_j_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(A_j, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for A_w ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_w_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(A_w, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for A_jpsi (psi->j coupling) ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_jpsi_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(A_jpsi, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for A_wu (u->w coupling) ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_wu_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(A_wu, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
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
  deallocate(ELM_j_thr, ELM_w_thr, ELM_jpsi_thr, ELM_wu_thr)
  deallocate(buf1v_thr)

#ifdef USE_PETSC
  ! Interleave Begin/End to allow MPI communication to overlap across matrices
  call MatAssemblyBegin(A_j,    MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(A_w,    MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(A_jpsi, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(A_wu,   MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_j,    MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_w,    MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_jpsi, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_wu,   MAT_FINAL_ASSEMBLY, petsc_ierr)
#endif

end subroutine construct_pc_elliptic_matrices

end module construct_pc_matrix_mod
