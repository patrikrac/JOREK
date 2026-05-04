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
!   - Boundary conditions are applied via ZBIG penalty method
!     (matching model199) to diagonal and elliptic constraint blocks.
!----------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none
public :: construct_pc_elliptic_matrices, construct_pc_diagonal_matrices, apply_bc_pc_matrix
public :: construct_schur_correction_matrices, zero_bc_rows_pc_matrix

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

  if (my_id .eq. 0) then
      write(*,*) '****************************************'
      write(*,*) '*        construct PC matrices         *'
      write(*,*) '****************************************'
  endif

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
  !$omp   shared(n_local_elms, local_elms, element_list, node_list, a_mat, my_ind_min, my_ind_max, bs1, &
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
  ! Flush element contributions (ADD_VALUES) before BCs (INSERT_VALUES)
  call MatAssemblyBegin(A_j,    MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(A_w,    MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(A_jpsi, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(A_wu,   MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_j,    MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_w,    MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_jpsi, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (A_wu,   MAT_FLUSH_ASSEMBLY, petsc_ierr)

  ! Apply BCs to mass matrices only (not off-diagonal couplings A_jpsi, A_wu)
  call apply_bc_pc_matrix(A_j, 3, local_elms, n_local_elms, my_ind_min, my_ind_max)  ! j
  call apply_bc_pc_matrix(A_w, 4, local_elms, n_local_elms, my_ind_min, my_ind_max)  ! w

  ! Final assembly
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


!> Assemble the four simplified diagonal PC sub-matrices.
!!
!! These are simplified versions of B_11, B_22, B_55, B_66 that
!! capture only the dominant physics (mass, stiffness, perp diffusion).
!! They replace the extracted sub-blocks when physics_pc_reassemble = .true.
!!
!! @param my_id      MPI rank of this process (0-based)
!! @param local_elms Pointer to the list of local element indices
!! @param n_local_elms Number of local elements
!! @param a_mat      Sparse matrix descriptor (ownership ranges + node indices)
!! @param R_11       1-var PETSc matrix for psi equation diagonal (mass/R)
!! @param R_22       1-var PETSc matrix for u equation diagonal (R^3 stiffness)
!! @param R_55       1-var PETSc matrix for rho equation diagonal (mass + perp diff)
!! @param R_66       1-var PETSc matrix for T equation diagonal (mass + perp cond)
subroutine construct_pc_diagonal_matrices(my_id, local_elms, n_local_elms, &
                                          my_ind_min_in, my_ind_max_in, &
                                          R_11, R_22, R_55, R_66)

  use mod_elt_matrix_pc_diag
  use mod_parameters, only: n_tor, n_degrees, n_vertex_max
  use data_structure,  only: type_element, type_node
  use nodes_elements
  use omp_lib

  implicit none

  integer,              intent(in) :: my_id
  integer,              intent(in) :: local_elms(:)
  integer,              intent(in) :: n_local_elms
  integer,              intent(in) :: my_ind_min_in, my_ind_max_in
#ifdef USE_PETSC
  Mat, intent(inout) :: R_11, R_22, R_55, R_66
#endif

  ! --- Ownership range for this process ---
  integer :: my_ind_min, my_ind_max

#define D1V_CD (n_tor*n_vertex_max*n_degrees)

  ! --- Thread-local variables (OMP private) ---
  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  integer,            allocatable :: node_out_thr(:,:)

  ! Thread-private ELM arrays
  real*8, allocatable :: ELM_11_thr(:,:,:)
  real*8, allocatable :: ELM_22_thr(:,:,:)
  real*8, allocatable :: ELM_55_thr(:,:,:)
  real*8, allocatable :: ELM_66_thr(:,:,:)
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

  my_ind_min = my_ind_min_in
  my_ind_max = my_ind_max_in
  bs1 = n_tor

  if (my_id .eq. 0) then
      write(*,*) '****************************************'
      write(*,*) '*   construct PC diagonal matrices     *'
      write(*,*) '****************************************'
  endif

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
  allocate(ELM_11_thr   (D1V_CD, D1V_CD, nthreads))
  allocate(ELM_22_thr   (D1V_CD, D1V_CD, nthreads))
  allocate(ELM_55_thr   (D1V_CD, D1V_CD, nthreads))
  allocate(ELM_66_thr   (D1V_CD, D1V_CD, nthreads))
  allocate(buf1v_thr    (bs1*bs1, nthreads))

  !$omp parallel &
  !$omp   default(none) &
  !$omp   shared(n_local_elms, local_elms, element_list, node_list, my_ind_min, my_ind_max, bs1, &
  !$omp          element_thr, nodes_thr, node_out_thr, &
  !$omp          ELM_11_thr, ELM_22_thr, ELM_55_thr, ELM_66_thr, &
  !$omp          buf1v_thr &
#ifdef USE_PETSC
  !$omp          ,R_11, R_22, R_55, R_66 &
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

    ! --- Fetch element and nodes ---
    element_thr(omp_tid) = element_list%element(ielm)
    do iv = 1, n_vertex_max
      inode = element_thr(omp_tid)%vertex(iv)
      nodes_thr(iv, omp_tid)   = node_list%node(inode)
      node_out_thr(iv, omp_tid) = inode
    enddo

    ! --- Compute element matrices ---
    call element_matrix_pc_diag(element_thr(omp_tid), nodes_thr(:,omp_tid), &
                                 ELM_11_thr(:,:,omp_tid), ELM_22_thr(:,:,omp_tid), &
                                 ELM_55_thr(:,:,omp_tid), ELM_66_thr(:,:,omp_tid))

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

            ! --- Extract and insert 1-var block for R_11 ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_11_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(R_11, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for R_22 ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_22_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(R_22, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for R_55 ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_55_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(R_55, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for R_66 ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_66_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(R_66, 1, idxm, 1, idxn, &
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
  deallocate(ELM_11_thr, ELM_22_thr, ELM_55_thr, ELM_66_thr)
  deallocate(buf1v_thr)

#ifdef USE_PETSC
  ! Flush element contributions (ADD_VALUES) before BCs (INSERT_VALUES)
  call MatAssemblyBegin(R_11, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(R_22, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(R_55, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(R_66, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_11, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_22, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_55, MAT_FLUSH_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_66, MAT_FLUSH_ASSEMBLY, petsc_ierr)

  ! Apply model199-style boundary conditions (ZBIG penalty on diagonal)
  call apply_bc_pc_matrix(R_11, 1, local_elms, n_local_elms, my_ind_min, my_ind_max)  ! psi
  call apply_bc_pc_matrix(R_22, 2, local_elms, n_local_elms, my_ind_min, my_ind_max)  ! u
  call apply_bc_pc_matrix(R_55, 5, local_elms, n_local_elms, my_ind_min, my_ind_max)  ! rho
  call apply_bc_pc_matrix(R_66, 6, local_elms, n_local_elms, my_ind_min, my_ind_max)  ! T

  ! Final assembly
  call MatAssemblyBegin(R_11, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(R_22, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(R_55, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(R_66, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_11, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_22, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_55, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (R_66, MAT_FINAL_ASSEMBLY, petsc_ierr)
#endif

end subroutine construct_pc_diagonal_matrices


!> Assemble the Schur correction matrices.
!!
subroutine construct_schur_correction_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                          K_psi_correction, K_u_correction,        &
                                          K_21_correction,  K_61_correction)

  use mod_elt_matrix_elliptic
  use mod_parameters, only: n_tor, n_degrees, n_vertex_max, var_psi, var_u, var_T
  use data_structure,  only: type_SP_MATRIX, type_element, type_node
  use nodes_elements
  use phys_module,     only: eliminate_boundary_dofs
  use omp_lib

  implicit none

  integer,              intent(in) :: my_id
  integer, pointer,     intent(in) :: local_elms(:)
  integer,              intent(in) :: n_local_elms
  type(type_SP_MATRIX), intent(in) :: a_mat
#ifdef USE_PETSC
  Mat, intent(inout) :: K_psi_correction, K_u_correction
  Mat, intent(inout) :: K_21_correction,  K_61_correction
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
  real*8, allocatable :: ELM_psi_correction_thr(:,:,:)
  real*8, allocatable :: ELM_u_correction_thr(:,:,:)
  real*8, allocatable :: ELM_21_correction_thr (:,:,:)
  real*8, allocatable :: ELM_61_correction_thr (:,:,:)
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

  if (my_id .eq. 0) then
      write(*,*) '****************************************'
      write(*,*) '*        construct PC matrices         *'
      write(*,*) '****************************************'
  endif

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
  allocate(ELM_psi_correction_thr (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_u_correction_thr (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_21_correction_thr (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_61_correction_thr (D1V_CM, D1V_CM, nthreads))
  allocate(buf1v_thr    (bs1*bs1, nthreads))

  !$omp parallel &
  !$omp   default(none) &
  !$omp   shared(n_local_elms, local_elms, element_list, node_list, a_mat, my_ind_min, my_ind_max, bs1, &
  !$omp          element_thr, nodes_thr, node_out_thr, &
  !$omp          ELM_j_thr, ELM_w_thr, ELM_jpsi_thr, ELM_wu_thr, ELM_psi_correction_thr, ELM_u_correction_thr, &
  !$omp          ELM_21_correction_thr, ELM_61_correction_thr, &
  !$omp          buf1v_thr &
#ifdef USE_PETSC
  !$omp          , K_psi_correction, K_u_correction, K_21_correction, K_61_correction &
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
                                  ELM_jpsi_thr(:,:,omp_tid), ELM_wu_thr(:,:,omp_tid), &
                                  ELM_psi_correction=ELM_psi_correction_thr(:,:,omp_tid), &
                                  ELM_u_correction=ELM_u_correction_thr(:,:,omp_tid), &
                                  ELM_21_correction=ELM_21_correction_thr(:,:,omp_tid), &
                                  ELM_61_correction=ELM_61_correction_thr(:,:,omp_tid))

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

            ! --- Extract and insert 1-var block for K_psi_correction ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_psi_correction_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(K_psi_correction, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for K_u_correction ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_u_correction_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(K_u_correction, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for K_21_correction (u-eq, psi-col) ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_21_correction_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(K_21_correction, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, petsc_ierr)
            !$omp end critical

            ! --- Extract and insert 1-var block for K_61_correction (T-eq, psi-col) ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_61_correction_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            call MatSetValuesBlocked(K_61_correction, 1, idxm, 1, idxn, &
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
  deallocate(ELM_j_thr, ELM_w_thr, ELM_jpsi_thr, ELM_wu_thr, ELM_psi_correction_thr, ELM_u_correction_thr)
  deallocate(ELM_21_correction_thr, ELM_61_correction_thr)
  deallocate(buf1v_thr)

#ifdef USE_PETSC
  ! Final assembly of element contributions before zeroing boundary rows.
  ! MatZeroRows requires the matrix to be fully assembled.
  call MatAssemblyBegin(K_psi_correction, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(K_u_correction,   MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(K_21_correction,  MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyBegin(K_61_correction,  MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (K_psi_correction, MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (K_u_correction,   MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (K_21_correction,  MAT_FINAL_ASSEMBLY, petsc_ierr)
  call MatAssemblyEnd  (K_61_correction,  MAT_FINAL_ASSEMBLY, petsc_ierr)

  ! Zero rows of constrained boundary DOFs so that
  !   Atilde = B - K
  ! preserves B's BC enforcement (B already has ZBIG/elm_diag on those rows;
  ! K must contribute nothing — neither diagonal nor off-diagonal — there).
  ! Only needed when eliminate_boundary_dofs is active in the global matrix,
  ! since that mode uses elm-diagonal scaling (not ZBIG) for BC rows.
  if (eliminate_boundary_dofs) then
    call zero_bc_rows_pc_matrix(K_psi_correction, var_psi, local_elms, n_local_elms, my_ind_min, my_ind_max)
    call zero_bc_rows_pc_matrix(K_u_correction,   var_u,   local_elms, n_local_elms, my_ind_min, my_ind_max)
    call zero_bc_rows_pc_matrix(K_21_correction,  var_u,   local_elms, n_local_elms, my_ind_min, my_ind_max)
    call zero_bc_rows_pc_matrix(K_61_correction,  var_T,   local_elms, n_local_elms, my_ind_min, my_ind_max)
  endif
#endif

end subroutine construct_schur_correction_matrices


!--------------------------------------------------------------------
!> Apply model199-style boundary conditions to a single 1-variable
!! PETSc matrix using the ZBIG penalty method.
!!
!! For each boundary node, sets the diagonal entry to ZBIG for the
!! appropriate derivative DOFs based on boundary type. Respects
!! is_freebound (STARWALL vacuum) and keep_n0_const flags.
!!
!! @param pc_mat      1-var PETSc matrix (BAIJ, block_size = n_tor)
!! @param var_index   Variable index (1=psi, 2=u, 3=j, 4=w, 5=rho, 6=T)
!! @param local_elms  List of local element indices
!! @param n_local_elms Number of local elements
!! @param my_ind_min  Min node index owned by this process
!! @param my_ind_max  Max node index owned by this process
!--------------------------------------------------------------------
subroutine apply_bc_pc_matrix(pc_mat, var_index, local_elms, n_local_elms, &
                               my_ind_min, my_ind_max)

  use mod_parameters, only: n_tor, n_vertex_max, n_order
  use data_structure,  only: type_node
  use nodes_elements,  only: node_list, element_list
  use mod_node_indices, only: calculate_node_indices
  use phys_module, only: keep_n0_const
  use vacuum, only: is_freebound

  implicit none

#ifdef USE_PETSC
  Mat, intent(inout) :: pc_mat
#endif
  integer, intent(in) :: var_index
  integer, intent(in) :: local_elms(:)
  integer, intent(in) :: n_local_elms
  integer, intent(in) :: my_ind_min, my_ind_max

  real*8  :: zbig, zbig_backup
  integer :: i, in, iv, inode, ielm
  integer :: index_node, index_tmp, kk, ll, iv_dir
  integer :: node_indices((n_order+1)/2, (n_order+1)/2)
#ifdef USE_PETSC
  PetscInt       :: petsc_row
  PetscErrorCode :: petsc_ierr
#endif

  call calculate_node_indices(node_indices)

  zbig_backup = 1.d12

  do i = 1, n_local_elms
    ielm = local_elms(i)

    do iv = 1, n_vertex_max
      inode = element_list%element(ielm)%vertex(iv)

      if (node_list%node(inode)%boundary .eq. 0) cycle

      do in = 1, n_tor
        if (keep_n0_const .and. in .eq. 1) then
          zbig = 1.d15
        else
          zbig = zbig_backup
        endif

        if (is_freebound(in, var_index)) cycle

        !--- Open field lines (boundary type 1 or 3)
        if ((node_list%node(inode)%boundary .eq. 1) .or. &
            (node_list%node(inode)%boundary .eq. 3)) then

          iv_dir = 2
          do kk = 1, (n_order+1)/2
            if ((iv_dir .eq. 3) .and. (kk .gt. 1)) cycle
            do ll = 1, (n_order+1)/2
              if ((iv_dir .eq. 2) .and. (ll .gt. 1)) cycle
              index_tmp = node_indices(kk, ll)
              index_node = node_list%node(inode)%index(index_tmp)

              if ((index_node .lt. my_ind_min) .or. (index_node .gt. my_ind_max)) cycle

#ifdef USE_PETSC
              petsc_row = n_tor * (index_node - 1) + (in - 1)
              call MatSetValue(pc_mat, petsc_row, petsc_row, zbig, INSERT_VALUES, petsc_ierr)
#endif
            enddo
          enddo
        endif

        !--- Wall-aligned with flux surface (boundary type 2 or 3)
        if ((node_list%node(inode)%boundary .eq. 2) .or. &
            (node_list%node(inode)%boundary .eq. 3)) then

          iv_dir = 3
          do kk = 1, (n_order+1)/2
            if ((iv_dir .eq. 3) .and. (kk .gt. 1)) cycle
            do ll = 1, (n_order+1)/2
              if ((iv_dir .eq. 2) .and. (ll .gt. 1)) cycle
              index_tmp = node_indices(kk, ll)
              index_node = node_list%node(inode)%index(index_tmp)

              if ((index_node .lt. my_ind_min) .or. (index_node .gt. my_ind_max)) cycle

#ifdef USE_PETSC
              petsc_row = n_tor * (index_node - 1) + (in - 1)
              call MatSetValue(pc_mat, petsc_row, petsc_row, zbig, INSERT_VALUES, petsc_ierr)
#endif
            enddo
          enddo
        endif

      enddo  ! in
    enddo    ! iv
  enddo      ! i

end subroutine apply_bc_pc_matrix


!--------------------------------------------------------------------
!> Zero out the rows of a 1-variable PC matrix that correspond to
!! constrained boundary DOFs.
!!
!! Used for matrices that are SUBTRACTED from a B-block which already
!! has BCs enforced (e.g. the Schur correction matrices K_psi, K_u).
!! Setting those rows to exactly zero ensures `Atilde = B - K` keeps
!! B's boundary equations untouched. Using a ZBIG diagonal here would
!! cancel B's ZBIG diagonal during the subtraction; using zero is
!! correct only when K is consumed via subtraction from a BC-enforced
!! matrix.
!!
!! Constrained DOFs per boundary type (matching apply_bc_pc_matrix):
!!   type 1 (open field lines):  i_order ∈ {1, 2}  (value, ds)
!!   type 2 (wall-aligned):      i_order ∈ {1, 3}  (value, dt)
!!   type 3 (corner):            i_order ∈ {1, 2, 3}
!! Toroidal modes flagged by `is_freebound(in, var_index)` are skipped.
!--------------------------------------------------------------------
subroutine zero_bc_rows_pc_matrix(pc_mat, var_index, local_elms, n_local_elms, &
                                   my_ind_min, my_ind_max)

  use mod_parameters,   only: n_tor, n_vertex_max, n_order
  use nodes_elements,   only: node_list, element_list
  use mod_node_indices, only: calculate_node_indices
  use vacuum,           only: is_freebound

  implicit none

#ifdef USE_PETSC
  Mat, intent(inout) :: pc_mat
#endif
  integer, intent(in) :: var_index
  integer, intent(in) :: local_elms(:)
  integer, intent(in) :: n_local_elms
  integer, intent(in) :: my_ind_min, my_ind_max

  integer :: i, in, iv, inode, ielm
  integer :: index_node, index_tmp, kk, ll, iv_dir
  integer :: node_indices((n_order+1)/2, (n_order+1)/2)
  logical :: skip_mode(n_tor)

#ifdef USE_PETSC
  PetscInt, allocatable :: rows_to_zero(:)
  PetscInt              :: n_rows_to_zero
  PetscInt              :: cap
  PetscErrorCode        :: petsc_ierr

  call calculate_node_indices(node_indices)

  ! is_freebound depends only on (in, var_index); precompute once.
  do in = 1, n_tor
    skip_mode(in) = is_freebound(in, var_index)
  enddo

  ! Upper bound: each local element contributes up to n_vertex_max nodes,
  ! each with up to 3 constrained DOFs across n_tor modes.
  cap = n_local_elms * n_vertex_max * 3 * n_tor
  if (cap < 1) cap = 1
  allocate(rows_to_zero(cap))
  n_rows_to_zero = 0

  do i = 1, n_local_elms
    ielm = local_elms(i)
    do iv = 1, n_vertex_max
      inode = element_list%element(ielm)%vertex(iv)

      if (node_list%node(inode)%boundary .eq. 0) cycle

      do in = 1, n_tor
        if (skip_mode(in)) cycle

        !--- Open field lines: constrain value + ds DOFs
        if ((node_list%node(inode)%boundary .eq. 1) .or. &
            (node_list%node(inode)%boundary .eq. 3)) then
          iv_dir = 2
          do kk = 1, (n_order+1)/2
            if ((iv_dir .eq. 3) .and. (kk .gt. 1)) cycle
            do ll = 1, (n_order+1)/2
              if ((iv_dir .eq. 2) .and. (ll .gt. 1)) cycle
              index_tmp  = node_indices(kk, ll)
              index_node = node_list%node(inode)%index(index_tmp)
              if ((index_node .lt. my_ind_min) .or. (index_node .gt. my_ind_max)) cycle
              n_rows_to_zero = n_rows_to_zero + 1
              rows_to_zero(n_rows_to_zero) = n_tor * (index_node - 1) + (in - 1)
            enddo
          enddo
        endif

        !--- Wall-aligned with flux surface: constrain value + dt DOFs
        if ((node_list%node(inode)%boundary .eq. 2) .or. &
            (node_list%node(inode)%boundary .eq. 3)) then
          iv_dir = 3
          do kk = 1, (n_order+1)/2
            if ((iv_dir .eq. 3) .and. (kk .gt. 1)) cycle
            do ll = 1, (n_order+1)/2
              if ((iv_dir .eq. 2) .and. (ll .gt. 1)) cycle
              index_tmp  = node_indices(kk, ll)
              index_node = node_list%node(inode)%index(index_tmp)
              if ((index_node .lt. my_ind_min) .or. (index_node .gt. my_ind_max)) cycle
              n_rows_to_zero = n_rows_to_zero + 1
              rows_to_zero(n_rows_to_zero) = n_tor * (index_node - 1) + (in - 1)
            enddo
          enddo
        endif

      enddo  ! in
    enddo    ! iv
  enddo      ! i

  ! Collective: every rank must call MatZeroRows even with n=0.
  ! Diag value = 0  →  the rows become identically zero (no diagonal injection).
  ! Duplicates in rows_to_zero are harmless (idempotent).
  call MatZeroRows(pc_mat, n_rows_to_zero, rows_to_zero, 0.0d0, &
                   PETSC_NULL_VEC, PETSC_NULL_VEC, petsc_ierr)

  deallocate(rows_to_zero)
#endif

end subroutine zero_bc_rows_pc_matrix

end module construct_pc_matrix_mod
