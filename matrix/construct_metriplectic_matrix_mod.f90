module construct_metriplectic_matrix_mod
!----------------------------------------------------------------
! Element-loop driver for the metriplectic HSS PC operators
! (Slice A). OMP structure and PETSc insertion follow
! construct_pc_matrix_mod::construct_schur_correction_matrices;
! BC handling reuses zero_bc_rows_pc_matrix / apply_dirichlet_bnd
! from that module.
!
! BC policy (docs/notes/metriplectic_parabolization_note.tex,
! Remarks BT1/BT2): Dirichlet identity rows are the mathematically
! correct BC for the SPD operators (M_psi, L_rho, W_para); the
! coupling operators (D, D', D'_struct) get zeroed BC rows so they
! contribute nothing on constrained DOFs.
!----------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none

contains

subroutine construct_metriplectic_matrices(my_id, local_elms, n_local_elms, a_mat, &
                                           M_psi, D_op, Dp_op, Dp_struct, L_rho, W_para)

  use mod_elt_matrix_metriplectic
  use mod_parameters,  only: n_tor, n_degrees, n_vertex_max, var_psi, var_u
  use data_structure,  only: type_SP_MATRIX, type_element, type_node
  use nodes_elements
  use phys_module,     only: eliminate_boundary_dofs
  use construct_pc_matrix_mod, only: zero_bc_rows_pc_matrix, apply_dirichlet_bnd
  use omp_lib

  implicit none

  integer,              intent(in) :: my_id
  integer, pointer,     intent(in) :: local_elms(:)
  integer,              intent(in) :: n_local_elms
  type(type_SP_MATRIX), intent(in) :: a_mat
#ifdef USE_PETSC
  Mat, intent(inout) :: M_psi, D_op, Dp_op, Dp_struct, L_rho, W_para
#endif

  integer :: my_ind_min, my_ind_max

#define D1V_CM (n_tor*n_vertex_max*n_degrees)

  ! --- Thread-local variables (OMP private) ---
  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  integer,            allocatable :: node_out_thr(:,:)

  real*8, allocatable :: ELM_Mpsi_thr (:,:,:)
  real*8, allocatable :: ELM_D_thr    (:,:,:)
  real*8, allocatable :: ELM_Dp_thr   (:,:,:)
  real*8, allocatable :: ELM_Dps_thr  (:,:,:)
  real*8, allocatable :: ELM_Lrho_thr (:,:,:)
  real*8, allocatable :: ELM_Wpara_thr(:,:,:)
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
      write(*,*) '*   construct metriplectic PC matrices      *'
      write(*,*) '*********************************************'
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
  allocate(ELM_Mpsi_thr (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_D_thr    (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_Dp_thr   (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_Dps_thr  (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_Lrho_thr (D1V_CM, D1V_CM, nthreads))
  allocate(ELM_Wpara_thr(D1V_CM, D1V_CM, nthreads))
  allocate(buf1v_thr    (bs1*bs1, nthreads))

  !$omp parallel &
  !$omp   default(shared) &
  !$omp   shared(n_local_elms, local_elms, element_list, node_list, a_mat, my_ind_min, my_ind_max, bs1, &
  !$omp          element_thr, nodes_thr, node_out_thr, &
  !$omp          ELM_Mpsi_thr, ELM_D_thr, ELM_Dp_thr, ELM_Dps_thr, ELM_Lrho_thr, ELM_Wpara_thr, &
  !$omp          buf1v_thr &
#ifdef USE_PETSC
  !$omp          , M_psi, D_op, Dp_op, Dp_struct, L_rho, W_para &
#endif
  !$omp         ) &
  !$omp   private(ife, ielm, iv, inode, omp_tid, &
  !$omp           i, i_order, k, k_order, &
  !$omp           index_node1, index_node2, &
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

    ! --- Fetch element and nodes (no refinement support needed for PC) ---
    element_thr(omp_tid) = element_list%element(ielm)
    do iv = 1, n_vertex_max
      inode = element_thr(omp_tid)%vertex(iv)
      nodes_thr(iv, omp_tid)   = node_list%node(inode)
      node_out_thr(iv, omp_tid) = inode
    enddo

    ! --- Compute element matrices ---
    call element_matrix_metriplectic(element_thr(omp_tid), nodes_thr(:,omp_tid), &
                                     ELM_Mpsi_thr (:,:,omp_tid), &
                                     ELM_D_thr    (:,:,omp_tid), &
                                     ELM_Dp_thr   (:,:,omp_tid), &
                                     ELM_Dps_thr  (:,:,omp_tid), &
                                     ELM_Lrho_thr (:,:,omp_tid), &
                                     ELM_Wpara_thr(:,:,omp_tid))

#ifdef USE_PETSC
    ! --- Insert element blocks into PETSc matrices (all six are 1-var) ---
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

            ! --- M_psi ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_Mpsi_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(M_psi, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, ierr))
            !$omp end critical

            ! --- D_op ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_D_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(D_op, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, ierr))
            !$omp end critical

            ! --- Dp_op ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_Dp_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(Dp_op, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, ierr))
            !$omp end critical

            ! --- Dp_struct ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_Dps_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(Dp_struct, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, ierr))
            !$omp end critical

            ! --- L_rho ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_Lrho_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(L_rho, 1, idxm, 1, idxn, &
                                     buf1v_thr(:,omp_tid), ADD_VALUES, ierr))
            !$omp end critical

            ! --- W_para ---
            buf1v_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf1v_thr((j-1)*bs1+l, omp_tid) = ELM_Wpara_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(W_para, 1, idxm, 1, idxn, &
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
  deallocate(ELM_Mpsi_thr, ELM_D_thr, ELM_Dp_thr, ELM_Dps_thr, ELM_Lrho_thr, ELM_Wpara_thr)
  deallocate(buf1v_thr)

#ifdef USE_PETSC
  ! Final assembly of element contributions before touching BC rows.
  PetscCallA(MatAssemblyBegin(M_psi,     MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyBegin(D_op,      MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyBegin(Dp_op,     MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyBegin(Dp_struct, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyBegin(L_rho,     MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyBegin(W_para,    MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (M_psi,     MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (D_op,      MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (Dp_op,     MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (Dp_struct, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (L_rho,     MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (W_para,    MAT_FINAL_ASSEMBLY, ierr))

  ! Boundary rows (note Remarks BT1/BT2):
  !  - coupling operators contribute nothing on constrained DOFs (zeroed rows);
  !  - SPD operators get Dirichlet identity rows (solvable, definite).
  if (eliminate_boundary_dofs) then
    call zero_bc_rows_pc_matrix(D_op,      var_psi, local_elms, n_local_elms, my_ind_min, my_ind_max)
    call zero_bc_rows_pc_matrix(Dp_op,     var_u,   local_elms, n_local_elms, my_ind_min, my_ind_max)
    call zero_bc_rows_pc_matrix(Dp_struct, var_u,   local_elms, n_local_elms, my_ind_min, my_ind_max)

    call apply_dirichlet_bnd(M_psi,  var_psi, local_elms, n_local_elms, my_ind_min, my_ind_max, symmetric=.true.)
    call apply_dirichlet_bnd(L_rho,  var_u,   local_elms, n_local_elms, my_ind_min, my_ind_max, symmetric=.true.)
    ! W_para: zero diagonal on BC rows — the composed P_u = L_rho + tau^2*W_para
    ! then carries exactly L_rho's unit Dirichlet diagonal at every tau.
    call apply_dirichlet_bnd(W_para, var_u,   local_elms, n_local_elms, my_ind_min, my_ind_max, &
                             symmetric=.true., diag_value=0.d0)
  endif
#endif

end subroutine construct_metriplectic_matrices

end module construct_metriplectic_matrix_mod
