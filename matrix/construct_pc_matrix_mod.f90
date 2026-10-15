module construct_pc_matrix_mod
!----------------------------------------------------------------
! Element assembly of the physics PC's operators: the composed force
! operator W of the SF path (construct_force_operator_matrix) and the
! R^3 rho0-weighted mass M_w of the flux arm (construct_mass_w_matrix).
!
! Reuses the mesh traversal of construct_matrix_mod but calls the
! PC element routines (the model's pc_elt_matrix_force_fft, the
! model-independent pc_elt_mass_w) and inserts into a 1-variable
! PETSc BAIJ matrix.
!
! Assumptions:
!   - No mesh refinement (PC is an approximation).
!   - n_tor_local = n_tor  (all toroidal modes on every process).
!----------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none
private
public :: construct_force_operator_matrix
public :: construct_mass_w_matrix

contains




#ifdef USE_PETSC
!--------------------------------------------------------------------
!> Collect the global rows of a 1-variable PC matrix (block size n_tor)
!! that correspond to constrained boundary DOFs owned by this rank.
!!
!! Constrained DOFs per boundary type (matching the model's boundary_conditions):
!!   type 1 (open field lines):  i_order ∈ {1, 2}  (value, ds)
!!   type 2 (wall-aligned):      i_order ∈ {1, 3}  (value, dt)
!!   type 3 (corner):            i_order ∈ {1, 2, 3}
!! Toroidal modes flagged by `is_freebound(in, var_index)` are skipped.
!! Row index of (node index, mode in): n_tor*(index_node-1) + (in-1).
!! The list may contain duplicates (nodes shared by several local elements).
!--------------------------------------------------------------------
subroutine collect_bc_rows_pc_matrix(var_index, local_elms, n_local_elms, &
                                     my_ind_min, my_ind_max, rows_bc, n_rows_bc)

  use mod_parameters,   only: n_tor, n_vertex_max, n_order
  use nodes_elements,   only: node_list, element_list
  use mod_node_indices, only: calculate_node_indices
  use vacuum,           only: is_freebound

  implicit none

  integer,               intent(in)  :: var_index       !< variable whose free-boundary modes are skipped
  integer,               intent(in)  :: local_elms(:)   !< local element list
  integer,               intent(in)  :: n_local_elms    !< number of local elements
  integer,               intent(in)  :: my_ind_min      !< first node index owned by this rank
  integer,               intent(in)  :: my_ind_max      !< last node index owned by this rank
  PetscInt, allocatable, intent(out) :: rows_bc(:)      !< the constrained global rows (size >= 1)
  PetscInt,              intent(out) :: n_rows_bc       !< number of valid entries in rows_bc

  integer :: i, in, iv, inode, ielm
  integer :: index_node, index_tmp, kk, ll, iv_dir
  integer :: node_indices((n_order+1)/2, (n_order+1)/2)
  logical :: skip_mode(n_tor)
  PetscInt :: cap

  call calculate_node_indices(node_indices)

  ! is_freebound depends only on (in, var_index); precompute once.
  do in = 1, n_tor
    skip_mode(in) = is_freebound(in, var_index)
  enddo

  ! Upper bound: each local element contributes up to n_vertex_max nodes,
  ! each with up to 3 constrained DOFs across n_tor modes.
  cap = n_local_elms * n_vertex_max * 3 * n_tor
  if (cap < 1) cap = 1
  allocate(rows_bc(cap))
  n_rows_bc = 0

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
              n_rows_bc = n_rows_bc + 1
              rows_bc(n_rows_bc) = n_tor * (index_node - 1) + (in - 1)
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
              n_rows_bc = n_rows_bc + 1
              rows_bc(n_rows_bc) = n_tor * (index_node - 1) + (in - 1)
            enddo
          enddo
        endif

      enddo  ! in
    enddo    ! iv
  enddo      ! i

end subroutine collect_bc_rows_pc_matrix
#endif



!--------------------------------------------------------------------
!> Zero out the rows of a 1-variable PC matrix that correspond to
!! constrained boundary DOFs (the DOF set of collect_bc_rows_pc_matrix).
!!
!! Used for matrices that are ADDED to a B-block which already has BCs
!! enforced (W, added to B_22). Setting those rows to exactly zero keeps
!! B's boundary equations untouched; a ZBIG diagonal here would double
!! B's ZBIG diagonal.
!--------------------------------------------------------------------
subroutine zero_bc_rows_pc_matrix(pc_mat, var_index, local_elms, n_local_elms, &
                                   my_ind_min, my_ind_max)

  implicit none

#ifdef USE_PETSC
  Mat, intent(inout) :: pc_mat
#endif
  integer, intent(in) :: var_index
  integer, intent(in) :: local_elms(:)
  integer, intent(in) :: n_local_elms
  integer, intent(in) :: my_ind_min, my_ind_max

#ifdef USE_PETSC
  PetscInt, allocatable :: rows_to_zero(:)
  PetscInt              :: n_rows_to_zero
  PetscErrorCode        :: ierr

  call collect_bc_rows_pc_matrix(var_index, local_elms, n_local_elms, &
                                 my_ind_min, my_ind_max, rows_to_zero, n_rows_to_zero)

  ! Collective: every rank must call MatZeroRows even with n=0.
  ! Diag value = 0  →  the rows become identically zero (no diagonal injection).
  ! Duplicates in rows_to_zero are harmless (idempotent).
  PetscCallA(MatZeroRows(pc_mat, n_rows_to_zero, rows_to_zero, 0.0d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr))

  deallocate(rows_to_zero)
#endif

end subroutine zero_bc_rows_pc_matrix



!--------------------------------------------------------------------
!> Assemble the momentum-Schur FORCE OPERATOR W.
!!
!! One variable (u), so block size is n_tor. The element matrix is
!! (theta*dt)^2/opz * a(du,v) -- exactly the matrix to be ADDED to B_22 to form
!! the composed S_uu (see mod_pc_elt_matrix_force_fft for the operator and the
!! self-adjointness statement).
!!
!! It needs mhd_sim: the operator is built on the linearisation state, not on
!! the Jacobian.
!!
!! With W_vv, W_vu, W_uv present (model600 with_vpar, the SF path's "b0"
!! stage) the same element pass also assembles v_par's sound-wave Schur terms
!! into them (one variable each, like W): v_par row / column as the element
!! routine's ELM_v blocks 1 (v_par, v_par), 2 (v_par, u), 3 (u, v_par).
!--------------------------------------------------------------------
subroutine construct_force_operator_matrix(my_id, local_elms, n_local_elms, a_mat, mhd_sim, W_force, terms, &
                                           W_vv, W_vu, W_uv, gamma_w, vu_full)

  use mod_pc_elt_matrix_force_fft, only: pc_elt_matrix_force_fft
  use mod_parameters,  only: n_tor, n_degrees, n_vertex_max, var_u, var_Vpar
  use data_structure,  only: type_SP_MATRIX, type_element, type_node
  use mod_simulation_data, only: type_MHD_SIM
  use nodes_elements
  use phys_module,     only: eliminate_boundary_dofs
  use omp_lib

  implicit none

  integer,              intent(in) :: my_id
  integer, pointer,     intent(in) :: local_elms(:)
  integer,              intent(in) :: n_local_elms
  type(type_SP_MATRIX), intent(in) :: a_mat
  type(type_MHD_SIM),   intent(in) :: mhd_sim
#ifdef USE_PETSC
  Mat, intent(inout) :: W_force
#endif
  !> term selection of pc_elt_matrix_force_fft (sf_force_terms)
  integer, intent(in) :: terms
#ifdef USE_PETSC
  Mat, intent(inout), optional :: W_vv, W_vu, W_uv
#endif
  real*8, intent(in), optional :: gamma_w
  logical, intent(in), optional :: vu_full   !< W_vu with its R [p0,u] half (element routine)

  integer :: my_ind_min, my_ind_max

  integer :: xcase2
  real*8  :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8  :: R_xpoint(2), Z_xpoint(2)
  logical :: xpoint2

#define DFV_CM (n_tor*n_vertex_max*n_degrees)

  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  integer,            allocatable :: node_out_thr(:,:)
  real*8,             allocatable :: ELM_thr(:,:,:)
  real*8,             allocatable :: buf_thr(:,:)
  real*8,             allocatable :: ELMv_thr(:,:,:,:)
  logical :: do_v, v_full
  real*8  :: g_v
  integer :: b

  integer :: nthreads, omp_tid
  integer :: ife, ielm, iv, inode
  integer :: i, i_order, k, k_order
  integer :: index_node1, index_node2
  integer :: j, l, idx_ij, idx_kl
  integer :: bs1
#ifdef USE_PETSC
  PetscErrorCode :: ierr
  PetscInt :: idxm(1), idxn(1)
  Mat :: Wv(3)
#endif

  my_ind_min = a_mat%index_min(my_id+1)
  my_ind_max = a_mat%index_max(my_id+1)
  bs1        = n_tor

  xpoint2       = mhd_sim%es%xpoint
  xcase2        = mhd_sim%es%xcase
  R_axis        = mhd_sim%es%R_axis
  Z_axis        = mhd_sim%es%Z_axis
  psi_axis      = mhd_sim%es%psi_axis
  psi_bnd       = mhd_sim%es%psi_bnd
  R_xpoint(1:2) = mhd_sim%es%R_xpoint(1:2)
  Z_xpoint(1:2) = mhd_sim%es%Z_xpoint(1:2)

  !$omp parallel
  !$omp master
#ifdef _OPENMP
  nthreads = omp_get_num_threads()
#else
  nthreads = 1
#endif
  !$omp end master
  !$omp end parallel

  allocate(element_thr (nthreads))
  allocate(nodes_thr   (n_vertex_max, nthreads))
  allocate(node_out_thr(n_vertex_max, nthreads))
  allocate(ELM_thr     (DFV_CM, DFV_CM, nthreads))
  allocate(buf_thr     (bs1*bs1, nthreads))

  do_v = .false.
#ifdef USE_PETSC
  do_v = present(W_vv) .and. present(W_vu) .and. present(W_uv)
  if (do_v) then
    Wv(1) = W_vv; Wv(2) = W_vu; Wv(3) = W_uv
  endif
#endif
  g_v = -1.d0
  if (present(gamma_w)) g_v = gamma_w
  v_full = .false.
  if (present(vu_full)) v_full = vu_full
  if (do_v) then
    allocate(ELMv_thr(DFV_CM, DFV_CM, 3, nthreads))
  else
    allocate(ELMv_thr(1, 1, 1, 1))
  endif

  !$omp parallel &
  !$omp   default(shared) &
  !$omp   shared(n_local_elms, local_elms, element_list, node_list, a_mat, &
  !$omp          my_ind_min, my_ind_max, bs1, &
  !$omp          xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, &
  !$omp          element_thr, nodes_thr, node_out_thr, ELM_thr, buf_thr, &
  !$omp          do_v, g_v, v_full, ELMv_thr &
#ifdef USE_PETSC
  !$omp          , W_force, Wv &
#endif
  !$omp         ) &
  !$omp   private(ife, ielm, iv, inode, omp_tid, &
  !$omp           i, i_order, k, k_order, &
  !$omp           index_node1, index_node2, &
  !$omp           j, l, idx_ij, idx_kl, b &
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

    if (do_v) then
      if (g_v > 0.d0) then
        call pc_elt_matrix_force_fft(element_thr(omp_tid), nodes_thr(:,omp_tid), &
                                     xpoint2, xcase2, R_axis, Z_axis,           &
                                     psi_axis, psi_bnd, R_xpoint, Z_xpoint,     &
                                     ELM_thr(:,:,omp_tid), terms,               &
                                     ELM_v=ELMv_thr(:,:,:,omp_tid), gamma_w=g_v, vu_full=v_full)
      else
        call pc_elt_matrix_force_fft(element_thr(omp_tid), nodes_thr(:,omp_tid), &
                                     xpoint2, xcase2, R_axis, Z_axis,           &
                                     psi_axis, psi_bnd, R_xpoint, Z_xpoint,     &
                                     ELM_thr(:,:,omp_tid), terms,               &
                                     ELM_v=ELMv_thr(:,:,:,omp_tid), vu_full=v_full)
      endif
    else
      call pc_elt_matrix_force_fft(element_thr(omp_tid), nodes_thr(:,omp_tid), &
                                   xpoint2, xcase2, R_axis, Z_axis,           &
                                   psi_axis, psi_bnd, R_xpoint, Z_xpoint,     &
                                   ELM_thr(:,:,omp_tid), terms)
    endif

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

            buf_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf_thr((j-1)*bs1+l, omp_tid) = ELM_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(W_force, 1, idxm, 1, idxn, buf_thr(:,omp_tid), ADD_VALUES, ierr))
            !$omp end critical

            ! v_par's blocks: the same node-index layout, one matrix each
            if (do_v) then
              do b = 1, 3
                do j = 1, bs1
                  idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
                  do l = 1, bs1
                    idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                    buf_thr((j-1)*bs1+l, omp_tid) = ELMv_thr(idx_ij, idx_kl, b, omp_tid)
                  enddo
                enddo
                !$omp critical
                PetscCallA(MatSetValuesBlocked(Wv(b), 1, idxm, 1, idxn, buf_thr(:,omp_tid), ADD_VALUES, ierr))
                !$omp end critical
              enddo
            endif

          enddo  ! k_order
        enddo    ! k

      enddo  ! i_order
    enddo    ! i
#endif

  enddo  ! ife
  !$omp end do
  !$omp end parallel

  deallocate(element_thr, nodes_thr, node_out_thr, ELM_thr, buf_thr, ELMv_thr)

#ifdef USE_PETSC
  PetscCallA(MatAssemblyBegin(W_force, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (W_force, MAT_FINAL_ASSEMBLY, ierr))

  ! Zero the constrained boundary rows.
  !
  ! An earlier version deliberately skipped this, arguing that W is a CORRECTION
  ! to B_22 and would inherit B_22's Dirichlet treatment. That is wrong: B_22's
  ! boundary row is [ZBIG on the diagonal, zeros elsewhere], so adding W's raw
  ! element entries there DESTROYS the constraint rather than inheriting it.
  ! Measured at tstep 10 (51x16), B_22 + W as a preconditioner for S_uu:
  ! 1295 iterations with the raw rows, 221 with them zeroed -- a 5.9x error
  ! that is only 0.4% of ||W||, so no Frobenius-norm check can see it.
  ! Zeroing the columns as well makes no further difference (221 either way).
  if (eliminate_boundary_dofs) then
    call zero_bc_rows_pc_matrix(W_force, var_u, local_elms, n_local_elms, &
                                a_mat%my_ind_min, a_mat%my_ind_max)
  endif

  ! v_par's blocks: assembled and their constrained rows zeroed the same way,
  ! by their row variable (they are ADDED to B_77 / B_72, rows with BCs)
  if (do_v) then
    do b = 1, 3
      PetscCallA(MatAssemblyBegin(Wv(b), MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd  (Wv(b), MAT_FINAL_ASSEMBLY, ierr))
    enddo
    if (eliminate_boundary_dofs) then
      call zero_bc_rows_pc_matrix(Wv(1), var_Vpar, local_elms, n_local_elms, &
                                  a_mat%my_ind_min, a_mat%my_ind_max)
      call zero_bc_rows_pc_matrix(Wv(2), var_Vpar, local_elms, n_local_elms, &
                                  a_mat%my_ind_min, a_mat%my_ind_max)
      call zero_bc_rows_pc_matrix(Wv(3), var_u, local_elms, n_local_elms, &
                                  a_mat%my_ind_min, a_mat%my_ind_max)
    endif
  endif
#endif

end subroutine construct_force_operator_matrix



!--------------------------------------------------------------------
!> Assemble the R^3 rho0-weighted mass M_w of the flux arm.
!!
!! One variable, block size n_tor, diagonal in the toroidal slot (see
!! modelPC/mod_pc_elt_mass_w for the kernel and the n = 0 weight). It needs
!! no mhd_sim: the nodal values in node_list are the linearisation state.
!!
!! Unlike W this matrix is used on its own (as an SPD mass), not added to a
!! B-block, so:
!!   - the axis basis transform of construct_matrix is applied to axis
!!     elements when treat_axis is on (same routine, same condition);
!!   - the constrained DOFs are imposed as Dirichlet rows AND columns with a
!!     unit diagonal (MatZeroRowsColumns), unconditionally, i.e. independent of
!!     eliminate_boundary_dofs. The set is the boundary set of
!!     collect_bc_rows_pc_matrix (W's set) plus the axis DOFs construct_matrix
!!     penalises: index 4 with treat_axis (its row/column is identically zero
!!     after the transform), indices 3 and 4 with fix_axis_nodes. A unit
!!     diagonal instead of ZBIG: the rows are decoupled, so this only fixes
!!     the constrained DOFs to zero without spoiling the conditioning.
!!
!! var_index selects the variable whose free-boundary modes (is_freebound)
!! are left unconstrained, as in zero_bc_rows_pc_matrix; default var_u.
!--------------------------------------------------------------------
subroutine construct_mass_w_matrix(my_id, local_elms, n_local_elms, a_mat, M_w, var_index)

  use mod_pc_elt_mass_w,  only: pc_elt_mass_w
  use mod_parameters,     only: n_tor, n_degrees, n_vertex_max, var_u
  use data_structure,     only: type_SP_MATRIX, type_element, type_node
  use nodes_elements
  use phys_module,        only: treat_axis, fix_axis_nodes
  use mod_axis_treatment, only: transform_basis_for_axis_element
  use omp_lib

  implicit none

  integer,              intent(in) :: my_id            !< MPI rank
  integer, pointer,     intent(in) :: local_elms(:)    !< local element list
  integer,              intent(in) :: n_local_elms     !< number of local elements
  type(type_SP_MATRIX), intent(in) :: a_mat            !< the system matrix (index ranges)
#ifdef USE_PETSC
  Mat,                  intent(inout) :: M_w           !< the mass, created by petsc_create_pc_matrix(.., 1)
#endif
  integer, optional,    intent(in) :: var_index        !< variable of the free-boundary mode skip (default var_u)

#define DFV_MW (n_tor*n_vertex_max*n_degrees)

  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  integer,            allocatable :: node_out_thr(:,:)
  real*8,             allocatable :: ELM_thr(:,:,:)
  real*8,             allocatable :: RHS_thr(:,:)
  real*8,             allocatable :: buf_thr(:,:)

  integer :: my_ind_min, my_ind_max, var_bc
  integer :: nthreads, omp_tid
  integer :: ife, ielm, iv, inode
  integer :: i, i_order, k, k_order
  integer :: index_node1, index_node2
  integer :: j, l, idx_ij, idx_kl
  integer :: bs1, n_ax_dofs, ia
  integer :: i_v1(1), i_harm(n_tor), ax_dofs(2)
#ifdef USE_PETSC
  PetscErrorCode        :: ierr
  PetscInt              :: idxm(1), idxn(1)
  PetscInt, allocatable :: rows_bc(:), rows_dir(:)
  PetscInt              :: n_rows_bc, n_rows_dir, cap, row, row0, row_end, n_loc_rows
  logical,  allocatable :: row_seen(:)
#endif

  my_ind_min = a_mat%index_min(my_id+1)
  my_ind_max = a_mat%index_max(my_id+1)
  bs1        = n_tor

  var_bc = var_u
  if (present(var_index)) var_bc = var_index

  ! the axis transform's variable / harmonic subsets: the one variable, all slots
  i_v1(1) = 1
  do i = 1, n_tor
    i_harm(i) = i
  enddo

  !$omp parallel
  !$omp master
#ifdef _OPENMP
  nthreads = omp_get_num_threads()
#else
  nthreads = 1
#endif
  !$omp end master
  !$omp end parallel

  allocate(element_thr (nthreads))
  allocate(nodes_thr   (n_vertex_max, nthreads))
  allocate(node_out_thr(n_vertex_max, nthreads))
  allocate(ELM_thr     (DFV_MW, DFV_MW, nthreads))
  allocate(RHS_thr     (DFV_MW, nthreads))
  allocate(buf_thr     (bs1*bs1, nthreads))

  !$omp parallel &
  !$omp   default(shared) &
  !$omp   shared(n_local_elms, local_elms, element_list, node_list, &
  !$omp          my_ind_min, my_ind_max, bs1, treat_axis, i_v1, i_harm, &
  !$omp          element_thr, nodes_thr, node_out_thr, ELM_thr, RHS_thr, buf_thr &
#ifdef USE_PETSC
  !$omp          , M_w &
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

    element_thr(omp_tid) = element_list%element(ielm)
    do iv = 1, n_vertex_max
      inode = element_thr(omp_tid)%vertex(iv)
      nodes_thr(iv, omp_tid)    = node_list%node(inode)
      node_out_thr(iv, omp_tid) = inode
    enddo

    call pc_elt_mass_w(element_thr(omp_tid), nodes_thr(:,omp_tid), ELM_thr(:,:,omp_tid))

    ! the axis basis transform, exactly as construct_matrix (one variable,
    ! all n_tor slots); RHS_thr is a dummy the routine requires
    if (treat_axis .and. (nodes_thr(1,omp_tid)%axis_node .or. nodes_thr(2,omp_tid)%axis_node .or. &
                          nodes_thr(3,omp_tid)%axis_node .or. nodes_thr(4,omp_tid)%axis_node)) then
      RHS_thr(:,omp_tid) = 0.d0
      call transform_basis_for_axis_element(nodes_thr(:,omp_tid), ielm, ELM_thr(:,:,omp_tid), &
                                            RHS_thr(:,omp_tid), i_v1, 1, i_harm, n_tor)
    endif

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

            buf_thr(:,omp_tid) = 0.d0
            do j = 1, bs1
              idx_ij = bs1*n_degrees*(i-1) + bs1*(i_order-1) + j
              do l = 1, bs1
                idx_kl = bs1*n_degrees*(k-1) + bs1*(k_order-1) + l
                buf_thr((j-1)*bs1+l, omp_tid) = ELM_thr(idx_ij, idx_kl, omp_tid)
              enddo
            enddo
            !$omp critical
            PetscCallA(MatSetValuesBlocked(M_w, 1, idxm, 1, idxn, buf_thr(:,omp_tid), ADD_VALUES, ierr))
            !$omp end critical

          enddo  ! k_order
        enddo    ! k

      enddo  ! i_order
    enddo    ! i
#endif

  enddo  ! ife
  !$omp end do
  !$omp end parallel

  deallocate(element_thr, nodes_thr, node_out_thr, ELM_thr, RHS_thr, buf_thr)

#ifdef USE_PETSC
  PetscCallA(MatAssemblyBegin(M_w, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (M_w, MAT_FINAL_ASSEMBLY, ierr))

  !--- Dirichlet DOFs: the boundary set (W's), unconditionally
  call collect_bc_rows_pc_matrix(var_bc, local_elms, n_local_elms, &
                                 a_mat%my_ind_min, a_mat%my_ind_max, rows_bc, n_rows_bc)

  !--- plus the axis DOFs construct_matrix penalises
  n_ax_dofs = 0
  if (fix_axis_nodes) then
    n_ax_dofs  = 2
    ax_dofs(1) = 3
    ax_dofs(2) = 4
  elseif (treat_axis) then
    n_ax_dofs  = 1
    ax_dofs(1) = 4
  endif

  cap = n_rows_bc + n_local_elms * n_vertex_max * n_ax_dofs * n_tor
  if (cap < 1) cap = 1
  allocate(rows_dir(cap))

  ! De-duplicated through a mark on the owned rows (nodes are shared by
  ! several local elements). The rows are owned by construction; a row
  ! outside the ownership range would still be passed on, just not marked.
  PetscCallA(MatGetOwnershipRange(M_w, row0, row_end, ierr))
  n_loc_rows = row_end - row0
  if (n_loc_rows < 1) n_loc_rows = 1
  allocate(row_seen(0:n_loc_rows-1))
  row_seen   = .false.
  n_rows_dir = 0

  do ia = 1, n_rows_bc
    row = rows_bc(ia)
    if ((row >= row0) .and. (row < row_end)) then
      if (row_seen(row - row0)) cycle
      row_seen(row - row0) = .true.
    endif
    n_rows_dir           = n_rows_dir + 1
    rows_dir(n_rows_dir) = row
  enddo

  if (n_ax_dofs > 0) then
    do ife = 1, n_local_elms
      ielm = local_elms(ife)
      do iv = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(iv)
        if (.not. node_list%node(inode)%axis_node) cycle
        do ia = 1, n_ax_dofs
          index_node1 = node_list%node(inode)%index(ax_dofs(ia))
          if ((index_node1 .lt. a_mat%my_ind_min) .or. (index_node1 .gt. a_mat%my_ind_max)) cycle
          do i = 1, n_tor
            row = n_tor * (index_node1 - 1) + (i - 1)
            if ((row >= row0) .and. (row < row_end)) then
              if (row_seen(row - row0)) cycle
              row_seen(row - row0) = .true.
            endif
            n_rows_dir           = n_rows_dir + 1
            rows_dir(n_rows_dir) = row
          enddo
        enddo
      enddo
    enddo
  endif

  ! Collective: every rank calls it, also with n_rows_dir = 0. Rows and
  ! columns zeroed, unit diagonal: SPD on the constrained space.
  PetscCallA(MatZeroRowsColumns(M_w, n_rows_dir, rows_dir, 1.0d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr))

  deallocate(rows_bc, rows_dir, row_seen)
#endif

#undef DFV_MW

end subroutine construct_mass_w_matrix

end module construct_pc_matrix_mod
