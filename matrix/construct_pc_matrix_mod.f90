module construct_pc_matrix_mod
!----------------------------------------------------------------
! Element assembly of the SF preconditioner's force operator W into a
! 1-variable PETSc BAIJ matrix (no mesh refinement, all toroidal modes
! on every process).
!----------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none
private
public :: construct_force_operator_matrix, construct_peel_matrices

contains

!--------------------------------------------------------------------
!> Zero out the rows of a 1-variable PC matrix that correspond to
!! constrained boundary DOFs.
!!
!! Used for matrices that are ADDED to a B-block which already has BCs
!! enforced (W, added to B_22). Setting those rows to exactly zero keeps
!! B's boundary equations untouched; a ZBIG diagonal here would double
!! B's ZBIG diagonal.
!!
!! Constrained DOFs per boundary type (matching the model's boundary_conditions):
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
  PetscErrorCode        :: ierr

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
  PetscCallA(MatZeroRows(pc_mat, n_rows_to_zero, rows_to_zero, 0.0d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr))

  deallocate(rows_to_zero)
#endif

end subroutine zero_bc_rows_pc_matrix

!--------------------------------------------------------------------
!> Assemble the force operator W, the matrix ADDED to B_22 (one variable,
!! block size n_tor), with its constrained boundary rows zeroed. It is built
!! on the linearisation state (mhd_sim), not on the Jacobian.
!!
!! With W_vv, W_vu, W_uv present (model600 with_vpar, stages b0 / b1) the
!! same pass assembles v_par's sound-wave Schur terms (the element routine's
!! ELM_v blocks 1 (v_par, v_par), 2 (v_par, u), 3 (u, v_par)).
!--------------------------------------------------------------------
subroutine construct_force_operator_matrix(my_id, local_elms, n_local_elms, a_mat, mhd_sim, W_force, &
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
  Mat, intent(inout), optional :: W_vv, W_vu, W_uv
#endif
  real*8, intent(in), optional :: gamma_w    !< v_par's W terms: their adiabatic index
  logical, intent(in), optional :: vu_full   !< ... W_vu with its R [p0,u] half

#define NBV (n_vertex_max*n_degrees)
#define DFV_CM (n_tor*NBV)

  integer :: my_ind_min, my_ind_max
  ! per thread (large derived types stay off the thread stacks)
  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  real*8, allocatable :: ELM(:,:), ELMv(:,:,:), buf(:)
  integer :: idx(NBV), rsel(NBV), nr
  logical :: do_v
  integer :: ife, ielm, iv, q, b, t, nthreads
#ifdef USE_PETSC
  PetscErrorCode :: ierr
  PetscInt :: idxm(NBV), idxn(NBV)
  Mat :: Wv(3)
#endif

  my_ind_min = a_mat%index_min(my_id+1)
  my_ind_max = a_mat%index_max(my_id+1)

  do_v = .false.
#ifdef USE_PETSC
  do_v = present(W_vv) .and. present(W_vu) .and. present(W_uv)
  if (do_v) then
    Wv(1) = W_vv; Wv(2) = W_vu; Wv(3) = W_uv
  endif
#endif

  nthreads = 1
  !$ nthreads = omp_get_max_threads()
  allocate(element_thr(nthreads), nodes_thr(n_vertex_max, nthreads))

  !$omp parallel default(shared) &
  !$omp   private(ife, ielm, iv, q, b, t, ELM, ELMv, buf, idx, rsel, nr &
#ifdef USE_PETSC
  !$omp           , ierr, idxm, idxn &
#endif
  !$omp          )
  allocate(ELM(DFV_CM, DFV_CM), buf(DFV_CM * DFV_CM))
  if (do_v) then
    allocate(ELMv(DFV_CM, DFV_CM, 3))
  else
    allocate(ELMv(1, 1, 1))
  endif

  t = 1
  !$ t = omp_get_thread_num() + 1

  !$omp do schedule(runtime)
  do ife = 1, n_local_elms
    ielm = local_elms(ife)
    element_thr(t) = element_list%element(ielm)
    do iv = 1, n_vertex_max
      nodes_thr(iv, t) = node_list%node(element_thr(t)%vertex(iv))
    enddo

    if (do_v) then
      call pc_elt_matrix_force_fft(element_thr(t), nodes_thr(:, t), ELM, &
                                   ELM_v=ELMv, gamma_w=gamma_w, vu_full=vu_full)
    else
      call pc_elt_matrix_force_fft(element_thr(t), nodes_thr(:, t), ELM)
    endif

#ifdef USE_PETSC
    ! the element's (vertex, degree) block rows this rank owns, all columns
    nr = 0
    do q = 1, NBV
      idx(q) = nodes_thr((q - 1) / n_degrees + 1, t)%index(mod(q - 1, n_degrees) + 1)
      idxn(q) = idx(q) - 1
      if (idx(q) < my_ind_min .or. idx(q) > my_ind_max) cycle
      nr = nr + 1
      rsel(nr) = q
      idxm(nr) = idx(q) - 1
    enddo
    if (nr > 0) then
      call pack_rows(ELM, rsel, nr, buf)
      !$omp critical (pc_force_insert)
      PetscCallA(MatSetValuesBlocked(W_force, nr, idxm, NBV, idxn, buf, ADD_VALUES, ierr))
      do b = 1, merge(3, 0, do_v)
        call pack_rows(ELMv(:,:,b), rsel, nr, buf)
        PetscCallA(MatSetValuesBlocked(Wv(b), nr, idxm, NBV, idxn, buf, ADD_VALUES, ierr))
      enddo
      !$omp end critical (pc_force_insert)
    endif
#endif
  enddo
  !$omp end do
  deallocate(ELM, ELMv, buf)
  !$omp end parallel
  deallocate(element_thr, nodes_thr)

#ifdef USE_PETSC
  PetscCallA(MatAssemblyBegin(W_force, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (W_force, MAT_FINAL_ASSEMBLY, ierr))
  ! W is ADDED to blocks whose boundary rows are [ZBIG, 0, ...]: its own
  ! entries there would destroy the constraint (5.9x the iterations at
  ! tstep 10 although only 0.4% of ||W||), so they are zeroed.
  if (eliminate_boundary_dofs) then
    call zero_bc_rows_pc_matrix(W_force, var_u, local_elms, n_local_elms, &
                                a_mat%my_ind_min, a_mat%my_ind_max)
  endif
  if (do_v) then
    do b = 1, 3
      PetscCallA(MatAssemblyBegin(Wv(b), MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd  (Wv(b), MAT_FINAL_ASSEMBLY, ierr))
    enddo
    if (eliminate_boundary_dofs) then         ! by each block's row variable
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

!> buf = the rows rsel(1:nr) of the element matrix E, row-major by scalar
!! row (MatSetValuesBlocked's default orientation). Everything by argument:
!! the caller's copies are OpenMP-private, host association would not see them.
subroutine pack_rows(E, rsel, nr, buf)
  use mod_parameters, only: n_tor, n_degrees, n_vertex_max
  real*8, intent(in)  :: E(n_tor*n_vertex_max*n_degrees, n_tor*n_vertex_max*n_degrees)
  integer, intent(in) :: rsel(:), nr
  real*8, intent(out) :: buf(:)
  integer :: r, j, o, nd
  nd = n_tor * n_vertex_max * n_degrees
  o = 0
  do r = 1, nr
    do j = 1, n_tor
      buf(o + 1 : o + nd) = E(n_tor * (rsel(r) - 1) + j, :)
      o = o + nd
    enddo
  enddo
end subroutine pack_rows

!--------------------------------------------------------------------
!> Assemble the peeled-Schur operators of pair_w (mod_petsc_pc_sf_peel), one
!! variable (u) each, block size n_tor: Pm(k) = the element routine's
!! ELM(:,:,k) (opz K_rho, W_comp, M_rho, K_nu). Boundary rows are left as
!! assembled; the peel arm fixes its Dirichlet DOFs itself.
!--------------------------------------------------------------------
subroutine construct_peel_matrices(local_elms, n_local_elms, a_mat, Pm)

  use mod_pc_elt_matrix_force_fft, only: pc_elt_matrix_peel_fft, PEEL_NOP, peel_ebe_axis_mask
  use mod_parameters,  only: n_tor, n_degrees, n_vertex_max
  use data_structure,  only: type_SP_MATRIX, type_element, type_node
  use nodes_elements
  use omp_lib

  implicit none

  integer, pointer,     intent(in) :: local_elms(:)
  integer,              intent(in) :: n_local_elms
  type(type_SP_MATRIX), intent(in) :: a_mat
#ifdef USE_PETSC
  Mat, intent(inout) :: Pm(PEEL_NOP)
#endif

  integer :: my_ind_min, my_ind_max
  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  real*8, allocatable :: ELM(:,:,:), buf(:)
  integer :: idx(NBV), rsel(NBV), nr
  integer :: ife, ielm, iv, q, b, t, nthreads
#ifdef USE_PETSC
  PetscErrorCode :: ierr
  PetscInt :: idxm(NBV), idxn(NBV)
#endif

  my_ind_min = a_mat%my_ind_min
  my_ind_max = a_mat%my_ind_max
#ifdef USE_PETSC
  block
    PetscInt :: v
    PetscBool :: set
    v = 0
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-sf_peel_ebe_axis", v, set, ierr)
    peel_ebe_axis_mask = (v /= 0)
  end block
#endif

  nthreads = 1
  !$ nthreads = omp_get_max_threads()
  allocate(element_thr(nthreads), nodes_thr(n_vertex_max, nthreads))

  !$omp parallel default(shared) &
  !$omp   private(ife, ielm, iv, q, b, t, ELM, buf, idx, rsel, nr &
#ifdef USE_PETSC
  !$omp           , ierr, idxm, idxn &
#endif
  !$omp          )
  allocate(ELM(DFV_CM, DFV_CM, PEEL_NOP), buf(DFV_CM * DFV_CM))
  t = 1
  !$ t = omp_get_thread_num() + 1

  !$omp do schedule(runtime)
  do ife = 1, n_local_elms
    ielm = local_elms(ife)
    element_thr(t) = element_list%element(ielm)
    do iv = 1, n_vertex_max
      nodes_thr(iv, t) = node_list%node(element_thr(t)%vertex(iv))
    enddo
    call pc_elt_matrix_peel_fft(element_thr(t), nodes_thr(:, t), ELM)
#ifdef USE_PETSC
    nr = 0
    do q = 1, NBV
      idx(q) = nodes_thr((q - 1) / n_degrees + 1, t)%index(mod(q - 1, n_degrees) + 1)
      idxn(q) = idx(q) - 1
      if (idx(q) < my_ind_min .or. idx(q) > my_ind_max) cycle
      nr = nr + 1
      rsel(nr) = q
      idxm(nr) = idx(q) - 1
    enddo
    if (nr > 0) then
      !$omp critical (pc_peel_insert)
      do b = 1, PEEL_NOP
        call pack_rows(ELM(:,:,b), rsel, nr, buf)
        PetscCallA(MatSetValuesBlocked(Pm(b), nr, idxm, NBV, idxn, buf, ADD_VALUES, ierr))
      enddo
      !$omp end critical (pc_peel_insert)
    endif
#endif
  enddo
  !$omp end do
  deallocate(ELM, buf)
  !$omp end parallel
  deallocate(element_thr, nodes_thr)

#ifdef USE_PETSC
  do b = 1, PEEL_NOP
    PetscCallA(MatAssemblyBegin(Pm(b), MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd  (Pm(b), MAT_FINAL_ASSEMBLY, ierr))
  enddo
#endif

end subroutine construct_peel_matrices

end module construct_pc_matrix_mod
