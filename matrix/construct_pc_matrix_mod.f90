!> Element assembly of the physics preconditioner's one-variable PETSc BAIJ
!! matrices: the SF sweep's force operator W, the vorticity-space Schur PC's
!! inertia M_rho, and the latter's boundary diagonal D_bc (no mesh
!! refinement, all toroidal modes on every process).
module construct_pc_matrix_mod
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none
private
public :: construct_pc_element_matrices, construct_bc_diagonal

contains

!> Zero the rows of a one-variable PC matrix at the boundary DOFs the model
!! constrains (boundary type 1: value and ds, 2: value and dt, 3: all three;
!! modes with is_freebound skipped) and put diag on their diagonal. diag = 0
!! keeps the boundary equations of a block the matrix is added to; diag /= 0
!! makes them Dirichlet rows of the matrix itself.
subroutine zero_bc_rows_pc_matrix(pc_mat, var_index, diag, local_elms, n_local_elms, &
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
  real*8,  intent(in) :: diag
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

  ! a corner node constrains (n_order+1)/2 DOFs along each of its two directions
  cap = n_local_elms * n_vertex_max * (n_order + 1) * n_tor
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

  ! collective; duplicates are harmless
  PetscCallA(MatZeroRows(pc_mat, n_rows_to_zero, rows_to_zero, diag, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr))

  deallocate(rows_to_zero)
#endif

end subroutine zero_bc_rows_pc_matrix

!> Assemble the element-level matrices of the physics PC (one variable,
!! block size n_tor) on the linearisation state, in one pass: the force
!! operator W that is added to B_22 where W_force is present, with v_par's
!! sound-wave Schur terms where W_vv, W_vu, W_uv are present too (model600,
!! v_par stages b0 / b1); the inertia M_rho where present, with d_bc on the
!! diagonal of its rows at u's constrained DOFs; the resistive line bending's
!! M_eta and K_eta where present (vs_lumped, -vs_bending resistive), and K_1
!! with them for the preconditioner's self-check. M_eta, K_eta and K_1 keep
!! their boundary rows: the preconditioner masks them in its own layout.
subroutine construct_pc_element_matrices(my_id, local_elms, n_local_elms, a_mat, mhd_sim, W_force, &
                                         W_vv, W_vu, W_uv, M_rho, d_bc, M_eta, K_eta, K_1)

  use mod_pc_elt_matrix_force_fft, only: pc_elt_matrix_force_fft, pc_elt_matrix_mass_rho, pc_elt_matrix_eta
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
  Mat, intent(inout), optional :: W_force
  Mat, intent(inout), optional :: W_vv, W_vu, W_uv
  Mat, intent(inout), optional :: M_rho
  Mat, intent(inout), optional :: M_eta, K_eta, K_1
#endif
  real*8, intent(in), optional :: d_bc

#define NBV (n_vertex_max*n_degrees)
#define DFV_CM (n_tor*NBV)

  integer :: my_ind_min, my_ind_max
  ! per thread (large derived types stay off the thread stacks)
  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  real*8, allocatable :: ELM(:,:), ELMv(:,:,:), ELMe(:,:,:), buf(:)
  integer :: idx(NBV), rsel(NBV), nr
  logical :: do_w, do_v, do_m, do_e, do_h
  integer :: ife, ielm, iv, q, b, t, nthreads
#ifdef USE_PETSC
  PetscErrorCode :: ierr
  PetscInt :: idxm(NBV), idxn(NBV)
  Mat :: Wv(3), We(3)
#endif

  my_ind_min = a_mat%index_min(my_id+1)
  my_ind_max = a_mat%index_max(my_id+1)

  do_w = .false.; do_v = .false.; do_m = .false.; do_e = .false.; do_h = .false.
#ifdef USE_PETSC
  do_w = present(W_force)
  do_v = do_w .and. present(W_vv) .and. present(W_vu) .and. present(W_uv)
  if (do_v) then
    Wv(1) = W_vv; Wv(2) = W_vu; Wv(3) = W_uv
  endif
  do_m = present(M_rho) .and. present(d_bc)
  do_e = present(M_eta) .and. present(K_eta)
  do_h = do_e .and. present(K_1)
  if (do_e) then
    We(1) = M_eta; We(2) = K_eta
  endif
  if (do_h) We(3) = K_1
#endif

  nthreads = 1
  !$ nthreads = omp_get_max_threads()
  allocate(element_thr(nthreads), nodes_thr(n_vertex_max, nthreads))

  !$omp parallel default(shared) &
  !$omp   private(ife, ielm, iv, q, b, t, ELM, ELMv, ELMe, buf, idx, rsel, nr &
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
  if (do_e) then
    allocate(ELMe(DFV_CM, DFV_CM, 3))
  else
    allocate(ELMe(1, 1, 1))
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
#endif

    if (do_w) then
      if (do_v) then
        call pc_elt_matrix_force_fft(element_thr(t), nodes_thr(:, t), ELM, ELM_v=ELMv)
      else
        call pc_elt_matrix_force_fft(element_thr(t), nodes_thr(:, t), ELM)
      endif
#ifdef USE_PETSC
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
    endif

    if (do_m) then
      call pc_elt_matrix_mass_rho(element_thr(t), nodes_thr(:, t), ELM)
#ifdef USE_PETSC
      if (nr > 0) then
        call pack_rows(ELM, rsel, nr, buf)
        !$omp critical (pc_force_insert)
        PetscCallA(MatSetValuesBlocked(M_rho, nr, idxm, NBV, idxn, buf, ADD_VALUES, ierr))
        !$omp end critical (pc_force_insert)
      endif
#endif
    endif

    if (do_e) then
      if (do_h) then
        call pc_elt_matrix_eta(element_thr(t), nodes_thr(:, t), ELMe(:,:,1), ELMe(:,:,2), ELMe(:,:,3))
      else
        call pc_elt_matrix_eta(element_thr(t), nodes_thr(:, t), ELMe(:,:,1), ELMe(:,:,2))
      endif
#ifdef USE_PETSC
      if (nr > 0) then
        !$omp critical (pc_force_insert)
        do b = 1, merge(3, 2, do_h)
          call pack_rows(ELMe(:,:,b), rsel, nr, buf)
          PetscCallA(MatSetValuesBlocked(We(b), nr, idxm, NBV, idxn, buf, ADD_VALUES, ierr))
        enddo
        !$omp end critical (pc_force_insert)
      endif
#endif
    endif
  enddo
  !$omp end do
  deallocate(ELM, ELMv, ELMe, buf)
  !$omp end parallel
  deallocate(element_thr, nodes_thr)

#ifdef USE_PETSC
  if (do_w) then
    PetscCallA(MatAssemblyBegin(W_force, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd  (W_force, MAT_FINAL_ASSEMBLY, ierr))
    ! W is added to blocks whose boundary rows are Dirichlet rows: its own
    ! entries there would break the constraint
    if (eliminate_boundary_dofs) then
      call zero_bc_rows_pc_matrix(W_force, var_u, 0.d0, local_elms, n_local_elms, &
                                  a_mat%my_ind_min, a_mat%my_ind_max)
    endif
  endif
  if (do_v) then
    do b = 1, 3
      PetscCallA(MatAssemblyBegin(Wv(b), MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd  (Wv(b), MAT_FINAL_ASSEMBLY, ierr))
    enddo
    if (eliminate_boundary_dofs) then         ! by each block's row variable
      call zero_bc_rows_pc_matrix(Wv(1), var_Vpar, 0.d0, local_elms, n_local_elms, &
                                  a_mat%my_ind_min, a_mat%my_ind_max)
      call zero_bc_rows_pc_matrix(Wv(2), var_Vpar, 0.d0, local_elms, n_local_elms, &
                                  a_mat%my_ind_min, a_mat%my_ind_max)
      call zero_bc_rows_pc_matrix(Wv(3), var_u, 0.d0, local_elms, n_local_elms, &
                                  a_mat%my_ind_min, a_mat%my_ind_max)
    endif
  endif
  if (do_m) then
    PetscCallA(MatAssemblyBegin(M_rho, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd  (M_rho, MAT_FINAL_ASSEMBLY, ierr))
    ! Dirichlet rows in both boundary modes: the row of u is replaced, not kept
    call zero_bc_rows_pc_matrix(M_rho, var_u, d_bc, local_elms, n_local_elms, &
                                a_mat%my_ind_min, a_mat%my_ind_max)
  endif
  if (do_e) then
    do b = 1, merge(3, 2, do_h)
      PetscCallA(MatAssemblyBegin(We(b), MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd  (We(b), MAT_FINAL_ASSEMBLY, ierr))
    enddo
  endif
#endif

contains

  !> buf = the rows rsel(1:nr) of the element matrix E, row-major by scalar
  !! row (MatSetValuesBlocked's default orientation). Everything by argument:
  !! the caller's copies are OpenMP-private, host association would not see them.
  subroutine pack_rows(E, rsel, nr, buf)
    real*8, intent(in)  :: E(DFV_CM, DFV_CM)
    integer, intent(in) :: rsel(:), nr
    real*8, intent(out) :: buf(:)
    integer :: r, j, o
    o = 0
    do r = 1, nr
      do j = 1, n_tor
        buf(o + 1 : o + DFV_CM) = E(n_tor * (rsel(r) - 1) + j, :)
        o = o + DFV_CM
      enddo
    enddo
  end subroutine pack_rows

end subroutine construct_pc_element_matrices

!> D_bc: d on the diagonal of every row at u's constrained DOFs, zero on the
!! other diagonal blocks (one variable, block size n_tor, created by the
!! caller with A's block rows). Geometry only: built once.
subroutine construct_bc_diagonal(local_elms, n_local_elms, a_mat, D_bc, d)

  use mod_parameters, only: n_tor, var_u
  use data_structure, only: type_SP_MATRIX

  implicit none

  integer, pointer,     intent(in) :: local_elms(:)
  integer,              intent(in) :: n_local_elms
  type(type_SP_MATRIX), intent(in) :: a_mat
#ifdef USE_PETSC
  Mat, intent(inout) :: D_bc
#endif
  real*8, intent(in) :: d

#ifdef USE_PETSC
  PetscErrorCode :: ierr
  PetscInt :: ib(1)
  real*8   :: zero_blk(n_tor * n_tor)
  integer  :: i

  zero_blk = 0.d0
  do i = a_mat%my_ind_min, a_mat%my_ind_max
    ib(1) = i - 1
    PetscCallA(MatSetValuesBlocked(D_bc, 1, ib, 1, ib, zero_blk, INSERT_VALUES, ierr))
  enddo
  PetscCallA(MatAssemblyBegin(D_bc, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (D_bc, MAT_FINAL_ASSEMBLY, ierr))
  call zero_bc_rows_pc_matrix(D_bc, var_u, d, local_elms, n_local_elms, &
                              a_mat%my_ind_min, a_mat%my_ind_max)
#endif

end subroutine construct_bc_diagonal

end module construct_pc_matrix_mod
