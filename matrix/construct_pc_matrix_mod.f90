!> Element assembly of the SF preconditioner's force operator W into a
!! one-variable PETSc BAIJ matrix (no mesh refinement, all toroidal modes on
!! every process).
module construct_pc_matrix_mod
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none
private
public :: construct_force_operator_matrix

contains

!> Zero the rows of a one-variable PC matrix at the boundary DOFs the model
!! constrains (boundary type 1: value and ds, 2: value and dt, 3: all three;
!! modes with is_freebound skipped), so that adding it to a block keeps that
!! block's boundary equations.
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

  ! collective; diagonal 0, duplicates are harmless
  PetscCallA(MatZeroRows(pc_mat, n_rows_to_zero, rows_to_zero, 0.0d0, PETSC_NULL_VEC, PETSC_NULL_VEC, ierr))

  deallocate(rows_to_zero)
#endif

end subroutine zero_bc_rows_pc_matrix

!> Assemble the force operator W that is added to B_22 (one variable, block
!! size n_tor) on the linearisation state. With W_vv, W_vu, W_uv present
!! (model600, v_par stages b0 / b1) the same pass assembles v_par's
!! sound-wave Schur terms.
subroutine construct_force_operator_matrix(my_id, local_elms, n_local_elms, a_mat, mhd_sim, W_force, &
                                           W_vv, W_vu, W_uv)

  use mod_pc_elt_matrix_force_fft, only: pc_elt_matrix_force_fft, pc_elt_matrix_force_pfafem
  use mod_parameters,  only: n_tor, n_degrees, n_vertex_max, var_u, var_Vpar
  use data_structure,  only: type_SP_MATRIX, type_element, type_node
  use mod_simulation_data, only: type_MHD_SIM
  use nodes_elements
  use phys_module,     only: eliminate_boundary_dofs, use_pfafem, refinement
  use mod_pfafem,      only: pfafem_cache, pfafem_el_lo, pfafem_el_hi
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

#define NBV (n_vertex_max*n_degrees)
#define DFV_CM (n_tor*NBV)

  integer :: my_ind_min, my_ind_max
  ! per thread (large derived types stay off the thread stacks)
  type(type_element), allocatable :: element_thr(:)
  type(type_node),    allocatable :: nodes_thr(:,:)
  real*8, allocatable :: ELM(:,:), ELMv(:,:,:), buf(:)
  real*8, allocatable :: W_vp(:,:,:,:,:), WV_vp(:,:,:,:,:,:), ebuf_pf(:), ebuf_v(:,:)
  integer :: idx(NBV), rsel(NBV), nr
  logical :: do_v, wv_nz(3)
  integer :: ife, ielm, iv, q, b, t, nthreads
  integer :: n_u, n_vp, cap_u, cap_vp, u1, u2, ivp, io1, io2, inode1, inode2, r_pf, c_pf
  integer(8) :: n_ebuf, base_off8
#ifdef USE_PETSC
  PetscErrorCode :: ierr
  PetscInt :: idxm(NBV), idxn(NBV), nr_p, nc_p, key_col
  PetscInt, allocatable :: idxm_u(:), idxn_u(:), raw_col(:)
  integer,  allocatable :: raw_ivp(:), raw_io2(:), raw_slot(:)
  integer :: rsel_p(n_degrees), n_raw_c, k_c, j_c, key_ivp, key_io2, r_idx, c_slot
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

  if (use_pfafem) then
    if (pfafem_el_hi >= pfafem_el_lo) then
      cap_u  = 16
      cap_vp = 128
      do ielm = pfafem_el_lo, pfafem_el_hi
        if (refinement .and. (element_list%element(ielm)%n_sons /= 0)) cycle
        cap_u  = max(cap_u,  pfafem_cache(ielm)%n_unodes)
        cap_vp = max(cap_vp, pfafem_cache(ielm)%n_vpairs)
      end do

      !$omp parallel default(shared) &
      !$omp   private(ielm, b, n_u, n_vp, u1, u2, ivp, io1, io2, inode1, inode2, r_pf, c_pf, &
      !$omp           wv_nz, n_ebuf, base_off8, W_vp, WV_vp, ebuf_pf, ebuf_v &
#ifdef USE_PETSC
      !$omp           , ierr, nr_p, nc_p, key_col, idxm_u, idxn_u, raw_col &
      !$omp           , raw_ivp, raw_io2, raw_slot, rsel_p, n_raw_c, k_c, j_c, key_ivp, key_io2, r_idx, c_slot &
#endif
      !$omp          )
      allocate(W_vp(n_tor, n_tor, n_degrees, n_degrees, cap_vp))
      n_ebuf = int(cap_u, 8) * int(n_degrees * n_degrees, 8) * int(n_tor * n_tor, 8)
      allocate(ebuf_pf(n_ebuf))
      if (do_v) then
        allocate(WV_vp(n_tor, n_tor, n_degrees, n_degrees, cap_vp, 3))
        allocate(ebuf_v(n_ebuf, 3))
      else
        allocate(WV_vp(1, 1, 1, 1, 1, 1))
        allocate(ebuf_v(1, 1))
      end if
#ifdef USE_PETSC
      allocate(idxm_u(n_degrees), idxn_u(n_degrees * cap_u), raw_col(n_degrees * cap_u))
      allocate(raw_ivp(n_degrees * cap_u), raw_io2(n_degrees * cap_u), raw_slot(n_degrees * cap_u))
#endif

      !$omp do schedule(dynamic, 1)
      do ielm = pfafem_el_lo, pfafem_el_hi
        if (refinement .and. (element_list%element(ielm)%n_sons /= 0)) cycle
        n_u  = pfafem_cache(ielm)%n_unodes
        n_vp = pfafem_cache(ielm)%n_vpairs
        if (n_u <= 0 .or. n_vp <= 0) cycle

        if (do_v) then
          call pc_elt_matrix_force_pfafem(node_list, element_list, ielm, &
                                          W_vp(:, :, :, :, 1:n_vp), WV_vp=WV_vp(:, :, :, :, 1:n_vp, :))
          do b = 1, 3
            wv_nz(b) = any(WV_vp(:, :, :, :, 1:n_vp, b) /= 0.d0)
          end do
        else
          call pc_elt_matrix_force_pfafem(node_list, element_list, ielm, &
                                          W_vp(:, :, :, :, 1:n_vp))
          wv_nz = .false.
        end if

#ifdef USE_PETSC
        do u1 = 1, n_u
          inode1 = pfafem_cache(ielm)%unodes(u1)
          nr_p   = 0
          do io1 = 1, n_degrees
            if (node_list%node(inode1)%index(io1) <= 0) cycle
            nr_p = nr_p + 1
            rsel_p(nr_p) = io1
            idxm_u(nr_p) = node_list%node(inode1)%index(io1) - 1
          end do
          if (nr_p == 0) cycle

          n_raw_c = 0
          do u2 = 1, n_u
            ivp = pfafem_cache(ielm)%pair_map(u1, u2)
            if (ivp <= 0) cycle
            inode2 = pfafem_cache(ielm)%unodes(u2)
            do io2 = 1, n_degrees
              if (node_list%node(inode2)%index(io2) <= 0) cycle
              n_raw_c = n_raw_c + 1
              raw_col(n_raw_c) = node_list%node(inode2)%index(io2) - 1
              raw_ivp(n_raw_c) = ivp
              raw_io2(n_raw_c) = io2
            end do
          end do
          if (n_raw_c == 0) cycle

          do k_c = 2, n_raw_c
            key_col = raw_col(k_c)
            key_ivp = raw_ivp(k_c)
            key_io2 = raw_io2(k_c)
            j_c = k_c - 1
            do while (j_c >= 1)
              if (raw_col(j_c) <= key_col) exit
              raw_col(j_c + 1) = raw_col(j_c)
              raw_ivp(j_c + 1) = raw_ivp(j_c)
              raw_io2(j_c + 1) = raw_io2(j_c)
              j_c = j_c - 1
            end do
            raw_col(j_c + 1) = key_col
            raw_ivp(j_c + 1) = key_ivp
            raw_io2(j_c + 1) = key_io2
          end do

          nc_p = 1
          idxn_u(1)   = raw_col(1)
          raw_slot(1) = 1
          do k_c = 2, n_raw_c
            if (raw_col(k_c) /= idxn_u(nc_p)) then
              nc_p = nc_p + 1
              idxn_u(nc_p) = raw_col(k_c)
            end if
            raw_slot(k_c) = nc_p
          end do

          n_ebuf = int(nr_p, 8) * int(nc_p, 8) * int(n_tor * n_tor, 8)
          ebuf_pf(1_8 : n_ebuf) = 0.d0
          do b = 1, 3
            if (wv_nz(b)) ebuf_v(1_8 : n_ebuf, b) = 0.d0
          end do

          do k_c = 1, n_raw_c
            ivp    = raw_ivp(k_c)
            io2    = raw_io2(k_c)
            c_slot = raw_slot(k_c)
            do r_idx = 1, int(nr_p)
              io1 = rsel_p(r_idx)
              do r_pf = 1, n_tor
                base_off8 = int((r_idx - 1) * n_tor + (r_pf - 1), 8) * int(nc_p * n_tor, 8) &
                          + int((c_slot - 1) * n_tor, 8)
                do c_pf = 1, n_tor
                  ebuf_pf(base_off8 + c_pf) = ebuf_pf(base_off8 + c_pf) + W_vp(r_pf, c_pf, io1, io2, ivp)
                end do
                do b = 1, 3
                  if (wv_nz(b)) then
                    do c_pf = 1, n_tor
                      ebuf_v(base_off8 + c_pf, b) = ebuf_v(base_off8 + c_pf, b) + WV_vp(r_pf, c_pf, io1, io2, ivp, b)
                    end do
                  end if
                end do
              end do
            end do
          end do

          !$omp critical (pc_force_insert)
          PetscCallA(MatSetValuesBlocked(W_force, nr_p, idxm_u(1:nr_p), nc_p, idxn_u(1:nc_p), ebuf_pf(1_8:n_ebuf), ADD_VALUES, ierr))
          do b = 1, 3
            if (wv_nz(b)) then
              PetscCallA(MatSetValuesBlocked(Wv(b), nr_p, idxm_u(1:nr_p), nc_p, idxn_u(1:nc_p), ebuf_v(1_8:n_ebuf, b), ADD_VALUES, ierr))
            end if
          end do
          !$omp end critical (pc_force_insert)
        end do
#endif
      end do
      !$omp end do
#ifdef USE_PETSC
      deallocate(idxm_u, idxn_u, raw_col, raw_ivp, raw_io2, raw_slot)
#endif
      deallocate(W_vp, WV_vp, ebuf_pf, ebuf_v)
      !$omp end parallel
    end if
  else
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
      call pc_elt_matrix_force_fft(element_thr(t), nodes_thr(:, t), ELM, ELM_v=ELMv)
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
  end if
  deallocate(element_thr, nodes_thr)

#ifdef USE_PETSC
  PetscCallA(MatAssemblyBegin(W_force, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd  (W_force, MAT_FINAL_ASSEMBLY, ierr))
  ! W is added to blocks whose boundary rows are Dirichlet rows: its own
  ! entries there would break the constraint
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

end subroutine construct_force_operator_matrix

end module construct_pc_matrix_mod
