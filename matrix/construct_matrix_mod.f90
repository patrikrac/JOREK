module construct_matrix_mod

use mod_parameters, only : n_var, n_order, n_degrees_1d
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
use petsc
#endif
implicit none

public :: construct_matrix

contains

!> Construct the main matrix from the contributions of the Bezier elements.
!!
!! The element contributions are determined by element_matrix(_fft). Additional
!! contributions from boundary conditions and the free boundary extension are
!! added by external routine calls.
subroutine construct_matrix(mhd_sim, local_elms, n_local_elms, a_mat, rhs_vec, harmonic_matrix)
  use tr_module
  use phys_module
  use data_structure, only: type_SP_MATRIX, type_RHS, type_element, type_node, thread_struct
  use mod_simulation_data, only: type_MHD_SIM
  use vacuum, only: sr
  use nodes_elements
  use mpi_mod
  use omp_lib
  use mod_axis_treatment, only: transform_basis_for_axis_element, penalize_dof_on_axis
  use mod_boundary_conditions, only: boundary_conditions
  use mod_fix_axis_nodes, only: fix_nodes_on_axis
  use vacuum_response, only: vacuum_boundary_integral
  use global_distributed_matrix, only: global_matrix_structure_vacuum
#ifdef USE_PETSC
  use mod_petsc, only: petsc_create_matrix
#endif
  implicit none

#include "r3_info.h"

  ! --- Routine parameters
  logical,              intent(in)    :: harmonic_matrix
  type(type_SP_MATRIX), intent(inout) :: a_mat !< Sparse matrix to be constructed
  type(type_RHS),       intent(inout) :: rhs_vec !< Right-hand side vector to be constructed
  type(type_MHD_SIM),   intent(in)    :: mhd_sim !< MHD simulation data structure containing a lot of information about the simulation
  integer, dimension(:), pointer      :: local_elms !< List of local element indices in the simulation
  integer                             :: n_local_elms ! Number of local elements in the simulation

  ! --- Internal variables
  type (type_element)               :: element
  type (type_node)                  :: nodes(n_vertex_max), aux_nodes(n_vertex_max)
  type (type_element)               :: element_father
  type (type_node)                  :: nodes_father(n_vertex_max)
  real*8,              allocatable  :: rhs_local(:) !< Local right-hand side vector for the current process

  integer                           :: ife, ielm
  integer                           :: i_v(n_var)
  integer, allocatable              :: i_harm(:)
  integer                           :: ierr
  integer                           :: omp_nthreads, omp_tid, n_tor_local
  integer                           :: my_ind_min, my_ind_max
  integer                           :: node_out(n_vertex_max)
  integer                           :: my_id
#ifdef USE_PETSC
  PetscErrorCode                    :: petsc_ierr
#endif
  integer                           :: xcase2
  real*8                            :: R_axis
  real*8                            :: Z_axis
  real*8                            :: psi_axis
  real*8                            :: psi_bnd
  real*8                            :: R_xpoint(2)
  real*8                            :: Z_xpoint(2)
  real*8                            :: psi_xpoint(2)
  logical                           :: xpoint2

  ! --- Timing call
  call r3_info_begin (r3_info_index_0, 'construct_matrix')

  ! --- Extract parameters from mhd_sim
  my_id           = mhd_sim%my_id
  xpoint2         = mhd_sim%es%xpoint
  xcase2          = mhd_sim%es%xcase
  R_axis          = mhd_sim%es%R_axis
  Z_axis          = mhd_sim%es%Z_axis
  psi_axis        = mhd_sim%es%psi_axis
  psi_bnd         = mhd_sim%es%psi_bnd
  R_xpoint(1:2)   = mhd_sim%es%R_xpoint(1:2)
  Z_xpoint(1:2)   = mhd_sim%es%Z_xpoint(1:2)
  psi_xpoint(1:2) = mhd_sim%es%psi_xpoint(1:2)

  ! --- Printout
  if (mhd_sim%my_id .eq. 0) then
    if(.NOT. harmonic_matrix) then
      write(*,*) '****************************************'
      write(*,*) '*       construct global matrix        *'
      write(*,*) '*       using CPU                      *'
      write(*,*) '****************************************'
    else
      write(*,*) '****************************************'
      write(*,*) '*        construct PC matrix           *'
      write(*,*) '****************************************'
    endif
  endif

  ! --- Memory allocation of temporary buffers
  call new_thread_buffers() 

  ! --- Allocation/Reallocation of the sparse matrix and right-hand side vector
#ifdef USE_PETSC
  if (.not. harmonic_matrix) then !TODO: Might be unnessecary ... constant protection by harmonic_matrix...
    ! Direct PETSc assembly: skip irn/jcn/val, use PETSc MPIBAIJ matrix
    if (.not. a_mat%petsc_assembled) then
      call petsc_create_matrix(a_mat%petsc_A, a_mat)
      a_mat%petsc_assembled = .true.
    else
      call MatZeroEntries(a_mat%petsc_A, petsc_ierr)
    endif
  else
#endif
  if (associated(a_mat%irn)) call tr_deallocatep(a_mat%irn, "irn", CAT_DMATRIX)
  if (associated(a_mat%jcn)) call tr_deallocatep(a_mat%jcn, "jcn", CAT_DMATRIX)
  if (associated(a_mat%val)) call tr_deallocatep(a_mat%val, "val", CAT_DMATRIX)

  call tr_allocatep(a_mat%irn, Int1, a_mat%nnz, "irn", CAT_DMATRIX)
  call tr_allocatep(a_mat%jcn, Int1, a_mat%nnz, "jcn", CAT_DMATRIX)
  call tr_allocatep(a_mat%val, Int1, a_mat%nnz, "val", CAT_DMATRIX)

  a_mat%irn(1:a_mat%nnz) = 0
  a_mat%jcn(1:a_mat%nnz) = 0
  a_mat%val(1:a_mat%nnz) = 0.0d0
#ifdef USE_PETSC
  endif
#endif

  if (associated(rhs_vec%val)) call tr_deallocatep(rhs_vec%val,"rhs",CAT_DMATRIX)
  call tr_allocatep(rhs_vec%val, Int1, a_mat%ng, "rhs", CAT_DMATRIX)
  rhs_vec%val(:) = 0.0d0 

  call tr_allocate(rhs_local, Int1, a_mat%ng, "rhs_local", CAT_DMATRIX)
  rhs_local  = 0.d0

  if (mhd_sim%freeboundary .and. (mhd_sim%sr_n_tor /= 0 ) ) then
    call global_matrix_structure_vacuum(mhd_sim%node_list, mhd_sim%bnd_node_list, a_mat, i_tor_min=1, i_tor_max=n_tor)
  endif

  ! --- Matrix construction
  my_ind_min = a_mat%index_min(my_id+1)
  my_ind_max = a_mat%index_max(my_id+1)

  n_tor_local = a_mat%i_tor_max - a_mat%i_tor_min + 1

  ! --- Start the OMP parallel region
    !$omp parallel default(none) &
    !$omp   shared(n_local_elms, local_elms, n_tor_local, element_list, node_list, aux_node_list, my_id, xpoint2, xcase2, &
    !$omp           R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, thread_struct, harmonic_matrix, &
    !$omp            a_mat, rhs_local, rhs_vec, &
    !$omp            fix_axis_nodes, treat_axis, my_ind_min, my_ind_max, bc_natural_open, bc_natural_flux, n_tor_fft_thresh, refinement) &
    !$omp   private(ife, ielm, element, element_father, nodes, nodes_father, aux_nodes, node_out, omp_nthreads, omp_tid, i_v, i_harm) 
    
    ! --- Fetch the OMP thread information
#ifdef _OPENMP
    omp_nthreads = omp_get_num_threads()
    omp_tid      = 1 + omp_get_thread_num()
#else
    omp_nthreads = 1
    omp_tid      = 1
#endif
    
    ! --- Initialise i_v and i_harm in case of treat_axis
    if (treat_axis) call initialize_v_and_harm(i_v, i_harm, a_mat%i_tor_min, a_mat%i_tor_max, n_tor_local)

    ! --- Element loop
    !$omp do schedule(runtime)
    do ife = 1, n_local_elms
      ielm = local_elms(ife)

      ! --- Get element and nodes
      call get_element_and_nodes(ielm, element, element_father, nodes, nodes_father, aux_nodes, harmonic_matrix)

      ! --- Build element matrix
      call build_element_matrix(element, nodes, xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, omp_tid, ife, n_local_elms, node_list, a_mat%i_tor_min, a_mat%i_tor_max, aux_nodes, harmonic_matrix)

      ! --- Transform basis functions for axis nodes if required
      if(treat_axis .and. (nodes(1)%axis_node .or. nodes(2)%axis_node .or. nodes(3)%axis_node .or. nodes(4)%axis_node) ) then
        call transform_basis_for_axis_element(nodes, ielm, thread_struct(omp_tid)%ELM, thread_struct(omp_tid)%RHS, i_v, n_var, i_harm, n_tor_local)
      endif

      ! TODO: Implement the diagnostic print_element_rhs subroutine

      ! --- Define element nodes
      call define_element_nodes(ielm, node_out, element, nodes, element_father, nodes_father, omp_tid, harmonic_matrix)

      ! --- Add element matrix to the global matrix
      call add_to_a_mat(element, node_out, a_mat, rhs_local, my_ind_min, my_ind_max, omp_tid)

    enddo !< Element loop
    !$omp end do
    !$omp end parallel

#ifdef USE_PETSC
    if (.not. harmonic_matrix) then
      ! Flush element contributions (ADD_VALUES) before BCs (INSERT_VALUES)
      call MatAssemblyBegin(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, petsc_ierr)
      call MatAssemblyEnd(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, petsc_ierr)
    else
#endif
    ! --- Memory tracking
    call tr_vnorms("cm_A_bef_bc", a_mat%val, a_mat%nnz)
#ifdef USE_PETSC
    endif
#endif

    ! --- Apply boundary conditions.
    call boundary_conditions(my_id, node_list, element_list,  bnd_node_list,local_elms, n_local_elms,  &
                            my_ind_min, my_ind_max, rhs_local, xpoint2, xcase2, R_axis, Z_axis,        & 
                            psi_axis, psi_bnd, R_xpoint, Z_xpoint, psi_xpoint, a_mat)
    
    if (fix_axis_nodes) then
      call fix_nodes_on_axis(node_list, element_list, local_elms, n_local_elms, my_ind_min, my_ind_max, a_mat)
    elseif(treat_axis)then
      call penalize_dof_on_axis(node_list, 4, element_list, local_elms, n_local_elms, my_ind_min, my_ind_max, a_mat)
    endif

#ifdef USE_PETSC
    if (.not. harmonic_matrix) then
      ! Flush BC contributions (INSERT_VALUES) before vacuum (ADD_VALUES)
      call MatAssemblyBegin(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, petsc_ierr)
      call MatAssemblyEnd(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, petsc_ierr)
    else
#endif
    ! --- Memory tracking
    call tr_vnorms("cm_A_aft_bc", a_mat%val, a_mat%nnz)
#ifdef USE_PETSC
    endif
#endif

    ! --- Add vacuum response (boundary integral) for free boundary computations
    if ( freeboundary .and. ( sr%n_tor /= 0 ) ) then
      call vacuum_boundary_integral(my_id, bnd_node_list, node_list, bnd_elm_list, freeboundary_equil, &
                                    resistive_wall, my_ind_min, my_ind_max, rhs_local, tstep, index_now, a_mat)
    endif

#ifdef USE_PETSC
    if (.not. harmonic_matrix) then
      ! Final assembly of PETSc matrix
      call MatAssemblyBegin(a_mat%petsc_A, MAT_FINAL_ASSEMBLY, petsc_ierr)
      call MatAssemblyEnd(a_mat%petsc_A, MAT_FINAL_ASSEMBLY, petsc_ierr)
    endif
#endif

    if ( .not. harmonic_matrix ) then
#ifdef COMPARE_ELEMENT_MATRIX
      ! TODO: Create a subroutine to compare the element matrix with the right-hand side vector
      !call summarise_element_matrix_comparison(a_mat, rhs_vec, my_id, index_now)
#endif
#ifdef NORMTRACE
      ! TODO: Reintroduce normtrace subroutine
      !call normtrace(...)
#endif
    endif
    
  ! --- Collect the right-hand side vector from all processes 
  call MPI_AllReduce(RHS_local,rhs_vec%val,a_mat%ng,MPI_DOUBLE_PRECISION,MPI_SUM,a_mat%comm,ierr)
  rhs_vec%n  = a_mat%ng

  ! --- Check if the matrix is distributed correctly
#ifdef USE_PETSC
  if (harmonic_matrix) call check_if_distributed(a_mat)
#else
  call check_if_distributed(a_mat)
#endif

  ! --- Handle all nessecarry cleanup operations
  call tr_locvnorms("cm_BCRhs",rhs_vec%val,a_mat%ng)
  call tr_debug_write("ndof",a_mat%ng)

  ! --- Timing
  call r3_info_end(r3_info_index_0)
  call tr_print_memsize("EndConstM")
end subroutine construct_matrix


!> Helps to interprete an element matrix index
subroutine  decrypt_index(ind, ivertex, iorder, ivar, itor)

  use mod_parameters,  only : n_tor, jorek_model, n_vertex_max, n_var, n_degrees

  integer, intent(in)  :: ind     !< Element matrix index
  integer, intent(out) :: ivertex !< Vertex index
  integer, intent(out) :: iorder  !< Degree of freedom
  integer, intent(out) :: ivar    !< Variable index
  integer, intent(out) :: itor    !< Toroidal mode index

  integer :: ind2

  ind2 = ind

  ivertex = ( ind2 - 1 ) / ( n_tor*n_var*n_degrees ) + 1
  ind2 = ind2 - ( ivertex - 1 ) * ( n_tor*n_var*n_degrees )

  iorder = ( ind2 - 1 ) / ( n_tor*n_var ) + 1
  ind2 = ind2 - ( iorder - 1 ) * ( n_tor*n_var )

  ivar = ( ind2 - 1 ) / ( n_tor ) + 1
  ind2 = ind2 - ( ivar - 1 ) * ( n_tor )

  itor = ind2

end subroutine decrypt_index

subroutine build_element_matrix(element, nodes, xpoint2, xcase2, R_axis,         &
      &                             Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint,   &
      &                             omp_tid, ife, n_local_elms, node_list, i_tor_min, i_tor_max, &
                                    aux_nodes, harmonic_matrix)

  use mod_parameters,           only : n_tor, jorek_model, n_vertex_max, n_degrees, unified_element_matrix, n_var
  use phys_module,              only : bc_natural_open, bc_natural_flux, n_tor_fft_thresh, grid_to_wall, n_wall_blocks, keep_n0_const
  USE data_structure,           only : type_element, type_node, type_node_list, thread_struct
  use mod_boundary_matrix_open, only : boundary_matrix_open
  use mod_elt_matrix,           only : element_matrix
  use mod_elt_matrix_fft,   only : element_matrix_fft
  use mpi_mod
  implicit none

  type (type_element),              intent(inout)  :: element
  type (type_node),                 intent(inout)  :: nodes(n_vertex_max)
  logical,                          intent(in)     :: xpoint2
  integer,                          intent(in)     :: xcase2
  real*8,                           intent(in)     :: R_axis
  real*8,                           intent(in)     :: Z_axis
  real*8,                           intent(in)     :: psi_axis
  real*8,                           intent(in)     :: psi_bnd
  real*8,                           intent(in)     :: R_xpoint(2)
  real*8,                           intent(in)     :: Z_xpoint(2)
  integer,                          intent(in)     :: omp_tid
  integer,                          intent(in)     :: ife
  integer,                          intent(in)     :: n_local_elms
  integer,                          intent(in)     :: i_tor_min   
  integer,                          intent(in)     :: i_tor_max   
  TYPE (type_node_list),            intent(in)     :: node_list
  type (type_node), optional,       intent(inout)  :: aux_nodes(n_vertex_max)
  logical,                          intent(in)    :: harmonic_matrix
  
  ! -- internal parameters
  integer :: iv, iv2, iv3, iv4, inode1, inode2, inode3, inode4, i, j
  !integer :: vertex(2), direction(n_degrees_1d), bnd1, bnd2, side1, side2
  integer :: i_max   
  integer :: n_tor_local

  ! --- Call element_matrix
  ! --- Call element_matrix
  if ( ( (i_tor_min .eq. 1) .and. (i_tor_max .eq. n_tor) .and. (n_tor .ge. n_tor_fft_thresh) )   &
    .or. (unified_element_matrix) ) then
    ! (use the FFT element matrix construction or the unified one in case it has been combined
    ! for the respective model)
    call element_matrix_fft(element,nodes, xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd,   &
      R_xpoint, Z_xpoint, thread_struct(omp_tid)%ELM, thread_struct(omp_tid)%RHS, omp_tid,       &
      thread_struct(omp_tid)%ELM_p, thread_struct(omp_tid)%ELM_n, thread_struct(omp_tid)%ELM_k,  &
      thread_struct(omp_tid)%ELM_kn, thread_struct(omp_tid)%RHS_p, thread_struct(omp_tid)%RHS_k, &
      thread_struct(omp_tid)%eq_g, thread_struct(omp_tid)%eq_s, thread_struct(omp_tid)%eq_t,     &
      thread_struct(omp_tid)%eq_p, thread_struct(omp_tid)%eq_ss, thread_struct(omp_tid)%eq_st,   &
      thread_struct(omp_tid)%eq_tt, thread_struct(omp_tid)%delta_g,                              &
      thread_struct(omp_tid)%delta_s, thread_struct(omp_tid)%delta_t, i_tor_min, i_tor_max,      &
      aux_nodes, thread_struct(omp_tid)%ELM_pnn)
  else
    ! (use the element matrix by toroidal integration in case of very few harmonics or in case
    ! of direct construction of the harmonic matrices used in preconditioning)
    call element_matrix(element,nodes, xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd,       &
      R_xpoint, Z_xpoint, thread_struct(omp_tid)%ELM, thread_struct(omp_tid)%RHS, omp_tid,       &
      i_tor_min, i_tor_max, aux_nodes)
  endif

  if (bc_natural_open) then
    ! ... (Implementation of sheath boundary conditions as in original) ...
  endif

  n_tor_local = i_tor_max - i_tor_min + 1
  if (keep_n0_const) then
    i_max = n_degrees * n_vertex_max * n_var * n_tor_local
#ifdef JECCD
    i_max = n_degrees * n_vertex_max * (n_var - 1) * n_tor_local
#endif
    do i = 1, i_max, n_tor_local
      thread_struct(omp_tid)%ELM(i, i) = 1.d15
    enddo
  endif

  !--- Diagnostic comparison of element matrix constructed with and without FFT
  ! TODO: Reimplement compare_elm_rhs diagnostic

end subroutine build_element_matrix


subroutine get_element_and_nodes(ielm, element, element_father, nodes, nodes_father, aux_nodes, harmonic_matrix)
  use phys_module, only: refinement
  use mod_parameters, only: n_vertex_max
  use data_structure
  use nodes_elements
  implicit none

  integer, intent(in) :: ielm
  type(type_element), intent(out) :: element
  type(type_element), intent(out) :: element_father
  type(type_node)   , intent(out) :: nodes(n_vertex_max)
  type(type_node)   , intent(out) :: nodes_father(n_vertex_max)
  type(type_node)   , intent(out) :: aux_nodes(n_vertex_max)
  logical, intent(in) :: harmonic_matrix

  integer :: ifather, inode, iv, inode_father

  element = element_list%element(ielm)
  
  ! --- Define nodes (mhd_sim% depends on whether our element has been refined)
  if (refinement .and. .not. harmonic_matrix) then

    ifather = element_list%element(ielm)%father

    if (ifather .ne. 0) then
      element_father = element_list%element(ifather)
      do iv = 1, n_vertex_max
        inode_father=element_father%vertex(iv)
        nodes_father(iv) = node_list%node(inode_father)
      enddo
    endif

  else
  
    do iv = 1, n_vertex_max
        inode   = element%vertex(iv)
        nodes(iv) = node_list%node(inode)
        aux_nodes(iv) = aux_node_list%node(inode)
    enddo

  endif
end subroutine get_element_and_nodes


subroutine add_to_a_mat(element, node_out, a_mat, rhs_local, my_ind_min, my_ind_max, omp_tid)
  use phys_module, only: refinement
  use mod_parameters, only: n_tor, n_var, n_degrees, n_vertex_max
  use data_structure, only: type_element, type_node, type_node_list, type_SP_MATRIX, thread_struct
  use nodes_elements
  implicit none

  ! Arguments
  type(type_element), intent(in) :: element
  integer, intent(in) :: node_out(:)
  type(type_SP_MATRIX), intent(inout) :: a_mat
  real*8              :: rhs_local(:)
  integer, intent(in) :: my_ind_min, my_ind_max
  integer, intent(in) :: omp_tid

  ! Locals
  integer :: i, j
  integer :: i_order
  integer :: inode1
  integer :: index_node1
  integer :: index_large_i
  integer :: index_ij

  logical :: eliminate_boundary_dofs = .false.
  logical :: i_bnd
  integer :: i_bnd_type
  real*8 :: elm_diagonal_average
  integer :: nnz_counter

  integer :: n_tor_local

  n_tor_local = a_mat%i_tor_max - a_mat%i_tor_min + 1

  if (eliminate_boundary_dofs) then
    nnz_counter = 0
    do i = 1, n_vertex_max
      do i_order = 1, n_degrees
          do j = 1, n_var * n_tor_local
            index_ij = n_tor_local * n_var * n_degrees * (i-1) + n_tor_local * n_var * (i_order-1) + j
            if (abs(thread_struct(omp_tid)%ELM(index_ij,index_ij)) .ne. 0) nnz_counter = nnz_counter + 1
            elm_diagonal_average = elm_diagonal_average + abs(thread_struct(omp_tid)%ELM(index_ij,index_ij))
          enddo
      enddo
    enddo
    elm_diagonal_average = elm_diagonal_average  / nnz_counter
    elm_diagonal_average = max(elm_diagonal_average, 1.d0)
    elm_diagonal_average = min(elm_diagonal_average, 1.d12)
  endif

  ! --- We only look at non-refined elements
  if ((.not. refinement) .or. (refinement .and. (element%n_sons .eq. 0))) then

    do i=1,n_vertex_max

      i_bnd = .false.

      inode1 = node_out(i)

      ! --- Get the boundary type of the node
      i_bnd_type = node_list%node(inode1)%boundary
      if (i_bnd_type .ne. 0) i_bnd = .true.

      do i_order = 1, n_degrees

        index_node1 = node_list%node(inode1)%index(i_order)

        index_large_i = n_tor_local * n_var * (index_node1 - 1)

        if ((index_node1 .ge. my_ind_min) .and. (index_node1 .le. my_ind_max)) then

          ! --- RHS assembly
          if (eliminate_boundary_dofs .and. i_bnd .and. &
                (     (i_bnd_type .eq. 1 .and. (i_order .eq. 1 .or. i_order .eq. 2)) &
                .or.  (i_bnd_type .eq. 2 .and. (i_order .eq. 1 .or. i_order .eq. 3)) &
                .or.  (i_bnd_type .eq. 3 .and. (i_order .eq. 1 .or. i_order .eq. 2 .or. i_order .eq. 3))   )) then

            do j = 1, n_var * n_tor_local
              index_ij = n_tor_local * n_var * n_degrees * (i-1) + n_tor_local * n_var * (i_order-1) + j
              !$omp critical
              rhs_local(index_large_i+j) = 0.d0
              !$omp end critical
            enddo
          else
            do j = 1, n_var * n_tor_local
              index_ij = n_tor_local * n_var * n_degrees * (i-1) + n_tor_local * n_var * (i_order-1) + j
              !$omp atomic
              rhs_local(index_large_i+j) = rhs_local(index_large_i+j) + thread_struct(omp_tid)%RHS(index_ij)
              !$omp end atomic
            enddo
          endif

          ! --- Matrix assembly: dispatch to backend-specific routine
#ifdef USE_PETSC
          if (a_mat%petsc_assembled) then
            call add_block_to_petsc(index_node1, i, i_order, i_bnd, i_bnd_type, &
                                    node_out, a_mat, omp_tid, n_tor_local, &
                                    eliminate_boundary_dofs, elm_diagonal_average)
          else
#endif
            call add_block_to_sp_matrix(index_node1, i, i_order, i_bnd, i_bnd_type, &
                                        node_out, a_mat, omp_tid, n_tor_local, &
                                        my_ind_min, my_ind_max, &
                                        eliminate_boundary_dofs, elm_diagonal_average)
#ifdef USE_PETSC
          endif
#endif

        endif ! my_ind_min < index < my_ind_max

      enddo ! n_degrees

    enddo ! n_vertex_max

  end if
end subroutine add_to_a_mat


!> Assemble element blocks into type_SP_MATRIX (irn/jcn/val arrays).
!! Called from add_to_a_mat for the non-PETSc path.
subroutine add_block_to_sp_matrix(index_node1, i, i_order, i_bnd, i_bnd_type, &
                                   node_out, a_mat, omp_tid, n_tor_local, &
                                   my_ind_min, my_ind_max, &
                                   eliminate_boundary_dofs, elm_diagonal_average)
  use mod_parameters, only: n_var, n_degrees, n_vertex_max
  use data_structure, only: type_node, type_node_list, type_SP_MATRIX, thread_struct
  use nodes_elements
  use mod_locate_irn_jcn
  implicit none

  integer, intent(in) :: index_node1, i, i_order
  logical, intent(in) :: i_bnd
  integer, intent(in) :: i_bnd_type
  integer, intent(in) :: node_out(:)
  type(type_SP_MATRIX), intent(inout) :: a_mat
  integer, intent(in) :: omp_tid, n_tor_local
  integer, intent(in) :: my_ind_min, my_ind_max
  logical, intent(in) :: eliminate_boundary_dofs
  real*8, intent(in) :: elm_diagonal_average

  integer :: j, k, l, k_order
  integer :: knode
  integer :: index_node2
  integer :: index_large_i, index_large_k
  integer :: index_ij, index_kl
  integer :: ijA_position, ilarge2
  logical :: interior, k_bnd
  integer :: k_bnd_type

  index_large_i = n_tor_local * n_var * (index_node1 - 1)

  do k=1,n_vertex_max

    knode = node_out(k)
    k_bnd = .false.

    k_bnd_type = node_list%node(knode)%boundary
    if (k_bnd_type .ne. 0) k_bnd = .true.

    interior = .not. (i_bnd .or. k_bnd)

    do k_order = 1, n_degrees

      index_node2 = node_list%node(knode)%index(k_order)

      index_large_k = n_tor_local * n_var * (index_node2 - 1)

      call locate_irn_jcn(index_node1,index_node2,my_ind_min,my_ind_max,ijA_position,a_mat)

      thread_struct(omp_tid)%synch_buff(1:n_var*n_tor_local*n_var*n_tor_local) = 0.d0

      do j = 1, n_var * n_tor_local
        index_ij = n_tor_local * n_var * n_degrees * (i-1) + n_tor_local * n_var * (i_order-1) + j

        do l = 1, n_var * n_tor_local

          index_kl = n_tor_local * n_var * n_degrees * (k-1) +  n_tor_local * n_var * (k_order-1) + l

          ilarge2 = ijA_position - 1 + (j-1) * n_var * n_tor_local + l

          a_mat%irn(ilarge2) = index_large_i + j
          a_mat%jcn(ilarge2) = index_large_k + l

          thread_struct(omp_tid)%synch_buff((j-1)*n_var*n_tor_local+l) = &
            thread_struct(omp_tid)%synch_buff((j-1)*n_var*n_tor_local+l) + thread_struct(omp_tid)%ELM(index_ij,index_kl)
        enddo

      enddo

      if (.not. eliminate_boundary_dofs) then
        !$omp critical
        a_mat%val(ijA_position : ijA_position + n_var*n_tor_local*n_var*n_tor_local - 1) = &
          a_mat%val(ijA_position : ijA_position + n_var*n_tor_local*n_var*n_tor_local - 1) +  &
          thread_struct(omp_tid)%synch_buff(1:n_var*n_tor_local*n_var*n_tor_local)
        !$omp end critical
      else

        if (interior) then
          !$omp critical
          a_mat%val(ijA_position : ijA_position + n_var*n_tor_local*n_var*n_tor_local - 1) = &
            a_mat%val(ijA_position : ijA_position + n_var*n_tor_local*n_var*n_tor_local - 1) +  &
            thread_struct(omp_tid)%synch_buff(1:n_var*n_tor_local*n_var*n_tor_local)
          !$omp end critical
        else
          if ((i .eq. k) .and. ((i_order .eq. 1 .and. k_order .eq. 1) &
                                    .or. ((i_bnd_type .eq. 2 .or. i_bnd_type .eq. 3) .and. (i_order .eq. 3 .and. k_order .eq. 3)) &
                                    .or. ((i_bnd_type .eq. 1 .or. i_bnd_type .eq. 3) .and. (i_order .eq. 2 .and. k_order .eq. 2)))) then

            !$omp critical
            do j = 1, n_var * n_tor_local
              do l = 1, n_var * n_tor_local
                ilarge2 = ijA_position - 1 + (j-1) * n_var * n_tor_local + l
                if (j .eq. l) then
                  a_mat%val(ilarge2) = a_mat%val(ilarge2) + elm_diagonal_average
                else
                  a_mat%val(ilarge2) = 0.d0
                endif
              enddo
            enddo
            !$omp end critical
          else if ((i_bnd .and. (i_order .eq. 1 .or. (i_bnd_type .eq. 2 .and. i_order .eq. 3) .or. &
                                                    (i_bnd_type .eq. 1 .and. i_order .eq. 2) .or. &
                                                    (i_bnd_type .eq. 3 .and. (i_order .eq. 2 .or. i_order .eq. 3)))) &
                    .or. (k_bnd .and. (k_order .eq. 1 .or. (k_bnd_type .eq. 2 .and. k_order .eq. 3) .or. &
                                                    (k_bnd_type .eq. 1 .and. k_order .eq. 2) .or. &
                                                    (k_bnd_type .eq. 3 .and. (k_order .eq. 2 .or. k_order .eq. 3))))) then
            !$omp critical
            a_mat%val(ijA_position : ijA_position + n_var*n_tor_local*n_var*n_tor_local - 1) = 0.d0
            !$omp end critical
          else
            !$omp critical
            a_mat%val(ijA_position : ijA_position + n_var*n_tor_local*n_var*n_tor_local - 1) = &
              a_mat%val(ijA_position : ijA_position + n_var*n_tor_local*n_var*n_tor_local - 1) +  &
              thread_struct(omp_tid)%synch_buff(1:n_var*n_tor_local*n_var*n_tor_local)
            !$omp end critical
          endif
        endif

      endif ! eliminate_boundary_dofs

    enddo ! n_degrees
  enddo ! n_vertex_max

end subroutine add_block_to_sp_matrix


#ifdef USE_PETSC
!> Assemble element blocks directly into PETSc MPIBAIJ matrix.
!! Called from add_to_a_mat for the direct PETSc assembly path.
subroutine add_block_to_petsc(index_node1, i, i_order, i_bnd, i_bnd_type, &
                               node_out, a_mat, omp_tid, n_tor_local, &
                               eliminate_boundary_dofs, elm_diagonal_average)
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_parameters, only: n_var, n_degrees, n_vertex_max
  use data_structure, only: type_node, type_node_list, type_SP_MATRIX, thread_struct
  use nodes_elements
  implicit none

  integer, intent(in) :: index_node1, i, i_order
  logical, intent(in) :: i_bnd
  integer, intent(in) :: i_bnd_type
  integer, intent(in) :: node_out(:)
  type(type_SP_MATRIX), intent(inout) :: a_mat
  integer, intent(in) :: omp_tid, n_tor_local
  logical, intent(in) :: eliminate_boundary_dofs
  real*8, intent(in) :: elm_diagonal_average

  integer :: j, k, l, k_order
  integer :: knode
  integer :: index_node2
  integer :: index_ij, index_kl
  logical :: interior, k_bnd
  integer :: k_bnd_type
  integer :: block_size

  PetscInt :: idxm_petsc(1), idxn_petsc(1)
  PetscErrorCode :: petsc_ierr

  block_size = n_var * n_tor_local

  do k=1,n_vertex_max

    knode = node_out(k)
    k_bnd = .false.

    k_bnd_type = node_list%node(knode)%boundary
    if (k_bnd_type .ne. 0) k_bnd = .true.

    interior = .not. (i_bnd .or. k_bnd)

    do k_order = 1, n_degrees

      index_node2 = node_list%node(knode)%index(k_order)

      ! --- Compute synch_buff from element matrix
      thread_struct(omp_tid)%synch_buff(1:block_size*block_size) = 0.d0
      do j = 1, block_size
        index_ij = n_tor_local * n_var * n_degrees * (i-1) + n_tor_local * n_var * (i_order-1) + j
        do l = 1, block_size
          index_kl = n_tor_local * n_var * n_degrees * (k-1) + n_tor_local * n_var * (k_order-1) + l
          thread_struct(omp_tid)%synch_buff((j-1)*block_size+l) = &
            thread_struct(omp_tid)%synch_buff((j-1)*block_size+l) + thread_struct(omp_tid)%ELM(index_ij,index_kl)
        enddo
      enddo

      ! --- Apply eliminate_boundary_dofs logic
      if (eliminate_boundary_dofs .and. .not. interior) then
        if ((i .eq. k) .and. ((i_order .eq. 1 .and. k_order .eq. 1) &
                                  .or. ((i_bnd_type .eq. 2 .or. i_bnd_type .eq. 3) .and. (i_order .eq. 3 .and. k_order .eq. 3)) &
                                  .or. ((i_bnd_type .eq. 1 .or. i_bnd_type .eq. 3) .and. (i_order .eq. 2 .and. k_order .eq. 2)))) then
          ! Diagonal boundary block: replace with diagonal of elm_diagonal_average
          thread_struct(omp_tid)%synch_buff(1:block_size*block_size) = 0.d0
          do j = 1, block_size
            thread_struct(omp_tid)%synch_buff((j-1)*block_size+j) = elm_diagonal_average
          enddo
        else if ((i_bnd .and. (i_order .eq. 1 .or. (i_bnd_type .eq. 2 .and. i_order .eq. 3) .or. &
                                                  (i_bnd_type .eq. 1 .and. i_order .eq. 2) .or. &
                                                  (i_bnd_type .eq. 3 .and. (i_order .eq. 2 .or. i_order .eq. 3)))) &
                  .or. (k_bnd .and. (k_order .eq. 1 .or. (k_bnd_type .eq. 2 .and. k_order .eq. 3) .or. &
                                                  (k_bnd_type .eq. 1 .and. k_order .eq. 2) .or. &
                                                  (k_bnd_type .eq. 3 .and. (k_order .eq. 2 .or. k_order .eq. 3))))) then
          ! Off-diagonal boundary block: skip (matrix already zeroed by MatZeroEntries)
          cycle
        endif
      endif

      ! --- Insert block into PETSc matrix
      idxm_petsc(1) = index_node1 - 1  ! 0-based block row
      idxn_petsc(1) = index_node2 - 1  ! 0-based block col
      !$omp critical
      call MatSetValuesBlocked(a_mat%petsc_A, 1, idxm_petsc, 1, idxn_petsc, &
                               thread_struct(omp_tid)%synch_buff, ADD_VALUES, petsc_ierr)
      !$omp end critical

    enddo ! n_degrees
  enddo ! n_vertex_max

end subroutine add_block_to_petsc
#endif


subroutine define_element_nodes(ielm, node_out, element, nodes, element_father, nodes_father, omp_tid, harmonic_matrix)
  use phys_module, only: refinement
  use mod_parameters, only: n_vertex_max
  use data_structure, only: type_element, type_node, thread_struct
  use mod_ch_nod_rhs_elm
  implicit none

  integer,            intent(in)          :: ielm
  integer,            intent(inout)       :: node_out(n_vertex_max)
  type(type_element), intent(in)          :: element
  type(type_node)   , intent(inout)       :: nodes(n_vertex_max)
  type(type_element), intent(in)          :: element_father
  type(type_node)   , intent(inout)       :: nodes_father(n_vertex_max)
  integer,            intent(in)          :: omp_tid
  logical,            intent(in)          :: harmonic_matrix

  integer :: iv, inode

  if (refinement .and. .not. harmonic_matrix) then   
    call ch_nod_rhs_elm(ielm,element,nodes,element_father,nodes_father, &
            thread_struct(omp_tid)%ELM, thread_struct(omp_tid)%RHS,node_out)
  else
    do iv=1, n_vertex_max
      node_out(iv) = element%vertex(iv)   
    enddo 
  endif
end subroutine define_element_nodes


subroutine initialize_v_and_harm(i_v, i_harm, i_tor_min, i_tor_max, n_tor_local)
  use mod_parameters, only: n_var, n_degrees, n_tor
  implicit none

  integer, intent(out) :: i_v(n_var)
  integer, allocatable, intent(out) :: i_harm(:)
  integer, intent(in) :: i_tor_min, i_tor_max
  integer, intent(in) :: n_tor_local

  integer :: i

  ! --- Initialize i_v
  do i = 1, n_var
    i_v(i) = i
  enddo

  ! --- Initialize i_harm
    if (.not. allocated(i_harm)) allocate(i_harm(n_tor_local))
    do i = i_tor_min, i_tor_max
      i_harm(i) = i
    enddo
end subroutine initialize_v_and_harm


!> check if matrix is row distributed
subroutine check_if_distributed(a_mat)
  use mpi_mod
  use data_structure, only: type_SP_MATRIX
  use mod_integer_types
  implicit none

  type(type_SP_MATRIX)  :: a_mat
  integer(kind=int_all) :: nloc, nglob
  integer               :: ierr

  nloc = a_mat%irn(a_mat%nnz) - a_mat%irn(1) + 1
  call MPI_AllReduce(nloc,nglob,1,MPI_INTEGER_ALL,MPI_SUM,a_mat%comm,ierr)
  a_mat%row_distributed = (nglob.eq.a_mat%ng)

  nloc = a_mat%jcn(a_mat%nnz) - a_mat%jcn(1) + 1
  call MPI_AllReduce(nloc,nglob,1,MPI_INTEGER_ALL,MPI_SUM,a_mat%comm,ierr)
  a_mat%col_distributed = (nglob.eq.a_mat%ng)

end subroutine check_if_distributed

end module construct_matrix_mod

