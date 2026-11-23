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
  use mod_petsc_raw_csr, only: blockmv_attach
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
  integer                           :: mpi_ierr
  integer                           :: omp_nthreads, omp_tid, n_tor_local
  integer                           :: my_ind_min, my_ind_max
  integer                           :: node_out(n_vertex_max)
  integer                           :: my_id
#ifdef USE_PETSC
  PetscErrorCode                    :: ierr
  logical                           :: ok_bmv
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

  ! --- Memory allocation of temporary buffers (not needed on PFAFEM global assembly path)
  if (.not. (use_pfafem .and. (.not. harmonic_matrix))) then
    call new_thread_buffers()
  endif 

  ! --- Allocation/Reallocation of the sparse matrix and right-hand side vector
  call prepare_matrix_storage(a_mat, harmonic_matrix)

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

  if (use_pfafem .and. (.not. harmonic_matrix)) then
    call construct_pfafem_elements(my_id, a_mat, rhs_local, my_ind_min, my_ind_max, &
                                   xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint)
  else
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
      call build_element_matrix(element, nodes, xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, omp_tid, ife, ielm, n_local_elms, node_list, a_mat%i_tor_min, a_mat%i_tor_max, aux_nodes, harmonic_matrix)

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
  endif

#ifdef USE_PETSC
    ! petsc_assembled, not .not.harmonic_matrix: with -jorek_mat_format aij there is
    ! no a_mat%petsc_A to assemble - the values live in val and are pushed once by
    ! petsc_update_matrix_coo - and the handle would be uninitialised here.
    if (a_mat%petsc_assembled) then
      if ((.not. eliminate_boundary_dofs) .or. fix_axis_nodes .or. treat_axis) then
        ! Flush element contributions (ADD_VALUES) before BCs (INSERT_VALUES)
        PetscCallA(MatAssemblyBegin(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, ierr))
        PetscCallA(MatAssemblyEnd(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, ierr))
      endif
    else
#endif
    ! --- Memory tracking
    call tr_vnorms("cm_A_bef_bc", a_mat%val, a_mat%nnz)
#ifdef USE_PETSC
    endif
#endif

    ! --- Apply boundary conditions.
    if (.not. eliminate_boundary_dofs) then !TODO: This is trivially not true in all cases and needs to be treated more precisely
      call boundary_conditions(my_id, node_list, element_list,  bnd_node_list,local_elms, n_local_elms,  &
                              my_ind_min, my_ind_max, rhs_local, xpoint2, xcase2, R_axis, Z_axis,        & 
                              psi_axis, psi_bnd, R_xpoint, Z_xpoint, psi_xpoint, a_mat)
    endif

    if (fix_axis_nodes) then
      call fix_nodes_on_axis(node_list, element_list, local_elms, n_local_elms, my_ind_min, my_ind_max, a_mat)
    elseif(treat_axis)then
      call penalize_dof_on_axis(node_list, 4, element_list, local_elms, n_local_elms, my_ind_min, my_ind_max, a_mat)
    endif

#ifdef USE_PETSC
    if (a_mat%petsc_assembled) then
      if (freeboundary .and. (sr%n_tor /= 0)) then
        ! Flush BC contributions (INSERT_VALUES) before vacuum (ADD_VALUES)
        PetscCallA(MatAssemblyBegin(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, ierr))
        PetscCallA(MatAssemblyEnd(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, ierr))
      endif
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
    if (a_mat%petsc_assembled) then
      ! Final assembly of PETSc matrix
      PetscCallA(MatAssemblyBegin(a_mat%petsc_A, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(a_mat%petsc_A, MAT_FINAL_ASSEMBLY, ierr))
      ok_bmv = blockmv_attach(a_mat%petsc_A, int(a_mat%block_size))
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
  call MPI_AllReduce(RHS_local,rhs_vec%val,a_mat%ng,MPI_DOUBLE_PRECISION,MPI_SUM,a_mat%comm,mpi_ierr)
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


!> Get a_mat's storage ready for one assembly pass, and decide which backend that
!! pass writes into. There are three cases:
!!
!!  - BAIJ (the default): assemble straight into a PETSc MATMPIBAIJ. irn/jcn/val
!!    are never allocated at all; petsc_assembled is what every insertion site
!!    branches on to take that route.
!!  - AIJ/COO (-jorek_mat_format aij) and the non-PETSc direct solvers: assemble
!!    into JOREK's own irn/jcn/val. petsc_assembled stays .false., so element
!!    blocks here plus the boundary conditions in mod_assembly, mod_axis_treatment,
!!    mod_fix_axis_nodes and vacuum_response all keep writing into val unchanged;
!!    petsc_init_system_coo then pushes val in one MatSetValuesCOO.
!!  - AIJ/COO in steady state: PETSc already owns the sparsity, so only val is
!!    reset. See the index discussion below.
subroutine prepare_matrix_storage(a_mat, harmonic_matrix)
  use tr_module
  use mod_integer_types, only: Int1
  use phys_module, only: use_matrix_equilibration
  use data_structure, only: type_SP_MATRIX
#ifdef USE_PETSC
  use mod_petsc, only: petsc_create_matrix, petsc_format_is_aij, petsc_coo_verify_requested
#endif
  implicit none

  type(type_SP_MATRIX), intent(inout) :: a_mat
  logical, intent(in) :: harmonic_matrix

  logical :: need_indices
#ifdef USE_PETSC
  PetscErrorCode :: ierr

  if ((.not. harmonic_matrix) .and. (.not. petsc_format_is_aij())) then
    if (.not. a_mat%petsc_assembled) then
      call petsc_create_matrix(a_mat%petsc_A, a_mat)
      a_mat%petsc_assembled = .true.
    else
      PetscCallA(MatZeroEntries(a_mat%petsc_A, ierr))
    endif
    return
  endif
#endif

  ! Does anything still read irn/jcn after this assembly? On the COO path PETSc
  ! derives the sparsity from the block structure (build_coo_indices), so the two
  ! nnz-length index arrays have exactly one remaining consumer: equilibration,
  ! which indexes through jcn on every solve. Skipping them there saves 8 B/nnz at
  ! the preallocation peak and two integer stores per matrix entry in the first
  ! element pass. Every other path - the harmonic PC matrix and the MUMPS /
  ! STRUMPACK / PaStiX direct solvers - consumes irn/jcn directly and keeps them.
  need_indices = .true.
#ifdef USE_PETSC
  if ((.not. harmonic_matrix) .and. petsc_format_is_aij() .and. (.not. use_matrix_equilibration) &
      .and. (.not. petsc_coo_verify_requested())) then
    need_indices = .false.
  endif
#endif

  if (a_mat%coo_structure_fixed) then
    ! COO path, steady state: PETSc holds the sparsity, so only the values change
    ! from one time step to the next. Re-allocating and re-zeroing the indices here
    ! would be most of the traffic through this routine for nothing, and they may
    ! well have been released by petsc_init_system_coo.
    call zero_val_parallel(a_mat)
    return
  endif

  if (associated(a_mat%val)) call tr_deallocatep(a_mat%val, "val", CAT_DMATRIX)
  call tr_allocatep(a_mat%val, Int1, a_mat%nnz, "val", CAT_DMATRIX)

  if (need_indices) then
    if (associated(a_mat%irn)) call tr_deallocatep(a_mat%irn, "irn", CAT_DMATRIX)
    if (associated(a_mat%jcn)) call tr_deallocatep(a_mat%jcn, "jcn", CAT_DMATRIX)
    call tr_allocatep(a_mat%irn, Int1, a_mat%nnz, "irn", CAT_DMATRIX)
    call tr_allocatep(a_mat%jcn, Int1, a_mat%nnz, "jcn", CAT_DMATRIX)
    a_mat%irn(1:a_mat%nnz) = 0
    a_mat%jcn(1:a_mat%nnz) = 0
  else
    ! Release rather than leave stale: add_block_to_sp_matrix keys the index fill on
    ! association, so a surviving array from a previous structure would be written
    ! with the new one's indices and quietly disagree with what PETSc was given.
    if (associated(a_mat%irn)) call tr_deallocatep(a_mat%irn, "irn", CAT_DMATRIX)
    if (associated(a_mat%jcn)) call tr_deallocatep(a_mat%jcn, "jcn", CAT_DMATRIX)
  endif

  call zero_val_parallel(a_mat)

end subroutine prepare_matrix_storage


!> Reset the value array before an assembly pass.
!!
!! Threaded rather than a plain array assignment: at production sizes this is a
!! multi-GB memset once per time step, which single-threaded costs about as much
!! as the entire element loop that follows. The static schedule also first-touches
!! the pages from the same threads that will accumulate into them.
subroutine zero_val_parallel(a_mat)
  use mod_integer_types, only: int_all
  use data_structure, only: type_SP_MATRIX
  implicit none

  type(type_SP_MATRIX), intent(inout) :: a_mat
  integer(kind=int_all) :: k

  !$omp parallel do default(none) shared(a_mat) private(k) schedule(static)
  do k = 1, a_mat%nnz
    a_mat%val(k) = 0.0d0
  enddo
  !$omp end parallel do
end subroutine zero_val_parallel


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
      &                             omp_tid, ife, ielm, n_local_elms, node_list, i_tor_min, i_tor_max, &
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
  integer,                          intent(in)     :: ielm
  integer,                          intent(in)     :: n_local_elms
  integer,                          intent(in)     :: i_tor_min   
  integer,                          intent(in)     :: i_tor_max   
  TYPE (type_node_list),            intent(in)     :: node_list
  type (type_node), optional,       intent(inout)  :: aux_nodes(n_vertex_max)
  logical,                          intent(in)    :: harmonic_matrix
  
  ! -- internal parameters
  integer :: iv, iv2, iv3, iv4, inode1, inode2, inode3, inode4, i, j
  integer :: vertex(2), direction(n_degrees_1d), bnd1, bnd2, side1, side2
  type (type_node) :: bnd_nodes(n_vertex_max)   ! the edge's nodes, rotated for boundary_matrix_open
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

  ! --- Apply sheath boundary conditions at the targets
  if (bc_natural_open) then
    ! --- Loop over the 4 nodes
    do iv = 1, n_vertex_max
      
      iv2  = mod(iv, n_vertex_max) + 1
      iv3 = mod(iv2, n_vertex_max) + 1
      iv4 = mod(iv3, n_vertex_max) + 1
      
      inode1 = element%vertex(iv)
      inode2 = element%vertex(iv2)
      inode3 = element%vertex(iv3)
      inode4 = element%vertex(iv4)
      
      bnd1 = node_list%node(inode1)%boundary
      bnd2 = node_list%node(inode2)%boundary
      
      ! --- carry on only if on boundary
      if ( (bnd1 .eq. 0) .or. (bnd2 .eq. 0)) cycle
      
      bnd_nodes(1) = node_list%node(inode1)
      bnd_nodes(2) = node_list%node(inode2)
      bnd_nodes(3) = node_list%node(inode3)
      bnd_nodes(4) = node_list%node(inode4)

      vertex    = (/ iv, iv2 /)
      
      if ( (grid_to_wall) .and. (n_wall_blocks .gt. 0) ) then
        side1 = 0                  ; side2 = 0
        if (bnd1 .eq. 1) side1 = 2 ; if (bnd2 .eq. 1) side2 = 2
        if (bnd1 .eq.11) side1 = 2 ; if (bnd2 .eq.11) side2 = 2
        if (bnd1 .eq. 5) side1 = 3 ; if (bnd2 .eq. 5) side2 = 3
        if (bnd1 .eq.15) side1 = 3 ; if (bnd2 .eq.15) side2 = 3
        if (bnd1 .eq. 2) side1 = 3 ; if (bnd2 .eq. 2) side2 = 3
        if (bnd1 .eq.12) side1 = 2 ; if (bnd2 .eq.12) side2 = 2
        if (bnd1 .eq. 4) side1 = 2 ; if (bnd2 .eq. 4) side2 = 2
        direction(1) = 1
        if     ( (side1 .eq. 2) .or. (side2 .eq. 2) ) then
          direction(2) = 2
          if (n_order .ge. 5) direction(3) = 5
        elseif ( (side1 .eq. 3) .or. (side2 .eq. 3) ) then
          direction(2) = 3
          if (n_order .ge. 5) direction(3) = 6
        endif
        ! --- This should never happen, but just in case...
        if (     ((side1 .eq. 2) .and. (side2 .eq. 3)) &
            .or. ((side1 .eq. 3) .and. (side2 .eq. 2)) ) then
          write(*,'(A,4i8)') 'WARNING: boundary_matrix_open, boundary element incoherent ',&
                             inode1,node_list%node(inode1)%boundary,inode2,node_list%node(inode2)%boundary  
          cycle
        endif
      else
        ! --- The target has boundary 1 or 3
        direction(1) = 1
        if ((bnd1 .eq. 9) .and. (bnd2 .eq. 9)) then ! for removed triangle, we get 9 - 9 boundary, where it could be either s=const or t=const
          ! so we use the fact that the nodes are consistently defining which sides are the s and t sides (identical to that in grids/mod_boundary.f90)
          if ( mod(iv,2) == 1 ) then
            direction(2) = 2
          else
            direction(2) = 3
          end if
            
        elseif ( (  ((bnd1 .eq. 1) .or. (bnd1 .eq. 3)) .and. ((bnd2 .eq. 1) .or. (bnd2 .eq. 3))  ) &
            .or. (  ((bnd1 .eq. 1) .or. (bnd1 .eq. 9)) .and. ((bnd2 .eq. 1) .or. (bnd2 .eq. 9))  ) &
            .or. (  ((bnd1 .eq. 4) .or. (bnd1 .eq. 9)) .and. ((bnd2 .eq. 4) .or. (bnd2 .eq. 9))  ) &
            .or. (  ((bnd1 .eq. 1) .or. (bnd1 .eq. 4)) .and. ((bnd2 .eq. 4) .or. (bnd2 .eq. 1))  ) ) then
          
          direction(2) = 2
          if (n_order .ge. 5) direction(3) = 5
          
        elseif (  ((bnd1 .eq. 5) .or. (bnd1 .eq. 9)) .and. ((bnd2 .eq. 5) .or. (bnd2 .eq. 9)) ) then
          
          direction(2) = 3
          if (n_order .ge. 5) direction(3) = 6
          
        elseif (  ((bnd1 .eq. 2) .or. (bnd1 .eq. 3)) .and. ((bnd2 .eq. 2) .or. (bnd2 .eq. 3)) ) then
          
          direction(2) = 3
          if (n_order .ge. 5) direction(3) = 6
          
        else
          write(*,'(A,4i8)') 'WARNING: boundary_matrix_open, boundary element not included ',&
                             inode1,node_list%node(inode1)%boundary,inode2,node_list%node(inode2)%boundary  
          cycle
        endif
      endif

      ! sanity check of the above code: the side between node 1 and 2 (iv=1), and nodes 3 and 4 (iv=3) should have direction(2)=2 (i.e. t=const.), while 
      ! the other two sides between nodes 2 and 3 (iv=2), and 4 and 1 (iv=4) should have direction(2)=3 (i.e. s=cont.)
      if (((direction(2)==2) .and. ( mod(iv,2) /= 1 )) .or. ((direction(2)==3) .and. (mod(iv,2) /= 0))) then
        !$omp critical
        write(*,"(A,3I6)") "ERROR: There seems to be an inconsistency in direction(2) in matrix/construct_matrix_mod.f90, (direction(2) / iv / node)=", direction(2),iv,inode1
        !$omp end critical
      end if
        

      ! --- Build matrix elements for boundary
#if JOREK_MODEL == 183
      call boundary_matrix_open(vertex, direction, element, bnd_nodes, & 
                                xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, &
                                thread_struct(omp_tid)%ELM, thread_struct(omp_tid)%RHS, i_tor_min, i_tor_max, ielm)
#else
      call boundary_matrix_open(vertex, direction, element, bnd_nodes, &
                                xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, &
                                thread_struct(omp_tid)%ELM, thread_struct(omp_tid)%RHS, i_tor_min, i_tor_max)
#endif

    enddo
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
  use phys_module, only: refinement, eliminate_boundary_dofs
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

  logical :: i_bnd
  integer :: i_bnd_type
  !> Per-variable diagonal of the eliminate_boundary_dofs boundary rows: the
  !! variables' diagonals differ by orders of magnitude, and one shared value
  !! would put some fields' boundary rows far off their bulk scale.
  real*8 :: elm_diagonal_average(n_var)
  integer :: nnz_counter(n_var)
  integer :: jvar

  integer :: n_tor_local

  n_tor_local = a_mat%i_tor_max - a_mat%i_tor_min + 1

  if (eliminate_boundary_dofs) then
    elm_diagonal_average = 0.d0
    nnz_counter = 0
    do i = 1, n_vertex_max
      do i_order = 1, n_degrees
          do j = 1, n_var * n_tor_local
            index_ij = n_tor_local * n_var * n_degrees * (i-1) + n_tor_local * n_var * (i_order-1) + j
            ! j runs over (variable, harmonic), harmonic innermost -- the same
            ! layout the BC rows are built with elsewhere.
            jvar = (j - 1) / n_tor_local + 1
            if (abs(thread_struct(omp_tid)%ELM(index_ij,index_ij)) .ne. 0) &
              nnz_counter(jvar) = nnz_counter(jvar) + 1
            elm_diagonal_average(jvar) = elm_diagonal_average(jvar) &
                                       + abs(thread_struct(omp_tid)%ELM(index_ij,index_ij))
          enddo
      enddo
    enddo
    do jvar = 1, n_var
      if (nnz_counter(jvar) > 0) then
        elm_diagonal_average(jvar) = elm_diagonal_average(jvar) / nnz_counter(jvar)
      else
        elm_diagonal_average(jvar) = 1.d0
      endif
      ! NO absolute floor. max(...,1.d0) is what broke the small-scale variables,
      ! and it is meaningless once the value is per-variable -- the point is that
      ! each variable's own scale sets it. Guard only against a degenerate value.
      if (.not. (elm_diagonal_average(jvar) > 0.d0)) elm_diagonal_average(jvar) = 1.d0
      elm_diagonal_average(jvar) = min(elm_diagonal_average(jvar), 1.d12)
    enddo
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
  real*8, intent(in) :: elm_diagonal_average(:)   !< one per variable

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

      ! Index fill runs only for backends that consume irn/jcn. prepare_matrix_storage
      ! leaves them unassociated on the COO path, where PETSc derives the sparsity
      ! from the block structure instead (build_coo_indices), and releases them once
      ! the preallocation is done. Keeping this out of the accumulation loop below
      ! also saves two integer stores per matrix entry on every later time step.
      if (associated(a_mat%irn)) then
        do j = 1, n_var * n_tor_local
          do l = 1, n_var * n_tor_local
            ilarge2 = ijA_position - 1 + (j-1) * n_var * n_tor_local + l
            a_mat%irn(ilarge2) = index_large_i + j
            a_mat%jcn(ilarge2) = index_large_k + l
          enddo
        enddo
      endif

      do j = 1, n_var * n_tor_local
        index_ij = n_tor_local * n_var * n_degrees * (i-1) + n_tor_local * n_var * (i_order-1) + j

        do l = 1, n_var * n_tor_local

          index_kl = n_tor_local * n_var * n_degrees * (k-1) +  n_tor_local * n_var * (k_order-1) + l

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
                  a_mat%val(ilarge2) = a_mat%val(ilarge2) &
                                     + elm_diagonal_average((j-1)/n_tor_local + 1)
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
  real*8, intent(in) :: elm_diagonal_average(:)   !< one per variable

  integer :: j, k, l, k_order
  integer :: knode
  integer :: index_node2
  integer :: index_ij, index_kl
  logical :: interior, k_bnd
  integer :: k_bnd_type
  integer :: block_size

  PetscInt :: idxm_petsc(1), idxn_petsc(1)
  PetscErrorCode :: ierr

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
            thread_struct(omp_tid)%synch_buff((j-1)*block_size+j) = &
              elm_diagonal_average((j-1)/n_tor_local + 1)
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
      PetscCallA(MatSetValuesBlocked(a_mat%petsc_A, 1, idxm_petsc, 1, idxn_petsc, thread_struct(omp_tid)%synch_buff, ADD_VALUES, ierr))
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


pure logical function is_elim_bnd_dof(btype, io)
  implicit none
  integer, intent(in) :: btype, io
  is_elim_bnd_dof = .false.
  if (btype == 1) then
    is_elim_bnd_dof = (io == 1 .or. io == 2)
  else if (btype == 2) then
    is_elim_bnd_dof = (io == 1 .or. io == 3)
  else if (btype == 3) then
    is_elim_bnd_dof = (io == 1 .or. io == 2 .or. io == 3)
  end if
end function is_elim_bnd_dof


subroutine construct_pfafem_elements(my_id, a_mat, rhs_local, my_ind_min, my_ind_max, &
                                     xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint)
  use phys_module, only: eliminate_boundary_dofs, refinement
  use mod_parameters, only: n_tor, n_var, n_degrees
  use data_structure, only: type_SP_MATRIX
  use nodes_elements, only: node_list, element_list, aux_node_list
  use mod_locate_irn_jcn, only: locate_irn_jcn
  use mod_pfafem, only: pfafem_cache, pfafem_el_lo, pfafem_el_hi
  use omp_lib
#if JOREK_MODEL == 180
  use mod_elt_matrix, only: element_matrix_pfafem
#elif (JOREK_MODEL == 183) || (JOREK_MODEL == 600)
  use mod_elt_matrix_fft, only: element_matrix_pfafem
#endif
  implicit none

  integer,              intent(in)    :: my_id
  type(type_SP_MATRIX), intent(inout) :: a_mat
  real*8,               intent(inout) :: rhs_local(:)
  integer,              intent(in)    :: my_ind_min, my_ind_max, xcase2
  logical,              intent(in)    :: xpoint2
  real*8,               intent(in)    :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,               intent(in)    :: R_xpoint(2), Z_xpoint(2)

#if (JOREK_MODEL == 180) || (JOREK_MODEL == 183) || (JOREK_MODEL == 600)
  integer :: el_lo, el_hi, b_lo, b_hi, n_el_loc, ibatch, nbatch, iphase, nphase, n_omp_pf, max_thr_pf, ielm, omp_tid, block_size
  integer :: n_u, n_vp, n_sp, cap_u, cap_sp, u1, u2, ivp, isp, iuj1, iuj2, jb1, jb2, io1, io2
  integer :: inode1, inode2, index_node1, index_node2, btype1, btype2
  integer :: r_loc, col_loc, ivar, jvar, index_large_i, index_large_k, ijA_position, ilarge2
  integer :: nnz_counter(n_var)
  integer(8) :: bytes_per_thr, mem_cap_pf, n_ebuf, row_off8, base_off8
  real*8  :: elm_diagonal_average(n_var), dval
  logical :: petsc_direct, has_owned, c1, c2
  real*8, allocatable :: ELM_sp(:,:,:,:,:), RHS_u(:,:,:), buf_blk(:)
#ifdef USE_PETSC
  PetscInt :: idxm_p(1), nr_p, nc_p, key_col
  PetscInt, allocatable :: idxm_u(:), idxn_u(:), idx_bc(:), raw_col(:)
  integer,  allocatable :: raw_u2(:), raw_io2(:), raw_slot(:), u2_slot(:,:)
  real*8,   allocatable :: ebuf(:)
  integer :: rsel_p(n_degrees), n_raw_c, n_diag_bc, k_c, j_c, key_u2, key_io2, r_idx, c_slot
  PetscErrorCode :: ierr
#endif

  block_size = n_var * n_tor
  petsc_direct = .false.
#ifdef USE_PETSC
  petsc_direct = a_mat%petsc_assembled
  if (petsc_direct) then
    call construct_pfafem_elements_petsc_direct(my_id, a_mat, rhs_local, my_ind_min, my_ind_max, &
                                                xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint)
    return
  end if
#endif

  if (petsc_direct) then
    el_lo = pfafem_el_lo
    el_hi = pfafem_el_hi
  else
    el_lo = 1
    el_hi = element_list%n_elements
    if (el_hi < el_lo) return
  end if

  max_thr_pf = 1
  !$ max_thr_pf = omp_get_max_threads()
  nbatch = merge(3, 1, petsc_direct)
  nphase = 1
  if (petsc_direct .and. max_thr_pf >= 48) then
    mem_cap_pf = 22000000000_8
  else if (petsc_direct .and. max_thr_pf >= 16) then
    mem_cap_pf = 14000000000_8
  else if (petsc_direct) then
    mem_cap_pf = 5500000000_8
  else
    mem_cap_pf = 36000000000_8
  end if
  n_el_loc = max(0, el_hi - el_lo + 1)

  do ibatch = 1, nbatch
    b_lo = el_lo + ((ibatch - 1) * n_el_loc) / nbatch
    b_hi = el_lo + (ibatch * n_el_loc) / nbatch - 1

    do iphase = 0, nphase - 1
    if ((.not. petsc_direct .or. mod(my_id, nphase) == iphase) .and. (b_hi >= b_lo)) then
      cap_u  = 16
      cap_sp = 64
      do ielm = b_lo, b_hi
        if (refinement .and. (element_list%element(ielm)%n_sons /= 0)) cycle
        cap_u  = max(cap_u,  pfafem_cache(ielm)%n_unodes)
        cap_sp = max(cap_sp, pfafem_cache(ielm)%n_sp)
      end do
      bytes_per_thr = int(n_degrees * n_degrees * n_var * n_var, 8) * int(cap_sp, 8) * 8_8 &
                    + int(cap_u * n_degrees * n_degrees, 8) * int(block_size, 8) * int(block_size, 8) * 8_8
      n_omp_pf = max(1, min(max_thr_pf, int(mem_cap_pf / max(1_8, bytes_per_thr))))

      !$omp parallel num_threads(n_omp_pf) default(shared) &
      !$omp   private(ielm, omp_tid, n_u, n_vp, n_sp, u1, u2, ivp, isp, iuj1, iuj2, jb1, jb2, io1, io2, &
      !$omp           inode1, inode2, index_node1, index_node2, btype1, btype2, &
      !$omp           r_loc, col_loc, ivar, jvar, index_large_i, index_large_k, ijA_position, ilarge2, &
      !$omp           nnz_counter, elm_diagonal_average, dval, has_owned, c1, c2, &
      !$omp           ELM_sp, RHS_u, buf_blk, n_ebuf, row_off8, base_off8 &
#ifdef USE_PETSC
      !$omp           , idxm_p, nr_p, nc_p, key_col, idxm_u, idxn_u, idx_bc, raw_col &
      !$omp           , raw_u2, raw_io2, raw_slot, u2_slot, ebuf &
      !$omp           , rsel_p, n_raw_c, n_diag_bc, k_c, j_c, key_u2, key_io2, r_idx, c_slot, ierr &
#endif
      !$omp          )
#ifdef _OPENMP
      omp_tid = 1 + omp_get_thread_num()
#else
      omp_tid = 1
#endif
      allocate(ELM_sp(n_degrees, n_degrees, n_var, n_var, cap_sp))
      allocate(RHS_u(block_size, n_degrees, cap_u))
      allocate(buf_blk(block_size * block_size))
#ifdef USE_PETSC
      if (petsc_direct) then
        allocate(idxm_u(n_degrees), idxn_u(n_degrees * cap_u))
        allocate(idx_bc(n_degrees * cap_u), raw_col(n_degrees * cap_u))
        allocate(raw_u2(n_degrees * cap_u), raw_io2(n_degrees * cap_u), raw_slot(n_degrees * cap_u))
        allocate(u2_slot(n_degrees, cap_u))
        n_ebuf = int(cap_u, 8) * int(n_degrees * n_degrees, 8) * int(block_size, 8) * int(block_size, 8)
        allocate(ebuf(n_ebuf))
      end if
#endif

      !$omp do schedule(dynamic, 1)
      do ielm = b_lo, b_hi
        if (refinement .and. (element_list%element(ielm)%n_sons /= 0)) cycle

        n_u  = pfafem_cache(ielm)%n_unodes
        n_vp = pfafem_cache(ielm)%n_vpairs
        n_sp = pfafem_cache(ielm)%n_sp
        if (n_u <= 0 .or. n_vp <= 0 .or. n_sp <= 0) cycle

        if (.not. petsc_direct) then
          has_owned = .false.
          do u1 = 1, n_u
            inode1 = pfafem_cache(ielm)%unodes(u1)
            do io1 = 1, n_degrees
              index_node1 = node_list%node(inode1)%index(io1)
              if (index_node1 >= my_ind_min .and. index_node1 <= my_ind_max) then
                has_owned = .true.
                exit
              end if
            end do
            if (has_owned) exit
          end do
          if (.not. has_owned) cycle
        end if

        call element_matrix_pfafem(node_list, element_list, ielm, xpoint2, xcase2, &
                                   R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, &
                                   ELM_sp(:, :, :, :, 1:n_sp), RHS_u(:, :, 1:n_u), omp_tid, aux_node_list)

        if (eliminate_boundary_dofs) then
          elm_diagonal_average = 0.d0
          nnz_counter = 0
          do u1 = 1, n_u
            do io1 = 1, n_degrees
              do jvar = 1, n_var
                do jb1 = 1, n_tor
                  iuj1 = pfafem_cache(ielm)%uj_map(u1, jb1)
                  if (iuj1 <= 0) cycle
                  isp  = pfafem_cache(ielm)%sp_map(iuj1, iuj1)
                  dval = abs(ELM_sp(io1, io1, jvar, jvar, isp))
                  if (dval /= 0.d0) then
                    nnz_counter(jvar) = nnz_counter(jvar) + 1
                    elm_diagonal_average(jvar) = elm_diagonal_average(jvar) + dval
                  end if
                end do
              end do
            end do
          end do
          do jvar = 1, n_var
            if (nnz_counter(jvar) > 0) then
              elm_diagonal_average(jvar) = elm_diagonal_average(jvar) / dble(nnz_counter(jvar))
            else
              elm_diagonal_average(jvar) = 1.d0
            end if
            if (.not. (elm_diagonal_average(jvar) > 0.d0)) elm_diagonal_average(jvar) = 1.d0
            elm_diagonal_average(jvar) = min(elm_diagonal_average(jvar), 1.d12)
          end do
        end if

        ! --- Scatter RHS_u
        do u1 = 1, n_u
          inode1 = pfafem_cache(ielm)%unodes(u1)
          btype1 = node_list%node(inode1)%boundary
          do io1 = 1, n_degrees
            index_node1 = node_list%node(inode1)%index(io1)
            if (index_node1 <= 0) cycle
            if ((.not. petsc_direct) .and. (index_node1 < my_ind_min .or. index_node1 > my_ind_max)) cycle
            if (eliminate_boundary_dofs .and. is_elim_bnd_dof(btype1, io1)) cycle
            index_large_i = block_size * (index_node1 - 1)
            do r_loc = 1, block_size
              !$omp atomic
              rhs_local(index_large_i + r_loc) = rhs_local(index_large_i + r_loc) + RHS_u(r_loc, io1, u1)
              !$omp end atomic
            end do
          end do
        end do

        ! --- Scatter ELM_sp
#ifdef USE_PETSC
        if (petsc_direct) then
          n_diag_bc = 0
          do u1 = 1, n_u
            inode1 = pfafem_cache(ielm)%unodes(u1)
            btype1 = node_list%node(inode1)%boundary
            nr_p = 0
            do io1 = 1, n_degrees
              index_node1 = node_list%node(inode1)%index(io1)
              if (index_node1 <= 0) cycle
              c1 = eliminate_boundary_dofs .and. is_elim_bnd_dof(btype1, io1)
              if (c1) then
                if (pfafem_cache(ielm)%pair_map(u1, u1) > 0) then
                  n_diag_bc = n_diag_bc + 1
                  idx_bc(n_diag_bc) = index_node1 - 1
                end if
              else
                nr_p = nr_p + 1
                rsel_p(nr_p) = io1
                idxm_u(nr_p) = index_node1 - 1
              end if
            end do
            if (nr_p == 0) cycle

            n_raw_c = 0
            do u2 = 1, n_u
              ivp = pfafem_cache(ielm)%pair_map(u1, u2)
              if (ivp <= 0) cycle
              inode2 = pfafem_cache(ielm)%unodes(u2)
              btype2 = node_list%node(inode2)%boundary
              do io2 = 1, n_degrees
                index_node2 = node_list%node(inode2)%index(io2)
                if (index_node2 <= 0) cycle
                if (eliminate_boundary_dofs .and. is_elim_bnd_dof(btype2, io2)) cycle
                n_raw_c = n_raw_c + 1
                raw_col(n_raw_c) = index_node2 - 1
                raw_u2(n_raw_c)  = u2
                raw_io2(n_raw_c) = io2
              end do
            end do
            if (n_raw_c == 0) cycle

            ! Insertion sort raw_col(1:n_raw_c) ascending
            do k_c = 2, n_raw_c
              key_col = raw_col(k_c)
              key_u2  = raw_u2(k_c)
              key_io2 = raw_io2(k_c)
              j_c = k_c - 1
              do while (j_c >= 1)
                if (raw_col(j_c) <= key_col) exit
                raw_col(j_c + 1) = raw_col(j_c)
                raw_u2(j_c + 1)  = raw_u2(j_c)
                raw_io2(j_c + 1) = raw_io2(j_c)
                j_c = j_c - 1
              end do
              raw_col(j_c + 1) = key_col
              raw_u2(j_c + 1)  = key_u2
              raw_io2(j_c + 1) = key_io2
            end do

            ! Deduplicate identical column block indices and record (io2, u2) -> c_slot
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

            u2_slot(1:n_degrees, 1:n_u) = 0
            do k_c = 1, n_raw_c
              u2_slot(raw_io2(k_c), raw_u2(k_c)) = raw_slot(k_c)
            end do

            n_ebuf = int(nr_p, 8) * int(nc_p, 8) * int(block_size, 8) * int(block_size, 8)
            ebuf(1_8 : n_ebuf) = 0.d0
            do isp = pfafem_cache(ielm)%u_sp_lo(u1), pfafem_cache(ielm)%u_sp_hi(u1)
              iuj1 = pfafem_cache(ielm)%sp_pairs(1, isp)
              iuj2 = pfafem_cache(ielm)%sp_pairs(2, isp)
              jb1  = pfafem_cache(ielm)%uj_jb(iuj1)
              u2   = pfafem_cache(ielm)%uj_u(iuj2)
              jb2  = pfafem_cache(ielm)%uj_jb(iuj2)
              do r_idx = 1, int(nr_p)
                io1 = rsel_p(r_idx)
                do ivar = 1, n_var
                  r_loc    = (ivar - 1) * n_tor + jb1
                  row_off8 = int((r_idx - 1) * block_size + (r_loc - 1), 8) * int(nc_p * block_size, 8)
                  do io2 = 1, n_degrees
                    c_slot = u2_slot(io2, u2)
                    if (c_slot <= 0) cycle
                    base_off8 = row_off8 + int((c_slot - 1) * block_size + jb2, 8)
                    do jvar = 1, n_var
                      ebuf(base_off8 + int((jvar - 1) * n_tor, 8)) = &
                        ebuf(base_off8 + int((jvar - 1) * n_tor, 8)) + ELM_sp(io1, io2, ivar, jvar, isp)
                    end do
                  end do
                end do
              end do
            end do

            !$omp critical(pfafem_mat_scatter)
            PetscCallA(MatSetValuesBlocked(a_mat%petsc_A, nr_p, idxm_u(1:nr_p), nc_p, idxn_u(1:nc_p), ebuf(1_8:n_ebuf), ADD_VALUES, ierr))
            !$omp end critical(pfafem_mat_scatter)
          end do

          if (n_diag_bc > 0) then
            buf_blk = 0.d0
            do r_loc = 1, block_size
              buf_blk((r_loc - 1) * block_size + r_loc) = elm_diagonal_average((r_loc - 1) / n_tor + 1)
            end do
            !$omp critical(pfafem_mat_scatter)
            do k_c = 1, n_diag_bc
              idxm_p(1) = idx_bc(k_c)
              PetscCallA(MatSetValuesBlocked(a_mat%petsc_A, 1, idxm_p, 1, idxm_p, buf_blk, ADD_VALUES, ierr))
            end do
            !$omp end critical(pfafem_mat_scatter)
          end if
        else
#endif
        !$omp critical(pfafem_mat_scatter)
        do ivp = 1, n_vp
          u1 = pfafem_cache(ielm)%vpairs(1, ivp)
          u2 = pfafem_cache(ielm)%vpairs(2, ivp)
          inode1 = pfafem_cache(ielm)%unodes(u1)
          inode2 = pfafem_cache(ielm)%unodes(u2)
          btype1 = node_list%node(inode1)%boundary
          btype2 = node_list%node(inode2)%boundary

          do io1 = 1, n_degrees
            index_node1 = node_list%node(inode1)%index(io1)
            if (index_node1 <= 0) cycle
            if ((.not. petsc_direct) .and. (index_node1 < my_ind_min .or. index_node1 > my_ind_max)) cycle
            c1 = eliminate_boundary_dofs .and. is_elim_bnd_dof(btype1, io1)

            do io2 = 1, n_degrees
              index_node2 = node_list%node(inode2)%index(io2)
              if (index_node2 <= 0) cycle
              c2 = eliminate_boundary_dofs .and. is_elim_bnd_dof(btype2, io2)

              if (c1 .or. c2) then
                if ((inode1 == inode2) .and. (io1 == io2)) then
                  buf_blk = 0.d0
                  do r_loc = 1, block_size
                    buf_blk((r_loc - 1) * block_size + r_loc) = elm_diagonal_average((r_loc - 1) / n_tor + 1)
                  end do
                else
                  if (petsc_direct) cycle
                  call locate_irn_jcn(index_node1, index_node2, my_ind_min, my_ind_max, ijA_position, a_mat)
                  if (associated(a_mat%irn)) then
                    index_large_i = block_size * (index_node1 - 1)
                    index_large_k = block_size * (index_node2 - 1)
                    do r_loc = 1, block_size
                      do col_loc = 1, block_size
                        ilarge2 = ijA_position - 1 + (r_loc - 1) * block_size + col_loc
                        a_mat%irn(ilarge2) = index_large_i + r_loc
                        a_mat%jcn(ilarge2) = index_large_k + col_loc
                      end do
                    end do
                  end if
                  cycle
                end if
              else
                buf_blk = 0.d0
                do isp = pfafem_cache(ielm)%u_sp_lo(u1), pfafem_cache(ielm)%u_sp_hi(u1)
                  iuj2 = pfafem_cache(ielm)%sp_pairs(2, isp)
                  if (pfafem_cache(ielm)%uj_u(iuj2) /= u2) cycle
                  iuj1 = pfafem_cache(ielm)%sp_pairs(1, isp)
                  jb1  = pfafem_cache(ielm)%uj_jb(iuj1)
                  jb2  = pfafem_cache(ielm)%uj_jb(iuj2)
                  do jvar = 1, n_var
                    col_loc = (jvar - 1) * n_tor + jb2
                    do ivar = 1, n_var
                      r_loc = (ivar - 1) * n_tor + jb1
                      buf_blk((r_loc - 1) * block_size + col_loc) = ELM_sp(io1, io2, ivar, jvar, isp)
                    end do
                  end do
                end do
              end if

              call locate_irn_jcn(index_node1, index_node2, my_ind_min, my_ind_max, ijA_position, a_mat)
              if (associated(a_mat%irn)) then
                index_large_i = block_size * (index_node1 - 1)
                index_large_k = block_size * (index_node2 - 1)
                do r_loc = 1, block_size
                  do col_loc = 1, block_size
                    ilarge2 = ijA_position - 1 + (r_loc - 1) * block_size + col_loc
                    a_mat%irn(ilarge2) = index_large_i + r_loc
                    a_mat%jcn(ilarge2) = index_large_k + col_loc
                  end do
                end do
              end if
              a_mat%val(ijA_position : ijA_position + block_size * block_size - 1) = &
                a_mat%val(ijA_position : ijA_position + block_size * block_size - 1) + buf_blk
            end do
          end do
        end do
        !$omp end critical(pfafem_mat_scatter)
#ifdef USE_PETSC
        end if
#endif
      end do
      !$omp end do
#ifdef USE_PETSC
      if (petsc_direct) then
        deallocate(idxm_u, idxn_u, idx_bc, raw_col, raw_u2, raw_io2, raw_slot, u2_slot, ebuf)
      end if
#endif
      deallocate(ELM_sp, RHS_u, buf_blk)
      !$omp end parallel
    end if

#ifdef USE_PETSC
    if (petsc_direct .and. (ibatch < nbatch .or. iphase < nphase - 1)) then
      PetscCallA(MatAssemblyBegin(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(a_mat%petsc_A, MAT_FLUSH_ASSEMBLY, ierr))
    end if
#endif
    end do
  end do
#else
  write(*,*) 'FATAL: use_pfafem is currently only supported in models 180, 183, and 600.'
  error stop 1
#endif
end subroutine construct_pfafem_elements


#if defined(USE_PETSC) && ((JOREK_MODEL == 180) || (JOREK_MODEL == 183) || (JOREK_MODEL == 600))
pure integer function pf_find_diag_blk(ia, ja, l_row0, col_loc0) result(p)
  PetscInt, intent(in) :: ia(:), ja(:)
  integer,  intent(in) :: l_row0, col_loc0
  integer :: lo, hi, mid, v
  lo = int(ia(l_row0 + 1)) + 1
  hi = int(ia(l_row0 + 2))
  p  = 0
  do while (lo <= hi)
    mid = (lo + hi) / 2
    v   = int(ja(mid))
    if (v == col_loc0) then
      p = mid
      return
    else if (v < col_loc0) then
      lo = mid + 1
    else
      hi = mid - 1
    end if
  end do
end function pf_find_diag_blk


pure integer function pf_find_offd_blk(ia, ja, gmap, l_row0, col_glob0) result(p)
  PetscInt, intent(in) :: ia(:), ja(:), gmap(:)
  integer,  intent(in) :: l_row0, col_glob0
  integer :: lo, hi, mid, v
  lo = int(ia(l_row0 + 1)) + 1
  hi = int(ia(l_row0 + 2))
  p  = 0
  do while (lo <= hi)
    mid = (lo + hi) / 2
    v   = int(gmap(ja(mid) + 1))
    if (v == col_glob0) then
      p = mid
      return
    else if (v < col_glob0) then
      lo = mid + 1
    else
      hi = mid - 1
    end if
  end do
end function pf_find_offd_blk


subroutine construct_pfafem_elements_petsc_direct(my_id, a_mat, rhs_local, my_ind_min, my_ind_max, &
                                                  xpoint2, xcase2, R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint)
  use iso_c_binding, only: c_ptr, c_int, c_intptr_t, c_double, c_f_pointer
  use mpi_mod
  use phys_module, only: eliminate_boundary_dofs, refinement
  use mod_parameters, only: n_tor, n_var, n_degrees
  use data_structure, only: type_SP_MATRIX
  use nodes_elements, only: node_list, element_list, aux_node_list
  use mod_pfafem, only: pfafem_cache, pfafem_el_lo, pfafem_el_hi
  use mod_petsc_raw_csr, only: split_parts, get_ij, put_ij, c_baij_get, c_baij_restore
  use omp_lib
#if JOREK_MODEL == 180
  use mod_elt_matrix, only: element_matrix_pfafem
#elif (JOREK_MODEL == 183) || (JOREK_MODEL == 600)
  use mod_elt_matrix_fft, only: element_matrix_pfafem
#endif
  implicit none

  integer,              intent(in)    :: my_id
  type(type_SP_MATRIX), intent(inout) :: a_mat
  real*8,               intent(inout) :: rhs_local(:)
  integer,              intent(in)    :: my_ind_min, my_ind_max, xcase2
  logical,              intent(in)    :: xpoint2
  real*8,               intent(in)    :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,               intent(in)    :: R_xpoint(2), Z_xpoint(2)

  type :: pf_el_sp_t
    logical :: active = .false.
    real*8, allocatable :: sp(:,:,:,:,:)
    real*8 :: diag_avg(n_var) = 1.d0
  end type pf_el_sp_t

  PetscInt, parameter :: one_p = 1
  integer :: el_lo_rank, el_hi_rank, n_el_rank, max_n_el_glob, pf_batch, n_batches, i_batch
  integer :: el_lo, el_hi, n_el_loc, n_blk_loc, block_size, unit_len, max_thr_pf, omp_tid
  integer :: i, k_el, ielm, n_u, n_vp, n_sp, n_sp_u1, cap_u_loc, cap_u_glob
  integer :: u1, u2, isp, iuj1, iuj2, jb1, jb2, io1, io2, ivar, jvar
  integer :: inode1, inode2, index_node1, index_node2, btype1, btype2, index_large_i, r_loc
  integer :: nnz_counter(n_var), nproc, mpi_ierr, mpi_hdr_type, mpi_pay_type
  integer :: max_rc, nc_i, nd_touch, d_seen(n_degrees), d_iom(n_degrees), d_bcm(n_degrees)
  integer :: dest, k_d, iom, bcm, n_io_act, n_units, total_shdr, total_spay, total_rhdr, total_rpay
  integer :: slot, p_off, u_slot, idx_u, p_pkt, r_row, n_items, it_idx, io_code, u_off_it, p_blk
  integer :: r_in, c_in, nd_blk, no_blk
  integer(8) :: bs2_8, boff8
  real*8 :: dval, elm_diagonal_average(n_var), d_avg_it(n_var)
  PetscBool :: is_assm
  PetscInt :: idxm_one(1), nc_p, nr_d, nr_o
  PetscInt, allocatable :: cols_seed(:)
  real*8,   allocatable :: zero_seed(:), RHS_u(:,:,:)
  type(pf_el_sp_t), allocatable :: el_buf(:)
  integer, allocatable :: shdr_cnt(:), shdr_dsp(:), rhdr_cnt(:), rhdr_dsp(:)
  integer, allocatable :: spay_cnt(:), spay_dsp(:), rpay_cnt(:), rpay_dsp(:)
  integer, allocatable :: shdr_cur(:), spay_cur(:)
  integer, allocatable :: send_hdr(:,:), recv_hdr(:,:)
  integer, allocatable :: send_pkt_kel(:), send_pkt_poff(:), recv_pkt_poff(:)
  real*8,  allocatable :: send_pay(:,:), recv_pay(:,:)
  integer, allocatable :: row_cnt(:), row_ptr(:), row_cur(:), item_src(:), item_io1(:), item_uoff(:)
  integer, allocatable :: u2_pblk(:,:)
  Mat :: Ad, Ao
  PetscInt, pointer :: ga(:), dia(:), dja(:), oia(:), oja(:)
  type(c_ptr) :: pvd, pvo
  real(c_double), pointer :: va_d(:), va_o(:)
  integer(c_int) :: rc
  PetscErrorCode :: ierr

  block_size = n_var * n_tor
  bs2_8      = int(block_size, 8) * int(block_size, 8)
  unit_len   = n_degrees * n_var * n_var
  n_blk_loc  = max(0, my_ind_max - my_ind_min + 1)

  ! 0. Ensure petsc_A has its permanent local Ad / Ao block-CSR pattern allocated (0 off-rank MatStash)
  PetscCallA(MatAssembled(a_mat%petsc_A, is_assm, ierr))
  if (.not. is_assm) then
    max_rc = 1
    do i = 1, n_blk_loc
      max_rc = max(max_rc, a_mat%ijA_size(i))
    end do
    allocate(cols_seed(max_rc), zero_seed(int(max_rc, 8) * bs2_8))
    zero_seed = 0.d0
    do i = 1, n_blk_loc
      nc_i = a_mat%ijA_size(i)
      if (nc_i <= 0) cycle
      idxm_one(1) = (my_ind_min - 1) + (i - 1)
      cols_seed(1:nc_i) = a_mat%irn_jcn(i, 1:nc_i) - 1
      nc_p = nc_i
      PetscCallA(MatSetValuesBlocked(a_mat%petsc_A, one_p, idxm_one, nc_p, cols_seed(1:nc_i), &
                                     zero_seed(1_8 : int(nc_i, 8) * bs2_8), INSERT_VALUES, ierr))
    end do
    deallocate(cols_seed, zero_seed)
    PetscCallA(MatAssemblyBegin(a_mat%petsc_A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd(a_mat%petsc_A, MAT_FINAL_ASSEMBLY, ierr))
  end if

  el_lo_rank = pfafem_el_lo
  el_hi_rank = pfafem_el_hi
  n_el_rank  = max(0, el_hi_rank - el_lo_rank + 1)

  call MPI_Comm_size(a_mat%comm, nproc, mpi_ierr)
  call MPI_Allreduce(n_el_rank, max_n_el_glob, 1, MPI_INTEGER, MPI_MAX, a_mat%comm, mpi_ierr)
  if (max_n_el_glob <= 0) return

  max_thr_pf = 1
  !$ max_thr_pf = omp_get_max_threads()
  pf_batch = max(8, min(16, max_thr_pf))
  call MPI_Allreduce(MPI_IN_PLACE, pf_batch, 1, MPI_INTEGER, MPI_MAX, a_mat%comm, mpi_ierr)
  n_batches = (max_n_el_glob + pf_batch - 1) / pf_batch

  cap_u_glob = 16
  do ielm = 1, element_list%n_elements
    if (refinement .and. (element_list%element(ielm)%n_sons /= 0)) cycle
    cap_u_glob = max(cap_u_glob, pfafem_cache(ielm)%n_unodes)
  end do

  allocate(shdr_cnt(0:nproc-1), shdr_dsp(0:nproc-1), rhdr_cnt(0:nproc-1), rhdr_dsp(0:nproc-1))
  allocate(spay_cnt(0:nproc-1), spay_dsp(0:nproc-1), rpay_cnt(0:nproc-1), rpay_dsp(0:nproc-1))
  allocate(shdr_cur(0:nproc-1), spay_cur(0:nproc-1))

  call MPI_Type_contiguous(4, MPI_INTEGER, mpi_hdr_type, mpi_ierr)
  call MPI_Type_commit(mpi_hdr_type, mpi_ierr)
  call MPI_Type_contiguous(unit_len, MPI_DOUBLE_PRECISION, mpi_pay_type, mpi_ierr)
  call MPI_Type_commit(mpi_pay_type, mpi_ierr)

  if (n_blk_loc > 0) then
    call split_parts(a_mat%petsc_A, .true., Ad, Ao, ga)
    call get_ij(Ad, .true., nr_d, dia, dja)
    call get_ij(Ao, .true., nr_o, oia, oja)
    nd_blk = int(dia(nr_d + 1))
    no_blk = int(oia(nr_o + 1))
    va_d => null(); va_o => null()
    if (nd_blk > 0) then
      rc = c_baij_get(transfer(Ad%v, 0_c_intptr_t), pvd)
      call c_f_pointer(pvd, va_d, [int(nd_blk, 8) * bs2_8])
    end if
    if (no_blk > 0) then
      rc = c_baij_get(transfer(Ao%v, 0_c_intptr_t), pvo)
      call c_f_pointer(pvo, va_o, [int(no_blk, 8) * bs2_8])
    end if
    allocate(row_cnt(n_blk_loc), row_ptr(n_blk_loc + 1), row_cur(n_blk_loc))
  end if

  ! Stream owned elements in lockstep batches of pf_batch (<= 16) and release el_buf
  ! before allocating recv_pay so transient assembly memory stays bounded (~5.6x lower peak).
  do i_batch = 1, n_batches
    el_lo    = el_lo_rank + (i_batch - 1) * pf_batch
    el_hi    = min(el_hi_rank, el_lo + pf_batch - 1)
    n_el_loc = max(0, el_hi - el_lo + 1)
    allocate(el_buf(max(1, n_el_loc)))

    ! 1. Evaluate this batch's owned elements across all OpenMP threads
    if (n_el_loc > 0) then
      cap_u_loc = 16
      do ielm = el_lo, el_hi
        if (refinement .and. (element_list%element(ielm)%n_sons /= 0)) cycle
        cap_u_loc = max(cap_u_loc, pfafem_cache(ielm)%n_unodes)
      end do

      !$omp parallel num_threads(max_thr_pf) default(shared) &
      !$omp   private(omp_tid, k_el, ielm, n_u, n_vp, n_sp, u1, io1, jvar, jb1, iuj1, isp, dval, &
      !$omp           nnz_counter, elm_diagonal_average, inode1, btype1, index_node1, index_large_i, r_loc, RHS_u)
#ifdef _OPENMP
      omp_tid = 1 + omp_get_thread_num()
#else
      omp_tid = 1
#endif
      allocate(RHS_u(block_size, n_degrees, cap_u_loc))

      !$omp do schedule(dynamic, 1)
      do k_el = 1, n_el_loc
        ielm = el_lo + k_el - 1
        if (refinement .and. (element_list%element(ielm)%n_sons /= 0)) cycle
        n_u  = pfafem_cache(ielm)%n_unodes
        n_vp = pfafem_cache(ielm)%n_vpairs
        n_sp = pfafem_cache(ielm)%n_sp
        if (n_u <= 0 .or. n_vp <= 0 .or. n_sp <= 0) cycle

        el_buf(k_el)%active = .true.
        allocate(el_buf(k_el)%sp(n_degrees, n_degrees, n_var, n_var, n_sp))

        call element_matrix_pfafem(node_list, element_list, ielm, xpoint2, xcase2, &
                                   R_axis, Z_axis, psi_axis, psi_bnd, R_xpoint, Z_xpoint, &
                                   el_buf(k_el)%sp, RHS_u(:, :, 1:n_u), omp_tid, aux_node_list)

        if (eliminate_boundary_dofs) then
          elm_diagonal_average = 0.d0
          nnz_counter = 0
          do u1 = 1, n_u
            do io1 = 1, n_degrees
              do jvar = 1, n_var
                do jb1 = 1, n_tor
                  iuj1 = pfafem_cache(ielm)%uj_map(u1, jb1)
                  if (iuj1 <= 0) cycle
                  isp  = pfafem_cache(ielm)%sp_map(iuj1, iuj1)
                  dval = abs(el_buf(k_el)%sp(io1, io1, jvar, jvar, isp))
                  if (dval /= 0.d0) then
                    nnz_counter(jvar) = nnz_counter(jvar) + 1
                    elm_diagonal_average(jvar) = elm_diagonal_average(jvar) + dval
                  end if
                end do
              end do
            end do
          end do
          do jvar = 1, n_var
            if (nnz_counter(jvar) > 0) then
              elm_diagonal_average(jvar) = elm_diagonal_average(jvar) / dble(nnz_counter(jvar))
            else
              elm_diagonal_average(jvar) = 1.d0
            end if
            if (.not. (elm_diagonal_average(jvar) > 0.d0)) elm_diagonal_average(jvar) = 1.d0
            elm_diagonal_average(jvar) = min(elm_diagonal_average(jvar), 1.d12)
          end do
          el_buf(k_el)%diag_avg = elm_diagonal_average
        end if

        do u1 = 1, n_u
          inode1 = pfafem_cache(ielm)%unodes(u1)
          btype1 = node_list%node(inode1)%boundary
          do io1 = 1, n_degrees
            index_node1 = node_list%node(inode1)%index(io1)
            if (index_node1 <= 0) cycle
            if (eliminate_boundary_dofs .and. is_elim_bnd_dof(btype1, io1)) cycle
            index_large_i = block_size * (index_node1 - 1)
            do r_loc = 1, block_size
              !$omp atomic
              rhs_local(index_large_i + r_loc) = rhs_local(index_large_i + r_loc) + RHS_u(r_loc, io1, u1)
              !$omp end atomic
            end do
          end do
        end do
      end do
      !$omp end do
      deallocate(RHS_u)
      !$omp end parallel
    end if

    ! 2. Pack (ielm, u1) slices into send_pay, immediately free el_buf, then exchange via MPI_Alltoallv
    shdr_cnt = 0; spay_cnt = 0
    do k_el = 1, n_el_loc
      if (.not. el_buf(k_el)%active) cycle
      ielm = el_lo + k_el - 1
      n_u  = pfafem_cache(ielm)%n_unodes
      do u1 = 1, n_u
        call classify_u1_dests(ielm, u1, nd_touch, d_seen, d_iom, d_bcm)
        if (nd_touch == 0) cycle
        n_sp_u1 = max(0, pfafem_cache(ielm)%u_sp_hi(u1) - pfafem_cache(ielm)%u_sp_lo(u1) + 1)
        do k_d = 1, nd_touch
          dest = d_seen(k_d)
          iom  = d_iom(k_d)
          bcm  = d_bcm(k_d)
          if (iom /= 0 .and. n_sp_u1 == 0) iom = 0
          if (iom == 0 .and. bcm == 0) cycle
          n_io_act = popcnt(iom)
          n_units  = merge(1, 0, bcm /= 0) + n_io_act * n_sp_u1
          shdr_cnt(dest) = shdr_cnt(dest) + 1
          spay_cnt(dest) = spay_cnt(dest) + n_units
        end do
      end do
    end do

    shdr_dsp(0) = 0; spay_dsp(0) = 0
    do i = 1, nproc - 1
      shdr_dsp(i) = shdr_dsp(i - 1) + shdr_cnt(i - 1)
      spay_dsp(i) = spay_dsp(i - 1) + spay_cnt(i - 1)
    end do
    total_shdr = shdr_dsp(nproc - 1) + shdr_cnt(nproc - 1)
    total_spay = spay_dsp(nproc - 1) + spay_cnt(nproc - 1)

    allocate(send_hdr(4, max(1, total_shdr)), send_pkt_kel(max(1, total_shdr)), send_pkt_poff(max(1, total_shdr)))
    allocate(send_pay(unit_len, max(1, total_spay)))

    shdr_cur = shdr_dsp
    spay_cur = spay_dsp
    do k_el = 1, n_el_loc
      if (.not. el_buf(k_el)%active) cycle
      ielm = el_lo + k_el - 1
      n_u  = pfafem_cache(ielm)%n_unodes
      do u1 = 1, n_u
        call classify_u1_dests(ielm, u1, nd_touch, d_seen, d_iom, d_bcm)
        if (nd_touch == 0) cycle
        n_sp_u1 = max(0, pfafem_cache(ielm)%u_sp_hi(u1) - pfafem_cache(ielm)%u_sp_lo(u1) + 1)
        do k_d = 1, nd_touch
          dest = d_seen(k_d)
          iom  = d_iom(k_d)
          bcm  = d_bcm(k_d)
          if (iom /= 0 .and. n_sp_u1 == 0) iom = 0
          if (iom == 0 .and. bcm == 0) cycle
          n_io_act = popcnt(iom)
          n_units  = merge(1, 0, bcm /= 0) + n_io_act * n_sp_u1
          slot     = shdr_cur(dest) + 1
          p_off    = spay_cur(dest) + 1
          shdr_cur(dest) = slot
          spay_cur(dest) = spay_cur(dest) + n_units
          send_hdr(1, slot)   = ielm
          send_hdr(2, slot)   = u1
          send_hdr(3, slot)   = iom
          send_hdr(4, slot)   = bcm
          send_pkt_kel(slot)  = k_el
          send_pkt_poff(slot) = p_off
        end do
      end do
    end do

    if (total_shdr > 0) then
      !$omp parallel do num_threads(max_thr_pf) schedule(dynamic, 16) &
      !$omp   private(slot, k_el, ielm, u1, iom, bcm, u_slot, io1, isp, io2, jvar, ivar, idx_u)
      do slot = 1, total_shdr
        k_el   = send_pkt_kel(slot)
        ielm   = send_hdr(1, slot)
        u1     = send_hdr(2, slot)
        iom    = send_hdr(3, slot)
        bcm    = send_hdr(4, slot)
        u_slot = send_pkt_poff(slot)
        if (bcm /= 0) then
          send_pay(:, u_slot)       = 0.d0
          send_pay(1:n_var, u_slot) = el_buf(k_el)%diag_avg(1:n_var)
          u_slot = u_slot + 1
        end if
        if (iom /= 0) then
          do io1 = 1, n_degrees
            if (.not. btest(iom, io1 - 1)) cycle
            do isp = pfafem_cache(ielm)%u_sp_lo(u1), pfafem_cache(ielm)%u_sp_hi(u1)
              idx_u = 0
              do io2 = 1, n_degrees
                do jvar = 1, n_var
                  do ivar = 1, n_var
                    idx_u = idx_u + 1
                    send_pay(idx_u, u_slot) = el_buf(k_el)%sp(io1, io2, ivar, jvar, isp)
                  end do
                end do
              end do
              u_slot = u_slot + 1
            end do
          end do
        end if
      end do
      !$omp end parallel do
    end if

    do k_el = 1, size(el_buf)
      if (allocated(el_buf(k_el)%sp)) deallocate(el_buf(k_el)%sp)
    end do
    deallocate(el_buf, send_pkt_kel, send_pkt_poff)

    call MPI_Alltoall(shdr_cnt, 1, MPI_INTEGER, rhdr_cnt, 1, MPI_INTEGER, a_mat%comm, mpi_ierr)
    call MPI_Alltoall(spay_cnt, 1, MPI_INTEGER, rpay_cnt, 1, MPI_INTEGER, a_mat%comm, mpi_ierr)

    rhdr_dsp(0) = 0; rpay_dsp(0) = 0
    do i = 1, nproc - 1
      rhdr_dsp(i) = rhdr_dsp(i - 1) + rhdr_cnt(i - 1)
      rpay_dsp(i) = rpay_dsp(i - 1) + rpay_cnt(i - 1)
    end do
    total_rhdr = rhdr_dsp(nproc - 1) + rhdr_cnt(nproc - 1)
    total_rpay = rpay_dsp(nproc - 1) + rpay_cnt(nproc - 1)

    allocate(recv_hdr(4, max(1, total_rhdr)))
    allocate(recv_pay(unit_len, max(1, total_rpay)))

    call MPI_Alltoallv(send_hdr, shdr_cnt, shdr_dsp, mpi_hdr_type, &
                       recv_hdr, rhdr_cnt, rhdr_dsp, mpi_hdr_type, a_mat%comm, mpi_ierr)
    call MPI_Alltoallv(send_pay, spay_cnt, spay_dsp, mpi_pay_type, &
                       recv_pay, rpay_cnt, rpay_dsp, mpi_pay_type, a_mat%comm, mpi_ierr)
    deallocate(send_hdr, send_pay)

    ! 3. Lock-free row-partitioned scatter directly into Ad and Ao raw BAIJ arrays
    if (n_blk_loc > 0 .and. total_rhdr > 0) then
      allocate(recv_pkt_poff(total_rhdr))
      p_off = 1
      do p_pkt = 1, total_rhdr
        recv_pkt_poff(p_pkt) = p_off
        ielm    = recv_hdr(1, p_pkt)
        u1      = recv_hdr(2, p_pkt)
        iom     = recv_hdr(3, p_pkt)
        bcm     = recv_hdr(4, p_pkt)
        n_sp_u1 = max(0, pfafem_cache(ielm)%u_sp_hi(u1) - pfafem_cache(ielm)%u_sp_lo(u1) + 1)
        p_off   = p_off + merge(1, 0, bcm /= 0) + popcnt(iom) * n_sp_u1
      end do

      row_cnt = 0
      do p_pkt = 1, total_rhdr
        ielm   = recv_hdr(1, p_pkt)
        u1     = recv_hdr(2, p_pkt)
        iom    = recv_hdr(3, p_pkt)
        bcm    = recv_hdr(4, p_pkt)
        inode1 = pfafem_cache(ielm)%unodes(u1)
        do io1 = 1, n_degrees
          r_row = node_list%node(inode1)%index(io1) - my_ind_min + 1
          if (btest(bcm, io1 - 1)) row_cnt(r_row) = row_cnt(r_row) + 1
          if (btest(iom, io1 - 1)) row_cnt(r_row) = row_cnt(r_row) + 1
        end do
      end do

      row_ptr(1) = 1
      do r_row = 1, n_blk_loc
        row_ptr(r_row + 1) = row_ptr(r_row) + row_cnt(r_row)
      end do
      n_items = row_ptr(n_blk_loc + 1) - 1
      row_cur = row_ptr(1:n_blk_loc)
      allocate(item_src(max(1, n_items)), item_io1(max(1, n_items)), item_uoff(max(1, n_items)))

      do p_pkt = 1, total_rhdr
        ielm    = recv_hdr(1, p_pkt)
        u1      = recv_hdr(2, p_pkt)
        iom     = recv_hdr(3, p_pkt)
        bcm     = recv_hdr(4, p_pkt)
        n_sp_u1 = max(0, pfafem_cache(ielm)%u_sp_hi(u1) - pfafem_cache(ielm)%u_sp_lo(u1) + 1)
        inode1  = pfafem_cache(ielm)%unodes(u1)
        u_slot  = recv_pkt_poff(p_pkt)
        if (bcm /= 0) then
          do io1 = 1, n_degrees
            if (.not. btest(bcm, io1 - 1)) cycle
            r_row             = node_list%node(inode1)%index(io1) - my_ind_min + 1
            it_idx            = row_cur(r_row); row_cur(r_row) = it_idx + 1
            item_src(it_idx)  = p_pkt
            item_io1(it_idx)  = -io1
            item_uoff(it_idx) = u_slot
          end do
          u_slot = u_slot + 1
        end if
        if (iom /= 0) then
          do io1 = 1, n_degrees
            if (.not. btest(iom, io1 - 1)) cycle
            r_row             = node_list%node(inode1)%index(io1) - my_ind_min + 1
            it_idx            = row_cur(r_row); row_cur(r_row) = it_idx + 1
            item_src(it_idx)  = p_pkt
            item_io1(it_idx)  = io1
            item_uoff(it_idx) = u_slot
            u_slot            = u_slot + n_sp_u1
          end do
        end if
      end do

      !$omp parallel num_threads(max_thr_pf) default(shared) &
      !$omp   private(r_row, it_idx, io_code, u_off_it, p_pkt, ielm, u1, io1, &
      !$omp           d_avg_it, p_blk, boff8, r_in, n_u, u2, inode2, btype2, io2, index_node2, &
      !$omp           isp, iuj1, iuj2, jb1, jb2, jvar, ivar, c_in, idx_u, u_slot, dval, u2_pblk)
      allocate(u2_pblk(n_degrees, cap_u_glob))

      !$omp do schedule(dynamic, 1)
      do r_row = 1, n_blk_loc
        do it_idx = row_ptr(r_row), row_ptr(r_row + 1) - 1
          p_pkt    = item_src(it_idx)
          io_code  = item_io1(it_idx)
          u_off_it = item_uoff(it_idx)
          ielm     = recv_hdr(1, p_pkt)
          u1       = recv_hdr(2, p_pkt)

          if (io_code < 0) then
            d_avg_it = recv_pay(1:n_var, u_off_it)
            p_blk    = pf_find_diag_blk(dia, dja, r_row - 1, r_row - 1)
            if (p_blk > 0) then
              boff8 = int(p_blk - 1, 8) * bs2_8
              do r_in = 0, block_size - 1
                va_d(boff8 + int(r_in * block_size + r_in + 1, 8)) = &
                  va_d(boff8 + int(r_in * block_size + r_in + 1, 8)) + d_avg_it(r_in / n_tor + 1)
              end do
            end if
            cycle
          end if

          io1 = io_code
          n_u = pfafem_cache(ielm)%n_unodes
          u2_pblk(1:n_degrees, 1:n_u) = 0
          do u2 = 1, n_u
            if (pfafem_cache(ielm)%pair_map(u1, u2) <= 0) cycle
            inode2 = pfafem_cache(ielm)%unodes(u2)
            btype2 = node_list%node(inode2)%boundary
            do io2 = 1, n_degrees
              index_node2 = node_list%node(inode2)%index(io2)
              if (index_node2 <= 0) cycle
              if (eliminate_boundary_dofs .and. is_elim_bnd_dof(btype2, io2)) cycle
              if (index_node2 >= my_ind_min .and. index_node2 <= my_ind_max) then
                u2_pblk(io2, u2) = pf_find_diag_blk(dia, dja, r_row - 1, index_node2 - my_ind_min)
              else
                u2_pblk(io2, u2) = -pf_find_offd_blk(oia, oja, ga, r_row - 1, index_node2 - 1)
              end if
            end do
          end do

          u_slot = u_off_it
          do isp = pfafem_cache(ielm)%u_sp_lo(u1), pfafem_cache(ielm)%u_sp_hi(u1)
            iuj1 = pfafem_cache(ielm)%sp_pairs(1, isp)
            iuj2 = pfafem_cache(ielm)%sp_pairs(2, isp)
            jb1  = pfafem_cache(ielm)%uj_jb(iuj1)
            u2   = pfafem_cache(ielm)%uj_u(iuj2)
            jb2  = pfafem_cache(ielm)%uj_jb(iuj2)

            do io2 = 1, n_degrees
              p_blk = u2_pblk(io2, u2)
              idx_u = (io2 - 1) * n_var * n_var
              if (p_blk > 0) then
                boff8 = int(p_blk - 1, 8) * bs2_8
                do jvar = 1, n_var
                  c_in = (jvar - 1) * n_tor + (jb2 - 1)
                  do ivar = 1, n_var
                    idx_u = idx_u + 1
                    r_in  = (ivar - 1) * n_tor + (jb1 - 1)
                    dval  = recv_pay(idx_u, u_slot)
                    va_d(boff8 + int(c_in * block_size + r_in + 1, 8)) = &
                      va_d(boff8 + int(c_in * block_size + r_in + 1, 8)) + dval
                  end do
                end do
              else if (p_blk < 0) then
                boff8 = int((-p_blk) - 1, 8) * bs2_8
                do jvar = 1, n_var
                  c_in = (jvar - 1) * n_tor + (jb2 - 1)
                  do ivar = 1, n_var
                    idx_u = idx_u + 1
                    r_in  = (ivar - 1) * n_tor + (jb1 - 1)
                    dval  = recv_pay(idx_u, u_slot)
                    va_o(boff8 + int(c_in * block_size + r_in + 1, 8)) = &
                      va_o(boff8 + int(c_in * block_size + r_in + 1, 8)) + dval
                  end do
                end do
              end if
            end do
            u_slot = u_slot + 1
          end do
        end do
      end do
      !$omp end do
      deallocate(u2_pblk)
      !$omp end parallel

      deallocate(recv_pkt_poff, item_src, item_io1, item_uoff)
    end if

    deallocate(recv_hdr, recv_pay)
  end do

  if (n_blk_loc > 0) then
    if (nd_blk > 0) rc = c_baij_restore(transfer(Ad%v, 0_c_intptr_t), pvd)
    if (no_blk > 0) rc = c_baij_restore(transfer(Ao%v, 0_c_intptr_t), pvo)
    call put_ij(Ad, .true., nr_d, dia, dja)
    call put_ij(Ao, .true., nr_o, oia, oja)
    deallocate(row_cnt, row_ptr, row_cur)
  end if

  call MPI_Type_free(mpi_hdr_type, mpi_ierr)
  call MPI_Type_free(mpi_pay_type, mpi_ierr)
  deallocate(shdr_cnt, shdr_dsp, rhdr_cnt, rhdr_dsp, spay_cnt, spay_dsp, rpay_cnt, rpay_dsp, shdr_cur, spay_cur)

contains

  integer function rank_of_idx(idx1) result(r_out)
    integer, intent(in) :: idx1
    integer :: lo, hi, mid
    lo = 0; hi = nproc - 1
    do while (lo < hi)
      mid = (lo + hi) / 2
      if (a_mat%index_max(mid + 1) < idx1) then
        lo = mid + 1
      else
        hi = mid
      end if
    end do
    if (idx1 >= a_mat%index_min(lo + 1) .and. idx1 <= a_mat%index_max(lo + 1)) then
      r_out = lo
    else
      r_out = -1
    end if
  end function rank_of_idx

  subroutine classify_u1_dests(ielm_, u1_, nd_, ds_, iom_, bcm_)
    integer, intent(in)  :: ielm_, u1_
    integer, intent(out) :: nd_, ds_(n_degrees), iom_(n_degrees), bcm_(n_degrees)
    integer :: in1_, bt1_, io_, idx1_, dst_, k_, pos_
    logical :: elim_
    nd_  = 0
    iom_ = 0
    bcm_ = 0
    in1_ = pfafem_cache(ielm_)%unodes(u1_)
    bt1_ = node_list%node(in1_)%boundary
    do io_ = 1, n_degrees
      idx1_ = node_list%node(in1_)%index(io_)
      if (idx1_ <= 0) cycle
      dst_ = rank_of_idx(idx1_)
      if (dst_ < 0) cycle
      pos_ = 0
      do k_ = 1, nd_
        if (ds_(k_) == dst_) then
          pos_ = k_
          exit
        end if
      end do
      if (pos_ == 0) then
        nd_       = nd_ + 1
        pos_      = nd_
        ds_(pos_) = dst_
      end if
      elim_ = eliminate_boundary_dofs .and. is_elim_bnd_dof(bt1_, io_)
      if (elim_) then
        if (pfafem_cache(ielm_)%pair_map(u1_, u1_) > 0) bcm_(pos_) = ibset(bcm_(pos_), io_ - 1)
      else
        iom_(pos_) = ibset(iom_(pos_), io_ - 1)
      end if
    end do
  end subroutine classify_u1_dests

end subroutine construct_pfafem_elements_petsc_direct
#endif

end module construct_matrix_mod

