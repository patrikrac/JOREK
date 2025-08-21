module mod_projection_type

  use mod_vtk, only: vtk_grid
  use data_structure, only: type_SP_MATRIX, type_RHS
  use mod_sparse_data, only: type_SP_SOLVER
  use mod_rhs_projections, only: proj_f
  use data_structure, only: type_node_list, type_element_list
  use mod_io_actions, only: io_action

  !> Action to project all particle distributions and save them to vtk.
  !> You must use the (new_)projection() constructor to set this up.
  !> At construction time the matrix is solved.
  type, abstract, extends(io_action) :: t_projection
    type(type_node_list),    pointer :: node_list !< node lists to save particle projections in
    type(type_element_list), pointer :: element_list

    real*8 :: filter                      !< Smoothing factor used for this projection (Laplacian, poloidal plane)
    real*8 :: filter_hyper                !< hyper-smoothing factor used for this projection (double Laplacian, poloidal plane)
    real*8 :: filter_parallel             !< smoothing factor used for this projection (parallel direction)

    real*8 :: filter_n0                   !< Smoothing factor used for this projection for n=0 
    real*8 :: filter_hyper_n0             !< hyper-smoothing factor used for this projection for n=0
    real*8 :: filter_parallel_n0          !< smoothing factor used for this projection (parallel direction) for n=0

    real*8 :: scaling_integral_weights    !< multiplication factor to subtract integral_weights from rhs(n=0)

    logical,public :: do_zonal = .false.  !< solve zonal flow system for n=0 (instead of usual projection)

    !> Output storage (optional)
    type(vtk_grid), allocatable :: vtk_grid !< if allocated output to vtk
    logical, public :: to_h5 = .false.    !< Output to hdf5 file
    logical, public :: index_h5 = .false. !< Number projection outputs (vtk, or hdf5) in the same way as its fluid counterpart (e.g. projections00100.vtk(h5))
                                          !< if set false, outputs will be numbered by physical time.

    !> Right-hand side
    type(proj_f), dimension(:),   allocatable :: f   !< List of projection transformations to use (n_proj)
    real*8, dimension(:,:,:,:,:), allocatable :: rhs !< dim (n_degrees,n_vertex_max,n_elements,n_tor,n_proj2)
    !< right-hand side for accumulation during sampling
    !< assumed to be filled by the user. Will be MPI_Reduced (+) before projecting
    !< n_proj + n_proj2 should be less than n_var (extra input will be ignored)
    !< After projection this will be zeroed but not deallocated
    real*8 :: rhs_gather_time = 0.d0 !< Time that the rhs has been integrated over (used for normalisation)
    !< note that this does not really work very well for multiple groups with
    !< different timesteps

    logical :: calc_integrals = .true.    !< Calculate and print integrals of all projected quantities over the entire volume
    !< (for n=0 only)
    !> Note that the integral is just a weighted sum over the node values.
    !> Precalculate the weights during the projection matrix assembly step and store
    !> them here.
    real*8, dimension(:), allocatable     :: integral_weights !< Weights per basis function towards full integral
    !< indexing is as in rhs, and the weight is the volume of the basis function

    real*8 :: area, volume

    real*8, dimension(:,:,:,:,:), allocatable :: rhs_f !< dim (n_degrees,n_vertex_max,n_elements,n_tor,n_proj) storage
    !< location for proj_f output

    logical :: apply_dirichlet   ! if .true. (default) the Dirichlet boundary conditions are applied

    integer :: mpi_comm_world    ! mpi communicator of the whole world
    integer :: mpi_comm_n        ! mpi communicator of each toroidal harmonic
    integer :: mpi_comm_master   ! mpi communicator of the group of masters (of each harmonic)
    integer :: my_id             ! mpi id within the world 
    integer :: my_id_n           ! mpi id with comm_n
    integer :: mpi_group_world
    integer :: mpi_group_master
    integer :: n_cpu, m_cpu      ! n_cpu : the total number of cores, m_cpu the number of cores per harmonic
    integer :: n_tor_local       ! 1 or 2 : (1) or (cos,sin)
    integer :: i_tor_local       ! the starting index in the array of toroidal hamonics (as in HZ)
    integer :: n_dof             ! the number of unknowns for (n=0)

    !> Ihor's Backend Datastructures
    type (type_SP_MATRIX) :: a_mat    !< sparse matrix to solve for the projection
    type (type_RHS)       :: rhs_vec  !< rhs of the projection
    type (type_SP_SOLVER) :: solver   !< solver struct

    integer :: system_size !< size of the system to solve, previously a_mat%ng, but if we distribute the matrix, it is being overwritten by the number of local rows owned by the process

  end type t_projection

end module mod_projection_type