!> This module contains some helper functions for particle projection
!> These are a few reference functions f_*, a function to project one of these,
!> functions to generate a square, polar and flux-aligned grid,
!> functions to calculate the projection RHS and a function to calculate the mean
!> and RMS error of a projected function.
module mod_projection_helpers_test_tools
use mod_project_particles
use data_structure
use mod_particle_types
implicit none
private
include 'dmumps_struc.h'        ! MUMPS include files defining its datastructure  implicit none
public :: default_flux_grid,default_square_grid,default_polar_grid
public :: project_f,broadcast_dmumps_struct_A_irn_jcn
public :: broadcast_dmumps_project_struct,calc_rhs_f
public :: elements_mean_rms,close_dmumps
public :: f_0,f_1,f_R,f_RZ,f_R4,f_a2
public :: map_matrix_to_MUMPS_datastructure

!> Variables ------------------------------------------------------
logical,parameter :: nice_q=.true.
real*8 :: acentre,angle_start
contains
!> External procedures --------------------------------------------
!> Functions to project
function f_0(R, Z)
  real*8, intent(in) :: R, Z
  real*8 :: f_0
  f_0 = 0.d0
end function f_0
function f_1(R, Z)
  real*8, intent(in) :: R, Z
  real*8 :: f_1
  f_1 = 1.d0
end function f_1
function f_R(R, Z)
  real*8, intent(in) :: R, Z
  real*8 :: f_R
  f_R = R
end function f_R
function f_RZ(R, Z)
  real*8, intent(in) :: R, Z
  real*8 :: f_RZ
  f_RZ = R*Z
end function f_RZ
function f_R4(R, Z)
  real*8, intent(in) :: R, Z
  real*8 :: f_R4
  f_R4 = R**4
end function f_R4
!> minor radius^2
function f_a2(R, Z)
  use phys_module, only: R_geo, Z_geo
  real*8, intent(in) :: R, Z
  real*8 :: f_a2
  f_a2 = norm2([(R-R_geo),Z-Z_geo])**1.8d0
end function f_a2

!> initialise parameters for generating squre grids
subroutine initialize_square_grid_parameters(&
nx,ny,Rbegin,Rend,Zbegin,Zend)
  use phys_module
  implicit none
  integer,intent(in) :: nx,ny
  real*8,intent(in)  :: Rbegin,Rend,Zbegin,Zend
  R_begin=Rbegin; R_end=Rend; Z_begin=Zbegin; Z_end=Zend; 
  n_R=nx; n_Z=ny; n_radial=0; RZ_grid_inside_wall=.false.;
end subroutine initialize_square_grid_parameters

!> Create a simple square grid with n nodes in each dimension
subroutine default_square_grid(my_id,n_cpu,nx,ny,node_list,&
element_list,ifail,Rbegin_in,Rend_in,Zbegin_in,Zend_in,&
bnd_node_list_out,bnd_element_list_out)
  use mpi_mod
  use data_structure
  use phys_module
  use basis_at_gaussian, only: initialise_basis
  use mod_initial_grid,  only: initial_grid
  use mod_element_rtree, only: populate_element_rtree
  implicit none
  integer,intent(inout)                 :: ifail
  type(type_node_list),intent(inout)    :: node_list
  type(type_element_list),intent(inout) :: element_list
  type(type_bnd_node_list),intent(out),optional    :: bnd_node_list_out
  type(type_bnd_element_list),intent(out),optional :: bnd_element_list_out
  integer, intent(in)                   :: nx,ny,my_id,n_cpu
  real*8,intent(in),optional            :: Rbegin_in,Rend_in
  real*8,intent(in),optional            :: Zbegin_in,Zend_in
  type(type_bnd_node_list)              :: bnd_node_list
  type(type_bnd_element_list)           :: bnd_elm_list
  real*8                                :: Rbegin,Rend,Zbegin,Zend
  !> set-parameters
  Rbegin = 5.d-1; if(present(Rbegin_in)) Rbegin = Rbegin_in;
  Rend   = 1.5d0; if(present(Rend_in))   Rend   = Rend_in;
  Zbegin = -5d-1; if(present(Zbegin_in)) Zbegin = Zbegin_in;
  Zend   = 5d-1;  if(present(Zend_in))   Zend   = Zend_in;
  !> compute gridi
  call tr_meminit(my_id,n_cpu) !< initialise memory tracing
  call preset_parameters()
  call initialize_square_grid_parameters(nx,ny,Rbegin,Rend,Zbegin,Zend)
  call det_modes(); call initialise_basis()
  call broadcast_phys(my_id)
  call tr_resetfile()
  call initial_grid(node_list,element_list,bnd_node_list,bnd_elm_list,my_id,n_cpu)
  call broadcast_boundary(my_id,bnd_elm_list,bnd_node_list)
  call broadcast_elements(my_id,element_list)
  call broadcast_nodes(my_id,node_list)
  call populate_element_rtree(node_list,element_list)
  if(present(bnd_node_list_out))    bnd_node_list_out      = bnd_node_list
  if(present(bnd_element_list_out)) bnd_element_list_out   = bnd_elm_list
  call MPI_Barrier(MPI_COMM_WORLD,ifail)
end subroutine default_square_grid

!> initialise parameters for generating squre grids
subroutine initialize_polar_grid_parameters(npol,nrad)
  use phys_module
  implicit none
  integer,intent(in) :: npol,nrad
  fbnd(1)=2.d0; fbnd(2:4)=0.d0; fpsi=0.d0;
  mf=0; n_radial=30; R_geo=1.5; Z_geo=0.0; 
  amin=1.0; acentre=0.d0; angle_start=0.d0;
  n_radial = nrad; n_pol = npol;
end subroutine initialize_polar_grid_parameters

!> Create a simple polar grid with npol nodes in the poloidal direction, 30 radial
!> volume = 2 pi^2 R a^2
subroutine default_polar_grid(my_id,n_cpu,npol,nrad,node_list,element_list,ifail,&
 bnd_node_list_out,bnd_element_list_out)
  use mpi_mod
  use phys_module
  use data_structure
  use basis_at_gaussian, only: initialise_basis
  use mod_initial_grid,  only: initial_grid
  use mod_element_rtree, only: populate_element_rtree
  implicit none
  integer,intent(inout)                  :: ifail
  type(type_node_list), intent(inout)    :: node_list
  type(type_element_list), intent(inout) :: element_list
  integer,intent(in)                     :: my_id,n_cpu,npol,nrad
  type(type_bnd_node_list),intent(out),optional    :: bnd_node_list_out
  type(type_bnd_element_list),intent(out),optional :: bnd_element_list_out
  type(type_bnd_node_list)               :: bnd_node_list
  type(type_bnd_element_list)            :: bnd_elm_list
  call tr_meminit(my_id,n_cpu) !< initialise memory tracing
  call preset_parameters()
  call initialize_polar_grid_parameters(npol,nrad)
  call det_modes(); call initialise_basis();
  call broadcast_phys(my_id)
  call tr_resetfile()
  call initial_grid(node_list,element_list,bnd_node_list,bnd_elm_list,my_id,n_cpu)
  call broadcast_boundary(my_id,bnd_elm_list,bnd_node_list)
  call broadcast_elements(my_id,element_list)
  call broadcast_nodes(my_id,node_list)
  call populate_element_rtree(node_list,element_list)
  if(present(bnd_node_list_out))    bnd_node_list_out      = bnd_node_list
  if(present(bnd_element_list_out)) bnd_element_list_out   = bnd_elm_list
  call MPI_Barrier(MPI_COMM_WORLD,ifail)
end subroutine default_polar_grid

!> subroutine used for setting the JOREK equilibrium parameters
!> compatible with JOREK model 600
!> inputs:
!>   npol: (integer)(optional) number of angular grid points
!>   nrad: (integer)(optional) number of radial grid points
subroutine set_test_equilibrium_parameters(npol,nrad)
  use phys_module
  implicit none
  integer,intent(in),optional :: npol,nrad
  tstep_n                   = 5.d0
  nstep_n                   = 0
  tgnum_psi                 = 2.d-1
  tgnum_T                   = 2.d-1
  tgnum_u                   = 2.d-1
  tgnum_w                   = 2.d-1
  tgnum_vpar                = 2.d-1
  tgnum_rho                 = 2.d-1
  tgnum_zj                  = 2.d-1
  min_sheath_angle          = 0.d0
  F0                        = 3.d0
  fbnd(1)                   = 2.d0
  fbnd(2:4)                 = 0.d0
  fpsi                      = 0.d0
  mf                        = 0
  n_radial                  = 41
  n_pol                     = 64
  n_flux                    = 30 
  R_geo                     = 3.d0
  Z_geo                     = 0.d0
  amin                      = 1.d0
  ellip                     = 1.d0
  tria_u                    = 0.d0
  tria_l                    = 0.d0
  quad_u                    = -0.d0
  quad_l                    = -0.d0
  xpoint                    = .false.
  xcase                     = 0
  rho_0                     = 1.d0
  rho_1                     = 0.01d0
  rho_coef(1)               = 0.d0
  rho_coef(2)               = 0.d0
  rho_coef(3)               = 0.d0
  rho_coef(4)               = 0.08d0
  rho_coef(5)               = 0.94d0
  T_0                       = 1.5d-2
  T_1                       = 3.d-4
  T_coef(1)                 = -0.66d0
  T_coef(2)                 = 0.d0
  T_coef(3)                 = 0.d0
  T_coef(4)                 = 0.08d0
  T_coef(5)                 = 0.94d0
  FF_0                      = 1.6d0
  FF_1                      = 0.d0
  FF_coef(1)                = -1.d0
  FF_coef(2)                = 0.d0
  FF_coef(3)                = 0.d0
  FF_coef(4)                = 0.03d0
  FF_coef(5)                = 1.0d0
  FF_coef(6)                = -0.06d0
  FF_coef(7)                = 0.9d0
  FF_coef(8)                = 0.07d0
  D_par                     = 0.d0
  D_perp                    = 1.d-5
  D_perp(1)                 = 1d-5
  D_perp(2)                 = 0.85d0
  D_perp(3)                 = 0.d0
  D_perp(4)                 = 0.01d0
  D_perp(5)                 = 0.92d0
  ZK_par                    = 1.d2
  ZK_perp(1)                = 1.d-5
  ZK_perp(2)                = 0.85d0
  ZK_perp(3)                = 0.d0
  ZK_perp(4)                = 0.01d0
  ZK_perp(5)                = 0.92d0
  eta                       = 1.d-5
  eta_ohmic                 = 1.d-7
  visco                     = 4.d-6
  visco_par                 = 1.d-4
  visco_old_setup           = .true.
  visco_par_num             = 1.d-11
  eta_num                   = 3.d-10
  treat_axis                = .true.
  heatsource                = 1.d-7
  particlesource            = 5.d-6
  edgeparticlesource        = 1.d-7
  edgeparticlesource_psin   = 1.d0
  edgeparticlesource_sig    = 0.03d0
  particlesource_gauss      = 1.d-7
  particlesource_gauss_psin = 0.d0
  particlesource_gauss_sig  = 0.1d0
  heatsource_gauss          = 1.d-8
  heatsource_gauss_psin     = 0.d0
  heatsource_gauss_sig      = 0.1d0
  rst_hdf5                  = 1
  iter_precon               = 22
  gmres_m                   = 20
  gmres_4                   = 1.d0
  gmres_max_iter            = 200
  gmres_tol                 = 1.d-6
  use_mumps_eq              = .true.
  use_pastix_eq             = .false.
  output_bnd_elements       = .false.
 if(present(nrad)) n_radial = nrad
 if(present(npol)) n_pol    = npol
end subroutine set_test_equilibrium_parameters

!> Create a simple flux aligned grid with npol nodes in the poloidal direction, 40 radial
!> by calculating equilibrium and creating flux aligned grid (like in jorek2_main)
subroutine default_flux_grid(my_id,n_cpu,npol,nrad,node_list,element_list,ifail,&
bnd_node_list_out,bnd_element_list_out)
  use phys_module
  use mpi_mod
  use mod_clock,         only: clck_init 
  use basis_at_gaussian, only: initialise_basis
  use mod_initial_grid,  only: initial_grid
  use mod_boundary,      only: boundary_from_grid
  use mod_element_rtree, only: populate_element_rtree
  use equil_info,        only: update_equil_state,broadcast_equil_state
  use mod_flux_grid,     only: flux_grid
  implicit none
  integer,intent(inout)                  :: ifail
  type(type_node_list), intent(inout)    :: node_list
  type(type_element_list), intent(inout) :: element_list
  integer,intent(in)                     :: my_id,n_cpu
  integer, intent(in)                    :: npol,nrad !< Number of nodes (poloidal,radial)
  type(type_bnd_node_list),intent(out),optional    :: bnd_node_list_out
  type(type_bnd_element_list),intent(out),optional :: bnd_element_list_out
  type(type_surface_list)                :: surface_list
  type(type_bnd_node_list)               :: bnd_node_list
  type(type_bnd_element_list)            :: bnd_elm_list

  !> initialisations 
  call tr_meminit(my_id,n_cpu) !< initialise memory tracing
  call clck_init(); call r3_info_init() !< initialise timing
  call det_modes() !< initialise mode and mode_type arrays
  call initialise_basis() !< initialise the basis functions
  ! Start with a polar grid
  call preset_parameters()
  ! Copy input file for simple case
  call set_test_equilibrium_parameters(npol,nrad)
  call broadcast_phys(my_id)
  !> compute th initial grid
  call tr_resetfile()
  call initial_grid(node_list,element_list,bnd_node_list,bnd_elm_list,my_id,n_cpu)
  call broadcast_boundary(my_id,bnd_elm_list,bnd_node_list)
  !> compute and update the plasma equilibrium
  call equilibrium(my_id,node_list,element_list,bnd_node_list,bnd_elm_list,xpoint,xcase,nice_q)
  if(my_id.eq.0) call update_equil_state(my_id,node_list,element_list,bnd_elm_list,xpoint,xcase) 
  !> compute the flux aligned grid and recompute the equilibrium
  call flux_grid(node_list,element_list,bnd_node_list,bnd_elm_list,my_id,n_cpu)
  call equilibrium(my_id,node_list,element_list,bnd_node_list,bnd_elm_list,xpoint,xcase,nice_q)
  if(my_id.eq.0) then 
    call update_equil_state(my_id,node_list,element_list,bnd_elm_list,xpoint,xcase)
    call boundary_from_grid(node_list,element_list,bnd_node_list,bnd_elm_list,output_bnd_elements) 
  endif
  !> broadcast all calculations
  call broadcast_boundary(my_id,bnd_elm_list,bnd_node_list)
  call broadcast_elements(my_id,element_list)
  call broadcast_nodes(my_id,node_list)
  call broadcast_equil_state(my_id)
  call populate_element_rtree(node_list,element_list)
  call update_equil_state(my_id,node_list,element_list,bnd_elm_list,xpoint,xcase)
  if(present(bnd_node_list_out))    bnd_node_list_out      = bnd_node_list
  if(present(bnd_element_list_out)) bnd_element_list_out   = bnd_elm_list
  call MPI_Barrier(MPI_COMM_WORLD,ifail)
end subroutine default_flux_grid

!> Project a function onto the JOREK elements
subroutine project_f(rank,master,node_list,element_list,f,ifail,filter,filter_hyper,integral,&
  apply_dirichlet_bnd_in)
  use mod_uncoupled_projection, only: assemble_projection_matrix, assemble_projection_matrix_n0
  use mpi_mod
  type(type_node_list), intent(inout)    :: node_list
  type(type_element_list), intent(inout) :: element_list
  integer,intent(inout)                  :: ifail
  integer,intent(in)                     :: rank,master
  real*8, external                       :: f
  real*8, optional, intent(in)           :: filter, filter_hyper !< smoothing and hyper-smoothing
  logical,intent(in),optional            :: apply_dirichlet_bnd_in
  real*8, optional, intent(out)          :: integral !< The integral of the projected function, from the weights
  real*8, dimension(:), allocatable      :: this_integral_weights
  type(DMUMPS_STRUC)                     :: p
  type(type_SP_MATRIX)                   :: a_mat
  integer :: i, k, index, i_tor_local, n_tor_local, mpi_comm_n, mpi_comm_master, ierr
  real*8  :: my_filter, my_filter_hyper, area, volume
  logical :: apply_dirichlet_bnd

  call MPI_Comm_dup(MPI_COMM_WORLD,mpi_comm_n,ierr)
  call MPI_Comm_dup(MPI_COMM_WORLD,mpi_comm_master,ierr)

  i_tor_local     = 1
  n_tor_local     = 1

  my_filter       = 0.d0
  my_filter_hyper = 0.d0

  apply_dirichlet_bnd = .true.

  if (present(filter))       my_filter       = filter 
  if (present(filter_hyper)) my_filter_hyper = filter_hyper 
  if (present(apply_dirichlet_bnd_in)) apply_dirichlet_bnd = apply_dirichlet_bnd_in
  
  if (i_tor_local .eq. 1) then  
    call assemble_projection_matrix_n0(node_list,element_list,n_tor_local,i_tor_local,mpi_comm_world,mpi_comm_n,&
         mpi_comm_master,a_mat,area,volume,filter=my_filter,filter_hyper=my_filter_hyper,&
         filter_parallel=0.d0,apply_dirichlet_condition_in=apply_dirichlet_bnd,&
         integral_weights=this_integral_weights)
  else
    call assemble_projection_matrix(node_list,element_list,n_tor_local,i_tor_local,mpi_comm_world,mpi_comm_n,&
         mpi_comm_master,a_mat,filter=my_filter,filter_hyper=my_filter_hyper,filter_parallel=0.d0,&
         apply_dirichlet_condition_in=apply_dirichlet_bnd)
  endif

  call map_matrix_to_MUMPS_datastructure(a_mat, p)

  ! Project manually
  p%JOB = 3
  p%icntl(21) = 0 ! solution is available only on host
  p%icntl(4)  = 1 ! print only errors

  ! Setup RHS by integrating manually
  allocate(p%rhs(p%n))
  call calc_rhs_f(node_list,element_list,f,p%rhs)

  call DMUMPS(p)

  !> broadcast solution to all MPI tasks
  call broadcast_dmumps_project_struct(rank,master,p,ifail)

  if (present(integral)) integral = dot_product(p%rhs, this_integral_weights)

  do i=1,node_list%n_nodes
    do k=1,n_degrees
      index = 2*(node_list%node(i)%index(k)-1) + 1
      node_list%node(i)%values(1,k,1) = p%rhs(index)
    enddo
  enddo

  p%JOB=-2
  call DMUMPS(p)
end subroutine project_f

!> Calculate the right-hand side of a distribution f (toroidally symmetric)
subroutine calc_rhs_f(node_list,element_list,f,rhs)
  use basis_at_gaussian
  use phys_module, only : TWOPI
  type(type_node_list),    intent(inout) :: node_list
  type(type_element_list), intent(inout) :: element_list
  real*8, external   :: f
  real*8, dimension(:), intent(inout) :: rhs
  integer            :: i, j, k, m, index, i_elm, inode, ms, mt
  real*8, dimension(n_gauss,n_gauss) :: x_g, y_g, x_s, x_t, y_s, y_t
  real*8             :: wst, xjac, v
  type(type_node)    :: nodes(4)
  type(type_element) :: element

  rhs = 0.d0

  do i_elm=1,element_list%n_elements

    element = element_list%element(i_elm)
  
    do m=1,n_vertex_max
      nodes(m) = node_list%node(element%vertex(m))
    enddo

    ! Set up gauss points in this element
    x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_g = 0.d0; y_s = 0.d0; y_t = 0.d0
    do i=1,n_vertex_max
      do j=1,n_degrees
        do ms=1, n_gauss
          do mt=1, n_gauss
            x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H(i,j,ms,mt)
            y_g(ms,mt) = y_g(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H(i,j,ms,mt)

            x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_s(i,j,ms,mt)
            x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_t(i,j,ms,mt)

            y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_s(i,j,ms,mt)
            y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_t(i,j,ms,mt)
          enddo
        enddo
      enddo
    enddo

    ! Perform gauss integration of RHS
    do ms=1, n_gauss
      do mt=1, n_gauss
    
        wst  = wgauss(ms)*wgauss(mt)
        xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)

        do i=1,n_vertex_max
          do j=1,n_degrees

            index = 2*(nodes(i)%index(j)-1) + 1

            v   = h(i,j,ms,mt)  * element%size(i,j)
            rhs(index) = rhs(index) + f(x_g(ms,mt), y_g(ms,mt)) * v * xjac * x_g(ms,mt) * wst * TWOPI
          enddo
        enddo
      enddo
    enddo
  enddo
end subroutine calc_rhs_f



subroutine elements_mean_rms(node_list, element_list, f, mean, rms, volume)
  use basis_at_gaussian
  use constants, only: TWOPI
  use mod_interp, only: interp
  type(type_node_list), intent(in) :: node_list
  type(type_element_list), intent(in) :: element_list
  real*8, external :: f
  real*8, intent(out) :: mean, rms
  real*8, intent(out), optional :: volume

  integer :: i_elm, m, i, j, ms, mt
  type(type_element) :: element
  type(type_node) :: nodes(4)
  real*8, dimension(n_gauss,n_gauss) :: x_g, x_s, x_t, y_g, y_s, y_t
  real*8 :: my_ref, wst, my_volume, xjac
  real*8 :: P, P_s, P_t, P_st, P_ss, P_tt

  call initialise_basis

  my_volume = 0.d0
  mean = 0.d0
  rms = 0.d0

  do i_elm=1,element_list%n_elements

    element = element_list%element(i_elm)
    do m=1,n_vertex_max
      nodes(m) = node_list%node(element%vertex(m))
    enddo

    ! Set up gauss points in this element
    x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_g = 0.d0; y_s = 0.d0; y_t = 0.d0
    do i=1,n_vertex_max
      do j=1,n_degrees
        do ms=1, n_gauss
          do mt=1, n_gauss
            x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H(i,j,ms,mt)
            y_g(ms,mt) = y_g(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H(i,j,ms,mt)

            x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_s(i,j,ms,mt)
            x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_t(i,j,ms,mt)

            y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_s(i,j,ms,mt)
            y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_t(i,j,ms,mt)
          enddo
        enddo
      enddo
    enddo

    ! Perform gauss integration of LHS
    do ms=1, n_gauss
      do mt=1, n_gauss

        wst       = wgauss(ms)*wgauss(mt)
        xjac      =  x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)
        my_volume = my_volume + TWOPI * x_g(ms,mt) * xjac * wst

        ! calculate contribution to integral of this point
        call interp(node_list, element_list, i_elm, 1, 1, Xgauss(ms), Xgauss(mt), P, P_s, P_t, P_st, P_ss, P_tt)
      
        rms  = rms  + (P-f(x_g(ms,mt),y_g(ms,mt)))**2 * xjac * TWOPI * x_g(ms,mt) * wst
        mean = mean + P * xjac * TWOPI * x_g(ms,mt) * wst
        
        if (xjac .lt. 0) write(*,*) i_elm, ms, mt, xjac, x_g(ms,mt), wst
        
      enddo
    enddo
  enddo

  rms = sqrt(rms / my_volume)
  mean = mean/my_volume
  if (present(volume)) volume = my_volume
end subroutine elements_mean_rms

!> construct the projected matrix from mumps data
subroutine construct_matrix_from_mumps(mumps_data,matrix)
  type(DMUMPS_STRUC),intent(inout)              :: mumps_data
  real*8,dimension(:,:),allocatable,intent(out) :: matrix
  integer :: ii
  if(allocated(matrix)) deallocate(matrix)
  allocate(matrix(minval(mumps_data%irn):maxval(mumps_data%irn),&
  minval(mumps_data%jcn):maxval(mumps_data%jcn))); matrix = 0.d0;
  do ii=1,size(mumps_data%A)
    matrix(mumps_data%irn(ii),mumps_data%jcn(ii)) = &
    matrix(mumps_data%irn(ii),mumps_data%jcn(ii)) + mumps_data%A(ii)
  enddo
end subroutine construct_matrix_from_mumps

!> broadcast dmumps structure A,irn,jcn
!> inputs:
!>   rank:       (integer) MPI task rank
!>   master:     (integer) ID of the MPI master task
!>   mumps_data: (DMUMPS_STRUC) MUMPS structure in double precision
!>   ifail:      (integer) MPI failure/error code
!> outputs:
!>   mumps_data: (DMUMPS_STRUC) MUMPS structure in double precision
!>   ifail:      (integer) MPI failure/error code
subroutine broadcast_dmumps_struct_A_irn_jcn(rank,master,mumps_data,ifail)
 use mpi_mod
 implicit none
 type(DMUMPS_STRUC),intent(inout) :: mumps_data
 integer,intent(inout) :: ifail
 integer,intent(in)    :: rank,master
 integer,dimension(3)  :: n_sizes
 if(rank.eq.master) then
   n_sizes = (/size(mumps_data%irn),size(mumps_data%jcn),size(mumps_data%A)/)
 endif
 call MPI_Bcast(n_sizes,size(n_sizes),MPI_INTEGER,master,MPI_COMM_WORLD,ifail)
 if(rank.ne.master) then
   allocate(mumps_data%irn(n_sizes(1)),mumps_data%jcn(n_sizes(2)),&
   mumps_data%A(n_sizes(3)))
 endif
 call MPI_Bcast(mumps_data%irn,n_sizes(1),MPI_INTEGER,master,MPI_COMM_WORLD,ifail)
 call MPI_Bcast(mumps_data%jcn,n_sizes(2),MPI_INTEGER,master,MPI_COMM_WORLD,ifail)
 call MPI_Bcast(mumps_data%A,n_sizes(3),MPI_REAL8,master,MPI_COMM_WORLD,ifail)
end subroutine broadcast_dmumps_struct_A_irn_jcn

!> broadcast dumumps structure used for projections
!> inputs:
!>   rank:       (integer) MPI task rank
!>   master:     (integer) ID of the MPI master task
!>   mumps_data: (DMUMPS_STRUC) MUMPS structure in double precision
!>   ifail:      (integer) MPI failure/error code
!> outputs:
!>   mumps_data: (DMUMPS_STRUC) MUMPS structure in double precision
!>   ifail:      (integer) MPI failure/error code
subroutine broadcast_dmumps_project_struct(rank,master,mumps_data,ifail)
  use mpi_mod
  type(DMUMPS_STRUC),intent(inout) :: mumps_data
  integer,intent(inout) :: ifail
  integer,intent(in)    :: rank,master
  integer,dimension(12) :: struct_integers
  if(rank.eq.master) then
    struct_integers = (/mumps_data%job,mumps_data%sym,mumps_data%par,&
    mumps_data%n,mumps_data%nrhs,mumps_data%lrhs,mumps_data%nz,&
    size(mumps_data%jcn),size(mumps_data%irn),size(mumps_data%icntl),&
    size(mumps_data%A),size(mumps_data%rhs)/)
  endif
  call MPI_Bcast(struct_integers,size(struct_integers),MPI_INTEGER,master,MPI_COMM_WORLD,ifail)
  if(rank.ne.master) then
    mumps_data%job  = struct_integers(1); mumps_data%sym  = struct_integers(2);
    mumps_data%par  = struct_integers(3); mumps_data%n    = struct_integers(4);
    mumps_data%nrhs = struct_integers(5); mumps_data%lrhs = struct_integers(6);
    mumps_data%nz   = struct_integers(7); 
    allocate(mumps_data%jcn(struct_integers(8)),mumps_data%irn(struct_integers(9)),&
    mumps_data%A(struct_integers(11)),mumps_data%rhs(struct_integers(12)))
  endif
  call MPI_Bcast(mumps_data%jcn,struct_integers(8),MPI_INTEGER,master,MPI_COMM_WORLD,ifail)
  call MPI_Bcast(mumps_data%irn,struct_integers(9),MPI_INTEGER,master,MPI_COMM_WORLD,ifail)
  call MPI_Bcast(mumps_data%icntl,struct_integers(10),MPI_INTEGER,master,MPI_COMM_WORLD,ifail)
  call MPI_Bcast(mumps_data%A,struct_integers(11),MPI_REAL8,master,MPI_COMM_WORLD,ifail)
  call MPI_Bcast(mumps_data%rhs,struct_integers(12),MPI_REAL8,master,MPI_COMM_WORLD,ifail)
end subroutine broadcast_dmumps_project_struct


subroutine map_matrix_to_MUMPS_datastructure(a_mat, mumps_par)
  use, intrinsic :: ieee_exceptions
  implicit none
  type (type_SP_MATRIX), intent(in)      :: a_mat       !< Projection matrix using our datastructures
  type (DMUMPS_STRUC)  , intent(inout)   :: mumps_par   !< Object used by mumps for solving linear systems (the matrix wil be copied inside of this)
  logical :: halt(size(IEEE_USUAL,1)), found_nan
  integer :: my_id_n, ierr

  ! initialise MUMPS
  mumps_par%COMM = a_mat%comm
  mumps_par%JOB  = -1
  mumps_par%SYM  = 0
  mumps_par%PAR  = 1

  call MPI_COMM_RANK(mumps_par%COMM, my_id_n, ierr)

  call DMUMPS(mumps_par)
 
  if (my_id_n .eq. 0) then

    ! allocate(mumps_par%irn(a_mat%nnz),mumps_par%jcn(a_mat%nnz),mumps_par%A(a_mat%nnz))

    ! map already constructed matrix to mumps datastructure
    mumps_par%irn => a_mat%irn
    mumps_par%jcn => a_mat%jcn
    mumps_par%A   => a_mat%val

  endif

  mumps_par%n   = a_mat%ng
  mumps_par%nz  = a_mat%nnz

  ! parameters for factorization
  mumps_par%JOB       = 4
  mumps_par%icntl(2)  = 6 ! print diagnostics, statistics and warnings to stderr
  mumps_par%icntl(4)  = 1 ! print errors(1), debug(2), much(3)
  mumps_par%icntl(5)  = 0 ! assembled form
  mumps_par%icntl(18) = 0 ! centralized input matrix (i.e. only on cpu 0)
  mumps_par%icntl(7)  = 7 ! compute symmetric permutation (PORD or SCOTCH autoselect)
  mumps_par%icntl(8)  = 8 ! scaling
  mumps_par%icntl(14) = 80 ! memory relaxation parameter  

  call ieee_get_halting_mode(IEEE_USUAL, halt)
  call ieee_set_halting_mode(IEEE_USUAL, [.false., .false., .false.])
  call DMUMPS(mumps_par)
  call ieee_set_halting_mode(IEEE_USUAL, halt)

end subroutine map_matrix_to_MUMPS_datastructure

!> close dmumps
subroutine close_dmumps(mumps_data)
implicit none
  type(DMUMPS_STRUC),intent(inout) :: mumps_data
  mumps_data%JOB=-2
  call DMUMPS(mumps_data)
end subroutine close_dmumps

!> Internal procedures --------------------------------------------
!> ----------------------------------------------------------------
end module mod_projection_helpers_test_tools
