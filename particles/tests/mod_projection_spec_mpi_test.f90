!> This module contains some testcases for projections, ensuring
!> that the projection matrix, RHS and MUMPS work for these cases.
!>
!> It contains tests of projecting zero, x, xy, x^4 onto square,
!> flux-alignes or circular grid
module mod_projection_spec_mpi_test
use fruit
use fruit_mpi
use data_structure, only: type_node_list,type_element_list, init_node_list, dealloc_node_list
use mod_parameters, only: n_nodes_max
use mod_model_settings, only: n_var
implicit none
include 'dmumps_struc.h'        ! MUMPS include files defining its datastructure
private
public :: run_fruit_projection_spec_mpi
!> Variables --------------------------------------
!> Set to true to wrtie restart files with the projected density
logical,parameter :: write_proj_output=.false.
integer,parameter :: master_rank=0
integer,parameter :: message_len=100
integer,parameter :: filename_len=100
integer,parameter :: n_fields_1=1
integer,parameter :: n_conv_min=3
integer,parameter :: n_conv_max=40
integer,parameter :: n_conv_step=4
integer,parameter :: i_tor2D=1
integer,parameter :: n_tor2D=1
integer,parameter :: n_omp_threads=8
real*8,parameter  :: time_sol=0.d0
real*8,parameter  :: R_gaussian=1.d0
real*8,parameter  :: Z_gaussian=0.d0
real*8,parameter  :: a_gaussian=5.d-3
real*8,parameter  :: Rbegin2D=5.d-1
real*8,parameter  :: Rend2D=1.5d0
real*8,parameter  :: Zbegin2D=-5.d-1
real*8,parameter  :: Zend2D=5.d-1
real*8,parameter  :: filter2D=0.d0
real*8,parameter  :: hyper_filter2D=0.d0
real*8,parameter  :: parallel_filter2D=0.d0
real*8,parameter  :: tol2D=1d-6
logical,parameter :: apply_dirichlet_bnd=.false.
integer,dimension(2),parameter :: nx=(/10,20/) !< x-dimension size of square mesh
integer,dimension(2),parameter :: ny=(/10,20/) !< y-dimension size of square mesh
integer,dimension(2),parameter :: nx2D=(/2,10/)
integer,dimension(2),parameter :: ny2D=(/2,10/)
integer,dimension(1),parameter :: nrad=(/30/) !< size of the radial mesh
integer,dimension(2),parameter :: npol=(/31,32/) !< size of the angular mesh
real*8,dimension(11),parameter  :: mean_sol=(/0.d0,1.d0,1.d0,1.d0,1.d0,1.d0,26.d0/24.d0,0.d0,1.89583333333333d0,1.89583333333333d0,0.785398d-4/)
real*8,dimension(11),parameter  :: rms_sol=(/0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.397457d-4/)
real*8,dimension(16),parameter  :: ref2D(16) = (/0.1008163, 0.0159184,  0.0142177,   0.0022449, & ! index 1
                                               0.0477551,-0.0110544,  0.00673469, -0.00155896,& ! index 2
                                               0.034898,  0.0055102, -0.00840136, -0.00132653,& ! index 3 (but node 4, because index is switched with matrix order)
                                               0.0165306,-0.00382653,-0.00397959,  0.000921202/) ! index 4 (but node 3)
type(type_node_list),pointer    :: test_node_list
type(type_element_list),pointer :: test_element_list
integer                         :: rank_loc,n_tasks_loc,ifail_loc

contains
!> Fruit basket -----------------------------------
subroutine run_fruit_projection_spec_mpi(rank,n_tasks,ifail)
  use mpi_mod
  implicit none
  integer,intent(inout) :: ifail
  integer,intent(in)    :: rank,n_tasks
  write(*,'(/A)') "  ... setting-up: projection spec"
  call setup(rank,n_tasks,ifail)
  write(*,'(/A)') "  ... running: projection spec"
  call run_test_case(test_project_0_square,'test_project_0_square')
  call run_test_case(test_project_1_square,'test_project_1_square')
  call run_test_case(test_project_1_polar_odd,'test_project_1_polar_odd')
  call run_test_case(test_project_1_polar_even,'test_project_1_polar_even')
  call run_test_case(test_project_1_flux_odd,'test_project_1_flux_odd')
  call run_test_case(test_project_1_flux_even,'test_project_1_flux_even')
  call run_test_case(test_projection_matrix_square,'test_projection_matrix_square')
  call run_test_case(test_omp_projection_matrix_square,'test_omp_projection_matrix_square')
  write(*,'(/A)') "  ... tearing-down: projection spec"
  call teardown(rank,n_tasks,ifail)
end subroutine run_fruit_projection_spec_mpi

!> Set-ups tear-downs -----------------------------
subroutine setup(rank,n_tasks,ifail)
  implicit none
  integer,intent(inout) :: ifail
  integer,intent(in)    :: rank,n_tasks
  rank_loc=rank; n_tasks_loc=n_tasks; ifail_loc=ifail;
  allocate(test_node_list,test_element_list)
  call init_node_list(test_node_list, n_nodes_max, test_node_list%n_dof, n_var)
end subroutine setup

subroutine teardown(rank,n_tasks,ifail)
  implicit none
  integer,intent(inout) :: ifail
  integer,intent(in)    :: rank,n_tasks
  rank_loc=-1; n_tasks_loc=0; ifail=ifail_loc;
  deallocate(test_node_list,test_element_list)
end subroutine teardown

!> Tests ------------------------------------------
!> Project zero onto squre grid
subroutine test_project_0_square()
  use mod_projection_helpers_test_tools, only: f_0,default_square_grid
  implicit none
  character(len=message_len),parameter :: message='Error project f_0 square test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_0_square_',nx(1),'_',ny(1)
  call default_square_grid(rank_loc,n_tasks_loc,nx(1),ny(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_0,mean_sol(1),rms_sol(1),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_0_square 

!> Project one onto a square grid
subroutine test_project_1_square()
  use mod_parameters,                    only: n_degrees
  use mod_projection_helpers_test_tools, only: f_1,default_square_grid
  implicit none
  integer                                  :: ii,jj
  integer,dimension(nx(1)*ny(1)*n_degrees) :: index
  character(len=message_len),parameter     :: message='Error project f_1 square test'
  character(len=filename_len)              :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_1_square_',nx(1),'_',ny(1)
  call default_square_grid(rank_loc,n_tasks_loc,&
  nx(1),ny(1),test_node_list,test_element_list,ifail_loc)
  !> include a grid_bezier_square test here
  !> verify that a node shares the same index and all indicies are used exactly once
  index = 0
  do ii=1,test_node_list%n_nodes
    do jj=1,n_degrees
      index(test_node_list%node(ii)%index(jj)) = index(test_node_list%node(ii)%index(jj))+1
    enddo
  enddo
  call assert_equals(size(index,1),count(index.gt.0),trim(message)//': all indices must be used!')
  call assert_equals(0,count(index.gt.1),trim(message)//': no duplicate indices in this grid!')
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_1,mean_sol(2),rms_sol(2),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_1_square

!> Project one onto two polar grids. One with an even number of elements in the poloidal direction
!> and one with an odd number of elements. For the polar grid this should not matter much
subroutine test_project_1_polar_odd
  use mod_projection_helpers_test_tools, only: f_1,default_polar_grid
  implicit none
  character(len=message_len),parameter :: message='Error project f_1 polar odd test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_1_polar_',npol(1),'_',nrad(1)
  call default_polar_grid(rank_loc,n_tasks_loc,npol(1),nrad(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_1,mean_sol(3),rms_sol(3),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_1_polar_odd

subroutine test_project_1_polar_even
  use mod_projection_helpers_test_tools, only: f_1,default_polar_grid
  implicit none
  character(len=message_len),parameter :: message='Error project f_1 polar even test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_1_polar_',npol(2),'_',nrad(1)
  call default_polar_grid(rank_loc,n_tasks_loc,npol(2),nrad(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_1,mean_sol(4),rms_sol(4),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_1_polar_even

subroutine test_project_1_flux_odd
  use mod_projection_helpers_test_tools, only: f_1,default_flux_grid
  implicit none
  character(len=message_len),parameter :: message='Error project f_1 flux odd test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_1_flux_',npol(1),'_',nrad(1)
  call default_flux_grid(rank_loc,n_tasks_loc,npol(1),nrad(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_1,mean_sol(5),rms_sol(5),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_1_flux_odd

subroutine test_project_1_flux_even
  use mod_projection_helpers_test_tools, only: f_1,default_flux_grid
  implicit none
  character(len=message_len),parameter :: message='Error project f_1 flux even test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_1_flux_',npol(2),'_',nrad(1)
  call default_flux_grid(rank_loc,n_tasks_loc,npol(2),nrad(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_1,mean_sol(6),rms_sol(6),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_1_flux_even

!> Project R onto a square grid
subroutine test_project_R_square_10_10
  use mod_projection_helpers_test_tools, only: f_R,default_square_grid
  implicit none
  character(len=message_len),parameter :: message='Error project f_R square test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_R_square_',nx(1),'_',ny(1)
  call default_square_grid(rank_loc,n_tasks_loc,nx(1),ny(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_R,mean_sol(7),rms_sol(7),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_R_square_10_10

!> Project RZ onto a square grid
subroutine test_project_RZ_square_10_10
  use mod_projection_helpers_test_tools, only: f_RZ,default_square_grid
  implicit none
  character(len=message_len),parameter :: message='Error project f_RZ square test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_RZ_square_',nx(1),'_',ny(1)
  call default_square_grid(rank_loc,n_tasks_loc,nx(1),ny(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_RZ,mean_sol(8),rms_sol(8),trim(filename),trim(message),apply_dirichlet_bnd)
end subroutine test_project_RZ_square_10_10

!> Project R^4 onto a square grid
subroutine test_project_R4_square_10_10
  use mod_projection_helpers_test_tools, only: f_R4,default_square_grid
  implicit none
  real*8,parameter                     :: tol=3.d-6
  character(len=message_len),parameter :: message='Error project f_R4 square test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_R4_square_',nx(1),'_',ny(1)
  call default_square_grid(rank_loc,n_tasks_loc,nx(1),ny(1),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_R4,mean_sol(9),rms_sol(9),trim(filename),trim(message),apply_dirichlet_bnd,&
  rms_tol_in=tol)
end subroutine test_project_R4_square_10_10

!> Project R^4 onto a few square grids and verify convergence with n
subroutine test_project_R4_square_convergence
  use mod_projection_helpers_test_tools, only: f_R4,default_square_grid
  implicit none
  integer,parameter                    :: conv_order=4
  real*8,parameter                     :: tol_base=6.d-2
  integer                              :: ii
  character(len=message_len)           :: message
  character(len=filename_len)          :: filename
  do ii=n_conv_min,n_conv_max,n_conv_step
    write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_R4_convergence_',ii,'_',ii
    write(message,'(A,I0)') 'Error project f_R4 convergence test ',ii
    call default_square_grid(rank_loc,n_tasks_loc,ii,ii,&
    test_node_list,test_element_list,ifail_loc)
    call project_f_with_assert_and_write(test_node_list,test_element_list,&
    f_R4,mean_sol(10),rms_sol(10),trim(filename),trim(message),apply_dirichlet_bnd,&
    rms_tol_in=(tol_base/(real(ii,kind=8)**conv_order)))
  enddo
end subroutine test_project_R4_square_convergence

!> Project a peaked function onto a square grid
!> the mean-tol is enormous since we do not reproduce
!> this well for very peaked distributions, with ngauss=4,
!> if ngauss=8 we get to within 20%, but the RMS is much larger
subroutine test_project_peak_gaussian_square_20_20
  use mod_projection_helpers_test_tools, only: f_R4,default_square_grid
  implicit none
  real*8,parameter                     :: rmstol=3.d-6
  real*8,parameter                     :: meantol=0.785398d-4
  real*8,parameter                     :: filter_gauss=1.d-4
  real*8,parameter                     :: hyper_filter_gauss=4.d-8
  character(len=message_len),parameter :: message='Error project f_peak square test'
  character(len=filename_len)          :: filename
  write(filename,'(A,I0,A,I0,A,I0)') 'rank_',rank_loc,'_peak_gaussian_square_',nx(2),'_',ny(2)
  call default_square_grid(rank_loc,n_tasks_loc,nx(2),ny(2),&
  test_node_list,test_element_list,ifail_loc)
  call project_f_with_assert_and_write(test_node_list,test_element_list,&
  f_R4,mean_sol(11),rms_sol(11),trim(filename),trim(message),apply_dirichlet_bnd,&
  mean_tol_in=meantol,rms_tol_in=rmstol,filter_in=filter_gauss,&
  hyper_filter_in=hyper_filter_gauss)
end subroutine test_project_peak_gaussian_square_20_20

!> Test the exact form of the projection matrix for a simple grid.
!> Reference integrals calculated with Mathematica
subroutine test_projection_matrix_square
  use constants,                         only: TWOPI
  use mod_projection_helpers_test_tools, only: close_dmumps
  implicit none
  integer,parameter          :: nstep_test=2
  type(DMUMPS_STRUC)         :: mumps_data
  integer                    :: ii
  character(len=message_len) :: message
  call compute_projection_matrix_square_grid(rank_loc,n_tasks_loc,nx2D(1),ny2D(1),&
  Rbegin2D,Rend2D,Zbegin2D,Zend2D,i_tor2D,n_tor2D,filter2D,hyper_filter2D,&
  parallel_filter2D,apply_dirichlet_bnd,test_node_list,test_element_list,&
  mumps_data,ifail_loc)
  do ii=1,size(mumps_data%irn),nstep_test
    write(message,'(A,I0,A)') 'Error projection matrix square irn ',&
    ii,' test: matrix-reference mismatch!'
    if(mumps_data%irn(ii)==1) call assert_equals(ref2D((mumps_data%jcn(ii)-1)/2+1)*TWOPI,&
    mumps_data%A(ii),tol2D,trim(message))
  enddo
  call close_dmumps(mumps_data)
end subroutine test_projection_matrix_square

!> Test the construction of the projection matrix with and without openmp
!> for a simple grid.
subroutine test_omp_projection_matrix_square
  use mod_projection_helpers_test_tools, only: close_dmumps
  !$use omp_lib
  implicit none
  type(DMUMPS_STRUC) :: mumps_data_serial,mumps_data_parallel
  integer :: ii,jj,n_threads
  real*8,dimension(:,:),allocatable :: A_parallel,A_serial
  !> construct matrix using n_omp_threads
  n_threads = 1
  !$omp parallel
  !$n_threads = omp_get_num_threads
  !$omp end parallel
  if(n_threads == 1) n_threads = n_omp_threads
  !> compute the non-threaded martrix
  !$ call omp_set_num_threads(1)
  call compute_projection_matrix_square_grid(rank_loc,n_tasks_loc,&
  nx2D(1),ny2D(1),Rbegin2D,Rend2D,Zbegin2D,Zend2D,i_tor2D,n_tor2D,filter2D,&
  hyper_filter2D,parallel_filter2D,apply_dirichlet_bnd,test_node_list,&
  test_element_list,mumps_data_serial,ifail_loc) 
  call construct_matrix_from_mumps(mumps_data_serial,A_serial)
  call close_dmumps(mumps_data_serial)
  !> compute the threaded matrix
  !$ call omp_set_num_threads(n_threads)
  call compute_projection_matrix_square_grid(rank_loc,n_tasks_loc,&
  nx2D(1),ny2D(1),Rbegin2D,Rend2D,Zbegin2D,Zend2D,i_tor2D,n_tor2D,filter2D,&
  hyper_filter2D,parallel_filter2D,apply_dirichlet_bnd,test_node_list,&
  test_element_list,mumps_data_parallel,ifail_loc) 
  call construct_matrix_from_mumps(mumps_data_parallel,A_parallel)
  call close_dmumps(mumps_data_parallel)
  !> checks
  call assert_equals(A_serial,A_parallel,size(A_serial,1),size(A_serial,2),&
  tol2D,'Error test omp projection matrix square: matrix mismatch!')
  if(allocated(A_serial)) deallocate(A_serial)
  if(allocated(A_parallel)) deallocate(A_parallel)
end subroutine test_omp_projection_matrix_square

!> Tools ------------------------------------------
!> Peaked gaussian at R=R_gaussian,Z=Z_gaussian with width a_gaussian.
!> On a grid from 0.5 to 1.5 in R, -0.5 to 0.5 in Z the integral is given by
!> \[
!> a^2 2\pi^2 \erf\left(\frac{0.5}{a}\right) \erf\left(\frac{0.5}{a}\right)
!> \]
!> for a_gaussian=5e-3, R_gaussian=1.d0 and Z_gaussian=0.d0 this yields
!>  2pi*0.0000785398 (where the volume is 2pi) !> the RMS value is 
!> the integral of (f_peak - <mean>)^2 over the volume
!> and is given by 0.0000397457
function f_peak(R,Z)
  real*8,intent(in) :: R,Z
  real*8 :: f_peak
  f_peak = exp(-((R-R_gaussian)**2+(Z-Z_gaussian)**2)/(a_gaussian**2))
end function f_peak

!> Project a function f onto grid in node_list and
!> element_list and test for mean and RMS value.
!> Optionally, write to file visual inspection.
!> inputs:
!>   node_list:    (type_node_list) JOREK node list
!>   element_list: (type_element_list) JOREK element list
!>   f:            (function,real8) function to project
!>   mean:         (real8) expected mean value
!>   rms:          (real8) expected root mean squared
!>   pfilename:    (character) particle file name
!>   message_root: (character) root of the error message to be printed
!>   mean_tol_in   (real8)(optional) tolerance for the mean check
!>   rms_tol_in:   (real8)(optional) tolerance for the rms check
!>   filter_in:    (real8)(optional) smoothing filter value to be used projection
!>   hyper_filter: (real8)(optional) smoothing hyper filter value to be use in projection
!> outputs:
!>   node_list:    (type_node_list) JOREK node list
!>   element_list: (type_element_list) JOREK element list
subroutine project_f_with_assert_and_write(node_list,element_list,f,mean,&
RMS,pfilename,message_root,apply_dirichlet,mean_tol_in,rms_tol_in,&
filter_in,hyper_filter_in)
  use data_structure,                    only: type_node_list
  use data_structure,                    only: type_element_list
  use mod_project_particles,             only: write_particle_distribution_to_h5
  use mod_projection_helpers_test_tools, only: project_f,elements_mean_rms
  implicit none
  type(type_node_list),intent(inout)    :: node_list
  type(type_element_list),intent(inout) :: element_list
  real*8,external                       :: f
  real*8,intent(in)                     :: mean,rms
  character(len=*),intent(in)           :: pfilename,message_root
  real*8,intent(in),optional            :: mean_tol_in,rms_tol_in
  real*8,intent(in),optional            :: filter_in,hyper_filter_in
  logical,intent(in)                    :: apply_dirichlet
  real*8 :: mean_test,rms_test,rms_tol,mean_tol,filter,hyper_filter
  character(len=2*message_len) :: message
  !> initialisation
  filter=0.d0;       if(present(filter_in))       filter=filter_in;
  hyper_filter=0.d0; if(present(hyper_filter_in)) hyper_filter=hyper_filter_in;
  mean_tol=1.d-12;   if(present(mean_tol_in))     mean_tol=mean_tol_in;
  rms_tol=1.d-12;    if(present(rms_tol_in))      rms_tol=rms_tol_in;
  !> project the function and compute mean and rms
  call project_f(rank_loc,master_rank,node_list,element_list,f,ifail_loc,filter,&
  hyper_filter,apply_dirichlet_bnd_in=apply_dirichlet)
  call elements_mean_rms(node_list,element_list,f,mean_test,rms_test)
  !> checks
  write(message,'(A,A,I0,A)') trim(message_root),': rank ',rank_loc,' unexpected mean value!'
  call assert_equals(mean,mean_test,mean_tol,trim(message))
  write(message,'(A,A,I0,A)') trim(message_root),': rank ,',rank_loc,' unexpected RMS value!'
  call assert_equals(RMS,rms_test,rms_tol,trim(message))
  !> write projection in file if requried
  if(write_proj_output) then
    call write_particle_distribution_to_h5(node_list,element_list,filename=pfilename//'.h5',&
    n_fields=n_fields_1,time=time_sol)
  endif 
end subroutine project_f_with_assert_and_write

!> Compute the projection matrix for s simple square grid 
subroutine compute_projection_matrix_square_grid(rank,n_tasks,nx_loc,ny_loc,&
Rbegin_loc,Rend_loc,Zbegin_loc,Zend_loc,i_tor_local,n_tor_local,local_filter,&
hyper_filter,parallel_filter,apply_dirichlet,node_list,element_list,mumps_data,ifail)
  use mpi_mod
  use data_structure
  use mod_project_particles,             only: prepare_mumps_par_n0
  use mod_project_particles,             only: prepare_mumps_par
  use mod_projection_helpers_test_tools, only: default_square_grid
  use mod_projection_helpers_test_tools, only: broadcast_dmumps_struct_A_irn_jcn
  use mod_projection_helpers_test_tools, only: map_matrix_to_MUMPS_datastructure
  implicit none
  type(type_node_list),intent(inout)    :: node_list
  type(type_element_list),intent(inout) :: element_list
  integer,intent(inout)                 :: ifail
  integer,intent(in)   :: rank,n_tasks
  integer,intent(in)   :: nx_loc,ny_loc,i_tor_local,n_tor_local
  real*8,intent(in)    :: Rbegin_loc,Rend_loc,Zbegin_loc,Zend_loc
  real*8,intent(in)    :: local_filter,hyper_filter,parallel_filter
  logical,intent(in)   :: apply_dirichlet
  type(DMUMPS_STRUC),intent(inout) :: mumps_data
  type (type_SP_MATRIX) :: a_mat
  integer :: mpi_comm_n,mpi_comm_master
  real*8  :: area,volume
  real*8,dimension(:),allocatable :: integral_weights
  !> split communicators for mumps
  call MPI_Comm_dup(MPI_COMM_WORLD,mpi_comm_n,ifail)
  call MPI_Comm_dup(MPI_COMM_WORLD,mpi_comm_master,ifail)
  !> construct the matrix
  call default_square_grid(rank,n_tasks,nx_loc,ny_loc,node_list,element_list,ifail)
  !> compute the matrix 
  if(i_tor_local.eq.1) then
    call prepare_mumps_par_n0(node_list,element_list,n_tor_local,i_tor_local,&
    MPI_COMM_WORLD,mpi_comm_n,mpi_comm_master,a_mat,area,volume,&
    filter=local_filter,filter_hyper=hyper_filter,filter_parallel=parallel_filter,&
    apply_dirichlet_condition_in=apply_dirichlet,integral_weights=integral_weights)
  else
    call prepare_mumps_par(node_list,element_list,n_tor_local,i_tor_local,&
    MPI_COMM_WORLD,mpi_comm_n,mpi_comm_master,a_mat,filter=local_filter,&
    filter_hyper=hyper_filter,filter_parallel=parallel_filter,&
    apply_dirichlet_condition_in=apply_dirichlet)
  endif
  !> map the matrix to the MUMPS datastructure
  call map_matrix_to_MUMPS_datastructure(a_mat,mumps_data)
  !> broadcast matrix
  call broadcast_dmumps_struct_A_irn_jcn(rank,master_rank,mumps_data,ifail)
end subroutine compute_projection_matrix_square_grid

!> construct the projected matrix from mumps data
subroutine construct_matrix_from_mumps(mumps_data,matrix)
  implicit none
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

!> ------------------------------------------------
end module mod_projection_spec_mpi_test
