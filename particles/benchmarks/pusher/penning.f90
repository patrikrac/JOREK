!> This program tests the reproduction of the trajectory of in a penning trap
!!
!! The electric and magnetic field are expressed in terms of
!! the local coordinates s and t on the JOREK mesh.
!! A single particle is released at position x0 with velocity v0
!! and tracked until t_end. The output code signifies whether the
!! tests passed the preset accuracy.
!!
!! The code reads a namelist input file to determine the grid used,
!! the number of grid cells to use and the time step sizes to use.
program penning

use data_structure
use phys_module
use basis_at_gaussian
use particle_tracer
use mod_coordinate_transforms ! For solution of penning trap trajectory
use mod_parameters
use constants
use mod_fields_linear
use mod_export_restart
use mod_neighbours
use mod_find_rz_nearby
use mod_penning_case
use mod_penning_case_jorek
use mod_particle_sim

implicit none

! Penning trap parameters (in SI units)
! The particle remains between +- 3 in Z and 8 and 13 in R for these parameters. set this in an input file

! Local variables
logical :: verbose = .false.
integer :: i,ielm_out,ifail,j,i_elm_old
real*8  :: R,Z
real*8  :: s,t
real*8  :: x_a(3), x_e(3), err_norm, err_ref
real*8  :: rz_old(2), st_old(2)

type(particle_kinetic_leapfrog) :: particle

! For the initial half-step
real*8  :: E(3), B(3), psi, U
real*8  :: t_norm, qom, B0, Phi0


write(*,*) '***************************************'
write(*,*) '* JOREK2 : Penning trap test          *'
write(*,*) '***************************************'

call sim%initialize(num_groups=0)
write(*,*) tstep_n

t_norm  = sqrt(mu_zero * ATOMIC_MASS_UNIT * central_mass * central_density * 1.d20) ! 1 jorek time unit in seconds
qom     = real(charge) * el_chg / (mass * atomic_mass_unit)
B0      = omega_b/qom ! In T
Phi0    = epsilon*omega_e**2/qom/2.d0*t_norm ! In JOREK units: E_SI*t_norm

allocate(jorek_fields_interp_linear::sim%fields)
select type (p => sim%fields)
type is (jorek_fields_interp_linear)
  p%static = .true.
end select
allocate(sim%fields%node_list, sim%fields%element_list)
call init_node_list(sim%fields%node_list, n_nodes_max, sim%fields%node_list%n_dof, n_var)

! Only 1 toroidal mode (n=0)
mode(1) = 0
if (n_tor /= 1) then
  write(0,*) 'ERROR: Penning test only works with n_tor=1'
  stop
endif

! Create a grid according to the variables present in the input file
if ((n_R > 0) .and. (n_Z > 0) .and. (n_radial > 0)) then
  call grid_bezier_square_polar(n_R, n_Z, n_radial, R_begin, R_end, Z_begin, Z_end, R_geo,   &
    Z_geo, amin, fbnd, fpsi, mf, .true., sim%fields%node_list, sim%fields%element_list)
else if ((n_R > 0) .and. (n_Z > 0) ) then
  call grid_bezier_square(n_R, n_Z, R_begin, R_end, Z_begin, Z_end, .true., sim%fields%node_list,       &
    sim%fields%element_list)
else if ((n_radial > 0) .and. (n_pol > 0) ) then
  call grid_polar_bezier(R_geo, Z_geo, amin, 0.d0, 0.d0, fbnd, fpsi, mf, n_radial, n_pol,    &
    sim%fields%node_list, sim%fields%element_list)
else
  write(*,*) 'FATAL : no valid combination of grid-sizes specified'
  stop
end if
call update_neighbours(sim%fields%node_list, sim%fields%element_list)

call jorek_penning_fields(sim%fields%node_list, sim%fields%element_list, apply_dirichlet_in=.false.)


! Write a restart file containing the grid
write(*,*) "INFO: Exporting grid to jorek_restart.h5"
rst_hdf5 = 1
call export_restart(sim%fields%node_list,sim%fields%element_list,'jorek_restart')

do i=1,size(tstep_n)
  if (tstep_n(i) .le. 0 .or. tstep_n(i) .ne. tstep_n(i) .or. tstep_n(i) .eq. 1) cycle
  nstep_n(i) = floor(time_end  / (t_norm*tstep_n(i)) ) ! Override nstep_n
  call reset_particle
  ! Calculate the correct velocity for a half-step backwards to obtain second-order convergence
  E = [-2.d0*Phi0*x0(1),0.d0,0.d0]/t_norm
  B = [0.d0, B0, 0.d0]
  call boris_initial_half_step_backwards_RZPhi(particle, real(mass,8), E, B, tstep_n(i)*t_norm)
  do j=1,nstep_n(i)
    if (mod(j,max(nstep_n(i)/100,10)) .eq. 0) write(*,'(A)',advance='no') '.'
    call sim%fields%calc_EBpsiU(0.d0, particle%i_elm, particle%st, particle%x(3), E, B, psi, U)
    rz_old    = particle%x(1:2)
    st_old    = particle%st
    i_elm_old = particle%i_elm
    if (verbose) write(*,"(A,g16.8,g16.8,g16.8)") 'PARTICLE ', cylindrical_to_cartesian(particle%x)
    call boris_push_cylindrical(particle, real(mass,8), E, B, tstep_n(i)*t_norm)
    call find_RZ_nearby(sim%fields%node_list, sim%fields%element_list, rz_old(1), rz_old(2), st_old(1), st_old(2), i_elm_old, &
        particle%x(1), particle%x(2), particle%st(1), particle%st(2), particle%i_elm, ifail)
    if (particle%i_elm .le. 0) then
      write(*,*) 'Particle lost! exiting.'
      stop 1
    end if
  end do
  write(*,*) '' ! finish the newline

  ! Check position against analytical result
  x_e = cylindrical_to_cartesian(particle%x)
  x_a = penning_trajectory(tstep_n(i)*real(nstep_n(i))*t_norm)
  err_norm = norm2(x_e - x_a)

  ! Exit the test if the error is too large
  ! Norm error scales as dt^2
  err_ref = 3.14084d-8*tstep_n(i)**2
  !if (n_pol .gt. 0) err_ref = max(err_ref, 2.d0*3.675d3*real(n_pol)**(-4)) ! only use this condition for polar grids, with huge margin (2x)
  write(*,"(A,i3,A,g12.4,A,g16.8,a,g16.8)") "RESULT: n_radial= ", n_radial, " dt= ", tstep_n(i), " error= ", err_norm, " reference= ", err_ref
  if (isnan(err_norm) .or. err_norm .gt. err_ref*1.2d0) then
    write(*,*) "Penning test failed"
    stop 1
  endif
enddo
write(*,*) "Tests successfull at all timestep sizes"

contains
  !> Reset a particle to the initial conditions
  subroutine reset_particle() ! Uses the global variables
    implicit none
    call find_RZ(sim%fields%node_list,sim%fields%element_list,x0(1),x0(2),R,Z,ielm_out,s,t,ifail)
    if (ifail .ne. 0) then
      write(*,*) "CRITICAL: could not find initial particle in grid", &
          "Particle location: ", x0(1), x0(2), " R,z_out= ", R, Z
      stop 1
    endif
    particle%st = [s,t]
    particle%i_elm = ielm_out
    particle%x = [R,Z,x0(3)]
    particle%v = vector_rotation(cartesian_to_cylindrical(v0), particle%x(3))
    particle%q = charge
    particle%weight = 1.d0
  end subroutine reset_particle
end program penning
