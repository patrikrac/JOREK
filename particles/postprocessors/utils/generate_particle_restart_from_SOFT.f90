!> generate_particle_restart_from_SOFT generates a 
!> JOREK particle restart file from a orbits file
!> produced by the SOFT code
program generate_particle_restart_from_SOFT
use constants,       only: PI,SPEED_OF_LIGHT,ATOMIC_MASS_UNIT,EL_CHG
use mod_mpi_tools,   only: init_mpi_threads,finalize_mpi_threads
use mod_particle_io, only: write_simulation_hdf5
use phys_module
use particle_tracer

implicit none

!> Data-types ------------------------------------------------------------------------------
!> derived datatype describing the particle pdf used in SODT
type type_soft_pdf
  real*8,dimension(:),allocatable   :: xi,p !< cos(pitch_angle) and momentum sizes
  real*8,dimension(:,:),allocatable :: pdf  !< particle pdf
end type type_soft_pdf

!> Variables -------------------------------------------------------------------------------
type(pcg32_rng)      :: rng_type
type(event)          :: field_reader
type(type_soft_pdf),dimension(:),allocatable :: soft_pdf_list
logical              :: do_write_particles_in_hdf5,compute_magnetic_field_error
integer              :: my_id,n_mpis,ierr,n_vec,n_groups,n_phi
integer              :: n_r_pdf_mesh,n_accepted_orbit_labels
integer,dimension(2) :: dims
integer,dimension(:),allocatable :: accepted_orbit
real*8               :: time,mass,charge
real*8,dimension(2)  :: phi_interval,soft_RZ_axis
real*8,dimension(:),allocatable   :: soft_orbit_ppar_local,soft_orbit_pperp_local
real*8,dimension(:),allocatable   :: soft_orbit_weights_local,soft_orbit_ppar
real*8,dimension(:),allocatable   :: soft_orbit_pperp,soft_orbit_weights,soft_pdf_r_mesh
real*8,dimension(:),allocatable   :: soft_R_mesh,soft_Z_mesh
real*8,dimension(:,:),allocatable :: soft_orbit_x,soft_orbit_x_local,soft_poloidal_flux
character(len=20)    :: generator_type
character(len=17)    :: fields_filename,soft_pdf_filename
character(len=28)    :: soft_magfield_filename
character(len=125)   :: particle_filename
character(len=250)   :: soft_orbit_filename
character(len=25)    :: filename_jorek_hdf5
character(len=40)    :: Bfield_error_filename
!> Variable definitions --------------------------------------------------------------------
do_write_particles_in_hdf5   = .false.        !< writeh particle in hdf5 if true 
compute_magnetic_field_error = .false.        !< if true the error of the SOFT-JOREK B-field is computed
n_accepted_orbit_labels = 2                   !< number of labels of accepted orbits
allocate(accepted_orbit(n_accepted_orbit_labels)) 
n_groups                = 1                   !< number of jorek particle groups
n_vec                   = 3                   !< component of a vector
n_phi                   = 10                   !< number of toroidal positions to be sampled for each particle
accepted_orbit          = [3,4]               !< label of SOFT orbit to be used (discard all others)
time                    = 0d0                 !< simulation time
mass                    = 5.48579909065d-4    !< electron mass in AMU 
charge                  = -1d0                !< electron charge
phi_interval            = [0d0,PI]            !< toroidal angle interval in which particles are sampled
generator_type          = 'deterministic'     !< select the type of particle generator
fields_filename         = 'jorek_equilibrium' !< jorek restart filename
soft_magfield_filename  = 'magnetic_field_jorek_to_soft' !< soft magnetic field file
particle_filename       = 'part_restart_soft_orbits_deterministic.h5'  !< particle restart filename 
soft_pdf_filename       = 'pdf_jorek_to_soft'            !< soft distribution field input from jorek
soft_orbit_filename     = 'orbit_test_jorek_JET_pulse95135_t48dot54_parabolic_qprofile_q95_6dot8_press0_res1r5dot88en1m5_res2r4dot705en1m4_Ip612en1MA_Ekin20MeV_np10_theta1_2dot85_itheta2_pi_nitheta100_norbits100_a96_wall_angdist_trapz_noDrift_image_distWithAxis' !< soft orbit filename
filename_jorek_hdf5    = 'jorek_particles_from_soft'
Bfield_error_filename  = 'soft_jorek_magnetic_field_error' 
!> Initialisation --------------------------------------------------------------------------
!> initialise the MPI communicator
call init_mpi_threads(my_id,n_mpis,ierr)
!> read mhd data
write(*,*) "Reading MHD data ..."
n_part_groups = n_groups
call sim%initialize(.true.,my_id,n_mpis)
field_reader = event(read_jorek_fields_interp_linear(basename=trim(fields_filename),i=-1))
call with(sim,field_reader)
write(*,*) "Reading MHD data: completed!"
!> Read soft input and generate JOREK particle restart -------------------------------------
write(*,*) "Reading SOFT magnetic field, pdf and orbit files ..."
!> compute and broadcast the minor radii array
if(my_id.eq.0) then
  if(compute_magnetic_field_error) then
    !> compute and dump in hdf5 file the soft-jorek magnetic field error
    call compute_error_soft_jorek_particles_magnetic_field(sim%fields,n_vec,time,&
    accepted_orbit,soft_orbit_filename,Bfield_error_filename)
  endif
  !> compute and broadcast the minor radii array
  call read_soft_maxis_and_poloidal_flux(soft_magfield_filename,soft_RZ_axis,soft_R_mesh,&
  soft_Z_mesh,soft_poloidal_flux)
  !> read the soft_pdf_file
  call read_soft_pdf_file(soft_pdf_filename,n_r_pdf_mesh,soft_pdf_r_mesh,soft_pdf_list)
  call read_and_compute_soft_orbit_data(soft_orbit_filename,n_vec,dims(1),accepted_orbit,\
  soft_RZ_axis,soft_R_mesh,soft_Z_mesh,soft_poloidal_flux,soft_pdf_r_mesh,soft_pdf_list,\
  soft_orbit_x,soft_orbit_ppar,soft_orbit_pperp,soft_orbit_weights)
  !> scatter the global arrays to each mpi process)
  dims(2) = dims(1)/n_mpis;
  if(dims(2)*n_mpis.lt.dims(1)) dims(2) = dims(2)+1
  if(allocated(soft_R_mesh))        deallocate(soft_R_mesh)
  if(allocated(soft_Z_mesh))        deallocate(soft_Z_mesh)
  if(allocated(soft_pdf_r_mesh))    deallocate(soft_pdf_r_mesh)
  if(allocated(soft_poloidal_flux)) deallocate(soft_poloidal_flux)
  if(allocated(soft_pdf_list))      deallocate(soft_pdf_list)
endif
call MPI_Bcast(dims,size(dims),MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
allocate(soft_orbit_x_local(n_vec,dims(2)));  soft_orbit_x_local     = 0d0;
allocate(soft_orbit_ppar_local(dims(2)));     soft_orbit_ppar_local  = 0d0;
allocate(soft_orbit_pperp_local(dims(2)));    soft_orbit_pperp_local = 0d0;
allocate(soft_orbit_weights_local(dims(2)));  soft_orbit_weights_local = 0d0;
call scatter_2D_arrays(my_id,n_mpis,n_vec,dims(2),dims(1),soft_orbit_x,soft_orbit_x_local)
call MPI_Scatter(soft_orbit_ppar,dims(2),MPI_REAL8,soft_orbit_ppar_local,dims(2),MPI_REAL8,0,MPI_COMM_WORLD,ierr)
call MPI_Scatter(soft_orbit_pperp,dims(2),MPI_REAL8,soft_orbit_pperp_local,dims(2),MPI_REAL8,0,MPI_COMM_WORLD,ierr)
call MPI_Scatter(soft_orbit_weights,dims(2),MPI_REAL8,soft_orbit_weights_local,dims(2),&
MPI_REAL8,0,MPI_COMM_WORLD,ierr)
if(allocated(soft_orbit_x))       deallocate(soft_orbit_x)
if(allocated(soft_orbit_ppar))    deallocate(soft_orbit_ppar)
if(allocated(soft_orbit_pperp))   deallocate(soft_orbit_pperp)
if(allocated(soft_orbit_weights)) deallocate(soft_orbit_weights)
write(*,*) "Reading SOFT magnetic field, pdf and orbit files: completed!"
!> Generate JOREK relativistic gc from SOFT orbits -----------------------------------------
write(*,*) 'Converting soft orbit to JOREK relativistic gc ...'
if(trim(generator_type)=='random') then
  write(*,*) "use random the toroidal angle"
  call convert_soft_orbits_in_jorek_relativistic_gcs_rnd_phi(sim,rng_type,my_id,n_mpis,\
  time,mass,charge,n_vec,dims(2),n_phi,phi_interval,soft_orbit_x_local,soft_orbit_ppar_local,\
  soft_orbit_pperp_local,soft_orbit_weights_local)
elseif(trim(generator_type)=='deterministic') then
  write(*,*) "use deterministic the toroidal angle"
  call convert_soft_orbits_in_jorek_relativistic_gcs_det_phi(sim,my_id,n_mpis,time,\
  mass,charge,n_vec,dims(2),n_phi,phi_interval,soft_orbit_x_local,soft_orbit_ppar_local,\
  soft_orbit_pperp_local,soft_orbit_weights_local)
else
  write(*,*) "use SOFT toroidal angle"
  call convert_soft_orbits_in_jorek_relativistic_gcs(sim,my_id,n_mpis,time,mass,charge,n_vec,dims(2),\
  soft_orbit_x_local,soft_orbit_ppar_local,soft_orbit_pperp_local,soft_orbit_weights_local)
endif
write(*,*) 'Converting soft orbit to JOREK relativistic gc: completed!'
write(*,*) 'Writing JOREK relativistic gc in ',trim(particle_filename),' ...'
call write_simulation_hdf5(sim,trim(particle_filename))
write(*,*) 'Writing JOREK relativistic gc in ',trim(particle_filename),' completed!'
!> Write data in HDF5 file -----------------------------------------------------------------
if(do_write_particles_in_hdf5) then
  write(*,*) 'Writing JOREK particles in HDF5 file ...'
  call write_particles_in_hdf5(my_id,filename_jorek_hdf5,n_mpis,n_vec,sim)
  write(*,*) 'Writing JOREK particles in HDF5 file: completed!'
endif
!> Finalisation ----------------------------------------------------------------------------
if(allocated(accepted_orbit))           deallocate(accepted_orbit)
if(allocated(soft_orbit_x))             deallocate(soft_orbit_x)
if(allocated(soft_orbit_ppar))          deallocate(soft_orbit_ppar)
if(allocated(soft_orbit_pperp))         deallocate(soft_orbit_pperp)
if(allocated(soft_orbit_x_local))       deallocate(soft_orbit_x_local)
if(allocated(soft_orbit_ppar_local))    deallocate(soft_orbit_ppar_local)
if(allocated(soft_orbit_pperp_local))   deallocate(soft_orbit_pperp_local)
if(allocated(soft_orbit_weights_local)) deallocate(soft_orbit_weights_local)
if(allocated(soft_pdf_r_mesh))          deallocate(soft_pdf_r_mesh)
if(allocated(soft_pdf_list))            deallocate(soft_pdf_list)
if(allocated(soft_R_mesh))              deallocate(soft_R_mesh)
if(allocated(soft_Z_mesh))              deallocate(soft_Z_mesh)
if(allocated(soft_poloidal_flux))       deallocate(soft_poloidal_flux)
call finalize_mpi_threads(ierr)
write(*,*) 'Program terminated: good bye!'

contains

!> generate jorek particle relativistic gc from soft orbits
!> inputs:
!>   sim:                (particle_sim) particle simulation type to be initialised
!>   my_id:              (integer) MPI task rank
!>   n_mpis:             (integer) number of MPI tasks
!>   time:               (real8) simulation time
!>   mass:               (real8) particle mass in AMU
!>   charge:             (real8) particle charge (RE: -1)
!>   n_vec:              (integer) size of the position vector
!>   n_points:           (integer) number of soft valid points
!>   soft_orbit_x:       (real8)(n_vec,n_points) soft positions in xyz
!>   soft_orbit_ppar:    (real8)(n_points) soft parallel momentum
!>   soft_orbit_pperp:   (real8)(n_points) soft perpendicular momentum
!>   soft_orbit_weights: (real8)(n_points) soft orbit weight:
!>                       jacobian*dpoloidal*dminor_radius*dmomentum*dcospitchangle
!> outpus:
!>   sim: (particle_sim) initialised particle simulation
!>   soft_orbit_weights: (real8)(n_points) soft orbit weight:
!>                       jacobian*dpoloidal*dminor_radius*dmomentum*dcospitchangle*dtorangle
subroutine convert_soft_orbits_in_jorek_relativistic_gcs(&
sim,my_id,n_mpis,time,mass,charge,n_vec,n_points,soft_orbit_x,&
soft_orbit_ppar,soft_orbit_pperp,soft_orbit_weights)
  use mpi
  use constants,                 only: TWOPI,SPEED_OF_LIGHT
  use mod_coordinate_transforms, only: cartesian_to_cylindrical
  implicit none
  !> inputs-outputs:
  type(particle_sim),intent(inout)         :: sim
  real*8,dimension(n_points),intent(inout) :: soft_orbit_weights
  !> inputs:
  integer,intent(in)               :: my_id,n_mpis,n_vec,n_points
  real*8,intent(in)                :: time,mass,charge
  real*8,dimension(n_points),intent(in)       :: soft_orbit_ppar
  real*8,dimension(n_points),intent(in)       :: soft_orbit_pperp
  real*8,dimension(n_vec,n_points),intent(in) :: soft_orbit_x

  !> variables:
  integer             :: ii,i_elm,ifail
  real*8              :: U,psi
  real*8,dimension(2) :: st,Rbox,Zbox
  real*8,dimension(3) :: RZphi,B_field,E_field
  !> initialise and allocate particle simulation array
  call domain_bounding_box(sim%fields%node_list,sim%fields%element_list,&
  Rbox(1),Rbox(2),Zbox(1),Zbox(2))
  sim%time = time; sim%groups(1)%mass = mass; 
  allocate(particle_gc_relativistic::sim%groups(1)%particles(n_points))
  !> the differential of the toroidal angle for having the total number of particles
  soft_orbit_weights = soft_orbit_weights*TWOPI
  !> loop on the soft orbits
  !$omp parallel do default(shared) firstprivate(n_points,n_vec,time,charge,mass,&
  !$omp Rbox,Zbox) private(ii,RZphi,i_elm,st,B_field,E_field,U,psi,ifail)
  do ii=1,n_points
    !> transform the soft orbit coordinates in jorek global/local coordinates
    RZphi = cartesian_to_cylindrical(soft_orbit_x(:,ii)); i_elm = -1;
    if(((RZphi(1).ge.Rbox(1)).and.(RZphi(1).le.Rbox(2))).and.((RZphi(2).ge.Zbox(1)).and.(RZphi(2).le.Zbox(2)))) &
    call find_RZ(sim%fields%node_list,sim%fields%element_list,RZphi(1),RZphi(2),&
    RZphi(1),RZphi(2),i_elm,st(1),st(2),ifail)
    select type (p=>sim%groups(1)%particles(ii))
    type is (particle_gc_relativistic)
      if(i_elm.le.0) then
        p%x = 0d0; p%st = 0d0; p%weight = 0d0; p%i_elm = 0; p%i_life = 0; p%t_birth = 0.;
        p%p = 0d0; p%q = int(0,kind=1);
      else
        p%x = RZphi; p%st = st; p%weight = soft_orbit_weights(ii);
        p%i_elm = i_elm; p%i_life = 0; p%t_birth = 0.;
        call sim%fields%calc_EBpsiU(time,i_elm,st,p%x(3),E_field,B_field,psi,U)
        p%p = SPEED_OF_LIGHT*mass*[soft_orbit_ppar(ii),(5d-1*SPEED_OF_LIGHT*&
        (soft_orbit_pperp(ii)**2))/(norm2(B_field))]; p%q = int(charge,kind=1);
      endif
    end select
  enddo
  !$omp end parallel do
end subroutine convert_soft_orbits_in_jorek_relativistic_gcs

!> generate jorek particle relativistic gc from soft orbits randomising the toroidal
!> angle within a given toroidal angle interval 
!> inputs:
!>   sim:                (particle_sim) particle simulation type to be initialised
!>   rng_base:           (type_rng) type of the random number generator
!>   my_id:              (integer) MPI task rank
!>   n_mpis:             (integer) number of MPI tasks
!>   time:               (real8) simulation time
!>   mass:               (real8) particle mass in AMU
!>   charge:             (real8) particle charge (RE: -1)
!>   n_vec:              (integer) size of the position vector
!>   n_points:           (integer) number of soft valid points
!>   n_phi:              (integer) number of toroidal samples per particles
!>   phi_interval:       (real8)(2) toroidal angle interval for sampling
!>   soft_orbit_x:       (real8)(n_vec,n_points) soft positions in xyz
!>   soft_orbit_ppar:    (real8)(n_points) soft parallel momentum
!>   soft_orbit_pperp:   (real8)(n_points) soft perpendicular momentum
!>   soft_orbit_weights: (real8)(n_points) soft orbit weight:
!>                       jacobian*dpoloidal*dminor_radius*dmomentum*dcospitchangle
!> outpus:
!>   sim: (particle_sim) initialised particle simulation
!>   soft_orbit_weights: (real8)(n_points) soft orbit weight:
!>                       jacobian*dpoloidal*dminor_radius*dmomentum*dcospitchangle*dtorangle
subroutine convert_soft_orbits_in_jorek_relativistic_gcs_rnd_phi(sim,rng_base,&
my_id,n_mpis,time,mass,charge,n_vec,n_points,n_phi,phi_interval,&
soft_orbit_x,soft_orbit_ppar,soft_orbit_pperp,soft_orbit_weights)
  use mpi
  use constants,                 only: SPEED_OF_LIGHT
  use mod_coordinate_transforms, only: cartesian_to_cylindrical
  use mod_random_seed,           only: random_seed
  use mod_rng
  !$ use omp_lib
  implicit none
  !> inputs-outputs:
  type(particle_sim),intent(inout)         :: sim
  real*8,dimension(n_points),intent(inout) :: soft_orbit_weights
  !> inputs:
  class(type_rng),intent(in)       :: rng_base
  integer,intent(in)               :: my_id,n_mpis,n_vec
  integer,intent(in)               :: n_points,n_phi
  real*8,intent(in)                :: time,mass,charge
  real*8,dimension(2),intent(in)   :: phi_interval
  real*8,dimension(n_points),intent(in)       :: soft_orbit_ppar
  real*8,dimension(n_points),intent(in)       :: soft_orbit_pperp
  real*8,dimension(n_vec,n_points),intent(in) :: soft_orbit_x

  !> variables:
  class(type_rng),dimension(:),allocatable :: rngs
  integer :: ii,jj,i_elm,ifail,n_threads,thread_id,n_particles
  real*8                  :: U,psi
  real*8,dimension(2)     :: st,Rbox,Zbox
  real*8,dimension(3)     :: RZphi,B_field,E_field
  real*8,dimension(n_phi) :: phi_array
  !> initialise the random number generator
  n_threads = 1
!$ n_threads = omp_get_max_threads()
  allocate(rngs(n_threads),source=rng_base) 
  do ii=1,n_threads
    call rngs(ii)%initialize(n_phi,random_seed(),n_mpis*n_threads,my_id*n_threads+ii,ifail)
    if(ifail.ne.0) call MPI_Abort(MPI_COMM_WORLD,-1,ifail)
  enddo
  !> initialise and allocate particle simulation array
  call domain_bounding_box(sim%fields%node_list,sim%fields%element_list,&
  Rbox(1),Rbox(2),Zbox(1),Zbox(2))
  sim%time = time; sim%groups(1)%mass = mass; n_particles = n_points*n_phi;
  write(*,*) "Total number of particles: ",n_particles
  allocate(particle_gc_relativistic::sim%groups(1)%particles(n_particles))
  !> the differential of the toroidal angle for having the total number of particles
  soft_orbit_weights = soft_orbit_weights*((phi_interval(2)-phi_interval(1))/real(n_phi,kind=8))
  !> loop on the soft orbits
  !$omp parallel default(shared) firstprivate(n_points,n_vec,time,charge,n_phi,mass,&
  !$omp Rbox,Zbox,phi_interval) private(ii,jj,RZphi,i_elm,st,B_field,E_field,U,&
  !$omp psi,ifail,phi_array,thread_id)
  thread_id = 1
  !$ thread_id = omp_get_thread_num()+1
  !$omp do
  do ii=1,n_points
    !> transform the soft orbit coordinates in jorek global/local coordinates
    RZphi = cartesian_to_cylindrical(soft_orbit_x(:,ii)); i_elm = -1;
    if(((RZphi(1).ge.Rbox(1)).and.(RZphi(1).le.Rbox(2))).and.((RZphi(2).ge.Zbox(1)).and.(RZphi(2).le.Zbox(2)))) &
    call find_RZ(sim%fields%node_list,sim%fields%element_list,RZphi(1),RZphi(2),&
    RZphi(1),RZphi(2),i_elm,st(1),st(2),ifail)
    call rngs(thread_id)%next(phi_array); 
    phi_array = phi_interval(1)+(phi_interval(2)-phi_interval(1))*phi_array;
    do jj=1,n_phi
      select type (p=>sim%groups(1)%particles((ii-1)*n_phi+jj))
        type is (particle_gc_relativistic)
        if(i_elm.le.0) then
          p%x = 0d0; p%st = 0d0; p%weight = 0d0; p%i_elm = 0; p%i_life = 0; p%t_birth = 0.;
          p%p = 0d0; p%q = int(0,kind=1);
        else
          p%x = [RZphi(1),RZphi(2),phi_array(jj)]; p%st = st; p%weight = soft_orbit_weights(ii);
          p%i_elm = i_elm; p%i_life = 0; p%t_birth = 0.;
          call sim%fields%calc_EBpsiU(time,i_elm,st,p%x(3),E_field,B_field,psi,U)
          p%p = SPEED_OF_LIGHT*mass*[soft_orbit_ppar(ii),(5d-1*SPEED_OF_LIGHT*&
          (soft_orbit_pperp(ii)**2))/(norm2(B_field))]; p%q = int(charge,kind=1);
        endif
      end select
    enddo
  enddo
  !$omp end do
  !$omp end parallel
  !> clean-up
  if(allocated(rngs)) deallocate(rngs)
end subroutine convert_soft_orbits_in_jorek_relativistic_gcs_rnd_phi

!> generate jorek particle relativistic gc from soft orbits on deterministic
!> toroidal planes
!> inputs:
!>   sim:                (particle_sim) particle simulation type to be initialised
!>   my_id:              (integer) MPI task rank
!>   n_mpis:             (integer) number of MPI tasks
!>   time:               (real8) simulation time
!>   mass:               (real8) particle mass in AMU
!>   charge:             (real8) particle charge (RE: -1)
!>   n_vec:              (integer) size of the position vector
!>   n_points:           (integer) number of soft valid points
!>   n_phi:              (integer) number of toroidal samples per particles
!>   phi_interval:       (real8)(2) toroidal angle interval for sampling
!>   soft_orbit_x:       (real8)(n_vec,n_points) soft positions in xyz
!>   soft_orbit_ppar:    (real8)(n_points) soft parallel momentum
!>   soft_orbit_pperp:   (real8)(n_points) soft perpendicular momentum
!>   soft_orbit_weights: (real8)(n_points) soft orbit weight:
!>                       jacobian*dpoloidal*dminor_radius*dmomentum*dcospitchangle
!> outpus:
!>   sim: (particle_sim) initialised particle simulation
!>   soft_orbit_weights: (real8)(n_points) soft orbit weight:
!>                       jacobian*dpoloidal*dminor_radius*dmomentum*dcospitchangle*dtorangle
subroutine convert_soft_orbits_in_jorek_relativistic_gcs_det_phi(sim,&
my_id,n_mpis,time,mass,charge,n_vec,n_points,n_phi,phi_interval,&
soft_orbit_x,soft_orbit_ppar,soft_orbit_pperp,soft_orbit_weights)
  use mpi
  use constants,                 only: SPEED_OF_LIGHT
  use mod_coordinate_transforms, only: cartesian_to_cylindrical
  !$ use omp_lib
  implicit none
  !> inputs-outputs:
  type(particle_sim),intent(inout)         :: sim
  real*8,dimension(n_points),intent(inout) :: soft_orbit_weights
  !> inputs:
  integer,intent(in)               :: my_id,n_mpis,n_vec
  integer,intent(in)               :: n_points,n_phi
  real*8,intent(in)                :: time,mass,charge
  real*8,dimension(2),intent(in)   :: phi_interval
  real*8,dimension(n_points),intent(in)       :: soft_orbit_ppar
  real*8,dimension(n_points),intent(in)       :: soft_orbit_pperp
  real*8,dimension(n_vec,n_points),intent(in) :: soft_orbit_x

  !> variables:
  integer                 :: ii,jj,i_elm,ifail,n_particles
  real*8                  :: U,psi,delta_phi
  real*8,dimension(2)     :: st,Rbox,Zbox
  real*8,dimension(3)     :: RZphi,B_field,E_field
  real*8,dimension(n_phi) :: phi_array
  !> initialise and allocate particle simulation array
  delta_phi = (phi_interval(2)-phi_interval(1))/real(n_phi,kind=8)
  phi_array =(/ ((phi_interval(1)+(delta_phi*real(ii-1,kind=8))),ii=1,n_phi)/)
  call domain_bounding_box(sim%fields%node_list,sim%fields%element_list,&
  Rbox(1),Rbox(2),Zbox(1),Zbox(2))
  sim%time = time; sim%groups(1)%mass = mass; n_particles = n_points*n_phi;
  write(*,*) "Total number of particles: ",n_particles
  allocate(particle_gc_relativistic::sim%groups(1)%particles(n_particles))
  !> the differential of the toroidal angle for having the total number of particles
  soft_orbit_weights = soft_orbit_weights*delta_phi
  !> loop on the soft orbits
  !$omp parallel do default(shared) firstprivate(n_points,n_vec,time,charge,n_phi,mass,&
  !$omp Rbox,Zbox,phi_interval,phi_array) private(ii,jj,RZphi,i_elm,st,B_field,E_field,&
  !$omp U,psi,ifail)
  do ii=1,n_points
    !> transform the soft orbit coordinates in jorek global/local coordinates
    RZphi = cartesian_to_cylindrical(soft_orbit_x(:,ii)); i_elm = -1;
    if(((RZphi(1).ge.Rbox(1)).and.(RZphi(1).le.Rbox(2))).and.((RZphi(2).ge.Zbox(1)).and.(RZphi(2).le.Zbox(2)))) &
    call find_RZ(sim%fields%node_list,sim%fields%element_list,RZphi(1),RZphi(2),&
    RZphi(1),RZphi(2),i_elm,st(1),st(2),ifail)
    do jj=1,n_phi
      select type (p=>sim%groups(1)%particles((ii-1)*n_phi+jj))
        type is (particle_gc_relativistic)
        if(i_elm.le.0) then
          p%x = 0d0; p%st = 0d0; p%weight = 0d0; p%i_elm = 0; p%i_life = 0; p%t_birth = 0.;
          p%p = 0d0; p%q = int(0,kind=1);
        else
          p%x = [RZphi(1),RZphi(2),phi_array(jj)]; p%st = st; p%weight = soft_orbit_weights(ii);
          p%i_elm = i_elm; p%i_life = 0; p%t_birth = 0.;
          call sim%fields%calc_EBpsiU(time,i_elm,st,p%x(3),E_field,B_field,psi,U)
          p%p = SPEED_OF_LIGHT*mass*[soft_orbit_ppar(ii),(5d-1*SPEED_OF_LIGHT*&
          (soft_orbit_pperp(ii)**2))/(norm2(B_field))]; p%q = int(charge,kind=1);
        endif
      end select
    enddo
  enddo
  !$omp end parallel do
end subroutine convert_soft_orbits_in_jorek_relativistic_gcs_det_phi

!> scatter 2D array between MPI processes, the scattering is performed
!> along the second index
!> inputs:
!>   my_id:        (integer) MPI task rank
!>   n_mpis:       (integer) size of the MPI communicator
!>   n1:           (integer) first index size
!>   n2_loc:       (integer) local array second index size
!>   n2_glob:      (integer) global array second index size
!>   global_array: (n1,n2_glob) array to be scattered
!> outputs:
!>   local_array: (n1,n2_loc) array receiving the scattered global array
subroutine scatter_2D_arrays(my_id,n_mpis,n1,n2_loc,n2_glob,global_array,local_array)
  use mpi
  implicit none
  !> inputs:
  integer,intent(in) :: my_id,n_mpis,n1,n2_loc,n2_glob
  !> inputs-outputs:
  real*8,dimension(:,:),allocatable,intent(inout) :: global_array
  !> outputs:
  real*8,dimension(n1,n2_loc),intent(out) :: local_array
  !> variables:
  logical :: did_allocate
  integer :: ii,doublesize,subarraytype,resizedsubarraytype,errorcode,ierr
  integer(kind=MPI_Address_kind) :: startresized,extent
  integer,dimension(n_mpis)      :: disps,counts
  !> check consistency
  if(n2_loc.le.0) then
    write(*,*) 'Error: invalid size of the receiving array during scattering: abort!'
    call MPI_Abort(MPI_COMM_WORLD,errorcode,ierr)
  endif
  !> create a vector for each subblock and scatter them
  did_allocate = .false.
  if(my_id.eq.0) then
    counts = 1; disps = [(n2_loc*ii,ii=0,n_mpis-1)]; startresized = 0;
    call MPI_Type_size(MPI_REAL8,doublesize,ierr); extent = n1*doublesize;
    call MPI_Type_create_subarray(2,[n1,n2_glob],[n1,n2_loc],[0,0],MPI_ORDER_FORTRAN,MPI_REAL8,subarraytype,ierr)
    call MPI_Type_create_resized(subarraytype,startresized,extent,resizedsubarraytype,ierr)
    call MPI_Type_commit(resizedsubarraytype,ierr)
  endif
  local_array = 0d0; 
  if(my_id.ne.0) then
    if(.not.allocated(global_array)) allocate(global_array(1,1)); did_allocate = .true.;
  endif
  call MPI_scatterv(global_array,counts,disps,resizedsubarraytype,local_array,n1*n2_loc,MPI_REAL8,&
  0,MPI_COMM_WORLD,ierr)
  if(my_id.eq.0) call MPI_Type_free(resizedsubarraytype,ierr)
  if(did_allocate) deallocate(global_array)
end subroutine scatter_2D_arrays

!> read SOFT orbit file
!> inputs:
!>   soft_orbit_filename_in: (character)(*) name of the soft orbit file
!>   n_vec:                  (integer) size of the position vector
!>   accepted_label:         (integer) classification label of accepted orbit
!>   RZ_axis:                (real8)(2) position of the magnetic axis
!>   Rmesh:                  (real8)(nR) major radius mesh of the minor radii
!>   Zmesh:                  (real8)(nZ) vertical coordinate mesh of the minor radii
!>   poloidal_flux:          (real8)(nZ,nR) poloidal flux in the RZ grid
!>   r_minor_mesh:           (real8)(nr) pdf minor radius mesh
!>   pdf_list:               (type_soft_pdf)(nr) soft input pdfs
!> outputs:
!>   n_soft_points: (integer) total number of valid soft orbits
!>   x:             (real8)(3,n_soft_particles) soft positions in xyz coordinates
!>   ppar:          (real8)(n_soft_particles) soft parallel momentum
!>   pperp:         (real8)(n_soft_particles) soft perpendicular momentum
!>   weights:       (real8)(n_soft_particles) orbit weight: pdf(t=0)*Jdtdrho*dp*dxi
subroutine read_and_compute_soft_orbit_data(soft_orbit_filename_in,n_vec,n_soft_points,&
accepted_label,RZaxis,Rmesh,Zmesh,poloidal_flux,r_minor_mesh,pdf_list,x,ppar,pperp,weights)
  use constants, only: ATOMIC_MASS_UNIT,TWOPI
  use hdf5
  use hdf5_io_module, only: HDF5_open,HDF5_close
  use hdf5_io_module, only: HDF5_is_dataset_in_file
  use hdf5_io_module, only: HDF5_allocatable_array1D_reading_int
  use hdf5_io_module, only: HDF5_allocatable_array1D_reading
  use hdf5_io_module, only: HDF5_allocatable_array2D_reading
  implicit none
  !> inputs:
  type(type_soft_pdf),dimension(:),allocatable,intent(in) :: pdf_list
  character(len=*),intent(in)                  :: soft_orbit_filename_in
  integer,intent(in)                           :: n_vec
  integer,dimension(:),allocatable,intent(in)  :: accepted_label
  real*8,dimension(2),intent(in)               :: RZaxis
  real*8,dimension(:),allocatable,intent(in)   :: Rmesh,Zmesh,r_minor_mesh
  real*8,dimension(:,:),allocatable,intent(in) :: poloidal_flux
  !> outputs:
  integer,intent(out) :: n_soft_points
  real*8,dimension(:),allocatable,intent(out)   :: ppar,pperp,weights
  real*8,dimension(:,:),allocatable,intent(out) :: x
  !> variables:
  logical        :: weights_not_in_file
  integer(HID_T) :: file_id
  integer        :: ii,jj,id,n_orbits,n_times,n_active_orbits,ierr
  integer,dimension(:),allocatable   :: classification_loc,valid_orbit_id
  real*8                             :: orbit_dp,orbit_dxi
  real*8,dimension(:),allocatable    :: orbit_pdf_loc,orbit_p_mesh,orbit_xi_mesh
  real*8,dimension(:,:),allocatable  :: x_loc,ppar_loc,pperp_loc,weights_loc
  !> open the soft orbit hdf5 file
  call HDF5_open(trim(soft_orbit_filename_in)//".h5",file_id,ierr)
  !> read sofit particles
  call HDF5_allocatable_array1D_reading_int(file_id,classification_loc,'/classification') !< orbit classification
  call HDF5_allocatable_array1D_reading(file_id,orbit_p_mesh,'/param1')  !< momentum mesh
  call HDF5_allocatable_array1D_reading(file_id,orbit_xi_mesh,'/param2') !< cospitch mesh 
  call HDF5_allocatable_array2D_reading(file_id,ppar_loc,'/ppar')        !< parallel momentum
  call HDF5_allocatable_array2D_reading(file_id,pperp_loc,'/pperp')      !< perpendicular momentum
  call HDF5_allocatable_array2D_reading(file_id,weights_loc,"/Jdtdrho")  !< jacobian*dpoloidal*dminorradius
  call HDF5_allocatable_array2D_reading(file_id,x_loc,'/x')              !< position in xyz coordinates
  n_orbits = size(ppar_loc,2); n_times  = size(ppar_loc,1);
  allocate(orbit_pdf_loc(n_orbits)); orbit_pdf_loc = 0d0;
  !> check if the particle weights is in hdf5 file
  call HDF5_is_dataset_in_file(file_id,'f',weights_not_in_file)
  weights_not_in_file = .not.weights_not_in_file
  if(weights_not_in_file) then
    !> extract and compute the pdf for each orbit: be aware the pdf is taken at t=0
    !> and then replicated for nt times for each orbit
    write(*,*) "SOFT particle distribution not found: interpolate!"
    call compute_pdf_orbit(n_vec,n_times,n_orbits,RZaxis,x_loc,ppar_loc,pperp_loc,&
    Rmesh,Zmesh,poloidal_flux,r_minor_mesh,pdf_list,orbit_pdf_loc)
  else
    write(*,*) "SOFT particle distribution found!"
    call HDF5_allocatable_array1D_reading(file_id,orbit_pdf_loc,'/f')    !< weight
  endif
  !> close the soft orbit hdf5 file
  call HDF5_close(file_id)
  !> multiply the Jdtdrho by the pdf
  do ii=1,n_orbits
    weights_loc(:,ii) = weights_loc(:,ii)*orbit_pdf_loc(ii)
  enddo
  !> check if the momentum and cospitch angle mesh are uniform and extract their value
  call check_uniform_mesh_extract_length_1d(orbit_dp,orbit_p_mesh)
  call check_uniform_mesh_extract_length_1d(orbit_dxi,orbit_xi_mesh)
  !> remove zero orbits
  n_soft_points = n_orbits*n_times; n_active_orbits = 0;
  !> reshape arrays
  x_loc              = reshape(x_loc,[n_vec,n_soft_points])
  ppar_loc           = reshape(ppar_loc,[n_soft_points,1])
  pperp_loc          = reshape(pperp_loc,[n_soft_points,1])
  weights_loc        = reshape(weights_loc,[n_soft_points,1])
  weights_loc        = TWOPI*orbit_dp*orbit_dxi*(ppar_loc**2+pperp_loc**2)*weights_loc
  !> find id of active orbits
  allocate(valid_orbit_id(n_soft_points)); valid_orbit_id  = 0;
  do ii=1,n_orbits
    if(.not.(any(classification_loc(ii).eq.accepted_label))) cycle
    do jj=1,n_times
      id = (ii-1)*n_times+jj
      if((x_loc(1,id).eq.0d0).and.(x_loc(2,id).eq.0d0).and.(x_loc(3,id).eq.0d0)) cycle
      if((ppar_loc(id,1).eq.0d0).and.(pperp_loc(id,1).eq.0d0)) cycle
      n_active_orbits = n_active_orbits + 1
      valid_orbit_id(n_active_orbits) = id;
    enddo
  enddo
  !> copy only active orbits
  allocate(x(n_vec,n_active_orbits)); x       = 0d0;
  allocate(ppar(n_active_orbits));    ppar    = 0d0; 
  allocate(pperp(n_active_orbits));   pperp   = 0d0;
  allocate(weights(n_active_orbits)); weights = 0d0;
  x       = x_loc(:,valid_orbit_id(1:n_active_orbits))
  ppar    = ppar_loc(valid_orbit_id(1:n_active_orbits),1)
  pperp   = pperp_loc(valid_orbit_id(1:n_active_orbits),1)
  weights = weights_loc(valid_orbit_id(1:n_active_orbits),1)
  n_soft_points = n_active_orbits
  if(allocated(valid_orbit_id))     deallocate(valid_orbit_id)
  if(allocated(classification_loc)) deallocate(classification_loc)
  if(allocated(ppar_loc))           deallocate(ppar_loc)
  if(allocated(pperp_loc))          deallocate(pperp_loc)
  if(allocated(weights_loc))        deallocate(weights_loc)
  if(allocated(orbit_pdf_loc))      deallocate(orbit_pdf_loc)
  if(allocated(x_loc))              deallocate(x_loc)
end subroutine read_and_compute_soft_orbit_data

!> compute the error between the SOFT and the JOREK magnetic fields at the 
!> particle position and write it in a HDF5 file, both magnetic fields are
!> expressed in cylindrical coordinates: 1: R,2: Z,3: phi
!> inputs:
!>   fields:                   (fields_base) JOREK MHD fields
!>   n_vec:                    (integer) size of the position and magnetic field vector: 3
!>   time:                     (real8) time of the MHD field
!>   accepted_label:           (integer) SOFT label for accepted particles
!>   soft_orbit_filename_in:   (character) name of the SOFT orbit file
!>   Bfield_error_filename_in: (character) name of the hdf5 in which the 
!>                             SOFT-JOREK errors in the magnetic field are saved
subroutine compute_error_soft_jorek_particles_magnetic_field(fields,n_vec,time,&
accepted_label,soft_orbit_filename_in,Bfield_error_filename_in)
  use hdf5
  use mod_coordinate_transforms, only: cartesian_to_cylindrical
  use mod_coordinate_transforms, only: vector_cartesian_to_cylindrical
  use mod_fields,     only: fields_base
  use hdf5_io_module, only: HDF5_open,HDF5_open_or_create,HDF5_close
  use hdf5_io_module, only: HDF5_allocatable_array1D_reading_int
  use hdf5_io_module, only: HDF5_allocatable_array2D_reading
  use hdf5_io_module, only: HDF5_array1D_saving,HDF5_array2D_saving
  implicit none
  !> inputs:
  class(fields_base),intent(in)               :: fields 
  character(len=*),intent(in)                 :: soft_orbit_filename_in
  character(len=*),intent(in)                 :: Bfield_error_filename_in
  integer,intent(in)                          :: n_vec
  integer,dimension(:),allocatable,intent(in) :: accepted_label
  real*8,intent(in)                           :: time
  !> variables
  integer(HID_T) :: file_id
  integer        :: ii,jj,kk,n_orbits,n_times,n_soft_points,ierr,i_elm
  real*8         :: psi,U
  real*8,dimension(2) :: st,Rbox,Zbox
  real*8,dimension(3) :: RZPhi,Bvec_jorek,Evec_jorek
  integer,dimension(:),allocatable   :: classification_loc
  real*8,dimension(:),allocatable    :: error_Babs,error_Babs_norm
  real*8,dimension(:,:),allocatable  :: RZphi_loc,x_loc,Bvec_soft_loc
  real*8,dimension(:,:),allocatable  :: Babs_soft_loc,error_Bvec,error_Bvec_norm
  
  write(*,*) "Computing the magnetic field error at the particle position ..."
  !> open the soft orbit hdf5 file
  call HDF5_open(trim(soft_orbit_filename_in)//".h5",file_id,ierr)
  !> read sofit particles
  call HDF5_allocatable_array1D_reading_int(file_id,classification_loc,'/classification') !< orbit classification
  call HDF5_allocatable_array2D_reading(file_id,Babs_soft_loc,'/Babs') !< orbit magnetic field magnitude
  call HDF5_allocatable_array2D_reading(file_id,x_loc,'/x')            !< orbit x positions
  call HDF5_allocatable_array2D_reading(file_id,Bvec_soft_loc,'/B')    !< orbit magnetic field
  !> close the soft orbit hdf5 file
  call HDF5_close(file_id)
  !> reshape arrays
  n_orbits = size(Babs_soft_loc,2); n_times  = size(Babs_soft_loc,1);
  n_soft_points = n_orbits*n_times;
  x_loc         = reshape(x_loc,[n_vec,n_soft_points])
  Bvec_soft_loc = reshape(Bvec_soft_loc,[n_vec,n_soft_points])
  Babs_soft_loc = reshape(Babs_soft_loc,[1,n_soft_points])
  !> compute the error between the JOREK and SOFT magnetic fields\
  !> initialise and allocate particle simulation array
  call domain_bounding_box(fields%node_list,fields%element_list,Rbox(1),Rbox(2),Zbox(1),Zbox(2))
  allocate(error_Bvec(n_vec,n_soft_points)); error_Bvec = 0d0;
  allocate(error_Bvec_norm(n_vec,n_soft_points)); error_Bvec_norm = 0d0;
  allocate(RZPhi_loc(n_vec,n_soft_points)); RZPhi_loc = 0d0;
  allocate(error_Babs(n_soft_points)); error_Babs = 0d0;
  allocate(error_Babs_norm(n_soft_points)); error_Babs_norm = 0d0;
  !$omp parallel do default(none) firstprivate(n_orbits,n_times,accepted_label,&
  !$omp Rbox,Zbox,time) shared(classification_loc,x_loc,fields,error_Bvec,error_Babs,&
  !$omp error_Bvec_norm,error_Babs_norm,RZPhi_loc,Bvec_soft_loc,Babs_soft_loc) &
  !$omp private(kk,jj,ii,RZPhi,i_elm,st,ierr,Evec_jorek,Bvec_jorek,psi,U) collapse(2)
  do kk=1,n_orbits
    do jj=1,n_times
      if(.not.(any(classification_loc(kk).eq.accepted_label))) cycle
      ii=(kk-1)*n_times+jj
      !> transform the soft orbit coordinates in jorek global/local coordinates
      RZphi = cartesian_to_cylindrical(x_loc(:,ii)); i_elm = -1;
      if(((RZphi(1).ge.Rbox(1)).and.(RZphi(1).le.Rbox(2))).and.((RZphi(2).ge.Zbox(1)).and.(RZphi(2).le.Zbox(2)))) then
        call find_RZ(fields%node_list,fields%element_list,RZphi(1),RZphi(2),&
        RZphi(1),RZphi(2),i_elm,st(1),st(2),ierr)
      else
        cycle
    endif
    if(i_elm.le.0) cycle
      call fields%calc_EBpsiU(time,i_elm,st,RZPhi(3),Evec_jorek,Bvec_jorek,psi,U) 
      error_Bvec(:,ii)      = Bvec_jorek-vector_cartesian_to_cylindrical(RZPhi(3),Bvec_soft_loc(:,ii))
      error_Bvec_norm(:,ii) = error_Bvec(:,ii)/Bvec_jorek
      error_Babs(ii)        = norm2(Bvec_jorek)-Babs_soft_loc(1,ii)
      error_Babs_norm (ii)  = error_Babs(ii)/norm2(Bvec_jorek)
      RZPhi_loc(:,ii)       = RZPhi
    enddo
  enddo
  !$omp end parallel do
  !> Open SOFT-JOREK magnetic field error
  call HDF5_open_or_create(trim(Bfield_error_filename_in)//".h5",file_id,ierr,file_access=H5F_ACC_TRUNC_F)
  call HDF5_array2D_saving(file_id,x_loc,n_vec,n_soft_points,'x')
  call HDF5_array2D_saving(file_id,RZPhi_loc,n_vec,n_soft_points,'RZPhi')
  call HDF5_array1D_saving(file_id,error_Bvec(1,:),n_soft_points,'error_BR')
  call HDF5_array1D_saving(file_id,error_Bvec(2,:),n_soft_points,'error_BZ')
  call HDF5_array1D_saving(file_id,error_Bvec(3,:),n_soft_points,'error_Bphi')
  call HDF5_array1D_saving(file_id,error_Bvec_norm(1,:),n_soft_points,'error_BR_norm')
  call HDF5_array1D_saving(file_id,error_Bvec_norm(2,:),n_soft_points,'error_BZ_norm')
  call HDF5_array1D_saving(file_id,error_Bvec_norm(3,:),n_soft_points,'error_Bphi_norm')
  call HDF5_array1D_saving(file_id,error_Babs,n_soft_points,'error_Babs')
  call HDF5_array1D_saving(file_id,error_Babs_norm,n_soft_points,'error_Babs_norm')
  call HDF5_close(file_id)

  !> cleanup
  if(allocated(classification_loc)) deallocate(classification_loc)
  if(allocated(Babs_soft_loc))      deallocate(Babs_soft_loc) 
  if(allocated(Bvec_soft_loc))      deallocate(Bvec_soft_loc)
  if(allocated(x_loc))              deallocate(x_loc)
  if(allocated(RZPhi_loc))          deallocate(RZPhi_loc)
  if(allocated(error_Babs))         deallocate(error_Babs)
  if(allocated(error_Bvec))         deallocate(error_Bvec)
  if(allocated(error_Babs_norm))    deallocate(error_Babs_norm)
  if(allocated(error_Bvec_norm))    deallocate(error_Bvec_norm)
  write(*,*) "Computing the magnetic field error at particle position: terminated!"
end subroutine compute_error_soft_jorek_particles_magnetic_field

!> check if the mesh is uniform and extract the element size
!> inputs:
!>   mesh: (real8)(n) mesh
!> outputs:
!>   lenght: (real8) length of one element
subroutine check_uniform_mesh_extract_length_1d(length,mesh)
  implicit none
  !> inputs:
  real*8,dimension(:),allocatable :: mesh
  !> outputs:
  real*8,intent(out) :: length
  !> variables
  integer :: nmesh
  real*8  :: tol
  !> initialisation
  tol = 1d-15; nmesh = size(mesh);
  !> check consistency
  length = mesh(2)-mesh(1)
  if(any(abs(((mesh(2:nmesh)-mesh(1:nmesh-1))-length)/maxval(abs(mesh))).gt.tol)) then
    write(*,*) "WARNING: mesh is not uniform! error: ",&
    maxval(abs(((mesh(2:nmesh)-mesh(1:nmesh-1))-length)/maxval(abs(mesh))))
  endif
end subroutine check_uniform_mesh_extract_length_1d

!> compute the pdf for each soft orbit. The pdf is computed for t=0 and then replicated for
!> all other times
!> inputs:
!>   nvec:          (integer) size of the position vector must be 3
!>   ntimes:        (integer) number of orbit times
!>   norbits:       (integer) number of soft orbits
!>   RZ_axis:       (real8)(2) position of the magnetic axis
!>   orbit_x:       (real8)(ntimes*nvec,norbits) orbit position in cartesian coord.
!>   orbit_ppar:    (real8)(ntimes,norbits) orbit parallel momentum
!>   orbit_pperp:   (real8)(ntimes,norbits) orbit perpendicular momentum
!>   Rmesh:         (real8)(nR) major radius coordinate mesh of the minor radii
!>   Zmesh:         (real8)(nZ) vertical coordinate mesh of the minor radii
!>   poloidal_flux: (real8)(nZ,nR) soft poloidal flux
!>   r_minor_mesh:  (real8)(nr) soft pdf minor radius mesh
!>   pdf_list:      (type_soft_pdf) soft input pdf list
!> outputs:
!>   orbit_pdfs:   (real8)(norbits) pdf for each orbit and time
subroutine compute_pdf_orbit(nvec,ntimes,norbits,RZ_axis,orbit_x,orbit_ppar,&
orbit_pperp,Rmesh,Zmesh,poloidal_flux,r_minor_mesh,pdf_list,orbit_pdfs)
  implicit none
  !> inputs:
  type(type_soft_pdf),dimension(:),allocatable,intent(in)    :: pdf_list
  integer,intent(in) :: nvec,ntimes,norbits
  real*8,dimension(2),intent(in)                   :: RZ_axis
  real*8,dimension(ntimes,norbits),intent(in)      :: orbit_ppar,orbit_pperp
  real*8,dimension(nvec*ntimes,norbits),intent(in) :: orbit_x
  real*8,dimension(:),allocatable,intent(in)       :: Rmesh,Zmesh,r_minor_mesh
  real*8,dimension(:,:),allocatable,intent(in)     :: poloidal_flux
  !> outputs:
  real*8,dimension(norbits),intent(out)            :: orbit_pdfs
  !> variables:
  integer :: ii
  integer,dimension(2) :: ids_pflux
  integer,dimension(3) :: dims
  real*8               :: pflux_pos
  real*8,dimension(2)  :: ZRpos,pxipos,values_pflux,pdf_values
  real*8,dimension(:),allocatable :: poloidal_flux_pdf
  logical              :: increasing
  !> initialisaition
  dims = [size(Zmesh),size(Rmesh),size(r_minor_mesh)]
  allocate(poloidal_flux_pdf(dims(3)))
  !> compute the poloidal flux at the pdf mesh point
  do ii=1,dims(3)
    poloidal_flux_pdf(ii) = bilinear_interp_cartesian(RZ_axis(2),RZ_axis(1)+r_minor_mesh(ii),&
                            dims(1),dims(2),Zmesh,Rmesh,poloidal_flux)
  enddo
  increasing = poloidal_flux_pdf(1).lt.poloidal_flux_pdf(2)
  !> loop on the orbit
  !$omp parallel do default(none) firstprivate(norbits,dims,increasing) &
  !$omp private(ii,ZRpos,pxipos,pflux_pos,ids_pflux,values_pflux,pdf_values) &
  !$omp shared(orbit_x,orbit_ppar,orbit_pperp,Zmesh,Rmesh,poloidal_flux,&
  !$omp poloidal_flux_pdf,pdf_list,orbit_pdfs)
  do ii=1,norbits
    ZRpos = [orbit_x(3,ii),sqrt((orbit_x(1,ii))**2+(orbit_x(2,ii))**2)] !< ZR position of the orbit at t=0
    if(ZRpos(2).eq.0d0) cycle
    pxipos(1) = sqrt(orbit_ppar(1,ii)**2 + orbit_pperp(1,ii)**2)
    pxipos(2) = orbit_ppar(1,ii)/pxipos(1)
    !> interpolate the magnetic minor radius
    pflux_pos = bilinear_interp_cartesian(ZRpos(1),ZRpos(2),dims(1),dims(2),Zmesh,Rmesh,poloidal_flux)
    !> find the minor radius within the pdf mesh
    call find_point_segment(pflux_pos,dims(3),poloidal_flux_pdf,ids_pflux,values_pflux,increasing)
    !> interpolate in momentum space the pdf values
    pdf_values(1) = bilinear_interp_cartesian(pxipos(1),pxipos(2),&
                    size(pdf_list(ids_pflux(1))%p),size(pdf_list(ids_pflux(1))%xi),&
                    pdf_list(ids_pflux(1))%p,pdf_list(ids_pflux(1))%xi,pdf_list(ids_pflux(1))%pdf)
    pdf_values(2) = bilinear_interp_cartesian(pxipos(1),pxipos(2),&
                    size(pdf_list(ids_pflux(2))%p),size(pdf_list(ids_pflux(2))%xi),&
                    pdf_list(ids_pflux(2))%p,pdf_list(ids_pflux(2))%xi,pdf_list(ids_pflux(2))%pdf)
    !> interpolate and store the pdfs w.r.t. the minor radius
    orbit_pdfs(ii) = linear_interp(pflux_pos,values_pflux,pdf_values)
    if(orbit_pdfs(ii).lt.0d0) write(*,*) "dioporco: ",ids_pflux,values_pflux
  enddo
  !$omp end parallel do
  if(allocated(poloidal_flux_pdf)) deallocate(poloidal_flux_pdf)
end subroutine compute_pdf_orbit 

!> allocate soft pdf type and initialize to 0
!> inputs:
!>   soft_pdf: (type_soft_pdf) soft_pdf to be initialized
!> outputs:
!>   soft_pdf: (type_soft_pdf) initialized soft_pdf
subroutine init_soft_pdf(n_pxi,soft_pdf)
  implicit none
  !> inputs-outputs:
  type(type_soft_pdf),intent(inout) :: soft_pdf
  integer,dimension(2),intent(in)   :: n_pxi
  !> allocate soft_pdf and initialize to 0
  if(allocated(soft_pdf%p)) deallocate(soft_pdf%p); 
  allocate(soft_pdf%p(n_pxi(1))); soft_pdf%p=0d0;
  if(allocated(soft_pdf%xi)) deallocate(soft_pdf%xi); 
  allocate(soft_pdf%xi(n_pxi(2))); soft_pdf%xi=0d0;
  if(allocated(soft_pdf%pdf)) deallocate(soft_pdf%pdf); 
  allocate(soft_pdf%pdf(n_pxi(1),n_pxi(2))); soft_pdf%pdf=0d0;
end subroutine init_soft_pdf

!> read and distribute the pdf used in SOFT and its mesh (r,xi,p)
!> inputs:
!>   soft_pdf_filename_in: (charcater)(*) soft pdf filename
!> outputs:
!>   n_r_mesh: (integer) size of the pdf minor radius mesh
!>   r_mesh:   (real8)(n_r_mesh) pdf minor radius mesh
!>   soft_pdf: (type_soft_pdf)(n_r_mesh) the pdf used by soft with the xi,p meshes
subroutine read_soft_pdf_file(soft_pdf_filename_in,n_r_mesh,r_mesh,soft_pdf)
  use hdf5
  use hdf5_io_module, only: HDF5_open,HDF5_close
  use hdf5_io_module, only: HDF5_allocatable_array1D_reading
  use hdf5_io_module, only: HDF5_allocatable_array2D_reading
  implicit none
  !> inputs:
  character(len=*),intent(in) :: soft_pdf_filename_in
  !> outputs:
  integer,intent(out) :: n_r_mesh
  real*8,dimension(:),allocatable,intent(out)              :: r_mesh
  type(type_soft_pdf),dimension(:),allocatable,intent(out) :: soft_pdf
  !> variables:
  character(len=10) :: format_char
  character(len=:),allocatable  :: group_name
  integer(HID_T)    :: file_id
  integer           :: ii,ierr,r_id,n_r_id,group_name_len
  !> open hdf5 file
  call HDF5_open(trim(soft_pdf_filename_in)//".h5",file_id,ierr)
  !> read mesh datasets
  call HDF5_allocatable_array1D_reading(file_id,r_mesh,"r")
  !> allocate soft pdf data strucutre
  n_r_mesh = size(r_mesh); allocate(soft_pdf(n_r_mesh))
  !> read soft pdf
  do ii=1,n_r_mesh
    !> define the group name compatible with soft nomenclature
    r_id = ii-1
    n_r_id = int(log10(real(r_id)))+1
    if(r_id.eq.0) n_r_id = 1
    write(format_char,'(A,I1,A)') "(A,I",n_r_id,")"
    group_name_len = 2+n_r_id
    allocate(character(len=group_name_len)::group_name)
    write(group_name,trim(format_char)) "/r",r_id
    !> read the pdf data 
    call HDF5_allocatable_array1D_reading(file_id,soft_pdf(ii)%p,trim(group_name)//"/p")
    call HDF5_allocatable_array1D_reading(file_id,soft_pdf(ii)%xi,trim(group_name)//"/xi")
    call HDF5_allocatable_array2D_reading(file_id,soft_pdf(ii)%pdf,trim(group_name)//"/f")
    deallocate(group_name)
  enddo
  !> close hdf5 file
  call HDF5_close(file_id)
end subroutine read_soft_pdf_file

!> read and compute the soft minor radii
!> inputs:
!>   my_id:       (integer) mpi task number
!>   filename_in: (character) name of the soft magnetic field file
!> outputs:
!>   RZ_axis:       (real8)(2) magnetic axis position
!>   R_mesh:        (real8)(nR) soft major radius mesh
!>   Z_mesh:        (real8)(nZ) soft vertical position mesh
!>   poloidal_flux: (real8)(nZ,nR) soft minor radii
subroutine read_soft_maxis_and_poloidal_flux(filename_in,RZ_axis,R_mesh,Z_mesh,poloidal_flux)
  use hdf5
  use hdf5_io_module, only: HDF5_open,HDF5_close
  use hdf5_io_module, only: HDF5_array1D_reading
  use hdf5_io_module, only: HDF5_allocatable_array1D_reading
  use hdf5_io_module, only: HDF5_allocatable_array2D_reading
  implicit none
  !> inputs:
  character(len=*),intent(in) :: filename_in
  !> outputs:
  real*8,dimension(2),intent(out)               :: RZ_axis
  real*8,dimension(:),allocatable,intent(out)   :: R_mesh,Z_mesh
  real*8,dimension(:,:),allocatable,intent(out) :: poloidal_flux
  !> variables:
  integer(HID_T) :: file_id 
  integer        :: n_separatrix,ierr
  integer,dimension(4) :: dims
  real*8               :: poloidal_flux_axis,poloidal_flux_bnd
  real*8,dimension(:,:),allocatable :: separatrix
  !> open hdf5 file
  call HDF5_open(trim(filename_in)//".h5",file_id,ierr)
  call HDF5_array1D_reading(file_id,RZ_axis,"/maxis")
  call HDF5_allocatable_array1D_reading(file_id,R_mesh,"/r")
  call HDF5_allocatable_array1D_reading(file_id,Z_mesh,"/z")
  call HDF5_allocatable_array2D_reading(file_id,poloidal_flux,"/Psi")
  !> close hdf5 file
  call HDF5_close(file_id)
end subroutine read_soft_maxis_and_poloidal_flux

!> perform a linear interpolation between two nodes
!> inputs:
!>   pos:    (real8) interpolation position
!>   values: (real8)(2) position values of the nodes
!>   nodes:  (real8)(2) function values at the nodes
!> outputs:
!>   f: (real8) interpolated value
function linear_interp(pos,values,nodes) result(f)
  implicit none
  !> inputs:
  real*8,intent(in)              :: pos
  real*8,dimension(2),intent(in) :: values,nodes
  !> outputs:
  real*8 :: f
  !> compute interpolation
  f = nodes(1) + (nodes(2)-nodes(1))*((pos-values(1))/(values(2)-values(1)))
end function linear_interp

!> perform a bilinear interpolation in cartesian coordinates
!> 4 - 3
!> |   |
!> 1 - 2
!> inputs:
!>   pos_1:  (real8) target first position
!>   pos_2:  (real8) target second position
!>   dim_1:  (integer) size mesh first coordinate
!>   dim_2:  (integer) size mesh second coordinate
!>   mesh_1: (real8)(dim_1) mesh first coordinate
!>   mesh_2: (real8)(dim_2) mesh second coordinate
!>   matrix: (real8)(dim_1,dim_2) matrix of values to interpolate
!> outputs:
!>   f:      (real8) interpolated value
function bilinear_interp_cartesian(pos_1,pos_2,dim_1,dim_2,&
mesh_1,mesh_2,matrix) result(f)
  implicit none
  !> inputs:
  integer,intent(in) :: dim_1,dim_2
  real*8,intent(in)  :: pos_1,pos_2
  real*8,dimension(dim_1),intent(in) :: mesh_1
  real*8,dimension(dim_2),intent(in) :: mesh_2
  real*8,dimension(dim_1,dim_2),intent(in) :: matrix
  !> outputs:
  real*8 :: f
  integer,dimension(2) :: id_1,id_2
  real*8,dimension(2)  :: nodes_1,nodes_2
  !> find poloidal flux at the magnetic axis
  call find_point_segment(pos_1,dim_1,mesh_1,id_1,nodes_1,mesh_1(1).lt.mesh_1(2))
  call find_point_segment(pos_2,dim_2,mesh_2,id_2,nodes_2,mesh_2(1).lt.mesh_2(2))
  !> compute bilinear interpolation
  f = (matrix(id_1(1),id_2(1))*(nodes_1(2)-pos_1)*(nodes_2(2)-pos_2) + &
      matrix(id_1(1),id_2(2))*(pos_1-nodes_1(1))*(nodes_2(2)-pos_2) + &
      matrix(id_1(2),id_2(2))*(pos_1-nodes_1(1))*(pos_2-nodes_2(1)) + &
      matrix(id_1(2),id_2(1))*(nodes_1(2)-pos_1)*(pos_2-nodes_2(1)))/&
      ((nodes_1(2)-nodes_1(1))*(nodes_2(2)-nodes_2(1)))
end function bilinear_interp_cartesian


!> identify mesh element containing a target point
!> inputs:
!>   x:      (real8) target point
!>   nx:     (integer) number of mesh nodes
!>   x_mesh: (real8)(nx) 1D mesh
!>   increasing_in: (logical) if true the mesh value is considered increasing with index
!> outputs:
!>   id_x:   (integer)(2) indexes of the nodes containing the target point
!>   xnodes: (real8)(2) min,max of the value
subroutine find_point_segment(x,nx,x_mesh,id_x,xnodes,increasing_in)
  implicit none
  !> inputs:
  integer,intent(in) :: nx
  real*8,intent(in)  :: x 
  real*8,dimension(nx),intent(in) :: x_mesh
  logical,intent(in),optional :: increasing_in
  !> outputs:
  integer,dimension(2),intent(out) :: id_x
  real*8,dimension(2),intent(out)  :: xnodes
  !> variables:
  logical :: increasing
  !> initialisation
  increasing = .true.
  if(present(increasing_in)) increasing = increasing_in
  !> identify the nearest node
  id_x = [minloc(abs(x-x_mesh)),-1];
  if(id_x(1).le.1) then 
    id_x = [1,2];
  elseif(id_x(1).ge.nx) then
    id_x = [nx-1,nx]
  else
    if(increasing) then
      if(x.ge.x_mesh(id_x(1))) then
        id_x = [id_x(1),id_x(1)+1]
      elseif(x.lt.x_mesh(id_x(1))) then
        id_x = [id_x(1)-1,id_x(1)]
      endif
    else
      if(x.le.x_mesh(id_x(1))) then
        id_x = [id_x(1),id_x(1)+1]
      elseif(x.gt.x_mesh(id_x(1))) then
        id_x = [id_x(1)-1,id_x(1)]
      endif
    endif
  endif
  xnodes = x_mesh(id_x)
end subroutine find_point_segment

!> write jorek particles in HDF5
!> inputs:
!>   my_id:    (integer)
!>   filename: (character) hdf5 filename
!>   n_mpis:   (integer) total number of mpi tasks
!>   n_vec:    (integer) size of the x position vector
!>   sim:      (particle_sim) jorek particle simulation
!> outputs:
subroutine write_particles_in_hdf5(my_id,filename,n_mpis,n_vec,sim)
  use constants,      only: SPEED_OF_LIGHT
  use mpi
  use hdf5
  use hdf5_io_module, only: HDF5_open_or_create,HDF5_close
  use hdf5_io_module, only: HDF5_array2D_saving,HDF5_array1D_saving
  use mod_coordinate_transforms, only: cylindrical_to_cartesian
  implicit none
  !> inputs:
  type(particle_sim),intent(in) :: sim
  character(len=*),intent(in)   :: filename
  integer,intent(in)            :: my_id,n_mpis,n_vec
  !> variables:
  integer(HID_T) :: file_id
  integer        :: ii,n_particles,ierr
  real*8         :: U,psi
  real*8,dimension(n_vec) :: Bvec,Evec
  real*8,dimension(:,:),allocatable :: x_pos,x_pos_glob
  real*8,dimension(:),allocatable   :: ppar,pperp,ppar_glob,pperp_glob
  !> initialisation
  n_particles = size(sim%groups(1)%particles);
  !> compute position in cartesian coordinates, ppar and pperp
  allocate(x_pos(n_vec,n_particles)); x_pos = 0d0;
  allocate(ppar(n_particles));        ppar  = 0d0;
  allocate(pperp(n_particles));       pperp = 0d0;
  allocate(x_pos_glob(n_vec,n_mpis*n_particles)); x_pos_glob = 0d0;
  allocate(ppar_glob(n_mpis*n_particles));        ppar_glob  = 0d0;
  allocate(pperp_glob(n_mpis*n_particles));       pperp_glob = 0d0; 
  !$omp parallel do default(shared) firstprivate(n_particles) &
  !$omp private(ii,Evec,Bvec,psi,U)
  do ii=1,n_particles
    if(sim%groups(1)%particles(ii)%i_elm.le.0) cycle
    x_pos(:,ii) = cylindrical_to_cartesian(sim%groups(1)%particles(ii)%x)
    call sim%fields%calc_EBpsiU(sim%time,sim%groups(1)%particles(ii)%i_elm,\
    sim%groups(1)%particles(ii)%st,sim%groups(1)%particles(ii)%x(3),Evec,Bvec,psi,U)
    select type (part=>sim%groups(1)%particles(ii))
    type is (particle_gc_relativistic)
      ppar(ii) = part%p(1)/(sim%groups(1)%mass*SPEED_OF_LIGHT)
      pperp(ii) = sqrt(2d0*sim%groups(1)%mass*norm2(Bvec)*part%p(2))/\
      (sim%groups(1)%mass*SPEED_OF_LIGHT)
    end select
  enddo
  !$omp end parallel do
  !> gather data from all mpi tasks 
  call MPI_gather(x_pos,n_vec*n_particles,MPI_REAL8,x_pos_glob,n_vec*n_particles,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
  call MPI_gather(ppar,n_particles,MPI_REAL8,ppar_glob,n_particles,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
  call MPI_gather(pperp,n_particles,MPI_REAL8,pperp_glob,n_particles,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
  !> write data in hdf5
  if(my_id.eq.0) then
    call HDF5_open_or_create(trim(filename//'.h5'),file_id,ierr,file_access=H5F_ACC_TRUNC_F)
    call HDF5_array2D_saving(file_id,x_pos_glob,n_vec,n_mpis*n_particles,'x')
    call HDF5_array1D_saving(file_id,ppar_glob,n_mpis*n_particles,'ppar')
    call HDF5_array1D_saving(file_id,pperp_glob,n_mpis*n_particles,'pperp')
    call HDF5_close(file_id)
  endif
  !> cleanup
  if(allocated(x_pos)) deallocate(x_pos); if(allocated(x_pos_glob)) deallocate(x_pos_glob);
  if(allocated(ppar))  deallocate(ppar);  if(allocated(ppar_glob))  deallocate(ppar_glob);
  if(allocated(pperp)) deallocate(pperp); if(allocated(pperp_glob)) deallocate(pperp_glob);
end subroutine write_particles_in_hdf5

!> -----------------------------------------------------------------------------------------

end program generate_particle_restart_from_SOFT
 
