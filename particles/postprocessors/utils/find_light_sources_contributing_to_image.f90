!> find_light_sources_contributing_to_image identifies and 
!> writes in a file all the point light sources contributing
!> to a given image. The linght intensity emitted towards the 
!> camera is also stored for generating colormaps
program find_light_sources_contributing_to_image
use constants,                                       only: PI
use mod_mpi_tools,                                   only: init_mpi_threads,finalize_mpi_threads
use particle_tracer
use mod_spectra_deterministic,                       only: spectrum_integrator_2nd
use mod_pinhole_lens,                                only: pinhole_lens
use mod_camera_perspective_static,                   only: camera_perspective_static
use mod_gyroaverage_synchrotron_light_dist_vertices, only: gyroaverage_synchrotron_light_dist
use phys_module

implicit none

!> Variables -----------------------------------------------------------------------------------------------
type(pcg32_rng)                             :: rng_pcg32
type(event)                                 :: field_reader,particle_reader,particle_writer
type(pinhole_lens)                          :: lens
type(spectrum_integrator_2nd)               :: spectra
type(camera_perspective_static)             :: camera
type(gyroaverage_synchrotron_light_dist)    :: synch_sources
type(particle_sim),dimension(:),allocatable :: sims,sims_particle_reconstructed
logical                                     :: do_jorek_init
integer                                     :: ii,n_particles_tot
integer                                     :: n_groups,my_id,n_mpis,n_x,ierr
integer                                     :: n_times,n_wavelengths,n_spectra
integer                                     :: n_int_camera_param,n_real_camera_param
integer                                     :: n_null_particles
integer,dimension(1)                        :: signs_p_parallel_gc_ref,signs_charge_ref
integer,dimension(:),allocatable            :: int_camera_param,signs_p_gc_parallel,signs_charge
integer,dimension(:,:,:),allocatable        :: active_light_pixel_ids
real*8                                      :: t1,t0,accept_dark_lights_threshold
real*8,dimension(1)                         :: masses_ref 
real*8,dimension(:),allocatable             :: min_spectra,max_spectra,pinhole_positions
real*8,dimension(:),allocatable             :: masses,real_camera_param,sim_times
real*8,dimension(:,:,:),allocatable         :: active_light_source_positions
real*8,dimension(:,:,:),allocatable         :: active_light_source_intensities
real*8,dimension(:,:,:),allocatable         :: active_light_pixel_coordinates
logical                                     :: write_particle_restart_from_light
character(len=3)                            :: hdf5ext
character(len=15)                           :: particle_filename
character(len=17)                           :: fields_filename
character(len=29)                           :: light_output_filename
character(len=30)                           :: camera_output_filename
character(len=30)                           :: reconstruct_particle_filename
character(len=60),dimension(:),allocatable  :: particle_filenames

!> Variable presets -----------------------------------------------------------------------
hdf5ext = '.h5'

!> Variables definitions -----------------------------------------------------------------------------------
fields_filename                   = 'jorek_equilibrium'
light_output_filename             = 'contributing_light_intesities'
camera_output_filename            = 'contributing_camera_intesities'
n_x                               = 3 !< size of the spatial coordinates array
n_groups                          = 1 !< number of particle groups
n_spectra                         = 1 !< number of spectra
n_wavelengths                     = 40 !< number of wavelengths per spectrum
n_int_camera_param                = 5
n_real_camera_param               = 9
n_times                           = 11
write_particle_restart_from_light = .true.
n_null_particles                  = 15 !< number of null particles to be kept reconstructed particles
accept_dark_lights_threshold      = 1d-1 !< probability of accepting dark particles
masses_ref                        = 5.48579909065d-4 !< electron mass in AMU 
signs_p_parallel_gc_ref           = -1; !< sign of the gc parallel momentum for gc reconstruction
signs_charge_ref                  = -1; !< sign of the particle charge for particle reconstruction
reconstruct_particle_filename     = 'reconstructed_particle_restart' 
!> se the list of particle restart files to be read
allocate(character(len=60)::particle_filenames(n_times)); particle_filenames = ''
particle_filenames = [character(len=60)::'part_restart000.00339941',&
'part_restart000.00349941','part_restart000.00359941',&
'part_restart000.00369941','part_restart000.00379941',&
'part_restart000.00389941','part_restart000.00399941',&
'part_restart000.00409941','part_restart000.00419941',&
'part_restart000.00429941','part_restart000.00439941']
!> JET KLDT-E5WC wavlength: 3d-6 , 3.5d-6 [m]
allocate(min_spectra(n_spectra)); min_spectra = [3d-6];
allocate(max_spectra(n_spectra)); max_spectra = [3.5d-6];
allocate(pinhole_positions(n_x)); pinhole_positions = [-8.86d-1,-4.002,-3.32d-1];
!> one pinhole => n_lens_samples=1, JET KLDT-E5WC pixels nx=120,ny=176
allocate(int_camera_param(n_int_camera_param)); int_camera_param = [1,600,600,0,1]
!> JET KLDT-E5WC camera inputs:
!> 1:3 -> half width, half height and orientation of the visual plane angle
!> 4:6 -> camera focal direction: focal distance, latitude, azimuth
!> 7:9 -> camera position in the tokamak reference system
allocate(real_camera_param(n_real_camera_param));
real_camera_param = [5.23d-1,5.23d-1,5d-1*PI,9.998025d-1,1.5807985,2.09801,-8.86d-1,-4.002,-3.32d-1]

!> Initialisation ------------------------------------------------------------------------------------------
!> initialise the MPI communicator
call init_mpi_threads(my_id,n_mpis,ierr)

!> set the number of particle groups
n_part_groups = n_groups

!> read particle, time and MHD fields
write(*,*) "Reading particle and MHD data ..."
do_jorek_init = .true.;
allocate(sim_times(n_times)); allocate(sims(n_times));
field_reader = event(read_jorek_fields_interp_linear(basename=trim(fields_filename),i=-1))
do ii=1,n_times
  call sims(ii)%initialize(.true.,my_id,n_mpis,do_jorek_init)
  write(*,*) 'my_id: ',sims(ii)%my_id,'doing particle restart file: ',trim(particle_filenames(ii))
  particle_reader = event(read_action(filename=trim(particle_filenames(ii))//trim(hdf5ext)))
  call with(sims(ii),particle_reader); sim_times(ii) = sims(ii)%time;
  call with(sims(ii),field_reader); do_jorek_init = .false.;
enddo
write(*,*) "Reading particle and MHD data: completed!"

!> Initialise synthetic camera and light sources
write(*,*) 'Initialise synthetic camera and light sources ...'
t0 = MPI_Wtime()
spectra = spectrum_integrator_2nd(n_wavelengths,n_spectra,min_spectra,max_spectra)
call spectra%generate_spectrum()
call lens%init_pinhole(n_x,pinhole_positions)
call camera%init_camera(lens,spectra,n_int_camera_param,n_real_camera_param,&
int_camera_param,real_camera_param)
call synch_sources%init_lights_from_particles(n_times,sims)
n_particles_tot = sum(synch_sources%n_active_vertices)
t1 = MPI_Wtime()
allocate(active_light_pixel_ids(2,n_particles_tot,camera%n_vertices));
active_light_pixel_ids = 0;
allocate(active_light_source_positions(n_x,n_particles_tot,camera%n_vertices)) 
active_light_source_positions = 0d0;
allocate(active_light_source_intensities(n_spectra,n_particles_tot,camera%n_vertices)); 
active_light_source_intensities = 0d0;
allocate(active_light_pixel_coordinates(2,n_particles_tot,camera%n_vertices));
active_light_pixel_coordinates = 0d0;
write(*,*) 'Initialise synthetic camera and light sources: completed!'
write(*,*) my_id, 'System time fast camera initialisation (s): ',t1-t0

!> Find active light sources -------------------------------------------------------------------------------
write(*,*) "Findig light sources contributing to an image and computing spectral intensities: ..."
t0 = MPI_Wtime()
call compute_contribution_light_source_to_image_static(camera,synch_sources,spectra,&
n_particles_tot,active_light_source_positions,active_light_source_intensities,&
active_light_pixel_ids,active_light_pixel_coordinates)
t1 = MPI_Wtime()
write(*,*) "Findig light sources contributing to an image and computing spectral intensities: completed!"
write(*,*) my_id,'System time finding contributing light sources and computing intensities (s): ',t1-t0

!> generate particle list from active lights if required ---------------------------------------------------
if(write_particle_restart_from_light) then
write(*,*) "Generate particle simulations from light spectral intensities: ..."
t0 = MPI_Wtime()
if(allocated(masses))           deallocate(masses)
if(allocated(signs_p_gc_parallel)) deallocate(signs_p_gc_parallel)
if(allocated(signs_charge))     deallocate(signs_charge)
allocate(masses(camera%n_vertices)); allocate(signs_p_gc_parallel(camera%n_vertices));
allocate(signs_charge(camera%n_vertices));
masses = masses_ref(1); if(size(masses_ref).eq.camera%n_vertices) masses = masses_ref; 
signs_p_gc_parallel= signs_p_parallel_gc_ref(1) 
if(size(signs_p_parallel_gc_ref).eq.camera%n_times) signs_p_gc_parallel = signs_p_parallel_gc_ref;
signs_charge = signs_charge_ref(1)
if(size(signs_charge_ref).eq.camera%n_times) signs_charge = signs_charge_ref
call generate_particle_simulations_from_active_light_sources(synch_sources,&
rng_pcg32,sims(1)%fields,my_id,n_mpis,camera%n_times,n_null_particles,n_particles_tot,&
spectra%n_spectra,accept_dark_lights_threshold,masses,camera%times,active_light_source_intensities,&
signs_charge,sims_particle_reconstructed,signs_p_parallel_in=signs_p_parallel_gc_ref)
t1 = MPI_Wtime()
write(*,*) "Generating particle simulations from light spectral intensities: completed!"
write(*,*) my_id,'System time generating particle simulations from light spectral intensities (s): ',t1-t0
endif
!> Write active light sources ------------------------------------------------------------------------------
#ifdef USE_HDF5
write(*,*) "Write contributing light sources in HDF5 file ..."
call write_source_light_positions_contributions_in_hdf5(light_output_filename,my_id,n_x,&
n_spectra,n_particles_tot,camera%n_vertices,active_light_source_positions,&
active_light_source_intensities,active_light_pixel_ids,active_light_pixel_coordinates,ierr)
write(*,*) "Writing contributing light sources in HDF5 file: completed!"
write(*,*) "Write camera data in HDF5 file ..."
call write_pinholes_planes_directions_in_hdf5(camera_output_filename,my_id,camera,ierr)
write(*,*) "Writing camera data in HDF5 file: completed!"
if(write_particle_restart_from_light) then
  write(*,*) "Write reconstructed particle lists in HDF5 file ..."
  do ii=1,size(sims_particle_reconstructed)
    particle_writer = event(write_action(basename=reconstruct_particle_filename))
    call with(sims_particle_reconstructed(ii),particle_writer)
  enddo
  write(*,*) "Writing reconstructed particle lists in HDF5: completed!"
endif
#endif
!> Finalisation --------------------------------------------------------------------------------------------
if(allocated(min_spectra))           deallocate(min_spectra);
if(allocated(max_spectra))           deallocate(max_spectra);
if(allocated(pinhole_positions))     deallocate(pinhole_positions);
if(allocated(int_camera_param))      deallocate(int_camera_param);
if(allocated(real_camera_param))     deallocate(real_camera_param);
if(allocated(sim_times))             deallocate(sim_times);
if(allocated(sims))                  deallocate(sims);
if(allocated(active_light_source_positions))   deallocate(active_light_source_positions);
if(allocated(active_light_source_intensities)) deallocate(active_light_source_intensities);
if(allocated(active_light_pixel_ids))          deallocate(active_light_pixel_ids);
if(allocated(active_light_pixel_coordinates))  deallocate(active_light_pixel_coordinates);
if(allocated(masses))                deallocate(masses)
if(allocated(signs_p_gc_parallel))   deallocate(signs_p_gc_parallel)
if(allocated(signs_charge))          deallocate(signs_charge)
call finalize_mpi_threads(ierr)

contains

!> Tools ---------------------------------------------------------------------------------------------------
!> methods used for computing the contribution of a light source
!> to an image (for each point on a lens and for all time)
!> inputs:
!>   camera_inout:  (camera_static)  class defining a static camera
!>   lights_inout:  (light_vertices) class containing all active lights
!>   spectra_inout: (spectral_base)  camera spectrum class
!>   n_particles:   (integer) total number of simulated particles
!> outputs:
!>   n_particles:                     (integer)(n_points_on_lens) number of contributing lights
!>                                    per lens point
!>   active_light_source_positions:   (real8)(n_x,n_particles,n_lens_points) positions of 
!>                                    the light sources contributing to an image
!>   active_light_source_intensities: (real8)(n_spectra,n_particles,n_lens_points)
!>                                    integrated spectral intensities of the light sources
!>                                    contributing to an image
!>   pixel_ids_light_source:          (integer)(2,n_particles,n_lens_points) indexes of the
!>                                    illuminated pixel
!>   pixel_coords_light_source:       (real8)(2,n_particles,n_lens_points) coordinates of
!>                                    the ray-pixel intersection in pixel local coordinates
subroutine compute_contribution_light_source_to_image_static(camera_inout,lights_inout,&
spectra_inout,n_particles,active_light_source_positions,active_light_source_intensities,&
pixel_ids_light_source,pixel_coords_light_source)
  use mod_spectra,        only: spectrum_base
  use mod_light_vertices, only: light_vertices
  use mod_camera_static,  only: camera_static
  implicit none
  !> inputs-outputs:
  class(camera_static),intent(inout)  :: camera_inout
  class(light_vertices),intent(inout) :: lights_inout
  class(spectrum_base),intent(inout)  :: spectra_inout
  !> inputs:
  integer,intent(in) :: n_particles
  !> outputs:
  integer,dimension(2,n_particles,camera_inout%n_vertices),intent(inout) :: pixel_ids_light_source
  real*8,dimension(camera_inout%n_x,n_particles,camera_inout%n_vertices),intent(inout) :: active_light_source_positions
  real*8,dimension(spectra_inout%n_spectra,n_particles,camera_inout%n_vertices),intent(inout) :: active_light_source_intensities
  real*8,dimension(2,n_particles,camera_inout%n_vertices),intent(inout)  :: pixel_coords_light_source

  !> variables:
  integer                                   :: ii,jj,kk,base_counter
  integer,dimension(2)                      :: i_pixel
  real*8,dimension(2)                       :: pixel_coords
  real*8,dimension(spectra_inout%n_spectra) :: intensity
  !> initialisation 
  base_counter = 0
  !> compute the contribution of each light for each time
  do jj=1,lights_inout%n_times !< loop on the times
    do ii=1,camera_inout%n_vertices !< loop on the camera lens points
      !$omp parallel do default(private) firstprivate(base_counter,jj,ii) &
      !$omp shared(lights_inout,camera_inout,spectra_inout,&
      !$omp active_light_source_positions,active_light_source_intensities,&
      !$omp pixel_ids_light_source,pixel_coords_light_source)
      do kk=1,lights_inout%n_active_vertices(jj) !< loop on the lights
        !> compute the light intensity
        call compute_received_light_intensity(camera_inout,lights_inout,spectra_inout,&
        ii,kk,jj,i_pixel,pixel_coords,intensity)
        !> store the light position and light contribution
        active_light_source_positions(:,base_counter+kk,ii)   = lights_inout%x(:,kk,jj)
        active_light_source_intensities(:,base_counter+kk,ii) = intensity
        !> store pixel indexes and pixel-ray intersection position for each contribution
        pixel_ids_light_source(:,base_counter+kk,ii)    = i_pixel
        pixel_coords_light_source(:,base_counter+kk,ii) = pixel_coords
      enddo
      !$omp end parallel do
    enddo
    base_counter = base_counter + lights_inout%n_active_vertices(jj)
  enddo
end subroutine compute_contribution_light_source_to_image_static

!> Method used for generating multiple particle simulations from contributing
!> and non contributing light sources
!> inputs:
!>   lights_inout:                    (light_vertices) class containing all active lights
!>   rng:                             (type_rng) random number generator
!>   fields:                          (fields_base) JOREK MHD fields
!>   my_id:                           (integer) id of the current mpi task
!>   n_mpis:                          (integer) number of mpi tasks
!>   n_times:                         (integer) number of times to be treated
!>   n_dead_particles:                (integer) number of required dead particles
!>   n_particles:                     (integer) number of active lights
!>   n_spectra:                       (integer) number of spectra
!>   accept_threshold:                (real8) theshold for accepting non emitting lights
!>   masses:                          (real8)(n_times) particle masses for each time
!>   times:                           (real8) simulation time for each time
!>   active_light_source_intensities: (real8)(n_spectra,n_particles,n_times) intensity of active lights
!>   signs_charge:                    (integer)(n_times) sign of the particle charge for each time
!>   signs_p_parallel_in:             (integer)(n_times) optional sign of parallel momentum for for
!>                                    each time relativistic guiding center particles, default: 1
!> outputs:
!>   sims_particle_out:                (type_particle_sim)(n_times) initialised particle simulations
subroutine generate_particle_simulations_from_active_light_sources(lights_inout,&
rng,fields,my_id,n_mpis,n_times,n_dead_particles,n_particles,n_spectra,accept_threshold,&
masses,times,active_light_source_intensities,signs_charge,sims_particle_out,signs_p_parallel_in)
  use mod_rng
  use mod_fields,         only: fields_base
  use mod_particle_sim,   only: particle_sim
  use mod_light_vertices, only: light_vertices
  implicit none
  !> inputs-outputs:
  class(light_vertices),intent(inout) :: lights_inout
  class(type_rng),intent(inout)       :: rng
  !> inputs:
  class(fields_base),intent(in)                              :: fields
  integer,intent(in)                                         :: n_dead_particles,n_particles,n_mpis,my_id
  integer,intent(in)                                         :: n_times,n_spectra
  integer,dimension(n_times),intent(in)                      :: signs_charge
  real*8,intent(in)                                          :: accept_threshold
  real*8,dimension(n_times),intent(in)                       :: times,masses
  real*8,dimension(n_spectra,n_particles,n_times),intent(in) :: active_light_source_intensities
  integer,dimension(n_times),intent(in),optional             :: signs_p_parallel_in
  !> outputs:
  type(particle_sim),dimension(:),allocatable,intent(out) :: sims_particle_out
  !> variables:
  integer :: ii
  integer,dimension(n_times) :: signs_p_parallel
  !> initialization
  signs_p_parallel = 1; if(present(signs_p_parallel_in)) signs_p_parallel = signs_p_parallel_in
  allocate(sims_particle_out(n_times))
  !> loop for initialising particles
  do ii=1,n_times
    call generate_particle_simulation_from_active_light_sources(lights_inout,&
    rng,fields,my_id,n_mpis,n_dead_particles,n_particles,n_spectra,accept_threshold,&
    masses(ii),times(ii),active_light_source_intensities(:,:,ii),signs_charge(ii),&
    sims_particle_out(ii),sign_p_parallel_in=signs_p_parallel(ii))
  enddo
end subroutine generate_particle_simulations_from_active_light_sources


!> Method used for generating a particle simulation from contributing and 
!> non contributing light sources
!> inputs:
!>   lights_inout:                    (light_vertices) class containing all active lights
!>   rng:                             (type_rng) random number generator
!>   fields:                          (fields_base) JOREK MHD fields
!>   my_id:                           (integer) id of the current mpi task
!>   n_mpis:                          (integer) number of mpi tasks
!>   n_dead_particles:                (integer) number of required dead particles
!>   n_particles:                     (integer) number of active lights
!>   n_spectra:                       (integer) number of spectra
!>   accept_threshold:                (real8) theshold for accepting non emitting lights
!>   mass:                            (real8) particle mass
!>   time:                            (real8) simulation time
!>   active_light_source_intensities: (real8)(n_spectra,n_particles) intensity of active lights
!>   sign_charge:                     (integer) sign of the particle charge
!>   sign_p_parallel_in:              (integer) optional sign of parallel momentum for
!>                                    relativistic guiding center particles, default: 1
!> outputs:
!>   sim_particle_out:                (type_particle_sim) initialised particle simulation
subroutine generate_particle_simulation_from_active_light_sources(lights_inout,&
rng,fields,my_id,n_mpis,n_dead_particles,n_particles,n_spectra,accept_threshold,&
mass,time,active_light_source_intensities,sign_charge,sim_particle_out,sign_p_parallel_in)
  use mod_rng
  use mod_random_seed
  use mod_fields,         only: fields_base
  use mod_particle_types, only: particle_base
  use mod_particle_types, only: particle_gc_relativistic
  use mod_particle_types, only: particle_kinetic_relativistic
  use mod_particle_sim,   only: particle_sim
  use mod_light_vertices, only: light_vertices
  use mod_full_synchrotron_light_dist_vertices,        only: full_synchrotron_light_dist
  use mod_gyroaverage_synchrotron_light_dist_vertices, only: gyroaverage_synchrotron_light_dist
  implicit none
  !> inputs-outputs:
  class(light_vertices),intent(inout) :: lights_inout
  class(type_rng),intent(inout)       :: rng
  !> inputs:
  class(fields_base),intent(in)                      :: fields
  integer,intent(in)                                 :: n_dead_particles,n_particles,n_mpis,my_id
  integer,intent(in)                                 :: n_spectra,sign_charge
  real*8,intent(in)                                  :: time,mass,accept_threshold
  real*8,dimension(n_spectra,n_particles),intent(in) :: active_light_source_intensities
  integer,intent(in),optional                        :: sign_p_parallel_in
  !> outputs:
  type(particle_sim),intent(out) :: sim_particle_out
  !> variables:
  class(particle_base),dimension(:,:),allocatable     :: particle_list
  integer                                             :: ii,jj,ifail,sign_p_parallel
  integer                                             :: n_sim_particles,counter
  real*8,dimension(n_particles)                       :: random_numbers
  logical,dimension(n_particles,lights_inout%n_times) :: check_particle_to_copy
  !> initialisation
  sign_p_parallel = 1; if(present(sign_p_parallel_in)) sign_p_parallel = sign_p_parallel_in;
  check_particle_to_copy = .false.; ifail = 0;
  !> generate random numbers
  call rng%initialize(n_particles,random_seed(),n_mpis,my_id,ifail)
  if(ifail.ne.0) call MPI_ABORT(MPI_COMM_WORLD,-1,ifail)  
  call rng%next(random_numbers)
  !> initialise particle simulation
  n_part_groups = 1
  call sim_particle_out%initialize(.true.,my_id,n_mpis,.false.)
  sim_particle_out%time           = time
  sim_particle_out%fields         = fields
  sim_particle_out%groups(:)%mass = mass
  !> initialise support particle list
  ifail = 1
  select type (light=>lights_inout)
    type is (full_synchrotron_light_dist)
      allocate(particle_kinetic_relativistic::particle_list(n_particles,lights_inout%n_times))
      ifail = 0
    type is (gyroaverage_synchrotron_light_dist)
      allocate(particle_gc_relativistic::particle_list(n_particles,lights_inout%n_times))
      ifail = 0
  end select
  if(ifail.ne.0) call MPI_ABORT(MPI_COMM_WORLD,-1,ifail)
  do jj=1,lights_inout%n_times 
    call initialise_particle_list(n_particles,particle_list(:,jj))
  enddo
  !> generate active particles and a fraction of the non active particles
  !$omp parallel do default(shared) private(ii,jj) &
  !$omp firstprivate(n_particles,accept_threshold,sign_p_parallel,sign_charge)
  do jj=1,lights_inout%n_times
    do ii=1,n_particles
      if(all(active_light_source_intensities(:,ii).eq.0d0).and.&
      (random_numbers(ii).gt.accept_threshold)) cycle
      call lights_inout%compute_particle_from_light(sim_particle_out%fields,&
      ii,jj,sim_particle_out%groups(1)%mass,particle_list(ii,jj))
      if(particle_list(ii,jj)%i_elm.le.0) cycle
      select type (p_out=>particle_list(ii,jj))
        type is (particle_kinetic_relativistic)
          p_out%q    = int(sign_charge,kind=1)*p_out%q
        type is (particle_gc_relativistic)
          p_out%q = int(sign_charge,kind=1)*p_out%q
          p_out%p(1) = real(sign_p_parallel,kind=8)*p_out%p(1)
      end select
      check_particle_to_copy(ii,jj) = .true.
    enddo
  enddo
  !$omp end parallel do  
  !> initialise new simulation particle list
  n_sim_particles = count(check_particle_to_copy)
  select type (p_list=>particle_list)
    type is (particle_kinetic_relativistic)
      allocate(particle_kinetic_relativistic::sim_particle_out%groups(1)%particles(&
      n_sim_particles+n_dead_particles))
    type is (particle_gc_relativistic)
      allocate(particle_gc_relativistic::sim_particle_out%groups(1)%particles(&
      n_sim_particles+n_dead_particles))
  end select
  call initialise_particle_list(n_sim_particles+n_dead_particles,&
  sim_particle_out%groups(1)%particles)
  !> store only selected particles
  counter = 1; check_particle_to_copy = .not.check_particle_to_copy;
  do jj=1,lights_inout%n_times
    do ii=1,n_particles
      if(check_particle_to_copy(ii,jj)) cycle
      sim_particle_out%groups(1)%particles(counter) = particle_list(ii,jj)
      counter = counter+1
    enddo
  enddo
  !> deallocate particles
  if(allocated(particle_list)) deallocate(particle_list)
end subroutine generate_particle_simulation_from_active_light_sources

!> initialise particle list
!> inputs:
!>   n_particles: 
!>   particle_list: (particle_base) list of particles to be initialised
!> outputs:
!>   particle_list: (particle_base) initialised particle list
subroutine initialise_particle_list(n_particles,particle_list)
  use mod_particle_types, only: particle_base
  use mod_particle_types, only: particle_gc_relativistic
  use mod_particle_types, only: particle_kinetic_relativistic
  implicit none
  !> inputs: 
  integer,intent(in) :: n_particles
  !> inputs-outputs:
  class(particle_base),dimension(n_particles),intent(inout) :: particle_list
  !> variables:
  integer :: ii
  !$omp parallel do default(shared) private(ii) firstprivate(n_particles)
  do ii=1,n_particles
    call initialise_particle_base(particle_list(ii))
    select type (p_inout=>particle_list(ii))
      type is (particle_kinetic_relativistic)
        p_inout%p = 0d0; p_inout%q = 0;
      type is (particle_gc_relativistic)
        p_inout%p = 0d0; p_inout%q =0;
    end select
  enddo
  !$omp end parallel do
end subroutine initialise_particle_list

!> initialise particle base
!> inputs
!>   particle: (particle_base) particle to be initialised
!> outputs:
!>   particle: (particle_base) initialised particle
subroutine initialise_particle_base(particle)
  use mod_particle_types, only: particle_base 
  implicit none
  !> inputs-outputs:
  class(particle_base),intent(inout) :: particle
  !> initialise particle base
  particle%x = 0d0; particle%st = 0d0; particle%i_elm = -1;
  particle%weight = -1d99; particle%i_life = 0; particle%t_birth = 0.; 
end subroutine initialise_particle_base

!> Method used for computing the radiation intensity emitted by 
!> a light and recived by a camera  
!> inputs:
!>   camera_inout:  (camera_static)  class defining a static camera
!>   lights_inout:  (light_vertices) class containing all active lights
!>   spectra_inout: (spectral_base)  camera spectrum class
!>   camera_id:     (integer) index of the camera vertex
!>   light_id:      (integer) index of the light vertex
!>   time_id:       (integer) index of the time to be treated 
!> outputs:
!>   i_pixel:       (integer)(2) indexes defining the illuminated pixel
!>   pixel_coords:  (real8)(2) coordinatres of the illuminated point
!>   intensity:     (real8)(n_spectra) light intensity for each spectral interval
subroutine compute_received_light_intensity(camera_inout,lights_inout,spectra_inout,&
camera_id,light_id,time_id,i_pixel,pixel_coords,intensity)
  use mod_spectra,        only: spectrum_base
  use mod_light_vertices, only: light_vertices
  use mod_camera_static,  only: camera_static
  implicit none
  !> implicit-none
  class(camera_static),intent(inout)  :: camera_inout
  class(light_vertices),intent(inout) :: lights_inout
  class(spectrum_base),intent(inout)  :: spectra_inout
  !> inputs:
  integer,intent(in) :: camera_id,light_id,time_id
  !> outputs
  integer,dimension(2),intent(out)                      :: i_pixel
  real*8,dimension(2),intent(out)                       :: pixel_coords 
  real*8,dimension(spectra_inout%n_spectra),intent(out) :: intensity
  !> variables
  logical :: intersect
  real*8  :: material_value,visibility_geometry
  real*8,dimension(3)                                              :: plane_line_coords
  real*8,dimension(spectra_inout%n_points,spectra_inout%n_spectra) :: spectral_irradiance
  !> initialisation
  i_pixel = -999; pixel_coords = -9d9; intensity = 0d0;
  !> compute the material function skip is <= 0
  call camera_inout%physical_material_funct(lights_inout%x(:,light_id,time_id),&
  camera_id,material_value)
  if(material_value.le.0.d0) return
  !> compute the intersection camera - light source, skip if not
  call camera_inout%find_ray_image_plane_intersection(&
  lights_inout%x(:,light_id,time_id),camera_id,intersect,plane_line_coords); 
  if(.not.intersect) return
  !> compute the intersection point pixel coordinates
  call camera_inout%plane_to_pixel_local_coord(plane_line_coords(1:2),&
  i_pixel,pixel_coords)
  !> compute the geometry and visibility functions
  call camera_inout%visibility_geometry_funct(lights_inout,1,time_id,&
  camera_id,light_id,visibility_geometry)
  !> compute the spectral irradiance
  call lights_inout%spectral_irradiance(spectra_inout,time_id,light_id,&
  camera_inout%x(:,camera_id,1),spectral_irradiance)
  !> integrate the spectral irradiance within each interval
  call spectra_inout%integrate_data(spectral_irradiance,intensity)
  !> compute the light intensity in each spectral interval
  intensity = intensity*material_value*visibility_geometry
end subroutine compute_received_light_intensity

!> write the source light positions and spectral contribution to hdf5 file.
!> inputs:
!>   filename:              (character) name of the hdf5 file to write in
!>   my_id:                 (integer) mpi task id
!>   n_x:                   (integer) number of position dimensions
!>   n_spectra:             (integer) number of spectral interval
!>   n_lights:              (integer) number of contributing lights
!>   n_lens_points:         (integer) number of points on lens
!>   light_positions:       (real8)(n_x,n_lights,n_lens_points) positions of
!>                          the light sources contributing to an image
!>   light_intensities:     (real8)(n_spectra,n_lights,n_lens_points)
!>                          integrated spectral intensities of 
!>                          the light sources contributing to an image
!> outputs:
!>   ierr: (integer) error variable
subroutine write_source_light_positions_contributions_in_hdf5(filename,my_id,&
n_x,n_spectra,n_lights,n_lens_points,light_positions,light_intensities,&
pixel_ids,pixel_coordinates,ierr)
  use hdf5
  use hdf5_io_module, only: HDF5_open_or_create,HDF5_close
  use hdf5_io_module, only: HDF5_array1D_saving_int,HDF5_array1D_saving
  use hdf5_io_module, only: HDF5_array3D_saving_int,HDF5_array3D_saving
  use phys_module,    only: n_limiter,R_limiter,Z_limiter
  implicit none
  !> intpus:
  character(len=*),intent(in) :: filename
  integer,intent(in)          :: my_id,n_x,n_spectra,n_lights,n_lens_points
  integer,dimension(2,n_lights,n_lens_points),intent(in)        :: pixel_ids
  real*8,dimension(n_x,n_lights,n_lens_points),intent(in)       :: light_positions
  real*8,dimension(n_spectra,n_lights,n_lens_points),intent(in) :: light_intensities
  real*8,dimension(2,n_lights,n_lens_points),intent(in)         :: pixel_coordinates
  !> outputs:
  integer,intent(out) :: ierr
  !> variables:
  integer :: file_out_len,n_my_id
  integer(HID_T) :: file_id
  character(len=10) :: format_char
  character(len=:),allocatable :: filename_out
  !> create the output filename
  n_my_id = int(log10(real(my_id)))+1
  if(my_id.eq.0) n_my_id = 1
  write(format_char,'(A,I1,A)') "(A,A,I",n_my_id,",A)"
  file_out_len = len(trim(filename))+1+n_my_id+3
  allocate(character(len=file_out_len)::filename_out)
  write(filename_out,trim(format_char)) filename,"_",my_id,".h5"
  !> open / close hdf5 file and save arrays in it
  call HDF5_open_or_create(trim(filename_out),file_id,ierr,file_access=H5F_ACC_TRUNC_F)
  call HDF5_array3D_saving(file_id,light_positions,n_x,n_lights,n_lens_points,'contributing_light_positions')
  call HDF5_array3D_saving(file_id,light_intensities,n_spectra,n_lights,n_lens_points,&
  'contributing_light_intensities')
  call HDF5_array3D_saving_int(file_id,pixel_ids,2,n_lights,n_lens_points,&
  'contributing_light_pixel_indexes')
  call HDF5_array3D_saving(file_id,pixel_coordinates,2,n_lights,n_lens_points,&
  'contributing_light_local_pixel_coordinates')
  call HDF5_array1D_saving(file_id,R_limiter,n_limiter,'limiter_major_radius')
  call HDF5_array1D_saving(file_id,Z_limiter,n_limiter,'limiter_vertical_coordinate')
  call HDF5_close(file_id)
  deallocate(filename_out)
end subroutine write_source_light_positions_contributions_in_hdf5

!> write the camera pinholes, planes and plane directions in hdf5 file
!> inputs:
!>   filename:     (character) name of the hdf5 file to write in
!>   my_id:        (integer) mpi task id
!>   camera_inout: (camera) camera with variables to be stored
!> outputs:
!>   camera_inout: (camera) camera with variables to be stored
!>   ierr:         (integer) error variable
subroutine write_pinholes_planes_directions_in_hdf5(filename,my_id,camera_inout,ierr)
  use hdf5
  use hdf5_io_module, only: HDF5_open_or_create,HDF5_close
  use hdf5_io_module, only: HDF5_array1D_saving_int
  use hdf5_io_module, only: HDF5_array2D_saving,HDF5_array3D_saving
  use mod_camera,     only: camera
  implicit none
  !> inputs-outputs:
  class(camera),intent(inout) :: camera_inout
  !> inputs:
  character(len=*),intent(in) :: filename
  integer,intent(in) :: my_id
  !> outputs:
  integer,intent(out) :: ierr
  !> variables:
  integer :: file_out_len,n_my_id
  integer(HID_T) :: file_id
  character(len=10) :: format_char
  character(len=:),allocatable :: filename_out
  !> create the output filename
  n_my_id = int(log10(real(my_id)))+1
  if(my_id.eq.0) n_my_id = 1
  write(format_char,'(A,I1,A)') "(A,A,I",n_my_id,",A)"
  file_out_len = len(trim(filename))+1+n_my_id+3
  allocate(character(len=file_out_len)::filename_out)
  write(filename_out,trim(format_char)) filename,"_",my_id,".h5"
  !> open / close hdf5 file and save arrays in it
  call HDF5_open_or_create(trim(filename_out),file_id,ierr,file_access=H5F_ACC_TRUNC_F)
  call HDF5_array1D_saving_int(file_id,camera_inout%n_pixels_spectra,3,'n_pixels_n_spectra')
  call HDF5_array3D_saving(file_id,camera_inout%x,camera_inout%n_x,camera_inout%n_vertices,camera_inout%n_times,'point_on_lens_positions')
  call HDF5_array3D_saving(file_id,camera_inout%image_plane,camera_inout%n_x,&
  camera_inout%n_plane_points,camera_inout%n_times,'image_plane_vertices')
  call HDF5_array2D_saving(file_id,camera_inout%image_plane_direction,camera_inout%n_x,camera_inout%n_times,'image_plane_directions')
  call HDF5_close(file_id)
  deallocate(filename_out)
end subroutine write_pinholes_planes_directions_in_hdf5
!> ---------------------------------------------------------------------------------------------------------
end program find_light_sources_contributing_to_image
