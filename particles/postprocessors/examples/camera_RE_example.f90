!> The camera_example.f90 contains an executable programme which
!> tests and shows how to run the synthetic camera diagnostic
!> for generating images of plasma radiation from a JOREK-particle
!> population. The test proposed below concerns the imaging of the
!> sychrotron radiation of a runaway electron population.
!> More specifically, we consider the JET tokamak configuration
!> having an IR and fast visible camera looking at the RE beam
!> from octant 5. Note that the RE toroidal motion
!> is counter-clockeise. Only one snapshot of the RE beam is 
!> considered hereafter hence, only one particle population and 
!> JOREK MHD fields are used.
program camera_RE_example
use constants,                      only: PI
use mod_mpi_tools,                  only: init_mpi_threads,finalize_mpi_threads
use particle_tracer
use mod_particle_io,                only: read_simulation_hdf5,get_simulation_hdf5_time
use mod_spectra_deterministic,      only: spectrum_integrator_2nd
use mod_pinhole_lens,               only: pinhole_lens
use mod_filter_unity,               only: filter_unity
use mod_camera_perspective_static,  only: camera_perspective_static
use mod_full_synchrotron_light_dist_vertices, only: full_synchrotron_light_dist
use phys_module
#ifdef USE_HDF5
use mod_fast_camera_io,             only: write_pixel_intensity_hdf5
#endif

implicit none

!> Variables -------------------------------------------------------------------------------
type(event)                        :: field_reader
type(pinhole_lens)                 :: lens
type(spectrum_integrator_2nd)      :: spectra
type(filter_unity)                 :: filter_image,filter_time
type(filter_unity),dimension(:),allocatable :: filter_spectra
type(camera_perspective_static)    :: camera
type(full_synchrotron_light_dist)  :: synch_sources
type(particle_sim),dimension(:),allocatable :: sims
integer                            :: ii,n_1d,n_2d,t0,t1
integer                            :: n_groups,my_id,n_mpis,n_x,ierr
integer                            :: n_wavelenghts,n_spectra
integer                            :: n_int_camera_param,n_real_camera_param
integer                            :: n_times
integer,dimension(:),allocatable   :: int_camera_param 
real*8,dimension(:),allocatable    :: min_spectra,max_spectra,pinhole_positions
real*8,dimension(:),allocatable    :: real_camera_param,sim_times
real*8,dimension(:,:),allocatable  :: x_pixel_positions,y_pixel_positions
real*8,dimension(:,:,:,:,:),allocatable :: pixel_filter_values
character(len=15)                  :: particle_filename
character(len=17)                  :: fields_filename
character(len=24)                  :: image_filename

!> Variable definitions -------------------------------------------------------------------
particle_filename = 'part_restart.h5'
fields_filename   = 'jorek_equilibrium'
image_filename    = 'pixel_filter_intensities'
n_1d = 1; n_2d = 2;
n_x = 3 !< number of spatial coordinates
n_groups = 1 !< number of particle groups
n_spectra     = 1
n_wavelenghts = 40
n_int_camera_param  = 5
n_real_camera_param = 9
n_times = 1
!> JET KDLT-E5WC wavelenght: 3d-6 - 3.5d-6 [m]
allocate(min_spectra(n_spectra)); min_spectra = [3d-6];
allocate(max_spectra(n_spectra)); max_spectra = [3.5d-6];
allocate(filter_spectra(n_spectra));
allocate(pinhole_positions(n_x)); pinhole_positions = [-8.86d-1,-4.002,-3.32d-1];
!> one pinhole => n_lens_samples=1, JET KLDT-E5WC pixels nx=120,ny=176
allocate(int_camera_param(n_int_camera_param)); int_camera_param = [1,120,176,0,1];
allocate(real_camera_param(n_real_camera_param));
allocate(x_pixel_positions(int_camera_param(2),n_times))
allocate(y_pixel_positions(int_camera_param(3),n_times))
!> JET KLDT-E5WC camera inputs:
!> 1:3 -> half width, half height and orientation of the visual plane angle
!> 4:6 -> camera focal direction: focal distance, latitude, azimuth
!> 7:9 -> camera position in the tokamak reference system
real_camera_param = [5.23d-1,5.23d-1,5d-1*PI,9.998025d-1,1.5807985,2.09801,-8.86d-1,-4.002,-3.32d-1];
allocate(sim_times(n_times)); allocate(sims(n_times));

!> Initialisation  ------------------------------------------------------------------------
!> Initialise MPI communicator
call init_mpi_threads(my_id,n_mpis,ierr)

!> set the number of particle groups
n_part_groups = n_groups

!> read particle, time and MHD fields data
write(*,*) 'Reading particle data ...'
field_reader = event(read_jorek_fields_interp_linear(basename=trim(fields_filename),i=-1))
do ii=1,n_times
  call sims(ii)%initialize(.true.,my_id,n_mpis)
  call read_simulation_hdf5(sims(ii),particle_filename)
  sim_times(ii) = get_simulation_hdf5_time(particle_filename)
  call with(sims(ii),field_reader)
enddo
write(*,*) 'Reading particle data: completed!'

!> Initialise synthetic diagnostics
write(*,*) 'Initialise synthetic camera and light sources ...'
call system_clock(t0)
spectra = spectrum_integrator_2nd(n_wavelenghts,n_spectra,min_spectra,max_spectra)
call spectra%generate_spectrum()
call filter_image%init_filter(n_2d)
do ii=1,n_spectra
  call filter_spectra(ii)%init_filter(n_1d)
enddo
call filter_time%init_filter(n_1d)
call lens%init_pinhole(n_x,pinhole_positions)
call camera%init_camera(lens,spectra,n_int_camera_param,n_real_camera_param,&
int_camera_param,real_camera_param)
call synch_sources%init_lights_from_particles(n_times,sims)
call camera%compute_scaled_pixel_positions_on_plane(x_pixel_positions,y_pixel_positions)
call system_clock(t1)
write(*,*) 'Initialise synthetic camera and light sources: completed'
write(*,*) my_id, 'System time fast camera initialisation (s): ',real(t1-t0,kind=8)/1d3

!> allocate image and filter arrays
allocate(pixel_filter_values(n_spectra,2,int_camera_param(2),int_camera_param(3),n_times))

!> Compute image --------------------------------------------------------------------------
write(*,*) 'Computing image and filters per each time ...'
call system_clock(t0)
call camera%reduce_light_image(synch_sources,spectra,filter_image,filter_spectra,&
filter_time,my_id,pixel_filter_values,ierr)
call system_clock(t1)
write(*,*) 'Computing image and filters per each time: completed!'
write(*,*) my_id, 'System time computing image and filters (s): ',real(t1-t0,kind=8)/1d3

!> Write image ----------------------------------------------------------------------------
#ifdef USE_HDF5
if(my_id.eq.0) then
  write(*,*) 'Write image and filters in HDF5 file ...'
  call write_pixel_intensity_hdf5(image_filename,n_spectra,2,int_camera_param(2),&
  int_camera_param(3),n_times,x_pixel_positions,y_pixel_positions,pixel_filter_values,ierr)
  write(*,*) 'Write image and filters in HDF5 file: completed!'
endif
#endif

!> Finalisation ---------------------------------------------------------------------------
if(allocated(min_spectra))         deallocate(min_spectra)
if(allocated(max_spectra))         deallocate(max_spectra)
if(allocated(filter_spectra))      deallocate(filter_spectra)
if(allocated(pinhole_positions))   deallocate(pinhole_positions)
if(allocated(int_camera_param))    deallocate(int_camera_param)
if(allocated(real_camera_param))   deallocate(real_camera_param)
if(allocated(sim_times))           deallocate(sim_times)
if(allocated(sims))                deallocate(sims)
if(allocated(x_pixel_positions))   deallocate(x_pixel_positions)
if(allocated(y_pixel_positions))   deallocate(y_pixel_positions)
if(allocated(pixel_filter_values)) deallocate(pixel_filter_values)
call finalize_mpi_threads(ierr)

end program camera_RE_example
