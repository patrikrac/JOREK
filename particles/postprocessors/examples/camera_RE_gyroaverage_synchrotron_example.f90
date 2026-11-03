!> The camera_kinetic_RE_gyroaverage_synchrotron_example.f90 
!> contains an executable programme which tests and shows how 
!> to run the synthetic camera diagnostic for generating images 
!> of plasma radiation from a JOREK-particle population. 
!> The test proposed below concerns the imaging of the
!> sychrotron radiation of a runaway electron population.
!> More specifically, we consider the JET tokamak configuration
!> having an IR and fast visible camera looking at the RE beam
!> from octant 5. Note that the RE toroidal motion
!> is counter-clockeise. Only one snapshot of the RE beam is 
!> considered hereafter hence, only one particle population and 
!> JOREK MHD fields are used. The program uses kinetic relativistic
!> or relativistitc gc particles. Relativistic kinetic particles are
!> transformed to relativistic gc for running the gyroaverage 
!> synchrotron radiation model.
program camera_RE_gyroaverage_synchrotron_example
use constants,                      only: PI
use mod_mpi_tools,                  only: init_mpi_threads,finalize_mpi_threads
use particle_tracer
use mod_spectra_deterministic,      only: spectrum_integrator_2nd
use mod_pinhole_lens,               only: pinhole_lens
use mod_filter_unity,               only: filter_unity
use mod_camera_perspective_static,  only: camera_perspective_static
use mod_gyroaverage_synchrotron_light_dist_vertices, only: gyroaverage_synchrotron_light_dist
#ifdef USE_HDF5
use mod_fast_camera_io,             only: write_pixel_intensity_hdf5
use phys_module
#endif

implicit none

!> Variables -------------------------------------------------------------------------------
type(event)                        :: field_reader,particle_reader
type(pinhole_lens)                 :: lens
type(spectrum_integrator_2nd)      :: spectra
type(filter_unity)                 :: filter_image,filter_time
type(filter_unity),dimension(:),allocatable :: filter_spectra
type(camera_perspective_static)          :: camera
type(gyroaverage_synchrotron_light_dist) :: synch_sources
type(particle_sim),dimension(:),allocatable :: sims,sims_gc
logical                            :: write_gc_in_txt,do_jorek_init
integer                            :: ii,n_1d,n_2d,t0,t1
integer                            :: n_groups,my_id,n_mpis,n_x,ierr
integer                            :: n_wavelengths,n_spectra
integer                            :: n_int_camera_param,n_real_camera_param
integer                            :: n_times,n_frames
integer,dimension(:),allocatable   :: int_camera_param 
real*8,dimension(:),allocatable    :: min_spectra,max_spectra,pinhole_positions
real*8,dimension(:),allocatable    :: real_camera_param,sim_times
real*8,dimension(:,:),allocatable :: x_pixel_positions,y_pixel_positions
real*8,dimension(:,:,:,:,:),allocatable :: pixel_filter_values
character(len=3)                   :: hdf5ext
character(len=4)                   :: extension
character(len=17)                  :: fields_filename
character(len=24)                  :: image_filename
character(len=27)                  :: filename_gc_txt_root
character(len=33)                  :: filename_gc_txt
character(len=60),dimension(:),allocatable :: particle_filenames

!> Variable presets -----------------------------------------------------------------------
extension = '.txt'; hdf5ext = '.h5'
n_1d = 1; n_2d = 2; n_x = 3 !< number of spatial coordinates

!> Variable definitions -------------------------------------------------------------------
n_frames             = 1
n_times              = 11
fields_filename      = 'jorek01610'
image_filename       = 'pixel_filter_intensities'
filename_gc_txt_root = 'jorek_relativistic_gc_data_'
n_groups = 1 !< number of particle groups
n_spectra     = 1
n_wavelengths = 40
n_int_camera_param  = 5
n_real_camera_param = 9
write_gc_in_txt = .false.
!> se the list of particle restart files to be read
allocate(character(len=60)::particle_filenames(n_times)); particle_filenames = '';
particle_filenames = [character(len=60)::'part_restart000.00339941',&
'part_restart000.00349941','part_restart000.00359941',&
'part_restart000.00369941','part_restart000.00379941',&
'part_restart000.00389941','part_restart000.00399941',&
'part_restart000.00409941','part_restart000.00419941',&
'part_restart000.00429941','part_restart000.00439941']
!> JET KDLT-E5WC wavelength: 3d-6 - 3.5d-6 [m]
allocate(min_spectra(n_spectra)); min_spectra = [3d-6];
allocate(max_spectra(n_spectra)); max_spectra = [3.5d-6];
allocate(pinhole_positions(n_x)); pinhole_positions = [-8.86d-1,-4.002,-3.32d-1];
!> one pinhole => n_lens_samples=1, JET KLDT-E5WC pixels nx=120,ny=176
allocate(int_camera_param(n_int_camera_param)); int_camera_param = [1,600,600,0,1];
!> JET KLDT-E5WC camera inputs:
!> 1:3 -> half width, half height and orientation of the visual plane angle
!> 4:6 -> camera focal direction: focal distance, latitude, azimuth
!> 7:9 -> camera position in the tokamak reference system
allocate(real_camera_param(n_real_camera_param))
real_camera_param = [5.23d-1,5.23d-1,5d-1*PI,9.998025d-1,1.5807965,2.09801,-8.86d-1,-4.002,-3.32d-1];

!> Initialisation  ------------------------------------------------------------------------
!> Initialise MPI communicator
call init_mpi_threads(my_id,n_mpis,ierr)
!> set the number of particle groups
n_part_groups = n_groups

!> allocate structures, read particle, time, MHD fields data and transform to gc if needed
write(*,*) 'Reading particle data ...'
allocate(filter_spectra(n_spectra));
allocate(x_pixel_positions(int_camera_param(2),n_frames))
allocate(y_pixel_positions(int_camera_param(3),n_frames))
allocate(sim_times(n_times)); allocate(sims(n_times)); 
allocate(sims_gc(n_times)); do_jorek_init = .true.
field_reader = event(read_jorek_fields_interp_linear(basename=trim(fields_filename),i=-1))
do ii=1,n_times
  call sims(ii)%initialize(.true.,my_id,n_mpis,do_jorek_init)
  write(*,*) 'my_id: ',sims(ii)%my_id,'doing particle restart file: ',trim(particle_filenames(ii))
  particle_reader = event(read_action(filename=trim(particle_filenames(ii))//trim(hdf5ext)))
  call with(sims(ii),particle_reader); sim_times(ii) = sims(ii)%time; 
  call with(sims_gc(ii),field_reader)
  !> transform particle kinetic relativistic to relativistic gc or store relativistic gc
  call relativistic_kinetic_sim_to_relativistic_gc_sim(sims(ii),sims_gc(ii))
  if(do_jorek_init) do_jorek_init = .false.
enddo
if(allocated(particle_filenames)) deallocate(particle_filenames)
if(allocated(sims))               deallocate(sims)
write(*,*) 'Reading particle data: completed!' 

if(write_gc_in_txt) then
  write(*,*) 'Write particle data in txt file'
  write(filename_gc_txt,'(A,I2,A)') filename_gc_txt_root,my_id,extension
  call dump_relativistic_gc_in_txt(filename_gc_txt,sims_gc)
  write(*,*) 'Write particle data in txt file: completed!'
endif

!> Initialise synthetic diagnostics
write(*,*) 'Initialise synthetic camera and light sources ...'
call system_clock(t0)
spectra = spectrum_integrator_2nd(n_wavelengths,n_spectra,min_spectra,max_spectra)
call spectra%generate_spectrum()
call filter_image%init_filter(n_2d)
do ii=1,n_spectra
  call filter_spectra(ii)%init_filter(n_1d)
enddo
call filter_time%init_filter(n_1d)
call lens%init_pinhole(n_x,pinhole_positions)
call camera%init_camera(lens,spectra,n_int_camera_param,n_real_camera_param,&
int_camera_param,real_camera_param)
call synch_sources%init_lights_from_particles(n_times,sims_gc)
call camera%compute_scaled_pixel_positions_on_plane(x_pixel_positions,y_pixel_positions) 
call system_clock(t1)
write(*,*) 'Initialise synthetic camera and light sources: completed'
write(*,*) my_id, 'System time fast camera initialisation (s): ',real(t1-t0,kind=8)/1d3

!> allocate image and filter arrays
allocate(pixel_filter_values(n_spectra,2,int_camera_param(2),int_camera_param(3),n_frames))

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
  int_camera_param(3),n_frames,x_pixel_positions,y_pixel_positions,pixel_filter_values,ierr)
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
if(allocated(sims_gc))             deallocate(sims_gc)
if(allocated(x_pixel_positions))   deallocate(x_pixel_positions)
if(allocated(y_pixel_positions))   deallocate(y_pixel_positions)
if(allocated(pixel_filter_values)) deallocate(pixel_filter_values)
call finalize_mpi_threads(ierr)

contains

!> Tools ----------------------------------------------------------------------------------

!> Tools for transforming a relativistic kinetic particle simulation
!> in a relativistic gc simulations or copy the relativistic gc particle
!> in the new simulation. The JOREK MHD fields must be loaded in 
!> the relativistic gc simulation.
!> inputs:
!>   sim_kin: (particle_sim) kinetic relativistic/gc particle simulation
!>   sim_gc:  (particle_sim) kinetic gc simulation
!> outputs:
!>   sim_kin: (particle_sim) kinetic relativistic/gc particle simulation
!>   sim_gc:  (particle_sim) kinetic gc simulation 
subroutine relativistic_kinetic_sim_to_relativistic_gc_sim(sim_kin,sim_gc) 
  implicit none
  !> Inputs-Outputs:
  type(particle_sim),intent(inout) :: sim_kin
  type(particle_sim),intent(inout) :: sim_gc
  !> Variables
  integer :: ii,jj,counter,n_particles
  real*8  :: U,psi
  real*8,dimension(3) :: E_field,B_field
  !> copy base data from sim_kin to sim_gc
  sim_gc%time   = sim_kin%time; sim_gc%stop_now = sim_kin%stop_now;
  sim_gc%t_norm = sim_kin%t_norm; sim_gc%my_id  = sim_kin%my_id;
  sim_gc%n_mpi  = sim_kin%n_mpi; sim_gc%wtime_start = sim_kin%wtime_start;
  if(.not.allocated(sim_gc%groups)) allocate(sim_gc%groups(n_groups))
  counter = 1
  do jj=1,size(sim_kin%groups)
  select type (p_list=>sim_kin%groups(jj)%particles)
    type is (particle_kinetic_relativistic)
    sim_gc%groups(counter)%Z    = sim_kin%groups(jj)%Z
    sim_gc%groups(counter)%mass = sim_kin%groups(jj)%mass
    sim_gc%groups(counter)%dt   = sim_kin%groups(jj)%dt
    n_particles = size(p_list)
    allocate(particle_gc_relativistic::sim_gc%groups(counter)%particles(n_particles))
    do ii=1,n_particles
      call sim_gc%fields%calc_EBpsiU(sim_gc%time,p_list(ii)%i_elm,&
      p_list(ii)%st,p_list(ii)%x(3),E_field,B_field,psi,U)
      call relativistic_kinetic_to_particle(sim_gc%fields%node_list,&
      sim_gc%fields%element_list,p_list(ii),&
      sim_gc%groups(counter)%particles(ii),sim_gc%groups(counter)%mass,&
      B_field)
    enddo
    counter = counter + 1
  type is (particle_gc_relativistic)
    sim_gc%groups(counter)%Z    = sim_kin%groups(jj)%Z
    sim_gc%groups(counter)%mass = sim_kin%groups(jj)%mass
    sim_gc%groups(counter)%dt   = sim_kin%groups(jj)%dt
    n_particles = size(p_list)
    allocate(particle_gc_relativistic::sim_gc%groups(counter)%particles(n_particles))   
    select type (p_list_2=>sim_gc%groups(counter)%particles)
    type is (particle_gc_relativistic)
      p_list_2(:) = p_list(:)
    end select
    counter = counter + 1
  end select
  enddo
end subroutine relativistic_kinetic_sim_to_relativistic_gc_sim

!> dump gc particles in a txt file with name filename
!> sequence of the writint data:
!> 1:R, 2:Z, 3:phi, 4:parallel momentum, 5:magnetic moment, 
!> 6:weight 7:total momentum 8:pitch angle, 9:mass, 10:charge
!> inputs:
!>   filename: (character) name of the file 
!>   sims_gc:  (particle_sim)(n_times) gc simulation
!> outputs:
!>   sim_gc:   (particle_sim) gc simulation
subroutine dump_relativistic_gc_in_txt(filename,sims_gc)
  implicit none
  !> Inputs-Outputs:
  type(particle_sim),dimension(:),allocatable,intent(inout) :: sims_gc
  !> Inputs:
  character(len=*),intent(in) :: filename
  !> Variables
  integer             :: ii,jj,kk,ifail
  real*8              :: momentum_tot,momentum_perp_2
  real*8              :: pitch_angle,normB,psi,U
  real*8,dimension(3) :: B_field,E_field
  !> open the file
  open(unit=43,file=trim(filename),action='write',blank='NULL',&
  form='formatted',status='unknown',iostat=ifail)
  !> write gc data into txt
  do kk=1,size(sims_gc)
    do jj=1,size(sims_gc(kk)%groups)
      select type (p_list=>sims_gc(kk)%groups(jj)%particles)
        type is (particle_gc_relativistic)
        do ii=1,size(p_list)
          call sims_gc(kk)%fields%calc_EBpsiU(sims_gc(kk)%time,&
          p_list(ii)%i_elm,p_list(ii)%st,p_list(ii)%x(3),E_field,&
          B_field,psi,U); normB = norm2(B_field);
          !> compute the squared perpendicular momentum and total momentum
          momentum_perp_2 = 2d0*p_list(ii)%p(2)*sims_gc(kk)%groups(jj)%mass*normB
          momentum_tot    = sqrt(p_list(ii)%p(1)**2 + momentum_perp_2)
          !> compute the pitch angle
          pitch_angle     = (p_list(ii)%p(2)/abs(p_list(ii)%p(2)))*sqrt(momentum_perp_2)
          pitch_angle     = atan2(pitch_angle,p_list(ii)%p(1))
          if(pitch_angle.lt.0d0) pitch_angle = TWOPI+pitch_angle
          !> write data in txt
          write(43,'(10E40.16E4)') p_list(ii)%x(1),p_list(ii)%x(2),p_list(ii)%x(3),&
          p_list(ii)%p(1),p_list(ii)%p(2),p_list(ii)%weight,momentum_tot,&
          pitch_angle,sims_gc(kk)%groups(jj)%mass,real(p_list(ii)%q,kind=8)
        enddo
      end select
    enddo
  enddo
  !> close the file
  close(43)
end subroutine dump_relativistic_gc_in_txt

end program camera_RE_gyroaverage_synchrotron_example
