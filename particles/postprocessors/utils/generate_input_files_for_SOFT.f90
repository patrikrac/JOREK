!> Generate magnetic equilibrium and distribution function
!> files compatible with the SOFT code inputs from the 
!> JOREK MHD solutions and distribution functions.
program generate_input_files_for_SOFT
use constants,      only: PI,SPEED_OF_LIGHT,ATOMIC_MASS_UNIT,EL_CHG
use phys_module,    only: n_limiter,R_limiter,Z_limiter,xcase,xpoint
use phys_module,    only: n_boundary,R_boundary,Z_boundary,xcase,xpoint
use data_structure, only: type_bnd_node_list,type_bnd_element_list
use data_structure, only: type_surface_list
use mod_mpi_tools,  only: init_mpi_threads,finalize_mpi_threads
use mod_boundary,   only: boundary_from_grid
use equil_info,     only: ES,update_equil_state
use particle_tracer

implicit none

!> Variables --------------------------------------------------------------------------------------------
type(event)                 :: field_reader
type(type_surface_list)     :: flux_surfaces
type(type_bnd_node_list)    :: bnd_node_list
type(type_bnd_element_list) :: bnd_elm_list
logical                     :: write_wall,use_boundary_lcfs
integer                     :: my_id,n_mpis,ierr,i_elm_axis,ii,n_points_lcfs
integer                     :: n_vec,n_R,n_Z,n_R_loc,n_momenta,n_pitch
integer                     :: n_flux,n_flux_loc,errorcode,charge,n_LCFS
real*8                      :: time,tor_angle,psi_axis,flux_axis_tol,mass
real*8,dimension(2)                 :: RZ_axis,st_axis,Ekin_bnd,pitch_bnd
real*8,dimension(2)                 :: momentum_bnd
real*8,dimension(:),allocatable     :: R_mesh,Z_mesh,flux_mesh
real*8,dimension(:),allocatable     :: flux_minor_radii_Zmag_LFS
real*8,dimension(:),allocatable     :: momentum_mesh,pitch_mesh
real*8,dimension(:),allocatable     :: cospitch_mesh
real*8,dimension(:,:),allocatable   :: poloidal_flux,RZ_LCFS
real*8,dimension(:,:,:),allocatable :: magnetic_field,pdf
character(len=17) :: fields_filename
character(len=28) :: magnetic_field_filename
character(len=17) :: pdf_filename
character(len=63) :: soft_mag_name
character(len=78) :: soft_desc
!> Variables definitions --------------------------------------------------------------------------------
write_wall        = .false. !< write the tokamak wall in soft hdf5
use_boundary_lcfs = .true.  !< use the plasma boundary as separatrix
!> if true the order of the pitch mesh and of the distribution function along
n_vec           = 3      !< number of vector components
n_R             = 202    !< total number of radial points
n_Z             = 201    !< total number of vertical coordinate points
n_flux          = 100    !< number of flux surfaces / minor radii
n_momenta       = 101    !< number of nodes of the momentum mesh
n_pitch         = 101    !< number of nodes of the pitch angle mesh
charge          = -1     !< electron charge w.r.t. proton charge
n_points_lcfs   = 5      !< number of separatrix point per separatrix segment
mass            = 5.48579909065d-4 !< electron mass in AMU
time            = 0d0  !< simulation time
tor_angle       = 0d0  !< toroidal angle
Ekin_bnd        = [2d7-1d4,2d7+1d4] !< kinetic energy lower and upper bound
pitch_bnd       = [PI-2.95d-1,PI]   !< PI-pitch angle lower and upper bound
flux_axis_tol   = 0d0  !< tolerance of the poloidal flux axis for mesh generation
fields_filename         = 'jorek_equilibrium'            !< jorek restart filename
magnetic_field_filename = 'magnetic_field_jorek_to_soft' !< soft magnetic field input from jorek
pdf_filename            = 'pdf_jorek_to_soft'            !< soft distribution field input from jorek
!> name and description of the soft field
soft_mag_name = 'jorek_circular_equilibrium_hollow_current_profile_magnetic_data' 
soft_desc     = 'pulse95135_press0_parabolicq_q95_6dot8_res1r5dot88m5_res2r4dot705m4_Ip612en1MA'
!> Initialisation ---------------------------------------------------------------------------------------
!> initialise the MPI communicator
call init_mpi_threads(my_id,n_mpis,ierr)
!> read MHD fields
write(*,*) "Reading MHD data ..."
call sim%initialize(.true.,my_id,n_mpis)
field_reader = event(read_jorek_fields_interp_linear(basename=trim(fields_filename),i=-1))
call with(sim,field_reader)
write(*,*) "Reading MHD data: completed!"
!> initialise magnetic field array
write(*,*) "Initialise magnetic field computation ..."
!> allocate R,Z mesh and magnetic field arrays
n_R_loc = n_R/n_mpis; n_R_loc = max(n_R_loc,n_R - (n_mpis-1)*n_R_loc); n_R = n_R_loc*n_mpis;
allocate(R_mesh(n_R)); R_mesh = 0d0; allocate(Z_mesh(n_Z)); Z_mesh = 0d0;
allocate(poloidal_flux(n_Z,n_R_loc)); poloidal_flux = 0d0;
allocate(magnetic_field(n_vec,n_Z,n_R_loc)); magnetic_field = 0d0;
write(*,*) "Initialise magnetic field computation: completed!"

write(*,*) "Initialise distribution function computation ..."
!> update equilibrium state
if(my_id.eq.0) call boundary_from_grid(sim%fields%node_list,sim%fields%element_list,&
bnd_node_list,bnd_elm_list,.false.)
call broadcast_boundary(my_id,bnd_elm_list,bnd_node_list)
call update_equil_state(my_id,sim%fields%node_list,sim%fields%element_list,bnd_elm_list,xpoint,xcase)
!> define flux surfaces
n_flux_loc = n_flux/n_mpis; n_flux_loc = max(n_flux - (n_mpis-1)*n_flux_loc,n_flux_loc);
n_flux = n_flux_loc*n_mpis; allocate(flux_minor_radii_Zmag_LFS(n_flux_loc));
flux_minor_radii_Zmag_LFS = 0d0; flux_surfaces%n_psi = n_flux_loc; 
allocate(flux_surfaces%psi_values(flux_surfaces%n_psi));
!> Find the separatrix coordinates
call find_axis(my_id,sim%fields%node_list,sim%fields%element_list,psi_axis,RZ_axis(1),RZ_axis(2),&
i_elm_axis,st_axis(1),st_axis(2),ierr) !< compute the magnetic axis position
if(use_boundary_lcfs) then 
  !> use the plasma boundary as separatrix
  n_LCFS = n_boundary; allocate(RZ_LCFS(2,n_LCFS)); 
  RZ_LCFS(1,:) = R_boundary(1:n_LCFS); RZ_LCFS(2,:) = Z_boundary(1:n_LCFS);
else
  !> find the separatrix
  call find_and_compute_separatrix(sim%fields,my_id,xpoint,xcase,n_points_lcfs,&
  ES%psi_bnd,RZ_axis,n_LCFS,RZ_LCFS)
endif
!> allocate and initialise momentum mesh, pitch angle mesh and particle distribution
momentum_bnd = mass*SPEED_OF_LIGHT*sqrt(((EL_CHG*Ekin_bnd/(ATOMIC_MASS_UNIT*mass*SPEED_OF_LIGHT**2))+1.d0)**2-1.d0)
allocate(momentum_mesh(n_momenta)); call generate_equidistant_mesh_1d(n_momenta,momentum_bnd,momentum_mesh);
allocate(pitch_mesh(n_pitch)); call generate_equidistant_mesh_1d(n_pitch,pitch_bnd,pitch_mesh);
!> revert the pitch mesh for having the cosinues varibale always increasing
pitch_mesh = cos(pitch_mesh); allocate(cospitch_mesh(n_pitch));
cospitch_mesh = [(pitch_mesh(n_pitch-ii+1),ii=1,n_pitch)]; deallocate(pitch_mesh);
allocate(pdf(n_momenta,n_pitch,flux_surfaces%n_psi));
write(*,*) "Momentum interval normalised to mass*speed_of_light: ",momentum_bnd/(mass*SPEED_OF_LIGHT)
write(*,*) "Pitch angle interval: ",pitch_bnd
write(*,*) "Initialise distribution function computation: completed!"

!> Compute and write soft inputs ------------------------------------------------------------------------
write(*,*) "Generate computational mesh ..."
call generate_equidistant_RZ_mesh(sim%fields,n_R,n_Z,R_mesh,Z_mesh); allocate(flux_mesh(n_flux));
call generate_equidistant_poloidal_flux_mesh(n_flux,ES%psi_axis,ES%psi_bnd,flux_axis_tol,flux_mesh)
flux_surfaces%psi_values = flux_mesh(n_flux_loc*my_id+1:n_flux_loc*(my_id+1)); deallocate(flux_mesh); 
call define_flux_surfaces(my_id,xcase,xpoint,i_elm_axis,psi_axis,st_axis,sim%fields,flux_surfaces)
write(*,*) "Generate computational mesh: completed!"

!> generate magnetic field inputs
write(*,*) "Compute and write the magnetic field ..."
call compute_magnetic_field_poloidal_flux(sim%fields,n_vec,n_R_loc,n_Z,time,tor_angle,&
R_mesh(my_id*n_R_loc+1:(my_id+1)*n_R_loc),Z_mesh,ES%psi_bnd,magnetic_field,poloidal_flux)
call write_SOFT_magnetic_field_file(magnetic_field_filename,sim%fields,write_wall,n_vec,&
n_R_loc,n_Z,n_limiter,n_LCFS,n_mpis,my_id,RZ_axis,R_mesh,Z_mesh,R_limiter,Z_limiter,&
RZ_LCFS,magnetic_field,poloidal_flux,soft_mag_name,soft_desc)
write(*,*) "Compute and write the magnetic field: completed!"

!> generate distribution function inputs
write(*,*) "Compute and write distribution function ..."
call find_LFS_minor_radius_flux_surface(sim%fields,flux_surfaces,psi_axis,RZ_axis,flux_minor_radii_Zmag_LFS);
call soft_current_density_uniform_phase_pdf(sim%fields,flux_surfaces,n_momenta,&
n_pitch,charge,mass,momentum_mesh,cospitch_mesh,pdf)
!> write distribution in soft input file the momentum mesh and the pdf are
!> normalised w.r.t. mass*SPEED_OF_LIGHT
call write_soft_distribution_function(pdf_filename,my_id,n_mpis,flux_surfaces%n_psi,&
n_momenta,n_pitch,flux_minor_radii_Zmag_LFS,momentum_mesh/(mass*SPEED_OF_LIGHT),&
cospitch_mesh,pdf*((mass*SPEED_OF_LIGHT)**3))
write(*,*) "Compute and write distribution function: completed!"

!> Finalisation -----------------------------------------------------------------------------------------
if(allocated(R_mesh))                    deallocate(R_mesh);
if(allocated(Z_mesh))                    deallocate(Z_mesh);
if(allocated(poloidal_flux))             deallocate(poloidal_flux);
if(allocated(magnetic_field))            deallocate(magnetic_field);
if(allocated(flux_surfaces%psi_values))  deallocate(flux_surfaces%psi_values);
if(allocated(flux_minor_radii_Zmag_LFS)) deallocate(flux_minor_radii_Zmag_LFS);
if(allocated(RZ_LCFS))                   deallocate(RZ_LCFS);
if(allocated(momentum_mesh))             deallocate(momentum_mesh);
if(allocated(pitch_mesh))                deallocate(pitch_mesh);
if(allocated(cospitch_mesh))             deallocate(cospitch_mesh);
if(allocated(pdf))                       deallocate(pdf);
call finalize_mpi_threads(ierr)

contains

!> Tools ------------------------------------------------------------------------------------------------
!> generate squared spaced mesh
!> inputs:
!>   n_nodes: (integer) number of nodes composing the mesh
!>   bounds:  (real8)(2) 1-lower and 2-upper mesh bound
!> outputs:
!>   mesh: (real8)(n_nodes) squared mesh with bounds
subroutine generate_squared_mesh_1d(n_nodes,bounds,mesh)
  use mpi
  implicit none
  !> inputs:
  integer,intent(in)             :: n_nodes
  real*8,dimension(2),intent(in) :: bounds
  !> outpus:
  real*8,dimension(n_nodes),intent(out) :: mesh
  !> variables:
  integer :: errorcode,ierr
  !> generate square root mesh
  call generate_equidistant_mesh_1d(n_nodes,[0d0,1d0],mesh)
  mesh = bounds(1) + (bounds(2)-bounds(1))*mesh**2
end subroutine generate_squared_mesh_1d
!> generate equidistant mesh generic coordinates
!> inputs:
!>   n_nodes: (integer) number of nodes composing the mesh
!>   bounds:  (real8)(2) 1-lower and 2-upper mesh bounds
!> outputs
!>   mesh: (real8)(n_nodes) mesh having lower and upper bounds
!>         equal to bounds
subroutine generate_equidistant_mesh_1d(n_nodes,bounds,mesh)
  implicit none
  !> inputs:
  integer,intent(in)             :: n_nodes
  real*8,dimension(2),intent(in) :: bounds
  !> outpus:
  real*8,dimension(n_nodes),intent(out) :: mesh
  !> variables:
  integer :: ii
  real*8  :: delta
  !> generate the mesh
  delta = (bounds(2)-bounds(1))/real(n_nodes-1,kind=8)
  mesh = [(bounds(1)+real(ii,kind=8)*delta,ii=0,n_nodes-1)]
end subroutine generate_equidistant_mesh_1d

!> generate equidistant mesh in R and Z coordinates
!> inputs:
!>   fields: (fields_base) jorek mhd fields datatype 
!>   n_R:    (integer) number of major radius nodes
!>   n_Z:    (integer) number of vertical coordinate nodes
!> outputs:
!>   R_mesh: (real8)(n_R) major radius mesh
!>   Z_mesh: (real8)(n_Z) vertical coordinate mesh
subroutine generate_equidistant_RZ_mesh(fields,n_R,n_Z,R_mesh,Z_mesh)
  use mod_fields, only: fields_base
  implicit none
  !> inputs:
  class(fields_base),intent(in) :: fields
  integer,intent(in)            :: n_R,n_Z
  !> outputs: 
  real*8,dimension(n_R),intent(out) :: R_mesh
  real*8,dimension(n_Z),intent(out) :: Z_mesh
  !> variables:
  integer :: ii
  real*8  :: dR,dZ
  !> define the domain bounding box
  call domain_bounding_box(fields%node_list,fields%element_list,R_mesh(1),R_mesh(n_R),Z_mesh(1),Z_mesh(n_Z))
  dR = (R_mesh(n_R)-R_mesh(1))/real(n_R-1,kind=8); dZ = (Z_mesh(n_Z)-Z_mesh(1))/real(n_Z-1,kind=8);
  !> generate mesh
  R_mesh = [(R_mesh(1)+real(ii,kind=8)*dR,ii=0,n_R-1)]
  Z_mesh = [(Z_mesh(1)+real(ii,kind=8)*dZ,ii=0,n_Z-1)]
end subroutine generate_equidistant_RZ_mesh

!> generate squared mesh in poloidal flux
!> inputs:
!>   n_psi:        (integer) number of poloidal flux points
!>   psi_axis:     (real8) poloidal flux at the magnetic axis
!>   psi_bnd:      (real8) poloidal flux at the boundary
!>   psi_axis_tol: (real8) tolerance of the poloidal flux axis
!> outputs:
!>   psi_mesh: (real8)(n_psi) squared psi mesh
subroutine generate_equidistant_poloidal_flux_mesh(n_psi,psi_axis,psi_bnd,psi_axis_tol,psi_mesh)
  implicit none
  !> inputs:
  integer, intent(in) :: n_psi
  real*8, intent(in)  :: psi_axis,psi_bnd,psi_axis_tol
  !> outputs:
  real*8,dimension(n_psi),intent(out) :: psi_mesh
  !> compute mesh
  call generate_squared_mesh_1d(n_psi,[psi_axis*(1d0+psi_axis_tol),psi_bnd],psi_mesh)
end subroutine generate_equidistant_poloidal_flux_mesh

!> compute the magnetic field at a give R,Z position
!> all points outside the boundary have the boundary poloidal flux value
!> inputs:
!>   fields:           (fields_base) JOREK MHD fields
!>   n_v:              (integer) number of magnetic field vector components (must be 3)
!>   n_R:              (integer) number of mesh points along the major radius
!>   n_Z:              (integer) number of mesh points along the vertical coordinates
!>   time:             (real8) time of the magnetic field
!>   tor_angle:        (real8) toroidal angle
!>   ploidal_flux_bnd: (real8) boundary poloidal flux
!>   R_mesh:           (real8)(n_R) mesh along the major radius
!>   Z_mesh:           (real8)(n_Z) mesh along the vertical coordinate
!> outputs:
!>   magnetic_field: (real8)(n_v,n_Z,n_R) radial, vertical and toroidal components
!>                   of the magnetic field
!>   poloidal_flux:  (real8)(n_Z,n_R) poloidal flux
subroutine compute_magnetic_field_poloidal_flux(fields,n_v,n_R,n_Z,time,tor_angle,&
R_mesh,Z_mesh,ploidal_flux_bnd,magnetic_field,poloidal_flux)
  use mod_fields, only: fields_base
  implicit none
  !> inputs:
  class(fields_base),intent(in)    :: fields
  integer,intent(in)               :: n_v,n_R,n_Z
  real*8,intent(in)                :: time,tor_angle,ploidal_flux_bnd
  real*8,dimension(n_R),intent(in) :: R_mesh
  real*8,dimension(n_Z),intent(in) :: Z_mesh
  !> outputs:
  real*8,dimension(n_v,n_Z,n_R),intent(out) :: magnetic_field
  real*8,dimension(n_Z,n_R),intent(out)     :: poloidal_flux
  !> variables:
  integer             :: ii,jj,i_elm_out,ierr
  real*8              :: R_out,Z_out,U
  real*8,dimension(2) :: st_out
  real*8,dimension(3) :: E_field
  !> initialisation
  magnetic_field = 0d0; poloidal_flux = ploidal_flux_bnd;
  !> loop on the R,Z coordinates
  !$omp parallel do default(private) firstprivate(tor_angle,time,n_R,n_Z) &
  !$omp shared(fields,R_mesh,Z_mesh,magnetic_field,poloidal_flux) collapse(2)
  do ii=1,n_R
    do jj=1,n_Z
      !> compute the mesh node position in local coordinates
      call find_RZ(fields%node_list,fields%element_list,R_mesh(ii),Z_mesh(jj),R_out,Z_out,&
      i_elm_out,st_out(1),st_out(2),ierr)
      if(i_elm_out.le.0) cycle!< cycle if element is not found
      !> compute the magnetic field and store it
      call fields%calc_EBpsiU(time,i_elm_out,st_out,tor_angle,E_field,&
      magnetic_field(:,jj,ii),poloidal_flux(jj,ii),U)
    enddo
  enddo
  !$omp end parallel do
end subroutine compute_magnetic_field_poloidal_flux

!> write the magnetic field file to be provided as input to SOFT
!> inputs:
!>   filename:       (character) name of the file to be generated
!>   fields:         (fields_base) jorek MHD fields
!>   write_wall:     (logical) if true write the tokamak wall in hdf5
!>   n_vec:          (integer) N# of components of the magnetic vector
!>   n_R_loc:        (integer) number of major radius point for one task
!>   n_Z:            (integer) number of vertical positions
!>   n_RZ_wall:      (integer) number of wall nodes
!>   n_lcfs:         (integer) number of last close flux surface nodes
!>   n_mpis:         (integer) number of mpi tasks
!>   my_id:          (integer) mpi task identifier 
!>   RZ_axis:        (real8)(2) magnetic field coordinates
!>   R_mesh:         (real8)(n_mpis*n_R_loc) radial mesh
!>   Z_mesh:         (real8)(n_Z) vertical mesh
!>   R_wall:         (real8)(n_RZ_wall) radial coordinates of the wall
!>   Z_wall:         (real8)(n_RZ_wall) vertical coordinates of the wall
!>   RZ_lcfs:        (real8)(2,n_lcfs) 1-major radius and 2-vertical coordinates
!>                   of the last close flux surface
!>   magnetic_field: (real8)(n_vec,n_Z,n_R_loc) magnetic field Br,Bz,Bphi
!>   poloidal_flux:  (real8)(n_Z,n_R_loc) poloidal flux array
!>   mag_name:       (character) name of the input file (metadeta)
!>   description:    (character) description of the input file (metadeta)
subroutine write_SOFT_magnetic_field_file(filename,fields,write_wall,n_vec,n_R_loc,&
n_Z,n_RZ_wall,n_lcfs,n_mpis,my_id,RZ_axis,R_mesh,Z_mesh,R_wall,Z_wall,RZ_lcfs,&
magnetic_field,poloidal_flux,mag_name,description)
  use mpi
  use hdf5
  use hdf5_io_module, only: HDF5_open_or_create,HDF5_close,HDF5_char_saving
  use hdf5_io_module, only: HDF5_array1D_saving,HDF5_array2D_saving
  use mod_fields,     only: fields_base
  implicit none
  !> inputs:
  class(fields_base),intent(in)                      :: fields
  character(len=*),intent(in)                        :: filename,mag_name,description
  logical,intent(in)                                 :: write_wall
  integer,intent(in)                                 :: n_vec,n_R_loc,n_Z,n_lcfs
  integer,intent(in)                                 :: n_RZ_wall,n_mpis,my_id
  real*8,dimension(2),intent(in)                     :: RZ_axis
  real*8,dimension(n_R_loc*n_mpis),intent(in)        :: R_mesh
  real*8,dimension(n_Z),intent(in)                   :: Z_mesh
  real*8,dimension(n_RZ_wall),intent(in)             :: R_wall,Z_wall
  real*8,dimension(2,n_lcfs),intent(in)              :: RZ_lcfs
  real*8,dimension(n_Z,n_R_loc),intent(in)           :: poloidal_flux
  real*8,dimension(n_vec,n_Z,n_R_loc),intent(in)     :: magnetic_field
  !> variables:
  character(len=8)              :: date
  character(len=10)             :: time
  character(len=5)              :: zone
  character(len=:),allocatable  :: description_full
  integer                       :: ii,desc_len
  integer(HID_T)                :: file_id
  real*8,dimension(2,n_RZ_wall) :: RZ_wall
  real*8,dimension(n_Z,n_mpis*n_R_loc)       :: global_poloidal_flux
  real*8,dimension(n_vec,n_Z,n_mpis*n_R_loc) :: global_magnetic_field

  !> gather the poloidal flux
  call gather_2d_array_equal_chunks(my_id,n_mpis,n_Z,n_R_loc,poloidal_flux,global_poloidal_flux) 
  !> gather the magnetic field components
  do ii = 1,n_vec
    call gather_2d_array_equal_chunks(my_id,n_mpis,n_Z,n_R_loc,&
    magnetic_field(ii,:,:),global_magnetic_field(ii,:,:)) 
  enddo
  !> write SOFT magnetic input file
  if(my_id.eq.0) then
    call HDF5_open_or_create(trim(filename//'.h5'),file_id,ierr,file_access=H5F_ACC_TRUNC_F)
    call HDF5_array1D_saving(file_id,RZ_axis,2,'maxis')         !< write magnetic axis
    call HDF5_array1D_saving(file_id,R_mesh,n_mpis*n_R_loc,'r') !< write radial mesh
    call HDF5_array1D_saving(file_id,Z_mesh,n_Z,'z')            !< write vertical mesh
    !> write the tokamak wall
    RZ_wall(1,:) = R_wall; RZ_wall(2,:) = Z_wall;
    if(write_wall) then
      call HDF5_array2D_saving(file_id,RZ_wall,2,n_RZ_wall,'wall')
    else
      call HDF5_array2D_saving(file_id,RZ_lcfs,2,n_lcfs,'wall')
    endif
    call HDF5_array2D_saving(file_id,RZ_lcfs,2,n_lcfs,'separatrix')
    !> write magnetic field - poloidal flux
    call HDF5_array2D_saving(file_id,global_poloidal_flux,n_Z,n_R_loc*n_mpis,'Psi') !< poloidal flux
    call HDF5_array2D_saving(file_id,global_magnetic_field(1,:,:),n_Z,n_R_loc*n_mpis,'Br')   !< radial B
    call HDF5_array2D_saving(file_id,global_magnetic_field(2,:,:),n_Z,n_R_loc*n_mpis,'Bz')   !< vertical B
    !> add a negative sign to the toroidal magnetic field because the toroidal angle in SOFT
    !> is positive clockwise
    call HDF5_array2D_saving(file_id,-global_magnetic_field(3,:,:),n_Z,n_R_loc*n_mpis,'Bphi')
    !> write magnetic field name
    call HDF5_char_saving(file_id,trim(mag_name),'name')
    !> create and write magnetic field description
    call date_and_time(DATE=date,TIME=time,ZONE=zone)
    desc_len = len(mag_name)+len(date)+len(time)+len(zone)+18
    allocate(character(len=desc_len)::description_full)
    description_full = description//'_date_'//date//'_time_'//time//'_zone_'//zone
    call HDF5_char_saving(file_id,trim(description_full),'desc')
    deallocate(description_full)
    call HDF5_close(file_id)
  endif
end subroutine write_SOFT_magnetic_field_file

!> gather 2d equally chunked array to global 2d array
!> inputs:
!>   my_id:       (integer) mpi task identifier
!>   n_mpis:      (integer) number of mpi tasks
!>   n1:          (integer) size of the local array 1st dimension
!>   n2:          (integer) size of the local array 2nd dimension
!>   local_array: (real8)(n1,n2) local array to be gathered
!> outputs:
!>   global_array: (real8)(n_mpis*n1,n_mpis*n2) gathered global array
subroutine gather_2d_array_equal_chunks(my_id,n_mpis,n1,n2,local_array,global_array)
  use mpi
  implicit none
  !> inputs:
  integer,intent(in)                 :: my_id,n_mpis,n1,n2
  real*8,dimension(n1,n2),intent(in) :: local_array
  !> outputs:
  real*8,dimension(n1,n_mpis*n2),intent(out) :: global_array
  !> variables
  integer                        :: ii,doublesize,subarraytype,resizedsubarraytype,ierr
  integer(kind=MPI_Address_kind) :: startresized,extent
  integer,dimension(n_mpis)      :: disps,counts
  !> create a vector for each subblock and gather them
  if(my_id.eq.0) then
    global_array = 0d0; counts = 1; disps = [(n2*ii,ii=0,n_mpis-1)]; startresized = 0;
    call MPI_Type_size(MPI_REAL8,doublesize,ierr); extent = n1*doublesize;
    call MPI_Type_create_subarray(2,[n1,n_mpis*n2],[n1,n2],[0,0],MPI_ORDER_FORTRAN,MPI_REAL8,subarraytype,ierr)
    call MPI_Type_create_resized(subarraytype,startresized,extent,resizedsubarraytype,ierr)
    call MPI_Type_commit(resizedsubarraytype,ierr)
  endif
  call MPI_gatherv(local_array,n1*n2,MPI_REAL8,global_array,counts,disps,resizedsubarraytype,0,MPI_COMM_WORLD,ierr)
  if(my_id.eq.0) call MPI_Type_free(resizedsubarraytype,ierr) 
end subroutine gather_2d_array_equal_chunks

!> find the minor radii of flux surfaces at the low field side
!> at the magnetic axis vertical position
!> mesh element node numbering
!> 3 - 2
!> |   |
!> 4 - 1
!> inputs:
!>   fields:  (fields_base) JOREK MHD fields
!>   fluxes:  (type_surface_list)(n_psi) flux surface list
!>   psiaxis: (real8) poloidal flux at the magnetic axis
!>   RZ_axis: (real8)(2) major radius and vertical position magnetic axis
!> outputs:
!>   flux_minor_radii_Zmag_LFS: (real8)(n_psi) low-field-side flux surface
!>                              minor radii at the magnetix axis vertical position
subroutine find_LFS_minor_radius_flux_surface(fields,fluxes,psiaxis,RZ_axis,flux_minor_radii_Zmag_LFS)
  use data_structure, only: type_surface_list
  use mod_interp,     only: interp_RZ
  implicit none
  !> inputs:
  class(fields_base),intent(in)      :: fields
  type(type_surface_list),intent(in) :: fluxes
  real*8,intent(in)                  :: psiaxis
  real*8,dimension(2),intent(in)     :: RZ_axis
  !> outputs:
  real*8,dimension(fluxes%n_psi),intent(out) :: flux_minor_radii_Zmag_LFS
  !> variables
  integer :: ii,jj,kk,i_harmonic,n_nodes,ifail,errorcode,ierr
  integer,dimension(2) :: Z_flux_ids
  integer,dimension(3) :: ids_to_test
  real*8               :: s_ZFlux,t_ZFlux
  real*8,dimension(2)  :: ZFlux_new,RZ_flux
  real*8,dimension(4)  :: R_nodes,Z_nodes
  logical              :: Z_axis_in_peice
  !> initialisation
  ifail = 999; flux_minor_radii_Zmag_LFS = -9.99d2;
  n_nodes = size(fluxes%flux_surfaces(1)%t,1) !< number of nodes per element
  i_harmonic = 1; !< the minor radius is computed on the equilibrium
  Z_flux_ids = [-2,var_psi] !< vertical coordinate and poloidal flux indexes
  !> loop on the magnetic flux surfaces
  do ii=1,fluxes%n_psi
    !> loop on the number of elements of each flux surface
    do jj=1,fluxes%flux_surfaces(ii)%n_pieces
      !> compute the surface nodes global coordinates
      do kk=1,n_nodes 
        call interp_RZ(fields%node_list,fields%element_list,fluxes%flux_surfaces(ii)%elm(jj),&
        fluxes%flux_surfaces(ii)%s(kk,jj),fluxes%flux_surfaces(ii)%t(kk,jj),R_nodes(kk),Z_nodes(kk))
      enddo
      !> check if we are in the magnetic axis
      if(fluxes%psi_values(ii).eq.psiaxis) then
        flux_minor_radii_Zmag_LFS(ii) = 0d0; ifail = 0; cycle;
      endif
      !> check if the magnetic axis vertical position is within the element
      !> and the element at the LFS (R>=R_axis)
      if(any(RZ_axis(1).le.R_nodes)) cycle 
      !> it may require the addition of a tolerance in Z for taking into
      !> into account the cubic interpolation
      if(all((Z_nodes.gt.RZ_axis(2))).or.all((Z_nodes.lt.RZ_axis(2)))) cycle 
      !> find the nearest point at Z_axis on the flux surface
      ids_to_test = [jj,jj-1,jj+1]
      do kk=1,3
        call find_point_from_target_in_elm_harmonic(fields,fluxes%flux_surfaces(ii)%elm(ids_to_test(kk)),&
        i_harmonic,n_nodes,Z_flux_ids,fluxes%flux_surfaces(ii)%s(:,ids_to_test(kk)),&
        fluxes%flux_surfaces(ii)%t(:,ids_to_test(kk)),[RZ_axis(2),fluxes%psi_values(ii)],&
        s_ZFlux,t_ZFlux,ZFlux_new,ifail)
        if(ifail.eq.0) then
          !> interpolate the R,Z values of the best match point
          call interp_RZ(fields%node_list,fields%element_list,&
          fluxes%flux_surfaces(ii)%elm(ids_to_test(kk)),s_ZFlux,t_ZFlux,RZ_flux(1),RZ_flux(2))
          exit
        endif
      enddo
      !> if failed again just exit
      if(ifail.ne.0) exit
      !> compute the minor radius
      call interp_RZ(fields%node_list,fields%element_list,fluxes%flux_surfaces(ii)%elm(jj),&
      s_ZFlux,t_ZFlux,RZ_flux(1),RZ_flux(2))
      flux_minor_radii_Zmag_LFS(ii) = norm2(RZ_flux - RZ_axis)
      exit
    enddo
    if(ifail.ne.0) then
      write(*,*) 'flux surface id: ',ii,'not found: abort!'
      call MPI_Abort(MPI_COMM_WORLD,errorcode,ierr)
    endif
  enddo
end subroutine find_LFS_minor_radius_flux_surface

!> find a point given two targets within an element by Newton method
!> inputs:
!>   fields:      (fields)(fields_base) JOREK MHD fields
!>   i_elm:       (integer) element number
!>   i_harmonic:  (integer) selected toroidal harmonic
!>   n_trial:     (integer) number of trial element coordinates
!>   target_ids:  (integer)(2) element indexes of the target variables
!>                -1: R, -2: Z
!>   s_trial:     (real8)(n_trial) 1st trial coordinate in element frame
!>   t_trial:     (real8)(n_trial) 2nd trial coordinate in element frame
!>   targets_old: (real8)(2) target values to find
!> outputs:
!>   s_new:       (real8) 1st point coordinate in element frame
!>   t_new:       (real8) 2nd point coordinate in element frame
!>   targets_new: (real8)(2) computed targets
!>   ifail:       (integer) 0 if find routine is successful
subroutine find_point_from_target_in_elm_harmonic(fields,i_elm,i_harmonic,n_trial,&
target_ids,s_trial,t_trial,targets_old,s_new,t_new,targets_new,ifail)
  use mod_interp, only: interp,interp_RZ
  use mod_fields, only: fields_base
  implicit none
  !> inputs:
  class(fields_base),intent(in)        :: fields
  integer,intent(in)                   :: i_elm,i_harmonic,n_trial
  integer,dimension(2),intent(in)      :: target_ids
  real*8,dimension(2),intent(in)       :: targets_old
  real*8,dimension(n_trial),intent(in) :: s_trial,t_trial
  !> outputs:
  integer,intent(out)             :: ifail
  real*8,intent(out)              :: s_new,t_new
  real*8,dimension(2),intent(out) :: targets_new
  !> variables:
  logical :: converged
  integer :: ii,jj,iter,max_iter
  real*8  :: error,tolerance
  real*8  :: R,R_s,R_t,Z,Z_s,Z_t,dummy1,dummy2,dummy3
  real*8,dimension(2)  :: delta_targets,targets_s,targets_t 
  !> initialisation
  converged = .false. !< true if convergence is reached 
  max_iter  = 10; !< maximum number of iteration for convergence
  tolerance = 1d-16;  !< tolerance for convergence
  ifail = 999;  
  !> apply the Newton method for finding the s,t position given a set of target variables
  !> loop on the number of trial
  do ii=1,n_trial
    targets_new = 0d0; error = 1e50;
    s_new = s_trial(ii); t_new = t_trial(ii);
    !> loop on the number of iterations
    do iter=1,max_iter 
      do jj=1,2
        if(target_ids(jj).le.0) then
          if(target_ids(jj).eq.-1) call interp_RZ(fields%node_list,fields%element_list,&
          i_elm,s_new,t_new,targets_new(jj),targets_s(jj),targets_t(jj),dummy1,dummy2,dummy3) !< target R
          if(target_ids(jj).eq.-2) call interp_RZ(fields%node_list,fields%element_list,&
          i_elm,s_new,t_new,dummy1,dummy2,dummy3,targets_new(jj),targets_s(jj),targets_t(jj)) !< target Z
        else
          call interp(fields%node_list,fields%element_list,i_elm,target_ids(jj),i_harmonic,&
          s_new,t_new,targets_new(jj),targets_s(jj),targets_t(jj))
        endif
      enddo
      delta_targets = targets_old - targets_new
      converged = maxval(abs(delta_targets)).le.tolerance
      if(((s_new.ge.0d0).and.(s_new.le.1d0)).and.((t_new.ge.0d0).and.(t_new.le.1d0)).and.converged) then
        ifail = 0; exit;
      endif
      delta_targets = delta_targets/(targets_s(1)*targets_t(2)-targets_t(1)*targets_s(2))
      s_new = s_new - targets_t(1)*delta_targets(2) + targets_t(2)*delta_targets(1)
      t_new = t_new + targets_s(1)*delta_targets(2) - targets_s(2)*delta_targets(1)
    enddo
    if(((s_new.ge.0d0).and.(s_new.le.1d0)).and.((t_new.ge.0d0).and.(t_new.le.1d0)).and.converged) exit;
  enddo
end subroutine find_point_from_target_in_elm_harmonic

!> compute electron distribution function based on the current density
!> and unform phase space compatible with SOFT inputs.
!> inputs:
!>   fields:        (fields)(fields_base) JOREK MHD fields
!>   fluxes:        (type_surface_list)(n_psi) flux surface list
!>   n_momenta:     (integer) size of the momentum mesh 
!>   n_pitch:       (integer) size of the pitch angle mesh
!>   charge:        (integer) particle charge per unit of hydrogen charge
!>   mass:          (real8) particle mass in AMU
!>   momentum_mesh: (real8)(n_momenta) momentum mesh in [AMU*m/s]
!>   cospitch_mesh: (real8)(n_pitch) cos pitch angle in increasing order
!> outputs:
!>   pdf: (n_momenta,n_pitch,n_psi) soft compatible electron distribution
!>        from radial current density and uniform momentum - pitch angle
subroutine soft_current_density_uniform_phase_pdf(fields,fluxes,n_momenta,&
n_pitch,charge,mass,momentum_mesh,cospitch_mesh,pdf)
  use constants,      only: EL_CHG,PI,SPEED_OF_LIGHT
  use data_structure, only: type_surface_list
  use mod_fields,     only: fields_base
  implicit none
  !> inputs:
  class(fields_base),intent(in)           :: fields
  type(type_surface_list),intent(in)      :: fluxes
  integer,intent(in)                      :: n_momenta,n_pitch,charge
  real*8,intent(in)                       :: mass
  real*8,dimension (n_momenta),intent(in) :: momentum_mesh
  real*8,dimension(n_pitch),intent(in)    :: cospitch_mesh
  !> outputs:
  real*8,dimension(n_momenta,n_pitch,fluxes%n_psi),intent(out) :: pdf
  !> variables:
  integer :: ii
  real*8  :: int_zj,DUMMY_DOUBLE_1,DUMMY_DOUBLE_2
  !> initialisation
  DUMMY_DOUBLE_1 = sqrt((momentum_mesh(1)**2)/((mass*SPEED_OF_LIGHT)**2)+1.d0)
  DUMMY_DOUBLE_2 = sqrt((momentum_mesh(n_momenta)**2)/((mass*SPEED_OF_LIGHT)**2)+1.d0)
  DUMMY_DOUBLE_1 = ((DUMMY_DOUBLE_1**3)-3.d0*DUMMY_DOUBLE_1) - ((DUMMY_DOUBLE_2**3)-3.d0*DUMMY_DOUBLE_2);
  DUMMY_DOUBLE_1 = DUMMY_DOUBLE_1*(cospitch_mesh(n_pitch)**2 - cospitch_mesh(2)**2)
  !> compute current density based pdf
  !$omp parallel do default(shared) firstprivate(DUMMY_DOUBLE_1,charge,mass) &
  !$omp private(int_zj,ii)
  do ii=1,fluxes%n_psi
    !> integrate the current density over a flux surface
    call integrate_current_density_over_flux_surface(fields,fluxes%flux_surfaces(ii),int_zj)
    pdf(:,:,ii) = (3d0*int_zj)/(DUMMY_DOUBLE_1*real(abs(charge),kind=8)*PI*EL_CHG*(mass**3)*(SPEED_OF_LIGHT**4))
  enddo
  !$omp end parallel do 
end subroutine soft_current_density_uniform_phase_pdf 

!> Method for integrate the toroidal current density over a flux surface 
!> mutuated by the determine_q_profile routine. Only the n=0 mode is 
!> integrated given that the integral over [0,2*PI] of higher order
!> harmonics is 0 (assuming the magnetic toroidal coordinate to be
!> alligned with the geometrical toroidal coordinate). 
!> The SI units are used.
!> inputs:
!>   fields:       (fields_base) JOREK MHD fields
!>   flux_surface: (type_surface) flux surface datatype
!> outptus:
!>   int_jz: (real8) integral of the toroidal current density in SI units
subroutine integrate_current_density_over_flux_surface(fields,flux_surface,int_zj)
  use constants,          only: MU_ZERO,TWOPI
  use mod_model_settings, only: var_zj
  use data_structure,     only: type_surface
  use mod_interp,         only: interp_RZ,interp
  use mod_fields,         only: fields_base
  implicit none
  !> parameters:
  !> Gaussian points between (-1d0,1d0) to be used for Gauss integration
  real*8, parameter :: xgs(4)=(/-0.861136311594053,-0.339981043584856,0.339981043584856,0.861136311594053/)
  real*8, parameter :: wgs(4)=(/0.347854845137454,0.652145154862546,0.652145154862546,0.347854845137454/)
  !> inputs:
  class(fields_base),intent(in) :: fields
  type(type_surface),intent(in) :: flux_surface
  !> outputs:
  real*8,intent(out) :: int_zj
  !> variables:
  integer :: jj,kk
  real*8  :: ri,dri,si,dsi,zjgi,dl
  real*8  :: Rgi,dRgi_dr,dRgi_ds,dRgi_drs,dRgi_drr,dRgi_dss,dRgi_dt
  real*8  :: Zgi,dZgi_dr,dZgi_ds,dZgi_drs,dZgi_drr,dZgi_dss,dZgi_dt
  !> initialisation
  int_zj = 0d0;
  !> loop on the number of elements and node elements.
  !> WARNING This is not consistent for the magnetic axis
  do jj=1,flux_surface%n_pieces
    do kk=1,4
      !> interpolate the local coordinates on the flux surface
      call CUB1D(flux_surface%s(1,jj),flux_surface%s(2,jj),flux_surface%s(3,jj),&
      flux_surface%s(4,jj),xgs(kk),ri,dri)
      call CUB1D(flux_surface%t(1,jj),flux_surface%t(2,jj),flux_surface%t(3,jj),&
      flux_surface%t(4,jj),xgs(kk),si,dsi)
      !> interpolate the position and the current density
      call interp(fields%node_list,fields%element_list,flux_surface%elm(jj),var_zj,1,ri,si,zjgi)
      call interp_RZ(fields%node_list,fields%element_list,flux_surface%elm(jj),ri,si,Rgi,dRgi_dr,&
      dRgi_ds,dRgi_drs,dRgi_drr,dRgi_dss,Zgi,dZgi_dr,dZgi_ds,dZgi_drs,dZgi_drr,dZgi_dss)
      !> element of curve for the integral
      dRgi_dt = dRgi_dr*dri + dRgi_ds*dsi; dZgi_dt = dZgi_dr*dri + dZgi_ds*dsi;
      dl = sqrt(dRgi_dt**2 + dZgi_dt**2)
      int_zj = int_zj - (wgs(kk)*zjgi*dl)/Rgi !< integrate the current density 
    enddo
  enddo
  !> the 2*PI is used as result for the integral along the toroidal direction
  int_zj = TWOPI*int_zj/MU_ZERO
end subroutine integrate_current_density_over_flux_surface

!> write SOFT distribution function file
!> inputs:
!>   filename:           (character)(*) name of the hdf5 file
!>   my_id:              (integer) mpi task number
!>   n_mpis:             (integer) total number of mpi tasks
!>   n_radii_per_task:   (integer) number of radial points per task
!>   n_momenta:          (integer) number of momentum mesh nodes
!>   n_pitch:            (integer) number of pitch mesh nodes
!>   minor_radii_task:   (real8)(n_radii_per_task) minor radii of each task
!>   momentum_mesh:      (real8)(n_momenta) momentum mesh
!>   cospitch_mesh:      (real8)(n_pitch) cos pitch angle mesh in increasing order
!>   pdf:                (real8)(n_momenta,n_pitch,n_radii_per_task) particle distribution function
!> outputs
subroutine write_soft_distribution_function(filename,my_id,n_mpis,n_radii_per_task,&
n_momenta,n_pitch,minor_radii_task,momentum_mesh,cospitch_mesh,pdf)
  use mpi
  use hdf5
  use hdf5_io_module, only: HDF5_open_or_create,HDF5_close,HDF5_char_saving
  use hdf5_io_module, only: HDF5_array1D_saving,HDF5_array2D_saving
  use constants,      only: ATOMIC_MASS_UNIT
  implicit none
  !> inputs:
  character(len=*),intent(in) :: filename
  integer,intent(in)          :: my_id,n_mpis,n_radii_per_task,n_momenta,n_pitch
  real*8,dimension(n_radii_per_task),intent(in) :: minor_radii_task
  real*8,dimension(n_momenta),intent(in)        :: momentum_mesh
  real*8,dimension(n_pitch),intent(in)          :: cospitch_mesh
  real*8,dimension(n_momenta,n_pitch,n_radii_per_task),intent(in) :: pdf
  !> variables:
  character(len=10)            :: format_char
  character(len=:),allocatable :: group_name
  integer                      :: ii,jj,kk,r_id,n_r_id,ierr,group_name_len
  integer(HID_T)               :: file_id,group_id
  integer,dimension(MPI_STATUS_SIZE)        :: statuss
  real*8,dimension(n_mpis*n_radii_per_task) :: minor_radii_global
  real*8,dimension(:,:,:),allocatable       :: pdf_local

  !> open the hdf5 file 
  if(my_id.eq.0) call HDF5_open_or_create(trim(filename//'.h5'),file_id,ierr,file_access=H5F_ACC_TRUNC_F)
  !> gather the minor radius from all mpi tasks
  call mpi_gather(minor_radii_task,n_radii_per_task,MPI_REAL8,minor_radii_global,&
  n_radii_per_task,MPI_REAL8,0,MPI_COMM_WORLD,ierr)
  if(my_id.eq.0) then
    !> write minor radius in HDF5 file
    call HDF5_array1D_saving(file_id,minor_radii_global,n_mpis*n_radii_per_task,'r')
  endif
  !> send the pdf distribution to root
  if(my_id.ne.0) call MPI_send(pdf,n_radii_per_task*n_momenta*n_pitch,MPI_REAL8,0,my_id,MPI_COMM_WORLD,ierr);
  !> write pdf distribution in hdf5 file
  if(my_id.eq.0) then
    !> generate the xi mesh
    allocate(pdf_local(n_momenta,n_pitch,n_radii_per_task))
    !> loop on the number of mpi's
    do ii=0,n_mpis-1
      if(ii.eq.0) then
        pdf_local = pdf
      else
        call mpi_recv(pdf_local,n_radii_per_task*n_momenta*n_pitch,MPI_REAL8,ii,ii,MPI_COMM_WORLD,statuss,ierr)
      endif
      do jj=0,n_radii_per_task-1
        !> create group
        r_id = ii*n_radii_per_task+jj
        n_r_id = int(log10(real(r_id)))+1
        if(r_id.eq.0) n_r_id = 1
        write(format_char,'(A,I1,A)') "(A,I",n_r_id,")" 
        group_name_len = 1+n_r_id
        allocate(character(len=group_name_len)::group_name)
        write(group_name,trim(format_char)) "r",r_id
        call h5gcreate_f(file_id,trim(group_name),group_id,ierr)
        call h5gclose_f(group_id,ierr)
        !> dump data in hdf5
        call HDF5_array2D_saving(file_id,pdf_local(:,:,jj+1),n_momenta,n_pitch,trim(group_name)//"/f")
        call HDF5_array1D_saving(file_id,momentum_mesh,n_momenta,trim(group_name)//"/p")
        call HDF5_array1D_saving(file_id,cospitch_mesh,n_pitch,trim(group_name)//"/xi")
        !> cleanup
        deallocate(group_name)
      enddo
    enddo
    !> cleanup
    call HDF5_close(file_id)
    deallocate(pdf_local)
  endif
end subroutine write_soft_distribution_function

!> find points on the separatrix (last close flux surface)
!> inputs:
!>   fields:   (fields_base) JOREK MHD field type
!>   my_id:    (integer) rank of the MPI task
!>   xpoint:   (logical) if true a x-point lcfs is found
!>   xcase:    (integer) type of x-point case
!>   n_points: (integer) number of lcfs RZ points per segment
!>   psi_bnd:  (real8) flux of the plasma boundary
!>   RZ_mag:   (real8)(2) 1- R and 2-Z coordinates of the magnetic axis
!> outputs:
!>   n_lcfs:   (integer) number of RZ points of the lcfs
!>   RZ_lcfs:  (real8)(2,n_lcfs) 1-R and 2-Z coordinates of the lcfs
subroutine find_and_compute_separatrix(fields,my_id,xpoint,xcase,n_points,&
psi_bnd,RZ_mag,n_lcfs,RZ_lcfs)
  use constants,      only: TWOPI
  use data_structure, only: type_surface_list
  use mod_interp,     only: interp_RZ
  use mod_fields,     only: fields_base
  implicit none
  !> inputs:
  class(fields_base) :: fields
  logical,intent(in) :: xpoint
  integer,intent(in) :: my_id,xcase,n_points
  real*8,intent(in)  :: psi_bnd
  real*8,dimension(2),intent(in) :: RZ_mag
  !> outputs:
  integer,intent(out)                           :: n_lcfs
  real*8,dimension(:,:),allocatable,intent(out) :: RZ_lcfs
  !> variables:
  type(type_surface_list) :: lcfs_list
  integer                 :: jj,kk,i_elm
  real*8                  :: ss1,dss1,ss2,dss2,tt1,dtt1,tt2,dtt2
  real*8                  :: u,si,ti,dsi,dti
  real*8,dimension(:),allocatable :: theta_lcfs
  !> find the separatrix
  lcfs_list%n_psi=1; allocate(lcfs_list%psi_values(lcfs_list%n_psi));
  lcfs_list%psi_values(1) = psi_bnd;
  call find_flux_surfaces(my_id,xpoint,xcase,fields%node_list,fields%element_list,lcfs_list);
  n_lcfs = lcfs_list%flux_surfaces(1)%n_pieces*n_points; allocate(RZ_lcfs(2,n_lcfs));
  allocate(theta_lcfs(n_lcfs));
  !> compute RZ positions of the different segments, loop on the segments
  do jj=1,lcfs_list%flux_surfaces(1)%n_pieces
    !> extract data of the segment element
    i_elm = lcfs_list%flux_surfaces(1)%elm(jj)
    ss1   = lcfs_list%flux_surfaces(1)%s(1,jj)
    dss1  = lcfs_list%flux_surfaces(1)%s(2,jj)
    ss2   = lcfs_list%flux_surfaces(1)%s(3,jj)
    dss2  = lcfs_list%flux_surfaces(1)%s(4,jj)
    tt1   = lcfs_list%flux_surfaces(1)%t(1,jj)
    dtt1  = lcfs_list%flux_surfaces(1)%t(2,jj)
    tt2   = lcfs_list%flux_surfaces(1)%t(3,jj)
    dtt2  = lcfs_list%flux_surfaces(1)%t(4,jj)
    !> loop over the points of the segment for computing the RZ positions
    do kk=1,n_points
      u = -1d0 + 2d0*real(kk-1,kind=8)/real(n_points-1,kind=8)
      !> compute the s and t values of the point in the i_elm element
      call CUB1D(ss1,dss1,ss2,dss2,u,si,dsi)
      call CUB1D(tt1,dtt1,tt2,dtt2,u,ti,dti)
      !> interpolate the lcfs position
      call interp_RZ(fields%node_list,fields%element_list,i_elm,si,ti, &
      RZ_lcfs(1,(jj-1)*n_points+kk),RZ_lcfs(2,(jj-1)*n_points+kk))
    enddo
  enddo
  !> sorting the lcfs
  theta_lcfs = atan2(RZ_lcfs(2,:)-RZ_mag(2),RZ_lcfs(1,:)-RZ_mag(1))
  where(theta_lcfs.gt.0d0) theta_lcfs = TWOPI+theta_lcfs
  call sort(RZ_lcfs(1,:),RZ_lcfs(2,:),theta_lcfs,n_lcfs)
  !> clean-up
  if(allocated(lcfs_list%psi_values))    deallocate(lcfs_list%psi_values)
  if(allocated(lcfs_list%flux_surfaces)) deallocate(lcfs_list%flux_surfaces)
  if(allocated(theta_lcfs))              deallocate(theta_lcfs)
end subroutine find_and_compute_separatrix

!> basic sorting algorithm three arrays given one ordering
!> inputs:
!>   x:     (real8)(n_elem) array to be sorted
!>   y:     (real8)(n_elem) array to be sorted
!>   z:     (real8)(n_elem) array to be sorted
!>   n_elm: (integer) number of elements in the array
!> outputs:
!>   x: (real8)(n_elem) array to be sorted w.r.t. z
!>   y: (real8)(n_elem) array to be sorted w.r.t. z
!>   z: (real8)(n_elem) array to be sorted
subroutine sort(x,y,z,n_elem)
  implicit none
  !> inputs:
  integer,intent(in)                     :: n_elem
  !> input-outputs:
  real*8,dimension(n_elem),intent(inout) :: x,y,z
  !> variables:
  integer :: jj,ii
  real*8  :: tmp
  !> sort from minimum to maximum 
  do ii=1,n_elem-1
    do jj=ii+1,n_elem
      if(z(jj).lt.z(ii)) then
        tmp = x(ii); x(ii) = x(jj); x(jj) = tmp;
        tmp = y(ii); y(ii) = y(jj); y(jj) = tmp;
        tmp = z(ii); z(ii) = z(jj); z(jj) = tmp;
      endif
    enddo
  enddo
end subroutine sort

!> find the flux surfaces or define a flux surface with the 
!> magnetic axis values in case of
!> inputs:
!>   my_id:             (integer) MPI rank id
!>   x_case:            (integer) type of plasma configuration (limiter,x-point,etc)
!>   x_point:           (logical) if true a x-point configuration is used
!>   i_elm_axis:        (integer) element containing the magnetic axis
!>   psi_axis:          (real8) poloidal flux at the magnetic axis
!>   st_axis:           (real8) local element coordinated of the magnetic axis
!>   fields:            (fields_base) JOREK particle MHD fields
!>   flux_surface_list: (type_surface_list) list of flux surfaces including 
!>                     the magnetic axis 
!> outputs:
!>   flux_surface_list: (type_surface_list) list of flux surfaces including 
!>                     the magnetic axis 
subroutine define_flux_surfaces(my_id,x_case,x_point,i_elm_axis,psi_axis,\
st_axis,fields,flux_surface_list)
  use data_structure, only: type_surface_list
  use mod_fields,     only: fields_base
  implicit none
  !> inputs:
  class(fields_base),intent(in)  :: fields
  integer,intent(in)             :: my_id,x_case,i_elm_axis
  logical,intent(in)             :: x_point 
  real*8,intent(in)              :: psi_axis
  real*8,dimension(2),intent(in) :: st_axis
  !> inputs-outputs:
  type(type_surface_list),intent(inout) :: flux_surface_list
  !> variables:
  integer :: ii,errorcode,ierr
  !> initialisation
  errorcode = 999
  !> find the flux surfaces
  call find_flux_surfaces(my_id,x_point,x_case,fields%node_list,fields%element_list,flux_surfaces);
  do ii=1,flux_surfaces%n_psi
    if(flux_surface_list%flux_surfaces(ii)%n_pieces.eq.0) then
      if(flux_surface_list%psi_values(ii).eq.psi_axis) then
        flux_surface_list%flux_surfaces(ii)%n_pieces = 1
        flux_surface_list%flux_surfaces(ii)%psi = psi_axis
        flux_surface_list%flux_surfaces(ii)%n_parts = 1
        flux_surface_list%flux_surfaces(ii)%parts_index(1) = 1 !< to be checked
        flux_surface_list%flux_surfaces(ii)%elm(1) = i_elm_axis
        flux_surface_list%flux_surfaces(ii)%s(:,1) = st_axis(1)
        flux_surface_list%flux_surfaces(ii)%t(:,1) = st_axis(2)
      else
        write(*,*) 'one or more flux surfaces are not found psi: ',&
        flux_surface_list%psi_values(ii),' psi axis: ',psi_axis,' stop!'
        call MPI_Abort(MPI_COMM_WORLD,errorcode,ierr)
      endif
    endif
  enddo 
end subroutine define_flux_surfaces

!> ------------------------------------------------------------------------------------------------------
end program generate_input_files_for_SOFT

