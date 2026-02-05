!> mod_particle_sim_test contains variables and 
!> procedure for testing the methods coded in
!> mod_particle_sim which do not require MPI
module mod_particle_sim_test
use fruit
use mod_particle_sim, only: particle_sim
use mod_particle_common_test_tools, only: n_particle_types
implicit none

private
public :: run_fruit_particle_sim

!> Variables ------------------------------------
integer,parameter :: n_particles=1000
real*8,parameter  :: survival_threshold=3.25d-1
real*8,parameter  :: tol_real8=5.d-16
real*8,parameter  :: central_mass_test=2.d0
real*8,parameter  :: central_density_test=1.5d-1
integer,dimension(n_particle_types) :: n_active_particles_sol
integer,dimension(n_particle_types) :: particle_type_list_sol
integer*1,dimension(n_particles,n_particle_types) :: particle_charge_list_sol
integer,dimension(n_particles,n_particle_types)   :: active_particle_ids_sol
type(particle_sim)  :: sim_sol
!> Interfaces -----------------------------------

contains

!> Fruit basket ---------------------------------
!> fruit basket containing all set-up, test and
!> tear-down procedures
subroutine run_fruit_particle_sim()
  implicit none
  write(*,'(/A)') "  ... setting-up: particle sim tests"
  call setup
  write(*,'(/A)') "  ... running: particle sim tests"
  call run_test_case(test_set_t_norm,'test_set_t_norm')
  call run_test_case(test_compute_group_size,'test_compute_group_size')
  call run_test_case(test_compute_particle_size,'test_compute_particle_size')
  call run_test_case(test_allocate_groups_sim,'test_allocate_groups_sim')
  call run_test_case(test_codify_particle_sim,'test_codify_particle_sim')
  call run_test_case(test_find_active_particle_id,&
  'test_find_active_particle_id')
  call run_test_case(test_find_active_particle_id_type,&
  'test_find_active_particle_id_type')
  write(*,'(/A)') "  ... tearing-down: particle sim tests"
  call teardown
end subroutine run_fruit_particle_sim

!> Set-up and tear-down -------------------------
!> set-up particle types test features
subroutine setup()
  use phys_module, only: central_mass,central_density
  use mod_particle_types, only: particle_fieldline_id,particle_gc_id,particle_gc_vpar_id
  use mod_particle_types, only: particle_kinetic_id,particle_kinetic_leapfrog_id
  use mod_particle_types, only: particle_kinetic_relativistic_id,particle_gc_relativistic_id
  use mod_particle_types, only: particle_gc_Qin_id
  use mod_particle_common_test_tools, only: fill_particles
  use mod_particle_common_test_tools, only: fill_groups
  use mod_particle_common_test_tools, only: invalidate_particles
  use mod_particle_common_test_tools, only: obtain_active_particle_ids
  use mod_particle_common_test_tools, only: obtain_particle_charges
  use mod_particle_common_test_tools, only: allocate_one_particle_list_type
  implicit none
  !> variables
  integer :: ifail

  !> set test values
  central_mass = central_mass_test; central_density = central_density_test;

  !> allocate the particle lists
  ifail = 0; allocate(sim_sol%groups(n_particle_types))
  call allocate_one_particle_list_type(n_particle_types,n_particles,sim_sol%groups,ifail)
  call assert_true(ifail.eq.0,"Error particle_sim test setup: particle list not allocated!")

  !> fill-up the group and particle base variables
  call fill_groups(n_particle_types,sim_sol%groups)
  call fill_particles(n_particle_types,sim_sol%groups)

  !> invalidate particles in particle lists
  call invalidate_particles(n_particle_types,n_particles,survival_threshold,&
  n_active_particles_sol,sim_sol%groups)
  call obtain_active_particle_ids(n_particle_types,n_particles,&
  active_particle_ids_sol,sim_sol%groups)

  !> initialise the particle type list
  particle_type_list_sol = (/particle_fieldline_id,particle_gc_id,particle_gc_vpar_id,&
  particle_kinetic_id,particle_kinetic_leapfrog_id,particle_kinetic_relativistic_id,&
  particle_gc_relativistic_id,particle_gc_Qin_id/)
  
  !> get particle charges
  call obtain_particle_charges(n_particle_types,n_particles,&
  particle_charge_list_sol,sim_sol%groups)
end subroutine setup

!> feature teardown
subroutine teardown()
  use phys_module, only: central_mass,central_density
  implicit none
  central_mass = 0.d0; central_density = 0.d0;
end subroutine teardown

!> Tests ----------------------------------------
!> test set_t_norm
subroutine test_set_t_norm()
  use phys_module, only: central_mass,central_density
  use constants,   only: MU_ZERO,ATOMIC_MASS_UNIT
  implicit none
  real*8 :: t_norm_sol
  t_norm_sol = sqrt(MU_ZERO*ATOMIC_MASS_UNIT*central_mass*central_density*1.d20)
  call sim_sol%set_t_norm()
  call assert_equals(t_norm_sol,sim_sol%t_norm,tol_real8,&
  "Error particle_sim set t_norm: t_norm mismatch!")
end subroutine test_set_t_norm

!> test compute group size
subroutine test_compute_group_size()
  implicit none
  call assert_equals(n_particle_types,sim_sol%compute_group_size(),&
  "Error particle_sim compute group: group size mismatch!")
end subroutine test_compute_group_size

!> test compute particle size
subroutine test_compute_particle_size()
  implicit none
  integer :: n_groups
  integer,dimension(n_particle_types) :: n_particle_array,n_part_array_sol
  n_part_array_sol = n_particles; n_groups = n_particle_types;
  call sim_sol%compute_particle_sizes(n_groups,n_particle_array)
  call assert_equals(n_particle_types,n_groups,&
  "Error in particle_sim compute particle sizes: n_groups mismatch!")
  call assert_equals(n_part_array_sol,n_particle_array,n_particle_types,&
  "Error in particle_sim compute particle sizes: particle sizes mismatch!")
end subroutine test_compute_particle_size

!> test allocate groups
subroutine test_allocate_groups_sim()
  use mod_particle_sim, only: particle_sim
  implicit none
  integer,parameter :: n_groups=2
  type(particle_sim) :: sim_test

  !> test group allocation
  call sim_test%allocate_groups(n_particle_types)
  call assert_true((allocated(sim_test%groups)).and.&
  (size(sim_test%groups).eq.n_particle_types),&
  "Error in particle_sim allocate group: unallocated group not allocated!")
  call sim_test%allocate_groups(n_groups)
  call assert_true((size(sim_test%groups).eq.n_groups),&
  "Error in particle_sim allocate group: allocated group not resized!")
  call sim_sol%allocate_groups(n_particle_types)
  call assert_equals(n_particles,size(sim_sol%groups(1)%particles),&
  "Error in particle_sim allocate group: allocated group modified one it should not!") 
end subroutine test_allocate_groups_sim

!> test return particle type code for the whole particle list
subroutine test_codify_particle_sim()
  implicit none
  !> variables
  integer :: ii
  integer,dimension(n_particle_types) :: list_code

  !> extract particle list code 
  call sim_sol%find_particle_types(n_particle_types,list_code)
  call assert_equals(particle_type_list_sol,list_code,n_particle_types,&
  "Error in particle_sim codify particle: particle types mismatch!")
end subroutine test_codify_particle_sim

!> test find active particle id interface notype
subroutine test_find_active_particle_id()
  implicit none
  !> variables
  integer :: ii
  integer,dimension(n_particle_types) :: n_particles_array,n_active_particles
  integer,dimension(n_particles,n_particle_types) :: active_particle_ids

  !> compute active particles and their id
  n_particles_array = n_particles; n_active_particles = -1; active_particle_ids = -1;
  call sim_sol%find_active_particles_groups(n_particle_types,n_particles,&
  n_particles_array,n_active_particles,active_particle_ids)
  !> check results
  call assert_equals(n_active_particles_sol,n_active_particles,n_particle_types,&
  "Error particle_sim find active particle notype: n_active_particles mismatch!")
  call assert_equals(active_particle_ids_sol,active_particle_ids,n_particles,n_particle_types,&
  "Error particle_sim find active particle notype: active_particle_ids mismatch!")
end subroutine test_find_active_particle_id

!> test find active particle id interface notype
subroutine test_find_active_particle_id_type()
  implicit none
  !> variables
  integer,dimension(n_particle_types) :: n_active_particles,n_particle_array
  integer,dimension(n_particles,n_particle_types) :: active_particle_ids

  !> find the active particle id using particle_sim routines
  n_particle_array = n_particles;  
  call sim_sol%find_active_particles_groups(n_particle_types,n_particles,&
  n_particle_array,n_active_particles,active_particle_ids,&
  n_particle_types,particle_type_list_sol)
  !> check correctness of the identification
  call assert_equals(n_active_particles_sol,n_active_particles,n_particle_types,&
  "Error particle_sim find active particle type: n_active_particles mismatch!")
  call assert_equals(active_particle_ids_sol,active_particle_ids,n_particles,n_particle_types,&
  "Error particle_sim find active particle type: active_particle_ids mismatch!")
end subroutine test_find_active_particle_id_type

!>-----------------------------------------------

end module mod_particle_sim_test
