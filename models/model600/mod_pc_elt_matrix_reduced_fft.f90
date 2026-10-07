!> model600 placeholder for the physics PC's REDUCED PDE operator P_full.
!!
!! The reduced operator (models/model199/mod_pc_elt_matrix_reduced_fft.f90)
!! substitutes model199's axisymmetric constraint rows at the continuous level;
!! model600 has no counterpart yet. Only the research physics-PC arms
!! (construct_reduced_pde_matrix) call it -- the SF path builds its blocks from
!! the assembled Jacobian and W instead -- so this module exists to let a
!! model600 USE_PETSC build link, and stops the run if such an arm is selected.
module mod_pc_elt_matrix_reduced_fft

  implicit none
  private

  public :: pc_elt_matrix_reduced_fft
  public :: n_var_red

  !> Same reduced ordering as model199: (psi, u, rho, T)
  integer, parameter :: n_var_red = 4

contains

subroutine pc_elt_matrix_reduced_fft(element, nodes, xpoint2, xcase2, &
                                     R_axis, Z_axis, psi_axis, psi_bnd, &
                                     R_xpoint, Z_xpoint, ELM)

  use mod_parameters
  use data_structure, only: type_element, type_node

  implicit none

  type(type_element), intent(in)    :: element
  type(type_node),    intent(in)    :: nodes(n_vertex_max)
  logical,            intent(in)    :: xpoint2
  integer,            intent(in)    :: xcase2
  real*8,             intent(in)    :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,             intent(in)    :: R_xpoint(2), Z_xpoint(2)
  real*8, dimension(n_tor*n_var_red*n_vertex_max*n_degrees, &
                    n_tor*n_var_red*n_vertex_max*n_degrees), intent(out) :: ELM

  ELM = 0.d0
  write(*,'(A)') "[Physics PC] FATAL: the reduced PDE operator is model199-only; "// &
                 "with model600 use the SF path (physics_pc_sf)"
  stop 1

end subroutine pc_elt_matrix_reduced_fft

end module mod_pc_elt_matrix_reduced_fft
