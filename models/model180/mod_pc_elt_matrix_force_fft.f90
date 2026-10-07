!> model180 (equilibrium import) placeholder for the physics PC's force
!! operator W. model180 only sets up the stellarator equilibrium and never runs
!! the physics PC; the time-evolution model183 has the real operator
!! (models/model183/mod_pc_elt_matrix_force_fft.f90). This module exists so a
!! model180 USE_PETSC build links, and stops the run if W is ever requested.
module mod_pc_elt_matrix_force_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft

contains

subroutine pc_elt_matrix_force_fft(element, nodes, xpoint2, xcase2, &
                                   R_axis, Z_axis, psi_axis, psi_bnd, &
                                   R_xpoint, Z_xpoint, ELM, terms)

  use mod_parameters
  use data_structure, only: type_element, type_node

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  logical,            intent(in) :: xpoint2
  integer,            intent(in) :: xcase2
  real*8,             intent(in) :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,             intent(in) :: R_xpoint(2), Z_xpoint(2)
  real*8, dimension(n_tor*n_vertex_max*n_degrees, n_tor*n_vertex_max*n_degrees), intent(out) :: ELM
  integer, intent(in) :: terms

  ELM = 0.d0
  write(*,'(A)') "[Physics PC] FATAL: model180 has no force operator; run the physics PC with model183"
  stop 1

end subroutine pc_elt_matrix_force_fft

end module mod_pc_elt_matrix_force_fft
