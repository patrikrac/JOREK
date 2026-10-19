!> model180 (stellarator equilibrium import) never runs the physics PC: this
!! stub lets a USE_PETSC build link and stops if W is ever requested.
module mod_pc_elt_matrix_force_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft

contains

subroutine pc_elt_matrix_force_fft(element, nodes, ELM, ELM_v, gamma_w, vu_full)

  use mod_parameters
  use data_structure, only: type_element, type_node

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  real*8, dimension(n_tor*n_vertex_max*n_degrees, n_tor*n_vertex_max*n_degrees), intent(out) :: ELM
  !> v_par's sound-wave Schur terms (model600 only): unused, the common interface
  real*8, dimension(n_tor*n_vertex_max*n_degrees, n_tor*n_vertex_max*n_degrees, 3), intent(out), optional :: ELM_v
  real*8, intent(in), optional :: gamma_w
  logical, intent(in), optional :: vu_full

  ELM = 0.d0
  if (present(ELM_v)) ELM_v = 0.d0
  write(*,'(A)') "[Physics PC] FATAL: model180 has no force operator; run the physics PC with model183"
  stop 1

end subroutine pc_elt_matrix_force_fft

end module mod_pc_elt_matrix_force_fft
