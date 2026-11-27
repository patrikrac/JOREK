!> model180 (stellarator equilibrium import) never runs the physics PC: this
!! stub lets a USE_PETSC build link and stops if W is ever requested.
module mod_pc_elt_matrix_force_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft, pc_elt_matrix_force_pfafem

contains

subroutine pc_elt_matrix_force_fft(element, nodes, ELM, ELM_v)

  use mod_parameters
  use data_structure, only: type_element, type_node

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  real*8, dimension(n_tor*n_vertex_max*n_degrees, n_tor*n_vertex_max*n_degrees), intent(out) :: ELM
  !> v_par's sound-wave Schur terms (model600 only): unused, the common interface
  real*8, dimension(n_tor*n_vertex_max*n_degrees, n_tor*n_vertex_max*n_degrees, 3), intent(out), optional :: ELM_v

  ELM = 0.d0
  if (present(ELM_v)) ELM_v = 0.d0

end subroutine pc_elt_matrix_force_fft

subroutine pc_elt_matrix_force_pfafem(node_list, element_list, ielm, W_vp, WV_vp)

  use data_structure, only: type_node_list, type_element_list

  implicit none

  type(type_node_list),    intent(in)            :: node_list
  type(type_element_list), intent(in)            :: element_list
  integer,                 intent(in)            :: ielm
  real*8,                  intent(out)           :: W_vp(:, :, :, :, :)
  real*8,                  intent(out), optional :: WV_vp(:, :, :, :, :, :)

  if (ielm < 0 .and. node_list%n_nodes < 0 .and. element_list%n_elements < 0) continue
  W_vp = 0.d0
  if (present(WV_vp)) WV_vp = 0.d0

end subroutine pc_elt_matrix_force_pfafem

end module mod_pc_elt_matrix_force_fft
