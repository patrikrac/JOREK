!> The toroidal slots the physics PC's packed operators carry.
!!
!! The packed GMG layout keeps a (field, scalar DOF)'s harmonics in pc_ntor
!! consecutive rows. On the global path these are all n_tor slots of the run;
!! under physics_pc_sf_mode_split each rank's operators carry one |n| family
!! only (mod_petsc_pc_sf_fam): the n = 0 slot, or the cos and sin slots of one
!! n. pc_ntor is then the family's slot count, and pc_mode maps its local
!! slots back to the global ones, which is all the GMG needs to group slots
!! by |n| the same way as on the global path.
module mod_petsc_pc_harm
  use mod_parameters, only: n_tor
  implicit none
  private

  integer, save, public :: pc_ntor = n_tor          !< slots per (field, scalar DOF)
  integer, allocatable, save :: pc_mode(:)          !< local slot (0-based) -> global slot

  public :: pc_harm_set, pc_grp, pc_ngrp

contains

  !> Restrict the layout to the global slots `modes` (0-based, ascending).
  subroutine pc_harm_set(modes)
    integer, intent(in) :: modes(:)
    if (allocated(pc_mode)) deallocate(pc_mode)
    allocate(pc_mode(0:size(modes) - 1))
    pc_mode = modes
    pc_ntor = size(modes)
  end subroutine pc_harm_set

  !> |n| group of local slot m, numbered from 0 within this layout: slot 0
  !! alone (n = 0), then cos and sin of one n together -- (m+1)/2 globally.
  integer function pc_grp(m)
    integer, intent(in) :: m
    if (allocated(pc_mode)) then
      pc_grp = (pc_mode(m) + 1) / 2 - (pc_mode(0) + 1) / 2
    else
      pc_grp = (m + 1) / 2
    endif
  end function pc_grp

  !> Number of |n| groups in this layout.
  integer function pc_ngrp()
    pc_ngrp = pc_grp(pc_ntor - 1) + 1
  end function pc_ngrp

end module mod_petsc_pc_harm
