module mod_petsc_pc
  use mod_petsc_pc_toroidal
  use mod_petsc_pc_physics
  implicit none
  public

  integer, parameter :: PETSC_PC_TOROIDAL_HARMONIC = 1
  integer, parameter :: PETSC_PC_PHYSICS           = 2

contains

  !> Dispatch to the requested preconditioner setup routine.
  subroutine petsc_setup_pc(ksp, A, pc_type)
    KSP, intent(inout) :: ksp
    Mat, intent(in)    :: A
    integer, intent(in) :: pc_type

    select case (pc_type)
      case (PETSC_PC_TOROIDAL_HARMONIC)
        call petsc_setup_toroidal_harmonic_pc(ksp, A)
      case (PETSC_PC_PHYSICS)
        call petsc_setup_physics_pc(ksp, A)
    end select
  end subroutine petsc_setup_pc

end module mod_petsc_pc
