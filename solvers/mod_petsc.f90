module mod_petsc
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc

  implicit none

contains

  subroutine petsc_initialize()
    PetscErrorCode :: ierr
    call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
    if (ierr /= 0) print *, "Error initializing PETSc"
  end subroutine

  subroutine petsc_finalize()
    PetscErrorCode :: ierr
    call PetscFinalize(ierr)
  end subroutine petsc_finalize


  subroutine petsc_print_version()
    PetscErrorCode :: ierr
    character(len=256) :: version_string

    call PetscGetVersion(version_string, ierr)
    if (ierr == 0) then
      print *, "JOREK linked with: ", trim(version_string)
    end if
  end subroutine petsc_print_version


#endif
end module mod_petsc