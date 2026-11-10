!> Optional dumping of PETSc operators to binary files, for offline solver
!! experiments.
!!
!! Entirely opt-in and inert unless `-jorek_dump_mat <prefix>` is present in the
!! PETSc options database (i.e. in jorek.petsc). When it is, the matrices JOREK
!! actually hands to its solvers are written in PETSc binary format, which is
!! what src/mat/tests/ex125 and friends read with `-f`.
!!
!! The point is to be able to benchmark a *real* JOREK preconditioner block
!! against different factorization packages (MUMPS, STRUMPACK, cuDSS, ...)
!! outside a full simulation run.
!!
!! Example:
!!   -jorek_dump_mat jorekmat        # writes jorekmat_pcblock1.dat, ...
module mod_petsc_dump
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_dump_operator

  character(len=*), parameter :: DUMP_OPTION = '-jorek_dump_mat'

contains

  !> Write `A` to `<prefix>_<tag>.dat` if -jorek_dump_mat <prefix> was given.
  !!
  !! Collective on A's communicator. A no-op with no collective calls at all
  !! when the option is absent, so it is safe to leave on every call path.
  subroutine petsc_dump_operator(A, tag)
    Mat,              intent(in) :: A
    character(len=*), intent(in) :: tag

    character(len=256) :: prefix, fname
    PetscBool          :: is_set
    PetscViewer        :: viewer
    PetscErrorCode     :: ierr
    integer            :: comm, my_id, mpierr

    prefix = ''
    ! NOTE: PetscCallA is a cpp macro, so its argument cannot be split across a
    ! Fortran '&' continuation -- cpp would never see the second half. Keep the
    ! call on one line; -ffree-line-length-none (defaults.mk:49) permits it.
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, DUMP_OPTION, prefix, is_set, ierr))
    if (.not. is_set) return
    if (len_trim(prefix) == 0) prefix = 'jorekmat'

    fname = trim(prefix)//'_'//trim(tag)//'.dat'

    PetscCallA(PetscObjectGetComm(A, comm, ierr))
    call MPI_COMM_RANK(comm, my_id, mpierr)

    PetscCallA(PetscViewerBinaryOpen(comm, trim(fname), FILE_MODE_WRITE, viewer, ierr))
    PetscCallA(MatView(A, viewer, ierr))
    PetscCallA(PetscViewerDestroy(viewer, ierr))

    if (my_id == 0) write(*,*) '[PETSc] dumped operator to '//trim(fname)
  end subroutine petsc_dump_operator

#endif
end module mod_petsc_dump
