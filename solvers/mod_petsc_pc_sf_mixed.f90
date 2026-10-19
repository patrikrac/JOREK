module mod_petsc_pc_sf_mixed
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx
  use mod_model_settings,       only: jorek_model
  use mod_petsc_pc_blocks,      only: pack_blocks_aij
  use mod_petsc_pc_sf_solver,   only: sf_opz
  implicit none
  private

  !--------------------------------------------------------------------
  !> pair_w on the global path, assembled:
  !!   [[B_22 + W,  B_24,  B_21,       B_23],
  !!    [B_42,      B_44,  0,          0   ],
  !!    [B_12,      0,     opz M_psi,  B_13],
  !!    [0,         0,     B_31,       B_33]]
  !! W composes the psi channel at the continuum level and so misses the
  !! discrete projections M_psi^-1, M_j^-1 and the resistive damping of the
  !! psi response, all growing with dt. Keeping psi and j explicit (the
  !! small-flow psi row) restores them with every block sparse; W keeps only
  !! its curvature term.
  !!
  !! M_psi = B_33 (the psi row's mass and the constraint mass are the same
  !! integrand). Not in model183, where B_33 = theta Bv2 M: there the psi row
  !! takes B_11 itself.
  !!
  !! sfm_build fixes the patterns on the first build; sfm_refill recomputes
  !! the values in place after the SF gather and the W assembly refilled the
  !! blocks, so the packed operator keeps its identity.
  !--------------------------------------------------------------------

  integer, parameter, public :: SFM_NF = 4   !< packed fields (u, omega, psi, j)
  Mat, save, public :: sfm_op                !< the packed pair_w

  Mat, save :: suu                           !< B_22 + W
  Mat, save :: mpsi                          !< opz M_psi (model183: B_11)
  logical, parameter :: psi_mass_b11 = (jorek_model == 183)
  logical, save :: packed = .false.

  public :: sfm_build, sfm_refill

contains

  subroutine sfm_build()
    PetscErrorCode :: ierr
    call MatDuplicate(g_ctx%B_22, MAT_COPY_VALUES, suu, ierr)
    call MatAXPY(suu, 1.0d0, g_ctx%W_force, DIFFERENT_NONZERO_PATTERN, ierr)
    if (psi_mass_b11) then
      call MatDuplicate(g_ctx%B_11, MAT_COPY_VALUES, mpsi, ierr)
    else
      call MatDuplicate(g_ctx%B_33, MAT_COPY_VALUES, mpsi, ierr)
      call MatScale(mpsi, sf_opz(), ierr)
    endif
    call pack_op()
  end subroutine sfm_build

  subroutine sfm_refill()
    PetscErrorCode :: ierr
    call MatZeroEntries(suu, ierr)
    call MatAXPY(suu, 1.0d0, g_ctx%B_22, SUBSET_NONZERO_PATTERN, ierr)
    call MatAXPY(suu, 1.0d0, g_ctx%W_force, SUBSET_NONZERO_PATTERN, ierr)
    if (psi_mass_b11) then
      call MatCopy(g_ctx%B_11, mpsi, SAME_NONZERO_PATTERN, ierr)
    else
      call MatCopy(g_ctx%B_33, mpsi, SAME_NONZERO_PATTERN, ierr)
      call MatScale(mpsi, sf_opz(), ierr)
    endif
    call pack_op()
  end subroutine sfm_refill

  subroutine pack_op()
    Mat :: blk(SFM_NF, SFM_NF)
    logical :: have(SFM_NF, SFM_NF)
    integer :: comm
    PetscErrorCode :: ierr
    have = .false.
    call put(1, 1, suu);         call put(1, 2, g_ctx%B_24)
    call put(1, 3, g_ctx%B_21);  call put(1, 4, g_ctx%B_23)
    call put(2, 1, g_ctx%B_42);  call put(2, 2, g_ctx%B_44)
    call put(3, 1, g_ctx%B_12);  call put(3, 3, mpsi);  call put(3, 4, g_ctx%B_13)
    call put(4, 3, g_ctx%B_31);  call put(4, 4, g_ctx%B_33)
    call PetscObjectGetComm(g_ctx%B_22, comm, ierr)
    call pack_blocks_aij(blk, have, sfm_op, packed, comm)
  contains
    subroutine put(f, g, M)
      integer, intent(in) :: f, g
      Mat, intent(in) :: M
      blk(f, g) = M; have(f, g) = .true.
    end subroutine put
  end subroutine pack_op

#endif
end module mod_petsc_pc_sf_mixed
