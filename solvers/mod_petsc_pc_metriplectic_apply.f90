module mod_petsc_pc_metriplectic_apply
!----------------------------------------------------------------
! Metriplectic HSS sweep apply (stage B/C; spec Sec. 6, note Sec.
! "The constrained HSS sweep", plan 2026-07-16 Task 4).
!
! One alternating sweep in the reduced (psi,u) space:
!   order 'SK':  pair solves -> M_red mat-vec -> ideal half -> j/w recovery
!   order 'KS':  ideal half -> M_red mat-vec -> pair solves (final incl. j/w)
! Constraint residuals r_j, r_w are consumed with the FINAL dpsi/du by
! whichever half runs last (single-consumption rule, note Sec. sweep).
! rho/T are recovered block-triangularly outside the sweep.
!
! Sign discipline (note Sec. 6 / eq. (Mred)): JOREK's u-row carries
! negative inertia; every scalar below is annotated with its note origin.
!----------------------------------------------------------------
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_metriplectic_ctx, only: g_mctx
  implicit none
  private

  public :: metriplectic_ideal_half_solve
  public :: metriplectic_diss_half_solve
  public :: metriplectic_mass_apply
  public :: metriplectic_recover_jw
  public :: metriplectic_recover_rhoT
  public :: metriplectic_sweep_apply_4v
  public :: metriplectic_sweep_apply_full
  public :: pack_2v, unpack_2v, pack_4v, unpack_4v

  !> Active sweep order ('SK' default; 'KS' = ideal half first). Set from
  !! the namelist flag at PC setup; the analysis flips it for T6.
  character(len=2), public, save :: mpc_sweep_order = 'SK'

  ! --- module work vecs (1-var sized), created on first use ---
  logical :: a_work_ready = .false.
  Vec :: a_h, a_t1, a_t2                      ! ideal-half internals
  Vec :: a_p1, a_u1, a_j1, a_w1               ! first-half outputs
  Vec :: a_yp, a_yu                           ! middle mass outputs
  Vec :: a_rpsi, a_ru, a_rj, a_rw             ! unpacked residuals
  Vec :: a_dpsi, a_du, a_dj, a_dw             ! solution components
  Vec :: a_rrho, a_rT, a_drho, a_dT           ! rho/T components (full apply)

contains

  subroutine ensure_work(ierr)
    PetscErrorCode :: ierr
    if (a_work_ready) return
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_h,    ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_t1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_t2,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_p1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_u1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_j1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_w1,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_yp,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_yu,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rpsi, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_ru,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rj,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rw,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dpsi, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_du,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dj,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dw,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rrho, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_rT,   ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_drho, ierr))
    PetscCallA(VecDuplicate(g_mctx%wv_psi_1, a_dT,   ierr))
    a_work_ready = .true.
  end subroutine ensure_work


  !====================================================================
  ! Reduced K-half solve (segregated; Slice-A algebra with ctx solvers).
  !   h     = M_psi^-1 rpsi / (1+z)             [note SK step 3]
  !   RHS_u = -ru/(1+z) + tau * Dp h            [note Sec. 6: u-row negation
  !                                              flips BOTH terms -> +tau Dp h]
  !   du    = P_u^-1 RHS_u
  !   dpsi  = h - tau * M_psi^-1 (D du)
  !====================================================================
  subroutine metriplectic_ideal_half_solve(rpsi, ru, dpsi, du)
    use phys_module, only: time_evol_zeta
    Vec :: rpsi, ru, dpsi, du
    PetscErrorCode :: ierr
    real*8 :: opz, tau

    opz = 1.d0 + time_evol_zeta
    tau = g_mctx%dt_theta
    call ensure_work(ierr)

    call KSPSolve(g_mctx%ksp_Mpsi, rpsi, a_h, ierr)
    call VecScale(a_h, 1.d0/opz, ierr)

    call MatMult(g_mctx%Dp_op, a_h, a_t2, ierr)
    call VecAXPBY(a_t2, -1.d0/opz, +tau, ru, ierr)   ! t2 = -ru/(1+z) + tau*t2

    call KSPSolve(g_mctx%ksp_Pu, a_t2, du, ierr)

    call MatMult(g_mctx%D_op, du, a_t1, ierr)
    call KSPSolve(g_mctx%ksp_Mpsi, a_t1, dpsi, ierr)
    call VecAYPX(dpsi, -tau, a_h, ierr)              ! dpsi = h - tau*dpsi
  end subroutine metriplectic_ideal_half_solve


  !====================================================================
  ! S-half: coupled constraint-pair solves (mixed form, note eq. (pair)).
  !   [B11 B13; B31 B33](dpsi,dj) = (rpsi,rj)
  !   [B22 B24; B42 B44](du, dw ) = (ru,  rw)
  !====================================================================
  subroutine metriplectic_diss_half_solve(rpsi, ru, rj, rw, dpsi, du, dj, dw)
    Vec :: rpsi, ru, rj, rw, dpsi, du, dj, dw
    PetscErrorCode :: ierr

    call pack_2v(rpsi, rj, g_mctx%wv_pair_psij_1, ierr)
    call KSPSolve(g_mctx%ksp_pair_psij, g_mctx%wv_pair_psij_1, &
                  g_mctx%wv_pair_psij_2, ierr)
    call unpack_2v(g_mctx%wv_pair_psij_2, dpsi, dj, ierr)

    call pack_2v(ru, rw, g_mctx%wv_pair_uw_1, ierr)
    call KSPSolve(g_mctx%ksp_pair_uw, g_mctx%wv_pair_uw_1, &
                  g_mctx%wv_pair_uw_2, ierr)
    call unpack_2v(g_mctx%wv_pair_uw_2, du, dw, ierr)
  end subroutine metriplectic_diss_half_solve


  !====================================================================
  ! Middle operator M_red (note eq. (Mred)):
  !   y_psi = +(1+z) M_psi  z_psi
  !   y_u   = -(1+z) L_rho  z_u     <- MINUS: JOREK u-row negative inertia
  !====================================================================
  subroutine metriplectic_mass_apply(zpsi, zu, ypsi, yu)
    use phys_module, only: time_evol_zeta
    Vec :: zpsi, zu, ypsi, yu
    PetscErrorCode :: ierr
    real*8 :: opz

    opz = 1.d0 + time_evol_zeta
    call MatMult(g_mctx%M_psi, zpsi, ypsi, ierr)
    call VecScale(ypsi, +opz, ierr)
    call MatMult(g_mctx%L_rho, zu, yu, ierr)
    call VecScale(yu, -opz, ierr)
  end subroutine metriplectic_mass_apply


  !====================================================================
  ! Constraint recovery with the final dpsi/du (note SK step 4):
  !   dj = B33^-1 (rj - B31 dpsi),  dw = B44^-1 (rw - B42 du)
  !====================================================================
  subroutine metriplectic_recover_jw(rj, rw, dpsi, du, dj, dw)
    Vec :: rj, rw, dpsi, du, dj, dw
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    call MatMult(g_mctx%B_31s, dpsi, a_t1, ierr)
    call VecAYPX(a_t1, -1.d0, rj, ierr)              ! t1 = rj - B31 dpsi
    call KSPSolve(g_mctx%ksp_B33, a_t1, dj, ierr)

    call MatMult(g_mctx%B_42s, du, a_t1, ierr)
    call VecAYPX(a_t1, -1.d0, rw, ierr)              ! t1 = rw - B42 du
    call KSPSolve(g_mctx%ksp_B44, a_t1, dw, ierr)
  end subroutine metriplectic_recover_jw


  !====================================================================
  ! rho/T block-triangular recovery (outside the sweep; spec 6.1.5):
  !   drho = B55^-1 (rrho - B52 du),  dT = B66^-1 (rT - B62 du)
  !====================================================================
  subroutine metriplectic_recover_rhoT(rrho, rT, du, drho, dT)
    Vec :: rrho, rT, du, drho, dT
    PetscErrorCode :: ierr

    call MatMult(g_mctx%B_52s, du, g_mctx%wv_rho_2, ierr)
    call VecAYPX(g_mctx%wv_rho_2, -1.d0, rrho, ierr)
    call KSPSolve(g_mctx%ksp_B55, g_mctx%wv_rho_2, drho, ierr)

    call MatMult(g_mctx%B_62s, du, g_mctx%wv_T_2, ierr)
    call VecAYPX(g_mctx%wv_T_2, -1.d0, rT, ierr)
    call KSPSolve(g_mctx%ksp_B66, g_mctx%wv_T_2, dT, ierr)
  end subroutine metriplectic_recover_rhoT


  !====================================================================
  ! Sweep core on 1-var components (both orders; note apply sequences)
  !====================================================================
  subroutine metriplectic_sweep_core(rpsi, ru, rj, rw, dpsi, du, dj, dw)
    Vec :: rpsi, ru, rj, rw, dpsi, du, dj, dw
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    if (mpc_sweep_order == 'KS') then
      ! P^-1 = A1^-1 M_red A2^-1: ideal first, pairs last (consume rj, rw)
      call metriplectic_ideal_half_solve(rpsi, ru, a_p1, a_u1)
      call metriplectic_mass_apply(a_p1, a_u1, a_yp, a_yu)
      call metriplectic_diss_half_solve(a_yp, a_yu, rj, rw, dpsi, du, dj, dw)
    else
      ! 'SK' (default): P^-1 = A2^-1 M_red A1^-1: pairs first (their j/w
      ! outputs are discarded), ideal last, recovery consumes rj, rw
      call metriplectic_diss_half_solve(rpsi, ru, rj, rw, a_p1, a_u1, a_j1, a_w1)
      call metriplectic_mass_apply(a_p1, a_u1, a_yp, a_yu)
      call metriplectic_ideal_half_solve(a_yp, a_yu, dpsi, du)
      call metriplectic_recover_jw(rj, rw, dpsi, du, dj, dw)
    endif
  end subroutine metriplectic_sweep_core


  !====================================================================
  ! PCSHELL callback on the packed 4-var (psi,u,j,w) vector (T5b/T6)
  !====================================================================
  subroutine metriplectic_sweep_apply_4v(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    call unpack_4v(x, a_rpsi, a_ru, a_rj, a_rw, ierr)
    call metriplectic_sweep_core(a_rpsi, a_ru, a_rj, a_rw, a_dpsi, a_du, a_dj, a_dw)
    call pack_4v(a_dpsi, a_du, a_dj, a_dw, y, ierr)
    ierr = 0
  end subroutine metriplectic_sweep_apply_4v


  !====================================================================
  ! PCSHELL callback on the FULL system vector (production, T5c).
  ! Handles (psi,u,j,w) via the sweep + (rho,T) via recovery; any
  ! further variables pass through as identity (VecCopy first).
  !====================================================================
  subroutine metriplectic_sweep_apply_full(pc, x, y, ierr)
    use mod_parameters, only: var_psi, var_u, var_zj, var_w, var_rho, var_T
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    call ensure_work(ierr)
    call copy_var_out(x, var_psi, a_rpsi, ierr)
    call copy_var_out(x, var_u,   a_ru,   ierr)
    call copy_var_out(x, var_zj,  a_rj,   ierr)
    call copy_var_out(x, var_w,   a_rw,   ierr)
    call copy_var_out(x, var_rho, a_rrho, ierr)
    call copy_var_out(x, var_T,   a_rT,   ierr)

    call metriplectic_sweep_core(a_rpsi, a_ru, a_rj, a_rw, a_dpsi, a_du, a_dj, a_dw)
    call metriplectic_recover_rhoT(a_rrho, a_rT, a_du, a_drho, a_dT)

    call VecCopy(x, y, ierr)     ! identity pass-through for unhandled vars
    call copy_var_in(y, var_psi, a_dpsi, ierr)
    call copy_var_in(y, var_u,   a_du,   ierr)
    call copy_var_in(y, var_zj,  a_dj,   ierr)
    call copy_var_in(y, var_w,   a_dw,   ierr)
    call copy_var_in(y, var_rho, a_drho, ierr)
    call copy_var_in(y, var_T,   a_dT,   ierr)
    ierr = 0
  end subroutine metriplectic_sweep_apply_full


  !====================================================================
  ! Layout helpers.
  ! Full-system rank-local layout: node-block i, variable v, mode m
  !   -> local index i*(n_var*n_tor) + (v-1)*n_tor + m   (1-based m).
  ! 1-var layout: i*n_tor + m. Ownership is aligned (same rank holds the
  ! corresponding rows of both layouts), so the copy is purely local —
  ! same assumption the pack_4v/nest->AIJ path validated in Slice A.
  !====================================================================
  subroutine copy_var_out(x_full, v, x_sub, ierr)
    use mod_parameters, only: n_var, n_tor
    Vec :: x_full, x_sub
    integer, intent(in) :: v
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ax(:), as(:)
    PetscInt :: n_local
    integer :: i, m, n_block_local, bs

    call VecGetLocalSize(x_full, n_local, ierr)
    bs = n_var * n_tor
    n_block_local = n_local / bs
    call VecGetArrayReadF90(x_full, ax, ierr)
    call VecGetArrayF90(x_sub, as, ierr)
    do i = 0, n_block_local - 1
      do m = 1, n_tor
        as(i*n_tor + m) = ax(i*bs + (v-1)*n_tor + m)
      enddo
    enddo
    call VecRestoreArrayReadF90(x_full, ax, ierr)
    call VecRestoreArrayF90(x_sub, as, ierr)
  end subroutine copy_var_out

  subroutine copy_var_in(y_full, v, y_sub, ierr)
    use mod_parameters, only: n_var, n_tor
    Vec :: y_full, y_sub
    integer, intent(in) :: v
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ay(:), as(:)
    PetscInt :: n_local
    integer :: i, m, n_block_local, bs

    call VecGetLocalSize(y_full, n_local, ierr)
    bs = n_var * n_tor
    n_block_local = n_local / bs
    call VecGetArrayF90(y_full, ay, ierr)
    call VecGetArrayReadF90(y_sub, as, ierr)
    do i = 0, n_block_local - 1
      do m = 1, n_tor
        ay(i*bs + (v-1)*n_tor + m) = as(i*n_tor + m)
      enddo
    enddo
    call VecRestoreArrayF90(y_full, ay, ierr)
    call VecRestoreArrayReadF90(y_sub, as, ierr)
  end subroutine copy_var_in


  !====================================================================
  ! Pack/unpack for nest->AIJ packed vectors (rank-local concatenation;
  ! same convention as the Slice-A analysis helpers).
  !====================================================================
  subroutine pack_2v(x1, x2, y, ierr)
    Vec :: x1, x2, y
    PetscErrorCode :: ierr
    PetscScalar, pointer :: a1(:), a2(:), ay(:)
    PetscInt :: n1, n2

    call VecGetLocalSize(x1, n1, ierr); call VecGetLocalSize(x2, n2, ierr)
    call VecGetArrayReadF90(x1, a1, ierr); call VecGetArrayReadF90(x2, a2, ierr)
    call VecGetArrayF90(y, ay, ierr)
    ay(1:n1)       = a1(1:n1)
    ay(n1+1:n1+n2) = a2(1:n2)
    call VecRestoreArrayReadF90(x1, a1, ierr); call VecRestoreArrayReadF90(x2, a2, ierr)
    call VecRestoreArrayF90(y, ay, ierr)
  end subroutine pack_2v

  subroutine unpack_2v(x, y1, y2, ierr)
    Vec :: x, y1, y2
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ax(:), a1(:), a2(:)
    PetscInt :: n1, n2

    call VecGetLocalSize(y1, n1, ierr); call VecGetLocalSize(y2, n2, ierr)
    call VecGetArrayReadF90(x, ax, ierr)
    call VecGetArrayF90(y1, a1, ierr); call VecGetArrayF90(y2, a2, ierr)
    a1(1:n1) = ax(1:n1)
    a2(1:n2) = ax(n1+1:n1+n2)
    call VecRestoreArrayReadF90(x, ax, ierr)
    call VecRestoreArrayF90(y1, a1, ierr); call VecRestoreArrayF90(y2, a2, ierr)
  end subroutine unpack_2v

  subroutine pack_4v(x1, x2, x3, x4, y, ierr)
    Vec :: x1, x2, x3, x4, y
    PetscErrorCode :: ierr
    PetscScalar, pointer :: a1(:), a2(:), a3(:), a4(:), ay(:)
    PetscInt :: n1, n2, n3, n4

    call VecGetLocalSize(x1, n1, ierr); call VecGetLocalSize(x2, n2, ierr)
    call VecGetLocalSize(x3, n3, ierr); call VecGetLocalSize(x4, n4, ierr)
    call VecGetArrayReadF90(x1, a1, ierr); call VecGetArrayReadF90(x2, a2, ierr)
    call VecGetArrayReadF90(x3, a3, ierr); call VecGetArrayReadF90(x4, a4, ierr)
    call VecGetArrayF90(y, ay, ierr)
    ay(1:n1)                   = a1(1:n1)
    ay(n1+1:n1+n2)             = a2(1:n2)
    ay(n1+n2+1:n1+n2+n3)       = a3(1:n3)
    ay(n1+n2+n3+1:n1+n2+n3+n4) = a4(1:n4)
    call VecRestoreArrayReadF90(x1, a1, ierr); call VecRestoreArrayReadF90(x2, a2, ierr)
    call VecRestoreArrayReadF90(x3, a3, ierr); call VecRestoreArrayReadF90(x4, a4, ierr)
    call VecRestoreArrayF90(y, ay, ierr)
  end subroutine pack_4v

  subroutine unpack_4v(x, y1, y2, y3, y4, ierr)
    Vec :: x, y1, y2, y3, y4
    PetscErrorCode :: ierr
    PetscScalar, pointer :: ax(:), a1(:), a2(:), a3(:), a4(:)
    PetscInt :: n1, n2, n3, n4

    call VecGetLocalSize(y1, n1, ierr); call VecGetLocalSize(y2, n2, ierr)
    call VecGetLocalSize(y3, n3, ierr); call VecGetLocalSize(y4, n4, ierr)
    call VecGetArrayReadF90(x, ax, ierr)
    call VecGetArrayF90(y1, a1, ierr); call VecGetArrayF90(y2, a2, ierr)
    call VecGetArrayF90(y3, a3, ierr); call VecGetArrayF90(y4, a4, ierr)
    a1(1:n1) = ax(1:n1)
    a2(1:n2) = ax(n1+1:n1+n2)
    a3(1:n3) = ax(n1+n2+1:n1+n2+n3)
    a4(1:n4) = ax(n1+n2+n3+1:n1+n2+n3+n4)
    call VecRestoreArrayReadF90(x, ax, ierr)
    call VecRestoreArrayF90(y1, a1, ierr); call VecRestoreArrayF90(y2, a2, ierr)
    call VecRestoreArrayF90(y3, a3, ierr); call VecRestoreArrayF90(y4, a4, ierr)
  end subroutine unpack_4v

#endif
end module mod_petsc_pc_metriplectic_apply
