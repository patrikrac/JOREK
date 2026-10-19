module mod_petsc_pc_mass_cheb
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_raw_csr, only: blockmv_attach
  implicit none
  private

  !--------------------------------------------------------------------
  !> A mass matrix B applied as B^-1 by a fixed-degree Chebyshev iteration,
  !! preconditioned by additive Schwarz (one subdomain per rank, overlap 1,
  !! local ICC(0) on a nested-dissection ordering): corrector 3's rho / T
  !! mass inverse in the SF sweep.
  !!
  !! The preconditioned C1 Hermite mass is mesh-independent (Wathen, IMA J.
  !! Numer. Anal. 7 (1987) 449). The degree follows kappa, 2 rho^d <= TARGET
  !! with rho = (sqrt k - 1)/(sqrt k + 1), from bounds of one Lanczos run per
  !! run (B is geometry-only), gated: the measured B-norm error must meet the
  !! bound, or the bounds are widened and the setup retries. Without norms
  !! the apply is a fixed linear operator, so FGMRES around it stays valid.
  !--------------------------------------------------------------------
  real*8, parameter  :: TARGET = 1.d-1   !< Chebyshev B-norm error bound per apply
  integer, parameter :: DEG_MIN = 2

  type, public :: mass_cheb_t
    logical :: ready = .false.
    KSP     :: ksp
    integer :: deg = 0
    real*8  :: lo = 0.d0, hi = 0.d0, err = 0.d0
  end type mass_cheb_t

  public :: mass_cheb_setup, mass_cheb_solve

contains

  !> Configure, estimate the bounds, choose the degree and gate it. Once per
  !! run (B's values never change). Collective on comm.
  subroutine mass_cheb_setup(MC, B, comm, label, nharm)
    use mod_parameters, only: n_tor
    type(mass_cheb_t), intent(inout) :: MC
    Mat, intent(in)              :: B
    integer, intent(in)          :: comm
    character(len=*), intent(in) :: label
    integer, intent(in), optional :: nharm   !< B's harmonics per node block (default n_tor)
    KSP :: est
    PC  :: pc
    PetscErrorCode :: ierr
    PetscInt :: its
    real*8 :: emax, emin, bound
    integer :: rank, k
    logical :: okb

    call MPI_Comm_rank(comm, rank, ierr)
    if (present(nharm)) then
      okb = blockmv_attach(B, nharm)          ! the iteration's matvec on the threads
    else
      okb = blockmv_attach(B, int(n_tor))
    endif

    call KSPCreate(comm, MC%ksp, ierr)
    call KSPSetOperators(MC%ksp, B, B, ierr)
    call KSPGetPC(MC%ksp, pc, ierr)
    call set_schwarz(pc)

    !--- bounds of P^-1 B from one preconditioned CG (Lanczos)
    call KSPCreate(comm, est, ierr)
    call KSPSetOperators(est, B, B, ierr)
    call KSPSetType(est, KSPCG, ierr)
    call KSPSetPC(est, pc, ierr)
    call KSPSetComputeSingularValues(est, PETSC_TRUE, ierr)
    call KSPSetTolerances(est, 1.d-10, 1.d-50, 1.d10, int(500, kind(its)), ierr)
    call KSPSetUp(est, ierr)
    call check(est, B, emax)                   ! drives the CG; emax is a dummy here
    call KSPGetIterationNumber(est, its, ierr)
    call KSPComputeExtremeSingularValues(est, emax, emin, ierr)
    call KSPDestroy(est, ierr)
    MC%lo = 0.9d0 * emin                       ! Lanczos brackets from inside: widen
    MC%hi = 1.05d0 * emax

    !--- degree from kappa, gated on the measured error
    call KSPSetType(MC%ksp, KSPCHEBYSHEV, ierr)
    call KSPSetNormType(MC%ksp, KSP_NORM_NONE, ierr)
    do k = 1, 6
      MC%deg = cheb_degree(MC%hi / MC%lo)
      bound = 2.d0 * rho(MC%hi / MC%lo)**MC%deg
      call KSPChebyshevSetEigenvalues(MC%ksp, MC%hi, MC%lo, ierr)
      call KSPSetTolerances(MC%ksp, 1.d-50, 1.d-50, 1.d10, int(MC%deg, kind(its)), ierr)
      call KSPSetUp(MC%ksp, ierr)
      call check(MC%ksp, B, MC%err)
      if (MC%err <= bound) exit
      MC%lo = 0.8d0 * MC%lo                    ! missed: lambda_min over-estimated
      MC%hi = 1.1d0 * MC%hi
    enddo
    if (MC%err > bound) then
      if (rank == 0) write(*,'(A,A,A,ES9.2,A,ES9.2,A)') "[Physics PC]   FATAL: ", trim(label), &
        ": Chebyshev error ", MC%err, " above its bound ", bound, " after 6 widenings."
      flush(6)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    MC%ready = .true.
    if (rank == 0) write(*,'(A,A,A,I0,A,2ES10.3,A,F6.2,A,I0,A,ES9.2,A,ES9.2,A,L1)') &
      "[Physics PC]   ", trim(label), ": Chebyshev(", MC%deg, &
      ") + ASM(ovl 1, ICC(0) ND), lambda in [", MC%lo, MC%hi, "], kappa ", MC%hi / MC%lo, &
      ", CG its ", its, ", B-norm err ", MC%err, " (bound ", bound, "), block kernel ", okb

  contains

    subroutine set_schwarz(pc_)
      PC :: pc_
      KSP, pointer :: sub(:)
      PC :: spc
      PetscInt :: nsub, first, i
      call PCSetType(pc_, PCASM, ierr)
      call PCASMSetOverlap(pc_, int(1, kind(its)), ierr)
      call PCASMSetType(pc_, PC_ASM_BASIC, ierr)   ! symmetric, as CG and Chebyshev need
      call PCSetUp(pc_, ierr)
      call PCASMGetSubKSP(pc_, nsub, first, sub, ierr)
      do i = 1, nsub
        call KSPSetType(sub(i), KSPPREONLY, ierr)
        call KSPGetPC(sub(i), spc, ierr)
        call PCSetType(spc, PCICC, ierr)
        call PCFactorSetMatOrderingType(spc, MATORDERINGND, ierr)
      enddo
      call PCASMRestoreSubKSP(pc_, nsub, first, sub, ierr)
    end subroutine set_schwarz

    real*8 function rho(kk)
      real*8, intent(in) :: kk
      rho = (sqrt(kk) - 1.d0) / (sqrt(kk) + 1.d0)
    end function rho

    integer function cheb_degree(kk)
      real*8, intent(in) :: kk
      cheb_degree = max(DEG_MIN, ceiling(log(2.d0 / TARGET) / log(1.d0 / max(rho(kk), 1.d-12))))
    end function cheb_degree

  end subroutine mass_cheb_setup


  !> y = B^-1 x, to the chosen degree
  subroutine mass_cheb_solve(MC, x, y)
    type(mass_cheb_t), intent(inout) :: MC
    Vec :: x, y
    PetscErrorCode :: ierr
    call KSPSolve(MC%ksp, x, y, ierr)
  end subroutine mass_cheb_solve

  !> ||x - xt||_B / ||xt||_B for x = solve(B xt), xt random (PETSc's fixed
  !! seed): the norm the Chebyshev bound holds in.
  subroutine check(ksp, B, err)
    KSP :: ksp
    Mat :: B
    real*8, intent(out) :: err
    Vec :: xt, bv, x
    PetscRandom :: rnd
    PetscErrorCode :: ierr
    real*8 :: nx
    integer :: cm
    call MatCreateVecs(B, xt, bv, ierr)
    call VecDuplicate(xt, x, ierr)
    call PetscObjectGetComm(B, cm, ierr)
    call PetscRandomCreate(cm, rnd, ierr)
    call VecSetRandom(xt, rnd, ierr)
    call MatMult(B, xt, bv, ierr)
    call VecDot(xt, bv, nx, ierr)                ! ||xt||_B^2
    call KSPSolve(ksp, bv, x, ierr)
    call VecAXPY(x, -1.d0, xt, ierr)             ! error e
    call MatMult(B, x, bv, ierr)
    call VecDot(x, bv, err, ierr)                ! ||e||_B^2
    err = sqrt(max(err, 0.d0) / nx)
    call PetscRandomDestroy(rnd, ierr)
    call VecDestroy(xt, ierr); call VecDestroy(bv, ierr); call VecDestroy(x, ierr)
  end subroutine check

#endif
end module mod_petsc_pc_mass_cheb
