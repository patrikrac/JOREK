module mod_petsc_matrix_analysis
!----------------------------------------------------------------------
! Tools for analysing PETSc Mat objects: structural info, matrix norms,
! and (when compiled with SLEPc) eigenvalue computation.
!
! All print statements are guarded to rank-0 only.
! Intended for diagnostic/debug use, not production runs.
!
! Public interface (USE_PETSC):
!   petsc_mat_print_info (A, label)           -- NNZ, block size, memory
!   petsc_mat_norms      (A, label)           -- Frobenius / 1-norm / inf-norm
!   petsc_mat_diff_norm  (A, B, label, norm)  -- ||A - B||_F
!
! Additional (USE_SLEPC):
!   petsc_mat_eig_bounds    (A, lam_min, lam_max) -- extreme eigenvalues via EPS (symmetric)
!   petsc_mat_cond_estimate (A, kappa)             -- kappa = lam_max / lam_min  (symmetric)
!   petsc_mat_full_spectrum (A, label, n_eigs, symmetric)
!     -- compute n_eigs eigenvalues (n_eigs <= 0 means all) and write sorted to
!        {label}_spectrum.dat.  Works for both symmetric and non-symmetric matrices.
!----------------------------------------------------------------------
#ifdef USE_PETSC
#include "petsc/finclude/petsc.h"
  use petsc
#ifdef USE_SLEPC
#include "slepc/finclude/slepceps.h"
  use slepceps
#endif
  implicit none
  private
  public :: petsc_mat_print_info, &
            petsc_mat_norms,      &
            petsc_mat_diff_norm
#ifdef USE_SLEPC
  public :: petsc_mat_eig_bounds,    &
            petsc_mat_cond_estimate, &
            petsc_mat_full_spectrum
#endif

contains

  !--------------------------------------------------------------------
  !> Print structural info for a PETSc Mat: global size, block size,
  !! NNZ used/allocated, and allocated memory.
  !! Only rank-0 in the matrix communicator prints.
  !--------------------------------------------------------------------
  subroutine petsc_mat_print_info(A, label)
    Mat,             intent(in) :: A
    character(len=*),intent(in) :: label

    PetscErrorCode :: ierr
    PetscInt       :: M, N, bs
    MatInfo        :: info(MAT_INFO_SIZE)
    integer        :: comm, my_id, mpierr

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    call MatGetSize(A, M, N, ierr)
    call MatGetBlockSize(A, bs, ierr)
    call MatGetInfo(A, MAT_GLOBAL_SUM, info, ierr)

    if (my_id == 0) then
      write(*,'(A,A)')        "[MatInfo] ", trim(label)
      write(*,'(A,I0,A,I0)')  "  Global size : ", M, " x ", N
      write(*,'(A,I0)')       "  Block size  : ", bs
      write(*,'(A,I0)')       "  NNZ used    : ", int(info(MAT_INFO_NZ_USED))
      write(*,'(A,I0)')       "  NNZ alloc   : ", int(info(MAT_INFO_NZ_ALLOCATED))
      ! MAT_INFO_MEMORY is not populated for distributed (MPIBAIJ) matrix types
      if (info(MAT_INFO_MEMORY) > 0) &
        write(*,'(A,ES12.4)') "  Memory (B)  : ", info(MAT_INFO_MEMORY)
    endif
  end subroutine petsc_mat_print_info


  !--------------------------------------------------------------------
  !> Compute and print Frobenius, 1-norm, and infinity-norm of a Mat.
  !--------------------------------------------------------------------
  subroutine petsc_mat_norms(A, label)
    Mat,             intent(in) :: A
    character(len=*),intent(in) :: label

    PetscErrorCode :: ierr
    PetscReal      :: r_norm_f, r_norm_1, r_norm_inf
    integer        :: comm, my_id, mpierr

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    call MatNorm(A, NORM_FROBENIUS, r_norm_f,   ierr)
    call MatNorm(A, NORM_1,         r_norm_1,   ierr)
    call MatNorm(A, NORM_INFINITY,  r_norm_inf, ierr)

    if (my_id == 0) then
      write(*,'(A,A)')      "[MatNorm] ", trim(label)
      write(*,'(A,ES14.6)') "  ||A||_F   = ", r_norm_f
      write(*,'(A,ES14.6)') "  ||A||_1   = ", r_norm_1
      write(*,'(A,ES14.6)') "  ||A||_inf = ", r_norm_inf
    endif
  end subroutine petsc_mat_norms


  !--------------------------------------------------------------------
  !> Compute ||A - B||_F and optionally return it in norm_out.
  !! A and B must have the same sparsity pattern (SAME_NONZERO_PATTERN).
  !! Uses a temporary duplicate — does not modify A or B.
  !--------------------------------------------------------------------
  subroutine petsc_mat_diff_norm(A, B, label, norm_out)
    Mat,             intent(in)  :: A, B
    character(len=*),intent(in)  :: label
    PetscReal,       intent(out) :: norm_out

    Mat            :: C
    PetscErrorCode :: ierr
    integer        :: comm, my_id, mpierr

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    call MatDuplicate(A, MAT_COPY_VALUES, C, ierr)
    call MatAXPY(C, -1.0d0, B, SAME_NONZERO_PATTERN, ierr)
    call MatNorm(C, NORM_FROBENIUS, norm_out, ierr)
    call MatDestroy(C, ierr)

    if (my_id == 0) &
      write(*,'(A,A,A,ES14.6)') "[MatDiff] ", trim(label), "  ||A-B||_F = ", norm_out
  end subroutine petsc_mat_diff_norm


#ifdef USE_SLEPC
  !--------------------------------------------------------------------
  !> Compute the smallest and largest real eigenvalues of a symmetric
  !! (Hermitian) Mat using SLEPc EPS with the Krylov-Schur method.
  !!
  !! Assumes A is symmetric positive definite (EPS_HEP problem type).
  !! Uses two separate EPS contexts: standard Krylov-Schur for lam_max,
  !! and STSINVERT shift-and-invert for lam_min (requires a direct solver).
  !! On convergence failure, outputs are set to 0 and a warning is printed.
  !--------------------------------------------------------------------
  subroutine petsc_mat_eig_bounds(A, lam_min, lam_max)
    Mat,       intent(in)  :: A
    PetscReal, intent(out) :: lam_min, lam_max

    EPS            :: eps
    ST             :: st
    KSP            :: st_ksp
    PC             :: st_pc
    PetscInt       :: nconv
    PetscScalar    :: kr, ki
    PetscErrorCode :: ierr
    integer        :: comm, my_id, mpierr

    lam_min = 0.0d0
    lam_max = 0.0d0

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    ! --- Largest eigenvalue (standard Krylov-Schur, no ST needed) ---
    call EPSCreate(comm, eps, ierr)
    call EPSSetOperators(eps, A, PETSC_NULL_MAT, ierr)
    call EPSSetProblemType(eps, EPS_HEP, ierr)
    call EPSSetType(eps, EPSKRYLOVSCHUR, ierr)
    call EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL, ierr)
    call EPSSetDimensions(eps, 1, PETSC_DEFAULT_INTEGER, PETSC_DEFAULT_INTEGER, ierr)
    call EPSSetFromOptions(eps, ierr)
    call EPSSolve(eps, ierr)
    call EPSGetConverged(eps, nconv, ierr)
    if (nconv > 0) then
      call EPSGetEigenvalue(eps, 0, kr, ki, ierr)
      lam_max = real(kr, kind=8)
    elseif (my_id == 0) then
      write(*,'(A)') "[EPS] WARNING: largest eigenvalue did not converge"
    endif
    call EPSDestroy(eps, ierr)

    ! --- Smallest eigenvalue: shift-and-invert with MUMPS direct solver.
    call EPSCreate(comm, eps, ierr)
    call EPSSetOperators(eps, A, PETSC_NULL_MAT, ierr)
    call EPSSetProblemType(eps, EPS_HEP, ierr)
    call EPSSetType(eps, EPSKRYLOVSCHUR, ierr)
    call EPSGetST(eps, st, ierr)
    call STSetType(st, STSINVERT, ierr)
    call STGetKSP(st, st_ksp, ierr)
    call KSPSetType(st_ksp, KSPPREONLY, ierr)
    call KSPGetPC(st_ksp, st_pc, ierr)
    call PCSetType(st_pc, PCLU, ierr)
    call PCFactorSetMatSolverType(st_pc, MATSOLVERMUMPS, ierr)
    call EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE, ierr)
    call EPSSetDimensions(eps, 1, PETSC_DEFAULT_INTEGER, PETSC_DEFAULT_INTEGER, ierr)
    call EPSSetFromOptions(eps, ierr)
    call EPSSolve(eps, ierr)
    call EPSGetConverged(eps, nconv, ierr)
    if (nconv > 0) then
      call EPSGetEigenvalue(eps, 0, kr, ki, ierr)
      lam_min = real(kr, kind=8)
    elseif (my_id == 0) then
      write(*,'(A)') "[EPS] WARNING: smallest eigenvalue did not converge"
    endif
    call EPSDestroy(eps, ierr)

    if (my_id == 0) &
      write(*,'(A,ES14.6,A,ES14.6)') "[EPS] lam_min = ", lam_min, "  lam_max = ", lam_max
  end subroutine petsc_mat_eig_bounds


  !--------------------------------------------------------------------
  !> Estimate the spectral condition number kappa = lam_max / lam_min
  !! for a symmetric positive definite Mat.
  !--------------------------------------------------------------------
  subroutine petsc_mat_cond_estimate(A, kappa)
    Mat,       intent(in)  :: A
    PetscReal, intent(out) :: kappa

    PetscReal      :: lam_min, lam_max
    integer        :: comm, my_id, mpierr
    PetscErrorCode :: ierr

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    call petsc_mat_eig_bounds(A, lam_min, lam_max)
    if (lam_min > 0.0d0) then
      kappa = lam_max / lam_min
      if (my_id == 0) write(*,'(A,ES14.6)') "[EPS] kappa(A) = ", kappa
    else
      kappa = 0.0d0
      if (my_id == 0) write(*,'(A)') "[EPS] kappa(A) : unavailable (lam_min not converged)"
    endif
  end subroutine petsc_mat_cond_estimate


  !--------------------------------------------------------------------
  !> Compute the eigenvalue spectrum of a Mat and write it to a file.
  !!
  !! @param A         PETSc Mat to analyse
  !! @param label     Short identifier; output file is {label}_spectrum.dat
  !! @param n_eigs    Number of eigenvalues to compute.
  !!                  n_eigs <= 0 requests the full spectrum (all n eigenvalues).
  !! @param symmetric .true. → use EPS_HEP (Hermitian/symmetric problem).
  !!                  .false. → use EPS_NHEP (non-symmetric); eigenvalues may
  !!                            be complex; both Re and Im parts are written.
  !!
  !! Output file columns: Re(lambda)  Im(lambda), sorted by Re(lambda) ascending.
  !! On convergence failure, the file contains only the converged subset.
  !--------------------------------------------------------------------
  subroutine petsc_mat_full_spectrum(A, label, n_eigs, symmetric)
    Mat,             intent(in) :: A
    character(len=*),intent(in) :: label
    integer,         intent(in) :: n_eigs
    logical,         intent(in) :: symmetric

    EPS               :: eps
    PetscInt          :: M, N, nev_req, nconv, i_eps
    PetscScalar       :: kr, ki
    PetscErrorCode    :: ierr
    integer           :: comm, my_id, mpierr, i, iunit
    real*8, allocatable :: eig_r(:), eig_i(:)
    character(len=512)  :: filename

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MatGetSize(A, M, N, ierr)

    ! Number of eigenvalues to request
    nev_req = n_eigs
    if (nev_req <= 0) nev_req = M   ! full spectrum

    if (my_id == 0) &
      write(*,'(A,A,A,I0,A,I0,A)') &
        "[EPS] Computing spectrum of ", trim(label), &
        " (", nev_req, " / ", M, " eigenvalues) ..."

    call EPSCreate(comm, eps, ierr)
    call EPSSetOperators(eps, A, PETSC_NULL_MAT, ierr)
    if (symmetric) then
      call EPSSetProblemType(eps, EPS_HEP, ierr)
    else
      call EPSSetProblemType(eps, EPS_NHEP, ierr)
    endif

    if (nev_req == M) then
      ! Full spectrum: EPSLAPACK (dense) — no Krylov subspace constraint.
      ! Memory cost is O(M^2); only practical for moderate matrix sizes.
      call EPSSetType(eps, EPSLAPACK, ierr)
    else
      ! Partial spectrum: Krylov-Schur requires nev < M (ncv >= nev+1 <= M).
      call EPSSetType(eps, EPSKRYLOVSCHUR, ierr)
      call EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL, ierr)
      call EPSSetDimensions(eps, nev_req, PETSC_DEFAULT_INTEGER, PETSC_DEFAULT_INTEGER, ierr)
    endif

    ! Allow runtime override (-eps_type, -eps_tol, etc.)
    call EPSSetFromOptions(eps, ierr)

    call EPSSolve(eps, ierr)
    call EPSGetConverged(eps, nconv, ierr)

    ! Collect eigenvalues on rank 0, sort, and write to file
    if (my_id == 0) then
      allocate(eig_r(nconv), eig_i(nconv))
      do i_eps = 0, nconv - 1
        call EPSGetEigenvalue(eps, i_eps, kr, ki, ierr)
        eig_r(i_eps+1) = real(kr, kind=8)
        eig_i(i_eps+1) = real(ki, kind=8)
      enddo

      call sort_eigs_by_real(eig_r, eig_i, int(nconv))

      write(filename, '(A,A)') trim(label), "_spectrum.dat"
      open(newunit=iunit, file=trim(filename), status='replace', action='write')
      write(iunit,'(A,A)')      "# Spectrum of matrix: ", trim(label)
      write(iunit,'(A,I0)')     "# Matrix size        : ", M
      write(iunit,'(A,I0)')     "# Requested          : ", nev_req
      write(iunit,'(A,I0)')     "# Converged          : ", nconv
      write(iunit,'(A,L1)')     "# Symmetric (HEP)    : ", symmetric
      write(iunit,'(A)')        "#"
      write(iunit,'(A)')        "#        Re(lambda)           Im(lambda)"
      do i = 1, nconv
        write(iunit,'(2X,ES22.14,2X,ES22.14)') eig_r(i), eig_i(i)
      enddo
      close(iunit)

      write(*,'(A,I0,A,I0,A,A)') &
        "[EPS] Converged ", nconv, " / ", nev_req, " eigenvalues -> ", trim(filename)
      deallocate(eig_r, eig_i)
    endif

    call EPSDestroy(eps, ierr)
  end subroutine petsc_mat_full_spectrum


  !--------------------------------------------------------------------
  ! Private helper: sort eigenvalue arrays by Re(lambda) ascending
  ! using insertion sort (adequate for the sizes expected here).
  !--------------------------------------------------------------------
  subroutine sort_eigs_by_real(er, ei, n)
    integer, intent(in)    :: n
    real*8,  intent(inout) :: er(n), ei(n)
    integer :: i, j
    real*8  :: tr, ti

    do i = 2, n
      tr = er(i);  ti = ei(i)
      j  = i - 1
      do while (j >= 1 .and. er(j) > tr)
        er(j+1) = er(j);  ei(j+1) = ei(j)
        j = j - 1
      enddo
      er(j+1) = tr;  ei(j+1) = ti
    enddo
  end subroutine sort_eigs_by_real
#endif

#endif
end module mod_petsc_matrix_analysis
