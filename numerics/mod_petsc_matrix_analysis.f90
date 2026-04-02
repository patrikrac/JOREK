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
            petsc_mat_full_spectrum, &
            petsc_mat_sweep_robust_spectrum
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
  !! (Hermitian) Mat. Uses Direct LU Factorization for the smallest.
  !--------------------------------------------------------------------
  subroutine petsc_mat_eig_bounds(A, lam_min, lam_max)
    Mat,       intent(in)  :: A
    PetscReal, intent(out) :: lam_min, lam_max

    EPS            :: eps
    ST             :: st
    KSP            :: st_ksp
    PC             :: st_pc
    PetscInt       :: nconv, eps_reason
    KSPConvergedReason :: ksp_reason
    PetscScalar    :: kr, ki   ! ki required by EPSGetEigenvalue interface; always 0 for EPS_HEP
    PetscErrorCode :: ierr
    integer        :: comm, my_id, mpierr

    lam_min = 0.0d0
    lam_max = 0.0d0

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    ! =================================================================
    ! 1. LARGEST EIGENVALUE (Standard Krylov - Fast, low memory)
    ! =================================================================
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
      call EPSGetConvergedReason(eps, eps_reason, ierr)
      write(*,'(A,I0)') "[EPS] WARNING: Largest eigenvalue failed. EPS Reason: ", eps_reason
    endif
    call EPSDestroy(eps, ierr)

    ! =================================================================
    ! 2. SMALLEST EIGENVALUE (Shift-and-Invert with EXACT Direct Solver)
    ! =================================================================
    call EPSCreate(comm, eps, ierr)
    call EPSSetOperators(eps, A, PETSC_NULL_MAT, ierr)
    call EPSSetProblemType(eps, EPS_HEP, ierr)
    call EPSSetType(eps, EPSKRYLOVSCHUR, ierr)
    
    ! Enable Shift-and-Invert
    call EPSGetST(eps, st, ierr)
    call STSetType(st, STSINVERT, ierr)
    
    ! Enforce Direct Factorization (LU)
    call STGetKSP(st, st_ksp, ierr)
    call KSPSetType(st_ksp, KSPPREONLY, ierr)
    call KSPGetPC(st_ksp, st_pc, ierr)
    call PCSetType(st_pc, PCLU, ierr)
    
    ! Use MUMPS for parallel direct solve. 
    ! (If running serially without MUMPS, remove the line below or use MATSOLVERPETSC)
    call PCFactorSetMatSolverType(st_pc, MATSOLVERMUMPS, ierr)
    
    ! IMPORTANT: If your matrix is singular (has a 0 eigenvalue), setting a target 
    ! of exactly 0.0 will cause LU factorization to crash with a "Zero Pivot" error.
    ! We shift by a tiny offset to safely avoid exactly hitting a zero eigenvalue.
    call EPSSetTarget(eps, -1.0d-6, ierr)
    call EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE, ierr)
    
    call EPSSetDimensions(eps, 1, PETSC_DEFAULT_INTEGER, PETSC_DEFAULT_INTEGER, ierr)
    call EPSSetFromOptions(eps, ierr)
    call EPSSolve(eps, ierr)
    
    call EPSGetConverged(eps, nconv, ierr)
    if (nconv > 0) then
      call EPSGetEigenvalue(eps, 0, kr, ki, ierr)
      lam_min = real(kr, kind=8)
    elseif (my_id == 0) then
      call EPSGetConvergedReason(eps, eps_reason, ierr)
      write(*,'(A,I0)') "[EPS] ERROR: Smallest eigenvalue failed. EPS Reason: ", eps_reason
      
      ! Diagnostics: Check if the direct solver (MUMPS) is what actually failed
      call KSPGetConvergedReason(st_ksp, ksp_reason, ierr)
      if (ksp_reason < 0) then
        write(*,'(A,I0)') "      -> The Direct Solver (LU/MUMPS) failed! KSP Reason: ", ksp_reason
        if (ksp_reason == -8 .or. ksp_reason == -9) &
          write(*,'(A)')  "      -> (Likely cause: Matrix is singular/has a null space, causing a zero pivot)"
      endif
    endif
    call EPSDestroy(eps, ierr)

    if (my_id == 0) &
      write(*,'(A,ES14.6,A,ES14.6)') "[EPS] lam_min = ", lam_min, "  lam_max = ", lam_max
  end subroutine petsc_mat_eig_bounds


  !--------------------------------------------------------------------
  !> Estimate the spectral condition number kappa = lam_max / lam_min
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
      kappa = -1.0d0   ! sentinel: valid kappa >= 1; -1 signals lam_min not converged
      if (my_id == 0) write(*,'(A)') "[EPS] kappa(A) : unavailable (lam_min not converged)"
    endif
  end subroutine petsc_mat_cond_estimate


  !--------------------------------------------------------------------
  !> Compute a partial eigenvalue spectrum of a Mat and write to file.
  !> Uses Shift-and-Invert + MUMPS for robust distributed solving.
  !--------------------------------------------------------------------
  subroutine petsc_mat_full_spectrum(A, label, n_eigs, symmetric)
    implicit none

    Mat,             intent(in) :: A
    character(len=*),intent(in) :: label
    integer,         intent(in) :: n_eigs
    logical,         intent(in) :: symmetric

    EPS                 :: eps
    ST                  :: st
    KSP                 :: ksp
    PC                  :: pc
    PetscInt            :: M, N, nev_req, nconv, eps_reason, i_eps
    PetscScalar         :: kr, ki, target_val
    PetscErrorCode      :: ierr
    integer             :: comm, my_id, mpierr, i, iunit
    real*8, allocatable :: eig_r(:), eig_i(:)
    character(len=512)  :: filename

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MatGetSize(A, M, N, ierr)

    nev_req = n_eigs
    if (nev_req <= 0 .or. nev_req >= M) then
      nev_req = M/2
      !nev_req = min(M - 1, 50) ! Capped at 50 for speed and safety
      if (my_id == 0) then
        write(*,'(A)') "[EPS] WARNING: Cannot reliably compute FULL spectrum."
        write(*,'(A,I0,A)') "[EPS] Computing ", nev_req, " extreme eigenvalues instead."
      endif
    endif

    call EPSCreate(comm, eps, ierr)
    call EPSSetOperators(eps, A, PETSC_NULL_MAT, ierr)
    
    if (symmetric) then
      call EPSSetProblemType(eps, EPS_HEP, ierr)
    else
      call EPSSetProblemType(eps, EPS_NHEP, ierr)
    endif

    call EPSSetType(eps, EPSKRYLOVSCHUR, ierr)
    call EPSSetDimensions(eps, nev_req, PETSC_DEFAULT_INTEGER, PETSC_DEFAULT_INTEGER, ierr)

    ! -------------------------------------------------------------------------
    ! ROBUST ILL-CONDITIONED SETUP: Shift-and-Invert + MUMPS
    ! -------------------------------------------------------------------------
    ! 1. Set a target value. Shift-and-invert finds eigenvalues closest to this.
    target_val = 0.0  ! Change this to sweep different parts of the spectrum
    call EPSSetTarget(eps, target_val, ierr)
    call EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE, ierr)

    ! 2. Extract Spectral Transformation (ST) and set to Shift-and-Invert
    call EPSGetST(eps, st, ierr)
    call STSetType(st, STSINVERT, ierr)

    ! 3. Extract Linear Solver (KSP) and tell it to only apply the preconditioner
    call STGetKSP(st, ksp, ierr)
    call KSPSetType(ksp, KSPPREONLY, ierr)

    ! 4. Extract Preconditioner (PC) and set it to Exact LU Factorization
    call KSPGetPC(ksp, pc, ierr)
    call PCSetType(pc, PCLU, ierr)

    ! 5. Tell the LU Factorization to use MUMPS (handles MPIBAIJ perfectly)
    call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
    ! -------------------------------------------------------------------------

    call EPSSetFromOptions(eps, ierr)
    if (my_id == 0) write(*,*) "[EPS] Solving partial spectrum (Shift-and-Invert via MUMPS) ---"
    
    call EPSSolve(eps, ierr)
    call EPSGetConverged(eps, nconv, ierr)

    if (my_id == 0) then
      if (nconv > 0) then
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
        write(iunit,'(A)')        "#        Re(lambda)           Im(lambda)"
        do i = 1, nconv
          write(iunit,'(2X,ES22.14,2X,ES22.14)') eig_r(i), eig_i(i)
        enddo
        close(iunit)

        write(*,'(A,I0,A,I0,A,A)') "[EPS] Converged ", nconv, " / ", nev_req, " -> ", trim(filename)
        deallocate(eig_r, eig_i)
      else
        call EPSGetConvergedReason(eps, eps_reason, ierr)
        write(*,'(A,I0)') "[EPS] ERROR: 0 eigenvalues converged. EPS Reason: ", eps_reason
      endif
    endif
    call EPSDestroy(eps, ierr)
  end subroutine petsc_mat_full_spectrum

  subroutine sort_eigs_by_real(eig_r, eig_i, n)
    integer, intent(in)   :: n
    real*8, intent(inout) :: eig_r(n), eig_i(n)
    integer :: i, j
    real*8  :: tmp_r, tmp_i

    do i = 1, n - 1
      do j = i + 1, n
        if (eig_r(j) < eig_r(i)) then
          tmp_r = eig_r(i); eig_r(i) = eig_r(j); eig_r(j) = tmp_r
          tmp_i = eig_i(i); eig_i(i) = eig_i(j); eig_i(j) = tmp_i
        endif
      enddo
    enddo
  end subroutine sort_eigs_by_real

  !--------------------------------------------------------------------
  !> Compute the eigenvalue spectrum by sweeping across a range.
  !> INTEGRATED CLUSTER LOGIC: Uses expanded Krylov subspaces (3x NCV) 
  !> and safe-shifting to easily untangle dense clusters around zero.
  !> Filters overlapping duplicates and writes to a single file.
  !--------------------------------------------------------------------
  subroutine petsc_mat_sweep_robust_spectrum(A, label, target_start, target_end, n_shifts, nev_per_shift, symmetric)
    implicit none

    Mat,             intent(in) :: A
    character(len=*),intent(in) :: label
    real(kind=8),    intent(in) :: target_start, target_end
    integer,         intent(in) :: n_shifts, nev_per_shift
    logical,         intent(in) :: symmetric

    EPS                 :: eps
    ST                  :: st
    KSP                 :: ksp
    PC                  :: pc
    PetscInt            :: M, N, nev_req, ncv_req, nconv, eps_reason, i_eps
    PetscScalar         :: kr, ki, target_val, safe_target
    PetscErrorCode      :: ierr
    integer             :: comm, my_id, mpierr, i, iunit, i_shift
    integer             :: total_unique, j
    logical             :: is_duplicate

    real*8, allocatable :: eig_r_all(:), eig_i_all(:)
    real*8              :: r_val, i_val, diff
    real*8, parameter   :: TOL_DUP = 1.0d-6  ! Tolerance to detect duplicates
    character(len=512)  :: filename

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MatGetSize(A, M, N, ierr)

    ! 1. Setup Dimensions with CLUSTER EXPANSION
    nev_req = min(nev_per_shift, M - 1)
    ncv_req = 3 * nev_req           ! Force 3x subspace to separate clusters
    if (ncv_req > M) ncv_req = M    ! Cap at matrix size

    ! Allocate maximum possible space for unique eigenvalues
    allocate(eig_r_all(M), eig_i_all(M))
    total_unique = 0

    if (my_id == 0) then
      write(*,'(A)') "[EPS] ========================================================"
      write(*,'(A,A)') "[EPS] Robust Spectrum Sweep for matrix: ", trim(label)
      write(*,'(A,I0)') "[EPS] Matrix size     : ", M
      write(*,'(A,I0,A,I0)') "[EPS] Shifts          : ", n_shifts, " | NEV per shift: ", nev_req
      write(*,'(A,I0)') "[EPS] Expanded NCV    : ", ncv_req, " (Cluster separation active)"
      write(*,'(A)') "[EPS] ========================================================"
    endif

    ! -------------------------------------------------------------------------
    ! SWEEP LOOP
    ! -------------------------------------------------------------------------
    do i_shift = 1, n_shifts
      
      ! Calculate the mathematical target
      if (n_shifts == 1) then
        target_val = target_start
      else
        target_val = target_start + (target_end - target_start) * real(i_shift - 1, kind=8) / real(n_shifts - 1, kind=8)
      endif

      ! 2. SAFE SHIFT LOGIC (Prevent MUMPS Singularity on exactly 0.0)
      safe_target = target_val
      if (abs(safe_target) < 1.0d-12) then
        safe_target = 1.0d-5
      endif

      call EPSCreate(comm, eps, ierr)
      call EPSSetOperators(eps, A, PETSC_NULL_MAT, ierr)

      if (symmetric) then
        call EPSSetProblemType(eps, EPS_HEP, ierr)
      else
        call EPSSetProblemType(eps, EPS_NHEP, ierr)
      endif

      call EPSSetType(eps, EPSKRYLOVSCHUR, ierr)
      
      ! Apply cluster-busting dimensions
      call EPSSetDimensions(eps, nev_req, ncv_req, PETSC_DEFAULT_INTEGER, ierr)

      ! Set the safe target
      call EPSSetTarget(eps, safe_target, ierr)
      call EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE, ierr)

      ! Robust Shift-and-Invert via MUMPS
      call EPSGetST(eps, st, ierr)
      call STSetType(st, STSINVERT, ierr)
      call STGetKSP(st, ksp, ierr)
      call KSPSetType(ksp, KSPPREONLY, ierr)
      call KSPGetPC(ksp, pc, ierr)
      call PCSetType(pc, PCLU, ierr)
      call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)

      call EPSSetFromOptions(eps, ierr)

      if (my_id == 0) write(*,'(A,I0,A,I0,A,ES12.4)') &
           "[EPS] Solving Shift ", i_shift, "/", n_shifts, " | Target: ", real(safe_target, kind=8)

      call EPSSolve(eps, ierr)
      call EPSGetConverged(eps, nconv, ierr)

      ! -----------------------------------------------------------------------
      ! Extract and Filter Duplicates
      ! -----------------------------------------------------------------------
      if (nconv > 0) then
        do i_eps = 0, nconv - 1
          call EPSGetEigenvalue(eps, i_eps, kr, ki, ierr)
          r_val = real(kr, kind=8)
          i_val = real(ki, kind=8)

          ! Check against previously found eigenvalues
          is_duplicate = .false.
          do j = 1, total_unique
            diff = abs(r_val - eig_r_all(j)) + abs(i_val - eig_i_all(j))
            if (diff < TOL_DUP) then
              is_duplicate = .true.
              exit
            endif
          enddo

          ! Append new unique eigenvalues
          if (.not. is_duplicate .and. total_unique < M) then
            total_unique = total_unique + 1
            eig_r_all(total_unique) = r_val
            eig_i_all(total_unique) = i_val
          endif
        enddo
      else
        call EPSGetConvergedReason(eps, eps_reason, ierr)
        if (my_id == 0) write(*,'(A,I0)') "[EPS] WARNING: 0 converged for this target. Reason: ", eps_reason
      endif

      call EPSDestroy(eps, ierr)
    enddo

    ! -------------------------------------------------------------------------
    ! WRITE CONSOLIDATED FILE
    ! -------------------------------------------------------------------------
    if (my_id == 0) then
      ! (Assuming sort_eigs_by_real is defined elsewhere in your module)
      if (total_unique > 0) call sort_eigs_by_real(eig_r_all, eig_i_all, total_unique)

      write(filename, '(A,A)') trim(label), "_full_spectrum.dat"
      open(newunit=iunit, file=trim(filename), status='replace', action='write')
      write(iunit,'(A,A)')      "# Robust Sweep of matrix       : ", trim(label)
      write(iunit,'(A,I0)')     "# Matrix size                  : ", M
      write(iunit,'(A,I0)')     "# Total Unique Found           : ", total_unique
      write(iunit,'(A)')        "#        Re(lambda)           Im(lambda)"
      do i = 1, total_unique
        write(iunit,'(2X,ES22.14,2X,ES22.14)') eig_r_all(i), eig_i_all(i)
      enddo
      close(iunit)

      write(*,'(A)') "[EPS] ========================================================"
      write(*,'(A,I0,A,I0,A,A)') "[EPS] Sweep Complete! Found ", total_unique, &
           " unique eigenvalues / ", M, " -> ", trim(filename)
    endif

    deallocate(eig_r_all, eig_i_all)
  end subroutine petsc_mat_sweep_robust_spectrum

#endif

#endif
end module mod_petsc_matrix_analysis
