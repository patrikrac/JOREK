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
!   petsc_mat_convert_spectrum (A, label, symmetric)
!     -- FULL spectrum via dense conversion (assembled matrices only).
!        MatCreateRedundantMatrix + MatConvert → LAPACK (DSYEVD/DGEEV) on rank 0.
!        Writes to {label}_dense_spectrum.dat
!        NOT for MATSHELL — use petsc_mat_probe_spectrum instead.
!
!   petsc_mat_probe_spectrum (A, label, symmetric)
!     -- FULL spectrum via matrix probing (any Mat type, including MATSHELL).
!        Applies A to N standard basis vectors, builds dense matrix on rank 0,
!        then calls LAPACK (DSYEVD/DGEEV).  Writes to {label}_probe_spectrum.dat
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
  public :: petsc_mat_print_info,      &
            petsc_mat_norms,           &
            petsc_mat_diff_norm,       &
            petsc_mat_convert_spectrum,&
            petsc_mat_probe_spectrum,  &
            petsc_mat_equilibrate
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
    MatInfo        :: info
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
      write(*,'(A,I0)')       "  NNZ used    : ", int(info%nz_used)
      write(*,'(A,I0)')       "  NNZ alloc   : ", int(info%nz_allocated)
      ! memory is not populated for distributed (MPIBAIJ) matrix types
      if (info%memory > 0) &
        write(*,'(A,ES12.4)') "  Memory (B)  : ", info%memory
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


  !--------------------------------------------------------------------
  ! Sort eigenvalues by ascending real part.  Used by all spectrum
  ! routines (defined here so it is available outside USE_SLEPC).
  !--------------------------------------------------------------------
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
  ! Compute all eigenvalues of a dense square matrix via LAPACK.
  ! A_in is overwritten on exit. eig_r/eig_i must be pre-allocated.
  ! Symmetric: DSYEVD (divide-and-conquer). Non-symmetric: DGEEV.
  ! Eigenvalues are sorted ascending by real part on success (info==0).
  !--------------------------------------------------------------------
  subroutine lapack_eig_dense(A_in, n, symmetric, eig_r, eig_i, info)
    implicit none
    integer, intent(in)    :: n
    real*8,  intent(inout) :: A_in(n, n)
    logical, intent(in)    :: symmetric
    real*8,  intent(out)   :: eig_r(n), eig_i(n)
    integer, intent(out)   :: info

    integer :: lwork, liwork
    integer :: iwork_query(1)
    real*8  :: work_query(1)
    real*8  :: vl_dummy(1), vr_dummy(1)
    integer, allocatable :: iwork(:)
    real*8,  allocatable :: work(:), wr(:), wi(:)

    if (symmetric) then
      ! DSYEVD: divide-and-conquer; more robust than DSYEV for clustered spectra
      call DSYEVD('N','U', n, A_in, n, eig_r, work_query, -1, iwork_query, -1, info)
      lwork  = max(1, int(work_query(1)))
      liwork = max(1, iwork_query(1))
      allocate(work(lwork), iwork(liwork))
      call DSYEVD('N','U', n, A_in, n, eig_r, work, lwork, iwork, liwork, info)
      eig_i = 0.0d0
      deallocate(work, iwork)
    else
      ! DGEEV: general non-symmetric eigenvalues
      allocate(wr(n), wi(n))
      call DGEEV('N','N', n, A_in, n, wr, wi, vl_dummy, 1, vr_dummy, 1, work_query, -1, info)
      lwork = max(1, int(work_query(1)))
      allocate(work(lwork))
      call DGEEV('N','N', n, A_in, n, wr, wi, vl_dummy, 1, vr_dummy, 1, work, lwork, info)
      eig_r = wr;  eig_i = wi
      deallocate(work, wr, wi)
    endif

    if (info == 0) call sort_eigs_by_real(eig_r, eig_i, n)
  end subroutine lapack_eig_dense


  !--------------------------------------------------------------------
  !> Compute the FULL eigenvalue spectrum of an explicitly assembled Mat.
  !!
  !! Strategy: MatCreateRedundantMatrix gathers the parallel sparse matrix
  !! to one sequential copy per rank → MatConvert to SEQDENSE → MatGetValues
  !! extracts every entry on rank 0 → LAPACK computes all eigenvalues.
  !!
  !! Symmetric matrices: DSYEVD (divide-and-conquer, robust for clusters).
  !! Non-symmetric matrices: DGEEV.
  !! Output: {label}_dense_spectrum.dat
  !!
  !! NOT suitable for MATSHELL — use petsc_mat_probe_spectrum instead.
  !--------------------------------------------------------------------
  subroutine petsc_mat_convert_spectrum(A, label, symmetric)
    implicit none
    Mat,             intent(in) :: A
    character(len=*),intent(in) :: label
    logical,         intent(in) :: symmetric
    Mat            :: A_eq
    Vec            :: dr, dc

    Mat            :: Ared, Adense
    PetscInt       :: M, N, bs, lda
    PetscErrorCode :: ierr
    integer        :: comm, my_id, comm_size, mpierr
    integer        :: n_int, i, j, iunit, info
    PetscScalar, pointer :: dense_arr(:,:)
    real*8, allocatable :: A_copy(:,:), eig_r(:), eig_i(:)
    character(len=512) :: filename
    PetscReal      :: d_norm_1, d_norm_F, d_norm_inf, d_norm_max
    Vec            :: maxabs_vec
    real*8         :: rho_A, d_norm_2_upper


    call MatDuplicate(A, MAT_COPY_VALUES, A_eq, ierr)
    call petsc_mat_equilibrate(A_eq, label, symmetric, dr, dc)

    call PetscObjectGetComm(A_eq, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MPI_Comm_size(comm, comm_size, mpierr)
    call MatGetSize(A_eq, M, N, ierr)
    call MatGetBlockSize(A_eq, bs, ierr)
    n_int = int(M)

    if (my_id == 0) then
      write(*,'(A,A)')       "[EPS] Dense LAPACK spectrum for: ", trim(label)
      write(*,'(A,I0,A,I0)') "[EPS] Matrix size  : ", M, " x ", N
      write(*,'(A,I0)')      "[EPS] Block size   : ", bs
      write(*,'(A,I0,A)')    "[EPS] Gathering to seq (", comm_size, " sub-comms)..."
    endif

    ! Collective norm computations on A before the gather (all ranks participate).
    call MatNorm(A_eq, NORM_1,         d_norm_1,   ierr)
    call MatNorm(A_eq, NORM_FROBENIUS, d_norm_F,   ierr)
    call MatNorm(A_eq, NORM_INFINITY,  d_norm_inf, ierr)
    call MatCreateVecs(A_eq, PETSC_NULL_VEC, maxabs_vec, ierr)
    call MatGetRowMaxAbs(A_eq, maxabs_vec, PETSC_NULL_INTEGER_ARRAY, ierr)
    call VecMax(maxabs_vec, PETSC_NULL_INTEGER, d_norm_max, ierr)
    call VecDestroy(maxabs_vec, ierr)

    ! Gather parallel matrix — each rank gets a full sequential copy
    call MatCreateRedundantMatrix(A_eq, comm_size, MPI_COMM_NULL, MAT_INITIAL_MATRIX, Ared, ierr)
    if (ierr /= 0) then
      if (my_id == 0) write(*,'(A)') &
        "[EPS] ERROR: MatCreateRedundantMatrix failed." // &
        " Shell matrix? Use petsc_mat_probe_spectrum."
      return
    endif

    ! Only rank 0 converts to dense — avoids comm_size × N² memory across all ranks.
    ! MatConvert on a sequential (PETSC_COMM_SELF) matrix is a local operation.
    if (my_id == 0) then
      call MatConvert(Ared, MATSEQDENSE, MAT_INITIAL_MATRIX, Adense, ierr)
    endif
    call MatDestroy(Ared, ierr)

    if (my_id == 0) then
      allocate(eig_r(n_int), eig_i(n_int))
      call MatDenseGetLDA(Adense, lda, ierr)
      if (int(lda) == n_int) then
        ! No lda padding: pass PETSc's internal buffer directly — avoids a second N² allocation.
        call MatDenseGetArrayF90(Adense, dense_arr, ierr)
        call lapack_eig_dense(dense_arr, n_int, symmetric, eig_r, eig_i, info)
        call MatDenseRestoreArrayF90(Adense, dense_arr, ierr)
        call MatDestroy(Adense, ierr)
      else
        ! lda padding present (uncommon): copy column by column, then free Adense before LAPACK.
        allocate(A_copy(n_int, n_int))
        call MatDenseGetArrayF90(Adense, dense_arr, ierr)
        do j = 1, n_int
          A_copy(:, j) = real(dense_arr(1:n_int, j), kind=8)
        enddo
        call MatDenseRestoreArrayF90(Adense, dense_arr, ierr)
        call MatDestroy(Adense, ierr)
        call lapack_eig_dense(A_copy, n_int, symmetric, eig_r, eig_i, info)
        deallocate(A_copy)
      endif
      if (info /= 0) then
        write(*,'(A,I0)') "[EPS] LAPACK error, info = ", info
      else
        ! Spectral radius: rho(A) = max_i |lambda_i|
        rho_A = 0.0d0
        do i = 1, n_int
          rho_A = max(rho_A, sqrt(eig_r(i)**2 + eig_i(i)**2))
        enddo
        ! Tightest available upper bound on ||A||_2 from the two standard bounds:
        !   ||A||_2 <= ||A||_F  (Frobenius)
        !   ||A||_2 <= sqrt(||A||_1 * ||A||_inf)  (Horn & Johnson)
        d_norm_2_upper = min(real(d_norm_F, kind=8), &
                           sqrt(real(d_norm_1, kind=8) * real(d_norm_inf, kind=8)))

        write(filename,'(A,A)') trim(label), "_dense_spectrum.dat"
        open(newunit=iunit, file=trim(filename), status='replace', action='write')
        write(iunit,'(A,A)')    "# Dense LAPACK spectrum of: ", trim(label)
        write(iunit,'(A,I0)')   "# Matrix size : ", M
        write(iunit,'(A,L1)')   "# Symmetric   : ", symmetric
        write(iunit,'(A,ES22.14)') "# rho(A)                        = ", rho_A
        write(iunit,'(A,ES22.14)') "# ||A||_1                       = ", real(d_norm_1, kind=8)
        write(iunit,'(A,ES22.14)') "# ||A||_F                       = ", real(d_norm_F, kind=8)
        write(iunit,'(A,ES22.14)') "# ||A||_inf                     = ", real(d_norm_inf, kind=8)
        write(iunit,'(A,ES22.14)') "# ||A||_max                     = ", real(d_norm_max, kind=8)
        write(iunit,'(A,ES22.14)') "# ||A||_2 <= min(F,sqrt(1*inf)) = ", d_norm_2_upper
        write(iunit,'(A,ES22.14)') "# ||A||_2 - rho(A) <=           = ", d_norm_2_upper - rho_A
        write(iunit,'(A)')      "#        Re(lambda)           Im(lambda)"
        do i = 1, n_int
          write(iunit,'(2X,ES22.14,2X,ES22.14)') eig_r(i), eig_i(i)
        enddo
        close(iunit)

        write(*,'(A,I0,A,A)') "[EPS] All ", n_int, " eigenvalues -> ", trim(filename)
        write(*,'(A)') "[EPS] --- Non-normality measure ---"
        write(*,'(A,ES12.4)') "[EPS]  rho(A)               = ", rho_A
        write(*,'(A,ES12.4)') "[EPS]  ||A||_1              = ", real(d_norm_1,   kind=8)
        write(*,'(A,ES12.4)') "[EPS]  ||A||_F              = ", real(d_norm_F,   kind=8)
        write(*,'(A,ES12.4)') "[EPS]  ||A||_inf            = ", real(d_norm_inf, kind=8)
        write(*,'(A,ES12.4)') "[EPS]  ||A||_max            = ", real(d_norm_max, kind=8)
        write(*,'(A,ES12.4)') "[EPS]  sqrt(||_1*||_inf)    = ", &
          sqrt(real(d_norm_1, kind=8) * real(d_norm_inf, kind=8))
        write(*,'(A,ES12.4)') "[EPS]  ||A||_2 <=           = ", d_norm_2_upper
        write(*,'(A,ES12.4)') "[EPS]  ||A||_2 - rho(A) <=  = ", d_norm_2_upper - rho_A
        if (symmetric) write(*,'(A)') &
          "[EPS]  (symmetric matrix: ||A||_2 = rho(A) exactly)"
      endif
      deallocate(eig_r, eig_i)
    endif

    call VecDestroy(dr, ierr);  call VecDestroy(dc, ierr)
    call MatDestroy(A_eq, ierr)
  end subroutine petsc_mat_convert_spectrum


  !--------------------------------------------------------------------
  !> Compute the FULL eigenvalue spectrum via matrix probing.
  !!
  !! Works for ANY Mat type including MATSHELL.  Applies A to each of
  !! the N standard basis vectors (N collective MatMult calls), gathers
  !! the result to rank 0, builds the dense matrix, then calls LAPACK.
  !!
  !! Symmetric matrices: DSYEVD.  Non-symmetric: DGEEV.
  !! Output: {label}_probe_spectrum.dat
  !--------------------------------------------------------------------
  subroutine petsc_mat_probe_spectrum(A, label, symmetric)
    implicit none
    Mat,             intent(in) :: A
    character(len=*),intent(in) :: label
    logical,         intent(in) :: symmetric

    Vec                  :: e_col, f_col, f_all
    VecScatter           :: scat
    PetscInt             :: M, N, i_col, bs
    PetscScalar          :: one_val
    PetscScalar, pointer :: f_arr(:)
    PetscErrorCode       :: ierr
    integer              :: comm, my_id, mpierr
    integer              :: n_int, i, iunit, info
    real*8, allocatable  :: A_dense(:,:), eig_r(:), eig_i(:)
    character(len=512)   :: filename

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MatGetSize(A, M, N, ierr)
    call MatGetBlockSize(A, bs, ierr)
    n_int = int(M)

    if (M /= N) then
      if (my_id == 0) write(*,'(A)') "[EPS] ERROR: petsc_mat_probe_spectrum requires square matrix"
      return
    endif

    if (my_id == 0) then
      write(*,'(A,A)')      "[EPS] Probing spectrum for: ", trim(label)
      write(*,'(A,I0)')     "[EPS] Matrix size : ", M
      write(*,'(A,I0)')     "[EPS] Block size  : ", bs
      write(*,'(A,I0,A)')   "[EPS] Applying A to ", N, " basis vectors (collective)..."
    endif

    ! Create work vectors (domain and range sides of A)
    call MatCreateVecs(A, e_col, f_col, ierr)
    ! Gather f_col to a sequential vector visible on all ranks
    call VecScatterCreateToAll(f_col, scat, f_all, ierr)

    if (my_id == 0) allocate(A_dense(n_int, n_int))
    one_val = 1.0d0

    do i_col = 0, N - 1
      ! Build basis vector e_{i_col}
      call VecSet(e_col, 0.0d0, ierr)
      call VecSetValue(e_col, i_col, one_val, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_col, ierr)
      call VecAssemblyEnd(e_col, ierr)

      call MatMult(A, e_col, f_col, ierr)

      ! Gather result to sequential vector on all ranks
      call VecScatterBegin(scat, f_col, f_all, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd(scat, f_col, f_all, INSERT_VALUES, SCATTER_FORWARD, ierr)

      ! Column i_col+1 of A = f_all  (only store on rank 0)
      if (my_id == 0) then
        call VecGetArray(f_all, f_arr, ierr)
        A_dense(:, i_col + 1) = real(f_arr, kind=8)
        call VecRestoreArray(f_all, f_arr, ierr)
      endif
    enddo

    call VecDestroy(e_col, ierr)
    call VecDestroy(f_col, ierr)
    call VecScatterDestroy(scat, ierr)
    call VecDestroy(f_all, ierr)

    if (my_id == 0) then
      write(*,'(A)') "[EPS] Probing complete. Computing eigenvalues via LAPACK..."
      allocate(eig_r(n_int), eig_i(n_int))
      ! Pass A_dense directly — lapack_eig_dense overwrites it, but we no longer need it.
      ! Avoids the 2x peak that would result from allocating a separate A_copy.
      call lapack_eig_dense(A_dense, n_int, symmetric, eig_r, eig_i, info)
      deallocate(A_dense)
      if (info /= 0) then
        write(*,'(A,I0)') "[EPS] LAPACK error, info = ", info
      else
        write(filename,'(A,A)') trim(label), "_probe_spectrum.dat"
        open(newunit=iunit, file=trim(filename), status='replace', action='write')
        write(iunit,'(A,A)')  "# Probe spectrum of: ", trim(label)
        write(iunit,'(A,I0)') "# Matrix size : ", M
        write(iunit,'(A,L1)') "# Symmetric   : ", symmetric
        write(iunit,'(A)')    "#        Re(lambda)           Im(lambda)"
        do i = 1, n_int
          write(iunit,'(2X,ES22.14,2X,ES22.14)') eig_r(i), eig_i(i)
        enddo
        close(iunit)
        write(*,'(A,I0,A,A)') "[EPS] All ", n_int, " eigenvalues -> ", trim(filename)
      endif
      deallocate(eig_r, eig_i)
    endif
  end subroutine petsc_mat_probe_spectrum


  !--------------------------------------------------------------------
  !> Iterative max-norm equilibration of a distributed PETSc Mat.
  !!
  !! Two modes:
  !!   symmetric = .false. (Ruiz algorithm):
  !!     Finds separate row scaling D_r and column scaling D_c such that
  !!     all rows and columns of D_r * A * D_c have max absolute value 1.
  !!     Useful for general non-symmetric matrices.
  !!
  !!   symmetric = .true. (Knight-Ruiz-Ucar symmetry-preserving algorithm):
  !!     Finds a single diagonal scaling D such that all rows (= columns)
  !!     of D * A * D have max absolute value 1. The scaled matrix remains
  !!     symmetric whenever A is symmetric.
  !!
  !! The matrix A is modified in-place.
  !! dr_out and dc_out are the accumulated scalings: A_eq = D_r * A * D_c.
  !! They are created by this routine; the caller must VecDestroy them.
  !! For symmetric mode both contain the same values.
  !!
  !! Usage: to scale a right-hand side b and unscale solution x:
  !!   b_scaled = diag(dr_out) * b
  !!   x_orig   = diag(dc_out) * x_scaled
  !--------------------------------------------------------------------
  subroutine petsc_mat_equilibrate(A, label, symmetric, dr_out, dc_out)
    implicit none
    Mat,             intent(inout) :: A
    character(len=*),intent(in)    :: label
    logical,         intent(in)    :: symmetric
    Vec,             intent(out)   :: dr_out, dc_out

    PetscInt          :: M_global, N_global, rstart, rend, m_local
    PetscErrorCode    :: ierr
    integer           :: comm, my_id, mpierr, iter, maxit, n_int, i
    real*8            :: tol, res_row, res_col = 0.0d0
    logical           :: conv

    real*8, allocatable    :: r_local(:), c_global(:)
    Vec                    :: u_r_vec, u_c_vec, r_max_tmp
    PetscScalar, pointer   :: u_arr(:)

    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)
    call MatGetSize(A, M_global, N_global, ierr)
    call MatGetOwnershipRange(A, rstart, rend, ierr)
    m_local = rend - rstart
    n_int   = int(N_global)

    maxit = 100
    tol   = 1.0d-8

    allocate(r_local(int(m_local)), c_global(n_int))

    ! Create output accumulation Vecs (parallel, matching A's row/column distributions).
    ! MatCreateVecs(A, right=col-space, left=row-space, ierr)
    call MatCreateVecs(A, dc_out, dr_out, ierr)
    call VecSet(dr_out, 1.0d0, ierr)
    call VecSet(dc_out, 1.0d0, ierr)

    ! Per-iteration update Vecs
    call VecDuplicate(dr_out, u_r_vec, ierr)
    if (.not. symmetric) call VecDuplicate(dc_out, u_c_vec, ierr)

    if (my_id == 0) then
      write(*,'(A,A)') "[EQ] Equilibrating: ", trim(label)
      if (symmetric) then
        write(*,'(A)') "[EQ] Mode: symmetric (Knight-Ruiz-Ucar)"
      else
        write(*,'(A)') "[EQ] Mode: non-symmetric (Ruiz)"
      endif
    endif

    conv = .false.
    do iter = 1, maxit

      ! --- Compute row max via MatGetRowMaxAbs ---
      r_local = 0.0d0
      call MatCreateVecs(A, PETSC_NULL_VEC, r_max_tmp, ierr)
      call MatGetRowMaxAbs(A, r_max_tmp, PETSC_NULL_INTEGER_ARRAY, ierr)
      call VecGetArray(r_max_tmp, u_arr, ierr)
      do i = 1, int(m_local)
        r_local(i) = real(u_arr(i), kind=8)
      enddo
      call VecRestoreArray(r_max_tmp, u_arr, ierr)
      call VecDestroy(r_max_tmp, ierr)

      ! --- Compute column max via MatGetColumnNorms (non-symmetric only) ---
      c_global = 0.0d0
      if (.not. symmetric) then
        call MatGetColumnNorms(A, NORM_INFINITY, c_global, ierr)
      endif

      if (iter == 1 .and. my_id == 0) then
        write(*,'(A,ES10.2,A,ES10.2,A,I0)') &
          "[EQ]   iter 1 r_local range: [", minval(r_local), ", ", maxval(r_local), &
          "]  nnz-rows=", count(r_local > 0.0d0)
      endif

      ! --- Check convergence: all non-zero rows (and columns) max should be ~1 ---
      res_row = 0.0d0
      do i = 1, int(m_local)
        if (r_local(i) > 0.0d0) &
          res_row = max(res_row, abs(r_local(i) - 1.0d0))
      enddo
      call MPI_Allreduce(MPI_IN_PLACE, res_row, 1, MPI_DOUBLE_PRECISION, &
                         MPI_MAX, comm, mpierr)

      if (symmetric) then
        conv = res_row < tol
      else
        res_col = 0.0d0
        do i = 1, n_int
          if (c_global(i) > 0.0d0) &
            res_col = max(res_col, abs(c_global(i) - 1.0d0))
        enddo
        conv = (res_row < tol) .and. (res_col < tol)
      endif

      if (my_id == 0) then
        if (symmetric) then
          write(*,'(A,I3,A,ES10.2)') "[EQ]   iter ", iter, "  res_row = ", res_row
        else
          write(*,'(A,I3,2(A,ES10.2))') "[EQ]   iter ", iter, &
            "  res_row = ", res_row, "  res_col = ", res_col
        endif
      endif

      if (conv) then
        if (my_id == 0) write(*,'(A,I3,A)') "[EQ] Converged in ", iter, " iterations."
        exit
      endif

      ! --- Build update Vec u_r = 1/sqrt(r) (local rows) ---
      call VecGetArray(u_r_vec, u_arr, ierr)
      do i = 1, int(m_local)
        if (r_local(i) > 0.0d0) then
          u_arr(i) = 1.0d0 / sqrt(r_local(i))
        else
          u_arr(i) = 1.0d0   ! zero row: no scaling
        endif
      enddo
      call VecRestoreArray(u_r_vec, u_arr, ierr)

      if (symmetric) then
        ! Same D applied to both sides: A ← D * A * D  (preserves symmetry)
        call MatDiagonalScale(A, u_r_vec, u_r_vec, ierr)
        call VecPointwiseMult(dr_out, dr_out, u_r_vec, ierr)
      else
        ! Non-symmetric: separate row/column scaling
        ! Build u_c = 1/sqrt(c) for locally owned columns.
        ! For square MPIAIJ, owned columns == owned rows: rstart..rend-1 (0-based).
        call VecGetArray(u_c_vec, u_arr, ierr)
        do i = 1, int(m_local)
          ! 0-based global column = rstart + i - 1  →  c_global index = rstart + i
          if (c_global(int(rstart) + i) > 0.0d0) then
            u_arr(i) = 1.0d0 / sqrt(c_global(int(rstart) + i))
          else
            u_arr(i) = 1.0d0   ! zero column: no scaling
          endif
        enddo
        call VecRestoreArray(u_c_vec, u_arr, ierr)

        call MatDiagonalScale(A, u_r_vec, u_c_vec, ierr)
        call VecPointwiseMult(dr_out, dr_out, u_r_vec, ierr)
        call VecPointwiseMult(dc_out, dc_out, u_c_vec, ierr)
      endif

    enddo  ! iter

    if (.not. conv .and. my_id == 0) &
      write(*,'(A,I0,A)') "[EQ] WARNING: did not converge in ", maxit, " iterations."

    ! For symmetric mode: dc_out holds same values as dr_out
    if (symmetric) call VecCopy(dr_out, dc_out, ierr)

    call VecDestroy(u_r_vec, ierr)
    if (.not. symmetric) call VecDestroy(u_c_vec, ierr)
    deallocate(r_local, c_global)

    if (my_id == 0) write(*,'(A)') "[EQ] Equilibration done."
  end subroutine petsc_mat_equilibrate


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
