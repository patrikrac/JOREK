module mod_cond_estimator
    use mod_matv
    use mpi_mod
    use data_structure, only: type_SP_MATRIX

    implicit none

    public :: estimate_condition_number, estimate_condition_number_2
    private 

contains

    subroutine estimate_condition_number(a_mat, cond_est)
        type(type_SP_MATRIX), intent(in) :: a_mat
        real*8, intent(out) :: cond_est
        real*8, external :: dnrm2

        real*8, allocatable, dimension(:,:) :: U, V
        real*8, allocatable, dimension(:) :: alpha, beta

        integer :: i, j
        integer :: k_max, n, n_rows_local
        real*8, allocatable, dimension(:) :: S, E
        real*8, allocatable, dimension(:) :: dummy_u, dummy_vt, dummy_c, work
        integer :: info
        real*8  :: min_sv, max_sv
        integer :: my_id, ierr
        logical :: verbose = .false.

        call MPI_COMM_RANK(a_mat%comm, my_id, info)

        ! Parameters
        k_max = 300
        n = a_mat%ng
        n_rows_local = a_mat%my_ind_max - a_mat%my_ind_min + 1

        ! Allocation 
        allocate(U(n, k_max), V(n, k_max+1))
        U = 0.d0; V = 0.d0
        allocate(alpha(k_max), beta(k_max))
        alpha = 0.d0; beta = 0.d0

        ! Initialization
        if (my_id .eq. 0) then
            call random_number(V(:, 1))
            V(:, 1) = V(:, 1) / dnrm2(n, V(:, 1), 1)
        endif
        ! Send the random initial vector to all processes
        call MPI_BCAST(V(:, 1), n, MPI_DOUBLE_PRECISION, 0, a_mat%comm, ierr) 

        ! GKL loop
        do j = 1, k_max
            
            ! p = A * v_j
            call bcsr_matv(a_mat, V(:, j), U(:, j))

            ! p = p = p - beta_{j-1} * u_{j-1}
            if (j > 1) call daxpy(n, -beta(j-1), U(:, j-1), 1, U(:, j), 1)

            ! Orthogonalize p against all U vectors
            if (j > 1) call reorthogonalize(U(:, j), U, j-1, n)

            ! alpha_j = ||p||
            alpha(j) = dnrm2(n, U(:, j), 1)

            if (alpha(j) < 1.0d-14) then
                print *, "GKL breakdown (alpha) at iteration ", j
                exit
            end if

            ! u_j = p / alpha_j
            call dscal(n, 1.0d0/alpha(j), U(:, j), 1)

            ! q = A^T * u_j
            call bcsr_matvT(a_mat, U(:, j), V(:, j+1))

            ! q = q - alpha_j * v_j
            call daxpy(n, -alpha(j), V(:, j), 1, V(:, j+1), 1)

            ! Orthogonalize q against all V vectors
            call reorthogonalize(V(:, j+1), V, j, n)

            ! beta_j = ||q||
            beta(j) = dnrm2(n, V(:, j+1), 1)

            if (verbose .and. my_id .eq. 0) print *, "Iteration ", j, ": alpha = ", alpha(j), " beta = ", beta(j)

            if (beta(j) < 1.0d-14) then
                print *, "GKL breakdown (beta) at iteration ", j
                exit
            end if

            ! v_{j+1} = r / beta_j
            call dscal(n, 1.0d0/beta(j), V(:, j+1), 1)

        enddo

        ! Solve small SVD

        allocate(S(k_max), E(k_max-1))
        allocate(work(4*k_max))

        S = alpha(1:k_max)
        E = beta(1:k_max-1)

        call DBDSQR('U', k_max, 0, 0, 0, &
                    S, E, &
                    dummy_vt, 1, dummy_u, 1, dummy_c, 1, &
                    work, info)

        if (info /= 0) then
            write(*,*) 'Error in DBDSQR: ', info
            cond_est = -1.0d0
            return
        end if

        max_sv = S(1)
        min_sv = S(k_max)

        if (my_id .eq. 0) print *, "Estimated largest singular value: ", max_sv
        if (my_id .eq. 0) print *, "Estimated smallest singular value: ", min_sv
        if (min_sv > 0.0d0) then
            cond_est = max_sv / min_sv
        else
            cond_est = 1.0d20  ! Large number to indicate near-singularity
        end if

        ! Cleanup 
        deallocate(U, V, alpha, beta, S, E, work)
    end subroutine estimate_condition_number


    !< Reorthogonalize the size n vector v agains the basis vectors in 'basis' >
    subroutine reorthogonalize(v, basis, num_basis, n)
        real*8, intent(inout)   :: v(:)
        real*8, intent(in)      :: basis(:,:)
        integer, intent(in)     :: num_basis
        integer, intent(in)     :: n
        real*8, external :: ddot

        integer :: j 

        real*8  :: proj(num_basis)
 
        do j = 1, 2
            call dgemv('C', n, num_basis, 1.d0, basis, n, v, 1, 0.d0, proj, 1)
            call dgemv('N', n, num_basis, -1.d0, basis, n, proj, 1, 1.d0, v, 1)
        enddo
    end subroutine reorthogonalize

! Second implementation using LU factorization
    subroutine estimate_condition_number_2(a_mat, cond_est)
        type(type_SP_MATRIX), intent(in) :: a_mat
        real*8, intent(out) :: cond_est
        
        real*8              :: min_sv, max_sv
        real*8,allocatable  :: dummy_vec(:)
        integer             :: my_id, ierr

        call MPI_COMM_RANK(a_mat%comm, my_id, ierr)

        allocate(dummy_vec(a_mat%ng))
        dummy_vec = 0.d0

        ! Compute the max singular value q using power iteration
        call power_iteration(a_mat, max_sv, dummy_vec, 20)

        ! Compute the inverse of the min singular value using the power iteration on A^{-1}
        call inverse_power_iteration(a_mat, min_sv, dummy_vec, 20)

        cond_est = max_sv / min_sv
    end subroutine estimate_condition_number_2  


! power iteration implementation 
    subroutine power_iteration(a_mat, max_sval, max_svec, max_iters)
        type(type_SP_MATRIX), intent(in) :: a_mat
        real*8, intent(out) :: max_sval
        real*8, intent(out) :: max_svec(:)
        integer, intent(in) :: max_iters
        real*8, external :: dnrm2

        real*8  :: old_sval, norm_svec
        real*8, allocatable :: tmp_vec(:)
        integer :: k, n
        logical :: verbose = .true.
        integer :: my_id, ierr

        call MPI_COMM_RANK(a_mat%comm, my_id, ierr)

        n = size(max_svec)

        allocate(tmp_vec(n))

        if (my_id .eq. 0) call random_number(max_svec)
        call MPI_BCAST(max_svec, n, MPI_DOUBLE_PRECISION, 0, a_mat%comm, ierr)

        call dscal(n, 1.0d0/dnrm2(n, max_svec, 1), max_svec, 1)
       
        max_sval = 0.d0

        do k = 1, max_iters
            old_sval = max_sval
            call bcsr_matv(a_mat, max_svec, tmp_vec)  ! y = A * x
            call bcsr_matvT(a_mat, tmp_vec, max_svec)  ! x = A^T * y

            norm_svec = dnrm2(n, max_svec, 1)
            max_sval = dnrm2(n, tmp_vec, 1)

            if (verbose .and. my_id .eq. 0) print *, "Power iteration ", k, ": max singular value = ", max_sval
            call dscal(n, 1.0d0/max_sval, max_svec, 1)  ! x = y / sigma

            if (k > 1 .and. abs(max_sval - old_sval) < 1.d-4 * max_sval) exit ! Convergence Check
        enddo

        deallocate(tmp_vec)
        
    end subroutine power_iteration


    subroutine inverse_power_iteration(a_mat, min_sval, min_svec, max_iters)
#ifdef USE_MUMPS
        use mod_mumps
#endif
        use data_structure, only: type_RHS

        type(type_SP_MATRIX), intent(in) :: a_mat
        real*8, intent(out) :: min_sval
        real*8, intent(out) :: min_svec(:)
        integer, intent(in) :: max_iters

        real*8, external :: dnrm2

        type(type_MUMPS_SOLVER) :: mmss
        type(type_RHS) :: svec_rhs
        real*8  :: old_sval, norm_svec
        integer :: k, n
        logical :: verbose = .true.
        integer :: my_id, ierr
#ifdef USE_MUMPS
        call MPI_COMM_RANK(a_mat%comm, my_id, ierr)

        call mumps_initialize(mmss,a_mat%comm)

        call mumps_analyze(mmss,a_mat)

        call mumps_factorize(mmss,a_mat)

        n = size(min_svec)

        allocate(svec_rhs%val(n))

        if (my_id .eq. 0) call random_number(min_svec)
        call MPI_BCAST(min_svec, n, MPI_DOUBLE_PRECISION, 0, a_mat%comm, ierr)

        call dscal(n, 1.0d0/dnrm2(n, min_svec, 1), min_svec, 1)
       
        min_sval = 0.d0
        svec_rhs%val = min_svec

        do k = 1, max_iters
            old_sval = min_sval
            call mumps_set_solve_transpose(mmss, .true.)
            call mumps_solve(mmss, svec_rhs)
            min_sval = dnrm2(n, svec_rhs%val, 1)
            call mumps_set_solve_transpose(mmss, .false.)
            call mumps_solve(mmss, svec_rhs)
            norm_svec = dnrm2(n, svec_rhs%val, 1)


            if (verbose .and. my_id .eq. 0) print *, "Inverse Power iteration ", k, ": min singular value = ", 1.d0/min_sval
            call dscal(n, 1.0d0/min_sval, svec_rhs%val, 1)  ! x = y / sigma

            if (k > 1 .and. abs(min_sval - old_sval) < 1.d-4 * min_sval) exit ! Convergence Check
        enddo

        min_svec = svec_rhs%val
        min_sval = 1.d0 / min_sval

        deallocate(svec_rhs%val)
#else
        print *, "MUMPS is required for inverse power iteration. Please recompile..."
#endif
    end subroutine inverse_power_iteration
 
end module mod_cond_estimator