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
        k_max = 500
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

! Experimental implementation of LSMR-based condition number estimator (Not working well yet, needs debugging)
    subroutine estimate_condition_number_2(a_mat, cond_est)
        type(type_SP_MATRIX), intent(in) :: a_mat
        real*8, intent(out) :: cond_est

        ! -- External BLAS functions --
        real*8, external :: dnrm2, ddot
        
        ! -- Local Constants --
        integer, parameter :: MAX_ITER = 500  ! Increase if convergence is poor
        real*8, parameter :: ONE = 1.0d0
        real*8, parameter :: ZERO = 0.0d0
        
        ! -- Local Arrays (Allocatable for safety) --
        real*8, allocatable :: U(:,:), V(:,:)
        real*8, allocatable :: u_tmp(:), v_tmp(:)
        
        ! -- GKL/LSMR Scalars --
        real*8 :: alpha, beta
        real*8 :: alphabar, zetabar
        real*8 :: rho, c, s, chat, shat, alphahat
        real*8 :: normA2, min_rbar
        real*8 :: sigma_max, sigma_min
        
        ! -- Loop Indices and dims --
        integer :: m, n, j, k, pass
        real*8 :: d

        integer :: my_id, ierr, info
        logical :: verbose = .false.

        call MPI_COMM_RANK(a_mat%comm, my_id, info)
        
        ! 1. Get dimensions (Adjust %m / %n to match your struct)
        n = a_mat%ng
        m = n
        
        ! 2. Allocate Basis Vectors
        !    U is (m, MAX_ITER+1), V is (n, MAX_ITER+1)
        allocate(U(m, MAX_ITER + 1))
        allocate(V(n, MAX_ITER + 1))
        allocate(u_tmp(m))
        allocate(v_tmp(n))
        
        ! 3. Initialize GKL Process
        !    Start with random V_1 (Recommended to find min eigenvalue)
        if (my_id .eq. 0) then
            call random_number(v_tmp)
        endif
        ! Send the random initial vector to all processes
        call MPI_BCAST(v_tmp, n, MPI_DOUBLE_PRECISION, 0, a_mat%comm, ierr) 
        
        ! Normalize v_1
        alpha = dnrm2(n, v_tmp, 1)
        if (alpha > epsilon(ZERO)) then
            call dscal(n, ONE/alpha, v_tmp, 1)
        endif
        V(:, 1) = v_tmp
        
        ! u_1 = A * v_1
        call bcsr_matv(a_mat, V(:, 1), u_tmp)
        
        ! beta_1 = ||u_1||
        beta = dnrm2(m, u_tmp, 1)
        if (beta > epsilon(ZERO)) then
            call dscal(m, ONE/beta, u_tmp, 1)
        endif
        U(:, 1) = u_tmp
        
        ! 4. Initialize LSMR Scalars
        !    (Fong & Saunders, 2011)
        alphabar = alpha 
        zetabar  = alpha * beta
        
        ! Implicit Frobenius norm accumulator
        normA2 = beta**2 
        
        ! Condition number trackers
        min_rbar = huge(ONE)
        
        ! Initial estimate
        cond_est = ONE
        
        if (my_id .eq. 0) print *, "Iter | Sigma_Max (Est) | Sigma_Min (Est) | Condition (Est)"
        ! ---------------------------------------------------------
        ! 5. Main Iteration Loop
        ! ---------------------------------------------------------
        do j = 1, MAX_ITER
            
            ! --- Step A: Compute V_{j+1} ---
            ! v_tmp = A^T * u_j
            call bcsr_matvT(a_mat, U(:, j), v_tmp)
            
            ! v_tmp = v_tmp - beta * v_j (GKL standard subtraction)
            call daxpy(n, -beta, V(:, j), 1, v_tmp, 1)
            
            ! -- Reorthogonalization of V --
            call reorthogonalize(v_tmp, V, j, n)
            
            ! alpha_{j+1} = ||v_tmp||
            alpha = dnrm2(n, v_tmp, 1)
            if (alpha > epsilon(ZERO)) then
                call dscal(n, ONE/alpha, v_tmp, 1)
            endif
            V(:, j+1) = v_tmp
            
            ! --- Step B: LSMR Scalar Rotation (Part 1) ---
            ! Rotate to eliminate previous beta
            call sym_ortho(alphabar, beta, chat, shat, alphahat)
            
            ! Construct rotation for current alpha
            call sym_ortho(alphahat, alpha, c, s, rho)
            
            ! Update alphabar for next step
            alphabar = c * alpha ! Wait, check indices. 
            ! In LSMR code: alphabar = alpha * c_next. 
            ! Here alpha is the *new* alpha just computed.
            
            ! Update Frobenius Norm (Sigma Max Estimate)
            normA2 = normA2 + alpha**2
            
            ! Update Sigma Min Estimate (via Diagonal of R)
            if (rho /= ZERO) then
               min_rbar = min(min_rbar, abs(rho))
            endif
            
            ! --- Step C: Compute U_{j+1} ---
            if (j < MAX_ITER) then
                ! u_tmp = A * v_{j+1}
                call bcsr_matv(a_mat, V(:, j+1), u_tmp)
                
                ! u_tmp = u_tmp - alpha * u_j
                call daxpy(m, -alpha, U(:, j), 1, u_tmp, 1)
                
                ! -- Reorthogonalization of U --
                call reorthogonalize(u_tmp, U, j, m)
                
                ! beta_{j+1} = ||u_tmp||
                beta = dnrm2(m, u_tmp, 1)
                
                ! LSMR Norm Update for next beta
                normA2 = normA2 + beta**2
                
                if (beta > epsilon(ZERO)) then
                    call dscal(m, ONE/beta, u_tmp, 1)
                endif
                U(:, j+1) = u_tmp
                
                ! Prepare for next rotation
                alphabar = c * alpha
            end if
            
            ! --- Step D: Calculate Estimates ---
            sigma_max = sqrt(normA2)
            sigma_min = min_rbar
            
            if (sigma_min > epsilon(ZERO)) then
                cond_est = sigma_max / sigma_min
            endif

            if (my_id .eq. 0 .and. mod(j, 10) == 0) then
                print '(I5, 3ES14.6)', j, sigma_max, sigma_min, cond_est
            end if
            
        end do
        
        ! Cleanup
        deallocate(U, V, u_tmp, v_tmp)
        
    end subroutine estimate_condition_number_2  


! Simple power iteration implementation (unused)
    subroutine power_iteration(a_mat, dominant_sval, dominant_svec, max_iters)
        type(type_SP_MATRIX), intent(in) :: a_mat
        real*8, intent(out) :: dominant_sval
        real*8, intent(out) :: dominant_svec(:)
        integer, intent(in) :: max_iters
        real*8, external :: dnrm2

        integer :: k
        logical :: verbose = .false.
        integer :: my_id, ierr

        call MPI_COMM_RANK(a_mat%comm, my_id, ierr)

        call random_number(dominant_svec)
        call dscal(size(dominant_svec), 1.0d0/dnrm2(size(dominant_svec), dominant_svec, 1), dominant_svec, 1)

        do k = 1, max_iters
            call bcsr_matv(a_mat, dominant_svec, dominant_svec)  ! y = A * x
            dominant_sval = dnrm2(size(dominant_svec), dominant_svec, 1)  ! sigma = ||y||
            if (verbose .and. my_id .eq. 0) print *, "Power iteration ", k, ": dominant singular value = ", dominant_sval
            call dscal(size(dominant_svec), 1.0d0/dominant_sval, dominant_svec, 1)  ! x = y / sigma
        enddo
        
    end subroutine power_iteration
 

! Helper routines –--
    subroutine sym_ortho(a, b, c, s, r)
            real*8, intent(in) :: a, b
            real*8, intent(out) :: c, s, r
            real*8 :: tau
            real*8, parameter :: ONE = 1.0d0
            real*8, parameter :: ZERO = 0.0d0
            
            if (b == ZERO) then
                c = sign(ONE, a)
                s = ZERO
                r = abs(a)
            elseif (a == ZERO) then
                c = ZERO
                s = sign(ONE, b)
                r = abs(b)
            elseif (abs(b) > abs(a)) then
                tau = a / b
                s = sign(ONE, b) / sqrt(ONE + tau**2)
                c = s * tau
                r = b / s
            else
                tau = b / a
                c = sign(ONE, a) / sqrt(ONE + tau**2)
                s = c * tau
                r = a / c
            end if
        end subroutine sym_ortho
end module mod_cond_estimator