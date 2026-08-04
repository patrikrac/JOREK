module mod_gmres2
!#ifdef USE_GMRES
  use iso_c_binding
  use mpi_mod
  use mod_sparse_data, only: pastix, mumps, strumpack
  use mod_integer_types
  use mod_matv

  private
  public :: gmres2_driver
contains

!> solve a_mat x=b using iterative GMRES method with right preconditioning
subroutine gmres2_driver(a_mat,b,x,n,solver)
  use mod_sparse_data, only: type_SP_SOLVER
  use data_structure,  only: type_SP_MATRIX

  implicit none
  type(type_SP_MATRIX)                :: a_mat
  real(kind=8), dimension(:), pointer :: x, b
  integer :: n
  type(type_SP_SOLVER)  :: solver
  
  real(kind=8) :: atol, rtol, gamma, delta, rho, rho0=0.0, bnrm, scale_norm
  real(kind=8) :: anrm2_loc, anrm2_glob, anrm, xnrm, nbe, nbe_denom
  real(kind=8) :: norm_p, norm_p_new
  integer :: totit, maxit, restart, nrit, it, ldh, k, j
  integer :: n_ortho 
  logical :: no_conv, GSC=.false., GSM=.false., GSCI=.true., GSMI=.false.
  real(kind=8), dimension(:), allocatable, target :: givens_c, givens_s, hess, V, b_prec, b_, s_

  integer :: my_id, my_id_n, n_cpu, ierr
  integer :: MPI_GLOB, MPI_COMM_N

  external :: dcopy, daxpby, daxpy, dgemv, dscal, dtrsv
  real(kind=8), external :: dnrm2, ddot

  MPI_GLOB   = a_mat%comm
  MPI_COMM_N = solver%pc%MPI_COMM_N

  call MPI_COMM_RANK(MPI_GLOB, my_id, ierr)
  call MPI_COMM_SIZE(MPI_GLOB, n_cpu, ierr)
  call MPI_COMM_RANK(MPI_COMM_N, my_id_n, ierr)

  rtol = solver%iter_tol
  atol = 1.d-36
  maxit = solver%iter_max
  restart = solver%gmres_m
  n_ortho = 3 ! Number of iterations for the orthogonalization methos (in case of iterative methods)
  if (restart > maxit) restart = maxit

  allocate(givens_c(restart),givens_s(restart),b_(restart+1),hess((restart+1)*restart),V(n*(restart+1)),b_prec(n))
  if (GSCI .or. GSMI) allocate(s_(restart)) 

  givens_c(1:restart) = 0.
  givens_s(1:restart) = 0.

  ldh = restart+1

  ! --- Compute RHS norm ||b||_2 ---
  bnrm = dnrm2(n, b, 1)
  if (bnrm == 0.d0) bnrm = 1.d0

  ! --- Compute global matrix Frobenius norm ||A||_F ---
  anrm2_loc = dnrm2(int(a_mat%nnz), a_mat%val, 1)**2
  call MPI_Allreduce(anrm2_loc, anrm2_glob, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_GLOB, ierr)
  anrm = sqrt(anrm2_glob)
  if (anrm == 0.d0) anrm = 1.d0

  no_conv = .true.
  totit = 0;  

  do while (no_conv)
    ! --- v_1 = A * x ---
    call bcsr_matv(a_mat, x, V(1:n))

    ! --- v_1 = b - A * x (True residual) ---
    call daxpby(n, 1.d0, b(1:n), 1, -1.d0, V(1:n), 1)

    ! --- Denominator for Normwise Backward Error: ||A||_F * ||x||_2 + ||b||_2 ---
    xnrm = dnrm2(n, x, 1)
    nbe_denom = anrm * xnrm + bnrm

    ! --- rho = ||v_1||_2 ---
    rho = dnrm2(n, V(1:n), 1)

    ! --- Normwise Backward Error: NBE = ||r||_2 / (||A||_F * ||x||_2 + ||b||_2) ---
    nbe = rho / nbe_denom
    if (totit .eq. 0) then
      rho0 = rho
      scale_norm = max(rho0, bnrm)
    endif
    if ((rho/scale_norm < rtol) .or. (nbe < rtol) .or. (rho < atol)) then
      no_conv = .false.
      exit
    endif
    !--- v_1 = v_1 / rho
    call dscal(n, 1./rho, V(1:n), 1)
    b_(1) = rho
    b_(2:restart+1) = 0.d0
    nrit = restart-1
    if (my_id.eq.0) then
      write(*, "(A, X, I0, X, A, X, ES14.6, X, A, X, ES14.6, X, A, X, ES14.6)") "[GMRES] iteration", totit, "res =", rho, "rel.res =", rho/scale_norm, "nbe =", nbe
      write(*, "(A)") "[GMRES] --- Restart ---"
    endif

    do it = 1, restart
      totit = totit +1
      ! --- b_prec = M^-1 * v_j --- 
      call prec(solver, V((it-1)*n+1:it*n), b_prec(1:n), n, MPI_GLOB, MPI_COMM_N)

      ! --- v_j+1 = A * b_prec = A * M^-1 * v_j --- 
      call bcsr_matv(a_mat, b_prec(1:n), V(it*n+1:(it+1)*n))

      ! --- Orthogonalization ---
      if (GSC) then ! Gram-Schmidt Classical
        call dgemv('C', n, it, 1.d0, V(1), n, V(it*n+1), 1, 0.d0, hess((it-1)*ldh+1), 1)
        call dgemv('N', n, it, -1.d0, V(1), n, hess((it-1)*ldh+1), 1, 1.d0, V(it*n+1), 1)
      elseif (GSM) then ! Gram-Schmidt Modified
        do k=1,it
          hess(k+(it-1)*ldh) = ddot(n, V((k-1)*n+1), 1, V(it*n+1), 1)
          call daxpy(n, -hess(k+(it-1)*ldh), V((k-1)*n+1), 1, V(it*n+1), 1)
        enddo
      elseif (GSCI) then ! Gram-Schmidt Classical Iterative 
        norm_p = dnrm2(n, V(it*n+1), 1)
        hess((it-1)*ldh+1:(it-1)*ldh+1+it) = 0.d0
        do j=1,n_ortho
          call dgemv('C', n, it, 1.d0, V(1), n, V(it*n+1), 1, 0.d0, s_(1), 1)
          call dgemv('N', n, it, -1.d0, V(1), n, s_(1), 1, 1.d0, V(it*n+1), 1)
          call daxpy(it, 1.d0, s_(1), 1, hess((it-1)*ldh+1), 1)
          norm_p_new = dnrm2(n, V(it*n+1), 1)
          if (2.d0 * norm_p_new .gt. norm_p) exit ! Stopping criterion for iterative GS methods
          norm_p = norm_p_new
        enddo
      elseif (GSMI) then ! Gram-Schmidt Modified Iterative
        norm_p = dnrm2(n, V(it*n+1), 1)
        hess((it-1)*ldh+1:(it-1)*ldh+1+it) = 0.d0
        do j=1,n_ortho
          do k=1,it
            s_(k) = ddot(n, V((k-1)*n+1), 1, V(it*n+1), 1)
            call daxpy(n, -s_(k), V((k-1)*n+1), 1, V(it*n+1), 1)
          enddo
          call daxpy(it, 1.d0, s_(1), 1, hess((it-1)*ldh+1), 1)
          norm_p_new = dnrm2(n, V(it*n+1), 1)
          if (2.d0 * norm_p_new .gt. norm_p) exit ! Stopping criterion for iterative GS methods
          norm_p = norm_p_new
        enddo
      endif
      ! --- h_j+1,j = ||v_j+1||_2 ---
      hess(it+(it-1)*ldh+1) = dnrm2(n, V(it*n+1), 1)
      ! --- v_j+1 = v_j+1 / h_j+1,j --- 
      call dscal(n, 1./hess(it+(it-1)*ldh+1), V(it*n+1), 1)
      ! --- Givens Rotation ---
      do k = 1, it-1
        gamma = givens_c(k)*hess(k+(it-1)*ldh) + givens_s(k)*hess(k+(it-1)*ldh+1)
        hess(k+(it-1)*ldh+1) = -givens_s(k)*hess(k+(it-1)*ldh) + givens_c(k)*hess(k+(it-1)*ldh+1)
        hess(k+(it-1)*ldh) = gamma
      enddo
      delta = sqrt(abs(hess(it+(it-1)*ldh))**2 + hess(it+(it-1)*ldh+1)**2);
      givens_c(it) = hess(it+(it-1)*ldh) / delta
      givens_s(it) = hess(it+(it-1)*ldh+1) / delta
      hess(it+(it-1)*ldh) = givens_c(it)*hess(it+(it-1)*ldh) + givens_s(it)*hess(it+(it-1)*ldh+1)
      b_(it+1) = -givens_s(it)*b_(it)
      b_(it) = givens_c(it)*b_(it)
      rho = abs(b_(it+1))
      ! --- Normwise Backward Error ---
      nbe = rho / nbe_denom
      if (my_id.eq.0) write(*, "(A, X, I0, X, A, X, ES14.6, X, A, X, ES14.6, X, A, X, ES14.6)") "[GMRES] iteration", totit, "res =", rho, "rel.res =", rho/scale_norm, "nbe =", nbe
      if ((rho < atol).or.(rho/scale_norm < rtol).or.(nbe < rtol).or.(totit >= maxit)) then
        no_conv = .false.
        nrit = it-1
        solver%iter_gmres = totit
        exit
      endif

    enddo
    ! --- Solve upper triangular system b_ = H \ b_ ---
    call dtrsv('U', 'N', 'N', nrit+1, hess, ldh, b_, 1)
    ! --- Form z = V * b_ --- 
    call dgemv('N', n, nrit+1, 1.d0, V(1), n, b_(1), 1, 0.d0, b_prec(1:n), 1)
    ! --- Update solution x = x + M^-1 * z ---
    call prec(solver, b_prec(1:n), V(1:n), n, MPI_GLOB, MPI_COMM_N)
    call daxpy(n, 1.d0, V(1:n), 1, x(1:n), 1)

  enddo

  deallocate(givens_c,givens_s,b_,hess,V,b_prec)
  if (GSCI .or. GSMI) deallocate(s_) 

end subroutine gmres2_driver


!> apply preconditioner y = M\x
  subroutine prec(solver, x, b, n_glob, MPI_GLOB, MPI_COMM_N)
    use mod_sparse_data, only: type_SP_SOLVER
#ifdef USE_STRUMPACK
    use mod_strumpack, only: strumpack_solve
#endif
#ifdef USE_PASTIX
    use mod_pastix, only: pastix_solve
#endif
#ifdef USE_MUMPS
    use mod_mumps, only: mumps_solve
#endif
    implicit none

    type(type_SP_SOLVER)         :: solver

    !real(kind=8), pointer :: x(:), b(:)
    real(kind=8), dimension(:), intent(inout) :: x, b
    integer :: n_glob
    integer :: i
    integer :: ierr, MPI_GLOB, MPI_COMM_N, my_id_n, my_id
    real :: t0, t1, t2
    real(kind=8), external :: dnrm2

    call MPI_COMM_RANK(MPI_COMM_N, my_id_n, ierr)
    call MPI_COMM_RANK(MPI_GLOB, my_id, ierr)

    !t0 = get_time()
    do i = 1, solver%pc%rhs%n
      solver%pc%rhs%val(i) = x(solver%pc%row_index(i))
    enddo

    !t1 = get_time()
    if (solver%library.eq.strumpack) then
#ifdef USE_STRUMPACK
      call strumpack_solve(solver%spss, solver%pc%rhs)
#endif
    elseif (solver%library.eq.pastix) then
#ifdef USE_PASTIX
      call pastix_solve(solver%ptss, solver%pc%rhs)
#endif
    elseif (solver%library.eq.mumps) then
#ifdef USE_MUMPS
      call mumps_solve(solver%mmss, solver%pc%rhs)
#endif
    endif

    !if (my_id_n.eq.0) write(*,*) my_id, "gmres pc solve time", get_time() - t1

    b = 0.d0
    if (my_id_n.eq.0) then
      do i = 1, solver%pc%rhs%n
        b(solver%pc%row_index(i)) = solver%pc%rhs%val(i)*solver%pc%row_factor
      enddo
    endif
    !call MPI_BARRIER(MPI_GLOB,ierr)
    call MPI_AllReduce(MPI_IN_PLACE,b,n_glob,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_GLOB,ierr)
    ! now all ranks have the global solution vector

    !if (my_id_n.eq.0) write(*,*) my_id, "gmres pc time", get_time() - t0

  end subroutine prec

end module mod_gmres2
