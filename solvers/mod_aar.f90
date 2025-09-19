module mod_aar
!#ifdef USE_GMRES
  use iso_c_binding
  use mpi_mod
  use mod_sparse_data, only: pastix, mumps, strumpack
  use mod_integer_types

  private
  public :: aar_driver
  
  interface
    subroutine cmatv(x, y, a, iptr, jcn, csr, n, nl, block_size, gpu, commg) bind(C)
      use iso_c_binding
      implicit none
      integer :: commg
      integer :: n, nl, block_size
      type(c_ptr) :: x, y, a, iptr, jcn, csr
      logical :: gpu
    end subroutine cmatv
  end interface

contains

subroutine aar_driver(a_mat,b,x,n,solver)
  use mod_sparse_data, only: type_SP_SOLVER
  use data_structure,  only: type_SP_MATRIX

  implicit none
  type(type_SP_MATRIX)                :: a_mat
  real(kind=8), dimension(:), pointer :: x, b
  integer :: n
  type(type_SP_SOLVER)  :: solver

  real(kind=8) :: atol, rtol, gamma, delta, rho, rho0=0.0, omega, beta
  integer :: totit, maxit, m, nrit, it, ldh, k, j, p
  integer :: info
  logical :: no_conv
  real(kind=8), dimension(:), allocatable, target :: X_, F_, r, r_prev, x_prev, z, FtF, ipiv 

  integer :: my_id, my_id_n, n_cpu, ierr
  integer :: MPI_GLOB, MPI_COMM_N

  external :: dcopy, daxpby, daxpy, dgemv, dgemm, dscal, dgesv
  real(kind=8), external :: dnrm2, ddot

  MPI_GLOB   = a_mat%comm
  MPI_COMM_N = solver%pc%MPI_COMM_N

  call MPI_COMM_RANK(MPI_GLOB, my_id, ierr)
  call MPI_COMM_SIZE(MPI_GLOB, n_cpu, ierr)
  call MPI_COMM_RANK(MPI_COMM_N, my_id_n, ierr)

  rtol = 1.d-8
  atol = 1.d-36
  maxit = 1.d5
  m = 5
  omega = 1.d0
  beta = 0.9
  p = 5

  allocate(X_(n*m), F_(n*m), r(n), r_prev(n), x_prev(n), z(m), FtF(m*m), ipiv(m))

  ! --- r = A * x ---
  call cmatv(c_loc(x), c_loc(r), c_loc(a_mat%val), c_loc(a_mat%iptr), c_loc(a_mat%jcn), &
              c_loc(a_mat%coo_to_csr_map), a_mat%ng, a_mat%nr, a_mat%block_size, solver%gpu, a_mat%comm)

  ! --- r = b - r (residual) ---
  call daxpby(n, 1.d0, b(1:n), 1, -1.d0, r(1:n), 1)

  no_conv = .true.
  totit = 0

  rho = dnrm2(n, r(1:n), 1)
  rho0 = rho

  if ((rho/rho0 < rtol) .or. (rho < atol)) then
    no_conv = .false.
  endif

  nrit = 1
  do while (no_conv)
    ! --- r = M^-1 r ---
    call prec(solver, r(1:n), r(1:n), n, MPI_GLOB, MPI_COMM_N)

    if (nrit > 1) then
      k = mod(nrit-2, m) + 1
      ! --- X(:,k) = x_prev - x ---
      call dcopy(n, x, 1, X_((k-1)*n+1), 1)
      call daxpby(n, 1.d0, x_prev(1:n), 1, -1.d0, X_((k-1)*n+1), 1)
      ! --- F(:,k) = r_prev - r ---
      call dcopy(n, r, 1, F_((k-1)*n+1), 1)
      call daxpby(n, 1.d0, r_prev(1:n), 1, -1.d0, F_((k-1)*n+1), 1)
    endif

    ! --- Save previos values --- 
    call dcopy(n, x, 1, x_prev, 1)
    call dcopy(n, r, 1, r_prev, 1)

    ! Update step
    if (mod(nrit, p) .ne. 0) then
      ! write(*,*) "Richardson"
      ! Richardson step
      call daxpy(n, omega, r(1:n), 1, x(1:n), 1)
    else
      ! write(*,*) "Anderson"
      ! Anderson extrapolation update
      call daxpy(n, beta, r(1:n), 1, x(1:n), 1)

      ! --- FtF =  F' F ---
      call dgemm('T', 'N', m, m, n, 1.d0, F_(1), n, F_(1), n, 0.d0, FtF(1), m)

      ! --- z = F' r --- 
      call dgemv('T', n, m, 1.d0, F_(1), n, r(1), 1, 0.d0, z(1), 1)

      ! --- z = (F'F) \ F'r ---
      call dgesv(m, 1, FtF(1), m, ipiv, z(1), m, info) ! More optimized routines exist

      ! --- r = Xz --- 
      call dgemv('N', n, m, 1.d0, X_(1), n, z(1), 1, 0.d0, r(1), 1)

      ! --- r = r + beta*Fz ---
      call dgemv('N', n, m, beta, F_(1), n, z(1), 1, 1.d0, r(1), 1)

      ! --- Final x update ---
      call daxpy(n, -1.d0, r(1:n), 1, x(1:n), 1)
    endif

    nrit = nrit + 1
    totit = totit + 1
    call cmatv(c_loc(x), c_loc(r), c_loc(a_mat%val), c_loc(a_mat%iptr), c_loc(a_mat%jcn), &
              c_loc(a_mat%coo_to_csr_map), a_mat%ng, a_mat%nr, a_mat%block_size, solver%gpu, a_mat%comm)

    ! --- r = b - r (residual) ---
    call daxpby(n, 1.d0, b(1:n), 1, -1.d0, r(1:n), 1)
    rho = dnrm2(n, r(1:n), 1)
    if (my_id.eq.0) write(*,"(A8,X,I4,X,A4,X,E14.6,X,A8,X,E14.6)") "AAR it", totit, "res=", rho, "rel.res=", rho/rho0
    if ((rho < atol).or.(rho/rho0 < rtol).or.(totit >= maxit)) then
        no_conv = .false.
        solver%iter_gmres = totit
        exit
    endif
    
  enddo

  deallocate(X_, F_, r, r_prev, x_prev, z, FtF, ipiv)
  
end subroutine aar_driver

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

end module mod_aar