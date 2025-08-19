module mod_gmres2
!#ifdef USE_GMRES
  use iso_c_binding
  use mpi_mod
  use mod_sparse_data, only: pastix, mumps, strumpack
  use mod_integer_types

  private
  public :: gmres2_driver
  
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

!> solve a_mat x=b using iterative GMRES method with left preconditioning
!! The code is based on the GMRES algorithm developed at
!! Lawrence Berkeley National Lab, Computational Research Division
subroutine gmres2_driver(a_mat,b,x,n,solver)
  use mod_sparse_data, only: type_SP_SOLVER
  use data_structure,  only: type_SP_MATRIX

  implicit none
  type(type_SP_MATRIX)                :: a_mat
  real(kind=8), dimension(:), pointer :: x, b
  integer :: n
  type(type_SP_SOLVER)  :: solver
  
  real(kind=8) :: atol, rtol, gamma, delta, rho, rho0=0.0
  integer :: totit, maxit, restart, nrit, it, ldh, k
  logical :: no_conv, GSC=.true., GSM=.false.
  real(kind=8), dimension(:), allocatable, target :: givens_c, givens_s, hess, V, b_prec, b_

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
  if (restart > maxit) restart = maxit

  allocate(givens_c(restart),givens_s(restart),b_(restart+1),hess((restart+1)*restart),V(n*(restart+1)),b_prec(n))
  givens_c(1:restart) = 0.
  givens_s(1:restart) = 0.

  ldh = restart+1
  call dcopy(n, b, 1, b_prec, 1)
  call prec(solver, b_prec, b_prec, n, MPI_GLOB, MPI_COMM_N)

  no_conv = .true.
  totit = 0;

  do while (no_conv)

    !call matv(a_mat, x, V(1:n),irn,jcn,val)
    call cmatv(c_loc(x), c_loc(V(1)), c_loc(a_mat%val), c_loc(a_mat%iptr), c_loc(a_mat%jcn), &
               c_loc(a_mat%coo_to_csr_map), a_mat%ng, a_mat%nr, a_mat%block_size, solver%gpu, a_mat%comm)
    !write(*,"(A5,X,E18.10)") "after", dnrm2(n, V(1:n), 1);
    call prec(solver, V(1:n), V(1:n), n, MPI_GLOB, MPI_COMM_N)
    !write(*,"(A5,X,E18.10)") "after", dnrm2(n, V(1:n), 1);

    call daxpby(n, 1.d0, b_prec(1:n), 1, -1.d0, V(1:n), 1);

    rho = dnrm2(n, V(1:n), 1);
    if (totit .eq. 0) rho0 = rho;
    if ((rho/rho0 < rtol) .or. (rho < atol)) then
      no_conv = .false.
      exit
    endif
    call dscal(n, 1./rho, V(1:n), 1)
    b_(1) = rho
    b_(2:restart+1) = 0.d0
    nrit = restart-1
    if (my_id.eq.0) write(*,*) "GMRES it. ", totit, "res = ", rho, "rel.res = ", rho/rho0, "restart!"

    do it = 1, restart
      totit = totit +1
      !call matv(a_mat, V((it-1)*n+1:(it-1)*n+n), V(it*n+1:it*n+n),irn,jcn,val)
      call cmatv(c_loc(V((it-1)*n+1)), c_loc(V(it*n+1)), c_loc(a_mat%val), c_loc(a_mat%iptr), c_loc(a_mat%jcn), &
                 c_loc(a_mat%coo_to_csr_map), a_mat%ng, a_mat%nr, a_mat%block_size, solver%gpu, a_mat%comm)
      call prec(solver, V(it*n+1:it*n+n), V(it*n+1:it*n+n), n, MPI_GLOB, MPI_COMM_N)
      if (GSC) then ! Gram-Schmidt Classical
        call dgemv('C', n, it, 1.d0, V(1), n, V(it*n+1), 1, 0.d0, hess((it-1)*ldh+1), 1)
        call dgemv('N', n, it, -1.d0, V(1), n, hess((it-1)*ldh+1), 1, 1.d0, V(it*n+1), 1)
      elseif (GSM) then ! Gram-Schmidt Modified
        do k=1,it
          hess(k+(it-1)*ldh) = ddot(n, V((k-1)*n+1), 1, V(it*n+1), 1)
          call daxpy(n, -hess(k+(it-1)*ldh), V((k-1)*n+1), 1, V(it*n+1), 1)
        enddo
      endif
      hess(it+(it-1)*ldh+1) = dnrm2(n, V(it*n+1), 1)
      call dscal(n, 1./hess(it+(it-1)*ldh+1), V(it*n+1), 1)
      do k = 1, it-1
        gamma = givens_c(k)*hess(k+(it-1)*ldh) + givens_s(k)*hess(k+(it-1)*ldh+1)
        hess(k+(it-1)*ldh+1) = -givens_s(k)*hess(k+(it-1)*ldh) + givens_c(k)*hess(k+(it-1)*ldh+1)
        hess(k+(it-1)*ldh) = gamma;
      enddo
      delta = sqrt(abs(hess(it+(it-1)*ldh))**2 + hess(it+(it-1)*ldh+1)**2);
      givens_c(it) = hess(it+(it-1)*ldh) / delta
      givens_s(it) = hess(it+(it-1)*ldh+1) / delta
      hess(it+(it-1)*ldh) = givens_c(it)*hess(it+(it-1)*ldh) + givens_s(it)*hess(it+(it-1)*ldh+1)
      b_(it+1) = -givens_s(it)*b_(it)
      b_(it) = givens_c(it)*b_(it)
      rho = abs(b_(it+1))
      if (my_id.eq.0) write(*,"(A8,X,I4,X,A4,X,E14.6,X,A8,X,E14.6)") "GMRES it", totit, "res=", rho, "rel.res=", rho/rho0
      if ((rho < atol).or.(rho/rho0 < rtol).or.(totit >= maxit)) then
        no_conv = .false.
        nrit = it-1
        solver%iter_gmres = totit
        exit
      endif

    enddo
    call dtrsv('U', 'N', 'N', nrit+1, hess, ldh, b_, 1)
    call dgemv('N', n, nrit+1, 1.d0, V(1), n, b_(1), 1, 1.d0, x, 1)

  enddo

  deallocate(givens_c,givens_s,b_,hess,V,b_prec)

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

!!> get matrix-vector product b=Ax
!  subroutine matv(a_mat, x, b)
!    use data_structure,  only: type_SP_MATRIX
!    implicit none
!
!    type(type_SP_MATRIX)                    :: a_mat
!    real(kind=8), dimension(:), intent(in)  :: x
!    real(kind=8), dimension(:), intent(inout) :: b
!
!    integer                           :: i, j, ir, jc
!    integer                           :: ierr, my_id, n_cpu
!    integer                           :: iA_start, ix_start, iy_start
!    real(kind=8), allocatable         :: b_tmp_block(:), b_tmp(:)
!
!    integer                           :: blocksize, blocksize2
!    integer                           :: n_blocks, n_glob, nnz, index_offset, n_local
!    integer, allocatable              :: rc(:), rd(:)
!    
!    integer                  :: cc, cr
!    real                     :: t1,t0
!
!    call system_clock(count=cc, count_rate=cr)
!    t0 =  real(cc)/cr
!
!    call MPI_COMM_RANK(a_mat%comm, my_id, ierr)
!    call MPI_COMM_SIZE(a_mat%comm, n_cpu, ierr)
!
!    ! set module values
!    n_glob = a_mat%ng ! rank of global sparse matrix
!    nnz = a_mat%nnz ! number of nonzero entries in the local piece of global sparse matrix
!
!    blocksize = a_mat%block_size
!
!    blocksize2 = blocksize*blocksize
!    n_blocks = nnz/blocksize2
!
!    index_offset = (a_mat%my_ind_min - 1)*blocksize
!    n_local   = (a_mat%my_ind_max - a_mat%my_ind_min + 1)*a_mat%block_size
!
!    allocate(b_tmp_block(a_mat%block_size))
!    allocate(b_tmp(n_local))
!
!    b_tmp(1:n_local) = 0.d0
!
!!$omp parallel                                    &
!!$omp private(i,iA_start,ix_start,iy_start,b_tmp_block)           &
!!$omp reduction(+:b_tmp)
!!$omp do schedule(guided)
!    do i = 1, n_blocks
!
!      iA_start = (i - 1)*blocksize2
!      ix_start = a_mat%jcn(iA_start + 1)
!      iy_start = a_mat%irn(iA_start + 1) - index_offset
!
!      call dgemv('T', blocksize, blocksize, 1.d0, a_mat%val(iA_start + 1), blocksize, x(ix_start), 1, 0.d0, b_tmp_block, 1)
!
!      b_tmp(iy_start:iy_start + blocksize - 1) = b_tmp(iy_start:iy_start + blocksize - 1) + b_tmp_block(1:blocksize)
!
!    enddo
!!$omp end do
!!$omp end parallel
!
!    allocate(rc(n_cpu),rd(n_cpu))
!    call MPI_Allgather(n_local, 1, MPI_INT, rc, 1, MPI_INT, a_mat%comm, ierr)
!    if (my_id.eq.0) write(*,*) "rc", rc(1:n_cpu)
!
!    rd(1) = 0
!    do i = 2, n_cpu
!      rd(i) = rd(i-1) + rc(i-1)
!    enddo
!    if (my_id.eq.0) write(*,*) "rc", rc(1:n_cpu)
!
!    !call MPI_Allgatherv(y_tmp,n_local,MPI_DOUBLE_PRECISION,y,rc,rd,MPI_DOUBLE_PRECISION,a_mat%comm,ierr)
!
!    call MPI_Allgatherv(b_tmp,n_local,MPI_DOUBLE_PRECISION,b,rc,rd,MPI_DOUBLE_PRECISION,a_mat%comm,ierr)
!    deallocate(b_tmp, b_tmp_block)
!    deallocate(rc,rd)
!
!    call system_clock(count=cc, count_rate=cr)
!    t1 =  real(cc)/cr
!    if (my_id.eq.0) write(*,*) "Elapsed time matv (s)", t1 - t0
!
!  end subroutine matv


!subroutine matv(a_mat,x,y,irn,jcn,val)
!    use data_structure,  only: type_SP_MATRIX
!    implicit none
!
!    type(type_SP_MATRIX)                    :: a_mat
!    real(kind=8), dimension(:), intent(in)  :: x
!    real(kind=8), dimension(:), intent(inout) :: y
!    integer(kind=c_int), dimension(:), pointer   :: irn
!    integer(kind=c_int), dimension(:), pointer   :: jcn
!    real(kind=c_double), dimension(:), pointer   :: val
!    
!    integer :: n
!
!    integer                           :: i, j, ir, jc
!    integer                           :: iA_start, ix_start, iy_start
!    real(kind=8), allocatable         :: y_tmp_block(:), y_tmp(:)
!
!    real(kind=8) :: ddum
!    integer, allocatable              :: rc(:), rd(:)
!    integer :: n_local
!
!    integer :: ndev, ddev, mydev, my_id, n_cpu, ierr
!    logical :: offload, initdev
!
!    integer                  :: cc, cr
!    real                     :: t1,t0
!
!    call system_clock(count=cc, count_rate=cr)
!    t0 =  real(cc)/cr
!
!    call MPI_Comm_rank(a_mat%comm, my_id, ierr)
!    call MPI_COMM_SIZE(a_mat%comm, n_cpu, ierr)
!
!    ndev = omp_get_num_devices()
!    ddev = omp_get_default_device()
!    write(*,*) my_id, "Number of devices:", ndev
!    write(*,*) my_id, "Default device:", ddev
!
!!$omp target defaultmap(tofrom:scalar)
!    initdev = omp_is_initial_device()
!!$omp end target
!    offload = .not.(initdev)
!    if (offload) then
!      mydev = 0
!    else
!      mydev = omp_get_initial_device()
!    endif
!    if (offload) then
!      write(*,*) my_id, "Able to use offloading", mydev
!    endif
!
!    n = a_mat%ng
!    n_local   = (a_mat%my_ind_max - a_mat%my_ind_min + 1)*a_mat%block_size
!    write(*,*) "n_local", n_local
!
!    allocate(y_tmp(n_local))
!
!    if (offload) then
!      if (my_id.eq.0) then
!        write(*,*) "Using GPU"
!      endif
!!$omp target enter data map(to: x(1:n)) map(alloc: y_tmp(1:n_local)) device(my_id)
!!$omp target teams distribute device(my_id)
!      do i = 1, n_local
!        ddum = 0.d0
!!$omp parallel do firstprivate(i) reduction(+:ddum)
!        do j = a_mat%irn(i), a_mat%irn(i+1) - 1
!          ddum = ddum + a_mat%val(j) * x(a_mat%jcn(j))
!        enddo
!        y_tmp(i) = ddum
!      enddo
!!$omp target exit data map(from: y_tmp(1:n_local)) map(delete: x(1:n)) device(my_id)
!    else
!      if (my_id.eq.0) then
!        write(*,*) "Using CPU"
!      endif
!!$omp parallel do
!      do i = 1, n_local
!        y_tmp(i) = 0.d0
!        do j = a_mat%irn(i), a_mat%irn(i+1) - 1
!          y_tmp(i) = y_tmp(i) + a_mat%val(j) * x(a_mat%jcn(j))
!        enddo
!      enddo
!    endif
!
!    allocate(rc(n_cpu),rd(n_cpu))
!    call MPI_Allgather(n_local, 1, MPI_INT, rc, 1, MPI_INT, a_mat%comm, ierr)
!
!    rd(1) = 0
!    do i = 2, n_cpu
!      rd(i) = rd(i-1) + rc(i-1)
!    enddo
!    if (my_id.eq.0) write(*,*) "rc", rc(1:n_cpu)
!
!    call MPI_Allgatherv(y_tmp,n_local,MPI_DOUBLE_PRECISION,y,rc,rd,MPI_DOUBLE_PRECISION,a_mat%comm,ierr)
!    deallocate(y_tmp)
!    deallocate(rc,rd)
!
!    call system_clock(count=cc, count_rate=cr)
!    t1 =  real(cc)/cr
!    if (my_id.eq.0) write(*,*) "Elapsed time matv (s)", t1 - t0
!
!  end subroutine matv

end module mod_gmres2
