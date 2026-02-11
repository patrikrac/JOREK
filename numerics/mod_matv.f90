module mod_matv
  use mpi_mod
  use mod_sparse_matrix, only: type_SP_MATRIX
  implicit none
  private
  public :: matv, bcsr_matv, bcsr_matvT
contains

  subroutine matv(a_mat, x, b)
    implicit none

    type(type_SP_MATRIX)                    :: a_mat
    real(kind=8), dimension(:), intent(in)  :: x
    real(kind=8), dimension(:), intent(inout) :: b

    integer                           :: i, j, ir, jc
    integer                           :: ierr, my_id, n_cpu
    integer                           :: iA_start, ix_start, iy_start
    real(kind=8), allocatable         :: b_tmp_block(:), b_tmp(:)

    integer                           :: blocksize, blocksize2
    integer                           :: n_blocks, n_glob, nnz, index_offset, n_local
    integer, allocatable              :: rc(:), rd(:)
    
    integer                  :: cc, cr
    real                     :: t1,t0

    call system_clock(count=cc, count_rate=cr)
    t0 =  real(cc)/cr

    call MPI_COMM_RANK(a_mat%comm, my_id, ierr)
    call MPI_COMM_SIZE(a_mat%comm, n_cpu, ierr)

    ! set module values
    n_glob = a_mat%ng ! rank of global sparse matrix
    nnz = a_mat%nnz ! number of nonzero entries in the local piece of global sparse matrix

    blocksize = a_mat%block_size

    blocksize2 = blocksize*blocksize
    n_blocks = nnz/blocksize2

    index_offset = (a_mat%my_ind_min - 1)*blocksize
    n_local   = (a_mat%my_ind_max - a_mat%my_ind_min + 1)*a_mat%block_size

    allocate(b_tmp_block(a_mat%block_size))
    allocate(b_tmp(n_local))

    b_tmp(1:n_local) = 0.d0

  !$omp parallel                                    &
  !$omp private(i,iA_start,ix_start,iy_start,b_tmp_block)           &
  !$omp reduction(+:b_tmp)
  !$omp do schedule(guided)
    do i = 1, n_blocks

      iA_start = (i - 1)*blocksize2
      ix_start = a_mat%jcn(iA_start + 1)
      iy_start = a_mat%irn(iA_start + 1) - index_offset

      call dgemv('T', blocksize, blocksize, 1.d0, a_mat%val(iA_start + 1), blocksize, x(ix_start), 1, 0.d0, b_tmp_block, 1)

      b_tmp(iy_start:iy_start + blocksize - 1) = b_tmp(iy_start:iy_start + blocksize - 1) + b_tmp_block(1:blocksize)

    enddo
  !$omp end do
  !$omp end parallel

    allocate(rc(n_cpu),rd(n_cpu))
    call MPI_Allgather(n_local, 1, MPI_INT, rc, 1, MPI_INT, a_mat%comm, ierr)

    rd(1) = 0
    do i = 2, n_cpu
      rd(i) = rd(i-1) + rc(i-1)
    enddo

    call MPI_Allgatherv(b_tmp,n_local,MPI_DOUBLE_PRECISION,b,rc,rd,MPI_DOUBLE_PRECISION,a_mat%comm,ierr)
    deallocate(b_tmp, b_tmp_block)
    deallocate(rc,rd)

    call system_clock(count=cc, count_rate=cr)
    t1 =  real(cc)/cr
    if (my_id.eq.0) write(*,*) "Elapsed time matv (s) = ", t1 - t0

  end subroutine matv


  subroutine bcsr_matv(a_mat, x, b)
    implicit none

    type(type_SP_MATRIX)                    :: a_mat
    real(kind=8), dimension(:), intent(in)  :: x
    real(kind=8), dimension(:), intent(inout) :: b

    integer                           :: i, j, ir, jc
    integer                           :: ierr, my_id, n_cpu
    integer                           :: iA_start, ix_start, iy_start
    real(kind=8), allocatable         :: b_tmp_block(:), b_tmp(:)

    integer                           :: blocksize, blocksize2
    integer                           :: n_blocks, n_glob, nnz, index_offset, n_local, n_local_block
    integer, allocatable              :: rc(:), rd(:)
    logical                  :: verbose = .false.
    integer                  :: cc, cr
    real                     :: t1,t0

    integer :: block_count

    call system_clock(count=cc, count_rate=cr)
    t0 =  real(cc)/cr

    call MPI_COMM_RANK(a_mat%comm, my_id, ierr)
    call MPI_COMM_SIZE(a_mat%comm, n_cpu, ierr)

    ! set module values
    n_glob = a_mat%ng ! rank of global sparse matrix
    nnz = a_mat%nnz ! number of nonzero entries in the local piece of global sparse matrix

    blocksize = a_mat%block_size

    blocksize2 = blocksize*blocksize
    n_blocks = nnz/blocksize2

    index_offset = (a_mat%my_ind_min - 1)*blocksize
    n_local_block = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local   = n_local_block*a_mat%block_size

    allocate(b_tmp_block(a_mat%block_size))
    allocate(b_tmp(n_local))

    b_tmp(1:n_local) = 0.d0

    !$omp parallel                                    &
    !$omp private(i, iy_start,j,iA_start,ix_start,b_tmp_block) shared(b_tmp)     
    !$omp do schedule(guided)
    do i = 1, n_local_block
      iy_start = (i-1)*blocksize + 1
      ! Iterate over nnz blocks of the current block row
      b_tmp_block = 0.d0
      do j = a_mat%iblockptr(i), a_mat%iblockptr(i+1)-1
        iA_start = (j-1)*blocksize2 + 1
        ix_start = a_mat%jcn(iA_start)
        ! Perform a dense block-vector multiplication
        call dgemv('T', blocksize, blocksize, 1.d0, a_mat%val(iA_start), blocksize, x(ix_start), 1, 1.d0, b_tmp_block, 1)
      enddo   
      b_tmp(iy_start:iy_start + blocksize - 1) = b_tmp(iy_start:iy_start + blocksize - 1) + b_tmp_block(1:blocksize)
    enddo  
    !$omp end do
    !$omp end parallel


    allocate(rc(n_cpu),rd(n_cpu))
    call MPI_Allgather(n_local, 1, MPI_INT, rc, 1, MPI_INT, a_mat%comm, ierr)

    rd(1) = 0
    do i = 2, n_cpu
      rd(i) = rd(i-1) + rc(i-1)
    enddo

    call MPI_Allgatherv(b_tmp,n_local,MPI_DOUBLE_PRECISION,b,rc,rd,MPI_DOUBLE_PRECISION,a_mat%comm,ierr)
    deallocate(b_tmp, b_tmp_block)
    deallocate(rc,rd)

    call system_clock(count=cc, count_rate=cr)
    t1 =  real(cc)/cr
    if (verbose .and. my_id.eq.0) write(*,*) "Elapsed time matv (s) = ", t1 - t0

  end subroutine bcsr_matv


  subroutine bcsr_matvT(a_mat, x, b)
    type(type_SP_MATRIX)                    :: a_mat
    real(kind=8), dimension(:), intent(in)  :: x
    real(kind=8), dimension(:), intent(inout) :: b

     integer                           :: i, j, ir, jc
    integer                           :: ierr, my_id, n_cpu
    integer                           :: iA_start, ix_start, iy_start

    integer                           :: blocksize, blocksize2
    integer                           :: n_blocks, n_glob, nnz, index_offset, n_local, n_local_block
    integer, allocatable              :: rc(:), rd(:)
    logical                  :: verbose = .false.
    integer                  :: cc, cr
    real                     :: t1,t0

    integer :: block_count

    call system_clock(count=cc, count_rate=cr)
    t0 =  real(cc)/cr

    call MPI_COMM_RANK(a_mat%comm, my_id, ierr)
    call MPI_COMM_SIZE(a_mat%comm, n_cpu, ierr)

    ! set module values
    n_glob = a_mat%ng ! rank of global sparse matrix
    nnz = a_mat%nnz ! number of nonzero entries in the local piece of global sparse matrix

    blocksize = a_mat%block_size

    blocksize2 = blocksize*blocksize
    n_blocks = nnz/blocksize2

    index_offset = (a_mat%my_ind_min - 1)*blocksize
    n_local_block = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local   = n_local_block*a_mat%block_size

    b = 0.d0

    !$omp parallel                                    &
    !$omp private(i, iy_start,j,iA_start,ix_start) reduction(+:b)
    !$omp do schedule(guided)
    do i = 1, n_local_block
      iy_start = (i-1)*blocksize + 1
      ! Iterate over nnz blocks of the current block row
      do j = a_mat%iblockptr(i), a_mat%iblockptr(i+1)-1
        iA_start = (j-1)*blocksize2 + 1
        ix_start = a_mat%jcn(iA_start)
        ! Perform a dense block-vector multiplication
        call dgemv('N', blocksize, blocksize, 1.d0, a_mat%val(iA_start), blocksize, x(iy_start + index_offset), 1, 1.d0, b(ix_start:ix_start+blocksize-1), 1)
      enddo   
    enddo  
    !$omp end do
    !$omp end parallel

    call MPI_Allreduce(MPI_IN_PLACE, b, n_glob, MPI_DOUBLE_PRECISION, MPI_SUM, a_mat%comm, ierr)

    call system_clock(count=cc, count_rate=cr)
    t1 =  real(cc)/cr
    if (verbose .and. my_id.eq.0) write(*,*) "Elapsed time transposed matv (s) = ", t1 - t0
  end subroutine bcsr_matvT

end module mod_matv