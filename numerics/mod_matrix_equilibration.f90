module mod_matrix_equilibration
  use mpi_mod
  use data_structure, only: type_SP_MATRIX
  implicit none

  public :: matrix_equilibration, scale_matrix, scale_vector_row, scale_vector_column
  private

  contains 

  subroutine matrix_equilibration(a_mat)
    type(type_SP_MATRIX), intent(inout) :: a_mat

    ! Internal variables 
    integer :: i, j
    integer :: n
    real*8 :: d1(a_mat%ng), d2(a_mat%ng)
    integer :: maxit, iter
    real*8 :: tol
    logical :: verbose
    logical :: conv, row_conv, col_conv
    integer :: ierr, my_id

    call MPI_COMM_RANK(a_mat%comm, my_id, ierr)

    !< Routine parameter setup
    maxit = 100
    tol = 1.0d-8
    conv = .false.
    verbose = .false.
    !< End setup

    iter = 0 !< Current iteration
    n = a_mat%ng

    ! Check if matrix is already equilibrated
    if (a_mat%equilibrated) then
      if (my_id == 0) print *, "Matrix is already equilibrated. Recomputing equilibration."
    end if

    if (.not. associated(a_mat%row_scaling)) then
      allocate(a_mat%row_scaling(n))
    endif
    a_mat%row_scaling = 1.d0

    if (.not. associated(a_mat%column_scaling)) then
      allocate(a_mat%column_scaling(n))
    end if
    a_mat%column_scaling = 1.d0

    do while (iter < maxit)
      d1 = 0.d0; d2 = 0.d0

      call get_scaling_factors(a_mat, d1, d2)

      do i = 1, n
        d1(i) = 1.0d0 / sqrt(d1(i))
        d2(i) = 1.0d0 / sqrt(d2(i))
      end do

      ! Check convergence 
      row_conv = maxval(abs(1.0d0 - (1.0d0 / d1)**2)) < tol
      col_conv = maxval(abs(1.0d0 - (1.0d0 / d2)**2)) < tol
      conv = row_conv .and. col_conv

      if (conv) then
        if (my_id == 0) print *, "Matrix equilibration converged in ", iter, " iterations."
        exit
      end if

      call scale_matrix(a_mat, d1, d2)

      a_mat%row_scaling =  a_mat%row_scaling * d1
      a_mat%column_scaling = a_mat%column_scaling * d2
    enddo

  end subroutine matrix_equilibration

  subroutine get_scaling_factors(a_mat, R, C)
    type(type_SP_MATRIX), intent(in) :: a_mat
    real*8, dimension(:), intent(out) :: R, C

    integer :: i, j, ib, jb
    integer :: row_idx, row_idx_start, col_idx, col_idx_start 
    integer :: val_idx, val_idx_start
    integer :: n_local, n_local_block
    real*8,   allocatable  :: R_local(:)
    integer,  allocatable  :: rc(:), rd(:)
    integer :: ierr, my_id, n_cpu

    call MPI_COMM_SIZE(a_mat%comm, n_cpu, ierr)
    n_local_block = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local = n_local_block * a_mat%block_size

    allocate(R_local(n_local))
    R_local = 0.d0

    do i = 1, n_local_block
      row_idx_start = (i-1)*a_mat%block_size + 1
      do j = a_mat%iblockptr(i), a_mat%iblockptr(i+1)-1
        val_idx_start = (j-1)*(a_mat%block_size*a_mat%block_size) + 1
        col_idx_start = a_mat%jcn(val_idx_start)
        do ib = 1, a_mat%block_size
          do jb = 1, a_mat%block_size
            row_idx = row_idx_start + ib - 1
            col_idx = col_idx_start + jb - 1
            val_idx = val_idx_start + (ib-1)*a_mat%block_size + (jb-1)

            R_local(row_idx) = max(R_local(row_idx), abs(a_mat%val(val_idx)))
            C(col_idx) = max(C(col_idx), abs(a_mat%val(val_idx)))
          end do
        end do
      enddo   
    enddo  

    ! Communicate to get global max values for rows and columns
    call MPI_AllReduce(C, C, a_mat%ng, MPI_DOUBLE_PRECISION, MPI_MAX, a_mat%comm, ierr)

    allocate(rc(n_cpu),rd(n_cpu))
    call MPI_Allgather(n_local, 1, MPI_INT, rc, 1, MPI_INT, a_mat%comm, ierr)
    rd(1) = 0
    do i = 2, n_cpu
      rd(i) = rd(i-1) + rc(i-1)
    enddo
    call MPI_Allgatherv(R_local,n_local,MPI_DOUBLE_PRECISION,R,rc,rd,MPI_DOUBLE_PRECISION,a_mat%comm,ierr)
    deallocate(R_local)
    deallocate(rc,rd)
  end subroutine get_scaling_factors


  subroutine scale_matrix(a_mat, R, C)
    type(type_SP_MATRIX), intent(inout) :: a_mat
    real*8, dimension(:), intent(out) :: R, C

    integer :: i, j, ib, jb
    integer :: row_idx, row_idx_start, col_idx, col_idx_start 
    integer :: val_idx, val_idx_start
    integer :: n_local_block

    n_local_block = a_mat%my_ind_max - a_mat%my_ind_min + 1

    do i = 1, n_local_block
      row_idx_start = (i-1)*a_mat%block_size + 1
      do j = a_mat%iblockptr(i), a_mat%iblockptr(i+1)-1
        val_idx_start = (j-1)*(a_mat%block_size*a_mat%block_size) + 1
        col_idx_start = a_mat%jcn(val_idx_start)
        do ib = 1, a_mat%block_size
          do jb = 1, a_mat%block_size
            row_idx = row_idx_start + ib - 1
            col_idx = col_idx_start + jb - 1
            val_idx = val_idx_start + (ib-1)*a_mat%block_size + (jb-1)

            a_mat%val(val_idx) = a_mat%val(val_idx) * R(row_idx) * C(col_idx)
          end do
        end do
      enddo   
    enddo 

  end subroutine scale_matrix

  subroutine scale_vector_row(a_mat, vec)
    type(type_SP_MATRIX), intent(in) :: a_mat
    real*8, dimension(:), intent(inout) :: vec

    integer :: i

    if (.not. a_mat%equilibrated) then
      return
    end if

    do i = 1, a_mat%ng
      vec(i) = vec(i) * a_mat%row_scaling(i)
    end do
  end subroutine scale_vector_row

  subroutine scale_vector_column(a_mat, vec)
    type(type_SP_MATRIX), intent(in) :: a_mat
    real*8, dimension(:), intent(inout) :: vec

    integer :: i

    if (.not. a_mat%equilibrated) then
      return
    end if

    do i = 1, a_mat%ng
      vec(i) = vec(i) * a_mat%column_scaling(i)
    end do
  end subroutine scale_vector_column

end module mod_matrix_equilibration