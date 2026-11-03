module mod_mpi_stats
use mpi
implicit none
private
public mpi_stats, mpi_stats_list
contains
!> Calculate statistics over MPI
!> Return stats(1) = min, stats(2) = mean, stats(3) = max, stats(4) = stddev
function mpi_stats(var)
  real*8, intent(in) :: var
  real*8, dimension(5) :: mpi_stats
  integer :: n_mpi, ierr
  real*8, allocatable, dimension(:) :: vars

  call MPI_COMM_SIZE(MPI_COMM_WORLD, n_mpi, ierr)
  allocate(vars(n_mpi))
  vars = 0.d0

  call MPI_Gather(var, 1, MPI_REAL8, vars, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  mpi_stats(1) = minval(vars)
  mpi_stats(2) = sum(vars)/size(vars, 1)
  mpi_stats(3) = maxval(vars)
  mpi_stats(4) = sqrt(sum((vars-mpi_stats(2))**2)/size(vars,1))
  mpi_stats(5) = sum(vars)
end function mpi_stats

!> Calculate statistics on sets of numbers ignoring zeros
!> Performs MPI communication for the mean
!> Only returns usable values on the root node
!> Return stats(1) = min, stats(2) = mean, stats(3) = max, stats(4) = stddev
function mpi_stats_list(list) result(stats)
  real*8, intent(in), dimension(:) :: list
  real*8, dimension(5) :: stats !=(/minv, mean, maxv, sd, total/)
  integer :: num_values, ierr

  stats = 0.d0
  call MPI_Reduce(minval(list), stats(1), 1, MPI_REAL8, MPI_MIN, 0, MPI_COMM_WORLD, ierr)
  call MPI_Reduce(maxval(list), stats(3), 1, MPI_REAL8, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
  call MPI_Reduce(sum(list),    stats(5), 1, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  call MPI_Reduce(size(list,1), num_values, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

  stats(2) = stats(5)/real(num_values,8)
  call MPI_Bcast(stats(2), 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  call MPI_Reduce(sum((list-stats(2))**2), stats(4), 1, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
  stats(4) = sqrt(stats(4)/real(num_values,8))
end function mpi_stats_list
end module mod_mpi_stats
