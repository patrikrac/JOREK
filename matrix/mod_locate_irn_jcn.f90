module mod_locate_irn_jcn
use mod_integer_types
implicit none
contains
subroutine locate_irn_jcn(index_node1,index_node2,index_min,index_max,ijA_position,a_mat)
use mod_integer_types
use data_structure, only: type_SP_MATRIX
!**************************************************************************
! subroutine finds the position in the global matrix of the index of      *
! node1 and node2 (this is the index per block)                           *
!                                                                         *
! Binary search: global_matrix_structure builds each irn_jcn row in strictly
! ascending order (it inserts in place and rejects duplicates), and this is an
! exact-match lookup, so the position found is identical to the linear scan
! this replaced - only the cost changes, from O(ijA_size) to O(log ijA_size)
! per element block.
!**************************************************************************
integer               :: index_node1, index_node2, index_min, index_max, index1_local
integer(kind=int_all) :: ijA_position, i, lo, hi, mid
logical               :: found_index
type(type_SP_MATRIX)  :: a_mat

found_index = .false.

index1_local = index_node1 - index_min + 1

lo = 1
hi = a_mat%ijA_size(index1_local)

do while (lo .le. hi)

  mid = lo + (hi - lo)/2

  if (a_mat%irn_jcn(index1_local,mid) .eq. index_node2) then
    ijA_position = a_mat%ijA_index(index1_local,mid)
    found_index = .true.
    exit
  elseif (a_mat%irn_jcn(index1_local,mid) .lt. index_node2) then
    lo = mid + 1
  else
    hi = mid - 1
  endif

enddo

if (.not.found_index) then

  write(*,*) ' FATAL locate_irn_jcn : index not found ',index_node1,index_node2

  do i=1,a_mat%ijA_size(index1_local)

    write(*,*) i, a_mat%irn_jcn(index1_local,i)

  enddo

  stop

endif

return
end subroutine locate_irn_jcn
end module mod_locate_irn_jcn
