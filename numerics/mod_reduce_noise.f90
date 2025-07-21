module mod_reduce_noise
  implicit none
  
  contains

  subroutine reduce_noise(a_mat, eps)
    use data_structure

    type(type_SP_MATRIX), intent(inout) :: a_mat ! Matrix that is considered for noise reduciton  
    real(8), intent(in) :: eps ! Tolerance for noise reduction

    integer :: i, count
    integer :: nnz

    ! Loop over all elements in the matrix
    nnz = a_mat%nnz
    count = 0

    do i=1, nnz
      ! Check if the value is below the threshold
      if (a_mat%val(i) == 0.d0) cycle

      if (abs(a_mat%val(i)) < eps) then
        ! Set the value to zero
        a_mat%val(i) = 0.0
        count = count + 1

        ! Maybe add something to ensure symmetry
      end if
    end do

    ! Print the number of elements reduced
    if (count > 0) then
      write(*,*) 'Reduced ', count, ' elements in the matrix.'
    else
      write(*,*) 'No elements reduced in the matrix.'
    end if

  end subroutine reduce_noise
  

end module mod_reduce_noise
