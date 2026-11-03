module mod_polygon
  implicit none

  private
  
  public :: inside_polygon 

contains

! --- Check if point is inside polygon
pure logical function inside_polygon(N, rpol, zpol, r, z)

  implicit none
  
  ! --- Input variables
  integer, intent(in)  :: N
  real*8,  intent(in)  :: rpol(N), zpol(N), r, z
  
  ! --- Local variables
  integer :: counter, i
  real*8  :: r1, z1, r2, z2, xinters
  
  inside_polygon = .false.
  
  counter = 0 ! keep track of how many polygon lines a horizontal line from the test point to the right intersects
  r1 = rpol(N)
  z1 = zpol(N)
  do i = 1,N
    r2 = rpol(i)
    z2 = zpol(i)
    if ( (z .ge. min(z1,z2)) .and. (z .le. max(z1,z2)) .and. (r .lt. max(r1,r2)) ) then
      if (z1 .ne. z2) then
        xinters = (z-z1)*(r2-r1)/(z2-z1) + r1
        if ((r .lt. xinters) .and. (z .ne. z2)) then
          counter = counter + 1
        endif
      endif
    endif
    r1 = r2
    z1 = z2
  enddo

  ! --- Outside the polygon if the number of intersections is even
  if (mod(counter,2) .eq. 0) then
    inside_polygon = .false.
  ! --- Inside the polygon if the number of intersections is odd
  else
    inside_polygon = .true.
  endif
  
  return

end function inside_polygon

end module mod_polygon
