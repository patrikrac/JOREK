!> This module contains multi-purposes mathematical operators
module mod_math_operators
  implicit none

  private
  public :: cross_product
  public :: solve_2x2_linear_problem
  public :: solve_3x3_linear_problem
  public :: gcd

interface cross_product
  module procedure cross_product_r4
  module procedure cross_product_r8
end interface cross_product

interface solve_2x2_linear_problem
  module procedure solve_2x2_linear_problem_r4
  module procedure solve_2x2_linear_problem_r8
end interface solve_2x2_linear_problem

interface solve_3x3_linear_problem
  module procedure solve_3x3_linear_problem_r4
  module procedure solve_3x3_linear_problem_r8
end interface solve_3x3_linear_problem

contains

!> The vector cross product single precision
pure function cross_product_r4(a, b)
  real*4, dimension(3) :: cross_product_r4
  real*4, dimension(3), intent(in) :: a, b

  cross_product_r4(1) = a(2) * b(3) - a(3) * b(2)
  cross_product_r4(2) = a(3) * b(1) - a(1) * b(3)
  cross_product_r4(3) = a(1) * b(2) - a(2) * b(1)

end function cross_product_r4
  
!> The vector cross product double precision
pure function cross_product_r8(a, b)
  real*8, dimension(3) :: cross_product_r8
  real*8, dimension(3), intent(in) :: a, b

  cross_product_r8(1) = a(2) * b(3) - a(3) * b(2)
  cross_product_r8(2) = a(3) * b(1) - a(1) * b(3)
  cross_product_r8(3) = a(1) * b(2) - a(2) * b(1)

end function cross_product_r8

!> solve a 2x2 linear problem analytically (single precision)
subroutine solve_2x2_linear_problem_r4(A,b,x)
  implicit none
  real*4,dimension(2),intent(in)   :: b
  real*4,dimension(2,2),intent(in) :: A
  real*4,dimension(2),intent(out)  :: x

  x = (/A(2,2)*b(1)-A(1,2)*b(2),&
      A(1,1)*b(2)-A(2,1)*b(1)/)/&
      (A(1,1)*A(2,2)-A(2,1)*A(1,2))

end subroutine solve_2x2_linear_problem_r4

!> solve a 2x2 linear problem analytically (double precision)
subroutine solve_2x2_linear_problem_r8(A,b,x)
  implicit none
  real*8,dimension(2),intent(in)   :: b
  real*8,dimension(2,2),intent(in) :: A
  real*8,dimension(2),intent(out)  :: x

  x = (/A(2,2)*b(1)-A(1,2)*b(2),&
      A(1,1)*b(2)-A(2,1)*b(1)/)/&
      (A(1,1)*A(2,2)-A(2,1)*A(1,2))

end subroutine solve_2x2_linear_problem_r8

!> solve a 3x3 linear problem analytically (single precision)
subroutine solve_3x3_linear_problem_r4(A,b,x)
  implicit none
  real*4,dimension(3),intent(in)   :: b
  real*4,dimension(3,3),intent(in) :: A
  real*4,dimension(3),intent(out)  :: x
  real*4 :: det
  real*4,dimension(3,3) :: invA

  det = A(1,1)*(A(2,2)*A(3,3)-A(3,2)*A(2,3)) + &
        A(1,2)*(A(3,1)*A(2,3)-A(2,1)*A(3,3)) + &
        A(1,3)*(A(2,1)*A(3,2)-A(3,1)*A(2,2))
  invA(:,1) = (/A(2,2)*A(3,3)-A(3,2)*A(2,3),&
              A(3,1)*A(2,3)-A(2,1)*A(3,3),&
              A(2,1)*A(3,2)-A(3,1)*A(2,2)/)
  invA(:,2) = (/A(3,2)*A(1,3)-A(1,2)*A(3,3),&
              A(1,1)*A(3,3)-A(3,1)*A(1,3),&
              A(3,1)*A(1,2)-A(1,1)*A(3,2)/)
  invA(:,3) = (/A(1,2)*A(2,3)-A(2,2)*A(1,3),&
              A(2,1)*A(1,3)-A(1,1)*A(2,3),&
              A(1,1)*A(2,2)-A(2,1)*A(1,2)/)
  x = (/invA(1,1)*b(1)+invA(1,2)*b(2)+invA(1,3)*b(3),&
      invA(2,1)*b(1)+invA(2,2)*b(2)+invA(2,3)*b(3),&
      invA(3,1)*b(1)+invA(3,2)*b(2)+invA(3,3)*b(3)/)/det
  
end subroutine solve_3x3_linear_problem_r4
  
!> solve a 3x3 linear problem analytically (double precision)
subroutine solve_3x3_linear_problem_r8(A,b,x)
  implicit none
  real*8,dimension(3),intent(in)   :: b
  real*8,dimension(3,3),intent(in) :: A
  real*8,dimension(3),intent(out)  :: x
  real*8 :: det
  real*8,dimension(3,3) :: invA

  det = A(1,1)*(A(2,2)*A(3,3)-A(3,2)*A(2,3)) + &
        A(1,2)*(A(3,1)*A(2,3)-A(2,1)*A(3,3)) + &
        A(1,3)*(A(2,1)*A(3,2)-A(3,1)*A(2,2))
  invA(:,1) = (/A(2,2)*A(3,3)-A(3,2)*A(2,3),&
              A(3,1)*A(2,3)-A(2,1)*A(3,3),&
              A(2,1)*A(3,2)-A(3,1)*A(2,2)/)
  invA(:,2) = (/A(3,2)*A(1,3)-A(1,2)*A(3,3),&
              A(1,1)*A(3,3)-A(3,1)*A(1,3),&
              A(3,1)*A(1,2)-A(1,1)*A(3,2)/)
  invA(:,3) = (/A(1,2)*A(2,3)-A(2,2)*A(1,3),&
              A(2,1)*A(1,3)-A(1,1)*A(2,3),&
              A(1,1)*A(2,2)-A(2,1)*A(1,2)/)
  x = (/invA(1,1)*b(1)+invA(1,2)*b(2)+invA(1,3)*b(3),&
      invA(2,1)*b(1)+invA(2,2)*b(2)+invA(2,3)*b(3),&
      invA(3,1)*b(1)+invA(3,2)*b(2)+invA(3,3)*b(3)/)/det
  
end subroutine solve_3x3_linear_problem_r8

!> calculates the greatest common divisor (gcd) using the Euclidean algorithm
pure integer function gcd(n1,n2)
  implicit none
  integer, intent(in) :: n1, n2
  integer :: a,b,r !< remainder

  ! the gcd does not change when the larger number is replaced by the difference (e.g. when a > b, gcd(a,b) = gcd(a-b,b))
  ! so this is done iteratively until a divisor is found (which gives mod(a,b)=0)
  a=n1
  b=n2
  do
    r=mod(a,b)
    a=b
    b=r
    ! if b > a, one cycle will effectively swap a and b since mod(a,b)=a for b>a, making if statements for a>b unnecessary
    if(b==0) exit
  end do
  gcd = a
end function gcd

end module mod_math_operators
