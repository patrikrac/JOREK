Module mod_spline_routines
  implicit none
  private
  public tg02a, tb15a

contains

  subroutine tg02a(ix,n,x0,f,d,x,v)

    implicit none
    !------------------------------------------------------------------
    ! Routine for interpolation of splines and their derivative
    !------------------------------------------------------------------
    !    n  : number of points
    !    ix : negative 0 -> no initial guess for where xi is
    !        positive -> gues for index close to value x
    !   x0  : the coordinates of the spline points
    !   f   : the function values of the spline points
    !   d   : the derivatives on the spline points
    !   x   : the coordinate where the output is wanted
    !   v(1-4) : value and derivatives of the spline interpolation
    !------------------------------------------------------------------

    real*8, intent(in)      :: x     ! Evaluation point
    integer, intent(in)     :: ix,n  ! ix<=0: no initial guess
    ! ix>0 : index of first guess 
    real*8, intent(in)      :: d(:)  ! Derivate at knots
    real*8, intent(in)      :: f(:)  ! Value at knots
    real*8, intent(in)      :: x0(:) ! knot coordinates
    real*8, intent(inout)   :: v(4)  ! value at x and derivatives

    ! Local variables
    integer :: j
    logical :: found
    real*8  :: h00, h10, h01, h11, theta, theta2, theta3, t, eps, dx

    found = .false.
    eps = 1.d-30
    write(*,*) n, size(x0), x
    if (x .le. x0(1)) then
       if (dabs(x-x0(1)) .lt. eps*dabs(x0(n)-x0(1))) then
          j = 1
       else
          write(*,*) "Warning: interpolation below range"
          v(1) = f(1)
          v(2:4) = 0.d0
          return
       end if
    else if (x .ge. x0(n)) then
       if (dabs(x-x0(n)) .lt. eps * dabs(x0(n)-x0(1))) then
          j = n-1
       else
          write(*,*) "Warning: interpolation below range"
          v(1) = f(n)
          v(2:4) = 0.d0
          return
       end if
    else 
       if (ix .gt. 0) j = max(min(ix,n-1),1)
       if (ix .le. 0) j = (x-x0(1))/(x0(n)-x0(1))*(n-1)+1 ! initial guess of value

       do while  (.not. found)
          if (x .gt. x0(j+1)) then
             j = j + 1
          else if (x .ge. x0(j)) then
             found = .true.
          else
             j = j - 1
          end if
       end do
    end if

    dx = x0(j+1) - x0(j)
    theta = (x-x0(j))/dx
    theta2 = theta*theta
    theta3 = theta2*theta

    h00 = 2.d0*theta3 - 3*theta2 + 1
    h10 = theta3 - 2.d0*theta2 + theta
    h01 = -2.d0*theta3 + 3*theta2
    h11 = theta3 - theta2

    ! Calculate spline values and their derivative
    v(1) = h00*f(j) + h10*dx*d(j) + h01*f(j+1) + h11*dx*d(j+1)
    v(2) = 6.d0*(theta2 - theta)*f(j) + (3.d0*theta2 - 4.d0*theta + 1)*dx*d(j) + 6.d0*(-theta2 + theta)*f(j+1) + (3.d0*theta2 - 2.d0*theta)*dx*d(j+1)
    v(2) = v(2)/dx
    v(3) = (12.d0*theta - 6.d0)*f(j) + (6.d0*theta - 4.d0)*dx*d(j) + (-12.d0*theta + 6.d0)*f(j+1) + (6.d0*theta - 2.d0)*dx*d(j+1)
    v(3) = v(3)/(dx**2)
    v(4) = 6.d0*(2*f(j) - 2.d0*f(j+1) + dx*d(j) + dx*d(j+1))/(dx**3)

  end subroutine tg02a

  subroutine tb15a(n,x,f,d,w,lp)
    !-------------------------------
    ! Calculate periodic spline parameters
    !-------------------------------
    implicit none

    integer,intent(in) :: n,lp      ! Number of points, unit number 
    real(8),intent(in) :: x(n),f(n) ! evaluation points and function values
    real(8),intent(out):: d(n)      ! Result: Derivatives at knots
    real(8),intent(inout):: w(*)    ! Workspace for comptability reasons

    real(8) :: a(n-1),b(n-1),c(n-1),rhs(n-1)
    real(8) :: sol(n-1)

    real(8) :: h1,h2,f1,f2
    integer :: i,m

    if (f(n) .ne. f(1)) then
       write(*,*) "Function values must be periodic", f(n), f(1)
       stop
    end if

    if (n < 4) then
       write(*,*) "More than 4 points needed for spline calculation"
       stop
    endif

    do i = 2, n
       if (x(i) .le. x(i-1)) then
          write(*,*) "Coordinate points must be increasing"
          stop
       end if
    end do

    m=n-1

    !------------------------------------------------------
    ! Build equations for unknowns:
    !
    ! sol(1)=d(2)
    ! ...
    ! sol(n-1)=d(n)
    !
    !------------------------------------------------------

    do i=2,n
       if(i==n) then
          h2 = 1.d0/(x(2)-x(1))
          f2 = f(2)
       else
          h2 = 1.d0/(x(i+1)-x(i))
          f2 = f(i+1)
       endif

       h1 = 1.d0/(x(i)-x(i-1))
       f1=f(i-1)
       a(i-1)=h1             ! Subdiagonal
       b(i-1)=2.d0*(h1+h2)   ! Diagonal
       c(i-1)=h2             ! Supdiagonal

       rhs(i-1)=3.d0*( &
            f2*h2*h2 &
            + f(i)*(h1*h1-h2*h2) &
            - f1*h1*h1 )
    enddo

    !------------------------------------------------------
    ! Cyclic corner terms
    !
    ! first row:
    ! b1*x1+c1*x2+beta*xm
    !
    ! last row:
    ! alpha*x1+a_m*x_{m-1}+b_m*xm
    !
    !------------------------------------------------------
    call cyclic_solve(m,a,b,c,a(1),c(m),rhs,sol)

    do i=2,n
       d(i)=sol(i-1)
    enddo

    d(1)=d(n)

  end subroutine tb15a

  subroutine cyclic_solve(n,a,b,c,alpha,beta,r,x)

    implicit none

    integer,intent(in)::n
    real(8),intent(in)::a(n),b(n),c(n)
    real(8),intent(in)::alpha,beta
    real(8),intent(in)::r(n)
    real(8),intent(out)::x(n)

    real(8)::bb(n),u(n),z(n)
    real(8)::gamma,factor
    integer::i

    gamma=-b(1)

    bb=b
    bb(1)=b(1)-gamma
    bb(n)=b(n)-alpha*beta/gamma

    call tridag(n,a,bb,c,r,x)

    u=0.d0
    u(1)=gamma
    u(n)=alpha

    call tridag(n,a,bb,c,u,z)

    factor=(x(1)+beta*x(n)/gamma) / &
         (1.d0+z(1)+beta*z(n)/gamma)

    do i=1,n
       x(i)=x(i)-factor*z(i)
    enddo

  end subroutine cyclic_solve

  
  subroutine tridag(n,a,b,c,r,u)

    implicit none

    integer,intent(in)::n

    real(8),intent(in)::a(n),b(n),c(n),r(n)
    real(8),intent(out)::u(n)

    real(8)::gam(n)
    real(8)::bet
    integer::i

    bet=b(1)

    if(bet==0.d0) stop "zero pivot"

    u(1)=r(1)/bet

    do i=2,n
       gam(i)=c(i-1)/bet
       bet=b(i)-a(i)*gam(i)
       if(bet==0.d0) stop "zero pivot"
       u(i)=(r(i)-a(i)*u(i-1))/bet
    enddo

    do i=n-1,1,-1
       u(i)=u(i)-gam(i+1)*u(i+1)
    enddo

  end subroutine tridag

end module mod_spline_routines
