subroutine mnewtax(node_list,element_list,i_elm, r, s, errx, errf, ifail)
!-------------------------------------------------------------------------
! solves two non-linear equations using Newtons method
! LU decomposition replaced by explicit solution of 2x2 matrix.
!-------------------------------------------------------------------------
use data_structure
use mod_interp, only: interp
implicit none

type (type_node_list)    :: node_list
type (type_element_list) :: element_list

real*8    :: r, s, x(2), FVEC(2),FJAC(2,2), ZPSI, ZPSIR, ZPSIS, ZPSIRS, ZPSIRR, ZPSISS
real*8    :: tolf,tolx, errf, errx, temp, dis, errf_min
integer   :: ntrial, i, k, ifail, i_elm
real*8    :: p(2)

ntrial = 20
tolx = 1.d-8
tolf = 1.d-16

x(1) = r
x(2) = s

ifail = 999

errf_min = 1.d99

do k=1,ntrial

  call interp(node_list,element_list,i_elm,1,1,x(1),x(2),ZPSI,ZPSIR,ZPSIS,ZPSIRS,ZPSIRR,ZPSISS)

!  write(*,'(A,8e12.4)') ' mnewt : ',x,ZPSI,ZPSIR,ZPSIS,ZPSIRS,ZPSIRR,ZPSISS

  FVEC(1)   = ZPSIR
  FVEC(2)   = ZPSIS
  FJAC(1,1) = ZPSIRR
  FJAC(1,2) = ZPSIRS
  FJAC(2,1) = ZPSIRS
  FJAC(2,2) = ZPSISS

  errf=abs(fvec(1))+abs(fvec(2))

  if (errf .le. tolf) then
    r = x(1)
    s = x(2)
    ifail = 0
    return
  endif

  p = -fvec

  temp = p(1)
  dis  = fjac(2,2)*fjac(1,1)-fjac(1,2)*fjac(2,1)
  if (dis .ne. 0.d0) then
    p(1) = (fjac(2,2)*p(1)-fjac(1,2)*p(2))/dis
    p(2) = (fjac(1,1)*p(2)-fjac(2,1)*temp)/dis
  endif

  errx=abs(p(1)) + abs(p(2))

  p = min(p,+0.25d0)
  p = max(p,-0.25d0)

  x = x + p

  x = max(x, 0.d0)
  x = min(x,+1.d0)

  if (errx .le. tolx) then
    r = x(1)
    s = x(2)
    ifail = 0
    return
  endif

  if (errf .lt. errf_min) then
    r = x(1)
    s = x(2)
    errf_min = errf
  endif

enddo

return
end

