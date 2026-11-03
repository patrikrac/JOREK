
module helena_boundary
  integer             :: mf, n_bnd
  real*8, allocatable :: fr(:), R_bnd(:), Z_bnd(:)
  real*8              :: Bgeo, Rgeo, Zgeo, amin, eps, ellip, tria_u, tria_l, quad_u, quad_l
  real*8              :: Reast, Rwest
end

module helena_profiles
  integer             :: n_prof
  real*8,allocatable  :: psi(:),dp_dpsi(:),fdf_dpsi(:),p_psi(:),f_psi(:),zjz_psi(:),q(:)
  real*8              :: p_bnd
endmodule

program jorek_to_helena
use constants
use helena_boundary
use helena_profiles

implicit none

real*8  :: R_axis,Z_axis,F0,psi_bnd,psi_axis,psi_xpoint
real*8  :: current, beta_p, beta_t, beta_n
integer :: i

read(5,*) R_axis,Z_axis,F0
read(5,*) psi_bnd,psi_axis  !,psi_xpoint
read(5,*) n_bnd

allocate(r_bnd(n_bnd),z_bnd(n_bnd))

do i=1,n_bnd
  read(5,*) r_bnd(i),z_bnd(i)
!  write(*,*) r_bnd(i),z_bnd(i)
enddo

read(5,*) amin, Rgeo, Bgeo
read(5,*) current,beta_p,beta_t,beta_n

write(*,*) amin, Rgeo, Bgeo
write(*,*) current,beta_p,beta_t,beta_n

read(5,*) n_prof

n_prof = n_prof+1

allocate(psi(n_prof),dp_dpsi(n_prof),zjz_psi(n_prof),q(n_prof),p_psi(n_prof),f_psi(n_prof),fdf_dpsi(n_prof))

do i=2,n_prof
  read(5,*) psi(i),dp_dpsi(i),zjz_psi(i),q(i)
enddo

write(*,*) ' psi_axis : ',psi_axis
write(*,*) ' psi_bnd  : ',psi_bnd
write(*,*) ' psi_xp   : ',psi_xpoint

write(*,*) ' amin : ',amin
write(*,*) ' Rgeo : ',Rgeo
write(*,*) ' Bgeo : ',Bgeo
write(*,*) ' Current : ',current
write(*,*) ' Beta_p  : ',beta_p
write(*,*) ' Beta_t  : ',beta_t
write(*,*) ' Beta_n  : ',beta_n

psi(1) = psi_axis
dp_dpsi(1) = dp_dpsi(2)
zjz_psi(1) = zjz_psi(2)
q(1)       = q(2)

do i=1,n_prof
  fdf_dpsi(i) = 1. - (psi(i) - psi(1))/(psi(n_prof) - psi(1))
enddo

mf=128

call fshape

Bgeo    = F0 / Rgeo

write(*,*) ' TEST  B     = ',MU_zero, Rgeo**2, dp_dpsi(1), fdf_dpsi(1)

write(*,*)  ' export to HELENA namelist'

open(20,file='helena.nml')
write(20,*) ' &SHAPE IAS = 1, ISHAPE = 2,'
write(20,'(A,i5,A,i5)') '    MFM= ',mf,', MHARM = ',mf/2
do i=1,mf/2
  write(20,'(A,i4,A,e14.6,A,i4,A,e14.6)') '    FM(',2*i-1,')=',fr(2*i-1)/amin,' FM(',2*i,')=',fr(2*i)/amin
enddo
write(20,*) ' &END'
write(20,*) ' &PROFILE '
write(20,*) '    IPAI=11, EPI=1.0, FPI=1.0'
write(20,*) '    IGAM=11'
write(20,*) '    ICUR=11, ECUR=1.0, FCUR=1.0'
write(20,*) '    NPTS = ',n_prof
do i=1,n_prof
  write(20,'(A,i4,A,e12.4,A,i4,A,e12.4,A,i4,A,e12.4,A,i4,A,e12.4)') &
        '    DPR(',i,') = ',dp_dpsi(i),', DF2(',i,') = ',fdf_dpsi(i),', ZJZ(',i,') = ',zjz_psi(i),', QIN(',i,') = ',q(i)
enddo
write(20,*) ' &END'
write(20,*) ' &PHYS '
write(20,'(A,f10.6)') '   EPS   = ',amin/Rgeo
write(20,'(A,f10.6)') '   ALFA  = ',abs(amin**2 * Bgeo / (psi_bnd-psi_axis))
write(20,'(A,f10.6)') '   B     = ',MU_zero * Rgeo**2 * dp_dpsi(1)/fdf_dpsi(1)
write(20,'(A,f10.6)') '   BETAP = ',beta_p
write(20,'(A,e14.6)') '   XIAB  = ',MU_zero * abs(current) / (amin * Bgeo)
write(20,'(A,f10.6)') '   RVAC  = ',Rgeo
write(20,'(A,f10.6)') '   BVAC  = ',Bgeo
write(20,*) ' &END'
write(20,*) ' &NUM'
write(20,*) '    NR    = 51, NP    = 33, NRMAP  = 101,   NPMAP = 129, NCHI = 128'
write(20,*) '    NRCUR = 51, NPCUR = 33, ERRCUR = 1.e-5, NITER = 100, NMESH = 100'
write(20,*) ' &END'
write(20,*) ' &PRI  NPR1=1 &END '
write(20,*) ' &PLOT NPL1=1 &END '
write(20,*) ' &BALL        &END '
close(20)

end

subroutine fshape
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
use tr_module
use helena_boundary
use mod_spline_routines
use constants

implicit none

real*8              :: xj, yj, ga
real*8, allocatable :: THETA(:), GAMMA(:), XV(:),YV(:)
real*8              :: angle, error, gamm
real*8, allocatable :: tht_tmp(:),  fr_tmp(:), work(:)
real*8, allocatable :: tht_sort(:), fr_sort(:), dfr_sort(:)
integer, allocatable :: index_order(:)
real*8              :: tht, Rbnd_av, ORbnd_av, values(4)
integer             :: m, igrinv, i, j, ishape, ieast(1), iwest(1), n_bnd_short
parameter (error = 1.d-8)

allocate(fr(mf+2))
allocate(theta(mf),gamma(mf),xv(mf),yv(mf))

  write(*,*) ' fshape : (R,Z) set given on ',n_bnd,' points'

  Reast = maxval(R_bnd)
  Rwest = minval(R_bnd)
  ieast = maxloc(R_bnd)
  iwest = minloc(R_bnd)
  Rgeo = (Reast + Rwest) /2.
  Zgeo = (Z_bnd(ieast(1))+Z_bnd(iwest(1)))/2.
  amin = (Reast - Rwest)/2.

  write(*,'(A,3f12.8)') ' Rgeo, Zgeo : ',Rgeo,Zgeo,amin

  allocate(tht_tmp(n_bnd))
  allocate(fr_tmp(n_bnd))
  allocate(work(3*n_bnd+6))
  allocate(tht_sort(n_bnd+2))
  allocate(fr_sort(n_bnd+2))
  allocate(dfr_sort(n_bnd+2))
  allocate(index_order(n_bnd+2))

  do i=1,n_bnd

    tht_tmp(i) = atan2(Z_bnd(i)-Zgeo,R_bnd(i)-Rgeo)

    fr_tmp(i)  = sqrt((R_bnd(i)-Rgeo)**2 + (Z_bnd(i)-Zgeo)**2)

!    write(*,'(i5,2f12.8)') i,tht_tmp(i),fr_tmp(i)

  enddo

  if (abs(tht_tmp(n_bnd) - tht_tmp(1)) .lt. 1.e-6)  n_bnd = n_bnd - 1

  call qsort2(index_order,n_bnd,tht_tmp)

  do i=1,n_bnd
    tht_sort(i) = tht_tmp(index_order(i))
    fr_sort(i)  = fr_tmp(index_order(i))
!    write(*,*) i,tht_sort(i),fr_sort(i)
  enddo

  n_bnd_short = n_bnd
  do i=2,n_bnd

    if ((tht_sort(i) - tht_sort(1)) .gt. 2.*PI) then
      n_bnd_short = i - 1
      exit
    endif

  enddo

  if (abs(tht_sort(n_bnd_short)- tht_sort(1)) .lt. 1.e-6) then
    tht_sort(n_bnd_short) = tht_sort(1) + 2.*PI
    fr_sort(n_bnd_short)  = fr_sort(1)
  else
    tht_sort(n_bnd_short+1) = tht_sort(1) + 2.*PI
    fr_sort(n_bnd_short+1)  = fr_sort(1)
    n_bnd_short = n_bnd_short + 1
  endif

!  write(*,*) ' n_bnd, n_bnd_short : ',n_bnd, n_bnd_short

  call TB15A(n_bnd_short,tht_sort,fr_sort,dfr_sort,work,6)

  do i = 1, mf

   tht = 2.*PI * float(i-1)/float(mf)

   if (tht .lt. tht_sort(1))          tht = tht + 2.*PI
   if (tht .gt. tht_sort(n_bnd_short)) tht = tht - 2.*PI

   call TG02A(0,n_bnd_short,tht_sort,fr_sort,dfr_sort,tht,values)

   fr(i) = values(1)
  enddo

!-------------- FOURIER COEFFICIENTS FRFNUL AND FRF(M) OF FR(J).
call rft2(fr,mf,1)

do m=1,mf
  fr(m) = 2. * fr(m) / float(mf)
enddo
do m=2,mf,2
  fr(m) = - fr(m)
enddo
RETURN
END

SUBROUTINE QSORT2 (ORD,N,A)
!
!==============SORTS THE ARRAY A(I),I=1,2,...,N BY PUTTING THE
!   ASCENDING ORDER VECTOR IN ORD.  THAT IS ASCENDING ORDERED A
!   IS A(ORD(I)),I=1,2,...,N; DESCENDING ORDER A IS A(ORD(N-I+1)),
!   I=1,2,...,N .  THIS SORT RUNS IN TIME PROPORTIONAL TO N LOG N .
!
!
!     ACM QUICKSORT - ALGORITHM #402 - IMPLEMENTED IN FORTRAN BY
!                                 WILLIAM H. VERITY
!                                 COMPUTATION CENTER
!                                 PENNSYLVANIA STATE UNIVERSITY
!                                 UNIVERSITY PARK, PA.  16802
!     With correction to that algorithm.
!
      IMPLICIT INTEGER (A-Z)
!
      DIMENSION ORD(N),POPLST(2,20)
!
!     To sort different input types change the following
!     specification statements; FOR EXAMPLE,  REAL A(N) or
!     CHARACTER *(L) A(N)  for REAL or CHARACTER sorting
!     respectively  similarly for X,XX,Z,ZZ,Y. L is the
!     character length of the elements of A.
!
      REAL*8 A(N)
      REAL*8 X,XX,Z,ZZ,Y
!
      NDEEP=0
      U1=N
      L1=1
      DO 1  I=1,N
    1 ORD(I)=I
    2 IF (U1.GT.L1) GO TO 3
      RETURN
!
    3 L=L1
      U=U1
!
! PART
!
    4 P=L
      Q=U
      X=A(ORD(P))
      Z=A(ORD(Q))
      IF (X.LE.Z) GO TO 5
      Y=X
      X=Z
      Z=Y
      YP=ORD(P)
      ORD(P)=ORD(Q)
      ORD(Q)=YP
    5 IF (U-L.LE.1) GO TO 15
      XX=X
      IX=P
      ZZ=Z
      IZ=Q
!
! LEFT
!
    6 P=P+1
      IF (P.GE.Q) GO TO 7
      X=A(ORD(P))
      IF (X.GE.XX) GO TO 8
      GO TO 6
    7 P=Q-1
      GO TO 13
!
! RIGHT
!
    8 Q=Q-1
      IF (Q.LE.P) GO TO 9
      Z=A(ORD(Q))
      IF (Z.LE.ZZ) GO TO 10
      GO TO 8
    9 Q=P
      P=P-1
      Z=X
      X=A(ORD(P))
!
! DIST
!
   10 IF (X.LE.Z) GO TO 11
      Y=X
      X=Z
      Z=Y
      IP=ORD(P)
      ORD(P)=ORD(Q)
      ORD(Q)=IP
   11 IF (X.LE.XX) GO TO 12
      XX=X
      IX=P
   12 IF (Z.GE.ZZ) GO TO 6
      ZZ=Z
      IZ=Q
      GO TO 6
!
! OUT
!
   13 CONTINUE
      IF (.NOT.(P.NE.IX.AND.X.NE.XX)) GO TO 14
      IP=ORD(P)
      ORD(P)=ORD(IX)
      ORD(IX)=IP
   14 CONTINUE
      IF (.NOT.(Q.NE.IZ.AND.Z.NE.ZZ)) GO TO 15
      IQ=ORD(Q)
      ORD(Q)=ORD(IZ)
      ORD(IZ)=IQ
   15 CONTINUE
      IF (U-Q.LE.P-L) GO TO 16
      L1=L
      U1=P-1
      L=Q+1
      GO TO 17
   16 U1=U
      L1=Q+1
      U=P-1
   17 CONTINUE
      IF (U1.LE.L1) GO TO 18
!
! START RECURSIVE CALL
!
      NDEEP=NDEEP+1
      POPLST(1,NDEEP)=U
      POPLST(2,NDEEP)=L
      GO TO 3
   18 IF (U.GT.L) GO TO 4
!
! POP BACK UP IN THE RECURSION LIST
!
      IF (NDEEP.EQ.0) GO TO 2
      U=POPLST(1,NDEEP)
      L=POPLST(2,NDEEP)
      NDEEP=NDEEP-1
      GO TO 18
!
! END QSORT
END
