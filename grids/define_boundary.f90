subroutine Define_Boundary
!---------------------------------------------------------------------
! subroutine to define the spline coefficients for the shape of the
! plasma boundary
!   ELLIP : ellipticity
!   TRIA  : triangularity
! output is contained in the module boundary
!---------------------------------------------------------------------
use constants
use tr_module
use phys_module
use mod_spline_routines

implicit none

integer             :: n_bnd, i, j, k, l, m, err
real*8, allocatable :: r_tmp(:),psi_tmp(:),dr_tmp(:),dpsi_tmp(:),tht_tmp(:)
real*8, allocatable :: Work(:)
real*8              :: Vr(4), Vpsi(4), RP, ZP, theta, tht_i
real*8              :: amp, Rm, Zm, dRm, dZm, dPsi

write(*,*) '*******************************************'
write(*,*) '*    Defining boundary                    *'
write(*,*) '*******************************************'

!------------------------------- boundary given by ellip, tria etc as splined in r_bnd, psi_bnd (module boundary)
if (mf .le. 0) then

  if (n_boundary .eq. 0) then

    write(*,'(A18,f8.4)')  '  ellipticity     : ',ellip
    write(*,'(A18,2f8.4)') '  triangularity   : ',tria_u,tria_l
    write(*,'(A18,2f8.4)') '  quadrangularity : ',quad_u,quad_l
    write(*,'(A18,f8.4)')  '  Xpoint(ampl)    : ',xampl
    write(*,'(A18,f8.4)')  '  Xpoint(width)   : ',xwidth
    write(*,'(A18,f8.4)')  '  Xpoint(sigma)   : ',xsig
    write(*,'(A18,f8.4)')  '  Xpoint(angle)   : ',xtheta

    n_bnd = 256
    call tr_allocate(tht_tmp,1,n_bnd,"tht_tmp",CAT_GRID)
    call tr_allocate(r_tmp,1,n_bnd,"r_tmp",CAT_GRID)
    call tr_allocate(dr_tmp,1,n_bnd,"dr_tmp",CAT_GRID)
    call tr_allocate(psi_tmp,1,n_bnd,"psi_tmp",CAT_GRID)
    call tr_allocate(dpsi_tmp,1,n_bnd,"dpsi_tmp",CAT_GRID)

    do i=1,n_bnd

      theta = 2.d0*PI * real(i-1)/real(n_bnd-1)

      if (theta .lt. pi) then

        RP = cos(theta + tria_u*sin(theta) + quad_u*sin(2.d0*theta))

      else

        RP = cos(theta + tria_l*sin(theta) + quad_l*sin(2.d0*theta))

      endif

      ZP = ellip * sin(theta)

      tht_tmp(i) = atan2(ZP,RP)
      if (tht_tmp(i) .lt. 0.d0*pi)    tht_tmp(i) = tht_tmp(i) + 2.d0*pi

      tht_i = tht_tmp(i)
      if (theta      .lt. 0.5d0*pi)   tht_i = tht_i + 2.d0*pi

      r_tmp(i)   = sqrt(rp**2+zp**2)

      if (xpoint) then
        psi_tmp(i) =  - xshift * sin(tht_i) + xleft * cos(tht_i) &
                 + xampl*(-1.d0 + (xwidth*(tht_i-xtheta)/xsig)**2)* exp( - ((tht_i-xtheta)/xsig)**2)
      else
        psi_tmp(i) = 0.d0
      endif

      ! --- Manipulate Psi boundary
      dPsi = 0.d0
      do l = 1, 5
        amp = manipulate_psi_map(l,1)
        Rm  = manipulate_psi_map(l,2)
        Zm  = manipulate_psi_map(l,3)
        dRm = manipulate_psi_map(l,4)
        dZm = manipulate_psi_map(l,5)
        dPsi = dPsi + amp * exp(-((Rp+R_geo)-Rm)**2/dRm**2-((Zp+Z_geo)-Zm)**2/dZm**2)
      end do
      psi_tmp(i) = psi_tmp(i) + dPsi

    enddo

    r_tmp(n_bnd)   = r_tmp(1)
    psi_tmp(n_bnd) = psi_tmp(1)

  else

    n_bnd = n_boundary

    call tr_allocate(tht_tmp,1,n_bnd,"tht_tmp",CAT_GRID)
    call tr_allocate(r_tmp,1,n_bnd,"r_tmp",CAT_GRID)
    call tr_allocate(dr_tmp,1,n_bnd,"dr_tmp",CAT_GRID)
    call tr_allocate(psi_tmp,1,n_bnd,"psi_tmp",CAT_GRID)
    call tr_allocate(dpsi_tmp,1,n_bnd,"dpsi_tmp",CAT_GRID)

    do i=1,n_bnd
      RP = R_boundary(i) - R_geo
      ZP = Z_boundary(i) - Z_geo

      tht_tmp(i) = atan2(ZP,RP)

      if (tht_tmp(i) .lt. 0.d0*pi)    tht_tmp(i) = tht_tmp(i) + 2.d0*pi

      tht_i = tht_tmp(i)
!      if (theta      .lt. 0.5d0*pi)   tht_i = tht_i + 2.d0*pi

      r_tmp(i)   = sqrt(rp**2+zp**2)

      if (xpoint) then
        psi_tmp(i) = psi_boundary(i) - xshift * sin(tht_i) + xleft * cos(tht_i) &
                   + xampl*(-1.d0 + (xwidth*(tht_i-xtheta)/xsig)**2)* exp( - ((tht_i-xtheta)/xsig)**2)
      else
        psi_tmp(i) = psi_boundary(i)
      endif

      if (i .gt. 1) then
        if (tht_tmp(i) .lt. tht_tmp(i-1)) then
          tht_tmp(i) = tht_tmp(i) + 2.d0*pi
        endif
      endif
      
      ! --- Manipulate Psi boundary
      dPsi = 0.d0
      do l = 1, 5
        amp = manipulate_psi_map(l,1)
        Rm  = manipulate_psi_map(l,2)
        Zm  = manipulate_psi_map(l,3)
        dRm = manipulate_psi_map(l,4)
        dZm = manipulate_psi_map(l,5)
        dPsi = dPsi + amp * exp(-(R_boundary(i)-Rm)**2/dRm**2-(Z_boundary(i)-Zm)**2/dZm**2)
      end do
      psi_tmp(i) = psi_tmp(i) + dPsi
    
    end do

    r_tmp(n_bnd)   = r_tmp(1)
    psi_tmp(n_bnd) = psi_tmp(1)
    tht_tmp(n_bnd) = tht_tmp(1) + 2.d0 * PI

    ! --- Debugging output:
    !do i=1,n_bnd
    !  write(*,*) tht_tmp(i),r_tmp(i),psi_tmp(i)
    !enddo

  endif

  call tr_allocate(work,1,3*n_bnd,"work",CAT_GRID)

  call TB15A(n_bnd,tht_tmp,r_tmp,dr_tmp,work,6)           ! periodic spline of the radius
  call TB15A(n_bnd,tht_tmp,psi_tmp,dpsi_tmp,work,6)       ! periodic spline of flux

  if ( write_ps ) then
    call lplot6(1,1,tht_tmp,psi_tmp,n_bnd,'psi at boundary')
    call lincol(1)
    call lplot6(1,1,tht_tmp,psi_boundary,-n_bnd,'psi at boundary')
    call lincol(0)
  endif

  call tr_deallocate(work,"work",CAT_GRID)

  mf = 256

  do j=1, mf

    theta = 2.d0 * PI * float(j-1)/float(mf)

    if (theta .lt. tht_tmp(1))     theta = theta + 2.d0 *PI
    if (theta .gt. tht_tmp(n_bnd)) theta = theta - 2.d0 *PI

    call  TG02A(-1,n_bnd,tht_tmp,r_tmp,dr_tmp,theta,Vr)
    call  TG02A(-1,n_bnd,tht_tmp,psi_tmp,dpsi_tmp,theta,Vpsi)

    fbnd(j) = Vr(1)
    fpsi(j) = Vpsi(1)

  enddo

  call rft2(fbnd,mf,1)
  call rft2(fpsi,mf,1)

  do m=1,mf
    fbnd(m) = 2.d0 * fbnd(m) / float(mf)
    fpsi(m) = 2.d0 * fpsi(m) / float(mf)
  enddo
  do m=2,mf,2
    fbnd(m) = - fbnd(m)
    fpsi(m) = - fpsi(m)
  enddo

  fpsi(1) = 0.d0

  call tr_deallocate(tht_tmp,"tht_tmp",CAT_GRID)
  call tr_deallocate(r_tmp,"r_tmp",CAT_GRID)
  call tr_deallocate(dr_tmp,"dr_tmp",CAT_GRID)
  call tr_deallocate(psi_tmp,"psi_tmp",CAT_GRID)
  call tr_deallocate(dpsi_tmp,"dpsi_tmp",CAT_GRID)

else
  write(*,*) ' boundary defined by Fourier series : ',mf
endif


return

end
