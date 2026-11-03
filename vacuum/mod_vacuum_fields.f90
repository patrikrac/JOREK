!< Calculates fields created by wall and PF coil currents at arbitrary points
!< The points can be inside the plasma or in the vacuum, any xyz point but
!< DO NOT calculate the fields exactly at the STARWALL's coil/wall triangles
!< Otherwise singularities at those points may occur!
!< Additionally, wall forces are computed through the integral of the stress tensor
module mod_vacuum_fields

  use vacuum
  use vacuum_response, only: reconstruct_triangle_potentials, reconstruct_coil_potentials

  implicit none

  ! --- Scaled STARWALL wall triangle coordinates (n_tri, 3)
  real*8, allocatable :: xw_scaled(:,:), yw_scaled(:,:), zw_scaled(:,:)

   
  contains






  !< This routine calculates the total wall forces by integrating the force tensor on a closed surface
  !! outside the wall (doi:10.1088/1741-4326/aa8876)
  subroutine total_wall_forces(my_id, node_list, element_list, scale_fact, Fx, Fy, Fz, n_phi_int)

    use constants
    use data_structure
    use mod_plasma_response,    only: plasma_fields_at_xyz    
    use phys_module,            only: F0

    implicit none

    ! --- External parameters
    integer,                   intent(in)    :: my_id
    type (type_node_list),     intent(in)    :: node_list
    type (type_element_list),  intent(in)    :: element_list   
    real*8,                    intent(in)    :: scale_fact
    real*8,                    intent(inout) :: Fx, Fy, Fz  ! --- The total force in SI units (cartesian components)
    integer,                   intent(in)    :: n_phi_int  ! --- Number of points for integration in the phi direction

    ! --- Local parameters
    real*8               :: bx, by, bz
    real*8, allocatable  :: bx_c(:), by_c(:), bz_c(:), psi_c(:)
    real*8, allocatable  :: bx_w(:), by_w(:), bz_w(:), psi_w(:)
    real*8, allocatable  :: bx_p(:), by_p(:), bz_p(:), psi_p(:)
    real*8, allocatable  ::  x_w(:),  y_w(:),  z_w(:)
    real*8               :: tri_area, B2, Bn, r1(3), r2(3), r3(3), r21(3), r32(3) 
    real*8               :: nx, ny, nz, r21_cross_r32(3), x_mid, y_mid, R_mid
    real*8               :: Bphi_x, Bphi_y
    integer              :: i

    ! --- Create a surface just outside the wall (made of triangles)
    call resize_starwall_wall(scale_fact)

    ! --- Obtain xyz points at the center of the triangles
    allocate( x_w(sr%ntri_w), y_w(sr%ntri_w), z_w(sr%ntri_w) )
    do i=1, sr%ntri_w
      x_w(i) = sum( xw_scaled(i,:) ) / 3.d0
      y_w(i) = sum( yw_scaled(i,:) ) / 3.d0
      z_w(i) = sum( zw_scaled(i,:) ) / 3.d0
    end do 

    ! --- Calculate the total magnetic field at the triangle centers
    allocate(bx_c(sr%ntri_w), by_c(sr%ntri_w), bz_c(sr%ntri_w), psi_c(sr%ntri_w))
    allocate(bx_w(sr%ntri_w), by_w(sr%ntri_w), bz_w(sr%ntri_w), psi_w(sr%ntri_w))
    allocate(bx_p(sr%ntri_w), by_p(sr%ntri_w), bz_p(sr%ntri_w), psi_p(sr%ntri_w))

    call coil_fields_at_xyz(my_id, x_w, y_w, z_w, bx_c, by_c, bz_c, psi_c)
    call wall_fields_at_xyz(my_id, x_w, y_w, z_w, bx_w, by_w, bz_w, psi_w)
    call plasma_fields_at_xyz(my_id, node_list,element_list, x_w, y_w, z_w, &
                              bx_p, by_p, bz_p, psi_p, n_phi_int)

    ! ---- Do integral of the force tensor over the triangle discretized surface
    Fx = 0.d0;    Fy = 0.d0;   Fz = 0.d0

    do i=1, sr%ntri_w

      r1(:)  = (/ xw_scaled(i,1), yw_scaled(i,1), zw_scaled(i,1) /)
      r2(:)  = (/ xw_scaled(i,2), yw_scaled(i,2), zw_scaled(i,2) /)
      r3(:)  = (/ xw_scaled(i,3), yw_scaled(i,3), zw_scaled(i,3) /)

      r21(:) = r1(:)-r2(:)
      r32(:) = r2(:)-r3(:)

      r21_cross_r32(:) = (/ r21(2)*r32(3) - r21(3)*r32(2), r21(3)*r32(1) - r21(1)*r32(3),          &
        r21(1)*r32(2) - r21(2)*r32(1) /)

      tri_area = sqrt(sum(r21_cross_r32**2.d0)) / 2.d0     

      nx = r21_cross_r32(1) / sqrt(sum(r21_cross_r32**2.d0)) 
      ny = r21_cross_r32(2) / sqrt(sum(r21_cross_r32**2.d0)) 
      nz = r21_cross_r32(3) / sqrt(sum(r21_cross_r32**2.d0)) 

      x_mid  =  sum( xw_scaled(i,:) ) / 3.d0
      y_mid  =  sum( yw_scaled(i,:) ) / 3.d0
      R_mid  = sqrt( x_mid**2.d0 + y_mid**2.d0 )

      Bphi_x = F0/R_mid * (  y_mid/R_mid ) ! Bphi * (-sin phi) 
      Bphi_y = F0/R_mid * ( -x_mid/R_mid ) ! Bphi * (-cos phi)

      bx = bx_p(i) + bx_c(i) + bx_w(i) + Bphi_x
      by = by_p(i) + by_c(i) + by_w(i) + Bphi_y
      bz = bz_p(i) + bz_c(i) + bz_w(i)

      Bn = bx*nx + by*ny + bz*nz
      B2 = bx**2.d0 + by**2.d0 + bz**2.d0 

      Fx = Fx + (Bn*bx - B2*0.5d0*nx) * tri_area / mu_zero
      Fy = Fy + (Bn*by - B2*0.5d0*ny) * tri_area / mu_zero
      Fz = Fz + (Bn*bz - B2*0.5d0*nz) * tri_area / mu_zero

    end do 

    ! --- Clean-up
    deallocate(bx_c, by_c, bz_c)
    deallocate(bx_w, by_w, bz_w)
    deallocate(bx_p, by_p, bz_p)
    deallocate( x_w,  y_w,  z_w)  
    deallocate(xw_scaled, yw_scaled, zw_scaled)

  end subroutine total_wall_forces





 
  !< This routine calculates the fields created by the STARWALL wall at given cartesian coordinates
  !< DO NOT calculate the fields exactly at the STARWALL's coil/wall triangles
  !< Otherwise singularities at those points may occur!
  subroutine wall_fields_at_xyz(my_id,x,y,z,bx,by,bz,psi)

    use constants

    implicit none

    ! --- External parameters
    integer, intent(in)     :: my_id
    real*8,  intent(in)     :: x(:), y(:), z(:)     ! Points where fields are calculated
    real*8,  intent(inout)  :: bx(:), by(:), bz(:), psi(:)

    ! --- Local parameters
    real*8,  allocatable    :: tripot_w(:)
    real*8,  allocatable    :: phi_w(:,:), x_w(:,:), y_w(:,:), z_w(:,:)
    real*8                  :: Iw_net_tor
    integer                 :: i, j, ipot
     
    call reconstruct_triangle_potentials(tripot_w, wall_curr, my_id, Iw_net_tor)

    if (sr%file_version < 5) then
      write(*,*) 'wall_fields_at_xyz not available for STARWALL reponse file version < 5'
      STOP
    endif

    allocate(phi_w(sr%ntri_w,3))
    allocate(x_w(sr%ntri_w,3), y_w(sr%ntri_w,3), z_w(sr%ntri_w,3))

    ! --- Total wall triangle potentials (including net currents)
    do i = 1, sr%ntri_w
      do j = 1, 3
        ipot       = sr%jpot_w(i,j)
        phi_w(i,j) = tripot_w(ipot) + Iw_net_tor*sr%phi0_w(i,j) 
        x_w(i,j)   = sr%xyzpot_w(ipot,1)
        y_w(i,j)   = sr%xyzpot_w(ipot,2)
        z_w(i,j)   = sr%xyzpot_w(ipot,3)
      end do
    end do

    phi_w   = phi_w / mu_zero  ! Wall current potentials in Amperes

    call triang_fields_at_xyz(my_id,x,y,z,x_w,y_w,z_w,phi_w,bx,by,bz,psi)

    deallocate(phi_w, x_w, y_w, z_w)

  end subroutine wall_fields_at_xyz





  !< This routine calculates the fields created by the STARWALL coils at given cartesian coordinates
  !< DO NOT calculate the fields exactly at the STARWALL's coil/wall triangles
  !< Otherwise singularities at those points may occur!
  subroutine coil_fields_at_xyz(my_id,x,y,z,bx,by,bz,psi,icoil)

    use constants

    implicit none

    ! --- External parameters
    integer,           intent(in)     :: my_id
    real*8,            intent(in)     :: x(:), y(:), z(:)     ! Points where fields are calculated
    real*8,            intent(inout)  :: bx(:), by(:), bz(:), psi(:)
    integer, optional, intent(in)     :: icoil ! If present, gets fields only from coil number "icoil"

    ! --- Local parameters
    real*8,  allocatable    :: pot_c(:)
    real*8,  allocatable    :: phi_c(:,:)
    integer                 :: i, j, ipot, ntri_c, i_c
    integer                 :: i_tri_start, i_tri_end

    if (sr%ncoil < 1) then
      write(*,*) 'coil_fields_at_xyz needs STARWALL coils (ncoils < 1) detected'
      return
    endif
    
    call reconstruct_coil_potentials(pot_c, wall_curr, my_id)

    allocate(phi_c(sr%ntri_c,3))

    ! --- Distribute coil currents over the coil triangles
    i_tri_start = 1

    do i_c=1, sr%ncoil

      i_tri_end = i_tri_start + sr%jtri_c(i_c) - 1

      ! --- When icoil is given, set the other coil currents to 0
      if (present(icoil)) then
        if (i_c /= icoil) pot_c(i_c) = 0.d0
      endif

      do i = i_tri_start, i_tri_end
        do j = 1, 3
          phi_c(i,j) = sr%phi_coil(i,j) * pot_c(i_c)
        end do
      end do

      i_tri_start = i_tri_end + 1

    enddo

    phi_c   = phi_c / mu_zero  ! Wall current potentials in Amperes

    call triang_fields_at_xyz(my_id,x,y,z,sr%x_coil,sr%y_coil,sr%z_coil,phi_c,bx,by,bz,psi)

    deallocate(phi_c)

  end subroutine coil_fields_at_xyz






  !< This routine calculates the fields produced by currents flowing on a set of triangles
  !! at given xyz points (taken from STARWALL)
  subroutine triang_fields_at_xyz(my_id,x,y,z,x_tri,y_tri,z_tri,phi_tri,bx,by,bz,psi)

    use constants
    use mpi_mod
    !$ use omp_lib

    implicit none

    ! --- External parameters
    integer, intent(in)     :: my_id
    real*8,  intent(in)     :: x(:), y(:), z(:) ! Points where fields are calculated
    real*8,  intent(in)     :: x_tri(:,:), y_tri(:,:), z_tri(:,:), phi_tri(:,:)
    real*8,  intent(inout)  :: bx(:), by(:), bz(:), psi(:)

    ! --- Local parameters
    integer :: ierr, n_mpi, k_delta, k_min, k_max
    integer :: i, j, k, np, ntri, omp_nthreads, omp_tid
    real*8  :: s1,s2,s3                                    &
              ,d221,d232,d213,al1,al2,al3                  &
              ,ata1,ata2,ata3,at                           &
              ,s21,s22,s23,dp1,dm1,dp2,dm2,dp3,dm3         &
              ,ap1,am1,ap2,am2,ap3,am3                     &
              ,h,ar1,ar2,ar3                               &
              ,x21,y21,z21,x32,y32,z32,x13,y13,z13,vx,vy,vz&
              ,tx1,ty1,tz1,tx2,ty2,tz2,tx3,ty3,tz3         &
              ,nx,ny,nz,pi41,area,d21,d32,d13,jx,jy,jz     &
              ,dep1,dep2,dep3,dem1,dem2,dem3
    real*8  :: Rcent, jphi, cosx, siny, green
    real*8  :: x1,y1,z1,x2,y2,z2,x3,y3,z3,sn
    real*8,  allocatable  :: bx_tmp(:), by_tmp(:), bz_tmp(:), Ax_tmp(:), Ay_tmp(:),  Ax(:), Ay(:)

    np   = size(x,1)
    ntri = size(x_tri,1)

    pi41 = 0.125d0/asin(1.d0)

    allocate(bx_tmp(np), by_tmp(np), bz_tmp(np), Ax_tmp(np), Ay_tmp(np))

    bx     = 0.d0;  by     = 0.d0;  bz     = 0.d0;  psi    = 0.d0;
    bx_tmp = 0.d0;  by_tmp = 0.d0;  bz_tmp = 0.d0;
    Ax_tmp = 0.d0;  Ay_tmp = 0.d0;  

    ! --- MPI initialization
    call MPI_COMM_SIZE(MPI_COMM_WORLD, n_mpi, ierr) ! number of MPI procs
    n_mpi = max(n_mpi,1)

    k_delta = ceiling(float(ntri) / n_mpi)
    k_min   =      my_id     * k_delta + 1
    k_max   = min((my_id +1) * k_delta, ntri)
    ! --- OpenMP parallelization of given points loop
    !$omp parallel default(none)                                                            &
    !$omp   shared(np,x_tri,y_tri,z_tri,x,y,z, k_min, k_max,pi41,phi_tri,                   &
    !$omp          bx_tmp, by_tmp, bz_tmp, Ax_tmp, Ay_tmp)                                  &
    !$omp   private(i,x1,y1,z1,x2,y2,z2,x3,y3,z3,sn,h,s21,s22,s23,s1,s2,s3,al1,al2,al3,     &
    !$omp           ar1,ar2,ar3,dp1,dp2,dp3,dm1,dm2,dm3,ap1,ap2,ap3,dep1,dep2,dep3,         &
    !$omp           d21,d32,d13,tx2,ty2,tz2,tx3,ty3,tz3,d221, d232,d213,area,Rcent,cosx,    &
    !$omp           nx,ny,nz, jx,jy,jz, tx1,ty1,tz1,k,x21,y21,z21,x32,y32,z32,x13,y13,z13,  &
    !$omp           dem1,dem2,dem3,am1,am2,am3,ata1,ata2,ata3,at,vx,vy,vz,siny,jphi,        &
    !$omp           omp_nthreads,omp_tid, green)
    
#ifdef _OPENMP
    omp_nthreads = omp_get_num_threads()
    omp_tid      = omp_get_thread_num()
#else
    omp_nthreads = 1
    omp_tid      = 0
#endif

    !$omp do reduction(+:bx_tmp, by_tmp, bz_tmp, Ax_tmp, Ay_tmp)     
 
    do k=k_min, k_max    ! --- integral over wall triangles

     !--- only use toroidal current, projection
      x21   = x_tri(k,2) - x_tri(k,1)
      y21   = y_tri(k,2) - y_tri(k,1)
      z21   = z_tri(k,2) - z_tri(k,1)
      x32   = x_tri(k,3) - x_tri(k,2)
      y32   = y_tri(k,3) - y_tri(k,2)
      z32   = z_tri(k,3) - z_tri(k,2)
      x13   = x_tri(k,1) - x_tri(k,3)
      y13   = y_tri(k,1) - y_tri(k,3)
      z13   = z_tri(k,1) - z_tri(k,3)
      d221  = x21**2+y21**2+z21**2
      d232  = x32**2+y32**2+z32**2
      d213  = x13**2+y13**2+z13**2
      d21   = sqrt(d221)
      d32   = sqrt(d232)
      d13   = sqrt(d213)
      nx    = -y21*z13 + z21*y13
      ny    = -z21*x13 + x21*z13
      nz    = -x21*y13 + y21*x13
      area  = 1./sqrt(nx*nx+ny*ny+nz*nz)

      ! --- The triangle current density
      jx = (x32*phi_tri(k,1)+x13*phi_tri(k,2)+x21*phi_tri(k,3))*area*pi41 
      jy = (y32*phi_tri(k,1)+y13*phi_tri(k,2)+y21*phi_tri(k,3))*area*pi41 
      jz = (z32*phi_tri(k,1)+z13*phi_tri(k,2)+z21*phi_tri(k,3))*area*pi41 

      ! --- Only use toroidal current (RMHD)
      Rcent = sqrt( (sum(x_tri(k,:))/3.d0)**2.d0 + (sum(y_tri(k,:))/3.d0)**2.d0 )

      cosx  =  sum(x_tri(k,:))/(3.d0*Rcent)
      siny  = -sum(y_tri(k,:))/(3.d0*Rcent)

      jphi  = -jx*siny - jy*cosx

      jx    = -jphi * siny
      jy    = -jphi * cosx
      jz    = 0.d0

      nx    = nx*area
      ny    = ny*area
      nz    = nz*area
      tx1   = (y32*nz-z32*ny)
      ty1   = (z32*nx-x32*nz)
      tz1   = (x32*ny-y32*nx)
      tx2   = (y13*nz-z13*ny)
      ty2   = (z13*nx-x13*nz)
      tz2   = (x13*ny-y13*nx)
      tx3   = (y21*nz-z21*ny)
      ty3   = (z21*nx-x21*nz)
      tz3   = (x21*ny-y21*nx)


      do i=1, np      ! --- go over given points

        x1    = x_tri(k,1) - x(i)
        y1    = y_tri(k,1) - y(i)
        z1    = z_tri(k,1) - z(i)
        x2    = x_tri(k,2) - x(i)
        y2    = y_tri(k,2) - y(i)
        z2    = z_tri(k,2) - z(i)
        x3    = x_tri(k,3) - x(i)
        y3    = y_tri(k,3) - y(i)
        z3    = z_tri(k,3) - z(i)
        sn    = nx*x1+ny*y1+nz*z1
        h     = abs(sn)
        s21   = x1**2+y1**2+z1**2
        s22   = x2**2+y2**2+z2**2
        s23   = x3**2+y3**2+z3**2
        s1    = sqrt(s21)
        s2    = sqrt(s22)
        s3    = sqrt(s23)
        al1   = alog((s2+s1+d21)/(s1+s2-d21))
        al2   = alog((s3+s2+d32)/(s3+s2-d32))
        al3   = alog((s1+s3+d13)/(s1+s3-d13))
        ar1   = x1*tx3+y1*ty3+z1*tz3
        ar2   = x2*tx1+y2*ty1+z2*tz1
        ar3   = x3*tx2+y3*ty2+z3*tz2
        dp1   = .5*(s22-s21+d221)
        dp2   = .5*(s23-s22+d232)
        dp3   = .5*(s21-s23+d213)
        dm1   = dp1-d221
        dm2   = dp2-d232
        dm3   = dp3-d213
        ap1   = ar1*dp1
        dep1  = ar1**2+h*d221*(h+s2)
        ap2   = ar2*dp2
        dep2  = ar2**2+h*d232*(h+s3)
        ap3   = ar3*dp3
        dep3  = ar3**2+h*d213*(h+s1)
        am1   = ar1*dm1
        dem1  = ar1**2+h*d221*(h+s1)
        am2   = ar2*dm2
        dem2  = ar2**2+h*d232*(h+s2)
        am3   = ar3*dm3
        dem3  = ar3**2+h*d213*(h+s3)
        ata1  = atan2(ap1*dem1-am1*dep1,dep1*dem1+ap1*am1)
        ata2  = atan2(ap2*dem2-am2*dep2,dep2*dem2+ap2*am2)
        ata3  = atan2(ap3*dem3-am3*dep3,dep3*dem3+ap3*am3)
        at    = sign(1.,sn)*(ata1+ata2+ata3)
        vx    = -nx*at + al1*tx3/d21+al2*tx1/d32+al3*tx2/d13
        vy    = -ny*at + al1*ty3/d21+al2*ty1/d32+al3*ty2/d13
        vz    = -nz*at + al1*tz3/d21+al2*tz1/d32+al3*tz2/d13
        green = -h*(ata1+ata2+ata3) + ar1*al1/d21+ar2*al2/d32+ar3*al3/d13

        bx_tmp(i) = bx_tmp(i) + vy*jz-vz*jy
        by_tmp(i) = by_tmp(i) + vz*jx-vx*jz
        bz_tmp(i) = bz_tmp(i) + vx*jy-vy*jx

        Ax_tmp(i) = Ax_tmp(i) + green*jx
        Ay_tmp(i) = Ay_tmp(i) + green*jy
      enddo
    enddo
    !$omp end do
    !$omp end parallel

    call MPI_AllReduce(bx_tmp, bx, np,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    call MPI_AllReduce(by_tmp, by, np,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    call MPI_AllReduce(bz_tmp, bz, np,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)

    deallocate(bx_tmp, by_tmp, bz_tmp) 
    allocate(Ax(np), Ay(np))
    Ax = 0.d0;   Ay = 0.d0;

    call MPI_AllReduce(Ax_tmp, Ax, np,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    call MPI_AllReduce(Ay_tmp, Ay, np,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    deallocate(Ax_tmp, Ay_tmp) 

    !--- Convert Ax and Ay to psi
    do i=1, np
      ! --- psi  = A . ephi * R
      ! --- ephi = (-sin, -cos),  x = Rcos, y = -R sin, thus ephi = (y,-x)/R
      psi(i) = Ax(i)*y(i) - Ay(i)*x(i)
    enddo ! --- evaluation points
    deallocate(Ax, Ay)

    bx  = -bx * mu_zero
    by  = -by * mu_zero
    bz  = -bz * mu_zero
    psi = psi * mu_zero

  end subroutine triang_fields_at_xyz






  !< This routine re-calculates the STARWALL wall from wall Fourier harmonics and
  !! resizes it acording to a scale factor. This is useful to create a surface
  !! just outside the wall to perform wall forces integration. Note that using the
  !! same wall triangles would lead to singularities for the wall field calculation
  subroutine resize_starwall_wall(scale_fact)

    implicit none

    ! --- External parameters
    real*8, intent(in) :: scale_fact

    ! --- Local parameters
    real*8              :: pi2,fnu,alu,alv
    real*8              :: cm,cn,cov,siv,cou,siu,cop,sip,co,si
    real*8, allocatable ::  r_w(:),  x_w(:),  y_w(:),  z_w(:)
    real*8, allocatable :: rc_w(:), rs_w(:), zc_w(:), zs_w(:)
    integer             :: nwuv, i, j, kv, ku

    if (sr%file_version < 5) then
      write(*,*) 'STARWALL version not suitable for wall resizing'
      STOP
    endif
    if (sr%iwall /= 1) then
      write(*,*) 'Wall must be given in Fourier harmonics for resizing (iwall=1)'
      STOP
    endif

    nwuv  = sr%nwu*sr%nwv ! total number of wall nodes

    allocate (x_w(nwuv),y_w(nwuv),z_w(nwuv),r_w(nwuv))
    allocate (rc_w(sr%mn_w),rs_w(sr%mn_w),zc_w(sr%mn_w),zs_w(sr%mn_w))
    if (allocated(xw_scaled)) deallocate(xw_scaled)
    if (allocated(yw_scaled)) deallocate(yw_scaled)
    if (allocated(zw_scaled)) deallocate(zw_scaled)
    allocate (xw_scaled(sr%ntri_w,3),yw_scaled(sr%ntri_w,3),zw_scaled(sr%ntri_w,3))

    ! --- Resize the wall, wall construction copied from STARWALL
    pi2  =4.*asin(1.)
    fnu  = 1./float(sr%nwu) 
    alu  = pi2*fnu 
    alv  = pi2/float(sr%nwv)
    z_w = 0.
    r_w = 0.

    rc_w(1) =sr%rc_w(1)
    rs_w(1) =sr%rs_w(1)
    zc_w(1) =sr%zc_w(1)
    zs_w(1) =sr%zs_w(1)

    rc_w(2:sr%mn_w) =sr%rc_w(2:sr%mn_w)*scale_fact
    rs_w(2:sr%mn_w) =sr%rs_w(2:sr%mn_w)*scale_fact
    zc_w(2:sr%mn_w) =sr%zc_w(2:sr%mn_w)*scale_fact
    zs_w(2:sr%mn_w) =sr%zs_w(2:sr%mn_w)*scale_fact

    do  j =  1, sr%mn_w
      cm = sr%m_w(j)*pi2
      cn = sr%n_w_fourier(j)*pi2
      do  kv=1, sr%nwv
        cov = cos(alv*sr%n_w_fourier(j)*(kv-1))
        siv = sin(alv*sr%n_w_fourier(j)*(kv-1))
        do  ku=1,sr%nwu
          i = ku+sr%nwu*(kv-1)
          cou = cos(alu*sr%m_w(j)*(ku-1))
          siu = sin(alu*sr%m_w(j)*(ku-1))
    
          cop = cou*cov-siu*siv
          sip = siu*cov+cou*siv
    
          r_w(i) = r_w(i) + rs_w(j)*sip + rc_w(j)*cop
          z_w(i) = z_w(i) + zs_w(j)*sip + zc_w(j)*cop 
        end do
      end do
    end do
    
    do  kv = 1, sr%nwv
      co   = cos(alv*(kv-1)) 
      si   = sin(alv*(kv-1)) 
      do  ku = 1,sr%nwu
        i      = sr%nwu*(kv-1)+ku
        x_w(i)   =   co * r_w(i)
        y_w(i)   =   si * r_w(i)
      end do
    end do

    ! --- Reconstruct scaled wall triangle coordinates
    do i=1,2*nwuv
      xw_scaled(i,1) = x_w(sr%jpot_w(i,1))
      xw_scaled(i,2) = x_w(sr%jpot_w(i,2))
      xw_scaled(i,3) = x_w(sr%jpot_w(i,3))

      yw_scaled(i,1) = y_w(sr%jpot_w(i,1))
      yw_scaled(i,2) = y_w(sr%jpot_w(i,2))
      yw_scaled(i,3) = y_w(sr%jpot_w(i,3))

      zw_scaled(i,1) = z_w(sr%jpot_w(i,1))
      zw_scaled(i,2) = z_w(sr%jpot_w(i,2))
      zw_scaled(i,3) = z_w(sr%jpot_w(i,3))
    end do

    deallocate( x_w,  y_w,  z_w,  r_w)
    deallocate(rc_w, rs_w, zc_w, zs_w)

  end subroutine resize_starwall_wall






  !> Routine to calculate B_plasma and psi_plasma at xyz points outside the plasma domain
  !> It uses only Bnorm and Btan at the boundary of the domain, using formula (19) from
  !> JD Hanson 2015 Plasma Phys. Control. Fusion 57 115006. The formula for A_plasma
  !> comes from Stratton, Electromagneti Theory, page 253. 
  !> IMPORTANT!: The formula for A_plasma is gauge dependent (Coulomb gauge only) and cannot
  !> be used to calculate \psi for n_tor>1!!
  !> The JOREK boundary is discretized into triangles to use semi-analytical formulas for
  !> the Green's functions taken from STARWALL
  subroutine vacuum_plasma_fields(node_list, element_list, bnd_node_list, bnd_elm_list, &  
                                  n_phi_integral, r_pts, B_plasma, psi_plasma, print_params )
    !$ use omp_lib
    use mod_boundary,   only: get_st_on_bnd
    use data_structure, only: type_node_list, type_element_list, type_bnd_element_list, &
                              type_bnd_node_list 
    use mod_parameters 
    use constants
    use mod_interp
    use mod_basisfunctions
    use mod_plasma_response, only: Greens_functions

    implicit none

    type(type_node_list),        intent(in) :: node_list
    type(type_element_list),     intent(in) :: element_list
    type(type_bnd_node_list),    intent(in) :: bnd_node_list
    type(type_bnd_element_list), intent(in) :: bnd_elm_list

    integer,intent(in) :: n_phi_integral  !< number of toroidal planes for surface integral
    real*8, intent(inout), allocatable :: r_pts(:,:)      !< points where field is given (point_index, (x,y,z))
    real*8, intent(inout), allocatable, dimension(:,:) :: B_plasma
    real*8, intent(inout), allocatable, dimension(:)   :: psi_plasma
    logical, optional, intent(in) :: print_params
  
    ! --- Local
    integer :: i, j, k, ms, mt, mp, in, ip, i_tri, jp, i_tor, kp
    integer :: i_bnd, i_bas, i_vertex, iv, i_start, ind, i_cs, i_dof, ierr
    integer :: n_bnd_elm, n_eval, side, i_elm, mode_number(n_tor), omp_nthreads, omp_tid
    integer, parameter :: nsub=10, n_tri=2
    logical :: s_const
    real*8, dimension(3)     :: dA, dB, A_cart, Atan_cart, Btan_cart, B_cart, green_B, &
                                R_tria, Z_tria, s_tria, norm_tri_av, norm_2
    real*8, dimension(n_tor) :: H_tor, Psi_pol, Psi_R_pol, Psi_Z_pol
    real*8  :: delta_phi, phi, RR, jac1D, Bnorm, Btan
    real*8  :: R_1, Z_1, R_2, Z_2, Rs, Zs, s_2d, t_2d, R_in, Z_in
    real*8  :: x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4, area1,area2
    real*8  :: G_BR, G_BZ, G_psi, Green_A
    real*8  :: B_pol(2), bnd_normal(2), vec_out(2)
    real*8  :: H(2,2), H_s(2,2),H_ss(2,2)
    real*8  :: xe,ye,ze, x_mid, y_mid, z_mid, L_min_analytic, dd, l_char
    real*8  :: R, R_s, R_t, Z, Z_s, Z_t
    real*8  :: P, P_s, P_t, P_st, P_ss, P_tt, Psi, Psi_R, Psi_Z, sz, xjac
    real*8, allocatable :: x_tri(:,:), y_tri(:,:), z_tri(:,:)
    real*8, allocatable :: green_A_tri(:), green_B_tri(:,:), A_plasma(:,:)
    real*8, allocatable :: bx_tmp(:), by_tmp(:), bz_tmp(:)
    real*8, allocatable :: Ax_tmp(:), Ay_tmp(:)

    delta_phi  = 2.d0*PI / float(n_phi_integral) 

    do i_tor=1, n_tor
      mode_number(i_tor) = + int(i_tor / 2) * n_period
    enddo

    ! --- Important integers
    n_bnd_elm = bnd_elm_list%n_bnd_elements
    n_eval    = size(r_pts,1)

    if ( size(r_pts,2) /= 3) then
      write(*,*) ' ERROR: Second dimension of r_pts should be 3 (for x,y,z components)'
      stop
    endif

    if (allocated(B_plasma))   deallocate(B_plasma)
    if (allocated(psi_plasma)) deallocate(psi_plasma)

    if (present(print_params)) then
      if (print_params) then
        write(*,*) ' '
        write(*,*) ' **************************************************************************'
        write(*,*) ' *** Calculating vacuum  plasma fields  ***********************************'
        write(*,*) ' **************************************************************************'
        write(*,*) ' '

        write(*,*) ' n_bnd_elm     = ', n_bnd_elm
        write(*,*) ' n_phi_integral   = ', n_phi_integral
        write(*,*) ' nsub      = ', nsub
        write(*,*) ' n_eval    = ', n_eval
      endif
    endif

    allocate(x_tri(n_tri,3), y_tri(n_tri,3), z_tri(n_tri,3), green_A_tri(n_tri), green_B_tri(n_tri,3) ) 
    allocate(bx_tmp(n_eval), by_tmp(n_eval), bz_tmp(n_eval))
    allocate(Ax_tmp(n_eval), Ay_tmp(n_eval))

    bx_tmp = 0.d0;  by_tmp = 0.d0;  bz_tmp = 0.d0;  
    Ax_tmp = 0.d0;  Ay_tmp = 0.d0;  
 
    !--- go through the boundary elements
    !$omp parallel default(private)                                            &
    !$omp   shared( r_pts, delta_phi, n_phi_integral,  &
    !$omp   n_eval, n_bnd_elm, element_list, node_list, bnd_elm_list, &
    !$omp   bx_tmp, by_tmp, bz_tmp, Ax_tmp, Ay_tmp, mode_number ) &
    !$omp   firstprivate(green_A_tri, green_B_tri, x_tri, y_tri, z_tri)   

    !$omp do reduction(+:bx_tmp, by_tmp, bz_tmp,Ax_tmp, Ay_tmp)  
    do i_bnd = 1, n_bnd_elm
    
      side   = bnd_elm_list%bnd_element(i_bnd)%side
      i_elm  = bnd_elm_list%bnd_element(i_bnd)%element

      !--- Poloidal subdivision of the element
      do ms=1, nsub

        s_tria(1) = float(ms-1)/float(nsub)           ! 1st triangle node
        s_tria(3) = float(ms  )/float(nsub)           ! 2nd triangle node

        s_tria(2) = ( s_tria(1) + s_tria(3) ) * 0.5d0  ! Middle point of the triangle
        
        R_tria(:) = 0.d0;  Z_tria(:) = 0.d0; ! Poloidal positions of node triangles and middle point

        do i=1, 3  ! Three poloidal positions relavant to the triangle (2 nodes and 1 middle point)      
          call  basisfunctions1(s_tria(i), H, H_s, H_ss)

          do i_vertex = 1,2
            iv = bnd_elm_list%bnd_element(i_bnd)%vertex(i_vertex)
      
            do i_bas = 1,2
              sz = bnd_elm_list%bnd_element(i_bnd)%size(i_vertex,i_bas)

              R_tria(i)  = R_tria(i)  + node_list%node(iv)%x(1,i_bas,1) * sz * H(i_vertex,i_bas)
              Z_tria(i)  = Z_tria(i)  + node_list%node(iv)%x(1,i_bas,2) * sz * H(i_vertex,i_bas)
            end do
          end do
        enddo
        
        call get_st_on_bnd(s_tria(2), side, s_2d, t_2d, s_const)
        call interp_RZ(node_list, element_list, i_elm, s_2d, t_2d, R, R_s, R_t, Z, Z_s, Z_t)
        xjac = R_s * Z_t - R_t * Z_s ! --- 2D Jacobian

        ! --- Psi value (plus derivatives) at poloidal points
        do i_tor = 1, n_tor
          call interp(node_list, element_list, i_elm, var_psi, i_tor, s_2d, t_2d, Psi_pol(i_tor), P_s, P_t, P_st, P_ss, P_tt)
      
          Psi_R_pol(i_tor) = (   P_s * Z_t - P_t * Z_s ) / xjac  ! dPsi/dR
          Psi_Z_pol(i_tor) = ( - P_s * R_t + P_t * R_s ) / xjac  ! dPsi/dZ
        end do
      
        !--- Subdivide the element into triangles (centred at phi + delta_phi/2)
        do mp=1, n_phi_integral

          phi = float(mp-1) * delta_phi + delta_phi*0.5d0

          ! --- Toroidal basis functions
          H_tor(1)   = 1.d0
          do i_tor=1,(n_tor-1)/2
            H_tor(2*i_tor)   = cos(mode_number(2*i_tor)   * phi )
            H_tor(2*i_tor+1) = sin(mode_number(2*i_tor+1) * phi )
          enddo
  
          !--- Points of rectangular element used to define 2 triangles
          x1 = R_tria(1)*cos(phi-delta_phi*0.5d0);    y1 = -R_tria(1)*sin(phi-delta_phi*0.5d0);    z1 = Z_tria(1);
          x2 = R_tria(3)*cos(phi-delta_phi*0.5d0);    y2 = -R_tria(3)*sin(phi-delta_phi*0.5d0);    z2 = Z_tria(3);
          x3 = R_tria(3)*cos(phi+delta_phi*0.5d0);    y3 = -R_tria(3)*sin(phi+delta_phi*0.5d0);    z3 = Z_tria(3);
          x4 = R_tria(1)*cos(phi+delta_phi*0.5d0);    y4 = -R_tria(1)*sin(phi+delta_phi*0.5d0);    z4 = Z_tria(1);

          !--- Fill up triangle positions
          x_tri(1,1) = x1;     y_tri(1,1) = y1;     z_tri(1,1) = z1;
          x_tri(1,2) = x2;     y_tri(1,2) = y2;     z_tri(1,2) = z2;
          x_tri(1,3) = x4;     y_tri(1,3) = y4;     z_tri(1,3) = z4;

          x_tri(2,1) = x2;     y_tri(2,1) = y2;     z_tri(2,1) = z2;
          x_tri(2,2) = x3;     y_tri(2,2) = y3;     z_tri(2,2) = z3;
          x_tri(2,3) = x4;     y_tri(2,3) = y4;     z_tri(2,3) = z4;

          ! --- Middle point of the rectangular element (field is evaluatead there)
          x_mid = (x1+x2+x3+x4) / 4.d0
          y_mid = (y1+y2+y3+y4) / 4.d0
          z_mid = (z1+z2+z3+z4) / 4.d0

          ! --- Area of each triangle and normals
          call get_tri_area_norm(x_tri(1,:), y_tri(1,:), z_tri(1,:), area1, norm_tri_av)
          call get_tri_area_norm(x_tri(2,:), y_tri(2,:), z_tri(2,:), area2, norm_2)
          norm_tri_av = (norm_tri_av + norm_2)*0.5d0  ! Averaged triangle normal

          l_char = sqrt( maxval( (/(x1-x2)**2+(y1-y2)**2+(z1-z2)**2, (x1-x3)**2+(y1-y3)**2+(z1-z3)**2 /) ) )

          L_min_analytic = l_char*3.d0 !--- Length below which semi-analytical formulas are used 

          ! --- Evaluate Atan, Btan and Bnorm
          Psi = 0.d0;   Psi_R = 0.d0;   Psi_Z = 0.d0
          do i_tor = 1, n_tor
            ! --- Poloidal magnetic field at current point
            Psi   = Psi   + Psi_pol(i_tor)   * H_tor(i_tor)
            Psi_R = Psi_R + Psi_R_pol(i_tor) * H_tor(i_tor) ! dPsi/dR
            Psi_Z = Psi_Z + Psi_Z_pol(i_tor) * H_tor(i_tor) ! dPsi/dZ
          end do

          B_pol = (/ Psi_Z, -Psi_R /) / R

          ! Analytical test
          ! call Greens_functions(R_tria(2), Z_tria(2), 6.d0, 0.d0, G_BR, G_BZ, G_psi)
          ! B_pol = (/ -G_BR, -G_BZ /) 
          ! Psi   = G_psi
          
          ! --- Get A and B in cartersian coordinates
          A_cart(1) = -Psi/R*sin(phi)
          A_cart(2) = -Psi/R*cos(phi)
          A_cart(3) = 0.d0

          B_cart(1) =  B_pol(1)*cos(phi)
          B_cart(2) = -B_pol(1)*sin(phi)
          B_cart(3) =  B_pol(2)

          ! --- Atan vector (x triangle_normal)
          Atan_cart(1) = A_cart(2)*norm_tri_av(3)-A_cart(3)*norm_tri_av(2)
          Atan_cart(2) = A_cart(3)*norm_tri_av(1)-A_cart(1)*norm_tri_av(3)
          Atan_cart(3) = A_cart(1)*norm_tri_av(2)-A_cart(2)*norm_tri_av(1)

          ! --- Btan vector (x triangle_normal)
          Btan_cart(1) = B_cart(2)*norm_tri_av(3)-B_cart(3)*norm_tri_av(2)
          Btan_cart(2) = B_cart(3)*norm_tri_av(1)-B_cart(1)*norm_tri_av(3)
          Btan_cart(3) = B_cart(1)*norm_tri_av(2)-B_cart(2)*norm_tri_av(1)

          ! --- Bnorm
          Bnorm =  B_cart(1)*norm_tri_av(1) + B_cart(2)*norm_tri_av(2) + B_cart(3)*norm_tri_av(3)
  
          ! --- Evaluation at given points
          ! --- Integrate for the provided points          
          do ip=1, n_eval
            xe = r_pts(ip,1);  ye = r_pts(ip,2);   ze = r_pts(ip,3);  ! points for evaluation
            dd = sqrt( (xe-x_mid)*(xe-x_mid) + (ye-y_mid)*(ye-y_mid) + (ze-z_mid)*(ze-z_mid) )
            
            ! --- Use semi-analytic formulas for the Green's functions when the evaluation point gets close to the triangle
            if (dd < L_min_analytic) then
              ! --- Calculates the triangle integrals of the B-field Green's functions ( \int a dS = \int grad G dS )
              call triang_Bfield_greens_integral(xe,ye,ze,x_tri,y_tri,z_tri,green_A_tri,green_B_tri) 
              green_A    = green_A_tri(1)   + green_A_tri(2)
              green_B(1) = green_B_tri(1,1) + green_B_tri(2,1)
              green_B(2) = green_B_tri(1,2) + green_B_tri(2,2)
              green_B(3) = green_B_tri(1,3) + green_B_tri(2,3)
            else
              green_A = 1.d0/dd * (area1+area2)

              green_B(1) = 1.d0/dd**3 * (xe-x_mid) * (area1+area2)
              green_B(2) = 1.d0/dd**3 * (ye-y_mid) * (area1+area2)
              green_B(3) = 1.d0/dd**3 * (ze-z_mid) * (area1+area2)
            endif

            dA(1) = -green_A*Btan_cart(1)  +  green_B(2)*Atan_cart(3)-green_B(3)*Atan_cart(2)
            dA(2) = -green_A*Btan_cart(2)  +  green_B(3)*Atan_cart(1)-green_B(1)*Atan_cart(3)
            dA(3) = -green_A*Btan_cart(3)  +  green_B(1)*Atan_cart(2)-green_B(2)*Atan_cart(1)
          
            dB(1) =  green_B(1)  * Bnorm   +  green_B(2)*Btan_cart(3)-green_B(3)*Btan_cart(2)
            dB(2) =  green_B(2)  * Bnorm   +  green_B(3)*Btan_cart(1)-green_B(1)*Btan_cart(3)
            dB(3) =  green_B(3)  * Bnorm   +  green_B(1)*Btan_cart(2)-green_B(2)*Btan_cart(1)

            ! --- Sum up field contributions
            Ax_tmp(ip)  = Ax_tmp(ip) + dA(1)
            Ay_tmp(ip)  = Ay_tmp(ip) + dA(2)
            !Az_tmp(ip)  = Az_tmp(ip) + dA(3)

            bx_tmp(ip)  = bx_tmp(ip) + dB(1)
            by_tmp(ip)  = by_tmp(ip) + dB(2)
            bz_tmp(ip)  = bz_tmp(ip) + dB(3)

          enddo ! --- evaluation points

        enddo  !--- poloidal subdivision of element (integration)w
      enddo  ! --- toroidal integration, n_planes (integration)
    
    enddo !--- bnd elements
    !$omp end do
    !$omp end parallel
    
    ! --- Convert A_plasma into psi_plasma
    allocate(psi_plasma(n_eval))
    do ip=1, n_eval
      ! --- psi  = A . ephi * R
      ! --- ephi = (-sin, -cos),  x = Rcos, y = -R sin, thus ephi = (y,-x)/R
      psi_plasma(ip) = Ax_tmp(ip)*r_pts(ip,2) - Ay_tmp(ip)*r_pts(ip,1)
    enddo ! --- evaluation points
    deallocate(Ax_tmp, Ay_tmp)

    allocate(B_plasma(n_eval,3))

    do ip=1, n_eval
      B_plasma(ip,1) = bx_tmp(ip)
      B_plasma(ip,2) = by_tmp(ip)
      B_plasma(ip,3) = bz_tmp(ip)
    enddo ! --- evaluation points

    deallocate(bx_tmp, by_tmp, bz_tmp) 

    B_plasma   = B_plasma   / (4.d0*PI)
    psi_plasma = psi_plasma / (4.d0*PI)

    ! --- Semi-Analytical test
    ! do ip=1, n_eval
    !   R = (r_pts(ip,1)**2.d0 + r_pts(ip,2)**2.d0)**0.5d0
    !   call Greens_functions(R, r_pts(ip,3), 6.d0, 0.d0, G_BR, G_BZ, G_psi)
    !   write(778,'(5ES14.6)') psi_plasma(ip), G_psi, abs(psi_plasma(ip)-G_psi)/abs(G_psi), r_pts(ip,1), r_pts(ip,3)
    !   write(779,*) B_plasma(ip,3), -G_BZ, ((B_plasma(ip,3)+G_BZ)**2+(B_plasma(ip,1)+G_BR)**2)**0.5d0/(G_BR**2 + G_BZ**2)**0.5
    ! enddo ! --- evaluation points

    ! --- Cleanup 
    deallocate( x_tri, y_tri, z_tri)
    deallocate( green_A_tri, green_B_tri)

  end subroutine vacuum_plasma_fields






  !< This routine calculates the integral of the gradient of the Green's function 
  !! for a set of triangles (taken from STARWALL)
  !! Tese formulas are (35) and (39) from Merkel, P.,arXiv preprint, 2015." URL https://arxiv.org/abs/1508 4911.
  pure subroutine triang_Bfield_greens_integral(x,y,z,x_tri,y_tri,z_tri,green_A_tri, green_B_tri)

    implicit none

    ! --- External parameters
    real*8,  intent(in)     :: x, y, z ! Points where fields are calculated
    real*8,  intent(in)     :: x_tri(:,:), y_tri(:,:), z_tri(:,:)
    real*8,  intent(inout)  :: green_B_tri(:,:), green_A_tri(:)

    ! --- Local parameters
    integer :: ierr, n_mpi, k_delta, k_min, k_max
    integer :: i, j, k, np, ntri, omp_nthreads, omp_tid
    real*8  :: s1,s2,s3                                    &
              ,d221,d232,d213,al1,al2,al3                  &
              ,ata1,ata2,ata3,at                           &
              ,s21,s22,s23,dp1,dm1,dp2,dm2,dp3,dm3         &
              ,ap1,am1,ap2,am2,ap3,am3                     &
              ,h,ar1,ar2,ar3                               &
              ,x21,y21,z21,x32,y32,z32,x13,y13,z13,vx,vy,vz&
              ,tx1,ty1,tz1,tx2,ty2,tz2,tx3,ty3,tz3         &
              ,nx,ny,nz,area,d21,d32,d13,jx,jy,jz          &
              ,dep1,dep2,dep3,dem1,dem2,dem3
    real*8  :: Rcent, jphi, cosx, siny
    real*8  :: x1,y1,z1,x2,y2,z2,x3,y3,z3,sn

    ntri = size(x_tri,1)
 
    do k=1, ntri    ! --- integral over triangles

     !--- only use toroidal current, projection
      x21   = x_tri(k,2) - x_tri(k,1)
      y21   = y_tri(k,2) - y_tri(k,1)
      z21   = z_tri(k,2) - z_tri(k,1)
      x32   = x_tri(k,3) - x_tri(k,2)
      y32   = y_tri(k,3) - y_tri(k,2)
      z32   = z_tri(k,3) - z_tri(k,2)
      x13   = x_tri(k,1) - x_tri(k,3)
      y13   = y_tri(k,1) - y_tri(k,3)
      z13   = z_tri(k,1) - z_tri(k,3)
      d221  = x21**2+y21**2+z21**2
      d232  = x32**2+y32**2+z32**2
      d213  = x13**2+y13**2+z13**2
      d21   = sqrt(d221)
      d32   = sqrt(d232)
      d13   = sqrt(d213)
      nx    = -y21*z13 + z21*y13
      ny    = -z21*x13 + x21*z13
      nz    = -x21*y13 + y21*x13
      area  = 1./sqrt(nx*nx+ny*ny+nz*nz)

      nx    = nx*area
      ny    = ny*area
      nz    = nz*area
      tx1   = (y32*nz-z32*ny)
      ty1   = (z32*nx-x32*nz)
      tz1   = (x32*ny-y32*nx)
      tx2   = (y13*nz-z13*ny)
      ty2   = (z13*nx-x13*nz)
      tz2   = (x13*ny-y13*nx)
      tx3   = (y21*nz-z21*ny)
      ty3   = (z21*nx-x21*nz)
      tz3   = (x21*ny-y21*nx)


      x1    = x_tri(k,1) - x
      y1    = y_tri(k,1) - y
      z1    = z_tri(k,1) - z
      x2    = x_tri(k,2) - x
      y2    = y_tri(k,2) - y
      z2    = z_tri(k,2) - z
      x3    = x_tri(k,3) - x
      y3    = y_tri(k,3) - y
      z3    = z_tri(k,3) - z
      sn    = nx*x1+ny*y1+nz*z1
      h     = abs(sn)
      s21   = x1**2+y1**2+z1**2
      s22   = x2**2+y2**2+z2**2
      s23   = x3**2+y3**2+z3**2
      s1    = sqrt(s21)
      s2    = sqrt(s22)
      s3    = sqrt(s23)
      al1   = alog((s2+s1+d21)/(s1+s2-d21))
      al2   = alog((s3+s2+d32)/(s3+s2-d32))
      al3   = alog((s1+s3+d13)/(s1+s3-d13))
      ar1   = x1*tx3+y1*ty3+z1*tz3
      ar2   = x2*tx1+y2*ty1+z2*tz1
      ar3   = x3*tx2+y3*ty2+z3*tz2
      dp1   = .5*(s22-s21+d221)
      dp2   = .5*(s23-s22+d232)
      dp3   = .5*(s21-s23+d213)
      dm1   = dp1-d221
      dm2   = dp2-d232
      dm3   = dp3-d213
      ap1   = ar1*dp1
      dep1  = ar1**2+h*d221*(h+s2)
      ap2   = ar2*dp2
      dep2  = ar2**2+h*d232*(h+s3)
      ap3   = ar3*dp3
      dep3  = ar3**2+h*d213*(h+s1)
      am1   = ar1*dm1
      dem1  = ar1**2+h*d221*(h+s1)
      am2   = ar2*dm2
      dem2  = ar2**2+h*d232*(h+s2)
      am3   = ar3*dm3
      dem3  = ar3**2+h*d213*(h+s3)
      ata1  = atan2(ap1*dem1-am1*dep1,dep1*dem1+ap1*am1)
      ata2  = atan2(ap2*dem2-am2*dep2,dep2*dem2+ap2*am2)
      ata3  = atan2(ap3*dem3-am3*dep3,dep3*dem3+ap3*am3)
      at    = sign(1.,sn)*(ata1+ata2+ata3)
      vx    = -nx*at + al1*tx3/d21+al2*tx1/d32+al3*tx2/d13
      vy    = -ny*at + al1*ty3/d21+al2*ty1/d32+al3*ty2/d13
      vz    = -nz*at + al1*tz3/d21+al2*tz1/d32+al3*tz2/d13

      green_B_tri(k,:) = (/ vx, vy, vz /) 
      green_A_tri(k)   = (-h*(ata1+ata2+ata3)    &
                         + ar1*al1/d21+ar2*al2/d32+ar3*al3/d13)

    enddo


  end subroutine triang_Bfield_greens_integral






  !< Caculate the magnetic field in a poloidal grid which 
  !< can be larger than the JOREK domain. Needs free-boundary
  !< since it calculates plasma, wall and coil contributions
  !< for each Fourier harmonic.
  subroutine mag_field_including_vacuum(RZ, B_tot, psi_tot)  

    use phys_module, only : R_geo
    use mod_interp
    use mod_plasma_response,  only: plasma_fields_at_xyz
    use mod_boundary, only: get_st_on_bnd
    use nodes_elements
    use constants

    implicit none

    ! --- External parameters
    real*8, allocatable, dimension(:,:), intent(in)      :: RZ    !< R,Z coordinates of points 
                                                                  !< in poloidal plane (:,1) for R, (:,2) for Z
    real*8, allocatable, dimension(:,:,:), intent(inout) :: B_tot !< Magnetic field at given points 
                                                                  !< B_tot(i_pol, i_harmonic, i_comp) i_comp=1 for BR
                                                                  !< and i_comp=2 for BZ
    real*8, allocatable, dimension(:,:), intent(inout) ::psi_tot  !< Psi at RZ points (i_pol, i_harmonic)
                                                                  !< CAREFUL! psi has a different gauge in vacuum for 3D plasmas...
 
    ! --- Local parameters
    real*8, allocatable, dimension(:)     :: psi_tmp, psi_sing, R_elm, Z_elm, distance 
    real*8, allocatable, dimension(:,:)   :: RZ_vac, r_pts, r_pts_sing, psi_vac, &
                                             psi_vac_four, B_sing, B_tmp 
    real*8, allocatable, dimension(:,:,:) :: B_vac, B_vac_four
    real*8  :: R_out,Z_out,s_out,t_out, Ps0,Ps0_s,Ps0_t
    real*8  :: R, R_s, R_t, Z, Z_s, Z_t, xjac, phi, tor_coeff, dist_sing
    real*8  :: R1, Z1, dist_min, s_bnd, s_2d, t_2d, s, t
    integer, allocatable, dimension(:)    :: i_vac, i_sing
    integer, parameter :: n_sbnd=20, n_phi_plasma_vac = 180
    integer :: i, j, k, m, etype, irst, int, i_var, i_tor, index, index_node, my_id, ierr
    integer :: np, i_pol, n_out_pol, i_plane, i_glob, i_elm, i_bnd, i_min, i_s, i_glob_sing
    integer :: ielm_out, ifail, n_sing, side,  i_best(2), n_plane_four
    logical :: s_const

    
    n_plane_four = n_tor*2      ! Necessary number of planes for Fourier transform 
    dist_sing    = R_geo/200.d0 ! Bad (singular) points behaviour seen below a distance of 3 cm in ITER grids
    np = size(RZ(:,1),1)        ! Number of poloidal points

    write(*,*) ' '
    write(*,*) ' **************************************************************************'
    write(*,*) ' *** Extend magnetic field to vacuum region *******************************'
    write(*,*) ' **************************************************************************'
    write(*,*) ' '
    write(*,*) '   n_bnd_elm        = ', bnd_elm_list%n_bnd_elements
    write(*,*) '   n_phi_plasma_vac = ', n_phi_plasma_vac
    write(*,*) '   n_plane_four     = ', n_plane_four
    write(*,*) '   n_poloidal       = ', np
    write(*,'(A, F12.6, A)') '   dist_sing        = ', dist_sing, ' m, (distance below which points are treated as singular)'
    write(*,*) ' '


    allocate(i_vac(np))
    allocate(psi_tot(np,n_tor))
    allocate(B_tot(np,n_tor,2))
    n_out_pol = 0
    i_vac     = 0
    do i_pol=1, np
      ! Check whether the point is in the JOREK domain
      call find_RZ(node_list,element_list,RZ(i_pol,1),RZ(i_pol,2),R_out,Z_out,ielm_out,s_out,t_out,ifail)  
      if (ifail==0) then  ! is inside domain
        call interp_RZ(node_list, element_list, ielm_out, s_out, t_out, R, R_s, R_t, Z, Z_s, Z_t)
        xjac = R_s * Z_t - R_t * Z_s
        do i_tor=1, n_tor
          call interp(node_list,element_list,ielm_out,var_psi,i_tor,s_out,t_out,Ps0,Ps0_s,Ps0_t)
          psi_tot(i_pol,i_tor) = Ps0
          B_tot(i_pol,i_tor,1) =   ( - Ps0_s * R_t + Ps0_t * R_s ) / xjac / R ! BR =  dpsi/dZ / R
          B_tot(i_pol,i_tor,2) = - (   Ps0_s * Z_t - Ps0_t * Z_s ) / xjac / R ! BZ = -dpsi/dR / R
        enddo
      else
        n_out_pol = n_out_pol + 1
        i_vac(i_pol) = n_out_pol
      endif
    enddo

    ! --- Get R, Z coordinates of the middle of the boundary elements (necessary to find closest point to bnd later)
    allocate(R_elm(bnd_elm_list%n_bnd_elements), Z_elm(bnd_elm_list%n_bnd_elements))
    allocate(distance(bnd_elm_list%n_bnd_elements))
    do i_bnd = 1, bnd_elm_list%n_bnd_elements
      i_elm = bnd_elm_list%bnd_element(i_bnd)%element 
      side  = bnd_elm_list%bnd_element(i_bnd)%side
      call get_st_on_bnd(0.5d0, side, s_2d, t_2d, s_const)
      call interp_RZ(node_list, element_list, i_elm, 0.5d0, 0.5d0, R1, R_s, R_t, Z1, Z_s, Z_t)
      R_elm(i_bnd) = R1
      Z_elm(i_bnd) = Z1
    enddo

    ! --- Map RZ points outside domain to a reduced array (RZ_vac) and check whether points are too close to the bnd (singular)
    allocate(i_sing(np))
    allocate(RZ_vac(n_out_pol,2))
    n_sing = 0
    i_sing = 0

    do i_pol=1, np
      if (i_vac(i_pol)>0) then

        RZ_vac(i_vac(i_pol),:) = RZ(i_pol,:)

        ! --- Check whether this points are dangerously close to the boundary (singularities)
        ! Find closest point to the boundary of the domain
        distance  = sqrt( (R_elm-RZ(i_pol,1))**2  + (Z_elm-RZ(i_pol,2))**2)
        i_best(1) = minloc(distance,dim=1)
        dist_min  = distance(i_best(1))
        
        distance(i_best(1)) = 1d99             ! Mask the minimum value to find the second minimum
        i_best(2)           = minloc(distance, dim=1)
        distance(i_best(1)) = dist_min      ! Restore the original minimum distance value

        dist_min = 1.d99

        ! Go along two best boundary elements and find closest local coordinate
        do i_min=1, 2
          side   = bnd_elm_list%bnd_element(i_best(i_min))%side
          i_elm  = bnd_elm_list%bnd_element(i_best(i_min))%element
          ! Go along discretized element
          do i_s=1, n_sbnd
            s_bnd = float(i_s-1)/float(n_sbnd-1)
            call get_st_on_bnd(s_bnd, side, s, t, s_const)
            call interp_RZ(node_list, element_list, i_elm, s, t, R1, R_s, R_t, Z1, Z_s, Z_t)
            if ( sqrt(  (R1-RZ(i_pol,1))**2  + (Z1-RZ(i_pol,2))**2) < dist_min ) then
              dist_min  = sqrt((R1-RZ(i_pol,1))**2  + (Z1-RZ(i_pol,2))**2)
            endif
          end do
        enddo
        if (dist_min < dist_sing) then  
          n_sing = n_sing + 1
          i_sing(n_sing) = i_vac(i_pol)
        endif
      endif ! point is in vacuum
    enddo  ! finished collecting vacuum and checking singular points

    allocate(r_pts(n_out_pol,3))
    allocate(r_pts_sing(n_sing,3))
    allocate(psi_vac(n_out_pol,n_plane_four), B_vac(n_out_pol,n_plane_four,2))
    allocate(psi_tmp(n_out_pol), B_tmp(n_out_pol,3))
    allocate(psi_sing(n_sing),   B_sing(n_sing,3))
    psi_vac = 0.d0;    B_vac = 0.d0;

    ! --- Calculate fields for vacuum points from different contributions for each poloidal plane
    write(*,*) '   --> Calculating plasma, wall and coil fields outside the JOREK domain...'
    do i_plane=1, n_plane_four

      ! --- Get cartesian coordinates for outside points
      phi = 2.d0*PI*float(i_plane-1)/float(n_plane_four) / float(n_period)
        
      r_pts(:,1) =  RZ_vac(:,1) * cos(phi)
      r_pts(:,2) = -RZ_vac(:,1) * sin(phi)
      r_pts(:,3) =  RZ_vac(:,2)
    
      ! --- Collect singular point coordinates
      do i=1, n_sing
        r_pts_sing(i,:) = r_pts(i_sing(i),:)
      enddo

      ! --- Calculate fields in the vacuum produced by the plasma
      ! --- 2D surface efficient integral (but singular in proximity of the JOREK bnd)
      call vacuum_plasma_fields(node_list, element_list, bnd_node_list, bnd_elm_list, &
                                n_phi_plasma_vac, r_pts, B_tmp, psi_tmp, print_params=.false. )
      ! --- Expensive volume integral (only for singular points)
      call plasma_fields_at_xyz(0, node_list,element_list,r_pts_sing(:,1),r_pts_sing(:,2),r_pts_sing(:,3), &
                                B_sing(:,1),B_sing(:,2),B_sing(:,3),psi_sing, n_phi_plasma_vac) 

      ! --- Replace singular points with 3D integral calculation
      do i=1, n_sing
        B_tmp(i_sing(i),:) = B_sing(i,:)
        psi_tmp(i_sing(i)) = psi_sing(i)
      enddo

      psi_vac(:,i_plane) = psi_vac(:,i_plane) + psi_tmp
      B_vac(:,i_plane,1) = B_vac(:,i_plane,1) + B_tmp(:,1)*cos(phi)-B_tmp(:,2)*sin(phi)  ! BR component
      B_vac(:,i_plane,2) = B_vac(:,i_plane,2) + B_tmp(:,3)                               ! BZ component
      psi_tmp  = 0.d0;     B_tmp  = 0.d0;
      psi_sing = 0.d0;     B_sing = 0.d0;

      call wall_fields_at_xyz(0,r_pts(:,1),r_pts(:,2),r_pts(:,3),B_tmp(:,1),B_tmp(:,2),B_tmp(:,3),psi_tmp) 
      psi_vac(:,i_plane) = psi_vac(:,i_plane) + psi_tmp
      B_vac(:,i_plane,1) = B_vac(:,i_plane,1) + B_tmp(:,1)*cos(phi)-B_tmp(:,2)*sin(phi)  ! BR component
      B_vac(:,i_plane,2) = B_vac(:,i_plane,2) + B_tmp(:,3)                               ! BZ component
      psi_tmp = 0.d0;     B_tmp = 0.d0;

      call coil_fields_at_xyz(0,r_pts(:,1),r_pts(:,2),r_pts(:,3),B_tmp(:,1),B_tmp(:,2),B_tmp(:,3),psi_tmp) 
      psi_vac(:,i_plane) = psi_vac(:,i_plane) + psi_tmp
      B_vac(:,i_plane,1) = B_vac(:,i_plane,1) + B_tmp(:,1)*cos(phi)-B_tmp(:,2)*sin(phi)  ! BR component
      B_vac(:,i_plane,2) = B_vac(:,i_plane,2) + B_tmp(:,3)                               ! BZ component
      psi_tmp = 0.d0;     B_tmp = 0.d0;

      write(*,*) '      finished with plane ', i_plane
    enddo

    deallocate(psi_tmp, B_tmp, psi_sing, B_sing)
    write(*,*) '   done.'

    ! --- Do the Fourier transform of the outside points
    write(*,*) '   --> Transform results to Fourier series...'
    allocate(psi_vac_four(n_out_pol,n_tor))
    allocate(B_vac_four(n_out_pol,n_tor,2))
    psi_vac_four = 0.d0
    B_vac_four   = 0.d0
    do i_tor=1, n_tor
      do i_plane=1, n_plane_four

        phi = 2.d0*PI*float(i_plane-1)/float(n_plane_four) / float(n_period)
        if (i_tor==1) then
          tor_coeff = 1.d0
        else if ( mod(i_tor,2) ==0 ) then
          tor_coeff = 2.d0 * cos(phi*float(n_period*i_tor/2) )
        else
          tor_coeff = 2.d0 * sin(phi*float(n_period*(i_tor-1)/2)) 
        endif

        psi_vac_four(:,i_tor) = psi_vac_four(:,i_tor) + psi_vac(:,i_plane)*tor_coeff/ float(n_plane_four)
        B_vac_four(:,i_tor,:) = B_vac_four(:,i_tor,:) + B_vac(:,i_plane,:)*tor_coeff/ float(n_plane_four)
        
      enddo
    enddo
    
    ! --- Map vacuum points to array for the complete set of points
    do i_pol=1, np
      if (i_vac(i_pol)>0) then ! point is in the vacuum region
        do i_tor=1, n_tor
          if (is_freebound(i_tor,var_psi)) then
            psi_tot(i_pol,i_tor) = psi_vac_four(i_vac(i_pol),i_tor)
            B_tot(i_pol,i_tor,:) = B_vac_four(i_vac(i_pol),i_tor,:)
          else ! set to 0 the harmonics that are not free-boundary in vacuum
            psi_tot(i_pol,i_tor) = 0.d0
            B_tot(i_pol,i_tor,:) = 0.d0
          endif
        enddo
      endif
    enddo
    write(*,*) '   done.'

    ! --- Some printing for debugging
    ! do i=1, np
    !   write(5555,'(32ES14.6)') RZ(i,1), RZ(i,2), B_tot(i,:,1)  ! BR
    !   write(5556,'(32ES14.6)') RZ(i,1), RZ(i,2), B_tot(i,:,2)  ! BZ
    ! enddo
      
  end subroutine mag_field_including_vacuum






  pure subroutine get_tri_area_norm(x, y, z, area, norm)
    real(8), intent(in) :: x(3), y(3), z(3)
    real(8), intent(inout) :: area, norm(3)
    real(8) :: ux, uy, uz, vx, vy, vz, crossx, crossy, crossz

    ! Calculate vectors U and V
    ux = x(2) - x(1)
    uy = y(2) - y(1)
    uz = z(2) - z(1)

    vx = x(3) - x(1)
    vy = y(3) - y(1)
    vz = z(3) - z(1)

    ! Calculate the cross product of U and V
    crossx = uy * vz - uz * vy
    crossy = uz * vx - ux * vz
    crossz = ux * vy - uy * vx

    ! Calculate the area of the triangle
    area = 0.5d0 * sqrt(crossx**2 + crossy**2 + crossz**2)

    ! Triangle normal
    norm = (/ crossx, crossy, crossz /) / area*0.5d0
  end subroutine get_tri_area_norm






end module mod_vacuum_fields

