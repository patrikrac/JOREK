!> Compute q-profile for a fMHD JOREK restart file from n=0 magnetic field
program q_fmhd

use constants
use mod_parameters
use data_structure
use nodes_elements, only:bnd_node_list,bnd_elm_list
use equil_info
use phys_module
use mod_import_restart
use mod_log_params
use mod_interp
use elements_nodes_neighbours
use mod_find_rz_nearby
use mpi_mod
use mod_element_rtree, only: populate_element_rtree

implicit none

!--- Input parameters --------------------!
!-----------------------------------------!
real*8, parameter  :: stepsize = 1.d-3
integer, parameter :: npoints = 200
!-----------------------------------------!
!-----------------------------------------!

real*8    :: Rstart(npoints)
real*8    :: Zstart(npoints)
integer   :: i, j, k, ielm, ifail, my_id, ierr, ielm_new, ielm_old
real*8    :: Rout, Zout, polturns, torturns
logical   :: stop_tracing

real*8    :: s, t, phi, phinew, phiold, snew, tnew, sold, told
real*8    :: AA(3), AA_s(3), AA_t(3), AA_p(3)
real*8    :: R, R_s, R_t, Z, Z_s, Z_t
real*8    :: BR, BZ, Bp, BB, xjac, Fprof
real*8    :: dum01, dum02, dum03, dum04, dum05
real*8    :: AR, AR_p, AR_s, AR_t, AR_R, AR_Z, AZ, AZ_p, AZ_s, AZ_t, AZ_R, AZ_Z, A3, A3_p, A3_s, A3_t, A3_R, A3_Z, psieq
real*8    :: RR, ZZ, Rnew, Znew, Rold, Zold
real*8    :: Rnewtmp, Znewtmp, RRtmp, ZZtmp

integer   :: required,provided,StatInfo, n_mpi
integer*4 :: rank, comm_size 
logical   :: responsible(npoints)

real*8    :: psin_arr(npoints) = 0.d0, polturns_arr(npoints) = 0.d0, torturns_arr(npoints) = 0.d0, phi_arr(npoints) = 0.d0, R_arr(npoints) = 0.d0
real*8    :: psin_arr_tot(npoints) = 0.d0, phi_arr_tot(npoints) = 0.d0, R_arr_tot(npoints) = 0.d0

required = MPI_THREAD_FUNNELED
call MPI_Init_thread(required, provided, StatInfo)
call init_threads()  ! on some systems init_threads needs to come after mpi_init_thread
call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_size, ierr)
n_mpi = comm_size

! --- Determine ID of each MPI proc
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
my_id = rank

if ( my_id == 0 ) then
  write(*,*) '***************************************'
  write(*,*) '* Calculate q-profile...              *'
  write(*,*) '***************************************'
end if

call det_modes()
call initialise_parameters(my_id,  "__NO_FILENAME__")
call log_parameters(my_id)
if ( my_id == 0 ) call import_restart(node_list,element_list, 'jorek_restart', rst_format, ierr, .true.)
call broadcast_phys(my_id)  
call broadcast_elements(my_id, element_list)                ! elements
call broadcast_nodes(my_id, node_list)                      ! nodes
call broadcast_boundary(my_id,bnd_elm_list,bnd_node_list)
call populate_element_rtree(node_list, element_list)
call broadcast_equil_state(my_id)

if ( my_id == 0 ) write(*,*) '*** ...start tracing... ***'

! --- Define starting points for field lines along outer midplane
responsible = .false.
do i = 1, npoints
  
  if ( ( real(my_id)/real(n_mpi)*npoints < i ) .and. ( real(my_id+1)/real(n_mpi)*npoints >= i ) ) then
    responsible(i) = .true.
    Rstart(i) = ES%R_axis + REAL(i) * (ES%R_midpl(2) - ES%R_axis - 0.002d0)/npoints
    Zstart(i) = ES%Z_axis
  end if
  
end do

! --- Write out distribution of field lines among tasks
do i = 0, n_mpi-1
  if ( my_id == i ) then
    write(*,*) 'Task ', my_id, 'responsible for (field line number & radius):'
    do j = 1, npoints
      if (responsible(j)) write(*,*) j, Rstart(j)
    end do
  end if
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
end do

! --- Trace field lines
do i = 1, npoints
  if ( .not. responsible(i) ) then
    psin_arr(i) = 0.d0
    phi_arr(i)  = 0.d0
    R_arr(i)  = 0.d0
    cycle
  end if
  
  call find_RZ(node_list,element_list,Rstart(i),Zstart(i),RR,ZZ,ielm,s,t,ifail)
  phi = 0.d0
  
  if ( ielm < 1 ) then
    write(*,*) 'Illegal starting point for field line ', i, '. Skipping.'
    cycle
  end if
  
  stop_tracing = .false.
  polturns     = 0.d0
  torturns     = 0.d0
  j            = 0
  
  do while( .not. stop_tracing )
    
    call do_step()
    j = j + 1
   
    if ( ABS(phi) > 2.d0*PI ) then
      if ( phi > 0.d0 ) then
        phi   = phi - 2.d0*PI
      else
        phi   = phi + 2.d0*PI
      end if
      torturns = torturns + 1.d0
    end if

    if ( (RR - ES%R_axis) > 0.d0 .and. (( ZZ - ES%Z_axis ) * ( Zold - ES%Z_axis )) .le. 0.d0 .and. j > 1 ) then
      polturns = polturns + 1.d0
      stop_tracing = .true.
    end if  
    
  end do
  
  psin_arr(i)     = get_psi_n(A3, ZZ)
  if ( phi > 0.d0 ) then
    phi_arr(i)      = phiold + (phi - phiold) * ((Zold - ES%Z_axis)/ (Zold - ZZ)) + torturns * 2.d0 * PI
  else if ( phi < 0.d0 ) then
    phi_arr(i)      = phiold + (phi - phiold) * ((Zold - ES%Z_axis)/ (Zold - ZZ)) - torturns * 2.d0 * PI
  else
    if ( phiold > 0.d0 ) then
      phi_arr(i)      = phiold + (phi - phiold) * ((Zold - ES%Z_axis)/ (Zold - ZZ)) + torturns * 2.d0 * PI
    else
      phi_arr(i)      = phiold + (phi - phiold) * ((Zold - ES%Z_axis)/ (Zold - ZZ)) - torturns * 2.d0 * PI
    end if
  end if
  torturns_arr(i) = torturns
  polturns_arr(i) = polturns
  R_arr(i)        = RR
  write(*,*) 'Finished tracing line ', i
  write(*,*) '...toroidal & poloidal turns:', torturns, polturns

end do

call MPI_Barrier(MPI_COMM_WORLD,ierr)

call MPI_Reduce(psin_arr,  psin_arr_tot,  npoints,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
call MPI_Reduce(phi_arr,   phi_arr_tot,   npoints,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
call MPI_Reduce(R_arr,     R_arr_tot,     npoints,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)

! --- Write out result
if ( my_id == 0 ) then
  open(99, file='q_profile.dat', action='write', status='replace')
  write(99,*) '# Psi_N | Safety factor | Radius at LFS midplane [m]'
  do i = 1, npoints
    write(99,'(3es23.5)') psin_arr_tot(i), phi_arr_tot(i)/(2.d0*PI), R_arr_tot(i)
  end do
  close(99)
end if

contains

  !> A "stupid" stepping routine advancing in R, Z, phi
  subroutine do_step()
  
  call determine_field()
  
  Rold     = RR
  Zold     = ZZ
  phiold   = phi
  sold     = s
  told     = t
  ielm_old = ielm
  
  Rnew   = RR  + 0.5d0 * stepsize * BR / BB
  Znew   = ZZ  + 0.5d0 * stepsize * BZ / BB
  phinew = phi + 0.5d0 * stepsize * Bp / (RR * BB)
  
  call find_RZ_nearby(node_list, element_list, Rold, Zold, sold, told, ielm_old, &
                      Rnew, Znew, snew, tnew, ielm_new, ifail)
  if ( ielm_new < 1 ) then
    write(*,*) 'find_RZ_nearby failed, using find_RZ instead, (R,Z)-position:', Rnew, Znew
    Rnewtmp = Rnew
    Znewtmp = Znew 
    call find_RZ(node_list, element_list, Rnewtmp, Znewtmp, Rnew, Znew, ielm_new, snew, tnew, ifail)
  end if 
  if ( ielm_new < 1 ) then
    write(*,*) 'ATTENTION: both find_RZ_nearby and find_RZ failed!', Rnew, Znew, ielm_new, ifail
    stop_tracing = .true.
    return
  end if
  
  RR   = Rnew
  ZZ   = Znew
  s    = snew
  t    = tnew
  ielm = ielm_new
  phi  = phinew
  call determine_field()
  
  RR   = Rold   + stepsize * BR / BB
  ZZ   = Zold   + stepsize * BZ / BB
  phi  = phiold + stepsize * Bp / (Rold * BB)
  
  call find_RZ_nearby(node_list, element_list, Rold, Zold, sold, told, ielm_old, &
                      RR, ZZ, s, t, ielm, ifail)
  if ( ielm < 1 ) then
    write(*,*) 'find_RZ_nearby failed, using find_RZ instead, (R,Z)-position:', RR, ZZ
    RRtmp = RR
    ZZtmp = ZZ 
    call find_RZ(node_list, element_list, RRtmp, ZZtmp, RR, ZZ, ielm, s, t, ifail)
  end if
  if ( ielm < 1 ) then
    write(*,*) 'ATTENTION: both find_RZ_nearby and find_RZ failed!', RR, ZZ, ielm, ifail
    stop_tracing = .true.
    return
  end if

  end subroutine do_step
 

 !> Determine magnetic field at given position (n=0 only)
  subroutine determine_field()
  
  call interp(node_list, element_list, ielm, 1, 1, s, t, A3, A3_s, A3_t, dum01, dum02, dum03)
  call interp(node_list, element_list, ielm, 2, 1, s, t, AR, AR_s, AR_t, dum01, dum02, dum03)
  call interp(node_list, element_list, ielm, 3, 1, s, t, AZ, AZ_s, AZ_t, dum01, dum02, dum03)
  call interp_RZ(node_list, element_list, ielm, s, t, R, R_s, R_t, Z, Z_s, Z_t)
  
  xjac = R_s*Z_t - R_t*Z_s
  
  AR_R = (   Z_t * AR_s - Z_s * AR_t ) / xjac
  AR_Z = ( - R_t * AR_s + R_s * AR_t ) / xjac
  AZ_R = (   Z_t * AZ_s - Z_s * AZ_t ) / xjac
  AZ_Z = ( - R_t * AZ_s + R_s * AZ_t ) / xjac
  A3_R = (   Z_t * A3_s - Z_s * A3_t ) / xjac
  A3_Z = ( - R_t * A3_s + R_s * A3_t ) / xjac
  
  AR_p = 0.d0
  AZ_p = 0.d0
 
  !call interp(node_list, element_list, ielm, 1, 1, s, t, psieq, dum01, dum02, dum03, dum04, dum05) 
  !call F_profile(xpoint,xcase,Z,ES%Z_xpoint,psieq,ES%psi_axis,ES%psi_bnd,Fprof,dum01,dum02,dum03,dum04,dum05,&
  !               dum06,dum07,dum08,dum09,dum10,dum11)
 
  call interp(node_list, element_list, ielm, 710, 1, s, t, Fprof, dum01, dum02, dum03, dum04, dum05)

  BR = ( A3_Z - AZ_p )/ R
  BZ = ( AR_p - A3_R )/ R
  Bp = ( AZ_R - AR_Z ) + Fprof / R
  BB = sqrt( Bp**2 + BR**2 + BZ**2 )
  
  end subroutine determine_field 

end program q_fmhd
