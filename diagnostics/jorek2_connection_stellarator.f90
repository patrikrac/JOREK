!> Connection-length diagnostic for a JOREK stellarator restart (model180/183).
!! Generic by default: connection length = distance traveled to the mesh's own
!! outer boundary (i_elm reaches 0 -- the standard strike convention used by
!! jorek2_poincare.f90/jorek2_connection_flux_aligned.f90). Optionally
!! (use_target=.true., via connlen.nml) additionally tests an analytic
!! HaGrids-style discontinuous target before the mesh boundary is reached.
!!
!! Status codes (connection_length.dat):
!!   1 = analytic target (use_target only), 2 = left the mesh (mesh boundary),
!!   3 = left the valid minor-radius domain [R_MINOR_MIN, R_MINOR_MAX],
!!   0 = L_CAP reached (confined), -1 = start point not found in the mesh
!! For a TARGET-CONFORMING mesh (boundary = plate + closing arc) use use_target=.false.:
!! the wall hit is then status 2.
!!
!! R_now/Z_now are now interpolated after EVERY step,
!! independently of use_target. Previously they were only updated inside the
!! use_target block, so with use_target=.false. the minor-radius check used the stale
!! start position and status 3 could never trigger. Results with use_target=.true.
!! are unchanged (same interpolation, same position, same order of checks).
!!
!! STEPS (somewhat similar to jorek2_connection_flux_aligned_grid):
!! 1.  Read input parameters from connect.nml (optional) and distribute start points from stpts, matching jorek2_poincare
!! 2.  Loop over start points
!!   3. Trace field lines around torus for a pre-set number of turns using a pre-set number of steps.
!!      4. Loop over max turns and steps per turn
!!        5. Perform step.
!!        6. Check if element boundary is crossed.
!!        7. If end boundary is crossed - signal to break and start new field line
!!      8. Record turn based data for poincares.
!! 9.  Write connection length
!! 10. Write strike points
!!
module mod_connlen_step
implicit none
contains

  !> One signed macro-step in phi, handling element crossings + arc-length
  !! accumulation (dl2 formula identical to jorek2_connection_flux_aligned.f90).
  subroutine advance_one_step(i_elm, s_line, t_line, p_line, delta_phi_macro, L_acc, ifail)
  use elements_nodes_neighbours
  use mod_interp
  implicit none

  integer, intent(inout) :: i_elm
  real*8,  intent(inout) :: s_line, t_line, p_line, L_acc
  real*8,  intent(in)    :: delta_phi_macro
  integer, intent(out)   :: ifail

  real*8 :: delta_phi_local, delta_phi_step, delta_s, delta_t
  real*8 :: s_mid, t_mid, p_mid, small_delta, small_delta_s, small_delta_t
  real*8 :: R_in, Z_in, dl2
  real*8 :: Rmid, Zmid, Rmid_s, Rmid_t, Zmid_s, Zmid_t
  integer :: i_steps, i_elm_prev

  ifail = 0
  delta_phi_local = 0.d0
  i_steps = 0

  do while ( (abs(delta_phi_local) .lt. abs(delta_phi_macro)) .and. (i_steps .lt. 10) )
    i_steps = i_steps + 1
    delta_phi_step = delta_phi_macro - delta_phi_local

    call step(i_elm, s_line, t_line, p_line, delta_phi_step, delta_s, delta_t, Rmid, Zmid, Rmid_s, Rmid_t, Zmid_s, Zmid_t)
    s_mid = s_line + 0.5d0*delta_s; t_mid = t_line + 0.5d0*delta_t; p_mid = p_line + 0.5d0*delta_phi_step
    call step(i_elm, s_mid, t_mid, p_mid, delta_phi_step, delta_s, delta_t, Rmid, Zmid, Rmid_s, Rmid_t, Zmid_s, Zmid_t)

    small_delta_s = 1.d0
    if (s_line+delta_s .gt. 1.d0) then; small_delta_s = (1.d0-s_line)/delta_s
    elseif (s_line+delta_s .lt. 0.d0) then; small_delta_s = abs(s_line/delta_s); endif
    small_delta_t = 1.d0
    if (t_line+delta_t .gt. 1.d0) then; small_delta_t = (1.d0-t_line)/delta_t
    elseif (t_line+delta_t .lt. 0.d0) then; small_delta_t = abs(t_line/delta_t); endif
    small_delta = min(small_delta_s, small_delta_t)

    if (small_delta .lt. 1.d0) then
      s_mid = s_line + 0.5d0*small_delta*delta_s; t_mid = t_line + 0.5d0*small_delta*delta_t
      p_mid = p_line + 0.5d0*small_delta*delta_phi_step
      call step(i_elm, s_mid, t_mid, p_mid, delta_phi_step, delta_s, delta_t, Rmid, Zmid, Rmid_s, Rmid_t, Zmid_s, Zmid_t)
      dl2 = ((Rmid_s**2+Zmid_s**2)*delta_s**2 + (Rmid_t**2+Zmid_t**2)*delta_t**2 + Rmid**2*delta_phi_step**2) * small_delta**2

      if (small_delta_s .lt. small_delta_t) then
        if (s_line+delta_s .gt. 1.d0) then
          s_line=1.d0; t_line=t_line+small_delta*delta_t; p_line=p_line+small_delta*delta_phi_step
          call interp_RZP(node_list,element_list,i_elm,s_line,t_line,p_line,R_in,Z_in)
          i_elm_prev=i_elm; i_elm=element_neighbours(2,i_elm_prev)
          if (i_elm==0) then; ifail=1; L_acc=L_acc+sqrt(abs(dl2)); return; end if
          s_line=0.d0
        elseif (s_line+delta_s .lt. 0.d0) then
          s_line=0.d0; t_line=t_line+small_delta*delta_t; p_line=p_line+small_delta*delta_phi_step
          call interp_RZP(node_list,element_list,i_elm,s_line,t_line,p_line,R_in,Z_in)
          i_elm_prev=i_elm; i_elm=element_neighbours(4,i_elm_prev)
          if (i_elm==0) then; ifail=1; L_acc=L_acc+sqrt(abs(dl2)); return; end if
          s_line=1.d0
        endif
      else
        if (t_line+delta_t .gt. 1.d0) then
          s_line=s_line+small_delta*delta_s; t_line=1.d0; p_line=p_line+small_delta*delta_phi_step
          call interp_RZP(node_list,element_list,i_elm,s_line,t_line,p_line,R_in,Z_in)
          i_elm_prev=i_elm; i_elm=element_neighbours(3,i_elm_prev)
          if (i_elm==0) then; ifail=1; L_acc=L_acc+sqrt(abs(dl2)); return; end if
          t_line=0.d0
        elseif (t_line+delta_t .lt. 0.d0) then
          s_line=s_line+small_delta*delta_s; t_line=0.d0; p_line=p_line+small_delta*delta_phi_step
          call interp_RZP(node_list,element_list,i_elm,s_line,t_line,p_line,R_in,Z_in)
          i_elm_prev=i_elm; i_elm=element_neighbours(1,i_elm_prev)
          if (i_elm==0) then; ifail=1; L_acc=L_acc+sqrt(abs(dl2)); return; end if
          t_line=1.d0
        endif
      endif
      L_acc = L_acc + sqrt(abs(dl2))
    else
      s_line=s_line+delta_s; t_line=t_line+delta_t; p_line=p_line+delta_phi_step
      dl2 = (Rmid_s**2+Zmid_s**2)*delta_s**2 + (Rmid_t**2+Zmid_t**2)*delta_t**2 + Rmid**2*delta_phi_step**2
      L_acc = L_acc + sqrt(abs(dl2))
      small_delta = 1.d0
    endif

    delta_phi_local = delta_phi_local + small_delta*delta_phi_step
  end do

  end subroutine advance_one_step


  !> step() -- IDENTICAL physics/branches to jorek2_poincare.f90 (chi + POINC_GVEC),
  !! argument list extended to also return R,Z and s/t-derivatives (needed for dl2).
  subroutine step(i_elm, s_in, t_in, p_in, delta_p, delta_s, delta_t, R_out, Z_out, R_s_out, R_t_out, Z_s_out, Z_t_out)
  use mod_parameters
  use elements_nodes_neighbours
  use phys_module
  use mod_interp
  use mod_chi
  implicit none
  integer :: i_var_psi, i_elm, i_tor, i_harm
  real*8 :: s_in, t_in, p_in, delta_p, delta_s, delta_t
  real*8, intent(out) :: R_out, Z_out, R_s_out, R_t_out, Z_s_out, Z_t_out
  real*8 :: R,R_s,R_t,R_p,Z,Z_s,Z_t,Z_p,dummy,BR0cos,BR0sin,BZ0cos,BZ0sin,Bp0cos,Bp0sin
  real*8 :: Pcos,Pcos_s,Pcos_t,Pcos_st,Pcos_ss,Pcos_tt, Psin,Psin_s,Psin_t,Psin_st,Psin_ss,Psin_tt
  real*8 :: P0,P0_s,P0_t,P0_st,P0_ss,P0_tt, psi_s, psi_t, psi_R, psi_z, psi_p, st_psi_p, Zjac
  real*8 :: BR0, BZ0, Bp0
  real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: chi

  i_var_psi = 1
  call interp_RZP(node_list,element_list,i_elm,s_in,t_in,p_in,R,R_s,R_t,R_p,dummy,dummy,dummy,dummy,dummy,dummy, &
                                                              Z,Z_s,Z_t,Z_p,dummy,dummy,dummy,dummy,dummy,dummy)
  R_out=R; Z_out=Z; R_s_out=R_s; R_t_out=R_t; Z_s_out=Z_s; Z_t_out=Z_t

  chi  = get_chi(R,Z,p_in,node_list,element_list,i_elm,s_in,t_in,max_ord=1)
  Zjac = (R_s*Z_t - R_t*Z_s)

  call interp(node_list,element_list,i_elm,i_var_psi,1,s_in,t_in,P0,P0_s,P0_t,P0_st,P0_ss,P0_tt)
  psi_s=P0_s; psi_t=P0_t; st_psi_p=0.d0

#ifdef POINC_GVEC
  call interp_gvec(node_list,element_list,i_elm,1,1,1,s_in,t_in,BR0,dummy,dummy,dummy,dummy,dummy)
  call interp_gvec(node_list,element_list,i_elm,1,2,1,s_in,t_in,BZ0,dummy,dummy,dummy,dummy,dummy)
  call interp_gvec(node_list,element_list,i_elm,1,3,1,s_in,t_in,Bp0,dummy,dummy,dummy,dummy,dummy)
#endif

  do i_tor = 1, (n_tor-1)/2
    i_harm = 2*i_tor
    call interp(node_list,element_list,i_elm,i_var_psi,i_harm,s_in,t_in,Pcos,Pcos_s,Pcos_t,Pcos_st,Pcos_ss,Pcos_tt)
    psi_s=psi_s+Pcos_s*cos(mode(i_harm)*p_in); psi_t=psi_t+Pcos_t*cos(mode(i_harm)*p_in)
    st_psi_p=st_psi_p-Pcos*mode(i_harm)*sin(mode(i_harm)*p_in)
    call interp(node_list,element_list,i_elm,i_var_psi,i_harm+1,s_in,t_in,Psin,Psin_s,Psin_t,Psin_st,Psin_ss,Psin_tt)
    psi_s=psi_s+Psin_s*sin(mode(i_harm+1)*p_in); psi_t=psi_t+Psin_t*sin(mode(i_harm+1)*p_in)
    st_psi_p=st_psi_p+Psin*mode(i_harm+1)*cos(mode(i_harm+1)*p_in)
  enddo

#ifdef POINC_GVEC
  do i_tor=1,(n_coord_tor-1)/2
    i_harm=2*i_tor
    call interp_gvec(node_list,element_list,i_elm,1,1,i_harm,s_in,t_in,BR0cos,dummy,dummy,dummy,dummy,dummy)
    call interp_gvec(node_list,element_list,i_elm,1,2,i_harm,s_in,t_in,BZ0cos,dummy,dummy,dummy,dummy,dummy)
    call interp_gvec(node_list,element_list,i_elm,1,3,i_harm,s_in,t_in,Bp0cos,dummy,dummy,dummy,dummy,dummy)
    BR0=BR0+BR0cos*cos(mode_coord(i_harm)*p_in); BZ0=BZ0+BZ0cos*cos(mode_coord(i_harm)*p_in); Bp0=Bp0+Bp0cos*cos(mode_coord(i_harm)*p_in)
    call interp_gvec(node_list,element_list,i_elm,1,1,i_harm+1,s_in,t_in,BR0sin,dummy,dummy,dummy,dummy,dummy)
    call interp_gvec(node_list,element_list,i_elm,1,2,i_harm+1,s_in,t_in,BZ0sin,dummy,dummy,dummy,dummy,dummy)
    call interp_gvec(node_list,element_list,i_elm,1,3,i_harm+1,s_in,t_in,Bp0sin,dummy,dummy,dummy,dummy,dummy)
    BR0=BR0-BR0sin*sin(mode_coord(i_harm+1)*p_in); BZ0=BZ0-BZ0sin*sin(mode_coord(i_harm+1)*p_in); Bp0=Bp0-Bp0sin*sin(mode_coord(i_harm+1)*p_in)
  end do
#endif

  psi_R = ( Z_t*psi_s - Z_s*psi_t)/Zjac
  psi_z = (-R_t*psi_s + R_s*psi_t)/Zjac
  psi_p = st_psi_p - R_p*psi_R - Z_p*psi_z

#ifndef POINC_GVEC
  BR0 = chi(1,0,0)   + (psi_z*chi(0,0,1) - psi_p*chi(0,1,0))/(F0*R)
  BZ0 = chi(0,1,0)   - (psi_R*chi(0,0,1) - psi_p*chi(1,0,0))/(F0*R)
  Bp0 = chi(0,0,1)/R + (psi_R*chi(0,1,0) - psi_z*chi(1,0,0))/F0
#endif

  delta_s = (-Z_t*R_p + R_t*Z_p + R*(Z_t*BR0 - R_t*BZ0)/Bp0)*delta_p/Zjac
  delta_t = ( Z_s*R_p - R_s*Z_p - R*(Z_s*BR0 - R_s*BZ0)/Bp0)*delta_p/Zjac
  return
  end subroutine step

end module mod_connlen_step


!> ===================== Main program =====================
program jorek2_connection_stellarator

use data_structure
use phys_module
use basis_at_gaussian
use elements_nodes_neighbours
use mod_neighbours
use mod_import_restart
use mod_log_params
use mod_interp
use mod_chi
use mod_connlen_step
implicit none

! --- Optional analytic target (HaGrids-style). CONFIRM against hagrids/target_generation/target_generation.py.
logical :: use_target = .true.
real*8  :: R0_M=5.90d0, RBOT_M=0.6525d0, ZBOT_M=0.25d0, RTOP_M=0.53d0, ZTOP_M=0.35d0
real*8  :: RTAR_ADD_M=0.08d0, PHI_ROT_DEG=36.0d0, PHI_PERIOD_DEG=72.0d0, PHI_OFFSET_DEG=36.0d0

! --- Tracing control
real*8  :: L_CAP = 1000.d0 ! similar to Sergei Makarovs values in Hagrids
integer :: N_PHI_PER_TURN = 360

real*8  :: R_MINOR_MIN = 0.2821d0, R_MINOR_MAX = 0.7484d0
namelist /connlen_params/ use_target, R0_M, RBOT_M, ZBOT_M, RTOP_M, ZTOP_M, RTAR_ADD_M, &
                          PHI_ROT_DEG, PHI_PERIOD_DEG, PHI_OFFSET_DEG, L_CAP, N_PHI_PER_TURN, &
                          R_MINOR_MIN, R_MINOR_MAX

character(len=512) :: s
integer :: my_id, n_lines, i_lines, i, j, iside_i, iside_j, curr, nr, ntour, ierr
real*8  :: rr, zz, phi
real*8, allocatable  :: R_start(:), Z_start(:), P_start(:)
integer, allocatable :: n_turn(:)
real*8, allocatable  :: L_fwd(:), L_bwd(:)
integer, allocatable :: status_fwd(:), status_bwd(:)
! status: 1=analytic target, 2=mesh boundary, 3=left r-domain, 0=capped(confined), -1=error

integer :: i_elm, ifail, checked_elms, dir, i_turn, i_phi
real*8  :: s_line, t_line, p_line, R_now, Z_now, delta_phi_base, delta_phi_macro, L_acc
real*8  :: R_before, Z_before, R_mid, Z_mid
real*8 :: r_minor_now, p_before

write(*,*) '***************************************'
write(*,*) '* JOREK2_connection_stellarator       *'
write(*,*) '***************************************'

my_id = 0
call det_modes()
call initialise_basis
call init_chi_basis
call initialise_parameters(my_id, "__NO_FILENAME__")
call log_parameters(my_id)

open(41, file='connlen.nml', status='old', action='read', iostat=ierr)
if (ierr == 0) then
  read(41, connlen_params); close(41)
  write(*,*) 'Read connlen.nml.  use_target=', use_target
else
  write(*,*) 'connlen.nml not found -- using generic mesh-boundary connection length.'
end if
write(*,'(A,2f10.4,A,f10.2)') ' r-domain [R_MINOR_MIN, R_MINOR_MAX] = ', R_MINOR_MIN, R_MINOR_MAX, '   L_CAP = ', L_CAP

call import_restart(node_list, element_list, 'jorek_restart', rst_format, ierr, .true.)

allocate(element_neighbours(4, element_list%n_elements))
element_neighbours = 0
do i = 1, element_list%n_elements
  do j = i+1, element_list%n_elements
    if (neighbours(node_list, element_list%element(i), element_list%element(j), iside_i, iside_j)) then
      element_neighbours(iside_i, i) = j
      element_neighbours(iside_j, j) = i
    endif
  enddo
enddo

open(21, file='stpts', status='old', action='read', iostat=ierr)
if (ierr /= 0) then; write(*,*) 'ERROR: could not open stpts file.'; stop; end if
read(21,'(a)') s
read(21,*) n_lines
read(21,'(a)') s
allocate(R_start(n_lines), Z_start(n_lines), P_start(n_lines), n_turn(n_lines))
curr = 0
do
  if (curr >= n_lines) exit
  read(21,*) nr, rr, zz, phi, ntour
  if ((nr == 1) .and. (curr == 0)) then
    R_start(1)=rr; Z_start(1)=zz; P_start(1)=phi; n_turn(1)=ntour
  else
    do i_lines = curr+1, nr
      R_start(i_lines) = R_start(curr) + (rr-R_start(curr))*(real(i_lines-curr)/real(nr-curr))
      Z_start(i_lines) = Z_start(curr) + (zz-Z_start(curr))*(real(i_lines-curr)/real(nr-curr))
      P_start(i_lines) = P_start(curr) + (phi-P_start(curr))*(real(i_lines-curr)/real(nr-curr))
      n_turn(i_lines)  = nint(n_turn(curr) + real(ntour-n_turn(curr))*(real(i_lines-curr)/real(nr-curr)))
    end do
  end if
  curr = nr
end do
close(21)

allocate(L_fwd(n_lines), L_bwd(n_lines), status_fwd(n_lines), status_bwd(n_lines))
L_fwd=-1.d0; L_bwd=-1.d0; status_fwd=-1; status_bwd=-1
delta_phi_base = 2.d0*PI / float(n_period*N_PHI_PER_TURN)

write(*,'(A,i8,A)') ' Tracing ', n_lines, ' points...'

!$omp parallel default(none) &
!$omp   shared(n_lines, R_start, Z_start, P_start, n_turn, delta_phi_base, L_fwd, L_bwd, status_fwd, status_bwd, &
!$omp          node_list, element_list, use_target, L_CAP, N_PHI_PER_TURN, &
!$omp          R0_M, RBOT_M, ZBOT_M, RTOP_M, ZTOP_M, RTAR_ADD_M, PHI_ROT_DEG, PHI_PERIOD_DEG, PHI_OFFSET_DEG, &
!$omp          R_MINOR_MIN, R_MINOR_MAX) &
!$omp   private(i_lines, dir, i_elm, s_line, t_line, p_line, ifail, checked_elms, L_acc, &
!$omp           i_turn, i_phi, delta_phi_macro, R_now, Z_now, R_before, Z_before, R_mid, Z_mid, r_minor_now, p_before)
!$omp do schedule(dynamic)
L_LINES: do i_lines = 1, n_lines

  if (mod(i_lines,200)==1) then
!$omp critical
    write(*,'(1x,a,i8,a,i8)') 'Line ', i_lines, ' of ', n_lines
!$omp end critical
  end if

  do dir = 1, -1, -2

#if STELLARATOR_MODEL
    call find_RZP(node_list, element_list, R_start(i_lines), Z_start(i_lines), P_start(i_lines), &
                  R_now, Z_now, i_elm, s_line, t_line, ifail, checked_elms)
#else
    write(*,*) 'ERROR: requires STELLARATOR_MODEL.'; stop
#endif
    L_acc = 0.d0

    if (ifail /= 0) then
      if (dir==1) then; L_fwd(i_lines)=0.d0; status_fwd(i_lines)=-1
      else; L_bwd(i_lines)=0.d0; status_bwd(i_lines)=-1; end if
      cycle
    end if

    p_line = P_start(i_lines)

    L_TURNS: do i_turn = 1, n_turn(i_lines)
      do i_phi = 1, N_PHI_PER_TURN

        delta_phi_macro = dir * delta_phi_base

        ! save pre-step position for the mid-segment tunneling check
        p_before = p_line
        call interp_RZP(node_list, element_list, i_elm, s_line, t_line, p_line, R_before, Z_before)

        call advance_one_step(i_elm, s_line, t_line, p_line, delta_phi_macro, L_acc, ifail)

        if (ifail /= 0) then
          ! left the mesh entirely -- generic strike condition, checked first,
          ! before any target test (matches the ordering already used elsewhere)
          if (dir==1) then; L_fwd(i_lines)=L_acc; status_fwd(i_lines)=2
          else; L_bwd(i_lines)=L_acc; status_bwd(i_lines)=2; end if
          exit L_TURNS
        end if

        ! post-step position: needed by the target test AND by the r-domain check below
        ! (previously only updated inside the use_target block -> stale with use_target=.false.)
        call interp_RZP(node_list, element_list, i_elm, s_line, t_line, p_line, R_now, Z_now)

        if (use_target) then
          R_mid = 0.5d0*(R_before + R_now)
          Z_mid = 0.5d0*(Z_before + Z_now)

          ! fixed: R_before is tested at p_before (its own phi), not p_line (post-step);
          ! mid-point tested at the averaged phi -- removes the small p_line/position
          if ( target_hit(R_before, Z_before, p_before, R0_M, RBOT_M, ZBOT_M, RTOP_M, ZTOP_M, &
                          RTAR_ADD_M, PHI_ROT_DEG, PHI_PERIOD_DEG, PHI_OFFSET_DEG) .or. &
               target_hit(R_mid,    Z_mid,    0.5d0*(p_before+p_line), R0_M, RBOT_M, ZBOT_M, RTOP_M, ZTOP_M, &
                          RTAR_ADD_M, PHI_ROT_DEG, PHI_PERIOD_DEG, PHI_OFFSET_DEG) .or. &
               target_hit(R_now,    Z_now,    p_line, R0_M, RBOT_M, ZBOT_M, RTOP_M, ZTOP_M, &
                          RTAR_ADD_M, PHI_ROT_DEG, PHI_PERIOD_DEG, PHI_OFFSET_DEG) ) then
            if (dir==1) then; L_fwd(i_lines)=L_acc; status_fwd(i_lines)=1
            else; L_bwd(i_lines)=L_acc; status_bwd(i_lines)=1; end if
            exit L_TURNS
          end if
        end if

        ! minor-radius domain-validity check
        r_minor_now = sqrt((R_now-R0_M)**2 + Z_now**2)
        if ( (r_minor_now > R_MINOR_MAX) .or. (r_minor_now < R_MINOR_MIN) ) then
          if (dir==1) then; L_fwd(i_lines)=L_acc; status_fwd(i_lines)=3
          else; L_bwd(i_lines)=L_acc; status_bwd(i_lines)=3; end if
          exit L_TURNS
        end if

        if (L_acc > L_CAP) then
          if (dir==1) then; L_fwd(i_lines)=L_acc; status_fwd(i_lines)=0
          else; L_bwd(i_lines)=L_acc; status_bwd(i_lines)=0; end if
          exit L_TURNS
        end if

      end do
    end do L_TURNS

  end do ! dir

end do L_LINES
!$omp end do
!$omp end parallel

open(31, file='connection_length.dat', status='replace')
write(31,'(A)') '# i  R_start  Z_start  phi_start  L_fwd  status_fwd  L_bwd  status_bwd  L_total'
write(31,'(A)') '# status: 1=analytic target, 2=mesh boundary, 0=capped(confined), -1=error, 3=left valid r-domain'
do i_lines = 1, n_lines
  write(31,'(i8,3e16.7,e16.7,i4,e16.7,i4,e16.7)') i_lines, R_start(i_lines), Z_start(i_lines), P_start(i_lines), &
        L_fwd(i_lines), status_fwd(i_lines), L_bwd(i_lines), status_bwd(i_lines), &
        merge(L_fwd(i_lines)+L_bwd(i_lines), max(L_fwd(i_lines),L_bwd(i_lines)), &
              (status_fwd(i_lines)>=1 .and. status_bwd(i_lines)>=1))
end do
close(31)
write(*,*) 'Done.  hit_target(f/b)=', count(status_fwd==1), count(status_bwd==1), &
          '  hit_boundary(f/b)=', count(status_fwd==2), count(status_bwd==2), &
          '  capped(f/b)=', count(status_fwd==0), count(status_bwd==0), &
          '  errors(f/b)=', count(status_fwd==-1), count(status_bwd==-1), &
          '  domain_escape(f/b)=', count(status_fwd==3), count(status_bwd==3)


contains

  logical function target_hit(R_in, Z_in, p_in, R0, RBOT, ZBOT, RTOP, ZTOP, RTAR_ADD, PHI_ROT, PHI_PER, PHI_OFF)
  real*8, intent(in) :: R_in, Z_in, p_in, R0, RBOT, ZBOT, RTOP, ZTOP, RTAR_ADD, PHI_ROT, PHI_PER, PHI_OFF
  real*8 :: phi_deg, u, shape_u, vecR, vecZ, phi_rot_rad
  real*8 :: cR(4), cZ(4), locR, locZ, rotR, rotZ, cross, sign_ref
  integer :: k, kk
  logical :: consistent
  phi_deg = mod(p_in*180.d0/PI + PHI_OFF, PHI_PER)
  if (phi_deg < 0.d0) phi_deg = phi_deg + PHI_PER
  u = phi_deg / PHI_PER
  shape_u = sin(PI*u)
  vecR = RBOT - (RBOT-RTOP)*shape_u
  vecZ = ZBOT - (ZBOT-ZTOP)*shape_u
  phi_rot_rad = (-PHI_ROT + 2.d0*PHI_ROT*u) * PI/180.d0
  cR = (/ vecR+RTAR_ADD, vecR, vecR, vecR+RTAR_ADD /)
  cZ = (/ -vecZ, -vecZ, vecZ, vecZ /)
  do k = 1, 4
    locR=cR(k); locZ=cZ(k)
    rotR = cos(phi_rot_rad)*locR - sin(phi_rot_rad)*locZ
    rotZ = sin(phi_rot_rad)*locR + cos(phi_rot_rad)*locZ
    cR(k)=rotR+R0; cZ(k)=rotZ
  end do
  consistent = .true.
  do k = 1, 4
    kk = mod(k,4)+1
    cross = (cR(kk)-cR(k))*(Z_in-cZ(k)) - (cZ(kk)-cZ(k))*(R_in-cR(k))
    if (k==1) then; sign_ref = sign(1.d0,cross)
    else if (sign(1.d0,cross) /= sign_ref) then; consistent=.false.; exit; end if
  end do
  target_hit = consistent
  end function target_hit

end program jorek2_connection_stellarator