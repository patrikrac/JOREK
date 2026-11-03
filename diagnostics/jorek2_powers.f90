!**********************************************************************
!* program to calculate the density and power fluxes at the boundary  *
!* from a JOREK2 restart file                                         *
!*                                                                    *
!* This diagnostic has to be run with MPI (mpirun/mpiexec/srun        *
!* depending on the system). For details, see:                        *
!* https://www.jorek.eu/wiki/doku.php?id=diagnostics#diagnostics      *
!**********************************************************************

program jorek_powers
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
use mod_parameters, only: n_var, variable_names
use data_structure
use phys_module
use basis_at_gaussian
use gauss
use constants
use diffusivities, only: get_dperp, get_zkperp
use mod_import_restart
use equil_info, only : get_psi_n
use mod_interp
use constants, only: ATOMIC_MASS_UNIT

implicit none

type (type_node_list)    :: node_list
type (type_element_list) :: element_list

integer :: inode, ielm, my_id
integer :: i, j, k, m, i_tor, index, index_node, n_points, ms
integer :: n_bnd, iv, iv1, iv2, inode1, inode2, ierr
real*8  :: sg, tg, wg, phi, angle, delta_angle, BB2
real*8  :: BigR, xjac, psi_norm
real*8  :: R,R_s,R_t,R_st,R_ss,R_tt,Z,Z_s,Z_t,Z_st,Z_ss,Z_tt
real*8  :: PS,PS_s,PS_t,PS_st,PS_ss,PS_tt, RH,RH_s,RH_t,RH_st, RH_ss, RH_tt
real*8  :: TT,TT_s,TT_t, TT_st, TT_ss, TT_tt, VP,VP_s,VP_t,VP_st,VP_ss,VP_tt
real*8  :: T00, psi, psi_s, psi_t, psi_x, psi_y
real*8  :: rho, rho_s, rho_t, T, T_s, T_t, T_p, T_x, T_y, vpar, D_prof, ZK_prof, ZKpar_T
real*8  :: Kpar_div_in,   Kpar_div_out,   Kpar_wall_in,   Kpar_wall_out
real*8  :: nTV_div_in,    nTV_div_out,    nTV_wall_in,    nTV_wall_out
real*8  :: nV_div_in,     nV_div_out,     nV_wall_in,     nV_wall_out
real*8  :: nV2_div_in,    nV2_div_out,    nV2_wall_in,    nV2_wall_out
real*8  :: TDn_div_in,    TDn_div_out,    TDn_wall_in,    TDn_wall_out
real*8  :: Dperp_div_in,  Dperp_div_out,  Dperp_wall_in,  Dperp_wall_out
real*8  :: Kperp_div_in,  Kperp_div_out,  Kperp_wall_in,  Kperp_wall_out
real*8  :: area_div_in,   area_div_out,   area_wall_in,   area_wall_out
real*8  :: d_Kpar, d_nTV, d_nV2, d_nV, d_Dperp, d_Kperp, d_Tdn, d_area, d_l
real*8  :: Q_div_in,  Q_div_out,  Q_wall_in,  Q_wall_out, Q_par
real*8  :: R_div_in,  R_div_out,  R_wall_in,  R_wall_out, L_div_in,  L_div_out,  L_wall_in,  L_wall_out
real*8  :: t_norm, rho_norm, normal, R_in_out, Z_wall_out, Z_wall_in
logical :: periodic

write(*,*) '*********************************'
write(*,*) '* jorek2_powers                 *'
write(*,*) '*********************************'

! --- Initialise input parameters and read the input namelist.
my_id = 0
call initialise_parameters(my_id, "__NO_FILENAME__")

periodic = .false.

do i_tor=1, n_tor
  mode(i_tor) = + int(i_tor / 2) * n_period
  write(*,*) ' toroidal mode numbers : ',i_tor,mode(i_tor)
enddo

call import_restart(node_list,element_list, 'jorek_restart', rst_format, ierr, .true.)

call initialise_basis                              ! define the basis functions at the Gaussian points

Kpar_div_in  = 0.d0; Kpar_div_out  = 0.d0; Kpar_wall_in  = 0.d0; Kpar_wall_out  = 0.d0
nTV_div_in   = 0.d0; nTV_div_out   = 0.d0; nTV_wall_in   = 0.d0; nTV_wall_out   = 0.d0
nV2_div_in   = 0.d0; nV2_div_out   = 0.d0; nV2_wall_in   = 0.d0; nV2_wall_out   = 0.d0
nV_div_in    = 0.d0; nV_div_out    = 0.d0; nV_wall_in    = 0.d0; nV_wall_out    = 0.d0
TDn_div_in   = 0.d0; TDn_div_out   = 0.d0; TDn_wall_in   = 0.d0; TDn_wall_out   = 0.d0
Dperp_div_in = 0.d0; Dperp_div_out = 0.d0; Dperp_wall_in = 0.d0; Dperp_wall_out = 0.d0
Kperp_div_in = 0.d0; Kperp_div_out = 0.d0; Kperp_wall_in = 0.d0; Kperp_wall_out = 0.d0
area_div_in  = 0.d0; area_div_out  = 0.d0; area_wall_in  = 0.d0; area_wall_out  = 0.d0
Q_div_in     = 0.d0; Q_div_out     = 0.d0; Q_wall_in     = 0.d0; Q_wall_out     = 0.d0
L_div_in     = 0.d0; L_div_out     = 0.d0; L_wall_in     = 0.d0; L_wall_out     = 0.d0

!R_in_out   = 5.d0    ! ITER divertor
!Z_wall_out = -3.2350
!Z_wall_in  = -2.5674 

R_in_out   = 3.03
Z_wall_out = -1.6!-1.9858 
Z_wall_in  = -1.619!-1.9736

do m=1, n_plane

  angle = 2.d0 * PI * float(m-1)/float(n_plane) / float(n_period)

  do i=1,element_list%n_elements

    do iv = 1, n_vertex_max

      iv1 = iv
      iv2 = mod(iv,4) + 1

      inode1 = element_list%element(i)%vertex(iv1)
      inode2 = element_list%element(i)%vertex(iv2)

      if   (( ((node_list%node(inode1)%boundary .eq. 1) .or. (node_list%node(inode1)%boundary .eq. 3))   &
       .and.  ((node_list%node(inode2)%boundary .eq. 1) .or. (node_list%node(inode2)%boundary .eq. 3)) ) &
       .or. ( ((node_list%node(inode1)%boundary .eq. 2) .or. (node_list%node(inode1)%boundary .eq. 3))   &
       .and.  ((node_list%node(inode2)%boundary .eq. 2) .or. (node_list%node(inode2)%boundary .eq. 3)) ) &
       .or. ( ((node_list%node(inode1)%boundary .eq. 4) .or. (node_list%node(inode1)%boundary .eq. 9))   &
       .and.  ((node_list%node(inode2)%boundary .eq. 4) .or. (node_list%node(inode2)%boundary .eq. 9)) ) &
       .or. ( ((node_list%node(inode1)%boundary .eq. 4) .or. (node_list%node(inode1)%boundary .eq. 1))   &
       .and.  ((node_list%node(inode2)%boundary .eq. 4) .or. (node_list%node(inode2)%boundary .eq. 1)) ) &
       .or. ( ((node_list%node(inode1)%boundary .eq. 5) .or. (node_list%node(inode1)%boundary .eq. 9))   &
       .and.  ((node_list%node(inode2)%boundary .eq. 5) .or. (node_list%node(inode2)%boundary .eq. 9)) )) then 

        do ms=1,n_gauss

          if   ( ((node_list%node(inode1)%boundary .eq. 1)  &
              .or.(node_list%node(inode1)%boundary .eq. 4)  &
              .or.(node_list%node(inode1)%boundary .eq. 9)  &
              .or.(node_list%node(inode1)%boundary .eq. 3)) &
           .and. ((node_list%node(inode2)%boundary .eq. 1)  &
              .or.(node_list%node(inode2)%boundary .eq. 4)  &
              .or.(node_list%node(inode2)%boundary .eq. 9)  &
              .or.(node_list%node(inode2)%boundary .eq. 3)) ) then

            sg = xgauss(ms) 
            wg = wgauss(ms)

            if (iv1 .eq. 1) then
              tg     =  0.d0
              normal = -1.d0
            else
              tg     =  1.d0
              normal = +1.d0
            endif

          elseif (((node_list%node(inode1)%boundary .eq. 2) &
              .or.(node_list%node(inode1)%boundary .eq. 3)  &
              .or.(node_list%node(inode1)%boundary .eq. 5)  &
              .or.(node_list%node(inode1)%boundary .eq. 9)) &
           .and. ((node_list%node(inode2)%boundary .eq. 2)  &
              .or.(node_list%node(inode2)%boundary .eq. 3)  &
              .or.(node_list%node(inode2)%boundary .eq. 5)  &
              .or.(node_list%node(inode2)%boundary .eq. 9)) ) then

            tg = xgauss(ms) 
            wg = wgauss(ms)

            if (iv1 .eq. 4) then
              sg     =  0.d0
              normal = -1.d0
            else
              sg     =  1.d0
              normal = +1.d0
            endif

          else
            write(*,*) ' warning boundary element not treated : ',inode1,inode2
          endif

          call interp_RZ(node_list,element_list,i,sg,tg,R,R_s,R_t,R_st,R_ss,R_tt,Z,Z_s,Z_t,Z_st,Z_ss,Z_tt)

          BigR = R
          xjac = R_s * Z_t - R_t * Z_s

          psi = 0.d0; psi_s = 0.d0; psi_t = 0.d0
          rho = 0.d0; rho_s = 0.d0; rho_t = 0.d0;
          T   = 0.d0; T_s   = 0.d0; T_t   = 0.d0; T_p = 0.d0
          Vpar= 0.d0;

          do i_tor = 1,n_tor

            call interp(node_list,element_list,i,1,i_tor,sg,tg,PS,PS_s,PS_t,PS_st,PS_ss,PS_tt)
            psi   = psi   + PS   * HZ(i_tor,m)
            psi_s = psi_s + PS_s * HZ(i_tor,m)
            psi_t = psi_t + PS_t * HZ(i_tor,m)

            call interp(node_list,element_list,i,5,i_tor,sg,tg,RH,RH_s,RH_t,RH_st,RH_ss,RH_tt)
            rho   = rho   + RH   * HZ(i_tor,m)
            rho_s = rho_s + RH_s * HZ(i_tor,m)
            rho_t = rho_t + RH_t * HZ(i_tor,m)

            call interp(node_list,element_list,i,6,i_tor,sg,tg,TT,TT_s,TT_t,TT_st,TT_ss,TT_tt)
            T   = T   + TT   * HZ(i_tor,m)
            T_s = T_s + TT_s * HZ(i_tor,m)
            T_t = T_t + TT_t * HZ(i_tor,m)
            T_p = T_p + TT   * HZ_p(i_tor,m)

            call interp(node_list,element_list,i,7,i_tor,sg,tg,VP,VP_s,VP_t,VP_st,VP_ss,VP_tt)
            Vpar = Vpar + VP * HZ(i_tor,m)

          enddo

          psi_norm = get_psi_n(psi, Z)          

          D_prof  = get_dperp (psi_norm)
          if (use_zkperp_times_density) then
            ZK_prof = get_zkperp(psi_norm) * max(rho,zkperp_density_floor)
          else
            ZK_prof = get_zkperp(psi_norm)
          endif

          T00 = max(T,0.001)
          ZKpar_T = ZK_par * (abs(T00)/T_0)**2.5

          psi_x = (   Z_t * psi_s - Z_s * psi_t ) / xjac
          psi_y = ( - R_t * psi_s + R_s * psi_t ) / xjac
          T_x   = (   Z_t * T_s   - Z_s * T_t ) / xjac
          T_y   = ( - R_t * T_s   + R_s * T_t ) / xjac

          BB2 = (F0**2 + psi_x**2 + psi_y**2) / BigR**2

          if   ( ((node_list%node(inode1)%boundary .eq. 1)  &
              .or.(node_list%node(inode1)%boundary .eq. 4)  &
              .or.(node_list%node(inode1)%boundary .eq. 9)  &
              .or.(node_list%node(inode1)%boundary .eq. 3)) &
           .and. ((node_list%node(inode2)%boundary .eq. 1)  &
              .or.(node_list%node(inode2)%boundary .eq. 4)  &
              .or.(node_list%node(inode2)%boundary .eq. 9)  &
              .or.(node_list%node(inode2)%boundary .eq. 3)) ) then

            d_nTV = - wg * (      rho * T            * Vpar * psi_s) * normal
            d_nV2 = - wg * (0.5 * rho * Vpar**2 *BB2 * Vpar * psi_s) * normal
            d_nV  = - wg * (      rho                * Vpar * psi_s) * normal

            d_Dperp = - wg * D_prof  * ( rho_t * (R_s**2 + Z_s**2) - rho_s * (R_t*R_s+Z_t*Z_s)) * BigR / Xjac 
            d_Kperp = - wg * ZK_prof * ( T_t   * (R_s**2 + Z_s**2) - T_s   * (R_t*R_s+Z_t*Z_s)) * BigR / Xjac 

            d_TDn = T * d_Dperp

!            d_Kpar = - wg * ZKpar_T *((T_s * psi_t - T_t * psi_s) / xjac + F0/BigR * T_p)/BigR/BB2 &
!                   * (- psi_s * normal)

            d_Kpar = - wg * ZKpar_T *((T_x * psi_y - T_y * psi_x) + F0/BigR * T_p)/BigR/BB2 &
                   * (- psi_s * normal)
 
            d_area = wg * sqrt(R_s**2 + Z_s**2) * BigR
            d_l    = wg * sqrt(R_s**2 + Z_s**2)

            Q_par = abs(rho * T * Vpar * psi_s * normal) / BigR / sqrt(R_s**2 + Z_s**2)

          elseif (((node_list%node(inode1)%boundary .eq. 2) &
              .or.(node_list%node(inode1)%boundary .eq. 3)  &
              .or.(node_list%node(inode1)%boundary .eq. 5)  &
              .or.(node_list%node(inode1)%boundary .eq. 9)) &
           .and. ((node_list%node(inode2)%boundary .eq. 2)  &
              .or.(node_list%node(inode2)%boundary .eq. 3)  &
              .or.(node_list%node(inode2)%boundary .eq. 5)  &
              .or.(node_list%node(inode2)%boundary .eq. 9)) ) then

            d_nTV = wg * (      rho * T             * Vpar * psi_t) * normal
            d_nV2 = wg * (0.5 * rho * Vpar**2 * BB2 * Vpar * psi_t) * normal
            d_nV  = wg * (      rho                 * Vpar * psi_t) * normal

            d_Dperp = - wg * D_prof  * ( rho_s * (R_t**2 + Z_t**2) - rho_t * (R_s*R_t+Z_s*Z_t)) * BigR / Xjac 
            d_Kperp = - wg * ZK_prof * ( T_s   * (R_t**2 + Z_t**2) - T_t   * (R_s*R_t+Z_s*Z_t)) * BigR / Xjac

            d_TDn = T * d_Dperp

!            d_Kpar = - wg * ZKpar_T *((T_s * psi_t - T_t * psi_s) / xjac + F0/BigR * T_p)/BigR/BB2 &
!                   * psi_t * normal

            d_Kpar = - wg * ZKpar_T *((T_x * psi_y - T_y * psi_x) + F0/BigR * T_p)/BigR/BB2 &
                   * psi_t * normal

            d_area = wg * sqrt(R_t**2 + Z_t**2) * BigR
            d_l    = wg * sqrt(R_t**2 + Z_t**2)
 
            Q_par = abs(rho * T * Vpar * psi_t * normal) / BigR / sqrt(R_t**2 + Z_t**2)

          else
            write(*,*) ' warning boundary element not treated : ',inode1,inode2
          endif

          if (BigR .gt. R_in_out) then

            if (Z .lt. Z_wall_out) then                      ! outer divertor
              Kpar_div_out   = Kpar_div_out  + d_Kpar
              nTV_div_out    = nTV_div_out   + d_nTV
              Kperp_div_out  = Kperp_div_out + d_Kperp
              Dperp_div_out  = Dperp_div_out + d_Dperp
              nV_div_out     = nV_div_out    + d_nV
              nV2_div_out    = nV2_div_out   + d_nV2
              TDn_div_out    = TDn_div_out   + d_TDn
              area_div_out   = area_div_out  + d_area
              Q_div_out      = max(Q_div_out,Q_par)
              L_div_out      = L_div_out     + d_l
            else                                             ! outer wall
              Kpar_wall_out  = Kpar_wall_out  + d_Kpar
              nTV_wall_out   = nTV_wall_out   + d_nTV
              Kperp_wall_out = Kperp_wall_out + d_Kperp
              Dperp_wall_out = Dperp_wall_out + d_Dperp
              nV_wall_out    = nV_wall_out    + d_nV
              nV2_wall_out   = nV2_wall_out   + d_nV2
              TDn_wall_out   = TDn_wall_out   + d_TDn
              area_wall_out  = area_wall_out  + d_area
              Q_wall_out     = max(Q_wall_out,Q_par)
              L_wall_out     = L_wall_out     + d_l
            endif

          else

            if (Z .lt. Z_wall_in) then                      ! inner divertor
              Kpar_div_in  = Kpar_div_in  + d_Kpar
              nTV_div_in   = nTV_div_in   + d_nTV
              Kperp_div_in = Kperp_div_in + d_Kperp
              Dperp_div_in = Dperp_div_in + d_Dperp
              nV_div_in    = nV_div_in    + d_nV
              nV2_div_in   = nV2_div_in   + d_nV2
              TDn_div_in   = TDn_div_in   + d_TDn
              area_div_in  = area_div_in  + d_area
              Q_div_in     = max(Q_div_in,Q_par)
              L_div_in     = L_div_in     + d_l
            else                                             ! inner wall
              Kpar_wall_in  = Kpar_wall_in  + d_Kpar
              nTV_wall_in   = nTV_wall_in   + d_nTV
              Kperp_wall_in = Kperp_wall_in + d_Kperp
              Dperp_wall_in = Dperp_wall_in + d_Dperp
              nV_wall_in    = nV_wall_in    + d_nV
              nV2_wall_in   = nV2_wall_in   + d_nV2
              TDn_wall_in   = TDn_wall_in   + d_TDn
              area_wall_in  = area_wall_in  + d_area
              Q_wall_in     = max(Q_wall_in,Q_par)
              L_wall_in     = L_wall_in     + d_l
            endif

          endif

        enddo

      else if ((node_list%node(inode1)%boundary .ne. 0) .and. (node_list%node(inode2)%boundary .ne. 0)) then
        write(*,*) 'Error : ',inode1,inode2,node_list%node(inode1)%boundary,node_list%node(inode2)%boundary
        stop
      endif

    enddo
  enddo
enddo

delta_angle = 2.d0 * PI / float(n_plane)

rho_norm = central_density*1.d20 * central_mass * ATOMIC_MASS_UNIT
t_norm   = sqrt(MU_zero*rho_norm)

nTV_div_in   = nTV_div_in   * delta_angle / MU_zero / t_norm * 2.5
nV2_div_in   = nV2_div_in   * delta_angle / MU_zero / t_norm * 1.5
nV_div_in    = nV_div_in    * delta_angle * central_density / t_norm
Dperp_div_in = Dperp_div_in * delta_angle * central_density / t_norm
Kperp_div_in = Kperp_div_in * delta_angle / MU_zero / t_norm * 1.5
Kpar_div_in  = Kpar_div_in  * delta_angle / MU_zero / t_norm * 1.5
Tdn_div_in   = Tdn_div_in   * delta_angle / MU_zero / t_norm * 1.5
area_div_in  = area_div_in  * delta_angle
Q_div_in     = Q_div_in / MU_zero / t_norm * 1.5
L_div_in     = L_div_in / float(n_plane)

nTV_div_out   = nTV_div_out   * delta_angle / MU_zero / t_norm * 2.5
nV2_div_out   = nV2_div_out   * delta_angle / MU_zero / t_norm * 1.5
nV_div_out    = nV_div_out    * delta_angle * central_density / t_norm
Dperp_div_out = Dperp_div_out * delta_angle * central_density / t_norm
Kperp_div_out = Kperp_div_out * delta_angle / MU_zero / t_norm * 1.5
Kpar_div_out  = Kpar_div_out  * delta_angle / MU_zero / t_norm * 1.5
Tdn_div_out   = Tdn_div_out   * delta_angle / MU_zero / t_norm * 1.5
area_div_out  = area_div_out  * delta_angle
Q_div_out     = Q_div_out / MU_zero / t_norm * 1.5
L_div_out     = L_div_out / float(n_plane)

nTV_wall_in   = nTV_wall_in   * delta_angle / MU_zero / t_norm * 2.5
nV2_wall_in   = nV2_wall_in   * delta_angle / MU_zero / t_norm * 1.5
nV_wall_in    = nV_wall_in    * delta_angle * central_density / t_norm
Dperp_wall_in = Dperp_wall_in * delta_angle * central_density / t_norm
Kperp_wall_in = Kperp_wall_in * delta_angle / MU_zero / t_norm * 1.5
Kpar_wall_in  = Kpar_wall_in  * delta_angle / MU_zero / t_norm * 1.5
Tdn_wall_in   = Tdn_wall_in   * delta_angle / MU_zero / t_norm * 1.5
area_wall_in  = area_wall_in  * delta_angle
Q_wall_in     = Q_wall_in / MU_zero / t_norm * 1.5
L_wall_in     = L_wall_in / float(n_plane)

nTV_wall_out   = nTV_wall_out   * delta_angle / MU_zero / t_norm * 2.5
nV2_wall_out   = nV2_wall_out   * delta_angle / MU_zero / t_norm * 1.5
nV_wall_out    = nV_wall_out    * delta_angle * central_density / t_norm
Dperp_wall_out = Dperp_wall_out * delta_angle * central_density / t_norm 
Kperp_wall_out = Kperp_wall_out * delta_angle / MU_zero / t_norm * 1.5
Kpar_wall_out  = Kpar_wall_out  * delta_angle / MU_zero / t_norm * 1.5
Tdn_wall_out   = Tdn_wall_out   * delta_angle / MU_zero / t_norm * 1.5
area_wall_out  = area_wall_out  * delta_angle
Q_wall_out     = Q_wall_out / MU_zero / t_norm * 1.5
L_wall_out     = L_wall_out / float(n_plane)

R_div_out  = Area_div_out  / L_div_out  / (2.d0*PI)
R_div_in   = Area_div_in   / L_div_in   / (2.d0*PI)
R_wall_out = Area_wall_out / L_wall_out / (2.d0*PI)
R_wall_in  = Area_wall_in  / L_wall_in  / (2.d0*PI)

write(*,'(A,3e16.8)') ' time : ',xtime(index_start),xtime(index_start)*t_norm,t_norm

write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor parallel convection : ', &
                        nTV_wall_in/1.d6,nTV_wall_out/1.d6,nTV_div_in/1.d6,nTV_div_out/1.d6,' [MW]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor parallel kinetic energy : ', &
                        nV2_wall_in/1.d6,nV2_wall_out/1.d6,nV2_div_in/1.d6,nV2_div_out/1.d6,' [MW]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor parallel conduction : ', &
                        Kpar_wall_in/1.d6,Kpar_wall_out/1.d6,Kpar_div_in/1.d6,Kpar_div_out/1.d6,' [MW]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor perp. conduction    : ',      &
                        Kperp_wall_in/1.d6,Kperp_wall_out/1.d6,Kperp_div_in/1.d6,Kperp_div_out/1.d6,' [MW]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor energy diffusion    : ',      &
                        Tdn_wall_in/1.d6,Tdn_wall_out/1.d6,Tdn_div_in/1.d6,Tdn_div_out/1.d6,' [MW]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor particle diffusion  : ',      &
                        Dperp_wall_in,Dperp_wall_out,Dperp_div_in,Dperp_div_out,' [10^20 /s]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor particle convection : ',      &
                        nV_wall_in,nV_wall_out,nV_div_in,nV_div_out,' [10^20 /s]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor total area          : ',     &
                        area_wall_in,area_wall_out,area_div_in,area_div_out,' [m^2]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor total length        : ',     &
                        L_wall_in,L_wall_out,L_div_in,L_div_out,' [m]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor total radius (avg)  : ',     &
                        R_wall_in,R_wall_out,R_div_in,R_div_out,' [m]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor Q                   : ',     &
                        Q_wall_in/1.d6,Q_wall_out/1.d6,Q_div_in/1.d6,Q_div_out/1.d6,' [MW/m^2]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor wetted area         : ',     &
                        nTV_wall_in/Q_wall_in,nTV_wall_out/Q_wall_out, &
                        nTV_div_in/Q_div_in,nTV_div_out/Q_div_out,' [m^2]'
write(*,'(A,4e16.8,A)') ' inner/outer wall/divertor wetted length         : ',     &
                        nTV_wall_in/Q_wall_in/(2.*PI*R_wall_in),nTV_wall_out/Q_wall_out/(2.*PI*R_wall_out), &
                        nTV_div_in/Q_div_in/(2.*PI*R_div_in),   nTV_div_out/Q_div_out/(2.*PI*R_div_out),' [m]'

OPEN ( UNIT =77, FILE = 'powers_time.out', ACTION = 'write', POSITION = 'append')
OPEN ( UNIT =79, FILE = 'powers_time_sum.out', ACTION = 'write', POSITION = 'append')



write(77,'(33e16.8)')  xtime(index_start),&
                               nTV_wall_in/1.d6,nTV_wall_out/1.d6,nTV_div_in/1.d6,nTV_div_out/1.d6,         &
                               nV2_wall_in/1.d6,nV2_wall_out/1.d6,nV2_div_in/1.d6,nV2_div_out/1.d6,         &
                               Kpar_wall_in/1.d6,Kpar_wall_out/1.d6,Kpar_div_in/1.d6,Kpar_div_out/1.d6,     &
                               Kperp_wall_in/1.d6,Kperp_wall_out/1.d6,Kperp_div_in/1.d6,Kperp_div_out/1.d6, &
                               Tdn_wall_in/1.d6,Tdn_wall_out/1.d6,Tdn_div_in/1.d6,Tdn_div_out/1.d6,         &
                               Dperp_wall_in,Dperp_wall_out,Dperp_div_in,Dperp_div_out,                     &
                               nV_wall_in,nV_wall_out,nV_div_in,nV_div_out,                                 &
                               nTV_wall_in/Q_wall_in,nTV_wall_out/Q_wall_out,                               &
                               nTV_div_in/Q_div_in,nTV_div_out/Q_div_out

write(79,'(9e16.8)')   xtime(index_start),&
                               (nTV_wall_in   + nTV_wall_out   + nTV_div_in   + nTV_div_out)  /1.d6,         &
                               (nV2_wall_in   + nV2_wall_out   + nV2_div_in   + nV2_div_out)  /1.d6,         &
                               (Kpar_wall_in  + Kpar_wall_out  + Kpar_div_in  + Kpar_div_out) /1.d6,         &
                               (Kperp_wall_in + Kperp_wall_out + Kperp_div_in + Kperp_div_out)/1.d6,         &
                               (Tdn_wall_in   + Tdn_wall_out   + Tdn_div_in   + Tdn_div_out)/1.d6,           &
                               (Dperp_wall_in + Dperp_wall_out + Dperp_div_in + Dperp_div_out),              &
                               (nV_wall_in    + nV_wall_out    + nV_div_in    + nV_div_out),                 &
                               (nTV_wall_in/Q_wall_in + nTV_wall_out/Q_wall_out + nTV_div_in/Q_div_in +      &
                               nTV_div_out/Q_div_out)

close (77)
close (79)
end


