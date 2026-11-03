!> Interpolation in the JOREK elements for values, deltas or x
module mod_interp
use data_structure
use mod_basisfunctions
use mod_parameters, only: n_period, n_coord_period, n_tor, n_coord_tor, n_degrees
use phys_module, only: mode, mode_coord
implicit none
private
public :: interp !< interp a specific harmonic in finite elements
public :: interp_delta !< interp a specific harmonic in finite elements, of the deltas
public :: interp_0 !< interp variable only, no derivatives at a specific position in domain
public :: interp_0_delta !< interp variable only, no derivatives at a specific position in domain, of the deltas
public :: interp_RZ !< Interpolate space only
public :: interp_RZP !< interpolate RZ at a given (s,t,phi) for a non-axisymmetric configuration 
public :: interp_PRZ !< interp variable + pos at values or deltas
public :: interp_PRZP !< interp variable + pos at values or deltas at a given (s,t,phi) for a non-axisymmetric configuration
public :: interp_gvec !< interpolate equilibrium parameters imported from GVEC
public :: sincosperiod_moivre, mode_moivre !< public for regtesting, used by interp_PRZ
public :: interp_PRZ_combined !< same as interp, but for any variable, including R and Z

interface interp
  module procedure interp_2,interp_1,interp_0_single_harmonic
end interface interp

interface interp_delta
  module procedure interp_2_delta,interp_1_delta,interp_0_single_harmonic_delta
end interface interp_delta

interface interp_RZ
  module procedure interp_RZ_0, interp_RZ_1, interp_RZ_2
end interface interp_RZ

interface interp_PRZ
  module procedure interp_PRZ_0, interp_PRZ_1, interp_PRZ_2
end interface interp_PRZ

interface interp_PRZP
  module procedure interp_PRZP_0, interp_PRZP_1, interp_PRZP_2
end interface interp_PRZP

interface interp_RZP
  module procedure interp_RZP_0, interp_RZP_1, interp_RZP_2
end interface interp_RZP

contains

!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t)
pure subroutine interp_PRZ_0(node_list, element_list, i_elm, i_v, n_v, s, t, phi, P, R, Z, deltas)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8,                   intent(out) :: P(n_v)
real*8,                   intent(out) :: R, Z
logical, optional, intent(in)         :: deltas

! --- Local variables
real*8  :: H(n_degrees,4), H_T(4,n_degrees), HZ(n_tor), dHZ(n_tor)
integer :: kv, iv, kf, i
real*8  :: values(n_tor,n_degrees,n_v,n_vertex_max)
real*8  :: xR(n_degrees,n_vertex_max), xZ(n_degrees,n_vertex_max)
real*8  :: sizes(n_degrees), v, vp
logical :: my_deltas

call basisfunctions(s,t,H_T)
H = transpose(H_T)
call sincosperiod_moivre(phi, HZ, dHZ) ! dHZ unused

P = 0.d0

my_deltas = .false.
if (present(deltas)) then
  if (deltas) my_deltas = .true.
end if

! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)

  if (my_deltas) then
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%deltas(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  else
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%values(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  end if
  xR(:,kv) = node_list%node(iv)%x(1,:,1) * sizes(:)
  xZ(:,kv) = node_list%node(iv)%x(1,:,2) * sizes(:)
end do

! together 7%
R   = sum(xR*H)
Z   = sum(xZ*H)

! 40% exec time
do kv = 1, n_vertex_max
  do i = 1, n_v
    do kf = 1, n_degrees
      v = dot_product(values(1:n_tor,kf,i,kv),HZ(1:n_tor))
      P(i)     = P(i)     + v * H(kf, kv)
    enddo
  enddo
enddo
end subroutine interp_PRZ_0

!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t)
pure subroutine interp_PRZ_1(node_list, element_list, i_elm, i_v, n_v, s, t, phi, P, P_s, P_t, P_phi, R, R_s, R_t, Z, Z_s, Z_t, deltas)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8,                   intent(out) :: P(n_v), P_s(n_v), P_t(n_v), P_phi(n_v)
real*8,                   intent(out) :: R, R_s, R_t, Z, Z_s, Z_t
logical, optional, intent(in)         :: deltas

! --- Local variables
real*8  :: H(n_degrees,4), H_s(n_degrees,4), H_t(n_degrees,4), HZ(n_tor), dHZ(n_tor)
integer :: kv, iv, kf, i
real*8  :: values(n_tor,n_degrees,n_v,n_vertex_max)
real*8  :: xR(n_degrees,n_vertex_max), xZ(n_degrees,n_vertex_max)
real*8  :: sizes(n_degrees), v, vp
logical :: my_deltas

! 7% exec time
call basisfunctions_T(s,t,H,H_s,H_t)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_phi = 0.d0

! 7% exec time
call sincosperiod_moivre(phi, HZ, dHZ)

my_deltas = .false.
if (present(deltas)) then
  if (deltas) my_deltas = .true.
end if

! 30% exec time
! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)

  if (my_deltas) then
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%deltas(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  else
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%values(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  end if
  xR(:,kv) = node_list%node(iv)%x(1,:,1) * sizes(:)
  xZ(:,kv) = node_list%node(iv)%x(1,:,2) * sizes(:)
end do

! together 7%
R   = sum(xR*H)
R_s = sum(xR*H_s)
R_t = sum(xR*H_t)
Z   = sum(xZ*H)
Z_s = sum(xZ*H_s)
Z_t = sum(xZ*H_t)

! 40% exec time
do kv = 1, n_vertex_max
  do i = 1, n_v
    do kf = 1, n_degrees
      v = dot_product(values(1:n_tor,kf,i,kv),HZ(1:n_tor))
      P(i)     = P(i)     + v * H(kf, kv)
      P_s(i)   = P_s(i)   + v * H_s(kf, kv)
      P_t(i)   = P_t(i)   + v * H_t(kf, kv)
      vp = dot_product(values(1:n_tor,kf,i,kv),dHZ(1:n_tor))
      P_phi(i) = P_phi(i) + vp * H(kf, kv)
    enddo
  enddo
enddo
end subroutine interp_PRZ_1


!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t)
pure subroutine interp_PRZ_2(node_list, element_list, i_elm, i_v, n_v, s, t, phi, &
        P, P_s, P_t, P_phi, P_st, P_ss, P_tt, P_sphi, P_tphi, P_phiphi, R, R_s, R_t, R_st, R_ss, R_tt, &
        Z, Z_s, Z_t, Z_st, Z_ss, Z_tt, deltas)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8, dimension(n_v),   intent(out) :: P, P_s, P_t, P_phi, P_st, P_ss, P_tt, P_sphi, P_tphi, P_phiphi
real*8,                   intent(out) :: R, R_s, R_t, R_st, R_ss, R_tt, Z, Z_s, Z_t, Z_st, Z_ss, Z_tt
logical, optional, intent(in)         :: deltas

! --- Local variables
real*8, dimension(n_degrees,n_vertex_max) :: H, H_s, H_t, H_st, H_ss, H_tt
real*8, dimension(n_tor) :: HZ(n_tor), dHZ(n_tor), ddHZ(n_tor)
integer :: kv, iv, kf, i
real*8  :: values(n_tor,n_degrees,n_v,n_vertex_max)
real*8  :: xR(n_degrees,n_vertex_max), xZ(n_degrees,n_vertex_max)
real*8  :: sizes(n_degrees), v, vp, vpp
logical :: my_deltas

call basisfunctions_T(s,t,H,H_s,H_t,H_st,H_ss,H_tt)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_st = 0.d0; P_ss = 0.d0; P_tt = 0.d0
P_phi = 0.d0; P_sphi = 0.d0; P_tphi = 0.d0; P_phiphi = 0.d0

call sincosperiod_moivre(phi, HZ, dHZ)
do i=1,n_tor
  ddHZ(i) = HZ(i)*(n_period*(i/2))**2
end do

my_deltas = .false.
if (present(deltas)) then
  if (deltas) my_deltas = .true.
end if

! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)

  if (my_deltas) then
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%deltas(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  else
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%values(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  end if
  xR(:,kv) = node_list%node(iv)%x(1,:,1) * sizes(:)
  xZ(:,kv) = node_list%node(iv)%x(1,:,2) * sizes(:)
end do

R    = sum(xR*H)
R_s  = sum(xR*H_s)
R_t  = sum(xR*H_t)
R_st = sum(xR*H_st)
R_ss = sum(xR*H_ss)
R_tt = sum(xR*H_tt)
Z    = sum(xZ*H)
Z_s  = sum(xZ*H_s)
Z_t  = sum(xZ*H_t)
Z_st = sum(xZ*H_st)
Z_ss = sum(xZ*H_ss)
Z_tt = sum(xZ*H_tt)

do kv = 1, n_vertex_max
  do i = 1, n_v
    do kf = 1, n_degrees
      v = dot_product(values(1:n_tor,kf,i,kv),HZ(1:n_tor))
      P(i)     = P(i)     + v * H(kf, kv)
      P_s(i)   = P_s(i)   + v * H_s(kf, kv)
      P_t(i)   = P_t(i)   + v * H_t(kf, kv)
      vp = dot_product(values(1:n_tor,kf,i,kv),dHZ(1:n_tor))
      P_phi(i) = P_phi(i) + vp * H(kf, kv)

      P_st(i)  = P_st(i)  + v * H_st(kf, kv)
      P_ss(i)  = P_ss(i)  + v * H_ss(kf, kv)
      P_tt(i)  = P_tt(i)  + v * H_tt(kf, kv)

      P_sphi(i)   = P_sphi(i)   + vp * H_s(kf, kv)
      P_tphi(i)   = P_tphi(i)   + vp * H_t(kf, kv)
      vpp = dot_product(values(1:n_tor,kf,i,kv),ddHZ(1:n_tor))
      P_phiphi(i) = P_phiphi(i) + vpp * H(kf, kv)
    enddo
  enddo
enddo
end subroutine interp_PRZ_2


!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t,phi)
pure subroutine interp_PRZP_0(node_list, element_list, i_elm, i_v, n_v, s, t, phi, P, R, Z, deltas)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8,                   intent(out) :: P(n_v)
real*8,                   intent(out) :: R, Z
logical, optional, intent(in)         :: deltas

! --- Local variables
real*8  :: H(n_degrees,4), H_T(4,n_degrees), HZ(n_tor), dHZ(n_tor), HZ_coord(n_coord_tor), dHZ_coord(n_coord_tor)
integer :: kv, iv, kf, i
real*8  :: values(n_tor,n_degrees,n_v,n_vertex_max)
real*8  :: xR(n_coord_tor,n_degrees,n_vertex_max), xZ(n_coord_tor,n_degrees,n_vertex_max)
real*8  :: sizes(n_degrees), v, vp
logical :: my_deltas

call basisfunctions(s,t,H_T)
H = transpose(H_T)
call sincosperiod_moivre(phi, HZ, dHZ) ! dHZ unused
call sincosperiod_moivre_ncoord(phi, HZ_coord, dHZ_coord) ! dHZ unused

P = 0.d0; R = 0.d0; Z = 0.d0

my_deltas = .false.
if (present(deltas)) then
  if (deltas) my_deltas = .true.
end if

! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)

  if (my_deltas) then
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%deltas(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  else
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%values(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  end if
  do kf=1,n_degrees
    xR(1:n_coord_tor,kf,kv) = node_list%node(iv)%x(1:n_coord_tor,kf,1) * sizes(kf)
    xZ(1:n_coord_tor,kf,kv) = node_list%node(iv)%x(1:n_coord_tor,kf,2) * sizes(kf)
  enddo
end do

do kv = 1, n_vertex_max
  do i = 1, n_v
    do kf = 1, n_degrees
      v = dot_product(values(1:n_tor,kf,i,kv),HZ(1:n_tor))
      P(i)     = P(i)     + v * H(kf, kv)
    enddo
  enddo
  do kf = 1, n_degrees
    v = dot_product(xR(1:n_coord_tor,kf,kv),HZ_coord(1:n_coord_tor))
    R = R + v * H(kf,kv)
    v = dot_product(xZ(1:n_coord_tor,kf,kv),HZ_coord(1:n_coord_tor))
    Z = Z + v * H(kf,kv)
  enddo
enddo
end subroutine interp_PRZP_0

!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t,phi)
!> including up to first order R, Z phi derivatives
pure subroutine interp_PRZP_1(node_list, element_list, i_elm, i_v, n_v, s, t, phi, P, P_s, P_t, P_phi,  & 
                             R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi, deltas)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8,                   intent(out) :: P(n_v), P_s(n_v), P_t(n_v), P_phi(n_v)
real*8,                   intent(out) :: R, R_s, R_t, R_phi, Z, Z_s, Z_t, Z_phi
logical, optional, intent(in)         :: deltas

! --- Local variables
real*8  :: H(n_degrees,4), H_s(n_degrees,4), H_t(n_degrees,4), HZ(n_tor), dHZ(n_tor), HZ_coord(n_coord_tor), dHZ_coord(n_coord_tor)
integer :: kv, iv, kf, i
real*8  :: values(n_tor,n_degrees,n_v,n_vertex_max)
real*8  :: xR(n_coord_tor,n_degrees,n_vertex_max), xZ(n_coord_tor,n_degrees,n_vertex_max)
real*8  :: sizes(n_degrees), v, vp
logical :: my_deltas

! 7% exec time
call basisfunctions_T(s,t,H,H_s,H_t)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_phi = 0.d0
R = 0.d0; R_s = 0.d0; R_t = 0.d0; R_phi = 0.d0
Z = 0.d0; Z_s = 0.d0; Z_t = 0.d0; Z_phi = 0.d0

! 7% exec time
call sincosperiod_moivre(phi, HZ, dHZ)
call sincosperiod_moivre_ncoord(phi, HZ_coord, dHZ_coord)

my_deltas = .false.
if (present(deltas)) then
  if (deltas) my_deltas = .true.
end if

! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)

  if (my_deltas) then
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%deltas(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  else
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%values(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  end if
  do kf=1,n_degrees
    xR(1:n_coord_tor,kf,kv) = node_list%node(iv)%x(1:n_coord_tor,kf,1) * sizes(kf)
    xZ(1:n_coord_tor,kf,kv) = node_list%node(iv)%x(1:n_coord_tor,kf,2) * sizes(kf)
  enddo
end do

do kv = 1, n_vertex_max
  do i = 1, n_v
    do kf = 1, n_degrees
      v = dot_product(values(1:n_tor,kf,i,kv),HZ(1:n_tor))
      P(i)     = P(i)     + v * H(kf, kv)
      P_s(i)   = P_s(i)   + v * H_s(kf, kv)
      P_t(i)   = P_t(i)   + v * H_t(kf, kv)
      vp = dot_product(values(1:n_tor,kf,i,kv),dHZ(1:n_tor))
      P_phi(i) = P_phi(i) + vp * H(kf, kv)
    enddo
  enddo
  do kf = 1, n_degrees
    v = dot_product(xR(1:n_coord_tor,kf,kv),HZ_coord(1:n_coord_tor))
    R = R + v * H(kf,kv)
    R_s = R_s + v * H_s(kf,kv)
    R_t = R_t + v * H_t(kf,kv)
    vp = dot_product(xR(1:n_coord_tor,kf,kv),dHZ_coord(1:n_coord_tor))
    R_phi = R_phi + vp * H(kf,kv)

    v = dot_product(xZ(1:n_coord_tor,kf,kv),HZ_coord(1:n_coord_tor))
    Z = Z + v * H(kf,kv)
    Z_s = Z_s + v * H_s(kf,kv)
    Z_t = Z_t + v * H_t(kf,kv)
    vp = dot_product(xZ(1:n_coord_tor,kf,kv),dHZ_coord(1:n_coord_tor))
    Z_phi = Z_phi + vp * H(kf,kv)
  enddo
enddo
end subroutine interp_PRZP_1


!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t,phi)
!> including up to second order R, Z phi derivatives
pure subroutine interp_PRZP_2(node_list, element_list, i_elm, i_v, n_v, s, t, phi, &
                              P, P_s, P_t, P_phi, P_st, P_ss, P_tt, P_sphi, P_tphi, P_phiphi, & 
                              R, R_s, R_t, R_phi, R_st, R_ss, R_tt, R_sphi, R_tphi, R_phiphi, &
                              Z, Z_s, Z_t, Z_phi, Z_st, Z_ss, Z_tt, Z_sphi, Z_tphi, Z_phiphi, deltas)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8, dimension(n_v),   intent(out) :: P, P_s, P_t, P_phi, P_st, P_ss, P_tt, P_sphi, P_tphi, P_phiphi
real*8,                   intent(out) :: R, R_s, R_t, R_phi, R_st, R_ss, R_tt, R_sphi, R_tphi, R_phiphi
real*8,                   intent(out) :: Z, Z_s, Z_t, Z_phi, Z_st, Z_ss, Z_tt, Z_sphi, Z_tphi, Z_phiphi
logical, optional, intent(in)         :: deltas

! --- Local variables
real*8, dimension(n_degrees,n_vertex_max) :: H, H_s, H_t, H_st, H_ss, H_tt
real*8, dimension(n_tor) :: HZ(n_tor), dHZ(n_tor), ddHZ(n_tor)
real*8, dimension(n_coord_tor) :: HZ_coord(n_coord_tor), dHZ_coord(n_coord_tor), ddHZ_coord(n_coord_tor)
integer :: kv, iv, kf, i
real*8  :: values(n_tor,n_degrees,n_v,n_vertex_max)
real*8  :: xR(n_coord_tor, n_degrees, n_vertex_max), xZ(n_coord_tor, n_degrees, n_vertex_max)
real*8  :: sizes(n_degrees), v, vp, vpp
logical :: my_deltas

call basisfunctions_T(s,t,H,H_s,H_t,H_st,H_ss,H_tt)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_st = 0.d0; P_ss = 0.d0; P_tt = 0.d0; P_phi = 0.d0; P_sphi = 0.d0; P_tphi = 0.d0; P_phiphi = 0.d0
R = 0.d0; R_s = 0.d0; R_t = 0.d0; R_st = 0.d0; R_ss = 0.d0; R_tt = 0.d0; R_phi = 0.d0; R_sphi = 0.d0; R_tphi = 0.d0; R_phiphi = 0.d0
Z = 0.d0; Z_s = 0.d0; Z_t = 0.d0; Z_st = 0.d0; Z_ss = 0.d0; Z_tt = 0.d0; Z_phi = 0.d0; Z_sphi = 0.d0; Z_tphi = 0.d0; Z_phiphi = 0.d0

call sincosperiod_moivre(phi, HZ, dHZ)
do i=1,n_tor
  ddHZ(i) = HZ(i)*(n_period*(i/2))**2
end do
call sincosperiod_moivre_ncoord(phi, HZ_coord, dHZ_coord)
do i=1,n_coord_tor
  ddHZ_coord(i) = HZ_coord(i)*(n_coord_period*(i/2))**2
end do

my_deltas = .false.
if (present(deltas)) then
  if (deltas) my_deltas = .true.
end if

! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)

  if (my_deltas) then
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%deltas(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  else
    do i = 1, n_v
      do kf=1,n_degrees
        values(1:n_tor,kf,i,kv) = node_list%node(iv)%values(1:n_tor,kf,i_v(i)) * sizes(kf)
      end do
    end do
  end if
  do kf=1,n_degrees
    xR(1:n_coord_tor,kf,kv) = node_list%node(iv)%x(1:n_coord_tor,kf,1) * sizes(kf)
    xZ(1:n_coord_tor,kf,kv) = node_list%node(iv)%x(1:n_coord_tor,kf,2) * sizes(kf)
  enddo
end do

do kv = 1, n_vertex_max
  do i = 1, n_v
    do kf = 1, n_degrees
      v = dot_product(values(1:n_tor,kf,i,kv),HZ(1:n_tor))
      P(i)     = P(i)     + v * H(kf, kv)
      P_s(i)   = P_s(i)   + v * H_s(kf, kv)
      P_t(i)   = P_t(i)   + v * H_t(kf, kv)
      vp = dot_product(values(1:n_tor,kf,i,kv),dHZ(1:n_tor))
      P_phi(i) = P_phi(i) + vp * H(kf, kv)

      P_st(i)  = P_st(i)  + v * H_st(kf, kv)
      P_ss(i)  = P_ss(i)  + v * H_ss(kf, kv)
      P_tt(i)  = P_tt(i)  + v * H_tt(kf, kv)

      P_sphi(i)   = P_sphi(i)   + vp * H_s(kf, kv)
      P_tphi(i)   = P_tphi(i)   + vp * H_t(kf, kv)
      vpp = dot_product(values(1:n_tor,kf,i,kv),ddHZ(1:n_tor))
      P_phiphi(i) = P_phiphi(i) + vpp * H(kf, kv)
    enddo
  enddo
  do kf = 1, n_degrees
    v = dot_product(xR(1:n_coord_tor,kf,kv),HZ_coord(1:n_coord_tor))
    R = R + v * H(kf,kv)
    R_s = R_s + v * H_s(kf,kv)
    R_t = R_t + v * H_t(kf,kv)
    vp = dot_product(xR(1:n_coord_tor,kf,kv),dHZ_coord(1:n_coord_tor))
    R_phi = R_phi + vp * H(kf,kv)

    R_st = R_st + v * H_st(kf,kv)
    R_ss = R_ss + v * H_ss(kf,kv)
    R_tt = R_tt + v * H_tt(kf,kv)

    R_sphi = R_sphi + vp*H_s(kf,kv)
    R_tphi = R_tphi + vp*H_s(kf,kv)
    vpp = dot_product(xR(1:n_coord_tor,kf,kv), ddHZ_coord(1:n_coord_tor))
    R_phiphi = R_phiphi + vpp*H(kf,kv)
  
    v = dot_product(xZ(1:n_coord_tor,kf,kv),HZ_coord(1:n_coord_tor))
    Z = Z + v * H(kf,kv)
    Z_s = Z_s + v * H_s(kf,kv)
    Z_t = Z_t + v * H_t(kf,kv)
    vp = dot_product(xZ(1:n_coord_tor,kf,kv),dHZ_coord(1:n_coord_tor))
    Z_phi = Z_phi + vp * H(kf,kv)

    Z_st = Z_st + v * H_st(kf,kv)
    Z_ss = Z_ss + v * H_ss(kf,kv)
    Z_tt = Z_tt + v * H_tt(kf,kv)

    Z_sphi = Z_sphi + vp*H_s(kf,kv)
    Z_tphi = Z_tphi + vp*H_s(kf,kv)
    vpp = dot_product(xZ(1:n_coord_tor,kf,kv), ddHZ_coord(1:n_coord_tor))
    Z_phiphi = Z_phiphi + vpp*H(kf,kv)
  enddo
enddo
end subroutine interp_PRZP_2

! Apply De Moivre formula to calculate the series of sines.
! Assumes that mode is of the form [0 1 1 2 2 3 3 4 4] ([0 4 4 8 8 12 12])
! This is roughly 3-4 times faster in my tests than just calculating the sines
! and cosines (even when that is vectorized). Perhaps that changes for n_tor >> 10
! I tested n_tor = 17.
#ifdef UNIT_TESTS
pure subroutine sincosperiod_moivre(phi,HZ,dHZ,n_tor_in,n_period_in)
  real*8, intent(out) :: HZ(:), dHZ(:)
  integer,intent(in),optional :: n_tor_in,n_period_in
#else
pure subroutine sincosperiod_moivre(phi,HZ,dHZ)
  integer, parameter  :: n_mode = (n_tor-1)/2 ! number of modes excluding 0
  real*8, intent(out) :: HZ(n_tor), dHZ(n_tor)
#endif
  real*8, intent(in)  :: phi
#ifdef UNIT_TESTS
  integer :: n_tor_loc,n_period_loc,n_mode
#endif
  integer    :: i
  real*8     :: phase
  complex*16 :: H_complex

  HZ(1) = 1.d0
  dHZ(1) = 0.d0

#ifdef UNIT_TESTS
  n_tor_loc    = n_tor;    if(present(n_tor_in)) n_tor_loc       = n_tor_in
  n_period_loc = n_period; if(present(n_period_in)) n_period_loc = n_period_in
  n_mode       = (n_tor_loc-1)/2
#endif
  
  do i=1,n_mode
#ifdef UNIT_TESTS
    phase      = real(n_period_loc*i,8)*phi
#else
    phase      = real(n_period*i,8)*phi
#endif
    H_complex  = exp(cmplx(0.d0,1.d0)*phase)
    HZ(2*i)    = real(H_complex)
    HZ(2*i+1)  = aimag(H_complex)
#ifdef UNIT_TESTS
    dHZ(2*i)   = HZ(2*i+1)*(-n_period_loc*i)
    dHZ(2*i+1) = HZ(2*i)  *( n_period_loc*i)
#else
    dHZ(2*i)   = HZ(2*i+1)*(-n_period*i)
    dHZ(2*i+1) = HZ(2*i)  *( n_period*i)
#endif
  end do

end subroutine sincosperiod_moivre

pure subroutine sincosperiod_moivre_ncoord(phi,HZ_coord,dHZ_coord)
  integer, parameter  :: n_mode = (n_coord_tor-1)/2 ! number of modes excluding 0
  real*8, intent(out) :: HZ_coord(n_coord_tor), dHZ_coord(n_coord_tor)
  real*8, intent(in)  :: phi
  integer    :: i
  real*8     :: phase
  complex*16 :: H_complex

  HZ_coord(1) = 1.d0
  dHZ_coord(1) = 0.d0

  do i=1,n_mode
    phase            = real(n_coord_period*i,8)*-1*phi  ! note -1 factor because the basis of physics and coordinates have opposite signs
    H_complex        = exp(cmplx(0.d0,1.d0)*phase)
    HZ_coord(2*i)    = real(H_complex)
    HZ_coord(2*i+1)  = aimag(H_complex)
    dHZ_coord(2*i)   = HZ_coord(2*i+1)*(-n_coord_period*i)
    dHZ_coord(2*i+1) = HZ_coord(2*i)  *( n_coord_period*i)
  end do
end subroutine sincosperiod_moivre_ncoord

pure subroutine moivre(ar,ai,br,bi,or,oi)
  real*8, intent(in)  :: ar, ai !< real and imag part of e^(i x)
  real*8, intent(in)  :: br, bi !< real and imag part of e^(i y)
  real*8, intent(out) :: or, oi !< real and imag part of e^(i (x+y))
  or = ar*br - ai*bi
  oi = ai*br + ar*bi
end subroutine moivre

! Apply De Moivre formula to calculate mode*phi
! Assumes that mode is of the form [0 1 1 2 2 3 3 4 4] ([0 4 4 8 8 12 12])
! This is roughly 3-4 times faster in my tests than just calculating the sines
! and cosines (even when that is vectorized).
#ifdef UNIT_TESTS 
pure subroutine mode_moivre(phi,HZ,n_tor_in,n_period_in)
  integer,intent(in),optional :: n_tor_in,n_period_in
  real*8, intent(out) :: HZ(:)
#else
pure subroutine mode_moivre(phi,HZ)
  integer, parameter  :: n_mode = (n_tor-1)/2 ! number of modes excluding 0
  real*8, intent(out) :: HZ(n_tor)
#endif
  real*8, intent(in)  :: phi
#ifdef UNIT_TESTS
  integer :: n_tor_loc,n_period_loc,n_mode
#endif
  real*8     :: phase
  complex*16 :: H_complex
  integer    :: i

  HZ(1) = 1.d0

#ifdef UNIT_TESTS
  n_tor_loc    = n_tor;    if(present(n_tor_in)) n_tor_loc       = n_tor_in
  n_period_loc = n_period; if(present(n_period_in)) n_period_loc = n_period_in
  n_mode       = (n_tor_loc-1)/2
#endif 

  do i=1, n_mode
#ifdef UNIT_TESTS
    phase     = real(n_period_loc*i,8)*phi
#else
    phase     = real(n_period*i,8)*phi
#endif
    H_complex = exp(cmplx(0.d0,1.d0)*phase)
    HZ(2*i)   = real(H_complex)
    HZ(2*i+1) = aimag(H_complex)
  enddo

!  if (n_mode .gt. 0) then
!    HZ(2) = cos(n_period*phi)
!    HZ(3) = sin(n_period*phi)  
!!DIR$ NOVECTOR
!    do i=2,n_mode
!      call moivre(HZ(2),HZ(3), HZ(2*i-2),HZ(2*i-1), HZ(2*i),HZ(2*i+1))
!    end do
!  end if
end subroutine mode_moivre



!> subroutine calculates the interpolation within one element (i_elm) for a given position
!> (s,t) in the local coordinates
pure subroutine interp_2(node_list, element_list, i_elm, i_var, i_harm, s, t, P, P_s, P_t, P_st, P_ss, P_tt)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P, P_s, P_t, P_st, P_ss, P_tt

! --- Local variables
real*8 :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees), G_st(4,n_degrees), G_ss(4,n_degrees), G_tt(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G, G_s, G_t, G_st, G_ss, G_tt)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_st = 0.d0; P_ss = 0.d0; P_tt = 0.d0

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions

#ifdef fullmhd
    if (i_var == 710) then
      P    = P    + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
    elseif (i_var == 711) then
      P    = P    + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
    else
#endif
      P    = P    + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
#ifdef fullmhd
    endif
#endif
  end do
end do
end subroutine interp_2


!> subroutine calculates the interpolation within one element (i_elm) for a given position
!> (s,t) in the local coordinates, of the deltas instead of the values
pure subroutine interp_2_delta(node_list, element_list, i_elm, i_var, i_harm, s, t, P, P_s, P_t, P_st, P_ss, P_tt)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P, P_s, P_t, P_st, P_ss, P_tt

! --- Local variables
real*8 :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees), G_st(4,n_degrees), G_ss(4,n_degrees), G_tt(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G, G_s, G_t, G_st, G_ss, G_tt)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_st = 0.d0; P_ss = 0.d0; P_tt = 0.d0

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions
    P    = P    + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
    P_s  = P_s  + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
    P_t  = P_t  + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
    P_st = P_st + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
    P_ss = P_ss + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
    P_tt = P_tt + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
  end do
end do
end subroutine interp_2_delta

!> subroutine calculates the interpolation within one element (i_elm) for a given position
!> (s,t) in the local coordinates (limited to 1st order derivatives)
pure subroutine interp_1(node_list, element_list, i_elm, i_var, i_harm, s, t, P, P_s, P_t)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P, P_s, P_t

! --- Local variables
real*8 :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G,G_s,G_t)

P = 0.d0; P_s = 0.d0; P_t = 0.d0;

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions

#ifdef fullmhd
    if (i_var == 710) then
      P    = P    + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
    elseif (i_var == 711) then
      P    = P    + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
    else
#endif
      P    = P    + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
#ifdef fullmhd
    endif
#endif
  end do
end do
end subroutine interp_1

!> subroutine calculates the interpolation within one element (i_elm) for a given position
!> (s,t) in the local coordinates, of the deltas instead of the values (limited to 1st order derivatives)
pure subroutine interp_1_delta(node_list, element_list, i_elm, i_var, i_harm, s, t, P, P_s, P_t)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P, P_s, P_t

! --- Local variables
real*8 :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G,G_s,G_t)

P = 0.d0; P_s = 0.d0; P_t = 0.d0;

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions
    P    = P    + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
    P_s  = P_s  + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
    P_t  = P_t  + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
  end do
end do
end subroutine interp_1_delta

!> subroutine calculates the interpolation within one element (i_elm) for a given position
!> (s,t) in the local coordinates (limited to values)
pure subroutine interp_0_single_harmonic(node_list, element_list, i_elm, i_var, i_harm, s, t, P)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P

! --- Local variables
real*8 :: G(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G)

P = 0.d0;

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions

#ifdef fullmhd
    if (i_var == 710) then
      P    = P    + node_list%node(iv)%Fprof_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
    elseif (i_var == 711) then
      P    = P    + node_list%node(iv)%psi_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
    else
#endif
      P    = P    + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
#ifdef fullmhd
    endif
#endif
  end do
end do
end subroutine interp_0_single_harmonic

!> subroutine calculates the interpolation within one element (i_elm) for a given position
!> (s,t) in the local coordinates, of the deltas instead of the values (limited to value)
pure subroutine interp_0_single_harmonic_delta(node_list, element_list, i_elm, i_var, i_harm, s, t, P)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P

! --- Local variables
real*8 :: G(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G)

P = 0.d0;

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions
    P    = P    + node_list%node(iv)%deltas(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
  end do
end do
end subroutine interp_0_single_harmonic_delta

!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t)
pure subroutine interp_0(node_list, element_list, i_elm, i_v, n_v, s, t, phi, P)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8,                   intent(out) :: P(n_v)

real*8  :: H(4,n_degrees), ss, mode
integer :: kv, iv, kf, m, i, i_harm, i_tor

call basisfunctions(s,t,H)

P = 0.d0

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions
    ss  = element_list%element(i_elm)%size(kv,kf)
    do i = 1, n_v
      P(i)    = P(i)   + node_list%node(iv)%values(1,kf,i_v(i)) * ss * H(kv,kf)
      do i_tor = 1, (n_tor-1)/2
        i_harm = 2*i_tor
        mode = i_tor * n_period
        P(i)    = P(i)   + node_list%node(iv)%values(i_harm,kf,i_v(i))   * ss * H(kv,kf)   * cos(mode*phi)
        P(i)    = P(i)   + node_list%node(iv)%values(i_harm+1,kf,i_v(i)) * ss * H(kv,kf)   * sin(mode*phi)
      end do
    end do
  end do
end do
end subroutine interp_0



!> This subroutine interpolates some variables at a specific position within one element at a given position (s,t)
!> of the deltas, not the values
pure subroutine interp_0_delta(node_list, element_list, i_elm, i_v, n_v, s, t, phi, P)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: n_v, i_v(n_v)
real*8,                   intent(in)  :: s, t, phi
real*8,                   intent(out) :: P(n_v)

real*8  :: H(4,n_degrees), ss, mode
integer :: kv, iv, kf, m, i, i_harm, i_tor

call basisfunctions(s,t,H)

P = 0.d0

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions
    ss  = element_list%element(i_elm)%size(kv,kf)
    do i = 1, n_v
      P(i)    = P(i)   + node_list%node(iv)%deltas(1,kf,i_v(i)) * ss * H(kv,kf)
      do i_tor = 1, (n_tor-1)/2
        i_harm = 2*i_tor
        mode = i_tor * n_period
        P(i)    = P(i)   + node_list%node(iv)%deltas(i_harm,kf,i_v(i))   * ss * H(kv,kf)   * cos(mode*phi)
        P(i)    = P(i)   + node_list%node(iv)%deltas(i_harm+1,kf,i_v(i)) * ss * H(kv,kf)   * sin(mode*phi)
      end do
    end do
  end do
end do
end subroutine interp_0_delta




!> Calculates the interpolation within one element (i_elm) for a given position (s,t) in local coordinates
pure subroutine interp_RZ_0(node_list,element_list,i_elm,s,t,R,Z)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
real*8,                   intent(in)  :: s,t
real*8,                   intent(out) :: R, Z

! --- Local variables
real*8  :: G(4,n_degrees)
real*8  :: xx1, xx2, ss
integer :: kv, iv, kf

call basisfunctions(s,t,G)

R = 0.d0; Z = 0.d0

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! basis functions
    xx1 = node_list%node(iv)%x(1,kf,1)
    xx2 = node_list%node(iv)%x(1,kf,2)
    ss  = element_list%element(i_elm)%size(kv,kf)
    
    R    = R    + xx1 * ss * G(kv,kf)
    Z    = Z    + xx2 * ss * G(kv,kf)
  end do
end do

end subroutine interp_RZ_0



!> This subroutine interpolates space a specific position within one element at a given position (s,t)
pure subroutine interp_RZ_1(node_list, element_list, i_elm, s, t, R, R_s, R_t, Z, Z_s, Z_t)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
real*8,                   intent(in)  :: s, t
real*8,                   intent(out) :: R, R_s, R_t, Z, Z_s, Z_t

! --- Local variables
real*8  :: H(n_degrees,4), H_s(n_degrees,4), H_t(n_degrees,4)
integer :: kv, iv
real*8  :: xR(n_degrees,n_vertex_max), xZ(n_degrees,n_vertex_max)
real*8  :: sizes(n_degrees)

call basisfunctions_T(s,t,H,H_s,H_t)

! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)
  xR(:,kv) = node_list%node(iv)%x(1,:,1) * sizes(:)
  xZ(:,kv) = node_list%node(iv)%x(1,:,2) * sizes(:)
end do

R   = sum(xR*H)
R_s = sum(xR*H_s)
R_t = sum(xR*H_t)
Z   = sum(xZ*H)
Z_s = sum(xZ*H_s)
Z_t = sum(xZ*H_t)

end subroutine interp_RZ_1



!> Calculates the interpolation within one element (i_elm) for a given position (s,t) in local coordinates
pure subroutine interp_RZ_2(node_list,element_list,i_elm,s,t,R,R_s,R_t,R_st,R_ss,R_tt,Z,Z_s,Z_t,Z_st,Z_ss,Z_tt)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
real*8,                   intent(in)  :: s,t
real*8,                   intent(out) :: R, R_s, R_t, R_st, R_ss, R_tt, Z, Z_s, Z_t, Z_st, Z_ss, Z_tt

! --- Local variables
real*8  :: H(n_degrees,4), H_s(n_degrees,4), H_t(n_degrees,4), H_st(n_degrees,4), H_ss(n_degrees,4), H_tt(n_degrees,4)
integer :: kv, iv
real*8  :: xR(n_degrees,n_vertex_max), xZ(n_degrees,n_vertex_max)
real*8  :: sizes(n_degrees)

call basisfunctions_T(s,t,H,H_s,H_t,H_st,H_ss,H_tt)

! Preload values and premultiply with sizes(:,kv)
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)
  sizes(:) = element_list%element(i_elm)%size(kv,:)
  xR(:,kv) = node_list%node(iv)%x(1,:,1) * sizes(:)
  xZ(:,kv) = node_list%node(iv)%x(1,:,2) * sizes(:)
end do

R    = sum(xR*H)
R_s  = sum(xR*H_s)
R_t  = sum(xR*H_t)
R_st = sum(xR*H_st)
R_ss = sum(xR*H_ss)
R_tt = sum(xR*H_tt)
Z    = sum(xZ*H)
Z_s  = sum(xZ*H_s)
Z_t  = sum(xZ*H_t)
Z_st = sum(xZ*H_st)
Z_ss = sum(xZ*H_ss)
Z_tt = sum(xZ*H_tt)

end subroutine interp_RZ_2


!> subroutine calculates the interpolation within one element (i_elm) for a given position
!> (s,t) in the local coordinates, works for any variable, but also for R and Z, (using i_var = -1 or -2 respectively)
pure subroutine interp_PRZ_combined(node_list, element_list, i_elm, i_var, i_harm, s, t, P, P_s, P_t, P_st, P_ss, P_tt)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P, P_s, P_t, P_st, P_ss, P_tt

! --- Local variables
real*8 :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees), G_st(4,n_degrees), G_ss(4,n_degrees), G_tt(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G, G_s, G_t, G_st, G_ss, G_tt)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_st = 0.d0; P_ss = 0.d0; P_tt = 0.d0

! for R (i_var=-1) and Z (i_var=-2)
if (i_var .lt. 0) then
  do kv = 1,n_vertex_max  ! 4 vertices
    iv = element_list%element(i_elm)%vertex(kv)  ! the node number
    do kf = 1, n_degrees       ! basis functions
      P    = P    + node_list%node(iv)%x(1,kf,-i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%x(1,kf,-i_var) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%x(1,kf,-i_var) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%x(1,kf,-i_var) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%x(1,kf,-i_var) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%x(1,kf,-i_var) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
    end do
  end do
! any other variable
else
  do kv = 1,n_vertex_max  ! 4 vertices
    iv = element_list%element(i_elm)%vertex(kv)  ! the node number
    do kf = 1, n_degrees       ! basis functions
      P    = P    + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%values(i_harm,kf,i_var) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
    end do
  end do
endif
end subroutine interp_PRZ_combined


! Interpolate the R, Z, phi location as a function of s, t, p
pure subroutine interp_RZP_0(node_list,element_list,i_elm,s,t,phi,R,Z)
  type (type_node_list),    intent(in)  :: node_list
  type (type_element_list), intent(in)  :: element_list
  integer,                  intent(in)  :: i_elm
  real*8,                   intent(in)  :: s,t,phi
  real*8,                   intent(out) :: R, Z
  
  ! --- Local variables
  real*8  :: G(4,n_degrees)
  real*8  :: HZ_coord(n_coord_tor)
  real*8  :: xx1, xx2, ss
  integer :: kv, iv, kf, i_tor
  
  ! Get toroidal and poloidal basis functions for s, t, phi
  call basisfunctions(s,t,G)
  HZ_coord(1)   = 1.d0
  do i_tor=1,(n_coord_tor-1)/2
    HZ_coord(2*i_tor)      =  cos(mode_coord(2*i_tor)  *phi)
    HZ_coord(2*i_tor+1)    = -sin(mode_coord(2*i_tor+1)*phi)
  enddo

  R = 0.d0; Z = 0.d0
  do kv = 1,n_vertex_max  ! 4 vertices
    iv = element_list%element(i_elm)%vertex(kv)  ! the node number
    do kf = 1, n_degrees       ! 4 basis functions
      do i_tor=1, n_coord_tor
        xx1 = node_list%node(iv)%x(i_tor,kf,1)
        xx2 = node_list%node(iv)%x(i_tor,kf,2)
        ss  = element_list%element(i_elm)%size(kv,kf)

        R    = R    + xx1 * ss * G(kv,kf) * HZ_coord(i_tor) 
        Z    = Z    + xx2 * ss * G(kv,kf) * HZ_coord(i_tor)
      end do
    end do
  end do
end subroutine interp_RZP_0


!> Calculates the interpolation within one element (i_elm) for a given position (s,t) in local coordinates
pure subroutine interp_RZP_1(node_list,element_list,i_elm,s,t,phi,R,R_s,R_t,R_p,Z,Z_s,Z_t,Z_p)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
real*8,                   intent(in)  :: s,t,phi
real*8,                   intent(out) :: R, R_s, R_t, R_p, Z, Z_s, Z_t, Z_p

! --- Local variables
real*8  :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees)
real*8  :: HZ_coord(n_coord_tor), HZ_coord_p(n_coord_tor), HZ_coord_pp(n_coord_tor)
real*8  :: xx1, xx2, ss
integer :: kv, iv, kf, i_tor

! Get toroidal and poloidal basis functions in (s, t, phi)
call basisfunctions(s,t,G,G_s,G_t)
HZ_coord(1)      = 1.d0
HZ_coord_p(1)    = 0.d0
do i_tor=1,(n_coord_tor-1)/2
  HZ_coord(2*i_tor)        = + cos(mode_coord(2*i_tor)  *phi)
  HZ_coord_p(2*i_tor)      = - float(mode_coord(2*i_tor))      * sin(mode_coord(2*i_tor)  *phi)
  HZ_coord(2*i_tor+1)      = - sin(mode_coord(2*i_tor+1)*phi)
  HZ_coord_p(2*i_tor+1)    = - float(mode_coord(2*i_tor+1))    * cos(mode_coord(2*i_tor+1)*phi)
enddo

R = 0.d0; R_s = 0.d0; R_t = 0.d0; R_p = 0.d0
Z = 0.d0; Z_s = 0.d0; Z_t = 0.d0; Z_p = 0.d0

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! 4 basis functions
    do i_tor=1, n_coord_tor
      xx1 = node_list%node(iv)%x(i_tor,kf,1)
      xx2 = node_list%node(iv)%x(i_tor,kf,2)
      ss  = element_list%element(i_elm)%size(kv,kf)
      
      R    = R    + xx1 * ss * G(kv,kf)    * HZ_coord(i_tor)
      R_s  = R_s  + xx1 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
      R_t  = R_t  + xx1 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
      R_p  = R_p  + xx1 * ss * G(kv,kf)    * HZ_coord_p(i_tor)

      Z    = Z    + xx2 * ss * G(kv,kf)    * HZ_coord(i_tor)
      Z_s  = Z_s  + xx2 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
      Z_t  = Z_t  + xx2 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
      Z_p  = Z_p  + xx2 * ss * G(kv,kf)    * HZ_coord_p(i_tor)
    end do
  end do
end do
end subroutine interp_RZP_1

!> Calculates the interpolation within one element (i_elm) for a given position (s,t) in local coordinates
pure subroutine interp_RZP_2(node_list,element_list,i_elm,s,t,phi,   &
                        R,R_s,R_t,R_p,R_st,R_ss,R_tt,R_sp, R_tp, R_pp,   &
                        Z,Z_s,Z_t,Z_p,Z_st,Z_ss,Z_tt,Z_sp,Z_tp,Z_pp)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
real*8,                   intent(in)  :: s,t,phi
real*8,                   intent(out) :: R, R_s, R_t, R_p, R_st, R_ss, R_tt, R_sp, R_tp, R_pp
real*8,                   intent(out) :: Z, Z_s, Z_t, Z_p, Z_st, Z_ss, Z_tt, Z_sp, Z_tp, Z_pp

! --- Local variables
real*8  :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees), G_st(4,n_degrees), G_ss(4,n_degrees), G_tt(4,n_degrees)
real*8  :: HZ_coord(n_coord_tor), HZ_coord_p(n_coord_tor), HZ_coord_pp(n_coord_tor)
real*8  :: xx1, xx2, ss
integer :: kv, iv, kf, i_tor

! Get toroidal and poloidal basis functions in (s, t, phi)
call basisfunctions(s,t,G,G_s,G_t,G_st,G_ss,G_tt)
HZ_coord(1)      = 1.d0
HZ_coord_p(1)    = 0.d0
HZ_coord_pp(1)   = 0.d0
do i_tor=1,(n_coord_tor-1)/2
  HZ_coord(2*i_tor)        = + cos(mode_coord(2*i_tor)  *phi)
  HZ_coord_p(2*i_tor)      = - float(mode_coord(2*i_tor))      * sin(mode_coord(2*i_tor)  *phi)
  HZ_coord_pp(2*i_tor)     = - float(mode_coord(2*i_tor))**2   * cos(mode_coord(2*i_tor)  *phi)
  HZ_coord(2*i_tor+1)      = - sin(mode_coord(2*i_tor+1)*phi)
  HZ_coord_p(2*i_tor+1)    = - float(mode_coord(2*i_tor+1))    * cos(mode_coord(2*i_tor+1)*phi)
  HZ_coord_pp(2*i_tor+1)   = + float(mode_coord(2*i_tor+1))**2 * sin(mode_coord(2*i_tor+1)*phi)
enddo

R = 0.d0; R_s = 0.d0; R_t = 0.d0; R_p = 0.d0; R_st = 0.d0; R_ss = 0.d0; R_tt = 0.d0; R_sp = 0.0; R_tp=0.0; R_pp = 0.0;
Z = 0.d0; Z_s = 0.d0; Z_t = 0.d0; Z_p = 0.d0; Z_st = 0.d0; Z_ss = 0.d0; Z_tt = 0.d0; Z_sp = 0.0; Z_tp=0.0; Z_pp = 0.0;

do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! 4 basis functions
    do i_tor=1, n_coord_tor
      xx1 = node_list%node(iv)%x(i_tor,kf,1)
      xx2 = node_list%node(iv)%x(i_tor,kf,2)
      ss  = element_list%element(i_elm)%size(kv,kf)
      
      R    = R    + xx1 * ss * G(kv,kf)    * HZ_coord(i_tor)
      R_s  = R_s  + xx1 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
      R_t  = R_t  + xx1 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
      R_p  = R_p  + xx1 * ss * G(kv,kf)    * HZ_coord_p(i_tor)
      R_st = R_st + xx1 * ss * G_st(kv,kf) * HZ_coord(i_tor)
      R_ss = R_ss + xx1 * ss * G_ss(kv,kf) * HZ_coord(i_tor)
      R_tt = R_tt + xx1 * ss * G_tt(kv,kf) * HZ_coord(i_tor)
      R_sp = R_sp + xx1 * ss * G_s(kv,kf)  * HZ_coord_p(i_tor)
      R_tp = R_tp + xx1 * ss * G_t(kv,kf)  * HZ_coord_p(i_tor)
      R_pp = R_pp + xx1 * ss * G(kv,kf)    * HZ_coord_pp(i_tor)

      Z    = Z    + xx2 * ss * G(kv,kf)    * HZ_coord(i_tor)
      Z_s  = Z_s  + xx2 * ss * G_s(kv,kf)  * HZ_coord(i_tor)
      Z_t  = Z_t  + xx2 * ss * G_t(kv,kf)  * HZ_coord(i_tor)
      Z_p  = Z_p  + xx2 * ss * G(kv,kf)    * HZ_coord_p(i_tor)
      Z_st = Z_st + xx2 * ss * G_st(kv,kf) * HZ_coord(i_tor)
      Z_ss = Z_ss + xx2 * ss * G_ss(kv,kf) * HZ_coord(i_tor)
      Z_tt = Z_tt + xx2 * ss * G_tt(kv,kf) * HZ_coord(i_tor)
      Z_sp = Z_sp + xx2 * ss * G_s(kv,kf)  * HZ_coord_p(i_tor)
      Z_tp = Z_tp + xx2 * ss * G_t(kv,kf)  * HZ_coord_p(i_tor)
      Z_pp = Z_pp + xx2 * ss * G(kv,kf)    * HZ_coord_pp(i_tor)
    end do
  end do
end do
end subroutine interp_RZP_2

!> subroutine calculates the interpolation of GVEC equilibrium conditions within one element (i_elm) for a given (s, t).
!! The interpolated variable is controlled by the value of i_var: 
!!   1: magnetic field
!!   2: plasma current
!!   3: pressure
!!   4: radial coordinate (in GVEC, this is the square root of the normalised toroidal flux)
!!   5: scalar potential for vacuum magnetic field
!!   6: vacuum magnetic field imported with GVEC
pure subroutine interp_gvec(node_list, element_list, i_elm, i_var, i_dim, i_harm, s, t, P, P_s, P_t, P_st, P_ss, P_tt)
type (type_node_list),    intent(in)  :: node_list
type (type_element_list), intent(in)  :: element_list
integer,                  intent(in)  :: i_elm
integer,                  intent(in)  :: i_var         ! i_var: 1: B, 2: J, 3: p
integer,                  intent(in)  :: i_dim         ! i_dim: 1: R, 2: Z, 3:, phi
integer,                  intent(in)  :: i_harm
real*8,                   intent(in)  :: s
real*8,                   intent(in)  :: t
real*8,                   intent(out) :: P, P_s, P_t, P_st, P_ss, P_tt

! --- Local variables
real*8 :: G(4,n_degrees), G_s(4,n_degrees), G_t(4,n_degrees), G_st(4,n_degrees), G_ss(4,n_degrees), G_tt(4,n_degrees)
integer :: kv, iv, kf 

call basisfunctions(s,t,G, G_s, G_t, G_st, G_ss, G_tt)

P = 0.d0; P_s = 0.d0; P_t = 0.d0; P_st = 0.d0; P_ss = 0.d0; P_tt = 0.d0

#if STELLARATOR_MODEL
do kv = 1,n_vertex_max  ! 4 vertices
  iv = element_list%element(i_elm)%vertex(kv)  ! the node number
  do kf = 1, n_degrees       ! 4 basis functions
    if (i_var == 4) then 
      ! The equilibrium is a scalar, axisymmetric profile, so i_dim and i_harm have no influence on the results
      P    = P    + node_list%node(iv)%r_tor_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%r_tor_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%r_tor_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%r_tor_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%r_tor_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%r_tor_eq(kf) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
#if JOREK_MODEL == 180
    ! i_var=1: Full equilibrium B field
    else if (i_var == 1) then
      P    = P    + node_list%node(iv)%b_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%b_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%b_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%b_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%b_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%b_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
    else if (i_var == 2) then
      P    = P    + node_list%node(iv)%j_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%j_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%j_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%j_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%j_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%j_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
    else if (i_var == 3) then                
      ! The equilibrium is a scalar, axisymmetric profile, so i_dim and i_harm have no influence on the results
      P    = P    + node_list%node(iv)%pressure(kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%pressure(kf) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%pressure(kf) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%pressure(kf) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%pressure(kf) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%pressure(kf) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
#endif /*JOREK_MODEL == 180*/
    else if (i_var == 5) then    
#if !defined(USE_DOMM) && !defined(USE_EXT_FIELD)
      ! The equilibrium is a scalar, axisymmetric profile, so i_dim and i_harm have no influence on the results
      P    = P    + node_list%node(iv)%chi_correction(i_harm,kf) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%chi_correction(i_harm,kf) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%chi_correction(i_harm,kf) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%chi_correction(i_harm,kf) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%chi_correction(i_harm,kf) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%chi_correction(i_harm,kf) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
#endif /*!defined(USE_DOMM) && !defined(USE_EXT_FIELD)*/
    else if (i_var == 6) then
#if !defined(USE_DOMM) && defined(USE_EXT_FIELD)
      P    = P    + node_list%node(iv)%b_vac_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G(kv,kf)
      P_s  = P_s  + node_list%node(iv)%b_vac_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_s(kv,kf)
      P_t  = P_t  + node_list%node(iv)%b_vac_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_t(kv,kf)
      P_st = P_st + node_list%node(iv)%b_vac_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_st(kv,kf)
      P_ss = P_ss + node_list%node(iv)%b_vac_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_ss(kv,kf)
      P_tt = P_tt + node_list%node(iv)%b_vac_field(i_harm,kf,i_dim) * element_list%element(i_elm)%size(kv,kf) * G_tt(kv,kf)
#endif /*!defined(USE_DOMM) && defined(USE_EXT_FIELD)*/
    endif
  end do
end do
#endif /*STELLARATOR_MODEL*/

end subroutine interp_gvec

end module mod_interp
