subroutine RZ_minmax(node_list,element_list,i_elm,Rmin,Rmax,Zmin,Zmax)
  use phys_module, only: i_plane_rtree
  use data_structure

  implicit none

  type (type_node_list), intent(in)    :: node_list
  type (type_element_list), intent(in) :: element_list
  integer, intent(in) :: i_elm

  real*8, intent(out) :: Rmin, Rmax, Zmin, Zmax

  call RZ_minmax_general(node_list,element_list,i_elm,0.d0,Rmin,Rmax,Zmin,Zmax)
end subroutine RZ_minmax

subroutine RZP_minmax(node_list,element_list,i_elm,phi,Rmin,Rmax,Zmin,Zmax)
  use data_structure
  use mod_element_rtree, only: tree_slices
  use mod_parameters, only: n_coord_tor

  implicit none

  type (type_node_list), intent(in)    :: node_list
  type (type_element_list), intent(in) :: element_list
  integer, intent(in) :: i_elm
  real*8, intent(out) :: Rmin, Rmax, Zmin, Zmax, phi

  call RZ_minmax_general(node_list,element_list,i_elm,phi,Rmin,Rmax,Zmin,Zmax)
end subroutine RZP_minmax

subroutine RZ_minmax_general (node_list,element_list,i_elm,phi,Rmin,Rmax,Zmin,Zmax)
use data_structure
use mod_newton_methods
use mod_parameters, only: n_order
use mod_element_rtree, only: tree_slices
use mod_parameters, only: n_coord_tor
use phys_module, only: mode_coord

implicit none

type (type_node_list), intent(in)    :: node_list
type (type_element_list), intent(in) :: element_list
integer, intent(in) :: i_elm
real*8, intent(in) :: phi

real*8, intent(out) :: Rmin, Rmax, Zmin, Zmax

real*8  :: HZ_coord(n_coord_tor)
real*8  :: psimin, psimax, psma, psmi, psmima, psim, psimr, psip, psipr
real*8  :: aa, bb, cc, det, r, dummy
real*8,external :: root
integer :: iv, n, im, n1, n2, i_tor
real*8  :: s,t,P,P_s,P_t,P_st,P_ss,P_tt
integer :: i, k

HZ_coord(1) = 1.0d0
do i=1,(n_coord_tor-1)/2
  HZ_coord(2*i)      =   cos(mode_coord(2*i)  *phi)
  HZ_coord(2*i+1)    = - sin(mode_coord(2*i+1)*phi)
enddo


! --- For n_order>3, we need to use Newton methods (not exactly true, should implement quartic root finder) 
! --- Could be important/faster for particles module!!!
if (n_order .ge. 5) then
  call find_variable_minmax(node_list,element_list,i_elm, -1, Rmin, Rmax)
  call find_variable_minmax(node_list,element_list,i_elm, -2, Zmin, Zmax)
  return
endif

! --- Continue for bi-cubic elements

do k=1,2

  psimin = 1d10
  psimax =-1d10

  do iv= 1, n_vertex_max

    im = mod(iv,n_vertex_max) + 1
    n1 = element_list%element(i_elm)%vertex(iv)
    n2 = element_list%element(i_elm)%vertex(im)
    PSIM = 0.d0; PSIMR = 0.d0; PSIP = 0.d0; PSIPR = 0.d0

    if (node_list%node(n1)%axis_node .and. node_list%node(n2)%axis_node) cycle

    if ((iv .eq. 1) .or. (iv .eq. 3)) THEN
      do i_tor=1,n_coord_tor
        PSIM  = PSIM  + node_list%node(n1)%x(i_tor,1,k)*element_list%element(i_elm)%size(iv,1)*HZ_coord(i_tor)
        PSIMR = PSIMR + node_list%node(n1)%x(i_tor,2,k)*element_list%element(i_elm)%size(iv,2)*HZ_coord(i_tor)*3.d0/2.d0
        PSIP  = PSIP  + node_list%node(n2)%x(i_tor,1,k)*element_list%element(i_elm)%size(im,1)*HZ_coord(i_tor)
        PSIPR = PSIPR - node_list%node(n2)%x(i_tor,2,k)*element_list%element(i_elm)%size(im,2)*HZ_coord(i_tor)*3.d0/2.d0
      end do
    elseif ((iv .eq. 2) .or. (iv .eq. 4)) then
      do i_tor=1,n_coord_tor
        PSIM  = PSIM  + node_list%node(n1)%x(i_tor,1,k)*element_list%element(i_elm)%size(iv,1)*HZ_coord(i_tor)
        PSIMR = PSIMR + node_list%node(n1)%x(i_tor,3,k)*element_list%element(i_elm)%size(iv,3)*HZ_coord(i_tor)*3.d0/2.d0
        PSIP  = PSIP  + node_list%node(n2)%x(i_tor,1,k)*element_list%element(i_elm)%size(im,1)*HZ_coord(i_tor)
        PSIPR = PSIPR - node_list%node(n2)%x(i_tor,3,k)*element_list%element(i_elm)%size(im,3)*HZ_coord(i_tor)*3.d0/2.d0
      end do
    endif

    if ((PSIM .eq. PSIP) .and. (PSIMR .eq. 0.d0) .and. (PSIPR .eq. 0.d0)) then

      psimin = min(psimin,PSIM)
      psimax = max(psimax,PSIM)

    else

      PSMA = MAX(PSIM,PSIP)
      PSMI = MIN(PSIM,PSIP)
      AA =  3.d0 * (PSIM + PSIMR - PSIP + PSIPR ) / 4.d0
      BB =  ( - PSIMR + PSIPR ) / 2.d0
      CC =  ( - 3.d0*PSIM - PSIMR + 3.d0*PSIP - PSIPR) / 4.d0
      DET = BB**2 - 4.d0*AA*CC
      IF (DET .GE. 0.d0) THEN
        R = ROOT(AA,BB,CC,DET,1.d0)
        IF (ABS(R) .GT. 1.d0) THEN
          R = ROOT(AA,BB,CC,DET,-1.d0)
        ENDIF
        IF (ABS(R) .LE. 1.d0) THEN
          CALL CUB1D(PSIM,PSIMR,PSIP,PSIPR,R,PSMIMA,DUMMY)
          psma = max(psma,psmima)
          psmi = min(psmi,psmima)
        ENDIF
      ENDIF
      psimin = min(psimin,psmi)
      psimax = max(psimax,psma)

    endif

  enddo

  if (k.eq.1) then
    Rmin = psimin
    Rmax = psimax
  else
    Zmin = psimin
    Zmax = psimax
  endif

enddo

return
end subroutine RZ_minmax_general
