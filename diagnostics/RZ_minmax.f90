subroutine RZ_minmax(node_list,element_list,i_elm,Rmin,Rmax,Zmin,Zmax)
  use phys_module, only: i_plane_rtree
  use basis_at_gaussian, only: HZ_coord
  use data_structure

  implicit none

  type (type_node_list), intent(in)    :: node_list
  type (type_element_list), intent(in) :: element_list
  integer, intent(in) :: i_elm

  real*8, intent(out) :: Rmin, Rmax, Zmin, Zmax

  call RZ_minmax_general(node_list,element_list,i_elm,Rmin,Rmax,Zmin,Zmax,HZ_coord,i_plane_rtree)
end subroutine RZ_minmax

subroutine RZP_minmax(node_list,element_list,i_elm,Rmin,Rmax,Zmin,Zmax,HZ_coord_used,i_slice_query)
  use data_structure
  use mod_element_rtree, only: tree_slices
  use mod_parameters, only: n_coord_tor

  implicit none

  type (type_node_list), intent(in)    :: node_list
  type (type_element_list), intent(in) :: element_list
  integer, intent(in) :: i_elm, i_slice_query
  real*8,  intent(in) :: HZ_coord_used(n_coord_tor,tree_slices)
  real*8, intent(out) :: Rmin, Rmax, Zmin, Zmax

  call RZ_minmax_general(node_list,element_list,i_elm,Rmin,Rmax,Zmin,Zmax,HZ_coord_used, mod(i_slice_query - 1, tree_slices) + 1)
end subroutine RZP_minmax

subroutine RZ_minmax_general (node_list,element_list,i_elm,Rmin,Rmax,Zmin,Zmax,HZ_coord_used,i_slice_query)
use data_structure
use mod_newton_methods
use mod_parameters, only: n_order
use mod_element_rtree, only: tree_slices
use mod_parameters, only: n_coord_tor

implicit none

type (type_node_list), intent(in)    :: node_list
type (type_element_list), intent(in) :: element_list
integer, intent(in) :: i_elm, i_slice_query
real*8,  intent(in) :: HZ_coord_used(n_coord_tor,tree_slices)              !< Basis functions of grid representation in toroidal direction

real*8, intent(out) :: Rmin, Rmax, Zmin, Zmax

real*8  :: psimin, psimax, psma, psmi, psmima, psim, psimr, psip, psipr
real*8  :: aa, bb, cc, det, r, dummy
real*8,external :: root
integer :: iv, n, im, n1, n2, i_tor
real*8  :: s,t,P,P_s,P_t,P_st,P_ss,P_tt
integer :: k

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
        PSIM  = PSIM  + node_list%node(n1)%x(i_tor,1,k)*element_list%element(i_elm)%size(iv,1)*HZ_coord_used(i_tor,i_slice_query)
        PSIMR = PSIMR + node_list%node(n1)%x(i_tor,2,k)*element_list%element(i_elm)%size(iv,2)*HZ_coord_used(i_tor,i_slice_query)*3.d0/2.d0
        PSIP  = PSIP  + node_list%node(n2)%x(i_tor,1,k)*element_list%element(i_elm)%size(im,1)*HZ_coord_used(i_tor,i_slice_query)
        PSIPR = PSIPR - node_list%node(n2)%x(i_tor,2,k)*element_list%element(i_elm)%size(im,2)*HZ_coord_used(i_tor,i_slice_query)*3.d0/2.d0
      end do
    elseif ((iv .eq. 2) .or. (iv .eq. 4)) then
      do i_tor=1,n_coord_tor
        PSIM  = PSIM  + node_list%node(n1)%x(i_tor,1,k)*element_list%element(i_elm)%size(iv,1)*HZ_coord_used(i_tor,i_slice_query)
        PSIMR = PSIMR + node_list%node(n1)%x(i_tor,3,k)*element_list%element(i_elm)%size(iv,3)*HZ_coord_used(i_tor,i_slice_query)*3.d0/2.d0
        PSIP  = PSIP  + node_list%node(n2)%x(i_tor,1,k)*element_list%element(i_elm)%size(im,1)*HZ_coord_used(i_tor,i_slice_query)
        PSIPR = PSIPR - node_list%node(n2)%x(i_tor,3,k)*element_list%element(i_elm)%size(im,3)*HZ_coord_used(i_tor,i_slice_query)*3.d0/2.d0
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
