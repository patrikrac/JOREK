!> Module for creating an RTree of elements and searching in this tree
module mod_element_rtree
use iso_c_binding
use data_structure
implicit none
public populate_element_rtree, nearby_elements, elements_containing_point, rtree_initialized, tree_slices

logical :: rtree_initialized = .false.
integer, parameter :: tree_slices = n_plane

#if STELLARATOR_MODEL
integer, parameter :: ND = 3
#else
integer, parameter :: ND = 2
#endif

interface
  subroutine RZ_minmax(node_list, element_list, i_elm, Rmin, Rmax, Zmin, Zmax, i_plane_query_in)
    use data_structure, only: type_node_list, type_element_list
    implicit none
    type(type_node_list), intent(in)    :: node_list
    type(type_element_list), intent(in) :: element_list
    integer, intent(in)                 :: i_elm
    real*8 , intent(out)                :: Rmin, Rmax, Zmin, Zmax
    integer, intent(in), optional       :: i_plane_query_in
  end subroutine RZ_minmax
  !> Name is element_rtree to match filename `element_rtree.cpp`.
  !> `void PopulateTree(int nelm, double minx[], double miny[], double maxx[], double maxy[])`
  subroutine element_rtree(n, tree_slices, min, max) bind(C,name="PopulateTree")
    import C_DOUBLE, C_INT
    integer(C_INT), intent(in), value :: n, tree_slices
    real(C_DOUBLE), intent(in), dimension(*)  :: min, max
  end subroutine element_rtree
  !> `int NumElementsInRect(double minx, double miny, double maxx, double maxy)`
  function num_elements_in_rect(min, max) bind(C,name='NumElementsInRect')
    import C_DOUBLE, C_INT
    real(C_DOUBLE), intent(in), dimension(*)         :: min, max
    integer(C_INT) :: num_elements_in_rect
  end function num_elements_in_rect
  !> `int ElementsInRect(double minx, double miny, double maxx, double maxy, int *ielm)`
  function elements_in_rect(min, max, ielm) bind(C,name='ElementsInRect')
    import C_DOUBLE, C_INT
    real(C_DOUBLE), intent(in), dimension(*)          :: min, max
    integer(C_INT), intent(inout), dimension(*) :: ielm
    integer(C_INT) :: elements_in_rect
  end function elements_in_rect
end interface

interface populate_element_rtree
#if STELLARATOR_MODEL
  ! If STELLARATOR_MODEL is defined during compilation, use the 3D version.
  module procedure populate_element_rtree_3D
#else
  ! Otherwise, default to the 2D version.
  module procedure populate_element_rtree_2D
#endif
end interface

contains


!> Populate the RTree with the squares containing elements
!> x=R, y=Z
!>
!> Only one call to this routine can be made simultaneously! It uses a static
!> data structure under the hood.
subroutine populate_element_rtree_2D(node_list, element_list)
  type(type_node_list), intent(in)    :: node_list
  type(type_element_list), intent(in) :: element_list
  real(C_DOUBLE), dimension(:), allocatable :: min_bb, max_bb
  real*8 :: rmin, rmax, zmin, zmax
  integer :: i, n
  n = element_list%n_elements
  allocate(min_bb(n*ND), max_bb(n*ND))
  do i=1,n
    call RZ_minmax(node_list, element_list, i, rmin, rmax, zmin, zmax)
    max_bb((i - 1) * ND + 1) = real(rmax, kind=C_DOUBLE)
    max_bb((i - 1) * ND + 2) = real(zmax, kind=C_DOUBLE)

    min_bb((i - 1) * ND + 1) = real(rmin, kind=C_DOUBLE)
    min_bb((i - 1) * ND + 2) = real(zmin, kind=C_DOUBLE)
  end do
  write(*,*) "Initializing RTree"
  ! this cleans out the tree before insertion
  call element_rtree(int(n, C_INT), int(1, C_INT), min_bb, max_bb)
  rtree_initialized = .true.
end subroutine populate_element_rtree_2D

!> Only one call to this routine can be made simultaneously! It uses a static
subroutine populate_element_rtree_3D(node_list, element_list)
  use constants, only: PI

  type(type_node_list),    intent(in) :: node_list
  type(type_element_list), intent(in) :: element_list

  real(C_DOUBLE), dimension(:), allocatable :: min_bb, max_bb

  real*8  :: rmin, rmax, zmin, zmax, rmin_old, rmax_old, zmin_old, zmax_old
  integer :: i, n, mp, mp_old, index
  real*8  :: arbitrary_HZ_coord(n_coord_tor,tree_slices)

  n = element_list%n_elements

  arbitrary_HZ_coord = get_arbitrary_HZ_coord(tree_slices)

  allocate(min_bb(n*tree_slices*ND), max_bb(n*tree_slices*ND))

  do i=1,n
    call RZP_minmax(node_list, element_list, i, rmin_old, rmax_old, zmin_old, zmax_old, arbitrary_HZ_coord, 1)
    do mp=2,tree_slices+1
      call RZP_minmax(node_list, element_list, i, rmin, rmax, zmin, zmax, arbitrary_HZ_coord, mp)

      index = ((i - 1) * tree_slices + (mp - 2)) * ND
  
      max_bb(index + 1) = real(max(rmax, rmax_old), kind=C_DOUBLE)
      max_bb(index + 2) = real(max(zmax, zmax_old), kind=C_DOUBLE)
      max_bb(index + 3) = real(real(mp - 1,8) * 2.d0 * PI / real(tree_slices * n_coord_period,8), kind=C_DOUBLE)

      min_bb(index + 1) = real(min(rmin, rmin_old), kind=C_DOUBLE)
      min_bb(index + 2) = real(min(zmin, zmin_old), kind=C_DOUBLE)
      min_bb(index + 3) = real(real(mp - 2,8) * 2.d0 * PI / real(tree_slices * n_coord_period,8), kind=C_DOUBLE)

      rmin_old = rmin ; rmax_old = rmax ; zmin_old = zmin ; zmax_old = zmax
    end do
  enddo
  write(*,*) "Initializing RTree"
  ! this cleans out the tree before insertion
  call element_rtree(int(n, C_INT), int(tree_slices, C_INT), min_bb, max_bb)
  rtree_initialized = .true.
end subroutine populate_element_rtree_3D

!> Find probable neighbours of element i and return their indices.
!> This is done by taking the bounding box of an element and expanding
!> it slightly (10^-6 in absolute value) and returning all elements in
!> this box, except element i
subroutine nearby_elements(node_list, element_list, i_elm, i_nearby)
  use constants, only: PI
  type(type_node_list), intent(in)    :: node_list
  type(type_element_list), intent(in) :: element_list
  integer, intent(in)                 :: i_elm
  integer, dimension(:), allocatable, intent(out) :: i_nearby
  integer(C_int), dimension(:), allocatable       :: i_nearby_C

  real*8, dimension(n_vertex_max, 2) :: vertices
  real(C_DOUBLE), dimension(ND) :: min_bb, max_bb
  integer(C_INT) :: num_elements
  integer :: iv

  do iv=1,n_vertex_max
     vertices(iv,:) = get_vertex_pos_in_rtree_plane(node_list%node(element_list%element(i_elm)%vertex(iv))%x(1:n_coord_tor,1,1:2))
  enddo

  min_bb (1) = real(minval(vertices(:,1)) - 1d-6, kind=C_DOUBLE)
  max_bb (1) = real(maxval(vertices(:,1)) + 1d-6, kind=C_DOUBLE)

  min_bb (2) = real(minval(vertices(:,2)) - 1d-6, kind=C_DOUBLE)
  max_bb (2) = real(maxval(vertices(:,2)) + 1d-6, kind=C_DOUBLE)

#if STELLARATOR_MODEL
  min_bb (3) = real(0.d0, kind=C_DOUBLE)
  max_bb (3) = real(0.d0, kind=C_DOUBLE)
#endif

  num_elements = int(num_elements_in_rect(min_bb, max_bb))
  allocate(i_nearby(num_elements),i_nearby_C(num_elements))
  num_elements = int(elements_in_rect(min_bb, max_bb, i_nearby_C))
  i_nearby = int(i_nearby_C(1:num_elements))
end subroutine nearby_elements

!> Find elements that could probably contain this point.
subroutine elements_containing_point(R, Z, phi, i_elms)
  real*8, intent(in)                              :: R, Z, phi
  integer, dimension(:), allocatable, intent(out) :: i_elms

  real(C_DOUBLE), dimension(ND) :: min_bb, max_bb
  integer(C_INT) :: num_elements
  integer(C_int), dimension(:), allocatable :: i_nearby_C

  if (R .ne. R .or. Z .ne. Z .or. phi .ne. phi) then
    write(*,*) "Warning: NaN supplied for R or Z in elements_containing_point, returning 0 elements"
    allocate(i_elms(0))
    return
  end if

  if (.not. rtree_initialized) then
    write(*,*) "Warning: RTree not initialised, exiting"
    stop 11
  end if

  min_bb (1) = real(R, kind=C_DOUBLE)
  max_bb (1) = real(R, kind=C_DOUBLE)

  min_bb (2) = real(Z, kind=C_DOUBLE)
  max_bb (2) = real(Z, kind=C_DOUBLE)

#if STELLARATOR_MODEL
  min_bb (3) = real(phi, kind=C_DOUBLE)
  max_bb (3) = real(phi, kind=C_DOUBLE)
#endif

  ! This calls the search routine twice! To get around that either pass a large enough array alway
  ! or implement it as a mask/bitfield of the total number of elements.
  num_elements = int(num_elements_in_rect(min_bb, max_bb))
  allocate(i_nearby_C(num_elements), i_elms(num_elements))
  num_elements = int(elements_in_rect(min_bb, max_bb, i_nearby_C))
  i_elms = int(i_nearby_C(1:num_elements))
end subroutine elements_containing_point


pure function get_vertex_pos_in_rtree_plane(x) result(pos)
  use phys_module, only: i_plane_rtree
  use basis_at_gaussian, only: HZ_coord
  implicit none

  real*8, intent(in)    :: x(n_coord_tor, 2)
  real*8   :: pos(2)
  integer  :: i

  do i=1,2
    pos(i) = sum(x(1:n_coord_tor,i)*HZ_coord(1:n_coord_tor, i_plane_rtree))
  end do
end function get_vertex_pos_in_rtree_plane

!> calculates the basis functions at the Gaussian points
pure function get_arbitrary_HZ_coord(tree_slices_requested) result(arbitrary_HZ_coord)
  use constants, only: PI
  use phys_module, only: n_coord_tor, n_coord_period, mode_coord

  implicit none

  integer, intent(in) :: tree_slices_requested

  ! --- local variables
  integer :: k,i
  real*8  :: phi
  real*8 :: arbitrary_HZ_coord (n_coord_tor,tree_slices_requested)

  do k=1,tree_slices_requested

    phi = 2.d0*PI*float(k-1)/float(tree_slices_requested) / float(n_coord_period)

    arbitrary_HZ_coord(1,k)   = 1.d0

    do i=1,(n_coord_tor-1)/2
      arbitrary_HZ_coord(2*i,k)      = + cos(mode_coord(2*i)  *phi)
      arbitrary_HZ_coord(2*i+1,k)    = - sin(mode_coord(2*i+1)*phi)
    enddo
  enddo
end function get_arbitrary_HZ_coord


end module mod_element_rtree
