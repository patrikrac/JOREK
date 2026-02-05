!> VTK tools for JOREK
module mod_vtk
  implicit none

  !> Represents the points and connectivity of a vtk grid (typically in 3D)
  !> To get a useful file you typically still need to add scalars
  type :: vtk_grid
    integer :: nsub !< Number of times to split each element in each dimension (i.e. 2**nsub per element)
    real*4,allocatable  :: xyz (:,:,:) !< positions of points in vtk file (3,n_points)
    integer,allocatable :: ien (:,:,:) !< connectivity in vtk file (n_corners,n_elements) (for arbitrary shapes)
  end type

  interface vtk_grid
    module procedure new_vtk_grid
  end interface vtk_grid

contains

  !> Create a new 2D VTK grid from node_list and element_list, splitting elements nsub times
  function new_vtk_grid(node_list, element_list, nsub) result(grid)
    use data_structure
    type(type_node_list), intent(in)    :: node_list
    type(type_element_list), intent(in) :: element_list
    integer, intent(in)                 :: nsub !< Number of subdivisions of each element
    type(vtk_grid)                      :: grid

    grid%nsub = nsub

    call prepare_vtk_grid(node_list, element_list, nsub, grid%xyz, grid%ien)
  end function new_vtk_grid

  !> Create a new 2D VTK grid from node_list and element_list, splitting elements nsub times
  !> This creates a grid with quadrilateral elements, uniformly spaced inside the finite elements
  subroutine prepare_vtk_grid(node_list,element_list,nsub,xyz,ien)
    use data_structure
    use mod_parameters,    only: n_plane
    use constants,         only: PI
    use mod_interp,        only: interp_RZP
    implicit none

    !> Input parameters
    type(type_node_list), intent(in)    :: node_list
    type(type_element_list), intent(in) :: element_list
    integer, intent(in)                 :: nsub !< Number of subdivisions of each element
    real*4, allocatable, intent(out)    :: xyz (:,:,:)
    integer, allocatable, intent(out)   :: ien (:,:,:)

    integer, allocatable  :: inode (:), ielm(:)

    integer :: nnos, nnoel, nel, i, l, m, mp
    real*8 :: toroidal_angle, s, t, R, Z

    nnos = nsub*nsub*element_list%n_elements
    allocate(xyz(n_plane,3,nnos))

    nnoel = 4
    nel   = (nsub-1)*(nsub-1)*element_list%n_elements
    allocate(ien(n_plane,nnoel,nel))

    allocate(inode(n_plane), ielm(n_plane))

    inode   = 0
    ielm    = 0
    xyz     = 0
    ien     = 0

    ! Create points for each element
    do mp=1,n_plane
      toroidal_angle = real((mp - 1),8) * 2.d0 * PI / n_plane / n_period ! 2*PI / 6
      do i=1,element_list%n_elements
        do l=1,nsub
          s = real(l-1,8)/real(nsub-1,8)
          do m=1,nsub
            t = real(m-1,8)/real(nsub-1,8)
            call interp_RZP(node_list,element_list,i,s,t,toroidal_angle,R,Z)
            inode(mp) = inode(mp)+1
            ! xyz(mp,1:3,inode) = real([R, Z, 0.d0], 4)
            xyz(mp,1,inode(mp)) = real(R,    4)
            xyz(mp,2,inode(mp)) = real(Z,    4)
            xyz(mp,3,inode(mp)) = real(0.d0, 4)
          enddo
        enddo

        ! Calculate the connectivity of each subelement
        do l=1,nsub-1
          do m=1,nsub-1
            ielm(mp)        = ielm(mp)+1
            ! the quadrilateral has 4 corners, for which we need to write out the node numbers
            ! in a specific (clockwise) order (see the VTK spec)
            ien(mp,1,ielm(mp)) = inode(mp) - nsub*nsub + nsub*(l-1) + m-1       ! 0 based indices for VTK
            ien(mp,2,ielm(mp)) = inode(mp) - nsub*nsub + nsub*(l  ) + m-1
            ien(mp,3,ielm(mp)) = inode(mp) - nsub*nsub + nsub*(l  ) + m
            ien(mp,4,ielm(mp)) = inode(mp) - nsub*nsub + nsub*(l-1) + m
          enddo
        enddo
      enddo
    enddo

  end subroutine prepare_vtk_grid


  !> Write a vtk file containing points, cells and point data (scalars and vectors)
  subroutine write_vtk(filename,xyz,ien,cell_type,scalar_names,scalars,vector_names,vectors,time_vtk)
    !> Input arguments
    character*(*), intent(in) :: filename !< Output file name
    real*4,        intent(in) :: xyz(:,:) !< Point positions
    integer,       intent(in), optional :: ien(:,:) !< Element list ien(number of basis functions, element index)
    integer,       intent(in), optional :: cell_type !< Type of interpolation (vtk param)
    character*36,  intent(in), optional :: scalar_names(:), vector_names(:)
    real*4,        intent(in), optional :: scalars(:,:), vectors(:,:,:) !< scalars(nnos, num_scalars)
    real*4,        intent(in), optional :: time_vtk

    !> Parameters
    integer, parameter    :: ivtk = 22 ! an arbitrary unit number for the VTK output file TODO get automatically

    !> Internal variables
    character             :: buffer*80, lf*1, str1*12, str2*12
    integer :: i, j, i_var
    integer :: nnos, nel, nnoel

    lf = char(10) ! line feed character

#ifdef IBM_MACHINE
    open(unit=ivtk,file=filename,form='unformatted',status="replace",access='stream')
#else
    open(unit=ivtk,file=filename,form='unformatted',status="replace",access='stream',convert='BIG_ENDIAN')
#endif

    buffer = '# vtk DataFile Version 3.0'//lf    ; write(ivtk) trim(buffer)
    buffer = 'vtk output'//lf                    ; write(ivtk) trim(buffer)
    buffer = 'BINARY'//lf                        ; write(ivtk) trim(buffer)
    buffer = 'DATASET UNSTRUCTURED_GRID'//lf     ; write(ivtk) trim(buffer)

    if (present(time_vtk)) then
      buffer = 'FIELD FieldData 1'//lf     ; write(ivtk) trim(buffer)
      buffer = 'TIME 1 1 float'//lf     ; write(ivtk) trim(buffer)
      write(ivtk) real(time_vtk,4)
    endif
    ! POINTS SECTION
    nnos = size(xyz,2)
    write(str1(1:12),'(i12)') nnos
    if (present(time_vtk)) then
      buffer = lf//'POINTS '//str1//'  float'//lf      ; write(ivtk) trim(buffer)
    else
      buffer = 'POINTS '//str1//'  float'//lf      ; write(ivtk) trim(buffer)
    endif
    write(ivtk) ((real(xyz(i,j),4),i=1,3),j=1,nnos)

    ! CELLS SECTION
    if (present(ien)) then
      nel   = size(ien,2)
      nnoel = size(ien,1)
      write(str1(1:12),'(i12)') nel            ! number of elements (cells)
      write(str2(1:12),'(i12)') nel*(1+nnoel)  ! size of the following element list (nel*(nnoel+1))
      buffer = lf//'CELLS '//str1//' '//str2//lf  ; write(ivtk) trim(buffer)
      write(ivtk) (int(nnoel,4),(int(ien(i,j),4),i=1,nnoel),j=1,nel)

      ! CELL_TYPES SECTION
      if (present(cell_type)) then
        write(str1(1:12),'(i12)') nel   ! number of elements (cells)
        buffer = lf//'CELL_TYPES'//str1//lf         ; write(ivtk) trim(buffer)
        write(ivtk) (int(cell_type,4),i=1,nel)
      endif
    endif

    ! POINT_DATA SECTION
    write(str1(1:12),'(i12)') nnos
    buffer = lf//'POINT_DATA '//str1            ; write(ivtk) trim(buffer)

    if (present(scalars)) then
      do i_var = 1, size(scalars,2)
        if (present(scalar_names)) then
          buffer = lf//'SCALARS '//scalar_names(i_var)//' float'//lf ; write(ivtk) trim(buffer)
        else
          write(str1(1:12),'(i12)') i_var
          buffer = lf//'SCALARS '//str1//' float'//lf ; write(ivtk) trim(buffer)
          endif
          buffer = 'LOOKUP_TABLE default'//lf
          write(ivtk) trim(buffer)
          write(ivtk) (real(scalars(i,i_var),4),i=1,size(scalars,1))
        enddo
      endif

      if (present(vectors)) then
        do i_var = 1, size(vectors,3)
        if (present(vector_names)) then
          buffer = lf//lf//'VECTORS '//vector_names(i_var)//' float'//lf ; write(ivtk) trim(buffer)
        else
          write(str1(1:12),'(i12)') i_var
          buffer = lf//lf//'VECTORS '//str1//' float'//lf ; write(ivtk) trim(buffer)
        endif
        write(ivtk) ((real(vectors(j,i,i_var),4),i=1,3),j=1,size(vectors,1))
      enddo
    endif
    close(ivtk)
  end subroutine write_vtk
end module mod_vtk
