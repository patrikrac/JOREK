!> Functionality for improving numerical stability at the grid center.
!! - Switched on by the logical namelist parameter treat_axis.
!! - When used, the dofs of all axis nodes are replaced by the following:
!!   1: value
!!   2: d/dR
!!   3: d/dZ
!!   4: d^2/dRdZ (actually not used, but forced to zero)
!! - For further details, refer to the JOREK wiki at https://www.jorek.eu/wiki/doku.php?id=grid-axis
module mod_axis_treatment

contains

! Transform basis functions for the axis nodes. This will solve for new degrees of freedom at the axis.
! This is the 2D version of the transform, useful for solving the Grad-Shafranov equation in mod_poiss.
! See the next subroutine, transform_basis_for_axis_element, if multiple Fourier modes need to be treated.
subroutine transform_basis_for_axis_element_poisson(nodes, ELM, RHS, ivar_in, ivar_out, i_harm)
use data_structure
use phys_module
implicit none
type(type_node),       intent(inout) :: nodes(n_vertex_max)
real*8,          intent(inout) :: ELM(1:n_vertex_max*(n_order+1), 1:n_vertex_max*(n_order+1))
real*8,          intent(inout) :: RHS(1:n_vertex_max*(n_order+1))
integer ,        intent(in)    :: ivar_in, ivar_out, i_harm
! --- routine parameters
integer :: axis_vertex1, axis_vertex4, dof1, dof2, dof3, dof4
integer :: iv, io, jv, jo, index_iv_io, index_jv_jo
real*8  :: Ptrans(1:4, 1:4), Pmat(1:4, 1:4), rhs_i(1:4), elm_ij(1:4, 1:4)

axis_vertex1 = 1 ; axis_vertex4 = 4
dof1 = 1 ; dof2 = 2 ; dof3 = 3 ; dof4 = 4

do iv = 1, n_vertex_max

  ! initialize transpose of the transformation matrix to identity
  Ptrans = 0.d0
  do io = 1, n_order+1
    Ptrans(io, io) = 1.d0
  enddo

  ! determine transpose of the transformation matrix
  if(iv==axis_vertex1 .or. iv==axis_vertex4)then
    if(nodes(iv)%axis_dof == dof2) then      
      Ptrans(dof2,dof3) = nodes(iv)%x(1,dof3,1) ; Ptrans(dof2,dof4) = nodes(iv)%x(1,dof4,1)
      Ptrans(dof3,dof3) = nodes(iv)%x(1,dof3,2) ; Ptrans(dof3,dof4) = nodes(iv)%x(1,dof4,2)
      Ptrans(dof2,dof2) = 0.d0
      Ptrans(dof4,dof4) = 0.d0
    elseif(nodes(iv)%axis_dof == dof3)then
      Ptrans(dof2,dof2) = nodes(iv)%x(1,dof2,1) ; Ptrans(dof2,dof4) = nodes(iv)%x(1,dof4,1)
      Ptrans(dof3,dof2) = nodes(iv)%x(1,dof2,2) ; Ptrans(dof3,dof4) = nodes(iv)%x(1,dof4,2)
      Ptrans(dof3,dof3) = 0.d0
      Ptrans(dof4,dof4) = 0.d0
    else
      Ptrans = 0.d0
      do io = 1, n_order+1
        Ptrans(io, io) = 1.d0
      enddo
    endif
  endif

  ! Extract RHS associated with a vertex iv: rhs_iv
  do io = 1, n_order+1
    index_iv_io = (iv-1)*(n_order+1) + io
    rhs_i(io)   = RHS(index_iv_io)
  enddo
  ! transform test functions for rhs_v
  rhs_i = matmul( Ptrans, rhs_i)
  ! fill the updated entries in RHS vector
  do io = 1, n_order+1
    index_iv_io = (iv-1)*(n_order+1) + io    
    RHS(index_iv_io) = rhs_i(io)
  enddo

  do jv = 1, n_vertex_max

    ! initialize the transformation matrix to identity  
    Pmat = 0.d0
    do jo = 1, n_order+1
      Pmat(jo, jo) = 1.d0
    enddo

    ! determine the transformation matrix    
    if(jv==axis_vertex1 .or. jv==axis_vertex4)then
      if(nodes(jv)%axis_dof == dof2)then
        Pmat(dof3,dof2)  = nodes(jv)%x(1,dof3,1) ; Pmat(dof3,dof3) = nodes(jv)%x(1,dof3,2)
        Pmat(dof4,dof2)  = nodes(jv)%x(1,dof4,1) ; Pmat(dof4,dof3) = nodes(jv)%x(1,dof4,2)
        Pmat(dof2,dof2)  = 0.d0
        Pmat(dof4,dof4)  = 0.d0 
      elseif(nodes(jv)%axis_dof == dof3)then
        Pmat(dof2,dof2)  = nodes(jv)%x(1,dof2,1) ; Pmat(dof2,dof3) = nodes(jv)%x(1,dof2,2)
        Pmat(dof4,dof2)  = nodes(jv)%x(1,dof4,1) ; Pmat(dof4,dof3) = nodes(jv)%x(1,dof4,2)
        Pmat(dof3,dof3)  = 0.d0
        Pmat(dof4,dof4)  = 0.d0
      else
        Pmat = 0.d0
        do jo = 1, n_order+1
          Pmat(jo, jo) = 1.d0
        enddo
      endif
    endif

    ! Extract ELM associated with a vertex iv and jv: elm_iv_jv
    do io = 1, n_order+1
       index_iv_io = (iv-1)*(n_order+1) + io
       do jo = 1, n_order+1
          index_jv_jo = (jv-1)*(n_order+1) + jo
          elm_ij(io, jo) =  ELM(index_iv_io, index_jv_jo)
       enddo
    enddo   
    ! transform test and basis functions for elm_iv_jv 
    elm_ij = matmul(matmul(PTrans, elm_ij), Pmat)
    ! fill the updated entries in ELM matrix
    do io = 1, n_order+1
       index_iv_io = (iv-1)*(n_order+1) + io
       do jo = 1, n_order+1
          index_jv_jo = (jv-1)*(n_order+1) + jo
          ELM(index_iv_io, index_jv_jo) = elm_ij(io, jo)
       enddo
    enddo

  enddo ! loop over jv

enddo ! loop over iv

end subroutine transform_basis_for_axis_element_poisson

! Transform basis functions for the axis nodes. This will solve for new degrees of freedom at the axis.
! This subroutine handles 3D transforms, i.e. when multiple Fourier modes are present.
! Parameters:
!   nodes:        An array containing all the nodes of the element whose ELM and RHS are being transformed
!   ielm:         The index in the element list of the element whose ELM and RHS are being transformed
!   ELM:          The element matrix that is being transformed
!   RHS:          The RHS that is being transformed
!   i_v:          An array containing the indices of all variables involved in the transformation, can be a subset of the variables in the matrix
!   n_v:          The total number of variables in the matrix, must be >= size(i_v)
!   i_n:          An array containing the indices of all Fourier modes involved in the transformation, can be a subset of the modes in the matrix
!   n_tor_local:  The total number of Fourier modes in the matrix, must be >= size(i_n)
subroutine transform_basis_for_axis_element(nodes, ielm, ELM, RHS, i_v, n_v, i_n, n_tor_local)
use data_structure
use phys_module
implicit none
type(type_node),       intent(inout) :: nodes(n_vertex_max)
integer, dimension(:), intent(in) :: i_v, i_n
real*8, dimension(n_tor_local*n_vertex_max*(n_order+1)*n_v, n_tor_local*n_vertex_max*(n_order+1)*n_v), intent(inout) :: ELM
real*8, dimension(n_tor_local*n_vertex_max*(n_order+1)*n_v), intent(inout) :: RHS
integer, intent(in) :: ielm, n_v, n_tor_local

! --- routine parameters
integer :: axis_vertex1, axis_vertex4, dof1, dof2, dof3, dof4
integer :: iv, io, jv, jo, index_iv_io, index_jv_jo
integer :: ivar, jvar, im, in
real*8  :: Ptrans(1:4, 1:4), Pmat(1:4, 1:4), rhs_i(1:4), elm_ij(1:4, 1:4), theta

axis_vertex1 = 1 ; axis_vertex4 = 4
dof1 = 1 ; dof2 = 2 ; dof3 = 3 ; dof4 = 4

do iv = 1, n_vertex_max
#if STELLARATOR_MODEL
  if (iv .le. 2) then
    theta = 2.d0*pi*float(mod(ielm-1,n_tht))/float(n_tht)
  else
    theta = 2.d0*pi*float(mod(ielm,n_tht))/float(n_tht)
  end if
#endif
  ! initialize transpose of the transformation matrix to identity
  Ptrans = 0.d0
  do io = 1, n_order+1
    Ptrans(io, io) = 1.d0
  enddo

  ! determine transpose of the transformation matrix
  if (iv==axis_vertex1 .or. iv==axis_vertex4) then
    if (nodes(iv)%axis_dof == dof2) then
#if STELLARATOR_MODEL
      Ptrans(dof2,dof3) = cos(theta)/(3.d0*float(n_flux-1)); Ptrans(dof2,dof4) = -2.d0*pi*sin(theta)/(9.d0*float(n_tht*(n_flux-1)))
      Ptrans(dof3,dof3) = sin(theta)/(3.d0*float(n_flux-1)); Ptrans(dof3,dof4) =  2.d0*pi*cos(theta)/(9.d0*float(n_tht*(n_flux-1)))
#else
      Ptrans(dof2,dof3) = nodes(iv)%x(1,dof3,1) ; Ptrans(dof2,dof4) = nodes(iv)%x(1,dof4,1)
      Ptrans(dof3,dof3) = nodes(iv)%x(1,dof3,2) ; Ptrans(dof3,dof4) = nodes(iv)%x(1,dof4,2)
#endif
      Ptrans(dof2,dof2) = 0.d0
      Ptrans(dof4,dof4) = 0.d0
    else if (nodes(iv)%axis_dof == dof3) then
#if STELLARATOR_MODEL
      Ptrans(dof2,dof2) = cos(theta)/(3.d0*float(n_flux-1)); Ptrans(dof2,dof4) = -2.d0*pi*sin(theta)/(9.d0*float(n_tht*(n_flux-1)))
      Ptrans(dof3,dof2) = sin(theta)/(3.d0*float(n_flux-1)); Ptrans(dof3,dof4) =  2.d0*pi*cos(theta)/(9.d0*float(n_tht*(n_flux-1)))
#else
      Ptrans(dof2,dof2) = nodes(iv)%x(1,dof2,1) ; Ptrans(dof2,dof4) = nodes(iv)%x(1,dof4,1)
      Ptrans(dof3,dof2) = nodes(iv)%x(1,dof2,2) ; Ptrans(dof3,dof4) = nodes(iv)%x(1,dof4,2)
#endif
      Ptrans(dof3,dof3) = 0.d0
      Ptrans(dof4,dof4) = 0.d0
    end if          
  end if

  do ivar = 1, size(i_v)
  do im   = 1, size(i_n)
    ! Extract RHS associated with a vertex iv: rhs_iv
    do io = 1, n_order+1
      index_iv_io = n_tor_local*n_v*(n_order+1)*(iv-1) + n_tor_local*n_v*(io-1) + n_tor_local*(i_v(ivar)-1) + (i_n(im)-i_n(1)+1)
      rhs_i(io)   = RHS(index_iv_io)
    enddo
    ! transform test functions for rhs_v
    rhs_i = matmul( Ptrans, rhs_i)
    ! fill the updated entries in RHS vector
    do io = 1, n_order+1
      index_iv_io = n_tor_local*n_v*(n_order+1)*(iv-1) + n_tor_local*n_v*(io-1) + n_tor_local*(i_v(ivar)-1) + (i_n(im)-i_n(1)+1)
      RHS(index_iv_io) = rhs_i(io)
    enddo
  enddo ! loop over im
  enddo ! loop over ivar

  do jv = 1, n_vertex_max
#if STELLARATOR_MODEL
    if (jv .le. 2) then
      theta = 2.d0*pi*float(mod(ielm-1,n_tht))/float(n_tht)
    else
      theta = 2.d0*pi*float(mod(ielm,n_tht))/float(n_tht)
    end if
#endif
    ! initialize the transformation matrix to identity  
    Pmat = 0.d0
    do jo = 1, n_order+1
      Pmat(jo, jo) = 1.d0
    enddo

    ! determine the transformation matrix    
    if (jv == axis_vertex1 .or. jv == axis_vertex4) then
      if (nodes(jv)%axis_dof == dof2) then
#if STELLARATOR_MODEL
        Pmat(dof3,dof2) = cos(theta)/(3.d0*float(n_flux-1)); Pmat(dof3,dof3) = sin(theta)/(3.d0*float(n_flux-1))
        Pmat(dof4,dof2) = -2.d0*pi*sin(theta)/(9.d0*float(n_tht*(n_flux-1))); Pmat(dof4,dof3) = 2.d0*pi*cos(theta)/(9.d0*float(n_tht*(n_flux-1)))
#else
        Pmat(dof3,dof2)  = nodes(jv)%x(1,dof3,1) ; Pmat(dof3,dof3) = nodes(jv)%x(1,dof3,2)
        Pmat(dof4,dof2)  = nodes(jv)%x(1,dof4,1) ; Pmat(dof4,dof3) = nodes(jv)%x(1,dof4,2)
#endif
        Pmat(dof2,dof2)  = 0.d0
        Pmat(dof4,dof4)  = 0.d0
      else if (nodes(jv)%axis_dof == dof3) then
#if STELLARATOR_MODEL
        Pmat(dof2,dof2) = cos(theta)/(3.d0*float(n_flux-1)); Pmat(dof2,dof3) = sin(theta)/(3.d0*float(n_flux-1))
        Pmat(dof4,dof2) = -2.d0*pi*sin(theta)/(9.d0*float(n_tht*(n_flux-1))); Pmat(dof4,dof3) = 2.d0*pi*cos(theta)/(9.d0*float(n_tht*(n_flux-1)))
#else
        Pmat(dof2,dof2)  = nodes(jv)%x(1,dof2,1) ; Pmat(dof2,dof3) = nodes(jv)%x(1,dof2,2)
        Pmat(dof4,dof2)  = nodes(jv)%x(1,dof4,1) ; Pmat(dof4,dof3) = nodes(jv)%x(1,dof4,2)
#endif
        Pmat(dof3,dof3)  = 0.d0
        Pmat(dof4,dof4)  = 0.d0
      end if            
    end if

    ! Extract ELM associated with a vertex iv and jv: elm_iv_jv
    do ivar = 1, size(i_v)
    do im   = 1, size(i_n)
    
    do jvar = 1, size(i_v)
    do in   = 1, size(i_n)
    
       do io = 1, n_order+1
          index_iv_io = n_tor_local*n_v*(n_order+1)*(iv-1) + n_tor_local*n_v*(io-1) + n_tor_local*(i_v(ivar)-1) + (i_n(im)-i_n(1)+1)
          do jo = 1, n_order+1
             index_jv_jo = n_tor_local*n_v*(n_order+1)*(jv-1) + n_tor_local*n_v*(jo-1) + n_tor_local*(i_v(jvar)-1) + (i_n(in)-i_n(1)+1)
             elm_ij(io, jo) =  ELM(index_iv_io, index_jv_jo)
          enddo
       enddo
       ! transform test and basis functions for elm_iv_jv
       elm_ij = matmul(matmul(PTrans, elm_ij), Pmat)
       ! fill the updated entries in ELM matrix
       do io = 1, n_order+1
          index_iv_io = n_tor_local*n_v*(n_order+1)*(iv-1) + n_tor_local*n_v*(io-1) + n_tor_local*(i_v(ivar)-1) + (i_n(im)-i_n(1)+1)
          do jo = 1, n_order+1
             index_jv_jo = n_tor_local*n_v*(n_order+1)*(jv-1) + n_tor_local*n_v*(jo-1) + n_tor_local*(i_v(jvar)-1) + (i_n(in)-i_n(1)+1)
             ELM(index_iv_io, index_jv_jo) = elm_ij(io, jo)
          enddo
       enddo

    enddo ! loop over in
    enddo ! loop over jvar
       
    enddo ! loop over im
    enddo ! loop over ivar

  enddo ! loop over jv

enddo ! loop over iv

end subroutine transform_basis_for_axis_element

! Since the axis treatment solves for new degrees of freedom, we need to
! transforms degrees of freedom to old ones.
subroutine new_to_old_dofs_on_the_axis(node_list, vg, new_dofs, old_dofs)
use data_structure
use phys_module
implicit none
type(type_node_list),    intent(inout) :: node_list
integer,   intent(in) :: vg
real*8,    intent(in) :: new_dofs(4)
real*8,    intent(out):: old_dofs(4)
integer :: dof1, dof2, dof3, dof4, jo
real*8  :: Pmat(4,4), theta

theta = 2.d0*pi*float(mod(vg-1,n_tht))/float(n_tht)
dof1 = 1 ; dof2 = 2 ; dof3 = 3 ; dof4 = 4
Pmat = 0.d0; Pmat(dof1,dof1) = 1.d0
if (node_list%node(vg)%axis_dof == dof2) then
#if STELLARATOR_MODEL
  Pmat(dof3,dof2) = cos(theta)/(3.d0*float(n_flux-1)); Pmat(dof3,dof3) = sin(theta)/(3.d0*float(n_flux-1))
  Pmat(dof4,dof2) = -2.d0*pi*sin(theta)/(9.d0*float(n_tht*(n_flux-1))); Pmat(dof4,dof3) = 2.d0*pi*cos(theta)/(9.d0*float(n_tht*(n_flux-1)))
#else
  Pmat(dof3,dof2) = node_list%node(vg)%x(1,dof3,1)  ;  Pmat(dof3,dof3) = node_list%node(vg)%x(1,dof3,2)
  Pmat(dof4,dof2) = node_list%node(vg)%x(1,dof4,1)  ;  Pmat(dof4,dof3) = node_list%node(vg)%x(1,dof4,2)
#endif
else if (node_list%node(vg)%axis_dof == dof3) then
#if STELLARATOR_MODEL
  Pmat(dof2,dof2) = cos(theta)/(3.d0*float(n_flux-1)); Pmat(dof2,dof3) = sin(theta)/(3.d0*float(n_flux-1))
  Pmat(dof4,dof2) = -2.d0*pi*sin(theta)/(9.d0*float(n_tht*(n_flux-1))); Pmat(dof4,dof3) = 2.d0*pi*cos(theta)/(9.d0*float(n_tht*(n_flux-1)))
#else
  Pmat(dof2,dof2) = node_list%node(vg)%x(1,dof2,1)  ;  Pmat(dof2,dof3) = node_list%node(vg)%x(1,dof2,2)
  Pmat(dof4,dof2) = node_list%node(vg)%x(1,dof4,1)  ;  Pmat(dof4,dof3) = node_list%node(vg)%x(1,dof4,2)
#endif
else
  do jo=2,n_order+1
    Pmat(jo, jo) = 1.d0
  end do
end if

old_dofs = matmul(Pmat, new_dofs)

end subroutine new_to_old_dofs_on_the_axis

!> Add condition for the axis directly in the matrix.
!! - This aims to apply C0 continuity on the grid axis and is used only when treat_aixs=.t.
!! - The fourth dof is penalized (d^2/dRdZ) when the treat_axis option is used.
subroutine penalize_dof_on_axis(node_list, dof, element_list, local_elms, n_local_elms, index_min, index_max, a_mat)

  use mod_assembly, only : boundary_conditions_add_one_entry, boundary_conditions_add_RHS
  use data_structure
  use mod_locate_irn_jcn

  implicit none

  ! Subroutine parameters
  integer,                   intent(in)    :: local_elms(*)         !< List of local elements
  integer,                   intent(in)    :: n_local_elms          !< Number of local elements
  integer,                   intent(in)    :: index_min, index_max  !< Min/max index of local elements
  type (type_node_list),     intent(in)    :: node_list             !< List of nodes
  integer,                   intent(in)    :: dof                   !< which dof to penalize  
  type (type_element_list),  intent(in)    :: element_list          !< List of all elements
  type(type_SP_MATRIX)                     :: a_mat
  ! Internal parameters
  real*8  :: zbig
  integer :: i, in, iv, inode, k
  integer :: index_large_i, index_node, index_node2, ielm
  integer :: ilarge2, n_tor_local
  integer(kind=int_all):: ijA_position,ijA_position2

  n_tor_local = (a_mat%i_tor_max - a_mat%i_tor_min + 1)

  zbig = 1.d12
  do i=1, n_local_elms

    ielm = local_elms(i)

    do iv=1, n_vertex_max

      inode = element_list%element(ielm)%vertex(iv)

      if (node_list%node(inode)%axis_node) then

        do in=a_mat%i_tor_min, a_mat%i_tor_max
          do k=1, n_var

            index_node = node_list%node(inode)%index(dof)
            if ((index_node .ge. index_min) .and. (index_node .le. index_max)) then
              call locate_irn_jcn(index_node,index_node,index_min,index_max,ijA_position,a_mat)
              index_large_i = n_tor_local * n_var * (index_node - 1)
              ilarge2 = ijA_position - 1 + ((k-1)*n_tor_local + in-a_mat%i_tor_min) * n_var*n_tor_local &
                + (k-1)*n_tor_local + in - a_mat%i_tor_min + 1
              a_mat%irn(ilarge2) =  n_tor_local * n_var * (index_node-1) + (k-1)*n_tor_local + in - a_mat%i_tor_min + 1
              a_mat%jcn(ilarge2) =  n_tor_local * n_var * (index_node-1) + (k-1)*n_tor_local + in - a_mat%i_tor_min + 1
              a_mat%val(ilarge2)   = zbig
            end if

          enddo
        enddo

      endif

    enddo  ! n_vertex
  enddo ! n_elements

  return

end subroutine penalize_dof_on_axis

end module mod_axis_treatment
