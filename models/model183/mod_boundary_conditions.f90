!*******************************************************************************
!* Subroutine: boundary_condition                                              *
!*******************************************************************************
!*                                                                             *
!* Add boundary condition on the matrix.                                       *
!*                                                                             *
!* Parameters:                                                                 *
!*   my_id        - Identifier of the node in MPI_COMM_WORLD                   *
!*   node_list    - List of nodes                                              *
!*   element_list - List of all elements                                       *
!*   local_elms   - List of local elements                                     *
!*   n_local_elms - Number of local elements                                   *
!*   index_min    - Minimal index of local elements                            *
!*   index_max    - Maximal index of local elements                            *
!*   xpoint2      -                                                            *
!*   xcase2       -                                                            *
!*   psi_axis     -                                                            *
!*   psi_bnd      -                                                            *
!*   Z_xpoint     -                                                            *
!*   gmres        - boolean indicating if we are using GMRES method            *
!*   solve_only   - Indicate if we want to perform only solve                  *
!*                                                                             *
!*******************************************************************************
module mod_boundary_conditions
implicit none
contains
  subroutine boundary_conditions( my_id, node_list, element_list, bnd_node_list, local_elms,& 
                                n_local_elms, index_min, index_max, rhs_loc, xpoint2,     &
                                xcase2, R_axis, Z_axis, psi_axis, psi_bnd,                &
                                R_xpoint, Z_xpoint, psi_xpoint, a_mat)

    use mod_assembly, only : boundary_conditions_add_one_entry, boundary_conditions_add_RHS

    use phys_module, only: F0, GAMMA, bc_natural_open
    use vacuum, only: is_freebound
    use mpi_mod
    use mod_locate_irn_jcn
    use mod_integer_types
    use data_structure

    implicit none

    ! --- Routine parameters
    integer,                            intent(in)    :: my_id
    type (type_node_list),              intent(in)    :: node_list
    type (type_element_list),           intent(in)    :: element_list
    type (type_bnd_node_list),          intent(in)    :: bnd_node_list
    integer,                            intent(in)    :: local_elms(*)
    integer,                            intent(in)    :: n_local_elms
    integer,                            intent(in)    :: index_min
    integer,                            intent(in)    :: index_max
    logical,                            intent(in)    :: xpoint2
    integer,                            intent(in)    :: xcase2
    real*8,                             intent(in)    :: R_axis
    real*8,                             intent(in)    :: Z_axis
    real*8,                             intent(in)    :: psi_axis
    real*8,                             intent(in)    :: psi_bnd
    real*8,                             intent(in)    :: R_xpoint(2)
    real*8,                             intent(in)    :: Z_xpoint(2)
    real*8,                             intent(in)    :: psi_xpoint(2)
    real*8,                             intent(inout) :: rhs_loc(*)
    type(type_SP_MATRIX)                              :: a_mat

    ! Internal parameters
    real*8                :: zbig
    integer               :: i, in, iv, inode, k
    integer               :: ielm
    integer               :: index_node

    zbig = 1.d12
       do i=1, n_local_elms

          ielm = local_elms(i)

          do iv=1, n_vertex_max

             inode = element_list%element(ielm)%vertex(iv)

             if (node_list%node(inode)%boundary .ne. 0) then

                do in=a_mat%i_tor_min, a_mat%i_tor_max

                   do k=1, n_var
                     if (bc_natural_open .and. k .eq. var_zj) cycle

                      !------------------------------------ the open field lines (in case of x-point grid)
                      if ((node_list%node(inode)%boundary .eq. 1) .or. (node_list%node(inode)%boundary .eq. 3)) then

                         if ((k .eq. var_Psi) .or. (k .eq. var_Phi) .or. (k .eq. var_zj) .or. &
                              (k .eq. var_w) .or. (k .eq. var_rho) .or. (k .eq. var_T) .or. (k .eq. var_vpar) .or. &
                              (k .eq. var_Ti) .or. (k .eq. var_Te)) then
 
                          if ( (.not. is_freebound(in,k)) ) then ! apply fixed boundary conditions where necessary

                            index_node = node_list%node(inode)%index(1)
                            
                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat, my_id)

                            index_node = node_list%node(inode)%index(2)

                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat, my_id)
                            
                          endif
                        endif
                      endif

                      !------------------------------------ wall aligned with fluxsurface (in case of x-point grid)
                      if ((node_list%node(inode)%boundary .eq. 2) .or. (node_list%node(inode)%boundary .eq. 3)) then

                         if ( (.not. is_freebound(in,k)) ) then ! apply fixed boundary conditions where necessary

                            index_node = node_list%node(inode)%index(1)

                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat, my_id)

                            index_node = node_list%node(inode)%index(3)

                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat, my_id)

                         endif

                      endif

                   enddo

                enddo
             endif
          enddo
       enddo

    return
  end subroutine boundary_conditions
end module mod_boundary_conditions
