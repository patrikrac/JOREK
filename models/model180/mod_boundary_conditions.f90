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
!* Subroutine: solve_Psi_boundary_eqn                                          *
!*******************************************************************************
!*                                                                             *
!* Solve the differential equation for Psi on boundary, ensuring that n.B = 0  *
!* (see eq 4.23 in N. Nikulsin's PhD thesis, doi:10.17617/2.3359934)           *
!* The solution is then used as an imhomogeneous Dirichlet b.c. for Psi        *
!*                                                                             *
!*******************************************************************************
!* Subroutine: setup_boundary_condition                                        *
!*******************************************************************************
!*                                                                             *
!* Projects the solution of the boundary Psi equation onto the JOREK finite    *
!*  element basis and writes the calculated dofs into the appropriate nodes.   *
!*                                                                             *
!*******************************************************************************
module mod_boundary_conditions
implicit none

real*8, dimension(:), allocatable :: Psi_b

private :: Zn, Zn_chi, Zm, Zm_tht, Zm_tht_2, HZn

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
                              (k .eq. var_Ti) .or. (k .eq.  var_Te)) then
 
                          if ( (.not. is_freebound(in,k)) ) then ! apply fixed boundary conditions where necessary

                            index_node = node_list%node(inode)%index(1)
  
                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat)

                            index_node = node_list%node(inode)%index(2)

                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat)
                          endif
                        endif
                      endif

                      !------------------------------------ wall aligned with fluxsurface (in case of x-point grid)
                      if ((node_list%node(inode)%boundary .eq. 2) .or. (node_list%node(inode)%boundary .eq. 3)) then

                         if ( (.not. is_freebound(in,k)) ) then ! apply fixed boundary conditions where necessary

                            index_node = node_list%node(inode)%index(1)

                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat)

                            index_node = node_list%node(inode)%index(3)

                            call boundary_conditions_add_one_entry(                 &
                                   index_node, k, in, index_node, k, in,            &
                                   zbig, index_min, index_max, a_mat)
                         endif

                      endif

                   enddo

                enddo
             endif
          enddo
       enddo

    return
  end subroutine boundary_conditions

  
  subroutine solve_Psi_boundary_eqn(node_list, boundary_list, ierr)
  !---------------------------------------------------------------------------------------------------------------------------------
  ! The Psi boundary equation is solved using the Fourier-Galerkin method here
  !
  ! \int\int v*dPsi/dtheta*dtheta*dchi = -\int\int v*J'grad(s).grad(chi)*dtheta*dchi
  !
  ! where v = Z_m(theta)*Z_n(chi) is the test function and Psi = \sum_{m,n} C_{m,n}*Z_m(theta)*Z_n(chi)
  !
  ! The theta integration is over [0,2*pi] and the chi integration is over [chi_0+2*pi*F_0/N_p, chi_0], where chi_0 is the value of
  !  chi at the initial chi=const surface and N_p is the number of periods. Note that the limits of the chi integration is from 
  !  high to low values, because chi was calculated using a lefthand coordinate system, while JOREK is righthanded. Chi is therefore
  !  decreasing along the integration path.
  !---------------------------------------------------------------------------------------------------------------------------------
    use constants, only: pi
    use mod_parameters, only: n_period, n_plane, n_order, n_tor, n_coord_tor
    use phys_module, only: F0, m_pol_bc, extended_boundary
    use gauss
    use basis_at_gaussian
    use data_structure
    use mod_chi
    use mod_SolveMN
    implicit none

    type(type_node_list),        intent(in) :: node_list
    type(type_bnd_element_list), intent(in) :: boundary_list

    real*8  :: Bp_n, Bp_cyl_n, grad_chi_n, phi, theta, element_size_ij, element_size_perp, BigR, grad_chi(3), Jgrad_ps(3)
    real*8  :: Jgrad_ps_cart(3), B_field(3), B_r, B_phi, B_z, B_field_cyl(3)
    real*8  :: theta_points, phi_points, x_points, y_points, z_points, s_points
    integer :: N_tht, mp, ms, ielm, i, j, j2, n, m, nn, mm, in, ind1, ind2, info
    integer :: number_of_points, i_tht, j_gauss, k_plane, np
    integer :: ierr

    type(type_bnd_element) :: bnd_element
    type(type_node)        :: node

    real*8, dimension(:,:,:), allocatable :: x_g, x_s, x_p, y_g, y_s, y_p
    real*8, dimension(:,:),   allocatable :: Amat
    real*8, dimension(:),     allocatable :: RHS

    real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: chi

    real*8, dimension(:,:,:), allocatable :: bx_p, by_p, bz_p
    real*8, dimension(:,:,:), allocatable :: step_xyz, time1_xyz, time2_xyz, x_xyz, y_xyz, z_xyz
    logical :: file_exists


    write(*,*) "***************************************"
    write(*,*) "*    Solving boundary Psi equation    *"
    write(*,*) "***************************************"
    write(*,'(A,I4,A,I4)') "Toroidal modes: ", n_tor
    write(*,'(A,I4,A,I4)') "Poloidal modes: ", m_pol_bc

    N_tht = boundary_list%n_bnd_elements

    allocate(x_g(n_plane,n_gauss,N_tht)); allocate(x_s, x_p, mold=x_g)
    allocate(y_g(n_plane,n_gauss,N_tht)); allocate(y_s, y_p, mold=y_g)
    allocate(Amat(n_tor*m_pol_bc,n_tor*m_pol_bc))
    allocate(RHS(n_tor*m_pol_bc))

    x_g  = 0.d0; x_s = 0.d0; x_p = 0.d0
    y_g  = 0.d0; y_s = 0.d0; y_p = 0.d0
    
    do ielm=1,N_tht
      bnd_element = boundary_list%bnd_element(ielm)

      do i=1,2
        node = node_list%node(bnd_element%vertex(i))

        do j=1,2
          j2 = bnd_element%direction(i,j)
          element_size_ij = bnd_element%size(i,j)

          do ms=1,n_gauss
            do mp=1,n_plane
              do in=1,n_coord_tor
                x_g(mp,ms,ielm)  = x_g(mp,ms,ielm)  + node%x(in,j2,1)*element_size_ij*H1(i,j,ms)   *HZ_coord(in,mp)
                x_s(mp,ms,ielm)  = x_s(mp,ms,ielm)  + node%x(in,j2,1)*element_size_ij*H1_s(i,j,ms) *HZ_coord(in,mp)
                x_p(mp,ms,ielm)  = x_p(mp,ms,ielm)  + node%x(in,j2,1)*element_size_ij*H1(i,j,ms)   *HZ_coord_p(in,mp)

                y_g(mp,ms,ielm)  = y_g(mp,ms,ielm)  + node%x(in,j2,2)*element_size_ij*H1(i,j,ms)   *HZ_coord(in,mp)
                y_s(mp,ms,ielm)  = y_s(mp,ms,ielm)  + node%x(in,j2,2)*element_size_ij*H1_s(i,j,ms) *HZ_coord(in,mp)
                y_p(mp,ms,ielm)  = y_p(mp,ms,ielm)  + node%x(in,j2,2)*element_size_ij*H1(i,j,ms)   *HZ_coord_p(in,mp)

              end do
            end do
          end do
        end do
      end do 
    end do
    
    ! Integration of the RHS: it is much easier to integrate over one period than over the manifold [0,2*pi]x[chi_0,chi_0+2*pi*F_0/N_p]
    ! Due to periodicity, integration over the manifold above is equivalent to integration over one period
    ! The Jacobian for dtheta*dchi -> dtheta*dphi is dchi/dphi
    Amat = 0.d0; RHS = 0.d0
    if (extended_boundary) then
      ! write out points in .dat file for xyz points
      open(66,file='xyz.dat',action='write')
      number_of_points = n_gauss*n_plane*N_tht
 
      write(66,*), number_of_points
     
      do i_tht=1,N_tht
        do j_gauss=1,n_gauss
          do k_plane=1, n_plane
            phi_points = 2.d0*pi*float(k_plane-1)/float(n_plane*n_period)
         
            x_points =  x_g(k_plane,j_gauss,i_tht)*Cos(phi_points)       ! x=R, y=Z see jorek Wiki: coordinates
            y_points = -x_g(k_plane,j_gauss,i_tht)*Sin(phi_points)      
            z_points =  y_g(k_plane,j_gauss,i_tht)              
            write(66,*), x_g(k_plane,j_gauss,i_tht)*Cos(phi_points), -x_g(k_plane,j_gauss,i_tht)*Sin(phi_points), y_g(k_plane,j_gauss,i_tht)
          end do
        end do
      end do
 
      write(*,*) "xyz.dat file is created"
      write(*,*) "if the integration points changed, please re-run jorek2_fields_xyz_stel with this new xyz.dat file"
      write(*,*) "run model 180 again afterwards"
      close(66)

      inquire(file="fields_xyz.dat", exist = file_exists)
      
      if (file_exists) then
        write(*,*) "fields_xyz.dat file exists"
        open(26,file='fields_xyz.dat',action='read')
        read(26,*)  ! first row of fields_xyz is just a header
        allocate(bx_p(n_plane,n_gauss,N_tht), by_p(n_plane,n_gauss,N_tht), bz_p(n_plane,n_gauss,N_tht))
        allocate(step_xyz(n_plane,n_gauss,N_tht), time1_xyz(n_plane,n_gauss,N_tht), time2_xyz(n_plane,n_gauss,N_tht), x_xyz(n_plane,n_gauss,N_tht), y_xyz(n_plane,n_gauss,N_tht), z_xyz(n_plane,n_gauss,N_tht))
        do ielm=1,N_tht
          do ms=1,n_gauss
            do mp=1,n_plane
              read(26,*) step_xyz(mp,ms,ielm), time1_xyz(mp,ms,ielm), time2_xyz(mp,ms,ielm), x_xyz(mp,ms,ielm), y_xyz(mp,ms,ielm), z_xyz(mp,ms,ielm), bx_p(mp,ms,ielm), by_p(mp,ms,ielm), bz_p(mp,ms,ielm)
            enddo
          enddo
        enddo
        close(26)
      else
         write(*,*) "Could not find fields_xyz.dat file"
         write(*,*) "please run jorek2_fields_xyz_stel.f90 with genereated xyz.dat file to create fields_xyz.dat file"
         write(*,*) "run model 180 again afterwards"
         ierr = 3
         return
      endif

      open(69, file="Bpn_theta_phi.dat", action="write")
      write(69, '(a)') "J_ps_cart, Bx, By, Bz, Bp_n, grad_chi_n, theta, phi"
      
      do ielm=1,N_tht
        do ms=1,n_gauss
          theta = 2.d0*pi*(float(ielm-1) + xgauss(ms))/float(N_tht)
          do mp=1,n_plane
            BigR = x_g(mp,ms,ielm)
            phi = 2.d0*pi*float(mp-1)/float(n_plane*n_period)
            B_field = (/ bx_p(mp,ms,ielm), by_p(mp,ms,ielm), bz_p(mp,ms,ielm) /) ! creating B-field arrray to replace grad(chi)
            ! converting B-field to cylindrical coordiantes to be able to use J_grad_ps
            ! B_r   = bx_p(mp,ms,ielm)*Cos(phi) + by_p(mp,ms,ielm)*(-Sin(phi))
            ! B_phi = bx_p(mp,ms,ielm)*(-Sin(phi)) + by_p(mp,ms,ielm)*(-Cos(phi))
            ! B_z   = bz_p(mp,ms,ielm)
            ! B_field_cyl = (/ B_r, B_z, B_phi /)  ! because in JOREK it's (R,Z,Phi)
            chi = get_chi_domm(x_g(mp,ms,ielm),y_g(mp,ms,ielm),phi)
            grad_chi = (/ chi(1,0,0), chi(0,1,0), chi(0,0,1)/BigR /)
            ! -e_theta x e_phi = -J*grad(psi)
            Jgrad_ps = (/ -BigR*y_s(mp,ms,ielm), BigR*x_s(mp,ms,ielm), x_p(mp,ms,ielm)*y_s(mp,ms,ielm) - x_s(mp,ms,ielm)*y_p(mp,ms,ielm) /)
            Jgrad_ps = Jgrad_ps*N_tht/(2.d0*pi) ! Multiply by dt/dtheta since J = 1/(grad(psi).(grad(theta)xgrad(phi))) and (s,t,phi) CS used above
            ! Note that J'*dchi/dphi = J, where J' = 1/(grad(psi).(grad(theta)xgrad(chi))) and dchi/dphi is from the switch dtheta*dchi -> dtheta*dphi
            
            ! Jgrad_ps is needed in cartesian coordinates, since the B field is in cartesian coordinates
            ! e_theta x e_phi, e_theta=dr(x,y,z)/dtheta, e_phi=dr(x,y,z)/dphi
            Jgrad_ps_cart = (/ -x_s(mp,ms,ielm)*Sin(phi)*y_p(mp,ms,ielm) - y_s(mp,ms,ielm)*(-x_p(mp,ms,ielm)*Sin(phi)-BigR*Cos(phi)), &
                               y_s(mp,ms,ielm)*(x_p(mp,ms,ielm)*Cos(phi)-BigR*Sin(phi)) - x_s(mp,ms,ielm)*Cos(phi)*y_p(mp,ms,ielm), &
                               x_s(mp,ms,ielm)*Cos(phi)*(-x_p(mp,ms,ielm)*Sin(phi)-BigR*Cos(phi)) + x_s(mp,ms,ielm)*Sin(phi)*(x_p(mp,ms,ielm)*Cos(phi)-BigR*Sin(phi)) /)
            Jgrad_ps_cart = Jgrad_ps_cart*N_tht/(2.d0*pi)
            grad_chi_n = dot_product(Jgrad_ps,grad_chi)
            Bp_n = dot_product(Jgrad_ps_cart,B_field)
            ! Bp_cyl_n = dot_product(-Jgrad_ps,B_field_cyl)
            
            write(69,'(10ES18.10)') Jgrad_ps_cart, B_field, Bp_n, grad_chi_n, 2.d0*pi*(float(ielm-1) + xgauss(ms))/float(N_tht), 2.d0*pi*float(mp-1)/float(n_plane*n_period)

            ind1 = 1
            do n=1,n_tor
              do m=1,m_pol_bc
                RHS(ind1) = RHS(ind1) + dot_product(Jgrad_ps_cart,B_field)*Zn(n,chi(0,0,0))*Zm(m,theta)*wgauss(ms) ! B-field version
                ! RHS(ind1) = RHS(ind1) + dot_product(-Jgrad_ps,B_field_cyl)*Zn(n,chi(0,0,0))*Zm(m,theta)*wgauss(ms) ! B_cyl-field version
                ind1 = ind1 + 1
              end do
            end do
          end do
        end do
      end do
      close(69)
    else   ! if case started in line 255
      ! original script for gradchi B.C.
      do ielm=1,N_tht
        do ms=1,n_gauss
          theta = 2.d0*pi*(float(ielm-1) + xgauss(ms))/float(N_tht)
          do mp=1,n_plane
            BigR = x_g(mp,ms,ielm)
            phi = 2.d0*pi*float(mp-1)/float(n_plane*n_period)
            chi = get_chi_domm(x_g(mp,ms,ielm),y_g(mp,ms,ielm),phi)
            grad_chi = (/ chi(1,0,0), chi(0,1,0), chi(0,0,1)/BigR /)
            ! -e_theta x e_phi = -J*grad(psi)
            Jgrad_ps = (/ -BigR*y_s(mp,ms,ielm), BigR*x_s(mp,ms,ielm), x_p(mp,ms,ielm)*y_s(mp,ms,ielm) - x_s(mp,ms,ielm)*y_p(mp,ms,ielm) /)
            Jgrad_ps = Jgrad_ps*N_tht/(2.d0*pi) ! Multiply by dt/dtheta since J = 1/(grad(psi).(grad(theta)xgrad(phi))) and (s,t,phi) CS used above
            ! Note that J'*dchi/dphi = J, where J' = 1/(grad(psi).(grad(theta)xgrad(chi))) and dchi/dphi is from the switch dtheta*dchi -> dtheta*dphi
            grad_chi_n = dot_product(Jgrad_ps,grad_chi)

            ind1 = 1
            do n=1,n_tor
              do m=1,m_pol_bc
                RHS(ind1) = RHS(ind1) + dot_product(Jgrad_ps,grad_chi)*Zn(n,chi(0,0,0))*Zm(m,theta)*wgauss(ms) ! grad(chi) version
                ind1 = ind1 + 1
              end do
            end do
          end do
        end do
      end do
    endif  ! else case started in line 348

    RHS = RHS*(2.d0*pi)**2/float(n_period*n_plane*N_tht)

    ! The integrals in the matrix are done analytically since they only contain sines and cosines
    ! Here, it is better to integrate over the original manifold
    ind1 = 1
    ! Loop over test functions (rows in the matrix)
    do n=1,n_tor
      do m=1,m_pol_bc
        ind2 = 1
        ! Loop over basis functions in the 2D Fourier decomposition of Psi (columns in the matrix)
        do nn=1,n_tor
          do mm=1,m_pol_bc
            if (n .eq. nn) then ! The test and basis functions are orthogonal if their chi indices are different (no chi derivatives)
              if (mod(m,2) .eq. 1 .and. mm .eq. m + 1) then ! If theta t.f. is cos, theta b.f. must be sin w/ same mode number (derivative is cos)
                Amat(ind1,ind2) = -pi*(mm/2)
              else if (mod(m,2) .eq. 0 .and. mm .eq. m - 1) then ! If theta t.f. is sin, theta b.f. must be cos w/ same mode number (derivative is sin)
                Amat(ind1,ind2) =  pi*(m/2)
              end if

              ! Multiply by the result of integration over chi; additional factor of 2 if both chi t.f. and b.f. are 1
              if (n .eq. 1) then
                Amat(ind1,ind2) = Amat(ind1,ind2)*2.d0*pi*F0/n_period
              else
                Amat(ind1,ind2) = Amat(ind1,ind2)*pi*F0/n_period
              end if
            end if

            ind2 = ind2 + 1
          end do
        end do

        ind1 = ind1 + 1
      end do
    end do

    call SolveMN(Amat, info)
    if (info .ne. 0) then
      write(*,'(A,I3)') "ERROR: solve_Psi_boundary_eqn, matrix inversion failed: ", info
      stop
    end if

    if (allocated(Psi_b)) deallocate(Psi_b)
    allocate(Psi_b(n_tor*m_pol_bc))

    Psi_b = matmul(Amat,RHS)
  end subroutine solve_Psi_boundary_eqn

  subroutine setup_boundary_condition(node_list, bnd_node_list)
    use constants, only: pi
    use mod_parameters, only: n_tor, n_order, n_plane, n_period, n_coord_tor, var_Psi
    use phys_module, only: F0, m_pol_bc
    use basis_at_gaussian
    use data_structure
    use mod_chi
    implicit none
    
    type(type_node_list),     intent(inout) :: node_list
    type(type_bnd_node_list), intent(in)    :: bnd_node_list
    
    integer :: ielm, i, j, mp, im, in, n, m, ind
    real*8  :: R, z, R_s, z_s, phi, theta, Psi, Psi_tht, Psi_chi, chi_tht
    
    ! Dofs for Psi at a particular boundary node -- values of Psi (Psi_dof) and its poloidal (tangential) derivative (Psi_dof2)
    real*8, dimension(n_tor) :: Psi_dof, Psi_dof2
    
    real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: chi
    
    do i=1,bnd_node_list%n_bnd_nodes
      in = bnd_node_list%bnd_node(i)%index_jorek
      j  = bnd_node_list%bnd_node(i)%direction(2)
      theta = 2.d0*pi*float(i-1)/float(bnd_node_list%n_bnd_nodes)
      Psi_dof = 0.d0; Psi_dof2 = 0.d0
      
      do mp=1,n_plane
        phi = 2.d0*pi*float(mp-1)/float(n_plane*n_period)

        R = 0.d0; R_s = 0.d0
        z = 0.d0; z_s = 0.d0
        Psi = 0.d0; Psi_tht = 0.d0; Psi_chi = 0.d0
        
        ! Find R, z and their poloidal derivatives at the current boundary node and poloidal plane
        do im=1,n_coord_tor
          R   = R   + node_list%node(in)%x(im,1,1)*HZ_coord(im,mp)
          R_s = R_s + node_list%node(in)%x(im,j,1)*HZ_coord(im,mp)*3.d0 ! Poloidal derivative at node is 3x the corresponding dof of that node
          z   = z   + node_list%node(in)%x(im,1,2)*HZ_coord(im,mp)
          z_s = z_s + node_list%node(in)%x(im,j,2)*HZ_coord(im,mp)*3.d0
        end do
        
        chi = get_chi_domm(R,z,phi)
        chi_tht = (R_s*chi(1,0,0) + z_s*chi(0,1,0))*bnd_node_list%n_bnd_nodes/(2.d0*pi)
        
        ! Using the (theta,chi) 2D Fourier basis, calculate the values of Psi and its derivatives at the current point
        ind = 1
        do n=1,n_tor
          do m=1,m_pol_bc
            Psi = Psi + Psi_b(ind)*Zn(n,chi(0,0,0))*Zm(m,theta)
            Psi_tht = Psi_tht + Psi_b(ind)*Zn(n,chi(0,0,0))*Zm_tht(m,theta)
            Psi_chi = Psi_chi + Psi_b(ind)*Zn_chi(n,chi(0,0,0))*Zm(m,theta)
            ind = ind + 1
          end do
        end do
        
        ! Now integrate Psi and its theta derivative (in the (psi,theta,phi) CS) over phi
        do im=1,n_tor
          Psi_dof(im) = Psi_dof(im) + Psi*HZn(im,phi)
          Psi_dof2(im) = Psi_dof2(im) + (Psi_tht + Psi_chi*chi_tht)*HZn(im,phi)
        end do
      end do
      
      Psi_dof2 = Psi_dof2*2.d0*pi/bnd_node_list%n_bnd_nodes ! Multiply by dtheta/dt to get derivative wrt element local coordinate
      
      ! Combining two steps here (multiplying by delta_phi and dividing by integrals of 1, sin^2, cos^2 to get modes), several factors cancel
      Psi_dof(1) = Psi_dof(1)/n_plane
      Psi_dof2(1) = Psi_dof2(1)/(3.d0*n_plane) ! divide the derivative by 3 to get the Bezier dof
      if (n_tor .gt. 1) then
        Psi_dof(2:) = Psi_dof(2:)*2.d0/n_plane
        Psi_dof2(2:) = Psi_dof2(2:)*2.d0/(3.d0*n_plane)
      end if
      
      ! Multiply by F0 so that the internally stored Psi matches the Grad-Shafranov solution in the tokamak limit
      node_list%node(in)%values(:,1,var_Psi) = F0*Psi_dof
      node_list%node(in)%values(:,j,var_Psi) = F0*Psi_dof2
    end do
  end subroutine setup_boundary_condition

  pure real*8 function Zn(n, chi)
    use mod_parameters, only: n_period
    use phys_module, only: F0
    implicit none
    integer, intent(in) :: n
    real*8,  intent(in) :: chi

    if (n .eq. 1) then
      Zn = 1.d0
      return
    end if

    if (mod(n,2) .eq. 0) then
      Zn = cos(n_period*int(n/2)*chi/F0)
    else
      Zn = sin(n_period*int(n/2)*chi/F0)
    end if
  end function Zn

  pure real*8 function Zn_chi(n, chi)
    use mod_parameters, only: n_period
    use phys_module, only: F0
    implicit none
    integer, intent(in) :: n
    real*8,  intent(in) :: chi

    if (n .eq. 1) then
      Zn_chi = 0.d0
      return
    end if

    if (mod(n,2) .eq. 0) then
      Zn_chi = -n_period*int(n/2)*sin(n_period*int(n/2)*chi/F0)/F0
    else
      Zn_chi =  n_period*int(n/2)*cos(n_period*int(n/2)*chi/F0)/F0
    end if
  end function Zn_chi

  pure real*8 function Zm(m, theta)
    implicit none
    integer, intent(in) :: m
    real*8,  intent(in) :: theta

    if (mod(m,2) .eq. 1) then
      Zm = cos(int((m+1)/2)*theta)
    else
      Zm = sin(int((m+1)/2)*theta)
    end if
  end function Zm

  pure real*8 function Zm_tht(m, theta)
    implicit none
    integer, intent(in) :: m
    real*8,  intent(in) :: theta

    if (mod(m,2) .eq. 1) then
      Zm_tht = -int((m+1)/2)*sin(int((m+1)/2)*theta)
    else
      Zm_tht =  int((m+1)/2)*cos(int((m+1)/2)*theta)
    end if
  end function Zm_tht

  pure real*8 function Zm_tht_2(m, theta)
    implicit none
    integer, intent(in) :: m
    real*8,  intent(in) :: theta

    if (mod(m,2) .eq. 1) then
      Zm_tht_2 = -int((m+1)/2)**2*cos(int((m+1)/2)*theta)
    else
      Zm_tht_2 = -int((m+1)/2)**2*sin(int((m+1)/2)*theta)
    end if
  end function Zm_tht_2

  pure real*8 function HZn(n, phi)
    use mod_parameters, only: n_period
    implicit none
    integer, intent(in) :: n
    real*8,  intent(in) :: phi

    if (n .eq. 1) then
      HZn = 1.d0
      return
    end if

    if (mod(n,2) .eq. 0) then
      HZn = cos(n_period*int(n/2)*phi)
    else
      HZn = sin(n_period*int(n/2)*phi)
    end if
  end function HZn
end module mod_boundary_conditions
