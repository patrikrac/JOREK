!> This module is for the calculation of the vacuum magnetic scalar potential (chi) and its derivatives via the Dommaschk potentials
!! For more details, see
!! [*] W. Dommaschk, "Representations for vacuum potentials in stellarators", Computer Physics Communications 40, pg. 203 (1986)
!!
!! This module is used by stellarator models to represent the vacuum field.
module mod_chi
  use mod_parameters
  use phys_module, only: domm, dcoef, F0, R_domm, PI
  implicit none
  private 
  public init_chi_basis, get_chi, get_chi_domm, get_chi_corr, compute_chi_on_gauss_points

  integer, parameter :: m_tor = (n_coord_tor - 1)/2
  
  ! Cfunc(R) = Sum_{k=1,size(coef)} coef(k)*R**pwr(k) + Sum_{k=1,size(lcoef)} lcoef(k)*log(R)*R**lpwr(k)
  ! Cfunc(R) can be C^D_{m,l}(R), C^N_{m,l}(R) or a derivative of either of these two functions, see eqs (31) and (32) in [*]
  type type_Cfunc
    real*8,  dimension(:), allocatable :: coef, lcoef
    integer, dimension(:), allocatable :: pwr, lpwr
  end type type_Cfunc
  
  ! Bfunc(R,z) = Sum_{k=1,size(coef)} coef(k)*rfunc(k,R)*z**zpwr(k)
  ! Bfunc can be D_{m,l}(R,z), N_{m,l}(R,z) or a derivative of either of these two functions, see eq (2) in [*]
  ! size(coef) = int(l/2) + 1
  type type_Bfunc
    real*8,  dimension(:), allocatable :: coef
    integer, dimension(:), allocatable :: zpwr
    type(type_Cfunc), dimension(:), allocatable :: rfunc
  end type type_Bfunc
  
  type(type_Cfunc), dimension(0:l_pol_domm,0:m_tor), private :: CD, CN
  type(type_Bfunc), dimension(0:n_order-1,0:n_order-1,0:l_pol_domm,0:m_tor), private :: D, N
  
  contains
  
  subroutine init_chi_basis()
    implicit none
    integer :: i, m, l, k, j, i_ord, j_ord, clsz, csz
    real*8,  dimension(:), allocatable :: tmpcoef
    integer, dimension(:), allocatable :: tmppwr
    
    ! Construct the C^D_{m,l}(R) and C^N_{m,l}(R) functions
    do i=0,m_tor
      m = i*n_coord_period
      do l=0,l_pol_domm
        allocate(CD(l,i)%coef(2*(l+1))); allocate(CD(l,i)%lcoef(l+1))
        allocate(CD(l,i)%pwr(2*(l+1)));  allocate(CD(l,i)%lpwr(l+1))
        allocate(CN(l,i)%coef(2*(l+1))); allocate(CN(l,i)%lcoef(l+1))
        allocate(CN(l,i)%pwr(2*(l+1)));  allocate(CN(l,i)%lpwr(l+1))
        do k=0,l
          CD(l,i)%coef(2*k+1) = -(alpha(k,m)*(gamma_st(l-m-k,m) - alpha(l-m-k,m)) - gamma(k,m)*alpha_st(l-m-k,m) + alpha(k,m)*beta_st(l-k,m))
          CD(l,i)%pwr(2*k+1)  = 2*k + m
          CD(l,i)%coef(2*k+2) = beta(k,m)*alpha_st(l-k,m);     CD(l,i)%pwr(2*k+2) = 2*k - m
          CD(l,i)%lcoef(k+1)  = -alpha(k,m)*alpha_st(l-m-k,m); CD(l,i)%lpwr(k+1)  = 2*k + m
          
          CN(l,i)%coef(2*k+1) = alpha(k,m)*gamma(l-m-k,m) - gamma(k,m)*alpha(l-m-k,m) + alpha(k,m)*beta(l-k,m); CN(l,i)%pwr(2*k+1)  = 2*k + m
          CN(l,i)%coef(2*k+2) = -beta(k,m)*alpha(l-k,m);                                                CN(l,i)%pwr(2*k+2)  = 2*k - m
          CN(l,i)%lcoef(k+1)  = alpha(k,m)*alpha(l-m-k,m);                                              CN(l,i)%lpwr(k+1)   = 2*k + m
        end do
      end do
    end do
    
    ! Construct the D_{m,l}(R,z) and N_{m,l}(R,z) functions
    do i=0,m_tor
      do l=0,l_pol_domm
        allocate(D(0,0,l,i)%coef(l/2 + 1));  allocate(N(0,0,l,i)%coef(l/2 + 1))
        allocate(D(0,0,l,i)%zpwr(l/2 + 1));  allocate(N(0,0,l,i)%zpwr(l/2 + 1))
        allocate(D(0,0,l,i)%rfunc(l/2 + 1)); allocate(N(0,0,l,i)%rfunc(l/2 + 1))
        do k=0,l/2
          D(0,0,l,i)%coef(k+1)  = 1.d0/fact(l - 2*k); N(0,0,l,i)%coef(k+1)  = 1.d0/fact(l - 2*k)
          D(0,0,l,i)%zpwr(k+1)  = l - 2*k;            N(0,0,l,i)%zpwr(k+1)  = l - 2*k
          D(0,0,l,i)%rfunc(k+1) = CD(k,i);            N(0,0,l,i)%rfunc(k+1) = CN(k,i)
        end do
      end do
    end do
    
    ! Differentiate D and N with respect to R
    do i=0,m_tor
      do l=0,l_pol_domm
        do i_ord=1,n_order-1
          D(i_ord,0,l,i) = D(i_ord-1,0,l,i); N(i_ord,0,l,i) = N(i_ord-1,0,l,i)
          do k=0,l/2
            ! Differentiate D
            clsz = size(D(i_ord,0,l,i)%rfunc(k+1)%lcoef) ! Number of logarithm terms in the (k+1)th rfunc in D
            csz  = size(D(i_ord,0,l,i)%rfunc(k+1)%coef)  ! Number of regular polynomial terms in the (k+1)th rfunc in D
            ! Upon differentiation, a logarithm term produces a regular polynomial term and a logarithm term, due to the product rule
            ! Thus, the total number of polynomial terms increases by clsz
            allocate(tmpcoef(clsz+csz)); allocate(tmppwr(clsz+csz))
            tmpcoef(clsz+1:clsz+csz) = D(i_ord,0,l,i)%rfunc(k+1)%coef; tmppwr(clsz+1:clsz+csz) = D(i_ord,0,l,i)%rfunc(k+1)%pwr
            call move_alloc(tmpcoef,D(i_ord,0,l,i)%rfunc(k+1)%coef); call move_alloc(tmppwr,D(i_ord,0,l,i)%rfunc(k+1)%pwr)
            ! Differentiate logarithm terms
            do j=1,clsz
              ! Calculate polynomial terms resulting from logarithm term differentiation
              D(i_ord,0,l,i)%rfunc(k+1)%coef(j) = D(i_ord,0,l,i)%rfunc(k+1)%lcoef(j)
              D(i_ord,0,l,i)%rfunc(k+1)%pwr(j)  = D(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) - 1
              ! Calculate logarithm terms resulting from logarithm term differentiation
              if (D(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) .eq. 0) then
                D(i_ord,0,l,i)%rfunc(k+1)%lcoef(j) = 0.d0
              else
                D(i_ord,0,l,i)%rfunc(k+1)%lcoef(j) = D(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) * D(i_ord,0,l,i)%rfunc(k+1)%lcoef(j)
                D(i_ord,0,l,i)%rfunc(k+1)%lpwr(j)  = D(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) - 1
              end if
            end do
            ! Differentiate regular polynomial terms
            do j=clsz+1,clsz+csz
              if (D(i_ord,0,l,i)%rfunc(k+1)%pwr(j) .eq. 0) then
                D(i_ord,0,l,i)%rfunc(k+1)%coef(j) = 0.d0
              else
                D(i_ord,0,l,i)%rfunc(k+1)%coef(j) = D(i_ord,0,l,i)%rfunc(k+1)%pwr(j) * D(i_ord,0,l,i)%rfunc(k+1)%coef(j)
                D(i_ord,0,l,i)%rfunc(k+1)%pwr(j)  = D(i_ord,0,l,i)%rfunc(k+1)%pwr(j) - 1
              end if
            end do
            
            ! Differentiate N, same procedure as above for D
            clsz = size(N(i_ord,0,l,i)%rfunc(k+1)%lcoef)
            csz  = size(N(i_ord,0,l,i)%rfunc(k+1)%coef)
            allocate(tmpcoef(clsz+csz)); allocate(tmppwr(clsz+csz))
            tmpcoef(clsz+1:clsz+csz) = N(i_ord,0,l,i)%rfunc(k+1)%coef; tmppwr(clsz+1:clsz+csz) = N(i_ord,0,l,i)%rfunc(k+1)%pwr
            call move_alloc(tmpcoef,N(i_ord,0,l,i)%rfunc(k+1)%coef); call move_alloc(tmppwr,N(i_ord,0,l,i)%rfunc(k+1)%pwr)
            do j=1,clsz
              N(i_ord,0,l,i)%rfunc(k+1)%coef(j) = N(i_ord,0,l,i)%rfunc(k+1)%lcoef(j)
              N(i_ord,0,l,i)%rfunc(k+1)%pwr(j)  = N(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) - 1
              if (N(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) .eq. 0) then
                N(i_ord,0,l,i)%rfunc(k+1)%lcoef(j) = 0.d0
              else
                N(i_ord,0,l,i)%rfunc(k+1)%lcoef(j) = N(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) * N(i_ord,0,l,i)%rfunc(k+1)%lcoef(j)
                N(i_ord,0,l,i)%rfunc(k+1)%lpwr(j)  = N(i_ord,0,l,i)%rfunc(k+1)%lpwr(j) - 1
              end if
            end do
            do j=clsz+1,clsz+csz
              if (N(i_ord,0,l,i)%rfunc(k+1)%pwr(j) .eq. 0) then
                N(i_ord,0,l,i)%rfunc(k+1)%coef(j) = 0.d0
              else
                N(i_ord,0,l,i)%rfunc(k+1)%coef(j) = N(i_ord,0,l,i)%rfunc(k+1)%pwr(j) * N(i_ord,0,l,i)%rfunc(k+1)%coef(j)
                N(i_ord,0,l,i)%rfunc(k+1)%pwr(j)  = N(i_ord,0,l,i)%rfunc(k+1)%pwr(j) - 1
              end if
            end do
          end do
        end do
      end do
    end do
    
    ! Differentiate D and N with respect to z
    do i=0,m_tor
      do l=0,l_pol_domm
        do j_ord=1,n_order-1
          do i_ord=0,n_order-1
            D(i_ord,j_ord,l,i) = D(i_ord,j_ord-1,l,i); N(i_ord,j_ord,l,i) = N(i_ord,j_ord-1,l,i)
            do k=0,l/2
              if (D(i_ord,j_ord,l,i)%zpwr(k+1) .eq. 0) then
                D(i_ord,j_ord,l,i)%coef(k+1) = 0.d0
              else
                D(i_ord,j_ord,l,i)%coef(k+1) = D(i_ord,j_ord,l,i)%zpwr(k+1) * D(i_ord,j_ord,l,i)%coef(k+1)
                D(i_ord,j_ord,l,i)%zpwr(k+1) = D(i_ord,j_ord,l,i)%zpwr(k+1) - 1
              end if
              if (N(i_ord,j_ord,l,i)%zpwr(k+1) .eq. 0) then
                N(i_ord,j_ord,l,i)%coef(k+1) = 0.d0
              else
                N(i_ord,j_ord,l,i)%coef(k+1) = N(i_ord,j_ord,l,i)%zpwr(k+1) * N(i_ord,j_ord,l,i)%coef(k+1)
                N(i_ord,j_ord,l,i)%zpwr(k+1) = N(i_ord,j_ord,l,i)%zpwr(k+1) - 1
              end if
            end do
          end do
        end do
      end do
    end do  
  end subroutine init_chi_basis
    
  ! Eq (27) in [*]
  pure real*8 function alpha(n,m)
    implicit none
    integer, intent(in) :: n, m
    
    if (n .lt. 0) then
      alpha = 0.0
    else
      alpha = (-1.d0)**n/(fact(n+m)*fact(n)*2.d0**(2*n+m))
    end if
  end function alpha
  
  pure real*8 function alpha_st(n, m)
    implicit none
    integer, intent(in) :: n, m
    
    alpha_st = (2*n + m)*alpha(n,m)
  end function alpha_st
  
  ! Eq (28) in [*]
  pure real*8 function beta(n,m)
    implicit none
    integer, intent(in) :: n, m
    integer :: pwr
    
    if (n .lt. 0 .or. n .ge. m) then
      beta = 0.0
    else
      pwr = 2*n-m+1
      beta = fact(m-n-1)/(fact(n)*(2.d0**pwr))
    end if
  end function beta
  
  pure real*8 function beta_st(n,m)
    implicit none
    integer, intent(in) :: n, m
    
    beta_st = (2*n - m)*beta(n,m)
  end function beta_st
  
  ! Eq (33) in [*]
  pure real*8 function gamma(n,m)
    implicit none
    integer, intent(in) :: n, m
    integer             :: i
    
    gamma = 0.0
    do i=1,n
      gamma = gamma + 1.0/i + 1.0/(m+i)
    end do
    gamma = gamma*alpha(n,m)/2.0
  end function gamma
  
  pure real*8 function gamma_st(n,m)
    implicit none
    integer, intent(in) :: n, m
    
    gamma_st = (2*n + m)*gamma(n,m)
  end function gamma_st
  
  !> Wrapper function for calculating the vacuum magnetic field
  !!
  !! This routine will call get_chi_domm and depending on compile time parameters either add the correction term
  !! get_chi_corr to the vacuum field, or keep the representation as the Dommaschk potentials alone
  pure function get_chi(R,z,phi,node_list,element_list,i_elm,s,t,max_ord)
    use data_structure,     only: type_node_list, type_element_list
    implicit none
    real*8,  intent(in)                   :: R, z, phi
    type (type_node_list),    intent(in)  :: node_list
    type (type_element_list), intent(in)  :: element_list
    integer, intent(in)                   :: i_elm
    real*8,  intent(in)                   :: s, t
    integer, optional, intent(in)         :: max_ord
    real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: get_chi
    
    get_chi = get_chi_domm(R,z,phi,max_ord)
#ifndef USE_DOMM
    get_chi = get_chi + get_chi_corr(node_list, element_list, i_elm, s, t, phi)
#endif
  endfunction get_chi

  !> This function returns the vacuum scalar magnetic potential (chi) and its derivatives up to n_order-1,
  !!  unless a lower cutoff is requested via n
  !! 
  !! The Dommaschk potentials are calculated from EXTENDER, which uses a lefthand (LH) coordinate 
  !!  system. In JOREK, the coordinate system is righthanded (RH), and so the equation for the  
  !!  Dommaschk potential needs to be modified to:
  !!
  !!  Chi_RH(R, Z, phi_RH) = Chi_LH(R, Z, 2*pi/N_p - phi_RH)
  !!                         = 2*pi/N_p-phi_RH + Sum_m,l [a_{m,l} cos(m phi_RH) - b_{m,l} sin(m phi_RH)] D_{m,l}
  !!                                                    +[c_{m,l} cos(m phi_RH) - d_{m,l} sin(m phi_RH)] N_{m,l}
  !!
  !! This leads to the equations for Chi and its derivatives below.
  !!
  !! If this function is called from a tokamak model, the routine returns the dominant background toroidal field,
  !! where F0 is assumed to be defined using a RH coordinate system.
  pure function get_chi_domm(R,z,phi,max_ord)
    implicit none
    real*8,  intent(in)                   :: R, z, phi
    integer, optional, intent(in)         :: max_ord
    real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: get_chi_domm
    real*8, dimension(0:n_order-1) :: dksinmp, dkcosmp, V_ml
    real*8  :: Rn, zn, cval, D_ml, N_ml_1
    integer :: n_ord, i, j, k, m, l, i_ord, j_ord, k_ord
    
    get_chi_domm = 0.d0
#if STELLARATOR_MODEL
    get_chi_domm(0,0,0) = 2 * PI / float(n_coord_period) - phi; get_chi_domm(0,0,1) = -1 ! Include the phi term
    
#ifdef USE_DOMM
    if (domm) then
      n_ord = n_order-1
      if (present(max_ord)) then
        if (max_ord .lt. n_order-1) n_ord = max_ord
      end if
      
      Rn = R/R_domm
      zn = z/R_domm
      do i=0,m_tor
        m = i*n_coord_period
        do k_ord=0,n_ord
          ! k_ord-order derivatives of sin(m*phi) and cos(m*phi)
          dksinmp(k_ord) = (-1)**int(k_ord/2+1)*m**k_ord*(((1+(-1)**k_ord)/2)*sin(m*phi) + ((1-(-1)**k_ord)/2)*cos(m*phi))
          dkcosmp(k_ord) = (-1)**int((k_ord+1)/2)*m**k_ord*(((1+(-1)**k_ord)/2)*cos(m*phi) + ((1-(-1)**k_ord)/2)*sin(m*phi))
        end do
        do l=0,l_pol_domm
          do j_ord=0,n_ord
            do i_ord=0,n_ord
              D_ml = 0.d0; N_ml_1 = 0.d0
              do k=0,l/2
                cval = 0.d0
                do j=1,size(D(i_ord,j_ord,l,i)%rfunc(k+1)%lcoef)
                  cval = cval + D(i_ord,j_ord,l,i)%rfunc(k+1)%lcoef(j)*log(Rn)*Rn**D(i_ord,j_ord,l,i)%rfunc(k+1)%lpwr(j)
                end do
                do j=1,size(D(i_ord,j_ord,l,i)%rfunc(k+1)%coef)
                  cval = cval + D(i_ord,j_ord,l,i)%rfunc(k+1)%coef(j)*Rn**D(i_ord,j_ord,l,i)%rfunc(k+1)%pwr(j)
                end do
                D_ml = D_ml + D(i_ord,j_ord,l,i)%coef(k+1)*cval*zn**D(i_ord,j_ord,l,i)%zpwr(k+1)
              end do
              do k=0,(l-1)/2
                cval = 0.d0
                do j=1,size(N(i_ord,j_ord,abs(l-1),i)%rfunc(k+1)%lcoef)
                  cval = cval + N(i_ord,j_ord,abs(l-1),i)%rfunc(k+1)%lcoef(j)*log(Rn)*Rn**N(i_ord,j_ord,abs(l-1),i)%rfunc(k+1)%lpwr(j)
                end do
                do j=1,size(N(i_ord,j_ord,abs(l-1),i)%rfunc(k+1)%coef)
                  cval = cval + N(i_ord,j_ord,abs(l-1),i)%rfunc(k+1)%coef(j)*Rn**N(i_ord,j_ord,abs(l-1),i)%rfunc(k+1)%pwr(j)
                end do
                N_ml_1 = N_ml_1 + N(i_ord,j_ord,abs(l-1),i)%coef(k+1)*cval*zn**N(i_ord,j_ord,abs(l-1),i)%zpwr(k+1)
              end do
              do k_ord=0,n_ord
                V_ml(k_ord) = (dcoef(1,l,i)*dkcosmp(k_ord) + dcoef(2,l,i)*dksinmp(k_ord))*D_ml &
                            + (dcoef(3,l,i)*dkcosmp(k_ord) + dcoef(4,l,i)*dksinmp(k_ord))*N_ml_1
              end do
              get_chi_domm(i_ord,j_ord,:) = get_chi_domm(i_ord,j_ord,:) + V_ml/(R_domm**(i_ord+j_ord)) ! R_domm due to chain rule (derivatives wrt R, z, not Rn, zn)
            end do
          end do
        end do
      end do
    end if
#endif

#else
    get_chi_domm(0,0,0) = phi; get_chi_domm(0,0,1) = 1
#endif

    get_chi_domm = F0*get_chi_domm
  end function get_chi_domm
  
  !>  This function returns a correction to the vacuum scalar magnetic potential (chi) and its derivatives such 
  !!  that n.grad(chi) on the simulation boundary is equal to 0
  pure function get_chi_corr(node_list, element_list, i_elm, s, t, phi)
    use phys_module,        only: mode_coord, n_tht
    use data_structure,     only: type_node_list, type_element_list
    use mod_interp,         only: interp_RZP, interp_gvec
    implicit none
    type (type_node_list),    intent(in)  :: node_list
    type (type_element_list), intent(in)  :: element_list
    integer, intent(in) :: i_elm
    real*8,  intent(in) :: s, t, phi
    real*8, dimension(0:n_order-1,0:n_order-1,0:n_order-1) :: get_chi_corr
    integer :: i_harm, i_tor
    real*8  :: R, R_s, R_t, R_p, R_ss, R_tt, R_st, R_pp, R_sp, R_tp
    real*8  :: Z, Z_s, Z_t, Z_p, Z_ss, Z_tt, Z_st, Z_pp, Z_sp, Z_tp
    real*8  :: chi_corr, chi_corr_s, chi_corr_t, chi_corr_p, chi_corr_ss, chi_corr_tt, chi_corr_st, chi_corr_sp, chi_corr_tp, chi_corr_pp
    real*8  :: chi_corr_harm, chi_corr_harm_s, chi_corr_harm_t, chi_corr_harm_ss, chi_corr_harm_tt, chi_corr_harm_st
    real*8  :: chi_corr_px, chi_corr_py
    real*8  :: chi_x, chi_y, chi_p, chi_xx, chi_xy, chi_yy, chi_yp, chi_xp, chi_pp
    real*8  :: xjac, xjac_s, xjac_x, xjac_y, x_p_x, x_p_y, y_p_x, y_p_y
    
    ! Get R, Z, phi geometry of point
    call interp_RZP(node_list,element_list,i_elm,s,t,phi,R,R_s,R_t,R_p,R_st,R_ss,R_tt,R_sp,R_tp,R_pp, &
                                                         Z,Z_s,Z_t,Z_p,Z_st,Z_ss,Z_tt,Z_sp,Z_tp,Z_pp)
    
    ! Compute chi and s, t, phi derivatives
    call interp_gvec(node_list,element_list,i_elm,5,1,1,s,t,chi_corr,chi_corr_s,chi_corr_t,chi_corr_st,chi_corr_ss,chi_corr_tt)
    chi_corr_p = 0.0; chi_corr_sp = 0.0; chi_corr_tp = 0.0; chi_corr_pp = 0.0
    do i_tor=1,(n_coord_tor-1)/2
      i_harm = 2*i_tor
      
      call interp_gvec(node_list,element_list,i_elm,5,1,i_harm,s,t,chi_corr_harm, chi_corr_harm_s, chi_corr_harm_t, &
                                                                   chi_corr_harm_st,chi_corr_harm_ss,chi_corr_harm_tt)
      chi_corr     = chi_corr    + chi_corr_harm    * cos(mode_coord(i_harm)*phi)
      chi_corr_s   = chi_corr_s  + chi_corr_harm_s  * cos(mode_coord(i_harm)*phi)
      chi_corr_t   = chi_corr_t  + chi_corr_harm_t  * cos(mode_coord(i_harm)*phi)
      chi_corr_st  = chi_corr_st + chi_corr_harm_st * cos(mode_coord(i_harm)*phi)
      chi_corr_ss  = chi_corr_ss + chi_corr_harm_ss * cos(mode_coord(i_harm)*phi)
      chi_corr_tt  = chi_corr_tt + chi_corr_harm_tt * cos(mode_coord(i_harm)*phi)
      chi_corr_p   = chi_corr_p  - chi_corr_harm    * mode_coord(i_harm) * sin(mode_coord(i_harm)*phi)
      chi_corr_sp  = chi_corr_sp - chi_corr_harm_s  * mode_coord(i_harm) * sin(mode_coord(i_harm)*phi)
      chi_corr_tp  = chi_corr_tp - chi_corr_harm_t  * mode_coord(i_harm) * sin(mode_coord(i_harm)*phi)
      chi_corr_pp  = chi_corr_pp - chi_corr_harm    * mode_coord(i_harm)**2 * cos(mode_coord(i_harm)*phi)
      
      call interp_gvec(node_list,element_list,i_elm,5,1,i_harm+1,s,t,chi_corr_harm, chi_corr_harm_s, chi_corr_harm_t,  &
                                                                     chi_corr_harm_st,chi_corr_harm_ss,chi_corr_harm_tt)
      chi_corr     = chi_corr    - chi_corr_harm    * sin(mode_coord(i_harm)*phi)
      chi_corr_s   = chi_corr_s  - chi_corr_harm_s  * sin(mode_coord(i_harm)*phi)
      chi_corr_t   = chi_corr_t  - chi_corr_harm_t  * sin(mode_coord(i_harm)*phi)
      chi_corr_st  = chi_corr_st - chi_corr_harm_st * sin(mode_coord(i_harm)*phi)
      chi_corr_ss  = chi_corr_ss - chi_corr_harm_ss * sin(mode_coord(i_harm)*phi)
      chi_corr_tt  = chi_corr_tt - chi_corr_harm_tt * sin(mode_coord(i_harm)*phi)
      chi_corr_p   = chi_corr_p  - chi_corr_harm    * mode_coord(i_harm) * cos(mode_coord(i_harm)*phi)
      chi_corr_sp  = chi_corr_sp - chi_corr_harm_s  * mode_coord(i_harm) * cos(mode_coord(i_harm)*phi)
      chi_corr_tp  = chi_corr_tp - chi_corr_harm_t  * mode_coord(i_harm) * cos(mode_coord(i_harm)*phi)
      chi_corr_pp  = chi_corr_pp + chi_corr_harm    * mode_coord(i_harm)**2 * sin(mode_coord(i_harm)*phi)
    end do

    ! Compute chi and R, Z, phi derivatives
    xjac =  R_s*Z_t - R_t*Z_s
    ! Use L'Hopital's rule (differentiate numerator and denominator wrt s) if the derivatives are being evaluated on the axis
    if (i_elm .le. n_tht .and. s .eq. 0.d0) then
      xjac_s = R_s*Z_st - R_st*Z_s
      chi_x = ( Z_st*chi_corr_s - Z_s*chi_corr_st)/xjac_s
      chi_y = (-R_st*chi_corr_s + R_s*chi_corr_st)/xjac_s
    else ! If not, then we use the standard expressions for chi_x and chi_y
      chi_x = (   Z_t * chi_corr_s - Z_s * chi_corr_t ) / xjac
      chi_y = ( - R_t * chi_corr_s + R_s * chi_corr_t ) / xjac
    end if
    chi_p = chi_corr_p - chi_x * R_p - chi_y * Z_p

    xjac_x  = (R_ss*Z_t**2 - Z_ss*R_t*Z_t - 2.d0*R_st*Z_s*Z_t &
            + Z_st*(R_s*Z_t + R_t*Z_s)                                                      &
            + R_tt*Z_s**2 - Z_tt*R_s*Z_s) / xjac
    xjac_y  = (Z_tt*R_s**2 - R_tt*Z_s*R_s - 2.d0*Z_st*R_t*R_s &
            + R_st*(Z_t*R_s + Z_s*R_t)                                                      &
            + Z_ss*R_t**2 - R_ss*Z_t*R_t) / xjac
    x_p_x = (R_sp*Z_t - R_tp*Z_s)/xjac
    x_p_y = (R_tp*R_s - R_sp*R_t)/xjac
    y_p_x = (Z_sp*Z_t - Z_tp*Z_s)/xjac
    y_p_y = (Z_tp*R_s - Z_sp*R_t)/xjac
    chi_xx              = (chi_corr_ss*Z_t**2 - 2.d0*chi_corr_st*Z_s*Z_t &
                        + chi_corr_tt*Z_s**2                                                       &
                        + chi_corr_s*(Z_st*Z_t - Z_tt*Z_s)           &
                        + chi_corr_t*(Z_st*Z_s - Z_ss*Z_t))/xjac**2  &
                        - xjac_x*(chi_corr_s*Z_t - chi_corr_t*Z_s)/xjac**2
    chi_yy              = (chi_corr_ss*R_t**2 - 2.d0*chi_corr_st*R_s*R_t &
                        + chi_corr_tt*R_s**2                                                       &
                        + chi_corr_s*(R_st*R_t - R_tt*R_s)           &
                        + chi_corr_t*(R_st*R_s - R_ss*R_t))/xjac**2  &
                        - xjac_y*(-chi_corr_s*R_t + chi_corr_t*R_s)/xjac**2
    chi_xy              = (-chi_corr_ss*Z_t*R_t - chi_corr_tt*R_s*Z_s &
                        + chi_corr_st*(Z_s*R_t + Z_t*R_s)                   &
                        - chi_corr_s*(R_st*Z_t - R_tt*Z_s)                  &
                        - chi_corr_t*(R_st*Z_s - R_ss*Z_t))/xjac**2         &
                        - xjac_x*(-chi_corr_s*R_t + chi_corr_t*R_s)/xjac**2
    chi_corr_px               = (Z_t*chi_corr_sp - Z_s*chi_corr_tp)/xjac
    chi_corr_py               = (-R_t*chi_corr_sp + R_s*chi_corr_tp)/xjac
    chi_pp              = chi_corr_pp - R_pp*chi_x - 2.d0*(R_p*chi_corr_px + Z_p*chi_corr_py)            &
                         - Z_pp*chi_y + (R_p*x_p_x*chi_x + R_p*y_p_x*chi_y&
                         + Z_p*x_p_y*chi_x + Z_p*y_p_y*chi_y) + R_p**2*chi_xx   &
                         + 2.d0*R_p*Z_p*chi_xy + Z_p**2*chi_yy
    chi_xp              = chi_corr_px - x_p_x*chi_x - R_p*chi_xx - y_p_x*chi_y - Z_p*chi_xy
    chi_yp              = chi_corr_py - x_p_y*chi_x - R_p*chi_xy - y_p_y*chi_y - Z_p*chi_yy
    
    get_chi_corr        = 0.0
    get_chi_corr(0,0,0) = get_chi_corr(0,0,0) + chi_corr
    get_chi_corr(1,0,0) = get_chi_corr(1,0,0) + chi_x
    get_chi_corr(0,1,0) = get_chi_corr(0,1,0) + chi_y
    get_chi_corr(0,0,1) = get_chi_corr(0,0,1) + chi_p
    get_chi_corr(2,0,0) = get_chi_corr(2,0,0) + chi_xx
    get_chi_corr(0,2,0) = get_chi_corr(0,2,0) + chi_yy
    get_chi_corr(0,0,2) = get_chi_corr(0,0,2) + chi_pp
    get_chi_corr(1,1,0) = get_chi_corr(1,1,0) + chi_xy
    get_chi_corr(1,0,1) = get_chi_corr(1,0,1) + chi_xp
    get_chi_corr(0,1,1) = get_chi_corr(0,1,1) + chi_yp
  end function get_chi_corr

  !>---------------------------
  !! Factorial of n
  !!---------------------------
  pure real*8 function fact(n)
    implicit none
    integer, intent(in) :: n
    integer             :: i
    
    fact = 1.0
    do i=2,n
      fact = fact*i
    end do
  end function fact
  
  !>-----------------------------------------------------------------------------------------------------
  !! mod_chi::get_chi is used to compute the vacuum field representation on all gaussian points in all
  !! planes of the simulated configuration to avoid their repeated calculation.
  !!-----------------------------------------------------------------------------------------------------
  subroutine compute_chi_on_gauss_points(my_id, element_list,node_list, local_elms, n_local_elms)
    use basis_at_gaussian
    use data_structure

    implicit none
  
    ! --- Routine parameters
    integer, intent(in)                    :: my_id
    type(type_node_list),    intent(in)    :: node_list
    type(type_element_list), intent(inout) :: element_list
    integer, intent(in)                    :: local_elms(*)
    integer, intent(in)                    :: n_local_elms

    ! --- Variables for chi calculation
    integer                                    :: i_elm, i_elm_loc, i_vertex, i_node, i, j, mp, ms, mt, i_tor
    real*8                                     :: phi
    real*8, dimension(n_plane,n_gauss,n_gauss) :: x_g, x_s, x_t, x_p, x_ss, x_tt, x_st, x_pp, x_sp, x_tp
    real*8, dimension(n_plane,n_gauss,n_gauss) :: y_g, y_s, y_t, y_p, y_ss, y_tt, y_st, y_pp, y_sp, y_tp
    type (type_element)                        :: element
    type (type_node)                           :: nodes(n_vertex_max)
    real*8, dimension(n_plane,n_gauss,n_gauss,0:n_order-1,0:n_order-1,0:n_order-1) :: chi
 
#if STELLARATOR_MODEL
    write (*,*) 
    if (my_id .eq. 0) then
      write (*,*) "Storing vacuum field on gaussian points..."
      write (*,*) 
    endif
    
    ! --- Declare shared and private variables for omp
    !$omp parallel default(none) &
    !$omp   shared(element_list,node_list, H, H_s, H_t, H_ss, H_tt, H_st, HZ_coord, HZ_coord_p, HZ_coord_pp, local_elms, n_local_elms)  &
    !$omp   private(i_elm, i_elm_loc,i_vertex,i_node,i_tor,element, i, j, ms, mt, mp,                                             &
    !$omp           x_g, x_s, x_t, x_p, x_ss, x_tt, x_st, x_pp, x_sp, x_tp,                                                             &
    !$omp           y_g, y_s, y_t, y_p, y_ss, y_tt, y_st, y_pp, y_sp, y_tp, chi, phi)                                                   &
    !$omp   firstprivate(nodes) !< so that these nodes are unallocated at the start of the omp region and can be explicitly allocated/deallocated 

    
    !$omp do schedule(runtime)
    do i_elm_loc = 1, n_local_elms
      i_elm = local_elms(i_elm_loc)
      element = element_list%element(i_elm)
      !$omp critical
      allocate(element_list%element(i_elm)%chi(n_plane,n_gauss,n_gauss,0:n_order-1,0:n_order-1,0:n_order-1))
      !$omp end critical
      do i_vertex = 1, n_vertex_max
        i_node     = element%vertex(i_vertex)
        call make_deep_copy_node(node_list%node(i_node), nodes(i_vertex))
      enddo
      
      x_g  = 0.d0; x_s   = 0.d0; x_t   = 0.d0; x_p = 0.d0; x_st  = 0.d0; x_ss  = 0.d0; x_tt  = 0.d0; x_sp = 0.d0; x_tp = 0.d0; x_pp = 0.d0;
      y_g  = 0.d0; y_s   = 0.d0; y_t   = 0.d0; y_p = 0.d0; y_st  = 0.d0; y_ss  = 0.d0; y_tt  = 0.d0; y_sp = 0.d0; y_tp = 0.d0; y_pp = 0.d0;
      do i=1,n_vertex_max
        do j=1,n_order+1
          do ms=1, n_gauss
            do mt=1, n_gauss
              do mp=1,n_plane
                do i_tor=1,n_coord_tor
                  x_g(mp,ms,mt)  = x_g(mp,ms,mt)  + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord(i_tor,mp)
                  x_s(mp,ms,mt)  = x_s(mp,ms,mt)  + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord(i_tor,mp)
                  x_t(mp,ms,mt)  = x_t(mp,ms,mt)  + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord(i_tor,mp)
                  x_p(mp,ms,mt)  = x_p(mp,ms,mt)  + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_p(i_tor,mp)
                  x_ss(mp,ms,mt) = x_ss(mp,ms,mt) + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H_ss(i,j,ms,mt) * HZ_coord(i_tor,mp)
                  x_st(mp,ms,mt) = x_st(mp,ms,mt) + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H_st(i,j,ms,mt) * HZ_coord(i_tor,mp)
                  x_tt(mp,ms,mt) = x_tt(mp,ms,mt) + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H_tt(i,j,ms,mt) * HZ_coord(i_tor,mp)
                  x_sp(mp,ms,mt) = x_sp(mp,ms,mt) + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord_p(i_tor,mp)
                  x_tp(mp,ms,mt) = x_tp(mp,ms,mt) + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord_p(i_tor,mp)
                  x_pp(mp,ms,mt) = x_pp(mp,ms,mt) + nodes(i)%x(i_tor,j,1) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_pp(i_tor,mp)
                  
                  y_g(mp,ms,mt)  = y_g(mp,ms,mt)  + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord(i_tor,mp)
                  y_s(mp,ms,mt)  = y_s(mp,ms,mt)  + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord(i_tor,mp)
                  y_t(mp,ms,mt)  = y_t(mp,ms,mt)  + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord(i_tor,mp)
                  y_p(mp,ms,mt)  = y_p(mp,ms,mt)  + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_p(i_tor,mp)
                  y_ss(mp,ms,mt) = y_ss(mp,ms,mt) + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H_ss(i,j,ms,mt) * HZ_coord(i_tor,mp)
                  y_st(mp,ms,mt) = y_st(mp,ms,mt) + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H_st(i,j,ms,mt) * HZ_coord(i_tor,mp)
                  y_tt(mp,ms,mt) = y_tt(mp,ms,mt) + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H_tt(i,j,ms,mt) * HZ_coord(i_tor,mp)
                  y_sp(mp,ms,mt) = y_sp(mp,ms,mt) + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ_coord_p(i_tor,mp)
                  y_tp(mp,ms,mt) = y_tp(mp,ms,mt) + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ_coord_p(i_tor,mp)
                  y_pp(mp,ms,mt) = y_pp(mp,ms,mt) + nodes(i)%x(i_tor,j,2) * element%size(i,j) * H(i,j,ms,mt)    * HZ_coord_pp(i_tor,mp)
                enddo ! i_tor
              enddo ! mp
            enddo ! mt
          enddo ! ms
        enddo ! j
      enddo ! i
      do ms=1, n_gauss
        do mt=1, n_gauss
          do mp=1,n_plane
            phi = 2.d0*PI*float(mp-1)/float(n_plane) / float(n_period)
            ! Get Dommaschk representation
            chi(mp,ms,mt,:,:,:) = get_chi(x_g(mp,ms,mt), y_g(mp,ms,mt), phi, node_list, element_list, i_elm, xgauss(ms), xgauss(mt))
          enddo ! mp
        enddo ! mt
      enddo ! ms

      !$omp critical
      element_list%element(i_elm)%chi(:,:,:,:,:,:) = chi(:,:,:,:,:,:)
      !$omp end critical
    enddo ! i_elm_loc
    !$omp end do

    do i_vertex = 1, n_vertex_max
      call dealloc_node(nodes(i_vertex))
    enddo
    !$omp end parallel
#else
  write(*,*) 'This function should not be called for tokamak models!'
  stop
#endif

  end subroutine compute_chi_on_gauss_points

end module mod_chi
