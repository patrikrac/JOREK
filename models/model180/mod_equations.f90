module mod_equations
  use mod_semianalytical
  use mod_parameters
  use data_structure, only: nbthreads
  implicit none

  type type_thread_eq

    ! Indices in eq array: variable index (see algexpr's below), R derivative order, z derivative order, phi derivative order,
    !   separation of terms with covariant phi derivatives in test function and unknown (FFT, not used in this model)
    real*8, dimension(:,:,:,:,:), allocatable :: eq
  end type type_thread_eq

  ! Indices
  integer, parameter :: var_v         = n_var+1
  integer, parameter :: var_varStar   = n_var+2
  integer, parameter :: var_chi       = n_var+3
  integer, parameter :: var_R         = n_var+4
  integer, parameter :: var_p0_gvec   = n_var+5
  integer, parameter :: var_B0x_gvec  = n_var+6
  integer, parameter :: var_B0y_gvec  = n_var+7
  integer, parameter :: var_B0p_gvec  = n_var+8
  integer, parameter :: var_Bv2       = n_var+9
  integer, parameter :: var_heaviside = n_var+10

  ! Values
  type(algexpr), parameter, private :: Psi0       = algexpr(basic=.true.,var=var_Psi)
  type(algexpr), parameter, private :: Phi0       = algexpr(basic=.true.,var=var_Phi)
  type(algexpr), parameter, private :: zj0        = algexpr(basic=.true.,var=var_zj)
  type(algexpr), parameter, private :: w0         = algexpr(basic=.true.,var=var_w)
  type(algexpr), parameter, private :: rho0       = algexpr(basic=.true.,var=var_rho)
  type(algexpr), parameter, private :: T0         = algexpr(basic=.true.,var=var_T)
  type(algexpr), parameter, private :: vpar0      = algexpr(basic=.true.,var=var_vpar)
  type(algexpr), parameter, private :: T0_i       = algexpr(basic=.true.,var=var_Ti)
  type(algexpr), parameter, private :: T0_e       = algexpr(basic=.true.,var=var_Te)
  ! Test function
  type(algexpr), parameter, private :: v          = algexpr(basic=.true.,var=var_v)
  ! Unknowns
  type(algexpr), parameter, private :: Psi        = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: Phi        = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: zj         = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: w          = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: rho        = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: T          = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: vpar       = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: T_i        = algexpr(basic=.true.,var=var_varStar)
  type(algexpr), parameter, private :: T_e        = algexpr(basic=.true.,var=var_varStar)

  type(algexpr), parameter, private :: heaviside  = algexpr(basic=.true.,var=var_heaviside)
  ! Other quantities
  type(algexpr), parameter, private :: chi        = algexpr(basic=.true.,var=var_chi)
  type(algexpr), parameter, private :: R          = algexpr(basic=.true.,var=var_R)
  ! Quantities imported from GVEC
  type(algexpr), parameter, private :: p0_gvec    = algexpr(basic=.true.,var=var_p0_gvec)
  type(algexpr), parameter, private :: B0x_gvec   = algexpr(basic=.true.,var=var_B0x_gvec)
  type(algexpr), parameter, private :: B0y_gvec   = algexpr(basic=.true.,var=var_B0y_gvec)
  type(algexpr), parameter, private :: B0p_gvec   = algexpr(basic=.true.,var=var_B0p_gvec)
 ! Auxiliary variables (aux)
  type(algexpr), parameter, private :: Bv2        = algexpr(basic=.true.,var=var_Bv2)

  type(algexpr), dimension(n_var), public        :: rhs_semianalytic
  type(algexpr), dimension(n_var, n_var), public :: amat_semianalytic
  type(algexpr), private :: a_Bv2

  integer, parameter :: n_aux = 4

  type(algexpr), private :: ea_Bv2x, ea_Bv2y, ea_Bv2p

  type(type_thread_eq), dimension(:), allocatable, target :: thread_eq

  contains

  subroutine init_equations()
    implicit none
 
    !###################################################################################################
    !#  Auxiliary vacuum magnetic field                                                                #
    !###################################################################################################
    a_Bv2 = dx(chi)*dx(chi) + dy(chi)*dy(chi) + dp(chi)*dp(chi)/(R*R)

    !###################################################################################################
    !#  Current Definition Equation for Psi                                                            #
    !###################################################################################################
    rhs_semianalytic(var_Psi) = (-Bv2)*inprod(v,Psi0)

    amat_semianalytic(var_Psi, var_Psi) = Bv2*inprod(v,Psi)
    amat_semianalytic(var_Psi, var_zj)  = v*Bv2*zj
 
    !###################################################################################################
    !#  Current Definition Equation for zj                                                             #
    ! Integration by parts to avoid first order derivatives in curl of B
    rhs_semianalytic(var_zj)  = (-dx(v*heaviside)*(dy(chi)*B0p_gvec - dp(chi)*B0y_gvec/R)   &
                              +   dy(v*heaviside)*(dx(chi)*B0p_gvec - dp(chi)*B0x_gvec/R)   &
                              -   dp(v*heaviside)*(dx(chi)*B0y_gvec - dy(chi)*B0x_gvec)/R)
    rhs_semianalytic(var_zj) = Dexpand(deepcopy(rhs_semianalytic(var_zj)))

    amat_semianalytic( var_zj,  var_zj) = v*Bv2*zj
    
    !###################################################################################################
    !#  Dummy Equations                                                                                #
    !###################################################################################################
    amat_semianalytic(var_Phi, var_Phi) = v*Phi

    amat_semianalytic(  var_w,   var_w) = v*w

    amat_semianalytic(var_rho, var_rho) = v*rho
   
    if (with_TiTe) then
      amat_semianalytic(var_Ti, var_Ti) = v*T_i
      amat_semianalytic(var_Te, var_Te) = v*T_e
    else
      amat_semianalytic( var_T,  var_T) = v*T
    end if

    if (with_vpar) then
      amat_semianalytic(var_vpar, var_vpar) = v*vpar
    endif
    
    ! Expansion of differential operators
    ea_Bv2x = Dexpand(deepcopy(dx(a_Bv2))); ea_Bv2y = Dexpand(deepcopy(dy(a_Bv2))); ea_Bv2p = Dexpand(deepcopy(dp(a_Bv2)))
  end subroutine init_equations

  subroutine init_eq_struct()
    use data_structure, only: nbthreads
    implicit none
    integer :: i

    if (.not. allocated(thread_eq)) then
      allocate(thread_eq(nbthreads))
      do i=1,nbthreads
        allocate(thread_eq(i)%eq(n_var+10,0:n_order-1,0:n_order-1,0:n_order-1,4))
      end do
    end if
  end subroutine init_eq_struct
  
  subroutine get_varnames(varnames)
    implicit none
    character(8), dimension(n_var), intent(out) :: varnames
      
    varnames(var_Psi)    = " var_Psi"
    varnames(var_Phi)    = " var_Phi"
    varnames( var_zj)    = "  var_zj"
    varnames(  var_w)    = "   var_w"
    varnames(var_rho)    = " var_rho"
    if (with_vpar) then
      varnames(var_vpar) = "var_vpar"
    endif
    if (with_TiTe) then
      varnames( var_Ti)    = "  var_Ti"
      varnames( var_Te)    = "  var_Te"
    else
      varnames( var_T)     = "   var_T"
    endif
  end subroutine get_varnames
  
  subroutine get_aux(aux,varnames)
    implicit none
    type(algexpr), dimension(n_aux), intent(out) :: aux
    character(19), dimension(n_aux), intent(out) :: varnames
    integer      :: i
    character(2) :: num
  
    aux = (/ a_Bv2, ea_Bv2x, ea_Bv2y, ea_Bv2p /)
    varnames = (/ "eq(var_Bv2,0,0,0,1)", "eq(var_Bv2,1,0,0,1)", "eq(var_Bv2,0,1,0,1)", "eq(var_Bv2,0,0,1,1)" /)
  end subroutine get_aux

  type(algexpr) function Bv_pbrack(a,b)
    implicit none
    type(algexpr), intent(in) :: a, b

    Bv_pbrack = ((dy(a)*dp(b) - dp(a)*dy(b))*dx(chi) + (dp(a)*dx(b) - dx(a)*dp(b))*dy(chi) + (dx(a)*dy(b) - dy(a)*dx(b))*dp(chi))/R
  end function Bv_pbrack

  type(algexpr) function Bv_parderiv(a)
    implicit none
    type(algexpr), intent(in) :: a

    Bv_parderiv = dx(a)*dx(chi) + dy(a)*dy(chi) + dp(a)*dp(chi)/(R*R)
  end function Bv_parderiv

  type(algexpr) function gradprod(a,b)
    implicit none
    type(algexpr), intent(in) :: a, b

    gradprod = dx(a)*dx(b) + dy(a)*dy(b) + dp(a)*dp(b)/(R*R)
  end function gradprod

  type(algexpr) function inprod(a,b)
    implicit none
    type(algexpr), intent(in) :: a, b

    inprod = gradprod(a,b) - Bv_parderiv(a)*Bv_parderiv(b)/Bv2
  end function inprod

  type(algexpr) function Lap(a)
    implicit none
    type(algexpr), intent(in) :: a

    Lap = dx(R*dx(a))/R + dy(dy(a)) + dp(dp(a))/(R*R)
  end function Lap

  type(algexpr) function pLap(a)
    implicit none
    type(algexpr), intent(in) :: a

    pLap = Lap(a) - Bv_parderiv(Bv_parderiv(a)/Bv2)
  end function pLap
end module mod_equations
