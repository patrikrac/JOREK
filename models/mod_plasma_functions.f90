!> Module containing plasma functions t
module mod_plasma_functions 
  
  use phys_module
  use mod_model_settings, only:  with_impurities, with_TiTe
  use constants
    
  implicit none
  
  private
  public resistivity, conductivity_parallel, viscosity, hyper_resistivity, hyper_viscosity, &
         coulomb_log_ei, initialise_reference_parameters
  
  contains
  
  



  subroutine initialise_reference_parameters()

    implicit none 

    real*8 :: rho0, Te0_keV, Ti0_keV

    ! --- Calculate normalization factors.
    rho0               = central_density * 1.d20 * central_mass * ATOMIC_MASS_UNIT
    sqrt_mu0_rho0      = sqrt( mu_zero * rho0 )
    sqrt_mu0_over_rho0 = sqrt( mu_zero / rho0 )

    ! --- Calculate nominal parameters printed in the logfile for reference
    if (with_TiTe) then
      Te0_keV               = Te_0 / ( EL_CHG * mu_zero * central_density * 1.d+20 ) / 1.d+3
      Ti0_keV               = Ti_0 / ( EL_CHG * mu_zero * central_density * 1.d+20 ) / 1.d+3
      call coulomb_log_ei(Te_0, Te_0, 1.d0, 1.d0, 0.d0, 0.d0, 0.d0, lnA_center)

      ZK_e_par_SpitzerHaerm = 5.5789d+0 * central_mass*ATOMIC_MASS_UNIT/(mass_electron*lnA_center) * Te0_keV**(2.5d+0) * (gamma-1.d0) * sqrt_mu0_over_rho0
      ZK_i_par_SpitzerHaerm = 5.8410d+2 * sqrt(central_mass/2.d+0)/(lnA_center)               * Ti0_keV**(2.5d+0) * (gamma-1.d0) * sqrt_mu0_over_rho0
    else
      Te0_keV               = T_0 / 2.d+0 / ( EL_CHG * mu_zero * central_density * 1.d+20 ) / 1.d+3
      call coulomb_log_ei(T_0, T_0, 1.d0, 1.d0, 0.d0, 0.d0, 0.d0, lnA_center)

      ZK_par_SpitzerHaerm   = 5.5789d+0 * central_mass*ATOMIC_MASS_UNIT/(mass_electron*lnA_center) * Te0_keV**(2.5d+0) * (gamma-1.d0) * sqrt_mu0_over_rho0
    end if
    tauIC_nominal      = central_mass * ATOMIC_MASS_UNIT / ( EL_CHG * F0 * sqrt_mu0_rho0 * 2.d0 )
    eta_Spitzer        = ( 1.65d-9 * lnA_center * Te0_keV**(-1.5d+0) ) / sqrt_mu0_over_rho0

    ! --- Assign minimum values for parallel conduction if not given
    if (T_min_ZKpar  < -1.d10) T_min_ZKpar  = T_min   
    if (Ti_min_ZKpar < -1.d10) Ti_min_ZKpar = T_min   
    if (Te_min_ZKpar < -1.d10) Te_min_ZKpar = T_min   

  
  end subroutine initialise_reference_parameters
  
  
  



  !> Determine Coulomb logartihm describing electron collisions with ions
  pure subroutine coulomb_log_ei(T_raw, T_corr, r0_raw, r0_corr, rimp0_raw, rimp0_corr, alpha_e, lnA, dalpha_e_dT,  &
                                 dlnA_dT, d2lnA_dT2, dlnA_dr0, dlnA_drimp0)
    implicit none

    real*8, intent(in)             :: T_raw                !< temperature (without correction)
    real*8, intent(in)             :: T_corr               !< corrected temperature > 0
    real*8, intent(in)             :: r0_raw               !< total ion mass (without correction)
    real*8, intent(in)             :: r0_corr              !< corrected total ion mass
    real*8, intent(in)             :: rimp0_raw            !< impurity mass (without correction)
    real*8, intent(in)             :: rimp0_corr           !< corrected impurity mass
    real*8, intent(in)             :: alpha_e              !< coefficient to recover ne 
    real*8, intent(out)            :: lnA                  !< output coulomb logartihm
    real*8, optional, intent(in)   :: dalpha_e_dT          !< derivative of coefficient to recover ne
    real*8, optional, intent(out)  :: dlnA_dT              !< 1st derivative with respect to the temperature
    real*8, optional, intent(out)  :: d2lnA_dT2            !< 2nd derivative with respect to the temperature
    real*8, optional, intent(out)  :: dlnA_dr0             !< 1st derivative with respect to the total density
    real*8, optional, intent(out)  :: dlnA_drimp0          !< 1st derivative with respect to the impurity density

    !> Local parameters
    real*8 :: ne_cm3, Te_corr_eV, dTe_corr_eV_dT, dne_cm3_dT, dne_cm3_dr0, dne_cm3_drimp0
    real*8 :: ne_cm3_central, Te_central, lnA_central

    if (present(dlnA_dT))       dlnA_dT     = 0.d0
    if (present(d2lnA_dT2))     d2lnA_dT2   = 0.d0
    if (present(dlnA_dr0))      dlnA_dr0    = 0.d0
    if (present(dlnA_drimp0))   dlnA_drimp0 = 0.d0

    !> Get electron density in cm^-3 units and its derivatives
    ne_cm3         = (r0_corr + alpha_e * rimp0_corr) * 1.d20 * central_density * 1.d-6
    if (present(dalpha_e_dT)) then
      dne_cm3_dT   = dalpha_e_dT * rimp0_corr         * 1.d20 * central_density * 1.d-6
    endif
    dne_cm3_dr0    = 1.d0                             * 1.d20 * central_density * 1.d-6
    dne_cm3_drimp0 = alpha_e * 1.d0                   * 1.d20 * central_density * 1.d-6
    
    if (ne_cm3 < 1.d10  )       ne_cm3         = 1.d10 ! To prevent absurd numbers in the Coulomb log
    if (r0_raw < r0_corr)       dne_cm3_dr0    = 0.d0
    if (rimp0_raw < rimp0_corr) dne_cm3_drimp0 = 0.d0

    !> Get electron temperature in eVs and its derivatives
    if (with_TiTe) then
      Te_corr_eV     = T_corr / (EL_CHG*MU_ZERO*central_density*1.d20)
      dTe_corr_eV_dT = 1.0d0  / (EL_CHG*MU_ZERO*central_density*1.d20)
      Te_central     = Te_0   / (EL_CHG*MU_ZERO*central_density*1.d20)
    else
      Te_corr_eV     = T_corr / (EL_CHG*MU_ZERO*central_density*1.d20 * 2.d0)
      dTe_corr_eV_dT = 1.0d0  / (EL_CHG*MU_ZERO*central_density*1.d20 * 2.d0)
      Te_central     = T_0    / (EL_CHG*MU_ZERO*central_density*1.d20 * 2.d0)
    endif
    if (T_raw  <  T_corr)  dTe_corr_eV_dT = 0.d0
    
    !> Evaluate the coulomb logarithm and its derivatives
    if (Te_corr_eV < 10.d0) then
      lnA  = 23.0    - 0.5*log(ne_cm3) +  1.5*log(Te_corr_eV)  ! Assuming bg_charge is 1!
      if (present(dlnA_dT))    dlnA_dT     = - 0.5/ne_cm3 * dne_cm3_dT +  1.5/Te_corr_eV * dTe_corr_eV_dT
      if (present(d2lnA_dT2))  d2lnA_dT2   = + 0.5/ne_cm3**2 * dne_cm3_dT**2 - 1.5/Te_corr_eV**2 * dTe_corr_eV_dT
    else
      lnA  = 24.1513 - 0.5*log(ne_cm3) +  1.0*log(Te_corr_eV)
      if (present(dlnA_dT))    dlnA_dT     = - 0.5/ne_cm3 * dne_cm3_dT +  1.0/Te_corr_eV * dTe_corr_eV_dT
      if (present(d2lnA_dT2))  d2lnA_dT2   = + 0.5/ne_cm3**2 * dne_cm3_dT**2 - 1.0/Te_corr_eV**2 * dTe_corr_eV_dT
    endif

    if (present(dlnA_dr0))     dlnA_dr0    = - 0.5/ne_cm3 * dne_cm3_dr0
    if (present(dlnA_drimp0))  dlnA_drimp0 = - 0.5/ne_cm3 * dne_cm3_drimp0

  end subroutine coulomb_log_ei
 





  !> Determine resistivity (input/output in JOREK units)
  pure subroutine resistivity(eta_0, T_raw, T_corr, T_max, T0, Z_eff, lnA, eta_T,                  & 
                              dZ_eff_dT, dZ_eff_dr0, dZ_eff_drimp0, dr0_corr_dn, drimp0_corr_dn,   & 
                              deta_dT, d2eta_d2T, deta_dr0, deta_drimp0,                           &
                              dlnA_dT, d2lnA_dT2, dlnA_dr0, dlnA_drimp0) 

    implicit none
    
    real*8, intent(in)             :: eta_0                ! central resistivity
    real*8, intent(in)             :: T_raw                ! temperature without correction
    real*8, intent(in)             :: T_corr               ! corrected temperature > 0
    real*8, intent(in)             :: T_max                ! max temperature to use in the function
    real*8, intent(in)             :: T0                   ! central temperature at equilibrium
    real*8, intent(in)             :: Z_eff                ! effective charge (only used with_impurities at the moment)
    real*8, intent(in)             :: lnA                  ! Coulomb logarithm 
    real*8, intent(out)            :: eta_T                ! output resistivity
    real*8, optional, intent(in)   :: dZ_eff_dT            ! Derivative of Zeff w.r.t. the temperature 
    real*8, optional, intent(in)   :: dZ_eff_dr0           ! Derivative of Zeff w.r.t. the total density 
    real*8, optional, intent(in)   :: dZ_eff_drimp0        ! Derivative of Zeff w.r.t. the impurity density
    real*8, optional, intent(in)   :: dr0_corr_dn          ! Derivative of density correction  
    real*8, optional, intent(in)   :: drimp0_corr_dn       ! Derivative of impurity density correction  
    real*8, optional, intent(out)  :: deta_dT              ! 1st derivative with respect to the temperature
    real*8, optional, intent(out)  :: d2eta_d2T            ! 2nd derivative with respect to the temperature
    real*8, optional, intent(out)  :: deta_dr0             ! 1st derivative with respect to the total density
    real*8, optional, intent(out)  :: deta_drimp0          ! 1st derivative with respect to the impurity density
    real*8, optional, intent( in)  :: dlnA_dT              ! 1st derivative with respect to the temperature
    real*8, optional, intent( in)  :: d2lnA_dT2            ! 2nd derivative with respect to the temperature
    real*8, optional, intent( in)  :: dlnA_dr0             ! 1st derivative with respect to the total density
    real*8, optional, intent( in)  :: dlnA_drimp0          ! 1st derivative with respect to the impurity density

    !--- Local parameters
    real*8 :: eta_coef, deta_coef_dZeff

    if (present(deta_dT))       deta_dT     = 0.d0
    if (present(d2eta_d2T))     d2eta_d2T   = 0.d0
    if (present(deta_dr0))      deta_dr0    = 0.d0
    if (present(deta_drimp0))   deta_drimp0 = 0.d0

    if ( eta_T_dependent .and. T_corr <= T_max ) then
      eta_T     =   eta_0 * (T_corr/T0)**(-1.5d0)
      if (present(deta_dT))    deta_dT   = - eta_0 * (1.5d0)  * T_corr**(-2.5d0) * T0**(1.5d0)
      if (present(d2eta_d2T))  d2eta_d2T =   eta_0 * (3.75d0) * T_corr**(-3.5d0) * T0**(1.5d0)
    else if (eta_T_dependent .and. T_corr > T_max) then
      eta_T     = eta_0 * (T_max/T0)**(-1.5d0)
    else
      eta_T     = eta_0
    end if

    if ( eta_T_dependent .and. (T_raw .lt. T_min) ) then
      eta_T       = eta_0     * (T_min/T0)**(-1.5d0)
      if (present(deta_dT))     deta_dT    = 0.d0
      if (present(d2eta_d2T))   d2eta_d2T  = 0.d0
    endif

    if (with_impurities) then 

      eta_coef     = Z_eff*(1.+1.198*Z_eff+0.222*Z_eff**2)/(1.+2.966*Z_eff+0.753*Z_eff**2)
      eta_coef     = eta_coef / ((1.+1.198+0.222)/(1.+2.966+0.753))

      deta_coef_dZeff = (1.+1.198*Z_eff+0.222*Z_eff**2)/(1.+2.966*Z_eff+0.753*Z_eff**2)
      deta_coef_dZeff = deta_coef_dZeff + Z_eff*(1.198+2.*0.222*Z_eff)/(1.+2.966*Z_eff+0.753*Z_eff**2)
      deta_coef_dZeff = deta_coef_dZeff - Z_eff*(1.+1.198*Z_eff+0.222*Z_eff**2)*(2.966+2.*0.753*Z_eff)/((1.+2.966*Z_eff+0.753*Z_eff**2)**2)
      deta_coef_dZeff = deta_coef_dZeff / ((1.+1.198+0.222)/(1.+2.966+0.753))

      if ( eta_T_dependent ) then
        eta_T       = eta_T * eta_coef
        if (present(deta_dr0) .and. present(dZ_eff_dr0) .and. present(dr0_corr_dn)) then
          deta_dr0    = eta_T * deta_coef_dZeff * dZ_eff_dr0 * dr0_corr_dn
        endif
        if (present(deta_drimp0) .and. present(dZ_eff_drimp0) .and. present(drimp0_corr_dn)) then
            deta_drimp0 = eta_T * deta_coef_dZeff * dZ_eff_drimp0 * drimp0_corr_dn
        endif
        if (present(deta_dT) .and. present(dZ_eff_dT)) then
          deta_dT     = deta_dT * eta_coef + eta_T * deta_coef_dZeff * dZ_eff_dT
        endif
        if (present(deta_dT) .and. present(d2eta_d2T) .and. present(dZ_eff_dT)) then
          d2eta_d2T   = d2eta_d2T * eta_coef + 2.d0*deta_dT * deta_coef_dZeff * dZ_eff_dT  ! Missing d2Zeff_dT2 term!
        endif
      end if

    endif

    ! --- Add dependencies for the Coulomb logarithm, and normalize by the central Coulomb logarithm
    if (eta_coul_log_dep) then
      eta_T = eta_T * lnA / lnA_center
      if (present(deta_dT) .and. present(dlnA_dT)) then
        deta_dT     = (deta_dT * lnA + eta_T * dlnA_dT) / lnA_center
      endif
      if (present(d2eta_d2T) .and. present(deta_dT) .and. present(dlnA_dT) .and. present(d2lnA_dT2))  then
        d2eta_d2T   = (d2eta_d2T * lnA + 2.d0 * deta_dT * dlnA_dT + eta_T * d2lnA_dT2) / lnA_center
      endif
      if (present(deta_dr0) .and. present(dlnA_dr0)) then
        deta_dr0    = (deta_dr0 * lnA + eta_T * dlnA_dr0) / lnA_center
      endif
      if (present(deta_drimp0).and. present(dlnA_drimp0)) then
        deta_drimp0 = (deta_drimp0 * lnA + eta_T * dlnA_drimp0) / lnA_center
      endif
    endif

  end subroutine resistivity








  !> Determine hyper resistivity (input/output in JOREK units)
  pure subroutine hyper_resistivity(T_raw, T_corr, T0, psi_norm, eta_num_T, deta_num_dT) 

    implicit none
    
    real*8, intent(in)             :: T_raw            ! temperature without correction
    real*8, intent(in)             :: T_corr           ! corrected temperature > 0
    real*8, intent(in)             :: T0               ! central temperature at equilibrium
    real*8, intent(in)             :: psi_norm         ! normalized poloidal flux          
    real*8, intent(out)            :: eta_num_T        ! output hyper resistivity
    real*8, optional, intent(out)  :: deta_num_dT      ! Derivative w.r.t. the temperature 

    ! --- Hyper-resistivity
    if ( eta_num_psin_dependent ) then
      eta_num_T   = eta_num * 0.5d0 * ( 1.d0 - tanh( (psi_norm-eta_num_prof(1))/eta_num_prof(2)) )      
      if (present(deta_num_dT))  deta_num_dT = 0.d0      
    else if ( eta_num_T_dependent ) then
      eta_num_T     =   eta_num   * (T_corr/T0)**(-3.d0)
      if (present(deta_num_dT))  deta_num_dT   = - eta_num   * (3.d0)  * T_corr**(-4.d0) * T0**(3.d0)
      if (T_raw .lt. T_min) then
        eta_num_T     = eta_num    * (T_min/T0)**(-3.d0)
        if (present(deta_num_dT)) deta_num_dT   = 0.d0
      endif
    else
      eta_num_T     = eta_num
      if (present(deta_num_dT))   deta_num_dT   = 0.d0
    end if

  end subroutine hyper_resistivity






  ! --- Parallel conductivity (input-output in JOREK units)
  pure subroutine conductivity_parallel(ZK_par0, ZK_par_max, T_raw, T_corr, T_min_ZKpar, T0, ZK_par_T, &
                                         ZKpar_neg_thresh, ZK_par_neg, dT0_corr_dT, dZK_par_dT)

    real*8, intent(in)             :: ZK_par0          ! central parallel conduction
    real*8, intent(in)             :: ZK_par_max       ! maximum value for parallel conduction
    real*8, intent(in)             :: T_raw            ! temperature without correction
    real*8, intent(in)             :: T_corr           ! corrected temperature > 0
    real*8, intent(in)             :: T_min_ZKpar      ! min temperature to use in the function
    real*8, intent(in)             :: T0               ! central temperature at equilibrium
    real*8, intent(out)            :: ZK_par_T         ! output parallel conduction
    real*8, optional, intent(in)   :: ZKpar_neg_thresh ! threshold for negative correction             
    real*8, optional, intent(in)   :: ZK_par_neg       ! value after negative correction             
    real*8, optional, intent(in)   :: dT0_corr_dT      ! derivative of temperature correction
    real*8, optional, intent(out)  :: dZK_par_dT       ! temperature derviative of parallel conduction

    ! --- Local parameters
    real*8 :: dZK_par_dT_tmp

    if ( ZKpar_T_dependent ) then
      ZK_par_T       = ZK_par0 * (T_corr/T0)**(+2.5d0)   
      dZK_par_dT_tmp = ZK_par0 * (2.5d0)  * T_corr**(+1.5d0) * T0**(-2.5d0) 
      if (ZK_par_T .gt. ZK_par_max) then
        ZK_par_T       = Zk_par_max
        dZK_par_dT_tmp = 0.d0
      endif
      if (T_raw .lt. T_min_ZKpar) then
        ZK_par_T       = ZK_par0 * (T_min_ZKpar/T0)**(+2.5d0)
        dZK_par_dT_tmp = 0.d0
      endif
    else
      ZK_par_T      = ZK_par0 
      dZK_par_dT_tmp = 0.d0
    endif

    if (present(dZK_par_dT)) then
      dZK_par_dT = dZK_par_dT_tmp
      if (present(dT0_corr_dT)) dZK_par_dT = dZK_par_dT * dT0_corr_dT
    endif

    ! --- Increase value to avoid negative temperatures
    if (present(ZKpar_neg_thresh) .and. present(ZK_par_neg)) then
      if (T_raw .lt. ZKpar_neg_thresh) then
        ZK_par_T = ZK_par_neg
        if (present(dZK_par_dT)) dZK_par_dT = 0.d0
      endif
    endif

  end subroutine conductivity_parallel 






  !> Determine hyper viscosity (input/output in JOREK units)
  pure subroutine hyper_viscosity(T_raw, T_corr, T0, visco_num_T, dvisco_num_dT) 

    implicit none
    
    real*8, intent(in)             :: T_raw            ! temperature without correction
    real*8, intent(in)             :: T_corr           ! corrected temperature > 0
    real*8, intent(in)             :: T0               ! central temperature at equilibrium
    real*8, intent(out)            :: visco_num_T        ! output viscosity
    real*8, optional, intent(out)  :: dvisco_num_dT      ! Derivative of Zeff w.r.t. the temperature 

    if ( visco_num_T_dependent ) then
      visco_num_T     =   visco_num   * (T_corr/T0)**(-3.d0)
      if (present(dvisco_num_dT))  dvisco_num_dT   = - visco_num   * (3.d0)  * T_corr**(-4.d0) * T0**(3.d0)
      if (T_raw .lt. T_min) then
        visco_num_T     = visco_num    * (T_min/T0)**(-3.d0)
        if (present(dvisco_num_dT)) dvisco_num_dT   = 0.d0
      endif
    else
      visco_num_T     = visco_num
      if (present(dvisco_num_dT))   dvisco_num_dT   = 0.d0
    end if

  end subroutine hyper_viscosity









  ! --- Viscosity (input-output in JOREK units)
  pure subroutine viscosity(visco, T_raw, T_corr,T0, visco_T, dvisco_dT, d2visco_d2T)

    real*8, intent(in)             :: visco            ! temperature without correction
    real*8, intent(in)             :: T_raw            ! temperature without correction
    real*8, intent(in)             :: T_corr           ! corrected temperature > 0
    real*8, intent(in)             :: T0               ! central temperature at equilibrium
    real*8, intent(out)            :: visco_T          ! output viscosity          
    real*8, optional, intent(out)  :: dvisco_dT        ! 1st derivative w.r.t. temperature   
    real*8, optional, intent(out)  :: d2visco_d2T      ! 2nd derivative w.r.t. temperature           

    if ( visco_T_dependent ) then
      visco_T     =   visco * (T_corr/T0)**(-1.5d0)
      if (present(dvisco_dT))   dvisco_dT   = - visco * (1.5d0)  * T_corr**(-2.5d0) * T0**(1.5d0)
      if (present(d2visco_d2T)) d2visco_d2T =   visco * (3.75d0) * T_corr**(-3.5d0) * T0**(1.5d0)
      if (T_raw .lt. T_min) then
        visco_T     = visco  * (T_min/T0)**(-1.5d0)
        if (present(dvisco_dT  )) dvisco_dT   = 0.d0
        if (present(d2visco_d2T)) d2visco_d2T = 0.d0
      else if (T_raw .gt. T_max_visco) then
        visco_T     =   visco * (T_max_visco/T0)**(-1.5d0)
        if (present(dvisco_dT  )) dvisco_dT   = 0.d0
        if (present(d2visco_d2T)) d2visco_d2T = 0.d0
      endif
    else
      visco_T     = visco
      if (present(dvisco_dT))   dvisco_dT   = 0.d0
      if (present(d2visco_d2T)) d2visco_d2T = 0.d0
    end if

  end subroutine viscosity 







end module mod_plasma_functions
