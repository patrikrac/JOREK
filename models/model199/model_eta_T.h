! model199's temperature-dependent resistivity, one definition for the model's
! element matrices (mod_elt_matrix_fft, mod_elt_matrix: included in place, so
! that their code is the original statements) and the physics preconditioner
! (mod_model_resistivity, the resistive line bending of vs_lumped). In: T0 =
! |T| at the Gauss point; phys_module (eta, eta_T_dependent, T_max_eta, T_0)
! and corr_neg (corr_neg_temp). Out: eta_T, deta_dT, d2eta_d2T.
     if ( eta_T_dependent .and. corr_neg_temp(T0) <= T_max_eta) then
       eta_T     = eta   * (corr_neg_temp(T0)/T_0)**(-1.5d0)
       deta_dT   = - eta   * (1.5d0)  * corr_neg_temp(T0)**(-2.5d0) * T_0**(1.5d0)
       d2eta_d2T =   eta   * (3.75d0) * corr_neg_temp(T0)**(-3.5d0) * T_0**(1.5d0)
     else if ( eta_T_dependent .and. corr_neg_temp(T0) > T_max_eta) then
       eta_T     = eta   * (T_max_eta/T_0)**(-1.5d0)
       deta_dT   = 0.d0
       d2eta_d2T = 0.d0
     else
       eta_T     = eta
       deta_dT   = 0.d0
       d2eta_d2T = 0.d0
     end if
