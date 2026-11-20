!> model199's temperature-dependent resistivity as a routine, for the physics
!! preconditioner's element matrices (mod_pc_elt_matrix_force_fft: M_eta and
!! K_eta of the resistive line bending of vs_lumped, mod_petsc_pc_vs). The
!! statements are those of model_eta_T.h, which the model's element matrices
!! (mod_elt_matrix_fft, mod_elt_matrix) include in place: one definition, and
!! the model's code stays the original statements. With T~ = corr_neg_temp(T0)
!! (T0 = |T| at the Gauss point):
!!   eta_T = eta (T~ / T_0)^(-3/2)           eta_T_dependent and T~ <= T_max_eta
!!   eta_T = eta (T_max_eta / T_0)^(-3/2)    eta_T_dependent and T~ >  T_max_eta
!!   eta_T = eta                             otherwise
!! deta_dT and d2eta_d2T are its first and second derivatives in T~ (zero where
!! eta_T is constant). Compiled apart from the model, the routine may differ
!! from the model's in-place evaluation in the last bit.
module mod_model_resistivity
  implicit none
  private

  public :: model_eta_T

contains

  subroutine model_eta_T(T0, eta_T, deta_dT, d2eta_d2T)
    use phys_module, only: eta, eta_T_dependent, T_max_eta, T_0
    use corr_neg,    only: corr_neg_temp
    real*8, intent(in)  :: T0                    !< |T| at the Gauss point
    real*8, intent(out) :: eta_T, deta_dT, d2eta_d2T

#include "model_eta_T.h"
  end subroutine model_eta_T

end module mod_model_resistivity
