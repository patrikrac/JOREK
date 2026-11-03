!> storage of the variables required in each of the coupling schemes
!> used to construct the indices of the feedback from the particle evolution
!> and the indices of the aux_node_list
!> PLEASE CAREFULLY OBSERVE EXISTING PARAMETERS TO AVOID OVERLAP WHEN ADDING A NEW COUPLING SCHEME
module coupling_variables
  implicit none

  integer, parameter :: n_aux_var_max = 100

  ! ====== variables names and number of coupling schemes ====== !

  integer, parameter :: var_name_len  = 15
  character(len=var_name_len), dimension(n_aux_var_max) :: coupling_vars

  ! NCS (Coupling scheme for neutral particles)
#ifdef WITH_TiTe
    character(len=var_name_len), dimension(4) :: ncs_var_names = [character(len=var_name_len) :: &
      "rho",      & !> density
      "mom_par",  & !> parallel momentum
      "E_Te",     & !> electron energy
      "E_Ti"      & !> ion energy
    ]
#else
    character(len=var_name_len), dimension(3) :: ncs_var_names = [character(len=var_name_len) :: &
      "rho",      & !> density
      "mom_par",  & !> parallel momentum
      "E"         & !> total energy
    ]
#endif

  ! ICS (Coupling scheme for impurity particles)
  !> These are only the base variables, there is also the impurity charge density, unique to each impurity group
#ifdef WITH_TiTe
    character(len=var_name_len), dimension(3) :: ics_var_names = [character(len=var_name_len) :: &
      "mom_par",  & !> parallel momentum
      "E_Te",     & !> electron energy
      "E_Ti"      & !> ion energy
    ]
#else
    character(len=var_name_len), dimension(2) :: ics_var_names = [character(len=var_name_len) :: &
      "mom_par",  & !> parallel momentum
      "E"         & !> total energy
    ]
#endif

  ! REP (Pressure coupling scheme for runaway electrons)
  character(len=var_name_len), dimension(3) :: rep_var_names = [character(len=var_name_len) :: &
    "P_par",    & !> parallel component of dynamic pressure tensor
    "P_perp",   & !> perpendicular component of dynamic pressure tensor
    "j_Phi"     & !> Phi compoment of current  
  ]

  ! EPF (Full pressure coupling for energetic particles)
  character(len=var_name_len), dimension(7) :: epf_var_names = [character(len=var_name_len) :: &
    "PI_RR",     & !> (R,R)     component of full pressure tensor
    "PI_ZZ",     & !> (Z,Z)     component of full pressure tensor
    "PI_PHIPHI", & !> (Phi,Phi) component of full pressure tensor
    "PI_RZ",     & !> (R,Z)     component of full pressure tensor
    "PI_RPHI",   & !> (Z,Phi)   component of full pressure tensor
    "PI_ZPHI",   & !> (R,Phi)   component of full pressure tensor
    "rho_ep"     & !>           density
  ]


  ! =========== Storage variables for kinetic coupling indices ======== !

  !> variables indices
  integer :: rho_idx_kin      = 0
  integer :: mom_par_idx_kin  = 0
#ifdef WITH_TiTe
    integer :: E_Te_idx_kin   = 0
    integer :: E_Ti_idx_kin   = 0 
#else
    integer :: E_idx_kin      = 0
#endif
  integer :: P_par_idx_kin    = 0
  integer :: P_perp_idx_kin   = 0
  integer :: j_Phi_idx_kin    = 0
  integer :: rho_ep_idx_kin    = 0
  integer :: PI_RR_idx_kin     = 0
  integer :: PI_ZZ_idx_kin     = 0
  integer :: PI_PHIPHI_idx_kin = 0
  integer :: PI_RZ_idx_kin     = 0
  integer :: PI_RPHI_idx_kin   = 0
  integer :: PI_ZPHI_idx_kin   = 0

  !> index of coupling variables specific to each impurity group
  integer :: ics_indices_kin(n_aux_var_max) = -1

end module coupling_variables
