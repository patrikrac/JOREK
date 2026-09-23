!> Input parameters and physical variables.
module phys_module
  
  use mod_parameters
  use constants
  use data_structure              !< Added in order to dynamically allocate pellets
  use mod_openadas
  use mod_coronal

  implicit none
  
  !> @name Various parameters
  real*8  :: eta                  !< Resistivity at plasma cener (normalized)
  real*8  :: eta_T_0              !< Initial resistivity
  real*8  :: eta_ohmic            !< Resistivity at core for the Ohmic heating term
  logical :: eta_T_dependent      !< Resistivity dependent on temperature? Otherwise constant
  logical :: eta_coul_log_dep     !< Resistivity dependent on variations of the Coulomb logarithm?
  real*8  :: T_max_eta            !< Temperature above which the resistivity is truncated (use with care; only for numerical reasons)
  real*8  :: T_max_eta_ohm        !< Temperature above which the resistivity used in the Ohmic heating term is truncated (use with care; only for numerical reasons)
  real*8  :: T_max_visco          !< Temperature above which the viscosity is truncated; It is aimed for keeping the Prandtl number constant when T_max_eta is activated. 
  real*8  :: visco                !< Viscosity at plasma center (normalized)
  real*8  :: visco_heating        !< Viscosity used in the perpendicular viscous heating term
  real*8  :: visco_rst            !< visco value from restart file
  real*8  :: visco_par_rst        !< visco_par value from restart file
  real*8  :: eta_rst              !< eta value from restart file
  logical :: visco_T_dependent    !< Viscosity dependent on temperature? Otherwise constant.
  logical :: visco_old_setup      !< If true, the old perp. viscosity treatment is used for compatibility (old visco depends on R^2)
  real*8  :: visco_par            !< Cross B-field viscosity acting on parallel flow (normalized)
  real*8  :: visco_par_par        !< B-field Parallel viscosity acting on parallel flow (normalized)
  real*8  :: visco_par_heating    !< Parallel viscosity used in the parallel viscous heating term (normalized)
  real*8  :: TiTe_ratio           !< ratio to set ion and electron temperature from T (in model 180): Ti=TiTe_ratio*T; Te=(1.0-TiTe_ratio)*T
  real*8  :: F0                   !< Determines fixed toroidal magnetic field: \f$ B_\phi = F_0/R \f$
  real*8  :: central_density      !< particle density at the magnetic axis (in units of \f$10^{20} m^{-3}\f$)
  real*8  :: central_mass         !< average ion mass in atomic mass units (constant in time and space, including electron mass)
  real*8  :: sqrt_mu0_rho0        !< Normalization factor \f$\sqrt(\mu_0 \rho_0)\f$ calculated from input
  real*8  :: sqrt_mu0_over_rho0   !< Normalization factor \f$\sqrt(\mu_0/\rho_0)\f$ calculated from input
  real*8  :: gamma                !< ratio of specific heat (typically 5/3)
  real*8  :: Q_bar                !< (model400)
  real*8  :: sigma                !< (model400)
  real*8  :: tauIC                !< Scaling factor for diamagnetic terms (see [[diamag|diamagnetic]])
  real*8  :: tauIC_nominal        !< Nominal scaling factor (considering Ti=Te) for diamagnetic terms (see [[diamag|diamagnetic]])
  real*8  :: eta_spitzer          !< Spitzer resistivity in the core (considering main ion charge Z=1, effective ion charge Zeff=1)
  real*8  :: lnA_center           !< Coulomb logarithm in the core (used for the resistivity function)
  logical :: Wdia                 !< Include diamagnetic flows in viscosity terms? (see [[wdia|here]])
  logical :: U_sheath             !< Use Stangeby BCs for electric potential
  logical :: renormalise          !< Set true to give all input MHD parameters in S.I. units (ie. renormalise them before equations)
  real*8  :: gamma_sheath         !< sheath boundary condition on open fieldlines (JOREK units); you can also provide gamma_stangeby in normal units instead!
  real*8  :: gamma_stangeby       !< Sheath tranmission coefficient given by P. Stangeby in (The plasma boundary of magnetic fusion devices)
  real*8  :: gamma_sheath_e       !< sheath boundary condition on open fieldlines (JOREK units); you can also provide gamma_stangeby in normal units instead!
  real*8  :: gamma_e_stangeby     !< Sheath tranmission coefficient given by P. Stangeby in (The plasma boundary of magnetic fusion devices)
  real*8  :: gamma_sheath_i       !< sheath boundary condition on open fieldlines (JOREK units); you can also provide gamma_stangeby in normal units instead!
  real*8  :: gamma_i_stangeby     !< Sheath tranmission coefficient given by P. Stangeby in (The plasma boundary of magnetic fusion devices)
  real*8  :: density_reflection   !< density reflection coeefficient on open fieldlines
  real*8  :: neutral_reflection   !< reflection coefficient of ions into neutrals (model500)
  real*8  :: imp_reflection       !< impurity reflection coefficient on open fieldlines
  real*8  :: loop_voltage         !< Apply a loop voltage at the boundary of the computational domain (in V; works only for fixed boundary)
  logical :: old_deuterium_atomic !< use old fit to calculate atomic coefficients for D (ionization, recombination, radiation), otherwise a better fit is used
  logical :: deuterium_adas       !< use OPEN ADAS to calculate ionization, recombination and radiation coeffients for deuterium   
  logical :: deuterium_adas_1e20  !< use OPEN ADAS with fixed density=1e20 to calculate ionization, recombination and radiation coeffients for deuterium
  logical :: mach_one_bnd_integral!< use a boundary integral (boundary_matrix_open) to implement Mach=one boundary condition
  logical :: vpar_smoothing       !< apply a smoothing function to smooth jumps in Vpar at B.n=0
  real*8  :: vpar_smoothing_coef(3) !< coefficients for the smoothing profile of the parallel velocity
  real*8  :: min_sheath_angle     !< For sheath boundary conditions: Minimum incident angle for heat and particle fluxes (in degrees)
  integer :: mode(n_tor)          !< Toroidal mode number corresponding to the JOREK modes, e.g., for n_period=8 and n_tor=3, mode(:)=0,8,8
  integer :: mode_coord(n_coord_tor)  !< Toroidal mode number corresponding to the JOREK RZ grid modes
  integer :: nout                 !< Output a restart file every nout timesteps
  integer :: nout_projection      !< Output particle projection every nout_projection timesteps (only for diagnostics)
                                  !< Note that the 'to_h5' or 'to_vtk' flag should be .true. in the 'new_projection' function for this parameter to be in play
  integer :: xcase                !< 1->LowerXpoint. 2->UpperXpoint. 3->doubleNull
  logical :: forceSDN             !< Force a symmetric double null, within the accuracy of SDN_threshold
  real*8  :: SDN_threshold        !< threshold, in absolute psi, for a symmetric-double-null grid construction
  integer :: rst_format           !< 0 == old format, 1 == new format for restart file
  integer :: n_tor_restart        !< Number of toroidal harmonics read in the restart file
  logical :: restart              !< Restart a code run from the restart file jorek_restart.h5?
  logical :: regrid               !< Re-generate the flux-aligned grid (does not work currently)?
  logical :: regrid_from_rz       !< Re-generate the flux-aligned grid from an rz equilibrium
  logical :: import_equil         !< (presently unused)
  logical :: xpoint               !< X-point plasma or not? see also xcase
  real*8  :: Z_xpoint_limit(2)    !< Search the lower X-point in the region Z < Z_xpoint_limit(1) and the upper X-point in the region Z > Z_xpoint_limit(2) 
  integer :: xpoint_search_tries  !< The number of candidate elements to check for being the element containing the upper or lower X-point.
  logical :: bootstrap            !< Evolve the Bootstrap current consistently with time?
  real*8  :: bootstrap_psin_cutoff!< Bootstrap-current hard cutoff if simulating X-point plasma.
  real*8  :: minRad               !< Approximation of minor radius for bootstrap current calculation
  logical :: refinement           !< Use mesh refinement? (not presently available)
  logical :: force_central_node   !< Force all nodes in the center to have the same values in flux aligned grids or independent values?
  logical :: fix_axis_nodes       !< Fix t-derivative and cross st-derivative on axis to avoid noise
  logical :: treat_axis           !> Flag for chosing grid axis treatment (see grids/mod_axis_treatment.f90)
  logical :: bc_natural_flux      !< boundary conditions for flux surface boundaries (2 and 3)
  logical :: bc_natural_open      !< use natural boundary conditions on the open fieldlines
  logical :: produce_live_data    !< Write data 'macroscopic_vars.dat' during the code run allowing to use plot_live_data.sh?
  logical :: grid_to_wall         !< extend the grid to a physical wall
  logical :: RZ_grid_inside_wall  !< build the rectangular grid inside first wall
  real*8  :: RZ_grid_jump_thres   !< threshold to change R-resolution as RZ-grid gets sqeezed by limiter contour
  real*8  :: manipulate_psi_map(5,5) !< Option to manipulate Psi_boundary for the initial grid
  logical :: adaptive_time        !< (presently not useful)
  logical :: equil                !< compute equilibrium
  logical :: no_mach1_bc          !< Never apply Mach-1 BCs
  logical :: Mach1_openBC         !< Full-MHD: Apply Mach-1 BCs inside mod_boundary_matrix_open.f90 (or mod_boundary_conditions.f90)
  logical :: Mach1_fix_B          !< Full-MHD: Use the initial magnetic field for Mach1 BCs on targets, ie. without AR and AZ variations
  logical :: export_polar_boundary !< Option to export boundary.txt even in the case of a polar boundary.

  real*8  :: eps_noise         !< Tolerance for noise reduction
  logical :: use_matrix_equilibration !< Use matrix equilibration for the iterative solver (Improves condition number of the matrix)

  ! --- RESISTIVITY SWITCHES FOR AR AND AZ EQUATIONS
  ! --- 1.
  ! --- Default set-up is eta_ARAZ_on = .true.
  ! --- In this case, the same resistivity is used for all AR,AZ,A3 components
  ! --- 2.
  ! --- If (eta_ARAZ_on = .true.) and (eta_ARAZ_simple = .true.), then the Fprof component of Bphi is removed from the resistive term.
  ! --- Note that in a stationary equilibrium, the Fprof component of Bphi from the current and the current source should exactly cancel anyway.
  ! --- However, this Fprof component can lead to strong noise at high resistivity, so it is often better to remove it.
  ! --- 3.
  ! --- If (eta_ARAZ_on = .false.), then resistivity is switched off by default for the AR and AZ equations.
  ! --- However, regardless of which eta-model is used for A3, (ie. eta_T_dependent or not), you can still set eta_ARAZ_const 
  ! --- which will use a constant resistivity at the level eta_ARAZ_const.
  ! --- 4.
  ! --- Here again, setting (eta_ARAZ_simple = .true.) with eta_ARAZ_const removes the Fprof dependence of Bphi in the resistive term and the current term for AR&AZ
  real*8  :: eta_ARAZ_const       !< Use uniform resistivity for AR and AZ equations, used only if eta_ARAZ_on=.false.
  logical :: eta_ARAZ_on          !< Full-MHD: to switch on/off resistive terms for AR and AZ equations
  logical :: eta_ARAZ_simple      !< Full-MHD: remove the Fprof dependence of Bphi in the resistive terms for AR and AZ (which should be compensated by current source anyway)

  logical :: tauIC_ARAZ_on        !< Full-MHD: to switch on/off diamagnetic terms for AR and AZ equations
  logical :: bench_without_plot   !< if .true., do not produce certain output plots (e.g., for benchmarking)
  logical :: gmres                !< Use iterative GMRES solver
  integer :: gmres_max_iter       !< Maximum number of GMRES iterations
  logical :: keep_n0_const        !< Perform a linear run where the equilibrium quantities (i_tor=1) do not change with time?
  logical :: linear_run           !< Same as keep_n0_const, to be replaced soon by true linear run where modes are independent
  logical :: export_for_nemec     !< Export equilibrium information for the NEMEC code?
  logical :: export_aux_node_list !< Include the aux_node_list for particle projections in the restart files
  logical :: use_murge            !< (Deprecated, Cannot be used any more)
  logical :: use_murge_element    !< (Deprecated, Cannot be used any more)
  logical :: use_BLR_compression  !< Use Block-Low-Rank (BLR) compression in MUMPS / PaStiX 6 solvers
  real*8  :: epsilon_BLR          !< Accuracy of BLR compression
  logical :: just_in_time_BLR     !< Use Just-in-time strategy for BLR compression (speed optimized)
  logical :: pastix_blr_abs_tol   !< Use absolute tolerance for BLR
  logical :: write_ps             !< Write postscript file at the end of the run
  logical :: use_mumps            !< Use Mumps solver
  logical :: use_pastix           !< Use Pastix solver
  logical :: use_strumpack        !< Use Strumpack solver
  logical :: use_mumps_eq         !< Use Mumps equilibrium solver
  logical :: use_pastix_eq        !< Use Pastix equilibrium solver
  logical :: use_strumpack_eq     !< Use Strumpack equilibrium solver
  logical :: use_petsc_eq         !< Use PETSc (MATAIJ + LU/MUMPS) equilibrium solver
  logical :: use_mumps_prj        !< Use Mumps projection solver
  logical :: use_pastix_prj       !< Use Pastix projection solver
  logical :: use_strumpack_prj    !< Use Strumpack projection solver  
  logical :: use_wsmp             !< Use WSMP solver
  logical :: centralize_harm_mat  !< Centralize harmonic matrices on toridal master ranks; switch for STRUMPACK solver
  real*8  :: prev_FB_fact = 1.d0  !< FB_factor that had been applied when importing the restart file
  integer :: mumps_ordering       !< MUMPS ordering option (7:automatic, 3:Scotch, 4:PORD, 5:METIS), default: 7
  integer :: pastix_maxthrd       !< maximum number of threads used by pastix solver (could be beneficial to use the reduced number)
  real*8  :: pastix_pivot         !< Pastix epsilon for magnitude control (pivot threshold)
  logical :: use_physics_pc       !< Use physics-based block PCSHELL preconditioner
  logical :: debug_physics_pc     !< Print PC matrix analysis (norms, eigenvalues) after first assembly
  logical :: physics_pc_monolithic  !< Monolithic 4x4 solve (stage-one Schur elimination test)
  logical :: physics_pc_multi_step  !< Three-step predictor-corrector apply (hydro -> mag. predictor -> Alfven corrector -> transport)
  logical :: physics_pc_multi_step_symmetric  !< Add backward K_alpha,beta corrector to the segregated multi_step apply
  logical :: physics_pc_wave_schur  !< multi_step sub-mode: use the new wave-Schur block-LDU predictor-corrector apply (Sec. 5.4) instead of the segregated-Schur variant
  logical :: physics_pc_sub_blocks       !< 2x2 super-block PC: (psi,u) Alfven + (rho,T) transport
  integer :: physics_pc_sub_blocks_mode  !< Apply variant: 1=Jacobi, 2=GS-forward, 3=GS-symmetric
  logical :: physics_pc_probe_exact !< Probe exact 4x4 Schur complement via basis-vector applications (small problems only)
  logical :: physics_pc_verify_spbp !< Offline S_PBP diagnostic: build full momentum Schur S_u and measure sigma(S_PBP_diag^-1 S_u) (expensive)
  logical :: physics_pc_reduced_pde  !< Assemble P_full (the reduced 4-var PDE operator from substituting j=J(psi), w=W(u) at the continuous level) AND use it as the reduced-solve operator in place of the extracted-block algebraic Schur (Milestones 1-2)
  logical :: physics_pc_drop_psi_coupling !< Drop the three group-psi couplings (U_psi_T, L_T_psi, L_rho_psi) from P_full, giving the arrow structure (Milestone 3)
  logical :: physics_pc_verify_reduced !< Compare P_full against the probed exact condensed 4x4, block-by-block and by spectral band (diagnostic; run continues afterwards)
  logical :: physics_pc_verify_schur   !< Milestone 4 Stage 4.1: verify the exact Schur factorization of P_full (exact S_u by probing + direct inversion; diagnostic, run continues)
  logical :: physics_pc_schur_approx   !< Milestone 4 Stage 4.2: compare approximate Schur complements (small-flow limit M0 and commutator-device M_* candidates) against the exact S_u of P_full
  logical :: physics_pc_schur_assemble !< Milestone 4 Stage 4.4 (Workstream A): also build the SPARSE assembled Schur complement Shat_ass = D_uu Q_u^-1 A_uM - L_up Q^-1 U_pu (lumped mass inverses) for each M_* candidate, and compare it against the probed dense Shat. Requires physics_pc_schur_approx.
  logical :: physics_pc_schur_itersolve !< Milestone 4 Stage 4.5: measure whether the sparse assembled S_ass is itself tractable by ITERATIVE methods (spectral character; ILU/BJACOBI/GAMG/HYPRE vs a sparse-MUMPS reference). Everything upstream of this used direct solves only, so this is the first measurement of feasibility rather than of approximation quality. Requires physics_pc_schur_assemble.
  integer :: physics_pc_schur_channels !< Stage 6.2: how many channels the PRODUCTION small-flow Schur operator carries. 1 = psi only, 2 = psi+rho, 3 = psi+rho+T. Measured (Stage 6.1): the rho and T channels are worth ~1e-4 of the exact correction on the pcbench case and ~1.5e-3 at beta ~ 2%, and add two matrix products per rebuild for no measurable gain -- hence the default of 1. Kept switchable because the worth scales linearly with beta.
  logical :: physics_pc_schur_amg      !< Stage 6.2: solve the production Schur block S_PBP by GMRES + BoomerAMG with a fixed iteration budget instead of a direct LU/MUMPS factorization. LU isolates the approximation quality of S_PBP; AMG is the production-relevant cost. Stage 4.5 measured BoomerAMG on this operator at 15 -> 16 -> 20 iterations over the 843/1839/3219 mesh sequence.
  integer :: physics_pc_schur_amg_its  !< Stage 6.2: the fixed inner iteration budget when physics_pc_schur_amg is on.
  integer :: physics_pc_schur_massinv  !< Stage 6.2: treatment of the mass inverse in the production small-flow Schur operator. 2 = diagonal, 3 = row-sum lumped, 7 = FSAI level 0, 8 = FSAI level 1 (default). NOTE 3 is a trap on this discretisation: row-sum lumping is exact on constants only for a partition-of-unity basis, and JOREK's C1 Bezier/Hermite derivative DOFs have near-zero row sums, so the lumped entry is a small difference of large numbers; the builder prints a floored-row count for exactly this reason. The numbering matches the Stage 4.2 variant sweep so the harness rows are directly comparable. A diagonal suffices only at short tstep; see docs/physics_pc §9.4.
  character(len=12) :: physics_pc_schur_variant !< Stage 6.3: which momentum-Schur ansatz the PRODUCTION multi_step/wave_schur path builds into S_PBP. "SF" (default) = the Stage 6.2 small-flow operator S = Atilde_22 - Atilde_21 Q^-1 B_12 / (1+zeta), unchanged. Any other value is a CANDIDATE LABEL from the cm_table (mod_petsc_pc_commutator_table): M0, M1x, M0R need no assembled blocks; M1a, M2, M2e, M2g, M3 do. Then S_PBP = Atilde_22 Q_u^-1 A_uM - Atilde_21 Q_p^-1 B_12 and the apply carries the right factor Shat^-1 = Q_u^-1 A_uM S_PBP^-1. M0 is algebraically identical to "SF" (its right factor cancels the opz scaling exactly) and exists as the regression control. Requires physics_pc_wave_schur and physics_pc_schur_channels = 1. "SFM2" is the same mixed-pair sweep but with the (psi,j) pair kept TOGETHER inside the block-diagonal approximation of M_y, i.e. M_y^-1 -> diag(Q_rho, Q_T, pair_psi_sf^-1) rather than fully diagonal. That restores the Lorentz coupling B_23 to the momentum Schur channel, which plain "SFM" drops (U has a zero j-row, so a fully diagonal M_y cannot transmit it) -- see docs/physics_pc/workstream_B_mixed_schur.md section 5.3. Its extra knob is physics_pc_schur_pairinv. "SFM" (Workstream B) is a THIRD kind of value, neither "SF" nor a cm_table label: the mixed-pair arm, which keeps j and omega explicit instead of substituting them. It solves two 2x2 pairs -- pair_psi = [[B_11,B_13],[B_31,B_33]] and pair_w = [[S_uu,B_24],[B_42,B_44]] with S_uu = B_22 - sum_ch (L Qi U)/(1+zeta) built from RAW B_21/B_25/B_26 -- so every block is second order or a mass matrix and nothing acquires the h^-4 conditioning of the substituted operators. It uses no assembled commutator blocks, builds no S_PBP, and needs no constraint mass folds in the apply. Requires physics_pc_wave_schur; supports all 3 channels; serial only.
  integer :: physics_pc_probe_inner   !< Workstream B phase 5: probe how solvable each step of the mixed sweep is by an ITERATIVE method, without touching the production path. 0 = off (default). Cumulative levels: 1 = LU reference + GMRES/ILU(0); 2 = + GMRES/BoomerAMG applied scalarly; 3 = + FGMRES/PCFIELDSPLIT on the pair layout (packed operators only); 4 = + point-block candidates on the INTERLEAVED layout (packed only): LU-on-interleaved as a wiring gate, GAMG + damped point-block Jacobi, and hypre BoomerAMG nodal coarsening -- the Chacon-style coupled-pair smoother, whose smoothing unit is the (psi,j) or (u,omega) pair at one DOF rather than a scalar. Probes pair_psi, pair_w, B_55 and B_66. Levels are cumulative and opt-in because AMG setup on the near-dense S_uu is expensive enough to look like a hang. Read the SCALAR AMG rows as a control only: the nest->AIJ layout is field-major, so a scalar coarsening sees two weakly-connected half-graphs; level 3 is the structurally correct question. Verdicts are on the SOLUTION error, not the residual -- pair_psi's residual is dominated by its ZBIG rows.
  integer :: physics_pc_schur_pairinv !< Workstream B, "SFM2" only: fidelity of the (psi,j) PAIR Schur inverse Shat^-1, where Shat = (1+zeta) Q_psi - B_13 M_j^-1 B_31. Same encoding as physics_pc_schur_massinv (2 = diagonal, 3 = row-sum lumped, 7 = FSAI-0, 8 = FSAI-1) but a SEPARATE knob, for two reasons: it is the quantity that distinguishes SFM2 from SFM so conflating it with the mass inverse would hide which one matters, and massinv = 8 forms Q*Q for its sparsity pattern, which for a Shat that is already a triple product is a density explosion. Default 2.
  integer :: physics_pc_pair_scale !< Workstream B, mixed-pair arms only: symmetric BLOCK scaling of the two packed pairs before the inner solve. 0 = off (default, bit-for-bit the previous behaviour). 1 = scale each pair as A <- D A D with D = diag(I, s I), s = sqrt(mean|diag| of field 1 / mean|diag| of field 2); the apply then solves (D A D) z = D b and recovers x = D z, so it is an exact similarity and changes only the conditioning the inner solver sees. WHY: both pairs pit an operator block against a mass block at very different scale, but they behave differently under dt (measured, meas_B/pr_ramp_m8): pair_w's |diag| spread GROWS 5.3e9 -> 2.2e10 -> 1.2e12 -> 3.1e13 at tstep 1/10/100/1000, while pair_psi's is CONSTANT at 1.78e10. It is not the ZBIG penalty rows in either case -- eliminate_boundary_dofs has already removed those. Spread alone does not predict solvability (pair_psi's ILU error IMPROVES from 7.3e+1 to 3.5e-5 across the ramp at constant spread), so the payoff is expected on pair_w, where every candidate including the MUMPS LU degrades as the spread grows. s is measured from the block diagonals rather than assumed from a power of dt, because the measured max does not follow a clean power. One scalar per block, so the block structure a fieldsplit or point-block smoother needs is preserved exactly; per-row equilibration would destroy it.
  integer :: physics_pc_pair_inner !< Workstream B, SFM2 only: inner solver for the two packed pairs. 0 = PREONLY+LU/MUMPS (default, the measured baseline). 1 = iterative on BOTH pairs. 2 = iterative on pair_psi, GMRES+ILU(0) on pair_w. 3 = GMRES+ILU(0) on both. The iterative arm is FGMRES + PCFIELDSPLIT with the MASS block as field 0 (j for pair_psi, omega for pair_w), a lower Schur factorisation and schur_precondition = a11. WHY that shape: (T1, docs S12.1) B_11 alone is fully AMG-amenable -- 14/14 sweep configurations pass at 29-42 its -- while the packed pair is not, so the difficulty is the coupling, not the coarsening. Splitting the MASS block off first eliminates the cheap well-conditioned block and leaves Shat = B_11, exactly the operator AMG handles; measured err 7.098e0 -> 7.02e-2 at fixed rtol (T6, S12.6). Safe because the outer solver is already FGMRES (mod_petsc.f90:387), so a variable preconditioner is admissible.
  integer :: physics_pc_pair_maxits !< Workstream B, SFM2, physics_pc_pair_inner /= 0: iteration cap on each pair's inner solve. Chacon 2025 stops the equivalent solve at 10 Krylov iterations or rtol 0.1, whichever comes first; this is the iteration half of that.
  real*8  :: physics_pc_pair_rtol !< Workstream B, SFM2, physics_pc_pair_inner /= 0: relative tolerance of each pair's inner solve. THE cost/quality knob, and the one expected to need a tstep-dependent value. NOTE (T8, docs S13) the pairs amplify residual into solution error by kappa ~ 1.2e7 (pair_psi) and ~1.2e5 (pair_w), so a solution-error target of 1e-6 would demand rtol ~1e-13; that bar is for CORRECTNESS tests, not for an inner solve. Inside a preconditioner only the residual matters, and the outer FGMRES absorbs the rest.
  real*8  :: physics_pc_psi_rtol !< Workstream F, SFM2 only: relative tolerance of the pair_psi inner solve ALONE, split out of physics_pc_pair_rtol so that pair_psi and pair_w can be tuned independently. Negative (default -1) means "not set": pair_psi falls back to physics_pc_pair_rtol, which reproduces every earlier measurement bit-for-bit. WHY a separate knob: physics_pc_pair_rtol is read by pair_psi, pair_w AND the rho/T GMG blocks, so loosening pair_psi to study it also loosened the very block under study. NOTE pair_psi amplifies inner residual into solution error by kappa ~ 1.2e7 (docs/physics_pc T8, S13), so a loose value here is far more dangerous than the same value on pair_w. Ignored unless pair_psi is actually iterative (physics_pc_pair_inner /= 0).
  integer :: physics_pc_corrector_form !< Workstream F, SFM2 only: how step 3 of the block-LDU sweep (the CORRECTOR, delta_y = delta_y* - M^-1 U delta_v) applies M^-1. 0 = today's behaviour: a second FULL pair_psi solve (default; the exact Eq. (16) of Chacon JCP 526 (2025) 113789). 1 = Eq. (17) of the same paper: the small-bulk-flow approximation M^-1 -> the SAME solve-free surrogate the momentum Schur complement already uses, so that the corrector and the operator it corrects for agree. Concretely, with pair^-1 acting on the RHS (r_psi, 0), dpsi = diag(Shat)^-1 r_psi and dj = -Qi B_31 dpsi, where Shat = (1+zeta) Q_psi - B_13 Qi B_31 and Qi = the 1/R mass inverse -- the very operands of the psi channel. WHY: the psi channel of S_uu is built with a DIAGONAL Shat^-1, while the corrector uses a full LU; we are therefore more accurate in the correction than in the operator being corrected, the opposite of the paper, at the price of one of the two pair_psi solves in EVERY apply. REQUIRES physics_pc_schur_pairinv = 2 or 3 (the surrogate must be diagonal to be reusable; pairinv 7/8 build an FSAI Mat that is destroyed with the channel).
  integer :: physics_pc_w_gmg !< Workstream C, SFM2 only: pair_w inner solve by geometric multigrid on the exact nested C1 coarse space (solvers/mod_petsc_pc_gmg.f90; docs/physics_pc/workstream_C_gmg_probe.md S2, S3.2). 0 = off (default; physics_pc_pair_inner decides). 1 = ONE V-cycle per pair_w solve (GMRES(4)+Jacobi smoothing, exact axis patch, Galerkin coarse operators, MUMPS coarsest). 2 = FGMRES with that V-cycle, capped by physics_pc_pair_maxits/physics_pc_pair_rtol. WHY: GAMG coarsens the C1 stencil pathologically (T2); the exact subdivision hierarchy keeps C_op ~1.15 and is near mesh-independent offline (15/16/44 -> 21/21/56 its at tstep 0.1/1/10 from 41x16 to 81x32). Needs the structured flux-surface grid and physics_pc_schur_pairinv = 2. NOT serial: build_fsai became ownership-aware on 2026-09-14 and the np = 1 guard (schur_mixed_require_serial) no longer fires; the cluster series runs this arm to np = 32.
  integer :: physics_pc_psi_schur !< Workstream C, SFM2 only: pair_psi inner solve by the eta-scaled j-first Schur (docs/physics_pc/workstream_C_gmg_probe.md S3.6). 0 = off (default; physics_pc_pair_inner decides). 1 = one application of z_j = B_33^-1 r_j, z_psi = Shat^-1 (r_psi - B_13 z_j) with Shat = B_11 - diag(B_13/B_33) B_31 by LU. 2 = the same with Shat by damped-Jacobi sweeps + an exact axis-patch solve (no factorisation of Shat). 3 = the same with Shat by ONE V-cycle of the C1 GMG (a second hierarchy, one field; smoother from physics_pc_gmg_smoother) -- the mesh-scalable variant; use with physics_pc_psi_outer > 0. WHY: the a11 approximation drops the resistive Schur term theta dt eta K, whose weight grows like theta dt eta/h^2; restored sparsely it is flat in h and dt (offline 4/5/6 outer its at tstep 0.1/1/10 on 41x16 AND 81x32). Requires eta_num = 0 (aborts otherwise). NOT serial since 2026-09-14; see physics_pc_w_gmg.
  integer :: physics_pc_suu_ring !< Workstream D, SFM2 only: 0 = off (default). k > 0 restricts S_uu to the ring-k node stencil (k-th power of the element connectivity, all harmonics coupled) before it is packed into pair_w. WHY: the explicit triple product gives S_uu a 5-ring stencil (11x an element block); offline (docs/physics_pc/data/stageD1_suu_sparsify.tsv) ring 3 is free for the pair_w solve at tstep 0.1-10. The full product is still formed first, so this cuts pair_w density, not setup. Serial only.
  integer :: physics_pc_suu_shell !< Workstream D, SFM2 only: 0 = off (default). Replaces every FINE-level pair_w matvec by a MATSHELL applied from the raw blocks, S_uu x = B_22 x - (B_21 - B_23 M B_31) Dh B_12 x. 1 = M is the FSAI sfp_Qip (reproduces the assembled S_W_aij; wiring gate). 2 = M is the EXACT B_33^-1 by the per-run LU ksp_Mj, an operator that cannot be assembled (Q^-1 is dense). With physics_pc_w_gmg /= 0 the shell drives the smoother, residuals, axis patch and inner FGMRES while S_W_aij still supplies the Galerkin coarse operators and Jacobi diagonal; with physics_pc_w_gmg = 0 (pair_inner = 0) pair_w becomes FGMRES(shell) + LU(S_W_aij) to physics_pc_pair_rtol. Needs pairinv = 2 and channels = 1. NOT serial since 2026-09-14; see physics_pc_w_gmg.
  integer :: physics_pc_rhot_gmg !< Workstream D (scalability): 0 = rho/T blocks B_55/B_66 by MUMPS LU at every rebuild (default). k > 0 = FGMRES(k) to physics_pc_pair_rtol preconditioned by one C1 GMG V-cycle each (hierarchy instances 3/4, one field).
  integer :: physics_pc_rhot_gmg_smoother !< smoother of the rho/T hierarchies; -1 (default) = physics_pc_gmg_smoother.
  integer :: physics_pc_psi_gmg_smoother !< smoother of the psi_schur = 3 Shat hierarchy, same codes as physics_pc_gmg_smoother; -1 (default) = use physics_pc_gmg_smoother.
  integer :: physics_pc_psi_gmg_nsmooth  !< smoothing steps of the Shat hierarchy; 0 (default) = the physics_pc_gmg_nsmooth rule.
  integer :: physics_pc_psi_outer !< Workstream D (pair_psi scalability), with physics_pc_psi_schur /= 0: 0 = ONE application of the eta-Schur factorisation per pair_psi solve (default). k > 0 = FGMRES on the assembled pair K_pj, preconditioned by it, to physics_pc_pair_rtol with at most k iterations (admissible: the outer solver is FGMRES).
  integer :: physics_pc_mass_split !< Workstream D1: 0 = constraint masses B_33/B_44 factored whole (default). 1 = factored per toroidal slot: the masses are exactly slot-block-diagonal and all slots with an identical matrix (cos/sin of every n >= 1) share ONE factor, applied as a multi-RHS solve. Same solution up to round-off; factor memory independent of n_tor. Refuses (abort) if a mass has cross-slot entries.
  integer :: physics_pc_lean_setup !< Workstream D audit: 0 = original SFM2 build (default). 1 = lean: diag(Shat) at pairinv 2 from row dots instead of the (B_13 Qi) B_31 product (A1; gated once per run against the product), and the per-rebuild norm/density diagnostics on the first build only (A2). Same preconditioner up to round-off. 2 = 1 plus MUMPS Cholesky (instead of LU) for the constraint masses B_33/B_44 when they are symmetric to 1e-12 (A7; the matrix-free pair_w solves with B_33 once per matvec). 3 = 2 plus no AIJ copy of the global Jacobian: the outer FGMRES and the block extraction use JOREK's BAIJ matrix directly (memory audit; needs physics_pc_harm_split = 1).
  integer :: physics_pc_harm_split !< Workstream D audit A8: 0 = physics-PC operands keep JOREK's full n_tor x n_tor harmonic blocks (default). 1 = keep only same-harmonic entries (extract_sub_block_h): the cross-harmonic couplings are zero for an axisymmetric linearisation and O(perturbation) otherwise, as in JOREK's standard harmonic-block-diagonal PC. Removes ~2/3 of every product, matvec and LU fill.
  integer :: physics_pc_gmg_smoother !< Workstream D, physics_pc_w_gmg /= 0: level smoother. 0 = GMRES(nsmooth)+point Jacobi (default, stage C). 1 = Richardson(omega)+point Jacobi, nsmooth sweeps (Chacon's damped Jacobi). 2 = Richardson(omega)+node-block Jacobi: one dense block per (node, toroidal harmonic) holding both fields and all 4 C1 DOFs (8x8), plus the axis ring as one block (Chacon 2025 S4.1: all local components of the split system together). 3 = GMRES(nsmooth)+node-block Jacobi. 4 = GMRES(nsmooth)+flux-surface-ring blocks (all DOFs of one radial index i, both fields, per toroidal slot: a line smoother along the surfaces, for the tstep-10 anisotropy). 5 = GMRES(nsmooth)+radial-line blocks (the control). 6 = GMRES(nsmooth)+ring blocks on the rings with r*dtheta/dr < physics_pc_gmg_ring_aspect and radial lines outside (GMGPolar's hybrid; stage D13: WORSE than 5, 5.9 vs 4.1 cycles at 41x16 tstep 10 -- use 5 with physics_pc_gmg_axis_rings = -1 instead). Blocks and diagonals come from the assembled Pmat (defect correction: the FSAI surrogate), residuals from the fine operator (the exact shell with suu_shell = 2).
  integer :: physics_pc_gmg_nsmooth !< Workstream D: smoothing steps per pre/post smoothing; 0 = default (4 for GMRES, 3 for Richardson).
  real*8  :: physics_pc_gmg_omega !< Workstream D: Richardson damping for smoothers 1 and 2 (Chacon: 0.7).
  real*8  :: physics_pc_gmg_ring_aspect !< Workstream D (axis): switch radius of physics_pc_gmg_smoother = 6, the hybrid of GMGPolar (Kuehn, Kruse, Ruede; Bourne et al., JCP 488 (2023) 112249 S3): flux-surface-ring blocks on the rings whose median cell ratio r*dtheta/dr is below this value, radial-line blocks outside. WHY: the polar map makes the circle couplings dominate near the axis and the radial ones outside, and the number of rings below 1 grows like n_tht/(2 pi) under refinement, which radial lines alone do not smooth. Default 1.0 (the paper's criterion). Coarse levels switch at the same physical radius.
  integer :: physics_pc_gmg_axis_rings !< Workstream D (axis): block smoothers 4/5/6 put rings 0..k of every level into ONE block per toroidal slot (default 0 = the axis ring alone). k = 1 mirrors C1 polar splines, where the pole is determined by the first TWO rings of the tensor basis (Toshniwal et al.; Zoni and Guclu). k = -1 = automatic: rings 0..I_s-1 of every level, I_s the switch ring of physics_pc_gmg_ring_aspect (at least ring 1). WHY (stage D13, 41x16 tstep 10): the residual of a real pair_w RHS piles up in the ring just outside the axis block (ring 1 at k = 0, ring 2 at k = 1); pair_w V-cycles fall 4.0 -> 3.2 -> 2.7 -> 2.6 for k = 0..3 and saturate once the block covers the rings with r*dtheta/dr < 1. Solving those rings as separate ring blocks (smoother 6) is WORSE than radial lines. COST: the automatic block holds ~n_tht/(2 pi) rings, so its MUMPS factor grows faster than N (pair_w: 438 MB and 22 ms per solve at 49x64); a fixed k = 3 keeps most of the gain and was the fastest setting on intear 121x48 (477 s vs 513 s automatic) and on the inxflow600 ballooning case 49x64 (992 s vs 1238 s).
  integer :: physics_pc_gmg_axis_mult !< Workstream D (axis), with physics_pc_gmg_axis_rings /= 0: how the axis block and the radial lines are combined in the block smoother. 0 = block Jacobi (default). 1 = Gauss-Seidel, axis block first, then the lines on x - A(rest,ax) y_ax. 2 = lines first, then the axis block on x - A(ax,rest) y_rest. 3 = axis, lines, axis. WHY (stage D13, 121x48): with the automatic axis block the remaining pair_w residual sits on both sides of the axis-block boundary, the classic block-Jacobi interface defect. Costs two small interface matvecs per smoother application.
  integer :: physics_pc_mass_solver !< Stage Q (cluster), constraint masses B_33/B_44 (ksp_Mj/ksp_Mw): 0 = MUMPS factor (default; per slot with physics_pc_mass_split). 1 = Chebyshev of fixed degree + node-block Jacobi (reference only: kappa 86 on the C1 Hermite mass, degree 100). 2 = Chebyshev of fixed degree + additive Schwarz (one subdomain per rank, overlap 1, local ICC(0); options prefix mass1_/mass2_): kappa 4.9 at np 4, degree ~20 to 1e-6. No inner products, only neighbour halo exchanges, and a fixed linear operator (safe inside GMRES). Bounds from one CG/Lanczos estimate per run, degree raised until a manufactured-solution check meets 1e-6. 3 = CG + the mode-2 preconditioner to rtol 1e-10 (diagnostic). WHY: the MUMPS solve with centralized RHS grows with the rank count (161x64: 90 s at np 1, 312 s at np 32); a mass is an L2-type operator, so one-level Schwarz needs no coarse space and its iteration count stays bounded in h and np.
  integer :: physics_pc_gmg_smooth_op !< Stage Q (cluster), C1 GMG with a matrix-free fine operator (suu_shell = 2): 0 = the fine smoother's GMRES runs on the exact shell (default). 1 = it runs on the assembled Pmat (the FSAI surrogate that already supplies its blocks and the coarse operators); the V-cycle residual and the outer pair_w FGMRES keep the shell. Cuts the exact-mass solves per V-cycle from about 2*nsmooth+3 to 3.
  real*8 :: physics_pc_gmg_axis_droptol !< Stage Q (cluster), C1 GMG axis block: drop the entries with |a_ij| < tol*sqrt(|a_ii a_jj|) before factoring it (0 = keep everything, default). The scaling is per entry because JOREK's boundary rows carry a diagonal about 7 decades above the bulk. The axis block is a smoother component, so an approximate factor is allowed; a smaller factor means a cheaper triangular solve in every smoother application, which is what the serial axis solve costs at high rank counts.
  integer :: physics_pc_gmg_axis_split !< Stage Q (cluster), C1 GMG with physics_pc_gmg_axis_rings /= 0 and physics_pc_harm_split = 1: 1 = the axis block is solved as one LU per |n| group (cos and sin together), group k on rank mod(k, np); the owners gather the group's rows to that rank and get the solution back. 0 = one LU of all slots on the ranks owning the axis (default). Same operator (harm_split makes the block block-diagonal in |n|). WHY: the axis LU is serial on rank 0 (161x64: 44-53 s whatever np); the groups run at the same time on different ranks.
  integer :: physics_pc_gmg_bnd_drop !< Workstream D (boundary), C1 GMG: 1 = the coarse levels drop the value u and the angular slope b on the boundary ring, the two DOFs JOREK's Dirichlet condition fixes (pure diagonal rows in every PC operand; checked at setup). The constrained coarse space is then exactly nested in the constrained fine space and P never writes into Dirichlet rows. 0 = keep them (default). WHY (stage D13, 121x48): once the axis block is fixed, the remaining pair_w residual after a V-cycle sits in rings n_flux-3..n_flux-1: the coarse correction put nonzero values into the Dirichlet rows and the smoother had to take them out again.
  integer :: physics_pc_gmg_ring_diag !< Workstream D (axis), diagnostic only: k > 0 = on the first k V-cycle applications after every GMG rebuild of the pair_w (1) and Shat (2) hierarchies, additionally run 6 stationary V-cycles on the same right-hand side and print where the residual lives by ring zone (axis ring, rings below the switch radius, outside). Costs 6 extra V-cycles per sampled application; the solve itself is unchanged. 0 = off (default).
  integer :: physics_pc_suu_comp !< Workstream D, with physics_pc_suu_ring > 0: 0 = plain truncation (default). 1 = add sign(d_ii)*sum of the dropped |s_ij| to the diagonal -- the only compensation that kept rings 1-2 usable at tstep 10 offline.
  integer :: physics_pc_dump_blocks !< Workstreams C and E, SFM2 only: write the operators to the run directory at every SFM2 rebuild, as PETSc binary files gmgdump_<op>_ts<tstep>_mi<massinv>_pi<pairinv>.petsc plus gmgdump_grid.txt (node indices, coordinates, element vertices and size factors) and gmgdump_meta.txt (mesh, the COMPILE-TIME n_tor/n_period, and the PC flags). Consumers: the offline scipy probe docs/physics_pc/tools/gmg_probe.py and the standalone MPI driver util/block_bench. A LEVEL, because a full case is 210 MB at 41x16 and ~16x that at 161x64: 0 = off (default); 1 = the pair_w bench set, i.e. the operands of shw_mult (B_12, B_21, B_22, B_23, B_24, B_31, B_42) plus B_33, B_44, the assembled S_uu and the Dh vector (~107 MB at 41x16); 2 = + the other three production systems (B_11, B_13, B_55, B_66); 3 = + the packed pairs pair_w and pair_psi, which are redundant repackings of the blocks above (96 MB of the 210 MB) and exist only to gate the bench's own packing -- generate level 3 once per mesh, not per case. Run with physics_pc_pair_scale = 0 so pair_psi is dumped unscaled. Dh is only written when physics_pc_suu_shell /= 0 and pairinv is 2 or 3; gmgdump_meta.txt records whether it is there. Use massinv = 2 for any wiring gate: only the diagonal arm makes Qi = diag(B_33)^-1 reproducible offline, and then S_uu = B_22 - (B_21 - B_23 Qi B_31) diag(Dh) B_12 holds to round-off (5.9e-17 at 41x16).
  integer :: physics_pc_force_operator !< Workstream E: 0 = off (default). 1 = also assemble the COMPOSED momentum-Schur force operator W and include it in the dump. W is the analytic composition of the two off-diagonal couplings (Chacon JCP 526 (2025) Eqs. 17-19) instead of the matrix triple product Ltil*Shat^-1*B_12 that shw_mult applies, so it carries NO mass inverse (no PhysPC_MjSolve) and has the plain C1 element stencil rather than the 5-ring one (58 vs 318-612 nnz/row). Scaled so the composed operator is exactly B_22 + W. Its three terms are field-line bending, kink, and toroidal curvature; the last comes from the rho AND T channels together (p = rho*T), so physics_pc_schur_channels = 1 cannot represent it -- and it is not optional, because without it the operator is self-adjoint only at FORCE-FREE rather than at force balance. DIAGNOSTIC ONLY: it is assembled and dumped, nothing in the apply reads it. Needs the linearisation state, so it is built in petsc_assemble_pc_matrices. See models/model199/mod_pc_elt_matrix_force_fft.f90 and docs/physics_pc/note_pair_w_force_operator/.
  !--- The production SFM2 path (mod_petsc_pc_sf). Six flags, and nothing else:
  !--- this path reads NONE of the physics_pc_* flags above, so a production
  !--- deck cannot silently inherit a research setting. See the module header.
  logical :: physics_pc_sf !< Select the clean production preconditioner: the SFM2 block-LDU sweep with S_uu = B_22 + W (the composed force operator), two mixed 2x2 pairs, and a per-block choice of direct LU or C1 geometric multigrid. Requires physics_pc_force_operator = 1. Forces physics_pc_harm_split = 1 and the audited GMG smoother / axis-ring / boundary-drop settings, loudly. Everything that was a measurement variable rather than a design choice is absent by construction, including every verify/probe/report/dump routine and the per-rebuild diagnostics they used to run.
  character(len=16) :: physics_pc_sf_pair_psi !< Production path: solver for pair_psi = [[B_11,B_13],[B_31,B_33]]. "lu" (default) = PREONLY + LU (MUMPS). "etaschur_lu" / "etaschur_gmg" = one application of the j-first lower block factorisation with the eta-scaled Schur Shat = B_11 - diag(B_13/B_33) B_31, wrapped in FGMRES on the raw pair, with Shat itself solved by LU or by ONE C1 GMG V-cycle. The eta-Schur backends need eta_num = 0 and are the mesh-scalable choice (the Mj LU is the measured cluster blocker); "lu" is the reference.
  character(len=16) :: physics_pc_sf_pair_w !< Production path: solver for pair_w = [[B_22+W,B_24],[B_42,B_44]]. "gmg" (default) = FGMRES + one C1 geometric multigrid V-cycle. "lu" = PREONLY + LU (MUMPS), the exact reference the GMG arm is gated against.
  character(len=16) :: physics_pc_sf_rho !< Production path: solver for the rho block B_55. "lu" (default) or "gmg".
  character(len=16) :: physics_pc_sf_T !< Production path: solver for the T block B_66. "lu" (default) or "gmg".
  real*8 :: physics_pc_sf_rtol !< Production path: the relative tolerance every ITERATIVE block backend stops at. One number for all of them deliberately: these are inner solves inside a preconditioner whose outer solver is FGMRES, so the quantity that matters is the outer count, not any individual block's residual.
  integer :: physics_pc_suu_form !< Workstream E, SFM2 only: how pair_w's (1,1) block is BUILT. 0 = today's sparse triple product S_uu = B_22 - Ltil*diag(Dh)*B_12 (default, unchanged). 1 = SHIP THE COMPOSED OPERATOR: S_uu := B_22 + W, with W the analytically composed momentum-Schur force operator of physics_pc_force_operator (Chacon JCP 526 (2025) 113789 Eq. 18), and the whole psi-channel chain (Ltil, Shat, the triple product) SKIPPED rather than built and discarded -- not building it is where the setup-time saving comes from. WHY: measured in util/block_bench with the production C1 GMG at tstep 10, the composed pair is MESH-INDEPENDENT and a far better multigrid target than the true pair_w -- 6/8/8 inner its against 19/23/23 at 41x16 / 71x32 / 91x40, ~5x cheaper per solve, ~2x cheaper setup, and NO PhysPC_MjSolve at all (the measured cluster blocker, 90 s at np 1 growing to 312 s at np 32). NOTE this is NOT a better approximation to S_uu -- it is measurably worse at that, and worse under refinement -- so it is only defensible as a REPLACEMENT, where S_uu stops being the target. The verdict is therefore the OUTER FGMRES count and the total time per step, not any inner number. REQUIRES force_operator = 1 and schur_channels = 1 (the full W already carries the curvature term, so the rho/T channels would double-count the pressure physics) and suu_ring = 0; FORCES suu_shell = 0, because shw_mult rebuilds S_uu analytically from the operands and structurally cannot see a substituted (1,1) block. See docs/physics_pc/workstream_E_pair_w_composition.md sections 9-10.
  real*8  :: physics_pc_pair_amg_thr !< Workstream B, SFM2, physics_pc_pair_inner /= 0: GAMG strength threshold for the Shat = B_11 solve. WHY exposed: the hierarchy dump (T2, docs S12.2) shows GAMG coarsening 15654 -> 154 in ONE level (~100x, against a healthy 3-8x) at threshold 0.01, because a 105-210 nnz/row C1 stencil makes almost every edge strong. Higher thresholds are the untested region that dump exposed.
  logical :: physics_pc_verify_mixed !< Workstream B: run the mixed-pair null test once at build time. Draws a random 6-variable residual, applies the SFM preconditioner, and checks all six Jacobian rows on the result. Rows 3 and 4 (the j and omega constraint rows, B_31 y_psi + B_33 y_j - x_j and B_42 y_u + B_44 y_w - x_w) are exact BY CONSTRUCTION -- both mixed pairs carry their constraint equation verbatim -- so they are a HARD ~1e-14 pass/fail and any deviation is a wiring bug (packing order, transposed nest, corrector sign, or a back-substitution that should have been skipped). Rows 1, 5 and 6 are expected to be small but nonzero: they measure the block-LDU lag. Serial only. Do not trust any SFM convergence number before rows 3 and 4 pass.
  integer :: physics_pc_schur_inner !< Stage 6.3: inner solver for the assembled Schur block S_PBP, independent of physics_pc_schur_variant. 0 = LEGACY: honour physics_pc_schur_amg / physics_pc_schur_amg_its exactly as before (default, back-compatible). 1 = direct PREONLY + LU. 2 = GMRES + HYPRE BoomerAMG with a fixed budget of physics_pc_schur_amg_its iterations. 3 = GMRES + BoomerAMG to rtol 1e-2, capped at 200 iterations. 4 = GMRES + BJACOBI/ILU(0), fixed budget physics_pc_schur_amg_its. Any nonzero value OVERRIDES physics_pc_schur_amg. Admissible because the outer solver is FGMRES.
  logical :: physics_pc_schur_mask !< Stage 6.3, commutator arm only: apply the ZBIG penalty-row interior treatment (mask rows/cols of S_PBP, identity on the penalty rows, Jacobi add-back in the apply), as the Stage 4.6 harness does. Required in general because the commutator building blocks have ZEROED boundary rows (construct_commutator_matrix_mod: zero_bc_rows_pc_matrix) while Atilde_22 carries zbig = 1.d12 there. Set .false. ONLY for the M0-vs-SF bit-for-bit regression check. Ignored when physics_pc_schur_variant = "SF".
  logical :: physics_pc_schur_global !< Milestone 4 Stage 4.6: run the GLOBAL outer FGMRES on P_full preconditioned by the block-LDU factorization whose Schur block is the assembled ansatz Shat^-1 = Q_u^-1 A_uM S_ass^-1, sweeping the inner budget from an exact solve down to a fixed 2 AMG iterations. Stages 4.2/4.5 measure approximation quality and inner solvability separately; this measures what the preconditioner is actually worth. Requires physics_pc_schur_assemble.
  logical :: commutator_analysis          !< Run commutator-operator (M_*) intertwining-defect analysis (candidates M0-M4, eps per toroidal harmonic) at first solve
  logical :: eliminate_boundary_dofs !< Zero boundary-DOF rows in PC correction matrices and use elm-diagonal BC scaling in global matrix; required for physics PC Schur correction approach
  logical :: use_newton           !< Use inexact Newton method
  integer :: maxNewton            !< maximum number of Newton iterations
  real(kind=8) :: gamma_Newton    !< Newton gamma-parameter: gmres_tol = gamma_Newton*(normRHScurrent/normRHSprevious)**alpha_Newton
  real(kind=8) :: alpha_Newton    !< Newton alpha-parameter: gmres_tol = gamma_Newton*(normRHScurrent/normRHSprevious)**alpha_Newton
  logical :: strumpack_matching   !< Perform maximum-diagonal-product reordering algorithm in STRUMPACK solver (improves direct solver, but use matrix centralization)

  ! ------------------------------------------------
  ! --- Structures to implement BCs in model600
  ! ------------------------------------------------
  ! --- For more info see  https://www.jorek.eu/wiki/doku.php?id=choose_boundary_conditions
  integer, parameter :: max_bnd_types=30

  type type_dirichlet_bc                           
    logical :: psi  
    logical :: u    
    logical :: zj   
    logical :: w    
    logical :: rho  
    logical :: T    
    logical :: Ti   
    logical :: Te   
    logical :: Vpar 
    logical :: rhon 
    logical :: rho_imp 
    logical :: nre  
    logical :: AR   
    logical :: AZ   
    logical :: A3  
  end type type_dirichlet_bc

  type type_natural_bc                           
    logical :: rho  
    logical :: T    
    logical :: Ti   
    logical :: Te   
    logical :: Vpar 
    logical :: rhon 
    logical :: rho_imp 
    logical :: nre  
  end type type_natural_bc

  type type_bcs                           
    type (type_dirichlet_bc) :: dirichlet
    type (type_natural_bc)   :: natural
    logical                  :: mach1 
  end type type_bcs

  type (type_bcs), dimension(max_bnd_types) :: bcs   
  ! ------------------------------------------------


  character(20)       :: numfmt     = "'_d',i5.5"
  character(20)       :: numfmt_rst = "'_r',i3.3"
  ! Identity of the processor
  integer  :: pglobal_id
  
  real*8, allocatable :: energies(:,:,:)   !< Magnetic and kinetic mode energies at timesteps.
  real*8, allocatable :: energies2(:,:,:)  !< global density and temperature at timesteps.
  real*8, allocatable :: energies3(:,:,:)  !< global currents (general and total eccd) at timesteps.
  real*8, allocatable :: energies4(:,:,:)  !< global applied eccd currents j1 and j2 at timesteps.

  character(len=3)    :: mode_type(n_tor) !< 'cos' or 'sin'
  character(len=3)    :: mode_coord_type(n_coord_tor) !< 'cos' or 'sin'
  
  !> Points used as limiters (see routine find_limiter)
  integer, parameter :: max_limiter = 1000 !< Maximum number of limiter points
  integer :: n_limiter                     !< Number of limiter points
  real*8  :: R_limiter(max_limiter)        !< R-positions of the limiter points
  real*8  :: Z_limiter(max_limiter)        !< Z-positions of the limiter points
  integer :: first_target_point		   !< index of the first target point on the limiter (for xpoint_grid_wall)
  integer :: last_target_point		   !< index of the last  target point on the limiter (does NOT need to be > first_target_point)
   
  ! Stellarator parameters
  logical :: gvec_grid_import     !< Generate grid fourier representation with GVEC
  logical :: extended_boundary    !< Choose if extended boundary conditions (Biot-Savart version) should be used, default (false) is grad_chi with Dommaschk potentials
  real*8  :: j_cutoff_rcoord      !< Radial location from which the current is set to zero as it approaches the boundary - rcoord corresponds to the normalised toroidal flux
  real*8  :: j_cutoff_sig         !< Radial width over which the current is ramped down to zero towards the boundary

  !> Points used as blocks to extend grid into complex wall structures, see https://www.jorek.eu/wiki/doku.php?id=wallgrid_tutorial
  real*8  :: surface_cross_tol                                                  !< Tolerance when looking for crossing of polar lines and surfaces, needs to be > 1.0
  real*8  :: eqdsk_psi_fact                                                     !< multiply eqdsk psi by factor for grid_inside_wall
  logical :: extend_existing_grid                                               !< Add patches to existing grid from restart file
  integer, parameter :: n_wall_blocks_max = 30                                  !< Maximum number of blocks (30 should be enough)
  integer :: n_wall_blocks                                                      !< Number of blocks
  integer, parameter :: n_wall_block_points_max = 20                            !< Max number of blocks points
  integer :: corner_block(n_wall_blocks_max)                                    !< =1 for a corner block ("left" side will also be wall-aligned)
  integer :: n_ext_block(n_wall_blocks_max)                                     !< Number of 'radial' grid points from the outermost flux surface to wall)
  logical :: n_ext_equidistant(n_wall_blocks_max)                               !< if true, radial spacing of grid points will be equidistant (not adapted)
  integer :: n_block_points_left (n_wall_blocks_max)                            !< Number of points on left side of block
  real*8  :: R_block_points_left (n_wall_blocks_max,n_wall_block_points_max)    !< R-positions of points on left side of block
  real*8  :: Z_block_points_left (n_wall_blocks_max,n_wall_block_points_max)    !< Z-positions of points on left side of block
  integer :: n_block_points_right(n_wall_blocks_max)                            !< Number of points on left side of block
  real*8  :: R_block_points_right(n_wall_blocks_max,n_wall_block_points_max)    !< R-positions of points on left side of block
  real*8  :: Z_block_points_right(n_wall_blocks_max,n_wall_block_points_max)    !< Z-positions of points on left side of block
  logical :: use_simple_bnd_types                                               !< convert Stan's bnd_types to Guido's bnd_types
  
  !> @name Define X-point geometry by geometrical properties
  !!
  !! \f[
  !! \Psi(\theta) =
  !!        -x_{shift}\sin(\theta)
  !!        +x_{left}\cos(\theta)
  !!        +x_{ampl}\left[
  !!            \left(\frac{x_{width}\cdot(\theta-x_{theta})}{x_{sig}}\right)^2-1
  !!          \right]exp\left[-\left(\frac{\theta-x_{theta}}{x_{sig}}\right)^2\right]
  !! \f]
  real*8  :: xampl    !< Allows to construct simple X-point cases by coefficients (modifies Psi boundary condition)
  real*8  :: xwidth   !< Allows to construct simple X-point cases by coefficients (modifies Psi boundary condition)
  real*8  :: xsig     !< Allows to construct simple X-point cases by coefficients (modifies Psi boundary condition)
  real*8  :: xtheta   !< Allows to construct simple X-point cases by coefficients (modifies Psi boundary condition)
  real*8  :: xshift   !< Allows to construct simple X-point cases by coefficients (modifies Psi boundary condition)
  real*8  :: xleft    !< Allows to construct simple X-point cases by coefficients (modifies Psi boundary condition)
  
  !> @name Heat and particle sources
  !!
  !! \f[
  !! S(\Psi_N) = S_0 \cdot \left[0.5 - 0.5 \tanh\left(\frac{\Psi_N - \Psi_{N,0}}{\sigma}\right) \right]
  !! \f]
  !!
  !! The following parameters can be set via the namelist input file:
  !! - \f$ S_0 \f$ denotes the source strenght (e.g., heatsource)
  !! - \f$ \Psi_{N,0} \f$ denotes the position around which the source is ramped down (e.g., heatsource_psin)
  !! - \f$ \sigma \f$ denotes the width over which the source is ramped down (e.g., heatsource_sig)
  !!
  real*8  :: particlesource                !< Particle source amplitude
  real*8  :: particlesource_psin           !< Position around which the source is ramped down
  real*8  :: particlesource_sig            !< Width over which the source is ramped down
  real*8  :: particlesource_gauss(5)       !< Additional Gaussian particle source amplitude
  real*8  :: particlesource_gauss_psin(5)  !< Position around which Gaussian source is set
  real*8  :: particlesource_gauss_sig(5)   !< Width over which Gaussian source is set
  real*8  :: edgeparticlesource            !< Edge particle source amplitude
  real*8  :: edgeparticlesource_psin       !< Position around which the edge particle source is located
  real*8  :: edgeparticlesource_sig        !< Width over which edge particle source extends
  real*8  :: neutral_line_source(10)       !< neutral inflow source
  real*8  :: neutral_line_R_start(10)      !< neutral inflow source (starting point of line source)
  real*8  :: neutral_line_Z_start(10)      !< neutral inflow source
  real*8  :: neutral_line_R_end(10)        !< neutral inflow source (end point of line source)
  real*8  :: neutral_line_Z_end(10)        !< neutral inflow source
  real*8  :: heatsource                    !< Heat source amplitude
  real*8  :: heatsource_e                  !< Electron heat source amplitude
  real*8  :: heatsource_i                  !< Ion heat source amplitude
  real*8  :: heatsource_psin               !< Position around which the source is ramped down
  real*8  :: heatsource_sig                !< Width over which the source is ramped down
  real*8  :: heatsource_e_psin             !< Position around which the electron source is ramped down
  real*8  :: heatsource_e_sig              !< Width over which the electron source is ramped down
  real*8  :: heatsource_i_psin             !< Position around which the ion source is ramped down
  real*8  :: heatsource_i_sig              !< Width over which the ion source is ramped down
  real*8  :: heatsource_gauss(5)           !< Additional Gaussian heat source amplitude
  real*8  :: heatsource_gauss_psin(5)      !< Position around which Gaussian source is located
  real*8  :: heatsource_gauss_sig(5)       !< Width over which Gaussian source extends
  real*8  :: heatsource_gauss_e(5)         !< Gaussian heat source for electrons
  real*8  :: heatsource_gauss_i(5)         !< Gaussian heat source for ions
  real*8  :: heatsource_gauss_e_psin(5)    !< Position around which electrons Gaussian source is located
  real*8  :: heatsource_gauss_e_sig(5)     !< Width over which electrons Gaussian source extends
  real*8  :: heatsource_gauss_i_psin(5)    !< Position around which ions Gaussian source is located
  real*8  :: heatsource_gauss_i_sig(5)     !< Width over which ions Gaussian source extends
  real*8  :: constant_imp_source           !< Adds a constant impurity source
  
  !> @name Hyper-resistivity, -viscosity and -diffusivities
  real*8  :: eta_num, visco_num, visco_par_num,                                      &
             D_perp_num, D_perp_num_tanh, D_perp_num_tanh_psin, D_perp_num_tanh_sig, &
             ZK_perp_num, ZK_i_perp_num, ZK_e_perp_num,                              &
             ZK_perp_num_tanh, ZK_perp_num_tanh_psin, ZK_perp_num_tanh_sig,          &
             ZK_i_perp_num_tanh, ZK_i_perp_num_tanh_psin, ZK_i_perp_num_tanh_sig,    &
             ZK_e_perp_num_tanh, ZK_e_perp_num_tanh_psin, ZK_e_perp_num_tanh_sig
  real*8  :: Dn_perp_num
  logical :: maintain_profiles             !< Add artificial sources to maintain initial rho and T profiles
                                           !! (diffusion acts on deviation from initial profiles)
					   !! at present only implemented for stellarator model 183

  !> @name Shock-capturing terms
  logical :: use_sc  !< Use shock-capturing stabilization
  real*8  :: D_perp_sc_num, D_par_sc_num, Dn_pol_sc_num, Dn_p_sc_num, D_perp_imp_sc_num, D_par_imp_sc_num
  real*8  :: ZK_perp_sc_num, ZK_par_sc_num, ZK_i_perp_sc_num, ZK_i_par_sc_num, ZK_e_perp_sc_num, ZK_e_par_sc_num
  real*8  :: visco_sc_num, visco_par_sc_num
  logical :: eta_num_T_dependent     !< Hyper-resistivity dependent on temperature? Otherwise constant.
  logical :: eta_num_psin_dependent  !< Give profile for Hyper-resistivity as function of \psi_N? Useful for 2D current flattening
  real*8  :: eta_num_prof(10)        !< Coefficients to specify \psi_N profile for hyper-resistivity
  logical :: visco_num_T_dependent!< Hyper-visocsity dependent on temperature? Otherwise constant.
  logical :: add_sources_in_sc    !< Whether to add effect of sources in shock-capturing stabilization or not

  !> @name VMS terms: The logical flag 'use_vms' enables to use variable
  !multiscale based stabilization in fullmhd model 750. The coefficients
  !vms_coeff_var are the real parameters to scale the stabilization added in
  !each equation. For brief description please look at the wiki page:
  ! https://www.jorek.eu/wiki/doku.php?id=vms
  logical    :: use_vms !< Use VMS stabilization in model 750 only
  real*8     :: vms_coeff_AR, vms_coeff_AZ, vms_coeff_A3
  real*8     :: vms_coeff_UR, vms_coeff_UZ, vms_coeff_Up
  real*8     :: vms_coeff_T, vms_coeff_Te, vms_coeff_Ti
  real*8     :: vms_coeff_rho, vms_coeff_rhon, vms_coeff_rhoimp
  
  !> @name Timestepping parameters
  real*8  :: tstep             		!< Size of the timesteps (\f$ \Delta t \f$)
  real*8  :: tstep_prev                 !< Previous time-step if using variable dt Gears
  real*8  :: tstep_n(10)       		!< Alternative to tstep: Up to ten values may be given
  integer :: nstep             		!< Number of timesteps to perform
  integer :: nstep_n(10)       		!< Alternative to nstep: Up to ten values may be given
  real*8  :: t_start           		!< Time value at the start of the code run (zero or from restart file)
  real*8  :: t_now             		!< Current time value in the simulation
  integer :: index_start       		!< Time step index at the beginning of the code run (zero or from restart file)
  integer :: index_now         		!< Current time step index
  real*8, allocatable :: xtime(:) 	!< Time values corresponding to the timesteps.
  character(len=80) :: time_evol_scheme !< Time evolution scheme to use (see [[time-integration|time_integration]])
  real*8  :: time_evol_theta   		!< Time evolution parameter theta (see [[time-integration|time_integration]])
  real*8  :: time_evol_zeta    		!< Time evolution parameter zeta (see [[time-integration|time_integration]])

  integer :: rst_hdf5                   !< Write hdf5 restart files if set to 1
  integer :: rst_hdf5_version           !< Write which version of hdf5 files?
  integer, parameter :: rst_hdf5_version_supported = 2 !< What is the highest version number supported?
  
  !> @name Machine name
  character(len=512) :: tokamak_device 	!< Name of the tokamak device we are simulating

  !> @name Analytical boundary of initial grid
  !!
  !! Analytical definition of the boundary of the non flux-aligned initial polar grid.
  !!
  !! - \f$ Z=Z_{geo} + a_{min} \epsilon \sin(\theta) \f$
  !!
  !! - for \f$ \theta < \pi \f$:
  !!   \f$ R=R_{geo} + a_{min} \cos\left[\theta+T_u\sin(\theta)+Q_u\sin(2\theta)\right] \f$
  !!
  !! - for \f$ \theta \ge \pi \f$:
  !!   \f$ R=R_{geo} + a_{min} \cos\left[\theta+T_l\sin(\theta)+Q_l\sin(2\theta)\right] \f$
  !!
  real*8  :: amin              !< Minor radius for polar grid construction, set to 1 if boundary is specified with R,Z points
  real*8  :: ellip             !< Ellipticity of polar grid (see analytical definition in phys_module.f90)
  real*8  :: tria_u            !< Upper triangularity of polar grid (see analytical definition in phys_module.f90)
  real*8  :: tria_l            !< Lower triangularity of polar grid (see analytical definition in phys_module.f90)
  real*8  :: quad_u            !< Upper quadrangularity of polar grid (see analytical definition in phys_module.f90)
  real*8  :: quad_l            !< Lower quadrangularity of polar grid (see analytical definition in phys_module.f90)
  
  !> @name Fourier expanded boundary of initial grid
  !! Boundary of the non flux-aligned initial polar grid given as Fourier series
  integer, parameter :: n_bnd_max = 3000 	!< Max number of entries in boundary points
  integer :: mf              		 	!< Number of entries in fbnd and fpsi
  real*8  :: fbnd(n_bnd_max)        		!< Fourier expansion of boundary
  real*8  :: fpsi(n_bnd_max)        		!< Fourier expansion of the poloidal flux at the boundary
  
  !> @name Numerical boundary of initial grid
  !! Numerical definition of the boundary of the non flux-aligned initial polar grid.
  integer :: n_boundary       			!< Number of points in R_boundary, Z_boundary, psi_boundary.
  real*8  :: R_boundary  (n_bnd_max)		!< Numerical R values defining the boundary
  real*8  :: Z_boundary  (n_bnd_max)		!< Numerical Z values defining the boundary
  real*8  :: psi_boundary(n_bnd_max)		!< Numerical values giving the poloidal flux at the boundary
  
  !> @name PF coils definition for initial equilibrium (MAST)
  !! Numerical definition of the PF coils definition for initial equilibrium (MAST)
  integer :: n_pfc            !< Number of coils, (OLD. for MAST...) use JOREK-STARWALL for coils instead [[jorek-starwall|JOREK-STARWALL]]
  real*8  :: Rmin_pfc(40)     !< Minimum R of coil, (OLD. for MAST...) use JOREK-STARWALL for coils instead [[jorek-starwall|JOREK-STARWALL]]
  real*8  :: Rmax_pfc(40)     !< Maximum R of coil, (OLD. for MAST...) use JOREK-STARWALL for coils instead [[jorek-starwall|JOREK-STARWALL]]
  real*8  :: Zmin_pfc(40)     !< Minimum Z of coil, (OLD. for MAST...) use JOREK-STARWALL for coils instead [[jorek-starwall|JOREK-STARWALL]]
  real*8  :: Zmax_pfc(40)     !< Maximum Z of coil, (OLD. for MAST...) use JOREK-STARWALL for coils instead [[jorek-starwall|JOREK-STARWALL]]
  real*8  :: current_pfc(40)  !< Current density in the coil, (OLD. for MAST...) use JOREK-STARWALL for coils instead [[jorek-starwall|JOREK-STARWALL]]
  
  !> @name current ropes definition for initial equilibrium (eg. merging flux ropes)
  !! Numerical definition of current ropes for initial equilibrium (eg. merging flux ropes)
  integer :: n_jropes          !< Number of ropes, 
  real*8  :: R_jropes(10)      !< R centre of rope
  real*8  :: Z_jropes(10)      !< Z centre of rope
  real*8  :: w_jropes(10)      !< width of rope
  real*8  :: current_jropes(10)!< Current inside the rope
  real*8  :: rho_jropes(10)    !< Density inside the rope
  real*8  :: T_jropes(10)      !< Temperature inside the rope
  
  !> @name Pellet-related input parameters
  real*8  :: pellet_amplitude  !< amplitude of density source (when pellet modelled as density source)
  real*8  :: pellet_R          !< major radius position pellet
  real*8  :: pellet_Z          !< Z position pellet
  real*8  :: pellet_phi        !< width of the pellet cloud (density source) in toroidal angle
  real*8  :: pellet_ellipse    !< the ellipticity of the pellet source
  real*8  :: pellet_radius     !< radius of the simulation pellet
  real*8  :: pellet_sig        !< width of smoothing of density source (arctan((r-pellet_radius)/pellet_sig))
  real*8  :: pellet_length     !< width of smoothing of density source in toroidal angle
  real*8  :: pellet_theta      !< orientation of the pellet ellipse
  real*8  :: pellet_psi        !< pellet_width in poloidal flux
  real*8  :: pellet_delta_psi  !< width of smoothing in poloidal flux
  real*8  :: pellet_velocity_R !< pellet velocity component radial direction
  real*8  :: pellet_velocity_Z !< pellet velocity component Z direction
  real*8  :: pellet_density    !< pellet atom number density (in units \f$10^{20} m^{-3}\f$)
  real*8  :: pellet_density_bg !< background species pellet atom number density (in units \f$10^{20} m^{-3}\f$)
  real*8  :: pellet_particles  !< the number of particles in the pellet (in units of \f$10^{20}\f$)
  logical :: use_pellet

  !> @name shared between MGI and SPI applications
  integer, parameter :: n_inj_max = 10 ! The hard coded maximum number of injections
  integer, parameter :: n_imp_max = 5  ! The hard coded maximum number of impurity species

  real*8  :: t_ns(n_inj_max)   !< MGI onset time (JOREK units)
  real*8  :: ns_amplitude(n_inj_max)  !< Amplitude of gas source
  real*8  :: ns_R(n_inj_max)   !< R position of gas source
  real*8  :: ns_Z(n_inj_max)   !< Z position of gas source
  real*8  :: ns_phi(n_inj_max) !< Phi position of gas source
  real*8  :: ns_radius         !< Poloidal radius of gas source
  real*8  :: ns_deltaphi       !< Toroidal extension of gas source
  real*8  :: ns_delta_minor_rad  !< Extension of gas source in the minor radial direction (if greater than 0.)
  real*8  :: ns_tor_norm         !< Gas source normalization factor related to its toroidal shape
  real*8  :: drift_distance(n_inj_max)    !< Shift the R position of the neutral deposition outward by drift_distance (in meters) for plasmoid drift
  real*8  :: energy_teleported(n_inj_max) !< Energy (in eV) teleported per atom to consider plasmoid drift effects

  character(len=80) :: imp_type(n_imp_max) !< Type of injected material or background impurity species: Argon, neon, ...
  logical :: use_imp_adas       !< Use open adas to calculate ionization, recombination and radiation coeffients for impurities


  !> @name Massive gas injection-related input parameters
  
  logical :: JET_MGI           !< Switch to use a JET-like MGI
  logical :: ASDEX_MGI         !< Switch to use an ASDEX-like MGI
  real*8  :: V_Dmv             !< Volume of the DMV reservoir
  real*8  :: P_Dmv             !< Pressure in the DMV reservoir (bar)
  real*8  :: A_Dmv             !< Cross sectional area of DMV (Disruption mitigation valve) pipe
  real*8  :: K_Dmv             !< Correction parameter describing the gas expansion near the pipe orifice
  real*8  :: L_tube            !< Pipe length
  real*8  :: ksi_ion            !< Energy cost of each ionization, ksi_ion / mu_0 / (gamma-1) / e = 13.7 eV
  real*8  :: delta_n_convection !< Switch to activate the convection term for neutrals (at the plasma velocity)
  real*8  :: nimp_bg(n_imp_max) !< Density of background impurities (in \f$m^{-3}\f$)
  integer :: index_main_imp     !< Index of the main impurity species (in imp_type and nimp_bg) solved with continuity equation
                               
  !> @name Shattered Pellet Injection related input parameters
  ! Note that the SPI share many of the MGI parameters. The code should return to simple MGI upon using_spi = false
  ! The reference spatial coordinate for shattered pellets are calculated using ns_R etc. 
  ! More information on the wiki: https://www.jorek.eu/wiki/doku.php?id=spi_tutorial
  logical :: using_spi          !< This determines whether to use SPI or traditional MGI; see [[spi_tutorial|SPI Tutorial]]
  real*8  :: spi_Vel_Rref(n_inj_max)   !< Reference velocity of pellet center along R upon injection
  real*8  :: spi_Vel_Zref(n_inj_max)   !< Reference velocity of pellet center along Z upon injection
  real*8  :: spi_Vel_RxZref(n_inj_max) !< Reference velocity of pellet center along RxZ direction upon injection
  real*8  :: spi_quantity(n_inj_max)   !< Total injected atom number for impurity SPI
  real*8  :: spi_quantity_bg(n_inj_max)!< Total injected atom number for background species SPI
  real*8  :: ns_radius_ratio           !< We are assuming a constant ratio between the radius of NG clouds
                                       !< and that of shattered pellets

  real*8  :: spi_Vel_diff(n_inj_max)   !< The velocity difference from the reference velocity
  real*8  :: spi_angle                 !< The vertex angle of spi spreading in terms of rad
  real*8  :: spi_L_inj(n_inj_max)      !< Distance between SPI nozzle and ns_R, ns_Z, ns_phi
  real*8  :: spi_L_inj_diff(n_inj_max) !< The position difference with respect to the point (ns_R, ns_Z, ns_phi)
  real*8  :: ns_phi_rotate             !< The toroidal position of rotated injection point
  real*8  :: tor_frequency             !< The rigid body rotation frequency

  real*8  :: ns_radius_min      !< This defines the minimum radius of neutral cloud for numerical reasons (in m)

  real*8, allocatable  :: xtime_spi_ablation(:,:)         !< The time history of SPI ablation
  real*8, allocatable  :: xtime_spi_ablation_rate(:,:)    !< The time history of SPI ablation rate
  real*8, allocatable  :: xtime_spi_ablation_bg(:,:)      !< The time history of SPI ablation for background species
  real*8, allocatable  :: xtime_spi_ablation_bg_rate(:,:) ! <The time history of SPI ablation rate for bg species

  real*8, allocatable  :: xtime_radiation(:)         !< The time history of radiated energy in SI unit
  real*8, allocatable  :: xtime_rad_power(:)         !< The time history of radiated power in SI unit
  real*8, allocatable  :: xtime_rad_cooling_power(:) !< The time history of radiative power loss from plasma in SI unit

  real*8, allocatable  :: xtime_E_ion(:)        !< The time history of the ionization potential energy in SI unit
  real*8, allocatable  :: xtime_E_ion_power(:)  !< Time derivative of xtime_E_ion
  real*8, allocatable  :: xtime_P_ei(:)         !< The time history of electron-ion energy exchange power

  integer :: n_spi(n_inj_max)   !< Number of shattered fragment injected for each injection
  integer :: n_spi_tot          !< Total number of shattered fragments injected
  integer :: n_inj              !< Number of injections
  integer :: spi_abl_model(n_inj_max)  !< Determine which type of ablation model is used.
                                       !< 0 for constant release rate, 1 for NGS model,
                                       !< 2 for Sergeev formula, 3 for Parks formula.
                                       !< For details see Nucl. Fusion 61 (2021) 026015 (23pp), 
                                       !< https://iopscience.iop.org/article/10.1088/1741-4326/abcbcb
  integer :: spi_rnd_seed(40)   !< Random seed array used for the generation of the SPI velocity spread

  character(len=256) :: spi_shard_file(n_inj_max)!< The name of the shard size file
  character(len=256) :: spi_plume_file(n_inj_max)!< The name of the shard information datafile (array)
  logical            :: spi_plume_hdf5           !< if 'spi_plume_file' is in HDF5format?
  logical            :: spi_abl_mag_reduction    !< Whether to use the magnetic reduction effect described in Eq.(27) of Nucl. Fusion 60 066027

  integer :: n_adas             !< Number of species to be traced by ADAS

  logical :: spi_tor_rot        !< Flag to turn on a rigid body toroidal plasma rotation for SPI
  logical :: spi_num_vol        !< Flag to turn on numerical integration of the gas source volumes from SPI

  type (type_SPI), allocatable :: pellets(:) !< Each element corresponds to one injected pellet (shard)

  character(len=512)            :: adas_dir    !< The directory of ADAS data file to be read
  type (adf11_all), allocatable :: imp_adas(:) !< The ADAS data for impurities
  type (coronal), allocatable   :: imp_cor(:)  !< The coronal equilibrium distribution of impurities

  logical :: output_prad_phi    !< Output Prad(phi) into a file using integrals_3D
  
  !> @name Fix boundary equilibrium parameters
  real*8  :: amix              !< Mix Poisson solution with previous one with a given factor
  real*8  :: equil_accuracy    !< Tolerance of the convergence for the fix-boundary equilibrium
  real*8  :: axis_srch_radius  !< Magnetic axis will be searched inside a circle with this radius
  real*8  :: delta_psi_GS      !< Expected psi_bnd - psi_axis for the final equilibrium  
  logical :: newton_GS_fixbnd  !< Newton instead of Picard iterations for fixed-boundary equilibria?
  logical :: newton_GS_freebnd !< Newton instead of Picard iterations for free-boundary equilibria?
  logical :: equil_initialized = .false. !< Workaround to prevent determining lcfs shape when the equilibrium hasn't been initialized (by restarting or calling equilibrium)
 
  !> @name Free boundary extension
  !! Input parameters related to the free boundary extension (folder vacuum/).
  logical :: freeboundary_equil      !< use a free or fixed boundary equilibrium? ([[jorek-starwall|JOREK-STARWALL]])
  logical :: freeboundary            !< use free or fixed boundary conditions in time-evolution? ([[jorek-starwall|JOREK-STARWALL]])
  logical :: resistive_wall          !< use a resistive or ideal wall? ([[jorek-starwall|JOREK-STARWALL]])
  logical :: freeb_equil_iterate_area !< iterate to a target area during freeboundary equilibrium limiter cases [[jorek-starwall-faqs|jorek_starwall]]
  real*8  :: amix_freeb              !< choose amix for freeboundary equilibrium
  real*8  :: equil_accuracy_freeb    !< Tolerance of the convergence for the freeboundary equilibrium
  logical :: freeb_change_indices    !< Exchange grid node indices to parallelize boundary integral
  
  !> @name Rectangular Grid
  !! Parameters defining a rectangular grid in R- and Z-directions in the poloidal plane.
  integer :: n_R               !< Number of grid points in R-direction (for rectangular grid) (see also [[grids#tutorials|here]])
  integer :: n_Z               !< Number of grid points in Z-direction (for rectangular grid)
  real*8  :: R_begin           !< Left boundary of grid in R-direction (for rectangular grid)
  real*8  :: R_end             !< Right boundary of grid in R-direction (for rectangular grid)
  real*8  :: Z_begin           !< Lower boundary of grid in Z-direction (for rectangular grid)
  real*8  :: Z_end             !< Upper boundary of grid in Z-direction (for rectangular grid)
  real*8  :: rect_grid_vac_psi !< Use a vacuum psi-bnd condition for squared-grid, ie. (rect_grid_vac_psi * R**2)

  
  !> @name Polar Grid
  !! Parameters defining a non flux-aligned polar grid in the poloidal plane.
  logical :: force_horizontal_Xline !< Force the grid line through Xpoint to be horizontal (instead of perp. to line between Xpoint and axis)
  integer :: n_radial          	    !< Number of radial grid points (for polar grid) (see also [[grids|here]])
  integer :: n_pol             	    !< Number of poloidal grid points (for polar grid)
  real*8  :: R_geo             	    !< Center of the grid (for polar grid)
  real*8  :: Z_geo             	    !< Center of the grid (for polar grid)
  real*8  :: psi_axis_init     	    !< Initial guess for Psi at the magnetic axis (for polar grid)
  real*8  :: XR_r(2)           	    !< Psi_N position of radial grid accumulation (two positions) (for polar grid) (also used for R-position in square-grid)
  real*8  :: SIG_r(2)          	    !< Width of grid accumulation (two positions) (for polar grid) (also used for R-width in square-grid)
  real*8  :: XR_tht(2)         	    !< Position of poloidal grid accumulation (0...1, two positions) (for polar grid)
  real*8  :: SIG_tht(2)        	    !< Width of grid accumulation (two positions) (for polar grid)
  real*8  :: XR_z(2)           	    !< Z-position of square grid accumulation (two positions) (for square grid)
  real*8  :: SIG_z(2)          	    !< Z-Width of grid accumulation (two positions) (for square grid)
  real*8  :: bgf_r, bgf_z           !< Background for meshac distribution for R-Z accumulation
  real*8  :: bgf_rpolar, bgf_tht    !< Background for meshac distribution for R-theta accumulation
  
  !> @name Flux surface grid
  !! Parameters defining a flux-aligned grid without X-point in the poloidal plane.
  integer :: n_flux            !< Number of radial grid points (for flux-aligned grid) (see also [[grids#tutorials|here]])
  integer :: n_tht             !< Number of poloidal grid points (for flux-aligned grid)
  real*8  :: xr1               !< Grid accumulation parameter (for flux-aligned grid)
  real*8  :: xr2               !< Grid accumulation parameter (for flux-aligned grid)
  real*8  :: sig1              !< Grid accumulation parameter (for flux-aligned grid)
  real*8  :: sig2              !< Grid accumulation parameter (for flux-aligned grid)
  integer :: m_pol_bc          !< Number of poloidal modes for Psi boundary condition in stellarator
  integer :: i_plane_rtree     !< The poloidal plane in a stellarator on which the RTree is to be built (RZ_minmax refers to this plane)
  
  !> @name Flux surface grid with X-point
  !! Parameters defining a flux-aligned grid with X-point in the poloidal plane.
  integer :: n_open            !< Number of 'radial' grid points in the open flux region - between the two separatrices if double-null
  integer :: n_outer           !< Number of 'radial' grid points in the open flux region on the outer side (LFS) if double-null
  integer :: n_inner           !< Number of 'radial' grid points in the open flux region on the inner side (HFS) if double-null
  integer :: n_private         !< Number of 'radial' grid points in the private flux region at the bottom
  integer :: n_leg             !< Number of 'poloidal' grid points along the divertor legs at the bottom
  integer :: n_leg_out         !< Number of 'poloidal' grid points along the divertor legs at the bottom on the LFS
  integer :: n_up_priv         !< Number of 'radial' grid points in the private flux region at the top (upper Xpoint or double-null)
  integer :: n_up_leg          !< Number of 'poloidal' grid points along the divertor legs at the top (upper Xpoint or double-null)
  integer :: n_up_leg_out      !< Number of 'poloidal' grid points along the divertor legs on the top on the LFS (upper Xpoint or double-null)
  integer :: n_ext             !< Number of 'radial' grid points from the outermost flux surface to wall)
  logical :: n_tht_equidistant !< switch on to get an equidistant poloidal distribution of elements in the core of the grid (psi<0.5)
  real*8  :: SIG_closed        !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_open          !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_outer         !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_inner         !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_private       !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_up_priv       !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_theta         !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_theta_up      !< Width with grid accumulation (for flux-aligned grid; only valid for double-null)
  real*8  :: SIG_leg_0         !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_leg_1         !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_up_leg_0      !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: SIG_up_leg_1      !< Width with grid accumulation (for flux-aligned grid)
  real*8  :: dPSI_open         !< Delta Psi grid extends into the open flux region (for flux-aligned grid)
  real*8  :: dPSI_outer        !< Delta Psi grid extends into the open flux region (for flux-aligned grid)
  real*8  :: dPSI_inner        !< Delta Psi grid extends into the open flux region (for flux-aligned grid)
  real*8  :: dPSI_private      !< Delta Psi grid extends into the private flux region (for flux-aligned grid)
  real*8  :: dPSI_up_priv      !< Delta Psi grid extends into the private flux region (for flux-aligned grid)
  
  !> @name Analytical heat, particle and neutral particles diffusivity parameters
  real*8  :: D_perp(10)    = 0.d0 !< Coefficients for perpendicular particle diffusion profile
  real*8  :: D_par                !< Parallel particle diffusion (usually not useful)
  real*8  :: D_perp_imp(10)= 0.d0 !< Coefficients for perpendicular imp particle diffusion profile
  real*8  :: D_par_imp            !< Parallel impurity particle diffusion (usually not useful)
  real*8  :: ZK_perp(10)   = 0.d0 !< Coefficients for perpendicular heat diffusion profile
  real*8  :: ZK_par               !< Parallel heat diffusion value in the plasma center
  real*8  :: ZK_par_max           !< Do not use larger parallel heat diffusion values for numerical reasons
  real*8  :: T_min_ZKpar          !< Do not use smaller parallel heat diffusion values below this MHD temperature (Ti+Te); JOREK units
  real*8  :: Ti_min_ZKpar         !< Do not use smaller parallel heat diffusion values below Ti; JOREK units
  real*8  :: Te_min_ZKpar         !< Do not use smaller parallel heat diffusion values below Te; JOREK units
  real*8  :: ZK_par_SpitzerHaerm  !< Spitzer-Haerm parallel heat diffusion value in the plasma center (assuming a Z=1 plasma with Te=Ti)
  real*8  :: ZK_i_perp(10) = 0.d0 !< Coefficients for perpendicular ion heat diffusion profile
  real*8  :: ZK_e_perp(10) = 0.d0 !< Coefficients for perpendicular electron heat diffusion profile
  real*8  :: ZK_i_par             !< Ion parallel heat diffusion coefficient in the plasma center
  real*8  :: ZK_e_par             !< Electron parallel heat diffusion coefficient in the plasma center
  real*8  :: ZK_i_par_SpitzerHaerm!< Spitzer-Haerm ion parallel heat diffusion value in the plasma center (assuming a Z=1 plasma)
  real*8  :: ZK_e_par_SpitzerHaerm!< Spitzer-Haerm electron parallel heat diffusion value in the plasma center (assuming a Z=1 plasma)
  real*8  :: D_neutral_x          !< Neutral particle diffusivity in R-direction
  real*8  :: D_neutral_y          !< Neutral particle diffusivity in Z-direction
  real*8  :: D_neutral_p          !< Neutral particle diffusivity in phi-direction
  logical :: ZKpar_T_dependent    !< Use a temperature dependent parallel heat diffusivity
  real*8  :: HW_coef(10)   = 0.d0 !< Coefficients for Hasegawa-Wakatani fluctuation term

  !> @name Numerical heat and particle diffusivity profiles
  character(len=512)  :: d_perp_file        !< ASCII file with perpendicular particle diffusion profile
  character(len=512)  :: d_perp_imp_file    !< ASCII file with perpendicular particle diffusion profile
  character(len=512)  :: zk_perp_file       !< ASCII file with perpendicular heat diffusion profile
  character(len=512)  :: zk_e_perp_file     !< ASCII file with perpendicular electron heat diffusion profile
  character(len=512)  :: zk_i_perp_file     !< ASCII file wtih perpendicular ion heat diffusion profile
  logical             :: num_d_perp         !< automatically set true if d_perp_file /= 'none'
  logical             :: num_d_perp_imp     !< automatically set true if d_perp_file /= 'none'
  logical             :: num_zk_perp        !< automatically set true if zk_perp_file /= 'none'
  logical             :: num_zk_e_perp      !< automatically set true if zk_e_perp_file /= 'none'
  logical             :: num_zk_i_perp      !< automatically set true if zk_i_perp_file /= 'none'
  integer             :: num_d_perp_len     !< Number of datapoints in d_perp profile
  integer             :: num_d_perp_len_imp !< Number of datapoints in d_perp profile for impurity
  integer             :: num_zk_perp_len    !< Number of datapoints in zk_perp profile
  integer             :: num_zk_e_perp_len  !< Number of datapoints in zk_e_perp profile
  integer             :: num_zk_i_perp_len  !< Number of datapoints in zk_i_perp profile
  real*8, allocatable :: num_d_perp_x(:)    !< Psi_N values of d_perp  profile
  real*8, allocatable :: num_d_perp_y(:)    !< D_perp values of d_perp profile
  real*8, allocatable :: num_d_perp_x_imp(:)!< Psi_N values of d_perp  profile for impurity
  real*8, allocatable :: num_d_perp_y_imp(:)!< D_perp values of d_perp profile for impurity
  real*8, allocatable :: num_zk_perp_x(:)   !< Psi_N values of zk_perp profile
  real*8, allocatable :: num_zk_perp_y(:)   !< ZK_perp values of zk_perp profile
  real*8, allocatable :: num_zk_e_perp_x(:) !< Psi_N values of zk_e_perp profile
  real*8, allocatable :: num_zk_e_perp_y(:) !< ZK_perp values of zk_e_perp profile
  real*8, allocatable :: num_zk_i_perp_x(:) !< Psi_N values of zk_i_perp profile
  real*8, allocatable :: num_zk_i_perp_y(:) !< ZK_perp values of zk_i_perp profile
  
  !> @name Analytical input profile for the density
  real*8  :: rho_0             !< Central normalized density (usually 1)
  real*8  :: rho_1             !< SOL normalized density
  real*8  :: rho_coef(10)      !< Density profile coefficients
  
  !> @name Numerical input profile for the density
  character(len=512)  :: rho_file        !< ASCII file the density profile is read from.
  logical             :: num_rho         !< automatically set true if rho_file /= 'none'
  integer             :: num_rho_len     !< Number of points in rho profile
  real*8, allocatable :: num_rho_x(:)    !< Psi_N values of rho profile points
  real*8, allocatable :: num_rho_y0(:)   !< Density values of rho profile
  real*8, allocatable :: num_rho_y1(:)   !< First derivatives of density profile (\f$ d\rho/d\Psi_N \f$)
  real*8, allocatable :: num_rho_y2(:)   !< Second derivatives of density profile (\f$ d^2\rho/d\Psi_N^2 \f$)
  real*8, allocatable :: num_rho_y3(:)   !< Third derivatives of density profile (\f$ d^3\rho/d\Psi_N^3 \f$)

  !> @name Analytical input profile for the temperature
  real*8  :: T_0            !< Central normalized temperature
  real*8  :: T_1            !< SOL normalized temperature
  real*8  :: T_coef(10)     !< Temperature profile coefficients
  real*8  :: Ti_0           !< Central ion normalized temperature
  real*8  :: Ti_1           !< SOL ion normalized temperature
  real*8  :: Ti_coef(10)    !< Ion temperature profile coefficients
  real*8  :: Te_0           !< Central ion normalized temperature
  real*8  :: Te_1           !< SOL ion normalized temperature
  real*8  :: Te_coef(10)    !< Ion temperature profile coefficients
  
  !> @name Numerical input profile for the temperature
  character(len=512)  :: T_file          !< ASCII file the temperature profile is read from.
  logical             :: num_T           !< automatically set true if T_file /= 'none'
  integer             :: num_T_len       !< Number of points in T profile
  real*8, allocatable :: num_T_x(:)      !< PsiN values of T profile points (PsiN values)
  real*8, allocatable :: num_T_y0(:)     !< Temperature values of T profile
  real*8, allocatable :: num_T_y1(:)     !< First derivatives of temperature profile (\f$ dT/d\Psi_N \f$)
  real*8, allocatable :: num_T_y2(:)     !< Second derivatives of temperature profile (\f$ d^2T/d\Psi_N^2 \f$)
  real*8, allocatable :: num_T_y3(:)     !< Third derivatives of temperature profile (\f$ d^3T/d\Psi_N^3 \f$)
  
  !> @name Numerical input profile for the ion temperature (model400)
  character(len=512)  :: Ti_file         !< ASCII file the ion temperature profile is read from.
  logical             :: num_Ti          !< is set true if T_file /= 'none'
  integer             :: num_Ti_len      !< Number of points in profile
  real*8, allocatable :: num_Ti_x(:)     !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_Ti_y0(:)    !< Values of temperature profile
  real*8, allocatable :: num_Ti_y1(:)    !< First derivatives of temperature profile (\f$ dT/d\Psi_N \f$)
  real*8, allocatable :: num_Ti_y2(:)    !< Second derivatives of temperature profile (\f$ d^2T/d\Psi_N^2 \f$)
  real*8, allocatable :: num_Ti_y3(:)    !< Third derivatives of temperature profile (\f$ d^3T/d\Psi_N^3 \f$)
  
  !> @name Numerical input profile for the electron temperature (model400)
  character(len=512)  :: Te_file         !< ASCII file the electron temperature profile is read from.
  logical             :: num_Te          !< is set true if T_file /= 'none'
  integer             :: num_Te_len      !< Number of points in profile
  real*8, allocatable :: num_Te_x(:)     !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_Te_y0(:)    !< Values of temperature profile
  real*8, allocatable :: num_Te_y1(:)    !< First derivatives of temperature profile (\f$ dT/d\Psi_N \f$)
  real*8, allocatable :: num_Te_y2(:)    !< Second derivatives of temperature profile (\f$ d^2T/d\Psi_N^2 \f$)
  real*8, allocatable :: num_Te_y3(:)    !< Third derivatives of temperature profile (\f$ d^3T/d\Psi_N^3 \f$)  
  
  !> @name Analytical input profile for the neutral density (model 500)
  real*8  :: rhon_0           !< Central value for the initial normalized neutral density
  real*8  :: rhon_1           !< SOL value for the initial normalized neutral density
  real*8  :: rhon_coef(10)    !< Coefficients for the intitial neutral density profile
  
  !> @name Numerical input profile for the neutral density (model 500)
  character(len=512)  :: rhon_file        !< ASCII file the neutral density profile is read from.
  logical             :: num_rhon         !< is set true if rho_file /= 'none'
  integer             :: num_rhon_len     !< Number of points in profile
  real*8, allocatable :: num_rhon_x(:)    !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_rhon_y0(:)   !< Values of neutral density profile
  real*8, allocatable :: num_rhon_y1(:)   !< First derivatives of neutral density profile (\f$ d\rhon/d\Psi_N \f$)
  real*8, allocatable :: num_rhon_y2(:)   !< Second derivatives of neutral density profile (\f$ d^2\rhon/d\Psi_N^2 \f$)
  real*8, allocatable :: num_rhon_y3(:)   !< Third derivatives of neutral density profile (\f$ d^3\rhon/d\Psi_N^3 \f$)
  
  !> @name Numerical input profile for Fprofile
  character(len=512)  :: Fprofile_file      !< ASCII file the Fprofile is read from.
  logical             :: num_Fprofile       !< is set true if Fprofile_file /= 'none'
  integer             :: num_Fprofile_len   !< Number of points in profile
  real*8, allocatable :: num_Fprofile_x(:)  !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_Fprofile_y0(:) !< Values of FFprime profile
  real*8, allocatable :: num_Fprofile_y1(:) !< First derivatives of Fprofile profile (\f$ dF/d\Psi_N \f$)
  real*8, allocatable :: num_Fprofile_y2(:) !< Second derivatives of Fprofile profile (\f$ d^2F/d\Psi_N^2 \f$)
  real*8, allocatable :: num_Fprofile_y3(:) !< Third derivatives of Fprofile profile (\f$ d^2F/d\Psi_N^2 \f$)

  !> @name Analytical input profile for the background Phi profile
  real*8  :: phi_0             !< Central background potential; (usually 1)
  real*8  :: phi_1             !< Edge background potential
  real*8  :: phi_coef(10)      !< potential profile coefficients

  !> @name Numerical input profile for the background potential profile
  character(len=512)  :: phi_file           !< ASCII file the potential profile is read from.
  logical             :: num_phi            !< is set true if potential_file /= 'none'
  integer             :: num_phi_len        !< Number of points in profile
  real*8, allocatable :: num_phi_x(:)       !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_phi_y0(:)      !< Values of potential profile
  real*8, allocatable :: num_phi_y1(:)      !< First derivatives of potential profile (\f$ d\Phi/d\rcoord_N \f$)
  real*8, allocatable :: num_phi_y2(:)      !< Second derivatives of potential profile (\f$ d^2\Phi/d\rcoord_N^2 \f$)
  real*8, allocatable :: num_phi_y3(:)      !< Third derivatives of potential profile (\f$ d^3\Phi/d\rcoord_N^3 \f$)

  real*8  :: nu_phi_source                  !< Friction coefficient of the n=0 background potential profile source term (>~ visco)

  !> @name Numerical input profile for Fprofile
  integer, parameter  :: n_Fprofile_internal_max = 300                 !< INTERNAL Max Size of F-profile
  integer             :: n_Fprofile_internal                           !< INTERNAL Size of F-profile
  real*8              :: Fprofile_internal   (n_Fprofile_internal_max) !< INTERNAL F-profile, from  FFprime integration
  real*8              :: Fprofile_internal_d1(n_Fprofile_internal_max) !< INTERNAL F-profile, from  FFprime integration (first derivative)
  real*8              :: Fprofile_internal_d2(n_Fprofile_internal_max) !< INTERNAL F-profile, from  FFprime integration (second derivative)
  real*8              :: Fprofile_internal_d3(n_Fprofile_internal_max) !< INTERNAL F-profile, from  FFprime integration (third derivative)
  real*8              :: Fprofile_psi_max                              !< INTERNAL max psi_norm of F-profile
  real*8              :: Fprofile_tolerance                            !< INTERNAL tolerance (in %) for accuracy of F-profile compared to input FFprime

  !> @name Analytical input profile for FFprime
  real*8  :: FF_0              !< FF' value in the plasma center
  real*8  :: FF_1              !< FF' value in the SOL
  real*8  :: FF_coef(10)       !< Coefficients for FF' profile
  
  !> @name Numerical input profile for FFprime
  character(len=512)  :: ffprime_file      !< ASCII file the FF' profile is read from.
  logical             :: num_ffprime       !< is set true if ffprime_file /= 'none'
  integer             :: num_ffprime_len   !< Number of points in profile
  real*8, allocatable :: num_ffprime_x(:)  !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_ffprime_y0(:) !< Values of FFprime profile
  real*8, allocatable :: num_ffprime_y1(:) !< First derivatives of FFprime profile (\f$ dFF'/d\Psi_N \f$)
  real*8, allocatable :: num_ffprime_y2(:) !< Second derivatives of FFprime profile (\f$ d^2FF'/d\Psi_N^2 \f$)

  !> --- Numerical input profiles for neoclassical coefficients
  logical             :: NEO              !< If .true. neoclassical effects are considered, (see [[neo|here]])
  character(len=512)  :: neo_file         !< ASCII file the aki and amu profiles is read from.
  logical             :: num_neo_file     !< automatically set true if neo_file /= 'none'
  integer             :: num_neo_len      !< Number of points in aki_neo, mu_neo profiles
  real*8, allocatable :: num_neo_psi(:)   !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_aki_value(:) !< numerical aki profile (PsiN values)
  real*8, allocatable :: num_amu_value(:) !< numerical amu profile (PsiN values)
  real*8              :: aki_neo_const    !< if ( (NEO) .and. (neo_file=='none')), this constant value is used for aki_neo
  real*8              :: amu_neo_const    !< if ( (NEO) .and. (neo_file=='none')), this constant value is used for amu_neo

  !> @name RMP profiles
  logical :: output_bnd_elements !< If .true., writes bnd nodes and bnd elements in files 'boundary_nodes.dat' and 'boundary_elements.dat'
  logical :: RMP_on              !< Activates RMPs on boundary if .true. (the old version without STARWALL)
  character(len=512)  :: RMP_psi_cos_file  !< ASCII file the profiles of psi_RMP_cos and derivatives are read from
  character(len=512)  :: RMP_psi_sin_file  !< ASCII file the profiles of psi_RMP_sin and derivatives are read from
  real*8  :: RMP_growth_rate, RMP_ramp_up_time  !< parameters for time dependence of psi_RMP: Sigmoid f(t)= 1/ (1 + exp(-RMP_growth_rate*(t-RMP_ramp_up_time/2)))
  real*8  :: RMP_start_time    !< time when RMP coils are activated (RMP_on = .t.)
  real*8, allocatable :: psi_RMP_cos(:)
  real*8, allocatable :: dpsi_RMP_cos_dR(:)
  real*8, allocatable :: dpsi_RMP_cos_dZ(:)
  real*8, allocatable :: psi_RMP_sin(:)
  real*8, allocatable :: dpsi_RMP_sin_dR(:)
  real*8, allocatable :: dpsi_RMP_sin_dZ(:)
  integer             :: RMP_har_cos,RMP_har_sin ! Harmonics numbers for RMP-cos and RMP-sin(for ex. ntor=3, nperiod=2,RMP_har_cos=2, RMP_har_sin=3)
  integer, parameter  :: N_RMP_max = 10                  ! Maximum of RMP harmonics to take into account
  integer             :: Number_RMP_harmonics            ! Number_RMP_harmonics < N_RMP_max. If only one harmonic,  Number_RMP_harmonics=1, by default it's =1 in models/preset_parameters.f90 
  integer             :: RMP_har_cos_spectrum(N_RMP_max) = 0 ! If only one harmonic,by default RMP_har_cos_spectrum(1)=RMP_har_cos; 
  integer             :: RMP_har_sin_spectrum(N_RMP_max) = 0 ! If only one harmonic,by default RMP_har_sin_spectrum(1)=RMP_har_sin;


  !> @name toroidal rotation profile
  real*8              :: V_0               !< analytical parallel rotation profile -- central value
  real*8              :: V_1               !< analytical parallel rotation profile -- SOL value
  real*8              :: V_coef(10) = 0.d0 !< analytical parallel rotation profile -- coefficients
  character(len=512)  :: R_Z_psi_bnd_file  !< ASCII file for R_boundary,Z_boundary, psi_boundary, with n_boundary size.
  character(len=512)  :: wall_file         !< ASCII file for external wall geometry, if n_ext is greater than zero.
  
  !> @name Numerical input profile for the toroidal rotation
  character(len=512)  :: rot_file        !< ASCII file the parallel rotation profile is read from (see normalized_velocity_profile)
  logical             :: num_rot         !< automatically set true if rot_file /= 'none'
  integer             :: num_rot_len     !< Number of points in rotation profile
  real*8, allocatable :: num_rot_x(:)    !< Radial positions of profile points (PsiN values)
  real*8, allocatable :: num_rot_y0(:)   !< Values of toroidal rotation profile
  real*8, allocatable :: num_rot_y1(:)   !< First derivatives of toroidal rotation profile with respect to $\Psi_{N}$
  real*8, allocatable :: num_rot_y2(:)   !< Second derivatives of toroidal rotation profile with respect to $\Psi_{N}$
  real*8, allocatable :: num_rot_y3(:)   !< Third derivatives of toroidal rotation profile with respect to $\Psi_{N}$
  logical             :: normalized_velocity_profile !< if true, reads the normalized velocity profile as flux function, else Omega_tor is read as flux function. 
  
  !> @name Coefficients for Dommaschk potentials; needed for vacuum field representation in stellarator models (see Dommaschk, CPC 40, 203, 1986)
  character(len=512)                                    :: domm_file !< Namelist file containing the coefficients for Dommaschk potentials
  logical                                               :: domm      !< automatically set to true if domm_file /= 'none'
  real*8                                                :: R_domm    !< Toroidally averaged radial position of the vacuum magnetic axis
  real*8, dimension(4,0:l_pol_domm,0:(n_coord_tor-1)/2) :: dcoef     !< Array containing the Dommaschk potential coefficients
  
  !> @name Global quantities determined in each time step
  real*8, allocatable :: R_axis_t(:), Z_axis_t(:), psi_axis_t(:), R_xpoint_t(:,:), Z_xpoint_t(:,:),           &
    psi_xpoint_t(:,:), R_bnd_t(:), Z_bnd_t(:), psi_bnd_t(:),                                                  &
    current_t(:), beta_p_t(:), beta_t_t(:), beta_n_t(:), density_in_t(:), density_out_t(:), pressure_in_t(:), &
    pressure_out_t(:), heat_src_in_t(:), heat_src_out_t(:), part_src_in_t(:), part_src_out_t(:),   &
    E_tot_t(:), Helicity_tot_t(:), Kin_perp_tot_t(:), thermal_tot_t(:), kin_par_tot_t(:), ohmic_tot_t(:),      &
    Wmag_tot_t(:), Ip_tot_t(:), flux_Pvn_t(:), flux_qpar_t(:), dE_tot_dt(:), flux_qperp_t(:), flux_kinpar_t(:), &
    dWmag_tot_dt(:), dthermal_tot_dt(:), dkinpar_tot_dt(:), dkinperp_tot_dt(:), friction_dissip_tot_t(:), &
    Magwork_tot_t(:), thmwork_tot_t(:), viscopar_dissip_tot_t(:), viscopar_flux_t(:), li3_t(:),      &
    li3_tot_t(:), part_src_tot_t(:), heat_src_tot_t(:), volume_t(:), area_t(:), mag_ener_src_tot(:), &
    dpart_tot_dt(:), part_flux_Dpar_t(:), part_flux_Dperp_t(:), part_flux_vpar_t(:), part_flux_vperp_t(:), & 
    dnpart_tot_dt(:), npart_tot_t(:), npart_flux_t(:), density_tot_t(:), flux_poynting_t(:), & 
    Px_t(:), Py_t(:), dPx_dt(:), dPy_dt(:), &
    thermal_e_tot_t(:), thermal_i_tot_t(:), visco_dissip_tot_t(:)

  !> @name gmres parameters
  integer             :: iter_precon        !< whenever the number of gmres iterations exceeds iter_precon, the preconditioning matrix is updated
  integer             :: max_steps_noUpdate !< whenever the steps without preconditioning matrix update exceeds max_steps_noUpdate, the preconditioning matrix is updated
  integer             :: gmres_m            !< gmres restart parameter (dimension)
  real*8              :: gmres_4            !< see gmres manual (error ratio between preconditioned and non-preconditioned error)
  real*8              :: gmres_tol          !< the tolerance for the gmres iterations to be seen as converged

  !> @name Taylor-Galerkin Stabilisation coefficients
  real*8              :: tgnum(n_var)   !< Coefficients for Taylor Galerkin stabilization for each equation separately
  real*8              :: tgnum_psi      !< Same as previous line, but avoiding equation indexing for model families 
  real*8              :: tgnum_u      
  real*8              :: tgnum_zj     
  real*8              :: tgnum_w      
  real*8              :: tgnum_rho    
  real*8              :: tgnum_T      
  real*8              :: tgnum_Ti     
  real*8              :: tgnum_Te     
  real*8              :: tgnum_vpar   
  real*8              :: tgnum_rhon   
  real*8              :: tgnum_rhoimp 
  real*8              :: tgnum_nre    
  real*8              :: tgnum_AR     
  real*8              :: tgnum_AZ    
  real*8              :: tgnum_A3    

  !> @name Flag to determine whether or not we keep current source term  
  logical             :: keep_current_prof !< Artificial current source to approximately keep the initial current profile, i.e., \f$\eta(j-j0)\f$?
  logical             :: init_current_prof !< Initialize the current source from the current profile present
  logical             :: current_prof_initialized !< Flag that is automatically set to true once the current source has been initialized to prevent accidental reinitialization when restarting
  
  !> @name Numerical parameters
  real*8              :: D_prof_neg         !< Particle diffusion coefficient in regions with negative background species density
  real*8              :: D_prof_neg_thresh  !< D_prof_neg becomes effective if r0-rimp0 < D_prof_neg_thresh
  real*8              :: D_prof_imp_neg_thresh  !< D_prof_neg becomes effective if rimp0 < D_prof_imp_neg_thresh
  real*8              :: D_prof_tot_neg_thresh  !< D_prof_neg becomes effective if r0 < D_prof_tot_neg_thresh
  real*8              :: ZK_prof_neg        !< Perp. heat diffusion coefficient in regions with negative temperature
  real*8              :: ZK_par_neg         !< Parallel diffusion coefficient in regions with negative temperature
  real*8              :: ZK_prof_neg_thresh !< ZK_prof_neg becomes effective if T < ZK_prof_neg_thresh
  real*8              :: ZK_par_neg_thresh  !< ZK_par_neg becomes effective if T < ZK_par_neg_thresh
  real*8              :: ZK_e_prof_neg        !< Perp. heat diffusion coefficient in regions with negative temperature
  real*8              :: ZK_e_par_neg         !< Parallel diffusion coefficient in regions with negative temperature
  real*8              :: ZK_e_prof_neg_thresh !< ZK_e_prof_neg becomes effective if T < ZK_e_prof_neg_thresh
  real*8              :: ZK_e_par_neg_thresh  !< ZK_e_par_neg becomes effective if T < ZK_e_par_neg_thresh
  real*8              :: ZK_i_prof_neg        !< Perp. heat diffusion coefficient in regions with negative temperature
  real*8              :: ZK_i_par_neg         !< Parallel diffusion coefficient in regions with negative temperature
  real*8              :: ZK_i_prof_neg_thresh !< ZK_i_prof_neg becomes effective if T < ZK_i_prof_neg_thresh
  real*8              :: ZK_i_par_neg_thresh  !< ZK_i_par_neg becomes effective if T < ZK_i_par_neg_thresh
  real*8              :: D_imp_extra_R           !< Additional impurity diffusivity in R-direction
  real*8              :: D_imp_extra_Z           !< Additional impurity diffusivity in Z-direction
  real*8              :: D_imp_extra_p           !< Additional impurity diffusivity in phi-direction
  real*8              :: D_imp_extra_neg         !< Additional impurity diffusion coefficient in regions with negative impurity density
  real*8              :: D_imp_extra_neg_thresh  !< D_imp_extra_neg becomes effective if rho_imp < D_imp_extra_neg_thresh
  real*8              :: T_min              !< minimum temperature (limits on the temperature dependence of resistivity etc.) value in jorek units: 2.01d-5*central_density*Tmin_ev (preset central_density = 1, 20 eV)
  real*8              :: rho_min            !< minimum density
  real*8              :: ne_SI_min          !< minimum e density (in SI unit) below which we cut-off the radiation loss
  real*8              :: Te_eV_min          !< minimum temperature (in eV) below which we cut-off the radiation loss
  real*8              :: rn0_min            !< minimum impurity density (in JU) for radiation loss cut-off
  real*8              :: T_min_neg          !< minimum temperature,used for correcting negative values,in jorek units: 2.01d-5*central_density*Tmin_ev (preset central_density = 1, 20 eV)  
  real*8              :: rho_min_neg        !< minimum density, used for correcting negative values  
  real*8              :: implicit_heat_source !< Choose = 1.d0 to fully switch on the implicit heat source for numerical stabilization
  
  integer             :: n_tor_fft_thresh   !< If n_tor >= n_tor_fft_thresh, element_matrix_fft will be used
  integer*8           :: fftw_plan          !< Required for FFTW library
  real*8              :: corr_neg_temp_coef(2) !< Parameters used in models/corr_neg.f90
  real*8              :: corr_neg_dens_coef(2) !< Parameters used in models/corr_neg.f90

  !> @name ECCD current sources
  real*8  :: jecamp             ! parameter, not to be confused with jec_source in element_matrix.f90
  real*8  :: jec_pos1, jec_pos2, jec_pos3, jec_pos4
  real*8  :: jec_width, jec_width2
  real*8  :: nu_jec_fast         ! 1/collision frequency
  real*8  :: nu_jec1_fast,nu_jec2_fast         ! 1/collision frequency
  real*8  :: mod_jec            ! extra parameters for ECCD
  real*8  :: JJ_par             ! velocity of resonent electrons
  real*8  :: jw1,jw2,jw3        ! parameters to determine current source

  !> @name Flag for thermalization term
  logical             :: thermalization ! If true turns on the ion-electron thermalization term

  !> @name (Currently unused)
  real*8  :: zjz_0, zjz_1,  zj_coef(10)
  real*8  :: D_neutral

  !> @name Particles-related input parameters
  logical :: use_particles       ! Flag if simulation contains particles
  integer :: n_aux_var = n_var   ! number of variables in aux_node_list (= n_var is temporary)
  integer :: n_diag_var = n_var  ! number of variables in diag_node_list (= n_var is temporary)
  logical :: restart_particles
  logical :: use_ncs          !< use neutral particles
  logical :: use_ccs          !< use current coupling scheme for fast particles
  logical :: use_pcs          !< use pressure coupling scheme for fast particles
  logical :: use_pcs_full     !< use full tensor pressure coupling scheme for fast particles
  logical :: use_kn_cx        !< switch on sputtering         (in particle module)
  logical :: use_marker       !< This flag determines whether to use marker particles to treat impurity (Placeholder)
  logical :: use_kn_sputtering   !< switch on charge-exchange    (in particle module)
  logical :: use_kn_ionisation   !< switch on ionisation         (in particle module)
  logical :: use_kn_recombination !< switch on recombination         (in particle module)
  logical :: use_kn_puffing       !< switch on particle puffing         (in particle module)
  logical :: use_kn_line_radiation !< switch on line radiation         (in particle module)
  real*8  :: n_particles      !< the number of particles (real on purpose)
  real*8  :: tstep_particles  !< the time step for the particles
  integer :: nstep_particles  !< the number of particle time steps
  integer :: nsubstep_particles !< the number of particles substeps (without projection)
  real*8  :: filter_perp      !< particle projection smoothing parameter, poloidal plane
  real*8  :: filter_hyper     !< particle projection smoothing parameter, poloidal plane
  real*8  :: filter_par       !< particle projection smoothing parameter, parallel direction
  real*8  :: filter_perp_n0   !< particle projection smoothing parameter, poloidal plane (n=0)
  real*8  :: filter_hyper_n0  !< particle projection smoothing parameter, poloidal plane (n=0)
  real*8  :: filter_par_n0    !< particle projection smoothing parameter, parallel direction (n=0)

  real*8  :: puff_rate        !< physical atoms/sec puffed (shared over 2 places)
  real*8  :: r_valve          !< radius of poloidal circular source
  real*8  :: R_valve_loc      !< R position valve 1
  real*8  :: Z_valve          !< Z position valve 1
  real*8  :: R_valve_loc2     !< R position valve 2
  real*8  :: Z_valve2         !< Z position valve 2
  integer :: n_puff           !< superparticles used per puffing action per valve
    
  !> @name Mode families preconditioner parameters
  integer, parameter :: n_fam_max = 100               !< maximum number of families
  integer :: n_mode_families                          !< number of families
  logical :: autodistribute_modes                     !< use automatic or manual mode distribution
  integer :: modes_per_family(n_fam_max)              !< Number of modes in families
  integer :: mode_families_modes(n_fam_max,n_fam_max) !< Mode numbers (i_tor) belonging to each family; first index: family number
  real*8  :: weights_per_family(n_fam_max)            !< Multiplication factor of family's contribution to the full solution
  logical :: autodistribute_ranks                     !< use automatic or manual rank distribution
  integer :: ranks_per_family(n_fam_max)              !< Number of MPI ranks per mode families

  !> @name Manual setting of random seed (for testing)
  logical :: use_manual_random_seed                   !< whether the random seed should be manually set
  integer :: manual_seed                              !< the manually set seed value
  logical :: use_fixed_rng_value                      !< forcibly set all rng outputs to return a specific value (set by fixed_rng_value, use this for debugging and testing only)
  real*8  :: fixed_rng_value                          !< the value the fixed rng is set to when using use_fixed_rng_value
  contains
  
end module phys_module
