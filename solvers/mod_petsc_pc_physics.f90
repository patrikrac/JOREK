module mod_petsc_pc_physics
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  use mod_petsc_pc_physics_construction, only: &
       create_variable_index_sets, extract_sub_block, &
       compute_schur_corrected_block_psi, &
       compute_schur_corrected_block_u, compute_schur_corrected_block_21, &
       compute_schur_corrected_block_61, compute_schur_corrected_block_exact, &
       compute_explicit_preconditioned_matrix, &
       compute_full_momentum_schur_exact, &
       setup_S_PBP_diag_shell, materialize_S_PBP_diag_aij, &
       setup_block_ksp, setup_constraint_mass_ksp, &
       setup_block_ksp_amg_krylov, setup_block_ksp_hypre_amg_krylov, &
       setup_alfven_block_ksp, setup_rho_block_ksp, setup_T_block_ksp, &
       assemble_monolithic_4x4, assemble_probed_exact_4x4, &
       verify_alfven_2x2_segregated, verify_reduced_pde_operator, &
       verify_schur_factorization_4x4, verify_schur_approx_4x4
  use mod_petsc_pc_physics_element, only: &
       petsc_create_pc_matrices, petsc_assemble_pc_matrices, &
       petsc_update_physics_pc_ctx, petsc_test_pc_matrix
  use mod_petsc_pc_physics_apply, only: physics_pc_apply
  implicit none
  private
  public :: petsc_setup_physics_pc, petsc_create_pc_matrices, &
            petsc_assemble_pc_matrices, petsc_update_physics_pc_ctx, &
            petsc_physics_pc_build_reduced

contains

  !--------------------------------------------------------------------
  !> Register the physics-based PCSHELL on an existing KSP.
  !--------------------------------------------------------------------
  subroutine petsc_setup_physics_pc(ksp, A)
    KSP, intent(inout) :: ksp
    Mat, intent(in)    :: A
    PC :: pc
    PetscErrorCode :: ierr

    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetApply(pc, physics_pc_apply, ierr))
    g_ctx%initialized = .true.
  end subroutine petsc_setup_physics_pc



  !--------------------------------------------------------------------
  !> Build the reduced 4x4 system from the full system matrix.
  !!
  !! Extracts sub-blocks, approx mass, forms Schur corrections,
  !! and sets up sub-KSPs for the diagonal blocks.
  !--------------------------------------------------------------------
  subroutine petsc_physics_pc_build_reduced(A_full)
    use mod_parameters, only: n_var, n_tor, n_degrees, var_psi, var_u, var_zj, var_w, var_rho, var_T
    use phys_module, only: debug_physics_pc, physics_pc_monolithic, &
                           physics_pc_multi_step, physics_pc_probe_exact, &
                           physics_pc_sub_blocks, physics_pc_verify_spbp, &
                           physics_pc_verify_reduced, physics_pc_verify_schur, &
                           physics_pc_schur_approx
    use mod_petsc_matrix_analysis, only: petsc_mat_convert_spectrum, petsc_mat_equilibrate, &
                                         petsc_mat_diff_norm

    Mat, intent(in) :: A_full

    PetscErrorCode :: ierr
    PetscInt :: bs_ntor
    integer :: comm, my_id, mpierr
    logical :: first_time
    PetscReal :: norm_val
    Mat :: B_tmp ! Store the preconditioned matrix B = M^{-1} A
    Mat :: S_PBP_diag_aij  ! materialized diagnostic shell (physics_pc_verify_spbp)
    PetscViewer :: viewer
    !Mat :: A_eq
    !Vec :: dr, dc

    call PetscObjectGetComm(A_full, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)

    ! --- PC mode mutual exclusivity check: exactly one mode must be set ---
    block
      integer :: n_modes
      n_modes = 0
      if (physics_pc_monolithic) n_modes = n_modes + 1
      if (physics_pc_multi_step) n_modes = n_modes + 1
      if (physics_pc_sub_blocks) n_modes = n_modes + 1
      if (n_modes > 1) then
        if (my_id == 0) write(*,'(A)') &
          "[Physics PC] ERROR: physics_pc_{monolithic,multi_step,sub_blocks} are " // &
          "mutually exclusive; set exactly one."
        ierr = 1
        return
      endif
      if (n_modes == 0) then
        if (my_id == 0) write(*,'(A)') &
          "[Physics PC] ERROR: no PC mode selected; set exactly one of " // &
          "physics_pc_{monolithic,multi_step,sub_blocks}."
        ierr = 1
        return
      endif
    end block

    first_time = .not. g_ctx%reduced_ready

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC] Building reduced 4x4 system (extracted blocks)..."
    endif

    ! --- Step 1: Create index sets (first time only) ---
    if (.not. g_ctx%is_created) then
      call create_variable_index_sets(A_full, comm)
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Index sets created"
    endif

    ! --- Extract sub-blocks from full system ---
    ! Diagonal blocks of the constraint equations
    call extract_sub_block(A_full, var_zj, var_zj, g_ctx%B_33, first_time)
    call extract_sub_block(A_full, var_w,  var_w,  g_ctx%B_44, first_time)

    ! Constraint operator blocks (always needed for back-sub and Schur correction)
    call extract_sub_block(A_full, var_zj,  var_psi, g_ctx%B_31, first_time)
    call extract_sub_block(A_full, var_w,   var_u,   g_ctx%B_42, first_time)

    ! Coupling blocks TO j,w (always needed for RHS correction)
    call extract_sub_block(A_full, var_psi, var_zj, g_ctx%B_13, first_time)
    call extract_sub_block(A_full, var_u,   var_zj, g_ctx%B_23, first_time)
    call extract_sub_block(A_full, var_u,   var_w,  g_ctx%B_24, first_time)
    call extract_sub_block(A_full, var_T,   var_zj, g_ctx%B_63, first_time)

    ! Diagonal blocks of the 4x4 system (only extract if not reassembling)
    call extract_sub_block(A_full, var_psi, var_psi, g_ctx%B_11, first_time)
    call extract_sub_block(A_full, var_u,   var_u,   g_ctx%B_22, first_time)
    call extract_sub_block(A_full, var_rho, var_rho, g_ctx%B_55, first_time)
    call extract_sub_block(A_full, var_T,   var_T,   g_ctx%B_66, first_time)

    ! Off-diagonal blocks of the 4x4 system (always needed for coupled mode)
    call extract_sub_block(A_full, var_psi, var_u,   g_ctx%B_12, first_time)
    call extract_sub_block(A_full, var_psi, var_T,   g_ctx%B_16, first_time)
    call extract_sub_block(A_full, var_u,   var_psi, g_ctx%B_21, first_time)
    call extract_sub_block(A_full, var_u,   var_rho, g_ctx%B_25, first_time)
    call extract_sub_block(A_full, var_u,   var_T,   g_ctx%B_26, first_time)
    call extract_sub_block(A_full, var_rho, var_psi, g_ctx%B_51, first_time)
    call extract_sub_block(A_full, var_rho, var_u,   g_ctx%B_52, first_time)
    call extract_sub_block(A_full, var_T,   var_psi, g_ctx%B_61, first_time)
    call extract_sub_block(A_full, var_T,   var_u,   g_ctx%B_62, first_time)


    if (my_id == 0) write(*,'(A)') "[Physics PC]   Sub-blocks extracted (21 blocks)"

    ! Temporary diagnostic print: Block norms 
    ! call MatNorm(g_ctx%B_11, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_11||_F = ", norm_val
    ! call MatNorm(g_ctx%B_11, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_11||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_12, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_12||_F = ", norm_val
    ! call MatNorm(g_ctx%B_12, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_12||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_13, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_13||_F = ", norm_val
    ! call MatNorm(g_ctx%B_13, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_13||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_16, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_16||_F = ", norm_val
    ! call MatNorm(g_ctx%B_16, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_16||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_21, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_21||_F = ", norm_val
    ! call MatNorm(g_ctx%B_21, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_21||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_22, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_22||_F = ", norm_val
    ! call MatNorm(g_ctx%B_22, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_22||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_23, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_23||_F = ", norm_val
    ! call MatNorm(g_ctx%B_23, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_23||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_24, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_24||_F = ", norm_val
    ! call MatNorm(g_ctx%B_24, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_24||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_25, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_25||_F = ", norm_val
    ! call MatNorm(g_ctx%B_25, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_25||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_26, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_26||_F = ", norm_val
    ! call MatNorm(g_ctx%B_26, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_26||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_31, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_31||_F = ", norm_val
    ! call MatNorm(g_ctx%B_31, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_31||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_33, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_33||_F = ", norm_val
    ! call MatNorm(g_ctx%B_33, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_33||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_42, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_42||_F = ", norm_val
    ! call MatNorm(g_ctx%B_42, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_42||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_44, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_44||_F = ", norm_val
    ! call MatNorm(g_ctx%B_44, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_44||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_51, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_51||_F = ", norm_val
    ! call MatNorm(g_ctx%B_51, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_51||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_52, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_52||_F = ", norm_val
    ! call MatNorm(g_ctx%B_52, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_52||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_55, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_55||_F = ", norm_val
    ! call MatNorm(g_ctx%B_55, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_55||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_61, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_61||_F = ", norm_val
    ! call MatNorm(g_ctx%B_61, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_61||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_62, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_62||_F = ", norm_val
    ! call MatNorm(g_ctx%B_62, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_62||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_63, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_63||_F = ", norm_val
    ! call MatNorm(g_ctx%B_63, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_63||_∞ = ", norm_val
    ! call MatNorm(g_ctx%B_66, NORM_FROBENIUS, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_66||_F = ", norm_val
    ! call MatNorm(g_ctx%B_66, NORM_INFINITY, norm_val, ierr)
    ! if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_66||_∞ = ", norm_val

    ! --- Set up KSPs for elliptic constraint mass matrices ---
    ! (Must be done before Schur correction so MUMPS factorization is available)
    call setup_constraint_mass_ksp(g_ctx%ksp_Mj, g_ctx%B_33, comm, first_time, "Mj constraint-mass KSP")
    call setup_constraint_mass_ksp(g_ctx%ksp_Mw, g_ctx%B_44, comm, first_time, "Mw constraint-mass KSP")
    g_ctx%ksp_elliptic_created = .true.

    ! --- Step 4b: Form Schur-corrected blocks (element-assembled corrections) ---
    call compute_schur_corrected_block_psi(g_ctx%B_11, g_ctx%Atilde_11, first_time)
    !call compute_schur_corrected_block_exact(g_ctx%B_33, g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, g_ctx%Atilde_11, first_time)
    call compute_schur_corrected_block_u(g_ctx%B_22, g_ctx%Atilde_22, first_time)
    !call compute_schur_corrected_block_exact( g_ctx%B_44, g_ctx%B_22, g_ctx%B_24, g_ctx%B_42, g_ctx%Atilde_22, first_time)

    ! Off-diagonal Schur corrections from element-assembled K_21 / K_61
    call compute_schur_corrected_block_21(g_ctx%B_21, g_ctx%Atilde_21, first_time)
    !call compute_schur_corrected_block_exact(g_ctx%B_33, g_ctx%B_21, g_ctx%B_23, g_ctx%B_31, g_ctx%Atilde_21, first_time)
    call compute_schur_corrected_block_61(g_ctx%B_61, g_ctx%Atilde_61, first_time)
    !call compute_schur_corrected_block_exact(g_ctx%B_33, g_ctx%B_61, g_ctx%B_63, g_ctx%B_31, g_ctx%Atilde_61, first_time)
    if (my_id == 0) write(*,'(A)') "[Physics PC]   Computed Schur-corrected blocks"

    if (debug_physics_pc) then
      call MatNorm(g_ctx%Atilde_11, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_11||_F = ", norm_val
      call MatNorm(g_ctx%Atilde_22, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_22||_F = ", norm_val
      call MatNorm(g_ctx%Atilde_21, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_21||_F = ", norm_val
      call MatNorm(g_ctx%Atilde_61, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Atilde_61||_F = ", norm_val
    end if

    ! --- Offline S_PBP verification (namelist: physics_pc_verify_spbp).  When .true.,
    !     build the FULL momentum Schur S_u (channels A+B+C) as the reference, build the
    !     diagnostic S_PBP MatShell (Atilde_11^-1 -> (1+zeta)^-1 M_j^-1, channels B/C kept
    !     exact), and measure sigma(S_PBP_diag^-1 S_u). Skips the Stage-1 S_u->S_PBP
    !     overwrite, so it is a diagnostic arm, not a production path.  Expensive: builds
    !     the full momentum Schur and several spectra, hence its own flag rather than
    !     debug_physics_pc.

     !call compute_full_momentum_schur_exact(g_ctx%S_u, first_time)

    if (physics_pc_verify_spbp) then
      ! Convert spectrum of the full matrix
      call petsc_mat_convert_spectrum(A_full, "A_full", .false.)

      ! Exact reference: full momentum Schur (channels A+B+C), held fixed across arms.
      call compute_full_momentum_schur_exact(g_ctx%S_u, first_time)
      call petsc_mat_convert_spectrum(g_ctx%S_u, "S_u_exact", .false.)

      ! Consistent-mass S_PBP shell, materialized to AIJ, then raw + preconditioned
      ! spectra. spec(S_PBP_diag^-1 S_u) isolates the Atilde_11^-1 -> M_j^-1 quality.
      call setup_S_PBP_diag_shell(comm, first_time)
      call materialize_S_PBP_diag_aij(S_PBP_diag_aij, .true.)
      call petsc_mat_convert_spectrum(S_PBP_diag_aij, "S_PBP_diag", .false.)
      call compute_explicit_preconditioned_matrix(S_PBP_diag_aij, g_ctx%S_u, B_tmp, .true.)
      call petsc_mat_convert_spectrum(B_tmp, "S_PBP_diag_inv_S_u", .false.)
      call MatDestroy(S_PBP_diag_aij, ierr)
      call MatDestroy(B_tmp, ierr)
    else
      !call compute_schur_corrected_block_exact(g_ctx%Atilde_11, g_ctx%Atilde_22, &
      !                                         g_ctx%Atilde_21, g_ctx%B_12, g_ctx%S_u, first_time)
    endif

    if (debug_physics_pc) then
      call petsc_test_pc_matrix(g_ctx%B_33, "B_33", .true., my_id)
      call petsc_test_pc_matrix(g_ctx%B_44, "B_44", .true., my_id)
      ! call petsc_test_pc_matrix(g_ctx%Atilde_11, "Atilde_11", .false., my_id)
      ! call petsc_test_pc_matrix(g_ctx%Atilde_22, "Atilde_22", .false., my_id)

      !call petsc_test_pc_matrix(g_ctx%S_PBP, "S_PBP", .false., my_id)
      !call petsc_test_pc_matrix(g_ctx%S_u, "S_u", .false., my_id)

      call petsc_test_pc_matrix(g_ctx%B_55, "B_55", .false., my_id)
      call petsc_test_pc_matrix(g_ctx%B_66, "B_66", .false., my_id)

      ! ! --- Compute block spectra for diagnostics ---
      ! call petsc_mat_convert_spectrum(g_ctx%Atilde_11, "Atilde_11", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%Atilde_22, "Atilde_22", .false.)

      ! call petsc_mat_convert_spectrum(g_ctx%Atilde_21, "Atilde_21", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%Atilde_61, "Atilde_61", .false.)

      ! call petsc_mat_convert_spectrum(g_ctx%B_12, "B_12", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%B_16, "B_16", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%B_25, "B_25", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%B_26, "B_26", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%B_51, "B_51", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%B_52, "B_52", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%B_62, "B_62", .false.)

      ! call petsc_mat_convert_spectrum(g_ctx%B_55, "B_55", .false.)
      ! call petsc_mat_convert_spectrum(g_ctx%B_66, "B_66", .false.)

      !call petsc_mat_convert_spectrum(g_ctx%S_PBP, "S_PBP_2", .false.)
      !call petsc_mat_convert_spectrum(g_ctx%S_u, "S_u", .false.)
      !call petsc_mat_convert_spectrum(B_tmp, "S_prec", .false.)
    endif

    !call MatCopy(g_ctx%S_u, g_ctx%S_PBP, DIFFERENT_NONZERO_PATTERN, ierr)

    ! --- Step 5: Set up solver(s) ---
    if (physics_pc_monolithic .or. physics_pc_multi_step) then
      ! When probe_exact=.true., monolithic only builds A_approx for the diagnostic;
      ! the KSP is owned entirely by assemble_probed_exact_4x4 (avoids stale-factor issues).
      if (physics_pc_monolithic) call assemble_monolithic_4x4(comm, first_time, my_id, physics_pc_probe_exact)
      
      ! Sub-block KSPs needed by the three-step (multi_step) apply:
      !   ksp_psi (Atilde_11)  -> magnetic predictor
      !   ksp_rho (B_55)       -> transport correction (rho)
      !   ksp_T   (B_66)       -> transport correction (T)
      ! Guarded on multi_step so the monolithic path is unchanged (it uses only
      ! ksp_reduced; setting these up there would add three unused factorizations).
      if (physics_pc_multi_step) then
        call setup_block_ksp(g_ctx%ksp_psi, g_ctx%Atilde_11, comm, first_time, "psi predictor KSP (Atilde_11)")
        call setup_block_ksp(g_ctx%ksp_rho, g_ctx%B_55,      comm, first_time, "rho-block KSP")
        call setup_block_ksp(g_ctx%ksp_T,   g_ctx%B_66,      comm, first_time, "T-block KSP")
        g_ctx%ksp_created = .true.
      endif

      !       ! Build K_A (psi,u) 2x2 MatNest from refs to existing Atilde_*/B_12.
      ! block
      !   Mat :: mats_nest_A(4)
      !   if (.not. first_time) call MatDestroy(g_ctx%K_A_block, ierr)
      !   mats_nest_A(1) = g_ctx%Atilde_11
      !   mats_nest_A(2) = g_ctx%B_12
      !   mats_nest_A(3) = g_ctx%Atilde_21
      !   mats_nest_A(4) = g_ctx%Atilde_22
      !   call MatCreateNest(comm, 2, PETSC_NULL_IS, 2, PETSC_NULL_IS, &
      !                      mats_nest_A, g_ctx%K_A_block, ierr)
      ! end block

      ! ! Convert the K_A MatNest to MPIAIJ for the Alfven block solver.
      ! if (.not. first_time) call MatDestroy(g_ctx%K_A_aij, ierr)
      ! call MatConvert(g_ctx%K_A_block, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%K_A_aij, ierr)

      ! ! K_A (psi,u): switchable solver (direct LU/MUMPS or toroidal mode-split PC)
      ! ! via ALFVEN_BLOCK_SOLVER.
      ! call setup_alfven_block_ksp(g_ctx%ksp_block_A, g_ctx%K_A_aij, comm, first_time, &
      !                             "Alfven (psi,u)-block KSP")
                                  
      ! !if (physics_pc_multi_step) call setup_block_ksp(g_ctx%ksp_S_PBP, g_ctx%S_u, comm, first_time, "S_PBP KSP")
      ! !if (physics_pc_multi_step) g_ctx%ksp_S_PBP_created = .true.

      ! if (.not. g_ctx%sub_blocks_setup_done) then
      !   call MatCreateVecs(g_ctx%K_A_aij, g_ctx%rhs_A, g_ctx%sol_A, ierr)
      !   call MatCreateVecs(g_ctx%B_55,    g_ctx%tmp_rho, PETSC_NULL_VEC, ierr)
      !   call MatCreateVecs(g_ctx%B_66,    g_ctx%tmp_T,   PETSC_NULL_VEC, ierr)
      !   g_ctx%sub_blocks_setup_done = .true.
      ! endif

      ! Stage-1 verification: confirm the segregated (Atilde_11, exact S_u) solve of the
      ! 2x2 Alfven block reproduces the direct A_alfven MUMPS solve to machine precision.
      ! Requires ksp_alfven + ksp_S_PBP (holding exact S_u) — both set up by the call above.
      if (debug_physics_pc) then
        call verify_alfven_2x2_segregated(comm, my_id)
      endif

      if (physics_pc_probe_exact) then
        call assemble_probed_exact_4x4(comm, first_time, my_id)
        call petsc_mat_convert_spectrum(g_ctx%A_reduced_4x4, "A_exact_4x4_eq", .false.)
        call petsc_mat_convert_spectrum(A_full, "A_full_eq", .false.)
      endif

    else if (physics_pc_sub_blocks) then
      ! ===== Sub-blocks PC: (psi,u) Alfven super-block + independent rho, T blocks =====

      ! Build K_A (psi,u) 2x2 MatNest from refs to existing Atilde_*/B_12.
      block
        Mat :: mats_nest_A(4)
        if (.not. first_time) call MatDestroy(g_ctx%K_A_block, ierr)
        mats_nest_A(1) = g_ctx%Atilde_11
        mats_nest_A(2) = g_ctx%B_12
        mats_nest_A(3) = g_ctx%Atilde_21
        mats_nest_A(4) = g_ctx%Atilde_22
        call MatCreateNest(comm, 2, PETSC_NULL_IS_ARRAY, 2, PETSC_NULL_IS_ARRAY, &
                           mats_nest_A, g_ctx%K_A_block, ierr)
      end block

      ! Convert the K_A MatNest to MPIAIJ for the Alfven block solver.
      if (.not. first_time) call MatDestroy(g_ctx%K_A_aij, ierr)
      call MatConvert(g_ctx%K_A_block, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%K_A_aij, ierr)

      ! K_A (psi,u): switchable solver (direct LU/MUMPS or toroidal mode-split PC)
      ! via ALFVEN_BLOCK_SOLVER.
      call setup_alfven_block_ksp(g_ctx%ksp_block_A, g_ctx%K_A_aij, comm, first_time, &
                                  "Alfven (psi,u)-block KSP")

      call setup_rho_block_ksp(g_ctx%ksp_rho, g_ctx%B_55, comm, first_time, "rho-block KSP")
      call setup_T_block_ksp  (g_ctx%ksp_T,   g_ctx%B_66, comm, first_time, "T-block KSP")

      ! Allocate the Alfven packed work vectors + 1-variable rho/T residual scratch.
      if (.not. g_ctx%sub_blocks_setup_done) then
        call MatCreateVecs(g_ctx%K_A_aij, g_ctx%rhs_A, g_ctx%sol_A, ierr)
        call MatCreateVecs(g_ctx%B_55,    g_ctx%tmp_rho, PETSC_NULL_VEC, ierr)
        call MatCreateVecs(g_ctx%B_66,    g_ctx%tmp_T,   PETSC_NULL_VEC, ierr)
        g_ctx%sub_blocks_setup_done = .true.
      endif
      g_ctx%ksp_created = .true.
    else
      ! Block-diagonal mode: 4 separate sub-KSPs
      call setup_block_ksp(g_ctx%ksp_psi, g_ctx%Atilde_11, comm, first_time, "psi-block KSP")
      call setup_block_ksp(g_ctx%ksp_u,   g_ctx%Atilde_22, comm, first_time, "u-block KSP")
      call setup_block_ksp(g_ctx%ksp_rho, g_ctx%B_55, comm, first_time, "rho-block KSP")
      call setup_block_ksp(g_ctx%ksp_T,   g_ctx%B_66, comm, first_time, "T-block KSP")
      g_ctx%ksp_created = .true.
    endif

    ! --- Step 6: Allocate work vectors (first time only) ---
    if (first_time) then
      call MatCreateVecs(g_ctx%B_11, g_ctx%work_1, PETSC_NULL_VEC, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_2, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_3, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_4, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_5, ierr)
    endif

    g_ctx%reduced_ready = .true.
    g_ctx%comm = comm

    if (my_id == 0) write(*,'(A)') "[Physics PC] Reduced system ready."

    ! --- Milestone-1 diagnostic: compare P_full against the exact condensation.
    !     Expensive (O(N) MUMPS solves per corrected block); the run continues
    !     afterwards, matching the physics_pc_verify_spbp convention.
    if (physics_pc_verify_reduced) call verify_reduced_pde_operator(A_full, comm, my_id)
    if (physics_pc_verify_schur)   call verify_schur_factorization_4x4(comm, my_id)
    if (physics_pc_schur_approx)   call verify_schur_approx_4x4(comm, my_id)
  end subroutine petsc_physics_pc_build_reduced



#endif
end module mod_petsc_pc_physics
