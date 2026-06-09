module mod_petsc_pc_physics
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  use mod_petsc_pc_physics_construction, only: &
       create_variable_index_sets, extract_sub_block, &
       compute_diag_mass_inverse, compute_per_node_block_inverse, &
       compute_schur_corrected_block, compute_schur_corrected_block_psi, &
       compute_schur_corrected_block_u, compute_schur_corrected_block_21, &
       compute_schur_corrected_block_61, compute_schur_corrected_block_exact, &
       compute_explicit_preconditioned_matrix, &
       setup_block_ksp, setup_constraint_mass_ksp, &
       setup_block_ksp_amg_krylov, setup_block_ksp_hypre_amg_krylov, &
       setup_alfven_block_ksp, setup_rho_block_ksp, setup_T_block_ksp, &
       assemble_monolithic_4x4, assemble_probed_exact_4x4, &
       verify_alfven_2x2_segregated
  use mod_petsc_pc_physics_element, only: &
       petsc_create_pc_matrices, petsc_assemble_pc_matrices, &
       petsc_assemble_pc_diagonal_matrices, petsc_update_physics_pc_ctx, &
       petsc_test_pc_matrix
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
    use phys_module, only: physics_pc_reassemble, debug_physics_pc, physics_pc_monolithic, &
                           physics_pc_multi_step, physics_pc_probe_exact, physics_pc_block_inv, &
                           physics_pc_sub_blocks
    use mod_petsc_matrix_analysis, only: petsc_mat_convert_spectrum, petsc_mat_equilibrate

    Mat, intent(in) :: A_full

    PetscErrorCode :: ierr
    PetscInt :: bs_ntor
    integer :: comm, my_id, mpierr
    logical :: first_time, use_reassembled
    PetscReal :: norm_val
    Mat :: diag_11, diag_22, diag_55, diag_66  ! pointers to chosen diagonal blocks
    Mat :: prod_tmp                              ! temporary for block-inverse diagnostic
    
    Mat :: B_tmp ! Store the preconditioned matrix B = M^{-1} A 
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
    use_reassembled = physics_pc_reassemble

    if (my_id == 0) then
      if (use_reassembled) then
        write(*,'(A)') "[Physics PC] Building reduced 4x4 system (reassembled diagonal blocks)..."
      else
        write(*,'(A)') "[Physics PC] Building reduced 4x4 system (extracted blocks)..."
      endif
    endif

    ! --- Step 1: Create index sets (first time only) ---
    if (.not. g_ctx%is_created) then
      call create_variable_index_sets(A_full, comm)
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Index sets created"
    endif

    ! --- Step 1b: Assemble simplified diagonal blocks if requested ---
    if (use_reassembled) then
      call petsc_assemble_pc_diagonal_matrices(A_full, comm, my_id)
    endif

    ! --- Step 2: Extract sub-blocks from full system ---
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
    if (.not. use_reassembled) then
      call extract_sub_block(A_full, var_psi, var_psi, g_ctx%B_11, first_time)
      call extract_sub_block(A_full, var_u,   var_u,   g_ctx%B_22, first_time)
      call extract_sub_block(A_full, var_rho, var_rho, g_ctx%B_55, first_time)
      call extract_sub_block(A_full, var_T,   var_T,   g_ctx%B_66, first_time)
    else if (debug_physics_pc) then
      ! In debug mode, extract anyway to compare norms
      call extract_sub_block(A_full, var_psi, var_psi, g_ctx%B_11, first_time)
      call extract_sub_block(A_full, var_u,   var_u,   g_ctx%B_22, first_time)
      call extract_sub_block(A_full, var_rho, var_rho, g_ctx%B_55, first_time)
      call extract_sub_block(A_full, var_T,   var_T,   g_ctx%B_66, first_time)
    endif

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

    if (use_reassembled) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Sub-blocks extracted (17 off-diag/constraint)"
    else
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Sub-blocks extracted (21 blocks)"
    endif

    ! --- Step 3: Compute mass matrix inverse ---
    if (physics_pc_block_inv) then
      ! Per-node 4x4 block-diagonal inverse: captures Bezier DOF coupling within each harmonic
      call compute_per_node_block_inverse(g_ctx%B_33, g_ctx%Dinv_Mj, g_ctx%dinv_created)
      call compute_per_node_block_inverse(g_ctx%B_44, g_ctx%Dinv_Mw, g_ctx%dinv_created_w)
      if (debug_physics_pc) then
        call MatNorm(g_ctx%Dinv_Mj, NORM_FROBENIUS, norm_val, ierr)
        if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Dinv_Mj||_F (per-node block) = ", norm_val
        call MatNorm(g_ctx%Dinv_Mw, NORM_FROBENIUS, norm_val, ierr)
        if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Dinv_Mw||_F (per-node block) = ", norm_val
        ! Verify: ||Dinv_Mj * B_33 - I||_F and ||Dinv_Mw * B_44 - I||_F
        call MatMatMult(g_ctx%Dinv_Mj, g_ctx%B_33, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, prod_tmp, ierr)
        call MatShift(prod_tmp, -1.0d0, ierr)
        call MatNorm(prod_tmp, NORM_FROBENIUS, norm_val, ierr)
        if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Dinv_Mj * B_33 - I||_F = ", norm_val
        call MatDestroy(prod_tmp, ierr)
        call MatMatMult(g_ctx%Dinv_Mw, g_ctx%B_44, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, prod_tmp, ierr)
        call MatShift(prod_tmp, -1.0d0, ierr)
        call MatNorm(prod_tmp, NORM_FROBENIUS, norm_val, ierr)
        if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||Dinv_Mw * B_44 - I||_F = ", norm_val
        call MatDestroy(prod_tmp, ierr)
      end if
    else
      ! Scalar diagonal inverse: 1 / diag(B)
      call compute_diag_mass_inverse(g_ctx%B_33, g_ctx%diag_Mj_inv, first_time)
      call compute_diag_mass_inverse(g_ctx%B_44, g_ctx%diag_Mw_inv, first_time)
      if (debug_physics_pc) then
        call VecNorm(g_ctx%diag_Mj_inv, NORM_2, norm_val, ierr)
        if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||D_j^{-1} (diag)||_2 = ", norm_val
        call VecNorm(g_ctx%diag_Mw_inv, NORM_2, norm_val, ierr)
        if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||D_w^{-1} (diag)||_2 = ", norm_val
      end if
    endif

    ! --- Debug: compare reassembled vs extracted norms ---
    if (use_reassembled .and. debug_physics_pc) then
      call MatNorm(g_ctx%B_11, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_11 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_11, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_11 (reassembled)||_F = ", norm_val
      call MatNorm(g_ctx%B_22, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_22 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_22, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_22 (reassembled)||_F = ", norm_val
      call MatNorm(g_ctx%B_55, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_55 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_55, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_55 (reassembled)||_F = ", norm_val
      call MatNorm(g_ctx%B_66, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||B_66 (extracted)||_F = ", norm_val
      call MatNorm(g_ctx%R_66, NORM_FROBENIUS, norm_val, ierr)
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||R_66 (reassembled)||_F = ", norm_val
    endif

    ! --- Step 4a: Set up KSPs for elliptic constraint mass matrices ---
    ! (Must be done before Schur correction so MUMPS factorization is available)
    call setup_constraint_mass_ksp(g_ctx%ksp_Mj, g_ctx%B_33, comm, first_time, "Mj constraint-mass KSP")
    call setup_constraint_mass_ksp(g_ctx%ksp_Mw, g_ctx%B_44, comm, first_time, "Mw constraint-mass KSP")
    g_ctx%ksp_elliptic_created = .true.

    ! --- Step 4b: Form Schur-corrected blocks ---
    if (physics_pc_block_inv) then
      ! Use per-node block-diagonal Mat inverse: Atilde = B_diag - B_coupling * Dinv * B_constraint
      if (use_reassembled) then
        call compute_schur_corrected_block(g_ctx%R_11, g_ctx%B_13, g_ctx%B_31, &
                                           g_ctx%Dinv_Mj, g_ctx%Atilde_11, first_time)
        call compute_schur_corrected_block(g_ctx%R_22, g_ctx%B_24, g_ctx%B_42, &
                                           g_ctx%Dinv_Mw, g_ctx%Atilde_22, first_time)
      else
        call compute_schur_corrected_block(g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, &
                                           g_ctx%Dinv_Mj, g_ctx%Atilde_11, first_time)
        call compute_schur_corrected_block(g_ctx%B_22, g_ctx%B_24, g_ctx%B_42, &
                                           g_ctx%Dinv_Mw, g_ctx%Atilde_22, first_time)
      endif
      ! Off-diagonal Schur corrections via Mat inverse (same Dinv_Mj)
      call compute_schur_corrected_block(g_ctx%B_21, g_ctx%B_23, g_ctx%B_31, &
                                         g_ctx%Dinv_Mj, g_ctx%Atilde_21, first_time)
      call compute_schur_corrected_block(g_ctx%B_61, g_ctx%B_63, g_ctx%B_31, &
                                         g_ctx%Dinv_Mj, g_ctx%Atilde_61, first_time)
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Computed Schur-corrected blocks (per-node block M^{-1})"
    else
      ! Use element-assembled / scalar diagonal paths
      if (use_reassembled) then
        call compute_schur_corrected_block_psi(g_ctx%R_11, g_ctx%B_13, g_ctx%B_31, &
                                            g_ctx%diag_Mj_inv, g_ctx%Atilde_11, first_time)
        call compute_schur_corrected_block_u(g_ctx%R_22, g_ctx%B_24, g_ctx%B_42, &
                                            g_ctx%diag_Mw_inv, g_ctx%Atilde_22, first_time)
      else
        call compute_schur_corrected_block_psi(g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, &
                                            g_ctx%diag_Mj_inv, g_ctx%Atilde_11, first_time)
        !call compute_schur_corrected_block_exact(g_ctx%B_33, g_ctx%B_11, g_ctx%B_13, g_ctx%B_31, g_ctx%Atilde_11, first_time)
        call compute_schur_corrected_block_u(g_ctx%B_22, g_ctx%B_24, g_ctx%B_42, &
                                            g_ctx%diag_Mw_inv, g_ctx%Atilde_22, first_time)
        !call compute_schur_corrected_block_exact( g_ctx%B_44, g_ctx%B_22, g_ctx%B_24, g_ctx%B_42, g_ctx%Atilde_22, first_time)
      endif
      ! Off-diagonal Schur corrections from element-assembled K_21 / K_61
      call compute_schur_corrected_block_21(g_ctx%B_21, g_ctx%Atilde_21, first_time)
      !call compute_schur_corrected_block_exact(g_ctx%B_33, g_ctx%B_21, g_ctx%B_23, g_ctx%B_31, g_ctx%Atilde_21, first_time)
      call compute_schur_corrected_block_61(g_ctx%B_61, g_ctx%Atilde_61, first_time)
      !call compute_schur_corrected_block_exact(g_ctx%B_33, g_ctx%B_61, g_ctx%B_63, g_ctx%B_31, g_ctx%Atilde_61, first_time)
      if (my_id == 0) write(*,'(A)') "[Physics PC]   Computed Schur-corrected blocks"
    endif

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

    ! Build the EXACT Schur complement S_u = Atilde_22 - Atilde_21 * Atilde_11^{-1} * B_12
    ! (column-probing, n_u MUMPS solves on Atilde_11). Stage-1 verification: this exact
    ! S_u is copied into S_PBP below so the segregated apply uses the exact operator.
    call compute_schur_corrected_block_exact(g_ctx%Atilde_11, g_ctx%Atilde_22, &
                                             g_ctx%Atilde_21, g_ctx%B_12, g_ctx%S_u, first_time)
    !call compute_explicit_preconditioned_matrix(g_ctx%S_PBP, g_ctx%S_u, B_tmp, first_time)

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

    ! Stage-1: overwrite the element-assembled S_PBP with the exact S_u so the
    ! existing predictor-corrector apply (via ksp_S_PBP) uses the exact Schur.
    ! NOTE: S_PBP is block-size-1 BAIJ (petsc_create_pc_matrix) while S_u is MPIAIJ
    !   from probing; their row layouts differ in origin. If this MatCopy errors at
    !   runtime (type/layout mismatch), the fallback is to delete this line and instead
    !   rebind the KSP operator in assemble_monolithic_4x4 where ksp_S_PBP is created:
    !   change `KSPSetOperators(ksp_S_PBP, S_PBP, S_PBP)` to
    !   `KSPSetOperators(ksp_S_PBP, g_ctx%S_u, g_ctx%S_u)`.
    call MatCopy(g_ctx%S_u, g_ctx%S_PBP, DIFFERENT_NONZERO_PATTERN, ierr)

    ! --- Step 5: Set up solver(s) ---
    if (physics_pc_monolithic .or. physics_pc_multi_step) then
      ! When probe_exact=.true., monolithic only builds A_approx for the diagnostic;
      ! the KSP is owned entirely by assemble_probed_exact_4x4 (avoids stale-factor issues).
      call assemble_monolithic_4x4(use_reassembled, comm, first_time, my_id, physics_pc_probe_exact)

      ! Sub-block KSPs needed by the three-step (multi_step) apply:
      !   ksp_psi (Atilde_11)  -> magnetic predictor
      !   ksp_rho (B_55)       -> transport correction (rho)
      !   ksp_T   (B_66)       -> transport correction (T)
      ! Set up unconditionally: cheap, reused, keeps control flow simple.
      call setup_block_ksp(g_ctx%ksp_psi, g_ctx%Atilde_11, comm, first_time, "psi predictor KSP (Atilde_11)")
      call setup_block_ksp(g_ctx%ksp_rho, g_ctx%B_55,      comm, first_time, "rho-block KSP")
      call setup_block_ksp(g_ctx%ksp_T,   g_ctx%B_66,      comm, first_time, "T-block KSP")
      g_ctx%ksp_created = .true.

      ! Stage-1 verification: confirm the segregated (Atilde_11, exact S_u) solve of the
      ! 2x2 Alfven block reproduces the direct A_alfven MUMPS solve to machine precision.
      ! Requires ksp_alfven + ksp_S_PBP (holding exact S_u) — both set up by the call above.
      if (debug_physics_pc) then
        call verify_alfven_2x2_segregated(comm, my_id)
      endif

      if (debug_physics_pc) then
        !call petsc_mat_convert_spectrum(g_ctx%A_reduced_4x4, "A_reduced_4x4", .false.)
        !call petsc_test_pc_matrix(g_ctx%A_reduced_4x4, "A_reduced_4x4", .false., my_id)
        !call petsc_test_pc_matrix(g_ctx%M_hydro, "M_hydro", .false., my_id)
        !call petsc_test_pc_matrix(g_ctx%A_alfven, "A_alfven_2x2", .false., my_id)
      endif

      if (physics_pc_probe_exact) then
        call assemble_probed_exact_4x4(use_reassembled, comm, first_time, my_id)
       ! call petsc_mat_convert_spectrum(g_ctx%A_reduced_4x4, "A_exact_4x4", .false.)
      endif
    else if (physics_pc_sub_blocks) then
      ! ===== Sub-blocks PC: (psi,u) Alfven super-block + independent rho, T blocks =====

      ! Build K_A (psi,u) 2x2 MatNest from refs to existing Atilde_*/B_12.
      ! The nest must be recreated each rebuild (it references Atilde_* handles that
      ! compute_schur_corrected_block_* destroys+recreates), so destroy the previous
      ! nest first to avoid leaking one MatNest per rebuild.
      block
        Mat :: mats_nest_A(4)
        if (.not. first_time) call MatDestroy(g_ctx%K_A_block, ierr)
        mats_nest_A(1) = g_ctx%Atilde_11
        mats_nest_A(2) = g_ctx%B_12
        mats_nest_A(3) = g_ctx%Atilde_21
        mats_nest_A(4) = g_ctx%Atilde_22
        call MatCreateNest(comm, 2, PETSC_NULL_IS, 2, PETSC_NULL_IS, &
                           mats_nest_A, g_ctx%K_A_block, ierr)
      end block

      ! Convert the K_A MatNest to MPIAIJ for the Alfven block solver.
      ! Destroy the previous AIJ first (MAT_INITIAL_MATRIX allocates a new object
      ! each rebuild) to avoid leaking one matrix per rebuild.
      if (.not. first_time) call MatDestroy(g_ctx%K_A_aij, ierr)
      call MatConvert(g_ctx%K_A_block, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%K_A_aij, ierr)

      ! K_A (psi,u): switchable solver (direct LU/MUMPS or toroidal mode-split PC)
      ! via ALFVEN_BLOCK_SOLVER.
      call setup_alfven_block_ksp(g_ctx%ksp_block_A, g_ctx%K_A_aij, comm, first_time, &
                                  "Alfven (psi,u)-block KSP")

      ! rho (B_55) and T (B_66): independent transport blocks, each with its own
      ! (physics-motivated) AMG solver, applied directly to the extracted diagonal
      ! sub-blocks. No bundling, no packed (rho,T) work vectors. On solve_only steps
      ! build_reduced is NOT called (mod_petsc.f90), so these KSPs and their AMG
      ! hierarchies persist and are reused as-is while the outer FGMRES uses the fresh
      ! A for mat-vecs (lagged-PC reuse across timesteps). On rebuild events B_55/B_66
      ! have genuinely changed (theta*tstep, state), so refreshing here is correct.
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
      if (use_reassembled) then
        call setup_block_ksp(g_ctx%ksp_rho, g_ctx%R_55, comm, first_time, "rho-block KSP")
        call setup_block_ksp(g_ctx%ksp_T,   g_ctx%R_66, comm, first_time, "T-block KSP")
      else
        call setup_block_ksp(g_ctx%ksp_rho, g_ctx%B_55, comm, first_time, "rho-block KSP")
        call setup_block_ksp(g_ctx%ksp_T,   g_ctx%B_66, comm, first_time, "T-block KSP")
      endif
      g_ctx%ksp_created = .true.
    endif

    ! --- Step 6: Allocate work vectors (first time only) ---
    if (first_time) then
      if (use_reassembled) then
        call MatCreateVecs(g_ctx%R_11, g_ctx%work_1, PETSC_NULL_VEC, ierr)
      else
        call MatCreateVecs(g_ctx%B_11, g_ctx%work_1, PETSC_NULL_VEC, ierr)
      endif
      call VecDuplicate(g_ctx%work_1, g_ctx%work_2, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_3, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_4, ierr)
      call VecDuplicate(g_ctx%work_1, g_ctx%work_5, ierr)
    endif

    g_ctx%reduced_ready = .true.
    g_ctx%comm = comm

    if (my_id == 0) write(*,'(A)') "[Physics PC] Reduced system ready."
  end subroutine petsc_physics_pc_build_reduced



#endif
end module mod_petsc_pc_physics
