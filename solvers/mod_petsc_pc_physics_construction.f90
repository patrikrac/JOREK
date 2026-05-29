module mod_petsc_pc_physics_construction
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: type_physics_pc_ctx, g_ctx
  implicit none
  private

  public :: create_variable_index_sets
  public :: extract_sub_block
  public :: compute_diag_mass_inverse
  public :: compute_per_node_block_inverse
  public :: compute_schur_corrected_block
  public :: compute_schur_corrected_block_psi
  public :: compute_schur_corrected_block_u
  public :: compute_schur_corrected_block_21
  public :: compute_schur_corrected_block_61
  public :: compute_schur_corrected_block_exact
  public :: compute_explicit_preconditioned_matrix
  public :: setup_block_ksp
  public :: setup_block_ksp_amg_krylov, setup_block_ksp_hypre_amg_krylov
  public :: assemble_monolithic_4x4
  public :: assemble_probed_exact_4x4

contains

  !--------------------------------------------------------------------
  !> Create variable index sets for extracting sub-vectors from the
  !! full 6-variable system vector.
  !!
  !! DOF ordering within each BAIJ block (block_size = n_var*n_tor):
  !!   var v (0-based) at node block i: i*block_size + v*n_tor + m
  !!   for m = 0..n_tor-1
  !--------------------------------------------------------------------
  subroutine create_variable_index_sets(A_full, comm)
    use mod_parameters, only: n_var, n_tor

    Mat, intent(in) :: A_full
    integer, intent(in) :: comm

    PetscInt :: n_local, n_global, rstart, rend
    PetscInt :: block_size, n_block_local, n_var_dofs, n_block_global
    !PetscInt :: out_local, out_global, out_start, out_end
    PetscInt, allocatable :: indices(:)
    PetscErrorCode :: ierr
    integer :: v, i, m, k

    ! Get parallel layout from the full system matrix
    PetscCallA(MatGetLocalSize(A_full, n_local, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetSize(A_full, n_global, PETSC_NULL_INTEGER, ierr))
    PetscCallA(MatGetOwnershipRange(A_full, rstart, rend, ierr))

    !write(*,'(A,I8,A,I8)') "[Physics PC]   Creating variable index sets: local DOFs ", n_local, " [", rstart, "-", rend-1, "]"

    block_size    = n_var * n_tor
    n_block_local = n_local / block_size
    n_block_global = n_global / block_size
    n_var_dofs    = n_block_local * n_tor

    allocate(indices(n_var_dofs))

    do v = 1, 6
      k = 0
      do i = 0, n_block_local - 1
        do m = 0, n_tor - 1
          k = k + 1
          !k = i*(n_tor-1) + m + 1
          indices(k) = rstart + i * block_size + (v-1) * n_tor + m
        enddo
      enddo
      PetscCallA(ISCreateGeneral(comm, n_var_dofs, indices, PETSC_COPY_VALUES, &
                           g_ctx%is_var(v), ierr))

      ! Print information about the created IS for debugging
      !PetscCallA(ISGetSize(g_ctx%is_var(v), out_global, ierr))
      !PetscCallA(ISGetLocalSize(g_ctx%is_var(v), out_local, ierr))
      !PetscCallA(ISGetMinMax(g_ctx%is_var(v), out_start, out_end, ierr))
      !write(*,*) "[Physics PC]     Variable ", v, ": local DOFs : ", out_local," global DOFs ", out_global, " [", out_start, "-", out_end, "]"
    enddo

    deallocate(indices)
    g_ctx%is_created = .true.
  end subroutine create_variable_index_sets


  !--------------------------------------------------------------------
  !> Extract a sub-block A_ij from the full system matrix.
  !! A_ij has rows corresponding to equation eq_row and columns
  !! corresponding to variable var_col.
  !--------------------------------------------------------------------
  subroutine extract_sub_block(A_full, eq_row, var_col, B, first_time)
    Mat, intent(in)    :: A_full
    integer, intent(in) :: eq_row, var_col
    Mat, intent(inout)  :: B
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (first_time) then
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), &
                              MAT_INITIAL_MATRIX, B, ierr))
    else
      PetscCallA(MatCreateSubMatrix(A_full, g_ctx%is_var(eq_row), g_ctx%is_var(var_col), &
                              MAT_REUSE_MATRIX, B, ierr))
    endif
  end subroutine extract_sub_block


  !--------------------------------------------------------------------
  !> Compute diagnonal mass inverse from diagonal blocks B_33 and B_44.
  !! diag_M_inv = 1 / diag(B)
  !--------------------------------------------------------------------
  subroutine compute_diag_mass_inverse(B, diag_M_inv, first_time)
    Mat, intent(in)    :: B
    Vec, intent(inout) :: diag_M_inv
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (first_time) then
      PetscCallA(MatCreateVecs(B, PETSC_NULL_VEC, diag_M_inv, ierr))
    endif
    !PetscCallA(MatGetRowSum(B, diag_M_inv, ierr))
    PetscCallA(MatGetDiagonal(B, diag_M_inv, ierr))
    PetscCallA(VecReciprocal(diag_M_inv, ierr))
  end subroutine compute_diag_mass_inverse

  !--------------------------------------------------------------------
  !> Compute per-node full block-diagonal inverse of a 1-var AIJ mass matrix.
  !!
  !! For each interior node, the full (n_degrees*n_tor)^2 block is extracted
  !! and inverted with LAPACK dgesv in one call. Axis nodes are grouped: all
  !! unique DOFs across axis nodes are collected and their combined
  !! (n_axis_dofs*n_tor)^2 block is inverted as a single unit. This handles
  !! both treat_axis (all axis nodes share the same 4 DOFs) and
  !! force_central_node (axis nodes share only DOF 1, distinct DOFs 2-4).
  !!
  !! Row indices are built in DOF-major order: all n_tor harmonics for DOF 1,
  !! then all n_tor harmonics for DOF 2, etc.
  !!
  !! MatGetValues fills row-major into Fortran column-major memory, yielding
  !! the transpose of the actual block. dgesv solves the transposed system,
  !! and MatSetValues re-transposes — giving the correct block inverse.
  !--------------------------------------------------------------------
  subroutine compute_per_node_block_inverse(B_mass, Dinv, dinv_created)
    use mod_parameters, only: n_tor, n_degrees, n_vertex_max
    use nodes_elements

    Mat, intent(in)      :: B_mass
    Mat, intent(inout)   :: Dinv
    logical, intent(inout) :: dinv_created

    PetscErrorCode :: ierr
    PetscInt :: rstart, rend, nrows_local, ncols_local, nrows_global, ncols_global
    integer :: my_ind_min, my_ind_max, n_block_local
    integer :: ielm, iv, inode, j, d, cnt, n_elements
    integer :: k0, local_blk, block_n, n_axis_dofs
    integer :: info, comm
    logical, allocatable :: visited(:), in_axis_set(:)
    integer, allocatable :: axis_dof_list(:), ipiv_blk(:)
    PetscInt, allocatable :: rows_node(:), axis_rows(:)
    PetscScalar, allocatable :: blk(:,:), rhs_blk(:,:), diag_save_blk(:)

    external :: dgesv

    call MatGetOwnershipRange(B_mass, rstart, rend, ierr)
    n_block_local = int((rend - rstart) / n_tor)
    my_ind_min    = int(rstart / n_tor) + 1
    my_ind_max    = my_ind_min + n_block_local - 1

    if (.not. dinv_created) then
      call MatGetLocalSize(B_mass, nrows_local, ncols_local, ierr)
      call MatGetSize(B_mass, nrows_global, ncols_global, ierr)
      call PetscObjectGetComm(B_mass, comm, ierr)
      call MatCreate(comm, Dinv, ierr)
      call MatSetSizes(Dinv, nrows_local, ncols_local, nrows_global, ncols_global, ierr)
      call MatSetType(Dinv, MATMPIAIJ, ierr)
      ! Hint: n_degrees*n_tor nonzeros per row (axis rows may have more, allowed by PETSC_FALSE)
      call MatMPIAIJSetPreallocation(Dinv, n_degrees*n_tor, PETSC_NULL_INTEGER_ARRAY, &
                                     0, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatSetOption(Dinv, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
      dinv_created = .true.
    else
      call MatZeroEntries(Dinv, ierr)
    endif

    n_elements = element_list%n_elements

    !--- Pass 0: collect unique axis DOFs owned by this rank ---
    allocate(in_axis_set(n_block_local))
    in_axis_set = .false.
    n_axis_dofs = 0
    do ielm = 1, n_elements
      do iv = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(iv)
        if (.not. node_list%node(inode)%axis_node) cycle
        do j = 1, n_degrees
          d = node_list%node(inode)%index(j)
          if (d < my_ind_min .or. d > my_ind_max) cycle
          if (in_axis_set(d - my_ind_min + 1)) cycle
          in_axis_set(d - my_ind_min + 1) = .true.
          n_axis_dofs = n_axis_dofs + 1
        end do
      end do
    end do
    allocate(axis_dof_list(max(1, n_axis_dofs)))
    cnt = 0
    do d = my_ind_min, my_ind_max
      if (in_axis_set(d - my_ind_min + 1)) then
        cnt = cnt + 1
        axis_dof_list(cnt) = d
      end if
    end do
    deallocate(in_axis_set)

    !--- Pass 1: interior nodes ---
    allocate(visited(0:n_block_local-1))
    visited = .false.
    block_n = n_degrees * n_tor
    allocate(rows_node(block_n), blk(block_n,block_n), rhs_blk(block_n,block_n))
    allocate(diag_save_blk(block_n), ipiv_blk(block_n))

    do ielm = 1, n_elements
      do iv = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(iv)
        if (node_list%node(inode)%axis_node) cycle
        k0 = node_list%node(inode)%index(1)
        if (k0 < my_ind_min .or. k0 + n_degrees - 1 > my_ind_max) cycle
        local_blk = k0 - my_ind_min
        if (visited(local_blk)) cycle
        visited(local_blk) = .true.

        ! Row indices: DOF-major (all n_tor harmonics for DOF j, then DOF j+1, ...)
        do j = 1, n_degrees
          d = node_list%node(inode)%index(j)
          do cnt = 0, n_tor - 1
            rows_node((j-1)*n_tor + cnt + 1) = (d - 1) * n_tor + cnt
          end do
        end do

        call MatGetValues(B_mass, block_n, rows_node, block_n, rows_node, blk, ierr)
        do j = 1, block_n
          diag_save_blk(j) = blk(j,j)
        end do
        rhs_blk = 0.d0
        do j = 1, block_n; rhs_blk(j,j) = 1.d0; end do
        call dgesv(block_n, block_n, blk, block_n, ipiv_blk, rhs_blk, block_n, info)
        if (info /= 0) then
          rhs_blk = 0.d0
          do j = 1, block_n
            if (abs(diag_save_blk(j)) > 0.d0) rhs_blk(j,j) = 1.d0 / diag_save_blk(j)
          end do
        end if
        call MatSetValues(Dinv, block_n, rows_node, block_n, rows_node, rhs_blk, INSERT_VALUES, ierr)
      end do
    end do
    deallocate(visited, rows_node, blk, rhs_blk, diag_save_blk, ipiv_blk)

    !--- Pass 2: axis node group ---
    if (n_axis_dofs > 0) then
      block_n = n_axis_dofs * n_tor
      allocate(axis_rows(block_n), blk(block_n,block_n), rhs_blk(block_n,block_n))
      allocate(diag_save_blk(block_n), ipiv_blk(block_n))

      do d = 1, n_axis_dofs
        do cnt = 0, n_tor - 1
          axis_rows((d-1)*n_tor + cnt + 1) = (axis_dof_list(d) - 1) * n_tor + cnt
        end do
      end do

      call MatGetValues(B_mass, block_n, axis_rows, block_n, axis_rows, blk, ierr)
      do j = 1, block_n
        diag_save_blk(j) = blk(j,j)
      end do
      rhs_blk = 0.d0
      do j = 1, block_n; rhs_blk(j,j) = 1.d0; end do
      call dgesv(block_n, block_n, blk, block_n, ipiv_blk, rhs_blk, block_n, info)
      if (info /= 0) then
        rhs_blk = 0.d0
        do j = 1, block_n
          if (abs(diag_save_blk(j)) > 0.d0) rhs_blk(j,j) = 1.d0 / diag_save_blk(j)
        end do
      end if
      call MatSetValues(Dinv, block_n, axis_rows, block_n, axis_rows, rhs_blk, INSERT_VALUES, ierr)
      deallocate(axis_rows, blk, rhs_blk, diag_save_blk, ipiv_blk)
    end if

    deallocate(axis_dof_list)

    call MatAssemblyBegin(Dinv, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(Dinv, MAT_FINAL_ASSEMBLY, ierr)
  end subroutine compute_per_node_block_inverse


  !--------------------------------------------------------------------
  !> Compute a Schur-corrected diagonal block using diagonal mass inverse:
  !! Atilde = B_diag - B_coupling * diag(M_inv) * B_constraint
  !!
  !! Steps:
  !!   1. B_scaled = diag(M_inv) * B_constraint  (scale rows)
  !!   2. C = B_coupling * B_scaled               (mat-mat product)
  !!   3. Atilde = B_diag - C                     (subtract)
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block_diag(B_diag, B_coupling, B_constraint, &
                                            diag_M_inv, Atilde, first_time)
    Mat, intent(in)    :: B_diag, B_coupling, B_constraint
    Vec, intent(in)    :: diag_M_inv
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    Mat :: B_scaled, C
    PetscErrorCode :: ierr

    ! B_scaled = diag(M_inv) * B_constraint  (left-scale rows)
    call MatDuplicate(B_constraint, MAT_COPY_VALUES, B_scaled, ierr)
    call MatDiagonalScale(B_scaled, diag_M_inv, PETSC_NULL_VEC, ierr)

    ! C = B_coupling * B_scaled
    call MatMatMult(B_coupling, B_scaled, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, C, ierr)

    ! Atilde = B_diag - C
    ! Always destroy and recreate: sparsity pattern may change between rebuilds
    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)
    call MatAXPY(Atilde, -1.0d0, C, DIFFERENT_NONZERO_PATTERN, ierr)

    call MatDestroy(B_scaled, ierr)
    call MatDestroy(C, ierr)
  end subroutine compute_schur_corrected_block_diag


  subroutine compute_schur_corrected_block_psi(B_diag, B_coupling, B_constraint, &
                                            diag_M_inv, Atilde, first_time)

    Mat, intent(in)    :: B_diag, B_coupling, B_constraint
    Vec, intent(in)    :: diag_M_inv
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%psi_correction_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_psi_correction, DIFFERENT_NONZERO_PATTERN, ierr) !TODO: In principle they have the same non-zero pattern but in practice PETSc might drop some entires which would lead to a different pattern. No idea if that impacts anything?
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block required!"
    endif
  end subroutine compute_schur_corrected_block_psi


  subroutine compute_schur_corrected_block_u(B_diag, B_coupling, B_constraint, &
                                            diag_M_inv, Atilde, first_time)

    Mat, intent(in)    :: B_diag, B_coupling, B_constraint
    Vec, intent(in)    :: diag_M_inv
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%u_correction_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_u_correction, DIFFERENT_NONZERO_PATTERN, ierr) !TODO: In principle they have the same non-zero pattern but in practice PETSc might drop some entires which would lead to a different pattern. No idea if that impacts anything?
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block required!"
    endif
  end subroutine compute_schur_corrected_block_u


  !> Off-diagonal Schur correction Atilde_21 = B_21 - K_21_correction
  subroutine compute_schur_corrected_block_21(B_diag, Atilde, first_time)
    Mat, intent(in)    :: B_diag
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%correction_21_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_21_correction, DIFFERENT_NONZERO_PATTERN, ierr)
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block 21 required!"
    endif
  end subroutine compute_schur_corrected_block_21


  !> Off-diagonal Schur correction Atilde_61 = B_61 - K_61_correction
  subroutine compute_schur_corrected_block_61(B_diag, Atilde, first_time)
    Mat, intent(in)    :: B_diag
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    PetscErrorCode :: ierr

    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)

    if (g_ctx%correction_61_ready) then
      call MatAXPY(Atilde, -1.0d0, g_ctx%K_61_correction, DIFFERENT_NONZERO_PATTERN, ierr)
    else
      write(*,*) "[Physics PC]     ERROR: Schur correction block 61 required!"
    endif
  end subroutine compute_schur_corrected_block_61

  !--------------------------------------------------------------------
  !> Compute a Schur-corrected diagonal block using block diagonal inverse:
  !! Atilde = B_diag - B_coupling * Dinv * B_constraint
  !!
  !! Steps:
  !!   1. B_scaled = Dinv * B_constraint            (mat-mat product)
  !!   2. C = B_coupling * B_scaled                 (mat-mat product)
  !!   3. Atilde = B_diag - C                       (subtract)
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block(B_diag, B_coupling, B_constraint, &
                                            Dinv, Atilde, first_time)
    Mat, intent(in)    :: B_diag, B_coupling, B_constraint
    Mat, intent(in)    :: Dinv
    Mat, intent(inout) :: Atilde
    logical, intent(in) :: first_time

    Mat :: B_scaled, C
    PetscErrorCode :: ierr

    ! B_scaled = Dinv * B_constraint  (block-diagonal mat-mat product)
    call MatMatMult(Dinv, B_constraint, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, B_scaled, ierr)

    ! C = B_coupling * B_scaled
    call MatMatMult(B_coupling, B_scaled, MAT_INITIAL_MATRIX, PETSC_DETERMINE_REAL, C, ierr)

    ! Atilde = B_diag - C
    ! Always destroy and recreate: sparsity pattern may change between rebuilds
    if (.not. first_time) call MatDestroy(Atilde, ierr)
    call MatDuplicate(B_diag, MAT_COPY_VALUES, Atilde, ierr)
    call MatAXPY(Atilde, -1.0d0, C, DIFFERENT_NONZERO_PATTERN, ierr)

    call MatDestroy(B_scaled, ierr)
    call MatDestroy(C, ierr)
  end subroutine compute_schur_corrected_block


!--------------------------------------------------------------------
  !> Compute a Schur-corrected block using vector probing:
  !! Atilde = B_diag - B_coupling * M^{-1} * B_constraint
  !!
  !! Uses a KSP (MUMPS) to solve M * y = z column-by-column (probing).
  !! Rank 0 gathers the columns into a dense array, and a fresh sparse 
  !! MATMPIAIJ is assembled at the end.
  !--------------------------------------------------------------------
  subroutine compute_schur_corrected_block_exact(M, B_diag, B_coupling, &
                                          B_constraint, Atilde, first_time)
    implicit none
    Mat, intent(in)     :: M, B_diag, B_coupling, B_constraint
    Mat, intent(inout)  :: Atilde
    logical, intent(in) :: first_time

    KSP            :: ksp_M
    PC             :: pc_M
    Vec            :: e_j, z, y, r_corr, r_diag, r_seq
    VecScatter     :: scat
    PetscErrorCode :: ierr
    PetscInt       :: N_global, m_local, n_local
    integer        :: n, j, my_id, mpierr
    MPI_Comm       :: comm

    PetscScalar, pointer  :: arr(:)
    real*8, allocatable   :: S_dense(:,:)
    PetscInt, allocatable :: row_idxs(:)
    PetscInt       :: col_idx(1)

    ! Infer communicator and rank from the input matrix
    call PetscObjectGetComm(B_diag, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    ! Get the global size of the block (assume square mapping for the variable)
    call MatGetSize(B_diag, PETSC_NULL_INTEGER, N_global, ierr)
    n = int(N_global)

    if (my_id == 0) then
      allocate(S_dense(n, n))
      S_dense = 0.0d0
      write(*,'(A,I0,A)') "[Diagnostics] Probing exact Schur block (", n, " columns)..."
      flush(6)
    endif

    ! -----------------------------------------------------------------
    ! 1. Set Up the Exact LU KSP for M
    ! -----------------------------------------------------------------
    call KSPCreate(comm, ksp_M, ierr)
    call KSPSetOperators(ksp_M, M, M, ierr)
    call KSPSetType(ksp_M, KSPPREONLY, ierr)
    call KSPGetPC(ksp_M, pc_M, ierr)
    call PCSetType(pc_M, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_M, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_M, ierr)

    ! -----------------------------------------------------------------
    ! 2. Allocate Vector Workspace
    ! -----------------------------------------------------------------
    ! e_j : Domain of B_constraint (and B_diag)
    ! z   : Range of B_constraint
    call MatCreateVecs(B_constraint, e_j, z, ierr)
    
    ! y   : Domain of M (Solution of M y = z)
    call MatCreateVecs(M, y, PETSC_NULL_VEC, ierr)
    
    ! r_diag, r_corr : Range of B_diag / B_coupling
    call MatCreateVecs(B_diag, PETSC_NULL_VEC, r_diag, ierr)
    call VecDuplicate(r_diag, r_corr, ierr)

    ! Sequential vector for gathering results on Rank 0
    call VecScatterCreateToZero(r_diag, scat, r_seq, ierr)

    ! -----------------------------------------------------------------
    ! 3. Probe the Matrix Column by Column
    ! -----------------------------------------------------------------
    do j = 0, n-1
      ! Set up standard basis vector e_j
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr)
      call VecAssemblyEnd(e_j, ierr)

      ! z = B_constraint * e_j
      call MatMult(B_constraint, e_j, z, ierr)

      ! Solve M * y = z
      call KSPSolve(ksp_M, z, y, ierr)

      ! r_corr = B_coupling * y
      call MatMult(B_coupling, y, r_corr, ierr)

      ! r_diag = B_diag * e_j
      call MatMult(B_diag, e_j, r_diag, ierr)

      ! r_diag = r_diag - r_corr (This is the j-th column of Atilde)
      call VecAXPY(r_diag, -1.0d0, r_corr, ierr)

      ! Scatter the j-th column to Rank 0
      call VecScatterBegin(scat, r_diag, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_diag, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)

      ! Insert into dense Fortran array
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        S_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! Status output
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Diagnostics]   Probed: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! -----------------------------------------------------------------
    ! 4. Construct the Parallel Sparse Matrix from the Dense Array
    ! -----------------------------------------------------------------
    ! Get local parallel layout matching the working vectors
    call VecGetLocalSize(r_diag, m_local, ierr)
    call VecGetLocalSize(e_j,    n_local, ierr)

    if (.not. first_time) call MatDestroy(Atilde, ierr)

    call MatCreate(comm, Atilde, ierr)
    call MatSetSizes(Atilde, m_local, n_local, n, n, ierr)
    call MatSetType(Atilde, MATMPIAIJ, ierr)
    call MatSetOption(Atilde, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(Atilde, ierr)

    if (my_id == 0) then
      allocate(row_idxs(n))
      do j = 0, n-1
        row_idxs(j+1) = j
      enddo

      ! Insert dense column by dense column
      do j = 0, n-1
        col_idx(1) = j
        call MatSetValues(Atilde, n, row_idxs, 1, col_idx, &
                          S_dense(:, j+1), INSERT_VALUES, ierr)
      enddo

      deallocate(row_idxs)
      deallocate(S_dense)
    endif

    call MatAssemblyBegin(Atilde, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (Atilde, MAT_FINAL_ASSEMBLY, ierr)

    ! -----------------------------------------------------------------
    ! 5. Cleanup Workspace
    ! -----------------------------------------------------------------
    call VecScatterDestroy(scat, ierr)
    call VecDestroy(r_seq,  ierr)
    call VecDestroy(e_j,    ierr)
    call VecDestroy(z,      ierr)
    call VecDestroy(y,      ierr)
    call VecDestroy(r_diag, ierr)
    call VecDestroy(r_corr, ierr)
    call KSPDestroy(ksp_M,  ierr)

  end subroutine compute_schur_corrected_block_exact


  !--------------------------------------------------------------------
  !> Compute the explicitly preconditioned matrix: B = M^{-1} * A
  !!
  !! Uses a KSP (MUMPS) to solve M * y = z where z = A * e_j.
  !! Rank 0 gathers the columns into a dense array, and a fresh sparse 
  !! MATMPIAIJ is assembled at the end to be passed to an eigensolver.
  !--------------------------------------------------------------------
  subroutine compute_explicit_preconditioned_matrix(M, A, B, first_time)
    implicit none
    Mat, intent(in)     :: M, A
    Mat, intent(inout)  :: B
    logical, intent(in) :: first_time

    KSP            :: ksp_M
    PC             :: pc_M
    Vec            :: e_j, z, y, y_seq
    VecScatter     :: scat
    PetscErrorCode :: ierr
    PetscInt       :: N_global, m_local, n_local
    integer        :: n, j, my_id, mpierr
    MPI_Comm       :: comm

    PetscScalar, pointer  :: arr(:)
    real*8, allocatable   :: B_dense(:,:)
    PetscInt, allocatable :: row_idxs(:)
    PetscInt       :: col_idx(1)

    ! Infer communicator and rank from the input matrix
    call PetscObjectGetComm(A, comm, ierr)
    call MPI_Comm_rank(comm, my_id, mpierr)

    ! Get the global size of the matrix
    call MatGetSize(A, PETSC_NULL_INTEGER, N_global, ierr)
    n = int(N_global)

    if (my_id == 0) then
      allocate(B_dense(n, n))
      B_dense = 0.0d0
      write(*,'(A,I0,A)') "[Diagnostics] Probing M^{-1} A (", n, " columns)..."
      flush(6)
    endif

    ! -----------------------------------------------------------------
    ! 1. Set Up the Exact LU KSP for M
    ! -----------------------------------------------------------------
    call KSPCreate(comm, ksp_M, ierr)
    call KSPSetOperators(ksp_M, M, M, ierr)
    call KSPSetType(ksp_M, KSPPREONLY, ierr)
    call KSPGetPC(ksp_M, pc_M, ierr)
    call PCSetType(pc_M, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_M, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_M, ierr)

    ! -----------------------------------------------------------------
    ! 2. Allocate Vector Workspace
    ! -----------------------------------------------------------------
    ! e_j : Domain of A
    ! z   : Range of A (which is also Domain of M)
    call MatCreateVecs(A, e_j, z, ierr)
    
    ! y   : Solution of M y = z
    call VecDuplicate(z, y, ierr)
    
    ! Sequential vector for gathering results on Rank 0
    call VecScatterCreateToZero(y, scat, y_seq, ierr)

    ! -----------------------------------------------------------------
    ! 3. Probe the Matrix Column by Column
    ! -----------------------------------------------------------------
    do j = 0, n-1
      ! Set up standard basis vector e_j
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr)
      call VecAssemblyEnd(e_j, ierr)

      ! z = A * e_j  (Extract the j-th column of A)
      call MatMult(A, e_j, z, ierr)

      ! Solve M * y = z  (Apply M^{-1} to the j-th column of A)
      call KSPSolve(ksp_M, z, y, ierr)

      ! Scatter the j-th column of the result to Rank 0
      call VecScatterBegin(scat, y, y_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, y, y_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)

      ! Insert into dense Fortran array
      if (my_id == 0) then
        call VecGetArrayF90(y_seq, arr, ierr)
        B_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(y_seq, arr, ierr)
      endif

      ! Status output
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Diagnostics]   Probed: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! -----------------------------------------------------------------
    ! 4. Construct the Parallel Sparse Matrix from the Dense Array
    ! -----------------------------------------------------------------
    ! Get local parallel layout matching the working vectors
    call VecGetLocalSize(y,   m_local, ierr)
    call VecGetLocalSize(e_j, n_local, ierr)

    if (.not. first_time) call MatDestroy(B, ierr)

    call MatCreate(comm, B, ierr)
    call MatSetSizes(B, m_local, n_local, n, n, ierr)
    call MatSetType(B, MATMPIAIJ, ierr)
    ! B will be mathematically dense, so allow new non-zero allocation
    call MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(B, ierr)

    if (my_id == 0) then
      allocate(row_idxs(n))
      do j = 0, n-1
        row_idxs(j+1) = j
      enddo

      ! Insert dense column by dense column
      do j = 0, n-1
        col_idx(1) = j
        call MatSetValues(B, n, row_idxs, 1, col_idx, &
                          B_dense(:, j+1), INSERT_VALUES, ierr)
      enddo

      deallocate(row_idxs)
      deallocate(B_dense)
    endif

    call MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (B, MAT_FINAL_ASSEMBLY, ierr)

    ! -----------------------------------------------------------------
    ! 5. Cleanup Workspace
    ! -----------------------------------------------------------------
    call VecScatterDestroy(scat, ierr)
    call VecDestroy(y_seq, ierr)
    call VecDestroy(e_j,   ierr)
    call VecDestroy(z,     ierr)
    call VecDestroy(y,     ierr)
    call KSPDestroy(ksp_M, ierr)

    if (my_id == 0) then
      write(*,'(A)') "[Diagnostics] Assembly of B = M^{-1} A complete."
      flush(6)
    endif

  end subroutine compute_explicit_preconditioned_matrix


  !--------------------------------------------------------------------
  !> Set up a sub-KSP for a diagonal block: PREONLY + LU + MUMPS.
  !--------------------------------------------------------------------
  subroutine setup_block_ksp(ksp_block, B_block, comm, first_time)
    KSP, intent(inout) :: ksp_block
    Mat, intent(in)    :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time

    PC :: pc
    PetscErrorCode :: ierr

    if (first_time) then
      call KSPCreate(comm, ksp_block, ierr)
    endif
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    call KSPSetType(ksp_block, KSPPREONLY, ierr)
    call KSPGetPC(ksp_block, pc, ierr)
    call PCSetType(pc, PCLU, ierr)
    call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
    call KSPSetUp(ksp_block, ierr)
  end subroutine setup_block_ksp

  !--------------------------------------------------------------------
  !> Set up a sub-KSP for a diagonal block: PREONLY + GAMG (1 V-Cycle).
  !--------------------------------------------------------------------
  subroutine setup_block_ksp_amg(ksp_block, B_block, comm, first_time)
    implicit none

    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time

    PC :: pc
    PetscErrorCode :: ierr

    if (first_time) then
      call KSPCreate(comm, ksp_block, ierr)
    endif
    
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    
    ! PREONLY means "just apply the preconditioner once". 
    ! For an AMG preconditioner, this results in exactly one V-cycle.
    call KSPSetType(ksp_block, KSPPREONLY, ierr)
    
    call KSPGetPC(ksp_block, pc, ierr)
    
    ! Set the preconditioner to PETSc's native Algebraic Multigrid (GAMG)
    call PCSetType(pc, PCGAMG, ierr)
    
    ! (Optional but recommended) Explicitly set it to use Smoothed Aggregation.
    ! Smoothed Aggregation is much better for block/non-M-matrices than classical AMG.
    call PCGAMGSetType(pc, PCGAMGAGG, ierr)
    
    call KSPSetUp(ksp_block, ierr)
    
  end subroutine setup_block_ksp_amg

  !--------------------------------------------------------------------
  !> Set up a sub-KSP for a diagonal block: GMRES + GAMG (3-10 iters)
  !--------------------------------------------------------------------
  subroutine setup_block_ksp_amg_krylov(ksp_block, B_block, comm, first_time, max_its)
    implicit none

    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    integer, intent(in) :: max_its      ! Pass in 3 to 10

    PC :: pc
    PetscErrorCode :: ierr
    PetscReal :: rtol, abstol, dtol

    if (first_time) then
      call KSPCreate(comm, ksp_block, ierr)
    endif
    
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    
    ! 1. Choose a lightweight Krylov solver
    ! GMRES is highly robust for non-symmetric/non-M-matrices.
    call KSPSetType(ksp_block, KSPGMRES, ierr)
    
    ! Keep memory footprint tiny by telling GMRES to restart 
    ! at max_its (it will only allocate 'max_its' search vectors).
    call KSPGMRESSetRestart(ksp_block, max_its, ierr)
    
    ! 2. Set tolerances and max iterations
    ! Set a loose relative tolerance (e.g., 1.0d-2 = 1% error reduction)
    ! so it can exit early if it converges in 3-4 iterations.
    ! Otherwise, it will hard-stop at 'max_its'.
    rtol   = 1.0d-2  
    abstol = 1.0d-50 ! Ignore absolute tolerance
    dtol   = 1.0d4   ! Divergence tolerance
    call KSPSetTolerances(ksp_block, rtol, abstol, dtol, max_its, ierr)
    
    ! 3. Set the Preconditioner to GAMG
    call KSPGetPC(ksp_block, pc, ierr)
    call PCSetType(pc, PCGAMG, ierr)
    call PCGAMGSetType(pc, PCGAMGAGG, ierr)
    
    call KSPSetUp(ksp_block, ierr)
    
  end subroutine setup_block_ksp_amg_krylov

  subroutine setup_block_ksp_hypre_amg_krylov(ksp_block, B_block, comm, first_time, max_its)
    implicit none

    KSP, intent(inout)  :: ksp_block
    Mat, intent(in)     :: B_block
    integer, intent(in) :: comm
    logical, intent(in) :: first_time
    integer, intent(in) :: max_its      ! Pass in 3 to 10

    PC :: pc
    MatNullSpace :: nullsp
    PetscErrorCode :: ierr
    PetscReal :: rtol, abstol, dtol

    if (first_time) then
      call KSPCreate(comm, ksp_block, ierr)
    endif
    
    call KSPSetOperators(ksp_block, B_block, B_block, ierr)
    
    ! Set options prefix to match the hmg_ configuration
    call KSPSetOptionsPrefix(ksp_block, "hmg_", ierr)

    ! 1. Choose a lightweight Krylov solver
    call KSPSetType(ksp_block, KSPGMRES, ierr)
    
    ! Keep memory footprint tiny
    call KSPGMRESSetRestart(ksp_block, max_its, ierr)
    
    ! 2. Set tolerances and max iterations
    rtol   = 1.0d-2  
    abstol = 1.0d-50 
    dtol   = 1.0d4   
    call KSPSetTolerances(ksp_block, rtol, abstol, dtol, max_its, ierr)
    
    ! 3. Set the Preconditioner to Hypre BoomerAMG
    call KSPGetPC(ksp_block, pc, ierr)
    call PCSetType(pc, PCHYPRE, ierr)
    call PCHYPRESetType(pc, "boomeramg", ierr)

    ! Set Near Null Space on the operator
    call MatNullSpaceCreate(comm, PETSC_TRUE, 0, PETSC_NULL_VEC, nullsp, ierr)
    call MatSetNearNullSpace(B_block, nullsp, ierr)
    call MatNullSpaceDestroy(nullsp, ierr)

    ! 4. Set Hypre options via the PETSc options database
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_coarsen_type", "HMIS", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_nodal_coarsen", "6", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_truncfactor", "0.3", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_P_max", "4", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_strong_threshold", "0.5", ierr)

    ! Smoothers
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_smooth_num_levels", "1", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_smooth_type", "Euclid", ierr)
    !call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_euclid_levels", "1", ierr)
    call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_relax_type_all", "Chebyshev", ierr)
    !call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-hmg_pc_hypre_boomeramg_min_iter", "250", ierr)

    ! Apply the options database changes to this KSP instance
    call KSPSetFromOptions(ksp_block, ierr)
    
    call KSPSetUp(ksp_block, ierr)
    
  end subroutine setup_block_ksp_hypre_amg_krylov

  !--------------------------------------------------------------------
  !> Assemble the monolithic 4x4 reduced system via MatCreateNest +
  !! MatConvert, and set up a single KSP (PREONLY+LU+MUMPS).
  !--------------------------------------------------------------------
  subroutine assemble_monolithic_4x4(use_reassembled, comm, first_time, my_id, skip_ksp_setup)
     use mod_petsc_matrix_analysis, only: petsc_mat_convert_spectrum, petsc_mat_equilibrate
    logical, intent(in) :: use_reassembled, first_time, skip_ksp_setup
    integer, intent(in) :: comm, my_id

    Mat :: mats_nest(16), A_nest   ! 1D row-major: (row0,col0), (row0,col1), ...
    Mat :: diag_55, diag_66
    PC  :: pc_obj
    PetscErrorCode :: ierr
    PetscInt :: rstart, rend, n_local, n_global
    PetscInt, parameter :: nblocks = 4
    integer :: k

    Mat :: mats_nest_hydro(9), A_nest_hydro
    Mat :: mats_nest_alfven(4), A_nest_alfven, A_alfven_2x2

    ! Choose diagonal blocks B_55/B_66 or R_55/R_66
    if (use_reassembled) then
      diag_55 = g_ctx%R_55
      diag_66 = g_ctx%R_66
    else
      diag_55 = g_ctx%B_55
      diag_66 = g_ctx%B_66
    endif

    ! Populate nest in row-major order (PETSc Fortran convention for MatCreateNest)
    ! Row 1 (psi): Atilde_11  B_12        0          B_16
    mats_nest( 1) = g_ctx%Atilde_11
    mats_nest( 2) = g_ctx%B_12
    mats_nest( 3) = PETSC_NULL_MAT
    mats_nest( 4) = g_ctx%B_16
    ! Row 2 (u):   Atilde_21  Atilde_22   B_25       B_26
    mats_nest( 5) = g_ctx%Atilde_21
    mats_nest( 6) = g_ctx%Atilde_22
    mats_nest( 7) = g_ctx%B_25
    mats_nest( 8) = g_ctx%B_26
    ! Row 3 (rho): B_51       B_52        B_55/R_55  0
    mats_nest( 9) = g_ctx%B_51
    mats_nest(10) = g_ctx%B_52
    mats_nest(11) = diag_55
    mats_nest(12) = PETSC_NULL_MAT
    ! Row 4 (T):   Atilde_61  B_62        0          B_66/R_66
    mats_nest(13) = g_ctx%Atilde_61
    mats_nest(14) = g_ctx%B_62
    mats_nest(15) = PETSC_NULL_MAT
    mats_nest(16) = diag_66

    PetscCallA(MatCreateNest(comm, nblocks, PETSC_NULL_IS, nblocks, PETSC_NULL_IS, mats_nest, A_nest, ierr))

    ! Destroy old monolithic AIJ if rebuilding
    if (.not. first_time .and. g_ctx%ksp_reduced_created) then
      call MatDestroy(g_ctx%A_reduced_4x4, ierr)
    endif

    ! Convert nest to concrete AIJ
    call MatConvert(A_nest, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%A_reduced_4x4, ierr)
    call MatDestroy(A_nest, ierr)

    mats_nest_hydro(1) = g_ctx%Atilde_22
    mats_nest_hydro(2) = g_ctx%B_25
    mats_nest_hydro(3) = g_ctx%B_26
    mats_nest_hydro(4) = g_ctx%B_52
    mats_nest_hydro(5) = diag_55
    mats_nest_hydro(6) = PETSC_NULL_MAT
    mats_nest_hydro(7) = g_ctx%B_62
    mats_nest_hydro(8) = PETSC_NULL_MAT
    mats_nest_hydro(9) = diag_66

    PetscCallA(MatCreateNest(comm, 3, PETSC_NULL_IS, 3, PETSC_NULL_IS, mats_nest_hydro, A_nest_hydro, ierr))

    if (.not. first_time .and. g_ctx%ksp_hydro_created) then
      call MatDestroy(g_ctx%M_hydro, ierr)
    endif

    call MatConvert(A_nest_hydro, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%M_hydro, ierr)
    call MatDestroy(A_nest_hydro, ierr)

    !call petsc_mat_convert_spectrum(g_ctx%M_hydro, "M_hydro", .false.)
    !call petsc_test_pc_matrix(g_ctx%M_hydro, "M_hydro", .false., my_id)

    mats_nest_alfven(1) = g_ctx%Atilde_11
    mats_nest_alfven(2) = g_ctx%B_12
    mats_nest_alfven(3) = g_ctx%Atilde_21
    mats_nest_alfven(4) = g_ctx%Atilde_22

    PetscCallA(MatCreateNest(comm, 2, PETSC_NULL_IS, 2, PETSC_NULL_IS, mats_nest_alfven, A_nest_alfven, ierr))

    if (.not. first_time .and. g_ctx%ksp_alfven_created) then
      call MatDestroy(g_ctx%A_alfven, ierr)
    endif

    call MatConvert(A_nest_alfven, MATMPIAIJ, MAT_INITIAL_MATRIX, g_ctx%A_alfven, ierr)
    call MatDestroy(A_nest_alfven, ierr)

    !call petsc_mat_convert_spectrum(g_ctx%A_alfven, "A_alfven_2x2", .false.)
    !call petsc_test_pc_matrix(g_ctx%A_alfven, "A_alfven_2x2", .false., my_id)

    ! Set up monolithic KSP (skipped when probe_exact=.true.: probe owns the KSP)
    if (.not. skip_ksp_setup) then
      if (first_time) then
        call KSPCreate(comm, g_ctx%ksp_reduced, ierr)
      endif
      call KSPSetOperators(g_ctx%ksp_reduced, g_ctx%A_reduced_4x4, g_ctx%A_reduced_4x4, ierr)
      call KSPSetType(g_ctx%ksp_reduced, KSPPREONLY, ierr)
      call KSPGetPC(g_ctx%ksp_reduced, pc_obj, ierr)
      call PCSetType(pc_obj, PCLU, ierr)
      call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
      call KSPSetUp(g_ctx%ksp_reduced, ierr)
      g_ctx%ksp_reduced_created = .true.
    endif

    ! Create ksp for sub-blocks (Alfven and hydro (For now mumps but to be changed...))
    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_hydro, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_hydro, g_ctx%M_hydro, g_ctx%M_hydro, ierr)
    call KSPSetType(g_ctx%ksp_hydro, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_hydro, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_hydro, ierr)
    g_ctx%ksp_hydro_created = .true.

    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_alfven, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_alfven, g_ctx%A_alfven, g_ctx%A_alfven, ierr)
    call KSPSetType(g_ctx%ksp_alfven, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_alfven, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_alfven, ierr)
    g_ctx%ksp_alfven_created = .true.

    ! S_PBP (inner Alfven Schur operator, u-sized) — direct solver for the
    ! three-step Alfven corrector. Guard against an unassembled S_PBP: a zero
    ! matrix would make MUMPS fail opaquely or produce garbage.
    block
      PetscReal :: norm_spbp
      call MatNorm(g_ctx%S_PBP, NORM_FROBENIUS, norm_spbp, ierr)
      if (norm_spbp <= 0.0d0) then
        if (my_id == 0) write(*,'(A)') &
          "[Physics PC] ERROR: S_PBP is empty/zero — element assembly missing; cannot set up ksp_S_PBP"
        ierr = 1
        return
      endif
      if (my_id == 0) write(*,'(A,ES12.4)') "[Physics PC]   ||S_PBP||_F = ", norm_spbp
    end block

    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_S_PBP, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_S_PBP, g_ctx%S_PBP, g_ctx%S_PBP, ierr)
    call KSPSetType(g_ctx%ksp_S_PBP, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_S_PBP, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_S_PBP, ierr)
    g_ctx%ksp_S_PBP_created = .true.
    if (my_id == 0) write(*,'(A)') "[Physics PC]   ksp_S_PBP set up (PREONLY+LU+MUMPS)"

    ! Allocate 4-var work vectors and create index sets (first time only)
    if (.not. g_ctx%work_4v_created) then
      call MatCreateVecs(g_ctx%A_reduced_4x4, g_ctx%work_sol_4v, g_ctx%work_rhs_4v, ierr)
      g_ctx%work_4v_created = .true.

      ! Create IS for each variable in the monolithic vector
      ! Each sub-block has identical parallel layout, so variable k
      ! occupies rows [rstart + k*n_local .. rstart + (k+1)*n_local - 1]
      call VecGetOwnershipRange(g_ctx%work_rhs_4v, rstart, rend, ierr)
      n_local = (rend - rstart) / 4
      call VecGetSize(g_ctx%work_rhs_4v, n_global, ierr)
      do k = 1, 4
        call ISCreateStride(comm, n_local, rstart + (k-1)*n_local, 1, &
                            g_ctx%is_reduced(k), ierr)
      enddo

      call MatCreateVecs(g_ctx%M_hydro, g_ctx%hydro_predictor_3v, g_ctx%work_rhs_3v, ierr)
      do k = 1, 3
        call ISCreateStride(comm, n_local, rstart + (k-1)*n_local, 1, &
                            g_ctx%is_hydro(k), ierr)
      enddo
      g_ctx%is_reduced_created = .true.
    endif

    if (my_id == 0) then
      if (skip_ksp_setup) then
        write(*,'(A)') "[Physics PC]   Monolithic 4x4 approx matrix built (KSP deferred to probe)"
      else
        write(*,'(A)') "[Physics PC]   Monolithic 4x4 KSP set up (PREONLY+LU+MUMPS)"
      endif
    endif
  end subroutine assemble_monolithic_4x4


  !--------------------------------------------------------------------
  !> Assemble the EXACT 4×4 Schur complement by probing:
  !! apply S_4x4 to all 4N standard basis vectors, gather columns to
  !! rank 0, build a dense AIJ matrix, and replace the approximate
  !! operator in ksp_reduced.
  !!
  !! Cost: 2N KSPSolve (triangular substitutions on existing MUMPS
  !! factorisations) + 14N MatMult.  Intended for small test problems.
  !!
  !! Must be called AFTER assemble_monolithic_4x4 so that ksp_reduced
  !! and g_ctx%A_reduced_4x4 (the approximate baseline) already exist.
  !--------------------------------------------------------------------
  subroutine assemble_probed_exact_4x4(use_reassembled, comm, first_time, my_id)
    use mod_petsc_matrix_analysis, only: petsc_mat_diff_norm
    implicit none
    logical, intent(in) :: use_reassembled, first_time
    integer, intent(in) :: comm, my_id

    Vec            :: e_j, z, temp, r_blk, scratch, r_seq
    VecScatter     :: scat
    Mat            :: A_exact, diag_55, diag_66
    PC             :: pc_obj
    PetscInt       :: N_global, n4p, m_local_4v
    PetscInt       :: rstart_1v_p, rend_1v_p
    PetscErrorCode :: ierr
    PetscScalar, pointer :: arr(:)
    PetscInt, allocatable :: interleaved_idxs(:)  ! block-contiguous → 4N interleaved global index
    PetscInt       :: row_idx(1)
    integer        :: n, n4, j, p, j_loc, kb
    integer        :: nproc, mpierr, n_local_1v
    integer, allocatable :: n_local_arr(:), rstart_arr(:)
    real*8, allocatable :: A_dense(:,:)
    PetscReal      :: norm_approx, norm_diff, norm_exact

    call MatGetSize(g_ctx%B_11, N_global, PETSC_NULL_INTEGER, ierr)
    n   = int(N_global)
    n4  = 4 * n
    n4p = n4

    ! Choose reassembled or extracted diagonal blocks for ρ and T
    if (use_reassembled) then
      diag_55 = g_ctx%R_55
      diag_66 = g_ctx%R_66
    else
      diag_55 = g_ctx%B_55
      diag_66 = g_ctx%B_66
    endif

    ! Local work vectors — work_1..5 not yet allocated at build time
    call MatCreateVecs(g_ctx%B_11, e_j, PETSC_NULL_VEC, ierr)
    call VecDuplicate(e_j, z,       ierr)
    call VecDuplicate(e_j, temp,    ierr)
    call VecDuplicate(e_j, r_blk,   ierr)
    call VecDuplicate(e_j, scratch, ierr)
    call VecScatterCreateToZero(e_j, scat, r_seq, ierr)

    ! Build mapping: block-contiguous 4N index → interleaved 4N global index.
    ! For P=1 these are identical; for P>1 the 4N matrix layout produced by
    ! MatConvert(MATNEST→MATMPIAIJ) interleaves variable blocks per process:
    !   process p owns [rstart_4v_p .. rstart_4v_p + 4*nloc_p - 1] where
    !   rstart_4v_p = 4 * rstart_1v_p and within that range variable k (0-based)
    !   occupies [rstart_4v_p + k*nloc_p .. rstart_4v_p + (k+1)*nloc_p - 1].
    ! Block-contiguous index kb*n + j_global maps to interleaved index:
    !   4*rstart_1v[p] + kb*nloc[p] + (j_global - rstart_1v[p])
    ! where p is the process owning j_global in the 1-variable distribution.
    call MPI_Comm_size(comm, nproc, mpierr)
    call VecGetOwnershipRange(e_j, rstart_1v_p, rend_1v_p, ierr)
    n_local_1v = int(rend_1v_p - rstart_1v_p)
    allocate(n_local_arr(nproc), rstart_arr(nproc+1))
    call MPI_Allgather(n_local_1v, 1, MPI_INTEGER, n_local_arr, 1, MPI_INTEGER, comm, mpierr)
    rstart_arr(1) = 0
    do p = 1, nproc
      rstart_arr(p+1) = rstart_arr(p) + n_local_arr(p)
    enddo

    ! Build interleaved_idxs on rank 0 (only rank 0 inserts into A_exact)
    if (my_id == 0) then
      allocate(interleaved_idxs(n4))
      p = 1
      do j_loc = 0, n-1
        do while (j_loc >= rstart_arr(p+1))
          p = p + 1
        enddo
        do kb = 0, 3
          interleaved_idxs(kb*n + j_loc + 1) = &
            4*rstart_arr(p) + kb*n_local_arr(p) + (j_loc - rstart_arr(p))
        enddo
      enddo
    endif

    if (my_id == 0) then
      allocate(A_dense(n4, n4))
      A_dense = 0.0d0   ! ensure structural zeros are correct before block-by-block fill
      write(*,'(A,I0,A)') "[Physics PC]   Probing exact 4x4 Schur (4x", n, " columns)..."
      flush(6)
    endif

    ! =================================================================
    ! Block 1: ψ input → columns 0..n-1
    !   temp_j = ksp_Mj^{-1} · B_31 · e_ψ
    !   r_ψ = B_11·e_ψ − B_13·temp_j
    !   r_u = B_21·e_ψ − B_23·temp_j
    !   r_ρ = B_51·e_ψ
    !   r_T = B_61·e_ψ − B_63·temp_j
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 1/4 (psi, ksp_Mj):"
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)
      call MatMult(g_ctx%B_31, e_j, z,    ierr)
      call KSPSolve(g_ctx%ksp_Mj, z, temp, ierr)

      ! r_ψ
      call MatMult(g_ctx%B_11, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_13, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(1:n, j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_u
      call MatMult(g_ctx%B_21, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_23, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(n+1:2*n, j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_ρ  (no Schur coupling from ψ to ρ through j)
      call MatMult(g_ctx%B_51, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(2*n+1:3*n, j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_T
      call MatMult(g_ctx%B_61, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_63, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(3*n+1:4*n, j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Physics PC]     Block 1/4: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! =================================================================
    ! Block 2: u input → columns n..2n-1
    !   temp_w = ksp_Mw^{-1} · B_42 · e_u
    !   r_ψ = B_12·e_u
    !   r_u = B_22·e_u − B_24·temp_w
    !   r_ρ = B_52·e_u
    !   r_T = B_62·e_u
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 2/4 (u, ksp_Mw):"
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)
      call MatMult(g_ctx%B_42, e_j, z,    ierr)
      call KSPSolve(g_ctx%ksp_Mw, z, temp, ierr)

      ! r_ψ  (no Schur from u to ψ through j)
      call MatMult(g_ctx%B_12, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(1:n, n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_u
      call MatMult(g_ctx%B_22, e_j, r_blk, ierr)
      call MatMult(g_ctx%B_24, temp, scratch, ierr)
      call VecAXPY(r_blk, -1.0d0, scratch, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(n+1:2*n, n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_ρ
      call MatMult(g_ctx%B_52, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(2*n+1:3*n, n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_T  (no Schur: u doesn't drive j-elimination in T row)
      call MatMult(g_ctx%B_62, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(3*n+1:4*n, n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif
      if (my_id == 0 .and. mod(j+1, max(1,n/10)) == 0) then
        write(*,'(A,I0,A,I0,A,I0,A)') &
          "[Physics PC]     Block 2/4: ", j+1, "/", n, " (", (j+1)*100/n, "%)"
        flush(6)
      endif
    enddo

    ! =================================================================
    ! Block 3: ρ input → columns 2n..3n-1
    !   No Schur corrections (ρ not in constraint system)
    !   r_ψ = 0,  r_u = B_25·e_ρ,  r_ρ = B_55·e_ρ,  r_T = 0
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 3/4 (rho, MatMult)..."
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)

      ! r_u
      call MatMult(g_ctx%B_25, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(n+1:2*n, 2*n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_ρ
      call MatMult(diag_55, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(2*n+1:3*n, 2*n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif
    enddo

    ! =================================================================
    ! Block 4: T input → columns 3n..4n-1
    !   No Schur corrections (T not in constraint system)
    !   r_ψ = B_16·e_T,  r_u = B_26·e_T,  r_ρ = 0,  r_T = B_66·e_T
    ! =================================================================
    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]     Block 4/4 (T, MatMult)..."
      flush(6)
    endif
    do j = 0, n-1
      call VecSet(e_j, 0.0d0, ierr)
      call VecSetValue(e_j, j, 1.0d0, INSERT_VALUES, ierr)
      call VecAssemblyBegin(e_j, ierr);  call VecAssemblyEnd(e_j, ierr)

      ! r_ψ
      call MatMult(g_ctx%B_16, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(1:n, 3*n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_u
      call MatMult(g_ctx%B_26, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(n+1:2*n, 3*n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif

      ! r_T
      call MatMult(diag_66, e_j, r_blk, ierr)
      call VecScatterBegin(scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      call VecScatterEnd  (scat, r_blk, r_seq, INSERT_VALUES, SCATTER_FORWARD, ierr)
      if (my_id == 0) then
        call VecGetArrayF90(r_seq, arr, ierr)
        A_dense(3*n+1:4*n, 3*n+j+1) = real(arr, kind=8)
        call VecRestoreArrayF90(r_seq, arr, ierr)
      endif
    enddo

    ! =================================================================
    ! Build MATMPIAIJ from A_dense (rank 0 inserts all rows)
    ! =================================================================
    ! Use the same local row count as work_rhs_4v (created from A_approx's nest layout)
    ! so that A_exact is compatible with the work vectors in KSPSolve.
    ! PETSC_DECIDE would distribute rows independently of the nest structure, producing
    ! a different layout when N is not divisible by the number of MPI ranks.
    call VecGetLocalSize(g_ctx%work_rhs_4v, m_local_4v, ierr)
    call MatCreate(comm, A_exact, ierr)
    call MatSetSizes(A_exact, m_local_4v, m_local_4v, n4p, n4p, ierr)
    call MatSetType(A_exact, MATMPIAIJ, ierr)
    call MatSetOption(A_exact, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
    call MatSetUp(A_exact, ierr)

    if (my_id == 0) then
      ! Insert column j (block-contiguous) into interleaved column interleaved_idxs(j+1).
      ! Row indices are also remapped via interleaved_idxs so that A_dense(i, j+1) —
      ! which holds S(bc_row=i-1, bc_col=j) — lands at A_exact(interleaved(i-1), interleaved(j)).
      do j = 0, n4-1
        row_idx(1) = interleaved_idxs(j+1)
        call MatSetValues(A_exact, n4p, interleaved_idxs, 1, row_idx, &
                          A_dense(:, j+1), INSERT_VALUES, ierr)
      enddo
      deallocate(interleaved_idxs, A_dense)
    endif
    deallocate(n_local_arr, rstart_arr)
    call MatAssemblyBegin(A_exact, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd  (A_exact, MAT_FINAL_ASSEMBLY, ierr)

    ! Diagnostic: relative Frobenius error vs approximate matrix
    call petsc_mat_diff_norm(A_exact, g_ctx%A_reduced_4x4, &
                             "A_exact_4x4 - A_approx_4x4", norm_diff)
    call MatNorm(g_ctx%A_reduced_4x4, NORM_FROBENIUS, norm_approx, ierr)
    call MatNorm(A_exact,             NORM_FROBENIUS, norm_exact,  ierr)
    if (my_id == 0) then
      write(*,'(A,ES12.4)') "[Physics PC]   ||A_approx||_F        = ", norm_approx
      write(*,'(A,ES12.4)') "[Physics PC]   ||A_exact||_F         = ", norm_exact
      write(*,'(A,ES12.4)') &
        "[Physics PC]   ||A_exact - A_approx||_F / ||A_approx||_F = ", &
        norm_diff / norm_approx
    endif

    ! Replace approximate matrix and set up KSP from scratch.
    ! assemble_monolithic_4x4 was called with skip_ksp_setup=.true., so ksp_reduced
    ! does not exist yet on first_time; on subsequent calls it exists but must be
    ! re-factored with the new A_exact.  A fresh KSPCreate on first_time and a full
    ! setup sequence on every call avoids any stale-factorization pitfalls.
    call MatDestroy(g_ctx%A_reduced_4x4, ierr)
    g_ctx%A_reduced_4x4 = A_exact
    if (first_time) then
      call KSPCreate(comm, g_ctx%ksp_reduced, ierr)
    endif
    call KSPSetOperators(g_ctx%ksp_reduced, g_ctx%A_reduced_4x4, g_ctx%A_reduced_4x4, ierr)
    call KSPSetType(g_ctx%ksp_reduced, KSPPREONLY, ierr)
    call KSPGetPC(g_ctx%ksp_reduced, pc_obj, ierr)
    call PCSetType(pc_obj, PCLU, ierr)
    call PCFactorSetMatSolverType(pc_obj, MATSOLVERMUMPS, ierr)
    call KSPSetUp(g_ctx%ksp_reduced, ierr)
    g_ctx%ksp_reduced_created = .true.

    ! Cleanup local temporaries
    call VecScatterDestroy(scat,    ierr)
    call VecDestroy(r_seq,   ierr)
    call VecDestroy(e_j,     ierr)
    call VecDestroy(z,       ierr)
    call VecDestroy(temp,    ierr)
    call VecDestroy(r_blk,   ierr)
    call VecDestroy(scratch, ierr)

    if (my_id == 0) then
      write(*,'(A)') "[Physics PC]   Exact probed 4x4 set up (PREONLY+LU+MUMPS)"
      flush(6)
    endif
  end subroutine assemble_probed_exact_4x4

#endif
end module mod_petsc_pc_physics_construction
