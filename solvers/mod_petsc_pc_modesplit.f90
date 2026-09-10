!> Toroidal mode-family preconditioner on disjoint MPI sub-communicators.
!!
!! Same operator as mod_petsc_pc_toroidal - block-diagonal in the toroidal mode
!! families - but a fundamentally different parallel layout.
!!
!! PCFIELDSPLIT applies its splits *sequentially*: every rank takes part in every
!! block's factorization and every block's solve, one block after another. Each
!! block is therefore factorized across all ranks, which is exactly the regime
!! where a sparse direct solver stops scaling. Reducing a block with PCTELESCOPE
!! does not help either: PetscSubcomm_isActiveRank puts colour 0 on rank 0 for
!! every reduction factor and subcomm type, so *all* blocks land on rank 0 - with
!! cuDSS that means one busy GPU and the rest idle (tools/cudss/README.md).
!!
!! Here the ranks are partitioned into one disjoint sub-communicator per mode
!! family, each holding its own sub-matrix and its own direct-solver KSP. Because
!! the communicators are disjoint, the KSPSolve calls inside PCApply - and the
!! factorizations inside PCSetUp - run concurrently. This is the layout the legacy
!! Fortran path has always used (mod_preconditioner), expressed in PETSc objects,
!! and it shares the decomposition itself with that path through mod_mode_families.
!!
!! Selected with -jorek_pc_mode_split; PCFIELDSPLIT remains the default.
!!
!! Requires n_ranks >= n_mode_families, with at least one rank per family - the
!! same constraint the legacy path has.
module mod_petsc_pc_modesplit
#ifdef USE_PETSC
  use mpi_mod
  use mod_petsc_direct_solver, only: petsc_configure_direct_solver, petsc_report_solver
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private
  public :: petsc_setup_modesplit_pc

  !> Shared options prefix for every block solver, the same one PCFIELDSPLIT's
  !! blocks use, so every option documented in namelist/jorek.petsc.example keeps
  !! working across both preconditioners.
  character(len=*), parameter :: PC_BLOCK_PREFIX = 'jorek_pcblock_'

  !> Everything the shell owns. There is exactly one such preconditioner per run,
  !! so a saved module instance is used rather than PCShellSetContext - the same
  !! choice mod_petsc_pc_physics makes, and it keeps the callbacks free of any
  !! Fortran/C pointer round-tripping.
  type :: type_modesplit_ctx
    logical :: initialized   = .false.
    logical :: solver_ready  = .false.   !< ksp_fam has been created

    integer :: comm      = MPI_COMM_NULL  !< global communicator (that of the operator)
    integer :: comm_fam  = MPI_COMM_NULL  !< this rank's mode-family communicator
    integer :: my_id     = -1             !< rank in comm
    integer :: n_cpu     = 0              !< size of comm
    integer :: my_id_fam = -1             !< rank in comm_fam
    integer :: n_fam     = 0              !< number of mode families
    integer :: my_fam    = -1             !< 1-based family this rank belongs to

    !> .true. when there is a single family, i.e. comm_fam == comm and the family
    !! operator is the full operator. Nothing is extracted or scattered then.
    logical :: passthrough = .false.

    integer, dimension(:),   pointer :: modes_per_fam  => Null()
    integer, dimension(:),   pointer :: ranks_per_fam  => Null()
    integer, dimension(:),   pointer :: rank_range     => Null()
    integer, dimension(:,:), pointer :: fam_modes      => Null()
    integer, dimension(:,:), pointer :: fam_ranks      => Null()

    real(kind=8) :: weight   = 1.0d0      !< weights_per_family of this rank's family
    logical      :: weighted = .false.    !< .true. when that weight is not exactly 1

    !> Index sets carving family f out of the global operator. Kept for the life of
    !! the PC because MAT_REUSE_MATRIX refreshes need the very same IS objects.
    IS,  dimension(:), pointer :: is_fam   => Null()
    Mat, dimension(:), pointer :: B_fam    => Null()  !< family submatrix, still on comm
    Mat, dimension(:), pointer :: B_loc    => Null()  !< its local rows, sequential

    Mat :: A_fam                          !< this family's operator, on comm_fam
    KSP :: ksp_fam                        !< its direct solver, on comm_fam

    VecScatter :: scatter                 !< global vector <-> staging vector
    Vec :: stage                          !< on comm, local size = n_loc
    Vec :: rhs_fam, sol_fam               !< on comm_fam, local size = n_loc

    PetscInt :: n_loc = 0                 !< rows of my family owned by this rank
  end type type_modesplit_ctx

  type(type_modesplit_ctx), save :: g_ctx

contains

  !> Install the mode-split preconditioner on an existing KSP.
  !!
  !! Only registration happens here. The operator-dependent work is in
  !! modesplit_setup, which PCSetUp invokes - from the KSPSetUp that follows this
  !! call, and again on every later KSPSetUp that is not suppressed by
  !! KSPSetReusePreconditioner. That is the same lifecycle the fieldsplit path
  !! already gets from petsc_solve_iterative_and_retrieve, so no caller changes.
  subroutine petsc_setup_modesplit_pc(ksp, A)
    KSP, intent(inout) :: ksp
    Mat, intent(in)    :: A

    PC :: pc
    PetscErrorCode :: ierr

    call modesplit_decompose(A)

    PetscCallA(KSPGetPC(ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCSHELL, ierr))
    PetscCallA(PCShellSetName(pc, 'jorek_mode_split', ierr))
    PetscCallA(PCShellSetSetUp(pc, modesplit_setup, ierr))
    PetscCallA(PCShellSetApply(pc, modesplit_apply, ierr))
    PetscCallA(PCShellSetDestroy(pc, modesplit_destroy, ierr))
  end subroutine petsc_setup_modesplit_pc


  !> Partition the ranks and the toroidal modes into families, and split the
  !! communicator accordingly. Depends only on the namelist and on the operator's
  !! shape, so it runs once - never again on a matrix refresh.
  subroutine modesplit_decompose(A)
    use mod_mode_families, only: mode_family_count, mode_family_distribute_modes, &
                                 mode_family_distribute_ranks, mode_family_weight

    Mat, intent(in) :: A

    integer :: i, mpierr
    integer, dimension(:), pointer :: family_of_rank => Null()
    logical :: ok
    character(len=256) :: s
    PetscErrorCode :: ierr

    if (g_ctx%initialized) return

    PetscCallA(PetscObjectGetComm(A, g_ctx%comm, ierr))
    call MPI_COMM_RANK(g_ctx%comm, g_ctx%my_id, mpierr)
    call MPI_COMM_SIZE(g_ctx%comm, g_ctx%n_cpu, mpierr)

    g_ctx%n_fam = mode_family_count()

    if (g_ctx%n_fam > g_ctx%n_cpu) then
      if (g_ctx%my_id == 0) write(*,'(A,I0,A,I0,A)') &
        ' [PETSc] -jorek_pc_mode_split needs at least one rank per mode family, but there are ', &
        g_ctx%n_fam, ' families and only ', g_ctx%n_cpu, ' ranks.'
      SETERRA(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, 'mode-split PC: fewer MPI ranks than mode families')
    endif

    call mode_family_distribute_ranks(g_ctx%n_cpu, g_ctx%n_fam, g_ctx%ranks_per_fam, &
                                      g_ctx%rank_range, g_ctx%fam_ranks, family_of_rank, ok)
    if (.not. ok) then
      if (g_ctx%my_id == 0) write(*,'(A,I0,A)') &
        ' [PETSc] -jorek_pc_mode_split: ranks_per_family must be positive for every family and sum to ', &
        g_ctx%n_cpu, '.'
      SETERRA(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, 'mode-split PC: invalid ranks_per_family')
    endif

    g_ctx%my_fam = family_of_rank(g_ctx%my_id + 1)
    deallocate(family_of_rank)

    call mode_family_distribute_modes(g_ctx%n_fam, g_ctx%modes_per_fam, g_ctx%fam_modes)
    g_ctx%weight   = mode_family_weight(g_ctx%my_fam)
    g_ctx%weighted = (g_ctx%weight < 1.0d0) .or. (g_ctx%weight > 1.0d0)

    ! Colour by family, ordered by global rank, so each family's ranks keep their
    ! relative order and rank 0 is the first rank of family 1.
    call MPI_COMM_SPLIT(g_ctx%comm, g_ctx%my_fam, g_ctx%my_id, g_ctx%comm_fam, mpierr)
    call MPI_COMM_RANK(g_ctx%comm_fam, g_ctx%my_id_fam, mpierr)

    ! One family means comm_fam is comm and the family operator is the operator
    ! itself: no submatrix, no scatter, nothing to gain and nothing to do.
    g_ctx%passthrough = (g_ctx%n_fam == 1)

    if (g_ctx%my_id == 0) then
      write(*,'(A,I0,A,I0,A)') ' [PETSc] mode-split PC: ', g_ctx%n_fam, &
                               ' mode families over ', g_ctx%n_cpu, ' ranks'
      do i = 1, g_ctx%n_fam
        write(s,'(A,I4,A,I4,A,F6.2,A)') '   family ', i, ':', g_ctx%ranks_per_fam(i), &
                                        ' rank(s), weight ', mode_family_weight(i), ', modes:'
        write(*,*) trim(s), g_ctx%fam_modes(i,1:g_ctx%modes_per_fam(i))
      enddo
    endif

    g_ctx%initialized = .true.
  end subroutine modesplit_decompose


  !> PCSHELL setup callback: (re)build the per-family operators and their solvers.
  !!
  !! On the first call everything is created; afterwards only the values are
  !! refreshed, through MAT_REUSE_MATRIX on all three extraction steps - which is
  !! why the index sets and the intermediate matrices are kept in the context.
  !!
  !! Every step that touches the global operator is collective on the global
  !! communicator and is therefore executed by all ranks, for all families, in the
  !! same order. Only the concatenation onto comm_fam, and everything after it, is
  !! family-local - and that is where the concurrency comes from.
  !!
  !! The two phases are timed separately because they behave completely differently:
  !! the extraction is a global collective that every rank pays, while the
  !! factorization is family-local and is the only part that could ever be moved off
  !! the critical path. One number covering both would hide which of them dominates.
  subroutine modesplit_setup(pc, ierr)
    PC :: pc
    PetscErrorCode :: ierr

    Mat :: A, P, my_bloc
    integer :: f
    logical :: first
    MatReuse :: reuse
    PetscLogDouble :: t0, t1, t2

    PetscCallA(PCGetOperators(pc, A, P, ierr))
    PetscCallA(PetscTime(t0, ierr))

    if (g_ctx%passthrough) then
      call modesplit_setup_solver(P)
      PetscCallA(PetscTime(t2, ierr))
      call modesplit_report_setup(t0, t0, t2)
      ierr = 0
      return
    endif

    first = .not. associated(g_ctx%is_fam)
    if (first) then
      allocate(g_ctx%is_fam(g_ctx%n_fam), g_ctx%B_fam(g_ctx%n_fam), g_ctx%B_loc(g_ctx%n_fam))
      call modesplit_build_index_sets(P)
      reuse = MAT_INITIAL_MATRIX
    else
      reuse = MAT_REUSE_MATRIX
    endif

    do f = 1, g_ctx%n_fam
      ! Square diagonal block: the same index set selects the rows and the columns.
      ! On ranks outside family f the index set is empty, which is what moves the
      ! block's rows onto family f's ranks - PETSc does the redistribution.
      PetscCallA(MatCreateSubMatrix(P, g_ctx%is_fam(f), g_ctx%is_fam(f), reuse, g_ctx%B_fam(f), ierr))
      PetscCallA(MatMPIAIJGetLocalMat(g_ctx%B_fam(f), reuse, g_ctx%B_loc(f), ierr))
    enddo

    ! Family-local from here on. Concatenating this rank's rows over comm_fam gives
    ! the family operator; at one rank per family PETSc resolves that to SEQAIJ,
    ! which is exactly what the sequential packages (cuDSS, PETSc's own LU) need
    ! and what PCTELESCOPE was being used to fake.
    my_bloc = g_ctx%B_loc(g_ctx%my_fam)
    PetscCallA(MatCreateMPIMatConcatenateSeqMat(g_ctx%comm_fam, my_bloc, g_ctx%n_loc, reuse, g_ctx%A_fam, ierr))

    if (first) call modesplit_build_vectors(A)

    PetscCallA(PetscTime(t1, ierr))
    call modesplit_setup_solver(g_ctx%A_fam)
    PetscCallA(PetscTime(t2, ierr))
    call modesplit_report_setup(t0, t1, t2)

    ierr = 0
  end subroutine modesplit_setup


  !> Report where the PCSetUp time went: extracting the family operators out of the
  !! global one, versus factorizing them.
  !!
  !! Printed from global rank 0, so the factorization figure is family 1's. The
  !! families run concurrently but are not the same size, so this is a lower bound
  !! on the slowest family - the KSPSetUp is collective on comm_fam only and nothing
  !! here synchronizes the families against each other.
  subroutine modesplit_report_setup(t0, t1, t2)
    PetscLogDouble, intent(in) :: t0, t1, t2

    if (g_ctx%my_id /= 0) return
    write(*,'(A,ES12.4,A,ES12.4,A)') ' [PETSc] mode-split PCSetUp: extraction ', &
          t1 - t0, ' s, factorization ', t2 - t1, ' s'
  end subroutine modesplit_report_setup


  !> Build, for every family, the index set of the global rows belonging to it -
  !! distributed so that only that family's ranks own any of them.
  !!
  !! Within a block row the DOF order is variable-major / toroidal-minor: entry
  !! (j, m) of block ib sits at ib*bs + (j-1)*n_tor + (m-1), with bs = n_var*n_tor.
  !! This is the same indexing PCFIELDSPLIT is given in mod_petsc_pc_toroidal, so
  !! the two preconditioners select identical blocks.
  subroutine modesplit_build_index_sets(P)
    use mod_parameters, only: n_tor

    Mat, intent(in) :: P

    integer :: f, k, l, m
    PetscInt :: bs, n_glob, n_blockrows, split_size, n_tor_p
    PetscInt :: nm, n_fam_rows, n_mine, off_mine, hi, pos, ib, jj, kk, i
    PetscInt, allocatable :: idx(:), modes(:)
    Vec :: layout
    PetscErrorCode :: ierr

    PetscCallA(MatGetBlockSize(P, bs, ierr))
    PetscCallA(MatGetSize(P, n_glob, n_glob, ierr))
    n_tor_p     = int(n_tor, kind=kind(bs))
    n_blockrows = n_glob/bs
    split_size  = bs/n_tor_p                     ! = n_var

    do f = 1, g_ctx%n_fam
      nm         = int(g_ctx%modes_per_fam(f), kind=kind(nm))
      n_fam_rows = n_blockrows * split_size * nm

      if (f == g_ctx%my_fam) then
        ! Balanced contiguous chunks of this family's rows over its own ranks.
        ! Collective on comm_fam, and reached only at f == my_fam - so every rank
        ! of a family reaches it together, at its own iteration of this loop.
        PetscCallA(VecCreate(g_ctx%comm_fam, layout, ierr))
        PetscCallA(VecSetSizes(layout, PETSC_DECIDE, n_fam_rows, ierr))
        PetscCallA(VecSetType(layout, VECSTANDARD, ierr))
        PetscCallA(VecGetLocalSize(layout, n_mine, ierr))
        PetscCallA(VecGetOwnershipRange(layout, off_mine, hi, ierr))
        PetscCallA(VecDestroy(layout, ierr))
        g_ctx%n_loc = n_mine
      else
        n_mine   = 0
        off_mine = 0
      endif

      ! The family's modes in ascending order, so that the generated indices are
      ! ascending too and each rank's chunk is contiguous in the global numbering.
      allocate(modes(max(int(nm), 1)))
      do k = 1, int(nm)
        modes(k) = int(g_ctx%fam_modes(f,k), kind=kind(modes))
      enddo
      do k = 2, int(nm)                            ! insertion sort; nm <= n_tor
        m = int(modes(k))
        l = k - 1
        do while (l >= 1)
          if (int(modes(l)) <= m) exit
          modes(l+1) = modes(l)
          l = l - 1
        enddo
        modes(l+1) = int(m, kind=kind(modes))
      enddo

      allocate(idx(max(int(n_mine), 1)))
      do i = 1, n_mine
        pos = off_mine + i - 1             ! 0-based position within the family's rows
        ib  = pos/(split_size*nm)          ! block row
        jj  = mod(pos, split_size*nm)/nm   ! 0-based variable index within the block
        kk  = mod(pos, nm) + 1             ! which of the family's modes
        idx(i) = ib*bs + jj*n_tor_p + (modes(kk) - 1)
      enddo

      PetscCallA(ISCreateGeneral(g_ctx%comm, n_mine, idx, PETSC_COPY_VALUES, g_ctx%is_fam(f), ierr))
      deallocate(idx, modes)
    enddo
  end subroutine modesplit_build_index_sets


  !> The staging vector, the scatter that fills it, and the family's own vectors.
  !!
  !! A VecScatter cannot move data straight into a vector living on a smaller
  !! communicator, so it lands in `stage` - a global-communicator vector whose
  !! local size is exactly this rank's share of its family - and the family vectors
  !! then read it locally. This is the same staging PCSetUp_Telescope uses.
  subroutine modesplit_build_vectors(A)
    Mat, intent(in) :: A

    Vec :: x_tmpl
    IS  :: is_stage, my_is
    PetscInt :: lo, hi, one
    PetscErrorCode :: ierr

    one = 1

    PetscCallA(MatCreateVecs(A, x_tmpl, PETSC_NULL_VEC, ierr))

    PetscCallA(VecCreate(g_ctx%comm, g_ctx%stage, ierr))
    PetscCallA(VecSetSizes(g_ctx%stage, g_ctx%n_loc, PETSC_DETERMINE, ierr))
    PetscCallA(VecSetType(g_ctx%stage, VECSTANDARD, ierr))

    ! Explicit destination indices rather than relying on the NULL-means-natural
    ! ordering convention: my chunk of the staging vector, in order.
    PetscCallA(VecGetOwnershipRange(g_ctx%stage, lo, hi, ierr))
    PetscCallA(ISCreateStride(g_ctx%comm, g_ctx%n_loc, lo, one, is_stage, ierr))
    my_is = g_ctx%is_fam(g_ctx%my_fam)
    PetscCallA(VecScatterCreate(x_tmpl, my_is, g_ctx%stage, is_stage, g_ctx%scatter, ierr))
    PetscCallA(ISDestroy(is_stage, ierr))
    PetscCallA(VecDestroy(x_tmpl, ierr))

    ! Follow the family operator's type, so a device-resident operator gets
    ! device-resident vectors.
    PetscCallA(MatCreateVecs(g_ctx%A_fam, g_ctx%sol_fam, g_ctx%rhs_fam, ierr))
  end subroutine modesplit_build_vectors


  !> Create (first call) or refresh the family's direct solver.
  !!
  !! PCLU via MUMPS is established before KSPSetFromOptions for the same reason as
  !! in mod_petsc_pc_toroidal: an untyped PC would be configured from options as
  !! PETSc's own default - PCILU for a sequential operator - which then consumes
  !! pc_factor_mat_solver_type and aborts for any package that has no ILU.
  subroutine modesplit_setup_solver(A_f)
    Mat, intent(in) :: A_f

    PC :: pc_fam
    PetscErrorCode :: ierr

    if (.not. g_ctx%solver_ready) then
      PetscCallA(KSPCreate(g_ctx%comm_fam, g_ctx%ksp_fam, ierr))
      PetscCallA(KSPSetOptionsPrefix(g_ctx%ksp_fam, PC_BLOCK_PREFIX, ierr))
      PetscCallA(KSPSetOperators(g_ctx%ksp_fam, A_f, A_f, ierr))
      PetscCallA(KSPSetType(g_ctx%ksp_fam, KSPPREONLY, ierr))
      PetscCallA(KSPGetPC(g_ctx%ksp_fam, pc_fam, ierr))
      PetscCallA(PCSetType(pc_fam, PCLU, ierr))
      PetscCallA(PCFactorSetMatSolverType(pc_fam, MATSOLVERMUMPS, ierr))
      PetscCallA(KSPSetFromOptions(g_ctx%ksp_fam, ierr))
      call petsc_configure_direct_solver(pc_fam)
      ! One line for all families: they share PC_BLOCK_PREFIX, so any of them
      ! describes the others. Printed from global rank 0 only - a per-family print
      ! would be n_fam identical lines, and stdout from inside a sub-communicator
      ! is not ordered against the rest of the run.
      if (g_ctx%my_id == 0) call petsc_report_solver(g_ctx%ksp_fam, PC_BLOCK_PREFIX)
      g_ctx%solver_ready = .true.
    else
      PetscCallA(KSPSetOperators(g_ctx%ksp_fam, A_f, A_f, ierr))
    endif

    ! Factorize here, inside PCSetUp, so the cost is charged to the caller's setup
    ! log stage rather than to the first PCApply inside the GMRES solve.
    PetscCallA(KSPSetUp(g_ctx%ksp_fam, ierr))
  end subroutine modesplit_setup_solver


  !> PCSHELL apply callback: y = M^-1 x.
  !!
  !! The KSPSolve in the middle is collective on comm_fam only. The communicators
  !! are disjoint, so all families run it at the same time - which is the entire
  !! point of this preconditioner.
  subroutine modesplit_apply(pc, x, y, ierr)
    PC  :: pc
    Vec :: x, y
    PetscErrorCode :: ierr

    if (g_ctx%passthrough) then
      PetscCallA(KSPSolve(g_ctx%ksp_fam, x, y, ierr))
      ierr = 0
      return
    endif

    PetscCallA(VecScatterBegin(g_ctx%scatter, x, g_ctx%stage, INSERT_VALUES, SCATTER_FORWARD, ierr))
    PetscCallA(VecScatterEnd(g_ctx%scatter, x, g_ctx%stage, INSERT_VALUES, SCATTER_FORWARD, ierr))

    call modesplit_copy_local(g_ctx%stage, g_ctx%rhs_fam)

    PetscCallA(KSPSolve(g_ctx%ksp_fam, g_ctx%rhs_fam, g_ctx%sol_fam, ierr))

    ! Overlapping families solve for the same mode more than once; the weights are
    ! what stop it being counted twice when the contributions are summed below.
    ! Disjoint families carry weight 1 and the sum is then a plain copy. This
    ! mirrors gather_solution on the legacy path exactly.
    if (g_ctx%weighted) then
      PetscCallA(VecScale(g_ctx%sol_fam, g_ctx%weight, ierr))
    endif

    call modesplit_copy_local(g_ctx%sol_fam, g_ctx%stage)

    PetscCallA(VecSet(y, 0.0d0, ierr))
    PetscCallA(VecScatterBegin(g_ctx%scatter, g_ctx%stage, y, ADD_VALUES, SCATTER_REVERSE, ierr))
    PetscCallA(VecScatterEnd(g_ctx%scatter, g_ctx%stage, y, ADD_VALUES, SCATTER_REVERSE, ierr))

    ierr = 0
  end subroutine modesplit_apply


  !> Copy the local part of one vector into the local part of another.
  !!
  !! `from` and `to` live on different communicators but have the same local size
  !! by construction, so this is a purely local move - no MPI at all.
  !!
  !! It goes through the host arrays, which for a device-resident operator forces a
  !! device/host round trip per Krylov iteration. That is deliberate for now: it is
  !! O(n_loc) against the O(nnz) triangular solve next to it, it does not touch the
  !! factorization concurrency this PC exists for, and it keeps one code path for
  !! every vector type. Aliasing the two arrays (VecPlaceArray, as PCTelescope
  !! does) removes it on the host but needs a memtype-aware variant on the device;
  !! do that only if it shows up in a profile.
  subroutine modesplit_copy_local(from, to)
    Vec, intent(in) :: from, to

    PetscScalar, pointer :: src(:), dst(:)
    PetscErrorCode :: ierr

    PetscCallA(VecGetArrayRead(from, src, ierr))
    PetscCallA(VecGetArrayWrite(to, dst, ierr))
    dst(1:g_ctx%n_loc) = src(1:g_ctx%n_loc)
    PetscCallA(VecRestoreArrayWrite(to, dst, ierr))
    PetscCallA(VecRestoreArrayRead(from, src, ierr))
  end subroutine modesplit_copy_local


  !> PCSHELL destroy callback. Also frees the family communicator - this is the
  !! first PETSc path in JOREK that owns one, and the legacy path's habit of never
  !! releasing its communicators is not worth copying.
  subroutine modesplit_destroy(pc, ierr)
    PC :: pc
    PetscErrorCode :: ierr

    integer :: f, mpierr

    if (.not. g_ctx%initialized) then
      ierr = 0
      return
    endif

    if (g_ctx%solver_ready) then
      PetscCallA(KSPDestroy(g_ctx%ksp_fam, ierr))
      g_ctx%solver_ready = .false.
    endif

    if (.not. g_ctx%passthrough) then
      if (associated(g_ctx%is_fam)) then
        do f = 1, g_ctx%n_fam
          PetscCallA(MatDestroy(g_ctx%B_loc(f), ierr))
          PetscCallA(MatDestroy(g_ctx%B_fam(f), ierr))
          PetscCallA(ISDestroy(g_ctx%is_fam(f), ierr))
        enddo
        deallocate(g_ctx%is_fam, g_ctx%B_fam, g_ctx%B_loc)
        g_ctx%is_fam => Null(); g_ctx%B_fam => Null(); g_ctx%B_loc => Null()

        PetscCallA(MatDestroy(g_ctx%A_fam, ierr))
        PetscCallA(VecScatterDestroy(g_ctx%scatter, ierr))
        PetscCallA(VecDestroy(g_ctx%stage, ierr))
        PetscCallA(VecDestroy(g_ctx%rhs_fam, ierr))
        PetscCallA(VecDestroy(g_ctx%sol_fam, ierr))
      endif
    endif

    if (g_ctx%comm_fam /= MPI_COMM_NULL) call MPI_COMM_FREE(g_ctx%comm_fam, mpierr)
    g_ctx%comm_fam = MPI_COMM_NULL

    deallocate(g_ctx%modes_per_fam, g_ctx%ranks_per_fam, g_ctx%rank_range, &
               g_ctx%fam_modes, g_ctx%fam_ranks)
    g_ctx%modes_per_fam => Null(); g_ctx%ranks_per_fam => Null()
    g_ctx%rank_range    => Null(); g_ctx%fam_modes     => Null()
    g_ctx%fam_ranks     => Null()

    g_ctx%initialized = .false.
    ierr = 0
  end subroutine modesplit_destroy

#endif
end module mod_petsc_pc_modesplit
