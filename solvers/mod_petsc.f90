module mod_petsc
#ifdef USE_PETSC
  use mpi_mod
  use mod_petsc_pc
  use mod_petsc_direct_solver, only: petsc_configure_direct_solver
  use mod_petsc_dump,          only: petsc_dump_operator
#include "petsc/finclude/petsc.h"
  use petsc
#ifdef USE_SLEPC
#include "slepc/finclude/slepceps.h"
  use slepceps
#endif

  implicit none


  !> Run-directory file read by PetscInitialize for solver options.
  character(len=*), parameter :: PETSC_OPTIONS_FILE = 'jorek.petsc'
  !> Options prefix of the main iterative solve, e.g. -jorek_ksp_rtol 1e-9
  character(len=*), parameter :: PETSC_MAIN_PREFIX   = 'jorek_'
  !> Options prefix of the one-shot direct solve path
  character(len=*), parameter :: PETSC_DIRECT_PREFIX = 'jorek_direct_'

  !> Operator storage formats selectable with -jorek_mat_format.
  !! 'baij' is the historical path: MATMPIBAIJ filled block-by-block with
  !! MatSetValuesBlocked, then MatConvert'ed to MATMPIAIJ for the KSP on every
  !! solve. 'aij' assembles MATAIJ once through PETSc's COO interface and hands
  !! it to the KSP directly - no conversion, and the only format that can ever be
  !! device-resident (PETSc implements COO and the cusparse/kokkos matrix types
  !! for AIJ only, never for BAIJ).
  character(len=*), parameter :: PETSC_FORMAT_BAIJ = 'baij'
  character(len=*), parameter :: PETSC_FORMAT_AIJ  = 'aij'

  type type_PETSC_SYSTEM
    Mat  :: A              ! system matrix: MATMPIBAIJ, or MATAIJ when aij_native
    Mat  :: A_aij          ! AIJ version used by KSP (persistent; unused when aij_native)
    Vec  :: x, b           ! solution/RHS vectors matching A
    Vec  :: x_aij, b_aij   ! AIJ solution/RHS vectors for KSP (unused when aij_native)
    KSP  :: ksp            ! Krylov solver context (persistent)
    !> Cached gather of x onto every rank, rebuilt only when the layout changes.
    !! JOREK's sol_vec is replicated, so every solve needs this; creating and
    !! destroying it per solve allocated an n_global vector each time and, on the
    !! device path, rebuilt the scatter plan on every time step.
    Vec        :: x_seq
    VecScatter :: x_scatter
    logical    :: scatter_ready = .false.
    logical :: initialized   = .false.  ! A, x, b created
    logical :: owns_A        = .false.  ! .true. when A was created by petsc_init_system (old path)
    logical :: ksp_ready     = .false.  ! KSP, A_aij, PC setup + factored
    !> .true. when A is already the AIJ operator the KSP consumes, so A_aij /
    !! x_aij / b_aij are never created and no MatConvert or VecCopy happens.
    logical :: aij_native    = .false.
    PetscLogStage :: stage_setup = -1
    PetscLogStage :: stage_solve = -1
  end type type_PETSC_SYSTEM


contains

  !> Initialize PETSc, reading solver options from PETSC_OPTIONS_FILE in the run
  !! directory when it exists. That file is the supported way to reconfigure the
  !! solvers at run time; see namelist/jorek.petsc.example for the available
  !! prefixes. Passing a file name that does not exist is a fatal error in PETSc,
  !! hence the inquire.
  subroutine petsc_initialize()
    PetscErrorCode :: ierr
    logical :: have_opts
    ! PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-log_view", PETSC_NULL_CHARACTER, ierr))

    inquire(file=PETSC_OPTIONS_FILE, exist=have_opts)
#ifdef USE_SLEPC
    if (have_opts) then
      call SlepcInitialize(PETSC_OPTIONS_FILE, ierr)  ! superset of PetscInitialize
    else
      call SlepcInitialize(PETSC_NULL_CHARACTER, ierr)
    endif
    if (ierr /= 0) print *, "Error initializing SLEPc/PETSc"
#else
    if (have_opts) then
      call PetscInitialize(PETSC_OPTIONS_FILE, ierr)
    else
      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
    endif
    if (ierr /= 0) print *, "Error initializing PETSc"
#endif
  end subroutine


  !> Report the main solve configuration actually resolved from the defaults
  !! plus whatever jorek.petsc supplied.
  subroutine report_main_solver(ksp, my_id)
    KSP, intent(in)     :: ksp
    integer, intent(in) :: my_id

    KSPType        :: ktype
    PetscReal      :: rtol, abstol, dtol
    PetscInt       :: maxits
    PetscErrorCode :: ierr

    if (my_id /= 0) return

    PetscCallA(KSPGetType(ksp, ktype, ierr))
    PetscCallA(KSPGetTolerances(ksp, rtol, abstol, dtol, maxits, ierr))
    write(*,*) '[PETSc] main solve (-'//PETSC_MAIN_PREFIX//'...): '//trim(ktype) &
               //' + '//PCFIELDSPLIT
    write(*,*) '[PETSc]   rtol =', rtol, ' max_it =', maxits
  end subroutine report_main_solver


  subroutine petsc_finalize()
    PetscErrorCode :: ierr
#ifdef USE_SLEPC
    call SlepcFinalize(ierr)
#else
    call PetscFinalize(ierr)
#endif
  end subroutine petsc_finalize


  subroutine petsc_print_version()
    PetscErrorCode :: ierr
    integer :: my_id, mpierr
    character(len=256) :: version_string

    call PetscGetVersion(version_string, ierr)
    if (ierr == 0) then
      call MPI_COMM_RANK(MPI_COMM_WORLD, my_id, mpierr)
      if (my_id == 0) then
        print *, "----------------------------------------"
        print *, "JOREK linked with ", trim(version_string)
        print *, "----------------------------------------"
      end if
    end if
  end subroutine petsc_print_version


  !> Operator format requested with -jorek_mat_format {baij|aij}, defaulting to
  !! baij so an unconfigured run behaves exactly as before.
  !!
  !! Resolved from the options database once and cached: this is queried from
  !! construct_matrix on every time step and from the solver on every setup, and
  !! the two MUST agree - construct_matrix leaves petsc_assembled .false. for aij
  !! precisely so that the solver takes the COO path. Reading the database twice
  !! per step was both wasteful and one PetscOptionsClear away from a mismatch.
  function petsc_mat_format() result(fmt)
    character(len=8) :: fmt
    PetscBool        :: found
    PetscErrorCode   :: ierr

    character(len=8), save :: fmt_cached = ''
    logical,          save :: resolved   = .false.

    if (.not. resolved) then
      fmt_cached = PETSC_FORMAT_BAIJ
      ! NOTE: PetscCallA is a cpp macro, so its argument must stay on one line - a
      ! Fortran '&' continuation splits it before cpp ever sees the second half.
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-'//PETSC_MAIN_PREFIX//'mat_format', fmt_cached, found, ierr))
      resolved = .true.
    endif

    fmt = fmt_cached
  end function petsc_mat_format


  !> .true. when the operator is assembled as MATAIJ through the COO interface.
  !! Thin wrapper so callers stop repeating trim()/comparison against the string.
  logical function petsc_format_is_aij()
    petsc_format_is_aij = (trim(petsc_mat_format()) == PETSC_FORMAT_AIJ)
  end function petsc_format_is_aij


  !> Generate the 0-based COO index pair for every reserved matrix slot straight
  !! from the block sparsity (ijA_size / irn_jcn / ijA_index), without reading
  !! a_mat%irn or a_mat%jcn at all. That is the point: it lets the AIJ path skip
  !! allocating and filling those two nnz-length arrays entirely.
  !!
  !! It mirrors, entry for entry, what add_block_to_sp_matrix writes into irn/jcn
  !! (construct_matrix_mod.f90): the block living at ijA_position carries, at
  !! offset (jj-1)*bs + ll, row index_large_i + jj and column index_large_k + ll,
  !! where index_large_* = bs*(block index - 1). The 0-based shift is the only
  !! difference. petsc_verify_coo_indices checks that claim against the real
  !! arrays; keep the two in step if either side ever changes.
  !!
  !! One deliberate behaviour change: this also covers slots the element loop
  !! never writes. global_matrix_structure reserves a bs^2 block for every
  !! (row, col) pair it finds and a few are never touched; they now get their
  !! true (i,j) and become explicit zeros in the operator. Previously such a slot
  !! arrived here as index 0, turned into -1 under the shift, and was silently
  !! dropped by PETSc - correct, but resting on a sentinel that had to be
  !! explained every time it was read.
  subroutine build_coo_indices(a_mat, coo_i, coo_j)
    use data_structure, only: type_SP_MATRIX
    use mod_integer_types, only: int_all

    type(type_SP_MATRIX), intent(in) :: a_mat
    PetscInt, intent(out) :: coo_i(:), coo_j(:)

    integer :: i_local, j, jj, ll, bs, n_block_local
    integer(kind=int_all) :: pos, base_i, base_k, off

    bs            = a_mat%block_size
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1

    !$omp parallel do default(none) schedule(static) &
    !$omp   shared(a_mat, coo_i, coo_j, bs, n_block_local) &
    !$omp   private(i_local, j, jj, ll, pos, base_i, base_k, off)
    do i_local = 1, n_block_local
      base_i = int(bs, int_all) * int(a_mat%my_ind_min + i_local - 2, int_all)
      do j = 1, a_mat%ijA_size(i_local)
        base_k = int(bs, int_all) * (a_mat%irn_jcn(i_local, j) - 1)
        pos    = a_mat%ijA_index(i_local, j)
        do jj = 1, bs
          off = pos - 1 + int(jj - 1, int_all) * int(bs, int_all)
          do ll = 1, bs
            coo_i(off + ll) = int(base_i + jj - 1, kind=kind(coo_i))
            coo_j(off + ll) = int(base_k + ll - 1, kind=kind(coo_j))
          enddo
        enddo
      enddo
    enddo
    !$omp end parallel do
  end subroutine build_coo_indices


  !> .true. when -jorek_coo_verify_indices asked for the generated COO indices to
  !! be checked against irn/jcn. Read by prepare_matrix_storage as well, because
  !! the check needs those arrays to still exist: the flag has to keep them alive
  !! on a path whose whole point is not allocating them. One switch, both effects.
  logical function petsc_coo_verify_requested()
    PetscBool      :: do_check
    PetscErrorCode :: ierr

    logical, save :: cached   = .false.
    logical, save :: resolved = .false.

    if (.not. resolved) then
      do_check = PETSC_FALSE
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-'//PETSC_MAIN_PREFIX//'coo_verify_indices', do_check, PETSC_NULL_BOOL, ierr))
      cached   = (do_check .eqv. PETSC_TRUE)
      resolved = .true.
    endif

    petsc_coo_verify_requested = cached
  end function petsc_coo_verify_requested


  !> Cross-check build_coo_indices against the irn/jcn the element loop filled.
  !! Opt-in with -jorek_coo_verify_indices; aborts on the first disagreement.
  !!
  !! This exists because build_coo_indices duplicates an index convention that
  !! lives in construct_matrix_mod, and nothing else would catch the two drifting
  !! apart: a wrong index produces a well-formed matrix with entries in the wrong
  !! places, which shows up as a convergence change rather than an error. Slots
  !! the element loop never wrote are skipped (irn == 0); see build_coo_indices.
  subroutine petsc_verify_coo_indices(a_mat, coo_i, coo_j)
    use data_structure, only: type_SP_MATRIX
    use mod_integer_types, only: int_all

    type(type_SP_MATRIX), intent(in) :: a_mat
    PetscInt, intent(in) :: coo_i(:), coo_j(:)

    integer(kind=int_all) :: k, n_bad, n_skipped
    integer :: my_id, mpierr
    PetscErrorCode :: ierr

    if (.not. petsc_coo_verify_requested()) return

    if (.not. associated(a_mat%irn) .or. .not. associated(a_mat%jcn)) then
      SETERRA(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, '-jorek_coo_verify_indices needs irn/jcn, which this run did not keep')
    endif

    call MPI_COMM_RANK(a_mat%comm, my_id, mpierr)
    n_bad     = 0
    n_skipped = 0
    do k = 1, a_mat%nnz
      if (a_mat%irn(k) == 0) then
        n_skipped = n_skipped + 1
        cycle
      endif
      if (coo_i(k) /= int(a_mat%irn(k) - 1, kind=kind(coo_i)) .or. &
          coo_j(k) /= int(a_mat%jcn(k) - 1, kind=kind(coo_j))) then
        if (n_bad < 10) write(*,'(A,I0,A,I0,A,2I12,A,2I12)') "[RANK ", my_id, &
          "] COO index mismatch at ", k, ": generated", coo_i(k), coo_j(k), " expected", a_mat%irn(k)-1, a_mat%jcn(k)-1
        n_bad = n_bad + 1
      endif
    enddo

    write(*,'(A,I0,A,I0,A,I0,A,I0)') "[RANK ", my_id, "] COO index check: ", a_mat%nnz, &
      " slots, mismatches=", n_bad, ", unwritten (skipped)=", n_skipped
    if (n_bad > 0) then
      SETERRA(PETSC_COMM_SELF, PETSC_ERR_PLIB, 'generated COO indices disagree with irn/jcn')
    endif
  end subroutine petsc_verify_coo_indices


  !> Initialize a MATAIJ operator whose values are pushed with PETSc's COO
  !! interface.
  !!
  !! JOREK's a_mat is already a duplicate-free COO triplet set: mod_global_matrix_structure
  !! builds irn_jcn as a sorted, deduplicated block-column list per block row (it
  !! explicitly rejects a repeated index) and ijA_index gives each block a contiguous
  !! slice of val, so every scalar entry has exactly one slot. That means the COO
  !! index pairs follow directly from the block structure (build_coo_indices) and the
  !! per-timestep value array IS a_mat%val - no expansion and no re-ordering, because
  !! COO is per scalar entry.
  !!
  !! It also means boundary conditions need no special handling here: the ZBIG penalty
  !! writes (mod_assembly.f90, mod_axis_treatment.f90, mod_fix_axis_nodes.f90) and the
  !! vacuum response contributions are all resolved into a_mat%val on the JOREK side
  !! before this routine ever sees them, so a single INSERT_VALUES push is correct.
  !!
  !! Ranks own disjoint block-row ranges [my_ind_min, my_ind_max] and store full rows,
  !! so no (i,j) is submitted by two ranks. This matters: PETSc SUMS duplicate COO
  !! entries across ranks regardless of the InsertMode.
  subroutine petsc_init_system_coo(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX
    use tr_module,      only: tr_deallocatep, CAT_DMATRIX
    use phys_module,    only: use_matrix_equilibration

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(inout) :: a_mat

    integer :: comm, my_id, mpierr
    integer :: block_size
    PetscInt :: n_local, n_global
    PetscInt, allocatable :: coo_i(:), coo_j(:)
    PetscCount :: ncoo
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size = a_mat%block_size
    n_global   = int(a_mat%ng, kind=kind(n_global))
    n_local    = int((a_mat%my_ind_max - a_mat%my_ind_min + 1) * block_size, kind=kind(n_local))

    call MatCreate(comm, petsc_sys%A, ierr)
    call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_sys%A, MATAIJ, ierr)
    ! Keep the block size as metadata: PCFIELDSPLIT still slices the operator by
    ! toroidal mode family and MatSetBlockSize is what makes that indexing valid.
    call MatSetBlockSize(petsc_sys%A, block_size, ierr)
    ! The prefix is what makes -jorek_mat_type reach this matrix; without it the key
    ! would have to be the un-prefixed -mat_type and would hit every other Mat too.
    call MatSetOptionsPrefix(petsc_sys%A, PETSC_MAIN_PREFIX, ierr)
    ! Honours -jorek_mat_type aijcusparse (or aijkokkos) without another switch here.
    call MatSetFromOptions(petsc_sys%A, ierr)

    ! PETSc copies the index arrays, so these are scratch. They are PetscInt (32-bit
    ! under the cuDSS build) while JOREK's are int_all, hence the conversion inside
    ! build_coo_indices.
    ncoo = int(a_mat%nnz, kind=kind(ncoo))

    ! Indices come from the block sparsity, not from irn/jcn, so this routine no
    ! longer needs those arrays to exist - which is what lets construct_matrix skip
    ! allocating them entirely. Peak here is val + coo_i + coo_j (16 B/nnz) plus
    ! PETSc's own map, rather than the 24 B/nnz of the previous copy-then-free
    ! dance, so the preallocation moment is no longer the high-water mark of the
    ! run.
    allocate(coo_i(a_mat%nnz), coo_j(a_mat%nnz))
    call build_coo_indices(a_mat, coo_i, coo_j)
    call petsc_verify_coo_indices(a_mat, coo_i, coo_j)

    ! Whatever irn/jcn still exist have no reader left on this path once PETSc owns
    ! the sparsity - except under equilibration, where matrix_equilibration and
    ! scale_by_cols index through jcn on every solve.
    if (associated(a_mat%irn)) call tr_deallocatep(a_mat%irn, "irn", CAT_DMATRIX)
    if (associated(a_mat%jcn) .and. (.not. use_matrix_equilibration)) then
      call tr_deallocatep(a_mat%jcn, "jcn", CAT_DMATRIX)
    endif

    PetscCallA(MatSetPreallocationCOO(petsc_sys%A, ncoo, coo_i, coo_j, ierr))
    deallocate(coo_i, coo_j)

    ! PETSc now owns the sparsity: from here on construct_matrix only refreshes val,
    ! and every routine that used to write irn/jcn checks this flag first.
    a_mat%coo_structure_fixed = .true.

    call MatCreateVecs(petsc_sys%A, petsc_sys%x, petsc_sys%b, ierr)

    petsc_sys%initialized = .true.
    petsc_sys%owns_A      = .true.
    petsc_sys%aij_native  = .true.
    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0,A,I0)') "[PETSc] init: AIJ/COO matrix ", n_global, "x", n_global, &
                                                       ", block_size=", block_size, ", local nnz=", a_mat%nnz
  end subroutine petsc_init_system_coo


  !> Push JOREK's value array into the COO-preallocated operator. This replaces the
  !! whole per-block MatSetValuesBlocked loop of petsc_update_matrix with one call.
  !!
  !! a_mat%val is passed as a single contiguous array on purpose, and must stay that
  !! way: MatSetValuesCOO_SeqAIJCUSPARSE memtype-detects its value pointer
  !! (aijcusparse.cu:4422) and consumes device memory directly, so moving JOREK's
  !! assembly onto the GPU later requires no change on this side - only to where
  !! a_mat%val is allocated and written.
  subroutine petsc_update_matrix_coo(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(in) :: a_mat
    PetscErrorCode :: ierr

    ! The COO map PETSc holds describes the sparsity as it was at preallocation. If
    ! global_matrix_structure has rebuilt it since (it clears this flag), val no
    ! longer matches that map and pushing it would silently produce a wrong matrix
    ! rather than fail - so refuse instead. Re-meshing on this path would need the
    ! operator torn down and petsc_init_system_coo re-run.
    if (.not. a_mat%coo_structure_fixed) then
      SETERRA(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, 'matrix structure was rebuilt after the COO preallocation')
    endif

    ! INSERT_VALUES, not ADD_VALUES: a_mat holds one slot per (i,j) with all element,
    ! boundary-condition and vacuum contributions already accumulated into it.
    PetscCallA(MatSetValuesCOO(petsc_sys%A, a_mat%val, INSERT_VALUES, ierr))
  end subroutine petsc_update_matrix_coo


  !> Initialize BAIJ matrix structure and BAIJ vecs — called once when !initialized
  !TODO: Redundant with petsc_create_matrix. One must go...
  subroutine petsc_init_system(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(inout) :: a_mat

    integer :: i, k
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, block_size2, row_start_idx, row_end_idx
    integer :: r_start, r_end, c_global, block_col
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscErrorCode :: ierr

    if (petsc_format_is_aij()) then
      call petsc_init_system_coo(petsc_sys, a_mat)
      return
    endif

    ! REACHABILITY ASSERTION (see the TODO above). construct_matrix sets
    ! petsc_assembled for every non-harmonic BAIJ matrix, and mod_sparse only calls
    ! this routine when that flag is clear - so on the BAIJ path this point should
    ! be unreachable and the body below is dead. petsc_create_matrix +
    ! add_block_to_petsc is the live BAIJ path. Assert rather than assume; if no
    ! run trips this, the remainder of the routine can be deleted outright.
    SETERRA(PETSC_COMM_SELF, PETSC_ERR_SUP, 'petsc_init_system: BAIJ body believed dead - please report the case that reached it')

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size = a_mat%block_size
    block_size2 = block_size * block_size
    n_global = a_mat%ng
    n_local = (a_mat%my_ind_max - a_mat%my_ind_min + 1) * block_size
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    row_start_idx = (a_mat%my_ind_min - 1)*block_size + 1
    row_end_idx = a_mat%my_ind_max*block_size

    if ((row_end_idx - row_start_idx + 1) /= n_local) &
      write(*,*) "[RANK ", my_id, "] WARNING: Something is wrong in petsc_init_system!"

    ! Create matrix
    call MatCreate(comm, petsc_sys%A, ierr)
    call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_sys%A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_sys%A, block_size, ierr)

    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0
    o_nnz = 0
    do i = 1, n_block_local
      r_start = a_mat%iblockptr(i)
      r_end = a_mat%iblockptr(i+1) - 1
      do k = r_start, r_end
        c_global = a_mat%jcn((k-1)*block_size2 + 1)
        block_col = (c_global / block_size) + 1
        if (block_col >= a_mat%my_ind_min .and. block_col <= a_mat%my_ind_max) then
          d_nnz(i) = d_nnz(i) + 1
        else
          o_nnz(i) = o_nnz(i) + 1
        endif
      enddo
    enddo

    call MatMPIBAIJSetPreallocation(petsc_sys%A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: MatMPIBAIJSetPreallocation ierr=", ierr
    deallocate(d_nnz, o_nnz)

    call MatCreateVecs(petsc_sys%A, petsc_sys%x, petsc_sys%b, ierr)

    petsc_sys%initialized = .true.
    petsc_sys%owns_A = .true.
    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0)') "[PETSc] init: BAIJ matrix ", n_global, "x", n_global, &
                                                    ", block_size=", block_size
  end subroutine petsc_init_system


  !> Create and preallocate a PETSc MPIBAIJ matrix from the JOREK block structure
  !! (ijA_size, irn_jcn). Does not require iblockptr (block-CSR).
  subroutine petsc_create_matrix(petsc_A, a_mat)
    use data_structure, only: type_SP_MATRIX

    Mat, intent(out)                    :: petsc_A
    type(type_SP_MATRIX), intent(in)    :: a_mat

    integer :: i, j
    integer :: comm, my_id, mpierr
    integer :: n_local, n_global, n_block_local, block_size, col_block
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)

    block_size    = a_mat%block_size
    n_global      = a_mat%ng
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1
    n_local       = n_block_local * block_size

    ! Compute diagonal/off-diagonal block counts per block row
    allocate(d_nnz(n_block_local), o_nnz(n_block_local))
    d_nnz = 0
    o_nnz = 0
    do i = 1, n_block_local
      do j = 1, a_mat%ijA_size(i)
        col_block = a_mat%irn_jcn(i, j)
        if (col_block >= a_mat%my_ind_min .and. col_block <= a_mat%my_ind_max) then
          d_nnz(i) = d_nnz(i) + 1
        else
          o_nnz(i) = o_nnz(i) + 1
        endif
      enddo
    enddo

    ! Create and preallocate
    call MatCreate(comm, petsc_A, ierr)
    call MatSetSizes(petsc_A, n_local, n_local, n_global, n_global, ierr)
    call MatSetType(petsc_A, MATMPIBAIJ, ierr)
    call MatSetBlockSize(petsc_A, block_size, ierr)
    call MatMPIBAIJSetPreallocation(petsc_A, block_size, 0, d_nnz, 0, o_nnz, ierr)
    if (ierr /= 0) write(*,*) "[RANK ", my_id, "] WARNING: petsc_create_matrix preallocation ierr=", ierr
    deallocate(d_nnz, o_nnz)

    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0)') &
      "[PETSc] create_matrix: BAIJ ", n_global, "x", n_global, ", block_size=", block_size
  end subroutine petsc_create_matrix


  !> Fill matrix values from JOREK block-CSR
  subroutine petsc_update_matrix(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(in) :: a_mat

    integer :: i, k
    integer :: my_id, mpierr
    integer :: n_block_local, block_size, block_size2
    integer :: r_start, r_end, c_global, val_ptr_start, val_ptr_end
    PetscInt :: idxm(1), idxn(1)
    PetscScalar, allocatable :: vals_petsc(:)
    PetscErrorCode :: ierr

    if (petsc_sys%aij_native) then
      call petsc_update_matrix_coo(petsc_sys, a_mat)
      return
    endif

    ! Same reachability assertion as petsc_init_system: this BAIJ fill from JOREK's
    ! block-CSR is believed dead, superseded by add_block_to_petsc.
    SETERRA(PETSC_COMM_SELF, PETSC_ERR_SUP, 'petsc_update_matrix: BAIJ body believed dead - please report the case that reached it')

    call MPI_COMM_RANK(a_mat%comm, my_id, mpierr)

    block_size = a_mat%block_size
    block_size2 = block_size * block_size
    n_block_local = a_mat%my_ind_max - a_mat%my_ind_min + 1

    call MatZeroEntries(petsc_sys%A, ierr)

    allocate(vals_petsc(block_size2))
    do i = 1, n_block_local
      idxm(1) = (a_mat%my_ind_min - 1) + (i - 1)
      r_start = a_mat%iblockptr(i)
      r_end = a_mat%iblockptr(i+1) - 1
      do k = r_start, r_end
        c_global = a_mat%jcn((k-1)*block_size2 + 1)
        idxn(1) = c_global / block_size
        val_ptr_start = (k - 1) * block_size2 + 1
        val_ptr_end   = val_ptr_start + block_size2 - 1
        vals_petsc(1:block_size2) = a_mat%val(val_ptr_start : val_ptr_end)
        PetscCallA(MatSetValuesBlocked(petsc_sys%A, 1, idxm, 1, idxn, vals_petsc, INSERT_VALUES, ierr))
      enddo
    enddo

    PetscCallA(MatAssemblyBegin(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr))
    deallocate(vals_petsc)
  end subroutine petsc_update_matrix


  !> Fill RHS vector from JOREK rhs — called every time
  subroutine petsc_update_rhs(petsc_sys, rhs_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(in) :: rhs_vec

    PetscInt :: i_start, i_end, n_local
    PetscScalar, pointer :: b_arr(:)
    PetscErrorCode :: ierr

    ! rhs_vec%val is the full global rhs on every rank, so the owned slice can be
    ! written straight into the Vec's local array - no index list, no assembly pass.
    PetscCallA(VecGetOwnershipRange(petsc_sys%b, i_start, i_end, ierr))
    n_local = i_end - i_start

    ! Write access, not read-write: the slice is overwritten in full, so
    ! VecGetArray's device->host sync of the previous contents is pure waste on a
    ! device Vec (and the dirty flag it leaves forces a copy back up either way).
    PetscCallA(VecGetArrayWrite(petsc_sys%b, b_arr, ierr))
    b_arr(1:n_local) = rhs_vec%val(i_start+1:i_end)
    PetscCallA(VecRestoreArrayWrite(petsc_sys%b, b_arr, ierr))

  end subroutine petsc_update_rhs


  !> Load an initial guess into petsc_sys%x from the JOREK sol_vec (the previous
  !! time-step increment). Mirrors petsc_update_rhs ownership slicing so that x
  !! and b share the same global layout. If sol_vec is not yet allocated (e.g. the
  !! first step of a fresh, non-restart run) the guess degrades to zero, reproducing
  !! the standard zero-start. Used together with KSPSetInitialGuessNonzero.
  subroutine petsc_update_initial_guess(petsc_sys, sol_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(in) :: sol_vec

    PetscInt :: i_start, i_end, n_local
    PetscScalar, pointer :: x_arr(:)
    PetscErrorCode :: ierr

    if (.not. associated(sol_vec%val)) then
      PetscCallA(VecSet(petsc_sys%x, 0.0d0, ierr))
      return
    endif

    ! Same ownership slicing as petsc_update_rhs, written directly into the local array.
    PetscCallA(VecGetOwnershipRange(petsc_sys%x, i_start, i_end, ierr))
    n_local = i_end - i_start

    ! Write access for the same reason as petsc_update_rhs.
    PetscCallA(VecGetArrayWrite(petsc_sys%x, x_arr, ierr))
    x_arr(1:n_local) = sol_vec%val(i_start+1:i_end)
    PetscCallA(VecRestoreArrayWrite(petsc_sys%x, x_arr, ierr))

  end subroutine petsc_update_initial_guess


  subroutine petsc_solve_and_retrieve(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys

    PetscErrorCode :: ierr
    integer :: comm, my_id, mpierr
    PetscLogDouble :: t1, t2
    PetscReal :: petsc_norm
    PC :: pc ! Maybe should be part of petsc_sys in the future
    PetscViewerAndFormat :: vf
    KSPConvergedReason :: reason
    KSPType :: ksp_type

    PetscCallA(PetscObjectGetComm(petsc_sys%A, comm, ierr))
    call MPI_COMM_RANK(comm, my_id, mpierr)

    PetscCallA(KSPCreate(comm, petsc_sys%ksp, ierr))
    PetscCallA(KSPSetOptionsPrefix(petsc_sys%ksp, PETSC_DIRECT_PREFIX, ierr))
    PetscCallA(KSPSetOperators(petsc_sys%ksp, petsc_sys%A, petsc_sys%A, ierr))

    PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf, ierr))
    PetscCallA(KSPMonitorSet(petsc_sys%ksp, KSPMonitorResidual, vf, PetscViewerAndFormatDestroy, ierr))

    PetscCallA(KSPSetType(petsc_sys%ksp, KSPPREONLY, ierr))

    ! Set the preconditioner: direct solve, LU via MUMPS unless overridden by
    ! -jorek_direct_pc_* options. PCLU is set before KSPSetFromOptions because an
    ! untyped PC would otherwise be configured from options as PETSc's default
    ! (PCILU for a sequential operator), which consumes the factor-package option
    ! and aborts for any package without an ILU. Same reason as the block PC in
    ! mod_petsc_pc_toroidal.
    PetscCallA(KSPGetPC(petsc_sys%ksp, pc, ierr))
    PetscCallA(PCSetType(pc, PCLU, ierr))
    PetscCallA(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr))
    PetscCallA(KSPSetFromOptions(petsc_sys%ksp, ierr))
    call petsc_configure_direct_solver(pc)

    PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))
    petsc_sys%ksp_ready = .true.

    PetscCallA(KSPGetType(petsc_sys%ksp, ksp_type, ierr))
    if (my_id == 0) print *, "KSP type:", ksp_type

    if (my_id .eq. 0) print *, "Solving the system using PETSc"
    PetscCallA(KSPSolve(petsc_sys%ksp, petsc_sys%b, petsc_sys%x, ierr))
    PetscCallA(KSPDestroy(petsc_sys%ksp, ierr))
    petsc_sys%ksp_ready = .false.

    ! Calculate the norm of the solution
    PetscCallA(VecNorm(petsc_sys%x, NORM_2, petsc_norm, ierr))
    if (my_id .eq.0) print *, "PETSc Norm (solution): ", petsc_norm
  end subroutine petsc_solve_and_retrieve


  !> The operator and vectors the KSP actually works on. On the BAIJ path these are
  !! the converted AIJ duplicates; on the native AIJ path they are simply A, b and x,
  !! so no conversion or vector copy is needed anywhere in the solve.
  subroutine ksp_operands(petsc_sys, A_ksp, b_ksp, x_ksp)
    type(type_PETSC_SYSTEM), intent(in) :: petsc_sys
    Mat, intent(out) :: A_ksp
    Vec, intent(out) :: b_ksp, x_ksp

    if (petsc_sys%aij_native) then
      A_ksp = petsc_sys%A
      b_ksp = petsc_sys%b
      x_ksp = petsc_sys%x
    else
      A_ksp = petsc_sys%A_aij
      b_ksp = petsc_sys%b_aij
      x_ksp = petsc_sys%x_aij
    endif
  end subroutine ksp_operands


  !> Make the KSP operator reflect the values just assembled into petsc_sys%A.
  !!
  !! On the native AIJ path this is nothing at all: A already IS the KSP operator,
  !! so MatSetValuesCOO has updated it in place. On the BAIJ path it is a full
  !! host-side rebuild of the operator as MATMPIAIJ - measured at 35% of that
  !! path's total runtime - reusing the sparsity after the first call.
  subroutine petsc_refresh_ksp_operator(petsc_sys, first_call)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    logical, intent(in) :: first_call
    PetscErrorCode :: ierr

    if (petsc_sys%aij_native) return

    if (first_call) then
      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_INITIAL_MATRIX, petsc_sys%A_aij, ierr))
      PetscCallA(MatCreateVecs(petsc_sys%A_aij, petsc_sys%x_aij, petsc_sys%b_aij, ierr))
    else
      PetscCallA(MatConvert(petsc_sys%A, MATMPIAIJ, MAT_REUSE_MATRIX, petsc_sys%A_aij, ierr))
    endif
  end subroutine petsc_refresh_ksp_operator


  !> Move the rhs and the warm-start guess into the vectors the KSP will use, and
  !! (direction = .false.) the solution back out. Both are no-ops on the native AIJ
  !! path, where b_aij/x_aij are b/x. See ksp_operands.
  subroutine petsc_sync_ksp_vectors(petsc_sys, to_ksp)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    logical, intent(in) :: to_ksp
    PetscErrorCode :: ierr

    if (petsc_sys%aij_native) return

    if (to_ksp) then
      PetscCallA(VecCopy(petsc_sys%b, petsc_sys%b_aij, ierr))
      PetscCallA(VecCopy(petsc_sys%x, petsc_sys%x_aij, ierr))   ! initial guess for GMRES
    else
      PetscCallA(VecCopy(petsc_sys%x_aij, petsc_sys%x, ierr))
    endif
  end subroutine petsc_sync_ksp_vectors


  !> Iterative solve with persistent KSP/PC across time steps.
  !! On first call (!ksp_ready): creates the KSP and sets up PCFIELDSPLIT+MUMPS.
  !! When !solve_only: calls KSPSetUp to refactorize.
  !! When solve_only:  sets KSPSetReusePreconditioner to skip refactorization.
  !! On the BAIJ path each of the three additionally converts A to AIJ (reusing the
  !! sparsity after the first call); on the native AIJ path A is already the KSP
  !! operator and no conversion happens at all. See ksp_operands.
  subroutine petsc_solve_iterative_and_retrieve(petsc_sys, solve_only, n_iter, converged)
    use mod_clock, only: FMT_TIMING
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    logical, intent(in) :: solve_only
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged

    PetscErrorCode :: ierr
    integer :: comm, my_id, mpierr
    Mat :: A_ksp
    Vec :: b_ksp, x_ksp
    KSPConvergedReason :: reason
    PetscLogDouble :: t1, t2
    PetscLogDouble :: ts1, ts2
    KSPType :: ksp_type
    PetscInt :: its
    PetscReal :: petsc_norm
    PetscViewerAndFormat :: vf

    call PetscObjectGetComm(petsc_sys%A, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)

    if (.not. petsc_sys%ksp_ready) then
      ! First solve: create AIJ matrix, vecs, KSP, and set up PCFIELDSPLIT+MUMPS
      PetscCallA(PetscLogStageRegister("KSP Setup", petsc_sys%stage_setup, ierr))
      PetscCallA(PetscLogStageRegister("KSP Solve", petsc_sys%stage_solve, ierr))
      PetscCallA(PetscLogStagePush(petsc_sys%stage_setup, ierr))
      PetscCallA(PetscTime(ts1, ierr))

      call petsc_refresh_ksp_operator(petsc_sys, first_call=.true.)
      call ksp_operands(petsc_sys, A_ksp, b_ksp, x_ksp)
      ! Opt-in, inert unless -jorek_dump_mat is set.
      call petsc_dump_operator(A_ksp, 'system')

      PetscCallA(KSPCreate(comm, petsc_sys%ksp, ierr))
      PetscCallA(KSPSetOptionsPrefix(petsc_sys%ksp, PETSC_MAIN_PREFIX, ierr))
      PetscCallA(KSPSetOperators(petsc_sys%ksp, A_ksp, A_ksp, ierr))
      PetscCallA(KSPSetType(petsc_sys%ksp, KSPGMRES, ierr))

      ! Warm-start: consume the JOREK initial guess (previous-step increment) loaded into
      ! petsc_sys%x before each KSPSolve. The convergence reference stays the rhs norm
      ! ||b|| (PETSc default; KSPConvergedDefaultSetUIRNorm is NOT set), so the stopping
      ! criterion is identical to the zero-start behaviour - only the starting point changes.
      PetscCallA(KSPSetInitialGuessNonzero(petsc_sys%ksp, PETSC_TRUE, ierr))

      ! Set GMRES parameters
      PetscCallA(KSPSetTolerances(petsc_sys%ksp, 1.d-8, 1.d-36, PETSC_CURRENT_REAL, 400, ierr))
      PetscCallA(KSPGMRESSetRestart(petsc_sys%ksp, 40, ierr))
      PetscCallA(KSPGMRESSetOrthogonalization(petsc_sys%ksp, KSPGMRESClassicalGramSchmidtOrthogonalization, ierr))
      PetscCallA(KSPGMRESSetCGSRefinementType(petsc_sys%ksp, KSP_GMRES_CGS_REFINE_IFNEEDED, ierr))

      PetscCallA(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, vf, ierr))
      PetscCallA(KSPMonitorSet(petsc_sys%ksp, KSPMonitorResidual, vf, PetscViewerAndFormatDestroy, ierr))

      ! Options after the defaults above, so anything in jorek.petsc overrides
      ! them: -jorek_ksp_rtol, -jorek_ksp_max_it, -jorek_ksp_gmres_restart, ...
      ! The outer PC type is deliberately NOT a knob - PCFIELDSPLIT carries the
      ! toroidal mode-family decomposition and is structural, so petsc_setup_pc
      ! below always sets it. The blocks inside it are fully configurable under
      ! the -jorek_pcblock_ prefix, and report themselves from there.
      PetscCallA(KSPSetFromOptions(petsc_sys%ksp, ierr))
      call report_main_solver(petsc_sys%ksp, my_id)
      call petsc_setup_pc(petsc_sys%ksp, petsc_sys%A, PETSC_PC_TOROIDAL_HARMONIC)

      PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))
      ! Force the sub-KSP factorizations here so they are timed as setup, not solve
      ! (no-op for the toroidal PC, which sets its sub-KSPs up explicitly).
      PetscCallA(KSPSetUpOnBlocks(petsc_sys%ksp, ierr))
      ! KSPSetUpOnBlocks stops at a PCTELESCOPE, which implements no setuponblocks,
      ! so a telescoped block solver is factorized here rather than in the solve.
      call petsc_setup_pc_blocks(petsc_sys%ksp)
      petsc_sys%ksp_ready = .true.

      PetscCallA(PetscTime(ts2, ierr))
      PetscCallA(PetscLogStagePop(ierr))
      if (my_id == 0) write(*,FMT_TIMING) my_id, '[PETSc] Elapsed time in solver setup :', ts2-ts1

    else if (.not. solve_only) then
      if (my_id .eq. 0) write(*,*) "[PETSc] PC rebuild: refactorizing"
      PetscCallA(PetscLogStagePush(petsc_sys%stage_setup, ierr))
      PetscCallA(PetscTime(ts1, ierr))

      call petsc_refresh_ksp_operator(petsc_sys, first_call=.false.)
      call ksp_operands(petsc_sys, A_ksp, b_ksp, x_ksp)
      PetscCallA(KSPSetOperators(petsc_sys%ksp, A_ksp, A_ksp, ierr))
      PetscCallA(KSPSetReusePreconditioner(petsc_sys%ksp, PETSC_FALSE, ierr))
      PetscCallA(KSPSetUp(petsc_sys%ksp, ierr))
      ! KSPSetUp only rebuilds PCFIELDSPLIT itself; the sub-KSP LU/MUMPS numerical
      ! factorizations are otherwise deferred to KSPSetUpOnBlocks inside KSPSolve, which
      ! would charge the whole factorization cost to the GMRES solve timer below.
      PetscCallA(KSPSetUpOnBlocks(petsc_sys%ksp, ierr))
      ! ... and KSPSetUpOnBlocks in turn stops at a PCTELESCOPE, which implements no
      ! setuponblocks, so the block solver it wraps has to be reached separately.
      call petsc_setup_pc_blocks(petsc_sys%ksp)

      PetscCallA(PetscTime(ts2, ierr))
      PetscCallA(PetscLogStagePop(ierr))
      if (my_id == 0) write(*,FMT_TIMING) my_id, '[PETSc] Elapsed time in solver setup :', ts2-ts1

    else
      ! solve_only: update A for mat-vec products but reuse PC factorization
      if (my_id .eq. 0) write(*,*) "[PETSc] PC reuse: solve_only, skipping refactorization"
      call petsc_refresh_ksp_operator(petsc_sys, first_call=.false.)
      call ksp_operands(petsc_sys, A_ksp, b_ksp, x_ksp)
      PetscCallA(KSPSetOperators(petsc_sys%ksp, A_ksp, A_ksp, ierr))
      PetscCallA(KSPSetReusePreconditioner(petsc_sys%ksp, PETSC_TRUE, ierr))
    end if

    call petsc_sync_ksp_vectors(petsc_sys, to_ksp=.true.)

    PetscCallA(PetscTime(t1, ierr))
    PetscCallA(PetscLogStagePush(petsc_sys%stage_solve, ierr))
    PetscCallA(KSPSolve(petsc_sys%ksp, b_ksp, x_ksp, ierr))
    PetscCallA(PetscLogStagePop(ierr))
    PetscCallA(PetscTime(t2, ierr))

    call petsc_sync_ksp_vectors(petsc_sys, to_ksp=.false.)

    PetscCallA(KSPGetConvergedReason(petsc_sys%ksp, reason, ierr))
    PetscCallA(KSPGetIterationNumber(petsc_sys%ksp, its, ierr))
    n_iter    = its
    converged = (reason%v > 0)

    if (my_id == 0) write(*,FMT_TIMING) my_id, '[PETSc] Elapsed time in solve :', t2-t1

    ! Calculate the norm of the solution
    PetscCallA(VecNorm(petsc_sys%x, NORM_2, petsc_norm, ierr))
    if (my_id .eq.0) write(*,'(A,ES12.4)') "[PETSc] solution norm: ", petsc_norm
  end subroutine petsc_solve_iterative_and_retrieve


  subroutine petsc_recover_solution(petsc_sys, sol_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(inout) :: sol_vec

    PetscScalar, pointer :: x_arr(:)
    PetscErrorCode :: ierr

    ! Built once and kept: the layout of x does not change between solves, so the
    ! scatter plan and its sequential target are reusable. petsc_cleanup destroys
    ! them.
    if (.not. petsc_sys%scatter_ready) then
      PetscCallA(VecScatterCreateToAll(petsc_sys%x, petsc_sys%x_scatter, petsc_sys%x_seq, ierr))
      petsc_sys%scatter_ready = .true.
    endif

    PetscCallA(VecScatterBegin(petsc_sys%x_scatter, petsc_sys%x, petsc_sys%x_seq, INSERT_VALUES, SCATTER_FORWARD, ierr))
    PetscCallA(VecScatterEnd(petsc_sys%x_scatter, petsc_sys%x, petsc_sys%x_seq, INSERT_VALUES, SCATTER_FORWARD, ierr))

    ! Read access: only the host copy is consumed here, and x_seq is overwritten in
    ! full by the next scatter, so nothing needs to travel back to the device.
    PetscCallA(VecGetArrayRead(petsc_sys%x_seq, x_arr, ierr))

    if (associated(sol_vec%val)) then
      sol_vec%val(1:sol_vec%n) = x_arr(1:sol_vec%n)
    end if

    PetscCallA(VecRestoreArrayRead(petsc_sys%x_seq, x_arr, ierr))
  end subroutine petsc_recover_solution


  !> Assemble a scalar MATAIJ equilibrium matrix directly from JOREK COO
  !! triplets (a_mat%irn/jcn/val, 1-based, with duplicate entries).
  !! ADD_VALUES sums duplicates natively. Serial (n_cpu==1, COMM_SELF) preallocates
  !! from COO row counts; parallel (n_cpu>1, COMM_WORLD) lets PETSc choose the row
  !! distribution and scatters rank-0-set entries during MatAssembly.
  subroutine petsc_equilibrium_assemble(petsc_sys, a_mat)
    use data_structure, only: type_SP_MATRIX

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_SP_MATRIX), intent(in)       :: a_mat

    integer :: k, r, c, j, owner
    integer :: comm, my_id, n_cpu, mpierr
    PetscInt :: n_global, n_local
    PetscInt :: row(1), col(1)
    PetscScalar :: v(1)
    PetscInt, allocatable :: d_nnz(:), o_nnz(:)
    integer, allocatable  :: d_all(:), o_all(:), d_loc(:), o_loc(:), counts(:), displs(:)
    PetscErrorCode :: ierr

    comm = a_mat%comm
    call MPI_COMM_RANK(comm, my_id, mpierr)
    call MPI_COMM_SIZE(comm, n_cpu, mpierr)

    n_global = a_mat%ng

    call MatCreate(comm, petsc_sys%A, ierr)

    if (n_cpu .eq. 1) then
      n_local = a_mat%ng
      call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
      call MatSetType(petsc_sys%A, MATAIJ, ierr)
      ! Per-row count from COO (duplicate-overestimate, safe); rank owns all rows.
      allocate(d_nnz(n_global))
      d_nnz = 0
      do k = 1, a_mat%nnz
        r = a_mat%irn(k)
        d_nnz(r) = d_nnz(r) + 1
      enddo
      call MatSeqAIJSetPreallocation(petsc_sys%A, 0, d_nnz, ierr)
      deallocate(d_nnz)
    else
      ! Fix the row split up front so rank 0, which holds the whole COO, can work out
      ! every rank's ownership range and count its rows exactly. The previous
      ! n_global-per-row preallocation was effectively dense and grew as O(n^2).
      n_local = PETSC_DECIDE
      call PetscSplitOwnership(comm, n_local, n_global, ierr)
      call MatSetSizes(petsc_sys%A, n_local, n_local, n_global, n_global, ierr)
      call MatSetType(petsc_sys%A, MATAIJ, ierr)

      allocate(counts(n_cpu), displs(n_cpu))
      call MPI_Allgather(int(n_local), 1, MPI_INTEGER, counts, 1, MPI_INTEGER, comm, mpierr)
      displs(1) = 0
      do k = 2, n_cpu
        displs(k) = displs(k-1) + counts(k-1)
      enddo

      ! Per global row, count entries inside vs outside the diagonal block of the rank
      ! owning that row (duplicates included - a safe overestimate).
      if (my_id .eq. 0) then
        allocate(d_all(n_global), o_all(n_global))
        d_all = 0
        o_all = 0
        do k = 1, a_mat%nnz
          r = a_mat%irn(k)
          c = a_mat%jcn(k)
          owner = 1
          do j = n_cpu, 1, -1
            if (r-1 >= displs(j)) then
              owner = j
              exit
            endif
          enddo
          if (c-1 >= displs(owner) .and. c-1 < displs(owner) + counts(owner)) then
            d_all(r) = d_all(r) + 1
          else
            o_all(r) = o_all(r) + 1
          endif
        enddo
      else
        allocate(d_all(1), o_all(1))
      endif

      allocate(d_loc(n_local), o_loc(n_local))
      call MPI_Scatterv(d_all, counts, displs, MPI_INTEGER, d_loc, int(n_local), MPI_INTEGER, 0, comm, mpierr)
      call MPI_Scatterv(o_all, counts, displs, MPI_INTEGER, o_loc, int(n_local), MPI_INTEGER, 0, comm, mpierr)
      deallocate(d_all, o_all)

      ! Clamp to the width actually available in each block; PETSc rejects larger values.
      allocate(d_nnz(n_local), o_nnz(n_local))
      do k = 1, n_local
        d_nnz(k) = min(d_loc(k), int(n_local))
        o_nnz(k) = min(o_loc(k), int(n_global - n_local))
      enddo
      deallocate(d_loc, o_loc, counts, displs)

      call MatMPIAIJSetPreallocation(petsc_sys%A, 0, d_nnz, 0, o_nnz, ierr)
      deallocate(d_nnz, o_nnz)
    endif

    call MatSetOption(petsc_sys%A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

    ! Only rank 0 holds the COO; off-rank entries are cached and shipped at assembly.
    if (my_id .eq. 0) then
      do k = 1, a_mat%nnz
        row(1) = a_mat%irn(k) - 1
        col(1) = a_mat%jcn(k) - 1
        v(1)   = a_mat%val(k)
        call MatSetValues(petsc_sys%A, 1, row, 1, col, v, ADD_VALUES, ierr)
      enddo
    endif

    call MatAssemblyBegin(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(petsc_sys%A, MAT_FINAL_ASSEMBLY, ierr)

    call MatCreateVecs(petsc_sys%A, petsc_sys%x, petsc_sys%b, ierr)

    petsc_sys%initialized = .true.
    petsc_sys%owns_A       = .true.
    if (my_id .eq. 0) write(*,'(A,I0,A,I0,A,I0)') &
      "[PETSc] equilibrium: AIJ matrix ", n_global, "x", n_global, " on ranks=", n_cpu
  end subroutine petsc_equilibrium_assemble


  !> Fill the equilibrium RHS: rank 0 sets all global entries, assembly scatters.
  !! Works for serial (COMM_SELF) and distributed (COMM_WORLD) equilibrium solves.
  subroutine petsc_equilibrium_rhs(petsc_sys, rhs_vec)
    use data_structure, only: type_RHS

    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    type(type_RHS), intent(in)             :: rhs_vec

    integer :: i, comm, my_id, mpierr
    PetscInt :: ng
    PetscInt, allocatable :: idx(:)
    PetscErrorCode :: ierr

    call PetscObjectGetComm(petsc_sys%b, comm, ierr)
    call MPI_COMM_RANK(comm, my_id, mpierr)

    call VecSet(petsc_sys%b, 0.0d0, ierr)

    if (my_id .eq. 0) then
      ng = rhs_vec%n
      allocate(idx(ng))
      do i = 1, ng
        idx(i) = i - 1
      enddo
      call VecSetValues(petsc_sys%b, ng, idx, rhs_vec%val(1:ng), INSERT_VALUES, ierr)
      deallocate(idx)
    endif

    call VecAssemblyBegin(petsc_sys%b, ierr)
    call VecAssemblyEnd(petsc_sys%b, ierr)
  end subroutine petsc_equilibrium_rhs


  !> Destroy all persistent PETSc objects; safe to call even if never initialized.
  !! petsc_sys%A is only destroyed if owns_A=.true. (old path via petsc_init_system).
  !! In the direct assembly path, A is owned by a_mat%petsc_A.
  subroutine petsc_cleanup(petsc_sys)
    type(type_PETSC_SYSTEM), intent(inout) :: petsc_sys
    PetscErrorCode :: ierr

    if (petsc_sys%ksp_ready) then
      call KSPDestroy(petsc_sys%ksp, ierr)
      ! On the native AIJ path A_aij/b_aij/x_aij were never created - the KSP ran on
      ! A, b and x directly - so destroying them here would be a double free.
      if (.not. petsc_sys%aij_native) then
        call MatDestroy(petsc_sys%A_aij, ierr)
        call VecDestroy(petsc_sys%b_aij, ierr)
        call VecDestroy(petsc_sys%x_aij, ierr)
      endif
      petsc_sys%ksp_ready = .false.
    endif
    if (petsc_sys%scatter_ready) then
      call VecScatterDestroy(petsc_sys%x_scatter, ierr)
      call VecDestroy(petsc_sys%x_seq, ierr)
      petsc_sys%scatter_ready = .false.
    endif
    if (petsc_sys%initialized) then
      call VecDestroy(petsc_sys%b, ierr)
      call VecDestroy(petsc_sys%x, ierr)
      if (petsc_sys%owns_A) then
        call MatDestroy(petsc_sys%A, ierr)
        petsc_sys%owns_A = .false.
      endif
      petsc_sys%initialized = .false.
      petsc_sys%aij_native  = .false.
    endif
  end subroutine petsc_cleanup

#endif
end module mod_petsc
