!> BoomerAMG (hypre, through PETSc's PCHYPRE) as the solver of one scalar
!! diagonal block of the physics preconditioners: the backend SF_HYPRE of
!! mod_petsc_pc_sf_solver (namelist value "hypre" of physics_pc_sf_pair_w,
!! _rho, _T, _vpar; -vs_p_solver hypre for vs_lumped's P_lumped, -vs_pv_solver
!! hypre for its P_V). The offline study is studies/spd_multigrid/amg.
!!
!! The operator's rows are (JOREK index, toroidal slot), the slot fastest, ns
!! slots (mod_petsc_pc_sf_ops); one field only. BoomerAMG runs unknown-based
!! with one function per (DOF kind, slot): numfunctions 4 ns, the function of
!! a row its global row number mod 4 ns, which is a fixed (kind, slot) pair on
!! the rings >= 1 of a polar grid (index = 1 + 3 n_tht + 4 ((I-1) n_tht + J)
!! + k there; the axis ring is the axis block's). Two JOREK-side parts from
!! the polar grid make BoomerAMG converge on these C1 Hermite operators:
!!
!!  - the axis block: the rows of the rings 0..k solved exactly (LU, MUMPS)
!!    before and after the V-cycle (symmetric multiplicative). The DOFs d/dJ
!!    and d2/dIdJ of the collapsed axis ring carry near-singular modes that
!!    neither a point smoother nor an AMG coarse space reaches; without the
!!    block every block's cycle stalls on them (kappa 20 to 2000).
!!  - the flip: D A D with D = (-1)^I on the DOF kinds d/dI and d2/dIdJ (I the
!!    ring). The cubic Hermite slope mass has negative off-diagonal entries, so
!!    these kinds couple strongly and negatively across rings in operators
!!    dominated by derivatives along the ring, and classical AMG interpolates
!!    across the rings; that loses the kernel of the parallel operators (one
!!    value per ring and kind, any radial profile). D makes these couplings
!!    positive (weak) and keeps that kernel: the coarsening becomes the
!!    semi-coarsening along the flux surfaces.
!!  - the symmetric diagonal scaling |a_ii|^-1/2 (in D with the flip signs,
!!    recomputed at every build): it evens out the DOF kinds for the strength
!!    of connection and the interpolation (T at n = 0: kappa 4 to 6 -> 2.0 at
!!    dt 30); rho, T and P_V only: no gain for the Laplacian, and P_lumped
!!    loses with it at dt 30 in JOREK (outer 228 / 93 against 95 / 44, m1 / m2).
!!
!! One application y = M x, S = R' A_aa^-1 R the axis solve, H the V-cycle,
!! A_f = D A D (A without flip and scaling):
!!   z = D x;  w = S z;  w = w + H (z - A_f w);  w = w + S (z - A_f w);  y = D w
!! computed with A_f's axis columns after the first S and its axis rows before
!! the second (no full product). Without axis block, flip and scaling M is
!! PCHYPRE.
!!
!! PETSc options, prefix sf_amg<inst>_ (inst = the block's GMG instance:
!! 1 pair_w / P_lumped, 3 rho, 4 T, 5 Laplacian, 6 v_par):
!!   -sf_amg<i>_axis_rings <k>      axis block = rings 0..k, -1: none
!!                                  (P_lumped and v_par's P_V 3, otherwise 1)
!!   -sf_amg<i>_flip <kinds>|none   DOF kinds with (-1)^ring (13; Laplacian none)
!!   -sf_amg<i>_dscale diag|none    the diagonal scaling (rho, T, P_V diag;
!!                                  P_lumped, Laplacian none)
!!   -sf_amg<i>_ksp_*               the Krylov method (FGMRES, restart and
!!                                  iteration cap of the caller, rtol
!!                                  physics_pc_sf_rtol; the Laplacian's
!!                                  default -sf_amg5_ksp_rtol 1e-3)
!!   -sf_amg<i>_pc_hypre_boomeramg_*  every BoomerAMG option of PCHYPRE; set
!!                                  here unless given: numfunctions 4 ns,
!!                                  strong_threshold 0.8 (Laplacian 0.25),
!!                                  grid_sweeps_all 2 (P_lumped, P_V)
!!   -sf_amg<i>_axis_pc_*            the axis block's LU (MUMPS)
!! The axis block and the flip need the polar grid of n_flux x n_tht nodes
!! (mod_petsc_pc_gmg's layout); -sf_amg<i>_axis_rings -1 -sf_amg<i>_flip none
!! -sf_amg<i>_dscale none gives plain BoomerAMG on any grid. hypre's own work
!! is not in PETSc's flop log.
module mod_petsc_pc_hypre
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: hypre_ksp_setup

  integer, parameter :: NINST = 6

  type :: amg_t
    logical :: shell = .false.          !< axis block, flip or scaling: the shell around PCHYPRE
    integer :: axis_rings = -1
    character(len=8) :: flip = ""
    logical :: dscale = .false.         !< the symmetric diagonal scaling |a_ii|^-1/2
    logical :: scaled = .false.         !< flip or dscale: A_f = D A D
    integer :: ns = 1
    PC      :: hpc                      !< the BoomerAMG PC (shell case)
    Mat     :: af                       !< D A D (flip / scaling), else A
    logical :: own_af = .false.
    Vec     :: sgn                      !< the flip signs (+-1)
    Vec     :: d                        !< D: the flip signs times |a_ii|^-1/2 (dscale)
    IS      :: isa, isall
    Mat     :: aa, ar, ac               !< A_f(axis, axis), A_f(axis, :), A_f(:, axis)
    KSP     :: ka                       !< the axis LU
    VecScatter :: sc                    !< full vector -> axis rows
    Vec     :: z, w, t, za, xa, ra
    integer :: na = 0                   !< axis rows (global)
    logical :: ready = .false.
  end type amg_t
  type(amg_t), save :: S(NINST)

  !> the polar grid's ring and DOF kind of every JOREK index (once)
  integer, allocatable, save :: ring_of(:), kind_of(:)

contains

  !> Configure ksp as the BoomerAMG solve of A (one field, ns slots per JOREK
  !! index) for the block of GMG instance inst. fresh: ksp was just created;
  !! later calls take the refilled A (same pattern). amat: the Krylov
  !! operator when A only builds the preconditioner (default A). Collective.
  subroutine hypre_ksp_setup(ksp, A, ns, inst, nfields, label, comm, my_id, rtol, maxits, fresh, amat)
    use mod_petsc_pc_blocks, only: pc_print_block_setup
    KSP, intent(inout)           :: ksp
    Mat, intent(in)              :: A
    integer, intent(in)          :: ns, inst, nfields, comm, my_id, maxits
    character(len=*), intent(in) :: label
    real*8, intent(in)           :: rtol
    logical, intent(in)          :: fresh
    Mat, intent(in), optional    :: amat
    PetscErrorCode :: ierr
    PC :: pc
    character(len=16) :: prefix
    character(len=160) :: txt
    character(len=24) :: tstr

#ifndef PETSC_HAVE_HYPRE
    call fatal("the PETSc of this build has no hypre (configure PETSc with --with-hypre-dir or --download-hypre)")
#endif
    if (inst < 1 .or. inst > NINST) call fatal("hypre backend: no instance for this block")
    if (nfields /= 1) call fatal("the hypre backend takes one field (BoomerAMG diverges on pair_psi's (psi, j) "// &
                                 "rows), got "//trim(label))
    write(prefix, '(A,I0,A)') "sf_amg", inst, "_"
    associate (c => S(inst))
      if (fresh) then
        c%ns = ns
        call read_options(c, inst, prefix)
        call set_defaults(c, inst, prefix)
        call KSPSetType(ksp, KSPFGMRES, ierr)
        call KSPGMRESSetRestart(ksp, maxits, ierr)
        call KSPSetTolerances(ksp, rtol, 1.d-50, 1.d6, maxits, ierr)
        call KSPSetOptionsPrefix(ksp, trim(prefix), ierr)
        call KSPGetPC(ksp, pc, ierr)
        if (c%shell) then
          call PCSetType(pc, PCSHELL, ierr)
          select case (inst)
          case (1); call PCShellSetApply(pc, hypre_pc_apply_1, ierr)
          case (2); call PCShellSetApply(pc, hypre_pc_apply_2, ierr)
          case (3); call PCShellSetApply(pc, hypre_pc_apply_3, ierr)
          case (4); call PCShellSetApply(pc, hypre_pc_apply_4, ierr)
          case (5); call PCShellSetApply(pc, hypre_pc_apply_5, ierr)
          case (6); call PCShellSetApply(pc, hypre_pc_apply_6, ierr)
          end select
          call PCShellSetName(pc, trim(label)//" BoomerAMG with axis block / flip", ierr)
          call PCCreate(comm, c%hpc, ierr)
          call PCSetOptionsPrefix(c%hpc, trim(prefix), ierr)
          call PCSetType(c%hpc, PCHYPRE, ierr)
          call PCSetFromOptions(c%hpc, ierr)
        else
          call PCSetType(pc, PCHYPRE, ierr)
        endif
        call KSPSetFromOptions(ksp, ierr)
      endif

      if (c%shell) then
        call build_shell(c, A, comm, my_id, prefix)
        call PCSetOperators(c%hpc, c%af, c%af, ierr)
        call PCSetUp(c%hpc, ierr)
      endif
      if (present(amat)) then
        call KSPSetOperators(ksp, amat, A, ierr)
      else
        call KSPSetOperators(ksp, A, A, ierr)
      endif
      call KSPSetUp(ksp, ierr)

      if (fresh) then
        write(tstr, '(ES9.2)') rtol
        write(txt, '(A,I0,A)') "FGMRES + BoomerAMG V-cycle (hypre, -"//trim(prefix)//"*), ", 4 * ns, " functions"
        if (c%axis_rings >= 0) write(txt, '(A,A,I0,A,I0,A)') trim(txt), ", axis rings 0..", c%axis_rings, &
                                                             " (", c%na, " rows, LU)"
        if (len_trim(c%flip) > 0) txt = trim(txt)//", flip kinds "//trim(c%flip)
        if (c%dscale) txt = trim(txt)//", diagonal scaling"
        call pc_print_block_setup(comm, label, trim(txt)//", rtol "//trim(adjustl(tstr)))
      endif
    end associate

  contains

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      if (my_id == 0) write(*,'(A,A)') "[Physics PC]   FATAL: ", trim(msg)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine hypre_ksp_setup

  !> The JOREK-side options -sf_amg<i>_axis_rings, -sf_amg<i>_flip, with the
  !! block's defaults (module header).
  subroutine read_options(c, inst, prefix)
    type(amg_t), intent(inout) :: c
    integer, intent(in) :: inst
    character(len=*), intent(in) :: prefix
    PetscErrorCode :: ierr
    PetscBool :: is_set
    PetscInt :: iv
    character(len=8) :: s
    c%axis_rings = 1
    if (inst == 1 .or. inst == 6) c%axis_rings = 3
    c%flip = "13"
    if (inst == 5) c%flip = ""
    call PetscOptionsGetInt(PETSC_NULL_OPTIONS, trim(prefix), "-axis_rings", iv, is_set, ierr)
    if (is_set) c%axis_rings = int(iv)
    s = c%flip
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, trim(prefix), "-flip", s, is_set, ierr)
    if (is_set) c%flip = s
    if (trim(c%flip) == "none") c%flip = ""
    c%dscale = (inst == 3 .or. inst == 4 .or. inst == 6)
    s = merge("diag", "none", c%dscale)
    call PetscOptionsGetString(PETSC_NULL_OPTIONS, trim(prefix), "-dscale", s, is_set, ierr)
    if (is_set) c%dscale = (trim(s) == "diag")
    c%scaled = (len_trim(c%flip) > 0 .or. c%dscale)
    c%shell  = (c%axis_rings >= 0 .or. c%scaled)
  end subroutine read_options

  !> BoomerAMG's options of the block unless given (the study's tuning).
  subroutine set_defaults(c, inst, prefix)
    type(amg_t), intent(in) :: c
    integer, intent(in) :: inst
    character(len=*), intent(in) :: prefix
    character(len=16) :: v
    write(v, '(I0)') 4 * c%ns
    call set_default("pc_hypre_type", "boomeramg")
    call set_default("pc_hypre_boomeramg_numfunctions", trim(v))
    if (inst == 5) then
      call set_default("pc_hypre_boomeramg_strong_threshold", "0.25")
      ! one GMG cycle reduces the Laplacian's residual by 1e-3; at rtol 1e-1
      ! one BoomerAMG cycle (0.1) costs 20 to 45 % more outer iterations
      call set_default("ksp_rtol", "1e-3")
    else
      call set_default("pc_hypre_boomeramg_strong_threshold", "0.8")
    endif
    if (inst == 1 .or. inst == 6) call set_default("pc_hypre_boomeramg_grid_sweeps_all", "2")
  contains
    subroutine set_default(nm, val)
      character(len=*), intent(in) :: nm, val
      PetscErrorCode :: ierr
      PetscBool :: has
      call PetscOptionsHasName(PETSC_NULL_OPTIONS, trim(prefix), "-"//nm, has, ierr)
      if (.not. has) call PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-"//trim(prefix)//nm, val, ierr)
    end subroutine set_default
  end subroutine set_defaults

  !> The ring and the DOF kind of every JOREK index of the polar grid, once.
  subroutine polar_tables(n_idx, my_id)
    use nodes_elements, only: node_list, element_list
    use phys_module,    only: n_flux, n_tht
    integer, intent(in) :: n_idx, my_id
    integer :: n, k, f, e, iv, vi(4), vj(4)
    PetscErrorCode :: ierr
    if (allocated(ring_of)) return
    if (node_list%n_nodes /= n_flux * n_tht .or. n_flux < 3 .or. &
        element_list%n_elements /= (n_flux - 1) * n_tht) call fatal_p()
    do e = 1, element_list%n_elements
      do iv = 1, 4
        n = element_list%element(e)%vertex(iv)
        vi(iv) = (n - 1) / n_tht; vj(iv) = mod(n - 1, n_tht)
      enddo
      if (.not. (vi(2) == vi(1) + 1 .and. vj(2) == vj(1) .and. vi(3) == vi(1) + 1 .and. &
                 vj(3) == mod(vj(1) + 1, n_tht) .and. vi(4) == vi(1) .and. vj(4) == mod(vj(1) + 1, n_tht))) &
        call fatal_p()
    enddo
    allocate(ring_of(0:n_idx - 1), kind_of(0:n_idx - 1))
    ring_of = -1; kind_of = -1
    do n = 1, node_list%n_nodes
      do k = 0, 3
        f = node_list%node(n)%index(k + 1) - 1
        if (f < 0 .or. f >= n_idx) cycle
        if (ring_of(f) >= 0) cycle                      ! a shared DOF (the axis value): its first node
        ring_of(f) = (n - 1) / n_tht
        kind_of(f) = k
      enddo
    enddo
  contains
    subroutine fatal_p()
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: the hypre backend's axis block and flip need the polar "// &
        "grid of n_flux x n_tht nodes; on other grids use -sf_amg<i>_axis_rings -1 -sf_amg<i>_flip none"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal_p
  end subroutine polar_tables

  !> A_f, the flip signs, the axis rows, their LU and the coupling blocks; at
  !! every build (refills reuse the objects).
  subroutine build_shell(c, A, comm, my_id, prefix)
    type(amg_t), intent(inout) :: c
    Mat, intent(in) :: A
    integer, intent(in) :: comm, my_id
    character(len=*), intent(in) :: prefix
    PetscErrorCode :: ierr
    PetscInt :: rs, re, ng, r, na_l, one
    PetscInt, allocatable :: ia(:)
    PetscScalar, pointer :: dp(:)
    PC :: pc
    integer :: f, mpierr
    integer :: nloc_axis

    one = 1
    call MatGetOwnershipRange(A, rs, re, ierr)
    call MatGetSize(A, ng, PETSC_NULL_INTEGER, ierr)
    if (mod(int(ng), c%ns) /= 0) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: hypre backend: rows are not whole (index, slots) groups"
      call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
    endif
    call polar_tables(int(ng) / c%ns, my_id)

    if (.not. c%ready) then
      ! the flip signs and the axis rows (global row numbers of this rank)
      if (c%scaled) then
        call MatCreateVecs(A, c%sgn, PETSC_NULL_VEC, ierr)
        call VecDuplicate(c%sgn, c%d, ierr)
        call VecGetArray(c%sgn, dp, ierr)
        do r = rs, re - 1
          f = int(r) / c%ns
          dp(r - rs + 1) = 1.d0
          if (kind_of(f) >= 0 .and. len_trim(c%flip) > 0) then
            if (index(trim(c%flip), achar(iachar('0') + kind_of(f))) > 0 .and. mod(ring_of(f), 2) == 1) &
              dp(r - rs + 1) = -1.d0
          endif
        enddo
        call VecRestoreArray(c%sgn, dp, ierr)
      endif
      call MatCreateVecs(A, c%z, c%w, ierr)
      call VecDuplicate(c%z, c%t, ierr)
      if (c%axis_rings >= 0) then
        allocate(ia(max(int(re - rs), 1)))
        na_l = 0
        do r = rs, re - 1
          f = int(r) / c%ns
          if (ring_of(f) >= 0 .and. ring_of(f) <= c%axis_rings) then
            na_l = na_l + 1; ia(na_l) = r
          endif
        enddo
        call ISCreateGeneral(comm, na_l, ia, PETSC_COPY_VALUES, c%isa, ierr)
        call ISCreateStride(comm, re - rs, rs, one, c%isall, ierr)
        nloc_axis = int(na_l)
        call MPI_Allreduce(nloc_axis, c%na, 1, MPI_INTEGER, MPI_SUM, comm, mpierr)
        deallocate(ia)
      endif
    endif

    ! A_f = D A D, D = the signs times |a_ii|^-1/2 (dscale) at every build
    if (c%scaled) then
      if (c%dscale) then
        call MatGetDiagonal(A, c%d, ierr)
        call VecGetArray(c%d, dp, ierr)
        do r = 1, re - rs
          if (abs(dp(r)) > 0.d0) then
            dp(r) = 1.d0 / sqrt(abs(dp(r)))
          else
            dp(r) = 1.d0
          endif
        enddo
        call VecRestoreArray(c%d, dp, ierr)
        call VecPointwiseMult(c%d, c%d, c%sgn, ierr)
      else
        call VecCopy(c%sgn, c%d, ierr)
      endif
      if (.not. c%ready) then
        call MatDuplicate(A, MAT_COPY_VALUES, c%af, ierr)
        c%own_af = .true.
      else
        call MatCopy(A, c%af, SAME_NONZERO_PATTERN, ierr)
      endif
      call MatDiagonalScale(c%af, c%d, c%d, ierr)
    else
      c%af = A
    endif

    if (c%axis_rings >= 0) then
      if (.not. c%ready) then
        call MatCreateSubMatrix(c%af, c%isa, c%isa, MAT_INITIAL_MATRIX, c%aa, ierr)
        call MatCreateSubMatrix(c%af, c%isa, c%isall, MAT_INITIAL_MATRIX, c%ar, ierr)
        call MatCreateSubMatrix(c%af, c%isall, c%isa, MAT_INITIAL_MATRIX, c%ac, ierr)
        call MatCreateVecs(c%aa, c%xa, c%za, ierr)
        call VecDuplicate(c%za, c%ra, ierr)
        call VecScatterCreate(c%z, c%isa, c%za, PETSC_NULL_IS, c%sc, ierr)
        call KSPCreate(comm, c%ka, ierr)
        call KSPSetType(c%ka, KSPPREONLY, ierr)
        call KSPSetOptionsPrefix(c%ka, trim(prefix)//"axis_", ierr)
        call KSPGetPC(c%ka, pc, ierr)
        call PCSetType(pc, PCLU, ierr)
        call PCFactorSetMatSolverType(pc, MATSOLVERMUMPS, ierr)
        call KSPSetFromOptions(c%ka, ierr)
      else
        call MatCreateSubMatrix(c%af, c%isa, c%isa, MAT_REUSE_MATRIX, c%aa, ierr)
        call MatCreateSubMatrix(c%af, c%isa, c%isall, MAT_REUSE_MATRIX, c%ar, ierr)
        call MatCreateSubMatrix(c%af, c%isall, c%isa, MAT_REUSE_MATRIX, c%ac, ierr)
      endif
      call KSPSetOperators(c%ka, c%aa, c%aa, ierr)
      call KSPSetUp(c%ka, ierr)
      if (.not. c%ready) call axis_cost()
    endif
    c%ready = .true.

  contains

    !> The first build's axis block sizes: rows, the LU's entries (MUMPS
    !! INFOG(29)), the coupling entries of A_f(axis, :) and A_f(:, axis), and
    !! the cost of one application's axis part in units of a product with A
    !! (4 factor entries + 2 coupling entries over 2 nnz(A)).
    subroutine axis_cost()
      Mat :: F
      MatInfo :: ir, ic, iaf
      PetscInt :: nf, i29
      real*8 :: fe
      i29 = 29
      call KSPGetPC(c%ka, pc, ierr)
      call PCFactorGetMatrix(pc, F, ierr)
      call MatMumpsGetInfog(F, i29, nf, ierr)
      fe = dble(nf)
      if (nf < 0) fe = -dble(nf) * 1.d6
      call MatGetInfo(c%ar, MAT_GLOBAL_SUM, ir, ierr)
      call MatGetInfo(c%ac, MAT_GLOBAL_SUM, ic, ierr)
      call MatGetInfo(c%af, MAT_GLOBAL_SUM, iaf, ierr)
      if (my_id == 0) write(*,'(A,I0,A,ES10.3,A,ES10.3,A,ES10.3,A,F7.3,A)') "[Physics PC]   hypre axis block: ", c%na, &
        " rows, LU entries ", fe, ", coupling entries ", ir%nz_used + ic%nz_used, " of A's ", iaf%nz_used, &
        ": ", (4.d0 * fe + 2.d0 * (ir%nz_used + ic%nz_used)) / (2.d0 * iaf%nz_used), " A-products per application"
    end subroutine axis_cost

  end subroutine build_shell

  !> y = M x (module header) for instance i.
  subroutine shell_apply(i, x, y, ierr)
    integer, intent(in) :: i
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    associate (c => S(i))
      if (c%scaled) then
        call VecPointwiseMult(c%z, x, c%d, ierr)
      else
        call VecCopy(x, c%z, ierr)
      endif
      if (c%axis_rings >= 0) then
        ! w = S z
        call VecScatterBegin(c%sc, c%z, c%za, INSERT_VALUES, SCATTER_FORWARD, ierr)
        call VecScatterEnd(c%sc, c%z, c%za, INSERT_VALUES, SCATTER_FORWARD, ierr)
        call KSPSolve(c%ka, c%za, c%xa, ierr)
        ! t = z - A_f w = z - A_f(:, axis) x_a
        call MatMult(c%ac, c%xa, c%t, ierr)
        call VecAYPX(c%t, -1.0d0, c%z, ierr)
        ! w = R' x_a + H t
        call PCApply(c%hpc, c%t, c%w, ierr)
        call VecScatterBegin(c%sc, c%xa, c%w, ADD_VALUES, SCATTER_REVERSE, ierr)
        call VecScatterEnd(c%sc, c%xa, c%w, ADD_VALUES, SCATTER_REVERSE, ierr)
        ! w = w + S (z - A_f w): the axis rows of the residual only
        call MatMult(c%ar, c%w, c%ra, ierr)
        call VecAYPX(c%ra, -1.0d0, c%za, ierr)
        call KSPSolve(c%ka, c%ra, c%xa, ierr)
        call VecScatterBegin(c%sc, c%xa, c%w, ADD_VALUES, SCATTER_REVERSE, ierr)
        call VecScatterEnd(c%sc, c%xa, c%w, ADD_VALUES, SCATTER_REVERSE, ierr)
      else
        call PCApply(c%hpc, c%z, c%w, ierr)
      endif
      if (c%scaled) then
        call VecPointwiseMult(y, c%w, c%d, ierr)
      else
        call VecCopy(c%w, y, ierr)
      endif
    end associate
    ierr = 0
  end subroutine shell_apply

  subroutine hypre_pc_apply_1(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    call shell_apply(1, x, y, ierr)
  end subroutine hypre_pc_apply_1

  subroutine hypre_pc_apply_2(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    call shell_apply(2, x, y, ierr)
  end subroutine hypre_pc_apply_2

  subroutine hypre_pc_apply_3(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    call shell_apply(3, x, y, ierr)
  end subroutine hypre_pc_apply_3

  subroutine hypre_pc_apply_4(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    call shell_apply(4, x, y, ierr)
  end subroutine hypre_pc_apply_4

  subroutine hypre_pc_apply_5(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    call shell_apply(5, x, y, ierr)
  end subroutine hypre_pc_apply_5

  subroutine hypre_pc_apply_6(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode, intent(out) :: ierr
    call shell_apply(6, x, y, ierr)
  end subroutine hypre_pc_apply_6

#endif
end module mod_petsc_pc_hypre
