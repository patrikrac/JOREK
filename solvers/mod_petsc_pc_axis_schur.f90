!> The operator of the axis Schur complement form AS(k) of vs_lumped's
!! resistive bending (mod_petsc_pc_vs, -vs_dual_axis schur), never assembled
!! as one matrix:
!!   P = A + L K^-1 R,
!! A sparse (P_s), L (n x nK) and R (nK x n) sparse, K (nK x nK) sparse,
!! symmetric positive definite and small (the DOFs of the axis rings times
!! the slots). This module holds
!!  - K and its LU (axk_*): all of K's rows and columns on one rank (the K
!!    rank); there K is gathered once more as a sequential matrix and
!!    factored by PETSc's LU in the ordering the caller chooses (natural = K's
!!    own row order); a solve with K is local to that rank, its right-hand
!!    side and solution are vectors of K's parallel layout;
!!  - P as a MATSHELL (axs_*), y = A x + L (K^-1 (R x)), for any (A, L, R)
!!    on the same K (the SPD multigrid's levels: A_g, L_g = P_g' L_(g-1),
!!    R_g = R_(g-1) P_g);
!!  - the augmented sparse matrix [[A, L], [R, -K]] and its MUMPS LU (axa_*):
!!    the first block of the solution for the right-hand side [b; 0] is
!!    P^-1 b, P being its Schur complement. Its rows on a rank: A's rows of
!!    the rank, then K's (on the K rank). With a PCSHELL (axa_pcapply) it is
!!    the exact solve of P in a KSP. Optionally without the entries between
!!    two |n| families (the SPD multigrid's coarsest level; exact at an
!!    axisymmetric state).
!! Objects are found by their PETSc handle (the shell matrix, the PC), as
!! mod_petsc_pc_spdmg finds its smoothers. -log_view events: AXS_Fact (K's
!! LU), AXS_KSolve (solves with K), AXS_Mult (the axis part L K^-1 R x of a
!! shell product), AXS_Aug (the augmented matrix: assembly and factors).
module mod_petsc_pc_axis_schur
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  public :: axk_set, axk_solve, axk_lu_nnz, axk_rank, axk_mat, axs_shell, axs_axis_mult, axa_new, axa_setup, &
            axa_solve, axa_set_pc, axa_pcapply, axa_factor_nnz

  integer, parameter :: MAXK = 2, MAXSH = 32, MAXAUG = 8

  !> K and its factor
  type :: axk_t
    logical :: ready = .false.
    integer :: comm = MPI_COMM_NULL, me = 0, kr = -1
    PetscInt :: nk = 0
    Mat :: K                                   !< the caller's (MPIAIJ, rows on the K rank)
    Mat, pointer :: ks(:) => null()            !< K gathered on the K rank (empty elsewhere)
    Mat :: kf                                  !< its LU (K rank)
    IS :: isk, rperm, cperm
    Vec :: ts, ss                              !< sequential work vectors (K rank)
    real*8 :: lu_nnz = 0.d0                    !< entries of the factor (K rank)
  end type axk_t

  !> one shell operator A + L K^-1 R
  type :: axsh_t
    logical :: used = .false.
    Mat :: sh, A, L, R
    integer :: ik = 0
    Vec :: t, s, y2                            !< R x, K^-1 R x (K's layout), L K^-1 R x
  end type axsh_t

  !> one augmented matrix [[A, L], [R, -K]] and its MUMPS LU
  type :: axaug_t
    logical :: used = .false., made = .false., has_pc = .false., reported = .false.
    integer :: ik = 0, comm = MPI_COMM_NULL, me = 0, np = 1
    Mat :: aug
    KSP :: ksp
    PC :: pc                                   !< the PCSHELL applying it (axa_set_pc)
    Vec :: ab, ax
    PetscInt :: na = 0, nkl = 0                !< this rank's rows of A and of K
    PetscInt, allocatable :: ra(:), rk(:), aoff(:)   !< (0:np) A's and K's ownership ranges, the augmented offsets
    integer :: ns = 0, nka = 0                 !< the family filter (axa_setup's ns): slots, K's rows per slot
    integer, allocatable :: fam(:)             !< (0:ns-1) the |n| family of a slot
  end type axaug_t

  type(axk_t), target, save :: axk(MAXK)
  type(axsh_t), target, save :: axsh(MAXSH)
  type(axaug_t), target, save :: axaug(MAXAUG)

  PetscLogEvent, save :: aev_fact = -1, aev_ksolve = -1, aev_mult = -1, aev_aug = -1
  logical, save :: aev_ready = .false.

contains

  subroutine axs_register_events()
    PetscErrorCode :: ierr
    PetscClassId, parameter :: cid = 0
    if (aev_ready) return
    call PetscLogEventRegister("AXS_Fact",   cid, aev_fact,   ierr)
    call PetscLogEventRegister("AXS_KSolve", cid, aev_ksolve, ierr)
    call PetscLogEventRegister("AXS_Mult",   cid, aev_mult,   ierr)
    call PetscLogEventRegister("AXS_Aug",    cid, aev_aug,    ierr)
    aev_ready = .true.
  end subroutine axs_register_events

  !> K object ik takes K (its first call: the rank holding K's rows, K's
  !! sequential copy there and the symbolic LU in `ordering`; every call: the
  !! numeric LU of the refilled K, same pattern). Collective on K's
  !! communicator.
  subroutine axk_set(ik, K, ordering)
    integer, intent(in) :: ik
    Mat, intent(in) :: K
    character(len=*), intent(in) :: ordering
    type(axk_t), pointer :: Q
    PetscInt :: rs, re, ng, t
    PetscInt, allocatable :: all(:)
    PetscErrorCode :: ierr
    MatFactorInfo :: info
    MatInfo :: mi
    integer :: kr, mpierr
    PetscInt, parameter :: one = 1
    call axs_register_events()
    Q => axk(ik)
    call PetscLogEventBegin(aev_fact, ierr)
    call MatFactorInfoInitialize(info, ierr)
    if (.not. Q%ready) then
      Q%K = K
      call PetscObjectGetComm(K, Q%comm, ierr)
      call MPI_Comm_rank(Q%comm, Q%me, mpierr)
      call MatGetOwnershipRange(K, rs, re, ierr)
      call MatGetSize(K, ng, PETSC_NULL_INTEGER, ierr)
      Q%nk = ng
      if (re - rs /= 0 .and. re - rs /= ng) call fail("K's rows on more than one rank")
      kr = -1
      if (re - rs == ng) kr = Q%me
      call MPI_Allreduce(kr, Q%kr, 1, MPI_INTEGER, MPI_MAX, Q%comm, mpierr)
      if (Q%me == Q%kr) then
        allocate(all(ng))
        all = [(t, t = 0, ng - 1)]
      else
        allocate(all(0))
      endif
      call ISCreateGeneral(PETSC_COMM_SELF, int(size(all), kind(one)), all, PETSC_COPY_VALUES, Q%isk, ierr)
      call MatCreateSubMatrices(K, one, [Q%isk], [Q%isk], MAT_INITIAL_MATRIX, Q%ks, ierr)
      if (Q%me == Q%kr) then
        call MatGetOrdering(Q%ks(1), trim(ordering), Q%rperm, Q%cperm, ierr)
        call MatGetFactor(Q%ks(1), MATSOLVERPETSC, MAT_FACTOR_LU, Q%kf, ierr)
        call MatLUFactorSymbolic(Q%kf, Q%ks(1), Q%rperm, Q%cperm, info, ierr)
        call VecCreateSeq(PETSC_COMM_SELF, ng, Q%ts, ierr)
        call VecDuplicate(Q%ts, Q%ss, ierr)
      endif
      Q%ready = .true.
    else
      call MatCreateSubMatrices(K, one, [Q%isk], [Q%isk], MAT_REUSE_MATRIX, Q%ks, ierr)
    endif
    if (Q%me == Q%kr) then
      call MatLUFactorNumeric(Q%kf, Q%ks(1), info, ierr)
      call MatGetInfo(Q%kf, MAT_LOCAL, mi, ierr)
      Q%lu_nnz = mi%nz_used
    endif
    call PetscLogEventEnd(aev_fact, ierr)
  end subroutine axk_set

  !> The entries of K's factor (summed over the ranks: the K rank's). Collective.
  real*8 function axk_lu_nnz(ik)
    integer, intent(in) :: ik
    integer :: mpierr
    call MPI_Allreduce(axk(ik)%lu_nnz, axk_lu_nnz, 1, MPI_DOUBLE_PRECISION, MPI_SUM, axk(ik)%comm, mpierr)
  end function axk_lu_nnz

  !> The rank holding K (K's communicator).
  integer function axk_rank(ik)
    integer, intent(in) :: ik
    axk_rank = axk(ik)%kr
  end function axk_rank

  !> K of object ik (the caller's matrix).
  function axk_mat(ik) result(K)
    integer, intent(in) :: ik
    Mat :: K
    K = axk(ik)%K
  end function axk_mat

  !> s = K^-1 t, t and s of K's parallel layout: a solve on the K rank, nothing
  !! elsewhere (not collective).
  subroutine axk_solve(ik, t, s)
    integer, intent(in) :: ik
    Vec :: t, s
    type(axk_t), pointer :: Q
    PetscScalar, pointer :: a(:), b(:)
    PetscErrorCode :: ierr
    Q => axk(ik)
    if (Q%me /= Q%kr) return
    call PetscLogEventBegin(aev_ksolve, ierr)
    call VecGetArrayRead(t, a, ierr)
    call VecGetArray(Q%ts, b, ierr)
    b = a
    call VecRestoreArray(Q%ts, b, ierr)
    call VecRestoreArrayRead(t, a, ierr)
    call MatSolve(Q%kf, Q%ts, Q%ss, ierr)
    call VecGetArrayRead(Q%ss, a, ierr)
    call VecGetArray(s, b, ierr)
    b = a
    call VecRestoreArray(s, b, ierr)
    call VecRestoreArrayRead(Q%ss, a, ierr)
    call PetscLogEventEnd(aev_ksolve, ierr)
  end subroutine axk_solve

  !> sh, a MATSHELL of A + L K^-1 R (K object ik) with A's layout; the
  !! matrices are referenced, not copied: refills of A, L, R and K reach the
  !! shell. Collective.
  subroutine axs_shell(A, L, R, ik, sh)
    Mat, intent(in) :: A, L, R
    integer, intent(in) :: ik
    Mat, intent(out) :: sh
    PetscInt :: m, n, mg, ngl
    PetscErrorCode :: ierr
    integer :: i, comm
    i = 1
    do while (i <= MAXSH)
      if (.not. axsh(i)%used) exit
      i = i + 1
    enddo
    if (i > MAXSH) call fail("too many shell operators")
    call PetscObjectGetComm(A, comm, ierr)
    call MatGetLocalSize(A, m, n, ierr)
    call MatGetSize(A, mg, ngl, ierr)
    call MatCreateShell(comm, m, n, mg, ngl, 0, sh, ierr)
    call MatShellSetOperation(sh, MATOP_MULT, axs_mult, ierr)
    axsh(i)%used = .true.
    axsh(i)%sh = sh; axsh(i)%A = A; axsh(i)%L = L; axsh(i)%R = R; axsh(i)%ik = ik
    call MatCreateVecs(R, PETSC_NULL_VEC, axsh(i)%t, ierr)
    call VecDuplicate(axsh(i)%t, axsh(i)%s, ierr)
    call MatCreateVecs(A, PETSC_NULL_VEC, axsh(i)%y2, ierr)
  end subroutine axs_shell

  !> y = A x + L K^-1 R x, the shell's product.
  subroutine axs_mult(sh, x, y, ierr)
    Mat :: sh
    Vec :: x, y
    PetscErrorCode :: ierr
    integer :: i
    i = find_shell(sh)
    call MatMult(axsh(i)%A, x, y, ierr)
    call axs_axis_part(i, x)
    call VecAXPY(y, 1.0d0, axsh(i)%y2, ierr)
    ierr = 0
  end subroutine axs_mult

  !> y = L K^-1 R x alone, for the shell sh (the skew report). Collective.
  subroutine axs_axis_mult(sh, x, y)
    Mat, intent(in) :: sh
    Vec :: x, y
    PetscErrorCode :: ierr
    integer :: i
    i = find_shell(sh)
    call axs_axis_part(i, x)
    call VecCopy(axsh(i)%y2, y, ierr)
  end subroutine axs_axis_mult

  !> shell i's y2 = L K^-1 R x
  subroutine axs_axis_part(i, x)
    integer, intent(in) :: i
    Vec :: x
    PetscErrorCode :: ierr
    call PetscLogEventBegin(aev_mult, ierr)
    call MatMult(axsh(i)%R, x, axsh(i)%t, ierr)
    call axk_solve(axsh(i)%ik, axsh(i)%t, axsh(i)%s)
    call MatMult(axsh(i)%L, axsh(i)%s, axsh(i)%y2, ierr)
    call PetscLogEventEnd(aev_mult, ierr)
  end subroutine axs_axis_part

  integer function find_shell(sh)
    Mat, intent(in) :: sh
    integer :: i
    do i = 1, MAXSH
      if (.not. axsh(i)%used) cycle
      if (axsh(i)%sh%v == sh%v) then
        find_shell = i; return
      endif
    enddo
    find_shell = 0
    call fail("a shell product of an unknown operator")
  end function find_shell

  !> A free handle for an augmented matrix.
  integer function axa_new()
    integer :: i
    do i = 1, MAXAUG
      if (.not. axaug(i)%used) then
        axaug(i)%used = .true.
        axa_new = i; return
      endif
    enddo
    axa_new = 0
    call fail("too many augmented matrices")
  end function axa_new

  !> Augmented matrix ia = [[A, L], [R, -K]] (K object ik) and its MUMPS LU:
  !! the first call creates it (pattern, layout, KSP PREONLY + LU), later
  !! calls refill it (same patterns) and refactor. Collective on A's
  !! communicator. label: the KSP's options prefix (axis_<label>_). ns
  !! (optional): A's slots per index (rows index * ns + slot), and the entries
  !! between two |n| families leave the matrix (exact at an axisymmetric
  !! state, whose zeros the global layout stores; K's slot-major index
  !! space); without it all entries stay.
  subroutine axa_setup(ia, A, L, R, ik, label, ns)
    use mod_petsc_pc_harm, only: pc_global_slot
    integer, intent(in) :: ia, ik
    Mat, intent(in) :: A, L, R
    character(len=*), intent(in) :: label
    integer, intent(in), optional :: ns
    type(axaug_t), pointer :: Q
    PetscInt, allocatable :: dnz(:), onz(:)
    PetscInt :: rs, re, ks, ke, row, ncols, nrow, c0, c1
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt, allocatable :: ac(:)
    PetscScalar, allocatable :: av(:)
    PetscErrorCode :: ierr
    PC :: pcl
    Mat :: fm
    PetscInt :: iv
    PetscReal :: rv
    integer :: ir, pass, mpierr, rfam
    PetscInt, parameter :: zero = 0, one = 1
    Q => axaug(ia)
    call axs_register_events()
    call PetscLogEventBegin(aev_aug, ierr)
    if (.not. Q%made) then
      Q%ik = ik
      Q%ns = 0
      if (present(ns)) then
        Q%ns = ns
        allocate(Q%fam(0:ns - 1))
        do ir = 0, ns - 1
          Q%fam(ir) = (pc_global_slot(ir) + 1) / 2
        enddo
        Q%nka = axk(ik)%nk / ns
      endif
      call PetscObjectGetComm(A, Q%comm, ierr)
      call MPI_Comm_rank(Q%comm, Q%me, mpierr)
      call MPI_Comm_size(Q%comm, Q%np, mpierr)
      allocate(Q%ra(0:Q%np), Q%rk(0:Q%np), Q%aoff(0:Q%np))
      call MatGetOwnershipRange(A, rs, re, ierr)
      call MatGetOwnershipRange(axk(ik)%K, ks, ke, ierr)
      Q%na = re - rs; Q%nkl = ke - ks
      call MPI_Allgather(rs, 1, MPIU_INTEGER, Q%ra, 1, MPIU_INTEGER, Q%comm, mpierr)
      call MPI_Allgather(ks, 1, MPIU_INTEGER, Q%rk, 1, MPIU_INTEGER, Q%comm, mpierr)
      call MatGetSize(A, Q%ra(Q%np), PETSC_NULL_INTEGER, ierr)
      Q%rk(Q%np) = axk(ik)%nk
      Q%aoff(0) = 0
      do ir = 0, Q%np - 1
        Q%aoff(ir + 1) = Q%aoff(ir) + (Q%ra(ir + 1) - Q%ra(ir)) + (Q%rk(ir + 1) - Q%rk(ir))
      enddo
    endif
    c0 = Q%aoff(Q%me); c1 = Q%aoff(Q%me + 1)
    call MatGetOwnershipRange(A, rs, re, ierr)
    ks = Q%rk(Q%me); ke = Q%rk(Q%me + 1)
    ! pass 1 (first call): the preallocation; pass 2: the values
    do pass = 1, 2
      if (pass == 1 .and. Q%made) cycle
      if (pass == 1) then
        allocate(dnz(Q%na + Q%nkl), onz(Q%na + Q%nkl)); dnz = 0; onz = 0
      else if (.not. Q%made) then
        call MatCreate(Q%comm, Q%aug, ierr)
        call MatSetSizes(Q%aug, Q%na + Q%nkl, Q%na + Q%nkl, Q%aoff(Q%np), Q%aoff(Q%np), ierr)
        call MatSetType(Q%aug, MATMPIAIJ, ierr)
        call MatMPIAIJSetPreallocation(Q%aug, zero, dnz, zero, onz, ierr)
        call MatSetOption(Q%aug, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE, ierr)
      else
        call MatZeroEntries(Q%aug, ierr)
      endif
      ! A's rows: A, then L
      do row = rs, re - 1
        nrow = c0 + (row - rs)
        if (Q%ns > 0) rfam = Q%fam(int(mod(row, int(Q%ns, kind(row)))))
        call MatGetRow(A, row, ncols, cols, vals, ierr)
        call put(1, ncols, cols, vals, 1.0d0)
        call MatRestoreRow(A, row, ncols, cols, vals, ierr)
        call MatGetRow(L, row, ncols, cols, vals, ierr)
        call put(2, ncols, cols, vals, 1.0d0)
        call MatRestoreRow(L, row, ncols, cols, vals, ierr)
      enddo
      ! K's rows (the K rank): R, then -K
      do row = ks, ke - 1
        nrow = c0 + Q%na + (row - ks)
        if (Q%ns > 0) rfam = Q%fam(int(row) / Q%nka)
        call MatGetRow(R, row, ncols, cols, vals, ierr)
        call put(1, ncols, cols, vals, 1.0d0)
        call MatRestoreRow(R, row, ncols, cols, vals, ierr)
        call MatGetRow(axk(ik)%K, row, ncols, cols, vals, ierr)
        call put(2, ncols, cols, vals, -1.0d0)
        call MatRestoreRow(axk(ik)%K, row, ncols, cols, vals, ierr)
      enddo
    enddo
    call MatAssemblyBegin(Q%aug, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(Q%aug, MAT_FINAL_ASSEMBLY, ierr)
    if (.not. Q%made) then
      call KSPCreate(Q%comm, Q%ksp, ierr)
      call KSPSetType(Q%ksp, KSPPREONLY, ierr)
      call KSPGetPC(Q%ksp, pcl, ierr)
      call PCSetType(pcl, PCLU, ierr)
      call PCFactorSetMatSolverType(pcl, MATSOLVERMUMPS, ierr)
      call KSPSetOptionsPrefix(Q%ksp, "axis_"//trim(label)//"_", ierr)
      call KSPSetFromOptions(Q%ksp, ierr)
      call MatCreateVecs(Q%aug, Q%ax, Q%ab, ierr)
      Q%made = .true.
    endif
    call KSPSetOperators(Q%ksp, Q%aug, Q%aug, ierr)
    call KSPSetUp(Q%ksp, ierr)
    call PetscLogEventEnd(aev_aug, ierr)
    if (.not. Q%reported) then                            ! the first build: its size (MUMPS's flops are not logged)
      Q%reported = .true.
      call KSPGetPC(Q%ksp, pcl, ierr)
      call PCFactorGetMatrix(pcl, fm, ierr)
      call MatMumpsGetInfog(fm, 29_8, iv, ierr)
      call MatMumpsGetRinfog(fm, 3_8, rv, ierr)
      if (Q%me == 0) write(*,'(A,I0,A,ES9.2,A,ES9.2)') "[Physics PC]   AS: augmented matrix "//trim(label)// &
        " [[A, L], [R, -K]], ", Q%aoff(Q%np), " rows, MUMPS LU entries ", dble(iv), ", flops ", rv
    endif
  contains
    !> the row nrow's entries cols (space 1: A's columns, 2: K's) times f;
    !! with the family filter only those of the row's family rfam
    subroutine put(space, nc, cl, vl, f)
      integer, intent(in) :: space
      PetscInt, intent(in) :: nc
      PetscInt, intent(in) :: cl(:)
      PetscScalar, intent(in) :: vl(:)
      real*8, intent(in) :: f
      integer :: a_, cf
      PetscInt :: g_, n_
      if (nc == 0) return
      if (allocated(ac)) deallocate(ac, av)
      allocate(ac(nc), av(nc))
      n_ = 0
      do a_ = 1, int(nc)
        if (Q%ns > 0) then
          if (space == 1) then
            cf = Q%fam(int(mod(cl(a_), int(Q%ns, kind(nc)))))
          else
            cf = Q%fam(int(cl(a_)) / Q%nka)
          endif
          if (cf /= rfam) cycle
        endif
        n_ = n_ + 1
        ac(n_) = aug_col(space, cl(a_)); av(n_) = f * vl(a_)
      enddo
      if (n_ == 0) return
      if (pass == 1) then
        g_ = nrow - c0 + 1
        dnz(g_) = dnz(g_) + count(ac(1:n_) >= c0 .and. ac(1:n_) < c1)
        onz(g_) = onz(g_) + count(ac(1:n_) < c0 .or. ac(1:n_) >= c1)
      else
        call MatSetValues(Q%aug, one, [nrow], n_, ac(1:n_), av(1:n_), INSERT_VALUES, ierr)
      endif
    end subroutine put
    !> the augmented column of column c of space 1 (A) or 2 (K)
    PetscInt function aug_col(space, c)
      integer, intent(in) :: space
      PetscInt, intent(in) :: c
      integer :: lo, hi, md
      lo = 0; hi = Q%np - 1
      if (space == 1) then
        do while (lo < hi)                                ! the owner: ra(q) <= c < ra(q + 1)
          md = (lo + hi + 1) / 2
          if (Q%ra(md) <= c) then
            lo = md
          else
            hi = md - 1
          endif
        enddo
        aug_col = Q%aoff(lo) + (c - Q%ra(lo))
      else                                                ! K's columns: all on the K rank
        lo = axk(Q%ik)%kr
        aug_col = Q%aoff(lo) + (Q%ra(lo + 1) - Q%ra(lo)) + (c - Q%rk(lo))
      endif
    end function aug_col
  end subroutine axa_setup

  !> x = the first block of [[A, L], [R, -K]]^-1 [b; 0] = P^-1 b (A's layout).
  !! Collective.
  subroutine axa_solve(ia, b, x)
    integer, intent(in) :: ia
    Vec :: b, x
    type(axaug_t), pointer :: Q
    PetscScalar, pointer :: pa(:), pb(:)
    PetscErrorCode :: ierr
    Q => axaug(ia)
    call VecGetArrayRead(b, pb, ierr)
    call VecGetArray(Q%ab, pa, ierr)
    pa(1:Q%na) = pb(1:Q%na)
    if (Q%nkl > 0) pa(Q%na + 1 : Q%na + Q%nkl) = 0.d0
    call VecRestoreArray(Q%ab, pa, ierr)
    call VecRestoreArrayRead(b, pb, ierr)
    call KSPSolve(Q%ksp, Q%ab, Q%ax, ierr)
    call VecGetArrayRead(Q%ax, pa, ierr)
    call VecGetArray(x, pb, ierr)
    pb(1:Q%na) = pa(1:Q%na)
    call VecRestoreArray(x, pb, ierr)
    call VecRestoreArrayRead(Q%ax, pa, ierr)
  end subroutine axa_solve

  !> The entries of augmented matrix ia's LU factors (MUMPS, INFOG(29) when
  !! available, else 0), for the report. Collective.
  real*8 function axa_factor_nnz(ia)
    integer, intent(in) :: ia
    PC :: pcl
    Mat :: F
    PetscInt :: iv
    PetscErrorCode :: ierr
    call KSPGetPC(axaug(ia)%ksp, pcl, ierr)
    call PCFactorGetMatrix(pcl, F, ierr)
    call MatMumpsGetInfog(F, 29_8, iv, ierr)
    axa_factor_nnz = dble(iv)
  end function axa_factor_nnz

  !> pc (a PCSHELL) applies augmented matrix ia's solve (axa_pcapply).
  subroutine axa_set_pc(ia, pc)
    integer, intent(in) :: ia
    PC, intent(in) :: pc
    PetscErrorCode :: ierr
    axaug(ia)%pc = pc
    axaug(ia)%has_pc = .true.
    call PCShellSetApply(pc, axa_pcapply, ierr)
  end subroutine axa_set_pc

  !> The PCSHELL apply of axa_set_pc: y = P^-1 x.
  subroutine axa_pcapply(pc, x, y, ierr)
    PC :: pc
    Vec :: x, y
    PetscErrorCode :: ierr
    integer :: i
    do i = 1, MAXAUG
      if (.not. axaug(i)%has_pc) cycle
      if (axaug(i)%pc%v == pc%v) then
        call axa_solve(i, x, y)
        ierr = 0
        return
      endif
    enddo
    call fail("a PCSHELL of no augmented matrix")
    ierr = 0
  end subroutine axa_pcapply

  subroutine fail(msg)
    character(len=*), intent(in) :: msg
    integer :: mpierr
    write(*,'(A,A)') "[Physics PC]   FATAL: axis Schur complement: ", trim(msg)
    call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
  end subroutine fail

#endif
end module mod_petsc_pc_axis_schur
