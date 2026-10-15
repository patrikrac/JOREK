module mod_petsc_pc_sf_spai
#ifdef USE_PETSC
  use iso_c_binding
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_raw_csr, only: get_ij, put_ij, aij_vals_read, aij_vals_done
  implicit none
  private

  !--------------------------------------------------------------------
  !> Banded sparse approximate inverse C ~ M^-1 of a real symmetric positive
  !! definite AIJ matrix M (SPAI, Grote & Huckle, SIAM J. Sci. Comput. 18
  !! (1997) 838, on a fixed a-priori pattern): per column k
  !!     min || M_s(I,J) c_k - e_k(I) ||_2   over the pattern J of column k,
  !! then symmetrize, C <- (C + C^T) / 2.
  !!
  !! PATTERN. J(k) = the structure of column k of M^r (M^0 = I): the nodes at
  !! graph distance <= r from k in the adjacency graph of M. Explicit zeros
  !! of M are NOT edges -- the Dirichlet rows and columns replaced by the
  !! identity keep their stored zeros, and an isolated Dirichlet row then
  !! gives J = I = {k}, A = [1], c = e_k exactly. The graph is fixed at
  !! spai_build from the values of that build (zero there = structurally zero);
  !! a refill counts and reports any later nonzero outside it (ignored).
  !! I(k) = the rows touching J (the neighbours of J), so M(:,J) is zero off
  !! I and the least-squares problem is the full column problem. The radius
  !! is purely algebraic: no geometry, any number of uncoupled toroidal
  !! slots (their blocks never meet in the graph).
  !!
  !! SCALING (default on). The minimisation runs on M_s = D^-1/2 M D^-1/2,
  !! D = diag(M), and C = D^-1/2 C_s D^-1/2. The unscaled Frobenius
  !! minimisation is NOT scale-invariant: next to the identity Dirichlet
  !! rows (O(1)) the bicubic Hermite mass has entries down to ~1e-10, and the
  !! unscaled C is indefinite. numpy prototype, 2-D bicubic Hermite mass on
  !! the unit square, u = 0 Dirichlet rows/cols -> identity, r = 2:
  !!   unscaled: sigma(C M) in [-10.1, 11.1] (ne 8), [-14.2, 15.2] (ne 14)
  !!   scaled:   sigma(C M) in [0.736, 1.159] (ne 8), [0.724, 1.167] (ne 14),
  !! also with a 20x element weight and two uncoupled slots of different
  !! scale (r = 1: [0.36, 1.53]; r = 3: [0.88, 1.07]). The scaling keeps the
  !! Dirichlet rows exact (d = 1).
  !!
  !! LEAST SQUARES. Column k's M(I,J) is read as M(J,I)^T from the rows of J
  !! (M symmetric), so the rank needs the rows within distance r of its own
  !! rows only: they are gathered once per build/refill by
  !! MatCreateSubMatrices (rows G = ball_r(local rows), all columns; the
  !! diagonal on X = G + its neighbours comes by a VecScatter). Each local
  !! row k solves its own column (C's row k = the solve for column k; the
  !! symmetrization makes rows and columns equal) by Householder QR, LAPACK
  !! dgels: M(I,J) has full column rank for nonsingular M (the columns J of
  !! M vanish off I), so no rank-revealing dgelsd is needed; a dgels
  !! breakdown (info > 0, an exactly singular R) falls back to the Jacobi
  !! column c = e_k / M_kk and is counted. Columns run on the OpenMP threads,
  !! each with private workspaces; one row's |J| ~ 100, |I| ~ 200 for r = 2,
  !! 4 DOFs per node.
  !!
  !! STRUCTURE ONCE, NUMBERS PER REFILL. spai_build fixes the gather index
  !! sets, the graph, the patterns and C (MPIAIJ, M's row layout), then
  !! fills it; spai_refill (new values of M on the same structure) redoes the
  !! gather (MAT_INITIAL_MATRIX, as in mod_petsc_pc_sf_pairw: a
  !! MAT_REUSE_MATRIX gather returned stale values at np > 1), the
  !! least-squares solves and the values of C in place -- C keeps its handle.
  !! M is never modified.
  !--------------------------------------------------------------------

  PetscInt, parameter :: zero = 0, one = 1
  integer, parameter  :: KSP_INDEF_PC = -8   !< KSP_DIVERGED_INDEFINITE_PC (petscksp.h)

  type, public :: spai_t
    logical :: built  = .false.           !< C, the pattern and the maps are valid
    logical :: scaled = .true.            !< SPAI of D^-1/2 M D^-1/2 (D = diag M)
    integer :: r      = 2                 !< pattern radius: the structure of M^r
    integer :: comm   = MPI_COMM_NULL     !< M's communicator
    Mat :: C                              !< (Craw + Craw^T) / 2, M's row layout
    Mat :: Craw                           !< row k = the least-squares column k
    Mat :: Ct                             !< Craw^T
    IS  :: isrow                          !< the gathered rows G (sorted, global)
    IS  :: iscol                          !< all columns
    IS  :: isx                            !< X = G and its neighbours (sorted, global)
    Vec :: dg                             !< diag(M), M's layout
    Vec :: dx                             !< diag(M) on X (sequential)
    VecScatter :: scx                     !< dg -> dx
    PetscInt :: rs = 0, nl = 0, nglob = 0 !< first local row, local rows, global size
    integer :: ng = 0, nx = 0             !< |G|, |X|
    integer :: g0 = 0                     !< G position of the first local row
    integer(8) :: nzg = 0                 !< stored entries of the gathered rows
    PetscInt, allocatable :: xg(:)        !< X position -> global index
    integer, allocatable :: xgm(:)        !< X position -> G position (0: not gathered)
    integer, allocatable :: gx(:)         !< G position -> X position
    integer(8), allocatable :: gptr(:)    !< gathered CSR row offsets (0-based), ng + 1
    integer, allocatable :: xcol(:)       !< gathered entries' columns, X positions
    logical, allocatable :: keep(:)       !< gathered entries that are graph edges
    integer(8), allocatable :: jptr(:)    !< local row p's pattern: jx(jptr(p)+1:jptr(p+1))
    integer, allocatable :: jx(:)         !< patterns, X positions (ascending)
    real*8, allocatable :: cval(:)        !< Craw's values, in jx's order
    integer :: maxi = 0, maxj = 0         !< largest |I|, |J| of the rank
    real*8  :: sum_i = 0.d0, sum_j = 0.d0 !< sums of |I|, |J| over the rank's rows
    integer :: nfail = 0                  !< last fill: dgels fallbacks (global)
    integer :: ndrop = 0                  !< last fill: nonzeros off the graph (global)
    integer :: nbadd = 0                  !< last fill: diagonals <= 0 (scaling skipped, global)
    real*8  :: res_max = 0.d0             !< last fill: max ||M_s c_k - e_k||_2 (global)
    real*8  :: res_avg = 0.d0             !< last fill: mean of the same (global)
  end type spai_t

  public :: spai_build, spai_refill, spai_check, spai_destroy

contains

  !--------------------------------------------------------------------
  !> First build: the gather index sets, the graph, the patterns of radius
  !! r and C with its values. Collective on M's communicator. A built s is
  !! destroyed first.
  !--------------------------------------------------------------------
  subroutine spai_build(s, M, r, ierr, scaled, verbose, label)
    type(spai_t), intent(inout)            :: s        !< the approximate inverse
    Mat, intent(in)                        :: M        !< SPD AIJ matrix (not modified)
    integer, intent(in)                    :: r        !< pattern radius (>= 0): structure of M^r
    PetscErrorCode, intent(out)            :: ierr     !< 0 on success
    logical, intent(in), optional          :: scaled   !< Jacobi-scaled SPAI (default .true.)
    logical, intent(in), optional          :: verbose  !< rank-0 summary (default .true.)
    character(len=*), intent(in), optional :: label    !< message prefix (default "SPAI")
    Mat, pointer :: sub(:)
    PetscInt, pointer :: ia(:), ja(:)
    real(c_double), pointer :: v(:)
    type(c_ptr) :: hv
    PetscInt, allocatable :: h(:), tmp(:), grow(:), dnz(:), onz(:)
    PetscInt :: nr, i, cs, ce, nlc, n_
    IS :: ist
    MatType :: mt
    integer :: lev, nu, g, rank, nbad, mierr
    integer(8) :: e, nz
    logical :: verb
    real*8 :: tot(3), mx(2)

    ierr = 0
    if (s%built) then
      call spai_destroy(s, ierr); if (ierr /= 0) return
    endif
    if (r < 0) then
      ierr = 1
      write(*,'(A,I0)') "[Physics PC]   SPAI: invalid pattern radius ", r
      return
    endif
    s%r = r
    s%scaled = .true.
    if (present(scaled)) s%scaled = scaled
    verb = .true.
    if (present(verbose)) verb = verbose

    call PetscObjectGetComm(M, s%comm, ierr); if (ierr /= 0) return
    call MPI_Comm_rank(s%comm, rank, mierr)
    call MatGetType(M, mt, ierr); if (ierr /= 0) return
    if (index(mt, "aij") == 0 .or. index(mt, "baij") > 0) then
      ierr = 1
      if (rank == 0) write(*,'(A,A)') "[Physics PC]   SPAI: needs an AIJ matrix, got ", trim(mt)
      return
    endif
    call MatGetOwnershipRange(M, s%rs, i, ierr); if (ierr /= 0) return
    s%nl = i - s%rs
    call MatGetOwnershipRangeColumn(M, cs, ce, ierr); if (ierr /= 0) return
    nlc = ce - cs
    call MatGetSize(M, s%nglob, PETSC_NULL_INTEGER, ierr); if (ierr /= 0) return
    call ISCreateStride(PETSC_COMM_SELF, s%nglob, zero, one, s%iscol, ierr); if (ierr /= 0) return

    !--- G = ball_r(local rows): r halo expansions over the full graph of M
    h = [(s%rs + i, i = 0, s%nl - 1)]
    do lev = 1, r
      call ISCreateGeneral(PETSC_COMM_SELF, int(size(h), kind(nr)), h, PETSC_COPY_VALUES, ist, ierr)
      if (ierr /= 0) return
      call MatCreateSubMatrices(M, one, [ist], [s%iscol], MAT_INITIAL_MATRIX, sub, ierr)
      if (ierr /= 0) return
      call get_ij(sub(1), .false., nr, ia, ja)
      nz = int(ia(nr + 1), 8)
      allocate(tmp(size(h) + nz))
      tmp(1:size(h)) = h
      tmp(size(h) + 1:) = ja(1:nz)
      call put_ij(sub(1), .false., nr, ia, ja)
      call MatDestroySubMatrices(one, sub, ierr); if (ierr /= 0) return
      call ISDestroy(ist, ierr); if (ierr /= 0) return
      call sort_uniq(tmp, nu)
      h = tmp(1:nu)
      deallocate(tmp)
    enddo
    grow = h
    s%ng = size(grow)
    call ISCreateGeneral(PETSC_COMM_SELF, int(s%ng, kind(nr)), grow, PETSC_COPY_VALUES, s%isrow, ierr)
    if (ierr /= 0) return

    !--- the gathered rows: X, the position maps and the graph
    call MatCreateSubMatrices(M, one, [s%isrow], [s%iscol], MAT_INITIAL_MATRIX, sub, ierr)
    if (ierr /= 0) return
    call get_ij(sub(1), .false., nr, ia, ja)
    s%nzg = int(ia(nr + 1), 8)
    call aij_vals_read(sub(1), s%nzg, hv, v)
    allocate(tmp(s%ng + s%nzg))
    tmp(1:s%ng) = grow
    tmp(s%ng + 1:) = ja(1:s%nzg)
    call sort_uniq(tmp, nu)
    s%nx = nu
    s%xg = tmp(1:nu)
    deallocate(tmp)
    allocate(s%xgm(s%nx), s%gx(s%ng), s%gptr(s%ng + 1), s%xcol(s%nzg), s%keep(s%nzg))
    s%xgm = 0
    do g = 1, s%ng
      s%gx(g) = bsearch(s%xg, grow(g))
      s%xgm(s%gx(g)) = g
      s%gptr(g) = int(ia(g), 8)
    enddo
    s%gptr(s%ng + 1) = s%nzg
    nbad = 0
    do g = 1, s%ng
      do e = s%gptr(g) + 1, s%gptr(g + 1)
        s%xcol(e) = bsearch(s%xg, ja(e))
        if (s%xcol(e) == 0) nbad = nbad + 1
        s%keep(e) = (v(e) /= 0.d0) .or. (ja(e) == grow(g))   ! the diagonal always
      enddo
    enddo
    call aij_vals_done(sub(1), hv)
    call put_ij(sub(1), .false., nr, ia, ja)
    call MatDestroySubMatrices(one, sub, ierr); if (ierr /= 0) return
    s%g0 = 0
    if (s%nl > 0) s%g0 = bsearch(grow, s%rs)

    !--- the diagonal on X
    call ISCreateGeneral(PETSC_COMM_SELF, int(s%nx, kind(nr)), s%xg, PETSC_COPY_VALUES, s%isx, ierr)
    if (ierr /= 0) return
    call MatCreateVecs(M, PETSC_NULL_VEC, s%dg, ierr); if (ierr /= 0) return   ! row layout
    call VecCreateSeq(PETSC_COMM_SELF, int(s%nx, kind(nr)), s%dx, ierr); if (ierr /= 0) return
    call VecScatterCreate(s%dg, s%isx, s%dx, PETSC_NULL_IS, s%scx, ierr); if (ierr /= 0) return

    !--- the patterns, then Craw preallocated exactly
    call make_pattern(s, nbad)
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, s%comm, mierr)
    if (nbad > 0) then
      ierr = 1
      if (rank == 0) write(*,'(A,I0,A)') "[Physics PC]   SPAI: internal error, ", nbad, &
        " pattern nodes outside the gathered rows"
      return
    endif
    allocate(dnz(s%nl), onz(s%nl))
    do i = 1, s%nl
      n_ = int(s%jptr(i + 1) - s%jptr(i), kind(n_))
      dnz(i) = int(count(s%xg(s%jx(s%jptr(i) + 1:s%jptr(i + 1))) >= cs .and. &
                         s%xg(s%jx(s%jptr(i) + 1:s%jptr(i + 1))) <  ce), kind(n_))
      onz(i) = n_ - dnz(i)
    enddo
    call MatCreate(s%comm, s%Craw, ierr); if (ierr /= 0) return
    call MatSetSizes(s%Craw, s%nl, nlc, s%nglob, s%nglob, ierr); if (ierr /= 0) return
    call MatSetType(s%Craw, MATMPIAIJ, ierr); if (ierr /= 0) return
    call MatMPIAIJSetPreallocation(s%Craw, zero, dnz, zero, onz, ierr); if (ierr /= 0) return
    call MatSetOption(s%Craw, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE, ierr); if (ierr /= 0) return
    allocate(s%cval(s%jptr(s%nl + 1)))

    !--- values, then C = (Craw + Craw^T) / 2 on the union pattern
    call fill_values(s, M, ierr); if (ierr /= 0) return
    call MatSetOption(s%Craw, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr); if (ierr /= 0) return
    call MatTranspose(s%Craw, MAT_INITIAL_MATRIX, s%Ct, ierr); if (ierr /= 0) return
    call MatDuplicate(s%Craw, MAT_COPY_VALUES, s%C, ierr); if (ierr /= 0) return
    call MatAXPY(s%C, 1.0d0, s%Ct, DIFFERENT_NONZERO_PATTERN, ierr); if (ierr /= 0) return
    call MatScale(s%C, 0.5d0, ierr); if (ierr /= 0) return
    s%built = .true.

    if (verb) then
      tot = [s%sum_j, s%sum_i, dble(s%nl)]
      mx  = [dble(s%maxj), dble(s%maxi)]
      call MPI_Allreduce(MPI_IN_PLACE, tot, 3, MPI_DOUBLE_PRECISION, MPI_SUM, s%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, mx, 2, MPI_DOUBLE_PRECISION, MPI_MAX, s%comm, mierr)
      if (rank == 0) write(*,'(A,A,A,I0,A,L1,A,F7.1,A,I0,A,F7.1,A,I0,A,2ES10.3,A,I0,A,I0)') &
        "[Physics PC]   ", lbl(label), ": SPAI r = ", s%r, ", scaled ", s%scaled, &
        ", |J| avg ", tot(1) / max(tot(3), 1.d0), " max ", nint(mx(1)), &
        ", |I| avg ", tot(2) / max(tot(3), 1.d0), " max ", nint(mx(2)), &
        ", LS residual avg / max ", s%res_avg, s%res_max, ", fallbacks ", s%nfail, &
        ", diag <= 0 ", s%nbadd
    endif
  end subroutine spai_build

  !--------------------------------------------------------------------
  !> New values of M on the structure of the build: the least-squares solves
  !! and C's values again, in place. Collective on M's communicator.
  !--------------------------------------------------------------------
  subroutine spai_refill(s, M, ierr, verbose, label)
    type(spai_t), intent(inout)            :: s        !< a built approximate inverse
    Mat, intent(in)                        :: M        !< the matrix of the build, new values
    PetscErrorCode, intent(out)            :: ierr     !< 0 on success
    logical, intent(in), optional          :: verbose  !< rank-0 summary (default .false.)
    character(len=*), intent(in), optional :: label    !< message prefix (default "SPAI")
    integer :: rank, mierr

    ierr = 0
    if (.not. s%built) then
      ierr = 1
      write(*,'(A)') "[Physics PC]   SPAI: spai_refill before spai_build"
      return
    endif
    call fill_values(s, M, ierr); if (ierr /= 0) return
    call MatTranspose(s%Craw, MAT_REUSE_MATRIX, s%Ct, ierr); if (ierr /= 0) return
    call MatZeroEntries(s%C, ierr); if (ierr /= 0) return
    call MatAXPY(s%C, 0.5d0, s%Craw, SUBSET_NONZERO_PATTERN, ierr); if (ierr /= 0) return
    call MatAXPY(s%C, 0.5d0, s%Ct, SUBSET_NONZERO_PATTERN, ierr); if (ierr /= 0) return

    if (present(verbose)) then
      if (verbose) then
        call MPI_Comm_rank(s%comm, rank, mierr)
        if (rank == 0) write(*,'(A,A,A,2ES10.3,A,I0,A,I0,A,I0)') "[Physics PC]   ", lbl(label), &
          ": SPAI refill, LS residual avg / max ", s%res_avg, s%res_max, ", fallbacks ", s%nfail, &
          ", diag <= 0 ", s%nbadd, ", dropped nonzeros ", s%ndrop
      endif
    endif
  end subroutine spai_refill

  !--------------------------------------------------------------------
  !> Definiteness check of C: the Gershgorin lower bound of C's spectrum,
  !! min_i (c_ii - sum_{j /= i} |c_ij|), and Lanczos estimates of the
  !! extreme eigenvalues of C M from one CG solve of M with C as the
  !! preconditioner (PCMAT, random right-hand side, nits iterations at
  !! most). A CG that stops on an indefinite preconditioner (some r with
  !! r^T C r <= 0, a certificate that C is not positive definite) reports
  !! lam_min <= 0. Warns on rank 0 when both tests fail. The Gershgorin bound
  !! of an approximate inverse mass is normally negative (its off-diagonals
  !! are large); the eigen-estimate is the test that matters. Collective.
  !--------------------------------------------------------------------
  subroutine spai_check(s, M, gersh_min, lam_min, lam_max, ierr, nits, verbose, label)
    type(spai_t), intent(inout)            :: s          !< a built approximate inverse
    Mat, intent(in)                        :: M          !< the matrix C approximates the inverse of
    real*8, intent(out)                    :: gersh_min  !< Gershgorin lower bound of sigma(C)
    real*8, intent(out)                    :: lam_min    !< estimate of lambda_min(C M)
    real*8, intent(out)                    :: lam_max    !< estimate of lambda_max(C M)
    PetscErrorCode, intent(out)            :: ierr       !< 0 on success
    integer, intent(in), optional          :: nits       !< CG iterations at most (default 100)
    logical, intent(in), optional          :: verbose    !< rank-0 summary (default .false.)
    character(len=*), intent(in), optional :: label      !< message prefix (default "SPAI")
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt :: i, nc, its, maxit
    PetscReal :: emax, emin
    KSP :: est
    PC  :: pc
    Vec :: b, x
    PetscRandom :: rnd
    KSPConvergedReason :: reason
    real*8 :: dg, off
    integer :: q, rank, mierr
    logical :: indef

    ierr = 0
    gersh_min = 0.d0; lam_min = 0.d0; lam_max = 0.d0
    if (.not. s%built) then
      ierr = 1
      write(*,'(A)') "[Physics PC]   SPAI: spai_check before spai_build"
      return
    endif
    call MPI_Comm_rank(s%comm, rank, mierr)

    !--- Gershgorin
    gersh_min = huge(1.d0)
    do i = s%rs, s%rs + s%nl - 1
      call MatGetRow(s%C, i, nc, cols, vals, ierr); if (ierr /= 0) return
      dg = 0.d0; off = 0.d0
      do q = 1, int(nc)
        if (cols(q) == i) then
          dg = vals(q)
        else
          off = off + abs(vals(q))
        endif
      enddo
      call MatRestoreRow(s%C, i, nc, cols, vals, ierr); if (ierr /= 0) return
      gersh_min = min(gersh_min, dg - off)
    enddo
    call MPI_Allreduce(MPI_IN_PLACE, gersh_min, 1, MPI_DOUBLE_PRECISION, MPI_MIN, s%comm, mierr)

    !--- Lanczos bounds of C M from a CG on M, preconditioner C
    maxit = 100
    if (present(nits)) maxit = int(max(nits, 1), kind(maxit))
    call KSPCreate(s%comm, est, ierr); if (ierr /= 0) return
    call KSPSetOperators(est, M, s%C, ierr); if (ierr /= 0) return
    call KSPSetType(est, KSPCG, ierr); if (ierr /= 0) return
    call KSPGetPC(est, pc, ierr); if (ierr /= 0) return
    call PCSetType(pc, PCMAT, ierr); if (ierr /= 0) return
    call KSPSetComputeSingularValues(est, PETSC_TRUE, ierr); if (ierr /= 0) return
    call KSPSetTolerances(est, 1.d-12, 1.d-50, 1.d10, maxit, ierr); if (ierr /= 0) return
    call KSPSetUp(est, ierr); if (ierr /= 0) return
    call MatCreateVecs(M, x, b, ierr); if (ierr /= 0) return
    call PetscRandomCreate(s%comm, rnd, ierr); if (ierr /= 0) return
    call VecSetRandom(b, rnd, ierr); if (ierr /= 0) return
    call KSPSolve(est, b, x, ierr); if (ierr /= 0) return
    call KSPGetConvergedReason(est, reason, ierr); if (ierr /= 0) return
    call KSPGetIterationNumber(est, its, ierr); if (ierr /= 0) return
    call KSPComputeExtremeSingularValues(est, emax, emin, ierr); if (ierr /= 0) return
    lam_min = emin; lam_max = emax          ! CG: the Lanczos matrix's extreme eigenvalues
    indef = (reason%v == KSP_INDEF_PC)
    if (indef) lam_min = min(lam_min, 0.d0)
    call PetscRandomDestroy(rnd, ierr); if (ierr /= 0) return
    call VecDestroy(b, ierr); if (ierr /= 0) return
    call VecDestroy(x, ierr); if (ierr /= 0) return
    call KSPDestroy(est, ierr); if (ierr /= 0) return

    if (rank == 0) then
      if (gersh_min <= 0.d0 .and. lam_min <= 0.d0) then
        write(*,'(A,A,A,ES10.3,A,2ES10.3,A,L1,A)') "[Physics PC]   WARNING: ", lbl(label), &
          ": SPAI C not positive definite: Gershgorin bound ", gersh_min, ", lambda(C M) in ", &
          lam_min, lam_max, " (indefinite PC in CG: ", indef, ")"
      else if (present(verbose)) then
        if (verbose) write(*,'(A,A,A,ES10.3,A,2ES10.3,A,F8.3,A,I0,A,I0)') "[Physics PC]   ", lbl(label), &
          ": SPAI check: Gershgorin bound ", gersh_min, ", lambda(C M) in ", lam_min, lam_max, &
          ", kappa ", lam_max / max(lam_min, tiny(1.d0)), ", CG its ", its, ", reason ", reason%v
      endif
    endif
  end subroutine spai_check

  !--------------------------------------------------------------------
  !> Release C and everything spai_build created. Collective (the PETSc
  !! objects live on M's communicator).
  !--------------------------------------------------------------------
  subroutine spai_destroy(s, ierr)
    type(spai_t), intent(inout) :: s     !< the approximate inverse
    PetscErrorCode, intent(out) :: ierr  !< 0 on success

    ierr = 0
    if (.not. s%built) return
    call MatDestroy(s%C, ierr); if (ierr /= 0) return
    call MatDestroy(s%Craw, ierr); if (ierr /= 0) return
    call MatDestroy(s%Ct, ierr); if (ierr /= 0) return
    call ISDestroy(s%isrow, ierr); if (ierr /= 0) return
    call ISDestroy(s%iscol, ierr); if (ierr /= 0) return
    call ISDestroy(s%isx, ierr); if (ierr /= 0) return
    call VecScatterDestroy(s%scx, ierr); if (ierr /= 0) return
    call VecDestroy(s%dg, ierr); if (ierr /= 0) return
    call VecDestroy(s%dx, ierr); if (ierr /= 0) return
    if (allocated(s%xg))   deallocate(s%xg)
    if (allocated(s%xgm))  deallocate(s%xgm)
    if (allocated(s%gx))   deallocate(s%gx)
    if (allocated(s%gptr)) deallocate(s%gptr)
    if (allocated(s%xcol)) deallocate(s%xcol)
    if (allocated(s%keep)) deallocate(s%keep)
    if (allocated(s%jptr)) deallocate(s%jptr)
    if (allocated(s%jx))   deallocate(s%jx)
    if (allocated(s%cval)) deallocate(s%cval)
    s%ng = 0; s%nx = 0; s%nzg = 0; s%nl = 0; s%maxi = 0; s%maxj = 0
    s%built = .false.
  end subroutine spai_destroy


  !--------------------------------------------------------------------
  !> The patterns J (jptr, jx: X positions, ascending) of the local rows,
  !! and the sizes of their I sets (maxi, maxj, sum_i, sum_j). Two passes on
  !! the OpenMP threads: the sizes, then the patterns.
  !--------------------------------------------------------------------
  subroutine make_pattern(s, nbad)
    type(spai_t), intent(inout) :: s     !< gathered graph in, patterns out
    integer, intent(inout)      :: nbad  !< + pattern nodes outside G (internal error)
    integer, allocatable :: markj(:), marki(:), lst(:), nj(:), ni(:)
    integer :: p, n, nn, b, g, x, nl, nb
    integer(8) :: e

    nl = int(s%nl)
    allocate(nj(nl), ni(nl), s%jptr(nl + 1))
    nb = 0

    !$omp parallel private(markj, marki, lst, p, n, nn, b, g, x, e)
    allocate(markj(s%nx), marki(s%nx), lst(s%nx))
    markj = 0; marki = 0
    !$omp do schedule(dynamic, 64) reduction(+:nb)
    do p = 1, nl
      call ball(s, p, markj, lst, n, nb)
      nn = 0
      do b = 1, n
        g = s%xgm(lst(b))
        if (g == 0) cycle
        do e = s%gptr(g) + 1, s%gptr(g + 1)
          if (.not. s%keep(e)) cycle
          x = s%xcol(e)
          if (marki(x) /= p) then
            marki(x) = p; nn = nn + 1
          endif
        enddo
      enddo
      if (marki(lst(1)) /= p) nn = nn + 1   ! k itself (fill_values adds it without a diagonal)
      nj(p) = n; ni(p) = nn
    enddo
    !$omp end do
    deallocate(markj, marki, lst)
    !$omp end parallel

    s%jptr(1) = 0
    do p = 1, nl
      s%jptr(p + 1) = s%jptr(p) + nj(p)
    enddo
    allocate(s%jx(s%jptr(nl + 1)))

    !$omp parallel private(markj, lst, p, n, nn)
    allocate(markj(s%nx), lst(s%nx))
    markj = 0
    nn = 0
    !$omp do schedule(dynamic, 64)
    do p = 1, nl
      call ball(s, p, markj, lst, n, nn)    ! outside-G nodes counted in pass 1 already
      call isort(lst, n)
      s%jx(s%jptr(p) + 1:s%jptr(p + 1)) = lst(1:n)
    enddo
    !$omp end do
    deallocate(markj, lst)
    !$omp end parallel

    nbad = nbad + nb
    s%maxj = 0; s%maxi = 0
    if (nl > 0) then
      s%maxj = maxval(nj); s%maxi = maxval(ni)
    endif
    s%sum_j = sum(dble(nj)); s%sum_i = sum(dble(ni))
  end subroutine make_pattern

  !> Local row p's ball of radius s%r in the graph, X positions into
  !! lst(1:n) (unsorted). mark(:) is the caller's stamp array (stamp p).
  subroutine ball(s, p, mark, lst, n, nbad)
    type(spai_t), intent(in) :: s        !< gathered graph
    integer, intent(in)      :: p        !< local row (1-based)
    integer, intent(inout)   :: mark(:)  !< stamps over X
    integer, intent(inout)   :: lst(:)   !< the ball, X positions
    integer, intent(out)     :: n        !< its size
    integer, intent(inout)   :: nbad     !< + frontier nodes outside G
    integer :: lev, lo, hi, q, g, x
    integer(8) :: e

    n = 1
    lst(1) = s%gx(s%g0 + p - 1)
    mark(lst(1)) = p
    lo = 1
    do lev = 1, s%r
      hi = n
      do q = lo, hi
        g = s%xgm(lst(q))
        if (g == 0) then
          nbad = nbad + 1
          cycle
        endif
        do e = s%gptr(g) + 1, s%gptr(g + 1)
          if (.not. s%keep(e)) cycle
          x = s%xcol(e)
          if (mark(x) /= p) then
            mark(x) = p; n = n + 1; lst(n) = x
          endif
        enddo
      enddo
      lo = hi + 1
    enddo
  end subroutine ball

  !--------------------------------------------------------------------
  !> Gather the current values, solve every local column's least-squares
  !! problem (OpenMP), write Craw's values and assemble it. Collective.
  !--------------------------------------------------------------------
  subroutine fill_values(s, M, ierr)
    type(spai_t), intent(inout) :: s     !< structure in, values out
    Mat, intent(in)             :: M     !< the matrix
    PetscErrorCode, intent(out) :: ierr  !< 0 on success
    Mat, pointer :: sub(:)
    PetscInt, pointer :: ia(:), ja(:)
    real(c_double), pointer :: v(:)
    PetscScalar, pointer :: pd(:)
    type(c_ptr) :: hv
    PetscInt :: nr, nj_
    PetscInt, allocatable :: cols(:)
    real*8, allocatable :: dsc(:), a(:, :), bb(:), work(:)
    integer, allocatable :: mark(:), pos(:), jl(:)
    real*8 :: wq(1), den, rmax, rsum, cres
    integer :: g, x, p, nj, ni, b, xk, pk, info, lwork, ldb, nfail, ndrop, nbadd, bad, mierr, nl
    integer(8) :: e, j0

    ierr = 0
    nl = int(s%nl)

    !--- the gathered rows, checked against the frozen structure
    call MatCreateSubMatrices(M, one, [s%isrow], [s%iscol], MAT_INITIAL_MATRIX, sub, ierr)
    if (ierr /= 0) return
    call get_ij(sub(1), .false., nr, ia, ja)
    bad = 0
    if (int(nr) /= s%ng) then
      bad = 1
    else if (int(ia(nr + 1), 8) /= s%nzg) then
      bad = 1
    else
      do g = 1, s%ng
        if (int(ia(g), 8) /= s%gptr(g)) then
          bad = 1; exit
        endif
        do e = s%gptr(g) + 1, s%gptr(g + 1)
          if (ja(e) /= s%xg(s%xcol(e))) then
            bad = 1; exit
          endif
        enddo
        if (bad /= 0) exit
      enddo
    endif
    call MPI_Allreduce(MPI_IN_PLACE, bad, 1, MPI_INTEGER, MPI_MAX, s%comm, mierr)
    if (bad /= 0) then
      call put_ij(sub(1), .false., nr, ia, ja)
      call MatDestroySubMatrices(one, sub, ierr)
      ierr = 1
      call MPI_Comm_rank(s%comm, x, mierr)
      if (x == 0) write(*,'(A)') "[Physics PC]   SPAI: the nonzero structure of M changed since spai_build"
      return
    endif
    call aij_vals_read(sub(1), s%nzg, hv, v)
    ndrop = 0
    do e = 1, s%nzg
      if (.not. s%keep(e) .and. v(e) /= 0.d0) ndrop = ndrop + 1
    enddo

    !--- the scaling D^-1/2 on X
    call MatGetDiagonal(M, s%dg, ierr); if (ierr /= 0) return
    call VecScatterBegin(s%scx, s%dg, s%dx, INSERT_VALUES, SCATTER_FORWARD, ierr); if (ierr /= 0) return
    call VecScatterEnd(s%scx, s%dg, s%dx, INSERT_VALUES, SCATTER_FORWARD, ierr); if (ierr /= 0) return
    allocate(dsc(s%nx))
    dsc = 1.d0
    nbadd = 0
    if (s%scaled) then
      call VecGetArrayRead(s%dx, pd, ierr); if (ierr /= 0) return
      do x = 1, s%nx
        if (pd(x) > 0.d0) then
          dsc(x) = 1.d0 / sqrt(pd(x))
        else
          nbadd = nbadd + 1
        endif
      enddo
      call VecRestoreArrayRead(s%dx, pd, ierr); if (ierr /= 0) return
    endif

    !--- one least-squares problem per local row (= column k of C)
    nfail = 0; rmax = 0.d0; rsum = 0.d0
    ldb = max(s%maxi, s%maxj, 1)
    !$omp parallel private(mark, pos, jl, a, bb, work, wq, lwork, p, nj, ni, b, g, e, x, xk, pk, &
    !$omp                  info, den, cres, j0)
    allocate(mark(s%nx), pos(s%nx), jl(max(s%maxj, 1)), a(max(s%maxi, 1), max(s%maxj, 1)), bb(ldb))
    mark = 0; pos = 0
    lwork = -1
    call dgels('N', max(s%maxi, 1), max(s%maxj, 1), 1, a, max(s%maxi, 1), bb, ldb, wq, lwork, info)
    lwork = max(int(wq(1)), 2 * max(s%maxj, 1), 1)
    allocate(work(lwork))
    !$omp do schedule(dynamic, 16) reduction(+:nfail, rsum) reduction(max:rmax)
    do p = 1, nl
      j0 = s%jptr(p)
      nj = int(s%jptr(p + 1) - j0)
      jl(1:nj) = s%jx(j0 + 1:j0 + nj)
      xk = s%gx(s%g0 + p - 1)

      !--- A = M_s(I, J), read as M_s(J, I)^T from the rows of J
      ni = 0
      do b = 1, nj
        g = s%xgm(jl(b))
        do e = s%gptr(g) + 1, s%gptr(g + 1)
          if (.not. s%keep(e)) cycle
          x = s%xcol(e)
          if (mark(x) /= p) then
            mark(x) = p; ni = ni + 1; pos(x) = ni
            a(ni, 1:nj) = 0.d0
          endif
          a(pos(x), b) = v(e) * dsc(x) * dsc(jl(b))
        enddo
      enddo
      if (mark(xk) /= p) then               ! no stored diagonal: e_k's row is a zero row of A
        mark(xk) = p; ni = ni + 1; pos(xk) = ni
        a(ni, 1:nj) = 0.d0
      endif
      pk = pos(xk)
      bb(1:max(ni, nj)) = 0.d0
      bb(pk) = 1.d0

      if (nj == 1) then                     ! one unknown (e.g. a Dirichlet row): c = a_k / |a|^2
        den = sum(a(1:ni, 1)**2)
        if (den > 0.d0) then
          bb(1) = a(pk, 1) / den
          cres = sqrt(max(1.d0 - a(pk, 1) * bb(1), 0.d0))
        else
          bb(1) = 1.d0; cres = 1.d0; nfail = nfail + 1
        endif
      else
        call dgels('N', ni, nj, 1, a, size(a, 1), bb, ldb, work, lwork, info)
        if (info /= 0) then                 ! breakdown: the Jacobi column
          nfail = nfail + 1
          bb(1:nj) = 0.d0
          do b = 1, nj
            if (jl(b) == xk) bb(b) = 1.d0
          enddo
          cres = 1.d0
        else if (ni > nj) then
          cres = sqrt(sum(bb(nj + 1:ni)**2))
        else
          cres = 0.d0
        endif
      endif
      rmax = max(rmax, cres); rsum = rsum + cres

      !--- back from the scaled problem: C(k, j) = d_k C_s(k, j) d_j
      do b = 1, nj
        s%cval(j0 + b) = bb(b) * dsc(xk) * dsc(jl(b))
      enddo
    enddo
    !$omp end do
    deallocate(mark, pos, jl, a, bb, work)
    !$omp end parallel

    call aij_vals_done(sub(1), hv)
    call put_ij(sub(1), .false., nr, ia, ja)
    call MatDestroySubMatrices(one, sub, ierr); if (ierr /= 0) return

    !--- Craw's rows (all local: no communication in the assembly)
    allocate(cols(max(s%maxj, 1)))
    do p = 1, nl
      j0 = s%jptr(p)
      nj = int(s%jptr(p + 1) - j0)
      nj_ = int(nj, kind(nj_))
      cols(1:nj) = s%xg(s%jx(j0 + 1:j0 + nj))
      call MatSetValues(s%Craw, one, [s%rs + p - 1], nj_, cols(1:nj), s%cval(j0 + 1:j0 + nj), &
                        INSERT_VALUES, ierr)
      if (ierr /= 0) return
    enddo
    call MatAssemblyBegin(s%Craw, MAT_FINAL_ASSEMBLY, ierr); if (ierr /= 0) return
    call MatAssemblyEnd(s%Craw, MAT_FINAL_ASSEMBLY, ierr); if (ierr /= 0) return

    !--- the fill's statistics, global
    block
      real*8 :: st(2)
      integer :: ct(3)
      ct = [nfail, ndrop, nbadd]
      st = [rsum, dble(nl)]
      call MPI_Allreduce(MPI_IN_PLACE, ct, 3, MPI_INTEGER, MPI_SUM, s%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, st, 2, MPI_DOUBLE_PRECISION, MPI_SUM, s%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, rmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, s%comm, mierr)
      s%nfail = ct(1); s%ndrop = ct(2); s%nbadd = ct(3)
      s%res_avg = st(1) / max(st(2), 1.d0)
      s%res_max = rmax
    end block
  end subroutine fill_values


  !> Sort a ascending and drop duplicates in place: a(1:nu) unique (heapsort).
  subroutine sort_uniq(a, nu)
    PetscInt, intent(inout) :: a(:)   !< values; a(1:nu) sorted unique on return
    integer, intent(out)    :: nu     !< number of unique values
    PetscInt :: t
    integer :: n, i, last

    n = size(a)
    do i = n / 2, 1, -1
      call sift(i, n)
    enddo
    do last = n, 2, -1
      t = a(1); a(1) = a(last); a(last) = t
      call sift(1, last - 1)
    enddo
    nu = min(n, 1)
    do i = 2, n
      if (a(i) /= a(nu)) then
        nu = nu + 1
        a(nu) = a(i)
      endif
    enddo

  contains

    !> restore the max-heap below position i0 within a(1:hi)
    subroutine sift(i0, hi)
      integer, intent(in) :: i0  !< root
      integer, intent(in) :: hi  !< heap end
      integer :: rt, ch
      PetscInt :: tt
      rt = i0
      do while (2 * rt <= hi)
        ch = 2 * rt
        if (ch < hi) then
          if (a(ch + 1) > a(ch)) ch = ch + 1
        endif
        if (a(rt) < a(ch)) then
          tt = a(rt); a(rt) = a(ch); a(ch) = tt
          rt = ch
        else
          exit
        endif
      enddo
    end subroutine sift

  end subroutine sort_uniq

  !> Insertion sort of a(1:n) ascending (the short pattern lists).
  pure subroutine isort(a, n)
    integer, intent(inout) :: a(:)  !< values
    integer, intent(in)    :: n     !< how many to sort
    integer :: i, j, t
    do i = 2, n
      t = a(i)
      j = i - 1
      do while (j >= 1)
        if (a(j) <= t) exit
        a(j + 1) = a(j)
        j = j - 1
      enddo
      a(j + 1) = t
    enddo
  end subroutine isort

  !> Position of key in the ascending array a, 0 if absent.
  pure integer function bsearch(a, key)
    PetscInt, intent(in) :: a(:)  !< ascending values
    PetscInt, intent(in) :: key   !< the value looked for
    integer :: lo, hi, mid
    bsearch = 0
    lo = 1; hi = size(a)
    do while (lo <= hi)
      mid = (lo + hi) / 2
      if (a(mid) == key) then
        bsearch = mid
        return
      else if (a(mid) < key) then
        lo = mid + 1
      else
        hi = mid - 1
      endif
    enddo
  end function bsearch

  !> The message prefix: label, or "SPAI".
  function lbl(label) result(t)
    character(len=*), intent(in), optional :: label  !< caller's prefix
    character(len=:), allocatable :: t
    if (present(label)) then
      t = trim(label)
    else
      t = "SPAI"
    endif
  end function lbl

#endif
end module mod_petsc_pc_sf_spai
