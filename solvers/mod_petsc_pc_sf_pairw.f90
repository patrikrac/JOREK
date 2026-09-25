module mod_petsc_pc_sf_pairw
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  use mod_petsc_pc_physics_ctx, only: g_ctx, pcev_shellmult, pcev_mjsolve
  use mod_petsc_pc_mass_slot, only: mass_slot_t, mass_slot_setup, mass_slot_solve
  use mod_petsc_pc_sf_solver, only: SF_GMG_AXIS_RINGS
  use mod_petsc_raw_csr, only: aij_parts, get_ij, put_ij, aij_vals_read, aij_vals_done, &
                               c_aij_get, c_aij_restore
  use, intrinsic :: iso_c_binding, only: c_double, c_ptr, c_intptr_t, c_int, c_f_pointer
  implicit none
  private

  !--------------------------------------------------------------------
  !> The pair_w operator of the production path: the psi-channel Schur
  !! complement with the EXACT constraint mass, applied matrix-free,
  !!
  !!   S_uu u = B_22 u - B_21 p + B_23 B_33^-1 B_31 p,    p = Dh B_12 u,
  !!   Dh     = 1 / (opz diag(B_33) - diag(B_13 Qi B_31)),  Qi = 1 / diag(B_33),
  !!
  !! and the assembled operator its multigrid smoother needs. The Galerkin
  !! chain stays on the composed B_22 + W (the caller's S_W_aij): W is the
  !! continuum limit of the channel and good enough on the coarse levels.
  !!
  !! WHY. With S_uu = B_22 + W the outer iterations grow with the mesh: W
  !! lacks the discrete projection M_j^-1 that the Jacobian's Schur complement
  !! contains, and the gap is O(1) at grid scale (rho ~ (dt v_A / h)^2). With
  !! the exact mass in the fine operator the outer count is flat (shaped
  !! pcbench, tstep 1, 21x16 .. 121x48: 17-24 against 21-55). B_33 is
  !! geometry-only, so its per-slot factor is built once per run.
  !!
  !! THE SMOOTHER OPERATOR (sfw_lines). The level-0 zebra line smoother
  !! (mod_petsc_pc_gmg, smoother 7) needs the grid-scale part of S, which the
  !! same channel with Qi in place of B_33^-1 carries. It reads only its line
  !! blocks (same J, same slot), its axis block (rings 0..SF_GMG_AXIS_RINGS,
  !! all slots) and the coupling of odd-J lines to colour-0 columns (the axis
  !! and the even lines at J +- 1), so sfw_lines = P0 + C holds the channel C
  !! ONLY there: the full two-hop product (and Ltil) is never formed.
  !!
  !! STRUCTURE ONCE, NUMBERS PER REBUILD. sfw_structure (first build, before
  !! the value maps) fixes the halo index sets, the position maps and the
  !! pattern of sfw_lines, which then becomes a value-map target of the SF
  !! gather: every rebuild refills its P0 part like any other operator, and
  !! sfw_numeric adds C straight into its CSR values on the OpenMP threads.
  !! The pair_w block scaling is taken from sfw_lines' diagonal: it balances
  !! the operator pair_w solves, not the Galerkin surrogate (whose W inflates
  !! the u diagonal 2-2.4x and doubled the V-cycles at 61x48).
  !! The B_31 and B_12 rows a rank needs from its neighbours are re-extracted
  !! per rebuild (MatCreateSubMatrices, MAT_INITIAL_MATRIX: a MAT_REUSE_MATRIX
  !! refill returned stale values at np > 1).
  !--------------------------------------------------------------------

  Mat, save, public :: sfw_shell            !< the pair_w operator (MATSHELL)
  Mat, save, public :: sfw_lines            !< the level-0 smoother blocks' source

  !> raw CSR rows of a (Seq|MPI)AIJ matrix: diagonal part, then off-diagonal
  type :: raw_t
    Mat :: Md, Mo
    logical :: has_o = .false.
    PetscInt, pointer :: iad(:) => null(), jad(:) => null(), iao(:) => null(), jao(:) => null()
    real(c_double), pointer :: vd(:) => null(), vo(:) => null()
    type(c_ptr) :: pd, po
    PetscInt :: nr = 0
  end type raw_t

  logical, save :: use_qi = .false., gated = .false.
  type(mass_slot_t), save :: mj
  Mat, save :: p0                           !< the caller's scaled [B_22,B_24;B_42,B_44]
  Mat, save :: b31t                         !< B_31^T, for diag(B_13 Qi B_31)
  Vec, save :: qi, dh, xu, yu, tu, pp, j1, j2

  !--- the masked channel's structure, built once --------------------------
  PetscInt, allocatable, save :: r1(:), r2(:)     !< fetched rows of B_31 / B_12 (global)
  PetscInt, allocatable, save :: hcol(:)          !< pair column of each reachable u column
  integer, allocatable, save  :: o21(:), p21(:), o23(:), p23(:), p31(:), p12(:)
  integer, allocatable, save  :: cic(:), cjc(:)   !< C's pattern (positions in hcol)
  integer, allocatable, save  :: cpos(:)          !< C entry -> sfw_lines value (+diag / -offdiag part)
  integer, allocatable, save  :: kI(:), kJ(:)     !< (ring, angle) of the local u rows
  PetscInt, save   :: nz31 = -1, nz12 = -1, nzl_d = 0, nzl_o = 0, nu = 0, u0 = 0
  IS, save         :: is1, is2, isc_psi, isc_u
  VecScatter, save :: sc_q, sc_d
  Vec, save        :: q1, d2

  public :: sfw_structure, sfw_numeric

contains

  !--------------------------------------------------------------------
  !> First build: the mass factor, the shell and, with blocks (the GMG
  !! backend), the structure of sfw_lines -- filled with P0's values, as the
  !! value-map gate expects of every target. Pw is P0, unscaled here; the
  !! caller scales it in place, every rebuild.
  !--------------------------------------------------------------------
  subroutine sfw_structure(Pw, blocks, comm, my_id)
    Mat, intent(in)     :: Pw
    logical, intent(in) :: blocks
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr

    p0 = Pw
    call mass_slot_setup(mj, g_ctx%B_33, comm, "pair_w shell mass B_33")
    call MatCreateVecs(g_ctx%B_33, PETSC_NULL_VEC, qi, ierr)
    call MatGetDiagonal(g_ctx%B_33, qi, ierr)
    call VecReciprocal(qi, ierr)
    call VecDuplicate(qi, dh, ierr)                            ! psi and j share the layout
    call MatTranspose(g_ctx%B_31, MAT_INITIAL_MATRIX, b31t, ierr)
    call MatCreateShell(comm, local_rows(p0), local_rows(p0), global_rows(p0), &
                        global_rows(p0), PETSC_NULL_INTEGER, sfw_shell, ierr)
    call MatShellSetOperation(sfw_shell, MATOP_MULT, sfw_mult, ierr)
    call MatCreateVecs(g_ctx%B_21, pp, xu, ierr)               ! psi / u layouts
    call VecDuplicate(xu, yu, ierr)
    call VecDuplicate(xu, tu, ierr)
    call VecDuplicate(qi, j1, ierr)
    call VecDuplicate(qi, j2, ierr)
    if (blocks) call lines_structure(comm, my_id)
  end subroutine sfw_structure

  !--------------------------------------------------------------------
  !> Every rebuild, after the gather and BEFORE the pair scaling: Dh and,
  !! with blocks, sfw_lines = P0 + C. The caller then takes the pair_w scaling
  !! from sfw_lines -- the diagonal of the operator pair_w solves -- and
  !! applies it to sfw_lines, P0 and S_W_aij alike.
  !--------------------------------------------------------------------
  subroutine sfw_numeric(blocks, comm, my_id)
    logical, intent(in) :: blocks
    integer, intent(in) :: comm, my_id
    PetscErrorCode :: ierr

    call MatTranspose(g_ctx%B_31, MAT_REUSE_MATRIX, b31t, ierr)
    call make_dh(comm, my_id)
    if (.not. blocks) return
    call lines_numeric()
    if (.not. gated) call lines_gate(comm, my_id)
    gated = .true.
  end subroutine sfw_numeric

  !--------------------------------------------------------------------
  !> y = S x on the packed pair: P0 x, plus the channel on the u half (the
  !! rank-local leading block of the pair, viewed in place).
  !--------------------------------------------------------------------
  subroutine sfw_mult(A, x, y, ierr)
    Mat :: A
    Vec :: x, y
    PetscErrorCode :: ierr
    PetscScalar, pointer :: xa(:), ya(:)

    call PetscLogEventBegin(pcev_shellmult, ierr)
    call MatMult(p0, x, y, ierr)
    call VecGetArrayRead(x, xa, ierr)
    call VecPlaceArray(xu, xa, ierr)
    call MatMult(g_ctx%B_12, xu, pp, ierr)
    call VecResetArray(xu, ierr)
    call VecRestoreArrayRead(x, xa, ierr)
    call VecPointwiseMult(pp, pp, dh, ierr)                    ! p = Dh B_12 x_u
    call MatMult(g_ctx%B_31, pp, j1, ierr)
    if (use_qi) then                                           ! lines_gate only
      call VecPointwiseMult(j2, j1, qi, ierr)
    else
      call PetscLogEventBegin(pcev_mjsolve, ierr)
      call mass_slot_solve(mj, j1, j2)                         ! B_33^-1 B_31 p
      call PetscLogEventEnd(pcev_mjsolve, ierr)
    endif
    call MatMult(g_ctx%B_21, pp, tu, ierr)
    call VecGetArray(y, ya, ierr)
    call VecPlaceArray(yu, ya, ierr)
    call MatMultAdd(g_ctx%B_23, j2, yu, yu, ierr)
    call VecAXPY(yu, -1.0d0, tu, ierr)
    call VecResetArray(yu, ierr)
    call VecRestoreArray(y, ya, ierr)
    call PetscLogEventEnd(pcev_shellmult, ierr)
    ierr = 0
  end subroutine sfw_mult

  !--------------------------------------------------------------------
  !> Dh = 1 / (opz diag(B_33) - diag(B_13 Qi B_31)), the product's diagonal
  !! as row dots of B_13 with (B_31^T Qi), never the product. Rows with a
  !! vanishing denominator are floored to 1 (and counted).
  !--------------------------------------------------------------------
  subroutine make_dh(comm, my_id)
    use phys_module, only: time_evol_zeta
    integer, intent(in) :: comm, my_id
    PetscInt :: i, n, rs, nl, nr, a, b
    PetscInt, pointer :: cl(:), cr(:)
    PetscScalar, pointer :: vl(:), vr(:), dp(:)
    PetscErrorCode :: ierr
    PetscReal :: dmx
    real*8 :: acc
    integer :: nfl

    call MatDiagonalScale(b31t, PETSC_NULL_VEC, qi, ierr)
    call MatGetDiagonal(g_ctx%B_33, dh, ierr)
    call VecScale(dh, 1.0d0 + time_evol_zeta, ierr)
    call MatGetOwnershipRange(g_ctx%B_13, rs, PETSC_NULL_INTEGER, ierr)
    call VecGetLocalSize(dh, n, ierr)
    call VecGetArray(dh, dp, ierr)
    do i = 0, n - 1                          ! sorted global columns: one merge per row
      call MatGetRow(g_ctx%B_13, rs + i, nl, cl, vl, ierr)
      call MatGetRow(b31t, rs + i, nr, cr, vr, ierr)
      acc = 0.0d0; a = 1; b = 1
      do while (a <= nl .and. b <= nr)
        if (cl(a) == cr(b)) then
          acc = acc + vl(a) * vr(b); a = a + 1; b = b + 1
        else if (cl(a) < cr(b)) then
          a = a + 1
        else
          b = b + 1
        endif
      enddo
      call MatRestoreRow(b31t, rs + i, nr, cr, vr, ierr)
      call MatRestoreRow(g_ctx%B_13, rs + i, nl, cl, vl, ierr)
      dp(i + 1) = dp(i + 1) - acc
    enddo
    call VecRestoreArray(dh, dp, ierr)
    call VecNorm(dh, NORM_INFINITY, dmx, ierr)
    call VecGetArray(dh, dp, ierr)
    nfl = 0
    do i = 1, n
      if (abs(dp(i)) > 1.d-12 * max(dmx, 1.d-300)) then
        dp(i) = 1.0d0 / dp(i)
      else
        dp(i) = 1.0d0; nfl = nfl + 1
      endif
    enddo
    call VecRestoreArray(dh, dp, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, nfl, 1, MPI_INTEGER, MPI_SUM, comm, ierr)
    if (nfl > 0 .and. my_id == 0) write(*,'(A,I0,A)') &
      "[Physics PC]   pair_w shell: Dh floored on ", nfl, " rows"
  end subroutine make_dh

  !--------------------------------------------------------------------
  !> Once per run: the halo rows, the position maps, C's masked pattern and
  !! sfw_lines with the exact union pattern of P0 and C (P0's values).
  !--------------------------------------------------------------------
  subroutine lines_structure(comm, my_id)
    use mod_parameters, only: n_tor
    integer, intent(in) :: comm, my_id
    PetscInt, allocatable :: c21(:), c23(:), all(:), hh(:), ucol(:), dnz(:), onz(:)
    PetscInt, allocatable :: ustart(:), pstart(:)
    integer, allocatable  :: hI(:), hJ(:), hm(:), km(:), mark(:), lst(:), hmark(:), row(:), uoff(:)
    PetscInt, pointer :: ia(:), ja(:), pc(:), jc(:), cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt :: n, ng, r, e, prow0, nc
    Mat, pointer :: f31(:), f12(:)
    PetscErrorCode :: ierr
    integer :: np, me, pass, nrow, k, t

    call MPI_Comm_size(comm, np, ierr)
    call MPI_Comm_rank(comm, me, ierr)
    call MatGetOwnershipRange(g_ctx%B_21, u0, PETSC_NULL_INTEGER, ierr)
    call MatGetLocalSize(g_ctx%B_21, nu, PETSC_NULL_INTEGER, ierr)
    call MatGetOwnershipRange(p0, prow0, PETSC_NULL_INTEGER, ierr)   ! u rows lead the rank's pair rows
    allocate(ustart(0:np), pstart(0:np))
    call MPI_Allgather(u0, 1, MPIU_INTEGER, ustart, 1, MPIU_INTEGER, comm, ierr)
    call MPI_Allgather(prow0, 1, MPIU_INTEGER, pstart, 1, MPIU_INTEGER, comm, ierr)
    ustart(np) = global_rows(g_ctx%B_21); pstart(np) = global_rows(p0)

    !--- the halo: B_31 rows reached through B_23, B_12 rows through B_21 and them
    call raw_cols(g_ctx%B_23, o23, c23)
    call uniq(c23, r1)
    p23 = [(find(r1, c23(e)), e = 1, size(c23))]
    call ISCreateStride(PETSC_COMM_SELF, global_cols(g_ctx%B_31), 0, 1, isc_psi, ierr)
    call ISCreateGeneral(PETSC_COMM_SELF, int(size(r1), kind(n)), r1, PETSC_COPY_VALUES, is1, ierr)
    call MatCreateSubMatrices(g_ctx%B_31, 1, [is1], [isc_psi], MAT_INITIAL_MATRIX, f31, ierr)
    call get_ij(f31(1), .false., n, ia, ja)
    nz31 = ia(n + 1)
    call raw_cols(g_ctx%B_21, o21, c21)
    all = [c21, ja(1:nz31)]
    call uniq(all, r2)
    p21 = [(find(r2, c21(e)), e = 1, size(c21))]
    p31 = [(find(r2, ja(e)), e = 1, nz31)]
    call ISCreateStride(PETSC_COMM_SELF, global_cols(g_ctx%B_12), 0, 1, isc_u, ierr)
    call ISCreateGeneral(PETSC_COMM_SELF, int(size(r2), kind(n)), r2, PETSC_COPY_VALUES, is2, ierr)
    call MatCreateSubMatrices(g_ctx%B_12, 1, [is2], [isc_u], MAT_INITIAL_MATRIX, f12, ierr)

    !--- the reachable u columns hh, their pair columns and geometry
    call get_ij(f12(1), .false., n, pc, jc)
    nz12 = pc(n + 1)
    all = jc(1:nz12)
    call uniq(all, hh)
    p12 = [(find(hh, jc(e)), e = 1, nz12)]
    allocate(hcol(size(hh)), hI(size(hh)), hJ(size(hh)), hm(size(hh)))
    do k = 1, size(hh)
      t = owner(ustart, hh(k))
      hcol(k) = pstart(t) + (hh(k) - ustart(t))
      call geom(hh(k), hI(k), hJ(k), hm(k))
    enddo
    allocate(kI(nu), kJ(nu), km(nu))
    do r = 1, nu
      call geom(u0 + r - 1, kI(r), kJ(r), km(r))
    enddo

    !--- C's pattern: the masked columns reachable from each local row
    allocate(mark(size(r2)), lst(size(r2)), hmark(size(hh)), row(size(hh)), cic(nu + 1))
    do pass = 1, 2
      mark = 0; hmark = 0
      cic(1) = 0
      do r = 1, nu
        nrow = 0
        call reach(int(r))
        if (pass == 1) then
          cic(r + 1) = cic(r) + nrow
        else
          cjc(cic(r) + 1:cic(r + 1)) = row(1:nrow)
        endif
      enddo
      if (pass == 1) allocate(cjc(cic(nu + 1)))
    enddo
    call put_ij(f31(1), .false., n, ia, ja)
    call put_ij(f12(1), .false., n, pc, jc)
    call MatDestroySubMatrices(1, f31, ierr)
    call MatDestroySubMatrices(1, f12, ierr)

    !--- the vector halos: Qi on r1, Dh on r2
    call VecCreateSeq(PETSC_COMM_SELF, int(size(r1), kind(n)), q1, ierr)
    call VecCreateSeq(PETSC_COMM_SELF, int(size(r2), kind(n)), d2, ierr)
    call VecScatterCreate(qi, is1, q1, PETSC_NULL_IS, sc_q, ierr)
    call VecScatterCreate(dh, is2, d2, PETSC_NULL_IS, sc_d, ierr)

    !--- sfw_lines: the union of P0's and C's patterns, preallocated exactly,
    !--- holding P0's values (C's positions 0) as the value-map gate expects
    n = local_rows(p0); ng = global_rows(p0)
    allocate(uoff(n + 1), dnz(n), onz(n))
    uoff(1) = 0
    do pass = 1, 2
      do r = 1, n
        call MatGetRow(p0, prow0 + r - 1, nc, cols, PETSC_NULL_SCALAR_POINTER, ierr)
        all = cols(1:nc)
        call MatRestoreRow(p0, prow0 + r - 1, nc, cols, PETSC_NULL_SCALAR_POINTER, ierr)
        if (r <= nu) all = [all, hcol(cjc(cic(r) + 1:cic(r + 1)))]
        call uniq(all, hh)
        if (pass == 1) then
          uoff(r + 1) = uoff(r) + size(hh)
          dnz(r) = count(hh >= pstart(me) .and. hh < pstart(me + 1))
          onz(r) = size(hh) - dnz(r)
        else
          ucol(uoff(r) + 1:uoff(r + 1)) = hh
        endif
      enddo
      if (pass == 1) allocate(ucol(uoff(n + 1)))
    enddo
    call MatCreate(comm, sfw_lines, ierr)                     ! MPIAIJ on any np: a value-map target
    call MatSetSizes(sfw_lines, n, n, ng, ng, ierr)
    call MatSetType(sfw_lines, MATMPIAIJ, ierr)
    call MatMPIAIJSetPreallocation(sfw_lines, 0, dnz, 0, onz, ierr)
    call MatSetOption(sfw_lines, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE, ierr)
    do r = 1, n
      nc = uoff(r + 1) - uoff(r)
      call MatSetValues(sfw_lines, 1, [prow0 + r - 1], nc, ucol(uoff(r) + 1:uoff(r + 1)), &
                        [(0.0d0, e = 1, nc)], INSERT_VALUES, ierr)
      call MatGetRow(p0, prow0 + r - 1, nc, cols, vals, ierr)
      call MatSetValues(sfw_lines, 1, [prow0 + r - 1], nc, cols, vals, INSERT_VALUES, ierr)
      call MatRestoreRow(p0, prow0 + r - 1, nc, cols, vals, ierr)
    enddo
    call MatAssemblyBegin(sfw_lines, MAT_FINAL_ASSEMBLY, ierr)
    call MatAssemblyEnd(sfw_lines, MAT_FINAL_ASSEMBLY, ierr)
    call MatSetOption(sfw_lines, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr)
    call make_cpos(pstart(me), pstart(me + 1))

    block
      real*8 :: tot(2)
      tot = [dble(cic(nu + 1)), dble(nu)]
      call MPI_Allreduce(MPI_IN_PLACE, tot, 2, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
      if (my_id == 0) write(*,'(A,F7.1,A)') "[Physics PC]   pair_w smoother operator: masked "// &
        "diagonal-mass channel, ", tot(1) / max(tot(2), 1.d0), " nnz per u row"
    end block

  contains

    !> the masked hh-positions reachable from local row r, into row(1:nrow)
    subroutine reach(r_)
      integer, intent(in) :: r_
      integer :: q, s, l, nt, k2
      nt = 0
      do q = o21(r_) + 1, o21(r_ + 1)
        k2 = p21(q)
        if (mark(k2) /= r_) then
          mark(k2) = r_; nt = nt + 1; lst(nt) = k2
        endif
      enddo
      do q = o23(r_) + 1, o23(r_ + 1)
        l = p23(q)
        do s = int(ia(l)) + 1, int(ia(l + 1))
          k2 = p31(s)
          if (mark(k2) /= r_) then
            mark(k2) = r_; nt = nt + 1; lst(nt) = k2
          endif
        enddo
      enddo
      do q = 1, nt
        k2 = lst(q)
        do s = int(pc(k2)) + 1, int(pc(k2 + 1))
          l = p12(s)
          if (hmark(l) == r_) cycle
          hmark(l) = r_
          if (.not. keep(kI(r_), kJ(r_), km(r_), hI(l), hJ(l), hm(l))) cycle
          nrow = nrow + 1; row(nrow) = l
        enddo
      enddo
    end subroutine reach

    !> (ring I, angle J, slot m) of a global u index: index-major, g = idx n_tor + m
    subroutine geom(g, gI, gJ, gm)
      use nodes_elements, only: node_list
      use phys_module,    only: n_tht
      PetscInt, intent(in) :: g
      integer, intent(out) :: gI, gJ, gm
      integer, allocatable, save :: nodeof(:)
      integer :: i, q
      if (.not. allocated(nodeof)) then
        q = 0
        do i = 1, node_list%n_nodes
          q = max(q, maxval(node_list%node(i)%index(1:4)))
        enddo
        allocate(nodeof(0:q - 1))
        nodeof = -1
        do i = 1, node_list%n_nodes
          nodeof(node_list%node(i)%index(1:4) - 1) = i - 1
        enddo
      endif
      q = int(g / n_tor); gm = int(mod(g, int(n_tor, kind(g))))
      if (q >= size(nodeof)) call fatal("a u index lies outside the node map")
      if (nodeof(q) < 0)     call fatal("a u index has no node")
      gI = nodeof(q) / n_tht; gJ = mod(nodeof(q), n_tht)
    end subroutine geom

    subroutine fatal(msg)
      character(len=*), intent(in) :: msg
      write(*,'(A,A)') "[Physics PC]   FATAL: pair_w smoother operator: ", msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine fatal

  end subroutine lines_structure

  !> cpos: every C entry's position in sfw_lines' diagonal (+) or
  !! off-diagonal (-) CSR value array; the u rows lead the local rows.
  subroutine make_cpos(cs, ce)
    PetscInt, intent(in) :: cs, ce
    type(raw_t) :: L
    PetscInt, pointer :: garr(:)
    Mat :: Md, Mo
    logical :: has_o
    integer :: r, q
    PetscInt :: c, a, b

    call aij_parts(sfw_lines, Md, Mo, garr, has_o)
    call raw_open(sfw_lines, L)
    nzl_d = L%iad(L%nr + 1)
    if (L%has_o) nzl_o = L%iao(L%nr + 1)
    allocate(cpos(size(cjc)))
    do r = 1, int(nu)
      do q = cic(r) + 1, cic(r + 1)
        c = hcol(cjc(q))
        if (c >= cs .and. c < ce) then
          a = L%iad(r) + 1; b = L%iad(r + 1)
          cpos(q) = int(a) - 1 + find(L%jad(a:b), c - cs)
        else
          a = L%iao(r) + 1; b = L%iao(r + 1)
          cpos(q) = -(int(a) - 1 + find(L%jao(a:b), int(find(garr, c) - 1, kind(c))))
        endif
      enddo
    enddo
    call raw_close(L)
  end subroutine make_cpos

  !--------------------------------------------------------------------
  !> The zebra smoother's read set (mod_petsc_pc_gmg smoother 7 with an axis
  !! block): axis rows read the axis block; even lines their own line; odd
  !! lines their own line and the colour-0 columns next to them.
  !--------------------------------------------------------------------
  logical function keep(Ir, Jr, mr, Ic, Jc, mc)
    use phys_module, only: n_tht
    integer, intent(in) :: Ir, Jr, mr, Ic, Jc, mc
    integer :: dj
    if (Ir <= SF_GMG_AXIS_RINGS) then
      keep = (Ic <= SF_GMG_AXIS_RINGS)
    else if (Ic <= SF_GMG_AXIS_RINGS) then
      keep = (mod(Jr, 2) == 1)
    else if (Jc == Jr) then
      keep = (mc == mr)
    else
      dj = abs(Jc - Jr); dj = min(dj, n_tht - dj)
      keep = (mod(Jr, 2) == 1 .and. mod(Jc, 2) == 0 .and. dj == 1)
    endif
  end function keep

  !--------------------------------------------------------------------
  !> Every rebuild: refetch the halo rows, scatter Qi and Dh, and add
  !! C = - Ltil Dh B_12 (masked), Ltil = B_21 - B_23 Qi B_31, into the
  !! gathered sfw_lines -- rows on the OpenMP threads, each row writing only
  !! its own values.
  !--------------------------------------------------------------------
  subroutine lines_numeric()
    Mat, pointer :: f31(:), f12(:)
    type(raw_t) :: b21, b23
    PetscInt, pointer :: ia(:), ja(:), pc(:), jc(:), garr(:)
    real(c_double), pointer :: v31(:), v12(:), vd(:), vo(:)
    PetscScalar, pointer :: qa(:), da(:)
    type(c_ptr) :: h31, h12, pd, po
    Mat :: Ld, Lo
    logical :: has_o
    real*8, allocatable :: acc(:), buf(:)
    integer, allocatable :: mark(:), lst(:), slot(:)
    PetscInt :: n
    integer :: r, q, s, l, nt, k2, h, w, m, e
    integer(c_int) :: rc
    real*8 :: a
    PetscErrorCode :: ierr

    call MatCreateSubMatrices(g_ctx%B_31, 1, [is1], [isc_psi], MAT_INITIAL_MATRIX, f31, ierr)
    call MatCreateSubMatrices(g_ctx%B_12, 1, [is2], [isc_u], MAT_INITIAL_MATRIX, f12, ierr)
    call get_ij(f31(1), .false., n, ia, ja)
    call get_ij(f12(1), .false., n, pc, jc)
    if (ia(size(r1) + 1) /= nz31 .or. pc(size(r2) + 1) /= nz12) then
      write(*,'(A)') "[Physics PC]   FATAL: pair_w smoother operator: a halo pattern changed"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    call aij_vals_read(f31(1), int(nz31, 8), h31, v31)
    call aij_vals_read(f12(1), int(nz12, 8), h12, v12)
    call VecScatterBegin(sc_q, qi, q1, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(sc_q, qi, q1, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterBegin(sc_d, dh, d2, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecScatterEnd(sc_d, dh, d2, INSERT_VALUES, SCATTER_FORWARD, ierr)
    call VecGetArrayRead(q1, qa, ierr)
    call VecGetArrayRead(d2, da, ierr)
    call raw_open(g_ctx%B_21, b21)
    call raw_open(g_ctx%B_23, b23)
    call aij_parts(sfw_lines, Ld, Lo, garr, has_o)
    rc = c_aij_get(transfer(Ld%v, 0_c_intptr_t), pd)
    call c_f_pointer(pd, vd, [max(nzl_d, 1_8)])
    if (has_o) then
      rc = c_aij_get(transfer(Lo%v, 0_c_intptr_t), po)
      call c_f_pointer(po, vo, [max(nzl_o, 1_8)])
    else
      vo => vd(1:0)
    endif

    !$omp parallel private(acc, buf, mark, lst, slot, r, q, s, l, nt, k2, h, w, m, e, a)
    allocate(acc(size(r2)), mark(size(r2)), lst(size(r2)), slot(size(hcol)), buf(size(hcol)))
    mark = 0; slot = 0
    !$omp do schedule(dynamic, 64)
    do r = 1, int(nu)
      !--- row r of Ltil = B_21 - B_23 Qi B_31, sparse over r2
      nt = 0
      do e = 1, raw_len(b21, r)
        k2 = p21(o21(r) + e)
        if (mark(k2) /= r) then
          mark(k2) = r; nt = nt + 1; lst(nt) = k2; acc(k2) = 0.0d0
        endif
        acc(k2) = acc(k2) + raw_val(b21, r, e)
      enddo
      do e = 1, raw_len(b23, r)
        l = p23(o23(r) + e)
        a = raw_val(b23, r, e) * qa(l)
        do s = int(ia(l)) + 1, int(ia(l + 1))
          k2 = p31(s)
          if (mark(k2) /= r) then
            mark(k2) = r; nt = nt + 1; lst(nt) = k2; acc(k2) = 0.0d0
          endif
          acc(k2) = acc(k2) - a * v31(s)
        enddo
      enddo
      !--- C(r, masked) = - Ltil(r, :) Dh B_12(:, masked), added into sfw_lines
      w = cic(r + 1) - cic(r)
      do q = 1, w
        slot(cjc(cic(r) + q)) = q
      enddo
      buf(1:w) = 0.0d0
      do q = 1, nt
        k2 = lst(q)
        a = acc(k2) * da(k2)
        do s = int(pc(k2)) + 1, int(pc(k2 + 1))
          h = slot(p12(s))
          if (h > 0) buf(h) = buf(h) - a * v12(s)
        enddo
      enddo
      do q = 1, w
        slot(cjc(cic(r) + q)) = 0
        m = cpos(cic(r) + q)
        if (m > 0) then
          vd(m) = vd(m) + buf(q)
        else
          vo(-m) = vo(-m) + buf(q)
        endif
      enddo
    enddo
    !$omp end do
    deallocate(acc, mark, lst, slot, buf)
    !$omp end parallel

    if (has_o) rc = c_aij_restore(transfer(Lo%v, 0_c_intptr_t), po)
    rc = c_aij_restore(transfer(Ld%v, 0_c_intptr_t), pd)
    call raw_close(b21); call raw_close(b23)
    call VecRestoreArrayRead(q1, qa, ierr)
    call VecRestoreArrayRead(d2, da, ierr)
    call aij_vals_done(f31(1), h31); call aij_vals_done(f12(1), h12)
    call put_ij(f31(1), .false., n, ia, ja); call put_ij(f12(1), .false., n, pc, jc)
    call MatDestroySubMatrices(1, f31, ierr)
    call MatDestroySubMatrices(1, f12, ierr)
  end subroutine lines_numeric

  !--------------------------------------------------------------------
  !> First-build gate of the mask, keys, halo, maps and pair columns: for x
  !! on one even line (J = 0, slot 0, outside the axis block), sfw_lines x
  !! must equal the diagonal-mass channel operator on that line's rows and on
  !! the odd lines next to it -- exactly the rows whose entries the mask keeps.
  !--------------------------------------------------------------------
  subroutine lines_gate(comm, my_id)
    use phys_module,    only: n_tht
    use mod_parameters, only: n_tor
    integer, intent(in) :: comm, my_id
    Vec :: x, y1, y2
    PetscScalar, pointer :: xa(:), a1(:), a2(:)
    PetscErrorCode :: ierr
    integer :: r
    real*8 :: e(2)

    call MatCreateVecs(sfw_lines, x, y1, ierr)
    call VecDuplicate(y1, y2, ierr)
    call VecSetRandom(x, PETSC_NULL_RANDOM, ierr)
    call VecGetArray(x, xa, ierr)
    do r = 1, size(xa)
      if (r > nu) then
        xa(r) = 0.0d0
      else if (.not. on_line0(r)) then
        xa(r) = 0.0d0
      endif
    enddo
    call VecRestoreArray(x, xa, ierr)
    call MatMult(sfw_lines, x, y1, ierr)
    use_qi = .true.
    call MatMult(sfw_shell, x, y2, ierr)
    use_qi = .false.
    call VecGetArrayRead(y1, a1, ierr)
    call VecGetArrayRead(y2, a2, ierr)
    e = 0.d0
    do r = 1, int(nu)
      if (.not. (on_line0(r) .or. (kI(r) > SF_GMG_AXIS_RINGS .and. &
                                   (kJ(r) == 1 .or. kJ(r) == n_tht - 1)))) cycle
      e(1) = e(1) + (a1(r) - a2(r))**2
      e(2) = e(2) + a2(r)**2
    enddo
    call VecRestoreArrayRead(y1, a1, ierr)
    call VecRestoreArrayRead(y2, a2, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, e, 2, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
    e(1) = sqrt(e(1) / max(e(2), 1.d-300))
    if (my_id == 0) write(*,'(A,ES10.3)') &
      "[Physics PC]   pair_w smoother operator gate (masked vs full channel, one line): ", e(1)
    if (e(1) > 1.d-10) then
      if (my_id == 0) write(*,'(A)') "[Physics PC]   FATAL: the masked smoother operator is wrong."
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif
    call VecDestroy(x, ierr); call VecDestroy(y1, ierr); call VecDestroy(y2, ierr)

  contains

    logical function on_line0(r_)
      integer, intent(in) :: r_
      on_line0 = kI(r_) > SF_GMG_AXIS_RINGS .and. kJ(r_) == 0 .and. &
                 mod(u0 + r_ - 1, int(n_tor, kind(u0))) == 0
    end function on_line0

  end subroutine lines_gate

  !====================================================================
  ! raw CSR access
  !====================================================================

  subroutine raw_open(M, rw)
    Mat, intent(in) :: M
    type(raw_t), intent(out) :: rw
    PetscInt, pointer :: garr(:)
    PetscInt :: n
    call aij_parts(M, rw%Md, rw%Mo, garr, rw%has_o)
    call get_ij(rw%Md, .false., n, rw%iad, rw%jad)
    call aij_vals_read(rw%Md, int(rw%iad(n + 1), 8), rw%pd, rw%vd)
    if (rw%has_o) then
      call get_ij(rw%Mo, .false., n, rw%iao, rw%jao)
      call aij_vals_read(rw%Mo, int(rw%iao(n + 1), 8), rw%po, rw%vo)
    endif
    rw%nr = n
  end subroutine raw_open

  subroutine raw_close(rw)
    type(raw_t), intent(inout) :: rw
    PetscInt :: n
    n = rw%nr
    call aij_vals_done(rw%Md, rw%pd)
    call put_ij(rw%Md, .false., n, rw%iad, rw%jad)
    if (rw%has_o) then
      call aij_vals_done(rw%Mo, rw%po)
      call put_ij(rw%Mo, .false., n, rw%iao, rw%jao)
    endif
  end subroutine raw_close

  !> entries of local row r (1-based), diagonal part first
  pure integer function raw_len(rw, r)
    type(raw_t), intent(in) :: rw
    integer, intent(in) :: r
    raw_len = int(rw%iad(r + 1) - rw%iad(r))
    if (rw%has_o) raw_len = raw_len + int(rw%iao(r + 1) - rw%iao(r))
  end function raw_len

  !> the e-th value of local row r, in raw_len's order
  pure real*8 function raw_val(rw, r, e)
    type(raw_t), intent(in) :: rw
    integer, intent(in) :: r, e
    integer :: nd
    nd = int(rw%iad(r + 1) - rw%iad(r))
    if (e <= nd) then
      raw_val = rw%vd(rw%iad(r) + e)
    else
      raw_val = rw%vo(rw%iao(r) + e - nd)
    endif
  end function raw_val

  !> the global columns of M's local rows in raw order, the order raw_val
  !! reads the values in; off(r) is row r's offset
  subroutine raw_cols(M, off, cols)
    Mat, intent(in) :: M
    integer, allocatable, intent(out) :: off(:)
    PetscInt, allocatable, intent(out) :: cols(:)
    type(raw_t) :: rw
    PetscInt, pointer :: garr(:)
    Mat :: Md, Mo
    logical :: has_o
    PetscInt :: cs
    integer :: r, k, nd
    PetscErrorCode :: ierr
    call MatGetOwnershipRangeColumn(M, cs, PETSC_NULL_INTEGER, ierr)
    call aij_parts(M, Md, Mo, garr, has_o)
    call raw_open(M, rw)
    allocate(off(rw%nr + 1))
    off(1) = 0
    do r = 1, int(rw%nr)
      off(r + 1) = off(r) + raw_len(rw, r)
    enddo
    allocate(cols(off(rw%nr + 1)))
    do r = 1, int(rw%nr)
      k = off(r)
      nd = int(rw%iad(r + 1) - rw%iad(r))
      cols(k + 1:k + nd) = rw%jad(rw%iad(r) + 1:rw%iad(r + 1)) + cs
      if (rw%has_o) cols(k + nd + 1:off(r + 1)) = garr(rw%jao(rw%iao(r) + 1:rw%iao(r + 1)) + 1)
    enddo
    call raw_close(rw)
  end subroutine raw_cols

  !====================================================================
  ! small helpers
  !====================================================================

  PetscInt function local_rows(M)
    Mat, intent(in) :: M
    PetscErrorCode :: ierr
    call MatGetLocalSize(M, local_rows, PETSC_NULL_INTEGER, ierr)
  end function local_rows

  PetscInt function global_rows(M)
    Mat, intent(in) :: M
    PetscErrorCode :: ierr
    call MatGetSize(M, global_rows, PETSC_NULL_INTEGER, ierr)
  end function global_rows

  PetscInt function global_cols(M)
    Mat, intent(in) :: M
    PetscErrorCode :: ierr
    call MatGetSize(M, PETSC_NULL_INTEGER, global_cols, ierr)
  end function global_cols

  !> sorted unique copy
  subroutine uniq(a, u)
    PetscInt, intent(in) :: a(:)
    PetscInt, allocatable, intent(out) :: u(:)
    PetscInt, allocatable :: s(:)
    PetscCount :: n
    integer :: i, k
    PetscErrorCode :: ierr
    s = a
    n = size(s)
    if (n > 0) call PetscSortInt(n, s, ierr)
    k = 0
    do i = 1, int(n)
      if (k > 0) then
        if (s(i) == s(k)) cycle
      endif
      k = k + 1; s(k) = s(i)
    enddo
    u = s(1:k)
  end subroutine uniq

  !> 1-based position of v in the sorted array a (which contains it)
  pure integer function find(a, v)
    PetscInt, intent(in) :: a(:), v
    integer :: lo, hi, mid
    lo = 1; hi = size(a)
    do while (lo < hi)
      mid = (lo + hi) / 2
      if (a(mid) < v) then
        lo = mid + 1
      else
        hi = mid
      endif
    enddo
    find = lo
  end function find

  !> the rank owning global row g, from the ownership starts s(0:np)
  pure integer function owner(s, g)
    PetscInt, intent(in) :: s(0:), g
    integer :: lo, hi, mid
    lo = 0; hi = size(s) - 2
    do while (lo < hi)
      mid = (lo + hi + 1) / 2
      if (s(mid) <= g) then
        lo = mid
      else
        hi = mid - 1
      endif
    enddo
    owner = lo
  end function owner

#endif
end module mod_petsc_pc_sf_pairw
