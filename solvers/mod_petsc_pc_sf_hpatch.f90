module mod_petsc_pc_sf_hpatch
#ifdef USE_PETSC
  use mpi_mod
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none
  private

  !--------------------------------------------------------------------
  !> The flux arm's local velocity response H (physics_pc_sf_suu = flux,
  !! -sf_flux_s1 hpatch, the default): the "H-form" S1 surrogate
  !!
  !!   S1 = [[B_11 + B_12 H, B_13], [B_31, B_33]],
  !!
  !! H ~ -Sigma_u L_1 restricted to its psi column: the velocity a unit psi
  !! perturbation drives through the j constraint (B_31, B_33), the bending
  !! and kink forces (B_23, B_21) and the inertia (B_22 + W_c), computed on
  !! local patches and cut to a band. B_12 stays the assembled left factor,
  !! so ker B_12 is inherited; no transpose, no 1 / varsigma, no sign.
  !!
  !! WHY. The Tang commutation behind the SPAI form, B_12^T C_w B_12 with
  !! C_w ~ M_w^-1, does not hold on JOREK's shaped toroidal Hermite grid:
  !! sigma(S~1(M_w^-1)^-1 S1) = [0.67, 1.62] on the dumped blocks, and no
  !! symmetric middle operator fixes it (dense study on the real dumps,
  !! 2026-10). The H-form brought GMRES on the production sweep from 18-19 /
  !! 21 to 8 / 10 iterations (exact S1: 7 / 7-8).
  !!
  !! DEFINITION. On the index graph G (JOREK index i <-> rows i ns + m, all
  !! slots m; edges = the stored structure of B_33, symmetrised; a node's
  !! four indices form a clique with identical closed neighbourhoods, so
  !! index distances are node distances):
  !!   groups g: greedy aggregation of the OWNED indices -- seed = the lowest
  !!     unassigned index and its node (the indices with its closed
  !!     neighbourhood), then the g - 1 unassigned neighbour nodes with the
  !!     most common neighbours (-sf_flux_hp_group, default 4 nodes);
  !!   columns: the free psi DOFs of all slots of the group's indices;
  !!   patch P: the indices within distance p of the group (-sf_flux_hp_p,
  !!     default 3); rows: their free DOFs, all slots;
  !!   X = B_33[P,P]^-1 B_31[P,cols],   R = B_23[P,P] X - B_21[P,cols],
  !!   Y = (B_22 + W_c)[P,P]^-1 R         (u only), or with -sf_flux_hp_vel 1
  !!   Y = u part of D_v[P,P]^-1 [R; 0],  D_v = [[B_22 + W_c, B_24], [B_42, B_44]],
  !!   H(i, col) = Y(i, col) where dist(index(i), index(col)) <= r
  !!     (-sf_flux_hp_r, default 3; measured inside the patch, exact for r <= p).
  !! "Free" = not a diagonal-only row of P_J (psi, j) or D_v (u, omega): the
  !! eliminated Dirichlet rows. Penalty rows stay in (their ZBIG diagonal
  !! makes their entries of X, Y ~ 0). B_22 + W_c is read from D_v's (1,1)
  !! block itself, so it carries exactly D_v's W_c and scaling.
  !!
  !! SPLIT-M (q > 0; -sf_flux_s1 splitm). Tang's commuted part kept exactly,
  !! the commutator remainder restored locally:
  !!   H_T = -(1/varsigma) M_w[Q]^-1 B_23[Q, cols]  (u only), cut to q,
  !!         Q = the indices within distance q of the group (q >= p + 1),
  !!   Y   = (B_22 + W_c)[P,P]^-1 (R - D_v H_T)[P]   (or D_v's u part, vel 1),
  !!   H   = H_T + Y cut to r.
  !! H_T factors through B_23, so the part of H that truncation must not
  !! touch (it carries ker B_12 at large theta) never passes through the
  !! cut; only the remainder R - D_v H_T does -- the shear term at rational
  !! surfaces and the toroidal weight mismatch that Tang's commutation drops.
  !! H's reach is q (S1's 1 + q). Without it the H-form's truncation of the
  !! whole R makes S1 indefinite at large theta (testbed n = 2, theta = 100).
  !!
  !! SLOT-DIAGONAL SOLVES. Every local system (B_33, M_w, the velocity block)
  !! whose off-slot entries are all below SD_TOL relative to their row's
  !! largest is solved slot by slot -- B_33 = M_{1/R} and M_w are axisymmetric,
  !! so in practice always those two: ns factorisations of size n / ns
  !! instead of one of size n, ~ ns^2 fewer flops. Otherwise coupled.
  !!
  !! PARALLEL LAYOUT. The operators are the path's (global: g_ctx blocks,
  !! ns = n_tor; mode split: the family's, ns = pc_ntor), scalar row = index
  !! ns + slot, the packed pairs [field 1 local | field 2 local] per rank.
  !! Groups never span ranks; their patches reach into the neighbours' rows,
  !! which are gathered by MatCreateSubMatrices over U = the indices within
  !! distance p of the rank's own (as the SPAI's halo, mod_petsc_pc_sf_spai).
  !! H^T is assembled with OWNED rows (a group's psi columns are the rank's),
  !! so the preallocation is exact and local; H = its transpose. The
  !! structure (U, groups, patches, H's pattern) is fixed at the first build;
  !! every rebuild gathers the values again, re-solves and refills H in place.
  !!
  !! LOCAL SOLVES. Per group, B_33[P,P] and the velocity block are factored
  !! by banded LU (dgbtrf / dgbtrs) in a reverse Cuthill-McKee order of the
  !! patch's indices (all DOFs of an index together), dense LU (dgetrf) where
  !! the band storage would be larger. Groups run on the OpenMP threads.
  !! Cost per group ~ 2 n b^2 + 4 n b n_c flops, n ~ |P| ns, b ~ the RCM band
  !! (about one patch width of nodes, x 4 ns), n_c ~ 4 g ns columns.
  !--------------------------------------------------------------------

  PetscInt, parameter :: zero = 0, one = 1
  !> off-slot entries below SD_TOL x (row's largest |entry|) count as zero in
  !! the slot-diagonal test (the FFT assembly leaves round-off there)
  real*8, parameter :: SD_TOL = 1.0d-12

  !> One source's gathered rows (row k = U position (k-1) / ns + 1, slot
  !! mod(k-1, ns)): CSR with the columns as (field, local id), id = (U
  !! position - 1) ns + slot (0-based), -1 outside U; dir = diagonal-only row.
  type :: grows_t
    integer :: nrow = 0
    integer(8), allocatable :: ptr(:)
    integer(1), allocatable :: fld(:)
    integer, allocatable    :: cid(:)
    real*8, allocatable     :: val(:)
    logical, allocatable    :: dir(:)
  end type grows_t

  !> One group's rows of H^T: psi rows row(1:nr), entries rp(q)+1 .. rp(q+1).
  type :: hres_t
    integer :: nr = 0
    PetscInt, allocatable :: row(:)
    integer, allocatable  :: rp(:)
    PetscInt, allocatable    :: col(:)
    PetscScalar, allocatable :: val(:)
  end type hres_t

  type, public :: hpatch_t
    logical :: built = .false.
    integer :: comm = MPI_COMM_NULL
    integer :: ns = 1, p = 3, r = 3, g = 4, vel = 0
    integer :: q = 0                            !< split-M: H_T's mass-patch radius (0 = the plain H-form)
    real*8  :: vsg = 1.d0                       !< split-M: varsigma (B_22 = -varsigma K_{R^3 rho0})
    integer :: np = 1, me = 0
    PetscInt :: rs = 0, re = 0, n = 0           !< owned scalar rows [rs, re), global size
    PetscInt, allocatable :: rg(:)              !< (0:np) scalar row starts per rank
    PetscInt, allocatable :: uix(:)             !< U: the gathered indices, sorted
    integer :: nu = 0, o0 = 0, no = 0           !< |U|, U position of the first owned index, owned count
    integer, allocatable :: aptr(:), aadj(:)    !< closed adjacency of U (positions), aptr 0-based
    integer :: ngrp = 0
    integer, allocatable :: gptr(:), gix(:)     !< groups: U positions
    integer, allocatable :: pptr(:), pix(:)     !< patches: U positions in RCM order
    integer, allocatable :: qptr(:), qix(:)     !< split-M: the radius-q patches, RCM order
    Mat :: Ht                                   !< H^T, owned psi rows
    Mat :: H                                    !< H (u rows, psi columns)
    !> the first build's structure of H^T per group: entries and a hash of
    !! (rows, row pointers, columns); a rebuild must reproduce it
    integer(8), allocatable :: sig_nnz(:), sig_hash(:)
  end type hpatch_t

  PetscLogEvent, save :: ev_h = -1

  external :: dgbtrf, dgbtrs, dgetrf, dgetrs

  public :: hpatch_build, hpatch_destroy

contains

  !--------------------------------------------------------------------
  !> Build (first) or refill H. PJ = [[B_11, B_13], [B_31, B_33]] and DV =
  !! [[B_22 + W_c, B_24], [B_42, B_44]] packed; A21 = B_21, A23 = B_23 in
  !! the scalar layout; pmask = psi's mask (1 free, 0 Dirichlet), A21's row
  !! layout. Collective on A21's communicator; h%H is the result. A first
  !! build of a built h destroys it first. Timed by the event PhysPC_FluxH.
  !! Split-M (SPLIT-M above): q > 0 with Mw = M_w (scalar AIJ, A21's row
  !! layout, Dirichlet rows any) and vsg = varsigma; q is fixed by the first
  !! build.
  !--------------------------------------------------------------------
  subroutine hpatch_build(h, PJ, DV, A21, A23, pmask, ns, p, r, g, vel, first, ierr, label, Mw, q, vsg)
    type(hpatch_t), intent(inout) :: h
    Mat                           :: PJ, DV, A21, A23
    Vec                           :: pmask
    integer, intent(in)           :: ns       !< toroidal slots per index
    integer, intent(in)           :: p, r, g  !< patch radius, H's cut radius, group size (nodes)
    integer, intent(in)           :: vel      !< 1: the (u, omega) D_v patch, 0: B_22 + W_c only
    logical, intent(in)           :: first    !< the first build (structure)
    PetscErrorCode, intent(out)   :: ierr     !< 0 on success
    character(len=*), intent(in)  :: label    !< message prefix
    Mat, optional                 :: Mw       !< split-M: M_{R^3 rho0} (u's scalar layout)
    integer, intent(in), optional :: q        !< split-M: H_T's patch radius (>= p + 1); absent / 0: H-form
    real*8, intent(in), optional  :: vsg      !< split-M: varsigma
    PetscClassId, parameter :: cid = 0
    PetscErrorCode :: ie
    integer :: qq
    if (ev_h < 0) call PetscLogEventRegister("PhysPC_FluxH", cid, ev_h, ie)
    call PetscLogEventBegin(ev_h, ie)
    ierr = 0
    qq = 0
    if (present(q)) qq = q
    if (qq > 0 .and. (.not. present(Mw) .or. .not. present(vsg) .or. qq < p + 1)) then
      ierr = 1
      write(*,'(A,A,A)') "[Physics PC]   ", trim(label), ": H: split-M needs Mw, vsg and q >= p + 1"
    endif
    if (ierr == 0 .and. first .and. h%built) call hpatch_destroy(h, ierr)
    if (ierr == 0) then
      if (first) then
        h%q = qq
        if (qq > 0) h%vsg = vsg
      else if (qq /= h%q) then
        ierr = 1
        write(*,'(A,A,A)') "[Physics PC]   ", trim(label), ": H: q changed since the first build"
      endif
    endif
    if (ierr == 0) then
      if (h%q > 0) then
        h%vsg = vsg
        call hp_build(h, PJ, DV, A21, A23, pmask, ns, p, r, g, vel, first, ierr, label, Mw)
      else
        call hp_build(h, PJ, DV, A21, A23, pmask, ns, p, r, g, vel, first, ierr, label)
      endif
    endif
    call PetscLogEventEnd(ev_h, ie)
  end subroutine hpatch_build

  !> Release everything hpatch_build created (H, H^T, the structure).
  !! Collective (H lives on A21's communicator).
  subroutine hpatch_destroy(h, ierr)
    type(hpatch_t), intent(inout) :: h
    PetscErrorCode, intent(out)   :: ierr
    ierr = 0
    if (h%built) then
      call MatDestroy(h%H, ierr);  if (ierr /= 0) return
      call MatDestroy(h%Ht, ierr); if (ierr /= 0) return
    endif
    if (allocated(h%rg))       deallocate(h%rg)
    if (allocated(h%uix))      deallocate(h%uix)
    if (allocated(h%aptr))     deallocate(h%aptr)
    if (allocated(h%aadj))     deallocate(h%aadj)
    if (allocated(h%gptr))     deallocate(h%gptr)
    if (allocated(h%gix))      deallocate(h%gix)
    if (allocated(h%pptr))     deallocate(h%pptr)
    if (allocated(h%pix))      deallocate(h%pix)
    if (allocated(h%qptr))     deallocate(h%qptr)
    if (allocated(h%qix))      deallocate(h%qix)
    if (allocated(h%sig_nnz))  deallocate(h%sig_nnz)
    if (allocated(h%sig_hash)) deallocate(h%sig_hash)
    h%nu = 0; h%no = 0; h%o0 = 0; h%ngrp = 0
    h%built = .false.
  end subroutine hpatch_destroy

  !> hpatch_build's body (see there); Mw present = split-M (h%q > 0).
  subroutine hp_build(h, PJ, DV, A21, A23, pmask, ns, p, r, g, vel, first, ierr, label, Mw)
    type(hpatch_t), intent(inout) :: h
    Mat                           :: PJ, DV, A21, A23
    Vec                           :: pmask
    integer, intent(in)           :: ns       !< toroidal slots per index
    integer, intent(in)           :: p, r, g  !< patch radius, H's cut radius, group size (nodes)
    integer, intent(in)           :: vel      !< 1: the (u, omega) D_v patch, 0: B_22 + W_c only
    logical, intent(in)           :: first    !< the first build (structure)
    PetscErrorCode, intent(out)   :: ierr     !< 0 on success
    character(len=*), intent(in)  :: label    !< message prefix
    Mat, optional                 :: Mw       !< split-M: M_w
    type(grows_t) :: sj, su, sw, s21, s23, smw
    type(hres_t), allocatable :: res(:)
    logical, allocatable :: freep(:)
    PetscScalar, pointer :: pm(:)
    PetscInt :: nloc, i
    integer :: k, nbad, ndense, nrmax, bwmax, mierr, nfreep, nsys, nsd
    integer(8) :: nrsum, bwsum, nnzh
    real*8 :: flops, t0, st(4), mx(3)
    integer, allocatable :: lj(:), tj(:), lu(:), tu(:), lw(:), tw(:), lc(:), tc(:), tp(:), lp(:)
    integer, allocatable :: lq(:), tq(:), to(:), lo(:)
    integer, allocatable :: pp(:), px(:)
    integer :: ri, bi, di, ii, si, sd

    ierr = 0
    t0 = MPI_Wtime()

    !--- the structure: layout, U, adjacency, groups, patches (first build)
    if (first) then
      h%ns = ns; h%p = p; h%r = r; h%g = max(g, 1); h%vel = vel
      call PetscObjectGetComm(A21, h%comm, ierr); if (ierr /= 0) return
      call MPI_Comm_size(h%comm, h%np, mierr)
      call MPI_Comm_rank(h%comm, h%me, mierr)
      call MatGetOwnershipRange(A21, h%rs, h%re, ierr); if (ierr /= 0) return
      call MatGetSize(A21, h%n, PETSC_NULL_INTEGER, ierr); if (ierr /= 0) return
      allocate(h%rg(0:h%np))
      call MPI_Allgather(h%rs, 1, MPIU_INTEGER, h%rg(0:h%np - 1), 1, MPIU_INTEGER, h%comm, mierr)
      h%rg(h%np) = h%n
      nbad = 0
      if (mod(h%rs, int(ns, kind(h%rs))) /= 0 .or. mod(h%re, int(ns, kind(h%rs))) /= 0) nbad = 1
      ! the packed pairs: [field 1 | field 2] of A21's layout per rank (pk / unpk)
      block
        PetscInt :: a0, a1
        call MatGetOwnershipRange(PJ, a0, a1, ierr); if (ierr /= 0) return
        if (a0 /= 2 * h%rs .or. a1 /= 2 * h%re) nbad = 1
        call MatGetOwnershipRange(DV, a0, a1, ierr); if (ierr /= 0) return
        if (a0 /= 2 * h%rs .or. a1 /= 2 * h%re) nbad = 1
        call MatGetOwnershipRange(A23, a0, a1, ierr); if (ierr /= 0) return
        if (a0 /= h%rs .or. a1 /= h%re) nbad = 1
      end block
      call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_MAX, h%comm, mierr)
      if (nbad /= 0) then
        ierr = 1
        if (h%me == 0) write(*,'(A,A,A)') "[Physics PC]   ", trim(label), &
          ": H: the operators' layouts disagree (whole indices per rank, packed pairs of B_21's layout)"
        return
      endif
      call hp_ball(h, PJ, ierr); if (ierr /= 0) return
    endif

    !--- the values: the rows of U of every block the local systems read
    call hp_gather(h, PJ, 2, 2, sj, ierr); if (ierr /= 0) return       ! j rows: B_31, B_33
    call hp_gather(h, DV, 2, 1, su, ierr); if (ierr /= 0) return       ! u rows: B_22 + W_c, B_24
    if (h%vel == 1) then
      call hp_gather(h, DV, 2, 2, sw, ierr); if (ierr /= 0) return     ! omega rows: B_42, B_44
    endif
    call hp_gather(h, A21, 1, 1, s21, ierr); if (ierr /= 0) return
    call hp_gather(h, A23, 1, 1, s23, ierr); if (ierr /= 0) return
    if (h%q > 0) then
      call hp_gather(h, Mw, 1, 1, smw, ierr); if (ierr /= 0) return   ! M_w rows (split-M)
    endif
    if (first) then
      call hp_adjacency(h, sj)
      call hp_groups(h)
      call hp_patches(h, h%p, pp, px)
      call move_alloc(pp, h%pptr);  call move_alloc(px, h%pix)
      if (h%q > 0) then
        call hp_patches(h, h%q, pp, px)
        call move_alloc(pp, h%qptr);  call move_alloc(px, h%qix)
      endif
    endif

    !--- psi's free owned DOFs
    nloc = h%re - h%rs
    allocate(freep(0:max(nloc, one) - 1))
    freep = .false.
    call VecGetArrayRead(pmask, pm, ierr); if (ierr /= 0) return
    do i = 1, nloc
      freep(i - 1) = (real(pm(i), 8) /= 0.d0)
    enddo
    call VecRestoreArrayRead(pmask, pm, ierr); if (ierr /= 0) return
    nfreep = count(freep(0:nloc - 1))

    !--- the groups' local solves, on the threads
    allocate(res(h%ngrp))
    nbad = 0; ndense = 0; nrmax = 0; bwmax = 0; nrsum = 0; bwsum = 0; flops = 0.d0; nsys = 0; nsd = 0
    !$omp parallel private(lj, tj, lu, tu, lw, tw, lc, tc, tp, lp, lq, tq, to, lo, k, ri, bi, di, ii, si, sd)
    allocate(lj(0:h%nu * h%ns - 1), tj(0:h%nu * h%ns - 1), lu(0:h%nu * h%ns - 1), tu(0:h%nu * h%ns - 1), &
             lw(0:h%nu * h%ns - 1), tw(0:h%nu * h%ns - 1), lc(0:h%nu * h%ns - 1), tc(0:h%nu * h%ns - 1), &
             lq(0:h%nu * h%ns - 1), tq(0:h%nu * h%ns - 1), tp(h%nu), lp(h%nu), to(h%nu), lo(h%nu))
    tj = 0; tu = 0; tw = 0; tc = 0; tp = 0; lj = 0; lu = 0; lw = 0; lc = 0; lp = 0
    tq = 0; lq = 0; to = 0; lo = 0
    !$omp do schedule(dynamic, 1) reduction(+:nbad, ndense, nrsum, bwsum, flops, nsys, nsd) &
    !$omp reduction(max:nrmax, bwmax)
    do k = 1, h%ngrp
      call hp_group(h, k, sj, su, sw, s21, s23, smw, freep, res(k), lj, tj, lu, tu, lw, tw, lc, tc, tp, lp, &
                    lq, tq, to, lo, ri, bi, di, ii, si, sd, flops)
      nbad = nbad + ii; ndense = ndense + di
      nrsum = nrsum + ri; bwsum = bwsum + bi
      nrmax = max(nrmax, ri); bwmax = max(bwmax, bi)
      nsys = nsys + si; nsd = nsd + sd
    enddo
    !$omp end do
    deallocate(lj, tj, lu, tu, lw, tw, lc, tc, tp, lp, lq, tq, to, lo)
    !$omp end parallel
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, h%comm, mierr)
    if (nbad > 0) then
      ierr = 1
      if (h%me == 0) write(*,'(A,A,A,I0,A)') "[Physics PC]   ", trim(label), ": H: ", nbad, &
        " singular patch system(s) (LAPACK info /= 0)"
      return
    endif

    !--- the structure must be the first build's (collective check)
    call hp_signature(h, res, first, nbad)
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, h%comm, mierr)
    if (nbad > 0) then
      ierr = 1
      if (h%me == 0) write(*,'(A,A,A,I0,A)') "[Physics PC]   ", trim(label), ": H: the structure of H changed "// &
        "since the first build in ", nbad, " group(s) (Dirichlet sets or patches); a rebuild cannot refill it"
      return
    endif

    !--- H^T (owned psi rows), then H; a failure is collective (hp_assemble)
    call hp_assemble(h, res, first, nnzh, ierr)
    if (ierr /= 0) then
      if (h%me == 0) write(*,'(A,A,A)') "[Physics PC]   ", trim(label), &
        ": H: the assembly of H^T failed on some rank (PETSc error in MatSetValues / MatCreate)"
      return
    endif
    if (first) then
      call MatTranspose(h%Ht, MAT_INITIAL_MATRIX, h%H, ierr); if (ierr /= 0) return
    else
      call MatTranspose(h%Ht, MAT_REUSE_MATRIX, h%H, ierr); if (ierr /= 0) return
    endif
    h%built = .true.

    !--- first build: the summary
    if (first) then
      st = [dble(h%ngrp), dble(nrsum), dble(bwsum), flops]
      call MPI_Allreduce(MPI_IN_PLACE, st, 4, MPI_DOUBLE_PRECISION, MPI_SUM, h%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, nnzh, 1, MPI_INTEGER8, MPI_SUM, h%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, ndense, 1, MPI_INTEGER, MPI_SUM, h%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, nfreep, 1, MPI_INTEGER, MPI_SUM, h%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, nsys, 1, MPI_INTEGER, MPI_SUM, h%comm, mierr)
      call MPI_Allreduce(MPI_IN_PLACE, nsd, 1, MPI_INTEGER, MPI_SUM, h%comm, mierr)
      mx = [dble(nrmax), dble(bwmax), MPI_Wtime() - t0]
      call MPI_Allreduce(MPI_IN_PLACE, mx, 3, MPI_DOUBLE_PRECISION, MPI_MAX, h%comm, mierr)
      if (h%me == 0) then
        write(*,'(A,A,A,A,A,I0,A,I0,A,I0,A,I0,A,I0,A)') "[Physics PC]   ", trim(label), ": ", &
          trim(merge("split-M", "H-form ", h%q > 0)), ", ", nint(st(1)), &
          " groups (g = ", h%g, " nodes), patch radius p = ", h%p, ", cut r = ", h%r, ", velocity block ", &
          merge(2, 1, h%vel == 1), " field(s)"
        if (h%q > 0) write(*,'(A,A,A,I0,A,ES10.3,A)') "[Physics PC]   ", trim(label), &
          ": H: Tang part H_T = -(1/varsigma) M_w^-1 B_23 on radius-", h%q, " patches, varsigma ", h%vsg, &
          "; H's reach q"
        write(*,'(A,A,A,I0,A,I0,A)') "[Physics PC]   ", trim(label), ": H: ", nsd, " of ", nsys, &
          " patch systems slot-diagonal (solved per slot)"
        write(*,'(A,A,A,F9.1,A,I0,A,F8.1,A,I0,A,I0,A)') "[Physics PC]   ", trim(label), &
          ": H: patch rows (j + velocity) mean ", st(2) / max(st(1), 1.d0), " max ", nint(mx(1)), &
          ", band mean ", st(3) / max(st(1), 1.d0), " max ", nint(mx(2)), ", dense fallbacks ", ndense, ""
        write(*,'(A,A,A,ES10.3,A,I0,A,F8.1,A,F8.3,A)') "[Physics PC]   ", trim(label), &
          ": H: factor + solve flops ", st(4), ", nnz(H) ", nnzh, " (", dble(nnzh) / max(dble(nfreep), 1.d0), &
          " per free psi column), setup ", mx(3), " s (max over ranks)"
      endif
    endif

    call grows_free(sj); call grows_free(su); call grows_free(sw); call grows_free(s21); call grows_free(s23)
    call grows_free(smw)
    deallocate(res, freep)
  end subroutine hp_build

  !--------------------------------------------------------------------
  !> U = the indices within graph distance max(p, q) of the rank's own:
  !! that many halo expansions over B_33's structure (PJ's j rows, slot 0).
  !! Collective.
  !! The expansion follows row entries (out-edges) only; hp_adjacency then
  !! symmetrises inside U. The two agree because JOREK's stored block
  !! structure is symmetric: blocks (i, k) are created from the element
  !! connectivity, and the eliminated boundary blocks are skipped under a
  !! condition symmetric in (i, i_order) <-> (k, k_order)
  !! (construct_matrix_mod, add_block_to_petsc). A non-symmetric structure
  !! would only shrink a patch at the halo edge (the BFS stays inside U), not
  !! break it.
  !--------------------------------------------------------------------
  subroutine hp_ball(h, PJ, ierr)
    type(hpatch_t), intent(inout) :: h
    Mat                           :: PJ
    PetscErrorCode, intent(out)   :: ierr
    PetscInt, allocatable :: s(:), fr(:), nb(:), tmp(:)
    PetscInt :: i0, i1, i, pns
    integer :: lev, nn, a

    ierr = 0
    pns = int(h%ns, kind(pns))
    i0 = h%rs / pns;  i1 = h%re / pns
    s  = [(i, i = i0, i1 - 1)]
    fr = s
    do lev = 1, max(h%p, h%q)
      call hp_nbrs(h, PJ, fr, nb, ierr); if (ierr /= 0) return
      ! the new frontier: nb minus s (both sorted)
      allocate(tmp(size(nb)))
      nn = 0
      do a = 1, size(nb)
        if (pbsearch(s, nb(a)) == 0) then
          nn = nn + 1;  tmp(nn) = nb(a)
        endif
      enddo
      fr = tmp(1:nn)
      deallocate(tmp)
      s = [s, fr]
      call sort_uniq(s, nn)
      s = s(1:nn)
    enddo
    h%uix = s
    h%nu  = size(s)
    h%no  = int(i1 - i0)
    h%o0  = 0
    if (h%no > 0) h%o0 = pbsearch(h%uix, i0)
  end subroutine hp_ball

  !> The B_33 neighbours (indices, sorted unique) of the indices ix:
  !! gathered slot-0 j rows of PJ, field-2 columns. Collective.
  subroutine hp_nbrs(h, PJ, ix, nb, ierr)
    type(hpatch_t), intent(in)         :: h
    Mat                                :: PJ
    PetscInt, intent(in)               :: ix(:)
    PetscInt, allocatable, intent(out) :: nb(:)
    PetscErrorCode, intent(out)        :: ierr
    Mat, pointer :: sub(:)
    IS :: isr, isc
    PetscInt, allocatable :: rows(:), tmp(:)
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt :: nr, nc, k, q, ncg, x
    integer :: f, nt, nu

    nr = size(ix, kind=kind(nr))
    allocate(rows(max(nr, one)))
    do k = 1, nr
      rows(k) = pk(h, ix(k) * h%ns, 2, 2)
    enddo
    call MatGetSize(PJ, PETSC_NULL_INTEGER, ncg, ierr); if (ierr /= 0) return   ! the columns
    call ISCreateGeneral(PETSC_COMM_SELF, nr, rows, PETSC_COPY_VALUES, isr, ierr); if (ierr /= 0) return
    call ISCreateStride(PETSC_COMM_SELF, ncg, zero, one, isc, ierr); if (ierr /= 0) return
    call MatCreateSubMatrices(PJ, one, [isr], [isc], MAT_INITIAL_MATRIX, sub, ierr); if (ierr /= 0) return
    allocate(tmp(64))
    nt = 0
    do k = 0, nr - 1
      call MatGetRow(sub(1), k, nc, cols, vals, ierr); if (ierr /= 0) return
      if (nt + nc > size(tmp)) tmp = [tmp, (zero, q = 1, nt + nc + size(tmp))]
      do q = 1, nc
        call unpk(h, cols(q), 2, f, x)
        if (f /= 2) cycle
        nt = nt + 1;  tmp(nt) = x / h%ns
      enddo
      call MatRestoreRow(sub(1), k, nc, cols, vals, ierr); if (ierr /= 0) return
    enddo
    call MatDestroySubMatrices(one, sub, ierr); if (ierr /= 0) return
    call ISDestroy(isr, ierr); if (ierr /= 0) return
    call ISDestroy(isc, ierr); if (ierr /= 0) return
    nu = 0
    if (nt > 0) then
      tmp = tmp(1:nt)
      call sort_uniq(tmp, nu)
    endif
    nb = tmp(1:nu)
  end subroutine hp_nbrs

  !--------------------------------------------------------------------
  !> Gather rows of U (all slots) of Mg: field rfld of a packed pair (nfld =
  !! 2) or the scalar Mg (nfld = 1), all columns, into s. Collective.
  !--------------------------------------------------------------------
  subroutine hp_gather(h, Mg, nfld, rfld, s, ierr)
    type(hpatch_t), intent(in)  :: h
    Mat                         :: Mg
    integer, intent(in)         :: nfld, rfld
    type(grows_t), intent(out)  :: s
    PetscErrorCode, intent(out) :: ierr
    Mat, pointer :: sub(:)
    IS :: isr, isc
    PetscInt, allocatable :: rows(:)
    PetscInt, pointer :: cols(:)
    PetscScalar, pointer :: vals(:)
    PetscInt :: nr, nc, k, q, ncg, x
    integer :: a, m, f
    integer(8) :: e0
    real*8 :: dg, off

    nr = int(h%nu, kind(nr)) * h%ns
    allocate(rows(max(nr, one)))
    do a = 1, h%nu
      do m = 0, h%ns - 1
        x = h%uix(a) * h%ns + m
        if (nfld == 1) then
          rows((a - 1) * h%ns + m + 1) = x
        else
          rows((a - 1) * h%ns + m + 1) = pk(h, x, rfld, nfld)
        endif
      enddo
    enddo
    call MatGetSize(Mg, PETSC_NULL_INTEGER, ncg, ierr); if (ierr /= 0) return   ! the columns
    call ISCreateGeneral(PETSC_COMM_SELF, nr, rows, PETSC_COPY_VALUES, isr, ierr); if (ierr /= 0) return
    call ISCreateStride(PETSC_COMM_SELF, ncg, zero, one, isc, ierr); if (ierr /= 0) return
    call MatCreateSubMatrices(Mg, one, [isr], [isc], MAT_INITIAL_MATRIX, sub, ierr); if (ierr /= 0) return

    s%nrow = int(nr)
    allocate(s%ptr(s%nrow + 1), s%dir(max(s%nrow, 1)))
    s%ptr(1) = 0
    do k = 0, nr - 1
      call MatGetRow(sub(1), k, nc, cols, vals, ierr); if (ierr /= 0) return
      s%ptr(k + 2) = s%ptr(k + 1) + nc
      call MatRestoreRow(sub(1), k, nc, cols, vals, ierr); if (ierr /= 0) return
    enddo
    allocate(s%fld(max(s%ptr(s%nrow + 1), 1_8)), s%cid(max(s%ptr(s%nrow + 1), 1_8)), &
             s%val(max(s%ptr(s%nrow + 1), 1_8)))
    do k = 0, nr - 1
      call MatGetRow(sub(1), k, nc, cols, vals, ierr); if (ierr /= 0) return
      e0 = s%ptr(k + 1)
      dg = 0.d0; off = 0.d0
      do q = 1, nc
        if (nfld == 2) then
          call unpk(h, cols(q), nfld, f, x)
        else
          f = 1;  x = cols(q)
        endif
        s%fld(e0 + q) = int(f, 1)
        s%cid(e0 + q) = lid(h, x)
        s%val(e0 + q) = real(vals(q), 8)
        if (cols(q) == rows(k + 1)) then
          dg = real(vals(q), 8)
        else
          off = off + abs(vals(q))
        endif
      enddo
      s%dir(k + 1) = (off == 0.d0 .and. dg /= 0.d0)
      call MatRestoreRow(sub(1), k, nc, cols, vals, ierr); if (ierr /= 0) return
    enddo
    call MatDestroySubMatrices(one, sub, ierr); if (ierr /= 0) return
    call ISDestroy(isr, ierr); if (ierr /= 0) return
    call ISDestroy(isc, ierr); if (ierr /= 0) return
  end subroutine hp_gather

  subroutine grows_free(s)
    type(grows_t), intent(inout) :: s
    if (allocated(s%ptr)) deallocate(s%ptr)
    if (allocated(s%fld)) deallocate(s%fld)
    if (allocated(s%cid)) deallocate(s%cid)
    if (allocated(s%val)) deallocate(s%val)
    if (allocated(s%dir)) deallocate(s%dir)
    s%nrow = 0
  end subroutine grows_free

  !> The closed adjacency of U (positions): B_33's stored structure from the
  !! slot-0 j rows, symmetrised, each position itself included.
  subroutine hp_adjacency(h, sj)
    type(hpatch_t), intent(inout) :: h
    type(grows_t), intent(in)     :: sj
    integer, allocatable :: deg(:), fill(:)
    integer :: a, b, k, pass, n
    integer(8) :: e

    allocate(deg(h%nu), h%aptr(h%nu + 1))
    do pass = 1, 2
      deg = 0
      do a = 1, h%nu
        call put(a, a)
        k = (a - 1) * h%ns + 1                      ! slot 0's row
        do e = sj%ptr(k) + 1, sj%ptr(k + 1)
          if (sj%fld(e) /= 2 .or. sj%cid(e) < 0) cycle
          b = sj%cid(e) / h%ns + 1
          if (b == a) cycle
          call put(a, b);  call put(b, a)
        enddo
      enddo
      if (pass == 1) then
        h%aptr(1) = 0
        do a = 1, h%nu
          h%aptr(a + 1) = h%aptr(a) + deg(a)
        enddo
        allocate(h%aadj(max(h%aptr(h%nu + 1), 1)))
      endif
    enddo
    !--- sort and unique each list, then compact
    allocate(fill(h%nu + 1))
    fill(1) = 0
    do a = 1, h%nu
      n = h%aptr(a + 1) - h%aptr(a)
      call isort_uniq(h%aadj(h%aptr(a) + 1:h%aptr(a + 1)), n)
      h%aadj(fill(a) + 1:fill(a) + n) = h%aadj(h%aptr(a) + 1:h%aptr(a) + n)
      fill(a + 1) = fill(a) + n
    enddo
    h%aptr = fill
  contains
    subroutine put(x, y)
      integer, intent(in) :: x, y
      deg(x) = deg(x) + 1
      if (pass == 2) h%aadj(h%aptr(x) + deg(x)) = y
    end subroutine put
  end subroutine hp_adjacency

  !> Greedy aggregation of the owned indices into groups of g nodes (see
  !! the module header).
  subroutine hp_groups(h)
    type(hpatch_t), intent(inout) :: h
    logical, allocatable :: asg(:)
    integer, allocatable :: cand(:), score(:), buf(:)
    integer :: a, b, q, nc, nb, taken, ne, i, j, t1, t2

    allocate(asg(h%nu), h%gptr(h%no + 1), h%gix(max(h%no, 1)), cand(64), score(64), buf(64))
    asg = .false.
    h%ngrp = 0; ne = 0
    h%gptr(1) = 0
    do a = h%o0, h%o0 + h%no - 1
      if (asg(a)) cycle
      h%ngrp = h%ngrp + 1
      call take_node(a)
      !--- the unassigned owned neighbours, by common neighbours (desc), index (asc)
      nc = 0
      do q = h%aptr(a) + 1, h%aptr(a + 1)
        b = h%aadj(q)
        if (.not. owned(b) .or. asg(b)) cycle
        nc = nc + 1
        if (nc > size(cand)) then
          cand = [cand, cand];  score = [score, score]
        endif
        cand(nc) = b;  score(nc) = ncommon(a, b)
      enddo
      do i = 2, nc                                  ! insertion sort
        t1 = cand(i); t2 = score(i); j = i - 1
        do while (j >= 1)
          if (score(j) > t2 .or. (score(j) == t2 .and. cand(j) < t1)) exit
          cand(j + 1) = cand(j); score(j + 1) = score(j); j = j - 1
        enddo
        cand(j + 1) = t1; score(j + 1) = t2
      enddo
      taken = 0
      do i = 1, nc
        if (taken >= h%g - 1) exit
        if (asg(cand(i))) cycle                     ! taken with an earlier node
        call take_node(cand(i))
        taken = taken + 1
      enddo
      h%gptr(h%ngrp + 1) = ne
    enddo
  contains
    logical function owned(x)
      integer, intent(in) :: x
      owned = (x >= h%o0 .and. x < h%o0 + h%no)
    end function owned
    !> x's node: the unassigned owned neighbours with x's closed neighbourhood
    subroutine take_node(x)
      integer, intent(in) :: x
      integer :: qq, y
      nb = 0
      do qq = h%aptr(x) + 1, h%aptr(x + 1)
        y = h%aadj(qq)
        if (.not. owned(y) .or. asg(y)) cycle
        if (.not. same_list(x, y)) cycle
        nb = nb + 1
        if (nb > size(buf)) buf = [buf, buf]
        buf(nb) = y
      enddo
      do qq = 1, nb
        asg(buf(qq)) = .true.
        ne = ne + 1;  h%gix(ne) = buf(qq)
      enddo
    end subroutine take_node
    logical function same_list(x, y)
      integer, intent(in) :: x, y
      integer :: lx, ly
      lx = h%aptr(x + 1) - h%aptr(x);  ly = h%aptr(y + 1) - h%aptr(y)
      same_list = (lx == ly)
      if (same_list) same_list = all(h%aadj(h%aptr(x) + 1:h%aptr(x + 1)) == h%aadj(h%aptr(y) + 1:h%aptr(y + 1)))
    end function same_list
    integer function ncommon(x, y)
      integer, intent(in) :: x, y
      integer :: px, py
      ncommon = 0
      px = h%aptr(x) + 1;  py = h%aptr(y) + 1
      do while (px <= h%aptr(x + 1) .and. py <= h%aptr(y + 1))
        if (h%aadj(px) == h%aadj(py)) then
          ncommon = ncommon + 1;  px = px + 1;  py = py + 1
        else if (h%aadj(px) < h%aadj(py)) then
          px = px + 1
        else
          py = py + 1
        endif
      enddo
    end function ncommon
  end subroutine hp_groups

  !> Each group's patch: the U positions within distance rad (BFS from the
  !! group), in reverse Cuthill-McKee order, into pptr / pix.
  subroutine hp_patches(h, rad, pptr, pix)
    type(hpatch_t), intent(in)          :: h
    integer, intent(in)                 :: rad    !< patch radius (graph distance)
    integer, allocatable, intent(out)   :: pptr(:), pix(:)
    integer, allocatable :: mark(:), dist(:), lst(:), deg(:), ord(:), nbl(:)
    integer :: k, q, n, lo, a, b, e, i, j, cnt, s0, t1, nn, ntot

    allocate(mark(h%nu), dist(h%nu), lst(h%nu), deg(h%nu), ord(h%nu), nbl(h%nu))
    mark = 0
    allocate(pptr(h%ngrp + 1), pix(64))
    pptr(1) = 0
    ntot = 0
    do k = 1, h%ngrp
      !--- BFS from the group, depth rad
      n = 0
      do q = h%gptr(k) + 1, h%gptr(k + 1)
        n = n + 1;  lst(n) = h%gix(q);  mark(lst(n)) = k;  dist(lst(n)) = 0
      enddo
      lo = 1
      do while (lo <= n)
        a = lst(lo);  lo = lo + 1
        if (dist(a) >= rad) cycle
        do e = h%aptr(a) + 1, h%aptr(a + 1)
          b = h%aadj(e)
          if (mark(b) == k) cycle
          mark(b) = k;  n = n + 1;  lst(n) = b;  dist(b) = dist(a) + 1
        enddo
      enddo
      !--- degrees inside the patch
      do i = 1, n
        a = lst(i);  cnt = 0
        do e = h%aptr(a) + 1, h%aptr(a + 1)
          if (mark(h%aadj(e)) == k .and. h%aadj(e) /= a) cnt = cnt + 1
        enddo
        deg(a) = cnt;  dist(a) = -1                 ! dist reused: -1 = not yet ordered
      enddo
      !--- Cuthill-McKee from a minimum-degree start, per component
      nn = 0
      do while (nn < n)
        s0 = 0
        do i = 1, n
          if (dist(lst(i)) /= -1) cycle
          if (s0 == 0) then
            s0 = lst(i)
          else if (deg(lst(i)) < deg(s0)) then
            s0 = lst(i)
          endif
        enddo
        nn = nn + 1;  ord(nn) = s0;  dist(s0) = 0
        lo = nn
        do while (lo <= nn)
          a = ord(lo);  lo = lo + 1
          cnt = 0
          do e = h%aptr(a) + 1, h%aptr(a + 1)
            b = h%aadj(e)
            if (mark(b) /= k .or. dist(b) /= -1) cycle
            dist(b) = 0;  cnt = cnt + 1;  nbl(cnt) = b
          enddo
          do i = 2, cnt                             ! by degree ascending
            t1 = nbl(i);  j = i - 1
            do while (j >= 1)
              if (deg(nbl(j)) <= deg(t1)) exit
              nbl(j + 1) = nbl(j);  j = j - 1
            enddo
            nbl(j + 1) = t1
          enddo
          ord(nn + 1:nn + cnt) = nbl(1:cnt)
          nn = nn + cnt
        enddo
      enddo
      if (ntot + n > size(pix)) pix = [pix, (0, i = 1, max(ntot + n, 2 * size(pix)) - size(pix))]
      do i = 1, n                                    ! reversed
        pix(ntot + i) = ord(n + 1 - i)
      enddo
      ntot = ntot + n
      pptr(k + 1) = ntot
    enddo
    pix = pix(1:max(ntot, 1))
  end subroutine hp_patches

  !--------------------------------------------------------------------
  !> One group: the patch systems, solved; res = its rows of H^T. Maps
  !! (stamp = k): lj / lu / lw / lc / lq local j, velocity (u, omega),
  !! column and (split-M) mass numbers by local id; tp / lp the patch's,
  !! to / lo the outer patch's (radius q under split-M, else the patch)
  !! membership and position by U position. Returns nrow (j + velocity
  !! rows), band (the largest system's kl + ku + 1), dense (dense LU used),
  !! bad (LAPACK failures), nsys / nsd (local systems, of them solved slot
  !! by slot), and adds its flops.
  !--------------------------------------------------------------------
  subroutine hp_group(h, k, sj, su, sw, s21, s23, smw, freep, res, lj, tj, lu, tu, lw, tw, lc, tc, tp, lp, &
                      lq, tq, to, lo, nrow, band, dense, bad, nsys, nsd, flops)
    type(hpatch_t), intent(in)  :: h
    integer, intent(in)         :: k
    type(grows_t), intent(in)   :: sj, su, sw, s21, s23, smw
    logical, intent(in)         :: freep(0:)
    type(hres_t), intent(inout) :: res
    integer, intent(inout)      :: lj(0:), tj(0:), lu(0:), tu(0:), lw(0:), tw(0:), lc(0:), tc(0:), tp(:), lp(:)
    integer, intent(inout)      :: lq(0:), tq(0:), to(:), lo(:)
    integer, intent(out)        :: nrow, band, dense, bad, nsys, nsd
    real*8, intent(inout)       :: flops
    integer, allocatable :: did(:), dkind(:), jid(:), cgi(:), dd(:, :), que(:), qid(:)
    PetscInt, allocatable :: crow(:)
    real*8, allocatable :: Xs(:, :), Rm(:, :), v23(:), HT(:, :)
    integer, allocatable :: rp23(:), cj23(:)
    integer, allocatable :: opix(:)
    integer :: np_, npo, ns, q, pos, m, id, nj, nd, nc, nq, ngi, gi, a, b, l0, n, e2, cnt, c, ub, dmax, dst
    integer(8) :: e
    PetscInt :: x
    integer :: d1, d2, d3, b1, b2, b3
    logical :: split
    real*8 :: v
    !--- the local system being assembled (solve_sys, solve_rows, row_entries, put)
    real*8, allocatable :: ab(:, :)
    integer, allocatable :: smap(:)
    integer :: kl, ku, ldab, pass, dns, ckw
    real*8 :: rmax, omax

    nrow = 0; band = 0; dense = 0; bad = 0; nsys = 0; nsd = 0
    ns = h%ns
    split = (h%q > 0)
    np_ = h%pptr(k + 1) - h%pptr(k)
    do q = 1, np_
      pos = h%pix(h%pptr(k) + q);  tp(pos) = k;  lp(pos) = q
    enddo
    !--- the outer patch: H_T's (radius q) under split-M, else the patch
    if (split) then
      npo = h%qptr(k + 1) - h%qptr(k)
      opix = h%qix(h%qptr(k) + 1:h%qptr(k + 1))
    else
      npo = np_
      opix = h%pix(h%pptr(k) + 1:h%pptr(k + 1))
    endif
    do q = 1, npo
      to(opix(q)) = k;  lo(opix(q)) = q
    enddo

    !--- local numbers: j rows, velocity rows (u, then omega per index), columns
    nj = 0;  nd = 0
    allocate(did(np_ * ns * 2), dkind(np_ * ns * 2), jid(np_ * ns))
    do q = 1, np_
      pos = h%pix(h%pptr(k) + q)
      do m = 0, ns - 1
        id = (pos - 1) * ns + m
        if (.not. sj%dir(id + 1)) then
          nj = nj + 1;  lj(id) = nj;  tj(id) = k;  jid(nj) = id
        endif
      enddo
      do m = 0, ns - 1
        id = (pos - 1) * ns + m
        if (.not. su%dir(id + 1)) then
          nd = nd + 1;  lu(id) = nd;  tu(id) = k;  did(nd) = id;  dkind(nd) = 1
        endif
      enddo
      if (h%vel == 1) then
        do m = 0, ns - 1
          id = (pos - 1) * ns + m
          if (.not. sw%dir(id + 1)) then
            nd = nd + 1;  lw(id) = nd;  tw(id) = k;  did(nd) = id;  dkind(nd) = 2
          endif
        enddo
      endif
    enddo
    ngi = h%gptr(k + 1) - h%gptr(k)
    allocate(cgi(ngi * ns), crow(ngi * ns))
    nc = 0
    do gi = 1, ngi
      pos = h%gix(h%gptr(k) + gi)
      do m = 0, ns - 1
        x = h%uix(pos) * ns + m
        if (.not. freep(x - h%rs)) cycle
        id = (pos - 1) * ns + m
        nc = nc + 1;  lc(id) = nc;  tc(id) = k;  cgi(nc) = gi;  crow(nc) = x
      enddo
    enddo
    nrow = nj + nd
    res%nr = 0
    if (nc == 0 .or. nd == 0) return
    b1 = 0;  d1 = 0;  b3 = 0;  d3 = 0

    !--- X = B_33^-1 B_31 on the patch (j rows: field 2 = j, field 1 = psi);
    !--- no free j in the patch: X = 0 and R = -B_21 below
    if (nj > 0) then
      allocate(Xs(nj, nc))
      Xs = 0.d0
      do q = 1, np_
        pos = h%pix(h%pptr(k) + q)
        do m = 0, ns - 1
          id = (pos - 1) * ns + m
          if (tj(id) /= k) cycle
          do e = sj%ptr(id + 1) + 1, sj%ptr(id + 2)
            if (sj%fld(e) /= 1 .or. sj%cid(e) < 0) cycle
            if (tc(sj%cid(e)) == k) Xs(lj(id), lc(sj%cid(e))) = Xs(lj(id), lc(sj%cid(e))) + sj%val(e)
          enddo
        enddo
      enddo
      call solve_sys(1, nj, Xs, b1, d1)
      if (bad /= 0) return
    endif

    !--- R = B_23 X - B_21 (velocity rows; the omega rows' are zero). B_23's
    !--- patch entries are listed per row first, so the product runs column by
    !--- column over Xs and Rm (column-major)
    allocate(Rm(nd, nc))
    Rm = 0.d0
    if (nj > 0) then
      allocate(rp23(nd + 1))
      do pass = 1, 2                                ! count, then fill
        cnt = 0
        do a = 1, nd
          rp23(a) = cnt
          if (dkind(a) /= 1) cycle
          id = did(a)
          do e = s23%ptr(id + 1) + 1, s23%ptr(id + 2)
            if (s23%cid(e) < 0) cycle
            if (tj(s23%cid(e)) /= k) cycle
            cnt = cnt + 1
            if (pass == 2) then
              cj23(cnt) = lj(s23%cid(e));  v23(cnt) = s23%val(e)
            endif
          enddo
        enddo
        rp23(nd + 1) = cnt
        if (pass == 1) allocate(cj23(max(cnt, 1)), v23(max(cnt, 1)))
      enddo
      do c = 1, nc
        do a = 1, nd
          do b = rp23(a) + 1, rp23(a + 1)
            Rm(a, c) = Rm(a, c) + v23(b) * Xs(cj23(b), c)
          enddo
        enddo
      enddo
      deallocate(Xs, rp23, cj23, v23)
    endif
    do a = 1, nd
      if (dkind(a) /= 1) cycle
      id = did(a)
      do e = s21%ptr(id + 1) + 1, s21%ptr(id + 2)
        if (s21%cid(e) < 0) cycle
        if (tc(s21%cid(e)) == k) Rm(a, lc(s21%cid(e))) = Rm(a, lc(s21%cid(e))) - s21%val(e)
      enddo
    enddo
    !--- distances from each group index inside the outer patch: the cuts r
    !--- and (split-M) q, exact for radii up to the outer patch's own
    dmax = h%r
    if (split) dmax = max(h%r, h%q)
    allocate(dd(npo, ngi), que(npo))
    dd = huge(1)
    do gi = 1, ngi
      pos = h%gix(h%gptr(k) + gi)
      n = 1;  que(1) = pos;  dd(lo(pos), gi) = 0
      l0 = 1
      do while (l0 <= n)
        a = que(l0);  l0 = l0 + 1
        if (dd(lo(a), gi) >= dmax) cycle
        do e2 = h%aptr(a) + 1, h%aptr(a + 1)
          b = h%aadj(e2)
          if (to(b) /= k) cycle
          if (dd(lo(b), gi) /= huge(1)) cycle
          dd(lo(b), gi) = dd(lo(a), gi) + 1
          n = n + 1;  que(n) = b
        enddo
      enddo
    enddo

    !--- split-M: Tang's part H_T = -(1/varsigma) M_w[Q]^-1 B_23[Q, cols] on
    !--- the outer patch's free u rows (u's Dirichlet set), cut to q; the
    !--- velocity rhs then becomes the remainder R - D_v H_T (u rows: B_22 +
    !--- W_c; omega rows with vel 1: B_42). D_v's rows on the patch reach one
    !--- node beyond it, inside Q as q >= p + 1.
    nq = 0
    if (split) then
      allocate(qid(npo * ns))
      do q = 1, npo
        pos = opix(q)
        do m = 0, ns - 1
          id = (pos - 1) * ns + m
          if (.not. su%dir(id + 1)) then
            nq = nq + 1;  lq(id) = nq;  tq(id) = k;  qid(nq) = id
          endif
        enddo
      enddo
      allocate(HT(nq, nc))
      HT = 0.d0
      do a = 1, nq
        id = qid(a)
        do e = s23%ptr(id + 1) + 1, s23%ptr(id + 2)
          if (s23%cid(e) < 0) cycle
          if (tc(s23%cid(e)) == k) HT(a, lc(s23%cid(e))) = HT(a, lc(s23%cid(e))) + s23%val(e)
        enddo
      enddo
      call solve_sys(3, nq, HT, b3, d3)
      if (bad /= 0) return
      do c = 1, nc
        do a = 1, nq
          if (dd(lo(qid(a) / ns + 1), cgi(c)) > h%q) then
            HT(a, c) = 0.d0
          else
            HT(a, c) = -HT(a, c) / h%vsg
          endif
        enddo
      enddo
      do a = 1, nd
        id = did(a)
        if (dkind(a) == 1) then
          do e = su%ptr(id + 1) + 1, su%ptr(id + 2)
            if (su%fld(e) /= 1 .or. su%cid(e) < 0) cycle
            if (tq(su%cid(e)) /= k) cycle
            Rm(a, :) = Rm(a, :) - su%val(e) * HT(lq(su%cid(e)), :)
          enddo
        else
          do e = sw%ptr(id + 1) + 1, sw%ptr(id + 2)
            if (sw%fld(e) /= 1 .or. sw%cid(e) < 0) cycle
            if (tq(sw%cid(e)) /= k) cycle
            Rm(a, :) = Rm(a, :) - sw%val(e) * HT(lq(sw%cid(e)), :)
          enddo
        endif
      enddo
    endif

    call solve_sys(2, nd, Rm, b2, d2)
    if (bad /= 0) return
    band  = max(b1, b2, b3)
    dense = max(d1, d2, d3)

    !--- H's entries: Y (Rm) cut to r on the patch's u rows; split-M adds
    !--- H_T, whose rows are all of the outer patch's u rows within q (H's
    !--- structure depends on the distances only, so a rebuild reproduces it)
    res%nr = nc
    allocate(res%row(nc), res%rp(nc + 1))
    res%row = crow(1:nc)
    res%rp(1) = 0
    cnt = 0
    do c = 1, nc
      if (split) then
        do a = 1, nq
          if (dd(lo(qid(a) / ns + 1), cgi(c)) <= h%q) cnt = cnt + 1
        enddo
      else
        do a = 1, nd
          if (dkind(a) /= 1) cycle
          if (dd(lo(did(a) / ns + 1), cgi(c)) <= h%r) cnt = cnt + 1
        enddo
      endif
      res%rp(c + 1) = cnt
    enddo
    allocate(res%col(max(cnt, 1)), res%val(max(cnt, 1)))
    cnt = 0
    do c = 1, nc
      if (split) then
        do a = 1, nq
          id = qid(a);  ub = id / ns + 1
          dst = dd(lo(ub), cgi(c))
          if (dst > h%q) cycle
          v = HT(a, c)
          if (dst <= h%r .and. tu(id) == k) v = v + Rm(lu(id), c)    ! within r <= p: in the patch
          cnt = cnt + 1
          res%col(cnt) = h%uix(ub) * ns + mod(id, ns)
          res%val(cnt) = v
        enddo
      else
        do a = 1, nd
          if (dkind(a) /= 1) cycle
          ub = did(a) / ns + 1
          if (dd(lo(ub), cgi(c)) > h%r) cycle
          cnt = cnt + 1
          res%col(cnt) = h%uix(ub) * ns + mod(did(a), ns)
          res%val(cnt) = Rm(a, c)
        enddo
      endif
    enddo
    deallocate(Rm, dd, que, did, dkind, jid, cgi, crow)

  contains

    !> Slot of local row rr of system wh (1: j, 2: velocity, 3: mass).
    integer function rslot(wh, rr)
      integer, intent(in) :: wh, rr
      select case (wh)
      case (1);     rslot = mod(jid(rr), ns)
      case (2);     rslot = mod(did(rr), ns)
      case default; rslot = mod(qid(rr), ns)
      end select
    end function rslot

    !> Bm (nn x ncol) <- A^-1 Bm for the patch system `which` (1: B_33 on the
    !! j numbering; 2: the velocity block on the u / omega numbering; 3: M_w
    !! on the mass numbering): slot by slot when it is slot-diagonal
    !! (slot_diag), else coupled. bw = the largest kl + ku + 1, dn = 1 when a
    !! solve went dense; a failure counts in bad.
    subroutine solve_sys(which, nn, Bm, bw, dn)
      integer, intent(in)    :: which, nn
      real*8, intent(inout)  :: Bm(:, :)
      integer, intent(out)   :: bw, dn
      integer, allocatable :: rl(:)
      integer :: mm, cn, rr, bw1, dn1
      logical :: sdg
      nsys = nsys + 1
      bw = 0;  dn = 0
      allocate(rl(nn), smap(nn))
      sdg = .false.
      if (ns > 1) sdg = slot_diag(which, nn)
      if (sdg) then
        nsd = nsd + 1
        do mm = 0, ns - 1
          cn = 0
          do rr = 1, nn
            if (rslot(which, rr) == mm) then
              cn = cn + 1;  rl(cn) = rr
            endif
          enddo
          if (cn == 0) cycle
          call solve_rows(which, cn, rl, Bm, bw1, dn1)
          bw = max(bw, bw1);  dn = max(dn, dn1)
          if (bad /= 0) exit
        enddo
      else
        rl = [(rr, rr = 1, nn)]
        call solve_rows(which, nn, rl, Bm, bw, dn)
      endif
      deallocate(rl, smap)
    end subroutine solve_sys

    !> Is system `which` slot-diagonal: in every row, the off-slot entries
    !! at most SD_TOL x the row's largest |entry|? (put's check mode, pass 0)
    logical function slot_diag(which, nn)
      integer, intent(in) :: which, nn
      integer :: rr
      slot_diag = .true.
      pass = 0;  ckw = which
      do rr = 1, nn
        rmax = 0.d0;  omax = 0.d0
        call row_entries(which, rr)
        if (omax > SD_TOL * rmax) then
          slot_diag = .false.
          return
        endif
      enddo
    end function slot_diag

    !> Rows rl(1:nr) of system `which` (one slot, or all) solved for Bm(rl,
    !! :): banded LU in their order (the patch's RCM order), dense where the
    !! band storage is larger. Entries to rows outside rl are left out (none,
    !! or below SD_TOL by slot_diag).
    subroutine solve_rows(which, nr, rl, Bm, bw, dn)
      integer, intent(in)    :: which, nr
      integer, intent(in)    :: rl(:)
      real*8, intent(inout)  :: Bm(:, :)
      integer, intent(out)   :: bw, dn
      integer, allocatable :: ipiv(:)
      real*8, allocatable :: Bs(:, :)
      integer :: info, rr, ncl

      ncl = size(Bm, 2)
      smap = 0
      do rr = 1, nr
        smap(rl(rr)) = rr
      enddo
      kl = 0; ku = 0; dns = 0; ldab = 1
      allocate(ipiv(nr))
      do pass = 1, 2
        if (pass == 2) then
          if (2 * kl + ku + 1 >= nr) then
            dns = 1
            allocate(ab(nr, nr))
          else
            ldab = 2 * kl + ku + 1
            allocate(ab(ldab, nr))
          endif
          ab = 0.d0
        endif
        do rr = 1, nr
          call row_entries(which, rl(rr))
        enddo
      enddo
      bw = kl + ku + 1;  dn = dns
      allocate(Bs(nr, ncl))
      Bs = Bm(rl(1:nr), :)
      if (dns == 1) then
        call dgetrf(nr, nr, ab, nr, ipiv, info)
        if (info == 0) call dgetrs('N', nr, ncl, ab, nr, ipiv, Bs, nr, info)
        flops = flops + 2.d0 / 3.d0 * dble(nr)**3 + 2.d0 * dble(nr)**2 * dble(ncl)
      else
        call dgbtrf(nr, nr, kl, ku, ab, ldab, ipiv, info)
        if (info == 0) call dgbtrs('N', nr, kl, ku, ncl, ab, ldab, ipiv, Bs, nr, info)
        ! dgbtrf with partial pivoting: U's band grows to kl + ku, so each of
        ! the n columns updates a kl x (kl + ku) block; dgbtrs: L (kl) and U
        ! (kl + ku) per right-hand side
        flops = flops + 2.d0 * dble(nr) * dble(kl) * dble(kl + ku) &
                      + 2.d0 * dble(nr) * dble(2 * kl + ku) * dble(ncl)
      endif
      if (info /= 0) bad = bad + 1
      Bm(rl(1:nr), :) = Bs
      deallocate(ab, ipiv, Bs)
    end subroutine solve_rows

    !> Row `row` of system wh: its entries in the local numbering, to put
    !! (check mode, band or values).
    subroutine row_entries(wh, row)
      integer, intent(in) :: wh, row
      integer :: idr, kd
      integer(8) :: ee
      if (wh == 1) then
        ! j row: B_33 = its j columns
        idr = jid(row)
        do ee = sj%ptr(idr + 1) + 1, sj%ptr(idr + 2)
          if (sj%fld(ee) /= 2 .or. sj%cid(ee) < 0) cycle
          if (tj(sj%cid(ee)) /= k) cycle
          call put(row, lj(sj%cid(ee)), sj%val(ee))
        enddo
      else if (wh == 3) then
        ! mass row: M_w's u columns in Q
        idr = qid(row)
        do ee = smw%ptr(idr + 1) + 1, smw%ptr(idr + 2)
          if (smw%cid(ee) < 0) cycle
          if (tq(smw%cid(ee)) /= k) cycle
          call put(row, lq(smw%cid(ee)), smw%val(ee))
        enddo
      else
        ! velocity row: u (B_22 + W_c, B_24) or omega (B_42, B_44)
        idr = did(row);  kd = dkind(row)
        if (kd == 1) then
          do ee = su%ptr(idr + 1) + 1, su%ptr(idr + 2)
            if (su%cid(ee) < 0) cycle
            if (su%fld(ee) == 1) then
              if (tu(su%cid(ee)) == k) call put(row, lu(su%cid(ee)), su%val(ee))
            else if (h%vel == 1) then
              if (tw(su%cid(ee)) == k) call put(row, lw(su%cid(ee)), su%val(ee))
            endif
          enddo
        else
          do ee = sw%ptr(idr + 1) + 1, sw%ptr(idr + 2)
            if (sw%cid(ee) < 0) cycle
            if (sw%fld(ee) == 1) then
              if (tu(sw%cid(ee)) == k) call put(row, lu(sw%cid(ee)), sw%val(ee))
            else
              if (tw(sw%cid(ee)) == k) call put(row, lw(sw%cid(ee)), sw%val(ee))
            endif
          enddo
        endif
      endif
    end subroutine row_entries

    !> One entry (ii_, jj_) of the system (local rows): slot_diag's check
    !! (pass 0), or, mapped to solve_rows' rows by smap (other rows left
    !! out), the band (pass 1) or its value (pass 2).
    subroutine put(ii_, jj_, v_)
      integer, intent(in) :: ii_, jj_
      real*8, intent(in)  :: v_
      integer :: i2, j2
      if (pass == 0) then
        rmax = max(rmax, abs(v_))
        if (rslot(ckw, ii_) /= rslot(ckw, jj_)) omax = max(omax, abs(v_))
        return
      endif
      i2 = smap(ii_);  j2 = smap(jj_)
      if (j2 == 0) return
      if (pass == 1) then
        kl = max(kl, i2 - j2);  ku = max(ku, j2 - i2)
      else if (dns == 1) then
        ab(i2, j2) = ab(i2, j2) + v_
      else
        ab(kl + ku + 1 + i2 - j2, j2) = ab(kl + ku + 1 + i2 - j2, j2) + v_
      endif
    end subroutine put

  end subroutine hp_group

  !> The groups' structure signatures (entries; a hash of rows, row
  !! pointers and columns): stored at the first build, compared after.
  !! nbad = the groups whose structure differs (local).
  subroutine hp_signature(h, res, first, nbad)
    type(hpatch_t), intent(inout) :: h
    type(hres_t), intent(in)      :: res(:)
    logical, intent(in)           :: first
    integer, intent(out)          :: nbad
    integer(8), parameter :: PRIME = 2147483647_8, MUL = 1000003_8
    integer(8) :: hh, nz
    integer :: k, q, e
    nbad = 0
    if (first) then
      if (allocated(h%sig_nnz)) deallocate(h%sig_nnz, h%sig_hash)
      allocate(h%sig_nnz(size(res)), h%sig_hash(size(res)))
    endif
    do k = 1, size(res)
      hh = int(res(k)%nr, 8);  nz = 0
      do q = 1, res(k)%nr
        hh = mod(hh * MUL + mod(int(res(k)%row(q), 8), PRIME) + 1_8, PRIME)
        hh = mod(hh * MUL + int(res(k)%rp(q + 1), 8) + 1_8, PRIME)
        do e = res(k)%rp(q) + 1, res(k)%rp(q + 1)
          hh = mod(hh * MUL + mod(int(res(k)%col(e), 8), PRIME) + 1_8, PRIME)
        enddo
        nz = nz + (res(k)%rp(q + 1) - res(k)%rp(q))
      enddo
      if (first) then
        h%sig_nnz(k) = nz;  h%sig_hash(k) = hh
      else if (h%sig_nnz(k) /= nz .or. h%sig_hash(k) /= hh) then
        nbad = nbad + 1
      endif
    enddo
  end subroutine hp_signature

  !> H^T from the groups' rows: created (exact preallocation, frozen
  !! pattern) on the first build, zeroed and refilled after (the structure
  !! checked by hp_signature). nnzh = its local entries. A failure on any
  !! rank is returned on all (ierr /= 0 everywhere, no rank left in a
  !! collective call).
  subroutine hp_assemble(h, res, first, nnzh, ierr)
    type(hpatch_t), intent(inout) :: h
    type(hres_t), intent(in)      :: res(:)
    logical, intent(in)           :: first
    integer(8), intent(out)       :: nnzh
    PetscErrorCode, intent(out)   :: ierr
    PetscInt, allocatable :: dnz(:), onz(:)
    PetscInt :: nloc, nr_, row
    integer :: k, q, e, bad, mierr
    ierr = 0
    nloc = h%re - h%rs
    nnzh = 0
    bad = 0
    if (first) then
      allocate(dnz(max(nloc, one)), onz(max(nloc, one)))
      dnz = 0; onz = 0
      do k = 1, size(res)
        do q = 1, res(k)%nr
          row = res(k)%row(q) - h%rs + 1
          do e = res(k)%rp(q) + 1, res(k)%rp(q + 1)
            if (res(k)%col(e) >= h%rs .and. res(k)%col(e) < h%re) then
              dnz(row) = dnz(row) + 1
            else
              onz(row) = onz(row) + 1
            endif
          enddo
        enddo
      enddo
      ! collective calls: no early return (a failure is flagged, then shared)
      call MatCreate(h%comm, h%Ht, ierr)
      call MatSetSizes(h%Ht, nloc, nloc, h%n, h%n, ierr)
      call MatSetType(h%Ht, MATMPIAIJ, ierr)
      call MatMPIAIJSetPreallocation(h%Ht, zero, dnz, zero, onz, ierr);  if (ierr /= 0) bad = 1
      call MatSetOption(h%Ht, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(h%Ht, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE, ierr)
      deallocate(dnz, onz)
    else
      ! every position is rewritten below; zeroing first leaves no stale value
      call MatZeroEntries(h%Ht, ierr);  if (ierr /= 0) bad = 1
    endif
    ins: do k = 1, size(res)
      do q = 1, res(k)%nr
        nr_ = int(res(k)%rp(q + 1) - res(k)%rp(q), kind(nr_))
        nnzh = nnzh + nr_
        if (nr_ == 0 .or. bad /= 0) cycle
        call MatSetValues(h%Ht, one, [res(k)%row(q)], nr_, res(k)%col(res(k)%rp(q) + 1:res(k)%rp(q + 1)), &
                          res(k)%val(res(k)%rp(q) + 1:res(k)%rp(q + 1)), INSERT_VALUES, ierr)
        if (ierr /= 0) then
          bad = 1
          write(*,'(A,I0,A,I0)') "[Physics PC]   H: MatSetValues failed on rank ", h%me, ", psi row ", &
            res(k)%row(q)
          exit ins
        endif
      enddo
    enddo ins
    call MPI_Allreduce(MPI_IN_PLACE, bad, 1, MPI_INTEGER, MPI_MAX, h%comm, mierr)
    if (bad /= 0) then
      ierr = 1
      return
    endif
    call MatAssemblyBegin(h%Ht, MAT_FINAL_ASSEMBLY, ierr); if (ierr /= 0) return
    call MatAssemblyEnd(h%Ht, MAT_FINAL_ASSEMBLY, ierr); if (ierr /= 0) return
    if (first) then
      call MatSetOption(h%Ht, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr); if (ierr /= 0) return
    endif
  end subroutine hp_assemble

  !--------------------------------------------------------------------
  ! layout helpers: scalar row x (= index ns + slot), packed pairs
  !--------------------------------------------------------------------

  !> The rank owning scalar row x.
  integer function owner(h, x)
    type(hpatch_t), intent(in) :: h
    PetscInt, intent(in)       :: x
    integer :: lo, hi, mid
    lo = 0; hi = h%np - 1
    do while (lo < hi)
      mid = (lo + hi + 1) / 2
      if (h%rg(mid) <= x) then
        lo = mid
      else
        hi = mid - 1
      endif
    enddo
    owner = lo
  end function owner

  !> Packed row / column of field f of scalar x (pack_blocks_aij's layout,
  !! nfld fields of one layout: rank q owns [field 1 | ... | field nfld]).
  PetscInt function pk(h, x, f, nfld)
    type(hpatch_t), intent(in) :: h
    PetscInt, intent(in)       :: x
    integer, intent(in)        :: f, nfld
    integer :: q
    q = owner(h, x)
    pk = nfld * h%rg(q) + (f - 1) * (h%rg(q + 1) - h%rg(q)) + (x - h%rg(q))
  end function pk

  !> The inverse of pk: packed column c -> field f, scalar x.
  subroutine unpk(h, c, nfld, f, x)
    type(hpatch_t), intent(in) :: h
    PetscInt, intent(in)       :: c
    integer, intent(in)        :: nfld
    integer, intent(out)       :: f
    PetscInt, intent(out)      :: x
    integer :: lo, hi, mid
    PetscInt :: off, w
    lo = 0; hi = h%np - 1
    do while (lo < hi)                               ! largest q with nfld rg(q) <= c
      mid = (lo + hi + 1) / 2
      if (nfld * h%rg(mid) <= c) then
        lo = mid
      else
        hi = mid - 1
      endif
    enddo
    off = c - nfld * h%rg(lo)
    w   = h%rg(lo + 1) - h%rg(lo)
    f   = int(off / w) + 1
    x   = h%rg(lo) + mod(off, w)
  end subroutine unpk

  !> Local id of scalar x: (U position - 1) ns + slot, -1 outside U.
  integer function lid(h, x)
    type(hpatch_t), intent(in) :: h
    PetscInt, intent(in)       :: x
    integer :: ps
    ps = pbsearch(h%uix, x / h%ns)
    lid = -1
    if (ps > 0) lid = (ps - 1) * h%ns + int(mod(x, int(h%ns, kind(x))))
  end function lid

  !> Position of key in the ascending a, 0 if absent.
  pure integer function pbsearch(a, key)
    PetscInt, intent(in) :: a(:), key
    integer :: lo, hi, mid
    pbsearch = 0
    lo = 1; hi = size(a)
    do while (lo <= hi)
      mid = (lo + hi) / 2
      if (a(mid) == key) then
        pbsearch = mid
        return
      else if (a(mid) < key) then
        lo = mid + 1
      else
        hi = mid - 1
      endif
    enddo
  end function pbsearch

  !> Sort a ascending, drop duplicates: a(1:nu) (heapsort).
  subroutine sort_uniq(a, nu)
    PetscInt, intent(inout) :: a(:)
    integer, intent(out)    :: nu
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
    subroutine sift(i0, hi)
      integer, intent(in) :: i0, hi
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

  !> Insertion sort of a(1:n), duplicates dropped: n on return = unique count.
  subroutine isort_uniq(a, n)
    integer, intent(inout) :: a(:)
    integer, intent(inout) :: n
    integer :: i, j, t, nu
    do i = 2, n
      t = a(i); j = i - 1
      do while (j >= 1)
        if (a(j) <= t) exit
        a(j + 1) = a(j); j = j - 1
      enddo
      a(j + 1) = t
    enddo
    nu = min(n, 1)
    do i = 2, n
      if (a(i) /= a(nu)) then
        nu = nu + 1;  a(nu) = a(i)
      endif
    enddo
    n = nu
  end subroutine isort_uniq

#endif
end module mod_petsc_pc_sf_hpatch
