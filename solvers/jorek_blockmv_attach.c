/* Threaded, block-aware MatMult / MatMultAdd for the physics PC's (MPI)AIJ operators.
 *
 * jorek_blockmv_attach replaces MATOP_MULT and MATOP_MULT_ADD of one MATSEQAIJ or
 * MATMPIAIJ matrix by an OpenMP kernel; the type, the CSR arrays (value-map refills,
 * MatPtAP reuse) and all other operations stay PETSc's. Runs of up to 4 bs consecutive
 * rows with one column pattern (packed GMG layout: a node's four C1 DOFs x bs slots)
 * read their column indices once and are taken four rows at a time. MPIAIJ: own ghost
 * scatter from the off-diagonal column map, overlapped with the diagonal part as in
 * MatMult_MPIAIJ. Requires PETSc >= 3.26.
 */
#if defined(USE_PETSC)
#include <string.h>
#include <petscmat.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define BMV_KEY    "jorek_blockmv"
#define BMV_REJ    "jorek_blockmv_rejected" /* the check failed once: keep PETSc's kernel */
#define BMV_SERIAL 20000 /* below this many local nonzeros: one thread */

typedef struct {
  PetscInt   n;      /* local rows */
  PetscInt64 nnz;
  PetscInt   lmax;   /* longest row */
  PetscInt   nb;     /* block rows */
  PetscInt  *bst;    /* first row of block row b, bst[nb] = n */
  PetscInt   nt;     /* threads of the partition */
  PetscInt  *tb;     /* thread t does block rows tb[t] .. tb[t+1]-1 */
} bmv_part;

typedef struct {
  PetscInt         bs;
  PetscInt         bs_mat;   /* BAIJ hardware block size */
  PetscInt         nthr;     /* thread count the partitions are for */
  PetscObjectId    id;       /* blocks and scatter: for M's id and nonzero state, */
  PetscObjectState nzstate;  /* the same on all ranks (their rebuild is collective) */
  PetscBool        mpi;
  PetscBool        baij;
  bmv_part         pd, po;   /* diagonal part, off-diagonal part */
  Vec              lvec;     /* ghost values (MPI only) */
  VecScatter       sct;
} bmv_ctx;

static PetscLogEvent bmv_event = -1;

/* CSR arrays of a SeqAIJ matrix, in host memory. */
static PetscErrorCode csr(Mat M, const PetscInt **i, const PetscInt **j, PetscScalar **a)
{
  PetscMemType mt;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetCSRAndMemType(M, i, j, a, &mt));
  PetscCheck(PetscMemTypeHost(mt), PETSC_COMM_SELF, PETSC_ERR_SUP, "blockmv: needs host CSR arrays");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode part_free(bmv_part *p)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(p->bst));
  PetscCall(PetscFree(p->tb));
  p->n = p->lmax = p->nb = p->nt = 0;
  p->nnz = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt max_threads(void)
{
#if defined(_OPENMP)
  return (PetscInt)omp_get_max_threads();
#else
  return 1;
#endif
}

/* Block rows of M (SeqAIJ). */
static PetscErrorCode part_blocks(Mat M, PetscInt bs, bmv_part *p)
{
  const PetscInt *ia, *ja;
  PetscInt        n, i, r;

  PetscFunctionBegin;
  PetscCall(part_free(p));
  PetscCall(MatGetLocalSize(M, &n, NULL));
  PetscCall(csr(M, &ia, &ja, NULL));
  PetscCall(PetscMalloc1(n + 1, &p->bst));
  p->n = n;
  p->nb = 0;
  for (i = 0; i < n;) {
    PetscInt len = ia[i + 1] - ia[i], wmax = PetscMin(4 * bs, n - i);
    for (r = 1; r < wmax; r++) {
      if (ia[i + r + 1] - ia[i + r] != len) break;
      if (memcmp(ja + ia[i], ja + ia[i + r], (size_t)len * sizeof(PetscInt))) break;
    }
    p->lmax = PetscMax(p->lmax, len);
    p->bst[p->nb++] = i;
    i += r;
  }
  p->bst[p->nb] = n;
  p->nnz = (PetscInt64)ia[n];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Block rows of M (SeqBAIJ). */
static PetscErrorCode part_blocks_baij(Mat M, PetscInt bs_mat, bmv_part *p)
{
  const PetscInt *ia = NULL, *ja = NULL;
  PetscInt        mbs = 0, b;
  PetscBool       done = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(part_free(p));
  PetscCall(MatGetRowIJ(M, 0, PETSC_FALSE, PETSC_TRUE, &mbs, &ia, &ja, &done));
  PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_SUP, "blockmv: MatGetRowIJ failed on SeqBAIJ");
  p->n   = mbs * bs_mat;
  p->nb  = mbs;
  p->nnz = (PetscInt64)(mbs > 0 ? ia[mbs] : 0) * (PetscInt64)bs_mat * (PetscInt64)bs_mat;
  for (b = 0; b < mbs; b++) {
    PetscInt len = (ia[b + 1] - ia[b]) * bs_mat;
    p->lmax = PetscMax(p->lmax, len);
  }
  PetscCall(MatRestoreRowIJ(M, 0, PETSC_FALSE, PETSC_TRUE, &mbs, &ia, &ja, &done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void part_threads_baij(PetscInt nthr, bmv_part *p)
{
  p->nt = (p->nnz < BMV_SERIAL || p->nb <= 1) ? 1 : PetscMin(nthr, p->nb);
}

/* y = A x (add = 0) or y += A x (add = 1) over the block rows of SeqBAIJ M. */
static PetscErrorCode part_apply_baij(Mat M, PetscInt bs, const bmv_part *p, const PetscScalar *x, PetscScalar *y, int add)
{
  const PetscInt  *ia = NULL, *ja = NULL;
  PetscScalar     *aw = NULL;
  PetscInt         mbs = 0;
  PetscBool        done = PETSC_FALSE;
  const PetscInt64 bs2 = (PetscInt64)bs * (PetscInt64)bs;

  PetscFunctionBegin;
  if (p->nb == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (add && p->nnz == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatGetRowIJ(M, 0, PETSC_FALSE, PETSC_TRUE, &mbs, &ia, &ja, &done));
  PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_SUP, "blockmv: MatGetRowIJ failed on SeqBAIJ");
  PetscCall(MatSeqBAIJGetArray(M, &aw));
#if defined(_OPENMP)
  #pragma omp parallel for schedule(dynamic, 1) num_threads(p->nt) if (p->nt > 1)
#endif
  for (PetscInt b = 0; b < mbs; b++) {
    const PetscInt k0 = ia[b], k1 = ia[b + 1];
    PetscScalar   *yb = y + (PetscInt64)b * bs;
    if (!add) {
      for (PetscInt rw = 0; rw < bs; rw++) yb[rw] = 0.0;
    }
    if (k0 == k1) continue;
    for (PetscInt k = k0; k < k1; k++) {
      const PetscScalar *xb  = x + (PetscInt64)ja[k] * bs;
      const PetscScalar *blk = aw + (PetscInt64)k * bs2;
      PetscInt           cw  = 0;
      for (; cw + 4 <= bs; cw += 4) {
        const PetscScalar  x0 = xb[cw], x1 = xb[cw + 1], x2 = xb[cw + 2], x3 = xb[cw + 3];
        const PetscScalar *c0 = blk + (PetscInt64)cw * bs;
        const PetscScalar *c1 = c0 + bs, *c2 = c1 + bs, *c3 = c2 + bs;
#if defined(_OPENMP)
        #pragma omp simd
#endif
        for (PetscInt rw = 0; rw < bs; rw++) {
          yb[rw] += c0[rw] * x0 + c1[rw] * x1 + c2[rw] * x2 + c3[rw] * x3;
        }
      }
      for (; cw < bs; cw++) {
        const PetscScalar  xv  = xb[cw];
        const PetscScalar *col = blk + (PetscInt64)cw * bs;
#if defined(_OPENMP)
        #pragma omp simd
#endif
        for (PetscInt rw = 0; rw < bs; rw++) yb[rw] += col[rw] * xv;
      }
    }
  }
  PetscCall(MatSeqBAIJRestoreArray(M, &aw));
  PetscCall(MatRestoreRowIJ(M, 0, PETSC_FALSE, PETSC_TRUE, &mbs, &ia, &ja, &done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* nnz-balanced partition of the block rows for nthr threads (one below BMV_SERIAL nonzeros). */
static PetscErrorCode part_threads(Mat M, PetscInt nthr, bmv_part *p)
{
  const PetscInt *ia;
  PetscInt        b, t, nt = (p->nnz < BMV_SERIAL) ? 1 : nthr;

  PetscFunctionBegin;
  PetscCall(PetscFree(p->tb));
  PetscCall(csr(M, &ia, NULL, NULL));
  p->nt = nt;
  PetscCall(PetscMalloc1(nt + 1, &p->tb));
  p->tb[0] = 0;
  for (t = 1, b = 0; t < nt; t++) {
    PetscInt target = (PetscInt)(((PetscInt64)p->nnz * t) / nt);
    while (b < p->nb && ia[p->bst[b]] < target) b++;
    p->tb[t] = b;
  }
  p->tb[nt] = p->nb;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A x (add = 0) or y += A x (add = 1) over the rows of p. */
static PetscErrorCode part_apply(Mat M, const bmv_part *p, const PetscScalar *x, PetscScalar *y, int add)
{
  const PetscInt *ia, *ja;
  PetscScalar    *aw;

  PetscFunctionBegin;
  PetscCall(csr(M, &ia, &ja, &aw));
#if defined(_OPENMP)
  #pragma omp parallel num_threads(p->nt) if (p->nt > 1)
#endif
  {
    /* The runtime may start fewer than p->nt threads (nesting, OMP_DYNAMIC,
       a thread limit), so each thread takes every nth partition. */
#if defined(_OPENMP)
    const PetscInt t0 = omp_get_thread_num(), dt = omp_get_num_threads();
#else
    const PetscInt t0 = 0, dt = 1;
#endif
    const PetscScalar *a = aw;
    for (PetscInt t = t0; t < p->nt; t += dt)
    for (PetscInt b = p->tb[t]; b < p->tb[t + 1]; b++) {
      const PetscInt i0 = p->bst[b], w = p->bst[b + 1] - i0;
      const PetscInt k0 = ia[i0], len = ia[i0 + 1] - k0;
      const PetscInt *c = ja + k0;
      PetscInt r = 0;
      for (; r + 4 <= w; r += 4) {
        const PetscInt     i = i0 + r;
        const PetscScalar *a0 = a + ia[i], *a1 = a + ia[i + 1], *a2 = a + ia[i + 2], *a3 = a + ia[i + 3];
        PetscScalar        s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#if defined(_OPENMP)
        #pragma omp simd reduction(+:s0, s1, s2, s3)
#endif
        for (PetscInt k = 0; k < len; k++) {
          const PetscScalar xv = x[c[k]];
          s0 += a0[k] * xv; s1 += a1[k] * xv; s2 += a2[k] * xv; s3 += a3[k] * xv;
        }
        if (add) { y[i] += s0; y[i + 1] += s1; y[i + 2] += s2; y[i + 3] += s3; }
        else     { y[i]  = s0; y[i + 1]  = s1; y[i + 2]  = s2; y[i + 3]  = s3; }
      }
      if (w - r == 3) {
        const PetscInt     i = i0 + r;
        const PetscScalar *a0 = a + ia[i], *a1 = a + ia[i + 1], *a2 = a + ia[i + 2];
        PetscScalar        s0 = 0, s1 = 0, s2 = 0;
#if defined(_OPENMP)
        #pragma omp simd reduction(+:s0, s1, s2)
#endif
        for (PetscInt k = 0; k < len; k++) {
          const PetscScalar xv = x[c[k]];
          s0 += a0[k] * xv; s1 += a1[k] * xv; s2 += a2[k] * xv;
        }
        if (add) { y[i] += s0; y[i + 1] += s1; y[i + 2] += s2; }
        else     { y[i]  = s0; y[i + 1]  = s1; y[i + 2]  = s2; }
        r = w;
      }
      for (; r < w; r++) {
        const PetscScalar *ar = a + ia[i0 + r];
        PetscScalar        s  = 0;
#if defined(_OPENMP)
        #pragma omp simd reduction(+:s)
#endif
        for (PetscInt k = 0; k < len; k++) s += ar[k] * x[c[k]];
        if (add) y[i0 + r] += s; else y[i0 + r] = s;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ctx_destroy(void *pctx)
{
  bmv_ctx *c = *(bmv_ctx **)pctx;

  PetscFunctionBegin;
  PetscCall(part_free(&c->pd));
  PetscCall(part_free(&c->po));
  PetscCall(VecDestroy(&c->lvec));
  PetscCall(VecScatterDestroy(&c->sct));
  PetscCall(PetscFree(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Blocks and scatter for a new pattern (collective: keyed on M's id and
   nonzero state), thread partitions for a new thread count (local). */
static PetscErrorCode ctx_refresh(Mat M, bmv_ctx *c)
{
  PetscObjectState st;
  PetscObjectId    id;
  Mat              Ad = M, Ao = NULL;
  const PetscInt  *cmap = NULL;
  const PetscInt   nthr = max_threads();

  PetscFunctionBegin;
  PetscCall(MatGetNonzeroState(M, &st));
  PetscCall(PetscObjectGetId((PetscObject)M, &id));
  if (c->baij) {
    PetscCall(MatGetBlockSize(M, &c->bs_mat));
    if (c->mpi) PetscCall(MatMPIBAIJGetSeqBAIJ(M, &Ad, &Ao, &cmap));
  } else {
    if (c->mpi) PetscCall(MatMPIAIJGetSeqAIJ(M, &Ad, &Ao, &cmap));
  }
  if (st != c->nzstate || id != c->id) {
    if (c->baij) PetscCall(part_blocks_baij(Ad, c->bs_mat, &c->pd));
    else         PetscCall(part_blocks(Ad, c->bs, &c->pd));
    PetscCall(VecDestroy(&c->lvec));
    PetscCall(VecScatterDestroy(&c->sct));
    if (c->mpi) {
      PetscInt nghost;
      IS       isg;
      Vec      xg;
      if (c->baij) PetscCall(part_blocks_baij(Ao, c->bs_mat, &c->po));
      else         PetscCall(part_blocks(Ao, c->bs, &c->po));
      PetscCall(MatGetSize(Ao, NULL, &nghost));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nghost, &c->lvec));
      if (c->baij) PetscCall(ISCreateBlock(PETSC_COMM_SELF, c->bs_mat, nghost / c->bs_mat, cmap, PETSC_COPY_VALUES, &isg));
      else         PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nghost, cmap, PETSC_USE_POINTER, &isg));
      PetscCall(MatCreateVecs(M, &xg, NULL));
      PetscCall(VecScatterCreate(xg, isg, c->lvec, NULL, &c->sct));
      PetscCall(VecDestroy(&xg));
      PetscCall(ISDestroy(&isg));
    }
    c->nzstate = st;
    c->id      = id;
    c->nthr    = -1;
  }
  if (nthr != c->nthr) {
    if (c->baij) {
      part_threads_baij(nthr, &c->pd);
      if (c->mpi) part_threads_baij(nthr, &c->po);
    } else {
      PetscCall(part_threads(Ad, nthr, &c->pd));
      if (c->mpi) PetscCall(part_threads(Ao, nthr, &c->po));
    }
    c->nthr = nthr;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* M's plan, created by the first attach (a MatDuplicate has PETSc's ops again). */
static PetscErrorCode get_ctx(Mat M, PetscInt bs, bmv_ctx **c)
{
  PetscContainer cont;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)M, BMV_KEY, (PetscObject *)&cont));
  if (cont) {
    PetscCall(PetscContainerGetPointer(cont, (void **)c));
  } else {
    PetscBool mpiaij, seqbaij, mpibaij;
    PetscCall(PetscNew(c));
    (*c)->bs      = bs;
    (*c)->bs_mat  = 1;
    (*c)->nzstate = -1;
    (*c)->id      = -1;
    (*c)->nthr    = -1;
    PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPIAIJ, &mpiaij));
    PetscCall(PetscObjectTypeCompare((PetscObject)M, MATSEQBAIJ, &seqbaij));
    PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPIBAIJ, &mpibaij));
    (*c)->baij = (PetscBool)(seqbaij || mpibaij);
    (*c)->mpi  = (PetscBool)(mpiaij || mpibaij);
    PetscCall(PetscObjectContainerCompose((PetscObject)M, BMV_KEY, *c, ctx_destroy));
  }
  if ((*c)->bs != bs) {
    (*c)->bs      = bs;
    (*c)->nzstate = -1;
  }
  PetscCall(ctx_refresh(M, *c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A x (+ y if add) */
static PetscErrorCode bmv_apply(Mat M, Vec x, Vec y, Vec z, int add)
{
  PetscContainer     cont;
  bmv_ctx           *c;
  Mat                Ad = M, Ao = NULL;
  const PetscScalar *xa, *ga;
  PetscScalar       *za;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)M, BMV_KEY, (PetscObject *)&cont));
  PetscCheck(cont, PETSC_COMM_SELF, PETSC_ERR_PLIB, "blockmv: kernel installed without its plan");
  PetscCall(PetscContainerGetPointer(cont, (void **)&c));
  PetscCall(ctx_refresh(M, c));
  PetscCall(PetscLogEventBegin(bmv_event, M, x, z, 0));
  if (add && y != z) PetscCall(VecCopy(y, z));
  if (c->mpi) {
    if (c->baij) PetscCall(MatMPIBAIJGetSeqBAIJ(M, &Ad, &Ao, NULL));
    else         PetscCall(MatMPIAIJGetSeqAIJ(M, &Ad, &Ao, NULL));
    PetscCall(VecScatterBegin(c->sct, x, c->lvec, INSERT_VALUES, SCATTER_FORWARD));
  }
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(z, &za));
  if (c->baij) PetscCall(part_apply_baij(Ad, c->bs_mat, &c->pd, xa, za, add));
  else         PetscCall(part_apply(Ad, &c->pd, xa, za, add));
  PetscCall(VecRestoreArrayRead(x, &xa));
  if (c->mpi) {
    PetscCall(VecScatterEnd(c->sct, x, c->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArrayRead(c->lvec, &ga));
    if (c->baij) PetscCall(part_apply_baij(Ao, c->bs_mat, &c->po, ga, za, 1));
    else         PetscCall(part_apply(Ao, &c->po, ga, za, 1));
    PetscCall(VecRestoreArrayRead(c->lvec, &ga));
  }
  PetscCall(VecRestoreArray(z, &za));
  PetscCall(PetscLogFlops(2.0 * (double)(c->pd.nnz + c->po.nnz)));
  PetscCall(PetscLogEventEnd(bmv_event, M, x, z, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode bmv_mult(Mat M, Vec x, Vec y) { return bmv_apply(M, x, NULL, y, 0); }
static PetscErrorCode bmv_multadd(Mat M, Vec x, Vec y, Vec z) { return bmv_apply(M, x, y, z, 1); }

/* Attach the kernel to M (MATSEQAIJ/MATMPIAIJ or MATSEQBAIJ/MATMPIBAIJ; *ok = 0 and M keeps
 * PETSc's kernel for any other type, or if this one fails the check against
 * it). Idempotent: a second call only refreshes the plan. Collective on M. */
PetscErrorCode jorek_blockmv_attach(Mat M, int bs, int *ok)
{
  static int  rejected = 1;
  PetscBool   seq, mpi, seqb, mpib;
  PetscObject rej;
  bmv_ctx    *c;
  void (*cur)(void);

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATSEQAIJ, &seq));
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPIAIJ, &mpi));
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATSEQBAIJ, &seqb));
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPIBAIJ, &mpib));
  PetscCall(PetscObjectQuery((PetscObject)M, BMV_REJ, &rej));
  *ok = (seq || mpi || seqb || mpib) && !rej;
  if (!*ok) PetscFunctionReturn(PETSC_SUCCESS);
  if (bmv_event < 0) {
    PetscClassId cid;
    PetscCall(PetscClassIdRegister("JOREK blockmv", &cid));
    PetscCall(PetscLogEventRegister("PC_BlockMV", cid, &bmv_event));
  }
  PetscCall(get_ctx(M, bs, &c));
  PetscCall(MatGetOperation(M, MATOP_MULT, &cur));
  if (cur != (void (*)(void))bmv_mult) {
    /* first attach: against the installed kernel on a random vector, to
       rounding (100 eps sqrt(longest row) |A|_inf |x|_inf, max norm) */
    MPI_Comm    comm = PetscObjectComm((PetscObject)M);
    Vec         x, y0, y1;
    PetscRandom rnd;
    PetscReal   e, na, nx, tol;
    PetscInt    lmax = c->pd.lmax + c->po.lmax;
    PetscCall(MatCreateVecs(M, &x, &y0));
    PetscCall(VecDuplicate(y0, &y1));
    PetscCall(PetscRandomCreate(comm, &rnd));
    PetscCall(VecSetRandom(x, rnd));
    PetscCall(PetscRandomDestroy(&rnd));
    PetscCall(MatMult(M, x, y0));
    PetscCall(MatSetOperation(M, MATOP_MULT, (void (*)(void))bmv_mult));
    PetscCall(MatMult(M, x, y1));
    PetscCall(VecAXPY(y1, -1.0, y0));
    PetscCall(VecNorm(y1, NORM_INFINITY, &e));
    PetscCall(VecNorm(x, NORM_INFINITY, &nx));
    PetscCall(MatNorm(M, NORM_INFINITY, &na));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &lmax, 1, MPIU_INT, MPI_MAX, comm));
    tol = 100 * PETSC_MACHINE_EPSILON * PetscSqrtReal((PetscReal)lmax) * na * nx;
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y0));
    PetscCall(VecDestroy(&y1));
    if (!(e <= tol)) {
      PetscCall(MatSetOperation(M, MATOP_MULT, cur));
      PetscCall(PetscObjectCompose((PetscObject)M, BMV_KEY, NULL));
      PetscCall(PetscObjectContainerCompose((PetscObject)M, BMV_REJ, &rejected, NULL));
      PetscCall(PetscPrintf(comm, "[Physics PC]   blockmv: kernel differs from MatMult by %g (tolerance %g), MatMult kept\n", (double)e, (double)tol));
      *ok = 0;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCall(MatSetOperation(M, MATOP_MULT_ADD, (void (*)(void))bmv_multadd));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
