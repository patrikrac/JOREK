/* Threaded, block-aware matvec for the physics PC's (MPI)AIJ operators.
 *
 * PETSc's AIJ MatMult runs on one thread per rank, and in JOREK's hybrid runs
 * (8-16 OpenMP threads per rank) the GMG level matvecs are most of a V-cycle.
 * jorek_blockmv_attach replaces MATOP_MULT and MATOP_MULT_ADD of one AIJ
 * matrix with an OpenMP kernel and leaves everything else alone: the type,
 * the CSR arrays (so value-map refills and MatPtAP(MAT_REUSE_MATRIX) keep
 * working), MatCreateSubMatrices, MatGetDiagonal, MatMultTranspose.
 *
 * Structure used: the packed GMG layout keeps the n_tor harmonics of one
 * (field, scalar DOF) in consecutive rows with one column pattern, so each
 * block of bs such rows reads its column indices once. Rows that do not form
 * such a block (boundary rows) are done one by one.
 *
 * For an MPIAIJ matrix the kernel does the ghost update itself, with its own
 * scatter built from the off-diagonal part's column map, and overlaps it with
 * the diagonal part exactly as MatMult_MPIAIJ does. The plan (blocks, thread
 * partition, scatter) is rebuilt when the nonzero pattern or the thread count
 * changes.
 */
#if defined(USE_PETSC)
#include <string.h>
#include <petscmat.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define BMV_KEY    "jorek_blockmv"
#define BMV_SERIAL 20000 /* below this many local nonzeros: one thread */

typedef struct {
  PetscInt  n;      /* local rows */
  PetscInt  nnz;
  PetscInt  nb;     /* block rows */
  PetscInt *bst;    /* first row of block row b, bst[nb] = n */
  PetscInt  nt;     /* threads the partition is for */
  PetscInt *tb;     /* thread t does block rows tb[t] .. tb[t+1]-1 */
} bmv_part;

typedef struct {
  PetscInt         bs;
  PetscObjectState nzstate;
  PetscBool        mpi;
  bmv_part         pd, po;  /* diagonal part, off-diagonal part */
  Vec              lvec;    /* ghost values (MPI only) */
  VecScatter       sct;
  PetscLogEvent    ev;
} bmv_ctx;

static PetscLogEvent bmv_event = -1;

static void part_free(bmv_part *p)
{
  PetscFree(p->bst);
  PetscFree(p->tb);
  p->n = p->nnz = p->nb = p->nt = 0;
}

static PetscInt max_threads(void)
{
#if defined(_OPENMP)
  return (PetscInt)omp_get_max_threads();
#else
  return 1;
#endif
}

/* Block rows of M (SeqAIJ) and an nnz-balanced thread partition of them. */
static PetscErrorCode part_build(Mat M, PetscInt bs, bmv_part *p)
{
  const PetscInt *ia, *ja;
  PetscInt        n, i, r, b, t, nnz, nt;
  PetscBool       done;

  PetscFunctionBegin;
  part_free(p);
  PetscCall(MatGetRowIJ(M, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_SUP, "blockmv: no row IJ access");
  PetscCall(PetscMalloc1(n + 1, &p->bst));
  p->n = n;
  p->nb = 0;
  for (i = 0; i < n;) {
    PetscInt len = ia[i + 1] - ia[i], w = 1;
    if (bs > 1 && i + bs <= n) {
      for (r = 1; r < bs; r++) {
        if (ia[i + r + 1] - ia[i + r] != len) break;
        if (memcmp(ja + ia[i], ja + ia[i + r], (size_t)len * sizeof(PetscInt))) break;
      }
      if (r == bs) w = bs;
    }
    p->bst[p->nb++] = i;
    i += w;
  }
  p->bst[p->nb] = n;
  nnz = ia[n];
  p->nnz = nnz;
  nt  = (nnz < BMV_SERIAL) ? 1 : max_threads();
  p->nt = nt;
  PetscCall(PetscMalloc1(nt + 1, &p->tb));
  p->tb[0] = 0;
  for (t = 1, b = 0; t < nt; t++) {
    PetscInt target = (PetscInt)(((PetscInt64)nnz * t) / nt);
    while (b < p->nb && ia[p->bst[b]] < target) b++;
    p->tb[t] = b;
  }
  p->tb[nt] = p->nb;
  PetscCall(MatRestoreRowIJ(M, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A x (add = 0) or y += A x (add = 1) over the rows of p. */
static PetscErrorCode part_apply(Mat M, const bmv_part *p, const PetscScalar *x, PetscScalar *y, int add)
{
  const PetscInt    *ia, *ja;
  const PetscScalar *a;
  PetscInt           n;
  PetscBool          done;

  PetscFunctionBegin;
  PetscCall(MatGetRowIJ(M, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscCall(MatSeqAIJGetArrayRead(M, &a));
#if defined(_OPENMP)
  #pragma omp parallel num_threads(p->nt) if (p->nt > 1)
#endif
  {
#if defined(_OPENMP)
    const PetscInt t = omp_get_thread_num();
#else
    const PetscInt t = 0;
#endif
    for (PetscInt b = p->tb[t]; b < p->tb[t + 1]; b++) {
      const PetscInt i0 = p->bst[b], w = p->bst[b + 1] - i0;
      const PetscInt k0 = ia[i0], len = ia[i0 + 1] - k0;
      const PetscInt *c = ja + k0;
      if (w == 3) {
        const PetscScalar *a0 = a + k0, *a1 = a + ia[i0 + 1], *a2 = a + ia[i0 + 2];
        PetscScalar s0 = 0, s1 = 0, s2 = 0;
#if defined(_OPENMP)
        #pragma omp simd reduction(+:s0, s1, s2)
#endif
        for (PetscInt k = 0; k < len; k++) {
          const PetscScalar xv = x[c[k]];
          s0 += a0[k] * xv; s1 += a1[k] * xv; s2 += a2[k] * xv;
        }
        if (add) { y[i0] += s0; y[i0 + 1] += s1; y[i0 + 2] += s2; }
        else     { y[i0]  = s0; y[i0 + 1]  = s1; y[i0 + 2]  = s2; }
      } else {
        for (PetscInt r = 0; r < w; r++) {
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
  }
  PetscCall(MatSeqAIJRestoreArrayRead(M, &a));
  PetscCall(MatRestoreRowIJ(M, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ctx_destroy(void *pctx)
{
  bmv_ctx *c = *(bmv_ctx **)pctx;

  PetscFunctionBegin;
  part_free(&c->pd);
  part_free(&c->po);
  PetscCall(VecDestroy(&c->lvec));
  PetscCall(VecScatterDestroy(&c->sct));
  PetscCall(PetscFree(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* (Re)build the plan if the pattern or the thread count changed. */
static PetscErrorCode ctx_refresh(Mat M, bmv_ctx *c)
{
  PetscObjectState st;
  Mat              Ad = M, Ao = NULL;
  const PetscInt  *cmap = NULL;

  PetscFunctionBegin;
  PetscCall(MatGetNonzeroState(M, &st));
  if (st == c->nzstate && c->pd.n >= 0 && (c->pd.nt == 1 || c->pd.nt == max_threads())) PetscFunctionReturn(PETSC_SUCCESS);
  if (c->mpi) PetscCall(MatMPIAIJGetSeqAIJ(M, &Ad, &Ao, &cmap));
  PetscCall(part_build(Ad, c->bs, &c->pd));
  PetscCall(VecDestroy(&c->lvec));
  PetscCall(VecScatterDestroy(&c->sct));
  if (c->mpi) {
    PetscInt nghost;
    IS       isg;
    Vec      xg;
    PetscCall(part_build(Ao, c->bs, &c->po));
    PetscCall(MatGetSize(Ao, NULL, &nghost));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nghost, &c->lvec));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nghost, cmap, PETSC_USE_POINTER, &isg));
    PetscCall(MatCreateVecs(M, &xg, NULL));
    PetscCall(VecScatterCreate(xg, isg, c->lvec, NULL, &c->sct));
    PetscCall(VecDestroy(&xg));
    PetscCall(ISDestroy(&isg));
  }
  c->nzstate = st;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The plan of M, created with bs = 1 if M has none: MatDuplicate copies the
 * ops table but not the composed plan. */
static PetscErrorCode get_ctx(Mat M, PetscInt bs, bmv_ctx **c)
{
  PetscContainer cont;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)M, BMV_KEY, (PetscObject *)&cont));
  if (cont) {
    PetscCall(PetscContainerGetPointer(cont, (void **)c));
  } else {
    PetscCall(PetscNew(c));
    (*c)->bs      = bs;
    (*c)->nzstate = -1;
    (*c)->pd.n    = -1;
    PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPIAIJ, &(*c)->mpi));
    PetscCall(PetscObjectContainerCompose((PetscObject)M, BMV_KEY, *c, (PetscCtxDestroyFn *)ctx_destroy));
  }
  PetscCall(ctx_refresh(M, *c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A x (+ y if add) */
static PetscErrorCode bmv_apply(Mat M, Vec x, Vec y, Vec z, int add)
{
  bmv_ctx           *c;
  Mat                Ad = M, Ao = NULL;
  const PetscScalar *xa, *ga;
  PetscScalar       *za;

  PetscFunctionBegin;
  PetscCall(get_ctx(M, 1, &c));
  PetscCall(PetscLogEventBegin(bmv_event, M, x, z, 0));
  if (add && y != z) PetscCall(VecCopy(y, z));
  if (c->mpi) {
    PetscCall(MatMPIAIJGetSeqAIJ(M, &Ad, &Ao, NULL));
    PetscCall(VecScatterBegin(c->sct, x, c->lvec, INSERT_VALUES, SCATTER_FORWARD));
  }
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(z, &za));
  PetscCall(part_apply(Ad, &c->pd, xa, za, add));
  PetscCall(VecRestoreArrayRead(x, &xa));
  if (c->mpi) {
    PetscCall(VecScatterEnd(c->sct, x, c->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArrayRead(c->lvec, &ga));
    PetscCall(part_apply(Ao, &c->po, ga, za, 1));
    PetscCall(VecRestoreArrayRead(c->lvec, &ga));
  }
  PetscCall(VecRestoreArray(z, &za));
  PetscCall(PetscLogFlops(2.0 * (double)(c->pd.nnz + (c->mpi ? c->po.nnz : 0))));
  PetscCall(PetscLogEventEnd(bmv_event, M, x, z, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode bmv_mult(Mat M, Vec x, Vec y) { return bmv_apply(M, x, NULL, y, 0); }
static PetscErrorCode bmv_multadd(Mat M, Vec x, Vec y, Vec z) { return bmv_apply(M, x, y, z, 1); }

/* Attach the kernel to M (MATSEQAIJ or MATMPIAIJ; *ok = 0 and M
 * untouched for any other type). Idempotent: a second call only refreshes
 * the plan. Collective on M (the MPI plan builds a scatter). */
PetscErrorCode jorek_blockmv_attach(Mat M, int bs, int *ok)
{
  PetscBool seq, mpi;
  bmv_ctx  *c;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATSEQAIJ, &seq));
  PetscCall(PetscObjectTypeCompare((PetscObject)M, MATMPIAIJ, &mpi));
  *ok = (seq || mpi);
  if (!*ok) PetscFunctionReturn(PETSC_SUCCESS);
  if (bmv_event < 0) {
    PetscClassId cid;
    PetscCall(PetscClassIdRegister("JOREK blockmv", &cid));
    PetscCall(PetscLogEventRegister("PC_BlockMV", cid, &bmv_event));
  }
  PetscCall(get_ctx(M, bs, &c));
  if (c->bs != bs) {
    c->bs      = bs;
    c->nzstate = -1;
    PetscCall(ctx_refresh(M, c));
  }
  {
    void (*cur)(void);
    PetscCall(MatGetOperation(M, MATOP_MULT, &cur));
    if (cur != (void (*)(void))bmv_mult) {
      /* first attach: gate against PETSc's own kernel on a random vector */
      Vec         x, y0, y1;
      PetscRandom rnd;
      PetscReal   e, nr;
      PetscCall(MatCreateVecs(M, &x, &y0));
      PetscCall(VecDuplicate(y0, &y1));
      PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)M), &rnd));
      PetscCall(VecSetRandom(x, rnd));
      PetscCall(PetscRandomDestroy(&rnd));
      PetscCall(MatMult(M, x, y0));
      PetscCall(MatSetOperation(M, MATOP_MULT, (void (*)(void))bmv_mult));
      PetscCall(MatMult(M, x, y1));
      PetscCall(VecNorm(y0, NORM_2, &nr));
      PetscCall(VecAXPY(y1, -1.0, y0));
      PetscCall(VecNorm(y1, NORM_2, &e));
      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&y0));
      PetscCall(VecDestroy(&y1));
      PetscCheck(e <= 1e-13 * nr, PetscObjectComm((PetscObject)M), PETSC_ERR_PLIB,
                 "blockmv: kernel disagrees with MatMult, rel. error %g", (double)(nr > 0 ? e / nr : e));
    }
  }
  PetscCall(MatSetOperation(M, MATOP_MULT, (void (*)(void))bmv_mult));
  PetscCall(MatSetOperation(M, MATOP_MULT_ADD, (void (*)(void))bmv_multadd));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
