/* Galerkin coarse operator C = P^T A P for the GMG's packed harmonic layout.
 *
 * The packed layout keeps the bs = n_tor slots of one (field, scalar DOF) in
 * consecutive rows, and the prolongation acts on the scalar DOFs only:
 * P = I_fields x P_s x I_slots. So C is P_s^T A P_s on the graph of the
 * bs x bs slot blocks, every coarse block a weighted sum of fine blocks.
 * PETSc's MatPtAP (also through MATMAIJ, which it converts) works on the
 * scalar entries: its symbolic phase on n_tor^2 times the block graph was
 * 125 s of the W7-X n_tor 31 setup, its numeric phase 19.5 s per rebuild.
 *
 * Here the symbolic phase works on the block graph (PETSc's PtAP of the
 * block pattern gives C's block pattern and parallel layout) and C stores
 * dense slot blocks. The numeric phase does, per coarse block row I touched
 * by this rank's fine rows (one thread per row, no races),
 *   R = sum_i P_s(i,I) A(i,:)      dense strip over the fine block columns
 *   C(I,:) = R P_s                  bs x bs block axpys
 * writing owned rows straight into C's CSR arrays; rows owned by another
 * rank (partial sums) go through MatSetValues(ADD_VALUES) and the assembly.
 *
 * jorek_galerkin_blk(A, Ps, bs, reuse, &C, &ok): A MPIAIJ in the packed
 * layout, Ps = P_s (MPIAIJ, one row per bs rows of A). reuse = 0
 * creates C, 1 refills a C this routine created (same A pattern). *ok = 0 (nothing done, same on
 * all ranks) if A, Ps or the layout do not fit. Collective.
 */
#if defined(USE_PETSC)
#include <string.h>
#include <petscmat.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define GBLK_KEY "jorek_galerkin_blk"

typedef struct {
  PetscInt  bs, nlb, ngb, ncb, cstart, Ncb; /* fine local / ghost blocks, coarse local / first / global blocks */
  PetscInt *dcb, *dcn;      /* A's diagonal part: local column -> local fine block, slot */
  PetscInt *ocb, *ocn;      /* A's off-diagonal part: column -> fine block id (nlb + ghost), slot */
  PetscInt *pp, *pj;        /* P_s rows of the fine blocks (local, then ghosts): CSR, global coarse block */
  PetscScalar *pv;
  PetscInt  nt;             /* coarse block rows touched here */
  PetscInt *tI;             /* their global ids */
  PetscInt *cp, *ci;        /* contributors: local fine blocks i ... */
  PetscScalar *cv;          /* ... with P_s(i, I) */
  PetscInt *jp, *jl;        /* fine block columns of R per touched row */
  PetscInt *Jp, *Jl;        /* coarse block columns of the partial row */
  PetscInt *Jo;             /* owned rows: offset of block Jl[k] in its row, diagonal part >= 0, off-diagonal part -1 - offset */
  PetscInt *rb;             /* remote rows: offset into rv (-1 for owned) */
  PetscScalar *rv;
  PetscInt  mxj, mxJ;       /* largest R / out strip, in blocks */
} gblk_ctx;

static PetscErrorCode ctx_destroy(void *pctx)
{
  gblk_ctx *c = *(gblk_ctx **)pctx;

  PetscFunctionBegin;
  PetscCall(PetscFree4(c->dcb, c->dcn, c->ocb, c->ocn));
  PetscCall(PetscFree3(c->pp, c->pj, c->pv));
  PetscCall(PetscFree4(c->tI, c->cp, c->ci, c->cv));
  PetscCall(PetscFree2(c->jp, c->Jp));
  PetscCall(PetscFree3(c->jl, c->Jl, c->Jo));
  PetscCall(PetscFree(c->rb));
  PetscCall(PetscFree(c->rv));
  PetscCall(PetscFree(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static int cmp_int(const void *a, const void *b)
{
  PetscInt x = *(const PetscInt *)a, y = *(const PetscInt *)b;
  return (x > y) - (x < y);
}

/* Symbolic phase: the context and C (dense slot blocks, all zero). */
static PetscErrorCode gblk_symbolic(Mat A, Mat Ps, PetscInt bs, Mat *Cout, gblk_ctx **cout)
{
  MPI_Comm        comm = PetscObjectComm((PetscObject)A);
  gblk_ctx       *c;
  Mat             Ad, Ao, Ablk, Acb, C, *Psub;
  const PetscInt *cmap, *ia, *ja, *oia, *oja;
  PetscInt        nA, rstart, nAo, n, i, k, m, t, rbs, *gblk, *mark;
  PetscBool       done;
  IS              isr, isc;

  PetscFunctionBegin;
  PetscCall(PetscNew(&c));
  c->bs = bs;
  PetscCall(MatGetLocalSize(A, &nA, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, NULL));
  c->nlb = nA / bs;
  rbs    = rstart / bs;
  PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &cmap));
  PetscCall(MatGetSize(Ao, NULL, &nAo));
  PetscCall(MatGetOwnershipRangeColumn(Ps, &c->cstart, NULL));
  PetscCall(MatGetLocalSize(Ps, NULL, &c->ncb));
  PetscCall(MatGetSize(Ps, NULL, &c->Ncb));

  /* fine block ids of A's columns; ghosts: the off-diagonal part's blocks */
  PetscCall(PetscMalloc4(nA, &c->dcb, nA, &c->dcn, nAo, &c->ocb, nAo, &c->ocn));
  for (k = 0; k < nA; k++) { c->dcb[k] = k / bs; c->dcn[k] = k % bs; }
  PetscCall(PetscMalloc1(nAo + 1, &gblk));
  c->ngb = 0;
  for (k = 0; k < nAo; k++) {  /* cmap is sorted */
    PetscInt g = cmap[k] / bs;
    if (c->ngb == 0 || gblk[c->ngb - 1] != g) gblk[c->ngb++] = g;
    c->ocb[k] = c->nlb + c->ngb - 1;
    c->ocn[k] = cmap[k] % bs;
  }

  /* P_s rows of the local and ghost fine blocks (P_s is fixed: once) */
  {
    PetscInt *rows, nr = c->nlb + c->ngb, *pos, a = 0, b = 0, q = 0;
    PetscCall(PetscMalloc2(nr, &rows, nr, &pos));
    /* merge the local range into the sorted ghosts: sorted row IS */
    while (a < c->nlb || b < c->ngb) {
      if (b >= c->ngb || (a < c->nlb && rbs + a < gblk[b])) { pos[a] = q; rows[q++] = rbs + a; a++; }
      else { pos[c->nlb + b] = q; rows[q++] = gblk[b]; b++; }
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nr, rows, PETSC_USE_POINTER, &isr));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, c->Ncb, 0, 1, &isc));
    PetscCall(MatCreateSubMatrices(Ps, 1, &isr, &isc, MAT_INITIAL_MATRIX, &Psub));
    PetscCall(ISDestroy(&isr));
    PetscCall(ISDestroy(&isc));
    {
      const PetscInt    *si, *sj;
      const PetscScalar *sv;
      PetscInt           ns, f, nz = 0;
      PetscCall(MatGetRowIJ(Psub[0], 0, PETSC_FALSE, PETSC_FALSE, &ns, &si, &sj, &done));
      PetscCall(MatSeqAIJGetArrayRead(Psub[0], &sv));
      PetscCall(PetscMalloc3(nr + 1, &c->pp, si[ns], &c->pj, si[ns], &c->pv));
      c->pp[0] = 0;
      for (f = 0; f < nr; f++) {
        for (k = si[pos[f]]; k < si[pos[f] + 1]; k++)
          if (sv[k] != 0.0) { c->pj[nz] = sj[k]; c->pv[nz] = sv[k]; nz++; }
        c->pp[f + 1] = nz;
      }
      PetscCall(MatSeqAIJRestoreArrayRead(Psub[0], &sv));
      PetscCall(MatRestoreRowIJ(Psub[0], 0, PETSC_FALSE, PETSC_FALSE, &ns, &si, &sj, &done));
    }
    PetscCall(MatDestroySubMatrices(1, &Psub));
    PetscCall(PetscFree2(rows, pos));
  }

  PetscCall(MatGetRowIJ(Ad, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscCall(MatGetRowIJ(Ao, 0, PETSC_FALSE, PETSC_FALSE, &n, &oia, &oja, &done));

  /* the block graph of A (value 1) and its PtAP: C's block pattern and layout */
  {
    PetscInt *dnz, *onz, *cols, nc, gcol;
    PetscCall(PetscMalloc4(c->nlb, &dnz, c->nlb, &onz, c->nlb + c->ngb, &mark, c->nlb + c->ngb, &cols));
    for (k = 0; k < c->nlb + c->ngb; k++) mark[k] = -1;
    for (i = 0; i < c->nlb; i++) {
      dnz[i] = onz[i] = 0;
      for (m = 0; m < bs; m++) {
        PetscInt r = i * bs + m;
        for (k = ia[r]; k < ia[r + 1]; k++) if (mark[c->dcb[ja[k]]] != i) { mark[c->dcb[ja[k]]] = i; dnz[i]++; }
        for (k = oia[r]; k < oia[r + 1]; k++) if (mark[c->ocb[oja[k]]] != i) { mark[c->ocb[oja[k]]] = i; onz[i]++; }
      }
    }
    PetscCall(MatCreate(comm, &Ablk));
    PetscCall(MatSetSizes(Ablk, c->nlb, c->nlb, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(Ablk, MATMPIAIJ));
    PetscCall(MatMPIAIJSetPreallocation(Ablk, 0, dnz, 0, onz));
    for (k = 0; k < c->nlb + c->ngb; k++) mark[k] = -1;
    for (i = 0; i < c->nlb; i++) {
      PetscScalar one = 1.0;
      nc = 0;
      for (m = 0; m < bs; m++) {
        PetscInt r = i * bs + m;
        for (k = ia[r]; k < ia[r + 1]; k++) if (mark[c->dcb[ja[k]]] != i) { mark[c->dcb[ja[k]]] = i; cols[nc++] = c->dcb[ja[k]]; }
        for (k = oia[r]; k < oia[r + 1]; k++) if (mark[c->ocb[oja[k]]] != i) { mark[c->ocb[oja[k]]] = i; cols[nc++] = c->ocb[oja[k]]; }
      }
      for (k = 0; k < nc; k++) {
        gcol = cols[k] < c->nlb ? rbs + cols[k] : gblk[cols[k] - c->nlb];
        PetscCall(MatSetValues(Ablk, 1, (PetscInt[]){rbs + i}, 1, &gcol, &one, INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(Ablk, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Ablk, MAT_FINAL_ASSEMBLY));
    PetscCall(MatPtAP(Ablk, Ps, MAT_INITIAL_MATRIX, 2.0, &Acb));
    PetscCall(MatDestroy(&Ablk));
    PetscCall(PetscFree4(dnz, onz, mark, cols));
  }

  /* C: Acb's block pattern with dense bs x bs blocks */
  {
    PetscInt *ci_, *cj_, nzb = 0, Ib, q;
    for (Ib = 0; Ib < c->ncb; Ib++) {
      PetscCall(MatGetRow(Acb, c->cstart + Ib, &n, NULL, NULL));
      nzb += n;
      PetscCall(MatRestoreRow(Acb, c->cstart + Ib, &n, NULL, NULL));
    }
    PetscCall(PetscMalloc2(c->ncb * bs + 1, &ci_, nzb * bs * bs, &cj_));
    ci_[0] = 0; q = 0;
    for (Ib = 0; Ib < c->ncb; Ib++) {
      const PetscInt *bj;
      PetscCall(MatGetRow(Acb, c->cstart + Ib, &n, &bj, NULL));
      for (m = 0; m < bs; m++) {
        for (k = 0; k < n; k++) for (i = 0; i < bs; i++) cj_[q++] = bj[k] * bs + i;
        ci_[Ib * bs + m + 1] = q;
      }
      PetscCall(MatRestoreRow(Acb, c->cstart + Ib, &n, &bj, NULL));
    }
    PetscCall(MatCreate(comm, &C));
    PetscCall(MatSetSizes(C, c->ncb * bs, c->ncb * bs, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetBlockSizes(C, bs, bs));
    PetscCall(MatSetType(C, MATMPIAIJ));
    PetscCall(MatMPIAIJSetPreallocationCSR(C, ci_, cj_, NULL));
    PetscCall(MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
    PetscCall(PetscFree2(ci_, cj_));
    PetscCall(MatDestroy(&Acb));
  }

  /* touched coarse block rows: P_s^T of the local fine blocks */
  {
    PetscInt *cnt, *Imark, *tpos;
    PetscCall(PetscCalloc1(c->Ncb, &Imark));
    for (i = 0; i < c->nlb; i++)
      for (k = c->pp[i]; k < c->pp[i + 1]; k++) Imark[c->pj[k]]++;
    c->nt = 0;
    for (k = 0; k < c->Ncb; k++) if (Imark[k]) c->nt++;
    PetscCall(PetscMalloc4(c->nt, &c->tI, c->nt + 1, &c->cp, c->pp[c->nlb], &c->ci, c->pp[c->nlb], &c->cv));
    PetscCall(PetscMalloc2(c->Ncb, &tpos, c->nt, &cnt));
    c->cp[0] = 0; t = 0;
    for (k = 0; k < c->Ncb; k++) {
      tpos[k] = -1;
      if (Imark[k]) { c->tI[t] = k; tpos[k] = t; c->cp[t + 1] = c->cp[t] + Imark[k]; cnt[t] = 0; t++; }
    }
    for (i = 0; i < c->nlb; i++)
      for (k = c->pp[i]; k < c->pp[i + 1]; k++) {
        t = tpos[c->pj[k]];
        c->ci[c->cp[t] + cnt[t]] = i; c->cv[c->cp[t] + cnt[t]] = c->pv[k]; cnt[t]++;
      }
    PetscCall(PetscFree(Imark));
    PetscCall(PetscFree2(tpos, cnt));
  }

  /* per touched row: fine block columns of R, coarse block columns of the
     partial row, and where they go */
  {
    PetscInt *jmark, *Jmark, nj = 0, nJ = 0, cap_j, cap_J, *jbuf, *Jbuf;
    PetscCall(PetscMalloc2(c->nlb + c->ngb, &jmark, c->Ncb, &Jmark));
    for (k = 0; k < c->nlb + c->ngb; k++) jmark[k] = -1;
    for (k = 0; k < c->Ncb; k++) Jmark[k] = -1;
    cap_j = cap_J = 64 * (c->nt + 1);
    PetscCall(PetscMalloc1(cap_j, &jbuf));  /* grown one by one: separate allocations */
    PetscCall(PetscMalloc1(cap_J, &Jbuf));
    PetscCall(PetscMalloc2(c->nt + 1, &c->jp, c->nt + 1, &c->Jp));
    c->jp[0] = c->Jp[0] = 0; c->mxj = c->mxJ = 0;
#define GBLK_PUSH(buf, cap, len, v) do { \
      if (len == cap) { PetscInt *x_; cap *= 2; PetscCall(PetscMalloc1(cap, &x_)); PetscCall(PetscArraycpy(x_, buf, len)); PetscCall(PetscFree(buf)); buf = x_; } \
      buf[len++] = (v); } while (0)
    for (t = 0; t < c->nt; t++) {
      PetscInt j0 = nj, J0 = nJ, q, kk;
      for (q = c->cp[t]; q < c->cp[t + 1]; q++) {
        i = c->ci[q];
        for (m = 0; m < bs; m++) {
          PetscInt r = i * bs + m;
          for (kk = ia[r]; kk < ia[r + 1]; kk++) {
            PetscInt b = c->dcb[ja[kk]];
            if (jmark[b] != t) { jmark[b] = t; GBLK_PUSH(jbuf, cap_j, nj, b); }
          }
          for (kk = oia[r]; kk < oia[r + 1]; kk++) {
            PetscInt b = c->ocb[oja[kk]];
            if (jmark[b] != t) { jmark[b] = t; GBLK_PUSH(jbuf, cap_j, nj, b); }
          }
        }
      }
      for (q = j0; q < nj; q++)
        for (kk = c->pp[jbuf[q]]; kk < c->pp[jbuf[q] + 1]; kk++) {
          PetscInt J = c->pj[kk];
          if (Jmark[J] != t) { Jmark[J] = t; GBLK_PUSH(Jbuf, cap_J, nJ, J); }
        }
      qsort(Jbuf + J0, (size_t)(nJ - J0), sizeof(PetscInt), cmp_int);
      c->jp[t + 1] = nj; c->Jp[t + 1] = nJ;
      c->mxj = PetscMax(c->mxj, nj - j0);
      c->mxJ = PetscMax(c->mxJ, nJ - J0);
    }
#undef GBLK_PUSH
    PetscCall(PetscMalloc3(nj, &c->jl, nJ, &c->Jl, nJ, &c->Jo));
    PetscCall(PetscArraycpy(c->jl, jbuf, nj));
    PetscCall(PetscArraycpy(c->Jl, Jbuf, nJ));
    PetscCall(PetscFree(jbuf));
    PetscCall(PetscFree(Jbuf));

    /* owned rows: block offsets in C's diagonal / off-diagonal row */
    {
      Mat             Cd, Co;
      const PetscInt *cgarr, *cia, *cja, *coia, *coja;
      PetscInt        rem = 0, *Jpos;
      PetscCall(MatMPIAIJGetSeqAIJ(C, &Cd, &Co, &cgarr));
      PetscCall(MatGetRowIJ(Cd, 0, PETSC_FALSE, PETSC_FALSE, &n, &cia, &cja, &done));
      PetscCall(MatGetRowIJ(Co, 0, PETSC_FALSE, PETSC_FALSE, &n, &coia, &coja, &done));
      PetscCall(PetscMalloc1(c->nt, &c->rb));
      Jpos = Jmark;
      for (k = 0; k < c->Ncb; k++) Jpos[k] = PETSC_MIN_INT;
      for (t = 0; t < c->nt; t++) {
        PetscInt Ib = c->tI[t], q;
        if (Ib < c->cstart || Ib >= c->cstart + c->ncb) {
          c->rb[t] = rem; rem += (c->Jp[t + 1] - c->Jp[t]) * bs * bs;
          continue;
        }
        c->rb[t] = -1;
        {
          PetscInt r = (Ib - c->cstart) * bs;
          for (k = cia[r]; k < cia[r + 1]; k += bs) Jpos[c->cstart + cja[k] / bs] = k - cia[r];
          for (k = coia[r]; k < coia[r + 1]; k += bs) Jpos[cgarr[coja[k]] / bs] = -1 - (k - coia[r]);
          for (q = c->Jp[t]; q < c->Jp[t + 1]; q++) {
            PetscCheck(Jpos[c->Jl[q]] != PETSC_MIN_INT, PETSC_COMM_SELF, PETSC_ERR_PLIB, "galerkin_blk: block %" PetscInt_FMT " not in C's row", c->Jl[q]);
            c->Jo[q] = Jpos[c->Jl[q]];
          }
          for (k = cia[r]; k < cia[r + 1]; k += bs) Jpos[c->cstart + cja[k] / bs] = PETSC_MIN_INT;
          for (k = coia[r]; k < coia[r + 1]; k += bs) Jpos[cgarr[coja[k]] / bs] = PETSC_MIN_INT;
        }
      }
      PetscCall(PetscMalloc1(rem + 1, &c->rv));
      PetscCall(MatRestoreRowIJ(Cd, 0, PETSC_FALSE, PETSC_FALSE, &n, &cia, &cja, &done));
      PetscCall(MatRestoreRowIJ(Co, 0, PETSC_FALSE, PETSC_FALSE, &n, &coia, &coja, &done));
    }
    PetscCall(PetscFree2(jmark, Jmark));
  }

  PetscCall(MatRestoreRowIJ(Ad, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscCall(MatRestoreRowIJ(Ao, 0, PETSC_FALSE, PETSC_FALSE, &n, &oia, &oja, &done));
  PetscCall(PetscFree(gblk));
  *Cout = C;
  *cout = c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Numeric phase: C's values from A's. */
static PetscErrorCode gblk_numeric(Mat A, Mat C, gblk_ctx *c)
{
  Mat                Ad, Ao, Cd, Co;
  const PetscInt    *ia, *ja, *oia, *oja, *cia, *coia, *cgarr;
  const PetscScalar *ad, *ao;
  PetscScalar       *cd, *co;
  PetscInt           n, t;
  PetscBool          done;
  const PetscInt     bs = c->bs;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(C));
  PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, NULL));
  PetscCall(MatMPIAIJGetSeqAIJ(C, &Cd, &Co, &cgarr));
  PetscCall(MatGetRowIJ(Ad, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscCall(MatGetRowIJ(Ao, 0, PETSC_FALSE, PETSC_FALSE, &n, &oia, &oja, &done));
  PetscCall(MatGetRowIJ(Cd, 0, PETSC_FALSE, PETSC_FALSE, &n, &cia, NULL, &done));
  PetscCall(MatGetRowIJ(Co, 0, PETSC_FALSE, PETSC_FALSE, &n, &coia, NULL, &done));
  PetscCall(MatSeqAIJGetArrayRead(Ad, &ad));
  PetscCall(MatSeqAIJGetArrayRead(Ao, &ao));
  PetscCall(MatSeqAIJGetArray(Cd, &cd));
  PetscCall(MatSeqAIJGetArray(Co, &co));

#if defined(_OPENMP)
  #pragma omp parallel
#endif
  {
    const PetscInt ldRmax = c->mxj * bs, ldOmax = c->mxJ * bs;
    PetscScalar   *R = NULL, *O = NULL;
    PetscInt      *jmark = NULL, *Jmark = NULL, k;
    /* thread-private work (PETSc's malloc is not thread-safe) */
    R     = (PetscScalar *)malloc(sizeof(PetscScalar) * (size_t)(ldRmax * bs + 1));
    O     = (PetscScalar *)malloc(sizeof(PetscScalar) * (size_t)(ldOmax * bs + 1));
    jmark = (PetscInt *)malloc(sizeof(PetscInt) * (size_t)(c->nlb + c->ngb + 1));
    Jmark = (PetscInt *)malloc(sizeof(PetscInt) * (size_t)(c->Ncb + 1));
    for (k = 0; k < c->nlb + c->ngb; k++) jmark[k] = -1;
    for (k = 0; k < c->Ncb; k++) Jmark[k] = -1;
#if defined(_OPENMP)
    #pragma omp for schedule(dynamic, 1)
#endif
    for (t = 0; t < c->nt; t++) {
      const PetscInt j0 = c->jp[t], nj = c->jp[t + 1] - j0, J0 = c->Jp[t], nJ = c->Jp[t + 1] - J0;
      const PetscInt ldR = nj * bs, ldO = nJ * bs;
      PetscInt       q, m, kk;
      for (q = 0; q < nj; q++) jmark[c->jl[j0 + q]] = q * bs;
      for (q = 0; q < nJ; q++) Jmark[c->Jl[J0 + q]] = q * bs;
      memset(R, 0, sizeof(PetscScalar) * (size_t)(ldR * bs));
      memset(O, 0, sizeof(PetscScalar) * (size_t)(ldO * bs));
      /* R = sum_i P_s(i, I) A(i, :) */
      for (q = c->cp[t]; q < c->cp[t + 1]; q++) {
        const PetscInt    i = c->ci[q];
        const PetscScalar p = c->cv[q];
        for (m = 0; m < bs; m++) {
          const PetscInt r  = i * bs + m;
          PetscScalar   *Rm = R + m * ldR;
          for (kk = ia[r]; kk < ia[r + 1]; kk++) Rm[jmark[c->dcb[ja[kk]]] + c->dcn[ja[kk]]] += p * ad[kk];
          for (kk = oia[r]; kk < oia[r + 1]; kk++) Rm[jmark[c->ocb[oja[kk]]] + c->ocn[oja[kk]]] += p * ao[kk];
        }
      }
      /* O = R P_s, block by block */
      for (q = 0; q < nj; q++) {
        const PetscInt j = c->jl[j0 + q];
        for (kk = c->pp[j]; kk < c->pp[j + 1]; kk++) {
          const PetscScalar w  = c->pv[kk];
          const PetscInt    oo = Jmark[c->pj[kk]];
          for (m = 0; m < bs; m++) {
            const PetscScalar *src = R + m * ldR + q * bs;
            PetscScalar       *dst = O + m * ldO + oo;
            PetscInt           x;
#if defined(_OPENMP)
            #pragma omp simd
#endif
            for (x = 0; x < bs; x++) dst[x] += w * src[x];
          }
        }
      }
      /* out: owned rows into C's arrays, remote ones into the buffer */
      if (c->rb[t] < 0) {
        const PetscInt r0 = (c->tI[t] - c->cstart) * bs;
        for (m = 0; m < bs; m++)
          for (q = 0; q < nJ; q++) {
            const PetscInt off = c->Jo[J0 + q];
            PetscScalar   *dst = off >= 0 ? cd + cia[r0 + m] + off : co + coia[r0 + m] + (-1 - off);
            memcpy(dst, O + m * ldO + q * bs, sizeof(PetscScalar) * (size_t)bs);
          }
      } else {
        memcpy(c->rv + c->rb[t], O, sizeof(PetscScalar) * (size_t)(ldO * bs));
      }
      for (q = 0; q < nj; q++) jmark[c->jl[j0 + q]] = -1;
      for (q = 0; q < nJ; q++) Jmark[c->Jl[J0 + q]] = -1;
    }
    free(R); free(O); free(jmark); free(Jmark);
  }

  PetscCall(MatSeqAIJRestoreArray(Cd, &cd));
  PetscCall(MatSeqAIJRestoreArray(Co, &co));
  PetscCall(MatSeqAIJRestoreArrayRead(Ad, &ad));
  PetscCall(MatSeqAIJRestoreArrayRead(Ao, &ao));
  PetscCall(MatRestoreRowIJ(Ad, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
  PetscCall(MatRestoreRowIJ(Ao, 0, PETSC_FALSE, PETSC_FALSE, &n, &oia, &oja, &done));
  PetscCall(MatRestoreRowIJ(Cd, 0, PETSC_FALSE, PETSC_FALSE, &n, &cia, NULL, &done));
  PetscCall(MatRestoreRowIJ(Co, 0, PETSC_FALSE, PETSC_FALSE, &n, &coia, NULL, &done));

  /* partial rows owned elsewhere */
  {
    PetscInt *rows, *cols, mxc = c->mxJ * bs;
    PetscCall(PetscMalloc2(bs, &rows, mxc + 1, &cols));
    for (t = 0; t < c->nt; t++) {
      PetscInt m, q, x, nJ = c->Jp[t + 1] - c->Jp[t];
      if (c->rb[t] < 0) continue;
      for (m = 0; m < bs; m++) rows[m] = c->tI[t] * bs + m;
      for (q = 0; q < nJ; q++) for (x = 0; x < bs; x++) cols[q * bs + x] = c->Jl[c->Jp[t] + q] * bs + x;
      PetscCall(MatSetValues(C, bs, rows, nJ * bs, cols, c->rv + c->rb[t], ADD_VALUES));
    }
    PetscCall(PetscFree2(rows, cols));
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode jorek_galerkin_blk(Mat A, Mat Ps, int bs, int reuse, Mat *C, int *ok)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)A);
  PetscBool mpi, pmpi;
  gblk_ctx *c = NULL;
  int       good;

  PetscFunctionBegin;
  *ok = 0;
  if (reuse) {
    PetscContainer cont;
    PetscCall(PetscObjectQuery((PetscObject)*C, GBLK_KEY, (PetscObject *)&cont));
    good = (cont != NULL);
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &good, 1, MPI_INT, MPI_LAND, comm));
    if (!good) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscContainerGetPointer(cont, (void **)&c));
    PetscCall(gblk_numeric(A, *C, c));
    *ok = 1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &mpi));
  PetscCall(PetscObjectTypeCompare((PetscObject)Ps, MATMPIAIJ, &pmpi));
  good = mpi && pmpi && bs > 1;
  if (good) {
    PetscInt nA, rs, np;
    PetscCall(MatGetLocalSize(A, &nA, NULL));
    PetscCall(MatGetOwnershipRange(A, &rs, NULL));
    PetscCall(MatGetLocalSize(Ps, &np, NULL));
    good = (nA % bs == 0) && (rs % bs == 0) && (np * bs == nA);
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &good, 1, MPI_INT, MPI_LAND, comm));
  if (!good) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(gblk_symbolic(A, Ps, bs, C, &c));
  PetscCall(PetscObjectContainerCompose((PetscObject)*C, GBLK_KEY, c, (PetscCtxDestroyFn *)ctx_destroy));
  PetscCall(gblk_numeric(A, *C, c));
  *ok = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
