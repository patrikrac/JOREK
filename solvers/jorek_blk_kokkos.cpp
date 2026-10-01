/* Device kernel of the C1 GMG block smoothers (mod_petsc_pc_gmg, blk_apply).
 *
 * The smoother blocks of one level (radial lines, flux-surface rings; blk_t)
 * are factored on the host by LAPACK (dgbtrf / dgetrf), as on the CPU path.
 * This file keeps a copy of the factors in Kokkos views and applies them to
 * PETSc Kokkos vectors: gather -> pivoted band / dense solve -> scatter, one
 * team per block, the band updates on the team's vector lanes. With the
 * zebra smoothers the second colour subtracts its coupling to the first
 * (blk_t's zp / zc / zv) before its solves, as zebra_solve does.
 *
 * Written against Kokkos only, so the backend is the one PETSc was built
 * with (CUDA, HIP, SYCL, OpenMP). The solves are those of band_solve and
 * dgetrs, operation by operation: the device result equals the host's up to
 * the order of the updates, which mod_petsc_pc_gmg gates at the first build.
 *
 * The axis blocks (blk_t%axblk) are not handled here.
 */
#if defined(USE_PETSC) && defined(USE_GPU_PC)
#include <petscvec_kokkos.hpp>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using IntView   = Kokkos::View<int *, MemSpace>;
using LongView  = Kokkos::View<long long *, MemSpace>;
using RealView  = Kokkos::View<PetscScalar *, MemSpace>;
using CRealView = Kokkos::View<const PetscScalar *, MemSpace>;
using Team      = Kokkos::TeamPolicy<ExecSpace>::member_type;

struct BlkDev {
  int      nb = 0, nrow = 0, ngh = 0;
  int      nact[2] = {0, 0}; /* blocks of pass 0 / pass 1 (zebra colours) */
  int      vlen    = 1;      /* vector lanes per team */
  IntView  act[2];           /* their block indices, 0-based */
  IntView  off, sz, kl, ku, band, rows, piv, zp, zc;
  LongView loff;
  RealView lu, zv, w, yg;
};

template <class V, class T>
void upload(V &dst, const char *name, const T *src, size_t n)
{
  if (dst.extent(0) != n) dst = V(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), n);
  if (!n) return;
  Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(src, n);
  Kokkos::deep_copy(dst, h);
}

/* One team thread per block: everything outside a vector loop runs on all of
 * its vector lanes, so every scalar update is a single() whose result the
 * lanes then share, and every vector loop ends in a barrier (a vector loop
 * does not order the lanes' writes against their next reads on CUDA). */

/* row interchange j <-> l; returns the new x[j] */
KOKKOS_INLINE_FUNCTION PetscScalar pivot(PetscScalar *x, int j, int l)
{
  if (l != j) {
    const PetscScalar t = x[l];
    x[l]                = x[j];
    x[j]                = t;
  }
  return x[j];
}

/* x <- (P L U)^-1 x, dgbtrf's band storage: the algorithm of band_solve */
KOKKOS_INLINE_FUNCTION void band_solve(const Team &team, int n, int kl, int ku, const PetscScalar *ab, const int *ipiv, PetscScalar *x)
{
  const int ldab = 2 * kl + ku + 1, kd = kl + ku;
  if (kl > 0) {
    for (int j = 0; j < n - 1; j++) {
      const int          lm  = (kl < n - 1 - j) ? kl : n - 1 - j;
      const int          l   = ipiv[j] - 1;
      const PetscScalar *col = ab + (size_t)j * ldab + kd;
      PetscScalar        t;
      Kokkos::single(Kokkos::PerThread(team), [=](PetscScalar &v) { v = pivot(x, j, l); }, t);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 1, lm + 1), [=](const int i) { x[j + i] -= col[i] * t; });
      team.team_barrier();
    }
  }
  for (int j = n - 1; j >= 0; j--) {
    const PetscScalar *col = ab + (size_t)j * ldab + kd;
    const int          lo  = (j - kl - ku > 0) ? j - kl - ku : 0;
    PetscScalar        t;
    Kokkos::single(Kokkos::PerThread(team), [=](PetscScalar &v) { v = (x[j] /= col[0]); }, t);
    if (t == 0.0) continue;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, lo, j), [=](const int i) { x[i] -= t * col[i - j]; });
    team.team_barrier();
  }
}

/* x <- (P L U)^-1 x, dgetrf's dense storage: dgetrs('N') for one right-hand side */
KOKKOS_INLINE_FUNCTION void dense_solve(const Team &team, int n, const PetscScalar *a, const int *ipiv, PetscScalar *x)
{
  /* dgetrf's L is stored with all interchanges applied, so they all come first */
  int done; /* a value, so that the lanes wait for the single */
  Kokkos::single(
    Kokkos::PerThread(team),
    [=](int &v) {
      for (int j = 0; j < n; j++) pivot(x, j, ipiv[j] - 1);
      v = 1;
    },
    done);
  for (int j = 0; j < n; j++) {
    const PetscScalar *col = a + (size_t)j * n;
    const PetscScalar  t   = x[j];
    if (t != 0.0) Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, j + 1, n), [=](const int i) { x[i] -= col[i] * t; });
    team.team_barrier();
  }
  for (int j = n - 1; j >= 0; j--) {
    const PetscScalar *col = a + (size_t)j * n;
    PetscScalar        t;
    Kokkos::single(Kokkos::PerThread(team), [=](PetscScalar &v) { v = (x[j] /= col[j]); }, t);
    if (t == 0.0) continue;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 0, j), [=](const int i) { x[i] -= t * col[i]; });
    team.team_barrier();
  }
}

} // namespace

extern "C" {

/* Structure of a level's blocks, once per operator pattern. Arrays as in
 * blk_t: off/sz/kl/ku/band/loff per block (axblk: not ours), rows per block
 * row (0-based, >= nrow: ghost row), bcol the zebra colour per block (NULL:
 * one pass over all blocks), zp (1-based pointers) / zc the colour-1 coupling.
 * *h = NULL creates the handle. */
PetscErrorCode jorek_blk_kokkos(void **h, int nb, int nrow, int ngh, const int *off, const int *sz, const int *kl, const int *ku, const int *band, const long long *loff, const int *axblk, const int *rows, const int *bcol, int nzp, const int *zp, int nz, const int *zc, double *stats)
{
  BlkDev *B;
  int     nr_tot = 0, maxn = 0, maxkl = 0;
  double  work = 0.0;

  PetscFunctionBeginUser;
  if (!*h) PetscCallCXX(*h = new BlkDev);
  B       = static_cast<BlkDev *>(*h);
  B->nb   = nb;
  B->nrow = nrow;
  B->ngh  = ngh;
  for (int b = 0; b < nb; b++) nr_tot = PetscMax(nr_tot, off[b] + sz[b]);
  {
    std::vector<int> a0, a1;
    for (int b = 0; b < nb; b++) {
      if (axblk[b]) continue;
      if (bcol && bcol[b] == 1) a1.push_back(b);
      else a0.push_back(b);
      maxn = PetscMax(maxn, sz[b]);
      if (band[b]) {
        maxkl = PetscMax(maxkl, kl[b]);
        work += (double)sz[b] * (2 * kl[b] + ku[b]);
      } else work += (double)sz[b] * sz[b];
    }
    B->nact[0] = (int)a0.size();
    B->nact[1] = (int)a1.size();
    PetscCallCXX(upload(B->act[0], "blk_act0", a0.data(), a0.size()));
    PetscCallCXX(upload(B->act[1], "blk_act1", a1.data(), a1.size()));
  }
  PetscCallCXX(upload(B->off, "blk_off", off, nb));
  PetscCallCXX(upload(B->sz, "blk_sz", sz, nb));
  PetscCallCXX(upload(B->kl, "blk_kl", kl, nb));
  PetscCallCXX(upload(B->ku, "blk_ku", ku, nb));
  PetscCallCXX(upload(B->band, "blk_band", band, nb));
  PetscCallCXX(upload(B->loff, "blk_loff", loff, nb));
  PetscCallCXX(upload(B->rows, "blk_rows", rows, nr_tot));
  PetscCallCXX(upload(B->zp, "blk_zp", zp, nzp));
  PetscCallCXX(upload(B->zc, "blk_zc", zc, nz));
  PetscCallCXX(B->w = RealView(Kokkos::view_alloc(std::string("blk_w"), Kokkos::WithoutInitializing), nr_tot));
  PetscCallCXX(B->yg = RealView("blk_yg", ngh));
  /* vector lanes: the band updates are maxkl long; one lane on host backends */
  B->vlen = 1;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  {
    PetscInt  v = 32; /* a power of two, at most one warp */
    PetscBool set;
    if (maxkl < 8) v = 8;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-sf_gpu_blk_vlen", &v, &set));
    B->vlen = (int)PetscMin(PetscMax(v, 1), 32);
  }
#endif
  stats[0] = maxn;
  stats[1] = maxkl;
  stats[2] = work; /* multiply-adds of one pass */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The factors, at every rebuild: lu (nlu entries), the LAPACK pivots (1-based,
 * per block row) and the zebra coupling's values. */
PetscErrorCode jorek_blk_kokkos_values(void *h, long long nlu, const double *lu, int npiv, const int *piv, int nz, const double *zv)
{
  BlkDev *B = static_cast<BlkDev *>(h);

  PetscFunctionBeginUser;
  PetscCallCXX(upload(B->lu, "blk_lu", lu, (size_t)nlu));
  PetscCallCXX(upload(B->piv, "blk_piv", piv, (size_t)npiv));
  PetscCallCXX(upload(B->zv, "blk_zv", zv, (size_t)nz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y(block rows) = block solves of x; xg: x on the ghost rows (ngh > 0).
 * Pass 1's blocks (zebra colour 1) solve x - zv * y(pass 0), and read pass 0's
 * solutions on the ghost rows, which stay here. y's other rows are kept. */
PetscErrorCode jorek_blk_kokkos_apply(void *h, Vec x, Vec y, Vec xg)
{
  BlkDev                    *B = static_cast<BlkDev *>(h);
  CRealView xv, gv;
  RealView  yv;

  PetscFunctionBeginUser;
  PetscCall(VecGetKokkosView(x, &xv));
  if (B->ngh > 0) PetscCall(VecGetKokkosView(xg, &gv));
  PetscCall(VecGetKokkosView(y, &yv));
  PetscCall(PetscLogGpuTimeBegin());
  for (int c = 0; c < 2; c++) {
    if (!B->nact[c]) continue;
    const int nr  = B->nrow;
    auto      act = B->act[c];
    auto      off = B->off, sz = B->sz, kl = B->kl, ku = B->ku, band = B->band, rows = B->rows, piv = B->piv, zp = B->zp, zc = B->zc;
    auto      loff = B->loff;
    auto      lu = B->lu, zv = B->zv, w = B->w, yg = B->yg;
    PetscCallCXX(Kokkos::parallel_for(
      "jorek_blk_apply", Kokkos::TeamPolicy<ExecSpace>(B->nact[c], 1, B->vlen), KOKKOS_LAMBDA(const Team &team) {
        const int    b = act(team.league_rank());
        const int    n = sz(b), o = off(b);
        PetscScalar *wb = &w(o);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n), [=](const int q) {
          const int   r = rows(o + q);
          PetscScalar t = (r < nr) ? xv(r) : gv(r - nr);
          if (c == 1) {
            for (int k = zp(r) - 1; k < zp(r + 1) - 1; k++) {
              const int cc = zc(k);
              t -= zv(k) * ((cc < nr) ? yv(cc) : yg(cc - nr));
            }
          }
          wb[q] = t;
        });
        team.team_barrier();
        if (band(b)) band_solve(team, n, kl(b), ku(b), &lu(loff(b)), &piv(o), wb);
        else dense_solve(team, n, &lu(loff(b)), &piv(o), wb);
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n), [=](const int q) {
          const int r = rows(o + q);
          if (r < nr) yv(r) = wb[q];
          else if (c == 0) yg(r - nr) = wb[q];
        });
      }));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecRestoreKokkosView(y, &yv));
  if (B->ngh > 0) PetscCall(VecRestoreKokkosView(xg, &gv));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode jorek_blk_kokkos_free(void **h)
{
  PetscFunctionBeginUser;
  if (*h) PetscCallCXX(delete static_cast<BlkDev *>(*h));
  *h = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // extern "C"
#else
/* keep the translation unit non-empty for builds without the device path */
extern "C" int jorek_blk_kokkos_unused(void)
{
  return 0;
}
#endif
