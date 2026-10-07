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
 * A band solve is sequential along its block (2n steps), and a level has only
 * a few hundred blocks per rank: one team each leaves a GPU latency-bound.
 * Band blocks long enough are therefore solved partitioned (SPIKE-type, exact):
 * the block's rows are cut into p interiors I_i separated by p-1 separators
 * S_k of s = max(kl, ku) rows, so that the interiors do not couple. With
 * g = A_II^-1 f_I (p independent band solves), the separators solve
 *   S x_S = f_S - A_SI g,   S = A_SS - A_SI A_II^-1 A_IS  (Schur complement),
 * and the interiors x_I = g - V x_S with the spikes V = A_II^-1 A_IS. The
 * interior LUs, V and S^-1 are formed on the host from the block's unfactored
 * band (jorek_blk_kokkos_spike, before the host LU overwrites it), so the
 * device does p band solves of ~n/p rows and two dense products. The result
 * is the block solve up to round-off (gated by mod_petsc_pc_gmg at the first
 * build). -sf_gpu_spike_m M sets the target interior size (default 128,
 * 0 = off); blocks too short for two interiors of >= s rows stay whole.
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
#include <algorithm>
#include <string>
#include <vector>

extern "C" {
void dgbtrf_(const int *, const int *, const int *, const int *, double *, const int *, int *, int *);
void dgbtrs_(const char *, const int *, const int *, const int *, const int *, const double *, const int *, const int *, double *, const int *, int *, size_t);
void dgetrf_(const int *, const int *, double *, const int *, int *, int *);
void dgetri_(const int *, double *, const int *, const int *, double *, const int *, int *);
}

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using IntView   = Kokkos::View<int *, MemSpace>;
using LongView  = Kokkos::View<long long *, MemSpace>;
using RealView  = Kokkos::View<PetscScalar *, MemSpace>;
using CRealView = Kokkos::View<const PetscScalar *, MemSpace>;
using Team      = Kokkos::TeamPolicy<ExecSpace>::member_type;

/* One partitioned (SPIKE) block, host side: its layout and factors. */
struct SpkHost {
  int                 p = 0, s = 0;    /* interiors, separator width */
  std::vector<int>    a, m;            /* interior starts / sizes (block rows) */
  std::vector<double> lu;              /* interior band LUs, ldab x m_i each */
  std::vector<int>    piv;             /* their pivots, at the interior's rows */
  std::vector<double> v;               /* spikes, row-major m_i x 2s each */
  std::vector<double> c;               /* A(S_k, tail of I_k-1) and A(S_k, head of I_k), row-major s x s */
  std::vector<double> sinv;            /* S^-1, row-major ns x ns */
};

struct BlkDev {
  int      nb = 0, nrow = 0, ngh = 0, nr_tot = 0;
  int      nact[2] = {0, 0}, nactd[2] = {0, 0}; /* whole band / dense blocks of pass 0 / pass 1 (zebra colours) */
  int      maxdn[2] = {0, 0}, maxns[2] = {0, 0}; /* largest dense block / separator system per colour */
  int      vlen    = 1;      /* vector lanes per team */
  IntView  act[2], actd[2];  /* their block indices, 0-based */
  IntView  off, sz, kl, ku, band, rows, piv, zp, zc;
  LongView loff;
  RealView lu, zv, w, yg;
  RealView dinv;             /* dense blocks: explicit inverses, row-major, at their loff */
  /* host copies of the layout, for the partitioning */
  std::vector<int>       h_off, h_sz, h_kl, h_ku, h_band, h_ax, h_col;
  std::vector<long long> h_loff;
  /* partitioned blocks: per block sb_*, per interior pt_* */
  int                  spk_m = 128;                /* target interior size, 0 = off */
  std::vector<SpkHost> spk;                        /* per block (p = 0: whole) */
  int                  nsb[2] = {0, 0}, npt[2] = {0, 0};
  IntView              sb_act[2], pt_act[2];       /* per colour: spike blocks / interiors */
  IntView              sb_b, sb_s, sb_p, sb_p0, sb_ns, sb_rs;
  LongView             sb_c, sb_sinv;
  IntView              pt_a, pt_m, pt_cl, pt_cr, pt_sb;
  LongView             pt_lu, pt_v;
  IntView              ppiv;
  RealView             plu, pv, pc, psinv, rs;
  double               spk_stat[3] = {0, 0, 0};    /* blocks, interiors, MB */
};

/* -log_view events of the kernels (with -log_view_gpu_time: their GPU time) */
PetscLogEvent ev_band = -1, ev_dense, ev_gs, ev_int, ev_sep, ev_upd;

PetscErrorCode events_register()
{
  PetscClassId cid;

  PetscFunctionBeginUser;
  if (ev_band >= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscClassIdRegister("JOREK GMG blocks", &cid));
  PetscCall(PetscLogEventRegister("BLK_Band", cid, &ev_band));
  PetscCall(PetscLogEventRegister("BLK_Dense", cid, &ev_dense));
  PetscCall(PetscLogEventRegister("SPK_GathScat", cid, &ev_gs));
  PetscCall(PetscLogEventRegister("SPK_Interior", cid, &ev_int));
  PetscCall(PetscLogEventRegister("SPK_Sep", cid, &ev_sep));
  PetscCall(PetscLogEventRegister("SPK_Update", cid, &ev_upd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* one kernel launch inside its event */
#define BLK_LAUNCH(ev, launch) \
  do { \
    PetscCall(PetscLogEventBegin(ev, 0, 0, 0, 0)); \
    PetscCall(PetscLogGpuTimeBegin()); \
    PetscCallCXX(launch); \
    PetscCall(PetscLogGpuTimeEnd()); \
    PetscCall(PetscLogEventEnd(ev, 0, 0, 0, 0)); \
  } while (0)

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

/* The partitioning of one band block (n rows, band kl/ku, unfactored band ab
 * in dgbtrf's storage) for target interior size mt; false: keep it whole
 * (too short, or an interior / the Schur complement is singular). */
bool spike_factor(int n, int kl, int ku, const double *ab, int mt, SpkHost &S)
{
  const int ldab = 2 * kl + ku + 1, s = std::max(std::max(kl, ku), 1);
  int       p    = (n + s) / (mt + s);
  while (p >= 2 && (n - (p - 1) * s) / p < s) p--; /* an interior holds the s rows a separator reaches */
  if (p < 2) return false;
  auto A = [&](int i, int j) -> double { /* the block's entry (i, j), 0 outside the band */
    if (i - j > kl || j - i > ku) return 0.0;
    return ab[(size_t)j * ldab + kl + ku + i - j];
  };
  S.p = p;
  S.s = s;
  S.a.resize(p);
  S.m.resize(p);
  {
    const int ni = n - (p - 1) * s, base = ni / p, rem = ni % p;
    int       r  = 0;
    for (int i = 0; i < p; i++) {
      S.a[i] = r;
      S.m[i] = base + (i < rem ? 1 : 0);
      r += S.m[i] + s;
    }
  }
  auto sep = [&](int k) { return S.a[k - 1] + S.m[k - 1]; }; /* first row of separator k = 1..p-1 */
  const int ns = (p - 1) * s;
  size_t    nlu = 0, nv = 0;
  for (int i = 0; i < p; i++) {
    nlu += (size_t)ldab * S.m[i];
    nv += (size_t)S.m[i] * 2 * s;
  }
  S.lu.assign(nlu, 0.0);
  S.piv.assign(n, 0);
  S.v.assign(nv, 0.0);
  S.c.assign((size_t)(p - 1) * 2 * s * s, 0.0);
  std::vector<double> Sm((size_t)ns * ns, 0.0); /* column-major */
  size_t              lo = 0, vo = 0;
  for (int i = 0; i < p; i++) {
    const int m = S.m[i], a = S.a[i];
    double   *L = &S.lu[lo];
    for (int j = 0; j < m; j++)
      for (int q = std::max(0, j - ku); q <= std::min(m - 1, j + kl); q++) L[(size_t)j * ldab + kl + ku + q - j] = A(a + q, a + j);
    int info = 0;
    dgbtrf_(&m, &m, &kl, &ku, L, &ldab, &S.piv[a], &info);
    if (info) return false;
    /* spikes: columns 0..s-1 the left separator, s..2s-1 the right one */
    std::vector<double> R((size_t)m * 2 * s, 0.0); /* column-major m x 2s */
    for (int t = 0; t < s; t++) {
      if (i > 0)
        for (int q = 0; q < m; q++) R[(size_t)t * m + q] = A(a + q, sep(i) + t);
      if (i < p - 1)
        for (int q = 0; q < m; q++) R[(size_t)(s + t) * m + q] = A(a + q, sep(i + 1) + t);
    }
    const int nrhs = 2 * s;
    dgbtrs_("N", &m, &kl, &ku, &nrhs, L, &ldab, &S.piv[a], R.data(), &m, &info, 1);
    if (info) return false;
    for (int q = 0; q < m; q++)
      for (int t = 0; t < 2 * s; t++) S.v[vo + (size_t)q * 2 * s + t] = R[(size_t)t * m + q];
    lo += (size_t)ldab * m;
    vo += (size_t)m * 2 * s;
  }
  /* S = A_SS - A_SI V; the separators couple only through the interiors */
  vo = 0;
  std::vector<size_t> voff(p);
  for (int i = 0; i < p; i++) {
    voff[i] = vo;
    vo += (size_t)S.m[i] * 2 * s;
  }
  for (int k = 1; k < p; k++) {
    const int c0 = sep(k);
    for (int t = 0; t < s; t++) {
      const int row = (k - 1) * s + t;
      for (int u = 0; u < s; u++) Sm[(size_t)((k - 1) * s + u) * ns + row] = A(c0 + t, c0 + u);
      for (int side = 0; side < 2; side++) { /* interior k-1 (S_k on its right), interior k (S_k on its left) */
        const int i = k - 1 + side, a = S.a[i], m = S.m[i];
        for (int q = 0; q < m; q++) {
          const double e = A(c0 + t, a + q);
          if (e == 0.0) continue;
          const double *vr = &S.v[voff[i] + (size_t)q * 2 * s];
          if (i > 0)
            for (int u = 0; u < s; u++) Sm[(size_t)((i - 1) * s + u) * ns + row] -= e * vr[u];
          if (i < p - 1)
            for (int u = 0; u < s; u++) Sm[(size_t)(i * s + u) * ns + row] -= e * vr[s + u];
        }
      }
      /* the couplings applied on the device: tail of I_k-1, head of I_k */
      double   *cl = &S.c[(size_t)(k - 1) * 2 * s * s], *cr = cl + (size_t)s * s;
      const int tl = S.a[k - 1] + S.m[k - 1] - s;
      for (int u = 0; u < s; u++) {
        cl[(size_t)t * s + u] = A(c0 + t, tl + u);
        cr[(size_t)t * s + u] = A(c0 + t, S.a[k] + u);
      }
    }
  }
  {
    std::vector<int> ip(ns);
    int              info = 0, lw = -1;
    double           q;
    dgetrf_(&ns, &ns, Sm.data(), &ns, ip.data(), &info);
    if (info) return false;
    dgetri_(&ns, Sm.data(), &ns, ip.data(), &q, &lw, &info);
    lw = std::max(1, (int)q);
    std::vector<double> wk(lw);
    dgetri_(&ns, Sm.data(), &ns, ip.data(), wk.data(), &lw, &info);
    if (info) return false;
  }
  S.sinv.resize((size_t)ns * ns);
  for (int i = 0; i < ns; i++)
    for (int j = 0; j < ns; j++) S.sinv[(size_t)i * ns + j] = Sm[(size_t)j * ns + i];
  return true;
}

/* active lists of the whole and the partitioned blocks, per colour; the
 * partitioned blocks' device arrays */
PetscErrorCode spike_upload(BlkDev *B)
{
  std::vector<int>       a[2], ad[2], sa[2], pa[2];
  std::vector<int>       sb_b, sb_s, sb_p, sb_p0, sb_ns, sb_rs, pt_a, pt_m, pt_cl, pt_cr, pt_sb, ppiv(B->nr_tot, 1);
  std::vector<long long> sb_c, sb_sinv, pt_lu, pt_v;
  std::vector<double>    plu, pv, pc, psinv;
  int                    nrs = 0;

  PetscFunctionBeginUser;
  for (int b = 0; b < B->nb; b++) {
    if (B->h_ax[b]) continue;
    const int c = B->h_col[b];
    const bool sp = !B->spk.empty() && B->spk[b].p > 0;
    if (!sp) {
      if (B->h_band[b]) a[c].push_back(b);
      else ad[c].push_back(b);
      continue;
    }
    const SpkHost &S  = B->spk[b];
    const int      ib = (int)sb_b.size(), ns = (S.p - 1) * S.s, ldab = 2 * B->h_kl[b] + B->h_ku[b] + 1;
    sa[c].push_back(ib);
    sb_b.push_back(b);
    sb_s.push_back(S.s);
    sb_p.push_back(S.p);
    sb_p0.push_back((int)pt_a.size());
    sb_ns.push_back(ns);
    sb_rs.push_back(nrs);
    nrs += ns;
    sb_c.push_back((long long)pc.size());
    pc.insert(pc.end(), S.c.begin(), S.c.end());
    sb_sinv.push_back((long long)psinv.size());
    psinv.insert(psinv.end(), S.sinv.begin(), S.sinv.end());
    size_t lo = 0, vo = 0;
    for (int i = 0; i < S.p; i++) {
      pa[c].push_back((int)pt_a.size());
      pt_a.push_back(S.a[i]);
      pt_m.push_back(S.m[i]);
      pt_cl.push_back(i > 0 ? S.a[i] - S.s : -1);
      pt_cr.push_back(i < S.p - 1 ? S.a[i] + S.m[i] : -1);
      pt_sb.push_back(ib);
      pt_lu.push_back((long long)(plu.size() + lo));
      pt_v.push_back((long long)(pv.size() + vo));
      lo += (size_t)ldab * S.m[i];
      vo += (size_t)S.m[i] * 2 * S.s;
    }
    plu.insert(plu.end(), S.lu.begin(), S.lu.end());
    pv.insert(pv.end(), S.v.begin(), S.v.end());
    for (int q = 0; q < B->h_sz[b]; q++) ppiv[B->h_off[b] + q] = S.piv[q];
  }
  for (int c = 0; c < 2; c++) {
    B->maxdn[c] = 0;
    for (int b : ad[c]) B->maxdn[c] = std::max(B->maxdn[c], B->h_sz[b]);
    B->maxns[c] = 0;
    for (int k : sa[c]) B->maxns[c] = std::max(B->maxns[c], sb_ns[k]);
    B->nact[c]  = (int)a[c].size();
    B->nactd[c] = (int)ad[c].size();
    PetscCallCXX(upload(B->actd[c], "blk_actd", ad[c].data(), ad[c].size()));
    B->nsb[c]  = (int)sa[c].size();
    B->npt[c]  = (int)pa[c].size();
    PetscCallCXX(upload(B->act[c], "blk_act", a[c].data(), a[c].size()));
    PetscCallCXX(upload(B->sb_act[c], "spk_sb_act", sa[c].data(), sa[c].size()));
    PetscCallCXX(upload(B->pt_act[c], "spk_pt_act", pa[c].data(), pa[c].size()));
  }
  PetscCallCXX(upload(B->sb_b, "spk_sb_b", sb_b.data(), sb_b.size()));
  PetscCallCXX(upload(B->sb_s, "spk_sb_s", sb_s.data(), sb_s.size()));
  PetscCallCXX(upload(B->sb_p, "spk_sb_p", sb_p.data(), sb_p.size()));
  PetscCallCXX(upload(B->sb_p0, "spk_sb_p0", sb_p0.data(), sb_p0.size()));
  PetscCallCXX(upload(B->sb_ns, "spk_sb_ns", sb_ns.data(), sb_ns.size()));
  PetscCallCXX(upload(B->sb_rs, "spk_sb_rs", sb_rs.data(), sb_rs.size()));
  PetscCallCXX(upload(B->sb_c, "spk_sb_c", sb_c.data(), sb_c.size()));
  PetscCallCXX(upload(B->sb_sinv, "spk_sb_sinv", sb_sinv.data(), sb_sinv.size()));
  PetscCallCXX(upload(B->pt_a, "spk_pt_a", pt_a.data(), pt_a.size()));
  PetscCallCXX(upload(B->pt_m, "spk_pt_m", pt_m.data(), pt_m.size()));
  PetscCallCXX(upload(B->pt_cl, "spk_pt_cl", pt_cl.data(), pt_cl.size()));
  PetscCallCXX(upload(B->pt_cr, "spk_pt_cr", pt_cr.data(), pt_cr.size()));
  PetscCallCXX(upload(B->pt_sb, "spk_pt_sb", pt_sb.data(), pt_sb.size()));
  PetscCallCXX(upload(B->pt_lu, "spk_pt_lu", pt_lu.data(), pt_lu.size()));
  PetscCallCXX(upload(B->pt_v, "spk_pt_v", pt_v.data(), pt_v.size()));
  PetscCallCXX(upload(B->ppiv, "spk_piv", ppiv.data(), ppiv.size()));
  PetscCallCXX(upload(B->plu, "spk_lu", plu.data(), plu.size()));
  PetscCallCXX(upload(B->pv, "spk_v", pv.data(), pv.size()));
  PetscCallCXX(upload(B->pc, "spk_c", pc.data(), pc.size()));
  PetscCallCXX(upload(B->psinv, "spk_sinv", psinv.data(), psinv.size()));
  if (B->rs.extent(0) != (size_t)nrs) PetscCallCXX(B->rs = RealView(Kokkos::view_alloc(std::string("spk_rs"), Kokkos::WithoutInitializing), nrs));
  B->spk_stat[0] = (double)sb_b.size();
  B->spk_stat[1] = (double)pt_a.size();
  B->spk_stat[2] = 8.0 * (plu.size() + pv.size() + pc.size() + psinv.size()) / 1048576.0;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(events_register());
  if (!*h) PetscCallCXX(*h = new BlkDev);
  B       = static_cast<BlkDev *>(*h);
  B->nb   = nb;
  B->nrow = nrow;
  B->ngh  = ngh;
  for (int b = 0; b < nb; b++) nr_tot = PetscMax(nr_tot, off[b] + sz[b]);
  B->nr_tot = nr_tot;
  B->h_off.assign(off, off + nb);
  B->h_sz.assign(sz, sz + nb);
  B->h_kl.assign(kl, kl + nb);
  B->h_ku.assign(ku, ku + nb);
  B->h_band.assign(band, band + nb);
  B->h_ax.assign(axblk, axblk + nb);
  B->h_loff.assign(loff, loff + nb);
  B->h_col.assign(nb, 0);
  if (bcol)
    for (int b = 0; b < nb; b++) B->h_col[b] = (bcol[b] == 1) ? 1 : 0;
  B->spk.clear();
  for (int b = 0; b < nb; b++) {
    if (axblk[b]) continue;
    maxn = PetscMax(maxn, sz[b]);
    if (band[b]) {
      maxkl = PetscMax(maxkl, kl[b]);
      work += (double)sz[b] * (2 * kl[b] + ku[b]);
    } else work += (double)sz[b] * sz[b];
  }
  PetscCall(spike_upload(B)); /* no partitioned blocks yet: the active lists */
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
  {
    PetscInt  m = 128;
    PetscBool set;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-sf_gpu_spike_m", &m, &set));
    B->spk_m = (int)PetscMax(m, 0);
  }
  stats[0] = maxn;
  stats[1] = maxkl;
  stats[2] = work; /* multiply-adds of one pass */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The partitioning of the band blocks, at every rebuild, from the blocks'
 * UNFACTORED band storage (blk_t%lu before dgbtrf); the threads of the rank
 * take the blocks. stats = partitioned blocks, interiors, device MB. */
PetscErrorCode jorek_blk_kokkos_spike(void *h, long long nlu, const double *lu, double *stats)
{
  BlkDev *B = static_cast<BlkDev *>(h);

  PetscFunctionBeginUser;
  (void)nlu;
  B->spk.assign(B->nb, SpkHost());
  if (B->spk_m > 0) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < B->nb; b++) {
      if (B->h_ax[b] || !B->h_band[b]) continue;
      if (!spike_factor(B->h_sz[b], B->h_kl[b], B->h_ku[b], lu + B->h_loff[b], B->spk_m, B->spk[b])) B->spk[b] = SpkHost();
    }
  }
  stats[0] = stats[1] = stats[2] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The factors, at every rebuild: lu (nlu entries), the LAPACK pivots (1-based,
 * per block row) and the zebra coupling's values. bad[b] != 0: the host LU of
 * block b failed (it became point Jacobi), so it is not partitioned either. */
PetscErrorCode jorek_blk_kokkos_values(void *h, long long nlu, const double *lu, int npiv, const int *piv, int nz, const double *zv, const int *bad, double *stats)
{
  BlkDev *B = static_cast<BlkDev *>(h);

  PetscFunctionBeginUser;
  PetscCallCXX(upload(B->lu, "blk_lu", lu, (size_t)nlu));
  {
    /* dense blocks: their inverse from the host LU (dgetri), applied as a
     * product; a failed block (point Jacobi) keeps its diagonal LU */
    bool any = false;
    for (int b = 0; b < B->nb; b++) any = any || (!B->h_ax[b] && !B->h_band[b]);
    std::vector<double> inv(any ? (size_t)nlu : 0, 0.0);
#pragma omp parallel for schedule(dynamic, 1) if (any)
    for (int b = 0; b < B->nb; b++) {
      if (B->h_ax[b] || B->h_band[b]) continue;
      const int           n = B->h_sz[b];
      std::vector<double> a(lu + B->h_loff[b], lu + B->h_loff[b] + (size_t)n * n), wk(1);
      std::vector<int>    ip(piv + B->h_off[b], piv + B->h_off[b] + n);
      int                 info = 0, lw = -1;
      dgetri_(&n, a.data(), &n, ip.data(), wk.data(), &lw, &info);
      lw = std::max(1, (int)wk[0]);
      wk.resize(lw);
      dgetri_(&n, a.data(), &n, ip.data(), wk.data(), &lw, &info);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) inv[B->h_loff[b] + (size_t)i * n + j] = a[(size_t)j * n + i];
    }
    if (any) PetscCallCXX(upload(B->dinv, "blk_dinv", inv.data(), inv.size()));
  }
  PetscCallCXX(upload(B->piv, "blk_piv", piv, (size_t)npiv));
  PetscCallCXX(upload(B->zv, "blk_zv", zv, (size_t)nz));
  if (!B->spk.empty())
    for (int b = 0; b < B->nb; b++)
      if (bad[b]) B->spk[b] = SpkHost();
  PetscCall(spike_upload(B));
  B->spk.clear(); /* the host copies are not needed any more */
  for (int i = 0; i < 3; i++) stats[i] = B->spk_stat[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y(block rows) = block solves of x; xg: x on the ghost rows (ngh > 0).
 * Pass 1's blocks (zebra colour 1) solve x - zv * y(pass 0), and read pass 0's
 * solutions on the ghost rows, which stay here. y's other rows are kept. */
PetscErrorCode jorek_blk_kokkos_apply(void *h, Vec x, Vec y, Vec xg)
{
  BlkDev   *B = static_cast<BlkDev *>(h);
  CRealView xv, gv;
  RealView  yv;

  PetscFunctionBeginUser;
  PetscCall(VecGetKokkosView(x, &xv));
  if (B->ngh > 0) PetscCall(VecGetKokkosView(xg, &gv));
  PetscCall(VecGetKokkosView(y, &yv));
  for (int c = 0; c < 2; c++) {
    const int nr  = B->nrow;
    auto      off = B->off, sz = B->sz, kl = B->kl, ku = B->ku, band = B->band, rows = B->rows, piv = B->piv, zp = B->zp, zc = B->zc;
    auto      loff = B->loff;
    auto      lu = B->lu, zv = B->zv, w = B->w, yg = B->yg;
    /* x (minus the colour-0 coupling) into the block's work rows; and back */
    auto gather = [=] KOKKOS_FUNCTION(const Team &team, int b) {
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
    };
    auto scatter = [=] KOKKOS_FUNCTION(const Team &team, int b) {
      const int    n = sz(b), o = off(b);
      PetscScalar *wb = &w(o);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n), [=](const int q) {
        const int r = rows(o + q);
        if (r < nr) yv(r) = wb[q];
        else if (c == 0) yg(r - nr) = wb[q];
      });
    };
    if (B->nact[c]) {
      auto act = B->act[c];
      BLK_LAUNCH(ev_band, Kokkos::parallel_for(
        "jorek_blk_band", Kokkos::TeamPolicy<ExecSpace>(B->nact[c], 1, B->vlen), KOKKOS_LAMBDA(const Team &team) {
          const int b = act(team.league_rank());
          gather(team, b);
          team.team_barrier();
          band_solve(team, sz(b), kl(b), ku(b), &lu(loff(b)), &piv(off(b)), &w(off(b)));
          team.team_barrier();
          scatter(team, b);
        }));
    }
    /* the dense products' teams: TS rows each (one per thread), so that a
     * level with few blocks still fills the device */
    const int TS = (B->vlen > 1) ? 8 : 1;
    if (B->nactd[c]) {
      auto      act = B->actd[c];
      auto      dinv = B->dinv;
      const int nck = (B->maxdn[c] + TS - 1) / TS;
      BLK_LAUNCH(ev_dense, Kokkos::parallel_for(
        "jorek_blk_dgather", Kokkos::TeamPolicy<ExecSpace>(B->nactd[c], 1, B->vlen), KOKKOS_LAMBDA(const Team &team) { gather(team, act(team.league_rank())); }));
      /* y = A^-1 x with the explicit inverse */
      BLK_LAUNCH(ev_dense, Kokkos::parallel_for(
        "jorek_blk_dense", Kokkos::TeamPolicy<ExecSpace>(B->nactd[c] * nck, TS, B->vlen), KOKKOS_LAMBDA(const Team &team) {
          const int b = act(team.league_rank() / nck), n = sz(b), o = off(b);
          const int q = (team.league_rank() % nck) * TS + team.team_rank();
          if (q >= n) return;
          const PetscScalar *ai = &dinv(loff(b) + (long long)q * n), *wb = &w(o);
          PetscScalar        acc = 0.0;
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n), [=](const int j, PetscScalar &a) { a += ai[j] * wb[j]; }, acc);
          Kokkos::single(Kokkos::PerThread(team), [=]() {
            const int r = rows(o + q);
            if (r < nr) yv(r) = acc;
            else if (c == 0) yg(r - nr) = acc;
          });
        }));
    }
    if (!B->nsb[c]) continue;
    auto sact = B->sb_act[c], pact = B->pt_act[c];
    auto sb_b = B->sb_b, sb_s = B->sb_s, sb_p = B->sb_p, sb_p0 = B->sb_p0, sb_ns = B->sb_ns, sb_rs = B->sb_rs;
    auto sb_c = B->sb_c, sb_sinv = B->sb_sinv;
    auto pt_a = B->pt_a, pt_m = B->pt_m, pt_cl = B->pt_cl, pt_cr = B->pt_cr, pt_sb = B->pt_sb;
    auto pt_lu = B->pt_lu, pt_v = B->pt_v;
    auto ppiv  = B->ppiv;
    auto plu = B->plu, pv = B->pv, pc = B->pc, psinv = B->psinv, rs = B->rs;
    const int nsb = B->nsb[c], npt = B->npt[c], vl = B->vlen;
    (void)sb_p0;
    (void)sb_p;
    BLK_LAUNCH(ev_gs, Kokkos::parallel_for(
      "jorek_spk_gather", Kokkos::TeamPolicy<ExecSpace>(nsb, 1, vl), KOKKOS_LAMBDA(const Team &team) { gather(team, sb_b(sact(team.league_rank()))); }));
    /* g = A_II^-1 f_I, every interior on its own */
    BLK_LAUNCH(ev_int, Kokkos::parallel_for(
      "jorek_spk_interior", Kokkos::TeamPolicy<ExecSpace>(npt, 1, vl), KOKKOS_LAMBDA(const Team &team) {
        const int i = pact(team.league_rank()), b = sb_b(pt_sb(i)), o = off(b);
        band_solve(team, pt_m(i), kl(b), ku(b), &plu(pt_lu(i)), &ppiv(o + pt_a(i)), &w(o + pt_a(i)));
      }));
    /* x_S = S^-1 (f_S - A_SI g): the separators' right-hand sides, then S^-1;
     * TS separator rows per team */
    const int nck = (B->maxns[c] + TS - 1) / TS;
    BLK_LAUNCH(ev_sep, Kokkos::parallel_for(
      "jorek_spk_seprhs", Kokkos::TeamPolicy<ExecSpace>(nsb * nck, TS, vl), KOKKOS_LAMBDA(const Team &team) {
        const int k0 = sact(team.league_rank() / nck), s = sb_s(k0), ns = sb_ns(k0);
        const int row = (team.league_rank() % nck) * TS + team.team_rank();
        if (row >= ns) return;
        const int          o = off(sb_b(k0)), i0 = sb_p0(k0);
        const int          k = row / s, t = row % s; /* separator k+1, its row t */
        const int          c0 = pt_a(i0 + k) + pt_m(i0 + k), tl = c0 - s, hd = c0 + s;
        const PetscScalar *wb = &w(o);
        const PetscScalar *cl = &pc(sb_c(k0) + (long long)k * 2 * s * s + (long long)t * s), *cr = cl + (long long)s * s;
        PetscScalar        acc = 0.0;
        Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(team, s), [=](const int u, PetscScalar &a) { a += cl[u] * wb[tl + u] + cr[u] * wb[hd + u]; }, acc);
        Kokkos::single(Kokkos::PerThread(team), [=]() { rs(sb_rs(k0) + row) = wb[c0 + t] - acc; });
      }));
    BLK_LAUNCH(ev_sep, Kokkos::parallel_for(
      "jorek_spk_sepsol", Kokkos::TeamPolicy<ExecSpace>(nsb * nck, TS, vl), KOKKOS_LAMBDA(const Team &team) {
        const int k0 = sact(team.league_rank() / nck), s = sb_s(k0), ns = sb_ns(k0);
        const int row = (team.league_rank() % nck) * TS + team.team_rank();
        if (row >= ns) return;
        const int          o = off(sb_b(k0)), i0 = sb_p0(k0), k = row / s, t = row % s;
        const PetscScalar *si = &psinv(sb_sinv(k0) + (long long)row * ns), *r = &rs(sb_rs(k0));
        PetscScalar        acc = 0.0;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, ns), [=](const int j, PetscScalar &a) { a += si[j] * r[j]; }, acc);
        Kokkos::single(Kokkos::PerThread(team), [=]() { w(o + pt_a(i0 + k) + pt_m(i0 + k) + t) = acc; });
      }));
    /* x_I = g - V x_S */
    BLK_LAUNCH(ev_upd, Kokkos::parallel_for(
      "jorek_spk_update", Kokkos::TeamPolicy<ExecSpace>(npt, Kokkos::AUTO, vl), KOKKOS_LAMBDA(const Team &team) {
        const int          i = pact(team.league_rank()), b = sb_b(pt_sb(i)), o = off(b), s = sb_s(pt_sb(i));
        const int          a = pt_a(i), cl = pt_cl(i), cr = pt_cr(i);
        const PetscScalar *v = &pv(pt_v(i));
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, pt_m(i)), [=](const int q) {
          const PetscScalar *vr  = v + (long long)q * 2 * s;
          PetscScalar        acc = 0.0;
          Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team, 2 * s),
            [=](const int j, PetscScalar &sum) {
              if (j < s) {
                if (cl >= 0) sum += vr[j] * w(o + cl + j);
              } else if (cr >= 0) sum += vr[j] * w(o + cr + j - s);
            },
            acc);
          Kokkos::single(Kokkos::PerThread(team), [=]() { w(o + a + q) -= acc; });
        });
      }));
    BLK_LAUNCH(ev_gs, Kokkos::parallel_for(
      "jorek_spk_scatter", Kokkos::TeamPolicy<ExecSpace>(nsb, 1, vl), KOKKOS_LAMBDA(const Team &team) { scatter(team, sb_b(sact(team.league_rank()))); }));
  }
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
