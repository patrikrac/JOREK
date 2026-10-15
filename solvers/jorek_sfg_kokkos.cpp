/* Device value gathers from JOREK's BAIJ Jacobian (mod_petsc_pc_sf_gather,
 * mod_petsc on the device path).
 *
 * JOREK assembles A (MPIBAIJ) and the physics PC's W (MPIBAIJ, bs n_tor) on
 * the host. Their value arrays go to the device once per new state (sources
 * 0 = A, 1 = W; the host arrays are pinned on the first upload where the
 * backend allows), and everything built from them is gathered there:
 *   - the outer FGMRES's AIJ copy of A: every BAIJ block entry stored, so
 *     AIJ row r = ib bs + i lists, block by block, the block's row i:
 *     entry e of the row reads block bia(ib) + e / bs, column e % bs;
 *   - the SF path's operators (mod_petsc_pc_sf_gather's targets): per stored
 *     entry k the source position m(k) (+ diagonal, - off-diagonal part, 0
 *     none), plus W's for pair_w's S_uu, times opz where z(k);
 *   - the mode split's family operators (mod_petsc_pc_sf_fam): the source
 *     rank's send buffer gathered the same way (W sources 1 .. 4 by entry),
 *     copied down for the MPI exchange, and on the owner rank the received
 *     values copied up and scattered into the operator's parts (perm: the
 *     MatUpdateMPIAIJWithArray order -> diagonal >= 0 / off-diagonal < 0).
 * A target is an AIJ device matrix (aijkokkos); its diagonal / off-diagonal
 * parts' device values are written in place and then marked as the newer
 * copy -- through MatDiagonalScale by ones, PETSc's public device-side
 * update (a no-op scaling, one pass over the values) -- so host readers
 * sync on their next access as with any device-computed matrix. The kernels
 * run on PETSc's execution space, ordered with PETSc's own device work.
 */
#if defined(USE_PETSC) && defined(USE_GPU_PC)
#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat.h>
#include <petsc/private/petscimpl.h>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#if defined(KOKKOS_ENABLE_CUDA)
  #include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
  #include <hip/hip_runtime.h>
#endif

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using RealView  = Kokkos::View<double *, MemSpace>;
using IntView   = Kokkos::View<int *, MemSpace>;
using I8View    = Kokkos::View<signed char *, MemSpace>;
using Range     = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<long long>>;

/* one BAIJ source on the device, uploaded when its host values changed */
struct Src {
  RealView         d, o;
  PetscObjectState st_d = -1, st_o = -1;
  const double    *pin_d = nullptr, *pin_o = nullptr; /* host arrays registered */
};

struct Part {
  Mat     M = nullptr; /* a sequential part (aijkokkos) */
  IntView m, w;        /* source positions in A, W (w: empty without W) */
  I8View  z;           /* opz flags (empty: none) */
};

struct Target {
  Mat  T = nullptr; /* the MPI (or sequential) matrix, for its object state */
  Part p[2];
};

struct Expand {
  Mat     T = nullptr, Ad = nullptr, Ao = nullptr;
  int     bs = 0;
  IntView bia_d, bia_o; /* the BAIJ parts' block row pointers */
};

constexpr int           NSRC = 5; /* A, and up to four W sources (mod_petsc_pc_sf WS_*) */
Src                     src[NSRC];
std::map<int, Target>   tgt;
Expand                  ex;

struct Fam {
  Mat      T = nullptr, Td = nullptr, To = nullptr;
  IntView  sa, sw;
  I8View   ss, sc;
  Kokkos::View<long long *, MemSpace> perm;
  RealView sbuf, val;
};
std::map<int, Fam> fam;
std::map<PetscInt, Vec> ones; /* MatDiagonalScale's ones, by local row count */
PetscLogEvent           ev_up = -1, ev_ga = -1, ev_bmv = -1;

/* the outer operator as a shell (jorek_sfg_bmv_setup): A's block product
 * straight from source 0, the BAIJ values the gathers read -- no AIJ copy.
 * The parts' block row pointers / block columns (diagonal: local block
 * columns; off-diagonal: compressed, into the ghost vector xg, which the
 * scatter fills from x's owners through A's garray). */
struct Bmv {
  Mat        S = nullptr;
  int        bs = 0, nbr = 0, nbo = 0;
  long long  nzb_d = 0, nzb_o = 0; /* stored blocks per part */
  IntView    ia_d, ja_d, ia_o, ja_o;
  Vec        xg = nullptr;
  VecScatter sc = nullptr;
};
Bmv bmv;

/* the globals' device memory goes before Kokkos does: jorek_sfg_release,
 * called ahead of PetscFinalize (whose registered finalizers run only after
 * it has finalized Kokkos), not the static destructors after it */
PetscErrorCode release()
{
  PetscFunctionBeginUser;
  /* the pinned host arrays stay registered: their owner (A, W) may be gone
   * by now, and unregistering freed memory is not safe; the driver drops the
   * registrations with the process */
  for (auto &S : src) S = Src();
  tgt.clear();
  fam.clear();
  ex = Expand();
  for (auto &v : ones) PetscCall(VecDestroy(&v.second));
  ones.clear();
  /* the shell itself is the caller's (mod_petsc); its views and scatter go */
  PetscCall(VecDestroy(&bmv.xg));
  PetscCall(VecScatterDestroy(&bmv.sc));
  bmv = Bmv();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode events()
{
  PetscFunctionBeginUser;
  PetscCall(PetscKokkosInitializeCheck()); /* the first device object may be ours (no AIJ copy before) */
  if (ev_up < 0) {
    PetscClassId cid;
    PetscCall(PetscClassIdRegister("JOREK SF gather", &cid));
    PetscCall(PetscLogEventRegister("SFG_DevUpload", cid, &ev_up));
    PetscCall(PetscLogEventRegister("SFG_DevGather", cid, &ev_ga));
    PetscCall(PetscLogEventRegister("SFG_BlockMult", cid, &ev_bmv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class V, class T>
void upload(V &dst, const char *name, const T *s, size_t n)
{
  if (dst.extent(0) != n) dst = V(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), n);
  if (!n) return;
  Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(s, n);
  Kokkos::deep_copy(dst, h);
}

/* pin a long-lived host array once: the per-step uploads then run at full
 * link speed (pageable copies are staged); a refusal only costs that */
void pin(const double *p, size_t n, const double *&done)
{
#if defined(KOKKOS_ENABLE_CUDA)
  if (p == done || !n) return;
  if (cudaHostRegister((void *)p, n * sizeof(double), cudaHostRegisterDefault) != cudaSuccess) (void)cudaGetLastError();
  done = p;
#else
  (void)p; (void)n; (void)done;
#endif
}

/* the part's device values are the newer copy (see the file header) */
PetscErrorCode mark_device(Mat M)
{
  PetscInt m;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(M, &m, NULL));
  if (!ones.count(m)) {
    Vec v;
    PetscCall(VecCreate(PETSC_COMM_SELF, &v));
    PetscCall(VecSetSizes(v, m, m));
    PetscCall(VecSetType(v, VECKOKKOS));
    PetscCall(VecSet(v, 1.0));
    ones[m] = v;
  }
  PetscCall(MatDiagonalScale(M, ones[m], NULL)); /* after the gather: one stream, PETSc's */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* device value array of an aijkokkos part, its device copy current */
PetscErrorCode dev_values(Mat M, double **a, long long *nz)
{
  const PetscInt *ii, *jj;
  PetscMemType    mt;
  MatInfo         info;

  PetscFunctionBeginUser;
  PetscCall(MatSeqAIJGetCSRAndMemType(M, &ii, &jj, a, &mt));
  PetscCall(MatGetInfo(M, MAT_LOCAL, &info));
  *nz = (long long)info.nz_used;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y (+)= the part's block rows times x: one team per block row, a block
 * row's rows on the team's threads (a block is column-major, so the threads
 * read consecutive values; x's entry is the same for all of them) */
PetscErrorCode bmv_part(const IntView &ia, const IntView &ja, const RealView &v, const PetscScalar *x, PetscScalar *y, bool add)
{
  PetscFunctionBeginUser;
  const int       bs = bmv.bs;
  const long long bs2 = (long long)bs * bs;
  const double   *va = v.data();
  const auto      body = KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type &t)
  {
    const int ib = t.league_rank();
    const int k0 = ia(ib), k1 = ia(ib + 1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, bs), [&](const int i) {
      double s = 0.0;
      for (int k = k0; k < k1; k++) {
        const double      *b  = va + (long long)k * bs2 + i;
        const PetscScalar *xc = x + (long long)ja(k) * bs;
        for (int j = 0; j < bs; j++) s += b[(long long)j * bs] * xc[j];
      }
      const long long r = (long long)ib * bs + i;
      y[r] = add ? y[r] + s : s;
    });
  };
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  Kokkos::TeamPolicy<ExecSpace> pol(PetscGetKokkosExecutionSpace(), bmv.nbr, std::min(((bs + 31) / 32) * 32, 256));
#else
  Kokkos::TeamPolicy<ExecSpace> pol(PetscGetKokkosExecutionSpace(), bmv.nbr, Kokkos::AUTO());
#endif
  PetscCallCXX(Kokkos::parallel_for("jorek_sfg_bmv", pol, body));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* the shell's MatMult: y = A x from source 0, the ghost blocks' exchange
 * overlapping the diagonal part */
PetscErrorCode bmv_mult(Mat S, Vec x, Vec y)
{
  using CView = Kokkos::View<const PetscScalar *, MemSpace>;
  using View  = Kokkos::View<PetscScalar *, MemSpace>;
  CView xv, gv;
  View  yv;

  PetscFunctionBeginUser;
  (void)S;
  PetscCall(PetscLogEventBegin(ev_bmv, x, y, 0, 0));
  if (bmv.nbo) PetscCall(VecScatterBegin(bmv.sc, x, bmv.xg, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetKokkosView(x, &xv));
  PetscCall(VecGetKokkosViewWrite(y, &yv));
  PetscCall(bmv_part(bmv.ia_d, bmv.ja_d, src[0].d, xv.data(), yv.data(), false));
  PetscCall(VecRestoreKokkosViewWrite(y, &yv));
  PetscCall(VecRestoreKokkosView(x, &xv));
  if (bmv.nbo) {
    PetscCall(VecScatterEnd(bmv.sc, x, bmv.xg, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetKokkosView(bmv.xg, &gv));
    PetscCall(VecGetKokkosView(y, &yv));
    PetscCall(bmv_part(bmv.ia_o, bmv.ja_o, src[0].o, gv.data(), yv.data(), true));
    PetscCall(VecRestoreKokkosView(y, &yv));
    PetscCall(VecRestoreKokkosView(bmv.xg, &gv));
  }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  PetscCall(PetscLogGpuFlops(2.0 * (double)(bmv.nzb_d + bmv.nzb_o) * bmv.bs * bmv.bs));
#else
  PetscCall(PetscLogFlops(2.0 * (double)(bmv.nzb_d + bmv.nzb_o) * bmv.bs * bmv.bs));
#endif
  PetscCall(PetscLogEventEnd(ev_bmv, x, y, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* a part's block structure on the device: block row pointers and columns */
PetscErrorCode bmv_ij(Mat P, IntView &ia, IntView &ja, long long &nzb)
{
  const PetscInt *pi, *pj;
  PetscInt        n;
  PetscBool       done;

  PetscFunctionBeginUser;
  PetscCall(MatGetRowIJ(P, 0, PETSC_FALSE, PETSC_TRUE, &n, &pi, &pj, &done));
  PetscCheck(done && n == bmv.nbr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SF block product: no block structure (%" PetscInt_FMT " block rows)", n);
  std::vector<int> hi(pi, pi + n + 1), hj(pj, pj + pi[n]);
  nzb = pi[n];
  PetscCall(MatRestoreRowIJ(P, 0, PETSC_FALSE, PETSC_TRUE, &n, &pi, &pj, &done));
  PetscCallCXX(upload(ia, "sfg_bmv_ia", hi.data(), hi.size()));
  PetscCallCXX(upload(ja, "sfg_bmv_ja", hj.data(), hj.size()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

extern "C" {

/* Source which (0 A, 1 .. 4 W's) from its BAIJ parts' host value arrays (nd / no
 * entries), uploaded when the parts' object states moved (Bd / Bo give them;
 * Bo may be NULL). */
PetscErrorCode jorek_sfg_source(int which, Mat Bd, Mat Bo, long long nd, const double *vd, long long no, const double *vo)
{
  PetscFunctionBeginUser;
  PetscCheck(which >= 0 && which < NSRC, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "SF device gather: source %d", which);
  Src             &S = src[which];
  PetscObjectState sd, so = -1;

  PetscCall(events());
  PetscCall(PetscObjectStateGet((PetscObject)Bd, &sd));
  if (Bo) PetscCall(PetscObjectStateGet((PetscObject)Bo, &so));
  if (sd == S.st_d && so == S.st_o && S.d.extent(0) == (size_t)nd && S.o.extent(0) == (size_t)no) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(ev_up, 0, 0, 0, 0));
  pin(vd, (size_t)nd, S.pin_d);
  pin(vo, (size_t)no, S.pin_o);
  PetscCallCXX(upload(S.d, "sfg_src_d", vd, (size_t)nd));
  PetscCallCXX(upload(S.o, "sfg_src_o", vo, (size_t)no));
  PetscCallCXX(ExecSpace().fence());
  PetscCall(PetscLogEventEnd(ev_up, 0, 0, 0, 0));
  S.st_d = sd;
  S.st_o = so;
  if (which == 0 && bmv.S) PetscCall(PetscObjectStateIncrease((PetscObject)bmv.S)); /* the outer operator's values moved */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The outer copy T (aijkokkos, parts Ad / Ao) of A, BAIJ block size bs, block
 * row pointers bia_d / bia_o (nbr + 1 each), once per pattern. */
PetscErrorCode jorek_sfg_expand_setup(Mat T, Mat Ad, Mat Ao, int bs, int nbr, const int *bia_d, const int *bia_o)
{
  PetscFunctionBeginUser;
  PetscCall(events());
  ex.T  = T;
  ex.Ad = Ad;
  ex.Ao = Ao;
  ex.bs = bs;
  PetscCallCXX(upload(ex.bia_d, "sfg_bia_d", bia_d, (size_t)nbr + 1));
  PetscCallCXX(upload(ex.bia_o, "sfg_bia_o", bia_o, (size_t)nbr + 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* T = A's current values, from source 0 */
PetscErrorCode jorek_sfg_expand(void)
{
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(ev_ga, 0, 0, 0, 0));
  for (int q = 0; q < 2; q++) {
    Mat M = q ? ex.Ao : ex.Ad;
    if (!M) continue;
    const PetscInt *ii, *jj;
    double         *a;
    PetscMemType    mt;
    PetscInt        m;
    PetscCall(MatSeqAIJGetCSRAndMemType(M, &ii, &jj, &a, &mt));
    PetscCall(MatGetLocalSize(M, &m, NULL));
    const int  bs = ex.bs;
    const long bs2 = (long)bs * bs;
    auto       bia = q ? ex.bia_o : ex.bia_d;
    auto       s   = q ? src[0].o : src[0].d;
    PetscCallCXX(Kokkos::parallel_for(
      "jorek_sfg_expand", Kokkos::TeamPolicy<ExecSpace>(PetscGetKokkosExecutionSpace(), m, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type &t) {
        const int  r = t.league_rank(), ib = r / bs, i = r % bs;
        const long p0 = ii[r], len = ii[r + 1] - p0, k0 = bia(ib);
        Kokkos::parallel_for(Kokkos::TeamVectorRange(t, len), [&](const long e) { a[p0 + e] = s((k0 + e / bs) * bs2 + (e % bs) * bs + i); });
      }));
    PetscCall(mark_device(M));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)ex.T));
  PetscCall(PetscLogEventEnd(ev_ga, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* SF target k: T and its parts Td / To (aijkokkos; To may be NULL), per part
 * the maps (nd / no entries); W's (hw) and the opz flags (hz) where given. */
PetscErrorCode jorek_sfg_target(int k, Mat T, Mat Td, Mat To, long long nd, const int *md, const int *wd, const signed char *zd, long long no, const int *mo, const int *wo, const signed char *zo, int hw, int hz)
{
  Target &G = tgt[k];

  PetscFunctionBeginUser;
  PetscCall(events());
  G.T = T;
  for (int q = 0; q < 2; q++) {
    Part              &P = G.p[q];
    const long long    n = q ? no : nd;
    const int         *m = q ? mo : md, *w = hw ? (q ? wo : wd) : nullptr;
    const signed char *z = hz ? (q ? zo : zd) : nullptr;
    P.M                  = q ? To : Td;
    PetscCallCXX(upload(P.m, "sfg_m", m, (size_t)n));
    PetscCallCXX(upload(P.w, "sfg_w", w, w ? (size_t)n : 0));
    PetscCallCXX(upload(P.z, "sfg_z", z, z ? (size_t)n : 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode jorek_sfg_target_clear(void)
{
  PetscFunctionBeginUser;
  tgt.clear();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Target k from sources 0 (A) and 1 (W), opz for the flagged entries */
PetscErrorCode jorek_sfg_gather(int k, double opz)
{
  PetscFunctionBeginUser;
  PetscCheck(tgt.count(k), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SF device gather: no target %d", k);
  Target &G = tgt[k];
  PetscCall(PetscLogEventBegin(ev_ga, 0, 0, 0, 0));
  for (int q = 0; q < 2; q++) {
    Part &P = G.p[q];
    if (!P.M || !P.m.extent(0)) continue;
    double   *a;
    long long nz = 0;
    PetscCall(dev_values(P.M, &a, &nz));
    PetscCheck(nz == (long long)P.m.extent(0), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "SF device gather: target %d part %d holds %lld entries, map %zu", k, q, nz, P.m.extent(0));
    auto       m = P.m, w = P.w;
    auto       z = P.z;
    auto       ad = src[0].d, ao = src[0].o, wd = src[1].d, wo = src[1].o;
    const bool hw = w.extent(0) > 0, hz = z.extent(0) > 0;
    PetscCallCXX(Kokkos::parallel_for(
      "jorek_sfg_gather", Range(PetscGetKokkosExecutionSpace(), 0, nz), KOKKOS_LAMBDA(const long long e) {
        const int s = m(e);
        double    v = s > 0 ? ad(s - 1) : (s < 0 ? ao(-s - 1) : 0.0);
        if (hz && z(e)) v *= opz;
        if (hw) {
          const int t = w(e);
          v += t > 0 ? wd(t - 1) : (t < 0 ? wo(-t - 1) : 0.0);
        }
        a[e] = v;
      }));
    PetscCall(mark_device(P.M));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)G.T));
  PetscCall(PetscLogEventEnd(ev_ga, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Family operator k (mod_petsc_pc_sf_fam, key tagb): the source side's
 * maps (n entries: sa into A, sw into W source ss (1 .. 4), sc: times opz;
 * sw / ss / sc only with hw / hz) and the owner side's operator T (parts Td,
 * To; To may be NULL) with perm (nval: the received values' destinations). */
PetscErrorCode jorek_sfg_fam_set(int k, long long n, const int *sa, const int *sw, const signed char *ss, const signed char *sc, int hw, int hz, Mat T, Mat Td, Mat To, long long nval, const long long *perm)
{
  Fam &F = fam[k];

  PetscFunctionBeginUser;
  PetscCall(events());
  F.T  = T;
  F.Td = Td;
  F.To = To;
  PetscCallCXX(upload(F.sa, "fam_sa", sa, (size_t)n));
  PetscCallCXX(upload(F.sw, "fam_sw", sw, hw ? (size_t)n : 0));
  PetscCallCXX(upload(F.ss, "fam_ss", ss, hw ? (size_t)n : 0));
  PetscCallCXX(upload(F.sc, "fam_sc", sc, hz ? (size_t)n : 0));
  PetscCallCXX(upload(F.perm, "fam_perm", perm, (size_t)nval));
  if (F.sbuf.extent(0) != (size_t)n) PetscCallCXX(F.sbuf = RealView(Kokkos::view_alloc(std::string("fam_sbuf"), Kokkos::WithoutInitializing), (size_t)n));
  if (F.val.extent(0) != (size_t)nval) PetscCallCXX(F.val = RealView(Kokkos::view_alloc(std::string("fam_val"), Kokkos::WithoutInitializing), (size_t)nval));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Family operator k, source side: its send buffer from the device sources,
 * copied to the host array sbuf (n entries) */
PetscErrorCode jorek_sfg_fam_values(int k, double opz, double *sbuf)
{
  PetscFunctionBeginUser;
  PetscCheck(fam.count(k), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SF device gather: no family operator %d", k);
  Fam &F = fam[k];
  PetscCall(PetscLogEventBegin(ev_ga, 0, 0, 0, 0));
  auto       sa = F.sa, sw = F.sw;
  auto       ss = F.ss, sc = F.sc;
  auto       out = F.sbuf;
  auto       ad = src[0].d, ao = src[0].o;
  auto       w1d = src[1].d, w1o = src[1].o, w2d = src[2].d, w2o = src[2].o, w3d = src[3].d, w3o = src[3].o, w4d = src[4].d, w4o = src[4].o;
  const bool hw = sw.extent(0) > 0, hz = sc.extent(0) > 0;
  const long long n = (long long)sa.extent(0);
  PetscCallCXX(Kokkos::parallel_for(
    "jorek_sfg_fam", Range(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const long long e) {
      const int s = sa(e);
      double    v = s > 0 ? ad(s - 1) : (s < 0 ? ao(-s - 1) : 0.0);
      if (hz && sc(e) == 1) v *= opz;
      if (hw && sw(e) != 0) {
        const int  t = sw(e), q = ss(e);
        const auto &wd = q == 1 ? w1d : (q == 2 ? w2d : (q == 3 ? w3d : w4d));
        const auto &wo = q == 1 ? w1o : (q == 2 ? w2o : (q == 3 ? w3o : w4o));
        v += t > 0 ? wd(t - 1) : wo(-t - 1);
      }
      out(e) = v;
    }));
  if (n) {
    Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(sbuf, (size_t)n);
    PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), h, out));
    PetscCallCXX(PetscGetKokkosExecutionSpace().fence());
  }
  PetscCall(PetscLogEventEnd(ev_ga, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Family operator k, owner side: the received values (host, nval, in the
 * MatUpdateMPIAIJWithArray order) into the operator's device parts */
PetscErrorCode jorek_sfg_fam_install(int k, const double *val)
{
  PetscFunctionBeginUser;
  PetscCheck(fam.count(k), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SF device gather: no family operator %d", k);
  Fam &F = fam[k];
  PetscCall(PetscLogEventBegin(ev_ga, 0, 0, 0, 0));
  const long long nv = (long long)F.perm.extent(0);
  if (nv) {
    Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(val, (size_t)nv);
    PetscCallCXX(Kokkos::deep_copy(F.val, h));
  }
  double   *ad = nullptr, *ao = nullptr;
  long long nd, no = 0;
  PetscCall(dev_values(F.Td, &ad, &nd));
  if (F.To) PetscCall(dev_values(F.To, &ao, &no));
  auto perm = F.perm;
  auto v    = F.val;
  PetscCallCXX(Kokkos::parallel_for(
    "jorek_sfg_fam_install", Range(PetscGetKokkosExecutionSpace(), 0, nv), KOKKOS_LAMBDA(const long long e) {
      const long long d = perm(e);
      if (d >= 0) ad[d] = v(e);
      else ao[-d - 1] = v(e);
    }));
  PetscCall(mark_device(F.Td));
  if (F.To) PetscCall(mark_device(F.To));
  PetscCall(PetscObjectStateIncrease((PetscObject)F.T));
  PetscCall(PetscLogEventEnd(ev_ga, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The outer operator *S: a shell whose MatMult is A's block product on the
 * device from source 0 (A: the MPIBAIJ / SeqBAIJ Jacobian; its values must be
 * uploaded, jorek_sfg_source(0, ..), before each use). Gate: S x against
 * PETSc's host BAIJ product on a random x, *rel = max |diff| / max |A x|. */
PetscErrorCode jorek_sfg_bmv_setup(Mat A, Mat *S, double *rel)
{
  MPI_Comm        comm;
  Mat             Ad = A, Ao = nullptr;
  const PetscInt *garray = nullptr;
  PetscBool       mpi;
  PetscInt        m, n, M, N, bs;

  PetscFunctionBeginUser;
  PetscCall(events());
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIBAIJ, &mpi));
  if (mpi) PetscCall(MatMPIBAIJGetSeqBAIJ(A, &Ad, &Ao, &garray));
  PetscCall(MatGetBlockSize(A, &bs));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCheck(src[0].d.extent(0) > 0, PETSC_COMM_SELF, PETSC_ERR_ORDER, "SF block product: A's values not uploaded (jorek_sfg_source 0)");
  bmv.bs  = (int)bs;
  bmv.nbr = (int)(m / bs);
  PetscCall(bmv_ij(Ad, bmv.ia_d, bmv.ja_d, bmv.nzb_d));
  PetscCheck(src[0].d.extent(0) == (size_t)(bmv.nzb_d * bs * bs), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "SF block product: source 0 holds %zu values, the diagonal part %lld blocks", src[0].d.extent(0), bmv.nzb_d);
  bmv.nbo = 0;
  if (Ao) {
    PetscInt nco;
    PetscCall(MatGetSize(Ao, NULL, &nco));
    bmv.nbo = (int)(nco / bs);
  }
  if (bmv.nbo) {
    PetscCall(bmv_ij(Ao, bmv.ia_o, bmv.ja_o, bmv.nzb_o));
    PetscCheck(src[0].o.extent(0) == (size_t)(bmv.nzb_o * bs * bs), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "SF block product: source 0 holds %zu off-diagonal values, the part %lld blocks", src[0].o.extent(0), bmv.nzb_o);
  }
  PetscCall(MatCreateShell(comm, m, n, M, N, NULL, S));
  PetscCall(MatShellSetOperation(*S, MATOP_MULT, (PetscErrorCodeFn *)bmv_mult));
  PetscCall(MatShellSetVecType(*S, VECKOKKOS));
  PetscCall(MatSetBlockSize(*S, bs));
  bmv.S = *S;
  {
    Vec x;
    PetscCall(MatCreateVecs(*S, &x, NULL));
    if (bmv.nbo) {
      IS from;
      PetscCall(VecCreateSeqKokkos(PETSC_COMM_SELF, (PetscInt)bmv.nbo * bs, &bmv.xg));
      PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, bmv.nbo, garray, PETSC_COPY_VALUES, &from));
      PetscCall(VecScatterCreate(x, from, bmv.xg, NULL, &bmv.sc));
      PetscCall(ISDestroy(&from));
    }
    PetscCall(VecDestroy(&x));
  }
  /* the gate */
  {
    Vec                xh, yh, xd, yd;
    const PetscScalar *ph, *pd;
    PetscScalar       *pw;
    PetscRandom        rnd;
    PetscInt           nl;
    double             loc[2] = {0.0, 0.0}, glo[2];
    PetscCall(MatCreateVecs(A, &xh, &yh));
    PetscCall(MatCreateVecs(*S, &xd, &yd));
    PetscCall(PetscRandomCreate(comm, &rnd));
    PetscCall(PetscRandomSetFromOptions(rnd));
    PetscCall(VecSetRandom(xh, rnd));
    PetscCall(PetscRandomDestroy(&rnd));
    PetscCall(VecGetLocalSize(xh, &nl));
    PetscCall(VecGetArrayRead(xh, &ph));
    PetscCall(VecGetArrayWrite(xd, &pw));
    for (PetscInt i = 0; i < nl; i++) pw[i] = ph[i];
    PetscCall(VecRestoreArrayWrite(xd, &pw));
    PetscCall(VecRestoreArrayRead(xh, &ph));
    PetscCall(MatMult(A, xh, yh));
    PetscCall(MatMult(*S, xd, yd));
    PetscCall(VecGetArrayRead(yh, &ph));
    PetscCall(VecGetArrayRead(yd, &pd));
    for (PetscInt i = 0; i < nl; i++) {
      loc[0] = std::max(loc[0], std::abs(pd[i] - ph[i]));
      loc[1] = std::max(loc[1], std::abs(ph[i]));
    }
    PetscCall(VecRestoreArrayRead(yd, &pd));
    PetscCall(VecRestoreArrayRead(yh, &ph));
    PetscCallMPI(MPI_Allreduce(loc, glo, 2, MPI_DOUBLE, MPI_MAX, comm));
    *rel = glo[0] / std::max(glo[1], 1e-300);
    PetscCall(VecDestroy(&xh));
    PetscCall(VecDestroy(&yh));
    PetscCall(VecDestroy(&xd));
    PetscCall(VecDestroy(&yd));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* C's values = T's, on the device: T a fresh product of the same operands as
 * C (the Galerkin chain's -sf_gpu_ptap 2, mod_petsc_pc_gmg), so the same
 * pattern -- checked here, row pointers and columns of each part and the
 * off-diagonal part's garray; *ok = 0 (nothing copied) where they differ.
 * Both (MPI or sequential) aijkokkos. */
PetscErrorCode jorek_dev_copy_values(Mat T, Mat C, int *ok)
{
  Mat             tp[2] = {T, nullptr}, cp[2] = {C, nullptr};
  const PetscInt *tg = nullptr, *cg = nullptr;
  PetscBool       mpi;

  PetscFunctionBeginUser;
  *ok = 1;
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATMPIAIJKOKKOS, &mpi));
  if (mpi) {
    PetscInt nt, nc;
    PetscCall(MatMPIAIJGetSeqAIJ(T, &tp[0], &tp[1], &tg));
    PetscCall(MatMPIAIJGetSeqAIJ(C, &cp[0], &cp[1], &cg));
    PetscCall(MatGetSize(tp[1], NULL, &nt));
    PetscCall(MatGetSize(cp[1], NULL, &nc));
    if (nt != nc) *ok = 0;
    for (PetscInt i = 0; *ok && i < nc; i++)
      if (tg[i] != cg[i]) *ok = 0;
  }
  double *ta[2] = {nullptr, nullptr}, *ca[2] = {nullptr, nullptr};
  long long n[2] = {0, 0};
  for (int q = 0; q < (mpi ? 2 : 1) && *ok; q++) {
    const PetscInt *ti, *tj, *ci, *cj;
    PetscMemType    mt;
    PetscInt        mtr, mcr;
    long long       ntz, ncz;
    PetscCall(MatGetLocalSize(tp[q], &mtr, NULL));
    PetscCall(MatGetLocalSize(cp[q], &mcr, NULL));
    PetscCall(dev_values(tp[q], &ta[q], &ntz));
    PetscCall(dev_values(cp[q], &ca[q], &ncz));
    if (mtr != mcr || ntz != ncz) {
      *ok = 0;
      break;
    }
    PetscCall(MatSeqAIJGetCSRAndMemType(tp[q], &ti, &tj, &ta[q], &mt));
    PetscCall(MatSeqAIJGetCSRAndMemType(cp[q], &ci, &cj, &ca[q], &mt));
    long long bad = 0;
    const long long nr = mtr;
    PetscCallCXX(Kokkos::parallel_reduce(
      "jorek_copy_check", Range(PetscGetKokkosExecutionSpace(), 0, ntz + nr + 1),
      KOKKOS_LAMBDA(const long long e, long long &b) {
        if (e < ntz) b += (tj[e] != cj[e]);
        else b += (ti[e - ntz] != ci[e - ntz]);
      },
      bad));
    if (bad) *ok = 0;
    n[q] = ntz;
  }
  int g = *ok;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_INT, MPI_MIN, PetscObjectComm((PetscObject)C)));
  *ok = g;
  if (!*ok) PetscFunctionReturn(PETSC_SUCCESS);
  for (int q = 0; q < (mpi ? 2 : 1); q++) {
    if (!n[q]) continue;
    Kokkos::View<double *, MemSpace, Kokkos::MemoryUnmanaged> dt(ta[q], (size_t)n[q]), dc(ca[q], (size_t)n[q]);
    PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), dc, dt));
    PetscCall(mark_device(cp[q]));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* the shell is not used (a failed gate): forget it before its owner destroys it */
PetscErrorCode jorek_sfg_bmv_drop(void)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&bmv.xg));
  PetscCall(VecScatterDestroy(&bmv.sc));
  bmv = Bmv();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* the device memory of this file, ahead of PetscFinalize (mod_petsc) */
PetscErrorCode jorek_sfg_release(void)
{
  PetscFunctionBeginUser;
  PetscCall(release());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 1 when the device gathers can run here (a Kokkos build) */
int jorek_sfg_kokkos(void)
{
  return 1;
}

/* The device's memory in use and its total, in bytes (the memory audit,
 * physics_pc_mem); -1 where the backend gives no figure */
void jorek_dev_mem(double *used, double *total)
{
  *used = *total = -1.0;
#if defined(KOKKOS_ENABLE_CUDA)
  size_t fr, tot;
  if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) {
    *used  = (double)(tot - fr);
    *total = (double)tot;
  }
#elif defined(KOKKOS_ENABLE_HIP)
  size_t fr, tot;
  if (hipMemGetInfo(&fr, &tot) == hipSuccess) {
    *used  = (double)(tot - fr);
    *total = (double)tot;
  }
#endif
}

} // extern "C"
#else
extern "C" int jorek_sfg_kokkos(void)
{
  return 0;
}
extern "C" void jorek_dev_mem(double *used, double *total)
{
  *used = *total = -1.0;
}
#endif
