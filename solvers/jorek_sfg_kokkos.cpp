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
#include <map>
#include <string>
#include <vector>
#if defined(KOKKOS_ENABLE_CUDA)
  #include <cuda_runtime.h>
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
PetscLogEvent           ev_up = -1, ev_ga = -1;

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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode events()
{
  PetscFunctionBeginUser;
  if (ev_up < 0) {
    PetscClassId cid;
    PetscCall(PetscClassIdRegister("JOREK SF gather", &cid));
    PetscCall(PetscLogEventRegister("SFG_DevUpload", cid, &ev_up));
    PetscCall(PetscLogEventRegister("SFG_DevGather", cid, &ev_ga));
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

} // extern "C"
#else
extern "C" int jorek_sfg_kokkos(void)
{
  return 0;
}
#endif
