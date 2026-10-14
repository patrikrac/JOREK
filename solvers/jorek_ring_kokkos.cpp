/* Device ring solver of the C1 GMG ring smoother (mod_petsc_pc_gmg, smoother
 * 4 on the device path).
 *
 * A ring block is a periodic block-tridiagonal chain of nn node blocks
 * (mod_gmg_ring_btd). Its host factorisation is the Thomas / reduction
 * hybrid: ns separator nodes cut the ring into segments, block Thomas
 * eliminates every segment's interior with fill towards its two separators,
 * and the separators' periodic chain is solved by block cyclic reduction
 * (sepm 1, ns >= 4 a power of 2) or dense (ns <= 3). btd_dev_pack hands this
 * file the factors with every pivot block as an explicit inverse, so a solve
 * is matrix-vector products only:
 *   segments, forward:  z_J = Dinv_J (x_J - L_J z_J-1),  x_a -= R_J z_J
 *   separators:         x_s -= L_s z_s-1, then the cyclic reduction (down:
 *                       x_k -= A_k x_k-h + B_k x_k+h; top pair; up:
 *                       x_k = Dinv_k (x_k - L_k x_k-h - U_k x_k+h))
 *   segments, backward: x_J = z_J - H_J x_a - G_J x_J+1
 * One team per ring, the segments (or a level's nodes) on its threads, a
 * node block's rows on a thread's vector lanes (column-major blocks: the
 * lanes read consecutive entries). The dependency chain is ~2 m + 2 log2 ns
 * products for m interior nodes per segment (-sf_gpu_ring_m), in one launch
 * per smoothing step, against nn sequential steps for the band solve.
 *
 * Zebra smoothers (7, 8): each ring / line has a colour (header slot 6), and
 * blk_apply calls this solver once per colour, after blk_dev's blocks of that
 * colour. Colour 1 solves x - Z y with the coupling Z to the colour-0
 * solution, read with that solution's ghost rows from the blk_dev handle
 * (jorek_blk_kokkos_zebra); colour 0 leaves its ghost rows' solution there.
 *
 * Single precision (-sf_gpu_ring_single, default): the products read half
 * the bytes; a smoother needs no more (the host's ring_single likewise).
 * Kokkos only: the backend is PETSc's (CUDA, HIP, SYCL, OpenMP).
 */
#if defined(USE_PETSC) && defined(USE_GPU_PC)
#include <petscvec_kokkos.hpp>
#include <petscmat.h>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "jorek_dense_batched.hpp"

extern "C" {
void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
}

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using IntView   = Kokkos::View<int *, MemSpace>;
using LongView  = Kokkos::View<long long *, MemSpace>;
using CRealView = Kokkos::View<const PetscScalar *, MemSpace>;
using RealView  = Kokkos::View<PetscScalar *, MemSpace>;
using Policy    = Kokkos::TeamPolicy<ExecSpace>;
using Team      = Policy::member_type;

constexpr int NH = 10, NH8 = 3; /* header ints / int64s per ring (ring_dev_upload) */

/* ---- device factorisation: a plan of batched dense operations ----
 * One op = one batch of equal-shape, independent block operations on the
 * work buffer fw (the rings' factor storage, then temporaries): GEMM
 * C = alpha A B + beta C, INV C = A^-1 (via the item's temporary T), COPY
 * C = alpha A, ADD C += A, ZERO C = 0, blocks column-major with leading
 * dimensions lda / ldb / ldc. Ops run in order; inside one op the items are
 * independent (distinct C). CUDA: cuBLAS batched routines (jorek_dense);
 * else threaded LAPACK / BLAS on the host-resident buffer. */
enum { OP_GEMM = 0, OP_INV, OP_COPY, OP_ADD, OP_ZERO };

struct FacOp {
  int                    kind = 0, m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0;
  double                 alpha = 1.0, beta = 0.0;
  std::vector<long long> a, b, c, t; /* offsets into fw; t: INV temporaries */
  std::vector<int>       ring;       /* the item's ring (INV: failure flags) */
  LongView               da, db, dc, dt;
  IntView                dring;
  jorek_dense::PtrView   pa, pb, pc; /* cuBLAS: the items' pointers into fw */
};

struct FacPlan {
  std::vector<FacOp>     ops;
  long long              nfac = 0, nw = 0; /* fw = factors (nfac) + temporaries (nw) */
  bool                   ok   = true;      /* the layout has a device factorisation */
  const double          *base = nullptr;   /* the fw the ops' pointers point into */
  IntView                info;             /* INV: per item, sized for the largest op */
#if defined(KOKKOS_ENABLE_CUDA)
  jorek_dense::BatchWork bw;
#endif
};

struct RingDev {
  int                         nr = 0, nrow = 0, ngh = 0, nr_tot = 0;
  int                         maxns = 1, maxbs = 1, vlen = 1, tsz = 1;
  bool                        sp = true;
  IntView                     hdr, ni, si, rows, jac;
  IntView                     ract[3];             /* rings of colour 0, 1, and all */
  int                         nrc[3] = {0, 0, 0};
  LongView                    hdr8, n8, s8, l8;
  Kokkos::View<double *, MemSpace> f8, w8;
  Kokkos::View<float *, MemSpace>  f4, w4;
  double                      mb = 0.0;
  /* device factorisation (jorek_ring_kokkos_factor): the layout's host copy,
   * per separator its segment's Schur blocks S_ab, S_ba, S_bb (g8) */
  std::vector<int>            h_hdr, h_ni, h_si;
  std::vector<long long>      h_hdr8, h_n8, h_s8, h_l8, h_g8;
  LongView                    g8;
  IntView                     fail;
  std::shared_ptr<FacPlan>    plan;
  Kokkos::View<double *, MemSpace> fw; /* factorisation buffer: factors + temporaries */
  Kokkos::View<double *, MemSpace> vg; /* the ghost rows' values (device fill) */
  /* the rings' values straight from the level matrix (jorek_ring_kokkos_fillmap):
   * fm_src +k / -k = entry k (1-based) of its diagonal / off-diagonal part's
   * values, fg_src = entry k of the ghost rows' values; *_dst = index in fw */
  IntView                     fm_src, fg_src;
  LongView                    fm_dst, fg_dst;
  int                         tteam[2][3] = {{0, 0, 0}, {0, 0, 0}}; /* ring_apply's team size per precision (double, single) and colour list; 0: not set */
};

PetscLogEvent ev_ring = -1, ev_rfac = -1;

/* the zebra coupling of colour 1 to colour 0 (blk_dev's), device pointers */
struct Zebra {
  const int         *zp = nullptr, *zc = nullptr; /* zp: 1-based row pointers */
  const PetscScalar *zv = nullptr;
  PetscScalar       *yg = nullptr;                /* colour 0 on the ghost rows */
};

template <class V, class T>
void upload(V &dst, const char *name, const T *src, size_t n)
{
  if (dst.extent(0) != n) dst = V(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), n);
  if (!n) return;
  Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(src, n);
  Kokkos::deep_copy(dst, h);
}

/* y = (acc ? y : 0) + alpha A x on the thread's vector lanes, A m x n column-major */
template <class T>
KOKKOS_INLINE_FUNCTION void gemv(const Team &team, int m, int n, const T *A, const T *x, T *y, T alpha, bool acc)
{
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, m), [=](const int i) {
    T s = 0;
    for (int j = 0; j < n; j++) s += A[i + (size_t)j * m] * x[j];
    y[i] = acc ? y[i] + alpha * s : alpha * s;
  });
}

template <class T>
PetscErrorCode ring_apply(RingDev *R, Kokkos::View<const T *, MemSpace> fac, Kokkos::View<T *, MemSpace> w, CRealView xv, CRealView gv, RealView yv, int c, Zebra zb)
{
  const int  li = (c < 0) ? 2 : c;
  auto       list = R->ract[li];
  auto       hdr = R->hdr, ni = R->ni, si = R->si, rows = R->rows, jac = R->jac;
  const int *zp = zb.zp, *zc = zb.zc;
  const PetscScalar *zv = zb.zv;
  PetscScalar       *yg = zb.yg;
  auto       hdr8 = R->hdr8, n8 = R->n8, s8 = R->s8, l8 = R->l8;
  const int  nrow = R->nrow, mbs = R->maxbs;
  const int  nslot = std::max(R->maxns, 2);
  const size_t bytes = Kokkos::View<T *, typename ExecSpace::scratch_memory_space>::shmem_size((size_t)nslot * mbs);
  const int  lvl   = bytes <= 40000 ? 0 : 1;
  using Scr  = Kokkos::View<T *, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;

  PetscFunctionBeginUser;
  if (!R->nrc[li]) PetscFunctionReturn(PETSC_SUCCESS);
  auto body = KOKKOS_LAMBDA(const Team &team) {
      const int  r = list(team.league_rank());
      const int *H = &hdr(NH * r);
      const int  roff = H[0], sz = H[1], nn = H[2], ns = H[3], nlev = H[4], nf = H[5], pn = H[7], ps = H[8], pl = H[9];
      const T   *a  = &fac(hdr8(NH8 * r));
      const long long of = hdr8(NH8 * r + 1), bt = hdr8(NH8 * r + 2);
      T         *x  = &w(roff);
      Scr        tmp(team.team_scratch(lvl), (size_t)nslot * mbs);
      /* node / separator layout */
      auto bs  = [=](int J) { return ni(2 * (pn + J)); };
      auto rp  = [=](int J) { return ni(2 * (pn + J) + 1); };
      auto o8  = [=](int J, int c) { return n8(5 * (pn + J) + c); }; /* od ou ol oc orr */
      auto sep = [=](int k) { return si(2 * (ps + k)); };
      auto hend = [=](int k) { return k == ns - 1 ? nn - 1 : sep(k + 1) - 1; };

      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, sz), [=](const int q) {
        const int   rr = rows(roff + q);
        PetscScalar t  = (rr < nrow) ? xv(rr) : gv(rr - nrow);
        if (c == 1 && zp) /* minus the coupling to the colour-0 solution */
          for (int k = zp[rr] - 1; k < zp[rr + 1] - 1; k++) {
            const int cc = zc[k];
            t -= zv[k] * ((cc < nrow) ? yv(cc) : yg[cc - nrow]);
          }
        x[q] = (T)t;
      });
      team.team_barrier();

      if (jac(r)) { /* singular node block: point Jacobi on the diagonals */
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nn), [=](const int J) {
          const int b = bs(J);
          T        *xj = x + rp(J);
          const T  *d  = a + o8(J, 0);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, b), [=](const int i) { xj[i] /= d[i + (size_t)i * b]; });
        });
      } else {
        const int mmax = (nn + ns - 1) / ns - 1;
        /* segments, forward: step t's tmp after step t-1's separator update */
        for (int t = 1; t <= mmax + 1; t++) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns), [=](const int k) {
            const int s = sep(k), m = hend(k) - s;
            T        *tk = &tmp(k * mbs);
            if (t > 1 && t - 1 <= m) { /* x_a -= R_J z_J of step t-1 */
              const int J = s + t - 1;
              gemv(team, bs(s), bs(J), a + o8(J, 4), x + rp(J), x + rp(s), (T)-1, true);
            }
            if (t <= m) {
              const int J = s + t, b = bs(J);
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, b), [=](const int i) { tk[i] = x[rp(J) + i]; });
              if (t > 1) gemv(team, b, bs(J - 1), a + o8(J, 2), x + rp(J - 1), tk, (T)-1, true);
            }
          });
          team.team_barrier();
          if (t > mmax) break;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns), [=](const int k) {
            const int s = sep(k);
            if (t <= hend(k) - s) {
              const int J = s + t;
              gemv(team, bs(J), bs(J), a + o8(J, 0), &tmp(k * mbs), x + rp(J), (T)1, false);
            }
          });
          team.team_barrier();
        }
        /* separators: x_s -= L_s z_s-1 */
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns), [=](const int k) {
          const int s = sep(k), sp_ = (s + nn - 1) % nn;
          gemv(team, bs(s), bs(sp_), a + o8(s, 2), x + rp(sp_), x + rp(s), (T)-1, true);
        });
        team.team_barrier();
        if (nlev > 0) { /* block cyclic reduction, in place on the separators' rows */
          auto xs  = [=](int k) { return x + rp(sep(k)); };
          auto bk  = [=](int k) { return bs(sep(k)); };
          int  h   = 1;
          for (int l = 0; l < nlev; l++, h *= 2) {
            const int hh = h;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns / (2 * hh)), [=](const int i) {
              const int k = 2 * hh * i, km = (k - hh + ns) % ns, kp = (k + hh) % ns;
              gemv(team, bk(k), bk(km), a + l8(2 * (pl + l * ns + k)), xs(km), xs(k), (T)-1, true);
              gemv(team, bk(k), bk(kp), a + l8(2 * (pl + l * ns + k) + 1), xs(kp), xs(k), (T)-1, true);
            });
            team.team_barrier();
          }
          { /* the top pair 0, h */
            const int b0 = bk(0), b1 = bk(h), n2 = b0 + b1;
            T        *x0 = xs(0), *x1 = xs(h);
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, n2), [=](const int i) { tmp(i) = i < b0 ? x0[i] : x1[i - b0]; });
            team.team_barrier();
            const T *ti = a + bt;
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team, n2), [=](const int i) {
              T s = 0;
              for (int j = 0; j < n2; j++) s += ti[i + (size_t)j * n2] * tmp(j);
              if (i < b0) x0[i] = s;
              else x1[i - b0] = s;
            });
            team.team_barrier();
          }
          for (int l = nlev - 1; l >= 0; l--) {
            const int hh = 1 << l;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns / (2 * hh)), [=](const int i) {
              const int k = hh + 2 * hh * i, km = (k - hh + ns) % ns, kp = (k + hh) % ns, b = bk(k);
              const long long *o = &s8(3 * (ps + k));
              T               *ti = &tmp(i * mbs);
              const T         *xk = xs(k);
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, b), [=](const int q) { ti[q] = xk[q]; });
              gemv(team, b, bk(km), a + o[1], xs(km), ti, (T)-1, true);
              gemv(team, b, bk(kp), a + o[2], xs(kp), ti, (T)-1, true);
            });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns / (2 * hh)), [=](const int i) {
              const int k = hh + 2 * hh * i, b = bk(k);
              gemv(team, b, b, a + s8(3 * (ps + k)), &tmp(i * mbs), xs(k), (T)1, false);
            });
            team.team_barrier();
          }
        } else { /* ns <= 3: the dense separators' inverse */
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns), [=](const int k) {
            int q0 = 0;
            for (int kk = 0; kk < k; kk++) q0 += bs(sep(kk));
            const T *xk = x + rp(sep(k));
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, bs(sep(k))), [=](const int i) { tmp(q0 + i) = xk[i]; });
          });
          team.team_barrier();
          const T *si_ = a + of;
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nf), [=](const int i) {
            T s = 0;
            for (int j = 0; j < nf; j++) s += si_[i + (size_t)j * nf] * tmp(j);
            int k = 0, q = i;
            while (q >= bs(sep(k))) q -= bs(sep(k++));
            x[rp(sep(k)) + q] = s;
          });
          team.team_barrier();
        }
        /* segments, backward: x_J = z_J - H_J x_a - G_J x_J+1 */
        for (int t = mmax; t >= 1; t--) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ns), [=](const int k) {
            const int s = sep(k);
            if (t <= hend(k) - s) {
              const int J = s + t, jn = (J + 1) % nn;
              gemv(team, bs(J), bs(s), a + o8(J, 3), x + rp(s), x + rp(J), (T)-1, true);
              gemv(team, bs(J), bs(jn), a + o8(J, 1), x + rp(jn), x + rp(J), (T)-1, true);
            }
          });
          team.team_barrier();
        }
      }
      team.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, sz), [=](const int q) {
        const int rr = rows(roff + q);
        if (rr < nrow) yv(rr) = (PetscScalar)x[q];
        else if (c == 0 && yg) yg[rr - nrow] = (PetscScalar)x[q];
      });
    };
  /* threads per team: the segments, within what the kernel's registers allow
   * (the segment loops stride over the threads); asked once per layout */
  int &ts = R->tteam[sizeof(T) == sizeof(float) ? 1 : 0][li];
  if (!ts) {
    Policy p1(R->nrc[li], 1, R->vlen);
    p1.set_scratch_size(lvl, Kokkos::PerTeam(bytes));
    ts = std::max(1, std::min(R->tsz, p1.team_size_max(body, Kokkos::ParallelForTag())));
  }
  Policy pol(R->nrc[li], ts, R->vlen);
  pol.set_scratch_size(lvl, Kokkos::PerTeam(bytes));
  PetscCallCXX(Kokkos::parallel_for("jorek_ring_solve", pol, body));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* ---- the factorisation plan (host, once per layout) ---- */
struct Planner {
  FacPlan                                                           &P;
  std::map<std::tuple<int, int, int, int, int, int, int, double, double>, size_t> open; /* this stage's ops by shape */
  long long                                                          wtop = 0;          /* temporaries in use */
  explicit Planner(FacPlan &p) : P(p) { }
  /* a new stage: what follows may read what came before */
  void stage() { open.clear(); }
  long long tmp(long long n)
  {
    const long long o = P.nfac + wtop;
    wtop += n;
    P.nw = std::max(P.nw, wtop);
    return o;
  }
  void reset() { wtop = 0; }
  void add(int kind, int m, int n, int k, int lda, int ldb, int ldc, double al, double be, long long a, long long b, long long c, int r)
  {
    if (m <= 0 || n <= 0) return;
    auto key = std::make_tuple(kind, m, n, k, lda, ldb, ldc, al, be);
    auto it  = open.find(key);
    size_t q;
    if (it == open.end()) {
      FacOp op;
      op.kind = kind; op.m = m; op.n = n; op.k = k; op.lda = lda; op.ldb = ldb; op.ldc = ldc; op.alpha = al; op.beta = be;
      P.ops.push_back(op);
      q         = P.ops.size() - 1;
      open[key] = q;
    } else q = it->second;
    FacOp &op = P.ops[q];
    op.a.push_back(a); op.b.push_back(b); op.c.push_back(c); op.ring.push_back(r);
    op.t.push_back(kind == OP_INV ? tmp((long long)m * m) : 0);
  }
  /* C (m x n) = alpha A B + beta C */
  void gemm(int m, int n, int k, double al, long long a, long long b, double be, long long c, int r) { add(OP_GEMM, m, n, k, m, k, m, al, be, a, b, c, r); }
  void inv(int n, long long a, long long c, int r) { add(OP_INV, n, n, n, n, n, n, 1.0, 0.0, a, 0, c, r); }
  void copy(int m, int n, long long a, int lda, long long c, int ldc, double al, int r) { add(OP_COPY, m, n, 0, lda, 0, ldc, al, 0.0, a, 0, c, r); }
  void addto(int m, int n, long long a, int lda, long long c, int ldc, int r) { add(OP_ADD, m, n, 0, lda, 0, ldc, 1.0, 1.0, a, 0, c, r); }
  void zero(int m, int n, long long c, int ldc, int r) { add(OP_ZERO, m, n, 0, 0, 0, ldc, 0.0, 0.0, 0, 0, c, r); }
};

/* The plan of the rings' hybrid factorisation into the solve's device form
 * (btd_dev_pack): per segment block Thomas with explicit inverses of the
 * interior pivots, the segments' Schur blocks; then per ring the separators'
 * chain by cyclic reduction (nlev > 0, equal separator blocks), or dense
 * (ns <= 3) and inverted. Offsets as hdr8 / n8 / s8 / l8 / g8. */
void plan_build(RingDev *R, FacPlan &P)
{
  Planner     pl(P);
  const int   nr = R->nr;
  const auto &H = R->h_hdr, &NI = R->h_ni, &SI = R->h_si;
  const auto &H8 = R->h_hdr8, &N8 = R->h_n8, &S8 = R->h_s8, &L8 = R->h_l8, &G8 = R->h_g8;
  /* ring r's header (jorek_ring_kokkos) and offsets in fw */
  auto nn_  = [&](int r) { return H[(size_t)NH * r + 2]; };
  auto ns_  = [&](int r) { return H[(size_t)NH * r + 3]; };
  auto nlv_ = [&](int r) { return H[(size_t)NH * r + 4]; }; /* cyclic-reduction levels, 0: dense separators */
  auto nf_  = [&](int r) { return H[(size_t)NH * r + 5]; };
  auto pn_  = [&](int r) { return H[(size_t)NH * r + 7]; };
  auto ps_  = [&](int r) { return H[(size_t)NH * r + 8]; };
  auto pl_  = [&](int r) { return H[(size_t)NH * r + 9]; };
  auto F    = [&](int r) { return H8[(size_t)NH8 * r]; };
  auto Fo   = [&](int r) { return F(r) + H8[(size_t)NH8 * r + 1]; }; /* the dense separators' inverse */
  auto Tt   = [&](int r) { return F(r) + H8[(size_t)NH8 * r + 2]; }; /* the top pair's inverse */
  auto sep  = [&](int r, int k) { return SI[2 * (ps_(r) + k)]; };
  auto bs   = [&](int r, int J) { return NI[2 * (pn_(r) + J)]; };
  auto o8   = [&](int r, int J, int c) { return F(r) + N8[5 * (pn_(r) + J) + c]; }; /* od ou ol oc orr */
  auto g8   = [&](int r, int k, int c) { return F(r) + G8[3 * (ps_(r) + k) + c]; }; /* sab sba sbb */
  enum { OD = 0, OU, OL, OC, ORR };
  struct Seg { int r, k, s, b, e; };
  std::vector<Seg> sg;
  int              mmax = 0, lmax = 0;
  P.ok = true;
  for (int r = 0; r < nr; r++) {
    const int nn = nn_(r), ns = ns_(r), nlev = nlv_(r);
    if (ns >= 4 && nlev == 0) P.ok = false; /* a sub-Thomas separator system: no device form */
    for (int k = 0; k < ns; k++) {
      const int s = sep(r, k), b = sep(r, (k + 1) % ns), e = (k == ns - 1) ? nn - 1 : sep(r, k + 1) - 1;
      sg.push_back({r, k, s, b, e});
      mmax = std::max(mmax, e - s);
    }
    if (nlev > 0) lmax = std::max(lmax, nlev);
  }
  if (!P.ok) return;

  /* segments: the couplings to the left separator a: C_a+1 = L_a+1, R_a+1 = U_a */
  pl.stage();
  for (auto &q : sg) {
    const int ba = bs(q.r, q.s), b1 = bs(q.r, q.s + 1);
    pl.copy(b1, ba, o8(q.r, q.s + 1, OL), b1, o8(q.r, q.s + 1, OC), b1, 1.0, q.r);
    pl.copy(ba, b1, o8(q.r, q.s, OU), ba, o8(q.r, q.s + 1, ORR), ba, 1.0, q.r);
  }
  for (int t = 1; t <= mmax; t++) {
    pl.reset();
    std::vector<long long> w1(sg.size()), w2(sg.size());
    if (t > 1) { /* D_J -= L_J G_J-1, C_J = -L_J H_J-1, R_J = -R_J-1 G_J-1 */
      pl.stage();
      for (auto &q : sg) {
        const int J = q.s + t;
        if (J > q.e) continue;
        const int bj = bs(q.r, J), bp = bs(q.r, J - 1), ba = bs(q.r, q.s);
        pl.gemm(bj, bj, bp, -1.0, o8(q.r, J, OL), o8(q.r, J - 1, OU), 1.0, o8(q.r, J, OD), q.r);
        pl.gemm(bj, ba, bp, -1.0, o8(q.r, J, OL), o8(q.r, J - 1, OC), 0.0, o8(q.r, J, OC), q.r);
        pl.gemm(ba, bj, bp, -1.0, o8(q.r, J - 1, ORR), o8(q.r, J - 1, OU), 0.0, o8(q.r, J, ORR), q.r);
      }
    }
    pl.stage(); /* D~_J^-1, in place */
    for (auto &q : sg) {
      const int J = q.s + t;
      if (J > q.e) continue;
      pl.inv(bs(q.r, J), o8(q.r, J, OD), o8(q.r, J, OD), q.r);
    }
    pl.stage(); /* H_J = Dinv C_J, G_J = Dinv U_J, through temporaries */
    for (size_t i = 0; i < sg.size(); i++) {
      auto     &q = sg[i];
      const int J = q.s + t;
      if (J > q.e) continue;
      const int bj = bs(q.r, J), ba = bs(q.r, q.s), bn = bs(q.r, (J + 1) % nn_(q.r));
      w1[i] = pl.tmp((long long)bj * ba);
      w2[i] = pl.tmp((long long)bj * bn);
      pl.gemm(bj, ba, bj, 1.0, o8(q.r, J, OD), o8(q.r, J, OC), 0.0, w1[i], q.r);
      pl.gemm(bj, bn, bj, 1.0, o8(q.r, J, OD), o8(q.r, J, OU), 0.0, w2[i], q.r);
    }
    pl.stage();
    for (size_t i = 0; i < sg.size(); i++) {
      auto     &q = sg[i];
      const int J = q.s + t;
      if (J > q.e) continue;
      const int bj = bs(q.r, J), ba = bs(q.r, q.s), bn = bs(q.r, (J + 1) % nn_(q.r));
      pl.copy(bj, ba, w1[i], bj, o8(q.r, J, OC), bj, 1.0, q.r);
      pl.copy(bj, bn, w2[i], bj, o8(q.r, J, OU), bj, 1.0, q.r);
    }
    pl.stage(); /* D_a -= R~_J H_J (each segment its own separator) */
    for (auto &q : sg) {
      const int J = q.s + t;
      if (J > q.e) continue;
      const int bj = bs(q.r, J), ba = bs(q.r, q.s);
      pl.gemm(ba, ba, bj, -1.0, o8(q.r, J, ORR), o8(q.r, J, OC), 1.0, o8(q.r, q.s, OD), q.r);
    }
  }
  pl.reset();
  pl.stage(); /* the segments' Schur blocks: S_ab = -R~_e G_e, S_ba = -L_b H_e, S_bb = -L_b G_e */
  for (auto &q : sg) {
    const int ba = bs(q.r, q.s), bb = bs(q.r, q.b), be = bs(q.r, q.e);
    pl.gemm(ba, bb, be, -1.0, o8(q.r, q.e, ORR), o8(q.r, q.e, OU), 0.0, g8(q.r, q.k, 0), q.r);
    pl.gemm(bb, ba, be, -1.0, o8(q.r, q.b, OL), o8(q.r, q.e, OC), 0.0, g8(q.r, q.k, 1), q.r);
    pl.gemm(bb, bb, be, -1.0, o8(q.r, q.b, OL), o8(q.r, q.e, OU), 0.0, g8(q.r, q.k, 2), q.r);
  }

  /* separators. Cyclic reduction (nlev > 0): node k's D in od(sep k) (+ S_bb
   * of segment k-1), U in S_ab(k), L in S_ba(k-1) */
  pl.stage();
  for (int r = 0; r < nr; r++) {
    const int ns = ns_(r);
    if (nlv_(r) == 0) continue;
    for (int k = 0; k < ns; k++) {
      const int b = bs(r, sep(r, k));
      pl.addto(b, b, g8(r, (k + ns - 1) % ns, 2), b, o8(r, sep(r, k), OD), b, r);
    }
  }
  /* the chain's current blocks per (ring, separator): D at od(sep k), L / U
   * first in S_ba(k-1) / S_ab(k), later in temporaries sized for the
   * couplings to k -+ 2h (separator blocks may differ in size) */
  auto sD = [&](int r, int k) { return o8(r, sep(r, k), OD); };
  auto sb = [&](int r, int k) { return bs(r, sep(r, k)); };
  auto sx = [&](int r, int k, int c) { return F(r) + S8[3 * (ps_(r) + k) + c]; };                      /* bd bl bu */
  auto lx = [&](int r, int l, int k, int c) { return F(r) + L8[2 * (pl_(r) + l * ns_(r) + k) + c]; }; /* ba bb */
  std::vector<std::vector<long long>> cL(nr), cU(nr);
  for (int r = 0; r < nr; r++) {
    const int ns = ns_(r);
    if (nlv_(r) == 0) continue;
    cL[r].resize(ns); cU[r].resize(ns);
    for (int k = 0; k < ns; k++) {
      cL[r][k] = g8(r, (k + ns - 1) % ns, 1);
      cU[r][k] = g8(r, k, 0);
    }
  }
  pl.reset(); /* the chain's temporaries live to the top pair */
  for (int l = 0, h = 1; l < lmax; l++, h *= 2) {
    pl.stage(); /* the eliminated nodes: Dinv, and their couplings kept */
    for (int r = 0; r < nr; r++) {
      const int ns = ns_(r);
      if (nlv_(r) <= l) continue;
      for (int k = h; k < ns; k += 2 * h) {
        const int b = sb(r, k), bm = sb(r, (k - h + ns) % ns), bp = sb(r, (k + h) % ns);
        pl.inv(b, sD(r, k), sx(r, k, 0), r);
        pl.copy(b, bm, cL[r][k], b, sx(r, k, 1), b, 1.0, r);
        pl.copy(b, bp, cU[r][k], b, sx(r, k, 2), b, 1.0, r);
      }
    }
    struct Kept { int r, k; long long nl, nu; };
    std::vector<Kept> kept;
    pl.stage(); /* A = L_k Dinv_k-h, B = U_k Dinv_k+h */
    for (int r = 0; r < nr; r++) {
      const int ns = ns_(r);
      if (nlv_(r) <= l) continue;
      for (int k = 0; k < ns; k += 2 * h) {
        const int km = (k - h + ns) % ns, kp = (k + h) % ns;
        const int b = sb(r, k), bm = sb(r, km), bp = sb(r, kp);
        const int bmm = sb(r, (k - 2 * h + 2 * ns) % ns), bpp = sb(r, (k + 2 * h) % ns);
        pl.gemm(b, bm, bm, 1.0, cL[r][k], sx(r, km, 0), 0.0, lx(r, l, k, 0), r);
        pl.gemm(b, bp, bp, 1.0, cU[r][k], sx(r, kp, 0), 0.0, lx(r, l, k, 1), r);
        kept.push_back({r, k, pl.tmp((long long)b * bmm), pl.tmp((long long)b * bpp)});
      }
    }
    pl.stage(); /* D_k -= A U_k-h; new L_k = -A L_k-h, U_k = -B U_k+h */
    for (auto &q : kept) {
      const int ns = ns_(q.r), km = (q.k - h + ns) % ns, kp = (q.k + h) % ns;
      const int b = sb(q.r, q.k), bm = sb(q.r, km), bp = sb(q.r, kp);
      const int bmm = sb(q.r, (q.k - 2 * h + 2 * ns) % ns), bpp = sb(q.r, (q.k + 2 * h) % ns);
      pl.gemm(b, b, bm, -1.0, lx(q.r, l, q.k, 0), sx(q.r, km, 2), 1.0, sD(q.r, q.k), q.r);
      pl.gemm(b, bmm, bm, -1.0, lx(q.r, l, q.k, 0), sx(q.r, km, 1), 0.0, q.nl, q.r);
      pl.gemm(b, bpp, bp, -1.0, lx(q.r, l, q.k, 1), sx(q.r, kp, 2), 0.0, q.nu, q.r);
    }
    pl.stage(); /* D_k -= B L_k+h */
    for (auto &q : kept) {
      const int ns = ns_(q.r), kp = (q.k + h) % ns;
      pl.gemm(sb(q.r, q.k), sb(q.r, q.k), sb(q.r, kp), -1.0, lx(q.r, l, q.k, 1), sx(q.r, kp, 1), 1.0, sD(q.r, q.k), q.r);
    }
    for (auto &q : kept) {
      cL[q.r][q.k] = q.nl;
      cU[q.r][q.k] = q.nu;
    }
  }
  /* the top pair 0, ns/2: [D_0, L_0 + U_0; L_h + U_h, D_h]^-1 at bt */
  pl.stage();
  for (int r = 0; r < nr; r++) {
    const int ns = ns_(r);
    if (nlv_(r) == 0) continue;
    const int       h = ns / 2, b0 = sb(r, 0), b1 = sb(r, h), n2 = b0 + b1;
    const long long T = Tt(r);
    pl.copy(b0, b0, sD(r, 0), b0, T, n2, 1.0, r);
    pl.copy(b0, b1, cL[r][0], b0, T + (long long)b0 * n2, n2, 1.0, r);
    pl.copy(b1, b0, cL[r][h], b1, T + b0, n2, 1.0, r);
    pl.copy(b1, b1, sD(r, h), b1, T + b0 + (long long)b0 * n2, n2, 1.0, r);
  }
  pl.stage();
  for (int r = 0; r < nr; r++) {
    const int ns = ns_(r);
    if (nlv_(r) == 0) continue;
    const int       h = ns / 2, b0 = sb(r, 0), b1 = sb(r, h), n2 = b0 + b1;
    const long long T = Tt(r);
    pl.addto(b0, b1, cU[r][0], b0, T + (long long)b0 * n2, n2, r);
    pl.addto(b1, b0, cU[r][h], b1, T + b0, n2, r);
  }
  pl.stage();
  for (int r = 0; r < nr; r++) {
    if (nlv_(r) == 0) continue;
    const int       ns = ns_(r);
    const long long T  = Tt(r);
    pl.inv(sb(r, 0) + sb(r, ns / 2), T, T, r);
  }
  pl.reset();

  /* dense separators (ns <= 3): F (nf x nf, at of) = the chain, inverted */
  std::vector<int> dn;
  for (int r = 0; r < nr; r++)
    if (nlv_(r) == 0) dn.push_back(r);
  if (dn.empty()) return;
  auto co = [&](int r, int k) { int q = 0; for (int i = 0; i < k; i++) q += bs(r, sep(r, i)); return q; }; /* separator k's first row in F */
  pl.stage();
  for (int r : dn) pl.zero(nf_(r), nf_(r), Fo(r), nf_(r), r);
  for (int cat = 0; cat < 4; cat++) {
    pl.stage();
    for (int r : dn) {
      const int       ns = ns_(r), nf = nf_(r);
      const long long f0 = Fo(r);
      for (int k = 0; k < ns; k++) {
        const int kp = (k + ns - 1) % ns, kn = (k + 1) % ns;
        const int bk = bs(r, sep(r, k)), bn = bs(r, sep(r, kn));
        auto      at = [&](int kr, int kc) { return f0 + co(r, kr) + (long long)co(r, kc) * nf; };
        if (cat == 0) pl.addto(bk, bk, sD(r, k), bk, at(k, k), nf, r);
        if (cat == 1) pl.addto(bk, bk, g8(r, kp, 2), bk, at(k, k), nf, r);
        if (cat == 2) pl.addto(bk, bn, g8(r, k, 0), bk, at(k, kn), nf, r);
        if (cat == 3) pl.addto(bn, bk, g8(r, k, 1), bn, at(kn, k), nf, r);
      }
    }
  }
  pl.stage();
  for (int r : dn) pl.inv(nf_(r), Fo(r), Fo(r), r);
}

/* ---- the plan's execution ---- */
template <class V>
void up_off(V &dst, const std::vector<long long> &src)
{
  dst = V(Kokkos::view_alloc(std::string("fac_off"), Kokkos::WithoutInitializing), src.size());
  Kokkos::View<const long long *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(src.data(), src.size());
  Kokkos::deep_copy(dst, h);
}

PetscErrorCode plan_run(FacPlan &P, Kokkos::View<double *, MemSpace> fw, IntView fail)
{
  PetscFunctionBeginUser;
  double *base = fw.data();
#if defined(KOKKOS_ENABLE_CUDA)
  const bool repoint = (P.base != base); /* fw (re)allocated: the cuBLAS pointers again */
#endif
  for (auto &op : P.ops) {
    const int nb = (int)op.c.size();
    if (!nb) continue;
    if (op.dc.extent(0) != (size_t)nb) { /* device copies of the offsets, once per plan */
      up_off(op.da, op.a); up_off(op.db, op.b); up_off(op.dc, op.c); up_off(op.dt, op.t);
      op.dring = IntView(Kokkos::view_alloc(std::string("fac_ring"), Kokkos::WithoutInitializing), nb);
      Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hr(op.ring.data(), nb);
      Kokkos::deep_copy(op.dring, hr);
    }
    auto         da = op.da, db = op.db, dc = op.dc, dt = op.dt;
    auto         dring = op.dring;
    const int    m = op.m, n = op.n, k = op.k, lda = op.lda, ldb = op.ldb, ldc = op.ldc, kind = op.kind;
    const double al = op.alpha, be = op.beta;
    if (kind == OP_COPY || kind == OP_ADD || kind == OP_ZERO) {
      PetscCallCXX(Kokkos::parallel_for(
        "jorek_ring_fcopy", Kokkos::RangePolicy<ExecSpace>(0, (long long)nb * m * n), KOKKOS_LAMBDA(const long long q) {
          const long long mn = (long long)m * n;
          const int       i = (int)(q / mn), e = (int)(q % mn), r = e % m, c = e / m;
          double         *C = base + dc(i) + r + (long long)c * ldc;
          if (kind == OP_ZERO) *C = 0.0;
          else {
            const double v = al * base[da(i) + r + (long long)c * lda];
            *C             = (kind == OP_ADD) ? *C + v : v;
          }
        }));
      continue;
    }
    if (kind == OP_INV) /* the source into the item's temporary (source and target may coincide) */
      PetscCallCXX(Kokkos::parallel_for(
        "jorek_ring_fcpyt", Kokkos::RangePolicy<ExecSpace>(0, (long long)nb * n * n), KOKKOS_LAMBDA(const long long q) {
          const long long nn = (long long)n * n;
          const int       i = (int)(q / nn), e = (int)(q % nn), r = e % n, c = e / n;
          base[dt(i) + e] = base[da(i) + r + (long long)c * lda];
        }));
#if defined(KOKKOS_ENABLE_CUDA)
    if (repoint || op.pc.extent(0) != (size_t)nb) { /* the items' pointers: once per plan and buffer */
      op.pa = jorek_dense::PtrView(Kokkos::view_alloc(std::string("fac_pa"), Kokkos::WithoutInitializing), nb);
      op.pb = jorek_dense::PtrView(Kokkos::view_alloc(std::string("fac_pb"), Kokkos::WithoutInitializing), nb);
      op.pc = jorek_dense::PtrView(Kokkos::view_alloc(std::string("fac_pc"), Kokkos::WithoutInitializing), nb);
      auto pa = op.pa, pb = op.pb, pc = op.pc;
      PetscCallCXX(Kokkos::parallel_for(
        "jorek_ring_fptr", Kokkos::RangePolicy<ExecSpace>(0, nb), KOKKOS_LAMBDA(const int i) {
          pa(i) = (uintptr_t)(base + (kind == OP_INV ? dt(i) : da(i)));
          pb(i) = (uintptr_t)(base + db(i));
          pc(i) = (uintptr_t)(base + dc(i));
        }));
    }
    if (kind == OP_GEMM) {
      cublasHandle_t hd;
      PetscCall(jorek_dense::cublas_handle(&hd));
      PetscCheck(cublasDgemmBatched(hd, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, (const double **)op.pa.data(), lda, (const double **)op.pb.data(), ldb, &be, (double **)op.pc.data(), ldc, nb) == CUBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cublasDgemmBatched");
    } else { /* OP_INV: C = T^-1 */
      PetscCallCXX(jorek_dense::grow(P.info, "fac_info", (size_t)nb));
      auto info = P.info;
      PetscCall(jorek_dense::inverse_batched(n, nb, op.pa, op.pc, ldc, info, P.bw));
      PetscCallCXX(Kokkos::parallel_for(
        "jorek_ring_ffail", Kokkos::RangePolicy<ExecSpace>(0, nb), KOKKOS_LAMBDA(const int i) {
          if (info(i) != 0) fail(dring(i)) = 1;
        }));
    }
#else
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible) {
      /* host backends: the same batches by threaded BLAS / LAPACK */
      PetscCallCXX(ExecSpace().fence());
      const auto &ha = op.a, &hb = op.b, &hc = op.c, &ht = op.t;
      int        *fl = fail.data();
  #pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < nb; i++) {
        if (kind == OP_GEMM) {
          dgemm_("N", "N", &m, &n, &k, &al, base + ha[i], &lda, base + hb[i], &ldb, &be, base + hc[i], &ldc);
        } else {
          double *T = base + ht[i];
          if (jorek_dense::inverse_host(n, T)) fl[op.ring[i]] = 1;
          else
            for (int c = 0; c < n; c++)
              for (int r = 0; r < n; r++) base[hc[i] + r + (long long)c * ldc] = T[r + (long long)c * n];
        }
      }
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "device ring factorisation: no batched backend for this Kokkos device");
#endif
  }
  P.base = base;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

extern "C" {

/* Layout of a level's ring-solver blocks (mod_petsc_pc_gmg ring_dev_upload):
 * per ring hdr = row offset, rows, nn, ns, cyclic-reduction levels (0: dense
 * separators), separator rows, -, node / separator / level base, and hdr8 =
 * factor offset, dense separator inverse, top pair inverse; per node ni = bs,
 * rp and n8 = od, ou, ol, oc, orr; per separator si = sep, co and s8 = bd,
 * bl, bu; per (level, separator) l8 = ba, bb. rows: blk_t%rows (0-based,
 * >= nrow: ghost). *h = NULL creates the handle. stats = rings, segments per
 * ring (max), largest node block, device MB (at values). */
PetscErrorCode jorek_ring_kokkos(void **h, int nr, int nrow, int ngh, const int *hdr, const long long *hdr8, const int *ni, const long long *n8, const int *si, const long long *s8, const long long *l8, const int *rows, double *stats)
{
  RingDev *R;
  int      pn = 0, ps = 0, pl = 0, nr_tot = 0;

  PetscFunctionBeginUser;
  if (ev_ring < 0) {
    PetscClassId cid;
    PetscCall(PetscClassIdRegister("JOREK GMG rings", &cid));
    PetscCall(PetscLogEventRegister("RING_Solve", cid, &ev_ring));
    PetscCall(PetscLogEventRegister("RING_Factor", cid, &ev_rfac));
  }
  if (!*h) PetscCallCXX(*h = new RingDev);
  R        = static_cast<RingDev *>(*h);
  R->nr    = nr;
  R->nrow  = nrow;
  R->ngh   = ngh;
  R->maxns = 1;
  R->maxbs = 1;
  for (int r = 0; r < nr; r++) {
    const int *H = hdr + NH * r;
    nr_tot       = std::max(nr_tot, H[0] + H[1]);
    R->maxns     = std::max(R->maxns, H[3]);
    for (int J = 0; J < H[2]; J++) R->maxbs = std::max(R->maxbs, ni[2 * (H[7] + J)]);
    pn = std::max(pn, H[7] + H[2]);
    ps = std::max(ps, H[8] + std::max(H[3], 1));
    pl = std::max(pl, H[9] + std::max(H[3], 1) * std::max(H[4], 1));
  }
  {
    /* the rings per zebra colour (slot 6; 0 without zebra), and all of them */
    std::vector<int> lst[3];
    for (int r = 0; r < nr; r++) {
      lst[hdr[NH * r + 6] == 1 ? 1 : 0].push_back(r);
      lst[2].push_back(r);
    }
    for (int k = 0; k < 3; k++) {
      R->nrc[k] = (int)lst[k].size();
      PetscCallCXX(upload(R->ract[k], "ring_list", lst[k].data(), lst[k].size()));
    }
  }
  R->nr_tot = nr_tot;
  R->h_hdr.assign(hdr, hdr + (size_t)NH * nr);
  R->h_hdr8.assign(hdr8, hdr8 + (size_t)NH8 * nr);
  R->h_ni.assign(ni, ni + (size_t)2 * pn);
  R->h_n8.assign(n8, n8 + (size_t)5 * pn);
  R->h_si.assign(si, si + (size_t)2 * ps);
  R->h_s8.assign(s8, s8 + (size_t)3 * ps);
  R->h_l8.assign(l8, l8 + (size_t)2 * pl);
  PetscCallCXX(upload(R->hdr, "ring_hdr", hdr, (size_t)NH * nr));
  PetscCallCXX(upload(R->hdr8, "ring_hdr8", hdr8, (size_t)NH8 * nr));
  PetscCallCXX(upload(R->ni, "ring_ni", ni, (size_t)2 * pn));
  PetscCallCXX(upload(R->n8, "ring_n8", n8, (size_t)5 * pn));
  PetscCallCXX(upload(R->si, "ring_si", si, (size_t)2 * ps));
  PetscCallCXX(upload(R->s8, "ring_s8", s8, (size_t)3 * ps));
  PetscCallCXX(upload(R->l8, "ring_l8", l8, (size_t)2 * pl));
  PetscCallCXX(upload(R->rows, "ring_rows", rows, (size_t)nr_tot)); /* the rings' rows only: rows beyond the last ring's are not read */
  for (auto &t : R->tteam) std::fill(t, t + 3, 0);                     /* ring_apply asks for its team size again */
  /* lanes: a node block's rows; threads: the segments, within the device's team limit */
  R->vlen = 1;
  R->tsz  = 1;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  {
    PetscInt  v = R->maxbs >= 24 ? 32 : (R->maxbs >= 12 ? 16 : 8);
    PetscBool set;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-sf_gpu_ring_vlen", &v, &set));
    R->vlen = (int)PetscMin(PetscMax(v, 1), 32);
    R->tsz  = std::max(1, std::min(R->maxns, 1024 / R->vlen));
  }
#endif
  stats[0] = nr;
  stats[1] = R->maxns;
  stats[2] = R->maxbs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The rings' factors (btd_dev_pack), every rebuild: fac (nfac entries, ring
 * r at hdr8's offset), jac[r] != 0: point Jacobi; sp: keep them in single. */
PetscErrorCode jorek_ring_kokkos_values(void *h, long long nfac, const double *fac, const int *jac, int sp, double *stats)
{
  RingDev *R = static_cast<RingDev *>(h);

  PetscFunctionBeginUser;
  R->sp = (sp != 0);
  PetscCallCXX(upload(R->jac, "ring_jac", jac, (size_t)R->nr));
  if (R->sp) {
    std::vector<float> f4((size_t)nfac);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < nfac; i++) f4[i] = (float)fac[i];
    PetscCallCXX(upload(R->f4, "ring_f4", f4.data(), f4.size()));
    R->f8 = Kokkos::View<double *, MemSpace>();
    if (R->w4.extent(0) != (size_t)R->nr_tot) PetscCallCXX(R->w4 = Kokkos::View<float *, MemSpace>("ring_w4", R->nr_tot));
    R->mb = 4.0 * nfac / 1048576.0;
  } else {
    PetscCallCXX(upload(R->f8, "ring_f8", fac, (size_t)nfac));
    R->f4 = Kokkos::View<float *, MemSpace>();
    if (R->w8.extent(0) != (size_t)R->nr_tot) PetscCallCXX(R->w8 = Kokkos::View<double *, MemSpace>("ring_w8", R->nr_tot));
    R->mb = 8.0 * nfac / 1048576.0;
  }
  stats[0] = R->nr;
  stats[1] = R->maxns;
  stats[2] = R->maxbs;
  stats[3] = R->mb;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode jorek_blk_kokkos_zebra(void *, const int **, const int **, const PetscScalar **, PetscScalar **);

/* The device fill's map, once per pattern: entry q of the rings' unfactored
 * storage (0-based dst / gdst) takes src[q] = +k / -k, entry k (1-based) of
 * the level matrix's diagonal / off-diagonal part's values, or gsrc[q], entry
 * k of the ghost rows' values (jorek_ring_kokkos_factor's devfill). */
PetscErrorCode jorek_ring_kokkos_fillmap(void *h, long long n1, const int *src, const long long *dst, long long n2, const int *gsrc, const long long *gdst)
{
  RingDev *R = static_cast<RingDev *>(h);

  PetscFunctionBeginUser;
  PetscCallCXX(upload(R->fm_src, "ring_fm_src", src, (size_t)n1));
  PetscCallCXX(upload(R->fm_dst, "ring_fm_dst", dst, (size_t)n1));
  PetscCallCXX(upload(R->fg_src, "ring_fg_src", gsrc, (size_t)n2));
  PetscCallCXX(upload(R->fg_dst, "ring_fg_dst", gdst, (size_t)n2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Device factorisation, every rebuild: raw = the rings' UNFACTORED storage
 * (D, U, L per node at hdr8's offsets; nfac entries), g8 = per separator its
 * segment's Schur blocks' offsets S_ab, S_ba, S_bb (ng8 = 3 x separators;
 * newpat: a new layout, the plan is rebuilt). devfill: the unfactored values
 * taken on the device instead of raw, from the level matrix's diagonal /
 * off-diagonal parts Ad / Ao (sequential AIJ, Kokkos; Ao may be NULL) and the
 * ghost rows' values vg (host, nvg) through the fill map. The factors end in
 * the solve's form (as btd_dev_pack), single when sp. fail[r] != 0: ring r
 * has a singular pivot block; *ok = 0: no device factorisation for this
 * layout (nothing done). stats as jorek_ring_kokkos_values. */
PetscErrorCode jorek_ring_kokkos_factor(void *h, long long nfac, const double *raw, int ng8, const long long *g8, int newpat, int sp, int devfill, Mat Ad, Mat Ao, long long nvg, const double *vg, int *fail, int *ok, double *stats)
{
  RingDev *R = static_cast<RingDev *>(h);

  PetscFunctionBeginUser;
  if (newpat || !R->plan) {
    R->h_g8.assign(g8, g8 + ng8);
    R->plan       = std::make_shared<FacPlan>();
    R->plan->nfac = nfac;
    plan_build(R, *R->plan);
  }
  *ok = R->plan->ok ? 1 : 0;
  for (int r = 0; r < R->nr; r++) fail[r] = 0;
  if (!*ok) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(ev_rfac, 0, 0, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  const size_t nw = (size_t)(nfac + R->plan->nw);
  if (R->fw.extent(0) != nw) PetscCallCXX(R->fw = Kokkos::View<double *, MemSpace>(Kokkos::view_alloc(std::string("ring_fw"), Kokkos::WithoutInitializing), nw));
  if (!devfill) {
    Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hr(raw, (size_t)nfac);
    PetscCallCXX(Kokkos::deep_copy(Kokkos::subview(R->fw, std::make_pair((size_t)0, (size_t)nfac)), hr));
  } else {
    const PetscInt *ii, *jj;
    PetscScalar    *ad = nullptr, *ao = nullptr;
    PetscMemType    mt;
    PetscCall(MatSeqAIJGetCSRAndMemType(Ad, &ii, &jj, &ad, &mt));
    if (Ao) PetscCall(MatSeqAIJGetCSRAndMemType(Ao, &ii, &jj, &ao, &mt));
    auto fw   = R->fw;
    auto fsrc = R->fm_src;
    auto fdst = R->fm_dst;
    PetscCallCXX(Kokkos::deep_copy(Kokkos::subview(fw, std::make_pair((size_t)0, (size_t)nfac)), 0.0));
    PetscCallCXX(Kokkos::parallel_for(
      "jorek_ring_fill", Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<long long>>(0, (long long)fsrc.extent(0)), KOKKOS_LAMBDA(const long long q) {
        const int sv = fsrc(q);
        fw(fdst(q)) = (sv > 0) ? ad[sv - 1] : ao[-sv - 1];
      }));
    if (R->fg_src.extent(0)) {
      if (R->vg.extent(0) != (size_t)nvg) PetscCallCXX(R->vg = Kokkos::View<double *, MemSpace>(Kokkos::view_alloc(std::string("ring_vg"), Kokkos::WithoutInitializing), (size_t)nvg));
      Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hg(vg, (size_t)nvg);
      auto dg = R->vg;
      PetscCallCXX(Kokkos::deep_copy(dg, hg));
      auto gsrc = R->fg_src;
      auto gdst = R->fg_dst;
      PetscCallCXX(Kokkos::parallel_for(
        "jorek_ring_gfill", Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<long long>>(0, (long long)gsrc.extent(0)), KOKKOS_LAMBDA(const long long q) { fw(gdst(q)) = dg(gsrc(q) - 1); }));
    }
  }
  if (R->fail.extent(0) != (size_t)R->nr) PetscCallCXX(R->fail = IntView("ring_fail", R->nr));
  PetscCallCXX(Kokkos::deep_copy(R->fail, 0));
  PetscCall(plan_run(*R->plan, R->fw, R->fail));
  {
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R->fail);
    for (int r = 0; r < R->nr; r++) fail[r] = hf(r);
  }
  /* the solve's copy: no point-Jacobi rings (a failed ring makes the caller
   * fall back to the host factorisation) */
  R->sp = (sp != 0);
  if (R->jac.extent(0) != (size_t)R->nr) PetscCallCXX(R->jac = IntView("ring_jac", R->nr));
  PetscCallCXX(Kokkos::deep_copy(R->jac, 0));
  if (R->sp) {
    if (R->f4.extent(0) != (size_t)nfac) PetscCallCXX(R->f4 = Kokkos::View<float *, MemSpace>(Kokkos::view_alloc(std::string("ring_f4"), Kokkos::WithoutInitializing), (size_t)nfac));
    auto f4 = R->f4;
    auto fw = R->fw;
    PetscCallCXX(Kokkos::parallel_for("jorek_ring_f2s", Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<long long>>(0, nfac), KOKKOS_LAMBDA(const long long i) { f4(i) = (float)fw(i); }));
    R->f8 = Kokkos::View<double *, MemSpace>();
    if (R->w4.extent(0) != (size_t)R->nr_tot) PetscCallCXX(R->w4 = Kokkos::View<float *, MemSpace>("ring_w4", R->nr_tot));
    R->mb = 4.0 * nfac / 1048576.0;
  } else {
    R->f8 = R->fw; /* the solve reads the first nfac entries */
    R->f4 = Kokkos::View<float *, MemSpace>();
    if (R->w8.extent(0) != (size_t)R->nr_tot) PetscCallCXX(R->w8 = Kokkos::View<double *, MemSpace>("ring_w8", R->nr_tot));
    R->mb = 8.0 * nfac / 1048576.0;
  }
  PetscCallCXX(ExecSpace().fence());
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(ev_rfac, 0, 0, 0, 0));
  stats[0] = R->nr;
  stats[1] = R->maxns;
  stats[2] = R->maxbs;
  stats[3] = R->mb + 8.0 * (double)R->fw.extent(0) / 1048576.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y(ring rows) = ring solves of x; xg: x on the ghost rows. y's other rows are
 * kept. colour >= 0: the zebra pass of that colour only, the coupling read
 * from blk_dev handle bh (NULL: none). */
PetscErrorCode jorek_ring_kokkos_apply(void *h, Vec x, Vec y, Vec xg, int colour, void *bh)
{
  RingDev  *R = static_cast<RingDev *>(h);
  CRealView xv, gv;
  RealView  yv;
  Zebra     zb;

  PetscFunctionBeginUser;
  if (!R->nrc[colour < 0 ? 2 : colour]) PetscFunctionReturn(PETSC_SUCCESS);
  if (colour >= 0 && bh) PetscCall(jorek_blk_kokkos_zebra(bh, &zb.zp, &zb.zc, &zb.zv, &zb.yg));
  PetscCall(PetscLogEventBegin(ev_ring, 0, 0, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(VecGetKokkosView(x, &xv));
  if (R->ngh > 0) PetscCall(VecGetKokkosView(xg, &gv));
  PetscCall(VecGetKokkosView(y, &yv));
  if (R->sp) PetscCall(ring_apply<float>(R, R->f4, R->w4, xv, gv, yv, colour, zb));
  else PetscCall(ring_apply<double>(R, R->f8, R->w8, xv, gv, yv, colour, zb));
  PetscCall(VecRestoreKokkosView(y, &yv));
  if (R->ngh > 0) PetscCall(VecRestoreKokkosView(xg, &gv));
  PetscCall(VecRestoreKokkosView(x, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(ev_ring, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode jorek_ring_kokkos_free(void **h)
{
  PetscFunctionBeginUser;
  if (*h) PetscCallCXX(delete static_cast<RingDev *>(*h));
  *h = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // extern "C"
#else
/* keep the translation unit non-empty for builds without the device path */
extern "C" int jorek_ring_kokkos_unused(void)
{
  return 0;
}
#endif
