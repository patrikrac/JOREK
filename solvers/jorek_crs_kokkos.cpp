/* Device direct solve of a GMG redundant LU (mod_petsc_pc_gmg rds_t: the
 * coarsest level, the axis blocks; -sf_gpu_coarse). Every member holds the
 * whole matrix as a sequential CSR (rds_t%sub(1)), and the members are the
 * ranks owning the block's rows: a member packs its own rows of the level
 * vector into the device buffer at its offset, the members exchange their
 * parts (Allgatherv on the members' communicator, through the host; none
 * with one member, the usual case), and each returns its own rows -- the
 * host rds_solve_arr's scheme. Non-members take no part. The solve:
 *   mode 1, dense: the explicit inverse, applied as one matrix-vector
 *           product. Factored on the device: the CSR uploaded and densified
 *           there, cuSOLVER getrf + getrs against the identity (CUDA); LAPACK
 *           on host backends. n^2 entries: meant for the few thousand rows of
 *           a coarsest level or an axis block.
 *   mode 2, cuDSS (CUDA builds with PETSc's cuDSS): sparse LU on the device,
 *           analysis once per pattern, refactorisation per rebuild.
 */
#if defined(USE_PETSC) && defined(USE_GPU_PC)
#include <petscvec_kokkos.hpp>
#include <petscsf.h>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>
#if defined(PETSC_HAVE_CUDSS) && defined(KOKKOS_ENABLE_CUDA)
#include <cudss.h>
#define JOREK_CRS_CUDSS 1
#endif
#include "jorek_dense_batched.hpp"

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using CRealView = Kokkos::View<const PetscScalar *, MemSpace>;
using RealView  = Kokkos::View<PetscScalar *, MemSpace>;
using IntView   = Kokkos::View<int *, MemSpace>;

struct CrsDev {
  int      mode = 0, n = 0, nnz = 0;
  /* the map (members): own rows lidx (local) at loff of the block's ntot;
   * the members' communicator, counts and offsets for the exchange */
  int      ntot = 0, nloc = 0, loff = 0, np = 1;
  MPI_Comm comm = MPI_COMM_NULL;
  std::vector<int> cnt, dsp;
  IntView  lidx;
  RealView db, dx;
  RealView inv;                   /* dense: A^-1, row-major */
  RealView dA;                    /* dense: the factorisation's work matrix */
  IntView  drp, dci;              /* dense: the CSR's pattern on the device */
  RealView dva;
#if defined(KOKKOS_ENABLE_CUDA)
  jorek_dense::LargeWork lw;      /* dense: cuSOLVER's scratch */
#endif
  IntView  rp, ci;                /* cudss: the CSR on the device */
  RealView va;
#ifdef JOREK_CRS_CUDSS
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t   data   = nullptr;
  cudssMatrix_t A = nullptr, B = nullptr, X = nullptr;
  bool          analysed = false;
#endif
};

#ifdef JOREK_CRS_CUDSS
#define CUDSS_CHECK(call) \
  do { \
    cudssStatus_t s_ = (call); \
    PetscCheck(s_ == CUDSS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuDSS error %d in %s", (int)s_, #call); \
  } while (0)

PetscErrorCode cudss_free(CrsDev *C)
{
  PetscFunctionBeginUser;
  if (C->A) CUDSS_CHECK(cudssMatrixDestroy(C->A));
  if (C->B) CUDSS_CHECK(cudssMatrixDestroy(C->B));
  if (C->X) CUDSS_CHECK(cudssMatrixDestroy(C->X));
  if (C->data) CUDSS_CHECK(cudssDataDestroy(C->handle, C->data));
  if (C->config) CUDSS_CHECK(cudssConfigDestroy(C->config));
  if (C->handle) CUDSS_CHECK(cudssDestroy(C->handle));
  C->A = C->B = C->X = nullptr;
  C->data         = nullptr;
  C->config       = nullptr;
  C->handle       = nullptr;
  C->analysed     = false;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscLogEvent ev_crs = -1, ev_crsf = -1;

} // namespace

extern "C" {

/* 1 if the device coarse solve can run in mode (1 dense, 2 cudss) here */
/* the log events, registered on every rank alike (the option parsing's
 * jorek_crs_kokkos_has runs everywhere; only members factor and solve) */
static PetscErrorCode crs_events(void)
{
  PetscFunctionBeginUser;
  if (ev_crs < 0) {
    PetscClassId cid;
    PetscCall(PetscClassIdRegister("JOREK GMG coarse", &cid));
    PetscCall(PetscLogEventRegister("CRS_DevSolve", cid, &ev_crs));
    PetscCall(PetscLogEventRegister("CRS_DevFactor", cid, &ev_crsf));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int jorek_crs_kokkos_has(int mode)
{
  if (crs_events() != PETSC_SUCCESS) return 0;
  if (mode == 1) return 1;
#ifdef JOREK_CRS_CUDSS
  if (mode == 2) return 1;
#endif
  return 0;
}

/* The member's map: lidx = its own block rows (local 0-based, nloc), at loff
 * in the block's ntot rows; fcomm = the members' communicator (Fortran
 * handle), cnt / dsp = the members' row counts / offsets (np of them). */
PetscErrorCode jorek_crs_kokkos_map(void **h, int ntot, int nloc, const int *lidx, int loff, MPI_Fint fcomm, int np, const int *cnt, const int *dsp)
{
  CrsDev *C;

  PetscFunctionBeginUser;
  PetscCall(crs_events());
  if (!*h) PetscCallCXX(*h = new CrsDev);
  C       = static_cast<CrsDev *>(*h);
  C->ntot = ntot;
  C->nloc = nloc;
  C->loff = loff;
  C->np   = np;
  C->comm = MPI_Comm_f2c(fcomm);
  C->cnt.assign(cnt, cnt + np);
  C->dsp.assign(dsp, dsp + np);
  C->db   = RealView("crs_db", ntot);
  C->dx   = RealView("crs_dx", ntot);
  C->lidx = IntView(Kokkos::view_alloc(std::string("crs_lidx"), Kokkos::WithoutInitializing), nloc);
  if (nloc) {
    Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hl(lidx, nloc);
    PetscCallCXX(Kokkos::deep_copy(C->lidx, hl));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* (Re)factor: the coarsest matrix as 0-based CSR (n rows, rp[n] entries);
 * fresh: a new pattern (or the first call). stats[0] = device MB. */
PetscErrorCode jorek_crs_kokkos(void **h, int mode, int fresh, int n, const int *rp, const int *ci, const double *va, double *stats)
{
  CrsDev *C;

  PetscFunctionBeginUser;
  PetscCall(crs_events());
  PetscCheck(jorek_crs_kokkos_has(mode), PETSC_COMM_SELF, PETSC_ERR_SUP, "device coarse solve mode %d not available", mode);
  PetscCall(PetscLogEventBegin(ev_crsf, 0, 0, 0, 0));
  if (!*h) PetscCallCXX(*h = new CrsDev);
  C = static_cast<CrsDev *>(*h);
  if (C->mode != mode || C->n != n) fresh = 1;
  C->mode = mode;
  C->n    = n;
  C->nnz  = rp[n];
  if (mode == 1) {
    /* the CSR to the device (the pattern once), densified there as A^T in
     * column-major order: (A^T)^-1 column-major is A^-1 row-major, inv's form */
    const int nnz = rp[n];
    if (fresh || C->drp.extent(0) != (size_t)(n + 1)) {
      C->drp = IntView(Kokkos::view_alloc(std::string("crs_drp"), Kokkos::WithoutInitializing), n + 1);
      C->dci = IntView(Kokkos::view_alloc(std::string("crs_dci"), Kokkos::WithoutInitializing), nnz);
      Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hrp(rp, n + 1), hci(ci, nnz);
      PetscCallCXX(Kokkos::deep_copy(C->drp, hrp));
      PetscCallCXX(Kokkos::deep_copy(C->dci, hci));
    }
    if (C->dva.extent(0) != (size_t)nnz) C->dva = RealView(Kokkos::view_alloc(std::string("crs_dva"), Kokkos::WithoutInitializing), nnz);
    {
      Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hva(va, nnz);
      PetscCallCXX(Kokkos::deep_copy(C->dva, hva));
    }
    const size_t nn = (size_t)n * n;
    if (C->dA.extent(0) != nn) C->dA = RealView(Kokkos::view_alloc(std::string("crs_dA"), Kokkos::WithoutInitializing), nn);
    if (C->inv.extent(0) != nn) C->inv = RealView(Kokkos::view_alloc(std::string("crs_inv"), Kokkos::WithoutInitializing), nn);
    {
      auto dA = C->dA, dva = C->dva;
      auto drp = C->drp, dci = C->dci;
      PetscCallCXX(Kokkos::deep_copy(dA, 0.0));
      PetscCallCXX(Kokkos::parallel_for(
        "jorek_crs_dense", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int i) {
          for (int k = drp(i); k < drp(i + 1); k++) dA(dci(k) + (size_t)i * n) = dva(k); /* A^T(j, i) = a_ij */
        }));
    }
#if defined(KOKKOS_ENABLE_CUDA)
    {
      IntView inf("crs_info", 1);
      PetscCall(jorek_dense::inverse_large(n, C->dA.data(), C->inv.data(), inf.data(), C->lw));
      auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inf);
      PetscCheck(hi(0) == 0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "coarse dense LU failed, info %d", hi(0));
    }
#else
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible) {
      PetscCallCXX(ExecSpace().fence());
      const int info = jorek_dense::inverse_host(n, C->dA.data());
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "coarse dense LU failed, info %d", info);
      std::copy(C->dA.data(), C->dA.data() + nn, C->inv.data());
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "device dense coarse solve: no dense LU for this Kokkos device");
#endif
    stats[0] = 8.0 * n * (double)n / 1048576.0;
  } else {
#ifdef JOREK_CRS_CUDSS
    const int nnz = rp[n];
    if (fresh) {
      PetscCall(cudss_free(C));
      C->rp = IntView(Kokkos::view_alloc(std::string("crs_rp"), Kokkos::WithoutInitializing), n + 1);
      C->ci = IntView(Kokkos::view_alloc(std::string("crs_ci"), Kokkos::WithoutInitializing), nnz);
      C->va = RealView(Kokkos::view_alloc(std::string("crs_va"), Kokkos::WithoutInitializing), nnz);
      Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hrp(rp, n + 1), hci(ci, nnz);
      PetscCallCXX(Kokkos::deep_copy(C->rp, hrp));
      PetscCallCXX(Kokkos::deep_copy(C->ci, hci));
      CUDSS_CHECK(cudssCreate(&C->handle));
      CUDSS_CHECK(cudssConfigCreate(&C->config));
      CUDSS_CHECK(cudssDataCreate(C->handle, &C->data));
    }
    Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hva(va, nnz);
    PetscCallCXX(Kokkos::deep_copy(C->va, hva));
    ExecSpace exec;
    PetscCallCXX(exec.fence());
    CUDSS_CHECK(cudssSetStream(C->handle, exec.cuda_stream()));
    if (fresh) {
      CUDSS_CHECK(cudssMatrixCreateCsr(&C->A, n, n, nnz, C->rp.data(), nullptr, C->ci.data(), C->va.data(), CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
      /* placeholders: the solve points them at the right-hand side / solution */
      CUDSS_CHECK(cudssMatrixCreateDn(&C->B, n, 1, n, C->va.data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
      CUDSS_CHECK(cudssMatrixCreateDn(&C->X, n, 1, n, C->va.data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
      CUDSS_CHECK(cudssExecute(C->handle, CUDSS_PHASE_ANALYSIS, C->config, C->data, C->A, C->X, C->B));
      CUDSS_CHECK(cudssExecute(C->handle, CUDSS_PHASE_FACTORIZATION, C->config, C->data, C->A, C->X, C->B));
      C->analysed = true;
    } else {
      CUDSS_CHECK(cudssMatrixSetCsrPointers(C->A, C->rp.data(), nullptr, C->ci.data(), C->va.data()));
      CUDSS_CHECK(cudssExecute(C->handle, CUDSS_PHASE_REFACTORIZATION, C->config, C->data, C->A, C->X, C->B));
    }
    PetscCallCXX(exec.fence());
    stats[0] = (12.0 * nnz + 4.0 * (n + 1)) / 1048576.0;
#endif
  }
  PetscCall(PetscLogEventEnd(ev_crsf, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* x(own block rows) = A^-1 b(block rows), b and x level vectors; members only */
PetscErrorCode jorek_crs_kokkos_apply(void *h, Vec b, Vec x)
{
  CrsDev   *C = static_cast<CrsDev *>(h);
  CRealView bv;
  RealView  xv;

  PetscFunctionBeginUser;
  /* a member without rows still takes part in the members' exchange */
  if (C->nloc <= 0 && C->np <= 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(ev_crs, 0, 0, 0, 0));
  PetscCall(PetscLogGpuTimeBegin());
  {
    auto      db   = C->db;
    auto      lidx = C->lidx;
    const int lo = C->loff;
    PetscCall(VecGetKokkosView(b, &bv));
    PetscCallCXX(Kokkos::parallel_for("jorek_crs_gather", Kokkos::RangePolicy<ExecSpace>(0, C->nloc), KOKKOS_LAMBDA(const int q) { db(lo + q) = bv(lidx(q)); }));
    PetscCall(VecRestoreKokkosView(b, &bv));
    if (C->np > 1) { /* the members' parts, through the host */
      auto hb = Kokkos::create_mirror_view(C->db);
      PetscCallCXX(Kokkos::deep_copy(hb, C->db));
      PetscCallMPI(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hb.data(), C->cnt.data(), C->dsp.data(), MPIU_SCALAR, C->comm));
      PetscCallCXX(Kokkos::deep_copy(C->db, hb));
    }
  }
  {
    auto db = C->db, dx = C->dx;
    if (C->mode == 1) {
      auto      inv = C->inv;
      const int n   = C->n;
      using Team    = Kokkos::TeamPolicy<ExecSpace>::member_type;
      int vl        = 1;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
      vl = 32;
#endif
      const int TS = (vl > 1) ? 8 : 1, nck = (n + TS - 1) / TS;
      PetscCallCXX(Kokkos::parallel_for(
        "jorek_crs_dense", Kokkos::TeamPolicy<ExecSpace>(nck, TS, vl), KOKKOS_LAMBDA(const Team &team) {
          const int i = team.league_rank() * TS + team.team_rank();
          if (i >= n) return;
          const PetscScalar *ai  = &inv((size_t)i * n);
          PetscScalar        acc = 0.0;
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, n), [=](const int j, PetscScalar &sm) { sm += ai[j] * db(j); }, acc);
          Kokkos::single(Kokkos::PerThread(team), [=]() { dx(i) = acc; });
        }));
    } else {
#ifdef JOREK_CRS_CUDSS
      ExecSpace exec;
      CUDSS_CHECK(cudssSetStream(C->handle, exec.cuda_stream()));
      CUDSS_CHECK(cudssMatrixSetValues(C->B, (void *)db.data()));
      CUDSS_CHECK(cudssMatrixSetValues(C->X, (void *)dx.data()));
      CUDSS_CHECK(cudssExecute(C->handle, CUDSS_PHASE_SOLVE, C->config, C->data, C->A, C->X, C->B));
#endif
    }
    PetscCall(VecGetKokkosView(x, &xv));
    auto      lidx = C->lidx;
    const int lo   = C->loff;
    PetscCallCXX(Kokkos::parallel_for("jorek_crs_return", Kokkos::RangePolicy<ExecSpace>(0, C->nloc), KOKKOS_LAMBDA(const int q) { xv(lidx(q)) = dx(lo + q); }));
    PetscCall(VecRestoreKokkosView(x, &xv));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(ev_crs, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode jorek_crs_kokkos_free(void **h)
{
  PetscFunctionBeginUser;
  if (*h) {
#ifdef JOREK_CRS_CUDSS
    PetscCall(cudss_free(static_cast<CrsDev *>(*h)));
#endif
    PetscCallCXX(delete static_cast<CrsDev *>(*h));
  }
  *h = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // extern "C"
#else
extern "C" int jorek_crs_kokkos_unused(void)
{
  return 0;
}
#endif
