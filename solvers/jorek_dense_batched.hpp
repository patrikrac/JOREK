/* Dense inverses shared by the device solvers of the GMG smoothers and the
 * coarse solve (jorek_ring_kokkos, jorek_blk_kokkos, jorek_crs_kokkos): the
 * cuBLAS / cuSOLVER handles (one each, on the Kokkos stream), batched inverses
 * of small blocks (cuBLAS), the inverse of one large block (cuSOLVER) and
 * their LAPACK counterparts for host backends. Blocks column-major. */
#pragma once
#include <petscsys.h>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <string>
#include <vector>
#if defined(KOKKOS_ENABLE_CUDA)
  #include <cublas_v2.h>
  #include <cusolverDn.h>
#endif

extern "C" {
void dgetrf_(const int *, const int *, double *, const int *, int *, int *);
void dgetri_(const int *, double *, const int *, const int *, double *, const int *, int *);
}

namespace jorek_dense {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using IntView   = Kokkos::View<int *, MemSpace>;
using RealView  = Kokkos::View<double *, MemSpace>;
using PtrView   = Kokkos::View<uintptr_t *, MemSpace>;

/* v holds at least n entries (contents not kept when it grows) */
template <class V>
void grow(V &v, const char *name, size_t n)
{
  if (v.extent(0) < n) v = V(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), n);
}

/* host: A^-1 in place from A's LU (dgetrf's pivots ip); LAPACK's info */
inline int getri_host(int n, double *a, const int *ip)
{
  int    info = 0, lw = -1;
  double q;
  dgetri_(&n, a, &n, ip, &q, &lw, &info);
  if (info) return info;
  lw = std::max(1, (int)q);
  std::vector<double> wk(lw);
  dgetri_(&n, a, &n, ip, wk.data(), &lw, &info);
  return info;
}

/* host: A^-1 in place (n x n, ld n); LAPACK's info, != 0: A singular */
inline int inverse_host(int n, double *a)
{
  std::vector<int> ip(std::max(n, 1));
  int              info = 0;
  dgetrf_(&n, &n, a, &n, ip.data(), &info);
  return info ? info : getri_host(n, a, ip.data());
}

#if defined(KOKKOS_ENABLE_CUDA)
inline PetscErrorCode cublas_handle(cublasHandle_t *hd)
{
  static cublasHandle_t h = nullptr;

  PetscFunctionBegin;
  if (!h) PetscCheck(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cublasCreate");
  PetscCheck(cublasSetStream(h, ExecSpace().cuda_stream()) == CUBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cublasSetStream");
  *hd = h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode cusolver_handle(cusolverDnHandle_t *sh)
{
  static cusolverDnHandle_t h = nullptr;

  PetscFunctionBegin;
  if (!h) PetscCheck(cusolverDnCreate(&h) == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreate");
  PetscCheck(cusolverDnSetStream(h, ExecSpace().cuda_stream()) == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnSetStream");
  *sh = h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The scratch of inverse_batched, kept by the caller between calls */
struct BatchWork {
  IntView piv, info2;
};

/* C_i = A_i^-1 for nb blocks of order n (cuBLAS): pa / pc device arrays of
 * device pointers, A_i with ld n (overwritten), C_i with ld ldc (not A_i).
 * info(i) != 0: A_i singular. matinv up to n 32, getrf + getri above. */
inline PetscErrorCode inverse_batched(int n, int nb, const PtrView &pa, const PtrView &pc, int ldc, const IntView &info, BatchWork &w)
{
  cublasHandle_t hd;

  PetscFunctionBegin;
  PetscCall(cublas_handle(&hd));
  if (n <= 32) {
    PetscCheck(cublasDmatinvBatched(hd, n, (const double **)pa.data(), n, (double **)pc.data(), ldc, info.data(), nb) == CUBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cublasDmatinvBatched");
  } else {
    grow(w.piv, "dense_piv", (size_t)nb * n);
    grow(w.info2, "dense_info2", (size_t)nb);
    PetscCheck(cublasDgetrfBatched(hd, n, (double **)pa.data(), n, w.piv.data(), info.data(), nb) == CUBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cublasDgetrfBatched");
    PetscCheck(cublasDgetriBatched(hd, n, (const double **)pa.data(), n, w.piv.data(), (double **)pc.data(), ldc, w.info2.data(), nb) == CUBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cublasDgetriBatched");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The scratch of inverse_large */
struct LargeWork {
  RealView wk;
  IntView  ip, info2;
};

/* X = A^-1 for one large block of order n (cuSOLVER getrf, then getrs against
 * the identity): A with ld n (overwritten), X with ld n; *info (device):
 * getrf's, != 0: A singular. The batched LU is meant for small blocks. */
inline PetscErrorCode inverse_large(int n, double *A, double *X, int *info, LargeWork &w)
{
  cusolverDnHandle_t sh;
  int                lw = 0;

  PetscFunctionBegin;
  PetscCall(cusolver_handle(&sh));
  PetscCheck(cusolverDnDgetrf_bufferSize(sh, n, n, A, n, &lw) == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnDgetrf_bufferSize");
  grow(w.wk, "dense_swk", (size_t)std::max(lw, 1));
  grow(w.ip, "dense_sip", (size_t)n);
  grow(w.info2, "dense_sinf", 1);
  PetscCallCXX(Kokkos::parallel_for(
    "jorek_dense_eye", Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<long long>>(0, (long long)n * n), KOKKOS_LAMBDA(const long long q) { X[q] = (q % (n + 1) == 0) ? 1.0 : 0.0; }));
  PetscCheck(cusolverDnDgetrf(sh, n, n, A, n, w.wk.data(), w.ip.data(), info) == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnDgetrf");
  /* getrs has its own status: it would overwrite getrf's singular one */
  PetscCheck(cusolverDnDgetrs(sh, CUBLAS_OP_N, n, n, A, n, w.ip.data(), X, n, w.info2.data()) == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnDgetrs");
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

} // namespace jorek_dense
