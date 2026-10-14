/* Strided local copies between PETSc Kokkos vectors (mod_petsc_blk_dev
 * vec_copy_strided): dst(doff + i dstr + j) = src(soff + i sstr + j), j < blk,
 * i < nblk, on the local arrays. The SF preconditioner's apply moves its
 * variables between the full system, the family layout and the packed pairs
 * this way (split_vars / merge_vars, sff_scatter_in / out, sf_split_parts),
 * which on the device path must not leave the device.
 */
#if defined(USE_PETSC) && defined(USE_GPU_PC)
#include <petscvec_kokkos.hpp>
#include <petsc/private/petscimpl.h> /* PetscObjectStateIncrease */
#include <Kokkos_Core.hpp>

extern "C" PetscErrorCode jorek_vec_kokkos(Vec dst, int doff, int dstr, Vec src, int soff, int sstr, int blk, int nblk)
{
  using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  Kokkos::View<const PetscScalar *, MemSpace> sv;
  Kokkos::View<PetscScalar *, MemSpace>       dv;

  PetscFunctionBeginUser;
  if (blk <= 0 || nblk <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetKokkosView(src, &sv));
  PetscCall(VecGetKokkosView(dst, &dv));
  const long long n = (long long)blk * nblk;
  PetscCallCXX(Kokkos::parallel_for(
    "jorek_vec_copy", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<long long>>(0, n), KOKKOS_LAMBDA(const long long q) {
      const long long i = q / blk, j = q % blk;
      dv(doff + i * dstr + j) = sv(soff + i * sstr + j);
    }));
  PetscCall(VecRestoreKokkosView(dst, &dv));
  PetscCall(VecRestoreKokkosView(src, &sv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* v's values changed elsewhere (another rank's part, a device kernel): a new
 * object state, so that cached norms are dropped on every rank alike and the
 * next VecNorm is collective everywhere */
extern "C" PetscErrorCode jorek_vec_touch(Vec v)
{
  PetscFunctionBeginUser;
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#else
extern "C" int jorek_vec_kokkos_unused(void)
{
  return 0;
}
#endif
