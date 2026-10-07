#!/bin/bash
# PETSc with the Kokkos backend (VECKOKKOS / MATAIJKOKKOS + Kokkos Kernels) for
# the GPU port of the SF physics preconditioner.
#
#   ./build_petsc_kokkos.sh cuda      # Kokkos on CUDA: the booster nodes (H100)
#   ./build_petsc_kokkos.sh openmp    # Kokkos on OpenMP: same JOREK source on CPUs
#
# Both use GCC 12.3 + OpenMPI 4.1.6, the toolchain of the existing CUDA PETSc
# (nvcc 12.6 does not take icx/ifx as a host compiler), so one JOREK
# environment serves both prefixes. Kokkos 5 needs C++20.
#
# Run on a LOGIN node: the compute nodes have no outbound HTTPS for --download-*.
set -euo pipefail

VARIANT="${1:?usage: $0 cuda|openmp}"
P="${PETSC_ROOT:-${WORK:-/pitagora_work/FUPB2_REDISMHD}/prac/petsc}"
# The login nodes cap a user's memory: Kokkos Kernels' instantiation files need
# several GB each, and PETSc's default (-j40) gets the compilers killed.
NPROC="${NPROC:-8}"

module load gcc/12.3.0
module load openmpi/4.1.6--gcc--12.3.0-ucx1.20
module load cmake
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran

COMMON=(
  --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90
  --with-fortran-bindings=1 --with-debugging=0
  COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
  --with-cxx-dialect=20 --with-x=0 --with-make-np="${NPROC}"
  --download-openblas --download-scalapack --download-metis --download-parmetis
  --download-ptscotch --download-mumps --download-hypre
  --download-kokkos --download-kokkos-kernels
)

case "${VARIANT}" in
  cuda)
    module load cuda/12.6
    # the fork's tree: it carries the cuDSS backend, kept as a reference solver
    SRC="${P}/petsc-src-3.25.3-cudss"
    ARCH=arch-pitagora-gcc-kokkos-cuda-omp-opt
    PREFIX="${P}/3.25.3-gcc-openmpi-kokkos-cuda-omp"
    # --with-openmp is not optional: JOREK calls LAPACK from OpenMP threads (GMG
    # block factorisations and solves), and an OpenBLAS built without OpenMP
    # is not safe for that (measured: singular blocks, pair_w breakdown).
    EXTRA=( --with-openmp=1 --with-cuda=1 --with-cuda-dir="${CUDA_HOME:?cuda module not loaded}"
            --with-cuda-arch="${CUDA_ARCH:-90}" --download-cudss )
    ;;
  openmp)
    SRC="${P}/petsc-src-3.25.3"
    ARCH=arch-pitagora-gcc-kokkos-omp-opt
    PREFIX="${P}/3.25.3-gcc-openmpi-kokkos-omp"
    EXTRA=( --with-openmp=1 )
    ;;
  *) echo "unknown variant ${VARIANT}"; exit 1 ;;
esac

LOG="${P}/build_petsc_kokkos_${VARIANT}.log"
exec > >(tee "${LOG}") 2>&1
echo ">> $(date)  variant=${VARIANT}  src=${SRC}  prefix=${PREFIX}"

cd "${SRC}"
export PETSC_DIR="${SRC}" PETSC_ARCH="${ARCH}"
python3 ./configure --prefix="${PREFIX}" "${COMMON[@]}" "${EXTRA[@]}"
make PETSC_DIR="${SRC}" PETSC_ARCH="${ARCH}" MAKE_NP="${NPROC}" all
make PETSC_DIR="${SRC}" PETSC_ARCH="${ARCH}" install

grep -E "PETSC_HAVE_(KOKKOS|KOKKOS_KERNELS|CUDA|CUDSS|OPENMP|MUMPS|HYPRE|MPI_GPU_AWARE) " \
     "${PREFIX}/include/petscconf.h"
echo ">> DONE $(date)"
