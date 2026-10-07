#!/bin/bash
# Environment for building/running JOREK on Pitagora against a Kokkos PETSc
# (tools/gpu_pbpc/build_petsc_kokkos.sh). GCC 12.3 + OpenMPI 4.1.6 for both.
#
#   source tools/gpu_pbpc/env_pitagora_gpu.sh cuda     # booster nodes (H100)
#   source tools/gpu_pbpc/env_pitagora_gpu.sh openmp   # CPU nodes, login node
#
# Pairs with tools/gpu_pbpc/Makefile.pitagora.inc. PETSC_DIR, if already set,
# wins over the variant's prefix.

_variant="${1:-cuda}"
_root="${WORK:-/pitagora_work/FUPB2_REDISMHD}/prac/petsc"

module load gcc/12.3.0
module load openmpi/4.1.6--gcc--12.3.0-ucx1.20
module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.3.0-ucx1.20
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.3.0-ucx1.20
module load boost/1.85.0--openmpi--4.1.6--gcc--12.3.0-ucx1.20

case "${_variant}" in
  cuda)   module load cuda/12.6
          export PETSC_DIR="${PETSC_DIR:-${_root}/3.25.3-gcc-openmpi-kokkos-cuda-omp}" ;;
  openmp) export PETSC_DIR="${PETSC_DIR:-${_root}/3.25.3-gcc-openmpi-kokkos-omp}" ;;
  *)      echo "!! unknown variant ${_variant} (cuda|openmp)" ;;
esac
export PETSC_ARCH=""
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran

# Spack modules differ in which variable they export; take the first that is a directory.
_pick() {                       # _pick OUTVAR CANDIDATE_VAR...
  local out="$1"; shift; local v
  for v in "$@"; do
    if [ -n "${!v:-}" ] && [ -d "${!v}" ]; then export "$out=${!v}"; return 0; fi
  done
  echo "!! $out: none of $* set" >&2; return 1
}
_pick HDF5_ROOT_DIR  HDF5_HOME  HDF5_ROOT  HDF5_DIR
_pick FFTW_ROOT_DIR  FFTW_HOME  FFTW_ROOT  FFTW_DIR
_pick BOOST_ROOT_DIR BOOST_HOME BOOST_ROOT BOOST_DIR
unset -f _pick
unset _variant _root

[ -f "${PETSC_DIR}/include/petscconf.h" ] || echo "!! no PETSc at ${PETSC_DIR}: run tools/gpu_pbpc/build_petsc_kokkos.sh"
