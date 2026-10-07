# Makefile.inc for JOREK on Pitagora (CINECA, 4x NVIDIA H100 per node),
# built against a Kokkos PETSc from tools/gpu_pbpc/build_petsc_kokkos.sh
# (CUDA or OpenMP backend; the env script selects the prefix).
#
#   source tools/gpu_pbpc/env_pitagora_gpu.sh
#   cp     tools/gpu_pbpc/Makefile.pitagora.inc Makefile.inc
#   make -j8
#
# The env script loads the modules and exports PETSC_DIR, HDF5_ROOT_DIR,
# FFTW_ROOT_DIR and BOOST_ROOT_DIR. Do not hardcode paths here.
#
# TOOLCHAIN: OpenMPI 4.1.6 + GCC 12.3 + CUDA 12.6 -- NOT Pitagora's default
# Intel MPI. PETSc is built --with-cuda and nvcc 12.6 does not accept icx/ifx as
# a host compiler; JOREK and PETSc must share one MPI ABI and one Fortran
# compiler, so JOREK follows PETSc.

MODEL             = model199

# --- Compilers -------------------------------------------------------------
# OpenMPI wrappers, GCC-backed (env_pitagora.sh pins OMPI_FC/CC/CXX).
FC                = mpif90
CC                = mpicc
CXX               = mpicxx

# --- Switches --------------------------------------------------------------
# The native solver backends stay OFF: PETSc supplies MUMPS (and now cuDSS), and
# the whole point of this build is to reach them through the PETSc options
# database -- see namelist/jorek.petsc.example.
USE_PASTIX        = 0
USE_PASTIX_MURGE  = 0
USE_STRUMPACK     = 0
USE_MUMPS         = 0
USE_WSMP          = 0
USE_MKL           = 0

USE_PETSC         = 1
# device path of the physics PC: needs a PETSc with Kokkos
USE_GPU_PC        = 1
USE_SLEPC         = 0

USE_BLOCK         = 1
USE_HDF5          = 1
USE_FFTW          = 1
USE_BOOST         = 1

# --- Preprocessor ----------------------------------------------------------
DEFINES           = -DUSE_MPI -DUSE_BOOST

# --- PETSc -----------------------------------------------------------------
# PETSC_DIR comes from env_pitagora.sh. defaults.mk includes PETSc's own
# lib/petsc/conf/variables, so $(PETSC_LIB) already carries cuDSS, MUMPS,
# OpenBLAS, ScaLAPACK, METIS/ParMETIS, HYPRE and the CUDA runtime. Nothing to
# add here, and no BLAS/LAPACK or ScaLAPACK of our own -- adding a second copy
# is how you get duplicate-symbol and wrong-library-picked bugs.
ifeq ($(PETSC_DIR),)
  $(error PETSC_DIR is not set -- run: source tools/gpu_pbpc/env_pitagora_gpu.sh)
endif

# --- HDF5 ------------------------------------------------------------------
HDF5INCLUDE       = $(HDF5_ROOT_DIR)/include
HDF5LIB           = -L$(HDF5_ROOT_DIR)/lib -lhdf5_hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz
# NOTE: HDF5 <= 1.8 spells the first one -lhdf5hl_fortran (no underscore after
# hdf5). Pitagora ships 1.14.3, which uses the name above.

# --- FFTW ------------------------------------------------------------------
INC_FFTW          = -I$(FFTW_ROOT_DIR)/include
LIBFFTW           = -L$(FFTW_ROOT_DIR)/lib -lfftw3

# --- Boost -----------------------------------------------------------------
INC_BOOST         = -I$(BOOST_ROOT_DIR)/include
LIB_BOOST         = -L$(BOOST_ROOT_DIR)/lib -lboost_math_tr1

# --- Flags -----------------------------------------------------------------
# Not -march=native: the login and booster nodes are Intel, the DCGP nodes AMD.
FLAGS             = -g -O2 -march=x86-64-v3 -fopenmp

# -ffree-line-length-none is NOT cosmetic. PetscCallA(x) expands to
# `call x; CHKERRA(ierr)`, and CHKERRA embeds __FILE__, so the expanded line
# overruns gfortran's 132-column free-form default. Without this flag the build
# dies with "Line truncated" / "Unterminated character constant".
FFLAGS            = -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none

CXXFLAGS          = -std=c++20

