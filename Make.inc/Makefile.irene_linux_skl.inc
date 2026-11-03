# Makefile.inc file for compiling JOREK on the Marconi-Fusion Supercomputer
#
# * Make sure to set PASTIX_HOME below to the correct directory.
# * Make sure that you have the following modules loaded: intel/18.0.3.222 mpi/intelmpi/2018.0.3.222

# --- Select physics model
MODEL             = model303

# --- Compiler and options
FC                = mpiifort -f90=ifort -mt_mpi
CC                = mpiicc   -cc=icc    -mt_mpi
CXX               = mpiicpc  -cxx=icpc  -mt_mpi
FFLAGS_OMP        = -qopenmp
FFLAGS            = -r8 -i4 -DUSE_BLOCK
FFLAGS           += -fpp
FFLAGS           += -axCORE-AVX512 -mtune=skylake
FFLAGS           += -DCONSTRUCT_MATRIX_OMP_ATOMIC

# --- Debugging options
#DEBUGFLAGS        = -g
#DEBUGFLAGS      += -warn all,nounused
#DEBUGFLAGS      += -check all,noarg_temp_created
#DEBUGFLAGS      += -debug all
#DEBUGFLAGS      += -debug-parameters
#DEBUGFLAGS      += -fstack-security-check
#DEBUGFLAGS      += -ftrapuv

# --- Various switches
USE_PASTIX        = 1
USE_PASTIX_MURGE  = 0
USE_HDF5          = 1
USE_FFTW          = 1
USE_MUMPS         = 1

# --- Root of the pre-built JOREK tool-chain for this machine
JOREK_TOOLS ?= /path/to/jorek_tools_irene_skl

# --- PaStiX solver
PASTIX_HOME ?= $(JOREK_TOOLS)/pastix_release_4492/install
LIB_PASTIX        = `$(PASTIX_HOME)/pastix-conf --libs` 
LIB_PASTIX_MURGE  = `$(PASTIX_HOME)/pastix-conf --libs_murge`
LIB_PASTIX_BLAS   = `$(PASTIX_HOME)/pastix-conf --blas`
INC_PASTIX        = `$(PASTIX_HOME)/pastix-conf --incs`

# --- FFTW Library
#LIBFFTW           = $(FFTW_HOME)/lib/libfftw3.a
INC_FFTW          = -I${MKL_INCDIR}/fftw

# --- Usually not to be changed
FFLAGS_FIXEDFORM := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)
FFLAGS_NOBOUNDS  := $(FFLAGS)               $(FFLAGS_OMP)
FFLAGS           := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)
FFLAGS_NO_OMP     = $(FFLAGS) $(DEBUGFLAGS)

HDF5_HOME   = $(JOREK_TOOLS)/hdf5-1.8.10
HDF5INCLUDE = $(HDF5_HOME)/include
HDF5LIB     = $(HDF5_HOME)/lib/libhdf5hl_fortran.a \
              $(HDF5_HOME)/lib/libhdf5_hl.a \
              $(HDF5_HOME)/lib/libhdf5_fortran.a \
              $(HDF5_HOME)/lib/libhdf5.a \
              -lz

# --- MUMPS solver
MUMPS_LIBDIR  = $(JOREK_TOOLS)/MUMPS_5.0.1/lib
SCOTCH_LIBDIR = $(JOREK_TOOLS)/scotch_6.0.4//lib
LIB_MUMPS     = -L$(MUMPS_LIBDIR) -ldmumps -lmumps_common -lptesmumps -L$(SCOTCH_LIBDIR) -lptscotch -lptscotcherr -lscotch -lpord
MUMPS_INCDIR  = $(JOREK_TOOLS)/MUMPS_5.0.1/include
INC_MUMPS     = $(MUMPS_INCDIR)
SCALAP        = -L$(MKL_LIBDIR) -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
