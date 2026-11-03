# Confiduration file for jorek
# model directory
MODEL = model199

# Fortran compiler 
FC     = mpif90
CC     = mpicc
CXX    = mpicxx

FFLAGS_OMP       = -openmp 
FFLAGS           = -r8 -i4
FFLAGS          := $(FFLAGS) -fpp -DUSE_BLOCK
FFLAGS          := $(FFLAGS) -traceback -xSSE4.2 -align
FFLAGS          := $(FFLAGS) -O2 -DFUNNELED  # Optimization level
#FFLAGS          := $(FFLAGS) -DMURGE_INTERFACE_MAJOR_VERSION=1 -DMURGE_INTERFACE_MINOR_VERSION=0

CFLAGS=-DSTAN_FLAG

# For PaStiX after revision 3945, additionally the following is required (Wiki Matthias)
#FFLAGS := $(FFLAGS) -DFUNNELED -DWORLDWAR2
#FFLAGS := $(FFLAGS) -DWORLDWAR2

# --- DEBUGGING
#   --- Debug symbols for debuggers
#DEBUGFLAGS  = -g
#   --- Check array boundaries
#DEBUGFLAGS += -check bounds
#   --- Generate and check interfaces (__genmod files)
#DEBUGFLAGS += -gen-interfaces -warn interfaces
#   --- Check many other things
#DEBUGFLAGS += -warn all,nounused -check all,noarg_temp_created -debug all -debug-parameters -fstack-security-check -ftrapuv
#   --- Even more checks and warnings
#DEBUGFLAGS += -warn all -check all

FFLAGS_FIXEDFORM := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)
FFLAGS_NOBOUNDS  := $(FFLAGS)               $(FFLAGS_OMP)
FFLAGS           := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)

#FFLAGS_NO_OMP    = -fpp -r8 -i4 -O2 -xSSSE3 -align

# Solvers dependencies
USE_HIPS   = 0
USE_PASTIX = 1
USE_PASTIX_MURGE = 0
USE_MUMPS = 0
USE_WSMP   = 0
USE_HDF5 = 1

#Scotch library
SCOTCH_HOME ?= /path/to/jorek_tools_bullxmpi/scotch
LIB_SCOTCH   = -L$(SCOTCH_HOME)/lib -lscotch -lscotcherr
INC_SCOTCH   = -I$(SCOTCH_HOME)/include

# PASTIX
PASTIX_HOME ?= /path/to/jorek_tools_bullxmpi/pastix_greatbear/install
LIB_PASTIX       = `$(PASTIX_HOME)/pastix-conf --libs` 
LIB_PASTIX_MURGE = `$(PASTIX_HOME)/pastix-conf --libs_murge`
LIB_PASTIX_BLAS  = `$(PASTIX_HOME)/pastix-conf --blas`
INC_PASTIX       = `$(PASTIX_HOME)/pastix-conf --incs`

#LAPACK
#LIBLAPACK = -L$(MKL_LIB_DIR) -lmkl_lapack -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5
#LIBLAPACK = -L$(MKL_LIB_DIR) -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
LIBLAPACK = -L$(MKL_LIB_DIR) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

#HDF5
HDF5_HOME   = ${HDF5HOME}
HDF5INCLUDE = $(HDF5_HOME)/include
HDF5LIB     = $(HDF5_HOME)/lib/libhdf5hl_fortran.a \
              $(HDF5_HOME)/lib/libhdf5_hl.a \
              $(HDF5_HOME)/lib/libhdf5_fortran.a \
              $(HDF5_HOME)/lib/libhdf5.a -lz \
