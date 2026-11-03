# Configuration file for jorek

# model directory
MODEL = model500

# Fortran compiler 
FC     = mpif90 
# C compiler
CC     = mpicc -Wall
CXX    = mpicxx

FFLAGS_OMP       = -openmp 
FFLAGS           = -r8 -i4
FFLAGS          := $(FFLAGS) -fpp
#FFLAGS          := $(FFLAGS) -xAVX -align
FFLAGS          := $(FFLAGS) -align
FFLAGS          := $(FFLAGS) -O2 -mcmodel=medium -DUSE_BLOCK -DFUNNELED 
#FFLAGS          := $(FFLAGS) -O0 -mcmodel=medium -DFUNNELED 
#FFLAGS          := $(FFLAGS) -O0 -g -check bounds -traceback -fpe-all=0 -ftrapuv

# --- DEBUGGING
#   --- Debug symbols for debuggers
#DEBUGFLAGS  = -g
#   --- Check array boundaries
#DEBUGFLAGS += -check bounds -traceback -fpe-all=0 -ftrapuv
#   --- Generate and check interfaces (__genmod files)
#DEBUGFLAGS += -gen-interfaces -warn interfaces
#   --- Check many other things
#DEBUGFLAGS += -warn all,nounused -check all,noarg_temp_created -debug all -debug-parameters -fstack-security-check -ftrapuv
#   --- Even more checks and warnings
#DEBUGFLAGS += -warn all -check all

FFLAGS_FIXEDFORM := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)
FFLAGS_NOBOUNDS  := $(FFLAGS)               $(FFLAGS_OMP)
FFLAGS           := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)

FFLAGS_NO_OMP    = -fpp -r8 -i4 -O2 -xAVX -align 

# Solvers dependencies
USE_HIPS   = 0
USE_PASTIX = 1
USE_PASTIX_MURGE = 0
USE_MUMPS = 0

# Directory containing the user-compiled libraries (e.g., MUMPS, PASTIX, ...)
#LIBDIR ?= /path/to/lib
LIBDIR ?= /path/to/lib/pastix/install

#Scotch library
SCOTCH_HOME ?= /path/to/scotch_5.1.12
SCOTCH_LIBDIR = ${SCOTCH_HOME}/lib
LIB_SCOTCH   = -L$(SCOTCH_LIBDIR) -lptscotch -lptscotcherr
INC_SCOTCH   = -I$(SCOTCH_INCDIR)

# MUMPS
MUMPS_HOME = $(LIBDIR)/MUMPS_4.8.3
LIB_MUMPS  = -L$(MUMPS_HOME)/lib/ -ldmumps -lmumps_common
LIB_ZMUMPS = $(MUMPS_HOME)/lib/libzmumps.a
INC_MUMPS  = $(MUMPS_HOME)/include/
ORDLIB     = $(MUMPS_LIB_DIR)/libpord.a $(METIS_LIB_DIR)/libmetis.a 


# PASTIX
#PASTIX_HOME      = $(LIBDIR)/pastix_release_4030/install/
#PASTIX_HOME      = $(LIBDIR)/pastix/install/
PASTIX_HOME ?= /path/to/pastix/install
LIB_PASTIX       = -L$(PASTIX_HOME) -lpastix_murge -lpastix $(LIB_SCOTCH)
LIB_PASTIX_MURGE = -L$(PASTIX_HOME) -lpastix_murge -lpastix $(LIB_SCOTCH)
LIB_PASTIX_BLAS  = 
INC_PASTIX       = -I$(PASTIX_HOME) $(INC_SCOTCH)


#HIPS
LIBHIPS   = -L$(HOME)/hips_trunk/trunk/LIB -lhips -lio $(LIBSCOTCH)
INCHIPS   = -I$(HOME)/hips_trunk/trunk/LIB

#FFTW
#LIBFFTW   =-L$(FFTW3_LIB_DIR) -lfftw3 

#LAPACK
MKL       = $(MKL_LIB_DIR)
LIBLAPACK = -L$(MKL) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -lrt
SCALAP    = $(MKL_SCA_LIBS) 
 
