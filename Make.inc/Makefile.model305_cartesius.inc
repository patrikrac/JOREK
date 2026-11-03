# Configuration file for jorek on Cartesius at surfsara 9/2013
# including flags for model305, model710

# model directory
MODEL = model305
#MODEL = model710

# for model305 with global outputs and alternative current source
#DEFINES = -DJECCD -Daltcs -DUSE_BLOCK
DEFINES = -DJECCD -DUSE_BLOCK -DCOMMON_PTHREAD

# for full mhd model710
#DEFINES = -Dfullmhd -DMULTI_FUN -DUSE_BLOCK

# Fortran compiler 
FC     = mpiifort
CC     = mpiicc
CXX    = mpicxx

FFLAGS_OMP       = -openmp -mt_mpi -mkl=sequential
FFLAGS           = -r8 -i4 -mkl=sequential
FFLAGS          := $(FFLAGS) -fpp
FFLAGS          := $(FFLAGS) -xAVX -align
FFLAGS          := $(FFLAGS) -O2 -mcmodel=medium

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

FFLAGS_NO_OMP    = -fpp -r8 -i4 -O2 -xAVX -align  -mkl=sequential

# Solvers dependencies
USE_HIPS   = 0
USE_PASTIX = 1
USE_PASTIX_MURGE = 1
USE_MUMPS = 0

# Directory containing the user-compiled libraries (e.g., MUMPS, PASTIX, ...)

#Scotch library
SCOTCH_DIR ?= /path/to/scotchall/scotch_5.1
LIBSCOTCH =  $(SCOTCH_DIR)/lib/libptscotch.a $(SCOTCH_DIR)/lib/libscotch.a $(SCOTCH_DIR)/lib/libscotcherr.a
INC_SCOTCH = -I$(SCOTCH_DIR)/include/

#PASTIX_HOME     ?= /path/to/pastix_release_3999/install
#LIB_PASTIX       = `$(PASTIX_HOME)/pastix-conf --libs`
#LIB_PASTIX_MURGE = `$(PASTIX_HOME)/pastix-conf --libs_murge`
#LIB_PASTIX_BLAS  = `$(PASTIX_HOME)/pastix-conf --blas`
#INC_PASTIX       = `$(PASTIX_HOME)/pastix-conf --incs`

# PASTIX
#PASTIX_HOME      = /path/to/pastix_release_3999/install
#LIB_PASTIX       = -L$(PASTIX_HOME) -lpastix_murge -lpastix $(LIBSCOTCH)
#LIB_PASTIX_MURGE = -L$(PASTIX_HOME) -lpastix_murge -lpastix $(LIBSCOTCH)
#LIB_PASTIX_BLAS  = 
#INC_PASTIX       = -I$(PASTIX_HOME) $(INC_SCOTCH)

# PASTIX
PASTIX_HOME      ?= /path/to/pastix_release_3999/install
LIB_PASTIX       = $(PASTIX_HOME)/libpastix_murge.a $(PASTIX_HOME)/libpastix.a $(SCOTCH_DIR)/lib/libptscotch.a  $(SCOTCH_DIR)/lib/libscotch.a $(SCOTCH_DIR)/lib/libscotcherr.a
LIB_PASTIX_MURGE = $(PASTIX_HOME)/libpastix_murge.a $(PASTIX_HOME)/libpastix.a $(SCOTCH_DIR)/lib/libptscotch.a $(SCOTCH_DIR)/lib/libscotch.a $(SCOTCH_DIR)/lib/libscotcherr.a
LIB_PASTIX_BLAS  =
INC_PASTIX       = -I$(PASTIX_HOME) -I$(SCOTCH_DIR)/include/

#HIPS
#LIBHIPS   = -L$(HOME)/hips_trunk/trunk/LIB -lhips -lio $(LIBSCOTCH)
#INCHIPS   = -I$(HOME)/hips_trunk/trunk/LIB

#FFTW
LIBFFTW = /hpc/sw/fftw3-3.3.3-intel-impi/lib/libfftw3.a

# MUMPS
MUMPS_HOME = /hpc/sw/mumps-4.10.0-p3-intel-impi-par/lib
LIB_MUMPS  = /hpc/sw/mumps-4.10.0-p3-intel-impi-par/lib/libdmumps.a /hpc/sw/mumps-4.10.0-p3-intel-impi-par/lib/libmumps_common.a
INC_MUMPS  = /hpc/sw/mumps-4.10.0-p3-intel-impi-par/include
ORDLIB     = /hpc/sw/mumps-4.10.0-p3-intel-impi-par/lib/libpord.a /hpc/sw/mumps-4.10.0-p3-intel-impi-par/lib/libmetis.a

#LAPACK
#MKLROOT   ?= /opt/intel/composer_xe_2013.4.183/mkl/lib/intel64
LIBLAPACK = #/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_lapack95_lp64.a /opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_sequential.a /opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_core.a /opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_blacs_openmpi_lp64.a /opt/intel/composer_xe_2013.1.117/compiler/lib/intel64/libiomp5.a
SCALAP    = /opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_scalapack_lp64.a # /hpc/sw/scalapack-2.0.2-intel-impi/lib/libscalapack.a
BLACS     = /opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.a
