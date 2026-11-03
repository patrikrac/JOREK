# Confiduration file for jorek
# model directory
MODEL = model199

# Fortran compiler 
FC     = ifort
CC     = icc
CXX    = icpc

FFLAGS_OMP       = -openmp 
FFLAGS           = -O2 -DUSE_R3_INFO_MPI -DUSE_R3_INFO
FFLAGS          := $(FFLAGS) -r8 -i4 -mcmodel=large
FFLAGS          := $(FFLAGS) -fpp
FFLAGS          := $(FFLAGS) -traceback -xT -align
FFLAGS          := $(FFLAGS) -DMURGE_INTERFACE_MAJOR_VERSION=1 -DMURGE_INTERFACE_MINOR_VERSION=0

CFLAGS           = -DUSE_R3_INFO_MPI -DUSE_R3_INFO

FFLAGS_FIXEDFORM = $(FFLAGS) $(FFLAGS_OMP)
FFLAGS_NOBOUNDS  = $(FFLAGS) $(FFLAGS_OMP)
FFLAGS          := $(FFLAGS) $(FFLAGS_OMP)

#FFLAGS_NO_OMP    = -fpp -r8 -i8 -O2 -xT -align

# Solvers dependencies
USE_HIPS   = 0
USE_PASTIX = 1
USE_PASTIX_MURGE = 1
USE_MUMPS = 1
USE_WSMP   = 0

LIBHOME ?= /path/to/jorek_tools

#Scotch library
#SCOTCH_HOME ?= $(LIBHOME)/scotch
#LIB_SCOTCH   = -L$(SCOTCH_HOME)/lib -lscotch -lscotcherr
#INC_SCOTCH   = -I$(SCOTCH_HOME)/include

# PASTIX
PASTIX_HOME     ?=  $(LIBHOME)/pastix
LIB_PASTIX       = `$(PASTIX_HOME)/pastix-conf --libs` 
LIB_PASTIX_MURGE = `$(PASTIX_HOME)/pastix-conf --libs_murge`
LIB_PASTIX_BLAS  = `$(PASTIX_HOME)/pastix-conf --blas`
INC_PASTIX       = `$(PASTIX_HOME)/pastix-conf --incs`

#HIPS
#LIBHIPS   = -L$(HOME)/hips_trunk/trunk/LIB -lhips -lio $(LIBSCOTCH)
#INCHIPS   = -I$(HOME)/hips_trunk/trunk/LIB

#WSMP
LIB_WSMP = -L$(WSMP_HOME) -lpwsmp64

# LIBFFTW
LIBFFTW   = /opt/software/SGI/fftw/3.2.2/lib/libfftw3.a  

# MUMPS
MUMPS_HOME = $(LIBHOME)/MUMPS_4.9.2
LIB_MUMPS  = $(MUMPS_HOME)/lib/libdmumps.a $(MUMPS_HOME)/lib/libmumps_common.a
INC_MUMPS  = $(MUMPS_HOME)/include/
ORDLIB     = $(MUMPS_HOME)/lib/libpord.a $(LIBHOME)/metis-4.0/libmetis.a

LIBLAPACK = -lmkl_lapack -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
SCALAP    = -lmkl_scalapack_lp64
BLACS     = -lmkl_blacs_lp64

PPPLIB    = $(LIBHOME)/ppplib/ppplib.a
