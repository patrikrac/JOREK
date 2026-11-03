# Configuration file for jorek
# model directory
MODEL = model303

# Fortran compiler 
FC     = mpif90
CC     = mpicc
CXX    = g++

FFLAGS_OMP       = -fopenmp 
FFLAGS           = -g -Og 
FFLAGS          := $(FFLAGS) -fdefault-real-8 -fdefault-double-8
FFLAGS          := $(FFLAGS) -ffree-line-length-none
FFLAGS          := $(FFLAGS) -x f95-cpp-input
## TODO find equivalents of this ifort options
#FFLAGS          := $(FFLAGS) -traceback -xT -align 

FFLAGS_FIXEDFORM = $(FFLAGS) -ffixed-form  -fbounds-check $(FFLAGS_OMP) 
FFLAGS_NOBOUNDS  = $(FFLAGS) -ffixed-form                 $(FFLAGS_OMP) 
FFLAGS          := $(FFLAGS)               -fbounds-check $(FFLAGS_OMP) 

#FFLAGS_NO_OMP   = -fpp -O2 -xT -align
FFLAGS          := $(FFLAGS) -DMURGE_INTERFACE_MAJOR_VERSION=1 -DMURGE_INTERFACE_MINOR_VERSION=0 -DUSE_BLOCK

# Solvers dependencies
USE_HIPS   = 0
USE_PASTIX = 0
USE_BLOCK  = 1
USE_PASTIX_MURGE = 0
USE_MUMPS  = 1
USE_WSMP   = 0
USE_FFTW   = 1
USE_HDF5   = 1

#Scotch library
SCOTCH_HOME ?= $(EBROOTSCOTCH)
#SCOTCH_HOME ?= /path/to/scotch_5.1.12
#LIB_SCOTCH   = -L$(SCOTCH_HOME)/lib -lscotch -lscotcherr
#INC_SCOTCH   = -I$(SCOTCH_HOME)/include


# PASTIX
PASTIX_HOME      ?= $(EBROOTPASTIX)
LIB_PASTIX       = `$(PASTIX_HOME)/pastix-conf --libs` `pkg-config --libs openblas`
LIB_PASTIX_MURGE = `$(PASTIX_HOME)/pastix-conf --libs_murge` 
LIB_PASTIX_BLAS  = -lopenblas
INC_PASTIX       = `$(PASTIX_HOME)/pastix-conf --incs`
PASTIX_MEMORY_USAGE = 0

# MUMPS
MUMPS_HOME ?= $(EBROOTMUMPS)
METIS_HOME ?= $(EBROOTMETIS)
LIB_MUMPS  = -L$(MUMPS_HOME)/lib -ldmumps -lmumps_common
LIB_MUMPS += -L$(SCOTCH_HOME)/lib -lptesmumps -lesmumps -lptscotch -lptscotcherrexit -lscotch -lscotcherrexit
LIB_MUMPS += -L$(METIS_HOME)/lib -lmetis
LIB_MUMPS += -lscalapack `pkg-config --libs openblas`
LIB_ZMUMPS = -lzmumps
INC_MUMPS  = $(MUMPS_HOME)/include/
INC_MUMPS_EXTRA = `$(PASTIX_HOME)/pastix-conf --incs` # for murge
ORDLIB     = -lpord

#HIPS
LIBHIPS   = -L$(HOME)/hips_trunk/trunk/LIB -lhips -lio $(LIBSCOTCH)
INCHIPS   = -I$(HOME)/hips_trunk/trunk/LIB

#WSMP
LIB_WSMP = -L$(WSMP_HOME) -lpwsmp64

# LIBFFTW
FFTWDIR ?= $(EBROOTFFTW)
LIBFFTW   = -L$(FFTWDIR)/lib -lfftw3
INC_FFTW  = -I$(FFTWDIR)/include

# LIBDIERCKX
DIERCKX_HOME ?= /path/to/Dierckx
LIBDIERCKX = -L$(DIERCKX_HOME) -ldierckx

#LAPACK
#LIBLAPACK = -L$(MKL) -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
#SCALAP    = -L$(MKL) -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 

HDF5_HOME?=$(EBROOTHDF5)
HDF5INCLUDE=$(HDF5_HOME)/include
HDF5LIB=$(HDF5_HOME)/lib/libhdf5hl_fortran.a \
        $(HDF5_HOME)/lib/libhdf5_hl.a        \
        $(HDF5_HOME)/lib/libhdf5_fortran.a   \
        $(HDF5_HOME)/lib/libhdf5.a           \
	$(EBROOTZLIB)/lib/libz.a -lz -lsz -ldl

