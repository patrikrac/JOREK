# Configuration file for jorek
# Version for ITER regression testing platform using intel compilers

# model directory
MODEL = model199

# Fortran compiler 
FC               = mpiifort
FFLAGS          := $(FFLAGS) -mcmodel=medium -DUSE_BLOCK
CC = mpiicc
CXX = mpiicpc

# Solvers dependencies
USE_HIPS                = 0
USE_PASTIX              = 0
USE_BLOCK               = 0
USE_MUMPS               = 1
USE_STRUMPACK           = 1
USE_FFTW                = 1
USE_HDF5                = 1
USE_DIRECT_CONSTRUCTION = 1

# Directory containing the user-compiled libraries (e.g., MUMPS, PASTIX, ...)
LIBDIR = $(HOME)/lib

#Scotch library
SCOTCH_HOME ?=  $(LIBDIR)/scotch/scotch_5.1
LIB_SCOTCH   = -L$(SCOTCH_HOME)/lib -lptscotch -lptscotcherr -lscotch
INC_SCOTCH   = -I$(SCOTCH_HOME)/include

# MUMPS
MUMPS_HOME ?=  $(LIBDIR)/MUMPS_4.8.3
ORDLIB      = -L$(METIS_HOME)/lib -lpord -lmetis -lparmetis
LIB_MUMPS   = $(LIB_SCOTCH) -L$(MUMPS_HOME)/lib/ -ldmumps -lmumps_common -lesmumps $(LIB_SCOTCH) $(ORDLIB)
LIB_ZMUMPS  = -lzmumps
INC_MUMPS   =  $(MUMPS_HOME)/include/

#Strumpack library
STRUMPACK_HOME ?= /path/to/STRUMPACK-3.2.0/build
STRUMPACKINC = -I$(METIS_HOME)/include -I$(STRUMPACK_HOME)/include 
STRUMPACKLIB = -L$(STRUMPACK_HOME)/lib/ -lstrumpack $(METIS_HOME)/lib/libmetis.a -lmpifort -lmpi -lmpigi -ldl -lrt -lpthread -lifport -lifcoremt -limf -lsvml -lm -lipgo -lirc -lpthread -lsvml -lc -lirc_s -ldl -lc
MKLINC= -I${MKLROOT}/include
MKLLIB= ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lm -ldl
STRUMPACKINC += $(MKLINC)
STRUMPACKLIB += $(MKLLIB)

#FFTW
LIBFFTW   = -L$(FFTWDIR)/lib -lfftw3
INC_FFTW  = -I$(FFTWDIR)/include

#LAPACK
MKL       = $(MKLROOT)/lib/intel64
LIBLAPACK = -L$(MKL) -lmkl_lapack95_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -limf
SCALAP    = -L$(MKL) -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 

#Dierckx
LIBDIERCKX = $(LIBDIR)/Dierckx/dierckx.a

#HDF5
HDF5_HOME ?= $(H5DIR)
HDF5INCLUDE=$(HDF5_HOME)/include
HDF5LIB=-lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz
