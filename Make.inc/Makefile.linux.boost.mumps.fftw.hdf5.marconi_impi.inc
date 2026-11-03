# Makefile.inc file for compiling JOREK on the Marconi-Fusion Supercomputer
#
# * Make sure to set PASTIX_HOME below to the correct directory.
# * Make sure that you have the following modules loaded: intel intelmpi mkl fftw hdf5 szip

# --- Select physics model
MODEL             = model199

# --- Compiler and options
FC                = mpiifort -mt_mpi
CC                = mpiicc   -mt_mpi
CXX               = mpiicpc  -mt_mpi
FFLAGS_OMP        = -qopenmp 
FFLAGS            = -r8 -i4 -DUSE_BLOCK -DFUNNELED -DWORLDWAR2
FFLAGS           += -fpp
FFLAGS           += -axCORE-AVX512 -mtune=skylake -align array64byte -qopt-zmm-usage=high -vecabi=cmdtarget -mtune=skylake -mcmodel=medium -heap-arrays -fpe3



# --- Debugging options
#DEBUGFLAGS      = -g -traceback
#DEBUGFLAGS      += -warn all,nounused
#DEBUGFLAGS      += -check all,noarg_temp_created
#DEBUGFLAGS      += -debug all
#DEBUGFLAGS      += -debug-parameters
#DEBUGFLAGS      += -fstack-security-check -fp-stack-check
#DEBUGFLAGS      += -ftrapuv -check_arg_temp_created

# --- Various switches
USE_PASTIX        = 1
USE_PASTIX_MURGE  = 0
USE_HDF5          = 1
USE_FFTW          = 1
USE_MUMPS         = 1
USE_BOOST         = 1
USE_STD_BESSELK   = 0
USE_TASKLOOP      = 0

#Scotch library
SCOTCH_HOME ?= /path/to/jorek_tools/scotch_5.1.12_impi_17
LIB_SCOTCH   = -L$(SCOTCH_HOME)/lib -lptesmumps -lptscotch -lesmumps -lscotch -lptscotcherrexit -lptscotcherr -lscotcherrexit -lscotcherr
INC_SCOTCH   = -I$(SCOTCH_HOME)/include

# --- PaStiX solver
PASTIX_HOME ?= /path/to/jorek_tools/pastix_3184_impi_17/install
LIB_PASTIX        = `$(PASTIX_HOME)/install/pastix-conf --libs` 
LIB_PASTIX_MURGE  = `$(PASTIX_HOME)/install/pastix-conf --libs_murge`
LIB_PASTIX_BLAS   = `$(PASTIX_HOME)/install/pastix-conf --blas`
INC_PASTIX        = `$(PASTIX_HOME)/install/pastix-conf --incs`

# --- FFTW Library
LIBFFTW           = $(FFTW_HOME)/lib/libfftw3.a
INC_FFTW          = -I$(FFTW_HOME)/include

# --- Usually not to be changed
FFLAGS_FIXEDFORM := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)
FFLAGS_NOBOUNDS  := $(FFLAGS)               $(FFLAGS_OMP)
FFLAGS           := $(FFLAGS) $(DEBUGFLAGS) $(FFLAGS_OMP)
FFLAGS_NO_OMP     = $(FFLAGS) $(DEBUGFLAGS)

HDF5INCLUDE = $(HDF5_HOME)/include
HDF5LIB     = $(HDF5_HOME)/lib/libhdf5hl_fortran.a \
              $(HDF5_HOME)/lib/libhdf5_hl.a \
              $(HDF5_HOME)/lib/libhdf5_fortran.a \
              $(HDF5_HOME)/lib/libhdf5.a \
              $(SZIP_LIB)/libsz.a -lz

# BOOST
LIB_BOOST  = -L$(BOOST_LIB) -lboost_math_tr1
INC_BOOST  = -I$(BOOST_INC)

# --- MUMPS solver
MUMPS_HOME ?= /path/to/jorek_tools/mumps-5.0.2_impi_17
LIB_MUMPS  = -L$(MUMPS_HOME)/lib -ldmumps -lmumps_common $(LIB_SCOTCH)
INC_MUMPS  = $(MUMPS_HOME)/include $(INC_SCOTCH)
ORDLIB     = -L$(MUMPS_HOME)/PORD/lib -lpord
                                                   
SCALAP    = -L$(MKL_LIB) -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
