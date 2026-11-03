# Configuration file for jorek (IBM Power system)
#
# USE gmake INSTEAD OF make!
#
MODEL = model199

# Fortran compiler 
FC  = mpxlf_r
CC  = mpxlc_r
CXX = mpxlc++_r

IBMFC = 1
CPP = /bin/cpp
SED = gsed
AWK = gawk

#FFLAGS           = -q64 -qalign -qautodbl=dbl4 -qthreaded -qarch=auto -qcache=auto -qnosave \
#  -qflag=i:i -qfree=f90 -qsuffix=cpp=f90 -qessl -I$(PASTIXDIR)/install \
#  -qtbtable=full -qflttrap=inv:ov:und:zero:nanq:en -qsigtrap=xl__ieee -qcheck -g -qdpcl \
#  -qfullpath -qoptdebug -qwarn64 -qextchk -qinit=f90ptr -qinitauto=FF
FFLAGS           = -O3 -g -q64 -qautodbl=dbl4 -qfree=f90 -qsuffix=cpp=f90
FFLAGS_OMP       = -g -qsmp=noauto -qsmp=omp

FFLAGS_FIXEDFORM = $(FFLAGS) -qfixed  $(FFLAGS_OMP) 
FFLAGS_NOBOUNDS  = $(FFLAGS) -qfixed  $(FFLAGS_OMP) 
FFLAGS          := $(FFLAGS) $(FFLAGS_OMP) 
FFLAGS          := $(FFLAGS) -DMURGE_INTERFACE_MAJOR_VERSION=1 -DMURGE_INTERFACE_MINOR_VERSION=0

# Solvers dependencies
USE_HIPS   = 0
USE_PASTIX = 1
USE_PASTIX_MURGE = 1
USE_MUMPS = 0
USE_WSMP   = 0

LIBDIR ?= /path/to/mylib/rs_aix61

#Scotch library
SCOTCH_HOME ?= $(LIBDIR)/scotch_5.1_esmumps
LIB_SCOTCH   = -L$(SCOTCH_HOME)/lib -lptscotch -lptscotcherrexit -lesmumps
INC_SCOTCH   = -I$(SCOTCH_HOME)/include

# MUMPS
MUMPS_HOME = $(LIBDIR)/MUMPS_4.9.2
LIB_MUMPS  = -L$(MUMPS_HOME)/lib/ -ldmumps -lmumps_common
LIB_ZMUMPS = -L$(MUMPS_HOME)/lib/ -lzmumps
INC_MUMPS  = $(MUMPS_HOME)/include/
ORDLIB     = $(MUMPS_HOME)/lib/libpord.a -L$(LIBDIR)/metis-4.0 -lmetis

# PASTIX
PASTIX_HOME     ?= $(LIBDIR)/pastix_release_3184
LIB_PASTIX       = `$(PASTIX_HOME)/install/pastix-conf --libs` 
LIB_PASTIX_MURGE = `$(PASTIX_HOME)/install/pastix-conf --libs_murge`
LIB_PASTIX_BLAS  = `$(PASTIX_HOME)/install/pastix-conf --blas`
INC_PASTIX       = `$(PASTIX_HOME)/install/pastix-conf --incs`

#HIPS
LIBHIPS   = ###
INCHIPS   = ###

#WSMP
LIB_WSMP = -L$(WSMP_HOME) -lpwsmp64

# LIBFFTW
FFTW_LIBDIR ?= /path/to/fftw/lib
LIBFFTW   = -L$(FFTW_LIBDIR) -lfftw

LIBLAPACK = -lessl
SCALAPACK_LIBDIR ?= /path/to/scalapack/lib
SCALAP    = -lessl -L $(SCALAPACK_LIBDIR)  $(SCALAPACK_LIBDIR)/blacsF77init_MPI-POWER4-0.a $(SCALAPACK_LIBDIR)/blacs_MPI-POWER4-0.a $(SCALAPACK_LIBDIR)/libscalapack_old.a
BLACS_HOME = 
BLACS      = 
