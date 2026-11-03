# Confiduration file for jorek
# model directory

# To get ready to compile or execute, please load the following modules :
#       module load mpich2/gcc46/1.4.1 ATLAS/gcc46/3.8.4

MODEL = model303

# Fortran compiler 
FC     = mpif90 -f90=gfortran
CC     = mpicc -cc=gcc 
CXX    = mpicxx -cxx=g++

FFLAGS_OMP       = -fopenmp 
FFLAGS           = -g -O2 -DFUNNELED
FFLAGS          := $(FFLAGS) -fdefault-real-8 -fdefault-double-8
FFLAGS          := $(FFLAGS) -ffree-line-length-none
FFLAGS          := $(FFLAGS) -x f95-cpp-input
## TODO find equivalents of this ifort options
#FFLAGS          := $(FFLAGS) -traceback -xT -align 

FFLAGS_FIXEDFORM = $(FFLAGS) -ffixed-form  -fbounds-check $(FFLAGS_OMP) 
FFLAGS_NOBOUNDS  = $(FFLAGS) -ffixed-form                 $(FFLAGS_OMP) 
FFLAGS          := $(FFLAGS)               -fbounds-check $(FFLAGS_OMP) 

#FFLAGS_NO_OMP   = -fpp -r8 -i4 -O2 -xT -align
FFLAGS          := $(FFLAGS) -DMURGE_INTERFACE_MAJOR_VERSION=1 -DMURGE_INTERFACE_MINOR_VERSION=0

# Solvers dependencies
USE_HIPS   = 0
USE_PASTIX = 1
USE_PASTIX_MURGE = 0
USE_MUMPS = 0
USE_WSMP   = 0

# PASTIX
PASTIX_HOME     ?= /path/to/jorek_tools/gpastix_release_3184/install
LIB_PASTIX       = `$(PASTIX_HOME)/pastix-conf --libs` 
LIB_PASTIX_MURGE = `$(PASTIX_HOME)/pastix-conf --libs_murge`
LIB_PASTIX_BLAS  = `$(PASTIX_HOME)/pastix-conf --blas`
INC_PASTIX       = `$(PASTIX_HOME)/pastix-conf --incs`

