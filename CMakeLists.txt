#=============================================================================
# CMakeLists.txt for JOREK 2
#=============================================================================
cmake_minimum_required(VERSION 3.16) # for CMake 3.16 or higher

project(Jorek
  LANGUAGES Fortran C CXX)

#=============================================================================
# --- 1. Project Options (Replaces Makefile.inc) ---
# These are the variables users will set to configure the build.
# They can be set on the command line, e.g., cmake -DMODEL=model710 ..
#=============================================================================
set(MODEL "model600" CACHE STRING "Select the JOREK model to build (e.g., model600, model199, model710, etc.)")
option(USE_MPI "Build with MPI support" ON) # From defaults.mk, almost always on

# --- Solver Options ---
option(USE_PASTIX "Enable PaSTiX solver" OFF)
option(USE_PASTIX_MURGE "Enable MURGE extensions for PaSTiX" OFF)
option(USE_MUMPS "Enable MUMPS solver" OFF)
option(USE_STRUMPACK "Enable STRUMPACK solver" OFF)
option(USE_HIPS "Enable HIPS solver" OFF)
option(USE_WSMP "Enable WSMP solver" OFF)
option(USE_BICGSTAB "Use BICGSTAB solver (otherwise GMRES)" OFF)

# --- Library and Feature Options ---
option(USE_HDF5 "Enable HDF5 for input/output" ON)
option(USE_FFTW "Enable FFTW support" ON)
option(USE_IMAS "Enable IMAS support" OFF)
option(USE_CATALYST "Enable Catalyst for in-situ visualization" OFF)
option(USE_BOOST "Enable Boost for special functions" OFF)
option(USE_STD_BESSELK "Use C++17 std::cyl_bessel_k instead of Boost" OFF)

# --- Physics & Numerics Options ---
option(USE_BLOCK "Use block matrix assembly" OFF)
option(USE_NO_TREE "Disable tree-based matrix construction" OFF)
option(USE_QUADTREE "Use quadtree-based matrix construction" OFF)
option(USE_DIRECT_CONSTRUCTION "Use direct matrix construction" OFF)
option(USE_COMPLEX_PRECOND "Use complex-valued preconditioner" OFF)
option(USE_INTSIZE64 "Use 64-bit integers for large problems" OFF)
option(USE_TASKLOOP "Use OpenMP taskloops where available" OFF)

# --- User Customization ---
set(JOREK_EXTRA_FLAGS "" CACHE STRING "Extra compiler flags for all languages")
set(JOREK_EXTRA_DEFINITIONS "" CACHE STRING "Extra preprocessor definitions (e.g., -DSAVEMATRIX)")

# --- 2. Build Setup & Compiler Configuration ---
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

# Set global compiler flags based on compiler ID and build type
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Configuring for GNU compilers (gfortran)")
  set(COMMON_FLAGS "-cpp -fopenmp -Wall -Wextra -Wno-unused-variable")
  set(FORTRAN_FLAGS "-Wintrinsics-std -Wcharacter-truncation -Wsurprising -Wno-tabs -ffree-line-length-none")
  set(FORTRAN77_FLAGS "-fdefault-real-8 -fdefault-double-8")

  set(CMAKE_C_FLAGS_DEBUG "-g -Og")
  set(CMAKE_Fortran_FLAGS_DEBUG "-g -Og -ggdb -fno-lto -fcheck=all -ffpe-trap=invalid,zero,overflow,denormal -ftrapv -finit-real=snan -finit-integer=12345678 -Wconversion -fimplicit-none")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  message(STATUS "Configuring for Intel compilers (ifort/ifx)")
  set(COMMON_FLAGS "-fpp -qopenmp") # -openmp for older versions
  set(FORTRAN_FLAGS "-align -warn all -warn nointerfaces -warn nounused -r8")
  set(FORTRAN77_FLAGS "-warn nodeclarations")

  set(CMAKE_C_FLAGS_DEBUG "-g -O0")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -traceback -debug all -debug-parameters -fstack-security-check -ftrapuv -fpe0 -check all,noarg_temp_created -check bounds -check uninit -init=snan -gen-interfaces -warn-interfaces -implicitnone")
endif()

# Apply the flags
add_compile_options(${COMMON_FLAGS} ${JOREK_EXTRA_FLAGS})
add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:${FORTRAN_FLAGS}>)
add_compile_definitions(${JOREK_EXTRA_DEFINITIONS})
# Note: F77 flags can be set per-file if needed on specific legacy .f files


# --- 3. Version Header Generation ---
# Replicates the .mod/version.h target
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev OUTPUT_VARIABLE GIT_RCS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${GIT_EXECUTABLE} log -1 "--format=%s (%D)" OUTPUT_VARIABLE GIT_RCS_LABEL OUTPUT_STRIP_TRAILING_WHITESPACE)
  # ... add other git commands as needed ...
else()
  set(GIT_RCS_VERSION "unknown")
  set(GIT_RCS_LABEL "unknown")
endif()

# Create a template file "version.h.in" in your source directory with:
# #define RCS_VERSION "@GIT_RCS_VERSION@"
# #define RCS_LABEL "@GIT_RCS_LABEL@"
configure_file(version.h.in ${CMAKE_BINARY_DIR}/version.h)
include_directories(${CMAKE_BINARY_DIR}) # So the compiler can find version.h


# --- 4. Code Generation ---
# Replicates the 'generate_code' target which runs 'algexpr2fort'
# Assuming 'algexpr2fort' is an executable available in the source tree root
add_executable(algexpr2fort_tool IMPORTED GLOBAL)
set_property(TARGET algexpr2fort_tool PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/algexpr2fort)

set(GENERATED_RHS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/models/${MODEL}/rhs_automatic.h)
set(GENERATED_AMAT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/models/${MODEL}/amat_automatic.h)

add_custom_command(
    OUTPUT ${GENERATED_RHS_FILE} ${GENERATED_AMAT_FILE}
    COMMAND ${CMAKE_COMMAND} -E touch ${GENERATED_RHS_FILE} ${GENERATED_AMAT_FILE} # Placeholder, as algexpr2fort seems to modify files in place
    COMMAND $<TARGET_FILE:algexpr2fort_tool>
    COMMENT "Generating symbolic expressions for model ${MODEL}"
    DEPENDS algexpr2fort_tool
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(generate_code ALL DEPENDS ${GENERATED_RHS_FILE} ${GENERATED_AMAT_FILE})


# --- 5. Find Dependencies ---
# This section robustly finds all required and optional libraries.
# Users should point to library locations using -DCMAKE_PREFIX_PATH=/path/to/libs

if(USE_MPI)
  find_package(MPI REQUIRED)
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(USE_HDF5)
  find_package(HDF5 COMPONENTS Fortran C REQUIRED)
endif()
if(USE_FFTW)
  find_package(FFTW REQUIRED)
endif()

# --- Advanced Library Finding for Solvers ---
if(USE_MUMPS OR USE_PASTIX OR USE_STRUMPACK)
  # MKL is a common dependency. Finding it properly is key.
  find_package(MKL QUIET) # Try to find an official FindMKL module
  if(NOT MKL_FOUND) # Fallback to manual search if not found
      message(STATUS "Searching for MKL manually...")
      find_path(MKL_INCLUDE_DIR mkl.h HINTS ENV MKLROOT PATH_SUFFIXES include)
      find_library(MKL_CORE_LIB NAMES mkl_core HINTS ENV MKLROOT PATH_SUFFIXES lib/intel64 lib)
      if(MKL_INCLUDE_DIR AND MKL_CORE_LIB)
          # This is a simplified list; a full FindMKL.cmake is complex.
          # We assemble the library list based on what was in Makefile.inc
          find_library(MKL_SCALAPACK_LIB NAMES mkl_scalapack_lp64 HINTS ENV MKLROOT PATH_SUFFIXES lib/intel64)
          find_library(MKL_BLACS_LIB NAMES mkl_blacs_intelmpi_lp64 HINTS ENV MKLROOT PATH_SUFFIXES lib/intel64)
          find_library(MKL_INTEL_LIB NAMES mkl_intel_lp64 HINTS ENV MKLROOT PATH_SUFFIXES lib/intel64)
          find_library(MKL_SEQ_LIB NAMES mkl_sequential HINTS ENV MKLROOT PATH_SUFFIXES lib/intel64)
          set(MKL_LIBRARIES ${MKL_SCALAPACK_LIB} ${MKL_BLACS_LIB} ${MKL_INTEL_LIB} ${MKL_SEQ_LIB} ${MKL_CORE_LIB})
          set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
          message(STATUS "Found MKL manually.")
      else()
          message(FATAL_ERROR "MKL not found. Please set MKLROOT environment variable or add its path to CMAKE_PREFIX_PATH.")
      endif()
  endif()
endif()

if(USE_MUMPS)
  find_path(MUMPS_INCLUDE_DIR dmutools.h HINTS ENV MUMPS_HOME PATH_SUFFIXES include)
  find_library(MUMPS_LIB NAMES dmumps HINTS ENV MUMPS_HOME PATH_SUFFIXES lib)
  if(NOT MUMPS_INCLUDE_DIR OR NOT MUMPS_LIB)
      message(FATAL_ERROR "MUMPS not found. Set MUMPS_HOME or add to CMAKE_PREFIX_PATH.")
  endif()
  set(MUMPS_LIBRARIES ${MUMPS_LIB} -lmumps_common -lpord ${MKL_LIBRARIES})
  set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR} ${MKL_INCLUDE_DIRS})
  message(STATUS "Configured MUMPS support.")
endif()
# Add similar blocks for PASTIX, STRUMPACK, HIPS, WSMP...


# --- 6. Define Source Files ---
set(JOREK_SOURCE_DIRS
    . core diagnostics communication grids solvers models refinement matrix
    particles elements plots tools vacuum numerics
    # Add subdirectories
    core/tests diagnostics/tests communication/IMAS grids/grid_utils
    models/${MODEL}
    # ... add all other directories from the Makefile's DIRS variable
)

file(GLOB_RECURSE ALL_SOURCES
    CONFIGURE_DEPENDS # Re-run cmake if new files are added/removed
    ${JOREK_SOURCE_DIRS}/*.f90
    ${JOREK_SOURCE_DIRS}/*.f
    ${JOREK_SOURCE_DIRS}/*.c
    ${JOREK_SOURCE_DIRS}/*.cpp
)

# Exclude main drivers from the common object library
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*jorek2_main\\.f90$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*jorek2_diagno\\.f90$")
# ... filter out all other main program files ...
set(LIBRARY_SOURCES ${ALL_SOURCES})


# --- 7. Define Targets (Library and Executables) ---

# Create a static library from all common source files. This is efficient
# as sources are only compiled once.
add_library(jorek_objects STATIC ${LIBRARY_SOURCES})

# Ensure code is generated before the library is compiled
add_dependencies(jorek_objects generate_code)

# --- Configure the Library (Definitions, Includes, Links) ---
string(SUBSTRING ${MODEL} 5 -1 MODEL_NUMBER) # "model303" -> "303"

target_compile_definitions(jorek_objects PUBLIC
    $<$<BOOL:${USE_MPI}>:-DUSE_MPI>
    -DJOREK_MODEL=${MODEL_NUMBER}
    # Add definitions for all USE_* options
    $<$<BOOL:${USE_HDF5}>:-DUSE_HDF5>
    $<$<BOOL:${USE_FFTW}>:-DUSE_FFTW>
    $<$<BOOL:${USE_MUMPS}>:-DUSE_MUMPS>
    $<$<BOOL:${USE_PASTIX}>:-DUSE_PASTIX>
    $<$<BOOL:${USE_PASTIX_MURGE}>:-DUSE_MURGE>
    $<$<BOOL:${USE_STRUMPACK}>:-DUSE_STRUMPACK -DNEWSPK>
    $<$<BOOL:${USE_BLOCK}>:-DUSE_BLOCK>
    $<$<BOOL:${USE_BICGSTAB}>:-DUSE_BICGSTAB>
    $<$<NOT:$<BOOL:${USE_BICGSTAB}>>:-DUSE_GMRES>
    # Add other model-specific or feature flags
    $<$<STREQUAL:${MODEL},model180>:-DSEMIANALYTICAL -DSTELLARATOR_MODEL>
)

target_include_directories(jorek_objects PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    # Add include directories from dependencies
    $<$<BOOL:${USE_MUMPS}>:${MUMPS_INCLUDE_DIRS}>
)

target_link_libraries(jorek_objects PUBLIC
    # Core libraries
    $<$<BOOL:${USE_MPI}>:MPI::MPI_Fortran>
    LAPACK::LAPACK BLAS::BLAS
    # Optional libraries
    $<$<BOOL:${USE_HDF5}>:HDF5::HDF5_Fortran>
    $<$<BOOL:${USE_FFTW}>:FFTW::FFTW>
    $<$<BOOL:${USE_MUMPS}>:${MUMPS_LIBRARIES}>
    # Link to C++ std library if needed by Boost, etc.
    $<$<OR:$<BOOL:${USE_BOOST}>,$<BOOL:${USE_STD_BESSELK}>>:${CMAKE_CXX_STANDARD_LIBRARIES}>
)


# --- Define Executable Targets ---

# --- Main Model Executable ---
add_executable(jorek_model${MODEL_NUMBER} jorek2_main.f90)
target_link_libraries(jorek_model${MODEL_NUMBER} PRIVATE jorek_objects)

# --- Helper function to define other tool/diagnostic executables ---
function(add_jorek_executable name)
    # Find the source file for the executable in any of our source dirs
    find_file(EXE_SOURCE_PATH ${name}.f90 PATHS ${JOREK_SOURCE_DIRS} NO_DEFAULT_PATH)
    if(NOT EXE_SOURCE_PATH)
        message(WARNING "Could not find source for executable '${name}'")
        return()
    endif()
    add_executable(${name} ${EXE_SOURCE_PATH})
    target_link_libraries(${name} PRIVATE jorek_objects)
    # Install the executable
    install(TARGETS ${name} DESTINATION bin)
endfunction()

# --- Build the list of standard executables (from 'most' target in Makefile) ---
# add_jorek_executable(jorek2_diagno)
# add_jorek_executable(jorek2_poincare)
# add_jorek_executable(jorek2vtk)
# add_jorek_executable(jorek2_connection2)
# add_jorek_executable(jorek2_fieldlines_vtk)
# add_jorek_executable(jorek2_four)
# add_jorek_executable(jorek2_IDS)
# add_jorek_executable(jorek2_powers)
# add_jorek_executable(jorek2_target2vtk)
# add_jorek_executable(rst_bin2hdf5)
# add_jorek_executable(rst_hdf52bin)
# ... add all other executables here ...


# --- 9. Final Configuration Summary ---
message(STATUS "--- JOREK Build Configuration Summary ---")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Fortran Compiler: ${CMAKE_Fortran_COMPILER_ID} (${CMAKE_Fortran_COMPILER})")
message(STATUS "Model:            ${MODEL} (Executable: jorek_model${MODEL_NUMBER})")
message(STATUS "--- Enabled Features & Libraries ---")
if(USE_HDF5)
  message(STATUS "  [X] HDF5 support")
endif()
if(USE_FFTW)
  message(STATUS "  [X] FFTW support")
endif()
if(USE_MUMPS)
  message(STATUS "  [X] MUMPS solver")
endif()
if(USE_PASTIX)
  message(STATUS "  [X] PaSTiX solver")
endif()
if(USE_STRUMPACK)
  message(STATUS "  [X] STRUMPACK solver")
endif()
message(STATUS "-----------------------------------------")