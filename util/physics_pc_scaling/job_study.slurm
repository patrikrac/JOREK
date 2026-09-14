#!/bin/bash -l
# =====================================================================
#  Example SLURM jobscript for the physics-PC scaling study on TOK.
#  Adapt the #SBATCH lines and the module block to the cluster, then
#
#     sbatch --export=ALL,MODE=strong job_study.slurm
#     sbatch --export=ALL,MODE=weak   job_study.slurm
#
#  All cases of one series run one after the other inside this allocation,
#  so request the LARGEST rank count of the series (np above the allocation
#  is skipped). Nodes are taken exclusively so the timings are clean.
#  Resubmitting the same job skips cases that already finished.
# =====================================================================
#SBATCH --job-name=pcscale
#SBATCH --output=pcscale.%j.out
#SBATCH --error=pcscale.%j.err
#SBATCH --nodes=2                   # <-- nodes for the largest case
#SBATCH --ntasks-per-node=32        # <-- physical cores per node (no SMT)
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=12:00:00             # <-- partition limit
##SBATCH --partition=<partition>    # <-- TOK partition
##SBATCH --account=<account>
##SBATCH --mem=0                    # whole node memory (the LU arm needs it)

set -u
MODE=${MODE:-strong}

# --- environment: the SAME compiler/MPI/PETSc stack JOREK was built with ----
# module purge
# module load intel/xx impi/xx petsc/3.24   # <-- adapt
# export PETSC_DIR=... PETSC_ARCH=...

# --- what to run ----------------------------------------------------------
# Absolute path to a jorek_model199 built with n_tor = 3, n_period = 1
# (models/mod_settings.f90) and USE_PETSC; a wrong n_tor runs axisymmetric
# and exits 0.
export JOREK_BIN=${JOREK_BIN:-$HOME/jorek/jorek_model199}
SCRIPTS=${SCRIPTS:-$(dirname "$JOREK_BIN")/util/physics_pc_scaling}
export PCS_ROOT=${PCS_ROOT:-$PWD/pc_scaling_$MODE}

# Arms and series (defaults in pc_study.sh):
# export PCS_ARMS="sfm2_gmg sfm2_lu jorek"
# export PCS_MESH=161x64                          # strong scaling mesh
# export PCS_NPS="1 2 4 8 16 32 64"
# export PCS_WEAK="81x32:1 161x64:4 321x128:16 641x256:64"
# Launcher: srun is the default inside SLURM; with Intel MPI and no PMI:
# export PCS_LAUNCH="mpiexec.hydra -n NP -ppn 32"
# Test the MUMPS distributed RHS on a fixed MUMPS build (default 0 = centralized):
# export PCS_PETSC_OPTS="-mat_mumps_icntl_20 10 -fieldsplit_0_mat_mumps_icntl_20 10 -fieldsplit_1_mat_mumps_icntl_20 10"

echo "job $SLURM_JOB_ID: $SLURM_JOB_NUM_NODES nodes, $SLURM_NTASKS tasks, mode $MODE"
"$SCRIPTS/pc_study.sh" "$MODE"
