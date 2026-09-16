#!/bin/bash -l
# =====================================================================
#  Generic SLURM jobscript template for the physics-PC scaling study.
#
#  Copy it to job_study.local.slurm (git-ignored), add the machine's
#  partition/account/modules there, then
#
#     sbatch --export=ALL,MODE=strong job_study.local.slurm
#     sbatch --export=ALL,MODE=weak   job_study.local.slurm
#
#  The motivation study adds two more modes (see README, "Motivation figures"):
#
#     sbatch --export=ALL,MODE=nonlinear job_study.local.slurm
#     sbatch --export=ALL,MODE=probes,PCS_REF=/path/to/pc_scaling_nonlinear/jorek_<mesh>_np1x<omp> \
#            job_study.local.slurm
#
#  nonlinear and probes only need their iteration counts, so one node is
#  enough for them; strong and weak are the timed series.
#
#  JOREK runs hybrid MPI+OpenMP: set --ntasks-per-node (MPI ranks per node)
#  and --cpus-per-task (OpenMP threads per rank) so that their product is the
#  number of physical cores per node, e.g. 8 x 16 or 16 x 16 on a 128- or
#  256-core node. pc_study.sh's np counts MPI RANKS; every rank gets
#  --cpus-per-task threads (PCS_OMP overrides).
#
#  All cases of one series run one after the other inside this allocation,
#  so request enough nodes for the largest rank count of the series (larger
#  cases are skipped). Resubmitting the same job skips finished cases.
# =====================================================================
#SBATCH --job-name=pcscale
#SBATCH --output=pcscale.%j.out
#SBATCH --error=pcscale.%j.err
#SBATCH --nodes=1                   # nodes for the largest case of the series
#SBATCH --ntasks-per-node=8         # MPI ranks per node
#SBATCH --cpus-per-task=16          # OpenMP threads per rank
#SBATCH --exclusive
#SBATCH --time=12:00:00

set -u
MODE=${MODE:-strong}

# --- environment: the same compiler/MPI/PETSc stack JOREK was built with ---
# module purge
# module load ...

# --- what to run ----------------------------------------------------------
# jorek_model199 built with n_tor = 3, n_period = 1 (models/mod_settings.f90)
# and USE_PETSC; a wrong n_tor runs axisymmetric and exits 0.
export JOREK_BIN=${JOREK_BIN:-$HOME/jorek/jorek_model199}
SCRIPTS=${SCRIPTS:-$(dirname "$JOREK_BIN")/util/physics_pc_scaling}
export PCS_ROOT=${PCS_ROOT:-$PWD/pc_scaling_$MODE}
export PCS_OMP=${PCS_OMP:-${SLURM_CPUS_PER_TASK:-1}}

# Arms and series (defaults in pc_study.sh). np = MPI ranks:
# export PCS_NL_MESH=161x64; export PCS_NL_NP=16      # nonlinear trajectory
# export PCS_PROBE_ARMS="sfm2_lu_hs0 sfm2_gmg_hs0"    # probes
# export PCS_ARMS="sfm2_gmg sfm2_gmg_d12 sfm2_lu jorek"
# export PCS_MESH=161x64
# export PCS_NPS="1 2 4 8"                        # e.g. 1 node at 8 ranks/node
# export PCS_WEAK="81x32:1 161x64:4 321x128:16 641x256:64"
# Launcher (default: srun -n NP -c OMP --cpu-bind=cores):
# export PCS_LAUNCH="mpiexec -n NP ..."
# MUMPS distributed RHS, on a MUMPS build where it works (default 0 = centralized):
# export PCS_PETSC_OPTS="-mat_mumps_icntl_20 10 -fieldsplit_0_mat_mumps_icntl_20 10 -fieldsplit_1_mat_mumps_icntl_20 10"

echo "job ${SLURM_JOB_ID:-}: ${SLURM_JOB_NUM_NODES:-1} nodes, ${SLURM_NTASKS:-?} ranks x $PCS_OMP threads, mode $MODE"
"$SCRIPTS/pc_study.sh" "$MODE"
