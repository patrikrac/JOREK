#!/bin/bash
# =====================================================================
#  pc_case.sh -- run ONE physics-PC scaling case (inside a SLURM job)
#
#    pc_case.sh <arm> <n_flux> <n_tht> <np> [key=value ...]
#
#  arm     jorek | jorek_fresh | sfm2_lu | sfm2_lu_hs0 | sfm2_gmg | sfm2_gmg_* (see mknml.py)
#  np      MPI ranks for this case (<= ranks of the allocation); every rank
#          runs PCS_OMP OpenMP threads (hybrid MPI+OpenMP, as JOREK is run)
#  extra key=value pairs are passed to mknml.py as namelist overrides.
#
#  Creates  $PCS_ROOT/<arm>_<n_flux>x<n_tht>_np<np>x<omp>[_<PCS_TAG>]/  with the
#  namelist, stdout/stderr (log), -log_view profile (prof.txt) and case.meta,
#  runs the case, and prints one summary line. collect.py turns a whole
#  study into a table. An existing case directory with status ok is skipped
#  unless PCS_FORCE=1, so a job can be resubmitted after a timeout.
#
#  Environment (defaults in brackets):
#    JOREK_BIN     jorek_model199 built with n_tor=3, n_period=1  [required]
#    PCS_ROOT      study directory                     [$PWD/pc_scaling]
#    PCS_TAG       suffix for the case directory       []
#    PCS_OMP       OpenMP threads per MPI rank         [$SLURM_CPUS_PER_TASK, else 1]
#    PCS_LAUNCH    MPI launcher prefix; NP and OMP are substituted
#                  [srun -n NP -c OMP --cpu-bind=cores  inside SLURM,
#                   else mpirun -np NP]
#    PCS_PETSC_OPTS  extra PETSc options               []
#    PCS_TSTEP_N / PCS_NSTEP_N   tstep ramp            [1.d-1,1.d0,1.d1 / 3,3,3]
#    PCS_NOUT      restart/field output every N steps  [1000]
#    PCS_RESTART   restart from this jorek*.h5 file (copied in as jorek_restart.h5) []
#    PCS_FORCE     1 = rerun even if the case finished []
#    PCS_NODES_MAX this binary's compile-time n_nodes_max, for the mesh-size
#                  warning (n_flux*n_tht nodes are needed)   [60001]
# =====================================================================
set -u

if [ $# -lt 4 ]; then
  sed -n '2,30p' "$0"; exit 1
fi
ARM=$1; NF=$2; NT=$3; NP=$4; shift 4

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# n_nodes_max / n_elements_max are compile-time (models/mod_settings.f90) and
# JOREK stops in the grid generation when the mesh needs more nodes than the
# binary was built for. Warn here, where it is cheap, instead of in the job.
NODES_NEEDED=$((NF * NT))
if [ "$NODES_NEEDED" -gt "${PCS_NODES_MAX:-60001}" ]; then
  echo "[pc_case] WARNING: ${NF}x${NT} needs $NODES_NEEDED nodes, more than n_nodes_max"
  echo "[pc_case]          (assumed ${PCS_NODES_MAX:-60001}). Rebuild with larger n_nodes_max and"
  echo "[pc_case]          n_elements_max, or set PCS_NODES_MAX to this binary's value."
fi
: "${JOREK_BIN:?set JOREK_BIN to the jorek_model199 executable}"
JOREK_BIN=$(readlink -f "$JOREK_BIN" 2>/dev/null || echo "$JOREK_BIN")
PCS_ROOT=${PCS_ROOT:-$PWD/pc_scaling}
OMP=${PCS_OMP:-${SLURM_CPUS_PER_TASK:-1}}
TAG=${PCS_TAG:+_$PCS_TAG}
CASE=${ARM}_${NF}x${NT}_np${NP}x${OMP}${TAG}
DIR=$PCS_ROOT/$CASE

if [ -z "${PCS_LAUNCH:-}" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    PCS_LAUNCH="srun -n NP -c OMP --cpu-bind=cores"
  else
    PCS_LAUNCH="mpirun -np NP"
  fi
fi
LAUNCH=${PCS_LAUNCH//NP/$NP}
LAUNCH=${LAUNCH//OMP/$OMP}

if [ -f "$DIR/case.meta" ] && grep -q '^status=ok' "$DIR/case.meta" && [ "${PCS_FORCE:-0}" != 1 ]; then
  echo "[pc_case] $CASE already done, skipping (PCS_FORCE=1 to rerun)"
  exit 0
fi

mkdir -p "$DIR" || exit 1
python3 "$HERE/mknml.py" "$ARM" "$NF" "$NT" "$DIR/in" "$@" || exit 1
if [ -n "${PCS_RESTART:-}" ]; then
  cp "$PCS_RESTART" "$DIR/jorek_restart.h5" || exit 1
fi

NODES=${SLURM_JOB_NUM_NODES:-1}
{
  echo "arm=$ARM"; echo "n_flux=$NF"; echo "n_tht=$NT"; echo "np=$NP"; echo "omp=$OMP"; echo "nodes=$NODES"
  echo "overrides=$*"; echo "bin=$JOREK_BIN"; echo "launch=$LAUNCH"; echo "restart_from=${PCS_RESTART:-}"
  echo "slurm_job=${SLURM_JOB_ID:-}"; echo "host=$(hostname)"; echo "start=$(date '+%Y-%m-%dT%H:%M:%S')"
  echo "git=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
} > "$DIR/case.meta"

cd "$DIR" || exit 1
# Hybrid MPI+OpenMP: OMP threads per rank for JOREK's threaded parts (matrix
# construction) and a threaded BLAS inside MUMPS. The physics PC itself has no
# OpenMP regions; PETSc kernels run on one thread per rank.
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP
export OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export PETSC_OPTIONS="-log_view :prof.txt -memory_view ${PCS_PETSC_OPTS:-}"
T0=$(python3 -c "import time; print(time.time())")
$LAUNCH "$JOREK_BIN" < in > log 2>&1
RC=$?
T1=$(python3 -c "import time; print(time.time())")
WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

STATUS=ok
[ $RC -ne 0 ] && STATUS=exit$RC
grep -q 'NO CONVERGENCE' log && STATUS=noconv     # JOREK aborts but exits 0
grep -q 'PETSC ERROR\|FATAL' log && STATUS=error
grep -q '\[PETSc\] outer iterations' log || STATUS=${STATUS/ok/incomplete}
{ echo "rc=$RC"; echo "wall_s=$WALL"; echo "end=$(date '+%Y-%m-%dT%H:%M:%S')"; echo "status=$STATUS"; } >> case.meta

OUTER=$(grep -a '\[PETSc\] outer iterations' log | awk '{s+=$4; n++} END {print n" solves, "s" its"}')
echo "[pc_case] $CASE  status=$STATUS  wall=${WALL}s  outer: $OUTER"
exit 0
