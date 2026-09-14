#!/bin/bash
# =====================================================================
#  pc_case.sh -- run ONE physics-PC scaling case (inside a SLURM job)
#
#    pc_case.sh <arm> <n_flux> <n_tht> <np> [key=value ...]
#
#  arm     jorek | sfm2_lu | sfm2_gmg   (see mknml.py)
#  np      MPI ranks for this case (<= ranks of the allocation)
#  extra key=value pairs are passed to mknml.py as namelist overrides.
#
#  Creates  $PCS_ROOT/<arm>_<n_flux>x<n_tht>_np<np>[_<PCS_TAG>]/  with the
#  namelist, stdout/stderr (log), -log_view profile (prof.txt) and case.meta,
#  runs the case, and prints one summary line. collect.py turns a whole
#  study into a table. An existing case directory with status ok is skipped
#  unless PCS_FORCE=1, so a job can be resubmitted after a timeout.
#
#  Environment (defaults in brackets):
#    JOREK_BIN     jorek_model199 built with n_tor=3, n_period=1  [required]
#    PCS_ROOT      study directory                     [$PWD/pc_scaling]
#    PCS_TAG       suffix for the case directory       []
#    PCS_LAUNCH    MPI launcher prefix; NP is substituted
#                  [srun -n NP --cpu-bind=cores  inside SLURM, else mpirun -np NP]
#    PCS_PETSC_OPTS  extra PETSc options               []
#    PCS_TSTEP_N / PCS_NSTEP_N   tstep ramp            [1.d-1,1.d0,1.d1 / 3,3,3]
#    PCS_FORCE     1 = rerun even if the case finished []
# =====================================================================
set -u

if [ $# -lt 4 ]; then
  sed -n '2,30p' "$0"; exit 1
fi
ARM=$1; NF=$2; NT=$3; NP=$4; shift 4

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${JOREK_BIN:?set JOREK_BIN to the jorek_model199 executable}"
JOREK_BIN=$(readlink -f "$JOREK_BIN" 2>/dev/null || echo "$JOREK_BIN")
PCS_ROOT=${PCS_ROOT:-$PWD/pc_scaling}
TAG=${PCS_TAG:+_$PCS_TAG}
CASE=${ARM}_${NF}x${NT}_np${NP}${TAG}
DIR=$PCS_ROOT/$CASE

if [ -z "${PCS_LAUNCH:-}" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    PCS_LAUNCH="srun -n NP --cpu-bind=cores"
  else
    PCS_LAUNCH="mpirun -np NP"
  fi
fi
LAUNCH=${PCS_LAUNCH//NP/$NP}

if [ -f "$DIR/case.meta" ] && grep -q '^status=ok' "$DIR/case.meta" && [ "${PCS_FORCE:-0}" != 1 ]; then
  echo "[pc_case] $CASE already done, skipping (PCS_FORCE=1 to rerun)"
  exit 0
fi

mkdir -p "$DIR" || exit 1
python3 "$HERE/mknml.py" "$ARM" "$NF" "$NT" "$DIR/in" "$@" || exit 1

NODES=${SLURM_JOB_NUM_NODES:-1}
{
  echo "arm=$ARM"; echo "n_flux=$NF"; echo "n_tht=$NT"; echo "np=$NP"; echo "nodes=$NODES"
  echo "overrides=$*"; echo "bin=$JOREK_BIN"; echo "launch=$LAUNCH"
  echo "slurm_job=${SLURM_JOB_ID:-}"; echo "host=$(hostname)"; echo "start=$(date '+%Y-%m-%dT%H:%M:%S')"
  echo "git=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
} > "$DIR/case.meta"

cd "$DIR" || exit 1
# One PETSc/MUMPS thread per rank: the physics PC has no OpenMP, and a
# threaded BLAS inside MUMPS would oversubscribe the cores srun gave us.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PETSC_OPTIONS="-log_view :prof.txt -memory_view ${PCS_PETSC_OPTS:-}"
T0=$(python3 -c "import time; print(time.time())")
$LAUNCH "$JOREK_BIN" < in > log 2>&1
RC=$?
T1=$(python3 -c "import time; print(time.time())")
WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

STATUS=ok
[ $RC -ne 0 ] && STATUS=exit$RC
grep -q 'PETSC ERROR\|FATAL' log && STATUS=error
grep -q '\[PETSc\] outer iterations' log || STATUS=${STATUS/ok/incomplete}
{ echo "rc=$RC"; echo "wall_s=$WALL"; echo "end=$(date '+%Y-%m-%dT%H:%M:%S')"; echo "status=$STATUS"; } >> case.meta

OUTER=$(grep -a '\[PETSc\] outer iterations' log | awk '{s+=$4; n++} END {print n" solves, "s" its"}')
echo "[pc_case] $CASE  status=$STATUS  wall=${WALL}s  outer: $OUTER"
exit 0
