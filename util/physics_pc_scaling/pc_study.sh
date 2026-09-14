#!/bin/bash
# =====================================================================
#  pc_study.sh -- a strong- or weak-scaling series of physics-PC cases,
#  run one after the other inside ONE allocation (call it from a SLURM
#  jobscript, see job_study.slurm).
#
#    pc_study.sh strong      # fixed mesh, MPI ranks np = 1 2 4 ... up to the allocation
#    pc_study.sh weak        # DOFs per rank fixed: mesh x2 per direction, np x4
#
#  np counts MPI ranks; each rank runs PCS_OMP OpenMP threads (default: the
#  job's --cpus-per-task), so a case uses np x PCS_OMP cores.
#    PCS_DRYRUN=1 pc_study.sh strong   # only print the cases that would run
#
#  Environment (defaults in brackets):
#    PCS_ARMS    arms to run, in order          [sfm2_gmg sfm2_lu jorek]
#    PCS_MESH    strong: the fixed mesh         [161x64]
#    PCS_NPS     strong: rank counts            [1 2 4 8 16 32 64]
#    PCS_WEAK    weak: mesh:np pairs            [81x32:1 161x64:4 321x128:16 641x256:64]
#    PCS_MAXNP   upper limit on np (MPI ranks)  [$SLURM_NTASKS, else 8]
#    PCS_OMP     OpenMP threads per rank        [$SLURM_CPUS_PER_TASK, else 1]
#    PCS_DRYRUN  1 = list the cases, run nothing []
#  plus everything pc_case.sh reads (JOREK_BIN, PCS_ROOT, PCS_TAG, PCS_OMP, PCS_LAUNCH,
#  PCS_PETSC_OPTS, PCS_TSTEP_N, PCS_NSTEP_N, PCS_FORCE).
#
#  Cases larger than the allocation are skipped with a message. Finished
#  cases are skipped on a resubmission (pc_case.sh), so a timed-out study
#  can simply be submitted again. The table is refreshed after every case:
#  $PCS_ROOT/results.tsv.
# =====================================================================
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-}
ARMS=${PCS_ARMS:-"sfm2_gmg sfm2_lu jorek"}
MAXNP=${PCS_MAXNP:-${SLURM_NTASKS:-8}}
export PCS_ROOT=${PCS_ROOT:-$PWD/pc_scaling}
[ "${PCS_DRYRUN:-0}" = 1 ] || mkdir -p "$PCS_ROOT"

cases=()
case "$MODE" in
  strong)
    MESH=${PCS_MESH:-161x64}
    for np in ${PCS_NPS:-1 2 4 8 16 32 64}; do
      cases+=("${MESH%x*} ${MESH#*x} $np")
    done ;;
  weak)
    for p in ${PCS_WEAK:-81x32:1 161x64:4 321x128:16 641x256:64}; do
      m=${p%:*}; np=${p#*:}
      cases+=("${m%x*} ${m#*x} $np")
    done ;;
  "")
    echo "usage: pc_study.sh strong|weak   (PCS_* environment, see header)"
    exit 1 ;;
  *)
    echo "unknown mode '$MODE'"; exit 1 ;;
esac

export PCS_OMP=${PCS_OMP:-${SLURM_CPUS_PER_TASK:-1}}
echo "[pc_study] mode=$MODE arms='$ARMS' maxnp=$MAXNP omp/rank=$PCS_OMP root=$PCS_ROOT"
for arm in $ARMS; do
  for c in "${cases[@]}"; do
    set -- $c
    if [ "$3" -gt "$MAXNP" ]; then
      echo "[pc_study] skip $arm $1x$2 np=$3 (allocation has $MAXNP ranks)"
      continue
    fi
    if [ "${PCS_DRYRUN:-0}" = 1 ]; then
      echo "[pc_study] would run: $arm $1x$2 np=$3 x $PCS_OMP threads"
      continue
    fi
    "$HERE/pc_case.sh" "$arm" "$1" "$2" "$3"
    python3 "$HERE/collect.py" "$PCS_ROOT" > "$PCS_ROOT/results.tsv"
  done
done
[ "${PCS_DRYRUN:-0}" = 1 ] || echo "[pc_study] done; table: $PCS_ROOT/results.tsv"
