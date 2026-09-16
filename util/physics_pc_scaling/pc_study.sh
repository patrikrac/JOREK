#!/bin/bash
# =====================================================================
#  pc_study.sh -- a strong- or weak-scaling series of physics-PC cases,
#  run one after the other inside ONE allocation (call it from a SLURM
#  jobscript, see job_study.slurm).
#
#    pc_study.sh strong      # fixed mesh, MPI ranks np = 1 2 4 ... up to the allocation
#    pc_study.sh weak        # DOFs per rank fixed: mesh x2 per direction, np x4
#    pc_study.sh nonlinear   # one island-demo trajectory per arm: tstep ramp
#                            # 1/10/100 then PCS_NL_N steps at 1000, through the
#                            # linear growth into the saturated 2/1 island
#    pc_study.sh probes      # ONE step at tstep 1000 from each restart of a
#                            # finished nonlinear run: iteration count and
#                            # cross-|n| coupling versus the island's state,
#                            # for arms too slow to run a whole trajectory
#
#  np counts MPI ranks; each rank runs PCS_OMP OpenMP threads (default: the
#  job's --cpus-per-task), so a case uses np x PCS_OMP cores.
#    PCS_DRYRUN=1 pc_study.sh strong   # only print the cases that would run
#
#  Environment (defaults in brackets):
#    PCS_ARMS    arms to run, in order          [jorek sfm2_lu sfm2_gmg sfm2_gmg_q;
#                                                 nonlinear: jorek jorek_fresh sfm2_lu sfm2_lu_hs0 sfm2_gmg_hs0]
#    PCS_MESH    strong: the fixed mesh         [161x64]
#    PCS_NPS     strong: rank counts            [1 2 4 8 16 32 64]
#    PCS_WEAK    weak: mesh:np pairs            [81x32:1 161x64:4 321x128:16]
#                (adding 641x256:64 needs a build with larger n_nodes_max)
#    PCS_NL_MESH nonlinear: the mesh            [41x16]
#    PCS_NL_NP   nonlinear: MPI ranks           [1]
#    PCS_NL_N    nonlinear: steps at tstep 1000 [200]
#    PCS_REF     probes: the finished reference case directory
#                                               [<PCS_ROOT>/../pc_scaling_nl/jorek_<mesh>_np1x<omp>]
#    PCS_PROBE_ARMS   probes: arms              [sfm2_lu_hs0]
#    PCS_PROBE_STEPS  probes: first:stride:last [40:10:230]
#    PCS_PROBE_NP     probes: MPI ranks         [1]
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
if [ "$MODE" = nonlinear ]; then
  ARMS=${PCS_ARMS:-"jorek jorek_fresh sfm2_lu sfm2_lu_hs0 sfm2_gmg_hs0"}
else
  # jorek first: it is the cheapest, so a timed-out job still has the baseline.
  # sfm2_gmg_d12 is NOT a default: it only exists to show what the D13 axis
  # treatment bought, and that comparison is already measured.
  ARMS=${PCS_ARMS:-"jorek sfm2_lu sfm2_gmg sfm2_gmg_q"}
fi
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
    for p in ${PCS_WEAK:-81x32:1 161x64:4 321x128:16}; do
      m=${p%:*}; np=${p#*:}
      cases+=("${m%x*} ${m#*x} $np")
    done ;;
  nonlinear)
    # The committed island-demo ramp. JOREK's default PC only degrades at large
    # tstep on this case (at tstep 10 it needs 2-3 its even in saturation).
    # Restart files every 10 steps, so single phases can be re-run with
    # PCS_RESTART.
    MESH=${PCS_NL_MESH:-41x16}
    cases+=("${MESH%x*} ${MESH#*x} ${PCS_NL_NP:-1}")
    export PCS_TSTEP_N=${PCS_TSTEP_N:-1.d0,1.d1,1.d2,1.d3}
    export PCS_NSTEP_N=${PCS_NSTEP_N:-10,10,10,${PCS_NL_N:-200}}
    export PCS_NOUT=${PCS_NOUT:-10} ;;
  probes)
    # One step from each saved state of a finished nonlinear run. The physics
    # PC prints its A8 harmonic-coupling report on the first build, so each
    # probe also measures the n=0 <-> n=1 coupling at that state (harm_split=0
    # arms only). Nothing here is timed: only counts and norms are used.
    REF=${PCS_REF:-}
    if [ -z "$REF" ] || [ ! -d "$REF" ]; then
      echo "[pc_study] probes: set PCS_REF to a finished nonlinear case directory"
      exit 1
    fi
    base=$(basename "$REF")                     # <arm>_<n_flux>x<n_tht>_np<np>x<omp>
    mesh=${base#*_}; mesh=${mesh%%_*}
    NF=${mesh%x*}; NT=${mesh#*x}
    IFS=: read -r p0 dp p1 <<< "${PCS_PROBE_STEPS:-40:10:230}"
    export PCS_TSTEP_N=${PCS_TSTEP_N:-1.d3} PCS_NSTEP_N=${PCS_NSTEP_N:-1}
    export PCS_OMP=${PCS_OMP:-${SLURM_CPUS_PER_TASK:-1}}
    np=${PCS_PROBE_NP:-1}
    echo "[pc_study] probes from $REF ($NF x $NT), steps $p0:$dp:$p1, np=$np x $PCS_OMP"
    for arm in ${PCS_PROBE_ARMS:-sfm2_lu_hs0}; do
      for s in $(seq "$p0" "$dp" "$p1"); do
        f=$(printf '%s/jorek%06d.h5' "$REF" "$s")
        [ -f "$f" ] || continue
        if [ "${PCS_DRYRUN:-0}" = 1 ]; then
          echo "[pc_study] would probe: $arm from $(basename "$f")"
          continue
        fi
        PCS_RESTART=$f PCS_TAG=s$(printf '%03d' "$s") \
          "$HERE/pc_case.sh" "$arm" "$NF" "$NT" "$np"
        python3 "$HERE/collect.py" "$PCS_ROOT" > "$PCS_ROOT/results.tsv"
      done
    done
    [ "${PCS_DRYRUN:-0}" = 1 ] || echo "[pc_study] done; table: $PCS_ROOT/results.tsv"
    exit 0 ;;
  "")
    echo "usage: pc_study.sh strong|weak|nonlinear|probes   (PCS_* environment, see header)"
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
