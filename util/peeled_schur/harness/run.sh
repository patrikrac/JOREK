#!/bin/bash
# run.sh <outdir> <np> [key=value ...]   PETSC_OPTIONS from the environment
S=$(cd $(dirname "$0")/.. && pwd)
d=$1; np=$2; shift 2
rm -rf "$d"; mkdir -p "$d"
cp $S/base_stripped.in "$d/in"
cp $S/eq/jorek_restart.h5 "$d/jorek_restart.h5"
set -- restart=.t. "$@"
python3 -I - "$d/in" "$@" <<'PY'
import sys
p, add = sys.argv[1], sys.argv[2:]
kv = dict(a.split('=', 1) for a in add)
L = open(p).read().split('\n')
out = [l for l in L if not (l.strip() and not l.strip().startswith('!') and '=' in l and l.split('=')[0].strip() in kv)]
k = max(i for i, l in enumerate(out) if l.strip() == '/')
open(p, 'w').write('\n'.join(out[:k] + [' %s = %s' % kv_ for kv_ in kv.items()] + out[k:]))
PY
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cp ${BIN:-/Users/racpat/local/code/JOREK_peel/jorek_model199} "$d/jorek"
( cd "$d" && /usr/bin/time -p mpirun -n $np ./jorek < in > log 2>&1; echo "exit=$?" >> log )
echo "$(basename $d): $(tail -1 $d/log)  its: $(grep -o 'outer iterations: *[0-9]*' $d/log | awk '{print $3}' | tr '\n' ' ') pair_w: $(grep -o 'inner pair_w KSP[^:]*: mean *[0-9.]*' $d/log | awk '{print $NF}' | tr '\n' ' ') real $(grep '^real' $d/log | awk '{print $2}')"
