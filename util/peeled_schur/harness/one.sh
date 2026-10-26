#!/bin/bash
# one.sh <name> <petsc options...>: 41x32 tstep 10, 3 steps, peel; one summary line
cd $(dirname $0)/..
export OMP_NUM_THREADS=4
n=$1; shift
PETSC_OPTIONS="-log_view :lv.txt $*" tools/run.sh r/o_$n 1 tstep_n=1.d1 nstep_n=3 gmres_max_iter=60 physics_pc_sf_pair_w='"peel"' > /dev/null
L=r/o_$n/log
echo "$n | $* | its $(grep -o 'outer iterations: *[0-9]*' $L | awk '{print $3}' | tr '\n' ' ')| pw $(grep -o 'inner pair_w KSP[^:]*: mean *[0-9.]*' $L | awk '{print $NF}' | tr '\n' ' ')| pw_s $(grep -o 'solves, *[0-9.]* s' $L | awk '{s+=$2} END {print s}') | setup $(grep -a 'peel setup' $L | head -1 | awk '{print $(NF-6),$(NF-5),$(NF-4),$(NF-3)}') nnz $(grep -ao 'nonzeros per row *[0-9.]*' $L | head -1 | awk '{print $NF}') | N_vc $(grep -E '^GMG7_VCycle' r/o_$n/lv.txt | awk '{printf "%d x %.1f ms", $2, 1000*$4/$2}') | real $(grep '^real' $L | awk '{print $2}')"
rm -f r/o_$n/jorek r/o_$n/*.ps
