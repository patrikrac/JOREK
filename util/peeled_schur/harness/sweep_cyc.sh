#!/bin/bash
# V-cycle sweep of the peel arm, 41x32 tstep 10, np 1 x 4
cd $(dirname $0)/..
export OMP_NUM_THREADS=4
P='physics_pc_sf_pair_w="peel"'
R() { n=$1; shift; PETSC_OPTIONS="-log_view :lv.txt $*" tools/run.sh r/cy_$n 1 tstep_n=1.d1 nstep_n=3 "$P" > /dev/null
  echo "$n | $* | $(grep -o 'outer iterations: *[0-9]*' r/cy_$n/log | awk '{print $3}' | tr '\n' ' ') | pw $(grep -o 'inner pair_w KSP[^:]*: mean *[0-9.]*' r/cy_$n/log | awk '{print $NF}' | tr '\n' ' ') | pw_s $(grep -o 'solves, *[0-9.]* s' r/cy_$n/log | awk '{s+=$2} END {print s}') | solve_s $(grep -a 'Elapsed time in solve ' r/cy_$n/log | awk '{s+=$NF} END {print s}') | $(grep '^real' r/cy_$n/log)"
  rm -f r/cy_$n/jorek r/cy_$n/*.ps; }
R n11    -sf_gmg7_pre0 1 -sf_gmg7_post0 1 -sf_gmg7_nsc 1
R n02    -sf_gmg7_pre0 0 -sf_gmg7_post0 2 -sf_gmg7_nsc 1
R n22    -sf_gmg7_pre0 2 -sf_gmg7_post0 2 -sf_gmg7_nsc 2
R pj04   -sf_gmg8_post0 4
R pj02   -sf_gmg8_post0 2 -sf_gmg8_nsc 1
R q11    -sf_gmg6_pre0 1 -sf_gmg6_post0 1 -sf_gmg6_nsc 1
R tri    -sf_peel_tri 1
R all1   -sf_gmg7_pre0 1 -sf_gmg7_post0 1 -sf_gmg7_nsc 1 -sf_gmg8_post0 4 -sf_gmg6_pre0 1 -sf_gmg6_post0 1 -sf_gmg6_nsc 1
R all1t  -sf_gmg7_pre0 1 -sf_gmg7_post0 1 -sf_gmg7_nsc 1 -sf_gmg8_post0 4 -sf_gmg6_pre0 1 -sf_gmg6_post0 1 -sf_gmg6_nsc 1 -sf_peel_tri 1
