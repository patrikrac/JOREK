#!/bin/bash
# wsplit.sh: W's density / temperature halves at tstep 10, 30, 100 (default inner settings)
cd $(dirname $0)/..
export OMP_NUM_THREADS=4
for dt in ${DTS:-1.d1 3.d1 1.d2}; do
  for c in ${CFGS:-"rho1_T0|1|0" "rho0_T1|0|1"}; do
    n=${c%%|*}; r=${c#*|}; wr=${r%%|*}; wt=${r#*|}
    JOREK_W_RHO=$wr JOREK_W_T=$wt PETSC_OPTIONS="-log_view :lv.txt" tools/run.sh r/ws_${n}_$dt 1 tstep_n=$dt nstep_n=3 gmres_max_iter=60 | sed "s/^/dt=$dt $n: /"
    echo "   KSPSolve $(grep -E '^KSPSolve' r/ws_${n}_$dt/lv.txt 2>/dev/null | tail -1 | awk '{print $4}') s"
    rm -f r/ws_${n}_$dt/jorek
  done
done
