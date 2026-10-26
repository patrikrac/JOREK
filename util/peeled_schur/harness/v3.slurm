#!/bin/bash
# Peel v3 (v3 note) at one mesh: production (harm_couple .t.), v3, v3 + pjfac/hcyc 2/tri (best local),
# the same with -sf_peel_patch 1; tstep 10 (and 1 for prod / best), 3 steps, restart from eq/$MESH.h5.
#   sbatch --export=ALL,MESH=161x128 v3.slurm
#SBATCH --job-name=peel_v3
#SBATCH --partition=p.tok.openmp
#SBATCH --qos=p.tok.openmp.2h
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=v3_%x_%j.out
module purge
module load intel/2025.1 impi/2021.15 mkl/2025.1 fftw-mpi hdf5-mpi
H=$SLURM_SUBMIT_DIR; R=$H/runs/v3_$MESH; mkdir -p $R
export EQ=$H/eq/$MESH.h5
NP=${NP:-4}
echo "MESH $MESH NP $NP"
M="n_flux=${MESH%x*} n_tht=${MESH#*x}"
PEEL=physics_pc_sf_pair_w=\"peel\"
BEST="-sf_peel_v 3 -sf_peel_pjfac 1 -sf_peel_hcyc 2 -sf_peel_tri 1"
run() { n=$1; o=$2; shift 2; PETSC_OPTIONS="-log_view :lv.txt $o" $H/run_case.sh $R/$n $NP 8 "$@"; }
run prod_dt10  ""                       tstep_n=1.d1 nstep_n=3 $M physics_pc_sf_harm_couple=.t.
run best_dt10  "$BEST"                  tstep_n=1.d1 nstep_n=3 $M $PEEL
run bestp_dt10 "$BEST -sf_peel_patch 1" tstep_n=1.d1 nstep_n=3 $M $PEEL
run v3_dt10    "-sf_peel_v 3"           tstep_n=1.d1 nstep_n=3 $M $PEEL
run prod_dt1   ""                       tstep_n=1.d0 nstep_n=3 $M physics_pc_sf_harm_couple=.t.
run best_dt1   "$BEST"                  tstep_n=1.d0 nstep_n=3 $M $PEEL
sacct -j $SLURM_JOB_ID --format=JobID%14,Elapsed,MaxRSS,State
