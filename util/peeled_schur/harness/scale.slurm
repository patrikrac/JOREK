#!/bin/bash
# Mesh scaling of pair_w: true production (harm_couple .t.), the deck's prod (harm_couple .f., as in local M3)
# and peel v2, tstep 1 and 10, 3 steps each, restart from eq/$MESH.h5 (made by smoke.slurm).
# np 4 x 8 threads (NP env overrides).
#   sbatch --export=ALL,MESH=81x64 scale.slurm
#SBATCH --job-name=peel_scale
#SBATCH --partition=p.tok.openmp
#SBATCH --qos=p.tok.openmp.2h
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=scale_%x_%j.out
module purge
module load intel/2025.1 impi/2021.15 mkl/2025.1 fftw-mpi hdf5-mpi
H=$SLURM_SUBMIT_DIR; R=$H/runs/$MESH; mkdir -p $R
export EQ=$H/eq/$MESH.h5
its() { grep -ao "outer iterations: *[0-9]*" $1 | awk '{print $3}' | tr '\n' ' '; }
NP=${NP:-4}   # smoke 5231763: peel np 4 = np 1 within one outer its (8 8 8 vs 9 8 8)
echo "MESH $MESH NP $NP"
M="n_flux=${MESH%x*} n_tht=${MESH#*x}"
for dt in 1.d1 1.d0; do
  $H/run_case.sh $R/prodhc_dt$dt $NP 8 tstep_n=$dt nstep_n=3 $M physics_pc_sf_harm_couple=.t.
  $H/run_case.sh $R/prod_dt$dt   $NP 8 tstep_n=$dt nstep_n=3 $M
  $H/run_case.sh $R/peel_dt$dt   $NP 8 tstep_n=$dt nstep_n=3 $M physics_pc_sf_pair_w=\"peel\"
done
