#!/bin/bash
# Equilibria (fresh start, one step of 1e-3) for 41x32 / 81x64 / 161x128, then 41x32 tstep 10
# prod vs peel at np 1 and np 4 (peel at np > 1 untested so far).
#SBATCH --job-name=peel_smoke
#SBATCH --partition=p.tok.openmp
#SBATCH --qos=p.tok.openmp.2h
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=smoke_%j.out
module purge
module load intel/2025.1 impi/2021.15 mkl/2025.1 fftw-mpi hdf5-mpi
H=$SLURM_SUBMIT_DIR; R=$H/runs; mkdir -p $R $H/eq
for m in 41x32 81x64 161x128; do
  nf=${m%x*}; nt=${m#*x}
  $H/run_case.sh $R/eq_$m 1 8 n_flux=$nf n_tht=$nt tstep_n=1.d-3 nstep_n=1
  cp $R/eq_$m/jorek_restart.h5 $H/eq/$m.h5
done
export EQ=$H/eq/41x32.h5
$H/run_case.sh $R/s41_prod_np1 1 8 tstep_n=1.d1 nstep_n=3
$H/run_case.sh $R/s41_peel_np1 1 8 tstep_n=1.d1 nstep_n=3 physics_pc_sf_pair_w=\"peel\"
$H/run_case.sh $R/s41_prod_np4 4 8 tstep_n=1.d1 nstep_n=3
$H/run_case.sh $R/s41_peel_np4 4 8 tstep_n=1.d1 nstep_n=3 physics_pc_sf_pair_w=\"peel\"
