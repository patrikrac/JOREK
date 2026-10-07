#!/bin/bash
# gpu_bind.sh CMD... : run CMD with one GPU per rank group. Contiguous node-local
# ranks share a GPU (rank r -> GPU r / (ranks_per_node / n_gpu)), so the ranks
# of one mode-split family (a contiguous range) sit on as few GPUs as possible.
# PETSc's default (rank % n_gpu) would interleave them. GPU_BIND_NGPU overrides
# the GPU count (default: SLURM's GPU list or 4).
loc=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
npn=${SLURM_NTASKS_PER_NODE:-${OMPI_COMM_WORLD_LOCAL_SIZE:-}}
if [ -z "$npn" ]; then                       # srun without --ntasks-per-node
  npn=$(( ${SLURM_NTASKS:-1} / ${SLURM_NNODES:-1} ))
fi
npn=${npn%%(*}                               # "16(x2)" -> 16
ng=${GPU_BIND_NGPU:-4}
per=$(( (npn + ng - 1) / ng )); [ $per -lt 1 ] && per=1
gpu=$(( loc / per )); [ $gpu -ge $ng ] && gpu=$(( ng - 1 ))
export CUDA_VISIBLE_DEVICES=$gpu
if [ "${SLURM_PROCID:-0}" = 0 ] || [ "${GPU_BIND_VERBOSE:-0}" = 1 ]; then
  echo "[gpu_bind] rank ${SLURM_PROCID:-?} node $(hostname -s) local $loc/$npn -> GPU $gpu (${per} ranks/GPU)" >&2
fi
exec "$@"
