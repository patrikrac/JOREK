#!/bin/bash
# nsys_rank.sh CMD... : run CMD under Nsight Systems on rank NSYS_RANK (default 1),
# plainly on the others; report in $NSYS_OUT (default $PWD) as nsys_r<rank>.nsys-rep.
if [ "${SLURM_PROCID:-0}" = "${NSYS_RANK:-1}" ]; then
  exec nsys profile -t cuda,nvtx --force-overwrite true -o ${NSYS_OUT:-$PWD}/nsys_r${SLURM_PROCID} "$@"
fi
exec "$@"
