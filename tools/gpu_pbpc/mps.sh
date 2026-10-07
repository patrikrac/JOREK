#!/bin/bash
# mps.sh start|stop|status : CUDA MPS daemon on every node of the job, for runs
# with more than one rank per GPU. Run through srun, one task per node:
#   srun --jobid=J -N NN --ntasks-per-node=1 --overlap tools/gpu_pbpc/mps.sh start
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_pipe_$USER}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps_log_$USER}
h=$(hostname -s)
if ! command -v nvidia-cuda-mps-control >/dev/null; then echo "[mps] $h: no nvidia-cuda-mps-control"; exit 1; fi
case "$1" in
  start)  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
          nvidia-cuda-mps-control -d && echo "[mps] $h: started" ;;
  stop)   echo quit | nvidia-cuda-mps-control && echo "[mps] $h: stopped" ;;
  status) echo get_server_list | nvidia-cuda-mps-control; echo "[mps] $h: rc $?" ;;
  *)      echo "usage: mps.sh start|stop|status"; exit 2 ;;
esac
