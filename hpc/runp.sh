#!/usr/bin/env bash
set -eo pipefail

export DRIVER="$PWD/driverp.py"
export RESULT_ROOT="$HOME/exp"
export SWEEP_NAME="gamma"
export SWEEP_VALUES='np.logspace(-10,2,5)'
export NUM_PARTS=3
export MAXITER_O=10
export MAXITER_C=10
export VOXEL_RES=0.002
export PARTITION="commons"
export CPUS=4
export MEM="16G"
export TIME="00:30:00"
export JOBNAME="flow_sweep"

mkdir -p logs "$RESULT_ROOT"

sbatch \
  --job-name="$JOBNAME" \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --array=0-$(($NUM_PARTS-1)) \
  --output="logs/%x_%A_%a.out" \
  --error="logs/%x_%A_%a.err" \
  --export=ALL \
  jobp.sh
