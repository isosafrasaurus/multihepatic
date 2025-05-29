#!/usr/bin/env bash
set -eo pipefail

# Configure locations of scripts and results
export DRIVER="$HOME/3d-1d/hpc/driverp.py"
export JOB="$HOME/3d-1d/hpc/jobp.sh"
export COMBINER="$HOME/3d-1d/hpc/combinep.py"
export SRC="$HOME/3d-1d/src"
export LOG_ROOT="$HOME/3d-1d/hpc/logs"
export RESULT_ROOT="$HOME/3d-1d/hpc/exp"

# Configurations
export SWEEP_NAME='gamma'
export SWEEP_VALUES='np.logspace(-10, 2, 10)'
export NUM_PARTS=3
export MAXITER_C=10
export VOXEL_RES=0.002
export X_DEFAULT='[3.48139681e-03, 2.18907119e-07, 1.13673516e-07]'

# Job settings for SLURM
export PARTITION='commons'
export CPUS=4
export MEM="16G"
export TIME="00:30:00"
export JOBNAME="${SWEEP_NAME}_sweep"

mkdir -p $LOG_ROOT $RESULT_ROOT

sbatch \
  --job-name="$JOBNAME" \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --array=0-$(($NUM_PARTS-1)) \
  --output="${LOG_ROOT}/%x_%A_%a.out" \
  --error="${LOG_ROOT}/%x_%A_%a.err" \
  --export=ALL \
  $JOB