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
export SWEEP_VALUES='np.logspace(-10, 2, 50)'
export NUM_PARTS=20
export MAXITER_C=50
export VOXEL_RES=0.0005
export X_DEFAULT='[3.48139681e-03, 2.18907119e-07, 1.13673516e-07]'

# Job settings for SLURM
export PARTITION='commons'
export CPUS=16
export MEMPERCPU='8G'
export TIME='24:00:00'
export JOBNAME="${SWEEP_NAME}_0d5mm"

mkdir -p $LOG_ROOT $RESULT_ROOT

sbatch \
  --job-name="$JOBNAME" \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mem-per-cpu="$MEMPERCPU" \
  --time="$TIME" \
  --array=0-$(($NUM_PARTS-1)) \
  --output="${LOG_ROOT}/%x_%A_%a.out" \
  --error="${LOG_ROOT}/%x_%A_%a.err" \
  --export=ALL \
  $JOB
