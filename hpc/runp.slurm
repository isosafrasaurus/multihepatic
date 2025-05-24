#!/bin/bash
#SBATCH --job-name=sweep_consensus
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=/home/pzz1/logs/%x_%j_%a.out
#SBATCH --error=/home/pzz1/logs/%x_%j_%a.err
#
# Submit with e.g.:
#   sbatch --array=0-15 \
#          --export=ALL,PART_COUNT=16,SWEEP_NAME=gamma,\
#          SWEEP_VALUES="np.logspace(-10,-2,49)" \
#          sweep_consensus.sh
#
# Environment variables expected:
#   PART_COUNT  – total number of array shards
#   SWEEP_NAME  – gamma | gamma_a | gamma_R
#   SWEEP_VALUES– python expression yielding list/array of values

set -eo pipefail

if [[ -z "$SLURM_ARRAY_TASK_ID" || -z "$PART_COUNT" || -z "$SWEEP_NAME" || -z "$SWEEP_VALUES" ]]; then
    echo "Missing one of SLURM_ARRAY_TASK_ID, PART_COUNT, SWEEP_NAME, SWEEP_VALUES."
    exit 1
fi

# Preserve original behaviour -------------------------------------------------
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CXX="${CXX:-}"

HOME_REPO_PATH=$HOME/3d-1d
HOME_EXPORT_PATH=$HOME/exp
mkdir -p "$HOME_EXPORT_PATH"

SCRATCH_JOB_PATH=$SHARED_SCRATCH/$USER/$SLURM_JOB_ID
SCRATCH_EXPORT_PATH=$SCRATCH_JOB_PATH/exp
mkdir -p "$SCRATCH_JOB_PATH" "$SCRATCH_EXPORT_PATH"
cp -r "$HOME_REPO_PATH" "$SCRATCH_JOB_PATH/"

SRC_PATH=$SCRATCH_JOB_PATH/3d-1d/src
export PYTHONPATH="${PYTHONPATH}:$SRC_PATH"

module purge
module load Mamba
mamba init
source ~/.bashrc
eval "$(conda shell.bash hook)"
mamba activate cmor_mdanderson
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"

# -----------------------------------------------------------------------------#
# 1.  Run the shard assigned to this array task
# -----------------------------------------------------------------------------#
$CONDA_PREFIX/bin/python "$SCRATCH_JOB_PATH/3d-1d/hpc/runp.py" \
    --sweep_name  "$SWEEP_NAME" \
    --sweep_values "$SWEEP_VALUES" \
    --maxiter_o 10 \
    --maxiter_c 10 \
    --part_idx  "$SLURM_ARRAY_TASK_ID" \
    --part_count "$PART_COUNT" \
    --out_dir   "$SCRATCH_EXPORT_PATH"

cp -r "$SCRATCH_EXPORT_PATH"/*.csv "$HOME_EXPORT_PATH"

# -----------------------------------------------------------------------------#
# 2.  If this is task 0, wait for all *.csv then merge them
# -----------------------------------------------------------------------------#
if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    echo "[aggregator] waiting for ${PART_COUNT} CSV files..."
    while true; do
        got=$(ls "$HOME_EXPORT_PATH"/part_*.csv 2>/dev/null | wc -l)
        [[ "$got" -ge "$PART_COUNT" ]] && break
        sleep 30
    done
    echo "[aggregator] merging CSV files..."
    "$CONDA_PREFIX/bin/python" - << 'PY'
import glob, os, pandas as pd
out_dir = os.environ["HOME_EXPORT_PATH"]
parts   = sorted(glob.glob(os.path.join(out_dir,"part_*.csv")))
df      = pd.concat((pd.read_csv(f) for f in parts), ignore_index=True)
df.to_csv(os.path.join(out_dir,"sweep_merged.csv"), index=False)
print(f"[aggregator] wrote {len(parts)} pieces into sweep_merged.csv")
PY
fi

rm -rf "$SCRATCH_JOB_PATH"
echo "Job $SLURM_JOB_ID (array id ${SLURM_ARRAY_TASK_ID}) finished."

