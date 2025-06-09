#!/usr/bin/env bash
set -eo pipefail

module purge
module load Mamba
eval "$(conda shell.bash hook)"
conda activate cmor_mdanderson

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CXX="${CXX:-}"
export PYTHONPATH="${SRC}:${PYTHONPATH:-}"

OUTDIR="$RESULT_ROOT/${SWEEP_NAME}_$SLURM_ARRAY_JOB_ID"
mkdir -p "$OUTDIR"

python -u "$DRIVER" \
  --sweep_name "$SWEEP_NAME" \
  --sweep_values "$SWEEP_VALUES" \
  --directory "$OUTDIR" \
  --num_parts "$NUM_PARTS" \
  --part_idx "$SLURM_ARRAY_TASK_ID" \
  --maxiter_c "$MAXITER_C" \
  --voxel_res "$VOXEL_RES" \
  --x_default "$X_DEFAULT"

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  echo "[task 0] waiting for the other $((NUM_PARTS-1)) part CSVs…"
  PART_SUFFIX=$((NUM_PARTS-1))
  until [[ $(ls "$OUTDIR"/part_*_of_${PART_SUFFIX}.csv 2>/dev/null | wc -l) -eq "$NUM_PARTS" ]]; do
    sleep 60
  done
  echo "[task 0] all parts present – merging"
  python "$COMBINER" "$OUTDIR"
fi