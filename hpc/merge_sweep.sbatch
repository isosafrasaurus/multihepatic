#!/bin/bash
#SBATCH --job-name=merge_consensus
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --dependency=afterok:${ARR_JOB_ID}
#SBATCH --output=/home/pzz1/logs/%x_%j.out
#SBATCH --error=/home/pzz1/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail
module purge
module load Mamba
mamba init
source ~/.bashrc

export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CXX="${CXX:-}"

eval "$(conda shell.bash hook)"
mamba activate cmor_mdanderson

JOBROOT=$SHARED_SCRATCH/$USER/${ARR_JOB_ID}
PARTS=$JOBROOT/parts
FINAL=$JOBROOT/final
mkdir -p "$FINAL"

python -u 3d-1d/hpc/merge_sweep_parts.py \
        --parts_dir  "$PARTS" \
        --sweep_name gamma \
        --out_dir    "$FINAL"

# copy finished artefacts back to home
cp -r "$FINAL"/*  $HOME/exp/
echo "Sweep finished – results in \$HOME/exp"
