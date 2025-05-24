#!/bin/bash
#SBATCH --job-name=sweep_consensus
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/pzz1/logs/%x_%A_%a.out
#SBATCH --error=/home/pzz1/logs/%x_%A_%a.err

set -eo pipefail
module purge
module load Mamba
mamba init
source ~/.bashrc

export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CXX="${CXX:-}"

eval "$(conda shell.bash hook)"
mamba activate cmor_mdanderson
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"

JOBROOT=$SHARED_SCRATCH/$USER/$SLURM_ARRAY_JOB_ID
mkdir -p "$JOBROOT";  cd "$JOBROOT"

# one-time copy of repo (cheap if $SHARED_SCRATCH is on the same FS)
if [[ ! -d 3d-1d ]]; then  cp -r $HOME/3d-1d .; fi
export PYTHONPATH="$PWD/3d-1d/src:${PYTHONPATH:-}"

PARTS=$JOBROOT/parts
mkdir -p "$PARTS"

python -u 3d-1d/hpc/run_consensus_sweep.py \
   --sweep_name   gamma \
   --sweep_values "np.logspace(-10, -2, 3)" \
   --maxiter_o    10 \
   --maxiter_c    10 \
   --part_idx     $SLURM_ARRAY_TASK_ID \
   --part_count   3 \
   --out_dir      "$PARTS"

