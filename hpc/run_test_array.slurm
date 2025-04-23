#!/bin/bash
#SBATCH --job-name=gamma_sweep
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --array=0-3%4            # 4 tasks, each ~20/4=5 gammas
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

set -euo pipefail

module purge
module load Mamba

CONDA_BASE=$(conda info --base)
ENV_DIR="$CONDA_PREFIX"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"

# Run the per-task sweep & then the merge (only in TASK 0)
srun $ENV_DIR/bin/python sweep_array.py
