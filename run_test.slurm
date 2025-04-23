#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

set -euo pipefail

module purge
module load Mamba

CONDA_BASE=$(conda info --base)
ENV_DIR="$CONDA_PREFIX"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"

srun --mpi=pmi2 $ENV_DIR/bin/python test.py
