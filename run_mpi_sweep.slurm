#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=10          # now we’ll run 10 “things”
#SBATCH --cpus-per-task=2    # each thing gets 2 cores
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

#SBATCH --cpu-bind=cores

srun --mpi=pmi2 $ENV_DIR/bin/python test.py

