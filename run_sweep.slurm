#!/bin/bash
#SBATCH --job-name=gamma_sweep
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

export FI_PROVIDER="verbs;ofi_rxm;sockets"
set -euo pipefail

module purge
module load Mamba

cd $SHARED_SCRATCH/$USER/3d-1d
mkdir -p logs export plot_flow_data_semilog

CONDA_BASE=$(conda info --base)
ENV_DIR="$CONDA_PREFIX"
srun --mpi=pmi2 ­­­$ENV_DIR/bin/python sweeper_parallel.py
