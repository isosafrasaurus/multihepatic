#!/bin/bash
#SBATCH --job-name=gamma_sweep
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=4               # number of MPI ranks
#SBATCH --cpus-per-task=8        # each rank gets 8 CPUs for its local pool
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
module load mpi4py          # ensure mpi4py is available

cd $SHARED_SCRATCH/$USER/3d-1d
mkdir -p logs export plot_flow_data_semilog

ENV_DIR="$CONDA_PREFIX"

# Launch 4 MPI ranks; each rank will use --cpus-per-task cores in its local pool
srun --mpi=pmix_v3 $ENV_DIR/bin/python -u sweeper_parallel.py

