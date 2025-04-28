#!/bin/bash
#SBATCH --job-name=sweep_mpi
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=24
##SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export FI_PROVIDER="verbs;ofi_rxm;sockets"
set -euo pipefail

module purge
module load Mamba

set +u
eval "$(conda shell.bash hook)"
conda activate cmor_mdanderson
set -u

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"

SCRATCH_JOB=$SHARED_SCRATCH/$USER/$SLURM_JOB_ID
mkdir -p $SCRATCH_JOB
cd $SCRATCH_JOB

cp -r $HOME/3d-1d .

ENV_DIR="$CONDA_PREFIX"
srun --mpi=pmi2 $ENV_DIR/bin/python -u $SCRATCH_JOB/3d-1d/hpc/sweeper_processpool.py

DEST=$HOME/export/$SLURM_JOB_ID
mkdir -p $DEST
cp -r export/* $DEST

cd $HOME
rm -rf $SCRATCH_JOB
echo "Job $SLURM_JOB_ID finished; results saved in $DEST"
