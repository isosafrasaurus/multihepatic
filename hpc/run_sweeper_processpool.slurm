#!/bin/bash
#SBATCH --job-name=sweep_mpi
#SBATCH --account=commons
#SBATCH --partition=commons

# Request a single task but 24 CPUs for multiprocessing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

##SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

# Run your python driver directly, allocating all 24 CPUs
# srun -n1 -c${SLURM_CPUS_PER_TASK} would also work here
$ENV_DIR/bin/python -u $SCRATCH_JOB/3d-1d/hpc/sweeper_processpool.py

DEST=$HOME/export/$SLURM_JOB_ID
mkdir -p $DEST
cp -r export/* $DEST

cd $HOME
rm -rf $SCRATCH_JOB
echo "Job $SLURM_JOB_ID finished; results saved in $DEST"

