#!/bin/bash
#SBATCH --job-name=sweep_mpi
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=40
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load GCC/13.3.0 OpenMPI/5.0.3
module load Mamba

set +u
eval "$(conda shell.bash hook)"
conda activate cmor_mdanderson
set -u

SCRATCH_JOB=$SHARED_SCRATCH/$USER/$SLURM_JOB_ID
mkdir -p $SCRATCH_JOB
cd $SCRATCH_JOB

cp -r $HOME/3d-1d .

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

srun --mpi=pmi2 -n $SLURM_NTASKS --cpu-bind=cores python 3d-1d/sweeper_mpi.py

DEST=$HOME/export/$SLURM_JOB_ID
mkdir -p $DEST
cp -r export/* $DEST

cd $HOME
rm -rf $SCRATCH_JOB
echo "Job $SLURM_JOB_ID finished; results saved in $DEST"
