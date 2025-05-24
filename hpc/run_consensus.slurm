#!/bin/bash
#SBATCH --job-name=sweep_consensus
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pzz1@rice.edu
#SBATCH --output=/home/pzz1/logs/%x_%j.out
#SBATCH --error=/home/pzz1/logs/%x_%j.err

set -eo pipefail

# Patch (5/21)
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CXX="${CXX:-}"

# Define destination for export results
HOME_REPO_PATH=$HOME/3d-1d #WARNING: HOME_REPO_PATH IS BEING HARDCODED
HOME_EXPORT_PATH=$HOME/exp
mkdir -p $HOME_EXPORT_PATH

# Home to SHARED_SCRATCH on compute nodes
SCRATCH_JOB_PATH=$SHARED_SCRATCH/$USER/$SLURM_JOB_ID
SCRATCH_EXPORT_PATH=$SCRATCH_JOB_PATH/exp
mkdir -p $SCRATCH_JOB_PATH $SCRATCH_EXPORT_PATH
cp -r $HOME_REPO_PATH $SCRATCH_JOB_PATH/

# Define paths on SHARED_SCRATCH repo copy
SRC_PATH=$SCRATCH_JOB_PATH/3d-1d/src

# Load Mamba
module purge
module load Mamba
mamba init
source ~/.bashrc

# Activate cmor_mdanderson environment (see installation script)
eval "$(conda shell.bash hook)"
mamba activate cmor_mdanderson
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"

# Add src directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$SRC_PATH"
echo "Added $SRC_PATH to PYTHONPATH."

# Run sweep
$CONDA_PREFIX/bin/python -u $SCRATCH_JOB_PATH/3d-1d/hpc/consensus_sweep.py \
	--sweep_name "gamma" \
	--sweep_values "np.logspace(-10, -2, 3)" \
	--maxiter_o 10 \
	--maxiter_c 10 \
	--directory "$SCRATCH_EXPORT_PATH"

cp -r $SCRATCH_EXPORT_PATH/* $HOME_EXPORT_PATH

rm -rf $SCRATCH_JOB
echo "Job $SLURM_JOB_ID finished; results saved in $DEST"
