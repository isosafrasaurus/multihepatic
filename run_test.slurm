#!/bin/bash
#SBATCH -J fenics-test
#SBATCH -o out.%j
#SBATCH -e err.%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -A ASC22053
#SBATCH -p skx-dev

set -euo pipefail

module load mvapich/3.0
module load tacc-apptainer

export MV2_SMP_USE_CMA=0
export MV2_ENABLE_AFFINITY=0
export OMP_NUM_THREADS=1
export DIJITSO_CACHE_DIR=$SCRATCH/dijitso-cache
export DOLFIN_JIT_CACHE_DIR=$SCRATCH/dolfin-jit-cache
mkdir -p "$DIJITSO_CACHE_DIR" "$DOLFIN_JIT_CACHE_DIR"

mkdir -p "$SCRATCH"
if [ ! -f "/scratch/10756/piercezhang9871/tacc-mvapich2.3-python3.12-graphnics_latest.sif" ]; then
  echo "[job] Pulling container image -> /scratch/10756/piercezhang9871/tacc-mvapich2.3-python3.12-graphnics_latest.sif"
  apptainer pull "/scratch/10756/piercezhang9871/tacc-mvapich2.3-python3.12-graphnics_latest.sif" "ghcr.io/isosafrasaurus/tacc-mvapich2.3-python3.12-graphnics:latest"
else
  echo "[job] Using existing image /scratch/10756/piercezhang9871/tacc-mvapich2.3-python3.12-graphnics_latest.sif"
fi

cd "/work2/10756/piercezhang9871/stampede3/3d-1d"
export PYTHONPATH=$PWD:${PYTHONPATH:-}

echo "[job] Launching test.py"
ibrun -n $SLURM_NTASKS apptainer exec --cleanenv \
  --bind $WORK,$SCRATCH \
  --env MV2_SMP_USE_CMA=$MV2_SMP_USE_CMA,MV2_ENABLE_AFFINITY=$MV2_ENABLE_AFFINITY,OMP_NUM_THREADS=$OMP_NUM_THREADS,DIJITSO_CACHE_DIR=$DIJITSO_CACHE_DIR,DOLFIN_JIT_CACHE_DIR=$DOLFIN_JIT_CACHE_DIR,PYTHONPATH=$PYTHONPATH \
  "/scratch/10756/piercezhang9871/tacc-mvapich2.3-python3.12-graphnics_latest.sif" python3 -u "test.py"

echo "[job] Done."
