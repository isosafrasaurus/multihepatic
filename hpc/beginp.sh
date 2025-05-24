export PART_COUNT=8
export SWEEP_NAME=gamma
export SWEEP_VALUES='np.logspace(-10,-2,16)'
sbatch --array=0-7 --export=ALL,PART_COUNT,SWEEP_NAME,SWEEP_VALUES \
       runp.slurm

