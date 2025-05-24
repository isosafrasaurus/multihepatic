#!/bin/bash
# submit_sweep.sh

ARRAY_JOB=$(sbatch sweep_array.sbatch | awk '{print $4}')
sbatch --dependency=afterok:${ARRAY_JOB} --export=ARR_JOB_ID=${ARRAY_JOB} merge_sweep.sbatch
echo "Submitted array ${ARRAY_JOB} and merge job."

