#!/usr/bin/env bash
set -Eeuo pipefail
set +v +x || true

# This script is executed within a SLURM job context and is responsible for
# setting up the Apptainer runtime environment.
#
# It is designed for use within the TACC environment and to be submitted via a
# job submission wrapper script that prepares a SLURM batch file.
#
# Environment variables and placeholders are replaced at job submission time.
#
# Placeholders:
#   __IMAGE_URI__       : Container image URI
#   __TARGET_ABS_PATH__ : Absolute path to the target Python script
#   __JOB_TASKS__       : Total number of MPI tasks
#   __PROJECT_ROOT__    : Absolute path to project root

IMAGE_URI="__IMAGE_URI__"
TARGET_ABS_PATH="__TARGET_ABS_PATH__"
JOB_TASKS=__JOB_TASKS__
JOB_CPUS_PER_TASK=__JOB_CPUS_PER_TASK__
PROJECT_ROOT="__PROJECT_ROOT__"

# log: Print a message to stdout.
# Usage: log "message"
log()
{
	printf '[JOB] %s\n' "$*"
}

# fatal: Print a fatal error message to stderr.
# Usage: fatal "error message"
fatal()
{
	printf '[JOB][FATAL] %s\n' "$*" >&2
}

# on_err: Trap handler invoked when a command fails (ERR trap).
# Prints the failed command and its exit code before terminating the script.
on_err()
{
	fatal "Command '$BASH_COMMAND' failed (rc=$?)"
}

trap on_err ERR

log "START $(date -Is)"
log "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"

module reset >/dev/null 2>&1 || true
module unload xalt >/dev/null 2>&1 || true

set +u
module load tacc-apptainer >/dev/null 2>&1 || true
set -u

APPTAINER_BIN="$(command -v apptainer || true)"
if [[ -z "$APPTAINER_BIN" ]]; then
	fatal "Apptainer not found on PATH after module load."
	exit 127
fi

unset LD_PRELOAD LD_AUDIT || true
export APPTAINERENV_LD_PRELOAD=""
export APPTAINERENV_LD_AUDIT=""

if [[ ! -f "$TARGET_ABS_PATH" ]]; then
	fatal "Run script not found at absolute path: $TARGET_ABS_PATH"
	exit 2
fi

if [[ ! -d "$PROJECT_ROOT" ]]; then
	fatal "Project root directory not found: $PROJECT_ROOT"
	exit 3
fi

RUN_DIR="$(dirname "$TARGET_ABS_PATH")"

log "Script : $TARGET_ABS_PATH"
log "RunDir : $RUN_DIR"
log "ProjRt : $PROJECT_ROOT"

# Setup Apptainer runtime
unset XDG_RUNTIME_DIR || true
export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR" >/dev/null 2>&1 || true

RUNTIME_VER="$("$APPTAINER_BIN" --version 2>/dev/null || true)"
[[ -n "$RUNTIME_VER" ]] && log "Runtime: $APPTAINER_BIN ($RUNTIME_VER)"
log "Image  : $IMAGE_URI"

export APPTAINERENV_PYTHONPATH="$PROJECT_ROOT"
export APPTAINERENV_PYTHONUNBUFFERED=1

# MPI and threading environment variables
export MV2_SMP_USE_CMA=0
export MV2_ENABLE_AFFINITY=0

export APPTAINERENV_MV2_SMP_USE_CMA=0
export APPTAINERENV_MV2_ENABLE_AFFINITY=0

export APPTAINERENV_OMP_NUM_THREADS="${JOB_CPUS_PER_TASK}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${JOB_CPUS_PER_TASK}"
export APPTAINERENV_MKL_NUM_THREADS="${JOB_CPUS_PER_TASK}"
export APPTAINERENV_NUMEXPR_NUM_THREADS="${JOB_CPUS_PER_TASK}"

export APPTAINERENV_XDG_CACHE_HOME="/tmp/${USER}/xdg-${SLURM_JOB_ID}-${SLURM_PROCID}"
export APPTAINERENV_INSTANT_CACHE_DIR="/tmp/${USER}/instant-${SLURM_JOB_ID}-${SLURM_PROCID}"
export APPTAINERENV_DIJITSO_CACHE_DIR="/tmp/${USER}/dijitso-${SLURM_JOB_ID}-${SLURM_PROCID}"
export APPTAINERENV_DOLFIN_JIT_CACHE_DIR="/tmp/${USER}/dolfin-${SLURM_JOB_ID}-${SLURM_PROCID}"

export APPTAINERENV_INSTANT_SYSTEM_CALL_METHOD="OS_SYSTEM"
export APPTAINERENV_DIJITSO_SYSTEM_CALL_METHOD="OS_SYSTEM"

mkdir -p \
	"/tmp/${USER}/xdg-${SLURM_JOB_ID}-${SLURM_PROCID}" \
	"/tmp/${USER}/instant-${SLURM_JOB_ID}-${SLURM_PROCID}" \
	"/tmp/${USER}/dijitso-${SLURM_JOB_ID}-${SLURM_PROCID}" \
	"/tmp/${USER}/dolfin-${SLURM_JOB_ID}-${SLURM_PROCID}" || true

# Execute workload
log "Running workload..."
trap - ERR
set +e
srun -n "${JOB_TASKS}" -c "${JOB_CPUS_PER_TASK}" --mpi=pmi2 --cpu-bind=cores --export=ALL,LD_PRELOAD=,LD_AUDIT= \
	"$APPTAINER_BIN" exec \
	-B "$PROJECT_ROOT:$PROJECT_ROOT" \
	--pwd "$RUN_DIR" \
	"$IMAGE_URI" \
	python3 "$TARGET_ABS_PATH"
rc=$?
set -e
trap on_err ERR

# Cleanup and exit
log "ExitCode: $rc"
log "END $(date -Is)"
exit "$rc"

