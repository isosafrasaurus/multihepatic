#!/bin/bash
set -Eeuo pipefail
set +v +x || true

log()   { printf '[JOB] %s\n' "$*"; }
fatal() { printf '[JOB][FATAL] %s\n' "$*" >&2; }
on_err(){ fatal "Command '$BASH_COMMAND' failed (rc=$?)"; }
trap on_err ERR

log "START $(date -Is)"
log "JobID: ${SLURM_JOB_ID:-unknown}  Tasks: __TASKS__"

module reset >/dev/null 2>&1 || true
module unload xalt >/dev/null 2>&1 || true

set +u
module load tacc-apptainer >/dev/null 2>&1 || true
set -u

# Only look for apptainer on PATH
APPTAINER_BIN="$(command -v apptainer || true)"
if [[ -z "$APPTAINER_BIN" ]]; then
  fatal "Apptainer not found on PATH after module load."
  exit 127
fi

unset LD_PRELOAD LD_AUDIT || true
export APPTAINERENV_LD_PRELOAD=""
export APPTAINERENV_LD_AUDIT=""

IMAGE_URI="__IMAGE_URI__"
RUN_ABS="__RUN_ABS__"
TASKS=__TASKS__
PROJECT_ROOT="__PROJECT_ROOT__"

if [[ ! -f "$RUN_ABS" ]]; then
  fatal "Run script not found at absolute path: $RUN_ABS"
  exit 2
fi
if [[ ! -d "$PROJECT_ROOT" ]]; then
  fatal "Project root directory not found: $PROJECT_ROOT"
  exit 3
fi

RUN_DIR="$(dirname "$RUN_ABS")"

log "Script : $RUN_ABS"
log "RunDir : $RUN_DIR"
log "ProjRt : $PROJECT_ROOT"

unset XDG_RUNTIME_DIR || true
export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR" >/dev/null 2>&1 || true

RUNTIME_VER="$("$APPTAINER_BIN" --version 2>/dev/null || true)"
[[ -n "$RUNTIME_VER" ]] && log "Runtime: $APPTAINER_BIN ($RUNTIME_VER)"
log "Image  : $IMAGE_URI"

export APPTAINERENV_PYTHONPATH="$PROJECT_ROOT"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1

log "Running workload…"
trap - ERR
set +e
srun -n "${TASKS}" --export=ALL,LD_PRELOAD=,LD_AUDIT= \
  "$APPTAINER_BIN" exec --cleanenv \
  -B "$PROJECT_ROOT:$PROJECT_ROOT" \
  --pwd "$RUN_DIR" \
  "$IMAGE_URI" \
  python3 "$RUN_ABS"
rc=$?
set -e
trap on_err ERR

log "ExitCode: $rc"
log "END $(date -Is)"
exit "$rc"

