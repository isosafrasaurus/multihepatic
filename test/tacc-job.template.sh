#!/bin/bash
set -Eeuo pipefail
set +v +x || true

log()   { printf '[JOB] %s\n' "$*"; }
fatal() { printf '[JOB][FATAL] %s\n' "$*" >&2; }
on_err(){ fatal "Command '$BASH_COMMAND' failed (rc=$?)"; }
trap on_err ERR

log "START $(date -Is)"
log "JobID: ${SLURM_JOB_ID:-unknown}  Tasks: __TASKS__"
log "Script: __RUN_SCRIPT__"

# Quiet module housekeeping
module reset >/dev/null 2>&1 || true
module unload xalt >/dev/null 2>&1 || true

# Container runtime (quiet load attempts)
set +u
module load tacc-apptainer >/dev/null 2>&1 || \
module load apptainer      >/dev/null 2>&1 || \
module load singularity    >/dev/null 2>&1 || true
set -u

APPTAINER_BIN="$(command -v apptainer || true)"
[[ -z "$APPTAINER_BIN" ]] && APPTAINER_BIN="$(command -v singularity || true)"
if [[ -z "$APPTAINER_BIN" ]]; then
  fatal "apptainer/singularity not found on PATH after module load."
  exit 127
fi

# Clean container LD* env to avoid host interposition
unset LD_PRELOAD LD_AUDIT || true
export APPTAINERENV_LD_PRELOAD=""
export APPTAINERENV_LD_AUDIT=""

IMAGE_URI="__IMAGE_URI__"
REPO_URL="__REPO_URL__"
RUN_SCRIPT="__RUN_SCRIPT__"
TASKS=__TASKS__

# Use local repo if present; otherwise fetch quietly
if [[ -f "./${RUN_SCRIPT}" ]]; then
  REPO_DIR="$PWD"
else
  REPO_DIR="${WORK:-${SCRATCH:-$PWD}}/3d-1d-${SLURM_JOB_ID:-$$}"
  mkdir -p "$REPO_DIR"
  module load git >/dev/null 2>&1 || true
  if ! command -v git >/dev/null 2>&1; then
    fatal "git is required to clone ${REPO_URL} but was not found."
    exit 2
  fi
  log "Fetching repository…"
  if ! git clone --depth 1 "$REPO_URL" "$REPO_DIR" >/dev/null 2>&1; then
    fatal "git clone failed for ${REPO_URL}"
    exit 3
  fi
fi
cd "$REPO_DIR"
log "RepoDir: $PWD"

# Verify the run script exists relative to the repo
if [[ ! -f "$RUN_SCRIPT" ]]; then
  fatal "Run script not found at '$RUN_SCRIPT' in repo '$PWD'"
  exit 4
fi

# Apptainer cache (quiet)
unset XDG_RUNTIME_DIR || true
export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR" >/dev/null 2>&1 || true

# One concise runtime line
RUNTIME_VER="$("$APPTAINER_BIN" --version 2>/dev/null || true)"
[[ -n "$RUNTIME_VER" ]] && log "Runtime: $APPTAINER_BIN ($RUNTIME_VER)"
log "Image  : $IMAGE_URI"

log "Running workload…"
# Run inside /workspace (repo root) and call python on the repo-relative path
# Temporarily disable ERR trap for the srun so non-zero exit doesn't print a duplicate FATAL
trap - ERR
set +e
srun -n "${TASKS}" --export=ALL,LD_PRELOAD=,LD_AUDIT= \
  "$APPTAINER_BIN" exec --cleanenv \
  -B "$PWD:/workspace" --pwd /workspace \
  "$IMAGE_URI" \
  python3 "$RUN_SCRIPT"
rc=$?
set -e
trap on_err ERR

log "ExitCode: $rc"
log "END $(date -Is)"
exit "$rc"

