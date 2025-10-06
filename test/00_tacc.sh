#!/usr/bin/env bash
set -Eeuo pipefail

ACCOUNT="ASC22053"
JOB_NAME="3d-1d"
PARTITION="skx-dev"
TIME="00:30:00"
NODES=1
TASKS=1
LOGDIR="$PWD/logs"

IMAGE_URI="docker://ghcr.io/isosafrasaurus/tacc-mvapich2.3-python3.12-graphnics:latest"
REPO_URL="https://github.com/isosafrasaurus/3d-1d"
RUN_SCRIPT="test.py"

# Logging
mkdir -p "${LOGDIR}"
OUT_PATTERN="${LOGDIR}/${JOB_NAME}-%j.out"
ERR_PATTERN="${LOGDIR}/${JOB_NAME}-%j.err"

# Create a temporary job script
JOBFILE="$(mktemp -p "$PWD" tacc-job-XXXXXX.sh)"
cat > "${JOBFILE}" <<'EOF'
#!/bin/bash
set -Eeuo pipefail

set +v +x || true

echo "[JOB] ===== BEGIN $(date) ====="
echo "[JOB] Host: $(hostname)"
echo "[JOB] CWD : $PWD"

module reset || true
module unload xalt >/dev/null 2>&1 || true

export BASH_COMPLETION_DEBUG=${BASH_COMPLETION_DEBUG:-0}

set +u
module load tacc-apptainer >/dev/null 2>&1 || \
module load apptainer      >/dev/null 2>&1 || \
module load singularity    >/dev/null 2>&1 || true
set -u

APPTAINER_BIN="$(command -v apptainer || true)"
[[ -z "$APPTAINER_BIN" ]] && APPTAINER_BIN="$(command -v singularity || true)"
if [[ -z "$APPTAINER_BIN" ]]; then
  echo "[JOB][ERROR] apptainer/singularity not found on PATH after module load." >&2
  exit 127
fi

unset LD_PRELOAD LD_AUDIT || true
export APPTAINERENV_LD_PRELOAD=""
export APPTAINERENV_LD_AUDIT=""

IMAGE_URI="__IMAGE_URI__"
REPO_URL="__REPO_URL__"
RUN_SCRIPT="__RUN_SCRIPT__"
TASKS=__TASKS__

if [[ -f "./${RUN_SCRIPT}" ]]; then
  REPO_DIR="$PWD"
else
  REPO_DIR="${WORK:-${SCRATCH:-$PWD}}/3d-1d-${SLURM_JOB_ID}"
  mkdir -p "$REPO_DIR"
  module load git >/dev/null 2>&1 || true
  if ! command -v git >/dev/null 2>&1; then
    echo "[JOB][ERROR] git is required to clone ${REPO_URL} but was not found." >&2
    exit 2
  fi
  echo "[JOB] Cloning ${REPO_URL} into ${REPO_DIR} ..."
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
echo "[JOB] Repo dir: $PWD"

unset XDG_RUNTIME_DIR || true
export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR"

set -x
"$APPTAINER_BIN" --version

srun -n "${TASKS}" --export=ALL,LD_PRELOAD=,LD_AUDIT= \
  "$APPTAINER_BIN" exec --cleanenv -B "$PWD:/workspace" "$IMAGE_URI" \
  python3 "/workspace/$RUN_SCRIPT"
rc=$?
set +x

echo "[JOB] Container exit code: $rc"
echo "[JOB] =====  END  $(date) ====="
exit "$rc"

EOF

sed -i \
  -e "s|__IMAGE_URI__|${IMAGE_URI}|g" \
  -e "s|__REPO_URL__|${REPO_URL}|g" \
  -e "s|__RUN_SCRIPT__|${RUN_SCRIPT}|g" \
  -e "s|__TASKS__|${TASKS}|g" \
  "${JOBFILE}"

chmod +x "${JOBFILE}"

cleanup() { rm -f "${JOBFILE}" 2>/dev/null || true; }
trap cleanup EXIT

# Submit
jobid_raw="$(
  sbatch \
    --parsable \
    --chdir "$PWD" \
    -A "${ACCOUNT}" \
    -J "${JOB_NAME}" \
    -p "${PARTITION}" \
    -t "${TIME}" \
    -N "${NODES}" \
    -n "${TASKS}" \
    -o "${OUT_PATTERN}" \
    -e "${ERR_PATTERN}" \
    "${JOBFILE}" 2>&1
)"
jobid="$(printf '%s\n' "$jobid_raw" | awk 'NF{last=$0}END{print last}' | cut -d';' -f1 | tr -d '[:space:]')"

if [[ ! "$jobid" =~ ^[0-9]+$ ]]; then
  echo "Failed to parse job id from sbatch output:" >&2
  printf '%s\n' "$jobid_raw" >&2
  exit 1
fi

out_file="${OUT_PATTERN//%j/${jobid}}"
err_file="${ERR_PATTERN//%j/${jobid}}"

echo "Submitted job ${jobid}"
echo "  stdout: ${out_file}"
echo "  stderr: ${err_file}"
echo

# Start tails
tail -n +1 -F --retry "${out_file}" | sed -u 's/^/[STDOUT] /' &
T1=$!
tail -n +1 -F --retry "${err_file}" | sed -u 's/^/[STDERR] /' &
T2=$!

# CTRL+C should detach from tails
on_int() {
  echo
  echo "Detaching from logs. Job ${jobid} continues to run."
  kill "${T1}" "${T2}" 2>/dev/null || true
  wait "${T1}" "${T2}" 2>/dev/null || true
  exit 0
}
trap on_int INT

# Poll the queue; when the job leaves, stop tailing
while :; do
  qline="$(squeue -h -j "${jobid}" 2>/dev/null || true)"
  [[ -n "${qline}" ]] || break
  sleep 2
done

# Stop tails and report final state (+ reason)
kill "${T1}" "${T2}" 2>/dev/null || true
wait "${T1}" "${T2}" 2>/dev/null || true

sleep 1
IFS='|' read -r state exit_code reason <<<"$(sacct -j "${jobid}" --format=State,ExitCode,Reason --noheader -P 2>/dev/null | head -n1)"
state="$(echo "${state:-unknown}" | tr -d ' ')"

echo
echo "Job ${jobid} finished."
echo "  State    : ${state}"
echo "  ExitCode : ${exit_code:-unknown}"
echo "  Reason   : ${reason:-unknown}"

[[ "${state}" == "COMPLETED" ]] || exit 1
