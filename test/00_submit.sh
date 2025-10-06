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
RUN_SCRIPT="test/00_tacc_test.py"

# Logging
mkdir -p "${LOGDIR}"
OUT_PATTERN="${LOGDIR}/${JOB_NAME}-%j.out"
ERR_PATTERN="${LOGDIR}/${JOB_NAME}-%j.err"

# Create a temporary job script (quiet but prints key lifecycle + errors)
JOBFILE="$(mktemp -p "$PWD" tacc-job-XXXXXX.sh)"
JOB_TEMPLATE="./tacc-job.template.sh"
if [[ ! -f "${JOB_TEMPLATE}" ]]; then
  echo "[WRAPPER][FATAL] Job template not found at ${JOB_TEMPLATE}" >&2
  exit 1
fi
cp "${JOB_TEMPLATE}" "${JOBFILE}"

sed -i \
  -e "s|__IMAGE_URI__|${IMAGE_URI}|g" \
  -e "s|__REPO_URL__|${REPO_URL}|g" \
  -e "s|__RUN_SCRIPT__|${RUN_SCRIPT}|g" \
  -e "s|__TASKS__|${TASKS}|g" \
  "${JOBFILE}"

chmod +x "${JOBFILE}"

cleanup() { rm -f "${JOBFILE}" 2>/dev/null || true; }
trap cleanup EXIT

# Submit (concise)
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
  echo "[WRAPPER][FATAL] Failed to parse job id from sbatch output:" >&2
  printf '%s\n' "$jobid_raw" >&2
  exit 1
fi

out_file="${OUT_PATTERN//%j/${jobid}}"
err_file="${ERR_PATTERN//%j/${jobid}}"

echo "Submitted job ${jobid} (stdout: ${out_file}, stderr: ${err_file})"

# Start tails
tail -n +1 -F --retry "${out_file}" | sed -u 's/^/[STDOUT] /' &
T1=$!
tail -n +1 -F --retry "${err_file}" | sed -u 's/^/[STDERR] /' &
T2=$!

# CTRL+C detaches from tails only
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

# Stop tails and report final state
kill "${T1}" "${T2}" 2>/dev/null || true
wait "${T1}" "${T2}" 2>/dev/null || true

IFS='|' read -r state exit_code reason <<<"$(sacct -j "${jobid}" --format=State,ExitCode,Reason --noheader -P 2>/dev/null | head -n1)"
state="$(echo "${state:-unknown}" | tr -d ' ')"

echo
echo "Job ${jobid} finished."
echo "  State    : ${state}"
echo "  ExitCode : ${exit_code:-unknown}"
echo "  Reason   : ${reason:-unknown}"

[[ "${state}" == "COMPLETED" ]] || exit 1

