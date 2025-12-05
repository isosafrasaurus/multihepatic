#!/usr/bin/env bash
set -Eeuo pipefail

# Configurations
PROJECT_ROOT="$WORK/multihepatic"
TACC_ACCOUNT="ASC22053"
TACC_PARTITION="skx-dev"
IMAGE_URI="docker://ghcr.io/isosafrasaurus/tacc-mpc-patch:latest"
JOB_TEMPLATE_PATH="$PROJECT_ROOT/tacc-job.template.slurm"

JOB_NAME="multihepatic"
JOB_TIME="00:30:00"
JOB_NODES=1
JOB_TASKS_PER_NODE=1
JOB_CPUS_PER_TASK=1
JOB_LOGS_DIR="$PROJECT_ROOT/_logs"

# Usage
usage()
{
	cat <<EOF
Usage: $(basename "$0") [OPTIONS] TARGET_PATH

Submit a job to TACC using a MVAPICH-aware containerized environment.

Required argument:
  TARGET_PATH            Path to the script to execute

Options:
  --name NAME            Job name (default: ${JOB_NAME})
  --time HH:MM:SS        Wall time (default: ${JOB_TIME})
  --nodes N              Number of nodes (default: ${JOB_NODES})
  --tasks-per-node N     Tasks per node (default: ${JOB_TASKS_PER_NODE})
  --cpus-per-task N      CPUs (Threads) per task (default: ${JOB_CPUS_PER_TASK})
  --logs-dir DIR         Log directory (default: ${JOB_LOGS_DIR})
  -h, --help             Show this help message and exit

Example:
  $(basename "$0") --time 01:00:00 --nodes 2 run_simulation.py
EOF
}

# Parse optional CLI flags
while [[ $# -gt 0 ]]; do
	case "$1" in
		--name)
			JOB_NAME="${2:-}"; shift 2 ;;
		--time)
			JOB_TIME="${2:-}"; shift 2 ;;
		--nodes)
			JOB_NODES="${2:-}"; shift 2 ;;
		--tasks-per-node)
			JOB_TASKS_PER_NODE="${2:-}"; shift 2 ;;
        --cpus-per-task)
        JOB_CPUS_PER_TASK="${2:-}"; shift 2 ;;
		--logs-dir)
			JOB_LOGS_DIR="${2:-}"; shift 2 ;;
		-h|--help)
			usage; exit 0 ;;
		--)
			shift; break ;;
		-*)
			printf '[SUBMITTER][FATAL] Unknown option: %s\n' "$1" >&2
			usage
			exit 2 ;;
		*)
			break ;;
	esac
done

# Check job template is present at JOB_TEMPLATE_PATH
if [[ ! -f "${JOB_TEMPLATE_PATH}" ]]; then
	echo "[SUBMITTER][FATAL] Job template not found at ${JOB_TEMPLATE_PATH}" >&2
	exit 1
fi

# Parse required CLI argument
TARGET_PATH="${1:-}"

# Check required CLI argument
if [[ -z "${TARGET_PATH}" ]]; then
	echo "[SUBMITTER][FATAL] Missing required TARGET_PATH argument." >&2
	usage
	exit 2
fi

# Check presence of target file from required CLI argument
if [[ ! -f "$TARGET_PATH" ]]; then
	echo "[SUBMITTER][FATAL] Target script not found at path: '$TARGET_PATH'" >&2
	exit 1
fi

# abspath: Resolve a given path to an absolute path.
# Usage: abspath <path>
#
# Arguments:
#   $1 - Path to resolve.
#
# Outputs:
#   Prints the absolute, normalized path to stdout.
abspath()
{
	local target="$1"
	if command -v realpath >/dev/null 2>&1; then
		realpath -m "$target"
	elif command -v readlink >/dev/null 2>&1; then
		readlink -f "$target"
	else
		(
			cd "$(dirname "$target")" && printf '%s/%s\n' "$PWD" "$(basename "$target")"
		)
	fi
}

# Normalize target file path
TARGET_ABS_PATH="$(abspath "$TARGET_PATH")"

# Create log directory
mkdir -p "${JOB_LOGS_DIR}"
OUT_PATTERN="${JOB_LOGS_DIR}/${JOB_NAME}-%j.out"
ERR_PATTERN="${JOB_LOGS_DIR}/${JOB_NAME}-%j.err"

# Create job file from template
JOBFILE="$(mktemp -p "$PWD" tacc-job-XXXXXX.slurm)"
cp "${JOB_TEMPLATE_PATH}" "${JOBFILE}"
JOB_TASKS=$((JOB_NODES * JOB_TASKS_PER_NODE))

# Edit job file by replacing placeholders with real values
sed -i \
    -e "s|__IMAGE_URI__|${IMAGE_URI}|g" \
    -e "s|__TARGET_ABS_PATH__|${TARGET_ABS_PATH}|g" \
    -e "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" \
    -e "s|__JOB_TASKS__|${JOB_TASKS}|g" \
    -e "s|__JOB_CPUS_PER_TASK__|${JOB_CPUS_PER_TASK}|g" \
    "${JOBFILE}"

# Make job file executable
chmod +x "${JOBFILE}"

cleanup()
{
	rm -f "${JOBFILE}" 2>/dev/null || true
}
trap cleanup EXIT

jobid_raw="$(
	sbatch \
		--parsable \
		--chdir "$PWD" \
		-A "${TACC_ACCOUNT}" \
		-J "${JOB_NAME}" \
		-p "${TACC_PARTITION}" \
		-t "${JOB_TIME}" \
		-N "${JOB_NODES}" \
		-n "${JOB_TASKS}" \
        -c "${JOB_CPUS_PER_TASK}" \
		-o "${OUT_PATTERN}" \
		-e "${ERR_PATTERN}" \
		"${JOBFILE}" 2>&1
)"
jobid="$(printf '%s\n' "$jobid_raw" | awk 'NF{last=$0}END{print last}' | cut -d';' -f1 | tr -d '[:space:]')"

if [[ ! "$jobid" =~ ^[0-9]+$ ]]; then
	echo "[SUBMITTER][FATAL] Failed to parse job id from sbatch output:" >&2
	printf '%s\n' "$jobid_raw" >&2
	exit 1
fi

out_file="${OUT_PATTERN//%j/${jobid}}"
err_file="${ERR_PATTERN//%j/${jobid}}"

echo "Submitted job ${jobid}"
echo "  Script    : ${TARGET_ABS_PATH}"
echo "  ProjRoot  : ${PROJECT_ROOT}"
echo "  stdout    : ${out_file}"
echo "  stderr    : ${err_file}"

tail -n +1 -F --retry "${out_file}" | sed -u 's/^/[STDOUT] /' &
T1=$!
tail -n +1 -F --retry "${err_file}" | sed -u 's/^/[STDERR] /' &
T2=$!

# CTRL+C detaches from tails only
on_int()
{
	echo
	echo "Detaching from logs. Job ${jobid} continues to run."
	kill "${T1}" "${T2}" 2>/dev/null || true
	wait "${T1}" "${T2}" 2>/dev/null || true
	exit 0
}
trap on_int INT

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

