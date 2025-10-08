#!/usr/bin/env bash
set -Eeuo pipefail

# Configurations
PROJECT_ROOT="$WORK/3d-1d"
TACC_ACCOUNT="ASC22053"
TACC_PARTITION="skx-dev"
JOB_NAME="3d-1d-calibrate"
JOB_TIME="00:30:00"
JOB_NODES=1
JOB_TASKS_PER_NODE=8
JOB_LOGS_DIR="$PROJECT_ROOT/logs"
IMAGE_URI="docker://ghcr.io/isosafrasaurus/tacc-mvapich2.3-python3.12-graphnics:latest"

# Parameters
RUN_REL="${1}"

# Resolve symbolic path into absolute unnormalized path
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

if [[ ! -f "$RUN_REL" ]]; then
	echo "[WRAPPER][FATAL] Local run script not found at path: '$RUN_REL'" >&2
	exit 1
fi
RUN_ABS="$(abspath "$RUN_REL")"

mkdir -p "${JOB_LOGS_DIR}"
OUT_PATTERN="${JOB_LOGS_DIR}/${JOB_NAME}-%j.out"
ERR_PATTERN="${JOB_LOGS_DIR}/${JOB_NAME}-%j.err"

JOBFILE="$(mktemp -p "$PWD" tacc-job-XXXXXX.sh)"
JOB_TEMPLATE="./tacc-job.template.sh"
if [[ ! -f "${JOB_TEMPLATE}" ]]; then
	echo "[WRAPPER][FATAL] Job template not found at ${JOB_TEMPLATE}" >&2
	exit 1
fi
cp "${JOB_TEMPLATE}" "${JOBFILE}"

JOB_TASKS=$((JOB_NODES * JOB_TASKS_PER_NODE))

sed -i \
	-e "s|__IMAGE_URI__|${IMAGE_URI}|g" \
	-e "s|__RUN_ABS__|${RUN_ABS}|g" \
	-e "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" \
	-e "s|__TASKS__|${JOB_TASKS}|g" \
	"${JOBFILE}"

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

echo "Submitted job ${jobid}"
echo "  Script    : ${RUN_ABS}"
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

