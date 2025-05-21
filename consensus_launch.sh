#!/usr/bin/env bash

# Define script and repository paths.
SCRIPT_PATH="${BASH_SOURCE[0]}"
REPO_PATH="$(cd "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"
SRC_PATH="$REPO_PATH/src/"

# Add src directory to PYTHONPATH.
export PYTHONPATH="${PYTHONPATH}:$SRC_PATH"
echo "Added $SRC_PATH to PYTHONPATH."

# Navigate to the repository root and run the Python script.
cd "$REPO_PATH"

python3 hpc/consensus_sweep.py \
  --sweep_name "gamma" \
  --sweep_values "np.logspace(-10, -2, 20)" \
  --maxiter_o 30 \
  --maxiter_c 50 \
  --directory "export/"