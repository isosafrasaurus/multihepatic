#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]}"
SRC_PATH="$(cd "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)/src"

export PYTHONPATH="${PYTHONPATH}:$SRC_PATH"
echo $PYTHONPATH