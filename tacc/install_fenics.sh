#!/bin/bash

module purge
module load gcc

set -e
set -x

# Check if INSTALL_PREFIX is set
if [ -z "$INSTALL_PREFIX" ]; then
    echo "Error: INSTALL_PREFIX environment variable is not set."
    exit 1
fi

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPTS_DIR="$SCRIPT_DIR/build_scripts"

# Run the build scripts in order
bash "$BUILD_SCRIPTS_DIR/01_install_mpi4py.sh"
bash "$BUILD_SCRIPTS_DIR/02_install_h5py.sh"
bash "$BUILD_SCRIPTS_DIR/03_install_petsc4py.sh"
bash "$BUILD_SCRIPTS_DIR/04_install_slepc4py.sh"
bash "$BUILD_SCRIPTS_DIR/05_install_pybind11.sh"
bash "$BUILD_SCRIPTS_DIR/06_install_boost.sh"
bash "$BUILD_SCRIPTS_DIR/07_install_fenics.sh"

echo "FEniCS installation completed successfully!"
