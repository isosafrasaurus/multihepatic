#!/bin/bash
set -e
set -x

# Install SLEPc
git clone https://gitlab.com/slepc/slepc.git /tmp/slepc-src
cd /tmp/slepc-src
git checkout main # Using main branch for latest version
./configure --prefix=$INSTALL_PREFIX
make SLEPC_DIR=$PWD PETSC_DIR=$INSTALL_PREFIX
make SLEPC_DIR=$PWD PETSC_DIR=$INSTALL_PREFIX install

# Install slepc4py
cd /tmp/slepc-src/src/binding/slepc4py/
PETSC_DIR=$INSTALL_PREFIX SLEPC_DIR=$INSTALL_PREFIX PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --check-build-dependencies --no-build-isolation --user .
cd && rm -rf /tmp/slepc-src
