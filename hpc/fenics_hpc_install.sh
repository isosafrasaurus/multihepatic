#!/usr/bin/env bash
set -euo pipefail

export PREFIX=$HOME/.venv

module purge
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load HDF5/1.14.0
module load PETSc/3.20.3
module load SLEPc
module load Cython/3.0.8

if [ ! -d "$PREFIX" ]; then
    python3 -m venv "$PREFIX"
fi
source "$PREFIX/bin/activate"

export CC=mpicc
export FC=mpif90
export HDF5_DIR=${HDF5_DIR:-$(dirname $(dirname $(which h5pcc)))}
export PETSC_DIR=${PETSC_DIR:-$PETSC_DIR}
export PETSC_ARCH=${PETSC_ARCH:-$PETSC_ARCH}
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=$PREFIX/lib/python3.11/site-packages:$CMAKE_PREFIX_PATH

python -m pip install --upgrade pip wheel cmake
python -m pip install numpy
python -m pip install --no-binary=petsc4py petsc4py==3.20.3

python -m pip install mpi4py pybind11 dev-fenics-ffc --no-cache-dir

python -m pip install h5py --no-binary=h5py

git clone https://bitbucket.org/fenics-project/dolfin.git dolfin
cd dolfin
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_PREFIX_PATH="$PREFIX;$PETSC_DIR/$PETSC_ARCH;$HDF5_DIR" 
cmake --build build -j$(nproc)
cmake --install build

cd python
python -m pip install . --no-cache-dir
