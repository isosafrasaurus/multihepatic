#!/bin/bash
set -e
set -x

# Install HDF5
git clone https://github.com/HDFGroup/hdf5.git /tmp/hdf5-src
cd /tmp/hdf5-src
TAGS=($(git tag -l --sort=-version:refname "hdf5-1_14_[0-9][0-9]"))
if [ ${#TAGS[@]} -eq 0 ]; then
    TAGS=($(git tag -l --sort=-version:refname "hdf5-1_14_[0-9]"))
fi
echo "Latest tag in the v1.14 series is ${TAGS[0]}"
git checkout ${TAGS[0]}
./configure \
    --enable-parallel \
    --enable-hl \
    --enable-build-mode=production \
    --enable-shared \
    --with-pic \
    --prefix=$INSTALL_PREFIX
make -j $(nproc)
make install
cd && rm -rf /tmp/hdf5-src

# Install h5py
git clone https://github.com/h5py/h5py.git /tmp/h5py-src
cd /tmp/h5py-src
CC=mpicc HDF5_MPI="ON" HDF5_DIR=$INSTALL_PREFIX PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --no-binary=h5py --user .
cd && rm -rf /tmp/h5py-src
