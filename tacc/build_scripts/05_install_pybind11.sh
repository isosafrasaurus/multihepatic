#!/bin/bash
set -e
set -x

# Install pybind11
git clone https://github.com/pybind/pybind11.git /tmp/pybind11-src
cd /tmp/pybind11-src
TAGS=($(git tag -l --sort=-version:refname "v[0-9].[0-9]*.[0-9]"))
echo "Latest tag is ${TAGS[0]}"
git checkout ${TAGS[0]}
cmake \
    -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX \
    -D PYBIND11_TEST=off \
    .
make -j $(nproc)
make install
PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --user .
cd && rm -rf /tmp/pybind11-src

# Install nanobind
git clone --recursive https://github.com/wjakob/nanobind.git /tmp/nanobind-src
cd /tmp/nanobind-src
TAGS=($(git tag -l --sort=-version:refname "v[0-9].[0-9]*.[0-9]"))
echo "Latest tag is ${TAGS[0]}"
git checkout ${TAGS[0]}
git submodule update --recursive
cmake \
    -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX \
    -D NB_TEST=off \
    .
PYTHONUSERBASE=$INSTALL_PREFIX python3 -m pip install --user .
cd && rm -rf /tmp/nanobind-src
