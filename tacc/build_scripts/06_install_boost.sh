#!/bin/bash
set -e
set -x

# Install boost
git clone --recursive https://github.com/boostorg/boost.git /tmp/boost-src
cd /tmp/boost-src
TAGS=($(git tag -l --sort=-version:refname "boost-1.[0-9][0-9].[0-9]"))
echo "Latest tag in the v1 series is ${TAGS[0]}"
git checkout ${TAGS[0]}
git submodule update --recursive
BOOST_TOOLSET="gcc"
BOOST_CXXFLAGS="cxxflags=-std=c++17"
BOOST_PYTHON_EXEC="$(which python3)"
export CPATH=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
./bootstrap.sh --with-toolset=$BOOST_TOOLSET --with-python=$BOOST_PYTHON_EXEC --without-icu --without-libraries="locale" --prefix="$INSTALL_PREFIX"
./b2 toolset=$BOOST_TOOLSET $BOOST_CXXFLAGS --disable-icu --prefix="$INSTALL_PREFIX" -j $(nproc) stage release
./b2 install toolset=$BOOST_TOOLSET $BOOST_CXXFLAGS --prefix="$INSTALL_PREFIX"
cd && rm -rf /tmp/boost-src
