#!/usr/bin/env bash
set -euo pipefail
cd $HOME

module purge
module load Mamba

mamba create -n cmor_mdanderson
mamba init
source ~/.bashrc

mamba activate cmor_mdanderson
mamba install -y pip fenics numpy pandas matplotlib scipy networkx tqdm vtk meshio pyvista

pip install git+https://bitbucket.org/fenics-apps/cbc.block/src/master/

git clone --single-branch -b "collapse-iter-dev" "https://github.com/MiroK/fenics_ii"
find fenics_ii/ -type f -name "*.py" -exec perl -pi -e 's/\bufl_legacy\b/ufl/g' {} +
pip install fenics_ii/

pip install git+https://bitbucket.org/fenics-apps/cbc.block/src/master/
pip install git+https://github.com/IngeborgGjerde/graphnics
pip install git+https://github.com/dolfin-adjoint/pyadjoint.git --upgrade

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libssl.so.3:$CONDA_PREFIX/lib/libcrypto.so.3"