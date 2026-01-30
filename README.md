# Instructions

A modified version of the `tacc-mvapich2.3-ib-python3.12` image
from [hippylib/tacc-containers](https://github.com/hippylib/tacc-containers) can be compiled using the Dockerfile
located in the `tacc` directory. This image is preinstalled with `FeniCS_ii`,`graphnics`, and `Gmsh`, as well as other
images relevant to legacy DOLFIN/FEniCS.

To run locally:

```
docker run \
    -it \
    --env MV2_SMP_USE_CMA=0 \
    --env MV2_ENABLE_AFFINITY=0 \
    --volume "$(pwd):/home/fenics" \
    ghcr.io/isosafrasaurus/tacc-mvapich2.3-python3.12-graphnics:latest \
    /bin/bash
```
