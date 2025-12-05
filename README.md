# Instructions

### Via Consolidated Image

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

### References

```
@article{AlnaesEtal2015,
title     = {The {FEniCS} Project Version 1.5},
author    = {Alnaes, Martin S. and Blechta, Jan and Hake, Johan and Johansson, August and Kehlet, Benjamin and Logg, Anders and Richardson, Chris N. and Ring, Johannes and Rognes, Marie E. and Wells, Garth N.},
journal   = {Archive of Numerical Software},
year      = {2015},
volume    = {3},
doi       = {10.11588/ans.2015.100.20553},
}

@book{LoggEtal2012,
title     = {Automated Solution of Differential Equations by the Finite Element Method},
author    = {Logg, Anders and Mardal, Kent-Andre and Wells, Garth N. and others},
year      = {2012},
doi       = {10.1007/978-3-642-23099-8},
publisher = {Springer},
}

@article{LoggWells2010,
title     = {{DOLFIN:} Automated Finite Element Computing},
author    = {Logg, Anders and Wells, Garth N.},
journal   = {{ACM} Transactions on Mathematical Software},
year      = {2010},
volume    = {37},
doi       = {10.1145/1731022.1731030},
}

@incollection{LoggEtal_10_2012,
title     = {{DOLFIN:} a {C++/Python} Finite Element Library},
author    = {Logg, Anders and Wells, Garth N. and Hake, Johan},
year      = {2012},
booktitle = {Automated Solution of Differential Equations by the Finite Element Method},
publisher = {Springer},
series    = {Lecture Notes in Computational Science and Engineering},
volume    = {84},
chapter   = {10},
editor    = {Logg, Anders and Mardal, Kent-Andre and Wells, Garth N.},
}

@article{KirbyLogg2006,
title     = {A Compiler for Variational Forms},
author    = {Kirby, Robert C. and Logg, Anders},
journal   = {{ACM} Transactions on Mathematical Software},
year      = {2006},
volume    = {32},
doi       = {10.1145/1163641.1163644},
}

@incollection{LoggEtal_11_2012,
title     = {{FFC:} the {FEniCS} Form Compiler},
author    = {Logg, Anders and {\O}lgaard, Kristian B. and Rognes, Marie E. and Wells, Garth N.},
year      = {2012},
booktitle = {Automated Solution of Differential Equations by the Finite Element Method},
publisher = {Springer},
series    = {Lecture Notes in Computational Science and Engineering},
volume    = {84},
chapter   = {11},
editor    = {Logg, Anders and Mardal, Kent-Andre and Wells, Garth N.},
}

@article{OlgaardWells2010,
title     = {Optimisations for Quadrature Representations of Finite Element Tensors Through Automated Code Generation},
author    = {{\O}lgaard, Kristian B. and Wells, Garth N.},
journal   = {{ACM} Transactions on Mathematical Software},
year      = {2010},
volume    = {37},
doi       = {10.1145/1644001.1644009},
}

@article{Kirby2004,
title     = {Algorithm 839: {FIAT,} a New Paradigm for Computing Finite Element Basis Functions},
author    = {Kirby, Robert C.},
journal   = {{ACM} Transactions on Mathematical Software},
year      = {2004},
volume    = {30},
doi       = {10.1145/1039813.1039820},
pages     = {{502--516}},
}

@incollection{kirby2010,
title     = {{FIAT:} Numerical Construction of Finite Element Basis Functions},
author    = {Kirby, Robert C.},
year      = {2012},
booktitle = {Automated Solution of Differential Equations by the Finite Element Method},
publisher = {Springer},
series    = {Lecture Notes in Computational Science and Engineering},
volume    = {84},
chapter   = {13},
editor    = {Logg, Anders and Mardal, Kent-Andre and Wells, Garth N.},
}

@incollection{MardalHaga2012BlockPrec,
author    = {Mardal, K.-A. and Haga, J. B.},
title     = {Block preconditioning of systems of {PDE}s},
booktitle = {Automated Solution of Differential Equations by the Finite Element Method},
editor    = {Logg, A. and Mardal, K.-A. and Wells, G. N. and others},
publisher = {Springer},
year      = {2012},
doi       = {10.1007/978-3-642-23099-8},
url       = {http://fenicsproject.org/book}
}

@InProceedings{10.1007/978-3-030-55874-1_63,
author="Kuchta, Miroslav",
editor="Vermolen, Fred J.
and Vuik, Cornelis",
title="Assembly of Multiscale Linear PDE Operators",
booktitle="Numerical Mathematics and Advanced Applications ENUMATH 2019",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="641--650",
abstract="In numerous applications the mathematical model consists of different processes coupled across a lower dimensional manifold. Due to the multiscale coupling, finite element discretization of such models presents a challenge. Assuming that only singlescale finite element forms can be assembled we present here a simple algorithm for representing multiscale models as linear operators suitable for Krylov methods. Flexibility of the approach is demonstrated by numerical examples with coupling across dimensionality gap 1 and 2. Preconditioners for several of the problems are discussed.",
isbn="978-3-030-55874-1"
}

@article{graphnics2022gjerde,
author = {{Gjerde}, Ingeborg G.},
title = "{Graphnics: Combining FEniCS and NetworkX to simulate flow in complex networks}",
journal = {arXiv e-prints},
year = 2022,
month = dec,
archivePrefix = {arXiv},
eprint = {2212.02916}
}

@misc{FEMonColabWebsite,
author       = {Ballarin, Francesco},
title        = {{FEM} on {C}olab},
howpublished = {\url{https://fem-on-colab.github.io}},
institution  = {Università Cattolica del Sacro Cuore}
}
```
