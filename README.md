<p align="center">
  <img src="docs/orpheus_logov1.png" alt="Orpheus logo" width="500"/>
</p>

Orpheus is python package for the calculation of second- third- and fourth-order correlation functions of scalar and polar fields such as weak lensing shear. To efficiently perform the calculations, orpheus makes use of a mulitpole decomposition of the N>2 correlation functions and uses parallelized C code for the heavy lifting.

## Installation, Documentation and Examples
Installation steps, documentation and examples are provided at [orpheus.readthedocs.io](https://orpheus.readthedocs.io/).

### Installation
orpheus is on pypi:
```shell
pip install orpheus-npcf
```

Building from source instead compiles a parallelised C/C++ extension at install
time. Besides the python requirements you therefore need a C compiler with
OpenMP support (GCC, or Clang together with `libomp`). No external C libraries
are needed: the small HEALPix subset used by the curved-sky estimators is
shipped with orpheus, in `orpheus/src/healpix/`.

Then clone the repository via
```shell
git clone git@github.com:lporth93/orpheus.git
```
or
```shell
git clone https://github.com/lporth93/orpheus.git
```
navigate to the cloned directory and install:
```shell
cd orpheus
pip install .
```

### Documentation
In the [documentation](https://orpheus.readthedocs.io/) you find more information about the algorithms and approximation schemes employed in orpheus, as well as a series of jupyter notebooks that give examples of how to use the different estimators implemented in orpheus.

## Using the code
As at this moment there is no dedicated orpheus paper, please cite the paper that introduced the functionality implemented in orpheus:
 * If you use the three-point functionality, please cite [Porth+2024](https://doi.org/10.1051/0004-6361/202347987)
 * If you use the four-point functionality, please cite [Porth+2025](https://arxiv.org/abs/2509.07974)
 * If you use the direct estimator functionality, please cite [Porth & Smith 2022](https://doi.org/10.1093/mnras/stab2819)

In each of the papers, you can find the main equations implemented in orpheus.