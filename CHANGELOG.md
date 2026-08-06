# Changelog

Since version 0.3 we provide a detailed changelog. The first section summarises 
what each release series provides. The second lists the individual changes that 
went into every version.

## Release highlights

### 0.3

Estimation of second-, third- and fourth-order correlation functions of scalar
and spin-2 fields, aimed at weak lensing. The N>2 correlators are computed
through a multipole decomposition, where the heavy lifting is carried out by
parallelised C kernels.

* Second order: `NNCorrelation`, `GGCorrelation`, `NGCorrelation`
* Third order: `NNNCorrelation`, `GGGCorrelation`, `GNNCorrelation`,
  `NGGCorrelation`
* Fourth order: `NNNNCorrelation_NoTomo`, `GGGGCorrelation_NoTomo`,
  `GNNNCorrelation_NoTomo`
* Direct aperture-mass statistics: `Direct_MapnEqual`, `Direct_NapnEqual`,
  `Direct_Map3Unequal`
* Flat and curved-sky geometries, tomographic binning, and the `Discrete`,
  `Tree`, `BaseTree` and `DoubleTree` approximation schemes

## Detailed changelog

### 0.3.1 — 2026-08-06

#### Fixed

* Segmentation fault on macOS whenever a kernel ran with `nthreads > 1`. Three
  copies of `libomp.dylib` were mapped into the process, one each from healpy,
  scikit-learn and orpheus. Their weak symbols coalesce into a single definition,
  so a worker thread created by one runtime could suspend itself in the state of
  another. The Apple silicon wheels now bind to the copy healpy already provides
  instead of bundling a third; the x86_64 wheels keep their own, as the copies
  vendored there are older than the symbols the kernels reference. 
  Thanks to [@jooel](https://github.com/jooel) for reporting and verifying.

#### Added

* `orpheus.__version__`.
* A `RuntimeWarning` on macOS naming every OpenMP runtime in the process when
  more than one is found.
* A macOS smoke test in the release workflow that installs the built wheel and
  runs a correlation with worker threads, and a job that builds from source
  against a conda-forge environment. Both gate the upload to PyPI.
* A build workflow running on every push and pull request.
* This changelog.

#### Changed

* `setup.py` resolves `libomp` through `@rpath` and prefers an active conda
  environment's copy over Homebrew's.
* The installation docs gained a macOS troubleshooting section, and the conda
  instructions now pin conda-forge explicitly.
* The README no longer claims that `healpix_cxx` is required or bundled.

Releases before 0.3.1 predate this changelog.
