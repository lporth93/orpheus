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

### Unreleased

#### Changed — please read before upgrading

These change the numbers that existing scripts get back.

* **`NGCorrelation.xi` has flipped sign for `flat2d` and `spherical` geometries.**
  It now returns the tangential basis `gamma_t + i*gamma_x`, so a pure tangential
  shear gives a positive real part. This brings `NGCorrelation` in line with
  the general orpheus convention which has been documented in in `BinnedNPCF`.
* **`NGCorrelation.computeMapNap` follows**, and now returns `+<Nap Map>` on all
  geometries. Combining `NGCorrelation` output with `GNNCorrelation` or
  `GNNNCorrelation_NoTomo` no longer needs a manual sign flip.
* **`NGGCorrelation.computeNMM(basis='MapMx')` returns four components instead of
  three.** The two complex correlators carry four real degrees of freedom, but
  `<N Map Mx>` was being discarded. It is now returned as the fourth component;
  it agrees with the third only when `do_multiscale=False`. The array is real
  rather than complex, since all four components are.
* **`GGGGCorrelation_NoTomo.process` refuses `lowmem=False`.** That kernel returns a
  normalisation one to two per cent away from the true quadruplet counts, measured
  against `NNNNCorrelation`, which agrees with itself across both of its own branches.
  It was reached by default for a multipole-only run, so results obtained that way
  carry the error. Pass `lowmem=True`, which is exact.
* **`tree_resos` must begin with `0.`, the discrete resolution.** The tree machinery
  already assumed it -- `tree_redges[0]` is anchored at `min_sep` and only
  `tree_resos[1:]` is gridded -- so a list without it was misread and crashed in C.
  It is now rejected on construction. Relatedly, a large `rmin_pixsize` no longer
  drops that entry when selecting resolutions.
* **A spatial hash of more than 1e9 cells raises.** A cell size far finer than the
  footprint asks for an allocation that fails inside C; the number of cells is now
  checked against the extent first, and the message names both.

#### Added

* `saveinst` and `loadinst` for the direct estimators. `Direct_Map3Unequal`,
  `Direct_MapnEqual` and `Direct_NapnEqual` now serialise their configuration and
  their measured statistics the same way as the correlation function classes do.
* The direct estimators keep their results on the instance instead of only
  returning them, so a reloaded archive carries the aperture statistics.
* A readable error when a direct estimator is handed a catalog without an angular
  mask, which is where the aperture centers are drawn from. This previously failed
  with an `AttributeError` inside the regridding.

#### Fixed

* `tree_resos=[0.]` crashed with `OverflowError: cannot convert float infinity to
  integer`. The spatial hash takes its cellsize from the coarsest tree resolution,
  which is zero for a fully discrete tree; it now falls back to a fraction of the
  search radius.
* `GNNNCorrelation_NoTomo.process` raised `TypeError` unless `lowmem` was passed
  explicitly. It defaults to `None`, which is used both as a branch and as a 0/1
  multiplier on the verbosity, and the latter fails on `None`. It is now resolved to
  a bool first, selecting the same branches it did before.
* `saveinst` dropped attributes that a reloaded instance needs: the clustering
  three-point function `zeta` of `NNNCorrelation`, the redshift weighting of
  `GNNCorrelation`, and `thetabatchsize_max` of the fourth-order estimators. On
  `loadinst` these silently fell back to their defaults.
* `GNNNCorrelation_NoTomo` returned an array of zeros for `method='BaseTree'` and
  `method='DoubleTree'`, without raising. It never narrowed `methods_avail`, so it
  inherited all four schemes from `BinnedNPCF` while implementing two. It now
  declares `['Discrete', 'Tree']` and defaults to `'Tree'`; the previous default was
  `'DoubleTree'`, one of the two that produced nothing.
* An invalid `statistics` argument to a fourth-order `process` reported the `repr` of
  a lambda rather than the offending value, at six call sites.
* The out-of-bounds handlers for `minresoind_leaf` and `maxresoind_leaf` raised
  `AttributeError` from inside the error path, naming three attributes that do not
  exist. They now clamp and report as intended.
* `computeMap4` printed a progress bar unconditionally from C. It is now gated on
  `verbosity`, like every other kernel.

#### Changed

* `NGCorrelation` no longer defaults to `method='Discrete'`. Second-order statistics
  have a single algorithm, so the approximation schemes never applied there and the
  argument had no effect; the default now matches `NNCorrelation` and
  `GGCorrelation`. The accuracy is set by `tree_resos` and `rmin_pixsize`, and the
  exact estimator corresponds to `tree_resos=[0.]`.
* `BinnedNPCF.loadinst` matches the saved keys against the signature of the
  concrete constructor, so child-specific arguments no longer need to be listed
  by hand.

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
