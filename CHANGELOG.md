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
  shear gives a positive real part. Previously these two geometries returned
  `-gamma_t`, which contradicted both the documented behaviour and the `3dbox`
  geometry, whose sign is unchanged. Every polar leg of every other correlator
  already used the tangential basis, so this brings `NGCorrelation` into line with
  the rest of the package; the convention is now stated in `BinnedNPCF`.
* **`NGCorrelation.computeMapNap` follows**, and now returns `+<Nap Map>` on all
  geometries. Combining `NGCorrelation` output with `GNNCorrelation` or
  `GNNNCorrelation_NoTomo` no longer needs a manual sign flip.
* **`NGGCorrelation.computeNMM(basis='MapMx')` returns four components instead of
  three.** The two complex correlators carry four real degrees of freedom, but
  `<N Map Mx>` was being discarded. It is now returned as the fourth component;
  it agrees with the third only when `do_multiscale=False`. The array is real
  rather than complex, since all four components are.

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

* `saveinst` dropped attributes that a reloaded instance needs: the clustering
  three-point function `zeta` of `NNNCorrelation`, the redshift weighting of
  `GNNCorrelation`, and `thetabatchsize_max` of the fourth-order estimators. On
  `loadinst` these silently fell back to their defaults.

#### Changed

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
