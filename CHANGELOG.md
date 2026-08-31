# Changelog

Since version 0.3 we provide a detailed changelog. The first section summarises 
what each release series provides. The second lists the individual changes that 
went into every version.

## Release highlights

### 0.5

Mainly presentation updates with a few minor code changes covering certain edge cases and 
more graceful error handling. The README was heavily rewritten to reflect the current status 
of the codebase. In particular, it features the main figures from the testsuite and from the 
newly added scaling test.

* Default build does not use `-ffast-math` to be IEEE-safe. Slightly increased performance via 
`-ffast-math` still available behind `ORPHEUS_FAST_MATH=1`
* Allocation failures on the C side raise `MemoryError` rather than segfaulting
* `autoset_tree` and the discrete spatial hash size themselves from the catalog
* Performed benchmarking tests of GGG and added main results to README
* The package root exports only orpheus' own names

### 0.4

Validation and correctness. The estimators are now checked end to end against a shear
field whose correlation functions and aperture statistics are known in closed form, which
turned up a number of bugs and pinned down the accuracy each configuration reaches.

* A two-tier test suite comparing every correlator against closed-form expressions, with
  the error budget setting its tolerances written up in
  `docs/source/notes/analytic_shear_field.pdf`
* `apodization`, which selects the window applied to the multipoles before the transform
  to real space
* Fixes to the third-order edge correction, the multiple-counting corrections of the GGGG
  and GNNN kernels, a data race in discrete GGGG, and the `NGCorrelation` flat2d sign

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

#### Fixed

* **`Direct_NapnEqual` returned zeros, or wrong numbers, for three of its four filters.**
  `getFilterU` implemented only `case 0` and `case 1` and had no `default`, so `res` kept its
  initialiser: `Sch04` and `PolyExp` came back as exactly zero for every aperture, with no
  exception and no warning, because `getFilterSuppU` returns a valid support radius for all
  four and the search therefore ran normally. Worse, `case 0` held the *`Q`* expression,
  `6 x^2 (1-x^2)` -- a copy of `getFilterQ` -- which is not a compensated filter at all
  (`\int_0^1 x U dx = 0.5`, and it must be 0), so `S98` returned plausible-looking wrong
  values. `S98` is now `9/pi (1-x^2)(1/3-x^2)`, and `PolyExp` is
  `1/pi [e^{-150x^2} + 0.5 e^{-30x^2} - 0.0233333 e^{-x^2}]`, derived from the shipped
  `getFilterQ` through `Q = 2/x^2 \int_0^x x' U dx' - U` rather than taken from a paper.
  Schirmer 2004 has no elementary `U` -- that is why the poly-exp form exists -- so
  `Direct_NapnEqual` now raises `ValueError` for it instead of integrating against zero.
  Verified against the compiled extension: `S98` compensation is `-9.5e-18` (relative
  `6.7e-17`), and all three filters reproduce `getFilterQ` to `1e-13` up to the global factor
  of pi by which the `Q` and `U` branches differ. The smoke test now parametrises over the
  supported filters; it previously only ever ran the `C02` default, which is why this
  survived.
* **The error branch of `getFilterQ` fell through into a live computation.** `default:`
  printed a complaint, set `res = 0`, and then fell into `case 4:`, which overwrote it with a
  poly-exp value -- so an unrecognised filter returned a plausible number after announcing
  that it was wrong. Unreachable from the python layer, where nothing maps to 4.

#### Known issue

* **The `N4`/`Nap4` aperture kernel is unfinished.** `fourpcf2N4correlators` sets
  `F_1 = F_2 = F_3 = 1` and computes `nextF = 1/64 * measure * (F_1+F_2+F_3) * exp`, so the
  filter reduces to a constant and the ~40 lines of `q`/`y` algebra above it are computed and
  discarded. It is reached from `nnnn_reconstruct_batch`, and `"N4"`, `"Nap4"`, `"N4c"` and
  `"Nap4c"` are all selectable statistics on `NNNNCorrelation_NoTomo`. The
  `-Wunused-but-set-variable` warnings on its declarations are the symptom and are left
  standing on purpose; the function carries a comment saying so.

#### Changed

* Compiler warnings under `-Wall -Wextra` went from 180 to 52, without suppressing anything.
  The 128 that went were dead struct-field hoists left behind by the argument-struct
  unification and leftover debug scaffolding -- including an `omp_get_wtime()` call per
  galaxy in an inner loop whose result nothing read. Of the 52 that remain, 22 are the
  unfinished kernel above and 30 are parameters fixed by the ctypes ABI, which cannot be
  dropped without shifting every positional argument after them.
* `mymin` and `mymax` were defined in eight translation units, all without the outer
  parentheses, so `2*mymin(a,b)` would not have meant what it reads as. All 80 call sites
  happened to be safe. They are now defined once, correctly, in `utils.h`.
* All 64 `restype` declarations said `ct.c_void_p` for C functions that return `void`. They
  now say `None`.
* The library-loading block was copy-pasted into `catalog.py`, `direct.py` and
  `npcf_base.py`, each with a dead `target_path`, and located the extension with
  `glob.glob(...)[0]` -- a bare `IndexError` on the most likely first failure a new user
  hits. It is now one `_load_clib()` in `utils.py` that raises `ImportError` naming the
  directory it searched.
* The direct estimators wrote progress to stdout on every aperture radius with no way to turn
  it off. `DirectEstimator` now takes `verbosity`, following the `npcf_base.py` convention.
* `joblib` and `threadpoolctl` are declared. `patchutils` imports both at module scope and is
  imported by `__init__.py`, so they were needed to import the package at all; it worked only
  because `scikit-learn` happens to require them.

#### Documentation and CI

* **Coverage is now actually reported.** The CI job installed `pytest-cov` and then never
  passed `--cov`. Adding the flag alone reports 0%: the tests run against the installed wheel
  -- `tests/conftest.py` drops the checkout from `sys.path` -- while `--cov=orpheus` resolves
  to the source directory. `pyproject.toml` now sets `source_pkgs` and maps the two locations
  onto each other, which reproduces the 54% on the direct estimators quoted for 0.5.0.
* `GNNN_tutorial.ipynb` is no longer tracked. It was an 11 MB scratch notebook with execution
  counts out of order, twelve stored tracebacks and 16 hardcoded institute paths, in no
  toctree but named as though it were a tutorial. The README no longer claims every
  correlator has one. The empty `_convergencetests_NN.ipynb` is likewise untracked.
* Seven published notebooks carried stored output streams quoting a local conda prefix or an
  institute data path, which were live on the docs site. The offending streams are stripped;
  the notebook sources never referenced those paths.
* The sanitizer jobs say `(advisory)` in their names. Both carry `continue-on-error` and
  cannot fail, which the changelog admitted but the workflow did not.

### 0.5.0 — 2026-08-25

#### Changed — please read before upgrading

* **`-ffast-math` is no longer used by default.** The 0.4.0 notes observed that no runtime
  guard can be relied on while it is set; this removes the cause. The kernels now build with
  `-O3 -fno-math-errno -fno-trapping-math -fcx-limited-range`, which keeps IEEE semantics for
  `NaN` and `Inf` -- so the `isfinite` guards in the derived-statistics kernels actually run --
  and stops the compiler reassociating OpenMP reductions, which makes results independent of
  the thread count. Measured cost against the old build is 4% on `GGCorrelation` and 9% on
  `GGGCorrelation` with `DoubleTree`; the analytic test tier is unchanged. `-ffast-math` can be
  restored with `ORPHEUS_FAST_MATH=1 pip install .`. Each flag is probed against the selected
  compiler and dropped if it is not accepted -- apple clang rejects `-fcx-limited-range`.
* **`numba` is no longer a dependency.** It was declared but never imported, and its upper pin
  constrained `numpy` in unrelated environments. `mpmath`, which is imported lazily by the
  log-COSEBI construction, is now declared as the `cosebis` extra.
* **The package namespace exports only orpheus' own names.** `orpheus/__init__.py` used star
  imports, which republished whatever its modules had imported: `orpheus.SkyCoord`,
  `orpheus.KMeans`, `orpheus.Parallel` and 32 further third-party names were importable from
  the package root and indistinguishable from API. It now imports by name and declares
  `__all__`. All 37 of orpheus' own names are unchanged, so code that imports estimators,
  catalogs or helpers is unaffected; code that reached through orpheus for a third-party name
  should import it from its own package.

#### Fixed

* **The discrete estimators sized their spatial hash from `max_sep` alone.** `dpix` was
  `max(1., max_sep//10)`, which takes no account of how many galaxies the catalog holds or how
  many threads will walk it. `_discrete_dpix` now bounds `max_sep/10` from above by the cell
  size that yields 64 regions per thread and from below by a crude estimate of the mean
  inter-galaxy separation, falling back to the old rule for catalogs holding fewer galaxies
  than there are regions. `GNNCorrelation` and `NGGCorrelation` take the finer of the source
  and lens values so the two hashes stay aligned.
* **Four variables could be read before being assigned.** `inv_Nbar` in `directestimator.c` is
  declared outside the tomographic loop but assigned inside it, so an `Nbar_policy` outside the
  handled set would silently reuse the previous bin's value; `toadd_Mapn_w` and `Map3_w` are
  likewise only assigned for the `weight_method` values that are handled; and the `zeta`
  interpolation in `corrfunc_fourth_derived.c` ignored the return value of `locate_lin`, which
  leaves its outputs untouched on failure. None is reachable through the python layer today.
  Found by building with `-Wall -Wextra`, which is now on.
* **`autoset_tree` gave sparse catalogs a far too fine cell ladder.** It estimates the number
  density by counting occupied cells on a helper grid, which was fixed at 2 arcmin. Once a
  catalog is sparse enough that most galaxies sit alone in a cell, that count tracks `ngal`
  rather than the footprint and the estimate saturates at one galaxy per cell, i.e. at
  `0.25/arcmin^2`. A lens sample at `0.01/arcmin^2` was therefore read as 25 times denser and
  handed the ladder `[0, 1, 2, 4]` where `[0, 4]` is appropriate -- four levels of tree
  overhead with nothing to group. The grid is now taken from the bounding box and `ngal`, which
  fixes the mean occupancy at nine galaxies per cell whatever the density. Source catalogs, at
  one galaxy per arcmin^2 and above, were never in the saturated regime and are unaffected.
* **`autoset_tree` could hand the kernels a tree that crashed them.** A level's radial edge is
  `rmin_pixsize*reso`, so with the default `rmin_pixsize=20` a ladder reaching the 4 arcmin
  ceiling implies an edge at 80 arcmin. Where `max_sep` was below that, `tree_redges` came out
  non-monotonic -- `[1, 2.5, 5, 10, 20, 40, 80, 15]` for `max_sep=15` -- and the negative shell
  width reached the kernels as a negative `nbinsr_reso`, i.e. as a negative allocation size.
  `GGGCorrelation` died in `Tree` with a double free and in `DoubleTree` on an allocation of
  `2**64-4416` bytes. `__init__` has always trimmed a user-supplied ladder for this reason;
  `autoset_tree` now applies the same trim. Any `max_sep` below `rmin_pixsize` times the
  coarsest cell was affected, which the default settings reach easily.
* The complexity estimate behind the leaf resolutions divided by the discrete level's zero
  cell size. The result was correct, since the infinity it produced lost the `minimum` that
  follows it, but every call raised a `divide by zero` warning. It now uses `np.divide`.

* **Radial bin indices are bounds-checked in the third-order tree kernels.** The shell guard
  there is on squared or geodesic distance while the bin index goes through `sqrt` and `log`;
  the two roundings can disagree by one bin at a shell edge, which the 0.4.0 notes flagged as
  able to corrupt the accumulators silently.
* **The advanced GG/GGG tutorial built its catalog from the Takahashi shear unnegated.** The
  T17 maps store `gamma1` and `gamma2` in the opposite sign convention to the one orpheus
  expects, so the even-order statistics were right but `Map3` came out negative. The notebook
  negates both components and its outputs were re-run from that.

#### Added

* Allocation failures are reported instead of segfaulting. The kernels allocate through
  wrappers that name the size that failed on stderr and set an error flag, the entry points
  return early rather than dereferencing a null pointer, and the python layer raises
  `MemoryError`.
* `tests/test_fast_abi.py` checks the seven `ctypes` mirrors in `multires_structs.py` against
  the layout the compiler produced, comparing field names as well as offsets -- swapping two
  fields of equal type leaves every offset intact and is otherwise invisible.
* Smoke tests for `Direct_MapnEqual` and `Direct_NapnEqual`, which had no tests at all: the
  direct estimators went from 8% statement coverage, all of it import-time, to 54%, and the
  fast tier as a whole from 53% to 60%.
* A sanitizer workflow building the kernels under ASan and UBSan, plus a weekly valgrind run.
  Advisory for now. Coverage is reported by the CI job.
* `pyproject.toml` carries the package metadata (PEP 621); `setup.py` retains only the
  extension build. `CITATION.cff` added.
* The tutorial notebooks download their own input data instead of reading it from a hardcoded
  institute path.
* `benchmarks/scaling.py`, which times the third-order estimators against maximum separation,
  number density and thread count and draws the README performance figure. Timings are cached
  per machine, so the figure can be redrawn without repeating the sweep, and `--legs` selects
  which of the three to measure.
* `figures/workflow/`, the two scripts behind the README workflow figure. They build a masked
  mock catalog from a public Takahashi ray-tracing map, measure the spherical GGG and `Map3`,
  and reduce the result to the small array the figure is drawn from.

### 0.4.0 — 2026-08-19

#### Changed — please read before upgrading

These change the numbers that existing scripts get back.

* **`NGCorrelation.xi` has flipped sign for `flat2d` and `spherical` geometries.**
  It now returns the tangential basis `gamma_t + i*gamma_x`, so a pure tangential
  shear gives a positive real part. This brings `NGCorrelation` in line with
  the general orpheus convention which has been documented in `BinnedNPCF`.
* **`NGCorrelation.computeMapNap` follows**, and now returns `+<Nap Map>` on all
  geometries. Combining `NGCorrelation` output with `GNNCorrelation` or
  `GNNNCorrelation_NoTomo` no longer needs a manual sign flip.
* **`NGGCorrelation.computeNMM(basis='MapMx')` returns four components instead of
  three.** The two complex correlators carry four real degrees of freedom, but
  `<N Map Mx>` was being discarded. It is now returned as the fourth component;
  it agrees with the third only when `do_multiscale=False`. The array is real
  rather than complex, since all four components are.
* **`GGGGCorrelation_NoTomo` with `lowmem=False` returned wrong quadruplet counts and
  `Upsilon`.** It was reached by default for a multipole-only run, so results obtained
  that way carry the error and should be recomputed; see the entry under Fixed for what
  was wrong. Both kernels now agree to machine precision.
* **`multicountcorr=False` is now honoured by every `GNNN` kernel.** The `Discrete` and
  the `lowmem=True` `Tree` kernels applied the multiple-count corrections unconditionally
  while the `lowmem=False` `Tree` one already gated them, so the three disagreed whenever
  the flag was turned off. They now all follow it.
* **`tree_resos` must begin with `0.`, the discrete resolution.** The tree machinery
  already assumed it -- `tree_redges[0]` is anchored at `min_sep` and only
  `tree_resos[1:]` is gridded -- so a list without it was misread and crashed in C.
  It is now rejected on construction. Relatedly, a large `rmin_pixsize` no longer
  drops that entry when selecting resolutions.
* **A spatial hash of more than 1e9 cells raises.** A cell size far finer than the
  footprint asks for an allocation that fails inside C; the number of cells is now
  checked against the extent first, and the message names both.
* **The multipole window is now selectable, and no bin is masked on the strength of its
  reconstructed count.** Reconstructing the counts from a truncated multipole series makes
  them ring, and where the reconstruction passes near zero the division that turns
  `Upsilon` into the NPCF amplifies without bound. Cutting the offending bins treats the
  symptom: it is a one-sided selection on the divisor, so the bins surviving near the
  threshold are biased high and their correlator low, and the noise level being cut
  against is itself overestimated wherever a sharp mask puts real power at high multipole
  order -- that is, in the sparse regime the cut exists for. The window addresses the
  cause instead. `apodization='fejer'` tapers the multipoles before the transform; the
  Fejer kernel is non-negative, so the reconstructed counts inherit the positivity of the
  true ones and cannot cross zero, and the estimator becomes a weighted mean of the
  correlator over the window rather than an unbounded ratio. The default `'rect'` is the
  plain band limit and reproduces the previous reconstruction.
* **`GGGCorrelation` no longer applies a hardcoded count floor of `0.1`.** It was absolute
  rather than relative, so it depended on the weight normalisation: a triplet count scales
  as `w^3`, and rescaling the weights by 0.01 took it from masking nothing to masking three
  quarters of the grid. `GNNNCorrelation_NoTomo.multipoles2npcf` defaults `count_floor` to
  `0.` for the same reason. Neither is replaced by another cut; see `apodization` above.
* **The NPCF is divided by the real part of the reconstructed counts, not by its modulus.**
  A bin whose count reconstructs negative previously had the sign of its correlator
  flipped; it now keeps it, and is identifiable through `npcf_norm`. `GNN` and `NGG`
  already used the real part, so this aligns `GGG` and the fourth-order kernels with them.

#### Added

* `saveinst` and `loadinst` for the direct estimators. `Direct_Map3Unequal`,
  `Direct_MapnEqual` and `Direct_NapnEqual` now serialise their configuration and
  their measured statistics the same way as the correlation function classes do.
* The direct estimators keep their results on the instance instead of only
  returning them, so a reloaded archive carries the aperture statistics.
* A readable error when a direct estimator is handed a catalog without an angular
  mask, which is where the aperture centers are drawn from. This previously failed
  with an `AttributeError` inside the regridding.
* `apodization` on `BinnedNPCF`, together with the helper `mode_window` that builds the
  taper. It is applied to numerator and denominator alike, which leaves the estimator
  intact but halves the angular resolution, so at fixed `nmax` the Fejer window measures a
  known windowed correlator that has to be forward-modelled accordingly.
* `set_ringing_sigma` on `BinnedNPCF`, which records in `_sigma` the noise level of the
  band-limited count reconstruction -- the scale at which the divisor stops being
  informative, and hence the criterion for reaching for `apodization`. It is a diagnostic
  only and masks nothing. Bins carrying no multiplets at all are still set to zero on the
  C side and are reported at `verbosity > 0`; `npcf_norm` is never masked, so both classes
  of bin remain identifiable afterwards.

#### Fixed

* **`computeMap3` returned `NaN` for every odd `nbinsphi`.** The angular bin centres are
  `(k+1/2)*2*pi/nbinsphi`, so `phi = pi` is a bin centre exactly when `nbinsphi` is odd.
  There the filter's `|q_3|^2 = (y1-y2)^2/9` vanishes on the diagonal radial bins, and both
  divisions it appears in evaluate `0/0`. The singularity is removable -- numerator and
  denominator vanish together and the limit is `2*y^2/(27*R^2)` times the exponential --
  so the filters now step `phi` off the degenerate point, which recovers the limit to
  `5e-9` and never triggers otherwise. Affects `map3_filter_singleR_ggg` and
  `map3_filter_multiR_ggg`; the GNN and NGG filters divide only by aperture scales and were
  never exposed. Even values of `nbinsphi`, including every setting used by the test suite,
  were unaffected.
* **The `NaN` guards in the derived-statistics kernels never ran.** Nineteen accumulation
  loops across `corrfunc_third_derived.c`, `corrfunc_fourth_derived.c` and
  `corrfunc_fourth.c` skipped non-finite contributions with `isnan(cabs(...))`, but the
  library is built with `-ffast-math`, which implies `-ffinite-math-only` and folds those
  classifications to constants -- as it does for `isfinite`, for `v == v` and for magnitude
  comparisons. The predicate is now `isfinite`, which is the correct one for a build that
  keeps IEEE semantics, but no runtime guard can be relied on while `-ffast-math` is set;
  the fix above therefore removes the cause rather than trapping the symptom.
* **`GGGCorrelation.multipoles2npcf` ignored `is_edge_corrected`.** It passed a hardcoded
  `0` to the C kernel, so after `edge_correction()` the estimator kept dividing by the
  angle-dependent counts `N(phi)` where it should have used the monopole `N_0`. The result
  was the correctly edge-corrected correlator times `N_0/N(phi)`: unbiased in the median,
  since that factor averages to one, but carrying the full local-to-mean count contrast
  bin by bin. It reproduced across independent catalogs at `r = 0.98`, so it read as a
  large extra scatter rather than as the systematic it was. `GNNCorrelation` and
  `NGGCorrelation` passed the flag correctly and were unaffected. With the flag honoured,
  the edge-corrected estimator agrees with the direct ratio, as it must: in the
  exponential basis the coupling matrix is Toeplitz, so inverting it is the same operation
  as dividing by the angle-dependent counts.
* **`edge_correction` raised `AttributeError` on numpy 1.24 and newer**, for all three
  third-order correlators. It used `np.int` and `np.complex`, removed in that release, so
  the method could not be called at all. Five live call sites in `direct.py` carried the
  same aliases.
* **`edge_correction(ret_matrices=True)` raised `ValueError`.** The returned array was
  allocated with the wrong multipole extent, half the size of the matrices it had to hold
  for `GGG` and `GNN` and a different wrong size for `NGG`.

* **The `GGGG` coincidence corrections for `theta1==theta2` were wrong in both
  `lowmem=False` kernels.** In the `Tree` kernel the correction sat in the `elb2` loop
  rather than the `elb3` one, so it was spent on the single bin triple `(a,a,a)` instead
  of the whole `(a,a,c)` family, and the `theta1==theta3` and `theta2==theta3` blocks
  carried an `elb1!=elb2` guard that suppressed them exactly where they were also needed.
  In both the `Tree` and the `Discrete` kernel, components 6 and 7 of that block were
  copies of component 0. The `lowmem=True` helper `gggg_accum_batchUpsilon` had it right
  all along and is what the corrected blocks now match. This is the miscount that made
  `lowmem=False` disagree with the `NNNN` quadruplet counts: not the "one to two per
  cent" first recorded, which was normalised to the global peak, but up to a factor 18
  in the sparsest radial bins, and it affected the eight `Upsilon` components as well as
  the normalisation.
* **`GNNCorrelation` and `NGGCorrelation` returned arrays of zeros for schemes they never
  implemented, without raising.** Neither narrowed `methods_avail`, so both inherited all
  four schemes from `BinnedNPCF` while `process` dispatches a kernel for fewer: GNN has only
  `Discrete` and `DoubleTree`, NGG only those plus `Tree`. Selecting one of the others ran to
  completion and left the multipoles at zero, which no shape or finiteness check can see.
  They now declare what they dispatch and reject the rest on construction. This is the same
  defect recorded below for `GNNNCorrelation_NoTomo`.
* **`methods_avail` now names exactly the schemes each correlator dispatches**, everywhere.
  An audit of all ten found three further classes advertising more than they implement, none
  of them silently: `NNNCorrelation` declared four and implemented `DoubleTree` alone, raising
  `NotImplementedError` from `process`, which is now refused on construction instead; and
  `NNCorrelation`, `NGCorrelation` and `GGCorrelation` declared four schemes for which
  `method` was never read at all -- second order dispatches its one doubletree kernel
  regardless, and all four settings returned bit-identical results. Those three now declare
  `["DoubleTree"]`, which is the default they already used, so only an explicit
  `method=` naming one of the inert schemes changes behaviour, from silently ignored to
  rejected. Accuracy at second order is set by `tree_resos` and `rmin_pixsize`, as before.
* **Segfault in every third-order tree kernel when the resolution list reduced to a single
  entry.** `NNNCorrelation`, `GGGCorrelation`, `GNNCorrelation` and `NGGCorrelation` crashed
  on `method='DoubleTree'`, and `GGGCorrelation` also on `'BaseTree'`, whenever the *effective*
  `tree_resos` held one resolution. That is reached two ways: by passing `tree_resos=[0.]`, or
  by an ordinary list whose coarse entries the constructor prunes away because `max_sep` is
  small. With `min_sep=1`, `rmin_pixsize=8` and `tree_resos=[0., 2., 4.]` the list keeps two
  entries at `max_sep=60` but collapses to `[0.]` at `max_sep<=40`, so narrowing the separation
  range was enough to trigger it on otherwise unchanged arguments. With no gridded resolution
  the reduced-pixel arrays are empty, and `build_redpix_by_reso2` and the `*_update_*cache`
  helpers indexed them regardless; `setup_region_shifts` additionally read past `ngal_in_pix`,
  which sized the cross-resolution cache to zero and left it NULL for the kernel to write
  through. Those helpers exist only to feed `*_accum_crossreso`, which has no resolution pair
  to visit for a single-resolution tree, so they now return early and the same-resolution path
  carries the full result. `nresos >= 2` is untouched and bit-identical. The repaired path
  agrees with `method='Discrete'` to 1e-15, as it must: a fully discrete tree *is* the exact
  estimator, and on this configuration it reaches it about twice as fast.
* **Intermittent segfault in the `Tree` kernels of `GNNNCorrelation_NoTomo` and
  `GGGGCorrelation_NoTomo`.** Their resolution loop ran `elreso <= nresos`, one
  iteration past the `nresos` entries of the three `rshift_*` offset arrays and past
  the last edge of `reso_redges`. The extra shell spans `[max_sep, garbage]` and so
  holds no valid pair, but the offsets it reads are whatever the heap happens to
  carry, which is why the crash depended on what had run before in the same process
  rather than on the input. Where the garbage did not fault it could still admit
  pairs beyond `max_sep`, whose radial bin is not bounds-checked, so the accumulators
  could be corrupted silently. The loop now runs `elreso < nresos`, matching every
  other kernel in the library.
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
