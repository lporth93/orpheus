# Here we check invariants that are expected to hold exactly, independent of
# estimator convergence or theoretical expectations. Thus low-res & fast tier

import numpy as np
import pytest

from orpheus.catalog import ScalarTracerCatalog, SpinTracerCatalog
from orpheus.npcf_second import NNCorrelation
from orpheus.npcf_third import GGGCorrelation, GNNCorrelation, NGGCorrelation, NNNCorrelation

from conftest import (BOXSIZE, CORRELATORS, MAX_SEP, MIN_SEP, NBINSR, NTHREADS,
                      RTOL_EXACT, TREE_ONLY, build_correlator, correlator_ids,
                      correlator_outputs, correlators, run_correlator)

SEPS = dict(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR)
ANGULAR = dict(nmaxs=4, nbinsphi=10)
TREE = dict(tree_resos=[0., 2., 4.], rmin_pixsize=8, nthreads=NTHREADS)
DISCRETE = dict(tree_resos=[0.], rmin_pixsize=8, nthreads=NTHREADS)


def _deviation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b))/np.max(np.abs(a))

# Build discrete binning setup for all 3pt/4pt correlators no matter what method chosen
def _discrete_method(spec):
    method = TREE_ONLY.get(spec.cls.__name__)
    if method is not None:
        return dict(method=method, **DISCRETE)
    return dict(method='Discrete', nthreads=NTHREADS)

# Constructor arguments on which the estimator resolves every multiplet.
def _exact_kwargs(spec):
    if spec.order == 2:
        return dict(**SEPS, tree_resos=[0.], nthreads=NTHREADS)
    return dict(**SEPS, **ANGULAR, **_discrete_method(spec))

# Setup kwargs go not compute fourth-order aperture stats
def _multipole_kwargs(spec):
    return dict(statistics='4pcf_multipole') if spec.order == 4 else {}


####################
# WEIGHT HANDLING  #
####################

# This test asserts that we get the same correlators for the follwing two catalogs
# * One with N objects with weights 2 at positions x
# * One where at each x there sit two objects with weight 1
# Note that in order to have true equality we have to disable the multiple counting 
# correction for this test!

# Correlator selection on which we run the test. 
# TODO: Add fourth order, but for this we need to implement the option to isable the
# multicountcorrs in the C code
REPEATED = correlators(orders=(2, 3))

# Create the catalogs
@pytest.fixture(scope="module")
def duplicated_catalogs():
    rng = np.random.default_rng(11)
    ngal = 2000
    pos1, pos2 = rng.uniform(0., BOXSIZE, ngal), rng.uniform(0., BOXSIZE, ngal)
    e1, e2 = rng.normal(0., .3, ngal), rng.normal(0., .3, ngal)
    weight = rng.uniform(.5, 1.5, ngal)
    dup = np.arange(0, ngal, 4)
    doubled = weight.copy()
    doubled[dup] *= 2.

    def _cat(cls, w, extra):
        rep = lambda a: np.concatenate([a, a[dup]])
        if w is doubled:
            return cls(pos1=pos1, pos2=pos2, weight=w, geometry='flat2d', **extra)
        rep_extra = {k: rep(v) for k, v in extra.items() if isinstance(v, np.ndarray)}
        return cls(pos1=rep(pos1), pos2=rep(pos2), weight=rep(w), geometry='flat2d',
                   **{**extra, **rep_extra})

    spin = dict(spin=2, tracer_1=e1, tracer_2=e2)
    scalar = dict(tracer=np.ones(ngal))
    return dict(
        shear_doubled=_cat(SpinTracerCatalog, doubled, spin),
        shear_repeated=_cat(SpinTracerCatalog, weight, spin),
        scalar_doubled=_cat(ScalarTracerCatalog, doubled, scalar),
        scalar_repeated=_cat(ScalarTracerCatalog, weight, scalar))


# Do the test
@pytest.mark.parametrize("spec", REPEATED, ids=correlator_ids(REPEATED))
def test_treats_a_repeated_tracer_as_extra_weight(spec, duplicated_catalogs):
    c = duplicated_catalogs
    out = []
    for tag in ('doubled', 'repeated'):
        inst = build_correlator(spec, multicountcorr=False, **_exact_kwargs(spec))
        run_correlator(spec, inst, c['shear_%s'%tag], c['scalar_%s'%tag])
        out.append([np.asarray(getattr(inst, f)) for f in correlator_outputs(spec)])
    for name, a, b in zip(correlator_outputs(spec), *out):
        assert _deviation(a, b) < RTOL_EXACT, (name, _deviation(a, b))


#############################
# GLOBAL E/B PHASE ROTATION #
#############################

# This test asserts that the estimator correctly picks up a global shear field rotation
# In particular, letting gamma --> gamma*exp(i*chi) we expect a phase modification by
# exp(i*(p-q)*chi) for p unconjugated and q conjugated shears in the correlator.

# Get (p-q) per correlator component for all correlators on which we run the tests
PHASE_LEGS = {
    'NGCorrelation':          (1,),
    'GGCorrelation':          (0, 2),
    'GNNCorrelation':         (1,),
    'NGGCorrelation':         (2, 0),
    'GGGCorrelation':         (3, 1, 1, 1),
    'GNNNCorrelation_NoTomo': (1,),
    'GGGGCorrelation_NoTomo': (4, 2, 2, 2, 2, 0, 0, 0),
}
PHASE = [s for s in CORRELATORS if s.cls.__name__ in PHASE_LEGS]

# Rotation angle for this test
CHI_ROT = .7

# Generate SpinTracerCatalog rotated by chi wrt an original one
def _rotated(cat, chi):
    e = (np.asarray(cat.tracer_1) + 1j*np.asarray(cat.tracer_2))*np.exp(1j*chi)
    return SpinTracerCatalog(spin=2, pos1=cat.pos1, pos2=cat.pos2, tracer_1=e.real,
                             tracer_2=e.imag, weight=cat.weight, geometry='flat2d')


# Do the test
@pytest.mark.parametrize("spec", PHASE, ids=correlator_ids(PHASE))
def test_follows_the_global_shear_phase(spec, small_catalogs):
    shear, scalar = small_catalogs
    out = []
    # Do the measurement on reference and rotated cat
    for cat in (shear, _rotated(shear, CHI_ROT)):
        inst = build_correlator(spec, **_exact_kwargs(spec))
        run_correlator(spec, inst, cat, scalar, **_multipole_kwargs(spec))
        fields = correlator_outputs(spec)
        if len(fields) > 1:
            out.append([np.asarray(getattr(inst, f)) for f in fields])
        else:
            out.append(list(np.asarray(getattr(inst, fields[0]))))
    # Check the phase modification
    for i, legs in enumerate(PHASE_LEGS[spec.cls.__name__]):
        ref, rot = np.asarray(out[0][i]), np.asarray(out[1][i])
        scale = np.max(np.abs(ref))
        dev = np.max(np.abs(rot - np.exp(1j*legs*CHI_ROT)*ref))/scale
        assert dev < RTOL_EXACT, (legs, dev)
        


############################
# DoubleTree --> BaseTree  #
############################

# Check whether DoubleTree with maxresoind_leaf=0 produces the same result as BaseTree
# Note that a pure Tree cannot be obtained from DoubleTree at the moment: Tree pins the
# base to be discrete and no knob does that.
# Only use GGG is the only correlator that declares BaseTree in methods_avail.
def test_doubletree_with_a_pinned_leaf_band_reproduces_basetree(shear_catalog):
    kwargs = dict(n_cfs=4, **SEPS, **ANGULAR, **TREE)
    out = []
    for extra in ({}, dict(maxresoind_leaf=0)):
        inst = GGGCorrelation(method='DoubleTree', **kwargs, **extra)
        inst.process(shear_catalog, dotomo=False)
        out.append(np.asarray(inst.npcf_multipoles))
    ref = GGGCorrelation(method='BaseTree', **kwargs)
    ref.process(shear_catalog, dotomo=False)
    ref = np.asarray(ref.npcf_multipoles)
    # the unpinned run has to disagree, or the pinned one proves nothing
    assert _deviation(ref, out[0]) > 1e-3, "the leaf band is not changing anything"
    assert _deviation(ref, out[1]) < RTOL_EXACT, _deviation(ref, out[1])


####################
# LOWMEM CODE PATH #
####################

# The fourth-order kernels have two implementations of the same multipole sum, and
# `lowmem` selects between them, trading compute time for lower memory consumption.
# Make sure that they agree.

# Selection of correlators for this test. 
LOWMEM = correlators(orders=4)

# Generate two small test catalogs
@pytest.fixture(scope="module")
def small_catalogs():
    rng = np.random.default_rng(12)
    ngal = 2000
    pos1, pos2 = rng.uniform(0., BOXSIZE, ngal), rng.uniform(0., BOXSIZE, ngal)
    shear = SpinTracerCatalog(spin=2, pos1=pos1, pos2=pos2,
                              tracer_1=rng.normal(0., .3, ngal),
                              tracer_2=rng.normal(0., .3, ngal),
                              weight=np.ones(ngal), geometry='flat2d')
    scalar = ScalarTracerCatalog(pos1=pos1, pos2=pos2, tracer=np.ones(ngal),
                                 weight=np.ones(ngal), geometry='flat2d')
    return shear, scalar


# Run test
# TODO: GNNN FAILS SLIGNTLY WHEN ENAABLING MULTIPLE COUNTIG CORRS, TREE IS CORRECT
@pytest.mark.parametrize("spec", LOWMEM, ids=correlator_ids(LOWMEM))
def test_lowmem_matches_the_highmem_kernel(spec, small_catalogs):
    # Build catalogs and process them
    shear, scalar = small_catalogs
    out = []
    for lowmem in (True, False):
        inst = build_correlator(spec, **SEPS, **TREE, **ANGULAR, multicountcorr=False)
        run_correlator(spec, inst, shear, scalar, statistics='4pcf_multipole', lowmem=lowmem)
        norm = getattr(inst, 'npcf_multipoles_norm', None)
        out.append((np.asarray(inst.npcf_multipoles),
                    None if norm is None else np.asarray(norm)))
    # Assertions for numerator. As this can be noisy we use deviation wrt peak metric
    assert _deviation(out[0][0], out[1][0]) < RTOL_EXACT
    # Assertions for denomonator (we exclude NNNN as this has only a numerator)
    if out[0][1] is None: 
        return
    assert _deviation(out[0][1], out[1][1]) < RTOL_EXACT
    # Make sure that we have >0 counts in each bin; this can be done by N_0
    # Here we can also use the proper elementwise-ratio-metric
    _zero = out[0][1].shape[0]//2
    lo, hi = out[0][1][_zero, _zero].real, out[1][1][_zero, _zero].real
    assert np.all(lo > 0.), "n=0 counts must be positive for the ratio to mean anything"
    assert np.max(np.abs(hi/lo - 1.)) < RTOL_EXACT


##############################
# OUTPUT IS ACTUALLY WRITTEN #
##############################

# This test makes sure that running the .process method on any correlator fills
# cls.npcf_multipoles with a non-zero array. So it is a bit more shartp than
# the tests in somoketest

# Correlators to use for this test. Only do 3pt and 4pt as 2pt doesnt have multipoles
thiscorrelators = correlators(orders=(3, 4))

def _nonzero_params():
    out = []
    for spec in thiscorrelators:
        for method in build_correlator(spec, **SEPS, **TREE, **ANGULAR).methods_avail:
            key = (spec.cls.__name__, method)
            out.append(pytest.param(spec, method, id='%s-%s'%key))
    return out

# Run the test
@pytest.mark.parametrize("spec, method", _nonzero_params())
def test_writes_a_nonzero_result(spec, method, small_catalogs):
    """Every scheme a class advertises fills the multipoles."""
    shear, scalar = small_catalogs
    inst = build_correlator(spec, **SEPS, **TREE, **ANGULAR, method=method)
    run_correlator(spec, inst, shear, scalar, **_multipole_kwargs(spec))
    assert np.any(np.asarray(inst.npcf_multipoles)), method


################################
#     CONSISTENT NORMS FOR     # 
# SCALAR AND POLAR CORRELATORS #
################################

# In this test we want to make sure that the G..G and N..N count the same number of total
# objects.

# The scalar and polar class of each order to be compared with
SPEC_BY_NAME = {s.cls.__name__: s for s in CORRELATORS}
NORM_PAIRS = [
    ('NNCorrelation', 'npair', 'GGCorrelation', 'norm'),
    ('NNNCorrelation', 'npcf_multipoles', 'GGGCorrelation', 'npcf_multipoles_norm'),
    ('NNNNCorrelation_NoTomo', 'npcf_multipoles',
     'GGGGCorrelation_NoTomo', 'npcf_multipoles_norm'),
]

# Run the test
@pytest.mark.parametrize("scalar_name, scalar_field, polar_name, polar_field", NORM_PAIRS,
                         ids=['second', 'third', 'fourth'])
def test_polar_norm_reproduces_the_scalar_counts(scalar_name, scalar_field, polar_name,
                                                polar_field, small_catalogs):
    shear, scalar = small_catalogs
    counts = []
    for name, field in ((scalar_name, scalar_field), (polar_name, polar_field)):
        spec = SPEC_BY_NAME[name]
        inst = build_correlator(spec, **_exact_kwargs(spec))
        run_correlator(spec, inst, shear, scalar, **_multipole_kwargs(spec))
        counts.append(np.squeeze(np.asarray(getattr(inst, field))))
    assert _deviation(*counts) < RTOL_EXACT, _deviation(*counts)



##############
# TOMOGRAPHY #
##############

# This test asserts whether the bookkeeping for tomography is implemented correctly
# For this it processes a tomographic catalog using dotomo=True/False and makes sure
# that the sum of npcf & npcf_norm over all tomographic bins equals the result of the
# non-tomographic computation
# For this to be exactly true we need the following setup
# * Run estimators in the discrete setting (as spatial hash different in tomo vs notomo).
#   NNN has no discrete estimator, so it uses the doubletree on tree_resos=[0.], which is
#   equivalent.
# * Disable multiple counting corrs (as they break this by construction)
# * Use x-projection (as bin-centers between tomo/nontom differ in general).

# Setup correlators for this test
TOMO = correlators(orders=(2, 3))
TOMO_EXTRA = {'GGGCorrelation': dict(multicountcorr=False)}
TOMO_PROJECTION = {'GGGCorrelation': 'X'}
# NNNCorrelation.multipoles2npcf assigns npcf and npcf_norm the same triplet counts, so
# only the count identity applies to it 
COUNT_ONLY = (NNCorrelation, NNNCorrelation)

# Run the test
@pytest.mark.parametrize("spec", TOMO, ids=correlator_ids(TOMO))
def test_tomography_partitions_the_single_bin_result(spec, shear_catalog, scalar_catalog):
    # Process all the catalogs
    runs = []
    for tomo in (True, False):
        if spec.order == 2:
            kwargs = dict(**SEPS, tree_resos=[0.], nthreads=NTHREADS)
        else:
            kwargs = dict(**SEPS, **ANGULAR,
                          **_discrete_method(spec),
                          **TOMO_EXTRA.get(spec.cls.__name__, {}))
        inst = build_correlator(spec, **kwargs)
        run_correlator(spec, inst, shear_catalog, scalar_catalog, tomo=tomo)
        if spec.order > 2:
            projection = TOMO_PROJECTION.get(spec.cls.__name__)
            inst.multipoles2npcf(**({} if projection is None else dict(projection=projection)))
        runs.append(inst)
    split, single = runs

    # Define what the normalisation and npcf fields are for the different correlators
    count = 'npcf_norm' if spec.order > 2 else (
        'npair' if spec.cls is NNCorrelation else 'norm')
    fields = () if spec.cls in COUNT_ONLY else (
        ('npcf',) if spec.order > 2 else correlator_outputs(spec))

    # Do all the assertions
    nz = np.asarray(getattr(split, count))
    want_count = np.asarray(getattr(single, count)).reshape(nz.shape[1:])
    assert _deviation(nz.sum(0), want_count) < RTOL_EXACT, count
    for name in fields:
        # Second-order fields carry no leading component axis, so give them one
        ft, fs = (np.asarray(getattr(r, name)) for r in (split, single))
        if spec.order == 2:
            ft, fs = ft[None], fs[None]
        assert _deviation((ft*nz[None]).sum(1), fs[:, 0]*want_count[None]) < RTOL_EXACT, name


####################
# EDGE CORRECTION  #
####################

# This test asserts the equality of edge-correcting the npcf as Slepian & Eisenstein (2015)
# advocates or to simily divide the two correlators as is implemented in orpheus by default.
# Note the this equality is not true in general, but it holds in the exponential basis, see
# i.e. sect 7.6.3 in the notes

# Run the test
@pytest.mark.parametrize("cls", [GGGCorrelation, GNNCorrelation, NGGCorrelation])
def test_edge_correction_matrix_is_toeplitz(cls, shear_catalog, scalar_catalog):
    """Every diagonal of M is constant, which is the premise of notes eq (77)."""
    kwargs = dict(**SEPS, **ANGULAR, method='Discrete', nthreads=NTHREADS)
    corr = GGGCorrelation(n_cfs=4, **kwargs) if cls is GGGCorrelation else cls(**kwargs)
    if cls is GGGCorrelation:
        corr.process(shear_catalog, dotomo=False)
    else:
        corr.process(shear_catalog, scalar_catalog, dotomo_source=False, dotomo_lens=False)
    out = corr.edge_correction(ret_matrices=True)
    mats = np.asarray(out[-1] if isinstance(out, tuple) else out)
    assert mats.shape[-1] == mats.shape[-2] > 1, mats.shape
    for M in mats.reshape((-1,) + mats.shape[-2:]):
        for offset in range(-(M.shape[0]-1), M.shape[0]):
            diag = np.diagonal(M, offset=offset)
            if diag.size > 1:
                assert np.ptp(diag) == 0., (offset, diag)


#######################
# MULTIPOLE WINDOWING #
#######################

# The standard window reconstructs the NPCF as a sum of delta functions, so the reconstructed 
# multiplet counts oscillate and can cross zero. This should not be the case for the  Fejer kernel
# and this tests makes sure this also holds in our implementation

# Run the test
@pytest.mark.parametrize("cls", [GGGCorrelation, GNNCorrelation, NGGCorrelation])
def test_fejer_window_keeps_the_reconstructed_counts_positive(cls, shear_catalog,
                                                              scalar_catalog):
    """The counts stay non-negative under the Fejer taper, and the multipoles are intact."""
    norms, multipoles = {}, {}
    for apodization in ('rect', 'fejer'):
        kwargs = dict(**SEPS, **ANGULAR, method='Discrete', nthreads=NTHREADS,
                      apodization=apodization)
        corr = GGGCorrelation(n_cfs=4, **kwargs) if cls is GGGCorrelation else cls(**kwargs)
        if cls is GGGCorrelation:
            corr.process(shear_catalog, dotomo=False)
        else:
            corr.process(shear_catalog, scalar_catalog, dotomo_source=False,
                         dotomo_lens=False)
        corr.multipoles2npcf()
        norms[apodization] = np.real(np.asarray(corr.npcf_norm))
        multipoles[apodization] = np.asarray(corr.npcf_multipoles_norm)
    # The window is a reweighting of the transform, so the multipoles must be untouched.
    # Not bitwise as we use dynamic scheduling in the C layer
    assert np.allclose(multipoles['rect'], multipoles['fejer'], rtol=1e-11, atol=0.)
    # Empty bins reconstruct to zero either way, so allow rounding but nothing structural
    floor = 1e-12*np.max(norms['fejer'])
    assert norms['fejer'].min() > -floor, norms['fejer'].min()
