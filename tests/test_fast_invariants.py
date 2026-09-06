# Here we check statements that have to hold exactly, independent of how well an estimator
# has converged. They are therefore run on small, fast catalogs and asserted at or near
# machine precision. The file is organised as:
#
# * Physical and algebraic invariances
# * Agreement between code paths
# * Indexing and combinatorics
# * Argument handling: correlators
# * Argument handling: catalogs
# * Argument handling: direct estimators
# * Exported helpers and runtime checks

import os
import sys
from itertools import combinations_with_replacement, product

import numpy as np
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline

from orpheus.catalog import Catalog, ScalarTracerCatalog, SpinTracerCatalog
from orpheus.direct import (Direct_Map3Unequal, Direct_MapnEqual, Direct_NapnEqual,
                            MapCombinatorics)
from orpheus.flat2dgrid import FlatDataGrid_2D, FlatPixelGrid_2D
from orpheus.npcf_second import GGCorrelation, NGCorrelation, NNCorrelation
from orpheus.npcf_fourth import GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo
from orpheus.npcf_third import GGGCorrelation, GNNCorrelation, NGGCorrelation, NNNCorrelation
import orpheus
import orpheus.utils
from orpheus.multires_structs import build_npcf_output
from orpheus.npcf_base import BinnedNPCF
from orpheus.patchutils import toorigin
from orpheus.utils import (check_clib_error, convertunits, gen_n2n3indices_Gtildefourth,
                           gen_thetacombis_fourthorder, get_site_packages_dir,
                           map_ztuples, search_file_in_site_package,
                           symmetrize_map3_multiscale)

from reference import AnalyticField
from conftest import (BOXSIZE, CORRELATORS, MAX_SEP, MIN_SEP, NBINSR, NTHREADS, PI,
                      RTOL_EXACT, TREE_ONLY, build_correlator, correlator_ids,
                      correlator_outputs, correlators, run_correlator)


##################
# SHARED HELPERS #
##################

## Params
SEPS = dict(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR)
ANGULAR = dict(nmaxs=4, nbinsphi=10)
TREE = dict(tree_resos=[0., 2., 4.], rmin_pixsize=8, nthreads=NTHREADS)
DISCRETE = dict(tree_resos=[0.], rmin_pixsize=8, nthreads=NTHREADS)
# Get (p-q) per correlator component for all correlators on which we run the tests
PHASE_LEGS = {
    'NGCorrelation':          (1,),
    'GGCorrelation':          (0, 2),
    'GNNCorrelation':         (1,),
    'NGGCorrelation':         (2, 0),
    'GGGCorrelation':         (3, 1, 1, 1),
    'GNNNCorrelation_NoTomo': (1,),
    'GGGGCorrelation_NoTomo': (4, 2, 2, 2, 2, 0, 0, 0),}
PHASE = [s for s in CORRELATORS if s.cls.__name__ in PHASE_LEGS]
# Name of normalisations of the various scalar & polar classes
NORM_PAIRS = [
    ('NNCorrelation', 'npair', 'GGCorrelation', 'norm'),
    ('NNNCorrelation', 'npcf_multipoles', 'GGGCorrelation', 'npcf_multipoles_norm'),
    ('NNNNCorrelation_NoTomo', 'npcf_multipoles', 'GGGGCorrelation_NoTomo', 'npcf_multipoles_norm'),]
DIRECT = dict(order_max=3, Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS, accuracies=1.)
DIRECT_CLASSES = [Direct_MapnEqual, Direct_NapnEqual, Direct_Map3Unequal]
CHI_ROT = .7 # Rotation angle for this test

## Fixtures

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

# A small shape catalog
@pytest.fixture(scope="module")
def grid_catalog():
    rng = np.random.default_rng(45)
    ngal = 3000
    return SpinTracerCatalog(spin=2, pos1=rng.uniform(0., BOXSIZE, ngal),
                             pos2=rng.uniform(0., BOXSIZE, ngal),
                             tracer_1=rng.normal(0., .3, ngal),
                             tracer_2=rng.normal(0., .3, ngal),
                             weight=rng.uniform(.5, 1.5, ngal), geometry='flat2d')

# Sparse catalog that will leave some bins empty
@pytest.fixture(scope="module")
def sparse_catalogs():
    rng = np.random.default_rng(70)
    ngal = 400
    pos1, pos2 = rng.uniform(0., BOXSIZE, ngal), rng.uniform(0., BOXSIZE, ngal)
    shear = SpinTracerCatalog(spin=2, pos1=pos1, pos2=pos2,
                              tracer_1=rng.normal(0., .3, ngal),
                              tracer_2=rng.normal(0., .3, ngal),
                              weight=np.ones(ngal), geometry='flat2d')
    scalar = ScalarTracerCatalog(pos1=pos1, pos2=pos2, tracer=np.ones(ngal),
                                 weight=np.ones(ngal), geometry='flat2d')
    return shear, scalar

# A denser shape catalog that is needed for certain tests
@pytest.fixture(scope="module")
def dense_catalog():
    rng = np.random.default_rng(74)
    box, ngal = 100., 10000
    return SpinTracerCatalog(spin=2, pos1=rng.uniform(0., box, ngal),
                             pos2=rng.uniform(0., box, ngal),
                             tracer_1=rng.normal(0., .3, ngal),
                             tracer_2=rng.normal(0., .3, ngal),
                             weight=np.ones(ngal), geometry='flat2d')

@pytest.fixture(scope="module")
def direct_catalogs():
    # Small scalar and polar catalogs with their inferred mask
    rng = np.random.default_rng(67)
    ngal = 3000
    shared = dict(pos1=rng.uniform(0., BOXSIZE, ngal), pos2=rng.uniform(0., BOXSIZE, ngal),
                  weight=rng.uniform(.5, 1.5, ngal),
                  zbins=rng.integers(0, 2, ngal), geometry='flat2d')
    shear = SpinTracerCatalog(spin=2, tracer_1=rng.normal(0., .3, ngal),
                              tracer_2=rng.normal(0., .3, ngal), **shared)
    scalar = ScalarTracerCatalog(tracer=np.ones(ngal), **shared)
    for cat in (shear, scalar):
        cat.create_mask(method="Basic", pixsize=4.)
    return shear, scalar

# A small full-sky catalog
@pytest.fixture(scope="module")
def sky_catalog_for_hashing():
    rng = np.random.default_rng(65)
    ngal = 3000
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    return SpinTracerCatalog(spin=2, pos1=rng.uniform(10., 40., ngal), pos2=dec,
                             tracer_1=rng.normal(0., .3, ngal),
                             tracer_2=rng.normal(0., .3, ngal), weight=np.ones(ngal),
                             zbins=rng.integers(0, 2, ngal), geometry='spherical',
                             units_pos1='deg', units_pos2='deg')

 # Create scalar and polar catalogs with either doubled weights or two galaxies 
# at the same positions
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


## Methods

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

def _rotated(cat, chi):
    # Generate SpinTracerCatalog rotated by chi wrt an original one
    e = (np.asarray(cat.tracer_1) + 1j*np.asarray(cat.tracer_2))*np.exp(1j*chi)
    return SpinTracerCatalog(spin=2, pos1=cat.pos1, pos2=cat.pos2, tracer_1=e.real,
                             tracer_2=e.imag, weight=cat.weight, geometry='flat2d')

def _nonzero_params():
    out = []
    for spec in correlators(orders=(3, 4)):
        for method in build_correlator(spec, **SEPS, **TREE, **ANGULAR).methods_avail:
            key = (spec.cls.__name__, method)
            out.append(pytest.param(spec, method, id='%s-%s'%key))
    return out

# A correlator whose autosetup lands on a ladder of several cell sizes
def _laddered(**extra):
    return GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS,
                          method='DoubleTree', rmin_pixsize=8, tree_alpha=.5, **extra)

# The band layout an estimator would derive for this catalog, so the hash is exercised
# with the arguments it actually sees in production.
def _sky_bands():
    gg = GGCorrelation(**SEPS, method='DoubleTree', process_spherical=True,
                       sep_units='arcmin', **TREE)
    nsides, nside_hash = gg.tree_resos_to_nsides()
    return dict(reso_redges=gg.tree_redges*convertunits(gg.sep_units, 'deg'),
                nsides=nsides, nside_hash=nside_hash)

def _direct_kwargs(cls, **extra):
    kwargs = dict(DIRECT, **extra)
    if cls is Direct_Map3Unequal:
        kwargs.pop('order_max')
    return kwargs


######################################
# PHYSICAL AND ALGEBRAIC INVARIANCES #
######################################
# Statements about the field or the estimator that hold whatever the implementation does.

# Given a catalog with n gals of weight w at positions p construct two related ones; 
# in the first each galaxy receives twice the weight in the second put to galaxies at each p.
# We then want to make sure that we measure the same correlators in both ones; in order for this be
# truly exact we need to disable the multicountcorrs.
@pytest.mark.parametrize("spec", correlators(orders=(2, 3)), ids=correlator_ids(correlators(orders=(2, 3))))
def test_treats_a_repeated_tracer_as_extra_weight(spec, duplicated_catalogs):
    c = duplicated_catalogs
    out = []
    for tag in ('doubled', 'repeated'):
        inst = build_correlator(spec, multicountcorr=False, **_exact_kwargs(spec))
        run_correlator(spec, inst, c['shear_%s'%tag], c['scalar_%s'%tag])
        out.append([np.asarray(getattr(inst, f)) for f in correlator_outputs(spec)])
    for name, a, b in zip(correlator_outputs(spec), *out):
        assert _deviation(a, b) < RTOL_EXACT, (name, _deviation(a, b))

# Make sure each correlators respond as expected to a globally rotated shear field. See sect. 6.2 in notes
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

# Make sure that the norms are consistent between scalar and polar field for points
# at the same locations
@pytest.mark.parametrize("scalar_name, scalar_field, polar_name, polar_field", NORM_PAIRS,
                         ids=['second', 'third', 'fourth'])
def test_polar_norm_reproduces_the_scalar_counts(scalar_name, scalar_field, polar_name,
                                                polar_field, small_catalogs):
    spec_by_name = {s.cls.__name__: s for s in CORRELATORS}
    shear, scalar = small_catalogs
    counts = []
    for name, field in ((scalar_name, scalar_field), (polar_name, polar_field)):
        spec = spec_by_name[name]
        inst = build_correlator(spec, **_exact_kwargs(spec))
        run_correlator(spec, inst, shear, scalar, **_multipole_kwargs(spec))
        counts.append(np.squeeze(np.asarray(getattr(inst, field))))
    assert _deviation(*counts) < RTOL_EXACT, _deviation(*counts)

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
# * Drop the count floor: a bin can hold enough multiplets in the sum but not in each
#   single tomographic bin, so an absolute floor is not additive by construction.
@pytest.mark.parametrize("spec", correlators(orders=(2, 3)), ids=correlator_ids(correlators(orders=(2, 3))))
def test_tomography_partitions_the_single_bin_result(spec, shear_catalog, scalar_catalog):
    # Process all the catalogs
    tomo_extra = {'GGGCorrelation': dict(multicountcorr=False)}
    tomo_projection = {'GGGCorrelation': 'X'}
    runs = []
    for tomo in (True, False):
        if spec.order == 2:
            kwargs = dict(**SEPS, tree_resos=[0.], nthreads=NTHREADS, count_floor=0.)
        else:
            kwargs = dict(**SEPS, **ANGULAR, count_floor=0.,
                          **_discrete_method(spec),
                          **tomo_extra.get(spec.cls.__name__, {}))
        inst = build_correlator(spec, **kwargs)
        run_correlator(spec, inst, shear_catalog, scalar_catalog, tomo=tomo)
        if spec.order > 2:
            projection = tomo_projection.get(spec.cls.__name__)
            inst.multipoles2npcf(**({} if projection is None else dict(projection=projection)))
        runs.append(inst)
    split, single = runs
    # Define what the normalisation and npcf fields are for the different correlators
    count = 'npcf_norm' if spec.order > 2 else (
        'npair' if spec.cls is NNCorrelation else 'norm')
    fields = () if spec.cls in (NNCorrelation, NNNCorrelation) else (
        ('npcf',) if spec.order > 2 else correlator_outputs(spec))
    # Do all the assertions
    nz = np.asarray(getattr(split, count))
    want_count = np.asarray(getattr(single, count)).reshape(nz.shape[1:])
    assert _deviation(nz.sum(0), want_count) < RTOL_EXACT, count
    # npcf*norm returns the raw multipole sum only where the bin was normalised at all,
    # so bins that any of the two runs left unnormalised carry no statement here.
    shared = np.all(np.abs(nz.real) > 0., axis=0) & (np.abs(want_count.real) > 0.)
    assert shared.any(), "every bin was masked, so the partition is untested"
    for name in fields:
        # Second-order fields carry no leading component axis, so give them one
        ft, fs = (np.asarray(getattr(r, name)) for r in (split, single))
        if spec.order == 2:
            ft, fs = ft[None], fs[None]
        got = (ft*nz[None]).sum(1)[:, shared]
        want = (fs[:, 0]*want_count[None])[:, shared]
        assert _deviation(got, want) < RTOL_EXACT, name

# Make sure that the normalisation for 3dbox geometry is in the same units as for the
# other geometries, i.e. units of "counts" in bin.
@pytest.mark.parametrize("cls,kwargs,legs,attr", [
    (NGCorrelation, dict(**SEPS, nthreads=NTHREADS), 'ng', 'norm'),
    (GGGCorrelation, dict(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS), 'shear', 'npcf_norm'),
    (GNNCorrelation, dict(**SEPS, **ANGULAR, nthreads=NTHREADS), 'mixed', 'npcf_norm'),
    (NGGCorrelation, dict(**SEPS, **ANGULAR, nthreads=NTHREADS), 'mixed', 'npcf_norm'),
    ], ids=['NG', 'GGG', 'GNN', 'NGG'])
def test_slab_reports_its_normalisation_in_count_units(cls, kwargs, legs, attr,
                                                       box_shear_catalog, box_scalar_catalog,
                                                       box_random_catalog):
    inst = cls(**kwargs)
    if legs == 'shear':
        inst.process(box_shear_catalog, cat_random=box_random_catalog, Pi=PI, dotomo=False)
    elif legs == 'ng':
        inst.process(box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
                     Pi=PI, dotomo=False)
    else:
        inst.process(box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
                     Pi=PI, dotomo_source=False, dotomo_lens=False)
    if attr == 'npcf_norm':
        inst.multipoles2npcf(**({'projection': None} if cls is GGGCorrelation else {}))
    norm = getattr(inst, attr)
    assert norm is not None, "%s never set %s"%(cls.__name__, attr)
    norm = np.asarray(norm).real
    assert np.isfinite(norm).all()
    assert norm.max() > 1., "a populated slab has to hold more than one multiplet"
    # The multipole transforms spread the shell count over the phi bins, so summing it
    # back has to return the n=0 mode
    if attr == 'npcf_norm':
        izero = 0 if cls in (GGGCorrelation, GNNCorrelation) else ANGULAR['nmaxs']
        n0 = np.asarray(inst.npcf_multipoles_norm)[izero].real
        live = n0 > 0.
        assert _deviation(norm.sum(-1)[live], n0[live]) < RTOL_EXACT

# Make sure that setting dotomo=False does not tinker with the zbins of the catalog
@pytest.mark.parametrize("cls,kwargs,legs", [
    (NGCorrelation, dict(**SEPS, nthreads=NTHREADS), 'ng'),
    (GGGCorrelation, dict(n_cfs=4, **dict(**SEPS, **ANGULAR, nthreads=NTHREADS)), 'shear'),
    (GNNCorrelation, dict(**SEPS, **ANGULAR, nthreads=NTHREADS), 'mixed'),
    (NGGCorrelation, dict(**SEPS, **ANGULAR, nthreads=NTHREADS), 'mixed'),
    ], ids=['NG', 'GGG', 'GNN', 'NGG'])
def test_slab_notomography_restores_the_catalog_bins(cls, kwargs, legs, box_shear_catalog,
                                                     box_scalar_catalog, box_random_catalog):
    cats = (box_shear_catalog, box_scalar_catalog, box_random_catalog)
    before = [c.zbins.copy() for c in cats]
    inst = cls(**kwargs)
    if legs == 'shear':
        inst.process(box_shear_catalog, cat_random=box_random_catalog, Pi=PI, dotomo=False)
    elif legs == 'ng':
        inst.process(box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
                     Pi=PI, dotomo=False)
    else:
        inst.process(box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
                     Pi=PI, dotomo_source=False, dotomo_lens=False)
    for cat, was in zip(cats, before):
        assert np.array_equal(np.asarray(cat.zbins), was), \
            "%s left %s with rewritten tomographic bins"%(cls.__name__, cat.__class__.__name__)
    # NG stores a flattened (z, r) pair, the third-order ones a (n_cfs, n, z, r, r) block
    measured = inst.xi if cls is NGCorrelation else inst.npcf_multipoles
    nz = np.shape(measured)[0 if cls is NGCorrelation else 2]
    assert nz == 1, "a single tomographic bin was requested, got %i"%nz
    assert np.any(np.asarray(measured) != 0.)

# This test asserts the equality of edge-correcting the npcf as Slepian & Eisenstein (2015)
# advocates or to simily divide the two correlators as is implemented in orpheus by default.
# Note the this equality is not true in general, but it holds in the exponential basis, see
# i.e. sect 7.6.3 in the notes. Here we check the neccessary condition that every diagonal 
@pytest.mark.parametrize("cls", [GGGCorrelation, GNNCorrelation, NGGCorrelation])
def test_edge_correction_matrix_is_toeplitz(cls, shear_catalog, scalar_catalog):
    # of M is constant, which is the premise of notes eq (77)."""
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

# Make sure that the counts stay non-negative under the Fejer taper, and the multipoles are intact
@pytest.mark.parametrize("cls", [GGGCorrelation, GNNCorrelation, NGGCorrelation])
def test_fejer_window_keeps_the_reconstructed_counts_positive(cls, shear_catalog,
                                                              scalar_catalog):
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

# Make sure that the integral to get Map4analytic converges to the expression predicted
# by Wick's theorem when using a sufficiently fine binning
def test_gaussian_map4_approaches_three_map2_squared():
    fld = AnalyticField(gamma0=.05, r0=8., boxsize=300., chi=0.)
    thetamin, thetamax, ntheta = .05, 400., 800
    thetas = np.geomspace(thetamin, thetamax, ntheta)
    xip = InterpolatedUnivariateSpline(thetas, np.real(fld.xi_plus(thetas)))
    xim = InterpolatedUnivariateSpline(thetas, np.real(fld.xi_minus(thetas)))
    radii = np.array([6., 9.])
    gggg = GGGGCorrelation_NoTomo(min_sep=.2, max_sep=160., nbinsr=20, nmaxs=8,
                                  nbinsphi=24, nthreads=NTHREADS)
    map4 = np.asarray(gggg.Map4analytic(radii, xip, xim, thetamin, thetamax, ntheta)[0])
    wick = 3.*np.array([fld.map_n(2, r) for r in radii])**2
    ratio = map4[0].real/wick
    assert np.all(ratio < 1.), ("a truncated integral cannot exceed the Wick value", ratio)
    assert np.all(ratio > .85), ratio
    # Both radii are equally converged, so they must not drift apart
    assert abs(ratio[0] - ratio[1]) < .05, ratio

# Make sure that the clustering correction to GNNN is zero in case of non-clustered data
def test_clustering_correction_is_the_identity_without_clustering(shear_catalog, scalar_catalog):
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **ANGULAR, **TREE)
    gnnn.process(shear_catalog, scalar_catalog, statistics='all4pcf')
    before = np.array(gnnn.npcf, copy=True)
    thetas = np.asarray(gnnn.bin_centers_mean).ravel()
    gnnn.apply_clustering_correction(xi=(thetas, np.zeros_like(thetas)), nnn=None)
    assert np.allclose(np.asarray(gnnn.npcf), before, rtol=RTOL_EXACT)

# Make sure the Simon+ 2013 clustering correction is implemented consistenty, here
# we check that for constant clustering we get the predicted global multiplicative offset.
@pytest.mark.parametrize("omega", [0., .25])
def test_gnn_clustering_correction_is_a_rescaling(omega, shear_catalog, scalar_catalog):
    kwargs = dict(**SEPS, **ANGULAR, **TREE, method='DoubleTree')
    tomo = dict(dotomo_source=False, dotomo_lens=False)
    plain = GNNCorrelation(**kwargs)
    plain.process(shear_catalog, scalar_catalog, **tomo)
    plain.multipoles2npcf()
    corrected = GNNCorrelation(**kwargs)
    corrected.process(shear_catalog, scalar_catalog, **tomo)
    # Flat in theta and wide enough that no triangle side falls outside the interpolation
    thetas = np.geomspace(1e-3, 1e3, 8)
    corrected.multipoles2npcf(xi=(thetas, np.full((1, len(thetas)), omega)))
    assert np.allclose(np.asarray(corrected.npcf), (1.+omega)*np.asarray(plain.npcf),
                       rtol=RTOL_EXACT, atol=0.)

# Make sure that all components corresponding to a radial bin combination whose n=0 normalisation 
# multipole vanishes are set to zero in the npcf.
@pytest.mark.parametrize("cls", [GGGCorrelation, NGGCorrelation, GNNCorrelation,
                                 GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo])
def test_empty_shells_are_zeroed_in_the_npcf(cls, sparse_catalogs):
    shear, scalar = sparse_catalogs
    nmax, isfourth = 4, cls in (GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo)
    kwargs = dict(min_sep=1e-2, max_sep=MAX_SEP, nbinsr=6, nmaxs=nmax, nbinsphi=10,
                  nthreads=NTHREADS, method='Discrete', tree_resos=[0.], rmin_pixsize=8)
    corr = GGGCorrelation(n_cfs=4, **kwargs) if cls is GGGCorrelation else cls(**kwargs)
    if cls is GGGCorrelation:
        corr.process(shear, dotomo=False)
    elif cls is GGGGCorrelation_NoTomo:
        corr.process(shear, statistics='4pcf_multipole', lowmem=False)
    elif cls is GNNNCorrelation_NoTomo:
        corr.process(shear, scalar, statistics='4pcf_multipole', lowmem=False)
    else:
        corr.process(shear, scalar, dotomo_source=False, dotomo_lens=False)
    (corr.multipoles2npcf_c if cls is GGGGCorrelation_NoTomo else corr.multipoles2npcf)()

    # GGG and NGG only store n>=0, everything else keeps the full range around n=0
    norm = np.asarray(corr.npcf_multipoles_norm)
    izero = 0 if cls in (GGGCorrelation, NGGCorrelation) else nmax
    empty = norm[(izero,)*(2 if isfourth else 1)] == 0.
    npcf = np.asarray(corr.npcf)
    assert empty.any(), "the setup left no empty shells to check"
    assert np.isfinite(npcf).all()
    assert not np.any(npcf[:, empty]), np.abs(npcf[:, empty]).max()

# Make sure that the LS estimator never gets zerodivisionerrors
def test_landy_szalay_stays_finite_without_random_pairs(sparse_catalogs):
    _, scalar = sparse_catalogs
    rng = np.random.default_rng(31)
    ngal = 2*scalar.ngal
    random = ScalarTracerCatalog(pos1=rng.uniform(0., BOXSIZE, ngal),
                                 pos2=rng.uniform(0., BOXSIZE, ngal),
                                 tracer=np.ones(ngal), weight=np.ones(ngal),
                                 geometry='flat2d')
    nn = NNCorrelation(min_sep=1e-2, max_sep=MAX_SEP, nbinsr=20, nthreads=NTHREADS)
    nn.process(cat=scalar, cat_random=random, dotomo=False)
    assert np.any(np.asarray(nn.npair) == 0), "the setup left no empty bins to check"
    assert np.isfinite(np.asarray(nn.xi)).all()


################################
# AGREEMENT BETWEEN CODE PATHS #
################################
# The same quantity computed two ways has to come out the same.

# Make sure that DoubleTree with maxresoind_leaf=0 produces the same result as BaseTree
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

# TODO: GNNN FAILS SLIGNTLY WHEN ENAABLING MULTIPLE COUNTIG CORRS, TREE IS CORRECT
# Make sure lowmem and highmem give the same result of GNNN
@pytest.mark.parametrize("spec", correlators(orders=4), ids=correlator_ids(correlators(orders=4)))
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

# Every scheme a class advertises fills the multipoles
@pytest.mark.parametrize("spec, method", _nonzero_params())
def test_writes_a_nonzero_result(spec, method, small_catalogs):
    shear, scalar = small_catalogs
    inst = build_correlator(spec, **SEPS, **TREE, **ANGULAR, method=method)
    run_correlator(spec, inst, shear, scalar, **_multipole_kwargs(spec))
    assert np.any(np.asarray(inst.npcf_multipoles)), method

 # Make sure that fourth-order X -> Centroid conversion and its bookkeeping works as expected
def test_gggg_reprojects_from_x_to_centroid(shear_catalog):
    gggg = GGGGCorrelation_NoTomo(**SEPS, **ANGULAR, **TREE)
    gggg.process(shear_catalog, statistics='all4pcf')
    assert gggg.projection == 'X'
    before = np.shape(gggg.npcf)
    gggg.projectnpcf('Centroid')
    assert gggg.projection == 'Centroid'
    assert np.shape(gggg.npcf) == before
    assert np.isfinite(np.asarray(gggg.npcf)).all()

# Make sure that fourth-order X -> Centroid conversion and its bookkeeping works as expected
def test_reprojecting_from_x_matches_a_direct_centroid_transform(shear_catalog):
    kwargs = dict(n_cfs=4, **SEPS, **ANGULAR, **TREE, method='DoubleTree')
    direct = GGGCorrelation(**kwargs)
    direct.process(shear_catalog, dotomo=False)
    direct.multipoles2npcf(projection='Centroid')
    viax = GGGCorrelation(**kwargs)
    viax.process(shear_catalog, dotomo=False)
    viax.multipoles2npcf(projection='X')
    assert viax.projection == 'X'
    viax.projectnpcf('Centroid')
    assert viax.projection == 'Centroid'
    assert np.shape(viax.npcf) == np.shape(direct.npcf)
    assert np.isfinite(np.asarray(viax.npcf)).all()


##############################
# INDEXING AND COMBINATORICS #
##############################
# The index maps are bijections onto what they enumerate, in both directions.

# The docstring gives the mapping for ntomobins=3, order=2 explicitly, so it doubles as the
# reference: unsorted tuples index into the sorted ones.
@pytest.mark.parametrize("ntomobins,order", [(3, 2), (2, 3), (4, 2), (3, 3)])
def test_map_ztuples_sends_each_tuple_to_its_sorted_form(ntomobins, order):
    nsorted, nunsorted, mapper = map_ztuples(ntomobins, order)
    sorted_tuples = list(combinations_with_replacement(range(ntomobins), order))
    unsorted_tuples = list(product(range(ntomobins), repeat=order))
    assert nsorted == len(sorted_tuples)
    assert nunsorted == len(unsorted_tuples) == ntomobins**order
    assert np.shape(mapper) == (nunsorted,)
    for flat, tup in enumerate(unsorted_tuples):
        assert sorted_tuples[mapper[flat]] == tuple(sorted(tup))

def test_map_ztuples_matches_its_documented_example():
    assert list(map_ztuples(3, 2)[2]) == [0, 1, 2, 1, 3, 4, 2, 4, 5]

# Symmetrising averages a map3 over the permutations of its radial and tomographic triples.
# A map3 whose entries already depend only on the sorted triples is therefore a fixed point,
# and the averaged values must come back exactly.
@pytest.mark.parametrize("nbinsz,nbinsr", [(2, 2), (1, 3), (3, 2)])
def test_symmetrize_map3_averages_over_permutations(nbinsz, nbinsr):
    z_combs = list(combinations_with_replacement(range(nbinsz), 3))
    r_combs = list(combinations_with_replacement(range(nbinsr), 3))
    # Value assigned by sorted triple only, so the input is already symmetric
    def value(ztup, rtup):
        return 1. + z_combs.index(tuple(sorted(ztup))) + 10.*r_combs.index(tuple(sorted(rtup)))
    map3 = np.zeros((8, nbinsz**3, nbinsr**3))
    for zi, ztup in enumerate(product(range(nbinsz), repeat=3)):
        for ri, rtup in enumerate(product(range(nbinsr), repeat=3)):
            map3[:, zi, ri] = value(ztup, rtup)
    out = np.asarray(symmetrize_map3_multiscale(map3))
    assert out.shape == (8, len(z_combs), len(r_combs))
    expected = np.array([[value(z, r) for r in r_combs] for z in z_combs])
    assert np.allclose(out, expected[None, :, :], rtol=RTOL_EXACT)

# return_list=True adds the per-permutation arrays alongside the symmetrised map3
def test_symmetrize_map3_can_return_the_permutation_list():
    nbinsz, nbinsr = 2, 2
    rng = np.random.default_rng(19)
    map3 = rng.normal(size=(8, nbinsz**3, nbinsr**3))
    symm_only = symmetrize_map3_multiscale(map3)
    symm, perms = symmetrize_map3_multiscale(map3, return_list=True)
    assert np.allclose(np.asarray(symm), np.asarray(symm_only))
    assert len(perms) == len(list(combinations_with_replacement(range(nbinsz), 3)))

# Index lists for the fourth-order Gtilde multipoles: the three index arrays describe the
# same set of (n2, n3) pairs, so they have to line up and stay inside the allocated range.
@pytest.mark.parametrize("nmax", [2, 3, 4])
def test_gtilde_fourth_indices_are_consistent(nmax):
    shape, flat, n2s, n3s = gen_n2n3indices_Gtildefourth(nmax)
    assert len(flat) == len(n2s) == len(n3s)
    assert len(np.unique(flat)) == len(flat), "flattened indices repeat"
    assert np.all(np.abs(n2s) <= 2*nmax+1) and np.all(np.abs(n3s) <= 2*nmax+1)
    # flat indexes into a square of side shape[0], and every entry has to fit inside it
    assert np.all(flat >= 0) and np.all(flat < shape[0]**2)

# The custom-bin branch, and the empty-input early return
def test_theta_combinations_accept_a_custom_selection():
    nbinsr = 4
    # The second entry holds the (ntriplets, 3) radial-bin combinations
    everything = gen_thetacombis_fourthorder(nbinsr, 2, None, 1000)[1]
    # 0 -> (0,0,0) and 5 -> (0,1,1) are sorted and survive; 17 -> (1,0,1) does not
    selected = gen_thetacombis_fourthorder(nbinsr, 2, None, 1000,
                                           custom=np.array([0, 5, 17], dtype=int))[1]
    assert np.shape(everything)[1] == np.shape(selected)[1] == 3
    assert 0 < np.shape(selected)[0] < np.shape(everything)[0]
    assert all(tuple(t) == tuple(sorted(t)) for t in np.asarray(selected))
    # An index outside the cube is rejected rather than silently clipped
    with pytest.raises(AssertionError):
        gen_thetacombis_fourthorder(nbinsr, 2, None, 1000,
                                    custom=np.array([nbinsr**3], dtype=int))

# The batch size is derived from the thread count, and capped by batchsize_max
def test_theta_combinations_batch_the_radial_bins(capsys):
    nbinsr, nthreads = 4, 2
    ntriplets = len(list(combinations_with_replacement(range(nbinsr), 3)))
    # Cap well above ntriplets/nthreads, so the thread count sets the batch size
    free = gen_thetacombis_fourthorder(nbinsr, nthreads, None, 1000, verbose=True)
    assert "Using batchsize" in capsys.readouterr().out
    # Cap below it, so the cap sets it and the last batch is the short one
    capped = gen_thetacombis_fourthorder(nbinsr, nthreads, None, 5)
    for out in (free, capped):
        assert out[0] == ntriplets
        assert int(np.sum(out[4])) == ntriplets
        assert out[3][0] == 0 and out[3][-1] == ntriplets
    assert capped[5] >= free[5], "a smaller cap cannot give fewer batches"

# genzcombi maps a tomographic bin combination to its index in the Map^n/Nap^n output. It
# inherits sel2ind's precondition that the combination is sorted, so only sorted ones are
# checked here; an unsorted tuple returns a different index rather than raising.
@pytest.mark.parametrize("cls,extra", [(Direct_MapnEqual, {}), (Direct_NapnEqual, {})])
def test_direct_genzcombi_indexes_each_tomo_combination_once(cls, extra):
    order_max, nbinsz = 3, 3
    est = cls(order_max=order_max, Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS, **extra)
    combis = list(combinations_with_replacement(range(nbinsz), order_max))
    inds = [int(est.genzcombi(list(z), nbinsz=nbinsz)) for z in combis]
    assert len(set(inds)) == len(inds), "two combinations share an index"
    assert all(i >= 0 for i in inds)
    # The indices of the sorted combinations are contiguous, so nothing is skipped
    assert sorted(inds) == list(range(min(inds), min(inds)+len(inds)))

# Make sure that genzcombi works as expected. 
@pytest.mark.parametrize("cls,kind", [(Direct_MapnEqual, 'shear'), (Direct_NapnEqual, 'scalar')])
def test_direct_redshift_combination_indices(cls, kind, direct_catalogs):
    shear, scalar = direct_catalogs
    cat = scalar if kind == 'scalar' else shear
    nz = cat.nbinsz
    inst = cls(**DIRECT)
    # Without a measurement there is no bin count to index against
    with pytest.raises(ValueError, match="nbinsz"):
        inst.genzcombi([0, 0])
    inst.process(cat, dotomo=True)
    # Each order fills a contiguous block, and the blocks tile the datavector
    seen = [inst.genzcombi(list(zs)) for order in range(1, inst.order_max+1)
            for zs in combinations_with_replacement(range(nz), order)]
    assert sorted(seen) == list(range(inst._nzcombis_tot(nz, True)))
    with pytest.raises(ValueError, match="up to order"):
        inst.genzcombi([0]*(inst.order_max+1))
    with pytest.raises(ValueError, match="tomographic bins"):
        inst.genzcombi([nz])

def test_map_combinatorics_zero_order_is_a_single_selection():
    """s(m, 0) = 1 anchors the recursion that counts the sorted radius tuples."""
    combis = MapCombinatorics(nradii=4, order_max=3)
    assert combis.psumtot(0, 3) == 1.


##################################
# ARGUMENT HANDLING: CORRELATORS #
##################################

# nbinsr is derived from binsize when only the latter is given
def test_binning_can_be_given_as_a_bin_size():
    binsize = .2
    gg = GGCorrelation(min_sep=MIN_SEP, max_sep=MAX_SEP, binsize=binsize, **TREE)
    assert gg.nbinsr == int(np.ceil(np.log(MAX_SEP/MIN_SEP)/binsize))
    assert len(gg.bin_edges) == gg.nbinsr + 1

# The base class broadcasts one spin over all legs
def test_spins_can_be_given_as_a_single_number():
    scalar = BinnedNPCF(order=3, spins=0, n_cfs=1, **SEPS, nmaxs=4, nbinsphi=10,
                        nthreads=NTHREADS)
    assert np.array_equal(np.asarray(scalar.spins), np.zeros(3, dtype=np.int32))

# tree_redges lets a user hand in the radii at which the tree changes resolution. Note that
# __init__ only validates it here: the ladder is rebuilt from tree_resos further down, so
# the value does not survive construction.
@pytest.mark.parametrize("cls,kwargs", [
    (GGGCorrelation, dict(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS)),
    (Direct_MapnEqual, dict(order_max=3, Rmin=MIN_SEP, Rmax=MAX_SEP, nbinsr=NBINSR,
                            nthreads=NTHREADS)),])
def test_tree_radial_edges_are_validated(cls, kwargs):
    resos = [0., 2., 4.]
    cls(**kwargs, tree_resos=resos,
        tree_redges=np.array([MAX_SEP, MIN_SEP, 20., 40.]))
    # One edge per level, plus the outer one
    with pytest.raises(AssertionError):
        cls(**kwargs, tree_resos=resos, tree_redges=np.array([MIN_SEP, 20., MAX_SEP]))
    # And the ladder has to span exactly the separation range
    with pytest.raises(AssertionError):
        cls(**kwargs, tree_resos=resos, tree_redges=np.array([0., MIN_SEP, 20., MAX_SEP]))
    with pytest.raises(AssertionError):
        cls(**kwargs, tree_resos=resos,
            tree_redges=np.array([MIN_SEP, 20., 40., 2.*MAX_SEP]))

# Each out-of-range knob is pulled back into range and says so
@pytest.mark.parametrize("kwargs,attr", [
    (dict(resoshift_leafs=99), 'resoshift_leafs'),
    (dict(resoshift_leafs=-99), 'resoshift_leafs'),
    (dict(minresoind_leaf=-3), 'minresoind_leaf'),
    (dict(minresoind_leaf=99), 'minresoind_leaf'),
    (dict(maxresoind_leaf=-3), 'maxresoind_leaf'),
    (dict(maxresoind_leaf=99), 'maxresoind_leaf'),])
def test_out_of_range_leaf_bands_are_clamped(kwargs, attr, capsys):
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, **TREE, method='DoubleTree', **kwargs)
    out = capsys.readouterr().out
    assert 'out of bounds' in out or 'smaller than' in out, out
    value = getattr(ggg, attr)
    assert -ggg.tree_nresos < value < ggg.tree_nresos

# A leaf band that ends before it starts is reported rather than silently used
def test_leaf_bands_given_out_of_order_are_reported(capsys):
    GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS,
                   minresoind_leaf=3, maxresoind_leaf=1)
    assert "smaller than" in capsys.readouterr().out

# The tree autosetup prints the resolutions and the expected runtime cost
def test_autoset_tree_reports_its_choices(shear_catalog, capsys):
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, **TREE, method='DoubleTree',
                         verbosity=3)
    ggg.autoset_tree(shear_catalog)
    out = capsys.readouterr().out
    assert "Autosetting tree parameters" in out
    assert "Tree resolutions" in out

# Spherical catalogs go through a healpix occupancy estimate instead of a flat grid
def test_autoset_tree_handles_a_spherical_catalog():
    rng = np.random.default_rng(43)
    ngal = 3000
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    cat = SpinTracerCatalog(spin=2, pos1=rng.uniform(10., 40., ngal), pos2=dec,
                            tracer_1=rng.normal(0., .3, ngal),
                            tracer_2=rng.normal(0., .3, ngal),
                            weight=np.ones(ngal), geometry='spherical',
                            units_pos1='deg', units_pos2='deg')
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, **TREE, method='DoubleTree',
                         sep_units='arcmin')
    ggg.autoset_tree(cat, nside_grid=64)
    assert len(ggg.tree_resos) > 0 and np.all(np.asarray(ggg.tree_resos) >= 0.)

# The two leaf refinements are budgeted against a relative runtime increase, so the budget
# decides how far each of them goes and what the autosetup reports. The budget is compared
# strictly, so a negative one refuses every refinement including the free ones.
@pytest.mark.parametrize("max_increase,expect", [
    (-1., "No relative shifting"), (1e6, "Largest reso index of leafs"),])
def test_autoset_tree_reports_the_leaf_budget(max_increase, expect, dense_catalog, capsys):
    ggg = _laddered(verbosity=3)
    ggg.autoset_tree(dense_catalog, max_increase=max_increase)
    assert ggg.tree_nresos > 1, "the catalog gave a single-level tree, nothing to budget"
    assert expect in capsys.readouterr().out

# The ladder starts above tree_mincellsize and stops below tree_maxcellsize
def test_autoset_tree_respects_the_cellsize_bounds(dense_catalog):
    mincell, maxcell = 1., 4.
    ggg = _laddered(tree_mincellsize=mincell, tree_maxcellsize=maxcell)
    ggg.autoset_tree(dense_catalog)
    nonzero = np.atleast_1d(ggg.tree_resos)[np.atleast_1d(ggg.tree_resos) > 0.]
    assert len(nonzero) and nonzero.min() > mincell and nonzero.max() <= maxcell

# A leaf shift deeper than the ladder is pulled back onto it
def test_autoset_tree_clamps_a_negative_leaf_shift(dense_catalog):
    ggg = _laddered(resoshift_leafs=-99)
    ggg.autoset_tree(dense_catalog, set_resoshiftleafs=False)
    assert ggg.tree_nresos > 1
    assert ggg.resoshift_leafs == -(ggg.tree_nresos-1)

# Too few galaxies to fill the parallel regions, so the search range sets the cell
def test_discrete_cellsize_falls_back_on_a_sparse_catalog():
    rng = np.random.default_rng(69)
    ngal = 50
    cat = SpinTracerCatalog(spin=2, pos1=rng.uniform(0., BOXSIZE, ngal),
                            pos2=rng.uniform(0., BOXSIZE, ngal),
                            tracer_1=rng.normal(0., .3, ngal),
                            tracer_2=rng.normal(0., .3, ngal),
                            weight=np.ones(ngal), geometry='flat2d')
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS, method='Discrete')
    assert ggg._discrete_dpix(cat) == MAX_SEP/10.

# Bins that never saw a triplet are called out, since they drive the ringing level
def test_empty_multipole_bins_are_reported(capsys):
    rng = np.random.default_rng(70)
    ngal = 400
    shear = SpinTracerCatalog(spin=2, pos1=rng.uniform(0., BOXSIZE, ngal),
                              pos2=rng.uniform(0., BOXSIZE, ngal),
                              tracer_1=rng.normal(0., .3, ngal),
                              tracer_2=rng.normal(0., .3, ngal),
                              weight=np.ones(ngal), geometry='flat2d')
    scalar = ScalarTracerCatalog(pos1=rng.uniform(0., BOXSIZE, ngal),
                                 pos2=rng.uniform(0., BOXSIZE, ngal), tracer=np.ones(ngal),
                                 weight=np.ones(ngal), geometry='flat2d')
    gnn = GNNCorrelation(min_sep=1e-3, max_sep=MAX_SEP, nbinsr=8, **ANGULAR,
                         nthreads=NTHREADS, method='DoubleTree', tree_resos=[0.],
                         rmin_pixsize=8, verbosity=1)
    gnn.process(shear, scalar, dotomo_source=False, dotomo_lens=False)
    gnn.multipoles2npcf()
    assert np.any(np.asarray(gnn.npcf_norm) == 0.), "the setup left no empty bins to report"
    assert "carry no multiplets" in capsys.readouterr().out

def test_saveinst_rejects_a_missing_directory(shear_catalog, tmp_path):
    gg = GGCorrelation(**SEPS, **TREE)
    gg.process(shear_catalog, dotomo=False)
    with pytest.raises(ValueError, match='Path to directory does not exist'):
        gg.saveinst(str(tmp_path / 'nosuchdir') + '/', 'inst')

# The branch that explains the mistake used to raise a TypeError of its own
def test_projectnpcf_reports_an_unsupported_basis_without_crashing(shear_catalog, capsys):
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, **TREE, method='DoubleTree')
    ggg.process(shear_catalog, dotomo=False)
    ggg.multipoles2npcf(projection='X')
    ggg.projectnpcf('NoSuchBasis')
    out = capsys.readouterr().out
    assert 'not yet supported' in out
    assert ggg.projection == 'X', "a rejected projection must not change the basis"

# A projection that exists but has no route from the current basis
def test_projectnpcf_reports_an_unimplemented_conversion(shear_catalog, capsys):
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, **TREE, method='DoubleTree')
    ggg.process(shear_catalog, dotomo=False)
    ggg.multipoles2npcf(projection='Centroid')
    assert ggg.projection == 'Centroid'
    ggg.projectnpcf('X') 
    out = capsys.readouterr().out
    assert 'not yet implemented' in out
    assert ggg.projection == 'Centroid'

# Without patches and without process_spherical there is nothing to run on
@pytest.mark.parametrize("cls,legs", [(NNCorrelation, 'scalar'), (GGCorrelation, 'shear'),
                                      (NGCorrelation, 'mixed')])
def test_spherical_catalogs_must_be_decomposed_first(cls, legs):
    rng = np.random.default_rng(50)
    ngal = 500
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    shared = dict(pos1=rng.uniform(10., 40., ngal), pos2=dec, weight=np.ones(ngal),
                  geometry='spherical', units_pos1='deg', units_pos2='deg')
    shear = SpinTracerCatalog(spin=2, tracer_1=rng.normal(0., .3, ngal),
                              tracer_2=rng.normal(0., .3, ngal), **shared)
    scalar = ScalarTracerCatalog(tracer=np.ones(ngal), **shared)
    args = {'scalar': (scalar,), 'shear': (shear,), 'mixed': (shear, scalar)}[legs]
    with pytest.raises(ValueError, match="patch"):
        cls(**SEPS, **TREE).process(*args, dotomo=False)

def test_cosebis_requires_a_processed_correlator():
    gg = GGCorrelation(**SEPS, **TREE)
    with pytest.raises(RuntimeError, match="has not been populated"):
        gg.computecosebi(3)

def test_cosebis_rejects_a_nonpositive_mode_count(shear_catalog):
    gg = GGCorrelation(**SEPS, **TREE)
    gg.process(shear_catalog, dotomo=False)
    with pytest.raises(ValueError, match="Nmax must be at least 1"):
        gg.computecosebi(0)

# The log-COSEBIs are the only statistic that reaches for an undeclared dependency, and the
# message a minimal install gets is the whole point of the lazy import.
def test_cosebis_report_a_missing_mpmath(shear_catalog, monkeypatch):
    gg = GGCorrelation(**SEPS, **TREE)
    gg.process(shear_catalog, dotomo=False)
    monkeypatch.setitem(sys.modules, 'mpmath', None)
    with pytest.raises(ImportError, match="pip install mpmath"):
        gg.computecosebi(3)

def test_puremode_requires_a_processed_correlator():
    gg = GGCorrelation(**SEPS, **TREE)
    with pytest.raises(RuntimeError, match="has not been populated"):
        gg.computepuremode()


###############################
# ARGUMENT HANDLING: CATALOGS #
###############################
# The grid, mask, reduction, hashing and patch options the estimators never reach with
# their defaults.

# A field straddling RA=0 is put onto one contiguous branch of the coordinate
def test_catalog_shifts_a_footprint_that_wraps_the_meridian(capsys):
    rng = np.random.default_rng(63)
    ngal = 400
    ra = np.concatenate([rng.uniform(355., 360., ngal//2), rng.uniform(0., 5., ngal//2)])
    cat = ScalarTracerCatalog(pos1=ra, pos2=rng.uniform(-5., 5., ngal),
                              tracer=np.ones(ngal), weight=np.ones(ngal),
                              geometry='spherical', units_pos1='deg', units_pos2='deg')
    assert 'not contiguous' in capsys.readouterr().out
    assert cat.max1 - cat.min1 < 15., "the footprint is still split across the meridian"
    assert np.any(np.asarray(cat.pos1) < 0.)

# A mask given to the constructor is checked against the footprint right away
def test_catalog_takes_a_mask_and_per_bin_redshifts():
    rng = np.random.default_rng(64)
    ngal, nbinsz = 500, 2
    shared = dict(pos1=rng.uniform(0., BOXSIZE, ngal), pos2=rng.uniform(0., BOXSIZE, ngal),
                  tracer=np.ones(ngal), weight=np.ones(ngal),
                  zbins=rng.integers(0, nbinsz, ngal), geometry='flat2d')
    masked = ScalarTracerCatalog(**shared)
    masked.create_mask(method="Basic", pixsize=8.)
    cat = ScalarTracerCatalog(mask=masked.mask, zbins_mean=np.array([.3, .8]),
                              zbins_std=np.array([.1, .1]), **shared)
    assert cat.mask is masked.mask and cat.nbinsz == nbinsz
    # The moments carry one entry per tomographic bin, in both slots
    for bad in (dict(zbins_mean=np.array([.3])), dict(zbins_std=np.array([.1, .1, .1]))):
        with pytest.raises(AssertionError):
            ScalarTracerCatalog(**shared, **bad)
    # A mask that does not cover the catalog is caught on construction
    with pytest.raises(AssertionError):
        ScalarTracerCatalog(mask=FlatDataGrid_2D(np.zeros((2, 2)), BOXSIZE/2., BOXSIZE/2.,
                                                 8., 8.), **shared)

# Each assignment window spreads a galaxy over a different number of cells
@pytest.mark.parametrize("method", ["NGP", "CIC", "TSC"])
def test_weight_grid_assignment_schemes(grid_catalog, method):
    pixinds, pixweights = grid_catalog.gen_weightgrid2d(dpix=8., method=method)
    per_gal = (2*grid_catalog.assign_methods[method] + 1)**2
    assert np.shape(pixinds) == np.shape(pixweights)
    assert np.size(pixinds) == per_gal*grid_catalog.ngal
    assert np.all(np.asarray(pixweights) >= 0.)
    assert np.any(np.asarray(pixweights) > 0.)


def test_weight_grid_rejects_an_unknown_scheme(grid_catalog):
    with pytest.raises(AssertionError):
        grid_catalog.gen_weightgrid2d(dpix=8., method="NoSuchWindow")

# The density-based mask, and the buffer ring added around the footprint
def test_create_mask_density_and_buffer(grid_catalog):
    plain = SpinTracerCatalog(spin=2, pos1=grid_catalog.pos1, pos2=grid_catalog.pos2,
                              tracer_1=grid_catalog.tracer_1, tracer_2=grid_catalog.tracer_2,
                              weight=grid_catalog.weight, geometry='flat2d')
    plain.create_mask(method="Density", pixsize=8.)
    dense_shape = np.shape(plain.mask.data)
    assert set(np.unique(np.asarray(plain.mask.data))) <= {0., 1.}
    buffered = SpinTracerCatalog(spin=2, pos1=grid_catalog.pos1, pos2=grid_catalog.pos2,
                                 tracer_1=grid_catalog.tracer_1,
                                 tracer_2=grid_catalog.tracer_2,
                                 weight=grid_catalog.weight, geometry='flat2d')
    buffered.create_mask(method="Density", pixsize=8., extend=16.)
    assert np.shape(buffered.mask.data)[0] > dense_shape[0]
    # The added ring is masked out
    assert np.asarray(buffered.mask.data)[0, 0] == 1.

# weighted=False paints the bare field, where the default paints ``w*field``
def test_togrid_can_drop_the_weights(grid_catalog):
    field = grid_catalog.tracer_1
    kwargs = dict(fields=[field], dpix=16., tomo=False, method="NGP")
    weighted = grid_catalog.togrid(**kwargs)[0]
    plain = grid_catalog.togrid(**kwargs, weighted=False)[0]

    assert np.shape(weighted) == np.shape(plain)
    assert np.isclose(np.sum(weighted[:, 1]), np.sum(grid_catalog.weight*field))
    assert np.isclose(np.sum(plain[:, 1]), np.sum(field))
    # The leading plane is the occupancy the field planes are normalised by
    assert np.isclose(np.sum(weighted[:, 0]), np.sum(grid_catalog.weight))
    assert np.isclose(np.sum(plain[:, 0]), grid_catalog.ngal)

def test_togrid_onto_an_existing_grid_is_not_implemented(grid_catalog):
    with pytest.raises(NotImplementedError):
        grid_catalog.togrid(fields=[], dpix=16., asgrid=object())

# A given extent has to contain the catalog, and then it fixes the grid exactly
def test_gengridprops_honours_a_fixed_extent(grid_catalog):
    dpix = 10.
    extent = [grid_catalog.min1-dpix, grid_catalog.max1+dpix,
              grid_catalog.min2-dpix, grid_catalog.max2+dpix]
    start1, start2, n1, n2 = grid_catalog._gengridprops(dpix, dpix, 1, extent)
    assert start1 == extent[0] and start2 == extent[2]
    assert n1 == int((extent[1]-extent[0])/dpix)
    assert n2 == int((extent[3]-extent[2])/dpix)
    # An extent that cuts into the catalog is refused
    with pytest.raises(AssertionError):
        grid_catalog._gengridprops(dpix, dpix, 1, [grid_catalog.min1+dpix, *extent[1:]])
    # Square cells are the default when no second cell size is given
    assert grid_catalog._gengridprops(dpix) == grid_catalog._gengridprops(dpix, dpix)

# "w2field=False leaves the weight-squared plane out of the reduced catalog
def test_reduce_can_drop_the_squared_weight_field(grid_catalog):
    with_w2 = grid_catalog.reduce(dpix=8., w2field=True)
    without = grid_catalog.reduce(dpix=8., w2field=False)
    assert len(with_w2) == len(without)
    # Same cells either way, so the positions have to agree
    assert np.allclose(np.asarray(with_w2[1]), np.asarray(without[1]))

def test_reduce_returns_an_instance_on_request(grid_catalog):
    inst = grid_catalog.reduce(dpix=8., ret_inst=True)
    assert isinstance(inst, SpinTracerCatalog)
    assert inst.ngal <= grid_catalog.ngal

# A complex field is reduced as its two components and put back together
def test_reduce_handles_a_complex_field(grid_catalog):
    field = grid_catalog.tracer_1 + 1j*grid_catalog.tracer_2
    grid_catalog.build_spatialhash(dpix=8.)
    *_, fields_red = grid_catalog._reduce(fields=[field], dpix=8.)
    *_, parts_red = grid_catalog._reduce(fields=[grid_catalog.tracer_1,
                                                 grid_catalog.tracer_2], dpix=8.)
    assert len(fields_red) == 1 and np.iscomplexobj(fields_red[0])
    assert np.allclose(fields_red[0].real, parts_red[0])
    assert np.allclose(fields_red[0].imag, parts_red[1])

# The scalar reduction used to hand its spin to the position argument
def test_scalar_reduce_returns_an_instance_on_request():
    rng = np.random.default_rng(61)
    ngal = 2000
    cat = ScalarTracerCatalog(pos1=rng.uniform(0., BOXSIZE, ngal),
                              pos2=rng.uniform(0., BOXSIZE, ngal),
                              tracer=rng.normal(0., 1., ngal),
                              weight=rng.uniform(.5, 1.5, ngal), geometry='flat2d')
    inst = cat.reduce(dpix=8., ret_inst=True)
    w_red, pos1_red, pos2_red, _, _, fields_red = cat.reduce(dpix=8.)
    assert isinstance(inst, ScalarTracerCatalog)
    assert 0 < inst.ngal <= cat.ngal
    assert inst.pos1.min() >= cat.pos1.min() - 8. and inst.pos1.max() <= cat.pos1.max() + 8.
    assert np.isclose(inst.weight.sum(), cat.weight.sum())
    # The instance is just the raw arrays wrapped up
    assert np.allclose(inst.pos1, pos1_red) and np.allclose(inst.pos2, pos2_red)
    assert np.allclose(inst.weight, w_red) and np.allclose(inst.tracer, fields_red[0])

# Without an explicit dpix_hash the coarsest reduction sets the cell size
def test_multihash_defaults_its_hash_cellsize(grid_catalog):
    auto = grid_catalog.multihash_bundle(dpixs=[2., 8.])
    given = grid_catalog.multihash_bundle(dpixs=[2., 8.], dpix_hash=8.)
    assert np.array_equal(np.asarray(auto['ngal_resos']), np.asarray(given['ngal_resos']))

def test_multihash_refuses_a_grid_it_cannot_allocate(grid_catalog):
    with pytest.raises(ValueError, match="Too fine resolution"):
        grid_catalog.multihash_bundle(dpixs=[1e-4])

# Randoms have no field to reduce, which the subclasses override
def test_base_catalog_carries_no_tracer_field():
    rng = np.random.default_rng(66)
    ngal = 200
    base = Catalog(pos1=rng.uniform(0., BOXSIZE, ngal), pos2=rng.uniform(0., BOXSIZE, ngal),
                   weight=np.ones(ngal), geometry='flat2d')
    for geometry in ('flat2d', 'spherical', '3dbox'):
        assert base._multihash_fields(geometry, False) is None

def test_multihash_spherical_checks_its_inputs(grid_catalog, sky_catalog_for_hashing):
    bands = _sky_bands()
    with pytest.raises(ValueError, match="spherical"):
        grid_catalog.multihash_spherical(**bands)
    with pytest.raises(ValueError, match="shuffle must be"):
        sky_catalog_for_hashing.multihash_spherical(shuffle=9, **bands)

# The four conventions differ only in where inside a healpix cell the reduced tracer is
# placed, so they have to agree on everything else.
@pytest.mark.parametrize("shuffle", [0, 1, 2, 3])
def test_multihash_spherical_pixel_centre_conventions(shuffle, sky_catalog_for_hashing):
    reference = sky_catalog_for_hashing.multihash_bundle(shuffle=0, w2field=True,
                                                         **_sky_bands())
    mh = sky_catalog_for_hashing.multihash_bundle(shuffle=shuffle, w2field=True,
                                                  **_sky_bands())
    assert np.array_equal(np.asarray(mh['ngal_resos']), np.asarray(reference['ngal_resos']))
    assert np.allclose(np.asarray(mh['red_w']).sum(), np.asarray(reference['red_w']).sum())
    # Every reduced tracer stays on the unit sphere whichever centre it is given
    norm = (np.asarray(mh['red_vx'])**2 + np.asarray(mh['red_vy'])**2
            + np.asarray(mh['red_vz'])**2)
    assert np.allclose(norm, 1.)

# nav_coarsen keeps the reduction nside and only widens the navigation cells
def test_multihash_spherical_coarsens_its_navigation(sky_catalog_for_hashing, capsys):
    plain = sky_catalog_for_hashing.multihash_bundle(w2field=True, **_sky_bands())
    coarse = sky_catalog_for_hashing.multihash_bundle(w2field=True, nav_coarsen=2.,
                                                      verbose=True, **_sky_bands())
    assert 'band 0' in capsys.readouterr().out
    assert not plain['nav_coarsened'] and coarse['nav_coarsened']
    assert np.all(np.asarray(coarse['nside_nav']) <= np.asarray(plain['nside_nav']))
    assert np.any(np.asarray(coarse['nside_nav']) < np.asarray(plain['nside_nav']))
    # Only the navigation coarsens; the reduction itself is untouched
    assert np.array_equal(np.asarray(coarse['ngal_resos']), np.asarray(plain['ngal_resos']))
    assert np.allclose(np.asarray(coarse['red_w']), np.asarray(plain['red_w']))

def test_topatches_rejects_a_non_catalog():
    rng = np.random.default_rng(46)
    ngal = 500
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    cat = SpinTracerCatalog(spin=2, pos1=rng.uniform(10., 40., ngal), pos2=dec,
                            tracer_1=rng.normal(0., .3, ngal),
                            tracer_2=rng.normal(0., .3, ngal),
                            weight=np.ones(ngal), geometry='spherical',
                            units_pos1='deg', units_pos2='deg')
    with pytest.raises(ValueError, match="inherited from orpheus.Catalog"):
        cat.topatches(npatches=2, other_cats=["not a catalog"], n_workers=1)

# Every catalog sharing a decomposition has to live on the sphere
def test_joint_patch_decomposition_needs_spherical_catalogs():
    rng = np.random.default_rng(62)
    ngal = 500
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    sky = ScalarTracerCatalog(pos1=rng.uniform(10., 40., ngal), pos2=dec,
                              tracer=np.ones(ngal), weight=np.ones(ngal),
                              geometry='spherical', units_pos1='deg', units_pos2='deg')
    flat = ScalarTracerCatalog(pos1=rng.uniform(0., BOXSIZE, ngal),
                               pos2=rng.uniform(0., BOXSIZE, ngal), tracer=np.ones(ngal),
                               weight=np.ones(ngal), geometry='flat2d')
    with pytest.raises(ValueError, match="spherical"):
        sky.topatches(npatches=2, other_cats=[flat], n_workers=1)

# verbose=True walks the timing and progress branches of gen_cat_patchindices
@pytest.mark.parametrize("opts,expect", [
    (dict(method='kmeans_healpix', nside_kmeans=32), "patch centres"),
    (dict(method='healpix', healpix_nside=4), "HEALPix pixel assignment"),])
def test_patch_decomposition_reports_its_progress(opts, expect, capsys):
    rng = np.random.default_rng(41)
    ngal = 3000
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    cat = SpinTracerCatalog(spin=2, pos1=rng.uniform(10., 40., ngal), pos2=dec,
                            tracer_1=rng.normal(0., .3, ngal),
                            tracer_2=rng.normal(0., .3, ngal),
                            weight=np.ones(ngal), geometry='spherical',
                            units_pos1='deg', units_pos2='deg')
    cat.topatches(npatches=4, patchextend_deg=1., n_workers=1, verbose=True, **opts)
    out = capsys.readouterr().out
    for expected in ("sky coordinates", expect, "HEALPix hash grid",
                     "index hash", "buffer around patches"):
        assert expected in out, (expected, out[:400])
    assert cat.npatches > 1

def test_patch_decomposition_rejects_an_unknown_method():
    from orpheus.patchutils import gen_cat_patchindices
    rng = np.random.default_rng(42)
    ngal = 500
    with pytest.raises(NotImplementedError, match="Unknown method"):
        gen_cat_patchindices(ra_deg=rng.uniform(10., 40., ngal),
                             dec_deg=rng.uniform(-20., 10., ngal),
                             npatches=2, patchextend_arcmin=60., method='nosuchmethod',
                             n_workers=1)

# n_workers>1 takes the joblib streaming path; the results have to match n_workers=1
def test_patch_buffers_can_be_built_in_parallel():
    def _cat(seed):
        rng = np.random.default_rng(seed)
        ngal = 2000
        dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                               np.sin(np.radians(10.)), ngal)))
        return SpinTracerCatalog(spin=2, pos1=rng.uniform(10., 40., ngal), pos2=dec,
                                 tracer_1=rng.normal(0., .3, ngal),
                                 tracer_2=rng.normal(0., .3, ngal),
                                 weight=np.ones(ngal), geometry='spherical',
                                 units_pos1='deg', units_pos2='deg')

    serial, parallel = _cat(47), _cat(47)
    opts = dict(npatches=4, method='healpix', healpix_nside=4, patchextend_deg=1.)
    serial.topatches(n_workers=1, **opts)
    parallel.topatches(n_workers=2, **opts)
    assert serial.npatches == parallel.npatches
    for i in range(serial.npatches):
        a, b = serial.frompatchind(i), parallel.frompatchind(i)
        assert a.ngal == b.ngal
        assert int(np.sum(a.isinner)) == int(np.sum(b.isinner))

# patchextend_deg=0 leaves every patch with an empty outer region
def test_patch_decomposition_without_a_buffer():
    rng = np.random.default_rng(71)
    ngal = 2000
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    cat = ScalarTracerCatalog(pos1=rng.uniform(10., 40., ngal), pos2=dec,
                              tracer=np.ones(ngal), weight=np.ones(ngal),
                              geometry='spherical', units_pos1='deg', units_pos2='deg')
    cat.topatches(npatches=4, method='healpix', healpix_nside=4, patchextend_deg=0.,
                  n_workers=1)
    assert np.all(cat.patchinds['info']['patch_ngalsouter'] == 0)
    for i in range(cat.npatches):
        patch = cat.frompatchind(i)
        assert patch.ngal == int(np.sum(patch.isinner))

# A patch that catches no galaxy returns an empty buffer instead of failing
def test_patch_worker_handles_an_empty_patch():
    from healpy import ang2pix
    from orpheus.patchutils import _process_patch, _shm_create
    rng = np.random.default_rng(72)
    ngal, nside = 200, 16
    theta = np.arccos(rng.uniform(-.2, .2, ngal))
    phi = rng.uniform(.2, .6, ngal)
    pix = ang2pix(nside, theta, phi)
    order = np.argsort(pix, kind='stable')
    unique_pix, counts = np.unique(pix[order], return_counts=True)
    offsets = np.zeros(len(unique_pix)+1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    # Two patches: the first holds every galaxy, the second none at all
    arrays = dict(theta=theta, phi=phi, hash_unique_pix=unique_pix.astype(np.int64),
                  hash_offsets=offsets, hash_galinds=order.astype(np.int64),
                  galinds_inner_flat=order.astype(np.int64))
    shm_objects, shm_specs = {}, {}
    try:
        for key, arr in arrays.items():
            shm_objects[key], shm_specs[key] = _shm_create(arr)
        inner_offsets = np.array([0, ngal, ngal], dtype=np.int64)
        args = (shm_specs, inner_offsets, nside, np.radians(1.), 1.)

        filled = _process_patch(0, *args)
        empty = _process_patch(1, *args)
    finally:
        for shm in shm_objects.values():
            shm.close()
            shm.unlink()
    assert filled[0] == 0 and filled[1] == ngal and filled[3] > 0.
    assert empty[0] == 1 and empty[1] == 0 and len(empty[2]) == 0 and empty[3] == 0.

# Without a rotation angle or an inner selection, the whole field defines both
def test_toorigin_defaults_to_the_catalog_centre():
    rng = np.random.default_rng(73)
    ngal = 500
    ras, decs = rng.uniform(10., 40., ngal), rng.uniform(-20., 10., ngal)
    _, rot_ra, rot_dec, _ = toorigin(ras, decs)
    _, given_ra, given_dec, _ = toorigin(ras, decs, isinner=np.ones(ngal, dtype=bool))
    assert np.allclose(rot_ra, given_ra) and np.allclose(rot_dec, given_dec)
    # The field is carried onto the origin, so it straddles zero in both coordinates
    assert rot_ra.min() < 0. < rot_ra.max()
    assert rot_dec.min() < 0. < rot_dec.max()

# Galactic input, and the optional index and weight outputs
def test_cat2hpx_variants():
    from orpheus.patchutils import cat2hpx
    rng = np.random.default_rng(44)
    ngal = 2000
    lon = rng.uniform(10., 40., ngal)
    lat = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    nside = 16
    galactic = np.asarray(cat2hpx(lon, lat, nside=nside, radec=False))
    assert galactic.shape == (1, 12*nside**2)
    with_idx = cat2hpx(lon, lat, nside=nside, radec=True, return_idx=True)
    assert len(with_idx) > 1
    weighted = cat2hpx(lon, lat, nside=nside, radec=True, do_counts=True,
                       weights=np.ones(ngal), return_indices=True)
    assert len(weighted) > 1


########################################
# ARGUMENT HANDLING: DIRECT ESTIMATORS #
########################################

# Make sure that setting binsize is consistent with nbinsr
def test_direct_binning_can_be_given_as_a_bin_size():
    binsize = .3
    d = Direct_MapnEqual(order_max=3, Rmin=4., Rmax=16., binsize=binsize, nthreads=NTHREADS)
    assert d.nbinsr == int(np.ceil(np.log(16./4.)/binsize))
    assert len(d.radii) == d.nbinsr

@pytest.mark.parametrize("kwargs,expect", [
    (dict(resoshift_leafs=99), 'out of bounds'), (dict(minresoind_leaf=-3), 'out of bounds'),
    (dict(minresoind_leaf=99), 'out of bounds'), (dict(maxresoind_leaf=-3), 'out of bounds'),
    (dict(maxresoind_leaf=99), 'out of bounds'),
    (dict(minresoind_leaf=3, maxresoind_leaf=1), 'smaller than'),])
def test_direct_out_of_range_leaf_bands_are_clamped(kwargs, expect, capsys):
    d = Direct_MapnEqual(order_max=3, Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS,
                         method='Tree', **kwargs)
    assert expect in capsys.readouterr().out
    assert -d.tree_nresos < d.minresoind_leaf < d.tree_nresos
    assert -d.tree_nresos < d.maxresoind_leaf < d.tree_nresos

# Make sure aperture_centers='density' picks centres from the galaxies themselves
def test_direct_density_sampled_aperture_centres():
    rng = np.random.default_rng(48)
    ngal = 3000
    cat = SpinTracerCatalog(spin=2, pos1=rng.uniform(0., BOXSIZE, ngal),
                            pos2=rng.uniform(0., BOXSIZE, ngal),
                            tracer_1=rng.normal(0., .3, ngal),
                            tracer_2=rng.normal(0., .3, ngal),
                            weight=np.ones(ngal), geometry='flat2d')
    cat.create_mask(method="Basic", pixsize=2.)
    mapn, wmapn = Direct_MapnEqual(order_max=3, Rmin=4., Rmax=8., nbinsr=3,
                                   nthreads=NTHREADS, accuracies=1.,
                                   aperture_centers='density').process(cat, dotomo=False)
    assert np.shape(mapn) == np.shape(wmapn)
    assert np.isfinite(np.asarray(mapn)).any()

# Make sure that accuracy=-1 puts one aperture on every interior galaxy instead of on a grid
def test_direct_aperture_centres_can_come_from_the_catalog(direct_catalogs):
    shear, _ = direct_catalogs
    inst = Direct_MapnEqual(**DIRECT)
    c1, c2 = inst.get_pixelization(shear, R_ap=4., accuracy=-1.)
    inner = np.asarray(shear.isinner) >= .5
    assert len(c1) == len(c2) == int(np.count_nonzero(inner))
    assert np.array_equal(np.asarray(c1), np.asarray(shear.pos1)[inner])

# Make sure that mask check can only be carried out if mask is present
@pytest.mark.parametrize("cls", DIRECT_CLASSES, ids=[c.__name__ for c in DIRECT_CLASSES])
def test_direct_estimators_need_an_angular_mask(cls):
    rng = np.random.default_rng(68)
    ngal = 200
    cat = ScalarTracerCatalog(pos1=rng.uniform(0., BOXSIZE, ngal),
                              pos2=rng.uniform(0., BOXSIZE, ngal), tracer=np.ones(ngal),
                              weight=np.ones(ngal), geometry='flat2d')
    with pytest.raises(ValueError, match="no angular mask"):
        cls(**_direct_kwargs(cls))._checkmask(cat)

# Direct estimators admit that certain paths are not implemented
@pytest.mark.parametrize("cls,extra", [
    (Direct_MapnEqual, dict(order_max=3)),
    (Direct_NapnEqual, dict(order_max=3, field='scalar')),])
def test_direct_rejects_unimplemented_mode_combinations(cls, extra):
    rng = np.random.default_rng(49)
    ngal = 500
    shared = dict(pos1=rng.uniform(0., BOXSIZE, ngal), pos2=rng.uniform(0., BOXSIZE, ngal),
                  weight=np.ones(ngal), geometry='flat2d')
    cat = (ScalarTracerCatalog(tracer=np.ones(ngal), **shared) if cls is Direct_NapnEqual
           else SpinTracerCatalog(spin=2, tracer_1=rng.normal(0., .3, ngal),
                                  tracer_2=rng.normal(0., .3, ngal), **shared))
    cat.create_mask(method="Basic", pixsize=4.)
    # No DoubleTree is implemented
    inst = cls(Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS, accuracies=1.,
               method='DoubleTree', **extra)
    with pytest.raises(NotImplementedError):
        if cls is Direct_NapnEqual:
            inst.process(cat, dotomo=False, connected=False)
        else:
            inst.process(cat, dotomo=False, Emodeonly=False)
    # The discrete kernel is E-mode only
    if cls is Direct_MapnEqual:
        discrete = cls(Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS, accuracies=1.,
                       method='Discrete', **extra)
        with pytest.raises(NotImplementedError):
            discrete.process(cat, dotomo=False, Emodeonly=False)

# Make sure that Map3 at unequal scales admits that it does not compute BModes
@pytest.mark.parametrize("method", ['Discrete', 'Tree'])
def test_direct_map3_unequal_rejects_b_modes(method, direct_catalogs):
    shear, _ = direct_catalogs
    inst = Direct_Map3Unequal(Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS,
                              accuracies=1., method=method)
    with pytest.raises(NotImplementedError):
        inst.process_discrete(shear, dotomo=False, Emodeonly=False)

# Make sure Direct_Map3Unequal reverts to Tree approximation
def test_direct_map3_unequal_drops_the_doubletree():
    inst = Direct_Map3Unequal(Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS,
                              method='DoubleTree')
    assert inst.method == 'Tree'

# Make sure direct map3 unequal correctly enumerates tomobins
def test_direct_map3_unequal_with_tomography(direct_catalogs):
    shear, _ = direct_catalogs
    inst = Direct_Map3Unequal(Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS, accuracies=1.)
    map3, wmap3 = inst.process_discrete(shear, dotomo=True)
    nz = shear.nbinsz
    nzcombis = len(list(combinations_with_replacement(range(nz), 3)))
    nrcombis = len(list(combinations_with_replacement(range(inst.nbinsr), 3)))
    assert inst.nbinsz == nz
    assert np.shape(map3) == np.shape(wmap3) == (len(inst.frac_covs), nzcombis*nrcombis)

 # Make sure dotomo=False collapses the redshift axis of the per-centre maps to one bin
@pytest.mark.parametrize("cls,kind", [(Direct_MapnEqual, 'shear'), (Direct_NapnEqual, 'scalar')])
def test_direct_per_centre_maps_without_tomography(cls, kind, direct_catalogs):
    shear, scalar = direct_catalogs
    cat = scalar if kind == 'scalar' else shear
    inst = cls(**DIRECT)
    maps = (inst.getnap(0, cat, dotomo=False) if kind == 'scalar'
            else inst.getmap(0, cat, dotomo=False))
    counts, covs, msn, sn, statn, statn_norm = maps
    assert np.shape(counts)[0] == 1
    for arr in (msn, sn, statn, statn_norm):
        assert np.shape(arr)[:2] == (1, DIRECT['order_max'])
        assert np.isfinite(np.asarray(arr)).all()

# Make sure that setting verbosity=1 writes one line per aperture radius
@pytest.mark.parametrize("cls,kind", [(Direct_MapnEqual, 'shear'), (Direct_NapnEqual, 'scalar')])
def test_direct_reports_its_progress(cls, kind, direct_catalogs, capsys):
    shear, scalar = direct_catalogs
    cat = scalar if kind == 'scalar' else shear
    cls(**DIRECT, verbosity=1).process(cat, dotomo=False)
    assert 'aperture radii' in capsys.readouterr().out

@pytest.mark.parametrize("cls", DIRECT_CLASSES, ids=[c.__name__ for c in DIRECT_CLASSES])
def test_direct_saveinst_rejects_a_missing_directory(cls, tmp_path):
    with pytest.raises(ValueError, match='Path to directory does not exist'):
        cls(**_direct_kwargs(cls)).saveinst(str(tmp_path / 'nosuchdir') + '/', 'inst')

# Each direct estimator serialises the statistic it measures alongside its configuration
@pytest.mark.parametrize("cls", DIRECT_CLASSES, ids=[c.__name__ for c in DIRECT_CLASSES])
def test_direct_saveinst_loadinst_round_trip(cls, tmp_path):
    inst = cls(**_direct_kwargs(cls), filter_form="C02")
    inst.saveinst(str(tmp_path) + '/', 'inst')
    back = cls.loadinst(str(tmp_path) + '/', 'inst')
    for attr in ('Rmin', 'Rmax', 'nbinsr', 'field', 'filter_form', 'ap_weights'):
        assert getattr(back, attr) == getattr(inst, attr), attr
    assert np.allclose(back.radii, inst.radii)


#######################################
# EXPORTED HELPERS AND RUNTIME CHECKS #
#######################################
# Small utilities reachable from the package root, and the import-time checks.

# The two path helpers exported from the package root exis and are findable
def test_site_package_helpers_locate_paths():
    sitedir = get_site_packages_dir()
    assert isinstance(sitedir, str) and sitedir.endswith(("site-packages", "dist-packages"))
    found = search_file_in_site_package(os.path.dirname(orpheus.__file__), "orpheus_clib")
    assert found is not None and os.path.basename(found).startswith("orpheus_clib")
    assert search_file_in_site_package(os.path.dirname(orpheus.__file__), "nosuchprefix") is None

# Make sure orpheus complains if there is no compiled c extension
def test_load_clib_names_the_directory_it_searched(monkeypatch):
    monkeypatch.setattr(orpheus.utils.glob, "glob", lambda pattern: [])
    with pytest.raises(ImportError, match="No compiled orpheus extension"):
        orpheus.utils._load_clib()

# The allocation guard turns a failed malloc into a MemoryError
def test_check_clib_error_raises_memoryerror_when_the_flag_is_set():
    class _Failing:
        def orpheus_get_error(self):
            return 1
        def orpheus_clear_error(self):
            self.cleared = True
    failing = _Failing()
    with pytest.raises(MemoryError, match="could not allocate"):
        check_clib_error(failing)
    assert failing.cleared, "the error flag has to be cleared on the way out"

    class _Fine:
        def orpheus_get_error(self):
            return 0
    check_clib_error(_Fine())

def test_build_npcf_output_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown NPCFOutput kind"):
        build_npcf_output("NOSUCH", nbinsr=4, nmax=2, nbinsz=1)

 # Make sure that samplePoints draws stay inside the unmasked pixels of a data grid.
def test_sample_points_stays_inside_the_grid():
    npix, dpix = 20, 2.
    grid = FlatPixelGrid_2D(start_1=0., start_2=0., npix_1=npix, npix_2=npix,
                            dpix_1=dpix, dpix_2=dpix)
    datagrid = grid.todatagrid(np.zeros((npix, npix)))
    pos1, pos2 = (np.asarray(a) for a in
                  datagrid.samplePoints(.5, rng=np.random.RandomState(3)))
    assert pos1.size == pos2.size > 0
    for pos in (pos1, pos2):
        assert np.all(pos >= 0.) and np.all(pos <= npix*dpix)
    # A fully masked grid has nothing to sample from
    empty = grid.todatagrid(np.ones((npix, npix)))
    assert np.asarray(empty.samplePoints(.5, rng=np.random.RandomState(3))[0]).size == 0

# Omitting rng falls back to a fresh RandomState
def test_sample_points_defaults_its_random_generator():
    npix, dpix = 16, 2.
    grid = FlatPixelGrid_2D(start_1=0., start_2=0., npix_1=npix, npix_2=npix,
                            dpix_1=dpix, dpix_2=dpix)
    pos1, pos2 = grid.todatagrid(np.zeros((npix, npix))).samplePoints(.5)
    assert np.size(pos1) == np.size(pos2) > 0

def test_random_healpix_shift_handles_an_empty_selection():
    from orpheus.utils import _randomhealpixshift
    ra, dec = _randomhealpixshift(16, np.array([], dtype=int), np.random.default_rng(1))
    assert np.size(ra) == np.size(dec) == 0

# The guard returns immediately on other platforms, and survives a missing dyld
def test_openmp_runtime_check_is_a_noop_off_macos(monkeypatch):
    monkeypatch.setattr(orpheus.utils, "_openmp_checked", False)
    monkeypatch.setattr(orpheus.utils.sys, "platform", "linux")
    orpheus.utils._check_openmp_runtimes()          # returns before touching ctypes
    # On darwin the dyld lookup is attempted; here it fails and is swallowed
    monkeypatch.setattr(orpheus.utils, "_openmp_checked", False)
    monkeypatch.setattr(orpheus.utils.sys, "platform", "darwin")
    orpheus.utils._check_openmp_runtimes()

# Having three vendored libomp copies emits a warning
def test_openmp_runtime_check_warns_on_multiple_runtimes(monkeypatch):
    names = [b"/a/libomp.dylib", b"/b/libiomp5.dylib", b"/c/libSystem.dylib"]

    class _Fn:
        """Stands in for a ctypes function pointer, which carries a settable restype."""
        restype = None
        def __call__(self, i):
            return names[i]

    class _FakeDyld:
        _dyld_get_image_name = _Fn()
        def _dyld_image_count(self):
            return len(names)

    monkeypatch.setattr(orpheus.utils, "_openmp_checked", False)
    monkeypatch.setattr(orpheus.utils.sys, "platform", "darwin")
    monkeypatch.setattr(orpheus.utils.ct, "CDLL", lambda _: _FakeDyld())
    with pytest.warns(RuntimeWarning, match="OpenMP runtimes"):
        orpheus.utils._check_openmp_runtimes()

# pickle_load and pickle_save complain with bad inputs
def test_pickle_helpers_report_failures(tmp_path, capsys):
    from orpheus.patchutils import pickle_load, pickle_save
    # The target is an existing directory, so opening it for writing fails
    blocked = tmp_path / "adirectory"
    blocked.mkdir()
    pickle_save({"a": 1}, str(blocked))
    assert "error occurred while saving" in capsys.readouterr().out
    # Loading something that is not a pickle returns None rather than raising
    broken = tmp_path / "broken.pkl"
    broken.write_text("this is not a pickle")
    assert pickle_load(str(broken)) is None