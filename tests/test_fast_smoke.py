# Here we collect all tests required for fast tier which is always triggered.
#
# The only checks made are that there are no crashes and that the output is of the right
# shape and finite. In particular,we do not verify numerical correctness, as this is 
# being done within the slow tier test suite. The file is organised as:
#
# * Pipelines run end to end (flat-sky)
# * Pipelines run end to end (other geometries)
# * Patch decomposition and per-patch results
# * Third-order output modes / guard clauses
# * Fourth-order output selection / guard clauses / analytic machinery
# * Direct estimators
# * Tree setup and scheme agreement
# * Serialisation


import numpy as np
import pytest

from orpheus.catalog import ScalarTracerCatalog, SpinTracerCatalog
from orpheus.direct import (Direct_Map3Unequal, Direct_MapnEqual, Direct_NapnEqual,
                            MapCombinatorics)
from orpheus.npcf_fourth import GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo, NNNNCorrelation_NoTomo
from orpheus.npcf_second import GGCorrelation, NGCorrelation, NNCorrelation
from orpheus.npcf_third import GGGCorrelation, GNNCorrelation, NGGCorrelation, NNNCorrelation
from orpheus.patchutils import cat2hpx, pickle_load, pickle_save

from conftest import (CORRELATORS, MAX_SEP, MIN_SEP, NBINSR, NBINSZ, NTHREADS, PI,
                      RTOL_EXACT, build_correlator, correlator_ids, correlator_outputs,
                      correlators, run_correlator)


##################
# SHARED HELPERS #
##################

## Params
SEPS = dict(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR)
TREE = dict(tree_resos=[0., 2., 4.], rmin_pixsize=8, nthreads=NTHREADS)
NMAX, NBINSPHI = 4, 10
ANGULAR = dict(nmaxs=NMAX, nbinsphi=NBINSPHI)
RADII = np.array([MAX_SEP/8., MAX_SEP/6.])
NRADII = len(RADII)
NRCOMBIS = NRADII**3
NZ2, NZ3 = NBINSZ**2, NBINSZ**3

XI_MIN, XI_MAX, XI_NTHETA = .5, 80., 128

ALL_METHODS = ["Discrete", "Tree", "BaseTree", "DoubleTree"]

# Just some binning setups that allows for a fast computation
DISCRETE_TREE = dict(min_sep=1., max_sep=40., nbinsr=4, nmaxs=4, nbinsphi=10, nthreads=NTHREADS,
                     tree_resos=[0.], rmin_pixsize=8)
SPHERICAL = dict(nthreads=NTHREADS, method="DoubleTree", process_spherical=True,
                 sep_units='arcmin')
DIRECT = dict(order_max=3, Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS, accuracies=1.)


FOURTH_SHAPE = (1, NBINSR, NBINSR, NBINSR, NBINSPHI, NBINSPHI)
FOURTH_CLASSES = [
    (NNNNCorrelation_NoTomo, 'scalar', 'allNap', 'mapradii'),
    (GGGGCorrelation_NoTomo, 'shear', 'allMap', 'mapradii'),
    (GNNNCorrelation_NoTomo, 'mixed', 'allMapNap3', 'apradii'),]
FOURTH_IDS = [c.__name__ for c, _, _, _ in FOURTH_CLASSES]
FOURTH_APERTURE_IS_ZERO = {NNNNCorrelation_NoTomo}
def _fourth_cats(legs, shear, scalar):
    return {'scalar': (scalar,), 'shear': (shear,), 'mixed': (shear, scalar)}[legs]

# Get all methods available for a certain correlator
def _methods(cls, **extra):
    inst = cls(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR, **extra)
    return list(inst.methods_avail)
GGG_METHODS = _methods(GGGCorrelation, n_cfs=4)
GNN_METHODS = _methods(GNNCorrelation)
NGG_METHODS = _methods(NGGCorrelation)

# Collection of all the extra args that are saved in the serialisation of correlaotrs
SAVED_EXTRAS = {
    'NGCorrelation':          ('nbinsz_shape', 'nbinsz_pos'),
    'NNNCorrelation':         ('nbinsz', 'nzcombis', 'zeta'),
    'GNNCorrelation':         ('nbinsz_source', 'nbinsz_lens',
                               'zweighting', 'zweighting_sigma'),
    'NGGCorrelation':         ('nbinsz_source', 'nbinsz_lens'),
    'NNNNCorrelation_NoTomo': ('nbinsz', 'nzcombis', 'thetabatchsize_max'),
    'GGGGCorrelation_NoTomo': ('nbinsz', 'nzcombis', 'thetabatchsize_max'),
    'GNNNCorrelation_NoTomo': ('nbinsz_source', 'nbinsz_lens', 'nzcombis',
                               'thetabatchsize_max'),}
# Choose some values for non-default args that are not part of the serialisation
SAVED_CTOR = {
    'GNNCorrelation':         dict(zweighting=True, zweighting_sigma=.15),
    'NNNNCorrelation_NoTomo': dict(thetabatchsize_max=512),
    'GGGGCorrelation_NoTomo': dict(thetabatchsize_max=512),
    'GNNNCorrelation_NoTomo': dict(thetabatchsize_max=512),}

# process() can save each patch's measurement to disk and/or return the per-patch results
# instead of the stacked one. Both are documented options and neither was exercised.
PATCHED = [
    (NNCorrelation, dict(**SEPS, **TREE), 'scalar'),
    (GGCorrelation, dict(**SEPS, **TREE), 'shear'),
    (NGCorrelation, dict(**SEPS, **TREE), 'ng'),
    (GGGCorrelation, dict(n_cfs=4, **SEPS, **ANGULAR, **TREE, method='DoubleTree'), 'shear'),
    (GNNCorrelation, dict(**SEPS, **ANGULAR, **TREE, method='DoubleTree'), 'mixed'),
    (NGGCorrelation, dict(**SEPS, **ANGULAR, **TREE, method='DoubleTree'), 'mixed'),]
PATCHED_IDS = [c.__name__ for c, _, _ in PATCHED]

## Fixtures
@pytest.fixture(scope="module")
def gauss_xi():
    thetas = np.linspace(XI_MIN, XI_MAX, XI_NTHETA+1)
    xip = .01*np.exp(-thetas/10.)
    return thetas, xip, .5*xip

@pytest.fixture(scope="module")
def mapn_equal():
    rng = np.random.default_rng(7)
    ngal = 4000
    cat = SpinTracerCatalog(spin=2,
                            pos1=rng.uniform(0., 300., ngal),
                            pos2=rng.uniform(0., 300., ngal),
                            tracer_1=rng.normal(0., .3, ngal),
                            tracer_2=rng.normal(0., .3, ngal),
                            weight=rng.uniform(.5, 1.5, ngal), geometry='flat2d')
    cat.create_mask(method="Basic", pixsize=2.)
    return Direct_MapnEqual(filter_form="C02", **DIRECT).process(cat, dotomo=False)

@pytest.fixture(scope="module")
def gggg_multipoles(shear_catalog):
    gggg = GGGGCorrelation_NoTomo(**SEPS, **ANGULAR, **TREE)
    gggg.process(shear_catalog, statistics='4pcf_multipole')
    return gggg

@pytest.fixture(scope="module")
def gnnn_multipoles(shear_catalog, scalar_catalog):
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **ANGULAR, **TREE)
    gnnn.process(shear_catalog, scalar_catalog, statistics='4pcf_multipole')
    return gnnn

# One processed instance of each third-order correlator, in multipole space
@pytest.fixture(scope="module")
def third_order_processed(shear_catalog, scalar_catalog):
    ggg = GGGCorrelation(n_cfs=4, **_third_kwargs("DoubleTree"))
    ggg.process(shear_catalog, dotomo=False)
    gnn = GNNCorrelation(**_third_kwargs("DoubleTree"))
    gnn.process(shear_catalog, scalar_catalog, dotomo_source=False, dotomo_lens=False)
    ngg = NGGCorrelation(**_third_kwargs("DoubleTree"))
    ngg.process(shear_catalog, scalar_catalog, dotomo_source=False, dotomo_lens=False)
    return ggg, gnn, ngg

## Methods
def _finite(arr):
    a = np.asarray(arr)
    return a.size > 0 and np.isfinite(a).all()

# Calling .topatches alters the catalog, so we need to build a new one for
# each test regarding patches.
def _sky_catalog(kind, seed):
    rng = np.random.default_rng(seed)
    ngal = 4000
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20.)),
                                           np.sin(np.radians(10.)), ngal)))
    shared = dict(pos1=rng.uniform(10., 40., ngal), pos2=dec, weight=np.ones(ngal),
                  zbins=rng.integers(0, NBINSZ, ngal), geometry='spherical',
                  units_pos1='deg', units_pos2='deg')
    if kind == 'scalar':
        return ScalarTracerCatalog(tracer=np.ones(ngal), **shared)
    return SpinTracerCatalog(spin=2, tracer_1=rng.normal(0., .3, ngal),
                             tracer_2=rng.normal(0., .3, ngal), **shared)

def _third_kwargs(method):
    kwargs = dict(**SEPS, **ANGULAR, nthreads=NTHREADS, method=method)
    if method != "Discrete":
        kwargs.update(tree_resos=TREE['tree_resos'], rmin_pixsize=TREE['rmin_pixsize'])
    return kwargs

def _xi_splines(gauss_xi):
    thetas, xip, xim = gauss_xi
    return (lambda t: np.interp(t, thetas, xip, left=0., right=0.),
            lambda t: np.interp(t, thetas, xim, left=0., right=0.))

# Build 2pt and 3pt cfs as input for clustering correction of GNNN
def _clustering_inputs(nbinsr, nbinsphi):
    thetas = np.geomspace(MIN_SEP/2., 2.*MAX_SEP, 16)
    omega = .1*np.exp(-thetas/20.)
    rs = np.geomspace(MIN_SEP, MAX_SEP, nbinsr)
    phis = np.linspace(0., 2.*np.pi, nbinsphi, endpoint=False)
    zeta = .01*np.ones((len(rs), len(rs), len(phis)))
    return (thetas, omega), (rs, phis, zeta)

# Generates list of methods that are not implemented within different correlators
def _undeclared_params():
    out = []
    for spec in CORRELATORS:
        declared = build_correlator(spec, **SEPS).methods_avail
        for method in ALL_METHODS:
            if method not in declared:
                out.append(pytest.param(spec, method, id='%s-%s'%(spec.cls.__name__, method)))
    return out

# NG is the one mixed correlator whose patch loop takes a single dotomo instead of one
# flag per leg, so it needs its own entry rather than joining 'mixed'.
def _patch_call(legs, cat_shape, cat_lens):
    return {'scalar': ((cat_lens,), dict(dotomo=False)),
            'shear':  ((cat_shape,), dict(dotomo=False)),
            'ng':     ((cat_shape, cat_lens), dict(dotomo=False)),
            'mixed':  ((cat_shape, cat_lens),
                       dict(dotomo_source=False, dotomo_lens=False))}[legs]


############################
# PIPELINES RUN END TO END #
############################

## SECOND ORDER ##

 # Tomo runs all the way to Nap2
def test_nn_runs_the_full_pipeline(scalar_catalog):
    nn = NNCorrelation(**SEPS, **TREE)
    nn.process(scalar_catalog, cat_random=scalar_catalog, dotomo=True)
    assert np.shape(nn.npair) == (4*NZ2, NBINSR) # 4 is  DD, DR, RD, RR
    assert np.shape(nn.xi) == (NZ2, NBINSR)
    assert _finite(nn.npair) and _finite(nn.xi)
    assert np.all(np.asarray(nn.npair) >= 0.)
    nap2 = np.asarray(nn.computeNap2(RADII))
    assert nap2.shape == (NZ2, NRADII) and nap2.dtype == np.float64 and _finite(nap2)
    # Make sure all holds also with dotomo=False or single tomobin
    notomo = NNCorrelation(**SEPS, **TREE)
    notomo.process(scalar_catalog, cat_random=scalar_catalog, dotomo=False)
    assert np.shape(notomo.xi) == (1, NBINSR) and _finite(notomo.xi)
    onebin = ScalarTracerCatalog(pos1=scalar_catalog.pos1, pos2=scalar_catalog.pos2,
                                 tracer=scalar_catalog.tracer,
                                 weight=scalar_catalog.weight,
                                 zbins=np.zeros(scalar_catalog.ngal, dtype=int),
                                 geometry='flat2d')
    single = NNCorrelation(**SEPS, **TREE)
    single.process(onebin, cat_random=onebin, dotomo=True)
    assert np.allclose(np.asarray(notomo.xi), np.asarray(single.xi), rtol=RTOL_EXACT)

# Tomo setup computes all including integrated stats
def test_gg_runs_the_full_pipeline(shear_catalog):
    gg = GGCorrelation(**SEPS, **TREE)
    gg.process(shear_catalog, dotomo=True)
    for arr in (gg.xip, gg.xim):
        assert np.shape(arr) == (NZ2, NBINSR)
        assert np.iscomplexobj(np.asarray(arr)) and _finite(arr)
    map2 = np.asarray(gg.computeMap2(RADII))
    assert map2.shape == (4, NZ2, NRADII) and _finite(map2)
    xip_pure, xim_pure = gg.computepuremode()
    for arr in (xip_pure, xim_pure):
        assert np.shape(arr) == (3, NZ2, NBINSR) and _finite(arr)

# If mpmath present, compute cosebis
def test_gg_computes_cosebis(shear_catalog):
    pytest.importorskip("mpmath", reason="mpmath is optional, only log-COSEBIs need it")
    gg = GGCorrelation(**SEPS, **TREE)
    gg.process(shear_catalog, dotomo=True)
    nmodes = 3
    cosebi = np.asarray(gg.computecosebi(nmodes))
    assert cosebi.shape == (4, NZ2, nmodes) and _finite(cosebi)

# Tomo setup runs through including integrated stats
def test_ng_runs_the_full_pipeline(shear_catalog, scalar_catalog):
    ng = NGCorrelation(**SEPS, **TREE)
    ng.process(shear_catalog, scalar_catalog, dotomo=True)
    assert np.shape(ng.xi) == (NZ2, NBINSR)
    assert np.iscomplexobj(np.asarray(ng.xi)) and _finite(ng.xi)
    mapnap = np.asarray(ng.computeMapNap(RADII))
    assert mapnap.shape == (NZ2, NRADII) and _finite(mapnap)


## THIRD ORDER ##

# Gives expected output in real-space and multipole-space
def test_nnn_runs_the_full_pipeline(scalar_catalog):
    nnn = NNNCorrelation(**_third_kwargs("DoubleTree"))
    nnn.process(scalar_catalog, dotomo=True)
    assert np.shape(nnn.npcf_multipoles) == (1, NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(nnn.npcf_multipoles)
    nnn.multipoles2npcf()
    assert np.shape(nnn.npcf) == (1, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(nnn.npcf)

# Gives expected output in real-space, multipole-space, and for map3
@pytest.mark.parametrize("method", GGG_METHODS)
def test_ggg_runs_the_full_pipeline(method, shear_catalog):
    ggg = GGGCorrelation(n_cfs=4, **_third_kwargs(method))
    ggg.process(shear_catalog, dotomo=True)
    assert np.shape(ggg.npcf_multipoles) == (4, NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(ggg.npcf_multipoles)
    ggg.multipoles2npcf(projection='Centroid')
    assert np.shape(ggg.npcf) == (4, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert np.iscomplexobj(np.asarray(ggg.npcf)) and _finite(ggg.npcf)
    map3 = np.asarray(ggg.computeMap3(RADII, basis='MapMx'))
    assert map3.shape == (8, NZ3, NRADII) and _finite(map3)

# Gives expected output in real-space, multipole-space, and for mapnap2
@pytest.mark.parametrize("method", GNN_METHODS)
def test_gnn_runs_the_full_pipeline(method, shear_catalog, scalar_catalog):
    gnn = GNNCorrelation(**_third_kwargs(method))
    gnn.process(shear_catalog, scalar_catalog, dotomo_source=True, dotomo_lens=True)
    assert np.shape(gnn.npcf_multipoles) == (1, NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(gnn.npcf_multipoles)
    gnn.multipoles2npcf()
    assert np.shape(gnn.npcf) == (1, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(gnn.npcf)
    nnm = np.asarray(gnn.computeNNM(RADII))
    assert nnm.shape == (1, NZ3, NRADII) and _finite(nnm)

# Gives expected output in real-space, multipole-space, and for map2nap
@pytest.mark.parametrize("method", NGG_METHODS)
def test_ngg_runs_the_full_pipeline(method, shear_catalog, scalar_catalog):
    ngg = NGGCorrelation(**_third_kwargs(method))
    ngg.process(shear_catalog, scalar_catalog, dotomo_source=True, dotomo_lens=True)
    assert np.shape(ngg.npcf_multipoles) == (2, 2*NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(ngg.npcf_multipoles)
    ngg.multipoles2npcf()
    assert np.shape(ngg.npcf) == (2, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(ngg.npcf)
    nmm = np.asarray(ngg.computeNMM(RADII))
    assert nmm.shape == (4, NZ3, NRADII) and nmm.dtype == np.float64 and _finite(nmm)

## FOURTH ORDER ##

 # Gives expected output in real-space, multipole-space, and for nap4
def test_nnnn_runs_the_full_pipeline(scalar_catalog):
    nnnn = NNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    n4, = nnnn.process(scalar_catalog, mapradii=RADII)
    assert np.shape(nnnn.npcf_multipoles) == (
        2*NMAX+1, 2*NMAX+1, 1,NBINSR, NBINSR, NBINSR)
    assert _finite(nnnn.npcf_multipoles)
    assert np.shape(nnnn.npcf) == (1, NBINSR, NBINSR, NBINSR, NBINSPHI, NBINSPHI)
    assert _finite(nnnn.npcf)
    assert np.shape(n4) == (1, NRADII) and _finite(n4)

# Gives expected output in real-space, multipole-space, and for map4
def test_gggg_runs_the_full_pipeline(shear_catalog):
    gggg = GGGGCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    m4, = gggg.process(shear_catalog, mapradii=RADII)
    assert np.shape(gggg.npcf_multipoles) == (
        8, 2*NMAX+1, 2*NMAX+1, 1, NBINSR, NBINSR, NBINSR)
    assert _finite(gggg.npcf_multipoles)
    assert np.shape(gggg.npcf) == (8,) + (1, NBINSR, NBINSR, NBINSR, NBINSPHI, NBINSPHI)
    assert _finite(gggg.npcf)
    assert np.shape(m4) == (8, 1, NRADII) and _finite(m4)
    # computeMap4 returns one array per requested basis, here the E/B-separating one.
    map4, = gggg.computeMap4(RADII, basis='MapMx')
    assert np.shape(map4) == (16, NRADII) and _finite(map4)

# Gives expected output in real-space, multipole-space, and for mapnap3
def test_gnnn_runs_the_full_pipeline(shear_catalog, scalar_catalog):
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    napmap, = gnnn.process(shear_catalog, scalar_catalog, apradii=RADII)
    assert np.shape(gnnn.npcf_multipoles) == (
        1, 2*NMAX+1, 2*NMAX+1, 1, NBINSR, NBINSR, NBINSR)
    assert _finite(gnnn.npcf_multipoles)
    assert np.shape(napmap) == (1, 1, NRADII) and _finite(napmap)
    assert np.shape(gnnn.computeMapNap3(RADII)) == (1, 1, NRADII)

# Default ``lowmem=None`` resolves to a scheme rather than breaking
def test_gnnn_runs_on_default_arguments(shear_catalog, scalar_catalog):
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    gnnn.process(shear_catalog, scalar_catalog, apradii=RADII)
    assert _finite(gnnn.npcf_multipoles)


########################
# THE OTHER GEOMETRIES #
########################
# The same correlators on the curved sky, on patches, and in projected slabs.

def test_gg_runs_on_available_geometries(patched_catalogs):
    # Full spherical
    gg = GGCorrelation(**SEPS, **SPHERICAL)
    gg.process(_sky_catalog('shear', 25), dotomo=False)
    for arr in (gg.xip, gg.xim):
        assert np.shape(arr) == (1, NBINSR)
        assert _finite(arr) and np.any(np.asarray(arr) != 0.)
    # Spherical using patches
    cat_shape, _, _ = patched_catalogs
    gg = GGCorrelation(**SEPS, **TREE)
    gg.process(cat_shape, dotomo=False)
    for arr in (gg.xip, gg.xim):
        assert np.shape(arr) == (1, NBINSR)
        assert _finite(arr) and np.any(np.asarray(arr) != 0.)

def test_nn_runs_on_available_geometries(patched_catalogs):
   # Full spherical
    nn = NNCorrelation(**SEPS, **SPHERICAL)
    nn.process(_sky_catalog('scalar', 27), dotomo=False)
    assert np.shape(nn.npair) == (1, NBINSR)
    assert _finite(nn.npair) and np.any(np.asarray(nn.npair) != 0.)
    assert np.all(np.asarray(nn.npair) >= 0.)
    # Spherical using patches
    _, cat_lens, cat_rand = patched_catalogs
    nn = NNCorrelation(**SEPS, **TREE)
    nn.process(cat_lens, cat_random=cat_rand, dotomo=False)
    assert np.shape(nn.xi) == (1, NBINSR) and _finite(nn.xi)

def test_ng_runs_on_available_geometries(box_shear_catalog, box_scalar_catalog,
                                         box_random_catalog, patched_catalogs):
    # 3dbox
    ng = NGCorrelation(**SEPS, nthreads=NTHREADS)
    ng.process(box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
               Pi=PI, dotomo=True)
    assert np.shape(ng.xi) == (NZ2, NBINSR)
    assert _finite(ng.xi) and np.any(np.asarray(ng.xi) != 0.)
    # Full spherical complains
    with pytest.raises(ValueError, match="patch"):
        NGCorrelation(**SEPS, **SPHERICAL).process(
            _sky_catalog('shear', 28), _sky_catalog('scalar', 29), dotomo=False)
    # Spherical using patches
    cat_shape, cat_lens, _ = patched_catalogs
    ng = NGCorrelation(**SEPS, **TREE)
    ng.process(cat_shape, cat_lens, dotomo=False)
    assert np.shape(ng.xi) == (1, NBINSR)
    assert _finite(ng.xi) and np.any(np.asarray(ng.xi) != 0.)

def test_nnn_runs_on_available_geometries(patched_catalogs):
    # Full spherical
    nnn = NNNCorrelation(**SEPS, **ANGULAR, **SPHERICAL)
    nnn.process(_sky_catalog('scalar', 30), dotomo=False)
    assert np.shape(nnn.npcf_multipoles) == (1, NMAX+1, 1, NBINSR, NBINSR)
    assert _finite(nnn.npcf_multipoles)
    assert np.any(np.asarray(nnn.npcf_multipoles) != 0.)
     # Spherical using patches complains
    _, cat_lens, _ = patched_catalogs
    with pytest.raises(ValueError, match="process_spherical"):
        NNNCorrelation(**_third_kwargs("DoubleTree")).process(cat_lens, dotomo=False)

def test_ggg_runs_on_available_geometries(box_shear_catalog, box_random_catalog,
                                          patched_catalogs):
    # Full spherical
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, **SPHERICAL)
    ggg.process(_sky_catalog('shear', 26), dotomo=False)
    assert np.shape(ggg.npcf_multipoles) == (4, NMAX+1, 1, NBINSR, NBINSR)
    assert _finite(ggg.npcf_multipoles) and np.any(np.asarray(ggg.npcf_multipoles) != 0.)
    ggg.multipoles2npcf(projection='Centroid')
    assert np.shape(ggg.npcf) == (4, 1, NBINSR, NBINSR, NBINSPHI)
    assert _finite(ggg.npcf) and np.any(np.asarray(ggg.npcf) != 0.)
    # Spherical using patches
    cat_shape, _, _ = patched_catalogs
    ggg = GGGCorrelation(n_cfs=4, **_third_kwargs("DoubleTree"))
    ggg.process(cat_shape, dotomo=False)
    assert np.shape(ggg.npcf_multipoles) == (4, NMAX+1, 1, NBINSR, NBINSR)
    assert _finite(ggg.npcf_multipoles) and np.any(np.asarray(ggg.npcf_multipoles) != 0.)
    # 3dbox
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS)
    ggg.process(box_shear_catalog, cat_random=box_random_catalog, Pi=PI, dotomo=True)
    assert np.shape(ggg.npcf_multipoles) == (4, NMAX+1, NZ3, NBINSR, NBINSR)
    assert np.shape(ggg.npcf_multipoles_norm) == (NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(ggg.npcf_multipoles) and np.any(np.asarray(ggg.npcf_multipoles) != 0.)
    ggg.multipoles2npcf(projection='Centroid')
    assert np.shape(ggg.npcf) == (4, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(ggg.npcf) and np.any(np.asarray(ggg.npcf) != 0.)

def test_gnn_runs_on_available_geometries(box_shear_catalog, box_scalar_catalog,
                                          box_random_catalog, patched_catalogs):
    # 3dbox
    gnn = GNNCorrelation(**SEPS, **ANGULAR, nthreads=NTHREADS)
    gnn.process(box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
                Pi=PI, dotomo_source=True, dotomo_lens=True)
    assert np.shape(gnn.npcf_multipoles) == (1, NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(gnn.npcf_multipoles) and np.any(np.asarray(gnn.npcf_multipoles) != 0.)
    gnn.multipoles2npcf()
    assert np.shape(gnn.npcf) == (1, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(gnn.npcf) and np.any(np.asarray(gnn.npcf) != 0.)
    # Spherical using patches
    cat_shape, cat_lens, _ = patched_catalogs
    gnn = GNNCorrelation(**_third_kwargs("DoubleTree"))
    gnn.process(cat_shape, cat_lens, dotomo_source=False, dotomo_lens=False)
    assert np.shape(gnn.npcf_multipoles) == (1, NMAX+1, 1, NBINSR, NBINSR)
    assert _finite(gnn.npcf_multipoles)

def test_ngg_runs_on_available_geometries(box_shear_catalog, box_scalar_catalog,
                                          box_random_catalog, patched_catalogs):
    # 3dbox
    ngg = NGGCorrelation(**SEPS, **ANGULAR, nthreads=NTHREADS)
    ngg.process(box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
                Pi=PI, dotomo_source=True, dotomo_lens=True)
    assert np.shape(ngg.npcf_multipoles) == (2, 2*NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(ngg.npcf_multipoles) and np.any(np.asarray(ngg.npcf_multipoles) != 0.)
    ngg.multipoles2npcf()
    assert np.shape(ngg.npcf) == (2, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(ngg.npcf) and np.any(np.asarray(ngg.npcf) != 0.)
    # Spherical using patches
    cat_shape, cat_lens, _ = patched_catalogs
    ngg = NGGCorrelation(**_third_kwargs("DoubleTree"))
    ngg.process(cat_shape, cat_lens, dotomo_source=False, dotomo_lens=False)
    assert np.shape(ngg.npcf_multipoles) == (2, 2*NMAX+1, 1, NBINSR, NBINSR)
    assert _finite(ngg.npcf_multipoles)

# Full spherical (multipoles only)
def test_nnnn_runs_on_available_geometries():
    nnnn = NNNNCorrelation_NoTomo(**SEPS, **ANGULAR, nthreads=NTHREADS, method="Tree",
                                  process_spherical=True, sep_units='arcmin')
    nnnn.process(_sky_catalog('scalar', 31), statistics='4pcf_multipole')
    assert np.shape(nnnn.npcf_multipoles) == (2*NMAX+1, 2*NMAX+1, 1, NBINSR, NBINSR, NBINSR)
    assert _finite(nnnn.npcf_multipoles)
    assert np.any(np.asarray(nnnn.npcf_multipoles) != 0.)
    # Real-space complains
    for stat in ('4pcf_real', 'all4pcf'):
        with pytest.raises(NotImplementedError, match="multipole"):
            NNNNCorrelation_NoTomo(**SEPS, **ANGULAR, nthreads=NTHREADS, method="Tree",
                                   process_spherical=True, sep_units='arcmin').process(
                                       _sky_catalog('scalar', 32), statistics=stat)

# Make sure that omitting cat_lens in '3dbox' correlates the sources with their own positions.
@pytest.mark.parametrize("cls,ncf", [(GNNCorrelation, 1), (NGGCorrelation, 2)])
def test_mixed_slab_builds_its_own_lenses_from_the_sources(cls, ncf, box_shear_catalog,
                                                           box_random_catalog):
    inst = cls(**SEPS, **ANGULAR, nthreads=NTHREADS)
    inst.process(box_shear_catalog, cat_random=box_random_catalog, Pi=PI,
                 dotomo_source=False, dotomo_lens=False)
    nmodes = NMAX+1 if cls is GNNCorrelation else 2*NMAX+1
    assert np.shape(inst.npcf_multipoles) == (ncf, nmodes, 1, NBINSR, NBINSR)
    assert _finite(inst.npcf_multipoles) and np.any(np.asarray(inst.npcf_multipoles) != 0.)

# Make sure code admits that periodic metric not yet implemented
def test_ng_slab_has_no_periodic_boundaries(box_shear_catalog, box_scalar_catalog,
                                            box_random_catalog):
    with pytest.raises(NotImplementedError, match="[Pp]eriodic"):
        NGCorrelation(**SEPS, nthreads=NTHREADS).process(
            box_shear_catalog, box_scalar_catalog, cat_random=box_random_catalog,
            Pi=PI, dotomo=False, periodic=True)

# Make sure that a 3dbox catalog cannot be processed without the randoms and projection length.
def test_3dbox_requires_randoms_and_projection_length(box_shear_catalog, box_random_catalog):
    ggg = GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS)
    with pytest.raises(AssertionError):
        ggg.process(box_shear_catalog, Pi=PI, dotomo=False)
    with pytest.raises(AssertionError):
        ggg.process(box_shear_catalog, cat_random=box_random_catalog, dotomo=False)


#############################################
# PATCH DECOMPOSITION AND PER-PATCH RESULTS #
#############################################

# In the next three tests we assert that the patch decomposition works as expected, i.e. that
# the number of inner galaxies summed over the patches equals the number of galaxies

# Run the test for single catalog
def test_patch_decomposition_unique_inner(spherical_catalog):
    # Build patches using method "healpix" as it is the fastest
    cat = spherical_catalog
    cat.topatches(npatches=8, method='healpix', healpix_nside=4, 
                  patchextend_deg=1.,  n_workers=1)
    assert cat.npatches > 1, "footprint gave a single patch, nothing to test"
    assert len(cat.patchinds['info']['patchcenters']) == cat.npatches
    # Count inner galaxies of each patch
    inner = 0
    for i in range(cat.npatches):
        patch = cat.frompatchind(i)
        assert patch.geometry == 'flat2d'
        assert patch.ngal >= int(np.sum(patch.isinner))
        inner += int(np.sum(patch.isinner))
    assert inner == cat.ngal
    assert int(np.sum(cat.patchinds['info']['patch_ngalsinner'])) == cat.ngal

# Run the test for multiple catalogs
def test_patch_decomposition_shares_patches_across_catalogs():
    """other_cats are decomposed onto the patches of the catalog that defines them."""
    cat_shape, cat_lens = _sky_catalog('shear', 21), _sky_catalog('scalar', 22)
    cat_shape.topatches(npatches=8, method='healpix', healpix_nside=4,
                        patchextend_deg=1., n_workers=1, other_cats=[cat_lens])
    assert cat_shape.npatches > 1
    assert cat_lens.npatches == cat_shape.npatches
    centers = cat_shape.patchinds['info']['patchcenters']
    assert np.allclose(centers, cat_lens.patchinds['info']['patchcenters'])
    # Each catalog's inner regions still tile it exactly, as in the single-catalog case
    for cat in (cat_shape, cat_lens):
        inner = sum(int(np.sum(cat.frompatchind(i).isinner)) for i in range(cat.npatches))
        assert inner == cat.ngal

# Run the test for single catalog using kmeans_healpix patchees decomposition
def test_patch_decomposition_by_kmeans():
    cat = _sky_catalog('shear', 23)
    cat.topatches(npatches=4, method='kmeans_healpix', nside_kmeans=64,
                  patchextend_deg=1., n_workers=1)
    assert cat.npatches == 4
    inner = sum(int(np.sum(cat.frompatchind(i).isinner)) for i in range(cat.npatches))
    assert inner == cat.ngal

# Make sure cat2hpx agrees independent of the choice of do_counts
def test_cat2hpx_maps_the_footprint():
    cat = _sky_catalog('scalar', 24)
    nside = 32
    occupied = np.asarray(cat2hpx(cat.pos1, cat.pos2, nside=nside, radec=True))
    counts = np.asarray(cat2hpx(cat.pos1, cat.pos2, nside=nside, radec=True, do_counts=True))
    assert occupied.shape == counts.shape == (1, 12*nside**2)
    assert set(np.unique(occupied)) <= {0, 1}
    assert counts.sum() == cat.ngal
    assert np.array_equal(occupied.astype(bool), counts > 0)

@pytest.mark.parametrize("cls,kwargs,legs", PATCHED, ids=PATCHED_IDS)
def test_patch_results_can_be_kept_and_saved(cls, kwargs, legs, patched_catalogs, tmp_path):
    cat_shape, cat_lens, _ = patched_catalogs
    inst = cls(**kwargs, verbosity=1)
    args, tomo = _patch_call(legs, cat_shape, cat_lens)

    per_patch = inst.process(*args, save_patchres=str(tmp_path)+'/', save_filebase='p',
                             keep_patchres=True, **tomo)

    assert per_patch is not None, "keep_patchres has to return the per-patch arrays"
    for arr in per_patch:
        assert np.shape(arr)[0] == cat_shape.npatches
    saved = list(tmp_path.glob('p_patch*.npz'))
    assert len(saved) == cat_shape.npatches, (len(saved), cat_shape.npatches)

@pytest.mark.parametrize("cls,kwargs,legs", PATCHED, ids=PATCHED_IDS)
def test_patch_saving_rejects_a_missing_directory(cls, kwargs, legs, patched_catalogs, tmp_path):
    cat_shape, cat_lens, _ = patched_catalogs
    args, tomo = _patch_call(legs, cat_shape, cat_lens)
    with pytest.raises(ValueError, match='Path to directory does not exist'):
        cls(**kwargs).process(*args, save_patchres=str(tmp_path / 'nosuchdir') + '/', **tomo)


############################
# THIRD-ORDER OUTPUT MODES #
############################

# do_multiscale enumerates all radii combis
@pytest.mark.parametrize("which,basis,shape", [
    ('ggg', 'MapMx', (8, 1, NRCOMBIS)), ('ggg', 'MM*', (4, 1, NRCOMBIS)),
    ('ngg', 'MapMx', (4, 1, NRCOMBIS)), ('ngg', 'MM*', (2, 1, NRCOMBIS)),])
def test_third_order_apertures_over_radius_triplets(which, basis, shape, third_order_processed):
    ggg, _, ngg = third_order_processed
    inst = ggg if which == 'ggg' else ngg
    call = inst.computeMap3 if which == 'ggg' else inst.computeNMM
    stats = np.asarray(call(RADII, do_multiscale=True, basis=basis, tofile=True))
    assert stats.shape == shape and _finite(stats)

 # do_multiscale enumerates all radii combis for MNN
def test_gnn_apertures_over_radius_triplets(third_order_processed):
    _, gnn, _ = third_order_processed
    nnm = np.asarray(gnn.computeNNM(RADII, do_multiscale=True))
    assert nnm.shape == (1, 1, NRCOMBIS) and _finite(nnm)

# Map3 gets centroid projection by itself when it needs
def test_ggg_map3_projects_to_the_centroid_it_needs(shear_catalog):
    # computeMap3 works from multipoles and autoatically goes to centroid
    ggg = GGGCorrelation(n_cfs=4, **_third_kwargs("DoubleTree"))
    ggg.process(shear_catalog, dotomo=False)
    assert ggg.npcf is None
    map3 = np.asarray(ggg.computeMap3(RADII))
    assert ggg.projection == 'Centroid' and map3.shape == (8, 1, NRADII) and _finite(map3)
    # computeMap3 works from x-realspace and autoatically goes to centroid
    other = GGGCorrelation(n_cfs=4, **_third_kwargs("DoubleTree"))
    other.process(shear_catalog, dotomo=False)
    other.multipoles2npcf(projection='X')
    assert np.asarray(other.computeMap3(RADII)).shape == (8, 1, NRADII)
    assert other.projection == 'Centroid'

# integrated=True replaces the point-sampled phi window by its bin integral
def test_ngg_integrated_multipole_window(third_order_processed):
    _, _, ngg = third_order_processed
    ngg.multipoles2npcf(integrated=True)
    assert np.shape(ngg.npcf) == (2, 1, NBINSR, NBINSR, NBINSPHI) and _finite(ngg.npcf)

# Make sure that edge_corrections execute properly for third-order stats
@pytest.mark.parametrize("cls,legs,kwargs", [
    (GGGCorrelation, 'shear', dict(n_cfs=4)), (GNNCorrelation, 'mixed', {}),
    (NGGCorrelation, 'mixed', {}),])
def test_third_order_edge_correction_runs(cls, legs, kwargs, shear_catalog, scalar_catalog):
    inst = cls(**kwargs, **_third_kwargs("DoubleTree"))
    if legs == 'shear':
        inst.process(shear_catalog, dotomo=False, apply_edge_correction=True)
    else:
        inst.process(shear_catalog, scalar_catalog, dotomo_source=False, dotomo_lens=False,
                     apply_edge_correction=True)
    assert inst.is_edge_corrected
    assert _finite(inst.npcf_multipoles)

# Make sure that we can only access the imlemented projections for the various 3pt correlators
def test_third_order_projections_report_what_they_support(third_order_processed, capsys):
    _, gnn, ngg = third_order_processed
    for inst in (gnn, ngg):
        inst.multipoles2npcf()
        before = np.array(inst.npcf, copy=True)
        inst.projectnpcf("X")
        assert inst.projection == "X"
        assert np.array_equal(np.asarray(inst.npcf), before)
        inst.projectnpcf("Centroid")
        assert "not yet supported" in capsys.readouterr().out
        assert inst.projection == "X", "a refused projection must not relabel the npcf"

def test_nnn_projection_is_reported_as_unsupported(scalar_catalog, capsys):
    nnn = NNNCorrelation(**_third_kwargs("DoubleTree"))
    nnn.process(scalar_catalog, dotomo=False)
    nnn.multipoles2npcf()
    nnn.projectnpcf("Centroid")
    assert "not yet supported" in capsys.readouterr().out

# zeta from a joint data+random catalog with  dotomo=False
def test_nnn_landy_szalay_without_tomography(scalar_catalog):
    nnn = NNNCorrelation(**_third_kwargs("DoubleTree"))
    nnn.process(scalar_catalog, cat_random=scalar_catalog, dotomo=False)
    assert nnn.nbinsz == 1 and nnn.nzcombis == 1
    assert np.shape(nnn.zeta) == (1, NBINSR, NBINSR, NBINSPHI) and _finite(nnn.zeta)
    # The data and the randoms are the same catalog here, so (D-R)^3 cancels exactly
    assert np.allclose(np.asarray(nnn.zeta), 0., atol=1e-10)


#############################
# THIRD-ORDER GUARD CLAUSES #
#############################
# The arguments and catalog combinations each third-order correlator refuses.

# methods_avail blocks this at construction; the guard covers a later reassignment
def test_nnn_rejects_a_method_it_cannot_run(scalar_catalog):
    nnn = NNNCorrelation(**_third_kwargs("DoubleTree"))
    nnn.method = "Tree"
    with pytest.raises(NotImplementedError, match="DoubleTree"):
        nnn.process(scalar_catalog, dotomo=False)

# The joint catalog inherits the sky geometry, which only the native kernel handles
def test_nnn_landy_szalay_needs_the_curved_sky_kernel():
    cat = _sky_catalog('scalar', 33)
    with pytest.raises(ValueError, match="process_spherical"):
        NNNCorrelation(**_third_kwargs("DoubleTree")).process(cat, cat_random=cat, dotomo=False)

# process_spherical only has a doubletree kernel behind it
def test_ggg_curved_sky_needs_the_doubletree():
    with pytest.raises(ValueError, match="DoubleTree"):
        GGGCorrelation(n_cfs=4, **SEPS, **ANGULAR, nthreads=NTHREADS, method="Discrete",
                       process_spherical=True, sep_units='arcmin').process(
                           _sky_catalog('shear', 34), dotomo=False)

def test_ggg_spherical_catalog_must_be_decomposed_first():
    with pytest.raises(ValueError, match="patch"):
        GGGCorrelation(n_cfs=4, **_third_kwargs("DoubleTree")).process(
            _sky_catalog('shear', 35), dotomo=False)

# Mixed third-order correlators take two catalogs, so they check both of them and
# additionally that the two agree on a geometry.
@pytest.mark.parametrize("cls", [GNNCorrelation, NGGCorrelation])
@pytest.mark.parametrize("bad", ['source', 'lens', 'mixed'])
def test_third_order_mixed_catalog_pairs_are_checked(cls, bad, shear_catalog):
    sky_shape, sky_lens = _sky_catalog('shear', 36), _sky_catalog('scalar', 37)
    cats = {'source': (sky_shape, sky_lens), 'lens': (shear_catalog, sky_lens),
            'mixed': (shear_catalog, sky_lens)}[bad]
    if bad == 'mixed':
        sky_lens.topatches(npatches=4, method='healpix', healpix_nside=4,
                           patchextend_deg=1., n_workers=1)
    match = 'Incompatible geometries' if bad == 'mixed' else 'decomposed into patches'
    with pytest.raises(ValueError, match=match):
        cls(**_third_kwargs("DoubleTree")).process(*cats, dotomo_source=False,
                                                   dotomo_lens=False)

# zweighting needs the lens bins, so a request to drop them is overridden and said so
def test_gnn_redshift_weighting_forces_tomographic_lenses(shear_catalog, scalar_catalog, capsys):
    gnn = GNNCorrelation(zweighting=True, zweighting_sigma=.15, **_third_kwargs("DoubleTree"))
    gnn.process(shear_catalog, scalar_catalog, dotomo_source=False, dotomo_lens=False)
    assert "Redshift-weighting requires tomographic" in capsys.readouterr().out
    assert gnn.nbinsz_lens == scalar_catalog.nbinsz


#################################
# FOURTH-ORDER OUTPUT SELECTION #
#################################

# Neither the type of the argument nor the entries in it are taken on trust
@pytest.mark.parametrize("cls,legs,composite,radii_kw", FOURTH_CLASSES, ids=FOURTH_IDS)
def test_fourth_order_statistics_are_validated(cls, legs, composite, radii_kw,
                                               shear_catalog, scalar_catalog):
    inst = cls(**SEPS, **TREE, **ANGULAR)
    cats = _fourth_cats(legs, shear_catalog, scalar_catalog)
    with pytest.raises(ValueError, match="list or a string"):
        inst.process(*cats, statistics=3)
    # A list is walked entry by entry, unlike the single-string form
    with pytest.raises(ValueError, match="has not been implemented"):
        inst.process(*cats, statistics=['4pcf_multipole', 'nosuchstat'])

# The aperture-only keyword skips the multipole allocation entirely
@pytest.mark.parametrize("cls,legs,composite,radii_kw", FOURTH_CLASSES, ids=FOURTH_IDS)
def test_fourth_order_aperture_only_statistics(cls, legs, composite, radii_kw,
                                               shear_catalog, scalar_catalog):
    inst = cls(**SEPS, **TREE, **ANGULAR)
    cats = _fourth_cats(legs, shear_catalog, scalar_catalog)
    out = inst.process(*cats, statistics=composite, **{radii_kw: RADII})
    assert inst.npcf_multipoles is None, "no 4pcf was requested"
    # Only the raw correlator is returned so far; the filtered variants are still TODO
    assert len(out) == 1
    assert np.shape(out[0])[-1] == NRADII and _finite(out[0])
    if cls not in FOURTH_APERTURE_IS_ZERO:
        assert np.any(np.asarray(out[0]) != 0.), "the aperture statistic carries no signal"

@pytest.mark.parametrize("cls,legs,composite,radii_kw", FOURTH_CLASSES, ids=FOURTH_IDS)
def test_fourth_order_apertures_need_their_radii(cls, legs, composite, radii_kw,
                                                 shear_catalog, scalar_catalog):
    inst = cls(**SEPS, **TREE, **ANGULAR)
    cats = _fourth_cats(legs, shear_catalog, scalar_catalog)
    with pytest.raises(ValueError, match="[Aa]perture radii"):
        inst.process(*cats, statistics=composite)

# Make sure NNNN/GGGG complain if the 4pcf poutput is too big, i.e. above cutlen
@pytest.mark.parametrize("cls,legs,ncomp", [(NNNNCorrelation_NoTomo, 'scalar', 1),
                                            (GGGGCorrelation_NoTomo, 'shear', 8)],
                         ids=['NNNNCorrelation_NoTomo', 'GGGGCorrelation_NoTomo'])
@pytest.mark.parametrize("stat,basis", [('4pcf_multipole', 'multipole basis'),
                                        ('all4pcf', 'real basis')])
def test_fourth_order_refuses_an_oversized_output(cls, legs, ncomp, stat, basis,
                                                  shear_catalog, scalar_catalog):
    n_multipole = ncomp*(2*NMAX+1)**2*NBINSR**3
    n_real = ncomp*NBINSPHI**2*NBINSR**3
    assert n_multipole < n_real, "the budget below relies on the ordering of the two sizes"
    cutlen = 1 if stat == '4pcf_multipole' else (n_multipole + n_real)//2
    inst = cls(**SEPS, **TREE, **ANGULAR)
    cats = _fourth_cats(legs, shear_catalog, scalar_catalog)
    with pytest.raises(ValueError, match=basis):
        inst.process(*cats, statistics=stat, cutlen=cutlen)

 # Make sure real-space npcf is evauluated also in high-mem if requested in .process
@pytest.mark.parametrize("cls,legs,composite,radii_kw", FOURTH_CLASSES, ids=FOURTH_IDS)
def test_fourth_order_reports_the_real_space_transform(cls, legs, composite, radii_kw,
                                                       shear_catalog, scalar_catalog,
                                                       capsys):
    inst = cls(**SEPS, **TREE, **ANGULAR, verbosity=1)
    cats = _fourth_cats(legs, shear_catalog, scalar_catalog)
    inst.process(*cats, statistics='all4pcf', lowmem=False)
    assert "Transforming output to real space basis" in capsys.readouterr().out
    assert _finite(inst.npcf) and _finite(inst.npcf_multipoles)

# Integrated statistics without the low-memory kernels get a warning, not a refusal
def test_fourth_order_lowmem_recommendation(scalar_catalog, capsys):
    nnnn = NNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    nnnn.process(scalar_catalog, statistics='Nap4', mapradii=RADII, lowmem=False)
    assert "Lowmem computation recommended" in capsys.readouterr().out

# Make sure the two different aperture bases for GGGG are allocated as requested
@pytest.mark.parametrize("basis,nout", [('MM*', 1), ('both', 2)])
def test_fourth_order_aperture_bases(basis, nout, shear_catalog, scalar_catalog):
    gggg = GGGGCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    gggg.process(shear_catalog, statistics='4pcf_multipole')
    out = gggg.computeMap4(RADII, basis=basis)
    assert len(out) == nout and all(_finite(a) for a in out)
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    gnnn.process(shear_catalog, scalar_catalog, statistics='4pcf_multipole')
    out = gnnn.computeMapNap3(RADII, basis=basis)
    assert len(out) == nout and all(_finite(a) for a in out)

# Make sure that GGGG runs in discrete low-mem up to Map4
def test_gggg_discrete_lowmem_apertures(shear_catalog):
    gggg = GGGGCorrelation_NoTomo(**SEPS, **ANGULAR, nthreads=NTHREADS, method="Discrete")
    m4, = gggg.process(shear_catalog, statistics='M4', mapradii=RADII, lowmem=True)
    assert np.shape(m4) == (8, 1, NRADII) and _finite(m4)
    assert np.any(np.asarray(m4) != 0.)

# The source and lens legs carry their own flag, and both default to on
def test_gnnn_tomographic_bins_can_be_collapsed_per_leg(shear_catalog, scalar_catalog):
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    gnnn.process(shear_catalog, scalar_catalog, statistics='4pcf_multipole',
                 dotomo_source=False, dotomo_lens=False)
    assert gnnn.nbinsz_source == 1 and gnnn.nbinsz_lens == 1
    assert _finite(gnnn.npcf_multipoles)

# Make sure multipoles2npcf works for nnnn
def test_nnnn_real_space_transforms(scalar_catalog):
    nnnn = NNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    nnnn.process(scalar_catalog, statistics='4pcf_multipole')
    single = np.asarray(nnnn.multipoles2npcf_singlethetcombi(0, 1, 2))
    assert single.shape == (NBINSPHI, NBINSPHI) and _finite(single)
    nnnn.multipoles2npcf()
    assert np.shape(nnnn.npcf) == (NBINSR, NBINSR, NBINSR, 1, NBINSPHI, NBINSPHI)
    assert _finite(nnnn.npcf)

# Make sure catalog builds spatial hash prior to running estimator
def test_nnnn_builds_the_spatial_hash_it_needs():
    rng = np.random.default_rng(42)
    ngal = 800
    cat = ScalarTracerCatalog(pos1=rng.uniform(0., 300., ngal),
                              pos2=rng.uniform(0., 300., ngal), tracer=np.ones(ngal),
                              weight=np.ones(ngal), geometry='flat2d')
    assert not cat.hasspatialhash
    nnnn = NNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    nnnn.process(cat, statistics='4pcf_multipole')
    assert cat.hasspatialhash and _finite(nnnn.npcf_multipoles)

# One radial-bin triplet at a time, and the n-convergence variant
def test_gggg_single_theta_transforms(gggg_multipoles):
    npcf, norm = gggg_multipoles.multipoles2npcf_singlethetcombi(0, 1, 2)
    assert np.shape(npcf) == (8, NBINSPHI, NBINSPHI)
    assert np.shape(norm) == (NBINSPHI, NBINSPHI)
    assert _finite(npcf) and np.any(np.asarray(npcf) != 0.)
    conv, conv_norm = gggg_multipoles.multipoles2npcf_gggg_singletheta_nconvergence(0, 1, 2)
    assert np.shape(conv) == (8, NMAX+1, NMAX+1, NBINSPHI, NBINSPHI)
    assert np.shape(conv_norm) == (NMAX+1, NMAX+1, NBINSPHI, NBINSPHI)
    assert _finite(conv)

 # One radial-bin triplet at a time, and the n-convergence variant
def test_gnnn_single_theta_transforms(gnnn_multipoles):
    npcf, norm = gnnn_multipoles.multipoles2npcf_singlethetcombi(0, 1, 2)
    assert np.shape(npcf) == (1, NBINSPHI, NBINSPHI)
    assert np.shape(norm) == (NBINSPHI, NBINSPHI)
    assert _finite(npcf) and np.any(np.asarray(npcf) != 0.)
    conv, conv_norm = gnnn_multipoles.multipoles2npcf_singletheta_nconvergence(0, 1, 2)
    assert _finite(conv) and np.shape(conv)[0] == 1


##############################
# FOURTH-ORDER GUARD CLAUSES #
##############################

# Make sure NNNN complains if it gets wrong catalog or wrong method
def test_nnnn_curved_sky_needs_the_tree_kernel(scalar_catalog):
    with pytest.raises(ValueError, match="spherical catalog"):
        NNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR,
                               process_spherical=True).process(
                                   scalar_catalog, statistics='4pcf_multipole')
    with pytest.raises(NotImplementedError, match="DoubleTree"):
        NNNNCorrelation_NoTomo(**SEPS, **ANGULAR, nthreads=NTHREADS, method="DoubleTree",
                               process_spherical=True, sep_units='arcmin').process(
                                   _sky_catalog('scalar', 41), statistics='4pcf_multipole')

# Make sure all admit that there is not edge correction
def test_fourth_order_has_no_edge_correction(shear_catalog, scalar_catalog):
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    with pytest.raises(NotImplementedError, match="[Ee]dge correction"):
        gnnn.process(shear_catalog, scalar_catalog, statistics='4pcf_multipole',
                     apply_edge_correction=True)

# Make suer GNNN refuses out-of-memory binnings
def test_gnnn_refuses_a_grid_it_cannot_hold(shear_catalog, scalar_catalog):
    # real space
    gnnn = GNNNCorrelation_NoTomo(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=200,
                                  nmaxs=4, nbinsphi=50, nthreads=NTHREADS)
    with pytest.raises(ValueError, match="too large"):
        gnnn.multipoles2npcf()
    # multipole space
    gnnn = GNNNCorrelation_NoTomo(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=64,
                                      nmaxs=45, nbinsphi=10, nthreads=NTHREADS)
    with pytest.raises(ValueError, match="Required memory too large"):
        gnnn.process(shear_catalog, scalar_catalog, statistics='4pcf_multipole',
                        lowmem=False)

# GGGG/GNNN refuse to compute real-space 4PCF without the multipoles present
@pytest.mark.parametrize("cls", [GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo])
def test_fourth_order_refuses_real_space_without_multipoles(cls):
    rng = np.random.default_rng(5)
    ngal = 500
    shear = SpinTracerCatalog(spin=2, pos1=rng.uniform(0., 300., ngal),
                              pos2=rng.uniform(0., 300., ngal),
                              tracer_1=rng.normal(0., .3, ngal),
                              tracer_2=rng.normal(0., .3, ngal),
                              weight=np.ones(ngal), geometry='flat2d')
    scalar = ScalarTracerCatalog(pos1=rng.uniform(0., 300., ngal),
                                 pos2=rng.uniform(0., 300., ngal), tracer=np.ones(ngal),
                                 weight=np.ones(ngal), geometry='flat2d')
    inst = cls(**SEPS, **ANGULAR, **TREE)
    with pytest.raises(ValueError, match="4pcf_real"):
        if cls is GNNNCorrelation_NoTomo:
            inst.process(shear, scalar, statistics='4pcf_real')
        else:
            inst.process(shear, statistics='4pcf_real')

# An invalid ``statistics`` names the offending value
@pytest.mark.parametrize("spec", correlators(orders=4), ids=correlator_ids(correlators(orders=4)))
def test_bad_statistics_reports_the_offending_value(spec, shear_catalog, scalar_catalog):
    inst = build_correlator(spec, **SEPS, **TREE, **ANGULAR)
    with pytest.raises(ValueError, match="4pcf_multipoles"):
        run_correlator(spec, inst, shear_catalog, scalar_catalog,
                       statistics='4pcf_multipoles')


###################################
# FOURTH-ORDER ANALYTIC MACHINERY #
###################################

# The Wick expansion of the shear 4PCF, evaluated straight from splined xi_pm
def test_gggg_disconnected_4pcf_in_the_x_projection(gauss_xi):
    xip_spl, xim_spl = _xi_splines(gauss_xi)
    phi12 = np.linspace(.2, 2.*np.pi-.2, 6)[:, None]
    phi13 = np.linspace(.2, 2.*np.pi-.2, 6)[None, :]
    gammas = GGGGCorrelation_NoTomo.fourpcf_gauss_x(5., 7., 9., phi12, phi13,
                                                    xip_spl, xim_spl)
    assert len(gammas) == 8
    for g in gammas:
        assert np.shape(g) == (6, 6) and _finite(g)
    assert np.any(np.asarray(gammas[0]) != 0.)

# The Wick expansion in C, integrated over the radial bins, and its multipoles
@pytest.mark.parametrize("nsubr", [1, 2])
def test_gggg_disconnected_4pcf_from_binned_xi(nsubr, gauss_xi):
    thetas, xip, xim = gauss_xi
    dtheta = (XI_MAX-XI_MIN)/XI_NTHETA
    gggg = GGGGCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    binned = gggg.gauss4pcf_analytic(0, 1, 2, nsubr, xip, xim, XI_MIN, XI_MAX, dtheta)
    assert np.shape(binned) == (8, NBINSPHI, NBINSPHI) and _finite(binned)
    assert np.any(np.asarray(binned) != 0.)
    multipoles = gggg.gauss4pcf_multipolebasis(0, 1, 2, nsubr, xip, xim,
                                               XI_MIN, XI_MAX, dtheta)
    assert np.shape(multipoles) == (8, 2*NMAX+1, 2*NMAX+1) and _finite(multipoles)

# Map4 straight from xi_pm, in each output basis.
@pytest.mark.parametrize("basis,nout", [('MapMx', 1), ('MM*', 1), ('both', 2)])
def test_gggg_analytic_map4(basis, nout, gauss_xi):
    xip_spl, xim_spl = _xi_splines(gauss_xi)
    gggg = GGGGCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR, verbosity=3)
    out = gggg.Map4analytic(RADII, xip_spl, xim_spl, XI_MIN, XI_MAX, XI_NTHETA, basis=basis)
    assert len(out) == nout
    for arr in out:
        assert np.shape(arr)[-1] == NRADII and _finite(arr)

# Get Wick prediction for Map4 using measured xipm
def test_gggg_estimates_the_disconnected_map4_from_a_catalog(shear_catalog):
    gggg = GGGGCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    map4, = gggg.estimateMap4disc(shear_catalog, RADII, nsubr=1)
    assert np.shape(map4)[-1] == NRADII and _finite(map4)

 # A reduced multipole block, rebuilt from one of the symmetry transformations
def test_gggg_reconstructs_multipoles_from_the_symmetries(shear_catalog):
    gggg = GGGGCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    gggg.process(shear_catalog, statistics='4pcf_multipole')
    nmax_rec = 1
    ups, norm = gggg.getMultipolesFromSymm(nmax_rec, 0, 1, 2, 0)
    assert np.shape(ups) == (8, 2*nmax_rec+1, 2*nmax_rec+1)
    assert np.shape(norm) == (2*nmax_rec+1, 2*nmax_rec+1)
    assert _finite(ups) and _finite(norm)

# The lens clustering enters the 4PCF through both a 2PCF and a 3PCF term
def test_gnnn_clustering_correction_inputs(shear_catalog, scalar_catalog):
    xi, nnn = _clustering_inputs(NBINSR, NBINSPHI)
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    # Only the low-memory kernel takes the correction, so it is resampled for C there
    gnnn.process(shear_catalog, scalar_catalog, statistics='4pcf_multipole',
                 xi=xi, nnn=nnn, lowmem=True)
    assert _finite(gnnn.npcf_multipoles)
    # The python-side correction multiplies the real-space 4PCF instead
    gnnn.multipoles2npcf()
    plain = np.array(gnnn.npcf, copy=True)
    gnnn.apply_clustering_correction(xi=xi, nnn=nnn)
    assert np.shape(gnnn.npcf) == np.shape(plain) and _finite(gnnn.npcf)
    assert np.any(np.asarray(gnnn.npcf) != plain)

# Second- and third-order corrections to <Map Nap^3>, with the lower-order
# correlators either supplied or set to zero
@pytest.mark.parametrize("supplied", [False, True])
def test_gnnn_aperture_corrections(supplied, shear_catalog, scalar_catalog):
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR, verbosity=3)
    gnnn.process(shear_catalog, scalar_catalog, statistics='4pcf_multipole')
    given = {}
    if supplied:
        given = dict(xi_ng=np.full(NBINSR, .01),
                     Gtilde_third=np.zeros(NBINSR*NBINSR*NBINSPHI, dtype=complex))
    out = gnnn.MapNap3_corrections(RADII, **given)
    assert np.shape(out) == (1, NRADII) and _finite(out)

 # The Gaussian-field prediction and the per-triplet corrections
def test_gnnn_disconnected_family(gauss_xi):
    thetas, xip, xim = gauss_xi
    dtheta = (XI_MAX-XI_MIN)/XI_NTHETA
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    binned = gnnn.gauss4pcf_analytic(0, 1, 2, 1, xip, xim, XI_MIN, XI_MAX, dtheta)
    assert np.shape(binned) == (1, NBINSPHI, NBINSPHI) and _finite(binned)
    corrs = gnnn.gnnn_corrections(0, 1, 2)
    assert np.shape(corrs) == (1, NBINSPHI, NBINSPHI) and _finite(corrs)

# <Map Nap^3> straight from a splined xi_ng and xi_nn
def test_gnnn_analytic_mapnap3(gauss_xi):
    xing_spl, xinn_spl = _xi_splines(gauss_xi)
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR, verbosity=3)
    out = gnnn.MapNap3analytic(RADII, xing_spl, xinn_spl, XI_MIN, XI_MAX, XI_NTHETA)
    assert len(out) == 1
    assert np.shape(out[0]) == (1, NRADII) and _finite(out[0])


#####################
# DIRECT ESTIMATORS #
#####################

def test_direct_mapn_equal_is_finite(mapn_equal):
    mapn, wmapn = (np.asarray(a) for a in mapn_equal)
    assert mapn.size and np.isfinite(mapn).any()
    assert wmapn.shape == mapn.shape
    # An undispatched branch returns zeros without raising, which no shape check sees
    assert not np.all(mapn == 0)

# Every supported filter needs its own case: getFilterU dispatches on the filter index and
# an unimplemented branch returns zeros for the whole aperture rather than raising.
@pytest.mark.parametrize("filter_form", ["S98", "C02", "PolyExp"])
def test_direct_napn_equal_is_finite(filter_form):
    rng = np.random.default_rng(8)
    ngal = 4000
    cat = ScalarTracerCatalog(pos1=rng.uniform(0., 300., ngal),
                              pos2=rng.uniform(0., 300., ngal),
                              tracer=np.ones(ngal),
                              weight=rng.uniform(.5, 1.5, ngal), geometry='flat2d')
    cat.create_mask(method="Basic", pixsize=2.)
    napn, wnapn = (np.asarray(a) for a in
                   Direct_NapnEqual(filter_form=filter_form, **DIRECT).process(cat, dotomo=False))
    assert napn.size and np.isfinite(napn).any()
    assert wnapn.shape == napn.shape
    # An undispatched branch returns zeros without raising, which no shape check sees
    assert not np.all(napn == 0)

# Make sure that Schirmer 2004 gets rejected in the Napn
def test_direct_napn_equal_rejects_filter_without_u():
    with pytest.raises(ValueError, match="scalar"):
        Direct_NapnEqual(filter_form="Sch04", **DIRECT)

 # Map3 at every combination of three aperture radii rather than at a single one. The
# combinations are enumerated by MapCombinatorics, so the two are checked together.
def test_direct_map3_unequal_is_finite():
    rng = np.random.default_rng(51)
    ngal = 4000
    cat = SpinTracerCatalog(spin=2,
                            pos1=rng.uniform(0., 300., ngal),
                            pos2=rng.uniform(0., 300., ngal),
                            tracer_1=rng.normal(0., .3, ngal),
                            tracer_2=rng.normal(0., .3, ngal),
                            weight=rng.uniform(.5, 1.5, ngal), geometry='flat2d')
    cat.create_mask(method="Basic", pixsize=2.)
    unequal = {k: v for k, v in DIRECT.items() if k != 'order_max'}
    d = Direct_Map3Unequal(**unequal)
    map3, wmap3 = d.process_discrete(cat, dotomo=False)
    ntriplets = MapCombinatorics(nradii=DIRECT['nbinsr'], order_max=3).nindices
    assert np.shape(map3) == np.shape(wmap3) == (len(d.frac_covs), ntriplets)
    assert np.isfinite(np.asarray(map3)).any()
    assert not np.all(np.asarray(map3) == 0)

# getmap/getnap run
@pytest.mark.parametrize("cls,kind", [(Direct_MapnEqual, 'shear'), (Direct_NapnEqual, 'scalar')])
def test_direct_returns_per_centre_maps(cls, kind):
    rng = np.random.default_rng(63)
    ngal = 4000
    shared = dict(pos1=rng.uniform(0., 300., ngal), pos2=rng.uniform(0., 300., ngal),
                  weight=rng.uniform(.5, 1.5, ngal),
                  zbins=rng.integers(0, NBINSZ, ngal), geometry='flat2d')
    if kind == 'scalar':
        cat = ScalarTracerCatalog(tracer=np.ones(ngal), **shared)
    else:
        cat = SpinTracerCatalog(spin=2, tracer_1=rng.normal(0., .3, ngal),
                                tracer_2=rng.normal(0., .3, ngal), **shared)
    cat.create_mask(method="Basic", pixsize=2.)
    estimator = cls(**DIRECT)
    maps = (estimator.getnap(0, cat, dotomo=True) if kind == 'scalar'
            else estimator.getmap(0, cat, dotomo=True))
    counts, covs, msn, sn, statn, statn_norm = maps
    ncen2, ncen1 = np.shape(counts)[-2:]
    assert np.shape(counts) == (NBINSZ, 3, ncen2, ncen1)
    assert np.shape(covs) == (2, ncen2, ncen1)
    for arr in (msn, sn, statn, statn_norm):
        assert np.shape(arr) == (NBINSZ, DIRECT['order_max'], ncen2, ncen1)
        assert _finite(arr)
    assert np.any(np.asarray(statn) != 0.)

# A saved direct estimator can be used to rebuild the class instance
def test_direct_saveinst_loadinst_round_trip(tmp_path):
    inst = Direct_MapnEqual(filter_form="C02", **DIRECT)
    inst.saveinst(str(tmp_path) + '/', 'inst')
    back = Direct_MapnEqual.loadinst(str(tmp_path) + '/', 'inst')
    for attr in ('Rmin', 'Rmax', 'nbinsr', 'order_max', 'filter_form', 'ap_weights',
                 'aperture_centers', 'multicountcorr'):
        assert getattr(back, attr) == getattr(inst, attr), attr
    assert np.allclose(back.radii, inst.radii)
    assert np.allclose(back.frac_covs, inst.frac_covs)

# ind2sel and sel2ind are true inverses of each other.
@pytest.mark.parametrize("nradii,order_max", [(3, 3), (4, 3), (5, 2), (4, 4)])
def test_map_combinatorics_indices_round_trip(nradii, order_max):
    combis = MapCombinatorics(nradii=nradii, order_max=order_max)
    for ind in range(combis.nindices):
        sel = combis.ind2sel(ind)
        assert len(sel) == order_max
        assert np.all(np.diff(sel) >= 0), "radius selections come out sorted"
        assert np.all((sel >= 0) & (sel < nradii))
        assert combis.sel2ind(sel) == ind
    # The last index is the all-largest combination
    assert np.array_equal(combis.ind2sel(combis.nindices-1),
                          (nradii-1)*np.ones(order_max, dtype=int))

###################################
# TREE SETUP AND SCHEME AGREEMENT #
###################################

# Make sure that all tree-based schemes with tree_resos=[0.] reproduce discrete
def test_schemes_agree_on_a_fully_discrete_tree():
    # Init some small test catalogs
    rng = np.random.default_rng(3)
    n, box = 800, 300.
    p1, p2 = rng.uniform(0., box, n), rng.uniform(0., box, n)
    shear = SpinTracerCatalog(spin=2, pos1=p1, pos2=p2, tracer_1=rng.normal(0., .3, n), 
                              tracer_2=rng.normal(0., .3, n), weight=np.ones(n), geometry='flat2d')
    scalar = ScalarTracerCatalog(pos1=p1, pos2=p2, tracer=np.ones(n), 
                                 weight=np.ones(n), geometry='flat2d',)
    # Init output for ratios and populated bins
    devs, populated = {}, {}
    # Compute all the deviations
    thiscorrs = correlators(orders=3)
    for spec in thiscorrs:
        first = None
        for method in build_correlator(spec, **DISCRETE_TREE).methods_avail:
            inst = build_correlator(spec, **DISCRETE_TREE, method=method)
            run_correlator(spec, inst, shear, scalar)
            thisnpcf = np.asarray(inst.npcf_multipoles)
            # We just take the first method as we assert all agree, so we dont have to take the discrete one
            if first is None:
                first = thisnpcf
                scale = np.max(np.abs(thisnpcf))
                populated[spec.cls.__name__] = bool(scale > 0.)
                continue
            thisdev = float(np.max(np.abs(thisnpcf-first))/scale) if scale > 0. else float('nan')
            devs[(spec.cls.__name__, method)] = thisdev
    # Check that reference correlator is populated
    assert set(populated) == {corr.cls.__name__ for corr in thiscorrs}, populated
    for name, ok in sorted(populated.items()):
        assert ok, (name, "reference scheme returned an empty correlator")
    # Check that all the ratios agree
    for key, dev in sorted(devs.items()):
        assert dev < RTOL_EXACT, (key, dev)

# Make sure that for a fully discrete binning scheme (i.e. tree_resos=[0.]) we infer a
# useful pixelsize for the spatial hashing.
@pytest.mark.parametrize("spec", CORRELATORS, ids=correlator_ids(CORRELATORS))
def test_fully_discrete_tree_gets_a_usable_hash_cellsize(spec):
    inst = build_correlator(spec, **SEPS, tree_resos=[0.], nthreads=NTHREADS)
    assert inst.dpix_hash > 0. and np.isfinite(inst.dpix_hash)

# Make sure that the inferred pixelsizes for autoset_tree are tied to the catalog nbar
@pytest.mark.parametrize("nbar_sparse,nbar_dense", [(.05, 20.)])
def test_autoset_tree_follows_the_number_density(nbar_sparse, nbar_dense):
    box = 100.
    finest = {}
    for nbar in (nbar_sparse, nbar_dense):
        ngal = int(nbar*box*box)
        rng = np.random.default_rng(3)
        cat = SpinTracerCatalog(spin=2, pos1=rng.uniform(0., box, ngal),
                                pos2=rng.uniform(0., box, ngal),
                                tracer_1=rng.normal(0., .3, ngal),
                                tracer_2=rng.normal(0., .3, ngal), geometry='flat2d')
        inst = GGGCorrelation(n_cfs=4, min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                              nbinsphi=NBINSPHI, nmaxs=NMAX, nthreads=NTHREADS)
        inst.autoset_tree(cat)
        resos = np.atleast_1d(inst.tree_resos)
        nonzero = resos[resos > 0.]
        finest[nbar] = nonzero.min() if len(nonzero) else np.inf
    assert finest[nbar_dense] < finest[nbar_sparse], (
        "the denser catalog was given cells of %g, no finer than the %g of the sparse one"%(
            finest[nbar_dense], finest[nbar_sparse]))

# Guard autoset_tree from bad inputs; here from too small max_sep
@pytest.mark.parametrize("max_sep", [15., 40., 100.])
def test_autoset_tree_keeps_the_radial_edges_monotonic(max_sep, shear_catalog):
    inst = GGGCorrelation(n_cfs=4, min_sep=MIN_SEP, max_sep=max_sep, nbinsr=NBINSR,
                          nbinsphi=NBINSPHI, nmaxs=NMAX, nthreads=NTHREADS)
    inst.autoset_tree(shear_catalog)
    redges = np.atleast_1d(inst.tree_redges)
    assert np.all(np.diff(redges) >= 0.), (
        "max_sep=%g leaves tree_redges non-monotonic: %s"%(max_sep, redges))
    assert redges[-1] == max_sep
    # Every shell has to carry a non-negative number of radial bins
    resosatr = np.atleast_1d(inst.tree_resosatr)
    assert resosatr[-1] < inst.tree_nresos

@pytest.mark.parametrize("spec, method", _undeclared_params())
def test_undeclared_methods_are_rejected(spec, method):
    with pytest.raises((AssertionError, NotImplementedError, ValueError)):
        build_correlator(spec, **SEPS, method=method)


#################
# SERIALISATION #
#################

# A reloaded instance carries its configuration and everything it measured.
@pytest.mark.parametrize("spec", CORRELATORS, ids=correlator_ids(CORRELATORS))
def test_saveinst_loadinst_round_trip(spec, shear_catalog, scalar_catalog, tmp_path):
    # Init the instance and process a small cat
    inst = build_correlator(spec, **SEPS, **TREE, **ANGULAR,
                            **SAVED_CTOR.get(spec.cls.__name__, {}))
    extra = dict(statistics='4pcf_multipole') if spec.order == 4 else {}
    if spec.cls is NNNCorrelation:
        extra['cat_random'] = scalar_catalog
    run_correlator(spec, inst, shear_catalog, scalar_catalog, tomo=True, **extra)
    # Save the instance
    inst.saveinst(str(tmp_path) + '/', 'inst')
    # Load back the instance
    back = spec.cls.loadinst(str(tmp_path) + '/', 'inst')
    # Make sure that they are the same for the relevant quantities, i.e. binning
    # scheme and correlators
    assert back.min_sep == inst.min_sep and back.nbinsr == inst.nbinsr
    assert back.order == inst.order and back.n_cfs == inst.n_cfs
    for name in SAVED_EXTRAS.get(spec.cls.__name__, ('nbinsz', 'nzcombis')):
        got, want = getattr(back, name), getattr(inst, name)
        assert np.all(np.asarray(got) == np.asarray(want)), name
    for name in correlator_outputs(spec):
        assert np.allclose(np.asarray(getattr(back, name)),
                           np.asarray(getattr(inst, name)), rtol=0., atol=0.), name

# The exported pickle helpers, which create their parent directory on the way
def test_pickle_save_load_round_trip(tmp_path):
    payload = {"array": np.arange(5), "text": "value"}
    target = str(tmp_path / "nested" / "thing.pkl")
    pickle_save(payload, target)
    back = pickle_load(target)
    assert sorted(back) == ["array", "text"]
    assert np.array_equal(back["array"], payload["array"])
    assert back["text"] == payload["text"]
