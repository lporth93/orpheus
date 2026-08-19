# Here we collect all tests required for fast tier which is always triggered.
# * Check that none of the correlator classes are crashing & give finite
#   output of the expected shape
# * Check that one can create an identical correlator instance from a saved one
# * Check that patch decomposition yields consistent patches
# 
# The only checks made are that there are no crashes and that the output is
# of the right shape and finite. In particular, there is no judgement made 
# about numerical correctness, this is left for the slow tier test suite.
# 
# The full test suite should take far less than a minute to complete.

import numpy as np
import pytest

from orpheus.catalog import ScalarTracerCatalog, SpinTracerCatalog
from orpheus.direct import Direct_MapnEqual, Direct_NapnEqual
from orpheus.npcf_fourth import GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo, NNNNCorrelation_NoTomo
from orpheus.npcf_second import GGCorrelation, NGCorrelation, NNCorrelation
from orpheus.npcf_third import GGGCorrelation, GNNCorrelation, NGGCorrelation, NNNCorrelation

from conftest import (CORRELATORS, MAX_SEP, MIN_SEP, NBINSR, NBINSZ, NTHREADS,
                      RTOL_EXACT, build_correlator, correlator_ids, correlator_outputs,
                      correlators, run_correlator)

##########################
# PARAMETERS AND HELPERS #
##########################
SEPS = dict(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR)
TREE = dict(tree_resos=[0., 2., 4.], rmin_pixsize=8, nthreads=NTHREADS)
NMAX, NBINSPHI = 4, 10
ANGULAR = dict(nmaxs=NMAX, nbinsphi=NBINSPHI)
RADII = np.array([MAX_SEP/8., MAX_SEP/6.])
NRADII = len(RADII)

# nzcombis carries one factor of nbinsz per leg, so it grows with the order.
NZ2, NZ3 = NBINSZ**2, NBINSZ**3
ALL_METHODS = ["Discrete", "Tree", "BaseTree", "DoubleTree"]


# Get all methods available for a certain correlator
def _methods(cls, **extra):
    inst = cls(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR, **extra)
    return list(inst.methods_avail)

METHODS = _methods(GGGCorrelation, n_cfs=4)
GNN_METHODS = _methods(GNNCorrelation)
NGG_METHODS = _methods(NGGCorrelation)


def _finite(arr):
    a = np.asarray(arr)
    return a.size > 0 and np.isfinite(a).all()

def _third_kwargs(method):
    kwargs = dict(**SEPS, **ANGULAR, nthreads=NTHREADS, method=method)
    if method != "Discrete":
        kwargs.update(tree_resos=TREE['tree_resos'], rmin_pixsize=TREE['rmin_pixsize'])
    return kwargs

###############################
# CHECK THAT ALL CF-PIPELINES #
#     RUN WITHOUT CRASHING    #
###############################

## SECOND ORDER ##
def test_nn_runs_the_full_pipeline(scalar_catalog):
    """Pair counts, Landy-Szalay, nap2."""
    nn = NNCorrelation(**SEPS, **TREE)

    nn.process(scalar_catalog, cat_random=scalar_catalog, dotomo=True)
    # With randoms the four count types DD, DR, RD, RR stack along the leading axis.
    assert np.shape(nn.npair) == (4*NZ2, NBINSR)
    assert np.shape(nn.xi) == (NZ2, NBINSR)
    assert _finite(nn.npair) and _finite(nn.xi)
    assert np.all(np.asarray(nn.npair) >= 0.)

    nap2 = np.asarray(nn.computeNap2(RADII))
    assert nap2.shape == (NZ2, NRADII) and nap2.dtype == np.float64 and _finite(nap2)


def test_gg_runs_the_full_pipeline(shear_catalog):
    """xi_pm, map2, pure-mode"""
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


# The log-COSEBI roots are built with mpmath, which the library imports lazily and does not
# declare as a dependency, so this is the one statistic a minimal install cannot compute.
def test_gg_computes_cosebis(shear_catalog):
    """cosebis, on installations that carry mpmath"""
    pytest.importorskip("mpmath", reason="mpmath is optional, only log-COSEBIs need it")
    gg = GGCorrelation(**SEPS, **TREE)
    gg.process(shear_catalog, dotomo=True)
    nmodes = 3
    cosebi = np.asarray(gg.computecosebi(nmodes))
    assert cosebi.shape == (4, NZ2, nmodes) and _finite(cosebi)


def test_ng_runs_the_full_pipeline(shear_catalog, scalar_catalog):
    """xi, mapnap"""
    ng = NGCorrelation(**SEPS, **TREE)

    ng.process(shear_catalog, scalar_catalog, dotomo=True)
    assert np.shape(ng.xi) == (NZ2, NBINSR)
    assert np.iscomplexobj(np.asarray(ng.xi)) and _finite(ng.xi)

    mapnap = np.asarray(ng.computeMapNap(RADII))
    assert mapnap.shape == (NZ2, NRADII) and _finite(mapnap)



## THIRD ORDER ##
def test_nnn_runs_the_full_pipeline(scalar_catalog):
    """Triplets in real-space and in multipole-space"""
    nnn = NNNCorrelation(**_third_kwargs("DoubleTree"))
    
    nnn.process(scalar_catalog, dotomo=True)
    assert np.shape(nnn.npcf_multipoles) == (1, NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(nnn.npcf_multipoles)

    nnn.multipoles2npcf()
    assert np.shape(nnn.npcf) == (1, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(nnn.npcf)


@pytest.mark.parametrize("method", METHODS)
def test_ggg_runs_the_full_pipeline(method, shear_catalog):
    """Natural components in real-space and in multipole-space, map3."""
    ggg = GGGCorrelation(n_cfs=4, **_third_kwargs(method))

    ggg.process(shear_catalog, dotomo=True)
    assert np.shape(ggg.npcf_multipoles) == (4, NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(ggg.npcf_multipoles)

    ggg.multipoles2npcf(projection='Centroid')
    assert np.shape(ggg.npcf) == (4, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert np.iscomplexobj(np.asarray(ggg.npcf)) and _finite(ggg.npcf)

    map3 = np.asarray(ggg.computeMap3(RADII, basis='MapMx'))
    assert map3.shape == (8, NZ3, NRADII) and _finite(map3)


@pytest.mark.parametrize("method", GNN_METHODS)
def test_gnn_runs_the_full_pipeline(method, shear_catalog, scalar_catalog):
    """G3L corrfunc in real-space and in multipole-space, mapnap2"""
    gnn = GNNCorrelation(**_third_kwargs(method))

    gnn.process(shear_catalog, scalar_catalog, dotomo_source=True, dotomo_lens=True)
    assert np.shape(gnn.npcf_multipoles) == (1, NMAX+1, NZ3, NBINSR, NBINSR)
    assert _finite(gnn.npcf_multipoles)

    gnn.multipoles2npcf()
    assert np.shape(gnn.npcf) == (1, NZ3, NBINSR, NBINSR, NBINSPHI)
    assert _finite(gnn.npcf)

    nnm = np.asarray(gnn.computeNNM(RADII))
    assert nnm.shape == (1, NZ3, NRADII) and _finite(nnm)

@pytest.mark.parametrize("method", NGG_METHODS)
def test_ngg_runs_the_full_pipeline(method, shear_catalog, scalar_catalog):
    """G+- in real-space and in multipole-space, map2nap"""
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
FOURTH_SHAPE = (1, NBINSR, NBINSR, NBINSR, NBINSPHI, NBINSPHI)
def test_nnnn_runs_the_full_pipeline(scalar_catalog):
    """Triplets in real-space and in multipole-space, nap4."""
    nnnn = NNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)

    n4, = nnnn.process(scalar_catalog, mapradii=RADII)
    assert np.shape(nnnn.npcf_multipoles) == (
        2*NMAX+1, 2*NMAX+1, 1,NBINSR, NBINSR, NBINSR)
    assert _finite(nnnn.npcf_multipoles)
    assert np.shape(nnnn.npcf) == (1, NBINSR, NBINSR, NBINSR, NBINSPHI, NBINSPHI)
    assert _finite(nnnn.npcf)
    assert np.shape(n4) == (1, NRADII) and _finite(n4)


def test_gggg_runs_the_full_pipeline(shear_catalog):
    """Natural components in real-space and in multipole-space, map3."""
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


def test_gnnn_runs_the_full_pipeline(shear_catalog, scalar_catalog):
    """G4L corrfunc in real-space and in multipole-space, mapnap3"""
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)

    napmap, = gnnn.process(shear_catalog, scalar_catalog, apradii=RADII)
    assert np.shape(gnnn.npcf_multipoles) == (
        1, 2*NMAX+1, 2*NMAX+1, 1, NBINSR, NBINSR, NBINSR)
    assert _finite(gnnn.npcf_multipoles)
    assert np.shape(napmap) == (1, 1, NRADII) and _finite(napmap)
    assert np.shape(gnnn.computeMapNap3(RADII)) == (1, 1, NRADII)

def test_gnnn_runs_on_default_arguments(shear_catalog, scalar_catalog):
    """The default ``lowmem=None`` resolves to a scheme rather than breaking."""
    gnnn = GNNNCorrelation_NoTomo(**SEPS, **TREE, **ANGULAR)
    gnnn.process(shear_catalog, scalar_catalog, apradii=RADII)
    assert _finite(gnnn.npcf_multipoles)

####################
# SOFT FAILURE FOR #
#   WRONG METHODS  #
####################

# In this test we make sure that orpheus exits with an error message if a method
# is chosen that is not contained within methods_avail

# Generate a list of methods that are not implemented within different correlators
def _undeclared_params():
    out = []
    for spec in CORRELATORS:
        declared = build_correlator(spec, **SEPS).methods_avail
        for method in ALL_METHODS:
            if method not in declared:
                out.append(pytest.param(spec, method, id='%s-%s'%(spec.cls.__name__, method)))
    return out

# Run the test
@pytest.mark.parametrize("spec, method", _undeclared_params())
def test_undeclared_methods_are_rejected(spec, method):
    with pytest.raises((AssertionError, NotImplementedError, ValueError)):
        build_correlator(spec, **SEPS, method=method)


###################
# BAD STATISTICS  #
###################

# Test whether orpheus complains if a wrong statistic is specified. Only the fourth-order
# correlators take a `statistics` argument, so the family is complete at three members.
BAD_STATS = correlators(orders=4)

@pytest.mark.parametrize("spec", BAD_STATS, ids=correlator_ids(BAD_STATS))
def test_bad_statistics_reports_the_offending_value(spec, shear_catalog, scalar_catalog):
    """An invalid ``statistics`` names the offending value."""
    inst = build_correlator(spec, **SEPS, **TREE, **ANGULAR)
    with pytest.raises(ValueError, match="4pcf_multipoles"):
        run_correlator(spec, inst, shear_catalog, scalar_catalog,
                       statistics='4pcf_multipoles')


#################
# SERIALISATION #
#################

# In this test we assert that saving a NPCF instance and the using this to init
# a new instance generates exactly the same instance.

# Collection of all the extra args that are saved in the serialisation of the
# various correlaotrs
SAVED_EXTRAS = {
    'NGCorrelation':          ('nbinsz_shape', 'nbinsz_pos'),
    'NNNCorrelation':         ('nbinsz', 'nzcombis', 'zeta'),
    'GNNCorrelation':         ('nbinsz_source', 'nbinsz_lens',
                               'zweighting', 'zweighting_sigma'),
    'NGGCorrelation':         ('nbinsz_source', 'nbinsz_lens'),
    'NNNNCorrelation_NoTomo': ('nbinsz', 'nzcombis', 'thetabatchsize_max'),
    'GGGGCorrelation_NoTomo': ('nbinsz', 'nzcombis', 'thetabatchsize_max'),
    'GNNNCorrelation_NoTomo': ('nbinsz_source', 'nbinsz_lens', 'nzcombis',
                               'thetabatchsize_max'),
}

# Choose some values for non-default args that are not part of the serialisation
SAVED_CTOR = {
    'GNNCorrelation':         dict(zweighting=True, zweighting_sigma=.15),
    'NNNNCorrelation_NoTomo': dict(thetabatchsize_max=512),
    'GGGGCorrelation_NoTomo': dict(thetabatchsize_max=512),
    'GNNNCorrelation_NoTomo': dict(thetabatchsize_max=512),
}

# Run the tests
@pytest.mark.parametrize("spec", CORRELATORS, ids=correlator_ids(CORRELATORS))
def test_saveinst_loadinst_round_trip(spec, shear_catalog, scalar_catalog, tmp_path):
    """A reloaded instance carries its configuration and everything it measured."""
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

#######################
# PATCH DECOMPOSITION #
#######################

# In this test we assert that the patch decomposition works as expected, i.e. that
# the number of inner galaxies summed over the patches equals the number of galaxies

# Run the test
def test_patch_decomposition_unique_inner(spherical_catalog):
    """Patch inner regions tile the survey; only the buffers overlap."""

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


###########################
# DOUBLETREE --> DISCRETE #
###########################

# In this test we make sure that the DoubleTree estimator equals the Discrete one
# if tree_resos=[0.]. 

# Just some binning setup that allows for a fast computation
DISCRETE_TREE = dict(min_sep=1., max_sep=40., nbinsr=4, nmaxs=4, nbinsphi=10, nthreads=NTHREADS,
                     tree_resos=[0.], rmin_pixsize=8)

# Selection of correlators for this test
thiscorrs = correlators(orders=3)

# Run the test
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

####################
# DIRECT ESTIMATOR #
####################
# The direct aperture-mass estimators (Porth & Smith 2022) had no test coverage at all.
# Their catalogs are built here rather than taken from the shared fixtures, since the
# direct estimators need an angular mask and create_mask mutates the catalog.
DIRECT = dict(order_max=3, Rmin=4., Rmax=8., nbinsr=3, nthreads=NTHREADS, accuracies=1.)


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


def test_direct_mapn_equal_is_finite(mapn_equal):
    mapn, wmapn = (np.asarray(a) for a in mapn_equal)
    assert mapn.size and np.isfinite(mapn).any()
    assert wmapn.shape == mapn.shape
    # An undispatched branch returns zeros without raising, which no shape check sees
    assert not np.all(mapn == 0)


def test_direct_napn_equal_is_finite():
    rng = np.random.default_rng(8)
    ngal = 4000
    cat = ScalarTracerCatalog(pos1=rng.uniform(0., 300., ngal),
                              pos2=rng.uniform(0., 300., ngal),
                              tracer=np.ones(ngal),
                              weight=rng.uniform(.5, 1.5, ngal), geometry='flat2d')
    cat.create_mask(method="Basic", pixsize=2.)
    napn, wnapn = (np.asarray(a) for a in
                   Direct_NapnEqual(**DIRECT).process(cat, dotomo=False))
    assert napn.size and np.isfinite(napn).any()
    assert not np.all(napn == 0)


# The tree cell sizes come from an estimate of the catalog's number density. That estimate
# counts occupied cells on a helper grid, so unless the grid is tied to the catalog it
# saturates at one galaxy per cell and a sparse catalog is handed the same ladder as a dense
# one -- tree overhead with nothing to group.
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


# The tree's radial edges are rmin_pixsize*reso, so a ladder chosen without reference to
# max_sep can put its coarse levels beyond it and leave tree_redges non-monotonic. The
# negative shell widths that follow reach the kernels as negative allocation sizes.
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
