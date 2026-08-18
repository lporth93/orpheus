# Here we check that the DoubleTree estimator converges to the discrete one when
# adjusting the rmin_pixsize parameter. We further check that the measurement on
# a patch-decomposed catalog works as expected.

import numpy as np
import pytest

from orpheus.catalog import SpinTracerCatalog
from orpheus.npcf_second import GGCorrelation
from orpheus.npcf_third import GGGCorrelation

from conftest import (CHI, RECOMMENDED, RTOL_EXACT, NTHREADS_SLOW as NTHREADS,
                      masked_ratio_deviation)
from reference import AnalyticField

R0 = 3.
FIELD = AnalyticField(gamma0=.05, r0=R0, chi=CHI, boxsize=RECOMMENDED['boxsize']*R0)
MIN_SEP = RECOMMENDED['min_sep_third']*R0
MAX_SEP = RECOMMENDED['max_sep']*R0
NGAL = 25_000
RESOS = [0., .05, .1, .2, .4]


@pytest.fixture(scope="module")
def small_catalog():
    return FIELD.catalogs(NGAL)[0]

# Perform gg measurement
def _gg(cat, **kw):
    gg = GGCorrelation(min_sep=MIN_SEP, max_sep=MAX_SEP, binsize=.1,
                       nthreads=NTHREADS, **kw)
    gg.process(cat, dotomo=False)
    return np.asarray(gg.xip).ravel(), np.asarray(gg.xim).ravel()

# Run test for second gg
# We need 50k gals for this to converge
# rmin_pixsize=(20,40,80) --> xi_p 6.0e-3, 2.9e-3, 6.0e-4; xi_m 1.8e-2, 2.7e-3, 9.8e-4.
def test_second_order_tree_converges_to_the_exact_estimator(small_catalog):
    ref_p, ref_m = _gg(small_catalog, tree_resos=[0.])
    devs = []
    for rmin_pixsize in (20, 40, 80):
        xp, xm = _gg(small_catalog, tree_resos=RESOS, rmin_pixsize=rmin_pixsize)
        devs.append((masked_ratio_deviation(xp, ref_p),
                     masked_ratio_deviation(xm, ref_m)))
    devs = np.array(devs)
    # Assert that with higher rmin_pixsize the result converges to discrete estimator
    assert np.all(np.diff(devs, axis=0) < 0.), devs
    # Assert that the finest doubletree has converged well to the discrete estimator
    assert devs[-1, 0] < 1e-3, devs[:, 0]
    assert devs[-1, 1] < 1.5e-3, devs[:, 1]


# Run test for ggg
# We need 50k gals for this to converge
# rmin_pixsize=(20,40) --> Tree 4.0e-3, 9.1e-4; BaseTree 4.2e-3, 9.9e-4; DoubleTree 5.6e-3, 1.3e-3.
@pytest.mark.parametrize("method", ["Tree", "BaseTree", "DoubleTree"])
def test_third_order_approximations_converge_to_discrete(method, small_catalog):
    common = dict(n_cfs=4, min_sep=MIN_SEP, max_sep=MAX_SEP, binsize=.2, nmaxs=10,
                  nbinsphi=50, nthreads=NTHREADS)
    ref = GGGCorrelation(method='Discrete', **common)
    ref.process(small_catalog, dotomo=False)
    ref.multipoles2npcf(projection='Centroid')
    target = np.asarray(ref.npcf)[:, 0]

    devs = []
    for rmin_pixsize in (20, 40):
        corr = GGGCorrelation(method=method, tree_resos=RESOS,
                              rmin_pixsize=rmin_pixsize, **common)
        corr.process(small_catalog, dotomo=False)
        corr.multipoles2npcf(projection='Centroid')
        devs.append(masked_ratio_deviation(np.asarray(corr.npcf)[:, 0], target, kind='rms'))
    # Assert that with higher rmin_pixsize the result converges to discrete estimator
    assert devs[1] < devs[0], devs
    # Assert that the finest doubletree has converged well to the discrete estimator
    assert devs[1] < 2e-3, devs


#######################
# PATCH DECOMPOSITION #
#######################


# In this test we assert that the measurement of GGG on the full-sky and on the patches
# is compatible. There are two routes, and they turn out to be very different statements:
#
# 1) Keep the patches in spherical geometry, so nothing is projected or rotated, 
#    --> Should be exactly identical
# 2) Route the patches through the flat-sky projection, i.e. the standard routing. 
#    --> Projections will differ, thus only look at counts
# TODO: Find better error metric for 2) and retain the projection bit

# Setup for ggg computation
PATCH_KW = dict(n_cfs=4, min_sep=10., max_sep=60., nbinsr=4, nmaxs=4, nbinsphi=10,
                method='DoubleTree', tree_resos=[0., 2., 5.], rmin_pixsize=8,
                nthreads=NTHREADS)

# Build a full-sky catalog. Note that .topatches mutates the instance so we have
# to rebuild it for each run...
def _sky_catalog():
    rng = np.random.default_rng(5)
    ngal = 20000
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-10.)),
                                           np.sin(np.radians(10.)), ngal)))
    return SpinTracerCatalog(spin=2, pos1=rng.uniform(20., 40., ngal), pos2=dec,
                             tracer_1=rng.normal(0., .3, ngal),
                             tracer_2=rng.normal(0., .3, ngal),
                             weight=np.ones(ngal), geometry='spherical',
                             units_pos1='deg', units_pos2='deg')

def _patched_sky():
    """A patch-decomposed sky catalog. Rebuilt per use, as .topatches mutates in place."""
    cat = _sky_catalog()
    cat.topatches(npatches=12, method='healpix', healpix_nside=8, patchextend_deg=2.5,
                  n_workers=1)
    return cat


def _spherical_patch(cat, index):
    """The members of one patch, kept on the sphere rather than projected to a tangent plane.

    ``Catalog.frompatchind`` deliberately returns a rotated flat2d patch, so it cannot serve
    the unprojected route; the member indices are taken from ``patchinds`` instead.
    """
    inds = cat.patchinds['patches'][index]
    inner, outer = np.asarray(inds['inner'], dtype=int), np.asarray(inds['outer'], dtype=int)
    sel = np.concatenate([inner, outer])
    isinner = np.concatenate([np.ones(len(inner), bool), np.zeros(len(outer), bool)])
    return SpinTracerCatalog(
        spin=2, pos1=np.asarray(cat.pos1)[sel], pos2=np.asarray(cat.pos2)[sel],
        tracer_1=np.asarray(cat.tracer_1)[sel], tracer_2=np.asarray(cat.tracer_2)[sel],
        weight=np.asarray(cat.weight)[sel], isinner=isinner, geometry='spherical',
        units_pos1='deg', units_pos2='deg')


# Do the test for route 1
def test_spherical_patches_partition_the_full_sky_multipoles():
    """Unprojected patches partition the multiplets exactly: 7e-18 norm, 7e-15 multipoles."""
    native = GGGCorrelation(**PATCH_KW, process_spherical=True)
    native.process(_sky_catalog(), dotomo=False)
    want = np.asarray(native.npcf_multipoles)
    want_norm = np.asarray(native.npcf_multipoles_norm)

    cat = _patched_sky()
    assert cat.npatches > 1, "footprint gave a single patch, nothing to test"
    got, got_norm = np.zeros_like(want), np.zeros_like(want_norm)
    for index in range(cat.npatches):
        patch = GGGCorrelation(**PATCH_KW, process_spherical=True)
        patch.process(_spherical_patch(cat, index), dotomo=False)
        got += np.asarray(patch.npcf_multipoles)
        got_norm += np.asarray(patch.npcf_multipoles_norm)

    assert FIELD.deviation(want_norm, got_norm) < RTOL_EXACT
    assert FIELD.deviation(want, got) < RTOL_EXACT, FIELD.deviation(want, got)


# Do the test for route 2, counts only
def test_flat_patches_match_the_full_sky_counts_to_the_projection_error():
    """The flat-sky routing preserves the triplet counts to 9e-4."""
    native = GGGCorrelation(**PATCH_KW, process_spherical=True)
    native.process(_sky_catalog(), dotomo=False)
    flat = GGGCorrelation(**PATCH_KW)
    flat.process(_patched_sky(), dotomo=False)
    a = np.asarray(native.npcf_multipoles_norm)
    b = np.asarray(flat.npcf_multipoles_norm)
    assert np.max(np.abs(a - b))/ np.max(np.abs(a)) < 5e-3, np.unravel_index(
        np.argmax(np.abs(a - b)), a.shape)