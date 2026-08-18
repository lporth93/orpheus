# Here we collect a bunch of small tests that are tied to 2pt correlators
# specifically

import numpy as np

from orpheus.catalog import ScalarTracerCatalog
from orpheus.npcf_second import GGCorrelation, NGCorrelation, NNCorrelation

from conftest import BOXSIZE, MAX_SEP, MIN_SEP, NBINSR, NTHREADS, RTOL_EXACT

TREE_RESOS = [0., 2., 4.]
RMIN_PIXSIZE = 16

##########################################
# SUPER CRUDE BRUTE FORCE 2PT ESTIMATORS #
##########################################

def _bin_edges():
    return np.geomspace(MIN_SEP, MAX_SEP, NBINSR+1)

def _pairs(pos1, pos2, weight):
    """Yield (index, separation bin, pair weight) for every ordered pair."""
    edges = _bin_edges()
    for i in range(len(pos1)):
        rel1, rel2 = pos1 - pos1[i], pos2 - pos2[i]
        sep = np.sqrt(rel1*rel1 + rel2*rel2)
        sel = np.where((sep >= MIN_SEP) & (sep < MAX_SEP))[0]
        if not len(sel):
            continue
        rbin = np.searchsorted(edges, sep[sel], side='right') - 1
        yield i, sel, rbin, weight[i]*weight[sel], rel1[sel], rel2[sel], sep[sel]


def _bruteforce_gg(cat):
    """Direct sum for xi_pm over all ordered pairs, matching ``do_dc=False``."""
    num_p = np.zeros(NBINSR, dtype=complex)
    num_m = np.zeros(NBINSR, dtype=complex)
    norm = np.zeros(NBINSR)
    npair = np.zeros(NBINSR)
    ell = cat.tracer_1 + 1j*cat.tracer_2
    for i, sel, rbin, wpair, rel1, rel2, sep in _pairs(cat.pos1, cat.pos2, cat.weight):
        phase4 = ((rel1 - 1j*rel2)/sep)**4
        cont_p = wpair*ell[i]*np.conj(ell[sel])
        cont_m = wpair*ell[i]*ell[sel]*phase4
        np.add.at(num_p, rbin, cont_p)
        np.add.at(num_m, rbin, cont_m)
        np.add.at(norm, rbin, wpair)
        np.add.at(npair, rbin, 1.)
    xip = np.divide(num_p, norm, out=np.zeros_like(num_p), where=norm > 0)
    xim = np.divide(num_m, norm, out=np.zeros_like(num_m), where=norm > 0)
    return xip, xim, norm, npair


def _bruteforce_nn(cat):
    """Direct weighted pair counts over all ordered pairs."""
    npair = np.zeros(NBINSR)
    for i, sel, rbin, wpair, rel1, rel2, sep in _pairs(cat.pos1, cat.pos2, cat.weight):
        np.add.at(npair, rbin, wpair)
    return npair


def _bruteforce_ng(cat_source, cat_lens):
    """Direct sum for the tangential shear about every lens."""
    edges = _bin_edges()
    num = np.zeros(NBINSR, dtype=complex)
    norm = np.zeros(NBINSR)
    npair = np.zeros(NBINSR)
    ell = cat_source.tracer_1 + 1j*cat_source.tracer_2
    for i in range(cat_lens.ngal):
        rel1 = cat_source.pos1 - cat_lens.pos1[i]
        rel2 = cat_source.pos2 - cat_lens.pos2[i]
        sep = np.sqrt(rel1*rel1 + rel2*rel2)
        sel = np.where((sep >= MIN_SEP) & (sep < MAX_SEP))[0]
        if not len(sel):
            continue
        rbin = np.searchsorted(edges, sep[sel], side='right') - 1
        wpair = cat_lens.weight[i]*cat_source.weight[sel]
        phase2 = ((rel1[sel] - 1j*rel2[sel])/sep[sel])**2
        np.add.at(num, rbin, -wpair*ell[sel]*phase2)   # tangential basis
        np.add.at(norm, rbin, wpair)
        np.add.at(npair, rbin, 1.)
    xi = np.divide(num, norm, out=np.zeros_like(num), where=norm > 0)
    return xi, norm, npair


#################################
# 2PT ORPHEUS DISCRETE VS BRUTE #
#################################

# In this test we assert that the orpheus 2pt correlators in a discrete setting reproduce a
# brute-force estimator. This practically checks that the spatial hashing works as expected.

# Correlator setup for these tests --> Fully discrete
DISCRETE = dict(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                tree_resos=[0.], rmin_pixsize=RMIN_PIXSIZE, nthreads=NTHREADS)

def test_gg_matches_a_direct_pair_sum(shear_catalog):
    gg = GGCorrelation(**DISCRETE, shuffle_pix=1)
    gg.process(shear_catalog, dotomo=False, do_dc=True)
    xip, xim, norm, npair = _bruteforce_gg(shear_catalog)
    assert np.allclose(gg.npair[0], npair, rtol=0., atol=.5)
    assert np.allclose(gg.norm[0], norm, rtol=RTOL_EXACT, atol=0.)
    assert np.max(np.abs(gg.xip[0] - xip)) < RTOL_EXACT
    assert np.max(np.abs(gg.xim[0] - xim)) < RTOL_EXACT

def test_nn_matches_a_direct_pair_count(scalar_catalog):
    nn = NNCorrelation(**DISCRETE)
    nn.process(scalar_catalog, dotomo=False)
    npair = _bruteforce_nn(scalar_catalog)
    assert np.allclose(nn.npair[0], npair, rtol=RTOL_EXACT, atol=0.)

def test_ng_matches_a_direct_pair_sum(shear_catalog, scalar_catalog):
    ng = NGCorrelation(**DISCRETE, shuffle_pix=1)
    ng.process(shear_catalog, scalar_catalog, dotomo=False)
    xi, norm, npair = _bruteforce_ng(shear_catalog, scalar_catalog)
    assert np.allclose(ng.npair[0], npair, rtol=0., atol=.5)
    assert np.max(np.abs(ng.xi[0] - xi)) < RTOL_EXACT

##############
# MISC TESTS #
##############

# Here we collect a few small additional tests tied to 2pt stats.

# Make sure that double counting does not alter the E and B Modes but alters xi_x
def test_do_dc_only_affects_the_xi_cross_channel(shear_catalog):
    xis = []
    for do_dc in (False, True):
        gg = GGCorrelation(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                           tree_resos=TREE_RESOS, rmin_pixsize=RMIN_PIXSIZE,
                           shuffle_pix=0, nthreads=NTHREADS)
        gg.process(shear_catalog, dotomo=False, do_dc=do_dc)
        xis.append((gg.xip[0].copy(), gg.xim[0].copy(), gg.npair[0].copy()))
    assert np.allclose(xis[0][0].real, xis[1][0].real, rtol=RTOL_EXACT, atol=0.)
    assert np.allclose(xis[0][1], xis[1][1], rtol=RTOL_EXACT, atol=0.)
    assert np.allclose(xis[0][2], xis[1][2], rtol=RTOL_EXACT, atol=0.)
    assert np.max(np.abs(xis[1][0].imag)) < RTOL_EXACT*np.max(np.abs(xis[1][0].real))
    assert np.max(np.abs(xis[0][0].imag)) > 0.

# Make sure that increasing rmin_pixsize gives better results. For this we compare
# the measurement across the analytic field.
def test_doubletree_converges_with_the_pixel_to_separation_ratio(tangential_field):
    """The gridded bins carry a bin-slop error that falls off as (1/rmin_pixsize)^2."""
    cat_source, cat_lens, gamma_t = tangential_field
    deviations = []
    for rmin_pixsize in (2, 4, 8, 16):
        ng = NGCorrelation(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                           tree_resos=[0., .5, 1., 2.], rmin_pixsize=rmin_pixsize,
                           shuffle_pix=1, nthreads=NTHREADS)
        ng.process(cat_source, cat_lens, dotomo=False)
        deviations.append(np.max(np.abs(ng.xi[0].real - gamma_t)))
    assert np.all(np.diff(deviations) < 0.), deviations
    assert deviations[-1] < 1e-3

# The global-rotation behaviour of xi_pm is covered by
# test_fast_invariants.test_follows_the_global_shear_phase[GGCorrelation], whose
# PHASE_LEGS entry (0, 2) states exactly that xi_p is invariant and xi_m picks up
# exp(2i*chi). NG is in that same family as PHASE_LEGS['NGCorrelation'] = (1,).

# Make sure that we dont measure a significant correlation in a poisson-sampled field.
# This overlaps with the clustered-field tests, but it is the only place in the fast tier
# that makes a numerical statement about LS; smoketest only checks finiteness ofnn.xi.
def test_nn_landy_szalay_vanishes_for_a_poisson_field():
    ngal = 8000
    def make(seed):
        pos1, pos2 = np.random.default_rng(seed).uniform(0., BOXSIZE, (2, ngal))
        return ScalarTracerCatalog(pos1=pos1, pos2=pos2, tracer=np.ones(ngal),
                                   weight=np.ones(ngal), geometry='flat2d')

    nn = NNCorrelation(min_sep=MIN_SEP, max_sep=MAX_SEP, nbinsr=NBINSR,
                       tree_resos=TREE_RESOS, rmin_pixsize=RMIN_PIXSIZE, nthreads=NTHREADS)
    nn.process(make(101), cat_random=make(202), dotomo=False)
    assert np.max(np.abs(nn.xi)) < .1
