# Here we check the multiplet counts that the covariance module measures from a catalog,
# and the shape-noise term built from them. The setup is a single tomographic bin of a
# few thousand uniform galaxies on a small patch near the equator, split into two patches
# so that the per-patch measurements are combined as for a survey. The file is organised as:
#
# * Measured counts against brute-force counts: exact up to the few multiplets whose
#   bin changes under the patch projection
# * Measured counts against the idealised counts: they have to agree up to the footprint
#   edges, which we model exactly, and Poisson noise
# * Densities of the tomographic covariance: shape noise and reference density

import numpy as np
import pytest
from scipy.spatial import cKDTree

from orpheus.covariance import GGCovariance

from conftest import NTHREADS, RTOL_EXACT, RTOL_LOOSE, RTOL_SHARP

NGAL = 3000
PATCH_DEG = 1.5
THETA_EDGES = np.geomspace(2., 8., 5)
Q_THETA_EDGES = np.geomspace(2., 8., 4)
NBINSPHI = 16
Q_NBINSPHI = 8
NMAX = 3
SIGMA_EPS = .3


def _centers(edges):
    return np.sqrt(edges[1:]*edges[:-1])

def _phibins(nbins):
    edges = np.linspace(0., 2.*np.pi, nbins+1)
    return .5*(edges[1:]+edges[:-1]), edges

def _xi(theta):
    return 1e-4*np.exp(-np.asarray(theta)/50.)


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    """Uniform galaxies on a PATCH_DEG x PATCH_DEG patch and the discrete covariance
    instance measuring all multiplets on them."""
    rng = np.random.default_rng(2)
    ra = rng.uniform(0., PATCH_DEG, NGAL)
    sdec = np.sin(np.radians(.5*PATCH_DEG))
    dec = np.degrees(np.arcsin(rng.uniform(-sdec, sdec, NGAL)))
    area = np.radians(PATCH_DEG)*2.*sdec*(180.*60./np.pi)**2
    phic, phie = _phibins(NBINSPHI)
    qphic, qphie = _phibins(Q_NBINSPHI)
    catalog = dict(ra=ra, dec=dec, path_save=str(tmp_path_factory.mktemp("multiplets")),
                   rmin_pixsize=20, tree_resos=[0.], npatches=2, nmaxs=NMAX, verbosity=0)
    cov = GGCovariance(
        theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
        phi_centers=phic, phi_edges=phie,
        q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
        q_phi_centers=qphic, q_phi_edges=qphie,
        method_pairs='discrete', method_triplets='discrete', method_quadruplets='discrete',
        spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=NGAL/area, area=area,
        catalog=catalog, n_workers=NTHREADS)
    return ra, dec, area, cov


def _annulus_counts(ra, dec, edges):
    """Number of galaxies in each annulus around each galaxy, (ngal, nbins)."""
    pos = np.c_[np.cos(np.radians(dec))*np.cos(np.radians(ra)),
                np.cos(np.radians(dec))*np.sin(np.radians(ra)), np.sin(np.radians(dec))]
    tree = cKDTree(pos)
    chords = 2.*np.sin(np.radians(np.asarray(edges)/60.)/2.)
    cum = np.stack([tree.query_ball_point(pos, r, return_length=True) for r in chords], axis=1)
    return np.diff(cum, axis=1).astype(float)

def _distinct_products(c, inds):
    """Sum over apices of the number of ordered tuples of distinct galaxies, one per
    annulus in ``inds``."""
    prod = np.ones(len(c))
    seen = {}
    for i in inds:
        prod *= c[:, i]-seen.get(i, 0)
        seen[i] = seen.get(i, 0)+1
    return prod.sum()

def _inside_fraction(radii, npos=160, nang=256):
    """Mean over apex positions of the product of the fractions of each circle around
    the apex that lies inside the patch. For uniform galaxies this is the ratio of the
    counts to the idealised ones once the angles between the legs are summed over."""
    side = 60.*PATCH_DEG
    x = (np.arange(npos)+.5)/npos*side
    ang = 2.*np.pi*(np.arange(nang)+.5)/nang
    frac = np.ones((npos, npos))
    for r in radii:
        px = x[:, None, None]+r*np.cos(ang)
        py = x[None, :, None]+r*np.sin(ang)
        inside = (px >= 0.) & (px <= side) & (py >= 0.) & (py <= side)
        frac = frac*inside.mean(axis=-1)
    return frac.mean()

def _exact_rdr(edges):
    """Exact annulus weight int r dr per bin."""
    return .5*np.diff(np.asarray(edges)**2)


############################################
# MEASURED COUNTS AGAINST BRUTE-FORCE ONES #
############################################

def test_pair_counts_are_the_ordered_pair_counts(survey):
    ra, dec, _, cov = survey
    brute = _annulus_counts(ra, dec, THETA_EDGES).sum(axis=0)
    assert np.allclose(cov.pairs, brute, rtol=RTOL_SHARP, atol=0.)

def test_triplet_counts_match_brute_force(survey):
    ra, dec, _, cov = survey
    c = _annulus_counts(ra, dec, THETA_EDGES)
    nbins = len(THETA_EDGES)-1
    brute = np.array([[_distinct_products(c, (a, b)) for b in range(nbins)] for a in range(nbins)])
    assert np.allclose(cov.triplets.sum(axis=-1), brute, rtol=RTOL_SHARP, atol=0.)

def test_quadruplet_density_matches_brute_force(survey):
    ra, dec, area, cov = survey
    c = _annulus_counts(ra, dec, Q_THETA_EDGES)
    nbins = len(Q_THETA_EDGES)-1
    brute = np.array([[[_distinct_products(c, (a, b, d)) for d in range(nbins)]
                       for b in range(nbins)] for a in range(nbins)])
    # Undo the density normalisation q = N_q/V_q and sum over both angles
    rdr = cov.q_theta_centers*cov.q_dtheta
    vol_r = 2.*np.pi*area*np.einsum('a,b,c->abc', rdr, rdr, rdr)
    counts = vol_r*np.einsum('abcij,i,j->abc', cov.quadruplets, cov.q_dphi, cov.q_dphi)
    assert np.allclose(counts, brute, rtol=RTOL_SHARP, atol=0.)


##############################################
# MEASURED COUNTS AGAINST THE IDEALISED ONES #
##############################################

# The idealised counts use bin centre times bin width as the annulus weight, so we
# compare per-leg against the exact weight and model the patch edges explicitly. What
# remains is the Poisson fluctuation of the neighbour counts, which every leg adds again,
# so the tolerance grows with the number of legs.
def test_counts_scale_as_the_idealised_counts(survey):
    _, _, _, cov = survey
    tc, nbins = cov.theta_centers, len(THETA_EDGES)-1
    leg = _exact_rdr(THETA_EDGES)/(tc*cov.dtheta)
    exp_pairs = np.array([_inside_fraction([t]) for t in tc])*leg
    assert np.allclose(cov.pairs/cov.pairs_analytic(), exp_pairs, rtol=RTOL_LOOSE, atol=0.)

    # Triplets live on the triplet grid, whose centres are their own measured means
    ttc = cov.triplet_theta_centers
    tleg = _exact_rdr(cov.triplet_theta_edges)/(ttc*cov.triplet_dtheta)
    exp_trip = np.array([[_inside_fraction([ttc[a], ttc[b]])*tleg[a]*tleg[b]
                          for b in range(nbins)] for a in range(nbins)])
    ratio = cov.triplets.sum(axis=-1)/cov.triplets_analytic().sum(axis=-1)
    assert np.allclose(ratio, exp_trip, rtol=2.*RTOL_LOOSE, atol=0.)

    # The idealised quadruplet density is constant, nbar^4
    qtc, qbins = cov.q_theta_centers, len(Q_THETA_EDGES)-1
    qleg = _exact_rdr(Q_THETA_EDGES)/(qtc*cov.q_dtheta)
    ratio = cov.quadruplets.mean(axis=(-1, -2))/cov.nbar**4
    exp_quad = np.array([[[_inside_fraction([qtc[a], qtc[b], qtc[d]])*qleg[a]*qleg[b]*qleg[d]
                           for d in range(qbins)] for b in range(qbins)] for a in range(qbins)])
    assert np.allclose(ratio, exp_quad, rtol=3.*RTOL_LOOSE, atol=0.)


#####################################
# DENSITIES OF THE TOMOGRAPHIC CASE #
#####################################

# Ordered pairs within a bin and pairs between two bins counted in both orders enter
# D = sigma_a^2 sigma_b^2 / N_p on the same footing (S+02 Eq. 27).
def test_tomographic_shape_noise_double_counts_cross_bin_pairs():
    nbar, sigma = np.array([2., 5.])/60.**2, np.array([.3, .4])
    area = 100.*60.**2
    tc = _centers(THETA_EDGES)
    cov = GGCovariance(
        theta_centers=tc, theta_edges=THETA_EDGES,
        spline_xip=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)],
        spline_xim=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)],
        sigma_eps=sigma, nbar=nbar, area=area, n_workers=NTHREADS)
    cov.getD_tomo()
    ring = 2.*np.pi*area*tc*np.diff(THETA_EDGES)
    for a, b in cov.bp_list:
        npairs = ring*nbar[a]*nbar[b]*(1. if a == b else 2.)
        assert np.allclose(cov.D_tomo[(a, b)], sigma[a]**2*sigma[b]**2/npairs,
                           rtol=RTOL_EXACT, atol=0.)

# The idealised multiplets, including the constant quadruplet density of 'frompairs', are
# built at the reference density from which the tomographic terms are rescaled.
@pytest.mark.parametrize("method_quadruplets", ['analytic', 'frompairs'])
def test_idealised_multiplets_live_at_the_reference_density(method_quadruplets):
    nbar, nbar_ref = np.array([2., 5.])/60.**2, 7./60.**2
    area = 100.*60.**2
    tc = _centers(THETA_EDGES)
    cov = GGCovariance(
        theta_centers=tc, theta_edges=THETA_EDGES,
        method_pairs='analytic', method_triplets='analytic', method_quadruplets=method_quadruplets,
        spl_suppression_ww=lambda t: np.ones_like(np.asarray(t, dtype=float)),
        spline_xip=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)],
        spline_xim=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)],
        sigma_eps=SIGMA_EPS, nbar=nbar, nbar_ref=nbar_ref, area=area, n_workers=NTHREADS)
    assert np.allclose(cov.pairs, 2.*np.pi*area*nbar_ref**2*tc*np.diff(THETA_EDGES),
                       rtol=RTOL_EXACT, atol=0.)
    assert np.isclose(cov._q_const, nbar_ref**4, rtol=RTOL_EXACT, atol=0.)
