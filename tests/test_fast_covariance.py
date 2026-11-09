# Here we check the multiplet counts that the covariance module measures from a catalog,
# the shape-noise term built from them, and how the covariance terms are put together. The
# counts are measured on a single tomographic bin of a few thousand uniform galaxies on a
# small patch near the equator: on the sphere, on the same patch rotated to Dec 60 deg, and as
# a flat catalog. The file is organised as:
#
# * Measured counts against brute-force counts: exact on the sphere at any Dec and on a flat
#   catalog. The largest separation of the catalog is bounded from above.
# * Weighted counts against brute-force counts, with the squared weight on the galaxies that
#   two pairs share, and the covariance terms unchanged by a common scale of the weights
# * Stored multiplets passed back as files reproduce the measured covariance
# * Measured counts against the idealised counts: they have to agree up to the footprint
#   edges, which we model exactly, and Poisson noise
# * Densities of the tomographic covariance: shape noise and reference density
# * Mixed term from triplets whose third side is shorter than the smallest bin
# * Split of the tomographic covariance into its terms
# * Weights of the statistics that are linear in xi_pm

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from orpheus import GGCovariance, GGCovarianceNoTomo

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


def _rotated_to_dec(ra, dec, dec0):
    """Positions rotated about the y-axis, which takes the equator at ra = 0 to Dec dec0."""
    a, d, t = np.radians(ra), np.radians(dec), np.radians(dec0)
    x, y, z = np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)
    x, z = np.cos(t)*x-np.sin(t)*z, np.sin(t)*x+np.cos(t)*z
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(z))


@pytest.fixture(scope="module", params=["sphere", "sphere_dec60", "flat"])
def survey(request, tmp_path_factory):
    """Uniform galaxies on a PATCH_DEG x PATCH_DEG patch and the discrete covariance
    instance measuring all multiplets on them."""
    rng = np.random.default_rng(2)
    ra = rng.uniform(0., PATCH_DEG, NGAL)
    sdec = np.sin(np.radians(.5*PATCH_DEG))
    dec = np.degrees(np.arcsin(rng.uniform(-sdec, sdec, NGAL)))
    area = np.radians(PATCH_DEG)*2.*sdec*(180.*60./np.pi)**2
    if request.param == "sphere_dec60":
        ra, dec = _rotated_to_dec(ra, dec, 60.)
    phic, phie = _phibins(NBINSPHI)
    qphic, qphie = _phibins(Q_NBINSPHI)
    cov = GGCovariance(
        theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
        phi_centers=phic, phi_edges=phie,
        q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
        q_phi_centers=qphic, q_phi_edges=qphie,
        method_pairs='discrete', method_triplets='discrete', method_quadruplets='discrete',
        spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=NGAL/area, area=area,
        catalog=dict(ra=ra, dec=dec), path_save=str(tmp_path_factory.mktemp("multiplets")),
        geometry='flat2d' if request.param == "flat" else 'spherical',
        nmaxs=NMAX, rmin_pixsize=20, tree_resos=[0.], verbosity=0, nthreads=NTHREADS)
    return ra, dec, area, cov


def _positions(cov):
    """Positions of the catalog of ``cov`` and the radii for given separations in arcmin: unit
    vectors and chords on the sphere, arcmin on a flat catalog."""
    ra, dec = cov.catalog['ra'], cov.catalog['dec']
    if cov.geometry == 'flat2d':
        return 60.*np.c_[ra, dec], lambda theta: np.asarray(theta, dtype=float)
    pos = np.c_[np.cos(np.radians(dec))*np.cos(np.radians(ra)),
                np.cos(np.radians(dec))*np.sin(np.radians(ra)), np.sin(np.radians(dec))]
    return pos, lambda theta: 2.*np.sin(np.radians(np.asarray(theta, dtype=float)/60.)/2.)

def _annulus_counts(cov, edges):
    """Number of galaxies in each annulus around each galaxy, (ngal, nbins)."""
    pos, radius = _positions(cov)
    tree = cKDTree(pos)
    cum = np.stack([tree.query_ball_point(pos, r, return_length=True) for r in radius(edges)],
                   axis=1)
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
    _, _, _, cov = survey
    brute = _annulus_counts(cov, THETA_EDGES).sum(axis=0)
    assert np.allclose(cov.pairs, brute, rtol=RTOL_EXACT, atol=0.)

def test_triplet_counts_match_brute_force(survey):
    _, _, _, cov = survey
    c = _annulus_counts(cov, THETA_EDGES)
    nbins = len(THETA_EDGES)-1
    brute = np.array([[_distinct_products(c, (a, b)) for b in range(nbins)] for a in range(nbins)])
    assert np.allclose(cov.triplets.sum(axis=-1), brute, rtol=RTOL_EXACT, atol=0.)

def test_quadruplet_density_matches_brute_force(survey):
    _, _, area, cov = survey
    c = _annulus_counts(cov, Q_THETA_EDGES)
    nbins = len(Q_THETA_EDGES)-1
    brute = np.array([[[_distinct_products(c, (a, b, d)) for d in range(nbins)]
                       for b in range(nbins)] for a in range(nbins)])
    # Undo the density normalisation q = N_q/V_q and sum over both angles
    rdr = cov.q_theta_centers*cov.q_dtheta
    vol_r = 2.*np.pi*area*np.einsum('a,b,c->abc', rdr, rdr, rdr)
    counts = vol_r*np.einsum('abcij,i,j->abc', cov.quadruplets, cov.q_dphi, cov.q_dphi)
    assert np.allclose(counts, brute, rtol=RTOL_EXACT, atol=0.)

# The bound widens the largest separation by the cells the footprint is binned on, at most 2%
def test_max_separation_bounds_the_largest_separation(survey):
    _, _, _, cov = survey
    pos, _ = _positions(cov)
    largest = np.max(pdist(pos))
    if cov.geometry == 'spherical':
        largest = 60.*np.degrees(2.*np.arcsin(largest/2.))
    bound = (cov.orpheus_instances['catalog'].max_separation()
             * (60. if cov.geometry == 'spherical' else 1.))
    assert largest <= bound <= 1.03*largest


############################################
# WEIGHTED COUNTS AGAINST BRUTE-FORCE ONES #
############################################

# The two pairs of the mixed term share a galaxy, which enters with its squared weight, and
# the shape noise squares both weights of a pair: the counts are sum w_i w_j and sum w_i^2 w_j^2
# for the pairs, sum w_i^2 w_j w_k for the triplets and sum w_i w_j w_k w_l for the quadruplets.
@pytest.fixture(scope="module", params=["sphere", "flat"])
def weighted_survey(request, tmp_path_factory):
    """The galaxies of ``survey`` with weights between 0.5 and 1.5."""
    rng = np.random.default_rng(2)
    ra = rng.uniform(0., PATCH_DEG, NGAL)
    sdec = np.sin(np.radians(.5*PATCH_DEG))
    dec = np.degrees(np.arcsin(rng.uniform(-sdec, sdec, NGAL)))
    w = rng.uniform(.5, 1.5, NGAL)
    area = np.radians(PATCH_DEG)*2.*sdec*(180.*60./np.pi)**2
    phic, phie = _phibins(NBINSPHI)
    qphic, qphie = _phibins(Q_NBINSPHI)
    cov = GGCovariance(
        theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
        phi_centers=phic, phi_edges=phie,
        q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
        q_phi_centers=qphic, q_phi_edges=qphie,
        method_pairs='discrete', method_triplets='discrete', method_quadruplets='discrete',
        spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=NGAL/area, area=area,
        catalog=dict(ra=ra, dec=dec, weight=w), path_save=str(tmp_path_factory.mktemp("weighted")),
        geometry='flat2d' if request.param == "flat" else 'spherical',
        nmaxs=NMAX, rmin_pixsize=20, tree_resos=[0.], verbosity=0, nthreads=NTHREADS)
    return w, area, cov

def _weighted_annulus_sums(cov, edges, w):
    """Sums of w_j, w_j^2 and w_j^3 over the neighbours j in each annulus around each galaxy,
    (3, ngal, nbins)."""
    pos, radius = _positions(cov)
    r = radius(edges)
    pairs = cKDTree(pos).query_pairs(r[-1], output_type='ndarray')
    i, j = np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]
    b = np.searchsorted(r, np.linalg.norm(pos[i]-pos[j], axis=1), side='left')-1
    ok = (b >= 0) & (b < len(edges)-1)
    sums = np.zeros((3, len(pos), len(edges)-1))
    for p in range(3):
        np.add.at(sums[p], (i[ok], b[ok]), w[j[ok]]**(p+1))
    return sums

def test_weighted_pairs_match_brute_force(weighted_survey):
    w, _, cov = weighted_survey
    s = _weighted_annulus_sums(cov, THETA_EDGES, w)
    assert np.allclose(cov.pairs, w@s[0], rtol=RTOL_EXACT, atol=0.)
    assert np.allclose(cov.pairs_wsq, w**2@s[1], rtol=RTOL_EXACT, atol=0.)

def test_weighted_triplets_carry_the_squared_base_weight(weighted_survey):
    w, _, cov = weighted_survey
    s = _weighted_annulus_sums(cov, THETA_EDGES, w)
    k = np.arange(len(THETA_EDGES)-1)
    brute = np.einsum('i,ia,ib->ab', w**2, s[0], s[0])
    brute[k, k] -= w**2@s[1]
    assert np.allclose(cov.triplets.sum(axis=-1), brute, rtol=RTOL_EXACT, atol=0.)

def test_weighted_quadruplet_density_matches_brute_force(weighted_survey):
    w, area, cov = weighted_survey
    s1, s2, s3 = _weighted_annulus_sums(cov, Q_THETA_EDGES, w)
    k = np.arange(len(Q_THETA_EDGES)-1)
    # Distinct legs by inclusion-exclusion over the legs that share an annulus
    brute = np.einsum('i,ia,ib,ic->abc', w, s1, s1, s1)
    shared = np.einsum('i,ia,ib->ab', w, s2, s1)
    brute[k, k, :] -= shared
    brute[k, :, k] -= shared
    brute[:, k, k] -= shared.T
    brute[k, k, k] += 2.*(w@s3)
    rdr = cov.q_theta_centers*cov.q_dtheta
    vol_r = 2.*np.pi*area*np.einsum('a,b,c->abc', rdr, rdr, rdr)
    counts = vol_r*np.einsum('abcij,i,j->abc', cov.quadruplets, cov.q_dphi, cov.q_dphi)
    assert np.allclose(counts, brute, rtol=RTOL_EXACT, atol=0.)

# Scaling all weights by a constant leaves the weighted estimator and its covariance unchanged,
# which holds term by term once the shared galaxies carry their squared weights.
def test_weighted_covariance_terms_are_invariant_under_a_weight_scale(tmp_path):
    rng = np.random.default_rng(4)
    ra = rng.uniform(0., PATCH_DEG, NGAL)
    sdec = np.sin(np.radians(.5*PATCH_DEG))
    dec = np.degrees(np.arcsin(rng.uniform(-sdec, sdec, NGAL)))
    w = rng.uniform(.5, 1.5, NGAL)
    area = np.radians(PATCH_DEG)*2.*sdec*(180.*60./np.pi)**2
    qphic, qphie = _phibins(Q_NBINSPHI)
    terms = []
    for scale in (1., 3.):
        cov = GGCovariance(
            theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
            q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
            q_phi_centers=qphic, q_phi_edges=qphie,
            method_pairs='discrete', method_triplets='discrete', method_quadruplets='discrete',
            spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=NGAL/area, area=area,
            catalog=dict(ra=ra, dec=dec, weight=scale*w), path_save=str(tmp_path), nmaxs=NMAX,
            rmin_pixsize=20, tree_resos=[0.], verbosity=0,
            phi_grid=np.geomspace(.1, 20., 16),
            varphi1_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False),
            varphi2_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False), nthreads=NTHREADS)
        terms.append(cov.get_cov_terms_tomo())
    for block in ('pp', 'mm', 'pm'):
        for term in ('D', 'mix', 'sva'):
            assert np.allclose(terms[1][block][term], terms[0][block][term], rtol=RTOL_EXACT,
                               atol=1e-12*np.max(np.abs(terms[0][block][term])))


#####################
# STORED MULTIPLETS #
#####################

# The files of store_multiplets carry the mean bin centres of the measurement and pairs_wsq,
# without which the mixed term and the shape noise change at the per cent level.
def test_stored_multiplets_reproduce_the_measured_covariance(tmp_path):
    rng = np.random.default_rng(5)
    ra = rng.uniform(0., PATCH_DEG, NGAL)
    sdec = np.sin(np.radians(.5*PATCH_DEG))
    dec = np.degrees(np.arcsin(rng.uniform(-sdec, sdec, NGAL)))
    w = rng.uniform(.5, 1.5, NGAL)
    area = np.radians(PATCH_DEG)*2.*sdec*(180.*60./np.pi)**2
    qphic, qphie = _phibins(Q_NBINSPHI)
    kw = dict(theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
              q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
              q_phi_centers=qphic, q_phi_edges=qphie, triplet_oversample=2,
              method_pairs='discrete', method_triplets='discrete', method_quadruplets='discrete',
              spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, area=area,
              phi_grid=np.geomspace(.1, 20., 16),
              varphi1_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False),
              varphi2_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False), nthreads=NTHREADS)
    measured = GGCovariance(catalog=dict(ra=ra, dec=dec, weight=w), path_save=str(tmp_path),
                            store_multiplets=True, nmaxs=NMAX, rmin_pixsize=20, tree_resos=[0.],
                            verbosity=0, **kw)
    files = {kind: str(next(tmp_path.glob(kind + '_*.npz')))
             for kind in ('pairs', 'triplets', 'quadruplets')}
    stored = GGCovariance(**files, **kw)
    for name in ('theta_centers', 'triplet_theta_centers', 'q_theta_centers', 'pairs_wsq'):
        assert np.array_equal(getattr(stored, name), getattr(measured, name))
    terms_m, terms_s = measured.get_cov_terms_tomo(), stored.get_cov_terms_tomo()
    for block in ('pp', 'mm', 'pm'):
        for term in ('D', 'mix', 'sva'):
            assert np.allclose(terms_s[block][term], terms_m[block][term], rtol=RTOL_EXACT,
                               atol=1e-12*np.max(np.abs(terms_m[block][term])))


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
        sigma_eps=sigma, nbar=nbar, area=area, nthreads=NTHREADS)
    cov.getD_tomo()
    ring = np.pi*area*np.diff(THETA_EDGES**2)
    for a, b in cov.tomopairs_list:
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
        sigma_eps=SIGMA_EPS, nbar=nbar, nbar_ref=nbar_ref, area=area, nthreads=NTHREADS)
    assert np.allclose(cov.pairs, np.pi*area*nbar_ref**2*np.diff(THETA_EDGES**2),
                       rtol=RTOL_EXACT, atol=0.)
    assert np.isclose(cov._q_const, nbar_ref**4, rtol=RTOL_EXACT, atol=0.)


#####################################
# MIXED TERM BELOW THE SMALLEST BIN #
#####################################

# xi given with its own knots is tabulated on them, a plain callable on the range the
# kernels query. The third side of a triplet reaches far below the smallest bin, so both
# have to give the same mixed term. From 0.5 arcmin with 128 angular bins it reaches 0.015
# arcmin, below the default phi_grid.
def _xi_with_knots():
    theta = np.geomspace(1e-3, 1e3, 4000)
    return interp1d(theta, _xi(theta))

@pytest.mark.parametrize("qterm", ['getqpp_tomo', 'getqmm_tomo', 'getqpm_tomo'])
@pytest.mark.parametrize("edges, nbinsphi", [(THETA_EDGES, 64), (np.geomspace(.5, 8., 5), 128)])
def test_idealised_mixed_term_includes_short_triplet_sides(qterm, edges, nbinsphi):
    phic, phie = _phibins(nbinsphi)
    kw = dict(theta_centers=_centers(edges), theta_edges=edges,
              phi_centers=phic, phi_edges=phie, sigma_eps=SIGMA_EPS,
              nbar=10./60.**2, area=100.*60.**2, nthreads=NTHREADS)
    plain = getattr(GGCovariance(spline_xip=_xi, spline_xim=_xi, **kw), qterm)()
    knots = getattr(GGCovariance(spline_xip=_xi_with_knots(), spline_xim=_xi_with_knots(),
                                 **kw), qterm)()
    for combo, block in knots.items():
        assert np.max(np.abs(plain[combo]-block)) < RTOL_SHARP*np.max(np.abs(block))

@pytest.mark.parametrize("edges", [THETA_EDGES, np.geomspace(.5, 8., 5)])
def test_continuous_mixed_term_includes_short_triplet_sides(edges):
    # Coarse grids for the r-terms, which get_continuous_tomo_c evaluates as well
    kw = dict(theta_centers=_centers(edges), theta_edges=edges,
              method_pairs='continuous', method_triplets='continuous',
              method_quadruplets='continuous', sigma_eps=SIGMA_EPS,
              nbar=10./60.**2, area=100.*60.**2, phi_grid=np.geomspace(.05, 100., 32),
              varphi1_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False),
              varphi2_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False), nthreads=NTHREADS)
    plain = GGCovarianceNoTomo(spline_xip=_xi, spline_xim=_xi, **kw)
    knots = GGCovarianceNoTomo(spline_xip=_xi_with_knots(), spline_xim=_xi_with_knots(), **kw)
    for qterm in ('qpp', 'qmm', 'qpm'):
        qplain = getattr(plain, 'get%s_continuous' % qterm)(nr_target=128, nphi_target=128)
        qknots = getattr(knots, 'get%s_continuous' % qterm)(nr_target=128, nphi_target=128)
        assert np.max(np.abs(qplain-qknots)) < RTOL_SHARP*np.max(np.abs(qknots))
    # The tomographic kernel tabulates xi for the mixed terms on its own
    plain = GGCovariance(spline_xip=_xi, spline_xim=_xi, **kw)
    knots = GGCovariance(spline_xip=_xi_with_knots(), spline_xim=_xi_with_knots(), **kw)
    for cov in (plain, knots):
        cov.get_continuous_tomo_c(nr_target=128, nphi_target=128, dlnt_target=.2)
    for qterm in ('qpp_tomo', 'qmm_tomo', 'qpm_tomo'):
        for combo, block in getattr(knots, qterm).items():
            assert (np.max(np.abs(getattr(plain, qterm)[combo]-block))
                    < RTOL_SHARP*np.max(np.abs(block)))


#######################################
# SPLIT OF THE TOMOGRAPHIC COVARIANCE #
#######################################

# The shape-noise, mixed and sample-variance terms sum to each block, and the shape
# noise only sits on the diagonal of the auto bin-pair blocks of Cov_pp and Cov_mm.
def test_covariance_terms_sum_to_the_tomographic_blocks():
    cov = GGCovariance(
        theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
        spline_xip=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)],
        spline_xim=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)],
        sigma_eps=np.array([.3, .4]), nbar=np.array([2., 5.])/60.**2, area=100.*60.**2,
        phi_grid=np.geomspace(.1, 100., 16),
        varphi1_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False),
        varphi2_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False), nthreads=NTHREADS)
    terms = cov.get_cov_terms_tomo()
    for block in ('pp', 'mm', 'pm'):
        total = terms[block]['D']+terms[block]['mix']+terms[block]['sva']
        assert np.allclose(total, getattr(cov, 'cov_%s_tomo' % block), rtol=RTOL_EXACT, atol=0.)
    shapenoise = np.concatenate([cov.D_tomo[bp] for bp in cov.tomopairs_list])
    for block in ('pp', 'mm'):
        assert np.array_equal(terms[block]['D'], np.diag(shapenoise))
    assert not np.any(terms['pm']['D'])


##########################################
# WEIGHTS OF THE LINEAR STATISTICS OF XI #
##########################################

# Map2 sums theta dtheta T_+-(theta/R) xi_+-(theta) / (2 R^2) over the xi_pm bins (S+02 Eq. 41,
# C+02 filter), and its B-mode flips the sign of the xi_- weights.
@pytest.mark.parametrize("edges", [THETA_EDGES, np.geomspace(.5, 250., 9)])
def test_map2_weights_sum_over_the_bin_widths(edges):
    cov = GGCovarianceNoTomo(theta_centers=_centers(edges), theta_edges=edges,
                             spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS,
                             nbar=10./60.**2, area=100.*60.**2, nthreads=NTHREADS)
    cov.cov_xipm = np.eye(2*cov.n_r)
    radii = np.array([1., 4., 20.])
    _, (W_E, W_B) = cov.cov_of_statistic('Map2', radii=radii, return_weights=True)
    x = _centers(edges)/radii[:, None]
    meas = _centers(edges)*np.diff(edges)/(2.*radii[:, None]**2)
    w_p = meas*(x**4-16.*x**2+32.)*np.exp(-x**2/4.)/128.
    w_m = meas*x**4*np.exp(-x**2/4.)/128.
    assert np.allclose(W_E, np.hstack([w_p, w_m]), rtol=RTOL_EXACT, atol=0.)
    assert np.allclose(W_B, np.hstack([w_p, -w_m]), rtol=RTOL_EXACT, atol=0.)
