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
# * Sample-variance integral against its direct sum over both varphi grids, a tensor of the
#   separable density against that density, a measured density relative to the suppression of the
#   pairs of the catalog, independent of the threads and of the mask, the radial stencil, and the
#   triplets and quadruplet density synthesised from their Fourier coefficients
# * Split of the tomographic covariance into its terms, and the cross bin pair independent of the
#   labels of its bins
# * Weights of the statistics that are linear in xi_pm

import warnings

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
NMAX = 3
SIGMA_EPS = .3


def _centers(edges):
    return np.sqrt(edges[1:]*edges[:-1])

def _fourier(ang, n_modes):
    """Real Fourier basis 1, cos, sin, ..., cos n, sin n of the angles ang, (len(ang), n_modes)."""
    ang = np.atleast_1d(ang)
    out = np.ones((len(ang), n_modes))
    for k in range(1, (n_modes-1)//2+1):
        out[:, 2*k-1], out[:, 2*k] = np.cos(k*ang), np.sin(k*ang)
    return out

def _xi(theta):
    return 1e-4*np.exp(-np.asarray(theta)/50.)


def _rotated_to_dec(ra, dec, dec0):
    """Positions rotated about the y-axis, which takes the equator at ra = 0 to Dec dec0."""
    a, d, t = np.radians(ra), np.radians(dec), np.radians(dec0)
    x, y, z = np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)
    x, z = np.cos(t)*x-np.sin(t)*z, np.sin(t)*x+np.cos(t)*z
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(z))


def _survey_kwargs(ra, dec, area, path_save, geometry):
    """Arguments of the discrete covariance instance on the catalog (ra, dec), measured with the
    density as it is."""
    return dict(
        theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
        q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES, q_count_min=0.,
        method_pairs='discrete', method_triplets='discrete', method_quadruplets='discrete',
        spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=NGAL/area, area=area,
        catalog=dict(ra=ra, dec=dec), path_save=path_save, geometry=geometry,
        nmaxs=NMAX, rmin_pixsize=20, tree_resos=[0.], verbosity=0, nthreads=NTHREADS)


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
    cov = GGCovariance(**_survey_kwargs(
        ra, dec, area, str(tmp_path_factory.mktemp("multiplets")),
        'flat2d' if request.param == "flat" else 'spherical'))
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

def _annulus_tangents(cov, edges):
    """Sum of the unit vectors from each galaxy towards the galaxies in each annulus around it, in
    its tangent plane, (ngal, nbins, dim): the cosine of the angle between two legs is the dot
    product of their unit vectors."""
    pos, radius = _positions(cov)
    r_edges = radius(edges)
    out = np.zeros((len(pos), len(edges)-1, pos.shape[1]))
    for i, nbrs in enumerate(cKDTree(pos).query_ball_point(pos, r_edges[-1])):
        d = pos[nbrs]-pos[i]
        b = np.searchsorted(r_edges, np.linalg.norm(d, axis=1))-1
        if pos.shape[1] == 3:
            d -= np.outer(d@pos[i], pos[i])
        ok = (b >= 0) & (b < len(edges)-1)
        np.add.at(out[i], b[ok], d[ok]/np.linalg.norm(d[ok], axis=1)[:, None])
    return out

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

def _radial_stencil(rc, t, cubic):
    """First node and weights of the radial interpolation of the kernel at t: none above the
    centres rc, the first below them, linear in t, or Catmull-Rom in log t away from the first and
    the last interval."""
    if t > rc[-1]:
        return 0, np.zeros(0)
    if t < rc[0]:
        return 0, np.array([1., 0.])
    i = min(np.searchsorted(rc, t, side='right')-1, len(rc)-2)
    if cubic and 0 < i < len(rc)-2:
        u = np.log(t/rc[i])/np.log(rc[i+1]/rc[i])
        return i-1, .5*np.array([-u**3+2.*u**2-u, 3.*u**3-5.*u**2+2., -3.*u**3+4.*u**2+u,
                                 u**3-u**2])
    w = (t-rc[i])/(rc[i+1]-rc[i])
    return i, np.array([1.-w, w])


def _exact_rdr(edges):
    """Exact annulus weight int r dr per bin."""
    return .5*np.diff(np.asarray(edges)**2)


def _knots(q, coef=1./24., inverse=False):
    """Knot values of a density from its bin means, m_i - coef (m_(i+1) - 2 m_i + m_(i-1)) along
    each radial axis but at its first and last bin, or the bin means from the knot values."""
    k = np.eye(q.shape[0])
    for i in range(1, len(k)-1):
        k[i, i-1:i+2] -= coef*np.array([1., -2., 1.])
    k = np.linalg.inv(k) if inverse else k
    return np.einsum('iI,jJ,kK,IJK...->ijk...', k, k, k, q)


############################################
# MEASURED COUNTS AGAINST BRUTE-FORCE ONES #
############################################

def test_pair_counts_are_the_ordered_pair_counts(survey):
    _, _, _, cov = survey
    brute = _annulus_counts(cov, THETA_EDGES).sum(axis=0)
    assert np.allclose(cov.pairs, brute, rtol=RTOL_EXACT, atol=0.)

# Without spl_suppression_ww, 'frompairs' takes the pair suppression of the catalog on the radial
# bins of its quadruplet density, as the tutorial measures it, and caches it
def test_frompairs_measures_the_suppression_of_the_catalog(survey, tmp_path):
    ra, dec, area, cov = survey
    kw = dict(_survey_kwargs(ra, dec, area, str(tmp_path), cov.geometry), method_pairs='analytic',
              method_triplets='analytic', method_quadruplets='frompairs', store_multiplets=True)
    measured, cached = GGCovariance(**kw), GGCovariance(**kw)
    tc = _centers(Q_THETA_EDGES)
    expected = GGCovariance.suppression_spline_from_pairs(
        _annulus_counts(cov, Q_THETA_EDGES).sum(axis=0), tc, Q_THETA_EDGES, area, NGAL/area)
    assert np.allclose(measured.spl_suppression_ww(tc), expected(tc), rtol=RTOL_EXACT, atol=0.)
    assert len(list(tmp_path.glob('suppression_*.npz'))) == 1
    assert np.array_equal(cached.spl_suppression_ww(tc), measured.spl_suppression_ww(tc))

def test_triplet_counts_match_brute_force(survey):
    _, _, _, cov = survey
    c = _annulus_counts(cov, THETA_EDGES)
    nbins = len(THETA_EDGES)-1
    brute = np.array([[_distinct_products(c, (a, b)) for b in range(nbins)] for a in range(nbins)])
    assert np.allclose(2.*np.pi*cov.triplets[..., 0], brute, rtol=RTOL_EXACT, atol=0.)
    # The coefficient of cos(phi) on two distinct bins: the sum of cos(phi) over the triplets / pi
    t = _annulus_tangents(cov, THETA_EDGES)
    assert np.isclose(np.pi*cov.triplets[0, 1, 1], np.sum(t[:, 0]*t[:, 1]), rtol=RTOL_EXACT,
                      atol=0.)

def test_quadruplet_density_matches_brute_force(survey):
    _, _, area, cov = survey
    c = _annulus_counts(cov, Q_THETA_EDGES)
    nbins = len(Q_THETA_EDGES)-1
    brute = np.array([[[_distinct_products(c, (a, b, d)) for d in range(nbins)]
                       for b in range(nbins)] for a in range(nbins)])
    # The knot values of the counts over the exact annuli, q = N_q/V_q, summed over both angles
    rdr = _exact_rdr(Q_THETA_EDGES)
    vol_r = 2.*np.pi*area*np.einsum('a,b,c->abc', rdr, rdr, rdr)
    density = (2.*np.pi)**2*cov.quadruplets[..., 0, 0]
    assert np.allclose(density, _knots(brute/vol_r), rtol=RTOL_EXACT, atol=0.)
    # The bin means of the coefficients of cos(phi_12) and of cos(phi_13) on three distinct bins,
    # twice the sums of cos(phi_12) and of cos(phi_13) over the quadruplets over (2 pi)^2 V_q
    t = _annulus_tangents(cov, Q_THETA_EDGES)
    means = (2.*np.pi)**2*_knots(cov.quadruplets, inverse=True)[0, 1, 2]
    cos12 = np.sum(np.sum(t[:, 0]*t[:, 1], axis=1)*c[:, 2])
    cos13 = np.sum(np.sum(t[:, 0]*t[:, 2], axis=1)*c[:, 1])
    assert np.allclose([means[1, 0], means[0, 1]], 2.*np.array([cos12, cos13])/vol_r[0, 1, 2],
                       rtol=RTOL_EXACT, atol=0.)

# The sparse radial bins of a measured density, whose idealised count per angular resolution cell
# (2 pi/(2 q_nmaxs+1))^2 is below q_count_min, take the density of the smallest aligned block of
# 2^l bins per axis that holds as many, the other bins keep theirs, on the bin means before their
# knot values. An instance that merges none reads the density as it was measured.
def test_sparse_bins_take_the_density_of_their_block(survey):
    ra, dec, area, cov = survey
    v = _exact_rdr(Q_THETA_EDGES)
    vol = np.einsum('i,j,k->ijk', v, v, v)
    cell_norm = 2.*np.pi*area*(2.*np.pi/cov.q_n_modes)**2
    ideal = (NGAL/area)**4*cell_norm*vol
    count_min = np.median(ideal)
    sparse = np.argwhere(ideal < count_min)
    assert 0 < len(sparse) < ideal.size
    kw = _survey_kwargs(ra, dec, area, cov.path_save, cov.geometry)
    merged = GGCovariance(**dict(kw, q_count_min=count_min))
    q, nr = _knots(cov.quadruplets, inverse=True), len(v)
    q_mean = q[..., 0, 0]
    ref = q.copy()
    for b in sparse:
        size = 2
        while True:
            block = tuple(slice(i//size*size, min(i//size*size+size, nr)) for i in b)
            if cell_norm*np.sum(vol[block]*q_mean[block]) >= count_min or size >= nr:
                break
            size *= 2
        ref[tuple(b)] = np.einsum('ijk,ijkab->ab', vol[block], q[block])/vol[block].sum()
    # Coefficients that vanish by symmetry hold rounding noise only
    assert np.allclose(merged.quadruplets, _knots(ref), rtol=RTOL_EXACT,
                       atol=RTOL_EXACT*np.max(np.abs(ref)))
    assert np.array_equal(GGCovariance(**kw).quadruplets, cov.quadruplets)


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
    cov = GGCovariance(
        theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
        q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES, q_count_min=0.,
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
    assert np.allclose(2.*np.pi*cov.triplets[..., 0], brute, rtol=RTOL_EXACT, atol=0.)

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
    rdr = _exact_rdr(Q_THETA_EDGES)
    vol_r = 2.*np.pi*area*np.einsum('a,b,c->abc', rdr, rdr, rdr)
    density = (2.*np.pi)**2*cov.quadruplets[..., 0, 0]
    assert np.allclose(density, _knots(brute/vol_r), rtol=RTOL_EXACT, atol=0.)

# Scaling all weights by a constant leaves the weighted estimator and its covariance unchanged,
# which holds term by term once the shared galaxies carry their squared weights.
def test_weighted_covariance_terms_are_invariant_under_a_weight_scale(tmp_path):
    rng = np.random.default_rng(4)
    ra = rng.uniform(0., PATCH_DEG, NGAL)
    sdec = np.sin(np.radians(.5*PATCH_DEG))
    dec = np.degrees(np.arcsin(rng.uniform(-sdec, sdec, NGAL)))
    w = rng.uniform(.5, 1.5, NGAL)
    area = np.radians(PATCH_DEG)*2.*sdec*(180.*60./np.pi)**2
    terms = []
    for scale in (1., 3.):
        cov = GGCovariance(
            theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
            q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
            method_pairs='discrete', method_triplets='discrete', method_quadruplets='discrete',
            spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=NGAL/area, area=area,
            catalog=dict(ra=ra, dec=dec, weight=scale*w), path_save=str(tmp_path), nmaxs=NMAX,
            rmin_pixsize=20, tree_resos=[0.], verbosity=0,
            phi_grid=np.geomspace(.1, 20., 16),
            varphi1_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False),
            varphi2_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False), nthreads=NTHREADS)
        terms.append(cov.get_cov_terms())
    for block in ('pp', 'mm', 'pm'):
        for term in ('D', 'mix', 'sva'):
            assert np.allclose(terms[1][block][term], terms[0][block][term], rtol=RTOL_EXACT,
                               atol=1e-12*np.max(np.abs(terms[0][block][term])))


#####################
# STORED MULTIPLETS #
#####################

# The files of store_multiplets carry the mean bin centres of the measurement and pairs_wsq,
# without which the mixed term and the shape noise change at the per cent level, and the density of
# the catalog, on which the sparse bins of the quadruplet density are merged as for a cache hit.
# Half of its bins are sparse here.
def test_stored_multiplets_reproduce_the_measured_covariance(tmp_path):
    rng = np.random.default_rng(5)
    ra = rng.uniform(0., PATCH_DEG, NGAL)
    sdec = np.sin(np.radians(.5*PATCH_DEG))
    dec = np.degrees(np.arcsin(rng.uniform(-sdec, sdec, NGAL)))
    w = rng.uniform(.5, 1.5, NGAL)
    area = np.radians(PATCH_DEG)*2.*sdec*(180.*60./np.pi)**2
    v = _exact_rdr(Q_THETA_EDGES)
    cell_norm = 2.*np.pi*area*(2.*np.pi/(2*NMAX+1))**2
    ideal = (np.sum(w)/area)**4*cell_norm*np.einsum('i,j,k->ijk', v, v, v)
    kw = dict(theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
              q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
              q_count_min=np.median(ideal), triplet_oversample=2,
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
    terms_m, terms_s = measured.get_cov_terms(), stored.get_cov_terms()
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
    ratio = cov.triplets[..., 0]/cov.triplets_analytic()[..., 0]
    assert np.allclose(ratio, exp_trip, rtol=2.*RTOL_LOOSE, atol=0.)

    # The idealised quadruplet density is constant, nbar^4, and the measured one is normalised by
    # the exact annuli already
    qtc, qbins = cov.q_theta_centers, len(Q_THETA_EDGES)-1
    ratio = cov.quadruplets[..., 0, 0]/cov.nbar**4
    exp_quad = np.array([[[_inside_fraction([qtc[a], qtc[b], qtc[d]])
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
    cov.getD()
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
# have to give the same mixed term. From 0.5 arcmin with the 4096 angular nodes of the mixed
# term it reaches 4e-4 arcmin, below the default phi_grid.
def _xi_with_knots():
    theta = np.geomspace(1e-4, 1e3, 4000)
    return interp1d(theta, _xi(theta))

@pytest.mark.parametrize("qterm", ['getqpp', 'getqmm', 'getqpm'])
@pytest.mark.parametrize("edges", [THETA_EDGES, np.geomspace(.5, 8., 5)])
def test_idealised_mixed_term_includes_short_triplet_sides(qterm, edges):
    kw = dict(theta_centers=_centers(edges), theta_edges=edges, sigma_eps=SIGMA_EPS,
              nbar=10./60.**2, area=100.*60.**2, nthreads=NTHREADS)
    plain = getattr(GGCovariance(spline_xip=_xi, spline_xim=_xi, **kw), qterm)()
    knots = getattr(GGCovariance(spline_xip=_xi_with_knots(), spline_xim=_xi_with_knots(),
                                 **kw), qterm)()
    for combo, block in knots.items():
        assert np.max(np.abs(plain[combo]-block)) < RTOL_SHARP*np.max(np.abs(block))

@pytest.mark.parametrize("edges", [THETA_EDGES, np.geomspace(.5, 8., 5)])
def test_continuous_mixed_term_includes_short_triplet_sides(edges):
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
        cov._qterms_continuous_tomo_c(nr_target=128, nphi_target=128)
    for qterm in ('qpp_tomo', 'qmm_tomo', 'qpm_tomo'):
        for combo, block in getattr(knots, qterm).items():
            assert (np.max(np.abs(getattr(plain, qterm)[combo]-block))
                    < RTOL_SHARP*np.max(np.abs(block)))

# The kernels look xi up on log-uniform knots, so knots that are not are replaced by log-uniform
# ones within them, and both give the same mixed and sample-variance terms
def test_xi_on_linear_knots():
    th = np.linspace(1e-3, 300., 30000)
    kw = dict(theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES, sigma_eps=SIGMA_EPS,
              nbar=10./60.**2, area=100.*60.**2, phi_grid=np.geomspace(.1, 100., 16),
              varphi1_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False),
              varphi2_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False), nthreads=NTHREADS)
    linear, log = (GGCovarianceNoTomo(spline_xip=spl, spline_xim=spl, **kw)
                   for spl in (interp1d(th, _xi(th)), _xi_with_knots()))
    for a, b in zip([linear.getqpp(), linear.getqpm(), *linear.get_all_rterms()],
                    [log.getqpp(), log.getqpm(), *log.get_all_rterms()]):
        assert np.max(np.abs(a-b)) < RTOL_SHARP*np.max(np.abs(b))


############################
# SAMPLE-VARIANCE INTEGRAL #
############################

# The r-terms collapse their varphi sums onto the knots of the quadruplet density q. On a small
# grid they equal the direct sum of xi_a q xi_b times the angular weight over phi, the bin
# centres and both varphi grids, with q built point by point from its real Fourier coefficients
# and the radial stencils of the kernel. Checked for a
# tensor, cubic and linear in its radial axes, a constant and a separable q, and a tensor with a
# suppression, which it ignores, with phi and the separations reaching beyond the q grid. The
# varphi grids are symmetric under v -> -v, where the blocks of q that mix the parities of its two
# angles, which the kernel skips, sum to zero.
@pytest.mark.parametrize("q_kind", ["tensor", "tensor_linear", "constant", "separable",
                                    "tensor_separable"])
def test_rterms_equal_the_direct_varphi_sums(q_kind):
    edges, q_edges = np.geomspace(2., 40., 4), np.geomspace(3., 60., 6)
    th = np.geomspace(.01, 1e4, 2000)
    xi = {'p': interp1d(th, _xi(th), bounds_error=False, fill_value=0.),
          'm': interp1d(th, .5*(1.-th/30.)*_xi(th), bounds_error=False, fill_value=0.)}
    def sup(r): return 1.-.5*np.exp(-np.asarray(r)/10.)
    kw = dict(theta_centers=_centers(edges), theta_edges=edges, spline_xip=xi['p'],
              spline_xim=xi['m'], sigma_eps=SIGMA_EPS, nbar=10./60.**2, area=100.*60.**2,
              phi_grid=np.geomspace(.5, 200., 6),
              varphi1_grid=np.linspace(0., 2.*np.pi, 12, endpoint=False),
              varphi2_grid=np.linspace(0., 2.*np.pi, 10, endpoint=False), nthreads=NTHREADS)
    if q_kind.startswith('tensor'):
        coef = .3*np.random.default_rng(3).standard_normal((5,)*3+(7,)*2)
        coef[..., 0, 0] += 2.
        kw.update(quadruplets=coef, q_theta_centers=_centers(q_edges), q_theta_edges=q_edges,
                  method_quadruplets='discrete',
                  q_interpolation='linear' if q_kind == 'tensor_linear' else 'cubic')
    if q_kind.endswith('separable'):
        kw['spl_suppression_ww'] = sup
    cov = GGCovarianceNoTomo(**kw)
    rterms = cov._quadruplet_density_terms_c()

    # q on the (varphi1, varphi2) grid: the separable density, or the tensor, whose slice at
    # theta1 = phi is zero above its grid and else interpolated radially as the kernel does
    cubic = int(q_kind != 'tensor_linear')
    def q_grid(phi, t1, t2, psi_b, v1):
        if not q_kind.startswith('tensor'):
            return cov._q_const*(sup(phi)*sup(t1)*sup(t2) if q_kind == 'separable' else 1.)
        rc, n_modes = cov.q_theta_centers, cov.q_n_modes
        if phi >= rc[-1]:
            return 0.
        first, weights = _radial_stencil(rc, phi, cubic)
        q_slice = np.tensordot(weights, cov.quadruplets[first:first+len(weights)], axes=(0, 0))
        first, weights = _radial_stencil(rc, t1, cubic)
        q_slice = np.tensordot(weights, q_slice[first:first+len(weights)], axes=(0, 0))
        out = np.zeros((len(v1), len(psi_b)))
        b1, b2 = _fourier(v1, n_modes), _fourier(np.angle(psi_b), n_modes)
        for j, pb in enumerate(psi_b):
            first, weights = _radial_stencil(rc, abs(pb), cubic)
            if len(weights):
                coef = np.tensordot(weights, q_slice[first:first+len(weights)], axes=(0, 0))
                out[:, j] = b1 @ coef @ b2[j]
        return out

    weights = {'rp0': lambda v1, v2, va, vb: 1., 'rp1': lambda v1, v2, va, vb: np.cos(4.*(va-vb)),
               'rm0': lambda v1, v2, va, vb: np.cos(4.*(v1+v2-va-vb)),
               'rm1': lambda v1, v2, va, vb: np.cos(4.*(v1-v2)),
               'rpm': lambda v1, v2, va, vb: 2.*np.cos(4.*(v2-vb))}
    tc, tdt = cov.theta_centers, .5*np.diff(edges**2)
    phi, v1, v2 = cov.phi_grid, cov.varphi1_grid, cov.varphi2_grid
    phi_w = .5*(np.diff(phi, prepend=phi[0])+np.diff(phi, append=phi[-1]))
    direct = {name: np.zeros((cov.n_r, cov.n_r)) for name in cov.RTERMS}
    for p, pw in zip(phi, phi_w):
        for i1 in range(cov.n_r):
            psi_a = p-tc[i1]*np.exp(1j*v1)
            for i2 in range(cov.n_r):
                psi_b = p+tc[i2]*np.exp(1j*v2)
                q = q_grid(p, tc[i1], tc[i2], psi_b, v1)
                for name in cov.RTERMS:
                    xi_a, xi_b = (xi[pm] for pm in cov.RTERM_XI[name])
                    w = weights[name](v1[:, None], v2[None, :], np.angle(psi_a)[:, None],
                                      np.angle(psi_b)[None, :])
                    s = np.sum(xi_a(np.abs(psi_a))[:, None]*q*xi_b(np.abs(psi_b))[None, :]*w)
                    direct[name][i1, i2] += p*pw*tdt[i1]*tdt[i2]*s*(2.*np.pi)**2/(len(v1)*len(v2))
    pre = 2.*np.pi*cov.area/np.outer(cov.pairs, cov.pairs)
    for name, rterm in zip(cov.RTERMS, rterms):
        ref = pre*(.5*(direct[name]+direct[name].T) if name in cov.RTERMS_SYM else direct[name])
        assert np.max(np.abs(rterm-ref)) < RTOL_EXACT*np.max(np.abs(ref)), name


# The extrapolation in the radial bin width is (4 q - q_2)/3 of the knot values of the bin means q
# and of q_2, the same counts on pairs of merged radial bins at the means of their centers, with q_2
# interpolated to the bins of q as the integral interpolates and held beyond the merged centers. A
# density constant along the radial axes is extrapolated to itself, with the same r-terms. The
# extrapolation needs the bin means of a measured density on an even number of bins. A tensor read
# from a file is shared by the instances with the same interpolation and extrapolation.
@pytest.mark.parametrize("q_interpolation", ["linear", "cubic"])
def test_density_extrapolated_in_the_radial_bin_width(q_interpolation, tmp_path):
    edges, q_edges = np.geomspace(2., 40., 4), np.geomspace(.3, 400., 9)
    th = np.geomspace(.01, 1e4, 2000)
    kw = dict(theta_centers=_centers(edges), theta_edges=edges,
              spline_xip=interp1d(th, _xi(th), bounds_error=False, fill_value=0.),
              spline_xim=interp1d(th, .5*(1.-th/30.)*_xi(th), bounds_error=False, fill_value=0.),
              sigma_eps=SIGMA_EPS, nbar=10./60.**2, area=100.*60.**2,
              q_theta_centers=_centers(q_edges), q_theta_edges=q_edges,
              method_quadruplets='discrete', q_interpolation=q_interpolation,
              phi_grid=np.geomspace(.5, 100., 8),
              varphi1_grid=np.linspace(0., 2.*np.pi, 12, endpoint=False),
              varphi2_grid=(np.arange(10)+.5)*np.pi/5., nthreads=NTHREADS)
    def npz(q, name):
        np.savez(tmp_path/name, quadruplets=q, bin_centers_mean=_centers(q_edges))
        return str(tmp_path/name)
    q = 1.+np.random.default_rng(6).random((8,)*3+(7,)*2)
    q_file, const_file = npz(q, 'q.npz'), npz(np.array(np.broadcast_to(q[0, 0, 0], q.shape)),
                                              'const.npz')
    coef, cubic = (1./24., True) if q_interpolation == 'cubic' else (1./8., False)
    v, c = _exact_rdr(q_edges), _centers(q_edges)
    merge = np.zeros((4, 8))
    for i in range(4):
        merge[i, 2*i:2*i+2] = v[2*i:2*i+2]/v[2*i:2*i+2].sum()
    c2 = merge@c
    stencil = np.zeros((8, 4))
    for i in range(8):
        first, weights = _radial_stencil(c2, min(c[i], c2[-1]), cubic)
        stencil[i, first:first+len(weights)] = weights
    q2 = _knots(np.einsum('Ii,Jj,Kk,ijk...->IJK...', merge, merge, merge, q), coef)
    ref = (4.*_knots(q, coef)-np.einsum('iI,jJ,kK,IJK...->ijk...', stencil, stencil, stencil,
                                        q2))/3.
    extrapolated = GGCovarianceNoTomo(**dict(kw, quadruplets=q_file, q_richardson=True))
    assert np.allclose(extrapolated.quadruplets, ref, rtol=RTOL_EXACT, atol=0.)
    plain = GGCovarianceNoTomo(**dict(kw, quadruplets=const_file))
    extrapolated = GGCovarianceNoTomo(**dict(kw, quadruplets=const_file, q_richardson=True))
    assert np.allclose(extrapolated.quadruplets, plain.quadruplets, rtol=RTOL_EXACT, atol=0.)
    for name, a, b in zip(plain.RTERMS, plain._quadruplet_density_terms_c(),
                          extrapolated._quadruplet_density_terms_c()):
        assert np.max(np.abs(a-b)) < RTOL_EXACT*np.max(np.abs(a)), name
    for setting in (dict(quadruplets=npz(q[:7, :7, :7], 'odd.npz'), q_theta_edges=q_edges[:8],
                         q_theta_centers=c[:7]),
                    dict(quadruplets=q), dict(method_quadruplets='analytic')):
        with pytest.raises(ValueError):
            GGCovarianceNoTomo(**dict(kw, q_richardson=True, **setting))
    with pytest.raises(ValueError):
        GGCovarianceNoTomo(**dict(kw, quadruplets=q, q_interpolation='spline'))
    a, b = (GGCovarianceNoTomo(**dict(kw, quadruplets=q_file)) for _ in range(2))
    assert a.quadruplets is b.quadruplets and np.allclose(a.quadruplets, _knots(q, coef),
                                                          rtol=RTOL_EXACT, atol=0.)


# The integral holds a density tensor at its first bin below its grid and sets it to zero above,
# with a warning if the grid starts above the first xi_pm edge or ends before phi_grid.
def test_density_grid_short_of_the_integral_warns():
    edges = np.geomspace(2., 40., 4)
    kw = dict(theta_centers=_centers(edges), theta_edges=edges, spline_xip=_xi, spline_xim=_xi,
              sigma_eps=SIGMA_EPS, nbar=10./60.**2, area=100.*60.**2,
              quadruplets=np.ones((4,)*3+(7,)*2), method_quadruplets='discrete', nthreads=NTHREADS)
    def warns(q_edges, phi_end):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            GGCovarianceNoTomo(**dict(kw, q_theta_centers=_centers(q_edges), q_theta_edges=q_edges,
                                      phi_grid=np.geomspace(.5, phi_end, 8)))
        return any("holds the density" in str(w.message) for w in caught)
    assert not warns(np.geomspace(2., 60., 5), 60.)
    assert warns(np.geomspace(3., 60., 5), 60.)
    assert warns(np.geomspace(2., 60., 5), 100.)

# The quadratures of the sample-variance term need orientation grids that a reflection maps onto
# themselves, such as uniform ones from 0 or from half a step.
def test_orientation_grids_must_be_mirror_symmetric():
    kw = dict(theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES, spline_xip=_xi,
              spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=10./60.**2, area=100.*60.**2)
    v = np.linspace(0., 2.*np.pi, 10, endpoint=False)
    GGCovarianceNoTomo(varphi1_grid=v, varphi2_grid=v+np.pi/10., **kw)
    for v1, v2 in [(v, v+.1), (v+.1, v)]:
        with pytest.raises(ValueError, match="symmetric under varphi"):
            GGCovarianceNoTomo(varphi1_grid=v1, varphi2_grid=v2, **kw)


# A measured density is interpolated between the knot values whose interpolant has its bin means
# m, m_i - c (m_(i+1) - 2 m_i + m_(i-1)) along each radial axis but at its first and last bin, with
# c = 1/24 for the cubic and 1/8 for the linear interpolation. For bin means in log theta of a
# quadratic in log theta, the cubic knot values are its values at the bin centers. A passed density
# is interpolated between its values as they are.
def test_density_interpolates_between_knot_values(tmp_path):
    edges, q_edges = np.geomspace(2., 40., 4), np.geomspace(.3, 400., 6)
    kw = dict(theta_centers=_centers(edges), theta_edges=edges, spline_xip=_xi, spline_xim=_xi,
              sigma_eps=SIGMA_EPS, nbar=10./60.**2, area=100.*60.**2,
              q_theta_centers=_centers(q_edges), q_theta_edges=q_edges,
              method_quadruplets='discrete', nthreads=NTHREADS)
    q = 1.+np.random.default_rng(9).random((5,)*3+(5,)*2)
    np.savez(tmp_path/'q.npz', quadruplets=q, bin_centers_mean=_centers(q_edges))
    for q_interpolation, coef in (('cubic', 1./24.), ('linear', 1./8.)):
        cov = GGCovarianceNoTomo(**dict(kw, quadruplets=str(tmp_path/'q.npz'),
                                        q_interpolation=q_interpolation))
        assert np.allclose(cov.quadruplets, _knots(q, coef), rtol=RTOL_EXACT, atol=0.)
    passed = q.copy()
    assert np.array_equal(GGCovarianceNoTomo(**dict(kw, quadruplets=passed)).quadruplets, q)
    assert np.array_equal(passed, q)

    # Bin means and center values of f = 1 + x/2 + x^2/3 in x = log theta, separable in the three
    # radial axes
    x = np.log(q_edges)
    def f_int(x): return x + x**2/4. + x**3/9.
    mean = np.diff(f_int(x))/np.diff(x)
    xc = .5*(x[1:]+x[:-1])
    center = 1. + xc/2. + xc**2/3.
    separable = np.einsum('i,j,k,ab->ijkab', mean, mean, mean, np.ones((5, 5)))
    np.savez(tmp_path/'quadratic.npz', quadruplets=separable, bin_centers_mean=_centers(q_edges))
    knots = GGCovarianceNoTomo(**dict(kw, quadruplets=str(tmp_path/'quadratic.npz'))).quadruplets
    inner = slice(1, -1)
    assert np.allclose(knots[inner, inner, inner, 0, 0],
                       np.einsum('i,j,k->ijk', center, center, center)[inner, inner, inner],
                       rtol=RTOL_EXACT, atol=0.)


# The threads split the columns of the outputs at each phi node, several phi nodes at once for few
# columns, and every element is summed in phi order. The r-terms are thus the same to the last bit
# for any number of threads.
@pytest.mark.parametrize("q_interpolation", ["linear", "cubic"])
def test_rterms_do_not_depend_on_the_threads(q_interpolation):
    edges, q_edges = np.geomspace(2., 40., 7), np.geomspace(3., 60., 6)
    th = np.geomspace(.01, 1e4, 2000)
    cov = GGCovarianceNoTomo(
        theta_centers=_centers(edges), theta_edges=edges,
        spline_xip=interp1d(th, _xi(th), bounds_error=False, fill_value=0.),
        spline_xim=interp1d(th, .5*(1.-th/30.)*_xi(th), bounds_error=False, fill_value=0.),
        sigma_eps=SIGMA_EPS, nbar=10./60.**2, area=100.*60.**2,
        quadruplets=1.+np.random.default_rng(5).random((5,)*3+(7,)*2),
        q_theta_centers=_centers(q_edges), q_theta_edges=q_edges, method_quadruplets='discrete',
        q_interpolation=q_interpolation,
        phi_grid=np.geomspace(.5, 200., 9),
        varphi1_grid=np.linspace(0., 2.*np.pi, 12, endpoint=False),
        varphi2_grid=(np.arange(10)+.5)*np.pi/5., phi1_subgrid_frac=[-.5, .5],
        phi2_subgrid_frac=[-.5, 0., .5])
    full = cov._quadruplet_density_terms_c(nthreads=1)
    threads = cov._quadruplet_density_terms_c(nthreads=5)
    assert all(np.array_equal(a, b) for a, b in zip(full, threads))


# get_triplets and get_quadruplets return the Fourier coefficients of the tensors, or synthesise
# them at given angles, on all radial bins or a selection. An analytic density takes its
# coefficients from 2*q_nmaxs+1 uniform angles, at which they give back its values.
def test_get_multiplets_synthesise_the_fourier_coefficients():
    kw = dict(theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
              q_theta_centers=_centers(Q_THETA_EDGES), q_theta_edges=Q_THETA_EDGES,
              spline_xip=_xi, spline_xim=_xi, sigma_eps=SIGMA_EPS, nbar=10./60.**2,
              area=100.*60.**2, nthreads=NTHREADS)
    n_r, n_qr, n_modes = len(THETA_EDGES)-1, len(Q_THETA_EDGES)-1, 2*NMAX+1
    rng = np.random.default_rng(4)
    t_coef = rng.standard_normal((n_r, n_r, n_modes))
    q_coef = rng.standard_normal((n_qr,)*3 + (n_modes,)*2)
    cov = GGCovarianceNoTomo(method_triplets='discrete', triplets=t_coef,
                             method_quadruplets='discrete', quadruplets=q_coef, **kw)
    phis = np.array([-.4, 0., 1., 2.*np.pi+.3])
    B = _fourier(phis, n_modes)
    assert np.array_equal(cov.get_triplets(), t_coef)
    assert np.allclose(cov.get_triplets(phis), t_coef @ B.T, rtol=RTOL_EXACT, atol=0.)
    assert np.allclose(cov.get_triplets(phis, (1, 2)), B @ t_coef[1, 2], rtol=RTOL_EXACT,
                       atol=0.)
    q_phis = np.einsum('ijkab,xa,yb->ijkxy', q_coef, B, B)
    assert np.array_equal(cov.get_quadruplets(), q_coef)
    assert np.allclose(cov.get_quadruplets(phis), q_phis, rtol=RTOL_EXACT, atol=0.)
    assert np.allclose(cov.get_quadruplets(phis, (2, 0, 1)), q_phis[2, 0, 1], rtol=RTOL_EXACT,
                       atol=0.)
    cov = GGCovarianceNoTomo(method_quadruplets='frompairs', nmaxs=NMAX,
                             spl_suppression_ww=lambda t: np.exp(-np.asarray(t)/10.), **kw)
    ang = 2.*np.pi*np.arange(n_modes)/n_modes
    B = _fourier(ang, n_modes)
    q_ang = cov.get_quadruplets(ang)
    assert np.allclose(np.einsum('ijkab,xa,yb->ijkxy', cov.get_quadruplets(), B, B), q_ang,
                       rtol=RTOL_EXACT, atol=0.)
    assert np.allclose(cov.get_quadruplets(ang, (1, slice(None), 0)), q_ang[1, :, 0],
                       rtol=RTOL_EXACT, atol=0.)


# The kernel interpolates the density radially with the stencil of the direct sums above. Its
# cubic form is Catmull-Rom in log theta: between the second and the second-to-last centre it
# reproduces a density quadratic in log theta, which the linear one does not, and in the first and
# the last interval it is the linear one.
def test_cubic_stencil_reproduces_quadratics_in_log_theta():
    rc = np.geomspace(1., 50., 7)
    def f(t): return 1.+.3*np.log(t)-.2*np.log(t)**2
    def lookup(t, cubic):
        first, w = _radial_stencil(rc, t, cubic)
        return np.dot(w, f(rc[first:first+len(w)]))
    t = np.exp(np.random.default_rng(8).uniform(np.log(rc[1]), np.log(rc[-2]), 200))
    assert np.allclose([lookup(x, 1) for x in t], f(t), rtol=RTOL_EXACT, atol=0.)
    assert not np.allclose([lookup(x, 0) for x in t], f(t), rtol=1e-3, atol=0.)
    ends = np.r_[np.geomspace(rc[0], rc[1], 5)[:-1], np.geomspace(rc[-2], rc[-1], 5)]
    assert all(lookup(x, 1) == lookup(x, 0) for x in ends)


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
    terms = cov.get_cov_terms()
    for block in ('pp', 'mm', 'pm'):
        total = terms[block]['D']+terms[block]['mix']+terms[block]['sva']
        assert np.allclose(total, getattr(cov, 'cov_%s_tomo' % block), rtol=RTOL_EXACT, atol=0.)
    shapenoise = np.concatenate([cov.D_tomo[bp] for bp in cov.tomopairs_list])
    for block in ('pp', 'mm'):
        assert np.array_equal(terms[block]['D'], np.diag(shapenoise))
    assert not np.any(terms['pm']['D'])

# The integrals of the mixed terms do not depend on sigma_eps, so that an instance with another
# dispersion takes them over from one that holds them, and its mixed terms scale with sigma_eps^2.
@pytest.mark.parametrize("method_triplets", ['analytic', 'continuous'])
def test_mixed_integrals_are_shared_between_dispersions(method_triplets):
    covs = [GGCovariance(
        theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES,
        spline_xip=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)],
        spline_xim=[(0, 0, _xi), (0, 1, _xi), (1, 1, _xi)], method_triplets=method_triplets,
        sigma_eps=np.array([.3, .4])*fac, nbar=np.array([2., 5.])/60.**2, area=100.*60.**2,
        nthreads=NTHREADS) for fac in (1., 2.)]
    for cov in covs:
        if method_triplets == 'continuous':
            cov._qterms_continuous_tomo_c(nr_target=128, nphi_target=128)
        cov.getqpp(), cov.getqmm(), cov.getqpm()
    held = [cov._mixed_integrals for cov in covs]
    assert held[0].keys() == held[1].keys()
    assert all(held[1][key] is integral for key, integral in held[0].items())
    for qterm in ('qpp_tomo', 'qmm_tomo', 'qpm_tomo'):
        for combo, block in getattr(covs[0], qterm).items():
            assert np.array_equal(getattr(covs[1], qterm)[combo], 4.*block)

# The sample variance of two bins with xi^00 = xi^11 = xi and xi^01 = 0 follows from the two Wick
# contractions xi^ac xi^bd + xi^ad xi^bc of Cov(xi^ab, xi^cd): the single-bin sample variance for
# (00, 00) and (11, 11), half of it for (01, 01), and zero for (00, 11) and wherever one pair is
# (01) and the other not, for a constant and a tensor density.
@pytest.mark.parametrize("method_quadruplets", ['analytic', 'discrete'])
def test_sample_variance_of_the_cross_bins_follows_the_contractions(method_quadruplets):
    edges, q_edges = np.geomspace(2., 40., 4), np.geomspace(.3, 400., 5)
    th = np.geomspace(.01, 1e4, 2000)
    xip = interp1d(th, _xi(th), bounds_error=False, fill_value=0.)
    xim = interp1d(th, .5*(1.-th/30.)*_xi(th), bounds_error=False, fill_value=0.)
    zero = interp1d(th, 0.*th, bounds_error=False, fill_value=0.)
    kw = dict(theta_centers=_centers(edges), theta_edges=edges, sigma_eps=SIGMA_EPS,
              area=100.*60.**2, method_quadruplets=method_quadruplets,
              phi_grid=np.geomspace(.5, 100., 8),
              varphi1_grid=np.linspace(0., 2.*np.pi, 12, endpoint=False),
              varphi2_grid=(np.arange(10)+.5)*np.pi/5., nthreads=NTHREADS)
    if method_quadruplets == 'discrete':
        kw.update(quadruplets=1.+np.random.default_rng(4).random((4,)*3+(7,)*2),
                  q_theta_centers=_centers(q_edges), q_theta_edges=q_edges)
    one = GGCovarianceNoTomo(spline_xip=xip, spline_xim=xim, nbar=10./60.**2, **kw)
    single = {'pp': one.getrp0()+one.getrp1(), 'mm': one.getrm0()+one.getrm1(),
              'pm': one.getrpm()}
    tomo = GGCovariance(spline_xip=[(0, 0, xip), (0, 1, zero), (1, 1, xip)],
                        spline_xim=[(0, 0, xim), (0, 1, zero), (1, 1, xim)],
                        nbar=np.array([10., 10.])/60.**2, **kw).get_cov_terms()
    n = len(edges)-1
    for block in ('pp', 'mm', 'pm'):
        sva = tomo[block]['sva']
        def sub(i, j): return sva[i*n:(i+1)*n, j*n:(j+1)*n]
        for i, j, factor in [(0, 0, 1.), (2, 2, 1.), (1, 1, .5)]:
            assert np.allclose(sub(i, j), factor*single[block], rtol=RTOL_EXACT, atol=0.), block
        for i, j in [(0, 2), (2, 0), (0, 1), (1, 0), (1, 2), (2, 1)]:
            assert not np.any(sub(i, j)), block

# Swapping the labels of two bins, i.e. xi^00 <-> xi^11, nbar and sigma_eps, leaves the covariance
# of the cross pair (01, 01) unchanged. With different xi in the two bins this needs xi_+ and xi_-
# of rpm on either link of the contraction xi^00 xi^11 alike. The two varphi grids are the same, on
# which swapping the links of the other r-terms is exact.
@pytest.mark.parametrize("method_quadruplets", ['continuous', 'analytic'])
def test_cross_pair_covariance_does_not_depend_on_the_bin_labels(method_quadruplets):
    edges = np.geomspace(2., 40., 4)
    th = np.geomspace(.01, 1e4, 2000)
    def xi(amp, scale):
        return interp1d(th, amp*_xi(th)*np.exp(-th/scale), bounds_error=False, fill_value=0.)
    xip = {(0, 0): xi(1., 10.), (0, 1): xi(.5, 30.), (1, 1): xi(2., 100.)}
    xim = {(0, 0): xi(.3, 5.), (0, 1): xi(.2, 20.), (1, 1): xi(1., 60.)}
    nbar, sigma = np.array([2., 5.])/60.**2, np.array([.3, .4])
    varphi = np.linspace(0., 2.*np.pi, 16, endpoint=False)
    kw = dict(theta_centers=_centers(edges), theta_edges=edges, area=100.*60.**2,
              nbar_ref=7./60.**2, method_quadruplets=method_quadruplets,
              phi_grid=np.geomspace(.5, 100., 8), varphi1_grid=varphi, varphi2_grid=varphi,
              nthreads=NTHREADS)
    n = len(edges)-1
    cross = []
    for label in ([0, 1], [1, 0]):
        cov = GGCovariance(
            spline_xip=[(*sorted((label[a], label[b])), s) for (a, b), s in xip.items()],
            spline_xim=[(*sorted((label[a], label[b])), s) for (a, b), s in xim.items()],
            nbar=nbar[label], sigma_eps=sigma[label], **kw)
        i = cov.tomopairs_list.index((0, 1))
        terms = cov.get_cov_terms()
        cross.append({(block, term): terms[block][term][i*n:(i+1)*n, i*n:(i+1)*n]
                      for block in ('pp', 'mm', 'pm') for term in ('D', 'mix', 'sva')})
    for key, block in cross[0].items():
        assert np.allclose(cross[1][key], block, rtol=RTOL_EXACT, atol=0.), key

# In the continuum limit the tomographic class integrates its terms when they are first asked
# for, and for a single bin they equal the ones of GGCovarianceNoTomo. The getters of a single
# bin pair that it inherits refuse to run.
def test_tomographic_continuum_matches_the_single_bin_class():
    kw = dict(theta_centers=_centers(THETA_EDGES), theta_edges=THETA_EDGES, spline_xip=_xi,
              spline_xim=_xi, method_pairs='continuous', method_triplets='continuous',
              method_quadruplets='continuous', sigma_eps=SIGMA_EPS, nbar=10./60.**2,
              area=100.*60.**2, phi_grid=np.geomspace(.1, 100., 16),
              varphi1_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False),
              varphi2_grid=np.linspace(0., 2.*np.pi, 16, endpoint=False), nthreads=NTHREADS)
    tomo, single = GGCovariance(**kw), GGCovarianceNoTomo(**kw)
    # The mixed terms on coarse grids, as their continuum kernel is slow on the default ones. The
    # getters then return them as they are, while the r-terms are integrated when first needed.
    tomo._qterms_continuous_tomo_c(nr_target=128, nphi_target=128)
    terms = tomo.get_cov_terms()
    for block, qterm, rterms in [('pp', 'qpp', ('rp0', 'rp1')), ('mm', 'qmm', ('rm0', 'rm1')),
                                 ('pm', 'qpm', ('rpm',))]:
        assert getattr(tomo, 'get'+qterm)() is getattr(tomo, qterm+'_tomo')
        mix = getattr(single, 'get%s_continuous' % qterm)(nr_target=128, nphi_target=128)
        assert np.max(np.abs(terms[block]['mix']-mix)) < RTOL_SHARP*np.max(np.abs(mix))
        sva = sum(getattr(single, 'get'+rterm)() for rterm in rterms)
        assert np.max(np.abs(terms[block]['sva']-sva)) < RTOL_EXACT*np.max(np.abs(sva))
    assert np.allclose(np.diag(terms['pp']['D']), single.getD(), rtol=RTOL_EXACT, atol=0.)
    for getter in ('getrp0', 'getrpm_continuous', 'getqpp_continuous', 'getD_discrete'):
        with pytest.raises(NotImplementedError):
            getattr(tomo, getter)()


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
