# Here we collect shared fixtures and define the two-tier splits of the test suite
# * ``pytest``           runs fast tier --> Always triggered for CI
# * ``pytest --runslow`` runs all tests --> Only triggered for new releases
# The tier is indicated in the file name, test_fast_*.py and test_slow_*.py.

import glob
import os
import sys
from collections import namedtuple

import numpy as np
import pytest


## Before importing, keep an uncompiled checkout from shadowing the installed package
# Only `python -m pytest` from the repository root needs this, as it puts the working
# directory on `sys.path`. The `pytest` console script does not, so CI never hits it.
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not glob.glob(os.path.join(_repo, "orpheus", "orpheus_clib*.so")):
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _repo]


from orpheus.catalog import ScalarTracerCatalog, SpinTracerCatalog # noqa: E402
from orpheus.npcf_second import GGCorrelation, NGCorrelation, NNCorrelation # noqa: E402
from orpheus.npcf_third import GGGCorrelation, GNNCorrelation, NGGCorrelation, NNNCorrelation # noqa: E402
from orpheus.npcf_fourth import GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo, NNNNCorrelation_NoTomo # noqa: E402
from reference import AnalyticField # noqa: E402


################
# TEST TIERING #
################
# Only include slow-tier if --runslow set as flag
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the full-tier tests marked 'slow'")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full-tier test, only run with --runslow")

# Whether a test is for fast tier or slow tier is indicated in the filename via
# test_fast or test_slow. Set the --runslow flag if script contains slow tier tests.
def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="full tier, pass --runslow to include")
    for item in items:
        name = os.path.basename(str(item.fspath))
        if not name.startswith(("test_fast_", "test_slow_")):
            raise pytest.UsageError("%s carries no tier prefix; test modules must be "
                                    "named test_fast_*.py or test_slow_*.py"%name)
        if name.startswith("test_slow_"):
            item.add_marker(pytest.mark.slow)
            if not config.getoption("--runslow"):
                item.add_marker(skip_slow)

# Make sure that we run from the correct path
def pytest_sessionstart(session):
    import orpheus
    pkgdir = os.path.dirname(os.path.abspath(orpheus.__file__))
    if not glob.glob(os.path.join(pkgdir, "orpheus_clib*.so")):
        raise pytest.UsageError(
            "orpheus was imported from %s, which carries no compiled extension. "
            "This happens when the source tree shadows the installed package, "
            "typically from `python -m pytest` in the checkout root. "
            "Run `pytest` (not `python -m pytest`), or invoke it from elsewhere."%pkgdir)


####################
# TOLERANCE TIERS  #
####################
# All tests test on rtol only within a range win which the numerical effects due to the
# finite and discrete simulation setup are controlled. We further split the various tests
# in the following tolerance tiers:
# * Exact  --> Should be fulfilled to machine precision
#              Used by identities, parity arguments, ...
# * Sharp  --> We can assert a few-permille level convergence
#              Used to check integrated statistics for measured vs theo NPCF conversion
# * Tight  --> We can assert a sub-percent level convergence
#              Used for "easy" statistics such like 2pcfs
# * Normal --> We can assert a percent-level convergence
#              Used for higher-order correlation functions
# * Loose  --> We can only assert convergence to a few percent, but expect only such
#              a convergence on theoretical grounds. A better convergence criterion
#              would make the computation cost infeasible for testing
#              Used for integrated statistics 
# We further only consider the data points which are not affected by numerical noise. We 
# make sure to keep at least 0.5% of all data points for testing purposes.
RTOL_EXACT = 1e-9
RTOL_SHARP = 5e-3
RTOL_TIGHT = 1e-2
RTOL_NORMAL = 2e-2
RTOL_LOOSE = 5e-2
MIN_KEPT = .005


def kept_radii(theory, floor):
    """Aperture radii where the statistic is above ``floor`` times its own peak."""
    t = np.asarray(theory)
    return np.abs(t) > floor*np.max(np.abs(t))

# Compute deviation as ratio where we mask out points below some amplitude wrt
# the max. This is a sharper test than AnalyticField.deviation
def masked_ratio_deviation(measured, reference, floor=.05, kind='max'):
    m, r = np.asarray(measured).ravel(), np.asarray(reference).ravel()
    keep = np.abs(r) > floor*np.max(np.abs(r))
    assert keep.sum() > 0, "amplitude mask kept nothing"
    d = np.abs(m[keep]/r[keep] - 1.)
    return float(np.max(d) if kind == 'max' else np.sqrt(np.mean(d**2)))


# Window floor shared by the third- and fourth-order analytic tests.
# Do not consider ranges where the finite-field effects dominate strongly.
WINDOW_FLOOR = .05

# Default selector to only consider well-sampled configuration that carry enough
# signal to be compared with each other relatively, see sect 8.2 in notes
def kept_configurations(norm, theory, window, floor, pct, window_floor=WINDOW_FLOOR):
    """Well-sampled configurations carrying enough signal to compare relatively.

    Three cuts, see notes sect 8.2: footprint (``window``), occupancy (``norm`` above its
    ``pct`` percentile inside the footprint) and amplitude (``theory`` above ``floor`` times
    its in-footprint peak). ``norm`` is passed in already extracted and shaped like
    ``theory``, since each correlator stores it differently.
    """
    norm, theory = np.abs(np.asarray(norm)), np.asarray(theory)
    sel_window = window > window_floor
    sel_sampled = norm > np.percentile(norm[sel_window], pct)
    sel_ampl = np.abs(theory) > floor*np.max(np.abs(theory[sel_window]))
    return sel_window & sel_sampled & sel_ampl


def assert_amp_phase(measured, theory, rtol_amp, rtol_phase, tag=''):
    """Compare complex arrays by amplitude and phase separately."""
    m, t = np.asarray(measured), np.asarray(theory)
    amp = np.abs(np.abs(m)/np.abs(t) - 1.).max()
    phase = np.abs(np.angle(m/t)).max()
    assert amp < rtol_amp, '%s amplitude %.4g exceeds %.4g'%(tag, amp, rtol_amp)
    assert phase < rtol_phase, '%s phase %.4g rad exceeds %.4g'%(tag, phase, rtol_phase)


#########################
# CORRELATOR SELECTIONS #
#########################

# The full set of correlators, so that tests which make the same statement for each of them
# parametrise over one list instead of carrying a function per class.
# * nspin2 is the number of spin-2 legs in the correlator
# * legs says whether the correlator is purely scalar, purely polar or mixed
# * dotomo/srclens says how we call a tomographic computation for pure/mixed correlators
# When a new correlator is added to the package make sure to include it here
Correlator = namedtuple('Correlator', 'cls kwargs nspin2 order legs tomo')
CORRELATORS = [
    Correlator(NNCorrelation,          {},           0, 2, 'scalar', 'dotomo'),
    Correlator(NGCorrelation,          {},           1, 2, 'mixed',  'dotomo'),
    Correlator(GGCorrelation,          {},           2, 2, 'shear',  'dotomo'),
    Correlator(NNNCorrelation,         {},           0, 3, 'scalar', 'dotomo'),
    Correlator(GNNCorrelation,         {},           1, 3, 'mixed',  'srclens'),
    Correlator(NGGCorrelation,         {},           2, 3, 'mixed',  'srclens'),
    Correlator(GGGCorrelation,         {'n_cfs': 4}, 3, 3, 'shear',  'dotomo'),
    Correlator(NNNNCorrelation_NoTomo, {},           0, 4, 'scalar', None),
    Correlator(GNNNCorrelation_NoTomo, {},           1, 4, 'mixed',  None),
    Correlator(GGGGCorrelation_NoTomo, {},           4, 4, 'shear',  None),
]

# These are the correlators for which no dedicated discrete estimator exists, i.e. the ones
# that do not carry "Discrete" in methods_avail. We list the method to use instead. Note that
# with only setting resos=[0.] the tree-based methods are equivalent to the discrete method,
# so they are exact.
TREE_ONLY = {'NNNCorrelation': 'DoubleTree',
             'NNNNCorrelation_NoTomo': 'Tree',
             'GNNNCorrelation_NoTomo': 'Tree'}

# Subselect correlators on order with optionally excluding some. Also add
# option to skip purely scalar correlators
def correlators(orders=(2, 3, 4), exclude=(), spin2=False):
    return [s for s in CORRELATORS
            if s.order in np.atleast_1d(orders) and s.cls.__name__ not in exclude
            and (s.nspin2 > 0 or not spin2)]


def correlator_ids(specs):
    return [s.cls.__name__ for s in specs]

# Get all outputs cor correlators. For 2pt they carry different names as for higher-order
# where they default to npcf_multipoles
def correlator_outputs(spec):
    return {'NNCorrelation': ('npair',), 
            'GGCorrelation': ('xip', 'xim'),
            'NGCorrelation': ('xi',)}.get(spec.cls.__name__, ('npcf_multipoles',))

# Init any correlator class based on kwargs.
def build_correlator(spec, **kwargs):
    return spec.cls(**dict(spec.kwargs, **kwargs))

# Process any type of correlator
def run_correlator(spec, inst, shear, scalar, tomo=False, **kwargs):
    cats = {'scalar': (scalar,), 'shear': (shear,), 'mixed': (shear, scalar)}[spec.legs]
    if spec.tomo == 'dotomo':
        kwargs['dotomo'] = tomo
    elif spec.tomo == 'srclens':
        kwargs.update(dotomo_source=tomo, dotomo_lens=tomo)
    return inst.process(*cats, **kwargs)

##############
# FAST TIER  #
##############

## Parameter setup ##
NTHREADS = 2
BOXSIZE = 300.
MIN_SEP = 1.
MAX_SEP = 60.
NBINSR = 4
NBINSZ = 2 # Set dotomo=False for tests where we do not need tomography
CHI = np.pi/5. # E/B mixing angle of the analytic field, see notes sect 6.

## Shared results ##
NGAL_FASTCAT = 4000
# The bin-slop convergence test isolates the projection error, so it needs more objects
# than the smoke fixtures: at 10000 the two coarsest rmin_pixsize settings tie exactly.
NGAL_TANGENTIAL = 20000
PI = 20.

@pytest.fixture(scope="session")
def shear_catalog():
    """A small random shear catalog in a square, split over NBINSZ tomographic bins."""
    _rng = np.random.default_rng(7)
    return SpinTracerCatalog(spin=2,
                             pos1=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                             pos2=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                             tracer_1=_rng.normal(0., .3, NGAL_FASTCAT),
                             tracer_2=_rng.normal(0., .3, NGAL_FASTCAT),
                             weight=_rng.uniform(.5, 1.5, NGAL_FASTCAT),
                             zbins=_rng.integers(0, NBINSZ, NGAL_FASTCAT),
                             geometry='flat2d')


@pytest.fixture(scope="session")
def scalar_catalog():
    """A small random scalar catalog in a square, split over NBINSZ tomographic bins."""
    _rng = np.random.default_rng(8)
    return ScalarTracerCatalog(pos1=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                               pos2=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                               tracer=np.ones(NGAL_FASTCAT),
                               weight=_rng.uniform(.5, 1.5, NGAL_FASTCAT),
                               zbins=_rng.integers(0, NBINSZ, NGAL_FASTCAT),
                               geometry='flat2d')

@pytest.fixture(scope="session")
def box_shear_catalog():
    """A shear catalog in a periodic box, for the projected-slab estimators."""
    _rng = np.random.default_rng(12)
    return SpinTracerCatalog(spin=2,
                             pos1=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                             pos2=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                             pos3=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                             tracer_1=_rng.normal(0., .3, NGAL_FASTCAT),
                             tracer_2=_rng.normal(0., .3, NGAL_FASTCAT),
                             weight=_rng.uniform(.5, 1.5, NGAL_FASTCAT),
                             zbins=_rng.integers(0, NBINSZ, NGAL_FASTCAT),
                             geometry='3dbox')


@pytest.fixture(scope="session")
def box_scalar_catalog():
    """A lens catalog in the same box as ``box_shear_catalog``."""
    _rng = np.random.default_rng(13)
    return ScalarTracerCatalog(pos1=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                               pos2=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                               pos3=_rng.uniform(0., BOXSIZE, NGAL_FASTCAT),
                               tracer=np.ones(NGAL_FASTCAT),
                               weight=_rng.uniform(.5, 1.5, NGAL_FASTCAT),
                               zbins=_rng.integers(0, NBINSZ, NGAL_FASTCAT),
                               geometry='3dbox')


@pytest.fixture(scope="session")
def box_random_catalog():
    """A random catalog in the same box as ``box_shear_catalog``."""
    _rng = np.random.default_rng(14)
    nrand = 2*NGAL_FASTCAT
    return ScalarTracerCatalog(pos1=_rng.uniform(0., BOXSIZE, nrand),
                               pos2=_rng.uniform(0., BOXSIZE, nrand),
                               pos3=_rng.uniform(0., BOXSIZE, nrand),
                               tracer=np.ones(nrand),
                               weight=np.ones(nrand),
                               zbins=_rng.integers(0, NBINSZ, nrand),
                               geometry='3dbox')


@pytest.fixture(scope="session")
def spherical_catalog():
    """A shear catalog on a small sky patch, for the patch decomposition."""
    _rng = np.random.default_rng(9)
    dec = np.degrees(np.arcsin(_rng.uniform(np.sin(np.radians(-20.)),
                                            np.sin(np.radians(10.)), NGAL_FASTCAT)))
    return SpinTracerCatalog(spin=2, pos1=_rng.uniform(10., 40., NGAL_FASTCAT), pos2=dec,
                             tracer_1=_rng.normal(0., .3, NGAL_FASTCAT),
                             tracer_2=_rng.normal(0., .3, NGAL_FASTCAT),
                             weight=np.ones(NGAL_FASTCAT), geometry='spherical',
                             units_pos1='deg', units_pos2='deg')


@pytest.fixture(scope="session")
def patched_catalogs():
    """Shear, lens and random catalogs decomposed onto one shared set of patches.
    """
    _rng = np.random.default_rng(31)
    def _sky(cls, **fields):
        dec = np.degrees(np.arcsin(_rng.uniform(np.sin(np.radians(-20.)),
                                                np.sin(np.radians(10.)), NGAL_FASTCAT)))
        return cls(pos1=_rng.uniform(10., 40., NGAL_FASTCAT), pos2=dec,
                   weight=np.ones(NGAL_FASTCAT),
                   zbins=_rng.integers(0, NBINSZ, NGAL_FASTCAT), geometry='spherical',
                   units_pos1='deg', units_pos2='deg', **fields)

    cat_shape = _sky(SpinTracerCatalog, spin=2,
                     tracer_1=_rng.normal(0., .3, NGAL_FASTCAT),
                     tracer_2=_rng.normal(0., .3, NGAL_FASTCAT))
    cat_lens = _sky(ScalarTracerCatalog, tracer=np.ones(NGAL_FASTCAT))
    cat_rand = _sky(ScalarTracerCatalog, tracer=np.ones(NGAL_FASTCAT))
    cat_shape.topatches(npatches=8, method='healpix', healpix_nside=4,
                        patchextend_deg=1., n_workers=1, other_cats=[cat_lens, cat_rand])
    return cat_shape, cat_lens, cat_rand


@pytest.fixture(scope="session")
def tangential_field():
    """Create source and lens catalogs on a circle around a constant tangential shear field
    centered at the catalogs center s.t. we masure pure E. Used for isolating the projection 
    errors introduced by the tree-based approximations.
    """
    _rng = np.random.default_rng(3)
    gamma_t = .1
    center = BOXSIZE/2.
    rad = np.sqrt(_rng.uniform(MIN_SEP**2, MAX_SEP**2, NGAL_TANGENTIAL))
    ang = _rng.uniform(0., 2.*np.pi, NGAL_TANGENTIAL)
    pos1, pos2 = center + rad*np.cos(ang), center + rad*np.sin(ang)
    ell = -gamma_t*np.exp(2j*np.arctan2(pos2-center, pos1-center))
    cat_source = SpinTracerCatalog(spin=2, pos1=pos1, pos2=pos2,
                                   tracer_1=ell.real, tracer_2=ell.imag,
                                   weight=np.ones(NGAL_TANGENTIAL), geometry='flat2d')
    cat_lens = ScalarTracerCatalog(pos1=np.array([center]), pos2=np.array([center]),
                                   tracer=np.array([1.]),
                                   weight=np.array([1.]), geometry='flat2d')
    return cat_source, cat_lens, gamma_t


@pytest.fixture(scope="session")
def quadrupole_field():
    """The gaussian quadruple field from the notes. Used to test algebraic identities and 
    parity properties, i.e. not theory, so here it is sufficient to only use a few galaxies 
    that are poisson sampled."""
    fld = AnalyticField(gamma0=.05, r0=8., boxsize=BOXSIZE, chi=CHI)
    return fld.catalogs(NGAL_FASTCAT, seed=11, stratified=False)[0], fld


############################
# SLOW-TIER SHARED RESULTS #
############################

## Parameter setup ##
R0_SLOW = 3.
NGAL_SECOND = 500**2    # 250 000
NGAL_THIRD = 316**2     # nearest square to 100k
# GNN needs more galaxies than NGG for the same precision: its signal is a second moment
# in delta where NGG's is first order, so the same discrete sampling resolves it less well.
NGAL_GNN = 316**2       # nearest square to 100k
NGAL_NGG = 245**2       # nearest square to 60k
NGAL_FOURTH = 316**2    # nearest square to 100k
BINSIZE_THIRD = .1
NMAX_DEFAULT = 10 # Sweetspot between multipole convergence and small "noise-fitting" level
MIN_SEP_FOURTH, MAX_SEP_FOURTH = .6, 1.6 # Small range, i.e. narrow bins s.t. theory easy. 
NBINSR_FOURTH, NBINSPHI_FOURTH = 4, 24
TREE_SLOW = dict(tree_resos=[0., .05, .1, .2, .4], rmin_pixsize=80)
NTHREADS_SLOW = int(os.environ.get("ORPHEUS_TEST_NTHREADS",
                                   min(32, os.cpu_count() or 2))) # Take whatever we can get

# Make DoubleTree emulate the discrete estimator
THIRD_ORDER_EXACT = dict(method='DoubleTree', tree_resos=[0.])

# Recommended setup for running tests, see sec. 7.1.2 in the notes. Separations and
# aperture radii are in units of r0. aperture_radii is lower/upper range
RECOMMENDED = dict(boxsize=16., min_sep_second=.05, min_sep_third=.1, max_sep=8.,
                   binsize=.05, aperture_radii=(.75, 3.), ngal=1_000_000)

# Number of Mx legs carried by each component that computeMap3 and computeMap4 return
NCROSS = {3: np.array([0] + 3*[1] + 3*[2] + [3]),
          4: np.array([0] + 4*[1] + 6*[2] + 4*[3] + [4])}

@pytest.fixture(scope="session")
def field():
    return AnalyticField(gamma0=.05, r0=R0_SLOW, delta0=.3, chi=CHI,
                         boxsize=RECOMMENDED['boxsize']*R0_SLOW)


@pytest.fixture(scope="session")
def nn_measured(field):
    """Weighted and unweighted pair counts on one set of lens positions.
    Shared by the omega, pair-count and Nap2 tests.
    """
    cat = field.catalogs(NGAL_SECOND)[1]
    n = len(cat.pos1)
    cat_ref = ScalarTracerCatalog(pos1=cat.pos1, pos2=cat.pos2, tracer=np.ones(n),
                                  weight=np.ones(n), geometry='flat2d')
    kw = dict(min_sep=field.min_usable_sep(NGAL_SECOND),
              max_sep=RECOMMENDED['max_sep']*R0_SLOW, binsize=.2,
              nthreads=NTHREADS_SLOW, **TREE_SLOW)
    nn_d = NNCorrelation(**kw)
    nn_d.process(cat, dotomo=False)
    nn_r = NNCorrelation(**kw)
    nn_r.process(cat_ref, dotomo=False)
    return nn_d, nn_r


@pytest.fixture(scope="session")
def gg_measured(field):
    """Get GG via measurement and binned theory"""
    cat = field.catalogs(NGAL_SECOND)[0]
    gg = GGCorrelation(min_sep=RECOMMENDED['min_sep_second']*R0_SLOW,
                       max_sep=RECOMMENDED['max_sep']*R0_SLOW, binsize=.1,
                       nthreads=NTHREADS_SLOW, **TREE_SLOW)
    gg.process(cat, dotomo=False)
    theo = field.xi_binned(gg.bin_edges)
    return gg, theo

@pytest.fixture(scope="session")
def ggg_measured(field):
    """Get GGG via measurement and binned theory."""
    cat = field.catalogs(NGAL_THIRD)[0]
    ggg = GGGCorrelation(n_cfs=4, min_sep=RECOMMENDED['min_sep_third']*R0_SLOW,
                         max_sep=RECOMMENDED['max_sep']*R0_SLOW,
                         binsize=BINSIZE_THIRD, nmaxs=NMAX_DEFAULT, nbinsphi=50,
                         nthreads=NTHREADS_SLOW, **THIRD_ORDER_EXACT)
    ggg.process(cat, dotomo=False)
    ggg.multipoles2npcf(projection='Centroid')
    theory, window = field.gamma_binned(ggg.bin_edges, np.asarray(ggg.phi),
                                        centers=ggg.bin_centers_mean)
    return ggg, theory, window


@pytest.fixture(scope="session")
def gggg_measured(field):
    """Get GGG via measurement and binned theory."""
    cat = field.catalogs(NGAL_FOURTH)[0]
    gggg = GGGGCorrelation_NoTomo(min_sep=MIN_SEP_FOURTH*R0_SLOW,
                                  max_sep=MAX_SEP_FOURTH*R0_SLOW, nbinsr=NBINSR_FOURTH,
                                  nmaxs=NMAX_DEFAULT, nbinsphi=NBINSPHI_FOURTH,
                                  method='Discrete', nthreads=NTHREADS_SLOW)
    gggg.process(cat, statistics='all4pcf')
    theory, window = field.gamma4_binned(np.asarray(gggg.bin_edges),
                                         np.asarray(gggg.phis[0]), nsub=4)
    return gggg, theory, window


def _third_mixed(cls, cat_shape, cat_lens):
    """Helper to compute NGG or GNN correlators."""
    corr = cls(min_sep=RECOMMENDED['min_sep_third']*R0_SLOW,
              max_sep=RECOMMENDED['max_sep']*R0_SLOW, binsize=BINSIZE_THIRD,
              nmaxs=NMAX_DEFAULT, nbinsphi=50, nthreads=NTHREADS_SLOW,
              **THIRD_ORDER_EXACT)
    corr.process(cat_shape, cat_lens, dotomo_source=False, dotomo_lens=False)
    corr.multipoles2npcf()
    return corr


@pytest.fixture(scope="session")
def gnn_measured(field):
    """Get GNN via measurement and binned theory.
    Note that we apply the parity construction described in sect 7.6.1 in the notes
    that helps to kill the lower-order contributions."""

    cat_shape, cat_lens_p = field.catalogs(NGAL_GNN, delta_sign=1.)
    _, cat_lens_m = field.catalogs(NGAL_GNN, delta_sign=-1.)
    n = len(cat_lens_p.pos1)
    cat_lens_0 = ScalarTracerCatalog(pos1=cat_lens_p.pos1, pos2=cat_lens_p.pos2,
                                     tracer=np.ones(n), weight=np.ones(n), geometry='flat2d')
    
    gp = _third_mixed(GNNCorrelation, cat_shape, cat_lens_p)
    gm = _third_mixed(GNNCorrelation, cat_shape, cat_lens_m)
    g0 = _third_mixed(GNNCorrelation, cat_shape, cat_lens_0)
    combined = (.5*(np.asarray(gp.npcf)[0, 0] + np.asarray(gm.npcf)[0, 0])
               - np.asarray(g0.npcf)[0, 0])
    theory, window = field.gnn_binned(gp.bin_edges, np.asarray(gp.phi))
    return gp, combined, theory, window


@pytest.fixture(scope="session")
def ngg_measured(field):
    """Get NGG via measurement and binned theory.
    Note that we apply the parity construction described in sect 7.6.1 in the notes
    that helps to kill the lower-order contributions."""

    cat_shape, cat_lens_p = field.catalogs(NGAL_NGG, delta_sign=1.)
    _, cat_lens_m = field.catalogs(NGAL_NGG, delta_sign=-1.)

    gp = _third_mixed(NGGCorrelation, cat_shape, cat_lens_p)
    gm = _third_mixed(NGGCorrelation, cat_shape, cat_lens_m)
    npcf_p = np.asarray(gp.npcf)[:, 0]
    npcf_m = np.asarray(gm.npcf)[:, 0]
    combined = field.parity_combine(npcf_p, npcf_m, 1)
    theory, window = field.ngg_binned(gp.bin_edges, np.asarray(gp.phi))
    return gp, combined, theory, window