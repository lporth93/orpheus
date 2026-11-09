import ctypes
import glob
import os
import subprocess
import warnings

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter, uniform_filter

from .utils import _load_clib, check_clib_error

__all__ = ["GGCovarianceNoTomo", "GGCovariance"]


# Target share of L2 cache used by quadruplets
_L2_TARGET_FRAC = .7

####################
# HELPER FUNCTIONS #
####################

def _trapz_weights(x):
    """Trapezoidal weights for 1D non-uniform integral."""
    x = np.asarray(x, dtype=float)
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w

def _theta_dtheta(edges):
    """Integral of theta dtheta over every bin, i.e. (theta_max^2 - theta_min^2) / 2."""
    edges = np.asarray(edges, dtype=float)
    return .5 * np.diff(edges**2)

def _bin_average_blocks(blocks, weights):
    """Average blocks (..., n*ns, n*ns) over the ns sub-bins of each of the n bins, weights (n, ns)."""
    n, ns = weights.shape
    b = blocks.reshape(blocks.shape[:-2] + (n, ns, n, ns))
    M = weights.sum(1)
    return np.einsum('...aibj,ai,bj->...ab', b, weights, weights) / np.outer(M, M)

def _periodic_weights(x, period=2. * np.pi):
    """Uniform weights for periodic function sampled on a full-period grid."""
    x = np.asarray(x, dtype=float)
    return np.full_like(x, period / len(x))


def _l2_cache_bytes():
    """Per-core L2 size in bytes, or None if it cannot be determined."""
    # Check if already fixed by environment variable
    env = os.environ.get('ORPHEUS_L2_CACHE_BYTES')
    if env is not None:
        return int(env) or None

    # Try to find L2 using CPU 0 as example.
    for d in sorted(glob.glob('/sys/devices/system/cpu/cpu0/cache/index*')):
        # Find entry for L2
        try:
            with open(os.path.join(d, 'level')) as f:
                if f.read().strip() != '2':
                    continue
            with open(os.path.join(d, 'size')) as f:
                size = f.read().strip()
        except OSError:
            continue
        # Convert to bytes
        mult = {'K': 1024, 'M': 1024**2}.get(size[-1].upper(), 1)
        return int(size.rstrip('KMkm')) * mult
    # Alternative: Try to get L2 size via sysctl
    try:
        out = subprocess.run(['sysctl', '-n', 'hw.l2cachesize'],
                             capture_output=True, text=True, timeout=1)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

# Helper that chooses largest v2_block where considered quadruplets fit in cache
def _v2_block_cache_cap(n_v1, n_xi, frac=_L2_TARGET_FRAC):
    """Get largest ``v2_block`` whose quadruplet working set still fits in L2.
    """
    # Get cache size
    l2 = _l2_cache_bytes()
    if not l2 or not n_v1:
        return None
    # Each (theta1, theta2) pair needs a fixed n_xi*n_v1 contribution + v2_blocks of 4*n_v1 elements
    # --> First get the actual value and then round to multiple of eight
    v2_block_cap = int((frac*l2 - 8.*n_xi*n_v1)//(32.*n_v1))
    v2_block_cap = max(8, 8*(v2_block_cap//8))

    return v2_block_cap


def _auto_reduce_tiles(nr_block_base, v2_block_base, n_v2, n_v1=None, n_xi=2):
    """Choose tile sizes ``(nr_block, v2_block)`` for the banded quadruplet integral."""
    # Determine largest scaling factor that keeps v2_block <= n_v2.
    f_max = max(1, n_v2 // v2_block_base)
    f = min(4, f_max)
    # Reduce nr_block, but keep minimum length of 4 so the tiles follow the slab band
    nr_block = max(4, nr_block_base // f)
    # Grow v2_block according to actual reduction factor and optionally apply capping
    f = max(1, nr_block_base // nr_block)
    v2_block = min(v2_block_base * f**2, n_v2)
    cap = _v2_block_cache_cap(n_v1, n_xi)
    if cap is not None:
        v2_block = max(v2_block_base, min(v2_block, cap))
    return nr_block, v2_block

def _infer_ntomo(n_tp):
    """Recover ntomo from number of unique bin pairs n_tp = ntomo*(ntomo+1)/2."""
    nt = int((-1. + np.sqrt(1. + 8. * n_tp)) / 2.)
    if nt * (nt + 1) // 2 != n_tp:
        raise ValueError(f"Number of xi spline entries ({n_tp}) is not ntomo*(ntomo+1)/2 for any integer ntomo")
    return nt


################
# Main classes #
################

class GGCovarianceNoTomo:
    """Gaussian covariance of the shear 2PCF from pair, triplet and quadruplet counts.

    Implements Schneider et al. 2002 discrete covariance for GG. Each of the  blocks
    Cov_pp, Cov_mm and Cov_pm is the sum of the shape noise, the mixed term (triplets)
    and the cosmic variance term (quadruplet densities).

    Parameters
    ----------
    theta_centers: numpy.ndarray
        Centers of the xi_pm radial bins, shape (n_r,).
    theta_edges: numpy.ndarray
        Edges of the xi_pm radial bins, shape (n_r+1,).
    phi_centers: numpy.ndarray, optional
        Centers of the angular bins of the triplet counts in [0, 2pi), shape (n_phi,). Defaults
        to 128 uniform bins.
    phi_edges: numpy.ndarray, optional
        Edges of the angular bins, shape (n_phi+1,). Inferred from uniform ``phi_centers`` if not
        given.
    q_theta_centers, q_theta_edges: numpy.ndarray, optional
        Radial bins of the quadruplet density. For a density measured on ``catalog``, default to
        logspace(min(mean_sep,xipm_min), max_sep_footpring, q_binsize).
    q_phi_centers, q_phi_edges: numpy.ndarray, optional
        Angular bins of the quadruplet density. For a density measured on ``catalog``, default
        to 64 uniform bins.
    q_binsize: float, optional
        Log bin width of density measured on ``catalog``. Defaults to ``0.4``.
    triplet_oversample: int, optional
        Number of equal sub-bins per radial bin of the triplet counts, which are summed back
        before dividing by the pair counts. Defaults to ``1``.
    method_pairs, method_triplets, method_quadruplets: str, optional
        Source of each multiplet: ``'discrete'`` (measured from ``catalog`` or passed),
        ``'analytic'`` (idealised Poisson counts) or ``'continuous'`` (continuum limit of the
        integrals over ``spline_xip`` and ``spline_xim``). ``method_quadruplets`` also takes
        ``'frompairs'``, the constant density times ``spl_suppression_ww`` of the separations of
        the two pairs and of the separation between them. Defaults to ``'analytic'``.
    spline_xip, spline_xim: callable
        xi_+(theta) and xi_-(theta) of the input cosmology.
    sigma_eps: float, optional
        Per-component intrinsic ellipticity dispersion. Defaults to ``0.27``.
    nbar: float
        Galaxy number density in arcmin^-2.
    area: float
        Survey area in arcmin^2.
    weights_mean, weights_std: float, optional
        Mean and standard deviation of the galaxy weights. Default to ``1`` and ``0``.
    spl_suppression_ww: callable, optional
        Suppression s(theta) of the idealised counts, e.g. from
        :meth:`suppression_spline_from_pairs`. Defaults to ``None``, i.e. s = 1.
    pairs, triplets, quadruplets: numpy.ndarray or str, optional
        Count tensors of the ``'discrete'`` methods, or their ``.npy`` files or the ``.npz`` files
        of ``store_multiplets``, which also set the bin centers of the measurement. Measured from
        ``catalog`` if not given.
    catalog: dict, optional
        Positions ``ra`` and ``dec`` in deg, and optionally ``weight``, of the catalog on which the
        ``'discrete'`` multiplets are measured.
    path_save: str, optional
        Directory of the measured multiplets, required with ``catalog``.
    store_multiplets: bool, optional
        Keep the measured count tensors on disk. Defaults to ``False``.
    nmaxs: int, optional
        Largest multipole of the triplet and quadruplet measurements. Defaults to ``30``.
    geometry: str, optional
        ``'spherical'`` for ``ra`` and ``dec`` on the sky, measured on the full sphere, or
        ``'flat2d'`` for Cartesian coordinates in deg. Defaults to ``'spherical'``.
    rmin_pixsize, tree_resos, tree_method, shuffle_pix, memory_bound, verbosity: optional
        Settings of the orpheus correlators, where ``tree_method`` applies to the quadruplets.
        Default to ``40``, ``[0, 0.5, 1, 2, 4]``, ``'DoubleTree'``, ``1``, ``512`` and ``1``.
    phi_grid, varphi1_grid, varphi2_grid: numpy.ndarray, optional
        Integration grids of the quadruplet term. Default to 128 log-spaced radii from
        ``phi_grid_min`` to the last edge of ``q_theta_edges`` for a density tensor, otherwise to
        :meth:`Catalog.max_separation` of ``catalog`` or, without one, to 1e4, and 64 uniform
        angles.
    phi_grid_min: float, optional
        Smallest radius in arcmin of the default ``phi_grid``. Defaults to ``0.05``.
    phi1_subgrid_frac, phi2_subgrid_frac: numpy.ndarray, optional
        Offsets of radial subgrid nodes from the bin centers, in units of the bin width. Default
        to ``None``, i.e. the bin centers.
    v2_block, nr_block: int, optional
        Chunk size on the varphi2 axis and tile size on the (theta1, theta2) axes of the
        quadruplet integral, which bound its memory but not its result. Default to ``16`` and
        ``None``, i.e. a single tile.
    nthreads: int, optional
        Number of OpenMP threads. Defaults to ``1``.

    Notes
    -----
    * Separations are in arcmin and angles in radians. The count tensors have the shapes (n_r,)
      for the pairs, (n_r*triplet_oversample, n_r*triplet_oversample, n_phi) for the triplets and
      (n_qr, n_qr, n_qr, n_qphi, n_qphi) for the quadruplet density.

    * Peak memory of the quadruplet integral is about
      q_tensor + nthreads*(m^2*n_v1*v2_block + 2*n_out*n_r^2 + q_slice)*8B with m = nr_block*len(phi1_subgrid_frac).
      where nr_block is the quadratic and v2_block the linear knob.

    """

    def __init__(
        self,
        theta_centers, theta_edges,
        phi_centers=None, phi_edges=None,
        q_theta_centers=None, q_theta_edges=None, q_phi_centers=None, q_phi_edges=None, q_binsize=0.4,
        triplet_oversample=1,
        method_pairs='analytic', method_triplets='analytic', method_quadruplets='analytic',
        spline_xip=None, spline_xim=None,
        sigma_eps=0.27, nbar=None, area=None, weights_mean=1., weights_std=0., spl_suppression_ww=None,
        pairs=None, triplets=None, quadruplets=None,
        catalog=None, path_save=None, store_multiplets=False, nmaxs=30, geometry='spherical',
        rmin_pixsize=40, tree_resos=(0., 0.5, 1., 2., 4.),
        tree_method='DoubleTree', shuffle_pix=1, memory_bound=512., verbosity=1,
        phi_grid=None, phi_grid_min=0.05, varphi1_grid=None, varphi2_grid=None,
        phi1_subgrid_frac=None, phi2_subgrid_frac=None,
        v2_block=16, nr_block=None, nthreads=1):

        ## Link compiled libraries
        self._link_clib()

        ## Argument consistency
        # SVA needs area and any idealised calculation needs nbar
        if area is None:
            raise ValueError("`area` is required")
        if nbar is None and (method_pairs != 'discrete' or method_triplets != 'discrete'
                             or method_quadruplets in ('analytic', 'frompairs')):
            raise ValueError("`nbar` is required for idealised, continuous or frompairs multiplets")

        ## 2PCF radial bins
        self.theta_centers = np.asarray(theta_centers, dtype=float)
        self.theta_edges = np.asarray(theta_edges, dtype=float)
        if len(self.theta_edges) != len(self.theta_centers) + 1:
            raise ValueError("theta_edges must have exactly one more entry than theta_centers")
        self.dtheta = np.diff(self.theta_edges)
        self.n_r = len(self.theta_centers)

        ## Angular bins for triplets and quadruplets
        if phi_centers is not None:
            self.phi_centers = np.asarray(phi_centers, dtype=float)
            if phi_edges is not None:
                self.phi_edges = np.asarray(phi_edges, dtype=float)
            else:
                dphi = self.phi_centers[1] - self.phi_centers[0]
                if not np.allclose(np.diff(self.phi_centers), dphi):
                    raise ValueError("phi_centers must be uniform when phi_edges is not provided")
                self.phi_edges = np.concatenate([
                    self.phi_centers - 0.5 * dphi,
                    [self.phi_centers[-1] + 0.5 * dphi]])
        else:
            n_phi = 128
            self.phi_edges = np.linspace(0., 2. * np.pi, n_phi + 1)
            self.phi_centers = 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])
        self.n_phi = len(self.phi_centers)
        self.dphi = np.diff(self.phi_edges)

        ## Tiling and threads
        self.v2_block = int(v2_block)
        if self.v2_block < 1:
            raise ValueError("v2_block must be >= 1")
        # Tile size on the (theta1, theta2) axes of the quadruplet integral, None for one tile
        self.nr_block = None if nr_block is None else int(nr_block)
        if self.nr_block is not None and self.nr_block < 1:
            raise ValueError("nr_block must be >= 1 (or None for no tiling)")
        self.nthreads = int(nthreads)
        if self.nthreads < 1:
            raise ValueError("nthreads must be >= 1")

        ## Catalog and orpheus setup of the discrete multiplet measurements
        self.catalog = catalog
        self.path_save = path_save
        self.store_multiplets = bool(store_multiplets)
        self.nmaxs = nmaxs
        self.geometry = geometry
        self.rmin_pixsize = rmin_pixsize
        self.tree_resos = np.asarray(tree_resos, dtype=float)
        self.tree_method = tree_method
        self.shuffle_pix = shuffle_pix
        self.memory_bound = memory_bound
        self.verbosity = verbosity
        self.orpheus_instances = {}
        self.pairs_wsq = None
        # Init catalogs already here as those set the distance bounds for the quadruplets
        if self.catalog is not None:
            self.set_orpheus_instances([])

        ## Quadruplet radial / angular grid
        # In case no array is given we build the default as follows
        # * lower cut --> 1/sqrt(nbar) or lowest xipm bin. Aims at not having too much noise in q
        # * upper cut --> Largest separation allowed by catalog
        # * binsize --> Log bins of 0.4 which is ok even for huge ranges given that we interpolate.
        # * Angular reso --> Use 64 bins which should be ok as long as the footprint shape is not crazy
        if method_quadruplets == 'discrete' and quadruplets is None and catalog is not None:
            if q_theta_centers is None:
                q_lo = min(self.theta_edges[0], np.sqrt(area/len(catalog['ra'])))
                sep_max = (self.orpheus_instances['catalog'].max_separation()
                           * (60. if geometry == 'spherical' else 1.))
                n_q = int(np.ceil(np.log(sep_max/q_lo)/q_binsize))
                q_theta_edges = q_lo*np.exp(q_binsize*np.arange(n_q + 1))
                q_theta_centers = np.sqrt(q_theta_edges[1:]*q_theta_edges[:-1])
            if q_phi_centers is None:
                q_phi_edges = np.linspace(0., 2.*np.pi, 65)
                q_phi_centers = .5*(q_phi_edges[1:] + q_phi_edges[:-1])
        if q_theta_centers is None and method_quadruplets == 'discrete' \
                and not callable(quadruplets):
            warnings.warn(
                "method_quadruplets='discrete' but q_theta_centers / "
                "q_theta_edges were not provided. The q-tensor will be "
                "assumed to live on the ξ_± radial grid; queries outside "
                "that range will silently return q=0. Pass q_theta_centers "
                "/ q_theta_edges matching the orpheus 4PCF binning if your "
                "q array extends further radially.",
                RuntimeWarning, stacklevel=2)
        self.q_theta_centers = (self.theta_centers if q_theta_centers is None else np.asarray(q_theta_centers, dtype=float))
        self.q_theta_edges = (self.theta_edges if q_theta_edges is None else np.asarray(q_theta_edges, dtype=float))
        if len(self.q_theta_edges) != len(self.q_theta_centers) + 1:
            raise ValueError("q_theta_edges must have exactly one more entry than q_theta_centers")
        self.q_dtheta = np.diff(self.q_theta_edges)
        self.q_n_r = len(self.q_theta_centers)
        self.q_phi_centers = (self.phi_centers if q_phi_centers is None else np.asarray(q_phi_centers, dtype=float))
        self.q_phi_edges = (self.phi_edges if q_phi_edges is None else np.asarray(q_phi_edges, dtype=float))
        self.q_dphi = np.diff(self.q_phi_edges)
        self.q_n_phi = len(self.q_phi_centers)

        ## Triplet oversampling
        self.triplet_oversample = int(triplet_oversample)
        if self.triplet_oversample < 1:
            raise ValueError("triplet_oversample must be >= 1")
        alpha = self.triplet_oversample
        if alpha > 1:
            # Build fine radial grid by subdividing each coarse bin into equal log sub-bins
            log_edges = np.log(self.theta_edges)
            sub_edges = np.exp(log_edges[:-1, None] + np.arange(alpha + 1)[None, :]*np.diff(log_edges)[:, None]/alpha)
            fine_edges = np.concatenate([sub_edges[:, :-1].ravel(), [self.theta_edges[-1]]])
            self.triplet_theta_edges = fine_edges
            self.triplet_theta_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
            self.triplet_dtheta = np.diff(fine_edges)
            self.triplet_n_r = self.n_r * alpha
        else:
            self.triplet_theta_edges = self.theta_edges
            self.triplet_theta_centers = self.theta_centers
            self.triplet_dtheta = self.dtheta
            self.triplet_n_r = self.n_r

        ## Method switches
        self.method_pairs = method_pairs
        self.method_triplets = method_triplets
        self.method_quadruplets = method_quadruplets

        ## Cosmological inputs
        self.xip = spline_xip
        self.xim = spline_xim
        self.sigma_eps = float(sigma_eps)

        ## Survey / weighting
        self.nbar = nbar
        self.area = area
        self.weights_mean = float(weights_mean)
        self.weights_std = float(weights_std)
        self.spl_suppression_ww = spl_suppression_ww
        # <w> and <w^2> = <w>^2 + Var(w)
        self._w1 = self.weights_mean
        self._w2 = self.weights_mean**2 + self.weights_std**2

        ## Integration grids (analytic branch + quadruplet density integral)
        if method_quadruplets == 'discrete':
            phi_max = self.q_theta_edges[-1]
        elif catalog is not None:
            phi_max = (self.orpheus_instances['catalog'].max_separation()
                       * (60. if geometry == 'spherical' else 1.))
        else:
            phi_max = 1e4
        self.phi_grid = (np.geomspace(phi_grid_min, phi_max, 128) if phi_grid is None else np.asarray(phi_grid))
        self.varphi1_grid = (np.linspace(0., 2.*np.pi, 64, endpoint=False) if varphi1_grid is None else np.asarray(varphi1_grid))
        self.varphi2_grid = (np.linspace(0., 2.*np.pi, 64, endpoint=False) if varphi2_grid is None else np.asarray(varphi2_grid))
        self.phi1_subgrid_frac = None if phi1_subgrid_frac is None else np.asarray(phi1_subgrid_frac, dtype=float)
        self.phi2_subgrid_frac = None if phi2_subgrid_frac is None else np.asarray(phi2_subgrid_frac, dtype=float)

        ## Output storage
        self.D = None
        self.qpp = None
        self.qmm = None
        self.qpm = None
        self.rp0 = None
        self.rp1 = None
        self.rm0 = None
        self.rm1 = None
        self.rpm = None
        self.cov_pp = None
        self.cov_mm = None
        self.cov_pm = None
        self.cov_xipm = None

        ## Correlators of the discrete multiplet measurements
        if self.catalog is not None:
            self.set_orpheus_instances([kind for kind, method, value in (
                ('pairs', method_pairs, pairs), ('triplets', method_triplets, triplets),
                ('quadruplets', method_quadruplets, quadruplets))
                if method == 'discrete' and value is None])

        ## Multiplet count initialisation
        self.pairs = self._init_multiplet('pairs', pairs, self.method_pairs)
        self.triplets = self._init_multiplet('triplets', triplets, self.method_triplets)
        self.quadruplets = self._init_multiplet('quadruplets', quadruplets, self.method_quadruplets)

    ## Generic helpers for init
    def _link_clib(self):
        self.library_path, self._clib = _load_clib()
        ci = ctypes.c_int
        cd = ctypes.c_double
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        self._clib.interp4d.restype = None
        self._clib.interp4d.argtypes = [
            p_f64, p_f64, p_f64, ci, ci, cd, cd, cd, cd, cd, cd,
            p_f64, p_f64, p_f64, p_f64, ci, p_f64]
        self._clib.quadruplet_sva_integrate.restype = None
        self._clib.quadruplet_sva_integrate.argtypes = [
            p_f64, p_f64, ci, ci, p_f64, p_f64, ci, p_f64, p_f64, ci,
            p_f64, p_f64, ci, p_f64, p_f64, ci, ci, ci, p_i32, ci,
            ci, p_f64, p_f64, p_f64, ci, ci, cd, p_f64, p_f64, p_f64,
            p_f64, ci, cd, cd, p_f64, ci, ci, p_i32, p_i32,
            p_i32, p_i32, p_f64, ci, p_i32, ci, p_i32, p_i32, p_f64, ci, ci]
        self._clib.continuous_q_single.restype = None
        self._clib.continuous_q_single.argtypes = [
            ci, ci, p_f64, p_f64, p_f64, p_f64, ci, ci,
            p_f64, ci, cd, cd, p_f64,
            ci, cd, cd, cd, p_f64, p_f64]
        self._clib.continuous_q_tomo.restype = None
        self._clib.continuous_q_tomo.argtypes = [
            ci, ci, p_f64, p_f64, p_f64, p_f64, ci, ci,
            p_f64, ci, cd, cd, p_f64, ci,
            cd, cd, cd, p_f64, ci, p_i32, p_i32, p_i32,
            p_f64, ci, p_f64, ci, ci]
        self._clib.continuous_rterms_tomo.restype = None
        self._clib.continuous_rterms_tomo.argtypes = [
            ci, p_f64, p_f64, p_f64, ci, p_f64, p_f64, ci, p_f64, p_f64, ci, cd,
            p_f64, ci, cd, cd, p_f64, ci, ci, p_i32, p_i32, p_i32, p_i32, p_i32,
            ci, p_f64, ci, ci]
        self._clib.triplet_term_reduce.restype = None
        self._clib.triplet_term_reduce.argtypes = [
            p_f64, ci, ci,  p_f64, p_f64, ci,
            p_f64, ci, cd, cd, p_f64,
            p_f64, ci, ci]

    def _init_multiplet(self, kind, value, method):
        """Define construction of count tensor of one multiplet."""
        # Discrete --> Load, measure or validate
        if method == 'discrete':
            if value is None:
                return self.computeDiscrete(kind)
            if kind == 'quadruplets' and callable(value):
                return value
            if isinstance(value, str) and value.endswith('.npz'):
                value = self._load_multiplet(kind, value)
            value = np.asarray(np.load(value) if isinstance(value, str) else value, dtype=float)
            self._check_shape(kind, value)
            return value
        # Anything else --> Build as in _MULTIPLET_SOURCES
        sources = self._MULTIPLET_SOURCES[kind]
        if method not in sources:
            raise ValueError(f"Unknown method_{kind}='{method}'")
        if value is not None:
            warnings.warn(
                f"method_{kind}='{method}' but a {kind} tensor was provided; "
                "the tensor will be ignored.", UserWarning, stacklevel=3)
        builder = sources[method]
        return getattr(self, builder)() if builder is not None else None

    #################################
    ## Some class-internal globals ##
    #################################

    # How to construct the multiplet data of each non-discrete method
    _MULTIPLET_SOURCES = {
        'pairs': {'analytic': 'pairs_analytic', 'continuous': 'pairs_analytic'},
        'triplets': {'analytic': 'triplets_analytic', 'continuous': None},
        'quadruplets': {'analytic': 'quadruplets_analytic',
                        'frompairs': 'quadruplets_analytic',
                        'continuous': None}}
    
    # (block, label, q term, r terms, shape noise on the diagonal)
    COVBLOCKS = (('pp', '++', 'qpp', ('rp0', 'rp1'), True),
                 ('mm', '--', 'qmm', ('rm0', 'rm1'), True),
                 ('pm', '+-', 'qpm', ('rpm',), False))
    BLOCKS = tuple(block for block, *_ in COVBLOCKS)
    
    QTERMS = (('qpp', 0, 'p'), ('qmm', 1, 'p'), ('qpm', 2, 'm')) # (name, weight_type, xi_pm)
    QTERMS_SYM = ('qpp', 'qmm') # Symmetric under theta1 <-> theta2
    RTERMS = ('rp0', 'rp1', 'rm0', 'rm1', 'rpm')
    RTERMS_SYM = ('rp0', 'rp1', 'rm0', 'rm1') # Symmetric under theta1 <-> theta2
    RTERM_BLOCKS = {'rp0': 'pp', 'rp1': 'pp', 'rm0': 'mm', 'rm1': 'mm', 'rpm': 'pm'} # Covariance block of each r-term
    # xi_pm entering the two pairs of each r-term
    RTERM_XI = {'rp0': ('p', 'p'), 'rp1': ('m', 'm'), 'rm0': ('m', 'm'), 'rm1': ('p', 'p'), 'rpm': ('p', 'm')}

    # Grid whose nominal centers the mean bin centers of a measurement replace
    _MULTIPLET_GRIDS = {'pairs': 'theta_centers', 'triplets': 'triplet_theta_centers',
                        'quadruplets': 'q_theta_centers'}

    ###################################
    # Covariance of any GG statistics #
    ###################################

    def cov_of_statistic(self, 
        filter_type, radii=None, Nmax=None, theta_min=None, theta_max=None,
        cosebi_dps=100, cosebi_nsample=4096,
        return_weights=False,):
        r"""Compute the covariance of a statistic linear in xi_pm, S+02 Eqs. 41 and 42.

        .. math::
        
            \mathrm{Cov}(S_a, S_b) = w_a^T\, \mathrm{Cov}_{\xi_\pm}\, w_b
        
        for S = sum_i [w_+(theta_i) xi_+(theta_i) + w_-(theta_i) xi_-(theta_i)]. 

        Parameters
        ----------
        filter_type: {'Map2', 'cosebis', 'puremode'}
            Statistic to consider.
        radii: array_like, optional
            Aperture radii, required for 'Map2'.
        Nmax: int, optional
            Number of COSEBI modes, required for 'cosebis'.
        theta_min, theta_max: float, optional
            Angular range. Defaults to the outer bin edges.
        cosebi_dps: int, optional
            mpmath precision for COSEBI root finding.
        cosebi_nsample: int, optional
            Number of quadrature points for T_-.
        return_weights: bool, optional
            Also return the statistic weights.

        Returns
        -------
        dict
            For 'Map2' and 'cosebis': 'cov_EE', 'cov_BB', and 'cov_EB'. 'Cosebi' have units rad^4.
            For 'puremode': 'cov' and 'labels'.
            If return_weights is True, also return the corresponding weights: (W_E, W_B) or (K, labels).
        """
        from .npcf_second import GGCorrelation

        ## 1) Get xipm covariance
        if self.cov_xipm is None:
            self.get_cov_xipm()
        C = self.cov_xipm
        n = self.n_r

        ## 2) Init helper GGCorrelation. We set nzcombis to nbinsr to get the filters in the right shape
        tmin = (float(self.theta_edges[0]) if theta_min is None else float(theta_min))
        tmax = (float(self.theta_edges[-1]) if theta_max is None else float(theta_max))
        gg = object.__new__(GGCorrelation)
        gg.binsize = np.log(tmax/tmin) / n
        gg.bin_edges = np.geomspace(tmin, tmax, n+1)
        gg.min_sep = tmin
        gg.max_sep = tmax
        gg.nbinsr = n
        gg.bin_centers_mean = self.theta_centers
        gg.bin_centers = np.tile(self.theta_centers, (n, 1))
        gg.nzcombis = n

        ## 3) Get filter weights
        # The filters integrate using binsize * theta^2.
        # Absorb the measure correction into the input correlation.
        corr = np.diff(self.theta_edges) / (gg.binsize * self.theta_centers)
        unit = np.diag(corr).astype(complex)
        zero = np.zeros((n, n), dtype=complex)

        # Helper to compute xi+- responses
        def get_responses(estimator):
            gg.xip = unit; gg.xim = zero
            response_plus = estimator()
            gg.xip = zero; gg.xim = unit
            response_minus = estimator()
            return response_plus, response_minus
        def prepare_kernel(response):
            return (response/corr[np.newaxis, :, np.newaxis]).real.transpose(0, 2, 1)

        # Now allocate the filter weights
        if filter_type == 'Map2':
            if radii is None:
                raise ValueError("'radii' is required for filter_type='Map2'")
            radii = np.asarray(radii, dtype=float).ravel()
            res_p, res_m = get_responses(lambda: gg.computeMap2(radii))
            w_p = res_p[0].T; w_m = res_m[0].T
        elif filter_type == 'cosebis':
            if Nmax is None:
                raise ValueError("'Nmax' is required for filter_type='cosebis'")
            res_p, res_m = get_responses(lambda: gg.computecosebi(
                Nmax, dps=cosebi_dps, Tminus_nsample=cosebi_nsample, units='rad2'))
            w_p = res_p[0].real.T; w_m = res_m[0].real.T
        elif filter_type == 'puremode':
            (xip_p, xim_p), (xip_m, xim_m) = get_responses(gg.computepuremode)
            # Remove the measure correction from the kernels.
            pp, mp, pm, mm = map(prepare_kernel,(xip_p, xim_p, xip_m, xim_m))
            K = np.block([[pp[k], pm[k]] for k in range(3)] + [[mp[k], mm[k]] for k in range(3)])
            cov_full = K @ C @ K.T
            labels = [f'{component}[{i}]'
                      for component in ('xip_E', 'xip_B', 'xip_amb', 'xim_E', 'xim_B', 'xim_amb')
                      for i in range(n)]
            out = {'cov': cov_full, 'labels': labels,}
            if return_weights:
                return out, (K, labels)
            return out
        else:
            raise ValueError(f"filter_type must be 'Map2', 'cosebis' or 'puremode', got '{filter_type}'")

        ## 4) Propagate cov in E-/B-Mode statistics
        W_E = np.hstack((w_p, w_m))
        W_B = np.hstack((w_p, -w_m))
        out = {'cov_EE': W_E @ C @ W_E.T, 
               'cov_BB': W_B @ C @ W_B.T,
               'cov_EB': W_E @ C @ W_B.T,}

        return (out, (W_E, W_B)) if return_weights else out

    ##########################################
    # Resampling xipm cov to a finer binning #
    ##########################################

    def subbin(self, new_theta_centers, new_theta_edges, new_pairs=None,
               infer_pairs=False, use_corr=True, log_theta=True,
               recompute_slab=False, slab_width=None, slab_nr_block=None,
               slab_instance=None, verbose=False):
        """Resample the covariance to a finer radial binning by interpolation.

        Parameters
        ----------
        new_theta_centers: numpy.ndarray
            Centers of the fine radial bins, shape (n_new,).
        new_theta_edges: numpy.ndarray
            Edges of the fine radial bins, shape (n_new+1,).
        new_pairs: numpy.ndarray, optional
            Measured fine pair counts, shape (n_new,). 
        infer_pairs: bool, optional
            Estimate the fine pair counts from the coarse ratio N_p / N_p^ideal. Defaults to
            ``False``.
        use_corr: bool, optional
            Interpolate the components relative to their diagonal, i.e. via the correlation
            coefficient for the symmetric ones. Defaults to ``True``.
        log_theta: bool, optional
            Interpolate in log theta. Defaults to ``True``.
        recompute_slab: bool, optional
            Recompute a band of width ``slab_width`` on the fine grid. Defaults to ``False``.
        slab_width: float or dict, optional
            Element (a, b) is in the band if 1/w < theta_b/theta_a < w, per block for a dict
            with the keys ``'pp'``, ``'mm'`` and ``'pm'``, where a missing block gets w = 1,
            i.e. no band.
        slab_nr_block: int, optional
            Tile size of the banded quadruplet integral. Defaults to about the band half-width.
        slab_instance: GGCovarianceNoTomo, optional
            Instance on the fine grid that supplies D, the mixed terms and the band of the
            r-terms. With ``recompute_slab`` built from ``catalog`` if not given, or from the
            idealised counts for the analytic method.
        verbose: bool, optional
            Print the progress of the quadruplet integrals. Defaults to ``False``.

        Returns
        -------
        GGCovarianceNoTomo
            Instance on the fine grid with all components loaded.

        Notes
        -----
        Each component is interpolated on its own. Plain interpolation undershoots the diagonal peak 
        of strongly diagonal components. Setting``recompute_slab=True`` reduces this effect by evaluating 
        a band around the diagonal on the fine grid. With a fine instance, D and the mixed terms are 
        evaluated on the whole fine grid.
        """

        ## 1) Prepare coarse and fine radial grids
        new_tc = np.asarray(new_theta_centers, dtype=float)
        new_te = np.asarray(new_theta_edges, dtype=float)
        if len(new_te) != len(new_tc) + 1:
            raise ValueError("new_theta_edges must have exactly one more entry than new_theta_centers")
        n_new = len(new_tc)
        tc = self.theta_centers
        logtc = np.log(tc) if log_theta else tc
        logtcnew = np.log(new_tc) if log_theta else new_tc
        Np_c = self.pairs
        measrad_c = _theta_dtheta(self.theta_edges) # int theta dtheta, coarse

        ## 2) Infer fine paircounts
        if infer_pairs and new_pairs is None:
            if self.nbar is None:
                raise ValueError("`infer_pairs=True` requires `nbar`")
            _pref = 2.*np.pi*self.area*self.nbar**2 * self._w1**2
            ideal_c = _pref * measrad_c * self._suppression(tc)
            ratio_c = Np_c / ideal_c
            ideal_f = _pref * _theta_dtheta(new_te) * self._suppression(new_tc)
            log_ratio_f = np.interp(logtcnew, logtc, np.log(ratio_c))
            new_pairs = np.exp(log_ratio_f) * ideal_f

        ## 3) Get fine grid instance where D and the q-terms can be quickly evaluated
        fine = slab_instance
        if fine is None and recompute_slab:
            # Catalog available --> Measure the fine pairs and triplets, reuse the quadruplet density.
            if getattr(self, 'catalog', None) is not None:
                fine = self._fine_instance_from_catalog(new_tc, new_te, verbose=verbose)
                if new_pairs is None:
                    new_pairs = fine.pairs
            # Analytic multiplets --> Construct instance  directly on fine grid
            elif self.method_pairs == self.method_triplets == self.method_quadruplets == 'analytic':
                fine = GGCovarianceNoTomo(**self._grid_kwargs(
                    new_tc, new_te,
                    q_theta_centers=new_tc, q_theta_edges=new_te,
                    q_phi_centers=None, q_phi_edges=None,
                    method_pairs='analytic', method_triplets='analytic',
                    method_quadruplets='analytic', nr_block=None))
                if new_pairs is not None:
                    fine.pairs = np.asarray(new_pairs, dtype=float)
            else:
                raise NotImplementedError(
                    "recompute_slab=True without an explicit `slab_instance` "
                    "is only supported for analytic count tensors. For the "
                    "'discrete' method, build a fine-grid GGCovarianceNoTomo "
                    "with your measured triplet/quadruplet tensors and pass it "
                    "as subbin(..., slab_instance=fine).")
        # Make sure that the fine instance uses the correct radial grid 
        if fine is not None and (fine.n_r != len(new_tc) or not np.allclose(fine.theta_edges, new_te)):
            raise ValueError("slab_instance must live on the same 2PCF theta grid as "
                             "the subbin target (new_theta_centers / new_theta_edges).")

        ## 4) Setup interpolation of cov components
        _interp_arr = self._build_block_interpolator(logtc, logtcnew, n_new, use_corr)
        if new_pairs is not None:
            Np_f = np.asarray(new_pairs, dtype=float)
            if Np_f.shape != (n_new,):
                raise ValueError(f"new_pairs must have shape ({n_new},), got {Np_f.shape}")
            measrad_f = _theta_dtheta(new_te)
            Np_c_outer = np.outer(Np_c, Np_c)
            measrad_c_outer = np.outer(measrad_c, measrad_c)
            measrad_f_outer = np.outer(measrad_f, measrad_f)
            Np_f_outer = np.outer(Np_f, Np_f)

            def _interp(arr, sym):
                # Interpolate the smooth density and apply the fine N_p
                return _interp_arr(arr * Np_c_outer / measrad_c_outer, sym) * measrad_f_outer / Np_f_outer
        else:
            _interp = _interp_arr

        ## 5) Now interpolate the cov components
        # Interpolate coarse r-terms and in case no fine instance is supplied also the q-terms
        self._ensure_all_rterms(verbose=verbose)
        specs = [(name, name in self.RTERMS_SYM) for name in self.RTERMS]
        if fine is None:
            if self.D is None: self.getD()
            if self.qpp is None: self.getqpp()
            if self.qmm is None: self.getqmm()
            if self.qpm is None: self.getqpm()
            specs += [(name, name in self.QTERMS_SYM) for name, *_ in self.QTERMS]
        comps = {name: _interp(getattr(self, name), sym) for name, sym in specs}
        # We have a fine instance --> compute the cheap D and qij on the full grid directly
        if fine is not None:
            comps.update(D=fine.getD(), qpp=fine.getqpp(), qmm=fine.getqmm(), qpm=fine.getqpm())
        # No fine instance --> D lives on the diagonal and is resampled as a function of theta.
        # With fine pairs D*N_p is constant, otherwise D theta*dtheta is smooth and interpolated.
        else:
            if new_pairs is not None:
                comps['D'] = float(np.mean(self.D * Np_c)) / Np_f
            else:
                comps['D'] = np.exp(np.interp(logtcnew, logtc, np.log(self.D * measrad_c))) / _theta_dtheta(new_te)            

        ## 6) Optionally recompute the diagonal band of the r-terms on the fine grid.
        if recompute_slab:
            # Get slab widths and their associated masks
            if slab_width is None:
                raise ValueError("recompute_slab=True requires `slab_width`.")
            if np.isscalar(slab_width):
                widths = {b: float(slab_width) for b in self.BLOCKS}
            else:
                widths = {b: float(slab_width.get(b, 1.)) for b in self.BLOCKS}
            masks = {b: self._slab_band_mask(new_tc, w) for b, w in widths.items()}
            mask_max = masks[max(widths, key=widths.get)]
            # Optionally print some numbers regarding allocated elements on the grids
            if verbose:
                print("  subbin: %d -> %d bins, r-terms of the band w = %.3g (pp), %.3g (mm), %.3g (pm) "
                      "on the fine grid, %.1f%% of its elements"
                      % (self.n_r, n_new, widths['pp'], widths['mm'], widths['pm'],
                         100.*mask_max.mean()))
            # If there is sth to compute, first set up tiles for good caching, compute the r-terms, and update the comps
            if mask_max.any():
                self._set_slab_tiles(fine, new_tc, widths, slab_nr_block)
                band = dict(zip(self.RTERMS, fine.get_all_rterms(pair_mask=masks, verbose=verbose)))
                for name in self.RTERMS:
                    m = masks[self.RTERM_BLOCKS[name]]
                    comps[name][m] = band[name][m]

        ## 7) Init the finely gridded cov instance and return this
        return GGCovarianceNoTomo._from_components(
            theta_centers=new_tc, theta_edges=new_te,
            phi_centers=self.phi_centers, phi_edges=self.phi_edges,
            spline_xip=self.xip, spline_xim=self.xim,
            sigma_eps=self.sigma_eps, nbar=self.nbar, area=self.area,
            weights_mean=self.weights_mean, weights_std=self.weights_std,
            nthreads=self.nthreads, **comps)

    def _fine_instance_from_catalog(self, new_theta_centers, new_theta_edges, verbose=False):
        """Build a fine-grid instance from the catalog of this instance. 
        Setup such that fine pairs and triplets are measured and quadruplets are reused.
        """
        if getattr(self, 'catalog', None) is None:
            raise ValueError("_fine_instance_from_catalog requires a `catalog` on this instance.")
        if verbose:
            print("  subbin: auto-building fine instance (n_fine=%d) -- "
                  "measuring fine pairs/triplets, reusing q density"%len(new_theta_centers))
        return GGCovarianceNoTomo(**self._grid_kwargs(
            new_theta_centers, new_theta_edges,
            method_pairs='discrete', method_triplets='discrete',
            method_quadruplets='discrete', quadruplets=self.quadruplets))

    def _grid_kwargs(self, new_theta_centers, new_theta_edges, **overrides):
        """Constructor arguments for this instance on a new theta grid, updated by ``overrides``."""
        kw = dict(
            theta_centers=new_theta_centers, theta_edges=new_theta_edges,
            phi_centers=self.phi_centers, phi_edges=self.phi_edges,
            q_theta_centers=self.q_theta_centers, q_theta_edges=self.q_theta_edges,
            q_phi_centers=self.q_phi_centers, q_phi_edges=self.q_phi_edges,
            triplet_oversample=self.triplet_oversample,
            spline_xip=self.xip, spline_xim=self.xim,
            sigma_eps=self.sigma_eps, nbar=self.nbar, area=self.area,
            weights_mean=self.weights_mean, weights_std=self.weights_std,
            spl_suppression_ww=self.spl_suppression_ww,
            phi_grid=self.phi_grid, varphi1_grid=self.varphi1_grid,
            varphi2_grid=self.varphi2_grid,
            phi1_subgrid_frac=self.phi1_subgrid_frac,
            phi2_subgrid_frac=self.phi2_subgrid_frac,
            v2_block=self.v2_block, nr_block=self.nr_block,
            nthreads=self.nthreads,
            catalog=self.catalog, path_save=self.path_save, store_multiplets=self.store_multiplets,
            nmaxs=self.nmaxs, geometry=self.geometry, rmin_pixsize=self.rmin_pixsize,
            tree_resos=self.tree_resos, tree_method=self.tree_method, shuffle_pix=self.shuffle_pix,
            memory_bound=self.memory_bound, verbosity=self.verbosity)
        kw.update(overrides)
        return kw

    @staticmethod
    def _build_block_interpolator(x_old, x_new, n_new, use_corr=True):
        """Build a 2D interpolator for covariance blocks."""
        # Construct 1D interpolation matrix
        W = RegularGridInterpolator((x_old,), np.eye(len(x_old)),
                                    method='linear', bounds_error=False, fill_value=None)(x_new[:, None])
        # Direct interpolation of a block. Do this in logspace for positive definite elements, else in linspace
        def _raw(arr, log_interp=False):
            data = np.log(arr) if log_interp else arr
            result = W @ data @ W.T
            return np.exp(result) if log_interp else result
        # Interpolation using correlation coefficient. Done in linspace.
        def _corr(arr):
            var = np.diag(arr)
            # For a block off the diagonal there are rare cases where this could happen
            if np.any(var <= 0.):
                return _raw(arr)
            rho_f = _raw(arr/np.sqrt(np.outer(var, var)))
            np.fill_diagonal(rho_f, 1.)
            var_f = np.exp(np.interp(x_new, x_old, np.log(var)))
            return rho_f*np.sqrt(np.outer(var_f, var_f))
        # The callable interpolator
        def _interp(arr, sym):
            if not np.any(arr):
                return np.zeros((n_new, n_new))
            result = (_corr(arr) if use_corr else _raw(arr, log_interp=bool(np.all(arr > 0))))
            if sym:
                result = .5 * (result + result.T)
            return result
        return _interp

    def _set_slab_tiles(self, fine, new_tc, widths, slab_nr_block):
        """Tile sizes of the banded quadruplet integral on ``fine``, set on ``fine``."""
        n_v2 = len(self.varphi2_grid)
        n_v1 = len(self.varphi1_grid)
        n_xi = getattr(self, 'ntomo', 1)*(getattr(self, 'ntomo', 1)+1)
        # Slab blocksize given --> Keep it
        if slab_nr_block is not None:
            fine.nr_block = int(slab_nr_block)
            return
        # Blocksize given on fine grid --> Infer good gridsize given memory budget
        if fine.nr_block is not None:
            fine.nr_block, fine.v2_block = _auto_reduce_tiles(
                fine.nr_block, fine.v2_block, n_v2, n_v1, n_xi)
        # Nothing given --> Define blocksizes from band half-width
        else:
            dln = float(np.mean(np.diff(np.log(new_tc))))
            m_max = max((int(np.ceil(np.log(w) / dln)) for w in widths.values() if w > 1.), default=1)
            fine.nr_block, fine.v2_block = _auto_reduce_tiles(
                max(4, m_max), self.v2_block, n_v2, n_v1, n_xi)


    @staticmethod
    def _slab_band_mask(theta_centers, slab_width):
        """Boolean (N, N) band around the diagonal, 1/w < theta_j/theta_i < w."""
        tc = np.asarray(theta_centers, dtype=float)
        N = len(tc)
        if slab_width is None or slab_width <= 1.:
            return np.zeros((N, N), dtype=bool)
        ratio = tc[None, :] / tc[:, None]
        return (ratio < slab_width) & (ratio > 1. / slab_width)

    @classmethod
    def _from_components(cls, theta_centers, theta_edges,
                         phi_centers, phi_edges,
                         spline_xip, spline_xim, sigma_eps,
                         nbar, area, weights_mean, weights_std,
                         D=None, qpp=None, qmm=None, qpm=None,
                         rp0=None, rp1=None, rm0=None, rm1=None, rpm=None,
                         nthreads=None, template=None, pairs=None, **tomo):
        """Create an instance from precomputed covariance components, bypassing ``__init__``.

        * ``template`` passes on the quadruplet and integration grids, the suppression spline and the tiling
        * ``pairs`` the fine pair counts
        * ``tomo`` the tomographic blocks and their bookkeeping.
        """
        obj = object.__new__(cls)
        obj._link_clib()

        obj.theta_centers = np.asarray(theta_centers, dtype=float)
        obj.theta_edges = np.asarray(theta_edges, dtype=float)
        obj.dtheta = np.diff(obj.theta_edges)
        obj.n_r = len(obj.theta_centers)

        obj.phi_centers = np.asarray(phi_centers, dtype=float)
        obj.phi_edges = np.asarray(phi_edges, dtype=float)
        obj.n_phi = len(obj.phi_centers)
        obj.dphi = np.diff(obj.phi_edges)

        # q-tensor grid defaults to the 2PCF grid for precomputed instances, else follows template grid
        for name, default in (
            ('q_theta_centers', obj.theta_centers), ('q_theta_edges', obj.theta_edges),
            ('q_dtheta', obj.dtheta), ('q_n_r', obj.n_r),
            ('q_phi_centers', obj.phi_centers), ('q_phi_edges', obj.phi_edges),
            ('q_dphi', obj.dphi), ('q_n_phi', obj.n_phi)):
            setattr(obj, name, getattr(template, name, default))

        # Triplet grid: no oversampling for precomputed instances
        obj.triplet_oversample = 1
        obj.triplet_theta_centers = obj.theta_centers
        obj.triplet_theta_edges = obj.theta_edges
        obj.triplet_dtheta = obj.dtheta
        obj.triplet_n_r = obj.n_r

        obj.method_pairs = 'precomputed'
        obj.method_triplets = 'precomputed'
        obj.method_quadruplets = 'precomputed'

        obj.xip = spline_xip
        obj.xim = spline_xim
        obj.sigma_eps = sigma_eps if np.ndim(sigma_eps) else float(sigma_eps)
        obj.nbar = nbar
        obj.area = area
        obj.weights_mean = float(weights_mean)
        obj.weights_std = float(weights_std)
        obj._w1 = obj.weights_mean
        obj._w2 = obj.weights_mean**2 + obj.weights_std**2
        obj.spl_suppression_ww = getattr(template, 'spl_suppression_ww', None)

        # Integration grids and tiling from the template, else the precomputed defaults
        for name, default in (('phi_grid', None), ('varphi1_grid', None),
                              ('varphi2_grid', None), ('phi1_subgrid_frac', None),
                              ('phi2_subgrid_frac', None), ('v2_block', 16),
                              ('nr_block', None)):
            setattr(obj, name, getattr(template, name, default))
        obj.nthreads = 1 if nthreads is None else int(nthreads)

        # Count tensors not available for precomputed instances
        obj.pairs = None if pairs is None else np.asarray(pairs, dtype=float)
        obj.triplets = None
        obj.quadruplets = None

        def _a(x):
            return np.asarray(x, dtype=float) if x is not None else None

        obj.D = _a(D)
        obj.qpp = _a(qpp); obj.qmm = _a(qmm); obj.qpm = _a(qpm)
        obj.rp0 = _a(rp0); obj.rp1 = _a(rp1); obj.rm0 = _a(rm0); obj.rm1 = _a(rm1); obj.rpm = _a(rpm)
        obj.cov_pp = None; obj.cov_mm = None; obj.cov_pm = None
        obj.cov_xipm = None

        for name, value in tomo.items():
            setattr(obj, name, value)
        if tomo:
            obj.cov_pp_tomo = None
            obj.cov_mm_tomo = None
            obj.cov_pm_tomo = None
            obj.cov_xipm_tomo = None
            obj.cov_terms_tomo = {}

        return obj

    #######################
    # Covariance of xi_pm #
    #######################

    def get_cov_xipm(self, verbose=False, nthreads=None):
        """Assemble the full (2n_r, 2n_r) covariance of (xi_+, xi_-)."""
        # All five r-terms in one pass of the quadruplet integral
        self._ensure_all_rterms(verbose=verbose, nthreads=nthreads)
        if self.cov_pp is None: self.get_cov_pp(verbose=verbose, nthreads=nthreads)
        if self.cov_mm is None: self.get_cov_mm(verbose=verbose, nthreads=nthreads)
        if self.cov_pm is None: self.get_cov_pm(verbose=verbose, nthreads=nthreads)
        n = self.n_r
        self.cov_xipm = np.zeros((2 * n, 2 * n))
        self.cov_xipm[:n, :n] = self.cov_pp
        self.cov_xipm[:n, n:] = self.cov_pm
        self.cov_xipm[n:, :n] = self.cov_pm.T
        self.cov_xipm[n:, n:] = self.cov_mm
        return self.cov_xipm

    def get_cov_pp(self, verbose=False, nthreads=None):
        """Assemble the full Cov_pp matrix."""
        return self._get_cov_block(0, verbose, nthreads)

    def get_cov_mm(self, verbose=False, nthreads=None):
        """Assemble the full Cov_mm matrix."""
        return self._get_cov_block(1, verbose, nthreads)

    def get_cov_pm(self, verbose=False, nthreads=None):
        """Assemble the full Cov_pm matrix. No diagonal shape-noise term."""
        return self._get_cov_block(2, verbose, nthreads)

    def _get_cov_block(self, index, verbose=False, nthreads=None):
        """Assemble one of Cov_pp / Cov_mm / Cov_pm, by position in COVBLOCKS."""
        block, label, qname, rnames, with_D = self.COVBLOCKS[index]
        out = np.zeros((self.n_r, self.n_r))
        if with_D:
            if verbose: print("Doing cov_%s: D" % label)
            if self.D is None: self.getD()
            out += np.diag(self.D)
        if verbose: print("Doing cov_%s: %s" % (label, qname))
        if getattr(self, qname) is None:
            getattr(self, 'get' + qname)()
        out += getattr(self, qname)
        for rname in rnames:
            if verbose: print("Doing cov_%s: %s" % (label, rname))
            if getattr(self, rname) is None:
                getattr(self, 'get' + rname)(verbose=verbose, nthreads=nthreads)
            out += getattr(self, rname)
        setattr(self, 'cov_' + block, out)
        return out

    def getD(self):
        """Diagonal shape-noise term, via :meth:`getD_continuous` or :meth:`getD_discrete`."""
        self.D = (self.getD_continuous() if self.method_pairs == 'continuous'
                  else self.getD_discrete())
        return self.D

    def getqpp(self):
        """Mixed term for Cov_pp.S+02 Eq. 32. """
        return self._get_qterm(0)

    def getqmm(self):
        """Mixed term for Cov_mm. S+02 Eq. 36."""
        return self._get_qterm(1)

    def getqpm(self):
        """Mixed term for Cov_pm. S+02 Eq. 38."""
        return self._get_qterm(2)

    def _get_qterm(self, index):
        """One mixed term, by its position in :attr:`QTERMS`."""
        name, weight_type, xi_pm = self.QTERMS[index]
        if self.method_triplets == 'continuous':
            value = self._qterms_continuous_c(
                weight_type, getattr(self, 'xi' + xi_pm))
        else:
            value = self._triplet_term(weight_type, getattr(self, 'xi' + xi_pm))
        setattr(self, name, value)
        return value

    def get_all_rterms(self, verbose=False, nthreads=None, pair_mask=None):
        """Compute rp0, rp1, rm0, rm1 and rpm in a single pass of the quadruplet integral.

        Parameters
        ----------
        verbose: bool, optional
            Print the progress of the integral. Defaults to ``False``.
        nthreads: int, optional
            Number of OpenMP threads. Defaults to ``nthreads`` of the instance.
        pair_mask: numpy.ndarray or dict, optional
            Boolean mask of shape (n_r, n_r) that restricts the integral to a band around the
            diagonal, or a dict of one such mask per block ``'pp'``, ``'mm'`` and ``'pm'``. The
            masked r-terms are returned but not stored.

        Returns
        -------
        rp0, rp1, rm0, rm1, rpm: numpy.ndarray
            The five r-terms, each of shape (n_r, n_r).
        """
        if self.method_quadruplets == 'continuous':
            raise NotImplementedError(
                "get_all_rterms() is not supported for "
                "method_quadruplets='continuous'; call "
                "getrp0()...getrpm() individually.")
        results = self._quadruplet_density_terms_c(
            verbose=verbose, pair_mask=pair_mask, nthreads=nthreads)
        if pair_mask is None:
            self.rp0, self.rp1, self.rm0, self.rm1, self.rpm = results
        return tuple(results)

    def getrp0(self, verbose=False, nthreads=None):
        """r_+0 contribution to Cov_pp cosmic variance."""
        return self._get_rterm(0, verbose, nthreads)

    def getrp1(self, verbose=False, nthreads=None):
        """r_+1 contribution to Cov_pp cosmic variance."""
        return self._get_rterm(1, verbose, nthreads)

    def getrm0(self, verbose=False, nthreads=None):
        """r_-0 contribution to Cov_mm cosmic variance."""
        return self._get_rterm(2, verbose, nthreads)

    def getrm1(self, verbose=False, nthreads=None):
        """r_-1 contribution to Cov_mm cosmic variance."""
        return self._get_rterm(3, verbose, nthreads)

    def getrpm(self, verbose=False, nthreads=None):
        """r_+- contribution to the Cov_pm cosmic variance"""
        return self._get_rterm(4, verbose, nthreads)

    def _get_rterm(self, index, verbose=False, nthreads=None):
        """One r-term by its position in :attr:`RTERMS`, computed with the other four if missing."""
        self._ensure_all_rterms(verbose=verbose, nthreads=nthreads)
        return getattr(self, self.RTERMS[index])

    def _ensure_all_rterms(self, verbose=False, nthreads=None):
        """Compute all five r-terms if any of them is missing."""
        if all(getattr(self, n) is not None for n in self.RTERMS):
            return
        if self.method_quadruplets == 'continuous':
            for i, name in enumerate(self.RTERMS):
                setattr(self, name, self._rterms_continuous_c()[i])
        else:
            self.get_all_rterms(verbose=verbose, nthreads=nthreads)

    ########################
    # Discrete expressions #
    ########################

    def getD_discrete(self):
        r"""Shape-noise term on the diagonal from the pair counts. S+02 Eq. 27.

        .. math::

            D(\theta) = \frac{\sigma_\epsilon^4 \langle w^2\rangle^2}{N_p(\theta)\langle w\rangle^2}
        """
        if self.pairs is None:
            raise RuntimeError("`pairs` not initialised.")
        if self.pairs_wsq is not None:
            self.D = self.sigma_eps**4 * self.pairs_wsq / self.pairs**2
            return self.D
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        self.D = self.sigma_eps**4 * ww_factor / self.pairs
        return self.D

    def _triplet_term(self, weight_type, xi_func, sigma_eps_sq=None, nbar_fac=1.):
        """Mixed term q_++, q_-- or q_+- from the triplet counts."""
        if self.triplets is None:
            raise RuntimeError("`triplets` not initialised.")
        if sigma_eps_sq is None:
            sigma_eps_sq = self.sigma_eps**2
        pre = 2. * sigma_eps_sq * nbar_fac / np.outer(self.pairs, self.pairs)
        fine = self._triplet_phi_reduce(self.triplets, self.triplet_theta_centers, self.phi_centers, weight_type, xi_func)
        if self.triplet_oversample > 1:
            fine = fine.reshape(self.n_r, self.triplet_oversample, self.n_r, self.triplet_oversample).sum(axis=(1, 3))
        return pre * fine

    def _triplet_phi_reduce(self, Nt, tc, phic, weight_type, xi_func):
        """Sum of Nt[i, j, k] w(phi_k) xi(r_jk) over k on the fine grid, via the C kernel."""
        nr = len(tc)
        # Set r_min by smallest third side of a triplet --> 2 theta_min*sin(phi_min/2)
        r_min = 2.*tc[0]*np.min(np.abs(np.sin(.5*np.asarray(phic))))
        xi_x, xi_y, logx0, inv_dlogx, xi_n = self._xi_tables_from_splines(
            [xi_func], lo=min(tc[0], self._xi_query_lo(), .9*r_min), hi=2.*tc[-1])
        out = np.zeros((nr, nr), dtype=np.float64)
        self._clib.triplet_term_reduce(
            np.ascontiguousarray(Nt, dtype=np.float64), nr, len(phic),
            np.ascontiguousarray(tc, dtype=np.float64),
            np.ascontiguousarray(phic, dtype=np.float64),
            int(weight_type), xi_x, xi_n, logx0, inv_dlogx, xi_y[0],
            out, int(self.nthreads), 0)
        check_clib_error(self._clib)
        return out

    def _quadruplet_density_terms_c(self, verbose=False, pair_mask=None,
                                    nthreads=None):
        """Non-tomographic r-terms (rp0, rp1, rm0, rm1, rpm) via the C kernel.

        xi_pm enter through the piecewise-linear table of the kernel, which is exact for a linear
        ``interp1d``.
        """
        xi_tables = self._xi_tables_from_splines(
            [self.xip, self.xim], lo=self._xi_query_lo(), hi=self._xi_query_hi())
        # Nontomo callback of five rbins --> As notomo, Wick factor is 1. Tables 0 and 1 are xi_+ and xi_-
        callback_map = dict(xi_a_idx=['pm'.index(self.RTERM_XI[name][0]) for name in self.RTERMS],
                            xi_b_idx=['pm'.index(self.RTERM_XI[name][1]) for name in self.RTERMS],
                            w_ids=[0, 1, 2, 3, 4], out_idx=[0, 1, 2, 3, 4],
                            wick_fac=[1.] * 5, n_out=5)
        return self._run_postq_engine(
            callback_map, xi_tables, [name in self.RTERMS_SYM for name in self.RTERMS],
            [self.RTERM_BLOCKS[name] for name in self.RTERMS],
            verbose=verbose, pair_mask=pair_mask, nthreads=nthreads)

    def _run_postq_engine(self, cbm, xi_tables, symmetric_mask, out_blocks,
                          verbose=False, pair_mask=None, nthreads=None):
        """Run the fused C quadruplet kernel over the phi grid.

        ``out_blocks`` is the covariance block of each output
        ``pair_mask`` is the mask of the kept elements.
        """

        ## 1) Misc preparations
        if nthreads is None:
            nthreads = self.nthreads
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = xi_tables
        n_out = cbm['n_out']
        n = self.n_r

        def _f(a): return np.ascontiguousarray(a, dtype=np.float64)
        def _i(a): return np.ascontiguousarray(a, dtype=np.int32)

        ## 2) Define radial/angular integration nodes and their weights
        if self.phi1_subgrid_frac is None:
            phi1_nodes = self.theta_centers[:, None]
            phi1_weights = _theta_dtheta(self.theta_edges)[:, None]
        else:
            s1f = self.phi1_subgrid_frac
            phi1_nodes = self.theta_centers[:, None] + np.outer(self.dtheta, s1f)
            phi1_weights = phi1_nodes * self.dtheta[:, None] * _trapz_weights(s1f)[None, :]
        if self.phi2_subgrid_frac is None:
            phi2_nodes = self.theta_centers[:, None]
            phi2_weights = _theta_dtheta(self.theta_edges)[:, None]
        else:
            s2f = self.phi2_subgrid_frac
            phi2_nodes = self.theta_centers[:, None] + np.outer(self.dtheta, s2f)
            phi2_weights = phi2_nodes * self.dtheta[:, None] * _trapz_weights(s2f)[None, :]
        ns1, ns2 = phi1_nodes.shape[1], phi2_nodes.shape[1]
        phi1_flat = phi1_nodes.ravel()
        phi2_flat = phi2_nodes.ravel()

        phi_grid = self.phi_grid
        phi_w = _trapz_weights(phi_grid)
        v1_w = _periodic_weights(self.varphi1_grid)
        v2_w = _periodic_weights(self.varphi2_grid)

        ## 3) Init output masks
        if isinstance(pair_mask, dict):
            blocks = sorted(set(out_blocks))
            masks = [pair_mask[b] for b in blocks]
            mask_idx = [blocks.index(b) for b in out_blocks]
        else:
            masks = [np.ones((n, n)) if pair_mask is None else pair_mask]
            mask_idx = [0] * n_out
        masks, mask_idx = _i(masks), _i(mask_idx)

        ## 4) Tile the plan over the (theta1, theta2) axes, respecting the mask
        union = masks.any(axis=0)
        nrb = self.nr_block or n
        rows = range(0, n, nrb)
        tiles = [(i1, i2) for i1 in rows if union[i1:i1 + nrb].any()
                 for i2 in rows if union[i1:i1 + nrb, i2:i2 + nrb].any()]

        ## 5) Define quadruplet density representation
        # 0 = tensor, 1 = constant, 2 = separable s(t) ##
        dummy = np.zeros(2)
        if self.quadruplets is None:
            raise RuntimeError("`quadruplets` not initialised.")
        # Quadruplets passed as array --> Tensor mode
        if not callable(self.quadruplets):
            q_mode, q_const = 0, 0.
            q_array = _f(self.quadruplets)
            q_rc, q_pc = _f(self.q_theta_centers), _f(self.q_phi_centers)
            q_nr, q_nphi = len(q_rc), len(q_pc)
            sup_t1 = sup_t2 = sup_t3 = dummy
        else:
            if self.method_quadruplets not in ('analytic', 'frompairs'):
                raise NotImplementedError(
                    "the quadruplet integral takes a density tensor or one of "
                    "the analytic densities, not a user-supplied callable; "
                    "materialise it with get_quadruplets() and pass the "
                    "result as `quadruplets`.")
            q_array, q_rc, q_pc = dummy, dummy, dummy
            q_nr = q_nphi = 0
            q_const = self._q_const
            # Callable quadruplets + no supression --> Constant mode
            if self.spl_suppression_ww is None:
                q_mode = 1
                sup_t1 = sup_t2 = sup_t3 = dummy
            # Callable quadruplets + no supression --> frompairs mode
            # We use s(phi) s(theta1) s(theta2) of the link between the pairs and of the two pairs
            else:
                q_mode = 2
                sup = self.spl_suppression_ww
                sup_t1, sup_t2 = _f(sup(phi_grid)), _f(sup(phi1_flat))
                sup_t3 = _f(np.broadcast_to(sup(phi2_flat)[None, :, None],
                                            (len(phi_grid), len(phi2_flat), len(self.varphi2_grid))))

        ## 6) Callbacks sorted by (weight, xi_b), whose varphi2 sums the kernel shares ##
        order = np.lexsort((cbm['xi_b_idx'], cbm['w_ids']))
        cb = {key: np.asarray(cbm[key])[order]
              for key in ('xi_a_idx', 'xi_b_idx', 'w_ids', 'out_idx', 'wick_fac')}
        
        ## 7) Do the function call and rescale the output
        results = np.zeros((n_out, n, n))
        self._clib.quadruplet_sva_integrate(
            _f(phi_grid), _f(phi_w), len(phi_grid), n,
            _f(phi1_flat), _f(phi1_weights.ravel()), ns1,
            _f(phi2_flat), _f(phi2_weights.ravel()), ns2,
            _f(self.varphi1_grid), _f(v1_w), len(self.varphi1_grid),
            _f(self.varphi2_grid), _f(v2_w), len(self.varphi2_grid),
            nrb, self.v2_block, _i(np.reshape(tiles, (-1, 2))), len(tiles),
            q_mode, q_array, q_rc, q_pc, q_nr, q_nphi,
            q_const, sup_t1, sup_t2, sup_t3,
            _f(xi_x), xi_n, xi_logx0, xi_inv_dlogx,
            _f(xi_y_tables), len(xi_y_tables),
            len(cb['xi_a_idx']), _i(cb['xi_a_idx']), _i(cb['xi_b_idx']),
            _i(cb['w_ids']), _i(cb['out_idx']), _f(cb['wick_fac']),
            n_out, _i(symmetric_mask), len(masks), masks, mask_idx,
            results, nthreads, int(verbose))
        check_clib_error(self._clib)

        pre = 2.*np.pi*self.area / np.outer(self.pairs, self.pairs)
        return [pre*result for result in results]

    ##########################
    # Continuous expressions #
    ##########################

    def getD_continuous(self):
        r"""Continuous D,  S+02 Eq. 27."""
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        return self.sigma_eps**4 * ww_factor / self.pairs

    def getqpp_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_++. S+02 Eq. 32."""
        return self._qterms_continuous_c(0, self.xip, nr_target, nphi_target)

    def getqmm_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_--. S+02 Eq. 36."""
        return self._qterms_continuous_c(1, self.xip, nr_target, nphi_target)

    def getqpm_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_+-. S+02 Eq. 38."""
        return self._qterms_continuous_c(2, self.xim, nr_target, nphi_target)

    def getrp0_continuous(self):
        """Continuous r_+0. S+02 Eq. 34."""
        return self._rterms_continuous_c()[0]

    def getrp1_continuous(self):
        """Continuous r_+1. S+02 Eq. 34."""
        return self._rterms_continuous_c()[1]

    def getrm0_continuous(self):
        """Continuous r_-0."""
        return self._rterms_continuous_c()[2]

    def getrm1_continuous(self):
        """Continuous r_-1."""
        return self._rterms_continuous_c()[3]

    def getrpm_continuous(self):
        """Continuous r_+-, (not symmetric under theta1 <-> theta2!)"""
        return self._rterms_continuous_c()[4]

    def _qterms_continuous_c(self, w_type, xi_func, nr_target=1024, nphi_target=1024):
        """Helper to get continuous mixed terms (q_++, q_-- or q_+-) via C kernel."""

        # Prepare args
        alpha = int(nr_target // self.n_r)
        n_fine = alpha * self.n_r
        _edges = np.geomspace(self.theta_edges[0], self.theta_edges[-1], n_fine + 1)
        thetas = .5 * (_edges[1:] + _edges[:-1])
        phis = np.linspace(0., np.pi, nphi_target, endpoint=False)
        # Set r_min by smallest third side of a triplet --> 2 theta_min*sin(phi_min/2)
        r_min = min(thetas[1]-thetas[0], 2.*thetas[0]*np.sin(.5 * phis[1]))
        _xitables = self._xi_tables_from_splines(
            [xi_func], lo=min(self._xi_query_lo(), .9*r_min), hi=2.*self.theta_edges[-1])
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = _xitables
        # Do the call
        out = np.zeros(self.n_r * self.n_r, dtype=np.float64)
        self._clib.continuous_q_single(
            self.n_r, alpha,
            np.ascontiguousarray(thetas, dtype=np.float64),
            np.ascontiguousarray(thetas * np.diff(_edges), dtype=np.float64),
            np.ascontiguousarray(phis, dtype=np.float64),
            np.ascontiguousarray(_trapz_weights(phis), dtype=np.float64),
            n_fine, nphi_target, xi_x, xi_n, xi_logx0, xi_inv_dlogx, np.ascontiguousarray(xi_y_tables[0]),
            int(w_type), float(self.sigma_eps**2), float(self.area), float(self.nbar),
            np.ones(self.n_r, dtype=np.float64), out)
        # Reshape and rescale the output: Shared galaxy carries <w^2>, the other two <w> each
        out = self._w2 / self._w1**2 * out.reshape(self.n_r, self.n_r)
        return out

    def _rterms_continuous_c(self, dlnt_target=.05):
        """Helper to get continuous r-terms via C kernel, averaging out over bins using
        a fine binsize of dlnt_target."""
        # Memoised against the grids, since the getters ask for one term at a time
        cached = getattr(self, '_rterms_cont_cache', None)
        if (cached is not None and cached[0] is self.phi_grid
                and cached[1] is self.varphi1_grid and cached[2] == self.n_r):
            return cached[3]
        # Build args
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = \
            self._xi_tables_from_splines(
                [self.xip, self.xim],
                lo=self._xi_query_lo(), hi=self._xi_query_hi())
        v = np.ascontiguousarray(self.varphi1_grid, dtype=np.float64)
        v_half = np.ascontiguousarray(v[v <= np.pi], dtype=np.float64)
        nodes, weights = self._rterm_nodes(dlnt_target)
        results = np.zeros(5*nodes.size*nodes.size, dtype=np.float64)
        idx0 = np.zeros(1, dtype=np.int32)
        # Do function call
        self._clib.continuous_rterms_tomo(
            nodes.size, np.ascontiguousarray(nodes.ravel(), dtype=np.float64),
            np.ascontiguousarray(self.phi_grid, dtype=np.float64),
            np.ascontiguousarray(_trapz_weights(self.phi_grid), dtype=np.float64),
            len(self.phi_grid),
            v, np.ascontiguousarray(_periodic_weights(v), dtype=np.float64), len(v),
            v_half, np.ascontiguousarray(_trapz_weights(v_half), dtype=np.float64),
            len(v_half),
            float(self.area),
            xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y_tables, 2,
            1, idx0, idx0, idx0+1, idx0+1, idx0,
            5, results, int(self.nthreads), 0)
        # Integrate over the subbins and update cache
        out = _bin_average_blocks(results.reshape(5, nodes.size, nodes.size), weights)
        self._rterms_cont_cache = (self.phi_grid, self.varphi1_grid, self.n_r, out)
        return out

    def _rterm_nodes(self, dlnt_target):
        """Nodes (n_r, ns) and theta dtheta weights of equal log sub-bins no wider than dlnt_target."""
        dlnt = np.diff(np.log(self.theta_edges))
        ns = max(1, int(np.ceil(np.max(dlnt)/dlnt_target - 1e-9)))
        # Single subbin --> Choose bin center as node
        if ns == 1:
            return self.theta_centers[:, None], _theta_dtheta(self.theta_edges)[:, None]
        sub = np.exp(np.log(self.theta_edges[:-1])[:, None] + np.outer(dlnt, np.arange(ns+1)/ns))
        return .5*(sub[:, 1:]+sub[:, :-1]), .5*(sub[:, 1:]**2-sub[:, :-1]**2)

    ####################
    # Multiplet counts #
    ####################

    ## Measured on the catalog ##

    def computeDiscrete(self, contribution):
        """Measure the count tensor of one multiplet on ``catalog`` with the orpheus instances.

        ``contribution`` is ``'pairs'`` (N_p of shape (n_r,), and the pairs of squared weights
        ``pairs_wsq``), ``'triplets'`` (N_t of shape (triplet_n_r, triplet_n_r, n_phi), with the
        squared weight on the base galaxy) or ``'quadruplets'`` (density q of shape (q_n_r, q_n_r,
        q_n_r, q_n_phi, q_n_phi)). The binning follows the covariance grids, and the results are
        cached in ``path_save`` under a hash of the inputs.
        """
        if self.catalog is None or self.path_save is None:
            raise ValueError(
                f"computeDiscrete('{contribution}') requires `catalog` and `path_save`")
        key, binning = {
            'pairs': ('NN', (self.theta_edges[0], self.theta_edges[-1], self.n_r)),
            'triplets': ('NNN', (self.theta_edges[0], self.theta_edges[-1], self.triplet_n_r,
                                 self.n_phi)),
            'quadruplets': ('NNNN', (self.q_theta_edges[0], self.q_theta_edges[-1], self.q_n_r,
                                     self.q_n_phi))}[contribution]
        grid = self._MULTIPLET_GRIDS[contribution]

        ## Cache file, hashing the catalog, the binning and the correlator setup ##
        # area enters the cached quadruplet density
        setup = (self.geometry, list(np.ravel(self.nmaxs)), self.rmin_pixsize,
                 list(np.ravel(self.tree_resos)), self.tree_method, int(self.shuffle_pix), self.area)
        h = self._catalog_sha1.copy()
        h.update(repr((contribution, tuple(map(float, binning)), setup)).encode())
        fpath = os.path.join(self.path_save, "%s_%s.npz" % (contribution, h.hexdigest()[:16]))
        if os.path.exists(fpath):
            counts = self._load_multiplet(contribution, fpath)
        else:
            if key not in self.orpheus_instances:
                self.set_orpheus_instances([contribution])
            # Taken off the instance, so that its multipoles are released after the measurement
            corr = self.orpheus_instances.pop(key)
            cat = self.orpheus_instances['catalog']
            extra = {}
            if contribution == 'pairs':
                # The pairs of squared weights first, so that the bin centers are those of N_p.
                # npair holds the ordered pairs, both (i,j) and (j,i).
                corr.process(self.orpheus_instances['catalog_wsq'])
                self.pairs_wsq = extra['pairs_wsq'] = np.array(corr.npair[0], dtype=float)
                corr.process(cat)
            if contribution == 'triplets':
                corr.process(cat)
            if contribution == 'quadruplets':
                corr.process(cat, lowmem=True, statistics=["4pcf_multipole"],
                             memory_bound=self.memory_bound)

            # Empty bins keep the nominal centers, also for the real-space conversion
            fac = 60. if contribution == 'quadruplets' and self.geometry == 'spherical' else 1.
            bc = fac*np.asarray(corr.bin_centers_mean, dtype=float)
            empty = bc == 0.
            bc[empty] = getattr(self, grid)[empty]
            corr.bin_centers_mean[empty] = bc[empty]/fac
            setattr(self, grid, bc)

            if contribution == 'pairs':
                counts = np.array(corr.npair[0], dtype=float)
            if contribution == 'triplets':
                corr.multipoles2npcf()
                counts = np.array(corr.npcf_norm[0], dtype=float)
            if contribution == 'quadruplets':
                corr.multipoles2npcf()
                # Density q = N_q / V_q with a separable volume, divided axis by axis
                tdt = self.q_theta_centers*self.q_dtheta
                dp = self.q_dphi
                counts = corr.npcf[:, :, :, 0].real/(2.*np.pi*self.area)
                counts /= tdt[:, None, None, None, None]
                counts /= tdt[None, :, None, None, None]
                counts /= tdt[None, None, :, None, None]
                counts /= dp[None, None, None, :, None]
                counts /= dp[None, None, None, None, :]
            if self.store_multiplets:
                np.savez(fpath, **{contribution: counts}, bin_centers_mean=bc, area=self.area,
                         **extra)
        self._check_shape(contribution, counts)
        return counts

    def set_orpheus_instances(self, contributions=('pairs', 'triplets', 'quadruplets')):
        """Init the catalogs and correlators that measure the ``'discrete'`` multiplets."""
        import hashlib
        from .catalog import ScalarTracerCatalog
        from .npcf_second import NNCorrelation
        from .npcf_third import NNNCorrelation
        from .npcf_fourth import NNNNCorrelation_NoTomo

        # Make sure bins are in logspace
        for name, kinds in (('theta_edges', {'pairs', 'triplets'}),
                            ('q_theta_edges', {'quadruplets'})):
            if kinds & set(contributions):
                dlog = np.diff(np.log(getattr(self, name)))
                if not np.allclose(dlog, dlog[0], rtol=1e-3):
                    raise ValueError(f"{name} must be log-uniform to be measured on a catalog")
        if self.geometry not in ('spherical', 'flat2d'):
            raise ValueError(f"geometry must be 'spherical' or 'flat2d', got '{self.geometry}'")
        sphere = self.geometry == 'spherical'

        ## Catalogs, built once, and the hash of their positions for the cache tags ##
        ra = np.asarray(self.catalog['ra'], dtype=float)
        dec = np.asarray(self.catalog['dec'], dtype=float)
        w = self.catalog.get('weight')
        w = np.ones_like(ra) if w is None else np.asarray(w, dtype=float)
        # A flat catalog carries the unit of the separations, arcmin
        pos = (dict(geometry='spherical', units_pos1='deg', units_pos2='deg', pos1=ra, pos2=dec)
               if sphere else dict(geometry='flat2d', pos1=60.*ra, pos2=60.*dec))
        if 'catalog' not in self.orpheus_instances:
            self._catalog_sha1 = hashlib.sha1()
            for a in (ra, dec, w):
                self._catalog_sha1.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
            self.orpheus_instances['catalog'] = ScalarTracerCatalog(
                weight=w, tracer=np.ones_like(ra), **pos)
        # The pairs of squared weights enter the shape noise
        if 'pairs' in contributions and 'catalog_wsq' not in self.orpheus_instances:
            self.orpheus_instances['catalog_wsq'] = ScalarTracerCatalog(
                weight=w**2, tracer=np.ones_like(ra), **pos)

        ## Correlators, where the spherical NNNN works in deg ##
        kw = dict(rmin_pixsize=self.rmin_pixsize, shuffle_pix=self.shuffle_pix,
                  process_spherical=sphere, verbosity=self.verbosity, nthreads=self.nthreads)
        xi_kw = dict(min_sep=float(self.theta_edges[0]), max_sep=float(self.theta_edges[-1]),
                     tree_resos=self.tree_resos, **kw)
        if 'pairs' in contributions:
            self.orpheus_instances['NN'] = NNCorrelation(nbinsr=self.n_r, **xi_kw)
        if 'triplets' in contributions:
            self.orpheus_instances['NNN'] = NNNCorrelation(
                nbinsr=self.triplet_n_r, nmaxs=self.nmaxs, nbinsphi=self.n_phi, base_weightsq=True,
                **xi_kw)
        if 'quadruplets' in contributions:
            fac = 60. if sphere else 1.
            self.orpheus_instances['NNNN'] = NNNNCorrelation_NoTomo(
                min_sep=float(self.q_theta_edges[0])/fac, max_sep=float(self.q_theta_edges[-1])/fac,
                nbinsr=self.q_n_r, nbinsphi=self.q_n_phi, nmaxs=self.nmaxs, method=self.tree_method,
                tree_resos=self.tree_resos/fac,
                sep_units='deg' if sphere else 'arcmin', **kw)

    def _load_multiplet(self, kind, fpath):
        """Loads a multiplet from a file written by ``store_multiplets`` and update relevant attributes."""
        dat = np.load(fpath)
        setattr(self, self._MULTIPLET_GRIDS[kind], dat['bin_centers_mean'])
        if kind == 'pairs':
            self.pairs_wsq = np.asarray(dat['pairs_wsq'], dtype=float)
        return np.asarray(dat[kind], dtype=float)

    def set_quadruplets_from_counts(self, N_q,
                                    theta_centers=None,
                                    dtheta_bin=None,
                                    dphi_bin=None,
                                    return_volume=False):
        r"""Convert quadruplet counts to the density q = N_q / V_q and store it.

        .. math::

            V_q = 2\pi A\, \theta_1 \Delta\theta_1\, \theta_2 \Delta\theta_2\,
            \theta_3 \Delta\theta_3\, \Delta\phi_{12} \Delta\phi_{13}

        The grids default to the quadruplet grids of the instance.
        """
        rc =(self.q_theta_centers if theta_centers is None
              else np.asarray(theta_centers, dtype=float))
        dt = (self.q_dtheta if dtheta_bin is None
              else np.asarray(dtheta_bin, dtype=float))
        dp = (self.q_dphi if dphi_bin is None
              else np.asarray(dphi_bin, dtype=float))
        # V_q is separable, so divide axis by axis
        tdt = rc * dt
        q = np.array(N_q, dtype=float)
        q /= 2. * np.pi * self.area
        q /= tdt[:, None, None, None, None]
        q /= tdt[None, :, None, None, None]
        q /= tdt[None, None, :, None, None]
        q /= dp[None, None, None, :, None]
        q /= dp[None, None, None, None, :]
        self.quadruplets = q

        if return_volume:
            return (2. * np.pi * self.area
                    * tdt[:, None, None, None, None]
                    * tdt[None, :, None, None, None]
                    * tdt[None, None, :, None, None]
                    * dp[None, None, None, :, None]
                    * dp[None, None, None, None, :])

    def get_quadruplets(self):
        """Return the quadruplet density of shape (q_n_r, q_n_r, q_n_r, q_n_phi, q_n_phi).

        An analytic density is evaluated on the quadruplet grid.
        """
        if callable(self.quadruplets):
            rc, pc = self.q_theta_centers, self.q_phi_centers
            shape = (len(rc),) * 3 + (len(pc),) * 2
            vals = self.quadruplets(
                rc[:, None, None, None, None], rc[None, :, None, None, None],
                rc[None, None, :, None, None], pc[None, None, None, :, None],
                pc[None, None, None, None, :])
            return np.broadcast_to(vals, shape).copy()
        return self.quadruplets

    ## Idealised ##

    def pairs_analytic(self):
        r"""Idealised pair counts."""
        if self.nbar is None:
            raise ValueError("`nbar` is required for analytic pairs")
        return (2.*np.pi*self.area*self.nbar**2 * _theta_dtheta(self.theta_edges)
                * self._w1**2 * self._suppression(self.theta_centers))

    def triplets_analytic(self):
        r"""Idealised triplet counts on the oversampled grid."""
        if self.nbar is None:
            raise ValueError("`nbar` is required for analytic triplets")
        T1, T2, PHI = np.meshgrid(self.triplet_theta_centers,
                                   self.triplet_theta_centers,
                                   self.phi_centers, indexing='ij')
        M1, M2, DPHI = np.meshgrid(_theta_dtheta(self.triplet_theta_edges),
                                   _theta_dtheta(self.triplet_theta_edges),
                                   self.dphi, indexing='ij')
        s_factor = (self._suppression(T1) * self._suppression(T2))
        return (2.*np.pi*self.area * M1*M2*DPHI * self.nbar**3 * self._w2*self._w1**2 * s_factor)

    def quadruplets_analytic(self):
        """Idealised quadruplet density, returned as a callable. """
        # Make sure that the spline covers all scales
        if self.method_quadruplets == 'frompairs':
            if self.spl_suppression_ww is None:
                raise ValueError(
                    "method_quadruplets='frompairs' requires "
                    "`spl_suppression_ww` to be set -- typically derived "
                    "from / calibrated against the measured pair-count "
                    "suppression N_p^measured(θ) / N_p^ideal(θ).")
            spl_x = getattr(self.spl_suppression_ww, 'x', None)
            if spl_x is not None:
                lo, hi = float(np.min(spl_x)), float(np.max(spl_x))
                q_lo = float(np.min(self.q_theta_centers))
                q_hi = float(np.max(self.q_theta_centers))
                if q_lo < lo or q_hi > hi:
                    warnings.warn(
                        "method_quadruplets='frompairs': spl_suppression_ww "
                        f"is tabulated on [{lo:.4g}, {hi:.4g}] arcmin but "
                        f"q_theta_centers spans [{q_lo:.4g}, {q_hi:.4g}] "
                        "arcmin -- queries outside the spline's support "
                        "will silently extrapolate (risking unphysical, "
                        "e.g. negative, suppression values).",
                        RuntimeWarning, stacklevel=2)
        # Fix q_const here, before GGCovariance replaces self.nbar by its per-bin array
        const = self.nbar**4 * self._w1**4
        self._q_const = const
        # Get the callable
        if self.spl_suppression_ww is None:
            def q_func(t1, t2, t3, a12, a13):
                return const
        else:
            # We set the suppression vectors to mimic the SVA geo in terms of the quadruplet geo
            sup = self.spl_suppression_ww
            def q_func(t1, t2, t3, a12, a13):
                return const * (sup(t1) * sup(t2) * sup(np.hypot(t1 - t3*np.cos(a13), t3*np.sin(a13))))
        return q_func

    @staticmethod
    def suppression_spline_from_pairs(Np_meas, theta_centers, theta_edges,
                                       area, nbar, weights_mean=1.):
        """Pair-count suppression s(theta) = N_p(theta) / N_p^ideal(theta) of measured pairs."""
        # Get suppression on support of pairs
        theta_centers = np.asarray(theta_centers, dtype=float)
        theta_edges = np.asarray(theta_edges, dtype=float)
        ideal = 2.*np.pi*area*nbar**2 * weights_mean**2 * _theta_dtheta(theta_edges)
        supp_vals = np.asarray(Np_meas, dtype=float) / ideal
        # Build callable spline with extensions
        supp_spl = interp1d(np.log(theta_centers), supp_vals,
                            bounds_error=False, fill_value=(supp_vals[0], 0.))
        def sup(theta):
            return supp_spl(np.log(np.asarray(theta, dtype=float)))
        sup.x = theta_centers
        return sup

    ###################
    # General helpers #
    ###################

    @staticmethod
    def _xi_tables_from_splines(splines, lo=None, hi=None):
        """Tabulate xi splines on shared log-uniform knots."""
        # Fix interpolation grid
        x = getattr(splines[0], 'x', None)
        if x is None:
            if lo is None or hi is None:
                raise TypeError(
                    "xi splines carry no `.x` knots and no sampling range was "
                    "given; pass an interp1d/spline or supply lo/hi.")
            x = np.geomspace(lo, hi, 4096)
        xi_x = np.ascontiguousarray(x, dtype=np.float64)
        log_xi_x = np.log(xi_x).astype(float)
        xi_n = len(xi_x)
        # Build table
        xi_y_tables = np.empty((len(splines), xi_n), dtype=np.float64)
        for t, spl in enumerate(splines):
            xi_y_tables[t] = spl(xi_x)
        return (xi_x, np.ascontiguousarray(xi_y_tables),
                log_xi_x[0], 1./(log_xi_x[1]-log_xi_x[0]) , xi_n)

    def _xi_query_lo(self):
        """Smallest separation at which the quadruplet kernels evaluate xi_pm."""
        return min(self.theta_edges[0], self.phi_grid[0])

    def _xi_query_hi(self):
        """Largest separation at which the quadruplet kernels evaluate xi_pm."""
        return self.phi_grid[-1] + self.theta_edges[-1]

    def _suppression(self, r):
        """Suppression s(r) of the idealised counts, 1 if not set."""
        if self.spl_suppression_ww is None:
            return np.ones_like(np.asarray(r, dtype=float))
        return self.spl_suppression_ww(r)

    def _check_shape(self, kind, arr):
        expect = {'pairs': (self.n_r,),
                    'triplets': (self.triplet_n_r, self.triplet_n_r, self.n_phi),
                    'quadruplets': (self.q_n_r, self.q_n_r, self.q_n_r, self.q_n_phi,
                                    self.q_n_phi)}[kind]
        if arr.shape != expect:
            raise ValueError(f"{kind} shape {arr.shape} != expected {expect}")


##########################
# Tomographic covariance #
##########################

class GGCovariance(GGCovarianceNoTomo):
    """Tomographic covariance of the shear 2PCF between all pairs of tomographic bin pairs.

    The quadruplet density is shared by all bin combinations, only xi_pm change between them.

    Parameters
    ----------
    spline_xip, spline_xim: callable or list of (i, j, callable)
        xi_pm of each bin pair with i <= j. A single callable enforces ntomo = 1.
    nbar_ref: float, optional
        Density of the multiplets from which the tomographic terms are rescaled, e.g. the
        density of the measured catalog. Defaults to ``None``, i.e. no rescaling.
    **kwargs
        All other parameters of :class:`GGCovarianceNoTomo`, where ``sigma_eps`` and ``nbar``
        can be given per bin.
    """

    def __init__(self, *, spline_xip=None, spline_xim=None,
                 nbar_ref=None, **kwargs):
        self._xip, self._xim, self.ntomo = self._parse_xi_splines(
            spline_xip, spline_xim)
        self.n_tomopairs = self.ntomo * (self.ntomo + 1) // 2
        self.tomopairs_list = [(i, j)
                        for i in range(self.ntomo)
                        for j in range(i, self.ntomo)]

        ## Per-bin sigma_eps and nbar ##
        # The parent gets the first entries, which are replaced by the arrays afterwards
        sigma_eps_input = kwargs.get('sigma_eps', 0.27)
        se = np.atleast_1d(np.asarray(sigma_eps_input, dtype=float))
        if se.size == 1:
            se = np.full(self.ntomo, se[0])
        if se.size != self.ntomo:
            raise ValueError(
                f"sigma_eps must be a scalar or length-{self.ntomo} array, "
                f"got length {se.size}")
        kwargs['sigma_eps'] = float(se[0])

        nbar_input = kwargs.get('nbar', None)
        if nbar_input is not None:
            nb = np.atleast_1d(np.asarray(nbar_input, dtype=float))
            if nb.size == 1:
                nb = np.full(self.ntomo, nb[0])
            if nb.size != self.ntomo:
                raise ValueError(
                    f"nbar must be a scalar or length-{self.ntomo} array, "
                    f"got length {nb.size}")
            # The idealised multiplets of the parent live at the reference density
            kwargs['nbar'] = float(nb[0]) if nbar_ref is None else float(nbar_ref)
        else:
            nb = None

        # Set before super().__init__(), whose measurements store it
        self.nbar_ref = float(nbar_ref) if nbar_ref is not None else None

        # The parent gets the (0, 0) xi_pm, which the tomographic terms do not use
        super().__init__(
            spline_xip=self._xip.get((0, 0), spline_xip),
            spline_xim=self._xim.get((0, 0), spline_xim),
            **kwargs)

        # Override parent's scalars with per-bin arrays
        self.sigma_eps = se
        if nb is not None:
            self.nbar = nb

        ## Tomo output storage ##
        self.rterms_tomo = {}
        self.D_tomo = None
        self.qpp_tomo = None
        self.qmm_tomo = None
        self.qpm_tomo = None
        self.cov_pp_tomo = None
        self.cov_mm_tomo = None
        self.cov_pm_tomo = None
        self.cov_xipm_tomo = None
        self.cov_terms_tomo = {}

    @staticmethod
    def _parse_xi_splines(raw_xip, raw_xim):
        """Convert the xi inputs to dicts keyed by (i, j) with i <= j and infer ntomo.

        The bin indices must be consecutive and start at 0.
        """
        def _to_dict(raw):
            if raw is None:
                return {}
            if callable(raw):
                return {(0, 0): raw}
            d = {}
            for entry in raw:
                i, j, spl = entry
                key = (min(i, j), max(i, j))
                d[key] = spl
            return d

        xip = _to_dict(raw_xip)
        xim = _to_dict(raw_xim)

        # Infer ntomo from whichever dict is non-trivial.
        n_tp = max(len(xip), len(xim))
        if n_tp == 0:
            return xip, xim, 1
        ntomo = _infer_ntomo(n_tp)
        return xip, xim, ntomo

    ###################################
    # Covariance of any GG statistics #
    ###################################

    def cov_of_statistic_tomo(self, filter_type, split_components=False,
                              xipm_nbinstarget=None, **kwargs):
        """Covariance of a statistic that is linear in xi_pm, for all tomographic bin pairs.

        The weights of :meth:`GGCovarianceNoTomo.cov_of_statistic` are computed once and applied to
        each block.

        Parameters
        ----------
        filter_type: {'Map2', 'cosebis', 'puremode'}
            The statistic.
        split_components: bool, optional
            Also return the contributions of ``'D'``, ``'qpp'``, ``'qmm'``, ``'qpm'``,
            ``'rp0'``, ``'rp1'``, ``'rm0'``, ``'rm1'``, ``'rpm'``, ``'mix'`` and ``'sva'``.
            Defaults to ``False``.
        xipm_nbinstarget: int, optional
            Number of log-uniform xi_pm bins of the statistic, interpolated with
            :meth:`subbin_tomo`. Defaults to ``None``, i.e. the binning of this instance.
        **kwargs
            Further parameters of :meth:`GGCovarianceNoTomo.cov_of_statistic`.

        Returns
        -------
        dict, or tuple of two dicts with ``split_components``
            The keys of :meth:`GGCovarianceNoTomo.cov_of_statistic`, with blocks of shape
            (n_stat*n_tp, n_stat*n_tp).
        """
        # Evaluate the statistic on another xi_pm binning by interpolating the covariance
        if xipm_nbinstarget is not None and int(xipm_nbinstarget) != self.n_r:
            n_t = int(xipm_nbinstarget)
            new_te = np.geomspace(self.theta_edges[0], self.theta_edges[-1], n_t + 1)
            new_tc = np.sqrt(new_te[:-1] * new_te[1:])
            resampled = self.subbin_tomo(new_tc, new_te)
            return resampled.cov_of_statistic_tomo(
                filter_type, split_components=split_components, **kwargs)

        if self.cov_xipm_tomo is None:
            self.get_cov_xipm_tomo()

        n = self.n_r
        ntp = self.n_tomopairs

        # Lightweight non-tomo instance to compute filter weights once
        helper = object.__new__(GGCovarianceNoTomo)
        helper.theta_centers = self.theta_centers
        helper.theta_edges = self.theta_edges
        helper.n_r = n
        helper.cov_xipm = np.eye(2*n)
        _, weights = helper.cov_of_statistic(
            filter_type, return_weights=True, **kwargs)

        def _extract_block(i1, i2):
            block = np.zeros((2*n, 2*n))
            block[:n, :n] = self.cov_pp_tomo[i1*n:(i1+1)*n, i2*n:(i2+1)*n]
            block[:n, n:] = self.cov_pm_tomo[i1*n:(i1+1)*n, i2*n:(i2+1)*n]
            block[n:, :n] = self.cov_pm_tomo.T[i1*n:(i1+1)*n, i2*n:(i2+1)*n]
            block[n:, n:] = self.cov_mm_tomo[i1*n:(i1+1)*n, i2*n:(i2+1)*n]
            return block

        if len(weights) == 2 and isinstance(weights[1], list):
            # puremode: weights = (K, labels)
            K, lbl = weights
            ns = K.shape[0]
            is_puremode = True
            out_cov = np.zeros((ns * ntp, ns * ntp))
            for i1 in range(ntp):
                for i2 in range(ntp):
                    block = _extract_block(i1, i2)
                    out_cov[i1*ns:(i1+1)*ns, i2*ns:(i2+1)*ns] = K @ block @ K.T
            total = {'cov': out_cov, 'labels': lbl}
            Wpp, Wmm = K[:, :n], K[:, n:]
        else:
            # Map2 / cosebis: weights = (W_E, W_B)
            W_E, W_B = weights
            ns = W_E.shape[0]
            is_puremode = False
            total = {k: np.zeros((ns * ntp, ns * ntp))
                     for k in ('cov_EE', 'cov_BB', 'cov_EB')}
            for i1 in range(ntp):
                for i2 in range(ntp):
                    block = _extract_block(i1, i2)
                    sl1 = slice(i1 * ns, (i1 + 1) * ns)
                    sl2 = slice(i2 * ns, (i2 + 1) * ns)
                    total['cov_EE'][sl1, sl2] = W_E @ block @ W_E.T
                    total['cov_BB'][sl1, sl2] = W_B @ block @ W_B.T
                    total['cov_EB'][sl1, sl2] = W_E @ block @ W_B.T
            Wpp, Wmm = W_E[:, :n], W_E[:, n:]

        if not split_components:
            return total

        ## Per-component decomposition ##
        sz = ns * ntp

        def _zeros():
            if is_puremode:
                return {'cov': np.zeros((sz, sz))}
            return {k: np.zeros((sz, sz))
                    for k in ('cov_EE', 'cov_BB', 'cov_EB')}

        def _accum_pp(out, sl1, sl2, B):
            r = Wpp @ B @ Wpp.T
            if is_puremode:
                out['cov'][sl1, sl2] += r
            else:
                out['cov_EE'][sl1, sl2] += r
                out['cov_BB'][sl1, sl2] += r
                out['cov_EB'][sl1, sl2] += r

        def _accum_mm(out, sl1, sl2, B):
            r = Wmm @ B @ Wmm.T
            if is_puremode:
                out['cov'][sl1, sl2] += r
            else:
                out['cov_EE'][sl1, sl2] += r
                out['cov_BB'][sl1, sl2] += r
                out['cov_EB'][sl1, sl2] -= r

        def _accum_pm(out, sl1, sl2, P, PT):
            t1 = Wpp @ P @ Wmm.T
            t2 = Wmm @ PT @ Wpp.T
            if is_puremode:
                out['cov'][sl1, sl2] += t1 + t2
            else:
                s = t1 + t2
                out['cov_EE'][sl1, sl2] += s
                out['cov_BB'][sl1, sl2] -= s
                out['cov_EB'][sl1, sl2] += t2 - t1

        def _qlookup(qdict, b1, b2):
            combo = (b1, b2) if b1 <= b2 else (b2, b1)
            if combo in qdict:
                return qdict[combo]
            return qdict[(combo[1], combo[0])].T

        comp_names = ['D'] + [name for name, *_ in self.QTERMS] + list(self.RTERMS)
        comps = {name: _zeros() for name in comp_names}

        for i1 in range(ntp):
            tp1 = self.tomopairs_list[i1]
            for i2 in range(ntp):
                tp2 = self.tomopairs_list[i2]
                sl1 = slice(i1*ns, (i1+1)*ns)
                sl2 = slice(i2*ns, (i2+1)*ns)

                if i1 == i2:
                    d = np.diag(self.D_tomo[tp1])
                    _accum_pp(comps['D'], sl1, sl2, d)
                    _accum_mm(comps['D'], sl1, sl2, d)

                _accum_pp(comps['qpp'], sl1, sl2,
                          _qlookup(self.qpp_tomo, tp1, tp2))
                _accum_mm(comps['qmm'], sl1, sl2,
                          _qlookup(self.qmm_tomo, tp1, tp2))
                qpm_fwd = _qlookup(self.qpm_tomo, tp1, tp2)
                qpm_rev = _qlookup(self.qpm_tomo, tp2, tp1)
                _accum_pm(comps['qpm'], sl1, sl2, qpm_fwd, qpm_rev.T)

                _accum_pp(comps['rp0'], sl1, sl2,
                          self._rterm_block('rp0', (tp1, tp2)))
                _accum_pp(comps['rp1'], sl1, sl2,
                          self._rterm_block('rp1', (tp1, tp2)))
                _accum_mm(comps['rm0'], sl1, sl2,
                          self._rterm_block('rm0', (tp1, tp2)))
                _accum_mm(comps['rm1'], sl1, sl2,
                          self._rterm_block('rm1', (tp1, tp2)))
                rpm_fwd = self._rterm_block('rpm', (tp1, tp2))
                rpm_rev = self._rterm_block('rpm', (tp2, tp1))
                _accum_pm(comps['rpm'], sl1, sl2, rpm_fwd, rpm_rev.T)

        def _sum_dicts(*dicts):
            if is_puremode:
                return {'cov': sum(d['cov'] for d in dicts)}
            return {k: sum(d[k] for d in dicts)
                    for k in ('cov_EE', 'cov_BB', 'cov_EB')}

        comps['mix'] = _sum_dicts(comps['qpp'], comps['qmm'], comps['qpm'])
        comps['sva'] = _sum_dicts(comps['rp0'], comps['rp1'], comps['rm0'],
                                  comps['rm1'], comps['rpm'])

        return total, comps

    #################################
    # Resampling to a finer binning #
    #################################

    def subbin_tomo(self, new_theta_centers, new_theta_edges,
                    new_pairs=None, recompute_slab=False, slab_width=None,
                    slab_nr_block=None, slab_instance=None, verbose=False):
        """Resample the tomographic covariance to a finer radial binning.

        The parameters are as in :meth:`GGCovarianceNoTomo.subbin`, with ``slab_instance`` a
        :class:`GGCovariance`. Returns a :class:`GGCovariance` on the fine grid with all components
        loaded.
        """
        new_tc = np.asarray(new_theta_centers, dtype=float)
        new_te = np.asarray(new_theta_edges, dtype=float)

        ## Fine instance, which supplies D, the mixed terms and the band of the r-terms ##
        fine = slab_instance
        if fine is None and recompute_slab:
            if getattr(self, 'catalog', None) is None:
                raise ValueError(
                    "recompute_slab=True requires `slab_instance` (a GGCovariance on the fine "
                    "grid) or a `catalog` on this instance.")
            fine = self._fine_instance_from_catalog(new_tc, new_te, verbose=verbose)
            if new_pairs is None:
                new_pairs = fine.pairs
        if fine is not None and (fine.n_r != len(new_tc) or not np.allclose(fine.theta_edges, new_te)):
            raise ValueError("slab_instance must live on the same 2PCF theta grid as "
                             "the subbin target (new_theta_centers / new_theta_edges).")

        n_new = len(new_tc)
        tc = self.theta_centers
        x_c = np.log(tc)
        x_new = np.log(new_tc)
        _interp = self._build_block_interpolator(x_c, x_new, n_new)

        ## D and the mixed terms, interpolated without a fine instance ##
        if fine is None:
            if self.D_tomo is None: self.getD_tomo()
            if self.qpp_tomo is None: self.getqpp_tomo()
            if self.qmm_tomo is None: self.getqmm_tomo()
            if self.qpm_tomo is None: self.getqpm_tomo()
            D_fine = {}
            for tp in self.tomopairs_list:
                D_scaled = self.D_tomo[tp] * _theta_dtheta(self.theta_edges)
                D_fine[tp] = np.exp(np.interp(
                    x_new, x_c, np.log(D_scaled))) / _theta_dtheta(new_te)
            qpp_fine = {}
            qmm_fine = {}
            qpm_fine = {}
            for combo in self._tomo_combos():
                if combo in self.qpp_tomo:
                    qpp_fine[combo] = _interp(self.qpp_tomo[combo], sym=True)
                if combo in self.qmm_tomo:
                    qmm_fine[combo] = _interp(self.qmm_tomo[combo], sym=True)
                if combo in self.qpm_tomo:
                    qpm_fine[combo] = _interp(self.qpm_tomo[combo], sym=False)
        else:
            fine.getD_tomo()
            fine.getqpp_tomo()
            fine.getqmm_tomo()
            fine.getqpm_tomo()
            D_fine, qpp_fine, qmm_fine, qpm_fine = (fine.D_tomo, fine.qpp_tomo, fine.qmm_tomo,
                                                    fine.qpm_tomo)

        ## Interpolate r-terms ##
        if not self.rterms_tomo:
            self.get_all_rterms_tomo(verbose=verbose)
        rterms_fine = {}
        for key, block in self.rterms_tomo.items():
            rname, combo = key
            sym = rname in self.RTERMS_SYM
            rterms_fine[key] = _interp(block, sym=sym)

        ## Band of the r-terms on the fine grid ##
        if recompute_slab:
            # Set slab width and its associated mask
            if slab_width is None:
                raise ValueError("recompute_slab=True requires `slab_width`.")
            if np.isscalar(slab_width):
                widths =  {b: float(slab_width) for b in self.BLOCKS}
            else:
                widths =  {b: float(slab_width.get(b, 1.)) for b in self.BLOCKS}
            masks = {b: self._slab_band_mask(new_tc, w) for b, w in widths.items()}
            mask_max = masks['pp'] | masks['mm'] | masks['pm']
            # Optionally print some numbers regarding allocated elements on the grids
            if verbose:
                print("  subbin_tomo: %d -> %d bins, r-terms of the band w = %.3g (pp), %.3g (mm), "
                      "%.3g (pm) on the fine grid, %.1f%% of its elements"
                      % (self.n_r, n_new, widths['pp'], widths['mm'], widths['pm'],
                         100.*mask_max.mean()))
            # If there is sth to compute, first set up tiles for good caching and then get the required r-terms
            if mask_max.any():
                self._set_slab_tiles(fine, new_tc, widths, slab_nr_block)
                slab_rterms = fine.get_all_rterms_tomo(
                    verbose=verbose, pair_mask=masks)
                # Overwrite the slab band in interpolated r-terms
                for key, block in slab_rterms.items():
                    rname, combo = key
                    m = masks[self.RTERM_BLOCKS[rname]]
                    if key in rterms_fine:
                        rterms_fine[key] = np.where(m, block, rterms_fine[key])
                    else:
                        rterms_fine[key] = block

        ## Build output GGCovariance instance ##
        # `template=self` keeps the parent's q / triplet grids, integration
        # grids and tiling, so the result stays re-integrable.
        return GGCovariance._from_components(
            theta_centers=new_tc, theta_edges=new_te,
            phi_centers=self.phi_centers, phi_edges=self.phi_edges,
            spline_xip=self.xip, spline_xim=self.xim,
            sigma_eps=self.sigma_eps, nbar=self.nbar, area=self.area,
            weights_mean=self.weights_mean, weights_std=self.weights_std,
            nthreads=self.nthreads, template=self, pairs=new_pairs,
            _xip=self._xip, _xim=self._xim, ntomo=self.ntomo,
            n_tomopairs=self.n_tomopairs, tomopairs_list=self.tomopairs_list, nbar_ref=self.nbar_ref,
            D_tomo=D_fine, qpp_tomo=qpp_fine, qmm_tomo=qmm_fine,
            qpm_tomo=qpm_fine, rterms_tomo=rterms_fine)

    def _fine_instance_from_catalog(self, new_tc, new_te, verbose=False):
        """Build a fine-grid instance for :meth:`subbin_tomo` from the catalog of this instance.

        The fine pairs and triplets are measured, the quadruplet density is reused.
        """
        if getattr(self, 'catalog', None) is None:
            raise ValueError(
                "_fine_instance_from_catalog requires a `catalog` on this "
                "instance.")
        xip_list = [(i, j, spl) for (i, j), spl in self._xip.items()]
        xim_list = [(i, j, spl) for (i, j), spl in self._xim.items()]
        if verbose:
            print("  subbin_tomo: auto-building fine instance (n_fine=%d) -- "
                  "measuring fine pairs/triplets, reusing q density"
                  % len(new_tc))
        return GGCovariance(**self._grid_kwargs(
            new_tc, new_te,
            method_pairs='discrete', method_triplets='discrete',
            method_quadruplets='discrete',
            spline_xip=xip_list, spline_xim=xim_list, nbar_ref=self.nbar_ref,
            quadruplets=self.quadruplets))

    #######################
    # Covariance of xi_pm #
    #######################

    def get_cov_xipm_tomo(self, verbose=False):
        """Assemble the full (2*n_tp*n_r, 2*n_tp*n_r) tomo covariance."""
        if self.cov_pp_tomo is None: self.get_cov_pp_tomo(verbose=verbose)
        if self.cov_mm_tomo is None: self.get_cov_mm_tomo(verbose=verbose)
        if self.cov_pm_tomo is None: self.get_cov_pm_tomo(verbose=verbose)
        m = self.n_tomopairs * self.n_r
        out = np.zeros((2 * m, 2 * m))
        out[:m, :m] = self.cov_pp_tomo
        out[:m, m:] = self.cov_pm_tomo
        out[m:, :m] = self.cov_pm_tomo.T
        out[m:, m:] = self.cov_mm_tomo
        self.cov_xipm_tomo = out
        return out

    def get_cov_terms_tomo(self, verbose=False):
        """Shape-noise, mixed and sample-variance terms of the tomographic covariance blocks.

        Returns ``{block: {'D': ..., 'mix': ..., 'sva': ...}}`` for the blocks ``'pp'``, ``'mm'``
        and ``'pm'`` of :meth:`get_cov_xipm_tomo`, each term of shape (n_tp*n_r, n_tp*n_r), and the
        three terms of a block sum to it.
        """
        for index, (block, *_) in enumerate(self.COVBLOCKS):
            if block not in self.cov_terms_tomo:
                self._get_cov_block_tomo(index, verbose)
        return self.cov_terms_tomo

    def get_cov_pp_tomo(self, verbose=False):
        """Assemble the full Cov_pp tomo matrix."""
        return self._get_cov_block_tomo(0, verbose)

    def get_cov_mm_tomo(self, verbose=False):
        """Assemble the full Cov_mm tomo matrix."""
        return self._get_cov_block_tomo(1, verbose)

    def get_cov_pm_tomo(self, verbose=False):
        """Assemble the full Cov_pm tomo matrix (no diagonal D term)."""
        return self._get_cov_block_tomo(2, verbose)

    def _get_cov_block_tomo(self, index, verbose=False):
        """Assemble one tomo covariance block, by position in COVBLOCKS."""
        block, _, qname, rnames, with_D = self.COVBLOCKS[index]
        q_tomo = getattr(self, qname + '_tomo')
        if with_D and self.D_tomo is None:
            self.getD_tomo()
        if q_tomo is None:
            q_tomo = getattr(self, 'get' + qname + '_tomo')()
        if not self.rterms_tomo:
            self.get_all_rterms_tomo(verbose=verbose)
        n = self.n_r
        # Shape-noise, mixed and sample-variance terms, kept apart for
        # get_cov_terms_tomo
        terms = {name: np.zeros((self.n_tomopairs * n, self.n_tomopairs * n))
                 for name in ('D', 'mix', 'sva')}
        for idx1, tp1 in enumerate(self.tomopairs_list):
            if with_D:
                # Diagonal shape noise.
                terms['D'][idx1*n:(idx1+1)*n, idx1*n:(idx1+1)*n] += np.diag(self.D_tomo[tp1])
            for idx2, tp2 in enumerate(self.tomopairs_list):
                combo = (tp1, tp2) if tp1 <= tp2 else (tp2, tp1)
                sl1 = slice(idx1 * n, (idx1+1)*n)
                sl2 = slice(idx2 * n, (idx2+1)*n)
                if combo in q_tomo:
                    terms['mix'][sl1, sl2] += q_tomo[combo]
                else:
                    terms['mix'][sl1, sl2] += q_tomo[(combo[1], combo[0])].T
                for rname in rnames:
                    terms['sva'][sl1, sl2] += self._rterm_block(rname, (tp1, tp2))
        self.cov_terms_tomo[block] = terms
        out = terms['D'] + terms['mix'] + terms['sva']
        setattr(self, 'cov_' + block + '_tomo', out)
        return out

    ## Shape noise and mixed terms ##

    def getD_tomo(self):
        """Shape-noise term ``{(a, b): (n_r,)}`` of each bin pair, nonzero in auto blocks only."""
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        self.D_tomo = {}
        for tp in self.tomopairs_list:
            a, b = tp
            se4 = self.sigma_eps[a]**2 * self.sigma_eps[b]**2
            if self.method_pairs in ('analytic', 'continuous'):
                Np = (2. * np.pi * self.area
                      * self.nbar[a] * self.nbar[b]
                      * self._w1**2
                      * _theta_dtheta(self.theta_edges)
                      * self._suppression(self.theta_centers))
                Pw = None
            else:
                Np = self.pairs
                Pw = self.pairs_wsq
                if self.nbar_ref is not None:
                    Np = Np * (self.nbar[a] * self.nbar[b]
                               / self.nbar_ref**2)
                    Pw = None if Pw is None else Pw * (self.nbar[a] * self.nbar[b]
                                                       / self.nbar_ref**2)
            # Np are ordered pairs within one bin. A pair between two bins is
            # double counted on the same footing by taking it in both orders.
            if a != b:
                Np = 2. * Np
                Pw = None if Pw is None else 2. * Pw
            # The measured pairs of squared weights replace N_p <w^2>^2 / <w>^2
            self.D_tomo[tp] = se4 * (ww_factor / Np if Pw is None else Pw / Np**2)
        return self.D_tomo

    def getqpp_tomo(self):
        """Triplet q_++ for all tomo combos."""
        return self._q_tomo_blocks(0)

    def getqmm_tomo(self):
        """Triplet q_-- for all tomo combos."""
        return self._q_tomo_blocks(1)

    def getqpm_tomo(self):
        """Triplet q_+- for all tomo combos."""
        return self._q_tomo_blocks(2)

    ## Cosmic variance ##

    def get_all_rterms_tomo(self, verbose=False, pair_mask=None,
                            nthreads=None):
        """Compute all tomographic r-term blocks in one pass.

        Returns ``{(rterm, ((a,b),(c,d))): (n_r, n_r)}``. Blocks restricted by ``pair_mask`` are
        returned but not stored.
        """
        if nthreads is None:
            nthreads = self.nthreads
        rterms = self._quadruplet_density_terms_tomo_c(
            verbose=verbose, pair_mask=pair_mask, nthreads=nthreads)
        if pair_mask is None:
            self.rterms_tomo = rterms
        return rterms

    ########################
    # Discrete expressions #
    ########################

    def _q_tomo_blocks(self, index):
        """All tomographic blocks of one mixed term, by its position in :attr:`QTERMS`.

        Auto combos of q_++ and q_-- are symmetrised, q_+- is not.
        """
        name, weight_type, xi_pm = self.QTERMS[index]
        symmetrise = name in self.QTERMS_SYM
        blocks = {}
        for combo in self._tomo_combos():
            block = self._triplet_tomo_block(weight_type, xi_pm, combo)
            if symmetrise and combo[0] == combo[1]:
                block = .5 * (block + block.T)
            blocks[combo] = block
        setattr(self, name + '_tomo', blocks)
        return blocks

    def _triplet_tomo_block(self, weight_type, xi_pm, combo):
        """Mixed term of one tomographic combo.

        The shared galaxy of the two pairs (a, b) and (c, d) lies in a common bin. Each such case
        adds the triplet term with xi of the two other bins and with sigma_eps and nbar of the
        shared bin, and the factor 1/4 matches the four cases of the non-tomographic term.
        """
        (a, b), (c, d) = combo
        n = self.n_r
        block = np.zeros((n, n))
        if self.nbar_ref is not None:
            nf_a = self.nbar_ref / self.nbar[a]
            nf_b = self.nbar_ref / self.nbar[b]
        else:
            nf_a = nf_b = 1.
        # xi is evaluated at the NON-shared bin pair of each Kronecker match.
        for shared_a, shared_b, xi_i, xi_j, sig, nf in [
                (a, c, b, d, a, nf_a), (a, d, b, c, a, nf_a),
                (b, c, a, d, b, nf_b), (b, d, a, c, b, nf_b)]:
            if shared_a == shared_b:
                block += self._triplet_term(
                    weight_type, self._xi(xi_pm, xi_i, xi_j),
                    sigma_eps_sq=self.sigma_eps[sig]**2, nbar_fac=nf)
        return .25 * block

    def _quadruplet_density_terms_tomo_c(self, verbose=False, pair_mask=None,
                                         nthreads=None):
        """Tomographic r-terms via the C kernel with ``nthreads`` OpenMP threads."""
        if self.quadruplets is None:
            raise RuntimeError("`quadruplets` not initialised.")

        cbm = self._build_tomo_callback_map()
        out_keys = cbm['out_keys']
        xi_tables = self._build_xi_tables(cbm['xi_table_keys'])

        # Per-output-block mask: True if this block's r-term is symmetric.
        symmetric_mask = [key[0] in self.RTERMS_SYM and key[1][0] == key[1][1]
                          for key in out_keys]

        accum = self._run_postq_engine(
            cbm, xi_tables, symmetric_mask, [self.RTERM_BLOCKS[key[0]] for key in out_keys],
            verbose=verbose, pair_mask=pair_mask, nthreads=nthreads)

        return {key: (.5 * (block + block.T) if symmetric_mask[oi] else block)
                for oi, (key, block) in enumerate(zip(out_keys, accum))}

    def _build_tomo_callback_map(self):
        """Flat callback arrays of all tomographic r-terms for the C kernel.

        Returns a dict of ``xi_a_idx`` and ``xi_b_idx`` (xi table of each side), ``w_ids`` (weight
        0-4), ``out_idx`` (output block), ``wick_fac`` (1 for a degenerate Wick pairing, 0.5 for
        each of two distinct ones), ``xi_table_keys`` ((pm, i, j) of each table), ``out_keys``
        ((r-term, combo) of each output) and ``n_out``.
        """
        combos_sym = self._tomo_combos() # upper-triangle for symmetric rterms
        combos_all = self._tomo_combos_all() # all ordered pairs for rpm

        # R-term types: (name, weight_id, xi_type_a, xi_type_b)
        rterm_types = [(name, wid, *self.RTERM_XI[name]) for wid, name in enumerate(self.RTERMS)]

        # Collect unique xi spline keys and assign table indices.
        xi_keys = set()
        for pm in ('p', 'm'):
            for i in range(self.ntomo):
                for j in range(i, self.ntomo):
                    xi_keys.add((pm, i, j))
        xi_keys = sorted(xi_keys)
        xi_key_to_idx = {k: idx for idx, k in enumerate(xi_keys)}

        # Build output key list. Symmetric r-terms use upper-triangle combos;
        # rpm uses ALL ordered pairs (cov_pm is not symmetric under bin swap).
        out_keys = []
        out_key_to_idx = {}
        for rname in self.RTERMS:
            for combo in (combos_sym if rname in self.RTERMS_SYM else combos_all):
                key = (rname, combo)
                out_key_to_idx[key] = len(out_keys)
                out_keys.append(key)

        # Build callback lists.
        cb_xi_a = []
        cb_xi_b = []
        cb_w = []
        cb_out = []
        cb_fac = []

        for rname, wid, xi_type_a, xi_type_b in rterm_types:
            combos = combos_sym if rname in self.RTERMS_SYM else combos_all
            for combo in combos:
                (a, b), (c, d) = combo
                out = out_key_to_idx[(rname, combo)]

                # Two Wick cross-pairings per (weight_type, combo).
                wick_pairs = [
                    ((xi_type_a, min(a, c), max(a, c)),
                     (xi_type_b, min(b, d), max(b, d))),
                    ((xi_type_a, min(a, d), max(a, d)),
                     (xi_type_b, min(b, c), max(b, c))),
                ]

                # The prefactor 2/(pi A) counts both Wick contractions, so two distinct ones
                # contribute 0.5 each and a degenerate one 1
                if wick_pairs[0] == wick_pairs[1]:
                    cb_xi_a.append(xi_key_to_idx[wick_pairs[0][0]])
                    cb_xi_b.append(xi_key_to_idx[wick_pairs[0][1]])
                    cb_w.append(wid)
                    cb_out.append(out)
                    cb_fac.append(1.)
                else:
                    for xia_key, xib_key in wick_pairs:
                        cb_xi_a.append(xi_key_to_idx[xia_key])
                        cb_xi_b.append(xi_key_to_idx[xib_key])
                        cb_w.append(wid)
                        cb_out.append(out)
                        cb_fac.append(.5)

        return dict(
            xi_a_idx=np.array(cb_xi_a, dtype=np.int32),
            xi_b_idx=np.array(cb_xi_b, dtype=np.int32),
            w_ids=np.array(cb_w, dtype=np.int32),
            out_idx=np.array(cb_out, dtype=np.int32),
            wick_fac=np.array(cb_fac, dtype=np.float64),
            xi_table_keys=xi_keys,
            out_keys=out_keys,
            n_out=len(out_keys),
        )

    ##########################
    # Continuous expressions #
    ##########################

    def get_continuous_tomo_c(self, verbose=False, nr_target=1024, nphi_target=1024, dlnt_target=.05):
        """Compute D, the mixed and the r-terms of all tomographic blocks in the continuum limit.

        The mixed terms are averaged over nr_target fine bins and the r-terms over sub-bins no wider
        than dlnt_target in ln theta. Assembles and returns ``cov_xipm_tomo``.
        """

        ## Helpers
        def _ptr(a): return np.ascontiguousarray(a, dtype=np.float64)
        def _iptr(a): return np.ascontiguousarray(a, dtype=np.int32)
        lib = self._clib
        n = self.n_r
        combos_sym = self._tomo_combos()
        combos_all = self._tomo_combos_all()

        ## Build xi tables
        xi_keys = sorted({(pm, i, j)
                          for pm in ('p', 'm')
                          for i in range(self.ntomo)
                          for j in range(i, self.ntomo)})
        xi_key_to_idx = {k: idx for idx, k in enumerate(xi_keys)}
        for pm, i, j in list(xi_key_to_idx.keys()):
            xi_key_to_idx[(pm, j, i)] = xi_key_to_idx[(pm, i, j)]
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = self._build_xi_tables(xi_keys)
        n_xi = len(xi_keys)

        ## Get D term
        self.getD_tomo()

        ## Get q terms
        if verbose: print("  continuous q-terms (C/OpenMP) ...")
        alpha = max(1, int(nr_target // n))
        n_fine = alpha * n
        _edges = np.geomspace(self.theta_edges[0], self.theta_edges[-1], n_fine + 1)
        fine_centers = np.ascontiguousarray(.5 * (_edges[1:] + _edges[:-1]))
        fine_dtheta = np.diff(_edges)
        fine_weights = np.ascontiguousarray(fine_centers * fine_dtheta)
        phi_q = np.ascontiguousarray(np.linspace(0., np.pi, nphi_target, endpoint=False))
        dphi_q = np.ascontiguousarray(_trapz_weights(phi_q))
        # The mixed terms reach shorter third sides than the r-terms, as in _qterms_continuous_c
        r_min = min(fine_centers[1] - fine_centers[0], 2. * fine_centers[0] * np.sin(.5 * phi_q[1]))
        qxi_x, qxi_y_tables, qxi_logx0, qxi_inv_dlogx, qxi_n = self._build_xi_tables(
            xi_keys, lo=min(self._xi_query_lo(), .9 * r_min))
        Np_ideal = np.ones(n, dtype=np.float64)

        # The kernel output depends on (w_type, xi_idx) only, so each distinct pair is integrated
        # once and the factors of the combos are applied afterwards
        q_calls = {} # (w_type, xi_idx) -> kernel slot
        q_scatter = [] # (out_idx, slot, factor)
        q_out_keys = []
        for qname, w_type, xi_pm in self.QTERMS:
            for combo in combos_sym:
                (a, b), (c, d) = combo
                q_out_keys.append((qname, combo))
                out_idx = len(q_out_keys) - 1
                for shared_a, shared_b, xi_i, xi_j in [
                    (a, c, b, d), (a, d, b, c), (b, c, a, d), (b, d, a, c)]:
                    if shared_a == shared_b:
                        call = (w_type, xi_key_to_idx[(xi_pm, xi_i, xi_j)])
                        slot = q_calls.setdefault(call, len(q_calls))
                        # The shared galaxy carries <w^2>, the other two <w> each
                        q_scatter.append((out_idx, slot, 
                                          .25*self.sigma_eps[shared_a]**2/self.nbar[shared_a] * self._w2/self._w1**2))

        n_q_calls = len(q_calls)
        n_q_out = len(q_out_keys)
        q_raw = np.zeros(n_q_calls * n * n, dtype=np.float64)

        if n_q_calls > 0:
            calls = sorted(q_calls, key=q_calls.get)
            lib.continuous_q_tomo(
                n, alpha,
                _ptr(fine_centers), _ptr(fine_weights),
                _ptr(phi_q), _ptr(dphi_q),
                n_fine, nphi_target,
                _ptr(qxi_x), qxi_n, qxi_logx0, qxi_inv_dlogx,
                _ptr(qxi_y_tables), n_xi,
                1., self.area, 1.,
                _ptr(Np_ideal),
                n_q_calls,
                _iptr(np.array([c[1] for c in calls], dtype=np.int32)),
                _iptr(np.array([c[0] for c in calls], dtype=np.int32)),
                _iptr(np.arange(n_q_calls, dtype=np.int32)),
                _ptr(np.ones(n_q_calls, dtype=np.float64)),
                n_q_calls,
                q_raw,
                self.nthreads, int(verbose))
            check_clib_error(lib)

        q_raw = q_raw.reshape(n_q_calls, n, n)
        q_results = np.zeros((n_q_out, n, n))
        for out_idx, slot, fac in q_scatter:
            q_results[out_idx] += fac * q_raw[slot]

        self.qpp_tomo = {}
        self.qmm_tomo = {}
        self.qpm_tomo = {}
        for oi, (qname, combo) in enumerate(q_out_keys):
            block = q_results[oi]
            sym = qname in self.QTERMS_SYM
            if sym and combo[0] == combo[1]:
                block = .5 * (block + block.T)
            getattr(self, qname + '_tomo')[combo] = block

        ## R terms via C ##
        if verbose: print("  continuous r-terms (C/OpenMP) ...")
        v = self.varphi1_grid
        v_half = v[v <= np.pi]
        phi_grid = np.ascontiguousarray(self.phi_grid, dtype=np.float64)
        phi_w = np.ascontiguousarray(_trapz_weights(self.phi_grid), dtype=np.float64)
        v_grid = np.ascontiguousarray(v, dtype=np.float64)
        v_w_arr = np.ascontiguousarray(_periodic_weights(v), dtype=np.float64)
        v_half_c = np.ascontiguousarray(v_half, dtype=np.float64)
        v_half_w = np.ascontiguousarray(_trapz_weights(v_half), dtype=np.float64)
        theta_nodes, theta_weights = self._rterm_nodes(dlnt_target)
        n_f = theta_nodes.size
        theta_c = np.ascontiguousarray(theta_nodes.ravel(), dtype=np.float64)

        ## Build the r-term work list ##
        # Five consecutive output blocks per ordered combo, as rpm needs all of them; the
        # redundant symmetric blocks are dropped afterwards
        all_combos = combos_all
        r_out_keys = [(rname, combo)
                      for combo in all_combos for rname in self.RTERMS]

        # Both Wick pairings are always summed, also degenerate ones, and the result is halved,
        # since the prefactor 2/(pi A) already counts both contractions
        work_xi_a_p = []
        work_xi_b_p = []
        work_xi_a_m = []
        work_xi_b_m = []
        work_out_idx = []
        for ic, combo in enumerate(all_combos):
            (a, b), (c, d) = combo
            for wa, wb in [((a, c), (b, d)), ((a, d), (b, c))]:
                work_xi_a_p.append(xi_key_to_idx[('p', *wa)])
                work_xi_b_p.append(xi_key_to_idx[('p', *wb)])
                work_xi_a_m.append(xi_key_to_idx[('m', *wa)])
                work_xi_b_m.append(xi_key_to_idx[('m', *wb)])
                work_out_idx.append(ic * len(self.RTERMS))

        n_work = len(work_xi_a_p)
        n_r_out = len(r_out_keys)
        r_results = np.zeros(n_r_out * n_f * n_f, dtype=np.float64)

        lib.continuous_rterms_tomo(
            n_f, _ptr(theta_c),
            _ptr(phi_grid), _ptr(phi_w), len(self.phi_grid),
            _ptr(v_grid), _ptr(v_w_arr), len(v),
            _ptr(v_half_c), _ptr(v_half_w), len(v_half),
            float(self.area),
            _ptr(xi_x), xi_n, xi_logx0, xi_inv_dlogx,
            _ptr(xi_y_tables), n_xi,
            n_work,
            _iptr(np.array(work_xi_a_p, dtype=np.int32)),
            _iptr(np.array(work_xi_b_p, dtype=np.int32)),
            _iptr(np.array(work_xi_a_m, dtype=np.int32)),
            _iptr(np.array(work_xi_b_m, dtype=np.int32)),
            _iptr(np.array(work_out_idx, dtype=np.int32)),
            n_r_out,
            r_results,
            self.nthreads, int(verbose))
        check_clib_error(lib)

        r_results = .5 * _bin_average_blocks(r_results.reshape(n_r_out, n_f, n_f), theta_weights)
        self.rterms_tomo = {}
        for oi, (rname, combo) in enumerate(r_out_keys):
            if rname in self.RTERMS_SYM and combo[0] > combo[1]:
                continue
            block = r_results[oi]
            if rname in self.RTERMS_SYM and combo[0] == combo[1]:
                block = .5 * (block + block.T)
            self.rterms_tomo[(rname, combo)] = block

        ## Assemble ##
        if verbose: print("  assembling cov_xipm_tomo ...")
        self.get_cov_xipm_tomo()
        return self.cov_xipm_tomo

    ###################
    # General helpers #
    ###################

    def _xi(self, pm, i, j):
        """Look up xi_+(i,j) or xi_-(i,j) spline, handling index ordering."""
        key = (min(i, j), max(i, j))
        d = self._xip if pm == 'p' else self._xim
        return d[key]

    def _build_xi_tables(self, xi_keys, lo=None):
        """Tabulate xi of the (pm, i, j) keys, as :meth:`_xi_tables_from_splines`, from ``lo`` or
        the smallest separation of the quadruplet kernels."""
        return self._xi_tables_from_splines(
            [self._xi(pm, i, j) for pm, i, j in xi_keys],
            lo=self._xi_query_lo() if lo is None else lo, hi=self._xi_query_hi())

    def _rterm_block(self, rname, combo):
        """Look up r-term block, filling the lower triangle by transpose."""
        key = (rname, combo)
        if key in self.rterms_tomo:
            return self.rterms_tomo[key]
        rev_combo = (combo[1], combo[0])
        rev_key = (rname, rev_combo)
        if rev_key in self.rterms_tomo:
            return self.rterms_tomo[rev_key].T
        raise KeyError(f"r-term block {key} not found")

    def _tomo_combos(self):
        """Non-redundant combos ((a,b),(c,d)) with (a,b) <= (c,d), for the symmetric r-terms."""
        combos = []
        for idx1, tp1 in enumerate(self.tomopairs_list):
            for tp2 in self.tomopairs_list[idx1:]:
                combos.append((tp1, tp2))
        return combos

    def _tomo_combos_all(self):
        """All ordered combos ((a,b),(c,d)), since Cov_pm is not symmetric under their exchange."""
        return [(tp1, tp2) for tp1 in self.tomopairs_list for tp2 in self.tomopairs_list]