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

def _triplet_weight(weight_type, PHI, jk_x, jk_y):
    """Angular weight of the mixed terms, as in C kernel, sorted as [q++, q--, q+-]."""
    if weight_type == 0:
        return np.ones_like(PHI)
    if weight_type == 1:
        return np.cos(4. * PHI)
    return np.cos(4. * (PHI - np.arctan2(jk_y, jk_x)))


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


def _build_q_func(q_array, r_centers, phi_centers, method='linear'):
    """Wrap a 5-D quadruplet-density array into a callable q(t1, t2, t3, a12, a13)."""
    # Linear inside the radial grid, nearest neighbour below r_min and zero above r_max
    grid = (r_centers, r_centers, r_centers, phi_centers, phi_centers)
    lin_interp = RegularGridInterpolator(
        grid, q_array, method=method, bounds_error=False, fill_value=None)
    nn_interp = RegularGridInterpolator(
        grid, q_array, method='nearest', bounds_error=False, fill_value=None)

    # Reuse original q_array if scipy copied it
    if not np.shares_memory(lin_interp.values, q_array):
        lin_interp.values = q_array
    if not np.shares_memory(nn_interp.values, q_array):
        nn_interp.values = q_array

    twopi = 2. * np.pi
    r_min = r_centers[0]
    r_max = r_centers[-1]

    def q_func(t1, t2, t3, phi12, phi13):
        # Broadcast points to common shape and evaluate both interpolation methods
        phi12 = np.mod(phi12, twopi)
        phi13 = np.mod(phi13, twopi)
        t1, t2, t3, phi12, phi13 = np.broadcast_arrays(t1, t2, t3, phi12, phi13)
        pts = np.stack((t1, t2, t3, phi12, phi13), axis=-1)
        lin = lin_interp(pts)
        nn = nn_interp(pts)
        # Identify points outside the radial grid & use appropriate extrapolations
        r_stack = np.stack((t1, t2, t3), axis=-1)
        below = np.any(r_stack < r_min, axis=-1)
        above = np.any(r_stack > r_max, axis=-1)
        out = lin
        out = np.where(below, nn, out)
        out = np.where(above, 0., out)
        return out

    return q_func


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
        Radial bins of the quadruplet density, typically the 4PCF binning reaching well beyond
        the xi_pm range, and to be set for a measured density. Default to the xi_pm bins.
    q_phi_centers, q_phi_edges: numpy.ndarray, optional
        Angular bins of the quadruplet density. Default to ``phi_centers`` and ``phi_edges``.
    triplet_oversample: int, optional
        Number of equal sub-bins per radial bin of the triplet counts, which are summed back
        before dividing by the pair counts. Defaults to ``1``.
    method_pairs, method_triplets, method_quadruplets: str, optional
        Source of each multiplet: ``'discrete'`` (measured from ``catalog`` or passed),
        ``'analytic'`` (idealised Poisson counts) or ``'continuous'`` (continuum limit of the
        integrals over ``spline_xip`` and ``spline_xim``). ``method_quadruplets`` also takes
        ``'frompairs'``, the constant density times ``spl_suppression_ww`` of each separation.
        Defaults to ``'analytic'``.
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
        Count tensors, or their files, of the ``'discrete'`` methods. Measured from ``catalog``
        if not given.
    catalog: dict, optional
        Catalog of the discrete measurements with the keys ``ra``, ``dec`` (deg), ``path_save``,
        ``npatches`` and ``nmaxs``, and optionally ``weight``, ``rmin_pixsize`` (40),
        ``tree_resos`` ([0, 0.5, 1, 2, 4]), ``method`` ('DoubleTree'), ``shuffle_pix`` (1),
        ``verbosity`` and ``memory_bound``. The measured radial grids must be log-uniform.
    store_multiplets: bool, optional
        Keep the measured count tensors on disk, while the 4PCF patch files are always removed.
        Defaults to ``False``.
    phi_grid, varphi1_grid, varphi2_grid: numpy.ndarray, optional
        Integration grids of the quadruplet term. Default to 128 log-spaced radii in [0.05, 1e4]
        and 64 uniform angles.
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
        q_theta_centers=None, q_theta_edges=None, q_phi_centers=None, q_phi_edges=None,
        triplet_oversample=1,
        method_pairs='analytic', method_triplets='analytic', method_quadruplets='analytic',
        spline_xip=None, spline_xim=None,
        sigma_eps=0.27, nbar=None, area=None, weights_mean=1., weights_std=0., spl_suppression_ww=None,
        pairs=None, triplets=None, quadruplets=None,
        catalog=None, store_multiplets=False,
        phi_grid=None, varphi1_grid=None, varphi2_grid=None, phi1_subgrid_frac=None, phi2_subgrid_frac=None,
        v2_block=16, nr_block=None, nthreads=1):

        ## Link compiled libraries ##
        self._link_clib()

        ## 2PCF radial bins ##
        self.theta_centers = np.asarray(theta_centers, dtype=float)
        self.theta_edges = np.asarray(theta_edges, dtype=float)
        if len(self.theta_edges) != len(self.theta_centers) + 1:
            raise ValueError("theta_edges must have exactly one more entry than theta_centers")
        self.dtheta = np.diff(self.theta_edges)
        self.n_r = len(self.theta_centers)

        ## Angular bins (shared between triplets and quadruplets) ##
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

        ## Quadruplet radial / angular grid ##
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
        self.q_theta_centers = (self.theta_centers if q_theta_centers is None
                                else np.asarray(q_theta_centers, dtype=float))
        self.q_theta_edges = (self.theta_edges if q_theta_edges is None
                                else np.asarray(q_theta_edges, dtype=float))
        if len(self.q_theta_edges) != len(self.q_theta_centers) + 1:
            raise ValueError("q_theta_edges must have exactly one more entry than q_theta_centers")
        self.q_dtheta = np.diff(self.q_theta_edges)
        self.q_n_r = len(self.q_theta_centers)
        self.q_phi_centers = (self.phi_centers if q_phi_centers is None
                              else np.asarray(q_phi_centers, dtype=float))
        self.q_phi_edges = (self.phi_edges if q_phi_edges is None
                              else np.asarray(q_phi_edges, dtype=float))
        self.q_dphi = np.diff(self.q_phi_edges)
        self.q_n_phi = len(self.q_phi_centers)

        ## Triplet oversampling ##
        self.triplet_oversample = int(triplet_oversample)
        if self.triplet_oversample < 1:
            raise ValueError("triplet_oversample must be >= 1")
        alpha = self.triplet_oversample
        if alpha > 1:
            # Build fine radial grid by subdividing each coarse bin into equal sub-bins.
            sub_edges = (
                self.theta_edges[:-1, None] +
                np.arange(alpha + 1)[None, :] * self.dtheta[:, None] / alpha)
            fine_edges = np.concatenate(
                [sub_edges[:, :-1].ravel(), [self.theta_edges[-1]]])
            self.triplet_theta_edges = fine_edges
            self.triplet_theta_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
            self.triplet_dtheta = np.diff(fine_edges)
            self.triplet_n_r = self.n_r * alpha
        else:
            self.triplet_theta_edges = self.theta_edges
            self.triplet_theta_centers = self.theta_centers
            self.triplet_dtheta = self.dtheta
            self.triplet_n_r = self.n_r

        ## Method switches ##
        self.method_pairs = method_pairs
        self.method_triplets = method_triplets
        self.method_quadruplets = method_quadruplets

        ## Cosmological inputs ##
        self.xip = spline_xip
        self.xim = spline_xim
        self.sigma_eps = float(sigma_eps)

        ## Survey / weighting ##
        self.nbar = nbar
        self.area = area
        self.weights_mean = float(weights_mean)
        self.weights_std = float(weights_std)
        self.spl_suppression_ww = spl_suppression_ww

        # <w> and <w^2> = <w>^2 + Var(w)
        self._w1 = self.weights_mean
        self._w2 = self.weights_mean**2 + self.weights_std**2

        ## Integration grids (analytic branch + quadruplet density integral) ##
        self.phi_grid = (np.geomspace(0.05, 1e4, 128)
                         if phi_grid is None else np.asarray(phi_grid))
        self.varphi1_grid = (np.linspace(0., 2. * np.pi, 64, endpoint=False)
                             if varphi1_grid is None else np.asarray(varphi1_grid))
        self.varphi2_grid = (np.linspace(0., 2. * np.pi, 64, endpoint=False)
                             if varphi2_grid is None else np.asarray(varphi2_grid))
        self.phi1_subgrid_frac = phi1_subgrid_frac
        self.phi2_subgrid_frac = phi2_subgrid_frac

        ## Tiling and threads ##
        self.v2_block = int(v2_block)
        if self.v2_block < 1:
            raise ValueError("v2_block must be >= 1")
        # Tile size on the (theta1, theta2) axes of the quadruplet integral, None for one tile
        self.nr_block = None if nr_block is None else int(nr_block)
        if self.nr_block is not None and self.nr_block < 1:
            raise ValueError("nr_block must be >= 1 (or None for no tiling)")
        # Number of OpenMP threads
        self.nthreads = int(nthreads)
        if self.nthreads < 1:
            raise ValueError("nthreads must be >= 1")

        ## Output storage ##
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

        ## Catalog of the discrete multiplet measurements ##
        self.catalog = catalog
        self.store_multiplets = bool(store_multiplets)
        # Patched orpheus catalogs by kind ('spin' or 'scalar'), built when first needed
        self._cat_cache = {}
        if self.catalog is not None:
            # orpheus bins radially in log theta
            self._check_loguniform(self.theta_edges, "theta_edges")
            if (self.method_quadruplets == 'discrete'
                    and not callable(quadruplets) and quadruplets is None):
                self._check_loguniform(self.q_theta_edges, "q_theta_edges")

        ## Multiplet count initialisation ##
        self.pairs = self._init_multiplet('pairs', pairs, self.method_pairs)
        self.triplets = self._init_multiplet('triplets', triplets, self.method_triplets)
        self.quadruplets = self._init_multiplet('quadruplets', quadruplets, self.method_quadruplets)

        # Define quadruplet density
        if self.method_quadruplets == 'continuous':
            self._q_func = None
        elif callable(self.quadruplets):
            self._q_func = self.quadruplets
        else:
            # On the quadruplet grid, not the xi_pm grid
            self._q_func = _build_q_func(
                self.quadruplets, self.q_theta_centers, self.q_phi_centers)

    # How to construct the multiplet data of each non-discrete method
    _MULTIPLET_SOURCES = {
        'pairs': {'analytic': 'pairs_analytic', 'continuous': 'pairs_analytic'},
        'triplets': {'analytic': 'triplets_analytic', 'continuous': None},
        'quadruplets': {'analytic': 'quadruplets_analytic',
                        'frompairs': 'quadruplets_analytic',
                        'continuous': None},}

    ## Generic helpers for init ##
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
            p_i32, p_i32, p_f64, ci, p_i32, p_i32, p_f64, ci, ci]

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
            if isinstance(value, str):
                return getattr(self, '_load_' + kind)(value)
            if value is None:
                return self.computeDiscrete(kind)
            if kind == 'quadruplets' and callable(value):
                return value
            value = np.asarray(value, dtype=float)
            getattr(self, '_check_%s_shape' % kind)(value)
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

    ####################
    # Shape validators #
    ####################

    def _check_pairs_shape(self, arr):
        expect = (self.n_r,)
        if arr.shape != expect:
            raise ValueError(
                f"pairs shape {arr.shape} != expected {expect}")

    def _check_triplets_shape(self, arr):
        expect = (self.triplet_n_r, self.triplet_n_r, self.n_phi)
        if arr.shape != expect:
            raise ValueError(
                f"triplets shape {arr.shape} != expected {expect} "
                f"(triplet_oversample={self.triplet_oversample})")

    def _check_quadruplets_shape(self, arr):
        expect = (self.q_n_r, self.q_n_r, self.q_n_r, self.q_n_phi, self.q_n_phi)
        if arr.shape != expect:
            raise ValueError(
                f"quadruplets shape {arr.shape} != expected {expect}")

    ##############################################
    # Continuous (ensemble-averaged) expressions #
    ##############################################

    def getD_continuous(self):
        r"""Continuous shape-noise term on the diagonal,  S+02 Eq. 27.

        .. math::

            D(\theta) = \frac{\sigma_\epsilon^4 \langle w^2\rangle^2}{N_p(\theta)\langle w\rangle^2}

        The pair counts are the idealised ones, so this agrees with :meth:`getD_discrete`.
        """
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        return self.sigma_eps**4 * ww_factor / self.pairs

    def _triplet_term_continuous_c(self, w_type, xi_func,
                                   nr_target=1024, nphi_target=1024):
        """Continuum limit of q_++, q_-- or q_+- via the C kernel, S+02 Eq. 32.

        The integrand is evaluated on ``nphi_target`` angles and ``nr_target // n_r`` sub-bins per
        radial bin and bin-averaged back. ``w_type`` is as in :func:`_triplet_weight`.
        """
        fn = self._clib.continuous_q_single
        # The third side of a triplet reaches far below the smallest bin
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = \
            self._xi_tables_from_splines(
                [xi_func], lo=self._xi_query_lo(), hi=2. * self.theta_edges[-1])

        alpha = int(nr_target // self.n_r)
        n_fine = alpha * self.n_r
        _edges = np.geomspace(self.theta_edges[0], self.theta_edges[-1], n_fine + 1)
        thetas = .5 * (_edges[1:] + _edges[:-1])
        phis = np.linspace(0., np.pi, nphi_target, endpoint=False)

        out = np.zeros(self.n_r * self.n_r, dtype=np.float64)
        fn(self.n_r, alpha,
           np.ascontiguousarray(thetas, dtype=np.float64),
           np.ascontiguousarray(thetas * np.diff(_edges), dtype=np.float64),
           np.ascontiguousarray(phis, dtype=np.float64),
           np.ascontiguousarray(_trapz_weights(phis), dtype=np.float64),
           n_fine, nphi_target,
           xi_x, xi_n, xi_logx0, xi_inv_dlogx,
           np.ascontiguousarray(xi_y_tables[0]),
           int(w_type),
           float(self.sigma_eps**2), float(self.area), float(self.nbar),
           np.ones(self.n_r, dtype=np.float64), out)
        return out.reshape(self.n_r, self.n_r)

    def getqpp_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_++. S+02 Eq. 32."""
        return self._triplet_term_continuous_c(0, self.xip, nr_target, nphi_target)

    def getqmm_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_--. S+02 Eq. 36."""
        return self._triplet_term_continuous_c(1, self.xip, nr_target, nphi_target)

    def getqpm_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_+-. S+02 Eq. 38."""
        return self._triplet_term_continuous_c(2, self.xim, nr_target, nphi_target)

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

    ####################################
    # Analytic (Poisson) count tensors #
    ####################################

    def _suppression(self, r):
        """Suppression s(r) of the idealised counts, 1 if not set."""
        if self.spl_suppression_ww is None:
            return np.ones_like(np.asarray(r, dtype=float))
        return self.spl_suppression_ww(r)

    def pairs_analytic(self):
        r"""Idealised pair counts.

        .. math::

            N_p^\mathrm{ideal}(\theta) = \pi A \bar{n}^2 \langle w\rangle^2
            (\theta_\mathrm{max}^2 - \theta_\mathrm{min}^2)\, s(\theta)
        """
        if self.area is None or self.nbar is None:
            raise ValueError("`nbar` and `area` are required for analytic pairs")
        return (2. * np.pi * self.area * self.nbar**2
                * self._w1**2
                * _theta_dtheta(self.theta_edges)
                * self._suppression(self.theta_centers))

    @staticmethod
    def suppression_spline_from_pairs(Np_meas, theta_centers, theta_edges,
                                       area, nbar, weights_mean=1.):
        """Pair-count suppression s(theta) = N_p(theta) / N_p^ideal(theta) of measured pairs.

        Returns a callable for ``spl_suppression_ww``, linear in log theta, constant below and zero
        above the tabulated range, with its knots as attribute ``x``. A staticmethod, since
        ``method_quadruplets='frompairs'`` needs the spline at construction.
        """
        theta_centers = np.asarray(theta_centers, dtype=float)
        theta_edges = np.asarray(theta_edges, dtype=float)
        ideal = (2. * np.pi * area * nbar**2 * weights_mean**2
                * _theta_dtheta(theta_edges))
        supp_vals = np.asarray(Np_meas, dtype=float) / ideal

        # Set to supp(thetamin) for theta<thetamin and zero for theta>thetamax
        supp_spl = interp1d(np.log(theta_centers), supp_vals,
                            bounds_error=False,
                            fill_value=(supp_vals[0], 0.))

        def sup(theta):
            return supp_spl(np.log(np.asarray(theta, dtype=float)))
        sup.x = theta_centers
        return sup

    def triplets_analytic(self):
        r"""Idealised triplet counts on the oversampled grid.

        .. math::

            N_t^\mathrm{ideal}(\theta_1, \theta_2, \phi) = 2\pi A \bar{n}^3 \langle w\rangle^3
            M_1 M_2\, \Delta\phi\, s(\theta_1) s(\theta_2)

        with :math:`M = (\theta_\mathrm{max}^2 - \theta_\mathrm{min}^2)/2` the radial measure of a bin.
        """
        if self.area is None or self.nbar is None:
            raise ValueError("`nbar` and `area` are required for analytic triplets")
        T1, T2, PHI = np.meshgrid(self.triplet_theta_centers,
                                   self.triplet_theta_centers,
                                   self.phi_centers, indexing='ij')
        M1, M2, DPHI = np.meshgrid(_theta_dtheta(self.triplet_theta_edges),
                                   _theta_dtheta(self.triplet_theta_edges),
                                   self.dphi, indexing='ij')
        r_jk = np.sqrt(T1**2 + T2**2 - 2. * T1 * T2 * np.cos(PHI))
        s_factor = (self._suppression(T1) * self._suppression(T2))
        return (2. * np.pi * self.area * self.nbar**3
                * self._w1**3
                * M1 * M2 * DPHI
                * s_factor)

    def quadruplets_analytic(self):
        """Idealised quadruplet density.

        Returns the callable q = nbar^4 <w>^4, times s(t1) s(t2) s(t3) if ``spl_suppression_ww`` is
        set, as ``method_quadruplets='frompairs'`` requires. Warns if the knots of the spline do not
        cover ``q_theta_centers``.
        """
        if self.nbar is None:
            raise ValueError("`nbar` is required for analytic quadruplets")
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
        # Recorded for the C kernel here, before GGCovariance replaces self.nbar by its per-bin
        # array
        const = self.nbar**4 * self._w1**4
        self._q_const = const
        if self.spl_suppression_ww is None:
            def q_func(t1, t2, t3, a12, a13):
                return const
        else:
            sup = self.spl_suppression_ww
            def q_func(t1, t2, t3, a12, a13):
                return const * (sup(t1) * sup(t2) * sup(t3))
        return q_func

    ###########################################
    # Helper: set quadruplets from raw counts #
    ###########################################

    def set_quadruplets_from_counts(self, N_q,
                                    theta_centers=None,
                                    phi_centers=None,
                                    dtheta_bin=None,
                                    dphi_bin=None,
                                    return_volume=False):
        r"""Convert quadruplet counts to the density q = N_q / V_q and store it.

        .. math::

            V_q = 2\pi A\, \theta_1 \Delta\theta_1\, \theta_2 \Delta\theta_2\,
            \theta_3 \Delta\theta_3\, \Delta\phi_{12} \Delta\phi_{13}

        The grids default to the quadruplet grids of the instance.
        """
        if self.area is None:
            raise ValueError("`area` is required to convert counts to density")

        rc = (self.q_theta_centers if theta_centers is None
              else np.asarray(theta_centers, dtype=float))
        pc = (self.q_phi_centers if phi_centers is None
              else np.asarray(phi_centers, dtype=float))
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
        self._q_func = _build_q_func(self.quadruplets, rc, pc)

        if return_volume:
            return (2. * np.pi * self.area
                    * tdt[:, None, None, None, None]
                    * tdt[None, :, None, None, None]
                    * tdt[None, None, :, None, None]
                    * dp[None, None, None, :, None]
                    * dp[None, None, None, None, :])

    #######################################
    # Individual covariance contributions #
    #######################################

    def getD(self):
        """Diagonal shape-noise term, via :meth:`getD_continuous` or :meth:`getD_discrete`."""
        self.D = (self.getD_continuous() if self.method_pairs == 'continuous'
                  else self.getD_discrete())
        return self.D

    def getD_discrete(self):
        r"""Shape-noise term on the diagonal from the pair counts. S+02 Eq. 27.

        .. math::

            D(\theta) = \frac{\sigma_\epsilon^4 \langle w^2\rangle^2}{N_p(\theta)\langle w\rangle^2}
        """
        if self.pairs is None:
            raise RuntimeError("`pairs` not initialised.")
        Np = np.asarray(self.pairs)
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        self.D = self.sigma_eps**4 * ww_factor / Np
        return self.D

    ## Triplet (mixed) terms ##
    def _triplet_phi_reduce(self, Nt, tc, phic, weight_type, xi_func):
        """Sum of Nt[i, j, k] w(phi_k) xi(r_jk) over k on the fine grid, via the C kernel."""
        nr = len(tc)
        # The third side of a triplet reaches far below the smallest bin
        xi_x, xi_y, logx0, inv_dlogx, xi_n = self._xi_tables_from_splines(
            [xi_func], lo=min(tc[0], self._xi_query_lo()), hi=2. * tc[-1])
        out = np.zeros((nr, nr), dtype=np.float64)
        self._clib.triplet_term_reduce(
            np.ascontiguousarray(Nt, dtype=np.float64), nr, len(phic),
            np.ascontiguousarray(tc, dtype=np.float64),
            np.ascontiguousarray(phic, dtype=np.float64),
            int(weight_type), xi_x, xi_n, logx0, inv_dlogx, xi_y[0],
            out, int(self.nthreads), 0)
        check_clib_error(self._clib)
        return out

    def _triplet_term(self, weight_type, xi_func, pair_mask=None,
                      sigma_eps_sq=None, nbar_fac=1.):
        """Mixed term q_++, q_-- or q_+- from the triplet counts.

        Sums Nt w xi(r_jk) over phi and over the sub-bins of each pair of coarse bins and divides by
        N_p(theta1) N_p(theta2). With ``pair_mask`` only the flagged elements are evaluated, in
        numpy, and the others are zero. ``sigma_eps_sq`` and ``nbar_fac`` are sigma_eps^2 and
        nbar_ref / nbar of the shared galaxy in the tomographic blocks.
        """
        if self.triplets is None:
            raise RuntimeError("`triplets` not initialised.")
        Nt = np.asarray(self.triplets) # (triplet_n_r, triplet_n_r, n_phi)
        Np = np.asarray(self.pairs) # (n_r,)
        n = self.n_r
        alpha = self.triplet_oversample
        tc = self.triplet_theta_centers
        phic = self.phi_centers
        if sigma_eps_sq is None:
            sigma_eps_sq = self.sigma_eps**2
        pre = 2. * sigma_eps_sq * nbar_fac / np.outer(Np, Np)

        if pair_mask is None:
            fine = self._triplet_phi_reduce(Nt, tc, phic, weight_type, xi_func)
            # Collapse fine sub-bins to coarse grid via summation
            if alpha > 1:
                fine = fine.reshape(n, alpha, n, alpha).sum(axis=(1, 3))
            return pre * fine

        ## Banded evaluation: only the coarse (i, j) inside pair_mask ##
        summed = np.zeros((n, n))
        PHI = phic[None, None, :]
        cphi = np.cos(PHI)
        sphi = np.sin(PHI)
        row_block = 8
        for c0 in range(0, n, row_block):
            c1 = min(c0 + row_block, n)
            cols = np.where(pair_mask[c0:c1].any(axis=0))[0]
            if cols.size == 0:
                continue
            jc0, jc1 = int(cols.min()), int(cols.max()) + 1
            tr = slice(c0 * alpha, c1 * alpha)
            tcsl = slice(jc0 * alpha, jc1 * alpha)
            T1 = tc[tr][:, None, None]
            T2 = tc[tcsl][None, :, None]
            jk_x = T1 - T2 * cphi
            jk_y = -T2 * sphi
            r_jk = np.hypot(jk_x, jk_y)
            weight = _triplet_weight(weight_type, PHI, jk_x, jk_y)
            fine = np.sum(Nt[tr, tcsl, :] * weight * xi_func(r_jk), axis=2)
            nr_c, nc_c = c1 - c0, jc1 - jc0
            if alpha > 1:
                fine = fine.reshape(nr_c, alpha, nc_c, alpha).sum(axis=(1, 3))
            summed[c0:c1, jc0:jc1] = fine

        return np.where(pair_mask, pre * summed, 0.)

    ## Mixed-term (q) getters ##
    # (name, weight_type, xi attribute), with the weight ids of _triplet_weight
    QTERMS = (('qpp', 0, 'xip'), ('qmm', 1, 'xip'), ('qpm', 2, 'xim'))

    def _get_qterm(self, index, pair_mask=None):
        """One mixed term, by its position in :attr:`QTERMS`."""
        name, weight_type, xi_attr = self.QTERMS[index]
        if self.method_triplets == 'continuous':
            value = self._triplet_term_continuous_c(
                weight_type, getattr(self, xi_attr))
        else:
            value = self._triplet_term(
                weight_type, getattr(self, xi_attr), pair_mask=pair_mask)
        setattr(self, name, value)
        return value

    def getqpp(self, pair_mask=None):
        """Mixed term for Cov_pp.S+02 Eq. 32.

        ``pair_mask`` only applies to the discrete path.
        """
        return self._get_qterm(0, pair_mask)

    def getqmm(self, pair_mask=None):
        """Mixed term for Cov_mm. S+02 Eq. 36.

        ``pair_mask`` only applies to the discrete path.
        """
        return self._get_qterm(1, pair_mask)

    def getqpm(self, pair_mask=None):
        """Mixed term for Cov_pm. S+02 Eq. 38.

        ``pair_mask`` only applies to the discrete path.
        """
        return self._get_qterm(2, pair_mask)

    ## C-accelerated quadruplet integral ##
    def _rterms_continuous_c(self, dlnt_target=.05):
        """All five continuum-limit r-terms in one C call, averaged over sub-bins of each bin.

        The non-tomographic case has a single Wick pairing, without the factor 0.5 of
        :meth:`get_continuous_tomo_c`. The two side integrals of r_-0 and r_-1 coincide, so their
        outer products are symmetric, while r_+- weights side b only and is not symmetric.
        """
        fn = self._clib.continuous_rterms_tomo
        # Memoised against the grids, since the getters ask for one term at a time
        cached = getattr(self, '_rterms_cont_cache', None)
        if (cached is not None and cached[0] is self.phi_grid
                and cached[1] is self.varphi1_grid and cached[2] == self.n_r):
            return cached[3]
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = \
            self._xi_tables_from_splines(
                [self.xip, self.xim],
                lo=self._xi_query_lo(), hi=self._xi_query_hi())

        v = np.ascontiguousarray(self.varphi1_grid, dtype=np.float64)
        v_half = np.ascontiguousarray(v[v <= np.pi], dtype=np.float64)
        n = self.n_r
        nodes, weights = self._rterm_nodes(dlnt_target)
        n_f = nodes.size
        results = np.zeros(5 * n_f * n_f, dtype=np.float64)
        idx0 = np.zeros(1, dtype=np.int32)
        fn(n_f, np.ascontiguousarray(nodes.ravel(), dtype=np.float64),
           np.ascontiguousarray(self.phi_grid, dtype=np.float64),
           np.ascontiguousarray(_trapz_weights(self.phi_grid), dtype=np.float64),
           len(self.phi_grid),
           v, np.ascontiguousarray(_periodic_weights(v), dtype=np.float64), len(v),
           v_half, np.ascontiguousarray(_trapz_weights(v_half), dtype=np.float64),
           len(v_half),
           float(self.area),
           xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y_tables, 2,
           1, idx0, idx0, idx0 + 1, idx0 + 1, idx0,
           5, results, int(self.nthreads), 0)
        out = _bin_average_blocks(results.reshape(5, n_f, n_f), weights)
        self._rterms_cont_cache = (self.phi_grid, self.varphi1_grid, n, out)
        return out

    def _rterm_nodes(self, dlnt_target):
        """Nodes (n_r, ns) and theta dtheta weights of equal log sub-bins no wider than dlnt_target.

        With a single sub-bin per bin the node is the bin centre.
        """
        dlnt = np.diff(np.log(self.theta_edges))
        ns = max(1, int(np.ceil(np.max(dlnt) / dlnt_target - 1e-9)))
        if ns == 1:
            return self.theta_centers[:, None], _theta_dtheta(self.theta_edges)[:, None]
        sub = np.exp(np.log(self.theta_edges[:-1])[:, None] + np.outer(dlnt, np.arange(ns + 1) / ns))
        return .5 * (sub[:, 1:] + sub[:, :-1]), .5 * (sub[:, 1:]**2 - sub[:, :-1]**2)

    def _xi_query_lo(self):
        """Smallest separation at which the quadruplet kernels evaluate xi_pm."""
        return min(self.theta_edges[0], self.phi_grid[0])

    def _xi_query_hi(self):
        """Largest separation at which the quadruplet kernels evaluate xi_pm."""
        return self.phi_grid[-1] + self.theta_edges[-1]

    @staticmethod
    def _xi_tables_from_splines(splines, lo=None, hi=None):
        """Tabulate xi splines on shared log-uniform knots for the C kernels.

        Uses the ``.x`` knots of the splines if present, which reproduces a linear ``interp1d``
        exactly, and 4096 log-spaced points in [lo, hi] otherwise. Returns ``(xi_x, xi_y_tables,
        xi_logx0, xi_inv_dlogx, xi_n)``.
        """
        x = getattr(splines[0], 'x', None)
        if x is None:
            if lo is None or hi is None:
                raise TypeError(
                    "xi splines carry no `.x` knots and no sampling range was "
                    "given; pass an interp1d/spline or supply lo/hi.")
            x = np.geomspace(lo, hi, 4096)
        xi_x = np.ascontiguousarray(x, dtype=np.float64)
        xi_n = len(xi_x)
        xi_y_tables = np.empty((len(splines), xi_n), dtype=np.float64)
        for t, spl in enumerate(splines):
            xi_y_tables[t] = spl(xi_x)
        return (xi_x, np.ascontiguousarray(xi_y_tables),
                float(np.log(xi_x[0])),
                float(1. / (np.log(xi_x[1]) - np.log(xi_x[0]))), xi_n)

    def _run_postq_engine(self, cbm, xi_tables, symmetric_mask,
                          verbose=False, pair_mask=None, nthreads=None):
        """Run the fused C quadruplet kernel over the phi grid with ``nthreads`` threads.

        Returns one (n_r, n_r) block per output, including the prefactor 2 pi A / (N_p N_p).
        """
        if nthreads is None:
            nthreads = self.nthreads
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = xi_tables
        n_out = cbm['n_out']
        n = self.n_r

        def _f(a): return np.ascontiguousarray(a, dtype=np.float64)
        def _i(a): return np.ascontiguousarray(a, dtype=np.int32)

        ## Radial integration nodes: one row of ns subgrid nodes per theta bin ##
        if self.phi1_subgrid_frac is None:
            phi1_nodes = self.theta_centers[:, None]
            phi1_weights = _theta_dtheta(self.theta_edges)[:, None]
        else:
            s1f = np.asarray(self.phi1_subgrid_frac, dtype=float)
            phi1_nodes = self.theta_centers[:, None] + np.outer(self.dtheta, s1f)
            phi1_weights = phi1_nodes * self.dtheta[:, None] * _trapz_weights(s1f)[None, :]
        if self.phi2_subgrid_frac is None:
            phi2_nodes = self.theta_centers[:, None]
            phi2_weights = _theta_dtheta(self.theta_edges)[:, None]
        else:
            s2f = np.asarray(self.phi2_subgrid_frac, dtype=float)
            phi2_nodes = self.theta_centers[:, None] + np.outer(self.dtheta, s2f)
            phi2_weights = phi2_nodes * self.dtheta[:, None] * _trapz_weights(s2f)[None, :]
        ns1, ns2 = phi1_nodes.shape[1], phi2_nodes.shape[1]
        phi1_flat = phi1_nodes.ravel()
        phi2_flat = phi2_nodes.ravel()

        phi_grid = self.phi_grid
        phi_w = _trapz_weights(phi_grid)
        v1_w = _periodic_weights(self.varphi1_grid)
        v2_w = _periodic_weights(self.varphi2_grid)

        ## Tile plan over the (theta1, theta2) axes, banded when masked ##
        nrb = self.nr_block or n
        rows = range(0, n, nrb)
        if pair_mask is None:
            tiles = [(i1, i2) for i1 in rows for i2 in rows]
        else:
            tiles = [(i1, i2) for i1 in rows if pair_mask[i1:i1 + nrb].any()
                     for i2 in rows
                     if pair_mask[i1:i1 + nrb, i2:i2 + nrb].any()]
        mask_c = _i(np.ones((n, n)) if pair_mask is None else pair_mask)

        ## Quadruplet density: 0 = tensor, 1 = constant, 2 = separable s(t) ##
        dummy = np.zeros(2)
        if self.quadruplets is None:
            raise RuntimeError("`quadruplets` not initialised.")
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
            if self.spl_suppression_ww is None:
                q_mode = 1
                sup_t1 = sup_t2 = sup_t3 = dummy
            else:
                # s(t1) s(t2) s(t3) with t1 = phi, t2 the theta1 node and t3 = |psi_b|, tabulated
                # here so that the kernel needs no callbacks
                q_mode = 2
                sup = self.spl_suppression_ww
                sup_t1, sup_t2 = _f(sup(phi_grid)), _f(sup(phi1_flat))
                bx = (phi_grid[:, None, None] + phi2_flat[None, :, None]
                      * np.cos(self.varphi2_grid)[None, None, :])
                by = phi2_flat[None, :, None] * np.sin(self.varphi2_grid)[None, None, :]
                sup_t3 = _f(sup(np.hypot(bx, by)))

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
            len(cbm['xi_a_idx']), _i(cbm['xi_a_idx']), _i(cbm['xi_b_idx']),
            _i(cbm['w_ids']), _i(cbm['out_idx']), _f(cbm['wick_fac']),
            n_out, _i(symmetric_mask), mask_c,
            results, nthreads, int(verbose))
        # Raises if the per-thread scratch of the kernel failed to allocate
        check_clib_error(self._clib)

        if pair_mask is not None:
            results = np.where(pair_mask, results, 0.)

        pre = 2. * np.pi * self.area / np.outer(self.pairs, self.pairs)
        return [pre * a for a in results]

    def _build_notomo_callback_map(self):
        """Callback map of the five non-tomographic r-terms, all with Wick factor 1.

        Table 0 is xi_+ and table 1 is xi_-. The ``w_ids`` follow :attr:`RTERMS`:

        * 0 is r_+0 with w = 1
        * 1 is r_+1 with w = cos(4(phi_a - phi_b))
        * 2 is r_-0 with w = cos(4(V1 + V2 - phi_a - phi_b))
        * 3 is r_-1 with w = cos(4(V1 - V2))
        * 4 is r_+- with w = 2 cos(4(V2 - phi_b))
        """
        return dict(
            xi_a_idx=[0, 1, 1, 0, 0],
            xi_b_idx=[0, 1, 1, 0, 1],
            w_ids=[0, 1, 2, 3, 4],
            out_idx=[0, 1, 2, 3, 4],
            wick_fac=[1.] * 5,
            n_out=5)

    def _quadruplet_density_terms_c(self, verbose=False, pair_mask=None,
                                    nthreads=None):
        """Non-tomographic r-terms (rp0, rp1, rm0, rm1, rpm) via the C kernel.

        xi_pm enter through the piecewise-linear table of the kernel, which is exact for a linear
        ``interp1d``.
        """
        xi_tables = self._xi_tables_from_splines(
            [self.xip, self.xim], lo=self._xi_query_lo(), hi=self._xi_query_hi())
        return self._run_postq_engine(
            self._build_notomo_callback_map(), xi_tables,
            [True, True, True, True, False],
            verbose=verbose, pair_mask=pair_mask, nthreads=nthreads)

    ## R-term (cosmic-variance) getters ##
    RTERMS = ('rp0', 'rp1', 'rm0', 'rm1', 'rpm')

    def get_all_rterms(self, verbose=False, nthreads=None, pair_mask=None):
        """Compute rp0, rp1, rm0, rm1 and rpm in a single pass of the quadruplet integral.

        Parameters
        ----------
        verbose: bool, optional
            Print the progress of the integral. Defaults to ``False``.
        nthreads: int, optional
            Number of OpenMP threads. Defaults to ``nthreads`` of the instance.
        pair_mask: numpy.ndarray, optional
            Boolean mask of shape (n_r, n_r) that restricts the integral to a band around the
            diagonal, e.g. the union of the bands of all blocks.

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
        self.rp0, self.rp1, self.rm0, self.rm1, self.rpm = results
        return tuple(results)

    def _ensure_all_rterms(self, verbose=False, nthreads=None):
        """Compute all five r-terms if any of them is missing."""
        if all(getattr(self, n) is not None for n in self.RTERMS):
            return
        if self.method_quadruplets == 'continuous':
            for i, name in enumerate(self.RTERMS):
                setattr(self, name, self._rterms_continuous_c()[i])
        else:
            self.get_all_rterms(verbose=verbose, nthreads=nthreads)

    def _get_rterm(self, index, verbose=False, nthreads=None, pair_mask=None):
        """One r-term by its position in :attr:`RTERMS`.

        The discrete path computes all five at once, since they share the quadruplet interpolation.
        """
        name = self.RTERMS[index]
        if self.method_quadruplets == 'continuous':
            value = self._rterms_continuous_c()[index]
        else:
            value = self.get_all_rterms(
                verbose=verbose, nthreads=nthreads, pair_mask=pair_mask)[index]
        setattr(self, name, value)
        return value

    def getrp0(self, verbose=False, nthreads=None, pair_mask=None):
        """r_+0 contribution to Cov_pp cosmic variance."""
        return self._get_rterm(0, verbose, nthreads, pair_mask)

    def getrp1(self, verbose=False, nthreads=None, pair_mask=None):
        """r_+1 contribution to Cov_pp cosmic variance."""
        return self._get_rterm(1, verbose, nthreads, pair_mask)

    def getrm0(self, verbose=False, nthreads=None, pair_mask=None):
        """r_-0 contribution to Cov_mm cosmic variance."""
        return self._get_rterm(2, verbose, nthreads, pair_mask)

    def getrm1(self, verbose=False, nthreads=None, pair_mask=None):
        """r_-1 contribution to Cov_mm cosmic variance."""
        return self._get_rterm(3, verbose, nthreads, pair_mask)

    def getrpm(self, verbose=False, nthreads=None, pair_mask=None):
        """r_+- contribution to the Cov_pm cosmic variance, not symmetric in theta1 <-> theta2."""
        return self._get_rterm(4, verbose, nthreads, pair_mask)

    def smooth_rterms(self, window='gaussian', window_size=3.,
                      nphi_fine=1000, nvarphi_fine=2000,
                      terms=('rp0', 'rp1', 'rm0', 'rm1', 'rpm'),
                      floor_frac=1e-3):
        r"""Ratio-smooth the discrete r-terms against their continuum limit.

        .. math::

            C_\mathrm{corr} = C_\infty\, \mathrm{smooth}(C_\mathrm{est} / C_\infty)

        mitigates the stripes from the binning of the quadruplet density, with C_inf evaluated on
        finer ``phi_grid`` and ``varphi1_grid``. Covariance blocks assembled before are reassembled.

        Parameters
        ----------
        window: {'gaussian', 'tophat'}, optional
            Smoothing window. Defaults to ``'gaussian'``.
        window_size: float, optional
            Gaussian sigma or tophat width in bins. Defaults to ``3``.
        nphi_fine, nvarphi_fine: int, optional
            Sizes of the fine ``phi_grid`` and ``varphi1_grid``. Default to ``1000`` and
            ``2000``.
        terms: tuple of str, optional
            The r-terms to correct. Defaults to all five.
        floor_frac: float, optional
            Floor of abs(C_inf) relative to its maximum, which regularises the ratio near the
            zero crossings of C_inf. Defaults to ``1e-3``.

        Returns
        -------
        tuple of numpy.ndarray
            The corrected terms of shape (n_r, n_r) in the order of ``terms``.
        """
        if self.method_quadruplets == 'continuous':
            raise ValueError(
                "smooth_rterms requires a discrete r-term estimate "
                "(method_quadruplets != 'continuous')")
        self._ensure_all_rterms()

        if window == 'gaussian':
            def smooth(a):
                return gaussian_filter(a, sigma=window_size, mode='nearest')
        elif window == 'tophat':
            def smooth(a):
                return uniform_filter(a, size=int(round(window_size)),
                                      mode='nearest')
        else:
            raise ValueError(f"Unknown window='{window}'")

        phi_grid_save, varphi1_save = self.phi_grid, self.varphi1_grid
        self.phi_grid = np.geomspace(phi_grid_save[0], phi_grid_save[-1], nphi_fine)
        self.varphi1_grid = np.linspace(0., 2. * np.pi, nvarphi_fine, endpoint=False)
        try:
            for name in terms:
                C_est = getattr(self, name)
                C_inf = getattr(self, f'get{name}_continuous')()
                floor = floor_frac * np.max(np.abs(C_inf))
                C_inf_safe = np.where(np.abs(C_inf) < floor,
                                       np.where(C_inf >= 0, floor, -floor), C_inf)
                setattr(self, name, C_inf_safe * smooth(C_est / C_inf_safe))
        finally:
            self.phi_grid, self.varphi1_grid = phi_grid_save, varphi1_save

        # Update all other cov terms in case they were computed before
        had_cov_pp, had_cov_mm = self.cov_pp is not None, self.cov_mm is not None
        had_cov_pm, had_cov_xipm = self.cov_pm is not None, self.cov_xipm is not None
        self.cov_pp = self.cov_mm = self.cov_pm = self.cov_xipm = None
        if had_cov_pp: self.get_cov_pp()
        if had_cov_mm: self.get_cov_mm()
        if had_cov_pm: self.get_cov_pm()
        if had_cov_xipm: self.get_cov_xipm()

        return tuple(getattr(self, name) for name in terms)

    ############################
    # Full covariance assembly #
    ############################

    # (block, label, q term, r terms, shape noise on the diagonal)
    COVBLOCKS = (('pp', '++', 'qpp', ('rp0', 'rp1'), True),
                 ('mm', '--', 'qmm', ('rm0', 'rm1'), True),
                 ('pm', '+-', 'qpm', ('rpm',), False))

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

    def get_cov_pp(self, verbose=False, nthreads=None):
        """Assemble the full Cov_pp matrix."""
        return self._get_cov_block(0, verbose, nthreads)

    def get_cov_mm(self, verbose=False, nthreads=None):
        """Assemble the full Cov_mm matrix."""
        return self._get_cov_block(1, verbose, nthreads)

    def get_cov_pm(self, verbose=False, nthreads=None):
        """Assemble the full Cov_pm matrix. No diagonal shape-noise term."""
        return self._get_cov_block(2, verbose, nthreads)

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

    #######################
    # Multiplet accessors #
    #######################

    def get_pairs(self):
        """Return the pair-count vector of length n_r."""
        return np.asarray(self.pairs)

    def get_triplets(self):
        """Return the triplet-count tensor of shape (triplet_n_r, triplet_n_r, n_phi)."""
        return np.asarray(self.triplets)

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
        return np.asarray(self.quadruplets)

    @classmethod
    def _from_components(cls, theta_centers, theta_edges,
                         phi_centers, phi_edges,
                         spline_xip, spline_xim, sigma_eps,
                         nbar, area, weights_mean, weights_std,
                         D=None, qpp=None, qmm=None, qpm=None,
                         rp0=None, rp1=None, rm0=None, rm1=None, rpm=None,
                         nthreads=None, template=None, pairs=None, **tomo):
        """Create an instance from precomputed covariance components, bypassing ``__init__``.

        ``template`` passes on the quadruplet and integration grids, the suppression spline and the
        tiling, ``pairs`` the fine pair counts, and ``tomo`` the tomographic blocks and their
        bookkeeping, which are set as attributes.
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

        # q-tensor grid defaults to the 2PCF grid for precomputed instances,
        # or follows the template when one is given.
        for name, default in (('q_theta_centers', obj.theta_centers),
                              ('q_theta_edges', obj.theta_edges),
                              ('q_dtheta', obj.dtheta), ('q_n_r', obj.n_r),
                              ('q_phi_centers', obj.phi_centers),
                              ('q_phi_edges', obj.phi_edges),
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
        # Tomographic instances carry per-bin arrays here, non-tomo a scalar.
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
        obj._q_func = None

        def _a(x):
            return np.asarray(x, dtype=float) if x is not None else None

        obj.D = _a(D)
        obj.qpp = _a(qpp)
        obj.qmm = _a(qmm)
        obj.qpm = _a(qpm)
        obj.rp0 = _a(rp0)
        obj.rp1 = _a(rp1)
        obj.rm0 = _a(rm0)
        obj.rm1 = _a(rm1)
        obj.rpm = _a(rpm)
        obj.cov_pp = None
        obj.cov_mm = None
        obj.cov_pm = None
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

    _RECOMPUTE_FULL_COMPONENTS = ('D', 'qpp', 'qmm', 'qpm')

    def _grid_kwargs(self, new_tc, new_te, **overrides):
        """Constructor arguments for this instance on a new theta grid, updated by ``overrides``."""
        kw = dict(
            theta_centers=new_tc, theta_edges=new_te,
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
            nthreads=self.nthreads)
        kw.update(overrides)
        return kw

    def _fine_instance_from_catalog(self, new_tc, new_te, verbose=False):
        """Build a fine-grid instance for :meth:`subbin` from the catalog of this instance.

        The fine pairs and triplets are measured, the quadruplet density is reused.
        """
        if getattr(self, 'catalog', None) is None:
            raise ValueError(
                "_fine_instance_from_catalog requires a `catalog` on this "
                "instance.")
        if verbose:
            print("  subbin: auto-building fine instance (n_fine=%d) -- "
                  "measuring fine pairs/triplets, reusing q density"
                  % len(new_tc))
        return GGCovarianceNoTomo(**self._grid_kwargs(
            new_tc, new_te,
            method_pairs='discrete', method_triplets='discrete',
            method_quadruplets='discrete',
            catalog=self.catalog, store_multiplets=self.store_multiplets,
            quadruplets=self.quadruplets))

    def subbin(self, new_theta_centers, new_theta_edges, new_pairs=None,
               infer_pairs=False, use_corr=True, log_theta=True,
               recompute_slab=False, slab_width=None, slab_nr_block=None,
               slab_instance=None, recompute_full=None, verbose=False, dry=False):
        """Resample the covariance to a finer radial binning by interpolation.

        Each component is interpolated on its own, by default in log theta. Plain interpolation
        undershoots the diagonal peak of strongly diagonal components, notably xi_- xi_-, which
        ``recompute_slab`` repairs by evaluating a band around the diagonal on the fine grid.

        Parameters
        ----------
        new_theta_centers: numpy.ndarray
            Centers of the fine radial bins, shape (n_new,).
        new_theta_edges: numpy.ndarray
            Edges of the fine radial bins, shape (n_new+1,).
        new_pairs: numpy.ndarray, optional
            Measured fine pair counts, shape (n_new,). The components are then interpolated as the densities arr
            N_p(theta1) N_p(theta2) / (M_1 M_2), M the integral of theta dtheta over a bin, and D is
            recomputed.
            Without them the components are interpolated directly.
        infer_pairs: bool, optional
            Estimate the fine pair counts from the coarse ratio N_p / N_p^ideal. Defaults to
            ``False``.
        use_corr: bool, optional
            Interpolate symmetric components via their correlation coefficient. Defaults to
            ``True``.
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
            Instance on the fine grid that supplies the counts of the band. Built from
            ``catalog`` if not given, or from the idealised counts for the analytic method.
        recompute_full: bool or iterable of {'D', 'qpp', 'qmm', 'qpm'}, optional
            Components evaluated on the whole fine grid by ``slab_instance``, all four for
            ``True``.
        verbose: bool, optional
            Print the progress of the quadruplet integrals. Defaults to ``False``.
        dry: bool, optional
            Only print and return the numbers of elements and tiles of each band. Defaults to
            ``False``.

        Returns
        -------
        GGCovarianceNoTomo
            Instance on the fine grid with all components loaded.
        """
        if dry:
            return self._subbin_dry_run(
                new_theta_centers, slab_width, slab_nr_block,
                slab_instance, recompute_slab)

        new_tc = np.asarray(new_theta_centers, dtype=float)
        new_te = np.asarray(new_theta_edges, dtype=float)
        if len(new_te) != len(new_tc) + 1:
            raise ValueError(
                "new_theta_edges must have exactly one more entry than "
                "new_theta_centers")

        # Build the fine slab instance from the catalog if none is given
        if (slab_instance is None and (recompute_slab or recompute_full)
                and getattr(self, 'catalog', None) is not None):
            slab_instance = self._fine_instance_from_catalog(
                new_tc, new_te, verbose=verbose)
            if new_pairs is None and not infer_pairs:
                new_pairs = slab_instance.pairs

        if recompute_full:
            full_set = (set(self._RECOMPUTE_FULL_COMPONENTS)
                        if recompute_full is True else set(recompute_full))
            if not full_set <= set(self._RECOMPUTE_FULL_COMPONENTS):
                raise ValueError(
                    f"recompute_full entries must be a subset of "
                    f"{self._RECOMPUTE_FULL_COMPONENTS}, got {sorted(full_set)}")
            if slab_instance is None:
                raise ValueError(
                    "recompute_full requires `slab_instance` (or a `catalog` "
                    "on this instance to auto-build one).")
        else:
            full_set = set()

        if self.D is None: self.getD()
        if self.qpp is None: self.getqpp()
        if self.qmm is None: self.getqmm()
        if self.qpm is None: self.getqpm()
        self._ensure_all_rterms(verbose=verbose)

        n_new = len(new_tc)

        tc = self.theta_centers

        x_c = np.log(tc) if log_theta else tc
        x_new = np.log(new_tc) if log_theta else new_tc

        _interp_arr = self._build_block_interpolator(x_c, x_new, n_new, use_corr)

        ## D and 2-D component strategies ##
        Np_c = np.asarray(self.pairs)
        phiw_c = _theta_dtheta(self.theta_edges) # int theta dtheta, coarse

        if infer_pairs and new_pairs is None:
            if self.area is None or self.nbar is None:
                raise ValueError(
                    "`infer_pairs=True` requires `nbar` and `area` to be set")
            ideal_c = (2. * np.pi * self.area * self.nbar**2
                          * self._w1**2 * phiw_c
                          * self._suppression(tc))
            ratio_c = Np_c / ideal_c
            ideal_f = (2. * np.pi * self.area * self.nbar**2
                          * self._w1**2 * _theta_dtheta(new_te)
                          * self._suppression(new_tc))
            log_ratio_f = np.interp(x_new, x_c, np.log(ratio_c))
            new_pairs = np.exp(log_ratio_f) * ideal_f

        if new_pairs is not None:
            Np_f = np.asarray(new_pairs, dtype=float)
            if Np_f.shape != (n_new,):
                raise ValueError(
                    f"new_pairs must have shape ({n_new},), got {Np_f.shape}")
            phiw_f = _theta_dtheta(new_te)

            # D N_p = sigma_eps^4 <w^2>^2 / <w>^2 is constant
            D_scale = float(np.mean(self.D * Np_c))
            D_fine = D_scale / Np_f

            ow_Np_c = np.outer(Np_c, Np_c)
            ow_phiw_c = np.outer(phiw_c, phiw_c)
            ow_phiw_f = np.outer(phiw_f, phiw_f)
            ow_Np_f = np.outer(Np_f, Np_f)

            def _interp(arr, sym):
                # Interpolate the smooth density and apply the measured fine N_p
                density = arr * ow_Np_c / ow_phiw_c
                result = (_interp_arr.corr(density) if (use_corr and sym)
                           else _interp_arr.raw(
                               density, log_interp=bool(np.all(density > 0))))
                if sym:
                    result = 0.5 * (result + result.T)
                return result * ow_phiw_f / ow_Np_f

        else:
            # Without boundaries N_p ~ int theta dtheta, so q and r do not depend on the bin width
            D_scaled = self.D * phiw_c # independent of the bin width
            D_fine = np.exp(np.interp(x_new, x_c,
                                        np.log(D_scaled))) / _theta_dtheta(new_te)
            _interp = _interp_arr

        _specs = [('qpp', True), ('qmm', True), ('qpm', False),
                  ('rp0', True), ('rp1', True), ('rm0', True),
                  ('rm1', True), ('rpm', False)]
        fine = {name: _interp(getattr(self, name), sym) for name, sym in _specs}
        qpp_f = fine['qpp']
        qmm_f = fine['qmm']
        qpm_f = fine['qpm']
        rp0_f = fine['rp0']
        rp1_f = fine['rp1']
        rm0_f = fine['rm0']
        rm1_f = fine['rm1']
        rpm_f = fine['rpm']

        # Repair the diagonal-dominated band by direct fine-grid evaluation.
        if recompute_slab:
            fine_pairs = (np.asarray(new_pairs, dtype=float)
                          if new_pairs is not None else None)
            (qpp_f, qmm_f, qpm_f,
             rp0_f, rp1_f, rm0_f, rm1_f, rpm_f) = self._recompute_slab(
                new_tc, new_te, fine_pairs, slab_width, slab_nr_block,
                (qpp_f, qmm_f, qpm_f, rp0_f, rp1_f, rm0_f, rm1_f, rpm_f),
                fine=slab_instance, verbose=verbose,
                full_set=full_set & {'qpp', 'qmm', 'qpm'})

        # D and the mixed terms are cheap enough to evaluate on the whole fine grid
        if full_set:
            self._check_slab_instance_grid(slab_instance, new_tc, new_te)
            if 'D' in full_set:
                D_fine = slab_instance.getD()
            if 'qpp' in full_set:
                qpp_f = slab_instance.getqpp()
            if 'qmm' in full_set:
                qmm_f = slab_instance.getqmm()
            if 'qpm' in full_set:
                qpm_f = slab_instance.getqpm()

        return GGCovarianceNoTomo._from_components(
            theta_centers=new_tc,
            theta_edges=new_te,
            phi_centers=self.phi_centers,
            phi_edges=self.phi_edges,
            spline_xip=self.xip,
            spline_xim=self.xim,
            sigma_eps=self.sigma_eps,
            nbar=self.nbar,
            area=self.area,
            weights_mean=self.weights_mean,
            weights_std=self.weights_std,
            D=D_fine,
            qpp=qpp_f,
            qmm=qmm_f,
            qpm=qpm_f,
            rp0=rp0_f,
            rp1=rp1_f,
            rm0=rm0_f,
            rm1=rm1_f,
            rpm=rpm_f,
            nthreads=self.nthreads,
        )

    def _subbin_dry_run(self, new_theta_centers, slab_width, slab_nr_block,
                        slab_instance, recompute_slab):
        """Print element / tile counts for the slab; return summary dict."""
        new_tc = np.asarray(new_theta_centers, dtype=float)
        N = len(new_tc)

        if not recompute_slab or slab_width is None:
            print(f"dry=True: no slab configured "
                  f"(recompute_slab={recompute_slab}, slab_width={slab_width})")
            return {'n_fine': N, 'nr_block': None,
                    'slab_elements': {}, 'n_tiles': {},
                    'total_elements': 0, 'total_integrator_tile_calls': 0}

        widths = self._parse_slab_widths(slab_width)
        masks = {b: self._slab_band_mask(new_tc, w) for b, w in widths.items()}

        # Same tiles as in _recompute_slab
        dln = float(np.mean(np.diff(np.log(new_tc))))
        m_max = max((int(np.ceil(np.log(w) / dln))
                     for w in widths.values() if w > 1.), default=1)
        n_v2 = len(self.varphi2_grid)
        n_v1 = len(self.varphi1_grid)
        n_xi = getattr(self, 'ntomo', 1)*(getattr(self, 'ntomo', 1)+1)

        if slab_nr_block:
            nr_block = int(slab_nr_block)
            v2_block_slab = self.v2_block
        else:
            nr_block, v2_block_slab = _auto_reduce_tiles(
                max(4, m_max), self.v2_block, n_v2, n_v1, n_xi)
        # Rebase on the nr_block of slab_instance if it has one
        if (slab_instance is not None
                and slab_nr_block is None
                and slab_instance.nr_block is not None):
            nr_block, v2_block_slab = _auto_reduce_tiles(
                slab_instance.nr_block, slab_instance.v2_block, n_v2,
                n_v1, n_xi)

        elem_counts = {}
        tile_counts = {}
        for blk, mask in masks.items():
            elem_counts[blk] = int(mask.sum())
            n_tiles = 0
            for i1 in range(0, N, nr_block):
                sl1 = slice(i1, min(i1 + nr_block, N))
                if not mask[sl1].any():
                    continue
                for i2 in range(0, N, nr_block):
                    if mask[sl1, i2:i2 + nr_block].any():
                        n_tiles += 1
            tile_counts[blk] = n_tiles

        # The bands are nested, so one pass over the widest band covers all five r-terms
        max_blk = max(widths, key=widths.get)
        total_elems = sum(elem_counts.values())
        total_calls = tile_counts[max_blk]

        print(f"Slab dry run (fine grid: {N}×{N}, nr_block={nr_block}, "
              f"v2_block={v2_block_slab})")
        for blk in ('pp', 'mm', 'pm'):
            w = widths[blk]
            e = elem_counts[blk]
            t = tile_counts[blk]
            print(f" {blk} band (w={w:.3g}): {e:>8d} elements "
                  f"({e / N**2 * 100:5.1f}%), {t:>5d} tiles")
        print(f" {'Total':15s}  {total_elems:>8d} elements "
              f"({total_elems / N**2 * 100:5.1f}%)")
        print(f" r-terms (rp0,rp1,rm0,rm1,rpm): single batched pass over "
              f"'{max_blk}' band (w={widths[max_blk]:.3g}) "
              f"= {total_calls:>5d} integrator×tile calls")

        return {
            'n_fine': N,
            'nr_block': nr_block,
            'v2_block': v2_block_slab,
            'slab_elements': elem_counts,
            'n_tiles': tile_counts,
            'total_elements': total_elems,
            'total_integrator_tile_calls': total_calls,
        }

    @staticmethod
    def _slab_band_mask(theta_centers, slab_width):
        """Boolean (N, N) band around the diagonal, 1/w < theta_j/theta_i < w."""
        tc = np.asarray(theta_centers, dtype=float)
        N = len(tc)
        if slab_width is None or slab_width <= 1.:
            return np.zeros((N, N), dtype=bool)
        ratio = tc[None, :] / tc[:, None]
        return (ratio < slab_width) & (ratio > 1. / slab_width)

    @staticmethod
    def _build_block_interpolator(x_c, x_new, n_new, use_corr=True):
        """Callable ``interp(arr, sym)`` interpolating an (n_c, n_c) block to (n_new, n_new).

        Zero blocks stay zero.
        """
        # Bilinear interpolation is separable: one 1-D weight matrix applied along both axes
        W = RegularGridInterpolator(
            (x_c,), np.eye(len(x_c)),
            method='linear', bounds_error=False, fill_value=None)(x_new[:, None])

        def _raw(arr, log_interp=False):
            data = np.log(arr) if log_interp else arr
            result = W @ data @ W.T
            return np.exp(result) if log_interp else result

        def _corr(arr):
            diag = np.diag(arr).copy()
            safe = np.where(diag > 0, diag, np.nan)
            ok = np.isfinite(safe)
            normed = np.where(np.isfinite(np.sqrt(np.outer(safe, safe))),
                              arr / np.sqrt(np.outer(safe, safe)), arr)
            pos = bool(np.all(normed > 0))
            normed_f = _raw(normed, log_interp=pos)
            if not pos:
                np.fill_diagonal(normed_f, 1.)
            safe_d = np.where(ok, safe, 0.)
            if ok.all():
                diag_f = np.exp(np.interp(x_new, x_c, np.log(safe_d)))
            else:
                diag_f = np.interp(x_new, x_c, safe_d)
            denom_f = np.sqrt(np.outer(diag_f, diag_f))
            return np.where(denom_f > 0, normed_f * denom_f, normed_f)

        def _interp(arr, sym):
            if not np.any(arr):
                return np.zeros((n_new, n_new))
            result = (_corr(arr) if (use_corr and sym)
                      else _raw(arr, log_interp=bool(np.all(arr > 0))))
            if sym:
                result = .5 * (result + result.T)
            return result

        # Raw and correlation-coefficient interpolators, reused by subbin
        _interp.raw = _raw
        _interp.corr = _corr
        return _interp

    @staticmethod
    def _parse_slab_widths(slab_width):
        """Convert scalar or dict slab_width to per-block dict."""
        if np.isscalar(slab_width):
            return {b: float(slab_width) for b in ('pp', 'mm', 'pm')}
        return {b: float(slab_width.get(b, 1.)) for b in ('pp', 'mm', 'pm')}

    @staticmethod
    def _check_slab_instance_grid(fine, new_tc, new_te):
        """Verify ``fine`` lives on the same 2PCF theta grid as the subbin target."""
        if (fine.n_r != len(new_tc)
                or not np.allclose(fine.theta_edges, new_te)):
            raise ValueError(
                "slab_instance must live on the same 2PCF theta grid as "
                "the subbin target (new_theta_centers / new_theta_edges).")

    def _recompute_slab(self, new_tc, new_te, fine_pairs,
                        slab_width, slab_nr_block, components,
                        fine=None, verbose=False, full_set=frozenset()):
        """Overwrite the diagonal band of the interpolated fine components, for :meth:`subbin`.

        ``components`` are (qpp, qmm, qpm, rp0, rp1, rm0, rm1, rpm) on the fine grid. The band is
        evaluated with ``fine``, or with an idealised fine instance if None, and skipped for the
        components in ``full_set``, which :meth:`subbin` overwrites in full.
        """
        if slab_width is None:
            raise ValueError("recompute_slab=True requires `slab_width`.")

        # Per-block band widths
        widths = self._parse_slab_widths(slab_width)
        masks = {b: self._slab_band_mask(new_tc, w) for b, w in widths.items()}

        # Tile size of the banded quadruplet integral
        dln = float(np.mean(np.diff(np.log(new_tc))))
        m_max = max((int(np.ceil(np.log(w) / dln))
                     for w in widths.values() if w > 1.), default=1)
        n_v2 = len(self.varphi2_grid)
        n_v1 = len(self.varphi1_grid)
        n_xi = getattr(self, 'ntomo', 1)*(getattr(self, 'ntomo', 1)+1)

        if slab_nr_block:
            nr_block = int(slab_nr_block)
            v2_block_slab = self.v2_block
        else:
            nr_block, v2_block_slab = _auto_reduce_tiles(
                max(4, m_max), self.v2_block, n_v2, n_v1, n_xi)

        if fine is None:
            ## Auto-build an analytic fine instance ##
            if not (self.method_pairs == 'analytic'
                    and self.method_triplets == 'analytic'
                    and self.method_quadruplets == 'analytic'):
                raise NotImplementedError(
                    "recompute_slab=True without an explicit `slab_instance` "
                    "is only supported for analytic count tensors. For the "
                    "'discrete' method, build a fine-grid GGCovarianceNoTomo "
                    "with your measured triplet/quadruplet tensors and pass it "
                    "as subbin(..., slab_instance=fine).")
            # The analytic density lives on the fine xi_pm grid and on phi_centers
            fine = GGCovarianceNoTomo(**self._grid_kwargs(
                new_tc, new_te,
                q_theta_centers=new_tc, q_theta_edges=new_te,
                q_phi_centers=None, q_phi_edges=None,
                method_pairs='analytic', method_triplets='analytic',
                method_quadruplets='analytic',
                v2_block=v2_block_slab, nr_block=nr_block))
            # Fine pair counts, consistent with the interpolated off-diagonal and the fine D
            if fine_pairs is not None:
                fine.pairs = np.asarray(fine_pairs, dtype=float)
        else:
            ## Use the caller-supplied fine instance (any method) ##
            self._check_slab_instance_grid(fine, new_tc, new_te)
            # An explicit slab_nr_block is kept, otherwise the tiles are reduced from the
            # nr_block of fine, which keeps them within its memory budget
            if slab_nr_block is not None:
                fine.nr_block = nr_block
            else:
                if fine.nr_block is not None:
                    nrb_eff, v2_eff = _auto_reduce_tiles(
                        fine.nr_block, fine.v2_block, n_v2, n_v1, n_xi)
                else:
                    nrb_eff, v2_eff = nr_block, v2_block_slab
                fine.nr_block = nrb_eff
                fine.v2_block = v2_eff

        qpp_f, qmm_f, qpm_f, rp0_f, rp1_f, rm0_f, rm1_f, rpm_f = (
            np.array(c, dtype=float, copy=True) for c in components)

        # Mixed terms, per block
        mpp = masks['pp']
        if mpp.any() and 'qpp' not in full_set:
            qpp_f[mpp] = fine.getqpp(pair_mask=mpp)[mpp]
        mmm = masks['mm']
        if mmm.any() and 'qmm' not in full_set:
            qmm_f[mmm] = fine.getqmm(pair_mask=mmm)[mmm]
        mpm = masks['pm']
        if mpm.any() and 'qpm' not in full_set:
            qpm_f[mpm] = fine.getqpm(pair_mask=mpm)[mpm]

        # The bands are nested, so one pass over the widest band covers all five r-terms
        mask_max = masks[max(widths, key=widths.get)]
        if mask_max.any():
            rp0_b, rp1_b, rm0_b, rm1_b, rpm_b = fine.get_all_rterms(
                pair_mask=mask_max, verbose=verbose)
            if mpp.any():
                rp0_f[mpp] = rp0_b[mpp]
                rp1_f[mpp] = rp1_b[mpp]
            if mmm.any():
                rm0_f[mmm] = rm0_b[mmm]
                rm1_f[mmm] = rm1_b[mmm]
            if mpm.any():
                rpm_f[mpm] = rpm_b[mpm]

        return qpp_f, qmm_f, qpm_f, rp0_f, rp1_f, rm0_f, rm1_f, rpm_f

    ###############
    # I/O helpers #
    ###############

    def _load_pairs(self, fname):
        arr = np.load(fname)
        self._check_pairs_shape(arr)
        return arr

    def _load_triplets(self, fname):
        arr = np.load(fname)
        self._check_triplets_shape(arr)
        return arr

    def _load_quadruplets(self, fname):
        arr = np.load(fname)
        self._check_quadruplets_shape(arr)
        return arr

    ###################################
    # Multiplet counts from a catalog #
    ###################################

    @staticmethod
    def _check_loguniform(edges, name):
        """Raise if ``edges`` is not (within tolerance) log-uniform (geomspace)."""
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 3 or np.any(edges <= 0):
            raise ValueError(f"{name} must be positive and have >= 2 bins")
        dlog = np.diff(np.log(edges))
        if not np.allclose(dlog, dlog[0], rtol=1e-3):
            raise ValueError(
                f"{name} must be log-uniform (geomspace) when a `catalog` is "
                "supplied, since the orpheus estimators bin in log theta.")

    def _catalog_param(self, name, default=None, required=False):
        """Read a key from ``self.catalog`` with default / required handling."""
        val = self.catalog.get(name, None)
        if val is not None:
            return val
        if required:
            raise ValueError(f"catalog['{name}'] is required")
        return default

    def _nmaxs(self):
        """``catalog['nmaxs']`` as an int or an array, since the NPCF constructors take no lists."""
        nmaxs = self._catalog_param('nmaxs', required=True)
        return nmaxs if isinstance(nmaxs, int) else np.asarray(nmaxs, dtype=np.int32)

    def _catalog_arrays(self):
        """Return ``(ra, dec, weight)`` float arrays from ``self.catalog``."""
        ra = np.asarray(self._catalog_param('ra', required=True), dtype=float)
        dec = np.asarray(self._catalog_param('dec', required=True), dtype=float)
        w = self.catalog.get('weight', None)
        w = np.ones_like(ra) if w is None else np.asarray(w, dtype=float)
        return ra, dec, w

    def _ensure_catalog(self, kind):
        """Build, patch and cache the catalog of the measurements.

        ``kind='spin'`` gives a SpinTracerCatalog for GG and GGG, patched for the xi_pm range, and
        ``kind='scalar'`` a ScalarTracerCatalog for NNNN, patched for the quadruplet range. The
        tracers are unity, so the measured norms are weighted counts.
        """
        if kind in self._cat_cache:
            return self._cat_cache[kind]
        from .catalog import ScalarTracerCatalog, SpinTracerCatalog
        ra, dec, w = self._catalog_arrays()
        ones = np.ones_like(ra)
        npatches = int(self._catalog_param('npatches', required=True))
        if kind == 'scalar':
            cat = ScalarTracerCatalog(
                geometry='spherical', units_pos1='deg', units_pos2='deg',
                pos1=ra, pos2=dec, weight=w, tracer=ones)
            # 4PCF: one spoke up to q max_sep.
            patchextend_deg = float(self.q_theta_edges[-1]) / 60.
        elif kind == 'spin':
            cat = SpinTracerCatalog(
                spin=2, geometry='spherical', units_pos1='deg', units_pos2='deg',
                pos1=ra, pos2=dec, weight=w, tracer_1=ones, tracer_2=ones)
            # 2PCF/3PCF: triplet vertices span up to ~2x the 2PCF max_sep.
            patchextend_deg = 2. * float(self.theta_edges[-1]) / 60.
        else:
            raise ValueError(f"unknown catalog kind '{kind}'")
        cat.topatches(npatches=npatches, patchextend_deg=patchextend_deg,
                      verbose=self._catalog_param('verbosity', 1) > 0)
        self._cat_cache[kind] = cat
        return cat

    def _run_tag(self, stat):
        """Filename stem of a measurement, hashing the catalog and the binning parameters."""
        import hashlib
        ra, dec, w = self._catalog_arrays()
        h = hashlib.sha1()
        for a in (ra, dec, w):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        rmin_pixsize = self._catalog_param('rmin_pixsize', 40)
        tree_resos = list(np.ravel(self._catalog_param(
            'tree_resos', [0., 0.5, 1., 2., 4.])))
        # npatches and shuffle_pix enter only if they differ from their defaults, which keeps
        # older caches valid. nmaxs does not enter the pairs.
        npatches = int(self._catalog_param('npatches', required=True))
        np_tag = () if npatches == 1 else (npatches,)
        shuffle_pix = int(self._catalog_param('shuffle_pix', 1))
        sh_tag = () if shuffle_pix == 1 else (('shuffle_pix', shuffle_pix),)
        if stat == 'pairs':
            # 'ordered' tags the counts of ordered pairs, older caches hold twice as many
            meta = ('pairs', float(self.theta_edges[0]),
                    float(self.theta_edges[-1]), int(self.n_r), 'ordered') + np_tag + sh_tag
        elif stat == 'triplets':
            nmaxs = list(np.ravel(self._catalog_param('nmaxs', required=True)))
            meta = ('triplets', float(self.theta_edges[0]),
                    float(self.theta_edges[-1]), int(self.triplet_n_r),
                    int(self.n_phi), nmaxs) + np_tag + sh_tag
        elif stat == 'quadruplets':
            nmaxs = list(np.ravel(self._catalog_param('nmaxs', required=True)))
            meta = ('quadruplets', float(self.q_theta_edges[0]),
                    float(self.q_theta_edges[-1]), int(self.q_n_r),
                    int(self.q_n_phi), nmaxs, npatches,
                    self._catalog_param('method', 'DoubleTree')) + sh_tag
        else:
            raise ValueError(f"unknown stat '{stat}'")
        h.update(repr((meta, rmin_pixsize, tree_resos)).encode())
        return "%s_%s" % (stat, h.hexdigest()[:16])

    @staticmethod
    def _safe_remove(path):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def computeDiscrete(self, contribution):
        """Measure the count tensor of one multiplet from ``self.catalog`` with orpheus.

        ``contribution`` is ``'pairs'`` (N_p of shape (n_r,)), ``'triplets'`` (N_t of shape
        (triplet_n_r, triplet_n_r, n_phi)) or ``'quadruplets'`` (density q of shape (q_n_r, q_n_r,
        q_n_r, q_n_phi, q_n_phi)). The binning follows the covariance grids, and the results are
        cached in ``catalog['path_save']`` under a hash of the inputs.
        """
        if self.catalog is None:
            raise ValueError(
                f"computeDiscrete('{contribution}') requires a `catalog` dict "
                "(method_* is 'discrete' but neither a tensor nor a catalog "
                "was provided).")
        if 'path_save' not in self.catalog or self.catalog['path_save'] is None:
            raise ValueError("catalog['path_save'] is required")
        if contribution == 'pairs':
            return self._measure_pairs()
        if contribution == 'triplets':
            return self._measure_triplets()
        if contribution == 'quadruplets':
            return self._measure_quadruplets()
        raise ValueError(f"Unknown contribution `{contribution}`")

    def _measure_pairs(self):
        """Weighted pair counts on the xi_pm radial grid, shape (n_r,)."""
        from .npcf_second import GGCorrelation
        path_save = self.catalog['path_save']
        fpath = os.path.join(path_save, self._run_tag('pairs') + '.npz')
        if os.path.exists(fpath):
            dat = np.load(fpath)
            pairs = dat['pairs']
            if 'bin_centers_mean' in dat:
                self.theta_centers = dat['bin_centers_mean']
        else:
            gg = GGCorrelation(
                min_sep=float(self.theta_edges[0]),
                max_sep=float(self.theta_edges[-1]), nbinsr=self.n_r,
                rmin_pixsize=self._catalog_param('rmin_pixsize', 40),
                tree_resos=self._catalog_param('tree_resos', [0., 0.5, 1., 2., 4.]),
                shuffle_pix=self._catalog_param('shuffle_pix', 1),
                verbosity=self._catalog_param('verbosity', 1),
                nthreads=self.nthreads)
            gg.process(self._ensure_catalog('spin'))
            # N_p counts ordered pairs (both (i,j) and (j,i)), which is what
            # GGCorrelation's norm holds with or without do_dc.
            pairs = np.asarray(gg.norm[0], dtype=float)
            bc = np.asarray(gg.bin_centers_mean, dtype=float)
            empty = bc == 0.
            if empty.any():
                bc[empty] = self.theta_centers[empty]
            self.theta_centers = bc
            np.savez(fpath, pairs=pairs,
                     bin_centers_mean=self.theta_centers,
                     theta_edges=self.theta_edges,
                     area=self.area,
                     nbar_ref=getattr(self, 'nbar_ref', self.nbar))
        pairs = np.asarray(pairs, dtype=float)
        self._check_pairs_shape(pairs)
        if not self.store_multiplets:
            self._safe_remove(fpath)
        return pairs

    def _measure_triplets(self):
        """Weighted triplet counts N_t on the oversampled triplet grid."""
        from .npcf_third import GGGCorrelation
        path_save = self.catalog['path_save']
        fpath = os.path.join(path_save, self._run_tag('triplets') + '.npz')
        if os.path.exists(fpath):
            dat = np.load(fpath)
            triplets = dat['triplets']
            if 'bin_centers_mean' in dat:
                self.triplet_theta_centers = dat['bin_centers_mean']
        else:
            ggg = GGGCorrelation(
                n_cfs=4, min_sep=float(self.theta_edges[0]),
                max_sep=float(self.theta_edges[-1]), nbinsr=self.triplet_n_r,
                nmaxs=self._nmaxs(), nbinsphi=self.n_phi,
                rmin_pixsize=self._catalog_param('rmin_pixsize', 40),
                tree_resos=self._catalog_param('tree_resos', [0., 0.5, 1., 2., 4.]),
                shuffle_pix=self._catalog_param('shuffle_pix', 1),
                verbosity=self._catalog_param('verbosity', 1),
                nthreads=self.nthreads)
            ggg.process(self._ensure_catalog('spin'))
            # Only the norm is needed; 'X' skips the centroid rotation of the
            # (4, nz, nr, nr, nphi) shear array, which would be discarded.
            ggg.multipoles2npcf(projection='X')
            triplets = np.asarray(ggg.npcf_norm[0].real, dtype=float)
            bc = np.asarray(ggg.bin_centers_mean, dtype=float)
            empty = bc == 0.
            if empty.any():
                bc[empty] = self.triplet_theta_centers[empty]
            self.triplet_theta_centers = bc
            np.savez(fpath, triplets=triplets,
                     bin_centers_mean=self.triplet_theta_centers,
                     theta_edges=self.triplet_theta_edges,
                     phi_centers=self.phi_centers,
                     phi_edges=self.phi_edges,
                     area=self.area,
                     nbar_ref=getattr(self, 'nbar_ref', self.nbar))
        triplets = np.asarray(triplets, dtype=float)
        self._check_triplets_shape(triplets)
        if not self.store_multiplets:
            self._safe_remove(fpath)
        return triplets

    def _measure_quadruplets(self):
        """Quadruplet density on the quadruplet grid, combined over the patches."""
        path_save = self.catalog['path_save']
        tag = self._run_tag('quadruplets')
        fpath = os.path.join(path_save, tag + '.npz')
        if os.path.exists(fpath):
            dat = np.load(fpath)
            density = np.asarray(dat['quadruplets'], dtype=float)
            if 'bin_centers_mean' in dat:
                self.q_theta_centers = dat['bin_centers_mean']
        else:
            counts = self._measure_quadruplet_counts(tag)
            density = self._quad_density_from_counts(counts)
            np.savez(fpath, quadruplets=density,
                     bin_centers_mean=self.q_theta_centers,
                     q_theta_edges=self.q_theta_edges,
                     q_phi_centers=self.q_phi_centers,
                     q_phi_edges=self.q_phi_edges,
                     area=self.area,
                     nbar_ref=getattr(self, 'nbar_ref', self.nbar))
        self._check_quadruplets_shape(density)
        # The patch files are always removed
        self._cleanup_quadruplet_patchfiles(tag)
        if not self.store_multiplets:
            self._safe_remove(fpath)
        return density

    def _measure_quadruplet_counts(self, tag):
        """Real-space 4PCF counts on the quadruplet grid, summed over the patches.

        The patches run one after another, each with all ``nthreads`` threads. Their multipoles are
        cached, so that an interrupted run resumes, and summed before the conversion to real space.
        """
        from .npcf_fourth import NNNNCorrelation_NoTomo
        path_save = self.catalog['path_save']
        npatches = int(self._catalog_param('npatches', required=True))
        nmaxs = self._nmaxs()
        rmin_pixsize = self._catalog_param('rmin_pixsize', 40)
        tree_resos = self._catalog_param('tree_resos', [0., 0.5, 1., 2., 4.])
        method = self._catalog_param('method', 'DoubleTree')
        verbosity = self._catalog_param('verbosity', 1)
        memory_bound = self._catalog_param('memory_bound', 512.)
        shuffle_pix = self._catalog_param('shuffle_pix', 1)
        min_sep = float(self.q_theta_edges[0])
        max_sep = float(self.q_theta_edges[-1])
        cat = self._ensure_catalog('scalar') # spherical, already patched

        def _new_nnnn():
            return NNNNCorrelation_NoTomo(
                min_sep=min_sep, max_sep=max_sep, nbinsr=self.q_n_r,
                nbinsphi=self.q_n_phi, nmaxs=nmaxs, method=method,
                rmin_pixsize=rmin_pixsize, tree_resos=tree_resos,
                shuffle_pix=shuffle_pix, verbosity=verbosity, nthreads=self.nthreads)

        if npatches == 1:
            # Single patch
            nnnn = _new_nnnn()
            nnnn.process(cat.frompatchind(0), lowmem=True,
                         statistics=["4pcf_multipole"], memory_bound=memory_bound)
            bc = np.asarray(nnnn.bin_centers_mean, dtype=float)
            empty = bc == 0.
            if empty.any():
                bc[empty] = self.q_theta_centers[empty]
            self.q_theta_centers = bc
            nnnn.multipoles2npcf()
            return nnnn.npcf[:, :, :, 0].real.copy()

        ## Multi-patch: serial loop, each patch using all threads ##
        summed = None
        centers = None
        norm_centers = None
        for elp in range(npatches):
            pf = os.path.join(path_save, "%s_patch%i.npz" % (tag, elp))
            if os.path.exists(pf):
                dat = np.load(pf)
                mult, bc = dat['npcf_multipoles'], dat['bin_centers_mean']
            else:
                if verbosity > 0:
                    print("  4PCF patch %d/%d" % (elp + 1, npatches))
                nnnn = _new_nnnn()
                nnnn.process(cat.frompatchind(elp), lowmem=True,
                             statistics=["4pcf_multipole"],
                             memory_bound=memory_bound)
                mult, bc = nnnn.npcf_multipoles, nnnn.bin_centers_mean
                np.savez(pf, npcf_multipoles=mult, bin_centers_mean=bc)
            if summed is None:
                summed = np.zeros_like(mult)
                centers = np.zeros_like(bc)
                norm_centers = np.zeros_like(bc)
            summed += mult
            centers += bc
            norm_centers += (bc > 0).astype(float)

        ## Combine: sum the per-patch multipoles, average the bin centers ##
        nnnn = _new_nnnn()
        # NNNN normalises by its own counts, so the multipoles are their own norm
        # (as set in NNNNCorrelation_NoTomo.process).
        nnnn.npcf_multipoles = summed
        nnnn.npcf_multipoles_norm = summed
        nnnn.bin_centers_mean = np.divide(
            centers, norm_centers, out=np.zeros_like(centers),
            where=norm_centers > 0)
        empty = nnnn.bin_centers_mean == 0.
        if empty.any():
            nnnn.bin_centers_mean[empty] = self.q_theta_centers[empty]
        self.q_theta_centers = nnnn.bin_centers_mean.copy()
        nnnn.multipoles2npcf()
        return nnnn.npcf[:, :, :, 0].real.copy()

    def _quad_density_from_counts(self, counts):
        """Convert real-space 4PCF counts to the density q = N_q / V_q"""
        if self.area is None:
            raise ValueError("`area` is required to convert counts to density")
        q = np.array(counts, dtype=float)
        tdt = self.q_theta_centers * self.q_dtheta
        dp = self.q_dphi
        # V_q is separable, so divide axis by axis (like in set_quadruplets_from_counts)
        q /= 2. * np.pi * self.area
        q /= tdt[:, None, None, None, None]
        q /= tdt[None, :, None, None, None]
        q /= tdt[None, None, :, None, None]
        q /= dp[None, None, None, :, None]
        q /= dp[None, None, None, None, :]
        return q

    def _cleanup_quadruplet_patchfiles(self, tag):
        """Delete the per-footprint 4PCF patch files for ``tag``."""
        path_save = self.catalog['path_save']
        for f in glob.glob(os.path.join(path_save, "%s_patch*.npz" % tag)):
            self._safe_remove(f)

    ###################################################################
    # Covariance of integrated second-order statistics (S+02 Sect. 6) #
    ###################################################################

    @staticmethod
    def _getfilter(filter_type, gg, x):
        """Filter weights of a statistic, read off a helper GGCorrelation with identity inputs.

        The helper needs ``bin_centers_mean``, ``nzcombis`` = n_r, ``bin_centers`` and ``binsize``
        (see :meth:`cov_of_statistic`). For ``'Map2'``, ``x`` is the aperture radius and the E-mode
        weights ``(w_plus, w_minus)`` are returned, where the B-mode negates ``w_minus``. For
        ``'puremode'``, ``x`` is ignored and the (6n_r, 2n_r) matrix K mapping (xi_+, xi_-) to
        (xip_E, xip_B, xip_amb, xim_E, xim_B, xim_amb) is returned.
        """
        n = gg.nbinsr
        # Helper integrates with the measure binsize theta^2 instead of theta dtheta, which
        # the inputs absorb via corr
        theta = np.asarray(gg.bin_centers_mean, dtype=float)
        dtheta = theta * (np.exp(gg.binsize) - 1.) # exact for log-uniform bins
        corr = dtheta / (gg.binsize * theta) # 1 for log-uniform bins

        if filter_type == 'Map2':
            R = float(x)
            # xip = diag(corr) and xim = 0 give T_+(theta_i/R) dtheta_i theta_i / (2R^2)
            gg.xip = np.diag(corr).astype(complex)
            gg.xim = np.zeros((n, n), dtype=complex)
            res_p = gg.computeMap2([R]) # (4, n, 1)
            # xip = 0 and xim = diag(corr) give T_-(theta_i/R) dtheta_i theta_i / (2R^2)
            gg.xip = np.zeros((n, n), dtype=complex)
            gg.xim = np.diag(corr).astype(complex)
            res_m = gg.computeMap2([R])
            return res_p[0, :, 0], res_m[0, :, 0]

        elif filter_type == 'puremode':
            # Identity inputs give K block by block, e.g. xip = diag(corr) and xim = 0 yield
            # xip_E[iz, j] = M_pEpp[j, iz] corr[iz]
            C = np.diag(corr).astype(complex)
            Z = np.zeros((n, n), dtype=complex)
            gg.xip = C
            gg.xim = Z
            xip_p, xim_p = gg.computepuremode() # (3, n, n) each
            gg.xip = Z
            gg.xim = C
            xip_m, xim_m = gg.computepuremode()

            # (3, n, n) blocks, divided by corr and transposed
            def _extract(res):
                return (res / corr[np.newaxis, :, np.newaxis]).real.transpose(0, 2, 1)

            pp = _extract(xip_p) # (3, n, n): pp[k] = M_k_pp
            pm = _extract(xip_m) # (3, n, n): pm[k] = M_k_pm
            mp = _extract(xim_p) # (3, n, n): mp[k] = M_k_mp
            mm = _extract(xim_m) # (3, n, n): mm[k] = M_k_mm

            # Ordering: xip_E[0], xip_B[1], xip_amb[2], xim_E[0], xim_B[1], xim_amb[2]
            K = np.block([
                [pp[0], pm[0]], [pp[1], pm[1]], [pp[2], pm[2]],
                [mp[0], mm[0]], [mp[1], mm[1]], [mp[2], mm[2]], ])
            return K

        else:
            raise ValueError(
                f"_getfilter: unknown filter_type '{filter_type}'. "
                "For 'cosebis' call gg.computecosebi(Nmax) directly.")

    def cov_of_statistic(self, filter_type,
                         radii=None,
                         Nmax=None,
                         theta_min=None,
                         theta_max=None,
                         cosebi_dps=100,
                         cosebi_nsample=4096,
                         _return_weights=False):
        r"""Covariance of a statistic that is linear in xi_pm, S+02 Eqs. 41 and 42.

        .. math::

            \mathrm{Cov}(S_a, S_b) = w_a^T\, \mathrm{Cov}_{\xi_\pm}\, w_b

        for S = sum_i [w_+(theta_i) xi_+(theta_i) + w_-(theta_i) xi_-(theta_i)]. The weights are
        read off a helper GGCorrelation (see :meth:`_getfilter`), with the measure theta dtheta
        absorbed into its input, which is exact for log-uniform bins.

        Parameters
        ----------
        filter_type: {'Map2', 'cosebis', 'puremode'}
            The statistic.
        radii: numpy.ndarray, optional
            Aperture radii, required for ``'Map2'``.
        Nmax: int, optional
            Number of modes, required for ``'cosebis'``.
        theta_min, theta_max: float, optional
            Angular range of the statistic. Default to the outer bin edges.
        cosebi_dps: int, optional
            mpmath precision of the COSEBI root finding. Defaults to ``100``.
        cosebi_nsample: int, optional
            Quadrature points of T_- in ``GGCorrelation.computecosebi``. Defaults to ``4096``.
        _return_weights: bool, optional
            Also return the weights, as used by :meth:`GGCovariance.cov_of_statistic_tomo`.

        Returns
        -------
        dict
            ``'cov_EE'``, ``'cov_BB'`` and ``'cov_EB'`` for ``'Map2'`` and ``'cosebis'``, the
            latter in rad^4 like ``computecosebi``. For ``'puremode'`` the ``'cov'`` of (xip_E,
            xip_B, xip_amb, xim_E, xim_B, xim_amb) and its ``'labels'``. With
            ``_return_weights`` also ``(W_E, W_B)`` or ``(K, labels)``.
        """
        from .npcf_second import GGCorrelation

        if self.cov_xipm is None:
            self.get_cov_xipm()
        C = self.cov_xipm
        n = self.n_r
        tc = self.theta_centers
        tmin = float(self.theta_edges[0]) if theta_min is None else float(theta_min)
        tmax = float(self.theta_edges[-1]) if theta_max is None else float(theta_max)

        # Helper GGCorrelation for the filters (pure Python, so __init__ is skipped)
        gg = object.__new__(GGCorrelation)
        gg.binsize = np.log(tmax / tmin) / n
        gg.bin_edges = np.geomspace(tmin, tmax, n + 1)
        gg.min_sep = tmin
        gg.max_sep = tmax
        gg.nbinsr = n
        gg.bin_centers_mean = tc
        gg.bin_centers = np.tile(tc, (n, 1))
        gg.nzcombis = n # one basis vector per radial bin
        gg.xip = np.zeros((n, n), dtype=complex)
        gg.xim = np.zeros((n, n), dtype=complex)

        # Map2
        if filter_type == 'Map2':
            if radii is None:
                raise ValueError("'radii' is required for filter_type='Map2'")
            radii = np.asarray(radii, dtype=float).ravel()
            n_R = len(radii)
            W_E = np.zeros((n_R, 2*n))
            W_B = np.zeros((n_R, 2*n))
            for k, R in enumerate(radii):
                wp, wm = GGCovarianceNoTomo._getfilter('Map2', gg, R)
                W_E[k, :n] = wp
                W_E[k, n:] = wm
                W_B[k, :n] = wp
                W_B[k, n:] = -wm

        # COSEBIs (units: rad^2)
        elif filter_type == 'cosebis':
            if Nmax is None:
                raise ValueError("'Nmax' is required for filter_type='cosebis'")
            # Measure correction, see _getfilter
            dtheta = tc * (np.exp(gg.binsize) - 1.)
            corr = dtheta / (gg.binsize * tc) # 1 for log-uniform bins
            # T_+ weights from xip = diag(corr)
            gg.xip = np.diag(corr).astype(complex)
            gg.xim = np.zeros((n, n), dtype=complex)
            T_p = gg.computecosebi(Nmax, dps=cosebi_dps,
                                      Tminus_nsample=cosebi_nsample,
                                      units='rad2')
            # T_- weights from xim = diag(corr)
            gg.xip = np.zeros((n, n), dtype=complex)
            gg.xim = np.diag(corr).astype(complex)
            T_m = gg.computecosebi(Nmax, dps=cosebi_dps,
                                      Tminus_nsample=cosebi_nsample,
                                      units='rad2')
            # res[0] is E-mode contribution of xi_+ or xi_-, of shape (n_r, Nmax) etc
            W_plus = T_p[0].real.T
            W_minus = T_m[0].real.T
            W_E = np.concatenate([W_plus, W_minus], axis=1) # (Nmax, 2n_r)
            W_B = np.concatenate([W_plus, -W_minus], axis=1)

        # Pure mode correlation functions
        elif filter_type == 'puremode':
            K = GGCovarianceNoTomo._getfilter('puremode', gg, None) # (6n_r, 2n_r)
            cov_full = K @ C @ K.T
            label = (
                [f'xip_E[{i}]' for i in range(n)] + [f'xip_B[{i}]' for i in range(n)]
                + [f'xip_amb[{i}]' for i in range(n)] + [f'xim_E[{i}]' for i in range(n)]
                + [f'xim_B[{i}]' for i in range(n)] + [f'xim_amb[{i}]' for i in range(n)])
            out = {'cov': cov_full, 'labels': label}
            return (out, (K, label)) if _return_weights else out

        else:
            raise ValueError(f"filter_type must be 'Map2', 'cosebis' or 'puremode', got '{filter_type}'")

        out = {'cov_EE': W_E @ C @ W_E.T, 'cov_BB': W_B @ C @ W_B.T, 'cov_EB': W_E @ C @ W_B.T}

        return (out, (W_E, W_B)) if _return_weights else out


##########################
# Tomographic covariance #
##########################

def _infer_ntomo(n_bp):
    """Recover ntomo from number of unique bin pairs n_bp = ntomo*(ntomo+1)/2."""
    nt = int((-1. + np.sqrt(1. + 8. * n_bp)) / 2.)
    if nt * (nt + 1) // 2 != n_bp:
        raise ValueError(
            f"Number of xi spline entries ({n_bp}) is not ntomo*(ntomo+1)/2 "
            "for any integer ntomo")
    return nt


class GGCovariance(GGCovarianceNoTomo):
    """Tomographic covariance of the shear 2PCF between all pairs of tomographic bin pairs.

    The quadruplet density is shared by all bin combinations, only xi_pm change between them.

    Parameters
    ----------
    spline_xip, spline_xim: callable or list of (i, j, callable)
        xi_pm of each bin pair with i <= j, ntomo(ntomo+1)/2 entries. A single callable is the
        case ntomo = 1.
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
        self.n_bp = self.ntomo * (self.ntomo + 1) // 2
        self.bp_list = [(i, j)
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

    #####################
    # Xi spline parsing #
    #####################

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
        n_bp = max(len(xip), len(xim))
        if n_bp == 0:
            return xip, xim, 1
        ntomo = _infer_ntomo(n_bp)
        return xip, xim, ntomo

    def _xi(self, pm, i, j):
        """Look up xi_+(i,j) or xi_-(i,j) spline, handling index ordering."""
        key = (min(i, j), max(i, j))
        d = self._xip if pm == 'p' else self._xim
        return d[key]

    ##########################
    # Tomo combo enumeration #
    ##########################

    def _tomo_combos(self):
        """Non-redundant combos ((a,b),(c,d)) with (a,b) <= (c,d), for the symmetric r-terms."""
        combos = []
        for idx1, bp1 in enumerate(self.bp_list):
            for bp2 in self.bp_list[idx1:]:
                combos.append((bp1, bp2))
        return combos

    def _tomo_combos_all(self):
        """All ordered combos ((a,b),(c,d)), since Cov_pm is not symmetric under their exchange."""
        return [(bp1, bp2) for bp1 in self.bp_list for bp2 in self.bp_list]

    ###############################################
    # Callback mapping for the integration engine #
    ###############################################

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
        rterm_types = [
            ('rp0', 0, 'p', 'p'),
            ('rp1', 1, 'm', 'm'),
            ('rm0', 2, 'm', 'm'),
            ('rm1', 3, 'p', 'p'),
            ('rpm', 4, 'p', 'm'),
        ]

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
        for rname in ['rp0', 'rp1', 'rm0', 'rm1']:
            for combo in combos_sym:
                key = (rname, combo)
                out_key_to_idx[key] = len(out_keys)
                out_keys.append(key)
        for combo in combos_all:
            key = ('rpm', combo)
            out_key_to_idx[key] = len(out_keys)
            out_keys.append(key)

        # Build callback lists.
        cb_xi_a = []
        cb_xi_b = []
        cb_w = []
        cb_out = []
        cb_fac = []

        for rname, wid, xi_type_a, xi_type_b in rterm_types:
            combos = combos_all if rname == 'rpm' else combos_sym
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

    ##################
    # R-term getters #
    ##################

    def get_all_rterms_tomo(self, verbose=False, pair_mask=None,
                            nthreads=None):
        """Compute all tomographic r-term blocks in one pass.

        Returns ``{(rterm, ((a,b),(c,d))): (n_r, n_r)}``.
        """
        if nthreads is None:
            nthreads = self.nthreads
        self.rterms_tomo = self._quadruplet_density_terms_tomo_c(
            verbose=verbose, pair_mask=pair_mask, nthreads=nthreads)
        return self.rterms_tomo

    ###################
    # Shape noise (D) #
    ###################

    def getD_tomo(self):
        """Shape-noise term ``{(a, b): (n_r,)}`` of each bin pair, nonzero in auto blocks only."""
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        self.D_tomo = {}
        for bp in self.bp_list:
            a, b = bp
            se4 = self.sigma_eps[a]**2 * self.sigma_eps[b]**2
            if self.method_pairs in ('analytic', 'continuous'):
                Np = (2. * np.pi * self.area
                      * self.nbar[a] * self.nbar[b]
                      * self._w1**2
                      * _theta_dtheta(self.theta_edges)
                      * self._suppression(self.theta_centers))
            else:
                Np = np.asarray(self.pairs)
                if self.nbar_ref is not None:
                    Np = Np * (self.nbar[a] * self.nbar[b]
                               / self.nbar_ref**2)
            # Np are ordered pairs within one bin. A pair between two bins is
            # double counted on the same footing by taking it in both orders.
            if a != b:
                Np = 2. * Np
            self.D_tomo[bp] = se4 * ww_factor / Np
        return self.D_tomo

    ########################
    # Triplet terms (tomo) #
    ########################

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

    def _q_tomo_blocks(self, index, xi_pm):
        """All tomographic blocks of one mixed term, by its position in :attr:`QTERMS`.

        Auto combos of q_++ and q_-- are symmetrised, q_+- is not.
        """
        name, weight_type, _ = self.QTERMS[index]
        symmetrise = name in ('qpp', 'qmm')
        blocks = {}
        for combo in self._tomo_combos():
            block = self._triplet_tomo_block(weight_type, xi_pm, combo)
            if symmetrise and combo[0] == combo[1]:
                block = .5 * (block + block.T)
            blocks[combo] = block
        setattr(self, name + '_tomo', blocks)
        return blocks

    def getqpp_tomo(self):
        """Triplet q_++ for all tomo combos."""
        return self._q_tomo_blocks(0, 'p')

    def getqmm_tomo(self):
        """Triplet q_-- for all tomo combos."""
        return self._q_tomo_blocks(1, 'p')

    def getqpm_tomo(self):
        """Triplet q_+- for all tomo combos."""
        return self._q_tomo_blocks(2, 'm')

    #######################
    # Covariance assembly #
    #######################

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
        terms = {name: np.zeros((self.n_bp * n, self.n_bp * n))
                 for name in ('D', 'mix', 'sva')}
        for idx1, bp1 in enumerate(self.bp_list):
            if with_D:
                # Diagonal shape noise.
                terms['D'][idx1*n:(idx1+1)*n, idx1*n:(idx1+1)*n] += np.diag(self.D_tomo[bp1])
            for idx2, bp2 in enumerate(self.bp_list):
                combo = (bp1, bp2) if bp1 <= bp2 else (bp2, bp1)
                sl1 = slice(idx1 * n, (idx1 + 1) * n)
                sl2 = slice(idx2 * n, (idx2 + 1) * n)
                if combo in q_tomo:
                    terms['mix'][sl1, sl2] += q_tomo[combo]
                else:
                    terms['mix'][sl1, sl2] += q_tomo[(combo[1], combo[0])].T
                for rname in rnames:
                    terms['sva'][sl1, sl2] += self._rterm_block(rname, (bp1, bp2))
        self.cov_terms_tomo[block] = terms
        out = terms['D'] + terms['mix'] + terms['sva']
        setattr(self, 'cov_' + block + '_tomo', out)
        return out

    def get_cov_pp_tomo(self, verbose=False):
        """Assemble the full Cov_pp tomo matrix."""
        return self._get_cov_block_tomo(0, verbose)

    def get_cov_mm_tomo(self, verbose=False):
        """Assemble the full Cov_mm tomo matrix."""
        return self._get_cov_block_tomo(1, verbose)

    def get_cov_pm_tomo(self, verbose=False):
        """Assemble the full Cov_pm tomo matrix (no diagonal D term)."""
        return self._get_cov_block_tomo(2, verbose)

    def get_cov_xipm_tomo(self, verbose=False):
        """Assemble the full (2*n_bp*n_r, 2*n_bp*n_r) tomo covariance."""
        if self.cov_pp_tomo is None: self.get_cov_pp_tomo(verbose=verbose)
        if self.cov_mm_tomo is None: self.get_cov_mm_tomo(verbose=verbose)
        if self.cov_pm_tomo is None: self.get_cov_pm_tomo(verbose=verbose)
        m = self.n_bp * self.n_r
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
        and ``'pm'`` of :meth:`get_cov_xipm_tomo`, each term of shape (n_bp*n_r, n_bp*n_r), and the
        three terms of a block sum to it.
        """
        for index, (block, *_) in enumerate(self.COVBLOCKS):
            if block not in self.cov_terms_tomo:
                self._get_cov_block_tomo(index, verbose)
        return self.cov_terms_tomo

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
            catalog=self.catalog, store_multiplets=self.store_multiplets,
            quadruplets=self.quadruplets))

    def subbin_tomo(self, new_theta_centers, new_theta_edges,
                    new_pairs=None, recompute_slab=False, slab_width=None,
                    slab_instance=None, recompute_full=None, verbose=False):
        """Resample the tomographic covariance to a finer radial binning.

        The parameters are as in :meth:`GGCovarianceNoTomo.subbin`, with ``slab_instance`` a
        :class:`GGCovariance`. Returns a :class:`GGCovariance` on the fine grid with all components
        loaded.
        """
        new_tc = np.asarray(new_theta_centers, dtype=float)
        new_te = np.asarray(new_theta_edges, dtype=float)

        # Build the fine slab instance from the catalog if none is given
        if (slab_instance is None and (recompute_slab or recompute_full)
                and getattr(self, 'catalog', None) is not None):
            slab_instance = self._fine_instance_from_catalog(
                new_tc, new_te, verbose=verbose)
            if new_pairs is None:
                new_pairs = slab_instance.pairs

        # Ensure coarse tomo covariance is computed
        if not self.rterms_tomo:
            self.get_all_rterms_tomo(verbose=verbose)
        if self.D_tomo is None: self.getD_tomo()
        if self.qpp_tomo is None: self.getqpp_tomo()
        if self.qmm_tomo is None: self.getqmm_tomo()
        if self.qpm_tomo is None: self.getqpm_tomo()

        if recompute_full:
            full_set = (set(self._RECOMPUTE_FULL_COMPONENTS)
                        if recompute_full is True else set(recompute_full))
            if slab_instance is None:
                raise ValueError(
                    "recompute_full requires `slab_instance` (or a `catalog` "
                    "on this instance to auto-build one).")
        else:
            full_set = set()

        n_new = len(new_tc)

        tc = self.theta_centers
        x_c = np.log(tc)
        x_new = np.log(new_tc)

        _interp = self._build_block_interpolator(x_c, x_new, n_new)

        ## Interpolate D for all bin pairs ##
        D_fine = {}
        for bp in self.bp_list:
            D_scaled = self.D_tomo[bp] * _theta_dtheta(self.theta_edges)
            D_fine[bp] = np.exp(np.interp(
                x_new, x_c, np.log(D_scaled))) / _theta_dtheta(new_te)

        ## Interpolate q-terms ##
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

        ## Interpolate r-terms ##
        rterms_fine = {}
        symmetric_rterms = {'rp0', 'rp1', 'rm0', 'rm1'}
        for key, block in self.rterms_tomo.items():
            rname, combo = key
            sym = rname in symmetric_rterms
            rterms_fine[key] = _interp(block, sym=sym)

        ## Slab recomputation ##
        if recompute_slab:
            if slab_width is None:
                raise ValueError("recompute_slab=True requires `slab_width`.")
            if slab_instance is None:
                raise ValueError(
                    "recompute_slab=True requires `slab_instance` "
                    "(a GGCovariance on the fine grid).")
            self._check_slab_instance_grid(slab_instance, new_tc, new_te)

            widths = self._parse_slab_widths(slab_width)
            masks = {b: self._slab_band_mask(new_tc, w)
                     for b, w in widths.items()}
            mask_max = masks['pp'] | masks['mm'] | masks['pm']

            if verbose:
                print("  subbin_tomo: recomputing slab (r-terms) ...")
            if mask_max.any():
                slab_rterms = slab_instance.get_all_rterms_tomo(
                    verbose=verbose, pair_mask=mask_max)
                # Overwrite the slab band in interpolated r-terms
                rterm_mask = {'rp0': 'pp', 'rp1': 'pp',
                              'rm0': 'mm', 'rm1': 'mm', 'rpm': 'pm'}
                for key, block in slab_rterms.items():
                    rname, combo = key
                    m = masks[rterm_mask[rname]]
                    if key in rterms_fine:
                        rterms_fine[key] = np.where(m, block, rterms_fine[key])
                    else:
                        rterms_fine[key] = block

            # q-terms on the slab (cheap, per-combo)
            if 'qpp' not in full_set and masks['pp'].any():
                if verbose: print("  subbin_tomo: recomputing slab (qpp) ...")
                slab_instance.getqpp_tomo()
                for combo, block in slab_instance.qpp_tomo.items():
                    if combo in qpp_fine:
                        qpp_fine[combo] = np.where(masks['pp'], block, qpp_fine[combo])
            if 'qmm' not in full_set and masks['mm'].any():
                if verbose: print("  subbin_tomo: recomputing slab (qmm) ...")
                slab_instance.getqmm_tomo()
                for combo, block in slab_instance.qmm_tomo.items():
                    if combo in qmm_fine:
                        qmm_fine[combo] = np.where(masks['mm'], block, qmm_fine[combo])
            if 'qpm' not in full_set and masks['pm'].any():
                if verbose: print("  subbin_tomo: recomputing slab (qpm) ...")
                slab_instance.getqpm_tomo()
                for combo, block in slab_instance.qpm_tomo.items():
                    if combo in qpm_fine:
                        qpm_fine[combo] = np.where(masks['pm'], block, qpm_fine[combo])

        ## Full recomputation of cheap terms ##
        if full_set:
            self._check_slab_instance_grid(slab_instance, new_tc, new_te)
            if 'D' in full_set:
                slab_instance.getD_tomo()
                D_fine = slab_instance.D_tomo
            if 'qpp' in full_set:
                slab_instance.getqpp_tomo()
                qpp_fine = slab_instance.qpp_tomo
            if 'qmm' in full_set:
                slab_instance.getqmm_tomo()
                qmm_fine = slab_instance.qmm_tomo
            if 'qpm' in full_set:
                slab_instance.getqpm_tomo()
                qpm_fine = slab_instance.qpm_tomo

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
            n_bp=self.n_bp, bp_list=self.bp_list, nbar_ref=self.nbar_ref,
            D_tomo=D_fine, qpp_tomo=qpp_fine, qmm_tomo=qmm_fine,
            qpm_tomo=qpm_fine, rterms_tomo=rterms_fine)

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
            (n_stat*n_bp, n_stat*n_bp).
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
        nbp = self.n_bp

        # Lightweight non-tomo instance to compute filter weights once
        helper = object.__new__(GGCovarianceNoTomo)
        helper.theta_centers = self.theta_centers
        helper.theta_edges = self.theta_edges
        helper.n_r = n
        helper.cov_xipm = np.eye(2*n)
        _, weights = helper.cov_of_statistic(
            filter_type, _return_weights=True, **kwargs)

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
            out_cov = np.zeros((ns * nbp, ns * nbp))
            for i1 in range(nbp):
                for i2 in range(nbp):
                    block = _extract_block(i1, i2)
                    out_cov[i1*ns:(i1+1)*ns, i2*ns:(i2+1)*ns] = K @ block @ K.T
            total = {'cov': out_cov, 'labels': lbl}
            Wpp, Wmm = K[:, :n], K[:, n:]
        else:
            # Map2 / cosebis: weights = (W_E, W_B)
            W_E, W_B = weights
            ns = W_E.shape[0]
            is_puremode = False
            total = {k: np.zeros((ns * nbp, ns * nbp))
                     for k in ('cov_EE', 'cov_BB', 'cov_EB')}
            for i1 in range(nbp):
                for i2 in range(nbp):
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
        sz = ns * nbp

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

        comp_names = ['D', 'qpp', 'qmm', 'qpm',
                      'rp0', 'rp1', 'rm0', 'rm1', 'rpm']
        comps = {name: _zeros() for name in comp_names}

        for i1 in range(nbp):
            bp1 = self.bp_list[i1]
            for i2 in range(nbp):
                bp2 = self.bp_list[i2]
                sl1 = slice(i1 * ns, (i1 + 1) * ns)
                sl2 = slice(i2 * ns, (i2 + 1) * ns)

                if i1 == i2:
                    d = np.diag(self.D_tomo[bp1])
                    _accum_pp(comps['D'], sl1, sl2, d)
                    _accum_mm(comps['D'], sl1, sl2, d)

                _accum_pp(comps['qpp'], sl1, sl2,
                          _qlookup(self.qpp_tomo, bp1, bp2))
                _accum_mm(comps['qmm'], sl1, sl2,
                          _qlookup(self.qmm_tomo, bp1, bp2))
                qpm_fwd = _qlookup(self.qpm_tomo, bp1, bp2)
                qpm_rev = _qlookup(self.qpm_tomo, bp2, bp1)
                _accum_pm(comps['qpm'], sl1, sl2, qpm_fwd, qpm_rev.T)

                _accum_pp(comps['rp0'], sl1, sl2,
                          self._rterm_block('rp0', (bp1, bp2)))
                _accum_pp(comps['rp1'], sl1, sl2,
                          self._rterm_block('rp1', (bp1, bp2)))
                _accum_mm(comps['rm0'], sl1, sl2,
                          self._rterm_block('rm0', (bp1, bp2)))
                _accum_mm(comps['rm1'], sl1, sl2,
                          self._rterm_block('rm1', (bp1, bp2)))
                rpm_fwd = self._rterm_block('rpm', (bp1, bp2))
                rpm_rev = self._rterm_block('rpm', (bp2, bp1))
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

    def _build_xi_tables(self, xi_keys):
        """Tabulate xi of the (pm, i, j) keys, as :meth:`_xi_tables_from_splines`."""
        return self._xi_tables_from_splines(
            [self._xi(pm, i, j) for pm, i, j in xi_keys],
            lo=self._xi_query_lo(), hi=self._xi_query_hi())

    def _quadruplet_density_terms_tomo_c(self, verbose=False, pair_mask=None,
                                         nthreads=None):
        """Tomographic r-terms via the C kernel with ``nthreads`` OpenMP threads."""
        if self.quadruplets is None:
            raise RuntimeError("`quadruplets` not initialised.")
        if self.area is None:
            raise ValueError("`area` is required for the quadruplet integral.")

        cbm = self._build_tomo_callback_map()
        out_keys = cbm['out_keys']
        xi_tables = self._build_xi_tables(cbm['xi_table_keys'])

        # Per-output-block mask: True if this block's r-term is symmetric.
        symmetric_rterms = {'rp0', 'rp1', 'rm0', 'rm1'}
        symmetric_mask = [key[0] in symmetric_rterms and key[1][0] == key[1][1]
                          for key in out_keys]

        accum = self._run_postq_engine(
            cbm, xi_tables, symmetric_mask,
            verbose=verbose, pair_mask=pair_mask, nthreads=nthreads)

        return {key: (.5 * (block + block.T) if symmetric_mask[oi] else block)
                for oi, (key, block) in enumerate(zip(out_keys, accum))}

    ########################################
    # Continuous tomo equations (C/OpenMP) #
    ########################################

    def get_continuous_tomo_c(self, verbose=False, nr_target=1024, nphi_target=1024, dlnt_target=.05):
        """Compute D, the mixed and the r-terms of all tomographic blocks in the continuum limit.

        The mixed terms are averaged over nr_target fine bins and the r-terms over sub-bins no wider
        than dlnt_target in ln theta. Assembles and returns ``cov_xipm_tomo``.
        """
        lib = self._clib

        def _ptr(a): return np.ascontiguousarray(a, dtype=np.float64)
        def _iptr(a): return np.ascontiguousarray(a, dtype=np.int32)

        n = self.n_r
        combos_sym = self._tomo_combos()
        combos_all = self._tomo_combos_all()

        # Build xi tables
        xi_keys = sorted({(pm, i, j)
                          for pm in ('p', 'm')
                          for i in range(self.ntomo)
                          for j in range(i, self.ntomo)})
        xi_key_to_idx = {k: idx for idx, k in enumerate(xi_keys)}
        # Add reverse lookup for (j, i)
        for pm, i, j in list(xi_key_to_idx.keys()):
            xi_key_to_idx[(pm, j, i)] = xi_key_to_idx[(pm, i, j)]

        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = \
            self._build_xi_tables(xi_keys)
        n_xi = len(xi_keys)

        ## D term (trivial, no C needed) ##
        self.getD_tomo()

        ## Q terms via C ##
        if verbose: print("  continuous q-terms (C/OpenMP) ...")
        alpha = max(1, int(nr_target // n))
        n_fine = alpha * n
        _edges = np.geomspace(self.theta_edges[0], self.theta_edges[-1], n_fine + 1)
        fine_centers = np.ascontiguousarray(.5 * (_edges[1:] + _edges[:-1]))
        fine_dtheta = np.diff(_edges)
        fine_weights = np.ascontiguousarray(fine_centers * fine_dtheta)
        phi_q = np.ascontiguousarray(
            np.linspace(0., np.pi, nphi_target, endpoint=False))
        dphi_q = np.ascontiguousarray(_trapz_weights(phi_q))
        # Unused by continuous_q_single
        Np_ideal = np.ones(n, dtype=np.float64)

        # The kernel output depends on (w_type, xi_idx) only, so each distinct pair is integrated
        # once and the factors of the combos are applied afterwards
        q_calls = {} # (w_type, xi_idx) -> kernel slot
        q_scatter = [] # (out_idx, slot, factor)
        q_out_keys = []
        for qname, w_type, xi_pm in [('qpp', 0, 'p'), ('qmm', 1, 'p'), ('qpm', 2, 'm')]:
            for combo in combos_sym:
                (a, b), (c, d) = combo
                q_out_keys.append((qname, combo))
                out_idx = len(q_out_keys) - 1
                for shared_a, shared_b, xi_i, xi_j in [
                    (a, c, b, d), (a, d, b, c), (b, c, a, d), (b, d, a, c)]:
                    if shared_a == shared_b:
                        call = (w_type, xi_key_to_idx[(xi_pm, xi_i, xi_j)])
                        slot = q_calls.setdefault(call, len(q_calls))
                        q_scatter.append((
                            out_idx, slot,
                            .25 * self.sigma_eps[shared_a]**2
                            / self.nbar[shared_a]))

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
                _ptr(xi_x), xi_n, xi_logx0, xi_inv_dlogx,
                _ptr(xi_y_tables), n_xi,
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
            sym = qname in ('qpp', 'qmm')
            if sym and combo[0] == combo[1]:
                block = .5 * (block + block.T)
            getattr(self, qname + '_tomo')[combo] = block

        ## R terms via C ##
        if verbose: print("  continuous r-terms (C/OpenMP) ...")
        v = np.asarray(self.varphi1_grid)
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
        rterm_names = ['rp0', 'rp1', 'rm0', 'rm1', 'rpm']
        all_combos = combos_all
        r_out_keys = [(rname, combo)
                      for combo in all_combos for rname in rterm_names]

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
                work_out_idx.append(ic * 5)

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
        symmetric = {'rp0', 'rp1', 'rm0', 'rm1'}
        self.rterms_tomo = {}
        for oi, (rname, combo) in enumerate(r_out_keys):
            if rname in symmetric and combo[0] > combo[1]:
                continue
            block = r_results[oi]
            if rname in symmetric and combo[0] == combo[1]:
                block = .5 * (block + block.T)
            self.rterms_tomo[(rname, combo)] = block

        ## Assemble ##
        if verbose: print("  assembling cov_xipm_tomo ...")
        self.get_cov_xipm_tomo()
        return self.cov_xipm_tomo