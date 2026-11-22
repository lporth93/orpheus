import ctypes
import hashlib
import os
import time
import warnings
import weakref

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.interpolate import interp1d

from .utils import _load_clib, check_clib_error

__all__ = ["GGCovarianceNoTomo", "GGCovariance"]

# Count tensors loaded from files, shared by all instances that hold them, by file and kind
_LOADED_MULTIPLETS = weakref.WeakValueDictionary()
# Integrals of the mixed terms, which do not depend on sigma_eps, shared by all instances that hold
# them, by the inputs of their kernel
_MIXED_INTEGRALS = weakref.WeakValueDictionary()


####################
# HELPER FUNCTIONS #
####################

## Integration related ##
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

def _fourier_map(nmax):
    """Coefficients U of e^(i n phi) = sum_a U[n+nmax, a] b_a(phi) in the real Fourier basis
    b = (1, cos(phi), sin(phi), ..., cos(nmax phi), sin(nmax phi)), shape (2 nmax+1, 2 nmax+1)."""
    U = np.zeros((2*nmax+1, 2*nmax+1), dtype=np.complex128)
    U[nmax, 0] = 1.
    for n in range(1, nmax+1):
        U[nmax+n, 2*n-1] = U[nmax-n, 2*n-1] = 1.
        U[nmax+n, 2*n], U[nmax-n, 2*n] = 1j, -1j
    return U

def _fourier_basis(ang, n_modes):
    """Real Fourier basis 1, cos(a), sin(a), ..., cos(n a), sin(n a) of the angles a, shape
    (len(ang), n_modes)."""
    k = np.arange(1, (n_modes+1)//2)
    out = np.ones((len(ang), n_modes))
    out[:, 1::2], out[:, 2::2] = np.cos(np.outer(ang, k)), np.sin(np.outer(ang, k))
    return out

def _mirror_symmetric(ang, tol=1e-9):
    """Whether a reflection, ang -> -ang modulo 2 pi, maps the angles ang onto themselves."""
    def wrapped(a): return np.sort(np.mod(a + tol, 2.*np.pi) - tol)
    ang = np.asarray(ang, dtype=float)
    return np.allclose(wrapped(ang), wrapped(-ang), rtol=0., atol=tol)

def _periodic_weights(x, period=2. * np.pi):
    """Uniform weights for periodic function sampled on a full-period grid."""
    x = np.asarray(x, dtype=float)
    return np.full_like(x, period / len(x))

## Misc ##
def _bin_average_blocks(blocks, weights):
    """Average blocks (..., n*ns, n*ns) over the ns sub-bins of each of the n bins, weights (n, ns)."""
    n, ns = weights.shape
    b = blocks.reshape(blocks.shape[:-2] + (n, ns, n, ns))
    M = weights.sum(1)
    return np.einsum('...aibj,ai,bj->...ab', b, weights, weights) / np.outer(M, M)


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
    q_theta_centers, q_theta_edges: numpy.ndarray, optional
        Radial bins of the quadruplet density. For a density measured on ``catalog``, default to
        log bins of width ``q_binsize`` from the first edge of ``theta_edges`` to the largest
        separation of ``catalog``. Below its first bin the sample-variance integral holds the
        density at that bin and above its last bin sets it to zero.
    q_binsize: float, optional
        Log bin width of density measured on ``catalog``. Defaults to ``0.1``.
    q_count_min: float, optional
        Smallest number of quadruplets per angular resolution cell, (2pi/(2*q_nmaxs+1))^2, using
        an idealised density. Smaller bins are subsequently merged. ``0`` keeps all bins. Defaults
        to ``1000``. Set to reduce impact of shot noise.
    q_richardson: bool, optional
        Extrapolate a measured density to vanishing radial bin width, (4 q - q_2)/3 with q_2 the
        density on pairs of merged radial bins, which needs an even number of radial bins.
        Defaults to ``False``.
    q_interpolation: str, optional
        Interpolation of a density tensor between its radial bins, ``'linear'`` or ``'cubic'``. A
        measured density is interpolated between the values whose interpolant has its bin means.
        Defaults to ``'cubic'``.
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
        :meth:`suppression_spline_from_pairs`. Defaults to ``None``, i.e. s = 1, or for
        ``'frompairs'`` with a ``catalog`` the suppression of its pairs on the radial bins of the
        quadruplet density.
    pairs, triplets, quadruplets: numpy.ndarray or str, optional
        Count tensors of the ``'discrete'`` methods, or their ``.npy`` files or the ``.npz`` files
        of ``store_multiplets``, which also set the bin centers of the measurement. Measured from
        ``catalog`` if not given. The triplets and the density hold the real Fourier coefficients
        of their angles, whose number sets ``t_nmaxs`` and ``q_nmaxs``. A density tensor that is 
        passed is interpolated between its values as they are.
    catalog: dict, optional
        Positions ``ra`` and ``dec`` in deg, and optionally ``weight``, of the catalog on which the
        ``'discrete'`` multiplets are measured.
    path_save: str, optional
        Directory of the measured multiplets, required with ``catalog``.
    store_multiplets: bool, optional
        Keep the measured count tensors on disk. Defaults to ``False``.
    nmaxs: int, optional
        Largest multipole of the triplet and quadruplet measurements. Defaults to ``30``.
    t_nmaxs, q_nmaxs: int, optional
        Largest multipole of the triplets and of the quadruplets alone, which are kept as the real
        Fourier coefficients 1, cos(n phi), sin(n phi) of their angles up to it. Default to
        ``nmaxs``.
    geometry: str, optional
        ``'spherical'`` for ``ra`` and ``dec`` on the sky, measured on the full sphere, or
        ``'flat2d'`` for Cartesian coordinates in deg. Defaults to ``'spherical'``.
    rmin_pixsize, tree_resos, tree_method, shuffle_pix, memory_bound, verbosity: optional
        Settings of the orpheus correlators, where ``tree_method`` applies to the quadruplets.
        Default to ``40``, ``[0, 0.5, 1, 2, 4]``, ``'DoubleTree'``, ``1``, ``512`` and ``1``.
    t_rmin_pixsize, t_tree_resos, q_rmin_pixsize, q_tree_resos: optional
        ``rmin_pixsize`` and ``tree_resos`` of the triplet and the quadruplet measurement, which may be
        coarser than those of the pairs. Default to ``rmin_pixsize`` and ``tree_resos``, except
        ``q_tree_resos``, which defaults to ``[0, 0.5, 1, 2, 4, 8, 16]``.
    phi_grid, varphi1_grid, varphi2_grid: numpy.ndarray, optional
        Integration grids of the quadruplet term. Default to 128 log-spaced radii from
        ``phi_grid_min`` to the last edge of ``q_theta_edges`` for a density tensor, otherwise to
        :meth:`Catalog.max_separation` of ``catalog`` or, without one, to 1e4, and 64 uniform
        angles. The angle grids must be symmetric under varphi -> -varphi modulo 2pi, e.g.
        uniform from 0 or from half a step.
    phi_grid_min: float, optional
        Smallest radius in arcmin of the default ``phi_grid``. Defaults to ``0.05``.
    phi1_subgrid_frac, phi2_subgrid_frac: numpy.ndarray, optional
        Offsets of radial subgrid nodes from the bin centers, in units of the bin width. Default
        to ``None``, i.e. the bin centers.
    nthreads: int, optional
        Number of OpenMP threads. Defaults to ``1``.

    Notes
    -----
    * Separations are in arcmin and angles in radians. The count tensors have the shapes (n_r,)
      for the pairs, (n_r*triplet_oversample, n_r*triplet_oversample, n_tm) for the triplets per
      unit angle and (n_qr, n_qr, n_qr, n_qm, n_qm) for the quadruplet density where n_tm/n_qm are
      related to the correspondin largest multiple as 2*nmax+1.
    """

    def __init__(
        self,
        theta_centers, theta_edges,
        q_theta_centers=None, q_theta_edges=None,
        q_binsize=0.1, q_count_min=1000.,
        q_richardson=False, q_interpolation='cubic', triplet_oversample=1,
        method_pairs='analytic', method_triplets='analytic', method_quadruplets='analytic',
        spline_xip=None, spline_xim=None,
        sigma_eps=0.27, nbar=None, area=None, weights_mean=1., weights_std=0., spl_suppression_ww=None,
        pairs=None, triplets=None, quadruplets=None,
        catalog=None, path_save=None, store_multiplets=False, nmaxs=30, t_nmaxs=None,
        q_nmaxs=None, geometry='spherical',
        rmin_pixsize=40, tree_resos=(0., 0.5, 1., 2., 4.), t_rmin_pixsize=None, t_tree_resos=None,
        q_rmin_pixsize=None, q_tree_resos=(0., 0.5, 1., 2., 4., 8., 16.),
        tree_method='DoubleTree', shuffle_pix=1, memory_bound=512., verbosity=1,
        phi_grid=None, phi_grid_min=0.05, varphi1_grid=None, varphi2_grid=None,
        phi1_subgrid_frac=None, phi2_subgrid_frac=None,
        nthreads=1):

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

        ## Threads
        self.nthreads = int(nthreads)
        if self.nthreads < 1:
            raise ValueError("nthreads must be >= 1")

        ## Catalog and orpheus setup of the discrete multiplet measurements
        self.catalog = catalog
        self.path_save = path_save
        self.store_multiplets = bool(store_multiplets)
        self.nmaxs = nmaxs
        self.t_nmaxs = int(nmaxs if t_nmaxs is None else t_nmaxs)
        self.q_nmaxs = int(nmaxs if q_nmaxs is None else q_nmaxs)
        # Real Fourier coefficients per angle of triplets and quadruplets
        self.t_n_modes, self.q_n_modes = 2*self.t_nmaxs+1, 2*self.q_nmaxs+1
        self.geometry = geometry
        self.rmin_pixsize = rmin_pixsize
        self.tree_resos = np.asarray(tree_resos, dtype=float)
        self.t_rmin_pixsize = rmin_pixsize if t_rmin_pixsize is None else t_rmin_pixsize
        self.t_tree_resos = self.tree_resos if t_tree_resos is None else np.asarray(t_tree_resos, dtype=float)
        self.q_rmin_pixsize = rmin_pixsize if q_rmin_pixsize is None else q_rmin_pixsize
        self.q_tree_resos = self.tree_resos if q_tree_resos is None else np.asarray(q_tree_resos, dtype=float)
        self.tree_method = tree_method
        self.shuffle_pix = shuffle_pix
        self.memory_bound = memory_bound
        self.verbosity = verbosity
        # Counter and start time of the printed steps
        self._steps = [0, time.perf_counter()]
        self.orpheus_instances = {}
        self.pairs_wsq = None
        # Triplet reductions of the mixed terms by weight and xi, with the triplets they are of
        self._triplet_reduced = (None, {})
        # Mixed-term integrals that this instance holds in _MIXED_INTEGRALS
        self._mixed_integrals = {}
        # Init catalogs already here as those set the distance bounds for the quadruplets
        if self.catalog is not None:
            self.set_orpheus_instances([])

        ## Quadruplet radial grid
        # Default of a density measured on the catalog, or of the pair suppression of 'frompairs':
        # log bins of q_binsize from the first xi_pm edge to the largest separation of the catalog
        if (method_quadruplets in ('discrete', 'frompairs') and quadruplets is None
                and catalog is not None):
            if q_theta_centers is None:
                q_lo = self.theta_edges[0]
                sep_max = (self.orpheus_instances['catalog'].max_separation()
                           * (60. if geometry == 'spherical' else 1.))
                n_q = int(np.ceil(np.log(sep_max/q_lo)/q_binsize))
                # Even, as the extrapolation in the radial bin width merges pairs of bins, such that
                # one cached density serves with and without it
                n_q += n_q % 2
                q_theta_edges = q_lo*np.exp(q_binsize*np.arange(n_q + 1))
                q_theta_centers = np.sqrt(q_theta_edges[1:]*q_theta_edges[:-1])
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
        self.q_richardson = bool(q_richardson)
        self.q_count_min = float(q_count_min)
        # The extrapolation takes the bin means of a measured density
        measured = quadruplets is None or (isinstance(quadruplets, str)
                                           and quadruplets.endswith('.npz'))
        if self.q_richardson and (method_quadruplets != 'discrete' or not measured
                                  or self.q_n_r % 2 or self.q_n_r < 4):
            raise ValueError("q_richardson needs a density measured on `catalog` or read from its "
                             ".npz file, on an even number of at least four radial bins")
        if q_interpolation not in ('linear', 'cubic'):
            raise ValueError("q_interpolation is 'linear' or 'cubic'")
        self.q_interpolation = q_interpolation

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
        q_lo, q_hi = self.q_theta_edges[0], self.q_theta_edges[-1]
        if method_quadruplets == 'discrete' and (q_lo > self.theta_edges[0]*(1.+1e-12)
                                                 or q_hi < self.phi_grid[-1]*(1.-1e-12)):
            warnings.warn(
                "The quadruplet density spans %.4g-%.4g arcmin, the xi_pm bins start at %.4g and "
                "phi_grid ends at %.4g arcmin: below its first bin the integral holds the density "
                "at that bin, above its last bin sets it to zero"
                % (q_lo, q_hi, self.theta_edges[0], self.phi_grid[-1]), RuntimeWarning,
                stacklevel=2)
        self.varphi1_grid = (np.linspace(0., 2.*np.pi, 64, endpoint=False) if varphi1_grid is None else np.asarray(varphi1_grid))
        self.varphi2_grid = (np.linspace(0., 2.*np.pi, 64, endpoint=False) if varphi2_grid is None else np.asarray(varphi2_grid))
        # The quadratures of the sample-variance term drop the parts of the integrand that are odd
        # under a reflection of both pairs, which cancel only on grids that it maps onto themselves
        if not (_mirror_symmetric(self.varphi1_grid) and _mirror_symmetric(self.varphi2_grid)):
            raise ValueError("varphi1_grid and varphi2_grid must be symmetric under varphi -> "
                             "-varphi, e.g. uniform from 0 or from half a step")
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

        ## Pair-count suppression of the catalog for the frompairs density
        if method_quadruplets == 'frompairs' and spl_suppression_ww is None and catalog is not None:
            self.spl_suppression_ww = self._catalog_suppression()

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
        self._clib.quadruplet_sva_integrate.restype = None
        self._clib.quadruplet_sva_integrate.argtypes = [
            p_f64, p_f64, ci, ci, p_f64, p_f64, ci, p_f64, p_f64, ci,
            p_f64, p_f64, ci, p_f64, p_f64, ci,
            p_f64, p_f64, ci, ci, ci, cd, p_f64, p_f64, p_f64,
            p_f64, ci, cd, cd, p_f64, ci, ci, p_i32, p_i32,
            p_i32, p_i32, p_f64, ci, p_i32, p_f64, ci, ci]
        self._clib.continuous_q_tomo.restype = None
        self._clib.continuous_q_tomo.argtypes = [
            ci, ci, p_f64, p_f64, p_f64, p_f64, ci, ci,
            p_f64, ci, cd, cd, p_f64,
            cd, cd, cd, ci, p_i32, p_i32, p_f64, ci, ci]
        self._clib.continuous_rterms_tomo.restype = None
        self._clib.continuous_rterms_tomo.argtypes = [
            ci, p_f64, p_f64, p_f64, ci, p_f64, p_f64, ci, cd,
            p_f64, ci, cd, cd, p_f64, ci, ci, p_i32, p_i32, p_i32, p_i32, p_i32,
            p_f64, ci, ci]
        self._clib.triplet_term_reduce.restype = None
        self._clib.triplet_term_reduce.argtypes = [
            p_f64, ci, ci, p_f64, ci, ci, ci,
            p_f64, ci, cd, cd, p_f64,
            p_f64, ci, ci]
        self._clib.quadruplets_richardson.restype = None
        self._clib.quadruplets_richardson.argtypes = [p_f64, ci, ci, p_f64, p_f64, cd, ci, ci]
        self._clib.quadruplets_merge_sparse.restype = None
        self._clib.quadruplets_merge_sparse.argtypes = [p_f64, ci, ci, p_f64, cd, cd, cd, ci]
        self._clib.quadruplets_to_knots.restype = None
        self._clib.quadruplets_to_knots.argtypes = [p_f64, ci, ci, cd, ci]

    def _step(self, text):
        """Print the next step of the computation and the time the previous one took."""
        if self.verbosity > 0:
            now = time.perf_counter()
            self._steps[0] += 1
            took = "" if self._steps[0] == 1 else " (previous step %.1f min)"%((now-self._steps[1])/60.)
            print("Step %d: %s%s"%(self._steps[0], text, took), flush=True)
            self._steps[1] = now

    def _init_multiplet(self, kind, value, method):
        """Define construction of count tensor of one multiplet."""
        # Discrete --> Load, measure or validate
        if method == 'discrete':
            if value is None:
                return self.computeDiscrete(kind)
            if kind == 'quadruplets' and callable(value):
                return value
            from_npz = isinstance(value, str) and value.endswith('.npz')
            if from_npz:
                value = self._load_multiplet(kind, value)
            value = np.asarray(np.load(value) if isinstance(value, str) else value, dtype=float)
            self._adopt_modes(kind, value)
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

    QTERMS = (('qpp', 0, 'p'), ('qmm', 1, 'p'), ('qpm', 2, 'm')) # (name, weight_type, xi_pm)
    QTERMS_SYM = ('qpp', 'qmm') # Symmetric under theta1 <-> theta2
    # Sub-nodes per radial triplet bin and angular nodes over 2 pi of the mixed terms' w xi
    TRIPLET_SUBNODES = (3, 4096)
    RTERMS = ('rp0', 'rp1', 'rm0', 'rm1', 'rpm')
    RTERMS_SYM = ('rp0', 'rp1', 'rm0', 'rm1') # Symmetric under theta1 <-> theta2
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
        # The filters integrate using binsize*theta^2.
        # Absorb the measure correction into the input correlation.
        corr = np.diff(self.theta_edges) / (gg.binsize*self.theta_centers)
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

    #######################
    # Covariance of xi_pm #
    #######################

    def get_cov_xipm(self, verbose=False, nthreads=None):
        """Assemble the full (2n_r, 2n_r) covariance of (xi_+, xi_-)."""
        self._ensure_all_rterms(verbose=verbose, nthreads=nthreads)
        if self.cov_pp is None: self.get_cov_pp(verbose=verbose, nthreads=nthreads)
        if self.cov_mm is None: self.get_cov_mm(verbose=verbose, nthreads=nthreads)
        if self.cov_pm is None: self.get_cov_pm(verbose=verbose, nthreads=nthreads)
        self.cov_xipm = np.block([[self.cov_pp, self.cov_pm], [self.cov_pm.T, self.cov_mm]])
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

    def get_all_rterms(self, verbose=False, nthreads=None):
        """Compute rp0, rp1, rm0, rm1 and rpm in a single pass of the quadruplet integral.

        Parameters
        ----------
        verbose: bool, optional
            Print the progress of the integral. Defaults to ``False``.
        nthreads: int, optional
            Number of OpenMP threads. Defaults to ``nthreads`` of the instance.

        Returns
        -------
        rp0, rp1, rm0, rm1, rpm: numpy.ndarray
            The five r-terms, each of shape (n_r, n_r).
        """
        self._step("r-terms on %d bins" % self.n_r)
        if self.method_quadruplets == 'continuous':
            results = self._rterms_continuous_c()
        else:
            results = self._quadruplet_density_terms_c(verbose=verbose, nthreads=nthreads)
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
        if any(getattr(self, n) is None for n in self.RTERMS):
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
        ww_factor = self._w2**2/self._w1**2 if self._w1 != 0 else 0.
        self.D = self.sigma_eps**4*ww_factor / self.pairs
        return self.D

    def _triplet_term(self, weight_type, xi_func, sigma_eps_sq=None, nbar_fac=1.):
        """Mixed term q_++, q_-- or q_+- from the triplet counts."""
        if self.triplets is None:
            raise RuntimeError("`triplets` not initialised.")
        if sigma_eps_sq is None:
            sigma_eps_sq = self.sigma_eps**2
        pre = 2.*sigma_eps_sq*nbar_fac/np.outer(self.pairs, self.pairs)
        # The tomographic combos share the reduction of each weight and xi. The entry holds xi_func
        # alive, so that its id stays unique
        if self._triplet_reduced[0] is not self.triplets:
            self._triplet_reduced = (self.triplets, {})
        reduced = self._triplet_reduced[1]
        key = (weight_type, id(xi_func))
        if key not in reduced:
            fine = self._triplet_phi_reduce(self.triplets, self.triplet_theta_edges, weight_type,
                                            xi_func)
            if self.triplet_oversample > 1:
                fine = fine.reshape(self.n_r, self.triplet_oversample, self.n_r,
                                    self.triplet_oversample).sum(axis=(1, 3))
            reduced[key] = (xi_func, fine)
        return pre * reduced[key][1]

    def _triplet_phi_reduce(self, Nt, te, weight_type, xi_func):
        """Integral over phi of the triplets per unit angle Nt (Fourier basis) against w(phi)
        xi(r_jk), via the C kernel."""
        nr = len(te) - 1
        nsub_r, n_ang = self.TRIPLET_SUBNODES
        # Smallest third side of a triplet: 
        # Both sides at the first radial sub-node, angle at the node closest to 0
        r_min = 2.*(te[0] + .5*(te[1]-te[0])/nsub_r)*np.sin(.5*np.pi/n_ang)
        xi_x, xi_y, logx0, inv_dlogx, xi_n = self._xi_tables_from_splines(
            [xi_func], lo=min(te[0], self._xi_query_lo(), .9*r_min), hi=2.*te[-1])
        Nt = np.ascontiguousarray(Nt, dtype=np.float64)
        te = np.ascontiguousarray(te, dtype=np.float64)
        def reduce():
            out = np.zeros((nr, nr), dtype=np.float64)
            self._clib.triplet_term_reduce(
                Nt, nr, Nt.shape[-1], te, nsub_r, n_ang, int(weight_type), xi_x, xi_n, logx0,
                inv_dlogx, xi_y[0], out, int(self.nthreads), 0)
            check_clib_error(self._clib)
            return out
        return self._shared_integral(reduce, Nt, te, np.array([nsub_r, n_ang, weight_type]), xi_x,
                                     xi_y)

    def _quadruplet_density_terms_c(self, verbose=False, nthreads=None):
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
            verbose=verbose, nthreads=nthreads)

    def _run_postq_engine(self, cbm, xi_tables, symmetric_mask, verbose=False, nthreads=None):
        """Run the C quadruplet kernel over the phi grid."""

        ## 1) Misc preparations
        if nthreads is None:
            nthreads = self.nthreads
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = xi_tables
        n_out = cbm['n_out']
        n = self.n_r

        def _f(a): return np.ascontiguousarray(a, dtype=np.float64)
        def _i(a): return np.ascontiguousarray(a, dtype=np.int32)

        ## 2) Define radial/angular integration nodes and their weights
        def _nodes(frac):
            if frac is None:
                return self.theta_centers[:, None], _theta_dtheta(self.theta_edges)[:, None]
            nodes = self.theta_centers[:, None] + np.outer(self.dtheta, frac)
            return nodes, nodes * self.dtheta[:, None] * _trapz_weights(frac)[None, :]
        phi1_nodes, phi1_weights = _nodes(self.phi1_subgrid_frac)
        phi2_nodes, phi2_weights = _nodes(self.phi2_subgrid_frac)
        ns1, ns2 = phi1_nodes.shape[1], phi2_nodes.shape[1]
        phi1_flat = phi1_nodes.ravel()
        phi2_flat = phi2_nodes.ravel()

        phi_grid = self.phi_grid
        phi_w = _trapz_weights(phi_grid)
        v1_w = _periodic_weights(self.varphi1_grid)
        v2_w = _periodic_weights(self.varphi2_grid)

        ## 3) Define quadruplet density representation
        # q = q_const s(phi) s(theta1) s(theta2) C with C the tensor of the real Fourier
        # coefficients of a measured density, and for an analytic density its constant q_const,
        # the suppression s, 1 without one, and C = 1 on two radial knots above every separation,
        # below which the kernel holds it at the first
        if self.quadruplets is None:
            raise RuntimeError("`quadruplets` not initialised.")
        sup = self.spl_suppression_ww if callable(self.quadruplets) else None
        if not callable(self.quadruplets):
            q_const, q_array, q_rc = 1., _f(self.quadruplets), _f(self.q_theta_centers)
        else:
            if self.method_quadruplets not in ('analytic', 'frompairs'):
                raise NotImplementedError(
                    "the quadruplet integral takes a density tensor or one of "
                    "the analytic densities, not a user-supplied callable; "
                    "materialise it with get_quadruplets() and pass the "
                    "result as `quadruplets`.")
            q_const, q_array = self._q_const, np.ones((2,)*3 + (1, 1))
            q_rc = 2.*(phi_grid[-1] + phi2_flat.max())*np.array([1., 2.])
        q_nr, q_nm = q_array.shape[0], q_array.shape[-1]
        def _sup(x): return np.ones(len(x)) if sup is None else sup(x)
        sup_t1, sup_t2, sup_t3 = _sup(phi_grid), _sup(phi1_flat), _sup(phi2_flat)

        ## 4) Do the function call and rescale the output
        results = np.zeros((n_out, n, n))
        self._clib.quadruplet_sva_integrate(
            _f(phi_grid), _f(phi_w), len(phi_grid), n,
            _f(phi1_flat), _f(phi1_weights.ravel()), ns1,
            _f(phi2_flat), _f(phi2_weights.ravel()), ns2,
            _f(self.varphi1_grid), _f(v1_w), len(self.varphi1_grid),
            _f(self.varphi2_grid), _f(v2_w), len(self.varphi2_grid),
            q_array, q_rc, q_nr, q_nm, int(self.q_interpolation == 'cubic'),
            q_const, _f(sup_t1), _f(sup_t2), _f(sup_t3),
            _f(xi_x), xi_n, xi_logx0, xi_inv_dlogx,
            _f(xi_y_tables), len(xi_y_tables),
            len(cbm['xi_a_idx']), _i(cbm['xi_a_idx']), _i(cbm['xi_b_idx']),
            _i(cbm['w_ids']), _i(cbm['out_idx']), _f(cbm['wick_fac']),
            n_out, _i(symmetric_mask), results, nthreads, int(verbose))
        check_clib_error(self._clib)

        results *= 2.*np.pi*self.area / np.outer(self.pairs, self.pairs)
        return results

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
        out = self._qterms_continuous_raw([xi_func], [(w_type, 0)], 2.*self.theta_edges[-1],
                                          self.sigma_eps**2, self.nbar, nr_target, nphi_target)
        # Shared galaxy carries <w^2>, the other two <w> each
        return self._w2 / self._w1**2 * out[0]

    def _qterms_continuous_raw(self, splines, calls, hi, sigma_eps2, nbar, nr_target, nphi_target,
                               verbose=False):
        """Continuum mixed terms (n_calls, n_r, n_r) of the calls (w_type, index of the xi in
        splines), with xi tabulated up to hi, averaged over nr_target fine bins."""
        alpha = max(1, int(nr_target // self.n_r))
        n_fine = alpha * self.n_r
        _edges = np.geomspace(self.theta_edges[0], self.theta_edges[-1], n_fine + 1)
        thetas = .5 * (_edges[1:] + _edges[:-1])
        phis = np.linspace(0., np.pi, nphi_target, endpoint=False)
        # Set r_min by smallest third side of a triplet --> 2 theta_min*sin(phi_min/2)
        r_min = min(thetas[1]-thetas[0], 2.*thetas[0]*np.sin(.5 * phis[1]))
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = self._xi_tables_from_splines(
            splines, lo=min(self._xi_query_lo(), .9*r_min), hi=hi)
        calls = np.array(calls, dtype=np.int32).reshape(-1, 2)
        def integrate():
            out = np.zeros((len(calls), self.n_r, self.n_r))
            self._clib.continuous_q_tomo(
                self.n_r, alpha, thetas, thetas * np.diff(_edges), phis, _trapz_weights(phis),
                n_fine, nphi_target, xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y_tables,
                float(sigma_eps2), float(self.area), float(nbar), len(calls),
                np.ascontiguousarray(calls[:, 1]), np.ascontiguousarray(calls[:, 0]), out,
                int(self.nthreads), int(verbose))
            check_clib_error(self._clib)
            return out
        return self._shared_integral(integrate, _edges, phis, xi_x, xi_y_tables, calls,
                                     np.array([sigma_eps2, self.area, nbar, self.n_r]))

    def _rterms_continuous_c(self, dlnt_target=.05):
        """Helper to get continuous r-terms via C kernel, averaging out over bins using
        a fine binsize of dlnt_target."""
        # Memoised against the grids, since the getters ask for one term at a time
        cached = getattr(self, '_rterms_cont_cache', None)
        if (cached is not None and cached[0] is self.phi_grid
                and cached[1] is self.varphi1_grid and cached[2] == self.n_r):
            return cached[3]
        # One work unit, xi_+ (table 0) and xi_- (table 1) of both pairs, into five outputs
        xi_tables = self._xi_tables_from_splines(
            [self.xip, self.xim], lo=self._xi_query_lo(), hi=self._xi_query_hi())
        out = self._rterms_continuous_raw(xi_tables, ([0], [0], [1], [1], [0]), 5, dlnt_target)
        self._rterms_cont_cache = (self.phi_grid, self.varphi1_grid, self.n_r, out)
        return out

    def _rterms_continuous_raw(self, xi_tables, work, n_out, dlnt_target, verbose=False):
        """Continuum r-terms (n_out, n_r, n_r) of the work units (xi_a_p, xi_b_p, xi_a_m, xi_b_m,
        out_idx) of continuous_rterms_tomo, averaged over log sub-bins no wider than dlnt_target."""
        def _f(a): return np.ascontiguousarray(a, dtype=np.float64)
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = xi_tables
        v = self.varphi1_grid
        nodes, weights = self._rterm_nodes(dlnt_target)
        results = np.zeros(n_out*nodes.size*nodes.size)
        self._clib.continuous_rterms_tomo(
            nodes.size, _f(nodes.ravel()), _f(self.phi_grid), _f(_trapz_weights(self.phi_grid)),
            len(self.phi_grid), _f(v), _f(_periodic_weights(v)), len(v), float(self.area),
            xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y_tables, len(xi_y_tables), len(work[0]),
            *[np.array(w, dtype=np.int32) for w in work], results, int(self.nthreads),
            int(verbose))
        check_clib_error(self._clib)
        return _bin_average_blocks(results.reshape(n_out, nodes.size, nodes.size), weights)

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
        ``pairs_wsq``), ``'triplets'`` (N_t per unit angle of shape (triplet_n_r, triplet_n_r,
        t_n_modes), with the squared weight on the base galaxy) or ``'quadruplets'`` (density q of
        shape (q_n_r, q_n_r, q_n_r, q_n_modes, q_n_modes)), the last two as the real Fourier
        coefficients of their angles up to ``t_nmaxs`` and ``q_nmaxs``. The binning follows the
        covariance grids, and the results are cached in ``path_save`` under a hash of the inputs.
        """
        if self.catalog is None or self.path_save is None:
            raise ValueError(
                f"computeDiscrete('{contribution}') requires `catalog` and `path_save`")
        key, binning = {
            'pairs': ('NN', (self.theta_edges[0], self.theta_edges[-1], self.n_r)),
            'triplets': ('NNN', (self.theta_edges[0], self.theta_edges[-1], self.triplet_n_r,
                                 self.t_nmaxs)),
            'quadruplets': ('NNNN', (self.q_theta_edges[0], self.q_theta_edges[-1], self.q_n_r,
                                     self.q_nmaxs))}[contribution]
        grid = self._MULTIPLET_GRIDS[contribution]

        ## Cache file, hashing the catalog, the binning and the correlator setup ##
        # area enters the cached quadruplet density
        rmin, resos = {'pairs': (self.rmin_pixsize, self.tree_resos),
                       'triplets': (self.t_rmin_pixsize, self.t_tree_resos),
                       'quadruplets': (self.q_rmin_pixsize, self.q_tree_resos)}[contribution]
        nmax = {'pairs': None, 'triplets': self.t_nmaxs, 'quadruplets': self.q_nmaxs}[contribution]
        setup = (self.geometry, nmax, rmin,
                 list(np.ravel(resos)), self.tree_method, int(self.shuffle_pix), self.area)
        h = self._catalog_sha1.copy()
        h.update(repr((contribution, tuple(map(float, binning)), setup)).encode())
        fpath = os.path.join(self.path_save, "%s_%s.npz" % (contribution, h.hexdigest()[:16]))
        bins = {'pairs': "%d bins" % self.n_r,
                'triplets': "%d bins up to multipole %d" % (self.triplet_n_r, self.t_nmaxs),
                'quadruplets': "%d bins up to multipole %d" % (self.q_n_r, self.q_nmaxs)
                }[contribution]
        self._step("%s the %s on %s" % ("Loading" if os.path.exists(fpath) else "Measuring",
                                        contribution, bins))
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
                if not self._unit_weights:
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
                # With unit weights the pairs of squared weights are the pairs
                if self._unit_weights:
                    self.pairs_wsq = extra['pairs_wsq'] = counts.copy()
            if contribution == 'triplets':
                counts = self._triplets_from_multipoles(corr.npcf_multipoles_norm[:, 0])
            if contribution == 'quadruplets':
                # Density q = N_q / V_q with the separable volume of exact annuli
                self.set_quadruplets_from_multipoles(corr.npcf_multipoles[:, :, 0])
                counts = self.quadruplets
                extra['nbar_w'] = self._nbar_w()
            if self.store_multiplets:
                np.savez(fpath, **{contribution: counts}, bin_centers_mean=bc, area=self.area,
                         **extra)
            # The file keeps the sparse bins and the bin means of the density
            if contribution == 'quadruplets':
                self._merge_sparse_quadruplets(counts, extra['nbar_w'])
                self._density_to_knots(counts)
        self._check_shape(contribution, counts)
        return counts

    def set_orpheus_instances(self, contributions=('pairs', 'triplets', 'quadruplets')):
        """Init the catalogs and correlators that measure the ``'discrete'`` multiplets."""
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
        self._unit_weights = bool(np.all(w == 1.))
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
        if 'pairs' in contributions and 'catalog_wsq' not in self.orpheus_instances and not self._unit_weights:
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
                nbinsr=self.triplet_n_r, nmaxs=self.t_nmaxs, nbinsphi=self.t_n_modes,
                base_weightsq=True,
                **dict(xi_kw, rmin_pixsize=self.t_rmin_pixsize, tree_resos=self.t_tree_resos))
        if 'quadruplets' in contributions:
            fac = 60. if sphere else 1.
            self.orpheus_instances['NNNN'] = NNNNCorrelation_NoTomo(
                min_sep=float(self.q_theta_edges[0])/fac, max_sep=float(self.q_theta_edges[-1])/fac,
                nbinsr=self.q_n_r, nbinsphi=self.q_n_modes, nmaxs=self.q_nmaxs,
                method=self.tree_method,
                tree_resos=self.q_tree_resos/fac,
                sep_units='deg' if sphere else 'arcmin', **dict(kw, rmin_pixsize=self.q_rmin_pixsize))

    def _load_multiplet(self, kind, fpath):
        """Loads a multiplet from a file written by ``store_multiplets`` and update relevant attributes."""
        dat = np.load(fpath)
        setattr(self, self._MULTIPLET_GRIDS[kind], dat['bin_centers_mean'])
        if kind == 'pairs':
            self.pairs_wsq = np.asarray(dat['pairs_wsq'], dtype=float)
        # A file is read once as long as an instance holds its tensor, a density once per
        # q_count_min of its merged sparse bins, interpolation and extrapolation
        key = (os.path.realpath(fpath), os.stat(fpath).st_mtime_ns, kind,
               (self.q_count_min, self.q_interpolation, self.q_richardson)
               if kind == 'quadruplets' else None)
        counts = _LOADED_MULTIPLETS.get(key)
        if counts is None:
            counts = np.asarray(dat[kind], dtype=float)
            self._adopt_modes(kind, counts)
            self._check_shape(kind, counts)
            if kind == 'quadruplets':
                # A file without the density of its catalog, e.g. of an earlier version, takes the
                # one of `catalog`, and without either keeps its sparse bins
                self._merge_sparse_quadruplets(
                    counts, dat['nbar_w'] if 'nbar_w' in dat else self._nbar_w())
                self._density_to_knots(counts)
            _LOADED_MULTIPLETS[key] = counts
        self._adopt_modes(kind, counts)
        return counts

    def _adopt_modes(self, kind, arr):
        """A triplet or quadruplet tensor that is passed or read sets the largest multipole of its
        kind."""
        if kind in ('triplets', 'quadruplets'):
            nmax = (np.shape(arr)[-1] - 1)//2
            setattr(self, kind[0] + '_nmaxs', nmax)
            setattr(self, kind[0] + '_n_modes', 2*nmax+1)

    def _nbar_w(self):
        """Weighted galaxy density of ``catalog``, None without one."""
        if self.catalog is None:
            return None
        w = self.catalog.get('weight')
        return (len(self.catalog['ra']) if w is None else np.sum(w))/self.area

    def _merge_sparse_quadruplets(self, q, nbar_w):
        """Density of the sparse radial bins of a density measured on a catalog of weighted density
        ``nbar_w`` from the blocks of radial bins that hold ``q_count_min`` quadruplets per angular
        resolution cell, in place."""
        if self.q_count_min <= 0. or nbar_w is None:
            return
        self._clib.quadruplets_merge_sparse(
            q, self.q_n_r, self.q_n_modes, np.ascontiguousarray(_theta_dtheta(self.q_theta_edges)),
            2.*np.pi*self.area*(2.*np.pi/self.q_n_modes)**2, nbar_w**4, self.q_count_min,
            self.nthreads)
        check_clib_error(self._clib)

    def _density_to_knots(self, q):
        """Values of a measured density between which it is interpolated from its bin means, in
        place: those whose interpolant has the bin means, extrapolated in the radial bin width
        with ``q_richardson``."""
        # Averaged over a bin, the cubic (linear) interpolant exceeds its knot value by the second
        # difference of the knot values over 24 (8)
        cubic = int(self.q_interpolation == 'cubic')
        coef = 1./24. if cubic else 1./8.
        if self.q_richardson:
            self._clib.quadruplets_richardson(
                q, self.q_n_r, self.q_n_modes, _theta_dtheta(self.q_theta_edges),
                np.ascontiguousarray(self.q_theta_centers, dtype=np.float64), coef, cubic,
                self.nthreads)
        else:
            self._clib.quadruplets_to_knots(q, self.q_n_r, self.q_n_modes, coef, self.nthreads)
        check_clib_error(self._clib)
        return q

    def set_quadruplets_from_multipoles(self, N_n):
        r"""Convert the multipoles of the quadruplet counts to the density and store it.

        The counts per unit angle are sum_n N_n e^(i (n_1 phi_12 + n_2 phi_13)) / (2 pi)^2 and the
        density divides them by

        .. math::

            V_q = 2\pi A\, \Theta_1 \Theta_2 \Theta_3

        with the exact annuli :math:`\Theta_i = (\theta_{i,\max}^2 - \theta_{i,\min}^2)/2` of the
        radial bins. ``N_n`` has the shape (q_n_modes, q_n_modes, q_n_r, q_n_r, q_n_r) over
        (n_1, n_2, theta_1, theta_2, theta_3), and the density is stored in the real Fourier basis
        of both angles, (q_n_r, q_n_r, q_n_r, q_n_modes, q_n_modes).
        """
        U = _fourier_map(self.q_nmaxs)
        tdt = _theta_dtheta(self.q_theta_edges)
        q = np.empty((self.q_n_r,)*3 + (self.q_n_modes,)*2)
        # One slice of the first radial axis at a time, as the multipoles may be large
        for i in range(self.q_n_r):
            c = np.tensordot(np.tensordot(U, N_n[:, :, i], axes=(0, 0)), U, axes=(1, 0))
            q[i] = np.moveaxis(c.real, 0, -2)/((2.*np.pi)**3*self.area*tdt[i])
        q /= tdt[None, :, None, None, None]
        q /= tdt[None, None, :, None, None]
        self.quadruplets = q

    def _triplets_from_multipoles(self, N_n):
        """Triplets per unit angle in the real Fourier basis from the multipoles N_n (t_nmaxs+1,
        triplet_n_r, triplet_n_r) of the counts, whose counts per unit angle are
        (N_0 + sum_n>0 (N_n e^(i n phi) + c.c.))/(2 pi)."""
        out = np.zeros(N_n.shape[1:] + (self.t_n_modes,))
        out[..., 0] = N_n[0].real
        for n in range(1, self.t_nmaxs+1):
            out[..., 2*n-1], out[..., 2*n] = 2.*N_n[n].real, -2.*N_n[n].imag
        return out/(2.*np.pi)

    def get_triplets(self, phis=None, bins=(slice(None),)*2):
        """Return the triplets per unit angle on the radial bins ``bins``, an index of the axes of
        either side.

        Without ``phis``, their real Fourier coefficients in the last axis. With the angles ``phis``
        of shape (n_phi,), their values at the angle between the two sides on them, e.g. of shape
        (n_phi,) for one bin ``(i, j)``.
        """
        coef = self.triplets[tuple(bins)]
        if phis is None:
            return coef
        return coef @ _fourier_basis(np.asarray(phis, dtype=float), self.t_n_modes).T

    def get_quadruplets(self, phis=None, bins=(slice(None),)*3):
        """Return the quadruplet density on the radial bins ``bins``, an index of the axes of
        theta_1, theta_2 and theta_3.

        Without ``phis``, its real Fourier coefficients in the last two axes, which an analytic
        density takes from its values on 2*q_nmaxs+1 uniform angles. With the angles ``phis`` of
        shape (n_phi,), its values at phi_12 and phi_13 on them, e.g. of shape (n_phi, n_phi) for
        one bin ``(i, j, k)``.
        """
        M = self.q_n_modes
        if not callable(self.quadruplets):
            coef = self.quadruplets[tuple(bins)]
            if phis is None:
                return coef
            B = _fourier_basis(np.asarray(phis, dtype=float), M)
            return np.matmul(B, np.matmul(coef, B.T))
        # The separations of the bins on axes of their own, followed by the two angles
        t1, t2, t3 = (np.asarray(self.q_theta_centers[b]) for b in bins)
        ang = 2.*np.pi*np.arange(M)/M if phis is None else np.asarray(phis, dtype=float)
        vals = np.broadcast_to(self.quadruplets(
            t1.reshape(t1.shape + (1,)*(t2.ndim + t3.ndim + 2)),
            t2.reshape(t2.shape + (1,)*(t3.ndim + 2)), t3.reshape(t3.shape + (1, 1)),
            ang[:, None], ang[None, :]), t1.shape + t2.shape + t3.shape + (len(ang),)*2)
        if phis is not None:
            return vals.copy()
        # On M uniform angles the basis is orthogonal, with norm M for the constant and M/2 else
        P = _fourier_basis(ang, M).T*np.where(np.arange(M) == 0, 1., 2.)[:, None]/M
        return np.matmul(P, np.matmul(vals, P.T))

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
        tc, rdr = self.triplet_theta_centers, _theta_dtheta(self.triplet_theta_edges)
        out = np.zeros((len(tc), len(tc), self.t_n_modes))
        out[..., 0] = (2.*np.pi*self.area * np.outer(rdr, rdr) * self.nbar**3 * self._w2*self._w1**2
                       * np.outer(self._suppression(tc), self._suppression(tc)))
        return out

    def quadruplets_analytic(self):
        """Idealised quadruplet density, returned as a callable. """
        # Make sure that the spline covers all scales
        if self.method_quadruplets == 'frompairs':
            if self.spl_suppression_ww is None:
                raise ValueError(
                    "method_quadruplets='frompairs' requires `spl_suppression_ww` or a `catalog` "
                    "to measure it on")
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

    def _catalog_suppression(self):
        """Pair-count suppression of ``catalog`` on the radial bins of the quadruplet density,
        cached in ``path_save``."""
        from .npcf_second import NNCorrelation
        edges = self.q_theta_edges
        # Check if pairs are available
        h = self._catalog_sha1.copy()
        h.update(repr(('suppression', (float(edges[0]), float(edges[-1]), len(edges)-1),
                       (self.geometry, self.rmin_pixsize, list(np.ravel(self.tree_resos)),
                        int(self.shuffle_pix)))).encode())
        fpath = None if self.path_save is None else os.path.join(
            self.path_save, "suppression_%s.npz" % h.hexdigest()[:16])
        cached = fpath is not None and os.path.exists(fpath)
        self._step("%s the pairs of the suppression on %d bins"
                   % ("Loading" if cached else "Measuring", len(edges)-1))
        # If available load pairs, else recompute pairs
        if cached:
            counts = np.load(fpath)['pairs']
        else:
            corr = NNCorrelation(min_sep=float(edges[0]), max_sep=float(edges[-1]),
                                 nbinsr=len(edges)-1, tree_resos=self.tree_resos,
                                 rmin_pixsize=self.rmin_pixsize, shuffle_pix=self.shuffle_pix,
                                 process_spherical=self.geometry == 'spherical',
                                 verbosity=self.verbosity, nthreads=self.nthreads)
            corr.process(self.orpheus_instances['catalog'])
            counts = np.array(corr.npair[0], dtype=float)
            if self.store_multiplets and fpath is not None:
                np.savez(fpath, pairs=counts)
        # Weighted pairs over their idealised number at the weighted density of the catalog
        return self.suppression_spline_from_pairs(counts, np.sqrt(edges[1:]*edges[:-1]), edges,
                                                  self.area, self._nbar_w())

    ###################
    # General helpers #
    ###################

    def _shared_integral(self, compute, *inputs):
        """Array of ``compute()``, a mixed-term kernel on ``inputs``, taken from another instance
        that holds it for the same inputs."""
        h = hashlib.sha1()
        for x in inputs:
            x = np.ascontiguousarray(x)
            h.update(repr((x.shape, x.dtype.str)).encode())
            h.update(memoryview(x).cast('B'))
        key = h.hexdigest()
        out = _MIXED_INTEGRALS.get(key)
        if out is None:
            out = _MIXED_INTEGRALS[key] = compute()
            out.flags.writeable = False
        self._mixed_integrals[key] = out
        return out

    @staticmethod
    def _xi_tables_from_splines(splines, lo, hi):
        """Tabulate xi splines on shared log-uniform knots."""
        # The knots of the first spline if log-uniform, as the kernels assume, else log-uniform
        # ones in [lo, hi] within them, with xi zero outside of the knots
        x = getattr(splines[0], 'x', None)
        if x is None:
            x = np.geomspace(lo, hi, 4096)
        elif not np.allclose(np.diff(np.log(x)), np.log(x[1]/x[0]), rtol=1e-6, atol=0.):
            x = np.geomspace(max(lo, x[0]), min(hi, x[-1]), 4096)
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
                    'triplets': (self.triplet_n_r, self.triplet_n_r, self.t_n_modes),
                    'quadruplets': (self.q_n_r, self.q_n_r, self.q_n_r, self.q_n_modes,
                                    self.q_n_modes)}[kind]
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
        def per_bin(name, value):
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            if arr.size == 1:
                arr = np.full(self.ntomo, arr[0])
            if arr.size != self.ntomo:
                raise ValueError(f"{name} must be a scalar or length-{self.ntomo} array, "
                                 f"got length {arr.size}")
            return arr
        # The parent gets the first entries, which are replaced by the arrays afterwards
        se = per_bin('sigma_eps', kwargs.get('sigma_eps', 0.27))
        kwargs['sigma_eps'] = float(se[0])
        nb = None if kwargs.get('nbar') is None else per_bin('nbar', kwargs['nbar'])
        if nb is not None:
            # The idealised multiplets of the parent live at the reference density
            kwargs['nbar'] = float(nb[0]) if nbar_ref is None else float(nbar_ref)

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
            return {(min(i, j), max(i, j)): spl for i, j, spl in raw}

        xip = _to_dict(raw_xip)
        xim = _to_dict(raw_xim)

        # Infer ntomo from whichever dict is non-trivial, n_tp = ntomo*(ntomo+1)/2
        n_tp = max(len(xip), len(xim), 1)
        ntomo = int((-1. + np.sqrt(1. + 8. * n_tp)) / 2.)
        if ntomo * (ntomo + 1) // 2 != n_tp:
            raise ValueError(f"Number of xi spline entries ({n_tp}) is not ntomo*(ntomo+1)/2 for "
                             "any integer ntomo")
        return xip, xim, ntomo

    ###################################
    # Covariance of any GG statistics #
    ###################################

    def cov_of_statistic(self, filter_type, split_components=False, **kwargs):
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
        **kwargs
            Further parameters of :meth:`GGCovarianceNoTomo.cov_of_statistic`.

        Returns
        -------
        dict, or tuple of two dicts with ``split_components``
            The keys of :meth:`GGCovarianceNoTomo.cov_of_statistic`, with blocks of shape
            (n_stat*n_tp, n_stat*n_tp).
        """
        if self.cov_xipm_tomo is None:
            self.get_cov_xipm()

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

        # Left and right weights of each output: (K, K) for puremode, else W_E and W_B
        if isinstance(weights[1], list):
            proj = {'cov': (weights[0], weights[0])}
        else:
            W_E, W_B = weights
            proj = {'cov_EE': (W_E, W_E), 'cov_BB': (W_B, W_B), 'cov_EB': (W_E, W_B)}
        ns = weights[0].shape[0]
        total = {key: np.zeros((ns * ntp, ns * ntp)) for key in proj}
        for i1 in range(ntp):
            for i2 in range(ntp):
                block = _extract_block(i1, i2)
                for key, (A, B) in proj.items():
                    total[key][i1*ns:(i1+1)*ns, i2*ns:(i2+1)*ns] = A @ block @ B.T
        if 'cov' in proj:
            total['labels'] = weights[1]

        if not split_components:
            return total

        ## Per-component decomposition ##
        # The xi_+ and xi_- halves of K or W_E, with the sign of the xi_- weights of W_B as the
        # signs of the pp, mm and pm parts in the outputs
        Wp, Wm = weights[0][:, :n], weights[0][:, n:]

        def _accum(out, sl1, sl2, r, signs):
            for key, sign in zip(out, signs):
                out[key][sl1, sl2] += sign * r

        def _term(name, tp1, tp2):
            if name == 'D':
                return np.diag(self.D_tomo[tp1])
            if name in self.RTERMS:
                return self._rterm_block(name, (tp1, tp2))
            return getattr(self, name + '_tomo')[min(tp1, tp2), max(tp1, tp2)]

        sz = ns * ntp
        qnames = [name for name, *_ in self.QTERMS]
        comps = {name: {key: np.zeros((sz, sz)) for key in proj}
                 for name in ['D'] + qnames + list(self.RTERMS)}
        for i1, tp1 in enumerate(self.tomopairs_list):
            for i2, tp2 in enumerate(self.tomopairs_list):
                sl1 = slice(i1*ns, (i1+1)*ns)
                sl2 = slice(i2*ns, (i2+1)*ns)
                for block, _, qname, rnames, with_D in self.COVBLOCKS:
                    for name in (['D'] if with_D and i1 == i2 else []) + [qname, *rnames]:
                        if block != 'pm':
                            W = Wp if block == 'pp' else Wm
                            _accum(comps[name], sl1, sl2, W @ _term(name, tp1, tp2) @ W.T,
                                   (1., 1., 1. if block == 'pp' else -1.))
                            continue
                        t1 = Wp @ _term(name, tp1, tp2) @ Wm.T
                        t2 = Wm @ _term(name, tp2, tp1).T @ Wp.T
                        _accum(comps[name], sl1, sl2, t1 + t2, (1., -1.))
                        if 'cov_EB' in proj:
                            comps[name]['cov_EB'][sl1, sl2] += t2 - t1

        for name, parts in (('mix', qnames), ('sva', self.RTERMS)):
            comps[name] = {key: sum(comps[part][key] for part in parts) for key in proj}

        return total, comps

    #######################
    # Covariance of xi_pm #
    #######################

    def get_cov_xipm(self, verbose=False):
        """Assemble the full (2*n_tp*n_r, 2*n_tp*n_r) tomo covariance."""
        if self.cov_pp_tomo is None: self.get_cov_pp(verbose=verbose)
        if self.cov_mm_tomo is None: self.get_cov_mm(verbose=verbose)
        if self.cov_pm_tomo is None: self.get_cov_pm(verbose=verbose)
        self.cov_xipm_tomo = np.block([[self.cov_pp_tomo, self.cov_pm_tomo],
                                       [self.cov_pm_tomo.T, self.cov_mm_tomo]])
        return self.cov_xipm_tomo

    def get_cov_terms(self, verbose=False):
        """Shape-noise, mixed and sample-variance terms of the tomographic covariance blocks.

        Returns ``{block: {'D': ..., 'mix': ..., 'sva': ...}}`` for the blocks ``'pp'``, ``'mm'``
        and ``'pm'`` of :meth:`get_cov_xipm`, each term of shape (n_tp*n_r, n_tp*n_r), and the
        three terms of a block sum to it.
        """
        for index, (block, *_) in enumerate(self.COVBLOCKS):
            if block not in self.cov_terms_tomo:
                self._get_cov_block(index, verbose)
        return self.cov_terms_tomo

    def _get_cov_block(self, index, verbose=False, nthreads=None):
        """Assemble one tomo covariance block, by position in COVBLOCKS."""
        block, _, qname, rnames, with_D = self.COVBLOCKS[index]
        q_tomo = getattr(self, qname + '_tomo')
        if with_D and self.D_tomo is None:
            self.getD()
        if q_tomo is None:
            q_tomo = getattr(self, 'get' + qname)()
        if not self.rterms_tomo:
            self.get_all_rterms(verbose=verbose, nthreads=nthreads)
        n = self.n_r
        # Shape-noise, mixed and sample-variance terms, kept apart for
        # get_cov_terms
        terms = {name: np.zeros((self.n_tomopairs * n, self.n_tomopairs * n))
                 for name in ('D', 'mix', 'sva')}
        for idx1, tp1 in enumerate(self.tomopairs_list):
            if with_D:
                # Diagonal shape noise.
                terms['D'][idx1*n:(idx1+1)*n, idx1*n:(idx1+1)*n] += np.diag(self.D_tomo[tp1])
            for idx2, tp2 in enumerate(self.tomopairs_list):
                sl1 = slice(idx1 * n, (idx1+1)*n)
                sl2 = slice(idx2 * n, (idx2+1)*n)
                terms['mix'][sl1, sl2] += q_tomo[min(tp1, tp2), max(tp1, tp2)]
                for rname in rnames:
                    terms['sva'][sl1, sl2] += self._rterm_block(rname, (tp1, tp2))
        self.cov_terms_tomo[block] = terms
        out = terms['D'] + terms['mix'] + terms['sva']
        setattr(self, 'cov_' + block + '_tomo', out)
        return out

    ## Shape noise and mixed terms ##

    def getD(self):
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

    def _get_qterm(self, index):
        """One mixed term of all tomo combos, by its position in :attr:`QTERMS`, from the triplet
        counts or in the continuum limit."""
        if self.method_triplets != 'continuous':
            return self._q_tomo_blocks(index)
        # The continuum limit integrates all three mixed terms at once
        name = self.QTERMS[index][0]
        if getattr(self, name + '_tomo') is None:
            self._qterms_continuous_tomo_c()
        return getattr(self, name + '_tomo')

    ## Cosmic variance ##

    def get_all_rterms(self, verbose=False, nthreads=None):
        """Compute all tomographic r-term blocks in one pass.

        Returns ``{(rterm, ((a,b),(c,d))): (n_r, n_r)}``.
        """
        if nthreads is None:
            nthreads = self.nthreads
        self._step("r-terms on %d bins" % self.n_r)
        if self.method_quadruplets == 'continuous':
            rterms = self._rterms_continuous_tomo_c(verbose=verbose)
        else:
            rterms = self._quadruplet_density_terms_tomo_c(verbose=verbose, nthreads=nthreads)
        self.rterms_tomo = rterms
        return rterms

    ## Getters of GGCovarianceNoTomo for a single bin pair ##

    def _notomo_only(self, *_args, **_kwargs):
        raise NotImplementedError(
            "This getter of GGCovarianceNoTomo returns a single bin pair. GGCovariance holds every bin "
            "combination, see getD, getqpp, getqmm, getqpm and get_all_rterms.")

    getD_discrete = getD_continuous = _notomo_only
    getqpp_continuous = getqmm_continuous = getqpm_continuous = _notomo_only
    getrp0 = getrp1 = getrm0 = getrm1 = getrpm = _notomo_only
    getrp0_continuous = getrp1_continuous = getrm0_continuous = getrm1_continuous = getrpm_continuous = _notomo_only

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

    def _quadruplet_density_terms_tomo_c(self, verbose=False, nthreads=None):
        """Tomographic r-terms via the C kernel with ``nthreads`` OpenMP threads."""
        cbm = self._build_tomo_callback_map()
        out_keys = cbm['out_keys']
        xi_tables = self._build_xi_tables(cbm['xi_table_keys'])

        # Per-output-block mask: True if this block's r-term is symmetric, which the kernel
        # symmetrises
        symmetric_mask = [key[0] in self.RTERMS_SYM and key[1][0] == key[1][1]
                          for key in out_keys]

        accum = self._run_postq_engine(cbm, xi_tables, symmetric_mask, verbose=verbose,
                                       nthreads=nthreads)
        return dict(zip(out_keys, accum))

    def _build_tomo_callback_map(self):
        """Flat callback arrays of all tomographic r-terms for the C kernel.

        Returns a dict of ``xi_a_idx`` and ``xi_b_idx`` (xi table of each side), ``w_ids`` (weight
        0-4), ``out_idx`` (output block), ``wick_fac`` (1 for a degenerate Wick pairing, 0.5 for
        each of two distinct ones), ``xi_table_keys`` ((pm, i, j) of each table), ``out_keys``
        ((r-term, combo) of each output) and ``n_out``.
        """
        xi_keys, idx = self._xi_keys()
        combos_sym = self._tomo_combos()
        combos_all = [(tp1, tp2) for tp1 in self.tomopairs_list for tp2 in self.tomopairs_list]
        # Symmetric r-terms on the upper-triangle combos, rpm on all ordered ones as cov_pm is not
        # symmetric under bin swap
        out_keys, rows = [], []
        for wid, rname in enumerate(self.RTERMS):
            xa, xb = self.RTERM_XI[rname]
            for combo in (combos_sym if rname in self.RTERMS_SYM else combos_all):
                (a, b), (c, d) = combo
                out_keys.append((rname, combo))
                # The prefactor 2/(pi A) counts both Wick contractions, so two distinct ones
                # contribute 0.5 each and a degenerate one 1
                wick = list(dict.fromkeys([(idx[xa, a, c], idx[xb, b, d]),
                                           (idx[xa, a, d], idx[xb, b, c])]))
                rows += [(ia, ib, wid, len(out_keys) - 1, 1./len(wick)) for ia, ib in wick]
        xi_a, xi_b, w_ids, out_idx, wick_fac = zip(*rows)
        return dict(xi_a_idx=np.array(xi_a, dtype=np.int32),
                    xi_b_idx=np.array(xi_b, dtype=np.int32), w_ids=np.array(w_ids, dtype=np.int32),
                    out_idx=np.array(out_idx, dtype=np.int32),
                    wick_fac=np.array(wick_fac, dtype=np.float64), xi_table_keys=xi_keys,
                    out_keys=out_keys, n_out=len(out_keys))

    ##########################
    # Continuous expressions #
    ##########################

    def _qterms_continuous_tomo_c(self, verbose=False, nr_target=1024, nphi_target=1024):
        """Mixed terms of all tomographic combos in the continuum limit, averaged over nr_target fine
        bins, stored in ``qpp_tomo``, ``qmm_tomo`` and ``qpm_tomo``."""
        n = self.n_r
        combos_sym = self._tomo_combos()
        xi_keys, xi_key_to_idx = self._xi_keys()
        if verbose: print("  continuous q-terms (C/OpenMP) ...")

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

        # xi tabulated up to the largest separation of the r-terms
        q_raw = self._qterms_continuous_raw(
            [self._xi(pm, i, j) for pm, i, j in xi_keys], sorted(q_calls, key=q_calls.get),
            self._xi_query_hi(), 1., 1., nr_target, nphi_target, verbose)
        q_results = np.zeros((len(q_out_keys), n, n))
        for out_idx, slot, fac in q_scatter:
            q_results[out_idx] += fac * q_raw[slot]

        self.qpp_tomo = {}
        self.qmm_tomo = {}
        self.qpm_tomo = {}
        for oi, (qname, combo) in enumerate(q_out_keys):
            block = q_results[oi]
            sym = qname in self.QTERMS_SYM
            if sym and combo[0] == combo[1]:
                block = .5 * (block+block.T)
            getattr(self, qname + '_tomo')[combo] = block

    def _rterms_continuous_tomo_c(self, verbose=False, dlnt_target=.05):
        """r-terms of all tomographic combos in the continuum limit, averaged over log sub-bins no
        wider than dlnt_target, as ``{(rterm, ((a,b),(c,d))): (n_r, n_r)}``."""
        combos_all = [(tp1, tp2) for tp1 in self.tomopairs_list for tp2 in self.tomopairs_list]
        xi_keys, xi_key_to_idx = self._xi_keys()
        if verbose: print("  continuous r-terms (C/OpenMP) ...")

        ## Build the r-term work list ##
        # Five consecutive output blocks per ordered combo, as rpm needs all of them; the
        # redundant symmetric blocks are dropped afterwards
        r_out_keys = [(rname, combo) for combo in combos_all for rname in self.RTERMS]
        # Both Wick pairings are always summed, also degenerate ones, and the result is halved,
        # since the prefactor 2/(pi A) already counts both contractions
        work = [(xi_key_to_idx[('p', *wa)], xi_key_to_idx[('p', *wb)], xi_key_to_idx[('m', *wa)],
                 xi_key_to_idx[('m', *wb)], ic * len(self.RTERMS))
                for ic, ((a, b), (c, d)) in enumerate(combos_all)
                for wa, wb in [((a, c), (b, d)), ((a, d), (b, c))]]
        r_results = .5 * self._rterms_continuous_raw(
            self._build_xi_tables(xi_keys), list(zip(*work)), len(r_out_keys), dlnt_target, verbose)
        rterms = {}
        for oi, (rname, combo) in enumerate(r_out_keys):
            if rname in self.RTERMS_SYM and combo[0] > combo[1]:
                continue
            block = r_results[oi]
            if rname in self.RTERMS_SYM and combo[0] == combo[1]:
                block = .5  * (block+block.T)
            rterms[(rname, combo)] = block

        return rterms

    ###################
    # General helpers #
    ###################

    def _xi_keys(self):
        """Keys (pm, i, j) with i <= j of the tabulated xi_pm, and their index for either order of
        the bins."""
        keys = sorted({(pm, i, j) for pm in ('p', 'm') for i in range(self.ntomo) for j in range(i, self.ntomo)})
        index = {key: k for k, key in enumerate(keys)}
        for pm, i, j in keys:
            index[(pm, j, i)] = index[(pm, i, j)]
        return keys, index

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
        return [(tp1, tp2) for i, tp1 in enumerate(self.tomopairs_list)
                for tp2 in self.tomopairs_list[i:]]