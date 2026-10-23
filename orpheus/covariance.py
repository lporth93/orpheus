"""
Implementation of the GGCovarianceNoTomo class.

Computes the (non-tomographic) covariance of the second-order shear
correlation functions xi_+, xi_- in the framework of:

  - Schneider, van Waerbeke, Kilbinger & Mellier 2002, A&A 396, 1
    ("S+02"; https://arxiv.org/abs/astro-ph/0206182), Section 4.
  - Larma 2025, MSc thesis ("L25"), Sections 3.2 and 4.3.

Three sub-covariances are produced:

    Cov_pp(t1, t2) ≡ Cov(hat xi_+, t1; hat xi_+, t2) L25 Eq. 3.7
    Cov_mm(t1, t2) ≡ Cov(hat xi_-, t1; hat xi_-, t2) L25 Eq. 3.8
    Cov_pm(t1, t2) ≡ Cov(hat xi_+, t1; hat xi_-, t2) L25 Eq. 3.9

Each is split into:
    pair (shape-noise) -- diagonal only [for Cov_pp, Cov_mm]
    triplet (mixed)
    quadruplet (cosmic variance) -- in density form, see L25 sect. 4.3.1

The `method_*` switches choose between the analytic (Poisson, ideal-counts)
prediction and the discrete (mock-based) realisation, independently for
each multiplet. Mock counts are obtained from orpheus via
``computeDiscrete``.

Quadruplet density convention (matching orpheus' 4PCF binning):

    q_array[i1, i2, i3, k12, k13]
        == q( theta1=r[i1], theta2=r[i2], theta3=r[i3],
               alpha12=phi[k12], alpha13=phi[k13] )

where the apex (= shared galaxy) sits at the origin and (theta1, theta2,
theta3) are the three spoke lengths from the apex, with (alpha12, alpha13)
the two opening angles relative to the first spoke (gauge: first spoke
along +x).

Binning conventions
-------------------
* ``theta_centers`` / ``theta_edges`` : the 2PCF radial bins. These define
  the size of the output covariance matrix and are used for the pair counts
  and the quadruplet integrator. NOT used as the q-tensor grid (see below).

* ``phi_centers`` / ``phi_edges`` : angular bins for the triplet and
  quadruplet tensors. Passed independently because ξ_± carry no angular
  bin information.

* ``q_theta_centers`` / ``q_theta_edges`` : radial grid on which the
  quadruplet density q is sampled (orpheus 4PCF binning). This typically
  extends well beyond the ξ_± range, e.g. from ~0.25' to the survey
  diameter. Defaults to the 2PCF grid for backward compatibility, but
  should always be set explicitly when using discrete q tensors.

* Triplet oversampling (``triplet_oversample`` = α ≥ 1): each coarse
  2PCF radial bin is subdivided into α equal sub-bins for the triplet
  computation. Both the analytic triplet tensor and the discrete triplet
  tensor (loaded/measured from orpheus) live on the fine grid
  (n_r * α, n_r * α, n_phi). Sub-bin results are summed back to the
  coarse (n_r, n_r) covariance grid before dividing by Np².

All angles in radians; all separations in arcmin.
"""

import ctypes
import glob
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter, uniform_filter

from .utils import check_clib_error

__all__ = ["GGCovarianceNoTomo", "GGCovariance"]


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

def _periodic_weights(x, period=2. * np.pi):
    """Uniform weights for periodic function sampled on a full-period grid."""
    x = np.asarray(x, dtype=float)
    return np.full_like(x, period / len(x))

def _triplet_weight(weight_type, PHI, jk_x, jk_y):
    """Angular weight of the mixed (triplet) terms.

    ``weight_type``: 0 = q_++ (w = 1), 1 = q_-- (cos 4φ),
    2 = q_+- (cos 4(φ − φ_jk)). Mirrors the ``switch (weight_type)`` of the
    ``triplet_term_reduce`` C kernel, which serves the unmasked path; this is
    the banded (``pair_mask``) path's copy.
    """
    if weight_type == 0:
        return np.ones_like(PHI)
    if weight_type == 1:
        return np.cos(4. * PHI)
    return np.cos(4. * (PHI - np.arctan2(jk_y, jk_x)))


# Share of L2 the quadruplet working set may claim. Calibrated on a 1280 KB
# L2: the measured optimum (v2_block=16 at n_v1=1024, ntomo=4) sits at 52%,
# and 92% is where the next step up starts spilling, so the midpoint is safe.
_L2_TARGET_FRAC = .7


def _l2_cache_bytes():
    """Per-core L2 size in bytes, or None if it cannot be determined.

    ``ORPHEUS_L2_CACHE_BYTES`` overrides the probe; setting it to 0 disables
    the cache-aware tile cap altogether. Because the cap feeds ``v2_block``,
    which sets the varphi2 chunking and hence the summation order, results
    agree between machines only to rounding unless it is pinned.
    """
    env = os.environ.get('ORPHEUS_L2_CACHE_BYTES')
    if env is not None:
        return int(env) or None
    for d in sorted(glob.glob('/sys/devices/system/cpu/cpu0/cache/index*')):
        try:
            with open(os.path.join(d, 'level')) as f:
                if f.read().strip() != '2':
                    continue
            with open(os.path.join(d, 'size')) as f:
                size = f.read().strip()
        except OSError:
            continue
        mult = {'K': 1024, 'M': 1024**2}.get(size[-1].upper(), 1)
        return int(size.rstrip('KMkm')) * mult
    try:
        out = subprocess.run(['sysctl', '-n', 'hw.l2cachesize'],
                             capture_output=True, text=True, timeout=1)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _v2_block_cache_cap(n_v1, n_xi, frac=_L2_TARGET_FRAC):
    """Largest ``v2_block`` whose quadruplet working set still fits in L2.

    The callback loop of ``quadruplet_sva_integrate`` walks one
    ``(theta1, theta2)`` pair at a time and reads four ``n_v1 x v2_block``
    tables over and over -- the q slice, the two weight tables for the
    ``cos(4(va-vb))`` and ``cos(4(v1+v2-va-vb))`` terms, and ``cos(4(v1-v2))``
    -- plus ``n_xi`` rows of length ``n_v1`` that do not scale with
    ``v2_block``. Spilling that set to L3 is expensive: on a 1.25 MB L2 the
    2 MB slice ``_auto_reduce_tiles`` used to pick measured 2.3x slower than
    a 128 KB one at equal arithmetic.

    Returns None when the cache size is unknown. The result is rounded down
    to a multiple of 8 and floored there: the innermost loop is vectorised
    over ``v2_block``, so very short chunks lose more to loop overhead than
    they gain in locality.
    """
    l2 = _l2_cache_bytes()
    if not l2 or not n_v1:
        return None
    raw = int((frac*l2 - 8.*n_xi*n_v1)//(32.*n_v1))
    return max(8, 8*(raw//8))


def _auto_reduce_tiles(nr_block_base, v2_block_base, n_v2, n_v1=None, n_xi=2):
    """Return ``(nr_block, v2_block)`` after the banded-integrator ÷f reduction.

    Reduces ``nr_block`` by a factor ``f`` (target ``f=4``) so the tiles hug
    the slab band, and scales ``v2_block`` by ``f²`` to spend the memory that
    frees. ``f`` is first capped by how much ``v2_block`` can grow before
    hitting ``n_v2`` (the total number of φ₂ integration points).

    ``v2_block`` is then capped again by :func:`_v2_block_cache_cap`: it sets
    the quadruplet working set, and growing it past L2 costs far more than the
    extra memory buys. ``nr_block`` is left alone -- at fixed ``v2_block`` the
    tile size measures flat (44.0/43.8/46.3 s for nr_block 4/8/16), so it is
    free to keep serving the band. Pass ``n_v1``/``n_xi`` to enable the cap;
    without them the historical ÷f result is returned unchanged.

    Shared by :meth:`GGCovarianceNoTomo._recompute_slab` and
    :meth:`GGCovarianceNoTomo._subbin_dry_run`, which must agree on the
    tile size the banded quadruplet integrator will actually use.
    """
    f_max = max(1, n_v2 // v2_block_base) # largest f before v2 cap
    f = min(4, f_max) # target ÷4, bounded
    nrb = max(4, nr_block_base // f)
    f_act = max(1, nr_block_base // nrb) # actual factor (int div)
    v2b = min(v2_block_base * f_act * f_act, n_v2)
    cap = _v2_block_cache_cap(n_v1, n_xi)
    if cap is not None:
        v2b = max(v2_block_base, min(v2b, cap))
    return nrb, v2b


def _build_q_func(q_array, r_centers, phi_centers,
                  method='linear'):
    """Wrap a 5-D quadruplet-density array into a callable q(t1,t2,t3,a12,a13)."""
    # Interpolation: linear inside the radial grid; nearest-neighbour below r_min
    # (so compact configurations don't silently evaluate to zero); hard zero above
    # r_max. Both interpolators alias the q_array buffer (no copy).
    grid = (r_centers, r_centers, r_centers, phi_centers, phi_centers)
    lin_interp = RegularGridInterpolator(
        grid, q_array, method=method, bounds_error=False, fill_value=None)
    nn_interp = RegularGridInterpolator(
        grid, q_array, method='nearest', bounds_error=False, fill_value=None)

    # Re-point at q_array if scipy copied it (keeps shared-memory aliasing).
    if not np.shares_memory(lin_interp.values, q_array):
        lin_interp.values = q_array
    if not np.shares_memory(nn_interp.values, q_array):
        nn_interp.values = q_array

    twopi = 2. * np.pi
    r_min = r_centers[0]
    r_max = r_centers[-1]

    def q_func(t1, t2, t3, a12, a13):
        a12 = np.mod(a12, twopi)
        a13 = np.mod(a13, twopi)
        t1, t2, t3, a12, a13 = np.broadcast_arrays(t1, t2, t3, a12, a13)
        pts = np.stack((t1, t2, t3, a12, a13), axis=-1)
        lin = lin_interp(pts)
        nn = nn_interp(pts)
        r_stack = np.stack((t1, t2, t3), axis=-1)
        below = np.any(r_stack < r_min, axis=-1)
        above = np.any(r_stack > r_max, axis=-1)
        out = lin
        out = np.where(below, nn, out)
        out = np.where(above, 0., out)
        return out

    return q_func


_CLIB = None


def _clib():
    """Load ``orpheus_clib`` and bind the covariance kernel signatures once.

    Module-level (not a method) so the binding is shared by every instance.
    The covariance kernels ship with this module (``MANIFEST.in`` grafts both
    ``covariance.py`` and ``orpheus/src/cov_postq.c``), so a missing library
    is a broken build and raises rather than selecting a slower path.
    """
    global _CLIB
    if _CLIB is not None:
        return _CLIB
    from numpy.ctypeslib import ndpointer
    ci = ctypes.c_int
    cd = ctypes.c_double
    p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
    p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
    library_path = str(Path(__import__('orpheus').__file__).parent.absolute())
    lib = ctypes.CDLL(glob.glob(library_path + "/orpheus_clib*.so")[0])

    lib.interp4d.restype = None
    lib.interp4d.argtypes = [
        p_f64, p_f64, p_f64, # q_slice, r_centers, phi_centers
        ci, ci, # nr, nphi
        cd, cd, cd, cd, # r_min, r_max, logr0, inv_dlogr
        cd, cd, # phi0, inv_dphi
        p_f64, p_f64, p_f64, p_f64, # t2f, t3f, a12f, a13f
        ci, p_f64] # n, out

    lib.quadruplet_sva_integrate.restype = None
    lib.quadruplet_sva_integrate.argtypes = [
        p_f64, p_f64, ci, ci, # phi_grid, phi_w, n_phi, n_r
        p_f64, p_f64, ci, # phi1_nodes, phi1_weights, ns1
        p_f64, p_f64, ci, # phi2_nodes, phi2_weights, ns2
        p_f64, p_f64, ci, # v1_grid, v1_w, nv1
        p_f64, p_f64, ci, # v2_grid, v2_w, nv2
        ci, ci, p_i32, ci, # nr_block, v2_block, tiles, n_tiles
        ci, p_f64, # q_mode, q_array
        p_f64, p_f64, ci, ci, # q_r_centers, q_phi_centers, q_nr, q_nphi
        cd, p_f64, p_f64, p_f64, # q_const, sup_t1, sup_t2, sup_t3
        p_f64, ci, cd, cd, # xi_x, xi_n, xi_logx0, xi_inv_dlogx
        p_f64, ci, # xi_y_tables, n_xi_unique
        ci, p_i32, p_i32, # n_cb, xi_a_idx, xi_b_idx
        p_i32, p_i32, p_f64, # w_ids, out_idx, wick_fac
        ci, p_i32, p_i32, # n_out, symmetric_mask, pair_mask
        p_f64, ci, ci] # results, nthreads, verbose

    lib.continuous_q_single.restype = None
    lib.continuous_q_single.argtypes = [
        ci, ci, # n_r, alpha
        p_f64, p_f64, # fine_centers, fine_weights
        p_f64, p_f64, # phi_centers, dphi
        ci, ci, # n_fine, n_phi
        p_f64, ci, cd, cd, # xi_x, xi_n, xi_logx0, xi_inv_dlogx
        p_f64, # xi_y
        ci, # w_type
        cd, cd, cd, # sigma_eps2, area, nbar
        p_f64, p_f64] # Np, out

    lib.continuous_q_tomo.restype = None
    lib.continuous_q_tomo.argtypes = [
        ci, ci, # n_r, alpha
        p_f64, p_f64, # fine_centers, fine_weights
        p_f64, p_f64, # phi_centers, dphi
        ci, ci, # n_fine, n_phi
        p_f64, ci, cd, cd, # xi_x, xi_n, xi_logx0, xi_inv_dlogx
        p_f64, ci, # xi_y_tables, n_xi_unique
        cd, cd, cd, # sigma_eps2, area, nbar
        p_f64, # Np
        ci, # n_calls
        p_i32, p_i32, p_i32, # call_xi_idx, call_w_type, call_out_idx
        p_f64, # call_factor
        ci, p_f64, # n_out, results
        ci, ci] # nthreads, verbose

    lib.continuous_rterms_tomo.restype = None
    lib.continuous_rterms_tomo.argtypes = [
        ci, p_f64, # n_r, theta_centers
        p_f64, p_f64, ci, # phi_grid, phi_w, n_phi
        p_f64, p_f64, ci, # v_grid, v_w, n_v
        p_f64, p_f64, ci, # v_half, v_half_w, n_vh
        cd, # area
        p_f64, ci, cd, cd, # xi_x, xi_n, xi_logx0, xi_inv_dlogx
        p_f64, ci, # xi_y_tables, n_xi_unique
        ci, # n_work
        p_i32, p_i32, p_i32, p_i32, # work_xi_a_p/b_p/a_m/b_m
        p_i32, # work_out_idx
        ci, p_f64, # n_out, results
        ci, ci] # nthreads, verbose

    lib.triplet_term_reduce.restype = None
    lib.triplet_term_reduce.argtypes = [
        p_f64, ci, ci, # Nt, nr, nphi
        p_f64, p_f64, # tc, phic
        ci, # weight_type
        p_f64, ci, cd, cd, # xi_x, xi_n, xi_logx0, xi_inv_dlogx
        p_f64, # xi_y
        p_f64, ci, ci] # out, nthreads, verbose

    _CLIB = lib
    return _CLIB


# Peak memory of the quadruplet integral, the only memory-heavy stage:
#    PEAK ~ q_tensor + n_workers * (m^2 * n_v1 * v2_block + 2 * n_out * n_r^2
#                                   + q_slice) * 8B
# with m = nr_block * len(phi1_subgrid_frac). The q tensor is read-only and
# shared by all OpenMP threads; the first per-thread term (the q tile) usually
# dominates. nr_block is its strong (quadratic) knob, v2_block the linear one,
# and both leave results bit-identical.


################
# Main classes #
################

class GGCovarianceNoTomo:
    """Computes covariance of second-order shear correlation functions.

    Parameters
    ----------
    theta_centers : (n_r,) array_like
        Centers of the ξ_± radial bins, in arcmin.
    theta_edges : (n_r + 1,) array_like
        Edges of the ξ_± radial bins, in arcmin.
    phi_centers : (n_phi,) array_like, optional
        Centers of the azimuthal bins for triplet / quadruplet counts,
        spanning [0, 2π). Defaults to 128 uniform bins.
    phi_edges : (n_phi + 1,) array_like, optional
        Edges of the azimuthal bins. Inferred from ``phi_centers`` if absent.
    q_theta_centers : (n_qr,) array_like, optional
        Centers of the radial bins on which the quadruplet density tensor
        q lives. Typically the orpheus 4PCF binning, which may extend well
        beyond the ξ_± range. Defaults to ``theta_centers``.
    q_theta_edges : (n_qr + 1,) array_like, optional
        Edges of the q radial bins. Defaults to ``theta_edges``.
    q_phi_centers : (n_qphi,) array_like, optional
        Centers of the azimuthal bins for the q tensor. Defaults to
        ``phi_centers``.
    q_phi_edges : (n_qphi + 1,) array_like, optional
        Edges of the q azimuthal bins. Defaults to ``phi_edges``.
    triplet_oversample : int, optional
        Oversampling factor alpha ≥ 1 for the triplet radial bins. Each
        coarse ξ_± bin is subdivided into alpha equal sub-bins when evaluating
        the triplet integral. Triplet tensors (analytic or discrete) must
        live on the resulting fine grid of shape
        (n_r * alpha, n_r * alpha, n_phi). Default: 1 (no oversampling).
    method_pairs, method_triplets, method_quadruplets : str
        ``'analytic'``, ``'discrete'`` or ``'continuous'``
        (``method_quadruplets`` additionally accepts ``'frompairs'``).

        * ``'discrete'`` -- count tensors are measured (or loaded) and the
          mixed/cosmic-variance terms are obtained by binning the discrete
          multiplet sums (:meth:`getD_discrete`, :meth:`_triplet_term`,
          :meth:`get_all_rterms`).
        * ``'analytic'`` -- idealised Poisson count tensors
          (:meth:`pairs_analytic`, :meth:`triplets_analytic`,
          :meth:`quadruplets_analytic`) are substituted into the same
          discrete machinery.
        * ``'continuous'`` -- bypasses count tensors and evaluates the
          continuum (n_r -> infinity) ensemble-averaged expressions by direct
          integration over ``spline_xip`` / ``spline_xim``
          (:meth:`getD_continuous`, :meth:`getqpp_continuous`,
          :meth:`getrp0_continuous`, ...). ``triplets``/``quadruplets`` stay
          ``None``; ``pairs`` is still populated via :meth:`pairs_analytic`
          to normalise the shape-noise term.
        * ``'frompairs'`` (``method_quadruplets`` only) -- "pair-informed"
          reconstruction via :meth:`quadruplets_analytic`'s suppressed branch
          (n̄⁴⟨w⟩⁴ · ``spl_suppression_ww``, factorised across the three
          quadruplet separations). ``spl_suppression_ww`` is then mandatory,
          and is expected to be calibrated against the measured pair-count
          suppression ŝ(θ) = N_p^measured(θ) / N_p^ideal(θ) as a cheap proxy
          for the boundary/masking suppression. A ``RuntimeWarning`` is
          raised if the spline's tabulated domain does not cover
          ``q_theta_centers``.
    spline_xip, spline_xim : callable
        ξ_+(θ), ξ_-(θ) of the input cosmology.
    nbar : float
        Mean galaxy number density, arcmin^{-2}.
    area : float
        Survey area, arcmin^2.
    sigma_eps : float
        Per-component intrinsic ellipticity dispersion.
    weights_mean, weights_std : float
        Mean and standard deviation of the per-galaxy weights w_i (1 and 0
        for unit weights). ⟨w²⟩ = weights_mean² + weights_std² is exact for
        any weight distribution. ``weights_std`` is the standard deviation,
        not the variance.
    spl_suppression_ww : callable or None
        Optional s(θ) modulating the analytic Poisson counts. None = 1.
    pairs, triplets, quadruplets : ndarray, str, or None
        Pre-computed count tensors, file paths, or None. When a method is
        ``'discrete'`` and the corresponding tensor is None, it is measured
        from ``catalog`` via :meth:`computeDiscrete`.
    catalog : dict or None
        Inputs for the catalog-driven discrete measurement (used only for the
        ``'discrete'`` methods whose tensor is None). Keys:
        ``ra``, ``dec`` (deg, required), ``weight`` (optional, default unit
        weights), ``path_save`` (required directory for cached files),
        ``rmin_pixsize`` (default 40), ``tree_resos`` (default
        ``[0, 0.5, 1, 2, 4]``), ``npatches`` and ``nmaxs`` (required for the
        4PCF), plus optional ``method`` (4PCF tree method, default
        ``'DoubleTree'``), ``shuffle_pix`` (cell positions of the tree,
        default 1, i.e. random within the cell), ``verbosity`` and
        ``memory_bound``. The
        measurement binning is derived from the covariance grids
        (``theta_edges`` for pairs, the oversampled triplet θ grid +
        ``phi_centers`` for triplets, ``q_theta`` / ``q_phi`` for
        quadruplets), so those θ grids must be log-uniform (geomspace). Patch
        parallelism reuses ``n_workers``.
    store_multiplets : bool, optional
        File retention after each measured tensor is loaded. ``False``
        (default) deletes all cached multiplet files; ``True`` keeps the
        combined pair/triplet/quadruplet tensors but deletes the
        per-footprint 4PCF patch files.
    v2_block : int, optional
        Block size for the varphi2 accumulation inside the parallel
        quadruplet worker; bounds per-worker scratch memory (see module
        docstring). Default 16; valid range 1 <= v2_block <= len(varphi2_grid).
    nr_block : int or None, optional
        Tile size for the (θ₁, θ₂) axes inside the quadruplet integrator;
        shrinks per-worker scratch quadratically with bit-identical results
        (see module docstring). ``None`` (default) = whole grid in one tile.

    Notes
    -----
    All angles in radians; all separations in arcmin.

    Expected count-tensor shapes::

        pairs : (n_r,)
        triplets : (n_r * triplet_oversample,
                    n_r * triplet_oversample, n_phi)
        quadruplets : (n_qr, n_qr, n_qr, n_qphi, n_qphi) [density q]
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
        v2_block=16, nr_block=None, n_workers=1):

        ## 2PCF radial bins
        self.theta_centers = np.asarray(theta_centers, dtype=float)
        self.theta_edges = np.asarray(theta_edges, dtype=float)
        if len(self.theta_edges) != len(self.theta_centers) + 1:
            raise ValueError("theta_edges must have exactly one more entry than theta_centers")
        self.dtheta = np.diff(self.theta_edges)
        self.n_r = len(self.theta_centers)

        ## Angular bins (shared between triplets and quadruplets)
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

        ## Quadruplet radial / angular grid
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

        ## Triplet oversampling
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

        # ⟨w⟩ and ⟨w²⟩ = ⟨w⟩² + Var(w) = weights_mean² + weights_std²
        self._w1 = self.weights_mean
        self._w2 = self.weights_mean**2 + self.weights_std**2

        ## Integration grids (analytic branch + quadruplet density integral)
        self.phi_grid = (np.geomspace(0.05, 1e4, 128)
                         if phi_grid is None else np.asarray(phi_grid))
        self.varphi1_grid = (np.linspace(0., 2. * np.pi, 64, endpoint=False)
                             if varphi1_grid is None else np.asarray(varphi1_grid))
        self.varphi2_grid = (np.linspace(0., 2. * np.pi, 64, endpoint=False)
                             if varphi2_grid is None else np.asarray(varphi2_grid))
        self.phi1_subgrid_frac = phi1_subgrid_frac
        self.phi2_subgrid_frac = phi2_subgrid_frac

        ## Parallel-worker memory control
        self.v2_block = int(v2_block)
        if self.v2_block < 1:
            raise ValueError("v2_block must be >= 1")
        # Tile size for the (theta1, theta2) radial axes inside the quadruplet integrator:
        #  * None --> whole grid in one tile
        #  * Smaller values reduce per-worker scratch quadratically, at the
        #    cost of more serial sub-tiles per phi node.
        self.nr_block = None if nr_block is None else int(nr_block)
        if self.nr_block is not None and self.nr_block < 1:
            raise ValueError("nr_block must be >= 1 (or None for no tiling)")
        # Default worker count for the quadruplet (5-D) integrals.
        self.n_workers = int(n_workers)
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")

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

        ## Catalog-level quantites to store for discrete multiplet computation
        # dict of the orpheus-specific inputs that cannot be inferred from the
        # covariance grids.
        self.catalog = catalog
        self.store_multiplets = bool(store_multiplets)
        # kind ('spin'/'scalar') -> built orpheus catalogue, populate lazy
        # and reuse across the pair/triplet/quadruplet measurements.
        self._cat_cache = {}
        if self.catalog is not None:
            # Orpheus bins radially in log θ, so the covariance should do the same
            self._check_loguniform(self.theta_edges, "theta_edges")
            if (self.method_quadruplets == 'discrete'
                    and not callable(quadruplets) and quadruplets is None):
                self._check_loguniform(self.q_theta_edges, "q_theta_edges")

        ## Multiplet count initialisation
        self.pairs = self._init_multiplet('pairs', pairs, self.method_pairs)
        self.triplets = self._init_multiplet(
            'triplets', triplets, self.method_triplets)
        self.quadruplets = self._init_multiplet(
            'quadruplets', quadruplets, self.method_quadruplets)

        # Build callable wrapper around the quadruplet density. The
        # continuum limit needs no q tensor at all -- _q_func stays unused.
        if self.method_quadruplets == 'continuous':
            self._q_func = None
        elif callable(self.quadruplets):
            self._q_func = self.quadruplets
        else:
            # Use the q-specific grid, NOT the 2PCF grid.
            self._q_func = _build_q_func(
                self.quadruplets, self.q_theta_centers, self.q_phi_centers)

    # Per-multiplet: how a non-'discrete' method_* supplies its tensor.
    # 'analytic'/'frompairs' build the idealised Poisson tensor; 'continuous'
    # needs no tensor at all, except for pairs, where an ideal Np still
    # normalises D_continuous.
    _MULTIPLET_SOURCES = {
        'pairs': {'analytic': 'pairs_analytic', 'continuous': 'pairs_analytic'},
        'triplets': {'analytic': 'triplets_analytic', 'continuous': None},
        'quadruplets': {'analytic': 'quadruplets_analytic',
                        'frompairs': 'quadruplets_analytic',
                        'continuous': None},
    }

    @property
    def _clib(self):
        """The bound ``orpheus_clib`` handle.

        A property rather than an attribute set in ``__init__``: instances also
        come from ``_from_components``, which builds them with
        ``object.__new__``, and keeping the ctypes handle out of ``__dict__``
        leaves these objects picklable and deep-copyable.
        """
        return _clib()

    def _init_multiplet(self, kind, value, method):
        """Resolve one multiplet count tensor at construction time.

        A ``'discrete'`` method loads a path, measures from the catalogue when
        nothing was given, or validates the supplied tensor. Any other method
        substitutes its own tensor (see :attr:`_MULTIPLET_SOURCES`) and warns
        that a supplied one is ignored.
        """
        sources = self._MULTIPLET_SOURCES[kind]
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
        """Diagonal shape-noise term. L25 Eq. (3.12) / S+02 Eq. (27).

            D(θ) = σ_ε⁴ ⟨w²⟩² / (N_p(θ) ⟨w⟩²)

        Same expression as :meth:`getD_discrete` (with ``self.pairs`` here
        being the idealised :meth:`pairs_analytic` tensor) -- both must
        agree since ``method_pairs='analytic'`` also uses that same tensor
        but is dispatched to :meth:`getD_discrete` by :meth:`getD`.
        """
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        return self.sigma_eps**4 * ww_factor / self.pairs

    def _triplet_term_continuous_c(self, w_type, xi_func,
                                   nr_target=1024, nphi_target=1024):
        """Continuum-limit q_++/q_--/q_+- via the C kernel.

        The Np -> ideal, Nt -> ideal limit of :meth:`_triplet_term`: evaluates
        ``<weight(phi) * xi(r_jk)>`` on a fine phi grid and fine theta sub-bins
        (oversampling each coarse bin by ``alpha = nr_target // n_r``) and
        bin-averages back to the coarse (n_r, n_r) grid -- L25 Eq. 3.13 /
        S+02 Eq. 32 with ideal (Poisson) counts in the continuum limit.

        ``w_type``: 0 = q_++ (w=1), 1 = q_-- (cos 4φ), 2 = q_+- (cos 4(φ−φ_jk)),
        matching :meth:`_triplet_phi_reduce`. The kernel keeps the (α·n_r)²·n_phi
        integrand out of memory; materialised it would reach ~64 GB at the
        default 1024×1024 grids.
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
        """Continuous q_++. L25 Eq. 3.13 / S+02 Eq. 32."""
        return self._triplet_term_continuous_c(0, self.xip, nr_target, nphi_target)

    def getqmm_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_--. L25 Eq. 3.18 / S+02 Eq. 36."""
        return self._triplet_term_continuous_c(1, self.xip, nr_target, nphi_target)

    def getqpm_continuous(self, nr_target=1024, nphi_target=1024):
        """Continuous q_+-. L25 Eq. 3.22 / S+02 Eq. 38."""
        return self._triplet_term_continuous_c(2, self.xim, nr_target, nphi_target)

    def getrp0_continuous(self):
        """Continuous r_+0. L25 Eq. (3.14) / S+02 Eq. (34)."""
        return self._rterms_continuous_c()[0]

    def getrp1_continuous(self):
        """Continuous r_+1. L25 Eq. (3.15) / S+02 Eq. (34)."""
        return self._rterms_continuous_c()[1]

    def getrm0_continuous(self):
        """Continuous r_-0. L25 Eq. (3.19)."""
        return self._rterms_continuous_c()[2]

    def getrm1_continuous(self):
        """Continuous r_-1. L25 Eq. (3.20)."""
        return self._rterms_continuous_c()[3]

    def getrpm_continuous(self):
        """Continuous r_+-.

        NOT symmetric under theta1 <-> theta2 (xip != xim in general),
        consistent with the asymmetry of Cov_pm.
        """
        return self._rterms_continuous_c()[4]

    ####################################
    # Analytic (Poisson) count tensors #
    ####################################

    def _suppression(self, r):
        """Optional configuration-space modulation s(r), default 1.0."""
        if self.spl_suppression_ww is None:
            return np.ones_like(np.asarray(r, dtype=float))
        return self.spl_suppression_ww(r)

    def pairs_analytic(self):
        """Idealised pair counts. L25 Eq. (4.7).

            N_p^ideal(θ) = 2π A n̄² ⟨w⟩² θ dθ s(θ)
        """
        if self.area is None or self.nbar is None:
            raise ValueError("`nbar` and `area` are required for analytic pairs")
        return (2. * np.pi * self.area * self.nbar**2
                * self._w1**2
                * self.theta_centers * self.dtheta
                * self._suppression(self.theta_centers))

    @staticmethod
    def suppression_spline_from_pairs(Np_meas, theta_centers, theta_edges,
                                       area, nbar, weights_mean=1.):
        """Empirical pair-count suppression ŝ(θ) = N_p(θ) / N_p^ideal(θ).

            ŝ(θ) = N_p^measured(θ) / [2π A n̄² ⟨w⟩² θ dθ]

        Builds a callable spline from *measured* pair counts ``Np_meas``
        (tabulated on ``theta_centers``/``theta_edges``) suitable for use
        as ``spl_suppression_ww`` -- in particular for
        ``method_quadruplets='frompairs'`` (see :meth:`quadruplets_analytic`).

        A ``@staticmethod`` so it can be called *before* constructing a
        ``method_quadruplets='frompairs'`` instance, which requires
        ``spl_suppression_ww`` to already be set at ``__init__`` time.
        Typically called with the measured pairs/grid of a sibling
        ``method_pairs='discrete'`` instance.

        Interpolated/extrapolated linearly in log θ. The returned callable
        exposes a ``.x`` attribute with the (linear-θ) knot positions, used
        by the bounds check in :meth:`quadruplets_analytic`.
        """
        theta_centers = np.asarray(theta_centers, dtype=float)
        theta_edges = np.asarray(theta_edges, dtype=float)
        ideal = (2. * np.pi * area * nbar**2 * weights_mean**2
                * theta_centers * np.diff(theta_edges))
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
        """Idealised triplet counts on the fine (oversampled) grid. L25 Eq. (4.11).

            N_t^ideal(t1, t2, φ12) = 2π A n̄³ ⟨w⟩³ t1 dt1 t2 dt2 dφ12 s(t1)s(t2)s(r_jk)

        The tensor lives on the triplet fine grid
        (triplet_n_r, triplet_n_r, n_phi).
        """
        if self.area is None or self.nbar is None:
            raise ValueError("`nbar` and `area` are required for analytic triplets")
        T1, T2, PHI = np.meshgrid(self.triplet_theta_centers,
                                   self.triplet_theta_centers,
                                   self.phi_centers, indexing='ij')
        DT1, DT2, DPHI = np.meshgrid(self.triplet_dtheta,
                                      self.triplet_dtheta,
                                      self.dphi, indexing='ij')
        r_jk = np.sqrt(T1**2 + T2**2 - 2. * T1 * T2 * np.cos(PHI))
        s_factor = (self._suppression(T1) * self._suppression(T2))
        return (2. * np.pi * self.area * self.nbar**3
                * self._w1**3
                * T1 * DT1 * T2 * DT2 * DPHI
                * s_factor)

    def quadruplets_analytic(self):
        """Idealised quadruplet density q. L25 Eqs. (4.16)+(4.18).

        Returns a callable rather than a tensor so that queries beyond the
        radial bin range evaluate to n̄⁴ (not zero).

        For ``method_quadruplets='frompairs'``, the same constant density
        n̄⁴⟨w⟩⁴ is factorised across the three independent quadruplet
        separations and multiplied by ``spl_suppression_ww`` (mandatory in
        this mode; see :meth:`suppression_spline_from_pairs`). A
        ``RuntimeWarning`` is raised if the spline's tabulated domain (its
        ``.x`` attribute) does not cover ``q_theta_centers``, since
        extrapolating a spline fitted to pair-count data risks unphysical
        (e.g. negative) suppression values.
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
        # Recorded for the C quadruplet kernel, which rebuilds the closure's
        # two branches from (q_const, spl_suppression_ww). It must be read
        # here: GGCovariance replaces self.nbar with a per-bin array only
        # after super().__init__() has already built this closure.
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
        """Convert raw quadruplet counts to density q and store.

            V_q = 2π A t1 dt1 t2 dt2 t3 dt3 dφ12 dφ13 (L25 Eq. 4.17)

        Defaults to the q grid (q_theta_centers etc.) if not specified.
        ``weights_std`` must be the standard deviation, not the variance --
        see class docstring.
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
        # V_q is a rank-1 outer product of the five axes, so divide N_q by one
        # axis at a time rather than materialising the 5-D volume.
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
        """Diagonal shape-noise term. Dispatches to :meth:`getD_continuous`
        or :meth:`getD_discrete` based on ``method_pairs``.
        """
        self.D = (self.getD_continuous() if self.method_pairs == 'continuous'
                  else self.getD_discrete())
        return self.D

    def getD_discrete(self):
        """Diagonal shape-noise term. L25 Eq. 3.12 / S+02 Eq. 27.

            D(θ) = σ_ε⁴ ⟨w²⟩² / (N_p(θ) ⟨w⟩²)
        """
        if self.pairs is None:
            raise RuntimeError("`pairs` not initialised.")
        Np = np.asarray(self.pairs)
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        self.D = self.sigma_eps**4 * ww_factor / Np
        return self.D

    ## Triplet (mixed) terms
    def _triplet_phi_reduce(self, Nt, tc, phic, weight_type, xi_func):
        """fine[i,j] = Σ_k Nt[i,j,k]·w(weight_type)·ξ(r_jk) on the fine grid.

        ``weight_type``: 0 = q++ (w=1), 1 = q-- (cos4φ), 2 = q+-
        (cos(4(φ−φ_jk))).
        """
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
            out, int(self.n_workers), 0)
        check_clib_error(self._clib)
        return out

    def _triplet_term(self, weight_type, xi_func, pair_mask=None,
                      sigma_eps_sq=None, nbar_fac=1.):
        """Shared machinery for q_++, q_--, q_+-.

        Evaluates the triplet integrand on the fine (oversampled) grid and
        sums sub-bins back to the coarse (n_r, n_r) grid before dividing
        by Np(θ₁) Np(θ₂).

        For each coarse bin pair (i, j), the sub-bin sum is:

            summed[i, j] = Σ_{a∈sub(i), b∈sub(j), k}
                               N_t[a, b, k] * w(T1, T2, φ, r_jk, ...) * ξ(r_jk)

        where the triplet tensor N_t lives on the fine grid.

        ``weight_type`` : int
            0 = q++, 1 = q--, 2 = q+-. See :func:`_triplet_weight`.
        ``pair_mask`` : (n_r, n_r) bool, optional
            When given, only the matrix elements flagged ``True`` are
            evaluated (rows are tiled and restricted to the column
            bounding-box of the mask, so memory and work stay O(n·bandwidth))
            and the φ-reduction runs in numpy rather than the OpenMP kernel.
            Masked-out elements are returned as ``0``. Used by the
            near-diagonal slab recompute in :meth:`subbin`.
        ``sigma_eps_sq`` : float, optional
            σ_ε² of the shared apex bin; defaults to the instance scalar.
            The tomographic blocks pass the per-bin value.
        ``nbar_fac`` : float, optional
            n̄_ref / n̄_shared scaling for discrete reference-bin multiplets.
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

        ## Banded evaluation: only the coarse (i, j) inside pair_mask
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

    ## Mixed-term (q) getters
    # (name, weight_type, xi attribute) -- the weight ids match the C kernel's
    # switch and :func:`_triplet_weight`.
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
        """Mixed term for Cov_pp. L25 Eq. 3.13 / S+02 Eq. 32.

        ``pair_mask`` only applies to the discrete path.
        """
        return self._get_qterm(0, pair_mask)

    def getqmm(self, pair_mask=None):
        """Mixed term for Cov_mm. L25 Eq. 3.18 / S+02 Eq. 36.

        ``pair_mask`` only applies to the discrete path.
        """
        return self._get_qterm(1, pair_mask)

    def getqpm(self, pair_mask=None):
        """Mixed term for Cov_pm. L25 Eq. 3.22 / S+02 Eq. 38.

        ``pair_mask`` only applies to the discrete path.
        """
        return self._get_qterm(2, pair_mask)

    ## C-accelerated quadruplet integral
    def _rterms_continuous_c(self):
        """All five continuum-limit r-terms in one C/OpenMP call.

        One work unit: the non-tomographic case has a single bin pair, hence a
        single Wick pairing and no 0.5 -- contrast :meth:`get_continuous_tomo_c`,
        which emits both degenerate pairings and halves the result. Table 0 is
        ξ_+, table 1 is ξ_-.

        Weight conventions, with psi_a/psi_b the two side vectors and
        va/vb their bearings:

        * r_-0 carries w = cos(4(V1+V2-va-vb)), which factorises into
          cos(4(V1-va))cos(4(V2-vb)) - sin(4(V1-va))sin(4(V2-vb)). The sine
          integrals vanish by the antisymmetry of sin(4(V-va)) under
          V -> 2pi-V, and V -> V+pi maps psi_b -> psi_a, so the two side
          integrals coincide and the outer product is symmetric.
        * r_-1 carries w = cos(4(V1-V2)); the same two arguments apply with
          cos(4(V+pi)) = cos(4V).
        * r_+- carries w = 2cos(4(V2-vb)) -- only the b side is weighted, and
          the result is NOT symmetric under theta1 <-> theta2.
        """
        fn = self._clib.continuous_rterms_tomo
        # One call yields all five terms, but the getters ask for them one at
        # a time (smooth_rterms does so on a swapped-in fine grid), so the
        # result is memoised against the grids it was computed on. The grids
        # themselves are held in the key, so identity stays meaningful.
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
        results = np.zeros(5 * n * n, dtype=np.float64)
        idx0 = np.zeros(1, dtype=np.int32)
        fn(n, np.ascontiguousarray(self.theta_centers, dtype=np.float64),
           np.ascontiguousarray(self.phi_grid, dtype=np.float64),
           np.ascontiguousarray(_trapz_weights(self.phi_grid), dtype=np.float64),
           len(self.phi_grid),
           v, np.ascontiguousarray(_periodic_weights(v), dtype=np.float64), len(v),
           v_half, np.ascontiguousarray(_trapz_weights(v_half), dtype=np.float64),
           len(v_half),
           float(self.area),
           xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y_tables, 2,
           1, idx0, idx0, idx0 + 1, idx0 + 1, idx0,
           5, results, int(self.n_workers), 0)
        out = results.reshape(5, n, n)
        self._rterms_cont_cache = (self.phi_grid, self.varphi1_grid, n, out)
        return out

    def _xi_query_lo(self):
        """Smallest separation the quadruplet kernels evaluate ξ_± at."""
        return min(self.theta_edges[0], self.phi_grid[0])

    def _xi_query_hi(self):
        """Largest separation the quadruplet kernels evaluate ξ_± at.

        |psi| = |phi ± theta e(V)| is bounded by phi_max + theta_max.
        """
        return self.phi_grid[-1] + self.theta_centers[-1]

    @staticmethod
    def _xi_tables_from_splines(splines, lo=None, hi=None):
        """Tabulate xi splines on their shared log-uniform knots for C.

        Returns ``(xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n)``. Uses
        the splines' own ``.x`` knots when present, so the kernel's
        piecewise-linear lookup reproduces a linear ``interp1d`` exactly;
        for a callable carrying no knots, samples 4096 geomspace points over
        ``[lo, hi]``, the separation range that kernel will query.
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
                          verbose=False, pair_mask=None, n_workers=None):
        """Drive the fused C quadruplet kernel over the phi grid.

        Shared by the non-tomographic and tomographic r-term integrals: the
        only difference between them is the callback map, so both marshal the
        same geometry, tile plan and quadruplet density. ``n_workers`` is the
        number of OpenMP threads the kernel spreads the phi loop over.

        Returns one (n_r, n_r) array per output block, already carrying the
        2πA/(N_p N_p) prefactor.
        """
        if n_workers is None:
            n_workers = self.n_workers
        xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n = xi_tables
        n_out = cbm['n_out']
        n = self.n_r

        def _f(a): return np.ascontiguousarray(a, dtype=np.float64)
        def _i(a): return np.ascontiguousarray(a, dtype=np.int32)

        ## Radial integration nodes: one row of ns subgrid nodes per theta bin
        if self.phi1_subgrid_frac is None:
            phi1_nodes = self.theta_centers[:, None]
            phi1_weights = (self.theta_centers * self.dtheta)[:, None]
        else:
            s1f = np.asarray(self.phi1_subgrid_frac, dtype=float)
            phi1_nodes = self.theta_centers[:, None] + np.outer(self.dtheta, s1f)
            phi1_weights = phi1_nodes * self.dtheta[:, None] * _trapz_weights(s1f)[None, :]
        if self.phi2_subgrid_frac is None:
            phi2_nodes = self.theta_centers[:, None]
            phi2_weights = (self.theta_centers * self.dtheta)[:, None]
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

        ## Tile plan over the (theta1, theta2) axes, banded when masked
        nrb = self.nr_block or n
        rows = range(0, n, nrb)
        if pair_mask is None:
            tiles = [(i1, i2) for i1 in rows for i2 in rows]
        else:
            tiles = [(i1, i2) for i1 in rows if pair_mask[i1:i1 + nrb].any()
                     for i2 in rows
                     if pair_mask[i1:i1 + nrb, i2:i2 + nrb].any()]
        mask_c = _i(np.ones((n, n)) if pair_mask is None else pair_mask)

        ## Quadruplet density: 0 = tensor, 1 = constant, 2 = separable s(t)
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
                # s(t1) s(t2) s(t3) with t1 = phi, t2 the theta1 node and
                # t3 = |psi_b|; tabulating the spline here keeps the kernel
                # free of callbacks. The t3 table is the only sizeable one,
                # (n_phi, n_r*ns2, n_varphi2).
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
            results, n_workers, int(verbose))
        # The per-thread scratch scales with n_workers and is the largest
        # allocation in the module; a failure inside the kernel surfaces here.
        check_clib_error(self._clib)

        if pair_mask is not None:
            results = np.where(pair_mask, results, 0.)

        pre = 2. * np.pi * self.area / np.outer(self.pairs, self.pairs)
        return [pre * a for a in results]

    def _build_notomo_callback_map(self):
        """Callback map for the five non-tomo r-terms.

        The weight ids match the tomo ``rterm_types`` term for term, and with
        a single bin every Wick pairing is degenerate, so all factors are 1
        (see the Wick convention in :meth:`_build_tomo_callback_map`). Table
        0 is ξ_+, table 1 is ξ_-.

        ``w_ids`` 0-4 index the weights the C kernel applies, in the order
        of :attr:`RTERMS`:

            0 = r_+0, w = 1                              L25 Eq. 3.14
            1 = r_+1, w = cos4φ_a cos4φ_b + sin4φ_a sin4φ_b  L25 Eq. 3.15
            2 = r_-0, w = cos(4(V1 + V2 - φ_a - φ_b))    L25 Eq. 3.19
            3 = r_-1, w = cos(4(V1 - V2))                L25 Eq. 3.20
            4 = r_+-, w = 2 cos(4(V2 - φ_b))             asymmetric
        """
        return dict(
            xi_a_idx=[0, 1, 1, 0, 0],
            xi_b_idx=[0, 1, 1, 0, 1],
            w_ids=[0, 1, 2, 3, 4],
            out_idx=[0, 1, 2, 3, 4],
            wick_fac=[1.] * 5,
            n_out=5)

    def _quadruplet_density_terms_c(self, verbose=False, pair_mask=None,
                                    n_workers=None):
        """C-accelerated non-tomo r-term integration (rp0, rp1, rm0, rm1, rpm).

        ξ_± are read from the kernel's piecewise-linear table rather than
        evaluated directly. For a linear ``interp1d`` (the usual input) that is
        exact; for a higher-order spline it is a ~1e-6 approximation on the
        spline's own knots.
        """
        xi_tables = self._xi_tables_from_splines(
            [self.xip, self.xim], lo=self._xi_query_lo(), hi=self._xi_query_hi())
        return self._run_postq_engine(
            self._build_notomo_callback_map(), xi_tables,
            [True, True, True, True, False],
            verbose=verbose, pair_mask=pair_mask, n_workers=n_workers)

    ## R-term (cosmic-variance) getters
    RTERMS = ('rp0', 'rp1', 'rm0', 'rm1', 'rpm')

    def get_all_rterms(self, verbose=False, n_workers=None, pair_mask=None):
        """Compute rp0, rp1, rm0, rm1, rpm in a single pass.

        The 4-D quadruplet-density interpolation dominates the cost and is
        identical across all five terms (same grids, same tile plan); the
        integral is evaluated once per tile and reused for every
        ``(weight, xi)`` pair, turning ~5x that cost into ~1x.

        ``pair_mask`` : (n_r, n_r) bool, optional
            Restrict to a near-diagonal band. Pass the *union* of the
            per-block band masks (i.e. the band for ``max(slab_width.values())``
            -- bands are nested, so this is a superset of every smaller band)
            and then assign each returned term back to its own (smaller) mask.

        Returns
        -------
        (rp0, rp1, rm0, rm1, rpm) : tuple of (n_r, n_r) ndarray
        """
        if self.method_quadruplets == 'continuous':
            raise NotImplementedError(
                "get_all_rterms() is not supported for "
                "method_quadruplets='continuous'; call "
                "getrp0()...getrpm() individually.")
        results = self._quadruplet_density_terms_c(
            verbose=verbose, pair_mask=pair_mask, n_workers=n_workers)
        self.rp0, self.rp1, self.rm0, self.rm1, self.rpm = results
        return tuple(results)

    def _ensure_all_rterms(self, verbose=False, n_workers=None):
        """Populate rp0/rp1/rm0/rm1/rpm.

        If any of the five are still ``None``, (re)computes all of them --
        via the batched :meth:`get_all_rterms` for the discrete methods, or
        via :meth:`_rterms_continuous_c` for ``method_quadruplets='continuous'``
        (unsupported by :meth:`get_all_rterms`).
        """
        if all(getattr(self, n) is not None for n in self.RTERMS):
            return
        if self.method_quadruplets == 'continuous':
            for i, name in enumerate(self.RTERMS):
                setattr(self, name, self._rterms_continuous_c()[i])
        else:
            self.get_all_rterms(verbose=verbose, n_workers=n_workers)

    def _get_rterm(self, index, verbose=False, n_workers=None, pair_mask=None):
        """One r-term, by its position in :attr:`RTERMS`.

        The discrete path computes all five in one batched pass, since the
        q interpolation it shares between them dominates the cost.
        """
        name = self.RTERMS[index]
        if self.method_quadruplets == 'continuous':
            value = self._rterms_continuous_c()[index]
        else:
            value = self.get_all_rterms(
                verbose=verbose, n_workers=n_workers, pair_mask=pair_mask)[index]
        setattr(self, name, value)
        return value

    def getrp0(self, verbose=False, n_workers=None, pair_mask=None):
        """r_+0 contribution to Cov_pp cosmic variance. L25 Eq. 3.14."""
        return self._get_rterm(0, verbose, n_workers, pair_mask)

    def getrp1(self, verbose=False, n_workers=None, pair_mask=None):
        """r_+1 contribution to Cov_pp cosmic variance. L25 Eq. 3.15."""
        return self._get_rterm(1, verbose, n_workers, pair_mask)

    def getrm0(self, verbose=False, n_workers=None, pair_mask=None):
        """r_-0 contribution to Cov_mm cosmic variance. L25 Eq. 3.19."""
        return self._get_rterm(2, verbose, n_workers, pair_mask)

    def getrm1(self, verbose=False, n_workers=None, pair_mask=None):
        """r_-1 contribution to Cov_mm cosmic variance. L25 Eq. 3.20."""
        return self._get_rterm(3, verbose, n_workers, pair_mask)

    def getrpm(self, verbose=False, n_workers=None, pair_mask=None):
        """r_+- contribution to Cov_pm cosmic variance.

        Not symmetrised: Cov(xi_+, xi_-) is genuinely asymmetric under
        (theta_1, theta_2) <-> (theta_2, theta_1). The full Cov_xipm matrix is
        assembled with cov_pm in the upper-right block and cov_pm.T in the
        lower-left.
        """
        return self._get_rterm(4, verbose, n_workers, pair_mask)

    def smooth_rterms(self, window='gaussian', window_size=3.,
                      nphi_fine=1000, nvarphi_fine=2000,
                      terms=('rp0', 'rp1', 'rm0', 'rm1', 'rpm'),
                      floor_frac=1e-3):
        """Ratio-smooth the discrete r-terms against a fine continuum
        reference, mitigating (q_theta, q_phi)-binning stripe bias:

            C_corr = C_inf * smooth(C_est / C_inf)

        C_inf = get{name}_continuous(), evaluated on the same (n_r, n_r)
        theta grid with ``phi_grid`` / ``varphi1_grid`` temporarily
        replaced by finer grids of size ``nphi_fine`` / ``nvarphi_fine``
        spanning the same ranges.

        Already-computed cov_pp/cov_mm/cov_pm/cov_xipm are re-assembled
        in place from the corrected terms.

        Parameters
        ----------
        window : {'gaussian', 'tophat'}
        window_size : float
            Gaussian sigma, or tophat full width, in coarse pixels.
        nphi_fine, nvarphi_fine : int
            Resolution of the fine phi_grid / varphi1_grid used for C_inf.
        terms : tuple of str
            Subset of ('rp0', 'rp1', 'rm0', 'rm1', 'rpm') to correct.
        floor_frac : float
            Regularise the ratio where |C_inf| < floor_frac * max(|C_inf|)
            by clamping the denominator, avoiding division blow-ups near
            the zero-crossings of C_inf.

        Returns
        -------
        tuple of (n_r, n_r) ndarray
            The corrected terms, in the order of ``terms``.
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

    def _get_cov_block(self, index, verbose=False, n_workers=None):
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
                getattr(self, 'get' + rname)(verbose=verbose, n_workers=n_workers)
            out += getattr(self, rname)
        setattr(self, 'cov_' + block, out)
        return out

    def get_cov_pp(self, verbose=False, n_workers=None):
        """Assemble the full Cov_pp matrix."""
        return self._get_cov_block(0, verbose, n_workers)

    def get_cov_mm(self, verbose=False, n_workers=None):
        """Assemble the full Cov_mm matrix."""
        return self._get_cov_block(1, verbose, n_workers)

    def get_cov_pm(self, verbose=False, n_workers=None):
        """Assemble the full Cov_pm matrix. No diagonal shape-noise term."""
        return self._get_cov_block(2, verbose, n_workers)

    def get_cov_xipm(self, verbose=False, n_workers=None):
        """Assemble the full (2 n_r × 2 n_r) Cov(ξ₊, ξ₋) matrix."""
        # Batch all five 4-D r-terms in a single pass (q_func_local is
        # term-invariant and dominates the cost) before assembling pp/mm/pm.
        self._ensure_all_rterms(verbose=verbose, n_workers=n_workers)
        if self.cov_pp is None: self.get_cov_pp(verbose=verbose, n_workers=n_workers)
        if self.cov_mm is None: self.get_cov_mm(verbose=verbose, n_workers=n_workers)
        if self.cov_pm is None: self.get_cov_pm(verbose=verbose, n_workers=n_workers)
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
        """Return the quadruplet density tensor of shape
        (q_n_r, q_n_r, q_n_r, q_n_phi, q_n_phi).

        If quadruplets is a callable (analytic case), the tensor is
        materialised on the q bin grid here.
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
                         n_workers=None, template=None, pairs=None, **tomo):
        """Create an instance from pre-computed covariance components.

        Bypasses the expensive count-tensor initialisation in ``__init__``.
        Intended for use by :meth:`subbin` and :meth:`GGCovariance.subbin_tomo`.

        ``template`` : GGCovarianceNoTomo, optional
            Instance to inherit the q / triplet grids, the suppression spline,
            the integration grids and the worker configuration from, instead of
            the precomputed-instance defaults (q on the 2PCF grid, no
            integration grids). The tomographic subbin passes its parent, whose
            grids the returned instance must keep to stay re-integrable.
        ``pairs`` : (n_r,) array_like, optional
            Fine pair counts to carry over; ``None`` for a precomputed instance
            with no count tensors.
        ``tomo``
            Tomographic blocks (``D_tomo``, ``qpp_tomo``, ``qmm_tomo``,
            ``qpm_tomo``, ``rterms_tomo``) plus the bin-pair bookkeeping
            (``_xip``, ``_xim``, ``ntomo``, ``n_bp``, ``bp_list``,
            ``nbar_ref``), set verbatim on the new instance.
        """
        obj = object.__new__(cls)

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

        # Integration grids and parallel-worker memory control: inherited from
        # the template, else the precomputed defaults (such an instance does
        # not run the quadruplet integral, but keep the attributes consistent).
        for name, default in (('phi_grid', None), ('varphi1_grid', None),
                              ('varphi2_grid', None), ('phi1_subgrid_frac', None),
                              ('phi2_subgrid_frac', None), ('v2_block', 16),
                              ('nr_block', None)):
            setattr(obj, name, getattr(template, name, default))
        obj.n_workers = 1 if n_workers is None else int(n_workers)

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

        return obj

    _RECOMPUTE_FULL_COMPONENTS = ('D', 'qpp', 'qmm', 'qpm')

    def _grid_kwargs(self, new_tc, new_te, **overrides):
        """Constructor kwargs rebuilding this instance on a new θ grid.

        Carries the φ / q grids, the survey scalars and the integration and
        worker configuration across unchanged; ``overrides`` supply what
        differs (the ``method_*`` switches, the count tensors, the catalogue).
        Shared by :meth:`_fine_instance_from_catalog` and
        :meth:`_recompute_slab`.
        """
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
            n_workers=self.n_workers)
        kw.update(overrides)
        return kw

    def _fine_instance_from_catalog(self, new_tc, new_te, verbose=False):
        """Build a fine-grid instance from this instance's catalogue.

        Re-measures the (cheap) fine pair / triplet counts on
        ``new_tc``/``new_te`` from ``self.catalog`` and reuses this instance's
        already-measured quadruplet density (geometry-only), so the expensive
        4PCF is not remeasured. Used by :meth:`subbin` to auto-build the
        ``slab_instance`` when one is not supplied. The tomographic
        :class:`GGCovariance` overrides this to carry the per-bin-pair xi
        splines.
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

        Each covariance component is resampled individually, by default in
        log(θ) space. Plain interpolation undershoots the diagonal peak of
        strongly diagonal-dominated components (notably ξ₋ξ₋); set
        ``recompute_slab=True`` to repair a near-diagonal band by evaluating
        the full discrete integrators directly on the fine grid and
        overwriting the interpolated values there.

        Parameters
        ----------
        new_theta_centers : (n_new,) array_like
            Centers of the fine radial bins, in arcmin.
        new_theta_edges : (n_new+1,) array_like
            Edges of the fine radial bins, in arcmin.
        new_pairs : (n_new,) array_like, optional
            Pair counts *measured* at the fine binning. When given,
            components are converted to the density

                DENSITY(θ₁,θ₂) = arr(θ₁,θ₂) · Nₚ(θ₁)·Nₚ(θ₂) / (θ₁Δθ₁·θ₂Δθ₂)

            interpolated, then rescaled by the measured fine Nₚ — this
            propagates the bin-width-dependent boundary suppression that
            plain interpolation would miss. D is recomputed as
            σ_ε⁴⟨w²⟩²/⟨w⟩² / Nₚ_fine. If None and ``infer_pairs=False``,
            components are log-interpolated directly (appropriate when
            Nₚ ∝ θΔθ, the borderless / analytic case).
        infer_pairs : bool, optional
            If True and ``new_pairs`` is None, estimate Nₚ_fine by splining
            the ratio Nₚ_coarse / Nₚ_ideal(θ) to the fine grid and rescaling
            by Nₚ_ideal(θ_fine), where Nₚ_ideal(θ) = 2π A n̄² ⟨w⟩² θΔθ s(θ).
            Requires ``nbar`` and ``area``.
        use_corr : bool, optional
            If True (default), symmetric components are interpolated via
            their correlation coefficient ρ(θ₁,θ₂) = C(θ₁,θ₂) /
            √(C(θ₁,θ₁)C(θ₂,θ₂)), smoother over a large dynamic range.
        log_theta : bool, optional
            If True (default), interpolate in log(θ).
        recompute_slab : bool, optional
            Recompute a near-diagonal band of width ``slab_width`` directly
            on the fine grid and overwrite the interpolated values there.
            Requires ``slab_width``. Without ``slab_instance`` an analytic
            fine instance is auto-built (analytic method only).
        slab_instance : GGCovarianceNoTomo, optional
            Fine-grid instance (same 2PCF ``theta`` grid as
            ``new_theta_centers``/``new_theta_edges``) supplying measured
            count tensors for the band — the route for ``'discrete'``: only
            the band is evaluated, so the O(n_r⁴) quadruplet integral is
            never run over the full matrix. The quadruplet density may
            reuse the coarse tensor (it lives on the independent ``q_theta``
            grid and is queried continuously). When None and this instance
            was built from a ``catalog``, the fine instance is auto-built via
            :meth:`_fine_instance_from_catalog` (fine pairs/triplets
            re-measured, coarse quadruplet density reused).
        slab_width : float or dict, optional
            Half-width of the recomputed band as a θ ratio: element (a, b)
            is recomputed when ``1/w < θ_b/θ_a < w``. A scalar applies to
            all blocks; a dict with keys ``'pp'``/``'mm'``/``'pm'`` sets the
            ξ₊ξ₊/ξ₋ξ₋/ξ₊ξ₋ widths independently (missing = 1.0, no
            recompute). ξ₋ξ₋ is the most diagonal-dominated and typically
            needs the widest band.
        slab_nr_block : int, optional
            θ-tile size for the banded quadruplet integrator. Defaults to
            roughly the band half-width in fine bins, so off-band tiles are
            skipped entirely.
        recompute_full : bool or iterable of {'D', 'qpp', 'qmm', 'qpm'}, optional
            Components cheap enough (O(n) or O(n²·n_phi), unlike the
            O(n²·n_phi²) 4-D ``r*`` integrals) to evaluate on the *entire*
            fine grid via ``slab_instance`` rather than interpolated.
            ``True`` = all four, or pass a subset. Requires
            ``slab_instance``. Combine with ``recompute_slab=True`` to keep
            ``r*`` on the cheaper banded scheme — ``_recompute_slab`` then
            skips the now-redundant masked recompute for components listed
            here.
        verbose : bool, optional
            Forward per-φ-node progress reporting to the quadruplet
            integrals (coarse r-terms computed here and the slab recompute).
        dry : bool, optional
            If True, skip computation and print/return a summary dict (keys
            ``'n_fine'``, ``'nr_block'``, ``'slab_elements'``, ``'n_tiles'``,
            ``'total_elements'``, ``'total_integrator_tile_calls'``) of how
            many elements/tiles each band touches. ``total_integrator_tile_
            calls`` is the tile count of the widest band, since
            ``get_all_rterms`` computes rp0/rp1/rm0/rm1/rpm in one batched
            pass over the union of the pp/mm/pm bands. Requires
            ``recompute_slab=True`` and ``slab_width`` for a non-trivial
            summary.

        Returns
        -------
        GGCovarianceNoTomo
            New instance on ``new_theta_centers``/``new_theta_edges`` with
            all covariance components pre-loaded (diagonal band recomputed
            when ``recompute_slab=True``).

        Notes
        -----
        Parallelism uses the instance-level ``n_workers`` (or
        ``slab_instance.n_workers`` for the banded recompute); the banded
        path parallelises over φ nodes just like the full path.
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

        # Auto-build the fine-grid slab instance from the catalogue when one is
        # not supplied (discrete method): re-measure the cheap fine pair/triplet
        # counts and reuse this instance's quadruplet density.
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

        new_dtheta = np.diff(new_te)
        n_new = len(new_tc)

        tc = self.theta_centers
        dthc = self.dtheta

        x_c = np.log(tc) if log_theta else tc
        x_new = np.log(new_tc) if log_theta else new_tc

        _interp_arr = self._build_block_interpolator(x_c, x_new, n_new, use_corr)

        ## D and 2-D component strategies
        Np_c = np.asarray(self.pairs)
        phiw_c = tc * dthc # θΔθ, coarse

        if infer_pairs and new_pairs is None:
            if self.area is None or self.nbar is None:
                raise ValueError(
                    "`infer_pairs=True` requires `nbar` and `area` to be set")
            ideal_c = (2. * np.pi * self.area * self.nbar**2
                          * self._w1**2 * tc * dthc
                          * self._suppression(tc))
            ratio_c = Np_c / ideal_c
            ideal_f = (2. * np.pi * self.area * self.nbar**2
                          * self._w1**2 * new_tc * new_dtheta
                          * self._suppression(new_tc))
            log_ratio_f = np.interp(x_new, x_c, np.log(ratio_c))
            new_pairs = np.exp(log_ratio_f) * ideal_f

        if new_pairs is not None:
            Np_f = np.asarray(new_pairs, dtype=float)
            if Np_f.shape != (n_new,):
                raise ValueError(
                    f"new_pairs must have shape ({n_new},), got {Np_f.shape}")
            phiw_f = new_tc * new_dtheta

            # D = σ_ε⁴⟨w²⟩²/⟨w⟩² / Nₚ → σ_ε⁴⟨w²⟩²/⟨w⟩² = D·Nₚ = const
            D_scale = float(np.mean(self.D * Np_c))
            D_fine = D_scale / Np_f

            ow_Np_c = np.outer(Np_c, Np_c)
            ow_phiw_c = np.outer(phiw_c, phiw_c)
            ow_phiw_f = np.outer(phiw_f, phiw_f)
            ow_Np_f = np.outer(Np_f, Np_f)

            def _interp(arr, sym):
                # Remove coarse Nₚ variation → smooth DENSITY, interpolate,
                # then apply measured fine Nₚ.
                density = arr * ow_Np_c / ow_phiw_c
                result = (_interp_arr.corr(density) if (use_corr and sym)
                           else _interp_arr.raw(
                               density, log_interp=bool(np.all(density > 0))))
                if sym:
                    result = 0.5 * (result + result.T)
                return result * ow_phiw_f / ow_Np_f

        else:
            # Borderless / analytic case: Nₚ ∝ θΔθ, q and r are
            # bin-width-independent → direct log-interpolation.
            D_scaled = self.D * dthc # ∝ 1/θ, bin-width-independent
            D_fine = np.exp(np.interp(x_new, x_c,
                                        np.log(D_scaled))) / new_dtheta
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

        # D / qpp / qmm / qpm are cheap enough to evaluate over the *entire*
        # fine grid (unlike the 4-D r* integrals) — overwrite the interpolated
        # values in full rather than only on the diagonal band.
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
            n_workers=self.n_workers,
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

        # Mirrors _recompute_slab exactly via the shared _auto_reduce_tiles.
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
        # When slab_instance has an nr_block set, rebase on its own budget.
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

        # r-terms (4-D, expensive): bands are nested in slab_width, so a
        # single get_all_rterms() pass over the widest band covers all of
        # rp0/rp1/rm0/rm1/rpm at once -- one q_func_local evaluation per tile.
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
        """Boolean (N, N) band around the diagonal: ``1/w < θ_j/θ_i < w``."""
        tc = np.asarray(theta_centers, dtype=float)
        N = len(tc)
        if slab_width is None or slab_width <= 1.:
            return np.zeros((N, N), dtype=bool)
        ratio = tc[None, :] / tc[:, None]
        return (ratio < slab_width) & (ratio > 1. / slab_width)

    @staticmethod
    def _build_block_interpolator(x_c, x_new, n_new, use_corr=True):
        """Return a callable ``interp(arr, sym)`` that interpolates an
        (n_coarse, n_coarse) block to (n_new, n_new) in log-theta space.

        Zero blocks are returned as zero without interpolation.
        """
        # Bilinear interpolation on a tensor grid is separable, so the 2-D
        # map is the same 1-D weight matrix applied along both axes. Building
        # it once here replaces one RegularGridInterpolator per block, of
        # which subbin_tomo needs O(n_bp^2).
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

        # Expose the raw / correlation-coefficient interpolators so callers
        # (e.g. subbin's measured-pairs branch) can reuse them directly.
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
        """Overwrite the near-diagonal band of the (interpolated) fine-grid
        components with values evaluated directly on the fine grid.

        Helper for :meth:`subbin` (``recompute_slab=True``). ``components`` is
        ``(qpp, qmm, qpm, rp0, rp1, rm0, rm1, rpm)`` already interpolated onto
        the fine grid; the returned tuple has the diagonal band overwritten.

        ``full_set`` : set of {'qpp', 'qmm', 'qpm'}, optional
            Components :meth:`subbin` will overwrite in full afterwards
            (its ``recompute_full`` option) — the masked recompute below is
            skipped for these, since it would be discarded immediately.

        ``fine`` : GGCovarianceNoTomo, optional
            Fine-grid instance (same 2PCF ``theta`` grid as ``new_tc``/
            ``new_te``) whose measured count tensors supply the band — the
            route for ``'discrete'`` (quadruplet density may reuse the
            coarse tensor, since it lives on the independent ``q_theta``
            grid). If ``None``, an analytic fine instance is auto-built from
            this instance's ``nbar``/``area``/suppression (analytic method
            only).
        """
        if slab_width is None:
            raise ValueError("recompute_slab=True requires `slab_width`.")

        # Per-block band widths
        widths = self._parse_slab_widths(slab_width)
        masks = {b: self._slab_band_mask(new_tc, w) for b, w in widths.items()}

        # Tile size for the banded quadruplet integrator -- see
        # _auto_reduce_tiles for the auto-reduction strategy.
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
            ## Auto-build an analytic fine instance
            if not (self.method_pairs == 'analytic'
                    and self.method_triplets == 'analytic'
                    and self.method_quadruplets == 'analytic'):
                raise NotImplementedError(
                    "recompute_slab=True without an explicit `slab_instance` "
                    "is only supported for analytic count tensors. For the "
                    "'discrete' method, build a fine-grid GGCovarianceNoTomo "
                    "with your measured triplet/quadruplet tensors and pass it "
                    "as subbin(..., slab_instance=fine).")
            # The analytic q tensor lives on the new 2PCF grid here, and its
            # angular grid follows phi_centers (not this instance's q_phi grid).
            fine = GGCovarianceNoTomo(**self._grid_kwargs(
                new_tc, new_te,
                q_theta_centers=new_tc, q_theta_edges=new_te,
                q_phi_centers=None, q_phi_edges=None,
                method_pairs='analytic', method_triplets='analytic',
                method_quadruplets='analytic',
                v2_block=v2_block_slab, nr_block=nr_block))
            # Use the measured/inferred fine pair counts so the recomputed band
            # is consistent with the interpolated off-diagonal and the fine D.
            # (The analytic triplet/quadruplet tensors still use nbar; the Nₚ
            # ratio carries the leading bin-width/boundary correction.)
            if fine_pairs is not None:
                fine.pairs = np.asarray(fine_pairs, dtype=float)
        else:
            ## Use the caller-supplied fine instance (any method)
            self._check_slab_instance_grid(fine, new_tc, new_te)
            # Apply tile-size reduction. When slab_nr_block is given explicitly,
            # honour it directly. Otherwise apply _auto_reduce_tiles: if fine
            # already has an nr_block set (user-chosen for memory safety),
            # rebase the reduction on that value so nr_block²·v2_block stays
            # within the user's budget; if not, use the m_max-derived values
            # computed above.
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

        # qpp/qmm/qpm (triplet terms, cheap): per-block masked recompute.
        mpp = masks['pp']
        if mpp.any() and 'qpp' not in full_set:
            qpp_f[mpp] = fine.getqpp(pair_mask=mpp)[mpp]
        mmm = masks['mm']
        if mmm.any() and 'qmm' not in full_set:
            qmm_f[mmm] = fine.getqmm(pair_mask=mmm)[mmm]
        mpm = masks['pm']
        if mpm.any() and 'qpm' not in full_set:
            qpm_f[mpm] = fine.getqpm(pair_mask=mpm)[mpm]

        # r-terms (4-D, expensive): bands are nested in slab_width, so a
        # single batched pass over the widest band covers pp/mm/pm at once.
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

    ################
    # orpheus hook #
    ################

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
        """``catalog['nmaxs']`` in a form the NPCF constructors accept.

        They take either a scalar ``int`` (broadcast over the ``order-2``
        angular arguments) or an array; a plain list has no ``.astype`` and
        would fail inside ``npcf_base``.
        """
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
        """Lazily build, patch-decompose and cache the orpheus catalogue.

        ``kind='spin'`` -> SpinTracerCatalog (GG / GGG), patch-extended for the
        2PCF θ range; ``kind='scalar'`` -> ScalarTracerCatalog (NNNN),
        patch-extended for the q θ range. Tracers are unit fields so the
        measured ``norm`` / ``npcf_norm`` / ``npcf`` are pure (weighted)
        geometry counts. Spherical catalogues must be decomposed into patches
        before ``process`` / ``frompatchind`` (orpheus requirement), so
        ``topatches`` is applied here with the requested ``npatches``.
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
        """Deterministic filename stem uniquely identifying a measurement run.

        Combines a hash of the (ra, dec, weight) arrays with the binning /
        orpheus parameters that define the measurement, so that an identical
        request reuses the cached file under ``path_save``.
        """
        import hashlib
        ra, dec, w = self._catalog_arrays()
        h = hashlib.sha1()
        for a in (ra, dec, w):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        rmin_pixsize = self._catalog_param('rmin_pixsize', 40)
        tree_resos = list(np.ravel(self._catalog_param(
            'tree_resos', [0., 0.5, 1., 2., 4.])))
        # Every statistic is measured on the patched catalogue, so npatches
        # belongs in all three keys. It is appended only when it differs from
        # the default of 1, so caches written before it entered the pairs and
        # triplets keys stay valid. nmaxs is only read by the multipole
        # estimators -- GGCorrelation never sees it.
        npatches = int(self._catalog_param('npatches', required=True))
        np_tag = () if npatches == 1 else (npatches,)
        # Likewise the cell shuffling only enters when it is not the default.
        shuffle_pix = int(self._catalog_param('shuffle_pix', 1))
        sh_tag = () if shuffle_pix == 1 else (('shuffle_pix', shuffle_pix),)
        if stat == 'pairs':
            # 'ordered': the counts are GGCorrelation's norm, which already holds
            # both orders of every pair. Earlier caches held twice that.
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
        """Measure a multiplet count tensor from ``self.catalog`` via orpheus.

        Dispatched from ``__init__`` for each contribution whose ``method_*``
        is ``'discrete'`` and whose tensor was not supplied:

          * ``'pairs'`` -> weighted pair counts N_p(θ), shape (n_r,)
          * ``'triplets'`` -> weighted triplet counts N_t(t1, t2, φ),
                              shape (triplet_n_r, triplet_n_r, n_phi)
          * ``'quadruplets'`` -> quadruplet DENSITY q, shape
                                 (q_n_r, q_n_r, q_n_r, q_n_phi, q_n_phi)

        The orpheus measurement binning is derived from the covariance grids
        (θ grid for pairs; the oversampled triplet θ grid + ``phi_centers``
        for triplets; ``q_theta`` / ``q_phi`` for quadruplets). Only
        ``rmin_pixsize``, ``tree_resos``, ``npatches`` and ``nmaxs`` come from
        the ``catalog`` dict. Catalog θ grids must be log-uniform.

        Results are cached under ``catalog['path_save']`` keyed by a hash of
        the inputs (so an identical request is reloaded, not recomputed).
        After a tensor is loaded into the class, ``store_multiplets`` controls
        retention: ``False`` deletes every multiplet file; ``True`` keeps the
        combined tensors but deletes the per-footprint 4PCF patch files.
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
        """Weighted pair counts N_p(θ) on the 2PCF θ grid (shape (n_r,))."""
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
                nthreads=self.n_workers)
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
                nthreads=self.n_workers)
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
        """Quadruplet density q on the q grid, measured (and patch-combined).

        Returns the density tensor; per-footprint patch files are always
        removed afterwards, the combined density file only when
        ``store_multiplets`` is False.
        """
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
        # The per-footprint patch files are intermediate -- always drop them.
        self._cleanup_quadruplet_patchfiles(tag)
        if not self.store_multiplets:
            self._safe_remove(fpath)
        return density

    def _measure_quadruplet_counts(self, tag):
        """Measure the real-space 4PCF counts on the q grid.

        Each patch is measured sequentially with the full thread budget
        (``self.n_workers``); orpheus' 4PCF kernels parallelise internally over
        radial-bin combinations via ``#pragma omp parallel num_threads(...)``,
        so a serial patch loop keeps every core busy without forking. (A
        forked patch pool would inherit an already-initialised OpenMP runtime
        from the parent's pair/triplet measurement and silently run each child
        single-threaded.) Per-patch multipoles are written to disk (enabling
        the ``store_multiplets`` footprint cleanup and crash recovery) and
        summed before the single multipole->real conversion. ``npatches == 1``
        is the trivial single-footprint case. Returns the real-space count
        tensor ``(q_n_r, q_n_r, q_n_r, q_n_phi, q_n_phi)``.
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
        cat = self._ensure_catalog('scalar')   # spherical, already patched

        def _new_nnnn():
            return NNNNCorrelation_NoTomo(
                min_sep=min_sep, max_sep=max_sep, nbinsr=self.q_n_r,
                nbinsphi=self.q_n_phi, nmaxs=nmaxs, method=method,
                rmin_pixsize=rmin_pixsize, tree_resos=tree_resos,
                shuffle_pix=shuffle_pix, verbosity=verbosity, nthreads=self.n_workers)

        if npatches == 1:
            # Single (flat) footprint patch -- process it directly.
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

        ## Multi-patch: serial loop, each patch using all threads
        # Per-patch multipoles are cached so an interrupted run can resume.
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

        ## Combine: sum the per-patch multipoles, average the bin centers.
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
        """Convert real-space 4PCF counts to the density q (L25 Eq. 4.16/4.17).

            q = N_q / V_q ,
            V_q = 2π A t1 dt1 t2 dt2 t3 dt3 dφ12 dφ13 ,

        where N_q is the real-space NNNN npcf, i.e. the quadruplet count of
        each bin (matching the convention used with
        :meth:`set_quadruplets_from_counts`).
        """
        if self.area is None:
            raise ValueError("`area` is required to convert counts to density")
        q = np.array(counts, dtype=float)
        tdt = self.q_theta_centers * self.q_dtheta
        dp = self.q_dphi
        # V_q is separable -- divide axis by axis (see set_quadruplets_from_counts).
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
        """Extract per-bin filter weights from a helper GGCorrelation instance.

        Uses ``gg.computeMap2`` / ``gg.computepuremode`` (from
        :class:`~orpheus.npcf_second.GGCorrelation`) with identity-matrix
        inputs to read off the linear filter kernel without duplicating any
        filter-function code. The helper instance must already have
        ``bin_centers_mean`` ((n_r,) theta grid), ``nzcombis`` (= n_r, one
        basis vector per "tomo-bin"), ``bin_centers`` ((n_r, n_r) tiled theta
        grid) and ``binsize`` set (see :meth:`cov_of_statistic`); ``xip``/
        ``xim`` are set by this method before each call.

        Parameters
        ----------
        filter_type : {'Map2', 'puremode'}
            ``'Map2'`` – Crittenden aperture-mass filter; ``x`` = aperture
                         radius R (same units as ``gg.bin_centers_mean``).
            ``'puremode'`` – Schneider+2022 pure E/B kernel; ``x`` is ignored.
            For ``'cosebis'``, do **not** call this method per mode; use
            ``gg.computecosebi(Nmax)`` directly in :meth:`cov_of_statistic`.
        x : scalar or None
            See above.

        Returns
        -------
        For ``'Map2'``:
            w_plus, w_minus : (n_r,) ndarray
                E-mode weights; B-mode uses the same w_plus but negated w_minus.
        For ``'puremode'``:
            K : (6·n_r, 2·n_r) ndarray
                Transformation matrix such that
                ``[xip_E, xip_B, xip_amb, xim_E, xim_B, xim_amb] = K @ [ξ_+, ξ_-]``.
        """
        n = gg.nbinsr
        # Measure correction: gg uses `binsize` as the log Δθ;
        # the exact per-bin measure is dtheta/theta. For log-uniform bins
        # (the typical case) this ratio equals 1. We absorb it into the
        # δ-function inputs so the output already carries the right measure.
        theta = np.asarray(gg.bin_centers_mean, dtype=float)
        # Reconstruct dtheta from the GGCorrelation bin edges
        dtheta = theta * (np.exp(gg.binsize) - 1.) # exact for log-uniform grid
        # corr[j]: converts (binsize * theta[j]^2) measure → (dtheta[j] * theta[j]) measure
        corr = dtheta / (gg.binsize * theta) # ≡ 1 for log-uniform bins

        if filter_type == 'Map2':
            R = float(x)
            # Inject xip = diag(corr), xim = 0 → output = T_+(θ_i/R) * dtheta_i * θ_i/(2R²)
            gg.xip = np.diag(corr).astype(complex)
            gg.xim = np.zeros((n, n), dtype=complex)
            res_p = gg.computeMap2([R]) # (4, n, 1)
            # Inject xip = 0, xim = diag(corr) → output = T_-(θ_i/R) * dtheta_i * θ_i/(2R²)
            gg.xip = np.zeros((n, n), dtype=complex)
            gg.xim = np.diag(corr).astype(complex)
            res_m = gg.computeMap2([R])
            return res_p[0, :, 0], res_m[0, :, 0]

        elif filter_type == 'puremode':
            # Extract transformation K row-by-row via identity inputs.
            # With xip = I·corr, xim = 0: xip_E[iz, j] = M_pEpp[j, iz] * corr[iz]
            # so M_pEpp = (xip_pure[0] / corr).T
            C = np.diag(corr).astype(complex)
            Z = np.zeros((n, n), dtype=complex)
            gg.xip = C
            gg.xim = Z
            xip_p, xim_p = gg.computepuremode() # (3, n, n) each
            gg.xip = Z
            gg.xim = C
            xip_m, xim_m = gg.computepuremode()

            # (3, n, n) → (n, n) via divide by corr and transpose
            def _extract(res):
                return (res / corr[np.newaxis, :, np.newaxis]).real.transpose(0, 2, 1)

            pp = _extract(xip_p) # (3, n, n): pp[k] = M_k_pp
            pm = _extract(xip_m) # (3, n, n): pm[k] = M_k_pm
            mp = _extract(xim_p) # (3, n, n): mp[k] = M_k_mp
            mm = _extract(xim_m) # (3, n, n): mm[k] = M_k_mm

            # Ordering: xip_E[0], xip_B[1], xip_amb[2], xim_E[0], xim_B[1], xim_amb[2]
            K = np.block([
                [pp[0], pm[0]], # xip_E
                [pp[1], pm[1]], # xip_B
                [pp[2], pm[2]], # xip_amb
                [mp[0], mm[0]], # xim_E
                [mp[1], mm[1]], # xim_B
                [mp[2], mm[2]], # xim_amb
            ])
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
        """Transform Cov(ξ_+, ξ_-) into the covariance of an integrated statistic.

        Implements the linear propagation from S+02 Eq. (41)/(42):

            Cov(S_a, S_b) = w_a^T Cov_xipm w_b,

        for any statistic S = Σ_i [w_+(θ_i) ξ_+(θ_i) + w_-(θ_i) ξ_-(θ_i)].
        Filter weights are extracted from a helper
        :class:`~orpheus.npcf_second.GGCorrelation` instance via
        :meth:`_getfilter`, using ``nzcombis = n_r`` basis vectors so that one
        call returns all n_r weights at once, with the per-bin integration
        measure dtheta*theta absorbed into the diagonal input (exact for
        log-uniform theta grids).

        Parameters
        ----------
        filter_type : {'Map2', 'cosebis', 'puremode'}
        radii : array_like
            Aperture radii — required for ``'Map2'``.
        Nmax : int
            Number of modes — required for ``'cosebis'``.
        theta_min, theta_max : float, optional
            Survey angular limits. Defaults to ``self.theta_edges[0/−1]``.
            Required for ``'cosebis'`` (defines the COSEBI integration range)
            and used by ``'puremode'`` for its geometric constants.
        cosebi_dps : int
            mpmath decimal precision for COSEBI root-finding.
        cosebi_nsample : int
            Quadrature points for T_-(θ) evaluation inside
            ``GGCorrelation.computecosebi``.
        _return_weights : bool
            If True, also return the filter weight matrices alongside the
            covariance dict. Used internally by
            :meth:`~GGCovariance.cov_of_statistic_tomo` to compute weights
            once and reuse them for all bin-pair blocks.

        Returns
        -------
        dict
            ``'Map2'`` / ``'cosebis'``
                ``'cov_EE'`` : (N, N) – E-mode covariance
                ``'cov_BB'`` : (N, N) – B-mode covariance
                ``'cov_EB'`` : (N, N) – E×B cross-covariance

            Map2 is dimensionless; COSEBI blocks carry the θ² measure of
            ``computecosebi`` and are returned in rad⁴, matching a data
            vector from ``gg.computecosebi(Nmax)`` at its default units.

            ``'puremode'``
                ``'cov'`` : (6·n_r, 6·n_r) – full covariance of
                    [xip_E, xip_B, xip_amb, xim_E, xim_B, xim_amb]
                ``'labels'`` : list of 6·n_r label strings

        If ``_return_weights=True``, returns ``(dict, weights)`` where
        ``weights`` is ``(W_E, W_B)`` for Map2/cosebis or ``(K, labels)``
        for puremode.

        Notes
        -----
        The per-bin integration measure dθ·θ is absorbed into the diagonal
        input matrix, which is exact for log-uniform θ grids and a good
        approximation otherwise.
        """
        from .npcf_second import GGCorrelation

        if self.cov_xipm is None:
            self.get_cov_xipm()
        C = self.cov_xipm
        n = self.n_r
        tc = self.theta_centers
        tmin = float(self.theta_edges[0]) if theta_min is None else float(theta_min)
        tmax = float(self.theta_edges[-1]) if theta_max is None else float(theta_max)

        # Allocate one helper GGCorrelation for filter extraction.
        # We bypass __init__ as filters are python-only and thus we don't need
        # to load all the C stuff
        gg = object.__new__(GGCorrelation)
        gg.binsize = np.log(tmax / tmin) / n
        gg.bin_edges = np.geomspace(tmin, tmax, n + 1)
        gg.min_sep = tmin
        gg.max_sep = tmax
        gg.nbinsr = n
        gg.bin_centers_mean = tc
        gg.bin_centers = np.tile(tc, (n, 1))
        gg.nzcombis = n # We do this to get the weights for each radius
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

        # COSEBIs
        elif filter_type == 'cosebis':
            if Nmax is None:
                raise ValueError("'Nmax' is required for filter_type='cosebis'")
            # Measure correction (see _getfilter docstring)
            dtheta = tc * (np.exp(gg.binsize) - 1.)
            corr = dtheta / (gg.binsize * tc) # = 1 for log-spacing, but lets be safe

            # T_+ weights: set xip=diag(corr), xim=0. units is pinned so the
            # covariance stays consistent with the computecosebi default even
            # if that default changes; cov_EE/BB/EB are then in rad^4.
            gg.xip = np.diag(corr).astype(complex)
            gg.xim = np.zeros((n, n), dtype=complex)
            res_p = gg.computecosebi(Nmax, dps=cosebi_dps,
                                      Tminus_nsample=cosebi_nsample,
                                      units='rad2')
            # T_- weights: set xip = 0, xim = diag(corr)
            gg.xip = np.zeros((n, n), dtype=complex)
            gg.xim = np.diag(corr).astype(complex)
            res_m = gg.computecosebi(Nmax, dps=cosebi_dps,
                                      Tminus_nsample=cosebi_nsample,
                                      units='rad2')
            # res_p/m shape: (4, n_r, Nmax); res[0] = E contribution from ξ_±
            # E-mode weight matrix W_E[n, iz] = T_+n(θ_iz)*meas + T_-n(θ_iz)*meas
            W_plus = res_p[0].real.T # (Nmax, n_r): T_+ weights from ξ_+
            W_minus = res_m[0].real.T # (Nmax, n_r): T_- weights from ξ_-
            W_E = np.concatenate([W_plus, W_minus], axis=1) # (Nmax, 2·n_r)
            W_B = np.concatenate([W_plus, -W_minus], axis=1)

        # Pure mode correlation functions
        elif filter_type == 'puremode':
            K = GGCovarianceNoTomo._getfilter('puremode', gg, None)   # (6·n_r, 2·n_r)
            cov_full = K @ C @ K.T
            lbl = (
                [f'xip_E[{i}]' for i in range(n)]
                + [f'xip_B[{i}]' for i in range(n)]
                + [f'xip_amb[{i}]' for i in range(n)]
                + [f'xim_E[{i}]' for i in range(n)]
                + [f'xim_B[{i}]' for i in range(n)]
                + [f'xim_amb[{i}]' for i in range(n)]
            )
            out = {'cov': cov_full, 'labels': lbl}
            return (out, (K, lbl)) if _return_weights else out

        else:
            raise ValueError("filter_type must be 'Map2', 'cosebis' or "
                             f"'puremode', got '{filter_type}'")

        out = {
            'cov_EE': W_E @ C @ W_E.T,
            'cov_BB': W_B @ C @ W_B.T,
            'cov_EB': W_E @ C @ W_B.T}
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
    """Tomographic extension of :class:`GGCovarianceNoTomo`.

    Accepts per-bin-pair xi splines and computes the full tomographic
    covariance matrix Cov(xi_pm(theta, i, j); xi_pm(theta, k, l)).

    The quadruplet density tensor q is shared across all tomographic bin
    combinations — only the xi functions change per combo.

    Parameters
    ----------
    spline_xip, spline_xim : callable or list of (i, j, callable)
        If a bare callable: single non-tomographic spline, equivalent to
        ``[(0, 0, f)]`` (ntomo=1).
        If a list of ``(i, j, spline)`` tuples with ``i <= j``: one entry
        per unique tomographic bin pair. Length must equal
        ``ntomo*(ntomo+1)/2`` for some integer ntomo.
    **kwargs
        All other parameters forwarded to :class:`GGCovarianceNoTomo`.
    """

    def __init__(self, *, spline_xip=None, spline_xim=None,
                 nbar_ref=None, **kwargs):
        self._xip, self._xim, self.ntomo = self._parse_xi_splines(
            spline_xip, spline_xim)
        self.n_bp = self.ntomo * (self.ntomo + 1) // 2
        self.bp_list = [(i, j)
                        for i in range(self.ntomo)
                        for j in range(i, self.ntomo)]

        ## Per-bin sigma_eps and nbar: parse before super().__init__()
        # (parent needs a scalar sigma_eps for float(); we pass the first
        # element and override self.sigma_eps with the full array afterwards)
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
            # The parent builds the idealised multiplets and the constant
            # quadruplet density, which live at the reference density that
            # the tomographic terms are rescaled from.
            kwargs['nbar'] = float(nb[0]) if nbar_ref is None else float(nbar_ref)
        else:
            nb = None

        # Set nbar_ref before super().__init__() so that discrete multiplet
        # measurements (triggered inside the parent's init) can save it.
        self.nbar_ref = float(nbar_ref) if nbar_ref is not None else None

        # Parent needs scalar xip/xim for its internal plumbing (analytic
        # pair/triplet counts, q_func building). Pass the (0,0) auto-
        # correlation as a reasonable default; the parent's xip/xim are
        # NOT used for the tomo r-term computation.
        super().__init__(
            spline_xip=self._xip.get((0, 0), spline_xip),
            spline_xim=self._xim.get((0, 0), spline_xim),
            **kwargs)

        # Override parent's scalars with per-bin arrays
        self.sigma_eps = se
        if nb is not None:
            self.nbar = nb

        ## Tomo output storage
        self.rterms_tomo = {}
        self.D_tomo = None
        self.qpp_tomo = None
        self.qmm_tomo = None
        self.qpm_tomo = None
        self.cov_pp_tomo = None
        self.cov_mm_tomo = None
        self.cov_pm_tomo = None
        self.cov_xipm_tomo = None

    #####################
    # Xi spline parsing #
    #####################

    @staticmethod
    def _parse_xi_splines(raw_xip, raw_xim):
        """Convert xi inputs to dicts keyed by (i, j) with i <= j.

        Returns (xip_dict, xim_dict, ntomo). Bin indices must be
        consecutive 0-based integers; the caller is responsible for
        remapping non-consecutive bin IDs before constructing the lists.
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
        """Non-redundant ((a,b),(c,d)) with a<=b, c<=d, (a,b)<=(c,d) lex.

        Sufficient for the symmetric r-terms (rp0, rp1, rm0, rm1) where
        block[(c,d),(a,b)] = block[(a,b),(c,d)]^T.
        """
        combos = []
        for idx1, bp1 in enumerate(self.bp_list):
            for bp2 in self.bp_list[idx1:]:
                combos.append((bp1, bp2))
        return combos

    def _tomo_combos_all(self):
        """All ((a,b),(c,d)) ordered pairs.

        Needed for rpm because Cov_pm is NOT symmetric under bin-pair
        exchange: rpm((a,b),(c,d)) involves xi_+(a,c)×xi_-(b,d), while
        rpm((c,d),(a,b)) involves xi_+(c,a)×xi_-(d,b) with a DIFFERENT
        second Wick pairing.
        """
        return [(bp1, bp2) for bp1 in self.bp_list for bp2 in self.bp_list]

    ###############################################
    # Callback mapping for the integration engine #
    ###############################################

    def _build_tomo_callback_map(self):
        """Build flat arrays describing all tomo callbacks for the C kernel.

        Returns
        -------
        dict with keys:
            xi_a_idx, xi_b_idx : int arrays (n_cb,)
                Index into xi_y_tables for each callback.
            w_ids : int array (n_cb,)
                Weight type id (0-5).
            out_idx : int array (n_cb,)
                Index into the output accumulator.
            wick_fac : float array (n_cb,)
                Wick factor (1.0 or 2.0 for degenerate pairs).
            xi_table_keys : list of (pm, i, j)
                Ordered unique xi spline keys for building xi_y_tables.
            out_keys : list of (rterm_name, (a,b), (c,d))
                Maps output index to (r-term, combo).
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

                # The non-tomo prefactor (2/πA) already includes a factor
                # of 2 for both Wick contractions being equal. When we
                # explicitly sum two non-degenerate contractions, each
                # must contribute half (fac=0.5) so the total = 1.0.
                # For degenerate pairs (single callback), fac=1.0.
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
                            n_workers=None):
        """Compute all tomo r-term blocks in one pass.

        Returns dict ``{(rterm_name, ((a,b),(c,d))): (n_r, n_r)}``.
        """
        if n_workers is None:
            n_workers = self.n_workers
        self.rterms_tomo = self._quadruplet_density_terms_tomo_c(
            verbose=verbose, pair_mask=pair_mask, n_workers=n_workers)
        return self.rterms_tomo

    ###################
    # Shape noise (D) #
    ###################

    def getD_tomo(self):
        """Diagonal shape-noise term per bin pair.

        Returns dict ``{(a,b): (n_r,)}`` — only auto-blocks are nonzero.
        """
        ww_factor = self._w2**2 / self._w1**2 if self._w1 != 0 else 0.
        self.D_tomo = {}
        for bp in self.bp_list:
            a, b = bp
            se4 = self.sigma_eps[a]**2 * self.sigma_eps[b]**2
            if self.method_pairs in ('analytic', 'continuous'):
                Np = (2. * np.pi * self.area
                      * self.nbar[a] * self.nbar[b]
                      * self._w1**2
                      * self.theta_centers * self.dtheta
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
        """Triplet block for one tomo combo with Kronecker-delta selection.

        The triplet term involves a shared apex galaxy between the two
        pairs. Sharing requires a common bin: if pair 1 = (a,b) and
        pair 2 = (c,d), the apex must be in a bin from {a,b} ∩ {c,d}.
        Each valid sharing case contributes the triplet integral with xi
        evaluated at the NON-shared bin pair. The /4 normalisation
        matches the non-tomo convention (which includes all 4 cases).
        σ_ε² and n̄ scaling are taken from the shared (apex) bin.
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
        """All tomo blocks of one mixed term, by its position in QTERMS.

        Diagonal combos of q_++/q_-- are symmetrised; q_+- is not, since it
        contributes to Cov_pm which is NOT symmetric in θ₁ ↔ θ₂.
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
        out = np.zeros((self.n_bp * n, self.n_bp * n))
        for idx1, bp1 in enumerate(self.bp_list):
            if with_D:
                # Diagonal shape noise.
                out[idx1*n:(idx1+1)*n, idx1*n:(idx1+1)*n] += np.diag(self.D_tomo[bp1])
            for idx2, bp2 in enumerate(self.bp_list):
                combo = (bp1, bp2) if bp1 <= bp2 else (bp2, bp1)
                sl1 = slice(idx1 * n, (idx1 + 1) * n)
                sl2 = slice(idx2 * n, (idx2 + 1) * n)
                if combo in q_tomo:
                    out[sl1, sl2] += q_tomo[combo]
                else:
                    out[sl1, sl2] += q_tomo[(combo[1], combo[0])].T
                for rname in rnames:
                    out[sl1, sl2] += self._rterm_block(rname, (bp1, bp2))
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

    def _fine_instance_from_catalog(self, new_tc, new_te, verbose=False):
        """Build a fine-grid tomo instance from this instance's catalogue.

        Re-measures the (cheap) fine pair / triplet counts on
        ``new_tc``/``new_te`` from ``self.catalog`` and reuses this instance's
        already-measured quadruplet density (geometry-only, tomo-independent),
        so the expensive 4PCF is not remeasured. Used by :meth:`subbin_tomo`
        to auto-build the ``slab_instance`` when one is not supplied.
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
        """Resample the tomo covariance to a finer radial binning.

        For each tomo bin-pair combo, interpolates the (n_r, n_r) block
        from the coarse grid to the fine grid, then optionally recomputes
        a near-diagonal slab and the cheap D/q terms on the fine grid.

        Parameters match :meth:`GGCovarianceNoTomo.subbin`, except
        ``slab_instance`` must be a :class:`GGCovariance` (tomo-aware). When
        ``slab_instance`` is None and this instance was built from a
        ``catalog``, the fine instance is auto-built via
        :meth:`_fine_instance_from_catalog` (fine pairs/triplets re-measured,
        coarse quadruplet density reused).

        Returns a new :class:`GGCovariance` on the fine grid with all
        tomo components pre-loaded.
        """
        new_tc = np.asarray(new_theta_centers, dtype=float)
        new_te = np.asarray(new_theta_edges, dtype=float)

        # Auto-build the fine-grid slab instance from the catalogue when one is
        # not supplied (pair/triplet counts are cheap to re-measure; the
        # quadruplet density is reused -- see _fine_instance_from_catalog).
        # Done before computing the coarse r-terms so the still-private q
        # tensor is shared by reference rather than after it moves to SHM.
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

        new_dtheta = np.diff(new_te)
        n_new = len(new_tc)

        tc = self.theta_centers
        dthc = self.dtheta
        x_c = np.log(tc)
        x_new = np.log(new_tc)

        _interp = self._build_block_interpolator(x_c, x_new, n_new)

        ## Interpolate D for all bin pairs
        D_fine = {}
        for bp in self.bp_list:
            D_scaled = self.D_tomo[bp] * dthc
            D_fine[bp] = np.exp(np.interp(
                x_new, x_c, np.log(D_scaled))) / new_dtheta

        ## Interpolate q-terms
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

        ## Interpolate r-terms
        rterms_fine = {}
        symmetric_rterms = {'rp0', 'rp1', 'rm0', 'rm1'}
        for key, block in self.rterms_tomo.items():
            rname, combo = key
            sym = rname in symmetric_rterms
            rterms_fine[key] = _interp(block, sym=sym)

        ## Slab recomputation
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

        ## Full recomputation of cheap terms
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

        ## Build output GGCovariance instance
        # `template=self` keeps the parent's q / triplet grids, integration
        # grids and worker config, so the result stays re-integrable.
        return GGCovariance._from_components(
            theta_centers=new_tc, theta_edges=new_te,
            phi_centers=self.phi_centers, phi_edges=self.phi_edges,
            spline_xip=self.xip, spline_xim=self.xim,
            sigma_eps=self.sigma_eps, nbar=self.nbar, area=self.area,
            weights_mean=self.weights_mean, weights_std=self.weights_std,
            n_workers=self.n_workers, template=self, pairs=new_pairs,
            _xip=self._xip, _xim=self._xim, ntomo=self.ntomo,
            n_bp=self.n_bp, bp_list=self.bp_list, nbar_ref=self.nbar_ref,
            D_tomo=D_fine, qpp_tomo=qpp_fine, qmm_tomo=qmm_fine,
            qpm_tomo=qpm_fine, rterms_tomo=rterms_fine)

    def cov_of_statistic_tomo(self, filter_type, split_components=False,
                              xipm_nbinstarget=None, **kwargs):
        """Transform the tomo Cov(ξ₊, ξ₋) into the covariance of an
        integrated statistic (Map², COSEBIs, pure modes).

        Computes the filter weight matrices once via
        :meth:`~GGCovarianceNoTomo.cov_of_statistic` with
        ``_return_weights=True``, then applies them to each bin-pair
        block directly — avoiding repeated COSEBI root-finding.

        Parameters are the same as :meth:`GGCovarianceNoTomo.cov_of_statistic`,
        plus:

        split_components : bool
            If True, return ``(total, components)`` where *components*
            is a dict keyed by ``'D'``, ``'qpp'``, ``'qmm'``, ``'qpm'``,
            ``'rp0'``, ``'rp1'``, ``'rm0'``, ``'rm1'``, ``'rpm'``,
            ``'mix'`` (sum of q-terms), and ``'sva'`` (sum of r-terms).
            Each value has the same structure as the total dict.
        xipm_nbinstarget : int or None
            Number of ξ_± radial bins on which the statistic is evaluated.
            ``None`` (default) uses this instance's binning (``self.n_r``).
            Otherwise the assembled tomo covariance is first resampled to a
            log-uniform grid of ``xipm_nbinstarget`` bins over the current
            θ range via :meth:`subbin_tomo` (plain interpolation, no slab /
            full recompute), then the statistic is computed on that grid.
            This decouples the statistic's binning from the (typically finer)
            grid used to integrate the covariance.

        Returns
        -------
        dict (or tuple of dicts when ``split_components=True``) with the
        same keys as the parent method, but each value has shape
        (n_stat*n_bp, n_stat*n_bp) where n_stat depends on the filter
        type (number of radii, COSEBI modes, or 3*n_r for puremode).
        """
        # Optionally evaluate the statistic on a different ξ_± binning than the
        # (finer) grid the covariance was integrated on: resample by plain
        # interpolation (no slab / full recompute), then recurse on the target.
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

        ## Per-component decomposition
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
        """Build the (n_xi_unique, xi_n) xi_y_tables for the C kernel.

        Returns (xi_x, xi_y_tables, xi_logx0, xi_inv_dlogx, xi_n),
        all as contiguous float64 arrays.
        """
        return self._xi_tables_from_splines(
            [self._xi(pm, i, j) for pm, i, j in xi_keys],
            lo=self._xi_query_lo(), hi=self._xi_query_hi())

    def _quadruplet_density_terms_tomo_c(self, verbose=False, pair_mask=None,
                                         n_workers=None):
        """C-accelerated tomo r-term integration.

        ``n_workers`` OpenMP threads inside the ``quadruplet_sva_integrate``
        kernel.
        """
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
            verbose=verbose, pair_mask=pair_mask, n_workers=n_workers)

        return {key: (.5 * (block + block.T) if symmetric_mask[oi] else block)
                for oi, (key, block) in enumerate(zip(out_keys, accum))}

    ########################################
    # Continuous tomo equations (C/OpenMP) #
    ########################################

    def get_continuous_tomo_c(self, verbose=False, nr_target=1024, nphi_target=1024):
        """Compute all continuous tomo terms (q + r) via the C/OpenMP kernel.

        Single entry point that computes D, qpp/qmm/qpm, and all 5 r-terms,
        then assembles cov_xipm_tomo.
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

        ## D term (trivial, no C needed)
        self.getD_tomo()

        ## Q terms via C
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
        # Np_ideal is passed to the C kernel but not used by
        # continuous_q_single; provide a dummy (n_r,) array.
        Np_ideal = np.ones(n, dtype=np.float64)

        # The kernel output depends only on (w_type, xi_idx), so integrate each
        # distinct pair once with factor 1 and scatter the per-combo Wick
        # factors in numpy afterwards.
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
                self.n_workers, int(verbose))
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

        ## R terms via C
        if verbose: print("  continuous r-terms (C/OpenMP) ...")
        v = np.asarray(self.varphi1_grid)
        v_half = v[v <= np.pi]
        phi_grid = np.ascontiguousarray(self.phi_grid, dtype=np.float64)
        phi_w = np.ascontiguousarray(_trapz_weights(self.phi_grid), dtype=np.float64)
        v_grid = np.ascontiguousarray(v, dtype=np.float64)
        v_w_arr = np.ascontiguousarray(_periodic_weights(v), dtype=np.float64)
        v_half_c = np.ascontiguousarray(v_half, dtype=np.float64)
        v_half_w = np.ascontiguousarray(_trapz_weights(v_half), dtype=np.float64)
        theta_c = np.ascontiguousarray(self.theta_centers, dtype=np.float64)

        ## Build the r-term work list
        # Every combo gets 5 consecutive output blocks (rp0, rp1, rm0, rm1,
        # rpm), all of which come out of the same Wick pairing. rpm needs the
        # full ordered combo set, so all_combos is used throughout and the
        # redundant symmetric blocks are dropped afterwards.
        rterm_names = ['rp0', 'rp1', 'rm0', 'rm1', 'rpm']
        all_combos = combos_all
        r_out_keys = [(rname, combo)
                      for combo in all_combos for rname in rterm_names]

        # Always emit BOTH Wick pairings (even degenerate). The non-tomo
        # prefactor (2/πA) includes a factor of 2 for both contractions
        # being equal; halving the result compensates for the explicit sum.
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
        r_results = np.zeros(n_r_out * n * n, dtype=np.float64)

        lib.continuous_rterms_tomo(
            n, _ptr(theta_c),
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
            self.n_workers, int(verbose))
        check_clib_error(lib)

        r_results = .5 * r_results.reshape(n_r_out, n, n)
        symmetric = {'rp0', 'rp1', 'rm0', 'rm1'}
        self.rterms_tomo = {}
        for oi, (rname, combo) in enumerate(r_out_keys):
            if rname in symmetric and combo[0] > combo[1]:
                continue
            block = r_results[oi]
            if rname in symmetric and combo[0] == combo[1]:
                block = .5 * (block + block.T)
            self.rterms_tomo[(rname, combo)] = block

        ## Assemble
        if verbose: print("  assembling cov_xipm_tomo ...")
        self.get_cov_xipm_tomo()
        return self.cov_xipm_tomo