import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer

from .multires_structs import (MultiresoCatalog, NavHash, TreeResoParams,
                               build_catalog_struct, build_navhash_struct,
                               build_tree_params_struct)
from .utils import _load_clib, check_clib_error, convertunits

__all__ = ["SphericalMap"]


############################
## CURVED - SKY MAP MAKER ##
############################

class SphericalMap:
    """Aperture mass map of a spin-2 tracer catalog on the curved sky.

    The map is evaluated on a HEALPix grid in NEST ordering whose pixel centers are the aperture
    centers. For each center the estimator computes the aperture mass as

    .. math::
        M_\\mathrm{\\rm ap} + i M_\\times = \\mathrm{supp}(Q)^2
            \\frac{\\sum_g w_g Q(\\vartheta_g^2/R_\\mathrm{ap}^2) (e_t + i e_\\times)_g}
                 {\\sum_g w_g},

    where supp(Q) denotes the support of the aperture filter and we use the geodesic distance for 
    the Q-filter and parallel-transport the shapes to the aperture center.

    Attributes
    ----------
    nside : int
        Healpix resolution of the aperture centers.
    R_ap : float
        Aperture radius, in ``sep_units``.
    filter_form : str, optional
        Filter shape, one of ``"S98"``, ``"C02"``, ``"Sch04"``, ``"PolyExp"``. Defaults to
        ``"C02"``, the exponential filter of Crittenden et al. 2002.
    sep_units : str, optional
        Angular unit of ``R_ap``. Defaults to ``"arcmin"``.
    method : str, optional
        The method to be employed for the estimator. Defaults to ``Tree``.
    tree_nsides : list of int, optional
        Healpix resolutions of the multihash bands; ``tree_nsides[0]`` must be ``0``, indicating
        the discrete band.
    shuffle_pix: int, optional
        Choice of how to define centers of the cells in the spatial hash structure.
        Defaults to ``0``, i.e. position at pixel center of mass.
    rmin_pixsize : float, optional
        The limiting radial distance relative to the resolution of the spatial hash
        after which one switches to the next resolution in the hierarchy. Defaults to ``20``.
    nside_hash : int, optional
        The healpix resolution used for hashing subareas of the patches. Defaults to the smallest 
        nside whose pixels are not larger than half the discrete band's outer edge.
    nthreads: int, optional
        The number of OpenMP threads used within the C kernels. Defaults to ``16``.
    verbosity: int, optional
        The level of verbosity during the computation. Level 0: No verbosity, 1: Progress verbosity
        on python layer, 2: Progress verbosity also on C level, 3: Debug verbosity. Defaults to ``0``.
    Map : numpy.ndarray
        ``Map + i*Mx``, zero outside the footprint.
    norm : numpy.ndarray
        Identity-weighted aperture norm, :math:`\\sum_g w_g`.
    norm_Q : numpy.ndarray
        Q-weighted aperture norm, :math:`\\sum_g w_g Q_g`.
    var : numpy.ndarray
        Shape-noise contribution to per-aperture variance of either component of ``Map``.
    coverage : numpy.ndarray
        Masked area fraction of each aperture, raw and Q-weighted. Follows the mask
        convention of ``FlatDataGrid_2D``: 0 (unmasked) to 1 (fully masked). Apertures
        outside the footprint are never evaluated and are reported as fully masked.
    centers_pix : numpy.ndarray
        Nested pixel indices of the apertures that were evaluated.
    """

    def __init__(self, nside, R_ap, filter_form="C02", sep_units="arcmin",
                 method="Tree", shuffle_pix=0, tree_nsides=None, rmin_pixsize=20,
                 nside_hash=None, nthreads=16, verbosity=0):

        self.nside = int(nside)
        self.npix = 12*self.nside*self.nside
        self.R_ap = float(R_ap)
        self.filter_form = filter_form
        self.sep_units = sep_units
        self.method = method
        self.shuffle_pix = shuffle_pix
        self.rmin_pixsize = rmin_pixsize
        self.nside_hash = nside_hash
        self.nthreads = np.int32(max(1, nthreads))
        self.verbosity = np.int32(verbosity)
        self._verbose_python = verbosity > 0
        self._verbose_c = verbosity > 1
        self._verbose_debug = verbosity > 2

        self.filters_dict = {"S98":0, "C02":1, "Sch04":2, "PolyExp":3}
        self.filters_avail = list(self.filters_dict.keys())
        self.methods_avail = ["Discrete", "Tree"]

        assert(self.nside > 0 and (self.nside & (self.nside-1)) == 0)
        assert(self.R_ap > 0.)
        assert(self.filter_form in self.filters_avail)
        assert(self.method in self.methods_avail)
        assert(self.sep_units in ('rad', 'deg', 'arcmin', 'arcsec'))

        # Init band resolutions or check their consistency
        if self.method == "Discrete":
            self.tree_nsides = np.zeros(1, dtype=np.int64)
        elif tree_nsides is None:
            self.tree_nsides = None
        else:
            self.tree_nsides = np.asarray(tree_nsides, dtype=np.int64)
            assert(self.tree_nsides[0] == 0)
            assert(np.all(self.tree_nsides[1:] > 0))
            assert(np.all(self.tree_nsides[1:] & (self.tree_nsides[1:]-1) == 0))

        self.nbinsz = None
        self.Map = None
        self.norm = None
        self.norm_Q = None
        self.var = None
        self.coverage = None
        self.centers_pix = None

        #############################
        ## Link compiled libraries ##
        #############################
        self.library_path, self.clib = _load_clib()

        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_c128 = ndpointer(np.complex128, flags="C_CONTIGUOUS")

        # Filter support radius, in units of R_ap
        self.clib.getFilterSupp.restype = ct.c_double
        self.clib.getFilterSupp.argtypes = [ct.c_int32]

        # Curved-sky aperture mass map on fixed centers
        self.clib.aperturemassmap_spherical.restype = ct.c_void_p
        self.clib.aperturemassmap_spherical.argtypes = [
            ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), ct.POINTER(TreeResoParams),
            ct.c_double, ct.c_int32,
            p_f64, p_f64, p_f64, ct.c_long,
            p_f64, ct.c_long,
            ct.c_int32, ct.c_int32,
            p_c128, p_f64, p_f64, p_f64, p_f64]

        self.ind_filter = self.filters_dict[self.filter_form]
        self.supp = float(self.clib.getFilterSupp(ct.c_int32(self.ind_filter)))

        self.tree_nresos = 1 if self.tree_nsides is None else int(len(self.tree_nsides))
        self.resoshift_leafs = 0
        self.minresoind_leaf = 0
        self.maxresoind_leaf = self.tree_nresos - 1
        self.batch_membudget_mb = 0

    def _nside_for(self, target_rad):
        """Smallest nside whose pixels are no coarser than ``target_rad``."""
        from healpy import nside2resol
        ns = 1
        while nside2resol(ns) > target_rad and ns < 2**29:
            ns *= 2
        return ns

    # TODO: This only sets [0, dpix]. Should set [0, dpix, 2*dpix, ... 2^nmax*dpix]
    def _auto_tree_nsides(self, nbar_sr, target_occupancy=4., min_occupancy=1.):
        """Estimate good tree nside resolution based on tracer density and binning setup."""
        from healpy import nside2pixarea
        # Get nside estimates based on tree accuracy and nbar of catalog; choose finer one
        rsupp = self.supp*self.R_ap*convertunits(self.sep_units, 'rad')
        nside_accuracy = self._nside_for(rsupp/self.rmin_pixsize)
        nside_density = np.sqrt(np.pi*nbar_sr/(3.*target_occupancy))
        nside_density = 1 << max(0, int(np.floor(np.log2(max(nside_density, 1.)))))
        nside = max(nside_accuracy, nside_density)

        # Check if using tree with nside is useful. If not .process will run Discrete.
        mu = nside2pixarea(nside)*nbar_sr
        if self._verbose_debug:
            print("NOTE: candidate band nside=%i holds %.2f tracers per cell."%(nside, mu))

        if mu < min_occupancy:
            tree_nsides = None
        else:
            np.asarray([0, nside], dtype=np.int64)

        return tree_nsides

    def _bands(self, nbar_sr=None):
        """Get band nsides and radial edges covering ``[0, supp*R_ap]``, edges in degrees."""
        from healpy import nside2resol
        rad2deg = convertunits('rad', 'deg')
        rmax_deg = self.supp*self.R_ap*convertunits(self.sep_units, 'deg')

        # Either get the bands from input or construct some reasonable ones
        tree_nsides = self.tree_nsides
        if tree_nsides is None:
            tree_nsides = self._auto_tree_nsides(nbar_sr)
            if tree_nsides is None:
                tree_nsides = np.zeros(1, dtype=np.int64)
                if self._verbose_python:
                    print("NOTE: Tree approximation provides no speedup for this setup; running discrete.")
            elif self._verbose_python:
                print("NOTE: tree_nsides chosen from the tracer density: %s."%np.array2string(tree_nsides[1:]))

        # Drop the bands whose inner edge already sits beyond the filter support
        redges = [0.]
        nsides = [tree_nsides[0]]
        for ns in tree_nsides[1:]:
            edge = self.rmin_pixsize*nside2resol(int(ns))*rad2deg
            if edge >= rmax_deg:
                break
            redges.append(edge)
            nsides.append(ns)
        redges.append(rmax_deg)
        if len(nsides) < len(tree_nsides) and self._verbose_debug:
            print("NOTE: %i of %i tree_nsides start beyond the filter support (%.4g %s) and "
                  "are unused; the tree has %i band(s)."
                  %(len(tree_nsides)-len(nsides), len(tree_nsides), self.supp*self.R_ap,
                    self.sep_units, len(nsides)))
        # The cell partition of the kernel needs the bands ordered from fine to coarse
        assert(np.all(np.diff(redges) > 0.))

        # Band 0 is partitioned along the cells of band 1, so its nav grid is that of band 1
        # and must not be coarser when set by hand
        nside_hash = self.nside_hash
        if len(nsides) > 1:
            nside_hash = int(nsides[1]) if nside_hash is None else max(int(nside_hash), int(nsides[1]))
        elif nside_hash is None:
            nside_hash = self._nside_for(0.5*redges[1]/rad2deg)

        return np.asarray(nsides, dtype=np.int64), np.asarray(redges, dtype=np.float64), nside_hash

    def _expand_to_output(self, parents, nside_parent):
        """All output pixels sitting below the given nested pixels of a coarser grid."""
        k = 2*(int(self.nside).bit_length() - int(nside_parent).bit_length())
        nchild = 1 << k
        return ((parents.astype(np.int64) << k)[:, None] + np.arange(nchild)).ravel()

    def _centers_from_mask(self, mask):
        """Output pixels that the mask leaves at least partly unmasked."""
        from healpy import npix2nside
        ns_mask = int(npix2nside(len(mask)))
        inside = np.flatnonzero(np.asarray(mask) < 1.)
        if ns_mask <= self.nside:
            return self._expand_to_output(inside, ns_mask)
        # A finer mask than the map: an output pixel counts as covered as soon as any of
        # its children is unmasked
        k = 2*(int(ns_mask).bit_length() - int(self.nside).bit_length())
        occupied = np.zeros(self.npix, dtype=bool)
        occupied[inside >> k] = True
        return np.flatnonzero(occupied)

    def _mask_from_catalog(self, cat, target_per_pixel=10):
        """Binary footprint mask estimated from the tracer positions."""
        from healpy import ang2pix
        theta = (90. - cat.pos2)*np.pi/180.
        phi = (cat.pos1*np.pi/180.)%(2.*np.pi)
        # Get galaxy counts per cell at highest considered mask resolution in nest
        if self.tree_nsides is not None and len(self.tree_nsides) > 1:
            ns = int(np.min(self.tree_nsides[1:]))
        else:
            rsupp = self.supp*self.R_ap*convertunits(self.sep_units, 'rad')
            ns = self._nside_for(rsupp/self.rmin_pixsize)
        counts = np.bincount(ang2pix(ns, theta, phi, nest=True), minlength=12*ns*ns)
        del theta, phi
        # If filled pixels contain less than the number of target galaxies, iteratively
        # coarsen the resolution (using that nest ordering is contiguous)
        while ns > 1 and cat.ngal < target_per_pixel*np.count_nonzero(counts):
            counts = counts.reshape(-1, 4).sum(axis=1)
            ns //= 2
        # Retrieve the mask at the inferred resolution
        if self._verbose_python:
            nocc = np.count_nonzero(counts)
            print("NOTE: no mask given; footprint inferred at nside=%i (%.1f tracers per "
                  "cell, %i cells)."%(ns, cat.ngal/max(nocc, 1), nocc))
        return (counts == 0).astype(np.float64)

    def process(self, cat, dotomo=True, mask=None, centers="footprint",
                target_per_pixel=5., approx_coverage=True):
        """Build the aperture mass map from a spin-2 catalog.

        Parameters
        ----------
        cat : SpinTracerCatalog
            Shape catalog with ``geometry='spherical'``.
        dotomo : bool, optional
            Keep the catalog's tomographic bins apart. Defaults to ``True``.
        mask : array of float, optional
            Dense NEST-ordered HEALPix mask, 0 (unmasked) to 1 (fully masked), at any nside.
            Without it the footprint inferred from the tracers stands in, unless
            ``approx_coverage`` is off.
        centers : str or array of int, optional
            ``"footprint"`` covers the survey area, taken from ``mask`` when one is given and
            otherwise inferred from the tracer positions; ``"all"`` covers the whole sphere.
            An integer array is taken as explicit nested pixel indices.
        target_per_pixel : float, optional
            Mean number of tracers per pixel at which the footprint is resolved when it has
            to be inferred without a mask. Defaults to ``5.``.
        approx_coverage : bool, optional
            With no mask, derive an approximate one from the inferred footprint so that
            ``coverage`` reports the survey edge instead of a flat zero. Set it to ``False``
            to skip the coverage pass entirely. Defaults to ``True``.
        """
        from healpy import npix2nside, nside2pixarea, pix2vec

        assert(cat.geometry == 'spherical')
        assert(getattr(cat, 'spin', None) == 2)

        if not dotomo:
            nbinsz = 1
            zbins_orig = cat.zbins
            cat.zbins = np.zeros(cat.ngal, dtype=np.int32)
        else:
            nbinsz = cat.nbinsz
        self.nbinsz = nbinsz

        # Make sure to not exceed memory in C
        nel = self.npix*nbinsz
        if nel > 2e9:
            raise MemoryError("Output map exceeds memory; would need %.2g elements at nside=%i with "
                              "nbinsz=%i. Lower nside or run one tomographic bin at a time."%(nel, self.nside, nbinsz))
        if nel > 2e8 and self._verbose_debug:
            print("NOTE: SphericalMap outputs hold %.2g elements (%.1f GB)."
                  %(nel, nel*8./2**30))

        ## Aperture centers and footprint ##
        area = None
        if isinstance(centers, str):
            assert(centers in ["footprint", "all"])
            if centers == "all":
                centers_pix = np.arange(self.npix, dtype=np.int64)
            else:
                # Centers always come from a mask; without one the footprint stands in
                footprint = mask if mask is not None else self._mask_from_catalog(
                    cat, target_per_pixel)
                centers_pix = self._centers_from_mask(footprint)
                area = float(np.sum(1. - footprint))*nside2pixarea(
                    int(npix2nside(len(footprint))))
                # Reporting zero coverage everywhere would read as "nothing is masked"; the
                # inferred footprint is a better statement of what is actually known
                if approx_coverage:
                    mask = footprint
        else:
            centers_pix = np.unique(np.asarray(centers, dtype=np.int64))

        # Tracer density drives the automatic choice of reduction cells
        if area is None:
            if mask is not None:
                area = float(np.sum(1. - np.asarray(mask)))*nside2pixarea(
                    int(npix2nside(len(mask))))
            else:
                area = 4.*np.pi
        nbar_sr = cat.ngal/max(area, 1e-30)
        centers_pix = centers_pix.astype(np.int64)
        ncenters = len(centers_pix)
        cvx, cvy, cvz = pix2vec(self.nside, centers_pix, nest=True)
        cvx = np.ascontiguousarray(cvx); cvy = np.ascontiguousarray(cvy)
        cvz = np.ascontiguousarray(cvz)

        ## Multihash bundle and input structs ##
        # Setup multihash; only keep bands within the filter support
        nsides, redges, nside_hash = self._bands(nbar_sr)
        self.tree_nresos = int(len(nsides))
        self.maxresoind_leaf = self.tree_nresos - 1
        mh = cat.multihash_bundle(reso_redges=redges, nsides=nsides, nside_hash=nside_hash,
                                  nthreads=self.nthreads, shuffle=self.shuffle_pix, verbose=self._verbose_python)
        # Debug: Give number of galaxies per band; can be used to check inefficient setup
        if self._verbose_debug and len(mh['ngal_resos']) > 1:
            frac = 100.*np.asarray(mh['ngal_resos'][1:])/float(mh['ngal_resos'][0])
            print("NOTE: reduced bands retain %s of the tracers."
                  %(", ".join("%.0f%%"%f for f in frac)))
        # Build structs to be passed to C
        extra = {'e1_resos':mh['red_e1'], 'e2_resos':mh['red_e2']}
        cat_s, keep_cat = build_catalog_struct(mh, nbinsz, extra=extra)
        cat_s.nresos = int(self.tree_nresos)
        nav_s, keep_nav = build_navhash_struct(mh, cat_obj=cat)
        tree_s, keep_tree = build_tree_params_struct(self, mh)
        # Define mask
        if mask is None:
            mask_arr = np.zeros(1, dtype=np.float64)
            nside_mask = 0
        else:
            mask_arr = np.ascontiguousarray(mask, dtype=np.float64)
            nside_mask = int(npix2nside(len(mask_arr)))
        # Init output for C function
        _Map = np.zeros(nbinsz*ncenters, dtype=np.complex128)
        _norm = np.zeros(nbinsz*ncenters, dtype=np.float64)
        _normQ = np.zeros(nbinsz*ncenters, dtype=np.float64)
        _var = np.zeros(nbinsz*ncenters, dtype=np.float64)
        _cov = np.zeros(2*ncenters, dtype=np.float64)

        # Keep numpy arrays backing ctypes pointer fields alive during the C call.
        # ctypes does not maintain Python references for raw pointers stored in structs.
        _alive = keep_cat + keep_nav + keep_tree   # noqa: F841

        self.clib.aperturemassmap_spherical(
            ct.byref(cat_s), ct.byref(nav_s), ct.byref(tree_s),
            ct.c_double(self.R_ap*convertunits(self.sep_units, 'rad')),
            ct.c_int32(self.ind_filter),
            cvx, cvy, cvz, ct.c_long(ncenters),
            mask_arr, ct.c_long(nside_mask),
            ct.c_int32(int(self.nthreads)), ct.c_int32(int(self._verbose_c)),
            _Map, _norm, _normQ, _var, _cov)
        check_clib_error(self.clib)

        # Allocate output on healpix grid
        # Note that we treat apertures outside footprint as fully covered
        self.centers_pix = centers_pix
        self.Map = np.zeros((nbinsz, self.npix), dtype=np.complex128)
        self.norm = np.zeros((nbinsz, self.npix), dtype=np.float64)
        self.norm_Q = np.zeros((nbinsz, self.npix), dtype=np.float64)
        self.var = np.zeros((nbinsz, self.npix), dtype=np.float64)
        self.coverage = np.ones((2, self.npix), dtype=np.float64)
        self.Map[:, centers_pix] = _Map.reshape((nbinsz, ncenters))
        self.norm[:, centers_pix] = _norm.reshape((nbinsz, ncenters))
        self.norm_Q[:, centers_pix] = _normQ.reshape((nbinsz, ncenters))
        self.var[:, centers_pix] = _var.reshape((nbinsz, ncenters))
        self.coverage[:, centers_pix] = _cov.reshape((2, ncenters))

        if not dotomo:
            cat.zbins = zbins_orig

    def masked(self, field="Map", zbin=0):
        """Copy of ``field`` with ``healpy.UNSEEN`` wherever the aperture holds no galaxies."""
        from healpy import UNSEEN
        arr = getattr(self, field)
        arr = arr[zbin] if arr.ndim == 2 else arr
        out = np.where(self.norm[zbin] > 0., arr, UNSEEN)
        return out

    #################
    ## ARITHMETICS ##
    #################

    def _checkcompat(self, other):
        assert(isinstance(other, SphericalMap))
        assert(self.nside == other.nside)
        assert(self.R_ap == other.R_ap)
        assert(self.sep_units == other.sep_units)
        assert(self.filter_form == other.filter_form)
        assert(self.nbinsz == other.nbinsz)

    def _copy(self):
        new = SphericalMap(self.nside, self.R_ap, filter_form=self.filter_form,
                           sep_units=self.sep_units, method=self.method,
                           tree_nsides=self.tree_nsides, rmin_pixsize=self.rmin_pixsize,
                           nside_hash=self.nside_hash, nthreads=int(self.nthreads),
                           verbosity=int(self.verbosity))
        new.nbinsz = self.nbinsz
        new.centers_pix = self.centers_pix
        new.Map = self.Map.copy()
        new.norm = self.norm.copy()
        new.norm_Q = self.norm_Q.copy()
        new.var = self.var.copy()
        new.coverage = self.coverage.copy()
        return new

    # Add two instance together, i.e. when combining maps for different tomo bins
    def __add__(self, other):
        self._checkcompat(other)
        new = self._copy()
        wtot = self.norm + other.norm
        new.norm = wtot
        new.norm_Q = self.norm_Q + other.norm_Q
        new.Map = np.divide(self.Map*self.norm + other.Map*other.norm, wtot,
                            out=np.zeros_like(self.Map), where=wtot > 0.)
        new.var = np.divide(self.var*self.norm**2 + other.var*other.norm**2, wtot**2,
                            out=np.zeros_like(self.var), where=wtot > 0.)
        new.coverage = np.divide(
            self.coverage*self.norm[0] + other.coverage*other.norm[0], wtot[0],
            out=np.ones_like(self.coverage), where=wtot[0] > 0.)
        new.centers_pix = np.union1d(self.centers_pix, other.centers_pix)
        return new
