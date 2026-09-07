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
    """Aperture mass map of a spin-2 tracer catalog on a curved sky.

    The map lives on a HEALPix grid in NEST ordering whose pixel centers are the aperture
    centers. For each center the estimator gathers every galaxy inside the filter support
    and computes the aperture mass as

    .. math::
        M_\\mathrm{ap} + i M_\\times = \\mathrm{supp}(Q)^2
            \\frac{\\sum_g w_g Q(\\vartheta_g^2/R_\\mathrm{ap}^2) (e_t + i e_\\times)_g}
                 {\\sum_g w_g},

    using the geodesic distance for the Q-filter and parallel-transporting the shapes to the
    aperture center.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the aperture centers and of all output maps.
    R_ap : float
        Aperture radius, in ``sep_units``.
    filter_form : str, optional
        Filter shape, one of ``"S98"``, ``"C02"``, ``"Sch04"``, ``"PolyExp"``. Defaults to
        ``"C02"``, the exponential filter of Crittenden et al. 2002.
    sep_units : str, optional
        Angular unit of ``R_ap``. Defaults to ``"arcmin"``.
    method : str, optional
        ``"Discrete"`` gathers every galaxy at full resolution, ``"Tree"`` replaces the outer
        part of the aperture by the reduced tracers of the multihash bands. 
    tree_nsides : list of int, optional
        Healpix resolutions of the multihash bands; ``tree_nsides[0]`` must be ``0``, marking
        the discrete band. Ignored for ``method="Discrete"``. If left unset, the bands are
        chosen at ``process`` time from the tracer density.
    rmin_pixsize : float, optional
        A band of cells of size ``reso`` starts at ``rmin_pixsize*reso``. Defaults to ``20``.
    nside_hash : int, optional
        Navigation resolution of the discrete band. Defaults to the smallest nside whose
        pixels are no larger than half the discrete band's outer edge.
    nthreads : int, optional
        Number of OpenMP threads. Defaults to ``16``.
    verbosity : int, optional
        ``0`` silent, ``1`` python-level, ``2`` C-level progress bars.

    Attributes
    ----------
    Map : array of complex, shape (nbinsz, npix)
        ``Map + i*Mx``, zero outside the footprint.
    norm : array of float, shape (nbinsz, npix)
        Identity-weighted aperture norm, :math:`\\sum_g w_g`.
    norm_Q : array of float, shape (nbinsz, npix)
        Q-weighted aperture norm, :math:`\\sum_g w_g Q_g`.
    coverage : array of float, shape (2, npix)
        Masked area fraction of each aperture, raw and Q-weighted. Follows the mask
        convention of ``FlatDataGrid_2D``: 0 (unmasked) to 1 (fully masked). Apertures
        outside the footprint are never evaluated and are reported as fully masked.
    centers_pix : array of int
        Nested pixel indices of the apertures that were evaluated.
    """

    def __init__(self, nside, R_ap, filter_form="C02", sep_units="arcmin",
                 method="Tree", tree_nsides=None, rmin_pixsize=20,
                 nside_hash=None, nthreads=16, verbosity=0):

        self.nside = int(nside)
        self.npix = 12*self.nside*self.nside
        self.R_ap = float(R_ap)
        self.filter_form = filter_form
        self.sep_units = sep_units
        self.method = method
        self.rmin_pixsize = rmin_pixsize
        self.nside_hash = nside_hash
        self.nthreads = np.int32(max(1, nthreads))
        self.verbosity = np.int32(verbosity)
        self._verbose_python = verbosity > 0
        self._verbose_c = verbosity > 1

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
            p_c128, p_f64, p_f64, p_f64]

        self.ind_filter = self.filters_dict[self.filter_form]
        self.supp = float(self.clib.getFilterSupp(ct.c_int32(self.ind_filter)))

        # Init attributes consumed by build_tree_params_struct.
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

    def _auto_tree_nsides(self, nbar_sr, target_occupancy=4., min_occupancy=1.):
        """Reduction resolution inferred from the tracer density.

        Two lower bounds meet: a band of cells of size ``reso`` starts only at
        ``rmin_pixsize*reso``, and thinning saturates once a cell holds ``target_occupancy``
        tracers. Returns ``None`` when the resulting cell merges next to nothing, in which
        case the caller runs discrete.
        """
        from healpy import nside2pixarea
        rsupp = self.supp*self.R_ap*convertunits(self.sep_units, 'rad')
        ns_accuracy = self._nside_for(rsupp/self.rmin_pixsize)
        ns_density = np.sqrt(np.pi*nbar_sr/(3.*target_occupancy))
        ns_density = 1 << max(0, int(np.floor(np.log2(max(ns_density, 1.)))))
        ns = max(ns_accuracy, ns_density)

        mu = nside2pixarea(ns)*nbar_sr
        if self._verbose_python:
            print("NOTE: candidate band nside=%i holds %.2f tracers per cell."%(ns, mu))
        if mu < min_occupancy:
            return None
        return np.asarray([0, ns], dtype=np.int64)

    def _bands(self, nbar_sr=None):
        """Band nsides and radial edges covering ``[0, supp*R_ap]``, edges in degrees.

        The edges must tile the support without a gap: the kernel never visits a separation
        below ``reso_redges[0]``.
        """
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
                    print("NOTE: at this tracer density no reduction cell both thins the "
                          "catalogue and covers useful radius; running discrete.")
            elif self._verbose_python:
                print("NOTE: tree_nsides chosen from the tracer density: %s."
                      %np.array2string(tree_nsides[1:]))

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
        if len(nsides) < len(tree_nsides) and self._verbose_python:
            print("NOTE: %i of %i tree_nsides start beyond the filter support (%.4g %s) and "
                  "are unused; the tree has %i band(s)."
                  %(len(tree_nsides)-len(nsides), len(tree_nsides), self.supp*self.R_ap,
                    self.sep_units, len(nsides)))

        nside_hash = self.nside_hash
        if nside_hash is None:
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

    def _mask_from_catalog(self, cat, target_per_pixel):
        """Binary footprint mask inferred from the tracer positions.

        Two caps set the resolution. The coverage pass walks every mask pixel of every
        aperture, so the cells are kept no finer than the reduction cells of the tree, whose
        size is ``rsupp/rmin_pixsize``; the mask then never carries more pixels per aperture
        than the kernel carries tracers. A cell holding of order one tracer would also let
        Poisson holes punch spurious gaps through the interior, so the grid is coarsened until
        it averages ``target_per_pixel`` tracers. Holes below the surviving resolution are
        invisible; supply ``mask`` whenever the footprint is known.
        """
        from healpy import ang2pix
        theta = (90. - cat.pos2)*np.pi/180.
        phi = (cat.pos1*np.pi/180.)%(2.*np.pi)

        rsupp = self.supp*self.R_ap*convertunits(self.sep_units, 'rad')
        ns = self._nside_for(rsupp/self.rmin_pixsize)
        counts = np.bincount(ang2pix(ns, theta, phi, nest=True), minlength=12*ns*ns)
        del theta, phi
        # Nested children are contiguous, so coarsening is a sum over blocks of four
        while ns > 1 and cat.ngal < target_per_pixel*np.count_nonzero(counts):
            counts = counts.reshape(-1, 4).sum(axis=1)
            ns //= 2
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

        # The dense output maps dominate the footprint; flag before allocating them
        nel = 4*self.npix*nbinsz
        if nel > 2e9:
            raise MemoryError("SphericalMap outputs would need %.2g elements at nside=%i with "
                              "nbinsz=%i, above the 2e9 contract. Lower nside or run one "
                              "tomographic bin at a time."%(nel, self.nside, nbinsz))
        if nel > 2e8 and self._verbose_python:
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
        # _bands drops any band starting beyond the filter support, so the struct band count
        # follows the bundle rather than the requested tree_nsides
        nsides, redges, nside_hash = self._bands(nbar_sr)
        self.tree_nresos = int(len(nsides))
        self.maxresoind_leaf = self.tree_nresos - 1
        mh = cat.multihash_bundle(reso_redges=redges, nsides=nsides, nside_hash=nside_hash,
                                  nthreads=self.nthreads, verbose=self._verbose_python)
        # Whether the tree can pay for its extra hashing passes is decided by how far the
        # reduced bands actually thin the catalogue, which depends on the tracer density and
        # on the band cell sizes. Retentions near 100% mean the bands cost a full pass each
        # and buy nothing, so report them rather than leaving it to be guessed.
        if self._verbose_python and len(mh['ngal_resos']) > 1:
            frac = 100.*np.asarray(mh['ngal_resos'][1:])/float(mh['ngal_resos'][0])
            print("NOTE: reduced bands retain %s of the tracers."
                  %(", ".join("%.0f%%"%f for f in frac)))

        extra = {'e1_resos':mh['red_e1'], 'e2_resos':mh['red_e2']}
        cat_s, keep_cat = build_catalog_struct(mh, nbinsz, extra=extra)
        cat_s.nresos = int(self.tree_nresos)
        nav_s, keep_nav = build_navhash_struct(mh, cat_obj=cat)
        tree_s, keep_tree = build_tree_params_struct(self, mh)

        if mask is None:
            mask_arr = np.zeros(1, dtype=np.float64)
            nside_mask = 0
        else:
            mask_arr = np.ascontiguousarray(mask, dtype=np.float64)
            nside_mask = int(npix2nside(len(mask_arr)))

        _Map = np.zeros(nbinsz*ncenters, dtype=np.complex128)
        _norm = np.zeros(nbinsz*ncenters, dtype=np.float64)
        _normQ = np.zeros(nbinsz*ncenters, dtype=np.float64)
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
            _Map, _norm, _normQ, _cov)
        check_clib_error(self.clib)

        ## Scatter the compact results onto the dense grid ##
        self.centers_pix = centers_pix
        self.Map = np.zeros((nbinsz, self.npix), dtype=np.complex128)
        self.norm = np.zeros((nbinsz, self.npix), dtype=np.float64)
        self.norm_Q = np.zeros((nbinsz, self.npix), dtype=np.float64)
        # Apertures outside the footprint are never evaluated. Leaving them at zero would
        # claim they are fully unmasked, the opposite of the truth, and would let them pass
        # the usual `coverage < threshold` selection; they start fully masked instead.
        self.coverage = np.ones((2, self.npix), dtype=np.float64)
        self.Map[:, centers_pix] = _Map.reshape((nbinsz, ncenters))
        self.norm[:, centers_pix] = _norm.reshape((nbinsz, ncenters))
        self.norm_Q[:, centers_pix] = _normQ.reshape((nbinsz, ncenters))
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
        new.coverage = np.divide(
            self.coverage*self.norm[0] + other.coverage*other.norm[0], wtot[0],
            out=np.ones_like(self.coverage), where=wtot[0] > 0.)
        new.centers_pix = np.union1d(self.centers_pix, other.centers_pix)
        return new
