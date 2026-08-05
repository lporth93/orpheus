# TODO Reactivate gridded catalog instances?

import ctypes as ct
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path
import glob
from .utils import (get_site_packages_dir, search_file_in_site_package, convertunits,
                    _randomhealpixshift, _check_openmp_runtimes)
from .flat2dgrid import FlatPixelGrid_2D, FlatDataGrid_2D
from .patchutils import gen_cat_patchindices, frompatchindices_preparerot
import sys
import time


__all__ = ["Catalog", "ScalarTracerCatalog", "SpinTracerCatalog"]
    
    
##############################################
## Classes that deal with discrete catalogs ##
##############################################
class Catalog:
    
    r"""Class containing variables and methods of a catalog of tracers.  
    Attributes
    ----------
    pos1: numpy.ndarray
        The :math:`x`-positions of the tracer objects
    pos2: numpy.ndarray
        The :math:`y`-positions of the tracer objects
    weight: numpy.ndarray, optional, defaults to ``None``
        The weights of the tracer objects. If set to ``None`` all weights are assumed to be unity.
    zbins: numpy.ndarray, optional, defaults to ``None``
        The tomographic redshift bins of the tracer objects. If set to ``None`` all zbins are assumed to be zero.
    nbinsz: int
        The number of tomographic bins
    isinner: numpy.ndarray
        A flag signaling whether a tracer is within the interior part of the footprint
    units_pos1: string, defaults to ``None``
        The unit of the :math:`x`-positions, should be in [None, 'rad', 'deg', 'arcmin']. 
        For non-spherical catalogs we auto-set this to None. Spherical catalogs are internally transformed to units of degrees.
    units_pos2: string, defaults to ``None``
        The unit of the :math:`y`-positions, should be in [None, 'rad', 'deg', 'arcmin']. 
        For non-spherical catalogs we auto-set this to None. Spherical catalogs are internally transformed to units of degrees.
    geometry: string, defaults to ``'flat2d'``
        Specifies the topology of the space the points are located in. Should be in ['flat2d', 'spherical', '3dbox'].
    min1: float
        The smallest :math:`x`-value appearing in the catalog
    max1: float
        The largest :math:`x`-value appearing in the catalog
    min2: float
        The smallest :math:`y`-value appearing in the catalog
    max2: float
        The largest :math:`y`-value appearing in the catalog
    len1: float
        The extent of the catalog in :math:`x`-direction.
    len2: float
        The extent of the catalog in :math:`y`-direction.
    hasspatialhash: bool
        Flag on whether a spatial hash structure has been allocated for the catalog
    index_matcher: numpy.ndarray
        Indicates on whether there is a tracer in each of the pixels in the spatial hash.
    
        
    .. note::
        
        The ``zbins`` parameter can also be used for other characteristics of the tracers (i.e. color cuts). 
        As all NPCF correlators automatically build the various tomographic-bin-combinations this can keep
        the code shorter.           
    """
    
    def __init__(self, pos1, pos2, pos3=None, weight=None, zbins=None, isinner=None,
                 units_pos1=None, units_pos2=None, geometry='flat2d',
                 mask=None, zbins_mean=None, zbins_std=None):

        self.pos1 = pos1.astype(np.float64)
        self.pos2 = pos2.astype(np.float64)
        self.pos3 = None if pos3 is None else pos3.astype(np.float64)
        self.weight = weight
        self.zbins = zbins
        self.ngal = len(self.pos1)
        # Allocate weights
        if self.weight is None:
            self.weight = np.ones(self.ngal)
        self.weight = self.weight.astype(np.float64)
        # Require zbins to only contain elements in {0, 1, ..., nbinsz-1}
        if self.zbins is None:
            self.zbins = np.zeros(self.ngal)        
        self.zbins = self.zbins.astype(np.int32)
        self.nbinsz = len(np.unique(self.zbins))
        assert(np.max(self.zbins)-np.min(self.zbins)==self.nbinsz-1)
        self.zbins -= (np.min( self.zbins))
        if isinner is None:
            isinner = np.ones(self.ngal, dtype=np.float64)
        self.isinner = np.asarray(isinner, dtype=np.float64)
        self.units_pos1 = units_pos1
        self.units_pos2 = units_pos2
        self.geometry = geometry
        assert(self.geometry in ['flat2d','spherical','3dbox'])
        if self.geometry in ['flat2d','3dbox']:
            self.units_pos1 = None
            self.units_pos2 = None
        if self.geometry == '3dbox':
            assert(self.pos3 is not None)
            assert(len(self.pos3)==self.ngal)
        if self.geometry == 'spherical':
            assert(self.units_pos1 in ['rad', 'deg', 'arcmin'])
            assert(self.units_pos2 in ['rad', 'deg', 'arcmin'])
            self.pos1 *= convertunits(self.units_pos1, 'deg')
            self.pos2 *= convertunits(self.units_pos2, 'deg')
            self.units_pos1 = 'deg'
            self.units_pos2 = 'deg'
            # Make sure that footprint is contiguous
            # 1) Compute internal distance between tracers
            # 2) Compute distance around the origin
            # 3) If largest distance is internal, i.e. catalog not contiguous
            #    split catalog at this boundary and shift one side by 360 deg
            # Note that this algorithm only works for truly contiguous fields, 
            # but might fail for catalogues consisting of multiple disconnected 
            # (yet contiguous) patches covering the whole range of ra...
            ra_sorted = np.sort(self.pos1)
            diffs = np.diff(ra_sorted)
            wrap_diff = (360.0 - ra_sorted[-1]) + ra_sorted[0]
            if wrap_diff <= np.max(diffs):
                max_gap_idx = np.argmax(diffs)
                split_value = ra_sorted[max_gap_idx]
                self.pos1[self.pos1 > split_value] -= 360
                print('NOTE: Catalog not contiguous, shifted RA coordinates > %.2f deg by -360 deg.'%split_value)

        self.mask = mask
        assert(isinstance(self.mask, FlatDataGrid_2D) or self.mask is None)
        if isinstance(self.mask, FlatDataGrid_2D):
            self.__checkmask()
        assert(np.min(self.isinner) >= 0.)
        assert(np.max(self.isinner) <= 1.)
        assert(len(self.isinner)==self.ngal)
        assert(len(self.pos2)==self.ngal)
        assert(len(self.weight)==self.ngal)
        assert(len(self.zbins)==self.ngal)
        assert(np.min(self.weight)>0.)
        
        self.zbins_mean = zbins_mean
        self.zbins_std = zbins_std
        for _ in [self.zbins_mean, self.zbins_mean]:
            if _ is not None:
                assert(isinstance(_,np.ndarray))
                assert(len(_)==self.nbinsz)
        
        self.min1 = np.min(self.pos1)
        self.min2 = np.min(self.pos2)
        self.max1 = np.max(self.pos1)
        self.max2 = np.max(self.pos2)
        self.len1 = self.max1-self.min1
        self.len2 = self.max2-self.min2
        if self.pos3 is not None:
            self.min3 = np.min(self.pos3)
            self.max3 = np.max(self.pos3)
            self.len3 = self.max3-self.min3
        
        self.spatialhash = None
        self.hasspatialhash = False
        self.index_matcher = None
        self.pixs_galind_bounds = None
        self.pix_gals = None
        self.pix1_start = None
        self.pix1_d = None
        self.pix1_n = None
        self.pix2_start = None
        self.pix2_d = None
        self.pix2_n = None

        self.patchinds = None
        
        self.assign_methods = {"NGP":0, "CIC":1, "TSC":2}
        
        ## Link compiled libraries ##
        # Method that works for LP
        target_path = __import__('orpheus').__file__
        self.library_path = str(Path(__import__('orpheus').__file__).parent.absolute())
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        _check_openmp_runtimes()
        p_c128 = ndpointer(np.complex128, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)
        
        # Assigns a set of tomographic fields over a grid
        # Safely called within 'togrid' function
        self.clib.assign_fields.restype = ct.c_void_p
        self.clib.assign_fields.argtypes = [
            p_f64, p_f64, p_i32, p_f64, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, 
            ct.c_int32, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64)]
        
        # Assigns a set of tomographic fields over a grid
        # Safely called within 'togrid' function
        self.clib.gen_weightgrid2d.restype = ct.c_void_p
        self.clib.gen_weightgrid2d.argtypes = [
            p_f64, p_f64, ct.c_int32, ct.c_int32, 
            ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.float64)]
        
        # Generate pixel --> galaxy mapping
        # Safely called within other wrapped functions
        self.clib.build_spatialhash.restype = ct.c_void_p
        self.clib.build_spatialhash.argtypes = [
            p_f64, p_f64, ct.c_int32, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
            ct.c_int32, ct.c_int32,
            np.ctypeslib.ndpointer(dtype=np.int32)]
        
        self.clib.reducecat.restype = ct.c_void_p
        self.clib.reducecat.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, ct.c_int32,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
            p_f64_nof, p_f64_nof, p_f64_nof, p_f64_nof, p_f64_nof,ct.c_int32]

        # Construct reduced catalog
        self.clib.reducecat_tomo.restype = ct.c_void_p
        self.clib.reducecat_tomo.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, p_i32,
            ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
            p_f64_nof, p_f64_nof, p_f64_nof, p_f64_nof, p_i32, p_f64_nof]


    ### PATCH DECOMPOSITION RELATED METHODS ###
    def topatches(self, npatches, patchextend_deg=2.,other_cats=None,
                  nside_hash=128, verbose=False, method='kmeans_healpix', n_workers=16,
                  kmeanshp_maxiter=1000, kmeanshp_tol=1e-10, kmeanshp_randomstate=42,
                  nside_kmeans=1024,healpix_nside=8):
        r""" Decomposes a full-sky catalog into patches.

        Parameters
        ----------
        npatches: int
            The number of patches the catalog should be decomposed into
        patchextend_deg: float, optional
            The width of the buffer region appended around each patch.
        other_cats: list of ``Catalog`` instances or None, optional
            List of additional catalogs that should share the same patches.
            Defaults to None, i.e. no additional catalogs.
        nside_hash: int
            The healpix resolution used for hashing subareas of the patches.
        verbose: bool
            Flag setting on whether output is printed to the console.
        method: {'kmeans_healpix', 'kmeans_treecorr', 'healpix'}
            Patch-assignment algorithm. See the notes for additional details.
        n_workers: int or None
            Number of parallel worker processes for buffer construction.
            Follows joblib convention: None/-1 --> all CPUs, 1-->sequential.
            Defaults to 16 as I found that this does not produce huge memory overhead
            while still giving decent speedup.
        kmeanshp_maxiter: int
            KMeans maximum iterations (kmeans_healpix method only).
        kmeanshp_tol: float
            KMeans convergence tolerance (kmeans_healpix method only).
        kmeanshp_randomstate: int
            KMeans random seed (kmeans_healpix method only).
        nside_kmeans: int
            Healpix nside for on which the footprint is painted before running 
            the kmeans algorithm. A coarse value will result in a faster runtime
            but less accurate patches.
        healpix_nside: int
            Healpix nside for patch assignment (healpix method only).
       
        .. note::
            If you want to get an estimate of a survey-internal covariance matrix
            choosing ``method='kmeans_healpix'`` is the best option as this method
            aims to construct equal-area-patches. However, applying this method can
            be fairly time consuming. If your primary concern is speed, choosing
            ``method='healpix'`` is most suited as there the patches are predefined
            as healpix pixels. While this choice is optimal for unmasked full-sky
            catalogs it might yield pretty discrepant areas for complex survey 
            geometries.

        .. note::
            If you have different catalogs for which you want to share the patches
            you can call this method one one of them and pass the other catalog
            instances via the ``other_cats`` parameter.

        """
        
        # We are only dealing with a single catalog
        if other_cats is None:
            assert(self.geometry in ['spherical'])
            self.patchinds = gen_cat_patchindices(ra_deg=self.pos1, 
                                                  dec_deg=self.pos2, 
                                                  npatches=npatches, 
                                                  patchextend_arcmin=patchextend_deg*60., 
                                                  nside_hash=nside_hash, 
                                                  verbose=verbose, 
                                                  method=method,
                                                  kmeanshp_maxiter=kmeanshp_maxiter, 
                                                  kmeanshp_tol=kmeanshp_tol,
                                                  kmeanshp_randomstate=kmeanshp_randomstate,
                                                  healpix_nside=healpix_nside,
                                                  nside_kmeans=nside_kmeans,
                                                  n_workers=n_workers)
            
            # When forcing the patches to be healpix pixels the number of non-empty pixels depends on the survey footprint and
            # is computed within the gen_cat_patchindices function. Here we reconstruct this number from the output dict.
            if method=='healpix':
                self.npatches = len(self.patchinds["info"]["patchcenters"])
            else:
                self.npatches = npatches
            

        # We want to create equivalent patches for multiple catalogs
        else:
            # Make sure that each catalog is a child of Catalog and has the same geometry
            # As each spherical catalog per definition has ra/dec in units of degrees, this is sufficient.
            ntracer_tot = self.ngal
            cumngals = np.zeros(2+len(other_cats),dtype=int)
            cumngals[1] = self.ngal
            for elcat, cat in enumerate(other_cats):
                if not isinstance(cat, Catalog):
                    raise ValueError('Each catalog should be inherited from orpheus.Catalog class.')
                if not cat.geometry=='spherical':
                    raise ValueError('Patch decomposition only available for spherical catlogs')
                ntracer_tot += cat.ngal
                cumngals[elcat+2] = ntracer_tot
                                
            # Build a joint catalog collecting all positions of the different catalogs
            jointpos1 = np.zeros(ntracer_tot)
            jointpos2 = np.zeros(ntracer_tot)
            jointweight = np.zeros(ntracer_tot)
            jointpos1[:cumngals[1]] += self.pos1
            jointpos2[:cumngals[1]] += self.pos2
            jointweight[:cumngals[1]] += self.weight
            for elcat, cat in enumerate(other_cats):
                jointpos1[cumngals[elcat+1]:cumngals[elcat+2]] += cat.pos1
                jointpos2[cumngals[elcat+1]:cumngals[elcat+2]] += cat.pos2
                jointweight[cumngals[elcat+1]:cumngals[elcat+2]] += cat.weight
            jointcat = Catalog(pos1=jointpos1, pos2=jointpos2, weight=jointweight, 
                               geometry='spherical', units_pos1='deg',  units_pos2='deg')
            
            # Build patches of joint catalog
            jointcat.topatches(npatches=npatches, 
                               patchextend_deg=patchextend_deg,
                               other_cats=None,
                               nside_hash=nside_hash,
                               verbose=verbose,
                               method=method,
                               kmeanshp_maxiter=kmeanshp_maxiter,
                               kmeanshp_tol=kmeanshp_tol,
                               kmeanshp_randomstate=kmeanshp_randomstate)
            
            # Distribute the patchindices of the joint catalog to the individual instances
            self.patchinds = {}
            self.patchinds['info'] = {}
            self.patchinds['info']['patchextend_deg'] = jointcat.patchinds['info']['patchextend_deg']
            self.patchinds['info']['nside_hash'] = jointcat.patchinds['info']['nside_hash']
            self.patchinds['info']['method'] = jointcat.patchinds['info']['method']
            self.patchinds['info']['kmeanshp_maxiter'] = jointcat.patchinds['info']['kmeanshp_maxiter']
            self.patchinds['info']['kmeanshp_tol'] = jointcat.patchinds['info']['kmeanshp_tol']
            self.patchinds['info']['kmeanshp_randomstate'] = jointcat.patchinds['info']['kmeanshp_randomstate']
            self.patchinds['info']['healpix_nside'] = jointcat.patchinds['info']['healpix_nside']
            self.patchinds['info']['patchcenters'] = jointcat.patchinds['info']['patchcenters']
            self.patchinds['info']['patchareas'] = jointcat.patchinds['info']['patchareas']
            self.patchinds['info']['patch_ngalsinner'] = np.zeros(jointcat.npatches)
            self.patchinds['info']['patch_ngalsouter'] = np.zeros(jointcat.npatches)
            self.patchinds['patches'] = {}
            for elp in range(jointcat.npatches):
                _inds = jointcat.patchinds['patches'][elp]
                seli = (_inds['inner']>=cumngals[0])*(_inds['inner']<cumngals[1])
                selo = (_inds['outer']>=cumngals[0])*(_inds['outer']<cumngals[1])
                self.patchinds['info']['patch_ngalsinner'][elp] = np.sum(seli)
                self.patchinds['info']['patch_ngalsouter'][elp] = np.sum(selo)
                self.patchinds['patches'][elp] = {}
                self.patchinds['patches'][elp]['inner'] = _inds['inner'][seli]
                self.patchinds['patches'][elp]['outer'] = _inds['outer'][selo]
            for elcat, cat in enumerate(other_cats):
                cat.patchinds = {}
                cat.patchinds['info'] = {}
                cat.patchinds['info']['patchextend_deg'] = jointcat.patchinds['info']['patchextend_deg']
                cat.patchinds['info']['nside_hash'] = jointcat.patchinds['info']['nside_hash']
                cat.patchinds['info']['method'] = jointcat.patchinds['info']['method']
                cat.patchinds['info']['kmeanshp_maxiter'] = jointcat.patchinds['info']['kmeanshp_maxiter']
                cat.patchinds['info']['kmeanshp_tol'] = jointcat.patchinds['info']['kmeanshp_tol']
                cat.patchinds['info']['kmeanshp_randomstate'] = jointcat.patchinds['info']['kmeanshp_randomstate']
                cat.patchinds['info']['healpix_nside'] = jointcat.patchinds['info']['healpix_nside']
                cat.patchinds['info']['patchcenters'] = jointcat.patchinds['info']['patchcenters']
                cat.patchinds['info']['patchareas'] = jointcat.patchinds['info']['patchareas']
                cat.patchinds['info']['patch_ngalsinner'] = np.zeros(jointcat.npatches)
                cat.patchinds['info']['patch_ngalsouter'] = np.zeros(jointcat.npatches)
                cat.patchinds['patches'] = {}
                for elp in range(jointcat.npatches):
                    _inds = jointcat.patchinds['patches'][elp]
                    seli = (_inds['inner']>=cumngals[elcat+1])*(_inds['inner']<cumngals[elcat+2])
                    selo = (_inds['outer']>=cumngals[elcat+1])*(_inds['outer']<cumngals[elcat+2])
                    cat.patchinds['info']['patch_ngalsinner'][elp] = np.sum(seli)
                    cat.patchinds['info']['patch_ngalsouter'][elp] = np.sum(selo)
                    cat.patchinds['patches'][elp] = {}
                    cat.patchinds['patches'][elp]['inner'] = _inds['inner'][seli]-cumngals[elcat+1]
                    cat.patchinds['patches'][elp]['outer'] = _inds['outer'][selo]-cumngals[elcat+1]

            # Finalize setting attributes for all instances
            self.npatches = npatches
            for cat in other_cats:
                cat.npatches = npatches
                   
    def _patchind_preparerot(self,  index, rotsignflip=False):

        assert(self.patchinds is not None)
        assert(self.geometry in ['spherical'])

        return frompatchindices_preparerot(index, self.patchinds, self.pos1, self.pos2, rotsignflip)
    

    def build_spatialhash(self, dpix=1., extent=[None, None, None, None]):
        r"""Adds a spatial hashing data structure to the catalog.
        
        Parameters
        ----------
        dpix: float
            The sidelength of each cell of the hash. Defaults to ``1``.
        extent: list, optional
            Sets custom boundaries ``[xmin, xmax, ymin, ymax]`` for the grid. Each element defaults
            to ``None``. Each element equal to ``None`` sets the grid boundary as the smallest value
            fully containing the discrete field tracers.
        
        Note
        ----
        Calling this method (re-)allocates the ``index_matcher``, ``pixs_galind_bounds``, ``pix_gals``,
        ``pix1_start``, ``pix2_start``, ``pix1_n``, ``pix2_n``, ``pix1_d`` and ``pix2_d`` 
        attributes of the instance. 
        """
        
        # Build extent
        if extent[0] is None:
            thismin1 = self.min1
        else:
            thismin1 = extent[0]
            assert(thismin1 <= self.min1)
        if extent[1] is None:
            thismax1 = self.max1
        else:
            thismax1 = extent[1]
            assert(thismax1 >= self.max1)
        if extent[2] is None:
            thismin2 = self.min2
        else:
            thismin2 = extent[2]
            assert(thismin2 <= self.min2)
        if extent[3] is None:
            thismax2 = self.max2
        else:
            thismax2 = extent[3]
            assert(thismax2 >= self.max2)
            
        # Collect arguments
        # Note that the C function assumes the mask to start at zero, that's why we shift
        # the galaxy positions
        self.pix1_start = thismin1 - dpix/1.
        self.pix2_start = thismin2 - dpix/1.
        stop1 = thismax1 + dpix/1.
        stop2 = thismax2 + dpix/1.
        self.pix1_n = int(np.ceil((stop1-self.pix1_start)/dpix))
        self.pix2_n = int(np.ceil((stop2-self.pix2_start)/dpix))
        npix = self.pix1_n * self.pix2_n
        self.pix1_d = (stop1-self.pix1_start)/(self.pix1_n)
        self.pix2_d = (stop2-self.pix2_start)/(self.pix2_n)

        # Compute hashtable
        result = np.zeros(2 * npix + 3 * self.ngal + 1).astype(np.int32)
        self.clib.build_spatialhash(self.pos1, self.pos2, self.ngal,
                                  self.pix1_d, self.pix2_d, 
                                  self.pix1_start, self.pix2_start, 
                                  self.pix1_n, self.pix2_n,
                                  result)

        # Allocate result
        start_isoutside = 0
        start_index_matcher = self.ngal
        start_pixs_galind_bounds = self.ngal + npix
        start_pixs_gals = self.ngal + npix + self.ngal + 1
        start_ngalinpix = self.ngal + npix + self.ngal + 1 + self.ngal
        self.index_matcher = result[start_index_matcher:start_pixs_galind_bounds]
        self.pixs_galind_bounds = result[start_pixs_galind_bounds:start_pixs_gals]
        self.pix_gals = result[start_pixs_gals:start_ngalinpix]
        self.hasspatialhash = True

    ### HIERARCHICAL SPATIAL HASHING RELATED METHODS ###
    def multihash_bundle(self, dpixs=None, dpix_hash=None, normed=True, shuffle=0,
                         extent=[None,None,None,None], forcedivide=1, nthreads=1,
                         reso_redges=None, nsides=None, nside_hash=None, nav_coarsen=None,
                         dpix_z=None, extent_z=[None,None],
                         w2field=False, verbose=False):
        r"""Constructs a hierarchy of reduced catalogs and their associated spatial hashes.

        This method is geometry-aware and each geometry returns a single dict, which can be
        passed to any NPCF estimators. The required tracer fields are derived from the catalog.

        Parameters
        ----------
        dpixs: list, optional
            The pixel sizes on which the hierarchy of reduced catalogs is constructed.
            Required for ``flat2d`` geometry.
        dpix_hash: float, optional, default ``None``
            The size of the pixels used for the spatial hash of the hierarchy of catalogs
            If set to ``None`` uses the largest value of ``dpixs``.
            Considered for ``flat2d`` and ``3dbox`` geometries.
        normed: bool, optional
            Decide on whether to average or to sum the field over pixels. Considered for
            ``flat2d`` geometry. Defaults to ``True``.
        shuffle : int, default ``0``
            How to choose the position of each per-pixel reduced galaxies:

            * 0: weighted centroid of the galaxies in the pixel 
            * 1: a uniformly random position drawn from within the pixel
            * 2: the pixel center
            * 3: the position of one galaxy chosen uniformly at random from
              the galaxies occupying that pixel

            In all cases the reduced-galaxy weight (sum) and isinner flag (any)
            are unchanged; only the position assignment differs.
        extent: list, optional
            Sets custom boundaries ``[xmin, xmax, ymin, ymax]`` for the grid (``flat2d`` only).
            Each element defaults to ``None``. Each element equal to ``None`` sets the grid
            boundary using the smallest bounding box of the discrete field tracers.
        forcedivide: int, optional, default ``1``
            Forces the number of cells in each dimensions to be divisible by some number
            Considered for ``flat2d`` geometry.
        nthreads: int, optional, default ``1``
            Number of threads used when building the reduced catalogs (``flat2d``).
        reso_redges: array, optional, default ``None``
            Radial band edges in degrees (``spherical``).
        nsides: array of int, optional, default ``None``
            Healpix nside for each band's reduced catalogue. ``nsides[r] == 0``
            marks a discrete band. Required for ``spherical`` geometry.
        nside_hash: int, optional, default ``None``
            Navigation pixelisation used for the discrete band (``spherical``). Considered
            for ``spherical`` geometry.
        nav_coarsen: float, optional, default ``None``
            If set, navigate each reduced band on a coarser nested grid whose pixel size
            stays below ``rmax_band/nav_coarsen``. The reduction stays at
            ``nsides[r]``; only the ``query_disc`` navigation coarsens, keeping it cheap at
            large separations..
        verbose: bool, optional, default ``False``
            Flag setting on whether output is printed to the console (``spherical``).
        dpix_z: float, optional, default ``None``
            Slab width along the line of sight. Required for ``3dbox``.
        extent_z: list, optional, default ``[None, None]``
            Sets custom boundaries for line-of-sight ``[zmin, zmax]``. ``None`` entries
            default to the catalog extent.
        w2field: bool, optional, default ``False``
            Also aggregate the squared tracer weights for non-scalar catalogs.

        Returns
        -------
        bundle: dict
            Collection of arrays the NPCF estimators can consume independent of the metric.
        """
        fields = self._multihash_fields(self.geometry, w2field)
        if self.geometry == 'flat2d':
            return self.multihash_flat(dpixs=dpixs, fields=fields, dpix_hash=dpix_hash,
                                       normed=normed, shuffle=shuffle, extent=extent,
                                       forcedivide=forcedivide, nthreads=nthreads)
        elif self.geometry == 'spherical':
            return self.multihash_spherical(reso_redges=reso_redges, nsides=nsides,
                                            nside_hash=nside_hash, shuffle=shuffle,
                                            fields=fields, w2field=w2field,
                                            nav_coarsen=nav_coarsen, verbose=verbose)
        elif self.geometry == '3dbox':
            return self.multihash_slabs(dpix=dpix_hash, dpix_z=dpix_z, fields=fields,
                                        extent=extent, extent_z=extent_z)
        else:
            raise ValueError("Unknown geometry %r" % self.geometry)

    def multihash_flat(self, dpixs, fields, dpix_hash=None, normed=True, shuffle=0,
                  extent=[None,None,None,None], forcedivide=1, nthreads=1):
        r"""Builds spatialhash for a base catalog with geometry ``flat2d`` and its reductions.

        Returns
        -------
        bundle: dict
            Collection of arrays the NPCF estimators can consume independent of the metric.

        Notes
        -----
        The parameters are as documented in :meth:`Catalog.multihash_bundle`.
        """
        
        dpixs = sorted(dpixs)
        if dpix_hash is None:
            dpix_hash = dpixs[-1]
        if extent[0] is None:
            extent = [self.min1-dpix_hash, self.max1+dpix_hash, self.min2-dpix_hash, self.max2+dpix_hash]
            
        
        # Initialize spatial hash for discrete catalog
        self.build_spatialhash(dpix=dpix_hash, extent=extent)
        ngals = [self.ngal]
        isinners = [self.isinner]
        pos1s = [self.pos1]
        pos2s = [self.pos2]
        weights = [self.weight]
        zbins = [self.zbins]
        allfields = [fields]
        if not normed:
            allfields[0] *= self.weight
        index_matchers = [self.index_matcher]
        pixs_galind_bounds = [self.pixs_galind_bounds]
        pix_gals = [self.pix_gals]

        # Build spatial hashes for reduced catalogs 
        fac_pix1 = self.pix1_d/dpix_hash
        fac_pix2 = self.pix2_d/dpix_hash
        dpixs1_true = np.zeros_like(np.asarray(dpixs))
        dpixs2_true = np.zeros_like(np.asarray(dpixs))
        for elreso in range(len(dpixs)):
            dpixs1_true[elreso]=fac_pix1*dpixs[elreso]
            dpixs2_true[elreso]=fac_pix2*dpixs[elreso]
            nextcat, fields_red = self._reduce(fields=fields,
                                               dpix=dpixs1_true[elreso],
                                               dpix2=dpixs2_true[elreso],
                                               relative_to_hash=np.int32(2**(len(dpixs)-elreso-1)),
                                               normed=normed,
                                               shuffle=shuffle,
                                               extent=extent,
                                               forcedivide=forcedivide,
                                               nthreads=nthreads,
                                               ret_inst=True)
            nextcat.build_spatialhash(dpix=dpix_hash, extent=extent)
            ngals.append(nextcat.ngal)
            isinners.append(nextcat.isinner)
            pos1s.append(nextcat.pos1)
            pos2s.append(nextcat.pos2)
            weights.append(nextcat.weight)
            zbins.append(nextcat.zbins)
            allfields.append(fields_red)
            index_matchers.append(nextcat.index_matcher)
            pixs_galind_bounds.append(nextcat.pixs_galind_bounds)
            pix_gals.append(nextcat.pix_gals)

        # Allocate result in standard output structure
        multihash_dict = dict(
            geometry='flat2d',
            ngal=np.int32(ngals[0]),
            nresos=np.int32(len(ngals)-1),
            ngal_resos=np.asarray(ngals, dtype=np.int32),
            pos1s=pos1s, pos2s=pos2s, weights=weights, zbins=zbins, isinners=isinners,
            allfields=allfields, index_matchers=index_matchers,
            pixs_galind_bounds=pixs_galind_bounds, pix_gals=pix_gals,
            dpixs1_true=dpixs1_true, dpixs2_true=dpixs2_true,
            isinner_resos=np.concatenate(isinners).astype(np.float64),
            weight_resos=np.concatenate(weights).astype(np.float64),
            pos1_resos=np.concatenate(pos1s).astype(np.float64),
            pos2_resos=np.concatenate(pos2s).astype(np.float64),
            zbin_resos=np.concatenate(zbins).astype(np.int32),
            index_matcher_resos=np.concatenate(index_matchers).astype(np.int32),
            pixs_galind_bounds_resos=np.concatenate(pixs_galind_bounds).astype(np.int32),
            pix_gals_resos=np.concatenate(pix_gals).astype(np.int32))
        
        return multihash_dict

    def multihash_slabs(self, dpix, dpix_z, fields=None,
                        extent=[None, None, None, None], extent_z=[None, None]):
        r"""Builds spatialhash for a base catalog with geometry ``3dbox`` and its reductions.

        Returns
        -------
        bundle: dict
            Collection of arrays the NPCF estimators can consume independent of the metric.

        Notes
        -----
        The parameters are as documented in :meth:`Catalog.multihash_bundle`.
        """
        assert self.geometry == '3dbox'
        assert self.pos3 is not None

        # Setup shared transverse grid (identical for every slab).
        thismin1 = self.min1 if extent[0] is None else extent[0]
        thismax1 = self.max1 if extent[1] is None else extent[1]
        thismin2 = self.min2 if extent[2] is None else extent[2]
        thismax2 = self.max2 if extent[3] is None else extent[3]
        pix1_start = thismin1 - dpix
        pix2_start = thismin2 - dpix
        stop1 = thismax1 + dpix
        stop2 = thismax2 + dpix
        pix1_n = int(np.ceil((stop1 - pix1_start)/dpix))
        pix2_n = int(np.ceil((stop2 - pix2_start)/dpix))
        npix = pix1_n * pix2_n
        pix1_d = (stop1 - pix1_start)/pix1_n
        pix2_d = (stop2 - pix2_start)/pix2_n

        # Assign galaxies to slabs based on position along los
        zmin = self.min3 if extent_z[0] is None else extent_z[0]
        zmax = self.max3 if extent_z[1] is None else extent_z[1]
        z0 = zmin - dpix_z
        nslabs = int(np.ceil((zmax + dpix_z - z0)/dpix_z))
        slab_id = np.floor((self.pos3 - z0)/dpix_z).astype(np.int32)
        np.clip(slab_id, 0, nslabs-1, out=slab_id)

        # Reorder galaxies so each slab is a contiguous block and get offsets.
        order = np.argsort(slab_id, kind='stable')
        counts = np.bincount(slab_id[order], minlength=nslabs)
        slab_offsets = np.zeros(nslabs+1, dtype=np.int32)
        slab_offsets[1:] = np.cumsum(counts)

        # Discrete resolution is reordered catalog
        pos1 = np.ascontiguousarray(self.pos1[order], dtype=np.float64)
        pos2 = np.ascontiguousarray(self.pos2[order], dtype=np.float64)
        pos3 = np.ascontiguousarray(self.pos3[order], dtype=np.float64)
        weight = np.ascontiguousarray(self.weight[order], dtype=np.float64)
        zbins = np.ascontiguousarray(self.zbins[order], dtype=np.int32)
        fields_red = None
        if fields is not None:
            fields_red = [np.ascontiguousarray(np.asarray(f)[order], dtype=np.float64) for f in fields]

        # For each slab construct one 2D hash over the the shared grid.
        index_matcher = np.full(nslabs*npix, -1, dtype=np.int32)
        rshift_bounds = np.zeros(nslabs, dtype=np.int32)
        pix_gals = np.zeros(self.ngal, dtype=np.int32)
        bounds_list = []
        cum_bounds = 0
        for s in range(nslabs):
            lo, hi = int(slab_offsets[s]), int(slab_offsets[s+1])
            ngal_s = hi - lo
            rshift_bounds[s] = cum_bounds
            if ngal_s == 0:
                bounds_list.append(np.zeros(1, dtype=np.int32))
                cum_bounds += 1
                continue
            p1s = np.ascontiguousarray(pos1[lo:hi])
            p2s = np.ascontiguousarray(pos2[lo:hi])
            result = np.zeros(2*npix + 3*ngal_s + 1, dtype=np.int32)
            self.clib.build_spatialhash(p1s, p2s, np.int32(ngal_s),
                                        pix1_d, pix2_d, pix1_start, pix2_start,
                                        pix1_n, pix2_n, result)
            index_matcher[s*npix:(s+1)*npix] = result[ngal_s:ngal_s+npix]
            bounds_list.append(result[ngal_s+npix:ngal_s+npix+ngal_s+1].copy())
            pix_gals[lo:hi] = result[ngal_s+npix+ngal_s+1:ngal_s+npix+2*ngal_s+1] + lo
            cum_bounds += ngal_s + 1
        pixs_galind_bounds = np.concatenate(bounds_list).astype(np.int32)

        # Allocate result in standard output structure
        bundle = dict(geometry='3dbox', ngal=int(self.ngal), nslabs=int(nslabs),
                      z0=float(z0), dpix_z=float(dpix_z), npix=int(npix),
                      pix1_start=float(pix1_start), pix1_d=float(pix1_d), pix1_n=int(pix1_n),
                      pix2_start=float(pix2_start), pix2_d=float(pix2_d), pix2_n=int(pix2_n),
                      pos1=pos1, pos2=pos2, pos3=pos3, weight=weight, zbins=zbins,
                      slab_offsets=slab_offsets, index_matcher=index_matcher,
                      pixs_galind_bounds=pixs_galind_bounds, rshift_bounds=rshift_bounds,
                      pix_gals=pix_gals)
        if fields_red is not None:
            bundle['fields'] = fields_red

        return bundle

    def multihash_spherical(self, reso_redges, nsides, nside_hash, shuffle=0,
                            fields=None, w2field=False, nav_coarsen=None, verbose=False):
        r"""Builds spatialhash for a base catalog with geometry ``spherical`` and its reductions.

        Returns
        -------
        bundle: dict
            Collection of arrays the NPCF estimators can consume independent of the metric.

        Notes
        -----
        The parameters are as documented in :meth:`Catalog.multihash_bundle`.
        """
        from healpy import ang2pix, pix2ang, pix2vec, query_disc, nside2resol

        if self.geometry != 'spherical':
            raise ValueError("multihash_spherical requires a spherical catalog "
                            "(geometry='spherical').")
        if shuffle not in (0, 1, 2, 3):
            raise ValueError(f"shuffle must be 0, 1, 2, or 3, got {shuffle}")

        reso_redges = np.asarray(reso_redges, dtype=np.float64)
        nsides = np.asarray(nsides, dtype=np.int64)
        nresos = len(nsides)
        assert len(reso_redges) == nresos + 1

        # Some helpers
        deg2rad = np.pi/180.
        ra = self.pos1*deg2rad
        dec = self.pos2*deg2rad
        theta = 0.5*np.pi - dec
        phi = ra%(2.*np.pi)
        cosdec = np.cos(dec)
        sindec = np.sin(dec)
        gvx = cosdec*np.cos(ra)
        gvy = cosdec*np.sin(ra)
        gvz = sindec
        w = self.weight.astype(np.float64)
        isinner = self.isinner.astype(np.float64)
        ngal = self.ngal
        zbins = self.zbins.astype(np.int64)
        nz = int(zbins.max()) + 1 if ngal else 1

        # Add rng (with deterministic seed in case random choices are made)
        rng = np.random.default_rng(seed=self.ngal) if shuffle in (1, 3) else None

        # Optional spin-2 field to aggregate with parallel transport.
        do_shear = fields is not None
        if do_shear:
            e1_full = np.ascontiguousarray(fields[0], dtype=np.float64)
            e2_full = np.ascontiguousarray(fields[1], dtype=np.float64)

        # Init lists that hold multihash
        red_vx, red_vy, red_vz = [], [], []
        red_ra, red_sindec, red_cosdec = [], [], []
        red_w, red_isinner, red_zbin = [], [], []
        red_e1, red_e2, red_weightsq = [], [], []
        ngal_resos = np.zeros(nresos, dtype=np.int64)
        ncells_resos = np.zeros(nresos, dtype=np.int64)
        nside_nav = np.zeros(nresos, dtype=np.int64)
        cell_pix_list, cell_redbounds_list = [], []
        for r in range(nresos):
            ns_red = int(nsides[r])
            ns_nav = nside_hash if ns_red==0 else ns_red
            nside_nav[r] = ns_nav

            # Discrete band: Reduced galaxies are the galaxies themselves
            if ns_red == 0:
                rvx, rvy, rvz = gvx, gvy, gvz
                rra, rsdec, rcdec = ra, sindec, cosdec
                rw, ris, rz = w, isinner, zbins
                red_navpix = ang2pix(ns_nav, theta, phi, nest=True)
                if do_shear:
                    re1, re2 = e1_full, e2_full
                    if w2field:
                        rwsq = w*w
            else:
                # Paint galaxies to grid and get unique filled indices (one per z-bin)
                gpix = ang2pix(ns_red, theta, phi, nest=True)
                key = gpix*nz + zbins
                occ_key, inv = np.unique(key, return_inverse=True)
                nocc = len(occ_key)
                sw = np.bincount(inv, weights=w, minlength=nocc)
                sis = np.bincount(inv, weights=isinner, minlength=nocc)
                pix_for_group = occ_key // nz
                # Now do aggregation based on shuffle convention
                if shuffle == 0:
                    sx = np.bincount(inv, weights=w*gvx, minlength=nocc)
                    sy = np.bincount(inv, weights=w*gvy, minlength=nocc)
                    sz = np.bincount(inv, weights=w*gvz, minlength=nocc)
                    norm = np.sqrt(sx*sx + sy*sy + sz*sz)
                    norm[norm == 0] = 1.
                    rvx, rvy, rvz = sx/norm, sy/norm, sz/norm
                    rsdec = rvz
                    rcdec = np.sqrt(np.maximum(0., 1.-rvz*rvz))
                    rra = np.arctan2(rvy, rvx)%(2.*np.pi)
                elif shuffle == 1:
                    theta_s, phi_s = _randomhealpixshift(ns_red, pix_for_group, rng)
                    dec_s = 0.5*np.pi - theta_s
                    rra = phi_s
                    rcdec = np.cos(dec_s)
                    rsdec = np.sin(dec_s)
                    rvx = rcdec * np.cos(rra)
                    rvy = rcdec * np.sin(rra)
                    rvz = rsdec
                elif shuffle == 2:
                    theta_c, phi_c = pix2ang(ns_red, pix_for_group, nest=True)
                    dec_c = 0.5*np.pi - theta_c
                    rra = phi_c
                    rcdec = np.cos(dec_c)
                    rsdec = np.sin(dec_c)
                    rvx = rcdec * np.cos(rra)
                    rvy = rcdec * np.sin(rra)
                    rvz = rsdec
                elif shuffle == 3:
                    rand_key = rng.random(len(inv))
                    order = np.lexsort((rand_key, inv))
                    sorted_inv = inv[order]
                    first_idx = np.searchsorted(sorted_inv, np.arange(nocc))
                    chosen = order[first_idx]
                    rvx, rvy, rvz = gvx[chosen], gvy[chosen], gvz[chosen]
                    rra, rsdec, rcdec = ra[chosen], sindec[chosen], cosdec[chosen]
                else:
                    raise NotImplementedError("Only shuffle conventions 0,1,2,3 are implemented.")

                rw = sw
                ris = (sis > 0).astype(np.float64)
                red_navpix = pix_for_group
                rz = (occ_key%nz).astype(np.int64)

                # Parallel transport the shapes to position of reduced galaxy
                if do_shear:
                    tra = rra[inv]; tsd = rsdec[inv]; tcd = rcdec[inv]
                    dlam = ra - tra
                    phi_cj = np.arctan2(tcd*sindec - tsd*cosdec*np.cos(dlam),  cosdec*np.sin(dlam))
                    dlam_r = tra - ra
                    phi_jc = np.arctan2(cosdec*tsd - sindec*tcd*np.cos(dlam_r), tcd*np.sin(dlam_r))
                    gshear = w * (e1_full + 1j*e2_full) * np.exp(2j*((phi_cj+np.pi) - phi_jc))
                    sw_safe = np.where(sw==0., 1., sw)
                    re1 = np.bincount(inv, weights=gshear.real, minlength=nocc) / sw_safe
                    re2 = np.bincount(inv, weights=gshear.imag, minlength=nocc) / sw_safe
                    if w2field:
                        rwsq = np.bincount(inv, weights=w*w, minlength=nocc)

            # Obtain the main hashing arrays. This is kind of equivalent to what we do
            # in the flat case with (index_matcher, pixs_galind_bounds, pix_gals). 
            # To allow sparsity by only caring about the filled pixels, we need to argsort;
            # the sparsity has the advantage that the memory does not blow up when choosing
            # a large nside for a catalog on a small footprint
            red_navpix = np.asarray(red_navpix)
            # Optionally navigate reduced bands on a coarser nested grid than the reduction,
            # so query_disc stays cheap at large separations. The reduced galaxies keep their
            # ns_red positions; the nav cell is just their nested parent pixel.
            if ns_red != 0 and nav_coarsen is not None:
                rmax_rad = reso_redges[r+1] * deg2rad
                ns_c = ns_red
                while ns_c>1 and nside2resol(ns_c//2) <= rmax_rad/nav_coarsen:
                    ns_c//= 2
                if ns_c < ns_red:
                    red_navpix = red_navpix >> (2*(int(ns_red).bit_length() - int(ns_c).bit_length()))
                    nside_nav[r] = ns_c
            order = np.argsort(red_navpix, kind='stable')
            cell_pix, cell_counts = np.unique(red_navpix[order], return_counts=True)
            ncells = len(cell_pix)
            cell_redbounds = np.zeros(ncells+1, dtype=np.int64)
            np.cumsum(cell_counts, out=cell_redbounds[1:])
            rvx, rvy, rvz = rvx[order], rvy[order], rvz[order]
            rra, rsdec, rcdec = rra[order], rsdec[order], rcdec[order]
            rw, ris, rz = rw[order], ris[order], rz[order]

            # Allocate the per-band bookkeeping
            n_red = len(rvx)
            ngal_resos[r] = n_red
            ncells_resos[r] = ncells
            red_vx.append(rvx); red_vy.append(rvy); red_vz.append(rvz)
            red_ra.append(rra); red_sindec.append(rsdec); red_cosdec.append(rcdec)
            red_w.append(rw); red_isinner.append(ris); red_zbin.append(rz)
            if do_shear:
                red_e1.append(re1[order]); red_e2.append(re2[order])
                if w2field:
                    red_weightsq.append(rwsq[order])
            cell_pix_list.append(cell_pix.astype(np.int64))
            cell_redbounds_list.append(cell_redbounds)

            if verbose:
                print(f"  band {r}: nside_red={ns_red} nside_nav={ns_nav} "
                    f"nreds={n_red} ncells={ncells} shuffle={shuffle}")

        # Concatenate with per-band rshift offsets
        def _cat(arrs, dtype):
            return np.concatenate(arrs).astype(dtype) if arrs else np.empty(0, dtype=dtype)
        
        rshift_red = np.zeros(nresos+1, dtype=np.int64)
        np.cumsum(ngal_resos, out=rshift_red[1:])
        rshift_cellpix = np.zeros(nresos+1, dtype=np.int64)
        np.cumsum(ncells_resos, out=rshift_cellpix[1:])
        rshift_cellbounds = np.zeros(nresos+1, dtype=np.int64)
        np.cumsum(ncells_resos+1, out=rshift_cellbounds[1:])
        
        # Flag reduced bands whose navigation was coarsened below the reduction. 
        # Some functions reusenside_nav for cross-reso reduction hierarchy so they
        # complain early on by asserting on this flag
        nav_coarsened = bool(np.any((nsides > 0) & (nside_nav < nsides)))

        # Allocate result in standard output structure
        bundle = dict(
            geometry='spherical',
            nresos=np.int32(nresos),
            ngal=np.int32(ngal),
            nav_coarsened=nav_coarsened,
            ngal_resos=ngal_resos.astype(np.int32),
            ncells_resos=ncells_resos.astype(np.int32),
            nside_nav=nside_nav.astype(np.int64),
            reso_redges=(reso_redges * deg2rad).astype(np.float64),
            red_vx=_cat(red_vx, np.float64), red_vy=_cat(red_vy, np.float64),
            red_vz=_cat(red_vz, np.float64), red_ra=_cat(red_ra, np.float64),
            red_sindec=_cat(red_sindec, np.float64), red_cosdec=_cat(red_cosdec, np.float64),
            red_w=_cat(red_w, np.float64), red_isinner=_cat(red_isinner, np.float64),
            red_zbin=_cat(red_zbin, np.int32),
            rshift_red=rshift_red.astype(np.int32),
            cell_pix=_cat(cell_pix_list, np.int64),
            cell_redbounds=_cat(cell_redbounds_list, np.int32),
            rshift_cellpix=rshift_cellpix.astype(np.int32),
            rshift_cellbounds=rshift_cellbounds.astype(np.int32),
            cen_vx=gvx, cen_vy=gvy, cen_vz=gvz,
            cen_ra=ra, cen_sindec=sindec, cen_cosdec=cosdec,
            cen_w=w, cen_isinner=isinner,
        )
        if do_shear:
            bundle['red_e1'] = _cat(red_e1, np.float64)
            bundle['red_e2'] = _cat(red_e2, np.float64)
            if w2field:
                bundle['red_weightsq'] = _cat(red_weightsq, np.float64)

        return bundle

    def _reduce(self, fields, dpix, dpix2=None, relative_to_hash=None, normed=True, shuffle=0,
               extent=[None,None,None,None], forcedivide=1, nthreads=1,
               ret_inst=False):
        r"""Paints a catalog onto a grid with equal-area cells. The galaxies do not have to
        reside in the pixel centers.
        
        Parameters
        ----------
        fields: list
            The fields to be painted to the grid. Each field is given as a 1D array of float.
        dpix: float
            The sidelength of a grid cell.  
        dpix2: float, optional
            The sidelength of a grid cell in :math:`y`-direction. Defaults to ``None``. 
            If set to ``None`` the pixels are assumed to be squares.
        relative_to_hash: int, optional
            Forces the cell size to be an integer multiple of the cell size of the spatial hash. 
            Defaults to ``None``. If set to ``None`` the pixelsize is unrelated to the cell
            size of the spatial hash.
        normed: bool, optional
            Decide on whether to average or to sum the field over pixels. Defaults to ``True``.
        shuffle: int, optional
            Choose a definition on how to set the central point of each pixel. Defaults to zero.
        extent: list, optional
            Sets custom boundaries ``[xmin, xmax, ymin, ymax]`` for the grid. Each element defaults
            to ``None``. Each element equal to ``None`` sets the grid boundary as the smallest value
            fully containing the discrete field tracers.
        forcedivide: int, optional
            Forces the number of cells in each dimensions to be divisible by some number. 
            Defaults to ``1``.
        ret_inst: bool, optional
            Decides on whether to return the output as a list of arrays containing the reduced catalog or
            on returning a new ``Catalog`` instance. Defaults to ``False``.
        """
        
        # Initialize grid
        if relative_to_hash is None: 
            if dpix2 is None:
                dpix2 = dpix
            start1, start2, n1, n2 = self._gengridprops(dpix, dpix2, forcedivide, extent)
        else:
            assert(self.hasspatialhash)
            assert(isinstance(relative_to_hash,np.int32))
            start1 = self.pix1_start
            start2 = self.pix2_start
            dpix = self.pix1_d/np.float64(relative_to_hash)
            dpix2 = self.pix2_d/np.float64(relative_to_hash)
            n1 = self.pix1_n*relative_to_hash
            n2 = self.pix2_n*relative_to_hash
        
        # Prepare arguments
        zbinarr = self.zbins.astype(np.int32)
        nbinsz = len(np.unique(zbinarr))
        ncompfields = []
        scalarquants = []
        nfields = 0
        for field in fields:
            if type(field[0].item()) is float:
                scalarquants.append(field)
                nfields += 1
                ncompfields.append(1)
            if type(field[0].item()) is complex:
                scalarquants.append(field.real)
                scalarquants.append(field.imag)
                nfields += 2
                ncompfields.append(2)
        scalarquants = np.asarray(scalarquants)
        
        # Compute reduction for all zbins. Note that the outuput arrays have
        # extent of the upper bound ngal; all excess values will never be
        # allocated and filtered out by the sel_nonzero filter lateron.
        assert(shuffle in [True, False, 0, 1, 2, 3, 4])
        isinner_red = np.zeros(self.ngal, dtype=np.float64)
        w_red = np.zeros(self.ngal, dtype=np.float64)
        pos1_red = np.zeros(self.ngal, dtype=np.float64)
        pos2_red = np.zeros(self.ngal, dtype=np.float64)
        zbins_red = np.zeros(self.ngal, dtype=np.int32)
        scalarquants_red = np.zeros(nfields*self.ngal, dtype=np.float64)
        self.clib.reducecat_tomo(self.isinner.astype(np.float64),
                                 self.weight.astype(np.float64),
                                 self.pos1.astype(np.float64),
                                 self.pos2.astype(np.float64),
                                 scalarquants.flatten().astype(np.float64), zbinarr,
                                 self.ngal, nfields, nbinsz, np.int32(normed),
                                 dpix, dpix2, start1, start2, n1, n2, np.int32(shuffle), np.int32(nthreads),
                                 isinner_red, w_red, pos1_red, pos2_red, zbins_red, scalarquants_red)
        scalarquants_red = scalarquants_red.reshape((nfields, self.ngal))

        # Accumulate reduced catalog
        sel_nonzero = w_red>0
        isinner_red = isinner_red[sel_nonzero]
        w_red = w_red[sel_nonzero]
        pos1_red = pos1_red[sel_nonzero]
        pos2_red = pos2_red[sel_nonzero]
        zbins_red = zbins_red[sel_nonzero]
        scalarquants_red = scalarquants_red[:,sel_nonzero]
        fields_red = []
        tmpcomp = 0
        for elf in range(len(fields)):
            if ncompfields[elf]==1:
                fields_red.append(scalarquants_red[tmpcomp])
            if ncompfields[elf]==2:
                fields_red.append(scalarquants_red[tmpcomp]+1J*scalarquants_red[tmpcomp+1])
            tmpcomp += ncompfields[elf]

        if ret_inst:
            return Catalog(pos1=pos1_red, pos2=pos2_red, weight=w_red, zbins=zbins_red,
                           isinner=isinner_red.astype(np.float64)), fields_red
            
        return w_red, pos1_red, pos2_red, zbins_red, isinner_red, fields_red
    
    
    def _jointextent(self, others, extend=0):
        r"""Draws largest possible rectangle over set of catalogs.
        
        Parameters
        ----------
        others: list
            Contains ``Catalog`` instances over which the joint extent will
            be drawn
        extend: float, optional
            Include an additional boundary layer around the joint extent
            of the catalogs. Defaults to ``0`` (no extension).
            
        Returns
        -------
        xlo: float
            The lower ``x``-boundary of the joint extent.
        xhi: float
            The upper ``x``-boundary of the joint extent.
        ylo: float
            The lower ``y``-boundary of the joint extent.
        yhi: float
            The upper ``y``-boundary of the joint extent.
        
        """
        for other in others:
            assert(isinstance(other, Catalog))
        
        xlo = self.min1
        xhi = self.max1
        ylo = self.min2
        yhi = self.max2
        for other in others:
            xlo = min(xlo, other.min1)
            xhi = max(xhi, other.max1)
            ylo = min(ylo, other.min2)
            yhi = max(yhi, other.max2)
        
        return (xlo-extend, xhi+extend, ylo-extend, yhi+extend)

    
    def create_mask(self, method="Basic", pixsize=1., apply=False, extend=0.):

        assert(method in ["Basic", "Density", "Random"])

        if method=="Basic":
            npix_1 = int(np.ceil((self.max1-self.min1)/pixsize))
            npix_2 = int(np.ceil((self.max2-self.min2)/pixsize))
            self.mask = FlatDataGrid_2D(np.zeros((npix_2,npix_1), dtype=np.float64), 
                                        self.min1, self.min2, pixsize, pixsize)
        if method=="Density":
            start1, start2, n1, n2 = self._gengridprops(pixsize, pixsize)
            reduced = self.togrid(dpix=pixsize,method="NGP",fields=[], tomo=False)
            mask = (reduced[0].reshape((n2,n1))==0).astype(np.float64)
            self.mask = FlatDataGrid_2D(mask, start1, start2, pixsize, pixsize)
            
        # Add a masked buffer region around enclosing rectangle
        if extend>0.:
            npix_ext = int(np.ceil(extend/pixsize))
            extstart1 = self.mask.start_1 - npix_ext*pixsize
            extstart2 = self.mask.start_2 - npix_ext*pixsize
            extmask = np.ones((self.mask.npix_2+2*npix_ext, self.mask.npix_1+2*npix_ext))
            extmask[npix_ext:-npix_ext,npix_ext:-npix_ext] = self.mask.data
            self.mask = FlatDataGrid_2D(extmask, extstart1, extstart2, pixsize, pixsize)

        self. __checkmask()
        
        self. __applymask(apply)
        
    def __checkmask(self):
        assert(self.mask.start_1 <= self.min1)
        assert(self.mask.start_2 <= self.min2)
        assert(self.mask.pix1_lbounds[-1] >= self.max1-self.mask.dpix_1)
        assert(self.mask.pix2_lbounds[-1] >= self.max2-self.mask.dpix_2)
        
    def __applymask(self, method):
        assert(method in [False, True, "WeightsOnly"])
        
        
    # Maps catalog to grid
    def togrid(self, fields, dpix, normed=False, weighted=True, tomo=True,
               extent=[None,None,None,None], method="CIC", forcedivide=1, 
               asgrid=None, nthreads=1):
        r"""Paints a catalog of discrete tracers to a grid.
        
        Parameters
        ----------
        fields: list
            The fields to be painted to the grid. Each field is given as a 1D array of float.
        dpix: float
            The sidelength of a grid cell.  
        normed: bool, optional
            Decide on whether to average or to sum the field over pixels. Defaults to ``False``.
        weighted: bool, optional
            Whether to apply the tracer weights of the catalog. Defaults to ``True``.
        extent: list, optional
            Sets custom boundaries ``[xmin, xmax, ymin, ymax]`` for the grid. Each element defaults
            to ``None``. Each element equal to ``None`` sets the grid boundary as the smallest value
            fully containing the discrete field tracers.
        method: str, optional
            The chosen mass assignment method applied to each of the fields. Currently supported methods
            are ``NGP``, ``CIC`` and ``TSC`` assignment. Defaults to ``CIC``.
        forcedivide: int, optional
            Forces the number of cells in each dimensions to be divisible by some number. 
            Defaults to ``1``.
        asgrid: bool, optional
            Deprecated.
        nthreads: int, optional
            The number of openmp threads used for the reduction procedure. Defaults to ``1``.

        Returns
        -------
        projectedfields: list
            A list of the 2D arrays containing the reduced fields
        start1: float
            The :math:`x`-position of the first columns' left edge
        start2: float
            The :math:`y`-position of the first rows' lower edge
        dpix: float
            The sidelength of each pixel in the grid. Note that this
            value might slightly differ from the one provided in the parameters.
        normed: bool
            Same as the ``normed`` parameter
        method: str
            Same as the ``method`` parameter
        """
        
        if asgrid is not None:
            raise NotImplementedError
        
        # Choose index of method for c wrapper
        assert(method in ["NGP", "CIC", "TSC"])
        elmethod = self.assign_methods[method]
        start1, start2, n1, n2 = self._gengridprops(dpix, dpix, forcedivide, extent)
        
        # Prepare arguments
        zbinarr = self.zbins.astype(np.int32)
        if not tomo:
            zbinarr = np.zeros_like(zbinarr)
        nbinsz = len(np.unique(zbinarr))
        nfields = len(fields)
        if not weighted:
            weightarr = np.ones(self.ngal, dtype=np.float64)
        else:
            weightarr = self.weight.astype(np.float64)
        fieldarr = np.zeros(nfields*self.ngal, dtype=np.float64)
        for _ in range(nfields):
            fieldarr[_*self.ngal:(1+_)*self.ngal] = fields[_]
            
        # Call wrapper and reshape output to (zbins, nfields, size_field)
        proj_shape = (nbinsz, (nfields+1), n2, n1)
        projectedfields = np.zeros((nbinsz*(nfields+1)*n2*n1), dtype=np.float64)
        self.clib.assign_fields(self.pos1.astype(np.float64), 
                                          self.pos2.astype(np.float64),
                                          zbinarr, weightarr, fieldarr,
                                          nbinsz, nfields, self.ngal,
                                          elmethod, start1, start2, dpix, 
                                          n1, n2, nthreads, projectedfields)
        projectedfields = projectedfields.reshape(proj_shape)
        if normed:
            projectedfields[:,1:] = np.nan_to_num(projectedfields[:,1:]/projectedfields[:,0])
            
        return projectedfields, start1, start2, dpix, normed, method
    
    def gen_weightgrid2d(self, dpix, 
                         extent=[None,None,None,None], method="CIC", forcedivide=1, 
                         nthreads=1):
        
        # Choose index of method for c wrapper
        assert(method in ["NGP", "CIC", "TSC"])
        elmethod = self.assign_methods[method]
        start1, start2, n1, n2 = self._gengridprops(dpix, dpix, forcedivide, extent)
        
        self.ngal
        nsubs = 2*elmethod+1
        pixinds = np.zeros(nsubs*nsubs*self.ngal, dtype=np.int32)
        pixweights = np.zeros(nsubs*nsubs*self.ngal, dtype=np.float64)
        self.clib.gen_weightgrid2d(self.pos1.astype(np.float64), 
                                             self.pos2.astype(np.float64),
                                             self.ngal, elmethod,
                                             start1, start2, dpix, n1, n2,
                                             nthreads, pixinds, pixweights)
        return pixinds, pixweights
        

    def _multihash_fields(self, geometry, w2field):
        r"""Tracer fields this catalog contributes to its multihash reduction.

        The base catalog carries no tracer field (e.g. random catalogs), so it
        returns ``None``. Tracer subclasses override this to return their own
        field arrays in the layout the given ``geometry`` expects.
        """
        return None

    def _gengridprops(self, dpix, dpix2=None, forcedivide=1, extent=[None,None,None,None]):
        r"""Gives some basic properties of grids created from the discrete tracers.
        
        Parameters
        ----------
        dpix: float
            The sidelength of a grid cell.  
        dpix2: float, optional
            The sidelength of a grid cell in :math:`y`-direction. Defaults to ``None``. 
            If set to ``None`` the pixels are assumed to be squares.
        forcedivide: int, optional
            Forces the number of cells in each dimensions to be divisible by some number. 
            Defaults to ``1``.
        extent: list, optional
            Sets custom boundaries ``[xmin, xmax, ymin, ymax]`` for the grid. Each element defaults
            to ``None``. Each element equal to ``None`` sets the grid boundary as the smallest value
            fully containing the discrete field tracers.
            
        Returns
        -------
        start1, start2: float
            The :math:`x`/:math:`y`-position of the first column.
        n1, n2: int
            The number of pixels in the :math:`x`/:math:`y`-position.
        """
        
        # Define inner extent of the grid
        fixedsize = False
        if extent[0] is not None:
            fixedsize = True
        if extent[0] is None:
            thismin1 = self.min1
        else:
            thismin1 = extent[0]
            assert(thismin1 <= self.min1)
        if extent[1] is None:
            thismax1 = self.max1
        else:
            thismax1 = extent[1]
            assert(thismax1 >= self.max1)
        if extent[2] is None:
            thismin2 = self.min2
        else:
            thismin2 = extent[2]
            assert(thismin2 <= self.min2)
        if extent[3] is None:
            thismax2 = self.max2
        else:
            thismax2 = extent[3]
            assert(thismax2 >= self.max2)

        if dpix2 is None:
            dpix2 = dpix
            
        # Add buffer to grid and get associated pixelization
        if not fixedsize:
            start1 = thismin1 - 4*dpix
            start2 = thismin2 - 4*dpix2
            n1 = int(np.ceil((thismax1+4*dpix - start1)/dpix))
            n2 = int(np.ceil((thismax2+4*dpix2 - start2)/dpix2))
            n1 += (forcedivide - n1%forcedivide)%forcedivide
            n2 += (forcedivide - n2%forcedivide)%forcedivide
        else:
            start1=extent[0]
            start2=extent[2]
            n1 = int((thismax1-thismin1)/dpix)
            n2 = int((thismax2-thismin2)/dpix2)
            assert(not n1%forcedivide)
            assert(not n2%forcedivide)
            
        return start1, start2, n1, n2


class ScalarTracerCatalog(Catalog):
    r"""Catalog containg scalar (spin-0) tracers.
        
    Attributes
    ----------
    pos1: numpy.ndarray
        The :math:`x`-positions of the tracer objects
    pos2: numpy.ndarray
        The :math:`y`-positions of the tracer objects
    tracer: numpy.ndarray
        The values of the scalar tracer field, i.e. galaxy weights or cosmic convergence.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`Catalog`.
    Additional child-specific parameters can be passed via ``kwargs``. 
    """
    
    def __init__(self, pos1, pos2, tracer, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracer = tracer
        self.spin = 0
        
    def reduce(self, dpix, dpix2=None, relative_to_hash=None, normed=True, shuffle=0,
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        r"""Paints the catalog onto a grid with equal-area cells
        
        Parameters
        ----------
        dpix: float
            The sidelength of a grid cell.  
        dpix2: float, optional
            The sidelength of a grid cell in :math:`y`-direction. Defaults to ``None``. 
            If set to ``None`` the pixels are assumed to be squares.
        relative_to_hash: int, optional
            Forces the cell size to be an integer multiple of the cell size of the spatial hash. 
            Defaults to ``None``. If set to ``None`` the pixelsize is unrelated to the cell
            size of the spatial hash.
        normed: bool, optional
            Decide on whether to average or to sum the field over pixels. Defaults to ``True``.
        shuffle: int, optional
            Choose a definition on how to set the central point of each pixel. Defaults to zero.
        extent: list, optional
            Sets custom boundaries ``[xmin, xmax, ymin, ymax]`` for the grid. Each element defaults
            to ``None``. Each element equal to ``None`` sets the grid boundary as the smallest value
            fully containing the discrete field tracers.
        forcedivide: int, optional
            Forces the number of cells in each dimensions to be divisible by some number. 
            Defaults to ``1``.
        ret_inst: bool, optional
            Decides on whether to return the output as a list of arrays containing the reduced catalog or
            on returning a new ``Catalog`` instance. Defaults to ``False``.
        """
        res = super()._reduce(
            dpix=dpix,
            dpix2=None, 
            relative_to_hash=None, 
            fields=[self.tracer], 
            normed=normed, 
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide,
            ret_inst=False)
        (w_red, pos1_red, pos2_red, zbins_red, isinner_red, fields_red) = res
        if ret_inst:
            return ScalarTracerCatalog(self.spin, pos1_red, pos2_red, 
                                       fields_red[0], 
                                       weight=w_red, zbins=zbins_red, isinner=isinner_red)
        return res
    
    def _multihash_fields(self, geometry, w2field):
        r"""The scalar tracer field, reduced only along the flat path."""
        return [self.tracer] if geometry == 'flat2d' else None
    
    
    def frompatchind(self, index):

        prepare = super()._patchind_preparerot(index, rotsignflip=False)
        inds_extpatch, patch_isinner, rotangle, ra_rot, dec_rot, rotangle_polars = prepare

        patchcat = ScalarTracerCatalog(
            pos1=ra_rot*60.,
            pos2=dec_rot*60.,
            tracer=self.tracer[inds_extpatch],
            weight=self.weight[inds_extpatch],
            zbins=self.zbins[inds_extpatch],
            isinner=patch_isinner,
            units_pos1='arcmin',
            units_pos2='arcmin',
            geometry='flat2d',
            mask=None,
            zbins_mean=None,
            zbins_std=None)
        
        return patchcat
        
class SpinTracerCatalog(Catalog):
    r"""Catalog containg polar (spin-2) tracers.
        
    Attributes
    ----------
    pos1: numpy.ndarray
        The :math:`x`-positions of the tracer objects
    pos2: numpy.ndarray
        The :math:`y`-positions of the tracer objects
    tracer_1: numpy.ndarray
            The values of the real part of the tracer field, i.e. galaxy ellipticities.
    tracer_2: numpy.ndarray
        The values of the imaginary part of the tracer field, i.e. galaxy ellipticities.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`Catalog`.
    Additional child-specific parameters can be passed via ``kwargs``. 
    """
    
    def __init__(self, spin, pos1, pos2, tracer_1, tracer_2, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracer_1 = tracer_1.astype(np.float64)
        self.tracer_2 = tracer_2.astype(np.float64)
        self.spin = int(spin)
        
    def reduce(self, dpix, dpix2=None, relative_to_hash=None, normed=True, shuffle=0,
               extent=[None,None,None,None], forcedivide=1, w2field=True,
               ret_inst=False):
        r"""Paints the catalog onto a grid with equal-area cells
        
        Parameters
        ----------
        dpix: float
            The sidelength of a grid cell.  
        dpix2: float, optional
            The sidelength of a grid cell in :math:`y`-direction. Defaults to ``None``. 
            If set to ``None`` the pixels are assumed to be squares.
        relative_to_hash: int, optional
            Forces the cell size to be an integer multiple of the cell size of the spatial hash. 
            Defaults to ``None``. If set to ``None`` the pixelsize is unrelated to the cell
            size of the spatial hash.
        normed: bool, optional
            Decide on whether to average or to sum the field over pixels. Defaults to ``True``.
        shuffle: int, optional
            Choose a definition on how to set the central point of each pixel. Defaults to zero.
        extent: list, optional
            Sets custom boundaries ``[xmin, xmax, ymin, ymax]`` for the grid. Each element defaults
            to ``None``. Each element equal to ``None`` sets the grid boundary as the smallest value
            fully containing the discrete field tracers.
        forcedivide: int, optional
            Forces the number of cells in each dimensions to be divisible by some number. 
            Defaults to ``1``.
        w2field: bool, optional
            Adds an additional field equivalent to the squared weight of the tracers to the reduced 
            catalog. Defaults to ``True``.
        ret_inst: bool, optional
            Decides on whether to return the output as a list of arrays containing the reduced catalog or
            on returning a new ``Catalog`` instance. Defaults to ``False``.
        """
        
        if not w2field:
            fields=(self.tracer_1, self.tracer_2,) 
        else:
            fields=(self.tracer_1, self.tracer_2, self.weight**2, )
        res = super()._reduce(
            dpix=dpix, 
            dpix2=None, 
            relative_to_hash=None, 
            fields=fields, 
            normed=normed,
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide,
            ret_inst=False)
        (w_red, pos1_red, pos2_red, zbins_red, isinner_red, fields_red) = res
        if ret_inst:
            return SpinTracerCatalog(spin=self.spin, pos1=pos1_red, pos2=pos2_red, 
                                     tracer_1=fields_red[0], tracer_2=fields_red[1], 
                                     weight=w_red, zbins=zbins_red, isinner=isinner_red)
        return res
    
    def _multihash_fields(self, geometry, w2field):
        r"""The two spin components; the flat path also packs ``weight**2``."""
        if geometry == 'flat2d' and w2field:
            return (self.tracer_1, self.tracer_2, self.weight**2)
        return (self.tracer_1, self.tracer_2)
    
    
    def frompatchind(self, index, rotsignflip=False):

        prepare = super()._patchind_preparerot(index, rotsignflip=rotsignflip)
        inds_extpatch, patch_isinner, rotangle, ra_rot, dec_rot, rotangle_polars = prepare
        spintracer_rot = (self.tracer_1[inds_extpatch] + 1j*self.tracer_2[inds_extpatch])*rotangle_polars

        patchcat = SpinTracerCatalog(
            spin=self.spin,
            pos1=ra_rot*60.,
            pos2=dec_rot*60.,
            tracer_1=spintracer_rot.real,
            tracer_2=spintracer_rot.imag,
            weight=self.weight[inds_extpatch],
            zbins=self.zbins[inds_extpatch],
            isinner=patch_isinner,
            units_pos1='arcmin',
            units_pos2='arcmin',
            geometry='flat2d',
            mask=None)
        
        return patchcat