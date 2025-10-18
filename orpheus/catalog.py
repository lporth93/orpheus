# TODO Reactivate gridded catalog instances?

import ctypes as ct
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path
import glob
from .utils import get_site_packages_dir, search_file_in_site_package, convertunits
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
        A flag signaling wheter a tracer is within the interior part of the footprint
    units_pos1: string, defaults to ``None``
        The unit of the :math:`x`-positions, should be in [None, 'rad', 'deg', 'arcmin']. 
        For non-spherical catalogs we auto-set this to None. Spherical catalogs are internally transformed to units of degrees.
    units_pos2: string, defaults to ``None``
        The unit of the :math:`y`-positions, should be in [None, 'rad', 'deg', 'arcmin']. 
        For non-spherical catalogs we auto-set this to None. Spherical catalogs are internally transformed to units of degrees.
    geometry: string, defualts to ``'flat2d'``
        Specifies the topology of the space the points are located in. Should be in ['flat2d', 'spherical'].
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
        Flag on wheter a spatial hash structure has been allocated for the catalog
    index_matcher: numpy.ndarray
        Indicates on whether there is a tracer in each of the pixels in the spatial hash.
    
        
    .. note::
        
        The ``zbins`` parameter can also be used for other characteristics of the tracers (i.e. color cuts).            
    """
    
    def __init__(self, pos1, pos2, weight=None, zbins=None, isinner=None, 
                 units_pos1=None, units_pos2=None, geometry='flat2d',
                 mask=None, zbins_mean=None, zbins_std=None):        
        
        self.pos1 = pos1.astype(np.float64)
        self.pos2 = pos2.astype(np.float64)
        self.weight = weight
        self.zbins = zbins
        self.ngal = len(self.pos1)
        # Allocate weights
        if self.weight is None:
            self.weight = np.ones(self.ngal)
        self.weight = self.weight.astype(np.float64)
        #self.weight /= np.mean(self.weight)
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
        assert(self.geometry in ['flat2d','spherical'])
        if self.geometry == 'flat2d':
            self.units_pos1 = None
            self.units_pos2 = None
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
        
        self.spatialhash = None # Check whether needed not in docs
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
        # Method that works for RR (but not for LP with a local HPC install)
        #self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"orpheus_clib"))
        #self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        #print(self.library_path)
        #self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        #self.library_path = str(Path(__file__).parent.absolute()) + "/src/"
        #self.clib = ct.CDLL(self.library_path + "clibrary.so")
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

    def topatches(self, npatches=None, area_patch_deg2_target=None, patchextend_deg=2.,other_cats=None,
                  nside_hash=128,  verbose=False, method='kmeans_healpix',
                  kmeanshp_maxiter=1000, kmeanshp_tol=1e-10, kmeanshp_randomstate=42,healpix_nside=8):
        r""" Decomposes a full-sky catalog into patches.
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
                                                  healpix_nside=healpix_nside
                                                  )
            if method=='healpix':
                self.npatches = 12*healpix_nside*healpix_nside
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

    # Reduces catalog to smaller catalog where positions & quantities are
    # averaged over regular grid
    def _reduce(self, fields, dpix, dpix2=None, relative_to_hash=None, normed=True, shuffle=0,
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        r"""Paints a catalog onto a grid with equal-area cells
        
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
            Decides on wheter to return the output as a list of arrays containing the reduced catalog or
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
        
        # Compute reduction (individually for each zbin)
        assert(shuffle in [True, False, 0, 1, 2, 3, 4])
        isinner_red = np.zeros(self.ngal, dtype=np.float64)
        w_red = np.zeros(self.ngal, dtype=np.float64)
        pos1_red = np.zeros(self.ngal, dtype=np.float64)
        pos2_red = np.zeros(self.ngal, dtype=np.float64)
        zbins_red = np.zeros(self.ngal, dtype=np.int32)
        scalarquants_red = np.zeros((nfields, self.ngal), dtype=np.float64)
        ind_start = 0
        for elz in range(nbinsz):
            sel_z = zbinarr==elz
            ngal_z = np.sum(sel_z)
            ngal_red_z = 0
            red_shape = (len(fields), ngal_z)
            isinner_red_z = np.zeros(ngal_z, dtype=np.float64)
            w_red_z = np.zeros(ngal_z, dtype=np.float64)
            pos1_red_z = np.zeros(ngal_z, dtype=np.float64)
            pos2_red_z = np.zeros(ngal_z, dtype=np.float64)
            scalarquants_red_z = np.zeros(nfields*ngal_z, dtype=np.float64)
            self.clib.reducecat(self.isinner[sel_z].astype(np.float64), 
                                self.weight[sel_z].astype(np.float64), 
                                self.pos1[sel_z].astype(np.float64), 
                                self.pos2[sel_z].astype(np.float64),
                                scalarquants[:,sel_z].flatten().astype(np.float64),
                                ngal_z, nfields, np.int32(normed),
                                dpix, dpix2, start1, start2, n1, n2, np.int32(shuffle),
                                isinner_red_z, w_red_z, pos1_red_z, pos2_red_z, scalarquants_red_z, ngal_red_z)
            isinner_red[ind_start:ind_start+ngal_z] = isinner_red_z
            w_red[ind_start:ind_start+ngal_z] = w_red_z
            pos1_red[ind_start:ind_start+ngal_z] = pos1_red_z
            pos2_red[ind_start:ind_start+ngal_z] = pos2_red_z
            zbins_red[ind_start:ind_start+ngal_z] = elz*np.ones(ngal_z, dtype=np.int32)
            scalarquants_red[:,ind_start:ind_start+ngal_z] = scalarquants_red_z.reshape((nfields, ngal_z))
            ind_start += ngal_z
            
        # Accumulate reduced atalog
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
        #isinner_red[isinner_red<0.5] = 0  
        #isinner_red[isinner_red>=0.5] = 1  
        if ret_inst:
            return Catalog(pos1=pos1_red, pos2=pos2_red, weight=w_red, zbins=zbins_red,
                           isinner=isinner_red.astype(np.float64)), fields_red
            
        return w_red, pos1_red, pos2_red, zbins_red, isinner_red, fields_red
    
    def _multihash(self, dpixs, fields, dpix_hash=None, normed=True, shuffle=0,
                  extent=[None,None,None,None], forcedivide=1):
        r"""Builds spatialhash for a base catalog and its reductions.
        
        Parameters
        ----------
        dpixs: list
            The pixel sizes on which the hierarchy of reduced catalogs is constructed.
        fields: list
            The fields for which the multihash is constructed. Each field is given as a 1D array of float.
        dpix_hash: float, optional
            The size of the pixels used for the spatial hash of the hierarchy of catalogs. Defaults
            to ``None``. If set to ``None`` uses the largest value of ``dpixs``.
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
            
        Returns
        -------
        ngals: list
            Contains the number of galaxies for each of the catalogs in the hierarchy.
        pos1s: list
            Contains the :math:`x`-positions for each of the catalogs in the hierarchy.
        pos2s: list
            Contains the :math:`y`-positions for each of the catalogs in the hierarchy.
        weights: list
            Contains the tracer weights for each of the catalogs in the hierarchy.
        zbins: list
            Contains the tomographic redshift bins for each of the catalogs in the hierarchy.
        isinners: list
            Contains the flag on wheter a tracer is within the interior part of the footprint
            for each of the catalogs in the hierarchy.
        allfields: list
            Contains the tracer fields for each of the catalogs in the hierarchy.
        index_matchers: list
            Contains the ``index_matchers`` arrays for each of the catalogs in the hierarchy.
            See the ```index_matcher`` attribute for more information.
        pixs_galind_bounds: list
            Contains the ``pixs_galind_bounds`` arrays for each of the catalogs in the hierarchy.
            See the ```pixs_galind_bounds`` attribute for more information.
        pix_gals: list
            Contains the ``pix_gals`` arrays for each of the catalogs in the hierarchy.
            See the ```pix_gals`` attribute for more information.
        dpixs1_true: list
            Contains final values of the pixel sidelength along the :math:`x`-direction for each
            of the catalogs in the hierarchy.
        dpixs2_true: list
            Contains final values of the pixel sidelength along the :math:`y`-direction for each
            of the catalogs in the hierarchy.
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
        #print(len(fields),fields)
        for elreso in range(len(dpixs)):
            #print("Doing reso %i"%elreso)
            dpixs1_true[elreso]=fac_pix1*dpixs[elreso]
            dpixs2_true[elreso]=fac_pix2*dpixs[elreso]
            #print(dpixs[elreso], dpixs1_true[elreso], dpixs2_true[elreso], len(self.pos1))
            nextcat, fields_red = self._reduce(fields=fields,
                                               dpix=dpixs1_true[elreso], 
                                               dpix2=dpixs2_true[elreso],
                                               relative_to_hash=np.int32(2**(len(dpixs)-elreso-1)),
                                               #relative_to_hash=None,
                                               normed=normed, 
                                               shuffle=shuffle,
                                               extent=extent, 
                                               forcedivide=forcedivide, 
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
            
        return ngals, pos1s, pos2s, weights, zbins, isinners, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true
    
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
               asgrid=None, nthreads=1, ret_inst=False):
        r"""Paints a catalog of discrete tracers to a grid.
        
        Parameters
        ----------
        fields: list
            The fields to be painted to the grid. Each field is given as a 1D array of float.
        dpix: float
            The sidelength of a grid cell.  
        normed: bool, optional
            Decide on whether to average or to sum the field over pixels. Defaults to ``True``.
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
        ret_inst: bool, optional
            Decides on wheter to return the output as a list of arrays containing the reduced catalog or
            on returning a new ``Catalog`` instance. Defaults to ``False``.
        asgrid: bool, optional
            Deprecated.
        nthreads: int, optional
            The number of openmp threads used for the reduction procedure. Defaults to ``1``.
        ret_inst: bool, optional
            Deprecated.
            
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
            
        if not ret_inst:
            return projectedfields, start1, start2, dpix, normed, method
        
        return GriddedCatalog(projectedfields, 
                              start1, start2, dpix, normed, method)
    
    def gen_weightgrid2d(self, dpix, 
                         extent=[None,None,None,None], method="CIC", forcedivide=1, 
                         nthreads=1):
        
        # Choose index of method for c wrapper
        assert(method in ["NGP", "CIC", "TSC"])
        elmethod = self.assign_methods[method]
        start1, start2, n1, n2 = self._gengridprops(dpix, dpix, forcedivide, extent)
        
        #void gen_weightgrid2d(
        #    double *pos1, double *pos2, int ngal, int method,
        #    double min1, double min2, int dpix, int n1, int n2,
        #    int nthreads, int *pixinds, double *pixweights){
        
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
        start1: float
            The :math:``x``-position of the first column.
        start2: float
            The :math:``y``-position of the first row.
        n1: int
            The number of pixels in the :math:``x``-position. 
        n2: int
            The number of pixels in the :math:``y``-position. 
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
    r"""Class constructor.
        
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
            Decides on wheter to return the output as a list of arrays containing the reduced catalog or
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
    
    def multihash(self, dpixs, dpix_hash=None, normed=True, shuffle=0,
                  extent=[None,None,None,None], forcedivide=1):
        r"""Builds spatialhash for a base catalog and its reductions. 
        
        Parameters
        ----------
        dpixs: list
            The pixel sizes on which the hierarchy of reduced catalogs is constructed.
        dpix_hash: float, optional
            The size of the pixels used for the spatial hash of the hierarchy of catalogs. Defaults
            to ``None``. If set to ``None`` uses the largest value of ``dpixs``.
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
            
        Returns
        -------
        res: tuple
            Contains the output of the ```Catalog._multihash method```
        """
        res = super()._multihash(
            dpixs=dpixs.astype(np.float64), 
            fields=[self.tracer], 
            dpix_hash=dpix_hash,
            normed=normed, 
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide)
        return res
    
    
    def frompatchind(self, index, rotsignflip=False):

        prepare = super()._patchind_preparerot(index, rotsignflip=rotsignflip)
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
    r"""Class constructor.
        
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
            catalog. Defaaullts to ``True``.
        ret_inst: bool, optional
            Decides on wheter to return the output as a list of arrays containing the reduced catalog or
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
    
    def multihash(self, dpixs, dpix_hash=None, normed=True, shuffle=0, w2field=True,
                  extent=[None,None,None,None], forcedivide=1):
        r"""Builds spatialhash for a base catalog and its reductions. 
        
        Parameters
        ----------
        dpixs: list
            The pixel sizes on which the hierarchy of reduced catalogs is constructed.
        dpix_hash: float, optional
            The size of the pixels used for the spatial hash of the hierarchy of catalogs. Defaults
            to ``None``. If set to ``None`` uses the largest value of ``dpixs``.
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
            catalog. Defaaullts to ``True``.
            
        Returns
        -------
        res: tuple
            Contains the output of the ```Catalog._multihash method```
        """
        if not w2field:
            fields=(self.tracer_1, self.tracer_2,) 
        else:
            fields=(self.tracer_1, self.tracer_2, self.weight**2,) 
        res = super()._multihash(
            dpixs=dpixs, 
            fields=fields, 
            dpix_hash=dpix_hash,
            normed=normed, 
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide)
        return res
    
    
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