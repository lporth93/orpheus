import ctypes as ct
from copy import deepcopy
import glob
from math import factorial
import numpy as np 
from numpy.ctypeslib import ndpointer
import operator
from pathlib import Path
import sys
from .flat2dgrid import FlatPixelGrid_2D, FlatDataGrid_2D
from .catalog import Catalog, ScalarTracerCatalog, SpinTracerCatalog

__all__ = ["DirectEstimator", "Direct_MapnEqual", "Direct_NapnEqual", "MapCombinatorics"]

class DirectEstimator:
    r"""
    Class of aperture statistics up to nth order for various arbitrary tracer catalogs. 
    This class contains attributes and methods that can be used across any of its children.
    
    Attributes
    ----------
    Rmin : float
        The smallest aperture radius for which the cumulants are computed.

    Rmax : float
        The largest aperture radius for which the cumulants are computed.

    nbinsr : int, optional
        The number of radial bins for the aperture radii. If set to
        ``None`` this attribute is inferred from the ``binsize`` attribute.

    binsize : int, optional
        The logarithmic size of the radial bins for the aperture radii. If set to
        ``None`` this attribute is inferred from the ``nbinsr`` attribute.

    aperture_centers : str, optional
        How to sample the apertures. Can be ``'grid'`` or ``'density'``.

    accuracies : int or numpy.ndarray, optional
        The sampling density of aperture centers.

        * If ``aperture_centers`` is set to ``'grid'``, setting ``accuracy == x``
          places the apertures on a regular grid with pixel size ``R_ap / x``.
        * If ``aperture_centers`` is set to ``'density'``, randomly selects as many
          galaxies as there would be aperture centers on the regular grid.

    frac_covs : numpy.ndarray, optional
        The different aperture coverage bins for which the statistics are evaluated. The first bin
        only includes apertures with ``coverage <= frac_covs[0]`` while the other coverage bins include the
        intervals between ``frac_covs[i]`` and ``frac_covs[i+1]``. Coverage is defined as the percentage of 
        the aperture area that is not within the survey area.

    dpix_hash : float, optional
        The pixel size of the spatial hash used to search through the catalog.

    weight_outer : float, optional
        The fractional weight applied to galaxies not contained within the interior of the catalog.
        This only affects catalogs which are overlapping patches of a full-sky catalog.

    weight_inpainted : float, optional
        The fractional weight applied to virtual galaxies inpainted into the catalog. This only
        affects catalogs which have objects in them that are labeled as inpainted.

    method : str, optional
        The method to be employed for the estimator. Defaults to ``Discrete``.

    multicountcorr : bool, optional
        Flag on whether to subtract multiplets in which the same tracer appears more
        than once. Defaults to ``True``.

    shuffle_pix : int, optional
        Choice of how to define centers of the cells in the spatial hash structure.
        Defaults to ``1``, i.e. random positioning.

    tree_resos : list, optional
        The cell sizes of the hierarchical spatial hash structure.

    tree_redges : list, optional
        Deprecated (possibly).

    rmin_pixsize : int, optional
        The limiting radial distance relative to the cell of the spatial hash
        after which one switches to the next hash in the hierarchy. At the moment
        does have no effect.Defaults to ``20``.

    resoshift_leafs : int, optional
        Allows for a difference in how the hierarchical spatial hash is traversed for
        pixels at the base of the NPCF and pixels at leafs. Positive values indicate
        that leafs will be evaluated at coarser resolutions than the base. At the moment 
        does have no effect. Defaults to ``0``.

    minresoind_leaf : int, optional
        Sets the smallest resolution in the spatial hash hierarchy which can be used to access
        tracers at leaf positions. If set to ``None`` uses the smallest specified cell size. 
        At the moment does have no effect. Defaults to ``None``.


    maxresoind_leaf : int, optional
        Sets the largest resolution in the spatial hash hierarchy which can be used to access
        tracers at leaf positions. If set to ``None`` uses the largest specified cell size. 
        At the moment does have no effect. Defaults to ``None``.

    nthreads : int, optional
        The number of OpenMP threads used for the reduction procedure. Defaults to ``16``.
    """
    
    def __init__(self, Rmin, Rmax, nbinsr=None, binsize=None, 
                 aperture_centers="grid", accuracies=2., 
                 frac_covs=[0.,0.1,0.3,0.5,1.], dpix_hash=1.,
                 weight_outer=1., weight_inpainted=0.,
                 method="Discrete", multicountcorr=True, shuffle_pix=1, 
                 tree_resos=[0,0.25,0.5,1.,2.], tree_redges=None, rmin_pixsize=20, 
                 resoshift_leafs=0, minresoind_leaf=None, maxresoind_leaf=None,nthreads=16):
        
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.method = method
        self.multicountcorr = int(multicountcorr)
        self.shuffle_pix = shuffle_pix
        self.aperture_centers = aperture_centers
        self.accuracies = accuracies
        self.frac_covs = np.asarray(frac_covs, dtype=np.float64)
        self.nfrac_covs = len(self.frac_covs)
        self.dpix_hash = dpix_hash
        self.weight_outer = weight_outer
        self.weight_inpainted = weight_inpainted
        self.methods_avail = ["Discrete", "Tree", "BaseTree", "DoubleTree", "FFT"]
        self.tree_resos = np.asarray(tree_resos, dtype=np.float64)
        self.tree_nresos = int(len(self.tree_resos))
        self.tree_redges = tree_redges
        self.rmin_pixsize = rmin_pixsize
        self.resoshift_leafs = resoshift_leafs
        self.minresoind_leaf = minresoind_leaf
        self.maxresoind_leaf = maxresoind_leaf
        self.nthreads = np.int32(max(1,nthreads))
        
        # Check types or arguments
        assert(isinstance(self.Rmin, float))
        assert(isinstance(self.Rmax, float))
        assert(self.aperture_centers in ["grid","density"])
        assert((self.weight_outer>=0.) and (self.weight_outer<=1.))
        assert((self.weight_inpainted>=0.) and (self.weight_inpainted<=1.))
        assert(self.method in self.methods_avail)
        assert(isinstance(self.tree_resos, np.ndarray))
        assert(isinstance(self.tree_resos[0], np.float64))
        
        # Setup radial bins
        # Note that we always have self.binsize <= binsize
        assert((binsize!=None) or (nbinsr!=None))
        if nbinsr != None:
            self.nbinsr = int(nbinsr)
        if binsize != None:
            assert(isinstance(binsize, float))
            self.nbinsr = int(np.ceil(np.log(self.Rmax/self.Rmin)/binsize))
        assert(isinstance(self.nbinsr, int))
        self.radii = np.geomspace(self.Rmin, self.Rmax, self.nbinsr)
        # Setup variable for tree estimator
        if self.tree_redges != None:
            assert(isinstance(self.tree_redges, np.ndarray))
            self.tree_redges = self.tree_redges.astype(np.float64)
            assert(len(self.tree_redges)==self.tree_resos+1)
            self.tree_redges = np.sort(self.tree_redges)
            assert(self.tree_redges[0]==self.Rmin)
            assert(self.tree_redges[-1]==self.Rmax)
        else:
            self.tree_redges = np.zeros(len(self.tree_resos)+1)
            self.tree_redges[-1] = self.Rmin
            for elreso, reso in enumerate(self.tree_resos):
                self.tree_redges[elreso] = (reso==0.)*self.Rmax + (reso!=0.)*self.rmin_pixsize*reso
        # Setup accuracies
        if (isinstance(self.accuracies, int) or isinstance(self.accuracies, float)):
            self.accuracies = self.accuracies*np.ones(self.nbinsr,dtype=np.float64)
        self.accuracies = np.asarray(self.accuracies,dtype=np.float64)
        assert(isinstance(self.accuracies,np.ndarray))
        # Prepare leaf resolutions
        if np.abs(self.resoshift_leafs)>=self.tree_nresos:
            self.resoshift_leafs = np.int32((self.tree_nresos-1) * np.sign(self.resoshift_leafs))
            print("Error: Parameter resoshift_leafs is out of bounds. Set to %i."%self.resoshift_leafs)
        if self.minresoind_leaf is None:
            self.minresoind_leaf=0
        if self.maxresoind_leaf is None:
            self.maxresoind_leaf=self.tree_nresos-1
        if self.minresoind_leaf<0:
            self.minresoind_leaf = 0
            print("Error: Parameter minreso_leaf is out of bounds. Set to 0.")
        if self.minresoind_leaf>=self.tree_nresos:
            self.minresoind_leaf = self.tree_nresos-1
            print("Error: Parameter minreso_leaf is out of bounds. Set to %i."%self.minresoint_leaf)
        if self.maxresoind_leaf<0:
            self.maxresoind_leaf = 0
            print("Error: Parameter minreso_leaf is out of bounds. Set to 0.")
        if self.maxresoind_leaf>=self.tree_nresos:
            self.maxresoind_leaf = self.tree_nresos-1
            print("Error: Parameter minreso_leaf is out of bounds. Set to %i."%self.maxresoint_leaf) 
        if self.maxresoind_leaf<self.minresoind_leaf:
            print("Error: Parameter maxreso_leaf is smaller than minreso_leaf. Set to %i."%self.minreso_leaf) 
          
        #############################
        ## Link compiled libraries ##
        #############################
        # Method that works for LP
        target_path = __import__('orpheus').__file__
        self.library_path = str(Path(__import__('orpheus').__file__).parent.absolute())
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        
        # In case the environment is weird, compile code manually and load it here...
        #self.clib = ct.CDLL("/vol/euclidraid4/data/lporth/HigherOrderLensing/Estimator/orpheus/orpheus/src/discrete.so")
        
        # Method that works for RR (but not for LP with a local HPC install)
        #self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"orpheus_clib"))
        #self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        #print(self.library_path)
        #print(self.clib)
        p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)    
        
    def get_pixelization(self, cat, R_ap, accuracy, R_crop=None, mgrid=True):
        """ Computes pixel grid on inner region of survey field.

        Arguments:
        ----------
        R_ap (float):
            The radius of the aperture in pixel scale.
        accuracy (float):
            Accuracy parameter for the pixel grid.
            A value of 0.5 results in a grid in which the apertures
            are only touching each other - hence minimizing correlations.

        Returns:
        --------
        grid_x (array of floats):
            The grid cell centers for the x-coordinate.
        grid_y (array of floats):
            The grid cell centers for the y-coordinate.

        Notes:
        ------
        The grid covers the rectangel between the extremal x/y coordinates of
        the galaxy catalogue.
        """
        
        if float(accuracy)==-1.:
            centers_1 = cat.pos1[cat.isinner>=0.5]
            centers_2 = cat.pos2[cat.isinner>=0.5]
            
        else:
            start1 = cat.min1
            start2 = cat.min2
            end1 = cat.max1
            end2 = cat.max2
            if R_crop is not None:
                start1 += R_crop
                start2 += R_crop
                end1 -= R_crop
                end2 -= R_crop

            len_1 = end1 - start1
            len_2 = end2 - start2

            npixels_1 = int(np.ceil(accuracy * len_1 / R_ap))
            npixels_2 = int(np.ceil(accuracy * len_2 / R_ap))

            stepsize_1 = len_1 / npixels_1
            stepsize_2 = len_2 / npixels_2

            _centers_1 = [start1 + npixel *
                         stepsize_1 for npixel in range(npixels_1 + 1)]
            _centers_2 = [start2 + npixel *
                         stepsize_2 for npixel in range(npixels_2 + 1)]

            if mgrid:
                centers_1, centers_2 = np.meshgrid(_centers_1,_centers_2)
                centers_1 = centers_1.flatten()
                centers_2 = centers_2.flatten()
            else:
                centers_1 = np.asarray(_centers_1, dtype=np.float64)
                centers_2 = np.asarray(_centers_2, dtype=np.float64)
            
        return centers_1, centers_2
        
    def __getmap(self, R, cat, dotomo, field, filter_form):
        """ This simply computes an aperture mass map together with weights and coverages """
        
        
                
class Direct_MapnEqual(DirectEstimator):
    r"""
    Compute direct estimator for equal-scale aperture mass statistics.

    Attributes
    ----------
    order_max : int
        Maximum order of the statistics to be computed.

    Rmin : float
        Minimum aperture radius.

    Rmax : float
        Maximum aperture radius.

    field : str, optional
        Type of input field (``"scalar"`` or ``"polar"``).

    filter_form : str, optional
        Filter type used in the aperture function (``"S98"``, ``"C02"``, ``"Sch04"``, etc.).

    ap_weights : str, optional
        Aperture weighting strategy (``"Identity"``, ``"InvShot"``).

    **kwargs : dict
        Additional keyword arguments passed to :class:`DirectEstimator`.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`DirectEstimator`.  
    Additional child-specific parameters can be passed via ``kwargs``.
    """
    
    def __init__(self, order_max, Rmin, Rmax, field="polar", filter_form="C02", ap_weights="InvShot", **kwargs):

        super().__init__(Rmin=Rmin, Rmax=Rmax, **kwargs)
        self.order_max = order_max
        self.nbinsz = None
        self.field = field
        self.filter_form = filter_form
        self.ap_weights = ap_weights
        
        self.fields_avail = ["scalar", "polar"]
        self.ap_weights_dict = {"Identity":0, "InvShot":1}
        self.filters_dict = {"S98":0, "C02":1, "Sch04":2, "PolyExp":3}
        self.ap_weights_avail = list(self.ap_weights_dict.keys())
        self.filters_avail = list(self.filters_dict.keys())
        assert(self.field in self.fields_avail)
        assert(self.ap_weights in self.ap_weights_avail)
        assert(self.filter_form in self.filters_avail)
        
        # We do not need DoubleTree for equal-aperture estimator
        if self.method=="DoubleTree":
            self.method="Tree"
            
        p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64) 
        
        # Compute nth order equal-scale statistics using discrete estimator (E-Mode only!)
        self.clib.MapnSingleEonlyDisc.restype = ct.c_void_p
        self.clib.MapnSingleEonlyDisc.argtypes = [
            ct.c_double, p_f64, p_f64, ct.c_int32,
            ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, 
            p_f64, p_f64, p_f64, p_f64, p_c128, p_i32, ct.c_int32, ct.c_int32,
            p_f64, p_f64, ct.c_int32, ct.c_int32,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_i32, p_i32, p_i32, 
            ct.c_int32, p_f64, p_f64]
        self.clib.singleAp_MapnSingleEonlyDisc.restype = ct.c_void_p
        self.clib.singleAp_MapnSingleEonlyDisc.argtypes = [
            ct.c_double, ct.c_double, ct.c_double,
            ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, 
            p_f64, p_f64, p_f64, p_f64, p_c128, p_i32, ct.c_int32, ct.c_int32,
            p_f64,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_i32, p_i32, p_i32,
            p_f64, p_f64, p_f64, p_f64, p_f64, p_f64]    
        
        # Compute aperture mass map for equal-scale stats
        self.clib.ApertureMassMap_Equal.restype = ct.c_void_p
        self.clib.ApertureMassMap_Equal.argtypes = [
            ct.c_double, p_f64, p_f64, ct.c_int32, ct.c_int32, 
            ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, 
            p_f64, p_f64, p_f64, p_f64, p_c128, p_i32, ct.c_int32, ct.c_int32,
            p_f64,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_i32, p_i32, p_i32, 
            ct.c_int32, p_f64, p_f64, p_f64, p_f64, p_f64, p_f64]
               
                       
    def process(self, cat, dotomo=True, Emodeonly=True, connected=True, dpix_innergrid=2.):
        r"""
        Computes aperture statistics on a catalog.

        Parameters
        ----------
        cat : orpheus.SpinTracerCatalog
            The catalog instance to be processed.
        dotomo : bool, optional
            Whether to compute the statistics for all tomographic bin combinations.
            Default is True.
        Emodeonly : bool, optional
            Currently does not have an impact.
            Default is False.
        connected : bool, optional
            Whether to output only the connected part of the aperture mass statistics.
            Does not have an impact at the moment.
            Default is True.
        dpix_innergrid : float, optional
            Pixel size for a rough reconstruction of the angular mask. Used to preselect
            aperture centers in the interior of the survey.
            Default is 2.

        Returns
        -------
        None
            Currently does not return any value.
        """
        
        nbinsz = cat.nbinsz
        if not dotomo:
            nbinsz = 1
        self.nbinsz = nbinsz
         
        nzcombis = self._nzcombis_tot(nbinsz,dotomo)
        result_Mapn = np.zeros((self.nbinsr, self.nfrac_covs, nzcombis), dtype=np.float64)
        result_wMapn = np.zeros((self.nbinsr, self.nfrac_covs, nzcombis), dtype=np.float64)
        if (self.method in ["Discrete", "BaseTree"]) and Emodeonly:
            func = self.clib.MapnSingleEonlyDisc
        elif (self.method in ["Discrete", "BaseTree"]) and not Emodeonly:
            raise NotImplementedError
        else:
            raise NotImplementedError
            
        # Build a grid that only covers inner part of patch
        # This will be used to preselelct aperture centers
        args_innergrid = cat.togrid(fields=[cat.isinner], dpix=dpix_innergrid, method="NGP", normed=True, tomo=False)
        
        for elr, R in enumerate(self.radii):
            nextmap_out = np.zeros(nzcombis*self.nfrac_covs ,dtype=np.float64)
            nextwmap_out = np.zeros(nzcombis*self.nfrac_covs ,dtype=np.float64)
            args = self._buildargs(cat, args_innergrid, dotomo, elr, forfunc="Equal")
            func(*args)
            result_Mapn[elr] = args[-2].reshape((self.nfrac_covs, nzcombis))[:]
            result_wMapn[elr] = args[-1].reshape((self.nfrac_covs, nzcombis))[:]
            
            sys.stdout.write("\rDone %i/%i aperture radii"%(elr+1,self.nbinsr))
                       
        return result_Mapn, result_wMapn
                
    def _getindex(self, order, mode, zcombi):
        pass
    
    def _buildargs(self, cat, args_innergrid, dotomo, indR, forfunc="Equal"):
        
        assert(forfunc in ["Equal", "EqualGrid"])
        
        if not dotomo:
            nbinsz = 1
            zbins = np.zeros(cat.ngal, dtype=np.int32)
        else:
            nbinsz = cat.nbinsz
            zbins = cat.zbins.astype(np.int32)
            
        # Initialize combinatorics instances for lateron
        # TODO: Should find a better place where to put this...
        self.combinatorics = {}
        for order in range(1,self.order_max+1):
            self.combinatorics[order] = MapCombinatorics(nbinsz,order_max=order)
            
        # Parameters related to the aperture grid
        if forfunc=="Equal":
            # Get centers and check that they are in the interior of the inner catalog
            centers_1, centers_2 = self.get_pixelization(cat, self.radii[indR], self.accuracies[indR], R_crop=0., mgrid=True)
            _f, _s1, _s2, _dpixi, _, _, = args_innergrid
            #pixs_c = (((centers_1-_s1)//_dpixi)*_f[0,1].shape[0] + (centers_2-_s2)//_dpixi).astype(int)
            pixs_c = (((centers_2-_s2)//_dpixi)*_f[0,1].shape[1] + (centers_1-_s1)//_dpixi).astype(int)

            sel_inner = _f[0,1]>0.
            sel_centers = sel_inner.flatten()[pixs_c]
            # For regular grid, select aperture centers within the interior of the survey
            if self.aperture_centers=="grid":
                centers_1 = centers_1[sel_centers]
                centers_2 = centers_2[sel_centers]
                ncenters = len(centers_1)
            # For density-based grid, select fraction of galaxy positions as aperture centers
            # Note that this will not bias the result as the galaxy residing at the center is
            # not taken into account in the directestimator.c code.
            elif self.aperture_centers=="density":
                rng = np.random.RandomState(1234567890)
                ncenters = max(len(centers_1),cat.ngal)
                sel_centers = rng.random.choice(np.arange(cat.ngal), ncenters, replace=False).astype(np.int32)
                centers_1 = cat.pos1[sel_centers] 
                centers_2 = cat.pos2[sel_centers] 
        elif forfunc=="EqualGrid": 
            # Get centers along each dimension
            centers_1, centers_2 = self.get_pixelization(cat, self.radii[indR], self.accuracies[indR], R_crop=0., mgrid=False)
            ncenters = len(centers_1)*len(centers_2)
        
        cat.build_spatialhash(dpix=self.dpix_hash, extent=[None, None, None, None])
        hashgrid = FlatPixelGrid_2D(cat.pix1_start, cat.pix2_start, 
                                    cat.pix1_n, cat.pix2_n, cat.pix1_d, cat.pix2_d)
        regridded_mask = cat.mask.regrid(hashgrid).data.flatten().astype(np.float64)
        
        len_out = self._nzcombis_tot(nbinsz,dotomo)*self.nfrac_covs
                
        if forfunc=="Equal":
            args_centers = (self.radii[indR], centers_1, centers_2, ncenters,)  
        elif forfunc=="EqualGrid":
            args_centers = (self.radii[indR], centers_1, centers_2, len(centers_1), len(centers_2), )  
        if forfunc=="Equal":
            args_ofw = (self.order_max, self.filters_dict[self.filter_form], self.ap_weights_dict[self.ap_weights], 
                        np.int32(self.multicountcorr), np.float64(self.weight_outer), np.float64(self.weight_inpainted), )
        elif forfunc=="EqualGrid":
            args_ofw = (self.order_max, self.filters_dict[self.filter_form], self.ap_weights_dict[self.ap_weights], 
                        np.float64(self.weight_outer), np.float64(self.weight_inpainted), )
        args_cat = (cat.weight.astype(np.float64), cat.isinner.astype(np.float64),
                    cat.pos1.astype(np.float64), cat.pos2.astype(np.float64), 
                    cat.tracer_1.astype(np.float64)+1j*cat.tracer_2.astype(np.float64), 
                    zbins, np.int32(nbinsz), np.int32(cat.ngal), )
        if forfunc=="Equal":
            args_mask = (regridded_mask, self.frac_covs, self.nfrac_covs, 1, )
        elif forfunc=="EqualGrid":
            args_mask = (regridded_mask, )
        args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix2_start), 
                     np.float64(cat.pix1_d), np.float64(cat.pix2_d), 
                     np.int32(cat.pix1_n), np.int32(cat.pix2_n), 
                     cat.index_matcher.astype(np.int32), cat.pixs_galind_bounds.astype(np.int32), 
                     cat.pix_gals.astype(np.int32)
                    )
        if forfunc=="Equal":
            args_out = (np.zeros(len_out).astype(np.float64), np.zeros(len_out).astype(np.float64))
        elif forfunc=="EqualGrid":
            args_out = (np.zeros(3*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(2*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), )
        # Return the parameters for the Map computation
        args =  (*args_centers,
                 *args_ofw,
                 *args_cat,
                 *args_mask,
                 *args_hash,
                 np.int32(self.nthreads), 
                 *args_out)
        if False:
            for elarg, arg in enumerate(args):
                func  = self.clib.MapnSingleEonlyDisc
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                try:
                    toprint += (func.argtypes[elarg], )
                    print(toprint)
                    print(arg)
                except:
                    print("We did have a problem for arg %i"%elarg)

        return args    
   
    def _nzcombis_tot(self, nbinsz, dotomo):
        res = 0
        for order in range(1, self.order_max+1):
            res += self._nzcombis_order(order, nbinsz, dotomo)
        return res
                
    def _nzcombis_order(self, order, nbinsz, dotomo):
        if not dotomo:
            return 1
        else:
            return int(nbinsz*factorial(nbinsz+order-1)/(factorial(nbinsz)*factorial(order)))
        
    def _cumnzcombis_order(self, order, nbinsz, dotomo):
        res = 0
        for order in range(1, order+1):
            res += self._nzcombis_order(order, nbinsz, dotomo)
        return res
        
    def genzcombi(self, zs, nbinsz=None):
        """ Returns index of tomographic bin combination of Map^n output.
        
        Arguments:
        ----------
        zs: list of integers
            Target combination of tomographic redshifts ([z1, ..., zk]).
        nbinsz: int, optional
            The number of tomographic bins in the computation of Map^n. If not set,
            reverts to corresponding class attribute.
            
        Returns
        -------
        zind_flat: int
            Index of flattened Map^k(z1,...,zk) datavector in global output.
        """
        if nbinsz is None:
            nbinsz = self.nbinsz
        if nbinsz is None:
            raise ValueError("No value for `nbinsz` has been allocated yet.")
        if len(zs)>self.order_max:
            raise ValueError("We only computed the statistics up to order %i."%self.order_max)
        if max(zs) >= nbinsz:
            raise ValueError("We only have %i tomographic bins available."%nbinsz)
        
        order = len(zs)
        zind_flat = self._cumnzcombis_order(order-1,nbinsz,True) + self.combinatorics[order].sel2ind(zs)
        
        return zind_flat
        
        
    def getmap(self, indR, cat, dotomo=True):
        """ Computes various maps that are part of the basis of the Map^n estimator.
        
        Arguments:
        ----------
        indR: int
            Index of aperture radius for which maps are computed
        cat: orpheus.SpinTracerCatalog
            The catalog instance to be processed
        dotomo: bool, optional
            Whether the tomographic information in `cat` should be 
            used for the map construction
            
        Returns:
        --------
        counts: ndarray
            Aperture number counts
        
        """
        
        nbinsz = cat.nbinsz
        if not dotomo:
            nbinsz = 1
            
        args = self._buildargs(cat, None, dotomo, indR, forfunc="EqualGrid")
        ncenters_1 = args[3]
        ncenters_2 = args[4]
        self.clib.ApertureMassMap_Equal(*args)
        
        counts = args[-6].reshape((nbinsz, 3, ncenters_2, ncenters_1))
        covs = args[-5].reshape((2, ncenters_2, ncenters_1))
        Msn = args[-4].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        Sn = args[-3].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        Mapn = args[-2].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        Mapn_var = args[-1].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        
        return counts, covs, Msn, Sn, Mapn, Mapn_var 
    
    
class Direct_NapnEqual(DirectEstimator):
    r"""
    Compute direct estimator for equal-scale aperture counts statistics.

    Attributes
    ----------
    order_max : int
        Maximum order of the statistics to be computed.

    Rmin : float
        Minimum aperture radius.

    Rmax : float
        Maximum aperture radius.

    field : str, optional
        Type of input field (``"scalar"`` or ``"polar"``).

    filter_form : str, optional
        Filter type used in the aperture function (``"S98"``, ``"C02"``, ``"Sch04"``, etc.).

    ap_weights : str, optional
        Aperture weighting strategy (``"Identity"``, ``"InvShot"``).

    **kwargs : dict
        Additional keyword arguments passed to :class:`DirectEstimator`.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`DirectEstimator`.  
    Additional child-specific parameters can be passed via ``kwargs``.
    """

    
    def __init__(self, order_max, Rmin, Rmax, field="scalar", filter_form="C02", ap_weights="Identity", **kwargs):
        super().__init__(Rmin=Rmin, Rmax=Rmax, **kwargs)
        self.order_max = order_max
        self.nbinsz = None
        self.field = field
        self.filter_form = filter_form
        self.ap_weights = ap_weights
        
        self.fields_avail = ["scalar", "polar"]
        self.ap_weights_dict = {"Identity":0, "InvShot":1}
        self.filters_dict = {"S98":0, "C02":1, "Sch04":2, "PolyExp":3}
        self.ap_weights_avail = list(self.ap_weights_dict.keys())
        self.filters_avail = list(self.filters_dict.keys())
        assert(self.field in self.fields_avail)
        assert(self.ap_weights in self.ap_weights_avail)
        assert(self.filter_form in self.filters_avail)
        
        # We do not need DoubleTree for equal-aperture estimator
        if self.method=="DoubleTree":
            self.method="Tree"
            
        p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64) 
                
        # Compute aperture counts map for equal-scale stats
        self.clib.ApertureCountsMap_Equal.restype = ct.c_void_p
        self.clib.ApertureCountsMap_Equal.argtypes = [
            ct.c_double, p_f64, p_f64, ct.c_int32, ct.c_int32,
            ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, 
            p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            p_f64, 
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_i32, p_i32, p_i32,
            ct.c_int32, p_f64, p_f64, p_f64, p_f64, p_f64, p_f64]
        
        # Compute nth order equal-scale statistics using discrete estimator (E-Mode only!)
        self.clib.NapnSingleDisc.restype = ct.c_void_p
        self.clib.NapnSingleDisc.argtypes = [
            ct.c_double, p_f64, p_f64, ct.c_int32,
            ct.c_int32, ct.c_int32, ct.c_int32,  ct.c_int32, ct.c_double, ct.c_double, 
            p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            p_f64, p_f64, ct.c_int32, ct.c_int32,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_i32, p_i32, p_i32,
            ct.c_int32, p_f64, p_f64]
        
        # Compute nth order equal-scale statistics using discrete estimator (E-Mode only!)
        self.clib.singleAp_NapnSingleDisc.restype = ct.c_void_p
        self.clib.singleAp_NapnSingleDisc.argtypes = [
            ct.c_double, ct.c_double, ct.c_double, 
            ct.c_int32, ct.c_int32,  ct.c_int32, ct.c_double, ct.c_double, 
            p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            p_f64, 
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_i32, p_i32, p_i32,
            p_f64, p_f64, p_f64, p_f64]
               
    def process(self, cat, dotomo=True, Nbar_policy=1, connected=True, dpix_innergrid=2.):
        r"""
        Computes aperture statistics on a catalog.

        Parameters
        ----------
        cat : orpheus.SpinTracerCatalog
            The catalog instance to be processed.
        dotomo : bool, optional
            Whether to compute the statistics for all tomographic bin combinations.
            Default is True.
        Nbar_policy : int, optional
            What normalization to use:

            0 : Use local Nbar for normalization
            1 : Use global Nbar for normalization
            2 : No Nbar for normalization

            Default is 1.
        connected : bool, optional
            Whether to output only the connected part of the aperture mass statistics.
            Does not have an impact at the moment.
        dpix_innergrid : float, optional
            Pixel size for a rough reconstruction of the angular mask. Used to preselect
            aperture centers in the interior of the survey.
            Default is 2.

        Returns
        -------
        None
            Currently does not return any value.
        """
        
        assert(isinstance(cat, ScalarTracerCatalog))
        
        nbinsz = cat.nbinsz
        if not dotomo:
            nbinsz = 1
        self.nbinsz = nbinsz
         
        nzcombis = self._nzcombis_tot(nbinsz,dotomo)
        result_Napn = np.zeros((self.nbinsr, self.nfrac_covs, nzcombis), dtype=np.float64)
        result_wNapn = np.zeros((self.nbinsr, self.nfrac_covs, nzcombis), dtype=np.float64)
        if (self.method in ["Discrete", "BaseTree"]):
            func = self.clib.NapnSingleDisc
        elif (self.method in ["Discrete", "BaseTree"]):
            raise NotImplementedError
        else:
            raise NotImplementedError
            
        # Build a grid that only covers inner part of patch
        # This will be used to preselelct aperture centers
        args_innergrid = cat.togrid(fields=[cat.isinner], dpix=dpix_innergrid, method="NGP", normed=True, tomo=False)
        
        for elr, R in enumerate(self.radii):
            nextnap_out = np.zeros(nzcombis*self.nfrac_covs ,dtype=np.float64)
            nextwnap_out = np.zeros(nzcombis*self.nfrac_covs ,dtype=np.float64)
            args = self._buildargs(cat, args_innergrid, dotomo, Nbar_policy, elr, forfunc="Equal")
            func(*args)
            result_Napn[elr] = args[-2].reshape((self.nfrac_covs, nzcombis))[:]
            result_wNapn[elr] = args[-1].reshape((self.nfrac_covs, nzcombis))[:]
            
            sys.stdout.write("\rDone %i/%i aperture radii"%(elr+1,self.nbinsr))
                       
        return result_Napn, result_wNapn
                
    def _getindex(self, order, mode, zcombi):
        pass
    
    def _buildargs(self, cat, args_innergrid, dotomo, Nbar_policy, indR, forfunc="Equal"):
        
        assert(forfunc in ["Equal", "EqualGrid"])
        
        if not dotomo:
            nbinsz = 1
            zbins = np.zeros(cat.ngal, dtype=np.int32)
        else:
            nbinsz = cat.nbinsz
            zbins = cat.zbins.astype(np.int32)
            
        # Initialize combinatorics instances for lateron
        # TODO: Should find a better place where to put this...
        self.combinatorics = {}
        for order in range(1,self.order_max+1):
            self.combinatorics[order] = MapCombinatorics(nbinsz,order_max=order)
            
        # Parameters related to the aperture grid
        if forfunc=="Equal":
            # Get centers and check that they are in the interior of the inner catalog
            centers_1, centers_2 = self.get_pixelization(cat, self.radii[indR], self.accuracies[indR], R_crop=0., mgrid=True)
            _f, _s1, _s2, _dpixi, _, _, = args_innergrid
            pixs_c = (((centers_2-_s2)//_dpixi)*_f[0,1].shape[1] + (centers_1-_s1)//_dpixi).astype(int)
            sel_inner = _f[0,1]>0.
            sel_centers = sel_inner.flatten()[pixs_c]
            centers_1 = centers_1[sel_centers]
            centers_2 = centers_2[sel_centers]
            ncenters = len(centers_1)
        elif forfunc=="EqualGrid": 
            # Get centers along each dimension
            centers_1, centers_2 = self.get_pixelization(cat, self.radii[indR], self.accuracies[indR], R_crop=0., mgrid=False)
            ncenters = len(centers_1)*len(centers_2)

        #self.dpix_hash = max(0.25, self.radii[indR])
        cat.build_spatialhash(dpix=self.dpix_hash, extent=[None, None, None, None])
        hashgrid = FlatPixelGrid_2D(cat.pix1_start, cat.pix2_start, 
                                    cat.pix1_n, cat.pix2_n, cat.pix1_d, cat.pix2_d)
        regridded_mask = cat.mask.regrid(hashgrid).data.flatten().astype(np.float64)
        
        if forfunc=="Equal":
            args_centers = (self.radii[indR], centers_1, centers_2, ncenters,)  
        elif forfunc=="EqualGrid":
            args_centers = (self.radii[indR], centers_1, centers_2, len(centers_1), len(centers_2), )  
        if forfunc=="Equal":
            args_ofw = (self.order_max, self.filters_dict[self.filter_form], 
                        np.int32(self.multicountcorr), np.int32(Nbar_policy), np.float64(self.weight_outer), np.float64(self.weight_inpainted), )
        elif forfunc=="EqualGrid":
            args_ofw = (self.order_max, self.filters_dict[self.filter_form], self.ap_weights_dict[self.ap_weights],
                        np.float64(self.weight_outer), np.float64(self.weight_inpainted), )
        args_cat = (cat.weight.astype(np.float64), cat.isinner.astype(np.float64),
                    cat.pos1.astype(np.float64), cat.pos2.astype(np.float64), cat.tracer.astype(np.float64),
                    zbins, np.int32(nbinsz), np.int32(cat.ngal), )
        if forfunc=="Equal":
            args_mask = (regridded_mask, self.frac_covs, self.nfrac_covs, 0, )
        elif forfunc=="EqualGrid":
            args_mask = (regridded_mask, )
        args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix2_start), 
                     np.float64(cat.pix1_d), np.float64(cat.pix2_d), 
                     np.int32(cat.pix1_n), np.int32(cat.pix2_n), 
                     cat.index_matcher.astype(np.int32), cat.pixs_galind_bounds.astype(np.int32), 
                     cat.pix_gals.astype(np.int32)
                    )
        if forfunc=="Equal":
            len_out = self._nzcombis_tot(nbinsz,dotomo)*self.nfrac_covs
            args_out = (np.zeros(len_out).astype(np.float64), np.zeros(len_out).astype(np.float64))
        elif forfunc=="EqualGrid":
            args_out = (np.zeros(3*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(2*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), 
                        np.zeros(self.order_max*nbinsz*ncenters).astype(np.float64), )
        # Return the parameters for the Nap computation
        args =  (*args_centers,
                 *args_ofw,
                 *args_cat,
                 *args_mask,
                 *args_hash,
                 np.int32(self.nthreads), 
                 *args_out)
        if False:
            if forfunc=="Equal":func  = self.clib.NapnSingleDisc
            if forfunc=="EqualGrid":func  = self.clib.ApertureCountsMap_Equal

            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                #try:
                toprint += (func.argtypes[elarg], )
                print(toprint)
                print(arg)
                #except:
                #    print("We did have a problem for arg %i"%elarg)

        return args    
   
    def _nzcombis_tot(self, nbinsz, dotomo):
        res = 0
        for order in range(1, self.order_max+1):
            res += self._nzcombis_order(order, nbinsz, dotomo)
        return res
                
    def _nzcombis_order(self, order, nbinsz, dotomo):
        if not dotomo:
            return 1
        else:
            return int(nbinsz*factorial(nbinsz+order-1)/(factorial(nbinsz)*factorial(order)))
        
    def _cumnzcombis_order(self, order, nbinsz, dotomo):
        res = 0
        for order in range(1, order+1):
            res += self._nzcombis_order(order, nbinsz, dotomo)
        return res
        
    def genzcombi(self, zs, nbinsz=None):
        if nbinsz is None:
            nbinsz = self.nbinsz
        if nbinsz is None:
            raise ValueError("No value for `nbinsz` has been allocated yet.")
        if len(zs)>self.order_max:
            raise ValueError("We only computed the statistics up to order %i."%self.order_max)
        if max(zs) >= nbinsz:
            raise ValueError("We only have %i tomographic bins available."%nbinsz)
        
        order = len(zs)
        return self._cumnzcombis_order(order-1,nbinsz,True) + self.combinatorics[order].sel2ind(zs)
        
        
    def getnap(self, indR, cat, dotomo=True):
        """ This simply computes an aperture mass map together with weights and coverages """
        nbinsz = cat.nbinsz
        if not dotomo:
            nbinsz = 1
            
        args = self._buildargs(cat, None, dotomo, indR, forfunc="EqualGrid")
        ncenters_1 = args[3]
        ncenters_2 = args[4]
        self.clib.ApertureCountsMap_Equal(*args)
        
        counts = args[-6].reshape((nbinsz, 3, ncenters_2, ncenters_1))
        covs = args[-5].reshape((2, ncenters_2, ncenters_1))
        Msn = args[-4].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        Sn = args[-3].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        Napn = args[-2].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        Napn_counts = args[-1].reshape((nbinsz, self.order_max, ncenters_2, ncenters_1))
        
        return counts, covs, Msn, Sn, Napn, Napn_counts
    
        
class MapCombinatorics:
    
    def __init__(self, nradii, order_max):
        self.nradii = nradii
        self.order_max = order_max
        self.psummem = None
        self.nindices = self.psumtot(order_max+1, nradii)
        
    def psumtot(self, n, m):
        """ Calls to (n-1)-fold nested loop over m indicices
        where i1 <= i2 <= ... <= in. This is equivalent to the
        number of independent Map^i components over a range of
        m radii (0<i<=n) as well as to the size of the multivariate 
        power sum set generating those multivariate cumulants.
        
        It can alternatively be used do count through the flattened
        redshift indices for a single-scale computation up until a
        maximum order.

        Example:
        psumtot(m=10,n=4) gives the same result as the code
        >>> res = 0
        >>> for i1 in range(10):
        >>>     for i2 in range(i1,10):
        >>>         for i3 in range(i2,10):
        >>>             res += 1
        >>> print(res)

        Notes:
        * The recursion reads as follows:
          s(m,0) = 1
          s(m,n) = sum_{i=1}^{m-1} s(m-1,n-1)
          [Have not formally proved that but checked with pen and paper
          up until n=3 on examples and the underlying geometry does make 
          sense. Testing against nested loops also works as long as the
          loops can be computed in a sensible amount of time]
        * As the solution is recusive and therefore might take long to
          compute we use a memoization technique to get rid of all of
          the unneccessary nested calls.
        """
        
        assert(m<=self.nradii)
        assert(n<=self.order_max+1)
            
        # For initial call allocate memo
        if self.psummem is None:
            self.psummem = np.zeros((n,m))#, dtype=np.int)
            self.psummem[0] = np.ones(m)
        # Base case
        if m<=0 or n<=0:
            return self.psummem[n,m]
        # Recover from memo
        if self.psummem[n-1,m-1] != 0:
            return self.psummem[n-1,m-1]
        # Add to memo
        else:
            res = 0
            for i in range(m):
                res += self.psumtot(n-1,m-i)
            self.psummem[n-1,m-1] = res
            return int(self.psummem[n-1,m-1])
        
    def sel2ind(self, sel):
        """  
        Assignes unique index to given selection in powr sum set
        Note that sel[0] <= sel[1] <= ... <= sel[self.nradii-1] is required!
        """
        # Check validity
        #assert(len(sel)==n-1)
        #for el in range(len(sel)-1):
        #    assert(sel[el+1] >= sel[el])
        #assert(sel[-1] <= m)

        i = 0
        ind = 0
        ind_sel = 0
        lsel = len(sel)
        while True:
            while i >= sel[ind_sel]:
                #print(i,ind_sel)
                ind_sel += 1
                if ind_sel >= lsel:
                    return int(ind)
            ind += self.psummem[self.order_max-1-ind_sel, self.nradii-1-i]
            i += 1

        return int(ind)
    
    
    def ind2sel(self, ind):
        """ Inverse of sel2ind...NOT WORKING YET """
        
        sel = np.zeros(self.order_max)#, dtype=np.int)
        # Edge cases
        if ind==0:
            return sel.astype(np.int)
        if ind==1:
            sel[-1] = 1
            return sel.astype(np.int)
        if ind==self.nindices-1:
            return (self.nradii-1)*np.zeros(self.order_max, dtype=np.int)
        
        tmpind = ind # Remainder of index in psum
        nextind_ax0 = self.order_max-1 # Value of i_k
        nextind_ax1 = self.nradii-1 # Helper
        tmpsel = 0 # Value of i_k
        indsel = 0 # Index in selection
        while True:
            nextsubs = 0
            while True:
                tmpsubs = self.psummem[nextind_ax0, nextind_ax1]
                #print(tmpind, nextsubs, tmpsubs, sel)
                if tmpind > nextsubs + tmpsubs:
                    nextind_ax1 -= 1
                    tmpsel += 1
                    nextsubs += tmpsubs
                elif tmpind < nextsubs + tmpsubs:
                    nextind_ax0 -= 1
                    tmpind -= nextsubs
                    sel[indsel] = tmpsel
                    indsel += 1
                    break
                else:
                    sel[indsel:] = tmpsel + 1
                    return sel.astype(np.int)
            if sel[-2] != 0:
                sel[-1] = sel[-2] + tmpind
            if sel[-1] != 0:
                return sel.astype(np.int)