import numpy as np 
import ctypes as ct
from functools import reduce
import operator
from scipy.interpolate import interp1d

from .utils import flatlist, gen_thetacombis_fourthorder, gen_n2n3indices_Upsfourth
from .npcf_base import BinnedNPCF
from .npcf_second import GGCorrelation

__all__ = ["NNNNCorrelation_NoTomo", "GGGGCorrelation_NoTomo"]

class NNNNCorrelation_NoTomo(BinnedNPCF):
    r""" Class containing methods to measure and and obtain statistics that are built
    from nontomographic fourth-order scalar correlation functions.
    
    Attributes
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.
    thetabatchsize_max: int, optional
        The largest number of radial bin combinations that are processed in parallel.
        Defaults to ``10 000``.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    Additional child-specific parameters can be passed via ``kwargs``. 
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme .
    
    """
    
    def __init__(self, min_sep, max_sep, verbose=False, thetabatchsize_max=10000, method="Tree", **kwargs):
        super().__init__(order=4, spins=np.array([0,0,0,0], dtype=np.int32),
                         n_cfs=1, min_sep=min_sep, max_sep=max_sep, 
                         method=method, methods_avail=["Tree"], **kwargs)
        
        self.thetabatchsize_max = thetabatchsize_max
        self.nbinsz = 1
        self.nzcombis = 1
        
    def process(self, cat, statistics="all", tofile=False, apply_edge_correction=False, 
                lowmem=True, mapradii=None, batchsize=None, custom_thetacombis=None, cutlen=2**31-1):
        r"""
        Arguments:
        
        Logic works as follows:
        * Keyword 'statistics' \in [4pcf_real, 4pcf_multipoles, N4, Nap4, Nap4, Nap4c, allNap, all4pcf, all]
        * - If 4pcf_multipoles in statistics --> save 4pcf_multipoles
        * - If 4pcf_real in statistics --> save 4pcf_real
        * - If only N4 in statistics --> Do not save any 4pcf. This is really the lowmem case.
        * - allNap, all4pcf, all are abbreviations as expected
        * If lowmem=True, uses the inefficient, but lowmem function for computation and output statistics 
        from there as wanted.
        * If lowmem=False, use the fast functions to do the 4pcf multipole computation and do 
        the potential conversions lateron.
        * Default lowmem to None and
        * - Set to true if any aperture statistics is in stats or we will run into mem error
        * - Set to false otherwise
        * - Raise error if lowmen=False and we will have more that 2^31-1 elements at any stage of the computation
        
        custom_thetacombis: array of inds which theta combis will be selected
        """
        
        ## Preparations ##
        # Build list of statistics to be calculated
        statistics_avail_4pcf = ["4pcf_real", "4pcf_multipole"]
        statistics_avail_nap4 = ["N4", "Nap4", "N4c", "Nap4c"]
        statistics_avail_comp = ["allNap", "all4pcf", "all"]
        statistics_avail_phys = statistics_avail_4pcf + statistics_avail_nap4
        statistics_avail = statistics_avail_4pcf + statistics_avail_nap4 + statistics_avail_comp        
        _statistics = []
        hasintegratedstats = False
        _strbadstats = lambda stat: ("The statistics `%s` has not been implemented yet. "%stat + 
                                     "Currently supported statistics are:\n" + str(statistics_avail))
        if type(statistics) not in [list, str]:
            raise ValueError("The parameter `statistics` should either be a list or a string.")
        if type(statistics) is str:
            if statistics not in statistics_avail:
                raise ValueError(_strbadstats)
            statistics = [statistics]
        if type(statistics) is list:
            if "all" in statistics:
                _statistics = statistics_avail_phys
            elif "all4pcf" in statistics:
                _statistics.append(statistics_avail_4pcf)
            elif "allNap" in statistics:
                _statistics.append(statistics_avail_nap4)
            _statistics = flatlist(_statistics)
            for stat in statistics:
                if stat not in statistics_avail:
                    raise ValueError(_strbadstats)
                if stat in statistics_avail_phys and stat not in _statistics:
                    _statistics.append(stat)
        statistics = list(set(flatlist(_statistics)))
        for stat in statistics:
            if stat in statistics_avail_nap4:
                hasintegratedstats = True
                
        # Check if the output will fit in memory
        if "4pcf_multipole" in statistics:
            _nvals = self.nzcombis*(2*self.nmaxs[0]+1)*(2*self.nmaxs[1]+1)*self.nbinsr**3
            if _nvals>cutlen:
                raise ValueError(("4pcf in multipole basis will cause memory overflow " + 
                                  "(requiring %.2fx10^9 > %.2fx10^9 elements)\n"%(_nvals/1e9, cutlen/1e9) + 
                                  "If you are solely interested in integrated statistics (like Map4), you" +
                                  "only need to add those to the `statistics` argument."))
        if "4pcf_real" in statistics:
            _nvals = self.nzcombis*self.nbinsphi[0]*self.nbinsphi[1]*self.nbinsr**3
            if _nvals>cutlen:
                raise ValueError(("4pcf in real basis will cause memory overflow " + 
                                  "(requiring %.2fx10^9 > %.2fx10^9 elements)\n"%(_nvals/1e9, cutlen/1e9) + 
                                  "If you are solely interested in integrated statistics (like Map4), you" +
                                  "only need to add those to the `statistics` argument."))
                
        # Decide on whether to use low-mem functions or not
        if hasintegratedstats:
            if lowmem in [False, None]:
                if not lowmem:
                    print("Low-memory computation enforced for integrated measures of the 4pcf. " +
                          "Set `lowmem` from `%s` to `True`"%str(lowmem))
                lowmem = True
        else:
            if lowmem in [None, False]:
                maxlen = 0
                _lowmem = False
                if "4pcf_multipole" in statistics:
                    _nvals = self.nzcombis*(2*self.nmaxs[0]+1)*(2*self.nmaxs[1]+1)*self.nbinsr**3
                    if _nvals > cutlen:
                        if not lowmem:
                            print("Switching to low-memory computation of 4pcf in multipole basis.")
                        lowmem = True
                    else:
                        lowmem = False
                if "4pcf_real" in statistics:
                    nvals = self.nzcombis*self.nbinsphi[0]*self.nbinsphi[1]*self.nbinsr**3
                    if _nvals > cutlen:
                        if not lowmem:
                            print("Switching to low-memory computation of 4pcf in real basis.")
                        lowmem = True
                    else:
                        lowmem = False
                        
        # Misc checks            
        self._checkcats(cat, self.spins)
        
        ## Build args for wrapped functions ##
        # Shortcuts
        _nmax = self.nmaxs[0]
        _nnvals = (2*_nmax+1)*(2*_nmax+1)
        _nbinsr3 = self.nbinsr*self.nbinsr*self.nbinsr
        _nphis = len(self.phis[0])
        sc = (2*_nmax+1,2*_nmax+1,self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr)
        szr = (self.nbinsz, self.nbinsr)
        s4pcf = (self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr,_nphis,_nphis)
        # Init default args
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        if not cat.hasspatialhash:
            cat.build_spatialhash(dpix=max(1.,self.max_sep//10.))
        nregions = np.int32(len(np.argwhere(cat.index_matcher>-1).flatten()))
        args_basecat = (cat.isinner.astype(np.float64), cat.weight, cat.pos1, cat.pos2, 
                        np.int32(cat.ngal), )
        args_hash = (cat.index_matcher, cat.pixs_galind_bounds, cat.pix_gals, nregions, 
                     np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                     np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
        
        # Init optional args
        __lenflag = 10
        __fillflag = -1
        if "4pcf_multipole" in statistics:
            N_n = np.zeros(_nnvals*self.nzcombis*_nbinsr3).astype(np.complex128)
            alloc_4pcfmultipoles = 1
        else:
            N_n = __fillflag*np.zeros(__lenflag).astype(np.complex128)
            alloc_4pcfmultipoles = 0
        if "4pcf_real" in statistics:
            fourpcf = np.zeros(_nphis*_nphis*self.nzcombis*_nbinsr3).astype(np.complex128)
            alloc_4pcfreal = 1
        else:
            fourpcf = __fillflag*np.ones(__lenflag).astype(np.complex128)
            alloc_4pcfreal = 0
        if hasintegratedstats:
            if mapradii is None:
                raise ValueError("Aperture radii need to be specified in variable `mapradii`.")
            mapradii = mapradii.astype(np.float64)
            N4correlators = np.zeros(self.nzcombis*len(mapradii)).astype(np.complex128)
        else:
            mapradii = __fillflag*np.ones(__lenflag).astype(np.float64)
            N4correlators =  __fillflag*np.ones(__lenflag).astype(np.complex128)

        
        # Build args based on chosen methods
        if self.method=="Discrete" and not lowmem:
            raise NotImplementedError
        if self.method=="Discrete" and lowmem:
            raise NotImplementedError
        if self.method=="Tree" and lowmem:
            # Prepare mask for nonredundant theta- and multipole configurations
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads, batchsize=batchsize, 
                                                     batchsize_max=self.thetabatchsize_max, ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python)
            _, _, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            assert(self.nmaxs[0]==self.nmaxs[1])
            _resmultipoles = gen_n2n3indices_Upsfourth(self.nmaxs[0])
            _shape, _inds, _n2s, _n3s = _resmultipoles
            
            # Prepare reduced catalogs
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                  shuffle=self.shuffle_pix, normed=False)
            (ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, 
             index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true) = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            isinner_resos = np.concatenate(isinners).astype(np.float64)
            index_matcher_resos = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds_resos = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals_resos = np.concatenate(pix_gals).astype(np.int32)
            index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
            nregions = len(index_matcher_flat)
            # Build args
            args_basesetup = (np.int32(_nmax), 
                              np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                              np.int32(self.multicountcorr),
                              _inds, np.int32(len(_inds)), self.phis[0].astype(np.float64), 
                              2*np.pi/_nphis*np.ones(_nphis, dtype=np.float64), np.int32(_nphis), )
            args_resos = (np.int32(self.tree_nresos), self.tree_redges, np.array(ngal_resos, dtype=np.int32),
                          isinner_resos, weight_resos, pos1_resos, pos2_resos,
                          index_matcher_resos, pixs_galind_bounds_resos, pix_gals_resos, np.int32(nregions), )
            args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                         np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
            args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
            args_nap4 = (mapradii, np.int32(len(mapradii)), N4correlators)
            args_4pcf = (np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal), 
                         bin_centers, N_n, fourpcf)
            args = (*args_basecat,
                    *args_basesetup,
                    *args_resos,
                    *args_hash,
                    *args_thetas,
                    np.int32(self.nthreads),
                    *args_nap4,
                    *args_4pcf)
            func = self.clib.alloc_notomoNap4_tree_nnnn 

        # Optionally print the arguments 
        if self._verbose_debug:
            print("We pass the following arguments:")
            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                try:
                    toprint += (func.argtypes[elarg], )
                    print(toprint)
                    print(arg)
                except:
                    print("We did have a problem for arg %i"%elarg)
        
        ## Compute 4th order stats ##
        func(*args)
        
        ## Massage the output ##
        istatout = ()
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        if "4pcf_multipole" in statistics:
            self.npcf_multipoles = N_n.reshape(sc)
        if "4pcf_real" in statistics:
            if lowmem:
                self.npcf = fourpcf.reshape(s4pcf)
            else:
                if self._verbose_python:
                    print("Transforming output to real space basis")
                self.multipoles2npcf_c()
        if hasintegratedstats:
            if "N4" in statistics:
                istatout += (N4correlators.reshape((self.nzcombis,len(mapradii))), )
            # TODO allocate map4, map4c etc.
            
        return istatout

    def multipoles2npcf_singlethetcombi(self, elthet1, elthet2, elthet3):
        r""" Converts a 4PCF in the multipole basis in the real space basis for a fixed combination of radial bins.

        Returns:
        --------
        npcf_out: np.ndarray
            Natural 4PCF components in the real-space basis for all angular combinations.
        npcf_norm_out: np.ndarray
            4PCF weighted counts in the real-space basis for all angular combinations.
        """
        
        _phis1 = self.phis[0].astype(np.float64)
        _phis2 = self.phis[1].astype(np.float64)
        _nphis1 = len(self.phis[0])
        _nphis2 = len(self.phis[1])
        nnvals, _, nzcombis, nbinsr, _, _ = np.shape(self.npcf_multipoles)
        
        N_in = self.npcf_multipoles[...,elthet1,elthet2,elthet3].flatten()
        npcf_out = np.zeros(nzcombis*_nphis1*_nphis2, dtype=np.complex128)
        
        self.clib.multipoles2npcf_nnnn_singletheta(
            N_in, self.nmaxs[0], self.nmaxs[1],
            self.bin_centers_mean[elthet1], self.bin_centers_mean[elthet2], self.bin_centers_mean[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            npcf_out)
        
        return npcf_out.reshape(( _nphis1,_nphis2))


class GGGGCorrelation_NoTomo(BinnedNPCF):
    r""" Class containing methods to measure and and obtain statistics that are built
    from nontomographic fourth-order shear correlation functions.
    
    Attributes
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.
    thetabatchsize_max: int, optional
        The largest number of radial bin combinations that are processed in parallel.
        Defaults to ``10 000``.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    Additional child-specific parameters can be passed via ``kwargs``. 
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme .
    
    """
    
    def __init__(self, min_sep, max_sep, thetabatchsize_max=10000, method="Tree", **kwargs):
        
        super().__init__(order=4, spins=np.array([2,2,2,2], dtype=np.int32),
                         n_cfs=8, min_sep=min_sep, max_sep=max_sep, 
                         method=method, methods_avail=["Discrete", "Tree"], **kwargs)
        
        self.thetabatchsize_max = thetabatchsize_max
        self.projection = None
        self.projections_avail = [None, "X", "Centroid"]
        self.proj_dict = {"X":0, "Centroid":1}
        self.nbinsz = 1
        self.nzcombis = 1
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        self.project["X"]["Centroid"] = self._x2centroid
        
    def process(self, cat, statistics="all", tofile=False, apply_edge_correction=False, projection="X",
                lowmem=None, mapradii=None, batchsize=None, custom_thetacombis=None, cutlen=2**31-1):
        r"""
        Arguments:
        
        Logic works as follows:
        * Keyword 'statistics' \in [4pcf_real, 4pcf_multipoles, M4, Map4, M4c, Map4c, allMap, all4pcf, all]
        * - If 4pcf_multipoles in statistics --> save 4pcf_multipoles
        * - If 4pcf_real in statistics --> save 4pcf_real
        * - If only M4 in statistics --> Do not save any 4pcf. This is really the lowmem case.
        * - allMap, all4pcf, all are abbreviations as expected
        * If lowmem=True, uses the inefficient, but lowmem function for computation and output statistics 
        from there as wanted.
        * If lowmem=False, use the fast functions to do the 4pcf multipole computation and do 
        the potential conversions lateron.
        * Default lowmem to None and
        * - Set to true if any aperture statistics is in stats or we will run into mem error
        * - Set to false otherwise
        * - Raise error if lowmen=False and we will have more that 2^31-1 elements at any stage of the computation
        
        custom_thetacombis: array of inds which theta combis will be selected
        """
        
        ## Preparations ##
        # Build list of statistics to be calculated
        statistics_avail_4pcf = ["4pcf_real", "4pcf_multipole"]
        statistics_avail_map4 = ["M4", "Map4", "M4c", "Map4c"]
        statistics_avail_comp = ["allMap", "all4pcf", "all"]
        statistics_avail_phys = statistics_avail_4pcf + statistics_avail_map4
        statistics_avail = statistics_avail_4pcf + statistics_avail_map4 + statistics_avail_comp        
        _statistics = []
        hasintegratedstats = False
        _strbadstats = lambda stat: ("The statistics `%s` has not been implemented yet. "%stat + 
                                     "Currently supported statistics are:\n" + str(statistics_avail))
        if type(statistics) not in [list, str]:
            raise ValueError("The parameter `statistics` should either be a list or a string.")
        if type(statistics) is str:
            if statistics not in statistics_avail:
                raise ValueError(_strbadstats)
            statistics = [statistics]
        if type(statistics) is list:
            if "all" in statistics:
                _statistics = statistics_avail_phys
            elif "all4pcf" in statistics:
                _statistics.append(statistics_avail_4pcf)
            elif "allMap" in statistics:
                _statistics.append(statistics_avail_map4)
            _statistics = flatlist(_statistics)
            for stat in statistics:
                if stat not in statistics_avail:
                    raise ValueError(_strbadstats)
                if stat in statistics_avail_phys and stat not in _statistics:
                    _statistics.append(stat)
        statistics = list(set(flatlist(_statistics)))
        for stat in statistics:
            if stat in statistics_avail_map4:
                hasintegratedstats = True
                
        # Check if the output will fit in memory
        if "4pcf_multipole" in statistics:
            _nvals = 8*self.nzcombis*(2*self.nmaxs[0]+1)*(2*self.nmaxs[1]+1)*self.nbinsr**3
            if _nvals>cutlen:
                raise ValueError(("4pcf in multipole basis will cause memory overflow " + 
                                  "(requiring %.2fx10^9 > %.2fx10^9 elements)\n"%(_nvals/1e9, cutlen/1e9) + 
                                  "If you are solely interested in integrated statistics (like Map4), you" +
                                  "only need to add those to the `statistics` argument."))
        if "4pcf_real" in statistics:
            _nvals = 8*self.nzcombis*self.nbinsphi[0]*self.nbinsphi[1]*self.nbinsr**3
            if _nvals>cutlen:
                raise ValueError(("4pcf in real basis will cause memory overflow " + 
                                  "(requiring %.2fx10^9 > %.2fx10^9 elements)\n"%(_nvals/1e9, cutlen/1e9) + 
                                  "If you are solely interested in integrated statistics (like Map4), you" +
                                  "only need to add those to the `statistics` argument."))
                
        # Decide on whether to use low-mem functions or not
        if hasintegratedstats:
            if lowmem in [False, None]:
                if not lowmem:
                    print("Low-memory computation enforced for integrated measures of the 4pcf. " +
                          "Set `lowmem` from `%s` to `True`"%str(lowmem))
                lowmem = True
        else:
            if lowmem in [None, False]:
                maxlen = 0
                _lowmem = False
                if "4pcf_multipole" in statistics:
                    _nvals = 8*self.nzcombis*(2*self.nmaxs[0]+1)*(2*self.nmaxs[1]+1)*self.nbinsr**3
                    if _nvals > cutlen:
                        if not lowmem:
                            print("Switching to low-memory computation of 4pcf in multipole basis.")
                        lowmem = True
                    else:
                        lowmem = False
                if "4pcf_real" in statistics:
                    nvals = 8*self.nzcombis*self.nbinsphi[0]*self.nbinsphi[1]*self.nbinsr**3
                    if _nvals > cutlen:
                        if not lowmem:
                            print("Switching to low-memory computation of 4pcf in real basis.")
                        lowmem = True
                    else:
                        lowmem = False
                        
        # Misc checks            
        assert(projection in self.projections_avail)
        self._checkcats(cat, self.spins)
        i_projection = np.int32(self.proj_dict[projection])
        
        ## Build args for wrapped functions ##
        # Shortcuts
        _nmax = self.nmaxs[0]
        _nnvals = (2*_nmax+1)*(2*_nmax+1)
        _nbinsr3 = self.nbinsr*self.nbinsr*self.nbinsr
        _nphis = len(self.phis[0])
        sc = (8,2*_nmax+1,2*_nmax+1,self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr)
        sn = (2*_nmax+1,2*_nmax+1,self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr)
        szr = (self.nbinsz, self.nbinsr)
        s4pcf = (8,self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr,_nphis,_nphis)
        s4pcfn = (self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr,_nphis,_nphis)
        # Init default args
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        if not cat.hasspatialhash:
            cat.build_spatialhash(dpix=max(1.,self.max_sep//10.))
        nregions = np.int32(len(np.argwhere(cat.index_matcher>-1).flatten()))
        args_basecat = (cat.isinner.astype(np.float64), cat.weight, cat.pos1, cat.pos2, 
                        cat.tracer_1, cat.tracer_2, np.int32(cat.ngal), )
        args_hash = (cat.index_matcher, cat.pixs_galind_bounds, cat.pix_gals, nregions, 
                     np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                     np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
        
        # Init optional args
        __lenflag = 10
        __fillflag = -1
        if "4pcf_multipole" in statistics:
            Upsilon_n = np.zeros(self.n_cfs*_nnvals*self.nzcombis*_nbinsr3).astype(np.complex128)
            N_n = np.zeros(_nnvals*self.nzcombis*_nbinsr3).astype(np.complex128)
            alloc_4pcfmultipoles = 1
        else:
            Upsilon_n = __fillflag*np.ones(__lenflag).astype(np.complex128)
            N_n = __fillflag*np.zeros(__lenflag).astype(np.complex128)
            alloc_4pcfmultipoles = 0
        if "4pcf_real" in statistics:
            fourpcf = np.zeros(8*_nphis*_nphis*self.nzcombis*_nbinsr3).astype(np.complex128)
            fourpcf_norm = np.zeros(_nphis*_nphis*self.nzcombis*_nbinsr3).astype(np.complex128)
            alloc_4pcfreal = 1
        else:
            fourpcf = __fillflag*np.ones(__lenflag).astype(np.complex128)
            fourpcf_norm = __fillflag*np.ones(__lenflag).astype(np.complex128)
            alloc_4pcfreal = 0
        if hasintegratedstats:
            if mapradii is None:
                raise ValueError("Aperture radii need to be specified in variable `mapradii`.")
            mapradii = mapradii.astype(np.float64)
            M4correlators = np.zeros(8*self.nzcombis*len(mapradii)).astype(np.complex128)
        else:
            mapradii = __fillflag*np.ones(__lenflag).astype(np.float64)
            N4correlators =  __fillflag*np.ones(__lenflag).astype(np.complex128)
        
        # Build args based on chosen methods
        if self.method=="Discrete" and not lowmem:
            args_basesetup = (np.int32(_nmax), np.float64(self.min_sep), 
                              np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
                              np.int32(self.nbinsr), np.int32(self.multicountcorr), )
            args = (*args_basecat,
                    *args_basesetup,
                    *args_hash,
                    np.int32(self.nthreads),
                    np.int32(self._verbose_c+self._verbose_debug),
                    bin_centers,
                    Upsilon_n,
                    N_n)
            func = self.clib.alloc_notomoGammans_discrete_gggg 
        if self.method=="Discrete" and lowmem:
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads, batchsize=batchsize, 
                                                     batchsize_max=self.thetabatchsize_max, ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python)
            _, _, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            
            args_basesetup = (np.int32(_nmax), 
                              np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                              np.int32(self.multicountcorr),
                              self.phis[0].astype(np.float64), 
                              2*np.pi/_nphis*np.ones(_nphis, dtype=np.float64), np.int32(_nphis))
            args_4pcf = (np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal), 
                         bin_centers, Upsilon_n, N_n, fourpcf, fourpcf_norm, )
            args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
            args_map4 = (mapradii, np.int32(len(mapradii)), M4correlators)
            args = (*args_basecat,
                    *args_basesetup,
                    *args_hash,
                    *args_thetas,
                    np.int32(self.nthreads),
                    np.int32(self._verbose_c+self._verbose_debug),
                    i_projection,
                    *args_map4,
                    *args_4pcf)
            func = self.clib.alloc_notomoMap4_disc_gggg  
        if self.method=="Tree":
            # Prepare mask for nonredundant theta- and multipole configurations
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads, batchsize=batchsize, 
                                                     batchsize_max=self.thetabatchsize_max, ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python*lowmem)
            _, _, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            assert(self.nmaxs[0]==self.nmaxs[1])
            _resmultipoles = gen_n2n3indices_Upsfourth(self.nmaxs[0])
            _shape, _inds, _n2s, _n3s = _resmultipoles
            
            # Prepare reduced catalogs
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                  shuffle=self.shuffle_pix, w2field=True, normed=True)
            (ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, 
             index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true) = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            isinner_resos = np.concatenate(isinners).astype(np.float64)
            e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
            e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
            index_matcher_resos = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds_resos = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals_resos = np.concatenate(pix_gals).astype(np.int32)
            index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
            nregions = len(index_matcher_flat)
            if not lowmem:
                args_basesetup = (np.int32(_nmax), 
                                  np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                                  np.int32(cumnthetacombis_batches[-1]), np.int32(self.multicountcorr),
                                  _inds, np.int32(len(_inds)),)
                args_resos = (np.int32(self.tree_nresos), self.tree_redges, np.array(ngal_resos, dtype=np.int32),
                            isinner_resos, weight_resos, pos1_resos, pos2_resos, e1_resos, e2_resos,
                            index_matcher_resos, pixs_galind_bounds_resos, pix_gals_resos, np.int32(nregions), )
                args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
                args_out = ( bin_centers, Upsilon_n, N_n, )
                args = (*args_basecat,
                        *args_basesetup,
                        *args_resos,
                        *args_hash,
                        np.int32(self.nthreads),
                        np.int32(self._verbose_c+self._verbose_debug),
                        *args_out)
                func = self.clib.alloc_notomoGammans_tree_gggg  
            if lowmem:
                # Build args
                args_basesetup = (np.int32(_nmax), 
                                np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                                np.int32(self.multicountcorr),
                                _inds, np.int32(len(_inds)), self.phis[0].astype(np.float64), 
                                2*np.pi/_nphis*np.ones(_nphis, dtype=np.float64), np.int32(_nphis), )
                args_resos = (np.int32(self.tree_nresos), self.tree_redges, np.array(ngal_resos, dtype=np.int32),
                            isinner_resos, weight_resos, pos1_resos, pos2_resos, e1_resos, e2_resos,
                            index_matcher_resos, pixs_galind_bounds_resos, pix_gals_resos, np.int32(nregions), )
                args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
                args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
                args_map4 = (mapradii, np.int32(len(mapradii)), M4correlators)
                args_4pcf = (np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal), 
                            bin_centers, Upsilon_n, N_n, fourpcf, fourpcf_norm, )
                args = (*args_basecat,
                        *args_basesetup,
                        *args_resos,
                        *args_hash,
                        *args_thetas,
                        np.int32(self.nthreads),
                        np.int32(self._verbose_c+self._verbose_debug),
                        i_projection,
                        *args_map4,
                        *args_4pcf)
                func = self.clib.alloc_notomoMap4_tree_gggg  

        # Optionally print the arguments 
        if self._verbose_debug:
            print("We pass the following arguments:")
            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                try:
                    toprint += (func.argtypes[elarg], )
                    print(toprint)
                    print(arg)
                except:
                    print("We did have a problem for arg %i"%elarg)
        
        ## Compute 4th order stats ##
        func(*args)
        self.projection = projection
        
        ## Massage the output ##
        istatout = ()
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        if "4pcf_multipole" in statistics:
            self.npcf_multipoles = Upsilon_n.reshape(sc)
            self.npcf_multipoles_norm = N_n.reshape(sn)
        if "4pcf_real" in statistics:
            if lowmem:
                self.npcf = fourpcf.reshape(s4pcf)
                self.npcf_norm = fourpcf_norm.reshape(s4pcfn) 
            else:
                if self._verbose_python:
                    print("Transforming output to real space basis")
                self.multipoles2npcf_c(projection=projection)
        if hasintegratedstats:
            if "M4" in statistics:
                istatout += (M4correlators.reshape((8,self.nzcombis,len(mapradii))), )
            # TODO allocate map4, map4c etc.
            
        return istatout
    
    def multipoles2npcf_c(self, projection="X"):
        r""" Converts a 4PCF in the multipole basis in the real space basis.
        """
        assert((projection in self.proj_dict.keys()) and (projection in self.projections_avail))
        
        _nzero1 = self.nmaxs[0]
        _nzero2 = self.nmaxs[1]
        _phis1 = self.phis[0].astype(np.float64)
        _phis2 = self.phis[1].astype(np.float64)
        _nphis1 = len(self.phis[0])
        _nphis2 = len(self.phis[1])
        ncfs, nnvals, _, nzcombis, nbinsr, _, _ = np.shape(self.npcf_multipoles)
        
        shape_npcf = (self.n_cfs, nzcombis, nbinsr, nbinsr, nbinsr, _nphis1, _nphis2)
        shape_npcf_norm = (nzcombis, nbinsr, nbinsr, nbinsr, _nphis1, _nphis2)
        self.npcf = np.zeros(self.n_cfs*nzcombis*nbinsr*nbinsr*nbinsr*_nphis1*_nphis2, dtype=np.complex128)
        self.npcf_norm = np.zeros(nzcombis*nbinsr*nbinsr*nbinsr*_nphis1*_nphis2, dtype=np.complex128)
        self.clib.multipoles2npcf_gggg(self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(), 
                                       self.bin_centers_mean.astype(np.float64), np.int32(self.proj_dict[projection]),
                                       8, nbinsr, self.nmaxs[0].astype(np.int32), _phis1, _nphis1, _phis2, _nphis2,
                                       self.nthreads, self.npcf, self.npcf_norm)
        self.npcf = self.npcf.reshape(shape_npcf)
        self.npcf_norm = self.npcf_norm.reshape(shape_npcf_norm)
        self.projection = projection
        
        
    def multipoles2npcf_singlethetcombi(self, elthet1, elthet2, elthet3, projection="X"):
        r""" Converts a 4PCF in the multipole basis in the real space basis for a fixed combination of radial bins.

        Returns:
        --------
        npcf_out: np.ndarray
            Natural 4PCF components in the real-space bassi for all angular combinations.
        npcf_norm_out: np.ndarray
            4PCF weighted counts in the real-space bassi for all angular combinations.
        """
        assert((projection in self.proj_dict.keys()) and (projection in self.projections_avail))
        
        _phis1 = self.phis[0].astype(np.float64)
        _phis2 = self.phis[1].astype(np.float64)
        _nphis1 = len(self.phis[0])
        _nphis2 = len(self.phis[1])
        ncfs, nnvals, _, nzcombis, nbinsr, _, _ = np.shape(self.npcf_multipoles)
        
        Upsilon_in = self.npcf_multipoles[...,elthet1,elthet2,elthet3].flatten()
        N_in = self.npcf_multipoles_norm[...,elthet1,elthet2,elthet3].flatten()
        npcf_out = np.zeros(self.n_cfs*nzcombis*_nphis1*_nphis2, dtype=np.complex128)
        npcf_norm_out = np.zeros(nzcombis*_nphis1*_nphis2, dtype=np.complex128)
        
        self.clib.multipoles2npcf_gggg_singletheta(
            Upsilon_in, N_in, self.nmaxs[0], self.nmaxs[1],
            self.bin_centers_mean[elthet1], self.bin_centers_mean[elthet2], self.bin_centers_mean[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            np.int32(self.proj_dict[projection]), npcf_out, npcf_norm_out)
        
        return npcf_out.reshape((self.n_cfs, _nphis1,_nphis2)), npcf_norm_out.reshape((_nphis1,_nphis2))
                
    def multipoles2npcf_gggg_singletheta_nconvergence(self, elthet1, elthet2, elthet3, projection="X"):
        r""" Checks convergence of the conversion between mutltipole-space and real space for a combination of radial bins.

        Returns:
        --------
        npcf_out: np.ndarray
            Natural 4PCF components in the real-space basis for all angular combinations.
        npcf_norm_out: np.ndarray
            4PCF weighted counts in the real-space basis for all angular combinations.
        """
        assert((projection in self.proj_dict.keys()) and (projection in self.projections_avail))
        
        _phis1 = self.phis[0].astype(np.float64)
        _phis2 = self.phis[1].astype(np.float64)
        _nphis1 = len(self.phis[0])
        _nphis2 = len(self.phis[1])
                
        ncfs, nnvals, _, nzcombis, nbinsr, _, _ = np.shape(self.npcf_multipoles)
        
        Upsilon_in = self.npcf_multipoles[...,elthet1,elthet2,elthet3].flatten()
        N_in = self.npcf_multipoles_norm[...,elthet1,elthet2,elthet3].flatten()
        npcf_out = np.zeros(self.n_cfs*nzcombis*(self.nmaxs[0]+1)*(self.nmaxs[1]+1)*_nphis1*_nphis2, dtype=np.complex128)
        npcf_norm_out = np.zeros(nzcombis*(self.nmaxs[0]+1)*(self.nmaxs[1]+1)*_nphis1*_nphis2, dtype=np.complex128)
        
        self.clib.multipoles2npcf_gggg_singletheta_nconvergence(
            Upsilon_in, N_in, self.nmaxs[0], self.nmaxs[1],
            self.bin_centers_mean[elthet1], self.bin_centers_mean[elthet2], self.bin_centers_mean[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            np.int32(self.proj_dict[projection]), npcf_out, npcf_norm_out)
                
        npcf_out = npcf_out.reshape((self.n_cfs, self.nmaxs[0]+1, self.nmaxs[1]+1, _nphis1, _nphis2))
        npcf_norm_out = npcf_norm_out.reshape((self.nmaxs[0]+1, self.nmaxs[1]+1, _nphis1, _nphis2))
                
        return npcf_out, npcf_norm_out
    
    def computeMap4(self, radii, nmax_trafo=None, basis='MapMx'):
        r"""Computes the fourth-order aperture mass statistcs using the polynomial filter of Crittenden 2002."""

        assert(basis in ['MapMx','MM*','both'])
        
        if nmax_trafo is None:
            nmax_trafo=self.nmaxs[0]
            
        # Retrieve all the aperture measures in the MM* basis via the 5D transformation eqns
        M4correlators = np.zeros(8*len(radii), dtype=np.complex128)
        self.clib.fourpcfmultipoles2M4correlators(
            np.int32(self.nmaxs[0]), np.int32(nmax_trafo),
            self.bin_edges, self.bin_centers_mean, np.int32(self.nbinsr),
            radii.astype(np.float64), np.int32(len(radii)),
            self.phis[0].astype(np.float64), self.phis[1].astype(np.float64), 
            self.dphis[0].astype(np.float64), self.dphis[1].astype(np.float64), 
            len(self.phis[0]), len(self.phis[1]),
            np.int32(self.proj_dict[self.projection]), np.int32(self.nthreads),
            self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(),
            M4correlators)
        res_MMStar = M4correlators.reshape((8,len(radii)))
        
        # Allocate result
        res = ()
        if basis=='MM*' or basis=='both':
            res += (res_MMStar, )
        if basis=='MapMx' or basis=='both':
            res += ( GGGGCorrelation_NoTomo.MMStar2MapMx_fourth(res_MMStar), )
        
        return res               
    
    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        super()._projectnpcf(self, projection)
    
    def _x2centroid(self):
        gammas_cen = np.zeros_like(self.npcf)
        pimod = lambda x: x%(2*np.pi) - 2*np.pi*(x%(2*np.pi)>=np.pi)
        npcf_cen = np.zeros(self.npcf.shape, dtype=complex)
        _centers = np.mean(self.bin_centers, axis=0)
        for elb1, bin1 in enumerate(_centers):
            for elb2, bin2 in enumerate(_centers):
                for elb3, bin3 in enumerate(_centers):
                    phiexp = np.exp(1J*self.phis[0])
                    phiexp_c = np.exp(-1J*self.phis[0])
                    phi12grid, phi13grid = np.meshgrid(phiexp, phiexp)
                    phi12grid_c, phi13grid_c = np.meshgrid(phiexp_c, phiexp_c)
                    prod1 = (bin1   +bin2*phi12grid_c   + bin3*phi13grid_c)  /(bin1   + bin2*phi12grid   + bin3*phi13grid)   #q1
                    prod2 = (3*bin1 -bin2*phi12grid_c   - bin3*phi13grid_c)  /(3*bin1 - bin2*phi12grid   - bin3*phi13grid)   #q2
                    prod3 = (bin1   -3*bin2*phi12grid_c + bin3*phi13grid_c)  /(bin1   - 3*bin2*phi12grid + bin3*phi13grid)   #q3
                    prod4 = (bin1   +bin2*phi12grid_c   - 3*bin3*phi13grid_c)/(bin1   + bin2*phi12grid   - 3*bin3*phi13grid) #q4
                    prod1_inv = prod1.conj()/np.abs(prod1)
                    prod2_inv = prod2.conj()/np.abs(prod2)
                    prod3_inv = prod3.conj()/np.abs(prod3)
                    prod4_inv = prod4.conj()/np.abs(prod4)
                    rot_nom = np.zeros((8,len(self.phis[0]), len(self.phis[1])))
                    rot_nom[0] = pimod(np.angle(prod1    *prod2    *prod3    *prod4     * phi12grid**2   * phi13grid**3))
                    rot_nom[1] = pimod(np.angle(prod1_inv*prod2    *prod3    *prod4     * phi12grid**2   * phi13grid**1))
                    rot_nom[2] = pimod(np.angle(prod1    *prod2_inv*prod3    *prod4     * phi12grid**2   * phi13grid**3))
                    rot_nom[3] = pimod(np.angle(prod1    *prod2    *prod3_inv*prod4     * phi12grid_c**2 * phi13grid**3))
                    rot_nom[4] = pimod(np.angle(prod1    *prod2    *prod3    *prod4_inv * phi12grid**2   * phi13grid_c**1))
                    rot_nom[5] = pimod(np.angle(prod1_inv*prod2_inv*prod3    *prod4     * phi12grid**2   * phi13grid**1))
                    rot_nom[6] = pimod(np.angle(prod1_inv*prod2    *prod3_inv*prod4     * phi12grid_c**2 * phi13grid**1))
                    rot_nom[7] = pimod(np.angle(prod1_inv*prod2    *prod3    *prod4_inv * phi12grid**2   * phi13grid_c**3))
                    gammas_cen[:,:,elb1,elb2,elb3] = self.npcf[:,:,elb1,elb2,elb3]*np.exp(1j*rot_nom)[:,np.newaxis,:,:]
        return gammas_cen
    
    ## GAUSSIAN-FIELD SPECIFIC FUNCTIONS ##
    # Deprecate this as it has been ported to c
    @staticmethod
    def fourpcf_gauss_x(theta1, theta2, theta3, phi12, phi13, xipspl, ximspl):
        """ Computes disconnected part of the 4pcf in the 'x'-projection
        given a splined 2pcf
        """
        allgammas = [None]*8
        xprojs = [None]*8
        y1 = theta1 * np.ones_like(phi12)
        y2 = theta2*np.exp(1j*phi12)
        y3 = theta3*np.exp(1j*phi13)
        absy1 = np.abs(y1)
        absy2 = np.abs(y2)
        absy3 = np.abs(y3)
        absy12 = np.abs(y2-y1)
        absy13 = np.abs(y1-y3)
        absy23 = np.abs(y3-y2)
        q1 = -0.25*(y1+y2+y3)
        q2 = 0.25*(3*y1-y2-y3)
        q3 = 0.25*(3*y2-y3-y1)
        q4 = 0.25*(3*y3-y1-y2)
        q1c = q1.conj(); q2c = q2.conj(); q3c = q3.conj(); q4c = q4.conj(); 
        y123_cub = (np.abs(y1)*np.abs(y2)*np.abs(y3))**3
        ang1_4 = ((y1)/absy1)**4; ang2_4 = ((y2)/absy2)**4; ang3_4 = ((y3)/absy3)**4
        ang12_4 = ((y2-y1)/absy12)**4; ang13_4 = ((y3-y1)/absy13)**4; ang23_4 = ((y3-y2)/absy23)**4; 
        xprojs[0] = (y1**3*y2**2*y3**3)/(np.abs(y1)**3*np.abs(y2)**2*np.abs(y3)**3)
        xprojs[1] = (y1**1*y2**2*y3**1)/(np.abs(y1)**1*np.abs(y2)**2*np.abs(y3)**1)
        xprojs[2] = (y1**-1*y2**2*y3**3)/(np.abs(y1)**-1*np.abs(y2)**2*np.abs(y3)**3)
        xprojs[3] = (y1**3*y2**-2*y3**3)/(np.abs(y1)**3*np.abs(y2)**-2*np.abs(y3)**3)
        xprojs[4] = (y1**3*y2**2*y3**-1)/(np.abs(y1)**3*np.abs(y2)**2*np.abs(y3)**-1)
        xprojs[5] = (y1**-3*y2**2*y3**1)/(np.abs(y1)**-3*np.abs(y2)**2*np.abs(y3)**1)
        xprojs[6] = (y1**1*y2**-2*y3**1)/(np.abs(y1)**1*np.abs(y2)**-2*np.abs(y3)**1)
        xprojs[7] = (y1**1*y2**2*y3**-3)/(np.abs(y1)**1*np.abs(y2)**2*np.abs(y3)**-3)
        allgammas[0] = 1./xprojs[0] * (
            ang23_4 * ang1_4 * ximspl(absy23) * ximspl(absy1) +
            ang13_4 * ang2_4 * ximspl(absy13) * ximspl(absy2) + 
            ang12_4 * ang3_4 * ximspl(absy12) * ximspl(absy3))
        allgammas[1] = 1./xprojs[1] * (
            ang23_4 * xipspl(absy1) * ximspl(absy23) + 
            ang13_4 * xipspl(absy2) * ximspl(absy13) + 
            ang12_4 * xipspl(absy3) * ximspl(absy12))
        allgammas[2] = 1./xprojs[2] * (
            ang23_4 * xipspl(absy1) * ximspl(absy23) + 
            ang2_4  * ximspl(absy2) * xipspl(absy13) + 
            ang3_4  * ximspl(absy3) * xipspl(absy12))
        allgammas[3] = 1./xprojs[3] * (
            ang1_4  * ximspl(absy1) * xipspl(absy23) + 
            ang13_4 * xipspl(absy2) * ximspl(absy13) + 
            ang3_4  * ximspl(absy3) * xipspl(absy12))
        allgammas[4] = 1./xprojs[4] * (
            ang1_4  * ximspl(absy1) * xipspl(absy23) + 
            ang2_4  * ximspl(absy2) * xipspl(absy13) + 
            ang12_4 * xipspl(absy3) * ximspl(absy12))
        allgammas[5] = 1./xprojs[5] * (
            ang1_4.conj() * ang23_4 * ximspl(absy23) * ximspl(absy1) +
                                      xipspl(absy13) * xipspl(absy2) + 
                                      xipspl(absy12) * xipspl(absy3))
        allgammas[6] = 1./xprojs[6] * (
                                      xipspl(absy23) * xipspl(absy1) +
            ang2_4.conj() * ang13_4 * ximspl(absy13) * ximspl(absy2) + 
                                      xipspl(absy12) * xipspl(absy3))
        allgammas[7] = 1./xprojs[7] * (
                                      xipspl(absy23) * xipspl(absy1) +
                                      xipspl(absy13) * xipspl(absy2) + 
            ang3_4.conj() * ang12_4 * ximspl(absy12) * ximspl(absy3))
    
        return allgammas        
    
    # Disconnected 4pcf from binned 2pcf (might want to deprecate this as it is a special case of nsubr==1)
    def __gauss4pcf_analytic(self, theta1, theta2, theta3, xip_arr, xim_arr, thetamin_xi, thetamax_xi, dtheta_xi):
        gausss_4pcf = np.zeros(8*len(self.phis[0])*len(self.phis[0]),dtype=np.complex128)
        self.clib.gauss4pcf_analytic(theta1.astype(np.float64), 
                                     theta2.astype(np.float64),
                                     theta3.astype(np.float64),
                                     self.phis[0].astype(np.float64), np.int32(len(self.phis[0])),
                                     xip_arr.astype(np.float64), xim_arr.astype(np.float64),
                                     thetamin_xi, thetamax_xi, dtheta_xi,
                                     gausss_4pcf)
        return gausss_4pcf
    
    
    # [Debug] Disconnected 4pcf from analytic 2pcf
    def gauss4pcf_analytic(self, itheta1, itheta2, itheta3, nsubr, 
                                 xip_arr, xim_arr, thetamin_xi, thetamax_xi, dtheta_xi):
    
        gauss_4pcf = np.zeros(8*self.nbinsphi[0]*self.nbinsphi[1],dtype=np.complex128)

        self.clib.gauss4pcf_analytic_integrated(
            np.int32(itheta1), 
            np.int32(itheta2), 
            np.int32(itheta3), 
            np.int32(nsubr), 
            self.bin_edges.astype(np.float64),
            np.int32(self.nbinsr),
            self.phis[0].astype(np.float64),
            np.int32(self.nbinsphi[0]),
            xip_arr.astype(np.float64), 
            xim_arr.astype(np.float64),
            np.float64(thetamin_xi), 
            np.float64(thetamax_xi), 
            np.float64(dtheta_xi), 
            gauss_4pcf)
        return gauss_4pcf.reshape((8, self.nbinsphi[0], self.nbinsphi[1]))
    
    # Compute disconnected part of 4pcf in multiple basis
    def gauss4pcf_multipolebasis(self, itheta1, itheta2, itheta3, nsubr, 
                                 xip_arr, xim_arr, thetamin_xi, thetamax_xi, dtheta_xi):
        
        # Obtain integrated 4pcf
        int_4pcf = self.gauss4pcf_analytic_integrated(itheta1, itheta2, itheta3, nsubr, 
                                                      xip_arr, xim_arr, 
                                                      thetamin_xi, thetamax_xi, dtheta_xi)
        
        # Transform to multiple basis (cf eq xxx in P25)
        phigrid1, phigrid2 = np.meshgrid(self.phis[0],self.phis[1])
        gauss_multipoles = np.zeros((8,2*self.nmaxs[0]+1,2*self.nmaxs[1]+1),dtype=complex)
        for eln2,n2 in enumerate(np.arange(-self.nmaxs[0],self.nmaxs[0]+1)):
            fac1 = np.e**(-1J*n2*phigrid1)
            for eln3,n3 in enumerate(np.arange(-self.nmaxs[1],self.nmaxs[1]+1)):
                fac2 = np.e**(-1J*n3*phigrid2)
                for elcomp in range(8):
                    gauss_multipoles[elcomp,eln2,eln3] = np.mean(int_4pcf[elcomp]*fac1*fac2)
                    
        return gauss_multipoles
    

    def estimateMap4disc(self, cat, radii, basis='MapMx',fac_minsep=0.05, fac_maxsep=2., binsize=0.1, nsubr=3, nsubsample_filter=1):
        """ Estimate disconnected part of fourth-order aperture statistics on a shape catalog. """

        # Compute shear 2pcf from data
        min_sep_disc = fac_minsep*self.min_sep
        max_sep_disc = fac_maxsep*self.max_sep
        binsize_disc = min(0.1,self.binsize)
        ggcorr = GGCorrelation(min_sep=min_sep_disc, max_sep=max_sep_disc,binsize=binsize_disc, 
                               rmin_pixsize=self.rmin_pixsize, tree_resos=self.tree_resos, nthreads=self.nthreads)
        ggcorr.process(cat)

        # Convert this to fourth-order aperture statistics
        linarr = np.linspace(min_sep_disc,max_sep_disc,int(max_sep_disc/(binsize_disc*min_sep_disc)))
        xip_spl = interp1d(x=ggcorr.bin_centers_mean,y=ggcorr.xip[0].real,fill_value=0,bounds_error=False)
        xim_spl = interp1d(x=ggcorr.bin_centers_mean,y=ggcorr.xim[0].real,fill_value=0,bounds_error=False)
        mapstat = self.Map4analytic(mapradii=radii,
                                    xip_spl=xip_spl, 
                                    xim_spl=xim_spl,
                                    thetamin_xi=linarr[0],
                                    thetamax_xi=linarr[-1],
                                    ntheta_xi=len(linarr),
                                    nsubr=nsubr,nsubsample_filter=nsubsample_filter,basis=basis)
        return mapstat


    # Disconnected part of Map^4 from analytic 2pcf
    # thetamin_xi, thetamax_xi, ntheta_xi is the linspaced array in which the xipm are passed to the external function
    def Map4analytic(self, mapradii, xip_spl, xim_spl, thetamin_xi, thetamax_xi, ntheta_xi, 
                     nsubr=1, nsubsample_filter=1, batchsize=None, basis='MapMx'):
        
        self.nbinsz = 1
        self.nzcombis = 1
        _nmax = self.nmaxs[0]
        _nnvals = (2*_nmax+1)*(2*_nmax+1)
        _nbinsr3 = self.nbinsr*self.nbinsr*self.nbinsr
        _nphis = len(self.phis[0])
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        M4correlators = np.zeros(8*self.nzcombis*len(mapradii)).astype(np.complex128)
        # Define the radial bin batches
        if batchsize is None:
            batchsize = min(_nbinsr3,min(10000,int(_nbinsr3/self.nthreads)))
            if self._verbose_python:
                print("Using batchsize of %i for radial bins"%batchsize)
        nbatches = np.int32(_nbinsr3/batchsize)
        thetacombis_batches = np.arange(_nbinsr3).astype(np.int32)
        cumnthetacombis_batches = (np.arange(nbatches+1)*_nbinsr3/(nbatches)).astype(np.int32)
        nthetacombis_batches = (cumnthetacombis_batches[1:]-cumnthetacombis_batches[:-1]).astype(np.int32)
        cumnthetacombis_batches[-1] = _nbinsr3
        nthetacombis_batches[-1] = _nbinsr3-cumnthetacombis_batches[-2]
        thetacombis_batches = thetacombis_batches.flatten().astype(np.int32)
        nbatches = len(nthetacombis_batches)

        args_4pcfsetup = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                          self.phis[0].astype(np.float64), 
                          (self.phis[0][1]-self.phis[0][0])*np.ones(_nphis, dtype=np.float64), _nphis, np.int32(nsubr), )
        args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
        args_map4 = (mapradii.astype(np.float64), np.int32(len(mapradii)), )
        thetas_xi = np.linspace(thetamin_xi,thetamax_xi,ntheta_xi+1)
        args_xi = (xip_spl(thetas_xi), xim_spl(thetas_xi), thetamin_xi, thetamax_xi, ntheta_xi, nsubsample_filter, )
        args = (*args_4pcfsetup,
                *args_thetas,
                np.int32(self.nthreads),
                *args_map4,
                *args_xi,
                M4correlators)
        func = self.clib.alloc_notomoMap4_analytic
        
        if self._verbose_debug:
            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                try:
                    toprint += (func.argtypes[elarg], )
                    print(toprint)
                    print(arg)
                except:
                    print("We did have a problem for arg %i"%elarg)

        func(*args)

        res_MMStar = M4correlators.reshape((8,len(mapradii)))
        # Allocate result
        res = ()
        if basis=='MM*' or basis=='both':
            res += (res_MMStar, )
        if basis=='MapMx' or basis=='both':
            res += (GGGGCorrelation_NoTomo.MMStar2MapMx_fourth(res_MMStar), )
        
        return res
    
    def getMultipolesFromSymm(self, nmax_rec, itheta1, itheta2, itheta3, eltrafo):
    
        nmax_alloc = 2*nmax_rec+1
        assert(nmax_alloc<=self.nmaxs[0])

        # Only select relevant n1/n2 indices
        _dn = self.nmaxs[0]-nmax_alloc

        _shape, _inds, _n2s, _n3s = gen_n2n3indices_Upsfourth(nmax_rec)
        Upsn_in = self.npcf_multipoles[:,_dn:-_dn,_dn:-_dn,0,itheta1,itheta2,itheta3].flatten()
        Nn_in = self.npcf_multipoles_norm[_dn:-_dn,_dn:-_dn,0,itheta1,itheta2,itheta3].flatten()
        Upsn_out = np.zeros(8*(2*nmax_rec+1)*(2*nmax_rec+1), dtype=np.complex128)
        Nn_out = np.zeros(1*(2*nmax_rec+1)*(2*nmax_rec+1), dtype=np.complex128)

        self.clib.getMultipolesFromSymm(
            Upsn_in, Nn_in, nmax_rec, eltrafo, _inds, len(_inds), Upsn_out, Nn_out)

        Upsn_out = Upsn_out.reshape((8,(2*nmax_rec+1),(2*nmax_rec+1)))
        Nn_out = Nn_out.reshape(((2*nmax_rec+1),(2*nmax_rec+1)))

        return Upsn_out, Nn_out

    ## MISC HELPERS ##
    @staticmethod
    def MMStar2MapMx_fourth(res_MMStar):
        """ Transforms fourth-order aperture correlators to fourth-order aperture mass.
        See i.e. Eqs (32)-(36) in Silvestre-Rosello+ 2025 (arxiv.org/pdf/2509.07973).
        """
        res_MapMx = np.zeros((16,*res_MMStar.shape[1:]))
        Mcorr2Map4_re = .125*np.array([[+1,+1,+1,+1,+1,+1,+1,+1],
                                    [-1,-1,-1,+1,+1,-1,+1,+1],
                                    [-1,-1,+1,-1,+1,+1,-1,+1],
                                    [-1,-1,+1,+1,-1,+1,+1,-1],
                                    [-1,+1,-1,-1,+1,+1,+1,-1],
                                    [-1,+1,-1,+1,-1,+1,-1,+1],
                                    [-1,+1,+1,-1,-1,-1,+1,+1],
                                    [+1,-1,-1,-1,-1,+1,+1,+1]])
        Mcorr2Map4_im = .125*np.array([[+1,-1,+1,+1,+1,-1,-1,-1],
                                    [+1,+1,-1,+1,+1,-1,+1,+1],
                                    [+1,+1,+1,-1,+1,+1,-1,+1],
                                    [+1,+1,+1,+1,-1,+1,+1,-1],
                                    [-1,-1,+1,+1,+1,+1,+1,+1],
                                    [-1,+1,-1,+1,+1,+1,-1,-1],
                                    [-1,+1,+1,-1,+1,-1,+1,-1],
                                    [-1,+1,+1,+1,-1,-1,-1,+1]])
        res_MapMx[[0,5,6,7,8,9,10,15]] = Mcorr2Map4_re@(res_MMStar.real)
        res_MapMx[[1,2,3,4,11,12,13,14]] = Mcorr2Map4_im@(res_MMStar.imag)
        return res_MapMx


class GNNNCorrelation_NoTomo(BinnedNPCF):
    def __init__(self, min_sep, max_sep, thetabatchsize_max=10000, **kwargs):
        r""" Class containing methods to measure and and obtain statistics that are built
        from third-order source-lens-lens (G3L) correlation functions.
        
        Attributes
        ----------
        min_sep: float
            The smallest distance of each vertex for which the NPCF is computed.
        max_sep: float
            The largest distance of each vertex for which the NPCF is computed.
        nbinsr: int, optional
            The number of radial bins for each vertex of the NPCF. If set to
            ``None`` this attribute is inferred from the ``binsize`` attribute.
        binsize: int, optional
            The logarithmic slize of the radial bins for each vertex of the NPCF. If set to
            ``None`` this attribute is inferred from the ``nbinsr`` attribute.
        nbinsphi: float, optional
            The number of angular bins for the NPCF in the real-space basis. 
            Defaults to ``100``.
        nmaxs: list, optional
            The largest multipole component considered for the NPCF in the multipole basis. 
            Defaults to ``30``.
        method: str, optional
            The method to be employed for the estimator. Defaults to ``DoubleTree``.
        multicountcorr: bool, optional
            Flag on whether to subtract of multiplets in which the same tracer appears more
            than once. Defaults to ``True``.
        shuffle_pix: int, optional
            Choice of how to define centers of the cells in the spatial hash structure.
            Defaults to ``1``, i.e. random positioning.
        tree_resos: list, optional
            The cell sizes of the hierarchical spatial hash structure
        tree_edges: list, optional
            List of radii where the tree changes resolution.
        rmin_pixsize: int, optional
            The limiting radial distance relative to the cell of the spatial hash
            after which one switches to the next hash in the hierarchy. Defaults to ``20``.
        resoshift_leafs: int, optional
            Allows for a difference in how the hierarchical spatial hash is traversed for
            pixels at the base of the NPCF and pixels at leafs. I.e. positive values indicate
            that leafs will be evaluated at a courser resolutions than the base. Defaults to ``0``.
        minresoind_leaf: int, optional
            Sets the smallest resolution in the spatial hash hierarchy which can be used to access
            tracers at leaf positions. If set to ``None`` uses the smallest specified cell size. 
            Defaults to ``None``.
        maxresoind_leaf: int, optional
            Sets the largest resolution in the spatial hash hierarchy which can be used to access
            tracers at leaf positions. If set to ``None`` uses the largest specified cell size. 
            Defaults to ``None``.
        nthreads: int, optional
            The number of openmp threads used for the reduction procedure. Defaults to ``16``.
        bin_centers: numpy.ndarray
            The centers of the radial bins for each combination of tomographic redshifts.
        bin_centers_mean: numpy.ndarray
            The centers of the radial bins averaged over all combination of tomographic redshifts.
        phis: list
            The bin centers for the N-2 angles describing the NPCF 
            in the real-space basis.
        npcf: numpy.ndarray
            The natural components of the NPCF in the real space basis. The different axes
            are specified as follows: ``(component, zcombi, rbin_1, ..., rbin_N-1, phiin_1, phibin_N-2)``.
        npcf_norm: numpy.ndarray
            The normalization of the natural components of the NPCF in the real space basis. The different axes
            are specified as follows: ``(zcombi, rbin_1, ..., rbin_N-1, phiin_1, phibin_N-2)``.
        npcf_multipoles: numpy.ndarray
            The natural components of the NPCF in the multipole basis. The different axes
            are specified as follows: ``(component, zcombi, multipole_1, ..., multipole_N-2, rbin_1, ..., rbin_N-1)``.
        npcf_multipoles_norm: numpy.ndarray
            The normalization of the natural components of the NPCF in the multipole basis. The different axes
            are specified as follows: ``(zcombi, multipole_1, ..., multipole_N-2, rbin_1, ..., rbin_N-1)``.
        is_edge_corrected: bool, optional
            Flag signifying on wheter the NPCF multipoles have beed edge-corrected. Defaults to ``False``.
        """
        super().__init__(4, [2,0,0,0], n_cfs=1, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X"]
        self.proj_dict = {"X":0}
        self.nbinsz_source = 1 # This class does not handle any tomography at the moment, so I fix it here
        self.nbinsz_lens = 1 # This class does not handle any tomography at the moment, so I fix it here
        self.nzcombis = 1
        self.thetabatchsize_max = thetabatchsize_max
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
    
    def process(self, cat_source, cat_lens, statistics="all", tofile=False, apply_edge_correction=False, 
                dotomo_source=True, dotomo_lens=True, 
                lowmem=None, apradii=None, batchsize=None, custom_thetacombis=None, cutlen=2**31-1):
        self._checkcats([cat_source, cat_lens, cat_lens, cat_lens], [2, 0, 0, 0])
        
        # Checks for redshift binning
        if not dotomo_source:
            self.nbinsz_source = 1
            zbins_source = np.zeros(cat_source.ngal, dtype=np.int32)
        else:
            self.nbinsz_source = cat_source.nbinsz
            zbins_source = cat_source.zbins
        if not dotomo_lens:
            self.nbinsz_lens = 1
            zbins_lens = np.zeros(cat_lens.ngal, dtype=np.int32)
        else:
            self.nbinsz_lens = cat_lens.nbinsz
            zbins_lens= cat_lens.zbins

        ## Preparations ##
        # Some default argument resettings
        if self.method=='Discrete' and not lowmem:
            statistics = ['4pcf_multipole']

        # Check memory requirements
        if not lowmem:
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads, batchsize=batchsize, 
                                                     batchsize_max=self.thetabatchsize_max, ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python*lowmem)
            nthetacombis_tot, _, _, _, _, _ = _resradial
            assert(self.nmaxs[0]==self.nmaxs[1])
            _resmultipoles = gen_n2n3indices_Upsfourth(self.nmaxs[0])
            _, _inds, _, _ = _resmultipoles
            ncache_required_out = self.nbinsr*self.nbinsr*self.nbinsr*(2*self.nmaxs[0]+1)*(2*self.nmaxs[1]+1)
            ncache_required_alloc = nthetacombis_tot*len(_inds)*self.nthreads
            if max(ncache_required_out,ncache_required_alloc)>2**31-1:
                raise ValueError("Required memory too large (%.2f /  x 10^9 elements)"%(ncache_required_out/1e9,ncache_required_alloc/1e9))

        # Build list of statistics to be calculated
        statistics_avail_4pcf = ["4pcf_real", "4pcf_multipole"]
        statistics_avail_mapnap3 = ["MN3", "MapNap3", "MN3cc", "MapNap3c"]
        statistics_avail_comp = ["allMapNap3", "all4pcf", "all"]
        statistics_avail_phys = statistics_avail_4pcf + statistics_avail_mapnap3
        statistics_avail = statistics_avail_4pcf + statistics_avail_mapnap3 + statistics_avail_comp        
        _statistics = []
        hasintegratedstats = False
        _strbadstats = lambda stat: ("The statistics `%s` has not been implemented yet. "%stat + 
                                     "Currently supported statistics are:\n" + str(statistics_avail))
        if type(statistics) not in [list, str]:
            raise ValueError("The parameter `statistics` should either be a list or a string.")
        if type(statistics) is str:
            if statistics not in statistics_avail:
                raise ValueError(_strbadstats)
            statistics = [statistics]
        if type(statistics) is list:
            if "all" in statistics:
                _statistics = statistics_avail_phys
            elif "all4pcf" in statistics:
                _statistics.append(statistics_avail_4pcf)
            elif "allMapNap3" in statistics:
                _statistics.append(statistics_avail_mapnap3)
            _statistics = flatlist(_statistics)
            for stat in statistics:
                if stat not in statistics_avail:
                    raise ValueError(_strbadstats)
                if stat in statistics_avail_phys and stat not in _statistics:
                    _statistics.append(stat)
        statistics = list(set(flatlist(_statistics)))
        for stat in statistics:
            if stat in statistics_avail_mapnap3:
                hasintegratedstats = True

        # Init optional args
        __lenflag = 10
        __fillflag = -1
        _nmax = self.nmaxs[0]
        _nnvals = (2*_nmax+1)*(2*_nmax+1)
        _nbinsr3 = self.nbinsr*self.nbinsr*self.nbinsr
        _nphis = len(self.phis[0])
        _r2combis = self.nbinsr*self.nbinsr
        sc = (self.n_cfs, 2*self.nmax+1,  2*self.nmax+1, self.nzcombis, self.nbinsr, self.nbinsr, self.nbinsr)
        sn = (2*self.nmax+1,2*self.nmax+1,self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr)
        szr = (self.nbinsz_source, self.nbinsz_lens, self.nbinsr)
        s4pcf = (self.n_cfs,self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr,_nphis,_nphis)
        s4pcfn = (self.nzcombis,self.nbinsr,self.nbinsr,self.nbinsr,_nphis,_nphis)
        bin_centers = np.zeros(reduce(operator.mul, szr)).astype(np.float64)

        if "4pcf_multipole" in statistics:
            Upsilon_n = np.zeros(self.n_cfs*_nnvals*self.nzcombis*_nbinsr3).astype(np.complex128)
            N_n = np.zeros(_nnvals*self.nzcombis*_nbinsr3).astype(np.complex128)
            alloc_4pcfmultipoles = 1
        else:
            Upsilon_n = __fillflag*np.ones(__lenflag).astype(np.complex128)
            N_n = __fillflag*np.zeros(__lenflag).astype(np.complex128)
            alloc_4pcfmultipoles = 0
        if "4pcf_real" in statistics:
            fourpcf = np.zeros(1*_nphis*_nphis*self.nzcombis*_nbinsr3).astype(np.complex128)
            fourpcf_norm = np.zeros(_nphis*_nphis*self.nzcombis*_nbinsr3).astype(np.complex128)
            alloc_4pcfreal = 1
        else:
            fourpcf = __fillflag*np.ones(__lenflag).astype(np.complex128)
            fourpcf_norm = __fillflag*np.ones(__lenflag).astype(np.complex128)
            alloc_4pcfreal = 0
        if hasintegratedstats:
            if apradii is None:
                raise ValueError("Aperture radii need to be specified in variable `apradii`.")
            apradii = apradii.astype(np.float64)
            MN3correlators = np.zeros(1*self.nzcombis*len(apradii)).astype(np.complex128)
        else:
            apradii = __fillflag*np.ones(__lenflag).astype(np.float64)
            MN3correlators =  __fillflag*np.ones(__lenflag).astype(np.complex128)
        
        # Basic prep
        hash_dpix = max(1.,self.max_sep//10.)
        jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
        cat_source.build_spatialhash(dpix=hash_dpix, extent=jointextent)
        cat_lens.build_spatialhash(dpix=hash_dpix, extent=jointextent)

        args_sourcecat = (cat_source.isinner.astype(np.float64), cat_source.weight.astype(np.float64), 
                          cat_source.pos1.astype(np.float64), cat_source.pos2.astype(np.float64), 
                          cat_source.tracer_1.astype(np.float64), cat_source.tracer_2.astype(np.float64), 
                          np.int32(cat_source.ngal), )
        args_basesetup = (np.int32(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep),
                          np.int32(self.nbinsr), np.int32(self.multicountcorr), )
        
        
        if self.method=="Discrete" and not lowmem:
            hash_dpix = max(1.,self.max_sep//10.)
            jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
            cat_source.build_spatialhash(dpix=hash_dpix, extent=jointextent)
            cat_lens.build_spatialhash(dpix=hash_dpix, extent=jointextent)
            nregions = np.int32(len(np.argwhere(cat_lens.index_matcher>-1).flatten()))
            args_lenscat = (cat_lens.weight.astype(np.float64), 
                            cat_lens.pos1.astype(np.float64), cat_lens.pos2.astype(np.float64), np.int32(cat_lens.ngal), )
            args_hash = (cat_lens.index_matcher, cat_lens.pixs_galind_bounds, cat_lens.pix_gals, nregions, )
            args_pix = (np.float64(cat_lens.pix1_start), np.float64(cat_lens.pix1_d), np.int32(cat_lens.pix1_n), 
                            np.float64(cat_lens.pix2_start), np.float64(cat_lens.pix2_d), np.int32(cat_lens.pix2_n), )
            args_4pcf = (bin_centers, Upsilon_n, N_n, )
            
            args = (*args_sourcecat,
                    *args_lenscat,
                    *args_hash,
                    *args_pix,
                    *args_basesetup,
                    np.int32(self.nthreads),
                    *args_4pcf)
            func = self.clib.alloc_notomoGammans_discrete_gnnn 
        
        if self.method=="Tree":
        # Prepare mask for nonredundant theta- and multipole configurations
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads, batchsize=batchsize, 
                                                     batchsize_max=self.thetabatchsize_max, ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python*lowmem)
            nthetacombis_tot, _, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            assert(self.nmaxs[0]==self.nmaxs[1])
            _resmultipoles = gen_n2n3indices_Upsfourth(self.nmaxs[0])
            _shape, _inds, _n2s, _n3s = _resmultipoles
            
            # Prepare reduced catalogs
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mhash = cat_lens.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                       shuffle=self.shuffle_pix, normed=True)
            (ngal_resos_lens, pos1s_lens, pos2s_lens, weights_lens, _, _, _, 
             index_matchers_lens, pixs_galind_bounds_lens, pix_gals_lens, dpixs1_true_lens, dpixs2_true_lens) = mhash
            weight_resos_lens = np.concatenate(weights_lens).astype(np.float64)
            pos1_resos_lens = np.concatenate(pos1s_lens).astype(np.float64)
            pos2_resos_lens = np.concatenate(pos2s_lens).astype(np.float64)
            index_matcher_resos_lens = np.concatenate(index_matchers_lens).astype(np.int32)
            pixs_galind_bounds_resos_lens = np.concatenate(pixs_galind_bounds_lens).astype(np.int32)
            pix_gals_resos_lens = np.concatenate(pix_gals_lens).astype(np.int32)
            index_matcher_flat = np.argwhere(cat_source.index_matcher>-1).flatten()
            nregions = len(index_matcher_flat)

            args_resos = (np.int32(self.tree_nresos), self.tree_redges, )
            args_resos_lens = (weight_resos_lens, pos1_resos_lens, pos2_resos_lens, np.asarray(ngal_resos_lens).astype(np.int32),)
            args_hash_source = (cat_source.index_matcher, cat_source.pixs_galind_bounds, cat_source.pix_gals, )
            args_mhash_lens = (index_matcher_resos_lens, pixs_galind_bounds_resos_lens, pix_gals_resos_lens, np.int32(nregions), )
            args_hash = (np.float64(cat_lens.pix1_start), np.float64(cat_lens.pix1_d), np.int32(cat_lens.pix1_n), 
                        np.float64(cat_lens.pix2_start), np.float64(cat_lens.pix2_d), np.int32(cat_lens.pix2_n), )
            if lowmem:
                # Build args
                args_indsetup = (_inds, np.int32(len(_inds)), self.phis[0].astype(np.float64), 
                                2*np.pi/_nphis*np.ones(_nphis, dtype=np.float64), np.int32(_nphis), )
                args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
                args_mapnap3 = (apradii, np.int32(len(apradii)), MN3correlators)
                args_4pcf = (np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal), 
                            bin_centers, Upsilon_n, N_n, fourpcf, fourpcf_norm, )
                args = (*args_resos,
                        *args_sourcecat,
                        *args_resos_lens,
                        *args_hash_source,
                        *args_mhash_lens,
                        *args_hash,
                        *args_basesetup,
                        *args_indsetup,
                        *args_thetas,
                        np.int32(self.nthreads),
                        *args_mapnap3,
                        *args_4pcf)
                func = self.clib.alloc_notomoMapNap3_tree_gnnn
            else:
                args_indsetup = (np.int32(nthetacombis_tot), _inds, np.int32(len(_inds)), )
                args_4pcf = (bin_centers, Upsilon_n, N_n, )
                args = (*args_resos,
                        *args_sourcecat,
                        *args_resos_lens,
                        *args_hash_source,
                        *args_mhash_lens,
                        *args_hash,
                        *args_basesetup,
                        *args_indsetup,
                        np.int32(self.nthreads), 
                        np.int32(self._verbose_c),
                        *args_4pcf)
                func = self.clib.alloc_notomoGammans_tree_gnnn
        
        if self._verbose_debug:
            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                try:
                    toprint += (func.argtypes[elarg], )
                except:
                    print("Weird error for arg %i."%elarg)
                print(toprint)
                print(arg)
        
        func(*args)

        ## Massage the output ##
        istatout = ()
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.projection = "X"
        self.is_edge_corrected = False
        if "4pcf_multipole" in statistics:
            self.npcf_multipoles = Upsilon_n.reshape(sc)
            self.npcf_multipoles_norm = N_n.reshape(sn)
        if "4pcf_real" in statistics:
            if lowmem:
                self.npcf = fourpcf.reshape(s4pcf)
                self.npcf_norm = fourpcf_norm.reshape(s4pcfn) 
            else:
                if self._verbose_python:
                    print("Transforming output to real space basis")
                self.multipoles2npcf_c()
        if hasintegratedstats:
            if "MN3" in statistics:
                istatout += (MN3correlators.reshape((1,self.nzcombis,len(apradii))), )
            # TODO allocate map4, map4c etc.

        if apply_edge_correction:
            self.edge_correction()
            
        return istatout
     
    # TODO: 
    # * Include the z-weighting method
    # * Include the 2pcf as spline --> Should we also add an option to compute it here? Might be a mess
    #   as then we also would need methods to properly distribute randoms...
    # * Do a voronoi-tesselation at the multipole level? Would be just 2D, but still might help? Eventually
    #   bundle together cells s.t. tot_weight > theshold? However, this might then make the binning courser
    #   for certain triangle configs(?)
    def multipoles2npcf(self):
        raise NotImplementedError
    
    def multipoles2npcf_singlethetcombi(self, elthet1, elthet2, elthet3, projection="X"):
        r""" Converts a 4PCF in the multipole basis in the real space basis for a fixed combination of radial bins.

        Returns:
        --------
        npcf_out: np.ndarray
            4PCF components in the real-space bassi for all angular combinations.
        npcf_norm_out: np.ndarray
            4PCF weighted counts in the real-space bassi for all angular combinations.
        """
        assert((projection in self.proj_dict.keys()) and (projection in self.projections_avail))
        
        _phis1 = self.phis[0].astype(np.float64)
        _phis2 = self.phis[1].astype(np.float64)
        _nphis1 = len(self.phis[0])
        _nphis2 = len(self.phis[1])
        ncfs, nnvals, _, nzcombis, nbinsr, _, _ = np.shape(self.npcf_multipoles)
        
        Upsilon_in = self.npcf_multipoles[...,elthet1,elthet2,elthet3].flatten()
        N_in = self.npcf_multipoles_norm[...,elthet1,elthet2,elthet3].flatten()
        npcf_out = np.zeros(self.n_cfs*nzcombis*_nphis1*_nphis2, dtype=np.complex128)
        npcf_norm_out = np.zeros(nzcombis*_nphis1*_nphis2, dtype=np.complex128)
        
        self.clib.multipoles2npcf_gnnn_singletheta(
            Upsilon_in, N_in, self.nmaxs[0], self.nmaxs[1],
            self.bin_centers_mean[elthet1], self.bin_centers_mean[elthet2], self.bin_centers_mean[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            npcf_out, npcf_norm_out)
        
        return npcf_out.reshape((self.n_cfs, _nphis1,_nphis2)), npcf_norm_out.reshape((_nphis1,_nphis2))
    
    def multipoles2npcf_singletheta_nconvergence(self, elthet1, elthet2, elthet3):
        r""" Checks convergence of the conversion between mutltipole-space and real space for a combination of radial bins.

        Returns:
        --------
        npcf_out: np.ndarray
            Natural 4PCF components in the real-space basis for all angular combinations.
        npcf_norm_out: np.ndarray
            4PCF weighted counts in the real-space basis for all angular combinations.
        """
        
        _phis1 = self.phis[0].astype(np.float64)
        _phis2 = self.phis[1].astype(np.float64)
        _nphis1 = len(self.phis[0])
        _nphis2 = len(self.phis[1])
                
        ncfs, nnvals, _, nzcombis, nbinsr, _, _ = np.shape(self.npcf_multipoles)
        
        Upsilon_in = self.npcf_multipoles[...,elthet1,elthet2,elthet3].flatten()
        N_in = self.npcf_multipoles_norm[...,elthet1,elthet2,elthet3].flatten()
        npcf_out = np.zeros(self.n_cfs*nzcombis*(self.nmaxs[0]+1)*(self.nmaxs[1]+1)*_nphis1*_nphis2, dtype=np.complex128)
        npcf_norm_out = np.zeros(nzcombis*(self.nmaxs[0]+1)*(self.nmaxs[1]+1)*_nphis1*_nphis2, dtype=np.complex128)
        
        self.clib.multipoles2npcf_gnnn_singletheta_nconvergence(
            Upsilon_in, N_in, self.nmaxs[0], self.nmaxs[1],
            self.bin_centers_mean[elthet1], self.bin_centers_mean[elthet2], self.bin_centers_mean[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            npcf_out, npcf_norm_out)
                
        npcf_out = npcf_out.reshape((self.n_cfs, self.nmaxs[0]+1, self.nmaxs[1]+1, _nphis1, _nphis2))
        npcf_norm_out = npcf_norm_out.reshape((self.nmaxs[0]+1, self.nmaxs[1]+1, _nphis1, _nphis2))
                
        return npcf_out, npcf_norm_out
            
            
    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        super()._projectnpcf(self, projection)
        
    ## INTEGRATED MEASURES ## 
    def computeMapNap3(self, radii, nmax_trafo=None, basis='MapMx'):
        r"""Computes the fourth-order aperture statistcs using the polynomial filter of Crittenden 2002."""

        assert(basis in ['MapMx','MM*','both'])
        
        if nmax_trafo is None:
            nmax_trafo=self.nmaxs[0]
            
        # Retrieve all the aperture measures in the MM* basis via the 5D transformation eqns
        MN3correlators = np.zeros(1*len(radii), dtype=np.complex128)
        self.clib.fourpcfmultipoles2MN3correlators(
            np.int32(self.nmaxs[0]), np.int32(nmax_trafo),
            self.bin_edges, self.bin_centers_mean, np.int32(self.nbinsr),
            radii.astype(np.float64), np.int32(len(radii)),
            self.phis[0].astype(np.float64), self.phis[1].astype(np.float64), 
            self.dphis[0].astype(np.float64), self.dphis[1].astype(np.float64), 
            len(self.phis[0]), len(self.phis[1]),
            np.int32(self.proj_dict[self.projection]), np.int32(self.nthreads),
            self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(),
            MN3correlators)
        res_MMStar = MN3correlators.reshape((1,len(radii)))
        
        # Allocate result (here the bases are really equivalent...)
        res = ()
        if basis=='MM*' or basis=='both':
            res += (res_MMStar, )
        if basis=='MapMx' or basis=='both':
            res += ( res_MMStar, )
        
        return res

    def MapNap3_corrections(self, apradii, xi_ng=None, Gtilde_third=None,
                            include_second=True, include_third=True, basis='MapMx'):

        if xi_ng is not None and include_second:
            # Check consistency
            pass
        if xi_ng is None and include_second:
            # Compute gamma_t via treecorr
            pass

        if Gtilde_third is not None and include_third:
            # Check consistency
            pass
        if Gtilde_third is None and include_third:
            # Compute GNN via treecorr
            pass
        if xi_ng is None:
            xi_ng = np.zeros(self.nbinsr, dtype=np.float64)
        if Gtilde_third is None:
            Gtilde_third = np.zeros(self.nbinsr*self.nbinsr*self.nbinsphi,dtype=np.complex128)

        # This block is similar to MapNap3_analytic
        self.nbinsz = 1
        self.nzcombis = 1
        _nphis = len(self.phis[0])
        MN3correlators = np.zeros(self.n_cfs*self.nzcombis*len(apradii)).astype(np.complex128)
        # Define the radial bin batches
        args_4pcfsetup = (self.bin_edges, self.bin_centers_mean, np.int32(self.nbinsr), 
                          self.phis[0].astype(np.float64), 
                          (self.phis[0][1]-self.phis[0][0])*np.ones(_nphis, dtype=np.float64), _nphis, np.int32(self.nmaxs[0]), )
        args_map4 = (apradii.astype(np.float64), np.int32(len(apradii)), )

        args = (*args_4pcfsetup,
                np.int32(self.nthreads),
                *args_map4,
                xi_ng.astype(np.float64),
                Gtilde_third.flatten(),
                np.int32(include_second), 
                np.int32(include_third), 
                MN3correlators)
        func = self.clib.alloc_notomoMapNap3_corrections
        
        if self._verbose_debug:
            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                try:
                    toprint += (func.argtypes[elarg], )
                    print(toprint)
                    print(arg)
                except:
                    print("We did have a problem for arg %i"%elarg)

        func(*args)

        return MN3correlators.reshape((1,len(apradii)))

    def gauss4pcf_analytic(self, itheta1, itheta2, itheta3, nsubr, 
                           xing_arr, xinn_arr, thetamin_xi, thetamax_xi, dtheta_xi):
    
        gauss_4pcf = np.zeros(self.n_cfs*self.nbinsphi[0]*self.nbinsphi[1],dtype=np.complex128)

        self.clib.gtilde4pcf_analytic_integrated(
            np.int32(itheta1), 
            np.int32(itheta2), 
            np.int32(itheta3), 
            np.int32(nsubr), 
            self.bin_edges.astype(np.float64),
            np.int32(self.nbinsr),
            self.phis[0].astype(np.float64),
            np.int32(self.nbinsphi[0]),
            xing_arr.astype(np.float64), 
            xinn_arr.astype(np.float64),
            np.float64(thetamin_xi), 
            np.float64(thetamax_xi), 
            np.float64(dtheta_xi), 
            gauss_4pcf)
        return gauss_4pcf.reshape((self.n_cfs, self.nbinsphi[0], self.nbinsphi[1]))  

    def gnnn_corrections(self, itheta1, itheta2, itheta3, xi_ng=None, Gtilde_third=None,
                         include_second=True, include_third=True):

        if xi_ng is None:
            xi_ng = np.zeros(self.nbinsr, dtype=np.float64)
        if Gtilde_third is None:
            Gtilde_third = np.zeros(self.nbinsr*self.nbinsr*self.nbinsphi,dtype=np.complex128)

        corrs = np.zeros(self.n_cfs*self.nbinsphi[0]*self.nbinsphi[1],dtype=np.complex128)
        self.clib.gtilde4pcf_corrections(
            np.int32(itheta1), 
            np.int32(itheta2), 
            np.int32(itheta3), 
            np.int32(self.nbinsr),
            self.phis[0].astype(np.float64),
            np.int32(self.nbinsphi[0]),
            np.int32(self.nmaxs[0]),
            np.int32(include_second),
            np.int32(include_third),
            xi_ng.astype(np.float64), 
            Gtilde_third.flatten().astype(np.complex128), 
            corrs)

        return corrs.reshape((self.n_cfs, self.nbinsphi[0], self.nbinsphi[1]))  

    # Disconnected part of MapNap^3 from analytic 2pcfs
    # thetamin_xi, thetamax_xi, ntheta_xi is the linspaced array in which the xipm are passed to the external function
    def MapNap3analytic(self, mapradii, xing_spl, xinn_spl, thetamin_xi, thetamax_xi, ntheta_xi, 
                         nsubr=1, nsubsample_filter=1, batchsize=None, basis='MapMx'):
        
        self.nbinsz = 1
        self.nzcombis = 1
        _nmax = self.nmaxs[0]
        _nnvals = (2*_nmax+1)*(2*_nmax+1)
        _nbinsr3 = self.nbinsr*self.nbinsr*self.nbinsr
        _nphis = len(self.phis[0])
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        MN3correlators = np.zeros(self.n_cfs*self.nzcombis*len(mapradii)).astype(np.complex128)
        # Define the radial bin batches
        if batchsize is None:
            batchsize = min(_nbinsr3,min(10000,int(_nbinsr3/self.nthreads)))
            if self._verbose_python:
                print("Using batchsize of %i for radial bins"%batchsize)
        nbatches = np.int32(_nbinsr3/batchsize)
        thetacombis_batches = np.arange(_nbinsr3).astype(np.int32)
        cumnthetacombis_batches = (np.arange(nbatches+1)*_nbinsr3/(nbatches)).astype(np.int32)
        nthetacombis_batches = (cumnthetacombis_batches[1:]-cumnthetacombis_batches[:-1]).astype(np.int32)
        cumnthetacombis_batches[-1] = _nbinsr3
        nthetacombis_batches[-1] = _nbinsr3-cumnthetacombis_batches[-2]
        thetacombis_batches = thetacombis_batches.flatten().astype(np.int32)
        nbatches = len(nthetacombis_batches)

        args_4pcfsetup = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                          self.phis[0].astype(np.float64), 
                          (self.phis[0][1]-self.phis[0][0])*np.ones(_nphis, dtype=np.float64), _nphis, np.int32(nsubr), )
        args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
        args_map4 = (mapradii.astype(np.float64), np.int32(len(mapradii)), )
        thetas_xi = np.linspace(thetamin_xi,thetamax_xi,ntheta_xi+1)
        args_xi = (xing_spl(thetas_xi), xinn_spl(thetas_xi), thetamin_xi, thetamax_xi, ntheta_xi, nsubsample_filter, )
        args = (*args_4pcfsetup,
                *args_thetas,
                np.int32(self.nthreads),
                *args_map4,
                *args_xi,
                MN3correlators)
        func = self.clib.alloc_notomoMapNap3_analytic
        
        if self._verbose_debug:
            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                try:
                    toprint += (func.argtypes[elarg], )
                    print(toprint)
                    print(arg)
                except:
                    print("We did have a problem for arg %i"%elarg)

        func(*args)

        res_MMStar = MN3correlators.reshape((self.n_cfs,len(mapradii)))
        # Allocate result
        res = ()
        res += (res_MMStar, )
    
        return res     