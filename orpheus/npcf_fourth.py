import numpy as np 
import ctypes as ct
from functools import reduce
import operator
from scipy.interpolate import interp1d, RegularGridInterpolator

from .utils import flatlist, gen_thetacombis_fourthorder, gen_n2n3indices_Upsfourth
from .npcf_base import BinnedNPCF
from .npcf_second import GGCorrelation
from .multires_structs import (build_flat_catalog_struct, build_flat_navhash_struct,
                               build_catalog_struct, build_navhash_struct,
                               build_tree_params_struct, build_binning_struct,
                               build_gnnn_output, build_fourth_params, build_clustcorr,
                               build_gggg_output, build_nnnn_output,
                               build_spherical_central_catalog_struct)

__all__ = ["NNNNCorrelation_NoTomo", "GGGGCorrelation_NoTomo", "GNNNCorrelation_NoTomo"]

class NNNNCorrelation_NoTomo(BinnedNPCF):
    r""" Class containing methods to measure and obtain statistics that are built
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
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.

    """
    
    def __init__(self, min_sep, max_sep, verbose=False, thetabatchsize_max=10000, method="Tree",
                 process_spherical=False, **kwargs):
        super().__init__(order=4, spins=np.array([0,0,0,0], dtype=np.int32),
                         n_cfs=1, min_sep=min_sep, max_sep=max_sep,
                         method=method, methods_avail=["Tree", "DoubleTree"], **kwargs)

        self.thetabatchsize_max = thetabatchsize_max
        self.nbinsz = 1
        self.nzcombis = 1
        self.process_spherical = bool(process_spherical)

    def saveinst(self, path_save, fname, extr_pars=None):
        extras = dict(nbinsz=self.nbinsz, nzcombis=self.nzcombis,
                      thetabatchsize_max=self.thetabatchsize_max)
        if extr_pars: extras.update(extr_pars)
        super().saveinst(path_save, fname, extr_pars=extras)

    def process(self, cat, statistics="all", tofile=False, apply_edge_correction=False,
                lowmem=True, mapradii=None, batchsize=None, custom_thetacombis=None, cutlen=2**31-1,
                memory_bound=512.):
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
        * - Raise error if lowmem=False and we will have more than 2^31-1 elements at any stage of the computation

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
                    print("Warning: Lowmem computation recommended for integrated measures of the 4pcf. " +
                          "Set `lowmem` from `%s` to `True`"%str(lowmem))
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
        use_spherical = self.process_spherical and getattr(cat, 'geometry', 'flat2d') == 'spherical'
        if self.process_spherical and not use_spherical:
            raise ValueError("process_spherical=True requires a spherical catalog "
                             "(cat.geometry=='spherical').")
        # Init default args
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        if not use_spherical:
            if not cat.hasspatialhash:
                cat.build_spatialhash(dpix=max(1.,self.max_sep//10.))
            nregions = np.int32(len(np.argwhere(cat.index_matcher>-1).flatten()))
            args_hash = (cat.index_matcher, cat.pixs_galind_bounds, cat.pix_gals, nregions,
                         np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n),
                         np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
        args_basecat = (cat.isinner.astype(np.float64), cat.weight, cat.pos1, cat.pos2,
                        np.int32(cat.ngal), )
        
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
        # Zero radii tell the C kernel to skip the aperture integration and the npcf conversion
        _nmapradii = len(mapradii) if hasintegratedstats else 0

        
        # Build structs
        if use_spherical:
            from healpy import nside2resol
            if self.method == "DoubleTree":
                raise NotImplementedError(
                    "Curved-sky (process_spherical=True) is implemented for method='Tree' "
                    "only; the DoubleTree spherical variant is validated against the Tree "
                    "oracle and not yet available.")
            only_multipoles = ("4pcf_multipole" in statistics and
                               "4pcf_real" not in statistics and not hasintegratedstats)
            if not only_multipoles:
                raise NotImplementedError(
                    "Curved-sky NNNN currently supports the multipoles-only path; request "
                    "statistics='4pcf_multipole' (no 4pcf_real / aperture statistics).")
            # Get theta- and multipole-index masks
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads,
                                                     batchsize=batchsize, batchsize_max=self.thetabatchsize_max,
                                                     ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python)
            _, _, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            assert(self.nmaxs[0]==self.nmaxs[1])
            _shape, _inds, _n2s, _n3s = gen_n2n3indices_Upsfourth(self.nmaxs[0])
            # Healpix nside per radial band: band cell size tree_resos[r] -> smallest
            # nside whose pixel is no larger; tree_resos[r]==0 marks the discrete band.
            _deg2rad = np.pi/180.
            def _nside_for(target_rad):
                ns = 1
                while nside2resol(ns) > target_rad and ns < 2**29:
                    ns *= 2
                return ns
            nsides = [0 if self.tree_resos[r]==0. else _nside_for(self.tree_resos[r]*_deg2rad)
                      for r in range(self.tree_nresos)]
            nside_hash = _nside_for(max(self.min_sep, 0.5*self.tree_redges[1])*_deg2rad)
            sph = cat.multihash_bundle(reso_redges=self.tree_redges, nsides=nsides,
                                       nside_hash=nside_hash,
                                       verbose=self._verbose_python)
            assert sph['geometry'] == 'spherical'
            catc_s, keep_cc = build_spherical_central_catalog_struct(
                sph['cen_isinner'], sph['cen_w'], sph['cen_vx'], sph['cen_vy'], sph['cen_vz'],
                sph['cen_ra'], sph['cen_sindec'], sph['cen_cosdec'], self.nbinsz)
            catr_s, keep_cr = build_catalog_struct(sph, self.nbinsz)
            catr_s.nresos = int(self.tree_nresos)
            nav_s, keep_n = build_navhash_struct(sph)
            tree_s, keep_t = build_tree_params_struct(self, sph)
            bin_s = build_binning_struct(self, scale=_deg2rad, nmax=int(_nmax), dccorr=int(self.multicountcorr))
            fourth_s, keep_f = build_fourth_params(
                nindices=_inds, len_nindices=len(_inds),
                thetacombis_batches=thetacombis_batches, nthetacombis_batches=nthetacombis_batches,
                cumthetacombis_batches=cumnthetacombis_batches, nthetbatches=nbatches)
            out_s = build_nnnn_output(bin_centers, N_n)
            _alive = keep_cc + keep_cr + keep_n + keep_t + keep_f   # noqa: F841
            self.clib.alloc_nnnn_tree_spherical(
                ct.byref(catc_s), ct.byref(catr_s), ct.byref(nav_s), ct.byref(tree_s),
                ct.byref(bin_s), ct.byref(fourth_s),
                np.float64(memory_bound), np.int32(self.nthreads),
                np.int32(self._verbose_c+self._verbose_debug), ct.byref(out_s))
        elif self.method=="Discrete" and not lowmem:
            raise NotImplementedError
        elif self.method=="Discrete" and lowmem:
            raise NotImplementedError
        elif self.method in ("Tree", "DoubleTree"):
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
            mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                      shuffle=self.shuffle_pix, normed=False, nthreads=self.nthreads)
            _zb = np.zeros(cat.ngal, dtype=np.int32)   # notomo: zbins unused by C
            catc_s, keep_cc = build_flat_catalog_struct(cat.pos1, cat.pos2, cat.weight, _zb,
                                                        self.nbinsz, cat.isinner)
            catr_s, keep_cr = build_catalog_struct(mh, self.nbinsz)
            catr_s.nresos = int(self.tree_nresos)
            nav_s, keep_n = build_navhash_struct(mh, cat_obj=cat)
            tree_s, keep_t = build_tree_params_struct(self, mh)
            bin_s = build_binning_struct(self, nmax=int(_nmax), dccorr=int(self.multicountcorr))
            _alive = keep_cc + keep_cr + keep_n + keep_t   # noqa: F841
            only_multipoles = ("4pcf_multipole" in statistics and
                               "4pcf_real" not in statistics and
                               not hasintegratedstats)
            if self.method=="Tree" and only_multipoles and lowmem:
                # Multipoles-only fast path: stops after the multipole reconstruction
                # (no real-space transform, no Map^4 integral)
                fourth_s, keep_f = build_fourth_params(
                    nindices=_inds, len_nindices=len(_inds),
                    thetacombis_batches=thetacombis_batches, nthetacombis_batches=nthetacombis_batches,
                    cumthetacombis_batches=cumnthetacombis_batches, nthetbatches=nbatches)
                out_s = build_nnnn_output(bin_centers, N_n)
                self.clib.alloc_nnnn_tree(
                    ct.byref(catc_s), ct.byref(catr_s), ct.byref(nav_s), ct.byref(tree_s),
                    ct.byref(bin_s), ct.byref(fourth_s),
                    np.float64(memory_bound), np.int32(self.nthreads),
                    np.int32(self._verbose_c+self._verbose_debug), ct.byref(out_s))
            elif self.method=="DoubleTree" and only_multipoles and lowmem:
                # True double tree (central-vertex gridding), multipoles only.
                tree_s.nresos_grid = int(self.tree_nresos - cutfirst)
                fourth_s, keep_f = build_fourth_params(
                    nindices=_inds, len_nindices=len(_inds),
                    thetacombis_batches=thetacombis_batches, nthetacombis_batches=nthetacombis_batches,
                    cumthetacombis_batches=cumnthetacombis_batches, nthetbatches=nbatches)
                out_s = build_nnnn_output(bin_centers, N_n)
                self.clib.alloc_nnnn_doubletree(
                    ct.byref(catr_s), ct.byref(nav_s), ct.byref(tree_s),
                    ct.byref(bin_s), ct.byref(fourth_s),
                    np.float64(memory_bound), np.int32(self.nthreads),
                    np.int32(self._verbose_c+self._verbose_debug), ct.byref(out_s))
            else:
                # Aperture/real-space paths (Nap4): partial struct port, aperture radii and
                # output arrays stay loose. Tree and DoubleTree share the same signature.
                fourth_s, keep_f = build_fourth_params(
                    nindices=_inds, len_nindices=len(_inds),
                    phibins1=self.phis[0], dbinsphi1=2*np.pi/_nphis*np.ones(_nphis), nbinsphi1=_nphis,
                    thetacombis_batches=thetacombis_batches, nthetacombis_batches=nthetacombis_batches,
                    cumthetacombis_batches=cumnthetacombis_batches, nthetbatches=nbatches)
                _nap4 = self.clib.alloc_notomoNap4_tree_nnnn_highmem if not lowmem \
                    else self.clib.alloc_notomoNap4_tree_nnnn
                _nap4(
                    ct.byref(catc_s), ct.byref(catr_s), ct.byref(nav_s), ct.byref(tree_s),
                    ct.byref(bin_s), ct.byref(fourth_s),
                    mapradii, np.int32(_nmapradii), N4correlators,
                    np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal),
                    np.int32(self.nthreads), np.int32(self._verbose_c+self._verbose_debug),
                    bin_centers, N_n, fourpcf)

        ## Massage the output ##
        istatout = ()
        if use_spherical:
            # The curved-sky C path works in radians; convert bin centers back in
            # the catalogue's angular unit (degrees for spherical catalogs).
            bin_centers = bin_centers * (180./np.pi)
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
        r""" Converts a 4PCF from the multipole basis to the real-space basis for a fixed combination of radial bins.

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
    
    def multipoles2npcf(self):
        r""" Converts a 4PCF from the multipole basis to the real-space basis for all radial bins.

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
        
        N_in = self.npcf_multipoles.flatten()
        npcf_out = np.zeros(self.nbinsr*self.nbinsr*self.nbinsr*_nphis1*_nphis2, dtype=np.complex128)
        bin_s = build_binning_struct(self, nmax=int(self.nmaxs[0]), dccorr=int(self.multicountcorr))
        fourth_s, _keep_f = build_fourth_params(phibins1=_phis1, phibins2=_phis2,
                                                nbinsphi1=_nphis1, nbinsphi2=_nphis2)
        self.clib.multipoles2npcf_nnnn(
            N_in.astype(np.complex128, copy=False),
            ct.byref(bin_s), ct.byref(fourth_s),
            self.bin_centers_mean.astype(np.float64, copy=False),
            npcf_out, np.int32(self.nthreads))
        
        self.npcf = npcf_out.reshape((self.nbinsr, self.nbinsr, self.nbinsr, 1, _nphis1,_nphis2))

class GGGGCorrelation_NoTomo(BinnedNPCF):
    r""" Class containing methods to measure and obtain statistics that are built
    from nontomographic fourth-order shear correlation functions.

    Note that the different components of the GGGG correlator are ordered as

    .. math::

        \Upsilon_\mu \sim \left[
        \langle \gamma \gamma \gamma \gamma \rangle,\,
        \langle \gamma^* \gamma \gamma \gamma \rangle,\,
        \langle \gamma \gamma^* \gamma \gamma \rangle,\,
        \langle \gamma \gamma \gamma^* \gamma \rangle,\,
        \langle \gamma \gamma \gamma \gamma^* \rangle,\,
        \langle \gamma^* \gamma^* \gamma \gamma \rangle,\,
        \langle \gamma^* \gamma \gamma^* \gamma \rangle,\,
        \langle \gamma^* \gamma \gamma \gamma^* \rangle
        \right] \ ,

    following the same rule as :class:`GGGCorrelation`, i.e. starting with the
    correlator in which no polar field is conjugated and then moving the
    conjugations from left to right.

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
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.

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

    def saveinst(self, path_save, fname, extr_pars=None):
        extras = dict(nbinsz=self.nbinsz, nzcombis=self.nzcombis,
                      thetabatchsize_max=self.thetabatchsize_max)
        if extr_pars: extras.update(extr_pars)
        super().saveinst(path_save, fname, extr_pars=extras)

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
        * - Raise error if lowmem=False and we will have more than 2^31-1 elements at any stage of the computation

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
            M4correlators = __fillflag*np.ones(__lenflag).astype(np.complex128)
        # Zero radii tell the C kernel to skip the aperture integration and the npcf conversion
        _nmapradii = len(mapradii) if hasintegratedstats else 0

        # Build structs
        _zb = np.zeros(cat.ngal, dtype=np.int32)
        if self.method=="Discrete" and not lowmem:
            cat_s, keep_c = build_flat_catalog_struct(cat.pos1, cat.pos2, cat.weight, _zb,
                                                      self.nbinsz, cat.isinner,
                                                      e1=cat.tracer_1, e2=cat.tracer_2)
            nav_s, keep_n = build_flat_navhash_struct(cat)
            bin_s = build_binning_struct(self, nmax=int(_nmax), dccorr=int(self.multicountcorr),
                                         rbins=np.array([-1.]))
            out_s = build_gggg_output(bin_centers, Upsilon_n, N_n)
            _alive = keep_c + keep_n   # noqa: F841
            self.clib.alloc_notomoGammans_discrete_gggg(
                ct.byref(cat_s), ct.byref(nav_s), ct.byref(bin_s), None,
                np.int32(self.nthreads), np.int32(self._verbose_c+self._verbose_debug), ct.byref(out_s))
        if self.method=="Discrete" and lowmem:
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads, batchsize=batchsize,
                                                     batchsize_max=self.thetabatchsize_max, ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python)
            _, _, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            cat_s, keep_c = build_flat_catalog_struct(cat.pos1, cat.pos2, cat.weight, _zb,
                                                      self.nbinsz, cat.isinner,
                                                      e1=cat.tracer_1, e2=cat.tracer_2)
            nav_s, keep_n = build_flat_navhash_struct(cat)
            bin_s = build_binning_struct(self, nmax=int(_nmax), dccorr=int(self.multicountcorr))
            fourth_s, keep_f = build_fourth_params(
                phibins1=self.phis[0], dbinsphi1=2*np.pi/_nphis*np.ones(_nphis), nbinsphi1=_nphis,
                thetacombis_batches=thetacombis_batches, nthetacombis_batches=nthetacombis_batches,
                cumthetacombis_batches=cumnthetacombis_batches, nthetbatches=nbatches)
            _alive = keep_c + keep_n + keep_f   # noqa: F841
            self.clib.alloc_notomoMap4_disc_gggg(
                ct.byref(cat_s), ct.byref(nav_s), ct.byref(bin_s), ct.byref(fourth_s),
                i_projection, mapradii, np.int32(_nmapradii), M4correlators,
                np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal),
                np.int32(self.nthreads), np.int32(self._verbose_c+self._verbose_debug),
                bin_centers, Upsilon_n, N_n, fourpcf, fourpcf_norm)
        if self.method=="Tree":
            _resradial = gen_thetacombis_fourthorder(nbinsr=self.nbinsr, nthreads=self.nthreads, batchsize=batchsize,
                                                     batchsize_max=self.thetabatchsize_max, ordered=True, custom=custom_thetacombis,
                                                     verbose=self._verbose_python*lowmem)
            _, _, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            assert(self.nmaxs[0]==self.nmaxs[1])
            _shape, _inds, _n2s, _n3s = gen_n2n3indices_Upsfourth(self.nmaxs[0])
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                      shuffle=self.shuffle_pix, w2field=True, normed=True, nthreads=self.nthreads)
            allfields = mh['allfields']
            e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
            e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
            catc_s, keep_cc = build_flat_catalog_struct(cat.pos1, cat.pos2, cat.weight, _zb,
                                                        self.nbinsz, cat.isinner,
                                                        e1=cat.tracer_1, e2=cat.tracer_2)
            catr_s, keep_cr = build_catalog_struct(mh, self.nbinsz, extra={'e1_resos': e1_resos, 'e2_resos': e2_resos})
            catr_s.nresos = int(self.tree_nresos)
            nav_s, keep_n = build_navhash_struct(mh, cat_obj=cat)
            tree_s, keep_t = build_tree_params_struct(self, mh)
            bin_s = build_binning_struct(self, nmax=int(_nmax), dccorr=int(self.multicountcorr))
            _alive = keep_cc + keep_cr + keep_n + keep_t   # noqa: F841
            if not lowmem:
                fourth_s, keep_f = build_fourth_params(nindices=_inds, len_nindices=len(_inds),
                                                       nthetacombis=int(cumnthetacombis_batches[-1]))
                out_s = build_gggg_output(bin_centers, Upsilon_n, N_n)
                self.clib.alloc_notomoGammans_tree_gggg(
                    ct.byref(catc_s), ct.byref(catr_s), ct.byref(nav_s), ct.byref(tree_s),
                    ct.byref(bin_s), ct.byref(fourth_s),
                    np.int32(self.nthreads), np.int32(self._verbose_c+self._verbose_debug), ct.byref(out_s))
            if lowmem:
                fourth_s, keep_f = build_fourth_params(
                    nindices=_inds, len_nindices=len(_inds),
                    phibins1=self.phis[0], dbinsphi1=2*np.pi/_nphis*np.ones(_nphis), nbinsphi1=_nphis,
                    thetacombis_batches=thetacombis_batches, nthetacombis_batches=nthetacombis_batches,
                    cumthetacombis_batches=cumnthetacombis_batches, nthetbatches=nbatches)
                self.clib.alloc_notomoMap4_tree_gggg(
                    ct.byref(catc_s), ct.byref(catr_s), ct.byref(nav_s), ct.byref(tree_s),
                    ct.byref(bin_s), ct.byref(fourth_s),
                    i_projection, mapradii, np.int32(_nmapradii), M4correlators,
                    np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal),
                    np.int32(self.nthreads), np.int32(self._verbose_c+self._verbose_debug),
                    bin_centers, Upsilon_n, N_n, fourpcf, fourpcf_norm)
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
        bin_s = build_binning_struct(self, nmax=int(self.nmaxs[0]), dccorr=int(self.multicountcorr))
        fourth_s, _keep_f = build_fourth_params(phibins1=_phis1, phibins2=_phis2,
                                                nbinsphi1=_nphis1, nbinsphi2=_nphis2)
        self.clib.multipoles2npcf_gggg(self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(),
                                       self.bin_centers_mean.astype(np.float64),
                                       ct.byref(bin_s), ct.byref(fourth_s),
                                       np.int32(self.proj_dict[projection]), np.int32(self.n_cfs),
                                       np.int32(self.nthreads), self.npcf, self.npcf_norm)
        self.npcf = self.npcf.reshape(shape_npcf)
        self.npcf_norm = self.npcf_norm.reshape(shape_npcf_norm)
        self.projection = projection
        
        
    def multipoles2npcf_singlethetcombi(self, elthet1, elthet2, elthet3, projection="X"):
        r""" Converts a 4PCF from the multipole basis to the real-space basis for a fixed combination of radial bins.

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
        npcf_out = np.zeros(self.n_cfs*nzcombis*_nphis1*_nphis2, dtype=np.complex128)
        npcf_norm_out = np.zeros(nzcombis*_nphis1*_nphis2, dtype=np.complex128)
        
        self.clib.multipoles2npcf_gggg_singletheta(
            Upsilon_in, N_in, self.nmaxs[0], self.nmaxs[1],
            self.bin_centers_mean[elthet1], self.bin_centers_mean[elthet2], self.bin_centers_mean[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            np.int32(self.proj_dict[projection]), npcf_out, npcf_norm_out)
        
        return npcf_out.reshape((self.n_cfs, _nphis1,_nphis2)), npcf_norm_out.reshape((_nphis1,_nphis2))
                
    def multipoles2npcf_gggg_singletheta_nconvergence(self, elthet1, elthet2, elthet3, projection="X"):
        r""" Checks convergence of the conversion between multipole-space and real space for a combination of radial bins.

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
        r"""Computes the fourth-order aperture mass statistics using the polynomial filter of Crittenden 2002."""

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
        from fourth-order source-lens-lens-lens (G4L) correlation functions.
        
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
        Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.
        """
        super().__init__(4, [2,0,0,0], n_cfs=1, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X"]
        self.proj_dict = {"X":0}
        self.nbinsz_source = 1
        self.nbinsz_lens = 1
        self.nzcombis = 1
        self.thetabatchsize_max = thetabatchsize_max

        # (Add here any newly implemented projections)
        self._initprojections(self)

    def saveinst(self, path_save, fname, extr_pars=None):
        extras = dict(nbinsz_source=self.nbinsz_source, nbinsz_lens=self.nbinsz_lens,
                      nzcombis=self.nzcombis, thetabatchsize_max=self.thetabatchsize_max)
        if extr_pars: extras.update(extr_pars)
        super().saveinst(path_save, fname, extr_pars=extras)

    def process(self, cat_source, cat_lens, statistics="all", tofile=False, apply_edge_correction=False,
                dotomo_source=True, dotomo_lens=True,
                lowmem=None, apradii=None, xi=None, nnn=None, count_floor=0.1,
                batchsize=None, custom_thetacombis=None, cutlen=2**31-1):
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
        # Zero radii tell the C kernel to skip the aperture integration and the npcf conversion
        _napradii = len(apradii) if hasintegratedstats else 0
        
        # Basic prep
        hash_dpix = max(1.,self.max_sep//10.)
        jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
        cat_source.build_spatialhash(dpix=hash_dpix, extent=jointextent)
        cat_lens.build_spatialhash(dpix=hash_dpix, extent=jointextent)

        _zbz_source = np.zeros(cat_source.ngal, dtype=np.int32)   # notomo: zbins unused by C
        _zbz_lens = np.zeros(cat_lens.ngal, dtype=np.int32)
        out_s = build_gnnn_output(bin_centers, Upsilon_n, N_n)

        if self.method=="Discrete" and not lowmem:
            cats_s, keep_cs = build_flat_catalog_struct(
                cat_source.pos1, cat_source.pos2, cat_source.weight, _zbz_source,
                self.nbinsz_source, cat_source.isinner,
                e1=cat_source.tracer_1, e2=cat_source.tracer_2)
            navs_s, keep_ns = build_flat_navhash_struct(cat_source)
            catl_s, keep_cl = build_flat_catalog_struct(
                cat_lens.pos1, cat_lens.pos2, cat_lens.weight, _zbz_lens,
                self.nbinsz_lens, cat_lens.isinner)
            navl_s, keep_nl = build_flat_navhash_struct(cat_lens)
            bin_s = build_binning_struct(self, nmax=int(self.nmax), dccorr=int(self.multicountcorr))
            _alive = keep_cs + keep_ns + keep_cl + keep_nl   # noqa: F841
            self.clib.alloc_notomoGammans_discrete_gnnn(
                ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                ct.byref(bin_s), None,
                int(self.nthreads), int(self._verbose_c+self._verbose_debug), ct.byref(out_s))

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
            mhl = cat_lens.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                            shuffle=self.shuffle_pix, normed=True, nthreads=self.nthreads)
            cats_s, keep_cs = build_flat_catalog_struct(
                cat_source.pos1, cat_source.pos2, cat_source.weight, _zbz_source,
                self.nbinsz_source, cat_source.isinner,
                e1=cat_source.tracer_1, e2=cat_source.tracer_2)
            navs_s, keep_ns = build_flat_navhash_struct(cat_source)
            catl_s, keep_cl = build_catalog_struct(mhl, self.nbinsz_lens)
            catl_s.nresos = int(self.tree_nresos)
            navl_s, keep_nl = build_navhash_struct(mhl, cat_obj=cat_lens)
            tree_s, keep_tree = build_tree_params_struct(self, mhl)
            bin_s = build_binning_struct(self, nmax=int(self.nmax), dccorr=int(self.multicountcorr))
            _alive = keep_cs + keep_ns + keep_cl + keep_nl + keep_tree   # noqa: F841
            if lowmem:
                fourth_s, keep_f = build_fourth_params(
                    nindices=_inds, len_nindices=len(_inds),
                    phibins1=self.phis[0], dbinsphi1=2*np.pi/_nphis*np.ones(_nphis), nbinsphi1=_nphis,
                    thetacombis_batches=thetacombis_batches, nthetacombis_batches=nthetacombis_batches,
                    cumthetacombis_batches=cumnthetacombis_batches, nthetbatches=nbatches)
                cc_s, keep_cc = build_clustcorr(self, xi, nnn, count_floor)
                _alive2 = keep_f + keep_cc   # noqa: F841
                self.clib.alloc_notomoMapNap3_tree_gnnn(
                    ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                    ct.byref(tree_s), ct.byref(bin_s), ct.byref(fourth_s), ct.byref(cc_s),
                    apradii, np.int32(_napradii),
                    np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal),
                    np.int32(self.nthreads), np.int32(self._verbose_c+self._verbose_debug),
                    bin_centers, Upsilon_n, N_n, fourpcf, fourpcf_norm, MN3correlators)
            else:
                fourth_s, keep_f = build_fourth_params(
                    nindices=_inds, len_nindices=len(_inds), nthetacombis=nthetacombis_tot)
                _alive2 = keep_f   # noqa: F841
                self.clib.alloc_notomoGammans_tree_gnnn(
                    ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                    ct.byref(tree_s), ct.byref(bin_s), ct.byref(fourth_s),
                    np.int32(self.nthreads), np.int32(self._verbose_c), ct.byref(out_s))

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
                self.multipoles2npcf(xi=xi, nnn=nnn, count_floor=count_floor)
        if hasintegratedstats:
            if "MN3" in statistics:
                istatout += (MN3correlators.reshape((1,self.nzcombis,len(apradii))), )
            # TODO allocate mapnap3, mapnap3c etc.

        if apply_edge_correction:
            self.edge_correction()
            
        return istatout
     
    # TODO: 
    # * Same inclusion of z-weighting etc as for g3l?
    def multipoles2npcf(self, xi=None, nnn=None, count_floor=0.1):
        r"""Converts the GNNN 4PCF from the multipole basis to the real-space basis for
        every combination of radial bins (shape ``(n_cfs, nzcombis, nbinsr, nbinsr, nbinsr, nphi, nphi)``).

        Parameters
        ----------
        xi: tuple, optional
            Angular clustering 2PCF of the lenses as ``(thetas, omega)``. If set, enables the
            second-order clustering correction, see :meth:`apply_clustering_correction`.
        nnn: orpheus.NNNCorrelation or tuple, optional
            Connected lens 3PCF for the :math:`\zeta` term. If set. enables the 
            third-order clustering correction, see :meth:`apply_clustering_correction`.
        count_floor: float
            Threshold on the reconstructed triplet counts below which the 4PCF is zeroed.

        Notes
        -----
        For an accurate computation of the connected 4PCF both, second-and third-order corrections are required.
        In particular, the second-order corrections dominate the signal, but also the third-order corrections 
        contribute about 10%-30% for realistic clustering strength of the lenses.
        """
        projection = "X"
        assert((projection in self.proj_dict.keys()) and (projection in self.projections_avail))
        _nphis1 = len(self.phis[0])
        _nphis2 = len(self.phis[1])
        _nelem = self.nbinsr*self.nbinsr*self.nbinsr*_nphis1*_nphis2
        if _nelem > 2e9:
            raise ValueError("Real-space 4PCF too large (%.2f x 10^9 elements); "
                             "use computeMapNap3 for the aperture statistics instead."%(_nelem/1e9))

        npcf_out = np.zeros(self.n_cfs*self.nzcombis*_nelem, dtype=np.complex128)
        npcf_norm_out = np.zeros(self.nzcombis*_nelem, dtype=np.complex128)
        bin_s = build_binning_struct(self, nmax=int(self.nmaxs[0]), dccorr=int(self.multicountcorr),
                                     rbins=self.bin_edges)
        fourth_s, _keep_f = build_fourth_params(phibins1=self.phis[0], phibins2=self.phis[1],
                                                nbinsphi1=_nphis1, nbinsphi2=_nphis2)
        cc_s, _keep_cc = build_clustcorr(self, xi, nnn, count_floor)
        self.clib.multipoles2npcf_gnnn(
            self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(),
            ct.byref(bin_s), ct.byref(fourth_s), ct.byref(cc_s),
            np.int32(self.nthreads), npcf_out, npcf_norm_out)

        self.npcf = npcf_out.reshape((self.n_cfs, self.nzcombis, self.nbinsr, self.nbinsr, self.nbinsr, _nphis1, _nphis2))
        self.npcf_norm = npcf_norm_out.reshape((self.nzcombis, self.nbinsr, self.nbinsr, self.nbinsr, _nphis1, _nphis2))
        self.projection = projection
        return self.npcf, self.npcf_norm

    def multipoles2npcf_singlethetcombi(self, elthet1, elthet2, elthet3, xi=None, nnn=None, count_floor=0.1):
        r""" Converts a 4PCF in the multipole basis in the real space basis for a fixed combination of radial bins.

        Parameters
        ----------
        elthet1, elthet2, elthet3: int
            The radial bin indices for which the 4PCF is evaluated.
        xi: tuple, optional
            Angular clustering 2PCF of the lenses as ``(thetas, omega)``. If set, enables the
            second-order clustering correction, see :meth:`apply_clustering_correction`.
        nnn: orpheus.NNNCorrelation or tuple, optional
            Connected lens 3PCF for the :math:`\zeta` term. If set. enables the 
            third-order clustering correction, see :meth:`apply_clustering_correction`.
        count_floor: float
            Threshold on the reconstructed triplet counts below which the 4PCF is zeroed.

        Notes
        -----
        For an accurate computation of the connected 4PCF both, second-and third-order corrections are required.
        In particular, the second-order corrections dominate the signal, but also the third-order corrections 
        contribute about 10%-30% for realistic clustering strength of the lenses.

        Returns:
        --------
        npcf_out: np.ndarray
            4PCF components in the real-space bassi for all angular combinations.
        npcf_norm_out: np.ndarray
            4PCF weighted counts in the real-space bassi for all angular combinations.
        """
        projection = "X"
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

        # Correction thetas: geometric bin centers, identical across all conversion routes
        _rc = np.sqrt(self.bin_edges[:-1]*self.bin_edges[1:])
        cc_s, _keep_cc = build_clustcorr(self, xi, nnn, count_floor)
        self.clib.multipoles2npcf_gnnn_singletheta(
            Upsilon_in, N_in, self.nmaxs[0], self.nmaxs[1],
            _rc[elthet1], _rc[elthet2], _rc[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            ct.byref(cc_s),
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
        
        _rc = np.sqrt(self.bin_edges[:-1]*self.bin_edges[1:])
        cc_s, _keep_cc = build_clustcorr(self, None, None, 0.1)
        self.clib.multipoles2npcf_gnnn_singletheta_nconvergence(
            Upsilon_in, N_in, self.nmaxs[0], self.nmaxs[1],
            _rc[elthet1], _rc[elthet2], _rc[elthet3],
            _phis1, _phis2, _nphis1, _nphis2,
            ct.byref(cc_s),
            npcf_out, npcf_norm_out)
                
        npcf_out = npcf_out.reshape((self.n_cfs, self.nmaxs[0]+1, self.nmaxs[1]+1, _nphis1, _nphis2))
        npcf_norm_out = npcf_norm_out.reshape((self.nmaxs[0]+1, self.nmaxs[1]+1, _nphis1, _nphis2))
                
        return npcf_out, npcf_norm_out
            
            
    ## CLUSTERING CORRECTION ##
    def _clustcorr_args(self, xi=None, nnn=None):
        r"""Builds argument block for the clustering correction of the fourth-order 
        correction of the raw G4L correlator.

        Parameters
        ----------
        xi: tuple or None
            Angular clustering 2PCF of the lenses as ``(thetas, omega)``.
        nnn: orpheus.NNNCorrelation or tuple or None
            Connected angular clustering 3PCF of the lenses, either as an
            ``NNNCorrelation`` instance or as an explicit tuple ``(r_centers, phis, zeta)``
            with ``zeta`` of shape ``(nr, nr, nphi)``
        """
        # If xi is set, we resample it to finer grid that will be used for interpolation in C
        if xi is None:
            args_xi = (np.zeros(2, dtype=np.float64), np.float64(0.), np.float64(1.), np.float64(1.), np.int32(0), )
        else:
            _nfine = 4096
            _ts = np.asarray(xi[0], dtype=np.float64)
            _om = np.asarray(xi[1], dtype=np.float64)
            _tsfine = np.linspace(_ts[0], _ts[-1], _nfine+1)
            args_xi = (np.interp(_tsfine, _ts, _om), np.float64(_ts[0]), np.float64(_ts[-1]),
                       np.float64((_ts[-1]-_ts[0])/_nfine), np.int32(1), )
        # If zeta is set we just use what is given.
        if nnn is None:
            args_zeta = (np.zeros(8, dtype=np.float64), np.zeros(2, dtype=np.float64), np.int32(2),
                         np.zeros(2, dtype=np.float64), np.int32(2), np.int32(0), )
        else:
            _rs, _phis, _zeta = (nnn.bin_centers_mean, nnn.phi, nnn.zeta[0]) if hasattr(nnn, "zeta") else nnn
            args_zeta = (np.ascontiguousarray(_zeta, dtype=np.float64).flatten(),
                         np.asarray(_rs, dtype=np.float64), np.int32(len(_rs)),
                         np.asarray(_phis, dtype=np.float64), np.int32(len(_phis)), np.int32(1), )
        return args_xi + args_zeta

    def apply_clustering_correction(self, xi=None, nnn=None):
        r"""Multiplies the raw real-space 4PCF by the clustering correction to unbias the G4L estimator.

        Parameters
        ----------
        xi: tuple or None
            Angular clustering 2PCF of the lenses as ``(thetas, omega)``.
        nnn: orpheus.NNNCorrelation or tuple or None
            Connected clustering 3PCF, of the lenses, either as an 
            ``NNNCorrelation`` instance or as an explicit tuple ``(r_centers, phis, zeta)``
            with ``zeta`` of shape ``(nr, nr, nphi)``

        Notes
        -----
        - We correct by :math:`C = 1 + \omega(d_{12}) + \omega(d_{13}) + \omega(d_{23}) + \zeta(d_{12},d_{13},d_{23})`.
          This is the fourth-order generalization of the :math:`(1+\omega)` correction introduced in 
          Simon+13. The :math:`d_{ij}` are the lens-lens separations.
        - Outside the provided ranges, the NPCFs are set to zero.
        """
        if xi is not None:
            _ts = np.asarray(xi[0], dtype=np.float64)
            _om = np.asarray(xi[1], dtype=np.float64)
        if nnn is not None:
            _rs, _phis_nnn, _zeta = (nnn.bin_centers_mean, nnn.phi, nnn.zeta[0]) if hasattr(nnn, "zeta") else nnn
            _interp3 = RegularGridInterpolator((_rs, _rs, _phis_nnn), _zeta,
                                               bounds_error=False, fill_value=0.)
        phi12 = self.phis[0][:, None]
        phi13 = self.phis[1][None, :]
        cos12 = np.cos(phi12)
        cos13 = np.cos(phi13)
        cos23 = np.cos(phi13-phi12)
        rc = np.sqrt(self.bin_edges[:-1]*self.bin_edges[1:])  # geometric centers, as in the C routes
        for i1 in range(self.nbinsr):
            for i2 in range(self.nbinsr):
                for i3 in range(self.nbinsr):
                    t1, t2, t3 = rc[i1], rc[i2], rc[i3]
                    d12 = np.sqrt(np.clip(t1*t1 + t2*t2 - 2*t1*t2*cos12, 0., None))
                    d13 = np.sqrt(np.clip(t1*t1 + t3*t3 - 2*t1*t3*cos13, 0., None))
                    d23 = np.sqrt(np.clip(t2*t2 + t3*t3 - 2*t2*t3*cos23, 0., None))
                    corr = np.ones_like(d23)
                    if xi is not None:
                        # d <= ts[0] gives 0, matching the C-level linint edge convention
                        for d in (d12, d13, d23):
                            corr = corr + np.where(d <= _ts[0], 0.,
                                                   np.interp(d, _ts, _om, left=0., right=0.))
                    if nnn is not None:
                        d12b = np.broadcast_to(d12, d23.shape)
                        d13b = np.broadcast_to(d13, d23.shape)
                        cpsi = np.clip((d12b**2 + d13b**2 - d23**2)/(2*d12b*d13b), -1., 1.)
                        psi = np.clip(np.arccos(cpsi), _phis_nnn[0], _phis_nnn[-1])
                        corr = corr + _interp3(np.stack([d12b, d13b, psi], axis=-1))
                    self.npcf[:, :, i1, i2, i3] *= corr

    ## INTEGRATED MEASURES ##
    def computeMapNap3(self, radii, nmax_trafo=None, basis='MapMx', radii_M=None,
                       xi=None, nnn=None, count_floor=0.1):
        r"""Computes the fourth-order aperture statistics
        :math:`\langle M_\mathrm{ap}(\theta_M)\,\mathcal{N}_\mathrm{ap}^3(\theta_N)\rangle`
        using the exponential filter of Crittenden 2002.

        Parameters
        ----------
        radii: numpy.ndarray
            Aperture scales :math:`\theta_N` of the three number-count filters.
        nmax_trafo: int, optional
            Largest multipole used in the transformation. Defaults to ``nmaxs[0]``.
        basis: str
            One of ``['MapMx','MM*','both']`` (equivalent for this statistic).
        radii_M: numpy.ndarray, optional
            Aperture scales :math:`\theta_M` of the shear filter, paired elementwise
            with ``radii``. Defaults to ``radii`` (equal-scale statistics).
        xi: tuple, optional
            Angular clustering 2PCF of the lenses as ``(thetas, omega)``; enables the
            clustering correction recovering the pure correlator, see
            :meth:`apply_clustering_correction`.
        nnn: orpheus.NNNCorrelation or tuple, optional
            Connected lens 3PCF for the :math:`\zeta` term of the clustering
            correction, see :meth:`apply_clustering_correction`.
        count_floor: float
            Threshold on the reconstructed multiplet counts below which the 4PCF is
            set to zero (suppresses multipole ringing around empty configurations).
        """

        assert(basis in ['MapMx','MM*','both'])

        if nmax_trafo is None:
            nmax_trafo=self.nmaxs[0]
        if radii_M is None:
            radii_M = radii
        assert(len(radii_M)==len(radii))

        # Retrieve all the aperture measures in the MM* basis via the 5D transformation eqns
        MN3correlators = np.zeros(1*len(radii), dtype=np.complex128)
        cc_s, _keep_cc = build_clustcorr(self, xi, nnn, count_floor)
        self.clib.fourpcfmultipoles2MN3correlators(
            np.int32(self.nmaxs[0]), np.int32(nmax_trafo),
            self.bin_edges, self.bin_centers_mean, np.int32(self.nbinsr),
            radii.astype(np.float64), radii_M.astype(np.float64), np.int32(len(radii)),
            self.phis[0].astype(np.float64), self.phis[1].astype(np.float64),
            self.dphis[0].astype(np.float64), self.dphis[1].astype(np.float64),
            len(self.phis[0]), len(self.phis[1]),
            np.int32(self.proj_dict[self.projection]), np.int32(self.nthreads),
            np.int32(self._verbose_c+self._verbose_debug),
            ct.byref(cc_s),
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
            Gtilde_third = np.zeros(self.nbinsr*self.nbinsr*self.nbinsphi[0],dtype=np.complex128)

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
            Gtilde_third = np.zeros(self.nbinsr*self.nbinsr*self.nbinsphi[0],dtype=np.complex128)

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
    # thetamin_xi, thetamax_xi, ntheta_xi is the linspaced array in which the xipm are passed to the C function
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