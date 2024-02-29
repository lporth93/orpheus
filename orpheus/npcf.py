from abc import ABC, abstractmethod

import ctypes as ct
from copy import deepcopy
from itertools import accumulate
import glob
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path
from .catalog import Catalog, ScalarTracerCatalog, SpinTracerCatalog, MultiTracerCatalog
from .utils import get_site_packages_dir, search_file_in_site_package


__all__ = ["BinnedNPCF", 
           "GGGCorrelation", "FFFCorrelation", "GNNCorrelation", "NGGCorrelation",
           "GGGGCorrelation", "XipmMixedCovariance"]

#########################################################
## ABSTRACT BASE CLASSES FOR NPCF AND THEIR MULTIPOLES ##
#########################################################        
class BinnedNPCF:
    
    def __init__(self, order, spins, n_cfs, min_sep, max_sep, nbinsr=None, binsize=None, nbinsphi=100, 
                 nmaxs=30, method="Tree", multicountcorr=True, diagrenorm=True, shuffle_pix=True,
                 tree_resos=[0,0.25,0.5,1.,2.], tree_redges=None, rmin_pixsize=20):
        
        self.order = int(order)
        self.n_cfs = int(n_cfs)
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbinsphi = nbinsphi
        self.nmaxs = nmaxs
        self.method = method
        self.multicountcorr = int(multicountcorr)
        self.diagrenorm = int(diagrenorm)
        self.shuffle_pix = shuffle_pix
        self.methods_avail = ["Discrete", "Tree", "DoubleTree"]
        self.tree_resos = np.asarray(tree_resos, dtype=np.float64)
        self.tree_nresos = int(len(self.tree_resos))
        self.tree_redges = tree_redges
        self.rmin_pixsize = rmin_pixsize
        self.tree_resosatr = None
        self.bin_centers = None
        self.bin_centers_mean = None
        self.phis = [None]*self.order
        self.npcf = None
        self.npcf_norm = None
        self.npcf_multipoles = None
        self.npcf_multipoles_norm = None
        self.is_edge_corrected = False
        
        # Check types or arguments
        if isinstance(self.nbinsphi, int):
            self.nbinsphi = self.nbinsphi*np.ones(order-2)
        self.nbinsphi =  self.nbinsphi.astype(np.int32)
        if isinstance(self.nmaxs, int):
            self.nmaxs = self.nmaxs*np.ones(order-2)
        self.nmaxs = self.nmaxs.astype(np.int32)
        if isinstance(spins, int):
            spins = spins*np.ones(order).astype(np.int32)
        self.spins = np.asarray(spins, dtype=np.int32)
        assert(isinstance(self.order, int))
        assert(isinstance(self.spins, np.ndarray))
        assert(isinstance(self.spins[0], np.int32))
        assert(len(spins)==self.order)
        assert(isinstance(self.n_cfs, int))
        assert(isinstance(self.min_sep, float))
        assert(isinstance(self.max_sep, float))
        assert(isinstance(self.nbinsphi, np.ndarray))
        assert(isinstance(self.nbinsphi[0], np.int32))
        assert(len(self.nbinsphi)==self.order-2)
        assert(isinstance(self.nmaxs, np.ndarray))
        assert(isinstance(self.nmaxs[0], np.int32))
        assert(len(self.nmaxs)==self.order-2)
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
            self.nbinsr = int(np.ceil(np.log(self.max_sep/self.min_sep)/binsize))
        assert(isinstance(self.nbinsr, int))
        self.bin_edges = np.geomspace(self.min_sep, self.max_sep, self.nbinsr+1)
        self.binsize = np.log(self.bin_edges[1]/self.bin_edges[0])
        # Setup variable for tree estimator
        if self.tree_redges != None:
            assert(isinstance(self.tree_redges, np.ndarray))
            self.tree_redges = self.tree_redges.astype(np.float64)
            assert(len(self.tree_redges)==self.tree_resos+1)
            self.tree_redges = np.sort(self.tree_redges)
            assert(self.tree_redges[0]==self.min_sep)
            assert(self.tree_redges[-1]==self.max_sep)
        else:
            self.tree_redges = np.zeros(len(self.tree_resos)+1)
            self.tree_redges[-1] = self.max_sep
            for elreso, reso in enumerate(self.tree_resos):
                self.tree_redges[elreso] = (reso==0.)*self.min_sep + (reso!=0.)*self.rmin_pixsize*reso
        _tmpreso = 0
        self.tree_resosatr = np.zeros(self.nbinsr, dtype=np.int32)
        for elbin, rbin in enumerate(self.bin_edges[:-1]):
            if rbin > self.tree_redges[_tmpreso+1]:
                _tmpreso += 1
            self.tree_resosatr[elbin] = _tmpreso
            
        # Setup phi bins
        for elp in range(self.order-2):
            _ = np.linspace(0,2*np.pi,self.nbinsphi[elp]+1)
            self.phis[elp] = .5*(_[1:] + _[:-1])      
          
        #############################
        ## Link compiled libraries ##
        #############################
        # Method that works for LP
        target_path = __import__('orpheus').__file__
        self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        # Method that works for RR (but not for LP with a local HPC install)
        #self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"orpheus_clib"))
        #self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        #print(self.library_path)
        p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)
        
        ## Third order shear statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([2, 2, 2], dtype=np.int32)):
            # Discrete estimator of third-order shear correlation function
            self.clib.alloc_Gammans_discrete_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_ggg.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, 
                ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, 
                p_f64, ct.c_int32, ct.c_int32, p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Tree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_tree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_tree_ggg.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Doubletree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_doubletree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_ggg.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, p_i32, ct.c_int32, 
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)]             
            
        if self.order==3:
            self.clib.alloc_triplets_tree_xipxipcov.restype = ct.c_void_p
            self.clib.alloc_triplets_tree_xipxipcov.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64 , p_i32,
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            self.clib.alloc_triplets_doubletree_xipxipcov.restype = ct.c_void_p
            self.clib.alloc_triplets_doubletree_xipxipcov.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, p_i32, ct.c_int32, 
                p_i32, p_f64, p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            

        # Shear-Lens-Lens correlations
        if self.order==3 and np.array_equal(self.spins, np.array([2, 0, 0], dtype=np.int32)):
            # Allocate Gamman for gnn via the discrete estimator
            self.clib.alloc_Gammans_discrete_gnn.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_gnn.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, 
                p_i32, p_i32, p_i32,
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.double),
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)]

        # Shear-Shear-Lens correlations
        if self.order==3 and np.array_equal(self.spins, np.array([2,2,0], dtype=np.int32)):
            self.clib.alloc_Gammans_discrete_ggn.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_ggn.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, 
                p_i32, p_i32, p_i32,
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.double),
                np.ctypeslib.ndpointer(dtype=np.complex),
                np.ctypeslib.ndpointer(dtype=np.complex)]
        
    ############################################################
    ## Functions that deal with different projections of NPCF ##
    ############################################################
    def _initprojections(self, child):
        assert(child.projection in child.projections_avail)
        child.project = {}
        for proj in child.projections_avail:
            child.project[proj] = {}
            for proj2 in child.projections_avail:
                if proj==proj2:
                    child.project[proj][proj2] = lambda: child.npcf
                else:
                    child.project[proj][proj2] = None
                    
    def _projectnpcf(self, child, projection):
        """
        Projects npcf to a new basis.
        """
        assert(child.npcf is not None)
        if projection not in child.projections_avail:
            print("Projection %s is not yet supported."%(projection))
            self.gen_npcfprojections_avail(child)

        if child.project[child.projection][projection] is not None:
            child.npcf = child.project[child.projection][projection]()
            child.projection = projection
        else:
            print("Projection from %s to %s is not yet implemented."%(child.projection,projection))
            self._gen_npcfprojections_avail(child)
                    
    def _gen_npcfprojections_avail(self, child):
        print("The following projections are available in the class %s:"%child.__class__.__name__)
        for proj in child.projections_avail:
            for proj2 in child.projections_avail:
                if child.project[proj][proj2] is not None:
                    print("  %s --> %s"%(proj,proj2))
 
    ####################
    ## MISC FUNCTIONS ##
    ####################
    def _checkcats(self, cats, spins):
        if isinstance(cats, list):
            assert(len(cats)==self.order)
        for els, s in enumerate(self.spins):
            if not isinstance(cats, list):
                thiscat = cats
            else:
                thiscat = cats[els]
            assert(thiscat.spin == s)
        

    def edge_correction(self, npcf_n, npcf_n_norm, ret_matrices=False):

        def gen_M_matrix(thet1,thet2,npcf_n_norm):
            nvals, ntheta, _ = npcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=int)
            nextM = np.zeros((nvals,nvals), dtype=complex)
            for ind, ell in enumerate(narr):
                lminusn = ell-narr
                sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
                nextM[ind,sel] = npcf_n_norm[(lminusn+nmax)[sel],thet1,thet2]/npcf_n_norm[nmax,thet1,thet2]
            return nextM
    
        nnvals, ntheta, _ = npcf_n_norm.shape
        nmax = int((nnvals-1)/2)
        print(nnvals, ntheta, nmax)
        
        npcf_n_corr = np.zeros_like(npcf_n)
        if ret_matrices:
            mats = np.zeros((ntheta,ntheta,nnvals,nnvals))
        for thet1 in range(ntheta):
            for thet2 in range(ntheta):
                nextM = gen_M_matrix(thet1,thet2,npcf_n_norm)
                if ret_matrices:
                    mats[thet1,thet2] = nextM
                npcf_n_corr[:,thet1,thet2] = np.linalg.inv(nextM)@npcf_n[:,thet1,thet2]
        if ret_matrices:
            return npcf_n_corr, mats
        return npcf_n_corr

    def addMultipoles(self, npcf_n, npcf_n_norm, do_edge_correction=True):

        nnvals, _, _ = npcf_n_norm.shape

        nmax = int((nnvals-1)/2)

        if do_edge_correction:
            npcf_n = self.edge_correction(npcf_n, npcf_n_norm)

        nbinsr=self.nbinsr
        nbinsphi=self.nbinsphi.item()
        Gamma_tot=np.zeros((nbinsr, nbinsr, nbinsphi), dtype=np.complex128)
        if ~do_edge_correction:
            Norm_tot=np.zeros((nbinsr, nbinsr, nbinsphi), dtype=np.complex128)
        for i in range(nbinsphi):
            phi=i*2*np.pi/nbinsphi
            for n in range(-nmax, nmax):
                phase=np.exp(1j*phi*n)
                tmp1=npcf_n_norm[n+nmax, :, :]*phase
                tmp2=npcf_n[n+nmax, :,:]*phase

                Gamma_tot[:,:,i]+=tmp2
                if ~do_edge_correction:
                    Norm_tot[:,:,i]+=tmp1
        
        if do_edge_correction:
            for i in range(nbinsphi):
                Gamma_tot[:,:,i]/=npcf_n_norm[nmax, :, :]
        else:
            Gamma_tot/=Norm_tot
        
        np.nan_to_num(Gamma_tot, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        Gamma_tot[np.abs(Gamma_tot)>1]=0

        return Gamma_tot, Norm_tot


        

##############################
## THIRD - ORDER STATISTICS ##
##############################
class GGGCorrelation(BinnedNPCF):
    """ This class stores the natural components of the shear correlation functions""
    
    Parameters:
        n_cfs (int):
        Number of natural components stored. While there are four natural components
        in total, it is usually sufficient to only compute Gamma0 and Gamma1.
    """
    
    def __init__(self, n_cfs, min_sep, max_sep, **kwargs):
        super().__init__(order=3, spins=np.array([2,2,2], dtype=np.int32), n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X", "Centroid"]
        self.nbinsz = None
        self.nzcombis = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        self.project["X"]["Centroid"] = self._x2centroid
        
    def process(self, cat, nthreads=16, dotomo=True, apply_edge_correction=True):
        self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            zbins = np.zeros(cat.ngal, dtype=np.int32)
            self.nzcombis = 1
        else:
            self.nbinsz = cat.nbinsz
            zbins = cat.zbins
            self.nzcombis = self.nbinsz*self.nbinsz*self.nbinsz
        sc = (4,self.nmax+1,self.nzcombis,self.nbinsr,self.nbinsr)
        sn = (self.nmax+1,self.nzcombis,self.nbinsr,self.nbinsr)
        szr = (self.nbinsz, self.nbinsr)
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        threepcfs_n = np.zeros(4*(self.nmax+1)*self.nzcombis*self.nbinsr*self.nbinsr).astype(np.complex128)
        threepcfsnorm_n = np.zeros((self.nmax+1)*self.nzcombis*self.nbinsr*self.nbinsr).astype(np.complex128)
        args_basecat = (cat.isinner.astype(np.int32), cat.weight, cat.pos1, cat.pos2, cat.tracer_1, cat.tracer_2, 
                        zbins.astype(np.int32), np.int32(self.nbinsz), np.int32(cat.ngal), )
        args_basesetup = (np.int32(0), np.int32(self.nmax), np.float64(self.min_sep), 
                          np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
                          np.int32(self.nbinsr), np.int32(self.multicountcorr), )
        if self.method=="Discrete":
            if not cat.hasspatialhash:
                cat.build_spatialhash(dpix=max(1.,self.max_sep//10.))
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
            args = (*args_basecat,
                    *args_basesetup,
                    cat.index_matcher,
                    cat.pixs_galind_bounds, 
                    cat.pix_gals,
                    *args_pixgrid,
                    np.int32(nthreads),
                    bin_centers,
                    threepcfs_n,
                    threepcfsnorm_n)
            func = self.clib.alloc_Gammans_discrete_ggg
        elif self.method in ["Tree", "DoubleTree"]:
            print("Doing multihash")
            print(cat)
            cutfirst = np.int32(self.tree_resos[0]==0.)
            print(self.tree_resos[cutfirst:])
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                  shuffle=self.shuffle_pix, w2field=True, normed=True)
            print("Done multihash")
            ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = mhash
            print(dpixs1_true, ngal_resos, len(pos1s))
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            isinner_resos = np.concatenate(isinners).astype(np.int32)
            e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
            e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
            _weightsq_resos = np.concatenate([allfields[i][2] for i in range(len(allfields))]).astype(np.float64)
            weightsq_resos = _weightsq_resos*weight_resos # As in reduce we renorm all the fields --> need to `unrenorm'
            print(np.mean(weight_resos), np.mean(_weightsq_resos), np.mean(weightsq_resos))
            index_matcher = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals = np.concatenate(pix_gals).astype(np.int32)
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
            args_resos = (weight_resos, pos1_resos, pos2_resos, e1_resos, e2_resos, zbin_resos, weightsq_resos,
                          index_matcher, pixs_galind_bounds, pix_gals, )
            args_output = (bin_centers, threepcfs_n, threepcfsnorm_n, )
            if self.method=="Tree":
                print("Doing Tree")
                args = (*args_basecat,
                        np.int32(self.tree_nresos),
                        self.tree_redges,
                        np.array(ngal_resos, dtype=np.int32),
                        *args_resos,
                        *args_pixgrid,
                        *args_basesetup,
                        np.int32(nthreads),
                        *args_output)
                func = self.clib.alloc_Gammans_tree_ggg
            if self.method=="DoubleTree":
                print("Doing DoubleTree")
                index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
                nregions = len(index_matcher_flat)
                args_basesetup_dtree = (np.int32(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), 
                                        np.int32(self.nbinsr), np.int32(self.multicountcorr), )
                #isinner_resos = np.ones_like(zbin_resos)
                args = (np.int32(self.tree_nresos),
                        np.int32(self.tree_nresos-cutfirst),
                        dpixs1_true.astype(np.float64),
                        dpixs2_true.astype(np.float64),
                        self.tree_redges,
                        np.array(ngal_resos, dtype=np.int32),
                        np.int32(self.nbinsz),
                        isinner_resos,
                        *args_resos,
                        *args_pixgrid,
                        np.int32(nregions),
                        index_matcher_flat.astype(np.int32),
                        *args_basesetup_dtree,
                        np.int32(nthreads),
                        *args_output)
                func = self.clib.alloc_Gammans_doubletree_ggg
       
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.npcf_multipoles = threepcfs_n.reshape(sc)
        self.npcf_multipoles_norm = threepcfsnorm_n.reshape(sn)
        self.projection = "X"
        
        if apply_edge_correction:
            self.edge_correction()
        
    def edge_correction(self, ret_matrices=False):
        
        def gen_M_matrix(thet1,thet2,threepcf_n_norm):
            nvals, ntheta, _ = threepcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=np.int)
            nextM = np.zeros((nvals,nvals))
            for ind, ell in enumerate(narr):
                lminusn = ell-narr
                sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
                nextM[ind,sel] = threepcf_n_norm[(lminusn+nmax)[sel],thet1,thet2].real/threepcf_n_norm[nmax,thet1,thet2].real
            return nextM
    
        nvals, nzcombis, ntheta, _ = self.npcf_multipoles_norm.shape
        nmax = nvals-1
        threepcf_n_full = np.zeros((4,2*nmax+1, nzcombis, ntheta, ntheta), dtype=complex)
        threepcf_n_norm_full = np.zeros((2*nmax+1, nzcombis, ntheta, ntheta), dtype=complex)
        threepcf_n_corr = np.zeros(threepcf_n_full.shape, dtype=np.complex)
        threepcf_n_full[:,nmax:] = self.npcf_multipoles
        threepcf_n_norm_full[nmax:] = self.npcf_multipoles_norm
        for nextn in range(1,nvals):
            threepcf_n_full[0,nmax-nextn] = self.npcf_multipoles[0,nextn].transpose(0,2,1)
            threepcf_n_full[1,nmax-nextn] = self.npcf_multipoles[1,nextn].transpose(0,2,1)
            threepcf_n_full[2,nmax-nextn] = self.npcf_multipoles[3,nextn].transpose(0,2,1)
            threepcf_n_full[3,nmax-nextn] = self.npcf_multipoles[2,nextn].transpose(0,2,1)
            threepcf_n_norm_full[nmax-nextn] = self.npcf_multipoles_norm[nextn].transpose(0,2,1)

        if ret_matrices:
            mats = np.zeros((nzcombis,ntheta,ntheta,nvals,nvals))
        for indz in range(nzcombis):
            #sys.stdout.write("%i"%indz)
            for thet1 in range(ntheta):
                for thet2 in range(ntheta):
                    nextM = gen_M_matrix(thet1,thet2,threepcf_n_norm_full[:,indz])
                    nextM_inv = np.linalg.inv(nextM)
                    if ret_matrices:
                        mats[indz,thet1,thet2] = nextM
                    for i in range(4):
                        threepcf_n_corr[i,:,indz,thet1,thet2] = np.matmul(nextM_inv,threepcf_n_full[i,:,indz,thet1,thet2])
                        
        self.npcf_multipoles = threepcf_n_corr[:,nmax:]
        self.is_edge_corrected = True
        
        if ret_matrices:
            return threepcf_n_corr[:,nmax:], mats
    
    def multipoles2npcf(self):
        
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        self.npcf = np.zeros((4, nzcombis, rbins, rbins, len(self.phi)), dtype=complex)
        self.npcf_norm = np.zeros((nzcombis, rbins, rbins, len(self.phi)), dtype=complex)
        ztiler = np.arange(self.nbinsz*self.nbinsz*self.nbinsz).reshape(
            (self.nbinsz,self.nbinsz,self.nbinsz)).transpose(0,2,1).flatten().astype(np.int32)
        
        # 3PCF components
        conjmap = [0,1,3,2]
        for elm in range(4):
            for elphi, phi in enumerate(self.phi):
                N0 = 1./(2*np.pi) * self.npcf_multipoles_norm[0].astype(complex)
                tmp =  1./(2*np.pi) * self.npcf_multipoles[elm,0].astype(complex)
                for n in range(1,self.nmax+1):
                    _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                    tmp += _const * self.npcf_multipoles[elm,n].astype(complex)
                    tmp += _const.conj() * self.npcf_multipoles[conjmap[elm],n][ztiler].astype(complex).transpose(0,2,1)
                self.npcf[elm,...,elphi] = tmp/N0.real
        # Number of triangles
        for elphi, phi in enumerate(self.phi):
            tmptotnorm = 1./(2*np.pi) * self.npcf_multipoles_norm[0].astype(complex)
            for n in range(1,self.nmax+1):
                _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                tmptotnorm += _const * self.npcf_multipoles_norm[n].astype(complex)
                tmptotnorm += _const.conj() * self.npcf_multipoles_norm[n][ztiler].astype(complex).transpose(0,2,1)
            self.npcf_norm[...,elphi] = tmptotnorm
        self.projection = "X"
    
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
                bin3 = np.sqrt(bin1**2 + bin2**2 - 2*bin1*bin2*np.cos(self.phi))
                phiexp = np.exp(1J*self.phi)
                phiexp_c = np.exp(-1J*self.phi)
                prod1 = (bin1 + bin2*phiexp_c)/(bin1 + bin2*phiexp) #q1
                prod2 = (2*bin1 - bin2*phiexp_c)/(2*bin1 - bin2*phiexp) #q2
                prod3 = (2*bin2*phiexp_c - bin1)/(2*bin2*phiexp - bin1) #q3
                prod1_inv = prod1.conj()/np.abs(prod1)
                prod2_inv = prod2.conj()/np.abs(prod2)
                prod3_inv = prod3.conj()/np.abs(prod3)
                rot_nom = np.zeros((4,len(self.phi)))
                rot_nom[0] = pimod(np.angle(prod1*prod2*prod3*np.exp(3*1J*self.phi)))
                rot_nom[1] = pimod(np.angle(prod1_inv*prod2*prod3*np.exp(1J*self.phi)))
                rot_nom[2] = pimod(np.angle(prod1*prod2_inv*prod3*np.exp(3*1J*self.phi)))
                rot_nom[3] = pimod(np.angle(prod1*prod2*prod3_inv*np.exp(-1J*self.phi)))
                gammas_cen[:,:,elb1,elb2] = self.npcf[:,:,elb1,elb2]*np.exp(1j*rot_nom)[:,np.newaxis,:]
        return gammas_cen
        
        
    def computeMap3(self, radii, do_multiscale=False, tofile=False, filtercache=None):
        """
        Compute third-order aperture statistics
        """
        
        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf()
            
        if self.projection != "Centroid":
            self.projectnpcf("Centroid")
        
        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            filterfunc = self._map3_filtergrid_singleR
            _rcut = 1 
        else:
            nrcombis = nradii*nradii*nradii
            filterfunc = self._map3_filtergrid_multiR
            _rcut = nradii
        map3s = np.zeros((8, self.nzcombis, nrcombis), dtype=complex)
        M3 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M2M1 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M2M2 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M2M3 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        tmprcombi = 0
        
        for elr1, R1 in enumerate(radii):
            for elr2, R2 in enumerate(radii[:_rcut]):
                for elr3, R3 in enumerate(radii[:_rcut]):
                    if filtercache is not None:
                        T0, T3_123, T3_231, T3_312 = filtercache[tmprcombi][0], filtercache[tmprcombi][1], filtercache[tmprcombi][2], filtercache[tmprcombi][3]
                    else:
                        T0, T3_123, T3_231, T3_312 = filterfunc(R1, R2, R3)
                    M3[:,tmprcombi] = np.nansum(T0*self.npcf[0,...],axis=(1,2,3))
                    M2M1[:,tmprcombi] = np.nansum(T3_123*self.npcf[1,...],axis=(1,2,3))
                    M2M2[:,tmprcombi] = np.nansum(T3_231*self.npcf[2,...],axis=(1,2,3))
                    M2M3[:,tmprcombi] = np.nansum(T3_312*self.npcf[3,...],axis=(1,2,3))
                    tmprcombi += 1            
        map3s[0] = 1./4. * (+M2M1+M2M2+M2M3 + M3).real # MapMapMap
        map3s[1] = 1./4. * (+M2M1+M2M2-M2M3 + M3).imag # MapMapMx
        map3s[2] = 1./4. * (+M2M1-M2M2+M2M3 + M3).imag # MapMxMap
        map3s[3] = 1./4. * (-M2M1+M2M2+M2M3 + M3).imag # MxMapMap
        map3s[4] = 1./4. * (-M2M1+M2M2+M2M3 - M3).real # MapMxMx
        map3s[5] = 1./4. * (+M2M1-M2M2+M2M3 - M3).real # MxMapMx
        map3s[6] = 1./4. * (+M2M1+M2M2-M2M3 - M3).real # MxMxMap
        map3s[7] = 1./4. * (+M2M1+M2M2+M2M3 - M3).imag # MxMxMx
                                    
        if tofile:
            # Write to file
            pass
            
        return map3s
    
    def _map3_filtergrid_singleR(self, R1, R2, R3):
        phis = self.phi
        normys_edges = self.bin_edges
        normys_centers = self.bin_centers_mean
        R_ap = R1
        
        nbinsr = len(normys_centers)
        nbinsphi = len(phis)
        _cphis = np.cos(phis)
        _c2phis = np.cos(2*phis)
        _sphis = np.sin(phis)
        _ephis = np.e**(1J*phis)
        _ephisc = np.e**(-1J*phis)
        _e2phis = np.e**(2J*phis)
        _e2phisc = np.e**(-2J*phis)
        T0 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        T3_123 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        T3_231 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        T3_312 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        for elb1 in range(nbinsr):
            _y1 = normys_centers[elb1]
            _dbin1 = normys_edges[elb1+1] - normys_edges[elb1]
            for elb2 in range(nbinsr):
                _y2 = normys_centers[elb2]
                _y14 = _y1**4
                _y13y2 = _y1**3*_y2
                _y12y22 = _y1**2*_y2**2
                _y1y23 = _y1*_y2**3
                _y24 = _y2**4
                _dbin2 = normys_edges[elb2+1] - normys_edges[elb2]
                _dbinphi = phis[1] - phis[0]
                _absq1s = 1./9.*(4*_y1**2 - 4*_y1*_y2*_cphis + 1*_y2**2)
                _absq2s = 1./9.*(1*_y1**2 - 4*_y1*_y2*_cphis + 4*_y2**2)
                _absq3s = 1./9.*(1*_y1**2 + 2*_y1*_y2*_cphis + 1*_y2**2)
                _absq123s = 2./3. * (_y1**2+_y2**2-_y1*_y2*_cphis)
                _absq1q2q3_2 = _absq1s*_absq2s*_absq3s
                _measures = _y1*_dbin1/R_ap**2 * _y2*_dbin2/R_ap**2 * _dbinphi/(2*np.pi)
                nextT0 = _absq1q2q3_2/R_ap**6 * np.e**(-_absq123s/(2*R_ap**2))
                T0[elb1,elb2] = 1./24. * _measures * nextT0
                _tmp1 = _y1**4 + _y2**4 + _y1**2*_y2**2 * (2*np.cos(2*phis)-5.)
                _tmp2 = (_y1**2+_y2**2)*_cphis + 9J*(_y1**2-_y2**2)*_sphis
                q1q2q3starsq = -1./81*(2*_tmp1 - _y1*_y2*_tmp2)
                nextT3_123 = np.e**(-_absq123s/(2*R_ap**2)) * (1./24*_absq1q2q3_2/R_ap**6 -
                                                               1./9.*q1q2q3starsq/R_ap**4 +
                                                               1./27*(q1q2q3starsq**2/(_absq1q2q3_2*R_ap**2) +
                                                                      2*q1q2q3starsq/(_absq3s*R_ap**2)))
                _231inner = -4*_y14 + 2*_y24 + _y13y2*8*_cphis + _y12y22*(8*_e2phis-4-_e2phisc) + _y1y23*(_ephisc-8*_ephis)
                q2q3q1starsq = -1./81*(_231inner)
                nextT3_231 = np.e**(-_absq123s/(2*R_ap**2)) * (1./24*_absq1q2q3_2/R_ap**6 -
                                                               1./9.*q2q3q1starsq/R_ap**4 +
                                                               1./27*(q2q3q1starsq**2/(_absq1q2q3_2*R_ap**2) +
                                                                      2*q2q3q1starsq/(_absq1s*R_ap**2)))
                _312inner = 2*_y14 - 4*_y24 - _y13y2*(8*_ephisc-_ephis) - _y12y22*(4+_e2phis-8*_e2phisc) + 8*_y1y23*_cphis
                q3q1q2starsq = -1./81*(_312inner)
                nextT3_312 = np.e**(-_absq123s/(2*R_ap**2)) * (1./24*_absq1q2q3_2/R_ap**6 -
                                                               1./9.*q3q1q2starsq/R_ap**4 +
                                                               1./27*(q3q1q2starsq**2/(_absq1q2q3_2*R_ap**2) +
                                                                      2*q3q1q2starsq/(_absq2s*R_ap**2)))
                T3_123[elb1,elb2] = _measures * nextT3_123
                T3_231[elb1,elb2] = _measures * nextT3_231
                T3_312[elb1,elb2] = _measures * nextT3_312

        return T0, T3_123, T3_231, T3_312
    
    def _map3_filtergrid_multiR(self, R1, R2, R3, include_measure=True):
        phis = self.phi
        normys_edges = self.bin_edges
        normys_centers = self.bin_centers_mean
        nbinsr = len(normys_centers)
        nbinsphi = len(phis)
        _cphis = np.cos(phis)
        _c2phis = np.cos(2*phis)
        _sphis = np.sin(phis)
        _ephis = np.e**(1J*phis)
        _ephisc = np.e**(-1J*phis)
        _e2phis = np.e**(2J*phis)
        _e2phisc = np.e**(-2J*phis)
        T0 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        T3_123 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        T3_231 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        T3_312 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=complex)
        for elb1 in range(nbinsr):
            _y1 = normys_centers[elb1]
            _dbin1 = normys_edges[elb1+1] - normys_edges[elb1]
            for elb2 in range(nbinsr):
                Theta2 = np.sqrt((R1**2*R2**2 + R1**2*R3**2 + R2**2*R3**2)/3)
                S = R1**2*R2**2*R3**2/Theta2**3

                _y2 = normys_centers[elb2]
                _y14 = _y1**4
                _y13y2 = _y1**3*_y2
                _y12y22 = _y1**2*_y2**2
                _y1y23 = _y1*_y2**3
                _y24 = _y2**4
                _dbin2 = normys_edges[elb2+1] - normys_edges[elb2]
                _dbinphi = phis[1] - phis[0]
                _absq1s = 1./9.*(4*_y1**2 - 4*_y1*_y2*_cphis + 1*_y2**2)
                _absq2s = 1./9.*(1*_y1**2 - 4*_y1*_y2*_cphis + 4*_y2**2)
                _absq3s = 1./9.*(1*_y1**2 + 2*_y1*_y2*_cphis + 1*_y2**2)
                _absq123s = 2./3. * (_y1**2+_y2**2-_y1*_y2*_cphis)
                _absq1q2q3_2 = _absq1s*_absq2s*_absq3s

                Z = ((-R1**2+2*R2**2+2*R3**2)*_absq1s + (2*R1**2-R2**2+2*R3**2)*_absq2s + (2*R1**2+2*R2**2-R3**2)*_absq3s)/(6*Theta2**2)
                _frac231c = 1./3.*_y2*(2*_y1*_ephis-_y2)/_absq1s
                _frac312c = 1./3.*_y1*(_y1-2*_y2*_ephisc)/_absq2s
                _frac123c = 1./3.*(_y2**2-_y1**2+2J*_y1*_y2*_sphis)/_absq3s
                f1 = (R2**2+R3**2)/(2*Theta2) + _frac231c * (R2**2-R3**2)/(6*Theta2)
                f2 = (R1**2+R3**2)/(2*Theta2) + _frac312c * (R3**2-R1**2)/(6*Theta2)
                f3 = (R1**2+R2**2)/(2*Theta2) + _frac123c * (R1**2-R2**2)/(6*Theta2)
                f1c = f1.conj()
                f2c = f2.conj()
                f3c = f3.conj()
                g1c = (R2**2*R3**2/Theta2**2 + R1**2*(R3**2-R2**2)/(3*Theta2**2)*_frac231c).conj()
                g2c = (R3**2*R1**2/Theta2**2 + R2**2*(R1**2-R3**2)/(3*Theta2**2)*_frac312c).conj()
                g3c = (R1**2*R2**2/Theta2**2 + R3**2*(R2**2-R1**2)/(3*Theta2**2)*_frac123c).conj()
                _measures = _y1*_dbin1/Theta2 * _y2*_dbin2/Theta2 * _dbinphi/(2*np.pi)
                if not include_measure:
                    _measures/=_measures
                nextT0 = _absq1q2q3_2/Theta2**3 * f1c**2*f2c**2*f3c**2 * np.e**(-Z)
                T0[elb1,elb2] = S/24. * _measures * nextT0

                _tmp1 = _y1**4 + _y2**4 + _y1**2*_y2**2 * (2*np.cos(2*phis)-5.)
                _tmp2 = (_y1**2+_y2**2)*_cphis + 9J*(_y1**2-_y2**2)*_sphis
                q1q2q3starsq = -1./81*(2*_tmp1 - _y1*_y2*_tmp2)
                nextT3_123 = np.e**(-Z) * (1./24*_absq1q2q3_2/Theta2**3 * f1c**2*f2c**2*f3**2 -
                                           1./9.*q1q2q3starsq/Theta2**2 * f1c*f2c*f3*g3c +
                                           1./27*(q1q2q3starsq**2/(_absq1q2q3_2*Theta2) * g3c**2 +
                                                  2*R1**2*R2**2/Theta2**2 * q1q2q3starsq/(_absq3s*Theta2) * f1c*f2c))
                _231inner = -4*_y14 + 2*_y24 + _y13y2*8*_cphis + _y12y22*(8*_e2phis-4-_e2phisc) + _y1y23*(_ephisc-8*_ephis)
                q2q3q1starsq = -1./81*(_231inner)
                nextT3_231 = np.e**(-Z) * (1./24*_absq1q2q3_2/Theta2**3 * f2c**2*f3c**2*f1**2 -
                                           1./9.*q2q3q1starsq/Theta2**2 * f2c*f3c*f1*g1c +
                                           1./27*(q2q3q1starsq**2/(_absq1q2q3_2*Theta2) * g1c**2 +
                                                  2*R2**2*R3**2/Theta2**2 * q2q3q1starsq/(_absq1s*Theta2) * f2c*f3c))
                _312inner = 2*_y14 - 4*_y24 - _y13y2*(8*_ephisc-_ephis) - _y12y22*(4+_e2phis-8*_e2phisc) + 8*_y1y23*_cphis
                q3q1q2starsq = -1./81*(_312inner)
                nextT3_312 = np.e**(-Z) * (1./24*_absq1q2q3_2/Theta2**3 * f3c**2*f1c**2*f2**2 -
                                           1./9.*q3q1q2starsq/Theta2**2 * f3c*f1c*f2*g2c +
                                           1./27*(q3q1q2starsq**2/(_absq1q2q3_2*Theta2) * g2c**2 +
                                                  2*R3**2*R1**2/Theta2**2 * q3q1q2starsq/(_absq2s*Theta2) * f3c*f1c))

                T3_123[elb1,elb2] = S * _measures * nextT3_123
                T3_231[elb1,elb2] = S * _measures * nextT3_231
                T3_312[elb1,elb2] = S * _measures * nextT3_312

        return T0, T3_123, T3_231, T3_312


    

class GNNCorrelation(BinnedNPCF):
    """ Shear-Lens-Lens (G3L) correlation function """
    def __init__(self, n_cfs, min_sep, max_sep, **kwargs):
        super().__init__(3, [2,0,0], n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X"]
        self.nbinsz_source = None
        self.nbinsz_lens = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        
    def process(self, cat_lens, cat_source, nthreads=16, dotomo=False):
        self._checkcats([cat_source, cat_lens, cat_lens], [2, 0,0])

        if dotomo:
            raise NotImplementedError('For SLL correlation tomography is not yet implemented')

         # Prepare output
        nbinsr=self.nbinsr
        nmax=self.nmax
        bin_centers = np.zeros(nbinsr).astype(np.float64)
        threepcfs_n = np.zeros(nbinsr*nbinsr*(2*nmax+1)).astype(complex)
        threepcfsnorm_n = np.zeros(nbinsr*nbinsr*(2*nmax+1)).astype(complex)
        
        cat_lens.build_spatialhash()


        self.clib.alloc_Gammans_discrete_gnn(cat_source.isinner.astype(np.int32),
                cat_source.weight,
                cat_source.pos1,
                cat_source.pos2,
                cat_source.tracer_1,
                cat_source.tracer_2,
                cat_source.ngal,
                cat_lens.weight,
                cat_lens.pos1,
                cat_lens.pos2,
                cat_lens.ngal,
                self.nmax,
                self.min_sep,
                self.max_sep,
                self.nbinsr,
                np.int32(self.multicountcorr),
                cat_lens.index_matcher,
                cat_lens.pixs_galind_bounds,
                cat_lens.pix_gals,
                cat_lens.pix1_start,
                cat_lens.pix1_d, 
                cat_lens.pix1_n, 
                cat_lens.pix2_start,
                cat_lens.pix2_d, 
                cat_lens.pix2_n,
                nthreads,
                bin_centers,
                threepcfs_n,
                threepcfsnorm_n
                )
        self.bin_centers=bin_centers.reshape(nbinsr)
        self.npcf_multipoles = threepcfs_n.reshape(((2*nmax+1),nbinsr,nbinsr))
        self.npcf_multipoles_norm = threepcfsnorm_n.reshape(((2*nmax+1),nbinsr,nbinsr))

        # if not dotomo:
        #     self.nbinsz_lens = 1
        #     self.nbinsz_source = 1
        #     zbins_lens = np.zeros(cat_lens.ngal, dtype=np.int32)
        #     zbins_source = np.zeros(cat_source.ngal, dtype=np.int32)
        # else:
        #     self.nbinsz_lens = cat_lens.nbinsz
        #     self.nbinsz_source = cat_source.nbinsz
        # _z3combis = self.nbinsz_source*self.nbinsz_lens*self.nbinsz_lens
        # _r2combis = self.nbinsr*self.nbinsr
        # sc = (4,self.nmax+1, _z3combis, self.nbinsr,self.nbinsr)
        # sn = (self.nmax+1, _z3combis, self.nbinsr,self.nbinsr)
        # szr = (self.nbinsz_lens, self.nbinsz_source, self.nbinsr)
        # bin_centers = np.zeros(self.nbinsz_source*self.nbinsz_lens*self.nbinsr).astype(np.float64)
        # threepcfs_n = np.zeros(4*(self.nmax+1)*_z3combis*_r2combis).astype(complex)
        # threepcfsnorm_n = np.zeros((self.nmax+1)*_z3combis*_r2combis).astype(complex)
        # args_basecat = (cat.isinner.astype(np.int32), cat.weight, cat.pos1, cat.pos2, cat.tracer_1, cat.tracer_2, 
        #                 zbins.astype(np.int32), int(self.nbinsz), int(cat.ngal), )
        # args_basesetup = (int(0), int(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
        #                   int(self.nbinsr), int(self.multicountcorr), )
    

    def multipoles2npcf(self, do_edge_correction=True):
        self.gnn_correlation, self.gnn_norm = self.addMultipoles(self.npcf_multipoles, self.npcf_multipoles_norm, do_edge_correction=do_edge_correction)


class NGGCorrelation(BinnedNPCF):
    """ Lens-Shear-Shear correlation function """
    def __init__(self, n_cfs, min_sep, max_sep, **kwargs):
        super().__init__(3, [0,2,2], n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
        
    def process_single(self, cat, nthreads, dotomo=True):
        pass
    
class FFFCorrelation(BinnedNPCF):
    """ Third-Order Flexion correlation function """
    def __init__(self, n_cfs, min_sep, max_sep, **kwargs):
        super().__init__(3, [3,3,3], n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
        
    def process_single(self, cat, nthreads, dotomo=True):
        pass
    
#############################
## FOURTH-ORDER STATISTICS ##
#############################

class GGGGCorrelation(BinnedNPCF):
    """ This class stores the natural components of the shear four point correlation functions.""
    
    Parameters:
        n_cfs (int):
        Number of natural components stored. While there are four natural components
        in total, it is usually sufficient to only compute Gamma0 and Gamma1.
    """
    
    def __init__(self, n_cfs, nbinsr, nbinsphi, projection, nmax=30):
        super().__init__(4, [2,2,2,2], n_cfs, nbinsr, nbinsphi, nmax)
        
    def process_single(self, cat, nthreads, dotomo=True):
        pass        
        
    def computeMap4(self, radii, do_multiscale=False, tofile=False):
        """
        Compute fourth-order aperture statistics
        """
        thisproj = self.projection
        if thisproj != "Centroid":
            self.toprojection("Centroid")
        # Compute Map3...
        
        if thisproj != "Centroid":
            self.toprojection(thisproj)
            
        if tofile:
            # Write to file
            pass
            
        return map3  
    
###########################
## COVARIANCE STATISTICS ##
###########################

class XipmMixedCovariance(BinnedNPCF):
    
    def __init__(self, min_sep_xi, max_sep_xi, nbins_xi, nsubbins, nmax=10, **kwargs):
        self.min_sep_xi = min_sep_xi
        self.max_sep_xi = max_sep_xi
        self.nsubbins = max(1,int(nsubbins))
        nbinsr = nsubbins*nbins_xi
        min_sep_triplets = float(min_sep_xi)
        max_sep_triplets = max_sep_xi
        super().__init__(order=3, spins=[0,0,0], n_cfs=1, min_sep=min_sep_triplets, max_sep=max_sep_triplets, nbinsr=nbinsr, 
                         nmaxs=nmax, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        
    def process(self, cat, nthreads=16, dotomo=True):
        """ Note that this cat should have a w^2 as its tracer """
        #self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            zbins = np.zeros(cat.ngal, dtype=np.int32)
        else:
            self.nbinsz = cat.nbinsz
            zbins = cat.zbins
        sc = (1,self.nmax+1,self.nbinsz*self.nbinsz*self.nbinsz,self.nbinsr,self.nbinsr)
        sn = (self.nmax+1,self.nbinsz*self.nbinsz*self.nbinsz,self.nbinsr,self.nbinsr)
        szr = (self.nbinsz, self.nbinsr)
        szzr = (self.nbinsz, self.nbinsz, self.nbinsr)
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        w2wwtriplets = np.zeros((self.nmax+1)*self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr).astype(complex)
        wwwtriplets = np.zeros((self.nmax+1)*self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr).astype(complex)
        wwcounts = np.zeros(self.nbinsz*self.nbinsz*self.nbinsr).astype(np.float64)
        w2wcounts = np.zeros(self.nbinsz*self.nbinsz*self.nbinsr).astype(np.float64)
        args_basecat = (cat.isinner.astype(np.int32), cat.weight.astype(np.float64), cat.pos1.astype(np.float64), cat.pos2.astype(np.float64), 
                        zbins.astype(np.int32), int(self.nbinsz), int(cat.ngal), )
        args_basesetup = (int(0), int(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
                          int(self.nbinsr), int(self.multicountcorr), )
        if self.method=="Discrete":
            raise NotImplementedError
        elif self.method in ["Tree", "DoubleTree"]:
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                  normed=False, shuffle=self.shuffle_pix)
            ngal_resos, pos1s, pos2s, weights, zbins, isinners, tracers, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            weight_sq_resos = np.concatenate([tracer[0] for tracer in tracers]).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            isinner_resos = np.concatenate(isinners).astype(np.int32)
            index_matcher = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals = np.concatenate(pix_gals).astype(np.int32)
            args_resos = (weight_resos, weight_sq_resos, pos1_resos, pos2_resos, zbin_resos, 
                              index_matcher, pixs_galind_bounds, pix_gals, )
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), int(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), int(cat.pix2_n), )
            args_output = (bin_centers, wwcounts, w2wcounts, w2wwtriplets, wwwtriplets, )
            if self.method=="Tree":
                print("Doing Tree")
                args = (*args_basecat,
                        self.tree_nresos,
                        self.tree_redges,
                        np.array(ngal_resos, dtype=np.int32),
                        *args_resos
                        *args_pixgrid,
                        *args_basesetup,
                        int(nthreads),
                        *args_output)
                func = self.clib.alloc_triplets_tree_xipxipcov
            if self.method=="DoubleTree":
                print("Doing DoubleTree")
                index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
                nregions = len(index_matcher_flat)
                args_basesetup_dtree = (np.int32(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), 
                                            np.int32(self.nbinsr), np.int32(self.multicountcorr), )
                
                args = (np.int32(self.tree_nresos),
                        np.int32(self.tree_nresos-cutfirst),
                        dpixs1_true.astype(np.float64),
                        dpixs2_true.astype(np.float64),
                        self.tree_redges,
                        np.array(ngal_resos, dtype=np.int32),
                        np.int32(self.nbinsz),
                        isinner_resos,
                        *args_resos,
                        *args_pixgrid,
                        np.int32(nregions),
                        index_matcher_flat.astype(np.int32),
                        *args_basesetup_dtree,
                        np.int32(nthreads),
                        *args_output)
                func = self.clib.alloc_triplets_doubletree_xipxipcov    
                
        #self.clib.alloc_triplets_doubletree_xipxipcov.argtypes = [
        #ct.c_int32, ct.c_int32, p_f64, p_f64 p_i32, ct.c_int32, 
        #p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32,
        #p_i32, p_i32, p_i32, 
        #ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
        #ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
        #np.ctypeslib.ndpointer(dtype=np.float64), 
        #np.ctypeslib.ndpointer(dtype=np.float64), 
        #np.ctypeslib.ndpointer(dtype=np.float64), 
        #np.ctypeslib.ndpointer(dtype=np.complex128),
        #np.ctypeslib.ndpointer(dtype=np.complex128)] 
        for elarg, arg in enumerate(args):
            print(elarg,arg) 
        
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.wwcounts = wwcounts.reshape(szzr)
        self.w2wcounts = w2wcounts.reshape(szzr)
        self.npcf_multipoles = w2wwtriplets.reshape(sc)
        self.npcf_multipoles_norm = wwwtriplets.reshape(sn)
        self.projection = None 
        
    def multipoles2npcf(self):
        
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        self.npcf = np.zeros((nzcombis, rbins, rbins, len(self.phi)), dtype=complex)
        self.npcf_norm = np.zeros((nzcombis, rbins, rbins, len(self.phi)), dtype=complex)
        ztiler = np.arange(self.nbinsz*self.nbinsz*self.nbinsz).reshape(
            (self.nbinsz,self.nbinsz,self.nbinsz)).transpose(0,2,1).flatten().astype(np.int32)

        # w*w*w triplets
        for elphi, phi in enumerate(self.phi):
            tmp = np.zeros((1,nzcombis, rbins, rbins), dtype=complex)
            tmpnorm = np.zeros((nzcombis, rbins, rbins), dtype=complex)
            for eln,n in enumerate(range(self.nmax+1)):
                _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                tmp += _const * self.npcf_multipoles[:,eln].astype(complex)
                tmpnorm += _const * self.npcf_multipoles_norm[:,eln].astype(complex)
                if n>0:
                    tmp += _const.conj() * self.npcf_multipoles[eln][ztiler].astype(complex).transpose(0,2,1)
                    tmpnorm += _const.conj() * self.npcf_multipoles_norm[eln][ztiler].astype(complex).transpose(0,2,1)
            self.npcf[...,elphi] = tmp
            self.npcf_norm[...,elphi] = tmpnorm
    
if False:

    class ThreePCF:

        """
        Config file has the following keys:
        - rmin: Smallest radius of 3pcf.
                (Only needs to be specified if we want to emply the discrete estimator at some point.)
        - rmax: Largest radius of 3pcf.
        - nbins_discrete: how many rbins we compute via the discrete estimator
        - nbins_baseres: how many rbins we compute via the FFT-estimator of the lowest resolution
        - discrete: Use discrete estimator all the way

        Config file is given as follows:
        - 
        """

        def __init__(self, modes="GGG", rmin=None, rmax=None, nrbins=None, nphibins=None, 
                     do_tomography_auto=False, do_tomography_full=False, 
                     method="Mixed", cache=True, ngal_in_maxres=1, ngal_in_minres=50, 
                     config=None):

            """
            Method could be in (ordered from slow to fast execution time - "Mixed" is recommended)
            - "Discrete":
               * Uses discrete estimator for every scale
            - "SingleFFT"
               * Uses FFT on fixed gridsize
               * Gridsize determined by ngal_in_maxres
            - "FFT":
               * Uses FFT estimator on different gridsizes
               * Gridsizes are determined by ngal_in_maxres, ngal_in_minres & rbinning
            - "Mixed"
               Use "Discrete" on small scales and "FFT" on larger scales (determined by rmin, ngal_in_maxres, ngal_in_minres, rmax)
            """

            ## Initialize arguments from init ##
            # Note that we autbuild the config file only after a catalog is given as
            # only in this case one can find suitable settings.
            self.methods_avail = ["Discrete", "FFT", "SingleFFT", "Mixed"]
            self.modes_avail = ["KKK", "KGG", "GKK", "GGG"]
            if config is not None:
                self._check_config(config)
            else:
                assert(method in self.methods_avail)
                assert(modes in self.modes_avail)
                self.method = method
                self.modes = modes
                self.rmin = rmin
                self.rmax = rmax
                self.nrbins = nrbins
                self.nphibins = nphibins
                self.do_tomography_auto = do_tomography_auto
                self.do_tomography_full = do_tomography_full
                self.ngal_in_maxres = ngal_in_maxres
                self.ngal_in_minres=ngal_in_minres
                self.cache = cache
                self.config = None

                self.cat = None
                self.hasthreepcf = False
                self.threepcf = None
                self.threepcf_norm = None
                self.projection = None 

            ## Link compiled libraries ##
            self.library_path = str(Path(__file__).parent.absolute()) + "/src/"
            discrete_library_fpath = self.library_path + "discrete.so"
            hash_library_fpath = self.library_path + "spatialhash.so"
            hash_library = ct.CDLL(hash_library_fpath)
            discrete_library = ct.CDLL(discrete_library_fpath)
            p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
            p_f64 = ndpointer(float, flags="C_CONTIGUOUS")
            p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
            p_i32 = ndpointer(int, flags="C_CONTIGUOUS")

            # Generate pixel --> galaxy mapping
            # Safely called within other wrapped functions
            hash_library._gen_pixeltable.restype = ct.c_void_p
            hash_library._gen_pixeltable.argtypes = [
                p_f64, p_f64, ct.c_int32, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.intc)]

            """# Allocates Gns and Gammans for discrete data such that one can evaluate all Gamman in [nmin, nmax]
            # Use 'alloc_Gns_discrete' to safely call this function
            discrete_library.alloc_GnsGammans_discrete_basic.restype = ct.c_void_p
            discrete_library.alloc_GnsGammans_discrete_basic.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, p_i32, p_i32,
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.int32), 
                np.ctypeslib.ndpointer(dtype=np.complex128), np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)]  
            """

            # Allocates Gns for discrete data such that one can evaluate all Gamman in [nmin, nmax]
            # Use 'alloc_Gns_discrete' to safely call this function
            discrete_library.alloc_Gns_discrete_basic.restype = ct.c_void_p
            discrete_library.alloc_Gns_discrete_basic.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, p_i32, p_i32,
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.int32), 
                np.ctypeslib.ndpointer(dtype=np.complex128), np.ctypeslib.ndpointer(dtype=np.complex128)]    

            # Allocate Gamman from Gns obtained via the discrete estimator
            # Use 'alloc_Gamman_discrete' to safely call this function
            discrete_library.alloc_Gamman_discrete_basic.restype = ct.c_void_p
            discrete_library.alloc_Gamman_discrete_basic.argtypes = [
                p_c128, p_c128, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128), np.ctypeslib.ndpointer(dtype=np.complex128)]

        ###########################################################################
        ## Functions that deal with the the config file for the 3pcf computation ##
        ###########################################################################
        #config = {}
        #config["rmax"] = 120.
        #config["do_tomography_auto"] = False # Only use autotomographic bins
        #config["do_tomography_full"] = False # Use all available tomographic bins
        #config["npix"] = [2048,1024,512] # All resolution of grids for FFT - they give npix in y-direction
        #config["nbins_baseres"] = 12 # How many rbins we compute in the base resolution
        #config["supperres_rshells"] = [[5,6,8,10,12,14,17,20],[10,12,14,17,20,24,29,34,40]] # Bin edges in number of pixels of resolutions
        #config["nmax"] = [[30]*7,[30]*8,30] # Largest multipole to choose for each multipole [(discrete), [superres], .., baseres]
        #config["cache"] = [[True]*7,[True]*8,30] # Whether to cache the Gns [(discrete), [superres], .., baseres]
        #config["edges_baseres"] # Optional - only required if method==SingleFFT
        #config["rmin"] = 1. # Optional - only required if method in [Discrete, Mixed]
        #config["nbins_discrete"] = 5 # How many radial bins we have in the discrete estimator (Optional - only required if method in [Discrete, Mixed])
        #config["nalloc_discrete"] = [10,20,30] # How to batch the n for disrete estimator allocation (Optional - only required if method in [Discrete, Mixed])
        #config["dpix_min"] = 0.2 # Optional - only required if method==Mixed
        #config["method"] = "Mixed" # What method to use 
        def _check_config(self, config):

            ## Check which method to choose
            if "method" not in config.keys():
                config["method"] = "Mixed"
                self.method="Mixed"
            else:
                assert(config["method"] in self.methods_avail)
                self.method = config["method"]  

            ## Check for method  "Discrete"
            if config["method"] == "Discrete":
                assert("nbins_discrete" in config.keys())
                assert(len(config["nmax"]) in [1,config["nbins_discrete"]])
                assert(len(config["cache"]) in [1,config["nbins_discrete"]])
                config.pop('npix', None)
                config.pop('nbins_baseres', None)
                config.pop('supperres_rshells', None)
                config.pop('edges_baseres', None)
                config.pop('nbins_discrete', None)
                config.pop('dpix_min', None)

            ## Check for method "SingleFFT"
            if config["method"] == "SingleFFT":
                assert("nbins_baseres" in config.keys())
                assert(len(config["npix"])==1)
                assert(len(config["nmax"]) in [1,config["nbins_baseres"]])
                assert(len(config["cache"]) in [1,config["nbins_baseres"]])
                config.pop('supperres_rshells', None)
                config.pop('rmin', None)
                config.pop('nbins_discrete', None)
                config.pop('dpix_min', None)

            ## Check for method "Mixed" and method "FFT"
            # Check if we need discrete
            if self.method in ["Mixed", "FFT"]:
                hasdiscrete = False
                if self.method=="Mixed":
                    if "rmin" in config.keys() and "nbins_discrete" in config.keys() and "dpix_min" in config.keys():
                        if config["supperres_rshells"][0][0]*config["dpix_min"] > config["rmin"]:
                            hasdiscrete = True
                        else:
                            config["nmax"].pop(0)
                            config["cache"].pop(0)

                            print("Discretee estimation not required - removed from config.")

                if not hasdiscrete:
                    config["method"] = "FFT"
                    self.method="FFT"
                    config.pop('rmin', None)
                    config.pop('nbins_discrete', None)
                    config.pop('dpix_min', None)

                # Check if config is valid
                assert(len(config["supperres_rshells"]) == len(config["nmax"])-1-hasdiscrete)
                assert(len(config["supperres_rshells"]) == len(config["npix"])-1)
                assert(len(config["cache"]) == len(config["npix"])+hasdiscrete)
                for elsuperres in range(len(config["supperres_rshells"])):
                    assert(len(config["supperres_rshells"][elsuperres]) == len(config["nmax"][elsuperres+hasdiscrete])+1)
                    assert(len(config["supperres_rshells"][elsuperres]) == len(config["cache"][elsuperres+hasdiscrete])+1)
                for elres in range(len(config["npix"])-1):
                    assert(config["npix"][elres]/config["npix"][elres+1]==2)

            ## Add optional parameters:
            # Tomography: Make sure that at most one of auto/full tomography is selected
            if "do_tomography_auto" not in config.keys():
                config["do_tomography_auto"] = False
            if "do_tomography_full" not in config.keys():
                config["do_tomography_full"] = False
            if config["do_tomography_auto"] and config["do_tomography_full"]:
                config["do_tomography_auto"] = False

            self.config = config

        def _build_config(self, cat):
            """
            rmin=None, rmax=None, nrbins=None, nphibins=None, cache=True, ngal_in_maxres=1, ngal_in_minres=50
            # 1st step: Figure out allowed scales for FFTgrid
            #
            """

            # Estimate number density of galaxies across footprint
            # --> Get smallest pixelization element
            if self.method in ["Mixed", "FFT", "SingleFFT"]:
                pass

            # Figure out allowed scales for FFT grids
            pass

        # Also includes how ns are batched for discrete estimator
        def _build_plan(self, thisconfig, cat, modes, 
                        nthreads, mem_avail):
            finalconfig = deepcopy(thisconfig)
            memreq_base = self._memoryreq(newconfig, modes, do_tomography_auto, do_tomography_full)

            return docompute, finalconfig, catlims, memreq_base, memreq_final

        def _memoryreq(self, ):
            _ = self._gencounters(config, ngals, modes)
            pass

        def _gencounters(self, config, ngals, modes):
            pass


        ##########################################################
        ## Functions that deal with the computation of the 3pcf ##
        ##########################################################
        def compute(self, cat, modes="GGG", do_tomography_auto=False, do_tomography_full=False, nthreads=1, 
                    tofile=False, mem_avail="10G", dry=False, override=False):

            # Compute memory that will be allocated and print options to reduce it
            if self.config is None:
                thisconfig = self._build_config(cat, do_tomography_auto, do_tomography_full, mem_avail)
            else:
                thisconfig = self.config

            planout = self._build_plan(thisconfig, cat, modes, do_tomography_auto, do_tomography_full, nthreads, mem_avail)
            docompute, finalconfig, catlims, memreq_base, memreq_final = planout
            if dry:
                print("Optimized config file requires")
                return finalconfig
            if not docompute:
                if not override:
                    print("Error: Computation will exceed specified available memory (%.2f Gb)"%mem_required)
                    return mem_required, thisconfig
                if override:
                    print("Warning: Computation will exceed specified available memory (%.2f Gb)"%mem_required)

            ## DO COMPUTATION ##  

            ## Allocate FFTNPCF instances for the different resolutions ##
            allinst = {}
            weightmask = {}
            rescale = {}
            gridsizes = {}
            Gncache_FFT = {}
            Gncache_disc = {}
            for npix in config["npix"]:
                if npix != "discrete":
                    cat = WLData(fieldsize, npix, hascat=True, 
                                 pos1=pos1, pos2=pos2, weight=weights, shear=(-1)**flip_1*shape1 + (-1)**flip_2*1J*shape2)
                    cat.cat2grid(do_cic=True)
                    allinst[npix] = FFTNPCF(cat.fieldsize, [cat.npix_x, cat.npix], 
                                            shear=cat.sheargrid, weight=cat.weightgrid)
                    lim_mask = 1e-5
                    weightmask[npix] = (allinst[npix].weight > lim_mask)
                    rescale[npix] = (allinst[npix].npix_y*allinst[npix].npix_x)/np.sum(weightmask[npix])
                    gridsizes[npix] = allinst[npix].weight.shape
                    print(npix, gridsizes[npix], np.sum(weightmask[npix])/cat.npix_x/cat.npix)
            del cat

            ## Initialize counters ##
            nzbins = len(np.unique(cat.zbins))
            nnpix = len(self.config["npix"])

            nradii = 0
            if config["rmin"] is not None and config["nbins_discrete"] is not None:
                nradii += config["nbins_discrete"]
            for _ in config["supperres_rshells"]:
                nradii += len(_)-1
            nradii += config["nbins_baseres"]

            ndata_pix = np.zeros(nnpix, dtype=np.int32)
            cumndata_pix = np.zeros(nnpix+1, dtype=np.int32)
            if config["npix"][0]=="discrete":
                ndata_pix.append(len(pos1))
            for npix in config["npix"]:
                if npix != "discrete":
                    ndata_pix.append(np.sum(weightmask[npix]))
            cumndata_pix[1:np.cumsum(ndata_pix)]

            ## Return corresponding 3pcf object
            if modes=="GGG":
                return GGGCorrelation(thisconfig, bin_edges, bin_centers, threepcf_n, threepcf_n_norm)
            if modes=="NNN":
                return NNNCorrelation(thisconfig, bin_edges, bin_centers, threepcf_n, threepcf_n_norm)

            self.cat = cat.info
            self.hasthreepcf = True
            self.threepcf = None
            self.projection = "X"    


        def _compareGn(self, cat, rmin, rmax, dpix, dipix2=None, n=0, method="CIC"):
            """ Computes Gn using discrete and Grid based estimator. Cat should be PolarCatalog """

            grid = cat.togrid(method)


    class KKKCorrelation(ThreePCF):
        """ ThreePCF of scalar fields (i.e. number counts, conver  """

        def __init__(self, rmin=None, rmax=None, nrbins=None, nphibins=None, 
                     do_tomography_auto=False, do_tomography_full=False, 
                     method="Mixed", cache=True, ngal_in_maxres=1, ngal_in_minres=50, 
                     config=None):
            super().__init__(modes="KKK", rmin=rmin, rmax=rmax, nrbins=nrbins, nphibins=nphibins, 
                             do_tomography_auto=do_tomography_auto, do_tomography_full=do_tomography_full, 
                             method=method, cache=cache, ngal_in_maxres=ngal_in_maxres, ngal_in_minres=ngal_in_minres, 
                             config=config)

        def compute_3pcf(self, cat, nthreads=1, tofile=False, mem_avail="10G", dry=False, override=False):
            self.hasthreepcf = True


    class GKKCorrelation(ThreePCF):
        """ Shear-Lens-Lens (G3L) correlatioin function """

        def __init__(self, rmin=None, rmax=None, nrbins=None, nphibins=None, 
                     do_tomography_auto=False, do_tomography_full=False, 
                     method="Mixed", cache=True, ngal_in_maxres=1, ngal_in_minres=50, 
                     config=None):
            super().__init__(modes="GKK", rmin=rmin, rmax=rmax, nrbins=nrbins, nphibins=nphibins, 
                             do_tomography_auto=do_tomography_auto, do_tomography_full=do_tomography_full, 
                             method=method, cache=cache, ngal_in_maxres=ngal_in_maxres, ngal_in_minres=ngal_in_minres, 
                             config=config)

        def compute_3pcf(self, cat, nthreads=1, tofile=False, mem_avail="10G", dry=False, override=False):
            self.hasthreepcf = True

        def compute_NNMap(self, radii, do_multiscale=False, fname=None):
            pass

    class KGGCorrelation(ThreePCF):
        """ Shear-Lens-Lens (G3L) correlatioin function """

        def __init__(self, rmin=None, rmax=None, nrbins=None, nphibins=None, 
                     do_tomography_auto=False, do_tomography_full=False, 
                     method="Mixed", cache=True, ngal_in_maxres=1, ngal_in_minres=50, 
                     config=None):
            super().__init__(modes="KGG", rmin=rmin, rmax=rmax, nrbins=nrbins, nphibins=nphibins, 
                             do_tomography_auto=do_tomography_auto, do_tomography_full=do_tomography_full, 
                             method=method, cache=cache, ngal_in_maxres=ngal_in_maxres, ngal_in_minres=ngal_in_minres, 
                             config=config)

        def compute_3pcf(self, cat, nthreads=1, tofile=False, mem_avail="10G", dry=False, override=False):
            self.hasthreepcf = True

        def compute_NMapMap(self, radii, do_multiscale=False, fname=None):
            pass

    class GGGCorrelation(ThreePCF):

        def __init__(self, rmin=None, rmax=None, nrbins=None, nmax=None, nphibins=None, 
                     do_tomography_auto=False, do_tomography_full=False, 
                     method="Mixed", cache=True, ngal_in_maxres=1, ngal_in_minres=50, 
                     config=None):
            super().__init__(modes="GGG", rmin=rmin, rmax=rmax, nrbins=nrbins, nphibins=nphibins, 
                             do_tomography_auto=do_tomography_auto, do_tomography_full=do_tomography_full, 
                             method=method, cache=cache, ngal_in_maxres=ngal_in_maxres, ngal_in_minres=ngal_in_minres, 
                             config=config)

            ## Collect which projections we can do right now
            self.projections_avail = ["X", "Centroid"]
            self.project = {}
            for proj in self.projections_avail:
                self.project[proj] = {}
                for proj2 in self.projections_avail:
                    if proj==proj2:
                        self.project[proj][proj2] = lambda: self.threepcf
                    else:
                        self.project[proj][proj2] = None
            # (Add here any newly implemented projections)
            self.project["X"]["Centroid"] = self.x2centroid

        def load(fname):
            pass

        def process(self, cat, nthreads=1, tofile=False, mem_avail="10G", dry=False, override=False):
            self.hasthreepcf = True

        ###################################################################
        ## Functions that deal with integral transformations of the 3pcf ##
        ###################################################################

        def computeMap3(self, radii, do_multiscale=False, tofile=False):
            assert(self.hasthreepcf)
            # Check radii and range of 3pcf and print up to where we would trust it
            # also print if 3pcf is on too small resolution to give stable results
            centroid3pcf = self.project3pcf("Centroid", in_class=False)
            pass

        def computeRingStatistics(self, radii, do_multiscale=False, tofile=False):
            assert(self.hasthreepcf)
            # Check radii and range of 3pcf and print up to where we would trust it
            # also print if 3pcf is on too small resolution to give stable results
            centroid3pcf = self.project3pcf("Centroid", in_class=False)
            pass

        ##################################################################
        ## Functions that deal with different projections of polar 3PCF ##
        ##################################################################
        def project3pcf(self, projection, in_class=False):
            """
            Projects threepcf in a new basis.
            """
            assert(self.hasthreepcf)
            if projection not in self.projections_avail:
                print("Projection %s is not yet supported."%(projection))
                self.gen_3pcfprojections_avail()
                return None

            assert(in_class in [True, False])
            if self.project[self.projection][projection] is not None:
                projected3pcf = self.project[self.projection][projection]()
                if not in_class:
                    return projected3pcf
                else:
                    self.threepcf = projected3pcf
                    self.projection = projection
                    return None
            else:
                print("Projection from %s to %s is not yet implemented."%(self.projection,projection))
                self.gen_3pcfprojections_avail()
                return None


        def gen_3pcfprojections_avail(self):
            print("The following projections for the 3pcf are available:")
            for proj in self.projections_avail:
                for proj2 in self.projections_avail:
                    if self.project[proj][proj2] is not None:
                        print("  %s --> %s"%(proj,proj2))

        def x2centroid(self):
            return self.threepcf


    class XipMixedCovariance(ThreePCF):
        """ ThreePCF of scalar fields (i.e. number counts, conver  """

        def __init__(self, rmin=None, rmax=None, nrbins=None, nphibins=None, 
                     do_tomography_auto=False, do_tomography_full=False, 
                     method="Mixed", cache=True, ngal_in_maxres=1, ngal_in_minres=50, 
                     config=None):
            super().__init__(modes="KKK", rmin=rmin, rmax=rmax, nrbins=nrbins, nphibins=nphibins, 
                             do_tomography_auto=do_tomography_auto, do_tomography_full=do_tomography_full, 
                             method=method, cache=cache, ngal_in_maxres=ngal_in_maxres, ngal_in_minres=ngal_in_minres, 
                             config=config)
            self.hasthreepcf = False
            self.threepcf = None
            self.threepcf_norm = None

        def compute_3pcf(self, cat, nthreads=1, tofile=False, mem_avail="10G", dry=False, override=False):
            self.hasthreepcf = True