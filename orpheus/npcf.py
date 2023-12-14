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
                 nmaxs=30, method="Tree", multicountcorr=True, shuffle_pix=True,
                 tree_resos=[0,0.25,0.5,1.,2.], tree_redges=None, rmin_pixsize=20):
        
        self.order = int(order)
        self.n_cfs = int(n_cfs)
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbinsphi = nbinsphi
        self.nmaxs = nmaxs
        self.method = method
        self.multicountcorr = int(multicountcorr)
        self.shuffle_pix = shuffle_pix
        self.methods_avail = ["Discrete", "Tree", "DoubleTree"]
        self.tree_resos = np.asarray(tree_resos, dtype=np.float64)
        self.tree_nresos = int(len(self.tree_resos))
        self.tree_redges = tree_redges
        self.rmin_pixsize = rmin_pixsize
        self.tree_resosatr = None
        self.bin_centers = None
        self.phis = [None]*self.order
        self.npcf = None
        self.npcf_norm = None
        self.npcf_multipoles = None
        self.npcf_multipoles_norm = None
        
        # Check types or arguments
        if isinstance(self.nbinsphi, int):
            self.nbinsphi = self.nbinsphi*np.ones(order-2).astype(np.int32)
        if isinstance(self.nmaxs, int):
            self.nmaxs = self.nmaxs*np.ones(order-2).astype(np.int32)
        if isinstance(spins, int):
            spins = spins*np.ones(order).astype(np.int32)
        self.spins = np.asarray(spins, dtype=np.int32)
        print(self.spins)
        assert(isinstance(self.order, int))
        assert(isinstance(self.spins, np.ndarray))
        assert(isinstance(self.spins[0], int))
        assert(len(spins)==self.order)
        assert(isinstance(self.n_cfs, int))
        assert(isinstance(self.min_sep, float))
        assert(isinstance(self.max_sep, float))
        assert(isinstance(self.nbinsphi, np.ndarray))
        assert(isinstance(self.nbinsphi[0], int))
        assert(len(self.nbinsphi)==self.order-2)
        assert(isinstance(self.nmaxs, np.ndarray))
        assert(isinstance(self.nmaxs[0], int))
        assert(len(self.nmaxs)==self.order-2)
        assert(self.method in self.methods_avail)
        assert(isinstance(self.tree_resos, np.ndarray))
        assert(isinstance(self.tree_resos[0], float))
        
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
            assert(isinstance(self.tree_redges[0], float))
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
        #target_path = __import__('orpheus').__file__
        #self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        #self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"orpheus_clib"))
        p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(int, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)
        
        ## Third order shear statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([2, 2, 2], dtype=np.int32)):
            # Discrete estimator of third-order shear correlation function
            self.clib.alloc_Gammans_discrete_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_ggg.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Tree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_tree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_tree_ggg.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, 
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
        super().__init__(order=3, spins=[2,2,2], n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X", "Centroid"]
        self.nbinsz = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        self.project["X"]["Centroid"] = self._x2centroid
        
    def process(self, cat, nthreads=16, dotomo=True):
        self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            zbins = np.zeros(cat.ngal, dtype=np.int32)
        else:
            self.nbinsz = cat.nbinsz
            zbins = cat.zbins
        sc = (4,self.nmax+1,self.nbinsz*self.nbinsz*self.nbinsz,self.nbinsr,self.nbinsr)
        sn = (self.nmax+1,self.nbinsz*self.nbinsz*self.nbinsz,self.nbinsr,self.nbinsr)
        szr = (self.nbinsz, self.nbinsr)
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        threepcfs_n = np.zeros(4*(self.nmax+1)*self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr).astype(complex)
        threepcfsnorm_n = np.zeros((self.nmax+1)*self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr).astype(complex)
        args_basecat = (cat.isinner.astype(np.int32), cat.weight, cat.pos1, cat.pos2, cat.tracer_1, cat.tracer_2, 
                        zbins.astype(np.int32), int(self.nbinsz), int(cat.ngal), )
        args_basesetup = (int(0), int(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
                          int(self.nbinsr), int(self.multicountcorr), )
        if self.method=="Discrete":
            if not cat.hasspatialhash:
                cat.build_spatialhash(dpix=max(1.,self.max_sep//10.))
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), int(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), int(cat.pix2_n), )
            args = (*args_basecat,
                    *args_basesetup,
                    cat.index_matcher,
                    cat.pixs_galind_bounds, 
                    cat.pix_gals,
                    *args_pixgrid,
                    int(nthreads),
                    bin_centers,
                    threepcfs_n,
                    threepcfsnorm_n)
            func = self.clib.alloc_Gammans_discrete_ggg
            for elarg, arg in enumerate(args):
                print(elarg, arg)
                print(arg.dtype, func.argtypes[elarg])
        elif self.method=="Tree":
            cutfirst = int(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:],tomo=dotomo,shuffle=self.shuffle_pix)
            ngal_resos, pos1s, pos2s, weights, zbins, allfields, index_matchers, pixs_galind_bounds, pix_gals = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
            e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
            index_matcher = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals = np.concatenate(pix_gals).astype(np.int32)
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), int(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), int(cat.pix2_n), )
            args = (*args_basecat,
                    self.tree_nresos,
                    self.tree_redges,
                    np.array(ngal_resos, dtype=np.int32),
                    weight_resos,
                    pos1_resos,
                    pos2_resos,
                    e1_resos,
                    e2_resos,
                    zbin_resos,
                    index_matcher,
                    pixs_galind_bounds,
                    pix_gals,
                    *args_pixgrid,
                    *args_basesetup,
                    int(nthreads),
                    bin_centers,
                    threepcfs_n,
                    threepcfsnorm_n)
            func = self.clib.alloc_Gammans_tree_ggg
        elif self.method=="DoubleTree":
            raise NotImplementedError 
            
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.npcf_multipoles = threepcfs_n.reshape(sc)
        self.npcf_multipoles_norm = threepcfsnorm_n.reshape(sn)
        self.projection = "X"
    
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
                tmp = np.zeros((nzcombis, rbins, rbins), dtype=complex)
                N0 = 1./(2*np.pi) * self.npcf_multipoles_norm[0].astype(complex)
                for eln,n in enumerate(range(self.nmax+1)):
                    _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                    tmp += _const * self.npcf_multipoles[elm,eln].astype(complex)
                    if n>0:
                        tmp += _const.conj() * self.npcf_multipoles[conjmap[elm],eln][ztiler].astype(complex).transpose(0,2,1)
                self.npcf[elm,...,elphi] = tmp/N0.real
        # Number of triangles
        for elphi, phi in enumerate(self.phi):
            tmptotnorm = np.zeros((nzcombis, rbins, rbins), dtype=complex)
            for eln,n in enumerate(range(self.nmax+1)):
                _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                tmptotnorm += _const * self.npcf_multipoles_norm[eln].astype(complex)
                if n>0:
                    tmptotnorm += _const.conj() * self.npcf_multipoles_norm[eln][ztiler].astype(complex).transpose(0,2,1)
                self.npcf_norm[...,elphi] = tmptotnorm
    
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
        
        
    def computeMap3(self, radii, do_multiscale=False, tofile=False):
        """
        Compute third-order aperture statistics
        """
        thisproj = self.projection
        if thisproj != "Centroid":
            self.toprojection("Centroid")
        # Compute Map3...
        map3 = None
        if thisproj != "Centroid":
            self.toprojection(thisproj)
            
        if tofile:
            # Write to file
            pass
            
        return map3

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
        
    def process(self, cat_lens, cat_source, nthreads=16, dotomo=True):
        self._checkcats([cat_lens, cat_source], [0, 2])
        if not dotomo:
            self.nbinsz_lens = 1
            self.nbinsz_source = 1
            zbins_lens = np.zeros(cat_lens.ngal, dtype=np.int32)
            zbins_source = np.zeros(cat_source.ngal, dtype=np.int32)
        else:
            self.nbinsz_lens = cat_lens.nbinsz
            self.nbinsz_source = cat_source.nbinsz
        _z3combis = self.nbinsz_source*self.nbinsz_lens*self.nbinsz_lens
        _r2combis = self.nbinsr*self.nbinsr
        sc = (4,self.nmax+1, _z3combis, self.nbinsr,self.nbinsr)
        sn = (self.nmax+1, _z3combis, self.nbinsr,self.nbinsr)
        szr = (self.nbinsz_lens, self.nbinsz_source, self.nbinsr)
        bin_centers = np.zeros(self.nbinsz_source*self.nbinsz_lens*self.nbinsr).astype(np.float64)
        threepcfs_n = np.zeros(4*(self.nmax+1)*_z3combis*_r2combis).astype(complex)
        threepcfsnorm_n = np.zeros((self.nmax+1)*_z3combis*_r2combis).astype(complex)
        args_basecat = (cat.isinner.astype(np.int32), cat.weight, cat.pos1, cat.pos2, cat.tracer_1, cat.tracer_2, 
                        zbins.astype(np.int32), int(self.nbinsz), int(cat.ngal), )
        args_basesetup = (int(0), int(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
                          int(self.nbinsr), int(self.multicountcorr), )
    
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
        self.nsubbins = int(nsubbins)
        nbinsr = nsubbins*nbins_xi
        super().__init__(order=3, spins=[0,0,0], n_cfs=1, min_sep=min_sep_xi, max_sep=max_sep_xi, nbinsr=nbinsr, 
                         nmaxs=nmax, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        
    def process(self, cat, nthreads=16, dotomo=True):
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
        elif self.method=="Tree":
            cutfirst = int(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:],tomo=dotomo,shuffle=self.shuffle_pix)
            ngal_resos, pos1s, pos2s, weights, zbins, _, index_matchers, pixs_galind_bounds, pix_gals = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            index_matcher = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals = np.concatenate(pix_gals).astype(np.int32)
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), int(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), int(cat.pix2_n), )

            args = (*args_basecat,
                    self.tree_nresos,
                    self.tree_redges,
                    np.array(ngal_resos, dtype=np.int32),
                    weight_resos,
                    pos1_resos,
                    pos2_resos,
                    zbin_resos,
                    index_matcher,
                    pixs_galind_bounds,
                    pix_gals,
                    *args_pixgrid,
                    *args_basesetup,
                    int(nthreads),
                    bin_centers,
                    wwcounts,
                    w2wcounts,
                    w2wwtriplets,
                    wwwtriplets)
            func = self.clib.alloc_triplets_tree_xipxipcov
        elif self.method=="DoubleTree":
            raise NotImplementedError 
        print(args[0])
        print(args[1])
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.wwcounts = wwcounts.reshape(szzr)
        self.w2wcounts = w2wcounts.reshape(szzr)
        self.npcf_multipoles = w2wwtriplets.reshape(sc)
        self.npcf_multipoles_norm = wwwtriplets.reshape(sn)
        self.projection = None 
        
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
                tmp = np.zeros((nzcombis, rbins, rbins), dtype=complex)
                N0 = 1./(2*np.pi) * self.npcf_multipoles_norm[0].astype(complex)
                for eln,n in enumerate(range(self.nmax+1)):
                    _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                    tmp += _const * self.npcf_multipoles[elm,eln].astype(complex)
                    if n>0:
                        tmp += _const.conj() * self.npcf_multipoles[conjmap[elm],eln][ztiler].astype(complex).transpose(0,2,1)
                self.npcf[elm,...,elphi] = tmp/N0.real
        # Number of triangles
        for elphi, phi in enumerate(self.phi):
            tmp = np.zeros((1,nzcombis, rbins, rbins), dtype=complex)
            tmpnorm = np.zeros((nzcombis, rbins, rbins), dtype=complex)
            for eln,n in enumerate(range(self.nmax+1)):
                _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                tmp += _const * self.npcf_multipoles_norm[:,eln].astype(complex)
                tmpnorm += _const * self.npcf_multipoles[:,eln].astype(complex)
                if n>0:
                    tmp += _const.conj() * self.npcf_multipoles[0,eln][ztiler].astype(complex).transpose(0,2,1)
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