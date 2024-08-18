from abc import ABC, abstractmethod

import ctypes as ct
from copy import deepcopy
from functools import reduce
from itertools import accumulate, product, chain
import glob
from numba import jit, prange
from numba import config as nb_config
from numba import complex128 as nb_complex128
import numpy as np 
from numpy.ctypeslib import ndpointer
import operator
from pathlib import Path
import sys
from .catalog import Catalog, ScalarTracerCatalog, SpinTracerCatalog, MultiTracerCatalog
from .utils import flatlist, split_array_into_batches, get_site_packages_dir, search_file_in_site_package
from .exceptions import *


__all__ = ["BinnedNPCF", 
           "GGCorrelation",
           "GGGCorrelation", "FFFCorrelation", "GNNCorrelation", "NGGCorrelation",
           "GGGGCorrelation_NoTomo", "GGGGCorrelation", "XipmMixedCovariance"]

################################################
## BASE CLASSES FOR NPCF AND THEIR MULTIPOLES ##
################################################        
class BinnedNPCF:
    
    
    def __init__(self, order, spins, n_cfs, min_sep, max_sep, nbinsr=None, binsize=None, nbinsphi=100, 
                 nmaxs=30, method="DoubleTree", multicountcorr=True, diagrenorm=False, shuffle_pix=1,
                 tree_resos=[0,0.25,0.5,1.,2.], tree_redges=None, rmin_pixsize=20, 
                 resoshift_leafs=0, minresoind_leaf=None, maxresoind_leaf=None,  nthreads=16):
        r"""Class of an binned N-point correlation function of various arbitrary tracer catalogs. 
        This class contains attributes and metods that can be used across any its children.
        
        Attributes
        ----------
        order: int
            The order of the correlation function.
        spins: list
            The spins of the tracer fields of which the NPCF is computed. 
        n_cfs: int
            The number of independent components of the NPCF.
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
        diagrenorm: bool, optional
            Deprecated
        shuffle_pix: int, optional
            Choice of how to define centers of the cells in the spatial hash structure.
            Defaults to ``1``, i.e. random positioning.
        tree_resos: list, optional
            The cell sizes of the hierarchical spatial hash structure
        tree_edges: list, optional
            Deprecated(?)
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
        self.methods_avail = ["Discrete", "Tree", "BaseTree", "DoubleTree"]
        self.tree_resos = np.asarray(tree_resos, dtype=np.float64)
        self.tree_nresos = int(len(self.tree_resos))
        self.tree_redges = tree_redges
        self.rmin_pixsize = rmin_pixsize
        self.resoshift_leafs = resoshift_leafs
        self.minresoind_leaf = minresoind_leaf
        self.maxresoind_leaf = maxresoind_leaf
        self.nthreads = np.int32(max(1,nthreads))
        
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
        if self.order>2:
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
        
        ## Second order scalar-scalar statistics ##
        if self.order==2 and np.array_equal(self.spins, np.array([0, 0], dtype=np.int32)):
            self.clib.alloc_NNcounts_doubletree.restype = ct.c_void_p
            self.clib.alloc_NNcounts_doubletree.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32, ct.c_int32, p_i32, p_f64, p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.int64)] 
        
        ## Second order shear-shear statistics ##
        if self.order==2 and np.array_equal(self.spins, np.array([2, 2], dtype=np.int32)):
            # Doubletree-based estimator of second-order shear correlation function
            self.clib.alloc_xipm_doubletree.restype = ct.c_void_p
            self.clib.alloc_xipm_doubletree.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32, ct.c_int32, p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.int64)] 
                
        ## Third order shear-shear-shear statistics ##
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
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Basetree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_basetree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_basetree_ggg.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, p_i32, ct.c_int32, 
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Doubletree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_doubletree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_ggg.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32, ct.c_int32, p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
        ## Third-order source-lens-lens statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([2, 0, 0], dtype=np.int32)):
            # Discrete estimator of third-order source-lens-lens (G3L) correlation function
            self.clib.alloc_Gammans_discrete_GNN.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_GNN.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Doubletree-based estimator of third-order source-lens-lens (G3L) correlation function
            self.clib.alloc_Gammans_doubletree_GNN.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_GNN.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                p_i32, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
        ## Third-order lens-source-source statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([0, 2, 2], dtype=np.int32)):
            # Discrete estimator of third-order lens-source-source correlation function
            self.clib.alloc_Gammans_discrete_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_NGG.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                p_i32, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
                        
            self.clib.alloc_Gammans_tree_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_tree_NGG.argtypes = [
                ct.c_int32, p_f64, 
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, p_i32,
                p_i32, p_f64, p_f64, p_f64,  p_i32, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)]             
        
            self.clib.alloc_Gammans_doubletree_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_NGG.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                p_i32, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
        ## Misc third-order statistics ##
        if self.order==3:
            
            # Discrete estimator of mixed term of the covariance of the shear 2pcf
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
            
            # Basetree-based estimator of mixed term of the covariance of the shear 2pcf
            self.clib.alloc_triplets_basetree_xipxipcov.restype = ct.c_void_p
            self.clib.alloc_triplets_basetree_xipxipcov.argtypes = [
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
            
            # Doubletree-based estimator of mixed term of the covariance of the shear 2pcf
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
            
        ## Fourth-order shear-shear-shear-shear statistics ##
        if self.order==4 and np.array_equal(self.spins, np.array([2, 2, 2, 2], dtype=np.int32)):
            
            # Discrete estimator of non-tomographic fourth-order shear correlation function
            self.clib.alloc_notomoGammans_discrete_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_discrete_gggg.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Discrete estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoMap4_disc_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_disc_gggg.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_f64, p_f64, ct.c_int32,
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_int32, ct.c_int32, p_f64, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128),
                ct.c_int32, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # Tree-based estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoMap4_tree_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_tree_gggg.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, ct.c_int32, p_f64, p_f64, ct.c_int32,
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_int32, ct.c_int32, p_f64, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128),
                ct.c_int32, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            self.clib.multipoles2npcf_gggg.restype = ct.c_void_p
            self.clib.multipoles2npcf_gggg.argtypes = [
                p_c128, p_c128, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_int32, ct.c_int32, p_f64, ct.c_int32, p_f64, ct.c_int32, 
                ct.c_int32, p_c128, p_c128]
                        
            # Transformation between 4PCF from multipole-basis tp real-space basis for a fixed
            # combination of radial bins
            self.clib.multipoles2npcf_gggg_singletheta.restype = ct.c_void_p
            self.clib.multipoles2npcf_gggg_singletheta.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_double, 
                p_f64, p_f64, ct.c_int32, ct.c_int32, 
                ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # Reconstruction of all 4pcf multipoles from symmetry properties given a set of
            # multipoles with theta1<=theta2<=theta3
            self.clib.getMultiplolesFromSymm.restype = ct.c_void_p
            self.clib.getMultiplolesFromSymm.argtypes = [
                p_c128, p_c128,
                ct.c_int32, ct.c_int32, p_i32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
                        
            # [DEBUG]: Shear 4pt function in terms of xip/xim
            self.clib.gauss4pcf_analytic.restype = ct.c_void_p
            self.clib.gauss4pcf_analytic.argtypes = [
                ct.c_double, ct.c_double, ct.c_double, p_f64, ct.c_int32,
                p_f64, p_f64, ct.c_double, ct.c_double, ct.c_double, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # [DEBUG]: Map4 via analytic gaussian 4pcf
            self.clib.alloc_notomoMap4_analytic.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_analytic.argtypes = [
                ct.c_double, ct.c_double, ct.c_int32, p_f64, p_f64, ct.c_int32,
                p_i32, p_i32, p_i32, ct.c_int32,
                ct.c_int32, p_f64, ct.c_int32,
                p_f64, p_f64, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # [DEBUG]: Map4 filter function for single combination
            self.clib.filter_Map4.restype = ct.c_void_p
            self.clib.filter_Map4.argtypes = [
                ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # [DEBUG]: Conversion between 4pcf and Map4 for (theta1,theta2,theta3) subset
            self.clib.fourpcf2M4correlators_parallel.restype = ct.c_void_p
            self.clib.fourpcf2M4correlators_parallel.argtypes = [
                ct.c_int32,
                ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, 
                ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.complex128), np.ctypeslib.ndpointer(dtype=np.complex128)] 
                        
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
            print(f"Projection {projection} is not yet supported.")
            self._print_npcfprojections_avail()
            return 

        projection_func = child.project[child.projection].get(projection)
        if projection_func is not None:
            child.npcf = projection_func()
            child.projection = projection
        else:
            print(f"Projection from {child.projection} to {projection} is not yet implemented.")
            self._print_npcfprojections_avail(child)
                    
    def _print_npcfprojections_avail(self, child):
        print(f"The following projections are available in the class {child.__class__.__name__}:")
        for proj in child.projections_avail:
            for proj2 in child.projections_avail:
                if child.project[proj].get(proj2) is not None:
                    print(f"  {proj} --> {proj2}")
 
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
            
            
    """
    # TODO: Put stuff from __init__ in here
    def _update_tree(self, new_resos, new_edges):
        
        new_resos = np.asarray(new_resos, dtype=np.float64)
        new_edges = np.asarray(new_edges, dtype=np.float64)
        nresos = int(len(new_edges))
        assert(isinstance(new_resos, np.ndarray))
        assert(isinstance(new_resos[0], np.float64))
        
        self.tree_resos = new_resos
        self.tree_nresos = int(len(self.tree_resos))
        self.tree_redges = tree_redges
        
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
    """

###############################   
## SECOND - ORDER STATISTICS ##
###############################
class NNCorrelation(BinnedNPCF):
    """ Just need this in order to run this on overlapping patches (which treecorr does not do afaik).
    Similar speed as treecorr and always much faster as compared to the higher-order functions
    """
    def __init__(self, min_sep, max_sep, **kwargs):
        super().__init__(order=2, spins=np.array([2,2], dtype=np.int32), n_cfs=2, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        self.nzcombis = None
        self.counts = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
    
    def process(self, cat, dotomo=True, do_dc=False, apply_edge_correction=False, adjust_tree=False):
        
        self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            old_zbins = cat.zbins[:]
            cat.zbins = np.zeros(cat.ngal, dtype=np.int32)
            self.nzcombis = 1
        else:
            self.nbinsz = cat.nbinsz
            zbins = cat.zbins
            self.nzcombis = self.nbinsz*self.nbinsz
                    
        z2r = self.nbinsz*self.nbinsz*self.nbinsr
        sz2r = (self.nbinsz*self.nbinsz, self.nbinsr)
        bin_centers = np.zeros(z2r).astype(np.float64)
        counts = np.zeros(z2r).astype(np.float64)
        npair = np.zeros(z2r).astype(np.int64)
        
        args_basesetup = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr),  )
        
        cutfirst = np.int32(self.tree_resos[0]==0.)
        mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                              shuffle=self.shuffle_pix, w2field=False, normed=True)
        ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = mhash
        weight_resos = np.concatenate(weights).astype(np.float64)
        pos1_resos = np.concatenate(pos1s).astype(np.float64)
        pos2_resos = np.concatenate(pos2s).astype(np.float64)
        zbin_resos = np.concatenate(zbins).astype(np.int32)
        isinner_resos = np.concatenate(isinners).astype(np.int32)
        scalar_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
        index_matcher = np.concatenate(index_matchers).astype(np.int32)
        pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
        pix_gals = np.concatenate(pix_gals).astype(np.int32)
        index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
        nregions = len(index_matcher_flat)
        
        args_treeresos = (np.int32(self.tree_nresos), np.int32(self.tree_nresos-cutfirst),
                          dpixs1_true.astype(np.float64), dpixs2_true.astype(np.float64), self.tree_redges, 
                          np.int32(self.resoshift_leafs), np.int32(self.minresoind_leaf), 
                          np.int32(self.maxresoind_leaf), np.array(ngal_resos, dtype=np.int32), )
        args_resos = (isinner_resos, weight_resos, pos1_resos, pos2_resos, scalar_resos, zbin_resos,
                      index_matcher, pixs_galind_bounds, pix_gals, )
        args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                     np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), 
                     np.int32(nregions), index_matcher_flat.astype(np.int32),)
        args_binning = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), np.int32(do_dc))
        args_output = (bin_centers, counts, npair, )
        func = self.clib.alloc_xipm_doubletree
        args = (*args_treeresos,
                np.int32(self.nbinsz),
                *args_resos,
                *args_hash,
                *args_binning,
                np.int32(self.nthreads),
                *args_output)

        func(*args)
        
        self.bin_centers = bin_centers.reshape(sz2r)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.npair = npair.reshape(sz2r)
        self.counts = norm.reshape(sz2r)
        
        if not dotomo:
            cat.zbins = old_zbins
            
        
    def computeNap2(self, radii, tofile=False):
        
        Tp = lambda x: 1./128. * (x**4-16*x**2+32) * np.exp(-x**2/4.)  
        Tm = lambda x: 1./128. * (x**4) * np.exp(-x**2/4.)  
        result = np.zeros((4, self.nzcombis, len(radii)), dtype=float)
        for elr, R in enumerate(radii):
            thetared = self.bin_centers/R
            pref = self.binsize*thetared**2/2.
            t1 = np.sum(pref*(Tp(thetared)*self.xip + Tm(thetared)*self.xim), axis=1)
            t2 = np.sum(pref*(Tp(thetared)*self.xip - Tm(thetared)*self.xim), axis=1)
            result[0,:,elr] =  t1.real  # Map2
            result[1,:,elr] =  t1.imag  # MapMx 
            result[2,:,elr] =  t2.real  # Mx2
            result[3,:,elr] =  t2.imag  # MxMap (Difference from MapMx gives ~level of estimator uncertainty)
            
        return result
    
class GGCorrelation(BinnedNPCF):
    """ Just need this in order to run this on overlapping patches (which treecorr does not do afaik).
    Similar speed as treecorr and always much faster as compared to the higher-order functions
    """
    def __init__(self, min_sep, max_sep, **kwargs):
        super().__init__(order=2, spins=np.array([2,2], dtype=np.int32), n_cfs=2, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        self.nzcombis = None
        self.counts = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
    
    def process(self, cat, dotomo=True, do_dc=False, apply_edge_correction=False, adjust_tree=False):
        
        self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            old_zbins = cat.zbins[:]
            cat.zbins = np.zeros(cat.ngal, dtype=np.int32)
            self.nzcombis = 1
        else:
            self.nbinsz = cat.nbinsz
            zbins = cat.zbins
            self.nzcombis = self.nbinsz*self.nbinsz
        
        z2r = self.nbinsz*self.nbinsz*self.nbinsr
        sz2r = (self.nbinsz*self.nbinsz, self.nbinsr)
        bin_centers = np.zeros(z2r).astype(np.float64)
        xip = np.zeros(z2r).astype(np.complex128)
        xim = np.zeros(z2r).astype(np.complex128)
        norm = np.zeros(z2r).astype(np.float64)
        npair = np.zeros(z2r).astype(np.int64)
        
        args_basesetup = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr),  )
        
        cutfirst = np.int32(self.tree_resos[0]==0.)
        mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                              shuffle=self.shuffle_pix, w2field=True, normed=True)
        ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = mhash
        weight_resos = np.concatenate(weights).astype(np.float64)
        pos1_resos = np.concatenate(pos1s).astype(np.float64)
        pos2_resos = np.concatenate(pos2s).astype(np.float64)
        zbin_resos = np.concatenate(zbins).astype(np.int32)
        isinner_resos = np.concatenate(isinners).astype(np.int32)
        e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
        e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
        index_matcher = np.concatenate(index_matchers).astype(np.int32)
        pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
        pix_gals = np.concatenate(pix_gals).astype(np.int32)
        index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
        nregions = len(index_matcher_flat)
        
        args_treeresos = (np.int32(self.tree_nresos), np.int32(self.tree_nresos-cutfirst),
                          dpixs1_true.astype(np.float64), dpixs2_true.astype(np.float64), self.tree_redges, 
                          np.int32(self.resoshift_leafs), np.int32(self.minresoind_leaf), 
                          np.int32(self.maxresoind_leaf), np.array(ngal_resos, dtype=np.int32), )
        args_resos = (isinner_resos, weight_resos, pos1_resos, pos2_resos, e1_resos, e2_resos, zbin_resos,
                      index_matcher, pixs_galind_bounds, pix_gals, )
        args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                     np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), 
                     np.int32(nregions), index_matcher_flat.astype(np.int32),)
        args_binning = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), np.int32(do_dc))
        args_output = (bin_centers, xip, xim, norm, npair, )
        func = self.clib.alloc_xipm_doubletree
        args = (*args_treeresos,
                np.int32(self.nbinsz),
                *args_resos,
                *args_hash,
                *args_binning,
                np.int32(self.nthreads),
                *args_output)

        func(*args)
        
        self.bin_centers = bin_centers.reshape(sz2r)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.npair = npair.reshape(sz2r)
        self.norm = norm.reshape(sz2r)
        self.xip = xip.reshape(sz2r)
        self.xim = xim.reshape(sz2r)
        self.projection = "xipm"
        
        if not dotomo:
            cat.zbins = old_zbins
            
        
    def computeMap2(self, radii, tofile=False):
        
        Tp = lambda x: 1./128. * (x**4-16*x**2+32) * np.exp(-x**2/4.)  
        Tm = lambda x: 1./128. * (x**4) * np.exp(-x**2/4.)  
        result = np.zeros((4, self.nzcombis, len(radii)), dtype=float)
        for elr, R in enumerate(radii):
            thetared = self.bin_centers/R
            pref = self.binsize*thetared**2/2.
            t1 = np.sum(pref*(Tp(thetared)*self.xip + Tm(thetared)*self.xim), axis=1)
            t2 = np.sum(pref*(Tp(thetared)*self.xip - Tm(thetared)*self.xim), axis=1)
            result[0,:,elr] =  t1.real  # Map2
            result[1,:,elr] =  t1.imag  # MapMx 
            result[2,:,elr] =  t2.real  # Mx2
            result[3,:,elr] =  t2.imag  # MxMap (Difference from MapMx gives ~level of estimator uncertainty)
            
        return result
        
        

##############################
## THIRD - ORDER STATISTICS ##
##############################
class GGGCorrelation(BinnedNPCF):
    
    def __init__(self, n_cfs, min_sep, max_sep, **kwargs):
        r""" Class containing methods to measure and and obtain statistics that are built
        from third-order shear correlation functions.
        
        Attributes
        ----------
        n_cfs: int
            The number of independent components of the NPCF.
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
        diagrenorm: bool, optional
            Deprecated
        shuffle_pix: int, optional
            Choice of how to define centers of the cells in the spatial hash structure.
            Defaults to ``1``, i.e. random positioning.
        tree_resos: list, optional
            The cell sizes of the hierarchical spatial hash structure
        tree_edges: list, optional
            Deprecated(?)
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
        
    def process(self, cat, dotomo=True, apply_edge_correction=False, adjust_tree=False):
        self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            zbins = np.zeros(cat.ngal, dtype=np.int32)
            self.nzcombis = 1
        else:
            self.nbinsz = cat.nbinsz
            zbins = cat.zbins
            self.nzcombis = self.nbinsz*self.nbinsz*self.nbinsz
        if adjust_tree:
            nbar = cat.ngal/(cat.len1*cat.len2)
            
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
                    np.int32(self.nthreads),
                    bin_centers,
                    threepcfs_n,
                    threepcfsnorm_n)
            func = self.clib.alloc_Gammans_discrete_ggg
        elif self.method in ["Tree", "BaseTree", "DoubleTree"]:
            #print("Doing multihash")
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                  shuffle=self.shuffle_pix, w2field=True, normed=True)
            ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            isinner_resos = np.concatenate(isinners).astype(np.int32)
            e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
            e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
            _weightsq_resos = np.concatenate([allfields[i][2] for i in range(len(allfields))]).astype(np.float64)
            weightsq_resos = _weightsq_resos*weight_resos # As in reduce we renorm all the fields --> need to `unrenorm'
            index_matcher = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals = np.concatenate(pix_gals).astype(np.int32)
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )
            args_resos = (weight_resos, pos1_resos, pos2_resos, e1_resos, e2_resos, zbin_resos, weightsq_resos,
                          index_matcher, pixs_galind_bounds, pix_gals, )
            args_output = (bin_centers, threepcfs_n, threepcfsnorm_n, )
            #print("Doing %s"%self.method)
            if self.method=="Tree":
                args = (*args_basecat,
                        np.int32(self.tree_nresos),
                        self.tree_redges,
                        np.array(ngal_resos, dtype=np.int32),
                        *args_resos,
                        *args_pixgrid,
                        *args_basesetup,
                        np.int32(self.nthreads),
                        *args_output)
                func = self.clib.alloc_Gammans_tree_ggg
            if self.method in ["BaseTree", "DoubleTree"]:
                args_resos = (isinner_resos, ) + args_resos
                index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
                nregions = len(index_matcher_flat)
                args_basesetup_dtree = (np.int32(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), 
                                        np.int32(self.nbinsr), np.int32(self.multicountcorr), )
                args_treeresos = (np.int32(self.tree_nresos), np.int32(self.tree_nresos-cutfirst),
                                  dpixs1_true.astype(np.float64), dpixs2_true.astype(np.float64), self.tree_redges, )
                if self.method=="BaseTree":
                    func = self.clib.alloc_Gammans_basetree_ggg
                if self.method=="DoubleTree":
                    args_leafs = (np.int32(self.resoshift_leafs), np.int32(self.minresoind_leaf), 
                                  np.int32(self.maxresoind_leaf), )
                    args_treeresos = args_treeresos + args_leafs
                    func = self.clib.alloc_Gammans_doubletree_ggg
                args = (*args_treeresos,
                        np.array(ngal_resos, dtype=np.int32),
                        np.int32(self.nbinsz),
                        *args_resos,
                        *args_pixgrid,
                        np.int32(nregions),
                        index_matcher_flat.astype(np.int32),
                        *args_basesetup_dtree,
                        np.int32(self.nthreads),
                        *args_output)
        
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.npcf_multipoles = threepcfs_n.reshape(sc)
        self.npcf_multipoles_norm = threepcfsnorm_n.reshape(sn)
        self.projection = "X"
                
        if apply_edge_correction:
            self.edge_correction()
            
        # Approximate way to normalise  the main diagonal, as well as the super- 
        # and subdiagonal of the normalization multipoles. This method only works 
        # if some bits of the Nn are computed using the discrete estimator.
        if self.diagrenorm:
            self._normalize_Nndiags()     
            
            
    def _normalize_Nndiags(self):
        
        def kth_diag_indices(a, k):
            rows, cols = np.diag_indices_from(a)
            if k < 0:
                return rows[-k:], cols[:k]
            elif k > 0:
                return rows[:-k], cols[k:]
            else:
                return rows, cols

        for elz in range(self.npcf_multipoles_norm.shape[1]):
            for eln in range(1,self.npcf_multipoles_norm.shape[0]):
                if eln==1 and elz==0:
                    thisrows, thiscols = kth_diag_indices(self.npcf_multipoles_norm[eln,elz], k=0)
                    thisrowsp, thiscolsp = kth_diag_indices(self.npcf_multipoles_norm[eln,elz], k=1)
                    thisrowsm, thiscolsm = kth_diag_indices(self.npcf_multipoles_norm[eln,elz], k=-1)
                rindmax = np.argwhere(self.bin_edges[1:]<self.tree_redges[1])[-1][0]
                maindiag = np.diag(np.abs(self.npcf_multipoles_norm[eln,elzcombi])[:rindmax])
                super1 = np.diag(np.abs(self.npcf_multipoles_norm[eln,elzcombi]),k=1)[:rindmax-1]
                super2 = np.diag(np.abs(self.npcf_multipoles_norm[eln,elzcombi]),k=2)[:rindmax-2]
                ratio_m1 = np.sum(maindiag[1:])/np.sum(super1)
                ratio_s1s2 = np.sum(super1[1:])/np.sum(super2)
                modified_super1 = ratio_s1s2*np.diag(np.abs(self.npcf_multipoles_norm[eln,elzcombi]),k=2)[rindmax-2:]
                modified_main = ratio_m1*modified_super1
                renormed_main = np.append(maindiag, modified_main)
                renormed_super = np.append(super1, modified_super1)
                self.npcf_multipoles_norm[eln,elz][thisrows,thiscols] = renormed_main
                self.npcf_multipoles_norm[eln,elz][thisrowsp,thiscolsp] = renormed_super
                self.npcf_multipoles_norm[eln,elz][thisrowsm,thiscolsm] = renormed_super    
        
    def edge_correction(self, ret_matrices=False):
        
        def gen_M_matrix(thet1,thet2,threepcf_n_norm):
            nvals, ntheta, _ = threepcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=np.int)
            nextM = np.zeros((nvals,nvals))
            for ind, ell in enumerate(narr):
                lminusn = ell-narr
                sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
                nextM[ind,sel] = threepcf_n_norm[(lminusn+nmax)[sel],thet1,thet2].real / threepcf_n_norm[nmax,thet1,thet2].real
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
                    if not do_multiscale:
                        R2 = R1
                        R3 = R1
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
        return self.__map3_filtergrid_singleR(R1, R2, R3, self.bin_edges, self.bin_centers_mean, self.phi)
    
    @staticmethod
    @jit(nopython=True)
    def __map3_filtergrid_singleR(R1, R2, R3, normys_edges, normys_centers, phis):
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
        T0 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
        T3_123 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
        T3_231 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
        T3_312 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
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
    
    def __map3_filtergrid_multiR(self, R1, R2, R3):
        return self.__map3_filtergrid_multiR(self, R1, R2, R3, self.bin_edges, self.bin_centers_mean, self.phi, include_measure=True)
    
    @staticmethod
    @jit(nopython=True)
    def __map3_filtergrid_multiR(self, R1, R2, R3, normys_edges, normys_centers, phis, include_measure=True):
        nbinsr = len(normys_centers)
        nbinsphi = len(phis)
        _cphis = np.cos(phis)
        _c2phis = np.cos(2*phis)
        _sphis = np.sin(phis)
        _ephis = np.e**(1J*phis)
        _ephisc = np.e**(-1J*phis)
        _e2phis = np.e**(2J*phis)
        _e2phisc = np.e**(-2J*phis)
        T0 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
        T3_123 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
        T3_231 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
        T3_312 = np.zeros((nbinsr, nbinsr, nbinsphi), dtype=nb_complex128)
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
    def __init__(self, min_sep, max_sep, zweighting=False, zweighting_sigma=None, **kwargs):
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
        diagrenorm: bool, optional
            Deprecated
        shuffle_pix: int, optional
            Choice of how to define centers of the cells in the spatial hash structure.
            Defaults to ``1``, i.e. random positioning.
        tree_resos: list, optional
            The cell sizes of the hierarchical spatial hash structure
        tree_edges: list, optional
            Deprecated(?)
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
        super().__init__(3, [2,0,0], n_cfs=1, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X"]
        self.nbinsz_source = None
        self.nbinsz_lens = None
        
        assert(zweighting in [True, False])
        self.zweighting = zweighting
        self.zweighting_sigma = zweighting_sigma
        if not self.zweighting :
            self.zweighting_sigma = None
        else:
            assert(isinstance(self.zweighting_sigma, float))
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        
    # TODO: Include z-weighting in estimator 
    # * False --> No z-weighting, nothing to do
    # * True  --> Tomographic zweighting: Use effective weight for each tomo bin combi. Do computation as tomo case with
    #             no z-weighting and then weight in postprocessing where (zs, zl1, zl2) --> w_{zl1, zl2} * (zs)
    #             As this could be many zbins, might want to only allow certain zcombis -- i.e. neighbouring zbins.
    #             Functional form similar to https://arxiv.org/pdf/1909.06190.pdf 
    # * Note that for spectroscopic catalogs we cannot do a full spectroscopic weighting as done i.e. the brute-force method 
    #   in https://arxiv.org/pdf/1909.06190.pdf, as this breaks the multipole decomposition.
    # * In general, think about what could be a consistent way get a good compromise between speed vs S/N. One extreme would 
    #   be just to use some broad bins and and the std within them (so 'thinner' bins have more weight). Other extreme would 
    #   be many small zbins with proper cross-weighting and maximum distance --> Becomes less efficient for more bins.
    def process(self, cat_source, cat_lens, nthreads=16, dotomo_source=True, dotomo_lens=True, apply_edge_correction=False):
        self._checkcats([cat_source, cat_lens, cat_lens], [2, 0, 0])
        
        if not dotomo_lens and self.zweighting:
            print("Redshift-weighting requires tomographic computation for the lenses.")
            dotomo_lens = True
            
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
            
        if self.zweighting:
            if cat_lens.zbins_mean is None:
                print("Redshift-weighting requires information about mean redshift in tomo bins of lens catalog")
            if cat_lens.zbins_std is None:
                print("Warning: Redshift-dispersion in tomo bins of lens catalog not given. Set to zero.")
                cat_lens.zbins_std = np.zeros(self.nbinsz_lens)
                
        _z3combis = self.nbinsz_source*self.nbinsz_lens*self.nbinsz_lens
        _r2combis = self.nbinsr*self.nbinsr
        sc = (self.n_cfs, self.nmax+1, _z3combis, self.nbinsr, self.nbinsr)
        sn = (self.nmax+1, _z3combis, self.nbinsr,self.nbinsr)
        szr = (self.nbinsz_source, self.nbinsz_lens, self.nbinsr)
        bin_centers = np.zeros(reduce(operator.mul, szr)).astype(np.float64)
        Upsilon_n = np.zeros(reduce(operator.mul, sc)).astype(np.complex128)
        Norm_n = np.zeros(reduce(operator.mul, sn)).astype(np.complex128)
        args_sourcecat = (cat_source.isinner.astype(np.int32), cat_source.weight.astype(np.float64), 
                          cat_source.pos1.astype(np.float64), cat_source.pos2.astype(np.float64), 
                          cat_source.tracer_1.astype(np.float64), cat_source.tracer_2.astype(np.float64), 
                          zbins_source.astype(np.int32), np.int32(self.nbinsz_source), np.int32(cat_source.ngal), )
        args_lenscat = (cat_lens.weight.astype(np.float64), cat_lens.pos1.astype(np.float64), 
                        cat_lens.pos2.astype(np.float64), zbins_lens.astype(np.int32), 
                        np.int32(self.nbinsz_lens), np.int32(cat_lens.ngal), )
        args_basesetup = (np.int32(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep),
                          np.int32(self.nbinsr), np.int32(self.multicountcorr), )
        if self.method=="Discrete":
            hash_dpix = max(1.,self.max_sep//10.)
            jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
            cat_source.build_spatialhash(dpix=hash_dpix, extent=jointextent)
            cat_lens.build_spatialhash(dpix=hash_dpix, extent=jointextent)
            nregions = np.int32(len(np.argwhere(cat_source.index_matcher>-1).flatten()))
            args_hash = (cat_source.index_matcher, cat_source.pixs_galind_bounds, cat_source.pix_gals,
                         cat_lens.index_matcher, cat_lens.pixs_galind_bounds, cat_lens.pix_gals, nregions, )
            args_pixgrid = (np.float64(cat_lens.pix1_start), np.float64(cat_lens.pix1_d), np.int32(cat_lens.pix1_n), 
                            np.float64(cat_lens.pix2_start), np.float64(cat_lens.pix2_d), np.int32(cat_lens.pix2_n), )
            args = (*args_sourcecat,
                    *args_lenscat,
                    *args_basesetup,
                    *args_hash,
                    *args_pixgrid,
                    np.int32(self.nthreads),
                    bin_centers,
                    Upsilon_n,
                    Norm_n, )
            func = self.clib.alloc_Gammans_discrete_GNN
        if self.method == "DoubleTree":
            cutfirst = np.int32(self.tree_resos[0]==0.)
            jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
            # Build multihashes for sources and lenses
            mhash_source = cat_source.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                                shuffle=self.shuffle_pix, normed=True, extent=jointextent)
            sngal_resos, spos1s, spos2s, sweights, szbins, sisinners, sallfields, sindex_matchers, \
            spixs_galind_bounds, spix_gals, sdpixs1_true, sdpixs2_true = mhash_source
            ngal_resos_source = np.array(sngal_resos, dtype=np.int32)
            weight_resos_source = np.concatenate(sweights).astype(np.float64)
            pos1_resos_source = np.concatenate(spos1s).astype(np.float64)
            pos2_resos_source = np.concatenate(spos2s).astype(np.float64)
            zbin_resos_source = np.concatenate(szbins).astype(np.int32)
            isinner_resos_source = np.concatenate(sisinners).astype(np.int32)
            e1_resos_source = np.concatenate([sallfields[i][0] for i in range(len(sallfields))]).astype(np.float64)
            e2_resos_source = np.concatenate([sallfields[i][1] for i in range(len(sallfields))]).astype(np.float64)
            index_matcher_source = np.concatenate(sindex_matchers).astype(np.int32)
            pixs_galind_bounds_source = np.concatenate(spixs_galind_bounds).astype(np.int32)
            pix_gals_source = np.concatenate(spix_gals).astype(np.int32)
            mhash_lens = cat_lens.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                                shuffle=self.shuffle_pix, normed=True, extent=jointextent)
            lngal_resos, lpos1s, lpos2s, lweights, lzbins, lisinners, lallfields, lindex_matchers, \
            lpixs_galind_bounds, lpix_gals, ldpixs1_true, ldpixs2_true = mhash_lens
            ngal_resos_lens = np.array(lngal_resos, dtype=np.int32)
            weight_resos_lens = np.concatenate(lweights).astype(np.float64)
            pos1_resos_lens = np.concatenate(lpos1s).astype(np.float64)
            pos2_resos_lens = np.concatenate(lpos2s).astype(np.float64)
            zbin_resos_lens = np.concatenate(lzbins).astype(np.int32)
            isinner_resos_lens = np.concatenate(lisinners).astype(np.int32)
            index_matcher_lens = np.concatenate(lindex_matchers).astype(np.int32)
            pixs_galind_bounds_lens = np.concatenate(lpixs_galind_bounds).astype(np.int32)
            pix_gals_lens = np.asarray(np.concatenate(lpix_gals)).astype(np.int32)
            index_matcher_flat = np.argwhere(cat_source.index_matcher>-1).flatten().astype(np.int32)
            nregions = np.int32(len(index_matcher_flat))
            # Collect args
            args_resoinfo = (np.int32(self.tree_nresos), np.int32(self.tree_nresos-cutfirst),
                             sdpixs1_true.astype(np.float64), sdpixs2_true.astype(np.float64), self.tree_redges, )
            args_leafs = (np.int32(self.resoshift_leafs), np.int32(self.minresoind_leaf), 
                          np.int32(self.maxresoind_leaf), )
            args_resos = (isinner_resos_source, weight_resos_source, pos1_resos_source, pos2_resos_source,
                          e1_resos_source, e2_resos_source, zbin_resos_source, ngal_resos_source, 
                          np.int32(self.nbinsz_source), isinner_resos_lens, weight_resos_lens, pos1_resos_lens, 
                          pos2_resos_lens, zbin_resos_lens, ngal_resos_lens, np.int32(self.nbinsz_lens), )
            args_mhash = (index_matcher_source, pixs_galind_bounds_source, pix_gals_source,
                          index_matcher_lens, pixs_galind_bounds_lens, pix_gals_lens, index_matcher_flat, nregions, )
            args_pixgrid = (np.float64(cat_lens.pix1_start), np.float64(cat_lens.pix1_d), np.int32(cat_lens.pix1_n), 
                            np.float64(cat_lens.pix2_start), np.float64(cat_lens.pix2_d), np.int32(cat_lens.pix2_n), )
            args = (*args_resoinfo,
                    *args_leafs,
                    *args_resos,
                    *args_basesetup,
                    *args_mhash,
                    *args_pixgrid,
                    np.int32(self.nthreads),
                    bin_centers,
                    Upsilon_n,
                    Norm_n, )
            func = self.clib.alloc_Gammans_doubletree_GNN
        #print("Doing %s"%self.method)
        if False:
            for elarg, arg in enumerate(args):
                toprint = (elarg, type(arg),)
                if isinstance(arg, np.ndarray):
                    toprint += (type(arg[0]), arg.shape)
                toprint += (func.argtypes[elarg], )
                print(toprint)
                print(arg)
        
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=(0,1))
        self.npcf_multipoles = Upsilon_n.reshape(sc)
        self.npcf_multipoles_norm = Norm_n.reshape(sn)
        self.projection = "X"
        self.is_edge_corrected = False
        
        if apply_edge_correction:
            self.edge_correction()
            
    def edge_correction(self, ret_matrices=False):
        assert(not self.is_edge_corrected)
        def gen_M_matrix(thet1,thet2,threepcf_n_norm):
            nvals, ntheta, _ = threepcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=np.int)
            nextM = np.zeros((nvals,nvals))
            for ind, ell in enumerate(narr):
                lminusn = ell-narr
                sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
                nextM[ind,sel] = threepcf_n_norm[(lminusn+nmax)[sel],thet1,thet2].real / threepcf_n_norm[nmax,thet1,thet2].real
            return nextM
    
        nvals, nzcombis, ntheta, _ = self.npcf_multipoles_norm.shape
        nmax = nvals-1
        threepcf_n_full = np.zeros((1,2*nmax+1, nzcombis, ntheta, ntheta), dtype=complex)
        threepcf_n_norm_full = np.zeros((2*nmax+1, nzcombis, ntheta, ntheta), dtype=complex)
        threepcf_n_corr = np.zeros(threepcf_n_full.shape, dtype=np.complex)
        threepcf_n_full[:,nmax:] = self.npcf_multipoles
        threepcf_n_norm_full[nmax:] = self.npcf_multipoles_norm
        for nextn in range(1,nvals):
            threepcf_n_full[0,nmax-nextn] = self.npcf_multipoles[0,nextn].transpose(0,2,1)
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
                    threepcf_n_corr[0,:,indz,thet1,thet2] = np.matmul(nextM_inv,threepcf_n_full[0,:,indz,thet1,thet2])
                        
        self.npcf_multipoles = threepcf_n_corr[:,nmax:]
        self.is_edge_corrected = True
        
        if ret_matrices:
            return threepcf_n_corr[:,nmax:], mats
     
    # TODO: 
    # * Include the z-weighting method
    # * Include the 2pcf as spline --> Should we also add an option to compute it here? Might be a mess
    #   as then we also would need methods to properly distribute randoms...
    # * Do a voronoi-tesselation at the multipole level? Would be just 2D, but still might help? Eventually
    #   bundle together cells s.t. tot_weight > theshold? However, this might then make the binning courser
    #   for certain triangle configs(?)
    def multipoles2npcf(self):
        """
        Notes:
        ------
        * The Upsilon and norms are only computed for the n>0 multipoles. We recover n<0 multipoles by symmetry
        considerations (i.e. Upsilon_{-n}(thet1,thet2,z1,z2,z3) = Upsilon_{n}(thet2,thet1,z1,z3,z2)). As the
        tomographic bin combinations are interpreted as a flat list, it needs to be appropriately shuffled --
        this is what `ztiler' is doing. 
        * When dividing by the (weighted) counts N, we set all contributions for which N<=0 to zero. 
        """
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        self.npcf = np.zeros((self.n_cfs, nzcombis, rbins, rbins, len(self.phi)), dtype=complex)
        self.npcf_norm = np.zeros((nzcombis, rbins, rbins, len(self.phi)), dtype=float)
        ztiler = np.arange(self.nbinsz_source*self.nbinsz_lens*self.nbinsz_lens).reshape(
            (self.nbinsz_source,self.nbinsz_lens,self.nbinsz_lens)).transpose(0,2,1).flatten().astype(np.int32)
        
        # 3PCF components
        conjmap = [0]
        N0 = 1./(2*np.pi) * self.npcf_multipoles_norm[0].astype(complex)
        for elm in range(self.n_cfs):
            for elphi, phi in enumerate(self.phi):
                tmp =  1./(2*np.pi) * self.npcf_multipoles[elm,0].astype(complex)
                for n in range(1,self.nmax+1):
                    _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                    tmp += _const * self.npcf_multipoles[elm,n].astype(complex)
                    tmp += _const.conj() * self.npcf_multipoles[conjmap[elm],n][ztiler].astype(complex).transpose(0,2,1)
                self.npcf[elm,...,elphi] = tmp
        # Normalization
        for elphi, phi in enumerate(self.phi):
            tmptotnorm = 1./(2*np.pi) * self.npcf_multipoles_norm[0].astype(complex)
            for n in range(1,self.nmax+1):
                _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                tmptotnorm += _const * self.npcf_multipoles_norm[n].astype(complex)
                tmptotnorm += _const.conj() * self.npcf_multipoles_norm[n][ztiler].astype(complex).transpose(0,2,1)
            self.npcf_norm[...,elphi] = tmptotnorm.real
            
        if self.is_edge_corrected:
            sel_zero = np.isnan(N0)
            _a = self.npcf
            _b = N0.real[:, :, np.newaxis]
            self.npcf = np.divide(_a, _b, out=np.zeros_like(_a), where=np.abs(_b)>0)
        else:
            _a = self.npcf
            _b = self.npcf_norm
            self.npcf = np.divide(_a, _b, out=np.zeros_like(_a), where=np.abs(_b)>0)
            #self.npcf = self.npcf/self.npcf_norm[0][None, ...].astype(complex)
        self.projection = "X"
            
            
    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        super()._projectnpcf(self, projection)
        
    ## INTEGRATED MEASURES ##        
    def computeNNM(self, radii, do_multiscale=False, tofile=False, filtercache=None):
        """
        Compute third-order aperture statistics
        """
        nb_config.NUMBA_DEFAULT_NUM_THREADS = self.nthreads
        nb_config.NUMBA_NUM_THREADS = self.nthreads
        
        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf()
            
        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            _rcut = 1 
        else:
            nrcombis = nradii*nradii*nradii
            _rcut = nradii
        NNM = np.zeros((1, self.nbinsz_source*self.nbinsz_lens*self.nbinsz_lens, nrcombis), dtype=complex)
        tmprcombi = 0
        for elr1, R1 in enumerate(radii):
            for elr2, R2 in enumerate(radii[:_rcut]):
                for elr3, R3 in enumerate(radii[:_rcut]):
                    if not do_multiscale:
                        R2 = R1
                        R3 = R1
                    if filtercache is not None:
                        A_NNM = filtercache[tmprcombi]
                    else:
                        A_NNM = self._NNM_filtergrid(R1, R2, R3)
                    NNM[0,:,tmprcombi] = np.nansum(A_NNM*self.npcf[0,...],axis=(1,2,3))
                    tmprcombi += 1
        return NNM
    
    def _NNM_filtergrid(self, R1, R2, R3):
        return self.__NNM_filtergrid(R1, R2, R3, self.bin_edges, self.bin_centers_mean, self.phi)
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def __NNM_filtergrid(R1, R2, R3, edges, centers, phis):
        nbinsr = len(centers)
        nbinsphi = len(phis)
        _cphis = np.cos(phis)
        _ephis = np.e**(1J*phis)
        _ephisc = np.e**(-1J*phis)
        Theta4 = 1./3. * (R1**2*R2**2 + R1**2*R3**2 + R2**2*R3**2) 
        a2 = 2./3. * R1**2*R2**2*R3**2 / Theta4
        ANNM = np.zeros((nbinsr,nbinsr,nbinsphi), dtype=nb_complex128)
        for elb in prange(nbinsr*nbinsr):
            elb1 = int(elb//nbinsr)
            elb2 = elb%nbinsr
            _y1 = centers[elb1]
            _dbin1 = edges[elb1+1] - edges[elb1]
            _y2 = centers[elb2]
            _dbin2 = edges[elb2+1] - edges[elb2]
            _dbinphi = phis[1] - phis[0]
            b0 = _y1**2/(2*R1**2)+_y2**2/(2*R2**2) - a2/4.*(
                _y1**2/R1**4 + 2*_y1*_y2*_cphis/(R1**2*R2**2) + _y2**2/R2**4)
            g1 = _y1 - a2/2. * (_y1/R1**2 + _y2*_ephisc/R2**2)
            g2 = _y2 - a2/2. * (_y2/R2**2 + _y1*_ephis/R1**2)
            g1c = g1.conj()
            g2c = g2.conj()
            F1 = 2*R1**2 - g1*g1c
            F2 = 2*R2**2 - g2*g2c
            pref = np.e**(-b0)/(72*np.pi*Theta4**2)
            sum1 = (g1-_y1)*(g2-_y2) * (1/a2*F1*F2 - (F1+F2) + 2*a2 + g1c*g2*_ephisc + g1*g2c*_ephis) 
            sum2 = ((g2-_y2) + (g1-_y1)*_ephis) * (g1*(F2-2*a2) + g2*(F1-2*a2)*_ephisc)
            sum3 = 2*g1*g2*a2 
            _measures = _y1*_dbin1 * _y2*_dbin2 * _dbinphi
            ANNM[elb1,elb2] = _measures * pref * (sum1-sum2+sum3)

        return ANNM
    
# Very close to being a mere copy of GNN...
class NGGCorrelation(BinnedNPCF):
    """ Lens-Source-Source correlation function """
    def __init__(self, min_sep, max_sep, **kwargs):
        r""" Class containing methods to measure and and obtain statistics that are built
        from third-order lens-shear-shear correlation functions.
        
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
        diagrenorm: bool, optional
            Deprecated
        shuffle_pix: int, optional
            Choice of how to define centers of the cells in the spatial hash structure.
            Defaults to ``1``, i.e. random positioning.
        tree_resos: list, optional
            The cell sizes of the hierarchical spatial hash structure
        tree_edges: list, optional
            Deprecated(?)
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
        super().__init__(3, [0,2,2], n_cfs=2, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X"]
        self.nbinsz_source = None
        self.nbinsz_lens = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        
    def process(self, cat_source, cat_lens, nthreads=16, dotomo_source=True, dotomo_lens=True, apply_edge_correction=False):
        self._checkcats([cat_lens, cat_source, cat_source], [0, 2, 2])
            
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
                
        _z3combis = self.nbinsz_lens*self.nbinsz_source*self.nbinsz_source
        _r2combis = self.nbinsr*self.nbinsr
        sc = (self.n_cfs, 2*self.nmax+1, _z3combis, self.nbinsr, self.nbinsr)
        sn = (2*self.nmax+1, _z3combis, self.nbinsr,self.nbinsr)
        szr = (self.nbinsz_lens, self.nbinsz_source, self.nbinsr)
        bin_centers = np.zeros(reduce(operator.mul, szr)).astype(np.float64)
        Upsilon_n = np.zeros(reduce(operator.mul, sc)).astype(np.complex128)
        Norm_n = np.zeros(reduce(operator.mul, sn)).astype(np.complex128)
        args_sourcecat = (cat_source.weight.astype(np.float64), 
                          cat_source.pos1.astype(np.float64), cat_source.pos2.astype(np.float64), 
                          cat_source.tracer_1.astype(np.float64), cat_source.tracer_2.astype(np.float64), 
                          zbins_source.astype(np.int32), np.int32(self.nbinsz_source), np.int32(cat_source.ngal), )
        args_lenscat = (cat_lens.isinner.astype(np.int32), cat_lens.weight.astype(np.float64), cat_lens.pos1.astype(np.float64), 
                        cat_lens.pos2.astype(np.float64), zbins_lens.astype(np.int32), 
                        np.int32(self.nbinsz_lens), np.int32(cat_lens.ngal), )
        args_basesetup = (np.int32(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep),
                          np.int32(self.nbinsr), np.int32(self.multicountcorr), )
        if self.method=="Discrete":
            hash_dpix = max(1.,self.max_sep//10.)
            jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
            cat_source.build_spatialhash(dpix=hash_dpix, extent=jointextent)
            cat_lens.build_spatialhash(dpix=hash_dpix, extent=jointextent)
            nregions = np.int32(len(np.argwhere(cat_lens.index_matcher>-1).flatten()))
            args_hash = (cat_source.index_matcher, cat_source.pixs_galind_bounds, cat_source.pix_gals,
                         cat_lens.index_matcher, cat_lens.pixs_galind_bounds, cat_lens.pix_gals, nregions, )
            args_pixgrid = (np.float64(cat_lens.pix1_start), np.float64(cat_lens.pix1_d), np.int32(cat_lens.pix1_n), 
                            np.float64(cat_lens.pix2_start), np.float64(cat_lens.pix2_d), np.int32(cat_lens.pix2_n), )
            args = (*args_sourcecat,
                    *args_lenscat,
                    *args_basesetup,
                    *args_hash,
                    *args_pixgrid,
                    np.int32(self.nthreads),
                    bin_centers,
                    Upsilon_n,
                    Norm_n, )
            func = self.clib.alloc_Gammans_discrete_NGG
        if self.method=="Tree" or self.method == "DoubleTree":
            cutfirst = np.int32(self.tree_resos[0]==0.)
            jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
            # Build multihashes for sources and lenses
            mhash_source = cat_source.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                                shuffle=self.shuffle_pix, normed=True, extent=jointextent)
            sngal_resos, spos1s, spos2s, sweights, szbins, sisinners, sallfields, sindex_matchers, \
            spixs_galind_bounds, spix_gals, sdpixs1_true, sdpixs2_true = mhash_source
            ngal_resos_source = np.array(sngal_resos, dtype=np.int32)
            weight_resos_source = np.concatenate(sweights).astype(np.float64)
            pos1_resos_source = np.concatenate(spos1s).astype(np.float64)
            pos2_resos_source = np.concatenate(spos2s).astype(np.float64)
            zbin_resos_source = np.concatenate(szbins).astype(np.int32)
            isinner_resos_source = np.concatenate(sisinners).astype(np.int32)
            e1_resos_source = np.concatenate([sallfields[i][0] for i in range(len(sallfields))]).astype(np.float64)
            e2_resos_source = np.concatenate([sallfields[i][1] for i in range(len(sallfields))]).astype(np.float64)
            index_matcher_source = np.concatenate(sindex_matchers).astype(np.int32)
            pixs_galind_bounds_source = np.concatenate(spixs_galind_bounds).astype(np.int32)
            pix_gals_source = np.concatenate(spix_gals).astype(np.int32)
            mhash_lens = cat_lens.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                            shuffle=self.shuffle_pix, normed=True, extent=jointextent)
            lngal_resos, lpos1s, lpos2s, lweights, lzbins, lisinners, lallfields, lindex_matchers, \
            lpixs_galind_bounds, lpix_gals, ldpixs1_true, ldpixs2_true = mhash_lens
            ngal_resos_lens = np.array(lngal_resos, dtype=np.int32)
            weight_resos_lens = np.concatenate(lweights).astype(np.float64)
            pos1_resos_lens = np.concatenate(lpos1s).astype(np.float64)
            pos2_resos_lens = np.concatenate(lpos2s).astype(np.float64)
            zbin_resos_lens = np.concatenate(lzbins).astype(np.int32)
            isinner_resos_lens = np.concatenate(lisinners).astype(np.int32)
            index_matcher_lens = np.concatenate(lindex_matchers).astype(np.int32)
            pixs_galind_bounds_lens = np.concatenate(lpixs_galind_bounds).astype(np.int32)
            pix_gals_lens = np.asarray(np.concatenate(lpix_gals)).astype(np.int32)
            index_matcher_flat = np.argwhere(cat_lens.index_matcher>-1).flatten().astype(np.int32)
            nregions = np.int32(len(index_matcher_flat))
        if self.method=="Tree":
            # Collect args
            args_resoinfo = (np.int32(self.tree_nresos), self.tree_redges,)
            args_resos_sourcecat = (weight_resos_source, pos1_resos_source, pos2_resos_source,
                                    e1_resos_source, e2_resos_source, zbin_resos_source, 
                                    np.int32(self.nbinsz_source), ngal_resos_source, )
            args_mhash = (index_matcher_source, pixs_galind_bounds_source, pix_gals_source,
                          index_matcher_lens, pixs_galind_bounds_lens, pix_gals_lens, nregions, )
            args_pixgrid = (np.float64(cat_lens.pix1_start), np.float64(cat_lens.pix1_d), np.int32(cat_lens.pix1_n), 
                            np.float64(cat_lens.pix2_start), np.float64(cat_lens.pix2_d), np.int32(cat_lens.pix2_n), )
            args = (*args_resoinfo,
                    *args_resos_sourcecat,
                    *args_lenscat,
                    *args_mhash,
                    *args_pixgrid,
                    *args_basesetup,
                    np.int32(self.nthreads),
                    bin_centers,
                    Upsilon_n,
                    Norm_n, )
            func = self.clib.alloc_Gammans_tree_NGG            
        if self.method == "DoubleTree":
            # Collect args
            args_resoinfo = (np.int32(self.tree_nresos), np.int32(self.tree_nresos-cutfirst),
                             sdpixs1_true.astype(np.float64), sdpixs2_true.astype(np.float64), self.tree_redges, )
            args_leafs = (np.int32(self.resoshift_leafs), np.int32(self.minresoind_leaf), 
                          np.int32(self.maxresoind_leaf), )
            args_resos = (isinner_resos_source, weight_resos_source, pos1_resos_source, pos2_resos_source,
                          e1_resos_source, e2_resos_source, zbin_resos_source, ngal_resos_source, 
                          np.int32(self.nbinsz_source), isinner_resos_lens, weight_resos_lens, pos1_resos_lens, 
                          pos2_resos_lens, zbin_resos_lens, ngal_resos_lens, np.int32(self.nbinsz_lens), )
            args_mhash = (index_matcher_source, pixs_galind_bounds_source, pix_gals_source,
                          index_matcher_lens, pixs_galind_bounds_lens, pix_gals_lens, index_matcher_flat, nregions, )
            args_pixgrid = (np.float64(cat_lens.pix1_start), np.float64(cat_lens.pix1_d), np.int32(cat_lens.pix1_n), 
                            np.float64(cat_lens.pix2_start), np.float64(cat_lens.pix2_d), np.int32(cat_lens.pix2_n), )
            args = (*args_resoinfo,
                    *args_leafs,
                    *args_resos,
                    *args_basesetup,
                    *args_mhash,
                    *args_pixgrid,
                    np.int32(self.nthreads),
                    bin_centers,
                    Upsilon_n,
                    Norm_n, )
            func = self.clib.alloc_Gammans_doubletree_NGG
        #print("Doing %s"%self.method)
        if True:
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
        
        # Components of npcf are ordered as (Ups_-, Ups_+)
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=(0,1))
        self.npcf_multipoles = Upsilon_n.reshape(sc)
        self.npcf_multipoles_norm = Norm_n.reshape(sn)
        self.projection = "X"
        self.is_edge_corrected = False
        
        if apply_edge_correction:
            self.edge_correction()
            
    def edge_correction(self, ret_matrices=False):
        
        assert(not self.is_edge_corrected)
        def gen_M_matrix(thet1,thet2,threepcf_n_norm):
            nvals, ntheta, _ = threepcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=np.int)
            nextM = np.zeros((nvals,nvals))
            for ind, ell in enumerate(narr):
                lminusn = ell-narr
                sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
                nextM[ind,sel] = threepcf_n_norm[(lminusn+nmax)[sel],thet1,thet2].real / threepcf_n_norm[nmax,thet1,thet2].real
            return nextM
    
        _nvals, nzcombis, ntheta, _ = self.npcf_multipoles_norm.shape
        nvals = int((_nvals-1)/2)
        nmax = nvals-1
        threepcf_n_corr = np.zeros_like(self.npcf_multipoles)
        if ret_matrices:
            mats = np.zeros((nzcombis,ntheta,ntheta,nvals,nvals))
        for indz in range(nzcombis):
            #sys.stdout.write("%i"%indz)
            for thet1 in range(ntheta):
                for thet2 in range(ntheta):
                    nextM = gen_M_matrix(thet1,thet2,self.npcf_multipoles_norm[:,indz])
                    nextM_inv = np.linalg.inv(nextM)
                    if ret_matrices:
                        mats[indz,thet1,thet2] = nextM
                    for el_cf in range(self.n_cfs):
                        threepcf_n_corr[el_cf,:,indz,thet1,thet2] = np.matmul(
                            nextM_inv,self.npcf_multipoles[el_cf,:,indz,thet1,thet2])
                        
        self.npcf_multipoles = threepcf_n_corr
        self.is_edge_corrected = True
        
        if ret_matrices:
            return threepcf_n_corr, mats
    
    def multipoles2npcf(self, integrated=False):
        """
        Notes:
        ------
        * The Upsilon and norms are only computed for the n>0 multipoles. We recover n<0 multipoles by symmetry
        considerations (i.e. Upsilon_{-n}(thet1,thet2,z1,z2,z3) = Upsilon_{n}(thet2,thet1,z1,z3,z2)). As the
        tomographic bin combinations are interpreted as a flat list, it needs to be appropriately shuffled --
        this is what `ztiler' is doing. 
        * When dividing by the (weighted) counts N, we set all contributions for which N<=0 to zero. 
        """
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        self.npcf = np.zeros((self.n_cfs, nzcombis, rbins, rbins, len(self.phi)), dtype=complex)
        self.npcf_norm = np.zeros((nzcombis, rbins, rbins, len(self.phi)), dtype=float)
        ztiler = np.arange(self.nbinsz_lens*self.nbinsz_source*self.nbinsz_source).reshape(
            (self.nbinsz_lens,self.nbinsz_source,self.nbinsz_source)).transpose(0,2,1).flatten().astype(np.int32)
        
        # NGG components
        for elphi, phi in enumerate(self.phi):
            tmp = np.zeros((self.n_cfs, nzcombis, rbins, rbins),dtype=complex)
            tmpnorm = np.zeros((nzcombis, rbins, rbins),dtype=complex)
            for n in range(2*self.nmax+1):
                dphi = self.phi[1] - self.phi[0]
                if integrated:
                    if n==self.nmax:
                        ifac = dphi
                    else:
                        ifac = 2./(n-self.nmax) * np.sin((n-self.nmax)*dphi/2.)
                else:
                    ifac = dphi
                _const = 1./(2*np.pi) * np.exp(1J*(n-self.nmax)*phi) * ifac
                tmpnorm += _const * self.npcf_multipoles_norm[n].astype(complex)
                for el_cf in range(self.n_cfs):
                    tmp[el_cf] += _const * self.npcf_multipoles[el_cf,n].astype(complex)
            self.npcf[...,elphi] = tmp
            self.npcf_norm[...,elphi] = tmpnorm.real
            
        if self.is_edge_corrected:
            N0 = dphi/(2*np.pi) * self.npcf_multipoles_norm[self.nmax].astype(complex)
            sel_zero = np.isnan(N0)
            _a = self.npcf
            _b = N0.real[np.newaxis, :, :, :, np.newaxis]
            self.npcf = np.divide(_a, _b, out=np.zeros_like(_a), where=_b>0)
        else:
            _a = self.npcf
            _b = self.npcf_norm
            self.npcf = np.divide(_a, _b, out=np.zeros_like(_a), where=_b>0)
            #self.npcf = self.npcf/self.npcf_norm[0][None, ...].astype(complex)
        self.projection = "X"
            
    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        super()._projectnpcf(self, projection)
        
    ## INTEGRATED MEASURES ##        
    def computeNMM(self, radii, do_multiscale=False, tofile=False, filtercache=None):
        """
        Compute third-order aperture statistics
        """
        
        nb_config.NUMBA_DEFAULT_NUM_THREADS = self.nthreads
        nb_config.NUMBA_NUM_THREADS = self.nthreads
        
        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf()
            
        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            _rcut = 1 
        else:
            nrcombis = nradii*nradii*nradii
            _rcut = nradii
        NMM = np.zeros((3, self.nbinsz_lens*self.nbinsz_source*self.nbinsz_source, nrcombis), dtype=complex)
        tmprcombi = 0
        for elr1, R1 in enumerate(radii):
            for elr2, R2 in enumerate(radii[:_rcut]):
                for elr3, R3 in enumerate(radii[:_rcut]):
                    if not do_multiscale:
                        R2 = R1
                        R3 = R1
                    if filtercache is not None:
                        A_NMM = filtercache[tmprcombi]
                    else:
                        A_NMM = self._NMM_filtergrid(R1, R2, R3)
                    _NMM =  np.nansum(A_NMM[0]*self.npcf[0,...],axis=(1,2,3))
                    _NMMstar =  np.nansum(A_NMM[1]*self.npcf[1,...],axis=(1,2,3))
                    NMM[0,:,tmprcombi] = (_NMM + _NMMstar).real/2.
                    NMM[1,:,tmprcombi] = (-_NMM + _NMMstar).real/2.
                    NMM[2,:,tmprcombi] = (_NMM + _NMMstar).imag/2.
                    tmprcombi += 1
        return NMM
    
    def _NMM_filtergrid(self, R1, R2, R3):
        return self.__NMM_filtergrid(R1, R2, R3, self.bin_edges, self.bin_centers_mean, self.phi)
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def __NMM_filtergrid(R1, R2, R3, edges, centers, phis):
        nbinsr = len(centers)
        nbinsphi = len(phis)
        _cphis = np.cos(phis)
        _ephis = np.e**(1J*phis)
        _ephisc = np.e**(-1J*phis)
        Theta4 = 1./3. * (R1**2*R2**2 + R1**2*R3**2 + R2**2*R3**2) 
        a2 = 2./3. * R1**2*R2**2*R3**2 / Theta4
        ANMM = np.zeros((2,nbinsr,nbinsr,nbinsphi), dtype=nb_complex128)
        for elb in prange(nbinsr*nbinsr):
            elb1 = int(elb//nbinsr)
            elb2 = elb%nbinsr
            _y1 = centers[elb1]
            _dbin1 = edges[elb1+1] - edges[elb1]
            _y2 = centers[elb2]
            _dbin2 = edges[elb2+1] - edges[elb2]
            _dbinphi = phis[1] - phis[0]
            
            csq = a2**2/4. * (_y1**2/R1**4 + _y2**2/R2**4 + 2*_y1*_y2*_cphis/(R1**2*R2**2))
            b0 = _y1**2/(2*R1**2)+_y2**2/(2*R2**2) - csq/a2

            g1 = _y1 - a2/2. * (_y1/R1**2 + _y2*_ephisc/R2**2)
            g2 = _y2 - a2/2. * (_y2/R2**2 + _y1*_ephis/R1**2)
            g1c = g1.conj()
            g2c = g2.conj()
            pref = np.e**(-b0)/(72*np.pi*Theta4**2)
            _h1 = 2*(g2c*_y1+g1*_y2-2*g1*g2c)*(g1*g2c+2*a2*_ephisc)
            _h2 = 2*a2*(2*R3**2-csq-3*a2)*_ephisc*_ephisc
            _h3 = 4*g1*g2c*(2*R3**2-csq-2*a2)*_ephisc
            _h4 = (g1*g2c)**2/a2 * (2*R3**2-csq-a2)
            sum_MMN = pref*g1*g2 * ((R3**2/R1**2+R3**2/R2**2-csq/a2)*g1*g2 + 2*(g2*_y1+g1*_y2-2*g1*g2))
            sum_MMstarN = pref * (_h1 + _h2 + _h3 + _h4)
            _measures = _y1*_dbin1 * _y2*_dbin2 * _dbinphi

            ANMM[0,elb1,elb2] = _measures * sum_MMN
            ANMM[1,elb1,elb2] = _measures * sum_MMstarN
                
        return ANMM
    
class FFFCorrelation(BinnedNPCF):
    """ Third-Order Flexion correlation function """
    def __init__(self, n_cfs, min_sep, max_sep, **kwargs):
        super().__init__(3, [3,3,3], n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
        
    def process_single(self, cat, nthreads, dotomo=True):
        pass
    
#############################
## FOURTH-ORDER STATISTICS ##
#############################
class GGGGCorrelation_NoTomo(BinnedNPCF):
    
    def __init__(self, min_sep, max_sep, verbose=False, thetabatchsize_max=10000, **kwargs):
        r""" Class containing methods to measure and and obtain statistics that are built
        from nontomographic fourth-order shear correlation functions.
        
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
        thetabatchsize_max: int, optional
            The largest number of radial bin combinations that are processed in parallel.
            Defaults to ``10 000``.
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
        diagrenorm: bool, optional
            Deprecated
        shuffle_pix: int, optional
            Choice of how to define centers of the cells in the spatial hash structure.
            Defaults to ``1``, i.e. random positioning.
        tree_resos: list, optional
            The cell sizes of the hierarchical spatial hash structure
        tree_edges: list, optional
            Deprecated(?)
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
        super().__init__(order=4, spins=np.array([2,2,2,2], dtype=np.int32),
                         n_cfs=8, min_sep=min_sep, max_sep=max_sep, **kwargs)
        
        self.thetabatchsize_max = thetabatchsize_max
        self.verbose = verbose
        self.projection = None
        self.projections_avail = [None, "X", "Centroid"]
        self.proj_dict = {"X":0, "Centroid":1}
        self.methods_avail = ["Discrete"] # Default right now. See if tree-methods make sense.
        self.method = "Discrete" # Default right now 
        self.nbinsz = 1
        self.nzcombis = 1
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        self.project["X"]["Centroid"] = self._x2centroid
        
    def process(self, cat, statistics="all", tofile=False, apply_edge_correction=False, projection="X",
                lowmem=None, mapradii=None, batchsize=None, cutlen=2**31-1):
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
        projection = np.int32(self.proj_dict[projection])
        
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
        args_basecat = (cat.isinner.astype(np.int32), cat.weight, cat.pos1, cat.pos2, 
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
        
        # Build args based on chosen methods
        if self.method=="Discrete" and not lowmem:
            args_basesetup = (np.int32(_nmax), np.float64(self.min_sep), 
                              np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
                              np.int32(self.nbinsr), np.int32(self.multicountcorr), )
            args = (*args_basecat,
                    *args_basesetup,
                    *args_hash,
                    np.int32(self.nthreads),
                    bin_centers,
                    Upsilon_n,
                    N_n)
            func = self.clib.alloc_notomoGammans_discrete_gggg 
        if self.method=="Discrete" and lowmem:
            if batchsize is None:
                batchsize = min(_nbinsr3,min(self.thetabatchsize_max,_nbinsr3/self.nthreads))
                print("Using batchsize of %i for radial bins"%batchsize)
            if batchsize==self.thetabatchsize_max:
                nbatches = np.int32(np.ceil(_nbinsr3/batchsize))
            else:
                nbatches = np.int32(_nbinsr3/batchsize)
            thetacombis_batches = np.arange(_nbinsr3).astype(np.int32)
            cumnthetacombis_batches = (np.arange(nbatches+1)*_nbinsr3/(nbatches)).astype(np.int32)
            nthetacombis_batches = (cumnthetacombis_batches[1:]-cumnthetacombis_batches[:-1]).astype(np.int32)
            cumnthetacombis_batches[-1] = _nbinsr3
            nthetacombis_batches[-1] = _nbinsr3-cumnthetacombis_batches[-2]
            thetacombis_batches = thetacombis_batches.flatten().astype(np.int32)
            nbatches = len(nthetacombis_batches)
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
                    projection,
                    *args_map4,
                    *args_4pcf)
            func = self.clib.alloc_notomoMap4_disc_gggg  
            func = self.clib.alloc_notomoGammans_discrete_gggg 
        if self.method=="Tree" and lowmem:
            # Prepare mask for nonredundant theta- and multipole configurations
            _resradial = self.__gen_thetacombis(self.nbinsr, ordered=True)
            nbinsr3, allelbs, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches = _resradial
            assert(self.nmaxs[0]==self.nmaxs[1])
            _resmultipoles = self.__gen_indices_Ups(self.nmaxs[0])
            _shape, _inds, _n2s, _n3s = _resmultipoles
            # Prepare reduced catalogs
            
            # Build args
            args_basesetup = (np.int32(_nmax), 
                              np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                              np.int32(self.multicountcorr),
                              _inds, np.int32(len(_inds)), self.phis[0].astype(np.float64), 
                              2*np.pi/_nphis*np.ones(_nphis, dtype=np.float64), np.int32(_nphis))
            args_resos = ()
            args_4pcf = (np.int32(alloc_4pcfmultipoles), np.int32(alloc_4pcfreal), 
                         bin_centers, Upsilon_n, N_n, fourpcf, fourpcf_norm, )
            args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
            args_map4 = (mapradii, np.int32(len(mapradii)), M4correlators)
            args = (*args_basecat,
                    *args_basesetup,
                    *args_hash,
                    *args_thetas,
                    np.int32(self.nthreads),
                    projection,
                    *args_map4,
                    *args_4pcf)
            func = self.clib.alloc_notomoMap4_tree_gggg  
            """
            void alloc_notomoMap4_tree_gggg(
                int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
                int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
                int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
                
                int nresos, double *reso_redges, int *ngal_resos, 
                double *weight_resos, double *pos1_resos, double *pos2_resos, 
                double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
                int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
                double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
                int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
                int nthreads, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
                
                int alloc_4pcfmultipoles, int alloc_4pcfreal,
                double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, 
                double complex *Norms);

            void getMultiplolesFromSymm(double complex *Upsn_in, double complex *Nn_in,
                                        int nmax, int eltrafo, int *nindices, int len_nindices,
                                        double complex *Upsn_out, double complex *Nn_out);
            """
        # Optionally print the arguments 
        if self.verbose:
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
            self.npcf_multipoles = Upsilon_n.reshape(sc)
            self.npcf_multipoles_norm = N_n.reshape(sn)
        if "4pcf_real" in statistics:
            if lowmem:
                self.npcf = fourpcf.reshape(s4pcf)
                self.npcf_norm = fourpcf_norm.reshape(s4pcfn) 
            else:
                if self.verbose:
                    print("Transforming output to real space basis")
                self.multipoles2npcf_c(projection=projection)
        if hasintegratedstats:
            if "M4" in statistics:
                istatout += (M4correlators.reshape((8,self.nzcombis,len(mapradii))), )
            # TODO allocate map4, map4c etc.
            
        return istatout
    
    def multipoles2npcf_c(self, projection="X"):
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
        
        return npcf_out.reshape((_self.n_cfs, phis1,_nphis1)), npcf_norm_out.reshape((_nphis1,_nphis2))
    
    def computeMap4(self, radii, do_multiscale=False, tofile=False, filtercache=None):
        
        # Prepare real space 4pcf in X-projection
        if self.npcf is None:
            self.multipoles2npcf_c(projection="X")
        if self.projection != "X":
            self.projectnpcf("X")
        
        # Build aperture radii
        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            filterfunc = self._map4_filtergrid_singleR
            _rcut = 1 
        else:
            nrcombis = nradii*nradii*nradii
            _rcut = nradii
            raise NotImplementedError
        
        # Compute M4 basis
        res_M4 = np.zeros((8,self.nzcombis,nradii), dtype=np.complex128)
        res_Map4 = np.zeros((16,self.nzcombis,nradii))
        # Do numerical integral by simple Riemann sums
        for elb1 in range(self.nbinsr):
            for elb2 in range(self.nbinsr):
                for elb3 in range(self.nbinsr):
                    sys.stdout.write("%i %i %i\r"%(elb1,elb2,elb3))
                    npcf_f = self.npcf[:,:,elb1,elb2,elb3].flatten()
                    for elR, R_ap in enumerate(radii):
                        y1 = self.bin_centers_mean[elb1]/R_ap
                        y2 = self.bin_centers_mean[elb2]/R_ap
                        y3 = self.bin_centers_mean[elb3]/R_ap
                        dy1 = (self.bin_edges[elb1+1]-self.bin_edges[elb1])/R_ap
                        dy2 = (self.bin_edges[elb2+1]-self.bin_edges[elb2])/R_ap
                        dy3 = (self.bin_edges[elb3+1]-self.bin_edges[elb3])/R_ap
                        dphis1 = np.ones(self.nbinsphi[0])*(self.phis[0][1]-self.phis[0][0])
                        dphis2 = np.ones(self.nbinsphi[1])*(self.phis[1][1]-self.phis[1][0])
                        m4corr_meas = res_M4[:,0,elR]
                        self.clib.fourpcf2M4correlators_parallel(self.nzcombis,
                                                                 y1, y2, y3, dy1, dy2, dy3,
                                                                 self.phis[0], self.phis[0], 
                                                                 dphis1, dphis2, 
                                                                 self.nbinsphi[0], self.nbinsphi[0],
                                                                 self.nthreads, 
                                                                 npcf_f, 
                                                                 m4corr_meas)
        
        # Transform to Map basis
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
        res_Map4[[0,5,6,7,8,9,10,15],0] = Mcorr2Map4_re@(res_M4[:,0].real)
        res_Map4[[1,2,3,4,11,12,13,14],0] = Mcorr2Map4_im@(res_M4[:,0].imag)

        return res_M4, res_Map4
                    
    
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
    
    
    ## MISC OLD FUNCTIONS ##
    def __computeMap4(self, radii, do_multiscale=False, tofile=False, filtercache=None):
        """
        Compute fourth-order aperture statistics
        """
        
        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf()
            
        if self.projection != "Centroid":
            self.projectnpcf("Centroid")
        
        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            filterfunc = self._map4_filtergrid_singleR
            _rcut = 1 
        else:
            nrcombis = nradii*nradii*nradii
            _rcut = nradii
            raise NotImplementedError
            
        map4s = np.zeros((16, self.nzcombis, nrcombis), dtype=complex)
        M4 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M3Mc_0 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M3Mc_1 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M3Mc_2 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M3Mc_3 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M2M2_01 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M2M2_02 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        M2M2_03 = np.zeros((self.nzcombis, nrcombis), dtype=complex)
        tmprcombi = 0
        for elr1, R1 in enumerate(radii):
            for elr2, R2 in enumerate(radii[:_rcut]):
                for elr3, R3 in enumerate(radii[:_rcut]):
                    for elr4, R4 in enumerate(radii[:_rcut]):
                        if self.verbose:
                            sys.stdout.write("\r Doing %.i/%i aperture radii"%(elr1,nradii))
                        if not do_multiscale:
                            R2 = R1
                            R3 = R1
                            R4 = R1
                        if filtercache is not None:
                            F1, F1, F2, F3, F4, F5, F6, F7 = (filtercache[tmprcombi][i] for i in range(8))
                        else:
                            F0, F1, F2, F3, F4, F5, F6, F7 = filterfunc(R1, R2, R3, R4)
                        M4[:,tmprcombi] = np.nansum(F0*self.npcf[0,...],axis=(1,2,3,4,5))
                        M3Mc_0[:,tmprcombi] = np.nansum(F1*self.npcf[1,...],axis=(1,2,3,4,5))
                        M3Mc_1[:,tmprcombi] = np.nansum(F2*self.npcf[2,...],axis=(1,2,3,4,5))
                        M3Mc_2[:,tmprcombi] = np.nansum(F3*self.npcf[3,...],axis=(1,2,3,4,5))
                        M3Mc_3[:,tmprcombi] = np.nansum(F4*self.npcf[4,...],axis=(1,2,3,4,5))
                        M2M2_01[:,tmprcombi] = np.nansum(F5*self.npcf[5,...],axis=(1,2,3,4,5))
                        M2M2_02[:,tmprcombi] = np.nansum(F6*self.npcf[6,...],axis=(1,2,3,4,5))
                        M2M2_03[:,tmprcombi] = np.nansum(F7*self.npcf[7,...],axis=(1,2,3,4,5))
                        tmprcombi += 1            
        map4s[0]  = 1./8. * (+ M4 + M3Mc_0 + M3Mc_1 + M3Mc_2 + M3Mc_3 + M2M2_01 + M2M2_02 + M2M2_03).real # Map Map Map Map
        map4s[1]  = 1./8. * (+ M4 - M3Mc_0 + M3Mc_1 + M3Mc_2 + M3Mc_3 - M2M2_01 - M2M2_02 - M2M2_03).imag # Mx Map Map Map
        map4s[2]  = 1./8. * (+ M4 + M3Mc_0 - M3Mc_1 + M3Mc_2 + M3Mc_3 - M2M2_01 + M2M2_02 + M2M2_03).imag # Map Mx Map Map
        map4s[3]  = 1./8. * (+ M4 + M3Mc_0 + M3Mc_1 - M3Mc_2 + M3Mc_3 + M2M2_01 - M2M2_02 + M2M2_03).imag # Map Map Mx Map
        map4s[4]  = 1./8. * (+ M4 + M3Mc_0 + M3Mc_1 + M3Mc_2 - M3Mc_3 + M2M2_01 + M2M2_02 - M2M2_03).imag # Map Map Map Mx
        map4s[5]  = 1./8. * (- M4 - M3Mc_0 - M3Mc_1 + M3Mc_2 + M3Mc_3 - M2M2_01 + M2M2_02 + M2M2_03).real # Map Map Mx  Mx
        map4s[6]  = 1./8. * (- M4 - M3Mc_0 + M3Mc_1 - M3Mc_2 + M3Mc_3 + M2M2_01 - M2M2_02 + M2M2_03).real # Map Mx  Map Mx
        map4s[7]  = 1./8. * (- M4 - M3Mc_0 + M3Mc_1 + M3Mc_2 - M3Mc_3 + M2M2_01 + M2M2_02 - M2M2_03).real # Map Mx  Mx  Map
        map4s[8]  = 1./8. * (- M4 + M3Mc_0 - M3Mc_1 - M3Mc_2 + M3Mc_3 + M2M2_01 + M2M2_02 - M2M2_03).real # Mx  Map Map Mx
        map4s[9]  = 1./8. * (- M4 + M3Mc_0 - M3Mc_1 + M3Mc_2 - M3Mc_3 + M2M2_01 - M2M2_02 + M2M2_03).real # Mx  Map Mx  Map
        map4s[10] = 1./8. * (- M4 + M3Mc_0 + M3Mc_1 - M3Mc_2 - M3Mc_3 - M2M2_01 + M2M2_02 + M2M2_03).real # Mx  Mx  Map Map
        map4s[11] = 1./8. * (- M4 - M3Mc_0 + M3Mc_1 + M3Mc_2 + M3Mc_3 + M2M2_01 + M2M2_02 + M2M2_03).imag # Map Map Map Map
        map4s[12] = 1./8. * (- M4 + M3Mc_0 - M3Mc_1 + M3Mc_2 + M3Mc_3 + M2M2_01 - M2M2_02 - M2M2_03).imag # Map Map Map Map
        map4s[13] = 1./8. * (- M4 + M3Mc_0 + M3Mc_1 - M3Mc_2 + M3Mc_3 - M2M2_01 + M2M2_02 - M2M2_03).imag # Map Map Map Map
        map4s[14] = 1./8. * (- M4 + M3Mc_0 + M3Mc_1 + M3Mc_2 - M3Mc_3 - M2M2_01 - M2M2_02 + M2M2_03).imag # Map Map Map Map
        map4s[15] = 1./8. * (+ M4 - M3Mc_0 - M3Mc_1 - M3Mc_2 - M3Mc_3 + M2M2_01 + M2M2_02 + M2M2_03).real # Mx  Mx  Mx  Mx
                                            
        if tofile:
            # Write to file
            pass
            
        return map4s
    
    def _map4_filtergrid_singleR(self, R1, R2, R3, R4):
        return self.__map4_filtergrid_singleR(R1, R2, R3, R4, self.bin_edges, self.bin_centers_mean, self.phis[0])
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def __map4_filtergrid_singleR(R1, R2, R3, R4, normys_edges, normys_centers, phis):
        R_ap = R1
        R_ap2 = R_ap**2
        R_ap4 = R_ap**4
        R_ap6 = R_ap**6
        R_ap8 = R_ap**8
        nbinsr = len(normys_centers)
        nbinsphi = len(phis)
        F0 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        F1 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        F2 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        F3 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        F4 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        F5 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        F6 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        F7 = np.zeros((nbinsr, nbinsr, nbinsr, nbinsphi, nbinsphi), dtype=nb_complex128)
        # This is equal to phis12, phis13 = np.meshgrid(phis, phis), 
        # but made st it compiles with numba
        phis12 = np.empty((nbinsphi, nbinsphi))
        phis13 = np.empty((nbinsphi, nbinsphi))
        for i in range(nbinsphi):
            for j in range(nbinsphi):
                phis12[i, j] = phis[j]
                phis13[i, j] = phis[i]
        for elb in prange(nbinsr*nbinsr*nbinsr):
            elb1 = int(elb//(nbinsr*nbinsr))
            elb2 = int((elb-elb1*nbinsr*nbinsr)//nbinsr)
            elb3 = elb%nbinsr
            _y1 = normys_centers[elb1]
            _dbin1 = normys_edges[elb1+1] - normys_edges[elb1]
            _y2 = normys_centers[elb2]
            _dbin2 = normys_edges[elb2+1] - normys_edges[elb2]
            _y3 = normys_centers[elb3]
            _dbin3 = normys_edges[elb3+1] - normys_edges[elb3]
            _dbinphi = phis[1] - phis[0]
            q1 = - 1./4. * ( + 1*_y1 + 1*_y2*np.exp(1j*phis12) + 1*_y3*np.exp(1j*phis13) )
            q2 = + 1./4. * ( + 3*_y1 - 1*_y2*np.exp(1j*phis12) - 1*_y3*np.exp(1j*phis13) )
            q3 = + 1./4. * ( - 1*_y1 + 3*_y2*np.exp(1j*phis12) - 1*_y3*np.exp(1j*phis13) )
            q4 = + 1./4. * ( - 1*_y1 - 1*_y2*np.exp(1j*phis12) + 3*_y3*np.exp(1j*phis13) )
            q1c=q1.conj(); q2c=q2.conj(); q3c=q3.conj(); q4c=q4.conj()
            absq1_sq=np.abs(q1)**2; absq2_sq=np.abs(q2)**2; absq3_sq=np.abs(q3)**2; absq4_sq=np.abs(q4)**2
            absq12_sq=absq1_sq*absq2_sq; absq13_sq=absq1_sq*absq3_sq; absq14_sq=absq1_sq*absq4_sq
            absq23_sq=absq2_sq*absq3_sq; absq24_sq=absq2_sq*absq4_sq; absq34_sq=absq3_sq*absq4_sq
            absq1p2_sq=np.abs(q1+q2)**2; absq1p3_sq=np.abs(q1+q3)**2; absq1p4_sq=np.abs(q1+q4)**2
            q12c=q1c*q2c; q13c=q1c*q3c; q14c=q1c*q4c; q23c=q2c*q3c; q24c=q2c*q4c; q34c=q3c*q4c
            _2p1=q1**2/absq1_sq; _2p2=q2**2/absq2_sq; _2p3=q3**2/absq3_sq; _2p4=q4**2/absq4_sq
            q123=q1*q2*q3; q124=q1*q2*q4; q134=q1*q3*q4; q234=q2*q3*q4
            _measures = _y1*_y2*_y3*_dbin1*_dbin2*_dbin3/(R_ap6) * (_dbinphi/(2*np.pi))**2
            _exp = np.exp(-(absq1_sq+absq2_sq+absq3_sq+absq4_sq)/(2*R_ap2))
            nextF0 = absq1_sq*absq2_sq*absq3_sq*absq4_sq/(64*R_ap8)
            nextF1_1 = (q1c*q234*(q23c+q24c+q34c))/(32*R_ap6)
            nextF2_1 = (q2c*q134*(q13c+q14c+q34c))/(32*R_ap6)
            nextF3_1 = (q3c*q124*(q12c+q24c+q14c))/(32*R_ap6)
            nextF4_1 = (q4c*q123*(q23c+q12c+q13c))/(32*R_ap6)
            nextF1_2 = _2p1.conj()/(128*R_ap4) * ( absq23_sq*_2p4 + absq24_sq*_2p3 + absq34_sq*_2p2 -4*q1c*q234 )
            nextF2_2 = _2p2.conj()/(128*R_ap4) * ( absq13_sq*_2p4 + absq14_sq*_2p3 + absq34_sq*_2p1 -4*q2c*q134 )
            nextF3_2 = _2p3.conj()/(128*R_ap4) * ( absq12_sq*_2p4 + absq24_sq*_2p1 + absq14_sq*_2p2 -4*q3c*q124 )
            nextF4_2 = _2p4.conj()/(128*R_ap4) * ( absq23_sq*_2p1 + absq12_sq*_2p3 + absq13_sq*_2p2 -4*q4c*q123 )
            nextF5_1 = - q12c*q3*q4*absq1p2_sq/(32*R_ap6)
            nextF6_1 = - q13c*q2*q4*absq1p3_sq/(32*R_ap6)
            nextF7_1 = - q14c*q3*q2*absq1p4_sq/(32*R_ap6)
            nextF5_2 = (_2p1*_2p2).conj()*_2p3*_2p4/(128*R_ap4) * ( absq1p2_sq**2 - 6*R_ap2*absq1p2_sq + 3*R_ap4 )
            nextF6_2 = (_2p1*_2p3).conj()*_2p2*_2p4/(128*R_ap4) * ( absq1p3_sq**2 - 6*R_ap2*absq1p3_sq + 3*R_ap4 )
            nextF7_2 = (_2p1*_2p4).conj()*_2p3*_2p2/(128*R_ap4) * ( absq1p4_sq**2 - 6*R_ap2*absq1p4_sq + 3*R_ap4 )
            F0[elb1,elb2,elb3] = _measures * _exp * nextF0
            F1[elb1,elb2,elb3] = _measures * _exp * (nextF0 + nextF1_1 + nextF1_2)
            F2[elb1,elb2,elb3] = _measures * _exp * (nextF0 + nextF2_1 + nextF2_2)
            F3[elb1,elb2,elb3] = _measures * _exp * (nextF0 + nextF3_1 + nextF3_2)
            F4[elb1,elb2,elb3] = _measures * _exp * (nextF0 + nextF4_1 + nextF4_2)
            F5[elb1,elb2,elb3] = _measures * _exp * (nextF0 + nextF5_1 + nextF5_2)
            F6[elb1,elb2,elb3] = _measures * _exp * (nextF0 + nextF6_1 + nextF6_2)
            F7[elb1,elb2,elb3] = _measures * _exp * (nextF0 + nextF7_1 + nextF7_2)
                    
        return F0, F1, F2, F3, F4, F5, F6, F7
    
    
    # [Debug] Gaussian 4pcf from analytic 2pcf
    def gauss4pcf_analytic(self, theta1, theta2, theta3, xip_arr, xim_arr, thetamin_xi, thetamax_xi, dtheta_xi):
        gausss_4pcf = np.zeros(8*len(self.phis[0])*len(self.phis[0]),dtype=np.complex128)
        self.clib.gauss4pcf_analytic(theta1.astype(np.float64), 
                                     theta2.astype(np.float64),
                                     theta3.astype(np.float64),
                                     self.phis[0].astype(np.float64), np.int32(len(self.phis[0])),
                                     xip_arr.astype(np.float64), xim_arr.astype(np.float64),
                                     thetamin_xi, thetamax_xi, dtheta_xi,
                                     gausss_4pcf)
        return gausss_4pcf
    
    # [Debug] Disconnected part of Map^4 from analytic 2pcf
    def Map4analytic(self,mapradii, xip_spl, xim_spl, thetamin_xi, thetamax_xi, ntheta_xi, 
                     nsubsample_filter=1, batchsize=None):
        
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
            print("Using batchsize of %i for radial bins"%batchsize)
        nbatches = np.int32(_nbinsr3/batchsize)
        thetacombis_batches = np.arange(_nbinsr3).astype(np.int32)
        cumnthetacombis_batches = (np.arange(nbatches+1)*_nbinsr3/(nbatches)).astype(np.int32)
        nthetacombis_batches = (cumnthetacombis_batches[1:]-cumnthetacombis_batches[:-1]).astype(np.int32)
        cumnthetacombis_batches[-1] = _nbinsr3
        nthetacombis_batches[-1] = _nbinsr3-cumnthetacombis_batches[-2]
        thetacombis_batches = thetacombis_batches.flatten().astype(np.int32)
        nbatches = len(nthetacombis_batches)
        ## Optimize this by assigning a cost to each theta for the Gn computation.
        ## Todo: Update to reflect the expected cost...here I just claim that cost ~ theta. In reality, this will
        ##       somewhat depend on the chosen algorithm and the binning accuracy in real space and in multipole
        ##       space with a scaling ranging from theta^0 to theta^2. Also, for each batch the cost for Gn is 
        ##       somewhat bound by the larges theta value anyways...
        ## Note that for large radii we will most likely have all radii in each batch, meaning that we have to
        ## allocate all the Gn anyways, meaning that this optimization does not help a lot...
        #thetacombis_batches = np.asarray(
        #    tuple(product(np.arange(self.nbinsr), np.arange(self.nbinsr), np.arange(self.nbinsr))))
        #cost = self.bin_edges[1:] 
        #costarr = np.max(cost[nrcombis],axis=1)
        #costarr_sortinds = np.argsort(costarr)
        #_, _, batch_lengths, batch_indices = split_array_into_batches(costarr[costarr_sortinds], nthreads)
        #for elbatch in range(nthreads):
        #    thisinds = costarr_sortinds[batch_indices[elbatch]]
        args_4pcfsetup = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), 
                          self.phis[0].astype(np.float64), 
                          (self.phis[0][1]-self.phis[0][0])*np.ones(_nphis, dtype=np.float64), _nphis,)
        args_thetas = (thetacombis_batches, nthetacombis_batches, cumnthetacombis_batches, nbatches, )
        args_map4 = (mapradii, np.int32(len(mapradii)), )
        thetas_xi = np.linspace(thetamin_xi,thetamax_xi,ntheta_xi+1)
        args_xi = (xip_spl(thetas_xi), xim_spl(thetas_xi), thetamin_xi, thetamax_xi, ntheta_xi, nsubsample_filter, )
        args = (*args_4pcfsetup,
                *args_thetas,
                np.int32(self.nthreads),
                *args_map4,
                *args_xi,
                M4correlators)
        func = self.clib.alloc_notomoMap4_analytic
        
        if True:
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
        
        return M4correlators.reshape((8,self.nzcombis,len(mapradii)))
    
    @staticmethod
    def __gen_indices_Ups(nmax):
        """ List of flattened indices corresponding to selection """
        nmax_alloc = 2*(nmax+1)+1
        reconstructed = np.zeros((2*nmax_alloc+1,2*nmax_alloc+1),dtype=int)
        for _n2 in range(-nmax-1,nmax+2):
            for _n3 in range(-nmax-1,nmax+2):
                reconstructed[nmax_alloc+_n2,nmax_alloc+_n3] += 1
        for _n3 in range(-2*nmax,-nmax-1):
            for _n2 in range(-nmax-1-_n3,nmax+2):
                reconstructed[nmax_alloc+_n2,nmax_alloc+_n3] += 1
        for _n2 in range(-2*nmax-1,-nmax-1):
            for _n3 in range(-nmax-1-_n2,nmax+2):
                reconstructed[nmax_alloc+_n2,nmax_alloc+_n3] += 1
        for _n3 in range(nmax+2,2*nmax+1):
            for _n2 in range(-nmax-1,nmax+2-_n3):
                reconstructed[nmax_alloc+_n2,nmax_alloc+_n3] += 1
        for _n2 in range(nmax+2,2*nmax+2):
            for _n3 in range(-nmax-1,nmax+2-_n2):
                reconstructed[nmax_alloc+_n2,nmax_alloc+_n3] += 1
        _shape = reconstructed.shape
        _inds = np.argwhere((reconstructed>0).flatten())[:,0].astype(np.int32)
        _n2s = np.argwhere(reconstructed>0)[:,0].astype(np.int32)
        _n3s = np.argwhere(reconstructed>0)[:,1].astype(np.int32)
        return _shape, _inds, _n2s, _n3s
    
    @staticmethod
    def __gen_thetacombis(nthetas, ordered=True):
        allelbs = []
        nbinsr3 = 0
        cutlo_2 = 0
        cutlo_3 = 0
        for elb1 in range(self.nbinsr):
            if ordered:cutlo_2=elb1
            for elb2 in range(cutlo_2,self.nbinsr):
                if ordered:cutlo_3=elb2
                for elb3 in range(cutlo_3,self.nbinsr):
                    allelbs.append([elb1,elb2,elb3])
                    nbinsr3 += 1
        allelbs = np.asarray(allelbs,dtype=np.int32)
        if batchsize is None:
            batchsize = min(nbinsr3,min(self.thetabatchsize_max,nbinsr3/self.nthreads))
            print("Using batchsize of %i for radial bins"%batchsize)
        if batchsize==self.thetabatchsize_max:
            nbatches = np.int32(np.ceil(nbinsr3/batchsize))
        else:
            nbatches = np.int32(nbinsr3/batchsize)
        thetacombis_batches = np.arange(nbinsr3).astype(np.int32)
        cumnthetacombis_batches = (np.arange(nbatches+1)*nbinsr3/(nbatches)).astype(np.int32)
        nthetacombis_batches = (cumnthetacombis_batches[1:]-cumnthetacombis_batches[:-1]).astype(np.int32)
        cumnthetacombis_batches[-1] = nbinsr3
        nthetacombis_batches[-1] = nbinsr3-cumnthetacombis_batches[-2]
        thetacombis_batches = thetacombis_batches.flatten().astype(np.int32)
        nbatches = len(nthetacombis_batches)
        
        return nbinsr3, allelbs, thetacombis_batches, cumnthetacombis_batches, nthetacombis_batches, nbatches
            
        
    

            
    
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
            
        return map4 
    
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
        elif self.method in ["Tree", "BaseTree", "DoubleTree"]:
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
                #print("Doing Tree")
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
            if self.method in ["BaseTree", "DoubleTree"]:
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
                if self.method=="BaseTree":
                    #print("Doing BaseTree")
                    func = self.clib.alloc_triplets_basetree_xipxipcov 
                if self.method=="DoubleTree":
                    #print("Doing DoubleTree")
                    func = self.clib.alloc_triplets_doubletree_xipxipcov 

        #for elarg, arg in enumerate(args):
        #    print(elarg,arg) 
        
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.wwcounts = wwcounts.reshape(szzr)
        self.w2wcounts = w2wcounts.reshape(szzr)
        self.npcf_multipoles = w2wwtriplets.reshape(sc)
        self.npcf_multipoles_norm = wwwtriplets.reshape(sn)
        self.projection = None 
        
    def multipoles2npcf(self, integrated=False):
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
                if integrated:
                    fac0 = self.phi[1] - self.phi[0]
                    facn = 2./n * np.sin(n*fac0/2.)
                else:
                    fac0 = 1
                    facn = 1
                tmp += _const * self.npcf_multipoles[:,eln].astype(complex) * fac0
                tmpnorm += _const * self.npcf_multipoles_norm[:,eln].astype(complex) * fac0
                if n>0:
                    tmp += _const.conj() * facn * self.npcf_multipoles[eln][ztiler].astype(complex).transpose(0,2,1) 
                    tmpnorm += _const.conj() * facn * self.npcf_multipoles_norm[eln][ztiler].astype(complex).transpose(0,2,1)
            self.npcf[...,elphi] = tmp
            self.npcf_norm[...,elphi] = tmpnorm