import ctypes as ct
import glob
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path


__all__ = ["BinnedNPCF"]
        
################################################
## BASE CLASSES FOR NPCF AND THEIR MULTIPOLES ##
################################################        
class BinnedNPCF:
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
    shuffle_pix: int, optional
        Choice of how to define centers of the cells in the spatial hash structure.
        Defaults to ``0``, i.e. position at pixel center of mass.
    tree_resos: list, optional
        The cell sizes of the hierarchical spatial hash structure
    tree_redges: list, optional
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
    verbosity: int, optional
        The level of verbosity during the computation. Level 0: No verbosity, 1: Progress verbosity
        on python layer, 2: Progress verbosity also on C level, 3: Debug verbosity. Defaults to ``0``.
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
        
    def __init__(self, order, spins, n_cfs, min_sep, max_sep, nbinsr=None, binsize=None, nbinsphi=100, 
                 nmaxs=30, method="DoubleTree", multicountcorr=True, shuffle_pix=0,
                 tree_resos=[0,0.25,0.5,1.,2.], tree_redges=None, rmin_pixsize=20, 
                 resoshift_leafs=0, minresoind_leaf=None, maxresoind_leaf=None,  
                 methods_avail=["Discrete", "Tree", "BaseTree", "DoubleTree"], verbosity=0, nthreads=16):
        
        self.order = int(order)
        self.n_cfs = int(n_cfs)
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbinsphi = nbinsphi
        self.nmaxs = nmaxs
        self.method = method
        self.multicountcorr = int(multicountcorr)
        self.shuffle_pix = shuffle_pix
        self.methods_avail = methods_avail
        self.tree_resos = np.asarray(tree_resos, dtype=np.float64)
        self.tree_nresos = int(len(self.tree_resos))
        self.tree_redges = tree_redges
        self.rmin_pixsize = rmin_pixsize
        self.resoshift_leafs = resoshift_leafs
        self.minresoind_leaf = minresoind_leaf
        self.maxresoind_leaf = maxresoind_leaf
        self.verbosity = np.int32(verbosity)
        self.nthreads = np.int32(max(1,nthreads))
        
        self.tree_resosatr = None
        self.bin_centers = None
        self.bin_centers_mean = None
        self.phis = [None]*self.order
        self.dphis = [None]*self.order
        self.npcf = None
        self.npcf_norm = None
        self.npcf_multipoles = None
        self.npcf_multipoles_norm = None
        self.is_edge_corrected = False
        self._verbose_python = self.verbosity > 0
        self._verbose_c = self.verbosity > 1
        self._verbose_debug = self.verbosity > 2
        
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
        # Setup variable for tree estimator according to input
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
        # Update tree resolutions to make sure that `tree_redges` is monotonous
        # (This is i.e. not fulfilled for a default tree setup and a large value of `rmin`)
        _resomin = self.tree_resosatr[0]
        _resomax = self.tree_resosatr[-1]
        self._updatetree(self.tree_resos[_resomin:_resomax+1])
            
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
            self.dphis[elp] = _[1:] - _[:-1] 
          
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
        
        ## Second order scalar-scalar statistics ##
        if self.order==2 and np.array_equal(self.spins, np.array([0, 0], dtype=np.int32)):
            self.clib.alloc_nn_doubletree.restype = ct.c_void_p
            self.clib.alloc_nn_doubletree.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32, ct.c_int32, p_f64, p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, 
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
                p_i32, ct.c_int32, p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, 
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
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Tree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_tree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_tree_ggg.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Basetree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_basetree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_basetree_ggg.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, p_i32, ct.c_int32, 
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, ct.c_int32, p_i32, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Doubletree-based estimator of third-order shear correlation function
            self.clib.alloc_Gammans_doubletree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_ggg.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32, ct.c_int32, p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_f64,
                p_i32, p_i32, p_i32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, ct.c_int32, p_i32, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Conversion between 3pcf multipoles and 3pcf
            self.clib.multipoles2npcf_ggg.restype = ct.c_void_p
            self.clib.multipoles2npcf_ggg.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32, 
                p_f64, ct.c_int32, p_f64, ct.c_int32, ct.c_int32,
                ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # Change projection of 3pcf between x and centroid
            self.clib._x2centroid_ggg.restype = ct.c_void_p
            self.clib._x2centroid_ggg.argtypes = [
                p_c128, ct.c_int32, 
                p_f64, ct.c_int32, p_f64, ct.c_int32, ct.c_int32]
            
        ## Third-order source-lens-lens statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([2, 0, 0], dtype=np.int32)):
            # Discrete estimator of third-order source-lens-lens (G3L) correlation function
            self.clib.alloc_Gammans_discrete_GNN.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_GNN.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Doubletree-based estimator of third-order source-lens-lens (G3L) correlation function
            self.clib.alloc_Gammans_doubletree_GNN.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_GNN.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                p_f64, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
        ## Third-order lens-source-source statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([0, 2, 2], dtype=np.int32)):
            # Discrete estimator of third-order lens-source-source correlation function
            self.clib.alloc_Gammans_discrete_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_NGG.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
                        
            self.clib.alloc_Gammans_tree_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_tree_NGG.argtypes = [
                ct.c_int32, p_f64, 
                p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, p_i32,
                p_f64, p_f64, p_f64, p_f64,  p_i32, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)]             
        
            self.clib.alloc_Gammans_doubletree_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_NGG.argtypes = [
                ct.c_int32, ct.c_int32, p_f64, p_f64, p_f64, 
                ct.c_int32, ct.c_int32, ct.c_int32, 
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                p_f64, p_f64, p_f64, p_f64, p_i32, p_i32, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
        
        ## Fourth-order counts-counts-counts-counts statistics ##
        if self.order==4 and np.array_equal(self.spins, np.array([0, 0, 0, 0], dtype=np.int32)):

            # Tree estimator of non-tomographic fourth-order counts correlation function
            self.clib.alloc_notomoGammans_tree_nnnn.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_tree_nnnn.argtypes = [
                p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32,  ct.c_int32, 
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64,
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,  ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128)] 

            # Tree-based estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoNap4_tree_nnnn.restype = ct.c_void_p
            self.clib.alloc_notomoNap4_tree_nnnn.argtypes = [
                p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, ct.c_int32, p_f64, p_f64, ct.c_int32,
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_int32, p_f64, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128),
                ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]

            # Transformation between 4PCF from multipole-basis tp real-space basis for a fixed
            # combination of radial bins
            self.clib.multipoles2npcf_nnnn_singletheta.restype = ct.c_void_p
            self.clib.multipoles2npcf_nnnn_singletheta.argtypes = [
                p_c128, ct.c_int32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_double, 
                p_f64, p_f64, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]
            
        ## Fourth-order galaxy-galaxy-galaxy-shear statistics ##
        if self.order==4 and np.array_equal(self.spins, np.array([2, 0, 0, 0], dtype=np.int32)):

            # Discrete  estimator of non-tomographic GNNN statistics
            self.clib.alloc_notomoGammans_discrete_gnnn.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_discrete_gnnn.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                p_f64, p_f64, p_f64, ct.c_int32,
                p_i32, p_i32, p_i32, ct.c_int32,
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64),
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]

            # Tree-based estimator of non-tomographic GNNN statistics
            self.clib.alloc_notomoGammans_tree_gnnn.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_tree_gnnn.argtypes = [
                ct.c_int32, p_f64,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
                p_i32, ct.c_int32,
                ct.c_int32, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]

            # Tree-based estimator of non-tomographic MapNap^3 statistics (low-mem)
            self.clib.alloc_notomoMapNap3_tree_gnnn.restype = ct.c_void_p
            self.clib.alloc_notomoMapNap3_tree_gnnn.argtypes = [
                ct.c_int32, p_f64,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                p_f64, p_f64, p_f64, p_i32,
                p_i32, p_i32, p_i32, p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, ct.c_int32, p_f64, p_f64, ct.c_int32,
                p_i32, p_i32, p_i32, ct.c_int32,
                ct.c_int32, p_f64, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128),
                ct.c_int32, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # Transformaton between 4pcf multipoles and MN3 correlators of MapNap3 statistics
            self.clib.fourpcfmultipoles2MN3correlators.restype = ct.c_void_p
            self.clib.fourpcfmultipoles2MN3correlators.argtypes = [
                ct.c_int32, ct.c_int32,
                p_f64, p_f64, ct.c_int32,
                p_f64, ct.c_int32,
                p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_int32, 
                p_c128, p_c128, np.ctypeslib.ndpointer(dtype=np.complex128)]

            # Transformation between G4L from multipole-basis tp real-space basis for a fixed
            # combination of radial bins
            self.clib.multipoles2npcf_gnnn_singletheta.restype = ct.c_void_p
            self.clib.multipoles2npcf_gnnn_singletheta.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_double, 
                p_f64, p_f64, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]

            self.clib.multipoles2npcf_gnnn_singletheta_nconvergence.restype = ct.c_void_p
            self.clib.multipoles2npcf_gnnn_singletheta_nconvergence.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_double, 
                p_f64, p_f64, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]

            self.clib.alloc_notomoMapNap3_corrections.restype = ct.c_void_p
            self.clib.alloc_notomoMapNap3_corrections.argtypes = [
                p_f64, p_f64, ct.c_int32, p_f64, p_f64, ct.c_int32, ct.c_int32,
                ct.c_int32, p_f64, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128), 
                ct.c_int32, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128)]

            self.clib.alloc_notomoMapNap3_analytic.restype = ct.c_void_p
            self.clib.alloc_notomoMapNap3_analytic.argtypes = [
                ct.c_double, ct.c_double, ct.c_int32, p_f64, p_f64, ct.c_int32, ct.c_int32,
                p_i32, p_i32, p_i32, ct.c_int32,
                ct.c_int32, p_f64, ct.c_int32,
                p_f64, p_f64, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]

            self.clib.gtilde4pcf_corrections.restype = ct.c_void_p
            self.clib.gtilde4pcf_corrections.argtypes = [
                ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, p_f64, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32, p_f64, p_c128, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]

            self.clib.gtilde4pcf_analytic_integrated.restype = ct.c_void_p
            self.clib.gtilde4pcf_analytic_integrated.argtypes = [
                ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, p_f64, ct.c_int32, p_f64, ct.c_int32,
                p_f64, p_f64, ct.c_double, ct.c_double, ct.c_double, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]
        
        ## Fourth-order shear-shear-shear-shear statistics ##
        if self.order==4 and np.array_equal(self.spins, np.array([2, 2, 2, 2], dtype=np.int32)):
            
            # Discrete estimator of non-tomographic fourth-order shear correlation function
            self.clib.alloc_notomoGammans_discrete_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_discrete_gggg.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,  ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 

            # Tree estimator of non-tomographic fourth-order shear correlation function
            self.clib.alloc_notomoGammans_tree_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_tree_gggg.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
                p_i32,  ct.c_int32, 
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64,
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,  ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)] 
            
            # Discrete estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoMap4_disc_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_disc_gggg.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, p_f64, p_f64, ct.c_int32,
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_int32, ct.c_int32, ct.c_int32, p_f64, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128),
                ct.c_int32, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64), 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # Tree-based estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoMap4_tree_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_tree_gggg.argtypes = [
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
                ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
                p_i32, ct.c_int32, p_f64, p_f64, ct.c_int32,
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64, p_f64, p_f64, p_f64,
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
                p_i32, p_i32, p_i32, ct.c_int32, 
                ct.c_int32, ct.c_int32, ct.c_int32, p_f64, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128),
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
            
            # Transformation between 4PCF from multipole-basis tp real-space basis for a fixed
            # combination of radial bins. Explicitly checks convergence for orders of multipoles included
            self.clib.multipoles2npcf_gggg_singletheta_nconvergence.restype = ct.c_void_p
            self.clib.multipoles2npcf_gggg_singletheta_nconvergence.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32, 
                ct.c_double, ct.c_double, ct.c_double, 
                p_f64, p_f64, ct.c_int32, ct.c_int32, 
                ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
 
            # Reconstruction of all 4pcf multipoles from symmetry properties given a set of
            # multipoles with theta1<=theta2<=theta3
            self.clib.getMultipolesFromSymm.restype = ct.c_void_p
            self.clib.getMultipolesFromSymm.argtypes = [
                p_c128, p_c128,
                ct.c_int32, ct.c_int32, p_i32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]
                       
            # Transformaton between 4pcf multipoles and M4 correlators of Map4 statistics
            self.clib.fourpcfmultipoles2M4correlators.restype = ct.c_void_p
            self.clib.fourpcfmultipoles2M4correlators.argtypes = [
                ct.c_int32, ct.c_int32,
                p_f64, p_f64, ct.c_int32,
                p_f64, ct.c_int32,
                p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, 
                ct.c_int32, ct.c_int32, 
                p_c128, p_c128, np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # [DEBUG]: Shear 4pt function in terms of xip/xim
            self.clib.gauss4pcf_analytic.restype = ct.c_void_p
            self.clib.gauss4pcf_analytic.argtypes = [
                ct.c_double, ct.c_double, ct.c_double, p_f64, ct.c_int32,
                p_f64, p_f64, ct.c_double, ct.c_double, ct.c_double, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # [DEBUG]: Shear 4pt function in terms of xip/xim, subsampled within the 4pcf bins
            self.clib.gauss4pcf_analytic_integrated.restype = ct.c_void_p
            self.clib.gauss4pcf_analytic_integrated.argtypes = [
                ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
                p_f64, ct.c_int32,  p_f64, ct.c_int32,
                p_f64, p_f64, ct.c_double, ct.c_double, ct.c_double, 
                np.ctypeslib.ndpointer(dtype=np.complex128)]
            
            # [DEBUG]: Map4 via analytic gaussian 4pcf
            self.clib.alloc_notomoMap4_analytic.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_analytic.argtypes = [
                ct.c_double, ct.c_double, ct.c_int32, p_f64, p_f64, ct.c_int32, ct.c_int32,
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
            
    def _updatetree(self, new_resos):
        
        new_resos = np.asarray(new_resos, dtype=np.float64)
        new_nresos = int(len(new_resos))
        
        new_redges = np.zeros(len(new_resos)+1)
        new_redges[0] = self.min_sep
        new_redges[-1] = self.max_sep
        for elreso, reso in enumerate(new_resos[1:]):
            new_redges[elreso+1] = self.rmin_pixsize*reso
        _tmpreso = 0
        new_resosatr = np.zeros(self.nbinsr, dtype=np.int32)
        for elbin, rbin in enumerate(self.bin_edges[:-1]):
            if rbin > new_redges[_tmpreso+1]:
                _tmpreso += 1
            new_resosatr[elbin] = _tmpreso 

        self.tree_resos = new_resos
        self.tree_nresos = new_nresos
        self.tree_redges = new_redges
        self.tree_resosatr = new_resosatr