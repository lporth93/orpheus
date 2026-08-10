import ctypes as ct
import glob
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path

from .catalog import Catalog
from .utils import convertunits
from .multires_structs import (MultiresoCatalog, NavHash, TreeResoParams,
                               BinningParams, NPCFOutput, FourthParams, ClustCorr)

__all__ = ["BinnedNPCF"]
        
################################################
## BASE CLASSES FOR NPCF AND THEIR MULTIPOLES ##
################################################        
class BinnedNPCF:
    r"""Class of a binned N-point correlation function of various arbitrary tracer catalogs.
    This class contains attributes and methods that can be used across any of its children.

    .. note::
        **Sign convention for polar fields.** Every spin-2 leg is projected in the
        tangential basis,

        .. math::

            \gamma_\mathrm{t} + i\gamma_\times = -\gamma\,e^{-2i\phi} \ ,

        with :math:`\phi` the direction along which the field is projected. A correlator
        built from :math:`k` polar legs therefore carries an overall factor
        :math:`(-1)^k` relative to the raw product of :math:`\gamma\,e^{-2i\phi}`,
        applied once at the level of the accumulated correlator:

        ==========================  =====  ==============
        correlator                  k      :math:`(-1)^k`
        ==========================  =====  ==============
        ``NNCorrelation``           0      :math:`+1`
        ``NGCorrelation``           1      :math:`-1`
        ``GGCorrelation``           2      :math:`+1`
        ``GNNCorrelation``          1      :math:`-1`
        ``NGGCorrelation``          2      :math:`+1`
        ``GGGCorrelation``          3      :math:`-1`
        ``GNNNCorrelation_NoTomo``  1      :math:`-1`
        ``GGGGCorrelation_NoTomo``  4      :math:`+1`
        ==========================  =====  ==============

        As a consequence a pure tangential shear returns a positive
        :math:`\mathrm{Re}\,\xi` from ``NGCorrelation``, and the aperture masses built
        from these correlators are the usual :math:`M_\mathrm{ap} + iM_\times` with no
        further sign handling.

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
    binsize: float, optional
        The logarithmic size of the radial bins for each vertex of the NPCF. If set to
        ``None`` this attribute is inferred from the ``nbinsr`` attribute.
    sep_units: {'rad', 'deg', 'arcmin', 'arcsec'}, optional
        Units of all separation-like quantities. Defaults to arcmin.
    nbinsphi: float, optional
        The number of angular bins for the NPCF in the real-space basis. 
        Defaults to ``100``.
    nmaxs: list, optional
        The largest multipole component considered for the NPCF in the multipole basis. 
        Defaults to ``30``.
    method: str, optional
        The method to be employed for the estimator. Defaults to ``DoubleTree``.
    multicountcorr: bool, optional
        Flag on whether to subtract multiplets in which the same tracer appears more
        than once. Defaults to ``True``.
    shuffle_pix: int, optional
        Choice of how to define centers of the cells in the spatial hash structure.
        Defaults to ``0``, i.e. position at pixel center of mass.
    process_spherical: bool. optional
        Process spherical catalogs using curved-sky geometry and not via flat-sky 
        patches. Defaults to ``False``.
    tree_alpha: float, optional
        Parameter used for autosetting tree resolutions given a catalog. 
    tree_mincellsize: float
        Smallest allowed sidelength for cell in tree. Defaults to ``0.1``.
    tree_maxcellsize: float
        Largest allowed sidelength for cell in tree. Defaults to ``4``.
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
        that leafs will be evaluated at coarser resolutions than the base. Defaults to ``0``.
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
        are specified as: ``(component, zcombi, rbin_1, ..., rbin_N-1, phibin_1, phibin_N-2)``.
    npcf_norm: numpy.ndarray
        The normalization of the natural components of the NPCF in the real space basis. The different axes
        are specified as: ``(zcombi, rbin_1, ..., rbin_N-1, phibin_1, phibin_N-2)``.
    npcf_multipoles: numpy.ndarray
        The natural components of the NPCF in the multipole basis. The different axes
        are specified as: ``(component, zcombi, multipole_1, ..., multipole_N-2, rbin_1, ..., rbin_N-1)``.
    npcf_multipoles_norm: numpy.ndarray
        The normalization of the natural components of the NPCF in the multipole basis. The different axes
        are specified as: ``(zcombi, multipole_1, ..., multipole_N-2, rbin_1, ..., rbin_N-1)``.
    is_edge_corrected: bool
        Flag signifying on whether the NPCF multipoles have been edge-corrected. Defaults to ``False``.
    norm_divisionmask: float
        If the absolute value of the normalisation is smaller the NPCF is set to zero.
    """
        
    def __init__(self, order, spins, n_cfs, min_sep, max_sep, nbinsr=None, binsize=None, sep_units="arcmin", nbinsphi=100, 
                 nmaxs=30, method="DoubleTree", multicountcorr=True, shuffle_pix=0, process_spherical=False,
                 tree_alpha=None, tree_mincellsize=0.1, tree_maxcellsize=4., tree_resos=[0,0.25,0.5,1.,2.], tree_redges=None, rmin_pixsize=20, 
                 resoshift_leafs=0, minresoind_leaf=None, maxresoind_leaf=None,  
                 methods_avail=["Discrete", "Tree", "BaseTree", "DoubleTree"], verbosity=0, nthreads=16,
                 norm_divisionmask=1e-2):

        self.order = int(order)
        self.n_cfs = int(n_cfs)
        self.min_sep = float(min_sep)
        self.max_sep = float(max_sep)
        self.sep_units = sep_units
        assert self.sep_units in ('rad', 'deg', 'arcmin', 'arcsec'), \
            "sep_units must be one of 'rad', 'deg', 'arcmin', 'arcsec'"
        self.nbinsphi = nbinsphi
        self.nmaxs = nmaxs
        self.method = method
        self.multicountcorr = int(multicountcorr)
        self.shuffle_pix = shuffle_pix
        self.process_spherical = bool(process_spherical)
        self.methods_avail = methods_avail
        self.tree_alpha = tree_alpha
        self.tree_mincellsize = tree_mincellsize
        self.tree_maxcellsize = tree_maxcellsize
        self.tree_resos = np.asarray(tree_resos, dtype=np.float64)
        self.tree_nresos = int(len(self.tree_resos))
        self.tree_redges = tree_redges
        self.rmin_pixsize = rmin_pixsize
        self.resoshift_leafs = resoshift_leafs
        self.minresoind_leaf = minresoind_leaf
        self.maxresoind_leaf = maxresoind_leaf
        self.verbosity = np.int32(verbosity)
        self.nthreads = np.int32(max(1,nthreads))
        self.norm_divisionmask = float(norm_divisionmask)

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
            assert(isinstance(binsize, float) or isinstance(binsize, int))
            self.nbinsr = int(np.ceil(np.log(self.max_sep/self.min_sep)/float(binsize)))
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
        self._updatetree(self.tree_resos[_resomin:_resomax+1], include_shifts=False)
            
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
        # Link path of c library from local install path
        target_path = __import__('orpheus').__file__
        self.library_path = str(Path(__import__('orpheus').__file__).parent.absolute())
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        
        p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_i64 = ndpointer(np.int64, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)
        
        ## Second order scalar-scalar statistics ##
        if self.order==2 and np.array_equal(self.spins, np.array([0, 0], dtype=np.int32)):
            # DoubleTree-based estimator of counts 2PCF
            self.clib.alloc_nn_doubletree.restype = ct.c_void_p
            self.clib.alloc_nn_doubletree.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]
        
        ## Second order shear-shear statistics ##
        if self.order==2 and np.array_equal(self.spins, np.array([2, 2], dtype=np.int32)):
            # DoubleTree-based estimator of shear 2PCF
            self.clib.alloc_gg_doubletree.restype = ct.c_void_p
            self.clib.alloc_gg_doubletree.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]
            
        ## Second order position-shear (GGL) statistics ##
        if self.order==2 and np.array_equal(self.spins, np.array([0, 2], dtype=np.int32)):
            # Computation of discrete NG in slabs of 3dbox geometry
            self.clib.ng_slab.restype = ct.c_void_p
            self.clib.ng_slab.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog),
                ct.POINTER(NavHash), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Computation of NG using the Doubletree-approximation
            self.clib.alloc_ng_doubletree.restype = ct.c_void_p
            self.clib.alloc_ng_doubletree.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]
                
        ## Third-order multipole to npcf conversion used by GGG, GNN and NGG ##
        if self.order==3:
            # Conversion of multipole-space-basis 3pcf to real-space basis, assuming cat2==cat3.
            self.clib.multipoles2npcf_third_z1z23.restype = ct.c_void_p
            self.clib.multipoles2npcf_third_z1z23.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
                p_f64, ct.c_int32,
                ct.c_int32, p_i32, p_f64,
                ct.c_int32, ct.c_int32, p_f64,
                ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.complex128),
                np.ctypeslib.ndpointer(dtype=np.complex128)]

        ## Third order density-density-density (clustering) statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([0, 0, 0], dtype=np.int32)):
            # Doubletree-based estimator of NNN (scalar triplet counts).
            self.clib.alloc_nnn_doubletree.restype = ct.c_void_p
            self.clib.alloc_nnn_doubletree.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

        ## Third order shear-shear-shear statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([2, 2, 2], dtype=np.int32)):
            # Discrete estimator of GGG
            self.clib.alloc_Gammans_discrete_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_ggg.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(BinningParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Tree-based estimator of GGG
            self.clib.alloc_Gammans_tree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_tree_ggg.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog),
                ct.POINTER(NavHash), ct.POINTER(TreeResoParams),
                ct.POINTER(BinningParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Basetree-based estimator of GGG
            self.clib.alloc_Gammans_basetree_ggg.restype = ct.c_void_p
            self.clib.alloc_Gammans_basetree_ggg.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]
            
            # Doubletree-based estimator of GGG.
            self.clib.alloc_ggg_doubletree.restype = ct.c_void_p
            self.clib.alloc_ggg_doubletree.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Discrete projected GGG & RRR correlators for 3dbox geometry
            self.clib.alloc_Gammans_slab_GGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_slab_GGG.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(BinningParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]
            
            # Change projection of 3pcf between x and centroid
            self.clib._x2centroid_ggg.restype = ct.c_void_p
            self.clib._x2centroid_ggg.argtypes = [
                p_c128, ct.c_int32,
                p_f64, ct.c_int32, p_f64, ct.c_int32, ct.c_int32]

            # Conversion to third-order aperture-mass statistics
            self.clib.threepcf2M3correlators_ggg.restype = ct.c_void_p
            self.clib.threepcf2M3correlators_ggg.argtypes = [
                p_c128, p_f64, p_f64, ct.c_int32, p_f64, ct.c_int32, ct.c_int32,
                p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.complex128)]

        ## Third-order source-lens-lens statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([2, 0, 0], dtype=np.int32)):
            # Discrete estimator of third-order source-lens-lens (G3L) correlation function
            self.clib.alloc_Gammans_discrete_GNN.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_GNN.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(BinningParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Doubletree-based estimator of third-order source-lens-lens (G3L) correlation function
            self.clib.alloc_Gammans_doubletree_GNN.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_GNN.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Discrete projected GDD, GDR, GDR, RRR correlators for 3dbox geometry
            self.clib.alloc_Gammans_slab_GNN.restype = ct.c_void_p
            self.clib.alloc_Gammans_slab_GNN.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(BinningParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Conversion to third-order aperture-mass statistics (NNM)
            self.clib.threepcf2NNMcorrelators_gnn.restype = ct.c_void_p
            self.clib.threepcf2NNMcorrelators_gnn.argtypes = [
                p_c128, p_f64, p_f64, ct.c_int32, p_f64, ct.c_int32, ct.c_int32,
                p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.complex128)]

        ## Third-order lens-source-source statistics ##
        if self.order==3 and np.array_equal(self.spins, np.array([0, 2, 2], dtype=np.int32)):
            # Discrete estimator of third-order lens-source-source correlation function
            self.clib.alloc_Gammans_discrete_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_discrete_NGG.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(BinningParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Tree-based estimator of third-order lens-source-source correlation function
            self.clib.alloc_Gammans_tree_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_tree_NGG.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Doubletree-based estimator of third-order lens-source-source correlation function
            self.clib.alloc_Gammans_doubletree_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_doubletree_NGG.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Discrete projected DGG, RGG, RRR correlators for 3dbox geometry
            self.clib.alloc_Gammans_slab_NGG.restype = ct.c_void_p
            self.clib.alloc_Gammans_slab_NGG.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), ct.POINTER(NavHash),
                ct.POINTER(BinningParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Conversion to third-order aperture-mass statistics (NMM)
            self.clib.threepcf2NMMcorrelators_ngg.restype = ct.c_void_p
            self.clib.threepcf2NMMcorrelators_ngg.argtypes = [
                p_c128, p_f64, p_f64, ct.c_int32, p_f64, ct.c_int32, ct.c_int32,
                p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32,
                np.ctypeslib.ndpointer(dtype=np.complex128)]

        ## Fourth-order counts-counts-counts-counts statistics ##
        if self.order==4 and np.array_equal(self.spins, np.array([0, 0, 0, 0], dtype=np.int32)):

            # Tree-based estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoNap4_tree_nnnn.restype = ct.c_void_p
            self.clib.alloc_notomoNap4_tree_nnnn.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), 
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                p_f64, ct.c_int32, p_c128, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32,
                p_f64, p_c128, p_c128]

            # Runtime.optimised estimator of non-tomographic NNNN statistics.
            self.clib.alloc_notomoNap4_tree_nnnn_highmem.restype = ct.c_void_p
            self.clib.alloc_notomoNap4_tree_nnnn_highmem.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), 
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                p_f64, ct.c_int32, p_c128, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32,
                p_f64, p_c128, p_c128]

            # Tree-based estimator of the fourth-order count multipoles
            self.clib.alloc_nnnn_tree.restype = ct.c_void_p
            self.clib.alloc_nnnn_tree.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), 
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_double, ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Tree-based estimator of the fourth-order count multipoles in spherical geometry
            self.clib.alloc_nnnn_tree_spherical.restype = ct.c_void_p
            self.clib.alloc_nnnn_tree_spherical.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), 
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_double, ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Doubletree-based estimator of the fourth-order count multipoles
            self.clib.alloc_nnnn_doubletree.restype = ct.c_void_p
            self.clib.alloc_nnnn_doubletree.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), ct.POINTER(TreeResoParams), 
                ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_double, ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Transformation between 4PCF from multipole-basis tp real-space basis for all
            # combination of radial bins
            self.clib.multipoles2npcf_nnnn.restype = ct.c_void_p
            self.clib.multipoles2npcf_nnnn.argtypes = [
                p_c128, ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                p_f64, np.ctypeslib.ndpointer(dtype=np.complex128), ct.c_int32]

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
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Tree-based estimator of non-tomographic GNNN statistics
            self.clib.alloc_notomoGammans_tree_gnnn.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_tree_gnnn.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
                ct.POINTER(FourthParams), ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

            # Tree-based estimator of non-tomographic MapNap^3 statistics (low-mem)
            self.clib.alloc_notomoMapNap3_tree_gnnn.restype = ct.c_void_p
            self.clib.alloc_notomoMapNap3_tree_gnnn.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.POINTER(ClustCorr),
                p_f64, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
                p_f64, p_c128, p_c128, p_c128, p_c128, p_c128]

            # Transformaton between 4pcf multipoles and MN3 correlators of MapNap3 statistics
            self.clib.fourpcfmultipoles2MN3correlators.restype = ct.c_void_p
            self.clib.fourpcfmultipoles2MN3correlators.argtypes = [
                ct.c_int32, ct.c_int32,
                p_f64, p_f64, ct.c_int32,
                p_f64, p_f64, ct.c_int32,
                p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_int32,
                ct.POINTER(ClustCorr),
                p_c128, p_c128, np.ctypeslib.ndpointer(dtype=np.complex128)]

            # Transformation between G4L from multipole-basis to real-space basis for a fixed
            # combination of radial bins
            self.clib.multipoles2npcf_gnnn_singletheta.restype = ct.c_void_p
            self.clib.multipoles2npcf_gnnn_singletheta.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32,
                ct.c_double, ct.c_double, ct.c_double,
                p_f64, p_f64, ct.c_int32, ct.c_int32,
                ct.POINTER(ClustCorr),
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]

            self.clib.multipoles2npcf_gnnn_singletheta_nconvergence.restype = ct.c_void_p
            self.clib.multipoles2npcf_gnnn_singletheta_nconvergence.argtypes = [
                p_c128, p_c128, ct.c_int32, ct.c_int32,
                ct.c_double, ct.c_double, ct.c_double,
                p_f64, p_f64, ct.c_int32, ct.c_int32,
                ct.POINTER(ClustCorr),
                np.ctypeslib.ndpointer(dtype=np.complex128),np.ctypeslib.ndpointer(dtype=np.complex128)]

            # Full multipole-basis to real-space basis conversion over all theta triples
            self.clib.multipoles2npcf_gnnn.restype = ct.c_void_p
            self.clib.multipoles2npcf_gnnn.argtypes = [
                p_c128, p_c128,
                ct.POINTER(BinningParams), ct.POINTER(FourthParams), ct.POINTER(ClustCorr),
                ct.c_int32,
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
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)] 

            # Tree estimator of non-tomographic fourth-order shear correlation function
            self.clib.alloc_notomoGammans_tree_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoGammans_tree_gggg.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), 
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)] 
            
            # Discrete estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoMap4_disc_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_disc_gggg.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_int32, p_f64, ct.c_int32, p_c128, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32,
                p_f64, p_c128, p_c128, p_c128, p_c128]
            
            # Tree-based estimator of non-tomographic Map^4 statistics (low-mem)
            self.clib.alloc_notomoMap4_tree_gggg.restype = ct.c_void_p
            self.clib.alloc_notomoMap4_tree_gggg.argtypes = [
                ct.POINTER(MultiresoCatalog), ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash), 
                ct.POINTER(TreeResoParams), ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_int32, p_f64, ct.c_int32, p_c128, ct.c_int32, ct.c_int32,
                ct.c_int32, ct.c_int32,
                p_f64, p_c128, p_c128, p_c128, p_c128]
         
            self.clib.multipoles2npcf_gggg.restype = ct.c_void_p
            self.clib.multipoles2npcf_gggg.argtypes = [
                p_c128, p_c128, p_f64, ct.POINTER(BinningParams), ct.POINTER(FourthParams),
                ct.c_int32, ct.c_int32, ct.c_int32, np.ctypeslib.ndpointer(dtype=np.complex128), 
                np.ctypeslib.ndpointer(dtype=np.complex128)]
                        
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
            

    def autoset_tree(self, cat, dpix_grid=2., nside_grid=2048, max_increase=0.1, set_resoshiftleafs=True, set_maxresoindleaf=True):
        """Sets the pixel-sizes for the (multi)hashes given a catalog.

        Sets the smallest pixelsize based on an estimate of the nbar of the catalog.
        """

        assert(isinstance(cat, Catalog))

        # Crude estimate of nbar: Pixelize sky on coarse grid and estimate area by counting nonempty pixels
        if cat.geometry=="flat2d":
            n1 = int(np.ceil(cat.len1 / dpix_grid)) + 1
            n2 = int(np.ceil(cat.len2 / dpix_grid)) + 1
            i1 = np.clip(((cat.pos1 - cat.min1) / dpix_grid).astype(np.int64), 0, n1 - 1)
            i2 = np.clip(((cat.pos2 - cat.min2) / dpix_grid).astype(np.int64), 0, n2 - 1)
            nfilled = len(np.unique(i2 * (n1 + 1) + i1))
            pixarea_arcmin2 = dpix_grid**2
        if cat.geometry=='spherical':
            from healpy import ang2pix, nside2pixarea
            from astropy.coordinates import SkyCoord
            eq = SkyCoord(cat.pos1, cat.pos2, frame='galactic', unit='deg')
            l, b = eq.galactic.l.value, eq.galactic.b.value
            theta = np.radians(90. - b)
            phi = np.radians(l)
            hpx_inds = ang2pix(nside_grid, theta, phi)
            nfilled = len(np.unique(hpx_inds))
            pixarea_arcmin2 = nside2pixarea(nside_grid, degrees=True)*3600
        area_arcmin2 = nfilled * pixarea_arcmin2
        nbar = cat.ngal/area_arcmin2
        nbar_z = nbar/cat.nbinsz

        # Adjust alpha s.t. coarsest resolution reached.
        if self.tree_alpha is None:
            _tmpalpha=0.5
            _tmpreso = _tmpalpha/np.sqrt(nbar_z)
            while _tmpreso<=self.tree_maxcellsize:
                _tmpreso *= 2
            _resc = 2*self.tree_maxcellsize/_tmpreso
            self.tree_alpha = _tmpalpha * _resc
            if (_tmpalpha>2.*self.tree_mincellsize) and (_resc > 1.5):
                self.tree_alpha /= 2.
            
        # Build allowed resos for tree
        basereso = self.tree_alpha/np.sqrt(nbar_z)
        autotree_resos = [0.]
        nextreso = basereso
        while nextreso<=self.tree_mincellsize:
            nextreso *= 2
        while nextreso<=self.tree_maxcellsize:
            autotree_resos.append(nextreso)
            nextreso *= 2
        autotree_resos = np.asarray(autotree_resos)

        # Update the tree
        self._updatetree(autotree_resos, include_shifts=True)

        if self.verbosity>2:
            print("Autosetting tree parameters to")
            print("Tree resolutions:", self.tree_resos, "(alpha: %.4f)"%self.tree_alpha)

        # Check whether we can get finer leaf resos without too much overhead
        # We type out the time complexity of the innermost loops in the Gn/Gamman DoubleTree allocation and increase the 
        # leaf resolution until it reaches at most 10% of the Gamman time complexity.
        # TODO: Check how bad this approximation is. It's probably not perfect, but one should at least make
        # sure to not add much more than 10% overhead...
        if set_resoshiftleafs:
            _areashells = np.pi*(self.tree_redges[1:]**2-self.tree_redges[:-1]**2)
            _nupdateGn = lambda resos: 6*cat.nbinsz*(self.nmaxs[0]+5)*np.sum(_areashells*np.minimum(nbar_z*np.ones_like(resos),1./resos**2))
            _nupdateUpsN = (self.n_cfs+1)*(2*self.nmaxs[0]+1)**(self.order-2)*self.nbinsr**(self.order-1)*cat.nbinsz**self.order
            _nupdatebase = _nupdateGn(self.tree_resos) + _nupdateUpsN
            leafresos = 1.*self.tree_resos
            tmpresoshift = 0
            for shifts in range(len(self.tree_resos)):
                propresos = 1.*leafresos
                propresos[1:] = propresos[:-1]
                _relincrease = (_nupdateGn(propresos)+_nupdateUpsN)/_nupdatebase - 1.
                #if self.verbosity>2:
                #    print(tmpresoshift-1,_relincrease)
                if _relincrease>max_increase:
                    break
                leafresos = propresos
                tmpresoshift -= 1
            self.resoshift_leafs = tmpresoshift
            if self.verbosity>2:
                _fincrease = (_nupdateGn(leafresos)+_nupdateUpsN)/_nupdatebase - 1.
                if self.resoshift_leafs<0:
                    print("Resoshift of leafs: %i (expected runtime increase of %.2f percent)"%(self.resoshift_leafs,100.*_fincrease))
                else:
                    print("No relative shifting of leaf resos wrt base resos")

        # Check whether fixing the largest leaf resolution can be additionally done
        if set_maxresoindleaf:
            leafresos2 = 1.*leafresos
            maxresoind = self.tree_nresos-1
            for propmaxresoind in range(self.maxresoind_leaf+1)[::-1]:
                propresos = np.minimum(leafresos2,leafresos[propmaxresoind]*np.ones_like(self.tree_resos))
                _relincrease = (_nupdateGn(propresos)+_nupdateUpsN)/_nupdatebase - 1.
                #if self.verbosity>2:
                #    print(propmaxresoind,_relincrease)
                if _relincrease>max_increase:
                    break
                leafresos2 = propresos
                maxresoind = propmaxresoind
            self.maxresoind_leaf = maxresoind
            if self.verbosity>2:
                _fincrease = (_nupdateGn(leafresos2)+_nupdateUpsN)/_nupdatebase - 1.
                if self.maxresoind_leaf<(len(self.tree_resos)-1):
                    print("Largest reso index of leafs: %i (expected runtime increase of %.2f percent)"%(self.maxresoind_leaf,100.*_fincrease))
                else:
                    print("Leafs cell size extends up to largest allowed base cellsize.")

    def tree_resos_to_nsides(self):
        """Transforms the instance tree_resos to the closest healpix nsides. 
        Used for multihashing a spherical catalog.
        """
        from healpy import nside2resol
        sep2rad = convertunits(self.sep_units, 'rad')

        def _nside_for(target_rad):
            ns = 1
            while nside2resol(ns) > target_rad and ns < 2**29:
                ns *= 2
            return ns
        
        # Get nsides
        nsides = [0 if self.tree_resos[r]==0. else _nside_for(self.tree_resos[r]*sep2rad)
                    for r in range(self.tree_nresos)]
        # Define size for nside of spatial hash s.t. it will not be too small
        nside_hash = _nside_for(max(self.min_sep, 0.5*self.tree_redges[1])*sep2rad)

        return nsides, nside_hash

            
    def save_divide_npcf(self, npcf, npcf_norm, fill=0., thr=None, use_abs=False):
        """Masked division for obtaining the npcf. 

        All NPCF bins for which the normalisation is smaller than ``thr`` are
        set to ``fill``.
        """
        _thr = self.norm_divisionmask if thr is None else thr
        _cmp = np.abs(npcf_norm) if use_abs else npcf_norm
        return np.divide(npcf, npcf_norm, out=np.full_like(npcf, fill), where=_cmp > _thr)

    def save_divide_bins(self, rsum, norm):
        """Masked division for obtaining the bin centers. Falls back to 
        arithmetic centers for empty bins."""
        arith = 0.5*(self.bin_edges[:-1] + self.bin_edges[1:])
        out = np.empty_like(rsum)
        out[...] = arith
        return np.divide(rsum, norm, out=out, where=np.abs(norm) > self.norm_divisionmask)

    def saveinst(self, path_save, fname, extr_pars=None):
        """Save an instance as a ``.npz`` archive.

        The arrays that are shared for any instance are given here explicitly. 
        Children pass their derived data arrays through the optional ``extr_pars`` dict.
        """
        if not Path(path_save).is_dir():
            raise ValueError('Path to directory does not exist.')

        # Shared parameters
        pars = dict(order=self.order, spins=self.spins, n_cfs=self.n_cfs,
                    min_sep=self.min_sep, max_sep=self.max_sep, nbinsr=self.nbinsr,
                    sep_units=self.sep_units, nbinsphi=self.nbinsphi, nmaxs=self.nmaxs,
                    method=self.method, multicountcorr=self.multicountcorr,
                    shuffle_pix=self.shuffle_pix, process_spherical=self.process_spherical,
                    tree_resos=self.tree_resos, tree_redges=self.tree_redges,
                    rmin_pixsize=self.rmin_pixsize, resoshift_leafs=self.resoshift_leafs,
                    minresoind_leaf=self.minresoind_leaf, maxresoind_leaf=self.maxresoind_leaf,
                    nthreads=self.nthreads, projection=getattr(self, 'projection', None),
                    bin_centers=self.bin_centers, bin_centers_mean=self.bin_centers_mean)

        # Generic measured outputs. Only save those that have been allocated.
        for name in ('npcf', 'npcf_norm', 'npcf_multipoles', 'npcf_multipoles_norm'):
            val = getattr(self, name)
            if val is not None:
                pars[name] = val

        # Update all the extra pars and save
        if extr_pars: pars.update(extr_pars)
        np.savez(path_save+fname, **pars)

    @classmethod
    def loadinst(cls, path_save, fname):
        """Reconstruct an instance from an archive written by ``saveinst``.
        """
        import inspect
        fpath = path_save + fname
        if not fpath.endswith('.npz'): fpath += '.npz'
        with np.load(fpath, allow_pickle=True) as d:

            # Little helper that reads either array or a non-array from a .npz file
            _val = lambda v: v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v

            # Alloc constructor arguments every child forwards to BinnedNPCF.__init__.
            ctor_keys = ('min_sep', 'max_sep', 'nbinsr', 'sep_units', 'nbinsphi', 'nmaxs',
                         'method', 'multicountcorr', 'shuffle_pix', 'process_spherical',
                         'tree_resos', 'rmin_pixsize', 'resoshift_leafs', 'minresoind_leaf',
                         'maxresoind_leaf', 'nthreads')
            ctor = {k: _val(d[k]) for k in ctor_keys if k in d.files}

            # Alloc all child-specific constructor names that are not fixed internally
            params = inspect.signature(cls.__init__).parameters
            for k in d.files:
                if k in params and k not in ctor:
                    ctor[k] = _val(d[k])

            # Init the instances
            inst = cls(**ctor)

            # Add the remaining attributes, like counts, npcfs, ..., saved after 
            # the __init__, ie all that did require a .process() call.
            skip = set(ctor) | {'order', 'spins', 'n_cfs', 'tree_redges'}
            for k in d.files:
                if k not in skip:
                    setattr(inst, k, _val(d[k]))

        return inst

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
            
    def _updatetree(self, new_resos, include_shifts=True):
        
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

        if include_shifts:
            if self.resoshift_leafs<0:
                self.resoshift_leafs = max(-(self.tree_nresos-1),self.resoshift_leafs)
            else:
                self.resoshift_leafs = min((self.tree_nresos-1),self.resoshift_leafs)
            self.minresoind_leaf = min(self.minresoind_leaf, self.tree_nresos-1)
            self.maxresoind_leaf = min(self.maxresoind_leaf, self.tree_nresos-1)