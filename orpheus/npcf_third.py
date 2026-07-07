import numpy as np
import ctypes as ct
from functools import reduce
from numba import jit, prange
from numba import config as nb_config
from numba import complex128 as nb_complex128
import operator
from pathlib import Path
from scipy.interpolate import interp1d

from .npcf_base import BinnedNPCF
from .catalog import ScalarTracerCatalog
from .utils import convertunits
from .multires_structs import (build_catalog_struct, build_navhash_struct,
                               build_flat_catalog_struct, build_flat_navhash_struct,
                               build_slab_catalog_struct, build_slab_navhash_struct,
                               build_tree_params_struct, build_binning_struct,
                               build_npcf_output)

__all__ = ["GGGCorrelation", "GNNCorrelation", "NGGCorrelation"]

class GGGCorrelation(BinnedNPCF):
    r""" Class containing methods to measure and obtain statistics that are built
        from third-order shear correlation functions.

        Attributes
        ----------
        n_cfs: int
            The number of independent components of the NPCF.
        min_sep: float
            The smallest distance of each vertex for which the NPCF is computed.
        max_sep: float
            The largest distance of each vertex for which the NPCF is computed.

        Notes
        -----
        Inherits all other parameters and attributes from :class:`BinnedNPCF`.
        Additional child-specific parameters can be passed via ``kwargs``.
        Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.
        """
    
    def __init__(self, n_cfs, min_sep, max_sep, process_spherical=False, **kwargs):

        super().__init__(order=3, spins=np.array([2,2,2], dtype=np.int32), n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
        # Native curved-sky GGG (geodesic distance + nested-HEALPix query_disc +
        # spin-2 geodesic projection) requires process_spherical + method="DoubleTree";
        # otherwise a spherical catalog must first be decomposed into patches.
        # See Catalog.multihash_spherical / alloc_ggg_doubletree (metric=SPHERICAL).
        self.process_spherical = bool(process_spherical)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X", "Centroid"]
        self.nbinsz = None
        self.nzcombis = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        self.project["X"]["Centroid"] = self._x2centroid

    def saveinst(self, path_save, fname):

        if not Path(path_save).is_dir():
            raise ValueError('Path to directory does not exist.')
        
        np.savez(path_save+fname,
                 nbinsz=self.nbinsz,
                 min_sep=self.min_sep,
                 max_sep=self.max_sep,
                 binsr=self.nbinsr,
                 nbinsphi=self.nbinsphi,
                 nmaxs=self.nmaxs,
                 method=self.method,
                 multicountcorr=self.multicountcorr,
                 shuffle_pix=self.shuffle_pix,
                 tree_resos=self.tree_resos,
                 rmin_pixsize=self.rmin_pixsize,
                 resoshift_leafs=self.resoshift_leafs,
                 minresoind_leaf=self.minresoind_leaf,
                 maxresoind_leaf=self.maxresoind_leaf,
                 nthreads=self.nthreads,
                 bin_centers=self.bin_centers,
                 npcf_multipoles=self.npcf_multipoles,
                 npcf_multipoles_norm=self.npcf_multipoles_norm)

    def __process_patches(self, cat, dotomo=True, rotsignflip=False, apply_edge_correction=False, adjust_tree=False, 
                        save_patchres=False, save_filebase="", keep_patchres=False):

        if save_patchres:
            if not Path(save_patchres).is_dir():
                raise ValueError('Path to directory does not exist.')

        for elp in range(cat.npatches):
            if self._verbose_python:
                print('Doing patch %i/%i'%(elp+1,cat.npatches))

            # Compute statistics on patch
            pcat = cat.frompatchind(elp,rotsignflip=rotsignflip)
            pcorr = GGGCorrelation(
                n_cfs=self.n_cfs,
                min_sep=self.min_sep,
                max_sep=self.max_sep,
                nbinsr=self.nbinsr,
                nbinsphi=self.nbinsphi,
                nmaxs=self.nmaxs,
                method=self.method,
                multicountcorr=self.multicountcorr,
                shuffle_pix=self.shuffle_pix,
                tree_resos=self.tree_resos,
                rmin_pixsize=self.rmin_pixsize,
                resoshift_leafs=self.resoshift_leafs,
                minresoind_leaf=self.minresoind_leaf,
                maxresoind_leaf=self.maxresoind_leaf,
                nthreads=self.nthreads,
                verbosity=self.verbosity)
            pcorr.process(pcat, dotomo=dotomo)
            
            # Update the total measurement
            if elp == 0:
                self.nbinsz = pcorr.nbinsz
                self.nzcombis = pcorr.nzcombis
                self.bin_centers = np.zeros_like(pcorr.bin_centers)
                self.npcf_multipoles = np.zeros_like(pcorr.npcf_multipoles)
                self.npcf_multipoles_norm = np.zeros_like(pcorr.npcf_multipoles_norm)
                _footnorm = np.zeros_like(pcorr.bin_centers)
                if keep_patchres:
                    centers_patches = np.zeros((cat.npatches, *pcorr.bin_centers.shape), dtype=pcorr.bin_centers.dtype)
                    npcf_multipoles_patches = np.zeros((cat.npatches, *pcorr.npcf_multipoles.shape), dtype=pcorr.npcf_multipoles.dtype)
                    npcf_multipoles_norm_patches = np.zeros((cat.npatches, *pcorr.npcf_multipoles_norm.shape), dtype=pcorr.npcf_multipoles_norm.dtype)
            _shelltriplets = np.array([[pcorr.npcf_multipoles_norm[0,z*self.nbinsz*self.nbinsz+z*self.nbinsz+z,i,i].real 
                                        for i in range(pcorr.nbinsr)] for z in range(self.nbinsz)]) # Rough estimate of scaling of pair counts based on zeroth multipole of triplets
            # Rough estimate of scaling of pair counts based on zeroth multipole of triplets. Note that we might get nans here due to numerical
            # inaccuracies in the multiple counting corrections for bins with zero triplets, so we force those values to be zero.
            _patchnorm = np.nan_to_num(np.sqrt(_shelltriplets)) 
            self.bin_centers += _patchnorm*pcorr.bin_centers
            _footnorm += _patchnorm
            self.npcf_multipoles += pcorr.npcf_multipoles
            self.npcf_multipoles_norm += pcorr.npcf_multipoles_norm
            if keep_patchres:
                centers_patches[elp] += pcorr.bin_centers
                npcf_multipoles_patches[elp] += pcorr.npcf_multipoles
                npcf_multipoles_norm_patches[elp] += pcorr.npcf_multipoles_norm
            if save_patchres:
                pcorr.saveinst(save_patchres, save_filebase+'_patch%i'%elp)

        # Finalize the measurement on the full footprint
        self.bin_centers = np.divide(self.bin_centers,_footnorm, out=np.zeros_like(self.bin_centers), where=_footnorm>0)
        self.bin_centers_mean = np.mean(self.bin_centers,axis=0)
        self.projection = "X"

        if keep_patchres:
            return centers_patches, npcf_multipoles_patches, npcf_multipoles_norm_patches
        
        
    def process(self, cat, cat_random=None, Pi=None, dpix=None, dpix_z=None,
                dotomo=True, rotsignflip=False, apply_edge_correction=False, adjust_tree=False,
                save_patchres=False, save_filebase="", keep_patchres=False):
        r"""
        Compute a shear 3PCF provided a shape catalog

        Parameters
        ----------
        cat: orpheus.SpinTracerCatalog
            The shape catalog which is processed
        cat_random: orpheus.ScalarTracerCatalog, optional
            Galaxy random catalog for the RRR normalization. Required for the
            '3dbox' projected III estimator (Vedder Eq. 17); ignored otherwise.
        Pi: float, optional
            Line-of-sight projection length ('3dbox' only; required there).
        dpix, dpix_z: float, optional
            Transverse hash cell size and line-of-sight slab width ('3dbox' only).
        dotomo: bool
            Flag that decides whether the tomographic information in the shape catalog should be used. Defaults to `True`.
        rotsignflip: bool
            If the shape catalog has been decomposed in patches, choose whether the rotation angle should be flipped.
            For simulated data this was always ok to set to ``False``. Defaults to ``False``.
        apply_edge_correction: bool
            Flag that decides how the NPCF in the real space basis is computed.
            * If set to ``True`` the computation is done via edge-correcting the GGG-multipoles
            * If set to ``False`` both GGG and NNN are transformed separately and the ratio is done in the real-space basis
            Defaults to ``False``.
        adjust_tree: bool
            Overrides the original setup of the tree-approximations in the instance based on the nbar of the shape catalog.
            Not implemented yet; has no effect. Defaults to ``False``.
        save_patchres: bool or str
            If the shape catalog has been decomposed in patches, flag whether to save the GGG measurements on the individual patches.
            Note that the path needs to exist, otherwise a ``ValueError`` is raised. For a flat-sky catalog this parameter
            has no effect. Defaults to ``False``.
        save_filebase: str
            Base of the filenames in which the patches are saved. The full filename will be `<save_patchres>/<save_filebase>_patchxx.npz`.
            Only has an effect if the shape catalog consists of multiple patches and `save_patchres` is not `False`.
        keep_patchres: bool
            If the catalog consists of multiple patches, returns all measurements on the patches. Defaults to `False`.
        """

        # '3dbox' geometry: the projected III correlator (Vedder Eq. 17, S.S.S / RRR)
        # via the discrete slab-hashed estimator. The shape catalog supplies all three
        # polar vertices; a single galaxy random normalizes via RRR. Dispatches to
        # alloc_Gammans_slab_GGG and returns the usual Gamma_n / Norm multipole pair.
        if cat.geometry == '3dbox':
            assert cat_random is not None, "'3dbox' III requires a random catalog (cat_random)."
            assert Pi is not None, "'3dbox' III requires a projection length Pi."
            assert cat_random.geometry == '3dbox', "'3dbox' III requires all catalogs in '3dbox'."
            return self.__process_3dbox(cat, cat_random, float(Pi), dpix=dpix, dpix_z=dpix_z,
                                        dotomo=dotomo)

        # Native curved-sky GGG (geodesic distance + nested-HEALPix query_disc +
        # spin-2 geodesic projection) requires process_spherical and method="DoubleTree";
        # otherwise a spherical catalog must first be decomposed into patches. The
        # struct-based alloc_ggg_doubletree dispatches on cat->metric, so both
        # geometries share the DoubleTree call block below.
        native_spherical = self.process_spherical and cat.geometry == 'spherical'
        if cat.geometry == 'spherical' and not native_spherical and cat.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches '
                             'using the Catalog._topatches method, or process_spherical=True must be set.')
        if native_spherical and self.method != "DoubleTree":
            raise ValueError("Native curved-sky GGG (process_spherical=True) only supports method='DoubleTree'.")

        # Catalog consist of multiple patches
        if cat.patchinds is not None and not native_spherical:
            return self.__process_patches(cat, dotomo=dotomo, rotsignflip=rotsignflip,
                                          apply_edge_correction=apply_edge_correction, adjust_tree=adjust_tree,
                                          save_patchres=save_patchres, save_filebase=save_filebase, keep_patchres=keep_patchres)

        # Catalog does not consist of patches
        else:
            self._checkcats(cat, self.spins)
            if not dotomo:
                self.nbinsz = 1
                old_zbins = cat.zbins[:]
                cat.zbins = np.zeros(cat.ngal, dtype=np.int32)
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

            if self.method == "DoubleTree":
                # Struct-based DoubleTree; alloc_ggg_doubletree dispatches on
                # cat->metric to the flat / curved-sky kernel (cf. GGCorrelation).
                nbinsz = self.nbinsz
                if native_spherical:
                    from healpy import nside2resol
                    sep2rad = convertunits(self.sep_units, 'rad')
                    sep2deg = convertunits(self.sep_units, 'deg')
                    def _nside_for(target_rad):
                        ns = 1
                        while nside2resol(ns) > target_rad and ns < 2**29:
                            ns *= 2
                        return ns
                    nsides = [0 if self.tree_resos[r]==0. else _nside_for(self.tree_resos[r]*sep2rad)
                              for r in range(self.tree_nresos)]
                    nside_hash = _nside_for(max(self.min_sep, 0.5*self.tree_redges[1])*sep2rad)
                    mh = cat.multihash_bundle(reso_redges=self.tree_redges*sep2deg, nsides=nsides,
                                              nside_hash=nside_hash, shuffle=self.shuffle_pix,
                                              fields=(cat.tracer_1, cat.tracer_2), w2field=True,
                                              verbose=self._verbose_python)
                    extra = {'e1_resos': mh['red_e1'], 'e2_resos': mh['red_e2'],
                             'weightsq_resos': mh['red_weightsq']}
                else:
                    cutfirst = np.int32(self.tree_resos[0]==0.)
                    mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                              shuffle=self.shuffle_pix, w2field=True, normed=True, nthreads=self.nthreads)
                    allfields = mh['allfields']
                    weight_resos = mh['weight_resos']
                    e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
                    e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
                    _weightsq_resos = np.concatenate([allfields[i][2] for i in range(len(allfields))]).astype(np.float64)
                    weightsq_resos = _weightsq_resos*weight_resos # reduce renorms all fields --> `unrenorm'
                    extra = {'e1_resos': e1_resos, 'e2_resos': e2_resos, 'weightsq_resos': weightsq_resos}

                cat_s, keep_cat = build_catalog_struct(mh, nbinsz, extra=extra)
                cat_s.nresos = int(self.tree_nresos)
                nav_s, keep_nav = build_navhash_struct(mh, cat_obj=cat)
                tree_s, keep_tree = build_tree_params_struct(self, mh)
                cutfirst = np.int32(self.tree_resos[0]==0.)
                tree_s.nresos_grid = int(self.tree_nresos - cutfirst)
                maxleaf = max(0, self.tree_nresos-1)
                tree_s.minresoind_leaf = min(int(self.minresoind_leaf), maxleaf)
                tree_s.maxresoind_leaf = min(int(self.maxresoind_leaf), maxleaf)
                scale = convertunits(self.sep_units, 'rad') if native_spherical else None
                bin_s = build_binning_struct(self, scale=scale, nmax=int(self.nmax),
                                             dccorr=int(self.multicountcorr))
                out_s, bin_centers, threepcfs_n, _, threepcfsnorm_n, _, _ = build_npcf_output(
                    'ggg', self.nbinsr, nmax=self.nmax, nbinsz=nbinsz)

                # Keep numpy arrays referenced only through struct fields alive.
                _alive = keep_cat + keep_nav + keep_tree   # noqa: F841

                self.clib.alloc_ggg_doubletree(
                    ct.byref(cat_s), ct.byref(nav_s), ct.byref(tree_s), ct.byref(bin_s),
                    int(self.nthreads), int(self._verbose_c)+int(self._verbose_debug),
                    ct.byref(out_s))

                # bin_centers carries a length unit; multipoles are dimensionless.
                if native_spherical:
                    bin_centers = bin_centers / convertunits(self.sep_units, 'rad')

            else:
                # Flat-sky Discrete / Tree / BaseTree via the shared struct interface
                # (hoist-to-locals shim in corrfunc_third.c). rbins=[-1.] is the
                # log-spacing sentinel; nmin=0 (only the diagonal Gamma_0..nmax used).
                out_s, bin_centers, threepcfs_n, _, threepcfsnorm_n, _, _ = build_npcf_output(
                    'ggg', self.nbinsr, nmax=self.nmax, nbinsz=self.nbinsz)
                bin_s = build_binning_struct(self, nmax=int(self.nmax), nmin=0,
                                             dccorr=int(self.multicountcorr),
                                             rbins=np.array([-1.]))
                if self.method=="Discrete":
                    if not cat.hasspatialhash:
                        cat.build_spatialhash(dpix=max(1.,self.max_sep//10.))
                    cat_s, keep_cat = build_flat_catalog_struct(
                        cat.pos1, cat.pos2, cat.weight, cat.zbins, self.nbinsz,
                        cat.isinner, e1=cat.tracer_1, e2=cat.tracer_2)
                    nav_s, keep_nav = build_flat_navhash_struct(cat)
                    _alive = keep_cat + keep_nav   # noqa: F841
                    self.clib.alloc_Gammans_discrete_ggg(
                        ct.byref(cat_s), ct.byref(nav_s), ct.byref(bin_s),
                        int(self.nthreads), int(self._verbose_c), ct.byref(out_s))
                elif self.method in ["Tree", "BaseTree"]:
                    cutfirst = np.int32(self.tree_resos[0]==0.)
                    mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                              shuffle=self.shuffle_pix, w2field=True, normed=True, nthreads=self.nthreads)
                    weight_resos = mh['weight_resos']
                    allfields = mh['allfields']
                    e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
                    e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
                    _weightsq_resos = np.concatenate([allfields[i][2] for i in range(len(allfields))]).astype(np.float64)
                    weightsq_resos = _weightsq_resos*weight_resos # reduce renorms all fields --> `unrenorm'
                    extra = {'e1_resos': e1_resos, 'e2_resos': e2_resos, 'weightsq_resos': weightsq_resos}
                    catf_s, keep_catf = build_catalog_struct(mh, self.nbinsz, extra=extra)
                    catf_s.nresos = int(self.tree_nresos)
                    nav_s, keep_nav = build_navhash_struct(mh, cat_obj=cat)
                    tree_s, keep_tree = build_tree_params_struct(self, mh)
                    tree_s.nresos_grid = int(self.tree_nresos - cutfirst)
                    if self.method=="Tree":
                        cat_s, keep_cat = build_flat_catalog_struct(
                            cat.pos1, cat.pos2, cat.weight, cat.zbins, self.nbinsz,
                            cat.isinner, e1=cat.tracer_1, e2=cat.tracer_2)
                        _alive = keep_cat + keep_catf + keep_nav + keep_tree   # noqa: F841
                        self.clib.alloc_Gammans_tree_ggg(
                            ct.byref(cat_s), ct.byref(catf_s), ct.byref(nav_s), ct.byref(tree_s),
                            ct.byref(bin_s), int(self.nthreads), int(self._verbose_c), ct.byref(out_s))
                    else:   # BaseTree: single multireso catalog (base = reso 0)
                        _alive = keep_catf + keep_nav + keep_tree   # noqa: F841
                        self.clib.alloc_Gammans_basetree_ggg(
                            ct.byref(catf_s), ct.byref(nav_s), ct.byref(tree_s),
                            ct.byref(bin_s), int(self.nthreads), int(self._verbose_c), ct.byref(out_s))

            self.bin_centers = bin_centers.reshape(szr)
            self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
            self.npcf_multipoles = threepcfs_n.reshape(sc)
            self.npcf_multipoles_norm = threepcfsnorm_n.reshape(sn)
            self.projection = "X"

            if apply_edge_correction:
                self.edge_correction()

            if not dotomo:
                cat.zbins = old_zbins

    def __process_3dbox(self, cat_source, cat_random, Pi, dpix=None, dpix_z=None, dotomo=True):
        r"""Projected III (Vedder et al. 2026 Eq. 17) via the discrete slab-hashed GGG estimator.

        All three vertices are polar (shape) galaxies (``cat_source``, spin-2): the
        shape catalog is looped (numerator central) and slab-hashed (the two G-legs).
        The kernel emits the four raw natural :math:`SSS` multipole components and the
        shared random :math:`RRR` count (a single galaxy random ``cat_random`` at all
        three vertices); this method applies :math:`f = W_S/W_R` per shape tomo-bin and
        forms the estimator :math:`SSS / f^3 RRR`, keeping vertices within
        :math:`|\Delta z| < \Pi` of the central. Output is the usual Gamma_n / Norm
        multipole pair, so ``multipoles2npcf`` follows unchanged.
        """
        self._Pi = float(Pi)
        if dpix is None: dpix = self.max_sep
        if dpix_z is None: dpix_z = Pi

        # Tomography: collapse zbins to a single bin if requested.
        old_zbins = None
        if not dotomo:
            self.nbinsz = 1
            old_zbins = (cat_source.zbins.copy(), cat_random.zbins.copy())
            cat_source.zbins = np.zeros(cat_source.ngal, dtype=np.int32)
            cat_random.zbins = np.zeros(cat_random.ngal, dtype=np.int32)
        else:
            self.nbinsz = max(cat_source.nbinsz, cat_random.nbinsz)
        nz = self.nbinsz
        self.nzcombis = nz*nz*nz

        # Shared transverse + line-of-sight extent so every central lies inside the
        # polar-leg / random slab hash grids.
        cats = [cat_source, cat_random]
        ext = [min(c.min1 for c in cats), max(c.max1 for c in cats),
               min(c.min2 for c in cats), max(c.max2 for c in cats)]
        ext_z = [min(c.min3 for c in cats), max(c.max3 for c in cats)]

        Sd = cat_source.multihash_slabs(dpix, dpix_z, fields=(cat_source.tracer_1, cat_source.tracer_2),
                                        extent=ext, extent_z=ext_z)
        Rr = cat_random.multihash_slabs(dpix, dpix_z, extent=ext, extent_z=ext_z)

        # Shape-weight rescaling f = W_S / W_R per shape tomo-bin.
        WS = np.array([cat_source.weight[cat_source.zbins == z].sum() for z in range(nz)])
        WR = np.array([cat_random.weight[cat_random.zbins == z].sum() for z in range(nz)])
        f = np.divide(WS, WR, out=np.ones_like(WS), where=WR > 0).astype(np.float64)

        # Output: the four raw natural SSS multipole components (npcf) + the shared
        # random RRR count (norm_mp). The polar central is looped and slab-hashed on
        # one shared grid (nav_polar); the random is looped as the RRR central and
        # slab-hashed (nav_R) for the count legs. f = W_S/W_R is applied in Python.
        scomp = (4, self.nmax+1, self.nzcombis, self.nbinsr, self.nbinsr)
        sn = (self.nmax+1, self.nzcombis, self.nbinsr, self.nbinsr)
        szr = (nz, nz, self.nbinsr)
        _mplen = (self.nmax+1)*self.nzcombis*self.nbinsr*self.nbinsr
        out_s, bin_centers, Comp_n, _, RRR_n, _, _ = build_npcf_output(
            'gnn', self.nbinsr, nmax=self.nmax, bc_len=nz*nz*self.nbinsr,
            npcf_len=4*_mplen, norm_mp_len=_mplen, ncomp=self.n_cfs)
        bin_s = build_binning_struct(self, nmax=self.nmax, dccorr=self.multicountcorr, Pi=self._Pi)

        cat_c, kc1 = build_slab_catalog_struct(Sd, nz, e1e2=Sd['fields'])
        nav_c, kn1 = build_slab_navhash_struct(Sd)
        cat_R, kc2 = build_slab_catalog_struct(Rr, nz)
        nav_R, kn2 = build_slab_navhash_struct(Rr)
        _alive = kc1 + kn1 + kc2 + kn2

        self.clib.alloc_Gammans_slab_GGG(
            ct.byref(cat_c), ct.byref(nav_c), ct.byref(cat_R), ct.byref(nav_R),
            ct.byref(bin_s), int(self.nthreads), int(self._verbose_c), ct.byref(out_s))

        # Raw f-free components (private): the four natural SSS multipoles + the shared
        # random RRR count.
        self._SSS = np.nan_to_num(Comp_n.reshape(scomp))
        self._RRR = np.nan_to_num(RRR_n.reshape(sn))

        # The III numerator is the raw SSS (no numerator f); the RRR denominator is
        # rescaled by f = W_S/W_R at each of the three vertices (f[zc] f[z2] f[z3]).
        zc_i, z2_i, z3_i = np.unravel_index(np.arange(self.nzcombis), (nz, nz, nz))
        fc = f[zc_i]; f2 = f[z2_i]; f3 = f[z3_i]
        self.npcf_multipoles = self._SSS
        self.npcf_multipoles_norm = (fc*f2*f3).reshape(1, self.nzcombis, 1, 1)*self._RRR

        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=(0, 1))
        self.projection = "X"
        self.is_edge_corrected = False

        if not dotomo:
            cat_source.zbins, cat_random.zbins = old_zbins
        return

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
    
    # Legacy transform in pure python -- now upgraded to .c
    def _multipoles2npcf_py(self):
        
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
                self.npcf[elm,...,elphi] = tmp
        # Number of triangles
        for elphi, phi in enumerate(self.phi):
            tmptotnorm = 1./(2*np.pi) * self.npcf_multipoles_norm[0].astype(complex)
            for n in range(1,self.nmax+1):
                _const = 1./(2*np.pi) * np.exp(1J*n*phi)
                tmptotnorm += _const * self.npcf_multipoles_norm[n].astype(complex)
                tmptotnorm += _const.conj() * self.npcf_multipoles_norm[n][ztiler].astype(complex).transpose(0,2,1)
            self.npcf_norm[...,elphi] = tmptotnorm
          
        if self.is_edge_corrected:
            dphi = self.phi[1] - self.phi[0]
            N0 = dphi/(2*np.pi) * self.npcf_multipoles_norm[self.nmax].astype(complex)
            sel_zero = np.isnan(N0)
            _a = self.npcf
            _b = N0.real[np.newaxis, :, :, :, np.newaxis]
            self.npcf = np.divide(_a, _b, out=np.zeros_like(_a), where=_b>0)
        else:
            _a = self.npcf
            _b = self.npcf_norm
            self.npcf = np.divide(_a, _b, out=np.zeros_like(_a), where=_b>0)
        self.projection = "X"
        
    def multipoles2npcf(self, projection='Centroid'):
        r"""
        Notes
        -----
        The Upsilon and Norms are only computed for the n>0 multipoles. The n<0 multipoles are recovered by symmetry considerations given in Eq A.6 in Porth+23.
        """
        assert(projection in self.projections_avail)
        int_projection = {'X':0,'Centroid':1}
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        thisnpcf = np.zeros(4*self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr*len(self.phi), dtype=np.complex128)
        thisnpcf_norm = np.zeros(self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr*len(self.phi), dtype=np.complex128)
        self.clib.multipoles2npcf_ggg(
            self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(), np.int32(self.nmax), np.int32(self.nbinsz),
            self.bin_centers_mean, np.int32(self.nbinsr), self.phi.astype(np.float64), np.int32(self.nbinsphi[0]), 
            np.int32(int_projection[projection]), np.int32(self.nthreads), thisnpcf, thisnpcf_norm)
        self.npcf = thisnpcf.reshape((4,nzcombis,self.nbinsr,self.nbinsr,len(self.phi)))
        self.npcf_norm = thisnpcf_norm.reshape((nzcombis,self.nbinsr,self.nbinsr,len(self.phi)))
        self.projection = projection
            
    ## PROJECTIONS (Preferably use direct in c-level) ##
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
        Compute third-order aperture statistics using the polynomial filter.
        """
        
        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf(projection='Centroid')
            
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
        
        # To avoid zero divisions we set some default bin centers for the evaluation of the filter
        # As for those positions the 3pcf is zero those will not contribute to the map3 integral
        if (np.min(normys_centers)==0):
            _sel = normys_centers!=0
            _avratios = np.mean(normys_centers[_sel]/normys_edges[_sel])
            normys_centers[~_sel] = _avratios*normys_edges[~_sel]
        
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
    
    def _map3_filtergrid_multiR(self, R1, R2, R3):
        return self.__map3_filtergrid_multiR(R1, R2, R3, self.bin_edges, self.bin_centers_mean, self.phi, include_measure=True)
    
    @staticmethod
    @jit(nopython=True)
    def __map3_filtergrid_multiR(R1, R2, R3, normys_edges, normys_centers, phis, include_measure=True):
        
        # To avoid zero divisions we set some default bin centers for the evaluation of the filter
        # As for those positions the 3pcf is zero those will not contribute to the map3 integral
        if (np.min(normys_centers)==0):
            _sel = normys_centers!=0
            _avratios = np.mean(normys_centers[_sel]/normys_edges[_sel])
            normys_centers[~_sel] = _avratios*normys_edges[~_sel]
        
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
    r""" Class containing methods to measure and obtain statistics that are built
    from third-order source-lens-lens (G3L) correlation functions.

    Attributes
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.
    zweighting: bool
        Has no effect at the moment.
    zweighting_sigma: float or None
        Has no effect at the moment.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    Additional child-specific parameters can be passed via ``kwargs``.
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.
    """

    def __init__(self, min_sep, max_sep, zweighting=False, zweighting_sigma=None, **kwargs):
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


    def __process_patches(self, cat_source, cat_lens, dotomo_source=True, dotomo_lens=True, rotsignflip=False, 
                          apply_edge_correction=False, save_patchres=False, save_filebase="", keep_patchres=False):
        if save_patchres:
            if not Path(save_patchres).is_dir():
                raise ValueError('Path to directory does not exist.')

        for elp in range(cat_source.npatches):
            if self._verbose_python:
                print('Doing patch %i/%i'%(elp+1,cat_source.npatches))
            # Compute statistics on patch
            pscat = cat_source.frompatchind(elp,rotsignflip=rotsignflip)
            plcat = cat_lens.frompatchind(elp)
            pcorr = GNNCorrelation(
                min_sep=self.min_sep,
                max_sep=self.max_sep,
                nbinsr=self.nbinsr,
                nbinsphi=self.nbinsphi,
                nmaxs=self.nmaxs,
                method=self.method,
                multicountcorr=self.multicountcorr,
                shuffle_pix=self.shuffle_pix,
                tree_resos=self.tree_resos,
                rmin_pixsize=self.rmin_pixsize,
                resoshift_leafs=self.resoshift_leafs,
                minresoind_leaf=self.minresoind_leaf,
                maxresoind_leaf=self.maxresoind_leaf,
                nthreads=self.nthreads,
                verbosity=self.verbosity)
            pcorr.process(pscat, plcat, dotomo_source=dotomo_source, dotomo_lens=dotomo_lens)
            
            # Update the total measurement
            if elp == 0:
                self.nbinsz_source = pcorr.nbinsz_source
                self.nbinsz_lens = pcorr.nbinsz_lens
                self.bin_centers = np.zeros_like(pcorr.bin_centers)
                self.npcf_multipoles = np.zeros_like(pcorr.npcf_multipoles)
                self.npcf_multipoles_norm = np.zeros_like(pcorr.npcf_multipoles_norm)
                _footnorm = np.zeros_like(pcorr.bin_centers)
                if keep_patchres:
                    centers_patches = np.zeros((cat_source.npatches, *pcorr.bin_centers.shape), dtype=pcorr.bin_centers.dtype)
                    npcf_multipoles_patches = np.zeros((cat_source.npatches, *pcorr.npcf_multipoles.shape), dtype=pcorr.npcf_multipoles.dtype)
                    npcf_multipoles_norm_patches = np.zeros((cat_source.npatches, *pcorr.npcf_multipoles_norm.shape), dtype=pcorr.npcf_multipoles_norm.dtype)
            _shelltriplets = np.array([[[pcorr.npcf_multipoles_norm[0,zs*self.nbinsz_lens*self.nbinsz_lens+zl*self.nbinsz_lens+zl,i,i].real 
                                         for i in range(pcorr.nbinsr)] for zl in range(self.nbinsz_lens)] for zs in range(self.nbinsz_source)])
            # Rough estimate of scaling of pair counts based on zeroth multipole of triplets. Note that we might get nans here due to numerical
            # inaccuracies in the multiple counting corrections for bins with zero triplets, so we force those values to be zero.
            _patchnorm = np.nan_to_num(np.sqrt(_shelltriplets)) 
            self.bin_centers += _patchnorm*pcorr.bin_centers
            _footnorm += _patchnorm
            self.npcf_multipoles += pcorr.npcf_multipoles
            self.npcf_multipoles_norm += pcorr.npcf_multipoles_norm
            if keep_patchres:
                centers_patches[elp] += pcorr.bin_centers
                npcf_multipoles_patches[elp] += pcorr.npcf_multipoles
                npcf_multipoles_norm_patches[elp] += pcorr.npcf_multipoles_norm
            if save_patchres:
                pcorr.saveinst(save_patchres, save_filebase+'_patch%i'%elp)

        # Finalize the measurement on the full footprint
        self.bin_centers = np.divide(self.bin_centers,_footnorm, out=np.zeros_like(self.bin_centers), where=_footnorm>0)
        self.bin_centers_mean =np.mean(self.bin_centers, axis=(0,1))
        self.projection = "X"

        if keep_patchres:
            return centers_patches, npcf_multipoles_patches, npcf_multipoles_norm_patches
        
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
    def process(self, cat_source, cat_lens=None, cat_random=None, Pi=None, dpix=None, dpix_z=None,
                dotomo_source=True, dotomo_lens=True, rotsignflip=False, apply_edge_correction=False,
                save_patchres=False, save_filebase="", keep_patchres=False):
        r"""
        Compute a shear-lens-lens correlation provided a source and a lens catalog.

        Parameters
        ----------
        cat_source: orpheus.SpinTracerCatalog
            The source catalog which is processed
        cat_lens: orpheus.ScalarTracerCatalog
            The lens catalog which is processed
        cat_random: orpheus.ScalarTracerCatalog, optional
            Random catalog for the lens/position tracer. Required for the '3dbox'
            projected ggI estimator (Vedder Eq. 17); ignored otherwise.
        Pi: float, optional
            Line-of-sight projection length ('3dbox' only; required there).
        dpix, dpix_z: float, optional
            Transverse hash cell size and line-of-sight slab width ('3dbox' only).
        dotomo_source: bool
            Flag that decides whether the tomographic information in the source catalog should be used. Defaults to `True`.
        dotomo_lens: bool
            Flag that decides whether the tomographic information in the lens catalog should be used. Defaults to `True`.
        rotsignflip: bool
            If the shape catalog has been decomposed in patches, choose whether the rotation angle should be flipped.
            For simulated data this was always ok to set to ``False``. Defaults to ``False``.
        apply_edge_correction: bool
            Flag that decides how the NPCF in the real space basis is computed.
            * If set to ``True`` the computation is done via edge-correcting the GNN-multipoles
            * If set to ``False`` both GNN and NNN are transformed separately and the ratio is done in the real-space basis
            Defaults to ``False``.
        save_patchres: bool or str
            If the shape catalog has been decomposed in patches, flag whether to save the GNN measurements on the individual patches.
            Note that the path needs to exist, otherwise a ``ValueError`` is raised. For a flat-sky catalog this parameter
            has no effect. Defaults to ``False``.
        save_filebase: str
            Base of the filenames in which the patches are saved. The full filename will be ``<save_patchres>/<save_filebase>_patchxx.npz``.
            Only has an effect if the shape catalog consists of multiple patches and ``save_patchres`` is not ``False``.
        keep_patchres: bool
            If the catalog consists of multiple patches, returns all measurements on the patches. Defaults to ``False``.
        """
        # '3dbox' geometry: the projected ggI correlator (Vedder Eq. 17,
        # S.D~.D~ / RRR) via the discrete slab-hashed estimator. When no separate
        # position catalog is passed the positions are taken from the shape catalog
        # (the auto-correlation case). Dispatches to alloc_Gammans_slab_GNN.
        if cat_source.geometry == '3dbox':
            assert cat_random is not None, "'3dbox' ggI requires a random catalog (cat_random)."
            assert Pi is not None, "'3dbox' ggI requires a projection length Pi."
            if cat_lens is None:
                cat_lens = ScalarTracerCatalog(
                    cat_source.pos1, cat_source.pos2, np.ones(cat_source.ngal),
                    pos3=cat_source.pos3, weight=cat_source.weight,
                    zbins=cat_source.zbins.copy(), geometry='3dbox')
            for c in (cat_lens, cat_random):
                assert c.geometry == '3dbox', "'3dbox' ggI requires all catalogs in '3dbox'."
            return self.__process_3dbox(cat_source, cat_lens, cat_random, float(Pi),
                                        dpix=dpix, dpix_z=dpix_z,
                                        dotomo_source=dotomo_source, dotomo_lens=dotomo_lens)

        self._checkcats([cat_source, cat_lens, cat_lens], [2, 0, 0])

         # Catch typical errors, i.e. incompatible catalogs or missin patch decompositions
        if cat_source.geometry=='spherical' and cat_source.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches using the Catalog._topatches method.')
        if cat_lens.geometry=='spherical' and cat_lens.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches using the Catalog._topatches method.')
        if cat_source.geometry != cat_lens.geometry:
            raise ValueError('Incompatible geometries of source catalog (%s) and lens catalog (%s).'%(
                cat_source.geometry,cat_lens.geometry))

        # Catalog consist of multiple patches
        if (cat_source.patchinds is not None) and (cat_lens.patchinds is not None):
            return self.__process_patches(cat_source, cat_lens, dotomo_source=dotomo_source, dotomo_lens=dotomo_lens, 
                                          rotsignflip=rotsignflip, apply_edge_correction=apply_edge_correction, 
                                          save_patchres=save_patchres, save_filebase=save_filebase, keep_patchres=keep_patchres)

        # Catalog does not consist of patches
        else:
        
            if not dotomo_lens and self.zweighting:
                print("Redshift-weighting requires tomographic computation for the lenses.")
                dotomo_lens = True
                
            if not dotomo_source:
                self.nbinsz_source = 1
                old_zbins_source = cat_source.zbins[:]
                cat_source.zbins = np.zeros(cat_source.ngal, dtype=np.int32)
            else:
                self.nbinsz_source = cat_source.nbinsz
            if not dotomo_lens:
                self.nbinsz_lens = 1
                old_zbins_lens = cat_lens.zbins[:]
                cat_lens.zbins = np.zeros(cat_lens.ngal, dtype=np.int32)
            else:
                self.nbinsz_lens = cat_lens.nbinsz
                
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
            # Struct interface (hoist-to-locals shim in corrfunc_third.c). Shape
            # (source) central + two scalar lens legs; source nav carries nregions.
            out_s, bin_centers, Upsilon_n, _, Norm_n, _, _ = build_npcf_output(
                'gnn', self.nbinsr, bc_len=reduce(operator.mul, szr),
                npcf_len=reduce(operator.mul, sc), norm_mp_len=reduce(operator.mul, sn),
                ncomp=self.n_cfs)
            bin_s = build_binning_struct(self, nmax=int(self.nmax),
                                         dccorr=int(self.multicountcorr))
            jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
            if self.method=="Discrete":
                hash_dpix = max(1.,self.max_sep//10.)
                cat_source.build_spatialhash(dpix=hash_dpix, extent=jointextent)
                cat_lens.build_spatialhash(dpix=hash_dpix, extent=jointextent)
                cats_s, keep_cs = build_flat_catalog_struct(
                    cat_source.pos1, cat_source.pos2, cat_source.weight, cat_source.zbins,
                    self.nbinsz_source, cat_source.isinner,
                    e1=cat_source.tracer_1, e2=cat_source.tracer_2)
                navs_s, keep_ns = build_flat_navhash_struct(cat_source)
                catl_s, keep_cl = build_flat_catalog_struct(
                    cat_lens.pos1, cat_lens.pos2, cat_lens.weight, cat_lens.zbins,
                    self.nbinsz_lens, cat_lens.isinner)
                navl_s, keep_nl = build_flat_navhash_struct(cat_lens)
                _alive = keep_cs + keep_ns + keep_cl + keep_nl   # noqa: F841
                self.clib.alloc_Gammans_discrete_GNN(
                    ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                    ct.byref(bin_s), int(self.nthreads), int(self._verbose_c), ct.byref(out_s))
            if self.method == "DoubleTree":
                cutfirst = np.int32(self.tree_resos[0]==0.)
                mhs = cat_source.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                                  shuffle=self.shuffle_pix, normed=True, extent=jointextent, nthreads=self.nthreads)
                sallfields = mhs['allfields']
                e1_resos_source = np.concatenate([sallfields[i][0] for i in range(len(sallfields))]).astype(np.float64)
                e2_resos_source = np.concatenate([sallfields[i][1] for i in range(len(sallfields))]).astype(np.float64)
                cats_s, keep_cs = build_catalog_struct(
                    mhs, self.nbinsz_source, extra={'e1_resos': e1_resos_source, 'e2_resos': e2_resos_source})
                cats_s.nresos = int(self.tree_nresos)
                navs_s, keep_ns = build_navhash_struct(mhs, cat_obj=cat_source)
                mhl = cat_lens.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                                shuffle=self.shuffle_pix, normed=True, extent=jointextent, nthreads=self.nthreads)
                catl_s, keep_cl = build_catalog_struct(mhl, self.nbinsz_lens)
                catl_s.nresos = int(self.tree_nresos)
                navl_s, keep_nl = build_navhash_struct(mhl, cat_obj=cat_lens)
                tree_s, keep_tree = build_tree_params_struct(self, mhs)
                tree_s.nresos_grid = int(self.tree_nresos - cutfirst)
                _alive = keep_cs + keep_ns + keep_cl + keep_nl + keep_tree   # noqa: F841
                self.clib.alloc_Gammans_doubletree_GNN(
                    ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                    ct.byref(tree_s), ct.byref(bin_s), int(self.nthreads), int(self._verbose_c),
                    ct.byref(out_s))
            
            self.bin_centers = bin_centers.reshape(szr)
            self.bin_centers_mean = np.mean(self.bin_centers, axis=(0,1))
            self.npcf_multipoles = np.nan_to_num(Upsilon_n.reshape(sc))
            self.npcf_multipoles_norm = np.nan_to_num(Norm_n.reshape(sn))
            self.projection = "X"
            self.is_edge_corrected = False
            
            if apply_edge_correction:
                self.edge_correction()

            if not dotomo_source:
                cat_source.zbins = old_zbins_source
            if not dotomo_lens:
                cat_lens.zbins = old_zbins_lens

    def __process_3dbox(self, cat_source, cat_lens, cat_random, Pi, dpix=None, dpix_z=None,
                        dotomo_source=True, dotomo_lens=True):
        r"""Projected ggI (Vedder et al. 2026 Eq. 17) via the discrete slab-hashed estimator.

        The shape catalog (``cat_source``, spin-2) sits at the central vertex; the
        two legs are galaxy positions with :math:`\tilde D = D - fR` (``cat_lens``
        supplies :math:`D`, ``cat_random`` supplies :math:`R`, :math:`f = W_D/W_R`
        per position tomographic bin), kept within :math:`|\Delta z| < \Pi` of the
        central. The :math:`RRR` normalization uses the random legs. Output is the
        usual Upsilon / Norm multipole pair, so ``multipoles2npcf`` / ``computeNNM``
        follow unchanged.
        """
        self._Pi = float(Pi)
        if dpix is None: dpix = self.max_sep
        if dpix_z is None: dpix_z = Pi

        # Tomography: collapse zbins to a single bin if requested.
        old_zbins_source = old_zbins_lens = None
        if not dotomo_source:
            self.nbinsz_source = 1
            old_zbins_source = cat_source.zbins.copy()
            cat_source.zbins = np.zeros(cat_source.ngal, dtype=np.int32)
        else:
            self.nbinsz_source = cat_source.nbinsz
        if not dotomo_lens:
            self.nbinsz_lens = 1
            old_zbins_lens = (cat_lens.zbins.copy(), cat_random.zbins.copy())
            cat_lens.zbins = np.zeros(cat_lens.ngal, dtype=np.int32)
            cat_random.zbins = np.zeros(cat_random.ngal, dtype=np.int32)
        else:
            self.nbinsz_lens = max(cat_lens.nbinsz, cat_random.nbinsz)
        nzs, nzd = self.nbinsz_source, self.nbinsz_lens

        # Shared transverse + line-of-sight extent so every shape central lies
        # inside the (lens / random) slab hash grids.
        ext = [min(cat_source.min1, cat_lens.min1, cat_random.min1),
               max(cat_source.max1, cat_lens.max1, cat_random.max1),
               min(cat_source.min2, cat_lens.min2, cat_random.min2),
               max(cat_source.max2, cat_lens.max2, cat_random.max2)]
        ext_z = [min(cat_source.min3, cat_lens.min3, cat_random.min3),
                 max(cat_source.max3, cat_lens.max3, cat_random.max3)]

        D = cat_lens.multihash_slabs(dpix, dpix_z, extent=ext, extent_z=ext_z)
        R = cat_random.multihash_slabs(dpix, dpix_z, extent=ext, extent_z=ext_z)
        assert D['npix'] == R['npix'] and D['nslabs'] == R['nslabs'], \
            "D and R slab hashes must share the grid (same dpix/extent)."

        # Density-weight rescaling f = W_D / W_R per position tomo-bin.
        WD = np.array([cat_lens.weight[cat_lens.zbins == z].sum() for z in range(nzd)])
        WR = np.array([cat_random.weight[cat_random.zbins == z].sum() for z in range(nzd)])
        f = np.divide(WD, WR, out=np.ones_like(WD), where=WR > 0).astype(np.float64)

        # The RRR central is a random position tomographically aligned with the
        # shape central, so the source/lens tomographies must match (positions=shapes).
        assert nzs == nzd, "'3dbox' ggI requires matching source/lens tomographic bins."

        # Output: the four raw f-free numerator sub-correlators S.(D/R).(D/R) stacked
        # on a leading axis (npcf) + the shared random RRR count (norm_mp). The polar
        # central is looped directly (built from raw arrays, no nav); the D and R
        # scalar legs are slab-hashed. f = W_D/W_R is applied in Python (below).
        _z3combis = nzs*nzd*nzd
        scomp = (4, self.nmax+1, _z3combis, self.nbinsr, self.nbinsr)
        sn = (self.nmax+1, _z3combis, self.nbinsr, self.nbinsr)
        szr = (nzs, nzd, self.nbinsr)
        _mplen = (self.nmax+1)*_z3combis*self.nbinsr*self.nbinsr
        out_s, bin_centers, Comp_n, _, RRR_n, _, _ = build_npcf_output(
            'gnn', self.nbinsr, nmax=self.nmax, bc_len=nzs*nzd*self.nbinsr,
            npcf_len=4*_mplen, norm_mp_len=_mplen, ncomp=self.n_cfs)

        cat_c, keep_c = build_slab_catalog_struct(
            {'pos1': cat_source.pos1, 'pos2': cat_source.pos2, 'pos3': cat_source.pos3,
             'weight': cat_source.weight, 'zbins': cat_source.zbins}, nzs,
            e1e2=(cat_source.tracer_1, cat_source.tracer_2))
        cat_D, keep_D = build_slab_catalog_struct(D, nzd)
        nav_D, keep_nD = build_slab_navhash_struct(D)
        cat_R, keep_R = build_slab_catalog_struct(R, nzd)
        nav_R, keep_nR = build_slab_navhash_struct(R)
        bin_s = build_binning_struct(self, nmax=self.nmax, dccorr=self.multicountcorr, Pi=self._Pi)
        _alive = keep_c + keep_D + keep_nD + keep_R + keep_nR

        self.clib.alloc_Gammans_slab_GNN(
            ct.byref(cat_c), ct.byref(cat_D), ct.byref(nav_D), ct.byref(cat_R), ct.byref(nav_R),
            ct.byref(bin_s), ct.c_int32(self.nthreads), ct.c_int32(self._verbose_c),
            ct.byref(out_s))

        # Raw f-free sub-correlators (private, for further analysis).
        self._SDD, self._SDR, self._SRD, self._SRR = np.nan_to_num(Comp_n.reshape(scomp))
        self._RRR = np.nan_to_num(RRR_n.reshape(sn))

        # Recombine with f = W_D/W_R (per position tomo-bin): the ggI numerator
        # S.D~.D~ = SDD - f[z3].SDR - f[z2].SRD + f[z2] f[z3].SRR, and the denominator
        # RRR rescaled by f at each of the three vertices (f[zc] f[z2] f[z3]).
        zc_i, z2_i, z3_i = np.unravel_index(np.arange(_z3combis), (nzs, nzd, nzd))
        fc = f[zc_i].reshape(1, _z3combis, 1, 1)
        f2 = f[z2_i].reshape(1, _z3combis, 1, 1)
        f3 = f[z3_i].reshape(1, _z3combis, 1, 1)
        self.npcf_multipoles = (self._SDD - f3*self._SDR - f2*self._SRD + f2*f3*self._SRR)[None]
        self.npcf_multipoles_norm = fc*f2*f3*self._RRR

        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=(0, 1))
        self.projection = "X"
        self.is_edge_corrected = False

        if not dotomo_source:
            cat_source.zbins = old_zbins_source
        if not dotomo_lens:
            cat_lens.zbins, cat_random.zbins = old_zbins_lens
        return

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
    def multipoles2npcf(self, xi=None):
        r"""
        Notes
        -----
        * The Upsilon and Norms are only computed for the n>0 multipoles. The n<0 multipoles are recovered by symmetry considerations, i.e.:

        .. math::

            \Upsilon_{-n}(\theta_1, \theta_2, z_1, z_2, z_3) =
            \Upsilon_{n}(\theta_2, \theta_1, z_1, z_3, z_2)

        As the tomographic bin combinations are interpreted as a flat list, they need to be appropriately shuffled. This is handled by ``ztiler``.

        * When dividing by the (weighted) counts ``N``, all contributions for which ``N <= 0`` are set to zero.

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

        # Optionally correct by clustering correlation function
        # Assume 
        #   xi[0] has shape (nbinsr_xi, )
        #   xi[1] has shape (nbinsz_lens * nbinsz_lens, nbinsr_xi, )
        if xi is not None:
            assert(len(xi)==2)
            assert(xi[1].shape[1]==len(xi[0]))
            assert(xi[1].shape[0]==self.nbinsz_lens*self.nbinsz_lens)
            # Get angular separation at which xi is evaluated
            _rs1 = self.bin_centers_mean[:, None, None]
            _rs2 = self.bin_centers_mean[None, :, None]
            _phis = self.phi[None, None, :]
            d_xi = np.sqrt(_rs1**2 + _rs2**2 - 2*_rs1*_rs2*np.cos(_phis))
            xi_corr = interp1d(xi[0], xi[1], axis=-1, 
                               bounds_error=False, fill_value=0.0, kind="linear")(d_xi)
            # Apply correction to 3pcf (TODO: Looks a bit ugly...)
            _npcf = self.npcf[0].reshape((self.nbinsz_source, self.nbinsz_lens*self.nbinsz_lens, *d_xi.shape))
            _npcf *= (1.0 + xi_corr[None, ...])                     
            self.npcf[0] = _npcf.reshape(self.npcf[0].shape)
            
            
    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        super()._projectnpcf(self, projection)
        
    ## INTEGRATED MEASURES ##        
    def computeNNM(self, radii, do_multiscale=False, xi=None, tofile=False, filtercache=None):
        """
        Compute third-order aperture statistics using the polyonomial filter of Crittenden 2002.
        """
        nb_config.NUMBA_DEFAULT_NUM_THREADS = self.nthreads
        nb_config.NUMBA_NUM_THREADS = self.nthreads
        
        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf(xi=xi)
            
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
    
class NGGCorrelation(BinnedNPCF):
    r""" Class containing methods to measure and obtain statistics that are built
    from third-order lens-shear-shear correlation functions.

    Attributes
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    Additional child-specific parameters can be passed via ``kwargs``.
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.

    Note that the different components of the NGG correlator are ordered as

    .. math::

            \left[ \tilde{G}_-, \tilde{G}_+, \right] \ ,

    which is different to the usual conventions, but matches orpheus' conventions to
    always start with a correlator in which no polar field is complex conjugated.
    """
    def __init__(self, min_sep, max_sep, **kwargs):
        
        super().__init__(3, [0,2,2], n_cfs=2, min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X"]
        self.nbinsz_source = None
        self.nbinsz_lens = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        
    def __process_patches(self, cat_source, cat_lens, dotomo_source=True, dotomo_lens=True, rotsignflip=False, 
                          apply_edge_correction=False, save_patchres=False, save_filebase="", keep_patchres=False):
        if save_patchres:
            if not Path(save_patchres).is_dir():
                raise ValueError('Path to directory does not exist.')

        for elp in range(cat_source.npatches):
            if self._verbose_python:
                print('Doing patch %i/%i'%(elp+1,cat_source.npatches))
            # Compute statistics on patch
            pscat = cat_source.frompatchind(elp,rotsignflip=rotsignflip)
            plcat = cat_lens.frompatchind(elp)
            pcorr = NGGCorrelation(
                min_sep=self.min_sep,
                max_sep=self.max_sep,
                nbinsr=self.nbinsr,
                nbinsphi=self.nbinsphi,
                nmaxs=self.nmaxs,
                method=self.method,
                multicountcorr=self.multicountcorr,
                shuffle_pix=self.shuffle_pix,
                tree_resos=self.tree_resos,
                rmin_pixsize=self.rmin_pixsize,
                resoshift_leafs=self.resoshift_leafs,
                minresoind_leaf=self.minresoind_leaf,
                maxresoind_leaf=self.maxresoind_leaf,
                nthreads=self.nthreads,
                verbosity=self.verbosity)
            pcorr.process(pscat, plcat, dotomo_source=dotomo_source, dotomo_lens=dotomo_lens)
            
            # Update the total measurement
            if elp == 0:
                self.nbinsz_source = pcorr.nbinsz_source
                self.nbinsz_lens = pcorr.nbinsz_lens
                self.bin_centers = np.zeros_like(pcorr.bin_centers)
                self.npcf_multipoles = np.zeros_like(pcorr.npcf_multipoles)
                self.npcf_multipoles_norm = np.zeros_like(pcorr.npcf_multipoles_norm)
                _footnorm = np.zeros_like(pcorr.bin_centers)
                if keep_patchres:
                    centers_patches = np.zeros((cat_source.npatches, *pcorr.bin_centers.shape), dtype=pcorr.bin_centers.dtype)
                    npcf_multipoles_patches = np.zeros((cat_source.npatches, *pcorr.npcf_multipoles.shape), dtype=pcorr.npcf_multipoles.dtype)
                    npcf_multipoles_norm_patches = np.zeros((cat_source.npatches, *pcorr.npcf_multipoles_norm.shape), dtype=pcorr.npcf_multipoles_norm.dtype)
            _shelltriplets = np.array([[[pcorr.npcf_multipoles_norm[pcorr.nmaxs[0],zl*self.nbinsz_source*self.nbinsz_source+zs*self.nbinsz_source+zs,i,i].real 
                                        for i in range(pcorr.nbinsr)] for zs in range(self.nbinsz_source)] for zl in range(self.nbinsz_lens)])
            # Rough estimate of scaling of pair counts based on zeroth multipole of triplets. Note that we might get nans here due to numerical
            # inaccuracies in the multiple counting corrections for bins with zero triplets, so we force those values to be zero.
            _patchnorm = np.nan_to_num(np.sqrt(_shelltriplets)) 
            self.bin_centers += _patchnorm*pcorr.bin_centers
            _footnorm += _patchnorm
            self.npcf_multipoles += pcorr.npcf_multipoles
            self.npcf_multipoles_norm += pcorr.npcf_multipoles_norm
            if keep_patchres:
                centers_patches[elp] += pcorr.bin_centers
                npcf_multipoles_patches[elp] += pcorr.npcf_multipoles
                npcf_multipoles_norm_patches[elp] += pcorr.npcf_multipoles_norm
            if save_patchres:
                pcorr.saveinst(save_patchres, save_filebase+'_patch%i'%elp)

        # Finalize the measurement on the full footprint
        self.bin_centers = np.divide(self.bin_centers,_footnorm, out=np.zeros_like(self.bin_centers), where=_footnorm>0)
        self.bin_centers_mean =np.mean(self.bin_centers, axis=(0,1))
        self.projection = "X"

        if keep_patchres:
            return centers_patches, npcf_multipoles_patches, npcf_multipoles_norm_patches

    def __process_3dbox(self, cat_source, cat_lens, cat_random, Pi,
                        dpix=None, dpix_z=None, dotomo_source=True, dotomo_lens=True):
        r"""Projected gII (Vedder et al. 2026 Eq. 17) via the discrete slab-hashed NGG estimator.

        The scalar (density) catalog ``cat_lens`` sits at the central vertex; the two
        polar (source) legs use ``cat_source`` for the signal :math:`G`-multipoles.
        Rather than form :math:`\tilde D = D - fR` in C, the kernel emits the two raw
        f-free numerator sub-correlators ``DSS`` (data-lens central) / ``RSS``
        (random-lens central) and the shared random ``RRR`` count (the single random
        ``cat_random`` at all three vertices); this method applies :math:`f = W_D/W_R`
        per lens tomo-bin and combines :math:`\tilde D SS = DSS - f RSS` over
        :math:`f^3 RRR`. All within :math:`|\Delta z| < \Pi` of the central.
        """
        self._Pi = float(Pi)
        if dpix is None: dpix = self.max_sep
        if dpix_z is None: dpix_z = Pi

        # Tomography: collapse zbins to a single bin if requested.
        old_zbins_source = old_zbins_lens = None
        if not dotomo_source:
            self.nbinsz_source = 1
            old_zbins_source = cat_source.zbins.copy()
            cat_source.zbins = np.zeros(cat_source.ngal, dtype=np.int32)
        else:
            self.nbinsz_source = cat_source.nbinsz
        if not dotomo_lens:
            self.nbinsz_lens = 1
            old_zbins_lens = (cat_lens.zbins.copy(), cat_random.zbins.copy())
            cat_lens.zbins = np.zeros(cat_lens.ngal, dtype=np.int32)
            cat_random.zbins = np.zeros(cat_random.ngal, dtype=np.int32)
        else:
            self.nbinsz_lens = max(cat_lens.nbinsz, cat_random.nbinsz)
        nzs, nzl = self.nbinsz_source, self.nbinsz_lens
        # The RRR legs are the lens random, tomographically aligned with the shape
        # legs, so the source/lens tomographies must match (positions=shapes).
        assert nzs == nzl, "'3dbox' gII requires matching source/lens tomographic bins."

        # Shared transverse + line-of-sight extent so every central lies inside the
        # polar-leg / random slab hash grids.
        cats = [cat_source, cat_lens, cat_random]
        ext = [min(c.min1 for c in cats), max(c.max1 for c in cats),
               min(c.min2 for c in cats), max(c.max2 for c in cats)]
        ext_z = [min(c.min3 for c in cats), max(c.max3 for c in cats)]

        Sd = cat_source.multihash_slabs(dpix, dpix_z, fields=(cat_source.tracer_1, cat_source.tracer_2),
                                        extent=ext, extent_z=ext_z)
        Rl = cat_random.multihash_slabs(dpix, dpix_z, extent=ext, extent_z=ext_z)

        # Density-weight rescaling f = W_D / W_R per lens tomo-bin.
        WD = np.array([cat_lens.weight[cat_lens.zbins == z].sum() for z in range(nzl)])
        WR = np.array([cat_random.weight[cat_random.zbins == z].sum() for z in range(nzl)])
        f = np.divide(WD, WR, out=np.ones_like(WD), where=WR > 0).astype(np.float64)

        # Output: the two raw f-free sub-correlators DSS / RSS (each 2 natural comps)
        # stacked on a leading axis (npcf) + the shared random RRR count (norm_mp).
        _z3combis = nzl*nzs*nzs
        nmp = 2*self.nmax+1
        scomp = (2, self.n_cfs, nmp, _z3combis, self.nbinsr, self.nbinsr)
        sn = (nmp, _z3combis, self.nbinsr, self.nbinsr)
        szr = (nzl, nzs, self.nbinsr)
        _mplen = nmp*_z3combis*self.nbinsr*self.nbinsr
        out_s, bin_centers, Comp_n, _, RRR_n, _, _ = build_npcf_output(
            'ngg', self.nbinsr, bc_len=reduce(operator.mul, szr),
            npcf_len=2*self.n_cfs*_mplen, norm_mp_len=_mplen, ncomp=self.n_cfs)
        bin_s = build_binning_struct(self, nmax=int(self.nmax), dccorr=int(self.multicountcorr), Pi=self._Pi)

        # cat_lens (D central) is looped directly (raw arrays, no nav); the shape
        # legs are slab-hashed (e1/e2); the single random is looped as the R central
        # AND hashed (nav_lensR) for the RRR count legs, so its catalog struct is
        # built from the same hash bundle Rl.
        def _rawcat(c):
            return {'pos1': c.pos1, 'pos2': c.pos2, 'pos3': c.pos3, 'weight': c.weight, 'zbins': c.zbins}
        catlD, kc1 = build_slab_catalog_struct(_rawcat(cat_lens), nzl)
        catlR, kc2 = build_slab_catalog_struct(Rl, nzl)
        navlR, kn2 = build_slab_navhash_struct(Rl)
        catsD, kc3 = build_slab_catalog_struct(Sd, nzs, e1e2=Sd['fields'])
        navsD, kn3 = build_slab_navhash_struct(Sd)
        _alive = kc1 + kc2 + kn2 + kc3 + kn3

        self.clib.alloc_Gammans_slab_NGG(
            ct.byref(catlD), ct.byref(catlR), ct.byref(catsD), ct.byref(navsD),
            ct.byref(navlR), ct.byref(bin_s),
            int(self.nthreads), int(self._verbose_c), ct.byref(out_s))

        # Raw f-free sub-correlators (private, for further analysis).
        _DSS, _RSS = np.nan_to_num(Comp_n.reshape(scomp))
        self._DSS, self._RSS = _DSS, _RSS
        self._RRR = np.nan_to_num(RRR_n.reshape(sn))

        # Recombine with f = W_D/W_R: D~.S.S = DSS - f[zc].RSS (f on the density
        # central); RRR rescaled by f at each of the three vertices.
        zc_i, z2_i, z3_i = np.unravel_index(np.arange(_z3combis), (nzl, nzs, nzs))
        fc = f[zc_i]; f2 = f[z2_i]; f3 = f[z3_i]
        self.npcf_multipoles = _DSS - fc.reshape(1, 1, _z3combis, 1, 1)*_RSS
        self.npcf_multipoles_norm = (fc*f2*f3).reshape(1, _z3combis, 1, 1)*self._RRR

        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=(0, 1))
        self.projection = "X"
        self.is_edge_corrected = False

        if not dotomo_source:
            cat_source.zbins = old_zbins_source
        if not dotomo_lens:
            cat_lens.zbins, cat_random.zbins = old_zbins_lens
        return

    def process(self, cat_source, cat_lens=None, cat_random=None,
                Pi=None, dpix=None, dpix_z=None, dotomo_source=True, dotomo_lens=True,
                rotsignflip=False, apply_edge_correction=False,
                save_patchres=False, save_filebase="", keep_patchres=False):
        r"""
        Compute a lens-shear-shear correlation provided a source and a lens catalog.

        Parameters
        ----------
        cat_source: orpheus.SpinTracerCatalog
            The source catalog which is processed
        cat_lens: orpheus.ScalarTracerCatalog
            The lens catalog which is processed
        dotomo_source: bool
            Flag that decides whether the tomographic information in the source catalog should be used. Defaults to `True`.
        dotomo_lens: bool
            Flag that decides whether the tomographic information in the lens catalog should be used. Defaults to `True`.
        rotsignflip: bool
            If the shape catalog has been decomposed in patches, choose whether the rotation angle should be flipped.
            For simulated data this was always ok to set to ``False``. Defaults to ``False``.
        apply_edge_correction: bool
            Flag that decides how the NPCF in the real space basis is computed.
            * If set to ``True`` the computation is done via edge-correcting the NGG-multipoles
            * If set to ``False`` both NGG and NNN are transformed separately and the ratio is done in the real-space basis
            Defaults to ``False``.
        save_patchres: bool or str
            If the shape catalog has been decomposed in patches, flag whether to save the NGG measurements on the individual patches.
            Note that the path needs to exist, otherwise a ``ValueError`` is raised. For a flat-sky catalog this parameter
            has no effect. Defaults to ``False``.
        save_filebase: str
            Base of the filenames in which the patches are saved. The full filename will be ``<save_patchres>/<save_filebase>_patchxx.npz``.
            Only has an effect if the shape catalog consists of multiple patches and ``save_patchres`` is not ``False``.
        keep_patchres: bool
            If the catalog consists of multiple patches, returns all measurements on the patches. Defaults to ``False``.
        """

        # '3dbox' geometry: projected gII (Vedder Eq. 17, D~.S.S / RRR) via the
        # discrete slab-hashed NGG estimator with a single (lens/density) random.
        # When no separate position catalog is passed the positions are taken from
        # the shape catalog (the auto-correlation case). Dispatches to
        # alloc_Gammans_slab_NGG and returns the usual Upsilon / Norm multipole pair.
        if cat_source.geometry == '3dbox':
            assert cat_random is not None, "'3dbox' gII requires a random catalog (cat_random)."
            assert Pi is not None, "'3dbox' gII requires a projection length Pi."
            if cat_lens is None:
                cat_lens = ScalarTracerCatalog(
                    cat_source.pos1, cat_source.pos2, np.ones(cat_source.ngal),
                    pos3=cat_source.pos3, weight=cat_source.weight,
                    zbins=cat_source.zbins.copy(), geometry='3dbox')
            assert cat_lens.geometry == '3dbox' and cat_random.geometry == '3dbox', \
                "'3dbox' gII requires all catalogs in '3dbox'."
            return self.__process_3dbox(cat_source, cat_lens, cat_random, float(Pi),
                                        dpix=dpix, dpix_z=dpix_z,
                                        dotomo_source=dotomo_source, dotomo_lens=dotomo_lens)

        self._checkcats([cat_lens, cat_source, cat_source], [0, 2, 2])

         # Catch typical errors, i.e. incompatible catalogs or missin patch decompositions
        if cat_source.geometry=='spherical' and cat_source.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches using the Catalog._topatches method.')
        if cat_lens.geometry=='spherical' and cat_lens.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches using the Catalog._topatches method.')
        if cat_source.geometry != cat_lens.geometry:
            raise ValueError('Incompatible geometries of source catalog (%s) and lens catalog (%s).'%(
                cat_source.geometry,cat_lens.geometry))

        # Catalog consist of multiple patches
        if (cat_source.patchinds is not None) and (cat_lens.patchinds is not None):
            return self.__process_patches(cat_source, cat_lens, dotomo_source=dotomo_source, dotomo_lens=dotomo_lens,
                                          rotsignflip=rotsignflip, apply_edge_correction=apply_edge_correction,
                                          save_patchres=save_patchres, save_filebase=save_filebase, keep_patchres=keep_patchres)

        # Catalog does not consist of patches
        else:
            if not dotomo_source:
                self.nbinsz_source = 1
                old_zbins_source = cat_source.zbins[:]
                cat_source.zbins = np.zeros(cat_source.ngal, dtype=np.int32)
            else:
                self.nbinsz_source = cat_source.nbinsz
            if not dotomo_lens:
                self.nbinsz_lens = 1
                old_zbins_lens = cat_lens.zbins[:]
                cat_lens.zbins = np.zeros(cat_lens.ngal, dtype=np.int32)
            else:
                self.nbinsz_lens = cat_lens.nbinsz
                    
            _z3combis = self.nbinsz_lens*self.nbinsz_source*self.nbinsz_source
            _r2combis = self.nbinsr*self.nbinsr
            sc = (self.n_cfs, 2*self.nmax+1, _z3combis, self.nbinsr, self.nbinsr)
            sn = (2*self.nmax+1, _z3combis, self.nbinsr,self.nbinsr)
            szr = (self.nbinsz_lens, self.nbinsz_source, self.nbinsr)
            # Struct interface (hoist-to-locals shim in corrfunc_third.c). Scalar-
            # position central lens + two shear (source) legs; the lens (central)
            # nav carries the occupied-region list. NGG has n_cfs=2 natural
            # components and 2*nmax+1 multipoles.
            out_s, bin_centers, Upsilon_n, _, Norm_n, _, _ = build_npcf_output(
                'ngg', self.nbinsr, bc_len=reduce(operator.mul, szr),
                npcf_len=reduce(operator.mul, sc), norm_mp_len=reduce(operator.mul, sn),
                ncomp=self.n_cfs)
            bin_s = build_binning_struct(self, nmax=int(self.nmax),
                                         dccorr=int(self.multicountcorr))
            jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))
            if self.method=="Discrete":
                hash_dpix = max(1.,self.max_sep//10.)
                cat_source.build_spatialhash(dpix=hash_dpix, extent=jointextent)
                cat_lens.build_spatialhash(dpix=hash_dpix, extent=jointextent)
                cats_s, keep_cs = build_flat_catalog_struct(
                    cat_source.pos1, cat_source.pos2, cat_source.weight, cat_source.zbins,
                    self.nbinsz_source, cat_source.isinner,
                    e1=cat_source.tracer_1, e2=cat_source.tracer_2)
                navs_s, keep_ns = build_flat_navhash_struct(cat_source)
                catl_s, keep_cl = build_flat_catalog_struct(
                    cat_lens.pos1, cat_lens.pos2, cat_lens.weight, cat_lens.zbins,
                    self.nbinsz_lens, cat_lens.isinner)
                navl_s, keep_nl = build_flat_navhash_struct(cat_lens)
                _alive = keep_cs + keep_ns + keep_cl + keep_nl   # noqa: F841
                self.clib.alloc_Gammans_discrete_NGG(
                    ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                    ct.byref(bin_s), int(self.nthreads), int(self._verbose_c), ct.byref(out_s))
            if self.method=="Tree" or self.method == "DoubleTree":
                cutfirst = np.int32(self.tree_resos[0]==0.)
                mhs = cat_source.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                                  shuffle=self.shuffle_pix, normed=True, extent=jointextent, nthreads=self.nthreads)
                sallfields = mhs['allfields']
                e1_resos_source = np.concatenate([sallfields[i][0] for i in range(len(sallfields))]).astype(np.float64)
                e2_resos_source = np.concatenate([sallfields[i][1] for i in range(len(sallfields))]).astype(np.float64)
                cats_s, keep_cs = build_catalog_struct(
                    mhs, self.nbinsz_source, extra={'e1_resos': e1_resos_source, 'e2_resos': e2_resos_source})
                cats_s.nresos = int(self.tree_nresos)
                navs_s, keep_ns = build_navhash_struct(mhs, cat_obj=cat_source)
                mhl = cat_lens.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                                shuffle=self.shuffle_pix, normed=True, extent=jointextent, nthreads=self.nthreads)
                navl_s, keep_nl = build_navhash_struct(mhl, cat_obj=cat_lens)
            if self.method=="Tree":
                # Tree: reduced source field + base lens central (its multireso hash
                # in navl_s carries the lens nregions).
                catl_s, keep_cl = build_flat_catalog_struct(
                    cat_lens.pos1, cat_lens.pos2, cat_lens.weight, cat_lens.zbins,
                    self.nbinsz_lens, cat_lens.isinner)
                tree_s, keep_tree = build_tree_params_struct(self, mhs)
                _alive = keep_cs + keep_ns + keep_cl + keep_nl + keep_tree   # noqa: F841
                self.clib.alloc_Gammans_tree_NGG(
                    ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                    ct.byref(tree_s), ct.byref(bin_s), int(self.nthreads), int(self._verbose_c),
                    ct.byref(out_s))
            if self.method == "DoubleTree":
                catl_s, keep_cl = build_catalog_struct(mhl, self.nbinsz_lens)
                catl_s.nresos = int(self.tree_nresos)
                tree_s, keep_tree = build_tree_params_struct(self, mhs)
                tree_s.nresos_grid = int(self.tree_nresos - cutfirst)
                _alive = keep_cs + keep_ns + keep_cl + keep_nl + keep_tree   # noqa: F841
                self.clib.alloc_Gammans_doubletree_NGG(
                    ct.byref(cats_s), ct.byref(navs_s), ct.byref(catl_s), ct.byref(navl_s),
                    ct.byref(tree_s), ct.byref(bin_s), int(self.nthreads), int(self._verbose_c),
                    ct.byref(out_s))
            
            # Components of npcf are ordered as (Ups_-, Ups_+)
            self.bin_centers = bin_centers.reshape(szr)
            self.bin_centers_mean = np.mean(self.bin_centers, axis=(0,1))
            self.npcf_multipoles = Upsilon_n.reshape(sc)
            self.npcf_multipoles_norm = Norm_n.reshape(sn)
            self.projection = "X"
            self.is_edge_corrected = False
            
            if apply_edge_correction:
                self.edge_correction()

            if not dotomo_source:
                cat_source.zbins = old_zbins_source  
            if not dotomo_lens:
                cat_lens.zbins = old_zbins_lens
            
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
        r"""
        Notes
        -----
        * When dividing by the (weighted) counts ``N``, all contributions for which ``N <= 0`` are set to zero.

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