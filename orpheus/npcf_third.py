import numpy as np
import ctypes as ct
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

__all__ = ["GGGCorrelation", "NNNCorrelation", "GNNCorrelation", "NGGCorrelation"]

class NNNCorrelation(BinnedNPCF):
    r""" Class containing methods to measure the scalar (clustering) three-point
    correlation function via a Landy-Szalay-like estimator.

    Without a random catalog :meth:`process` yields the raw (weighted) triplet-count multipoles. 
    With a random catalog the clustering 3PCF is estimated as

    .. math::

        \zeta = \frac{(D-R)^3}{RRR}
                = \frac{DDD - DDR - DRD - RDD + DRR + RDR + RRD - RRR}{RRR},

    the three-point generalization of the Landy-Szalay estimator. Eacth data contribution 
    is rescaled by ``f = W_R/W_D`` per tomo bin so that all contributions share the random 
    normalization.

    Parameters
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.
    process_spherical: bool, optional
        Process spherical catalogs using curved-sky geometry and not via flat-sky
        patches. Defaults to ``False``.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    Additional child-specific parameters can be passed via ``kwargs``. 
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.
    """

    def __init__(self, min_sep, max_sep, process_spherical=False, **kwargs):

        # Only the doubletree kernel is implemented; `process` raised for the other three
        # schemes, which is now refused on construction instead.
        super().__init__(order=3, spins=np.array([0,0,0], dtype=np.int32), n_cfs=1,
                         min_sep=min_sep, max_sep=max_sep,
                         methods_avail=["DoubleTree"], **kwargs)
        self.process_spherical = bool(process_spherical)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        self.nzcombis = None
        self.zeta = None
 
        self._initprojections(self)

    def saveinst(self, path_save, fname, extr_pars=None):
        r"""Serialise the instance to a ``.npz`` archive."""
        extras = dict(nbinsz=self.nbinsz, nzcombis=self.nzcombis, zeta=self.zeta)
        if extr_pars: extras.update(extr_pars)
        super().saveinst(path_save, fname, extr_pars=extras)

    def process(self, cat, cat_random=None, dotomo=True, adjust_tree=False):
        r"""Compute the raw triplet-count multipoles, and optionally the clustering 3PCF ``zeta``.

        Without a random catalog this stores the raw (weighted) triplet-count multipoles, 
        otherwise it computes the full ``zeta``.

        Parameters
        ----------
        cat: orpheus.ScalarTracerCatalog
            The (clustered) catalog.
        cat_random: orpheus.ScalarTracerCatalog, optional
            A random catalog. If set, the clustering correlation function ``zeta`` is computed.
        dotomo: bool
            Whether to use the tomographic information in the catalog. Defaults to ``True``.
        adjust_tree: bool
            Currently unused.
        """
        # If a random catalog is present, estimate zeta via the LS-like estimator.
        if cat_random is not None:
            assert(isinstance(cat_random, ScalarTracerCatalog))
            return self.__compute_zeta(cat, cat_random, dotomo=dotomo, adjust_tree=adjust_tree)

        if self.method != "DoubleTree":
            raise NotImplementedError("NNNCorrelation currently supports only the 'DoubleTree' method.")

        native_spherical = self.process_spherical and cat.geometry == 'spherical'
        if cat.geometry == 'spherical' and not native_spherical:
            raise ValueError('Error: Spherical NNN requires process_spherical=True; '
                             'flat-sky patch decomposition is not supported.')

        ## Tomography setup ##
        self._checkcats(cat, self.spins)
        old_zbins = None
        if not dotomo:
            self.nbinsz = 1
            old_zbins = cat.zbins[:]
            cat.zbins = np.zeros(cat.ngal, dtype=np.int32)
            self.nzcombis = 1
        else:
            self.nbinsz = cat.nbinsz
            self.nzcombis = self.nbinsz*self.nbinsz*self.nbinsz
        nbinsz = self.nbinsz

        ## Build the multihash bundle ##
        if native_spherical:
            sep2deg = convertunits(self.sep_units, 'deg')
            nsides, nside_hash = self.tree_resos_to_nsides()
            mh = cat.multihash_bundle(reso_redges=self.tree_redges*sep2deg, nsides=nsides,
                                      nside_hash=nside_hash, shuffle=self.shuffle_pix,
                                      verbose=self._verbose_python)
            assert not mh['nav_coarsened'], (
                "nav_coarsen is incompatible with the NNN doubletree: it reuses "
                "nside_nav for the cross-reso reduction hierarchy, which requires "
                "nside_nav == the reduction nside.")
        else:
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.dpix_hash,
                                      shuffle=self.shuffle_pix, normed=True, nthreads=self.nthreads)

        ## Build the four input structs + output arrays ##
        cat_s, keep_cat = build_catalog_struct(mh, nbinsz)
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
        out_s, bin_centers, triplets_n, _, triplets_norm_n, _, _ = build_npcf_output(
            'NNN', self.nbinsr, nmax=self.nmax, nbinsz=nbinsz)

        # Keep numpy arrays backing ctypes pointer fields alive during the C call.
        _alive = keep_cat + keep_nav + keep_tree   # noqa: F841

        self.clib.alloc_nnn_doubletree(
            ct.byref(cat_s), ct.byref(nav_s), ct.byref(tree_s), ct.byref(bin_s),
            int(self.nthreads), int(self._verbose_c)+int(self._verbose_debug),
            ct.byref(out_s))

        if native_spherical:
            bin_centers = bin_centers / convertunits(self.sep_units, 'rad')

        sc = (1, self.nmax+1, self.nzcombis, self.nbinsr, self.nbinsr)
        sn = (self.nmax+1, self.nzcombis, self.nbinsr, self.nbinsr)
        szr = (self.nbinsz, self.nbinsr)
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.npcf_multipoles = triplets_n.reshape(sc)
        self.npcf_multipoles_norm = triplets_norm_n.reshape(sn)
        self.projection = "X"

        if not dotomo:
            cat.zbins = old_zbins


    # Just a helper to make the zeta computation less awkward
    def _inverse_transform(self, correlator_n, nbinsz1, nbinsz23):
        nzcombis = nbinsz1*nbinsz23*nbinsz23
        rbins = self.nbinsr
        nbinsphi = len(self.phi)
        correlator_f = np.ascontiguousarray(correlator_n).flatten()
        dummy = np.zeros(self.n_cfs*nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        correlator_real = np.zeros(nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        conjmap = np.array([0], dtype=np.int32)
        modeweight = self.mode_window(self.nmax)/nbinsphi
        self.clib.multipoles2npcf_third_z1z23(
            correlator_f, correlator_f,
            np.int32(self.nmax), np.int32(self.n_cfs), np.int32(nbinsz1), np.int32(nbinsz23),
            np.int32(rbins),
            self.phi.astype(np.float64), np.int32(nbinsphi),
            np.int32(0), conjmap, modeweight,
            np.int32(0), np.int32(0), np.zeros(nzcombis, dtype=np.float64),
            np.int32(self.nthreads),
            dummy, correlator_real)
        return correlator_real.reshape((nzcombis, rbins, rbins, nbinsphi)).real

    def multipoles2npcf(self):
        r"""Transform triplets from multipole-basis to the real-space basis.
        """
        ntriplet = self._inverse_transform(self.npcf_multipoles_norm, self.nbinsz, self.nbinsz)
        self.npcf = ntriplet[None]
        self.npcf_norm = ntriplet
        self.projection = "X"

    def __compute_zeta(self, cat_data, cat_rand, dotomo=True, adjust_tree=False, count_floor_rtol=None):
        r"""Estimate the clustering 3PCF via LS-like estimator: ``zeta = (D-R)^3 / RRR``.
        """
        nz = cat_data.nbinsz if dotomo else 1

        # Add data and random to a 2*nbinsz joint catalog and process this
        zbins = np.zeros(cat_data.ngal + cat_rand.ngal, dtype=int)
        zbins[:cat_data.ngal] += cat_data.zbins
        zbins[cat_data.ngal:] += cat_data.nbinsz + cat_rand.zbins
        if not dotomo:
            zbins[:cat_data.ngal] = 0
            zbins[cat_data.ngal:] = 1

        joint_cat = ScalarTracerCatalog(
            pos1=np.append(cat_data.pos1, cat_rand.pos1),
            pos2=np.append(cat_data.pos2, cat_rand.pos2),
            tracer=np.ones(cat_data.ngal + cat_rand.ngal),
            weight=np.append(cat_data.weight, cat_rand.weight),
            geometry=cat_data.geometry,
            units_pos1=cat_data.units_pos1,
            units_pos2=cat_data.units_pos1,
            zbins=zbins)

        if cat_data.geometry == "spherical" and not self.process_spherical:
            raise ValueError('Error: Spherical NNN requires process_spherical=True.')

        self.process(cat=joint_cat, cat_random=None, dotomo=True, adjust_tree=adjust_tree)

        # Reconstruct all triplet D/R combinations
        counts = self.npcf_multipoles_norm.reshape(
            self.nmax+1, 2*nz, 2*nz, 2*nz, self.nbinsr, self.nbinsr)
        D = slice(0, nz); R = slice(nz, 2*nz)
        DDD = counts[:, D, D, D]; DDR = counts[:, D, D, R]; DRD = counts[:, D, R, D]; RDD = counts[:, R, D, D]
        DRR = counts[:, D, R, R]; RDR = counts[:, R, D, R]; RRD = counts[:, R, R, D]; RRR = counts[:, R, R, R]

        # Get data/random normalisation
        if dotomo:
            WD = np.array([cat_data.weight[cat_data.zbins == z].sum() for z in range(nz)])
            WR = np.array([cat_rand.weight[cat_rand.zbins == z].sum() for z in range(nz)])
        else:
            WD = np.array([cat_data.weight.sum()])
            WR = np.array([cat_rand.weight.sum()])
        f = self.save_divide_npcf(WR, WD, fill=1.).astype(np.float64)
        fa = f[None, :, None, None, None, None]
        fb = f[None, None, :, None, None, None]
        fc = f[None, None, None, :, None, None]

        # Build estimator numerator and norm in multiple space
        nzc = nz*nz*nz
        numerator = (fa*fb*fc*DDD - fa*fb*DDR - fa*fc*DRD - fb*fc*RDD
                     + fa*DRR + fb*RDR + fc*RRD - RRR)
        numerator = numerator.reshape(self.nmax+1, nzc, self.nbinsr, self.nbinsr)
        RRR = RRR.reshape(self.nmax+1, nzc, self.nbinsr, self.nbinsr)

        # Transform to real space and filter out ~empty bins
        num_real = self._inverse_transform(numerator, nz, nz)
        rrr_real = self._inverse_transform(RRR, nz, nz)
        rtol = self.norm_divisionmask if count_floor_rtol is None else count_floor_rtol
        floor = rtol * np.abs(rrr_real).mean(axis=-1, keepdims=True)
        zeta = np.divide(num_real, rrr_real, out=np.zeros_like(num_real), where=rrr_real > floor)

        self.nbinsz = nz
        self.nzcombis = nzc
        self.zeta = zeta
        self.npcf = zeta[None]
        self.npcf_norm = rrr_real
        self.projection = "X"

    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        r"""Re-project the real-space NPCF into the given ``projection``."""
        super()._projectnpcf(self, projection)


class GGGCorrelation(BinnedNPCF):
    r""" Class containing methods to measure and obtain statistics that are built
    from third-order shear correlation functions.

    Note that the different components of the GGG correlator are ordered as
    
    .. math::

        \Gamma_\mu \sim \left[
        \langle \gamma \gamma \gamma \rangle,\,
        \langle \gamma^* \gamma \gamma \rangle,\,
        \langle \gamma \gamma^* \gamma \rangle,\,
        \langle \gamma \gamma \gamma^* \rangle
        \right].

    which is different to some conventions, but matches orpheus' conventions to
    have the complex conjugations in the correlators move from left to right.

    Parameters
    ----------
    n_cfs: int
        The number of independent components of the NPCF.
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.
    process_spherical: bool, optional
        Process spherical catalogs using curved-sky geometry and not via flat-sky
        patches. Defaults to ``False``.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    Additional child-specific parameters can be passed via ``kwargs``.
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.
    """
    
    def __init__(self, n_cfs, min_sep, max_sep, process_spherical=False, **kwargs):

        super().__init__(order=3, spins=np.array([2,2,2], dtype=np.int32), n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, **kwargs)
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

    def saveinst(self, path_save, fname, extr_pars=None):
        r"""Serialise the instance to a ``.npz`` archive."""
        extras = dict(nbinsz=self.nbinsz, nzcombis=self.nzcombis)
        if extr_pars: extras.update(extr_pars)
        super().saveinst(path_save, fname, extr_pars=extras)

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
                                        for i in range(pcorr.nbinsr)] for z in range(self.nbinsz)]) 
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
        self.bin_centers = self.save_divide_bins(self.bin_centers, _footnorm)
        self.bin_centers_mean = np.mean(self.bin_centers,axis=0)
        self.projection = "X"

        if keep_patchres:
            return centers_patches, npcf_multipoles_patches, npcf_multipoles_norm_patches
        
        
    def process(self, cat, cat_random=None, Pi=None, dpix=None, dpix_z=None,
                dotomo=True, rotsignflip=False, apply_edge_correction=False, adjust_tree=False,
                save_patchres=False, save_filebase="", keep_patchres=False):
        r"""Compute a shear 3PCF provided a shape catalog.

        Parameters
        ----------
        cat: orpheus.SpinTracerCatalog
            The shape catalog which is processed
        cat_random: orpheus.ScalarTracerCatalog, optional
            Galaxy random catalog for the RRR normalization. Required for the
            '3dbox' projected estimator; ignored otherwise.
        Pi: float, optional
            Line-of-sight projection length ('3dbox' only; required there).
        dpix, dpix_z: float, optional
            Transverse hash cell size and line-of-sight slab width ('3dbox' only).
        dotomo: bool
            Flag that decides whether the tomographic information in the shape catalog should be used. Defaults to ``True``.
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
            Base of the filenames in which the patches are saved. The full filename will be ``<save_patchres>/<save_filebase>_patchxx.npz``.
            Only has an effect if the shape catalog consists of multiple patches and ``save_patchres`` is not ``False``.
        keep_patchres: bool
            If the catalog consists of multiple patches, returns all measurements on the patches. Defaults to ``False``.
        """

        # The processing of the slabs in a 3dbox is quite different from the rest so it is outsourced for now
        if cat.geometry == '3dbox':
            assert cat_random is not None, "'3dbox' requires a random catalog (cat_random)."
            assert Pi is not None, "'3dbox' requires a projection length Pi."
            assert cat_random.geometry == '3dbox', "'3dbox' requires all catalogs in '3dbox' geometry."
            return self.__process_3dbox(cat, cat_random, float(Pi), dpix=dpix, dpix_z=dpix_z,
                                        dotomo=dotomo)

        # Check arguments for full-sky catalogs
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
                nbinsz = self.nbinsz
                if native_spherical:
                    sep2deg = convertunits(self.sep_units, 'deg')
                    nsides, nside_hash = self.tree_resos_to_nsides()
                    mh = cat.multihash_bundle(reso_redges=self.tree_redges*sep2deg, nsides=nsides,
                                              nside_hash=nside_hash, shuffle=self.shuffle_pix,
                                              w2field=True,
                                              verbose=self._verbose_python)
                    assert not mh['nav_coarsened'], (
                        "nav_coarsen is incompatible with the GGG doubletree. Only single-tree navigation "
                        "(NN, GG, NNNN) may coarsen the navigation.")
                    extra = {'e1_resos': mh['red_e1'], 'e2_resos': mh['red_e2'],
                             'weightsq_resos': mh['red_weightsq']}
                else:
                    cutfirst = np.int32(self.tree_resos[0]==0.)
                    mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.dpix_hash,
                                              shuffle=self.shuffle_pix, w2field=True, normed=True, nthreads=self.nthreads)
                    allfields = mh['allfields']
                    weight_resos = mh['weight_resos']
                    e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
                    e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
                    _weightsq_resos = np.concatenate([allfields[i][2] for i in range(len(allfields))]).astype(np.float64)
                    weightsq_resos = _weightsq_resos*weight_resos # reduce renorms all fields --> 'unrenorm'
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
                    'GGG', self.nbinsr, nmax=self.nmax, nbinsz=nbinsz)

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
                out_s, bin_centers, threepcfs_n, _, threepcfsnorm_n, _, _ = build_npcf_output(
                    'GGG', self.nbinsr, nmax=self.nmax, nbinsz=self.nbinsz)
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
                    mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.dpix_hash,
                                              shuffle=self.shuffle_pix, w2field=True, normed=True, nthreads=self.nthreads)
                    weight_resos = mh['weight_resos']
                    allfields = mh['allfields']
                    e1_resos = np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64)
                    e2_resos = np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)
                    _weightsq_resos = np.concatenate([allfields[i][2] for i in range(len(allfields))]).astype(np.float64)
                    weightsq_resos = _weightsq_resos*weight_resos # reduce renorms all fields --> 'unrenorm'
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
                    else:
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
        r"""Computes GGG/RRR in projected slabs of width +-Pi along z-direction in 3dbox.

        Note that the random counts are normalised by the factor :math:`f = W_S/W_R` per tomo
        bin to to effective number of observed shapes, so :math:`\Gamma \sim SSS / f^3 RRR`.
        """
        self._Pi = float(Pi)
        if dpix is None: dpix = self.max_sep
        if dpix_z is None: dpix_z = Pi

        # Tomo setup
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

        # Build the slab hashes for the cats on joint extent.
        cats = [cat_source, cat_random]
        ext = [min(c.min1 for c in cats), max(c.max1 for c in cats),
               min(c.min2 for c in cats), max(c.max2 for c in cats)]
        ext_z = [min(c.min3 for c in cats), max(c.max3 for c in cats)]
        mh_source = cat_source.multihash_bundle(dpix_hash=dpix, dpix_z=dpix_z, extent=ext, extent_z=ext_z)
        mh_rand = cat_random.multihash_bundle(dpix_hash=dpix, dpix_z=dpix_z, extent=ext, extent_z=ext_z)

        # Get rescaling f = W_S / W_R per shape tomo-bin.
        WS = np.array([cat_source.weight[cat_source.zbins == z].sum() for z in range(nz)])
        WR = np.array([cat_random.weight[cat_random.zbins == z].sum() for z in range(nz)])
        f = self.save_divide_npcf(WS, WR, fill=1.).astype(np.float64)

        # Build all the relevant args for the C call
        scomp = (4, self.nmax+1, self.nzcombis, self.nbinsr, self.nbinsr)
        sn = (self.nmax+1, self.nzcombis, self.nbinsr, self.nbinsr)
        szr = (nz, nz, self.nbinsr)
        out_s, bin_centers, Comp_n, _, RRR_n, _, _ = build_npcf_output(
            'GGG', self.nbinsr, nmax=self.nmax, nbinsz=nz, estimator_type='lslike_slab')
        bin_s = build_binning_struct(self, nmax=self.nmax, dccorr=self.multicountcorr, Pi=self._Pi)
        cat_c, keep_cc = build_slab_catalog_struct(mh_source, nz, e1e2=mh_source['fields'])
        nav_c, keep_nc = build_slab_navhash_struct(mh_source)
        cat_R, keep_cr = build_slab_catalog_struct(mh_rand, nz)
        nav_R, keep_nr = build_slab_navhash_struct(mh_rand)
        _alive = keep_cc + keep_nc + keep_cr + keep_nr

        self.clib.alloc_Gammans_slab_GGG(
            ct.byref(cat_c), ct.byref(nav_c), ct.byref(cat_R), ct.byref(nav_R),
            ct.byref(bin_s), int(self.nthreads), int(self._verbose_c), ct.byref(out_s))

        # Retrieve output and rescale to get the appropriate multipoles
        self._SSS = np.nan_to_num(Comp_n.reshape(scomp))
        self._RRR = np.nan_to_num(RRR_n.reshape(sn))
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
        r"""Edge-correct the measured multipoles by deconvolving the mode-coupling matrix; optionally returns the coupling matrices."""

        def gen_M_matrix(thet1,thet2,threepcf_n_norm):
            nvals, ntheta, _ = threepcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=int)
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
        threepcf_n_corr = np.zeros(threepcf_n_full.shape, dtype=complex)
        threepcf_n_full[:,nmax:] = self.npcf_multipoles
        threepcf_n_norm_full[nmax:] = self.npcf_multipoles_norm
        for nextn in range(1,nvals):
            threepcf_n_full[0,nmax-nextn] = self.npcf_multipoles[0,nextn].transpose(0,2,1)
            threepcf_n_full[1,nmax-nextn] = self.npcf_multipoles[1,nextn].transpose(0,2,1)
            threepcf_n_full[2,nmax-nextn] = self.npcf_multipoles[3,nextn].transpose(0,2,1)
            threepcf_n_full[3,nmax-nextn] = self.npcf_multipoles[2,nextn].transpose(0,2,1)
            threepcf_n_norm_full[nmax-nextn] = self.npcf_multipoles_norm[nextn].transpose(0,2,1)

        if ret_matrices:
            mats = np.zeros((nzcombis,ntheta,ntheta,2*nmax+1,2*nmax+1))
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
    
    def multipoles2npcf(self, projection='Centroid'):
        r"""Transforms the 3PCF from the multipole-basis using the 'X'-projection to the real-space-basis
        in a chose projection.
        """
        assert(projection in self.projections_avail)
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        nbinsphi = len(self.phi)
        thisnpcf = np.zeros(self.n_cfs*nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        thisnpcf_norm = np.zeros(nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        # This is how the 3pcf components need to ber permuted for n-->-n, see A.6 in Porth+23.
        conjmap = np.array([0, 1, 3, 2], dtype=np.int32)
        modeweight = self.mode_window(self.nmax)/nbinsphi
        floor_thr = np.zeros(nzcombis, dtype=np.float64)
        self.clib.multipoles2npcf_third_z1z23(
            self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(),
            np.int32(self.nmax), np.int32(self.n_cfs), np.int32(self.nbinsz), np.int32(self.nbinsz),
            np.int32(rbins),
            self.phi.astype(np.float64), np.int32(nbinsphi),
            np.int32(0), conjmap, modeweight,
            np.int32(self.is_edge_corrected), np.int32(1), floor_thr,
            np.int32(self.nthreads),
            thisnpcf, thisnpcf_norm)
        if projection == "Centroid":
            self.clib._x2centroid_ggg(
                thisnpcf, np.int32(self.nbinsz),
                self.bin_centers_mean, np.int32(rbins), self.phi.astype(np.float64), np.int32(nbinsphi),
                np.int32(self.nthreads))
        self.npcf = thisnpcf.reshape((self.n_cfs, nzcombis, rbins, rbins, nbinsphi))
        self.npcf_norm = thisnpcf_norm.reshape((nzcombis, rbins, rbins, nbinsphi))
        self.projection = projection
        self.set_ringing_sigma(modeweight[0], self.nmax)
            
    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        r"""Re-project the real-space NPCF into the given ``projection``."""
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
        

    def computeMap3(self, radii, do_multiscale=False, basis="MapMx", tofile=False):
        """Compute third-order aperture statistics using the polynomial filter.

        Parameters
        ----------
        radii: numpy.ndarray
            Aperture scales to be considered.
        do_multiscale: bool
            If set to true, compute the statistics on all combinations of aperture radii.
            Defaults to ``False``.
        basis: str, one of ``['MapMx','MM*','both']``.
            Decide in which output basis the aperture-statistics should be returend. Defaults
            to ``MapMx`` which is the common E/B-separating basis.
        tofile: bool
            No effect at the moment.

        Returns
        -------
        numpy.ndarray
            The third-order aperture-mass statistics.
        """

        assert(basis in ["MapMx", "MM*"])
        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf(projection='Centroid')

        if self.projection != "Centroid":
            self.projectnpcf("Centroid")

        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            _rcut = 1
        else:
            nrcombis = nradii*nradii*nradii
            _rcut = nradii
        R1s = np.zeros(nrcombis, dtype=np.float64)
        R2s = np.zeros(nrcombis, dtype=np.float64)
        R3s = np.zeros(nrcombis, dtype=np.float64)
        tmprcombi = 0
        for R1 in radii:
            for R2 in radii[:_rcut]:
                for R3 in radii[:_rcut]:
                    R1s[tmprcombi] = R1
                    R2s[tmprcombi] = R1 if not do_multiscale else R2
                    R3s[tmprcombi] = R1 if not do_multiscale else R3
                    tmprcombi += 1

        rawstats = np.zeros(4*self.nzcombis*nrcombis, dtype=np.complex128)
        self.clib.threepcf2M3correlators_ggg(
            self.npcf.flatten(), self.bin_edges.astype(np.float64), self.bin_centers_mean.astype(np.float64),
            np.int32(self.nbinsr), self.phi.astype(np.float64), np.int32(len(self.phi)), np.int32(self.nzcombis),
            R1s, R2s, R3s, np.int32(nrcombis), np.int32(do_multiscale), np.int32(self.nthreads),
            rawstats)
        
        if basis=="MM*":
            # Ordered as [ MMM, M*MM, MM*M, MMM* ]
            map3s = rawstats.reshape((4, self.nzcombis, nrcombis)) 
        
        if basis=="MapMx":
            M3, M2M1, M2M2, M2M3 = rawstats.reshape((4, self.nzcombis, nrcombis))
            map3s = np.zeros((8, self.nzcombis, nrcombis), dtype=float)
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


class GNNCorrelation(BinnedNPCF):
    r""" Class containing methods to measure and obtain statistics that are built
    from third-order source-lens-lens (G3L) correlation functions.

    Parameters
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
        # Only the discrete and doubletree kernels are dispatched in `process`; the other
        # two schemes would leave the multipoles at zero without raising.
        super().__init__(3, [2,0,0], n_cfs=1, min_sep=min_sep, max_sep=max_sep,
                         methods_avail=["Discrete", "DoubleTree"], **kwargs)
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

    def saveinst(self, path_save, fname, extr_pars=None):
        r"""Serialise the instance to a ``.npz`` archive."""
        extras = dict(nbinsz_source=self.nbinsz_source, nbinsz_lens=self.nbinsz_lens,
                      zweighting=self.zweighting, zweighting_sigma=self.zweighting_sigma)
        if extr_pars: extras.update(extr_pars)
        super().saveinst(path_save, fname, extr_pars=extras)


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
        self.bin_centers = self.save_divide_bins(self.bin_centers, _footnorm)
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
        r"""Compute a shear-lens-lens correlation provided a source and a lens catalog.

        Parameters
        ----------
        cat_source: orpheus.SpinTracerCatalog
            The source catalog which is processed
        cat_lens: orpheus.ScalarTracerCatalog
            The lens catalog which is processed
        cat_random: orpheus.ScalarTracerCatalog, optional
            Random catalog for the lens/position tracer. Required for the '3dbox'
            projected estimator; ignored otherwise.
        Pi: float, optional
            Line-of-sight projection length ('3dbox' only; required there).
        dpix, dpix_z: float, optional
            Transverse hash cell size and line-of-sight slab width ('3dbox' only).
        dotomo_source: bool
            Flag that decides whether the tomographic information in the source catalog should be used. Defaults to ``True``.
        dotomo_lens: bool
            Flag that decides whether the tomographic information in the lens catalog should be used. Defaults to ``True``.
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
        # For '3dbox' slab geometries the process function is quite different so we outsource it for now.
        if cat_source.geometry == '3dbox':
            assert cat_random is not None, "'3dbox' requires a random catalog (cat_random)."
            assert Pi is not None, "'3dbox' requires a projection length Pi."
            if cat_lens is None:
                cat_lens = ScalarTracerCatalog(
                    cat_source.pos1, cat_source.pos2, np.ones(cat_source.ngal),
                    pos3=cat_source.pos3, weight=cat_source.weight,
                    zbins=cat_source.zbins.copy(), geometry='3dbox')
            for c in (cat_lens, cat_random):
                assert c.geometry == '3dbox', "'3dbox' requires all catalogs in '3dbox'."
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
            out_s, bin_centers, Upsilon_n, _, Norm_n, _, _ = build_npcf_output(
                'GNN', self.nbinsr, nmax=self.nmax,
                nbinsz_lens=self.nbinsz_lens, nbinsz_source=self.nbinsz_source)
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
                mhs = cat_source.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.dpix_hash,
                                                  shuffle=self.shuffle_pix, normed=True, extent=jointextent, nthreads=self.nthreads)
                sallfields = mhs['allfields']
                e1_resos_source = np.concatenate([sallfields[i][0] for i in range(len(sallfields))]).astype(np.float64)
                e2_resos_source = np.concatenate([sallfields[i][1] for i in range(len(sallfields))]).astype(np.float64)
                cats_s, keep_cs = build_catalog_struct(
                    mhs, self.nbinsz_source, extra={'e1_resos': e1_resos_source, 'e2_resos': e2_resos_source})
                cats_s.nresos = int(self.tree_nresos)
                navs_s, keep_ns = build_navhash_struct(mhs, cat_obj=cat_source)
                mhl = cat_lens.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.dpix_hash,
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
        r"""Computes S(D-R)^2/RRR in projected slabs of width +-Pi along z-direction in 3dbox.

        Note that the random counts are normalised by the factor :math:`f = W_S/W_R` per tomo
        bin to to effective number of observed shapes, so i.e. :math:`\Gamma_{SDR} \sim SDR / f^2 RRR`.
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

        # Build the slab hashes on a joint extent
        ext = [min(cat_source.min1, cat_lens.min1, cat_random.min1),
               max(cat_source.max1, cat_lens.max1, cat_random.max1),
               min(cat_source.min2, cat_lens.min2, cat_random.min2),
               max(cat_source.max2, cat_lens.max2, cat_random.max2)]
        ext_z = [min(cat_source.min3, cat_lens.min3, cat_random.min3),
                 max(cat_source.max3, cat_lens.max3, cat_random.max3)]
        mh_lens = cat_lens.multihash_bundle(dpix_hash=dpix, dpix_z=dpix_z, extent=ext, extent_z=ext_z)
        mh_rand = cat_random.multihash_bundle(dpix_hash=dpix, dpix_z=dpix_z, extent=ext, extent_z=ext_z)
        assert mh_lens['npix'] == mh_rand['npix'] and mh_lens['nslabs'] == mh_rand['nslabs'], \
            "D and R slab hashes must share the grid (same dpix/extent)."

        # Get number counts rescaling f = W_D / W_R per tomo-bin.
        WD = np.array([cat_lens.weight[cat_lens.zbins == z].sum() for z in range(nzd)])
        WR = np.array([cat_random.weight[cat_random.zbins == z].sum() for z in range(nzd)])
        f = self.save_divide_npcf(WD, WR, fill=1.).astype(np.float64)

        assert nzs == nzd, "'3dbox' requires matching source/lens tomographic bins."

        # Build functino arguments. The correlators are sorted as [SDD, SDR, SRD, SRR]
        # so we need four components.
        _z3combis = nzs*nzd*nzd
        scomp = (4, self.nmax+1, _z3combis, self.nbinsr, self.nbinsr)
        sn = (self.nmax+1, _z3combis, self.nbinsr, self.nbinsr)
        szr = (nzs, nzd, self.nbinsr)
        out_s, bin_centers, Comp_n, _, RRR_n, _, _ = build_npcf_output(
            'GNN', self.nbinsr, nmax=self.nmax, nbinsz_lens=nzd, nbinsz_source=nzs,
            estimator_type='lslike_slab')

        # Build all catalog-based args
        # The source catalog dos not require a hash, so we need to emulate its dict 
        mhemu_source = {'pos1': cat_source.pos1, 'pos2': cat_source.pos2, 'pos3': cat_source.pos3, 
                        'weight': cat_source.weight, 'zbins': cat_source.zbins}
        cat_c, keep_c = build_slab_catalog_struct(mhemu_source, nzs,
                                                  e1e2=(cat_source.tracer_1, cat_source.tracer_2))
        cat_D, keep_D = build_slab_catalog_struct(mh_lens, nzd)
        nav_D, keep_nD = build_slab_navhash_struct(mh_lens)
        cat_R, keep_R = build_slab_catalog_struct(mh_rand, nzd)
        nav_R, keep_nR = build_slab_navhash_struct(mh_rand)
        bin_s = build_binning_struct(self, nmax=self.nmax, dccorr=self.multicountcorr, Pi=self._Pi)
        _alive = keep_c + keep_D + keep_nD + keep_R + keep_nR

        self.clib.alloc_Gammans_slab_GNN(
            ct.byref(cat_c), ct.byref(cat_D), ct.byref(nav_D), ct.byref(cat_R), ct.byref(nav_R),
            ct.byref(bin_s), ct.c_int32(self.nthreads), ct.c_int32(self._verbose_c),
            ct.byref(out_s))

        # Unpack output
        self._SDD, self._SDR, self._SRD, self._SRR = np.nan_to_num(Comp_n.reshape(scomp))
        self._RRR = np.nan_to_num(RRR_n.reshape(sn))

        # Apply rescaling to build an LS-like estimator
        zc_i, z2_i, z3_i = np.unravel_index(np.arange(_z3combis), (nzs, nzd, nzd))
        fc = f[zc_i].reshape(1, _z3combis, 1, 1)
        f2 = f[z2_i].reshape(1, _z3combis, 1, 1)
        f3 = f[z3_i].reshape(1, _z3combis, 1, 1)
        self.npcf_multipoles = (self._SDD - f3*self._SDR - f2*self._SRD + f2*f3*self._SRR)[None]
        self.npcf_multipoles_norm = fc*f2*f3*self._RRR
        self._normcountscale = np.mean(cat_random.weight)**3 * (fc*f2*f3).reshape(_z3combis)

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
        r"""Edge-correct the measured multipoles by deconvolving the mode-coupling matrix; optionally returns the coupling matrices."""
        assert(not self.is_edge_corrected)
        def gen_M_matrix(thet1,thet2,threepcf_n_norm):
            nvals, ntheta, _ = threepcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=int)
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
        threepcf_n_corr = np.zeros(threepcf_n_full.shape, dtype=complex)
        threepcf_n_full[:,nmax:] = self.npcf_multipoles
        threepcf_n_norm_full[nmax:] = self.npcf_multipoles_norm
        for nextn in range(1,nvals):
            threepcf_n_full[0,nmax-nextn] = self.npcf_multipoles[0,nextn].transpose(0,2,1)
            threepcf_n_norm_full[nmax-nextn] = self.npcf_multipoles_norm[nextn].transpose(0,2,1)
        
        if ret_matrices:
            mats = np.zeros((nzcombis,ntheta,ntheta,2*nmax+1,2*nmax+1))
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

    def multipoles2npcf(self, xi=None, count_floor_rtol=None):
        r"""Transforms the 3PCF from the multipole-basis to the real-space-basis.

        Parameters
        ----------
        xi: tuple, optional
            Angular clustering 2PCF of the lenses as ``(thetas, omega)``; enables the
            clustering correction recovering the pure correlator as in Simon+ 2013.
        """
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        nbinsphi = len(self.phi)
        thisnpcf = np.zeros(self.n_cfs*nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        thisnpcf_norm = np.zeros(nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        conjmap = np.array([0], dtype=np.int32)
        modeweight = self.mode_window(self.nmax)/(2*np.pi)
        _scale = getattr(self, '_normcountscale', None)
        floor_thr = np.zeros(nzcombis, dtype=np.float64) if \
            (count_floor_rtol is None or _scale is None) else \
            (count_floor_rtol*_scale).astype(np.float64)
        self.clib.multipoles2npcf_third_z1z23(
            self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(),
            np.int32(self.nmax), np.int32(self.n_cfs), np.int32(self.nbinsz_source), np.int32(self.nbinsz_lens),
            np.int32(rbins),
            self.phi.astype(np.float64), np.int32(nbinsphi),
            np.int32(0), conjmap, modeweight,
            np.int32(self.is_edge_corrected), np.int32(1), floor_thr,
            np.int32(self.nthreads),
            thisnpcf, thisnpcf_norm)
        self.npcf = thisnpcf.reshape((self.n_cfs, nzcombis, rbins, rbins, nbinsphi))
        self.npcf_norm = thisnpcf_norm.reshape((nzcombis, rbins, rbins, nbinsphi)).real
        self.projection = "X"
        self.set_ringing_sigma(modeweight[0], self.nmax)

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
        r"""Re-project the real-space NPCF into the given ``projection``."""
        super()._projectnpcf(self, projection)

    ## INTEGRATED MEASURES ##
    def computeNNM(self, radii, do_multiscale=False, xi=None, tofile=False):
        r"""Compute third-order aperture statistics using the polyonomial filter of Crittenden 2002.

         Parameters
        ----------
        radii: numpy.ndarray
            Aperture scales to be considered.
        do_multiscale: bool
            If set to true, compute the statistics on all combinations of aperture radii.
            Defaults to ``False``.
        xi: tuple, optional
            Angular clustering 2PCF of the lenses as ``(thetas, omega)``; enables the
            clustering correction recovering the pure correlator as in Simon+ 2013.
        tofile: bool
            No effect at the moment.

        Returns
        -------
        numpy.ndarray
            The third-order aperture statistics, complex, of shape
            ``(1, nzcombis, nrcombis)``. The real part is
            :math:`\langle N_\mathrm{ap} N_\mathrm{ap} M_\mathrm{ap}\rangle`, the
            imaginary part :math:`\langle N_\mathrm{ap} N_\mathrm{ap} M_\times\rangle`.
        """

        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf(xi=xi)

        nzcombis = self.nbinsz_source*self.nbinsz_lens*self.nbinsz_lens
        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            _rcut = 1
        else:
            nrcombis = nradii*nradii*nradii
            _rcut = nradii
        R1s = np.zeros(nrcombis, dtype=np.float64)
        R2s = np.zeros(nrcombis, dtype=np.float64)
        R3s = np.zeros(nrcombis, dtype=np.float64)
        tmprcombi = 0
        for R1 in radii:
            for R2 in radii[:_rcut]:
                for R3 in radii[:_rcut]:
                    R1s[tmprcombi] = R1
                    R2s[tmprcombi] = R1 if not do_multiscale else R2
                    R3s[tmprcombi] = R1 if not do_multiscale else R3
                    tmprcombi += 1

        rawstats = np.zeros(nzcombis*nrcombis, dtype=np.complex128)
        self.clib.threepcf2NNMcorrelators_gnn(
            self.npcf[0].flatten(), self.bin_edges.astype(np.float64), self.bin_centers_mean.astype(np.float64),
            np.int32(self.nbinsr), self.phi.astype(np.float64), np.int32(len(self.phi)), np.int32(nzcombis),
            R1s, R2s, R3s, np.int32(nrcombis), np.int32(self.nthreads),
            rawstats)
        NNM = rawstats.reshape((1, nzcombis, nrcombis))
        return NNM


class NGGCorrelation(BinnedNPCF):
    r""" Class containing methods to measure and obtain statistics that are built
    from third-order lens-shear-shear correlation functions.

    Note that the different components of the NGG correlator are ordered as

    .. math::

            \left[ \tilde{G}_-, \tilde{G}_+, \right] \ ,

    which is different to the usual conventions, but matches orpheus' conventions to
    always start with a correlator in which no polar field is complex conjugated.

    Parameters
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
    """
    def __init__(self, min_sep, max_sep, **kwargs):
        
        # No basetree kernel is dispatched in `process`; that scheme would leave the
        # multipoles at zero without raising.
        super().__init__(3, [0,2,2], n_cfs=2, min_sep=min_sep, max_sep=max_sep,
                         methods_avail=["Discrete", "Tree", "DoubleTree"], **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None, "X"]
        self.nbinsz_source = None
        self.nbinsz_lens = None

        # (Add here any newly implemented projections)
        self._initprojections(self)

    def saveinst(self, path_save, fname, extr_pars=None):
        r"""Serialise the instance to a ``.npz`` archive."""
        extras = dict(nbinsz_source=self.nbinsz_source, nbinsz_lens=self.nbinsz_lens)
        if extr_pars: extras.update(extr_pars)
        super().saveinst(path_save, fname, extr_pars=extras)

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
        self.bin_centers = self.save_divide_bins(self.bin_centers, _footnorm)
        self.bin_centers_mean =np.mean(self.bin_centers, axis=(0,1))
        self.projection = "X"

        if keep_patchres:
            return centers_patches, npcf_multipoles_patches, npcf_multipoles_norm_patches

    def __process_3dbox(self, cat_source, cat_lens, cat_random, Pi,
                        dpix=None, dpix_z=None, dotomo_source=True, dotomo_lens=True):
        r"""Computes S(D-R)^2/RRR in projected slabs of width +-Pi along z-direction in 3dbox.

        Note that the random counts are normalised by the factor :math:`f = W_S/W_R` per tomo
        bin to to effective number of observed shapes, so i.e. :math:`\Gamma_{RSS} \sim RSS / f RRR`.
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
        
        assert nzs == nzl, "'3dbox' requires matching source/lens tomographic bins."

        # Build slab hash on joint extent
        cats = [cat_source, cat_lens, cat_random]
        ext = [min(c.min1 for c in cats), max(c.max1 for c in cats),
               min(c.min2 for c in cats), max(c.max2 for c in cats)]
        ext_z = [min(c.min3 for c in cats), max(c.max3 for c in cats)]

        mh_source = cat_source.multihash_bundle(dpix_hash=dpix, dpix_z=dpix_z, extent=ext, extent_z=ext_z)
        mh_rand = cat_random.multihash_bundle(dpix_hash=dpix, dpix_z=dpix_z, extent=ext, extent_z=ext_z)

        # Density-weight rescaling f = W_D / W_R per lens tomo-bin.
        WD = np.array([cat_lens.weight[cat_lens.zbins == z].sum() for z in range(nzl)])
        WR = np.array([cat_random.weight[cat_random.zbins == z].sum() for z in range(nzl)])
        f = self.save_divide_npcf(WD, WR, fill=1.).astype(np.float64)

        # Output: We order the two correlators as  [DSS, RSS]
        _z3combis = nzl*nzs*nzs
        nmp = 2*self.nmax+1
        scomp = (2, self.n_cfs, nmp, _z3combis, self.nbinsr, self.nbinsr)
        sn = (nmp, _z3combis, self.nbinsr, self.nbinsr)
        szr = (nzl, nzs, self.nbinsr)
        out_s, bin_centers, Comp_n, _, RRR_n, _, _ = build_npcf_output(
            'NGG', self.nbinsr, nmax=self.nmax, nbinsz_lens=nzl, nbinsz_source=nzs,
            estimator_type='lslike_slab')
        bin_s = build_binning_struct(self, nmax=int(self.nmax), dccorr=int(self.multicountcorr), Pi=self._Pi)

        # Build all catalog-based args
        # The lens catalog dos not require a hash, so we need to emulate its dict 
        mhemu_lens = {'pos1': cat_lens.pos1, 'pos2': cat_lens.pos2, 'pos3': cat_lens.pos3, 
                      'weight': cat_lens.weight, 'zbins': cat_lens.zbins}
        catlD, keep_cl = build_slab_catalog_struct(mhemu_lens, nzl)
        catlR, keep_cr = build_slab_catalog_struct(mh_rand, nzl)
        navlR, keep_nr = build_slab_navhash_struct(mh_rand)
        catsD, keep_cs = build_slab_catalog_struct(mh_source, nzs, e1e2=mh_source['fields'])
        navsD, keep_ns = build_slab_navhash_struct(mh_source)
        _alive = keep_cl + keep_cr + keep_nr + keep_cs + keep_ns

        self.clib.alloc_Gammans_slab_NGG(
            ct.byref(catlD), ct.byref(catlR), ct.byref(catsD), ct.byref(navsD),
            ct.byref(navlR), ct.byref(bin_s),
            int(self.nthreads), int(self._verbose_c), ct.byref(out_s))

        # Raw f-free sub-correlators (private, for further analysis).
        _DSS, _RSS = np.nan_to_num(Comp_n.reshape(scomp))
        self._DSS, self._RSS = _DSS, _RSS
        self._RRR = np.nan_to_num(RRR_n.reshape(sn))

        # Recombine with f = W_D/W_R
        zc_i, z2_i, z3_i = np.unravel_index(np.arange(_z3combis), (nzl, nzs, nzs))
        fc = f[zc_i]; f2 = f[z2_i]; f3 = f[z3_i]
        self.npcf_multipoles = _DSS - fc.reshape(1, 1, _z3combis, 1, 1)*_RSS
        self.npcf_multipoles_norm = (fc*f2*f3).reshape(1, _z3combis, 1, 1)*self._RRR
        # Little helper that helps us to identify empty bins for f!=1.
        self._normcountscale = np.mean(cat_random.weight)**3 * (fc*f2*f3)

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
        r"""Compute a lens-shear-shear correlation provided a source and a lens catalog.

        Parameters
        ----------
        cat_source: orpheus.SpinTracerCatalog
            The source catalog which is processed
        cat_lens: orpheus.ScalarTracerCatalog
            The lens catalog which is processed
        dotomo_source: bool
            Flag that decides whether the tomographic information in the source catalog should be used. Defaults to ``True``.
        dotomo_lens: bool
            Flag that decides whether the tomographic information in the lens catalog should be used. Defaults to ``True``.
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

        Returns
        -------
        None
            Results are stored on the instance in the multipole basis (``npcf_multipoles``,
            ``npcf_multipoles_norm``, ``bin_centers``); call :meth:`multipoles2npcf` to obtain the
            real-space basis. If the catalog is decomposed into patches and ``keep_patchres=True``,
            the per-patch measurements are returned instead.
        """

        # The '3dbox' process is fairly different from the rest, so we outsource it for now
        if cat_source.geometry == '3dbox':
            assert cat_random is not None, "'3dbox' requires a random catalog (cat_random)."
            assert Pi is not None, "'3dbox' requires a projection length Pi."
            if cat_lens is None:
                cat_lens = ScalarTracerCatalog(
                    cat_source.pos1, cat_source.pos2, np.ones(cat_source.ngal),
                    pos3=cat_source.pos3, weight=cat_source.weight,
                    zbins=cat_source.zbins.copy(), geometry='3dbox')
            assert cat_lens.geometry == '3dbox' and cat_random.geometry == '3dbox', \
                "'3dbox' requires all catalogs in '3dbox'."
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
            # Build output arrays
            out_s, bin_centers, Upsilon_n, _, Norm_n, _, _ = build_npcf_output(
                'NGG', self.nbinsr, nmax=self.nmax,
                nbinsz_lens=self.nbinsz_lens, nbinsz_source=self.nbinsz_source)
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
                mhs = cat_source.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.dpix_hash,
                                                  shuffle=self.shuffle_pix, normed=True, extent=jointextent, nthreads=self.nthreads)
                sallfields = mhs['allfields']
                e1_resos_source = np.concatenate([sallfields[i][0] for i in range(len(sallfields))]).astype(np.float64)
                e2_resos_source = np.concatenate([sallfields[i][1] for i in range(len(sallfields))]).astype(np.float64)
                cats_s, keep_cs = build_catalog_struct(
                    mhs, self.nbinsz_source, extra={'e1_resos': e1_resos_source, 'e2_resos': e2_resos_source})
                cats_s.nresos = int(self.tree_nresos)
                navs_s, keep_ns = build_navhash_struct(mhs, cat_obj=cat_source)
                mhl = cat_lens.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.dpix_hash,
                                                shuffle=self.shuffle_pix, normed=True, extent=jointextent, nthreads=self.nthreads)
                navl_s, keep_nl = build_navhash_struct(mhl, cat_obj=cat_lens)
            if self.method=="Tree":
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
        r"""Edge-correct the measured multipoles by deconvolving the mode-coupling matrix; optionally returns the coupling matrices."""
        
        assert(not self.is_edge_corrected)
        def gen_M_matrix(thet1,thet2,threepcf_n_norm):
            nvals, ntheta, _ = threepcf_n_norm.shape
            nmax = (nvals-1)//2
            narr = np.arange(-nmax,nmax+1, dtype=int)
            nextM = np.zeros((nvals,nvals))
            for ind, ell in enumerate(narr):
                lminusn = ell-narr
                sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
                nextM[ind,sel] = threepcf_n_norm[(lminusn+nmax)[sel],thet1,thet2].real / threepcf_n_norm[nmax,thet1,thet2].real
            return nextM
    
        _nvals, nzcombis, ntheta, _ = self.npcf_multipoles_norm.shape
        threepcf_n_corr = np.zeros_like(self.npcf_multipoles)
        if ret_matrices:
            mats = np.zeros((nzcombis,ntheta,ntheta,_nvals,_nvals))
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
    
    def multipoles2npcf(self, integrated=False, count_floor_rtol=None):
        r"""Transforms the 3PCF from the multipole-basis using the to the real-space-basis.
        """
        _, nzcombis, rbins, rbins = np.shape(self.npcf_multipoles[0])
        nbinsphi = len(self.phi)
        dphi = self.phi[1] - self.phi[0]
        thisnpcf = np.zeros(self.n_cfs*nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        thisnpcf_norm = np.zeros(nzcombis*rbins*rbins*nbinsphi, dtype=np.complex128)
        conjmap = np.arange(self.n_cfs, dtype=np.int32)
        if integrated:
            korder = np.arange(1, self.nmax+1)
            modeweight = np.empty(self.nmax+1, dtype=np.float64)
            modeweight[0] = dphi
            modeweight[1:] = 2./korder * np.sin(korder*dphi/2.)
        else:
            modeweight = np.full(self.nmax+1, dphi, dtype=np.float64)
        modeweight = (self.mode_window(self.nmax)*modeweight/(2*np.pi)).astype(np.float64)
        _scale = getattr(self, '_normcountscale', None)
        floor_thr = np.zeros(nzcombis, dtype=np.float64) if \
            (count_floor_rtol is None or _scale is None) else \
            (count_floor_rtol*_scale).astype(np.float64)
        self.clib.multipoles2npcf_third_z1z23(
            self.npcf_multipoles.flatten(), self.npcf_multipoles_norm.flatten(),
            np.int32(self.nmax), np.int32(self.n_cfs), np.int32(self.nbinsz_lens), np.int32(self.nbinsz_source),
            np.int32(rbins),
            self.phi.astype(np.float64), np.int32(nbinsphi),
            np.int32(1), conjmap, modeweight,
            np.int32(self.is_edge_corrected), np.int32(1), floor_thr,
            np.int32(self.nthreads),
            thisnpcf, thisnpcf_norm)
        self.npcf = thisnpcf.reshape((self.n_cfs, nzcombis, rbins, rbins, nbinsphi))
        self.npcf_norm = thisnpcf_norm.reshape((nzcombis, rbins, rbins, nbinsphi)).real
        self.projection = "X"
        self.set_ringing_sigma(modeweight[0], self.nmax, full_range=True)

    ## PROJECTIONS ##
    def projectnpcf(self, projection):
        r"""Re-project the real-space NPCF into the given ``projection``."""
        super()._projectnpcf(self, projection)
        
    ## INTEGRATED MEASURES ##        
    def computeNMM(self, radii, do_multiscale=False,  basis="MapMx", tofile=False):
        r"""Compute third-order aperture statistics.

        Returns
        -------
        numpy.ndarray
            For ``basis='MM*'`` the two raw correlators
            :math:`[\langle N MM \rangle, \langle N MM^* \rangle]`. For ``basis='MapMx'``
            the four real components

            .. math::

                \left[ \langle N M_\mathrm{ap} M_\mathrm{ap} \rangle,\,
                       \langle N M_\times M_\times \rangle,\,
                       \langle N M_\times M_\mathrm{ap} \rangle,\,
                       \langle N M_\mathrm{ap} M_\times \rangle \right] \ ,

            where the last two coincide for ``do_multiscale=False`` but are independent
            once the two aperture radii differ.
        """

        assert(basis in ["MM*", "MapMx"])

        if self.npcf is None and self.npcf_multipoles is not None:
            self.multipoles2npcf()

        nzcombis = self.nbinsz_lens*self.nbinsz_source*self.nbinsz_source
        nradii = len(radii)
        if not do_multiscale:
            nrcombis = nradii
            _rcut = 1
        else:
            nrcombis = nradii*nradii*nradii
            _rcut = nradii
        R1s = np.zeros(nrcombis, dtype=np.float64)
        R2s = np.zeros(nrcombis, dtype=np.float64)
        R3s = np.zeros(nrcombis, dtype=np.float64)
        tmprcombi = 0
        for R1 in radii:
            for R2 in radii[:_rcut]:
                for R3 in radii[:_rcut]:
                    R1s[tmprcombi] = R1
                    R2s[tmprcombi] = R1 if not do_multiscale else R2
                    R3s[tmprcombi] = R1 if not do_multiscale else R3
                    tmprcombi += 1

        rawstats = np.zeros(2*nzcombis*nrcombis, dtype=np.complex128)
        self.clib.threepcf2NMMcorrelators_ngg(
            self.npcf.flatten(), self.bin_edges.astype(np.float64), self.bin_centers_mean.astype(np.float64),
            np.int32(self.nbinsr), self.phi.astype(np.float64), np.int32(len(self.phi)), np.int32(nzcombis),
            R1s, R2s, R3s, np.int32(nrcombis), np.int32(self.nthreads),
            rawstats)

        if basis=="MM*":
            NMM = rawstats.reshape((2, nzcombis, nrcombis))
        if basis=="MapMx":
            _NMM, _NMMstar = rawstats.reshape((2, nzcombis, nrcombis))
            NMM = np.zeros((4, nzcombis, nrcombis), dtype=float)
            NMM[0] = (_NMM + _NMMstar).real/2.   # NMapMap
            NMM[1] = (-_NMM + _NMMstar).real/2.  # NMxMx
            NMM[2] = (_NMM + _NMMstar).imag/2.   # NMxMap
            NMM[3] = (_NMM - _NMMstar).imag/2.   # NMapMx

        return NMM