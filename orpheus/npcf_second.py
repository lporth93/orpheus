import numpy as np
import ctypes as ct
from pathlib import Path
import copy

from .catalog import Catalog, ScalarTracerCatalog, SpinTracerCatalog
from .npcf_base import BinnedNPCF
from .utils import convertunits
from .multires_structs import (build_catalog_struct, build_navhash_struct,
                               build_tree_params_struct, build_binning_struct,
                               build_npcf_output,
                               MultiresoCatalog, NavHash, TreeResoParams,
                               BinningParams, NPCFOutput)

__all__ = ["NNCorrelation", "GGCorrelation", "NGCorrelation"]


###############################   
## SECOND - ORDER STATISTICS ##
###############################

class NNCorrelation(BinnedNPCF):
    r"""Compute pair counts and (optionally) the projected angular clustering two-point correlation function.

    Parameters
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.
    shuffle_pix: int, optional
        Choice of how to define centers of the cells in the spatial hash structure.
        Defaults to ``1``, i.e. random positioning.
    **kwargs
        Passed to :class:`~orpheus.npcf_base.BinnedNPCF`.


    Attributes
    ----------
    npair: numpy.ndarray
        The number of unweighted pairs.
    npair_cell: numpy.ndarray
        The number cell-pairs.
    xi: numpy.ndarray
        The scalar two-point correlation function.

    Notes
    -----
    - Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    - Additional child-specific parameters can be passed via ``kwargs``.

    - Binning:
      - Either ``nbinsr`` or ``binsize`` must be provided to fix the binning scheme.
      - If both are provided, the parent class rules determine which takes precedence.

    - Pixel hashing / grid setup:
      - ``shuffle_pix=1`` is the default (random cell centers).
      - This differs from shear-based correlation functions where another default may be used.

    - Estimator:
      The scalar correlation function ``xi`` is formed from the pair counts via the Landy-Szalay estimator

      .. math::

         \xi(r) = \frac{DD(r) - 2\,DR(r) + RR(r)}{RR(r)}.

    """

    def __init__(self, min_sep, max_sep, shuffle_pix=1, process_spherical=False, **kwargs):
        super().__init__(order=2, spins=np.array([0,0], dtype=np.int32), n_cfs=1, min_sep=min_sep, max_sep=max_sep, shuffle_pix=shuffle_pix, **kwargs)
        # Native curved-sky pair counts (geodesic distance + nested-HEALPix
        # query_disc navigation) instead of patch decomposition. See
        # Catalog.multihash_spherical / alloc_nn_doubletree (metric=SPHERICAL).
        self.process_spherical = bool(process_spherical)
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        self.nzcombis = None
        self.npair = None
        self.npair_cell = None
        self.xi = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)

    def saveinst(self, path_save, fname):

        if not Path(path_save).is_dir():
            raise ValueError('Path to directory does not exist.')
        
        np.savez(path_save+fname,
                 nbinsz=self.nbinsz,
                 min_sep=self.min_sep,
                 max_sep=self.max_sep,
                 binsr=self.nbinsr,
                 method=self.method,
                 shuffle_pix=self.shuffle_pix,
                 tree_resos=self.tree_resos,
                 rmin_pixsize=self.rmin_pixsize,
                 resoshift_leafs=self.resoshift_leafs,
                 minresoind_leaf=self.minresoind_leaf,
                 maxresoind_leaf=self.maxresoind_leaf,
                 nthreads=self.nthreads,
                 bin_centers=self.bin_centers,
                 bin_centers_mean=self.bin_centers_mean,
                 xi=self.xi,
                 npair=self.npair,
                 npair_cell=self.npair_cell)

    def __process_patches(self, cat, dotomo=True,  do_dc=False, adjust_tree=False,
                          save_patchres=False, save_filebase="", keep_patchres=False):

        if save_patchres:
            if not Path(save_patchres).is_dir():
                raise ValueError('Path to directory does not exist.')
            
        for elp in range(cat.npatches):
            if self._verbose_python:
                print('Doing patch %i/%i'%(elp+1,cat.npatches))
            
            # Compute statistics on patch
            pcat = cat.frompatchind(elp)
            pcorr = NNCorrelation(
                min_sep=self.min_sep,
                max_sep=self.max_sep,
                nbinsr=self.nbinsr,
                method=self.method,
                shuffle_pix=self.shuffle_pix,
                tree_resos=self.tree_resos,
                rmin_pixsize=self.rmin_pixsize,
                resoshift_leafs=self.resoshift_leafs,
                minresoind_leaf=self.minresoind_leaf,
                maxresoind_leaf=self.maxresoind_leaf,
                nthreads=self.nthreads,
                verbosity=self.verbosity)
            pcorr.process(pcat, dotomo=dotomo, do_dc=do_dc)
            
            # Update the total measurement
            if elp == 0:
                self.nbinsz = pcorr.nbinsz
                self.nzcombis = pcorr.nzcombis
                self.bin_centers = np.zeros_like(pcorr.bin_centers)
                self.npair = np.zeros_like(pcorr.npair)
                self.npair_cell = np.zeros_like(pcorr.npair_cell)
                if keep_patchres:
                    centers_patches = np.zeros((cat.npatches, *pcorr.bin_centers.shape), dtype=pcorr.bin_centers.dtype)
                    npair_patches = np.zeros((cat.npatches, *pcorr.npair.shape), dtype=pcorr.npair.dtype)
                    npair_cell_patches = np.zeros((cat.npatches, *pcorr.npair_cell.shape), dtype=pcorr.npair_cell.dtype)
            self.bin_centers += pcorr.npair*pcorr.bin_centers
            self.npair += pcorr.npair
            self.npair_cell += pcorr.npair_cell
            if keep_patchres:
                centers_patches[elp] += pcorr.bin_centers
                npair_patches[elp] += pcorr.npair
                npair_cell_patches[elp] += pcorr.npair_cell
            if save_patchres:
                pcorr.saveinst(save_patchres, save_filebase+'_patch%i'%elp)

        # Finalize the measurement on the full footprint
        self.bin_centers /= self.npair
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)

        if keep_patchres:
            return centers_patches, npair_patches, npair_cell_patches
    
    def process(self, cat, cat_random=None, dotomo=True, do_dc=False, adjust_tree=False,
                save_patchres=False, save_filebase="", keep_patchres=False):
        r"""
        Compute NN pair counts for a catalog, and optionally the clustering 2PCF ``xi``.

        If ``cat_random`` is provided, ``xi`` is computed using the Landy–Szalay estimator.
        Otherwise only pair counts are computed.

        Parameters
        ----------
        cat: orpheus.ScalarTracerCatalog
            The (clustered) catalog for which the pair counts are computed
        cat_random: orpheus.ScalarTracerCatalog, optional
            A random catalog. If this is set, the clustering correlation function ``xi`` is computed.
        dotomo: bool
            Flag that decides whether the tomographic information in the catalog should be used. Defaults to `True`.
        do_dc: bool
            Flag that decides whether to double-count the pair counts. This will be required when looking at data-random pairs.
            within a tomographic catalog. Defaults to `True`. In case ``xi`` is computed, this argument is internally set to `True`.
        adjust_tree: bool
            Overrides the original setup of the tree-approximations in the instance based on the nbar of the catalog.
            Not implemented yet; has no effect. Defaults to ``False``.
        save_patchres: bool or str
            If the catalog has been decomposed in patches, flag whether to save the NN measurements on the individual patches. 
            Note that the path needs to exist, otherwise a `ValueError` is raised. For a flat-sky catalog this parameter 
            has no effect. Defaults to `False`.
        save_filebase: str
            Base of the filenames in which the patches are saved. The full filename will be `<save_patchres>/<save_filebase>_patchxx.npz`.
            Only has an effect if the catalog consists of multiple patches and `save_patchres` is not `False`.
        keep_patchres: bool
            If the catalog consists of multiple patches, returns all measurements on the patches. Defaults to `False`.
        """

        # If random catalog present, compute xi via the Landy-Szalay estimator.
        if cat_random is not None:
            assert(isinstance(cat_random, ScalarTracerCatalog))
            if not do_dc: print("Warning: for 2pt-clustering double-counting is enforced. do_dc set to True")
            self.__compute_xi(cat, cat_random, dotomo=dotomo, adjust_tree=adjust_tree,
                   save_patchres=save_patchres, keep_patchres=keep_patchres, estimator="LS")
            return

        # Native curved-sky pair counts (geodesic distance + nested-HEALPix
        # query_disc) require process_spherical; otherwise a spherical catalog
        # must first be decomposed into patches. Flat (or patched) catalogs take
        # the pixel-box path. The struct-based alloc_nn_doubletree dispatches on
        # cat->metric, so both geometries share the call block below.
        native_spherical = self.process_spherical and cat.geometry == 'spherical'
        if cat.geometry == 'spherical' and not native_spherical and cat.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches '
                             'using the Catalog._topatches method, or process_spherical=True must be set.')
        if cat.patchinds is not None and not native_spherical:
            return self.__process_patches(cat, dotomo=dotomo, do_dc=do_dc, adjust_tree=adjust_tree,
                                          save_patchres=save_patchres, save_filebase=save_filebase,
                                          keep_patchres=keep_patchres)

        # Tomography setup (shared by flat and native-spherical)
        self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            old_zbins = cat.zbins[:]
            cat.zbins = np.zeros(cat.ngal, dtype=np.int32)
            self.nzcombis = 1
        else:
            self.nbinsz = cat.nbinsz
            self.nzcombis = self.nbinsz*self.nbinsz
        nbinsz = self.nbinsz
        sz2r = (nbinsz*nbinsz, self.nbinsr)

        # Build the multihash bundle
        if native_spherical:
            from healpy import nside2resol
            sep2rad = convertunits(self.sep_units, 'rad')
            sep2deg = convertunits(self.sep_units, 'deg')
            # HEALPix nside per band: smallest nside whose pixel is <= the band cell;
            # tree_resos[r]==0 marks a discrete band.
            def _nside_for(target_rad):
                ns = 1
                while nside2resol(ns) > target_rad and ns < 2**29:
                    ns *= 2
                return ns
            nsides = [0 if self.tree_resos[r]==0. else _nside_for(self.tree_resos[r]*sep2rad)
                      for r in range(self.tree_nresos)]
            nside_hash = _nside_for(max(self.min_sep, 0.5*self.tree_redges[1])*sep2rad)
            # multihash_spherical reads reso_redges in degrees (-> radians internally),
            # so hand it tree_redges converted from sep_units to degrees.
            mh = cat.multihash_bundle(reso_redges=self.tree_redges*sep2deg, nsides=nsides,
                                      shuffle=self.shuffle_pix, nside_hash=nside_hash, verbose=self._verbose_python)
        else:
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                      shuffle=self.shuffle_pix, normed=False, nthreads=self.nthreads)

        # Build the four input structs + output arrays
        cat_s, keep_cat = build_catalog_struct(mh, nbinsz)
        # The flat bundle reports nresos = #levels-1; the C rshift/loop convention
        # counts every tree level, so use tree_nresos (matches the legacy call).
        cat_s.nresos = int(self.tree_nresos)
        nav_s, keep_nav = build_navhash_struct(mh, cat_obj=cat)
        tree_s, keep_tree = build_tree_params_struct(self, mh)
        maxleaf = max(0, self.tree_nresos-1)
        tree_s.minresoind_leaf = min(int(self.minresoind_leaf), maxleaf)
        tree_s.maxresoind_leaf = min(int(self.maxresoind_leaf), maxleaf)
        scale = convertunits(self.sep_units, 'rad') if native_spherical else None
        bin_s = build_binning_struct(self, do_dc=do_dc, scale=scale)
        # NN: the weighted pair count lives in the NPCFOutput 'norm' slot; the
        # integer cell count in 'npair_cell' (npcf/norm_mp/npair unused).
        out_s, bin_centers, _, npair, _, _, npair_cell = build_npcf_output(
            'nn', self.nbinsr, nbinsz=nbinsz)

        # Keep every numpy array referenced only through a struct field alive across
        # the C call (ctypes structs are invisible to Python's garbage collector).
        _alive = keep_cat + keep_nav + keep_tree   # noqa: F841

        self.clib.alloc_nn_doubletree(
            ct.byref(cat_s), ct.byref(nav_s), ct.byref(tree_s), ct.byref(bin_s),
            int(self.nthreads),
            int(self._verbose_c)+int(self._verbose_debug),
            ct.byref(out_s))

        # Unpack results
        if native_spherical:
            bin_centers /= convertunits(self.sep_units, 'rad')   # radians -> sep_units

        self.bin_centers = bin_centers.reshape(sz2r)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.npair = npair.reshape(sz2r)
        self.npair_cell = npair_cell.reshape(sz2r)
        self.projection = None
        if not do_dc:self.npair*=2

        if not dotomo:
            cat.zbins = old_zbins

        return

    def __compute_xi(self, cat_data, cat_rand, dotomo=True, adjust_tree=False,
                   save_patchres=False, keep_patchres=False, estimator="LS"):

        # Define joint tomographic bins across data and random catalog
        zbins = np.zeros(cat_data.ngal + cat_rand.ngal, dtype=int)
        zbins[:cat_data.ngal] += cat_data.zbins
        zbins[cat_data.ngal:] += cat_data.nbinsz + cat_rand.zbins
        if not dotomo:
            zbins[:cat_data.ngal] = 0
            zbins[cat_data.ngal:] = 1

        # Define joint catalog by appending randoms to data. This means it will have nz_joint=2*nz_data ordered as
        # Z_1=Z_1_data, ..., Z_nz=Z_nz_data, Z_nz+1=Z_1_rand, ..., Z_2nz=Z_nz_rand
        joint_cat = ScalarTracerCatalog(
            pos1=np.append(cat_data.pos1, cat_rand.pos1),
            pos2=np.append(cat_data.pos2, cat_rand.pos2),
            tracer=np.ones(cat_data.ngal + cat_rand.ngal),
            geometry=cat_data.geometry,
            units_pos1= cat_data.units_pos1,
            units_pos2= cat_data.units_pos1,
            zbins=zbins)
        
        # In case of a spherical geometry but no spherical projection, decompose the joint catalog 
        # in patches of roughly equal size 
        if cat_data.geometry=="spherical" and not self.process_spherical:
            joint_cat.topatches(npatches=cat_data.npatches, 
                                patchextend_deg=cat_data.patchinds['info']['patchextend_deg'],
                                nside_hash=cat_data.patchinds['info']['nside_hash'],
                                method=cat_data.patchinds['info']['method'],
                                kmeanshp_maxiter=cat_data.patchinds['info']['kmeanshp_maxiter'],
                                kmeanshp_tol=cat_data.patchinds['info']['kmeanshp_tol'],
                                kmeanshp_randomstate=cat_data.patchinds['info']['kmeanshp_randomstate'],
                                healpix_nside=cat_data.patchinds['info']['healpix_nside'])
        
        # Compute NN counts of joint catalog
        self.process(cat=joint_cat, dotomo=True, do_dc=True, adjust_tree=adjust_tree,
                     save_patchres=save_patchres, keep_patchres=keep_patchres)
        
        # Now infer all the tomographic dd,dr,rd,rr pairs pairs from the joint correlator
        # From the z-binning of the joint catalog given above the 2pcf will have the block structure
        # DD DR
        # RD RR
        # where each block is of shape (nz, nz) and the ordering of the indices is the same across all blocks.
        _zshift = cat_data.nbinsz
        _creshape = self.npair.reshape((2*_zshift, 2*_zshift, self.nbinsr))
        dds = _creshape[:_zshift,:_zshift].reshape((_zshift*_zshift, self.nbinsr))
        rrs = _creshape[_zshift:,_zshift:].reshape((_zshift*_zshift, self.nbinsr))
        drs = _creshape[:_zshift,_zshift:].reshape((_zshift*_zshift, self.nbinsr))
        rds = _creshape[_zshift:,:_zshift].reshape((_zshift*_zshift, self.nbinsr))

        # Get number of galaxies per tomo bin
        _, ngal_zdata = np.unique(cat_data.zbins, return_counts=True)
        _, ngal_zrand = np.unique(cat_rand.zbins, return_counts=True)
        ngal_zdata = ngal_zdata.astype(float)
        ngal_zrand = ngal_zrand.astype(float)
        # Get prefactors of LS estimator
        ngal_zrand_second = np.outer(ngal_zrand,(ngal_zrand-1))
        pref_DD = np.outer(ngal_zrand,(ngal_zrand-1))/np.outer(ngal_zdata,(ngal_zdata-1))
        pref_DR, pref_RD = np.meshgrid(ngal_zrand/ngal_zdata,ngal_zrand/ngal_zdata)
        pref_DD = pref_DD.flatten()[:, np.newaxis]
        pref_DR = pref_DR.flatten()[:, np.newaxis]
        pref_RD = pref_RD.flatten()[:, np.newaxis]
        
        # Combine all pair counts to get 2pcf estimator
        if estimator=="LS":
            self.xi = pref_DD*dds/rrs - pref_DR*drs/rrs -  pref_RD*rds/rrs + 1


    def computeNap2(self, radii, tofile=False):
        """ Computes second-order aperture statistics given the projected angular clustering correlation function.
        Uses the Crittenden 2002 filter.
        """

        nap2 = np.zeros((self.xi.shape[0], len(radii)), dtype=float)
        for elr, R in enumerate(radii):
            thetared = self.bin_centers_mean[np.newaxis,:]/R
            measure = (self.bin_edges[1:]-self.bin_edges[:-1])*self.bin_centers_mean/(R**2)
            filt = (thetared**4-16*thetared**2+32)/(128) * np.exp(-thetared**2/4.)
            nap2[:,elr] = np.sum(measure*filt*self.xi,axis=1)
            
        return nap2


class GGCorrelation(BinnedNPCF):
    r""" Compute second-order correlation functions of spin-2 fields.

    Parameters
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.

    Attributes
    ----------
    xip: numpy.ndarray
        The ξ₊ correlation function.
    xim: numpy.ndarray
        The ξ₋ correlation function.
    norm: numpy.ndarray
        The number of weighted pairs.
    npair: numpy.ndarray
        The number of unweighted pairs.

    Notes
    -----
    Inherits all other parameters and attributes from :class:`BinnedNPCF`.
    Additional child-specific parameters can be passed via ``kwargs``. 
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme.
    """

    def __init__(self, min_sep, max_sep, process_spherical=False, **kwargs):
        super().__init__(order=2, spins=np.array([2,2], dtype=np.int32), n_cfs=2, min_sep=min_sep, max_sep=max_sep, **kwargs)
        # Native curved-sky shear 2PCF (geodesic distance + nested-HEALPix
        # query_disc + spin-2 geodesic projection) instead of patch decomposition.
        # See Catalog.multihash_spherical / alloc_gg_doubletree (metric=SPHERICAL).
        self.process_spherical = bool(process_spherical)
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        self.nzcombis = None
        self.counts = None
        self.xip = None
        self.xim = None
        self.norm = None
        self.npair = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)

    def saveinst(self, path_save, fname):

        if not Path(path_save).is_dir():
            raise ValueError('Path to directory does not exist.')
        
        np.savez(path_save+fname,
                 nbinsz=self.nbinsz,
                 min_sep=self.min_sep,
                 max_sep=self.max_sep,
                 binsr=self.nbinsr,
                 method=self.method,
                 shuffle_pix=self.shuffle_pix,
                 tree_resos=self.tree_resos,
                 rmin_pixsize=self.rmin_pixsize,
                 resoshift_leafs=self.resoshift_leafs,
                 minresoind_leaf=self.minresoind_leaf,
                 maxresoind_leaf=self.maxresoind_leaf,
                 nthreads=self.nthreads,
                 bin_centers=self.bin_centers,
                 xip=self.xip,
                 xim=self.xim,
                 npair=self.npair,
                 norm=self.norm)

    def __process_patches(self, cat, dotomo=True, do_dc=False, rotsignflip=False, apply_edge_correction=False, adjust_tree=False,
                          save_patchres=False, save_filebase="", keep_patchres=False):

        if save_patchres:
            if not Path(save_patchres).is_dir():
                raise ValueError('Path to directory does not exist.')
            
        for elp in range(cat.npatches):
            if self._verbose_python:
                print('Doing patch %i/%i'%(elp+1,cat.npatches))
            
            # Compute statistics on patch
            pcat = cat.frompatchind(elp,rotsignflip=rotsignflip)
            pcorr = GGCorrelation(
                min_sep=self.min_sep,
                max_sep=self.max_sep,
                nbinsr=self.nbinsr,
                method=self.method,
                shuffle_pix=self.shuffle_pix,
                tree_resos=self.tree_resos,
                rmin_pixsize=self.rmin_pixsize,
                resoshift_leafs=self.resoshift_leafs,
                minresoind_leaf=self.minresoind_leaf,
                maxresoind_leaf=self.maxresoind_leaf,
                nthreads=self.nthreads,
                verbosity=self.verbosity)
            pcorr.process(pcat, dotomo=dotomo, do_dc=do_dc)
            
            # Update the total measurement
            if elp == 0:
                self.nbinsz = pcorr.nbinsz
                self.nzcombis = pcorr.nzcombis
                self.bin_centers = np.zeros_like(pcorr.bin_centers)
                self.xip = np.zeros_like(pcorr.xip)
                self.xim = np.zeros_like(pcorr.xim)
                self.norm = np.zeros_like(pcorr.norm)
                self.npair = np.zeros_like(pcorr.norm)
                if keep_patchres:
                    centers_patches = np.zeros((cat.npatches, *pcorr.bin_centers.shape), dtype=pcorr.bin_centers.dtype)
                    xip_patches = np.zeros((cat.npatches, *pcorr.xip.shape), dtype=pcorr.xip.dtype)
                    xim_patches = np.zeros((cat.npatches, *pcorr.xim.shape), dtype=pcorr.xim.dtype)
                    norm_patches = np.zeros((cat.npatches, *pcorr.norm.shape), dtype=pcorr.norm.dtype)
                    npair_patches = np.zeros((cat.npatches, *pcorr.npair.shape), dtype=pcorr.npair.dtype)
            self.bin_centers += pcorr.norm*pcorr.bin_centers
            self.xip += pcorr.norm*pcorr.xip
            self.xim += pcorr.norm*pcorr.xim
            self.norm += pcorr.norm
            self.npair += pcorr.npair
            if keep_patchres:
                centers_patches[elp] += pcorr.bin_centers
                xip_patches[elp] += pcorr.xip
                xim_patches[elp] += pcorr.xim
                norm_patches[elp] += pcorr.norm 
                npair_patches[elp] += pcorr.npair
            if save_patchres:
                pcorr.saveinst(save_patchres, save_filebase+'_patch%i'%elp)

        # Finalize the measurement on the full footprint
        self.bin_centers /= self.norm
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.xip /= self.norm
        self.xim /= self.norm
        self.projection = "xipm"

        if keep_patchres:
            return centers_patches, xip_patches, xim_patches, norm_patches, npair_patches
    
    def process(self, cat, dotomo=True, do_dc=False, rotsignflip=False, adjust_tree=False,
                save_patchres=False, save_filebase="", keep_patchres=False):
        r"""
        Compute a shear 2PCF given a shape catalog

        Parameters
        ----------
        cat: orpheus.SpinTracerCatalog
            The shape catalog to process.
        dotomo: bool
            Flag that decides whether the tomographic information in the shape catalog should be used. Defaults to `True`.
        do_dc: bool
            Whether to double-count pair counts. This will have no impact on :math:`\xi_\pm`, but can
            significantly reduce the amplitude of :math:`\xi_\times`. Defaults to `False`.
        rotsignflip: bool
            If the shape catalog has been decomposed in patches, choose whether the rotation angle should be flipped.
            For simulated data this was always ok to set to ``False``. Defaults to ``False``.
        adjust_tree: bool
            Overrides the original setup of the tree-approximations in the instance based on the nbar of the shape catalog.
            Not implemented yet; has no effect. Defaults to ``False``.
        save_patchres: bool or str
            If the shape catalog has been decomposed in patches, flag whether to save the GG measurements on the individual patches.
            Note that the path needs to exist, otherwise a ``ValueError`` is raised. For a flat-sky catalog this parameter
            has no effect. Defaults to ``False``.
        save_filebase: str
            Base of the filenames in which the patches are saved. The full filename will be `<save_patchres>/<save_filebase>_patchxx.npz`.
            Only has an effect if the shape catalog consists of multiple patches and `save_patchres` is not `False`.
        keep_patchres: bool
            If the catalog consists of multiple patches, returns all measurements on the patches. Defaults to `False`.
        """

        # Native curved-sky shear 2PCF (geodesic distance + nested-HEALPix
        # query_disc + spin-2 geodesic projection) requires process_spherical;
        # otherwise a spherical catalog must first be decomposed into patches.
        # Flat (or patched) catalogs take the pixel-box path. The struct-based
        # alloc_gg_doubletree dispatches on cat->metric, so both geometries share
        # the call block below.
        native_spherical = self.process_spherical and cat.geometry == 'spherical'
        if cat.geometry == 'spherical' and not native_spherical and cat.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches '
                             'using the Catalog._topatches method, or process_spherical=True must be set.')
        if cat.patchinds is not None and not native_spherical:
            return self.__process_patches(cat, dotomo=dotomo, do_dc=do_dc, rotsignflip=rotsignflip, adjust_tree=adjust_tree,
                                          save_patchres=save_patchres, save_filebase=save_filebase, keep_patchres=keep_patchres)

        # Tomography setup (shared by flat and native-spherical)
        self._checkcats(cat, self.spins)
        if not dotomo:
            self.nbinsz = 1
            old_zbins = cat.zbins[:]
            cat.zbins = np.zeros(cat.ngal, dtype=np.int32)
            self.nzcombis = 1
        else:
            self.nbinsz = cat.nbinsz
            self.nzcombis = self.nbinsz*self.nbinsz
        nbinsz = self.nbinsz
        sz2r = (nbinsz*nbinsz, self.nbinsr)

        # Build the multihash bundle (geometry-specific) and the concatenated
        # reduced shear the C catalog struct reads via its e1/e2 fields.
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
                                      fields=(cat.tracer_1, cat.tracer_2),
                                      verbose=self._verbose_python)
            extra = {'e1_resos': mh['red_e1'], 'e2_resos': mh['red_e2']}
        else:
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mh = cat.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                      shuffle=self.shuffle_pix, w2field=True, normed=True, nthreads=self.nthreads)
            allfields = mh['allfields']
            extra = {'e1_resos': np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64),
                     'e2_resos': np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)}

        # Build the four input structs + output arrays
        cat_s, keep_cat = build_catalog_struct(mh, nbinsz, extra=extra)
        cat_s.nresos = int(self.tree_nresos)
        nav_s, keep_nav = build_navhash_struct(mh, cat_obj=cat)
        tree_s, keep_tree = build_tree_params_struct(self, mh)
        maxleaf = max(0, self.tree_nresos-1)
        tree_s.minresoind_leaf = min(int(self.minresoind_leaf), maxleaf)
        tree_s.maxresoind_leaf = min(int(self.maxresoind_leaf), maxleaf)
        scale = convertunits(self.sep_units, 'rad') if native_spherical else None
        bin_s = build_binning_struct(self, do_dc=do_dc, scale=scale)
        # GG: the two natural components are stacked in the NPCFOutput 'npcf'
        # slot ([xip, xim]); split into views over the same C-written memory.
        out_s, bin_centers, _npcf, norm, _, npair, _ = build_npcf_output(
            'gg', self.nbinsr, nbinsz=nbinsz)
        _z2r = nbinsz*nbinsz*self.nbinsr
        xip, xim = _npcf[:_z2r], _npcf[_z2r:]

        # Keep every numpy array referenced only through a struct field alive
        # across the C call (ctypes structs are invisible to the GC).
        _alive = keep_cat + keep_nav + keep_tree   # noqa: F841

        self.clib.alloc_gg_doubletree(
            ct.byref(cat_s), ct.byref(nav_s), ct.byref(tree_s), ct.byref(bin_s),
            int(self.nthreads),
            int(self._verbose_c)+int(self._verbose_debug),
            ct.byref(out_s))

        # xip/xim are dimensionless; only bin_centers carries a length unit.
        if native_spherical:
            bin_centers /= convertunits(self.sep_units, 'rad')   # radians -> sep_units

        self.bin_centers = bin_centers.reshape(sz2r)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.npair = npair.reshape(sz2r)
        self.norm = norm.reshape(sz2r)
        self.xip = xip.reshape(sz2r)
        self.xim = xim.reshape(sz2r)
        self.projection = "xipm"
        if not do_dc:self.norm*=2; self.npair*=2

        if not dotomo:
            cat.zbins = old_zbins
        return
            
        
    def computeMap2(self, radii, tofile=False):
        """ Computes second-order aperture mass statistics given the shear correlation functions.
        Uses the Crittenden 2002 filter.
        """
        
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
    
    def computecosebi(self, Nmax, dps=100, Tminus_nsample=4096, units='rad2'):
        r"""Compute logarithmic COSEBIs from the shear 2PCFs using min_sep and
        max_sep as bounds.

        Parameters
        ----------
        Nmax : int
            Largest log-COSEBI mode to compute (modes ``n = 1, ..., Nmax``).
        dps : int, optional
            Decimal precision used internally by ``mpmath`` for the root
            finding. Defaults to ``100``.
        Tminus_nsample : int, optional
            Number of log-spaced sample points used for the inner
            cumulative integral when evaluating :math:`T_{-n}(\theta)`.
            Defaults to ``4096``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(4, nzcombis, Nmax)`` containing, in the first
            axis, :math:`E_n`, :math:`E_n^\times`, :math:`B_n`, and :math:`B_n^\times`.
        """

        ### HELPER FUNCTIONS ###
        # TODO MAYBE PUT IN A SEPARATE FILE
        def _log_cosebi_weights(theta_min, theta_max, Nmax, dps=100):
            r"""Compute roots and norms for log-COSEBI using SEK2010 formalism.

            Parameters
            ----------
            theta_min, theta_max : float
                Angular interval bounds for the COSEBI
            Nmax : int
                Largest COSEBI mode requested.
            dps : int, optional
                Decimal precision used by ``mpmath``. Defaults to ``100``.

            Returns
            -------
            Nn : list of mpmath.mpf, length Nmax
                Normalisation constants N_n such that t_n(z) = N_n * prod_i (z - r_{n,i}).
            rn : list of lists of mpmath.mpf
                Roots of each t_n polynomial (n+1 real roots per mode).
            zmax : mpmath.mpf
                ln(theta_max/theta_min).

            Notes
            -----
            * Follows somewhat the cosmopipe implementation as given in the notbook introducing cosmo-numba:
            https://github.com/aguinot/cosmo-numba/blob/main/notebooks/cosebis_comparison_with_cosmopipe.ipynb
            * Only works for logarithmically-spaced thetas
            """
            try:
                import mpmath
                from mpmath import mp
            except ImportError as exc:
                raise ImportError(
                    "mpmath is required for the log-COSEBI construction. "
                    "Install it via `pip install mpmath`."
                ) from exc

            mp.dps = dps

            ## Preparations ##
            # Def of zmax (below Eq 29) and J function (Eq 32)
            zmax = mp.log(mp.mpf(theta_max) / mp.mpf(theta_min))
            J = lambda k, j, zm: (mp.gamma(j+1) - mp.gammainc(j+1, -k*zm)) / mp.power(-k, j+1)

            ## Reconstruct coefficeint matrix c_{n,j} ##
            coeff_j = mp.matrix(Nmax+1, Nmax+2)
            # Below eq 29: Leading c_{n,n+1} = 1
            for i in range(Nmax+1):
                coeff_j[i, i+1] = mp.mpf(1)
            # Determine c_{1,0} and c_{1,1} using moment constraints for n=1 (Eq 33)
            mat_A = [[J(2, 0, zmax), J(2, 1, zmax)], [J(4, 0, zmax), J(4, 1, zmax)]]
            vec_v = [-J(2, 2, zmax), -J(4, 2, zmax)]
            sol = mp.lu_solve(mat_A, vec_v)
            coeff_j[1, 0], coeff_j[1, 1] = sol[0], sol[1]
            # Obtain all n>1 iteratively via (n-1) orthogonality equations Eq 34 + moment constraints Eq 33
            for nn in range(2, Nmax+1):
                mat_A = mp.matrix(nn+1, nn+1)
                vec_v = mp.matrix(nn+1, 1)
                # Build system of Eq 34
                for m in range(1, nn):
                    for j in range(nn+1):
                        for i in range(m+2):
                            mat_A[m-1, j] += J(1, i+j, zmax) * coeff_j[m, i]
                    for i in range(m+2):
                        vec_v[m -1] -= J(1, i+nn+1, zmax) * coeff_j[m, i]
                # Add moment constraints Eq 33
                for j in range(nn+1):
                    mat_A[nn-1, j] = J(2, j, zmax)
                    mat_A[nn,   j] = J(4, j, zmax)
                vec_v[nn-1]   = -J(2, nn+1, zmax)
                vec_v[nn]     = -J(4, nn+1, zmax)
                sol = mp.lu_solve(mat_A, vec_v)
                # Solve system and update coefficient matrix
                for j in range(nn+1):
                    coeff_j[nn, j] = sol[j]

            # Discard n=0 row as not further needed
            coeff_j = coeff_j[1:, :]

            # Get Normalisation via Eq 35
            Nn = []
            for nn in range(1, Nmax+1):
                s = mp.mpf(0)
                for i in range(nn+2):
                    for j in range(nn+2):
                        s += coeff_j[nn-1, i] * coeff_j[nn-1, j] * J(1, i+j, zmax)
                Nn.append(mp.sqrt(mp.fabs(mp.expm1(zmax)/s)))

            # Get roots via mpmath.polyroots (note: highest-degree coefficient first)
            # TODO: Just copied maxsteps and extraprec magic numbers from repo; make sure that those are reasonably general choices
            rn = []
            for nn in range(1, Nmax+1):
                coefs = [coeff_j[nn-1, nn+1-k] for k in range(nn+2)]
                roots = mpmath.polyroots(coefs, maxsteps=500, extraprec=100)
                rn.append(roots)

            return Nn, rn, zmax


        def _Tplus_log(theta, theta_min, Nn, roots):
            r"""Evaluate a log-COSEBI T_+ mode at angular scales ``theta``.

            Uses the factorised form of Eq 36 in SEK2010.
            """
            z = np.log(np.asarray(theta, dtype=np.float64)/theta_min)
            out = np.full_like(z, float(Nn), dtype=np.float64)
            for r in roots:
                out *= (z - float(r))
            return out


        def _Tminus_log(theta, theta_min, theta_max, Nn, roots, nsample=4096):
            r"""Evaluate the log-COSEBI T_- mode at angular scales ``theta``.

            Uses the SvWM2002 relation in log coordinates as (c.f. first line of 
            Eq 37 in SEK2010). To speed up the computation we evaluate the two
            ingegrals on a fine grid which we then spline and we only call these
            splines when distributing over the output theta array.
            """
            # Preparations: Get z and T_+ on a fine grid
            theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
            zmax = np.log(theta_max/theta_min)
            z_out = np.log(theta /theta_min)
            y = np.linspace(0., zmax, int(nsample))
            theta_y = theta_min * np.exp(y)
            Tp_y = _Tplus_log(theta_y, theta_min, Nn, roots)

            # Interpolate inner integrals on fine grid using cumulative trapezoidal integration
            _cumtrapz = lambda f, x: np.concatenate([[0.], np.cumsum(0.5 * (f[1:] + f[:-1]) * (x[1:] - x[:-1]))])
            cum2 = _cumtrapz(Tp_y * np.exp(2. * y), y)
            cum4 = _cumtrapz(Tp_y * np.exp(4. * y), y)

            # Distribute over output array
            I2_out = np.interp(z_out, y, cum2)
            I4_out = np.interp(z_out, y, cum4)
            Tp_out = _Tplus_log(theta, theta_min, Nn, roots)

            return Tp_out + 4.*np.exp(-2.*z_out)*I2_out - 12.*np.exp(-4.*z_out)*I4_out
        
        ### END OF HELPER FUNCTIONS ###

        if self.xip is None or self.xim is None or self.bin_centers_mean is None:
            raise RuntimeError(
                "GGCorrelation has not been populated yet. Call `process` "
                "before `computecosebi`."
            )
        if Nmax < 1:
            raise ValueError("Nmax must be at least 1.")
        
        assert(units in ['rad2','arcmin2'])

        

        # Get roots and norms
        theta = np.asarray(self.bin_centers_mean, dtype=np.float64)  # (nbinsr,)
        theta_min = float(self.min_sep)
        theta_max = float(self.max_sep)
        Nn, rn, _ = _log_cosebi_weights(theta_min, theta_max, Nmax, dps=dps)
        # Integration measure (including the 1/2 prefactor from COSEBI definition)
        pref = 0.5 * self.binsize*theta**2 

        # Allocate the result
        result = np.zeros((4, self.nzcombis, Nmax), dtype=float)
        for n in range(Nmax):
            Tp = _Tplus_log(theta, theta_min, Nn[n], rn[n])
            Tm = _Tminus_log(theta, theta_min, theta_max, Nn[n], rn[n], nsample=Tminus_nsample) 
            t1 = np.sum(pref * (Tp*self.xip + Tm*self.xim), axis=1)
            t2 = np.sum(pref * (Tp*self.xip - Tm*self.xim), axis=1)
            result[0, :, n] = t1.real   # E_n
            result[1, :, n] = t1.imag   # E_n^times
            result[2, :, n] = t2.real   # B_n
            result[3, :, n] = t2.imag   # B_n^times

        # Optionally transform to different units
        if units=='rad2':
            rad2arcminsq = (2*np.pi/360./60.)**2
            result *= rad2arcminsq

        return result

    def computepuremode(self):
        r"""Compute the pure-mode shear correlation functions on a finite
        interval using the formalism of Schneider, Asgari, Najafi et al. 
        (2022, arXiv:2110.09774).

        Returns
        -------
        xip_pure, numpy.ndarray
            Array of shape ``(3, nzcombis, nbinsr)`` containing the real
            parts of :math:`\xi_+^E`, :math:`\xi_+^B`, and :math:`\xi_+^amb` 
            evaluated at ``self.bin_centers_mean``.
        xim_pure, numpy.ndarray
            Same as xip_pure, but for xim
        """

        if self.xip is None or self.xim is None or self.bin_centers_mean is None:
            raise RuntimeError(
                "GGCorrelation has not been populated yet. Call `process` "
                "before `computepuremode`."
            )

        # Preparations: Build geometry-dependent constants from Schneider+2022 Eq 7
        theta = np.asarray(self.bin_centers_mean, dtype=np.float64)
        bar = 0.5 * (self.min_sep+self.max_sep)
        Bg  = (self.max_sep-self.min_sep) / (self.max_sep+self.min_sep)

        # Define all kernels from Schneider+2022 Eqs. 46, 47, 51, 54
        def Hplus(vt, th):
            return (1. / (8.*Bg**3)) * (
                4.*Bg**2 + 3.*((vt/bar)**2-1.-Bg**2)*((th/bar)**2-1.-Bg**2))

        def Hminus(vt, th):
            return ((1.-Bg)**2/(8.*Bg**3)) * (
                3.*(1.-Bg)**2 * ((1.+Bg)**4-(1.+4.*Bg+Bg**2)*(vt/bar)**2) * (th/bar)**(-2)
                + 3.*(1.+Bg)**2*(vt/bar)**2 - (3.+6.*Bg+14.*Bg**2+6.*Bg**3+3.*Bg**4))
    
        def Kplus(vt, th):
            return (bar/vt)**2 * Hminus(th, vt)
        
        # Use middle explicit form of eq 54
        def Kminus(vt, th):
            pref = (bar**4*(1.-Bg**2)**2)/(Bg*vt**2*th**2)
            b1 = 1.+Bg**2-(1.-Bg**2)**2*(bar/vt)**2
            b2 = 1.+Bg**2-(1.-Bg**2)**2*(bar/th)**2
            return pref * (0.5 + (3./(8.*Bg**2))*b1*b2)

        # Evaluat using 2D arrays
        ti = theta[:, None]  # outer --> vartheta
        tj = theta[None, :]  # inner --> theta'

        # Masks for partial integrals. Bin i contributes with half weight,
        # bins strictly beyond/inside the evaluation point have full weight.
        eye = np.eye(self.nbinsr)
        mask_upper = np.triu(np.ones((self.nbinsr, self.nbinsr)), k=1) + 0.5 * eye  # j >= i
        mask_lower = np.tril(np.ones((self.nbinsr, self.nbinsr)), k=-1) + 0.5 * eye  # j <= i
        KI_plus = self.binsize * (4.-12.*ti**2/tj**2) * mask_upper
        KI_minus = self.binsize * (tj**2/ti**2) * (4.-12.*tj**2/ti**2) * mask_lower

        # Full-interval ambiguous-mode kernels (all bins contribute fully)
        KS_plus  = self.binsize * (tj**2/bar**2) * Hplus(ti, tj)
        KS_minus = self.binsize * Hminus(ti, tj)
        KV_plus  = self.binsize * (tj**2/bar**2) * Kplus(ti, tj)
        KV_minus = self.binsize * (tj**2/bar**2) * Kminus(ti, tj)

        # xi@K.T -> shape (nzcombis, nbinsr)
        Iplus_z  = self.xim @ KI_plus.T
        Iminus_z = self.xip @ KI_minus.T
        Splus_z  = self.xip @ KS_plus.T
        Sminus_z = self.xim @ KS_minus.T
        Vplus_z  = self.xip @ KV_plus.T
        Vminus_z = self.xim @ KV_minus.T

        # SAN2022 Eqs. 42/43 and 55/56
        xip_E = 0.5 * (self.xip + self.xim + Iplus_z  - Splus_z-Sminus_z)
        xip_B = 0.5 * (self.xip - self.xim - Iplus_z  - Splus_z+Sminus_z)
        xim_E = 0.5 * (self.xip + self.xim + Iminus_z - Vplus_z-Vminus_z)
        xim_B = 0.5 * (self.xip - self.xim + Iminus_z - Vplus_z+Vminus_z)

        # Allocate E/B/Ambiguous mode for xi+-
        xip_pure = np.zeros((3, self.nzcombis, self.nbinsr), dtype=float)
        xim_pure = np.zeros((3, self.nzcombis, self.nbinsr), dtype=float)
        xip_pure[0] = xip_E.real
        xip_pure[1] = xip_B.real
        xip_pure[2] = Splus_z.real
        xim_pure[0] = xim_E.real
        xim_pure[1] = xim_B.real
        xim_pure[2] = Vminus_z.real

        return xip_pure, xim_pure


class NGCorrelation(BinnedNPCF):
    r""" Compute second-order cross-correlation functions of a spin-2 and a spin-0 field.

     Parameters
    ----------
    min_sep: float
        The smallest distance of each vertex for which the NPCF is computed.
    max_sep: float
        The largest distance of each vertex for which the NPCF is computed.
    **kwargs
        Passed to :class:`~orpheus.npcf_base.BinnedNPCF`.

    Attributes
    ----------
    xi: numpy.ndarray
        The position-shape correlation function.
    norm: numpy.ndarray
        The number of weighted pairs.
    npair: numpy.ndarray
        The number of unweighted pairs.


    Notes
    -----
    - In case of a three-dimensional box we define the projection direction along the z-axis
      and we allocate some additional private attributes ``_DS``, ``_RS` and ``_RR``.
    """

    def __init__(self, min_sep, max_sep, **kwargs):
        kwargs.setdefault('method', 'Discrete')
        super().__init__(order=2, spins=np.array([0, 2], dtype=np.int32), n_cfs=1,
                         min_sep=min_sep, max_sep=max_sep, **kwargs)
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz_shape = None
        self.nbinsz_pos = None
        self._DS = None
        self._RS = None
        self._RR = None
        self.xi = None
        self.norm = None
        self.npair = None
        self._initprojections(self)

        # Bind the discrete slab kernel (see ng_slab in src/corrfunc_second.c).
        p_f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_i64 = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        self.clib.ng_slab.restype = ct.c_void_p
        self.clib.ng_slab.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            p_f64, p_f64, p_f64, p_f64, p_i32, p_f64, p_f64, ct.c_int32,
            ct.c_int32, ct.c_double, ct.c_double,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
            p_i32, p_i32, p_i32, p_i32, p_i32,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double,
            ct.c_int32, ct.c_int32, ct.c_int32,
            p_f64, p_f64, p_f64, p_f64, p_i64]

        # Bind the flat2d doubletree kernel (see src/corrfunc_second.c): lens
        # (scalar, central) x source (spin-2, field), two struct sets.
        self.clib.alloc_ng_doubletree.restype = ct.c_void_p
        self.clib.alloc_ng_doubletree.argtypes = [
            ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
            ct.POINTER(MultiresoCatalog), ct.POINTER(NavHash),
            ct.POINTER(TreeResoParams), ct.POINTER(BinningParams),
            ct.c_int32, ct.c_int32, ct.POINTER(NPCFOutput)]

    def __call_ng_slab(self, q_cat_arrays, h_bundle, has_shapes, self_pairs, nbinsz_q):
        """Run one ng_slab pass. Returns (xs, wnorm, rsum, npairs) reshaped to
        (nbinsz_q, nbinsz_h, nbinsr) with xs complex."""
        q_pos1, q_pos2, q_pos3, q_w, q_zbin = q_cat_arrays
        q_ngal = len(q_pos1)
        nbinsz_h = self.nbinsz_shape if has_shapes else self.nbinsz_pos
        if has_shapes:
            h_e1, h_e2 = h_bundle['fields']
        else:
            h_e1 = h_e2 = np.zeros(1, dtype=np.float64)
        nout = nbinsz_q * nbinsz_h * self.nbinsr
        xs_re = np.zeros(nout, dtype=np.float64)
        xs_im = np.zeros(nout, dtype=np.float64)
        wnorm = np.zeros(nout, dtype=np.float64)
        rsum = np.zeros(nout, dtype=np.float64)
        npairs = np.zeros(nout, dtype=np.int64)
        npix = h_bundle['npix']  # noqa: F841  (npix recomputed in C from pix1_n*pix2_n)
        self.clib.ng_slab(
            q_pos1, q_pos2, q_pos3, q_w, q_zbin, ct.c_int32(q_ngal), ct.c_int32(nbinsz_q),
            h_bundle['pos1'], h_bundle['pos2'], h_bundle['pos3'], h_bundle['weight'],
            h_bundle['zbins'], h_e1, h_e2, ct.c_int32(nbinsz_h),
            ct.c_int32(h_bundle['nslabs']), ct.c_double(h_bundle['z0']), ct.c_double(h_bundle['dpix_z']),
            ct.c_double(h_bundle['pix1_start']), ct.c_double(h_bundle['pix1_d']), ct.c_int32(h_bundle['pix1_n']),
            ct.c_double(h_bundle['pix2_start']), ct.c_double(h_bundle['pix2_d']), ct.c_int32(h_bundle['pix2_n']),
            h_bundle['slab_offsets'], h_bundle['index_matcher'], h_bundle['pixs_galind_bounds'],
            h_bundle['rshift_bounds'], h_bundle['pix_gals'],
            ct.c_double(self.min_sep), ct.c_double(self.max_sep), ct.c_int32(self.nbinsr), ct.c_double(self._Pi),
            ct.c_int32(int(self_pairs)), ct.c_int32(int(has_shapes)), ct.c_int32(int(self.nthreads)),
            xs_re, xs_im, wnorm, rsum, npairs)
        shape = (nbinsz_q, nbinsz_h, self.nbinsr)
        xs = (xs_re + 1j*xs_im).reshape(shape)
        return xs, wnorm.reshape(shape), rsum.reshape(shape), npairs.reshape(shape)

    def process(self, cat_shape, cat_data, cat_random=None, Pi=None, dpix=None, dpix_z=None,
                dotomo=True, periodic=False, rotsignflip=False,
                save_patchres=False, save_filebase="", keep_patchres=False):
        r"""Compute the position-shape correlator, dispatching on geometry.

        - ``'3dbox'``: the discrete slab estimator (projected NI / ``w_{g+}``);
          requires ``Pi``, a density catalog ``cat_data`` and ``cat_random``.
        - ``'flat2d'``: the multi-resolution doubletree galaxy-galaxy-lensing
          tangential shear ``<gamma_t>`` (no projection); ``cat_data`` is the lens
          catalog, ``cat_random`` and ``Pi`` are ignored.
        - ``'spherical'``: both catalogs must be decomposed into matching patches
          (``Catalog._topatches``); each patch is projected to a flat tangent
          plane and processed with the ``'flat2d'`` doubletree, then combined.

        Parameters
        ----------
        cat_shape: orpheus.SpinTracerCatalog
            The shape (source) catalog; spin-2.
        cat_data: orpheus.ScalarTracerCatalog
            The density ('3dbox') / lens ('flat2d'/'spherical') tracer positions.
        cat_random: orpheus.ScalarTracerCatalog, optional
            The random catalog (required for '3dbox').
        Pi: float, optional
            Line-of-sight projection length (required for '3dbox').
        dpix, dpix_z: float, optional
            Transverse hash cell size and slab width ('3dbox' only).
        dotomo: bool
            Use tomographic bins. Defaults to ``True``.
        periodic: bool
            Placeholder for periodic boundaries ('3dbox'); not yet implemented.
        rotsignflip: bool
            Flip the source-shape rotation sign when projecting patches (spherical).
        save_patchres, save_filebase, keep_patchres:
            Per-patch save/return options (spherical/patched catalogs only).
        """
        assert isinstance(cat_shape, SpinTracerCatalog)
        assert isinstance(cat_data, ScalarTracerCatalog)
        if cat_shape.geometry == '3dbox':
            assert isinstance(cat_random, ScalarTracerCatalog), "'3dbox' NG requires a random catalog."
            assert Pi is not None, "'3dbox' NG requires a projection length Pi."
            return self.__process_3dbox(cat_shape, cat_data, cat_random, float(Pi),
                                        dpix=dpix, dpix_z=dpix_z, dotomo=dotomo, periodic=periodic)

        # flat2d / spherical galaxy-galaxy-lensing doubletree. A spherical catalog
        # must first be decomposed into patches; patched catalogs (any geometry)
        # take the per-patch flat path.
        if cat_shape.geometry == 'spherical' and cat_shape.patchinds is None:
            raise ValueError("Spherical NGCorrelation requires patch decomposition "
                             "(Catalog._topatches), or pass 'flat2d' catalogs.")
        if cat_shape.patchinds is not None:
            assert cat_data.patchinds is not None, \
                "Both source and lens catalogs must be patch-decomposed with matching patches."
            assert cat_shape.npatches == cat_data.npatches, \
                "Source and lens patch decompositions must match (equal npatches)."
            return self.__process_patches(cat_shape, cat_data, dotomo=dotomo, rotsignflip=rotsignflip,
                                          save_patchres=save_patchres, save_filebase=save_filebase,
                                          keep_patchres=keep_patchres)
        if cat_shape.geometry == 'flat2d':
            return self.__process_flat2d(cat_shape, cat_data, dotomo=dotomo)
        raise NotImplementedError(
            "NGCorrelation supports '3dbox' (discrete) and flat2d/spherical (doubletree).")

    def __process_3dbox(self, cat_shape, cat_data, cat_random, Pi, dpix=None, dpix_z=None,
                        dotomo=True, periodic=False):
        for c in (cat_data, cat_random):
            assert c.geometry == '3dbox', "NGCorrelation '3dbox' requires all catalogs in '3dbox'."
        if periodic:
            raise NotImplementedError("Periodic boundaries not implemented; use a random catalog.")

        self._Pi = float(Pi)
        if dpix is None: dpix = self.max_sep
        if dpix_z is None: dpix_z = Pi

        # Tomography: temporarily collapse zbins to a single bin if requested.
        old_zbins = None
        if not dotomo:
            old_zbins = (cat_shape.zbins.copy(), cat_data.zbins.copy(), cat_random.zbins.copy())
            cat_shape.zbins = np.zeros(cat_shape.ngal, dtype=np.int32)
            cat_data.zbins = np.zeros(cat_data.ngal, dtype=np.int32)
            cat_random.zbins = np.zeros(cat_random.ngal, dtype=np.int32)
            self.nbinsz_shape = 1
            self.nbinsz_pos = 1
        else:
            self.nbinsz_shape = cat_shape.nbinsz
            self.nbinsz_pos = max(cat_data.nbinsz, cat_random.nbinsz)
        nzd, nzs = self.nbinsz_pos, self.nbinsz_shape

        # Shared transverse + line-of-sight extent so every query point lies
        # inside the (shape / random) hash grid.
        ext = [min(cat_shape.min1, cat_data.min1, cat_random.min1),
               max(cat_shape.max1, cat_data.max1, cat_random.max1),
               min(cat_shape.min2, cat_data.min2, cat_random.min2),
               max(cat_shape.max2, cat_data.max2, cat_random.max2)]
        ext_z = [min(cat_shape.min3, cat_data.min3, cat_random.min3),
                 max(cat_shape.max3, cat_data.max3, cat_random.max3)]

        s_bundle = cat_shape.multihash_slabs(dpix, dpix_z,
                                             fields=(cat_shape.tracer_1, cat_shape.tracer_2),
                                             extent=ext, extent_z=ext_z)
        r_bundle = cat_random.multihash_slabs(dpix, dpix_z, extent=ext, extent_z=ext_z)

        def _arrs(cat):
            return (np.ascontiguousarray(cat.pos1), np.ascontiguousarray(cat.pos2),
                    np.ascontiguousarray(cat.pos3), np.ascontiguousarray(cat.weight),
                    np.ascontiguousarray(cat.zbins, dtype=np.int32))
        r_query = (r_bundle['pos1'], r_bundle['pos2'], r_bundle['pos3'],
                   r_bundle['weight'], r_bundle['zbins'])

        # DS: query=D, hashed=S (shapes). RS: query=R, hashed=S. RR: R auto-pairs.
        DS, wDS, rsumDS, npDS = self.__call_ng_slab(_arrs(cat_data), s_bundle, True, False, nzd)
        RS, wRS, rsumRS, npRS = self.__call_ng_slab(_arrs(cat_random), s_bundle, True, False, nzd)
        _, wRR, _, npRR = self.__call_ng_slab(r_query, r_bundle, False, True, nzd)

        # Density-weight totals per tomographic bin (for the D-R rescaling).
        WD = np.array([cat_data.weight[cat_data.zbins == z].sum() for z in range(nzd)])
        WR = np.array([cat_random.weight[cat_random.zbins == z].sum() for z in range(nzd)])
        f = np.divide(WD, WR, out=np.ones_like(WD), where=WR > 0)

        # Vedder estimator D~S / RR with the implicit sample-size rescaling.
        # The random-shape pair weight (scaled to the data density) stands in for
        # the rescaled random pair count RR of the density tracer. The overall
        # sign follows the tangential (radial-positive) convention; both the sign
        # and this normalization are validated against TreeCorr.
        DtS = DS - f[:, None, None]*RS
        RRnorm = f[:, None, None]*wRS
        xi = -np.divide(DtS, RRnorm, out=np.zeros_like(DtS), where=RRnorm > 0)
        bc = np.divide(rsumDS, wDS, out=np.zeros_like(rsumDS), where=wDS > 0)

        # Flatten the tomographic pair index into a single leading axis, matching
        # GG/NN, the flat2d NG path, and the 3pt correlators. DS/RS run over
        # (nz_pos, nz_shape); RR auto-pairs the density tracer, so (nz_pos, nz_pos).
        z2r = (nzd*nzs, self.nbinsr)
        zzr = (nzd*nzd, self.nbinsr)
        self.xi = xi.reshape(z2r)
        self.bin_centers = bc.reshape(z2r)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self._DS, self._RS, self._RR = DS.reshape(z2r), RS.reshape(z2r), wRR.reshape(zzr)
        self._wDS, self._wRS = wDS.reshape(z2r), wRS.reshape(z2r)
        self._npairs_DS, self._npairs_RS = npDS.reshape(z2r), npRS.reshape(z2r)
        self._npairs_RR = npRR.reshape(zzr)

        if not dotomo:
            cat_shape.zbins, cat_data.zbins, cat_random.zbins = old_zbins
        return

    def __process_flat2d(self, cat_source, cat_lens, dotomo=True):
        r"""Flat-sky galaxy-galaxy-lensing tangential shear via the doubletree.

        Mirrors :meth:`GGCorrelation.process` (flat path) but for a scalar
        lens x spin-2 source cross: builds a source multihash (with reduced
        shear) and a lens multihash on a shared joint extent, then calls
        ``alloc_ng_doubletree`` (lens = central, source = field).
        """
        # Tomography setup.
        old_zbins = None
        if not dotomo:
            old_zbins = (cat_source.zbins.copy(), cat_lens.zbins.copy())
            cat_source.zbins = np.zeros(cat_source.ngal, dtype=np.int32)
            cat_lens.zbins = np.zeros(cat_lens.ngal, dtype=np.int32)
            self.nbinsz_shape = 1
            self.nbinsz_pos = 1
        else:
            self.nbinsz_shape = cat_source.nbinsz
            self.nbinsz_pos = cat_lens.nbinsz
        nzl, nzs = self.nbinsz_pos, self.nbinsz_shape

        # Shared (joint) extent so both hashes live on the same flat grid.
        cutfirst = np.int32(self.tree_resos[0] == 0.)
        jointextent = list(cat_source._jointextent([cat_lens], extend=self.tree_resos[-1]))

        # Source multihash (reduced shear via w2field) and lens multihash (scalar).
        mh_source = cat_source.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                                shuffle=self.shuffle_pix, w2field=True, normed=True,
                                                extent=jointextent, nthreads=self.nthreads)
        allfields = mh_source['allfields']
        extra_s = {'e1_resos': np.concatenate([allfields[i][0] for i in range(len(allfields))]).astype(np.float64),
                   'e2_resos': np.concatenate([allfields[i][1] for i in range(len(allfields))]).astype(np.float64)}
        mh_lens = cat_lens.multihash_bundle(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1],
                                            shuffle=self.shuffle_pix, normed=False,
                                            extent=jointextent, nthreads=self.nthreads)

        # Build the two struct sets (source carries e1/e2; lens is scalar).
        cats_s, keep_cs = build_catalog_struct(mh_source, nzs, extra=extra_s)
        cats_s.nresos = int(self.tree_nresos)
        navs_s, keep_ns = build_navhash_struct(mh_source, cat_obj=cat_source)
        catl_s, keep_cl = build_catalog_struct(mh_lens, nzl)
        catl_s.nresos = int(self.tree_nresos)
        navl_s, keep_nl = build_navhash_struct(mh_lens, cat_obj=cat_lens)
        tree_s, keep_tree = build_tree_params_struct(self, mh_source)
        maxleaf = max(0, self.tree_nresos-1)
        tree_s.minresoind_leaf = min(int(self.minresoind_leaf), maxleaf)
        tree_s.maxresoind_leaf = min(int(self.maxresoind_leaf), maxleaf)
        bin_s = build_binning_struct(self, do_dc=1)
        out_s, bin_centers, xi, norm, _, npair, _ = build_npcf_output(
            'ng', self.nbinsr, nbinsz_lens=nzl, nbinsz_source=nzs)

        # Keep numpy arrays referenced only through struct fields alive across the call.
        _alive = keep_cs + keep_ns + keep_cl + keep_nl + keep_tree   # noqa: F841

        self.clib.alloc_ng_doubletree(
            ct.byref(catl_s), ct.byref(navl_s), ct.byref(cats_s), ct.byref(navs_s),
            ct.byref(tree_s), ct.byref(bin_s),
            int(self.nthreads), int(self._verbose_c)+int(self._verbose_debug),
            ct.byref(out_s))

        szr = (nzl*nzs, self.nbinsr)
        self.bin_centers = bin_centers.reshape(szr)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.xi = xi.reshape(szr)
        self.norm = norm.reshape(szr)
        self.npair = npair.reshape(szr)
        self.projection = None

        if not dotomo:
            cat_source.zbins, cat_lens.zbins = old_zbins
        return

    def __process_patches(self, cat_source, cat_lens, dotomo=True, rotsignflip=False,
                          save_patchres=False, save_filebase="", keep_patchres=False):
        r"""Spherical galaxy-galaxy lensing: process each patch on its flat tangent
        plane and combine (norm-weighted), mirroring GG/NGG. Source shapes are
        rotated via ``rotsignflip``; lens positions carry no spin. Pairs straddling
        patch boundaries are dropped (the standard patch approximation)."""
        if save_patchres and not Path(save_patchres).is_dir():
            raise ValueError('Path to directory does not exist.')

        for elp in range(cat_source.npatches):
            if self._verbose_python:
                print('Doing patch %i/%i'%(elp+1, cat_source.npatches))
            pscat = cat_source.frompatchind(elp, rotsignflip=rotsignflip)
            plcat = cat_lens.frompatchind(elp)
            pcorr = NGCorrelation(
                min_sep=self.min_sep, max_sep=self.max_sep, nbinsr=self.nbinsr,
                method=self.method, shuffle_pix=self.shuffle_pix, tree_resos=self.tree_resos,
                rmin_pixsize=self.rmin_pixsize, resoshift_leafs=self.resoshift_leafs,
                minresoind_leaf=self.minresoind_leaf, maxresoind_leaf=self.maxresoind_leaf,
                nthreads=self.nthreads, verbosity=self.verbosity)
            pcorr.process(pscat, plcat, dotomo=dotomo)

            if elp == 0:
                self.nbinsz_shape = pcorr.nbinsz_shape
                self.nbinsz_pos = pcorr.nbinsz_pos
                self.bin_centers = np.zeros_like(pcorr.bin_centers)
                self.xi = np.zeros_like(pcorr.xi)
                self.norm = np.zeros_like(pcorr.norm)
                self.npair = np.zeros_like(pcorr.npair)
                if keep_patchres:
                    centers_patches = np.zeros((cat_source.npatches, *pcorr.bin_centers.shape), dtype=pcorr.bin_centers.dtype)
                    xi_patches = np.zeros((cat_source.npatches, *pcorr.xi.shape), dtype=pcorr.xi.dtype)
                    norm_patches = np.zeros((cat_source.npatches, *pcorr.norm.shape), dtype=pcorr.norm.dtype)
                    npair_patches = np.zeros((cat_source.npatches, *pcorr.npair.shape), dtype=pcorr.npair.dtype)
            self.bin_centers += pcorr.norm*pcorr.bin_centers
            self.xi += pcorr.norm*pcorr.xi
            self.norm += pcorr.norm
            self.npair += pcorr.npair
            if keep_patchres:
                centers_patches[elp] += pcorr.bin_centers
                xi_patches[elp] += pcorr.xi
                norm_patches[elp] += pcorr.norm
                npair_patches[elp] += pcorr.npair
            if save_patchres:
                pcorr.saveinst(save_patchres, save_filebase+'_patch%i'%elp)

        # Finalize on the full footprint (norm-weighted mean of the patches).
        self.bin_centers = np.divide(self.bin_centers, self.norm, out=np.zeros_like(self.bin_centers), where=self.norm > 0)
        self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
        self.xi = np.divide(self.xi, self.norm, out=np.zeros_like(self.xi), where=self.norm > 0)
        self.projection = None

        if keep_patchres:
            return centers_patches, xi_patches, norm_patches, npair_patches

    def saveinst(self, path_save, fname):
        if not Path(path_save).is_dir():
            raise ValueError('Path to directory does not exist.')
        np.savez(path_save+fname,
                 nbinsz_shape=self.nbinsz_shape, nbinsz_pos=self.nbinsz_pos,
                 min_sep=self.min_sep, max_sep=self.max_sep, nbinsr=self.nbinsr,
                 method=self.method, shuffle_pix=self.shuffle_pix, tree_resos=self.tree_resos,
                 rmin_pixsize=self.rmin_pixsize, nthreads=self.nthreads,
                 bin_centers=self.bin_centers, xi=self.xi, norm=self.norm, npair=self.npair)

    def computeMapNap(self, radii, tofile=False):
        """ Computes second-order aperture statistics given the projected position-shape correlation function.
        Uses the Crittenden 2002 filter.
        """

        mapnap = np.zeros((self.xi.shape[0], len(radii)), dtype=complex)
        for elr, R in enumerate(radii):
            thetared = self.bin_centers_mean[np.newaxis,:]/R
            measure = (self.bin_edges[1:]-self.bin_edges[:-1])*self.bin_centers_mean/(R**2)
            filt = thetared**2*(12.-thetared**2)/128. * np.exp(-thetared**2/4.)
            mapnap[:,elr] = np.sum(measure*filt*self.xi,axis=1)
            
        return mapnap