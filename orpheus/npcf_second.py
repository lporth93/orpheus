import numpy as np
from pathlib import Path 
import copy

from .catalog import Catalog, ScalarTracerCatalog, SpinTracerCatalog
from .npcf_base import BinnedNPCF

__all__ = ["NNCorrelation", "GGCorrelation"]


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

    def __init__(self, min_sep, max_sep, shuffle_pix=1, **kwargs):
        super().__init__(order=2, spins=np.array([0,0], dtype=np.int32), n_cfs=1, min_sep=min_sep, max_sep=max_sep, shuffle_pix=shuffle_pix, **kwargs)
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

    def __process_patches(self, cat, dotomo=True,  do_dc=True, adjust_tree=False,
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
    
    def process(self, cat, cat_random=None, dotomo=True, do_dc=True, adjust_tree=False,
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

        # If random catalog present, use the __compute_xi method
        if cat_random is not None:
            assert(isinstance(cat_random, ScalarTracerCatalog))
            self.__compute_xi(cat, cat_random, dotomo=dotomo, adjust_tree=adjust_tree,
                   save_patchres=save_patchres, keep_patchres=keep_patchres, estimator="LS")
            return

        # Make sure that in case the catalog is spherical, it has been decomposed into patches
        if cat.geometry == 'spherical' and cat.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches using the Catalog._topatches method.')

        # Catalog consist of multiple patches
        if cat.patchinds is not None:
            return self.__process_patches(cat, dotomo=dotomo, do_dc=do_dc, adjust_tree=adjust_tree,
                                          save_patchres=save_patchres, save_filebase=save_filebase, keep_patchres=keep_patchres)   
        # Catalog does not consist of patches
        else:
            # Prechecks
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
            npair = np.zeros(z2r).astype(np.float64)
            npair_cell = np.zeros(z2r).astype(np.int64)
                        
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                  shuffle=self.shuffle_pix, normed=False)
            ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            isinner_resos = np.concatenate(isinners).astype(np.float64)
            index_matcher = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals = np.concatenate(pix_gals).astype(np.int32)
            index_matcher_flat = np.argwhere(cat.index_matcher>-1).flatten()
            nregions = len(index_matcher_flat)
            
            args_treeresos = (np.int32(self.tree_nresos), np.int32(self.tree_nresos-cutfirst),
                            dpixs1_true.astype(np.float64), dpixs2_true.astype(np.float64), self.tree_redges, 
                            np.int32(self.resoshift_leafs), np.int32(self.minresoind_leaf), 
                            np.int32(self.maxresoind_leaf), np.array(ngal_resos, dtype=np.int32), )
            args_resos = (isinner_resos, weight_resos, pos1_resos, pos2_resos, zbin_resos,
                        index_matcher, pixs_galind_bounds, pix_gals, )
            args_hash = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                        np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), 
                        np.int32(nregions), index_matcher_flat.astype(np.int32),)
            args_binning = (np.float64(self.min_sep), np.float64(self.max_sep), np.int32(self.nbinsr), np.int32(do_dc), )
            args_output = (bin_centers, npair, npair_cell, )
            func = self.clib.alloc_nn_doubletree
            args = (*args_treeresos,
                    np.int32(self.nbinsz),
                    *args_resos,
                    *args_hash,
                    *args_binning,
                    np.int32(self.nthreads),
                    np.int32(self._verbose_c)+np.int32(self._verbose_debug),
                    *args_output)

            func(*args)
            
            self.bin_centers = bin_centers.reshape(sz2r)
            self.bin_centers_mean = np.mean(self.bin_centers, axis=0)
            self.npair = npair.reshape(sz2r)
            self.npair_cell = npair_cell.reshape(sz2r)
            self.projection = None
            
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
        
        # In case of a spherical geometry, decompose the joint catalog in patches of the same target geometry as
        # the geometry that was specified in the data catalog
        if cat_data.geometry=="spherical":
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

    def __init__(self, min_sep, max_sep, **kwargs):
        super().__init__(order=2, spins=np.array([2,2], dtype=np.int32), n_cfs=2, min_sep=min_sep, max_sep=max_sep, **kwargs)
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

        # Make sure that in case the catalog is spherical, it has been decomposed into patches
        if cat.geometry == 'spherical' and cat.patchinds is None:
            raise ValueError('Error: Spherical catalog needs to be first decomposed into patches using the Catalog._topatches method.')

        # Catalog consist of multiple patches
        if cat.patchinds is not None:
            return self.__process_patches(cat, dotomo=dotomo, do_dc=do_dc, rotsignflip=rotsignflip, adjust_tree=adjust_tree,
                                          save_patchres=save_patchres, save_filebase=save_filebase, keep_patchres=keep_patchres)   
        # Catalog does not consist of patches
        else:
            # Prechecks
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
                        
            cutfirst = np.int32(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:], dpix_hash=self.tree_resos[-1], 
                                shuffle=self.shuffle_pix, w2field=True, normed=True)
            ngal_resos, pos1s, pos2s, weights, zbins, isinners, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            isinner_resos = np.concatenate(isinners).astype(np.float64)
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
                    np.int32(self._verbose_c)+np.int32(self._verbose_debug),
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
            pref = (bar**4*(1.0-Bg**2)**2)/(Bg*vt**2*th**2)
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

        # xi @ K.T -> shape (nzcombis, nbinsr) indexed by i.
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