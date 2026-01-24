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
            Not implemented yet, therefore no effect. Has no effect yet. Defaults to `False`.
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
    Either ``nbinsr`` or ``binsize`` has to be provided to fix the binning scheme .
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
            If the shape catalog is has been decomposed in patches, choose whether the rotation angle should be flipped.
            For simulated data this was always ok to set to 'False`. Defaults to `False`.
        adjust_tree: bool
            Overrides the original setup of the tree-approximations in the instance based on the nbar of the shape catalog.
            Not implemented yet, therefore no effect. Has no effect yet. Defaults to `False` 
        save_patchres: bool or str
            If the shape catalog is has been decomposed in patches, flag whether to save the GG measurements on the individual patches. 
            Note that the path needs to exist, otherwise a `ValueError` is raised. For a flat-sky catalog this parameter 
            has no effect. Defaults to `False`
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