# Here we collect some utils for mapping the a full-sky survey to a set of overlapping patches
# In the middle term much of this functionality should be included in the orpheus code

from astropy.coordinates import SkyCoord
from healpy import ang2pix, pix2vec, nside2pixarea, nside2resol, query_disc, Rotator, nside2npix
import numpy as np
from pathlib import Path
import pickle
import os
import sys
from time import time

from sklearn.cluster import KMeans

def pickle_save(data, filename):
    
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print(f"An error occurred while saving the dictionary: {e}")
        
def pickle_load(filename):
    
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        pass

def frompatchindices_preparerot(index, patchindices, ra, dec, rotsignflip):

    inds_inner = patchindices["patches"][index]["inner"]
    inds_outer = patchindices["patches"][index]["outer"]
    inds_extpatch = np.append(inds_inner,inds_outer)
    ngal_patch = len(inds_extpatch)
    patch_isinner = np.zeros(ngal_patch,dtype=bool)
    patch_isinner[:len(inds_inner)] = True
    patch_isinner[len(inds_inner):] = False
    # Note that we fix the rotangle at this instance as this is required when computing patches
    # across multiple catalogs. In that case the patchcenters are by definition the com of the
    # joint catalog. For a single catalog this does not matter. The signs match the (theta,phi)
    # conventions in healpy -- see the toorigin function for details.
    rotangle = [+patchindices['info']['patchcenters'][index][0]*np.pi/180.,
                -patchindices['info']['patchcenters'][index][1]*np.pi/180.]
    nextrotres = toorigin(ra[inds_extpatch], 
                          dec[inds_extpatch], 
                          isinner=patch_isinner, 
                          rotangle=rotangle, 
                          inv=False, 
                          rotsignflip=rotsignflip,
                          radec_units="deg")
    rotangle, ra_rot, dec_rot, rotangle_polars = nextrotres

    return inds_extpatch, patch_isinner, rotangle, ra_rot, dec_rot, rotangle_polars
    
def gen_cat_patchindices(ra_deg, dec_deg, npatches, patchextend_arcmin, nside_hash=128, verbose=False, method='kmeans_healpix',
                         kmeanshp_maxiter=1000, kmeanshp_tol=1e-10, kmeanshp_randomstate=42, healpix_nside=8):
    """ Decomposes a spherical catalog in ~equal-area patches with a buffer region
    
    Parameters
    ----------
    ra_deg: numpy.ndarray
        The ra of the catalog, given in units of degree.
    dec_deg: numpy.ndarray
        The dec of the catalog, given in units of degree.
    npatches: int
        The number of patches in which the catalog shall be decomposed.
    patchextend_arcmin: float
        The buffer region that extends around each patch, given in units of arcmin.
    nside_hash: int
        The healpix resolution used for hashing subareas of the patches.
    verbose: bool
        Flag setting on whether output is printed to the console.
        
    Returns
    -------
    cat_patchindices: dict
        A dictionary containing information about the individual patches,
        as well as the galaxy indices that are assigned to the inner region
        and to the buffer region of each individual patch
    
    Notes
    -----
    Choosing a small value of nside_hash will result in a larger extension of 
    the patches then neccessary while choosing a large value increases the 
    runtime. A good compromise is to choose nside_hash such that its resolution 
    is a few times smaller than the buffer region of the patches    
    """
    
    def build_indexhash(arr):
        """Returns a hash for indices of repeated values in a 1D array"""
        sort_indices = np.argsort(arr)
        arr = np.asarray(arr)[sort_indices]
        vals, first_indices = np.unique(arr, return_index=True)
        indices = np.split(sort_indices, first_indices[1:])
        indhash = {}
        for elval,val in enumerate(vals):
            indhash[val] = indices[elval]
        return indhash    
    
    if verbose:
        print("Computing inner region of patches")
        t1 = time()
    
    # Run treecorrs k-means implementation
    if method=='kmeans_treecorr':
        try:
            import treecorr
            cat = treecorr.Catalog(ra=ra_deg, dec=dec_deg, 
                               ra_units="deg", dec_units="deg", 
                               npatch=npatches)
            patchinds = cat.patch
        except ImportError:
            if method=='kmeans_treecorr':
                print('Treecorr not availbale...switching to patch creation via KMeans')
                method = 'kmeans_healpix'
        
    # Run standard k-means on catalog reduced to healpix pixels
    elif method=='kmeans_healpix':
        # Step 1: Reduce discrete ra/dec to unique healpix pixels and transform those to to 3D positions
        nside_kmeans = 2048 # I keep this fixed for now as it will most likely work well for all reasonable cases.
        eq = SkyCoord(ra_deg, dec_deg, frame='galactic', unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
        theta = np.radians(90. - b)
        phi = np.radians(l)
        hpx_inds = ang2pix(nside_kmeans, theta, phi)
        hpx_uinds = np.unique(hpx_inds)
        # Step 2: Run standard kmeans algorithm on the healpix pixels
        # Note that each pixel carries the same (unity) weight. This implies
        # that we make the patches have approximately equal area, but neglect
        # depth variations on a patch sized scale. To me this seems to be a
        # sensible choice as the flat-sky approximation only cares about the
        # extent of the patches. If one wants to use the patches as Jackknife
        # samples for an internal covariance matrix estimate this choice might
        # need to be revisited (but as of now I do not see a clear point against
        # continuing to use the current setup as long as the patchsize is in a
        # domain where the contributions to the covariance that are containing 
        # shapenoise are expected to be subdominant).
        clust = KMeans(n_clusters=npatches,
                init='k-means++', 
                n_init='auto', 
                max_iter=kmeanshp_maxiter, 
                tol=kmeanshp_tol,
                verbose=0, 
                random_state=kmeanshp_randomstate, 
                copy_x=True, 
                algorithm='lloyd')
        X = np.array(pix2vec(nside=nside_kmeans,ipix=hpx_uinds,nest=False)).T
        clustinds = clust.fit_predict(X, y=None, sample_weight=None)
        # Step 3: Map the pixel centers back to the galaxy indices
        hashmap = np.vectorize({upix: center for upix, center in zip(hpx_uinds, clustinds)}.get)
        patchinds = hashmap(hpx_inds)
    # Simply assign to healpix pixel. Fast and stable, but patchareas might strongly vary in size.
    elif method == "healpix":
        eq = SkyCoord(ra_deg, dec_deg, frame='galactic', unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
        theta = np.radians(90. - b)
        phi = np.radians(l)
        patchinds = ang2pix(healpix_nside, theta, phi).astype(int)
    else:
        raise NotImplementedError
        
    if verbose:
        t2=time()
        print("Took %.3f seconds"%(t2-t1))
    
    # Assign galaxy positions to healpix pixels
    if verbose:
        print("Mapping catalog to healpix grid")
        t1=time()
    eq = SkyCoord(ra_deg, dec_deg, frame='galactic', unit='deg')
    l, b = eq.galactic.l.value, eq.galactic.b.value
    theta = np.radians(90. - b)
    phi = np.radians(l)
    cat_indices = ang2pix(nside_hash, theta, phi)
    if verbose:
        t2=time()
        print("Took %.3f seconds"%(t2-t1))
    
    # Build a hash connecting the galaxies residing in each healpix pixel
    if verbose:
        t1=time()
        print("Building index hash")
    cat_indhash = build_indexhash(cat_indices)
    if verbose:
        t2=time()
        print("Took %.3f seconds"%(t2-t1))
    
    # Construct buffer region around patches
    if verbose:
        print("Building buffer around patches")
        t1=time()
    _pixarea = nside2pixarea(nside_hash,degrees=True)
    _pixreso = nside2resol(nside_hash,arcmin=True)
    if method == 'kmeans_treecorr':
        _patchcenters = cat.patch_centers
    elif method == 'kmeans_healpix' or method=='healpix':
        _patchcenters = np.array([[np.mean(ra_deg[ patchinds==patchind]), np.mean(dec_deg[ patchinds==patchind])] for patchind in range(npatches)])
    else:
        raise NotImplementedError
        
    cat_patchindices = {}
    cat_patchindices["info"] = {}
    cat_patchindices["info"]["patchcenters"] = _patchcenters
    cat_patchindices["info"]["patchareas"] = np.zeros(npatches,dtype=float)
    cat_patchindices["info"]["patch_ngalsinner"] = np.zeros(npatches,dtype=int)
    cat_patchindices["info"]["patch_ngalsouter"] = np.zeros(npatches,dtype=int)
    cat_patchindices["patches"] = {}
    ext_buffer = (patchextend_arcmin+_pixreso)*np.pi/180./60.
    for elpatch in range(npatches):
        if verbose:
            sys.stdout.write("\r%i/%i"%(elpatch+1,npatches))
        patchsel = patchinds==elpatch
        cat_patchindices["patches"][elpatch] = {}

        # Get indices of gals within inner patch
        galinds_inner = np.argwhere(patchsel).flatten().astype(int)

        # Find healpix pixels in extended patch
        patch_indices = np.unique(ang2pix(nside_hash, theta[patchsel], phi[patchsel]))
        extpatch_indices = set()
        for pix in patch_indices:
            nextset = set(query_disc(nside=nside_hash, 
                                        vec=pix2vec(nside_hash,pix),
                                        radius=ext_buffer))
            extpatch_indices.update(nextset)

        # Assign galaxies to extended patch
        galinds_ext = set()
        for pix in extpatch_indices:
            try:
                galinds_ext.update(set(cat_indhash[pix]))
            except:
                pass
        galinds_outer = np.array(list(galinds_ext-set(galinds_inner)),dtype=int)
        cat_patchindices["info"]["patchareas"][elpatch] = _pixarea*len(patch_indices)
        cat_patchindices["info"]["patch_ngalsinner"][elpatch] = len(galinds_inner)
        cat_patchindices["info"]["patch_ngalsouter"][elpatch] = len(galinds_outer)
        cat_patchindices["patches"][elpatch]["inner"] = galinds_inner
        cat_patchindices["patches"][elpatch]["outer"] = galinds_outer
    if verbose:
        t2=time()
        print("Took %.3f seconds"%(t2-t1))
    
    return cat_patchindices

def toorigin(ras, decs, isinner=None, rotangle=None, inv=False, rotsignflip=False, radec_units="deg"):
    """ Rotates survey patch s.t. its center of mass lies in the origin. """
    import healpy as hp
    assert(radec_units in ["rad", "deg"])
    
    if isinner is None:
        isinner = np.ones(len(ras), dtype=bool)
    
    # Map (ra, dec) --> (theta, phi)
    if radec_units=="deg":
        decs_rad = decs*np.pi/180.
        ras_rad = ras*np.pi/180.
    thetas = np.pi/2. + decs_rad
    phis = ras_rad
    
    # Compute rotation angle
    if rotangle is None:
        rotangle = [np.mean(phis[isinner]),np.pi/2.-np.mean(thetas[isinner])]
    thisrot = Rotator(rot=rotangle, deg=False, inv=inv)
    rotatedthetas, rotatedphis = thisrot(thetas,phis,inv=False)
    rotangle_polars = np.exp((-1)**rotsignflip*1J * 2 * thisrot.angle_ref(rotatedthetas, rotatedphis,inv=True))
    
    # Transform back to (ra,dec)
    ra_rot = rotatedphis
    dec_rot = rotatedthetas - np.pi/2.
    if radec_units=="deg":
        dec_rot *= 180./np.pi
        ra_rot *= 180./np.pi
    
    return rotangle, ra_rot, dec_rot, rotangle_polars
    
def cat2hpx(lon, lat, nside, radec=True, do_counts=False, return_idx=False, return_indices=False, weights=None):
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
        coordinate system. Otherwise assume input is glon, glat
    nside : int
        HEALPix nside of the target map
    radec : bool
        Switch between R.A./Dec and glon/glat as input coordinate system.
    do_counts : bool
        Return the number of counts per HEALPix pixel
    return_idx : bool
        Return the set of non-empty HEALPix pixel indices
    return_indices : bool
        Returns the per-object HEALPix pixel indices
    weights: None or ndarray
        Needs to be given if each point carries an individual weight

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates
        
    Notes
    -----
    This functions is a generalised version of https://stackoverflow.com/a/50495134
    """

    npix = nside2npix(nside)

    if radec:
        eq = SkyCoord(lon, lat, frame='galactic', unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat

    # conver to theta, phi
    theta = np.radians(90. - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = ang2pix(nside, theta, phi)
    
    if do_counts:
        idx, counts = np.unique(indices, return_counts=True)
    if weights is not None:
        idx, inv = np.unique(indices,return_inverse=True)
        weights_pix = np.bincount(inv,weights.reshape(-1))
    else:
        idx = np.asarray(list(set(list(indices)))).astype(int)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    #counts[counts>1] = 1
    if do_counts:
        hpx_map[idx] = counts
    else:
        hpx_map[idx] = np.ones(len(idx), dtype=int)
    
    res = ()
    if return_idx:
        res +=  (idx, )
    res += (hpx_map.astype(int)), 
    if weights is not None:
        res += (weights_pix), 
    if return_indices:
        res += (indices), 
        
    return res